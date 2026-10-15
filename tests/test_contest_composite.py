"""Composite scenes (scripts/contest_composite.py): the --scene grammar, the
grids and supports of each image, and the Joint penalty whose overall factor
the discrepancy principle chooses. Tiny NumPy/virgil evaluations only; the
fits themselves run on OzSTAR."""

import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("virgil")

ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import contest_composite as cc  # noqa: E402
import contest_images as ci  # noqa: E402

GUARD = pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")


@GUARD
def test_parse_scene_puts_the_star_first_and_fills_defaults():
    parts = cc.parse_scene("disk=image[hole=0.5, ell=0.5:1, prior=gp] + planet=stamp + star=point")
    assert [(p.name, p.kind) for p in parts] == [("star", "point"), ("disk", "image"), ("planet", "stamp")]
    disk, planet = parts[1].opts, parts[2].opts
    assert disk["hole"] == 0.5 and disk["ell"] == (0.5, 1.0) and disk["sigma"] == cc.IMAGE_DEFAULTS["sigma"]
    assert planet["at"] == "search" and planet["fov"] == 2.0 and planet["pixel"] == 0.125 and planet["hole"] == 0.0
    fixed = cc.parse_scene("env=image[prior=tsv,w=3] + c=stamp[at=1.5:-2]")
    assert fixed[0].opts["prior"] == "tsv" and fixed[0].opts["w"] == 3.0 and fixed[1].opts["at"] == (1.5, -2.0)
    assert cc.scene_tag(fixed) == "_c-envitsv-cs"


@GUARD
@pytest.mark.parametrize("text", [
    "star=point",  # no image
    "a=point + b=ud + c=image",  # two stars
    "a=image + a=stamp",  # repeated name
    "a=blob",  # unknown kind
    "a=image[colour=red]",  # unknown key
    "a=image[prior=clean]",  # not a prior
    "a=image[at=1:2]",  # an image is centred
    "a=image + b=stamp[at=1]",  # at needs two numbers
    "a=image[hole=0.5",  # unclosed
    "s=point[fov=2] + a=image",  # a star takes no options
    "a=image +",  # empty part
])
def test_parse_scene_rejects(text):
    with pytest.raises(ValueError):
        cc.parse_scene(text)


@GUARD
def test_composite_and_comma_scenes_are_told_apart():
    assert cc.is_composite("star=point + env=image") and not cc.is_composite("point,image")
    assert ci.scene_options("point,image")["star"]  # the old grammar is unchanged


@GUARD
def test_odd_grid_and_support():
    assert cc.odd_grid(10.0, 1.0, 256) == (11, 1.0)
    n, pixel = cc.odd_grid(100.0, 0.1, 101)
    assert n == 101 and np.isclose(pixel, 100.0 / 101)  # coarsened to fit the cap
    assert cc.support_mask(11, 1.0, 0.0, None) is None  # the whole grid
    keep = np.asarray(cc.support_mask(11, 1.0, 2.0, 4.0))
    x = np.arange(11) - 5.0
    r = np.hypot(x[None, :], x[:, None])
    assert np.array_equal(keep, (r >= 2.0) & (r <= 4.0))
    with pytest.raises(ValueError):
        cc.support_mask(5, 1.0, 10.0, None)


@GUARD
def test_joint_penalty_is_the_weighted_sum_of_its_members():
    import jax.numpy as jnp
    import virgil.models as vm

    y, x = np.mgrid[:12, :12] - 6.0
    image = vm.Image(jnp.asarray(-(x**2 + y**2) / 8.0 + 0.1 * np.sin(x)), 1.0, flux=1.0)
    model = vm.System(a=image, b=vm.Image(jnp.asarray(0.2 * np.cos(y)), 1.0, flux=0.5))
    pa, pb = ci.regulariser("tsv", 3.0, "a"), ci.regulariser("tsv", 0.5, "b")
    joint = cc.Joint(jnp.asarray(2.0), (pa, pb))
    assert np.isclose(joint.value(model), 2.0 * (pa.value(model) + pb.value(model)))
    if hasattr(pa, "residuals"):  # least squares, virgil's convention: ½ Σ residuals² is the value
        assert np.isclose(0.5 * np.sum(np.square(joint.residuals(model))), joint.value(model), rtol=1e-6)
    mixed = cc.Joint(jnp.asarray(1.0), (pa, ci.regulariser("mem", 1.0, "b")))
    assert hasattr(mixed, "residuals") == all(hasattr(m, "residuals") for m in mixed.members)


@GUARD
def test_validate_routes_a_composite_scene():
    parser = ci.build_parser()
    args = parser.parse_args(["--task", "0", "--scene", "star=point + env=image"])
    _, star, scene = ci.validate(parser, args)
    assert star is None and [p.name for p in scene["composite"]] == ["star", "env"]
    with pytest.raises(SystemExit):
        ci.validate(parser, parser.parse_args(["--task", "0", "--scene", "env=image", "--prior", "gp"]))


@GUARD
def test_bench_composite_arms_parse_and_are_opt_in():
    import contest_bench as cb

    assert cb.COMPOSITE and set(cb.COMPOSITE) <= set(cb.OPT_IN)
    assert not set(cb.COMPOSITE) & set(cb.DEFAULT_CONFIGS)  # earlier runs' task numbering is unchanged
    for name in cb.COMPOSITE:
        cc.parse_scene(cb.CONFIGS[name]["scene"])



TEMPLATE = pathlib.Path("~/data/imaging_contests/2004/2004-data2.fits").expanduser()


@pytest.mark.validates("pipeline:contest-imaging", "virgil.grid_fit.linear_flux_grid", roots=["self-consistency"],
                       kind="reference")
@pytest.mark.skipif(not TEMPLATE.exists(), reason="contest data not fetched")
def test_marginal_search_finds_a_companion_beside_a_gp(tmp_path):
    """A star with a small GP image on it, and a companion at (6, -3) mas of
    flux 0.1 in the data: the map with the GP's latents marginalized peaks at
    the companion; without a GP it is virgil's linear_flux_grid."""
    import virgil.models as vm
    from crosscheck import sky
    from virgil.fields import GaussianField
    from virgil.grid_fit import linear_flux_grid
    from virgil.oidata import OIData

    import contest_bench as cb

    path = tmp_path / "binary.fits"
    cb.simulate_file(TEMPLATE, path, sky.mix([sky.point(), sky.point(6.0, -3.0)], [1.0, 0.1]), np.random.default_rng(7))
    data = OIData(str(path))
    env = vm.Image(GaussianField(np.zeros((9, 9)), 1.0, 2.0), 1.0, flux=0.05)
    sc = cc.Scene([], vm.System(star=vm.PointSource(), env=env), {}, {"env": (2.0, 1.0)}, {})
    axis = np.arange(-9.0, 9.01, 1.0)
    pos = np.stack([x.ravel() for x in np.meshgrid(axis, axis, indexing="ij")], 1)
    flux, snr = cc.marginal_flux_map(sc, data, pos)
    peak = np.argmax(snr)
    assert np.hypot(*(pos[peak] - (6.0, -3.0))) <= 1.0 and flux[peak] == pytest.approx(0.1, rel=0.5)

    small = np.array([-3.0, 0.0, 6.0])
    pos = np.stack([x.ravel() for x in np.meshgrid(small, small, indexing="ij")], 1)
    flux0, _ = cc.marginal_flux_map(cc.Scene([], vm.System(star=vm.PointSource()), {}, {}, {}), data, pos)
    grid = linear_flux_grid(vm.System(star=vm.PointSource(), comp=vm.PointSource(1e-6)), data,
                            {"comp.dra": small, "comp.ddec": small, "comp.flux": np.array([1e-6])})
    assert np.allclose(flux0, np.ravel(np.asarray(grid.flux)), rtol=1e-3, atol=1e-6, equal_nan=True)


@pytest.mark.validates("pipeline:contest-imaging", roots=["self-consistency"], kind="guard")
@pytest.mark.skipif(not TEMPLATE.exists(), reason="contest data not fetched")
def test_build_fit_and_stamp_a_composite_scene(tmp_path):
    """build_scene, a few fit steps, the joint evidence and a stamp added at
    the search's peak, on a simulated binary (a hand-made setup, not setup())."""
    import virgil.models as vm
    from crosscheck import sky
    from virgil.imaging import beam
    from virgil.oidata import OIData

    import contest_bench as cb

    path = tmp_path / "binary.fits"
    cb.simulate_file(TEMPLATE, path, sky.mix([sky.point(), sky.point(6.0, -3.0)], [1.0, 0.1]), np.random.default_rng(7))
    data = OIData(str(path))
    res = beam(data)
    n, pixel = 31, res.minor_mas / 3  # a 15.5 mas field: the search covers the companion
    img0 = vm.Image(np.zeros((n, n)), pixel, flux=0.05)
    s = dict(resolution=res, data=data, npts=1000, img0=img0, npix=n, pixel=pixel, star=True,
             start=vm.System(star=vm.PointSource(), env=img0), star_priors={})
    parts = cc.parse_scene("star=point + env=image[ell=1,sigma=1] + comp=stamp[fov=1,pixel=0.25]")
    sc = cc.build_scene(parts, s)
    env = sc.model.components["env"]
    assert set(sc.model.names) == {"star", "env"} and set(sc.gp) == {"env"}
    assert env.support is not None and not bool(np.asarray(env.support)[n // 2, n // 2])  # the star's hole
    assert {"env.flux", "env.log_brightness.latent"} <= set(sc.priors)
    sc, r = cc.fit_scene(sc, data, max_steps=5)
    log_z, occam = cc.log_evidence_all(sc, r, data)
    assert np.isfinite(log_z) and np.isfinite(occam)
    held = cc.Scene(sc.parts, sc.model, {"env.flux": sc.priors["env.flux"]}, {}, {})  # env as if penalised
    assert np.isfinite(cc.log_evidence_all(held, r, data)[0])  # conditional on env: -½χ² - norm + Occam
    peaks = cc.search_candidates(sc, data, s, k=1)
    assert peaks and np.hypot(peaks[0][0] - 6.0, peaks[0][1] + 3.0) <= res.minor_mas
    with_stamp = cc.add_stamp(sc, parts[2], s, *peaks[0][:3])
    assert set(with_stamp.gp) == {"env", "comp"} and {"comp.dra", "comp.ddec", "comp.flux"} <= set(with_stamp.priors)
    with_stamp, r2 = cc.fit_scene(with_stamp, data, init=r.values, max_steps=5)
    assert np.isfinite(cc.log_evidence_all(with_stamp, r2, data)[0])
