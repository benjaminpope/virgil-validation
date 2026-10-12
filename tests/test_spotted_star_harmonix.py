"""The harmonix model of the 2004 contest's spotted star
(scripts/spotted_star_harmonix.py): the elliptical wrapper's conventions
against virgil's own models, the spot sign convention, MacKay's error scale on
a linear model, recovery on simulated data with data2's uv coverage, and the
command line. Small maps (ydeg <= 3) and a handful of visibilities only."""

import importlib.util
import pathlib

import numpy as np
import pytest

pytest.importorskip("harmonix")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "spotted_star_harmonix", pathlib.Path(__file__).parents[1] / "scripts" / "spotted_star_harmonix.py"
)
ssh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ssh)

DATA2 = pathlib.Path("~/data/imaging_contests/2004/2004-data2.fits").expanduser()
U = np.array([1.2e7, 3.1e7, -2.4e7, 4.0e7, -0.7e7, 0.0])  # wavelengths
V = np.array([2.0e7, -1.3e7, 3.6e7, 0.5e7, -4.2e7, 0.0])


def _star(ydeg=2, radius=2.0, u1=0.4, ratio=1.0, pa=0.0, seed=0, amplitude=0.1):
    data = amplitude * np.random.default_rng(seed).normal(size=(ydeg + 1) ** 2 - 1)
    return ssh.star_from(ssh.template_star(ydeg), radius, u1, ratio, pa, data)


@pytest.mark.validates("pipeline:contest-imaging", "virgil.models.HarmonixModel", roots=["self-consistency"],
                       kind="reference")
def test_ratio_one_is_harmonix_model_for_any_pa():
    import virgil.models as vm

    for pa in (0.0, 37.0, 140.0):
        star = _star(ratio=1.0, pa=pa)
        plain = vm.HarmonixModel(star.source, observation_time=0.0)
        np.testing.assert_allclose(np.asarray(star.model(U, V, 1.0)), np.asarray(plain.model(U, V, 1.0)),
                                   rtol=1e-12, atol=1e-14)
        if pa == 37.0:  # the image too (one angle: rendering is the slow part)
            np.testing.assert_allclose(np.asarray(star.render(33, 6.0)), np.asarray(plain.render(33, 6.0)), atol=1e-14)


@pytest.mark.validates("pipeline:contest-imaging", "virgil.models.EllipticalLimbDarkenedDisk",
                       roots=["self-consistency"], kind="reference")
def test_stretched_unspotted_star_is_the_elliptical_limb_darkened_disk():
    import virgil.models as vm

    for ratio, pa in ((0.6, 30.0), (0.35, 125.0)):
        star = _star(radius=2.0, u1=0.4, ratio=ratio, pa=pa, amplitude=0.0)
        disk = vm.EllipticalLimbDarkenedDisk(4.0, ratio=ratio, pa=pa, u=[0.4])
        np.testing.assert_allclose(np.asarray(star.model(U, V, 1.0)), np.asarray(disk.model(U, V, 1.0)),
                                   rtol=1e-9, atol=1e-12)
        # The rendered outline too: same pixels lit, same profile.
        np.testing.assert_allclose(np.asarray(star.render(41, 6.0)), np.asarray(disk.render(41, 6.0)), atol=2e-4)


@pytest.mark.validates("pipeline:contest-imaging", roots=["self-consistency"], kind="reference")
def test_spot_level_sign_and_position():
    """level > 1 is a bright spot, < 1 a dark one (jaxoplanet's contrast is
    1 - level); a spot at positive latitude and longitude lies North-East."""
    spot = ssh.ylm_spot(3)
    xx = (16 - np.arange(33)) * (6.0 / 33)  # East to the left: column 0 is the most positive dra
    x, y = np.meshgrid(xx, xx)
    plain = np.asarray(ssh.star_from(ssh.template_star(3), 2.0, 0.4, 1.0, 0.0, np.zeros(15)).render(33, 6.0))
    for level, sign in ((4.0, 1.0), (0.2, -1.0)):
        data = ssh.spot_map(spot, [(level, 0.6, 0.5, 0.5)])
        img = np.asarray(ssh.star_from(ssh.template_star(3), 2.0, 0.4, 1.0, 0.0, data).render(33, 6.0))
        diff = sign * (img / img.sum() - plain / plain.sum())
        weight = np.clip(diff, 0, None)
        assert np.sum(weight * x) > 0 and np.sum(weight * y) > 0  # East (+dra) and North (+ddec)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_mackay_error_scale_on_a_linear_model():
    """Flat priors on k measured parameters: s² = χ² / (N - k) exactly; a
    tight Gaussian prior on one of them removes it from γ."""
    rng = np.random.default_rng(1)
    a = rng.normal(size=(40, 3))
    y = a @ np.array([1.0, -2.0, 0.5]) + 0.7 * rng.normal(size=40)
    theta = jnp.asarray(np.linalg.lstsq(a, y, rcond=None)[0])
    residuals = lambda t: jnp.asarray(a) @ t - jnp.asarray(y)  # noqa: E731
    chi2 = float(jnp.sum(residuals(theta) ** 2))
    s, gamma = ssh.mackay_error_scale(residuals, theta, np.zeros(3))
    assert gamma == pytest.approx(3.0, abs=1e-8)
    assert s == pytest.approx(np.sqrt(chi2 / 37), rel=1e-8)
    _, gamma_tight = ssh.mackay_error_scale(residuals, theta, np.array([0.0, 0.0, 1e9]))
    assert gamma_tight == pytest.approx(2.0, abs=1e-3)


@pytest.mark.validates("pipeline:contest-imaging", "virgil.models.HarmonixModel", roots=["self-consistency"],
                       kind="reference")
@pytest.mark.skipif(not DATA2.exists(), reason="contest data not fetched")
def test_simulated_spotted_star_prefers_its_truth():
    """A one-spot elliptical harmonix star plus companion on data2's uv
    coverage: χ² at the truth beats χ² with the spot moved, the outline
    turned, or the spot made dark."""
    from virgil.oidata import OIData

    companion = {"dra": 6.0, "ddec": 1.0, "flux": 0.1}
    scene = ssh.Scene("spots1", 3, companion)
    truth = dict(radius=2.6, u=0.5, ratio=0.7, pa=120.0, comp_dra=6.0, comp_ddec=1.0, comp_flux=0.1,
                 spot0_level=4.0, spot0_size=0.5, spot0_lat=-0.6, spot0_lon=0.4, v2_scale=1.0)
    data = OIData(str(DATA2)).with_model(scene(**truth), key=jax.random.PRNGKey(0))
    chi2_truth, n = ssh.chi2_of(scene(**truth), data)
    assert chi2_truth / n < 2.0
    for change in ({"spot0_lat": 0.6}, {"spot0_lon": -0.4}, {"pa": 30.0}, {"spot0_level": 0.25}):
        chi2, _ = ssh.chi2_of(scene(**(truth | change)), data)
        assert chi2 > chi2_truth + 25.0, change


@pytest.mark.validates("pipeline:contest-imaging", roots=["self-consistency"], kind="reference")
def test_command_line():
    args = ssh.parse_args(["--config", "map", "--ydeg", "4", "--smoke", "--out", "x"])
    assert (args.config, args.ydeg, args.smoke, args.out) == ("map", 4, True, "x")
    args = ssh.parse_args(["--config", "spots2"])
    assert args.ydeg is None and not args.smoke and args.data.endswith("2004-data2.fits")
    assert ssh.DEFAULT_YDEG["spots2"] == 12 and ssh.DEFAULT_YDEG["map"] == 4
    with pytest.raises(SystemExit):
        ssh.parse_args(["--config", "spots3"])
    with pytest.raises(SystemExit):
        ssh.parse_args(["--config", "map", "--ydeg", "0"])
    # The prior on the map's coefficients falls as C_l = σ² (1 + l)^-2.
    np.testing.assert_allclose(ssh.map_prior_scales(2, 1.0), [1 / 2] * 3 + [1 / 3] * 5)


@pytest.mark.validates("pipeline:contest-imaging", roots=["self-consistency"], kind="reference")
def test_v2_scale_multiplies_v2_and_keeps_closure_phases():
    inner = _star(ratio=0.7, pa=40.0)
    scaled = ssh.V2Scaled(inner, 1.21)
    v, v0 = np.asarray(scaled.model(U, V, 1.0)), np.asarray(inner.model(U, V, 1.0))
    np.testing.assert_allclose(np.abs(v) ** 2, 1.21 * np.abs(v0) ** 2, rtol=1e-12)
    np.testing.assert_allclose(np.angle(v), np.angle(v0), atol=1e-12)
    np.testing.assert_allclose(np.asarray(scaled.render(17, 6.0)), np.asarray(inner.render(17, 6.0)))
    assert float(ssh.V2Scaled(inner).scale) == 1.0


@pytest.mark.validates("pipeline:contest-imaging", roots=["self-consistency"], kind="reference")
def test_scene_applies_v2_scale_and_frozen_parameters():
    """A Scene's v2_scale scales V²; ``fixed`` parameters need not be passed
    and give the same model as passing them."""
    outline = dict(radius=2.2, u=0.4, ratio=0.7, pa=50.0)
    free = dict(v2_scale=1.3, data=0.05 * np.random.default_rng(2).normal(size=8))
    whole = ssh.Scene("map", 2, None)(**outline, **free)
    frozen = ssh.Scene("map", 2, None, fixed=outline)(**free)
    np.testing.assert_allclose(np.asarray(frozen.model(U, V, 1.0)), np.asarray(whole.model(U, V, 1.0)), rtol=1e-13)
    unit = ssh.Scene("map", 2, None, fixed=outline)(**(free | {"v2_scale": 1.0}))
    np.testing.assert_allclose(np.abs(np.asarray(whole.model(U, V, 1.0))) ** 2,
                               1.3 * np.abs(np.asarray(unit.model(U, V, 1.0))) ** 2, rtol=1e-12)


@pytest.mark.validates("pipeline:contest-imaging", roots=["self-consistency"], kind="reference")
@pytest.mark.skipif(not DATA2.exists(), reason="contest data not fetched")
def test_v2_scale_is_recovered_and_matters():
    """Data2's uv coverage with V² 15% above a normalized model: the
    uniform-disk scan finds the scale and radius, and χ² at the
    truth is far worse without it."""
    import virgil.models as vm
    from virgil.oidata import OIData

    truth = vm.UniformDisk(2.8)
    data = OIData(str(DATA2)).with_model(ssh.V2Scaled(truth, 1.15), key=jax.random.PRNGKey(3))
    radius, scale = ssh.uniform_disk_fit(data, 2.0, 12.0, n=150)
    assert scale == pytest.approx(1.15, abs=0.03) and radius == pytest.approx(1.4, abs=0.1)
    chi2_scaled, n = ssh.chi2_of(ssh.V2Scaled(truth, 1.15), data)
    chi2_plain, _ = ssh.chi2_of(truth, data)
    assert chi2_scaled / n < 2.0 and chi2_plain / n > 10.0


@pytest.mark.validates("pipeline:contest-imaging", roots=["self-consistency"], kind="reference")
def test_on_bounds_reports_parameters_on_a_prior_bound():
    import numpyro.distributions as dist

    priors = {"level": dist.LogUniform(0.05, 20.0), "u": dist.Uniform(0.0, 1.0), "ratio": dist.Uniform(0.2, 1.0),
              "scale": dist.LogUniform(0.5, 2.0), "data": dist.Normal(jnp.zeros(2), 1.0)}
    values = {"level": 19.98, "u": 0.0002, "ratio": 0.7, "scale": 1.15, "data": np.zeros(2)}
    assert ssh.on_bounds(values, priors) == ["level", "u"]
    assert ssh.on_bounds(values | {"level": 1.0, "u": 0.3}, priors) == []
    assert ssh.on_bounds(values | {"u": 0.02, "level": 17.0}, priors) == []  # outside the 1% tolerance
