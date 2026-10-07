"""The star arms' saved images and their star-or-no-star choice (C2g).

The smoke run of the star arms (job 18207630) found three problems: the
image saved for scoring did not reproduce the fit (its own χ²/N was 3-11
against the fit's 1.2: SourceModel.render resamples the fit's pixels
bilinearly onto the reference grid, which smooths them); the saved npz
lacked the fitted star's parameters; and the star decomposition was
degenerate, with no test that the star earned its parameters. Here the
reference image is rendered by Fourier synthesis (``render_scene``) and its
direct NumPy DFT (crosscheck.sky) must reproduce the model's χ²; every
component, the companion included, must be in it and in ``star_record``;
and the Occam-penalised evidence (``log_evidence_full``) must keep a star
the data need and reject one they do not. The fits are tiny (a 13 x 13
Gaussian-field image).
"""

import importlib.util
import pathlib

import numpy as np
import pytest

from crosscheck import sky

ROOT = pathlib.Path(__file__).parents[1]
TEMPLATE = pathlib.Path("~/data/imaging_contests").expanduser() / "2004" / "2004-data2.fits"
needs_2004 = pytest.mark.skipif(not TEMPLATE.exists(), reason="contest data not fetched")
LAM, BMAX = 0.55e-6, 60.0  # an NPOI-like coverage, as 2004's: a 1.9 mas beam


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scene():
    """An elliptical limb-darkened star with a point companion, plus a
    rotated pixel image on a coarse grid (as the GP's), in a System as the
    star arms build it."""
    import virgil.models as vm

    n, pixel = 21, 0.35
    x = (np.arange(n) - (n - 1) / 2) * pixel
    xx, yy = np.meshgrid(x, x)
    eta = np.log(np.exp(-0.5 * ((xx - 1.0) ** 2 / 2.0 + (yy + 0.5) ** 2 / 0.5))
                 + 0.3 * np.exp(-0.5 * ((xx + 1.5) ** 2 + (yy - 1.5) ** 2) / 0.3))
    star = vm.System(star=vm.EllipticalLimbDarkenedDisk(3.0, ratio=0.7, pa=150.0, u=(0.5,)),
                     comp=vm.PointSource(0.05, dra=3.33, ddec=-2.17))
    return vm.System(star=star, env=vm.Image(eta, pixel, rotation_deg=30.0, flux=0.6))


def _observables(vis_fn, u1, v1, u2, v2):
    """V² on the three baselines of each triangle, and closure phases."""
    v_1, v_2, v_3 = vis_fn(u1, v1), vis_fn(u2, v2), vis_fn(-(u1 + u2), -(v1 + v2))
    return np.abs(np.concatenate([v_1, v_2, v_3])) ** 2, np.angle(v_1 * v_2 * v_3)


def _chi2_red(model_obs, data):
    (v2, cp), (d_v2, e_v2, d_cp, e_cp) = model_obs, data
    r_cp = np.angle(np.exp(1j * (cp - d_cp))) / e_cp
    return (np.sum(((v2 - d_v2) / e_v2) ** 2) + np.sum(r_cp**2)) / (v2.size + cp.size)


@pytest.mark.validates("pipeline:contest-imaging", "virgil.models.Image", roots=["mathematics"], kind="reference")
def test_saved_image_reproduces_the_fitted_chi2():
    # Fix A: the DFT of the image saved on the reference grid gives the χ²/N
    # of the model itself (what a fit logs), within 10%; the bilinear
    # resampling it replaces does not.
    ci = _load("contest_images")
    model = _scene()
    rng = np.random.default_rng(3)
    r, t = np.exp(rng.uniform(np.log(3.0), np.log(BMAX), (2, 150))), rng.uniform(0, 2 * np.pi, (2, 150))
    u1, v1, u2, v2 = r[0] * np.sin(t[0]), r[0] * np.cos(t[0]), r[1] * np.sin(t[1]), r[1] * np.cos(t[1])
    true_v2, true_cp = _observables(lambda u, v: np.asarray(model.model(u, v, LAM)), u1, v1, u2, v2)
    e_v2, e_cp = 0.001 + 0.01 * true_v2, np.full(true_cp.shape, np.radians(1.0))
    data = (true_v2 + e_v2 * rng.standard_normal(true_v2.shape), e_v2,
            true_cp + e_cp * rng.standard_normal(true_cp.shape), e_cp)
    logged = _chi2_red((true_v2, true_cp), data)
    fov, npix = 30.0, 257
    q_max = max(np.max(np.hypot(a, b)) for a, b in ((u1, v1), (u2, v2), (u1 + u2, v1 + v2))) / LAM * sky.MAS
    rendered = ci.render_scene(model, npix, fov, q_max)
    saved = {}
    for name, image in (("render_scene", rendered),
                        ("bilinear", np.asarray(model.render(npix, fov)))):
        cloud = sky.pixel_image(image, fov / npix)
        saved[name] = _chi2_red(_observables(lambda u, v: sky.visibility(cloud, u, v, LAM), u1, v1, u2, v2), data)
    assert 0.5 < logged < 2.0
    assert saved["render_scene"] == pytest.approx(logged, rel=0.1)
    assert saved["bilinear"] > 1.5 * logged  # the smoke run's failure, reproduced
    assert np.sum(rendered) == pytest.approx(1.0)
    assert np.sum(np.clip(rendered, None, 0.0)) > -0.02  # smoothed beyond the data, not ringing


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_every_component_is_drawn_and_recorded():
    # Fix B: the companion is in the saved image (its share of the flux,
    # where it is), and the npz record has the star and the companion.
    import virgil.models as vm

    ci = _load("contest_images")
    model = _scene()
    alone = vm.System(star=vm.System(star=model.star.star), env=model.env)
    fov, npix = 30.0, 257
    pixel = fov / npix
    q_max = BMAX / LAM * sky.MAS
    diff = ci.render_scene(model, npix, fov, q_max) - ci.render_scene(alone, npix, fov, q_max)
    col = int(round((npix - 1) / 2 - 3.33 / pixel))  # East is left
    row = int(round((npix - 1) / 2 + 2.17 / pixel))  # North is up
    i, j = np.unravel_index(np.argmax(diff), diff.shape)
    assert abs(i - row) <= 1 and abs(j - col) <= 1
    # diff = share x (companion - disk): the disk is far from the companion,
    # so a box about it holds the companion's share of the flux.
    share = (0.05 / 1.05) / 1.6  # within the star (weight 1), beside the env (0.6)
    assert np.sum(diff[row - 6:row + 7, col - 6:col + 7]) == pytest.approx(share, rel=0.1)
    rec = ci.star_record(model)
    assert rec["star_frac"] == pytest.approx(1 / 1.6)
    assert (rec["star_diam"], rec["star_ratio"], rec["star_pa"]) == pytest.approx((3.0, 0.7, 150.0))
    assert (rec["comp_dra"], rec["comp_ddec"], rec["comp_flux"]) == pytest.approx((3.33, -2.17, 0.05))
    point = ci.star_record(vm.System(star=vm.PointSource(), env=model.env))
    assert point["star_frac"] == pytest.approx(1 / 1.6) and np.isnan(point["star_diam"]) and np.isnan(point["comp_flux"])
    assert np.isnan(ci.star_record(model.env)["star_frac"])


def _gp_fit(ci, data, star, n=13, pixel=0.5):
    """A tiny GP fit (σ 2, ℓ 1.5 mas) with or without a point star; returns
    the FitResult and the non-image priors."""
    import numpyro.distributions as dist
    import virgil.models as vm
    from virgil.fields import GaussianField
    from virgil.fitting import fit
    from virgil.imaging import image_priors

    env = vm.Image(GaussianField(np.zeros((n, n)), 2.0, 1.5), pixel, flux=1.0)
    scene = vm.System(star=vm.PointSource(), env=env) if star else vm.System(env=env)
    extra = {"env.flux": dist.LogUniform(ci.FLUX_FLOOR, ci.FLUX_CAP)} if star else {}
    return fit(scene, image_priors(scene) | extra, data, max_steps=300), extra


@pytest.mark.validates("pipeline:contest-imaging", roots=["self-consistency"], kind="reference")
@needs_2004
@pytest.mark.parametrize("star_frac", [0.0, 0.5])
def test_evidence_keeps_a_needed_star_and_rejects_a_spare_one(tmp_path, star_frac):
    # Fix C: on a resolved Gaussian (2 mas FWHM, about a beam) with no star,
    # the star's evidence gain is below ln 100 and it is rejected; with half
    # the flux in a point star it is kept. The Occam term is a few nats at
    # most, and about ½ ln(π/6) for a parameter the data do not constrain.
    from virgil.oidata import OIData

    bench, ci = _load("contest_bench"), _load("contest_images")
    gauss = sky.elliptical_gaussian(2.0, 0.8, 30.0)
    cloud = sky.mix([sky.point(), gauss], [star_frac, 1 - star_frac]) if star_frac else gauss
    path = tmp_path / "sim.fits"
    bench.simulate_file(TEMPLATE, path, cloud, np.random.default_rng(5))
    data = OIData(str(path))
    with_star, extra = _gp_fit(ci, data, True)
    without, _ = _gp_fit(ci, data, False)
    log_z_star, occam = ci.log_evidence_full(with_star, data, extra)
    log_z_none, occam_none = ci.log_evidence_full(without, data, {})
    assert occam_none == 0.0 and -10.0 < occam < 0.0
    log_bayes = log_z_star - log_z_none
    if star_frac:
        assert log_bayes >= ci.STAR_LOG_BAYES
    else:
        assert log_bayes < ci.STAR_LOG_BAYES
