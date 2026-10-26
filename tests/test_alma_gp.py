"""The IM Lup GP image (``virgil_bridge.alma_gp``, ``scripts/alma_imlup_gp.py``) on a small synthetic set (CPU, seconds).

Stage 0 of the GP plan: ``Image(MultipoleField(...))`` goes through virgil's Toeplitz chi^2 (``ComplexVisData.toeplitz``) with the gate's nuisances and no taper,
and the loss and every gradient leaf equal the direct chi^2's. Skipped on a virgil without ``radio_toeplitz`` or ``MultipoleField``."""

import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("virgil.radio_toeplitz")
pytest.importorskip("virgil.fields")

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import virgil_bridge.alma as va  # noqa: E402
import virgil_bridge.alma_gp as vg  # noqa: E402

if not hasattr(__import__("virgil.fields", fromlist=["x"]), "MultipoleField"):
    pytest.skip("virgil has no MultipoleField", allow_module_level=True)

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from test_alma_maxent import synthetic_dir  # noqa: E402

NPIX, N_R, N_TH = 64, 12, 24
FIELD = dict(npix=NPIX, n_r=N_R, n_theta=N_TH, r_max_mas=140.0, length_r=20.0, length_theta=1.0, inc=47.5, pa=144.5)


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    return synthetic_dir(tmp_path_factory.mktemp("gp"), n=1000, n_obs=2, per_obs=2)  # 4000 visibilities, two observations


def _rel(a, b):
    return float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))


def _values(prob, seed=1, scale=1.0):
    rng = np.random.default_rng(seed)
    v = {k: jnp.asarray(x) for k, x in prob.init.items()}
    v["field.latent"] = jnp.asarray(scale * rng.normal(size=(N_R, N_TH)))
    v["field.sigma"] = jnp.asarray(1.3)
    v["cal.flux"] = jnp.asarray(0.2)
    if "cal.gain" in v:
        v["cal.gain"] = v["cal.gain"] * (1.0 + 0.03 * rng.normal(size=v["cal.gain"].shape))
        v["cal.offset"] = jnp.asarray(rng.uniform(-4, 4, size=v["cal.offset"].shape))
        v["cal.sigma"] = v["cal.sigma"] * (1.0 + 0.1 * rng.uniform(size=v["cal.sigma"].shape))
    return v


def _vg(term, values):
    return eqx.filter_value_and_grad(lambda x: vg.term_loss(term, x))(values)


@pytest.mark.validates("virgil.radio_toeplitz.ToeplitzVisData", roots=["self-consistency"])
def test_the_multipole_field_through_the_toeplitz_chi2_equals_the_direct_chi2_in_value_and_gradient(syn):
    """Stage 0 of the GP plan: ``Image(MultipoleField)`` (no taper, gains, offsets and per-ddid sigma free, reference observation fixed) at a random latent: the
    Toeplitz loss equals the direct one to 1e-8 relative, and so does every gradient leaf (latent, sigma, flux, gains, offsets, sigmas) to 1e-8 of its own largest component."""
    data = va.load_dir(syn)
    direct = vg.build_gp_problem(data, chi2="direct", **FIELD)
    toep = vg.build_gp_problem(data, chi2="toeplitz", **FIELD)
    assert direct.nuis.taper_fwhm_mas is None and toep.chi2_info["engine"] == "toeplitz" and toep.chi2_info["offsets"] == "spread"
    assert set(direct.init) == {"field.latent", "field.sigma", "cal.flux", "cal.gain", "cal.offset", "cal.sigma"}
    v = _values(direct)
    with va.ft_eps(1e-12):
        ld, gd = _vg(direct.terms[0], v)
        lt, gt = _vg(toep.terms[0], v)
    assert float(abs(ld - lt) / abs(ld)) < 1e-8, (float(ld), float(lt))
    worst = {k: _rel(gt[k], gd[k]) for k in gd}
    assert all(r < 1e-8 for r in worst.values()), worst
    assert float(jnp.max(jnp.abs(gd["field.latent"]))) > 0 and float(jnp.max(jnp.abs(gd["field.sigma"]))) > 0  # the comparison is not of zeros
    from evidence.plugin import record

    record("gp_toeplitz_loss_rel_diff", float(abs(ld - lt) / abs(ld)))
    record("gp_toeplitz_grad_rel_diff_max", max(worst.values()))


@pytest.mark.validates("virgil_bridge.alma_gp", roots=["self-consistency"], kind="control")
def test_the_toeplitz_comparison_sees_a_different_field(syn):
    """Control: the same comparison with the latent of one engine perturbed by 1e-3 differs by far more than the 1e-8 of the check, so it can fail."""
    data = va.load_dir(syn)
    direct = vg.build_gp_problem(data, chi2="direct", **FIELD)
    toep = vg.build_gp_problem(data, chi2="toeplitz", **FIELD)
    v = _values(direct)
    w = {**v, "field.latent": v["field.latent"] + 1e-3}
    with va.ft_eps(1e-12):
        ld, _ = _vg(direct.terms[0], v)
        lt, _ = _vg(toep.terms[0], w)
    assert float(abs(ld - lt) / abs(ld)) > 1e-6


@pytest.mark.validates("virgil_bridge.alma_gp", roots=["self-consistency"], kind="guard")
def test_held_nuisances_give_the_full_problems_loss_and_latent_gradient(syn):
    """Stage 1 holds gains, offsets and sigmas: its term at (latent, sigma, flux) equals the free problem's at the same values with those held, in loss and in the
    gradient of the fitted sites; the held sites are not parameters."""
    data = va.load_dir(syn)
    free = vg.build_gp_problem(data, chi2="toeplitz", **FIELD)
    v = _values(free)
    held = vg.hold_nuisances(free, {k: v[k] for k in vg.HELD_SITES})
    assert set(held.init) == {"field.latent", "field.sigma", "cal.flux"} and set(held.priors) == set(held.init) and held.held
    vh = {k: v[k] for k in held.init}
    with va.ft_eps(1e-12):
        lf, gf = _vg(free.terms[0], v)
        lh, gh = _vg(held.terms[0], vh)
    assert float(abs(lf - lh) / abs(lf)) < 1e-12
    assert all(_rel(gh[k], gf[k]) < 1e-10 for k in gh)
    with pytest.raises(ValueError, match="lack"):
        vg.hold_nuisances(free, {"cal.gain": v["cal.gain"]})


@pytest.mark.validates("virgil_bridge.alma_gp", roots=["self-consistency"], kind="guard")
def test_image_of_is_the_flux_times_the_unit_sum_brightness_and_flux_beyond_counts_the_outer_pixels(syn):
    data = va.load_dir(syn)
    prob = vg.build_gp_problem(data, chi2="direct", **FIELD)
    v = _values(prob)
    img = vg.image_of(prob, v)
    assert img.shape == (NPIX, NPIX) and img.sum() * (prob.pixel_mas / 1000.0) ** 2 == pytest.approx(0.2, rel=1e-9)
    assert vg.flux_beyond(img, prob.pixel_mas, -1.0) == pytest.approx(0.2, rel=1e-9)  # every pixel is beyond a negative radius
    assert vg.flux_beyond(img, prob.pixel_mas, 10.0) == 0.0
    dark = np.zeros((NPIX, NPIX))
    dark[0, 0] = 1.0  # a corner pixel is ~0.22 arcsec from the centre cell: inside 0.3, beyond 0.2
    assert vg.flux_beyond(dark, 5.0, 0.2) > 0 and vg.flux_beyond(dark, 5.0, 0.3) == 0.0


# ------------------------------------------------------------------ the script (alma_imlup_gp.py) end to end, toy size

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import alma_imlup_gp as script  # noqa: E402
from crosscheck import alma_maxent as cx  # noqa: E402

SMALL = ["--npix", "32", "--n-r", "8", "--n-theta", "12", "--r-max-mas", "70", "--length-r-mas", "15", "--length-theta-rad", "1", "--max-steps", "8", "--time-limit", "300"]


def _fake_clean(path, npix=121, pixel_mas=3.0, beam=(40.0, 30.0, 20.0), flux=0.2):
    """A CLEAN output directory with only ``image.fits`` (Jy/beam, a Gaussian blob on the middle pixel, the beam in its header)."""
    path.mkdir(parents=True, exist_ok=True)
    k = (np.arange(npix) - (npix - 1) / 2) * pixel_mas
    e, n = np.meshgrid(k, k)
    blob = np.exp(-0.5 * (e**2 + n**2) / 25.0**2)
    jy_beam = blob / blob.sum() * flux * cx.beam_area_arcsec2(beam[0], beam[1]) / (pixel_mas / 1000.0) ** 2
    cx.write_fits(path / "image.fits", jy_beam, pixel_mas, 0.0, 0.0, bunit="Jy/beam", BMAJ=beam[0] / 3.6e6, BMIN=beam[1] / 3.6e6, BPA=beam[2])
    return path, flux, beam


@pytest.mark.validates("virgil_bridge.alma_gp", roots=["mathematics"])
def test_restored_clean_is_in_jy_per_arcsec2_on_the_fit_grid_with_its_flux_and_beam(tmp_path):
    """The restored CLEAN image (Jy/beam, a Gaussian blob) regridded to the fit's grid: its flux is the blob's inside the field (to 1 %) and is conserved by the resampling, its peak is on the
    phase-centre cell, and the beam is the header's."""
    d, flux, beam = _fake_clean(tmp_path / "clean")
    img, b, info = vg.restored_clean(d, npix=32, pixel_mas=5.0)
    assert img.shape == (32, 32) and b == pytest.approx(beam, rel=1e-6)
    assert info["flux_in_field_jy"] == pytest.approx(flux, rel=0.01)  # the blob is 3 sigma across the 160 mas field
    assert img.sum() * (5.0 / 1000.0) ** 2 == pytest.approx(info["flux_in_field_jy"], rel=1e-9)  # regrid is flux-preserving within the field
    assert np.unravel_index(np.argmax(img), img.shape) == (15, 15)  # cx.phase_centre_cell(32)


@pytest.mark.validates("virgil_bridge.alma_gp", roots=["self-consistency"], kind="guard")
def test_the_gp_fit_runs_end_to_end_and_reports_the_scored_chi2_the_nuisances_and_the_timings(syn, tmp_path):
    """``alma_imlup_gp.py`` on the toy set (32^2, a few L-BFGS steps per stage, a CLEAN mean and nuisances from a file): both stages run and are recorded (steps, time, s/step,
    s/eval, C8), ``result.json`` leads with the raw chi^2/N, and its chi^2 at the fitted values equals virgil's own direct term's (so the report scores what the fit fitted); the
    outputs exist and the image has the fitted flux."""
    import json

    clean, _, beam = _fake_clean(tmp_path / "clean")
    held = tmp_path / "r" / "values.npz"
    held.parent.mkdir()
    np.savez(held, **{"cal.gain": np.array([1.02]), "cal.offset": np.array([[1.5, -2.0]]), "cal.sigma": np.array([1.0, 1.1, 0.9, 1.05]), "cal.flux": np.asarray(0.18), "log_brightness": np.zeros((4, 4))})
    out = tmp_path / "gp"
    script.main(["--data", str(syn), "--out", str(out), "--mean-from-clean", str(clean), "--nuisances-from", str(held.parent)] + SMALL)
    res = json.loads((out / "result.json").read_text())
    assert next(iter(res)) == "chi2_raw_over_n" and res["complete"] and res["chi2_raw_over_n"] > 0 and res["n_points"] == 4000
    assert [s["name"] for s in res["stages"]] == ["held", "all"] and all(s["steps"] >= 1 and s["wall_s"] > 0 and s["eval_s"] > 0 and s["s_per_step"] > 0 for s in res["stages"])
    assert all(set(("stopped_because", "fit_converged", "c8_converged")) <= set(s) for s in res["timing"]["stages"]) and res["timing"]["total_s"] >= res["timing"]["fit_s"] > 0
    hp = res["hyperparameters"]
    assert hp["length_r_mas"] == 15 and hp["length_theta_rad"] == 1 and hp["n_r"] == 8 and hp["mean_from_clean"] == str(clean) and 0.05 <= hp["sigma_field"] <= 5.0
    nu = res["nuisances"]
    assert nu["gain"][0] == 1.0 and len(nu["gain"]) == 2 and np.shape(nu["offset_mas"]) == (2, 2) and len(nu["sigma"]) == 4
    assert res["flux_beyond_1p8_arcsec_jy"] == 0.0  # the 32 x 5 mas grid is 80 mas across
    assert (res["dots"] is None) == (not hasattr(va, "dots_report"))
    z = np.load(out / "values.npz")
    assert {"field.latent", "field.sigma", "cal.flux", "cal.gain", "cal.offset", "cal.sigma"} <= set(z.files) and z["field.latent"].shape == (8, 12)
    image = np.load(out / "image.npy")
    assert image.shape == (32, 32) and image.sum() * (5.0 / 1000.0) ** 2 == pytest.approx(float(z["cal.flux"]), rel=1e-5)
    assert np.load(out / "image_clean_beam.npy").shape == (32, 32) and (out / "image.fits").exists() and (out / "image_tapered.npy").exists()
    assert {p.name for p in out.glob("resid_dirty_obs*.npy")} == {"resid_dirty_obs0.npy", "resid_dirty_obs1.npy"}
    # the report's chi^2 (fitted s) is virgil's direct term's at the same values
    data = va.load_dir(syn)
    prob = vg.build_gp_problem(data, chi2="direct", npix=32, n_r=8, n_theta=12, r_max_mas=70.0, length_r=15.0, length_theta=1.0, inc=47.5, pa=144.5,
                               mean=vg.restored_clean(clean, 32, 5.0)[0])
    with va.ft_eps(1e-12):
        r, _ = prob.terms[0].residuals_and_log_norm({k: jnp.asarray(z[k]) for k in prob.init})
    assert float(jnp.sum(r**2)) == pytest.approx(res["chi2_fit"], rel=1e-8)


@pytest.mark.validates("virgil_bridge.alma_gp", roots=["self-consistency"], kind="control")
def test_nuisances_of_the_wrong_shape_or_gauge_are_refused(syn, tmp_path):
    """Control: a ``values.npz`` with the wrong number of error scales, and one from a fit with another reference observation, stop the script before any fit."""
    bad = tmp_path / "bad"
    bad.mkdir()
    np.savez(bad / "values.npz", **{"cal.gain": np.array([1.0]), "cal.offset": np.zeros((1, 2)), "cal.sigma": np.ones(26)})
    with pytest.raises(SystemExit, match="cal.sigma has shape"):
        script.main(["--data", str(syn), "--out", str(tmp_path / "o1"), "--nuisances-from", str(bad)] + SMALL)
    (bad / "result.json").write_text('{"ref_obs": 3}')
    with pytest.raises(SystemExit, match="reference observation 3"):
        script.main(["--data", str(syn), "--out", str(tmp_path / "o2"), "--nuisances-from", str(bad)] + SMALL)
