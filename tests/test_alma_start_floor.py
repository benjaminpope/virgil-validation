"""The cold-start floor and the dead-pixel diagnostic of the IM Lup MaxEnt fit (R ob100, job 18968246_0, ended with 64 % of its pixels dead).
Synthetic data on the CPU, seconds."""

import json
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("virgil.radio")

import virgil_bridge.alma as va  # noqa: E402
import virgil_bridge.alma_start as vs  # noqa: E402
from crosscheck import alma_maxent as cx  # noqa: E402
from evidence.plugin import record  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import alma_imlup_maxent as script  # noqa: E402
from test_alma_maxent import synthetic_dir  # noqa: E402


def _call(*argv):
    script.main([str(a) for a in argv])


class _Prob:
    init = {}

    def __init__(self, npix, pixel_mas):
        self.npix, self.pixel_mas = npix, pixel_mas


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
@pytest.mark.parametrize("f_config", [False, True])
def test_floor_start_raises_every_pixel_and_reads_back_through_image_of(f_config):
    """The 1" Gaussian start has pixels at 1e-8 of the peak and below; floored, every pixel is at least the floor and ``image_of`` returns
    exactly the floored image (in config F as well, where ``cal.flux`` is the apparent flux)."""
    n, px, floor = 512, 10.0, 1e-4  # a 5.1 arcsec field, as IM Lup's 5.1 arcsec at 1028 x 5 mas
    v = va.start_values(_Prob(n, px), flux=0.25, fwhm_mas=1000.0)
    log_pb = va.beam_log(n, px) if f_config else None
    before = va.image_of(v, n, px, log_pb)
    assert before.min() < 1e-6 * before.max()
    eta, flux, info = va.floor_start(v, n, px, floor, log_pb=log_pb)
    after = va.image_of({"log_brightness": eta, "cal.flux": flux}, n, px, log_pb)
    assert after.min() >= floor * (1 - 1e-12)
    assert np.allclose(after, np.maximum(before, floor), rtol=1e-10, atol=0)
    assert info["n_raised"] == int((before < floor).sum()) and info["flux_added_jy"] > 0
    record("flux_added_jy" + ("_F" if f_config else "_R"), info["flux_added_jy"])


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_dead_pixel_report_flags_the_bare_gaussian_and_passes_the_floored_start():
    n, px = 512, 10.0
    v = va.start_values(_Prob(n, px), flux=0.25, fwhm_mas=1000.0)
    bare = va.dead_pixel_report(va.image_of(v, n, px), px)
    eta, flux, _ = va.floor_start(v, n, px, 1e-4)
    live = va.dead_pixel_report(va.image_of({"log_brightness": eta, "cal.flux": flux}, n, px), px)
    assert bare["warning"] and bare["frac_below_1e-6_peak"] > 0.3
    assert live["warning"] is None and live["frac_below_1e-6_peak"] == 0.0 and live["frac_below_1e-3_floor"] == 0.0
    assert live["largest_live_radius_arcsec"] > bare["largest_live_radius_arcsec"]
    record("dead_fraction_bare_gaussian", bare["frac_below_1e-6_peak"])


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
@pytest.mark.parametrize("floor", [None, 0.0])
def test_fit_p3_records_the_floor_start_by_default_and_the_dead_pixels(tmp_path, floor):
    """``fit`` raises the p3 start to the entropy floor by default (recorded in result.json and the first stage) and reports the dead pixels;
    ``--start-floor 0`` is the bare Gaussian and records no floor."""
    syn = synthetic_dir(tmp_path / "s")
    extra = [] if floor is None else ["--start-floor", floor]
    out = tmp_path / "o"
    _call("fit", "--config", "R", "--data", syn, "--out", out, "--npix", 128, "--max-steps", 3, "--no-dirty", *extra)
    r = json.loads((out / "result.json").read_text())
    if floor is None:
        assert r["args"]["start_floor"] == va.MPOL["prior"]
        assert r["start_floor"]["floor_jy_arcsec2"] == va.MPOL["prior"] and r["stages"][0]["start_floor"] == r["start_floor"]
        assert r["dead_pixels"]["warning"] is None and r["dead_pixels"]["frac_below_1e-6_peak"] == 0.0
    else:
        assert r["start_floor"] is None and "start_floor" not in r["stages"][0]  # (a 0.64 arcsec test field is inside the Gaussian: nothing dead; the report is tested above)
    assert set(r["dead_pixels"]) >= {"frac_below_1e-6_peak", "frac_below_1e-3_floor", "largest_live_radius_arcsec", "median_beyond_2p4_arcsec_over_peak"}


# ------------------------------------------------------------------ the cold-start gate


GOOD = dict(objective_ours=-10.0, objective_mpol=-9.0, chi2n_ours=2.70, chi2n_mpol=2.71, stages_converged=[True, True], dead_fraction=0.0, flux_ours=0.25, flux_ref=0.255, flux_ref_name="DSHARP")


@pytest.mark.validates("virgil_bridge.alma_start", roots=["self-consistency"], kind="guard")
def test_the_gate_passes_only_when_all_five_checks_pass_and_names_the_failing_one():
    """a objective, b chi^2/N, c C8 in every stage, d dead pixels under 1%, e flux within 10% of the reference: PASS needs all five; each single failure is FAIL
    on that item alone; a missing input (compare without --data) is INCOMPLETE, never PASS; the boundaries are at or below / under as defined."""
    assert vs.cold_start_gate(**GOOD)["verdict"] == "PASS"
    bad = dict(a=dict(objective_ours=-8.0), b=dict(chi2n_ours=2.72), c=dict(stages_converged=[True, False]), d=dict(dead_fraction=0.64), e=dict(flux_ours=0.205))
    for key, kw in bad.items():
        g = vs.cold_start_gate(**{**GOOD, **kw})
        assert g["verdict"] == "FAIL" and [i["key"] for i in g["items"] if i["status"] == "FAIL"] == [key], key
    assert vs.cold_start_gate(**{**GOOD, "objective_ours": -9.0, "chi2n_ours": 2.71})["verdict"] == "PASS"  # equal is at or below
    assert vs.cold_start_gate(**{**GOOD, "flux_ours": 0.255 * 1.099})["verdict"] == "PASS" and vs.cold_start_gate(**{**GOOD, "flux_ours": 0.255 * 1.101})["verdict"] == "FAIL"
    assert vs.cold_start_gate(**{**GOOD, "dead_fraction": 0.0099})["verdict"] == "PASS" and vs.cold_start_gate(**{**GOOD, "dead_fraction": 0.01})["verdict"] == "FAIL"
    inc = vs.cold_start_gate(**{**GOOD, "objective_ours": None, "objective_mpol": None, "chi2n_ours": None, "chi2n_mpol": None})
    assert inc["verdict"] == "INCOMPLETE" and [i["status"] for i in inc["items"]][:2] == ["NOT EVALUATED"] * 2
    assert vs.cold_start_gate(**{**GOOD, "objective_ours": None, "objective_mpol": None, "dead_fraction": 0.5})["verdict"] == "FAIL"  # a failure outranks a missing check
    assert "FAIL" in "\n".join(vs.gate_markdown(vs.cold_start_gate(**{**GOOD, "flux_ours": 0.1})))


# ------------------------------------------------------------------ the data-driven start


def _gauss_data(tmp_path, truth, n=3000, q_hi=3e4, seed=1, noise=1e-3):
    """Per-ddid files (the vv format) holding the analytic visibilities of ``truth`` (flux, east, north, major, minor, pa) at baselines up to ``q_hi``."""
    rng = np.random.default_rng(seed)
    q, phi = 10 ** rng.uniform(3.5, np.log10(q_hi), n), rng.uniform(0, np.pi, n)
    u, v = q * np.cos(phi), q * np.sin(phi)
    w = np.full(n, 1 / noise**2)
    vis = vs.gaussian_visibility(u, v, *truth) + noise * (rng.normal(size=n) + 1j * rng.normal(size=n))  # sigma 1/sqrt(w) per component, as the repo's weights
    tmp_path.mkdir(parents=True, exist_ok=True)
    np.savez(tmp_path / "ddid_00.npz", u=u, v=v, vis=vis, weight=w, freq=np.full(n, 2.3e11), ddid=0, obsid=0)
    return tmp_path


@pytest.mark.validates("virgil_bridge.alma_start", roots=["mathematics"])
def test_fit_gaussian_recovers_an_elliptical_gaussian_and_orders_the_axes(tmp_path):
    """Noisy visibilities of a known elliptical Gaussian (flux 0.3 Jy, 200 mas East and -150 mas North, 2200 x 1300 mas FWHM at PA 35 deg): the fit gets the flux,
    centroid, FWHMs and PA back (also when started with the axes swapped: major >= minor and PA in [0, 180)), and chi^2 per point is about 1."""
    truth = (0.30, 200.0, -150.0, 2200.0, 1300.0, 35.0)
    d = va.load_dir(_gauss_data(tmp_path, truth))
    f = vs.fit_gaussian(d.u, d.v, d.vis, d.weight, q_max=3e4)
    got = (f["flux_jy"], f["east_mas"], f["north_mas"], f["major_mas"], f["minor_mas"], f["pa_deg"])
    np.testing.assert_allclose(got, truth, rtol=0.02, atol=20.0)
    assert abs(f["pa_deg"] - 35.0) < 1.5 and abs(f["east_mas"] - 200) < 15 and abs(f["north_mas"] + 150) < 15 and 0.8 < f["chi2_per_point"] < 1.2
    swapped = (0.30, 200.0, -150.0, 1300.0, 2200.0, 125.0)  # the same ellipse with the axes named the other way round
    d2 = va.load_dir(_gauss_data(tmp_path / "s", swapped))
    f2 = vs.fit_gaussian(d2.u, d2.v, d2.vis, d2.weight, q_max=3e4)
    assert f2["major_mas"] > f2["minor_mas"] and abs(f2["pa_deg"] - 35.0) < 1.5
    record("pa_error_deg", abs(f["pa_deg"] - 35.0))
    with pytest.raises(ValueError):
        vs.fit_gaussian(d.u, d.v, d.vis, d.weight, q_max=1.0)


@pytest.mark.validates("virgil_bridge.alma_start", roots=["self-consistency"], kind="guard")
def test_the_fitted_gaussian_start_is_the_image_virgils_model_sees_not_its_mirror(tmp_path):
    """End to end through virgil's own forward model: the start image rendered from the fitted parameters (``gaussian_log_image`` on the fit's grid, East towards
    column 0) gives a chi^2 on the data far below the same image mirrored East-West or North-South; the conventions of the image, the sign of u and the PA agree."""
    truth = (0.30, 80.0, -50.0, 220.0, 130.0, 35.0)
    syn = _gauss_data(tmp_path, truth, n=1500, q_hi=1.5e5, noise=1e-3)
    data = va.load_dir(syn)
    n, px = 128, 5.0
    prob = va.build_problem(data, "R", lam=1.0, npix=n, pixel_mas=px)
    fit = vs.fit_gaussian(data.u, data.v, data.vis, data.weight, q_max=1.5e5)
    values = dict(prob.init)
    values["cal.flux"] = np.asarray(fit["flux_jy"])

    def chi2(eta):
        return va.score(data, prob.scene, {**values, "log_brightness": np.asarray(eta)}, prob.nuis)["chi2_raw"] / (2 * len(data.u))

    args = (fit["east_mas"], fit["north_mas"], fit["major_mas"], fit["minor_mas"], fit["pa_deg"])
    good = chi2(vs.gaussian_log_image(n, px, *args))
    flip_ew = chi2(vs.gaussian_log_image(n, px, -args[0], args[1], args[2], args[3], -args[4]))
    flip_ns = chi2(vs.gaussian_log_image(n, px, args[0], -args[1], args[2], args[3], -args[4]))
    assert good < 0.05 * min(flip_ew, flip_ns), (good, flip_ew, flip_ns)
    record("chi2n_fitted_start", good)
    record("chi2n_mirrored_start", min(flip_ew, flip_ns))


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_fit_start_gauss_fit_records_the_fitted_gaussian_and_is_refused_where_there_is_another_start(tmp_path):
    """``fit --start gauss-fit`` on the p3 workflow: the fitted Gaussian (parameters, n used, source) is in result.json and the first stage, the start is then floored,
    and the default (``--start gauss``) records none; --workflow mpol, clean and --init refuse it."""
    syn = synthetic_dir(tmp_path / "s")
    out = tmp_path / "o"
    _call("fit", "--config", "R", "--data", syn, "--out", out, "--npix", 64, "--max-steps", 2, "--no-dirty", "--start", "gauss-fit", "--gauss-fit-max-klambda", 100)
    r = json.loads((out / "result.json").read_text())
    g = r["start_gauss_fit"]
    assert g["n_used"] > 10 and g["major_mas"] >= g["minor_mas"] and "uvmodelfit" in g["source"] and r["stages"][0]["start_gauss_fit"] == g
    assert r["start_floor"]["floor_jy_arcsec2"] == va.MPOL["prior"] and r["args"]["start"] == "gauss-fit"
    out2 = tmp_path / "o2"
    _call("fit", "--config", "R", "--data", syn, "--out", out2, "--npix", 64, "--max-steps", 2, "--no-dirty")
    assert json.loads((out2 / "result.json").read_text())["start_gauss_fit"] is None
    for extra in (["--workflow", "mpol"], ["--init", out2]):
        with pytest.raises(SystemExit):
            _call("fit", "--config", "R", "--data", syn, "--out", tmp_path / "x", "--npix", 64, "--max-steps", 2, "--start", "gauss-fit", *extra)
