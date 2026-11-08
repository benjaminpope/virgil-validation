"""The registered lambda rule (``lambda-rule`` of ``scripts/alma_imlup_maxent.py``, ``virgil_bridge.alma_lambda``): every stage and every failure path on a toy set
(8^2 pixels, a few thousand visibilities, CPU). The toy is too small for the registered sigma and D4 precision, so those tests lower the threshold they exercise
(``SIGMA_REL_MAX``, ``D4_REL_MAX``) and a separate test shows the registered threshold failing on it. No IM Lup data."""

import json
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("virgil.radio_toeplitz")
pytest.importorskip("virgil.fitting")

import virgil_bridge.alma_lambda as vl  # noqa: E402
from virgil.fitting import kkt_report  # noqa: E402,F401  (the registered convergence test needs it)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import alma_imlup_maxent as script  # noqa: E402

NPIX, PIX = 8, 12.0
pytestmark = pytest.mark.validates("virgil_bridge.alma_lambda", roots=["self-consistency"], kind="guard")


def toy_columns(seed=3, nant=8, ntime=20, nscan=2, gain1=1.0, w0=3000.0, flux=0.15, rotate=7.3e-5):
    """Four ddids (two observations) of baselines x times x scans: a Gaussian source seen through earth rotation, noise sd 1/sqrt(w), obs 1 scaled by ``gain1``."""
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(nant) for j in range(i + 1, nant)]
    q, ph = 10 ** rng.uniform(4.3, 6.0, len(pairs)), rng.uniform(0, np.pi, len(pairs))
    cols = {k: [] for k in "u v vis weight freq ddid obsid time ant1 ant2 scan model".split()}
    for d in range(4):
        for sc in range(nscan):
            for t in range(ntime):
                T = sc * 1000 + t * 6.0
                for k, (a, b) in enumerate(pairs):
                    u, v = q[k] * np.cos(ph[k] + rotate * T), q[k] * np.sin(ph[k] + rotate * T)
                    w = w0 * np.exp(rng.normal(0, 0.2))
                    m = flux * np.exp(-((np.hypot(u, v) / 1.2e6) ** 2)) * (gain1 if d // 2 == 1 else 1.0)
                    z = (rng.normal() + 1j * rng.normal()) / np.sqrt(w)
                    for key, val in zip(cols, (u, v, m + z, w, 2.3e11 + 1e9 * d, d, d // 2, T, a, b, sc, m)):
                        cols[key].append(val)
    return {k: np.array(v) for k, v in cols.items()}


def write_dir(path, cols):
    path.mkdir(parents=True, exist_ok=True)
    for d in np.unique(cols["ddid"]):
        s = cols["ddid"] == d
        np.savez(path / f"ddid_{d:02d}.npz", **{k: (v[s] if k != "ddid" and k != "obsid" else int(v[s][0])) for k, v in cols.items() if k != "model"})
    return path


@pytest.fixture(scope="module")
def cols():
    return toy_columns()


@pytest.fixture(scope="module")
def data(cols, tmp_path_factory):
    import virgil_bridge.alma as va

    return va.load_dir(write_dir(tmp_path_factory.mktemp("toy"), cols))


@pytest.fixture(scope="module")
def sigma(data):
    """A sigma record for the toy (the registered precision is out of reach of a few thousand rows)."""
    mp = pytest.MonkeyPatch()
    mp.setattr(vl, "SIGMA_REL_MAX", 0.05)
    try:
        return vl.sigma_stage(data)
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def swept(data, sigma):
    return vl.sweep_stage(data, sigma, npix=NPIX, pixel_mas=PIX, log=lambda *a: None)


# ---------------------------------------------------------------- the registered text and grid


def test_the_constants_are_the_registered_text():
    block = vl.rule()
    assert block["rule"] == "discrepancy_dof_corrected" and block["crop"] == {"npix": 384, "pixel_mas": 7.5}
    with pytest.raises(SystemExit, match="no longer says"):
        vl.rule(dict(block, text=block["text"].replace("at most 5 steps", "at most 9 steps")))
    with pytest.raises(SystemExit, match="not the registered"):
        vl.rule(dict(block, rule="discrepancy"))


def test_the_grid_runs_from_1e8_to_1e2_in_steps_of_root_ten():
    g = vl.grid(2.0)
    assert len(g) == 13 and g[0] == pytest.approx(2e8) and g[-1] == pytest.approx(2e2)
    assert np.allclose(g[:-1] / g[1:], np.sqrt(10.0))


def test_the_fast_loop_runs_on_the_full_data_with_no_way_to_thin_it(data, sigma):
    """D6: no subsample function, no --stride option, and the sweep takes no stride."""
    import inspect

    assert not hasattr(vl, "subsample") and "stride" not in inspect.signature(vl.sweep_stage).parameters
    with pytest.raises(SystemExit):
        script.main(["lambda-rule", "--stage", "sweep", "--stride", "4"])


# ---------------------------------------------------------------- sigma stage and D4


def test_sigma_stage_records_the_scatter_and_the_reference(data, sigma):
    assert sigma["s2"] == pytest.approx(1.0, abs=0.05)  # noise sd is 1/sqrt(w)
    assert sigma["sigma_s"] > 0 and sigma["raw_chi2_over_N_pairs"] == sigma["s2"] and sigma["status"] == "ok"
    assert sigma["lag1_correlation"] == pytest.approx(-0.5, abs=0.03)  # white noise
    ref, sums = vl.reference_obs(data)
    assert sigma["ref_obs"] == ref == int(np.argmax(sums)) and sigma["fingerprint"] == vl.fingerprint(data)


def test_sigma_fails_when_the_relative_error_exceeds_2e_3(data):
    with pytest.raises(vl.RuleFailure, match="sigma_s / s\\^2") as e:
        vl.sigma_stage(data)  # the registered 2e-3: a few thousand rows give about 1e-2
    assert e.value.record["status"] == "failed" and e.value.record["sigma_rel"] > 2e-3


def test_sigma_needs_the_scan_columns(cols, tmp_path):
    import virgil_bridge.alma as va

    bare = {k: v for k, v in cols.items() if k not in ("scan", "time")}
    with pytest.raises(SystemExit, match="sigma stage"):
        vl.sigma_stage(va.load_dir(write_dir(tmp_path / "bare", bare)))


def test_d4_passes_on_unbiased_scatter_and_fails_on_a_biased_one(data, cols, monkeypatch):
    monkeypatch.setattr(vl, "D4_REL_MAX", 0.05)
    model = cols["model"]
    rec = vl.d4_stage(data, None, PIX, model=model, log=lambda *a: None)
    assert rec["pass"] and rec["n"] == 20 and abs(rec["bias"]) <= 2 * rec["se_mean"] and rec["lag1_correlation_real"] == pytest.approx(-0.5, abs=0.03)
    rapid = model * np.exp(1j * 0.5 * np.asarray(data.time) / 6.0)  # signal that changes inside every pair: the scatter reads high
    with pytest.raises(vl.RuleFailure, match="D4 failed") as e:
        vl.d4_stage(data, None, PIX, model=rapid, log=lambda *a: None)
    assert e.value.record["pass"] is False and e.value.record["bias"] > 0
    vl.require_d4(rec, data)
    for bad in (None, dict(rec, **{"pass": False}), dict(rec, fingerprint="other")):
        with pytest.raises(SystemExit):
            vl.require_d4(bad, data)


# ---------------------------------------------------------------- sweep and final


def test_sweep_finds_the_root_at_the_largest_crossing(swept, sigma):
    rec, start = swept
    s2 = sigma["s2"]
    assert rec["status"] == "ok" and rec["points"][0]["m"] > 0 and rec["points"][-1]["m"] < 0 and len(rec["points"]) == 13
    assert rec["w_over_s2"] == pytest.approx(rec["weight"] / s2) and abs(rec["m"]) <= 3 * rec["g_se"] and rec["n_secant"] <= 5
    assert rec["crossings"] and rec["weight"] == pytest.approx(max(rec["crossings"]), rel=0.5)  # D5: the refit stays at the largest crossing
    assert len(rec["weight_band"]) == 2  # w at s2 + 2 sigma_s and s2 - 2 sigma_s (an end is None when its root is off the grid)
    assert rec["h"] is not None and rec["n_real"] == pytest.approx(2 * 4480, rel=0.03) and set(start) == {"pixels"}


def test_sweep_fails_when_m_is_not_positive_at_the_first_point(data, sigma):
    with pytest.raises(vl.RuleFailure, match="D1: m = .* <= 0 at the first grid point"):
        vl.sweep_stage(data, dict(sigma, s2=100.0 * sigma["s2"]), npix=NPIX, pixel_mas=PIX, log=lambda *a: None)


def test_sweep_fails_without_a_sign_change_and_never_extends_the_grid(data, sigma):
    with pytest.raises(vl.RuleFailure, match="D1: m\\(w\\) has no sign change") as e:  # R1: a noise scale 20% too small has no root
        vl.sweep_stage(data, dict(sigma, s2=0.8 * sigma["s2"]), npix=NPIX, pixel_mas=PIX, log=lambda *a: None)
    assert len(e.value.record["points"]) == 13 and e.value.record["status"] == "failed"


def test_sweep_fails_when_a_fast_loop_fit_has_not_converged(data, sigma, monkeypatch):
    """Item 6 holds on the fast loop: a grid point or secant refit with kkt_w >= 1e-4 ends the run."""
    import functools

    monkeypatch.setattr(vl, "fit_point", functools.partial(vl.fit_point, max_rounds=1))
    monkeypatch.setattr(vl, "kkt_of", lambda report: (1.0, float("nan")))
    with pytest.raises(vl.RuleFailure, match="item 6") as e:
        vl.sweep_stage(data, sigma, npix=NPIX, pixel_mas=PIX, log=lambda *a: None)
    assert e.value.record["unconverged_fits"]


def test_sweep_fails_when_m_stays_above_3_se_after_the_secant_steps(data, sigma, monkeypatch):
    """virgil's registered call (16 probes, CG 1e-4, at most 5 secant steps) is made; a G whose error is far smaller than |m| ends the run (D2)."""
    import dataclasses

    from virgil.imaging import LCurve

    calls, original = [], LCurve.dof_discrepancy

    def tight(self, *a, **kw):
        calls.append(kw)
        out = original(self, *a, **kw)
        return dataclasses.replace(out, g_se=1e-6)

    monkeypatch.setattr(LCurve, "dof_discrepancy", tight)
    with pytest.raises(vl.RuleFailure, match="D2: \\|m\\|") as e:
        vl.sweep_stage(data, sigma, npix=NPIX, pixel_mas=PIX, log=lambda *a: None)
    assert (calls[0]["nprobe"], calls[0]["tol"], calls[0]["max_secant"]) == (16, 1e-4, 5)
    assert e.value.record["status"] == "failed" and e.value.record["secant_trace"]


def finite_band(monkeypatch):
    """The toy's image can be flat at the root, where virgil's band limit is NaN (platform-dependent); the happy paths take a finite one, the NaN failure has its own test."""
    real = vl.kkt_of
    monkeypatch.setattr(vl, "kkt_of", lambda report: (lambda k, b: (k, 1.0 if np.isnan(b) else b))(*real(report)))


@pytest.fixture(scope="module")
def finished(data, sigma, swept):
    rec, start = swept
    with pytest.MonkeyPatch.context() as mp:
        finite_band(mp)
        return vl.final_stage(data, rec, sigma, start, npix=NPIX, pixel_mas=PIX, log=lambda *a: None)


@pytest.mark.slow
def test_final_fails_on_a_nan_band_limit(data, sigma, swept, monkeypatch):
    real = vl.kkt_of
    monkeypatch.setattr(vl, "kkt_of", lambda report: (real(report)[0], float("nan")))
    rec, start = swept
    with pytest.raises(vl.RuleFailure, match="f_eta/f_floor"):
        vl.final_stage(data, rec, sigma, start, npix=NPIX, pixel_mas=PIX, log=lambda *a: None)


def test_final_reports_item_8_and_checks_g_against_the_fast_loop(finished, swept, sigma):
    rec, values = finished
    sweep = swept[0]
    assert list(rec)[0] == "stage" and rec["raw_chi2_over_N"] > 0 and rec["s2"] == sigma["s2"] and rec["weight"] == sweep["weight"] and rec["w_over_s2"] == pytest.approx(sweep["w_over_s2"])
    assert rec["kkt_w"] < 1e-4 and rec["band_limit"] < 2 and rec["ref_obs"] == sigma["ref_obs"]
    assert set(rec["psi_hp"]) == {"disc", "lo", "hi"} and set(rec["nuisances"]) == {"cal.gain", "cal.offset"}
    assert all(v["edge_frac"].min() > 0.05 and not v["at_edge"].any() for v in rec["nuisances"].values())
    assert set(values) >= {"pixels", "cal.gain", "cal.offset"}
    assert "m_full" not in rec and "g_check_z" not in rec and rec["g"] > 0 and rec["sigma_factors"].startswith("not fitted")


@pytest.mark.slow
def test_final_fails_when_a_nuisance_ends_at_its_box_edge(sigma, swept):
    import virgil_bridge.alma as va
    import tempfile

    bad = toy_columns(gain1=1.6)  # obs 1's gain is outside the 0.7-1.3 box
    with tempfile.TemporaryDirectory() as d:
        data = va.load_dir(write_dir(pathlib.Path(d), bad))
    sweep = swept[0]
    prob = vl.build(data, npix=NPIX, pixel_mas=PIX, flux=sigma["flux_jy"], ref_obs=vl.reference_obs(data)[0])
    with pytest.raises(vl.RuleFailure, match="box edge") as e:
        vl.final_stage(data, sweep, dict(sigma, ref_obs=vl.reference_obs(data)[0]), prob.start(), npix=NPIX, pixel_mas=PIX, log=lambda *a: None)
    assert e.value.record["nuisances"]["cal.gain"]["at_edge"].any()


@pytest.mark.slow
@pytest.mark.parametrize("name, value, match", [("KKT_W_MAX", 1e-30, "kkt_w"), ("BAND_MAX", 0.0, "f_eta/f_floor")])
def test_final_fails_on_the_convergence_tests(data, sigma, swept, monkeypatch, name, value, match):
    monkeypatch.setattr(vl, name, value)
    rec, start = swept
    with pytest.raises(vl.RuleFailure, match=match):
        vl.final_stage(data, rec, sigma, start, npix=NPIX, pixel_mas=PIX, log=lambda *a: None)


# ---------------------------------------------------------------- the command


def call(*argv):
    script.main(["lambda-rule", *[str(a) for a in argv]])


def test_the_command_runs_the_stages_in_order_and_refuses_the_sweep_without_d4(cols, tmp_path, monkeypatch):
    finite_band(monkeypatch)
    from virgil_bridge import alma_synth as syn

    d, out = write_dir(tmp_path / "data", cols), tmp_path / "run"
    geom = ["--data", d, "--out", out, "--npix", NPIX, "--pixel-mas", PIX]
    with pytest.raises(SystemExit, match="sigma FAILED"):
        call("--stage", "sigma", *geom)  # registered precision: fails, writes the reason
    failed = json.loads((out / "sigma.json").read_text())
    assert failed["status"] == "failed" and "sigma_s / s^2" in failed["reason"] and failed["criteria_commit"]
    with pytest.raises(SystemExit, match="failed stage"):
        call("--stage", "sweep", *geom)
    monkeypatch.setattr(vl, "SIGMA_REL_MAX", 0.05)
    call("--stage", "sigma", *geom)
    with pytest.raises(SystemExit, match="passing D4 verdict"):
        call("--stage", "sweep", *geom)
    monkeypatch.setattr(vl, "D4_REL_MAX", 0.05)
    toy_truth = np.zeros((16, 16))
    toy_truth[7:9, 7:9] = 1.0
    monkeypatch.setattr(syn, "registered_truth", lambda *a, **k: (toy_truth, "0" * 64, {}))
    call("--stage", "d4", *geom, "--clean-dir", tmp_path)
    verdict = json.loads((out / "d4.json").read_text())
    assert verdict["pass"] is True and verdict["clean_model_sha256"] == "0" * 64
    call("--stage", "sweep", *geom)
    sweep = json.loads((out / "sweep.json").read_text())
    assert sweep["status"] == "ok" and sweep["weight"] > 0 and (out / "sweep_values.npz").exists()
    call("--stage", "final", *geom)
    final = json.loads((out / "lambda_final.json").read_text())
    assert final["status"] == "ok" and final["weight"] == sweep["weight"] and final["criteria_commit"] and np.load(out / "image.npy").shape == (NPIX, NPIX)
