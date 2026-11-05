"""The registered lambda rule (``lambda-rule`` of ``scripts/alma_imlup_maxent.py``, ``virgil_bridge.alma_lambda``) end to end on a toy set (a 16^2 grid, a few
hundred visibilities, CPU, seconds): sigma -> sweep (sharded) -> final, with the discrepancy weight from virgil's ``LCurve``. The registered grid is
replaced by a short one; everything else is the shipped code path."""

import json
import pathlib
import sys

import numpy as np
import pytest
import yaml

pytest.importorskip("virgil.radio_toeplitz")

import virgil_bridge.alma_lambda as vl  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import alma_imlup_maxent as script  # noqa: E402
from test_alma_maxent import synthetic_dir  # noqa: E402

NPIX, PIX = 8, 12.0  # the toy's baselines (4e6 wavelengths) need pixels below ~14 mas for the Toeplitz precompute
PER = 1  # points per decade of the toy grid
pytestmark = pytest.mark.validates("virgil_bridge.alma_lambda", roots=["self-consistency"], kind="guard")


DRAFT = yaml.safe_load("""\nlambda_selection:
  rule: discrepancy
  statement: 'Choose the regulariser weight w at which the raw chi^2/N of the image fit, with the per-ddid
    sigma factors FIXED (below), equals 1 (Morozov). virgil.imaging.l_curve over the grid below, then
    LCurve.discrepancy(target=N), log-interpolated between the bracketing grid points; the final image is
    refitted at that w.'
  sigma_factors: 'per-ddid factors s_k fixed once, before any regularised fit, from an unregularised fit
    (regularizers=[], same crop, same nuisances, L-BFGS to C8): s_k^2 = chi^2_k / N_k of that fit. Never
    refitted jointly with w (joint fitting makes chi^2/N = 1 for every w and the rule empty).
    Overfitting bias of s_k^2 is n_pix / N, below 0.5 % here; recorded, not corrected.'
  grid: 'log-uniform, 8 points per decade; MEM 1e3-1e7, StarletL1 1e3-1e7 (in the #614 normalised units at
    this crop). If the crossing falls at a grid edge, extend the grid by one decade and record the extension;
    never stop at an edge.'
  arms: [mem, starlet_l1]   # MEM (flat prior) and StarletL1: best two in the synthetic fast loop (2026-11-02)
  crop: {npix: 384, pixel_mas: 7.5}
  optimiser: 'L-BFGS, warm-started along the grid from strong to weak w, each point run to C8 (no step cap;
    the 900 s cap of the fast loop is not used)'
  secondary_reported_not_selecting: [corner, classic_maxent_if_feasible]
  validation: 'the same rule on the synthetic ring_gap truth (fast loop, 2026-11-02) gave NRMSE(0.5 beam)
    0.059 (MEM) and 0.061 (StarletL1) against 0.076 unregularised; that run used known sigma and is the
    development check, not a held-out one'
  changed_after_seeing_a_result: 'no real-data result exists for this rule; the arm choice uses only the
    synthetic fast loop'
""")["lambda_selection"]  # the withdrawn first draft of the rule, which this driver implements (the registered block is the revised one)


def call(*argv):
    script.main(["lambda-rule", *[str(a) for a in argv]])


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    return synthetic_dir(tmp_path_factory.mktemp("syn"), n=30, n_obs=3, per_obs=2)


def test_the_driver_refuses_the_revised_registered_rule():
    with pytest.raises(SystemExit, match="withdrawn draft"):
        vl.rule()


def test_the_draft_grid_is_parsed():
    r = vl.rule(DRAFT)
    assert r["per_decade"] == 8 and r["arms"] == ["mem", "starlet_l1"] and r["crop"] == {"npix": 384, "pixel_mas": 7.5}
    assert r["k_range"] == {"mem": (24, 56), "starlet_l1": (24, 56)}  # 1e3 to 1e7 at 8 per decade
    assert vl.weight(24, 8) == pytest.approx(1e3) and vl.weight(56, 8) == pytest.approx(1e7)


def test_discrepancy_interpolates_in_log_weight_and_reports_a_missing_crossing():
    rec = lambda w, c: dict(weight=w, chi2=c * 100, chi2_over_N=c, penalty=1.0)  # noqa: E731
    pts = [rec(1e4, 3.0), rec(1e3, 0.5)]
    assert vl.interpolate(pts) == pytest.approx(10 ** (4 - 2.0 / 2.5))  # linear in log w between the bracketing points
    assert vl.interpolate([rec(1e4, 3.0), rec(1e3, 2.0)]) is None and vl.bracketed([rec(1e4, 3.0), rec(1e3, 2.0)]) == (False, True)


def test_sigma_sweep_final_end_to_end(syn, tmp_path, monkeypatch):
    out = tmp_path / "run"
    base = vl.rule(DRAFT)
    monkeypatch.setattr(vl, "rule", lambda: dict(base, per_decade=PER, k_range={a: (0, 2) for a in base["arms"]}))  # the draft with a toy grid: w = 1, 10, 100
    geom = ["--data", syn, "--out", out, "--npix", NPIX, "--pixel-mas", PIX]
    with pytest.raises(SystemExit, match="did not reach C8"):  # a cut sigma stage is refused (its s_k would be too large), but its record is kept
        call("--stage", "sigma", *geom, "--time-limit", 2)
    with pytest.raises(SystemExit, match="did not reach C8"):
        call("--stage", "sweep", *geom, "--arm", "mem", "--k", 1)
    sig = json.loads((out / "sigma.json").read_text())
    assert sig["c8_converged"] is False and len(sig["sigma"]) == 6 and all(s > 0 for s in sig["sigma"])
    assert sig["chi2_over_N"] == pytest.approx(sum(np.array(sig["sigma"]) ** 2 * np.array(sig["n_k"])) / sum(sig["n_k"]))  # s_k^2 = chi^2_k / N_k
    assert sig["overfit_bias"] == pytest.approx(NPIX**2 / sig["n_real"])
    (out / "sigma.json").write_text(json.dumps(dict(sig, sigma=[1.11] * 6, c8_converged=True)))  # the toy model leaves chi^2/N = 1.2 even unregularised (s = 1.11 puts that at 1); the cut fit above would give s too large for a crossing

    call("--stage", "sweep", *geom, "--arm", "starlet_l1", "--k", 1)  # the second arm: one shard
    assert json.loads((out / "sweep" / "starlet_l1" / "k+001" / "point.json").read_text())["arm"] == "starlet_l1"
    call("--stage", "sweep", *geom, "--arm", "mem", "--k", 2, 1)  # one call, two points: the second warm-starts from the first
    assert sorted(p.parent.name for p in (out / "sweep" / "mem").glob("k*/point.json")) == ["k+001", "k+002"]
    p2 = json.loads((out / "sweep" / "mem" / "k+002" / "point.json").read_text())
    assert p2["start_from"] == "flat" and p2["sigma_hash"] == vl.sigma_hash([1.11] * 6)
    assert json.loads((out / "sweep" / "mem" / "k+001" / "point.json").read_text())["start_from"] == 2  # warm start from the stronger neighbour
    assert p2["weight"] == pytest.approx(100.0) and p2["c8"]["window_steps"] is not None
    assert {"n_eff_ratio", "base_dead_in_disk_1e4", "sub_taper_power"} <= set(p2["dots"])  # the dots metrics of every point
    assert p2["chi2_over_N"] == pytest.approx(p2["virgil_chi2"] / p2["n_real"], rel=1e-6)  # the term's chi^2 is virgil's own
    (out / "sigma.json").write_text(json.dumps(dict(sig, sigma=[1.12] * 6, c8_converged=True)))
    with pytest.raises(SystemExit, match="other s_k"):  # a cached point from other s_k is refused
        call("--stage", "sweep", *geom, "--arm", "mem", "--k", 2)
    (out / "sigma.json").write_text(json.dumps(dict(sig, sigma=[1.11] * 6, c8_converged=True)))
    call("--stage", "final", *geom, "--arm", "mem", "--beam-mas", 120, 90, 30, "--max-extend", 2)  # fits k = 0 itself, then the discrepancy weight
    res = json.loads((out / "final" / "mem" / "lambda_final.json").read_text())
    assert res["registered"] is True and res["criteria_hash"] and res["registered_grid"]["k_range"] == [0, 2]
    assert res["max_extend"] == 2 and "not in the registered block" in res["max_extend_note"] and res["unconverged_points"] == []
    assert res["status"] == "ok"
    chi = [p["chi2_over_N"] for p in res["points"]]
    print("bracketing chi2/N:", chi, "weights:", [p["weight"] for p in res["points"]], "selected", res["selected_weight"])
    ws = [p["weight"] for p in res["points"]]
    below = [p["weight"] for p in res["points"] if p["chi2_over_N"] <= 1]
    above = [p["weight"] for p in res["points"] if p["chi2_over_N"] > 1]
    assert max(below) <= res["selected_weight"] <= min(above) and min(ws) <= res["selected_weight"] <= max(ws)
    f = res["final"]
    assert f["weight"] == pytest.approx(res["selected_weight"]) and f["chi2_over_N_fixed_s"] == pytest.approx(f["chi2_over_N"], rel=1e-3)  # direct and Toeplitz agree
    assert abs(f["chi2_over_N"] - 1) < 0.25 and f["c8_converged"]  # the refit at the interpolated weight lands near the discrepancy (coarse toy grid: 1 point per decade)
    assert f["chi2_over_N_s1"] > 0 and f["flux_jy"] > 0 and f["blur_fraction"] == 0.5
    d = out / "final" / "mem"
    raw, blur = np.load(d / "image_raw.npy"), np.load(d / "image_blur.npy")
    assert raw.shape == blur.shape == (NPIX, NPIX) and blur.max() < raw.max()
