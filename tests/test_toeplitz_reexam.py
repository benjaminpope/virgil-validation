"""``scripts/toeplitz_reexam.py`` (virgil 0.4.2 gate G2; ``design/toeplitz_reexam_042.md``), CPU only, seconds.

The registered criteria hash and its enforcement; the independent reference ``crosscheck.radio_exact`` against the brute-force
``crosscheck.radio_offsets`` and against central differences; the Tier I comparison on two tiny unregistered scenes (the registered
ones run only in the job), with a control that a looser kernel fails the documented budget; Tier II on a tiny IM Lup-like set; the
verdict logic. None of these runs a registered scene.
"""

import hashlib
import json
import math
import pathlib
import sys

import numpy as np
import pytest

from crosscheck import radio_exact as rx
from crosscheck import radio_offsets as ro
from crosscheck import toeplitz_scenes as ts

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import toeplitz_reexam as tr  # noqa: E402

TCFG = dict(modes=["fixed", "fitted", "profiled"], points=["optimum", "moved", "edge"], eps_kernel=1e-14, eps_psi_hat=4e-14)
SMOKE = [dict(name="smoke_spread", image="ringed_disc", npix=12, pixel_mas=4.0, n_ant=8, n_times=8, n_obs=3, n_ddid=4, reach=0.4,
              offsets="spread", offset_bound_mas=10.0),
         dict(name="smoke_none", image="sources_on_halo", npix=11, pixel_mas=3.0, n_ant=8, n_times=8, n_obs=2, n_ddid=3, reach=1.3,
              offsets="none")]


def _has_toeplitz():
    try:
        import virgil.radio_toeplitz  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


needs_toeplitz = pytest.mark.skipif(not _has_toeplitz(), reason="virgil.radio_toeplitz is not installed")


# ------------------------------------------------------------------ the registered criteria


@pytest.mark.validates("virgil.radio_toeplitz.ToeplitzVisData", roots=["self-consistency"], kind="guard")
def test_the_registered_hash_is_the_criteria_file_and_the_registered_scenes_are_held_out():
    """REGISTERED_SHA256 is the sha256 of the criteria file; at least three distinct held-out images; no registered scene uses a
    development image (random log-normal pixels on a uniform uv disc) or IM Lup."""
    assert tr.REGISTERED_SHA256 == hashlib.sha256(tr.CRITERIA_FILE.read_bytes()).hexdigest()
    cfg = tr.criteria()
    images = {s["image"] for s in cfg["tier1"]["scenes"]}
    assert len(images) >= 3 and images <= {"ringed_disc", "sources_on_halo", "edge_filled"}
    assert {s["offsets"] for s in cfg["tier1"]["scenes"]} == {"spread", "none"}
    assert cfg["verdict"]["combine_runs"] is False


@pytest.mark.validates("virgil.radio_toeplitz.ToeplitzVisData", roots=["self-consistency"], kind="control")
def test_a_changed_criteria_file_is_refused(tmp_path, monkeypatch):
    changed = tmp_path / "c.yml"
    changed.write_text(tr.CRITERIA_FILE.read_text().replace("speedup_min: 30.0", "speedup_min: 15.0"))
    monkeypatch.setattr(tr, "CRITERIA_FILE", changed)
    with pytest.raises(SystemExit, match="not the registered"):
        tr.criteria()
    with pytest.raises(SystemExit):
        tr.main(["summary", "--out", str(tmp_path)])


@pytest.mark.validates("crosscheck.toeplitz_scenes", roots=["self-consistency"], kind="guard")
def test_seeds_follow_the_registered_rule():
    assert ts.seed("S1_ringed_disc", "truth") == int(hashlib.sha256(b"toeplitz-reexam-042/S1_ringed_disc/truth").hexdigest()[:8], 16)
    assert ts.seed("a", "truth") != ts.seed("a", "moved")


# ------------------------------------------------------------------ the independent reference


def _small(spec=SMOKE[0]):
    case, truth = ts.make_case(spec)
    return case, truth, ts.points(case, truth, spec)


@pytest.mark.validates("crosscheck.radio_exact", roots=["mathematics"], kind="reference")
def test_the_factorised_reference_equals_the_brute_force_pixel_sum():
    """chi2 and the gradient (image logits, flux, gains, offsets) of ``radio_exact`` (matrix products over the factorised phase)
    equal ``radio_offsets`` (the full ``(points, pixels)`` exponential) at the moved point, to float64 rounding of these sums
    (about 1e-13 of each site's norm)."""
    case, _, pts = _small()
    v = pts["moved"]
    out = rx.evaluate(case, v, "fixed")
    chi2, g = ro.chi2_and_gradient(case.u, case.v, case.vis, case.weight, case.obs, v["eta"], case.pixel_mas, v["flux"], v["gain"],
                                   v["offset"])
    assert 2 * out["loss"] == pytest.approx(chi2, rel=1e-13)
    for site in ("eta", "flux", "gain", "offset"):
        assert np.linalg.norm(2 * np.asarray(out["grad"][site]) - g[site]) <= 1e-12 * np.linalg.norm(g[site]), site


@pytest.mark.validates("crosscheck.radio_exact", roots=["mathematics"], kind="reference")
@pytest.mark.parametrize("mode", ["fitted", "profiled"])
def test_the_reference_gradient_equals_central_differences(mode):
    """The error-scale modes: d L / d s_d (fitted) and d L / d flux (both) against central differences of the reference loss,
    h = 1e-6 relative; truncation ~h^2 and rounding ~u L / h are both below 1e-7 of the gradient here. The profiled loss equals
    the fitted loss at s = sigma_hat (its definition)."""
    case, _, pts = _small()
    v = dict(pts["moved"])
    part = rx.partials(case, v)
    s = 1.1 * rx.sigma_hat(part["chi2_d"], case.n_d)
    out = rx.combine(case, part, mode, s if mode == "fitted" else None)

    def loss(sigma=s, **kw):
        vals = dict(v, **kw)
        return rx.evaluate(case, dict(vals, sigma=sigma) if mode == "fitted" else vals, mode)["loss"]

    h = 1e-6 * v["flux"]
    fd = (loss(flux=v["flux"] + h) - loss(flux=v["flux"] - h)) / (2 * h)
    assert fd == pytest.approx(out["grad"]["flux"], rel=1e-6)
    if mode == "fitted":
        for d in range(case.n_ddid):
            hs = 1e-6 * s[d]
            up, dn = s.copy(), s.copy()
            up[d] += hs
            dn[d] -= hs
            fd = (loss(sigma=up) - loss(sigma=dn)) / (2 * hs)
            assert fd == pytest.approx(out["grad"]["sigma"][d], rel=1e-6, abs=1e-6 * np.abs(out["grad"]["sigma"]).max())
    else:
        sh = rx.sigma_hat(part["chi2_d"], case.n_d)
        assert out["loss"] == pytest.approx(rx.combine(case, part, "fitted", sh)["loss"], rel=1e-14)


@pytest.mark.validates("crosscheck.radio_exact", roots=["mathematics"], kind="reference")
def test_the_budget_constants():
    """gamma_n, the FFT factor and the probabilistic bound from their formulas (Higham 2002, Lemma 3.1, Thm 24.2; Higham & Mary 2019)."""
    assert rx.U == 2.0**-53
    assert rx.gamma(10**6) == pytest.approx(1e6 * 2.0**-53, rel=1e-9)
    assert rx.fft_factor(2**20) == pytest.approx(20 * rx.ETA_FFT, rel=1e-9)
    assert rx.gamma_probabilistic(10**6) == pytest.approx(math.sqrt(2 * math.log(2e12)) * 1e3 * 2.0**-53)
    assert rx.gamma_probabilistic(10**6) < rx.gamma(10**6)


# ------------------------------------------------------------------ Tier I on tiny unregistered scenes


@needs_toeplitz
@pytest.mark.validates("virgil.radio_toeplitz.ToeplitzVisData", roots=["mathematics"])
@pytest.mark.parametrize("spec", SMOKE, ids=[s["name"] for s in SMOKE])
def test_tier1_rows_pass_on_tiny_unregistered_scenes(spec):
    """Every mode and point: loss, chi2 per ddid (fixed mode), each gradient site within the derived budget; the optimum's fitted
    error-scale gradient is a cancellation (relative error far above 1e-8) and still passes its absolute budget."""
    rows, facts = tr.tier1_scene(spec, TCFG)
    bad = [r for r in rows if r["status"] != "PASS"]
    assert not bad, bad
    n_points = 3 if spec["offsets"] == "spread" else 2
    assert len({r["where"] for r in rows}) == 3 * n_points
    if spec["offsets"] == "spread":
        assert facts["nyquist_fraction"] < facts["margin"]
    opt = [r for r in rows if r["where"].endswith("fitted/optimum") and r["quantity"] == "grad sigma"]
    assert opt and opt[0]["rel"] > 1e-8  # what the 2026-10-23 relative criterion would have called a failure


@needs_toeplitz
@pytest.mark.validates("virgil.radio_toeplitz.ToeplitzVisData", roots=["mathematics"], kind="control")
def test_a_kernel_looser_than_documented_fails_the_budget():
    """The precompute at eps 1e-7 instead of the documented 1e-14: the budget (which assumes 1e-14) must catch it."""
    rows, _ = tr.tier1_scene(SMOKE[0], dict(TCFG, modes=["fixed"]), toeplitz_eps=1e-7)
    assert any(r["status"] == "FAIL" for r in rows)


# ------------------------------------------------------------------ Tier II and the summary on a tiny IM Lup-like set


@needs_toeplitz
@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_tier2_and_summary_on_a_tiny_set(tmp_path, monkeypatch):
    """Tier II agreement on a 32^2 config-R problem passes its budget; on a CPU the performance rows are NOT EVALUATED, so with no
    Tier I files the verdict is INCOMPLETE, never ACCEPTED; every JSON records the hash and the virgil commit."""
    from test_alma_maxent import synthetic_dir

    import virgil_bridge.alma as va

    syn = synthetic_dir(tmp_path / "s", n=150, n_obs=3, per_obs=2)
    data = va.load_dir(syn)
    prob = va.build_problem(data, "R", lam=va.lam_r(len(data.u)), npix=32)
    rng = np.random.default_rng(1)
    values = {k: np.asarray(v) for k, v in prob.init.items()}
    values["log_brightness"] = rng.normal(size=(32, 32))
    values["cal.gain"] = 1.0 + 0.02 * rng.normal(size=values["cal.gain"].shape)
    values["cal.offset"] = rng.uniform(-3, 3, size=values["cal.offset"].shape)
    fit = tmp_path / "fit"
    fit.mkdir()
    np.savez(fit / "values.npz", **values)
    out = tmp_path / "o"
    common = ["--data", syn, "--fit", fit, "--out", out, "--npix", 32, "--max-offset-mas", 20]
    tr.main(["tier2-agree", *map(str, common)])
    for method in ("toeplitz", "direct"):
        tr.main(["tier2-time", "--method", method, "--repeats", "1", *map(str, common)])
    tr.main(["summary", "--out", str(out)])
    a = json.loads((out / "tier2_agree.json").read_text())
    assert a["criteria_sha256"] == tr.REGISTERED_SHA256 and a["virgil_commit"]
    assert all(r["status"] == "PASS" for r in a["rows"]), [r for r in a["rows"] if r["status"] != "PASS"]
    assert {r["where"] for r in a["rows"]} == {"R_mpol/optimum", "R_mpol/moved", "R_mpol/edge"}
    s = json.loads((out / "reexam.json").read_text())
    assert s["verdict"] == "INCOMPLETE"
    assert any(r["status"] == "NOT EVALUATED" and "speed-up" in r["quantity"] for r in s["gated"])
    assert "VERDICT: INCOMPLETE" in (out / "reexam.txt").read_text()


@pytest.mark.validates("virgil.radio_toeplitz.ToeplitzVisData", roots=["self-consistency"], kind="guard")
def test_verdicts_and_the_no_combining_rule(tmp_path):
    assert tr.verdict(["PASS", "PASS"]) == "ACCEPTED"
    assert tr.verdict(["PASS", "NOT EVALUATED"]) == "INCOMPLETE"
    assert tr.verdict(["NOT EVALUATED", "FAIL"]) == "NOT ACCEPTED"
    base = dict(criteria_sha256=tr.REGISTERED_SHA256, rows=[], facts=dict(nyquist_fraction=0.1, margin=0.9))
    (tmp_path / "tier1_S1_ringed_disc.json").write_text(json.dumps(dict(base, virgil_commit="a" * 40)))
    (tmp_path / "tier1_S2_sources_on_halo.json").write_text(json.dumps(dict(base, virgil_commit="b" * 40)))
    with pytest.raises(SystemExit, match="several virgil commits"):
        tr.main(["summary", "--out", str(tmp_path)])
    (tmp_path / "tier1_S2_sources_on_halo.json").write_text(json.dumps(dict(base, virgil_commit="a" * 40, criteria_sha256="0" * 64)))
    with pytest.raises(SystemExit, match="other criteria"):
        tr.main(["summary", "--out", str(tmp_path)])


@pytest.mark.validates("virgil.radio_toeplitz.ToeplitzVisData", roots=["self-consistency"], kind="control")
def test_performance_off_the_registered_gpu_is_not_evaluated_and_on_it_is_judged():
    perf = dict(gpu="A100", speedup_min=30.0, peak_gb_max=20.0)
    t = {"toeplitz": dict(device="NVIDIA RTX PRO 6000 Blackwell Server Edition", s_per_vg=0.044, peak_gb=13.75),
         "direct": dict(device="NVIDIA RTX PRO 6000 Blackwell Server Edition", s_per_vg=0.7244, peak_gb=21.92)}
    rows = tr.performance_rows(t, perf)
    assert [r["status"] for r in rows] == ["NOT EVALUATED", "NOT EVALUATED", "REPORT"]
    t = {m: dict(x, device="NVIDIA A100-SXM4-80GB") for m, x in t.items()}
    assert [r["status"] for r in tr.performance_rows(t, perf)] == ["FAIL", "PASS", "REPORT"]  # 16x on an A100 would fail
    t["toeplitz"]["device"] = "NVIDIA A1000"
    assert tr.performance_rows(t, perf)[0]["status"] == "NOT EVALUATED"
