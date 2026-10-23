"""``scripts/alma_imlup_toeplitz_check.py`` on a tiny synthetic IM Lup-like set (CPU, seconds): the Toeplitz chi^2 of virgil#516 against the
direct term, the timing runs and the acceptance table. Skipped on a virgil without ``radio_toeplitz``."""

import json
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("virgil.radio_toeplitz")

import virgil_bridge.alma as va  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import alma_imlup_toeplitz_check as chk  # noqa: E402
from test_alma_maxent import synthetic_dir  # noqa: E402


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_toeplitz_check_agrees_with_the_direct_term_and_writes_the_acceptance_table(tmp_path):
    """R at 32^2 with fitted-looking nuisances: chi^2, loss and every gradient leaf of the Toeplitz term equal the direct term's, and the
    summary carries a PASS/FAIL row for each acceptance limit and an overall verdict."""
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
    chk.main(["agree", *map(str, common)])
    for method in ("toeplitz", "direct"):
        chk.main(["time", "--method", method, "--repeats", "1", *map(str, common)])
    chk.main(["summary", "--out", str(out)])
    a = json.loads((out / "agree.json").read_text())
    t = {m: json.loads((out / f"time_{m}.json").read_text()) for m in ("toeplitz", "direct")}
    assert a["max_rel"] < 1e-10 and a["dloss_rel"] < 1e-6 and max(a["grad_rel_moved"].values()) < 1e-8
    assert set(a["grad_scale"]) == set(a["grad_scale_moved"]) == set(a["grad_rel"]) and all(v > 0 for v in a["grad_scale_moved"].values())
    assert a["grad_scale_moved"]["cal.sigma"] != a["grad_scale"]["cal.sigma"]  # the moved point is a different point
    assert a["max_rel"] == max(a["chi2_rel"], a["loss_rel"]) and np.isfinite(a["max_rel_virgil"])  # the table's number is this script's own
    assert t["direct"]["precompute_s"] is None and t["direct"]["s_per_vg_score"] > 0 and t["direct"]["eps_fit"] == va.EPS_FIT
    text = (out / "toeplitz_check.txt").read_text()
    rows = [line for line in text.splitlines() if line.startswith(("PASS", "FAIL", "REPORT"))]
    assert len(rows) == 7 and "accept <= 1e-10" in text and ("ALL ACCEPTED" in text or "NOT ACCEPTED" in text)
    assert t["toeplitz"]["gpu"] == t["direct"]["gpu"] == chk.gpu_name() and chk.gpu_name() in text  # the device is recorded and shown
    assert any(line.startswith("REPORT") and "speed-up" in line for line in rows)  # not an A100: the speed-up is reported, not judged
    table = {q: ok for q, _, _, ok in chk.acceptance(a, t)}
    agreement = {q: ok for q, ok in table.items() if not q.startswith(("speed-up", "GPU peak"))}  # speed and memory can't be met on CPU at 32^2
    assert len(agreement) == 5 and all(agreement.values()), agreement


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="control")
def test_the_acceptance_table_fails_a_slow_or_large_or_disagreeing_run():
    """Control: the verdict is not always PASS. Each limit is breached in turn."""
    good = dict(max_rel=1e-13, dloss_rel=1e-9, dloss_abs=1e-6, grad_rel={"a": 1e-12}, grad_rel_moved={"a": 1e-12}, nyquist_fraction=0.6, margin=0.99)
    a100 = "NVIDIA A100-SXM4-40GB"
    t = {"toeplitz": dict(s_per_vg=0.01, peak_gb=8.0, gpu=a100), "direct": dict(s_per_vg=0.5, peak_gb=6.0, gpu=a100)}
    assert all(ok is True for *_, ok in chk.acceptance(good, t))
    for patch, tpatch in [(dict(max_rel=1e-9), {}), (dict(dloss_rel=1e-5), {}), (dict(dloss_abs=0.1), {}), (dict(grad_rel_moved={"a": 1e-7}), {}),
                          (dict(grad_rel_moved={"a": 1e-12, "b": float("nan")}), {}), (dict(grad_rel_moved={"a": float("nan"), "b": 1e-12}), {}), (dict(max_rel=float("nan")), {}),
                          (dict(nyquist_fraction=0.995), {}), ({}, dict(toeplitz=dict(s_per_vg=0.1, peak_gb=8.0, gpu=a100))), ({}, dict(toeplitz=dict(s_per_vg=0.01, peak_gb=25.0, gpu=a100)))]:
        assert any(ok is False for *_, ok in chk.acceptance({**good, **patch}, {**t, **tpatch}))
    # what is deliberately not gated: the gradient at the optimum (dL/ds ~ 0 there) and the dense direct path's memory
    assert all(ok is True for *_, ok in chk.acceptance({**good, "grad_rel": {"a": 1e-3}}, t))
    assert all(ok is True for *_, ok in chk.acceptance(good, {**t, "direct": dict(s_per_vg=0.5, peak_gb=40.0, gpu=a100)}))


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="control")
def test_the_speedup_limit_is_judged_on_an_a100_and_reported_elsewhere():
    """Both branches: on an A100 the 30x limit is PASS/FAIL (34x passes, 16x fails); on another GPU, an unrecorded one or a mixed pair the same
    speed-ups are REPORT, with the GPU named, and the run is still ACCEPTED while every other limit holds; REPORT never hides a real failure."""
    good = dict(max_rel=1e-13, dloss_rel=1e-9, dloss_abs=1e-6, grad_rel={"a": 1e-12}, grad_rel_moved={"a": 1e-12}, nyquist_fraction=0.6, margin=0.99)
    a100, blackwell = "NVIDIA A100-SXM4-40GB", "NVIDIA RTX PRO 6000 Blackwell Server Edition"

    def table(speedup, tgpu, dgpu=None, peak=8.0):
        t = {"toeplitz": dict(s_per_vg=0.01, peak_gb=peak), "direct": dict(s_per_vg=0.01 * speedup, peak_gb=6.0)}
        if tgpu is not None:
            t["toeplitz"]["gpu"], t["direct"]["gpu"] = tgpu, dgpu or tgpu
        return {q: (lim, ok) for q, _, lim, ok in chk.acceptance(good, t)}

    key = "speed-up per value+grad (direct at eps 1e-6)"
    assert table(34, a100)[key] == (">= 30x on A100", True)
    assert table(34, "NVIDIA A100-PCIE-40GB")[key][1] is True  # any A100 variant
    assert table(16, a100)[key] == (">= 30x on A100", False)
    for gpu in (blackwell, "cpu", None, "NVIDIA H100", "NVIDIA RTX A1000 Laptop GPU"):  # A1000 is not an A100  # None: a report from before the GPU was recorded
        for speedup in (16, 34):
            lim, ok = table(speedup, gpu)[key]
            assert ok is None and "report only" in lim and (gpu or "unknown") in lim
    assert table(34, a100, dgpu=blackwell)[key][1] is None  # the pair must be one device type
    assert not any(ok is False for _, ok in table(16, blackwell).values())  # a slow non-A100 run is ACCEPTED
    assert table(16, blackwell, peak=25.0)["GPU peak, Toeplitz path"][1] is False  # but a real failure still fails there
