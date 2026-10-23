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
    assert a["max_rel"] < 1e-10 and a["dloss_rel"] < 1e-6 and max(a["grad_rel"].values()) < 1e-8
    assert a["max_rel"] == max(a["chi2_rel"], a["loss_rel"]) and np.isfinite(a["max_rel_virgil"])  # the table's number is this script's own
    assert t["direct"]["precompute_s"] is None and t["direct"]["s_per_vg_score"] > 0 and t["direct"]["eps_fit"] == va.EPS_FIT
    text = (out / "toeplitz_check.txt").read_text()
    rows = [line for line in text.splitlines() if line.startswith(("PASS", "FAIL"))]
    assert len(rows) == 7 and "accept <= 1e-10" in text and ("ALL ACCEPTED" in text or "NOT ACCEPTED" in text)
    table = {q: ok for q, _, _, ok in chk.acceptance(a, t)}
    agreement = {q: ok for q, ok in table.items() if not q.startswith(("speed-up", "GPU peak"))}  # speed and memory can't be met on CPU at 32^2
    assert len(agreement) == 5 and all(agreement.values()), agreement


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="control")
def test_the_acceptance_table_fails_a_slow_or_large_or_disagreeing_run():
    """Control: the verdict is not always PASS. Each limit is breached in turn."""
    good = dict(max_rel=1e-13, dloss_rel=1e-9, dloss_abs=1e-6, grad_rel={"a": 1e-12}, nyquist_fraction=0.6, margin=0.99)
    t = {"toeplitz": dict(s_per_vg=0.01, peak_gb=8.0), "direct": dict(s_per_vg=0.5, peak_gb=6.0)}
    assert all(ok for *_, ok in chk.acceptance(good, t))
    for patch, tpatch in [(dict(max_rel=1e-9), {}), (dict(dloss_rel=1e-5), {}), (dict(dloss_abs=0.1), {}), (dict(grad_rel={"a": 1e-7}), {}),
                          (dict(grad_rel={"a": 1e-12, "b": float("nan")}), {}), (dict(grad_rel={"a": float("nan"), "b": 1e-12}), {}), (dict(max_rel=float("nan")), {}),
                          (dict(nyquist_fraction=0.995), {}), ({}, dict(toeplitz=dict(s_per_vg=0.1, peak_gb=8.0))), ({}, dict(toeplitz=dict(s_per_vg=0.01, peak_gb=25.0)))]:
        assert not all(ok for *_, ok in chk.acceptance({**good, **patch}, {**t, **tpatch}))
