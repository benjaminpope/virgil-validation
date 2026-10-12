"""Cold-start campaign (scripts/coldstart_pull.py; criteria and design registered in
design/coldstart_pull.yml). The first test reads the committed summary (it skips until
the OzSTAR job has run and been aggregated); the rest check the registered file and its
hash, the classification and the aggregator on synthetic rows, and that a real draw runs
the user path with no truth in the fit."""

import argparse
import hashlib
import importlib.util
import json
import pathlib
import subprocess

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec_ = importlib.util.spec_from_file_location("coldstart_pull", ROOT / "scripts" / "coldstart_pull.py")
cp = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(cp)

S = cp.spec()
SUMMARY = ROOT / "trust" / "campaigns" / "coldstart_pull.json"
# SHA-256 of design/coldstart_pull.yml as registered (commit "Register the cold-start pull campaign",
# then amendment 1 before any campaign draw, see the file's header). Changing the file changes the
# hash: register a new version before any run instead.
ORIGINAL_HASH = "5c0229024d9d45770754c5f45c64b34023ce8ff799aee7320dc602460751a26b"  # superseded by amendment 1 (grids)
REGISTERED_HASH = "96fb09a785f999f840c41d0c7cfc2d7373705f248d0526dfdc87c10b1d8ddba2"


@pytest.mark.skipif(not SUMMARY.exists(), reason="the cold-start campaign has not run (OzSTAR)")
@pytest.mark.validates("virgil.grid_fit.likelihood_grid", roots=["statistics"], tier="C")
def test_coldstart_campaign_failure_rates_within_bounds():
    s = json.loads(SUMMARY.read_text())
    assert s["criteria_hash"] == cp.criteria_hash() and s["design_hash"] == cp.design_hash()
    assert s["pass"], {c: (e["failures"], e["n"], e["bound"]) for c, e in s["cells"].items()}


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_registered_hash():
    """The registered file hashes to the registered value, and that file was committed
    (the original too, so that the amendment is visible in history)."""
    assert cp.criteria_hash() == REGISTERED_HASH

    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True)

    commits = git("log", "--format=%H", "--", "design/coldstart_pull.yml").stdout.decode().split()
    if len(commits) < 2:  # shallow clone, or the amendment is not yet committed
        pytest.skip("history of design/coldstart_pull.yml not available")
    blobs = {hashlib.sha256(git("show", f"{c}:design/coldstart_pull.yml").stdout).hexdigest() for c in commits}
    assert {ORIGINAL_HASH, REGISTERED_HASH} <= blobs


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_registered_design():
    assert S["family_alpha"] == 0.01 and S["failure_delta_chi2"] == 1.0
    assert S["counts"]["draws_per_task"] * S["counts"]["tasks_per_cell"] >= 1000
    assert S["counts"]["min_draws_per_cell"] >= 900
    assert {c: v["failure_bound"] for c, v in S["cells"].items()} == {
        "binary_bright": 0.01, "binary_faint": 0.05, "binary_wide": 0.05, "binary_sparse": 0.05,
        "resolved_companion": 0.02, "envelope": 0.02}
    for cell in S["cells"]:
        assert len(cp.param_names(cell)) in (3, 4)
        cell_spec = S["cells"][cell]
        grid = cp.grid_axes(cell_spec) | (cp.stage1_axes(cell_spec) or {})
        assert set(grid) == set(cp.param_names(cell))  # one axis per fitted parameter


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_no_truth_in_the_cold_path():
    """grid, priors and start depend on the cell alone: two draws of a cell with different
    truths see identical grids and priors (the grid_axes/priors_of signatures take no truth)."""
    import inspect

    for fn in (cp.grid_axes, cp.stage1_axes, cp.priors_of, cp.generic_template):
        assert "truth" not in str(inspect.signature(fn)) and "scene" not in str(inspect.signature(fn))
    assert list(inspect.signature(cp.cold_start).parameters) == ["cell_spec", "data"]  # data only


def row(cold, ref, far=(0.0,), **kw):
    return {"seed": 0, "cold_chi2": cold, "ref_chi2": ref, "ref_distance_sigma": list(far), "pulls": [0.0], "snr": 10.0,
            "sep": 5.0, "contrast": 0.05, **kw}


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_classification():
    assert cp.classify(row(100.5, 100.0), S) == "same_mode"
    assert cp.classify(row(102.0, 100.0), S) == "failure"
    assert cp.classify(row(90.0, 100.0), S) == "better"
    assert cp.classify(row(100.0, 100.0, far=(7.0,)), S) == "alias"
    assert cp.classify({"cold_failed": True}, S) == "failure"
    assert cp.classify({"ref_failed": True}, S) == "reference_failed"


def synthetic(rate, n, rng):
    return [row(100.0 + (5.0 if rng.random() < rate else 0.0), 100.0) for _ in range(n)]


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_analysis_flags_rates_over_bound():
    rng = np.random.default_rng(1)
    names = {c: (cp.param_names(c), None) for c in S["cells"]}
    ok = {c: (cp.param_names(c), synthetic(0.3 * S["cells"][c]["failure_bound"], 1000, rng)) for c in S["cells"]}
    out = cp.analyse(ok, S)
    assert out["pass"] and out["rejected"] == []
    bad = dict(ok)
    bad["binary_bright"] = (cp.param_names("binary_bright"), synthetic(0.04, 1000, rng))
    out = cp.analyse(bad, S)
    assert not out["pass"] and out["rejected"] == ["binary_bright"]
    few = {c: (n, r[:50]) for c, (n, r) in ok.items()}
    assert not cp.analyse(few, S)["pass"]  # too few draws is not a pass
    assert names  # keep the helper used


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_aggregate_refuses_smoke_and_changed_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("PIN_COMMIT", "a" * 40)
    a = argparse.Namespace(cell="binary_bright", task=0, draws=2, seed_base=1, out=str(tmp_path), smoke=True)
    cp.run(a, draw_fn=lambda cell, seed, tmp: row(100.0, 100.0))
    with pytest.raises(SystemExit, match="smoke"):
        cp.aggregate(argparse.Namespace(dirs=[str(tmp_path)], summary=None))
    f = next(tmp_path.glob("coldstart_*.json"))
    r = json.loads(f.read_text())
    r["smoke"], r["criteria_hash"] = False, "0" * 64
    f.write_text(json.dumps(r))
    with pytest.raises(SystemExit, match="criteria changed"):
        cp.aggregate(argparse.Namespace(dirs=[str(tmp_path)], summary=None))


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("PIN_COMMIT", "a" * 40)
    a = argparse.Namespace(cell="binary_bright", task=0, draws=6, seed_base=1, out=str(tmp_path))
    calls = []

    def f(cell, seed, tmp):
        calls.append(seed)
        if len(calls) == 4:
            raise KeyboardInterrupt
        return row(100.0, 100.0)

    with pytest.raises(KeyboardInterrupt):
        cp.run(a, draw_fn=f)
    calls.clear()
    cp.run(a, draw_fn=f)
    assert len(calls) == 3


@pytest.mark.validates("virgil.grid_fit.likelihood_grid", roots=["statistics"])
def test_smoke_real_draw(tmp_path):
    """One real draw of the binary cell: the cold fit finds the reference fit's mode."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    r = cp.draw("binary_bright", 12345, tmp_path)
    assert not r["cold_failed"] and not r["ref_failed"]
    assert cp.classify(r, S) in ("same_mode", "better", "alias", "failure")
    assert np.isfinite(r["snr"]) and len(r["pulls"]) == 3
