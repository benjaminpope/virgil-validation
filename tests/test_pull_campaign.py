"""Pull campaign (scripts/pull_campaign.py; criteria and design registered in
design/pull_campaign.yml). The first test reads the committed summary (it
skips until the OzSTAR job pull_campaign has run and been aggregated); the
rest check the registered file, the aggregator and the task files on
synthetic input."""

import argparse
import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec_ = importlib.util.spec_from_file_location("pull_campaign", ROOT / "scripts" / "pull_campaign.py")
pc = importlib.util.module_from_spec(spec_)
spec_.loader.exec_module(pc)

S = pc.spec()
SUMMARY = ROOT / "trust" / "campaigns" / "pull_campaign.json"


@pytest.mark.skipif(not SUMMARY.exists(), reason="the pull campaign has not run (ozstar_scripts job pull_campaign)")
@pytest.mark.validates("virgil.inference.laplace_cov", roots=["statistics"], tier="C")
def test_pull_campaign_pulls_are_standard_normal():
    s = json.loads(SUMMARY.read_text())
    assert s["criteria_hash"] == pc.criteria_hash() and s["design_hash"] == pc.design_hash()
    assert s["pass"], s["rejected"]


def fake_draw(scale=1.0, shift=0.0):
    def f(cell, seed, tmp):
        rng = np.random.default_rng(seed)
        return list(shift + scale * rng.standard_normal(len(pc.param_names(cell))))

    return f


def run_all(out, draws, **kw):
    for ci, cell in enumerate(S["cells"]):
        a = argparse.Namespace(cell=cell, task=0, draws=draws, seed_base=1, out=str(out))
        pc.run(a, draw_fn=fake_draw(**kw))


def agg(out, monkeypatch, min_draws):
    monkeypatch.setitem(S["counts"], "min_draws_per_cell", min_draws)
    monkeypatch.setattr(pc, "spec", lambda: S)
    monkeypatch.setenv("PIN_COMMIT", "a" * 40)
    a = argparse.Namespace(dirs=[str(out)], summary=str(out / "s.json"))
    pc.aggregate(a)
    return json.loads((out / "s.json").read_text())


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_registered_design():
    assert S["family_alpha"] == 0.01
    assert S["counts"]["min_draws_per_cell"] >= 1000
    assert S["counts"]["draws_per_task"] * S["counts"]["tasks_per_cell"] >= 1250
    assert sum(len(pc.param_names(c)) for c in S["cells"]) * 2 == 40


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_calibrated_passes(tmp_path, monkeypatch, capsys):
    run_all(tmp_path, 300)
    s = agg(tmp_path, monkeypatch, 300)
    assert s["pass"] and s["rejected"] == [] and s["tests"] == 40
    assert s["criteria_hash"] == pc.criteria_hash() and s["design_hash"] == pc.design_hash()


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
@pytest.mark.parametrize("kw", [{"scale": 1.15}, {"shift": 0.2}])
def test_miscalibration_fails(tmp_path, monkeypatch, capsys, kw):
    run_all(tmp_path, 1000, **kw)
    s = agg(tmp_path, monkeypatch, 1000)
    assert not s["pass"] and s["rejected"]


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_too_few_draws_not_a_pass(tmp_path, monkeypatch, capsys):
    run_all(tmp_path, 50)
    assert not agg(tmp_path, monkeypatch, 1000)["pass"]


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_failed_fits_counted(tmp_path, monkeypatch, capsys):
    def bad(cell, seed, tmp):
        raise RuntimeError("no")

    a = argparse.Namespace(cell="binary_fit", task=0, draws=5, seed_base=1, out=str(tmp_path))
    monkeypatch.setenv("PIN_COMMIT", "a" * 40)
    pc.run(a, draw_fn=bad)
    r = json.loads(next(tmp_path.glob("pull_*.json")).read_text())
    assert all(x["pulls"] == [None] * 3 for x in r["rows"])
    out = pc.analyse({"binary_fit": (r["params"], r["rows"])}, S)
    assert not out["pass"] and out["cells"]["binary_fit"]["nonfinite_fraction"] == 1


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_mixed_runs_refused(tmp_path, monkeypatch, capsys):
    run_all(tmp_path, 20)
    f = next(tmp_path.glob("pull_*.json"))
    r = json.loads(f.read_text())
    r["criteria_hash"] = "0" * 64
    f.write_text(json.dumps(r))
    with pytest.raises(SystemExit):
        agg(tmp_path, monkeypatch, 20)


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("PIN_COMMIT", "a" * 40)
    a = argparse.Namespace(cell="binary_fit", task=0, draws=6, seed_base=1, out=str(tmp_path))
    calls = []

    def f(cell, seed, tmp):
        calls.append(seed)
        if len(calls) == 4:
            raise KeyboardInterrupt
        return [0.0, 0.0, 0.0]

    with pytest.raises(KeyboardInterrupt):
        pc.run(a, draw_fn=f)
    calls.clear()
    pc.run(a, draw_fn=f)
    assert len(calls) == 3  # the three finished draws are not redone


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_holm():
    assert pc.holm({"a": 1e-9, "b": 0.5}, 0.01) == {"a": True, "b": False}


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_smoke_real_draw(tmp_path):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    for cell in S["cells"]:
        p = pc.draw(cell, 7, tmp_path)
        assert len(p) == len(pc.param_names(cell)) and np.all(np.isfinite(p))
