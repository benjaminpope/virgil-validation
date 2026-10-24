"""``fit --method adam``: Adam through ``virgil.fit`` with C8 from its loss history, and the per-stage dead-pixel record. Small synthetic data on the CPU.
The Adam keywords of ``virgil.fit`` (``va.ADAM_KEYWORDS``) are stubbed, so these run against any virgil; a real Adam run is checked where the installed one has them."""

import inspect
import json
import pathlib
import sys
import types

import numpy as np
import pytest

pytest.importorskip("virgil.radio")

import virgil  # noqa: E402

import virgil_bridge.alma as va  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import alma_imlup_maxent as script  # noqa: E402
from test_alma_maxent import synthetic_dir  # noqa: E402


def _call(*argv):
    script.main([str(a) for a in argv])


def _stub_adam(monkeypatch, history, seen=None):
    """A ``virgil.fit`` with the Adam keywords that returns ``init`` unchanged and the given per-step loss history."""
    def fit(model, priors, data, regularizers=(), *, likelihoods=(), init=None, method=None, max_steps=None, learning_rate=1e-2, adam_eps=1e-8,
            ftol_abs=None, ftol_window=None, time_limit=None, precondition=None, **kw):
        if seen is not None:
            seen.update(method=method, max_steps=max_steps, learning_rate=learning_rate, adam_eps=adam_eps, ftol_abs=ftol_abs, ftol_window=ftol_window)
        return types.SimpleNamespace(values=dict(init), info=dict(loss=float(history[-1]), loss_history=list(history), stop=None, converged=None, method=method))

    monkeypatch.setattr(virgil, "fit", fit)


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_adam_loss_history_gives_the_c8_verdict_and_records_the_stage(monkeypatch):
    """A history that has flattened (change < 0.1 over 100 steps) is C8 and ``why`` is ``c8``; one still falling is not and ``why`` is ``max_steps``;
    the options reach ``virgil.fit`` and the stage record has lr, epsilon, steps, final loss and the verdict."""
    seen = {}
    flat = 100.0 + 50.0 * np.exp(-np.arange(1, 801) / 50.0)
    _stub_adam(monkeypatch, flat, seen)
    prob = types.SimpleNamespace(model=None, priors={}, regs=(), terms=())
    _, info, traj, why = va.run_adam(prob, {"x": np.zeros(2)}, steps=800, lr=3e-3, adam_eps=1e-6)
    assert why == "c8" and info["converged"] and info["steps"] == 800 and va.convergence(traj)["converged"]
    assert seen == dict(method="adam", max_steps=800, learning_rate=3e-3, adam_eps=1e-6, ftol_abs=0.1, ftol_window=100)
    a = info["adam"]
    assert a["lr"] == 3e-3 and a["eps"] == 1e-6 and a["steps"] == 800 and a["loss_final"] == pytest.approx(flat[-1]) and a["c8"]["converged"]
    _stub_adam(monkeypatch, 1000.0 - np.arange(1, 801))
    _, info, traj, why = va.run_adam(prob, {"x": np.zeros(2)}, steps=800)
    assert why == "max_steps" and not info["converged"] and not info["adam"]["c8"]["converged"]


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_method_adam_needs_a_virgil_with_the_adam_keywords(tmp_path, monkeypatch):
    """Without the keywords ``fit --method adam`` stops before reading any data, naming them; the default method needs none."""
    monkeypatch.setattr(virgil, "fit", lambda model, priors, data, **kw: None)
    with pytest.raises(SystemExit, match="adam_eps"):
        _call("fit", "--config", "R", "--data", tmp_path / "nodata", "--out", tmp_path / "o", "--method", "adam")
    with pytest.raises(RuntimeError, match="ftol_abs"):
        va.run_adam(types.SimpleNamespace(model=None, priors={}, regs=(), terms=()), {}, steps=1)


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_stage_dead_pixels_are_recorded_for_lbfgs_and_adam(tmp_path, monkeypatch):
    """Each stage of an L-BFGS fit (the default, unchanged) and of an Adam fit has ``dead_pixels`` for the base and tapered images with the 1e-6 and 1e-4
    fractions and the largest live radius, and the Adam stage has its loss history verdict; compare's notes tabulate them."""
    syn = synthetic_dir(tmp_path / "s")
    out = tmp_path / "lbfgs"
    _call("fit", "--config", "R", "--data", syn, "--out", out, "--npix", 64, "--max-steps", 3, "--no-dirty", "--chi2", "direct")
    r = json.loads((out / "result.json").read_text())
    assert r["args"]["method"] == "lbfgs" and all(s["method"] == "lbfgs" and s["adam"] is None for s in r["stages"])
    for s in r["stages"]:
        for img in ("base", "tapered"):
            d = s["dead_pixels"][img]
            assert {"frac_below_1e-6_peak", "frac_below_1e-4_peak", "largest_live_radius_arcsec"} <= set(d)
    assert "| sb | lbfgs |" in "\n".join(script.fit_notes(r))
    if not all(k in inspect.signature(virgil.fit).parameters for k in va.ADAM_KEYWORDS):
        flat = 100.0 + 50.0 * np.exp(-np.arange(1, 301) / 20.0)
        _stub_adam(monkeypatch, flat)
    out = tmp_path / "adam"
    _call("fit", "--config", "R", "--data", syn, "--out", out, "--npix", 64, "--method", "adam", "--adam-steps", 300, "--adam-lr", 1e-2, "--no-dirty")
    r = json.loads((out / "result.json").read_text())
    assert r["args"]["method"] == "adam"
    for s in r["stages"]:
        assert s["method"] == "adam" and s["adam"]["steps"] > 0 and s["adam"]["c8"]["converged"] in (True, False) and s["convergence"] == s["adam"]["c8"]
        assert s["dead_pixels"]["base"]["frac_below_1e-6_peak"] >= 0
