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
            ftol_abs=None, ftol_window=None, time_limit=None, progress=None, precondition=None, **kw):
        if seen is not None:
            seen.update(method=method, max_steps=max_steps, learning_rate=learning_rate, adam_eps=adam_eps, ftol_abs=ftol_abs, ftol_window=ftol_window)
        return types.SimpleNamespace(values=dict(init), info=dict(loss=float(history[-1]), loss_history=list(history), stop=None, converged=True, method=method))

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
    assert a["lr"] == 3e-3 and a["eps"] == 1e-6 and a["steps"] == 800 and a["loss_final"] == pytest.approx(flat[-1]) and a["c8"]["converged"] and a["virgil_converged"] is True
    assert info["converged"] is True and all(t["elapsed"] is None for t in traj)  # virgil's flag unchanged; no made-up per-step time
    _stub_adam(monkeypatch, 1000.0 - np.arange(1, 801))
    _, info, traj, why = va.run_adam(prob, {"x": np.zeros(2)}, steps=800)
    assert why == "max_steps" and not info["adam"]["c8"]["converged"]


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


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_adam_c8_is_a_window_test_that_an_oscillating_loss_fails():
    """100 + 5 sin(2 pi t / 100) passes a two-point test at some steps but ranges by 10 over a window: not converged. A flat loss with noise passes; a loss
    still drifting down by more than 0.1 per window (a small range, a falling mean) does not; fewer than two windows never does."""
    t = np.arange(1, 1001)
    assert not va.convergence_window(100 + 5 * np.sin(2 * np.pi * t / 100))["converged"]
    rng = np.random.default_rng(0)
    assert va.convergence_window(100 + 0.01 * rng.normal(size=1000))["converged"]
    drift = va.convergence_window(100 - 0.0015 * t)  # 0.15 per 100 steps, no noise
    assert not drift["converged"] and drift["mean_drop"] > 0.1
    assert not va.convergence_window(np.ones(150))["converged"]


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_every_adam_keyword_and_precondition_are_checked_before_data(tmp_path, monkeypatch):
    """A virgil with adam_eps and ftol_abs but not ftol_window, or not precondition when asked for, is refused with that name, before data is read."""
    def fit(model, priors, data, *, adam_eps=1e-8, ftol_abs=None, time_limit=None, progress=None, **kw):
        return None

    monkeypatch.setattr(virgil, "fit", fit)
    with pytest.raises(SystemExit, match="ftol_window"):
        _call("fit", "--config", "R", "--data", tmp_path / "nodata", "--out", tmp_path / "o", "--method", "adam")

    def fit2(model, priors, data, *, adam_eps=1e-8, ftol_abs=None, ftol_window=100, time_limit=None, progress=None):
        return None

    monkeypatch.setattr(virgil, "fit", fit2)
    with pytest.raises(SystemExit, match="precondition"):
        _call("fit", "--config", "R", "--data", tmp_path / "nodata", "--out", tmp_path / "o", "--method", "adam", "--precondition", "diag")
    va.require_adam(fit2)  # without precondition it is complete


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_fit_converged_is_virgils_and_c8_converged_is_vvs_for_both_methods():
    """Both methods' stage records carry ``fit_converged`` (virgil's ``info['converged']``) and ``c8_converged`` (vv's verdict)."""
    adam = dict(steps=300, rounds=1, stop="limit", converged=True, loss=1.0, adam=dict(c8=dict(converged=False, change=9.0)))
    s = script.stage_summary("all", adam, [], 1.0, "max_steps")
    assert s["fit_converged"] is True and s["c8_converged"] is False and s["convergence"]["converged"] is False
    lb = script.stage_summary("all", dict(steps=300, rounds=1, stop=None, converged=False, loss=1.0), [dict(steps=1.0, loss=5.0), dict(steps=300.0, loss=4.99)], 1.0, "c8")
    assert lb["fit_converged"] is False and lb["c8_converged"] is True


@pytest.mark.validates("virgil_bridge.alma.blur_post_step", roots=["self-consistency"], kind="guard")
def test_post_step_blur_conserves_flux_revives_a_dead_pixel_and_is_off_at_zero():
    """dorito's blur: the normalised image keeps unit mass and ``cal.flux`` is untouched; a pixel at 1e-12 of the peak beside live ones is raised above 1e-6 of
    the peak in one step; width 0 or None gives no callable; absent ``log_brightness`` (a held-image stage) passes through."""
    import jax
    import jax.numpy as jnp

    assert va.blur_post_step(0) is None and va.blur_post_step(None) is None
    with pytest.raises(ValueError):
        va.blur_post_step(-1.0)
    step = va.blur_post_step(0.25)
    n = 16
    lb = np.full((n, n), -30.0)
    lb[4:12, 4:12] = 0.0
    lb[8, 8] = -30.0  # a dead pixel inside the live block
    params = {"log_brightness": jnp.asarray(lb), "cal.flux": jnp.asarray(0.3), "cal.gain": jnp.ones(3)}
    out = step(params)
    p0 = jax.nn.softmax(params["log_brightness"].reshape(-1))
    p1 = jax.nn.softmax(out["log_brightness"].reshape(-1))
    assert float(p1.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(out["cal.flux"]) == 0.3 and np.array_equal(out["cal.gain"], params["cal.gain"])
    out_img = np.asarray(p1).reshape(n, n)
    in_img = np.asarray(p0).reshape(n, n)
    assert in_img[8, 8] < 1e-6 * in_img.max()
    assert out_img[8, 8] > 1e-6 * out_img.max()
    held = {"cal.flux": jnp.asarray(0.3)}
    assert step(held) is held


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_post_step_blur_is_adam_only_and_reaches_virgil_fit(tmp_path, monkeypatch):
    """lbfgs refuses it before any data is read; Adam needs ``post_step`` in virgil.fit (named in the error); with it the callable is passed and the width recorded."""
    monkeypatch.setattr(virgil, "fit", lambda model, priors, data, **kw: None)
    with pytest.raises(SystemExit, match="needs --method adam"):
        _call("fit", "--config", "R", "--data", tmp_path / "nodata", "--out", tmp_path / "o", "--post-step-blur-px", "0.25")
    seen = {}

    def fit(model, priors, data, regularizers=(), *, likelihoods=(), init=None, method=None, max_steps=None, learning_rate=1e-2, adam_eps=1e-8,
            ftol_abs=None, ftol_window=None, time_limit=None, progress=None, precondition=None, post_step=None, **kw):
        seen["post_step"] = post_step
        return types.SimpleNamespace(values=dict(init), info=dict(loss=1.0, loss_history=[1.0] * 3, stop=None, converged=True, method=method))

    prob = types.SimpleNamespace(model=None, priors={}, regs=(), terms=())
    monkeypatch.setattr(virgil, "fit", lambda model, priors, data, **kw: None)
    with pytest.raises(RuntimeError, match="post_step"):
        va.run_adam(prob, {}, steps=1, post_step_blur_px=0.25)
    monkeypatch.setattr(virgil, "fit", fit)
    _, info, _, _ = va.run_adam(prob, {"x": np.zeros(2)}, steps=3, post_step_blur_px=0.25)
    assert callable(seen["post_step"]) and info["adam"]["post_step_blur_px"] == 0.25
    seen.clear()
    _, info, _, _ = va.run_adam(prob, {"x": np.zeros(2)}, steps=3)
    assert seen["post_step"] is None and info["adam"]["post_step_blur_px"] is None
