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


def _stub_adam(monkeypatch, history, seen=None, doc=None):
    """A ``virgil.fit`` with the Adam keywords that returns ``init`` unchanged and the given per-step loss history."""
    def fit(model, priors, data, regularizers=(), *, likelihoods=(), init=None, method=None, max_steps=None, learning_rate=1e-2, adam_eps=1e-8,
            ftol_abs=None, ftol_window=None, time_limit=None, progress=None, precondition=None, **kw):
        if seen is not None:
            seen.update(method=method, max_steps=max_steps, learning_rate=learning_rate, adam_eps=adam_eps, ftol_abs=ftol_abs, ftol_window=ftol_window)
        return types.SimpleNamespace(values=dict(init), info=dict(loss=float(history[-1]), loss_history=list(history), stop=None, converged=True, method=method))

    fit.__doc__ = doc
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
    assert seen == dict(method="adam", max_steps=800, learning_rate=3e-3, adam_eps=1e-6, ftol_abs=None, ftol_window=100)  # this stub has no windowed ftol: no virgil stop, the full budget
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


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
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
    lb[0, :] = lb[:, 0] = lb[n - 1, 5] = 0.0  # flux on the border, where a leaky padding shows
    params = {"log_brightness": jnp.asarray(lb), "cal.flux": jnp.asarray(0.3), "cal.gain": jnp.ones(3)}
    out = step(params)
    p0 = jax.nn.softmax(params["log_brightness"].reshape(-1))
    p1 = jax.nn.softmax(out["log_brightness"].reshape(-1))
    from scipy.ndimage import gaussian_filter

    img0 = np.asarray(p0, float).reshape(n, n)
    ref = gaussian_filter(img0, 0.25, mode="reflect", truncate=3.0)  # scipy's reflect repeats the edge pixel: mass-conserving
    assert ref.sum() == pytest.approx(img0.sum(), rel=1e-12)
    np.testing.assert_allclose(np.asarray(p1, float).reshape(n, n), ref / ref.sum(), rtol=1e-4)  # the step's float32 softmax
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


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_adam_ftol_only_when_asked_and_stops_on_c8_only_when_the_window_passes(monkeypatch):
    """No ``ftol_abs`` unless ``--adam-ftol-abs``; then it reaches virgil. A virgil stop on a loss still falling is ``ftol``, not ``c8``."""
    prob = types.SimpleNamespace(model=None, priors={}, regs=(), terms=())
    seen = {}
    flat = 100.0 + 50.0 * np.exp(-np.arange(1, 401) / 20.0)
    _stub_adam(monkeypatch, flat, seen)
    monkeypatch.setitem(va._PROBED, virgil.fit, (True, None))
    _, info, _, why = va.run_adam(prob, {"x": np.zeros(2)}, steps=1000)
    assert seen["ftol_abs"] is None and why == "c8" and info["adam"]["ftol_abs"] is None
    _, info, _, why = va.run_adam(prob, {"x": np.zeros(2)}, steps=1000, ftol_abs=0.1)
    assert seen["ftol_abs"] == 0.1 and why == "c8"
    _stub_adam(monkeypatch, 1000.0 - np.arange(1, 401), seen)
    monkeypatch.setitem(va._PROBED, virgil.fit, (True, None))
    _, info, _, why = va.run_adam(prob, {"x": np.zeros(2)}, steps=1000, ftol_abs=0.1)
    assert why == "ftol" and not info["adam"]["c8"]["converged"]


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_adam_lr_schedule_const_then_decay_and_the_probe_refuses_an_old_virgil(monkeypatch, tmp_path):
    """Default a float lr; decay alone is ``exponential_decay``; with ``const_steps`` the lr is constant then decays from there. A virgil failing the probe refuses every new feature by
    name, and the flags' own checks hold before any data is read."""
    optax = pytest.importorskip("optax")
    prob = types.SimpleNamespace(model=None, priors={}, regs=(), terms=())
    seen = {}
    _stub_adam(monkeypatch, [1.0] * 300, seen)
    monkeypatch.setitem(va._PROBED, virgil.fit, (True, None))
    _, info, _, _ = va.run_adam(prob, {"x": np.zeros(2)}, steps=300, lr=1e-2)
    assert seen["learning_rate"] == 1e-2 and info["adam"]["lr_decay_per_1000"] is None
    va.run_adam(prob, {"x": np.zeros(2)}, steps=300, lr=1e-2, lr_decay=0.5)
    sched = seen["learning_rate"]
    assert float(sched(0)) == pytest.approx(1e-2) and float(sched(2000)) == pytest.approx(2.5e-3)
    _, info, _, _ = va.run_adam(prob, {"x": np.zeros(2)}, steps=300, lr=1e-2, lr_decay=0.5, const_steps=10000)
    sched = seen["learning_rate"]
    assert float(sched(9999)) == pytest.approx(1e-2) and float(sched(12000)) == pytest.approx(2.5e-3, rel=1e-3) and info["adam"]["const_steps"] == 10000
    monkeypatch.setitem(va._PROBED, virgil.fit, (False, "TypeError: no schedule"))
    for kw in (dict(lr_decay=0.5), dict(const_steps=100), dict(ftol_abs=0.1)):
        with pytest.raises(RuntimeError, match="virgil#562"):
            va.run_adam(prob, {"x": np.zeros(2)}, steps=300, **kw)
    va.run_adam(prob, {"x": np.zeros(2)}, steps=300)  # plain Adam needs no probe
    with pytest.raises(SystemExit, match="virgil#562"):
        _call("fit", "--config", "R", "--data", tmp_path / "nodata", "--out", tmp_path / "o", "--method", "adam", "--adam-lr-decay", "0.5")
    monkeypatch.setitem(va._PROBED, virgil.fit, (True, None))
    with pytest.raises(SystemExit, match=r"in \(0, 1\)"):
        _call("fit", "--config", "R", "--data", tmp_path / "nodata", "--out", tmp_path / "o", "--method", "adam", "--adam-lr-decay", "1.5")
    with pytest.raises(SystemExit, match="--method adam"):
        _call("fit", "--config", "R", "--data", tmp_path / "nodata", "--out", tmp_path / "o", "--adam-lr-decay", "0.5")
    with pytest.raises(SystemExit, match="--method adam"):
        _call("fit", "--config", "R", "--data", tmp_path / "nodata", "--out", tmp_path / "o", "--adam-const-steps", "5")


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_feature_probe_against_the_installed_virgil():
    """The probe is a real 3-step Adam fit with a schedule and ``ftol_abs``: it passes on a virgil with virgil#562 (CI's pin once it includes it) and the test is skipped on an older one."""
    pytest.importorskip("optax")
    ok, err = va.adam_features_probe(virgil.fit)
    if not ok:
        pytest.skip(f"installed virgil predates virgil#562: {err}")


def _ring(n=256, px=5.0, r_mas=600.0, w_mas=100.0, sparse=0, seed=0):
    from crosscheck.alma_maxent import phase_centre_cell

    r0, c0 = phase_centre_cell(n)
    rr, cc = np.indices((n, n), dtype=float)
    rad = np.hypot(rr - r0, cc - c0) * px
    ring = np.exp(-0.5 * ((rad - r_mas) / (w_mas / 2.355)) ** 2) + 1e-6
    if not sparse:
        return ring
    rng = np.random.default_rng(seed)
    pick = rng.choice(ring.size, sparse, replace=False, p=(ring / ring.sum()).ravel())
    out = np.full(ring.size, 1e-12)
    out[pick] = ring.sum() / sparse
    return out.reshape(ring.shape)


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_dots_report_separates_a_smooth_ring_from_the_same_ring_as_sparse_points():
    """A face-on-ish smooth ring has no holes, N_eff close to its tapered one and an azimuthal CV ratio near 1; the ring sampled as 400 points (closer than the taper) has holes, a small N_eff, a base peak far above the tapered one and power below the taper. (The sector CV ratio is blind to structure smaller than a sector, which the taper also averages: it is asserted finite only.)"""
    smooth = _ring()
    dots = _ring(sparse=400)
    rs = va.dots_report(smooth, va.tapered_image(smooth, 5.0), 5.0, inc=0.0)
    rd = va.dots_report(dots, va.tapered_image(dots, 5.0), 5.0, inc=0.0)
    assert rs["hole_fraction"] < 0.05 and rs["ring_intermittency"]["median"] < 1.5 and rs["n_eff"]["base"] > 0.8 * rs["n_eff"]["tapered"] and rs["sub_taper_power"] < 0.05
    assert rd["hole_fraction"] > 0.5 and np.isfinite(rd["ring_intermittency"]["median"]) and rd["n_eff"]["base"] < 0.2 * rd["n_eff"]["tapered"] and rd["sub_taper_power"] > rs["sub_taper_power"]
    assert rd["peak_ratio"] > 5 * rs["peak_ratio"] and rd["dead"]["base_below_1e-06"]["in_disk"] > rs["dead"]["base_below_1e-06"]["in_disk"]
    assert rd["n_eff_ratio"] < 0.2 < 0.8 < rs["n_eff_ratio"] and rd["n_eff_ratio"] == pytest.approx(rd["n_eff"]["base"] / rd["n_eff"]["tapered"])
    assert rd["base_dead_in_disk_1e4"] == rd["dead"]["base_below_0.0001"]["in_disk"] and rd["tapered_dead_outside_1e4"] == rd["dead"]["tapered_below_0.0001"]["outside"]


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_snapshot_post_step_calls_the_sink_every_n_steps_and_passes_params_through():
    """The host callback fires once per call and the sink gets the image logit at every ``every``-th call; the params are returned unchanged."""
    import jax.numpy as jnp

    got = []
    ps = va.snapshot_post_step(lambda k, lb: got.append((k, lb.copy())), every=3)
    params = {"log_brightness": jnp.arange(4.0), "cal.flux": jnp.asarray(1.0)}
    for _ in range(7):
        out = ps(params)
    import jax

    jax.effects_barrier()
    assert [k for k, _ in got] == [3, 6] and out is params and np.allclose(got[0][1], np.arange(4.0))


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_classic_mem_arm_writes_summary_first_dots_and_the_standard_r_objective(tmp_path, monkeypatch):
    """``--taper-mas 0`` (classic MEM: no taper on the model or the entropy) still records the standard R (30 mas taper) of the final image; result.json starts with the raw chi^2/N
    and has ``dots``, the tapered peak and the flux beyond 1.8 arcsec; with Adam's snapshots (a real Adam run) the trajectory has the loss and a dots_report per snapshot."""
    syn = synthetic_dir(tmp_path / "s")
    real = all(k in inspect.signature(virgil.fit).parameters for k in va.ADAM_KEYWORDS + ("post_step",))
    if not real:
        _stub_adam(monkeypatch, 100.0 + 50.0 * np.exp(-np.arange(1, 301) / 20.0))
    out = tmp_path / "mem"
    _call("fit", "--config", "R", "--data", syn, "--out", out, "--npix", 64, "--method", "adam", "--adam-steps", 120, "--adam-snapshot-every", 50, "--taper-mas", 0, "--no-dirty")
    r = json.loads((out / "result.json").read_text())
    assert next(iter(r)) == "summary" and {"chi2n_raw", "peak_tapered_jy_arcsec2", "flux_beyond_1p8_arcsec_jy"} <= set(r["summary"]) and r["summary"]["taper_mas"] == 0
    assert {"n_eff", "hole_fraction", "ring_intermittency", "sub_taper_power", "peak_ratio", "dead"} <= set(r["dots"])
    assert np.isfinite(r["objective_R_taper30"]) and r["objective_R_taper30"] != r["objective_R"]
    if real:
        assert [t["step"] for t in r["trajectory"] if t["stage"] == "all"][:2] == [50, 100] and all("dots" in t and t["loss"] is not None for t in r["trajectory"])


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_real_adam_with_const_then_decay_schedule_ftol_and_snapshots(tmp_path):
    """The release arms' flags together on a real (small) Adam run: constant lr then decay, the windowed ftol, snapshots and Toeplitz-free direct chi^2; skipped on a virgil without virgil#562."""
    pytest.importorskip("optax")
    ok, err = va.adam_features_probe(virgil.fit)
    if not ok or "post_step" not in inspect.signature(virgil.fit).parameters:
        pytest.skip(f"installed virgil predates virgil#562: {err}")
    syn = synthetic_dir(tmp_path / "s")
    out = tmp_path / "o"
    _call("fit", "--config", "R", "--data", syn, "--out", out, "--npix", 64, "--method", "adam", "--adam-steps", 150, "--adam-lr", 1e-2, "--adam-const-steps", 50, "--adam-lr-decay", 0.7,
          "--adam-ftol-abs", 1e-9, "--adam-snapshot-every", 100, "--no-dirty")
    r = json.loads((out / "result.json").read_text())
    a = r["stages"][-1]["adam"]
    assert a["const_steps"] == 50 and a["lr_decay_per_1000"] == 0.7 and a["ftol_abs"] == 1e-9 and a["steps"] == 150
    assert [t["step"] for t in r["trajectory"] if t["stage"] == "all"] == [100]
    out2 = tmp_path / "o2"  # --init with Adam, the MPoL-warm release arm's mode: one all-data stage from the saved values
    _call("fit", "--config", "R", "--data", syn, "--out", out2, "--npix", 64, "--method", "adam", "--adam-steps", 60, "--adam-lr-decay", 0.7, "--adam-const-steps", 20, "--init", out, "--no-dirty")
    assert len(json.loads((out2 / "result.json").read_text())["stages"]) == 1
