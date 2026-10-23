"""The robustness of the IM Lup MaxEnt run (P3 first-run fixes): chunked model evaluation, saving before reporting, and
stopping on C8. Small synthetic data on the CPU; the design is ``design/alma_imlup_p3.md`` ("Built: first run")."""

import json
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("virgil.radio")

import jax.numpy as jnp  # noqa: E402

import virgil_bridge.alma as va  # noqa: E402
from evidence.plugin import record  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import alma_imlup_maxent as script  # noqa: E402
from test_alma_maxent import synthetic_dir  # noqa: E402


def _call(*argv):
    script.main([str(a) for a in argv])


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    return synthetic_dir(tmp_path_factory.mktemp("syn"))


def _state(syn, npix=32):
    data = va.load_dir(syn)
    prob = va.build_problem(data, "R", lam=0.5, npix=npix)
    rng = np.random.default_rng(1)
    v = {k: np.asarray(x) for k, x in va.start_values(prob).items()}
    v["log_brightness"] = v["log_brightness"] + 0.3 * rng.normal(size=v["log_brightness"].shape)
    v["cal.sigma"] = 1.0 + 0.1 * rng.uniform(size=np.shape(v["cal.sigma"]))
    return data, prob, v


@pytest.mark.validates("virgil_bridge.alma", roots=["mathematics"])
def test_chunked_score_and_dirty_images_equal_the_unchunked_ones(syn):
    """Model evaluation in chunks of 37 points (every observation split many times, the last chunk short) gives the
    same per-point decomposition, tallies and dirty images as one call per observation, to 1e-12; and no chunk is larger
    than ``chunk_size``."""
    data, prob, v = _state(syn)
    rng = np.random.default_rng(0)
    mask = rng.uniform(size=len(data.u)) < 0.6
    ann = rng.integers(0, 12, len(data.u))
    whole = va.score(data, prob.scene, v, prob.nuis, mask=mask, annulus=ann, chunk_size=10**9)
    sizes = []
    orig = va.model_chunks

    def spy(*a, **k):
        for idx, sub, m in orig(*a, **k):
            sizes.append(len(idx))
            yield idx, sub, m

    va.model_chunks = spy
    try:
        parts = va.score(data, prob.scene, v, prob.nuis, mask=mask, annulus=ann, chunk_size=37)
        d_parts = va.residual_dirty_images(data, prob.scene, v, prob.nuis, npix=64, pixel_mas=20.0, chunk_size=37)
    finally:
        va.model_chunks = orig
    record("max_chunk", max(sizes))
    assert max(sizes) <= 37 and min(sizes) >= 1 and len(sizes) > 50
    assert abs(parts["nll"] - whole["nll"]) < 1e-12 * abs(whole["nll"])
    for key in ("chi2_raw", "chi2_fit"):
        assert abs(parts[key] - whole[key]) < 1e-12 * abs(whole[key])
    for tally in ("by_ddid", "by_annulus"):
        for k in ("n", "nll", "chi2_raw", "chi2_fit"):
            assert np.allclose(parts[tally][k], whole[tally][k], rtol=1e-12, atol=0)
    d_whole = va.residual_dirty_images(data, prob.scene, v, prob.nuis, npix=64, pixel_mas=20.0, chunk_size=10**9)
    assert d_parts.keys() == d_whole.keys()
    for e in d_whole:
        assert np.allclose(d_parts[e], d_whole[e], rtol=1e-10, atol=1e-12 * np.abs(d_whole[e]).max())


@pytest.mark.validates("virgil_bridge.alma", roots=["mathematics"])
def test_a_failed_report_loses_no_fit_and_report_recomputes_it(syn, tmp_path, monkeypatch):
    """The fit's values, image (npy and fits) and a provisional result.json (stages, loss history, complete false) are on disk
    when the reporting step raises, as the out-of-memory did on the A100; ``report`` then writes the full result, equal to the
    one an uninterrupted fit writes, without refitting (its wall time is the fit's)."""
    out, ref = tmp_path / "fit", tmp_path / "ref"
    args = ["fit", "--config", "R", "--data", syn, "--npix", 32, "--max-steps", 20, "--time-limit", 600, "--no-dirty"]
    real = script.write_report

    def boom(*a, **k):
        raise RuntimeError("RESOURCE_EXHAUSTED (simulated)")

    monkeypatch.setattr(script, "write_report", boom)
    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        _call(*args, "--out", out)
    monkeypatch.setattr(script, "write_report", real)
    for f in ("values.npz", "image.npy", "image.fits", "result.json"):
        assert (out / f).exists(), f
    prov = json.loads((out / "result.json").read_text())
    assert prov["complete"] is False and [s["name"] for s in prov["stages"]] == ["sb", "all"]
    assert prov["stages"][-1]["trajectory"] and prov["stopped_because"] in ("max_steps", "c8", "time_limit", "stalled")
    assert not list(out.glob("*.tmp"))
    before = (out / "values.npz").read_bytes()
    _call("report", "--data", syn, "--out", out, "--chunk-size", 500)
    full = json.loads((out / "result.json").read_text())
    assert (out / "values.npz").read_bytes() == before  # not refitted
    assert full["complete"] is True and full["stages"] == prov["stages"]
    _call(*args, "--out", ref)  # the uninterrupted fit: same fit, same report
    whole = json.loads((ref / "result.json").read_text())
    assert abs(full["nll"] - whole["nll"]) < 1e-9 * abs(whole["nll"]) and abs(full["objective_R"] - whole["objective_R"]) < 1e-9 * abs(whole["objective_R"])
    assert full["chi2_s1"]["chi2_raw_over_n"] == pytest.approx(whole["chi2_s1"]["chi2_raw_over_n"], rel=1e-9)


def _fake_fit(rounds):
    """A stand-in for ``va.run_fit``: each call returns the next ``(steps, stop, [losses at record steps])``."""
    it = iter(rounds)
    state = dict(loss=0.0)

    def fake(prob, values, *, max_steps, time_limit=None, eps=None, log=None, gtol=1e-4):
        steps, stop, drops = next(it)
        steps = min(steps, max_steps)
        traj = []
        for k, drop in enumerate(drops):
            state["loss"] -= drop
            traj.append(dict(steps=float(min(200 * (k + 1), steps)), loss=state["loss"], grad_norm=0.0, elapsed=0.0))
        if not traj or traj[-1]["steps"] != steps:
            traj.append(dict(steps=float(steps), loss=state["loss"], grad_norm=0.0, elapsed=0.0))
        return values, dict(steps=steps, stop=stop, converged=stop is None, loss=state["loss"], grad_norm=0.0), traj

    return fake


@pytest.mark.validates("virgil_bridge.alma", roots=["mathematics"])
def test_c8_loop_restarts_until_the_loss_has_stopped_moving(monkeypatch):
    """`fit` reporting converged (stop None) while the loss still falls by 50 is not C8: the loop restarts it, and stops only
    when the loss has changed by less than 0.1 over >= 100 steps, saying so. Controls: a loop that always gains runs to
    ``max_steps``, one that stalls stops with ``stalled`` and one out of time with ``time_limit``."""
    p = va
    monkeypatch.setattr(p, "run_fit", _fake_fit([(149, None, [100.0]), (300, None, [50.0, 10.0]), (200, None, [0.04, 0.02])]))
    rounds = []
    _, info, traj, why = va.run_fit_c8(None, {}, max_steps=1000, on_round=lambda v, i, t, r: rounds.append(r))
    assert why == "c8" and info["rounds"] == 3 and info["steps"] == 649 and rounds == [1, 2, 3]
    assert [t["steps"] for t in traj] == sorted(t["steps"] for t in traj)  # steps are cumulative across rounds
    assert va.convergence(traj)["converged"]
    # control: the first round alone (stop None, one record) is not C8
    monkeypatch.setattr(p, "run_fit", _fake_fit([(149, None, [100.0]), (300, None, [60.0, 20.0, 5.0])]))
    _, info, traj, why = va.run_fit_c8(None, {}, max_steps=449)
    assert why == "max_steps" and not va.convergence(traj)["converged"]
    monkeypatch.setattr(p, "run_fit", _fake_fit([(100, "stalled", [5.0])]))
    assert va.run_fit_c8(None, {}, max_steps=1000)[3] == "stalled"
    monkeypatch.setattr(p, "run_fit", _fake_fit([(100, "time", [5.0])]))
    assert va.run_fit_c8(None, {}, max_steps=1000, time_limit=10.0)[3] == "time_limit"


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"])
def test_c8_loop_goes_further_than_fits_own_loose_test(syn):
    """On a real (small) problem a loose ``gtol`` makes `fit` stop with stop None well short of the minimum; the C8 loop restarts
    it and reaches a lower loss, in more than one round, with C8 true or the step budget spent."""
    data = va.load_dir(syn)
    prob = va.build_problem(data, "R", lam=0.5, npix=24)
    start = va.start_values(prob)
    _, plain_info, plain_traj = va.run_fit(prob, start, max_steps=2000, gtol=3e-2)
    _, info, traj, why = va.run_fit_c8(prob, start, max_steps=2000, gtol=3e-2)
    record("loss_plain", plain_info["loss"])
    record("loss_c8", info["loss"])
    record("rounds", info["rounds"])
    assert info["rounds"] > 1 and info["loss"] < plain_info["loss"] - 1.0
    assert why in ("c8", "max_steps") and (why != "c8" or va.convergence(traj)["converged"])


@pytest.mark.validates("virgil_bridge.alma", roots=["mathematics"])
def test_precondition_needs_a_virgil_that_has_it_and_is_recorded_when_it_does(syn, monkeypatch):
    """``precondition="diag"`` raises a clear error on a virgil whose fit() lacks the argument; with one that has it the
    C8 loop records the scaling's per-round stats and each restart's steps and wall time."""
    import inspect

    import virgil

    data = va.load_dir(syn)
    prob = va.build_problem(data, "R", lam=0.5, npix=16)
    start = va.start_values(prob)
    if "precondition" not in inspect.signature(virgil.fit).parameters:
        with pytest.raises(RuntimeError, match="precondition"):
            va.run_fit(prob, start, max_steps=10, precondition="diag")
        return
    _, info, _, _ = va.run_fit_c8(prob, start, max_steps=300, precondition="diag")
    assert info["precondition"] == "diag" and info["precondition_rounds"] and info["restarts"]
    _, plain, _, _ = va.run_fit_c8(prob, start, max_steps=50)
    assert plain["precondition_rounds"] == [] and plain["precondition"] is None
    monkeypatch.setattr(virgil, "fit", lambda model, priors, data, **kw: virgil.__dict__["fit"])  # unused


@pytest.mark.validates("virgil_bridge.alma", roots=["mathematics"])
def test_c8_needs_100_steps_and_a_bad_round_is_not_saved(monkeypatch):
    """Short rounds (30 steps each, loss still) are not C8 (window under 100 steps); a non-finite round stops the loop without
    calling ``on_round`` (no NaN save); a time limit already spent returns instead of raising; virgil's real ``time`` stop
    is what the loop maps to ``time_limit``."""
    monkeypatch.setattr(va, "run_fit", _fake_fit([(30, None, [0.01]), (30, None, [0.01]), (30, None, [0.01]), (30, None, [0.01])]))
    _, info, traj, why = va.run_fit_c8(None, {}, max_steps=120)
    assert why == "max_steps" and info["steps"] == 120
    saved = []
    monkeypatch.setattr(va, "run_fit", lambda *a, **k: ({"x": np.array([np.nan])}, dict(steps=10, stop="non-finite", loss=float("nan")), []))
    _, _, _, why = va.run_fit_c8(None, {}, max_steps=100, on_round=lambda *a: saved.append(1))
    assert why == "non_finite" and not saved
    assert va.run_fit_c8(None, {}, max_steps=100, time_limit=0.0)[3] == "time_limit"


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"])
def test_virgils_real_time_stop_is_spelled_time(syn):
    data = va.load_dir(syn)
    prob = va.build_problem(data, "F", lam=1000.0, npix=32)  # F needs more than a 200-step chunk, where the time test runs
    v = va.start_values(prob)
    _, info, _ = va.run_fit(prob, v, max_steps=5000, time_limit=1e-6, gtol=1e-12)  # the time test runs after a 200-step chunk
    assert info["stop"] == "time", info


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_f_fixes_alpha_unless_opted_in(syn):
    data = va.load_dir(syn)
    fixed = va.build_problem(data, "F", lam=1000.0, npix=16)
    assert "cal.alpha" not in fixed.init and "cal.alpha" not in fixed.priors
    free = va.build_problem(data, "F", lam=1000.0, npix=16, alpha_free=True)
    assert "cal.alpha" in free.init and "cal.alpha" in free.priors
    assert va.alpha_report(fixed.init) == (None, "fixed 0 (MPoL)")
    a, label = va.alpha_report(free.init)
    assert a == float(free.init["cal.alpha"]) and "None" not in label


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_warm_start_from_free_alpha_values_drops_alpha(syn):
    data = va.load_dir(syn)
    old = {**va.build_problem(data, "F", lam=1000.0, npix=16, alpha_free=True).init, "cal.alpha": 5.0}
    prob = va.build_problem(data, "F", lam=1000.0, npix=16)
    kept = va.drop_unused_sites({**old, "log_brightness": 0.0}, prob)
    assert "cal.alpha" not in kept and set(kept) == set(prob.init) | {"log_brightness"}
    assert va.alpha_report(kept) == (None, "fixed 0 (MPoL)")
    assert "cal.alpha" in va.drop_unused_sites(old, va.build_problem(data, "F", lam=1000.0, npix=16, alpha_free=True))
    # report on a pre-PR free-alpha fit: args carry no alpha_free, so the setting is inferred from the saved values
    import argparse

    ns = argparse.Namespace(alpha_free=None, config="F")
    import importlib.util, pathlib

    spec = importlib.util.spec_from_file_location("maxent_cli", pathlib.Path(__file__).parents[1] / "scripts" / "alma_imlup_maxent.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    cli.resolve_alpha(ns)
    assert ns.alpha_free is False


# ------------------------------------------------------------------ the MPoL workflow (--workflow mpol)


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_mpol_start_reproduces_the_uniform_softplus_image_and_flux(syn):
    """`mpol_start_values` is uniform through our softmax parametrisation (zero logits, flux = value dx^2 npix^2): `image_of` returns one
    value in every pixel and the pixels sum to the flux, to 1e-12; gains 1, offsets 0, s = 1; the C9 start is not uniform. That the value is
    softplus(-3) and that sgd.py starts there is transcribed from mpol 0.3.1 images.py:65-76 and sgd.py:49-59 (cited in the design note), not
    checked against MPoL here."""
    npix, px = 24, 5.0
    data = va.load_dir(syn)
    prob = va.build_problem(data, "R", lam=0.5, npix=npix, pixel_mas=px)
    v = {k: np.asarray(x) for k, x in va.mpol_start_values(prob).items()}
    img = va.image_of(v, npix, px)
    sp = np.log1p(np.exp(-3.0))
    record("rel_pixel_error", float(np.max(np.abs(img / sp - 1))))
    assert np.max(np.abs(img / img.flat[0] - 1)) < 1e-12 and abs(img.flat[0] / sp - 1) < 1e-12
    flux = sp * (px / 1000) ** 2 * npix**2
    assert abs(float(v["cal.flux"]) / flux - 1) < 1e-12
    assert abs(img.sum() * (px / 1000) ** 2 / flux - 1) < 1e-12
    assert np.all(v["cal.gain"] == 1) and np.all(v["cal.offset"] == 0) and np.all(v["cal.sigma"] == 1)
    c9 = {k: np.asarray(x) for k, x in va.start_values(prob).items()}
    assert abs(va.image_of(c9, npix, px).max() / sp - 1) > 1e-3


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_workflow_mpol_runs_exactly_one_stage_and_records_itself(syn, tmp_path, monkeypatch):
    calls = []
    real = va.mpol_start_values
    monkeypatch.setattr(va, "mpol_start_values", lambda prob: calls.append(1) or real(prob))
    monkeypatch.setattr(va, "start_values", lambda *a, **k: pytest.fail("the mpol workflow must not use C9's start"))
    out = tmp_path / "m"
    _call("fit", "--config", "R", "--workflow", "mpol", "--data", syn, "--out", out, "--npix", 32, "--max-steps", 5, "--no-dirty")
    r = json.loads((out / "result.json").read_text())
    assert r["workflow"] == "mpol" and calls == [1]
    assert [s["name"] for s in r["stages"]] == ["all"]
    assert r["n"] == 5600  # all data, no SB subset
    monkeypatch.undo()
    p3 = tmp_path / "p"
    _call("fit", "--config", "R", "--data", syn, "--out", p3, "--npix", 32, "--max-steps", 5, "--no-dirty")
    r3 = json.loads((p3 / "result.json").read_text())
    assert r3["workflow"] == "p3" and [s["name"] for s in r3["stages"]] == ["sb", "all"]
    with pytest.raises(SystemExit):
        _call("fit", "--config", "F", "--workflow", "mpol", "--data", syn, "--out", tmp_path / "f")
    with pytest.raises(SystemExit):
        _call("fit", "--config", "R", "--workflow", "mpol", "--sb-only", "--data", syn, "--out", tmp_path / "g")


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_grad_table_states_physical_units_and_sigma_per_ln_s():
    row = dict(n=2, gz_max=1e4, gz_norm=1.5e4, dxdz_median=2500.0, gx_max=4.0)
    sig = dict(n=3, gz_max=3.0, gz_norm=4.0, dxdz_median=1.5, gx_max=2.0, gx_lns_max=4.0)
    md = "\n".join(script.grad_table({"x": {"cal.offset": row, "cal.sigma": sig}}))
    assert "| x | cal.offset | 2 | 1e+04 | 1.5e+04 | 2500 | 4 | per mas |" in md
    assert "| 4 | per ln s |" in md  # max |s g_x| over the ddids, stored by grad_groups


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_offset_bound_reaches_the_priors_and_default_is_virgils_own(syn):
    data = va.load_dir(syn)
    default = va.build_problem(data, "R", lam=0.5, npix=16)
    bounded = va.build_problem(data, "R", lam=0.5, npix=16, offset_bound_mas=100.0)
    assert bounded.offset_bound_mas == 100.0 and bounded.offset_bound_requested == 100.0
    assert default.offset_bound_requested is None and 0 < default.offset_bound_mas < default.nuis.offset_bound_mas
    d = default.priors["cal.offset"]  # virgil 0.4.1: ±3 beams of each observation; the reported bound is the smallest
    assert np.allclose(np.asarray(d.low), -np.asarray(d.high)) and default.offset_bound_mas == np.asarray(d.high).min()
    d = bounded.priors["cal.offset"]
    assert np.allclose(np.asarray(d.low), -100.0) and np.allclose(np.asarray(d.high), 100.0)
    for k in default.priors:  # nothing else changes
        if k != "cal.offset" and hasattr(default.priors[k], "low"):
            assert np.allclose(np.asarray(default.priors[k].low), np.asarray(bounded.priors[k].low))


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_report_records_the_bound_and_warns_when_an_offset_is_within_5_percent_of_it(syn, tmp_path, capsys):
    out = tmp_path / "ob"
    _call("fit", "--config", "R", "--workflow", "mpol", "--data", syn, "--out", out, "--npix", 16, "--max-steps", 3, "--no-dirty", "--offset-bound-mas", 100)
    r = json.loads((out / "result.json").read_text())
    assert r["args"]["offset_bound_mas"] == 100 and r["offset_bound"]["bound_mas"] == 100 and r["offset_bound"]["requested_mas"] == 100
    assert r["offset_bound"]["warning"] is None
    z = dict(np.load(out / "values.npz"))
    z["cal.offset"] = np.zeros_like(z["cal.offset"])
    z["cal.offset"][0, 1] = 97.0  # 3% from the bound
    np.savez(out / "values.npz", **z)
    capsys.readouterr()
    _call("report", "--data", syn, "--out", out, "--no-dirty")
    shown = capsys.readouterr()
    assert "OFFSET BOUND BINDING" in shown.out + shown.err
    r = json.loads((out / "result.json").read_text())  # the saved --offset-bound-mas is reused by report
    assert r["offset_bound"]["bound_mas"] == 100 and r["offset_bound"]["near_bound"] == [dict(obs=1, axis="dec", offset_mas=97.0)]
    assert va.offset_bound_report(np.full((2, 2), 94.0), 100.0)["warning"] is None
    assert va.offset_bound_report(np.full((2, 2), -95.0), 100.0)["warning"] is not None
