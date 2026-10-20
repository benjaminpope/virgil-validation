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


@pytest.mark.validates("virgil_bridge.alma", "virgil.fitting.fit", roots=["mathematics"])
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
