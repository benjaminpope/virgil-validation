"""The speed comparison of virgil with MPoL on IM Lup (``crosscheck.alma_speed``, ``alma_imlup_maxent.py speed``;
design/alma_imlup_p3.md, "Built: speed comparison"). Everything is synthetic: a trajectory built from a known cost per step, a known
compile time, known restarts and a known constant between the fit's loss and the objective, so the numbers the functions return can be
checked against the construction. Laptop, numpy only except the CLI test (which needs the script's imports)."""

import json
import pathlib
import sys

import numpy as np
import pytest

from crosscheck import alma_speed as sp
from evidence.plugin import record

ROOT = pathlib.Path(__file__).resolve().parents[1]
C = -189_545_489.39  # the constant of the real fits: R = loss + C
RATE = 0.8  # s per step
COMPILE = 12.0  # extra seconds in the first chunk of a round
N = 42_421_131


def make_result(*, workflow="p3", init=None, rounds=((1000, 0.0), (800, 4.0)), sb_steps=400, loss0=8.2e7, drop=3e5, with_restarts=True, shift=0.0):
    """A fit result.json (a round that ends off a 200-step boundary gets ``run_fit_c8``'s closing record, whose ``elapsed`` runs from the start of the stage): an SB stage (``p3``) then an all-data stage in rounds of (steps, gap): each round reports every 200 steps with
    ``elapsed`` restarting at 0, the first chunk carrying ``COMPILE`` seconds; the loss falls as ``loss0 - drop (1 - exp(-steps/500))``."""
    stages = []
    if workflow == "p3" and not init:
        traj = [dict(steps=float(s), loss=1.0e6 - s, grad_norm=1.0, elapsed=COMPILE + RATE * s, round=1) for s in range(200, sb_steps + 1, 200)]
        stages.append(dict(name="sb", steps=sb_steps, wall_s=COMPILE + RATE * sb_steps + 1.0, stopped_because="max_steps", fit_converged=False, convergence=dict(converged=False),
                           loss_final=traj[-1]["loss"], trajectory=traj, restarts=[]))
    traj, total, restarts, before = [], 0, [], 0.0
    for r, (steps, gap) in enumerate(rounds, 1):
        for s in range(200, steps + 1, 200):
            traj.append(dict(steps=float(total + s), loss=loss0 - drop * (1 - np.exp(-(total + s) / 500.0)) + shift, grad_norm=1.0, elapsed=COMPILE + RATE * s, round=r))
        if steps % 200:  # the closing record: elapsed since the start of the stage, i.e. the earlier rounds' time plus this round's
            traj.append(dict(steps=float(total + steps), loss=loss0 - drop * (1 - np.exp(-(total + steps) / 500.0)) + shift, grad_norm=1.0, elapsed=before + COMPILE + RATE * steps, round=r))
        restarts.append(dict(steps=steps, wall_s=COMPILE + RATE * steps + gap, loss=traj[-1]["loss"]))
        before += COMPILE + RATE * steps + gap
        total += steps
    wall = sum(x["wall_s"] for x in restarts)
    stages.append(dict(name="all", steps=total, wall_s=wall, stopped_because="max_steps", fit_converged=False, convergence=dict(converged=False),
                       loss_final=traj[-1]["loss"], trajectory=traj, restarts=restarts if with_restarts else []))
    return dict(command="fit", config="R", workflow=workflow, args=dict(init=init, workflow=workflow), n=N, stages=stages, wall_s=sum(s["wall_s"] for s in stages) + 13.0,
                objective_R=traj[-1]["loss"] + C, chi2_raw=2.7 * 2 * N, stopped_because="max_steps")


def make_mpol(chk0=True, w0=7000.0, w1=14000.0, o0=-107_900_000.0, o1=-108_232_183.7):
    stages = [dict(stage=0, epochs=10, wall_s=w0, job="1"), dict(stage=1, epochs=10, wall_s=w1, job="2")]
    obj = dict(n=N, objective_R=o1, chi2_raw_total=2.7062 * 2 * N, dtype="float64")
    return sp.mpol_summary(stages, obj, dict(objective_R=o0, chi2_raw_total=3.0 * 2 * N) if chk0 else None)


# ------------------------------------------------------------------ the bookkeeping


@pytest.mark.validates("crosscheck.alma_speed", roots=["mathematics"], kind="reference")
def test_cumulative_time_adds_rounds_and_stages():
    """Round 2's ``elapsed`` restarts at 0: its records sit after round 1's whole duration (``restarts[0].wall_s``, which includes a 4 s gap), and the
    all-data stage after the SB stage's ``wall_s``. Without ``restarts`` (older result.json) the last ``elapsed`` of the earlier rounds is used."""
    res = make_result(rounds=((600, 3.0), (400, 0.0)))
    pts, recon = sp.fit_curve(res)
    sb_wall = res["stages"][0]["wall_s"]
    r1_wall = COMPILE + RATE * 600 + 3.0
    allp = [p for p in pts if p["stage"] == "all"]
    last_r1, first_r2 = [p for p in allp if p["round"] == 1][-1], [p for p in allp if p["round"] == 2][0]
    assert last_r1["t_s"] == pytest.approx(sb_wall + COMPILE + RATE * 600)
    assert first_r2["t_s"] == pytest.approx(sb_wall + r1_wall + COMPILE + RATE * 200)
    assert allp[-1]["steps"] == 400 + 1000  # SB's 400 steps come first: steps are cumulative over stages
    assert all(abs(r["diff_s"]) < 3.5 for r in recon if r["stage"] == "all")  # the 3 s gap is inside restarts, not inside elapsed
    pts_old, _ = sp.fit_curve(make_result(rounds=((600, 3.0), (400, 0.0)), with_restarts=False))
    r2_old = [p for p in pts_old if p["stage"] == "all" and p["round"] == 2][0]
    assert r2_old["t_s"] == pytest.approx(first_r2["t_s"] - 3.0)  # no restart record: the gap is lost, and the reconciliation shows it
    record("speed.cumulative_time_gap_s", first_r2["t_s"] - r2_old["t_s"])


@pytest.mark.validates("crosscheck.alma_speed", roots=["mathematics"], kind="reference")
def test_sb_stage_costs_time_but_is_not_on_the_curve():
    res = make_result()
    out = sp.build({"cold": res}, make_mpol())
    f = out["fits"]["cold"]
    assert f["stage_wall_s"] == pytest.approx(sum(s["wall_s"] for s in res["stages"]))
    assert f["startup_s"] == pytest.approx(13.0)
    assert min(p["t_s"] for p in f["curve"]) > res["stages"][0]["wall_s"]  # every curve point is after the SB stage
    assert len(f["curve"]) == 9  # all-data records only (5 + 4)


@pytest.mark.validates("crosscheck.alma_speed", roots=["mathematics"], kind="reference")
def test_cost_per_step_and_compile_estimate():
    """The median seconds per step is the construction's RATE; the compile estimate of each round is its first chunk minus a steady chunk, i.e. COMPILE."""
    c = sp.steady_cost(make_result(rounds=((1000, 0.0), (800, 4.0))))
    assert c["s_per_step_median"] == pytest.approx(RATE, rel=1e-9)
    assert c["n_chunks"] == (5 - 1) + (4 - 1)
    assert [x["round"] for x in c["compile"]] == [1, 2]
    assert [x["compile_s_est"] for x in c["compile"]] == pytest.approx([COMPILE, COMPILE])
    record("speed.cost_per_step_recovered", c["s_per_step_median"])


@pytest.mark.validates("crosscheck.alma_speed", roots=["mathematics"], kind="reference")
def test_time_to_target_first_record_and_interpolation():
    """A straight line of objective against time: the first record at or below the target is found, and the interpolated time is the exact crossing."""
    pts = [dict(t_s=100.0 * k, steps=200 * k, objective=-1000.0 * k) for k in range(1, 11)]
    got = sp.time_to(pts, -3450.0)
    assert got["reached"] and got["t_first_s"] == 400.0 and got["steps"] == 800
    assert got["t_interp_s"] == pytest.approx(345.0)
    miss = sp.time_to(pts, -1e6)
    assert not miss["reached"] and miss["best_objective"] == -10000.0
    assert sp.time_to(pts, -1000.0)["t_interp_s"] == 100.0  # reached at the first record: no earlier record to interpolate from


@pytest.mark.validates("crosscheck.alma_speed", roots=["mathematics"], kind="reference")
def test_loss_is_converted_with_the_constant_of_the_final_point():
    res = make_result()
    assert sp.constant_of(res) == pytest.approx(C)
    out = sp.build({"a": res}, make_mpol())
    assert out["fits"]["a"]["curve"][-1]["objective"] == pytest.approx(res["objective_R"])
    assert out["constant_check"]["ok"] is None  # one fit, no anchor: nothing to compare with, so not "consistent"
    assert "NOT CHECKED" in sp.markdown(out, sp.fairness_notes(out))
    assert sp.build({"a": res, "b": make_result(loss0=8.1e7)}, make_mpol())["constant_check"]["ok"] is True


@pytest.mark.validates("crosscheck.alma_speed", roots=["mathematics"], kind="control")
def test_a_different_constant_between_fits_or_an_anchor_is_flagged():
    """A fit whose loss is not R plus the same constant (a different term, a different data column) must fail the check, as must an anchor that disagrees."""
    a, b = make_result(), make_result(shift=250.0)
    b["objective_R"] = a["objective_R"]  # the same R at the end, but its loss is shifted: the constant differs by 250
    out = sp.build({"a": a, "b": b}, make_mpol())
    assert not out["constant_check"]["ok"] and out["constant_check"]["spread"] == pytest.approx(250.0)
    anchored = dict(make_result(), anchors=[dict(steps=400, objective=-1.0e8, loss=8.0e7 - 1.0e4)])
    assert not sp.build({"a": anchored}, make_mpol())["constant_check"]["ok"]
    ok = dict(make_result(), anchors=[dict(steps=400, objective=8.0e7 + C, loss=8.0e7)])
    assert sp.build({"a": ok}, make_mpol())["constant_check"]["ok"]


# ------------------------------------------------------------------ targets and the MPoL side


@pytest.mark.validates("crosscheck.alma_speed", roots=["mathematics"], kind="reference")
def test_targets_mpol_checkpoints_and_warm_starts():
    """MPoL reaches its own final target at the end of stage 1 and its stage-0 target at the end of stage 0; a target between them only at stage 1.
    A warm start gives no time to target; a cold start that falls through the final target gets a time, and the best known is the lowest final objective."""
    cold = make_result(drop=3.0e5)
    cold["objective_R"] = -108_300_000.0
    cold["stages"][-1]["loss_final"] = cold["objective_R"] - C  # a final point below MPoL's
    cold["stages"][-1]["trajectory"][-1]["loss"] = cold["objective_R"] - C
    warm = make_result(init="/x/R_mpol_init")
    out = sp.build({"cold": cold, "warm": warm}, make_mpol())
    rows = {r["target"]: r for r in out["targets"]}
    fin = rows["MPoL final (chk1)"]
    assert fin["mpol"]["reached"] and fin["mpol"]["t_s"] == 21000.0 and fin["mpol"]["where"] == "end of stage 1"
    assert fin["virgil"]["warm"]["reached"] is None
    assert fin["virgil"]["cold"]["reached"]  # the last record is below MPoL's final objective
    c0 = rows["MPoL end of stage 0 (chk0)"]
    assert c0["mpol"]["t_s"] == 7000.0 and c0["mpol"]["where"] == "end of stage 0"
    best = [r for r in out["targets"] if r["target"].startswith("best known")]
    assert len(best) == 1 and best[0]["objective"] == pytest.approx(-108_300_000.0) and "cold" in best[0]["target"]
    below = sp.build({"cold": cold}, make_mpol(), extra_targets={"far": -109e6})
    far = [r for r in below["targets"] if r["target"] == "far"][0]
    assert not far["mpol"]["reached"] and not far["virgil"]["cold"]["reached"]


@pytest.mark.validates("crosscheck.alma_speed", roots=["mathematics"], kind="reference")
def test_mpol_cost_per_epoch_and_stage_ratio():
    m = make_mpol()
    assert m["s_per_epoch"] == [700.0, 1400.0] and m["stage_ratio"] == pytest.approx(2.0)
    assert m["batches_per_epoch"] == pytest.approx(N / 40000) and m["total_wall_s"] == 21000.0
    assert m["final_raw_chi2_over_n"] == pytest.approx(2.7062) and m["chk0_raw_chi2_over_n"] == pytest.approx(3.0)
    out = sp.build({"a": make_result()}, m)
    pf = out["cost"]["per_fit"]["a"]
    assert pf["s_per_step"] == pytest.approx(RATE) and pf["evals_recorded"] is False
    lo, hi = pf["evals_per_step_est"]
    assert lo == pytest.approx(RATE / 0.53) and hi == pytest.approx(RATE / 0.44)
    assert pf["mpol_epoch_over_virgil_step"] == pytest.approx([700 / RATE, 1400 / RATE])


@pytest.mark.validates("crosscheck.alma_speed", roots=["self-consistency"], kind="guard")
def test_log_parsing_reads_host_and_gpu_and_attributes_by_owner():
    """A log is whose run its ``mpol=``/``virgil=`` prefix says; without a prefix it is MPoL's only if its job is in stage<N>.time.json, else it is
    reported as unattributed and never credited to virgil."""
    text = ("  NVIDIA A100-SXM4-80GB, 81920 MiB, 615.71.09\n  job=18913982 stage=0 host=gina1 scripts=9f526ad pins=\"x\" start=Wed\n"
            "  job=18913983 stage=1 host=gina2 scripts=9f526ad\n")
    got = sp.parse_logs({"a.txt": text, "b.txt": "nothing here"})
    assert [(h["job"], h["host"], h["stage"], h["gpu"], h["owner"]) for h in got[:2]] == [
        ("18913982", "gina1", "0", "NVIDIA A100-SXM4-80GB", "unattributed"), ("18913983", "gina2", "1", "NVIDIA A100-SXM4-80GB", "unattributed")]
    assert got[2]["host"] is None and got[2]["gpu"] is None
    # stage<N>.time.json without a job key (jobs [None, None]): MPoL's own log must not become virgil's hardware
    nojob = sp.mpol_summary([dict(stage=0, epochs=10, wall_s=7000.0), dict(stage=1, epochs=10, wall_s=14000.0)], dict(n=N, objective_R=-1.0, chi2_raw_total=2.7 * 2 * N))
    notes = sp.fairness_notes(sp.build({"a": make_result()}, nojob, hardware=got))
    assert notes[0].startswith("Hardware, MPoL: not recorded") and notes[1].startswith("Hardware, virgil: not recorded")
    assert any(n.startswith("Hardware, unattributed") and "gina1" in n for n in notes)
    # explicit owners, and matching by job id for an unprefixed log
    own = sp.parse_logs({"m.txt": text, "v.txt": "job=7 task=3 host=gina9\n"}, {"m.txt": "mpol", "v.txt": "virgil"})
    notes = sp.fairness_notes(sp.build({"a": make_result()}, nojob, hardware=own))
    assert "job 18913982 (stage 0) on gina1" in notes[0] and notes[1] == "Hardware, virgil: job 7 on gina9, GPU model not in the log."
    notes = sp.fairness_notes(sp.build({"a": make_result()}, make_mpol(), hardware=sp.parse_logs({"x": "job=1 stage=0 host=gina3\n"})))
    assert "job 1 (stage 0) on gina3" in notes[0]  # job 1 is stage 0's job in make_mpol


@pytest.mark.validates("crosscheck.alma_speed", roots=["mathematics"], kind="reference")
def test_closing_record_of_a_restarted_round_does_not_count_earlier_rounds_twice():
    """``run_fit_c8`` adds a closing record whose ``elapsed`` runs from the start of the stage. A round of 850 steps (off the 200-step grid) in round 2
    ends with one: its time must be the SB stage + round 1's whole duration + round 2's own time, with or without ``restarts``, and the stage's
    wall_s must reconcile with the rounds (the old double count gave a difference of minus round 1's duration)."""
    rounds = ((1000, 3.0), (850, 0.0))
    for with_restarts in (True, False):
        res = make_result(rounds=rounds, with_restarts=with_restarts)
        pts, recon = sp.fit_curve(res)
        sb_wall = res["stages"][0]["wall_s"]
        r1 = COMPILE + RATE * 1000 + (3.0 if with_restarts else 0.0)
        last = [p for p in pts if p["stage"] == "all"][-1]
        assert last["steps"] == 400 + 1850
        # without ``restarts`` the 3 s gap between rounds is unknown, and the closing record's elapsed (which holds it) is then 3 s too late
        assert last["t_s"] == pytest.approx(sb_wall + r1 + COMPILE + RATE * 850, abs=1e-6 if with_restarts else 3.0 + 1e-6)
        rec = [r for r in recon if r["stage"] == "all"][0]
        assert rec["closing_records_corrected"] == 1
        assert abs(rec["diff_s"]) < (0.5 if with_restarts else 3.5)
    raw_cumulative = COMPILE + RATE * 1000 + 3.0 + COMPILE + RATE * 850  # what the closing record's elapsed holds
    assert [p for p in make_result(rounds=rounds)["stages"][-1]["trajectory"] if p["steps"] == 1850][0]["elapsed"] == pytest.approx(raw_cumulative)
    c = sp.steady_cost(make_result(rounds=rounds))
    assert c["s_per_step_median"] == pytest.approx(RATE)  # the closing record is not a chunk
    record("speed.closing_record_diff_s", rec["diff_s"])


@pytest.mark.validates("crosscheck.alma_speed", roots=["mathematics"], kind="reference")
def test_mpol_startup_is_subtracted_from_every_derived_time():
    """With a start-up of 60 s per stage the per-epoch times, the total wall, the checkpoint times (hence the time to target) all drop by it, together."""
    a, b = make_mpol(), sp.mpol_summary([dict(stage=0, epochs=10, wall_s=7000.0, job="1"), dict(stage=1, epochs=10, wall_s=14000.0, job="2")],
                                       dict(n=N, objective_R=-108_232_183.7, chi2_raw_total=2.7062 * 2 * N), dict(objective_R=-107_900_000.0, chi2_raw_total=3.0 * 2 * N), startup_s=60.0)
    assert b["s_per_epoch"] == pytest.approx([694.0, 1394.0]) and b["total_wall_s"] == 20880.0
    assert [p["t_s"] for p in b["points"]] == [6940.0, 20880.0] and [p["t_s"] for p in a["points"]] == [7000.0, 21000.0]
    assert b["s_per_batch"][0] == pytest.approx(694.0 / b["batches_per_epoch"])


@pytest.mark.validates("crosscheck.alma_speed", roots=["self-consistency"], kind="guard")
def test_fairness_notes_flag_the_stage_ratio_and_the_missing_chk0():
    m = sp.mpol_summary([dict(stage=0, epochs=10, wall_s=6983.0, job="18913982"), dict(stage=1, epochs=10, wall_s=14745.0, job="18913983")],
                        dict(n=N, objective_R=-108_232_183.7, chi2_raw_total=2.7062 * 2 * N))
    hw = [dict(source="s", owner="mpol", job="18913982", host="gina1", stage="0", gpu="NVIDIA A100-SXM4-80GB"), dict(source="s", owner="mpol", job="18913983", host="gina1", stage="1", gpu="NVIDIA A100-SXM4-80GB")]
    res = sp.build({"a": make_result()}, m, hardware=hw)
    notes = sp.fairness_notes(res)
    assert any("factor 2.11" in n and "same node (gina1)" in n for n in notes)
    md = sp.markdown(res, notes)
    assert "objective_chk0.json" in md  # the missing stage-0 evaluation is called out
    assert md.index("Raw chi^2/N") < md.index("## Definitions")  # raw chi^2/N comes first


# ------------------------------------------------------------------ the command


@pytest.mark.validates("crosscheck.alma_speed", roots=["self-consistency"], kind="guard")
def test_speed_command_writes_json_markdown_and_png(tmp_path):
    """Plumbing only: result.json files and an MPoL run directory in, speed.json, speed.md and speed.png out; the numbers are the construction's."""
    pytest.importorskip("virgil.radio")
    sys.path.insert(0, str(ROOT / "scripts"))
    import alma_imlup_maxent as script

    for name, kw in (("cold", {}), ("warm", dict(init="/x"))):
        d = tmp_path / name
        d.mkdir()
        (d / "result.json").write_text(json.dumps(make_result(**kw)))
    run = tmp_path / "mpol"
    (run / "export").mkdir(parents=True)
    for s, w in ((0, 7000.0), (1, 14000.0)):
        (run / f"stage{s}.time.json").write_text(json.dumps(dict(stage=s, epochs=10, wall_s=w, job=str(s + 1))))
    (run / "export" / "objective.json").write_text(json.dumps(dict(n=N, objective_R=-108_232_183.7, chi2_raw_total=2.7062 * 2 * N)))
    (run / "export" / "objective_chk0.json").write_text(json.dumps(dict(n=N, objective_R=-107_900_000.0, chi2_raw_total=3.0 * 2 * N)))
    log = tmp_path / "log.txt"
    log.write_text("NVIDIA A100-SXM4-80GB, 81920 MiB\njob=1 stage=0 host=gina1 start=now\njob=2 stage=1 host=gina1 start=now\n")
    out = tmp_path / "out"
    script.main(["speed", "--fit", f"cold_p3={tmp_path / 'cold'}", "--fit", str(tmp_path / "warm"), "--mpol", str(run), "--out", str(out), "--log", f"mpol={log}", "--target", "mine=-108000000"])
    res = json.loads((out / "speed.json").read_text())
    assert set(res["fits"]) == {"cold_p3", "warm"} and res["fits"]["cold_p3"]["start"] == "p3" and res["fits"]["warm"]["start"] == "warm"
    assert res["constant_check"]["ok"] and [r["target"] for r in res["targets"]][-1] == "mine"
    assert (out / "speed.png").stat().st_size > 5000 and "## Fairness notes" in (out / "speed.md").read_text()
    assert res["mpol"]["s_per_epoch"] == [700.0, 1400.0] and any(h["host"] == "gina1" for h in res["hardware"])
