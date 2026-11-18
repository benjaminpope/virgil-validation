"""Serial measured matrix (WP-R): the row list, its priority order, the time estimate and the executor.

    BENCH_PYTHON matrix.py --dry-run             list every row with its estimate and the 7 h cap; runs nothing
    BENCH_PYTHON matrix.py --run                 execute (use ``run_all.sh``, which adds caffeinate and the machine checks)

One process at a time. Every measured process goes through ``common.timed_process(wrapper="pyslot")`` (binary and
imaging runners do this themselves with ``--wrapper pyslot``; orbit rows are launched here). Within a case the codes
alternate (virgil, other, virgil, other, ...). Rows are written to ``results/raw``. Processes per row: 3, or 1 when one
call exceeds 10 min (``common.repeat_plan``; the first process decides). Rows still unrun when the 7 h budget ends get a
``not_run`` row with reason ``budget``. A row is rerun once, keeping both, only if flagged ``noisy`` or ``threads_not_held``;
never because of a value. Estimates are rough (see the comparison READMEs); they order the work, they decide nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

RAW = HERE / "results" / "raw"
LOG = HERE / "results" / "matrix_log.jsonl"
BUDGET_S = 7 * 3600
PY = os.environ.get("BENCH_PYTHON", sys.executable)
JULIA_PROJECT = HERE / "orbit" / "julia"
ORBIT_RUNS = Path.home() / ".cache" / "virgil-bench" / "runs" / "orbit" / "measured"
RESUME = False
TEST = {"tiny": False, "no_wrapper": False, "out": None}   # test-only switches (--tiny, --no-wrapper, --out-dir); never used for a measured run

B_CASES = ["B-sim-6", "B-sim-60", "B-sim-600", "B-sim-ud", "B-real-pionier"]
B_T3 = ["B-sim-6", "B-sim-60", "B-real-pionier"]
I_FACTOR = {"I-sim-24": 1.0, "I-sim-48": 2.5, "I-sim-96": 6.0, "I-sim-ring": 2.5, "I-sim-large": 15.0}
# ladder pass (all 8 budgets, 6375 steps) seconds at 1 thread, from imaging/README.md
I_PASS = {"I-sim-24": (11, 50), "I-sim-48": (30, 130), "I-sim-96": (70, 280), "I-sim-ring": (30, 130),
          "I-sim-large": (115, 2100)}


def n_calls(call_s):
    p = common.repeat_plan(call_s)
    return p["processes"], p["processes"] * p["warm"]


def cost(call_s, startup_s=10.0):
    procs, calls = n_calls(call_s)
    return procs * startup_s + calls * call_s


@dataclass
class Spec:
    prio: int
    comp: str
    case: str
    code: str            # virgil | other | virgil2 (imaging default-method row)
    label: str           # task/config
    est_s: float
    call_s: float
    threads: int = 1
    argv: list = field(default_factory=list)
    runner: str = "self"           # "self": the runner writes its own rows; "orbit": launched here; "untimed": gate step
    not_run_reason: str | None = None
    cmeta: dict = field(default_factory=dict)
    def __post_init__(self):
        if self.runner not in ("untimed", "not_run"):
            self.est_s = 60.0 * est_min(self)


    @property
    def key(self):
        return f"{self.comp}|{self.case}|{self.label}|{self.code}|t{self.threads}"

    @property
    def other_name(self):
        return {"binary": "pmoired", "imaging": "ehtim", "orbit": "octofitter"}[self.comp]


# ----------------------------------------------------------------------------------------------------------------
# row list
# ----------------------------------------------------------------------------------------------------------------

def b_args(code, task, case, config="matched", threads=1):
    script = HERE / "binary" / ("run_virgil.py" if code == "virgil" else "run_pmoired.py")
    return [PY, str(script), "--task", task, "--case", case, "--config", config, "--threads", str(threads),
            "--wrapper", "pyslot"]


def i_args(code, case, task, threads=1, method=None):
    script = HERE / "imaging" / ("run_virgil.py" if code in ("virgil", "virgil2") else "run_ehtim.py")
    a = [PY, str(script), "run", "--case", case, "--task", task, "--threads", str(threads), "--wrapper", "pyslot"]
    if method:
        a += ["--method", method]
    return a


def o_cmd(code, case, config, threads, out, pigeons_rounds=None):
    common_args = ["--case", case, "--config", config, "--out", str(out)]
    if code == "virgil":
        return [PY, str(HERE / "orbit" / "run_virgil.py"), *common_args, "--threads", str(threads)]
    extra = ["--pigeons-rounds", str(pigeons_rounds)] if pigeons_rounds else []
    return ["julia", "-t", str(threads), f"--project={JULIA_PROJECT}", str(HERE / "orbit" / "run_octofitter.jl"),
            *common_args, *extra]


# Estimated minutes per row (all processes of the row, 1 thread unless the row says 2). Anchored on the block totals
# the three WPs wrote in their READMEs (binary 1.5 h, imaging 3-3.5 h, orbit 90-120 min), spread over the rows; they
# order the work and size the 7 h cap, nothing more. Where a README block total could be per case rather than in
# all (PMOIRED T3), the dry run uses the README total in all; the digest states this.
LADDER_MIN = {"I-sim-24": 13.8, "I-sim-48": 36.4, "I-sim-96": 48.3, "I-sim-ring": 36.4, "I-sim-large": 0.0}


def est_min(sp):
    c, t = sp.comp, sp.label
    thr = 0.5 if sp.threads == 2 and c == "binary" and t.startswith("T3") else 1.0
    if c == "binary":
        if t.startswith("T1") or t.startswith("T2"):
            return 0.5
        if t.startswith("pulls"):
            return 2.5
        if sp.code == "virgil":
            return 1.5 * thr
        return {"T3/matched": 11.0, "T3/matched-coarse": 25.0, "T3/native": 27.0}[t] * thr
    if c == "orbit":
        big = {"O-B-sim": 8.0, "O-A-sim-pulls": 15.0}.get(sp.case)
        if big:
            if sp.case == "O-B-sim" and t == "native" and sp.code != "virgil":
                return 15.0
            return big
        return 3.0
    # imaging
    case = sp.case
    f = I_FACTOR[case]
    if case == "I-sim-large":
        return {"ladder/lbfgsb": 5, "ladder/matched": 40, "ladder/default-method": 5, "per_eval": 0.5 if sp.code == "virgil" else 2,
                "stop_default": 1 if sp.code == "virgil" else 4, "native": 1 if sp.code == "virgil" else 3}[t]
    if t.startswith("ladder"):
        m = LADDER_MIN[case]
        return m * (0.71 if sp.code == "other" else 0.145)
    if t == "per_eval":
        return 0.5
    return 0.18 * f * (3 if sp.code == "other" else 1)


def add_pair(out, prio, comp, case, label, est_v, est_o, call_v, call_o, mk, threads=1, reason_o=None):
    """virgil then the other code (alternation inside a case)."""
    out.append(Spec(prio, comp, case, "virgil", label, est_v, call_v, threads, mk("virgil")))
    out.append(Spec(prio, comp, case, "other", label, est_o, call_o, threads, mk("other"), not_run_reason=reason_o))


def build():
    s: list[Spec] = []
    # ---- priority 1: gates (untimed) ----
    s.append(Spec(1, "binary", "B-sim-6,B-sim-ud", "both", "gate", 90, 0, 1,
                  [PY, str(HERE / "binary" / "compare.py"), "--gate"], runner="untimed"))
    s.append(Spec(1, "imaging", "all", "both", "gate", 300, 0, 1,
                  ["bash", "-c", " && ".join(
                      f"{PY} {HERE}/imaging/run_{c}.py gate --case {k}" for k in I_FACTOR for c in ("virgil", "ehtim"))
                   + f" && {PY} {HERE}/imaging/compare.py gate-report"], runner="untimed"))
    s.append(Spec(1, "orbit", "all", "both", "gate", 300, 0, 1,
                  ["bash", "-c", f"cd {HERE}/orbit && {PY} gates.py prepare && julia -t 1 --project=julia gate_octofitter.jl "
                   f"&& {PY} gates.py virgil && {PY} gates.py compare"], runner="untimed"))

    # ---- binary building blocks ----
    def b_t2(prio, cfg, cases):
        for c in cases:
            add_pair(s, prio, "binary", c, f"T2/{cfg}", 30, 30, 1, 1,
                     lambda code, c=c, cfg=cfg: b_args(code, "T2", c, cfg))

    def b_t1(prio):
        for c in B_CASES:
            add_pair(s, prio, "binary", c, "T1/matched", 30, 30, 0.1, 0.1, lambda code, c=c: b_args(code, "T1", c))

    def b_t3(prio, cfg, threads, only=B_T3):
        for c in only:
            v_call = 60 if cfg != "matched-coarse" else 15
            o_call = {"matched": 660, "matched-coarse": 160, "native": 180}[cfg] * (1 if threads == 1 else 0.5)
            add_pair(s, prio, "binary", c, f"T3/{cfg}", cost(v_call), cost(o_call), v_call, o_call,
                     lambda code, c=c, cfg=cfg, t=threads: b_args(code, "T3", c, cfg, t), threads)

    # ---- priority 2 ----
    b_t2(2, "matched", B_CASES)
    for c in ("O-A-sim",):
        add_pair(s, 2, "orbit", c, "matched", 270 * 3 / 2, 270 * 3 / 2, 130, 140,
                 lambda code, c=c: [code, c, "matched", 1, None], reason_o=None)
    for c in ("I-sim-24", "I-sim-48"):
        iladder(s, 2, c)
    # ---- priority 3 ----
    b_t1(3)
    b_t3(3, "matched", 1)
    add_pair(s, 3, "orbit", "O-A-real", "matched", 450, 450, 150, 150, lambda code: [code, "O-A-real", "matched", 1, None])
    add_pair(s, 3, "orbit", "O-A-sim", "native", 450, 450, 150, 150, lambda code: [code, "O-A-sim", "native", 1, None])
    iladder(s, 3, "I-sim-96")
    for c in ("I-sim-24", "I-sim-48", "I-sim-96", "I-sim-ring"):
        f = I_FACTOR[c]
        add_pair(s, 3, "imaging", c, "per_eval", 60, 60, 1, 1, lambda code, c=c: i_args(code, c, "per_eval"))
        add_pair(s, 3, "imaging", c, "stop_default", 100 * f / 2, 150 * f, 10, 15 * f,
                 lambda code, c=c: i_args(code, c, "stop_default"))
        add_pair(s, 3, "imaging", c, "native", 200 * f / 2, 200 * f, 20, 20 * f, lambda code, c=c: i_args(code, c, "native"))
    b_t3(3, "native", 1)
    # ---- priority 4 ----
    add_pair(s, 4, "orbit", "O-B-sim", "matched", 1800, 1800, 600, 600, lambda code: [code, "O-B-sim", "matched", 1, None])
    add_pair(s, 4, "orbit", "O-B-sim", "native", 1800, 3000, 600, 1800, lambda code: [code, "O-B-sim", "native", 1, 8])
    for c in ("B-pulls-tri", "B-pulls-base"):
        add_pair(s, 4, "binary", c, "pulls/matched", 150, 150, 150, 150, lambda code, c=c: b_args(code, "pulls", c))
    add_pair(s, 4, "orbit", "O-A-sim-pulls", "matched", 1200, 1200, 1200, 1200,
             lambda code: [code, "O-A-sim-pulls", "matched", 1, None])
    # ---- priority 5: 2-thread pass ----
    b_t3(5, "matched", 2)
    b_t3(5, "native", 2)
    for c in ("O-A-sim", "O-A-real"):
        add_pair(s, 5, "orbit", c, "native", 450, 450, 150, 150, lambda code, c=c: [code, c, "native", 2, None], 2)
    add_pair(s, 5, "orbit", "O-B-sim", "native", 1800, 3000, 600, 1800, lambda code: [code, "O-B-sim", "native", 2, 8], 2)
    for c in ("I-sim-24", "I-sim-48", "I-sim-96", "I-sim-ring"):
        f = I_FACTOR[c]
        add_pair(s, 5, "imaging", c, "native", 200 * f / 2, 200 * f, 20, 20 * f,
                 lambda code, c=c: i_args(code, c, "native", 2), 2)
    # ---- priority 6 ----
    b_t2(6, "tight", B_CASES)
    b_t3(6, "matched-coarse", 1)
    iladder(s, 6, "I-sim-ring")
    for c in ("I-sim-24", "I-sim-48", "I-sim-96", "I-sim-ring"):
        f = I_FACTOR[c]
        add_pair(s, 6, "imaging", c, "stop_tight", 200 * f / 2, 500 * f, 40, 60 * f,
                 lambda code, c=c: i_args(code, c, "stop_tight"))
    add_pair(s, 6, "orbit", "O-A-real", "native", 450, 450, 150, 150, lambda code: [code, "O-A-real", "native", 1, None])
    iladder(s, 6, "I-sim-large")
    c = "I-sim-large"
    f = I_FACTOR[c]
    add_pair(s, 6, "imaging", c, "per_eval", 60, 120, 1, 2, lambda code: i_args(code, c, "per_eval"))
    add_pair(s, 6, "imaging", c, "stop_default", 150, 600, 15, 150, lambda code: i_args(code, c, "stop_default"))
    add_pair(s, 6, "imaging", c, "native", 300, 600, 30, 200, lambda code: i_args(code, c, "native"))
    for cc in ("O-A-sim", "O-B-sim"):
        s.append(Spec(6, "orbit", cc, "both", "matched-truthstart", 0, 0, 1, [], runner="not_run",
                      not_run_reason="no documented route for user-supplied starting values in Octofitter's initialize! "
                                     "was recorded by WP-O; both codes use their own initialisation (plan section 5)"))
    # attach orbit command builders
    for sp in s:
        if sp.comp == "orbit" and sp.runner == "self" and sp.argv and not isinstance(sp.argv[0], str):
            pass
    return s


def iladder(s, prio, case):
    pv, po = I_PASS[case]
    cv, co = n_calls(pv)[1] * pv, n_calls(po)[1] * po
    s.append(Spec(prio, "imaging", case, "virgil", "ladder/lbfgsb", cv, pv, 1, i_args("virgil", case, "ladder")))
    s.append(Spec(prio, "imaging", case, "other", "ladder/matched", co, po, 1, i_args("ehtim", case, "ladder")))
    s.append(Spec(prio, "imaging", case, "virgil2", "ladder/default-method", cv, pv, 1,
                  i_args("virgil", case, "ladder", method="default")))


# ----------------------------------------------------------------------------------------------------------------
# orbit specs hold a callable (code, case, config, threads, rounds); resolve to a command at run time
# ----------------------------------------------------------------------------------------------------------------

def resolve_orbit(sp):
    code, case, config, threads, rounds = sp.argv
    out = ORBIT_RUNS / f"{case}_{config}_t{threads}_{code}"
    return o_cmd("virgil" if code == "virgil" else "octofitter", case, config, threads, out, rounds), out


def finalise(specs):
    """Orbit specs were created with a placeholder argv (code, case, config, threads, rounds); mark their runner."""
    for sp in specs:
        if sp.comp == "orbit" and sp.runner == "self" and len(sp.argv) == 5:
            sp.runner = "orbit"
            sp.argv = [("virgil" if sp.code == "virgil" else "octofitter"), *sp.argv[1:]]
    return specs


def ordered(specs):
    """Stable priority order; within a priority keep the construction order (codes alternate inside a case)."""
    return sorted(specs, key=lambda x: x.prio)


# ----------------------------------------------------------------------------------------------------------------
# dry run
# ----------------------------------------------------------------------------------------------------------------

def total_est(sp):
    return 0.0 if sp.runner == "not_run" else sp.est_s


def dry_run(specs):
    print(f"{'#':>3} {'P':>1} {'comparison':<8} {'case':<16} {'task/config':<22} {'code':<8} {'thr':>3} "
          f"{'est min':>8} {'cum h':>6}")
    cum, n, per_prio, cut = 0.0, 0, {}, None
    for i, sp in enumerate(specs, 1):
        e = total_est(sp)
        cum += e
        if sp.runner != "not_run":
            n += 1
        per_prio[sp.prio] = per_prio.get(sp.prio, 0.0) + e
        mark = ""
        if cut is None and cum > BUDGET_S:
            cut = i
            mark = "  <-- 7 h cap reached here"
        note = f"  [not_run: {sp.not_run_reason[:40]}...]" if sp.runner == "not_run" else ""
        print(f"{i:>3} {sp.prio:>1} {sp.comp:<8} {sp.case:<16} {sp.label:<22} {sp.code:<8} {sp.threads:>3} "
              f"{e / 60:>8.1f} {cum / 3600:>6.2f}{mark}{note}")
    print()
    for p in sorted(per_prio):
        print(f"priority {p}: {per_prio[p] / 3600:.2f} h")
    runnable = [x for x in specs if x.runner != "not_run"]
    timed = [x for x in runnable if x.runner != "untimed"]
    print(f"\nrows listed: {len(specs)} ({len(runnable)} to run: {len(timed)} timed + "
          f"{len(runnable) - len(timed)} untimed gate steps; {len(specs) - len(runnable)} predeclared not_run)")
    print(f"total estimate: {cum / 3600:.2f} h (cap {BUDGET_S / 3600:.0f} h); first row past the cap: "
          f"{'none' if cut is None else cut}")
    if cut:
        dropped = [x for x in specs[cut - 1:] if x.runner not in ("not_run",)]
        print(f"rows past the cap at the estimate (would be not_run, reason budget): {len(dropped)}")


# ----------------------------------------------------------------------------------------------------------------
# executor
# ----------------------------------------------------------------------------------------------------------------

def newest_rows(before):
    after = set(RAW.glob("*.json")) if RAW.exists() else set()
    return sorted(after - before)


def run_one(sp):
    """One process of a spec. Returns the list of row paths written."""
    before = set(RAW.glob("*.json")) if RAW.exists() else set()
    if sp.runner == "untimed":
        r = subprocess.run(sp.argv, capture_output=True, text=True)
        (RAW.parent / "gates").mkdir(parents=True, exist_ok=True)
        (RAW.parent / "gates" / f"{sp.comp}.log").write_text(r.stdout + "\n" + r.stderr)
        return []
    if sp.runner == "orbit":
        code, case, config, threads, rounds = sp.argv
        cmd = o_cmd(code, case, config, threads, ORBIT_RUNS / f"{case}_{config}_t{threads}_{code}",
                         rounds if code == "octofitter" else None)
        res = common.timed_process(cmd, threads=threads, timeout=1800, wrapper=None if TEST["no_wrapper"] else "pyslot")
        row = common.row_from_process(res, case=case, comparison="orbit", code=code, config=config, threads=threads,
                                      size="full")
        row["settings"].update({"task": "orbit-" + config, "case_id": case})
        common.write_row(row, RAW)
    else:
        argv = list(sp.argv)
        if TEST["no_wrapper"] and "--wrapper" in argv:
            i = argv.index("--wrapper")
            del argv[i:i + 2]
        if TEST["tiny"]:
            argv += ["--size", "tiny"]
        if TEST["out"]:
            argv += ["--out-dir" if sp.comp == "imaging" else "--out", str(TEST["out"])]
        r = subprocess.run(argv, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stderr[-400:], file=sys.stderr)
    return [str(p) for p in newest_rows(before)]


def write_not_run(sp, reason):
    for code in (["virgil", sp.other_name] if sp.code == "both" else [sp.code if sp.code == "virgil" else sp.other_name]):
        row = common.new_row(case=sp.case, comparison=sp.comp, code=code, config=sp.label.split("/")[-1],
                             threads=sp.threads, status="not_run", settings={"reason": reason, "task": sp.label})
        common.write_row(row, RAW)


def execute(specs):
    done_keys, prev_h, prev_rows = set(), 0.0, {}
    if RESUME and LOG.exists():     # continue after a crash: skip specs already logged, keep the elapsed time
        for l in LOG.read_text().splitlines():
            d = json.loads(l)
            if "proc" in d:
                prev_rows.setdefault(d["key"], []).extend(d["rows"])
            if "proc" in d or d.get("status") == "not_run":
                done_keys.add(d["key"])
            prev_h = max(prev_h, d.get("elapsed_h", 0.0))
    t0 = time.time() - prev_h * 3600
    RAW.mkdir(parents=True, exist_ok=True)
    done_marker = RAW.parent / "DONE"
    if TEST["tiny"] or TEST["no_wrapper"]:
        (RAW.parent).mkdir(parents=True, exist_ok=True)
        (RAW.parent / "TEST_RUN_BYPASSES").write_text(f"test-only run (tiny={TEST['tiny']}, no_wrapper={TEST['no_wrapper']}); idle, AC and pyslot checks were NOT applied; rows are not measurements\n")
    done_marker.unlink(missing_ok=True)
    log = LOG.open("a")
    produced = {}
    # blocks of consecutive specs of one (comparison, case, task/config, threads): processes alternate between the
    # codes of a block (virgil, other, virgil, other, ...) so drift affects both alike
    blocks, cur = [], []
    for sp in specs:
        k = (sp.comp, sp.case, sp.label.split("/")[0], sp.threads, sp.prio)
        if cur and (k != cur[0][0] or sp.runner in ("untimed", "not_run")):
            blocks.append([x for _, x in cur]); cur = []
        cur.append((k, sp))
        if sp.runner in ("untimed", "not_run"):
            blocks.append([x for _, x in cur]); cur = []
    if cur:
        blocks.append([x for _, x in cur])
    for blk in blocks:
        n_proc = {}
        for sp in blk:
            produced[sp.key] = []
            n_proc[sp.key] = 0 if sp.key in done_keys else 3
            if sp.key in done_keys:
                produced[sp.key] = prev_rows.get(sp.key, [])
            if sp.runner == "not_run":
                write_not_run(sp, sp.not_run_reason)
        for i in range(3):
            for sp in blk:
                if sp.runner == "not_run" or i >= n_proc[sp.key]:
                    continue
                if time.time() - t0 > BUDGET_S:
                    if i == 0:
                        if sp.runner != "untimed":
                            write_not_run(sp, "budget")
                        log.write(json.dumps({"key": sp.key, "status": "not_run", "reason": "budget"}) + "\n")
                    n_proc[sp.key] = 0
                    continue
                ts = time.time()
                got = run_one(sp)
                produced[sp.key] += got
                log.write(json.dumps({"key": sp.key, "proc": i, "wall_s": time.time() - ts, "rows": got,
                                      "elapsed_h": (time.time() - t0) / 3600}) + "\n")
                log.flush()
                if sp.runner == "untimed" or sp.code == "both":
                    n_proc[sp.key] = 0
                elif i == 0 and got:
                    r = json.loads(Path(got[0]).read_text())
                    call = r.get("t_cold") or 0
                    if r.get("t_warm") and "pulls" not in sp.label:
                        call = max(call, sorted(r["t_warm"])[len(r["t_warm"]) // 2])
                    n_proc[sp.key] = common.repeat_plan(call)["processes"]
                    if r.get("status") in ("timeout", "oom", "error", "gate_failed") or "pulls" in sp.label:
                        n_proc[sp.key] = 1   # a failed row is not repeated; a pull loop is timed once as a whole
    # one rerun, only for noisy / threads_not_held rows
    rows = {p: json.loads(Path(p).read_text()) for ps in produced.values() for p in ps}
    for p, r in rows.items():
        r["language"] = "julia" if r["code"] == "octofitter" else "python"
    common.flag_noisy(list(rows.values()))
    for p, r in rows.items():             # persist the noisy flag (t_calib vs the session median per language)
        Path(p).write_text(json.dumps(r, indent=1, default=float) + "\n")
    for sp in specs:
        if time.time() - t0 > BUDGET_S:
            break
        ps = produced.get(sp.key, [])
        if any(("noisy" in rows[p]["flags"] or "threads_not_held" in rows[p]["flags"]) for p in ps):
            got = run_one(sp)
            log.write(json.dumps({"key": sp.key, "rerun": True, "rows": got}) + "\n")
    done_marker.write_text(f"finished after {(time.time() - t0) / 3600:.2f} h\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue a crashed run: skip specs already in the log")
    ap.add_argument("--only", nargs="*", default=None, help="TEST ONLY: run just the specs whose key contains one of these")
    ap.add_argument("--tiny", action="store_true", help="TEST ONLY: --size tiny for binary/imaging runners")
    ap.add_argument("--no-wrapper", action="store_true", help="TEST ONLY: no pyslot wrapper (run inside pyquick)")
    ap.add_argument("--out-dir", default=None, help="TEST ONLY: rows/log/marker go here instead of results/")
    a = ap.parse_args()
    global RAW, LOG, RESUME
    if a.out_dir:
        RAW, LOG = Path(a.out_dir) / "raw", Path(a.out_dir) / "matrix_log.jsonl"
        TEST["out"] = RAW
    TEST.update(tiny=a.tiny, no_wrapper=a.no_wrapper)
    specs = ordered(finalise(build()))
    if a.resume:
        RESUME = True
    if a.only:
        specs = [x for x in specs if any(o in x.key for o in a.only)]
    if a.dry_run:
        dry_run(specs)
    elif a.run:
        execute(specs)
    else:
        ap.error("--dry-run or --run")


if __name__ == "__main__":
    main()
