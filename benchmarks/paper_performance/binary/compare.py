"""Gate and summary for the binary comparison. Imports neither virgil nor PMOIRED.

    $BENCH_PYTHON binary/compare.py --gate [--out gate_results.json]    run both gate workers, apply cases.yml thresholds
    $BENCH_PYTHON binary/compare.py --rows DIR                           pair virgil / PMOIRED rows, print the table

Gate (plan section 4): both codes' model V2 and closure phase at the truth, on B-sim-6 and B-sim-ud, against the
noise-free file written by crosscheck (1e-8 in V2, 1e-6 deg in closure phase); chi2 at the truth against
``neutral.chi2_v2_cp`` (PMOIRED to 1e-6 relative; virgil's difference is recorded as ``likelihood_mismatch``, not gated).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _shared as sh  # noqa: E402

common = sh.common
G = sh.CASES["gate"]
V2_TOL = G["model_vs_crosscheck_noise_free"]["v2_abs_max"]
CP_TOL = G["model_vs_crosscheck_noise_free"]["cp_abs_max_deg"]
CHI2_TOL = G["chi2_at_truth_vs_neutral"]["pmoired_rel"]


def run_gate(out=None):
    res = {}
    for code in ("virgil", "pmoired"):
        script = sh.HERE / f"run_{code}.py"
        r = common.timed_process([sys.executable, str(script), "--worker", "--task", "gate", "--case", "B-sim-6",
                                  "--config", "matched", "--size", "full", "--threads", "1"], threads=1)
        if r["status"] != "ok":
            raise SystemExit(f"{code} gate worker {r['status']}:\n{r['stderr_tail']}")
        res[code] = r["inner"]["accuracy"]
    summary = {"thresholds": {"v2_abs": V2_TOL, "cp_abs_deg": CP_TOL, "pmoired_chi2_rel": CHI2_TOL}, "cases": {}}
    failed = []
    for case in ("B-sim-6", "B-sim-ud"):
        row = {}
        for code in ("virgil", "pmoired"):
            a = res[code][case]
            ok = a["v2_max_abs_diff"] <= V2_TOL and a["cp_max_abs_diff_deg"] <= CP_TOL
            row[code] = {**a, "model_gate_pass": bool(ok)}
            if not ok:
                failed.append(f"{code} {case} model")
        p = row["pmoired"]
        p["chi2_gate_pass"] = bool(abs(p["chi2_rel_diff"]) <= CHI2_TOL)
        if not p["chi2_gate_pass"]:
            failed.append(f"pmoired {case} chi2")
        summary["cases"][case] = row
    summary["failed"] = failed
    summary["status"] = "pass" if not failed else "gate_failed"
    text = json.dumps(summary, indent=1)
    if out:
        Path(out).write_text(text + "\n")
    print(text)
    return summary


def load_rows(d):
    return [json.loads(p.read_text()) for p in sorted(Path(d).glob("binary__*.json"))]


def summarise(rows):
    """Pair rows by (case, config, threads); warm and cold times, ratio other/virgil, accuracy differences."""
    by = {}
    for r in rows:
        by.setdefault((r["case"], r["config"], r["threads"]), {})[r["code"]] = r
    lines = ["case | config | thr | virgil cold/warm [s] | pmoired cold/warm [s] | ratio warm | ratio cold | max |dbest| [combined sigma]"]
    for (case, config, thr), d in sorted(by.items()):
        v, p = d.get("virgil"), d.get("pmoired")
        if not v or not p:
            continue
        wv, wp = common.warm_stats(v["t_warm"]), common.warm_stats(p["t_warm"])
        rw = common.ratio_with_spread(wp, wv)
        rc = None if None in (v["t_cold"], p["t_cold"]) or v["t_cold"] == 0 else p["t_cold"] / v["t_cold"]
        dmax = None
        av, ap = v["accuracy"].get("best"), p["accuracy"].get("best")
        if av and ap:
            sv, sp = v["accuracy"]["sigma"], p["accuracy"]["sigma"]
            dmax = max(abs(av[k] - ap[k]) / np.hypot(sv[k], sp[k]) for k in av)
        fmt = lambda x, n=3: "-" if x is None else f"{x:.{n}g}"  # noqa: E731
        lines.append(f"{case} | {config} | {thr} | {fmt(v['t_cold'])}/{fmt(wv['median'])} | "
                     f"{fmt(p['t_cold'])}/{fmt(wp['median'])} | {fmt(rw['ratio'])} | {fmt(rc)} | {fmt(dmax)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rows", default=None)
    a = ap.parse_args()
    if a.gate:
        s = run_gate(a.out)
        sys.exit(0 if s["status"] == "pass" else 1)
    if a.rows:
        print(summarise(load_rows(a.rows)))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
