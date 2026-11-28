"""WP-O smoke matrix: both codes, Level A matched and native (200 + 200, 4 chains) and Level B (small), one process each,
through ``common.timed_process``; validated rows go to ``--rows`` (default ``~/.cache/virgil-bench/runs/orbit/rows``),
never to results/raw/. Smoke rows are not headline numbers.

    ~/.claude/bin/pyslot $BENCH_PYTHON orbit/smoke.py        (run_in_background; PYSLOT_MEM=4G)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

RUNS = Path.home() / ".cache" / "virgil-bench" / "runs" / "orbit"
JULIA = HERE / "julia"

# (case, config, warmup, draws, chains, extra octofitter args)
PLAN = [
    ("O-A-sim", "matched", 200, 200, 4, []),
    ("O-A-sim", "native", 200, 200, 4, []),
    ("O-B-sim", "matched", 30, 30, 2, []),
    ("O-B-sim", "native", 30, 30, 1, ["--pigeons-rounds", "4"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(RUNS / "rows"))
    ap.add_argument("--only", default=None, help="substring of case_config to run, e.g. O-B-sim_native")
    a = ap.parse_args()
    py = sys.executable
    for case, config, warm, draws, chains, extra in PLAN:
        if a.only and a.only not in f"{case}_{config}":
            continue
        common_args = ["--case", case, "--config", config, "--warmup", warm, "--draws", draws, "--chains", chains,
                       "--no-warm"]
        for code, cmd in (
            ("virgil", [py, HERE / "run_virgil.py", *common_args, "--out", RUNS / f"{case}_{config}_virgil"]),
            ("octofitter", ["julia", "-t", "1", f"--project={JULIA}", HERE / "run_octofitter.jl", *common_args, *extra,
                            "--out", RUNS / f"{case}_{config}_octofitter"]),
        ):
            if code == "virgil" and "--pigeons-rounds" in extra:
                pass
            res = common.timed_process(cmd, threads=1, timeout=1800)
            row = common.row_from_process(res, case=case, comparison="orbit", code=code, config=config, threads=1,
                                          size="tiny")
            row["settings"]["smoke"] = True
            if res["status"] != "ok":
                row["settings"].setdefault("reason", "process failed; see stderr tail")
            p = common.write_row(row, a.rows)
            print(code, case, config, res["status"], f"t_process={res['t_process']:.1f}",
                  f"t_init={row['t_init']}", f"t_cold={row['t_cold']}", f"peak_mb={row['peak_footprint_mb']}", p.name,
                  flush=True)


if __name__ == "__main__":
    main()
