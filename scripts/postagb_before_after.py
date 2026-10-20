"""Before / after table of a post-AGB rerun: the strategies of every target in ``docs/imaging/postagb`` (as published) against a new results
directory laid out as ``NEW/<target>/<target>_strategies.json`` (what ``postagb_imaging.py select`` writes).

    python scripts/postagb_before_after.py NEW_DIR [OUT.md]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "docs/imaging/postagb"
TARGETS = ("IW_Car", "IRAS08544", "HR4049", "HD108015", "AI_Sco", "EN_TrA", "HD95767", "IRAS15469-5311", "PS_Gem")
STRATS = ("mem", "gp-iso", "gp-aniso", "gp-polar", "disc+res")


def num(x, fmt):
    return "–" if x is None else format(x, fmt)


def main(new_dir, out=None):
    lines = ["| target | strategy | NCC before → after | χ²/pt before → after | ΔlnZ before → after | f_env % before → after | grid edge before → after |", "|---|---|---|---|---|---|---|"]
    for t in TARGETS:
        old_f, new_f = OLD / f"{t}_strategies.json", Path(new_dir) / t / f"{t}_strategies.json"
        old = json.load(open(old_f))["strategies"] if old_f.exists() else {}
        new = json.load(open(new_f))["strategies"] if new_f.exists() else {}
        for s in STRATS:
            o, n = old.get(s), new.get(s)
            if o is None and n is None:
                continue
            g = lambda d, k, f: num(d.get(k), f) if d else "–"  # noqa: E731
            edge = lambda d: "–" if not d or d.get("gp_on_edge") is None else ("edge" if d["gp_on_edge"] else "interior")  # noqa: E731
            lines.append(f"| {t} | {s} | {g(o, 'ncc', '.2f')} → {g(n, 'ncc', '.2f')} | {g(o, 'chi2_per_point', '.2f')} → {g(n, 'chi2_per_point', '.2f')} | "
                         f"{g(o, 'dlogz', '.0f')} → {g(n, 'dlogz', '.0f')} | {g(o, 'f_env_pct', '.1f')} → {g(n, 'f_env_pct', '.1f')} | {edge(o)} → {edge(n)} |")
    text = "\n".join(lines)
    if out:
        Path(out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main(*sys.argv[1:3])
