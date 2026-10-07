"""Track B summary: the registered pass criteria (design/trackb_criteria.md) over all systems.

Reads <dir>/<system>.json from scripts/trackb_fit.py and writes <dir>/summary.json:
per-epoch flags (d2 > 13.816), per-system chi2_{2N} tests, the overall KS test of the
counted d2 against chi2_2 (fails if the one-sided p for d2 too large is below 0.01),
epochs whose fitted error scale exceeds 3, counted epochs without a fit, and the criteria
block of every result (they must all name the same registered file).

The verdict is "incomplete" (not "pass") unless the run is whole: 20 systems with a
.json and a .done marker each, 121 counted epochs (design/trackb_criteria.md) and no
counted epoch without a fit.

  python scripts/trackb_summary.py <dir>
"""

import json
import pathlib
import sys

N_SYSTEMS = 20  # system.json files in the Track B data (19 contribute counted epochs)
N_COUNTED = 121  # counted epochs fixed by the registered criteria

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from crosscheck import astrometry as A  # noqa: E402


def main():
    d = pathlib.Path(sys.argv[1])
    results = [
        json.loads(p.read_text())
        for p in sorted(d.glob("*.json"))
        if p.name != "summary.json"
    ]
    crit = {json.dumps(r["criteria"], sort_keys=True) for r in results}
    if len(crit) != 1:
        raise SystemExit(f"results from different criteria versions: {crit}")
    counted, flags, scale_fail, unfitted, systems = [], [], [], [], {}
    for r in results:
        for e in r["epochs"]:
            tag = f"{r['system']} {e['date']}"
            if e.get("counted") and "d2" not in e:
                unfitted.append(dict(epoch=tag, error=e.get("error")))
            if e.get("scale_failure"):
                scale_fail.append(
                    dict(epoch=tag, scales=e.get("scales"), counted=e.get("counted"))
                )
            if e.get("counted") and "d2" in e:
                counted.append(e["d2"])
                if e["d2"] > A.D2_FLAG:
                    flags.append(
                        dict(
                            epoch=tag,
                            d2=e["d2"],
                            delta=e["delta"],
                            two_dloss_at_reference=(e.get("at_reference") or {}).get(
                                "two_dloss"
                            ),
                        )
                    )
        systems[r["system"]] = A.system_statistic(
            [e["d2"] for e in r["epochs"] if e.get("counted") and "d2" in e]
        )
    ks = A.overall_ks(counted)
    done = {p.stem for p in d.glob("*.done")}
    missing = sorted(
        {r["system"] for r in results} ^ done
    )  # a .json without .done, or the reverse
    incomplete = bool(
        unfitted or missing or ks["n"] != N_COUNTED or len(results) != N_SYSTEMS
    )
    out = dict(
        criteria=json.loads(crit.pop()),
        n_systems=len(results),
        overall=ks,
        epoch_flags=flags,
        system_flags=[k for k, v in systems.items() if v["flagged"]],
        systems_conservative=[k for k, v in systems.items() if v.get("conservative")],
        systems=systems,
        scale_failures=scale_fail,
        counted_without_fit=unfitted,
        systems_without_json_or_done=missing,
        verdict="incomplete" if incomplete else ("fail" if ks["fails"] else "pass"),
    )
    (d / "summary.json").write_text(json.dumps(out, indent=1))
    print(
        f"{len(results)} systems, {ks['n']} counted epochs: KS p(d2 too large) {ks['p_larger']}, two-sided "
        f"{ks['p_two_sided']}; {len(flags)} epoch flags (expected {ks.get('expected_flagged', 0):.2f}); "
        f"system flags {out['system_flags']}; {len(unfitted)} counted epochs without a fit; verdict {out['verdict']}"
    )


if __name__ == "__main__":
    main()
