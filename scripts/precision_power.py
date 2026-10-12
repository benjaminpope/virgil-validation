"""The power calculations behind the precision targets, and the precision pull v1 achieved.

    python scripts/precision_power.py power          # all registered designs
    python scripts/precision_power.py pull-achieved  # from trust/campaigns/pull_campaign.json

Registered in design/coldstart_pull_precision.yml, design/pull_campaign_precision.yml and
design/sbc_v2_precision.yml (rationale: design/precision_targets.md). Formulas:
src/crosscheck/precision.py. tests/test_precision.py checks the registered numbers
against this module's functions.
"""

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck import precision as P  # noqa: E402

SBC_L = 99  # draws per replicate: ranks 0..99
# Each claim is a conjunction ("every parameter of every cell is calibrated to the margin"), so the
# equivalence tests form an intersection-union test: the claim is made when ALL reject non-equivalence,
# each at the family level, with no correction (Berger 1982). The power of the claim is the product of
# the tests' powers (they use different quantities or cells; treated as independent).
COLD_ALPHA = 0.01   # per cell, one-sided exact upper limit
PULL_ALPHA = 0.05   # per side, per pull sd and mean
SBC_ALPHA = 0.05    # per side, per coverage and mean rank


def coldstart_table(n_list=(1000, 1450, 1625), alpha=COLD_ALPHA):
    """Power that the failure-rate upper limit is below the rule-out rate when the true rate is the bound."""
    out = {}
    for cell, (bound, rule_out) in COLD.items():
        out[cell] = {n: {"max_failures": P.max_failures_below(n, rule_out, alpha),
                         "power": P.power_upper_below(n, rule_out, bound, alpha)} for n in n_list}
    return out


COLD = {"binary_bright": (0.01, 0.025), "binary_faint": (0.05, 0.075), "binary_wide": (0.05, 0.075),
        "binary_sparse": (0.05, 0.075), "resolved_companion": (0.02, 0.04), "envelope": (0.02, 0.04)}


def pull_power(n, mean_margin=0.1, sd_interval=(0.9, 1.1), alpha=PULL_ALPHA):
    a = alpha
    return {"mean": P.power_mean_equivalence(n, mean_margin, 0.0, a),
            "sd": P.power_sd_equivalence(n, *sd_interval, 1.0, a)}


def sbc_acceptance(level):
    """The rank set of the central `level`% interval and its nominal probability (scripts/sbc_numpyro.py's definition)."""
    ranks = np.arange(SBC_L + 1)
    accepted = np.abs(ranks - SBC_L / 2) < level / 100 * (SBC_L + 1) / 2
    return int(accepted.sum()), float(accepted.mean())


def sbc_power(n, alpha=SBC_ALPHA, margins=None):
    margins = margins or {"coverage_68": 0.06, "coverage_95": 0.03, "mean_rank": 0.04}
    a = alpha
    out = {}
    for level in (68, 95):
        _, p = sbc_acceptance(level)
        out[f"coverage_{level}"] = P.power_ci_inside(n, p, margins[f"coverage_{level}"], p, a)
    var = ((SBC_L + 1) ** 2 - 1) / 12.0 / SBC_L**2  # normalised ranks, discrete uniform on 0..L
    out["mean_rank"] = P.power_mean_equivalence(n, margins["mean_rank"], 0.0, a, sd=float(np.sqrt(var)))
    return out


def width_equivalents():
    """Coverage of the central 68% and 95% intervals of a Gaussian posterior whose width is f times the true width."""
    from scipy import stats

    return {f: {level: float(2 * stats.norm.cdf(stats.norm.isf((1 - level / 100) / 2) * f) - 1) for level in (68, 95)}
            for f in (0.85, 0.9, 1.1, 1.15)}


def pull_achieved(path=ROOT / "trust" / "campaigns" / "pull_campaign.json", alpha=PULL_ALPHA):
    d = json.loads(pathlib.Path(path).read_text())
    a = alpha
    rows, worst_mean, worst_sd = [], 0.0, 0.0
    for cell, e in d["cells"].items():
        n = e["finite"]
        for name, p in e["params"].items():
            sd_i = P.sd_ci(p["sd"], p["kurtosis"], n, a)
            mean_i = P.mean_ci(p["mean"], p["sd"], n, a)
            need_mean = max(abs(mean_i[0]), abs(mean_i[1]))
            need_sd = max(sd_i[1] - 1.0, 1.0 - sd_i[0])
            worst_mean, worst_sd = max(worst_mean, need_mean), max(worst_sd, need_sd)
            rows.append({"cell": cell, "param": name, "n": n, "mean": p["mean"], "mean_ci": mean_i, "sd": p["sd"], "sd_ci": sd_i,
                         "mean_inside_0.1": P.inside(mean_i, -0.1, 0.1), "sd_inside_0.9_1.1": P.inside(sd_i, 0.9, 1.1)})
    return rows, {"smallest_mean_margin_certified": worst_mean, "smallest_sd_margin_certified": worst_sd}


def joint(*powers):
    return float(np.prod(powers))


def cold_joint(n):
    return joint(*(P.power_upper_below(n, ro, b, COLD_ALPHA) for b, ro in COLD.values()))


def pull_joint(n, tests_per_kind=20, **kw):
    pw = pull_power(n, **kw)
    return joint(*([pw["mean"]] * tests_per_kind + [pw["sd"]] * tests_per_kind))


def sbc_joint(n, regimes=2, quantities=6):
    pw = sbc_power(n)
    return joint(*([v for v in pw.values()] * (regimes * quantities)))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "power"
    if cmd == "power":
        print(f"cold-start failure rate: one-sided exact upper limit (alpha {COLD_ALPHA}) <= rule-out, true rate = bound")
        for cell, t in coldstart_table().items():
            print(f"  {cell:20s} bound {COLD[cell][0]} rule-out {COLD[cell][1]}: "
                  + "; ".join(f"N={n}: k<={v['max_failures']}, power {v['power']:.3f}" for n, v in t.items()))
        print("  all six cells at once: " + ", ".join(f"N={n}: {cold_joint(n):.3f}" for n in (1000, 1450, 1500, 1625)))
        print(f"pull campaign (alpha {PULL_ALPHA} per side): sd in [0.9,1.1], |mean| < 0.1; joint = all 40 tests")
        for n in (1000, 1250, 1500, 1625, 1750, 1875, 2000):
            print(f"  N={n}: {pull_power(n)} joint {pull_joint(n):.3f}")
        print("cold-start pull flag (mean margin 0.15, 40 tests):")
        for n in (450, 940, 1300):
            print(f"  N={n}: {pull_power(n, mean_margin=0.15)}")
        print("N for per-test power 0.8, pull mean 0.1:", P.n_for_power(lambda n: pull_power(n)["mean"], 0.8, 100, 5000, 10),
              "; sd:", P.n_for_power(lambda n: pull_power(n)["sd"], 0.8, 100, 5000, 10),
              "; cold-start flag mean 0.15:", P.n_for_power(lambda n: pull_power(n, 0.15)["mean"], 0.8, 100, 5000, 10))
        print(f"SBC v2 (alpha {SBC_ALPHA} per side; 36 tests = 2 regimes x 6 quantities x 3): coverage_68 +-0.06, coverage_95 +-0.03, mean rank +-0.04")
        for n in (1000, 1024, 1250, 1270, 1280):
            print(f"  N={n}: " + ", ".join(f"{k} {v:.3f}" for k, v in sbc_power(n).items()) + f"; all 36: {sbc_joint(n):.3f}")
        print("  nominal rank-set coverages (size, probability):", {lv: sbc_acceptance(lv) for lv in (68, 95)})
        print("  Gaussian posterior width f x true -> coverage:", {f: {k: round(v, 4) for k, v in c.items()}
                                                                    for f, c in width_equivalents().items()})
    elif cmd == "pull-achieved":
        rows, worst = pull_achieved()
        for r in rows:
            print(f"{r['cell']:28s}{r['param']:14s} n={r['n']} mean {r['mean']:+.3f} [{r['mean_ci'][0]:+.3f},{r['mean_ci'][1]:+.3f}]"
                  f" in0.1={r['mean_inside_0.1']}  sd {r['sd']:.3f} [{r['sd_ci'][0]:.3f},{r['sd_ci'][1]:.3f}] in={r['sd_inside_0.9_1.1']}")
        print(worst, "mean inside: %d/%d" % (sum(r["mean_inside_0.1"] for r in rows), len(rows)),
              "sd inside: %d/%d" % (sum(r["sd_inside_0.9_1.1"] for r in rows), len(rows)))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
