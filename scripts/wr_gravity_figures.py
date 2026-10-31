"""Figures for docs/wr_gravity/index.md from the real1 catalogue of the WR GRAVITY reanalysis.

Writes docs/wr_gravity/delta_chi2.png (best-companion delta-chi2 per epoch against the frozen 83.1 rule and the 51.1
sensitivity value, coloured by whether the fit is good) and docs/wr_gravity/flux_vs_chi2.png (physical flux against reduced
chi-squared for the epochs that pass the rule).

    python scripts/wr_gravity_figures.py [--catalogue ~/data/wr_gravity/products/real1/catalogue.csv]
"""

import argparse
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "wr_gravity"
CATALOGUE = pathlib.Path("~/data/wr_gravity/products/real1/catalogue.csv").expanduser()
RULE, SENSITIVITY = 83.1, 51.1
CHI2_CUT, FLUX_CUT = 2.98, 0.037
FLOOR = 1.0  # where delta-chi2 <= 0 is drawn on the log axis
GOOD, BAD = plt.cm.viridis(0.15), plt.cm.viridis(0.85)


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("dchi2_max", "f_phys", "chi2red_H1"):
            r[k] = float(r[k])
        for k in ("detect_rule", "candidate", "good_fit"):
            r[k] = r[k] == "True"
    return rows


def delta_chi2(rows, path):
    rows = sorted(rows, key=lambda r: (int(r["wr"].rstrip("ab")), r["wr"], int(r["night"])))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for good, colour, label in ((True, GOOD, "good fit"), (False, BAD, "poor fit or halo at bound")):
        # a best companion that does not improve on the null has dchi2_max = 0, which a log axis would drop
        sel = [(i, max(r["dchi2_max"], FLOOR)) for i, r in enumerate(rows) if r["good_fit"] == good]
        ax.scatter([i for i, _ in sel], [y for _, y in sel], color=colour, s=28, label=label, zorder=3)
    ax.axhline(RULE, color="k", ls="--", lw=1, label=f"frozen rule ({RULE})")
    ax.axhline(SENSITIVITY, color="0.4", ls=":", lw=1, label=f"sensitivity ({SENSITIVITY})")
    # one tick label per star, at the middle of its epochs
    stars = {}
    for i, r in enumerate(rows):
        stars.setdefault(r["wr"], []).append(i)
    ax.set_xticks([sum(v) / len(v) for v in stars.values()])
    ax.set_xticklabels([f"WR {k}" for k in stars], rotation=90, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel(r"best-companion $\Delta\chi^2$")
    ax.set_xlabel("epochs, grouped by star")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def flux_vs_chi2(rows, path):
    rows = [r for r in rows if r["detect_rule"]]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for cand, marker in ((True, "o"), (False, "x")):
        sel = [r for r in rows if r["candidate"] == cand]
        ax.scatter([r["chi2red_H1"] for r in sel], [r["f_phys"] for r in sel], marker=marker, s=40,
                   c=[GOOD if r["good_fit"] else BAD for r in sel], label="candidate" if cand else "not a candidate",
                   zorder=3)
    for r in rows:
        if r["candidate"]:
            ax.annotate(f"WR {r['wr']}", (r["chi2red_H1"], r["f_phys"]), textcoords="offset points", xytext=(5, 4),
                        fontsize=7)
    ax.axvline(CHI2_CUT, color="k", ls=":", lw=1)
    ax.axhline(FLUX_CUT, color="k", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"reduced $\chi^2$ (companion model)")
    ax.set_ylabel("physical flux ratio")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--catalogue", type=pathlib.Path, default=CATALOGUE)
    a = ap.parse_args()
    rows = load(a.catalogue.expanduser())
    OUT.mkdir(parents=True, exist_ok=True)
    delta_chi2(rows, OUT / "delta_chi2.png")
    flux_vs_chi2(rows, OUT / "flux_vs_chi2.png")
