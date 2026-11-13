"""Stage 1 of the M87* replication (design/m87_replication.md): ring features of Lai et al.'s
published 225 μas reconstructions by three extractors, against their Table 1.

    .venv/bin/python scripts/m87_stage1.py [--their]     # writes trust/campaigns/m87_stage1.json

(X) crosscheck.ring_extraction; (V) virgil.metrics.ring_features (virgil_bridge.m87);
(T) their RingFit through external_bridge/m87_worker.py in .venv-basis, with --their (needs
`scripts/setup_m87.sh basis`). Inputs from $M87_DATA (default ~/data/m87_eht; `setup_m87.sh data`).
Criteria and thresholds are those of the plan, by its last commit; the run refuses an edited plan.
"""

import argparse
import glob
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from crosscheck.preregistration import criteria_commit  # noqa: E402
from crosscheck.ring_extraction import ring_features  # noqa: E402

PLAN = "design/m87_replication.md"
DATA = pathlib.Path(os.environ.get("M87_DATA", pathlib.Path.home() / "data" / "m87_eht"))
PS = 225.0 / 64  # μas per pixel of the published samples
DAYS = {"2017-04-11": "M87_2017_101", "2018-04-21": "M87_2018_111", "2021-04-18": "M87_2021_108"}
# Table 1 of Lai et al. 2026 (mean ± SD over three fields of view).
TABLE1 = {
    "2017-04-11": {"diameter": (43.3, 1.0), "width": (16, 3), "pa": (167, 6), "asymmetry": (0.20, 0.01), "central_fraction": (0.037, 0.028)},
    "2018-04-21": {"diameter": (42.7, 0.5), "width": (11, 1), "pa": (205, 10), "asymmetry": (0.31, 0.01), "central_fraction": (0.004, 0.004)},
    "2021-04-18": {"diameter": (39.9, 1.8), "width": (12, 1), "pa": (197, 4), "asymmetry": (0.26, 0.01), "central_fraction": (0.004, 0.002)},
}
# Their RingFit on the 2021-04-18 median, as printed in GenDIReCT's runGenDIReCT.ipynb (archive outputs).
NOTEBOOK_2021 = {"diameter": 2 * 20.33, "width": 7.98, "pa": 203.12, "asymmetry": 0.3047}


def dangle(a, b):
    return abs((a - b + 180) % 360 - 180)


def samples(day):
    (path,) = glob.glob(str(DATA / "zenodo/gendirect/x/*/results" / f"{DAYS[day]}_225_sample.npy"))
    return np.load(path, allow_pickle=False), path


def their(images):
    from external_bridge import _subprocess

    (path,) = glob.glob(str(DATA / "zenodo/gendirect/x/*/data/ringFitting.py"))
    out = _subprocess.run("basis", "m87_worker.py", {"task": "ring_features", "images": [im.tolist() for im in images],
                                                      "psize_uas": PS, "ringfitting_path": path})
    res = []
    for r in out["results"]:
        v = lambda x: x[0] if isinstance(x, list) else x
        res.append({"diameter": 2 * v(r["mean_rpk_uas"]), "width": v(r["ring_width_uas"]),
                    "pa": float(np.degrees(v(r["ring_orientation_rad"])) % 360),
                    "asymmetry": v(r["azimuthal_asymmetry"]), "central_fraction": v(r["frac_central_brightness"]),
                    "raw": r})
    return res, out.get("ehtim_version")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--their", action="store_true", help="also run their RingFit in .venv-basis")
    ap.add_argument("--out", default=str(ROOT / "trust/campaigns/m87_stage1.json"))
    args = ap.parse_args()
    commit = criteria_commit(PLAN)  # refuses an uncommitted plan before any measurement

    from virgil_bridge import m87 as vb

    summary = {"criteria_file": PLAN, "criteria_commit": commit, "pixel_scale_uas": PS, "days": {},
               **(vb.provenance() if vb.has_ring_features() else {"virgil_commit": None, "virgil_dirty": None})}
    medians = {}
    for day in DAYS:
        s, path = samples(day)
        med = np.median(s, axis=0)
        medians[day] = med
        x = ring_features(med, PS)
        per = [ring_features(im, PS) for im in s]
        d = {"input": os.path.relpath(path, DATA), "n_samples": len(s), "X": x,
             "X_samples": {k: [float(np.mean([p[k] for p in per])), float(np.std([p[k] for p in per]))]
                           for k in ("diameter", "width", "asymmetry", "central_fraction")},
             "table1": TABLE1[day]}
        pas = np.radians([p["pa"] for p in per])
        d["X_samples"]["pa"] = [float(np.degrees(np.angle(np.mean(np.exp(1j * pas)))) % 360),
                                float(np.degrees(np.sqrt(-2 * np.log(abs(np.mean(np.exp(1j * pas)))))))]
        if vb.has_ring_features():
            d["V"] = vb.ring_features(med, PS)
        t1 = TABLE1[day]
        d["1a"] = {"diameter": abs(x["diameter"] - t1["diameter"][0]) <= max(t1["diameter"][1], 1.0),
                   "pa": dangle(x["pa"], t1["pa"][0]) <= max(t1["pa"][1], 5.0)}
        if "V" in d:
            v = d["V"]
            d["1c"] = {"diameter": abs(v["diameter"] - x["diameter"]) <= 0.5 * PS,
                       "pa": dangle(v["pa"], x["pa"]) <= 2.0,
                       "asymmetry": abs(v["asymmetry"] - x["asymmetry"]) <= 0.02}
        summary["days"][day] = d
    summary["days"]["2021-04-18"]["T_notebook"] = NOTEBOOK_2021
    if args.their:
        res, ver = their([medians[d] for d in DAYS])
        summary["ehtim_version"] = ver
        for day, t in zip(DAYS, res):
            x = summary["days"][day]["X"]
            summary["days"][day]["T"] = t
            summary["days"][day]["1b"] = {"diameter": abs(t["diameter"] - x["diameter"]) <= 0.5 * PS,
                                          "pa": dangle(t["pa"], x["pa"]) <= 2.0,
                                          "width": abs(t["width"] - x["width"]) <= PS,
                                          "asymmetry": abs(t["asymmetry"] - x["asymmetry"]) <= 0.02}
    summary["vv_commit"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                          text=True).stdout.strip()
    pathlib.Path(args.out).write_text(json.dumps(summary, indent=1, default=float) + "\n")
    for day, d in summary["days"].items():
        print(day, "X d=%.2f w=%.2f pa=%.1f A=%.3f fc=%.3f" % tuple(d["X"][k] for k in
              ("diameter", "width", "pa", "asymmetry", "central_fraction")),
              "| V d=%.2f pa=%.1f A=%.3f" % tuple(d["V"][k] for k in ("diameter", "pa", "asymmetry")) if "V" in d else "",
              "| 1a", d["1a"], "| 1c", d.get("1c"), "| 1b", d.get("1b"))


if __name__ == "__main__":
    main()
