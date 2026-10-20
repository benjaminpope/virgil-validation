"""Calibrate the GRAVITY data of the gallery binaries with virgil-vlti (design/orbit_noise_models.md
§3.1.1-3, virgil PR #436).

Per system: screen the downloaded calibrators (``calpca.screen_calibrators``), calibrate each
science night with ``virgil-vlti-calibrate`` (propagated transfer-function error, the shipped
template, HIGH binned by 8 and MEDIUM by 1 (by each file's resolution mode), 2.05-2.38 um; Brgamma is masked by the loader's
``ranges``, not here), and list the nights with no usable calibrator: those use closure phases
only. Calibrator diameters are the pipeline's own (``ESO QC TF CAL_SC DIAM``), so each file
gets its ``.modes.npz`` diameter modes.

Reads ``<root>/<slug>/oifits`` and ``<root>/<slug>/calibrators``; writes
``<root>/<slug>/calibrated/<RES>`` and ``<root>/<slug>/calibration_summary.json``. Usage:

    OMP_NUM_THREADS=2 python scripts/gravity_calibrate_gallery.py hd_9312 [--night 2020-12-09]
"""

import argparse
import csv
import glob
import json
import os
import time

from astropy.io import fits
from astropy.time import Time

from virgil_vlti import cli
from virgil_vlti.calpca import screen_calibrators
from virgil_vlti.products import read_exposure

ROOT = os.path.expanduser("~/data/eso_binaries/trackb")
INSNAME = "GRAVITY_SC"
WINDOW = ("2.05", "2.38")
# spectral-resolution mode -> (template, channels per bin)
SETUPS = {
    "HIGH": ("gravity_single_high_combined_2023-2025", 8),
    "MEDIUM": ("gravity_single_medium_combined_2023-2025", 1),
}


def night_of(exposure):
    return exposure.date[:10]


# Calibrated files whose V² is not usable (Ben, 2026-10-20: treat as closure-phase only), by the
# stem of the calibrated file name; checked on the 115 first-pass files (V² above 1.1 on over 5% of
# samples, or over 20% NaN or flagged).
V2_UNUSABLE = {
    "ll_aqr_20211108T011826": "8% of V² above 1.1 (max 1.56)",
    "hd188088_20210817T051919": "25% of V² NaN or flagged",
    "hr7578_20221027T010724": "22% of V² NaN or flagged",
    "hip_10258_20210907T080044": "35% of V² NaN or flagged",
    "ll_aqr_20211111T000644": "25% of V² NaN or flagged",
}


def resolution_of(path):
    return str(fits.getheader(path).get("ESO INS SPEC RES", "")).upper()


def manifest_ids(manifest, slug):
    return {c["dp_id"] for e in manifest[slug] for c in e["candidates"]}


def run(slug, root, manifest, night=None, out_name="calibrated", use_list=True):
    """Calibrate one system, one spectral-resolution mode at a time (a system can mix them)."""
    sci_all = sorted(glob.glob(f"{root}/{slug}/oifits/*.fits"))
    cal_all = [
        p
        for p in sorted(glob.glob(f"{root}/{slug}/calibrators/*.fits"))
        if os.path.basename(p)[: -len(".fits")] in manifest_ids(manifest, slug)
        and fits.getheader(p).get("ESO PRO CATG") == "SINGLE_CAL_VIS"
    ]
    summary = {"slug": slug, "modes": {}}
    for res in sorted({resolution_of(p) for p in sci_all}):
        sci = [p for p in sci_all if resolution_of(p) == res]
        if res not in SETUPS:
            summary["modes"][res] = {
                "unsupported": "no shipped template",
                "science": names(sci),
            }
            continue
        template, nbin = SETUPS[res]
        cal_exposures = [
            read_exposure(p, INSNAME) for p in cal_all if resolution_of(p) == res
        ]
        if (
            night
        ):  # the UT date of any science file, and everything within half a day of it
            first = [p for p in sci if read_exposure(p, INSNAME).date.startswith(night)]
            if not first:
                continue
            t = _mjd(read_exposure(first[0], INSNAME))
            sci = [p for p in sci if abs(_mjd(read_exposure(p, INSNAME)) - t) < 0.5]
            cal_exposures = [e for e in cal_exposures if abs(_mjd(e) - t) < 0.5]
        stamps = {
            read_exposure(p, INSNAME).date[:19].replace(":", "").replace("-", "")
            for p in sci
        }
        marked = (
            [k.split("_")[-1] for k in V2_UNUSABLE if k.split("_")[-1] in stamps]
            if use_list
            else []
        )
        t0 = time.time()
        try:
            _, excluded = screen_calibrators(
                cal_exposures, INSNAME, window_um=tuple(map(float, WINDOW)), field="SINGLE",
                allow_unrecorded=True,
            )  # fmt: skip
            screened = True
        except ValueError as err:
            if not str(err).startswith("No night has two calibrators"):
                raise  # anything else is a bug or a bad file, not "nothing to score"
            excluded, screened = [], f"UNSCREENED: {err}"
            print(f"WARNING {slug} {res}: calibrators unscreened ({err})", flush=True)
        bad = {e["target"] for e in excluded}
        kept = [e.path for e in cal_exposures if e.target not in bad]
        out = f"{root}/{slug}/{out_name}" + (f"_{night}" if night else "") + f"/{res}"
        info = {"template": template, "bin": nbin, "out": out, "excluded_calibrators": excluded,
                "screened": screened, "calibrator_files": {"read": len(cal_exposures), "kept": len(kept)},
                "screen_seconds": round(time.time() - t0)}  # fmt: skip
        if os.path.exists(f"{out}/index.csv"):
            os.remove(f"{out}/index.csv")  # never read a previous run's index
        try:
            cli.main(
                [*sci, *map(str, kept), "--out", out, "--tf-error", "propagated",
                 "--template", template, "--bin", str(nbin), "--window", *WINDOW,
                 "--allow-unrecorded", "--cp-only-when-uncalibrated",
                 *[x for n in marked for x in ("--cp-only", n)]]
            )  # fmt: skip
            with open(f"{out}/index.csv", newline="") as f:
                rows = list(csv.DictReader(f))
        except (
            SystemExit
        ) as err:  # e.g. no night had a calibrator, or a configuration mismatch
            info["error"] = str(err)
            rows = []
        # V² calibrated; closure phases calibrated with V² set aside (--cp-only); closure phases
        # only and uncalibrated (no calibrator that night).
        cp = [r for r in rows if r.get("cp_only") == "True"]
        info["v2_calibrated"] = [r["source"] for r in rows if r not in cp]
        info["cp_calibrated"] = [r["source"] for r in cp if int(r["n_cal_frames"])]
        info["cp_uncalibrated"] = [
            r["source"] for r in cp if not int(r["n_cal_frames"])
        ]
        info["v2_unusable"] = {
            r["source"]: V2_UNUSABLE[stem]
            for r in cp
            if (stem := os.path.splitext(os.path.basename(r["calibrated"]))[0])
            in V2_UNUSABLE
        }
        info["not_written"] = [
            n for n in names(sci) if n not in {r["source"] for r in rows}
        ]
        summary["modes"][res] = info
    with open(
        f"{root}/{slug}/{out_name.replace('calibrated', 'calibration_summary')}.json",
        "w",
    ) as f:
        json.dump(summary, f, indent=1, default=float)
    for res, info in summary["modes"].items():
        print(
            slug,
            res,
            {
                k: (len(v) if isinstance(v, list) else v)
                for k, v in info.items()
                if k != "excluded_calibrators"
            },
            flush=True,
        )


def names(paths):
    return [os.path.basename(p) for p in paths]


def _mjd(exposure):
    return Time(exposure.date[:19]).mjd


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("slugs", nargs="+")
    p.add_argument("--root", default=ROOT)
    p.add_argument(
        "--manifest",
        default=os.path.join(
            os.path.dirname(__file__), "data", "gravity_calibrators.json"
        ),
    )
    p.add_argument(
        "--out-name", default="calibrated", help="output folder under each system"
    )
    p.add_argument(
        "--ignore-v2-list", action="store_true", help="do not set V2_UNUSABLE aside"
    )
    p.add_argument("--night", help="YYYY-MM-DD: only that night (a test)")
    a = p.parse_args()
    with open(a.manifest) as f:
        manifest = json.load(f)
    for s in a.slugs:
        run(s, a.root, manifest, a.night, a.out_name, not a.ignore_v2_list)
