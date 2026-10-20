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
import glob
import json
import os
import time

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


def resolution_of(path):
    from astropy.io import fits

    return str(fits.getheader(path).get("ESO INS SPEC RES", "")).upper()


def run(slug, root, night=None):
    """Calibrate one system, one spectral-resolution mode at a time (a system can mix them)."""
    sci_all = sorted(glob.glob(f"{root}/{slug}/oifits/*.fits"))
    cal_all = sorted(glob.glob(f"{root}/{slug}/calibrators/*.fits"))
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
        if night:
            sci = [p for p in sci if read_exposure(p, INSNAME).date.startswith(night)]
            t = _mjd(read_exposure(sci[0], INSNAME))
            cal_exposures = [e for e in cal_exposures if abs(_mjd(e) - t) < 0.5]
        t0 = time.time()
        try:
            _, excluded = screen_calibrators(
                cal_exposures, INSNAME, window_um=tuple(map(float, WINDOW)), field="SINGLE",
                allow_unrecorded=True,
            )  # fmt: skip
            screened = True
        except (
            ValueError
        ) as err:  # no night with two calibrator stars: nothing to score
            excluded, screened = [], str(err)
        bad = {e["target"] for e in excluded}
        kept = [e.path for e in cal_exposures if e.target not in bad]
        out = f"{root}/{slug}/calibrated" + (f"_{night}" if night else "") + f"/{res}"
        info = {"template": template, "bin": nbin, "out": out, "excluded_calibrators": excluded,
                "screened": screened, "calibrator_files": {"read": len(cal_exposures), "kept": len(kept)},
                "screen_seconds": round(time.time() - t0)}  # fmt: skip
        try:
            cli.main(
                [*sci, *map(str, kept), "--out", out, "--tf-error", "propagated",
                 "--template", template, "--bin", str(nbin), "--window", *WINDOW,
                 "--allow-unrecorded"]
            )  # fmt: skip
            rows = open(f"{out}/index.csv").read().splitlines()[1:]
            done = {r.split(",")[2] for r in rows}
        except (
            SystemExit
        ) as err:  # e.g. no night had a calibrator, or a configuration mismatch
            info["error"] = str(err)
            done = set()
        info["calibrated"] = sorted(done)
        info["closure_phase_only"] = [n for n in names(sci) if n not in done]
        summary["modes"][res] = info
    with open(f"{root}/{slug}/calibration_summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    for res, info in summary["modes"].items():
        print(
            slug, res, {k: v for k, v in info.items() if k != "calibrated"}, flush=True
        )


def names(paths):
    return [os.path.basename(p) for p in paths]


def _mjd(exposure):
    from astropy.time import Time

    return Time(exposure.date[:19]).mjd


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("slugs", nargs="+")
    p.add_argument("--root", default=ROOT)
    p.add_argument("--night", help="YYYY-MM-DD: only that night (a test)")
    a = p.parse_args()
    for s in a.slugs:
        run(s, a.root, a.night)
