"""PMOIRED cross-check of the HD 45166 raw (uncalibrated) closure phases (design/hd45166_diagnosis.md).

    python scripts/pmoired_hd45166.py sci|cal DATA_DIR

Fits one unresolved companion to the T3PHI of the two science (``sci``) or two calibrator (``cal``) raw files
``GRAVI.*_single{sci,cal}vis_single{sci,cal}tf.fits`` in DATA_DIR, on the continuum of oidb_fit.HD45166_LINES, from
virgil's 3.5 mas minimum and from the published position (dE 4.83, dN 9.75 mas), then with that position fixed
(flux ratio free). Needs ``pmoired`` (the ``[external]`` extra)."""
import glob
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import pmoired  # noqa: E402

LINES = [(2.030, 2.045), (2.050, 2.066), (2.068, 2.090), (2.100, 2.120), (2.155, 2.195), (2.335, 2.355)]
BASE = {"*,f": 1.0, "*,ud": 0.0, "c,ud": 0.0}
NOT_FITTED = ["*,f", "*,ud", "c,ud"]


def main(which, data_dir):
    files = sorted(glob.glob(f"{data_dir}/*single{which}vis_single{which}tf.fits"))
    oi = pmoired.OI(files, insname="GRAVITY_SC", verbose=0)
    edges = [2.02] + [x for r in LINES for x in r] + [2.40]
    oi.setupFit({"obs": ["T3PHI"], "wl ranges": [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)]})

    def run(tag, x, y, f, fixed=False):
        oi.doFit(dict(BASE, **{"c,f": f, "c,x": x, "c,y": y}), doNotFit=NOT_FITTED + (["c,x", "c,y"] if fixed else []),
                 verbose=0)
        b = oi.bestfit["best"]
        print(f"{tag}: x={b['c,x']:.3f} y={b['c,y']:.3f} rho={np.hypot(b['c,x'], b['c,y']):.3f} "
              f"PA={np.degrees(np.arctan2(b['c,x'], b['c,y'])) % 360:.1f} f={b['c,f']:.3f} "
              f"chi2_red={oi.bestfit['chi2']:.3f}", flush=True)

    run("start at virgil minimum (-3.0, 1.9)", -3.0, 1.9, 0.4)
    run("start at the published position", 4.83, 9.75, 0.8)
    run("fixed at the published position", 4.83, 9.75, 0.8, fixed=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
