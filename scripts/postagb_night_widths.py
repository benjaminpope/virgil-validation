"""Night-level calibration widths for ``postagb_imaging.py --noise night`` (design/gp_polar_hyperparameters.md §9).

Writes ``postagb_night_widths.json``: per target, the width (1σ) of a gain on log|V| shared by every frame of one night on one
baseline, and of a closure-phase offset shared by every frame of one night on one triangle.

The widths come from the data's residuals against the *published* model (SPARCO + ORGANIC, the ``pub_v2`` / ``pub_t3`` that the
``compare`` stage stores), never from a virgil image, so they do not depend on the hyperparameter picks they are meant to test.
A two-level method-of-moments ANOVA separates what a night shares from what a frame has on its own:

1. Each (frame, baseline) group gives one value: the mean over its channels of the relative V² residual (V² - V²_pub) / V²_pub,
   which is 2 × the log|V| gain. Closure phases: the mean over channels of (φ - φ_pub), wrapped, per (frame, triangle).
2. These are grouped by (night, baseline) or (night, triangle) with the frames as replicates. The between-cell component is
   τ² = var(cell means) - mean(within-cell variance / n_frames), from cells with at least two frames. Per-frame gains and white
   noise are in the within-cell variance, so they are not counted again.

Whatever the published model gets wrong at (u,v) shared by one night also lands in τ, so these widths are upper estimates; a
negative τ² gives zero. Run once, locally, on the stored compare outputs:

    python scripts/postagb_night_widths.py <results>/<vv>+<virgil>
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import postagb_imaging as P  # noqa: E402

OUT = Path(__file__).with_name("postagb_night_widths.json")


def between_cell_width(values, frames, cells):
    """τ of a two-level ANOVA: ``values`` per sample, averaged first per ``frames`` group, then compared across ``cells``."""
    keys = np.unique(np.stack([cells, frames], 1), axis=0)
    frame_mean = np.array([values[(cells == c) & (frames == f)].mean() for c, f in keys])
    means, within = [], []
    for c in np.unique(keys[:, 0]):
        x = frame_mean[keys[:, 0] == c]
        if x.size >= 2:
            means.append(x.mean())
            within.append(x.var(ddof=1) / x.size)
    if len(means) < 2:
        return 0.0, len(means)
    tau2 = np.var(means, ddof=1) - np.mean(within)
    return float(np.sqrt(max(tau2, 0.0))), len(means)


def target_widths(target, results):
    cfg = P.TARGETS[target]
    data, _ = P.load(cfg, None)
    lab = P.night_labels(data)  # data.vis and data.phi hold the unflagged samples, as do pub_v2 and pub_t3
    z = np.load(results / target / f"{target}_gp-polar_compare_images.npz")
    rel = (np.asarray(data.vis) - z["pub_v2"]) / z["pub_v2"]
    tau_v2, n_v = between_cell_width(rel, lab["v2_frame"], lab["v2_cell"])
    dphi = (np.degrees(np.asarray(data.phi)) - z["pub_t3"] + 180.0) % 360.0 - 180.0
    tau_t3, n_t = between_cell_width(dphi, lab["t3_frame"], lab["t3_cell"])
    return dict(night_gain_baseline=0.5 * tau_v2, night_t3_offset_deg=tau_t3, n_cells_v2=n_v, n_cells_t3=n_t,
                nights=int(len(np.unique(lab["v2_night"]))))


if __name__ == "__main__":
    results = Path(sys.argv[1])
    rows = {}
    for t in sorted(P.TARGETS):
        if (results / t / f"{t}_gp-polar_compare_images.npz").exists():
            rows[P.TARGETS[t].get("data_key", P.TARGETS[t]["key"])] = target_widths(t, results)
            print(t, rows[P.TARGETS[t].get("data_key", P.TARGETS[t]["key"])], flush=True)
    json.dump(dict(source=f"residuals against the published model, {results.name}", targets=rows), open(OUT, "w"), indent=1)
