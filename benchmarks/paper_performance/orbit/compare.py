"""Posterior comparison of the two codes for one orbit case (plan section 5, "Accuracy"). Reads the saved draws of
``run_virgil.py`` (samples.npz, shape chain x draw) and ``run_octofitter.jl`` (chain<k>.csv), puts both on the same
quantities and reports, for each quantity: R-hat and bulk ESS per code (ArviZ, same estimator, 4 chains), the
cross-code difference of the medians in units of the pooled sd, the sd ratio and a two-sample KS statistic.

Quantities compared (all invariant to the omega/Omega branch choice): P, M_tot, e, inc, a_mas and the Thiele-Innes
constants A, B, F, G. The phase is compared as M0 for ``matched`` only through the circular mean resultant length.
Octofitter's a_mas is derived here with the same Kepler constant as virgil (gate 1 shows the two agree).

    compare.py --virgil DIR --octofitter DIR --case O-A-sim [--out FILE.json]
    compare.py --pulls ...    (pulls: see ``pulls`` below; needs the per-seed run directories)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "src"))

import make_data as md  # noqa: E402

QUANT = ["P", "M_tot", "e", "inc", "a_mas", "A", "B", "F", "G"]


def load_virgil(d):
    z = np.load(Path(d) / "samples.npz")
    out = {k: z[k] for k in QUANT if k in z.files}
    ph = "M0" if "M0" in z.files else "theta"
    out["phase_deg"] = z[ph]
    return out, z["diverging"] if "diverging" in z.files else None


def load_octofitter(d, plx):
    from crosscheck.orbits import thiele_innes

    chains = []
    for f in sorted(Path(d).glob("chain*.csv")):
        with open(f) as fh:
            rd = csv.reader(fh)
            head = next(rd)
            rows = np.array([[float(x) for x in r] for r in rd])
        chains.append({h: rows[:, i] for i, h in enumerate(head)})
    n = min(c["b_P"].size for c in chains)

    def stack(k, f=lambda x: x):
        return np.stack([f(c[k][:n]) for c in chains])

    P, mass, e, inc = stack("b_P"), stack("A_mass"), stack("b_e"), stack("b_i", np.rad2deg)
    om, Om = stack("b_ω", np.rad2deg), stack("b_Ω", np.rad2deg)
    ph = stack("b_M0", np.rad2deg) if "b_M0" in chains[0] else stack("b_θ", np.rad2deg)
    a = plx * (mass * P**2 / md.KEPLER_K) ** (1.0 / 3.0)
    A, B, F, G = thiele_innes(a, inc, om, Om)
    div = stack("numerical_error") if "numerical_error" in chains[0] else None
    return {"P": P, "M_tot": mass, "e": e, "inc": inc, "a_mas": a, "A": A, "B": B, "F": F, "G": G, "phase_deg": ph}, div


def diag(x):
    import arviz as az
    import xarray as xr

    da = xr.DataArray(np.asarray(x), dims=("chain", "draw"))
    return float(az.rhat(da)), float(az.ess(da, method="bulk"))


def compare(v, o):
    from scipy import stats

    res = {}
    for k in QUANT:
        a, b = v[k], o[k]
        fa, fb = a.reshape(-1), b.reshape(-1)
        sd = np.sqrt(0.5 * (fa.var() + fb.var()))
        rv, ev = diag(a)
        ro, eo = diag(b)
        res[k] = dict(median_virgil=float(np.median(fa)), median_octofitter=float(np.median(fb)),
                      median_diff_in_sd=float((np.median(fa) - np.median(fb)) / sd), sd_ratio=float(fa.std() / fb.std()),
                      ks=float(stats.ks_2samp(fa, fb).statistic), rhat_virgil=rv, rhat_octofitter=ro,
                      ess_bulk_virgil=ev, ess_bulk_octofitter=eo)
    for name, d in (("virgil", v), ("octofitter", o)):
        z = np.exp(1j * np.deg2rad(d["phase_deg"].reshape(-1)))
        res.setdefault("phase_mean_resultant_length", {})[name] = float(np.abs(z.mean()))
    return res


def pulls(truth, runs):
    """z = (posterior mean - truth) / posterior sd over noise seeds; ``runs`` is a list of dicts of flat arrays."""
    out = {}
    for k in ("P", "e", "inc", "a_mas"):
        z = [(r[k].mean() - truth[k]) / r[k].std() for r in runs]
        out[k] = dict(mean=float(np.mean(z)), sd=float(np.std(z, ddof=1)), n=len(z))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--virgil", required=True)
    ap.add_argument("--octofitter", required=True)
    ap.add_argument("--case", default="O-A-sim")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    plx = md.TRUTH["plx"] if a.case != "O-A-real" else 16.176
    v, vdiv = load_virgil(a.virgil)
    o, odiv = load_octofitter(a.octofitter, plx)
    res = compare(v, o)
    res["divergences"] = {"virgil": None if vdiv is None else int(np.asarray(vdiv).sum()),
                          "octofitter": None if odiv is None else int((np.asarray(odiv) != 0).sum())}
    res["n_draws"] = {"virgil": list(v["P"].shape), "octofitter": list(o["P"].shape)}
    txt = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
