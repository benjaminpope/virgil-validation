"""V3b: DSHARP AS 209, frank's documented example, against PolarDisk with a RadialGP profile.

Data: frank's ``docs/tutorials/AS209_continuum.npz`` (discsim/frank, 37.4 MB,
977 774 complex visibilities of the DSHARP AS 209 continuum; fetched with
``curl -L -o $V3_DATA/frank/AS209_continuum.npz
https://github.com/discsim/frank/raw/master/docs/tutorials/AS209_continuum.npz``). Plain ``.npz`` arrays, read
with ``allow_pickle=False``. Procedure and parameters are those of frank's
``fitting_procedure`` tutorial: FixedGeometry(inc 34.97, PA 85.76, dRA 1.9e-3,
dDec 2.5e-3 arcsec; the published DSHARP values), FrankFitter(Rmax = 1.6 arcsec,
N = 250, alpha = 1.05, weights_smooth = 1e-4).

virgil has no loader for complex visibilities and this benchmark adds none
(design/radial_profiles.md section 13): the geometry correction and the
deprojected radial binning (2 klambda bins to 4 Mlambda, real part, summed
weights) are done here with frank's own ``apply_correction``, and PolarDisk
(inc = 0, pa = 0: the data are deprojected) is evaluated on the binned
baselines. What is compared is therefore the FORWARD MODEL and the recovered
PROFILE SHAPE, not a posterior from raw data. The fit is a MAP of the RadialGP
latents (200 nodes on 2 mas to 1.6 arcsec; Matern 3/2, sigma = 2.5, ell = 30
mas, fixed; free log-flux), not tuned.

PASS CRITERIA (fixed before the run)
  B1  raw chi2/N (real part, on the quoted errors, same bins) of the RadialGP
      MAP <= 1.5 x that of frank's profile through frank's own predictor.
  B2  ring radii: every local maximum of the frank profile (0.03-0.6 arcsec,
      prominence >= 5 percent of the maximum) has a RadialGP maximum within 5 mas.
  B3  profile shape: after normalizing both to unit flux, the rms difference
      over 0.02-0.6 arcsec is <= 10 percent of the frank peak.
"""

import os
import pathlib
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks

from common import save
from virgil import PolarDisk, RadialGP

DATA = pathlib.Path(os.environ.get("V3_DATA", pathlib.Path.home() / "data" / "polar_disk_v3"))
GEOM = (34.97, 85.76, 1.9e-3, 2.5e-3)  # inc, PA (deg), dRA, dDec (arcsec)
RMAX, N, ALPHA, WSMOOTH = 1.6, 250, 1.05, 1e-4
BIN, QMAX = 2e3, 4e6
N_R, SIGMA, ELL = 200, 2.5, 30.0


def run_frank(u, v, vis, w):
    from frank.geometry import FixedGeometry
    from frank.radial_fitters import FrankFitter

    geom = FixedGeometry(*GEOM)
    t0 = time.time()
    sol = FrankFitter(RMAX, N, geometry=geom, alpha=ALPHA, weights_smooth=WSMOOTH).fit(u, v, vis, w)
    return geom, sol, time.time() - t0


def bin_deprojected(geom, u, v, vis, w):
    up, vp, vp_vis = geom.apply_correction(u, v, vis)
    q = np.hypot(up, vp)
    keep = q < QMAX
    q, re, w = q[keep], vp_vis.real[keep], w[keep]
    edges = np.arange(0.0, QMAX + BIN, BIN)
    idx = np.digitize(q, edges) - 1
    n = len(edges) - 1
    sw = np.bincount(idx, w, n)
    ok = sw > 0
    qb = np.bincount(idx, w * q, n)[ok] / sw[ok]
    reb = np.bincount(idx, w * re, n)[ok] / sw[ok]
    return qb, reb, sw[ok]


def main():
    d = np.load(DATA / "frank" / "AS209_continuum.npz", allow_pickle=False)
    u, v, vis, w = (d[k] for k in ("u", "v", "V", "weights"))
    geom, sol, t_frank = run_frank(u, v, vis, w)
    qb, reb, wb = bin_deprojected(geom, u, v, vis, w)
    n = len(qb)
    pred = np.asarray(sol.predict_deprojected(qb)).real if np.iscomplexobj(sol.predict_deprojected(qb)) else sol.predict_deprojected(qb)
    chi2_frank = float(np.sum((reb - pred) ** 2 * wb))

    r_frank = np.asarray(sol.r)  # arcsec
    i_frank = np.asarray(sol.mean)  # Jy / sr
    flux_frank = 2 * np.pi * np.trapezoid(i_frank * r_frank, r_frank) * (np.pi / 180 / 3600) ** 2

    disk0 = PolarDisk(RadialGP(2.0, RMAX * 1e3, sigma=SIGMA, ell=ELL, n=N_R), inc=0.0, pa=0.0, n_r=N_R)
    qj = jnp.asarray(qb)
    wl, zero = jnp.ones_like(qj), jnp.zeros_like(qj)

    def model(x):
        disk = eqx.tree_at(lambda dd: dd.profile.z, disk0, x[:-1])
        return jnp.exp(x[-1]) * disk.model(qj, zero, wl).real  # unit-flux disk times the flux

    def loss(x):
        return 0.5 * jnp.sum(wb * (reb - model(x)) ** 2) + 0.5 * jnp.sum(x[:-1] ** 2)

    vg = jax.jit(jax.value_and_grad(loss))
    x0 = np.concatenate([np.zeros(N_R), [np.log(flux_frank)]])
    t0 = time.time()
    res = minimize(lambda x: tuple(np.asarray(a, float) for a in vg(jnp.asarray(x))), x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 3000, "maxfun": 6000})
    t_virgil = time.time() - t0
    chi2_v = float(np.sum(wb * (reb - np.asarray(model(jnp.asarray(res.x)))) ** 2))

    disk = eqx.tree_at(lambda dd: dd.profile.z, disk0, jnp.asarray(res.x[:-1]))
    rg = np.linspace(0.005, 1.0, 1991)  # arcsec
    i_v = np.asarray(disk.profile.intensity(rg * 1e3, 1.0))
    i_f = np.interp(rg, r_frank, i_frank)

    def unit(i):
        return i / np.trapezoid(2 * np.pi * rg * i, rg)

    uf, uv_ = unit(i_f), unit(i_v)
    sel = (rg >= 0.02) & (rg <= 0.6)

    def peaks(i):
        p, _ = find_peaks(i[(rg >= 0.03) & (rg <= 0.6)], prominence=0.05 * i.max())
        return rg[(rg >= 0.03) & (rg <= 0.6)][p]

    pf, pv = peaks(uf), peaks(uv_)
    miss = [float(np.min(np.abs(pv - x)) * 1e3) if len(pv) else float("inf") for x in pf]
    rms = float(np.sqrt(np.mean((uf[sel] - uv_[sel]) ** 2)) / uf[sel].max())
    out = {
        "data": {"file": "AS209_continuum.npz", "source": "github.com/discsim/frank/docs/tutorials", "bytes": 37433007,
                 "n_vis": int(len(u)), "n_bins": int(n), "bin_klambda": BIN / 1e3, "q_max": QMAX},
        "frank": {"geometry": GEOM, "Rmax": RMAX, "N": N, "alpha": ALPHA, "weights_smooth": WSMOOTH,
                  "seconds": t_frank, "flux_Jy": float(flux_frank), "chi2_per_bin": chi2_frank / n},
        "virgil": {"n_r": N_R, "sigma": SIGMA, "ell_mas": ELL, "flux_Jy": float(np.exp(res.x[-1])), "optimizer": res.message,
                   "nit": int(res.nit), "seconds": t_virgil, "chi2_per_bin": chi2_v / n},
        "criteria": {"B1_ratio": 1.5, "B2_mas": 5.0, "B3_rms_frac": 0.10},
        "B1_ratio": chi2_v / chi2_frank, "B1_pass": bool(chi2_v <= 1.5 * chi2_frank),
        "B2_frank_peaks_arcsec": pf.tolist(), "B2_virgil_peaks_arcsec": pv.tolist(), "B2_offsets_mas": miss,
        "B2_pass": bool(max(miss) <= 5.0 if miss else False),
        "B3_rms_fraction_of_peak": rms, "B3_pass": bool(rms <= 0.10),
        "profile": {"r_arcsec": rg[::10].tolist(), "frank_unit_flux": uf[::10].tolist(), "virgil_unit_flux": uv_[::10].tolist()},
        "scope": "forward model and profile shape on frank-geometry-corrected, radially binned real parts; no raw-data posterior",
    }
    print(save("v3_as209", out))
    for k in ("frank", "virgil", "B1_ratio", "B1_pass", "B2_frank_peaks_arcsec", "B2_virgil_peaks_arcsec", "B2_offsets_mas",
              "B2_pass", "B3_rms_fraction_of_peak", "B3_pass"):
        print(k, out[k])


if __name__ == "__main__":
    main()
