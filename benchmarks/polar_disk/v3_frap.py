"""V3c: HD 169142 ALMA band 3 and band 6 pickles from FRAP's example_data, single-band I(r).

Data: ``example_data/HD169142_band3.pkl`` (6 679 bytes) and
``HD169142_band6.pkl`` (23 557 bytes) of tomyoshida/frap, fetched with
``curl -L -o $V3_DATA/frap/<name> https://github.com/tomyoshida/frap/raw/main/example_data/<name>``.
They are pickles, so they are read with the whitelisting unpickler in
src/external_bridge/safe_load.py. They hold azimuthally averaged, deprojected
real visibilities per frequency channel (q in wavelengths, V in Jy, s in Jy).
Only the first channel of each band is fitted: a single-band I(r), with no
dust model and no spectral index (FRAP's physical model, opacities and
multi-band inference are out of scope).

The reference is FRAP's Hankel matrix H (``set_visibility`` without the primary beam,
Fourier-Bessel nodes to r_out = 1.3 arcsec, N = 130 as in FRAP's tutorial)
with a Gaussian-process MAP written here: I = exp(g), g = c + sigma L z, z ~ N(0, I),
squared-exponential kernel (FRAP's), sigma = 2, ell = 0.03 arcsec, the same prior
and statistics as the RadialGP profile; L-BFGS on (z, c). The tutorial's physical
parameters (fixed T and q, inferred Sigma_d and a_max) are not used. PolarDisk uses
RadialGP(nu = inf, same sigma and ell) on 130 Gauss-Legendre nodes
(2 mas to 1.3 arcsec), free flux.

PASS CRITERIA (fixed before the run)
  F1  forward model: FRAP's H applied to virgil's MAP profile (evaluated at FRAP's nodes) vs
      virgil's own visibilities: max |dV| <= 2e-3 x V(0 baseline).
  F2  fit quality: raw chi2/N (on the quoted s) of the two MAPs differ by <= 5 percent.
  F3  profile: both normalized to unit flux, rms difference over 0.005-0.8 arcsec <= 10 percent
      of the FRAP-basis peak.
"""

import os
import pathlib
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from common import save
from external_bridge.radial_codes import Frap
from external_bridge.safe_load import load_pickle
from virgil import PolarDisk, RadialGP

DATA = pathlib.Path(os.environ.get("V3_DATA", pathlib.Path.home() / "data" / "polar_disk_v3"))
R_OUT, N, SIGMA, ELL = 1300.0, 130, 2.0, 30.0  # mas


def minimize_jax(loss, x0):
    vg = jax.jit(jax.value_and_grad(loss))
    res = minimize(lambda x: tuple(np.asarray(a, float) for a in vg(jnp.asarray(x))), x0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 5000, "maxfun": 10000})
    return res


def one_band(name):
    obs = load_pickle(DATA / "frap" / f"HD169142_{name}.pkl")
    q, v, s = (np.asarray(obs[k][0], float) for k in ("q", "V", "s"))
    n = len(q)
    w = 1.0 / s**2
    fr = Frap(R_OUT, N, q)
    h, r_fb = jnp.asarray(fr.h), np.asarray(fr.r_mas)
    k = SIGMA**2 * np.exp(-0.5 * (r_fb[:, None] - r_fb[None, :]) ** 2 / ELL**2) + 1e-6 * SIGMA**2 * np.eye(N)
    lfac = jnp.asarray(np.linalg.cholesky(k))

    def frap_model(x):
        return h @ jnp.exp(lfac @ x[:-1] + x[-1])

    scale = float(np.max(np.abs(v)))
    # start with a unit-flux model of the correct overall scale: V(0) ~ max |V|
    h0 = np.asarray(h) @ np.ones(N)
    c0 = np.log(scale * N / max(h0.max(), 1e-300) / N)

    def frap_loss(x):
        return 0.5 * jnp.sum(w * (v - frap_model(x)) ** 2) + 0.5 * jnp.sum(x[:-1] ** 2)

    t0 = time.time()
    rf = minimize_jax(frap_loss, np.concatenate([np.zeros(N), [c0]]))
    t_f = time.time() - t0
    chi2_f = float(np.sum(w * (v - np.asarray(frap_model(jnp.asarray(rf.x)))) ** 2))

    disk0 = PolarDisk(RadialGP(2.0, R_OUT, sigma=SIGMA, ell=ELL, nu=float("inf"), n=N), inc=0.0, pa=0.0, n_r=N)
    qj = jnp.asarray(q)
    wl, zero = jnp.ones_like(qj), jnp.zeros_like(qj)

    def virgil_model(x):
        d = eqx.tree_at(lambda dd: dd.profile.z, disk0, x[:-1])
        return jnp.exp(x[-1]) * d.model(qj, zero, wl).real

    def v_loss(x):
        return 0.5 * jnp.sum(w * (v - virgil_model(x)) ** 2) + 0.5 * jnp.sum(x[:-1] ** 2)

    t0 = time.time()
    rv = minimize_jax(v_loss, np.concatenate([np.zeros(N), [np.log(scale)]]))
    t_v = time.time() - t0
    vv = np.asarray(virgil_model(jnp.asarray(rv.x)))
    chi2_v = float(np.sum(w * (v - vv) ** 2))

    # F1: virgil's profile at FRAP's nodes through FRAP's H, vs virgil's own transform.
    d = eqx.tree_at(lambda dd: dd.profile.z, disk0, jnp.asarray(rv.x[:-1]))
    i_at_fb = np.asarray(d.profile.intensity(r_fb, 1.0))
    via_h = np.asarray(h) @ i_at_fb
    # H is up to a constant: match at the shortest baseline, then compare the shape of V(q)/V(q0)
    f1 = float(np.abs(via_h / via_h[0] - vv / vv[0]).max())

    rg = np.linspace(5.0, 800.0, 1591)
    i_v = np.asarray(d.profile.intensity(rg, 1.0))
    g_f = np.asarray(lfac @ jnp.asarray(rf.x[:-1]) + rf.x[-1])
    i_f = np.interp(rg, r_fb, np.exp(g_f))

    def unit(i):
        return i / np.trapezoid(rg * i, rg)

    uf, uv_ = unit(i_f), unit(i_v)
    rms = float(np.sqrt(np.mean((uf - uv_) ** 2)) / uf.max())
    return {
        "n_baselines": int(n), "q_range": [float(q.min()), float(q.max())], "nu_Hz": float(obs["nu"][0]),
        "frap_basis_map": {"chi2_per_point": chi2_f / n, "nit": int(rf.nit), "seconds": t_f, "message": str(rf.message)},
        "virgil_map": {"chi2_per_point": chi2_v / n, "nit": int(rv.nit), "seconds": t_v, "message": str(rv.message)},
        "F1_max_rel_dV_shape": f1, "F1_pass": bool(f1 <= 2e-3),
        "F2_chi2_ratio": chi2_v / chi2_f, "F2_pass": bool(abs(chi2_v / chi2_f - 1) <= 0.05),
        "F3_rms_fraction_of_peak": rms, "F3_pass": bool(rms <= 0.10),
        "profile": {"r_mas": rg[::20].tolist(), "frap_basis_unit": uf[::20].tolist(), "virgil_unit": uv_[::20].tolist()},
    }


def main():
    out = {"criteria": {"F1": 2e-3, "F2": 0.05, "F3": 0.10},
           "data": {"HD169142_band3.pkl": 6679, "HD169142_band6.pkl": 23557, "source": "github.com/tomyoshida/frap/example_data"},
           "setup": {"r_out_mas": R_OUT, "N": N, "sigma": SIGMA, "ell_mas": ELL, "kernel": "squared exponential",
                     "channel": 0}}
    for b in ("band3", "band6"):
        out[b] = one_band(b)
        print(b, {k: out[b][k] for k in out[b] if k != "profile"})
    print(save("v3_frap", out))


if __name__ == "__main__":
    main()
