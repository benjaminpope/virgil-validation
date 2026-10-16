"""V3a: HR 4796 ALMA visibilities, vis-r's example, against PolarDisk.

Data: vis-r's ``data/hr4796.selfcal.npy`` (drgmk/vis-r, 12.9 MB, 321 322 complex
visibilities, one SPW group; fetched with
``curl -L -o $V3_DATA/visr/hr4796.selfcal.npy https://github.com/drgmk/vis-r/raw/main/data/hr4796.selfcal.npy``).
It is an object array: loaded with the whitelisting unpickler in
src/external_bridge/safe_load.py, never ``allow_pickle=True``.
Reference values: the authors' own notebook (``notebooks/mcmc_fitting.ipynb``)
output for the image-based Gaussian ring, medians and standard deviations of
their MCMC, reproduced below. These are NOT numbers read from the RASTI paper
(not fetched); the check is against the code authors' published run.

Same procedure as vis-r's ``examples/hello_world.py``: reweight so the null
model has chi^2 = N, bin in (u, v) with ``size_arcsec`` = 4, Gaussian ring
(dra, ddec, PA, inc, flux, r0, dr, dz) with the vertical Gaussian factor
exp(-0.5 (2 pi dz r0 sin(i) u_minor)^2). PolarDisk has no scale height, so for
the virgil model the harness multiplies by the same factor (as in V1/A4).

CONVENTION. vis-r's (dra, ddec) is a phase rotation exp(+2 pi i (u dra + v ddec)) of
the model visibilities (CASA sign of u, v). In virgil's sky convention (East, North offset of the source,
visibility exp(-2 pi i (u dra + v ddec))) it is the offset (-dra, -ddec). The first run of this script
passed (+dra, +ddec) to virgil and recovered the same fit with both offsets of opposite sign (a
convention, not a model difference); the harness now maps the sign, and that first result is kept in
the docstring only.

PASS CRITERIA (fixed before the run)
  H1  forward models: at the authors' parameters, max |V_virgil - V_visr| over
      all binned baselines <= 1e-3 flux (vis-r interpolates linearly in q).
  H2  MAP fits (scipy least squares from the authors' starting point, Laplace
      errors from J^T J): virgil MAP agrees with the vis-r MAP within 1 Laplace
      sigma in each of the 8 parameters, and |chi2_virgil - chi2_visr| <= 10.
  H3  both MAPs within 3 sigma (sigma of the authors' MCMC and ours, in
      quadrature) of the authors' notebook values (image-based model: expected
      to differ slightly, reported).
  Reported: raw chi2/N on the quoted (reweighted) errors.
"""

import os
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from common import save
from crosscheck import radial as R
from external_bridge.safe_load import load_object_npy
from external_bridge.radial_codes import quiet
from virgil import GaussianRing, PolarDisk

DATA = pathlib.Path(os.environ.get("V3_DATA", pathlib.Path.home() / "data" / "polar_disk_v3"))
ARCSEC = np.pi / 180 / 3600
NAMES = ["dra", "ddec", "PA", "inc", "flux", "r0", "dr", "dz"]
NOTEBOOK = np.array([0.030333, 0.0306575, 26.6274, 76.614, 0.0150856, 1.07795, 0.059447, 0.0388962])
NOTEBOOK_STD = np.array([0.00133613, 0.00160744, 0.0966579, 0.10023, 0.000143231, 0.0021021, 0.00286252, 0.0022413])
P0 = np.array([0.03, 0.03, 26.7, 76.7, 0.015, 1.08, 0.06, 0.04])
SZ = 4.0


def load():
    import vis_r

    u, v, re, im, w, wavelength, ms = load_object_npy(DATA / "visr" / "hr4796.selfcal.npy")
    u, v, re, im, w = (np.asarray(x, float) for x in (u, v, re, im, w))
    w = w * 2 * len(w) / np.sum((re**2 + im**2) * w)
    with quiet():
        u, v, re, im, w = vis_r.bin_uv(u, v, re, im, w, size_arcsec=SZ, verb=False)
    ok = np.isfinite(re) & np.isfinite(im) & (w > 0)
    return u[ok], v[ok], re[ok], im[ok], w[ok], float(wavelength)


def visr_model(u, v):
    from vis_r import functions as vf

    Rnk, Qzero, Ykm = vf.setup_dht(SZ, u, v)

    def model(p):
        urot, ruv = vf.uv_trans(u, v, np.deg2rad(p[2]), np.deg2rad(p[3]))
        f = np.exp(-0.5 * np.square((Rnk / ARCSEC - p[5]) / p[6]))
        fth = Ykm @ f
        fth = fth * p[4] / fth[0]
        vis = np.interp(ruv, Qzero, fth)
        vis = vis * np.exp(-0.5 * np.square(p[7] * p[5] * np.sin(np.deg2rad(p[3])) * urot * ARCSEC * 2 * np.pi))
        return vis * np.exp(1j * (u * p[0] + v * p[1]) * ARCSEC * 2 * np.pi)

    return model


def virgil_model(u, v):
    uj, vj = jnp.asarray(u), jnp.asarray(v)
    wl = jnp.ones_like(uj)  # u, v already in wavelengths: baseline in metres = wavelengths at wavel = 1

    @jax.jit
    def model(p):
        disk = PolarDisk(GaussianRing(p[5] * 1e3, p[6] * 1e3), inc=p[3], pa=p[2],
                         dra=-p[0] * 1e3, ddec=-p[1] * 1e3, n_r=128)  # sign: see CONVENTION in the docstring
        vis = p[4] * disk.model(uj, vj, wl)  # PolarDisk is unit flux: the flux is the component weight
        urot = uj * jnp.cos(jnp.deg2rad(p[2])) - vj * jnp.sin(jnp.deg2rad(p[2]))
        return vis * jnp.exp(-0.5 * jnp.square(p[7] * p[5] * jnp.sin(jnp.deg2rad(p[3])) * urot * ARCSEC * 2 * jnp.pi))

    return lambda p: np.asarray(model(jnp.asarray(p, float)))


def fit(model, data, p0):
    u, v, re, im, w = data
    sw = np.sqrt(w)
    scale = np.maximum(np.abs(p0), 1e-3)

    def resid(x):
        vis = model(x * scale)
        return np.concatenate([(re - vis.real) * sw, (im - vis.imag) * sw])

    t0 = time.time()
    sol = least_squares(resid, p0 / scale, method="lm", x_scale=1.0)
    p = sol.x * scale
    j = sol.jac / scale
    cov = np.linalg.inv(j.T @ j)
    return {"p": p, "sigma": np.sqrt(np.diag(cov)), "chi2": float(np.sum(sol.fun**2)), "nfev": sol.nfev,
            "seconds": time.time() - t0}


def main():
    u, v, re, im, w, wavelength = load()
    n = 2 * len(u)
    mv, mg = visr_model(u, v), virgil_model(u, v)
    h1 = float(np.abs(mg(NOTEBOOK) - mv(NOTEBOOK)).max())
    # attribution of any H1 difference: each model against our own Gauss-Legendre Hankel transform
    pk = NOTEBOOK
    q_dep, _, u_minor = R.deproject(u, v, pk[3], pk[2])
    truth = pk[4] * R.hankel(R.gaussian_ring(pk[5] * 1e3, pk[6] * 1e3), q_dep, q_max=float(q_dep.max()))
    truth = truth * np.exp(-0.5 * (2 * np.pi * pk[7] * pk[5] * np.sin(np.deg2rad(pk[3])) * u_minor * ARCSEC) ** 2)
    err_visr = float(np.abs(mv(NOTEBOOK) * np.exp(-1j * (u * pk[0] + v * pk[1]) * ARCSEC * 2 * np.pi) - truth).max())
    err_virgil = float(np.abs(mg(NOTEBOOK) * np.exp(-1j * (u * pk[0] + v * pk[1]) * ARCSEC * 2 * np.pi) - truth).max())
    fv = fit(mv, (u, v, re, im, w), P0)
    fg = fit(mg, (u, v, re, im, w), P0)
    dsig = np.abs(fg["p"] - fv["p"]) / fv["sigma"]
    comb = lambda f: np.abs(f["p"] - NOTEBOOK) / np.hypot(f["sigma"], NOTEBOOK_STD)  # noqa: E731
    out = {
        "data": {"file": "hr4796.selfcal.npy", "source": "github.com/drgmk/vis-r/data", "bytes": 12853509,
                 "n_binned": int(len(u)), "n_dof_data": n, "wavelength_m": wavelength, "bin_sz_arcsec": SZ},
        "criteria": {"H1_max_dV_over_flux": 1e-3, "H2_param_sigma": 1.0, "H2_dchi2": 10.0, "H3_sigma": 3.0},
        "H1_max_abs_dV": h1,
        "H1_pass": bool(h1 <= 1e-3 * NOTEBOOK[4]),
        "H1_amplitude_error_vs_truth": {"vis_r": err_visr, "virgil": err_virgil},
        "vis_r_fit": {k: (x.tolist() if hasattr(x, "tolist") else x) for k, x in fv.items()},
        "virgil_fit": {k: (x.tolist() if hasattr(x, "tolist") else x) for k, x in fg.items()},
        "names": NAMES,
        "H2_param_diff_in_sigma": dsig.tolist(),
        "H2_dchi2": fg["chi2"] - fv["chi2"],
        "H2_pass": bool(dsig.max() <= 1.0 and abs(fg["chi2"] - fv["chi2"]) <= 10.0),
        "chi2_per_data_point_visr": fv["chi2"] / n,
        "chi2_per_data_point_virgil": fg["chi2"] / n,
        "notebook": {"median": NOTEBOOK.tolist(), "std": NOTEBOOK_STD.tolist()},
        "H3_visr_sigma": comb(fv).tolist(),
        "H3_virgil_sigma": comb(fg).tolist(),
        "H3_pass": bool(comb(fv).max() <= 3 and comb(fg).max() <= 3),
        "note": "reference is the authors' notebook output (image-based model), not the RASTI paper table",
    }
    print(save("v3_hr4796", out))
    for k in ("H1_max_abs_dV", "H1_pass", "H2_param_diff_in_sigma", "H2_dchi2", "H2_pass", "H3_visr_sigma",
              "H3_virgil_sigma", "H3_pass", "chi2_per_data_point_visr"):
        print(k, out[k])


if __name__ == "__main__":
    main()
