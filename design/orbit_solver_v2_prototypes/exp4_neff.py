"""Exp 4: effective number of independent spectral channels from the wavelength structure of residuals.

For each Gl 229 night, at its best single-night static binary (exp1 peak 0, blind), the whitened V2 and
closure-phase residuals are grouped into spectra (one exposure frame x baseline or triangle, ordered by
wavelength). Two estimators of the dof fraction f = nu_eff / nu:
  * ACF: f = 1 / tau, tau = 1 + 2 sum_k rho_k (pooled autocorrelation, summed until it first drops below 0);
  * binning: f = 1 / max_b [ b * mean(binned residual)^2 / mean(residual^2) ] over b = 2..64 adjacent channels.
Also a 'mean-removed' variant (each spectrum's mean subtracted) which separates per-spectrum offsets
(calibration) from in-band structure.
"""
import os, sys, json, pathlib
import numpy as onp
ROOT = pathlib.Path(__file__).resolve().parents[2]  # the virgil-validation checkout
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(pathlib.Path(__file__).resolve().parent)]
import jax
jax.config.update("jax_enable_x64", True)
import reanalyse_orbits as R
from virgil.models import BinaryModelCartesian

HERE = pathlib.Path(os.environ.get("PROTO_WORK", pathlib.Path(__file__).parent))  # inputs and outputs
import common_data as CD
peaks = json.load(open(HERE / f"exp1_{CD.TAG}_k25.json"))["nights"]
L = CD.load()


def spectra(res, keys, wl):
    out = []
    for key in sorted(set(keys)):
        idx = [i for i, k in enumerate(keys) if k == key]
        idx = sorted(idx, key=lambda i: wl[i])
        if len(idx) >= 8:
            out.append(res[idx])
    return out


def tau_acf(specs, demean):
    kmax = min(len(s) for s in specs) // 2
    num = onp.zeros(kmax); den = 0.0
    for s in specs:
        s = s - s.mean() if demean else s
        den += onp.sum(s * s)
        for k in range(1, kmax):
            num[k] += onp.sum(s[:-k] * s[k:])
    rho = num / den
    tau = 1.0
    for k in range(1, kmax):
        if rho[k] <= 0:
            break
        tau += 2 * rho[k]
    return tau, rho[1:6]


def tau_bin(specs, demean):
    best = 1.0; curve = {}
    for b in (2, 4, 8, 16, 32, 64):
        num = den = 0.0
        for s in specs:
            s = s - s.mean() if demean else s
            n = len(s) // b
            if n < 1:
                continue
            m = s[: n * b].reshape(n, b).mean(1)
            num += b * onp.sum(m * m); den += onp.sum(s[: n * b] ** 2) / b * 1.0
        r = num / den * 1.0 / 1.0 if den else onp.nan
        r = num / (den) if den else onp.nan
        curve[b] = round(float(r), 1)
        best = max(best, r)
    return best, curve


rows = []
for d, p in zip(L.data, peaks):
    pk = p["peaks"][0]
    pred = onp.asarray(d.model(BinaryModelCartesian(pk["dra"], pk["ddec"], pk["flux"])))
    nv = onp.size(d.vis)
    v2m, cpm = pred[:nv], pred[nv:]
    obs, err = onp.asarray(d.vis), onp.asarray(d.d_vis)
    w = 1 / err**2
    amp = float(onp.clip(onp.sum(w * obs * v2m) / onp.sum(w * v2m**2), 0.01, 1.5))
    rv = (obs - amp * v2m) / err
    dphi = onp.angle(onp.exp(1j * (onp.asarray(d.phi) - cpm)))
    rc = dphi / onp.asarray(d.d_phi)
    st, fr, wl = onp.asarray(d.stations), onp.asarray(d.frame), onp.asarray(d.wavel)
    vi = onp.asarray(d.vis_index); ci = onp.asarray(d.i_cps1); ci2 = onp.asarray(d.i_cps2)
    kv = [(int(fr[i]), tuple(st[i])) for i in vi]
    kc = [(int(fr[i]), tuple(st[i]), tuple(st[j])) for i, j in zip(ci, ci2)]
    row = dict(night=p["night"], amp=amp, chi2nu_v2=float(onp.mean(rv**2)), chi2nu_cp=float(onp.mean(rc**2)))
    for name, res, keys, idx in (("v2", rv, kv, vi), ("cp", rc, kc, ci)):
        sp = spectra(res, keys, wl[idx])
        for dm in (False, True):
            t, rho = tau_acf(sp, dm)
            tb, curve = tau_bin(sp, dm)
            row[f"{name}{'_dm' if dm else ''}"] = dict(n_spec=len(sp), len_spec=int(onp.median([len(s) for s in sp])),
                                                     f_acf=round(1 / t, 4), rho1_5=onp.round(rho, 2).tolist(),
                                                     f_bin=round(1 / tb, 4), bin_curve=curve)
    rows.append(row)
    print(json.dumps(row), flush=True)
json.dump(rows, open(HERE / f"exp4_neff_{CD.TAG}.json", "w"), indent=1)
