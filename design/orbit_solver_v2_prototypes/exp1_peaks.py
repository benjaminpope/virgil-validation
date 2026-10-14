"""Exp 1: per-night multimodal positions for Gl 229 (all 7 nights), with per-peak covariances.

Blind: the grid, n_peaks and the scoring never see the reference. Xuan+2024 is used only AFTER, to
report the rank of the peak nearest the published track on each night.
usage: python exp1_peaks.py [n_peaks] [half_mas]
"""
import os, sys, time, json, pathlib
import numpy as onp
ROOT = pathlib.Path(__file__).resolve().parents[2]  # the virgil-validation checkout
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(pathlib.Path(__file__).resolve().parent)]
import jax
jax.config.update("jax_enable_x64", True)
import reanalyse_orbits as R
from virgil.epochs import Epochs, epoch_positions, _Surface, _negative_marginal_hessian
HERE = pathlib.Path(os.environ.get("PROTO_WORK", pathlib.Path(__file__).parent))  # inputs and outputs

n_peaks = int(sys.argv[1]) if len(sys.argv) > 1 else 8
half = float(sys.argv[2]) if len(sys.argv) > 2 else 18.0
import common_data as CD
L = CD.load()
times = onp.array([float(onp.mean(onp.asarray(d.mjd, float))) for d in L.data])
wmin = min(float(onp.min(onp.asarray(d.wavel, float))) for d in L.data)
bmax = max(float(onp.max(onp.hypot(onp.asarray(d.u, float), onp.asarray(d.v, float)))) for d in L.data)
step = wmin / (4 * bmax) * 206264.806e3
n = 2 * int(onp.ceil(half / step)) + 1
ax = onp.linspace(-half, half, n)
grid = dict(dra=ax, ddec=ax, flux=onp.geomspace(0.02, 1.0, 13))
print(f"grid {n}x{n}x13, step {step:.3f} mas, half {half} mas; channels/night",
      [int(onp.size(onp.unique(onp.asarray(d.wavel)))) for d in L.data], flush=True)
t0 = time.time()
ep = epoch_positions(Epochs(dict(zip(L.names, L.data))), grid, n_peaks=n_peaks)
t_pos = time.time() - t0
ref = onp.stack(R.ref_track(CD.reference(), times), -1)[:, :2] if True else None
out = []
for i, (nm, pk) in enumerate(zip(L.names, ep.peaks)):
    surf = _Surface(L.data[i])
    covs = []
    for k in range(len(pk)):
        x = jax.numpy.array([pk.dra[k], pk.ddec[k], pk.flux[k]])
        H = onp.asarray(_negative_marginal_hessian(x, 1.0, L.data[i], surf))
        try:
            C = onp.linalg.inv(H)[:2, :2]
            ok = bool(onp.all(onp.linalg.eigvalsh(H) > 0))
        except onp.linalg.LinAlgError:
            C, ok = onp.full((2, 2), onp.nan), False
        covs.append((C.tolist(), ok))
    d = onp.hypot(pk.dra - ref[i, 0], pk.ddec - ref[i, 1])
    j = int(onp.argmin(d))
    row = dict(night=nm, mjd=times[i], ref=ref[i].tolist(), n=len(pk), nearest_rank=j, nearest_mas=float(d[j]),
               dm_best_minus_nearest=float(pk.marginal[0] - pk.marginal[j]),
               peaks=[dict(dra=float(pk.dra[k]), ddec=float(pk.ddec[k]), flux=float(pk.flux[k]), m=float(pk.marginal[k]),
                           w=float(pk.weight[k]), edge=bool(pk.edge[k]), cov=covs[k][0], pd=covs[k][1]) for k in range(len(pk))])
    out.append(row)
    sig = [onp.sqrt(onp.diag(onp.asarray(c[0]))).round(3).tolist() for c in covs[:3]]
    print(f"{nm}: true-track peak rank {j} at {d[j]:.2f} mas, dm(best-it) {row['dm_best_minus_nearest']:.1f}; "
          f"m: {onp.round(pk.marginal[:6]-pk.marginal[0],1).tolist()}; sig(top3) {sig}", flush=True)
print(f"epoch_positions wall {t_pos:.1f} s for {len(L.data)} nights, n_peaks={n_peaks}")
json.dump(dict(t_pos=t_pos, n_peaks=n_peaks, grid_n=n, step=step, nights=out),
          open(HERE / (f"exp1_{CD.TAG}_k{n_peaks}.json"), "w"), indent=1)
