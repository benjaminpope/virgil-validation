"""Exp 3: direct-to-data scoring of the exp-2 candidates, then a vmapped Levenberg-Marquardt multistart.

Coordinates (all unconstrained, no angle wraps): x = (log P, tau, u, A, B, F, G, log flux), with
e = 0.95 sigmoid(u), dt_peri = tau P, and the Thiele-Innes constants (mas) Cartesian. Objective: the
tempered, scale-marginalised log likelihood  l(x) = - sum_n sum_b (f_n nu_b / 2) ln chi2_nb  over every
night n and observable block b (V2 with the night's amplitude profiled, closure phases).
LM: IRLS Gauss-Newton matrix  H = sum_b w_b J_b^T J_b,  w_b = f nu_b / chi2_b  (the gradient is exact;
the curvature drops the residual term and the rank-one -w_b/chi2_b (J^T r)(J^T r)^T term).
usage: python exp3_lm.py SELECTION SCREEN_JSON [n_starts] [n_iter]
"""
import os, sys, json, time, pathlib
import numpy as onp
ROOT = pathlib.Path(__file__).resolve().parents[2]  # the virgil-validation checkout
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(pathlib.Path(__file__).resolve().parent)]
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as np
import reanalyse_orbits as R
import fastcvis
fastcvis.enable()
from virgil.models import OrbitalBinary
from virgil.orbits import ThieleInnesOrbit, _unit_orbit
from virgil.likelihood import whitened_residuals
from virgil.epochs import _Surface

HERE = pathlib.Path(os.environ.get("PROTO_WORK", pathlib.Path(__file__).parent))  # inputs and outputs
sel, screen = sys.argv[1], json.load(open(HERE / sys.argv[2]))
n_starts = int(sys.argv[3]) if len(sys.argv) > 3 else 32
n_iter = int(sys.argv[4]) if len(sys.argv) > 4 else 40
import common_data as CD
L = CD.load()
keep = [i for i, nm in enumerate(L.names) if nm in screen["nights"]]
data = [L.data[i] for i in keep]
times = onp.array([float(onp.mean(onp.asarray(d.mjd, float))) for d in data])
t_ref = screen["t_ref"]
f = onp.asarray(screen["f"])
surfs = [_Surface(d) for d in data]
AMP = (0.01, 1.5)


class TIOrbit(ThieleInnesOrbit):
    def _relative(self, dt):
        x, y = _unit_orbit(dt, self.period, self.dt_peri, self.ecc)
        return np.stack([self.B * x + self.G * y, self.A * x + self.F * y, np.zeros_like(x)])


def scene(x):
    P = np.exp(x[0]); e = 0.95 * jax.nn.sigmoid(x[2])
    return OrbitalBinary(TIOrbit(P, x[1] * P, e, x[3], x[4], x[5], x[6], t_ref=t_ref), np.exp(x[7]))


def night_resid(x, d, s):
    """Whitened residuals of one night, V2 with its amplitude profiled (variable projection)."""
    r = whitened_residuals(scene(x), d)
    n = s.blocks[0][2]
    obs, err = np.reshape(d.vis, -1), np.reshape(d.d_vis, -1)
    mod = obs + r[:n] * err
    w = 1 / err**2
    amp = np.clip(np.sum(w * obs * mod) / np.sum(w * mod**2), *AMP)
    return np.concatenate([(obs - amp * mod) / err, r[n:]])


def all_resid(x):
    return [night_resid(x, d, s) for d, s in zip(data, surfs)]


def block_chi2(rs):
    return [np.stack([np.sum(r[a:b] ** 2) for _, a, b, _, _ in s.blocks]) for r, s in zip(rs, surfs)]


def loglike(x):
    return sum(-0.5 * f[n] * np.sum(np.asarray(s.nu) * np.log(c)) for n, (c, s) in enumerate(zip(block_chi2(all_resid(x)), surfs)))


def lm_step(state):
    x, lam, ll = state
    rs = all_resid(x)
    Js = jax.jacfwd(all_resid)(x)
    H = np.zeros((8, 8)); g = np.zeros(8)
    for n, (r, J, s) in enumerate(zip(rs, Js, surfs)):
        for _, a, b, nu, _ in s.blocks:
            c = np.sum(r[a:b] ** 2)
            w = f[n] * nu / c
            H = H + w * J[a:b].T @ J[a:b]
            g = g + w * J[a:b].T @ r[a:b]
    D = np.diag(H) + 1e-12
    delta = -np.linalg.solve(H + lam * np.diag(D), g)
    ll_new = loglike(x + delta)
    ok = np.isfinite(ll_new) & (ll_new > ll)
    return (np.where(ok, x + delta, x), np.where(ok, lam / 3, lam * 4), np.where(ok, ll_new, ll)), (ll_new, ok)


@jax.jit
def lm(x0):
    def body(state, _):
        return lm_step(state)
    (x, lam, ll), (hist, oks) = jax.lax.scan(body, (x0, 1e-2, loglike(x0)), None, length=n_iter)
    return x, ll, lam, hist


ll_batch = jax.jit(jax.vmap(loglike))
# --- stage A: score every screen candidate on all the data, over a flux scan
fluxes = onp.geomspace(0.05, 1.0, 10)
X0 = []
for c in screen["cands"]:
    u = onp.log(max(c["e"], 1e-3) / 0.95) - onp.log1p(-max(c["e"], 1e-3) / 0.95)
    for fl in fluxes:
        X0.append([onp.log(c["P"]), c["tau"], u, *c["beta"], onp.log(fl)])
X0 = onp.asarray(X0)
t0 = time.time()
sc = onp.concatenate([onp.asarray(ll_batch(np.asarray(X0[i:i + 256]))) for i in range(0, len(X0), 256)])
t_score = time.time() - t0
sc[~onp.isfinite(sc)] = -onp.inf
sc = sc.reshape(len(screen["cands"]), len(fluxes))
best_flux = sc.argmax(1)
cand_ll = sc.max(1)
order = onp.argsort(-cand_ll)
print(f"stage A: {X0.shape[0]} rows scored in {t_score:.1f} s ({1e3 * t_score / X0.shape[0]:.2f} ms/row)", flush=True)
sel_idx = list(order[:n_starts])
PERBAND = int(os.environ.get("PERBAND", 0))  # also start from the best PERBAND candidates of every band
if PERBAND:
    per = {}
    for i in order:
        bnd = screen["cands"][i]["band"]
        if per.get(bnd, 0) < PERBAND:
            per[bnd] = per.get(bnd, 0) + 1
            if i not in sel_idx:
                sel_idx.append(i)
order = onp.array(sel_idx + [i for i in order if i not in set(sel_idx)])
n_starts = len(sel_idx)
print(f"{n_starts} LM starts ({len(set(c['band'] for c in screen['cands']))} bands)", flush=True)
starts = onp.array([X0[i * len(fluxes) + best_flux[i]] for i in order[:n_starts]])
# --- stage B: vmapped LM from the top n_starts (blind)
lm_v = jax.jit(jax.vmap(lm))
t0 = time.time()
xs, lls, lams, hist = map(onp.asarray, jax.block_until_ready(lm_v(np.asarray(starts))))
t_lm = time.time() - t0
print(f"stage B: {n_starts} starts x {n_iter} LM iterations in {t_lm:.1f} s (incl. compile)", flush=True)
t0 = time.time()
jax.block_until_ready(lm_v(np.asarray(starts)))
t_lm2 = time.time() - t0
print(f"stage B again (compiled): {t_lm2:.1f} s = {1e3 * t_lm2 / n_iter / n_starts:.1f} ms per start-iteration", flush=True)
# --- curvature at the best modes: GN vs full Hessian of -l
ref_tr = onp.stack(R.ref_track(CD.reference(), times), -1)[:, :2]


def track(x):
    o = TIOrbit(onp.exp(x[0]), x[1] * onp.exp(x[0]), 0.95 / (1 + onp.exp(-x[2])), *x[3:7], t_ref=t_ref)
    return onp.stack(o.sky(times), -1)


rows = []
for k in onp.argsort(-lls):
    x = xs[k]
    o = TIOrbit(onp.exp(x[0]), x[1] * onp.exp(x[0]), 0.95 / (1 + onp.exp(-x[2])), *x[3:7], t_ref=t_ref).to_kepler()
    rows.append(dict(start_rank=int(k), start_ll=float(cand_ll[order[k]]), ll=float(lls[k]), P=float(o.period),
                     e=float(o.ecc), inc=float(o.inc), a=float(o.a_mas), flux=float(onp.exp(x[7])),
                     band=int(round((times.max() - times.min()) / float(o.period))),
                     ref_rms=float(onp.sqrt(onp.mean(onp.sum((track(x) - ref_tr) ** 2, -1)))),
                     lam=float(lams[k]), x=x.tolist()))
best = rows[0]["ll"]
for r in rows[:12]:
    print(f"  dll {r['ll'] - best:8.2f} (start #{r['start_rank']}, start dll {r['start_ll'] - best:8.2f}) N={r['band']} "
          f"P {r['P']:.4f} e {r['e']:.3f} i {r['inc']:.1f} a {r['a']:.2f} flux {r['flux']:.3f} ref_rms {r['ref_rms']:.2f} lam {r['lam']:.1e}")
xu = [i for i, r in enumerate(rows) if r["ref_rms"] < 1.5]
print("Xuan-like refined modes (track rms < 1.5 mas):", [(i, round(rows[i]["ll"] - best, 2), round(rows[i]["ref_rms"], 2)) for i in xu])
negH = jax.jit(jax.hessian(lambda x: -loglike(x)))


def gn(x):
    rs = all_resid(x); Js = jax.jacfwd(all_resid)(x); H = np.zeros((8, 8))
    for n, (r, J, s) in enumerate(zip(rs, Js, surfs)):
        for _, a, b, nu, _ in s.blocks:
            H = H + f[n] * nu / np.sum(r[a:b] ** 2) * J[a:b].T @ J[a:b]
    return H


gn = jax.jit(gn)
curv = []
for r in rows[:4] + ([rows[xu[0]]] if xu and xu[0] >= 4 else []):
    x = np.asarray(r["x"])
    t0 = time.time(); Hf = onp.asarray(negH(x)); th = time.time() - t0
    t0 = time.time(); Hg = onp.asarray(gn(x)); tg = time.time() - t0
    ef, eg = onp.linalg.eigvalsh(Hf), onp.linalg.eigvalsh(Hg)
    curv.append(dict(ll=r["ll"], ref_rms=r["ref_rms"], full_min_eig=float(ef.min()), gn_min_eig=float(eg.min()),
                     logdet_full=float(onp.linalg.slogdet(Hf)[1]) if ef.min() > 0 else None,
                     logdet_gn=float(onp.linalg.slogdet(Hg)[1]), t_full=th, t_gn=tg))
    print(f"  curvature at dll {r['ll'] - best:.2f}: full min eig {ef.min():.3g} (pd {ef.min() > 0}), GN min eig {eg.min():.3g}; "
          f"logdet full {curv[-1]['logdet_full']} GN {curv[-1]['logdet_gn']:.2f}; t {th:.1f}/{tg:.1f} s", flush=True)
json.dump(dict(sel=sel, t_score=t_score, t_lm=t_lm, t_lm_compiled=t_lm2, n_starts=n_starts, n_iter=n_iter,
               rows=rows, curvature=curv),
          open(HERE / f"exp3_{CD.TAG}_{sel}.json", "w"), indent=1)
