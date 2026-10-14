"""Exp 2: deterministic orbit candidates from per-night peak catalogues (blind).

Screen score of an orbit (P, e, tau, beta=(A,B,F,G)) is the sum over nights of
    s_n(mu) = max( max_k [ h_nk - 0.5 (mu - x_nk)^T W_nk (mu - x_nk) ],  b_n ),
with h_nk = f_n (m_nk - m_n0) the tempered peak height, W_nk = (C_nk / f_n + sg^2 I)^-1 (a grid floor sg),
and b_n = h_n,K-1 the 'none of the listed peaks' bound (an unlisted basin is no higher than the lowest
listed peak). At each (P, e, tau) grid point, beta is found exactly from every pair of nights x every
pair of their top-Kp peaks (4 linear equations, 4 unknowns), then 3 hard-EM steps (assign each night to
its best component, re-solve weighted LS). No random numbers: the candidate list is a function of the data.

usage: python exp2_screen.py SELECTION [f_mode] [Kp] [sg]
  SELECTION: all | sec5 ; f_mode: measured | 1 ; Kp: peaks per night used for pair seeds ; sg: floor (mas)
The reference (Xuan+2024) is read only after the screen, to report where the published orbit ranks.
"""
import os, sys, json, time, pathlib
import numpy as onp
ROOT = pathlib.Path(__file__).resolve().parents[2]  # the virgil-validation checkout
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(pathlib.Path(__file__).resolve().parent)]
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as np
from virgil.orbits import _unit_orbit, period_grid, ThieleInnesOrbit

HERE = pathlib.Path(os.environ.get("PROTO_WORK", pathlib.Path(__file__).parent))  # inputs and outputs
sel = sys.argv[1] if len(sys.argv) > 1 else "all"
f_mode = sys.argv[2] if len(sys.argv) > 2 else "measured"
Kp = int(sys.argv[3]) if len(sys.argv) > 3 else 5
sg = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
EXCL = {"sec5": {"2024-02-27", "2024-03-28", "2024-12-18"}, "all": set()}[sel]

import common_data as CD
cat = json.load(open(HERE / f"exp1_{CD.TAG}_k25.json"))["nights"]
neff = {r["night"]: r for r in json.load(open(HERE / f"exp4_neff_{CD.TAG}.json"))}
cat = [c for c in cat if c["night"] not in EXCL]
N, K = len(cat), max(c["n"] for c in cat)
t = onp.array([c["mjd"] for c in cat])
t_ref = float(onp.round(t.mean(), 1))
f = onp.ones(N)
if f_mode == "measured":  # per-night dof fraction: count-weighted binning estimate over V2 and CP (exp 4)
    for n, c in enumerate(cat):
        r = neff[c["night"]]
        nv, nc = r["v2"]["n_spec"], r["cp"]["n_spec"]
        f[n] = (nv * r["v2"]["f_bin"] + nc * r["cp"]["f_bin"]) / (nv + nc)
pos = onp.zeros((N, K, 2)); W = onp.tile(onp.eye(2) * 1e-12, (N, K, 1, 1)); h = onp.full((N, K), -1e30)
b = onp.zeros(N)
for n, c in enumerate(cat):
    m0 = c["peaks"][0]["m"]
    for k, p in enumerate(c["peaks"]):
        C = onp.asarray(p["cov"])
        if not (p["pd"] and onp.all(onp.isfinite(C))):
            C = onp.eye(2) * 0.05**2
        pos[n, k] = (p["dra"], p["ddec"])
        W[n, k] = onp.linalg.inv(C / f[n] + sg**2 * onp.eye(2))
        h[n, k] = f[n] * (p["m"] - m0)
    b[n] = f[n] * (c["peaks"][-1]["m"] - m0)
PLO, PHI = float(os.environ.get("PLO", 10.0)), float(os.environ.get("PHI", 14.0))
periods = period_grid(t, PLO, PHI)
eccs = onp.arange(0.0, 0.91, float(os.environ.get("ESTEP", 0.05)))
n_tau = int(os.environ.get("NTAU", 36))
grid = onp.array([(p, e, j / n_tau) for p in periods for e in eccs for j in range(n_tau)])
pairs = onp.array([(i, j, ki, kj) for i in range(N) for j in range(i + 1, N) for ki in range(Kp) for kj in range(Kp)])
print(f"{sel}: N={N} nights, f={onp.round(f,3).tolist()}, grid {len(periods)}P x {len(eccs)}e x {n_tau} = {len(grid)}, "
      f"seeds/point {len(pairs)}, sg={sg}", flush=True)
pos_j, W_j, h_j, b_j, t_j = map(np.asarray, (pos, W, h, b, t - t_ref))


def design(g):
    P, e, tau = g
    x, y = _unit_orbit(t_j, P, tau * P, e)
    z = np.zeros_like(x)
    return np.stack([np.stack([z, x, z, y], -1), np.stack([x, z, y, z], -1)], -2)  # (N, 2[dra,ddec], 4[A,B,F,G])


def comp_scores(M, beta):
    mu = M @ beta  # (N, 2)
    r = mu[:, None, :] - pos_j  # (N, K, 2)
    return h_j - 0.5 * np.einsum("nki,nkij,nkj->nk", r, W_j, r)


def solve(M, wsel):  # wsel (N, K) 0/1 assignment weights
    Wn = np.einsum("nk,nkij->nij", wsel, W_j)
    xn = np.einsum("nk,nkij,nkj->ni", wsel, W_j, pos_j)
    A = np.einsum("nia,nij,njb->ab", M, Wn, M) + 1e-9 * np.eye(4)
    rhs = np.einsum("nia,ni->a", M, xn)
    return np.linalg.solve(A, rhs)


def seed_score(M, pr):
    i, j, ki, kj = pr
    wsel = np.zeros((N, K)).at[i, ki].set(1.0).at[j, kj].set(1.0)
    beta = solve(M, wsel)
    for _ in range(3):
        s = comp_scores(M, beta)
        best = np.argmax(s, 1)
        inlier = np.max(s, 1) > b_j
        wsel = jax.nn.one_hot(best, K) * inlier[:, None]
        beta = solve(M, wsel)
    s = comp_scores(M, beta)
    tot = np.sum(np.maximum(np.max(s, 1), b_j))
    return tot, beta, np.where(np.max(s, 1) > b_j, np.argmax(s, 1), -1)


@jax.jit
def point(g):
    M = design(g)
    tot, beta, assign = jax.vmap(lambda pr: seed_score(M, pr))(np.asarray(pairs))
    k = np.argmax(tot)
    return tot[k], beta[k], assign[k]


t0 = time.time()
tot, beta, assign = jax.lax.map(point, np.asarray(grid), batch_size=256)
tot, beta, assign = map(onp.asarray, jax.block_until_ready((tot, beta, assign)))
dt_screen = time.time() - t0
print(f"screen wall {dt_screen:.1f} s ({len(grid) * len(pairs) / dt_screen:.3g} seeds/s)", flush=True)

# --- candidates: grid points within D of the best, deduplicated by sky track at the observed nights
D = float(os.environ.get("DSCREEN", 25.0))
order = onp.argsort(-tot)
best = tot[order[0]]
cands, tracks = [], []
PERCELL = int(os.environ.get("PERCELL", 0))  # >0: keep the best PERCELL per (band, e-bin of 0.1), no global cap
cell_n, cell_tr = {}, {}
span = t.max() - t.min()
for g in order:
    if tot[g] < best - D or (not PERCELL and len(cands) >= int(os.environ.get("CAP", 400))):
        break
    P, e, tau = grid[g]
    if PERCELL:
        key = (int(round(span / P)), int(e / 0.1))
        if cell_n.get(key, 0) >= PERCELL:
            continue
    M = onp.asarray(design(grid[g]))
    tr = M @ beta[g]
    pool = cell_tr.setdefault(key, []) if PERCELL else tracks  # per-cell de-duplication when PERCELL
    if pool and (onp.max(onp.hypot(*(onp.asarray(pool) - tr).transpose(2, 0, 1)), axis=1) < 0.5).any():
        continue
    ko = ThieleInnesOrbit(P, tau * P, e, *beta[g], t_ref=t_ref).to_kepler()
    cands.append(dict(score=float(tot[g] - best), P=float(P), e=float(e), tau=float(tau), beta=beta[g].tolist(),
                      inc=float(ko.inc), a=float(ko.a_mas), Omega=float(ko.Omega), omega=float(ko.omega),
                      assign=assign[g].tolist(), band=int(round((t.max() - t.min()) / P))))
    tracks.append(tr)
    if PERCELL:
        cell_tr[key].append(tr)
        cell_n[key] = cell_n.get(key, 0) + 1
# --- only now: the reference, for the report
import reanalyse_orbits as R
ref_tr = onp.stack(R.ref_track(CD.reference(), t), -1)[:, :2]
for c in cands:
    M = onp.asarray(design((c["P"], c["e"], c["tau"])))
    c["ref_rms_mas"] = float(onp.sqrt(onp.mean(onp.sum((M @ onp.asarray(c["beta"]) - ref_tr) ** 2, -1))))
near = min(range(len(cands)), key=lambda i: cands[i]["ref_rms_mas"]) if cands else None
print(f"{len(cands)} distinct candidates within {D} of best (tempered nats)")
for i, c in enumerate(cands[:12]):
    print(f"  #{i}: dS {c['score']:7.2f} N={c['band']} P {c['P']:.4f} e {c['e']:.2f} i {c['inc']:.1f} a {c['a']:.2f} "
          f"assign {c['assign']} ref_rms {c['ref_rms_mas']:.2f}")
if near is not None:
    c = cands[near]
    print(f"nearest to Xuan: #{near} dS {c['score']:.2f} N={c['band']} P {c['P']:.4f} e {c['e']:.2f} i {c['inc']:.1f} "
          f"a {c['a']:.2f} assign {c['assign']} ref_rms {c['ref_rms_mas']:.2f} mas")
bands = {}
for c in cands:
    bands.setdefault(c["band"], c["score"])
print("best dS per band:", {k: round(v, 1) for k, v in sorted(bands.items())})
json.dump(dict(sel=sel, f=f.tolist(), Kp=Kp, sg=sg, grid=len(grid), seeds=len(pairs), wall=dt_screen,
               nights=[c["night"] for c in cat], t_ref=t_ref, cands=cands, nearest=near),
          open(HERE / f"exp2_{CD.TAG}_{sel}_{f_mode}_K{Kp}_sg{sg}{os.environ.get('SUFFIX', '')}.json", "w"), indent=1)
