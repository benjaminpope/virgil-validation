"""Injection at the real Gl 229 (u, v, t) by residual transplant: data_new = data - model_night + model_inj.
model_night is a static binary per night, refined locally (L-BFGS on the V2-amplitude-profiled,
scale-marginalised likelihood) from the blind 7-night best mode's predicted position on that night
(exp3_gl229_all.json rows[0]); not the night's best peak, which is the wrong peak on 4 of 7 nights and
would leave the real companion in the data. V2 keeps each night's fitted amplitude; closure phases are
wrapped. flux = 0 gives the null injection (no companion).
usage: python inject.py NAME P e inc omega Omega a flux t_peri_offset_days"""
import os, sys, json, pickle, pathlib
import numpy as onp
import common_data as CD
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as np, equinox as eqx
from scipy.optimize import minimize
from virgil.models import OrbitalBinary, BinaryModelCartesian
from virgil.orbits import KeplerOrbit, ThieleInnesOrbit, _unit_orbit
from virgil.likelihood import whitened_residuals
from virgil.epochs import _Surface
import fastcvis; fastcvis.enable()
HERE = pathlib.Path(os.environ.get("PROTO_WORK", pathlib.Path(__file__).parent))  # inputs and outputs
name = sys.argv[1]; P, e, inc, om, Om, a, fl, dtp = map(float, sys.argv[2:10])
best = json.load(open(HERE / "exp3_gl229_all.json"))["rows"][0]
x = onp.asarray(best["x"]); t_ref = json.load(open(HERE / "exp2_gl229_all_measured_K5_sg0.5.json"))["t_ref"]
static = json.load(open(HERE / "exp1_gl229_k25.json"))["nights"]


class TIOrbit(ThieleInnesOrbit):
    def _relative(self, dt):
        X, Y = _unit_orbit(dt, self.period, self.dt_peri, self.ecc)
        return np.stack([self.B * X + self.G * Y, self.A * X + self.F * Y, np.zeros_like(X)])


mode = TIOrbit(onp.exp(x[0]), x[1] * onp.exp(x[0]), 0.95 / (1 + onp.exp(-x[2])), *x[3:7], t_ref=t_ref)


def amp_of(obs, err, m):
    w = 1 / err**2
    return np.clip(np.sum(w * obs * m) / np.sum(w * m**2), 0.01, 1.5)


def refine(d, x0):
    s = _Surface(d); nv = s.blocks[0][2]
    obs, err = np.reshape(d.vis, -1), np.reshape(d.d_vis, -1)

    def negll(p):
        r = whitened_residuals(BinaryModelCartesian(p[0], p[1], np.exp(p[2])), d)
        mod = obs + r[:nv] * err
        r = np.concatenate([(obs - amp_of(obs, err, mod) * mod) / err, r[nv:]])
        return 0.5 * sum(nu * np.log(np.sum(r[a_:b_] ** 2)) for _, a_, b_, nu, _ in s.blocks)

    vg = jax.jit(jax.value_and_grad(negll))
    res = minimize(lambda p: tuple(map(onp.asarray, vg(np.asarray(p)))), x0, jac=True, method="L-BFGS-B",
                   options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 500})
    return res.x


mt = OrbitalBinary(KeplerOrbit(P, dtp, e, inc, om, Om, a, t_ref=t_ref), fl)
L = CD.load()
out, log = [], []
for d, st in zip(L.data, static):
    tn = float(onp.mean(onp.asarray(d.mjd, float)))
    pr = onp.asarray(mode.sky(onp.array([tn])))[:2, 0]
    p = refine(d, onp.array([pr[0], pr[1], x[7]]))
    pk = st["peaks"][0]
    log.append(dict(night=st["night"], pred=pr.round(3).tolist(), refined=p[:2].round(3).tolist(),
                    flux=round(float(onp.exp(p[2])), 3), moved=round(float(onp.hypot(*(p[:2] - pr))), 3),
                    best_peak_dist=round(float(onp.hypot(p[0] - pk["dra"], p[1] - pk["ddec"])), 3)))
    print(log[-1], flush=True)
    mb = BinaryModelCartesian(p[0], p[1], onp.exp(p[2]))
    pb, pt = onp.asarray(d.model(mb)), onp.asarray(d.model(mt))
    nv = onp.size(d.vis)
    obs, err = onp.asarray(d.vis), onp.asarray(d.d_vis)
    amp = float(amp_of(obs, err, pb[:nv]))
    vis = obs - amp * pb[:nv] + amp * pt[:nv]
    phi = onp.angle(onp.exp(1j * (onp.asarray(d.phi) - pb[nv:] + pt[nv:])))
    out.append(eqx.tree_at(lambda q: (q.vis, q.phi), d, (np.asarray(vis), np.asarray(phi))))
pickle.dump(out, open(HERE / f"inj_{name}.pkl", "wb"))
truth = dict(P=P, t_peri=t_ref + dtp, ecc=e, inc=inc, omega=om, Omega=Om, a_mas=a, flux=fl)
json.dump(dict(truth=truth, t_ref=t_ref, best_x=best["x"], subtracted=log), open(HERE / f"inj_{name}.json", "w"), indent=1)
print("injected", name, truth)
