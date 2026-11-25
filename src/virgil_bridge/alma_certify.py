"""The external convergence evaluator of the DSHARP convergence plan (``reports/dsharp-convergence-plan.md`` §4.1, §5.3, §5.4): the stop test, scored on snapshots.

A method's own stop never counts. ``Evaluator.evaluate`` scores one iterate ``f`` (pixel flux, Jy) in flux coordinates, where ``H_f = T + (w / F) diag(1 / f)`` is
positive definite everywhere (``alma_optim`` documents the objective). The tests, thresholds fixed before any run, ``kappa_0 = 1e-4``:

1. ``kkt_w < 1e-4``: rms of ``g`` (the gradient of ``J`` per pixel flux; ``mu = 0``, the flux is owned by the pixels) over pixels above 1e-3 of the peak, in units of ``w / F``;
   ``want_grow = 0`` (no pixel below 1e-6 of the peak with ``g < -1e-3 w / F``) and ``zero_violation = 0`` (no pixel at or below zero). These are the components of virgil's
   ``kkt_report``, not its ``converged`` flag.
2. Per-class residual: rms ``|g|`` in units of ``w / F`` below 1e-4 above 1e-3 of the peak and below 1e-3 in each of the classes 1e-3 to 1e-6 and below 1e-6.
3. Newton decrement ``D = 1/2 g' H_f^-1 g < 1/2 kappa_0^2 w``, solved in the scaled variables (matrix ``4 X H_f X`` with ``X = diag sqrt f``, preconditioner ``4 f_i T_ii + 4 w / F``) by CG to
   relative residual 1e-3. CG's running estimate of ``D`` rises to the truth, so a solve that has not reached its tolerance within ``cg_maxit`` is *not accepted*: the test fails.
4. Nuisances: the Newton decrement of the dense gain-and-offset block below the same threshold (absent: not applicable).

``band_limit`` is reported by ``kkt_report`` and does not gate. There is no fallback criterion; a run stopped by its step cap is failed (``run_failed``). A method is *certified* only if
the tests pass at a snapshot **and at every later snapshot** (``certify_series``). ``VirgilObjective`` is the objective on a virgil problem (the 25394917 pin) for the cluster runs and for the
chain test, ``fit_result`` rebuilds the ``FitResult`` the gate gives ``kkt_report`` (vv ``alma_lambda.py`` ``fit_point``).
"""

import dataclasses
import glob
import time

import numpy as np
from scipy import linalg

from . import alma_optim as ao

KAPPA0 = 1e-4
KKT_W_MAX = 1e-4
RES_HI, RES_LOW = 1e-4, 1e-3  # class residual limits: above 1e-3 of the peak; below it
GROW_TOL = 1e-3  # want_grow: g below -GROW_TOL w/F in a pixel under 1e-6 of the peak
ACTIVE, FROZEN = 1e-3, 1e-6  # the brightness classes (virgil's _ACTIVE, _FROZEN)
CG_RTOL, CG_MAXIT = 1e-3, 1500
# C3 (plan §5.4): beam-convolved image within 1e-3 of peak; J within the decrement threshold; flux 0.1%; gains 1e-3; offsets 0.5 mas
C3 = dict(image=1e-3, flux=1e-3, gain=1e-3, offset=0.5)


def decrement_threshold(w, kappa0=KAPPA0):
    """``1/2 kappa_0^2 w``: the Newton decrement that ``kkt_w < kappa_0`` implies (the entropy alone gives ``H_ii >= w / (F f_i)``)."""
    return 0.5 * kappa0**2 * float(w)


def newton_decrement(obj, f, g, *, rtol=CG_RTOL, maxit=CG_MAXIT):
    """``(D, info)``: ``D = 1/2 g' H_f^-1 g`` by CG in the scaled variables; ``info['converged']`` is False when CG did not reach ``rtol`` (the decrement is then not accepted)."""
    matvec, b, pre = ao.newton_system(obj, f, g)
    delta, iters, ok, rel = ao._cg(matvec, b, pre, rtol, maxit)
    return 0.5 * float(b @ delta), dict(iterations=iters, converged=bool(ok), rel_residual=rel)


def nuisance_decrement(obj, f):
    """``(D_theta, n)``: ``1/2 g' H^-1 g`` of the nuisance block (dense), or ``(0.0, 0)`` for an objective without one."""
    if not hasattr(obj, "nuisance_gh") or np.asarray(obj.theta).size == 0:
        return 0.0, 0
    g, H = obj.nuisance_gh(f)
    try:
        cf = linalg.cho_factor(0.5 * (H + H.T), lower=True)
    except (np.linalg.LinAlgError, linalg.LinAlgError, ValueError):
        return float("inf"), int(g.size)  # not positive definite: a saddle or a maximum, never stationary-and-minimal, so the nuisance test fails
    return 0.5 * float(g @ linalg.cho_solve(cf, g)), int(g.size)


class Evaluator:
    """Scores iterates of one objective. ``beam`` (optional) maps a flat flux to the beam-convolved image used in the image comparisons; ``reference`` is the reference ``f*`` and its ``J*``."""

    def __init__(self, obj, *, beam=None, reference=None, cg_rtol=CG_RTOL, cg_maxit=CG_MAXIT):
        self.obj, self.beam = obj, beam
        self.cg_rtol, self.cg_maxit = cg_rtol, cg_maxit
        self.reference, self.J_ref, self.ref_image = None, None, None
        if reference is not None:
            self.set_reference(reference)

    def set_reference(self, f, theta=None):
        if theta is not None:
            self.obj.set_theta(theta)
        self.reference = np.array(f, float)
        self.J_ref = self.obj.value_grad(self.reference)[0]
        self.ref_image = self._image(self.reference)

    def _image(self, f):
        return np.asarray(self.beam(f) if self.beam is not None else f, float).ravel()

    def evaluate(self, f, theta=None, *, extra=None):
        """The record of one iterate: every statistic and the booleans of tests 1 to 4 (``certified`` all four)."""
        obj = self.obj
        if theta is not None and np.asarray(theta).size:
            obj.set_theta(theta)
        f = np.asarray(f, float)
        J, g = obj.value_grad(f)
        unit = obj.weight / float(f.sum())  # w / F as kkt_report has it: F is the iterate's own flux (b.sum()), not the reference flux
        peak = float(f.max())
        rel = f / peak
        hi, mid, low = rel > ACTIVE, (rel <= ACTIVE) & (rel >= FROZEN), rel < FROZEN
        rms = lambda m: float(np.sqrt(np.mean(g[m] ** 2)) / unit) if m.any() else 0.0  # noqa: E731
        rec = dict(J=float(J), flux=float(f.sum()), peak=peak, kkt_w=rms(hi), res_hi=rms(hi), res_mid=rms(mid), res_low=rms(low),
                   want_grow=float(np.mean(low & (g < -GROW_TOL * unit))), zero_violation=float(np.mean(f <= 0)),
                   n_hi=int(hi.sum()), n_mid=int(mid.sum()), n_low=int(low.sum()))
        thr = decrement_threshold(obj.weight)
        D, info = newton_decrement(obj, f, g, rtol=self.cg_rtol, maxit=self.cg_maxit)
        Dn, nn = nuisance_decrement(obj, f)
        rec.update(decrement=D, decrement_threshold=thr, cg=info, nuisance_decrement=Dn, n_nuisance=nn)
        if self.J_ref is not None:
            rec["J_minus_Jstar"] = float(J - self.J_ref)
            scale = float(np.max(np.abs(self.ref_image)))
            rec["image_distance"] = float(np.max(np.abs(self._image(f) - self.ref_image)) / scale)
        rec["tests"] = dict(
            kkt_w=rec["kkt_w"] < KKT_W_MAX and rec["want_grow"] == 0 and rec["zero_violation"] == 0,
            classes=rec["res_hi"] < RES_HI and rec["res_mid"] < RES_LOW and rec["res_low"] < RES_LOW,
            decrement=bool(info["converged"] and D < thr),
            nuisance=bool(Dn < thr) if nn else True,
        )
        rec["certified"] = bool(all(rec["tests"].values()) and np.isfinite(J))
        if extra:
            rec.update(extra)
        return rec


def certify_series(records, budget_s=None):
    """Certification of a run from its scored snapshots (each with ``certified``, ``t`` seconds and ``geq`` gradient-equivalents), in time order: certified only if the tests pass at a snapshot
    and at **every later snapshot**. Returns ``dict(certified, index, t, geq, n)``; ``t`` and ``geq`` are those of the first snapshot of the final unbroken run of passes."""
    if budget_s is not None:  # a snapshot taken after the budget is not the method's: only those inside it count
        records = [r for r in records if r["t"] <= budget_s * (1 + 1e-3)]
    n = len(records)
    first = None
    for i in range(n - 1, -1, -1):
        if not records[i]["certified"]:
            break
        first = i
    if first is None:
        return dict(certified=False, index=None, t=None, geq=None, n=n)
    return dict(certified=True, index=first, t=records[first]["t"], geq=records[first]["geq"], n=n)


def run_failed(result):
    """A run that hit its step cap is failed whatever its snapshots say (the plan: caps are safety limits, a capped fit is not a result)."""
    return bool(result.capped)


def score_snapshots(evaluator, paths, *, log=None):
    """Score ``snap_NNNN.npz`` files (a run's directory or a list), in order; the records carry the meter's ``t``, ``geq`` and ``steps``. The evaluator's time is not any method's."""
    if isinstance(paths, str):
        paths = sorted(glob.glob(paths + "/snap_*.npz"))
    out = []
    for p in paths:
        z = np.load(p)
        theta = np.asarray(z["theta"], float)
        rec = evaluator.evaluate(z["f"], theta if theta.size else None, extra=dict(t=float(z["t"]), geq=int(z["geq"]), steps=int(z["steps"]), path=str(p)))
        out.append(rec)
        if log:
            log(f"{p}: t {rec['t']:.0f} s, kkt_w {rec['kkt_w']:.3g}, decrement {rec['decrement']:.3g} / {rec['decrement_threshold']:.3g}, certified {rec['certified']}")
    return out


# ---------------------------------------------------------------- reference, restarts (C3), perturbation (C4)


def reference_agreement(evaluator, f_a, f_b, theta_a=None, theta_b=None):
    """The reference's own test (plan §4.1): ``|dJ| < 1e-6`` (each ``J`` at its own nuisances) and the beam-convolved image difference below 1e-5 of the peak; with nuisances, they also agree to a
    tenth of the C3 tolerances (gains, offsets). Returns ``(agree, dJ, distance)``."""
    obj = evaluator.obj
    theta0 = np.array(obj.theta, float)
    try:
        if theta_a is not None and np.size(theta_a):
            obj.set_theta(theta_a)
        Ja = obj.value_grad(np.asarray(f_a))[0]
        if theta_b is not None and np.size(theta_b):
            obj.set_theta(theta_b)
        Jb = obj.value_grad(np.asarray(f_b))[0]
    finally:
        if theta0.size:
            obj.set_theta(theta0)
    dJ = abs(Ja - Jb)
    ia, ib = evaluator._image(f_a), evaluator._image(f_b)
    dist = float(np.max(np.abs(ia - ib)) / np.max(np.abs(ia)))
    ok = bool(dJ < 1e-6 and dist < 1e-5)
    if theta_a is not None and theta_b is not None and np.size(theta_a):
        for name, a, b in getattr(obj, "layout", [("gain", 0, len(theta_a))]):
            if b > a:
                d = float(np.max(np.abs(np.asarray(theta_b)[a:b] - np.asarray(theta_a)[a:b])))
                ok = ok and d < 0.1 * C3["offset" if "offset" in name else "gain"]
    return ok, float(dJ), dist


def perturb_lognormal(f, dex=0.5, rng=None, flux=None):
    """C4's perturbation: ``f * 10**(dex * N(0,1))``, rescaled to the flux of ``f`` (or ``flux``)."""
    rng = np.random.default_rng(0) if rng is None else rng
    g = np.asarray(f, float) * 10.0 ** (dex * rng.standard_normal(np.shape(f)))
    return g * (float(np.sum(f)) if flux is None else flux) / g.sum()


def gaussian_floor_start(npix, pixel_mas, flux, fwhm_mas=1000.0, floor_jy_arcsec2=1e-4):
    """The 1 arcsec Gaussian raised to the entropy floor (``alma.floor_start``'s start), flat flux ``flux`` (Jy), pixel flux ``(npix**2,)``; renormalised to ``flux``."""
    c = (np.arange(npix) - (npix // 2 - 1)) * pixel_mas  # the sky phase centre under the half-pixel convention (crosscheck.alma_maxent.phase_centre_cell)
    r2 = c[:, None] ** 2 + c[None, :] ** 2
    img = np.exp(-4.0 * np.log(2.0) * r2 / fwhm_mas**2)
    img *= flux / img.sum()
    pix_arcsec2 = (pixel_mas / 1000.0) ** 2
    img = np.maximum(img, floor_jy_arcsec2 * pix_arcsec2)
    return (img * flux / img.sum()).ravel()


def nuisance_starts(obj, theta_ref=None, offset_mas=10.0, gain_frac=0.05):
    """The two nuisance starts of C3: offsets by ``+/- offset_mas``, gains by ``1 +/- gain_frac`` (opposite signs). ``obj.layout`` names the sites (a ``VirgilObjective``);
    for a one-gain toy the single parameter is a gain."""
    ref = np.asarray(obj.theta if theta_ref is None else theta_ref, float)
    layout = getattr(obj, "layout", [("gain", 0, ref.size)])
    out = []
    for sign in (+1.0, -1.0):
        th = ref.copy()
        for name, a, b in layout:
            if "offset" in name:
                th[a:b] = ref[a:b] + sign * offset_mas
            elif "gain" in name:
                th[a:b] = ref[a:b] * (1.0 + sign * gain_frac)
        if hasattr(obj, "theta_bounds"):  # a start outside the prior box would abort the Toeplitz shift: clip inside it
            lo, hi = obj.theta_bounds()
            th = np.clip(th, lo, hi)
        out.append(th)
    return out


def compare_fits(evaluator, ref, other, *, theta_ref=None, theta_other=None):
    """C3/C4 against the converged ``ref``: beam-convolved image within 1e-3 of peak, ``J`` within the decrement threshold, flux within 0.1%, gains within 1e-3, offsets within 0.5 mas.
    Returns ``dict(passes, ...)`` with the measured numbers."""
    obj = evaluator.obj
    theta0 = np.array(obj.theta, float)  # restored on return: a later ``evaluate`` without ``theta`` must see the nuisances it had
    try:
        if theta_ref is not None:
            obj.set_theta(theta_ref)
        Jr = obj.value_grad(np.asarray(ref))[0]
        if theta_other is not None:
            obj.set_theta(theta_other)
        Jo = obj.value_grad(np.asarray(other))[0]
    finally:
        if theta0.size:
            obj.set_theta(theta0)
    ir, io = evaluator._image(ref), evaluator._image(other)
    out = dict(image=float(np.max(np.abs(io - ir)) / np.max(np.abs(ir))), dJ=float(Jo - Jr), flux=float(abs(np.sum(other) / np.sum(ref) - 1.0)))
    ok = out["image"] < C3["image"] and abs(out["dJ"]) < decrement_threshold(obj.weight) and out["flux"] < C3["flux"]
    if theta_ref is not None and theta_other is not None:
        layout = getattr(obj, "layout", [("gain", 0, len(theta_ref))])
        for name, a, b in layout:
            d = float(np.max(np.abs(np.asarray(theta_other)[a:b] - np.asarray(theta_ref)[a:b]))) if b > a else 0.0
            key = "offset" if "offset" in name else "gain"
            out[f"d_{name}"] = d
            ok = ok and d < C3[key]
    out["passes"] = bool(ok)
    return out


# ---------------------------------------------------------------- the objective on a virgil problem


def build_s2_problem(data, *, npix, pixel_mas, flux, ref_obs, gains=False, offsets=False, max_offset_mas=20.0):
    """``alma_lambda.Problem`` for S2: ``LinearImage`` sqrt pixels (flux owned by the pixels, ``reference_flux=flux``), the explicit Toeplitz chi^2, scales fixed
    (``sigmas=False``), gains and offsets fitted or not. With offsets the Toeplitz precompute spreads the box ``+/- max_offset_mas``."""
    import numpyro.distributions as dist
    from virgil import LinearImage
    from virgil.radio import CalibrationNuisance

    from . import alma as va
    from . import alma_lambda as vl

    shift = -pixel_mas / 2.0

    def scene(values):
        return LinearImage(values["pixels"], pixel_mas, dra=shift, ddec=shift, ft="auto", parametrization="sqrt", reference_flux=flux)

    nuis = CalibrationNuisance(data, ref_obs=ref_obs, alpha=False, gains=gains, offsets=offsets, sigmas=False, fit_flux=False)
    term, _ = va.toeplitz_term(data, scene, nuis, npix=npix, pixel_mas=pixel_mas, mask=None, max_offset_mas=float(max_offset_mas))  # resolved Toeplitz term also without offsets
    priors = {"pixels": dist.ImproperUniform(dist.constraints.real, (), event_shape=(npix, npix))}
    if gains or offsets:
        priors.update(nuis.priors(**({"offset": (-float(max_offset_mas), float(max_offset_mas))} if offsets else {})))
    return vl.Problem(data, scene, nuis, term, priors, int(npix), float(pixel_mas), float(flux))


class VirgilObjective(ao.Objective):
    """``J = 1/2 chi2 + MaxEntropy(w)`` of a ``Problem`` (``build_s2_problem``) in pixel flux. The data term is exactly quadratic in ``f`` (the model is linear in flux), so ``hvp`` is the
    constant ``T v``. With ``free`` nuisance sites (default none) ``theta`` holds them, flattened in ``layout`` order; the rest stay at ``fixed`` (default: the nuisance defaults).
    ``diag`` takes ``T_ii`` of the centre pixel for all (shift invariance of the Toeplitz term: an estimate, used only to precondition)."""

    def __init__(self, prob, weight, *, free=(), fixed=None):
        import jax
        import jax.numpy as jnp

        if not jax.config.jax_enable_x64:
            raise RuntimeError("float64 is required: set jax_enable_x64 before building the problem")
        self.prob = prob
        self.term = getattr(prob, "term", None) or prob.terms[0]  # a Problem has one, dsharp_fit's production namespace a list
        nuisance = getattr(prob, "nuisance", None) or prob.nuis
        super().__init__(prob.npix**2, prob.flux, weight)
        defaults = {k: jnp.asarray(v, float) for k, v in nuisance.defaults().items()}
        fixed = {**defaults, **{k: jnp.asarray(v, float) for k, v in (fixed or {}).items()}}
        free = tuple(free)
        self.layout, a = [], 0
        for k in free:
            size = int(np.size(fixed[k]))
            self.layout.append((k, a, a + size))
            a += size
        self._shapes = {k: np.shape(fixed[k]) for k in free}
        self.theta = np.concatenate([np.asarray(fixed[k], float).ravel() for k in free]) if free else np.zeros(0)
        self._fixed, self._free = {k: v for k, v in fixed.items() if k not in free}, free
        shape = (prob.npix, prob.npix)

        def values(f, theta):
            v = dict(self._fixed)
            for k, a0, b0 in self.layout:
                v[k] = theta[a0:b0].reshape(self._shapes[k])
            v["pixels"] = jnp.sqrt(jnp.maximum(f, 1e-300)).reshape(shape)
            return v

        def half(f, theta):
            return 0.5 * self.term.chi2_and_log_norm(values(f, theta))[0]  # less the parameter-independent constants: J ~ 1e4, not 5e7

        self._vg = jax.jit(jax.value_and_grad(half))
        gr = jax.grad(half)
        self._hvp = jax.jit(lambda f, v, theta: jax.jvp(lambda x: gr(x, theta), (f,), (v,))[1])
        self._ng = jax.jit(lambda f, theta: jax.grad(half, argnums=1)(f, theta)) if free else None
        self._nh = jax.jit(lambda f, theta: jax.hessian(half, argnums=1)(f, theta)) if free else None
        self._f0 = jnp.full(self.n, self.q)  # the data term is quadratic: its curvature at any positive point
        self._diag, self._lip = None, None

    def set_theta(self, theta):
        theta = np.asarray(theta, float).copy()
        if theta.shape != self.theta.shape or not np.array_equal(theta, self.theta):
            self._diag, self._lip = None, None  # the curvature T(theta) scales with the gains
        self.theta = theta

    def theta_bounds(self):
        """``(lo, hi)`` over ``theta``: gains positive, offsets inside the problem's box (mas), else unbounded."""
        lo, hi = np.full(self.theta.size, -np.inf), np.full(self.theta.size, np.inf)
        for k, a, b in self.layout:
            if k == "cal.gain":
                lo[a:b] = 1e-3
            elif k == "cal.offset":
                bound = getattr(self.prob, "offset_bound_mas", None)
                if bound is None:  # a Problem: the box is the prior's support
                    pr = self.prob.priors.get("cal.offset")
                    bound = np.inf if pr is None else np.maximum(np.abs(np.asarray(pr.low, float)), np.abs(np.asarray(pr.high, float)))
                bound = np.broadcast_to(np.asarray(bound, float), (b - a,))
                lo[a:b], hi[a:b] = -bound, bound
        return lo, hi

    def data_vg(self, f):
        import jax.numpy as jnp

        v, g = self._vg(jnp.asarray(f), jnp.asarray(self.theta))
        return float(v), np.asarray(g, float)

    def hvp(self, v):
        import jax.numpy as jnp

        return np.asarray(self._hvp(self._f0, jnp.asarray(v, float), jnp.asarray(self.theta)), float)

    def diag(self):
        if self._diag is None:
            e = np.zeros(self.n)
            c = (self.prob.npix // 2) * self.prob.npix + self.prob.npix // 2
            e[c] = 1.0
            self._diag = np.full(self.n, float(self.hvp(e)[c]))
        return self._diag

    def lipschitz(self, iters=30):
        if self._lip is None:
            v = np.random.default_rng(0).standard_normal(self.n)
            lam = 0.0
            for _ in range(iters):
                v /= np.linalg.norm(v)
                tv = self.hvp(v)
                lam = float(v @ tv)
                v = tv
            self._lip = 1.05 * lam  # a margin: power iteration approaches from below
        return self._lip

    def nuisance_gh(self, f):
        import jax.numpy as jnp

        return np.asarray(self._ng(jnp.asarray(f), jnp.asarray(self.theta)), float), np.asarray(self._nh(jnp.asarray(f), jnp.asarray(self.theta)), float)

    def values(self, f):
        """The ``values`` dict of a ``FitResult`` for ``f`` at the current nuisances."""
        import jax.numpy as jnp

        out = {k: jnp.asarray(v) for k, v in self._fixed.items() if k in self.prob.priors}
        for k, a, b in self.layout:
            out[k] = jnp.asarray(self.theta[a:b].reshape(self._shapes[k]))
        out["pixels"] = jnp.sqrt(jnp.asarray(f)).reshape(self.prob.npix, self.prob.npix)
        return out


def fit_result(objective, f, *, n_real=None):
    """The ``FitResult`` of ``f`` as the gate rebuilds it for ``kkt_report`` (``fit_point``: the result's model replaced by ``prob.model``), method ``lbfgsb``."""
    from virgil.fitting import FitResult

    info = dict(method="lbfgsb", steps=0, converged=True, stop=None)
    if n_real is not None:
        info["ndata"] = [float(n_real)]
    return FitResult(model=objective.prob.model, values=objective.values(f), info=info)


def virgil_kkt(objective, f):
    """``kkt_report`` of ``f`` through virgil: ``(kkt_w, band_limit, want_grow, converged)`` of the image block, via the rebuilt ``FitResult``."""
    from virgil.fitting import kkt_report
    from virgil.imaging import MaxEntropy

    import types

    data = objective.prob.data

    class _Likelihood:  # a resolved Toeplitz term keeps no (u, v) in its arrays, which kkt_report's band limit reads: give it the data's own
        arrays = {"u": np.asarray(data.u), "v": np.asarray(data.v)}

        def __init__(self, term):
            self._t = term

        def __getattr__(self, name):
            return getattr(self._t, name)

    reg = MaxEntropy(float(objective.weight))
    rep = kkt_report(fit_result(objective, f), objective.prob.priors, (), [reg], likelihoods=[_Likelihood(objective.term)], weight=float(objective.weight))
    (img,) = rep["images"].values()
    return dict(kkt_w=float(img["kkt_w"]), band_limit=float(img.get("band_limit", np.nan)), want_grow=float(img["want_grow"]), converged=bool(rep["converged"]), report=rep)


def provenance(env=None):
    """The commits under test (virgil's pinned snapshot and this repository's), as ``dsharp_fit.provenance`` records them."""
    import os
    import pathlib
    import re

    from evidence import meta

    env = os.environ if env is None else env
    src = dict(meta.virgil_source())
    pinned = env.get("PIN_VIRGIL_SRC")
    if pinned and re.fullmatch(r"[0-9a-f]{40}", pathlib.Path(pinned).parent.name):
        src = dict(commit=pathlib.Path(pinned).parent.name, url=pinned, dirty=False)
    return dict(virgil_commit=src.get("commit"), virgil_dirty=src.get("dirty"), virgil_source=src.get("url"),
                validation_commit=meta.own_commit() or env.get("PIN_COMMIT"), pins=env.get("PIN_DESC"), when=time.strftime("%Y-%m-%dT%H:%M:%S%z"))


# ---------------------------------------------------------------- the production hook: `dsharp_fit.py --certify alma_certify:certify`

ENV_RUNNER_UP, ENV_BUDGET, ENV_BEAM, ENV_APERTURE = "ALMA_CERTIFY_RUNNER_UP", "ALMA_CERTIFY_BUDGET_S", "ALMA_CERTIFY_BEAM_MAS", "ALMA_CERTIFY_APERTURE_MAS"
C5_RTOL = 1e-10
C6_EDGE = 0.05


def production_objective(prob, values):
    """The ``VirgilObjective`` of dsharp_fit's production problem (its ``terms``, ``nuis``, ``lam``, ``init``) at ``values``: every fitted site but the pixels is a free nuisance."""
    free = tuple(k for k in prob.init if k != "pixels" and k in values)
    return VirgilObjective(prob, float(prob.lam), free=free, fixed={k: np.asarray(values[k], float) for k in free})


def _flux_pixels(values):
    return np.asarray(values["pixels"], float).ravel() ** 2


def _converge(obj, method, f0, theta0, ev, *, budget_s, check_s):
    """Reconverge to the stop test from ``f0`` at ``theta0`` with S2 method ``method``: the method runs in slices of ``check_s``, a Newton-CG polish and the evaluator follow each, and the
    run ends at the first certified slice or at ``budget_s``. Returns ``(f, theta, record, seconds)``."""
    obj.set_theta(theta0)
    f, rec = np.asarray(f0, float), None
    t0 = time.perf_counter()
    spent = 0.0  # wall clock of everything below (method, polish, nuisance step, evaluation), so a method that stalls at once cannot loop uncharged
    while spent < budget_s:
        res = ao.run(method, obj, f, budget_s=min(check_s, budget_s - spent))
        f = res.f
        obj.set_theta(res.theta)
        f, _ = ao.polish(obj, f, rounds=20)
        if np.size(obj.theta) and hasattr(obj, "nuisance_gh"):
            ao.newton_nuisance(obj, f, max_steps=5)  # the polished image moved: bring the nuisances to it
        rec = ev.evaluate(f, np.array(obj.theta, float))
        spent = time.perf_counter() - t0
        if rec["certified"]:
            break
    return f, np.array(obj.theta, float), rec, spent


def _restart_row(ev, f_ref, th_ref, f, th, rec, spent):
    cmp = compare_fits(ev, f_ref, f, theta_ref=th_ref, theta_other=th)
    return dict(cmp, certified=bool(rec["certified"]), seconds=float(spent), kkt_w=rec["kkt_w"])


def _far_field(obj, f, aperture_mas, n_ann=8):
    n, px = obj.prob.npix, obj.prob.pixel_mas
    c = (np.arange(n) - (n // 2 - 1)) * px
    r = np.hypot(c[:, None], c[None, :]).ravel()
    flux = float(np.sum(f))
    edges = np.linspace(0.0, float(np.max(np.abs(c))), n_ann + 1)
    prof = [float(np.sum(f[(r >= a) & (r < b)]) / flux) for a, b in zip(edges[:-1], edges[1:])]
    return dict(aperture_mas=float(aperture_mas), flux_beyond=float(np.sum(f[r > aperture_mas]) / flux), profile_edges_mas=edges.tolist(), profile_flux_fraction=prof)


def certify(prob, values, data, result, out):
    """C1 to C6 of plan section 5.4 at the selected rung: the signature ``dsharp_fit.py --certify module:function`` calls, ``certify(prob=, values=, data=, result=, out=)``.

    C1 the stop test (section 5.3 tests 1-4) at ``values``; C2 test 2 (far-field classes) plus the flux beyond ``ALMA_CERTIFY_APERTURE_MAS`` (default 0.8 of the half field) with its radial profile;
    C3 three restarts (the 1 arcsec Gaussian with floor image start, offsets +/-10 mas and gains +/-5% with opposite signs), taken from ``values_restart_<name>.npz`` in ``out`` when the driver
    wrote them (s5) and otherwise run here with the method named by ``result['args']['optimiser']`` if it is an S2 method, else ``lbfgsb``; C4 the converged image perturbed by 0.5 dex of
    log-normal noise and reconverged with G2's runner-up (``ALMA_CERTIFY_RUNNER_UP``, default ``lbfgsb``); C5 the direct NUFFT at eps 1e-12 against the Toeplitz loss (1e-10 relative);
    C6 ``at_bound_report`` and ``offset_bound_report``. Returns a JSON-able dict with ``certified`` (all of C1 to C6) and one entry per check; reconvergence is bounded by ``ALMA_CERTIFY_BUDGET_S``
    (default 1200 s) per restart. A failed check never raises: it is recorded and ``certified`` is false."""
    import os
    import pathlib

    from . import alma as va

    out = pathlib.Path(out)
    args = (result or {}).get("args", {}) or {}
    budget = float(os.environ.get(ENV_BUDGET, 1200.0))
    runner_up = os.environ.get(ENV_RUNNER_UP, "lbfgsb")
    main_method = args.get("optimiser") if args.get("optimiser") in ao.METHODS else "lbfgsb"
    for m in (runner_up, main_method):
        if m not in ao.METHODS:
            raise ValueError(f"{m!r} is not an S2 method: {sorted(ao.METHODS)}")
    obj = production_objective(prob, values)
    px, npix = obj.prob.pixel_mas, obj.prob.npix
    from scipy.ndimage import gaussian_filter

    sigma = float(os.environ.get(ENV_BEAM, 35.0)) / 2.3548 / px
    ev = Evaluator(obj, beam=lambda f: gaussian_filter(np.asarray(f, float).reshape(npix, npix), sigma, mode="constant"))
    f_ref, th_ref = _flux_pixels(values), np.array(obj.theta, float)
    ev.set_reference(f_ref, th_ref)
    rec = ev.evaluate(f_ref, th_ref)
    report = dict(method=main_method, runner_up=runner_up, budget_s=budget, provenance=provenance())
    report["C1"] = dict(passed=bool(rec["certified"]), tests=rec["tests"], kkt_w=rec["kkt_w"], decrement=rec["decrement"], decrement_threshold=rec["decrement_threshold"], nuisance_decrement=rec["nuisance_decrement"],
                        classes={k: rec[k] for k in ("res_hi", "res_mid", "res_low", "want_grow", "zero_violation")}, band_limit=rec.get("band_limit"))
    aperture = float(os.environ.get(ENV_APERTURE, 0.8 * npix * px / 2.0))
    report["C2"] = dict(passed=bool(rec["tests"]["classes"]), **_far_field(obj, f_ref, aperture))

    def restarts():
        rows, f0 = {}, {}
        f0["image_gaussian"], t0 = gaussian_floor_start(npix, px, obj.flux), th_ref
        plus, minus = nuisance_starts(obj, th_ref)
        starts = {"image_gaussian": (f0["image_gaussian"], th_ref), "nuisance_plus": (f_ref, plus), "nuisance_minus": (f_ref, minus)}
        for name, (fs, ts) in starts.items():
            saved = out / f"values_restart_{name}.npz"
            if saved.exists():
                z = dict(np.load(saved))
                th = np.concatenate([np.asarray(z[k], float).ravel() for k, _, _ in obj.layout]) if obj.layout else np.zeros(0)
                f, spent = np.asarray(z["pixels"], float).ravel() ** 2, 0.0
                obj.set_theta(th)
                rows[name] = _restart_row(ev, f_ref, th_ref, f, th, ev.evaluate(f, th), spent) | dict(source="driver")
            else:
                f, th, r, spent = _converge(obj, main_method, fs, ts, ev, budget_s=budget, check_s=min(120.0, budget))
                rows[name] = _restart_row(ev, f_ref, th_ref, f, th, r, spent) | dict(source=f"certify:{main_method}")
        return rows

    def guarded(label, fn):
        try:
            return fn()
        except Exception as e:  # a check that cannot run is a check that failed, recorded, never a silent pass
            return dict(passed=False, error=f"{type(e).__name__}: {e}", check=label)

    rows = guarded("C3", restarts)
    report["C3"] = dict(passed=("error" not in rows) and all(r["passes"] and r["certified"] for r in rows.values()), restarts=rows) if "error" not in rows else rows

    def perturbed():
        fp = perturb_lognormal(f_ref, 0.5, np.random.default_rng(0), flux=float(f_ref.sum()))
        f, th, r, spent = _converge(obj, runner_up, fp, th_ref, ev, budget_s=budget, check_s=min(120.0, budget))
        row = _restart_row(ev, f_ref, th_ref, f, th, r, spent)
        return dict(passed=bool(row["passes"] and row["certified"]), optimiser=runner_up, **row)

    report["C4"] = guarded("C4", perturbed)

    def direct():
        obj.set_theta(th_ref)
        toeplitz = 2.0 * obj.data_vg(f_ref)[0] + float(obj.term.chi2_constant)  # the full chi2: the objective leaves the constant out
        sc = va.score(data, prob.scene, {k: np.asarray(v) for k, v in values.items()}, prob.nuis, eps=1e-12)
        rel = abs(float(sc["chi2_raw"]) - toeplitz) / abs(toeplitz)
        return dict(passed=bool(rel < C5_RTOL), rel_diff=float(rel), chi2_direct=float(sc["chi2_raw"]), chi2_toeplitz=float(toeplitz), eps=1e-12)

    report["C5"] = guarded("C5", direct)

    def bounds():
        ab = va.at_bound_report(values, prob.priors, edge=C6_EDGE)
        ob = va.offset_bound_report(values.get("cal.offset"), prob.offset_bound_mas, getattr(prob, "offset_bound_requested", None), edge=C6_EDGE,
                                    bound_obs=getattr(prob, "offset_bound_obs", None), ref_obs=prob.nuis.ref_obs)
        return dict(passed=not ab["at_bound"] and ob.get("warning") is None, at_bound=ab, offset_bound=ob)

    report["C6"] = guarded("C6", bounds)
    report["certified"] = bool(all(report[c]["passed"] for c in ("C1", "C2", "C3", "C4", "C5", "C6")))
    return report
