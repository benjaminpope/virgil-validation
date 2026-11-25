"""The S2 optimiser candidates of the DSHARP convergence plan (``reports/dsharp-convergence-plan.md`` §4), on one objective, one start and one external evaluator.

Every method minimises the same convex-in-flux objective with the nuisances held fixed (tier A)::

    J(f) = 1/2 chi2(f) + (w / F) sum_i [ f_i ln(f_i / q) - f_i + q ],      q = F / n     (flat default, Skilling's form)

``f`` is the pixel flux (Jy), ``F`` the reference flux of the image and ``n`` the pixel count; the data term is ``1/2 f'Tf - r'f`` (``T`` the Toeplitz normal operator).
This is ``0.5 * chi2 + MaxEntropy(w)`` of virgil on a ``LinearImage(parametrization="sqrt")`` up to a constant (``alma_certify.VirgilObjective`` checks it).

**Interface.** ``run(name, objective, f0, budget_s=..., out=...)`` returns a ``RunResult``. A method never sees a stop test of its own: tolerances are zero, early
exits (line-search failure) restart the method from its last iterate, and the run ends only at the wall-clock budget, at the step cap (``max_steps``; a capped run is
*failed*, not converged) or when the method cannot move at all (``reason="stalled"``). The meter writes a snapshot ``snap_NNNN.npz`` of the iterate every ``snap_every`` seconds;
``alma_certify`` scores them afterwards, so its time is not charged. Time and gradient-equivalents (one Hessian-vector product counts 2) are counted by the meter.

An *objective* offers ``n``, ``flux``, ``weight``, ``data_vg(f)`` (value and gradient of the data term at the objective's current nuisances), ``hvp(v)`` (the data curvature ``T v``),
``diag()`` (``diag T``) and ``lipschitz()`` (largest eigenvalue of ``T``). Joint (tier B) objectives add ``theta``, ``set_theta`` and ``nuisance_gh(f)`` (gradient and dense Hessian of the
data term in the nuisance block). ``CirculantObjective`` is a self-contained one for tests and tuning on the laptop.
"""

import dataclasses
import pathlib
import time

import numpy as np
from scipy import linalg, optimize
from scipy.special import xlogy

TINY = 1e-300
METHODS = {}  # name -> function(objective, meter, f0, **options)
# Settings the tuning stage may vary (a method's own defaults when it does not).
DEFAULTS = {
    "lbfgsb": dict(maxcor=100),
    "lbfgs": dict(memory=20),
    "lbfgsb_newton": dict(maxcor=100, phase1=0.5, cg_tol=1e-3),
    "newton_cg": dict(cg_tol=1e-3),
    "skilling_bryan": dict(ndir=5, radius=0.1),
    "mirror": dict(step=1.0),
    "mirror_accel": dict(step=1.0),
    "fista": dict(step=1.0),
    "adam": dict(lr=0.05, beta1=0.9, beta2=0.999),
    "softmax_lbfgs": dict(maxcor=100),
}


class Budget(Exception):
    """The wall-clock budget is spent."""


class Cap(Exception):
    """The step cap was reached: the run is failed."""


class Stalled(Exception):
    """The method cannot move (a restart made no progress)."""


# ---------------------------------------------------------------- the objective


class Objective:
    """The shared base: entropy, total value and gradient. Subclasses give ``data_vg``, ``hvp``, ``diag``, ``lipschitz``."""

    theta = np.zeros(0)

    def __init__(self, n, flux, weight):
        self.n, self.flux, self.weight = int(n), float(flux), float(weight)
        self.q = self.flux / self.n

    # entropy: (w/F) sum f ln(f/q) - f + q, gradient (w/F) ln(f/q), Hessian (w/F)/f
    def entropy(self, f):
        return self.weight / self.flux * float(np.sum(xlogy(f, f / self.q) - f + self.q))

    def entropy_grad(self, f):
        return self.weight / self.flux * np.log(np.maximum(f, TINY) / self.q)

    def value_grad(self, f):
        d, g = self.data_vg(f)
        return d + self.entropy(f), g + self.entropy_grad(f)

    def flat(self):
        return np.full(self.n, self.q)

    def set_theta(self, theta):
        raise NotImplementedError("a joint objective sets its nuisances")


class CirculantObjective(Objective):
    """A shift-invariant normal operator on a periodic ``shape`` grid (``T`` has spectrum ``spectrum``, real and non-negative): a stand-in for the Toeplitz term, exact and cheap.
    ``r = T f_true + T^{1/2} eps``. Optional single gain ``theta[0]`` scaling the model (``chi2 = (y - g A f)'...``) makes the objective joint: the data term is
    ``1/2 g^2 f'Tf - g r'f + c`` with ``r`` fixed."""

    def __init__(self, shape, spectrum, r, flux, weight, gain=None):
        super().__init__(int(np.prod(shape)), flux, weight)
        self.shape, self.spectrum, self.r = tuple(shape), np.asarray(spectrum, float), np.asarray(r, float).ravel()
        self.theta = np.zeros(0) if gain is None else np.array([float(gain)])

    def _T(self, v):
        return np.fft.irfft2(np.fft.rfft2(np.asarray(v).reshape(self.shape)) * self.spectrum, s=self.shape).ravel()

    @property
    def gain(self):
        return 1.0 if self.theta.size == 0 else float(self.theta[0])

    def data_vg(self, f):
        Tf = self._T(f)
        g = self.gain
        return 0.5 * g * g * float(f @ Tf) - g * float(self.r @ f), g * g * Tf - g * self.r

    def hvp(self, v):
        return self.gain**2 * self._T(v)

    def diag(self):
        return np.full(self.n, self.gain**2 * float(self.spectrum_mean()))

    def spectrum_mean(self):
        full = np.fft.irfft2(self.spectrum, s=self.shape)  # the kernel; its centre value is T_ii
        return full[0, 0]

    def lipschitz(self):
        return self.gain**2 * float(self.spectrum.max())

    def dense(self):
        """The dense ``T`` (n x n), for the reference solution of the tests."""
        eye = np.eye(self.n)
        return np.stack([self._T(eye[i]) for i in range(self.n)], axis=1) * self.gain**2

    def set_theta(self, theta):
        self.theta = np.asarray(theta, float).copy()

    def theta_bounds(self):
        return np.full(self.theta.size, 1e-3), np.full(self.theta.size, np.inf)  # the gain stays positive

    def nuisance_gh(self, f):
        """Gradient and Hessian of the data term in the gain (one parameter): d/dg = g f'Tf - r'f, d2/dg2 = f'Tf."""
        Tf = self._T(f)
        ftf = float(f @ Tf)
        g = self.gain
        return np.array([g * ftf - float(self.r @ f)]), np.array([[ftf]])


# ---------------------------------------------------------------- the meter


class Meter:
    """Wraps the objective: counts gradient-equivalents (a Hessian-vector product is 2), enforces the budget and the step cap, writes snapshots of accepted iterates.
    Snapshot time is excluded from the wall clock. The evaluator, not this class, judges convergence."""

    def __init__(self, obj, *, budget_s, snap_every=30.0, out=None, max_steps=None, clock=time.perf_counter):
        self.obj, self.budget_s, self.snap_every, self.max_steps, self.clock = obj, float(budget_s), float(snap_every), max_steps, clock
        self.out = None if out is None else pathlib.Path(out)
        if self.out is not None:
            self.out.mkdir(parents=True, exist_ok=True)
        self.t0, self.paused, self.grads, self.hvps, self.steps = clock(), 0.0, 0, 0, 0
        self.last, self.last_snap, self.snaps, self.deadline = None, 0.0, [], None
        self.trace = []  # (t, gradient-equivalents) at every accepted iterate

    def elapsed(self):
        return self.clock() - self.t0 - self.paused

    @property
    def geq(self):
        return self.grads + 2 * self.hvps

    def _check(self):
        t = self.elapsed()
        if t > self.budget_s or (self.deadline is not None and t > self.deadline):
            raise Budget

    def vg(self, f):
        self._check()
        self.grads += 1
        return self.obj.value_grad(f)

    def data_vg(self, f):
        self._check()
        self.grads += 1
        return self.obj.data_vg(f)

    def hvp(self, v):
        self._check()
        self.hvps += 1
        return self.obj.hvp(v)

    def accept(self, f):
        self.last = np.array(f, float, copy=True)
        self.steps += 1
        t = self.elapsed()
        self.trace.append((t, self.geq))
        if self.out is not None and t - self.last_snap >= self.snap_every:
            self.snapshot()
        if self.max_steps is not None and self.steps >= self.max_steps:
            raise Cap

    def snapshot(self):
        if self.last is None or self.out is None:
            return
        t = self.elapsed()
        p = self.out / f"snap_{len(self.snaps):04d}.npz"
        c0 = self.clock()
        np.savez(p, f=self.last, theta=np.asarray(self.obj.theta, float), t=t, grads=self.grads, hvps=self.hvps, geq=self.geq, steps=self.steps)
        self.paused += self.clock() - c0
        self.last_snap = t
        self.snaps.append(dict(path=str(p), t=t, geq=self.geq, steps=self.steps))


@dataclasses.dataclass
class RunResult:
    method: str
    f: np.ndarray
    reason: str  # "budget", "capped", "stalled"
    wall: float
    grads: int
    hvps: int
    geq: int
    steps: int
    options: dict
    snapshots: list
    trace: list
    theta: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(0))

    @property
    def capped(self):
        return self.reason == "capped"


# ---------------------------------------------------------------- shared numerics


def _boundary_step(f, df, frac=0.9):
    """Largest ``alpha <= 1`` keeping ``f + alpha df`` at least ``1 - frac`` of the way from the boundary (never more than ``frac`` of the way to f = 0)."""
    neg = df < 0
    if not neg.any():
        return 1.0
    return float(min(1.0, frac * np.min(f[neg] / -df[neg])))


def _cg(matvec, b, precond, rtol, maxit, count=None):
    """Preconditioned CG for an SPD system to relative residual ``rtol``; returns ``(x, iterations, converged, relative residual)``."""
    x = np.zeros_like(b)
    r = b.copy()
    z = precond(r)
    p = z.copy()
    rz = float(r @ z)
    nb = float(np.linalg.norm(b))
    if nb == 0:
        return x, 0, True, 0.0
    for k in range(1, maxit + 1):
        Ap = matvec(p)
        denom = float(p @ Ap)
        if denom <= 0:
            return x, k, False, float(np.linalg.norm(r) / nb)
        a = rz / denom
        x += a * p
        r -= a * Ap
        rel = float(np.linalg.norm(r) / nb)
        if rel < rtol:
            return x, k, True, rel
        z = precond(r)
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, maxit, False, rel


def lambertw_exp(L):
    """``W(exp(L))`` for real ``L`` (any size, no overflow): Newton on ``w + ln w = L``."""
    L = np.asarray(L, float)
    w = np.where(L > 1.0, L - np.log(np.maximum(L, 1.0)), np.exp(np.minimum(L, 1.0)))
    w = np.maximum(w, TINY)
    for _ in range(40):
        step = w * (1.0 + L - np.log(w)) / (1.0 + w)
        new = np.maximum(step, TINY)
        if np.all(np.abs(new - w) <= 1e-15 * np.abs(new) + TINY):
            w = new
            break
        w = new
    return w


def entropy_prox(z, a, q):
    """``argmin_f a [f ln(f/q) - f] + 1/2 (f - z)^2`` elementwise (closed form: ``f = a W((q/a) exp(z/a))``), the entropy's proximal map for step weight ``a = t w / F``."""
    return a * lambertw_exp(np.log(q / a) + np.asarray(z, float) / a)


def _lipschitz(obj):
    return obj.lipschitz()


def _quad_data(obj, f, df):
    """Data-term value change ``g.df + 1/2 df'T df`` is not needed; this is the curvature along ``df`` (one product, uncounted: used only by tests)."""
    return float(df @ obj.hvp(df))


def newton_system(obj, f, g):
    """The scaled Newton system at ``f`` (plan §5.3): ``X = diag(sqrt f)``; matrix ``4 X T X + (4 w / F) I`` and right side ``-2 X g``; the step is ``df = 2 X delta``.
    Returns ``(matvec, b, precond)`` with the preconditioner ``1 / (4 f_i T_ii + 4 w / F)``."""
    X = np.sqrt(np.maximum(f, 0.0))
    shift = 4.0 * obj.weight / obj.flux
    diag = 4.0 * f * obj.diag() + shift
    return (lambda v: 4.0 * X * obj.hvp(X * v) + shift * v), -2.0 * X * g, (lambda r: r / diag)


# ---------------------------------------------------------------- the methods


def _restarting(run_once, meter, f0):
    """Run ``run_once(x0)`` (returns when its own line search gives up) from the last iterate until the budget or the step cap; two exits without progress mean ``Stalled``."""
    f = np.array(f0, float)
    strikes = 0
    while True:
        before = meter.steps
        f_new = run_once(f)
        if f_new is not None:
            f = f_new
        strikes = strikes + 1 if meter.steps - before <= 1 else 0
        if strikes >= 2:
            raise Stalled


def lbfgsb(obj, meter, f0, maxcor=100, **_):
    """scipy L-BFGS-B on ``x = sqrt f`` (the gate's method and coordinates), memory ``maxcor``; own stops off."""

    def once(f):
        def fun(x):
            J, g = meter.vg(x * x)
            return J, 2.0 * x * g

        res = optimize.minimize(fun, np.sqrt(f), jac=True, method="L-BFGS-B", callback=lambda xk: meter.accept(xk * xk),
                                options=dict(maxcor=int(maxcor), maxiter=10**9, maxfun=10**9, ftol=0.0, gtol=0.0, maxls=40))
        return res.x**2

    _restarting(once, meter, f0)


def lbfgs(obj, meter, f0, memory=20, **_):
    """L-BFGS in ``x = sqrt f`` with no box and a strong-Wolfe line search (scipy ``line_search``); a failed search resets the memory (the 'restarts' of the plan)."""
    x = np.sqrt(np.asarray(f0, float))
    S, Y = [], []

    cache = {}

    def vg(x):  # one evaluation serves the value and the gradient of the line search's trial point
        if cache.get("x") is None or not np.array_equal(cache["x"], x):
            cache["x"], cache["vg"] = x.copy(), meter.vg(x * x)
        return cache["vg"]

    def fun(x):
        return vg(x)[0]

    def jac(x):
        return 2.0 * x * vg(x)[1]

    J, g = meter.vg(x * x)
    gx = 2.0 * x * g
    strikes = 0
    while True:
        q = gx.copy()
        alphas = []
        for s, y in zip(reversed(S), reversed(Y)):
            a = (s @ q) / (y @ s)
            alphas.append(a)
            q -= a * y
        gamma = (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1]) if S else 1.0 / max(np.linalg.norm(gx), TINY)
        r = gamma * q
        for (s, y), a in zip(zip(S, Y), reversed(alphas)):
            r += s * (a - (y @ r) / (y @ s))
        p = -r
        if p @ gx >= 0:
            S, Y, p = [], [], -gx / max(np.linalg.norm(gx), TINY)
        alpha, _, _, Jn, _, gnew = optimize.line_search(fun, jac, x, p, gfk=gx, old_fval=J, c1=1e-4, c2=0.9, maxiter=30)
        if alpha is None or gnew is None:
            strikes += 1
            if S:
                S, Y = [], []
            if strikes >= 2:
                raise Stalled
            continue
        strikes = 0
        xn = x + alpha * p
        s, y = xn - x, gnew - gx
        if s @ y > 1e-300:
            S.append(s)
            Y.append(y)
            S, Y = S[-int(memory):], Y[-int(memory):]
        x, J, gx = xn, Jn, gnew
        meter.accept(x * x)


def _newton_cg_loop(obj, meter, f, cg_tol=1e-3, cg_maxit=300):
    """Truncated Newton-CG in the scaled flux coordinates of the plan, fraction-to-boundary and Armijo backtracking (steps in f, then 'mapped' to sqrt f by the caller's snapshots)."""
    J, g = meter.vg(f)
    strikes = 0
    while True:
        matvec, b, pre = newton_system(obj, f, g)
        delta, _, _, _ = _cg(matvec, b, pre, cg_tol, cg_maxit)
        df = 2.0 * np.sqrt(f) * delta
        if g @ df >= 0:
            df = -f * g / np.maximum(np.max(np.abs(f * g)), TINY) * np.min(f)  # steepest entropy-metric step as a fallback
        alpha = _boundary_step(f, df)
        ok = False
        for _ in range(40):
            fn = f + alpha * df
            Jn, gn = meter.vg(fn)
            if Jn <= J + 1e-4 * alpha * float(g @ df):
                ok = True
                break
            alpha *= 0.5
        if not ok:
            strikes += 1
            if strikes >= 2:
                raise Stalled
            continue
        strikes = 0
        f, J, g = fn, Jn, gn
        meter.accept(f)
        # the hvps of the CG were counted through ``meter.hvp`` via obj; see ``newton_cg`` which wires that


class _CountedHvp:
    """Routes the objective's ``hvp`` through the meter so a CG product is counted and respects the budget."""

    def __init__(self, obj, meter):
        self._obj, self._meter = obj, meter

    def __getattr__(self, name):
        return getattr(self._obj, name)

    def hvp(self, v):
        return self._meter.hvp(v)


def newton_cg(obj, meter, f0, cg_tol=1e-3, **_):
    """Truncated Newton-CG from ``f0`` (the plan starts it flat), CG to relative residual ``cg_tol`` on the scaled system."""
    _newton_cg_loop(_CountedHvp(obj, meter), meter, np.array(f0, float), cg_tol=cg_tol)


def lbfgsb_newton(obj, meter, f0, maxcor=100, phase1=0.5, cg_tol=1e-3, **_):
    """L-BFGS-B for the fraction ``phase1`` of the budget, then Newton-CG polish in flux coordinates (the first draft's choice)."""
    outer = meter.deadline  # a joint slice's deadline, else None (the whole budget): phase 1 takes ``phase1`` of what is left of it
    start = meter.elapsed()
    meter.deadline = start + phase1 * ((meter.budget_s if outer is None else min(outer, meter.budget_s)) - start)
    try:
        lbfgsb(obj, meter, f0, maxcor=maxcor)
    except Budget:
        pass
    except Stalled:
        pass
    finally:
        meter.deadline = outer
    f = meter.last if meter.last is not None else np.array(f0, float)
    _newton_cg_loop(_CountedHvp(obj, meter), meter, f, cg_tol=cg_tol)


def skilling_bryan(obj, meter, f0, ndir=5, radius=0.1, **_):
    """Skilling-Bryan subspace trust-region steps in the entropy metric ``M = diag(1/f)``: directions ``f.grad(data)``, ``f.grad(entropy)``, then ``f . (T e)`` of earlier ones,
    ``ndir`` in all, normalised in the metric; the quadratic model is solved exactly in the subspace with a multiplier ``mu`` that holds ``|df|_M^2 <= radius * F``."""
    ob = _CountedHvp(obj, meter)
    ndir = max(int(ndir), 2)
    f = np.array(f0, float)
    ell2 = float(radius) * obj.flux
    J, g = meter.vg(f)
    strikes = 0
    while True:
        dd = g - obj.entropy_grad(f)  # the data gradient, from the total (no extra product)
        ge = obj.entropy_grad(f)
        dirs = [f * dd, f * ge]
        Tdirs = []
        for k in range(int(ndir)):
            Tdirs.append(ob.hvp(dirs[k]))
            if len(dirs) < ndir:
                dirs.append(f * Tdirs[k])
        nrm = np.array([np.sqrt(np.sum(d * d / f)) for d in dirs])
        keep = nrm > 1e-12 * max(nrm.max(), 1e-300)  # the entropy gradient vanishes at a flat start: drop empty directions
        if not keep.any():
            raise Stalled
        E = np.stack([d for d, k in zip(dirs, keep) if k], axis=1) / nrm[keep]
        TE = np.stack([d for d, k in zip(Tdirs, keep) if k], axis=1) / nrm[keep]
        D = obj.weight / (obj.flux * f)
        Hs = E.T @ TE + E.T @ (D[:, None] * E)
        Hs = 0.5 * (Hs + Hs.T)
        Gm = E.T @ (E / f[:, None])
        gs = E.T @ g
        accepted = False
        for _ in range(30):
            mu = 0.0
            for _ in range(80):
                try:
                    c = -np.linalg.solve(Hs + mu * Gm, gs)
                except np.linalg.LinAlgError:
                    c = np.zeros(len(gs))
                if c @ Gm @ c <= ell2 or mu > 1e30:
                    break
                mu = 1e-12 if mu == 0 else mu * 4.0
            df = E @ c
            alpha = _boundary_step(f, df)
            fn = f + alpha * df
            Jn, gn = meter.vg(fn)
            pred = -(float(g @ df) * alpha + 0.5 * alpha**2 * float(c @ Hs @ c))
            if Jn < J:
                accepted = True
                if pred > 0 and (J - Jn) > 0.75 * pred:
                    ell2 *= 2.0
                break
            ell2 *= 0.25
        if not accepted:
            strikes += 1
            if strikes >= 2:
                raise Stalled
            continue
        strikes = 0
        f, J, g = fn, Jn, gn
        meter.accept(f)


def _step_L(obj, step):
    """Mirror and prox steps take ``t = step / L``; L is the data term's smoothness: ``lambda_max(T)`` (FISTA, Euclidean) or ``lambda_max(T) * F`` (mirror descent: relative to the
    entropy's Bregman distance on a flux-F image, by Pinsker's inequality)."""
    return float(step) / obj.lipschitz()


def mirror(obj, meter, f0, step=1.0, accel=False, base="diag", **_):
    """Entropic mirror descent: the exact proximal-Bregman step ``ln f+ = (ln f - eta d + c ln q) / (1 + c)``, ``c = eta w / F``. The KL divergence is ``1/F``-strongly convex in the
    l1 norm on a flux-``F`` set, so the matching smoothness constant of the quadratic data term is ``max|T_ij| = T_ii`` (``base="diag"``, the default): ``eta = step / (T_ii F)``;
    ``base="lambda"`` uses ``lambda_max(T)`` instead (the l2 constant, far more conservative). ``eta`` is scaled by ``F / max(||y||_1, F)`` where the iterate's flux exceeds ``F``.
    ``accel=True`` adds Nesterov momentum on ``ln f`` with a gradient restart that tests the **total** gradient and only once there is momentum (k > 1)."""
    L = float(obj.diag()[0]) if base == "diag" else float(obj.lipschitz())
    eta0 = float(step) / (L * obj.flux)
    lnq = np.log(obj.q)
    lf = np.log(np.asarray(f0, float))
    lf_prev = lf.copy()
    k = 0
    while True:
        k += 1
        beta = (k - 1.0) / (k + 2.0) if accel else 0.0
        ly = lf + beta * (lf - lf_prev)
        y = np.exp(ly)
        _, d = meter.data_vg(y)
        eta = eta0 * obj.flux / max(float(y.sum()), obj.flux)
        c = eta * obj.weight / obj.flux
        lf_new = (ly - eta * d + c * lnq) / (1.0 + c)
        f_new = np.exp(lf_new)
        if accel and k > 1 and float((d + obj.entropy_grad(y)) @ (f_new - np.exp(lf))) > 0:
            k = 0  # restart: the momentum pushed uphill (total gradient, so the entropy is in the test)
            lf_prev = lf.copy()
            continue
        lf_prev, lf = lf, lf_new
        meter.accept(f_new)


def mirror_accel(obj, meter, f0, step=1.0, **_):
    mirror(obj, meter, f0, step=step, accel=True)


def fista(obj, meter, f0, step=1.0, **_):
    """FISTA with the entropy as the prox (``entropy_prox``, Lambert W) and gradient adaptive restart; ``t = step / lambda_max(T)`` (power iteration on the normal operator)."""
    t = _step_L(obj, step)
    a = t * obj.weight / obj.flux
    f = np.array(f0, float)
    f_prev = f.copy()
    k = 0
    while True:
        k += 1
        y = f + (k - 1.0) / (k + 2.0) * (f - f_prev)
        y = np.maximum(y, TINY)
        _, d = meter.data_vg(y)
        f_new = entropy_prox(y - t * d, a, obj.q)
        if float((y - f_new) @ (f_new - f)) > 0:  # gradient-mapping restart
            k = 0
        f_prev, f = f, f_new
        meter.accept(f)


def adam(obj, meter, f0, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, **_):
    """Adam in float64 on ``x = sqrt f`` with a cosine schedule from ``lr`` (relative to the start's mean ``x``) to 1e-3 of it over the run's budget, **read off the meter's clock**
    (``meter.elapsed() / meter.budget_s``), so a method restarted in slices (tier B) keeps annealing; constant at the floor after."""
    x = np.sqrt(np.asarray(f0, float))
    scale = float(x.mean())
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    for k in range(1, 10**12):
        _, g = meter.vg(x * x)
        gx = 2.0 * x * g
        m = beta1 * m + (1 - beta1) * gx
        v = beta2 * v + (1 - beta2) * gx * gx
        frac = min(max(meter.elapsed() / meter.budget_s, 0.0), 1.0)
        lr_k = scale * lr * (1e-3 + (1 - 1e-3) * 0.5 * (1 + np.cos(np.pi * frac)))
        x = x - lr_k * (m / (1 - beta1**k)) / (np.sqrt(v / (1 - beta2**k)) + eps)
        meter.accept(x * x)


def softmax_lbfgs(obj, meter, f0, maxcor=100, **_):
    """Control: L-BFGS-B on softmax logits with a free log-flux, ``f = exp(xi) softmax(eta)`` (the old image parametrisation, now minimising the same unconstrained ``J`` as the others,
    so it can reach a certified point), to be scored under the new test."""

    def split(f):
        f = np.maximum(np.asarray(f, float), TINY)
        return np.concatenate([np.log(f / f.sum()), [np.log(f.sum())]])

    def image(z):
        e = z[:-1] - z[:-1].max()
        s = np.exp(e)
        s /= s.sum()
        return np.exp(z[-1]) * s, s

    def once(f):
        def fun(z):
            fl, s = image(z)
            J, g = meter.vg(fl)
            gs = fl * g
            return J, np.concatenate([gs - s * gs.sum(), [gs.sum()]])

        res = optimize.minimize(fun, split(f), jac=True, method="L-BFGS-B", callback=lambda z: meter.accept(image(z)[0]),
                                options=dict(maxcor=int(maxcor), maxiter=10**9, maxfun=10**9, ftol=0.0, gtol=0.0, maxls=40))
        return image(res.x)[0]

    _restarting(once, meter, np.asarray(f0, float))


METHODS.update(lbfgsb=lbfgsb, lbfgs=lbfgs, lbfgsb_newton=lbfgsb_newton, newton_cg=newton_cg, skilling_bryan=skilling_bryan, mirror=mirror,
               mirror_accel=mirror_accel, fista=fista, adam=adam, softmax_lbfgs=softmax_lbfgs)


# ---------------------------------------------------------------- nuisances (tier B) and the common entry point


def newton_nuisance(obj, f, *, max_steps=3, lm0=1e-9):
    """A damped dense Newton step (up to ``max_steps``) on the nuisance block at the image ``f``. The damped Hessian ``H + lm I`` must be positive definite (Cholesky; ``lm`` grows tenfold
    until it is), the trial point must lie inside the objective's ``theta_bounds()`` (if it has them) before the data term is evaluated, and the data value must not rise."""
    for _ in range(max_steps):
        g, H = obj.nuisance_gh(f)
        if not g.size:
            return
        v0 = obj.data_vg(f)[0]
        theta = obj.theta.copy()
        lo, hi = obj.theta_bounds() if hasattr(obj, "theta_bounds") else (None, None)
        lm = lm0 * max(np.abs(np.diag(H)).max(), 1e-300)
        for _ in range(40):
            try:
                cf = linalg.cho_factor(H + lm * np.eye(len(g)), lower=True)
            except (np.linalg.LinAlgError, linalg.LinAlgError, ValueError):
                lm = max(lm * 10, 1e-12)
                continue
            trial = theta - linalg.cho_solve(cf, g)
            inside = np.all(np.isfinite(trial)) and (lo is None or np.all(trial >= lo)) and (hi is None or np.all(trial <= hi))
            if inside:
                obj.set_theta(trial)
                if obj.data_vg(f)[0] <= v0:
                    break
            lm = max(lm * 10, 1e-12)
        else:
            obj.set_theta(theta)
            return


def run(name, obj, f0, *, budget_s, out=None, snap_every=30.0, max_steps=None, joint_rounds=20, clock=time.perf_counter, **options):
    """Run method ``name`` from ``f0`` on ``obj`` with its own stop disabled; ``RunResult``. A joint objective (one with ``nuisance_gh`` and a non-empty ``theta``) alternates the method
    for ``budget_s / joint_rounds`` with dense Newton steps on the nuisances (tier B). ``options`` override ``DEFAULTS[name]``."""
    opts = {**DEFAULTS.get(name, {}), **options}
    meter = Meter(obj, budget_s=budget_s, snap_every=snap_every, out=out, max_steps=max_steps, clock=clock)
    f = np.array(f0, float)
    joint = obj.theta.size > 0 and hasattr(obj, "nuisance_gh")
    reason = "budget"
    try:
        if not joint:
            METHODS[name](obj, meter, f, **opts)
        else:
            slice_s = meter.budget_s / int(joint_rounds)
            reserve = 0.0  # the last nuisance step's duration: each slice stops that early, so the final nuisance step ends inside the budget
            while meter.elapsed() + reserve < meter.budget_s:
                meter.deadline = min(meter.elapsed() + slice_s, meter.budget_s - reserve)
                try:
                    METHODS[name](obj, meter, f, **opts)
                except Budget:
                    pass
                except Stalled:
                    pass
                finally:
                    meter.deadline = None
                if meter.last is not None:
                    f = meter.last
                t1 = meter.elapsed()
                newton_nuisance(obj, f)
                reserve = max(reserve, 2.0 * (meter.elapsed() - t1))
                meter.accept(f)
    except Budget:
        pass
    except Cap:
        reason = "capped"
    except Stalled:
        reason = "stalled"
    if meter.last is None:
        meter.last = f
    meter.snapshot()
    return RunResult(name, meter.last, reason, meter.elapsed(), meter.grads, meter.hvps, meter.geq, meter.steps, opts, meter.snaps, meter.trace, np.asarray(obj.theta, float).copy())


def polish(obj, f, *, rounds=60, cg_tol=1e-6):
    """Newton-CG to a tight tolerance with no budget (the reference's polish): in flux coordinates, fraction-to-boundary, Armijo; returns the polished flux and the iterations used."""
    ob = obj
    f = np.array(f, float)
    J, g = ob.value_grad(f)
    for k in range(rounds):
        matvec, b, pre = newton_system(ob, f, g)
        delta, _, _, _ = _cg(matvec, b, pre, cg_tol, 2000)
        df = 2.0 * np.sqrt(f) * delta
        alpha = _boundary_step(f, df)
        for _ in range(40):
            fn = f + alpha * df
            Jn, gn = ob.value_grad(fn)
            if Jn <= J + 1e-4 * alpha * float(g @ df) + 1e-14 * abs(J):
                break
            alpha *= 0.5
        else:
            return f, k
        done = np.max(np.abs(fn - f) / np.maximum(f, TINY)) < 1e-13
        f, J, g = fn, Jn, gn
        if done:
            return f, k + 1
    return f, rounds
