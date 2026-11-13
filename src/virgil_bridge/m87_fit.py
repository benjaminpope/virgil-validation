"""virgil fits of the EHT M87* data of Lai et al. 2026 (design/m87_replication.md, stages 3-5).

The only module here that imports virgil for those fits. Three parts:

1. **Closure invariants in JAX** (stage 3). ``build_triangles`` lists, per scan (one time stamp), the triangles
   (r, a, b) of BASIS's closure-invariant likelihood: r is the first station of the array table present in the
   scan, a < b run over the other stations in table order, and a triangle is kept when its three baselines are
   present. ``ci_vector`` turns complex visibilities into the L2-normalised real invariants
   [Re(adv), Im(adv)] with adv = V_ra conj(V_ab)^-1 V_br, normalised per scan over both halves. A scan with N
   stations has (N-1)(N-2) = N^2 - 3N + 2 such numbers on its unit sphere, so N^2 - 3N + 1 independent ones.
2. **The xsringauss-equivalent virgil System** (stage 3): ``xsringauss_system`` maps BASIS's parameter names to
   ``Crescent`` + thick-side ``EllipticalGaussian`` + floor ``GaussianDisk`` (+ background ``GaussianDisk``) inside
   ``Blurred`` and ``Rotated``, plus two nuisance ``EllipticalGaussian``. ``Crescent`` and ``Blurred`` come with
   virgil PR #716 and are imported lazily, so this module imports on virgil main.
3. **Likelihoods and priors**: ``CIFit`` (a) diagonal sigma from BASIS, (b) the full covariance C = J Sigma J^T of
   the invariants, J from ``jax.jacfwd`` of ``ci_vector`` at the data, projected on the tangent space of the sphere
   because C is singular along the invariants themselves; ``BoxPrior`` (uniform and log-uniform boxes sampled in a
   logistic coordinate, for ``virgil.smc``); ``native_*`` builders for stage 4 and RML/CLEAN runners for stage 5.

Parameter conventions are read off BASIS's documentation and analytic visibilities (BASIS v0.1.0, MIT): the
mapping below is pinned by criterion 3a (invariants of a fixed model, BASIS against virgil, <= 1e-8), not assumed.
The frame is the data's: x = dra (East), y = ddec (North), V = sum I exp(-2 pi i (u x + v y)). In BASIS's own
crescent frame the bright (thick) side is +x', the crescent's axis is at angle ``phi`` counter-clockwise from +x,
so its position angle (North through East) is 90 deg - phi, and BASIS's y' is virgil's West (-dra) direction.
"""

import dataclasses
import math
import pathlib
import subprocess

import numpy as onp

FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))
UAS = 1e-3  # mas per microarcsecond
UAS_BASIS = 1e-3 * 206264.80624709636 / 206265.0  # mas per BASIS "microarcsecond": its models convert with 206265 arcsec per radian

# --------------------------------------------------------------------------------------------------------------
# BASIS parameter table (BASIS v0.1.0 MODEL_PARAMS: names, uniform bounds). ``basis_worker`` task "parameters"
# returns BASIS's own table, and the 3a test compares the two.
# --------------------------------------------------------------------------------------------------------------

XS_BOUNDS = {
    "I0": (0.05, 10.0), "Rp": (10.0, 40.0), "Rn_frac": (0.05, 0.95), "ecn": (0.0, 0.99), "f": (0.0, 1.0),
    "phi": (0.0, 2 * math.pi), "gax": (0.05, 2.0), "aq": (0.05, 1.0), "gq": (0.0, 0.99),
    "floor_sigma": (1.0, 100.0), "floor_brightness": (0.0, 2.0), "bg_gauss_flux": (0.0, 2.0),
    "bg_gauss_sigma": (1.0, 100.0), "gauss_blur_sigma": (0.0, 10.0),
}
G_BOUNDS = {
    "I0": (0.05, 10.0), "gauss_sigma": (5.0, 40.0), "aq": (0.05, 1.0), "pa": (0.0, math.pi),
    "xoff": (-80.0, 80.0), "yoff": (-80.0, 80.0), "gauss_blur_sigma": (0.0, 10.0),
}
BASIS_MODELS = ("xsringauss", "gauss", "gauss")


def basis_param_bounds(nuisance_blur=True):
    """``{name: (low, high)}`` of the free BASIS parameters, names ``"0_Rp"``, ``"1_xoff"`` ...

    ``nuisance_blur=False`` leaves out the two nuisance Gaussians' ``gauss_blur_sigma`` (the plan's "no
    per-Gaussian blur": an elliptical Gaussian blurred by a circle is again an elliptical Gaussian, so the blur is
    degenerate with its width and aspect ratio, and 26 parameters remain).
    """
    out = {f"0_{k}": v for k, v in XS_BOUNDS.items()}
    for i in (1, 2):
        out.update({f"{i}_{k}": v for k, v in G_BOUNDS.items() if nuisance_blur or k != "gauss_blur_sigma"})
    return out


def _virgil_models():
    """The virgil model classes, imported when first needed (``Crescent``/``Blurred`` need PR #716)."""
    import virgil.models as m

    missing = [n for n in ("Crescent", "Blurred") if not hasattr(m, n)]
    if missing:
        raise ImportError(f"virgil.models lacks {missing}: use a virgil main commit after PR #716 merged")
    return m


def crescent_available():
    """Whether this virgil has ``Crescent`` and ``Blurred`` (for ``skipif``)."""
    try:
        import virgil.models as m
    except ImportError:
        return False
    return hasattr(m, "Crescent") and hasattr(m, "Blurred")


def _ellipse(sigma, ratio, blur):
    """(sigma_major, ratio) of an elliptical Gaussian (std sigma, ratio) after a circular blur of std ``blur``."""
    import jax.numpy as jnp

    smaj = jnp.sqrt(sigma**2 + blur**2)
    smin = jnp.sqrt((ratio * sigma) ** 2 + blur**2)
    return smaj, smin / smaj


def xsringauss_system(p, nuisance_blur=True):
    """The virgil ``System`` equivalent to BASIS's ``xsringauss,gauss,gauss`` model.

    ``p`` maps BASIS parameter names (``"0_Rp"``, ``"1_gauss_sigma"`` ...; microarcseconds, radians, Jy) to
    numbers or JAX tracers. Positions are virgil's, in mas. The xsringauss is built with its thick side North and
    rotated by ``90 - phi`` degrees; BASIS's floor Gaussian sits at +d along the thick axis in its analytic
    visibility (and at -d in its rendered image: see ``basis_worker`` task "render_check"), which is what the
    likelihood uses and what is mapped here.
    """
    import jax.numpy as jnp

    m = _virgil_models()
    I0, Rp, Rnf, ecn, f, phi = (p[f"0_{k}"] for k in ("I0", "Rp", "Rn_frac", "ecn", "f", "phi"))
    gax, aq, gq = p["0_gax"], p["0_aq"], p["0_gq"]
    Rn = Rnf * Rp
    d = ecn * (Rp - Rn)
    a = gax * Rp
    parts = dict(
        crescent=m.Crescent(r_out=Rp * UAS_BASIS, r_in=Rn * UAS_BASIS, ecc=ecn, fade=f, pa=0.0, flux=(1.0 - gq) * I0),
        thick=m.EllipticalGaussian(
            fwhm=FWHM_PER_SIGMA * a * UAS_BASIS, ratio=aq, pa=90.0, flux=gq * I0, ddec=(Rn - d) * UAS_BASIS
        ),
        floor=m.GaussianDisk(sigma=p["0_floor_sigma"] * UAS_BASIS, flux=p["0_floor_brightness"], ddec=d * UAS_BASIS),
        background=m.GaussianDisk(sigma=p["0_bg_gauss_sigma"] * UAS_BASIS, flux=p["0_bg_gauss_flux"]),
    )
    ring = m.Rotated(
        m.Blurred(
            m.System(flux=I0 + p["0_floor_brightness"] + p["0_bg_gauss_flux"], **parts),
            fwhm=FWHM_PER_SIGMA * p["0_gauss_blur_sigma"] * UAS_BASIS,
        ),
        rotation_deg=90.0 - jnp.rad2deg(phi),
    )
    scene = dict(ring=ring)
    for i in (1, 2):
        sigma, ratio = p[f"{i}_gauss_sigma"], p[f"{i}_aq"]
        blur = p[f"{i}_gauss_blur_sigma"] if nuisance_blur else 0.0
        smaj, rat = _ellipse(sigma, ratio, blur)
        scene[f"nuisance{i}"] = m.EllipticalGaussian(
            fwhm=FWHM_PER_SIGMA * smaj * UAS_BASIS, ratio=rat, pa=jnp.rad2deg(p[f"{i}_pa"]),
            flux=p[f"{i}_I0"], dra=p[f"{i}_xoff"] * UAS_BASIS, ddec=p[f"{i}_yoff"] * UAS_BASIS,
        )
    return m.System(**scene)


def basis_defaults(nuisance_blur=True):
    """A mid-bounds parameter dict (for tests and starting points); the blur of the nuisances 0 if not free."""
    p = {k: 0.5 * (lo + hi) for k, (lo, hi) in basis_param_bounds(True).items()}
    if not nuisance_blur:
        p["1_gauss_blur_sigma"] = p["2_gauss_blur_sigma"] = 0.0
    return p


def model_vis(system, u, v):
    """Visibilities of a virgil model at (u, v) in wavelengths (virgil normalises V(0) = 1)."""
    return system.model(u, v, 1.0)


# --------------------------------------------------------------------------------------------------------------
# Closure invariants
# --------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Triangles:
    """Triangles (r, a, b) per scan, as indices into the baseline arrays; static (numpy) data.

    ``i1, i2, i3`` index the baselines (r, a), (a, b), (b, r); ``c1, c2, c3`` are True where the stored baseline is
    the reverse of the one the triangle needs (so its visibility is conjugated). ``scan`` numbers the scans
    0..n_scan-1 in time order; ``labels`` holds (scan time, r, a, b) station ids per triangle.
    """

    i1: onp.ndarray
    i2: onp.ndarray
    i3: onp.ndarray
    c1: onp.ndarray
    c2: onp.ndarray
    c3: onp.ndarray
    scan: onp.ndarray
    n_scan: int
    times: onp.ndarray
    labels: onp.ndarray

    @property
    def n_tri(self):
        return len(self.scan)

    def n_per_scan(self):
        """Triangles per scan."""
        return onp.bincount(self.scan, minlength=self.n_scan)

    def n_independent(self):
        """Independent invariants per scan: 2 * triangles - 1 (N^2 - 3N + 1 for a full scan of N stations)."""
        return 2 * self.n_per_scan() - 1


def n_invariants(n_stations):
    """(N-1)(N-2) = N^2 - 3N + 2 normalised real numbers per full scan of N stations; N^2 - 3N + 1 independent."""
    return (n_stations - 1) * (n_stations - 2)


def n_independent_invariants(n_stations):
    """N^2 - 3N + 1: the normalised invariants of a full scan lie on a unit sphere."""
    return n_stations**2 - 3 * n_stations + 1


def build_triangles(time, ant1, ant2, order=None, time_decimals=3):
    """The triangles of each scan, pinned on the first station of ``order`` present, as BASIS does.

    ``time`` (seconds), ``ant1``, ``ant2`` give one entry per baseline row. A scan is one distinct ``time``
    (rounded to ``time_decimals``). ``order`` ranks the stations (a sequence of ids in array-table order; default
    ascending id): the reference is the first of them present at that time and a < b follow the same order.
    Where a baseline occurs twice in a scan the last row wins (BASIS's dictionary does the same).
    """
    time, ant1, ant2 = onp.asarray(time, float), onp.asarray(ant1).astype(int), onp.asarray(ant2).astype(int)
    rank = {int(s): i for i, s in enumerate(order)} if order is not None else None
    key = lambda s: rank[int(s)] if rank is not None else int(s)  # noqa: E731
    tkey = onp.round(time, time_decimals)
    uniq = onp.unique(tkey)
    cols = {k: [] for k in ("i1", "i2", "i3", "c1", "c2", "c3", "scan")}
    labels = []
    keep_times = []
    for t in uniq:
        rows = onp.nonzero(tkey == t)[0]
        lookup = {}
        for r in rows:
            lookup[(int(ant1[r]), int(ant2[r]))] = (r, False)
            lookup[(int(ant2[r]), int(ant1[r]))] = (r, True)
        stations = sorted({int(s) for s in ant1[rows]} | {int(s) for s in ant2[rows]}, key=key)
        if len(stations) < 3:
            continue
        ref, rest = stations[0], stations[1:]
        found = 0
        sid = len(keep_times)
        for ia in range(len(rest)):
            for ib in range(ia + 1, len(rest)):
                a, b = rest[ia], rest[ib]
                legs = [lookup.get(pair) for pair in ((ref, a), (a, b), (b, ref))]
                if any(leg is None for leg in legs):
                    continue
                for n, (r, c) in enumerate(legs, 1):
                    cols[f"i{n}"].append(r)
                    cols[f"c{n}"].append(c)
                cols["scan"].append(sid)
                labels.append((t, ref, a, b))
                found += 1
        if found:
            keep_times.append(t)
    arr = {k: onp.asarray(v) for k, v in cols.items()}
    return Triangles(
        arr["i1"].astype(int), arr["i2"].astype(int), arr["i3"].astype(int),
        arr["c1"].astype(bool), arr["c2"].astype(bool), arr["c3"].astype(bool),
        arr["scan"].astype(int), len(keep_times), onp.asarray(keep_times, float), onp.asarray(labels, float),
    )


def advariants(vis, tri):
    """Complex advariants V_ra conj(V_ab)^-1 V_br of the triangles of ``tri``, from baseline visibilities."""
    import jax.numpy as jnp

    vis = jnp.asarray(vis)
    cj = lambda x, c: jnp.where(c, jnp.conj(x), x)  # noqa: E731
    v1 = cj(vis[tri.i1], tri.c1)
    v2 = cj(vis[tri.i2], tri.c2)
    v3 = cj(vis[tri.i3], tri.c3)
    return v1 * v3 / jnp.conj(v2)


def ci_vector(vis, tri):
    """L2-normalised real closure invariants ``[Re(adv), Im(adv)]`` (length 2 * n_tri).

    Each scan's entries (both halves) are divided by their L2 norm, as BASIS's ``advariants_to_ci``. Invariant
    under ``V_ij -> g_i conj(g_j) V_ij`` for any complex station gains.
    """
    import jax
    import jax.numpy as jnp

    adv = advariants(vis, tri)
    sq = adv.real**2 + adv.imag**2
    norm = jnp.sqrt(jax.ops.segment_sum(sq, tri.scan, num_segments=tri.n_scan))[tri.scan]
    return jnp.concatenate([adv.real / norm, adv.imag / norm])


def apply_station_gains(vis, ant1, ant2, gains):
    """``g[ant1] conj(g[ant2]) vis``: the corruption closure invariants are blind to."""
    gains = onp.asarray(gains)
    return onp.asarray(vis) * gains[onp.asarray(ant1)] * onp.conj(gains[onp.asarray(ant2)])


# --------------------------------------------------------------------------------------------------------------
# Likelihoods on closure invariants
# --------------------------------------------------------------------------------------------------------------


def ci_jacobian(vis, tri):
    """d(ci_vector)/d[Re V, Im V] at ``vis``: shape (2 n_tri, 2 n_vis), by ``jax.jacfwd``."""
    import jax
    import jax.numpy as jnp

    n = len(vis)

    def fn(x):
        return ci_vector(x[:n] + 1j * x[n:], tri)

    x0 = jnp.concatenate([jnp.asarray(vis).real, jnp.asarray(vis).imag])
    return onp.asarray(jax.jacfwd(fn)(x0))


def ci_covariance(vis, weight, tri):
    """Full covariance C = J Sigma J^T of the invariants for independent noise of variance 1/weight on each of
    Re V and Im V (virgil's ``ComplexVisData.weight``), J at the data. Singular along each scan's invariant
    vector (its norm is 1): see ``whitening``."""
    J = ci_jacobian(vis, tri)
    var = onp.concatenate([1.0 / onp.asarray(weight), 1.0 / onp.asarray(weight)])
    return (J * var) @ J.T


def _scan_index(tri):
    """Row indices (into the 2 n_tri vector) of each scan: Re entries then Im entries."""
    n = tri.n_tri
    return [onp.concatenate([onp.nonzero(tri.scan == s)[0], n + onp.nonzero(tri.scan == s)[0]]) for s in range(tri.n_scan)]


def whitening(C, ci, tri):
    """Whitening of the tangent-space residuals of each scan.

    For scan s with invariants x (length n, unit norm) take Q, an orthonormal basis of x's orthogonal complement
    (n-1 columns); C_red = Q^T C_s Q is positive definite, L its Cholesky factor. Returns (W, logdet, k): the block
    diagonal matrix W (sum(n-1) x 2 n_tri) with W r = L^-1 Q^T r_s stacked over scans, the summed log det C_red and
    the number k of independent invariants. The Gaussian log likelihood of a residual r (data minus model) is
    -0.5 (|W r|^2 + logdet + k log 2 pi).
    """
    import scipy.linalg as sl

    ci = onp.asarray(ci)
    blocks, logdet, k = [], 0.0, 0
    idx = _scan_index(tri)
    W = onp.zeros((sum(len(i) - 1 for i in idx), len(ci)))
    row = 0
    for ii in idx:
        x = ci[ii]
        n = len(ii)
        q, _ = onp.linalg.qr(onp.column_stack([x, onp.eye(n)]))
        Q = q[:, 1:n]
        red = Q.T @ C[onp.ix_(ii, ii)] @ Q
        L = onp.linalg.cholesky(0.5 * (red + red.T))
        W[row : row + n - 1, ii] = sl.solve_triangular(L, Q.T, lower=True)
        logdet += 2.0 * onp.sum(onp.log(onp.diag(L)))
        row += n - 1
        k += n - 1
        blocks.append(n - 1)
    return W, float(logdet), int(k)


@dataclasses.dataclass
class CIFit:
    """Closure-invariant likelihoods of one observing day for virgil models.

    Build with ``CIFit.from_arrays``. ``sigma`` is the diagonal standard deviation of each invariant (BASIS's, in
    ``[Re..., Im...]`` order); ``loglike_diag`` is BASIS's likelihood (a), ``loglike_full`` the one with the full
    covariance (b). ``chi2_raw`` first: chi^2/N on the quoted sigma.
    """

    u: onp.ndarray
    v: onp.ndarray
    vis: onp.ndarray
    weight: onp.ndarray
    tri: Triangles
    ci: onp.ndarray
    sigma: onp.ndarray = None
    W: onp.ndarray = None
    logdet: float = 0.0
    k: int = 0

    @classmethod
    def from_arrays(cls, u, v, vis, weight, time, ant1, ant2, order=None, sigma=None, full=True):
        tri = build_triangles(time, ant1, ant2, order=order)
        vis = onp.asarray(vis, complex)
        ci = onp.asarray(ci_vector(vis, tri))
        self = cls(onp.asarray(u, float), onp.asarray(v, float), vis, onp.asarray(weight, float), tri, ci, sigma)
        if full:
            self.W, self.logdet, self.k = whitening(ci_covariance(vis, weight, tri), ci, tri)
        return self

    def model_ci(self, system):
        import jax.numpy as jnp

        return ci_vector(model_vis(system, jnp.asarray(self.u), jnp.asarray(self.v)), self.tri)

    def loglike_diag(self, system):
        """BASIS's Gaussian log likelihood on the diagonal sigma: -0.5 sum[(r/sigma)^2 + log(2 pi sigma^2)]."""
        import jax.numpy as jnp

        r = (jnp.asarray(self.ci) - self.model_ci(system)) / self.sigma
        return -0.5 * (jnp.sum(r**2) + jnp.sum(jnp.log(2 * jnp.pi * jnp.asarray(self.sigma) ** 2)))

    def loglike_full(self, system):
        """Log likelihood with the full covariance, on each scan's tangent space."""
        import jax.numpy as jnp

        r = jnp.asarray(self.W) @ (jnp.asarray(self.ci) - self.model_ci(system))
        return -0.5 * (jnp.sum(r**2) + self.logdet + self.k * jnp.log(2 * jnp.pi))

    def chi2_raw(self, system):
        """(chi^2, N) on the quoted diagonal sigma, at the model."""
        r = (self.ci - onp.asarray(self.model_ci(system))) / self.sigma
        return float(onp.sum(r**2)), int(len(r))

    def chi2_full(self, system):
        """(chi^2, k): the tangent-space chi^2 with the full covariance and its number of independent invariants."""
        r = self.W @ (self.ci - onp.asarray(self.model_ci(system)))
        return float(onp.sum(r**2)), int(self.k)


# --------------------------------------------------------------------------------------------------------------
# Priors for virgil.smc: boxes (uniform or log-uniform) sampled in a logistic coordinate
# --------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass
class BoxPrior:
    """Independent uniform / log-uniform priors on boxes, as the flat vector ``virgil.smc`` wants.

    ``spec`` is a list of ``(name, low, high, kind)`` with kind ``"uniform"`` or ``"log"`` (log-uniform, a Jeffreys
    prior for a scale). The flat coordinate is z with x = low + (high - low) sigmoid(z) (in log x for ``"log"``),
    so a uniform prior in x has a logistic density in z and every z is valid. Fixed values go in ``fixed``.
    """

    spec: list
    fixed: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.names = [s[0] for s in self.spec]
        self.lo = onp.array([math.log(s[1]) if s[3] == "log" else s[1] for s in self.spec], float)
        self.hi = onp.array([math.log(s[2]) if s[3] == "log" else s[2] for s in self.spec], float)
        self.is_log = onp.array([s[3] == "log" for s in self.spec])

    @property
    def dim(self):
        return len(self.spec)

    def to_values(self, z):
        """Dict of constrained values (JAX arrays) for flat z, with the fixed values added."""
        import jax
        import jax.numpy as jnp

        u = jax.nn.sigmoid(z)
        raw = jnp.asarray(self.lo) + (jnp.asarray(self.hi) - jnp.asarray(self.lo)) * u
        x = jnp.where(jnp.asarray(self.is_log), jnp.exp(raw), raw)
        out = dict(self.fixed)
        out.update({n: x[i] for i, n in enumerate(self.names)})
        return out

    def logprior(self, z):
        """Log density of z (the logistic Jacobian): the prior in x is uniform on its box (or in log x)."""
        import jax.numpy as jnp

        return jnp.sum(-jnp.abs(z) - 2.0 * jnp.log1p(jnp.exp(-jnp.abs(z))))

    def init_particles(self, key, n):
        """n exact draws of z from the prior."""
        import jax
        import jax.numpy as jnp

        u = jax.random.uniform(key, (n, self.dim), minval=1e-6, maxval=1 - 1e-6)
        return jnp.log(u) - jnp.log1p(-u)


def basis_box_prior(nuisance_blur=True):
    """The uniform prior of BASIS's ``runBilby.py`` on its bounds (stage 3 priors)."""
    spec = [(k, lo, hi, "uniform") for k, (lo, hi) in basis_param_bounds(nuisance_blur).items()]
    fixed = {} if nuisance_blur else {"1_gauss_blur_sigma": 0.0, "2_gauss_blur_sigma": 0.0}
    return BoxPrior(spec, fixed)


# --------------------------------------------------------------------------------------------------------------
# Stage 3 runner
# --------------------------------------------------------------------------------------------------------------


def smc_run(prior, loglike_of_values, key, n_particles=1000, max_seconds=float("inf"), **kw):
    """``virgil.smc.tempered_smc`` on ``prior`` with ``loglike_of_values(values_dict)``.

    Returns the ``SMCResult`` and the particles as a dict name -> (n,) numpy array of constrained values.
    """
    import jax
    import jax.numpy as jnp

    from virgil.smc import tempered_smc

    loglike = jax.jit(lambda z: loglike_of_values(prior.to_values(z)))
    res = tempered_smc(
        prior.logprior, loglike, prior.init_particles, key=key, n_particles=n_particles,
        max_seconds=max_seconds, **kw,
    )
    out = None
    if res.particles is not None:
        vals = jax.vmap(prior.to_values)(jnp.asarray(res.particles))
        out = {k: onp.asarray(v) for k, v in vals.items()}
    return res, out


def weighted_summary(samples, weights=None):
    """``{name: {"median", "sd", "q16", "q84"}}`` of weighted samples (weights sum to 1; None for equal)."""
    out = {}
    for k, x in samples.items():
        x = onp.asarray(x, float)
        if x.ndim != 1 or len(x) < 2:
            continue
        w = onp.full(len(x), 1.0 / len(x)) if weights is None else onp.asarray(weights, float)
        order = onp.argsort(x)
        cw = onp.cumsum(w[order])
        q = lambda p: float(x[order][min(onp.searchsorted(cw, p), len(x) - 1)])  # noqa: E731
        mean = float(onp.sum(w * x))
        out[k] = {"median": q(0.5), "sd": float(math.sqrt(onp.sum(w * (x - mean) ** 2))), "q16": q(0.16), "q84": q(0.84)}
    return out


def resample(samples, weights, n, seed=0):
    """n draws (with replacement) from weighted samples, as a dict of arrays."""
    rng = onp.random.default_rng(seed)
    w = onp.asarray(weights, float)
    idx = rng.choice(len(w), size=n, p=w / w.sum())
    return {k: onp.asarray(v)[idx] for k, v in samples.items()}


def render_posterior(build, samples, n=200, npix=128, fov_uas=225.0, seed=0, weights=None):
    """Unit-sum images (row 0 = North, column 0 = East) of ``n`` posterior draws; ``build(values) -> System``.

    The pixel scale is ``fov_uas / npix`` microarcseconds.
    """
    if weights is not None:
        samples = resample(samples, weights, n, seed)
    total = len(next(iter(samples.values())))
    pick = onp.random.default_rng(seed).choice(total, size=min(n, total), replace=False)
    return onp.stack([
        onp.asarray(build({k: v[i] for k, v in samples.items()}).render(npix=npix, fov_mas=fov_uas * UAS)) for i in pick
    ])


def ring_summary(images, fov_uas=225.0):
    """REX features (``crosscheck.ring_extraction.ring_features``) of each image: mean and SD over images.

    Keys as ring_features returns them (diameter, width, pa, asymmetry, central_fraction; micro-arcseconds and
    degrees). The position angle is averaged as a circular mean.
    """
    from crosscheck.ring_extraction import ring_features

    scale = fov_uas / images.shape[-1]
    rows = [ring_features(img, scale) for img in images]
    out = {}
    for k in ("diameter", "width", "pa", "asymmetry", "central_fraction"):
        x = onp.array([r[k] for r in rows], float)
        if k == "pa":
            ang = onp.deg2rad(x)
            mean = math.degrees(math.atan2(onp.mean(onp.sin(ang)), onp.mean(onp.cos(ang)))) % 360.0
            d = (x - mean + 180.0) % 360.0 - 180.0
            out[k] = {"mean": float(mean), "sd": float(onp.std(d))}
        else:
            out[k] = {"mean": float(onp.mean(x)), "sd": float(onp.std(x))}
    return out


def git_commit(path):
    """Full HEAD commit of the checkout containing ``path`` (``"+dirty"`` appended if it has local changes)."""
    path = pathlib.Path(path)
    try:
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return None
    return (head + ("+dirty" if dirty else "")) if head else None


# --------------------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------------------


def load_vis(path):
    """A ``ComplexVisData`` from a directory of ``ddid_NN.npz`` (written by ``scripts/m87_uvfits_to_ms.py``; OzSTAR
    has no casatools), from ``eht_reading.npz`` (``basis_worker`` task ``dump_uvfits``: eht-imaging's own reading
    of the UVFITS, for checks without CASA) or from a measurement set (``*.ms``, needs casatools and a parallel-hand
    pair: not a Stokes I MS)."""
    path = pathlib.Path(path).expanduser()
    if path.is_dir() and list(path.glob("ddid_*.npz")):
        from virgil_bridge.alma import ddid_files, load_npz

        return load_npz(ddid_files(path))
    from virgil.radio import ComplexVisData

    if path.suffix == ".npz":
        z = dict(onp.load(path))
        return ComplexVisData.from_arrays(
            z["u"], z["v"], z["vis"], 1.0 / z["sigma"] ** 2, float(z.get("rf_hz", 2.3e11)), time=z["time"],
            ant1=z["t1"], ant2=z["t2"],
        )
    return ComplexVisData.read_ms(str(path))


def ci_fit_from_data(data, sigma=None, full=True):
    """``CIFit`` of a ``ComplexVisData`` (needs ``time``, ``ant1``, ``ant2``), stations ranked by id."""
    a = {k: onp.asarray(getattr(data, k)) for k in ("u", "v", "vis", "weight", "time", "ant1", "ant2")}
    return CIFit.from_arrays(a["u"], a["v"], a["vis"], a["weight"], a["time"], a["ant1"], a["ant2"], sigma=sigma, full=full)


def match_labels(fit, labels, tol_s=0.5):
    """Index into BASIS's invariants for each triangle of ``fit.tri``.

    ``labels`` (n, 4) = (time in MJD seconds, ref, a, b) station ids (indices of BASIS's array table) of BASIS's
    triangles. Times match within ``tol_s`` and stations exactly. Raises ``ValueError`` if any triangle is missing
    or the two lists differ in length (so the pick is a permutation).
    """
    labels = onp.asarray(labels, float)
    own = fit.tri.labels
    if len(own) != len(labels):
        raise ValueError(f"{len(own)} triangles here but {len(labels)} in BASIS")
    buckets = {}
    for i, (t, r, a, b) in enumerate(labels):
        buckets.setdefault((int(r), int(a), int(b)), []).append((t, i))
    pick = []
    for t, r, a, b in own:
        hit = [i for tt, i in buckets.get((int(r), int(a), int(b)), []) if abs(tt - t) <= tol_s]
        if len(hit) != 1:
            raise ValueError(f"triangle ({int(r)}, {int(a)}, {int(b)}) at t={t:.3f}: {len(hit)} matches in BASIS")
        pick.append(hit[0])
    pick = onp.asarray(pick)
    if len(set(pick.tolist())) != len(pick):
        raise ValueError("BASIS triangles matched more than once")
    return pick


def reorder(values, pick):
    """BASIS's ``[Re..., Im...]`` vector in this module's triangle order (``values[pick]`` for each half)."""
    values = onp.asarray(values, float)
    n = len(pick)
    return onp.concatenate([values[pick], values[n + pick]])


# --------------------------------------------------------------------------------------------------------------
# Stage 4: native models (complex visibilities, marginalised antenna gains, Jeffreys priors)
# --------------------------------------------------------------------------------------------------------------

# Priors in micro-arcseconds / fractions: log-uniform scales, uniform angles and bounded shape parameters.
RING_SPEC = [  # shared by both stage 4 models: ring-like part
    ("R", 5.0, 40.0, "log"), ("w", 1.0, 20.0, "log"), ("dx", -30.0, 30.0, "uniform"), ("dy", -30.0, 30.0, "uniform"),
]
MRING_SPEC = [
    ("beta1", 0.0, 0.5, "uniform"), ("phi1", 0.0, 2 * math.pi, "uniform"),
    ("beta2", 0.0, 0.5, "uniform"), ("phi2", 0.0, 2 * math.pi, "uniform"),
]
COMPOSITE_SPEC = [
    ("Rp", 5.0, 40.0, "log"), ("Rn_frac", 0.05, 0.95, "uniform"), ("ecn", 0.0, 0.99, "uniform"),
    ("f", 0.0, 1.0, "uniform"), ("phi", 0.0, 2 * math.pi, "uniform"), ("gax", 0.05, 2.0, "log"),
    ("aq", 0.05, 1.0, "uniform"), ("gq", 0.0, 0.99, "uniform"), ("floor_sigma", 1.0, 100.0, "log"),
    ("floor_frac", 1e-3, 3.0, "log"), ("blur_sigma", 0.5, 10.0, "log"),
]
BACKGROUND_SPEC = [("bg_sigma", 30.0, 150.0, "log"), ("bg_frac", 1e-3, 3.0, "log")]
NUISANCE_SPEC = [  # per nuisance Gaussian i (relative flux to the ring)
    ("frac", 1e-3, 1.0, "log"), ("sigma", 5.0, 40.0, "log"), ("aq", 0.05, 1.0, "uniform"),
    ("pa", 0.0, math.pi, "uniform"), ("x", -80.0, 80.0, "uniform"), ("y", -80.0, 80.0, "uniform"),
]


def _prefixed(spec, prefix):
    return [(f"{prefix}{n}", lo, hi, k) for n, lo, hi, k in spec]


def native_spec(model, n_nuis):
    """Prior ``spec`` for ``BoxPrior`` of stage 4 ``model`` ("mring" or "composite") with ``n_nuis`` Gaussians."""
    if model == "mring":
        spec = RING_SPEC + MRING_SPEC + BACKGROUND_SPEC
    elif model == "composite":
        spec = COMPOSITE_SPEC + BACKGROUND_SPEC
    else:
        raise ValueError(f"model must be 'mring' or 'composite', not {model!r}")
    for i in range(1, n_nuis + 1):
        spec = spec + _prefixed(NUISANCE_SPEC, f"n{i}_")
    return spec


def _nuisance_gaussians(m, v, n_nuis):
    import jax.numpy as jnp

    out = {}
    for i in range(1, n_nuis + 1):
        k = lambda s: v[f"n{i}_{s}"]  # noqa: E731
        out[f"nuisance{i}"] = m.EllipticalGaussian(
            fwhm=FWHM_PER_SIGMA * k("sigma") * UAS, ratio=k("aq"), pa=jnp.rad2deg(k("pa")), flux=k("frac"),
            dra=k("x") * UAS, ddec=k("y") * UAS,
        )
    return out


def native_scene(model, n_nuis):
    """Function ``values -> System`` (unit total flux; ``cal.flux`` sets the scale) for a stage 4 model."""
    import jax.numpy as jnp

    def scene(v):
        import virgil.models as m

        if model == "mring":
            c = (v["beta1"] * jnp.cos(v["phi1"]), v["beta2"] * jnp.cos(v["phi2"]))
            s = (v["beta1"] * jnp.sin(v["phi1"]), v["beta2"] * jnp.sin(v["phi2"]))
            ring = m.MultipoleDisk(
                m.GaussianRing(radius=v["R"] * UAS, sigma=v["w"] * UAS), harmonics=m.Harmonics(c=c, s=s),
                dra=v["dx"] * UAS, ddec=v["dy"] * UAS, flux=1.0,
            )
            parts = dict(ring=ring)
        else:
            parts = dict(ring=composite_ring(v))
        parts["background"] = m.GaussianDisk(sigma=v["bg_sigma"] * UAS, flux=v["bg_frac"])
        parts.update(_nuisance_gaussians(m, v, n_nuis))
        return m.System(**parts)

    return scene


def composite_ring(v):
    """The stage 3 composite as a unit-flux group: Crescent + thick-side Gaussian + floor Gaussian, blurred and
    rotated, parameters as ``COMPOSITE_SPEC`` (floor flux relative to the ring, as a fraction ``floor_frac``)."""
    import jax.numpy as jnp

    m = _virgil_models()
    Rp, Rn = v["Rp"], v["Rn_frac"] * v["Rp"]
    d = v["ecn"] * (Rp - Rn)
    parts = dict(
        crescent=m.Crescent(r_out=Rp * UAS, r_in=Rn * UAS, ecc=v["ecn"], fade=v["f"], pa=0.0, flux=1.0 - v["gq"]),
        thick=m.EllipticalGaussian(
            fwhm=FWHM_PER_SIGMA * v["gax"] * Rp * UAS, ratio=v["aq"], pa=90.0, flux=v["gq"], ddec=(Rn - d) * UAS
        ),
        floor=m.GaussianDisk(sigma=v["floor_sigma"] * UAS, flux=v["floor_frac"], ddec=d * UAS),
    )
    inner = m.System(flux=1.0 + v["floor_frac"], **parts)
    return m.Rotated(
        m.Blurred(inner, fwhm=FWHM_PER_SIGMA * v["blur_sigma"] * UAS), rotation_deg=90.0 - jnp.rad2deg(v["phi"])
    )


def gain_nuisance(data, sigmas=True, **kw):
    """``CalibrationNuisance`` of one-observation EHT data: per-scan marginalised antenna gains (amplitude width
    0.5, phase width 3 rad: broad, as a priori station calibration of the HOPS network is uncertain), flux, and
    ``cal.sigma`` per ddid; no offsets, no spectral index."""
    from virgil.radio import CalibrationNuisance

    opts = dict(
        ref_obs=0, gains=False, offsets=False, alpha=False, sigmas=sigmas, antenna_gains=True, solint="scan",
        gain_amp_width=0.5, gain_phase_width=3.0,
    )
    opts.update(kw)
    return CalibrationNuisance(data, **opts)


def term_values(nuis, values, n_ddid):
    """Site dict for a radio term from flat named scalars: ``cal.flux`` and ``cal.sigma_d`` -> ``cal.sigma``."""
    import jax.numpy as jnp

    out = dict(nuis.defaults())
    out.update({k: v for k, v in values.items() if not k.startswith("cal.sigma_")})
    if nuis.sigmas is True:
        out["cal.sigma"] = jnp.stack([values[f"cal.sigma_{d}"] for d in range(n_ddid)])
    return out


def native_prior(model, n_nuis, data, flux0, sigmas=True):
    """``BoxPrior`` of a stage 4 model including ``cal.flux`` (log-uniform, a decade each side of ``flux0``) and,
    with ``sigmas``, ``cal.sigma_d`` (log-uniform 0.3-30: the error scale of ddid d)."""
    spec = native_spec(model, n_nuis) + [("cal.flux", flux0 / 10.0, flux0 * 10.0, "log")]
    if sigmas:
        spec += [(f"cal.sigma_{d}", 0.3, 30.0, "log") for d in range(len(data.ddid_ids))]
    return BoxPrior(spec)


def native_loglike(model, n_nuis, data, nuis):
    """``values -> log L`` of a stage 4 model: -(chi^2/2 + log_norm/2 + 1/2 sum log 2 pi / w) on complex
    visibilities, the gains Laplace-marginalised (``virgil`` radio term)."""
    scene_fn = native_scene(model, n_nuis)
    n_ddid = len(data.ddid_ids)
    term = data.term(lambda values: scene_fn(values), nuis, chi2="direct")

    def loglike(values):
        chi2, log_norm = term.full_chi2_and_log_norm(term_values(nuis, values, n_ddid))
        return -0.5 * (chi2 + log_norm + term._log_2pi_w)

    return loglike, term


def raw_chi2(term, nuis, values, n_ddid):
    """(chi^2, N) of the data on the quoted sigma at ``values`` (``cal.sigma`` set to 1): the model's raw
    misfit, antenna gains at their solved value. N = number of real data (2 per visibility)."""
    import jax.numpy as jnp

    vals = dict(term_values(nuis, values, n_ddid))
    if nuis.sigmas is True:
        vals["cal.sigma"] = jnp.ones(n_ddid)
    chi2, _ = term.full_chi2_and_log_norm(vals)
    return float(chi2), 2 * int(term.n)


def flux_guess(data):
    """Total-flux starting guess: the weighted mean |V| of the shortest tenth of the baselines."""
    q = onp.hypot(onp.asarray(data.u), onp.asarray(data.v))
    short = q <= onp.quantile(q, 0.1)
    return float(onp.average(onp.abs(onp.asarray(data.vis))[short], weights=onp.asarray(data.weight)[short]))


# --------------------------------------------------------------------------------------------------------------
# Stage 5: nonparametric imaging (CLEAN, RML)
# --------------------------------------------------------------------------------------------------------------

FOVS_UAS = (128.0, 225.0, 450.0)
STARTS = ("flat", "gauss40", "gauss100")
NPIX = 65  # odd: CLEAN needs it


def start_image(start, npix, fov_uas):
    """Unit-sum starting image: flat or a circular Gaussian of FWHM 40 or 100 microarcseconds at the centre."""
    if start == "flat":
        return onp.full((npix, npix), 1.0 / npix**2)
    fwhm = {"gauss40": 40.0, "gauss100": 100.0}[start]
    x = (onp.arange(npix) - (npix - 1) / 2) * fov_uas / npix
    g = onp.exp(-0.5 * (x[:, None] ** 2 + x[None, :] ** 2) / (fwhm / FWHM_PER_SIGMA) ** 2)
    return g / g.sum()


def resolution_mas(data):
    """Nominal resolution lambda / 2 B_max in mas from the longest baseline (u, v in wavelengths)."""
    q = float(onp.hypot(onp.asarray(data.u), onp.asarray(data.v)).max())
    return 1.0 / (2.0 * q) * 206264806.247


def clean_run(data, start, fov_uas, npix=NPIX, deconvolver="hogbom", niter=2000):
    """CLEAN of ``data`` on ``npix`` pixels across ``fov_uas``. Returns a dict with ``image`` (Jy per pixel,
    row 0 North), ``beam`` and the stop reason. CLEAN has no starting image: the start sets the CLEAN window,
    all of the field for ``flat`` and a disc of radius equal to the Gaussian's FWHM for the others."""
    from virgil.radio_clean import clean

    scale = fov_uas * UAS / npix
    mask = None
    if start != "flat":
        x = (onp.arange(npix) - (npix - 1) / 2) * fov_uas / npix
        radius = {"gauss40": 40.0, "gauss100": 100.0}[start]
        mask = (x[:, None] ** 2 + x[None, :] ** 2) <= radius**2
    sigma_img = 1.0 / math.sqrt(float(onp.sum(onp.asarray(data.weight))))
    kw = dict(deconvolver=deconvolver, niter=niter, threshold=3 * sigma_img)
    if mask is None:
        kw["usemask"] = "auto-multithresh"
    else:
        kw.update(mask=mask, usemask="user")
    res = clean(data, npix, scale, **kw)
    return {"image": onp.asarray(res.model), "beam": [float(b) for b in res.beam], "stop": str(res.stop),
            "iterations": int(res.iterations), "pixel_scale_uas": scale / UAS}


def rml_run(data, regularizer, start, fov_uas, npix=NPIX, weight=None, max_steps=5000, seed=0):
    """Regularised maximum likelihood of ``data`` with a pixel ``Image`` ("mem" or "tv"), antenna gains
    marginalised, quoted sigma. Returns ``image`` (Jy per pixel, row 0 North), the raw chi^2/N and fit flags.

    ``weight`` is the regulariser's strength (default ``0.1 N`` for MEM and ``0.01 N`` for TV, N real data:
    untuned, so the raw chi^2/N must be read before trusting the image)."""
    import numpyro.distributions as dist
    from virgil import fit
    from virgil.imaging import TV, MaxEntropy
    from virgil.models import Image

    scale = fov_uas * UAS / npix
    nuis = gain_nuisance(data, sigmas=False)
    n_real = 2 * len(data)
    scene = lambda values: Image(values["log_brightness"], pixel_scale_mas=scale)  # noqa: E731
    priors = nuis.priors()
    priors["log_brightness"] = dist.Normal(0.0, 3.0)
    init = nuis.defaults()
    base = start_image(start, npix, fov_uas)
    init["log_brightness"] = onp.log(base + 1e-4 * base.max())
    flux0 = flux_guess(data)
    init["cal.flux"] = flux0
    term = data.term(scene, nuis, chi2="direct")
    if regularizer == "mem":
        reg = MaxEntropy((0.1 if weight is None else weight) * n_real)
    elif regularizer == "tv":
        reg = TV((0.01 if weight is None else weight) * n_real, beam_mas=resolution_mas(data))
    else:
        raise ValueError(f"regularizer must be 'mem' or 'tv', not {regularizer!r}")
    result = fit(lambda **values: scene(values), priors, (), likelihoods=[term], regularizers=[reg], init=init,
                 method="lbfgs", max_steps=max_steps)
    vals = result.values
    image = onp.asarray(scene(vals).brightness) * float(vals["cal.flux"])
    chi2, _ = term.full_chi2_and_log_norm(vals)
    return {"image": image, "chi2": float(chi2), "n_real": n_real, "chi2_per_n": float(chi2) / n_real,
            "converged": bool(result.info.get("converged", False)), "steps": int(result.info.get("steps", -1)),
            "flux": float(vals["cal.flux"]), "pixel_scale_uas": scale / UAS}


def image_ring(image, pixel_scale_uas):
    """REX features of one image (row 0 North): ``ring_features`` on the Jy image, microarcsecond pixels."""
    from crosscheck.ring_extraction import ring_features

    return ring_features(onp.asarray(image, float), pixel_scale_uas)
