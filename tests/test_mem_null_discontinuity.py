"""The image-fit objective is singular where a model visibility is zero (finding F20).

The imaging-contest MEM fits stall (design/contest_convergence_diagnosis.md):
at the stall JAX gradients disagree with finite differences, the finite
difference scales as 1/eps, and the loss rises along -grad. Those are the
signs of a jump in the loss. Here the jump is reproduced on the smallest
data, and localised.

Where it comes from. Every closure phase is the argument of a product of
three model visibilities. A complex visibility that passes through zero
changes its phase by pi in an arbitrarily short step, so the closure
phase of every triangle that holds that baseline jumps by pi, and so does
its residual and the chi-squared. Near a null the derivative of the phase
is ~ 1/|V|. A closure-phase chi-squared therefore has no smooth, bounded
gradient at the nulls of the model. The data bring nothing to prevent a fit
from reaching them: where the true source is nearly resolved on a baseline
its V² is ~ 0 too, a V² term does not object to a null, and at a null the
closure phase of any triangle through it is free. The fit slides to model
visibilities of 1e-9 and less (below, 1e-18 on the tiny problem), where one
step of any size changes the closure phases completely and L-BFGS reports
"line search ran out of float64 precision".

It is not the correlated closure-phase whitening (F11, fixed): it happens
for three telescopes (one triangle, nothing correlated), and for six. It is
not the entropy, the softmax or the pixel parameterisation: those terms are
smooth at the same point. V² data alone (no closure phases) never go there.

The construction. An image that is centrosymmetric about its centre has a
real visibility on every baseline, so along a straight line in
log-brightness a baseline's visibility changes sign, passing through
exactly zero at a point we bisect for (|V| ~ 1e-18). At that point every
closure phase through the baseline is 0 or pi, and flips. The
passing tests pin what is established (which terms are smooth, that the
closure phase flips by pi, that the problem is confined to the nulls); the
strict xfail tests state what is wrong, the loss's AD and finite-difference
gradients, and flip when virgil's closure-phase residual is made smooth
(see trust/ledger.yml F20 for the fix).
"""

import warnings

import numpy as np
import pytest

from crosscheck import simulate, sky
from evidence.plugin import record

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

vm = pytest.importorskip("virgil.models")
from virgil.fitting import fit  # noqa: E402
from virgil.imaging import MaxEntropy, image_priors  # noqa: E402
from virgil.likelihood import whitened_residuals  # noqa: E402
from virgil.oidata import OIData  # noqa: E402

pytestmark = pytest.mark.x64

STATIONS = np.array(
    [[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999], [-30.0, 80.0], [60.0, -40.0]]
)
WAVEL = 1.65e-6
N, PIXEL, WEIGHT = 16, 1.0, 10.0
SIGMA_V2, SIGMA_CP_DEG = 0.005, 0.25
FD_EPS = (1e-5, 1e-7)


def source(u, v, w):
    """An asymmetric source, so that its closure phases are not 0 or pi."""
    return (
        sky.vis_gaussian(u, v, w, 3.0)
        + 0.5 * sky.vis_gaussian(u, v, w, 1.0, dra=6.0, ddec=-3.0)
        + 0.3 * sky.vis_gaussian(u, v, w, 1.0, dra=-5.0, ddec=4.0)
    ) / 1.8


def make_data(path, n_tel, *, closure_phases=True, rng=None, n_ha=1, n_wl=1):
    simulate.observe(
        path, source, STATIONS[:n_tel], hour_angles_h=np.linspace(-3, 3, n_ha) if n_ha > 1 else [0.0],
        wavelengths=np.linspace(1.5e-6, 2.4e-6, n_wl) if n_wl > 1 else [WAVEL], dec_deg=-50.0,
        sigma_v2=SIGMA_V2, sigma_cp_deg=SIGMA_CP_DEG, rng=rng, closure_phases=closure_phases,
    )
    return OIData(str(path))


def centrosymmetric(rng):
    eta = rng.normal(size=(N, N))
    return eta + eta[::-1, ::-1]


class NullLine:
    """A line eta(t) = a + t (b - a) of centrosymmetric log-brightness
    images, and the point t* on it where one baseline's visibility is zero.

    Visibilities come from our own direct transform of the pixel fluxes
    (crosscheck.sky), whose V² is checked against virgil's below."""

    def __init__(self, data, seed=5):
        rng = np.random.default_rng(seed)
        self.a, self.b = centrosymmetric(rng), centrosymmetric(rng)
        self.data = data
        self.n_v2 = int(np.asarray(data.vis).size)
        self.u = np.asarray(data.u).ravel()[: self.n_v2]
        self.v = np.asarray(data.v).ravel()[: self.n_v2]
        self.wavel = float(np.asarray(data.wavel).ravel()[0])
        ts = np.linspace(0.0, 1.0, 1001)
        vis = np.array([self.vis(t) for t in ts])
        assert np.abs(vis.imag).max() < 1e-12  # centrosymmetric: V is real
        for k in range(self.n_v2):
            s = np.sign(vis[:, k].real)
            hit = np.flatnonzero(s[:-1] * s[1:] < 0)
            if hit.size:
                self.baseline, lo, hi = k, ts[hit[0]], ts[hit[0] + 1]
                break
        else:
            raise AssertionError("no baseline's visibility changes sign on the line")
        sign = np.sign(self.vis(lo)[self.baseline].real)
        for _ in range(200):  # bisection to adjacent floats
            mid = 0.5 * (lo + hi)
            if np.sign(self.vis(mid)[self.baseline].real) == sign:
                lo = mid
            else:
                hi = mid
        self.t_null = 0.5 * (lo + hi)

    def log_brightness(self, t):
        return self.a + t * (self.b - self.a)

    def flux(self, t):
        e = self.log_brightness(t)
        p = np.exp(e - e.max())
        return p / p.sum()

    def vis(self, t):
        cloud = sky.pixel_image(self.flux(t), PIXEL)
        return sky.visibility(cloud, self.u, self.v, self.wavel)

    def image(self, t):
        return vm.Image(jnp.asarray(self.log_brightness(t)), PIXEL)


def terms(line, regulariser=None):
    """The objective's terms as functions of t: V², closure phases (the
    whitened ones and the periodic penalty, if the triangles are correlated),
    the entropy, and their sum (the loss the fitter minimises)."""
    data, n_v2 = line.data, line.n_v2
    n_phi = int(np.asarray(data.phi).size)
    n_cp = int(data.cp_noise.size) if data.cp_noise is not None else n_phi
    regulariser = regulariser or MaxEntropy(WEIGHT)

    def residuals(t):
        return whitened_residuals(line.image(t), data)

    out = {
        "v2": lambda t: 0.5 * jnp.sum(residuals(t)[:n_v2] ** 2),
        "cp": lambda t: 0.5 * jnp.sum(residuals(t)[n_v2 : n_v2 + n_cp] ** 2),
        "cp_penalty": lambda t: 0.5 * jnp.sum(residuals(t)[n_v2 + n_cp :] ** 2),
        "entropy": lambda t: regulariser.value(line.image(t)),
        "loss": lambda t: 0.5 * jnp.sum(residuals(t) ** 2) + regulariser.value(line.image(t)),
    }
    return out


def ad_and_fd(f, t, eps=FD_EPS):
    return float(jax.grad(f)(t)), [float((f(t + e) - f(t - e)) / (2 * e)) for e in eps]


@pytest.fixture(scope="module", params=[3, 6], ids=["3T", "6T"])
def line(request, tmp_path_factory):
    path = tmp_path_factory.mktemp("mem") / f"{request.param}t.fits"
    return NullLine(make_data(path, request.param))


# ----------------------------------------------------------------- passing


@pytest.mark.validates("virgil.models.Image", "virgil.likelihood.whitened_residuals", roots=["mathematics"])
def test_constructed_null_is_a_null_of_virgils_model(line):
    """Our own transform and virgil's model V² agree on the line, and the
    bisected point is a visibility null of 1e-15 or less."""
    for t in (0.0, 0.3, line.t_null, 0.9):
        mine = np.abs(line.vis(t)) ** 2
        theirs = np.asarray(line.data.model(line.image(t)))[: line.n_v2]
        assert np.max(np.abs(mine - theirs)) < 1e-12
    null = abs(line.vis(line.t_null)[line.baseline])
    record("null_abs_v", null)
    assert null < 1e-15


@pytest.mark.validates("virgil.likelihood.whitened_residuals", "virgil.imaging.MaxEntropy", roots=["mathematics"])
def test_v2_and_entropy_terms_are_smooth_at_the_null(line):
    """The localisation, first half: V² (|V|² is smooth through V = 0) and the
    entropy (softmax over the pixels) have AD = FD at the null."""
    f = terms(line)
    for name in ("v2", "entropy"):
        ad, fds = ad_and_fd(f[name], line.t_null)
        record(f"{name}_ad", ad)
        for fd in fds:
            assert abs(ad - fd) <= 1e-6 * max(abs(ad), 1.0), (name, ad, fd)


@pytest.mark.validates("virgil.likelihood.whitened_residuals", roots=["mathematics"])
def test_closure_phase_flips_by_pi_across_the_null(line):
    """The localisation, second half: the model closure phases of the
    triangles through the null baseline change by pi across it (a
    centrosymmetric image has closure phases 0 or pi; V → -V flips them),
    while the others and every V² are continuous. This is the observable's
    behaviour, not virgil's choice."""
    delta = 1e-9
    below = np.asarray(line.data.model(line.image(line.t_null - delta)))
    above = np.asarray(line.data.model(line.image(line.t_null + delta)))
    n = line.n_v2
    assert np.max(np.abs(below[:n] - above[:n])) < 1e-6  # V² continuous
    step = np.abs(np.angle(np.exp(1j * (above[n:] - below[n:]))))  # wrapped
    jumped = step > 1.0
    record("n_closure_phases_flipped", int(jumped.sum()))
    record("n_closure_phases", step.size)
    assert jumped.any()
    assert np.allclose(step[jumped], np.pi, atol=1e-3)
    assert np.all(step[~jumped] < 1e-6)


@pytest.mark.validates("virgil.likelihood.whitened_residuals", "virgil.imaging.MaxEntropy", roots=["mathematics"])
def test_loss_gradient_is_right_away_from_nulls(line):
    """The problem is confined to the nulls: a tenth of the way along, and
    0.02 either side of the null (where V ≫ eps), AD = FD."""
    f = terms(line)["loss"]
    for t in (0.1, line.t_null - 0.02, line.t_null + 0.02):
        ad, fds = ad_and_fd(f, t, eps=(1e-7,))
        record("ad_at_%.4f" % t, ad)
        assert abs(ad - fds[0]) <= 1e-4 * max(abs(ad), 1.0)


@pytest.mark.validates("virgil.likelihood.whitened_residuals", "virgil.imaging.MaxEntropy", roots=["mathematics"])
def test_loss_changes_by_a_fixed_amount_across_the_null(line):
    """The signature the contest stalls show, here on the model line: the
    loss changes by an amount that does not shrink with the distance to the
    null, so its finite difference scales as 1/eps. Pinned as measured
    numbers (not the finding: the finding is that AD ≠ FD, below); the
    jump lies entirely in the closure-phase terms."""
    f = terms(line)
    jumps = {}
    for name in ("v2", "entropy", "cp", "cp_penalty", "loss"):
        g = f[name]
        jumps[name] = [abs(float(g(line.t_null + d) - g(line.t_null - d))) for d in (1e-4, 1e-6, 1e-8)]
        record(f"jump_{name}_1e-8", jumps[name][2])
    for name in ("v2", "entropy"):
        assert jumps[name][2] < 1e-5  # continuous: the change is ~ 2 d dV
    assert jumps["loss"][0] > 100
    assert jumps["loss"][2] == pytest.approx(jumps["loss"][0], rel=1e-2)  # does not shrink
    fd_small, fd_large = (jumps["loss"][i] / (2 * d) for i, d in ((2, 1e-8), (0, 1e-4)))
    assert fd_small / fd_large == pytest.approx(1e4, rel=1e-2)  # FD ∝ 1/eps


@pytest.mark.validates("virgil.imaging.MaxEntropy", "virgil.fitting.fit", roots=["mathematics"])
def test_v2_only_mem_fit_has_no_null_stall(tmp_path):
    """Control: the same kind of MEM fit on V² alone (six telescopes, no
    closure phases) converges to a point where AD = FD."""
    data = make_data(tmp_path / "v2.fits", 6, closure_phases=False, rng=np.random.default_rng(1), n_ha=5, n_wl=4)
    start = vm.Image(jnp.zeros((32, 32)), PIXEL)
    reg = MaxEntropy(1e3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit(start, image_priors(start), data, [reg], max_steps=3000)
    assert result.info["converged"]
    x = result.model.log_brightness

    def loss(x):
        m = start.__class__(x, PIXEL)
        return 0.5 * jnp.sum(whitened_residuals(m, data) ** 2) + reg.value(m)

    d = np.random.default_rng(1).normal(size=x.shape)
    d /= np.linalg.norm(d)
    ad = float(jnp.vdot(jax.grad(loss)(x), d))
    fd = float((loss(x + 1e-5 * d) - loss(x - 1e-5 * d)) / 2e-5)
    record("ad", ad)
    assert abs(ad - fd) <= 1e-4 * max(abs(fd), 1e-3)


# ------------------------------------------------------------- reference


@pytest.mark.validates("virgil.likelihood.whitened_residuals", roots=["mathematics"], kind="reference")
def test_bispectrum_residual_is_continuous_across_the_null(line):
    """What a smooth closure-phase objective looks like, written with our own
    transform: the complex bispectrum against the data's, weighted by the
    data's amplitude and phase error (as BSMEM and MiRA do),

        r = (B_model - B_data) / (|B_data| σ_φ),   B = V_ab V_bc conj(V_ac),

    which is a polynomial in the model visibilities, so continuous and
    smooth through V = 0, and equal to the phase residual to first order
    where the model amplitude matches the data. Its change across the null
    shrinks linearly with the distance, where the phase residual's does not."""
    from crosscheck import array

    n_tel = 3 if line.n_v2 == 3 else 6
    tris = array.triangles(n_tel)
    pairs = array.baselines(n_tel)
    index = {p: i for i, p in enumerate(pairs)}
    sigma = np.deg2rad(SIGMA_CP_DEG)
    v_data = source(line.u, line.v, line.wavel)  # noise-free truth, per baseline

    def bispectra(v):
        return np.array([v[index[(a, b)]] * v[index[(b, c)]] * np.conj(v[index[(a, c)]]) for a, b, c in tris])

    b_data = bispectra(v_data)

    def loss(t):
        r = (bispectra(line.vis(t)) - b_data) / (np.abs(b_data) * sigma)
        return 0.5 * float(np.sum(np.abs(r) ** 2))

    changes = [abs(loss(line.t_null + d) - loss(line.t_null - d)) for d in (1e-4, 1e-6, 1e-8)]
    record("bispectrum_change_1e-8", changes[2])
    assert changes[2] < 1e-3 * changes[0]
    assert changes[2] < 1e-2 * max(loss(line.t_null), 1.0)


# ------------------------------------------------------------------ xfail


@pytest.mark.xfail(
    strict=True,
    reason="F20: the closure-phase residual is singular where a model visibility is zero, so AD ≠ FD at such a point",
)
@pytest.mark.validates("virgil.likelihood.whitened_residuals", roots=["mathematics"], kind="finding")
def test_f20_loss_gradient_is_right_at_a_visibility_null(line):
    """At a model-visibility null, AD and a central finite difference of the
    loss must agree at eps = 1e-5 and 1e-7. Today the finite difference
    scales as 1/eps and the AD gradient is 1e15 or more."""
    ad, fds = ad_and_fd(terms(line)["loss"], line.t_null)
    record("ad", ad)
    record("fd_1e-5", fds[0])
    record("fd_1e-7", fds[1])
    for fd in fds:
        assert abs(ad - fd) <= 1e-3 * max(abs(fd), 1.0)


@pytest.mark.xfail(
    strict=True,
    reason="F20: a MEM fit of closure-phase data runs into a visibility null (|V| ~ 1e-9 and below), where AD ≠ FD",
)
@pytest.mark.validates("virgil.imaging.MaxEntropy", "virgil.fitting.fit", roots=["mathematics"], kind="finding")
def test_f20_mem_fit_stalls_at_a_point_where_ad_equals_fd(tmp_path):
    """The contest-stall reproducer: a MEM fit (flat start, three telescopes,
    one snapshot and channel, 16 × 16 pixels, noisy data) for 3000 L-BFGS
    steps. The fit ends at a model with a visibility of ~1e-9 or less and
    does not converge; AD and FD of the loss disagree by orders of magnitude
    along a random direction. Once the objective is smooth, the fit goes
    elsewhere (or converges) and AD = FD, so this flips."""
    data = make_data(tmp_path / "stall.fits", 3, rng=np.random.default_rng(1))
    start = vm.Image(jnp.zeros((N, N)), PIXEL)
    reg = MaxEntropy(1e3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit(start, image_priors(start), data, [reg], max_steps=3000)
    x = result.model.log_brightness

    def loss(x):
        m = vm.Image(x, PIXEL)
        return 0.5 * jnp.sum(whitened_residuals(m, data) ** 2) + reg.value(m)

    n_v2 = int(np.asarray(data.vis).size)
    record("min_model_v2", float(np.min(np.asarray(data.model(result.model))[:n_v2])))
    d = np.random.default_rng(1).normal(size=x.shape)
    d /= np.linalg.norm(d)
    ad = float(jnp.vdot(jax.grad(loss)(x), d))
    fd = float((loss(x + 1e-5 * d) - loss(x - 1e-5 * d)) / 2e-5)
    record("ad", ad)
    record("fd", fd)
    assert abs(ad - fd) <= 1e-3 * max(abs(fd), 1.0)
