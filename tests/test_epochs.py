"""virgil.epochs (multi-epoch orbits from interferometric data), against our
own chi-squared (crosscheck.chi2, three-telescope OIFITS files from
crosscheck.simulate read with astropy), our own Kepler positions
(crosscheck.orbits) and SciPy quadrature.

* marginal_loglike: each block's likelihood (Gaussian in V², and in the
  chords 2 sin(Δ/2)/σ of the closure phases), with its error scale s
  integrated out under the Jeffreys prior ds/s by scipy.integrate.quad in
  ln s. Differences between models are compared, so the constants (the
  quoted errors, 2π, the prior's normalisation) cancel. ``dof`` tempers
  the likelihood (L^dof) before the integral; ``s_max`` bounds the
  integral to [1/s_max, s_max] and uses the exact von Mises density
  exp(κ cos Δ)/(2π I₀(κ)), κ = 1/(sσ)², for the closure phases. The
  control integrates under a uniform prior in s instead, which must not
  agree.
* epoch_positions: our own grid search of the scale-marginalised surface
  m = -Σ_b (ν_b/2) ln χ²_b on the same grid: the best grid point, its
  refinement by SciPy, the curvature, gap and gap_marginal; and the
  invariance of the positions and gap_marginal when every quoted error is
  three times too small (virgil#281), while gap grows ninefold.
* Epochs.loglike: the full normalised log likelihood of all epochs, the
  companion at our own orbit positions.
* rank_orbits ("quoted" and "marginal"): the order and the differences of
  the scores of trial orbits, against our own orbit χ² and m.
* chain_starts: our own greedy choice of distinct orbits (best first; one
  mode when the positions at every snapshot time agree within half the
  finest λ/B_max) on two clusters of trial orbits, the mirror orbit
  (Ω + 180°, ω + 180°) belonging to the true one.
* start_from_positions (slow): the start recovers a synthetic orbit, by
  our own positions and chi-squared.
"""

import shutil
import warnings

import numpy as np
import pytest
from astropy.io import fits
from scipy import integrate, optimize, special

from crosscheck import chi2 as ours, orbits as co, simulate, sky
from evidence.plugin import record

ve = pytest.importorskip("virgil.epochs")
pytest.importorskip("jaxoplanet")
from virgil.models import BinaryModelCartesian, OrbitalBinary  # noqa: E402
from virgil.oidata import OIData  # noqa: E402
from virgil.orbits import KeplerOrbit  # noqa: E402

pytestmark = pytest.mark.x64

UTS3 = np.array([[-9.925, -20.335], [14.887, 30.502], [103.306, 43.999]])
WAVELENGTHS = np.linspace(2.0e-6, 2.4e-6, 4)
T0 = 60000.0
ELEMENTS = (900.0, 100.0, 0.3, 50.0, 60.0, 120.0, 8.0)  # P, dt_peri, e, i, omega, Omega, a
FLUX = 0.1


def binary_vis(f, x, y):
    def vis(u, v, w):
        return (sky.vis_point(u, v, w) + f * sky.vis_point(u, v, w, x, y)) / (1 + f)

    return vis


def observe(path, f, x, y, seed, sigma_v2=0.01, sigma_cp_deg=1.0, hours=7):
    """A three-telescope night (independent closure phases), noise drawn
    per closure phase so that the quoted errors are the true ones."""
    simulate.observe(path, binary_vis(f, x, y), UTS3, hour_angles_h=np.linspace(-3, 3, hours),
                     wavelengths=WAVELENGTHS, dec_deg=-50.0, sigma_v2=sigma_v2, sigma_cp_deg=sigma_cp_deg,
                     rng=np.random.default_rng(seed), phase_noise="triangle")
    return path


def shrink_errors(src, dst, factor):
    """The same data with every quoted error divided by ``factor``."""
    shutil.copy(src, dst)
    with fits.open(dst, mode="update") as h:
        h["OI_VIS2"].data["VIS2ERR"] /= factor
        h["OI_T3"].data["T3PHIERR"] /= factor
    return dst


def blocks(d, f, x, y):
    """Whitened residuals of our binary, (V² block, closure-phase chords)."""
    r = ours.residuals(d, binary_vis(f, x, y))
    n = d["v2"].size
    return r[:n], r[n:]


def m_ours(d, f, x, y):
    """-Σ_b (ν_b/2) ln χ²_b, ν_b the size of each block (all independent
    for three telescopes)."""
    return -sum(0.5 * r.size * np.log(np.sum(r**2)) for r in blocks(d, f, x, y))


def resolution_mas(ds):
    """The finest λ/B over the samples of the datasets (mas)."""
    return min(np.min(d["wl"] / np.hypot(d["u"], d["v"])) for d in ds) * 180 / np.pi * 3.6e6


def orbit_position(elements, t):
    """Our own (dRA, dDec) of the companion at MJD t."""
    period, dt_peri, *rest = elements
    x, y = co.position(np.atleast_1d(t) - T0, period, dt_peri, *rest)
    return np.stack([x, y], -1)


def elements_of(o):
    return tuple(float(getattr(o, k)) for k in ("period", "dt_peri", "ecc", "inc", "omega", "Omega", "a_mas"))


# ------------------------------------------------- marginal_loglike: quadrature


def log_integral(logl, lo=None, hi=None, uniform=False):
    """ln ∫ L(s) p(s) ds by quad in t = ln s, p = 1/s (Jeffreys) or 1
    (``uniform``), over [lo, hi] in ln s (unbounded: the peak ± 12)."""

    def g(t):
        return logl(np.exp(t)) + (t if uniform else 0.0)

    peak = optimize.minimize_scalar(lambda t: -g(t), bounds=(-12.0, 12.0) if lo is None else (lo, hi),
                                    method="bounded", options={"xatol": 1e-10}).x
    a, b = (peak - 12.0, peak + 12.0) if lo is None else (lo, hi)
    gmax = g(peak)
    val, err = integrate.quad(lambda t: np.exp(g(t) - gmax), a, b, points=[peak] if a < peak < b else None,
                              epsabs=0.0, epsrel=1e-12, limit=400)
    return gmax + np.log(val)


def gaussian_logl(r, sigma, dof=1.0):
    """ln of the Gaussian likelihood of whitened residuals r with errors
    s σ, tempered by dof."""
    return lambda s: dof * np.sum(-0.5 * (r / s) ** 2 - np.log(s * sigma) - 0.5 * np.log(2 * np.pi))


def von_mises_logl(chord, sigma):
    """ln of the von Mises likelihood exp(κ cos Δ)/(2π I₀(κ)), κ = 1/(sσ)²,
    from the whitened chords c = 2 sin(Δ/2)/σ: κ cos Δ = κ - (c/s)²/2."""

    def logl(s):
        kappa = 1.0 / (s * sigma) ** 2
        return np.sum(-0.5 * (chord / s) ** 2 - np.log(2 * np.pi) - np.log(special.i0e(kappa)))

    return logl


def marginal_by_quad(d, model, dof=1.0, s_max=None, uniform=False):
    rv, rp = blocks(d, *model)
    sv, sp = d["dv2"].ravel(), d["dcp"].ravel()
    if s_max is None:
        return (log_integral(gaussian_logl(rv, sv, dof), uniform=uniform)
                + log_integral(gaussian_logl(rp, sp, dof), uniform=uniform))
    lo, hi = -np.log(s_max), np.log(s_max)
    return (log_integral(gaussian_logl(rv, sv), lo, hi, uniform=uniform)
            + log_integral(von_mises_logl(rp, sp), lo, hi, uniform=uniform))


TRUTH1 = (0.1, 6.0, -4.0)
MODELS1 = [TRUTH1, (0.1, 6.5, -4.3), (0.1, -6.0, 4.0)]  # truth, offset, mirror image


@pytest.fixture(scope="module")
def single(tmp_path_factory):
    path = observe(tmp_path_factory.mktemp("marg") / "a.fits", *TRUTH1, seed=1, sigma_v2=0.02, sigma_cp_deg=2.0,
                   hours=5)
    return OIData(str(path)), ours.load(path)


@pytest.fixture(scope="module")
def weak(tmp_path_factory):
    """Weak closure phases (σ = 40°), where s σ is not small for s ≫ 1."""
    truth = (0.3, 6.0, -4.0)
    path = observe(tmp_path_factory.mktemp("weak") / "w.fits", *truth, seed=2, sigma_v2=0.02, sigma_cp_deg=40.0,
                   hours=5)
    return OIData(str(path)), ours.load(path), [truth, (0.3, 6.5, -4.3), (0.3, -6.0, 4.0)]


def virgil_m(data, model, **kw):
    f, x, y = model
    return float(ve.marginal_loglike(BinaryModelCartesian(x, y, f), data, **kw))


@pytest.mark.parametrize("dof", [1.0, 0.5])
@pytest.mark.validates("virgil.epochs.marginal_loglike", roots=["mathematics"])
def test_marginal_loglike_is_the_jeffreys_integral(single, dof):
    """Differences of m between models equal those of ln ∫ L^dof(s) ds/s,
    one scale per block, by quad; and m is the documented closed form
    -dof Σ_b (ν_b/2) ln χ²_b (up to a constant, which is reported)."""
    data, d = single
    got = [virgil_m(data, m, dof=dof) for m in MODELS1]
    want = [marginal_by_quad(d, m, dof=dof) for m in MODELS1]
    closed = [dof * m_ours(d, *m) for m in MODELS1]
    worst = max(abs((got[k] - got[0]) - (want[k] - want[0])) for k in (1, 2))
    record("max_abs_diff_vs_quad", worst)
    record("closed_form_offset", got[0] - closed[0])
    record("delta_m_mirror", got[2] - got[0])
    assert worst < 1e-9
    assert max(abs((got[k] - got[0]) - (closed[k] - closed[0])) for k in (1, 2)) < 1e-9


@pytest.mark.validates("virgil.epochs.marginal_loglike", roots=["mathematics"], kind="control")
def test_marginal_loglike_control_uniform_prior_in_s(single):
    """A uniform prior in s (∫ L ds) gives (ν_b - 1)/2 in place of ν_b/2:
    its differences must miss virgil's by far more than the tolerance."""
    data, d = single
    got = [virgil_m(data, m) for m in MODELS1]
    wrong = [marginal_by_quad(d, m, uniform=True) for m in MODELS1]
    miss = min(abs((got[k] - got[0]) - (wrong[k] - wrong[0])) for k in (1, 2))
    record("min_abs_miss", miss)
    assert miss > 1e-3


@pytest.fixture(scope="module")
def inflated(tmp_path_factory):
    """The weak-closure-phase data with every quoted error 2 times too large
    (s ≈ 0.5 in both blocks): the lower bound binds."""
    tmp = tmp_path_factory.mktemp("infl")
    src = observe(tmp / "a.fits", 0.3, 6.0, -4.0, seed=2, sigma_v2=0.02, sigma_cp_deg=40.0, hours=5)
    path = shrink_errors(src, tmp / "b.fits", 0.5)
    return OIData(str(path)), ours.load(path)


MODELS_WEAK = [(0.3, 6.0, -4.0), (0.3, 6.5, -4.3), (0.3, -6.0, 4.0)]  # truth, offset, mirror image
# (data, s_max, tolerance on the mirror's m difference): the limits in ln s are
# about one posterior width (1/sqrt(2 nu) ~ 0.16) from the peak, so the bound
# changes the answer (``upper``: s ≈ 1.1 against s_max 1.2; ``lower``: s ≈ 0.5
# against 1/1.2).
BOUND_CASES = {"upper": ("weak", 1.2), "lower": ("inflated", 1.2)}
BOUND_TOL = 5e-3  # virgil integrates on a grid in ln s; see the finding below


def bound_case(request_data, case):
    name, s_max = BOUND_CASES[case]
    data, d = request_data[name][:2]
    return data, d, s_max


def mirror_difference(fn, data_models):
    return fn(data_models[2]) - fn(data_models[0])


@pytest.mark.validates("virgil.epochs.marginal_loglike", roots=["mathematics"])
def test_marginal_loglike_bounded_scales(weak):
    """With s_max, each block's scale is bounded to [1/s_max, s_max] and
    integrated out with the exact von Mises density for the closure
    phases: differences against quad of that integral (loose bound, 10)."""
    data, d, models = weak
    s_max = 10.0
    got = [virgil_m(data, m, s_max=s_max) for m in models]
    want = [marginal_by_quad(d, m, s_max=s_max) for m in models]
    gauss = [virgil_m(data, m) for m in models]
    worst = max(abs((got[k] - got[0]) - (want[k] - want[0])) for k in (1, 2))
    record("max_abs_diff_vs_quad", worst)
    record("unbounded_minus_bounded_mirror_gap", (gauss[2] - gauss[0]) - (got[2] - got[0]))
    assert worst < 1e-6


@pytest.mark.parametrize("case", ["upper", "lower"])
@pytest.mark.validates("virgil.epochs.marginal_loglike", roots=["mathematics"])
def test_marginal_loglike_bound_binds(weak, inflated, case):
    """A tight s_max, where the limits of ln s are about one posterior width
    from the peak, so the bound changes the answer. virgil matches quad of
    the bounded integral for the mirror image's m, and the bound is shown to
    matter: the bounded difference differs from the unbounded one by far
    more than the tolerance (and the wrong-bound controls below miss)."""
    data, d, s_max = bound_case({"weak": weak, "inflated": inflated}, case)
    got = mirror_difference(lambda m: virgil_m(data, m, s_max=s_max), MODELS_WEAK)
    want = mirror_difference(lambda m: marginal_by_quad(d, m, s_max=s_max), MODELS_WEAK)
    free = mirror_difference(lambda m: marginal_by_quad(d, m), MODELS_WEAK)
    record(f"abs_diff_vs_quad_{case}", abs(got - want))
    record(f"bound_effect_{case}", abs(want - free))
    assert abs(want - free) > 0.1
    assert abs(got - want) < BOUND_TOL
    assert abs(want - free) > 10 * BOUND_TOL


@pytest.mark.parametrize("case,wrong", [("upper", "no_upper"), ("lower", "no_lower"), ("lower", "squared_lower")])
@pytest.mark.validates("virgil.epochs.marginal_loglike", roots=["mathematics"], kind="control")
def test_marginal_loglike_control_wrong_bound(weak, inflated, case, wrong):
    """The wrong bound (no upper limit; no lower limit; [1/s_max², s_max]) on
    the side where the data reach it must miss virgil's by far more than the
    tolerance."""
    data, d, s_max = bound_case({"weak": weak, "inflated": inflated}, case)
    got = mirror_difference(lambda m: virgil_m(data, m, s_max=s_max), MODELS_WEAK)
    lo, hi = -np.log(s_max), np.log(s_max)
    lo, hi = {"no_lower": (-12.0, hi), "no_upper": (lo, 12.0), "squared_lower": (2 * lo, hi)}[wrong]

    def wrong_m(m):
        rv, rp = blocks(d, *m)
        sv, sp = d["dv2"].ravel(), d["dcp"].ravel()
        return log_integral(gaussian_logl(rv, sv), lo, hi) + log_integral(von_mises_logl(rp, sp), lo, hi)

    miss = abs(got - mirror_difference(wrong_m, MODELS_WEAK))
    record(f"abs_miss_{case}_{wrong}", miss)
    assert miss > 10 * BOUND_TOL


@pytest.mark.parametrize("case", ["upper", "lower"])
@pytest.mark.validates("virgil.epochs.marginal_loglike", roots=["mathematics"], kind="finding")
@pytest.mark.xfail(strict=True, reason="the ln s grid of the bounded integral is too coarse where the likelihood "
                   "falls steeply from a bound (a model with s ≈ 5 against s_max = 1.2): errors of 0.1-0.7 in m")
def test_marginal_loglike_bounded_steep_edge(weak, inflated, case):
    """The offset model (χ²_V²/ν ≈ 30 or more, so its likelihood is cut off
    steeply at s_max): virgil's m differences against the truth should match
    quad of the bounded integral to 1e-3, as they do for the mirror. They
    miss by 0.1-0.7 (our dense trapezoid sums converge on quad)."""
    data, d, s_max = bound_case({"weak": weak, "inflated": inflated}, case)
    s_max = 1.2
    got = [virgil_m(data, m, s_max=s_max) for m in MODELS_WEAK[:2]]
    want = [marginal_by_quad(d, m, s_max=s_max) for m in MODELS_WEAK[:2]]
    miss = abs((got[1] - got[0]) - (want[1] - want[0]))
    record(f"abs_miss_offset_{case}", miss)
    assert miss < 1e-3


# -------------------------------------------------------- epoch_positions


NIGHTS = {"n1": (0.1, 6.0, -4.0), "n2": (0.1, -3.0, 7.0)}
GRID = {"dra": np.arange(-12.0, 12.01, 1.0), "ddec": np.arange(-12.0, 12.01, 1.0), "flux": np.array([0.05, 0.1, 0.2])}
GAP_MAS = 3.0


@pytest.fixture(scope="module")
def nights(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("nights")
    paths = {n: observe(tmp / f"{n}.fits", *t, seed=10 + k) for k, (n, t) in enumerate(NIGHTS.items())}
    small = {n: shrink_errors(p, tmp / f"{n}_small.fits", 3.0) for n, p in paths.items()}
    times = {"n1": T0, "n2": T0 + 100.0}
    quoted = ve.Epochs({n: OIData(str(p)) for n, p in paths.items()}, times=times)
    under = ve.Epochs({n: OIData(str(p)) for n, p in small.items()}, times=times)
    return quoted, under, {n: ours.load(p) for n, p in paths.items()}


def our_grid(d):
    """m and the quoted log likelihood (-χ²/2) on GRID: arrays (dra, ddec, flux)."""
    shape = (GRID["dra"].size, GRID["ddec"].size, GRID["flux"].size)
    m, q = np.empty(shape), np.empty(shape)
    for i, x in enumerate(GRID["dra"]):
        for j, y in enumerate(GRID["ddec"]):
            for k, f in enumerate(GRID["flux"]):
                rv, rp = blocks(d, f, x, y)
                cv, cp = np.sum(rv**2), np.sum(rp**2)
                m[i, j, k] = -0.5 * (rv.size * np.log(cv) + rp.size * np.log(cp))
                q[i, j, k] = -0.5 * (cv + cp)
    return m, q


def gap_on(surface):
    """Best grid value minus the best more than GAP_MAS from it (flux
    maximised out), and the best grid point."""
    s = surface.max(axis=2)
    i, j = np.unravel_index(np.argmax(s), s.shape)
    X, Y = np.meshgrid(GRID["dra"], GRID["ddec"], indexing="ij")
    far = np.hypot(X - X[i, j], Y - Y[i, j]) > GAP_MAS
    return s[i, j] - s[far].max(), (X[i, j], Y[i, j], GRID["flux"][np.argmax(surface[i, j])])


@pytest.fixture(scope="module")
def positions(nights):
    quoted, under, _ = nights
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pq = ve.epoch_positions(quoted, GRID, gap_mas=GAP_MAS)
    with pytest.warns(UserWarning, match="n1.*n2"):
        pu = ve.epoch_positions(under, GRID, gap_mas=GAP_MAS)
    grid_only = ve.epoch_positions(quoted, GRID, gap_mas=GAP_MAS, refine=False)
    return pq, pu, grid_only


@pytest.mark.validates("virgil.epochs.epoch_positions", "virgil.epochs.EpochPositions", "virgil.epochs.Epochs",
                       roots=["mathematics"])
def test_epoch_positions_grid_and_gaps(nights, positions):
    """Our own grid search of m and of -χ²/2: the same best grid point
    and flux, the same gap_marginal (on m) and gap (on the quoted errors)."""
    _, _, ds = nights
    pq, _, grid_only = positions
    worst_gap = 0.0
    for k, name in enumerate(pq.names):
        m, q = our_grid(ds[name])
        gm, best = gap_on(m)
        gq, _ = gap_on(q)
        assert (grid_only.dra[k], grid_only.ddec[k], grid_only.flux[k]) == pytest.approx(best, abs=1e-12)
        worst_gap = max(worst_gap, abs(pq.gap_marginal[k] / gm - 1), abs(pq.gap[k] / gq - 1))
        assert grid_only.gap_marginal[k] == pytest.approx(pq.gap_marginal[k], rel=1e-12)
    record("max_rel_gap", worst_gap)
    record("min_gap_marginal", float(np.min(pq.gap_marginal)))
    assert worst_gap < 1e-8


def curvature(d, x, h):
    """Second derivatives of m in (dra, ddec, flux) at x, central differences with steps h."""
    H = np.empty((3, 3))
    for a in range(3):
        for b in range(3):
            ea, eb = np.eye(3)[a] * h[a], np.eye(3)[b] * h[b]
            f = [m_ours(d, p[2], p[0], p[1]) for p in (x + ea + eb, x + ea - eb, x - ea + eb, x - ea - eb)]
            H[a, b] = (f[0] - f[1] - f[2] + f[3]) / (4 * h[a] * h[b])
    return H


@pytest.mark.validates("virgil.epochs.epoch_positions", "virgil.epochs.EpochPositions", roots=["mathematics"])
def test_epoch_positions_refined_maximum_of_m(nights, positions):
    """The refined position and flux maximise our m (SciPy, from the best
    grid point); the covariance is the position block of the inverse of the full
    (position and flux) curvature of m there (our Richardson-extrapolated
    finite differences), and differs from the flux-fixed inverse of the
    position block by far more than the tolerance; chi2_raw and scale are χ²_b/ν_b and its root at
    the fitted point; and the truth lies within a few σ."""
    _, _, ds = nights
    pq, _, _ = positions
    worst_pos, worst_flux, worst_cov, worst_cond, worst_chi2 = 0.0, 0.0, 0.0, np.inf, 0.0
    for k, name in enumerate(pq.names):
        d = ds[name]
        x0 = [pq.dra[k], pq.ddec[k], pq.flux[k]]
        res = optimize.minimize(lambda p: -m_ours(d, p[2], p[0], p[1]), x0, method="Nelder-Mead",
                                options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 4000})
        worst_pos = max(worst_pos, np.max(np.abs(res.x[:2] - x0[:2])))
        worst_flux = max(worst_flux, abs(res.x[2] - x0[2]))
        # curvature in (dra, ddec, flux) by central differences, Richardson-extrapolated over h and h/2
        H1, H2 = (curvature(d, res.x, np.array([2e-3, 2e-3, 2e-4]) / n) for n in (1, 2))
        H = (4 * H2 - H1) / 3
        cov = np.linalg.inv(-H)[:2, :2]
        worst_cov = max(worst_cov, np.max(np.abs(pq.cov[k] - cov)) / np.max(np.abs(cov)))
        # the flux-conditioned covariance, inv(-H[:2,:2]), is what a bug that held the flux fixed would return
        worst_cond = min(worst_cond, np.max(np.abs(np.linalg.inv(-H[:2, :2]) / cov - 1)))
        rv, rp = blocks(d, pq.flux[k], pq.dra[k], pq.ddec[k])
        for key, r in (("vis", rv), ("phi", rp)):
            worst_chi2 = max(worst_chi2, abs(pq.chi2_raw[k][key] / (np.sum(r**2) / r.size) - 1),
                             abs(pq.scale[k][f"{key}_scale"] / np.sqrt(np.sum(r**2) / r.size) - 1))
        truth = np.array(NIGHTS[name][1:])
        delta = np.array([pq.dra[k], pq.ddec[k]]) - truth
        assert delta @ np.linalg.solve(pq.cov[k], delta) < 25.0
    record("max_abs_position_mas", worst_pos)
    record("max_abs_flux", worst_flux)
    record("max_rel_cov", worst_cov)
    record("min_rel_flux_conditioned_minus_marginalised_cov", worst_cond)
    record("max_rel_chi2_raw_scale", worst_chi2)
    assert worst_pos < 1e-6
    assert worst_flux < 1e-6
    assert worst_cov < 1e-4
    assert worst_cond > 100 * worst_cov  # the tolerance separates flux marginalised from flux fixed
    assert worst_chi2 < 1e-6


@pytest.mark.validates("virgil.epochs.epoch_positions", "virgil.epochs.EpochPositions", roots=["mathematics"])
def test_epoch_positions_invariant_to_underestimated_errors(positions):
    """Every quoted error three times too small (virgil#281): positions,
    covariances and gap_marginal are unchanged, gap grows by 9, χ²/N by 9,
    and the documented warning names both datasets (the fixture)."""
    pq, pu, _ = positions
    rel = max(np.max(np.abs(pu.gap_marginal / pq.gap_marginal - 1)), np.max(np.abs(pu.cov / pq.cov - 1)))
    shift = max(np.max(np.abs(pu.dra - pq.dra)), np.max(np.abs(pu.ddec - pq.ddec)))
    ratio = pu.gap / pq.gap
    record("max_rel_change_gap_marginal_cov", rel)
    record("max_position_shift_mas", shift)
    record("max_abs_gap_ratio_minus_9", float(np.max(np.abs(ratio - 9.0))))
    assert rel < 1e-6 and shift < 1e-6
    assert ratio == pytest.approx(9.0, rel=1e-6)
    for a, b in zip(pq.chi2_raw, pu.chi2_raw):
        assert b["all"] / a["all"] == pytest.approx(9.0, rel=1e-6)


# ------------------------------------------------------ multi-epoch orbits


EPOCH_TIMES = {"e0": T0, "e1": T0 + 150.0, "e2": T0 + 300.0, "e3": T0 + 500.0}


@pytest.fixture(scope="module")
def orbit_epochs(tmp_path_factory):
    """Four nights of a binary on ELEMENTS; e1's quoted errors are three
    times too small, so the quoted and marginal rankings weigh it apart."""
    tmp = tmp_path_factory.mktemp("orbit")
    paths = {}
    for k, (n, t) in enumerate(EPOCH_TIMES.items()):
        x, y = orbit_position(ELEMENTS, t)[0]
        paths[n] = observe(tmp / f"{n}.fits", FLUX, x, y, seed=20 + k)
    paths["e1"] = shrink_errors(paths["e1"], tmp / "e1_small.fits", 3.0)
    epochs = ve.Epochs({n: OIData(str(p)) for n, p in paths.items()}, times=EPOCH_TIMES)
    return epochs, {n: ours.load(p) for n, p in paths.items()}


def trial_orbits():
    """The truth, its mirror (Ω + 180°, ω + 180°) and perturbed orbits."""
    rng = np.random.default_rng(3)
    el = np.array(ELEMENTS)
    trials = [ELEMENTS, (*ELEMENTS[:4], ELEMENTS[4] + 180.0, ELEMENTS[5] + 180.0, ELEMENTS[6])]
    trials += [tuple(el * (1 + 0.05 * rng.standard_normal(7))) for _ in range(6)]
    return trials


def our_orbit_scores(ds, elements, flux=FLUX):
    """Our quoted log likelihood (-χ²/2, up to a constant) and m of all epochs."""
    q = m = 0.0
    for name, t in EPOCH_TIMES.items():
        x, y = orbit_position(elements, t)[0]
        rv, rp = blocks(ds[name], flux, x, y)
        q += -0.5 * (np.sum(rv**2) + np.sum(rp**2))
        m += -0.5 * (rv.size * np.log(np.sum(rv**2)) + rp.size * np.log(np.sum(rp**2)))
    return q, m


@pytest.mark.validates("virgil.epochs.Epochs", "virgil.models.OrbitalBinary", roots=["mathematics"])
def test_epochs_loglike_normalised(orbit_epochs):
    """Epochs.loglike of the orbital binary: the sum over epochs of the
    Gaussian V² density and the von Mises closure-phase density, κ = 1/σ²,
    with the companion at our orbit position at each snapshot time."""
    epochs, ds = orbit_epochs
    assert np.array_equal(epochs.times, list(EPOCH_TIMES.values()))
    got = float(epochs.loglike(OrbitalBinary(KeplerOrbit(*ELEMENTS, t_ref=T0), FLUX)))
    want = 0.0
    for name, t in EPOCH_TIMES.items():
        d = ds[name]
        x, y = orbit_position(ELEMENTS, t)[0]
        rv, rp = blocks(d, FLUX, x, y)
        want += gaussian_logl(rv, d["dv2"].ravel())(1.0) + von_mises_logl(rp, d["dcp"].ravel())(1.0)
    record("abs_diff", abs(got - want))
    assert got == pytest.approx(want, rel=1e-10, abs=1e-8)


@pytest.mark.parametrize("scales", ["quoted", "marginal"])
@pytest.mark.validates("virgil.epochs.rank_orbits", "virgil.epochs.RankedOrbits", "virgil.models.OrbitalBinary",
                       roots=["mathematics"])
def test_rank_orbits_against_our_orbit_chi2(orbit_epochs, scales):
    """The ranking and score differences of trial orbits equal ours: -Δχ²/2
    on the quoted errors, or Δm with each block's scale integrated out;
    RankedOrbits.positions are our Kepler positions at the snapshot times."""
    epochs, ds = orbit_epochs
    trials = trial_orbits()
    ranked = ve.rank_orbits(lambda o: OrbitalBinary(o, FLUX), epochs, [KeplerOrbit(*e, t_ref=T0) for e in trials],
                            scales=scales)
    mine = np.array([our_orbit_scores(ds, e)[0 if scales == "quoted" else 1] for e in trials])
    got = np.empty(len(trials))
    got[ranked.order] = ranked.loglike
    worst = np.max(np.abs((got - got[0]) - (mine - mine[0])))
    record("max_abs_diff_scores", worst)
    assert worst < 1e-6 * max(1.0, np.max(np.abs(mine - mine[0])))
    assert np.all(np.diff(mine[ranked.order]) <= 1e-6)  # best first, ties (the mirror) in any order
    pos = ranked.positions()
    want = np.stack([orbit_position(trials[i], list(EPOCH_TIMES.values())) for i in ranked.order])
    worst_pos = np.max(np.abs(pos - want))
    record("max_abs_position_mas", worst_pos)
    assert worst_pos < 1e-9


@pytest.mark.validates("virgil.epochs.rank_orbits", roots=["mathematics"], kind="guard")
def test_rank_orbits_quoted_and_marginal_differ_here(orbit_epochs):
    """Guard on the test data: the epoch with underestimated errors makes
    the quoted and marginal rankings disagree, so the two checks above test
    different things."""
    _, ds = orbit_epochs
    trials = trial_orbits()
    q, m = np.array([our_orbit_scores(ds, e) for e in trials]).T
    assert not np.array_equal(np.argsort(-q, kind="stable")[2:], np.argsort(-m, kind="stable")[2:])


CLUSTER_B = (*ELEMENTS[:5], ELEMENTS[5] + 35.0, ELEMENTS[6])  # another orbit: positions several mas away


def two_clusters():
    """Mode A (the truth, its mirror, two near copies) and mode B (another
    orbit and two near copies), interleaved."""
    rng = np.random.default_rng(5)

    def near(e):
        return tuple(np.array(e) * (1 + 0.004 * rng.standard_normal(7)))

    mirror = (*ELEMENTS[:4], ELEMENTS[4] + 180.0, ELEMENTS[5] + 180.0, ELEMENTS[6])
    return [near(CLUSTER_B), ELEMENTS, near(ELEMENTS), CLUSTER_B, mirror, near(CLUSTER_B), near(ELEMENTS)]


def our_chain_starts(ds, trials, n, min_distance):
    """Best first by our own quoted χ²; keep an orbit if, for every kept
    one, some snapshot position differs by at least ``min_distance``."""
    scores = np.array([our_orbit_scores(ds, e)[0] for e in trials])
    times = list(EPOCH_TIMES.values())
    kept = []
    for i in np.argsort(-scores, kind="stable"):
        p = orbit_position(trials[i], times)
        if all(np.max(np.hypot(*(p - orbit_position(trials[j], times)).T)) >= min_distance for j in kept):
            kept.append(i)
    return [kept[k % len(kept)] for k in range(n)]


@pytest.mark.validates("virgil.epochs.chain_starts", "virgil.epochs.RankedOrbits", "virgil.epochs.Epochs",
                       roots=["mathematics"])
def test_chain_starts_cover_both_modes(orbit_epochs):
    """Two clusters of trial orbits: two chains start one in each (the best
    of each, by our own χ²), never on the mirror of the truth, and a third
    chain repeats the first mode. The default distance is half our own
    finest λ/B_max."""
    epochs, ds = orbit_epochs
    trials = two_clusters()
    res = resolution_mas(ds.values())
    record("resolution_mas", res)
    assert epochs.resolution_mas == pytest.approx(res, rel=1e-6)
    ranked = ve.rank_orbits(lambda o: OrbitalBinary(o, FLUX), epochs, [KeplerOrbit(*e, t_ref=T0) for e in trials],
                            scales="quoted")
    times = list(EPOCH_TIMES.values())
    for n in (2, 3):
        starts = ve.chain_starts(ranked, n)
        mine = our_chain_starts(ds, trials, n, res / 2)
        got = np.array([orbit_position(elements_of(o), times) for o in starts.orbits])
        want = np.array([orbit_position(trials[i], times) for i in mine])
        worst = np.max(np.abs(got - want))
        record(f"max_abs_position_mas_n{n}", worst)
        assert worst < 1e-9
    a, b = (orbit_position(e, times) for e in (ELEMENTS, CLUSTER_B))
    assert np.max(np.hypot(*(a - b).T)) > res  # the clusters are distinct modes


# ------------------------------------------------- start_from_positions (slow)


@pytest.mark.slow
@pytest.mark.validates("virgil.epochs.start_from_positions", "virgil.epochs.OrbitStart", roots=["mathematics"],
                       tier="B")
def test_start_from_positions_recovers_the_orbit(orbit_epochs):
    """From positions to a refined orbit: the best fit's companion lies at
    our own orbit positions of the truth (within 0.1 mas at each epoch)
    and fits at least as well as the truth by our own χ² (the noise); the
    seeding positions are decisive."""
    dist = pytest.importorskip("numpyro.distributions")
    epochs, ds = orbit_epochs
    keys = ("period", "dt_peri", "ecc", "inc", "omega", "Omega", "a_mas")

    def scene(period, dt_peri, ecc, inc, omega, Omega, a_mas, flux):
        return OrbitalBinary(KeplerOrbit(period, dt_peri, ecc, inc, omega, Omega, a_mas, t_ref=T0), flux)

    priors = {"period": dist.Uniform(300.0, 3000.0), "dt_peri": dist.Uniform(-1500.0, 1500.0),
              "ecc": dist.Uniform(0.0, 0.95), "inc": dist.Uniform(0.0, 180.0), "omega": dist.Uniform(-360.0, 360.0),
              "Omega": dist.Uniform(-360.0, 360.0), "a_mas": dist.Uniform(1.0, 30.0), "flux": dist.Uniform(0.0, 1.0)}

    def start_values(orbit, flux):
        return {**dict(zip(keys, elements_of(orbit))), "flux": float(flux)}

    with pytest.warns(UserWarning, match="e1"):  # e1's errors are three times too small
        start = ve.start_from_positions(scene, priors, epochs, start_values, grid=GRID,
                                        periods=np.geomspace(500, 1500, 15),
                                        t_ref=T0, scales="marginal", eccs=np.arange(0.0, 0.6, 0.1), n_phase=24,
                                        n_candidates=50, n_refine=2)
    v = {k: float(x) for k, x in start.best.values.items()}
    fitted = tuple(v[k] for k in keys)
    times = list(EPOCH_TIMES.values())
    miss = np.max(np.hypot(*(orbit_position(fitted, times) - orbit_position(ELEMENTS, times)).T))
    q_fit, _ = our_orbit_scores(ds, fitted, v["flux"])
    q_true, _ = our_orbit_scores(ds, ELEMENTS)
    record("max_position_miss_mas", miss)
    record("delta_loglike_fit_minus_truth", q_fit - q_true)
    assert np.all(start.positions.gap_marginal > 5.0)
    assert miss < 0.1
    assert abs(v["flux"] / FLUX - 1) < 0.05
    assert q_fit >= q_true - 1e-6
    assert len(start.chain_values(3)) == 3
