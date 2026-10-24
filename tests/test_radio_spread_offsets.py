"""Independent check of ``ComplexVisData.toeplitz(..., offsets="spread", max_offset_mas=...)`` (register B5, derived).

Extends ``test_radio_innovations.py`` (chi-squared against a direct DFT at a few points, a control, and the
refusal guard) with the gradient, the fit and the documented bound. The reference is numpy only
(``crosscheck.radio_offsets``): the exact phase ramp ``exp(-2 pi i (u dRA_e + v dDec_e))`` per observation on a
pixel DFT, its analytic gradient (itself checked against central differences here), and the Fisher information.

Where the tolerances come from. virgil documents ``eps = 1e-14`` for the spreading kernel (``toeplitz``'s
docstring; w = 16) and a 4e-14 error of the kernel transform psi-hat "at 64^2 with 20-pixel shifts" (the design
note's table of numerical choices): ``EPS_EFF = 1e-14 + 4e-14``. A type-1/type-3 NUFFT with tolerance eps has an
error bounded by eps times the l1 norm of its input strengths, here the weights times the data and the weights
alone (``w_j V_j`` into the data term ``D``, ``w_j`` into the beam ``B``). For a ddid of observation ``e`` with
``g = flux gain_e`` the chi-squared ``C - 2 g I.D + g^2 I.B.I`` (unit-sum image ``I``) therefore differs from
the exact value by at most ``SCALE_d = 2 g sum w|V| + g^2 sum w`` times ``EPS_EFF``, plus float64 rounding of
the sums, ``u sqrt(n) (C + SCALE_d)`` with ``u = 2.2e-16``. The same errors enter the derivatives, each a
contraction with the same strengths: d chi2 / d b_p, d/d flux and d/d gain_e are bounded by twice SCALE
(divided by flux or gain_e for the last two), and the derivative with respect to an offset component by twice
SCALE times the largest phase rate ``2 pi MAS q_max`` (rad per mas), the factor the ramp's derivative brings;
the image gradient is in ``eta`` through the softmax, whose Jacobian has norm <= 1, so its l2 error is at most
``sqrt(P)`` times the per-pixel bound. The tolerances are these quantities, with no safety factor; the l1 norm
bounds the l2-relative error a NUFFT promises, so the bound is already conservative. Every gradient tolerance is
stated and checked relative to the l2 norm of that site's reference gradient, at a moved point (image, flux,
gains and offsets all away from the start, as vv#179 requires, because the start of a fit has near-cancelling
gradients).
"""

import numpy as np
import pytest
import test_radio_innovations as base

from crosscheck import radio_offsets as ro
from crosscheck.radio_innovations import MAS, softmax_image
from evidence.plugin import record

radio, toeplitz = base.radio, base.toeplitz
needs_toeplitz = base.needs_toeplitz
PIXEL, NPIX, BOUND, GAINS, FLUX = base.PIXEL, base.NPIX, base.OFFSET_BOUND, base.GAINS, base.FLUX
EPS_EFF = 1e-14 + 4e-14  # documented kernel eps plus documented psi-hat error
U64 = 2.2e-16
FRACTION = 0.5  # longest baseline at half the pixel Nyquist limit: well inside the margin


# ------------------------------------------------------------------ the reference itself


def _reference_case(seed=3, n=200):
    rng = np.random.default_rng(seed)
    q = 0.4 / (PIXEL * MAS) * np.sqrt(rng.uniform(0, 1, n))
    phi = rng.uniform(0, 2 * np.pi, n)
    obs = rng.integers(0, 3, n)
    return dict(u=q * np.cos(phi), v=q * np.sin(phi), obs=obs, weight=10 ** rng.uniform(2.0, 3.5, n),
                vis=0.1 * (rng.normal(size=n) + 1j * rng.normal(size=n)), eta=rng.normal(size=(6, 6)),
                gain=np.array([1.0, 1.1, 0.9]), offset=np.array([[0.0, 0.0], [3.0, -2.0], [-4.0, 5.0]]), flux=0.8)


@pytest.mark.validates("crosscheck.radio_offsets", roots=["mathematics"], kind="reference")
def test_the_reference_gradient_equals_central_differences_of_the_reference_chi2():
    """Analytic gradient of the numpy chi-squared against central differences, every site (36 pixels, flux, 3
    gains, 3 x 2 offsets), steps h = 1e-5 (pixels, flux, gains) and 1e-4 mas. The central difference has
    truncation error (h^2/6) f''' and rounding u chi2 / h; f''' is at most (phase rate 2 pi MAS q_max <= 0.5 rad /
    mas)^2 times the gradient for offsets and O(1) times it for the others, so truncation is <= 1e-9 of the gradient and
    rounding about 1e-11 chi2 / |grad| (chi2/|grad| ~ 10): the tolerance 1e-7 of the l2 norm of each site's gradient
    is two orders above both."""
    c = _reference_case()
    args = (c["u"], c["v"], c["vis"], c["weight"], c["obs"])

    def f(eta=c["eta"], flux=c["flux"], gain=c["gain"], offset=c["offset"]):
        return ro.chi2_and_gradient(*args, eta, PIXEL, flux, gain, offset)[0]

    _, grads = ro.chi2_and_gradient(*args, c["eta"], PIXEL, c["flux"], c["gain"], c["offset"])
    worst = 0.0
    for site, h in (("eta", 1e-5), ("flux", 1e-5), ("gain", 1e-5), ("offset", 1e-4)):
        x = np.array(c["eta"] if site == "eta" else c[site], float)
        fd = np.zeros_like(x)
        for i in np.ndindex(x.shape):
            plus, minus = x.copy(), x.copy()
            plus[i] += h
            minus[i] -= h
            fd[i] = (f(**{site: plus}) - f(**{site: minus})) / (2 * h)
        err = np.linalg.norm(fd - grads[site]) / np.linalg.norm(grads[site])
        record(f"reference_gradient_vs_finite_difference_{site}", err)
        worst = max(worst, err)
    assert worst < 1e-7


# ------------------------------------------------------------------ B5 (1): chi2 and gradient


def _case(offsets):
    case = base._toeplitz_case(FRACTION, offsets)
    case["tdata"] = case["data"].toeplitz(case["template"], case["nuis"], offsets="spread", max_offset_mas=BOUND)
    return case


def _values_at(case, point, rng):
    """The three points: the case's own start; a moved point (image, flux, gains, offsets all changed, offsets
    inside half the bound); the edge, every offset component at +-0.9 of the bound."""
    v = dict(case["values"])
    if point == "start":
        return v
    v["log_brightness"] = case["eta"] + 0.7 * rng.normal(size=case["eta"].shape)
    v["cal.flux"] = 1.15 * FLUX
    v["cal.gain"] = np.array([1.05, 0.96]) * v["cal.gain"]
    if point == "moved":
        v["cal.offset"] = rng.uniform(-0.5 * BOUND, 0.5 * BOUND, size=(2, 2))
    else:
        v["cal.offset"] = 0.9 * BOUND * rng.choice([-1.0, 1.0], size=(2, 2))
    return v


def _tolerances(case, values):
    """Per-ddid chi2 tolerance and per-site gradient (l2) tolerance, from the docstring's derivation."""
    flux, gain = float(values["cal.flux"]), np.concatenate([[1.0], np.asarray(values["cal.gain"])])
    w, absv, ddid = case["weight"], np.abs(case["vis"]), case["ddid"]
    g = flux * gain
    chi2_tol, scale_obs = [], np.zeros(3)
    for d in range(4):
        m, e = ddid == d, base.DDIDS[d]
        scale = 2 * g[e] * np.sum(w[m] * absv[m]) + g[e] ** 2 * np.sum(w[m])
        c = np.sum(w[m] * absv[m] ** 2)
        chi2_tol.append(EPS_EFF * scale + U64 * np.sqrt(m.sum()) * (c + scale))
        scale_obs[e] += scale
    rate = 2 * np.pi * MAS * np.max(np.hypot(case["u"], case["v"]))
    pixels = case["eta"].size
    tol = {"log_brightness": np.sqrt(pixels) * 2 * EPS_EFF * scale_obs.sum(),
           "cal.flux": 2 * EPS_EFF * np.sum(scale_obs / flux),
           "cal.gain": 2 * EPS_EFF * scale_obs[1:] / gain[1:],
           "cal.offset": 2 * EPS_EFF * rate * scale_obs[1:, None] * np.ones((1, 2))}
    tol["cal.gain"], tol["cal.offset"] = np.linalg.norm(tol["cal.gain"]), np.linalg.norm(tol["cal.offset"])
    return np.array(chi2_tol), tol


@needs_toeplitz
@pytest.mark.validates("virgil.radio_toeplitz.ToeplitzVisData", roots=["mathematics"])
@pytest.mark.parametrize("point", ["start", "moved", "edge"])
def test_spread_offsets_chi2_and_gradient_equal_the_exact_reference_within_the_documented_error(point):
    """Per-ddid chi-squared and the gradient of the chi-squared in every fitted site (image logits, flux,
    gains, offsets), 600 points, 16 x 16 image, longest baseline at half Nyquist, at the start, at a moved point
    and at an edge point (offsets +-0.9 of the 20 mas bound). Chi-squared must be within ``SCALE_d EPS_EFF``
    (+ rounding) per ddid, and each site's gradient within its derived l2 tolerance (module docstring: kernel
    eps 1e-14 + psi-hat 4e-14, l1 norm of the strengths, phase-rate factor for offsets); both are also
    asserted relative to the gradient norm at 1e-8, the vv#179 limit, and the gradient norm must be large
    (the point is not a stationary one) so that the relative statement means something. The moved and edge
    points change image, flux, gains and offsets; a gradient that is right at the start only fails there."""
    import equinox as eqx
    import jax.numpy as jnp

    case = _case([[3.1, -2.2], [-1.7, 4.4]])
    rng = np.random.default_rng({"start": 0, "moved": 1, "edge": 2}[point])
    values = _values_at(case, point, rng)
    tdata, nuis, scene = case["tdata"], case["nuis"], case["scene"]
    gain = np.concatenate([[1.0], np.asarray(values["cal.gain"])])
    offset = np.vstack([[0.0, 0.0], np.asarray(values["cal.offset"])])
    flux = float(values["cal.flux"])
    chi2_ref = base.ri.direct_chi2_by_ddid(case["u"], case["v"], case["vis"], case["weight"], case["ddid"], case["obs"],
                                           softmax_image(values["log_brightness"]), PIXEL, flux, gain, offset)
    chi2_got = tdata.chi2_by_ddid(scene, values, nuis)[0]
    _, ref = ro.chi2_and_gradient(case["u"], case["v"], case["vis"], case["weight"], case["obs"], values["log_brightness"],
                                  PIXEL, flux, gain, offset)
    jvalues = {k: jnp.asarray(v, float) for k, v in values.items()}
    term = tdata.term(scene, nuis)
    got = eqx.filter_grad(lambda v: term.chi2_and_log_norm(v)[0])(jvalues)
    chi2_tol, tol = _tolerances(case, values)
    chi2_err = np.abs(chi2_got - chi2_ref)
    record(f"spread_chi2_error_over_tolerance_{point}", float((chi2_err / chi2_tol).max()))
    assert np.all(chi2_err < chi2_tol), (chi2_err, chi2_tol)
    assert np.abs(chi2_got / chi2_ref - 1).max() < 1e-8
    reference = {"log_brightness": ref["eta"], "cal.flux": np.array(ref["flux"]), "cal.gain": ref["gain"][1:],
                 "cal.offset": ref["offset"][1:]}
    for site, exact in reference.items():
        diff = np.linalg.norm(np.asarray(got[site]) - exact)
        norm = np.linalg.norm(exact)
        record(f"spread_grad_error_over_tolerance_{site}_{point}", float(diff / tol[site]))
        record(f"spread_grad_relative_error_{site}_{point}", float(diff / norm))
        assert norm > 1.0, (site, norm)  # a gradient of order one or more on a loss of ~3e4: not a stationary point
        assert diff < tol[site], (site, diff, tol[site])
        assert diff / norm < 1e-8, (site, diff / norm)


# ------------------------------------------------------------------ B5 (2): the fit recovers injected offsets

K_FITS = 16
TRUE_OFFSETS = np.array([[0.0, 0.0], [3.1, -2.2], [-1.7, 4.4]])


@needs_toeplitz
@pytest.mark.validates("virgil.radio_toeplitz.ToeplitzVisData", roots=["statistics", "mathematics"])
def test_fitted_offsets_recover_injected_offsets_within_the_fisher_scatter():
    """Simulate (numpy, exact DFT, Gaussian noise of variance 1/w in each of Re and Im, so that exp(-chi2/2) is
    the likelihood) 600 visibilities of a 16 x 16 image in three observations with injected offsets (3.1, -2.2)
    and (-1.7, 4.4) mas, flux 0.7 and gains 1.08, 0.93, and fit flux, gains and offsets (image known) with
    virgil's spread-offsets Toeplitz term and L-BFGS from flux 1, unit gains, zero offsets, on K = 16 noise
    realisations. The expected scatter is the Cramer-Rao one, the inverse of the Fisher information F of
    (flux, gains, offsets) from the reference's analytic derivatives (about 0.17 mas per offset component here;
    the maximum-likelihood estimator is unbiased and attains it to first order at this signal-to-noise). So:
    (a) d^2 = (theta_hat - theta)^T F (theta_hat - theta) ~ chi-squared with 7 degrees of freedom per fit, so its
    mean over K fits, divided by 7, is 1 +- sqrt(2 / (7 K)) = 0.134: accepted within 3 of those (0.40);
    (b) no bias: K times the Mahalanobis norm of the mean error is chi-squared with 7 dof, accepted below its
    99.9% point (24.3); (c) no component more than 4.5 standard deviations off in any fit (two-sided
    probability 7e-6 per component, 112 components); (d) the fitted values equal the maximum likelihood of the
    numpy chi-squared (scipy BFGS, analytic gradient) from the same start to 1e-4 sigma in the Fisher metric,
    which is the exactness of the optimizer's stopping rule (the documented floor sqrt(eps) of the largest
    starting gradient per data point; measured 1e-5 sigma or less), so a wrong or biased spread term moves
    the estimate by order sigma, four orders above."""
    from scipy.optimize import minimize
    from scipy.stats import chi2 as chi2_law
    from virgil import Image, fit

    case = base._toeplitz_case(FRACTION, TRUE_OFFSETS[1:])
    eta, nuis, weight, obs = case["eta"], case["nuis"], case["weight"], case["obs"]
    tdata = case["data"].toeplitz(case["template"], nuis, offsets="spread", max_offset_mas=BOUND)
    args = (case["u"], case["v"], weight, obs, eta, PIXEL)
    model, _ = ro.model_visibilities(case["u"], case["v"], obs, eta, PIXEL, FLUX, GAINS, TRUE_OFFSETS)
    fisher = ro.nuisance_fisher(*args, FLUX, GAINS, TRUE_OFFSETS)
    sigma = np.sqrt(np.diag(np.linalg.inv(fisher)))
    truth = np.concatenate([[FLUX], GAINS[1:], TRUE_OFFSETS[1:].ravel()])
    record("fisher_sigma_offset_mas_min", float(sigma[3:].min()))
    record("fisher_sigma_offset_mas_max", float(sigma[3:].max()))
    scene = lambda values: Image(eta, pixel_scale_mas=PIXEL)  # noqa: E731
    priors = nuis.priors(offset=(-BOUND, BOUND))
    init = {"cal.flux": 1.0, "cal.gain": np.ones(2), "cal.offset": np.zeros((2, 2))}
    x0 = np.concatenate([[1.0], np.ones(2), np.zeros(4)])

    def reference_loss(x, vis):
        c, g = ro.chi2_and_gradient(case["u"], case["v"], vis, weight, obs, eta, PIXEL, x[0], np.concatenate([[1.0], x[1:3]]),
                                    np.vstack([[0.0, 0.0], x[3:].reshape(2, 2)]))
        return 0.5 * c, 0.5 * np.concatenate([[g["flux"]], g["gain"][1:], g["offset"][1:].ravel()])

    rng = np.random.default_rng(11)
    errors, apart = [], []
    for _ in range(K_FITS):
        vis = model + (rng.normal(size=len(model)) + 1j * rng.normal(size=len(model))) / np.sqrt(weight)
        term = tdata.rebuild(vis, weight).term(scene, nuis)
        res = fit(lambda **values: scene(values), priors, (), init=init, likelihoods=[term], method="lbfgs", gtol=1e-9, max_steps=500)
        est = np.concatenate([[float(res.values["cal.flux"])], np.asarray(res.values["cal.gain"]), np.ravel(res.values["cal.offset"])])
        ref = minimize(reference_loss, x0, args=(vis,), jac=True, method="BFGS", options={"gtol": 1e-9, "maxiter": 1000}).x
        errors.append(est - truth)
        apart.append(np.sqrt((est - ref) @ fisher @ (est - ref)))
    errors = np.array(errors)
    d2 = np.einsum("ki,ij,kj->k", errors, fisher, errors)
    mean_d2 = d2.mean() / 7
    bias = K_FITS * errors.mean(0) @ fisher @ errors.mean(0)
    worst_z = np.abs(errors / sigma).max()
    record("recovery_mean_d2_over_7", float(mean_d2))
    record("recovery_bias_statistic", float(bias))
    record("recovery_max_abs_z", float(worst_z))
    record("recovery_fit_vs_reference_ml_max_sigma", float(max(apart)))
    assert abs(mean_d2 - 1) < 3 * np.sqrt(2 / (7 * K_FITS))
    assert bias < chi2_law.ppf(0.999, 7)
    assert worst_z < 4.5
    assert max(apart) < 1e-4


# ------------------------------------------------------------------ B5 (3): the bound


def _bound_case(offsets, max_offset_mas=BOUND):
    case = base._toeplitz_case(FRACTION, offsets)
    case["tdata"] = case["data"].toeplitz(case["template"], case["nuis"], offsets="spread", max_offset_mas=max_offset_mas)
    return case


@needs_toeplitz
@pytest.mark.validates("virgil.radio_toeplitz.ToeplitzVisData", roots=["mathematics"])
@pytest.mark.parametrize("offsets", [[[BOUND, -BOUND], [-BOUND, BOUND]], [[0.0, BOUND], [-BOUND, 0.0]]],
                         ids=["corners", "axes"])
def test_offsets_exactly_at_the_bound_evaluate_and_equal_the_reference(offsets):
    """The documented bound is the largest offset of any observation (``max_offset_mas``, per axis for an
    unrotated image): offsets of exactly +-20 mas, in every sign and on both axes, evaluate, and chi-squared and
    the offset gradient equal the reference within the tolerances of the module docstring (the point of the
    design is that the spread grid is largest at the bound)."""
    import equinox as eqx
    import jax.numpy as jnp

    case = _bound_case(offsets)
    values = dict(case["values"])
    gain, off = np.concatenate([[1.0], values["cal.gain"]]), np.vstack([[0.0, 0.0], values["cal.offset"]])
    ref = base.ri.direct_chi2_by_ddid(case["u"], case["v"], case["vis"], case["weight"], case["ddid"], case["obs"],
                                      softmax_image(case["eta"]), PIXEL, FLUX, gain, off)
    got = case["tdata"].chi2_by_ddid(case["scene"], values, case["nuis"])[0]
    chi2_tol, tol = _tolerances(case, values)
    term = case["tdata"].term(case["scene"], case["nuis"])
    g = eqx.filter_grad(lambda v: term.chi2_and_log_norm(v)[0])({k: jnp.asarray(v, float) for k, v in values.items()})
    _, exact = ro.chi2_and_gradient(case["u"], case["v"], case["vis"], case["weight"], case["obs"], case["eta"], PIXEL, FLUX, gain, off)
    diff = np.linalg.norm(np.asarray(g["cal.offset"]) - exact["offset"][1:])
    record("at_bound_chi2_error_over_tolerance", float((np.abs(got - ref) / chi2_tol).max()))
    record("at_bound_offset_gradient_error_over_tolerance", float(diff / tol["cal.offset"]))
    assert np.all(np.abs(got - ref) < chi2_tol)
    assert diff < tol["cal.offset"]


@needs_toeplitz
@pytest.mark.validates("virgil.radio_toeplitz.ToeplitzVisData", roots=["mathematics"], kind="guard")
@pytest.mark.parametrize("which", [(0, 0, 1.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 1.0)],
                         ids=["obs1_x+", "obs1_y-", "obs2_x-", "obs2_y+"])
def test_an_offset_just_past_the_bound_is_refused_in_value_and_gradient(which):
    """Documented: offsets beyond the precompute's bound raise at run time (``max_offset_mas``: the largest
    offset of any observation). 1e-6 mas past +-20 mas, on either axis of either free observation and with either
    sign, the chi-squared and its gradient both raise an error naming the bound, while the same offsets 1e-6 mas
    inside it evaluate to finite numbers."""
    import equinox as eqx
    import jax.numpy as jnp

    row, col, sign = which
    inside = np.zeros((2, 2))
    inside[row, col] = sign * (BOUND - 1e-6)
    past = np.zeros((2, 2))
    past[row, col] = sign * (BOUND + 1e-6)
    case = _bound_case(inside)
    tdata, nuis, scene = case["tdata"], case["nuis"], case["scene"]
    assert np.all(np.isfinite(tdata.chi2_by_ddid(scene, case["values"], nuis)[0]))
    values = dict(case["values"], **{"cal.offset": past})
    with pytest.raises(Exception, match="(?i)beyond the bound"):  # noqa: B017  (a checkify error from inside jit)
        tdata.chi2_by_ddid(scene, values, nuis)
    term = tdata.term(scene, nuis)
    with pytest.raises(Exception, match="(?i)beyond the bound"):  # noqa: B017
        eqx.filter_grad(lambda v: term.chi2_and_log_norm(v)[0])({k: jnp.asarray(v, float) for k, v in values.items()})


@needs_toeplitz
@pytest.mark.validates("virgil.radio_toeplitz.ToeplitzVisData", roots=["mathematics"], kind="guard")
def test_the_default_bound_is_the_prior_box_of_the_nuisance():
    """Documented: without ``max_offset_mas`` the bound is that of ``nuisance.priors()`` (the offsets' prior
    box, read here from the prior's own limits, 60.18 mas for this nuisance): an offset a part in 1e9 inside the
    box's largest edge evaluates and equals the reference (to 1e-8 relative), and 1% beyond it raises. (Exactly
    at a bound that is not a binary fraction the comparison is at the mercy of the last bit: an offset equal
    to this 60.18 raised, while the exactly representable 20.0 evaluates; the claim is therefore made a part
    in 1e9 inside.)"""
    case = base._toeplitz_case(FRACTION, [[0.0, 0.0], [0.0, 0.0]])
    tdata = case["data"].toeplitz(case["template"], case["nuis"], offsets="spread")
    prior = case["nuis"].priors()["cal.offset"]
    edge = float(np.max(np.abs([np.asarray(prior.low), np.asarray(prior.high)])))
    record("default_bound_mas", edge)
    ok = (1 - 1e-9) * np.array([[edge, 0.0], [0.0, 0.0]])
    gain, off = np.concatenate([[1.0], case["values"]["cal.gain"]]), np.vstack([[0.0, 0.0], ok])
    values = dict(case["values"], **{"cal.offset": ok})
    ref = base.ri.direct_chi2_by_ddid(case["u"], case["v"], case["vis"], case["weight"], case["ddid"], case["obs"],
                                      softmax_image(case["eta"]), PIXEL, FLUX, gain, off)
    got = tdata.chi2_by_ddid(case["scene"], values, case["nuis"])[0]
    assert np.abs(got / ref - 1).max() < 1e-8
    with pytest.raises(Exception, match="(?i)beyond the bound"):  # noqa: B017
        tdata.chi2_by_ddid(case["scene"], dict(values, **{"cal.offset": 1.01 * edge * np.array([[1.0, 0.0], [0.0, 0.0]])}), case["nuis"])
