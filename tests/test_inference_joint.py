"""Inference helpers, joint likelihoods over several datasets and small
detection helpers, against NumPy and SciPy on problems with closed forms
and against our own observables (crosscheck.sky, read with astropy by
crosscheck.chi2) on two simulated three-telescope files.

* hessian_matrix: the analytic Hessian of a polynomial-and-sine objective.
* regularized_inverse: numpy.linalg.inv(M + ridge I); a RuntimeWarning for
  an indefinite matrix.
* laplace_covariance: for a model linear in its parameters, the inverse
  Hessian of the negative log-likelihood is exactly (J^T C^-1 J)^-1.
  Control: the unweighted (J^T J)^-1 and the wrongly weighted (J^T C J)^-1
  must not match.
* gaussian_fisher: J^T Sigma^-1 J with J the central-difference Jacobian of
  our own V^2 and closure phases; and exactly on a linear model. Control:
  the unweighted J^T J must not match.
* fisher_projection: P P^T = F^-1 and P^T F P = I; flat directions floored
  at eps times the largest eigenvalue, with a RuntimeWarning.
* laplace_parameter_uncertainty: (1/2 d^2 chi^2 / d theta^2)^-1/2, our
  chi-squared by central differences, the other parameters fixed.
* joint_loglike: the sum over datasets of SciPy's Gaussian (V^2) and von
  Mises (closure phase, concentration 1/sigma^2) log densities, which is
  -1/2 our chi-squared plus their normalisations, with shared and
  per-dataset parameters, and with inflated errors. Control: swapping the
  per-dataset parameters must not match.
* joint_prediction, joint_data, joint_errors: the same values as our
  files and our observables, in an order where (prediction - data) /
  errors gives our chi-squared (V^2 residuals, closure-phase chords).
* noise_sites, noise_for: the documented site names, and the per-dataset
  terms they give reproduce our likelihood with each dataset's errors
  scaled.
* posterior_predictive_summary: the mean and standard deviation over the
  samples of our own V^2 and closure phases.
* rescale_errors: s = sqrt(chi^2 / n) per block of our chi-squared, after
  which each block has chi^2 / n = 1.
* injection_grid: every separation and flux, ordered, at PAs recovered by
  the angle of (dra, ddec), uniform on [0, 360) (Kolmogorov-Smirnov).
* bootstrap_null: a sign flip keeps each whitened residual's magnitude
  about our prediction (phases wrapped), flipping about half of them; a
  resample draws only residuals from the same block.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from scipy import special, stats

from crosscheck import chi2 as ours, simulate as sim, sky
from evidence.plugin import record

vm = pytest.importorskip("virgil.models")
from virgil import detection, inference, likelihood  # noqa: E402
from virgil.oidata import OIData  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

pytestmark = pytest.mark.x64

UTS3 = np.array([[-9.925, -20.335], [14.887, 30.502], [103.306, 43.999]])
POS = (6.0, -4.0)
FLUXES = (0.03, 0.02)
BANDS = ((1.5e-6, 1.8e-6), (2.0e-6, 2.4e-6))


def binary_vis(x, y, f):
    def vis(u, v, w):
        return (sky.vis_point(u, v, w) + f * sky.vis_point(u, v, w, x, y)) / (1 + f)

    return vis


def our_observables(d, vis):
    """Our model V^2 and closure phases (sum of the three baseline phases),
    flattened as our loaded arrays."""
    v2 = np.abs(vis(d["u"], d["v"], d["wl"])) ** 2
    cp = (np.angle(vis(d["u1"], d["v1"], d["wl3"])) + np.angle(vis(d["u2"], d["v2_"], d["wl3"]))
          - np.angle(vis(d["u1"] + d["u2"], d["v1"] + d["v2_"], d["wl3"])))
    return v2.ravel(), cp.ravel()


def our_loglike(d, vis, vis_scale=1.0, phi_scale=1.0):
    """SciPy's Gaussian in V^2 and von Mises (kappa = 1/sigma^2) in the
    closure phases, the errors multiplied by the scales."""
    v2, cp = our_observables(d, vis)
    sv, sp = vis_scale * d["dv2"].ravel(), phi_scale * d["dcp"].ravel()
    return float(np.sum(stats.norm.logpdf(d["v2"].ravel(), v2, sv))
                 + np.sum(stats.vonmises.logpdf(d["cp"].ravel(), 1.0 / sp**2, loc=cp)))


def order(values, reference):
    """Indices into ``reference`` of each of ``values`` (all distinct)."""
    idx = np.array([np.argmin(np.abs(reference - x)) for x in values])
    assert np.allclose(reference[idx], values, rtol=0, atol=1e-12) and len(set(idx)) == len(idx)
    return idx


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("joint")
    obs, loaded = [], []
    for k, (band, f) in enumerate(zip(BANDS, FLUXES)):
        path = tmp / f"d{k}.fits"
        sim.observe(path, binary_vis(*POS, f), UTS3, hour_angles_h=np.linspace(-3, 3, 5),
                    wavelengths=np.linspace(*band, 3), dec_deg=-50.0, sigma_v2=0.01, sigma_cp_deg=0.5,
                    rng=np.random.default_rng(20 + k))
        # Per-element errors that differ inside a file and between the two files, so that
        # any misalignment of errors with data (or of one dataset with the other) shows.
        rng = np.random.default_rng(100 + k)
        with fits.open(path, mode="update") as hdul:
            for ext, col in (("OI_VIS2", "VIS2ERR"), ("OI_T3", "T3PHIERR")):
                err = hdul[ext].data[col]
                err[...] = err * rng.uniform(0.5, 2.0, err.shape)
        obs.append(OIData(str(path)))
        loaded.append(ours.load(path))
    return obs, loaded


def model_fn(p, i):
    return vm.BinaryModelCartesian(p["dra"], p["ddec"], p["flux"][i])


PARAMS = {"dra": jnp.array(5.8), "ddec": jnp.array(-4.1), "flux": jnp.array([0.028, 0.021])}


def our_vis(p, i):
    return binary_vis(float(p["dra"]), float(p["ddec"]), float(p["flux"][i]))


# ---------------------------------------------------------------- inference


@pytest.mark.validates("virgil.inference.hessian_matrix", roots=["mathematics"])
def test_hessian_matrix_is_the_analytic_hessian():
    A = np.array([[3.0, 1.0, 0.5], [1.0, 2.0, -0.3], [0.5, -0.3, 1.5]])

    def f(x):
        return 0.5 * x @ (jnp.asarray(A) @ x) + jnp.sum(x**4) + x[0] * x[1] * jnp.sin(x[2])

    worst = 0.0
    for x in (np.array([0.3, -0.7, 1.1]), np.array([-1.2, 0.4, 2.5])):
        want = A + np.diag(12 * x**2)
        want[0, 1] += np.sin(x[2])
        want[1, 0] += np.sin(x[2])
        want[0, 2] += x[1] * np.cos(x[2])
        want[2, 0] += x[1] * np.cos(x[2])
        want[1, 2] += x[0] * np.cos(x[2])
        want[2, 1] += x[0] * np.cos(x[2])
        want[2, 2] += -x[0] * x[1] * np.sin(x[2])
        worst = max(worst, np.max(np.abs(np.asarray(inference.hessian_matrix(f, jnp.asarray(x))) - want)))
    record("max_abs_dH", worst)
    assert worst < 1e-12


@pytest.mark.validates("virgil.inference.regularized_inverse", roots=["mathematics"])
def test_regularized_inverse_and_its_warning():
    rng = np.random.default_rng(3)
    B = rng.normal(size=(4, 4))
    M = B @ B.T + 0.1 * np.eye(4)
    worst = 0.0
    for ridge in (0.0, 1e-10, 0.3):
        got = np.asarray(inference.regularized_inverse(jnp.asarray(M), ridge=ridge))
        want = np.linalg.inv(M + ridge * np.eye(4))
        worst = max(worst, np.max(np.abs(got - want)) / np.max(np.abs(want)))
    record("max_rel_dinv", worst)
    assert worst < 1e-12
    with pytest.warns(RuntimeWarning):
        inference.regularized_inverse(jnp.asarray(np.diag([1.0, -2.0, 3.0])), ridge=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        inference.regularized_inverse(jnp.asarray(M))  # positive definite: silent


def linear_problem(seed=4, n=30, p=3):
    rng = np.random.default_rng(seed)
    J = rng.normal(size=(n, p)) * np.array([1.0, 10.0, 0.1])
    sigma = rng.uniform(0.5, 2.0, n)
    y = J @ np.array([1.0, -0.2, 3.0]) + sigma * rng.normal(size=n)
    return J, sigma, y


@pytest.mark.validates("virgil.inference.laplace_covariance", roots=["mathematics"])
def test_laplace_covariance_is_exact_for_a_linear_model():
    J, sigma, y = linear_problem()
    Ci = np.diag(1 / sigma**2)
    want = np.linalg.inv(J.T @ Ci @ J)
    xhat = want @ J.T @ Ci @ y

    def nll(x):
        r = (jnp.asarray(y) - jnp.asarray(J) @ x) / jnp.asarray(sigma)
        return 0.5 * jnp.sum(r**2)

    worst = 0.0
    for x in (xhat, xhat + 0.5):  # exact anywhere: the Hessian is constant
        got = np.asarray(inference.laplace_covariance(nll, jnp.asarray(x), ridge=0.0))
        worst = max(worst, np.max(np.abs(got - want) / np.sqrt(np.outer(np.diag(want), np.diag(want)))))
    got = np.asarray(inference.laplace_covariance(nll, jnp.asarray(xhat)))  # default ridge 1e-10
    default = np.max(np.abs(got - want) / np.sqrt(np.outer(np.diag(want), np.diag(want))))
    record("max_rel_dcov", worst)
    record("max_rel_dcov_default_ridge", default)
    assert worst < 1e-10
    assert default < 1e-6


@pytest.mark.validates("virgil.inference.laplace_covariance", roots=["mathematics"], kind="control")
def test_laplace_covariance_is_not_unweighted():
    J, sigma, y = linear_problem()

    def nll(x):
        r = (jnp.asarray(y) - jnp.asarray(J) @ x) / jnp.asarray(sigma)
        return 0.5 * jnp.sum(r**2)

    got = np.asarray(inference.laplace_covariance(nll, jnp.zeros(3), ridge=0.0))
    for wrong in (np.linalg.inv(J.T @ J), np.linalg.inv(J.T @ np.diag(sigma**2) @ J)):
        assert np.max(np.abs(got - wrong) / np.sqrt(np.outer(np.diag(got), np.diag(got)))) > 0.05


@pytest.mark.validates("virgil.inference.gaussian_fisher", roots=["mathematics"])
def test_gaussian_fisher_is_exact_for_a_linear_model():
    J, sigma, _ = linear_problem()
    params = {"a": jnp.array(0.1), "b": jnp.array([0.2, -0.4])}

    def pred(p):
        return jnp.asarray(J) @ jnp.concatenate([p["a"][None], p["b"]])

    F, unravel = inference.gaussian_fisher(pred, params, jnp.asarray(sigma))
    restored = unravel(jnp.array([1.0, 2.0, 3.0]))
    assert float(restored["a"]) == 1.0 and np.array_equal(np.asarray(restored["b"]), [2.0, 3.0])
    want = J.T @ np.diag(1 / sigma**2) @ J
    rel = np.max(np.abs(np.asarray(F) - want)) / np.max(np.abs(want))
    record("max_rel_dF", rel)
    assert rel < 1e-12
    F2, _ = inference.gaussian_fisher(pred, params, jnp.asarray(sigma), ridge=0.25)
    assert np.allclose(np.asarray(F2), want + 0.25 * np.eye(3), rtol=1e-12, atol=0)


def our_fisher(loaded, p, unravel, flat, h=1e-6):
    """J^T Sigma^-1 J, summed over both datasets, with J by central
    differences of our V^2 and closure phases in the flattened parameters."""
    F = np.zeros((flat.size, flat.size))
    for i, d in enumerate(loaded):
        cols = []
        for k in range(flat.size):
            up, down = unravel(flat + h * np.eye(flat.size)[k]), unravel(flat - h * np.eye(flat.size)[k])
            a, b = (np.concatenate(our_observables(d, our_vis(q, i))) for q in (up, down))
            cols.append((a - b) / (2 * h))
        Jk = np.array(cols).T
        w = 1 / np.concatenate([d["dv2"].ravel(), d["dcp"].ravel()]) ** 2
        F += Jk.T @ (w[:, None] * Jk)
    return F


@pytest.mark.validates("virgil.inference.gaussian_fisher", "virgil.likelihood.joint_prediction",
                       "virgil.likelihood.joint_errors", roots=["mathematics"])
def test_gaussian_fisher_of_the_joint_prediction(pair):
    obs, loaded = pair

    def pred(p):
        return likelihood.joint_prediction(p, obs, model_fn)

    F, unravel = inference.gaussian_fisher(pred, PARAMS, likelihood.joint_errors(obs))
    flat = jax.flatten_util.ravel_pytree(PARAMS)[0]
    want = our_fisher(loaded, PARAMS, unravel, np.asarray(flat))
    rel = np.max(np.abs(np.asarray(F) - want) / np.sqrt(np.outer(np.diag(want), np.diag(want))))
    record("max_rel_dF", rel)
    assert rel < 1e-6


@pytest.mark.validates("virgil.inference.gaussian_fisher", roots=["mathematics"], kind="control")
def test_gaussian_fisher_is_not_unweighted(pair):
    obs, _ = pair

    def pred(p):
        return likelihood.joint_prediction(p, obs, model_fn)

    F, _ = inference.gaussian_fisher(pred, PARAMS, likelihood.joint_errors(obs))
    J = np.asarray(jax.jacfwd(lambda x: pred(jax.flatten_util.ravel_pytree(PARAMS)[1](x)))(
        jax.flatten_util.ravel_pytree(PARAMS)[0]))
    wrong = J.T @ J
    assert np.max(np.abs(np.asarray(F) - wrong) / np.sqrt(np.outer(np.diag(F), np.diag(F)))) > 0.5


@pytest.mark.validates("virgil.inference.fisher_projection", roots=["mathematics"])
def test_fisher_projection_whitens_the_fisher_matrix():
    J, sigma, _ = linear_problem()
    F = J.T @ np.diag(1 / sigma**2) @ J
    P = np.asarray(inference.fisher_projection(jnp.asarray(F)))
    cov = np.linalg.inv(F)
    a = np.max(np.abs(P @ P.T - cov) / np.sqrt(np.outer(np.diag(cov), np.diag(cov))))
    b = np.max(np.abs(P.T @ F @ P - np.eye(3)))
    record("max_rel_dPPt", a)
    record("max_abs_dPtFP", b)
    assert a < 1e-10 and b < 1e-10
    # a flat direction: floored at eps * largest eigenvalue, with a warning
    with pytest.warns(RuntimeWarning):
        P = np.asarray(inference.fisher_projection(jnp.diag(jnp.array([4.0, 0.0])), eps=1e-6))
    assert np.all(np.isfinite(P))
    assert np.allclose(np.sort(np.diag(P @ P.T)), [0.25, 1 / 4e-6], rtol=1e-10)


@pytest.mark.validates("virgil.inference.laplace_parameter_uncertainty", roots=["mathematics"])
def test_laplace_parameter_uncertainty_is_the_curvature_of_our_chi2(pair):
    obs, loaded = pair
    data, d = obs[0], loaded[0]
    names = ["dra", "ddec", "flux"]
    x0 = np.array([5.9, -4.05, 0.029])
    worst = 0.0
    for k, target in enumerate(names):
        got = float(inference.laplace_parameter_uncertainty(jnp.asarray(x0), names, data,
                                                            vm.BinaryModelCartesian, target))
        h = np.array([1e-3, 1e-3, 1e-5])[k]
        e = np.eye(3)[k] * h

        def half(x):
            return 0.5 * ours.chi2(d, binary_vis(*x))

        curv = (half(x0 + e) - 2 * half(x0) + half(x0 - e)) / h**2
        worst = max(worst, abs(got * np.sqrt(curv) - 1))
    record("max_rel_dsigma", worst)
    assert worst < 1e-5


# --------------------------------------------------------------- likelihood


@pytest.mark.validates("virgil.likelihood.joint_loglike", roots=["mathematics"])
def test_joint_loglike_is_the_sum_of_independent_gaussians(pair):
    obs, loaded = pair
    worst = 0.0
    for p in (PARAMS, {"dra": jnp.array(6.0), "ddec": jnp.array(-4.0), "flux": jnp.array(FLUXES)}):
        terms = [our_loglike(d, our_vis(p, i)) for i, d in enumerate(loaded)]
        for i, d in enumerate(loaded):  # our densities are -1/2 our chi-squared plus their normalisations
            norm = terms[i] + 0.5 * ours.chi2(d, our_vis(p, i))
            sv, sp = d["dv2"].ravel(), d["dcp"].ravel()
            want_norm = (np.sum(-np.log(sv) - 0.5 * np.log(2 * np.pi))
                         + np.sum(-np.log(2 * np.pi) - np.log(special.i0e(1 / sp**2))))
            assert abs(norm - want_norm) < 1e-8 * abs(want_norm)
        got = float(likelihood.joint_loglike(p, obs, model_fn))
        worst = max(worst, abs(got - sum(terms)))
    record("max_abs_dloglike", worst)
    assert worst < 1e-8


@pytest.mark.validates("virgil.likelihood.joint_loglike", roots=["mathematics"])
def test_joint_loglike_with_inflated_errors(pair):
    obs, loaded = pair
    got = float(likelihood.joint_loglike(PARAMS, obs, model_fn, vis_scale=1.7, phi_scale=1.3))
    want = sum(our_loglike(d, our_vis(PARAMS, i), 1.7, 1.3) for i, d in enumerate(loaded))
    record("abs_dloglike", abs(got - want))
    assert abs(got - want) < 1e-8


@pytest.mark.validates("virgil.likelihood.joint_loglike", roots=["mathematics"], kind="control")
def test_joint_loglike_keeps_each_dataset_its_own_parameters(pair):
    obs, loaded = pair
    got = float(likelihood.joint_loglike(PARAMS, obs, model_fn))
    swapped = dict(PARAMS, flux=PARAMS["flux"][::-1])
    wrong = sum(our_loglike(d, our_vis(swapped, i)) for i, d in enumerate(loaded))
    assert abs(got - wrong) > 1.0


@pytest.mark.validates("virgil.likelihood.joint_prediction", "virgil.likelihood.joint_data",
                       "virgil.likelihood.joint_errors", roots=["mathematics"])
def test_joint_vectors_match_our_files_and_observables(pair):
    obs, loaded = pair
    y, e = np.asarray(likelihood.joint_data(obs)), np.asarray(likelihood.joint_errors(obs))
    m = np.asarray(likelihood.joint_prediction(PARAMS, obs, model_fn))
    assert y.shape == e.shape == m.shape
    start, worst, chi2 = 0, 0.0, 0.0
    for i, d in enumerate(loaded):
        nv, nc = d["v2"].size, d["cp"].size
        sl_v, sl_c = slice(start, start + nv), slice(start + nv, start + nv + nc)
        start += nv + nc
        iv = order(y[sl_v], d["v2"].ravel())  # visibilities first, then phases, dataset by dataset
        ic = order(y[sl_c], d["cp"].ravel())
        assert np.array_equal(e[sl_v], d["dv2"].ravel()[iv]) and np.allclose(e[sl_c], d["dcp"].ravel()[ic], rtol=1e-12)
        v2, cp = our_observables(d, our_vis(PARAMS, i))
        worst = max(worst, np.max(np.abs(m[sl_v] - v2[iv])),
                    np.max(np.abs(np.angle(np.exp(1j * (m[sl_c] - cp[ic]))))))
        chi2 += np.sum(((m[sl_v] - y[sl_v]) / e[sl_v]) ** 2) + np.sum((2 * np.sin((m[sl_c] - y[sl_c]) / 2) / e[sl_c]) ** 2)
    assert start == y.size
    want = sum(ours.chi2(d, our_vis(PARAMS, i)) for i, d in enumerate(loaded))
    record("max_abs_dprediction", worst)
    record("rel_dchi2", abs(chi2 - want) / want)
    assert worst < 1e-10
    assert abs(chi2 - want) < 1e-10 * want


@pytest.mark.validates("virgil.likelihood.noise_sites", "virgil.likelihood.noise_for", roots=["mathematics"])
def test_noise_sites_and_terms_per_dataset(pair):
    obs, loaded = pair
    prior = dist.LogUniform(0.1, 10.0)
    shared = likelihood.noise_sites({"vis_scale": prior}, 2)
    assert set(shared) == {"noise.vis_scale"}
    pr, datasets, term = shared["noise.vis_scale"]
    assert pr is prior and tuple(datasets) == (0, 1) and term == "vis_scale"
    each = likelihood.noise_sites([{"vis_scale": prior}, {"vis_scale": prior, "phi_scale": prior}], 2)
    assert set(each) == {"noise[0].vis_scale", "noise[1].vis_scale", "noise[1].phi_scale"}
    assert tuple(each["noise[1].phi_scale"][1]) == (1,) and each["noise[1].phi_scale"][2] == "phi_scale"
    values = {"noise[0].vis_scale": 1.6, "noise[1].vis_scale": 0.8, "noise[1].phi_scale": 1.4}
    assert likelihood.noise_for(each, values, 0) == {"vis_scale": 1.6}
    assert likelihood.noise_for(each, values, 1) == {"vis_scale": 0.8, "phi_scale": 1.4}
    assert likelihood.noise_for(shared, {"noise.vis_scale": 2.0}, 1) == {"vis_scale": 2.0}
    worst = 0.0
    for i, d in enumerate(loaded):
        terms = likelihood.noise_for(each, values, i)
        got = float(likelihood.model_loglike(model_fn(PARAMS, i), obs[i], **terms))
        want = our_loglike(d, our_vis(PARAMS, i), terms.get("vis_scale", 1.0), terms.get("phi_scale", 1.0))
        worst = max(worst, abs(got - want))
    record("max_abs_dloglike", worst)
    assert worst < 1e-8


@pytest.mark.validates("virgil.likelihood.posterior_predictive_summary", roots=["mathematics"])
def test_posterior_predictive_summary_over_our_observables(pair):
    obs, loaded = pair
    data, d = obs[0], loaded[0]
    rng = np.random.default_rng(8)
    n = 25
    samples = {"dra": 6.0 + 0.1 * rng.normal(size=n), "ddec": -4.0 + 0.1 * rng.normal(size=n),
               "flux": 0.03 + 0.002 * rng.normal(size=n)}
    got = likelihood.posterior_predictive_summary({k: jnp.asarray(v) for k, v in samples.items()},
                                                  vm.BinaryModelCartesian, data)
    y = np.asarray(data.flatten_data()[0])
    nv = d["v2"].size
    iv, ic = order(y[:nv], d["v2"].ravel()), order(y[nv:], d["cp"].ravel())
    v2s, cps = zip(*(our_observables(d, binary_vis(samples["dra"][k], samples["ddec"][k], samples["flux"][k]))
                     for k in range(n)))
    v2s, cps = np.array(v2s)[:, iv], np.array(cps)[:, ic]  # closure phases far from +-pi: no wrapping
    worst = max(np.max(np.abs(np.asarray(got["vis_mean"]) - v2s.mean(0))),
                np.max(np.abs(np.asarray(got["vis_std"]) - v2s.std(0))),
                np.max(np.abs(np.asarray(got["phi_mean"]) - cps.mean(0))),
                np.max(np.abs(np.asarray(got["phi_std"]) - cps.std(0))))
    record("max_abs_dsummary", worst)
    assert worst < 1e-10


# ---------------------------------------------------------------- detection


@pytest.mark.validates("virgil.detection.rescale_errors", roots=["mathematics"])
def test_rescale_errors_sets_each_block_to_unit_reduced_chi2(pair):
    obs, loaded = pair
    data, d = obs[0], loaded[0]
    null = vm.BinaryModelCartesian(0.0, 0.0, 0.0)
    star = binary_vis(0.0, 0.0, 0.0)
    v2, cp = our_observables(d, star)
    chi_v = np.sum(((v2 - d["v2"].ravel()) / d["dv2"].ravel()) ** 2)
    chi_c = np.sum((2 * np.sin((cp - d["cp"].ravel()) / 2) / d["dcp"].ravel()) ** 2)
    s_v, s_c = np.sqrt(chi_v / v2.size), np.sqrt(chi_c / cp.size)
    new, factors = detection.rescale_errors(data, null)
    rel = max(abs(factors["vis"] / s_v - 1), abs(factors["phi"] / s_c - 1))
    record("max_rel_dfactor", rel)
    assert rel < 1e-10
    _, e_new = new.flatten_data()
    _, e_old = data.flatten_data()
    nv = v2.size
    assert np.allclose(np.asarray(e_new)[:nv], s_v * np.asarray(e_old)[:nv], rtol=1e-10)
    assert np.allclose(np.asarray(e_new)[nv:], s_c * np.asarray(e_old)[nv:], rtol=1e-10)
    scaled = dict(d, dv2=d["dv2"] * s_v, dcp=d["dcp"] * s_c)
    assert abs(ours.chi2(scaled, star) / (v2.size + cp.size) - 1) < 1e-10


@pytest.mark.validates("virgil.detection.injection_grid", roots=["mathematics", "statistics"])
def test_injection_grid_order_geometry_and_uniform_angles():
    seps, fluxes, n_pa = np.array([5.0, 12.0, 30.0]), np.array([0.0, 0.01]), 300
    g = detection.injection_grid(seps, fluxes, n_pa, 7)
    n = seps.size * fluxes.size * n_pa
    assert all(np.shape(g[k]) == (n,) for k in ("dra", "ddec", "flux"))
    sep = np.hypot(g["dra"], g["ddec"])
    assert np.allclose(sep, np.repeat(seps, fluxes.size * n_pa), rtol=1e-12)
    assert np.array_equal(g["flux"], np.tile(np.repeat(fluxes, n_pa), seps.size))
    # Only the angle distribution is tested: a uniform angle stays uniform under any rotation or
    # reflection, so this does not pin the PA convention (dra = sep sin PA is virgil's documented
    # choice; the convention is validated through the binary model tests, not here).
    pa = np.degrees(np.arctan2(g["dra"], g["ddec"])) % 360
    p = stats.kstest(pa / 360, "uniform").pvalue
    record("ks_pvalue", p)
    assert p > 1e-3
    again = detection.injection_grid(seps, fluxes, n_pa, 7)
    assert np.array_equal(again["dra"], g["dra"])
    assert not np.allclose(detection.injection_grid(seps, fluxes, n_pa, 8)["dra"], g["dra"])


def wrap(x):
    return np.angle(np.exp(1j * x))


@pytest.mark.validates("virgil.detection.bootstrap_null", roots=["mathematics", "statistics"])
def test_bootstrap_null_sign_flip_keeps_our_whitened_residuals(pair):
    obs, loaded = pair
    data, d = obs[0], loaded[0]
    null = vm.BinaryModelCartesian(6.0, -4.0, 0.025)
    y, e = (np.asarray(a) for a in data.flatten_data())
    nv = d["v2"].size
    iv, ic = order(y[:nv], d["v2"].ravel()), order(y[nv:], d["cp"].ravel())
    v2, cp = our_observables(d, binary_vis(6.0, -4.0, 0.025))
    r_v = (d["v2"].ravel() - v2)[iv] / d["dv2"].ravel()[iv]
    r_c = wrap(d["cp"].ravel() - cp)[ic] / d["dcp"].ravel()[ic]
    simulate = detection.bootstrap_null(data, null)
    signs, worst = [], 0.0
    for k in range(20):
        draw = simulate(jax.random.key(k))
        yd, ed = (np.asarray(a) for a in draw.flatten_data())
        assert np.array_equal(ed, e)
        s_v = (yd[:nv] - v2[iv]) / d["dv2"].ravel()[iv]
        s_c = wrap(yd[nv:] - cp[ic]) / d["dcp"].ravel()[ic]
        worst = max(worst, np.max(np.abs(np.abs(s_v) - np.abs(r_v))), np.max(np.abs(np.abs(s_c) - np.abs(r_c))))
        signs.append(np.sign(np.concatenate([s_v * r_v, s_c * r_c])))
    signs = np.concatenate(signs)
    record("max_abs_dwhitened", worst)
    assert worst < 1e-8
    flipped = np.mean(signs < 0)
    record("fraction_flipped", flipped)
    assert abs(flipped - 0.5) < 4 * 0.5 / np.sqrt(signs.size)
    # about another scene: its prediction plus the same magnitudes
    other = vm.BinaryModelCartesian(10.0, 3.0, 0.05)
    v2o, cpo = our_observables(d, binary_vis(10.0, 3.0, 0.05))
    yo = np.asarray(simulate(jax.random.key(99), other).flatten_data()[0])
    assert np.allclose(np.abs(yo[:nv] - v2o[iv]) / d["dv2"].ravel()[iv], np.abs(r_v), atol=1e-8)
    assert np.allclose(np.abs(wrap(yo[nv:] - cpo[ic])) / d["dcp"].ravel()[ic], np.abs(r_c), atol=1e-8)


@pytest.mark.validates("virgil.detection.bootstrap_null", roots=["mathematics"])
def test_bootstrap_null_resample_draws_within_each_block(pair):
    obs, loaded = pair
    data, d = obs[0], loaded[0]
    null = vm.BinaryModelCartesian(6.0, -4.0, 0.025)
    y = np.asarray(data.flatten_data()[0])
    nv = d["v2"].size
    iv, ic = order(y[:nv], d["v2"].ravel()), order(y[nv:], d["cp"].ravel())
    v2, cp = our_observables(d, binary_vis(6.0, -4.0, 0.025))
    r_v = (d["v2"].ravel() - v2)[iv] / d["dv2"].ravel()[iv]
    r_c = wrap(d["cp"].ravel() - cp)[ic] / d["dcp"].ravel()[ic]
    simulate = detection.bootstrap_null(data, null, method="resample")
    moved, repeated = 0, 0
    for k in range(5):
        yd = np.asarray(simulate(jax.random.key(k)).flatten_data()[0])
        s_v = (yd[:nv] - v2[iv]) / d["dv2"].ravel()[iv]
        s_c = wrap(yd[nv:] - cp[ic]) / d["dcp"].ravel()[ic]
        assert all(np.min(np.abs(r_v - x)) < 1e-8 for x in s_v)  # visibilities from visibilities
        assert all(np.min(np.abs(r_c - x)) < 1e-8 for x in s_c)  # phases from phases
        moved += not (np.allclose(s_v, r_v, atol=1e-8) and np.allclose(s_c, r_c, atol=1e-8))
        repeated += len(np.unique(np.round(s_v, 8))) < nv  # with replacement: a value recurs
    record("draws_changed_of_5", moved)
    assert moved == 5  # a method that returned the data unchanged would give 0
    assert repeated >= 1  # a pure permutation (no replacement) would give 0
