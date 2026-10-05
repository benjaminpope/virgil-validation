"""Light checks of small helpers: their stated behaviour, against our own
chi-squared (crosscheck.chi2, a three-telescope file read with astropy).

* best_grid_point: numpy's nanargmax over the grid.
* build_model and loglike: a class and a template build the same scene, and
  differences of loglike are -1/2 the differences of our chi-squared.
* inflated_errors: hypot(scale sigma, absolute, relative |V|) for the
  visibility observables and hypot(scale sigma, phi_error) for the phases,
  relative to the model or the data; or the largest of them ("max").
* fisher: the Hessian of 1/2 our chi-squared by central differences.
* simulate: noiseless, the template with our own V^2 and closure phases;
  noisy, V^2 residuals over the errors are N(0, noise_scale^2).
* bias_test: each entry is the fit to the simulation drawn with that key.
"""

import jax
import numpy as np
import pytest
from scipy import stats

from crosscheck import chi2 as ours, simulate as sim, sky
from evidence.plugin import record

vm = pytest.importorskip("virgil.models")
from virgil import fitting, grid_fit, inference, likelihood, simulate  # noqa: E402
from virgil.oidata import OIData  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

pytestmark = pytest.mark.x64

UTS3 = np.array([[-9.925, -20.335], [14.887, 30.502], [103.306, 43.999]])
TRUTH = (6.0, -4.0, 0.03)


def binary_vis(x, y, f):
    def vis(u, v, w):
        return (sky.vis_point(u, v, w) + f * sky.vis_point(u, v, w, x, y)) / (1 + f)

    return vis


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("helpers") / "d.fits"
    sim.observe(path, binary_vis(*TRUTH), UTS3, hour_angles_h=np.linspace(-3, 3, 7),
                wavelengths=np.linspace(1.5e-6, 2.4e-6, 6), dec_deg=-50.0, sigma_v2=0.01, sigma_cp_deg=0.5,
                rng=np.random.default_rng(11))
    return OIData(str(path)), ours.load(path)


PARAMS = ["dra", "ddec", "flux"]


@pytest.mark.validates("virgil.grid_fit.best_grid_point", roots=["mathematics"])
def test_best_grid_point_is_the_nanargmax():
    rng = np.random.default_rng(0)
    axes = {"a": np.linspace(0, 1, 5), "b": np.linspace(-2, 2, 7), "c": np.logspace(-3, 0, 4)}
    grid = rng.normal(size=(5, 7, 4))
    grid[1, 2, 3] = np.nan
    i, j, k = np.unravel_index(np.nanargmax(grid), grid.shape)
    best = grid_fit.best_grid_point(grid, axes)
    assert list(best) == ["a", "b", "c"]
    assert (float(best["a"]), float(best["b"]), float(best["c"])) == (axes["a"][i], axes["b"][j], axes["c"][k])


@pytest.mark.validates("virgil.likelihood.loglike", "virgil.likelihood.build_model", roots=["mathematics"])
def test_build_model_and_loglike_against_our_chi2(dataset):
    data, d = dataset
    a, b = np.array([5.0, -3.0, 0.02]), np.array(TRUTH)
    template = vm.BinaryModelCartesian(0.0, 0.0, 0.01)
    for v in (a, b):
        x = likelihood.build_model(vm.BinaryModelCartesian, PARAMS, v)
        y = likelihood.build_model(template, PARAMS, v)
        assert all(float(getattr(x, p)) == float(getattr(y, p)) == v[n] for n, p in enumerate(PARAMS))
    for model in (vm.BinaryModelCartesian, template):
        got = float(likelihood.loglike(a, PARAMS, data, model) - likelihood.loglike(b, PARAMS, data, model))
        want = -0.5 * (ours.chi2(d, binary_vis(*a)) - ours.chi2(d, binary_vis(*b)))
        record("abs_dloglike", abs(got - want))
        assert abs(got - want) < 1e-8 * max(1.0, abs(want))


@pytest.mark.parametrize("where,combine", [("model", "quadrature"), ("data", "quadrature"), ("data", "max")])
@pytest.mark.validates("virgil.likelihood.inflated_errors", roots=["mathematics"])
def test_inflated_errors(dataset, where, combine):
    data, _ = dataset
    pred = np.asarray(data.model(vm.BinaryModelCartesian(*TRUTH)))
    obs, sig = (np.asarray(x) for x in data.flatten_data())
    n_vis = np.asarray(data.vis).size
    ref = (pred if where == "model" else obs)[:n_vis]
    got = np.asarray(likelihood.inflated_errors(data, pred, vis_error_rel=0.05, phi_error=0.01, vis_scale=1.3,
                                                phi_scale=0.8, vis_error=0.004, where=where, combine=combine))
    op = (lambda *t: np.sqrt(sum(x**2 for x in t))) if combine == "quadrature" else (lambda *t: np.max(t, axis=0))
    want_vis = op(1.3 * sig[:n_vis], np.full(n_vis, 0.004), 0.05 * np.abs(ref))
    want_phi = op(0.8 * sig[n_vis:], np.full(sig.size - n_vis, 0.01))
    np.testing.assert_allclose(got, np.concatenate([want_vis, want_phi]), rtol=1e-14)
    np.testing.assert_array_equal(np.asarray(likelihood.inflated_errors(data, pred)), sig)


@pytest.mark.validates("virgil.inference.fisher", roots=["mathematics"])
def test_fisher_is_the_hessian_of_half_our_chi2(dataset):
    data, d = dataset
    x0 = np.array([5.8, -4.1, 0.028])
    got = np.asarray(inference.fisher(x0, PARAMS, data, vm.BinaryModelCartesian))
    h = np.array([1e-3, 1e-3, 1e-5])

    def half(x):
        return 0.5 * ours.chi2(d, binary_vis(*x))

    want = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            ei, ej = np.eye(3)[i] * h[i], np.eye(3)[j] * h[j]
            want[i, j] = (half(x0 + ei + ej) - half(x0 + ei - ej) - half(x0 - ei + ej) + half(x0 - ei - ej)) / (4 * h[i] * h[j])
    rel = np.max(np.abs(got - want) / np.sqrt(np.outer(np.diag(want), np.diag(want))))
    record("max_rel_fisher", rel)
    assert rel < 1e-5


@pytest.mark.validates("virgil.simulate.simulate", roots=["mathematics", "statistics"])
def test_simulate_noiseless_and_noisy(dataset):
    data, d = dataset
    scene = vm.BinaryModelCartesian(4.0, 3.0, 0.05)
    clean = simulate.simulate(scene, data)
    vis = binary_vis(4.0, 3.0, 0.05)
    np.testing.assert_allclose(np.asarray(clean.vis).ravel(), np.abs(vis(d["u"], d["v"], d["wl"])).ravel() ** 2,
                               atol=1e-12)
    np.testing.assert_array_equal(np.asarray(clean.d_vis), np.asarray(data.d_vis))
    for scale in (1.0, 2.0):
        pulls = np.concatenate([
            ((np.asarray(simulate.simulate(scene, data, key=jax.random.key(k), noise_scale=scale).vis)
              - np.asarray(clean.vis)) / np.asarray(data.d_vis)).ravel() for k in range(30)])
        assert abs(np.std(pulls) / scale - 1) < 4 / np.sqrt(2 * pulls.size)
        assert stats.kstest(pulls / scale, "norm").pvalue > 1e-3


@pytest.mark.validates("virgil.simulate.bias_test", roots=["self-consistency"])
def test_bias_test_bookkeeping(dataset):
    data, _ = dataset
    scene = vm.BinaryModelCartesian(*TRUTH)
    priors = {"dra": dist.Uniform(0.0, 10.0), "ddec": dist.Uniform(-8.0, 0.0), "flux": dist.LogUniform(1e-3, 0.1)}
    key = jax.random.key(7)
    out = simulate.bias_test(scene, data, scene, priors, 2, key)
    assert set(out) == set(PARAMS) | {"chi2_red"}
    for n, draw in enumerate(jax.random.split(key, 2)):
        one = fitting.fit(scene, priors, simulate.simulate(scene, data, key=draw))
        for p in PARAMS:
            assert float(out[p][n]) == pytest.approx(float(one.values[p]), rel=1e-10, abs=1e-12)
        assert float(out["chi2_red"][n]) == pytest.approx(float(one.info["chi2_red"]), rel=1e-10)
