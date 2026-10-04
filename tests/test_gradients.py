"""Stage 2: virgil's derivatives against finite differences.

Every fit, Laplace covariance and posterior virgil computes depends on
JAX's derivatives of its models and likelihood. virgil's own tests only
check that they are finite. Here `jax.test_util.check_grads` compares
forward- and reverse-mode derivatives with central finite differences,
in float64, at generic points away from any singularity.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from evidence.plugin import record

vm = pytest.importorskip("virgil.models")
from virgil.coverage import vlti_oidata  # noqa: E402
from virgil.likelihood import model_loglike, whitened_residuals  # noqa: E402

pytestmark = pytest.mark.x64

RNG = np.random.default_rng(17)
U, V = RNG.uniform(-100, 100, (2, 24))
WL = 2.0e-6


def _visibility_function(template, paths):
    """x -> real and imaginary parts of the visibilities of the template
    with the leaves at ``paths`` set to ``x`` (flattened in order)."""
    sizes = [int(np.size(template.get(p))) for p in paths]
    shapes = [np.shape(template.get(p)) for p in paths]

    def f(x):
        values, start = [], 0
        for size, shape in zip(sizes, shapes):
            values.append(jnp.reshape(x[start : start + size], shape))
            start += size
        model = template.set(paths, values)
        vis = model.model(jnp.asarray(U), jnp.asarray(V), WL)
        return jnp.concatenate([jnp.real(vis), jnp.imag(vis)])

    x0 = jnp.concatenate(
        [jnp.ravel(jnp.asarray(template.get(p), float)) for p in paths]
    )
    return f, x0


def _check(template, paths, **tol):
    f, x0 = _visibility_function(template, paths)
    check_grads(f, (x0,), order=1, modes=("fwd", "rev"), **tol)


CASES = {
    "PointSource": (vm.PointSource(dra=3.0, ddec=-2.0), ["dra", "ddec"]),
    "GaussianDisk": (
        vm.GaussianDisk(1.3, dra=0.4, ddec=-0.7),
        ["sigma", "dra", "ddec"],
    ),
    "EllipticalGaussian": (
        vm.EllipticalGaussian(3.0, 0.4, 35.0, dra=1.0, ddec=2.0),
        ["fwhm", "ratio", "pa", "dra", "ddec"],
    ),
    "UniformDisk": (
        vm.UniformDisk(3.0, dra=0.4, ddec=-0.7),
        ["diam", "dra", "ddec"],
    ),
    "ModulatedGaussianRim": (
        vm.ModulatedGaussianRim(
            4.0, 0.8, 50.0, 30.0, np.array([0.4, 0.25]), np.array([40.0, 110.0]),
            dra=0.3, ddec=-0.2,
        ),
        ["diam", "fwhm", "inc", "pa", "az_amps", "az_pas", "dra", "ddec"],
    ),
    "GaussianArc": (
        vm.GaussianArc(5.0, 0.6, 4.0, 250.0, dra=0.5, ddec=0.3),
        ["radius", "width", "length", "pa", "dra", "ddec"],
    ),
    "BinaryModelCartesian": (
        vm.BinaryModelCartesian(4.97, -3.36, 0.05),
        ["dra", "ddec", "flux"],
    ),
    "BinaryModelAngular": (
        vm.BinaryModelAngular(6.0, 124.0, 0.05),
        ["sep", "pa", "flux"],
    ),
    "System": (
        vm.System(
            star=vm.UniformDisk(1.8),
            comp=vm.PointSource(0.02, -12.0, 8.0),
            env=vm.GaussianDisk(2.0, flux=0.3),
            halo=vm.Resolved(0.1),
        ),
        ["star.diam", "comp.flux", "comp.dra", "comp.ddec", "env.flux",
         "env.sigma", "halo.flux"],
    ),
    "Rotated": (
        vm.Rotated(
            vm.System(a=vm.GaussianDisk(1.0, ddec=3.0), b=vm.PointSource(0.5, dra=1.0)),
            70.0,
        ),
        ["rotation_deg"],
    ),
    "GravityDarkenedStar": (
        vm.GravityDarkenedStar(1.0, omega=0.7, inc=50.0, pa=30.0, n_lat=16),
        ["diam_eq", "omega", "inc", "pa"],
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_model_visibility_gradients(name):
    template, paths = CASES[name]
    _check(template, paths)


# tag each case with its object (markers cannot depend on parameters)
test_model_visibility_gradients = pytest.mark.validates(
    *[f"virgil.models.{n}" for n in CASES], roots=["mathematics"]
)(test_model_visibility_gradients)


@pytest.mark.validates("virgil.models.Image", roots=["mathematics"])
def test_image_pixel_gradients():
    """Derivatives with respect to the log-brightness of every pixel (the
    softmax normalisation couples them all)."""
    rng = np.random.default_rng(4)
    log_b = rng.normal(size=(5, 6))

    def f(x):
        vis = vm.Image(x, 0.6, dra=0.2).model(jnp.asarray(U), jnp.asarray(V), WL)
        return jnp.concatenate([jnp.real(vis), jnp.imag(vis)])

    check_grads(f, (jnp.asarray(log_b),), order=1, modes=("fwd", "rev"))


def _noisy_data(model):
    data = vlti_oidata(
        hour_angles_h=(-2.0, 0.0, 2.0),
        wavelengths_m=np.linspace(1.6e-6, 2.3e-6, 3),
        sigma_v2=0.02,
        sigma_cp_deg=1.0,
    )
    return data.with_model(model, key=jax.random.PRNGKey(2))


@pytest.mark.validates(
    "virgil.likelihood.whitened_residuals", "virgil.likelihood.model_loglike",
    roots=["mathematics"],
)
@pytest.mark.parametrize("which", ["whitened_residuals", "model_loglike"])
def test_likelihood_gradients(which):
    """The residual vector every fit uses, and the log-likelihood, with
    respect to model parameters, on noisy data with closure phases (so the
    correlated closure-phase whitening is differentiated too)."""
    template = vm.System(
        star=vm.UniformDisk(1.8), comp=vm.PointSource(0.05, -6.0, 4.0)
    )
    data = _noisy_data(template.set("comp.flux", 0.03))
    paths = ["star.diam", "comp.flux", "comp.dra", "comp.ddec"]

    def f(x):
        model = template.set(paths, list(x))
        if which == "whitened_residuals":
            return whitened_residuals(model, data)
        return model_loglike(model, data)

    x0 = jnp.array([1.8, 0.05, -6.0, 4.0])
    check_grads(f, (x0,), order=1, modes=("fwd", "rev"))


@pytest.mark.validates(
    "virgil.likelihood.model_loglike", "virgil.inference.laplace_cov",
    roots=["mathematics"],
)
def test_hessian_of_the_likelihood():
    """Laplace covariances and Fisher matrices use second derivatives.
    jax.hessian of the log-likelihood against a Richardson-extrapolated
    central difference of its gradient (error O(h^4)). The plain
    order-2 check_grads is limited by finite-difference error at ~1e-5
    relative for a log-likelihood of this size."""
    template = vm.BinaryModelCartesian(4.97, -3.36, 0.05)
    data = _noisy_data(template.set("flux", 0.04))
    paths = ["dra", "ddec", "flux"]

    def loglike(x):
        return model_loglike(template.set(paths, list(x)), data)

    x0 = jnp.array([4.9, -3.3, 0.045])
    hess = np.asarray(jax.hessian(loglike)(x0))
    grad = jax.jit(jax.grad(loglike))
    steps = np.array([1e-3, 1e-3, 1e-5])  # mas, mas, flux ratio

    def central(h):
        cols = []
        for k in range(3):
            e = np.zeros(3)
            e[k] = h[k]
            cols.append((np.asarray(grad(x0 + e)) - np.asarray(grad(x0 - e))) / (2 * h[k]))
        return np.stack(cols, axis=1)

    richardson = (4 * central(steps / 2) - central(steps)) / 3
    scale = np.sqrt(np.outer(np.abs(np.diag(hess)), np.abs(np.diag(hess))))
    err = np.max(np.abs(hess - richardson) / scale)
    np.testing.assert_allclose(hess, hess.T, rtol=1e-10)
    record("max_scaled_hessian_error", err)
    assert err < 1e-7, err
