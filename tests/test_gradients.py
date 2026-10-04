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


M = "virgil.models."

# (objects credited, factory, paths). Factories run inside each test, so
# the templates are built in float64 (tests/conftest.py's x64 context).
CASES = {
    "PointSource": (
        [M + "PointSource"],
        lambda: vm.PointSource(dra=3.0, ddec=-2.0),
        ["dra", "ddec"],
    ),
    "GaussianDisk": (
        [M + "GaussianDisk"],
        lambda: vm.GaussianDisk(1.3, dra=0.4, ddec=-0.7),
        ["sigma", "dra", "ddec"],
    ),
    "EllipticalGaussian": (
        [M + "EllipticalGaussian"],
        lambda: vm.EllipticalGaussian(3.0, 0.4, 35.0, dra=1.0, ddec=2.0),
        ["fwhm", "ratio", "pa", "dra", "ddec"],
    ),
    "UniformDisk": (
        [M + "UniformDisk"],
        lambda: vm.UniformDisk(3.0, dra=0.4, ddec=-0.7),
        ["diam", "dra", "ddec"],
    ),
    "ModulatedGaussianRim": (
        [M + "ModulatedGaussianRim"],
        lambda: vm.ModulatedGaussianRim(
            4.0, 0.8, 50.0, 30.0, np.array([0.4, 0.25]),
            np.array([40.0, 110.0]), dra=0.3, ddec=-0.2,
        ),
        ["diam", "fwhm", "inc", "pa", "az_amps", "az_pas", "dra", "ddec"],
    ),
    "GaussianArc": (
        [M + "GaussianArc"],
        lambda: vm.GaussianArc(5.0, 0.6, 4.0, 250.0, dra=0.5, ddec=0.3),
        ["radius", "width", "length", "pa", "dra", "ddec"],
    ),
    "BinaryModelCartesian": (
        [M + "BinaryModelCartesian"],
        lambda: vm.BinaryModelCartesian(4.97, -3.36, 0.05),
        ["dra", "ddec", "flux"],
    ),
    "BinaryModelAngular": (
        [M + "BinaryModelAngular"],
        lambda: vm.BinaryModelAngular(6.0, 124.0, 0.05),
        ["sep", "pa", "flux"],
    ),
    "System": (
        [M + "System", M + "Resolved", M + "UniformDisk", M + "PointSource",
         M + "GaussianDisk"],
        lambda: vm.System(
            star=vm.UniformDisk(1.8),
            comp=vm.PointSource(0.02, -12.0, 8.0),
            env=vm.GaussianDisk(2.0, flux=0.3),
            halo=vm.Resolved(0.1),
        ),
        ["star.diam", "comp.flux", "comp.dra", "comp.ddec", "env.flux",
         "env.sigma", "halo.flux"],
    ),
    "Rotated": (
        [M + "Rotated"],
        lambda: vm.Rotated(
            vm.System(
                a=vm.GaussianDisk(1.0, ddec=3.0), b=vm.PointSource(0.5, dra=1.0)
            ),
            70.0,
        ),
        ["rotation_deg"],
    ),
    "GravityDarkenedStar": (
        [M + "GravityDarkenedStar"],
        lambda: vm.GravityDarkenedStar(1.0, omega=0.7, inc=50.0, pa=30.0, n_lat=16),
        ["diam_eq", "omega", "inc", "pa"],
    ),
}

_DISK = dict(radius=40.0, fwhm=20.0, inc=50.0, pa=30.0, skew=1.5, aspect=0.1,
             flaring=1.25, symmetric=0.3, npix=24, pixel_scale_mas=6.0, dra=1.0, ddec=-0.5)
_DISK_PATHS = ["radius", "fwhm", "inc", "pa", "skew", "aspect", "flaring", "symmetric", "dra", "ddec"]
for _name, _key, _value in [("FlaredDiskHG", "g", 0.4), ("FlaredDiskGaussian", "sigma_theta", 40.0),
                            ("FlaredDiskPowerLaw", "n", 3.0)]:
    CASES[_name] = (
        [M + _name],
        lambda _name=_name, _key=_key, _value=_value: getattr(vm, _name)(**{_key: _value}, **_DISK),
        [_key, *_DISK_PATHS],
    )


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name, marks=pytest.mark.validates(*objs, roots=["mathematics"])
        )
        for name, (objs, _, _) in CASES.items()
    ],
)
def test_model_visibility_gradients(name):
    """HarmonixModel is not covered: its derivatives are harmonix's."""
    _, factory, paths = CASES[name]
    template = factory()
    assert jnp.asarray(template.get(paths[0])).dtype == jnp.float64
    _check(template, paths)


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


@pytest.mark.parametrize(
    "which",
    [
        pytest.param(
            w, marks=pytest.mark.validates(f"virgil.likelihood.{w}", roots=["mathematics"])
        )
        for w in ("whitened_residuals", "model_loglike")
    ],
)
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
