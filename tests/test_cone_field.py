"""TruncatedCone, GaussianField and the invariants every SourceModel shares.

* TruncatedCone: a direct Fourier sum over a 3-D cloud of points on the
  cone's walls, built from its documented geometry (axis towards ``pa``,
  tilted ``tilt`` out of the sky, apex ``tip`` behind the reference point,
  slant distances s from ``s0`` to ``s0 + 5 length`` weighted by
  rho exp(-(s - s0)/length) and uniformly round each ring, cross-section
  stretched by ``ratio`` in the plane of the axis and the line of sight),
  projected onto the sky and blurred by a Gaussian of FWHM ``width``. The
  walls are integrated by Gauss-Legendre in s and the trapezoid rule in
  azimuth, not virgil's midpoint rings and J0. The documented second-order
  convergence in n_rings is checked against that reference.
* GaussianField: the field's covariance (from its linear response to each
  latent) against our own dense construction, Pi (kappa^2 + L)^(-order) Pi
  with L the reflecting-boundary (Neumann) Laplacian built as a matrix of
  second differences and Pi the projection off the constant mode, scaled to
  a pixel-averaged variance sigma^2; the anisotropic form; the template;
  and an Image of it as the softmax of the field.
* SourceModel invariants, for each analytic component: V(0) = 1, Hermitian
  symmetry V(-u, -v) = conj V(u, v) for a real brightness, and the shift
  theorem for dra, ddec.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import fft as sfft
from scipy.special import roots_legendre

from crosscheck import sky
from evidence.plugin import record

vm = pytest.importorskip("virgil.models")
vf = pytest.importorskip("virgil.fields")

pytestmark = pytest.mark.x64

WL = 2.2e-6


def baselines(n=40, b_max=60.0, seed=0):
    rng = np.random.default_rng(seed)
    r = b_max * np.sqrt(rng.uniform(0.01, 1.0, n))
    t = rng.uniform(0, 2 * np.pi, n)
    return r * np.sin(t), r * np.cos(t)


def cone_cloud(tip, alpha, s0, length, tilt, pa, ratio, n_s=240, n_phi=256):
    """Points on the cone's walls, in East and North mas, with weights."""
    a, b, p = np.deg2rad(alpha), np.deg2rad(tilt), np.deg2rad(pa)
    axis = np.array([np.sin(p) * np.cos(b), np.cos(p) * np.cos(b), np.sin(b)])  # East, North, towards us
    across = np.array([np.cos(p), -np.sin(p), 0.0])
    in_plane = np.cross(across, axis)  # perpendicular to the axis, in its plane with the line of sight
    x, w = roots_legendre(n_s)
    s = s0 + 2.5 * length * (x + 1)
    w_s = w * 2.5 * length
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    rho = s * np.sin(a)
    S, PHI = np.meshgrid(s, phi, indexing="ij")
    R = S * np.sin(a)
    centre = (S * np.cos(a) - tip)[..., None] * axis
    pts = centre + R[..., None] * (np.cos(PHI)[..., None] * across + ratio * np.sin(PHI)[..., None] * in_plane)
    weight = (w_s * rho * np.exp(-(s - s0) / length))[:, None] * np.ones(n_phi)
    weight = weight / weight.sum()
    return sky.Cloud(pts[..., 0].ravel(), pts[..., 1].ravel(), weight.ravel())


CONES = [
    dict(tip=5.0, alpha=30.0, s0=4.0, length=10.0, width=1.0, tilt=20.0, pa=90.0, ratio=1.0),
    dict(tip=2.0, alpha=62.5, s0=1.0, length=6.0, width=0.7, tilt=-35.0, pa=200.0, ratio=1.4),
    dict(tip=3.0, alpha=45.0, s0=2.0, length=5.0, width=1.5, tilt=90.0, pa=10.0, ratio=0.8),
    dict(tip=0.0, alpha=15.0, s0=0.5, length=8.0, width=0.5, tilt=0.0, pa=300.0, ratio=1.0),
]


def cone_reference(u, v, c, dra=0.0, ddec=0.0):
    geom = {k: c[k] for k in ("tip", "alpha", "s0", "length", "tilt", "pa", "ratio")}
    cloud = cone_cloud(**geom).shifted(dra, ddec)
    return sky.visibility(cloud, u, v, WL) * sky.gaussian_blur_factor(u, v, WL, c["width"])


@pytest.mark.parametrize("c", CONES, ids=["side", "wide-negative-tilt", "down-axis", "in-sky"])
@pytest.mark.validates("virgil.models.TruncatedCone", roots=["mathematics"])
def test_truncated_cone_against_a_3d_cloud(c):
    u, v = baselines()
    want = cone_reference(u, v, c, dra=1.5, ddec=-2.0)
    errs = {}
    for n in (64, 128, 1024):
        got = np.asarray(vm.TruncatedCone(**c, dra=1.5, ddec=-2.0, n_rings=n).model(jnp.asarray(u), jnp.asarray(v), WL))
        errs[n] = np.max(np.abs(got - want))
    record("max_abs_dV_1024", errs[1024])
    assert errs[1024] < 2e-5
    if errs[64] > 1e-6:  # resolved fringes: second order, so doubling cuts the error by about 4
        record("convergence_ratio", errs[64] / errs[128])
        assert 2.5 < errs[64] / errs[128] < 6.0


@pytest.mark.validates("virgil.models.TruncatedCone", roots=["mathematics"])
def test_truncated_cone_tilt_sign_does_not_change_the_image():
    u, v = baselines()
    c = dict(CONES[1])
    a = np.asarray(vm.TruncatedCone(**c).model(jnp.asarray(u), jnp.asarray(v), WL))
    c["tilt"] = -c["tilt"]
    b = np.asarray(vm.TruncatedCone(**c).model(jnp.asarray(u), jnp.asarray(v), WL))
    np.testing.assert_allclose(a, b, atol=1e-14)


# ----------------------------------------------------------------- fields


def neumann_laplacian(n, h):
    """Second differences with reflecting ends: -(f[i+1] - 2 f[i] + f[i-1]) / h^2."""
    L = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    L[0, 0] = L[-1, -1] = 1.0
    return L / h**2


def our_covariance(shape, h, sigma, length, order):
    n, m = shape
    Ln, Lm = neumann_laplacian(n, h), neumann_laplacian(m, h)
    if np.ndim(length) == 0:
        A = np.kron(Ln, np.eye(m)) + np.kron(np.eye(n), Lm) + np.eye(n * m) / length**2
    else:
        A = np.eye(n * m) + length[0] ** 2 * np.kron(Ln, np.eye(m)) + length[1] ** 2 * np.kron(np.eye(n), Lm)
    lam, vec = np.linalg.eigh(A)
    C = vec @ np.diag(lam ** (-float(order))) @ vec.T
    P = np.eye(n * m) - np.ones((n * m, n * m)) / (n * m)
    C = P @ C @ P
    return C * sigma**2 * (n * m) / np.trace(C)


def virgil_covariance(shape, h, sigma, length, order):
    def eta(z):
        return vf.GaussianField(z.reshape(shape), sigma=sigma, length_mas=length, order=order).evaluate(h).ravel()

    A = np.asarray(jax.jacfwd(eta)(jnp.zeros(shape[0] * shape[1])))
    return A @ A.T


@pytest.mark.parametrize("length,order", [(2.0, 1), (2.0, 2), (0.7, 3), ((3.0, 0.8), 2)])
@pytest.mark.validates("virgil.fields.GaussianField", "virgil.fields.field_spectrum", roots=["mathematics"])
def test_gaussian_field_covariance(length, order):
    shape, h, sigma = (7, 5), 0.6, 1.7
    got = virgil_covariance(shape, h, sigma, length, order)
    want = our_covariance(shape, h, sigma, length, order)
    rel = np.max(np.abs(got - want)) / np.max(np.abs(want))
    record("max_rel_cov", rel)
    assert rel < 1e-10
    assert np.trace(got) / got.shape[0] == pytest.approx(sigma**2, rel=1e-12)


@pytest.mark.validates("virgil.fields.GaussianField", roots=["mathematics"])
def test_order_one_prior_is_squared_variation_plus_l2():
    """With order 1, the whitened prior 1/2 |z|^2 is c (|eta|^2 / l^2 +
    sum of squared neighbour differences / h^2) for a zero-mean field."""
    shape, h, sigma, ell = (6, 8), 0.5, 1.3, 1.5
    rng = np.random.default_rng(2)
    spectrum = np.asarray(vf.field_spectrum(shape, h, sigma, ell, order=1))
    ratios = []
    for _ in range(5):
        z = rng.normal(size=shape)
        z[0, 0] = 0.0  # the constant mode does not reach the field
        eta = np.asarray(vf.GaussianField(z, sigma=sigma, length_mas=ell, order=1).evaluate(h))
        penalty = (np.sum(eta**2) / ell**2 + (np.sum(np.diff(eta, axis=0) ** 2) + np.sum(np.diff(eta, axis=1) ** 2)) / h**2)
        ratios.append(0.5 * np.sum(z**2) / penalty)
    assert np.ptp(ratios) < 1e-10 * np.mean(ratios)
    assert spectrum[0, 0] == 0.0


@pytest.mark.validates("virgil.fields.GaussianField", "virgil.models.Image", roots=["mathematics"])
def test_gaussian_field_template_and_image():
    shape, h = (9, 9), 0.4
    rng = np.random.default_rng(3)
    z = rng.normal(size=shape)
    yy, xx = np.mgrid[:9, :9] - 4.0
    template = np.exp(-(xx**2 + yy**2) / 6.0)
    field = vf.GaussianField(z, sigma=0.8, length_mas=1.2, mean=template, mean_floor=1e-2)
    eta = np.asarray(field.evaluate(h))
    S = np.asarray(vf.field_spectrum(shape, h, 0.8, 1.2))
    want = sfft.idctn(np.sqrt(S) * z, type=2, norm="ortho") + np.log(template / template.max() + 1e-2)
    np.testing.assert_allclose(eta, want, atol=1e-12)
    image = vm.Image(field, pixel_scale_mas=h)
    soft = np.exp(eta - eta.max())
    np.testing.assert_allclose(np.asarray(image.brightness), soft / soft.sum(), rtol=1e-12)


# --------------------------------------------------------- source models

COMPONENTS = {
    "PointSource": lambda: vm.PointSource(),
    "GaussianDisk": lambda: vm.GaussianDisk(sigma=1.3),
    "EllipticalGaussian": lambda: vm.EllipticalGaussian(2.0, 0.6, 30.0),
    "UniformDisk": lambda: vm.UniformDisk(1.5),
    "TruncatedCone": lambda: vm.TruncatedCone(**CONES[0]),
    "GaussianArc": lambda: vm.GaussianArc(3.0, 1.0, 90.0, 40.0),
}


@pytest.mark.parametrize("name", list(COMPONENTS))
@pytest.mark.validates("virgil.models.SourceModel", "virgil.models.System", roots=["mathematics"])
def test_source_model_invariants(name):
    comp = COMPONENTS[name]()
    u, v = baselines(seed=5)
    U, V = jnp.asarray(u), jnp.asarray(v)
    assert abs(complex(comp.model(jnp.zeros(1), jnp.zeros(1), WL)[0]) - 1) < 1e-12
    pos = np.asarray(comp.model(U, V, WL))
    neg = np.asarray(comp.model(-U, -V, WL))
    np.testing.assert_allclose(neg, np.conj(pos), atol=1e-12)
    shifted = np.asarray(comp.set(["dra", "ddec"], [2.5, -1.0]).model(U, V, WL))
    np.testing.assert_allclose(shifted, pos * sky.shift_phase(u, v, WL, 2.5, -1.0), atol=1e-12)
    scene = vm.System(a=comp, b=vm.PointSource(0.3, 4.0, 1.0))
    np.testing.assert_allclose(np.asarray(scene.model(U, V, WL)),
                               (pos + 0.3 * sky.vis_point(u, v, WL, 4.0, 1.0)) / 1.3, atol=1e-12)
