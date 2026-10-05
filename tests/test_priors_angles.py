"""virgil.priors, virgil.angles and fit's flat coordinates.

* IsotropicInclination (degrees, density proportional to sin i) and
  IsotropicLatitude (radians, proportional to cos lat), on full, narrow
  and polar ranges: log density, CDF and mean against closed forms and
  SciPy quadrature; the quantile function inverts the CDF; samples pass a
  Kolmogorov-Smirnov test against the CDF.
* AngleVector: the density of v = r (cos t, sin t) is normalised over the
  plane, its angle is exactly uniform (or von Mises, or axial von Mises)
  whatever the ring width, half the squared residuals is -log p up to a
  constant, and samples of the angle and radius follow their densities.
* fit: a parameter with a flat-coordinate prior (LogUniform, an isotropic
  inclination) is optimised where its prior is constant, so the MAP is the
  maximum of the likelihood ("Priors and the MAP" in virgil's
  conventions); a Gaussian prior, which has no flat coordinate, is not.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from scipy import integrate, optimize, stats

from crosscheck import simulate, sky
from evidence.plugin import record

pr = pytest.importorskip("virgil.priors")
ang = pytest.importorskip("virgil.angles")
vm = pytest.importorskip("virgil.models")
from virgil.fitting import fit  # noqa: E402
from virgil.likelihood import whitened_residuals  # noqa: E402
from virgil.oidata import OIData  # noqa: E402

pytestmark = pytest.mark.x64

INC_RANGES = [(0.0, 180.0), (0.0, 90.0), (20.0, 70.0), (0.0, 0.01), (179.98, 180.0), (89.0, 91.0)]
LAT_RANGES = [(-np.pi / 2, np.pi / 2), (0.0, np.pi / 2), (-0.3, 0.4), (np.pi / 2 - 1e-4, np.pi / 2)]


def _cos_diff(a, b):
    """cos a - cos b, exactly as 2 sin((a+b)/2) sin((b-a)/2) (no cancellation)."""
    return 2 * np.sin(0.5 * (a + b)) * np.sin(0.5 * (b - a))


def _sin_diff(a, b):
    """sin b - sin a = 2 cos((a+b)/2) sin((b-a)/2)."""
    return 2 * np.cos(0.5 * (a + b)) * np.sin(0.5 * (b - a))


def inc_pdf(i, lo, hi):  # per degree
    a, b = np.deg2rad(lo), np.deg2rad(hi)
    return np.sin(np.deg2rad(i)) * np.pi / 180 / _cos_diff(a, b)


def inc_cdf(i, lo, hi):
    a, b, x = np.deg2rad(lo), np.deg2rad(hi), np.deg2rad(i)
    return _cos_diff(a, x) / _cos_diff(a, b)


def lat_pdf(x, lo, hi):
    return np.cos(x) / _sin_diff(lo, hi)


def lat_cdf(x, lo, hi):
    return _sin_diff(lo, x) / _sin_diff(lo, hi)


@pytest.mark.parametrize("lo,hi", INC_RANGES)
@pytest.mark.validates("virgil.priors.IsotropicInclination", roots=["mathematics", "statistics"])
def test_isotropic_inclination(lo, hi):
    p = pr.IsotropicInclination(lo, hi)
    x = np.linspace(lo, hi, 41)[1:-1]
    np.testing.assert_allclose(np.exp(np.asarray(p.log_prob(x))), inc_pdf(x, lo, hi), rtol=1e-9)
    np.testing.assert_allclose(np.asarray(p.cdf(x)), inc_cdf(x, lo, hi), rtol=1e-8, atol=1e-12)
    total = integrate.quad(lambda t: float(np.exp(p.log_prob(t))), lo, hi, epsabs=1e-13)[0]
    mean = integrate.quad(lambda t: t * inc_pdf(t, lo, hi), lo, hi, epsabs=1e-13)[0]
    q = np.linspace(0.01, 0.99, 25)
    record("abs_norm_err", abs(total - 1))
    assert abs(total - 1) < 1e-8
    assert abs(float(p.mean) - mean) < 1e-8 * max(1.0, mean)
    np.testing.assert_allclose(inc_cdf(np.asarray(p.icdf(q)), lo, hi), q, atol=1e-8)
    if hi - lo > 1:
        draws = np.asarray(p.sample(jax.random.key(0), (20000,)))
        assert stats.kstest(draws, lambda t: inc_cdf(t, lo, hi)).pvalue > 1e-3
    assert float(p.log_prob(lo - 1.0 if lo > 0 else hi + 1.0)) == -np.inf or hi == 180.0


@pytest.mark.parametrize("lo,hi", LAT_RANGES)
@pytest.mark.validates("virgil.priors.IsotropicLatitude", roots=["mathematics", "statistics"])
def test_isotropic_latitude(lo, hi):
    p = pr.IsotropicLatitude(lo, hi)
    x = np.linspace(lo, hi, 41)[1:-1]
    np.testing.assert_allclose(np.exp(np.asarray(p.log_prob(x))), lat_pdf(x, lo, hi), rtol=1e-8)
    np.testing.assert_allclose(np.asarray(p.cdf(x)), lat_cdf(x, lo, hi), rtol=1e-7, atol=1e-12)
    total = integrate.quad(lambda t: float(np.exp(p.log_prob(t))), lo, hi, epsabs=1e-13)[0]
    q = np.linspace(0.01, 0.99, 25)
    assert abs(total - 1) < 1e-8
    np.testing.assert_allclose(lat_cdf(np.asarray(p.icdf(q)), lo, hi), q, atol=1e-7)
    if hi - lo > 0.1:
        draws = np.asarray(p.sample(jax.random.key(1), (20000,)))
        assert stats.kstest(draws, lambda t: lat_cdf(t, lo, hi)).pvalue > 1e-3


@pytest.mark.validates("virgil.priors.IsotropicInclination", "virgil.priors.IsotropicLatitude", roots=["mathematics"])
def test_flat_coordinates_are_affine_in_cos_i_and_sin_lat():
    """fit's flat coordinate is the CDF, which must be affine in cos i
    (inclination) and in sin lat (latitude)."""
    for p, f, xs in [(pr.IsotropicInclination(10.0, 150.0), lambda t: np.cos(np.deg2rad(t)), np.linspace(10, 150, 9)),
                     (pr.IsotropicLatitude(-0.5, 1.2), np.sin, np.linspace(-0.5, 1.2, 9))]:
        cdf, icdf, lo, hi = p.flat_coordinate()
        c = np.asarray(cdf(xs))
        coef = np.polyfit(f(xs), c, 1)
        assert np.max(np.abs(np.polyval(coef, f(xs)) - c)) < 1e-12
        assert (lo, hi) == (0.0, 1.0)
        np.testing.assert_allclose(np.asarray(icdf(c)), xs, atol=1e-8)


# ------------------------------------------------------------------ angles


def polar_integral(f, r_max=8.0):
    return integrate.dblquad(lambda r, t: r * f(r, t), 0, 2 * np.pi, 0, r_max, epsabs=1e-10)[0]


@pytest.mark.parametrize("kw", [dict(), dict(mean=40.0, kappa=3.0), dict(mean=40.0, kappa=3.0, axial=True)],
                         ids=["uniform", "von-mises", "axial"])
@pytest.mark.validates("virgil.angles.AngleVector", roots=["mathematics", "statistics"])
def test_angle_vector_density_and_marginal(kw):
    """p(v) normalised over the plane; the angle's marginal is the stated
    density (uniform, von Mises, or von Mises of 2t), for two ring widths."""
    for width in (None, 0.3):
        prior = ang.AngleVector(**kw) if width is None else ang.AngleVector(**kw, ring_width=width)

        def density(r, t, prior=prior):
            v = jnp.array([r * np.cos(t), r * np.sin(t)])
            return float(np.exp(prior.log_prob(v)))

        assert abs(polar_integral(density) - 1) < 1e-6
        for t in (0.3, 2.0, 4.5):
            marginal = integrate.quad(lambda r: r * density(r, t), 0, 8.0, epsabs=1e-12)[0]
            if not kw:
                want = 1 / (2 * np.pi)
            elif kw.get("axial"):
                want = stats.vonmises(kw["kappa"], loc=2 * np.deg2rad(kw["mean"])).pdf(2 * t)  # integrates to 1 over [0, 2 pi)
            else:
                want = stats.vonmises(kw["kappa"], loc=np.deg2rad(kw["mean"])).pdf(t)
            assert abs(marginal / want - 1) < 1e-6, (kw, width, t)


@pytest.mark.validates("virgil.angles.AngleVector", roots=["mathematics"])
def test_angle_vector_residuals_are_minus_log_density():
    """Half the squared residuals is -log p(v) up to a constant, so least
    squares fits optimise the same density."""
    prior = ang.AngleVector(mean=40.0, kappa=3.0)
    rng = np.random.default_rng(3)
    vs = rng.normal(size=(12, 2)) + 0.5
    half = np.array([0.5 * float(np.sum(np.asarray(prior.residuals(jnp.asarray(v))) ** 2)) for v in vs])
    lp = np.array([float(prior.log_prob(jnp.asarray(v))) for v in vs])
    const = half + lp
    assert np.ptp(const) < 1e-10


@pytest.mark.validates("virgil.angles.AngleVector", roots=["statistics"])
def test_angle_vector_samples():
    """Angles of samples follow the von Mises; radii follow r exp(-(r-1)^2/2s^2)."""
    prior = ang.AngleVector(mean=40.0, kappa=3.0)
    v = np.asarray(prior.sample(jax.random.key(4), (20000,)))
    theta = np.arctan2(v[:, 1], v[:, 0])
    assert stats.kstest(theta, stats.vonmises(3.0, loc=np.deg2rad(40.0)).cdf).pvalue > 1e-3
    s = 0.25  # the documented default ring width
    norm = integrate.quad(lambda r: r * np.exp(-((r - 1) ** 2) / (2 * s**2)), 0, np.inf)[0]

    def rcdf(r):
        return integrate.quad(lambda x: x * np.exp(-((x - 1) ** 2) / (2 * s**2)), 0, r)[0] / norm

    radius = np.hypot(v[:, 0], v[:, 1])
    assert stats.kstest(radius[:4000], np.vectorize(rcdf)).pvalue > 1e-3
    np.testing.assert_allclose(np.asarray(ang.vector_angle(jnp.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, -1e-9]]))),
                               [0.0, 90.0, 180.0], atol=1e-6)


# --------------------------------------------------------- MAP in flat coords

UTS3 = np.array([[-9.925, -20.335], [14.887, 30.502], [103.306, 43.999]])


@pytest.fixture(scope="module")
def binary_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("map") / "b.fits"

    def scene(u, v, w):
        return (sky.vis_uniform_disk(u, v, w, 1.2) + 0.04 * sky.vis_point(u, v, w, 6.0, -4.0)) / 1.04

    simulate.observe(path, scene, UTS3, hour_angles_h=np.linspace(-3, 3, 7), wavelengths=np.linspace(1.5e-6, 2.4e-6, 6),
                     dec_deg=-50.0, sigma_v2=0.02, sigma_cp_deg=1.0, rng=np.random.default_rng(6))
    return OIData(str(path))


def max_likelihood(data, template, paths, start):
    """The maximum of the likelihood by direct search (SciPy), on virgil's
    whitened residuals, independent of any prior or coordinate."""
    def chi2(x):
        return float(jnp.sum(whitened_residuals(template.set(paths, list(x)), data) ** 2))

    res = optimize.minimize(chi2, start, method="Nelder-Mead",
                            options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000, "maxfev": 40000})
    return res.x


@pytest.mark.validates("virgil.fitting.fit", "virgil.priors.IsotropicInclination", roots=["mathematics"])
def test_map_with_flat_coordinate_priors_is_the_likelihood_maximum(binary_file):
    """LogUniform priors on the diameter and the companion flux: their
    densities 1/x would pull a fit in x towards zero; in log x they are
    flat, so the MAP must be the likelihood maximum."""
    template = vm.System(star=vm.UniformDisk(1.0), comp=vm.PointSource(0.03, 6.2, -3.8))
    paths = ["star.diam", "comp.flux", "comp.dra", "comp.ddec"]
    priors = {"star.diam": dist.LogUniform(0.01, 10.0), "comp.flux": dist.LogUniform(1e-4, 1.0),
              "comp.dra": dist.Uniform(-20.0, 20.0), "comp.ddec": dist.Uniform(-20.0, 20.0)}
    got = fit(template, priors, binary_file)
    best = np.array([float(got.values[p]) for p in paths])
    want = max_likelihood(binary_file, template, paths, best * 1.01)
    sigma = np.sqrt(np.diag(np.linalg.inv(0.5 * np.asarray(jax.hessian(
        lambda x: jnp.sum(whitened_residuals(template.set(paths, list(x)), binary_file) ** 2))(jnp.asarray(want))))))
    pulls = (best - want) / sigma
    record("max_abs_dmap_over_sigma", float(np.max(np.abs(pulls))))
    assert np.max(np.abs(pulls)) < 1e-3


@pytest.mark.validates("virgil.fitting.fit", roots=["mathematics"], kind="control")
def test_a_gaussian_prior_moves_the_map(binary_file):
    """A Normal prior has no flat coordinate: it enters the loss, and a
    tight one pulls the MAP away from the likelihood maximum."""
    template = vm.System(star=vm.UniformDisk(1.0), comp=vm.PointSource(0.03, 6.2, -3.8))
    paths = ["star.diam", "comp.flux", "comp.dra", "comp.ddec"]
    flat = {"star.diam": dist.LogUniform(0.01, 10.0), "comp.flux": dist.LogUniform(1e-4, 1.0),
            "comp.dra": dist.Uniform(-20.0, 20.0), "comp.ddec": dist.Uniform(-20.0, 20.0)}
    tight = flat | {"star.diam": dist.Normal(0.8, 0.01)}
    a = float(fit(template, flat, binary_file).values["star.diam"])
    b = float(fit(template, tight, binary_file).values["star.diam"])
    assert abs(a - b) > 0.05
