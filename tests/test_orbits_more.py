"""virgil.orbits beyond KeplerOrbit's positions (which orbitize! covers),
against our own textbook Kepler code (crosscheck.orbits) and SciPy.

* KeplerOrbit.relative: positions from crosscheck.orbits (Newton's method
  on Kepler's equation, the visual-binary projection).
* AxialVonMises: exp(kappa cos 2(t - mean)) / (360 I0(kappa)) per degree,
  normalised on [0, 360); t and t + 180 equally likely; samples by KS.
* orientation_from_varpi, orientation_priors: varpi = Omega + omega, the
  node in [0, 180) from 2 Omega; the priors are uniform angle vectors.
* position_angle_log_jacobian: log |dM/d theta| by finite differences of our
  own projection (M from theta through the true anomaly), and its integral
  over a full turn of theta is 2 pi.
* starting_orbits: each grid point is an exact weighted least-squares solve
  in the Thiele-Innes constants, so its chi-squared equals our own linear
  solve at that period, eccentricity and periastron time; the best orbit
  of positions from a known orbit is that orbit.
* models.Attached: a companion attached to an orbit, evaluated at a time,
  is the static binary with the companion at that orbit position.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, optimize, special, stats

from crosscheck import orbits as co, sky
from evidence.plugin import record

vo = pytest.importorskip("virgil.orbits")
pytest.importorskip("jaxoplanet")
vm = pytest.importorskip("virgil.models")

pytestmark = pytest.mark.x64

ORBITS = [
    (800.0, 120.0, 0.3, 60.0, 40.0, 110.0, 25.0),
    (3000.0, -900.0, 0.75, 130.0, 300.0, 15.0, 80.0),
    (40.0, 3.0, 0.0, 89.0, 0.0, 200.0, 4.0),
]


@pytest.mark.parametrize("elements", ORBITS)
@pytest.mark.validates("virgil.orbits.KeplerOrbit", roots=["mathematics"])
def test_kepler_positions_from_the_textbook(elements):
    o = vo.KeplerOrbit(*elements, t_ref=59000.0)
    t = np.linspace(59000.0, 59000.0 + 2.3 * elements[0], 37)
    got = np.asarray(o.relative(t))
    dra, ddec = co.position(t - 59000.0, elements[0], elements[1], *elements[2:])
    worst = max(np.max(np.abs(got[0] - dra)), np.max(np.abs(got[1] - ddec))) / elements[-1]
    record("max_rel_position", worst)
    assert worst < 1e-12


@pytest.mark.parametrize("mean,kappa", [(30.0, 0.5), (130.0, 4.0), (350.0, 20.0)])
@pytest.mark.validates("virgil.orbits.AxialVonMises", roots=["mathematics", "statistics"])
def test_axial_von_mises(mean, kappa):
    p = vo.AxialVonMises(mean, kappa)
    t = np.linspace(0.5, 359.5, 50)
    want = np.exp(kappa * np.cos(2 * np.deg2rad(t - mean))) / (360 * special.i0(kappa))
    np.testing.assert_allclose(np.exp(np.asarray(p.log_prob(t))), want, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(p.log_prob(t)), np.asarray(p.log_prob(t + 180.0)), atol=1e-12)
    assert abs(integrate.quad(lambda x: float(np.exp(p.log_prob(x))), 0, 360, limit=200)[0] - 1) < 1e-10
    draws = np.asarray(p.sample(jax.random.key(2), (20000,)))
    grid = np.linspace(0, 360, 3601)
    cdf = np.concatenate([[0], np.cumsum(want_at(grid, mean, kappa)[1:] * np.diff(grid))])
    assert stats.kstest(draws, lambda x: np.interp(x, grid, cdf / cdf[-1])).pvalue > 1e-3


def want_at(t, mean, kappa):
    return np.exp(kappa * np.cos(2 * np.deg2rad(t - mean))) / (360 * special.i0(kappa))


def circ(x, period=360.0):
    """Distance of x from 0 on a circle of this period."""
    r = x % period
    return min(r, period - r)


@pytest.mark.validates("virgil.orbits.orientation_from_varpi", "virgil.orbits.orientation_priors", roots=["mathematics"])
def test_orientation_bookkeeping():
    rng = np.random.default_rng(1)
    for varpi, Omega in rng.uniform(0, 360, (20, 2)):
        omega, W = (float(x) for x in vo.orientation_from_varpi(varpi, Omega=Omega))
        assert W == pytest.approx(Omega) and circ(omega + W - varpi) < 1e-9
        omega2, W2 = (float(x) for x in vo.orientation_from_varpi(varpi, two_Omega=2 * Omega))
        assert 0 <= W2 < 180 and circ(W2 - Omega, 180.0) < 1e-9
        assert circ(omega2 + W2 - varpi) < 1e-9
    priors = vo.orientation_priors()
    assert set(priors) == {"two_Omega", "varpi"}
    assert set(vo.orientation_priors(positions_only=False, prefix="orbit.")) == {"orbit.Omega", "orbit.varpi"}
    with pytest.raises(ValueError):
        vo.orientation_from_varpi(10.0)


def mean_anomaly_of_theta(theta_rad, ecc, inc, omega, Omega):
    """M at which our projected orbit has position angle theta (radians):
    the true anomaly from a root search of our own position angle."""
    def gap(f):
        return np.angle(np.exp(1j * (co.position_angle(f, ecc, inc, omega, Omega) - theta_rad)))

    fs = np.linspace(-np.pi, np.pi, 721)
    g = np.array([gap(f) for f in fs])
    k = np.flatnonzero((np.sign(g[:-1]) != np.sign(g[1:])) & (np.abs(g[:-1] - g[1:]) < np.pi))[0]
    f = optimize.brentq(gap, fs[k], fs[k + 1], xtol=1e-15)
    E = 2 * np.arctan2(np.sqrt(1 - ecc) * np.sin(f / 2), np.sqrt(1 + ecc) * np.cos(f / 2))
    return E - ecc * np.sin(E)


@pytest.mark.parametrize("ecc,inc,omega,Omega", [(0.3, 50.0, 40.0, 110.0), (0.7, 130.0, 250.0, 20.0), (0.0, 20.0, 0.0, 300.0)])
@pytest.mark.validates("virgil.orbits.position_angle_log_jacobian", roots=["mathematics"])
def test_position_angle_jacobian(ecc, inc, omega, Omega):
    thetas = np.linspace(5.0, 355.0, 15)
    got = np.asarray(vo.position_angle_log_jacobian(thetas, ecc, inc, omega, Omega))
    h = 1e-5
    want = []
    for th in thetas:
        dM = (mean_anomaly_of_theta(np.deg2rad(th + h), ecc, inc, omega, Omega)
              - mean_anomaly_of_theta(np.deg2rad(th - h), ecc, inc, omega, Omega))
        dM = np.angle(np.exp(1j * dM))
        want.append(np.log(abs(dM / np.deg2rad(2 * h))))
    np.testing.assert_allclose(got, want, atol=1e-6)
    turn = integrate.quad(lambda th: float(np.exp(vo.position_angle_log_jacobian(th, ecc, inc, omega, Omega))),
                          0, 360, limit=400)[0] * np.pi / 180
    record("abs_turn_minus_2pi", abs(turn - 2 * np.pi))
    assert abs(turn - 2 * np.pi) < 1e-8


def our_grid_chi2(t, dra, ddec, sigma, period, t_peri, ecc):
    """Our own weighted least squares at one grid point: dRA and dDec are
    each a linear combination of the unit orbit X = cos E - e,
    Y = sqrt(1 - e^2) sin E (the Thiele-Innes form, whatever its labels)."""
    E = co.eccentric_anomaly(np.mod(2 * np.pi * (t - t_peri) / period + np.pi, 2 * np.pi) - np.pi, ecc)
    X = np.c_[np.cos(E) - ecc, np.sqrt(1 - ecc**2) * np.sin(E)] / sigma[:, None]
    total = 0.0
    for d in (dra, ddec):
        coef = np.linalg.lstsq(X, d / sigma, rcond=None)[0]
        total += np.sum((X @ coef - d / sigma) ** 2)
    return total


@pytest.mark.validates("virgil.orbits.starting_orbits", "virgil.orbits.PositionData", roots=["mathematics"])
def test_starting_orbits_grid_is_an_exact_linear_solve():
    truth = (900.0, 150.0, 0.4, 55.0, 70.0, 120.0, 30.0)
    rng = np.random.default_rng(3)
    mjd = np.sort(rng.uniform(59000, 60800, 14))
    dra, ddec = co.position(mjd - 59000.0, truth[0], truth[1], *truth[2:])
    sigma = np.full(mjd.size, 0.05)
    dra, ddec = dra + sigma * rng.normal(size=mjd.size), ddec + sigma * rng.normal(size=mjd.size)
    cov = np.zeros((mjd.size, 2, 2))
    cov[:, 0, 0] = cov[:, 1, 1] = sigma**2
    data = vo.PositionData(mjd, dra, ddec, cov, t_ref=59000.0)
    periods = np.array([700.0, 800.0, 900.0, 1000.0])
    best = vo.starting_orbits(data, periods, eccs=np.array([0.0, 0.2, 0.4, 0.6]), n_phase=36, n_best=3)
    worst = 0.0
    for orbit, chi2 in best:
        want = our_grid_chi2(mjd - 59000.0, dra, ddec, sigma, float(orbit.period), float(orbit.dt_peri), float(orbit.ecc))
        worst = max(worst, abs(chi2 - want) / max(1.0, want))
    record("max_rel_chi2", worst)
    assert worst < 1e-8
    o = best[0][0]
    assert float(o.period) == 900.0 and float(o.ecc) == pytest.approx(0.4)
    assert abs(((float(o.dt_peri) - 150.0 + 450.0) % 900.0) - 450.0) <= 900.0 / 36  # within one phase step
    assert best[0][1] < mjd.size * 2 * 3  # a good fit, chi2 ~ number of data


@pytest.mark.validates("virgil.models.Attached", roots=["mathematics"])
def test_attached_companion_is_the_static_binary_at_each_time():
    elements = (800.0, 120.0, 0.3, 60.0, 40.0, 110.0, 25.0)
    orbit = vo.KeplerOrbit(*elements, t_ref=59000.0)
    scene = vm.System(star=vm.UniformDisk(1.0), comp=vm.Attached(vm.PointSource(0.05), orbit))
    rng = np.random.default_rng(4)
    u, v = rng.uniform(-100, 100, (2, 20))
    worst = 0.0
    for mjd in (59000.0, 59333.0, 60010.0):
        dra, ddec = co.position(mjd - 59000.0, elements[0], elements[1], *elements[2:])
        got = np.asarray(scene.at(mjd).model(jnp.asarray(u), jnp.asarray(v), 2e-6))
        want = (sky.vis_uniform_disk(u, v, 2e-6, 1.0) + 0.05 * sky.vis_point(u, v, 2e-6, float(dra[0]), float(ddec[0]))) / 1.05
        worst = max(worst, np.max(np.abs(got - want)))
    record("max_abs_dV", worst)
    assert worst < 1e-12
