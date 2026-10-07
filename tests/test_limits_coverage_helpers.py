"""Small public helpers in virgil.limits, virgil.orbits and virgil.coverage,
against SciPy, closed forms and our own mask geometry.

* chi2ppf: scipy.stats.chi2.ppf for one degree of freedom; for several it
  needs tensorflow_probability, which virgil does not declare (F17).
* delta_mag_to_flux, contrast_to_flux: 10^(-Δm/2.5) and 1/contrast, and
  the inverses of the existing flux_to_* nodes.
* radial_profile: our own NumPy annuli, equal widths from 0 to r_max.
* position_angle_prior: the term's loglike is log|dM/dθ| by finite
  differences of our own projection (crosscheck.orbits), and log_norm its
  negative.
* mask_transfer: the autocorrelation of identical circular holes, in closed
  form: Σ over ordered hole pairs of the overlap area of two circles at
  |b - (x_j - x_i)|, over N times the area of one hole.
* nrm_oidata: the V² sit at every hole-pair baseline (either sign) of the
  mask turned to ``rotation_deg``, the closure phases at every hole
  triangle, each the closing sum b_ab + b_bc = b_ac.
"""

from itertools import combinations

import numpy as np
import pytest
from scipy import optimize, stats

from crosscheck import array, orbits as co
from evidence.plugin import record

vl = pytest.importorskip("virgil.limits")
from virgil import coverage, orbits as vo  # noqa: E402

pytestmark = pytest.mark.x64


# ---------------------------------------------------------------- limits


def _chi2ppf_against_scipy(df):
    p = np.array([0.1, 0.5, 0.6827, 0.9, 0.99, 0.9999])
    got = np.asarray(vl.chi2ppf(p, df))
    want = stats.chi2.ppf(p, df)
    err = record("max_rel_dppf", np.max(np.abs(got / want - 1)))
    assert err < 1e-9


@pytest.mark.validates("virgil.limits.chi2ppf", roots=["statistics"])
def test_chi2ppf_is_scipys_for_one_degree_of_freedom():
    _chi2ppf_against_scipy(1)


@pytest.mark.validates("virgil.limits.chi2ppf", roots=["statistics"])
def test_f17_chi2ppf_works_for_several_degrees_of_freedom():
    """The documented fallback for df != 1 (numpyro's gammaincinv) raises
    ImportError ("Please install tensorflow_probability>=0.18") on an
    install of virgil-astro with its declared dependencies."""
    _chi2ppf_against_scipy(3)


@pytest.mark.validates("virgil.limits.delta_mag_to_flux", "virgil.limits.contrast_to_flux", roots=["mathematics"])
def test_magnitudes_and_contrasts():
    dm = np.array([0.0, 2.5, 5.0, 7.3])
    np.testing.assert_allclose(np.asarray(vl.delta_mag_to_flux(dm)), 10 ** (-dm / 2.5), rtol=1e-14)
    c = np.array([1.0, 10.0, 250.0])
    np.testing.assert_allclose(np.asarray(vl.contrast_to_flux(c)), 1 / c, rtol=1e-14)
    f = np.array([0.5, 0.01, 3e-4])
    np.testing.assert_allclose(np.asarray(vl.delta_mag_to_flux(vl.flux_to_delta_mag(f))), f, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(vl.contrast_to_flux(vl.flux_to_contrast(f))), f, rtol=1e-12)


@pytest.mark.validates("virgil.limits.radial_profile", roots=["mathematics"])
def test_radial_profile_is_annular_statistics():
    rng = np.random.default_rng(4)
    dra = np.linspace(-40, 40, 41)
    ddec = np.linspace(-30, 30, 31)
    centre = (2.0, -1.0)
    values = rng.normal(size=(dra.size, ddec.size)) + 0.05 * np.hypot(*np.meshgrid(dra - 2, ddec + 1, indexing="ij"))
    values[3, 4] = np.nan
    r_max, bins = 30.0, 6
    got = vl.radial_profile(values, dra, ddec, center=centre, r_max=r_max, bins=bins)
    r = np.hypot(*np.meshgrid(dra - centre[0], ddec - centre[1], indexing="ij"))
    edges = np.linspace(0, r_max, bins + 1)
    np.testing.assert_allclose(np.asarray(got["r"]), (edges[:-1] + edges[1:]) / 2, rtol=1e-12)
    worst = 0.0
    for k in range(bins):
        inside = (r >= edges[k]) & (r < edges[k + 1]) & np.isfinite(values)
        x = values[inside]
        assert int(got["count"][k]) == inside.sum()
        for key, want in (("mean", x.mean()), ("std", x.std()), ("median", np.median(x)),
                          ("q16", np.percentile(x, 16)), ("q84", np.percentile(x, 84))):
            worst = max(worst, abs(float(got[key][k]) - want))
    record("max_abs_dstat", worst)
    assert worst < 1e-12


# ---------------------------------------------------------------- orbits


def mean_anomaly_of_theta(theta_rad, ecc, inc, omega, Omega):
    """M at which our projected orbit has position angle theta (radians)."""
    def gap(f):
        return np.angle(np.exp(1j * (co.position_angle(f, ecc, inc, omega, Omega) - theta_rad)))

    fs = np.linspace(-np.pi, np.pi, 721)
    g = np.array([gap(f) for f in fs])
    k = np.flatnonzero((np.sign(g[:-1]) != np.sign(g[1:])) & (np.abs(g[:-1] - g[1:]) < np.pi))[0]
    f = optimize.brentq(gap, fs[k], fs[k + 1], xtol=1e-15)
    E = 2 * np.arctan2(np.sqrt(1 - ecc) * np.sin(f / 2), np.sqrt(1 + ecc) * np.cos(f / 2))
    return E - ecc * np.sin(E)


@pytest.mark.validates("virgil.orbits.position_angle_prior", roots=["mathematics"])
def test_position_angle_prior_term_is_the_jacobian():
    inc, omega, Omega = 60.0, 40.0, 110.0

    def orbit_fn(v):
        return vo.KeplerOrbit.from_position_angle(400.0, v["theta"], v["ecc"], inc, omega, Omega, 20.0, t_ref=60500.0)

    term = vo.position_angle_prior(orbit_fn)
    assert term.has_log_norm
    worst, h = 0.0, 1e-5
    for theta, ecc in [(15.0, 0.3), (140.0, 0.6), (260.0, 0.0)]:
        values = {"theta": theta, "ecc": ecc}
        dM = (mean_anomaly_of_theta(np.deg2rad(theta + h), ecc, inc, omega, Omega)
              - mean_anomaly_of_theta(np.deg2rad(theta - h), ecc, inc, omega, Omega))
        want = np.log(abs(np.angle(np.exp(1j * dM)) / np.deg2rad(2 * h)))
        got = float(term.loglike(values))
        worst = max(worst, abs(got - want))
        assert float(term.log_norm(values)) == pytest.approx(-got, abs=1e-12)
    record("max_abs_dlogjac", worst)
    assert worst < 1e-6


# -------------------------------------------------------------- coverage


def circle_overlap(s, d):
    """Area common to two circles of diameter d whose centres are s apart."""
    r = d / 2
    s = np.minimum(np.abs(s), 2 * r)
    return 2 * r**2 * np.arccos(s / (2 * r)) - (s / 2) * np.sqrt(4 * r**2 - s**2)


def mtf_closed_form(u, v, holes, d):
    total = np.zeros_like(u)
    for xi in holes:
        for xj in holes:
            total += circle_overlap(np.hypot(u - (xj[0] - xi[0]), v - (xj[1] - xi[1])), d)
    return total / (len(holes) * circle_overlap(0.0, d))


HOLES = np.array([[0.0, -2.64], [-2.28631, 0.0], [2.28631, -1.32], [-2.28631, 1.32],
                  [-1.14315, 1.98], [2.28631, 1.32], [1.14315, 1.98]])


@pytest.mark.validates("virgil.coverage.mask_transfer", roots=["mathematics"])
def test_mask_transfer_is_the_pupil_autocorrelation():
    d = 0.82
    rng = np.random.default_rng(9)
    u, v = rng.uniform(-6, 6, (2, 2000))
    bu, bv = array.pupil_uv(HOLES)
    u = np.concatenate([u, [0.0], bu, -bu])
    v = np.concatenate([v, [0.0], bv, -bv])
    got = np.asarray(coverage.mask_transfer(u, v, holes=HOLES, diameter=d))
    want = mtf_closed_form(u, v, HOLES, d)
    err = record("max_abs_dmtf", np.max(np.abs(got - want)))
    assert err < 1e-12
    assert got[2000] == pytest.approx(1.0) and np.allclose(got[2001:], 1 / 7)


def rotate(holes, rotation_deg):
    """The mask's "up" (+y) axis turned to position angle ``rotation_deg``
    North through East, a proper rotation: (x, y) -> East, North."""
    t = np.deg2rad(rotation_deg)
    x, y = holes[:, 0], holes[:, 1]
    return np.stack([x * np.cos(t) + y * np.sin(t), -x * np.sin(t) + y * np.cos(t)], 1)


@pytest.mark.parametrize("rotation", [0.0, 37.0])
@pytest.mark.validates("virgil.coverage.nrm_oidata", roots=["mathematics"])
def test_nrm_oidata_baselines_and_triangles(rotation):
    data = coverage.nrm_oidata(wavelength_m=3.8e-6, rotation_deg=rotation, sigma_v2=0.02, sigma_cp_deg=0.7,
                               holes=HOLES)
    u, v = np.asarray(data.u), np.asarray(data.v)
    sky = rotate(HOLES, rotation)
    pairs = [sky[j] - sky[i] for i, j in combinations(range(7), 2)]
    # each V² baseline is one hole pair, either sign, each pair exactly once
    match = [min(range(21), key=lambda k: min(np.hypot(*(np.array([a, b]) - pairs[k])),
                                              np.hypot(*(np.array([a, b]) + pairs[k])))) for a, b in zip(u, v)]
    dist = max(min(np.hypot(*(np.array([a, b]) - pairs[k])), np.hypot(*(np.array([a, b]) + pairs[k])))
               for a, b, k in zip(u, v, match))
    record("max_baseline_error_m", dist)
    assert sorted(match) == list(range(21)) and dist < 1e-9
    assert np.allclose(np.asarray(data.wavel), 3.8e-6)
    assert np.allclose(np.asarray(data.d_vis), 0.02) and np.allclose(np.asarray(data.d_phi), np.deg2rad(0.7))
    # closure phases: 35 triangles, each closing, every hole triple once
    i1, i2, i3 = (np.asarray(getattr(data, k)) for k in ("i_cps1", "i_cps2", "i_cps3"))
    assert len(i1) == 35
    np.testing.assert_allclose(u[i1] + u[i2], u[i3], atol=1e-12)
    np.testing.assert_allclose(v[i1] + v[i2], v[i3], atol=1e-12)
    triples = set()
    for a, b, c in zip(i1, i2, i3):
        holes_used = set()
        for k in (a, b, c):
            i, j = list(combinations(range(7), 2))[match[k]]
            holes_used |= {i, j}
        triples.add(tuple(sorted(holes_used)))
    assert triples == set(combinations(range(7), 3))
