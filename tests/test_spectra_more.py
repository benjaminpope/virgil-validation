"""virgil.spectra: Nodes, GaussianLine, LorentzianLine and Sum.

* Nodes, linear: numpy.interp, and the documented outside behaviour (held
  at the end values, or a fixed number).
* Nodes, cubic: a natural cubic spline, against SciPy's
  CubicSpline(bc_type="natural"); is_physical, which looks for negative
  overshoot between the nodes from the spline's extrema, against a dense
  evaluation of SciPy's spline.
* GaussianLine and LorentzianLine: their profiles, and the integrals over
  wavelength their docs state, by SciPy quadrature.
* Sum: the sum of its parts at every wavelength, and its reference flux at
  wavel0.
"""

import numpy as np
import pytest
from scipy import integrate, interpolate

from evidence.plugin import record

sp = pytest.importorskip("virgil.spectra")

pytestmark = pytest.mark.x64

NODES = np.array([2.00e-6, 2.05e-6, 2.12e-6, 2.20e-6, 2.31e-6])
VALUES = np.array([0.4, 0.9, 0.3, 0.6, 0.5])
WL = np.linspace(1.9e-6, 2.4e-6, 301)


def ev(s, w):
    return np.asarray(s(np.asarray(w)))


@pytest.mark.validates("virgil.spectra.Nodes", roots=["mathematics"])
def test_linear_nodes_and_outside():
    inside = (WL >= NODES[0]) & (WL <= NODES[-1])
    held = sp.Nodes(VALUES, NODES)
    np.testing.assert_allclose(ev(held, WL), np.interp(WL, NODES, VALUES), rtol=0, atol=1e-14)
    zero = sp.Nodes(VALUES, NODES, outside=0.0)
    np.testing.assert_allclose(ev(zero, WL)[inside], np.interp(WL, NODES, VALUES)[inside], atol=1e-14)
    assert np.all(ev(zero, WL)[~inside] == 0.0)
    assert float(held()) == pytest.approx(VALUES[0])  # wavel0 is the first node


@pytest.mark.validates("virgil.spectra.Nodes", roots=["mathematics"])
def test_cubic_nodes_are_scipys_natural_spline():
    s = sp.Nodes(VALUES, NODES, kind="cubic")
    ref = interpolate.CubicSpline(NODES, VALUES, bc_type="natural")
    inside = (WL >= NODES[0]) & (WL <= NODES[-1])
    diff = np.max(np.abs(ev(s, WL)[inside] - ref(WL[inside])))
    record("max_abs_diff", diff)
    assert diff < 1e-13
    np.testing.assert_allclose(ev(s, WL)[WL < NODES[0]], VALUES[0], atol=1e-14)
    np.testing.assert_allclose(ev(s, WL)[WL > NODES[-1]], VALUES[-1], atol=1e-14)


@pytest.mark.parametrize("values", [VALUES, np.array([0.05, 0.9, 0.02, 0.9, 0.05]), np.array([0.5, 0.01, 0.5, 0.5, 0.5])])
@pytest.mark.validates("virgil.spectra.Nodes", roots=["mathematics"])
def test_cubic_is_physical_sees_overshoot_between_nodes(values):
    """Every node may be positive while the spline dips below zero between
    them; is_physical must say so exactly when a dense evaluation of the
    natural spline goes negative."""
    s = sp.Nodes(values, NODES, kind="cubic")
    ref = interpolate.CubicSpline(NODES, values, bc_type="natural")
    dense = ref(np.linspace(NODES[0], NODES[-1], 200001))
    assert bool(s.is_physical()) == bool(dense.min() >= 0.0), (dense.min(), values)


@pytest.mark.parametrize("cls,integral", [
    ("GaussianLine", lambda a, f: a * f * np.sqrt(np.pi / (4 * np.log(2)))),
    ("LorentzianLine", lambda a, f: a * np.pi * f / 2),
])
@pytest.mark.validates("virgil.spectra.GaussianLine", "virgil.spectra.LorentzianLine", roots=["mathematics"])
def test_lines_and_their_integrals(cls, integral):
    amp, centre, fwhm = 0.4, 2.1661e-6, 1.0e-9
    line = getattr(sp, cls)(amp, line_wavel=centre, fwhm=fwhm)
    assert float(line(centre)) == pytest.approx(amp, rel=1e-14)
    assert float(line(centre + fwhm / 2)) == pytest.approx(amp / 2, rel=1e-12)  # half maximum
    x = (WL - centre) / fwhm
    profile = np.exp(-4 * np.log(2) * x**2) if cls == "GaussianLine" else 1 / (1 + 4 * x**2)
    np.testing.assert_allclose(ev(line, WL), amp * profile, rtol=1e-13, atol=1e-300)
    # in units of the width, x = (wavel - centre) / fwhm, out to |x| = edge
    edge = 30.0 if cls == "GaussianLine" else 1e4
    total = fwhm * integrate.quad(lambda x: float(line(centre + fwhm * x)), -edge, edge, points=[0.0], limit=2000,
                                  epsabs=1e-13)[0]
    if cls == "LorentzianLine":  # its wings beyond |x| = edge, analytically
        total += amp * fwhm * (np.pi / 2 - np.arctan(2 * edge))
    record("rel_integral", abs(total / integral(amp, fwhm) - 1))
    assert abs(total / integral(amp, fwhm) - 1) < 1e-8


@pytest.mark.validates("virgil.spectra.Sum", roots=["mathematics"])
def test_sum_is_the_sum_of_its_parts():
    parts = dict(
        continuum=sp.PowerLaw(1.0, index=-2.0, wavel0=2.2e-6),
        brg=sp.GaussianLine(-0.3, line_wavel=2.1661e-6, fwhm=1.0e-9),
        excess=sp.Nodes([0.0, 0.5, 0.0], [2.16e-6, 2.166e-6, 2.172e-6], outside=0.0),
    )
    total = sp.Sum(**parts)
    want = sum(ev(p, WL) for p in parts.values())
    np.testing.assert_allclose(ev(total, WL), want, rtol=1e-14, atol=1e-15)
    assert float(total()) == pytest.approx(float(sum(np.asarray(p(2.2e-6)) for p in parts.values())), rel=1e-14)
    assert set(total.components) == set(parts)
