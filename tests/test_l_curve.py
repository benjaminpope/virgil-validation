"""virgil.imaging.l_curve against our own χ², penalty and optimality.

l_curve fits an Image over a sweep of weights of one regularizer and
reports, per weight, the total χ², χ² per point and the unweighted penalty
``value / weight`` (its docstring and LCurve's). Here the regularizer is
TSV, ``w Σ (Δx b)² + (Δy b)²`` with zeros beyond the edges (the table in
virgil.imaging's docstring), on an 8 x 8 Image whose pixel fluxes are
``softmax(log_brightness)`` (virgil.models.Image). The data are three VLTI
UTs (uncorrelated closure phases), a Gaussian blob with a point beside it.

Everything is recomputed here, independently of virgil:

* χ² of each fitted image with crosscheck.chi2 (V² plus chord closure
  phases) and our direct-sum visibilities of its pixels (crosscheck.sky);
* TSV with NumPy from its documented formula;
* optimality: at each fitted image the gradient, with respect to the
  log-brightness, of the loss ``χ² + κ w TSV`` must vanish. We solve for κ
  by least squares from our own finite-difference gradients of χ² and TSV.
  fit's documentation adds the regularizer to "the loss" without saying
  whether that is χ² or ½χ² (the negative log likelihood); κ = 2 is the
  latter, κ = 1 the former. The tests show κ = 2.
* the L-curve's monotonicity, a property of exact minimizers of
  χ² + κ w R: χ² cannot fall and R cannot rise as w grows;
* LCurve.discrepancy from its documented definition: linear interpolation
  in log w of χ² per point to the target.
"""

import numpy as np
import pytest

from crosscheck import chi2 as ours, simulate, sky
from evidence.plugin import record

imaging = pytest.importorskip("virgil.imaging")
vm = pytest.importorskip("virgil.models")
from virgil.oidata import OIData  # noqa: E402

pytestmark = pytest.mark.x64

UTS3 = np.array([[-9.925, -20.335], [14.887, 30.502], [103.306, 43.999]])
N, PIXEL = 8, 1.0
WEIGHTS = [1e3, 3e2, 1e2, 3e1, 1e1]


def scene(u, v, w):
    return (sky.vis_gaussian(u, v, w, 2.0) + 0.3 * sky.vis_point(u, v, w, 2.0, -1.0)) / 1.3


def softmax(log_b):
    e = np.exp(log_b - log_b.max())
    return e / e.sum()


def tsv(b):
    """Σ (Δx b)² + (Δy b)² between neighbours, zeros beyond the edges."""
    p = np.pad(b, 1)
    return float(np.sum(np.diff(p, axis=1) ** 2) + np.sum(np.diff(p, axis=0) ** 2))


def chi2_image(d, b):
    return ours.chi2(d, lambda u, v, w: sky.visibility(sky.pixel_image(b, PIXEL), u, v, w))


def gradient(f, x, h=1e-6):
    g = np.zeros(x.size)
    for i in range(x.size):
        e = np.zeros(x.size)
        e[i] = h
        e = e.reshape(x.shape)
        g[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    path = tmp_path_factory.mktemp("lcurve") / "lc.fits"
    simulate.observe(path, scene, UTS3, hour_angles_h=np.linspace(-3, 3, 5), wavelengths=np.array([1.65e-6, 2.2e-6]),
                     dec_deg=-30.0, sigma_v2=0.01, sigma_cp_deg=1.0, rng=np.random.default_rng(1))
    model = vm.Image(np.zeros((N, N)), PIXEL)
    lc = imaging.l_curve(model, imaging.image_priors(model), OIData(str(path)), imaging.TSV(1.0), WEIGHTS)
    log_b = [np.asarray(r.values["log_brightness"]) for r in lc.results]
    return lc, log_b, ours.load(path)


@pytest.mark.validates("virgil.imaging.l_curve", roots=["mathematics"])
def test_points_match_our_chi2_and_penalty(sweep):
    """Weights largest first; per weight, χ², χ² per point and the penalty
    equal ours of the fitted image (rel. 1e-8)."""
    lc, log_b, d = sweep
    images = [softmax(x) for x in log_b]
    assert list(np.asarray(lc.weights)) == sorted(WEIGHTS, reverse=True)
    chi2 = np.array([chi2_image(d, b) for b in images])
    pen = np.array([tsv(b) for b in images])
    rel_chi2 = np.max(np.abs(np.asarray(lc.chi2) / chi2 - 1))
    rel_red = np.max(np.abs(np.ravel(lc.chi2_red) / (chi2 / ours.n_data(d)) - 1))
    rel_pen = np.max(np.abs(np.asarray(lc.penalty) / pen - 1))
    record("max_rel_chi2", rel_chi2)
    record("max_rel_chi2_red", rel_red)
    record("max_rel_penalty", rel_pen)
    assert rel_chi2 < 1e-8 and rel_red < 1e-8 and rel_pen < 1e-8


@pytest.mark.validates("virgil.imaging.l_curve", roots=["mathematics"])
def test_each_point_minimizes_chi2_plus_twice_the_penalty(sweep):
    """At each fitted image our gradients satisfy g_χ² + κ w g_TSV = 0 with
    κ = 2 (to 1 %), and what is left over is below 1e-5 of the gradient
    of χ² at the flat starting image."""
    lc, log_b, d = sweep
    flat = np.zeros((N, N))
    scale = np.max(np.abs(gradient(lambda x: chi2_image(d, softmax(x)), flat)))
    worst_kappa, worst_left = 0.0, 0.0
    for w, x in zip(np.asarray(lc.weights), log_b):
        g_chi = gradient(lambda y: chi2_image(d, softmax(y)), x)
        g_pen = w * gradient(lambda y: tsv(softmax(y)), x)
        kappa = -(g_chi @ g_pen) / (g_pen @ g_pen)
        left = np.max(np.abs(g_chi + 2.0 * g_pen)) / scale
        worst_kappa, worst_left = max(worst_kappa, abs(kappa - 2.0)), max(worst_left, left)
    record("max_abs_kappa_minus_2", worst_kappa)
    record("max_rel_gradient_left", worst_left)
    assert worst_kappa < 0.02
    assert worst_left < 1e-5


@pytest.mark.validates("virgil.imaging.l_curve", roots=["mathematics"], kind="control")
def test_chi2_plus_once_the_penalty_is_not_stationary(sweep):
    """The other reading of "added to the loss", χ² + w TSV, leaves a
    gradient ten times the tolerance above at the strongest weight."""
    lc, log_b, d = sweep
    flat = np.zeros((N, N))
    scale = np.max(np.abs(gradient(lambda x: chi2_image(d, softmax(x)), flat)))
    w, x = float(np.asarray(lc.weights)[0]), log_b[0]
    g = gradient(lambda y: chi2_image(d, softmax(y)) + w * tsv(softmax(y)), x)
    assert np.max(np.abs(g)) / scale > 1e-4


@pytest.mark.validates("virgil.imaging.l_curve", roots=["mathematics"])
def test_the_curve_is_monotone_and_discrepancy_interpolates(sweep):
    """As the weight falls, our χ² does not rise and our TSV does not fall
    (exact minimizers; slack 1e-6 relative). LCurve.discrepancy at a target
    between two fitted χ² per point is our linear interpolation in log w."""
    lc, log_b, d = sweep
    images = [softmax(x) for x in log_b]
    w = np.asarray(lc.weights)
    chi2 = np.array([chi2_image(d, b) for b in images])
    pen = np.array([tsv(b) for b in images])
    assert np.all(np.diff(chi2) <= 1e-6 * chi2[:-1])
    assert np.all(np.diff(pen) >= -1e-6 * pen[:-1])
    red = chi2 / ours.n_data(d)
    target = 0.5 * (red[1] + red[2])
    t = (target - red[1]) / (red[2] - red[1])
    want = float(np.exp(np.log(w[1]) + t * (np.log(w[2]) - np.log(w[1]))))
    got = lc.discrepancy(target)
    record("rel_discrepancy", abs(got / want - 1))
    assert got == pytest.approx(want, rel=1e-8)
    assert lc.discrepancy(0.5 * red.min()) is None
