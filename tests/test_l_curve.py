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


def kkt(d, b, w, kappa, scale):
    """Stationarity of χ² + κ w TSV on the simplex of pixel fluxes b.

    The fitted log-brightness runs far along its softmax gauge (shifts of
    1e7 or more) and the weakest pixels are driven to exactly zero flux, where
    the gradient with respect to log-brightness vanishes identically. A
    gradient in log-brightness is then zero for every loss and shows nothing,
    so we differentiate in flux b instead and apply the Karush-Kuhn-Tucker
    condition for Σ b = 1: on the pixels with b > 0 the gradient of the loss
    is one common number λ. Returns the largest departure from λ, relative to
    the gradient of χ² at the flat image, and the number of such pixels."""
    g = gradient(lambda y: chi2_image(d, y) + kappa * w * tsv(y), b, h=1e-8)
    live = b.ravel() > 1e-12
    r = g[live] - np.mean(g[live])
    return np.max(np.abs(r)) / scale, int(live.sum())


def flux_scale(d):
    flat = np.full((N, N), 1.0 / N**2)
    return np.max(np.abs(gradient(lambda y: chi2_image(d, y), flat, h=1e-8)))


@pytest.mark.validates("virgil.imaging.l_curve", roots=["mathematics"])
def test_each_point_minimizes_chi2_plus_twice_the_penalty(sweep):
    """At each fitted image the loss χ² + 2 w TSV is stationary on the
    pixels with flux (KKT, see kkt): what is left is below 1e-5 of the
    gradient of χ² at the flat image. Solving for κ in g_χ² + κ w g_TSV = λ
    gives κ = 2 to 1 % at the two strongest weights. At the weaker weights
    the penalty gradient is itself below that 1e-5 floor, so κ is not
    determined there (it wanders from 2 to 3) and only the residual is
    asserted."""
    lc, log_b, d = sweep
    scale = flux_scale(d)
    worst_left, worst_kappa = 0.0, 0.0
    for i, (w, x) in enumerate(zip(np.asarray(lc.weights), log_b)):
        b = softmax(x)
        left, n_live = kkt(d, b, w, 2.0, scale)
        assert n_live > 10  # not a vacuous check
        worst_left = max(worst_left, left)
        if i < 2:
            gc = gradient(lambda y: chi2_image(d, y), b, h=1e-8)
            gp = w * gradient(tsv, b, h=1e-8)
            live = b.ravel() > 1e-12
            design = np.c_[gp[live], -np.ones(live.sum())]
            kappa = np.linalg.lstsq(design, -gc[live], rcond=None)[0][0]
            worst_kappa = max(worst_kappa, abs(kappa - 2.0))
    record("max_abs_kappa_minus_2_strong_weights", worst_kappa)
    record("max_rel_gradient_left", worst_left)
    assert worst_kappa < 0.02
    assert worst_left < 1e-5


@pytest.mark.validates("virgil.imaging.l_curve", roots=["mathematics"], kind="control")
def test_chi2_plus_once_the_penalty_is_not_stationary(sweep):
    """The other reading of "added to the loss", χ² + w TSV, leaves a
    gradient on the pixels with flux at the strongest weight that is more
    than 100 times what χ² + 2 w TSV leaves."""
    lc, log_b, d = sweep
    scale = flux_scale(d)
    w, b = float(np.asarray(lc.weights)[0]), softmax(log_b[0])
    once, _ = kkt(d, b, w, 1.0, scale)
    twice, _ = kkt(d, b, w, 2.0, scale)
    record("kkt_left_kappa1_over_kappa2", once / twice)
    assert once > 3e-5 and once > 100 * twice


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
