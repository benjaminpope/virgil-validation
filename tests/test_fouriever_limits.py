"""Detection limits and significance against fouriever (Stage 8, part 1).

fouriever (J. Kammerer; github.com/kammerje/fouriever) computes detection
limits by both of the methods virgil has: Absil et al. (2011), the flux at
which a companion would be ruled out at sigma, and the injection method of
Gallenne et al. (2015), the flux at which an injected companion would be
detected at sigma. It runs in its own environment through
src/external_bridge/fouriever_worker.py and reads our files with its own
reader; conventions in docs/method/fouriever.md.

* nsigma: fouriever's util.nsigma, with SciPy and with mpmath, against
  virgil.limits.nsigma.
* absil_limits and injection_limits: fouriever's own criteria
  (uvfit.lim_absil, lim_injection, with its injection uvfit.inj_companion),
  solved exactly by Brent's method in the worker, against virgil's, on
  three-UT files (independent closure phases) and on four-UT files with
  fouriever's closure-phase covariance (virgil's correlated closure
  phases). The primary is unresolved, so the uniform disk is held at 0.
* fouriever's public detlim (grid start, then L-BFGS-B on |nsigma -
  sigma|^2) against that exact solution of its own criteria: it fails
  under NumPy 2 and SciPy 1.18 (problem P8), pinned as a strict xfail.
* chi-squared maps on a seven-hole aperture mask (35 closure phases, 15
  independent, as for JWST NIRISS AMI) with fouriever's covariance against
  virgil's correlated likelihood_grid.

Differences of definition (docs/method/fouriever.md, the ledger): D7, the
degrees of freedom of correlated closure phases (fouriever counts all of
them, virgil the independent ones), 3 % in the four-UT limits, so the
comparison uses virgil's count in fouriever's criteria; D5, the closure-phase
residual (fouriever's plain difference, virgil's sine plus a periodic
penalty), negligible where closure phases are small, as they are at a
detection limit; D6, the generalised inverse, which agrees for equal
closure-phase errors (used here).
"""

import numpy as np
import pytest

from crosscheck import array, simulate, sky
from evidence.plugin import record
from external_bridge import _subprocess as sp

vm = pytest.importorskip("virgil.models")
lim = pytest.importorskip("virgil.limits")
from virgil.grid_fit import likelihood_grid  # noqa: E402
from virgil.oidata import OIData  # noqa: E402

pytestmark = [
    pytest.mark.x64,
    pytest.mark.external,
    pytest.mark.skipif(not sp.available("fouriever"), reason="fouriever not installed (scripts/setup_external.sh)"),
]

UTS3 = np.array([[-9.925, -20.335], [14.887, 30.502], [103.306, 43.999]])
UTS4 = np.array([[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]])
# seven holes of a non-redundant mask, metres on the sky-oriented pupil (NIRISS-like layout)
HOLES = np.array([[0.0, -2.64], [-2.29, 0.0], [2.29, -1.32], [-2.29, 1.32], [-1.14, 1.98], [2.29, 1.32], [1.14, 1.98]])
XS, YS = np.array([-8.0, 5.0]), np.array([-7.0, 6.0])


def binary_vis(f, x, y):
    def vis(u, v, w):
        return (sky.vis_point(u, v, w) + f * sky.vis_point(u, v, w, x, y)) / (1 + f)

    return vis


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """Companion-free files: three and four UTs (Earth rotation), and the
    seven-hole mask (one pointing per snapshot) with a 1 % companion."""
    root = tmp_path_factory.mktemp("fv_limits")
    out = {}
    for name, uts in (("uts3", UTS3), ("uts4", UTS4)):
        out[name] = root / f"{name}.fits"
        simulate.observe(out[name], binary_vis(0.0, 0, 0), uts, hour_angles_h=np.linspace(-3, 3, 5),
                         wavelengths=np.linspace(1.5e-6, 2.4e-6, 4), dec_deg=-50.0, sigma_v2=0.01, sigma_cp_deg=0.5,
                         rng=np.random.default_rng(11))
    out["mask"] = root / "mask.fits"
    simulate.observe(out["mask"], binary_vis(0.01, 120.0, -80.0), HOLES, hour_angles_h=[0.0, 1.0],
                     wavelengths=np.array([4.8e-6]), fixed_uv=array.pupil_uv(HOLES), sigma_v2=0.02,
                     sigma_cp_deg=1.0, rng=np.random.default_rng(12))
    return out


def fouriever(task, **kw):
    return sp.run("fouriever", "fouriever_worker.py", {"task": task, **kw})


@pytest.mark.validates("virgil.limits.nsigma", roots=["fouriever"])
def test_nsigma_matches_fouriever():
    """Both are Absil et al.'s eq. 1: the chi-squared CDF of ndof * ratio as a
    two-sided Gaussian significance. fouriever's SciPy branch saturates near
    8 sigma (its CDF reaches 1 - 1e-15); virgil works in log space."""
    cases = [(r, 1.0, n) for n in (30, 200, 1000) for r in (1.02, 1.05, 1.1, 1.2, 1.4)]
    got = np.array([float(lim.nsigma(a, b, n)) for a, b, n in cases])
    fv = fouriever("nsigma", cases=cases)
    want = np.array(fv["mpmath"])
    ok = want < 7.5  # where fouriever's own SciPy branch is exact too
    record("max_abs_dsigma", float(np.max(np.abs(got[ok] - want[ok]))))
    np.testing.assert_allclose(got[ok], want[ok], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.array(fv["scipy"])[ok], want[ok], rtol=1e-7)
    assert np.all(np.array(fv["scipy"])[~ok] <= 8.1)  # saturated, not virgil's concern
    assert np.all(got[~ok] > 7.5)


def virgil_limits(path, method):
    data = OIData(str(path))
    fn = lim.absil_limits if method == "absil" else lim.injection_limits
    out = np.asarray(fn(data, vm.BinaryModelCartesian, {"dra": XS, "ddec": YS, "flux": [1e-3]}, sigma=3.0))
    return np.array([out[i, j] for i in range(XS.size) for j in range(YS.size)])


POSITIONS = [[float(x), float(y)] for x in XS for y in YS]


@pytest.mark.parametrize("name,cov", [("uts3", False), ("uts4", True)], ids=["3UT", "4UT-correlated"])
@pytest.mark.validates("virgil.limits.absil_limits", "virgil.limits.injection_limits", roots=["fouriever"])
def test_limits_match_fouriever_exactly(files, name, cov):
    """With the same degrees of freedom. fouriever counts every closure
    phase as one (definition D7); virgil counts the independent ones
    (n_independent: 3 of the 4 triangles of four telescopes per snapshot
    and channel), the rank of the covariance and so the number of degrees
    of freedom of the chi-squared. On three telescopes the two agree."""
    ndof = OIData(str(files[name])).n_independent
    own = fouriever("limits_exact", path=str(files[name]), positions=POSITIONS, cov=cov, sigma=3.0)
    same = fouriever("limits_exact", path=str(files[name]), positions=POSITIONS, cov=cov, sigma=3.0, ndof=ndof)
    worst = {}
    for method in ("absil", "injection"):
        got = virgil_limits(files[name], method)
        worst[method] = float(np.max(np.abs(got / np.array(same[method]) - 1)))
        record(f"max_rel_{method}", worst[method])
        record(f"max_rel_{method}_with_fouriever_ndof", float(np.max(np.abs(got / np.array(own[method]) - 1))))
    assert (own["ndof"] == ndof) == (name == "uts3")
    # virgil bisects to 1e-4; D5 is ~1e-5 of chi-squared at these closure phases
    assert worst["absil"] < 3e-4 and worst["injection"] < 3e-4, worst


@pytest.mark.xfail(strict=True, raises=RuntimeError, reason="P8: fouriever's detlim fails under NumPy 2 / SciPy 1.18")
@pytest.mark.validates("fouriever", roots=["mathematics"], kind="upstream")
def test_fouriever_public_detlim_against_its_exact_criteria(files):
    """fouriever's detlim minimises |nsigma - sigma|^2 with L-BFGS-B from the
    best of 200 log-spaced fluxes; how close that comes to its own exact
    solution, on the three-UT file's closure phases. (With V² detlim stops
    at a print of the fitted diameter under NumPy 2: problem P8. With
    closure phases alone it reports its Absil limits for both methods.)"""
    pub = fouriever("detlim", path=str(files["uts3"]), cov=False, sigma=3.0, sep_range=[2.0, 10.0], step_size=3.0,
                    observables=["cp"])
    dra, ddec = np.array(pub["dra"]), np.array(pub["ddec"])
    absil, inj = np.array(pub["absil"]), np.array(pub["injection"])
    keep = (absil > 0) & (np.hypot(dra, ddec) > 2.0)
    positions = [[float(x), float(y)] for x, y in zip(dra[keep], ddec[keep])]
    exact = fouriever("limits_exact", path=str(files["uts3"]), positions=positions, cov=False, sigma=3.0,
                      observables=["cp"])
    rel_a = absil[keep] / np.array(exact["absil"]) - 1
    record("median_rel_absil", float(np.median(rel_a)))
    record("max_abs_rel_absil", float(np.max(np.abs(rel_a))))
    np.testing.assert_array_equal(inj[keep], absil[keep])  # its closure-phase branch copies Absil
    assert np.max(np.abs(rel_a)) < 0.05


@pytest.mark.validates("virgil.grid_fit.likelihood_grid", "virgil.oidata.OIData.cp_noise", roots=["fouriever"])
def test_mask_chi2_map_with_correlated_closure_phases(files):
    """A seven-hole mask: 35 closure phases per snapshot, 15 independent.
    virgil's correlated likelihood map against fouriever's chi2_bin with
    its closure-phase covariance, as chi-squared differences from the
    null (both codes' map statistic), over a grid around the companion."""
    data = OIData(str(files["mask"]))
    xs, ys = np.linspace(80.0, 160.0, 9), np.linspace(-120.0, -40.0, 9)
    flux = 0.01
    grid = np.asarray(likelihood_grid(data, vm.BinaryModelCartesian, {"dra": xs, "ddec": ys, "flux": [flux]}))
    got = -2 * grid.reshape(xs.size, ys.size)
    null = -2 * float(np.asarray(likelihood_grid(data, vm.BinaryModelCartesian,
                                                 {"dra": [0.0], "ddec": [0.0], "flux": [0.0]})).ravel()[0])
    positions = [[float(x), float(y)] for x in xs for y in ys]
    fv = fouriever("chi2_grid", path=str(files["mask"]), positions=positions, flux=flux, cov=True)
    want = (np.array(fv["chi2"]) - fv["chi2_null"]).reshape(xs.size, ys.size)
    gap = np.max(np.abs((got - null) - want)) / np.max(np.abs(want))
    record("max_rel_dchi2_map", float(gap))
    assert np.unravel_index(np.argmin(got), got.shape) == np.unravel_index(np.argmin(want), want.shape)
    assert gap < 1e-3
