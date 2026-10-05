"""virgil's correlated closure phases against fouriever.

With four or more telescopes, closure phases of one snapshot and channel
that share baselines are correlated. virgil's model is Kammerer et al.
(2020, A&A 644, A110, sec. 2.2): C = D^1/2 R D^1/2 with R = T T^T / 3.
fouriever (J. Kammerer; github.com/kammerje/fouriever) is that paper's own
code. It runs in its own environment (scripts/setup_external.sh) through
src/external_bridge/fouriever_worker.py, and reads our files with its own
reader.

Two differences of definition, both pinned here:

* D5 (as with CANDID): fouriever's closure-phase residual is the plain
  difference, virgil's (since virgil#174) the sine sin(delta), correlated,
  plus a periodic penalty (1 - cos delta) / sigma per closure phase.
* D6: C is singular, and the two codes use different generalised inverses.
  fouriever takes the pseudo-inverse, r^T C^+ r; virgil whitens by sigma
  first, r^T D^-1/2 R^+ D^-1/2 r. They agree whenever r is in C's column
  space, which is always the case when the triangles of a group have equal
  errors, and differ by a few per cent when they do not. On true
  closure-phase noise (closures of baseline phases), virgil's form is the
  closer to the nominal chi-squared.

Our own reference (crosscheck.chi2 with correlated=True) implements the
paper's model with either residual.
"""

import shutil

import jax
import numpy as np
import pytest
from astropy.io import fits

from crosscheck import chi2 as ours, simulate, sky
from evidence.plugin import record
from external_bridge import _subprocess as sp

vm = pytest.importorskip("virgil.models")
from virgil.likelihood import whitened_residuals  # noqa: E402
from virgil.oidata import OIData  # noqa: E402

pytestmark = [
    pytest.mark.x64,
    pytest.mark.external,
    pytest.mark.skipif(not sp.available("fouriever"), reason="fouriever not installed (scripts/setup_external.sh)"),
]

UTS = np.array([[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]])
FLUX, DRA, DDEC = 0.03, 6.0, -4.0


def binary_vis(f, x, y):
    def vis(u, v, w):
        return (sky.vis_point(u, v, w) + f * sky.vis_point(u, v, w, x, y)) / (1 + f)

    return vis


def params(n=10, seed=2):
    rng = np.random.default_rng(seed)
    out = [[FLUX, DRA, DDEC], [0.0, 0.0, 0.0]]
    return out + [[float(f), float(x), float(y)] for f, x, y in
                  zip(rng.uniform(0, 0.08, n), rng.uniform(-15, 15, n), rng.uniform(-15, 15, n))]


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """A four-UT file with equal closure-phase errors, and a copy whose
    errors are scaled by random factors 0.5-2 per triangle and channel."""
    root = tmp_path_factory.mktemp("fouriever")
    equal = root / "equal.fits"
    simulate.observe(
        equal, binary_vis(FLUX, DRA, DDEC), UTS, hour_angles_h=np.linspace(-3, 3, 5),
        wavelengths=np.linspace(1.5e-6, 2.4e-6, 4), dec_deg=-50.0, sigma_v2=0.01,
        sigma_cp_deg=0.5, rng=np.random.default_rng(8),
    )
    unequal = root / "unequal.fits"
    shutil.copy(equal, unequal)
    with fits.open(unequal, mode="update") as h:
        err = h["OI_T3"].data["T3PHIERR"]
        h["OI_T3"].data["T3PHIERR"] = err * np.random.default_rng(4).uniform(0.5, 2.0, err.shape)
    return {"equal": equal, "unequal": unequal}


def virgil_chi2(path, ps):
    data = OIData(str(path))
    return np.array([
        float(np.sum(np.asarray(whitened_residuals(vm.BinaryModelCartesian(x, y, f), data)) ** 2))
        for f, x, y in ps
    ])


def fouriever(path, ps, cov=True):
    return sp.run("fouriever", "fouriever_worker.py", {"task": "chi2", "path": path, "params": ps, "cov": cov})


@pytest.mark.validates("virgil.oidata.OIData.cp_noise", roots=["fouriever", "mathematics"])
def test_closure_phase_correlation_matches_fouriever(files):
    """virgil's correlation matrix of the closure phases, fouriever's
    (its CPCOV divided by the errors) and ours (T T^T / 3 per snapshot and
    channel, from the file's station indices) are the same."""
    path = files["equal"]
    d = ours.load(path)
    nrow, nwl = d["cp"].shape
    ours_r = np.zeros((d["cp"].size,) * 2)
    blocks = []
    for mjd in np.unique(d["t3_mjd"]):
        rows = np.flatnonzero(d["t3_mjd"] == mjd)
        T = ours.triangle_matrix(d["t3_sta"][rows])
        R = T @ T.T / 3.0
        blocks.append(R)
        for k in range(nwl):
            idx = rows * nwl + k
            ours_r[np.ix_(idx, idx)] = R
    virgil_r = OIData(str(path)).cp_noise.correlation(d["cp"].size)
    assert np.max(np.abs(virgil_r - ours_r)) < 1e-14
    res = fouriever(path, [[0.0, 0.0, 0.0]])
    worst = 0.0
    for cov, err, R in zip(np.array(res["cpcov"]), np.array(res["dcp"]), blocks):
        s = err.ravel()  # fouriever orders (triangle, channel)
        worst = max(worst, np.max(np.abs(cov / np.outer(s, s) - np.kron(R, np.eye(nwl)))))
    assert worst < 1e-14


@pytest.mark.validates(
    "virgil.likelihood.whitened_residuals", "virgil.oidata.OIData.cp_noise",
    roots=["fouriever", "mathematics"],
)
def test_correlated_chi2_matches_fouriever(files):
    """Equal errors: fouriever equals our plain-residual r^T C^+ r, and
    virgil our sine form (sin Δ correlated, plus the periodic penalty), to
    rounding; near the truth, virgil and fouriever agree directly (the
    residuals are small, so the forms coincide to O(Δ³))."""
    path = files["equal"]
    ps = params()
    res = fouriever(path, ps)
    got_f = np.array(res["chi2"])
    d = ours.load(path)
    plain = np.array([ours.chi2(d, binary_vis(*p), correlated=True, chord=False) for p in ps])
    sine = np.array([ours.chi2(d, binary_vis(*p), correlated=True, sine=True) for p in ps])
    got_v = virgil_chi2(path, ps)
    record("rel_fouriever_vs_plain", np.max(np.abs(got_f / plain - 1)))
    record("rel_virgil_vs_sine", np.max(np.abs(got_v / sine - 1)))
    record("rel_virgil_vs_fouriever_at_truth", abs(got_v[0] / got_f[0] - 1))
    assert np.max(np.abs(got_f / plain - 1)) < 1e-12
    assert np.max(np.abs(got_v / sine - 1)) < 1e-12
    assert abs(got_v[0] / got_f[0] - 1) < 1e-4


@pytest.mark.validates("virgil.likelihood.whitened_residuals", roots=["fouriever"], kind="control")
def test_independent_closure_phases_overcount(files):
    """fouriever without its covariance treats every triangle as
    independent: its chi-squared then differs from virgil's by several per
    cent, so the agreement above is not an accident."""
    path = files["equal"]
    ps = params(4)
    got_f = np.array(fouriever(path, ps, cov=False)["chi2"])
    got_v = virgil_chi2(path, ps)
    assert np.min(np.abs(got_v / got_f - 1)) > 0.01


@pytest.mark.validates("external_bridge.fouriever_worker", roots=["fouriever", "mathematics"], kind="reference")
def test_plain_residual_across_the_phase_cut(tmp_path):
    """A companion brighter than the primary (flux ratio 1.5, so baseline
    phases reach beyond +-90 deg) puts summed baseline phases beyond +-180
    deg, where an unwrapped plain residual and a wrapped one differ by 2 pi.
    fouriever's model closure phase is that sum, not wrapped; our reference
    must (and does) follow it there."""
    path = tmp_path / "bright.fits"
    simulate.observe(
        path, binary_vis(1.5, DRA, DDEC), UTS, hour_angles_h=np.linspace(-3, 3, 5),
        wavelengths=np.linspace(1.5e-6, 2.4e-6, 4), dec_deg=-50.0, sigma_v2=0.01,
        sigma_cp_deg=0.5, rng=np.random.default_rng(9),
    )
    d = ours.load(path)
    ps = [[1.5, DRA, DDEC], [1.4, DRA + 0.2, DDEC - 0.1], [2.0, -DRA, DDEC]]
    for p in ps:  # the summed baseline phases leave (-pi, pi] somewhere
        vis = binary_vis(*p)
        summed = (np.angle(vis(d["u1"], d["v1"], d["wl3"])) + np.angle(vis(d["u2"], d["v2_"], d["wl3"]))
                  - np.angle(vis(d["u1"] + d["u2"], d["v1"] + d["v2_"], d["wl3"])))
        assert np.max(np.abs(summed)) > np.pi
    got = np.array(fouriever(path, ps)["chi2"])
    plain = np.array([ours.chi2(d, binary_vis(*p), correlated=True, chord=False) for p in ps])
    record("rel_fouriever_vs_plain", np.max(np.abs(got / plain - 1)))
    assert np.max(np.abs(got / plain - 1)) < 1e-12


@pytest.mark.xfail(strict=True, reason="P5: fouriever's closure-phase residual is not wrapped")
@pytest.mark.validates("fouriever", roots=["mathematics"], kind="upstream")
def test_p5_residual_wraps_across_the_phase_cut(tmp_path):
    """A near-equal binary (flux ratio 0.99, within fouriever's fitting
    range) has a closure phase of 178.4 deg. A measurement 3 deg away is
    reported as -178.6 deg. Its residual against the true model is 3 deg
    (0.04 of the chi-squared budget below), but fouriever takes the plain
    difference of 357 deg, so the true binary scores a huge chi-squared.
    virgil's residuals are 2 pi periodic and score it correctly."""
    path = tmp_path / "cut.fits"
    f = 0.99
    simulate.observe(path, binary_vis(f, DRA, DDEC), UTS, hour_angles_h=np.linspace(-3, 3, 5),
                     wavelengths=np.linspace(1.5e-6, 2.4e-6, 4), dec_deg=-50.0, sigma_v2=0.01, sigma_cp_deg=0.5)
    with fits.open(path, mode="update") as h:
        cp = h["OI_T3"].data["T3PHI"]
        row, col = np.unravel_index(np.argmax(np.abs(cp)), cp.shape)
        assert abs(cp[row, col]) > 178
        cp[row, col] = (cp[row, col] + np.sign(cp[row, col]) * 3.0 + 180) % 360 - 180  # across the cut
        h["OI_T3"].data["T3PHI"] = cp
    assert virgil_chi2(path, [[f, DRA, DDEC]])[0] < 100
    assert fouriever(path, [[f, DRA, DDEC]])["chi2"][0] < 100


@pytest.mark.validates(
    "virgil.likelihood.whitened_residuals", "virgil.oidata.OIData.cp_noise",
    roots=["fouriever", "mathematics"], kind="definition",  # D6: asserts the codes differ
)
def test_unequal_errors_use_different_generalised_inverses(files):
    """D6: with unequal errors in a group, fouriever's r^T C^+ r and virgil's
    whitened form differ (by a few per cent here); each equals its own
    definition, written independently, to rounding."""
    path = files["unequal"]
    ps = params(6)
    got_f = np.array(fouriever(path, ps)["chi2"])
    got_v = virgil_chi2(path, ps)
    d = ours.load(path)
    pinv_plain = np.array([ours.chi2(d, binary_vis(*p), correlated=True, chord=False) for p in ps])
    whitened = np.array([ours.chi2(d, binary_vis(*p), correlated=True, sine=True, whitened=True) for p in ps])
    record("rel_virgil_vs_fouriever", np.max(np.abs(got_v / got_f - 1)))
    assert np.max(np.abs(got_f / pinv_plain - 1)) < 1e-12
    assert np.max(np.abs(got_v / whitened - 1)) < 1e-12
    assert np.max(np.abs(got_v / got_f - 1)) > 0.01


@pytest.mark.validates("virgil.oidata.OIData.cp_noise", roots=["statistics"])
@pytest.mark.parametrize("which", ["one noisy baseline", "random"])
def test_whitened_form_is_better_calibrated_on_baseline_noise(tmp_path, which):
    """True closure-phase noise is the closure T b of baseline-phase noise
    b. With unequal baseline errors and the reported errors
    sqrt(diag(T S T^T)), the mean chi-squared of four telescopes' closure
    phases should be 3 (the independent combinations). On a one-snapshot,
    one-channel file with those errors, virgil's whitening
    (OIData.cp_noise.whiten) gives a mean closer to 3 than the
    pseudo-inverse of C; the exact model T S T^T is the control."""
    path = tmp_path / "one.fits"
    simulate.observe(path, binary_vis(0.0, 0.0, 0.0), UTS, hour_angles_h=[0.0], wavelengths=[2.0e-6],
                     dec_deg=-50.0, sigma_v2=0.01, sigma_cp_deg=1.0)
    d = ours.load(path)
    T = ours.triangle_matrix(d["t3_sta"])
    rng = np.random.default_rng(0)
    s_base = np.array([1, 1, 1, 1, 1, 3.0]) if which != "random" else rng.uniform(0.5, 2.0, 6)
    s_base = np.deg2rad(s_base)
    S = np.diag(s_base**2)
    sig = np.sqrt(np.diag(T @ S @ T.T))
    with fits.open(path, mode="update") as h:
        h["OI_T3"].data["T3PHIERR"] = np.rad2deg(sig).reshape(h["OI_T3"].data["T3PHIERR"].shape)
    data = OIData(str(path))
    n = 100_000
    r = (rng.normal(size=(n, s_base.size)) * s_base) @ T.T
    whiten = jax.jit(jax.vmap(lambda x: data.cp_noise.whiten(x, data.d_phi)[0]))
    virgil = float(np.mean(np.sum(np.asarray(whiten(r)) ** 2, axis=1)))
    C = sig[:, None] * (T @ T.T / 3) * sig[None, :]
    pinv = float(np.mean(np.einsum("ni,ij,nj->n", r, np.linalg.pinv(C, rcond=1e-10), r)))
    exact = float(np.mean(np.einsum("ni,ij,nj->n", r, np.linalg.pinv(T @ S @ T.T, rcond=1e-10), r)))
    record("mean_virgil", virgil)
    record("mean_pinv", pinv)
    record("mean_exact", exact)
    tol = 4 * np.sqrt(2 * 3 / n)  # 4 sigma on the mean of chi2_3
    assert abs(exact - 3) < tol
    assert abs(virgil - 3) < abs(pinv - 3)
