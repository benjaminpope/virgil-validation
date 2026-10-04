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
  difference, virgil's the chord 2 sin(delta/2).
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
    """Equal errors: fouriever equals our plain-residual r^T C^+ r and
    virgil our chord one, to rounding; near the truth, virgil and fouriever
    agree directly (the residuals are small, so chord and plain coincide)."""
    path = files["equal"]
    ps = params()
    res = fouriever(path, ps)
    got_f = np.array(res["chi2"])
    d = ours.load(path)
    plain = np.array([ours.chi2(d, binary_vis(*p), correlated=True, chord=False) for p in ps])
    chord = np.array([ours.chi2(d, binary_vis(*p), correlated=True) for p in ps])
    got_v = virgil_chi2(path, ps)
    record("rel_fouriever_vs_plain", np.max(np.abs(got_f / plain - 1)))
    record("rel_virgil_vs_chord", np.max(np.abs(got_v / chord - 1)))
    record("rel_virgil_vs_fouriever_at_truth", abs(got_v[0] / got_f[0] - 1))
    assert np.max(np.abs(got_f / plain - 1)) < 1e-12
    assert np.max(np.abs(got_v / chord - 1)) < 1e-12
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


def whitened_form(d, vis):
    """virgil's documented generalised inverse, written independently:
    r^T D^-1/2 R^+ D^-1/2 r per snapshot and channel, chord residuals."""
    t3 = (vis(d["u1"], d["v1"], d["wl3"]) * vis(d["u2"], d["v2_"], d["wl3"])
          * np.conj(vis(d["u1"] + d["u2"], d["v1"] + d["v2_"], d["wl3"])))
    r = 2 * np.sin(np.angle(np.exp(1j * (np.angle(t3) - d["cp"]))) / 2)
    total = np.sum(((np.abs(vis(d["u"], d["v"], d["wl"])) ** 2 - d["v2"]) / d["dv2"]) ** 2)
    for mjd in np.unique(d["t3_mjd"]):
        rows = np.flatnonzero(d["t3_mjd"] == mjd)
        T = ours.triangle_matrix(d["t3_sta"][rows])
        Rp = np.linalg.pinv(T @ T.T / 3.0, rcond=1e-10)
        for k in range(r.shape[1]):
            x = r[rows, k] / d["dcp"][rows, k]
            total += x @ Rp @ x
    return float(total)


@pytest.mark.validates(
    "virgil.likelihood.whitened_residuals", "virgil.oidata.OIData.cp_noise",
    roots=["fouriever", "mathematics"],
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
    whitened = np.array([whitened_form(d, binary_vis(*p)) for p in ps])
    record("rel_virgil_vs_fouriever", np.max(np.abs(got_v / got_f - 1)))
    assert np.max(np.abs(got_f / pinv_plain - 1)) < 1e-12
    assert np.max(np.abs(got_v / whitened - 1)) < 1e-12
    assert np.max(np.abs(got_v / got_f - 1)) > 0.01


@pytest.mark.validates("virgil.oidata.OIData.cp_noise", roots=["statistics"], kind="reference")
def test_whitened_form_is_better_calibrated_on_baseline_noise():
    """True closure-phase noise is the closure T b of baseline-phase noise
    b. With unequal baseline errors and the reported errors
    sqrt(diag(T S T^T)), the mean chi-squared of four telescopes' closure
    phases should be 3 (independent combinations). Virgil's whitened form
    is closer to it than the pseudo-inverse of C (the exact T S T^T is the
    control)."""
    T = ours.triangle_matrix([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    R = T @ T.T / 3
    rng = np.random.default_rng(0)
    n = 200_000
    for s_base in (np.array([1, 1, 1, 1, 1, 3.0]), rng.uniform(0.5, 2.0, 6)):
        S = np.diag(s_base**2)
        sig = np.sqrt(np.diag(T @ S @ T.T))
        C = sig[:, None] * R * sig[None, :]
        r = (rng.normal(size=(n, 6)) * s_base) @ T.T
        means = {}
        for name, G in {
            "pinv": np.linalg.pinv(C, rcond=1e-10),
            "whitened": np.diag(1 / sig) @ np.linalg.pinv(R, rcond=1e-10) @ np.diag(1 / sig),
            "exact": np.linalg.pinv(T @ S @ T.T, rcond=1e-10),
        }.items():
            means[name] = float(np.mean(np.einsum("ni,ij,nj->n", r, G, r)))
        for name, value in means.items():
            record(f"mean_{name}_{s_base.max():.1f}", value)
        tol = 4 * np.sqrt(2 * 3 / n)  # 4 sigma on the mean of chi2_3
        assert abs(means["exact"] - 3) < tol
        assert abs(means["whitened"] - 3) < abs(means["pinv"] - 3)
