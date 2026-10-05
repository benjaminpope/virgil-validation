"""Grid search, significance, detection limits and fits against CANDID.

CANDID (Gallenne et al. 2015; github.com/amerand/CANDID, by A. Mérand and
A. Gallenne) is the binary-search code virgil's grid search and limits
follow. It runs in its own environment (scripts/setup_candid.sh) through
src/external_bridge/candid_bridge.py, and reads our files with its own
OIFITS reader.

Conventions (pinned below): CANDID's x is East and y North, in mas; f is
the companion flux in percent of the primary; the primary is a uniform
disk of diameter diam*. Bandwidth smearing is switched off and CANDID uses
V² and closure phases, as virgil does.

Definitions that differ (ledger D5): CANDID's closure-phase residual is
the plain difference of phases, virgil's the chord 2 sin(Δ/2) (equal to
O(Δ³)). On V²-only files the two must agree to the precision of the file.

The data are three-telescope files (one closure phase per snapshot, so
closure phases are independent and CANDID's diagonal chi-squared is the
right one).
"""

import numpy as np
import pytest

from crosscheck import chi2 as ours, simulate, sky
from evidence.plugin import record
from external_bridge import candid_bridge as cb

vm = pytest.importorskip("virgil.models")
import virgil_bridge as vb  # noqa: E402
from virgil.grid_fit import likelihood_grid  # noqa: E402
from virgil.likelihood import model_loglike, whitened_residuals  # noqa: E402
from virgil.limits import absil_limits, nsigma  # noqa: E402
from virgil.oidata import OIData  # noqa: E402

pytestmark = [
    pytest.mark.x64,
    pytest.mark.external,
    pytest.mark.skipif(not cb.available(), reason="CANDID not installed (scripts/setup_candid.sh)"),
]

UTS3 = np.array([[-9.925, -20.335], [14.887, 30.502], [103.306, 43.999]])
HA = np.linspace(-3, 3, 7)
WL = np.linspace(1.5e-6, 2.4e-6, 6)
DIAM, DRA, DDEC, FLUX = 0.8, 6.0, -4.0, 0.03
RMIN, RMAX = 2.0, 12.0


def scene_vis(flux, diam=DIAM, dra=DRA, ddec=DDEC):
    def vis(u, v, w):
        return (sky.vis_uniform_disk(u, v, w, diam) + flux * sky.vis_point(u, v, w, dra, ddec)) / (1 + flux)

    return vis


def template(diam, flux=FLUX):
    return vm.System(star=vm.UniformDisk(diam), comp=vm.PointSource(flux, 0.0, 0.0))


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """{(companion?, closure phases?): path}"""
    root = tmp_path_factory.mktemp("candid")
    out = {}
    for comp in (True, False):
        for cps in (True, False):
            path = root / f"{'c' if comp else 'n'}{'cp' if cps else 'v2'}.fits"
            simulate.observe(
                path, scene_vis(FLUX if comp else 0.0), UTS3, hour_angles_h=HA,
                wavelengths=WL, dec_deg=-50.0, sigma_v2=0.01,
                sigma_cp_deg=0.5 if cps else 0.0, rng=np.random.default_rng(8),
                closure_phases=cps,
            )
            out[comp, cps] = path
    return out


def observables(cps):
    return ["v2", "cp"] if cps else ["v2"]


def virgil_chi2(model, data):
    return float(np.sum(np.asarray(whitened_residuals(model, data)) ** 2))


def random_params(n, seed=1):
    rng = np.random.default_rng(seed)
    return [
        {"x": float(x), "y": float(y), "f": float(f), "diam*": float(d)}
        for x, y, f, d in zip(
            rng.uniform(-15, 15, n), rng.uniform(-15, 15, n), rng.uniform(0, 8, n), rng.uniform(0.3, 1.5, n)
        )
    ]


def as_virgil(p):
    return vm.System(star=vm.UniformDisk(p["diam*"]), comp=vm.PointSource(p["f"] / 100, p["x"], p["y"]))


# ------------------------------------------------------------- chi-squared


@pytest.mark.validates(
    "virgil.likelihood.model_loglike", "virgil.likelihood.whitened_residuals",
    "virgil.models.System", "virgil.models.UniformDisk", "virgil.models.PointSource",
    roots=["candid"],
)
def test_chi2_matches_candid_on_v2(files):
    """With V² alone the two chi-squareds have one definition: they must
    agree to the file's float32 precision at random binaries."""
    path = files[True, False]
    params = random_params(20) + [{"x": DRA, "y": DDEC, "f": 3.0, "diam*": DIAM}]
    res = cb.run({"task": "chi2", "path": path, "params": params, "observables": ["v2"]})
    candid = np.array(res["chi2r"]) * res["ndata"]
    data = OIData(str(path))
    ours_ = np.array([virgil_chi2(as_virgil(p), data) for p in params])
    worst = np.max(np.abs(ours_ / candid - 1))
    record("max_rel_chi2_difference", worst)
    assert res["ndata"] == ours.n_data(ours.load(path))
    assert worst < 1e-6


@pytest.mark.validates("virgil.likelihood.whitened_residuals", "external_bridge.candid_bridge", roots=["candid", "mathematics"])
def test_closure_phase_residual_definitions(files):
    """D5: CANDID's chi-squared is the plain-difference one (computed
    independently here), virgil's the chord one; far from the data they
    differ, as they should."""
    path = files[True, True]
    params = random_params(20)
    res = cb.run({"task": "chi2", "path": path, "params": params})
    candid = np.array(res["chi2r"]) * res["ndata"]
    d = ours.load(path)

    def plain(p):
        vis = scene_vis(p["f"] / 100, p["diam*"], p["x"], p["y"])
        total = np.sum(((np.abs(vis(d["u"], d["v"], d["wl"])) ** 2 - d["v2"]) / d["dv2"]) ** 2)
        t3 = vis(d["u1"], d["v1"], d["wl3"]) * vis(d["u2"], d["v2_"], d["wl3"]) * np.conj(
            vis(d["u1"] + d["u2"], d["v1"] + d["v2_"], d["wl3"])
        )
        return total + np.sum(((np.angle(t3) - d["cp"]) / d["dcp"]) ** 2)

    plain_ = np.array([plain(p) for p in params])
    data = OIData(str(path))
    chord = np.array([virgil_chi2(as_virgil(p), data) for p in params])
    record("max_rel_candid_vs_plain", np.max(np.abs(candid / plain_ - 1)))
    record("max_rel_virgil_vs_plain", np.max(np.abs(chord / plain_ - 1)))
    assert np.max(np.abs(candid / plain_ - 1)) < 1e-6
    assert np.max(np.abs(chord / plain_ - 1)) > 1e-5  # the definitions do differ
    assert np.all(chord <= plain_ * (1 + 1e-12))  # |2 sin(Δ/2)| <= |Δ|


@pytest.mark.validates("virgil.limits.nsigma", roots=["candid"])
def test_nsigma_matches_candid():
    """Both take the two-sided Gaussian equivalent of the chi-squared tail
    (CANDID with chi2.sf and chdtri, virgil with gammaincc and ndtri), up
    to 27 sigma. Beyond ~37 sigma the tail underflows: CANDID returns inf,
    virgil saturates."""
    cases = [(1.2, 50), (1.25, 300), (1.05, 2000), (1.3, 168), (1.0, 100), (1.5, 126), (3.0, 300), (5.0, 300)]
    got = np.array([float(nsigma(r, 1.0, n)) for r, n in cases])
    want = np.array(cb.run({"task": "nsigma", "cases": cases})["nsigma"])
    worst = np.max(np.abs(got / want - 1))
    record("max_rel_dsigma", worst)
    assert want.max() > 25.0
    assert worst < 1e-10


# -------------------------------------------------------------------- maps


@pytest.mark.parametrize(
    "cps",
    [
        pytest.param(False, id="v2", marks=pytest.mark.validates("virgil.grid_fit.likelihood_grid", roots=["candid"])),
        pytest.param(True, id="v2+cp", marks=pytest.mark.validates("virgil.grid_fit.likelihood_grid", roots=["candid"])),
    ],
)
def test_chi2_map_matches_candid(files, cps):
    """CANDID's chi2Map at a 3 % companion (diameter fitted by CANDID)
    against virgil's likelihood_grid on the same positions, as the ratio
    chi2(binary) / chi2(single star) that both turn into a significance.
    V²-only: one definition, float32 precision. With closure phases: the
    chord-versus-plain difference (D5), < 2e-3."""
    path = files[True, cps]
    step = 0.75
    r = cb.run({"task": "chi2map", "path": path, "step": step, "fratio": 100 * FLUX,
                "rmin": RMIN, "rmax": RMAX, "observables": observables(cps)})
    x, y, cmap = np.array(r["x"]), np.array(r["y"]), np.array(r["chi2r"])
    data = OIData(str(path))
    tmpl = template(r["diam"])
    loglike = np.asarray(likelihood_grid(data, tmpl, {"comp.dra": x, "comp.ddec": y, "comp.flux": [FLUX]}))[..., 0].T
    null = tmpl.set("comp.flux", 0.0)
    offset = -2 * float(model_loglike(null, data)) - virgil_chi2(null, data)  # normalisation
    ratio_v = (-2 * loglike - offset) / virgil_chi2(null, data)
    ratio_c = cmap / r["chi2r_ud"]
    X, Y = np.meshgrid(x, y)
    inside = (X**2 + Y**2 <= RMAX**2) & (X**2 + Y**2 >= RMIN**2)
    worst = np.max(np.abs(ratio_v / ratio_c - 1)[inside])
    record("max_rel_ratio_difference", worst)
    best_v = np.unravel_index(np.argmin(np.where(inside, ratio_v, np.inf)), X.shape)
    best_c = np.unravel_index(np.argmin(np.where(inside, ratio_c, np.inf)), X.shape)
    if cps:
        assert best_v == best_c
        assert np.hypot(X[best_v] - DRA, Y[best_v] - DDEC) < step  # x is East
    else:  # V² cannot tell (x, y) from (-x, -y): either twin
        twin = tuple(n - 1 - k for n, k in zip(X.shape, best_c))
        assert best_v in (best_c, twin)
        assert min(np.hypot(X[best_v] - s * DRA, Y[best_v] - s * DDEC) for s in (1, -1)) < step
    assert worst < (1e-6 if not cps else 2e-3)


# ------------------------------------------------------------------ limits

POSITIONS = [(-8.0, -7.0), (-3.0, 2.0), (5.0, 8.0), (11.0, -3.0), (2.5, -6.0), (-7.0, 7.0)]


@pytest.mark.parametrize(
    "cps",
    [
        pytest.param(c, id=i, marks=pytest.mark.validates(
            "virgil.limits.absil_limits", "virgil.limits.nsigma", roots=["candid"]))
        for c, i in [(False, "v2"), (True, "v2+cp")]
    ],
)
def test_absil_limits_match_candids_criterion(files, cps):
    """virgil's 3-sigma Absil limits against CANDID's own criterion (its
    chi-squared, its significance, its number of data points) solved
    exactly at each position, on companion-free data with the diameter
    fixed. Before virgil#191 the two agreed to 2e-7 (V²) and 4e-6; since,
    to virgil's documented bisection precision."""
    path = files[False, cps]
    want = np.array(cb.run({"task": "absil_exact", "path": path, "diam": DIAM, "positions": POSITIONS,
                            "observables": observables(cps)})["f3"]) / 100
    data = OIData(str(path))
    got = np.array([
        float(np.asarray(absil_limits(data, template(DIAM, 0.01), {"comp.dra": [x], "comp.ddec": [y],
                                      "comp.flux": np.logspace(-4, -1, 8)}, sigma=3.0)).ravel()[0])
        for x, y in POSITIONS
    ])
    worst = np.max(np.abs(got / want - 1))
    record("max_rel_limit_difference", worst)
    # virgil bisects a log-flux decade 14 times after bracketing (virgil#191)
    # and returns the midpoint, so its limit is within half a step,
    # 10^(1/2^15) - 1 = 7.0e-5, of the root
    assert worst < 1e-4


def _public_vs_exact(path, cps):
    r = cb.run({"task": "limits", "path": path, "step": 2.0, "diam": DIAM, "rmin": RMIN, "rmax": RMAX,
                "observables": observables(cps)})
    axis, f3 = np.array(r["x"]), np.array(r["f3s"]["Absil"]) / 100
    cells = [(i, j) for i in range(axis.size) for j in range(axis.size)
             if RMIN**2 <= axis[i] ** 2 + axis[j] ** 2 <= RMAX**2][::5]
    positions = [[float(axis[i]), float(axis[j])] for i, j in cells]
    exact = np.array(cb.run({"task": "absil_exact", "path": path, "diam": DIAM, "positions": positions,
                             "observables": observables(cps)})["f3"]) / 100
    public = np.array([f3[j, i] for i, j in cells])  # CANDID maps are [y, x]
    return public / exact - 1


@pytest.mark.validates("candid", roots=["mathematics"], kind="reference")
def test_candid_public_limits_are_within_their_interpolation(files):
    """CANDID's detectionLimit brackets the 3-sigma flux by factors of 1.4
    and interpolates linearly; against its own criterion solved exactly it
    is ~1 % low (optimistic). Pinned so a change shows."""
    rel = _public_vs_exact(files[False, True], True)
    record("mean_rel", np.mean(rel))
    record("max_abs_rel", np.max(np.abs(rel)))
    assert np.max(np.abs(rel)) < 0.03


@pytest.mark.xfail(strict=True, reason="P4: CANDID's Absil limits are ~1 % low (bracketing and linear interpolation)")
@pytest.mark.validates("candid", roots=["mathematics"], kind="upstream")
def test_p4_candid_public_limits_solve_its_criterion(files):
    rel = _public_vs_exact(files[False, True], True)
    assert np.max(np.abs(rel)) < 1e-3


# -------------------------------------------------------------------- fits


@pytest.mark.validates("virgil.fitting.fit", "virgil.inference.laplace_cov", roots=["candid"])
def test_fit_matches_candid_fitmap(files):
    """CANDID's fitMap (fits started on a grid, best kept) against virgil's
    MAP fit and Laplace errors. CANDID scales its errors by sqrt(chi2r)
    with chi2r = chi2 / (N - n_fit + 1) (like PMOIRED, ledger D4); that is
    undone here."""
    path = files[True, True]
    r = cb.run({"task": "fitmap", "path": path, "step": 3.0, "rmin": RMIN, "rmax": RMAX})
    scene = vb.resolved_star_companion(diam=DIAM, dra=DRA, ddec=DDEC, flux=FLUX)
    result, cov = vb.fit_scene(scene, OIData(str(path)))
    values = vb.flat_values(scene, result.values)
    sigma = np.sqrt(np.diag(cov))
    key = {"star.diam": ("diam*", 1.0), "comp.dra": ("x", 1.0), "comp.ddec": ("y", 1.0), "comp.flux": ("f", 0.01)}
    scale = np.sqrt(r["chi2r"])
    pulls, ratios = [], []
    for name, value, s in zip(scene.truth, values, sigma):
        k, unit = key[name]
        pulls.append((value - r["best"][k] * unit) / s)
        ratios.append(r["uncer"][k] * unit / scale / s)
    record("max_abs_dbest_over_sigma", np.max(np.abs(pulls)))
    record("max_abs_sigma_ratio_minus_1", np.max(np.abs(np.array(ratios) - 1)))
    assert np.max(np.abs(pulls)) < 0.01
    assert np.max(np.abs(np.array(ratios) - 1)) < 0.03
