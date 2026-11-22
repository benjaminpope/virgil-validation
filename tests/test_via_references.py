"""First-principles reference checks for ``trust/graph.yml`` ``via`` modules.

Every ``layer: references`` module exercised through ``via`` on a trust check needs
its own ``kind="reference"`` validation against mathematical, statistical, or
standards invariants so that ``scripts/trust.py`` upgrades dependent ``ok-via``
nodes rather than holding them at ``relies-unverified``.
"""

from __future__ import annotations

import hashlib
import subprocess

import numpy as np
import pytest
from scipy.special import j1
from scipy.stats import chi2 as chi2_dist
from scipy.stats import norm

from evidence.plugin import record
from external_bridge import _subprocess as ext_sp

# The crosscheck reference modules and external_bridge.npz_compare are imported inside the test that checks
# each, not here: evidence.plugin stamps every module-level import onto every check in a file, which made
# each of them look checked through all the others (a false cycle, scripts/trust.py).


# ------------------------------------------------ 1. crosscheck.radio_innovations


@pytest.mark.validates(
    "crosscheck.radio_innovations",
    roots=["mathematics"],
    kind="reference",
    headline="gn_vs_fd_rel_err",
)
def test_radio_innovations_softmax_gn_and_entropy_metric():
    """``softmax_image`` is shift-invariant with unit sum, ``gn_diagonal`` matches a central
    finite-difference Jacobian ``J^T J`` of the whitened DFT residuals, and
    ``entropy_metric_diagonal`` equals the Skilling-Bryan pullback ``w * b * (1 - b)``."""
    from crosscheck import radio_innovations as ri
    rng = np.random.default_rng(101)
    eta = rng.normal(0.0, 1.2, (5, 5))
    b = ri.softmax_image(eta)
    b_shifted = ri.softmax_image(eta + 42.5)
    shift_err = float(np.max(np.abs(b - b_shifted)))
    assert abs(float(b.sum()) - 1.0) < 1e-15
    assert shift_err < 1e-15

    n_vis = 40
    u = rng.normal(0.0, 2.5e5, n_vis)
    v = rng.normal(0.0, 2.5e5, n_vis)
    weight = rng.uniform(10.0, 100.0, n_vis)
    pixel_mas, flux = 8.0, 0.65
    east, north = ri.pixel_positions(5, pixel_mas)
    dft = ri.dft_matrix(u, v, east, north)

    def whitened_model(flat_eta):
        bb = ri.softmax_image(flat_eta.reshape(5, 5)).ravel()
        m = flux * (dft @ bb)
        sw = np.sqrt(weight)
        return np.concatenate([sw * m.real, sw * m.imag])

    h = 1e-6
    x0 = eta.ravel()
    cols = []
    for k in range(x0.size):
        step = np.zeros_like(x0)
        step[k] = h
        cols.append((whitened_model(x0 + step) - whitened_model(x0 - step)) / (2.0 * h))
    jac = np.column_stack(cols)
    fd_diag = (jac**2).sum(axis=0).reshape(5, 5)
    gn = ri.gn_diagonal(u, v, weight, eta, pixel_mas, flux)
    gn_rel_err = float(np.max(np.abs(gn / fd_diag - 1.0)))

    w_ent = 3.7
    ent = ri.entropy_metric_diagonal(w_ent, eta)
    ent_closed = w_ent * b * (1.0 - b)
    ent_rel_err = float(np.max(np.abs(ent / ent_closed - 1.0)))

    record("gn_vs_fd_rel_err", gn_rel_err)
    record("entropy_closed_form_rel_err", ent_rel_err)
    assert gn_rel_err < 1e-6
    assert ent_rel_err < 1e-13


@pytest.mark.validates(
    "crosscheck.radio_innovations",
    roots=["mathematics"],
    kind="reference",
)
def test_radio_innovations_direct_chi2_by_ddid_against_two_point_interference():
    """``direct_chi2_by_ddid`` reproduces the analytical two-point interference visibility
    with per-observation amplitude gains and sky-plane phase ramps."""
    from crosscheck import radio_innovations as ri
    brightness = np.zeros((3, 3))
    # Row 0 North, Col 0 East; centre is (1, 1); (1, 0) is +pixel East, (0, 1) is +pixel North
    brightness[1, 0] = 0.75
    brightness[0, 1] = 0.25
    pixel_mas = 10.0
    u = np.array([1.0e5, -2.0e5, 1.5e5, -0.5e5])
    v = np.array([-0.8e5, 1.2e5, 0.9e5, 2.1e5])
    ddid = np.array([0, 0, 1, 1])
    obs = np.array([0, 1, 1, 2])
    flux = 1.4
    gain = np.array([1.0, 1.12, 0.88])
    offset = np.array([[0.0, 0.0], [3.5, -2.0], [-1.5, 4.0]])
    weight = np.array([2.0, 5.0, 3.0, 7.0])
    vis_data = np.array([0.4 - 0.1j, -0.2 + 0.3j, 0.1 + 0.5j, 0.6 - 0.4j])

    scene = (
        0.75 * np.exp(-2j * np.pi * ri.MAS * (u * pixel_mas))
        + 0.25 * np.exp(-2j * np.pi * ri.MAS * (v * pixel_mas))
    )
    ramp = np.exp(-2j * np.pi * ri.MAS * (u * offset[obs, 0] + v * offset[obs, 1]))
    exact_model = flux * gain[obs] * ramp * scene
    exact_resid = weight * np.abs(vis_data - exact_model) ** 2
    want = np.array([exact_resid[:2].sum(), exact_resid[2:].sum()])

    got = ri.direct_chi2_by_ddid(u, v, vis_data, weight, ddid, obs, brightness, pixel_mas, flux, gain, offset)
    rel_err = float(np.max(np.abs(got / want - 1.0)))
    record("two_point_chi2_rel_err", rel_err)
    assert rel_err < 1e-14


@pytest.mark.validates(
    "crosscheck.radio_innovations.DeltaProblem",
    roots=["mathematics"],
    kind="reference",
    headline="weight_homogeneity_rel_err",
)
def test_delta_problem_zero_step_and_weight_homogeneity():
    """At ``z == z_ref``, ``DeltaProblem`` gives identically zero difference and scale;
    for ``z != z_ref``, ``delta_chi2_float32`` scales linearly with the visibility weights."""
    from crosscheck import radio_innovations as ri
    rng = np.random.default_rng(12)
    east, north = ri.pixel_positions(4, 6.0)
    u, v = rng.normal(0.0, 2e5, 30), rng.normal(0.0, 2e5, 30)
    dft = ri.dft_matrix(u, v, east, north)
    z_ref = rng.normal(-2.0, 0.5, 16)
    vis = dft @ np.exp(z_ref) + 0.01 * (rng.normal(size=30) + 1j * rng.normal(size=30))
    w = rng.uniform(5.0, 25.0, 30)
    p1 = ri.DeltaProblem(dft, z_ref, vis, w)
    p2 = ri.DeltaProblem(dft, z_ref, vis, 3.5 * w)
    assert p1.exact_delta(z_ref) == 0.0
    assert p1.delta_chi2_float32(z_ref) == 0.0
    assert p1.absolute_scale(z_ref) == 0.0

    z = z_ref + 1e-3 * rng.normal(size=16)
    hom_err = abs(p2.delta_chi2_float32(z) / (3.5 * p1.delta_chi2_float32(z)) - 1.0)
    record("weight_homogeneity_rel_err", float(hom_err))
    record("zero_step_abs_delta", abs(p1.delta_chi2_float32(z_ref)))
    assert hom_err < 1e-14


# ------------------------------------------------ 2. crosscheck.preregistration


@pytest.mark.validates(
    "crosscheck.preregistration",
    roots=["standards"],
    kind="reference",
)
def test_preregistration_criteria_commit_and_aggregator_rules(tmp_path, monkeypatch):
    """``criteria_commit`` returns the exact last commit SHA of a clean tracked file, ignores
    ``CRITERIA_COMMIT`` inside a git repo, rejects dirty or untracked files with ``CriteriaError``,
    uses ``CRITERIA_COMMIT`` only outside git (archive snapshots), and ``require_same_commit``
    rejects mismatched or missing SHAs."""
    from crosscheck import preregistration as prereg
    repo = tmp_path / "repo"
    repo.mkdir()
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    git("init")
    git("config", "user.email", "test@example.org")
    git("config", "user.name", "Test")
    crit = repo / "design" / "gate.yml"
    crit.parent.mkdir()
    crit.write_text("threshold: 1.0\n")
    git("add", "design/gate.yml")
    git("commit", "-m", "add gate")
    want_sha = git("log", "-n1", "--format=%H", "--", "design/gate.yml").stdout.strip()

    monkeypatch.setenv("CRITERIA_COMMIT", "deadbeef" * 5)
    got_sha = prereg.criteria_commit("design/gate.yml", root=repo)
    assert got_sha == want_sha

    # Dirty tracked file is refused
    crit.write_text("threshold: 2.0\n")
    with pytest.raises(prereg.CriteriaError, match="uncommitted"):
        prereg.criteria_commit("design/gate.yml", root=repo)
    git("checkout", "--", "design/gate.yml")

    # Untracked file is refused
    untracked = repo / "design" / "untracked.yml"
    untracked.write_text("threshold: 1.0\n")
    with pytest.raises(prereg.CriteriaError, match="not tracked"):
        prereg.criteria_commit("design/untracked.yml", root=repo)

    # Archive directory outside git requires CRITERIA_COMMIT
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "gate.yml").write_text("threshold: 1.0\n")
    monkeypatch.delenv("CRITERIA_COMMIT", raising=False)
    with pytest.raises(prereg.CriteriaError, match="CRITERIA_COMMIT"):
        prereg.criteria_commit("gate.yml", root=archive)
    monkeypatch.setenv("CRITERIA_COMMIT", want_sha)
    assert prereg.criteria_commit("gate.yml", root=archive) == want_sha

    # Aggregator enforces single non-null SHA
    assert prereg.require_same_commit([{"criteria_commit": want_sha}, {"criteria_commit": want_sha}]) == want_sha
    with pytest.raises(prereg.CriteriaError):
        prereg.require_same_commit([{"criteria_commit": want_sha}, {"criteria_commit": "0" * 40}])
    with pytest.raises(prereg.CriteriaError):
        prereg.require_same_commit([{"criteria_commit": want_sha}, {}])
    record("verified_sha_len", float(len(got_sha)))


# ------------------------------------------------ 3. external_bridge.npz_compare


@pytest.mark.validates(
    "external_bridge.npz_compare",
    roots=["mathematics"],
    kind="reference",
    headline="vis_max_abs",
)
def test_npz_compare_matches_channels_by_frequency_rank_and_handles_zero_u():
    """``compare_ddid`` aligns channels by ascending frequency rank regardless of whether
    either input stores channels in increasing or decreasing frequency order, avoids 0/0
    at ``u = 0`` via ``U_FLOOR = 1.0``, flags conjugation mismatches, and refuses missing ``freq``."""
    from external_bridge import npz_compare as npzc
    freqs_asc = np.array([230.0e9, 230.5e9, 231.0e9])
    rows = []
    for t in (100.0, 200.0):
        for a1, a2 in ((0, 1), (1, 2)):
            for ch, f in enumerate(freqs_asc):
                u_val = 0.0 if (a1 == 0 and ch == 1) else (1e4 * (ch + 1) * (a1 + 1))
                v_val = -2e4 * (a2 + 1) + 50.0 * ch
                vis_val = (0.1 * (ch + 1) + 0.02 * a1) - 1j * (0.05 * (a2 + 1) + 1e-4 * t)
                w_val = 100.0 + 10.0 * ch + a2
                rows.append((t, a1, a2, f, u_val, v_val, vis_val, w_val))

    def pack_rows(order):
        sub = [rows[i] for i in order]
        return dict(
            time=np.array([r[0] for r in sub]),
            ant1=np.array([r[1] for r in sub]),
            ant2=np.array([r[2] for r in sub]),
            freq=np.array([r[3] for r in sub]),
            u=np.array([r[4] for r in sub]),
            v=np.array([r[5] for r in sub]),
            vis=np.array([r[6] for r in sub]),
            weight=np.array([r[7] for r in sub]),
        )

    rng = np.random.default_rng(7)
    # Reverse channel order and permute rows in 'a', keep ascending in 'b'
    a = pack_rows(rng.permutation(len(rows)))
    b = pack_rows(np.arange(len(rows)))
    res = npzc.compare_ddid(a, b)
    assert res["matching"] == "keys"
    assert res["key_mismatch"] == 0
    assert res["pass"] is True
    assert np.isfinite(res["u_max_rel"]) and res["u_max_rel"] == 0.0

    # Conjugating 'a' fails pass while reporting zero vis_conj_max_abs
    a_conj = dict(a, vis=np.conj(a["vis"]))
    res_conj = npzc.compare_ddid(a_conj, b)
    assert res_conj["pass"] is False
    assert res_conj["vis_conj_max_abs"] < 1e-15

    # Missing freq raises ValueError rather than guessing channel direction
    bad = {k: v for k, v in a.items() if k != "freq"}
    with pytest.raises(ValueError, match="freq"):
        npzc.compare_ddid(bad, b)

    record("vis_max_abs", float(res["vis_max_abs"]))
    record("conj_signature_abs", float(res_conj["vis_conj_max_abs"]))


# ------------------------------------------------ 4. crosscheck.informed_starts


def _sky_gaussian_vis_quadrature(u, v, fwhm_mas, ratio, pa_deg, n_grid=201):
    """Direct 2D sky-plane numerical quadrature of a unit-flux elliptical Gaussian.

    Sky coordinates ``dra`` (East positive) and ``ddec`` (North positive) are in mas,
    converted independently via ``pi / (180 * 3600 * 1000)``, with major axis along
    ``dra * sin(pa) + ddec * cos(pa)``. Shares no formula or constant with ``informed_starts``.
    """
    mas_to_rad = np.pi / (180.0 * 3600.0 * 1000.0)
    sigma_maj = fwhm_mas / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sigma_min = ratio * sigma_maj
    half = 6.0 * fwhm_mas
    axis = np.linspace(-half, half, n_grid)
    step = float(axis[1] - axis[0])
    dra, ddec = np.meshgrid(axis, axis, indexing="xy")
    pa = np.deg2rad(pa_deg)
    along = dra * np.sin(pa) + ddec * np.cos(pa)
    across = dra * np.cos(pa) - ddec * np.sin(pa)
    density = (
        np.exp(-0.5 * ((along / sigma_maj) ** 2 + (across / sigma_min) ** 2))
        / (2.0 * np.pi * sigma_maj * sigma_min)
        * step**2
    )
    dra_rad = (dra * mas_to_rad).ravel()
    ddec_rad = (ddec * mas_to_rad).ravel()
    w_sky = density.ravel()
    # Centro-symmetric source has real visibility int I(dra, ddec) cos(2 pi (u dra + v ddec))
    phase = 2.0 * np.pi * (np.outer(u, dra_rad) + np.outer(v, ddec_rad))
    return np.cos(phase) @ w_sky


@pytest.mark.validates(
    "crosscheck.informed_starts",
    roots=["mathematics"],
    kind="reference",
    headline="moment_fit_fwhm_rel_err",
)
def test_informed_starts_moments_recover_analytic_elliptical_gaussian():
    """Against direct 2D sky-plane numerical quadrature of an elliptical Gaussian (East positive,
    North positive, major axis at ``pa`` East of North), ``gaussian_visibility`` and
    ``moment_model_v2`` match the 2D Fourier integral, ``second_moments`` recovers the exact
    covariance eigenvalues and PA from ``-ln V^2``, and ``moment_fit`` recovers
    ``(compact, extended, fwhm, ratio, pa)`` of a point-plus-Gaussian sky."""
    from crosscheck import informed_starts as ist
    rng = np.random.default_rng(19)
    u = rng.uniform(-3.0e7, 3.0e7, 80)
    v = rng.uniform(-3.0e7, 3.0e7, 80)
    fwhm_true, ratio_true, pa_true = 6.5, 0.6, 37.0
    vis_quad = _sky_gaussian_vis_quadrature(u, v, fwhm_true, ratio_true, pa_true)
    vis_model_err = float(np.max(np.abs(ist.gaussian_visibility(u, v, fwhm_true, ratio_true, pa_true) - vis_quad)))
    assert vis_model_err < 1e-12

    v2_gauss = vis_quad**2
    err = np.full_like(v2_gauss, 1e-3)
    sm = ist.second_moments(u, v, v2_gauss, err, v2_min=0.2)
    sigma_maj_true = fwhm_true / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sm_err = max(
        abs(sm["sigma_major"] / sigma_maj_true - 1.0),
        abs(sm["ratio"] / ratio_true - 1.0),
    )
    assert not sm["isotropic"]
    assert sm_err < 1e-9
    assert abs((sm["pa"] - pa_true + 90.0) % 180.0 - 90.0) < 1e-7

    # Point + elliptical Gaussian + over-resolved flux from independent 2D sky quadrature
    q = np.geomspace(2.0e6, 1.5e8, 120)
    phi = rng.uniform(0.0, 2.0 * np.pi, 120)
    u2, v2_coord = q * np.cos(phi), q * np.sin(phi)
    fp_true, fg_true = 0.25, 0.55
    vis_quad2 = _sky_gaussian_vis_quadrature(u2, v2_coord, fwhm_true, ratio_true, pa_true)
    v2_mix = (fp_true + fg_true * vis_quad2) ** 2
    assert np.max(np.abs(ist.moment_model_v2(u2, v2_coord, fp_true, fg_true, fwhm_true, ratio_true, pa_true) - v2_mix)) < 1e-12
    mf = ist.moment_fit(u2, v2_coord, v2_mix, np.full(120, 5e-3), theta_range=(1.5, 25.0))
    assert mf is not None
    fwhm_rel_err = abs(mf["fwhm"] / fwhm_true - 1.0)
    flux_abs_err = max(abs(mf["compact"] - fp_true), abs(mf["extended"] - fg_true))

    record("moment_fit_fwhm_rel_err", float(fwhm_rel_err))
    record("gaussian_vis_quad_abs_err", vis_model_err)
    record("second_moments_rel_err", float(sm_err))
    record("moment_fit_flux_abs_err", float(flux_abs_err))
    assert fwhm_rel_err < 2e-2
    assert abs(mf["ratio"] - ratio_true) < 2e-2
    assert abs((mf["pa"] - pa_true + 90.0) % 180.0 - 90.0) < 2.0
    assert flux_abs_err < 2e-2


@pytest.mark.validates(
    "crosscheck.informed_starts",
    roots=["mathematics"],
    kind="control",
    headline="mirrored_pa_max_v2_diff",
)
def test_informed_starts_control_rejects_mirrored_position_angle():
    """Negative control: mirroring the sky position angle (``180 - pa``) flips the sign of the
    ``u * v`` cross-term in Fourier space and is strongly separated from ``pa`` by both
    ``moment_model_v2`` and ``moment_fit``."""
    from crosscheck import informed_starts as ist
    rng = np.random.default_rng(29)
    q = np.geomspace(4.0e6, 1.2e8, 100)
    phi = rng.uniform(0.0, 2.0 * np.pi, 100)
    u, v = q * np.cos(phi), q * np.sin(phi)
    fwhm_true, ratio_true, pa_true = 7.0, 0.5, 37.0
    pa_mirror = 180.0 - pa_true

    v2_mirror = (0.2 + 0.65 * _sky_gaussian_vis_quadrature(u, v, fwhm_true, ratio_true, pa_mirror)) ** 2
    v2_unmirrored = ist.moment_model_v2(u, v, 0.2, 0.65, fwhm_true, ratio_true, pa_true)
    max_diff = float(np.max(np.abs(v2_mirror - v2_unmirrored)))
    mf = ist.moment_fit(u, v, v2_mirror, np.full(100, 5e-3), theta_range=(1.5, 25.0))
    assert mf is not None
    pa_err_vs_true = abs((mf["pa"] - pa_true + 90.0) % 180.0 - 90.0)
    pa_err_vs_mirror = abs((mf["pa"] - pa_mirror + 90.0) % 180.0 - 90.0)

    record("mirrored_pa_max_v2_diff", max_diff)
    record("mirrored_pa_fit_sep_deg", float(pa_err_vs_true))
    assert max_diff > 0.1
    assert pa_err_vs_true > 60.0
    assert pa_err_vs_mirror < 2.0


@pytest.mark.validates(
    "crosscheck.informed_starts",
    roots=["mathematics"],
    kind="reference",
    headline="coarsened_npix",
)
def test_informed_starts_scales_tiled_peaks_and_diagnostic_checks():
    """``scales_of`` enforces Nyquist sampling and ``MAX_PIXELS`` coarsening, ``tiled_map`` +
    ``map_peaks`` separate local maxima while respecting origin exclusion, and the section-5
    diagnostic predicates enforce bound, nesting, agreement, completeness, and collapse thresholds."""
    from crosscheck import informed_starts as ist
    b_max, b_min = 1.0e8, 2.0e6
    sc = ist.scales_of(b_max, b_min, moments={"fwhm": 50.0, "short_v2": 0.5, "at_upper_edge": False, "extended": 0.6})
    want_res = 1.0 / (2.0 * b_max) / ist.MAS
    assert sc.resolution == pytest.approx(want_res, rel=1e-12)
    assert sc.extent == pytest.approx(200.0, rel=1e-12)
    assert sc.npix == ist.MAX_PIXELS and sc.npix_wanted > ist.MAX_PIXELS

    # Tiled map with peak at (0, 0) (excluded) and two separated peaks at (+6, -4) and (-8, +6)
    def surface_fn(dra, ddec):
        x, y = np.meshgrid(dra, ddec, indexing="ij")
        return (
            10.0 * np.exp(-((x**2 + y**2) / 1.5**2))
            + 5.0 * np.exp(-(((x - 6.0) ** 2 + (y + 4.0) ** 2) / 1.5**2))
            + 3.0 * np.exp(-(((x + 8.0) ** 2 + (y - 6.0) ** 2) / 1.5**2))
        )

    axis, surf, step = ist.tiled_map(surface_fn, half=12.0, step=0.5, tile=16)
    peaks = ist.map_peaks(axis, surf, k=2, separation=4.0, exclude=3.0)
    assert len(peaks) == 2
    assert np.hypot(peaks[0][0] - 6.0, peaks[0][1] + 4.0) <= step
    assert np.hypot(peaks[1][0] + 8.0, peaks[1][1] - 6.0) <= step

    # Diagnostic checks
    ok_b, _ = ist.no_bounds({"r": 5.0, "pa": 0.001}, {"r": (1.0, 10.0, False, "real"), "pa": (0.0, 180.0, False, "angle")})
    bad_b, hits = ist.no_bounds({"r": 1.02}, {"r": (1.0, 10.0, False, "real")})
    assert ok_b and not bad_b and hits[0]["end"] == "low"

    ok_n, _ = ist.nested([{"rung": "point", "minus2lnl": 120.0, "contains": []}, {"rung": "binary", "minus2lnl": 95.0, "contains": ["point"]}], n=200)
    bad_n, _ = ist.nested([{"rung": "point", "minus2lnl": 95.0, "contains": []}, {"rung": "binary", "minus2lnl": 105.0, "contains": ["point"]}], n=200)
    assert ok_n and not bad_n
    assert ist.starts_agree([100.0, 100.4, 130.0], n=200)[0] is True
    assert ist.starts_agree([100.0, 110.0, 130.0], n=200)[0] is False
    assert ist.family_complete(0.12)[0] is True and ist.family_complete(0.35)[0] is False and ist.family_complete(None)[0] is None
    assert ist.no_collapse([{"kind": "companion", "dra": 1.0, "ddec": 0.5}], resolution=1.0)[0] is False
    assert ist.no_collapse([{"kind": "companion", "dra": 2.5, "ddec": 0.0}], resolution=1.0)[0] is True

    record("coarsened_npix", float(sc.npix))
    record("peak0_pos_err_mas", float(np.hypot(peaks[0][0] - 6.0, peaks[0][1] + 4.0)))


# ------------------------------------------------ 5. crosscheck.orbit_population


@pytest.mark.validates(
    "crosscheck.orbit_population",
    roots=["mathematics", "standards"],
    kind="reference",
    headline="ellipse_quadrature_abs_err",
)
def test_orbit_population_stream_isolation_elliptical_disk_and_band_laws(tmp_path):
    """``system_streams`` isolates parameter draws from noise draws, ``vis_elliptical_disk``
    matches 2D numerical quadrature over the sky ellipse and the circular Airy formula,
    ``band_of`` and ``p_cut`` satisfy their analytical definitions, and ``write_system``
    produces readable OIFITS files."""
    from crosscheck import orbit_population as opop
    from astropy.io import fits

    rng1, noise1 = opop.system_streams("A", 3)
    _ = noise1.standard_normal(5000)  # consuming noise draws never perturbs rng2
    rng2, _ = opop.system_streams("A", 3)
    assert np.array_equal(rng1.uniform(size=16), rng2.uniform(size=16))

    # Circular limit: ratio = 1 equals 2 J1(pi * major * q) / (pi * major * q)
    u = np.array([0.0, 20.0, -45.0, 80.0])
    v = np.array([0.0, -30.0, 50.0, 35.0])
    wavel = 2.2e-6
    major, ratio, pa = 4.0, 0.55, 32.0
    circ = opop.vis_elliptical_disk(u, v, wavel, major, 1.0, pa)
    q_rad = np.hypot(u, v) / wavel * (major / opop.MAS)
    airy = np.where(q_rad > 0, 2.0 * j1(np.pi * q_rad) / np.where(q_rad > 0, np.pi * q_rad, 1.0), 1.0)
    circ_err = float(np.max(np.abs(circ - airy)))

    # 2D Riemann sum over the rotated uniform ellipse in sky offsets (dra East, ddec North)
    grid_1d = np.linspace(-major, major, 1201)
    dra, ddec = np.meshgrid(grid_1d, grid_1d, indexing="xy")
    p = np.deg2rad(pa)
    along = dra * np.sin(p) + ddec * np.cos(p)
    across = dra * np.cos(p) - ddec * np.sin(p)
    inside = (along / (0.5 * major)) ** 2 + (across / (0.5 * major * ratio)) ** 2 <= 1.0
    dra_in, ddec_in = dra[inside] / opop.MAS, ddec[inside] / opop.MAS
    riemann = np.array([
        np.mean(np.exp(-2j * np.pi * ((uk / wavel) * dra_in + (vk / wavel) * ddec_in)))
        for uk, vk in zip(u, v)
    ])
    got_ell = opop.vis_elliptical_disk(u, v, wavel, major, ratio, pa)
    quad_err = float(np.max(np.abs(got_ell - riemann)))

    # Alias band edges and p_cut
    assert opop.band_of(25.0, 100.0) == 4
    assert opop.band_of(250.0, 100.0) == 0
    assert opop.p_cut([60000.0, 60003.0, 60010.0], a_mas=2.0, night_days=0.1, res_mas=1.0) == pytest.approx(
        max(6.0, 2.0 * np.pi * 2.0 * 0.1 / (0.5 * 1.0)), rel=1e-12
    )

    # End-to-end draw and OIFITS write
    sys_a = opop.draw_system("A", 0)
    paths = opop.write_system(sys_a, tmp_path / "sys_a0")
    assert len(paths) == len(sys_a["nights"])
    with fits.open(paths[0]) as hdul:
        extnames = {h.name for h in hdul}
        assert {"OI_WAVELENGTH", "OI_ARRAY", "OI_VIS2", "OI_T3"} <= extnames

    record("ellipse_quadrature_abs_err", quad_err)
    record("circular_airy_abs_err", circ_err)
    assert circ_err < 1e-14
    assert quad_err < 3e-4


# ------------------------------------------------ 6. crosscheck.references


@pytest.mark.validates(
    "crosscheck.references",
    roots=["mathematics", "standards"],
    kind="reference",
    headline="committed_reference_errors",
)
def test_references_secondary_omega_time_systems_and_catalog_completeness():
    """``secondary_omega`` shifts primary omega by +180 deg mod 360 and refuses missing
    conventions, ``missing_omega_of`` validates all committed ``oidb/references/*.json``
    files, and ``trackb_reference`` converts JD, BJD-2400000, and HJD-2445000 to MJD."""
    from crosscheck import references as refs
    assert refs.secondary_omega(25.0, "primary") == pytest.approx(205.0)
    assert refs.secondary_omega(250.0, "primary") == pytest.approx(70.0)
    assert refs.secondary_omega(250.0, "secondary") == pytest.approx(250.0)
    with pytest.raises(ValueError, match="omega_of"):
        refs.secondary_omega(25.0, None)

    assert refs.missing_omega_of({"orbit": {"omega_deg": 10.0}}) != []
    assert refs.missing_omega_of({"orbit": {"omega_primary_deg": 10.0, "omega_of": "secondary"}}) != []
    assert refs.missing_omega_of({"orbit": {"omega_primary_deg": 10.0, "omega_of": "primary"}}) == []

    all_refs = refs.load_references()
    assert len(all_refs) > 0
    bad_committed = sum(len(refs.missing_omega_of(doc)) for doc in all_refs.values())
    assert bad_committed == 0

    def mk_sys(t0_val, unit, omega_of="primary"):
        return {
            "reference_orbit": {
                "P_day": [10.0],
                "T0": [t0_val, 0.1, unit],
                "e": [0.3],
                "omega_deg": [40.0],
                "Omega_deg": [110.0],
                "i_deg": [55.0],
                "a_mas": [5.0],
                "omega_of": omega_of,
            }
        }

    r_jd = refs.trackb_reference(mk_sys(2459000.5, "JD"))
    r_bjd = refs.trackb_reference(mk_sys(59000.5, "BJD-2400000"))
    r_hjd = refs.trackb_reference(mk_sys(14000.5, "HJD-2445000"))
    assert r_jd["t_peri"] == pytest.approx(59000.0)
    assert r_jd["omega"] == pytest.approx(220.0)
    assert r_bjd["t_peri"] == pytest.approx(59000.0)
    assert r_hjd["t_peri"] == pytest.approx(59000.0)

    record("committed_reference_errors", float(bad_committed))
    record("n_committed_reference_files", float(len(all_refs)))


# ------------------------------------------------ 7. crosscheck.skyplane


@pytest.mark.validates(
    "crosscheck.skyplane",
    roots=["mathematics"],
    kind="reference",
    headline="twin_rms_mas",
)
def test_skyplane_twin_invariance_point_reflection_and_verdict_rules():
    """The ``(Omega + 180, omega + 180)`` twin produces identical sky-plane positions
    (``SAME-SKY-ORBIT`` with samples, ``INCOMPLETE`` without), a single ``omega + 180``
    point reflection gives ``UNEXPLAINED-FLIP`` unless ``flipped`` or ``labels_free`` is set,
    and ``good_fit`` separates ``REFERENCE-DISAGREES`` from ``DIFFERENT`` via ``chi2.sf``."""
    from crosscheck import skyplane as sp_sky
    mjd = np.linspace(60000.0, 60045.0, 12)
    ref = dict(P=18.5, t_peri=60003.2, ecc=0.35, inc=52.0, omega=40.0, Omega=75.0, a_mas=8.0)
    meta = dict(mjd=mjd, reference=ref)
    t_ref = 60000.0

    # Exact (Omega + 180, omega + 180) twin
    twin_best = dict(period=18.5, dt_peri=3.2, ecc=0.35, inc=52.0, omega=220.0, Omega=255.0, a_mas=8.0)
    rng = np.random.default_rng(23)
    n_samp = 64
    samples = {
        "period": np.full(n_samp, 18.5) + rng.normal(0.0, 1e-3, n_samp),
        "dt_peri": np.full(n_samp, 3.2) + rng.normal(0.0, 1e-3, n_samp),
        "ecc": np.full(n_samp, 0.35),
        "inc": np.full(n_samp, 52.0),
        "omega": np.full(n_samp, 220.0),
        "Omega": np.full(n_samp, 255.0),
        "a_mas": np.full(n_samp, 8.0) + rng.normal(0.0, 1e-2, n_samp),
    }
    sc_twin = sp_sky.score(meta, twin_best, t_ref=t_ref, samples=samples)
    sc_no_samp = sp_sky.score(meta, twin_best, t_ref=t_ref, samples=None)
    assert sc_twin["verdict"] == "SAME-SKY-ORBIT"
    assert sc_twin["mirror"] == "none"
    assert sc_no_samp["verdict"] == "INCOMPLETE"

    # Point reflection: omega + 180 alone negates every sky offset
    refl_best = dict(period=18.5, dt_peri=3.2, ecc=0.35, inc=52.0, omega=220.0, Omega=75.0, a_mas=8.0)
    refl_samples = dict(samples, Omega=np.full(n_samp, 75.0))
    sc_unexpl = sp_sky.score(meta, refl_best, t_ref=t_ref, samples=refl_samples, flipped=False, labels_free=False)
    sc_flipped = sp_sky.score(meta, refl_best, t_ref=t_ref, samples=refl_samples, flipped=True)
    assert sc_unexpl["verdict"] == "UNEXPLAINED-FLIP"
    assert sc_flipped["verdict"] == "SAME-SKY-ORBIT"

    # Far orbit: good chi2/N -> REFERENCE-DISAGREES, bad chi2/N -> DIFFERENT
    far_best = dict(period=31.0, dt_peri=1.0, ecc=0.1, inc=20.0, omega=10.0, Omega=15.0, a_mas=12.0)
    n_data = 100
    dof = n_data - sp_sky.N_ORBIT_FREE
    crit_chi2_n = float(chi2_dist.isf(sp_sky.GOOD_FIT_ALPHA, dof) / n_data)
    assert sp_sky.good_fit(0.95 * crit_chi2_n, n_data) is True
    assert sp_sky.good_fit(1.05 * crit_chi2_n, n_data) is False
    assert sp_sky.score(meta, far_best, t_ref=t_ref, chi2_n=0.95 * crit_chi2_n, n_data=n_data)["verdict"] == "REFERENCE-DISAGREES"
    assert sp_sky.score(meta, far_best, t_ref=t_ref, chi2_n=1.05 * crit_chi2_n, n_data=n_data)["verdict"] == "DIFFERENT"

    record("twin_rms_mas", float(sc_twin["rms"]))
    record("good_fit_critical_chi2_n", crit_chi2_n)
    assert sc_twin["rms"] < 1e-12


# ------------------------------------------------ 8. crosscheck.radio_trust


@pytest.mark.validates(
    "crosscheck.radio_trust",
    roots=["mathematics"],
    kind="reference",
    headline="three_ant_closed_triangle_abs_err",
)
def test_radio_trust_dirty_briggs_taper_image_model_and_closed_triangle_gain_marginal():
    """``dirty`` peaks at 1 on the phase centre and matches an analytical cosine fringe,
    ``density_weights`` folds Hermitian partners and interpolates between uniform and natural
    limits, ``uv_taper`` matches 2D sky-plane Gaussian quadrature at an oblique PA, ``image_model``
    matches analytic pixel phase ramps on both even and odd ``npix`` grids, and
    ``gain_marginal_brute_force`` matches exact integrals at ``M = 0`` and on a closed 3-antenna
    triangle in both coupled phase-closure and coupled 3-antenna log-amplitude limits."""
    from crosscheck import radio_trust as rt
    # Dirty beam & off-centre point source
    u = np.array([1.0e5, -2.0e5, 3.0e5])
    v = np.array([0.5e5, 1.5e5, -1.0e5])
    w = np.array([2.0, 3.0, 5.0])
    beam = rt.dirty(u, v, np.ones(3), w, npix=17, pixel_mas=10.0)
    assert beam[8, 8] == pytest.approx(1.0, abs=1e-15)
    # Source at row 6 (+20 mas North), col 5 (+30 mas East)
    dra, ddec = 30.0 * rt.MAS, 20.0 * rt.MAS
    vis_pt = 2.5 * np.exp(-2j * np.pi * (u * dra + v * ddec))
    img_pt = rt.dirty(u, v, vis_pt, w, npix=17, pixel_mas=10.0)
    assert img_pt[6, 5] == pytest.approx(2.5, rel=1e-14)

    # Briggs Hermitian partner folding & robust limits
    u_b = np.array([50.0, -50.0, 250.0])  # first two share cell after Hermitian fold
    v_b = np.array([50.0, -50.0, 50.0])
    w_b = np.array([2.0, 6.0, 4.0])
    w_unif = rt.density_weights(u_b, v_b, w_b, cell=100.0, kind="uniform")
    assert np.allclose(w_unif, [0.25, 0.75, 1.0], rtol=1e-14)
    w_rob_nat = rt.density_weights(u_b, v_b, w_b, cell=100.0, kind="briggs", robust=10.0)
    assert np.allclose(w_rob_nat / w_rob_nat.sum(), w_b / w_b.sum(), rtol=1e-12)
    w_rob_uni = rt.density_weights(u_b, v_b, w_b, cell=100.0, kind="briggs", robust=-10.0)
    assert np.allclose(w_rob_uni / w_rob_uni.sum(), w_unif / w_unif.sum(), rtol=1e-12)

    # uv_taper against direct 2D sky-plane elliptical Gaussian quadrature at oblique PA = 37 deg
    rng = np.random.default_rng(71)
    u_tap = rng.uniform(-4.0e6, 4.0e6, 24)
    v_tap = rng.uniform(-4.0e6, 4.0e6, 24)
    bmaj_mas, bmin_mas, pa_deg = 35.0, 18.0, 37.0
    got_taper = rt.uv_taper(u_tap, v_tap, bmaj_mas, bmin_mas, pa_deg)
    want_taper = _sky_gaussian_vis_quadrature(u_tap, v_tap, bmaj_mas, bmin_mas / bmaj_mas, pa_deg)
    taper_err = float(np.max(np.abs(got_taper - want_taper)))
    taper_wrong_pa = rt.uv_taper(u_tap, v_tap, bmaj_mas, bmin_mas, -pa_deg)
    taper_mirrored_u = rt.uv_taper(-u_tap, v_tap, bmaj_mas, bmin_mas, pa_deg)
    assert taper_err < 1e-12
    assert float(np.max(np.abs(taper_wrong_pa - want_taper))) > 0.05
    assert float(np.max(np.abs(taper_mirrored_u - want_taper))) > 0.05

    # image_model against closed-form pixel phase ramps on both odd (npix=17) and even (npix=16) grids
    mas_to_rad = np.pi / (180.0 * 3600.0 * 1000.0)
    img_model_err = 0.0
    for npix, pix_a, pix_b in ((17, (5, 11), (12, 3)), (16, (4, 10), (11, 2))):
        eta = np.full((npix, npix), -80.0)
        eta[pix_a] = np.log(0.7)
        eta[pix_b] = np.log(0.3)
        pixel_mas, flux = 8.0, 1.75
        got_im = rt.image_model(eta, pixel_mas, u_tap, v_tap, flux=flux)
        c0 = (npix - 1) / 2.0
        xa, ya = (c0 - pix_a[1]) * pixel_mas * mas_to_rad, (c0 - pix_a[0]) * pixel_mas * mas_to_rad
        xb, yb = (c0 - pix_b[1]) * pixel_mas * mas_to_rad, (c0 - pix_b[0]) * pixel_mas * mas_to_rad
        want_im = flux * (
            0.7 * np.exp(-2j * np.pi * (u_tap * xa + v_tap * ya))
            + 0.3 * np.exp(-2j * np.pi * (u_tap * xb + v_tap * yb))
        )
        img_model_err = max(img_model_err, float(np.max(np.abs(got_im - want_im))))
    assert img_model_err < 1e-14

    # 1. Exact closed-form integral of gain_marginal_brute_force when model == 0:
    vis0 = np.array([0.3 - 0.4j, -0.2 + 0.5j, 0.1 + 0.2j])
    w0 = np.array([4.0, 7.0, 3.0])
    a1, a2 = np.array([0, 0, 1]), np.array([1, 2, 2])
    got_zero, mode_zero = rt.gain_marginal_brute_force(
        np.zeros(3, complex), vis0, w0, a1, a2, n_ant=3, tau_a=0.15, tau_p=0.4, nodes=7
    )
    want_zero = float(0.5 * np.sum(w0 * np.abs(vis0) ** 2))
    zero_err = abs(float(got_zero) - want_zero)
    assert np.max(np.abs(mode_zero)) < 1e-7

    # 2. Closed 3-antenna triangle (p = 6, off-diagonal Cholesky Hessian across all 3 baselines):
    from scipy.integrate import dblquad

    m3 = np.array([0.7 + 0.2j, -0.4 + 0.5j, 0.3 - 0.6j])
    v3 = np.array([0.82 + 0.15j, -0.35 + 0.58j, 0.26 - 0.52j])
    w3 = np.array([6.0, 9.0, 5.0])

    # (a) Coupled phase-closure limit (tau_a = 1e-6, tau_p = 0.32):
    # d1 = phi_0 - phi_1, d2 = phi_0 - phi_2 (so phi_1 - phi_2 = d2 - d1) have correlated
    # Gaussian prior covariance tau_p^2 * [[2, 1], [1, 2]] (det = 3 * tau_p^4).
    tau_p_tri = 0.32
    got_tri_phase, _ = rt.gain_marginal_brute_force(
        m3, v3, w3, a1, a2, n_ant=3, tau_a=1e-6, tau_p=tau_p_tri, nodes=9
    )
    norm_d = 2.0 * np.pi * np.sqrt(3.0) * tau_p_tri**2
    lim_d = 7.0 * np.sqrt(2.0) * tau_p_tri
    int_phase, _ = dblquad(
        lambda d2, d1: np.exp(
            -0.5
            * (
                w3[0] * abs(v3[0] - np.exp(1j * d1) * m3[0]) ** 2
                + w3[1] * abs(v3[1] - np.exp(1j * d2) * m3[1]) ** 2
                + w3[2] * abs(v3[2] - np.exp(1j * (d2 - d1)) * m3[2]) ** 2
            )
            - (d1**2 - d1 * d2 + d2**2) / (3.0 * tau_p_tri**2)
        )
        / norm_d,
        -lim_d,
        lim_d,
        -lim_d,
        lim_d,
        epsabs=1e-11,
        epsrel=1e-11,
    )
    tri_phase_err = abs(float(got_tri_phase) - float(-np.log(int_phase)))

    # (b) Coupled 3-antenna log-amplitude limit (tau_a = 0.14, tau_p = 1e-6):
    # On an odd cycle, (lam_0 + lam_1, lam_0 + lam_2, lam_1 + lam_2) is full-rank (det = -2)
    # with full 3x3 off-diagonal coupling; integrate directly over (lam_0, lam_1, lam_2) by
    # unwhitened 3D Gauss-Legendre quadrature (shares no mode, Hessian, or Cholesky with rt).
    tau_a_tri = 0.14
    got_tri_amp, _ = rt.gain_marginal_brute_force(
        m3, v3, w3, a1, a2, n_ant=3, tau_a=tau_a_tri, tau_p=1e-6, nodes=9
    )
    gl_x, gl_w = np.polynomial.legendre.leggauss(48)
    half_a = 7.5 * tau_a_tri
    pts_a = half_a * gl_x
    wts_a = half_a * gl_w / (np.sqrt(2.0 * np.pi) * tau_a_tri) * np.exp(-0.5 * (pts_a / tau_a_tri) ** 2)
    l0, l1, l2 = np.meshgrid(pts_a, pts_a, pts_a, indexing="ij")
    wt3 = np.einsum("i,j,k->ijk", wts_a, wts_a, wts_a)
    chi2_amp = (
        w3[0] * np.abs(v3[0] - np.exp(l0 + l1) * m3[0]) ** 2
        + w3[1] * np.abs(v3[1] - np.exp(l0 + l2) * m3[1]) ** 2
        + w3[2] * np.abs(v3[2] - np.exp(l1 + l2) * m3[2]) ** 2
    )
    want_tri_amp = float(-np.log(np.sum(wt3 * np.exp(-0.5 * chi2_amp))))
    tri_amp_err = abs(float(got_tri_amp) - want_tri_amp)
    tri_err = max(tri_phase_err, tri_amp_err)

    record("three_ant_closed_triangle_abs_err", float(tri_err))
    record("uv_taper_quad_abs_err", taper_err)
    record("image_model_exact_abs_err", img_model_err)
    record("zero_model_marginal_abs_err", float(zero_err))
    assert zero_err < 1e-12
    assert tri_phase_err < 1e-7
    assert tri_amp_err < 1e-7


# ------------------------------------------------ 9. crosscheck.toeplitz_scenes


@pytest.mark.validates(
    "crosscheck.toeplitz_scenes",
    roots=["mathematics"],
    kind="reference",
    headline="nyquist_reach_rel_err",
)
def test_toeplitz_scenes_hash_seed_unit_brightness_nyquist_reach_and_points():
    """``seed`` matches SHA-256, all three held-out ``image`` families are non-negative with
    unit sum, ``coverage`` places its longest baseline at ``reach`` of Nyquist, and
    ``make_case`` / ``points`` preserve the reference observation and ``+-0.9 * bound`` edge offsets."""
    from crosscheck import toeplitz_scenes as ts
    from crosscheck import radio_innovations as ri
    want_seed = int(hashlib.sha256(b"toeplitz-reexam-042/ringed/truth").hexdigest()[:8], 16)
    assert ts.seed("ringed", "truth") == want_seed

    for kind in ("ringed_disc", "sources_on_halo", "edge_filled"):
        im = ts.image(kind, npix=32, pixel_mas=5.0, rng=np.random.default_rng(1))
        assert im.shape == (32, 32)
        assert np.all(im >= 0.0)
        assert float(im.sum()) == pytest.approx(1.0, rel=1e-14)

    pixel_mas, reach = 5.0, 0.73
    u, v, ddid = ts.coverage(n_ant=8, n_times=6, freqs_hz=[220e9, 240e9], reach=reach, pixel_mas=pixel_mas, rng=np.random.default_rng(2))
    q_nyq = 1.0 / (2.0 * pixel_mas * ri.MAS)
    reach_err = abs(float(np.max(np.hypot(u, v)) / (reach * q_nyq)) - 1.0)

    spec = dict(
        name="mini",
        image="ringed_disc",
        npix=16,
        pixel_mas=pixel_mas,
        n_ant=6,
        n_times=4,
        n_obs=3,
        n_ddid=4,
        reach=0.6,
        offsets="spread",
        offset_bound_mas=15.0,
    )
    case, truth = ts.make_case(spec)
    pts = ts.points(case, truth, spec, sigma_opt=truth["sigma"])
    assert set(pts) == {"optimum", "moved", "edge"}
    for p in pts.values():
        assert p["gain"][0] == 1.0 and np.all(p["offset"][0] == 0.0)
    assert np.all(np.abs(pts["edge"]["offset"][1:]) == pytest.approx(0.9 * 15.0))

    record("nyquist_reach_rel_err", float(reach_err))
    record("edge_offset_abs_mas", float(np.max(np.abs(pts["edge"]["offset"][1:]))))
    assert reach_err < 1e-13


# ------------------------------------------------ 10. virgil_bridge.alma_lambda


@pytest.mark.validates(
    "virgil_bridge.alma_lambda",
    roots=["mathematics", "statistics"],
    kind="reference",
    headline="lag1_white_noise_abs_bias",
)
def test_alma_lambda_grid_flux_scale_reference_obs_and_white_noise_lag1():
    """``rule()`` verifies the committed criteria phrases, ``grid(s2)`` steps by ``sqrt(10)``
    from ``1e8 * s2`` to ``1e2 * s2``, ``reference_obs`` and ``flux_scale`` match analytical
    sums, and ``lag1_correlation`` on white complex Gaussian noise has expectation ``-1/2``."""
    radio = pytest.importorskip("virgil.radio")
    from virgil_bridge import alma_lambda as al

    blk = al.rule()
    assert blk["rule"] == "discrepancy_dof_corrected"
    with pytest.raises(SystemExit):
        al.rule(dict(blk, text="tampered criteria text missing required phrases"))

    s2 = 1.35
    g = al.grid(s2)
    assert g[0] == pytest.approx(1e8 * s2, rel=1e-14)
    assert g[-1] == pytest.approx(1e2 * s2, rel=1e-14)
    assert np.allclose(g[:-1] / g[1:], np.sqrt(10.0), rtol=1e-14)

    # Synthetic multi-scan time series of i.i.d. complex Gaussian noise: first differences have lag-1 rho = -0.5
    rng = np.random.default_rng(31)
    n_times = 400
    rows = []
    for obs_id, w_val in ((0, 2.0), (1, 5.0)):
        for ant2 in (1, 2, 3):
            t = np.arange(n_times, dtype=float) + 1000.0 * obs_id
            vis = (rng.standard_normal(n_times) + 1j * rng.standard_normal(n_times)) / np.sqrt(w_val)
            u = np.full(n_times, 1e4 * (ant2 + 10 * obs_id))
            v = np.zeros(n_times)
            rows.append((u, v, vis, np.full(n_times, w_val), np.full(n_times, 230e9), t, np.zeros(n_times, int), np.full(n_times, ant2, int), np.full(n_times, obs_id, int)))

    def cat(idx):
        return np.concatenate([r[idx] for r in rows])
    data = radio.ComplexVisData.from_arrays(
        cat(0), cat(1), cat(2), cat(3), cat(4),
        ddid=cat(8), obs=cat(8), time=cat(5), ant1=cat(6), ant2=cat(7), scan=cat(8),
    )
    ref_obs, sums = al.reference_obs(data)
    assert ref_obs == 1 and sums[1] > sums[0]

    r1, n_lag = al.lag1_correlation(data)
    se_r1 = 1.0 / np.sqrt(n_lag)
    bias = abs(r1 - (-0.5))

    # Deterministic flux_scale on shortest baselines
    vis_det = np.where(np.hypot(cat(0), cat(1)) <= 1.5e4, 0.42 + 0.1j, 0.05 + 0.0j)
    data_det = radio.ComplexVisData.from_arrays(cat(0), cat(1), vis_det, cat(3), cat(4))
    assert al.flux_scale(data_det, fraction=0.1) == pytest.approx(0.42, rel=1e-14)

    # Problem.chi2 and Problem.curvature HVP vs explicit NumPy Vandermonde Gauss-Newton matrix Re(D^H W D) v
    npix_c, px_c, flux_c = 6, 12.0, 0.36
    m_vis = 28
    u_c = rng.uniform(-2.5e5, 2.5e5, m_vis)
    v_c = rng.uniform(-2.5e5, 2.5e5, m_vis)
    w_c = rng.uniform(1.5, 6.0, m_vis)
    vis_c = rng.normal(0.0, 0.2, m_vis) + 1j * rng.normal(0.0, 0.2, m_vis)
    data_c = radio.ComplexVisData.from_arrays(u_c, v_c, vis_c, w_c, np.full(m_vis, 230e9))
    prob = al.build(data_c, npix=npix_c, pixel_mas=px_c, flux=flux_c, ref_obs=0)
    b0 = rng.uniform(0.004, 0.025, (npix_c, npix_c))
    vec = rng.normal(0.0, 1.0, npix_c * npix_c)

    mas_rad = np.pi / (180.0 * 3600.0 * 1000.0)
    coords_1d = ((npix_c - 1) / 2.0 - np.arange(npix_c) - 0.5) * px_c * mas_rad
    east_g, north_g = np.meshgrid(coords_1d, coords_1d)
    d_mat = np.exp(-2j * np.pi * (u_c[:, None] * east_g.ravel()[None, :] + v_c[:, None] * north_g.ravel()[None, :]))
    want_chi2 = float(np.sum(w_c * np.abs(vis_c - d_mat @ b0.ravel()) ** 2))
    got_chi2 = prob.chi2({"pixels": np.sqrt(b0)})
    assert got_chi2 == pytest.approx(want_chi2, rel=1e-11)

    want_kv = ((d_mat.conj().T * w_c[None, :]) @ d_mat).real @ vec
    got_kv = prob.curvature(prob.model(pixels=np.sqrt(b0)))(vec)
    curv_rel_err = float(np.linalg.norm(got_kv - want_kv) / np.linalg.norm(want_kv))

    record("lag1_white_noise_abs_bias", float(bias))
    record("lag1_white_noise_r1", float(r1))
    record("curvature_hvp_rel_err", curv_rel_err)
    assert n_lag > 2000
    assert bias < 3.5 * se_r1
    assert curv_rel_err < 1e-11


# ------------------------------------------------ 11. virgil_bridge.mpol


def _mpol_direct_visibilities(packed_intensity, pixel_mas, u, v, obs, log10_amp, delta_ra_arcsec, delta_dec_arcsec):
    """Independent NumPy forward model in MPoL's unpacked sky coordinates:
    unpacked pixel ``(r, c)`` sits at East ``(c - N/2) * dx`` and North ``(r - N/2) * dx``
    (in arcsec), scaled by ``g_obs`` and shifted by ``exp(-2 pi i (u * dra + v * ddec))``."""
    arcsec_rad = np.pi / (180.0 * 3600.0)
    cell_arcsec = pixel_mas / 1000.0
    unpacked = np.fft.fftshift(np.asarray(packed_intensity, float))
    npix = unpacked.shape[0]
    east_rad = (np.arange(npix) - npix // 2) * cell_arcsec * arcsec_rad
    north_rad = (np.arange(npix) - npix // 2) * cell_arcsec * arcsec_rad
    flux_jy = unpacked * cell_arcsec**2
    phase_sky = -2.0 * np.pi * (
        u[:, None, None] * east_rad[None, None, :] + v[:, None, None] * north_rad[None, :, None]
    )
    vis_sky = np.sum(flux_jy[None, :, :] * np.exp(1j * phase_sky), axis=(1, 2))
    gains = np.r_[1.0, 10.0 ** np.asarray(log10_amp, float)]
    dra_rad = np.r_[0.0, np.asarray(delta_ra_arcsec, float)] * arcsec_rad
    ddec_rad = np.r_[0.0, np.asarray(delta_dec_arcsec, float)] * arcsec_rad
    phase_off = -2.0 * np.pi * (u * dra_rad[obs] + v * ddec_rad[obs])
    return gains[obs] * vis_sky * np.exp(1j * phase_off)


@pytest.mark.validates(
    "virgil_bridge.mpol",
    roots=["mathematics"],
    kind="reference",
    headline="nll_vs_mpol_dft_abs_err",
)
def test_virgil_bridge_mpol_bijection_parameter_map_and_exact_log_likelihood():
    """``unpack``/``pack`` map MPoL's ``(c - N/2, r - N/2)`` unpacked sky coordinates onto
    virgil's ``(row 0 North, col 0 East)`` grid with ``-dx/2`` half-pixel shift,
    ``virgil_values`` maps every MPoL site (including ``cal.gain`` and ``log_brightness``),
    and ``neg_log_likelihood`` matches an independent NumPy pixel-DFT plus ``scipy.stats.norm.logpdf``."""
    import jax
    import jax.numpy as jnp

    radio = pytest.importorskip("virgil.radio")
    from virgil_bridge import mpol as vbm

    rng = np.random.default_rng(44)
    npix = 8
    unpacked_mpol = rng.uniform(0.05, 0.8, (npix, npix))
    # Distinct asymmetric off-axis peaks in MPoL unpacked coordinates: (r=1, c=6) -> SE-ish (c>4 East, r<4 South)
    unpacked_mpol[1, 6] = 7.5
    unpacked_mpol[6, 2] = 4.2
    packed = np.fft.ifftshift(unpacked_mpol)

    sky_im = vbm.unpack(packed)
    assert np.array_equal(sky_im, unpacked_mpol[::-1, ::-1])
    assert sky_im[npix - 1 - 1, npix - 1 - 6] == 7.5
    assert sky_im[npix - 1 - 6, npix - 1 - 2] == 4.2
    roundtrip_layout_err = float(np.max(np.abs(vbm.pack(sky_im) - packed)))
    base_packed = vbm.softplus_inverse(packed)
    softplus_err = float(np.max(np.abs(np.asarray(jax.nn.softplus(jnp.asarray(base_packed))) - packed)))
    assert roundtrip_layout_err == 0.0
    assert softplus_err < 1e-13

    obs_map = vbm.obs_of_ddid({0: [0, 1], 1: [2], 2: [3]}, 4)
    assert list(obs_map) == [0, 0, 1, 2]
    pixel_mas = 5.0
    s_m = np.array([1.1, 1.3, 0.9, 1.5])
    amps = np.array([1.08, 0.92])
    dra_arcsec = np.array([0.002, -0.003])
    ddec_arcsec = np.array([-0.001, 0.004])
    theta = {
        "base_cube": jnp.asarray(base_packed),
        "log10_sigma": jnp.log10(jnp.asarray(s_m)),
        "log10_amp": jnp.log10(jnp.asarray(amps)),
        "delta_ra": jnp.asarray(dra_arcsec),
        "delta_dec": jnp.asarray(ddec_arcsec),
    }
    vals = vbm.virgil_values(theta, obs_map, pixel_mas, sigma_map="g_times_s")
    vals_ctrl = vbm.virgil_values(theta, obs_map, pixel_mas, sigma_map="s_only")
    assert float(vals["cal.flux"]) == pytest.approx(float(unpacked_mpol.sum() * (pixel_mas / 1000.0) ** 2), rel=1e-13)
    assert np.allclose(vals["cal.gain"], [1.08, 0.92], rtol=1e-13)
    assert np.allclose(np.exp(np.asarray(vals["log_brightness"])), sky_im, rtol=1e-13)
    assert np.allclose(vals["cal.sigma"], [1.1, 1.3, 0.9 * 1.08, 1.5 * 0.92], rtol=1e-13)
    assert np.allclose(vals_ctrl["cal.sigma"], [1.1, 1.3, 0.9, 1.5], rtol=1e-13)
    assert np.allclose(vals["cal.offset"], [[2.0, -1.0], [-3.0, 4.0]], rtol=1e-13)

    u = rng.normal(0.0, 2e5, 24)
    v = rng.normal(0.0, 2e5, 24)
    ddid = np.arange(24) % 4
    obs = obs_map[ddid]
    w = rng.uniform(10.0, 50.0, 24)
    vis = 0.05 * (rng.normal(size=24) + 1j * rng.normal(size=24))
    data = radio.ComplexVisData.from_arrays(u, v, vis, w, np.full(24, 230e9), ddid=ddid, obs=obs)
    nuis = radio.CalibrationNuisance(data, ref_obs=0, alpha=False)
    scene = vbm.imlup_scene(npix, pixel_mas)
    term = data.term(scene, nuis)
    got_nll = float(vbm.neg_log_likelihood(term, vals, data))

    # Independent MPoL forward model from theta alone (no call to imlup_scene, data.model, or vals)
    mod_vis_mpol = _mpol_direct_visibilities(
        packed, pixel_mas, u, v, obs, np.log10(amps), dra_arcsec, ddec_arcsec
    )
    got_vis = np.asarray(data.model(scene(vals), vals, nuis))
    vis_dft_err = float(np.max(np.abs(got_vis - mod_vis_mpol)))
    gains_full = np.r_[1.0, amps]
    sig_mpol = (s_m * gains_full[obs_map])[ddid] / np.sqrt(w)
    want_nll = float(
        -(norm.logpdf(vis.real, mod_vis_mpol.real, sig_mpol).sum()
          + norm.logpdf(vis.imag, mod_vis_mpol.imag, sig_mpol).sum())
    )
    nll_err = abs(got_nll - want_nll)

    record("nll_vs_mpol_dft_abs_err", nll_err)
    record("vis_vs_mpol_dft_abs_err", vis_dft_err)
    record("softplus_roundtrip_abs_err", softplus_err)
    assert vis_dft_err < 1e-11
    assert nll_err < 1e-10


@pytest.mark.validates(
    "virgil_bridge.mpol",
    roots=["mathematics"],
    kind="control",
    headline="min_wrong_map_vis_err",
)
def test_control_virgil_bridge_mpol_rejects_missing_half_pixel_or_unflipped_axes():
    """Dropping the ``-pixel_mas / 2`` half-pixel shift or omitting the ``[::-1, ::-1]`` axis
    reversal breaks agreement with MPoL's ``(c - N/2, r - N/2)`` sky visibilities on an asymmetric cube."""
    import jax.numpy as jnp

    radio = pytest.importorskip("virgil.radio")
    from virgil_bridge import mpol as vbm

    rng = np.random.default_rng(45)
    npix, pixel_mas = 8, 25.0
    unpacked_mpol = np.full((npix, npix), 0.05)
    unpacked_mpol[1, 6] = 12.0
    packed = np.fft.ifftshift(unpacked_mpol)
    obs_map = np.array([0, 1])
    theta = {
        "base_cube": jnp.asarray(vbm.softplus_inverse(packed)),
        "log10_sigma": jnp.zeros(2),
        "log10_amp": jnp.zeros(1),
        "delta_ra": jnp.zeros(1),
        "delta_dec": jnp.zeros(1),
    }
    vals = vbm.virgil_values(theta, obs_map, pixel_mas)
    u = rng.uniform(3e5, 8e5, 16)
    v = rng.uniform(3e5, 8e5, 16)
    ddid = np.arange(16) % 2
    obs = obs_map[ddid]
    data = radio.ComplexVisData.from_arrays(u, v, np.zeros(16, complex), np.ones(16), np.full(16, 230e9), ddid=ddid, obs=obs)
    nuis = radio.CalibrationNuisance(data, ref_obs=0, alpha=False)
    want_vis = _mpol_direct_visibilities(packed, pixel_mas, u, v, obs, [0.0], [0.0], [0.0])

    no_shift_vis = np.asarray(data.model(vbm.imlup_scene(npix, pixel_mas, half_pixel=False)(vals), vals, nuis))
    unflipped_vals = {**vals, "log_brightness": jnp.log(jnp.asarray(unpacked_mpol))}
    unflipped_vis = np.asarray(data.model(vbm.imlup_scene(npix, pixel_mas)(unflipped_vals), unflipped_vals, nuis))

    err_no_shift = float(np.max(np.abs(no_shift_vis - want_vis)) / np.max(np.abs(want_vis)))
    err_unflipped = float(np.max(np.abs(unflipped_vis - want_vis)) / np.max(np.abs(want_vis)))
    min_err = min(err_no_shift, err_unflipped)
    record("min_wrong_map_vis_err", min_err)
    assert err_no_shift > 0.05
    assert err_unflipped > 0.2


# ------------------------------------------------ 12. mpol (external package)


@pytest.mark.skipif(not ext_sp.available("mpol", "mpol"), reason="MPoL not installed (scripts/setup_external.sh mpol)")
@pytest.mark.external
@pytest.mark.validates(
    "mpol",
    roots=["mathematics"],
    kind="reference",
    headline="max_grad_fd_rel_err",
)
@pytest.mark.validates(
    "external_bridge.mpol_worker",
    roots=["mathematics"],
    kind="reference",
    headline="max_grad_fd_rel_err",
)
def test_mpol_losses_match_analytic_entropy_tv_tsv_and_central_differences():
    """MPoL's ``losses.entropy``, ``losses.TV_image``, and ``losses.TSV`` match their closed-form
    sums on a positive unit-sum image, and their PyTorch autograd gradients match central finite
    differences."""
    rng = np.random.default_rng(57)
    raw = np.exp(rng.normal(0.0, 0.6, (7, 7)))
    im = raw / raw.sum()
    prior = 1.0 / im.size
    eps = 1e-6

    out = ext_sp.run(
        "mpol",
        "mpol_worker.py",
        {"task": "regularisers", "images": [im.tolist()], "priors": [prior], "tv_epsilon": eps},
    )["results"][0]

    # Analytic values (MPoL uses interior forward steps for TV and TSV, tot_flux=1 for entropy)
    want_ent = float(np.sum(im * np.log(im / prior)))
    dx = im[1:, :-1] - im[:-1, :-1]
    dy = im[:-1, 1:] - im[:-1, :-1]
    want_tsv = float(np.sum(dx**2 + dy**2))
    want_tv = float(np.sum(np.sqrt(dx**2 + dy**2 + eps)))

    val_err = max(
        abs(out["entropy"] / want_ent - 1.0),
        abs(out["tsv"] / want_tsv - 1.0),
        abs(out["tv"] / want_tv - 1.0),
    )

    # Central finite differences of the three analytical regularisers
    def eval_np(x):
        a = x.reshape(im.shape)
        ddx = a[1:, :-1] - a[:-1, :-1]
        ddy = a[:-1, 1:] - a[:-1, :-1]
        return np.array([
            np.sum(a * np.log(a / prior)),
            np.sum(ddx**2 + ddy**2),
            np.sum(np.sqrt(ddx**2 + ddy**2 + eps)),
        ])

    h = 1e-7
    x0 = im.ravel()
    fd = np.zeros((3, x0.size))
    for k in range(x0.size):
        step = np.zeros_like(x0)
        step[k] = h
        fd[:, k] = (eval_np(x0 + step) - eval_np(x0 - step)) / (2.0 * h)

    max_grad_err = 0.0
    for idx, key in enumerate(("entropy_grad", "tsv_grad", "tv_grad")):
        got_g = np.asarray(out[key], float).ravel()
        max_grad_err = max(max_grad_err, float(np.max(np.abs(got_g - fd[idx])) / np.max(np.abs(fd[idx]))))

    record("max_grad_fd_rel_err", max_grad_err)
    record("max_value_rel_err", float(val_err))
    assert val_err < 1e-12
    assert max_grad_err < 1e-8


# ------------------------------------------------ 13. virgil_bridge.alma


@pytest.mark.validates(
    "virgil_bridge.alma",
    roots=["mathematics"],
    kind="reference",
    headline="mpol_entropy_spatial_rel_err",
)
def test_virgil_bridge_alma_lambda_scales_ring_geometry_and_mpol_entropy():
    """``lam_f_equivalent`` gives ``MaxEntropy`` the exact stationary-point gradient pull of
    untapered ``mpol_entropy_jax`` with respect to ``log_brightness``, ``ring_and_blob`` has unit
    sum in the expected sky quadrant, and ``mpol_entropy_jax`` matches direct 2D sky-plane spatial
    Gaussian PSF convolution (no FFTs) plus Shannon relative entropy."""
    import jax
    import jax.numpy as jnp

    pytest.importorskip("virgil.radio")
    from virgil import Image
    from virgil.imaging import MaxEntropy
    from virgil_bridge import alma as vba

    # 1. Stationary-point gradient identity: d/d(eta) [lam_R S_M(eta; taper=0)] == d/d(eta) [MaxEntropy(lam_F)(eta)]
    rng = np.random.default_rng(61)
    npix = 31
    pixel_mas, taper_mas, flux, prior, f_tot = 4.0, 20.0, 0.32, 1e-4, 0.25
    eta = rng.normal(0.0, 0.8, (npix, npix))
    eta_j = jnp.asarray(eta)

    lam_r_val = vba.lam_r(50_000, lam_m=1e-9)
    lam_f_val = vba.lam_f_equivalent(lam_r_val, flux=flux, pixel_mas=pixel_mas, f_tot=f_tot)
    grad_r = np.asarray(
        jax.grad(
            lambda e: lam_r_val * vba.mpol_entropy_jax(e, flux, pixel_mas=pixel_mas, taper_mas=0.0, prior=prior, f_tot=f_tot)
        )(eta_j),
        float,
    )
    grad_f = np.asarray(
        jax.grad(lambda e: MaxEntropy(lam_f_val).value(Image(e, pixel_scale_mas=pixel_mas)))(eta_j),
        float,
    )
    grad_pull_rel_err = float(np.linalg.norm(grad_r - grad_f) / np.linalg.norm(grad_r))
    assert grad_pull_rel_err < 1e-12

    scene_im = vba.ring_and_blob(npix=33, pixel_mas=20.0, blob=(0.4, 140.0, -160.0, 40.0))
    assert scene_im.shape == (33, 33)
    assert np.all(scene_im >= 0.0)
    assert float(scene_im.sum()) == pytest.approx(1.0, rel=1e-14)
    # Blob at (+140 mas East -> col < 16, -160 mas North -> row > 16): SE quadrant brighter than NW
    assert float(scene_im[17:, :16].sum()) > float(scene_im[:16, 17:].sum())

    # 2. Direct 2D sky-plane spatial Gaussian convolution (zero FFTs) vs mpol_entropy_jax
    got_ent = float(
        vba.mpol_entropy_jax(
            eta_j, flux, pixel_mas=pixel_mas, taper_mas=taper_mas, prior=prior, f_tot=f_tot
        )
    )
    dx_arcsec = pixel_mas / 1000.0
    fwhm_arcsec = taper_mas / 1000.0
    b = np.exp(eta - eta.max())
    b /= b.sum()
    sb = flux * b / dx_arcsec**2
    half = npix // 2
    lags = np.arange(-half, half + 1)
    dr_g, dc_g = np.meshgrid(lags, lags, indexing="ij")
    ln2 = np.log(2.0)
    norm_psf = (4.0 * ln2 / (np.pi * fwhm_arcsec**2)) * (dx_arcsec**2)
    kernel = np.zeros((npix, npix))
    for mr in (-1, 0, 1):
        for mc in (-1, 0, 1):
            r2_arcsec = (((dr_g + mr * npix) ** 2 + (dc_g + mc * npix) ** 2)) * (dx_arcsec**2)
            kernel += norm_psf * np.exp(-4.0 * ln2 * r2_arcsec / (fwhm_arcsec**2))
    sb_tap = np.zeros_like(sb)
    for ir, lr in enumerate(lags):
        for ic, lc in enumerate(lags):
            sb_tap += kernel[ir, ic] * np.roll(sb, (int(lr), int(lc)), axis=(0, 1))
    sb_tap = np.maximum(sb_tap, 1e-30)
    want_ent = float(np.sum(sb_tap * np.log(sb_tap / prior)) / f_tot)
    ent_rel_err = abs(got_ent / want_ent - 1.0)
    # Negative control: mistaking FWHM for sigma (factor ~2.355) shifts the tapered entropy substantially
    wrong_ent = float(
        vba.mpol_entropy_jax(
            eta_j,
            flux,
            pixel_mas=pixel_mas,
            taper_mas=taper_mas / (2.0 * np.sqrt(2.0 * ln2)),
            prior=prior,
            f_tot=f_tot,
        )
    )
    assert abs(wrong_ent / want_ent - 1.0) > 1e-3

    record("mpol_entropy_spatial_rel_err", ent_rel_err)
    record("lam_f_grad_pull_rel_err", grad_pull_rel_err)
    record("ring_and_blob_sum_abs_err", abs(float(scene_im.sum()) - 1.0))
    assert ent_rel_err < 1e-9


# ------------------------ 14. subprocess workers & safe_load (external_bridge)


@pytest.mark.skipif(
    not ext_sp.available("alma", "casatools"),
    reason="CASA not installed (scripts/setup_external.sh alma)",
)
@pytest.mark.external
@pytest.mark.validates(
    "external_bridge.casa_clean_worker",
    roots=["casa", "mathematics"],
    kind="reference",
    headline="orient_wcs_dft_abs_err",
)
def test_casa_clean_worker_orient_wcs_mapping_and_permuted_spw_columns(tmp_path):
    """``casa_clean_worker._orient`` maps CASA's ``[x (East->West), y (South->North), pol, chan]``
    WCS axes onto ``[row (North->South), col (East->West)]`` with identical sky-plane Fourier
    phases, and ``columns`` indexes ``CHAN_FREQ`` through ``SPECTRAL_WINDOW_ID`` rather than row index."""
    import shutil
    from external_bridge import casa_clean_worker as ccw

    # 1. WCS sky-plane Fourier invariance of _orient on an asymmetric 4D CASA cube
    nx, ny, dx_mas = 12, 10, 5.0
    mas_rad = np.pi / (180.0 * 3600.0 * 1000.0)
    x0, y0 = (nx - 1) / 2.0, (ny - 1) / 2.0
    rng = np.random.default_rng(71)
    casa_cube = np.zeros((nx, ny, 1, 1))
    # Place distinct off-axis sources in CASA [ix, iy]: CDELT1 = -dx (East at ix=0), CDELT2 = +dx (North at iy=ny-1)
    casa_cube[2, 8, 0, 0] = 3.4  # NE
    casa_cube[9, 1, 0, 0] = 1.9  # SW
    casa_cube[1, 2, 0, 0] = 0.8  # SE
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    east_casa = (x0 - ix) * dx_mas * mas_rad
    north_casa = (iy - y0) * dx_mas * mas_rad

    oriented = ccw._orient(casa_cube)
    assert oriented.shape == (ny, nx)
    assert oriented[ny - 1 - 8, 2] == 3.4
    assert oriented[ny - 1 - 1, 9] == 1.9
    assert oriented[ny - 1 - 2, 1] == 0.8
    rr, cc = np.indices((ny, nx), dtype=float)
    east_repo = (x0 - cc) * dx_mas * mas_rad
    north_repo = ((ny - 1 - y0) - rr) * dx_mas * mas_rad

    u = rng.uniform(-3e5, 3e5, 20)
    v = rng.uniform(-3e5, 3e5, 20)
    vis_casa = np.sum(
        casa_cube[:, :, 0, 0][None, :, :]
        * np.exp(-2j * np.pi * (u[:, None, None] * east_casa[None] + v[:, None, None] * north_casa[None])),
        axis=(1, 2),
    )
    vis_repo = np.sum(
        oriented[None, :, :]
        * np.exp(-2j * np.pi * (u[:, None, None] * east_repo[None] + v[:, None, None] * north_repo[None])),
        axis=(1, 2),
    )
    orient_err = float(np.max(np.abs(vis_casa - vis_repo)))

    # 2. Real CASA MeasurementSet with reversed SPECTRAL_WINDOW_ID = [1, 0]
    ms_orig = tmp_path / "orig.ms"
    ms_perm = tmp_path / "perm.ms"
    ext_sp.run("alma", "alma_worker.py", {"task": "simulate_ms", "path": str(ms_orig)})
    shutil.copytree(ms_orig, ms_perm)

    import subprocess
    perm_code = (
        "import casatools, numpy as np, sys; "
        "tb = casatools.table(); tb.open(sys.argv[1] + '/DATA_DESCRIPTION', nomodify=False); "
        "tb.putcol('SPECTRAL_WINDOW_ID', np.array([1, 0], dtype=np.int32)); tb.close(); tb.done()"
    )
    subprocess.run(
        [ext_sp.python("alma"), "-c", perm_code, str(ms_perm)],
        check=True,
        env=ext_sp._env("alma"),
        capture_output=True,
    )

    out_orig = dict(
        np.load(ext_sp.run("alma", "casa_clean_worker.py", {"task": "columns", "ms": str(ms_orig), "out": str(tmp_path / "c0.npz")})["npz"])
    )
    out_perm = dict(
        np.load(ext_sp.run("alma", "casa_clean_worker.py", {"task": "columns", "ms": str(ms_perm), "out": str(tmp_path / "c1.npz")})["npz"])
    )
    spw_swap_err = float(
        max(
            np.max(np.abs(out_perm["ddid_freqs"][0] - out_orig["ddid_freqs"][1])),
            np.max(np.abs(out_perm["ddid_freqs"][1] - out_orig["ddid_freqs"][0])),
        )
    )
    assert np.max(np.abs(out_orig["ddid_freqs"][0] - out_orig["ddid_freqs"][1])) > 1e9
    assert np.array_equal(out_perm["UVW"], out_orig["UVW"])
    assert np.array_equal(out_perm["DATA"], out_orig["DATA"])

    record("orient_wcs_dft_abs_err", orient_err)
    record("permuted_spw_freq_hz_err", spw_swap_err)
    assert orient_err < 1e-14
    assert spw_swap_err == 0.0


@pytest.mark.validates(
    "external_bridge.casa_clean_worker",
    roots=["mathematics"],
    kind="control",
    headline="unflipped_y_vis_miss",
)
def test_control_casa_clean_worker_transpose_without_y_flip_inverts_north_south():
    """Omitting ``[:, ::-1]`` before ``.T`` maps CASA's South-to-North ``y`` axis upside down."""
    from external_bridge import casa_clean_worker as ccw

    nx, ny, dx_mas = 10, 10, 8.0
    mas_rad = np.pi / (180.0 * 3600.0 * 1000.0)
    cube = np.zeros((nx, ny, 1, 1))
    cube[3, 8, 0, 0] = 2.5
    right = ccw._orient(cube)
    wrong = cube[:, :, 0, 0].T
    c = (nx - 1) / 2.0
    rr, _ = np.indices((ny, nx), dtype=float)
    north = (c - rr) * dx_mas * mas_rad
    v = np.linspace(1e5, 6e5, 12)
    vis_right = np.sum(right[None] * np.exp(-2j * np.pi * v[:, None, None] * north[None]), axis=(1, 2))
    vis_wrong = np.sum(wrong[None] * np.exp(-2j * np.pi * v[:, None, None] * north[None]), axis=(1, 2))
    miss = float(np.max(np.abs(vis_right - vis_wrong)))
    record("unflipped_y_vis_miss", miss)
    assert miss > 1.0


@pytest.mark.skipif(
    not ext_sp.available("alma", "casatools"),
    reason="CASA not installed (scripts/setup_external.sh alma)",
)
@pytest.mark.external
@pytest.mark.validates(
    "external_bridge.alma_worker",
    roots=["casa", "mathematics"],
    kind="reference",
    headline="max_two_source_vis_abs_err",
)
def test_alma_worker_simulate_and_read_match_analytic_two_source_visibilities_and_weights(tmp_path):
    """``alma_worker`` simulates two point sources in CASA and reads ``u = +UVW[0] * nu / c``,
    ``v = +UVW[1] * nu / c``, conjugated ``DATA``, and ``WEIGHT_SPECTRUM`` parallel-hand sums
    matching the analytic sky phase sum."""
    ms = tmp_path / "aw.ms"
    info = ext_sp.run("alma", "alma_worker.py", {"task": "simulate_ms", "path": str(ms)})
    ext_sp.run("alma", "alma_worker.py", {"task": "add_weight_spectrum", "path": str(ms)})
    read_out = ext_sp.run(
        "alma",
        "alma_worker.py",
        {"task": "read", "path": str(ms), "out": str(tmp_path / "npz"), "conjugate": True},
    )
    unconj_out = ext_sp.run(
        "alma",
        "alma_worker.py",
        {"task": "read", "path": str(ms), "out": str(tmp_path / "npz_raw"), "conjugate": False},
    )
    mas_rad = np.pi / (180.0 * 3600.0 * 1000.0)
    worst_vis = 0.0
    worst_w = 0.0
    for entry, raw_entry in zip(read_out["files"], unconj_out["files"]):
        z = dict(np.load(entry["file"]))
        z_raw = dict(np.load(raw_entry["file"]))
        assert str(z["weight_source"]) == "WEIGHT_SPECTRUM"
        vis_got = z["re"].astype(float) + 1j * z["im"].astype(float)
        vis_raw = z_raw["re"].astype(float) + 1j * z_raw["im"].astype(float)
        vis_want = sum(
            flux * np.exp(-2j * np.pi * (z["u"] * east + z["v"] * north) * mas_rad)
            for flux, east, north in info["components"]
        )
        assert float(np.max(np.abs(vis_raw - np.conj(vis_got)))) == 0.0
        assert float(np.max(np.abs(vis_raw - vis_want))) > 1e-3
        w_want = 2.0 * (z["chan"].astype(float) + 1.0)
        worst_vis = max(worst_vis, float(np.max(np.abs(vis_got - vis_want))))
        worst_w = max(worst_w, float(np.max(np.abs(z["weight"].astype(float) - w_want))))

    record("max_two_source_vis_abs_err", worst_vis)
    record("max_weight_sum_abs_err", worst_w)
    assert worst_vis < 5e-7
    assert worst_w == 0.0


@pytest.mark.skipif(
    not ext_sp.available("ehtim", "ehtim"),
    reason="eht-imaging not installed (scripts/setup_external.sh ehtim)",
)
@pytest.mark.external
@pytest.mark.validates(
    "external_bridge.ehtim_worker",
    roots=["mathematics"],
    kind="reference",
    headline="max_ft_abs_err",
)
def test_ehtim_worker_ft_and_cphase_chisq_match_analytic_pixel_sum_and_chord_formula():
    """``ehtim_worker.ft`` evaluates ``sum b exp(+2 pi i (u x + v y))`` in row-0-North /
    col-0-East sky coordinates, and ``chisq_cphase`` negates ``(uv1, uv2)`` so closure phases
    match OIFITS ``exp(-2 pi i (u x + v y))`` with ``(2 / N) sum (1 - cos Delta) / sigma^2``."""
    rng = np.random.default_rng(73)
    npix, scale_mas = 6, 1.2
    mas_rad = np.pi / (180.0 * 3600.0 * 1000.0)
    pdim = scale_mas * mas_rad
    im = rng.uniform(0.02, 0.5, (npix, npix))
    im[1, 4] = 3.0
    im /= im.sum()
    offs = ((npix - 1) / 2.0 - np.arange(npix)) * pdim
    east, north = np.meshgrid(offs, offs)

    uv = rng.uniform(-4e7, 4e7, (14, 2))
    res_ft = ext_sp.run("ehtim", "ehtim_worker.py", {"task": "ft", "image": im.tolist(), "pdim": pdim, "uv": uv.tolist()})
    got_ft = np.asarray(res_ft["re"]) + 1j * np.asarray(res_ft["im"])
    want_ft = np.sum(
        im[None, :, :] * np.exp(2j * np.pi * (uv[:, 0, None, None] * east[None] + uv[:, 1, None, None] * north[None])),
        axis=(1, 2),
    )
    ft_err = float(np.max(np.abs(got_ft - want_ft)))

    uv1 = rng.uniform(-3e7, 3e7, (8, 2))
    uv2 = rng.uniform(-3e7, 3e7, (8, 2))
    uv3 = -(uv1 + uv2)

    def oifits_vis(uv_arr):
        return np.sum(
            im[None, :, :]
            * np.exp(-2j * np.pi * (uv_arr[:, 0, None, None] * east[None] + uv_arr[:, 1, None, None] * north[None])),
            axis=(1, 2),
        )

    mod_cp = np.angle(oifits_vis(uv1) * oifits_vis(uv2) * oifits_vis(uv3))
    cp_deg = np.rad2deg(mod_cp) + rng.normal(0.0, 4.0, 8)
    sigma_deg = rng.uniform(1.5, 3.5, 8)
    res_cp = ext_sp.run(
        "ehtim",
        "ehtim_worker.py",
        {
            "task": "chisq_cphase",
            "image": im.tolist(),
            "pdim": pdim,
            "uv1": uv1.tolist(),
            "uv2": uv2.tolist(),
            "cp_deg": cp_deg.tolist(),
            "sigma_deg": sigma_deg.tolist(),
        },
    )
    delta_rad = np.deg2rad(cp_deg) - mod_cp
    want_chisq = float(np.mean(2.0 * (1.0 - np.cos(delta_rad)) / np.deg2rad(sigma_deg) ** 2))
    cp_rel_err = abs(res_cp["chisq"] / want_chisq - 1.0)

    record("max_ft_abs_err", ft_err)
    record("cphase_chisq_rel_err", cp_rel_err)
    assert ft_err < 1e-13
    assert cp_rel_err < 1e-12


@pytest.mark.skipif(
    not ext_sp.available("candid", "candid"),
    reason="CANDID not installed (scripts/setup_candid.sh)",
)
@pytest.mark.external
@pytest.mark.validates(
    "external_bridge.candid_worker",
    roots=["mathematics"],
    kind="reference",
    headline="max_nsigma_rel_err",
)
def test_candid_worker_chi2_and_nsigma_match_analytic_binary_and_chi2_tail(tmp_path):
    """``candid_worker`` maps ``x`` East (mas), ``y`` North (mas), and ``f = 100 * flux_ratio``
    with monochromatic ``dwavel = 0``, and ``nsigma`` matches the two-sided Gaussian equivalent
    of ``scipy.stats.chi2.sf``."""
    from scipy.stats import chi2 as sp_chi2
    from crosscheck import simulate as cx_sim, sky as cx_sky

    uts3 = np.array([[-9.925, -20.335], [14.887, 30.502], [103.306, 43.999]])
    wl = np.linspace(1.6e-6, 2.2e-6, 4)
    dra, ddec, flux, diam = 5.5, -3.2, 0.04, 0.8

    def bin_vis(uu, vv, wavel):
        return (cx_sky.vis_uniform_disk(uu, vv, wavel, diam) + flux * cx_sky.vis_point(uu, vv, wavel, dra, ddec)) / (1.0 + flux)

    fits_path = tmp_path / "cand_ref.fits"
    cx_sim.observe(
        fits_path,
        bin_vis,
        uts3,
        hour_angles_h=[-1.5, 0.0, 1.5],
        wavelengths=wl,
        dec_deg=-45.0,
        sigma_v2=0.005,
        sigma_cp_deg=0.3,
        rng=None,
        closure_phases=True,
    )

    out_chi2 = ext_sp.run(
        "candid",
        "candid_worker.py",
        {
            "task": "chi2",
            "path": str(fits_path),
            "params": [
                {"x": dra, "y": ddec, "f": 100.0 * flux, "diam*": diam},
                {"x": -dra, "y": ddec, "f": 100.0 * flux, "diam*": diam},
                {"x": dra, "y": ddec, "f": flux, "diam*": diam},
            ],
        },
    )
    chi2r_true, chi2r_mirror, chi2r_unscaled = out_chi2["chi2r"]
    assert chi2r_true < 1e-8
    assert chi2r_mirror > 5.0
    assert chi2r_unscaled > 5.0

    cases = [(1.25, 80), (1.4, 200), (2.2, 120)]
    got_ns = np.asarray(ext_sp.run("candid", "candid_worker.py", {"task": "nsigma", "cases": cases})["nsigma"])
    want_ns = np.array([float(norm.isf(0.5 * sp_chi2.sf(n * r, n))) for r, n in cases])
    ns_rel_err = float(np.max(np.abs(got_ns / want_ns - 1.0)))

    record("max_nsigma_rel_err", ns_rel_err)
    record("truth_chi2r", float(chi2r_true))
    assert ns_rel_err < 1e-11


@pytest.mark.skipif(
    not ext_sp.available("orbitize", "orbitize"),
    reason="orbitize! not installed (scripts/setup_external.sh orbitize)",
)
@pytest.mark.external
@pytest.mark.validates(
    "external_bridge.orbitize_worker",
    roots=["mathematics"],
    kind="reference",
    headline="max_pos_over_sma_err",
)
def test_orbitize_worker_ephemeris_and_period_match_kepler_third_law_and_sky_projection():
    """``orbitize_worker`` returns Kepler's-third-law periods (IAU ``G M_sun``) and relative
    ``(raoff, deoff, vz)`` matching direct Newton-Raphson Keplerian projection."""
    from crosscheck import orbits as cx_orb

    au_m = 149597870700.0
    gm_sun = 1.3271244e20
    sma, ecc, inc_deg, aop_deg, pan_deg, tau, plx, mtot = 12.5, 0.34, 52.0, 73.0, 128.0, 0.18, 25.0, 1.65
    tau_ref = 58849.0
    epochs = np.linspace(58900.0, 61500.0, 15)
    el = {
        "sma": sma,
        "ecc": ecc,
        "inc": np.deg2rad(inc_deg),
        "aop": np.deg2rad(aop_deg),
        "pan": np.deg2rad(pan_deg),
        "tau": tau,
        "plx": plx,
        "mtot": mtot,
    }
    res = ext_sp.run(
        "orbitize",
        "orbitize_worker.py",
        {
            "task": "batch",
            "tasks": {
                "p": {"task": "period", "cases": [[sma, mtot]]},
                "e": {
                    "task": "ephemeris",
                    "orbits": [el],
                    "epochs": epochs.tolist(),
                    "tau_ref_epoch": tau_ref,
                    "tolerance": 1e-14,
                },
            },
        },
    )["results"]
    p_got = res["p"]["results"][0]["period_day"]
    sma_back = res["p"]["results"][0]["sma_back"]
    p_want = 2.0 * np.pi * np.sqrt((sma * au_m) ** 3 / (gm_sun * mtot)) / 86400.0
    period_rel_err = abs(p_got / p_want - 1.0)
    assert abs(sma_back / sma - 1.0) < 1e-14

    ep = res["e"]["results"][0]
    a_mas = sma * plx
    t_peri = tau_ref + tau * p_got
    dra_want, ddec_want = cx_orb.position(epochs, p_got, t_peri, ecc, inc_deg, aop_deg, pan_deg, a_mas)
    vz_want = cx_orb.radial_velocity(epochs, p_got, t_peri, ecc, inc_deg, aop_deg, sma) * (au_m / 1000.0 / 86400.0)
    pos_err = float(
        max(np.max(np.abs(np.asarray(ep["raoff"]) - dra_want)), np.max(np.abs(np.asarray(ep["deoff"]) - ddec_want)))
        / a_mas
    )
    rv_err = float(np.max(np.abs(np.asarray(ep["vz"]) - vz_want)) / np.max(np.abs(vz_want)))

    record("max_pos_over_sma_err", pos_err)
    record("period_kepler_rel_err", period_rel_err)
    record("max_rv_rel_err", rv_err)
    assert period_rel_err < 1e-13
    assert pos_err < 1e-12
    assert rv_err < 1e-12


@pytest.mark.validates(
    "external_bridge.safe_load",
    roots=["standards"],
    kind="reference",
    headline="roundtrip_max_abs_err",
)
def test_safe_load_reconstructs_numpy_arrays_and_blocks_arbitrary_globals(tmp_path):
    """``safe_load.load_pickle`` and ``load_object_npy`` (format 1.0, 2.0, and 3.0 headers)
    reconstruct allowlisted NumPy arrays, OrderedDict, and scalar/container builtins while
    rejecting non-constructor globals across ``os``, ``builtins``, and ``numpy``."""
    from collections import OrderedDict
    import io
    import pickle
    from external_bridge import safe_load

    payload = {
        "u": np.linspace(-5.0, 5.0, 9),
        "meta": ("hr4796", 3, frozenset({"Band7"})),
        "order": OrderedDict([("a", np.float64(1.5)), ("b", np.int64(4))]),
    }
    pkl_path = tmp_path / "ok.pkl"
    pkl_path.write_bytes(pickle.dumps(payload))
    got_pkl = safe_load.load_pickle(pkl_path)
    err_pkl = float(np.max(np.abs(got_pkl["u"] - payload["u"])))
    assert got_pkl["meta"] == payload["meta"]
    assert list(got_pkl["order"].items()) == [("a", 1.5), ("b", 4)]

    obj_arr = np.empty(2, dtype=object)
    obj_arr[0] = np.array([1.25, -3.5, 4.0])
    obj_arr[1] = {"k": 7}
    err_npy = 0.0
    for ver, name in (((1, 0), "v1.npy"), ((2, 0), "v2.npy"), ((3, 0), "v3.npy")):
        npy_path = tmp_path / name
        with open(npy_path, "wb") as fh:
            np.lib.format.write_array(fh, obj_arr, version=ver, allow_pickle=True)
        got_npy = safe_load.load_object_npy(npy_path)
        err_npy = max(err_npy, float(np.max(np.abs(got_npy[0] - obj_arr[0]))))
        assert got_npy[1] == {"k": 7}

    # Non-object .npy and unsupported format versions fail closed with ValueError
    plain_npy = tmp_path / "plain.npy"
    np.save(plain_npy, np.arange(4.0))
    with pytest.raises(ValueError, match="object-dtype"):
        safe_load.load_object_npy(plain_npy)
    bad_ver_npy = tmp_path / "v9.npy"
    bad_ver_npy.write_bytes(b"\x93NUMPY\x09\x00\x00\x00")
    with pytest.raises(ValueError, match="unsupported .npy format"):
        safe_load.load_object_npy(bad_ver_npy)

    # Every non-constructor callable across os, builtins, and numpy must raise UnpicklingError
    forbidden_payloads = [
        b"cos\nsystem\n(S'echo nope'\ntR.",
        b"cbuiltins\ndelattr\n(S'x'\nS'y'\ntR.",
        b"cbuiltins\nglobals\n(tR.",
        b"cnumpy\nmemmap\n(S'nope.dat'\ntR.",
        b"cnumpy\nloadtxt\n(S'nope.txt'\ntR.",
    ]
    for idx, bad_bytes in enumerate(forbidden_payloads):
        bad_path = tmp_path / f"bad_{idx}.pkl"
        bad_path.write_bytes(bad_bytes)
        with pytest.raises(pickle.UnpicklingError, match="blocked global"):
            safe_load.load_pickle(bad_path)

        bad_npy = tmp_path / f"bad_{idx}.npy"
        buf = io.BytesIO()
        np.lib.format.write_array_header_1_0(buf, {"descr": "|O", "fortran_order": False, "shape": (1,)})
        bad_npy.write_bytes(buf.getvalue() + bad_bytes)
        with pytest.raises(pickle.UnpicklingError, match="blocked global"):
            safe_load.load_object_npy(bad_npy)

    worst = max(err_pkl, err_npy)
    record("roundtrip_max_abs_err", worst)
    assert worst == 0.0


@pytest.mark.validates(
    "virgil_bridge.alma_lambda_report",
    roots=["mathematics"],
    kind="reference",
    headline="blur_kernel_max_abs_err",
)
def test_alma_lambda_report_blur_moments_and_gate_checks_match_analytic_beam_and_thresholds():
    """``alma_lambda_report.blur`` convolves with the exact rotated elliptical Gaussian
    ``(4 ln 2 / (pi BMAJ BMIN)) dx^2 exp(-4 ln 2 ((da sin PA + dd cos PA)^2 / BMAJ^2 + ...))``,
    ``moments`` recovers centroid and PA from exact sky-plane second moments, and
    ``checks_of`` enforces the pre-registered thresholds of ``alma_lambda``."""
    from virgil_bridge import alma_lambda as vl, alma_lambda_report as rep

    npix = 96
    px = 10.0
    bmaj, bmin, pa_deg = 120.0, 65.0, 37.0
    r0 = npix // 2 - 1
    delta = np.zeros((npix, npix))
    delta[r0, r0] = 1.0
    got_blur = rep.blur(delta, px, (bmaj, bmin, pa_deg))

    rr, cc = np.indices((npix, npix))
    east = (r0 - cc) * px
    north = (r0 - rr) * px
    th = np.deg2rad(pa_deg)
    along = east * np.sin(th) + north * np.cos(th)
    across = east * np.cos(th) - north * np.sin(th)
    ln2 = np.log(2.0)
    want_blur = (
        (4.0 * ln2 / (np.pi * bmaj * bmin))
        * (px**2)
        * np.exp(-4.0 * ln2 * ((along / bmaj) ** 2 + (across / bmin) ** 2))
    )
    blur_err = float(np.max(np.abs(got_blur - want_blur)))

    # Shifted rotated elliptical Gaussian inside the 1.2 arcsec aperture for moments()
    e_true, n_true, pa_true = 45.0, -30.0, 128.0
    th2 = np.deg2rad(pa_true)
    de, dn = east - e_true, north - n_true
    u_maj = de * np.sin(th2) + dn * np.cos(th2)
    u_min = de * np.cos(th2) - dn * np.sin(th2)
    sky_gauss = np.exp(-0.5 * ((u_maj / 45.0) ** 2 + (u_min / 20.0) ** 2))
    e_got, n_got, pa_got = rep.moments(sky_gauss, px, radius_arcsec=1.2)
    mom_pos_err = float(max(abs(e_got - e_true), abs(n_got - n_true)))
    mom_pa_err = float(abs((pa_got - pa_true + 90.0) % 180.0 - 90.0))

    # Pre-registered gate checks_of() pass/fail boundaries
    w_grid = vl.grid(1.0)
    pts = [dict(weight=float(w), w_over_s2=float(w), chi2=1e5, m=float(10.0 - 5.0 * i), g=100.0, kkt_w=1e-5, band_limit=0.8) for i, w in enumerate(w_grid)]
    w_root = float(w_grid[2])
    rec_ok = dict(
        sigma=dict(sigma_rel=0.5 * vl.SIGMA_REL_MAX, ref_obs=1, sum_weights=[1.0, 4.0, 2.0], n_vis=1000),
        d4=dict(bias=1e-4, se_mean=1e-3, s2_injected=1.0, bias_in_se=0.1, bias_rel=1e-4),
        sweep=dict(
            s2=1.0, n_rows=1000, n_real=2000, points=pts, crossings=[w_root / 10.0, w_root],
            crossings_w_over_s2=[w_root / 10.0, w_root], weight=w_root, w_over_s2=w_root,
            m=0.5, g=100.0, g_se=1.0, n_secant=2, unconverged_fits=[],
        ),
        final=dict(
            kkt_w=0.5 * vl.KKT_W_MAX, band_limit=0.5 * vl.BAND_MAX,
            nuisances={"cal.gain": dict(edge_frac=[0.5, 0.2], at_edge=[False, False])},
        ),
    )
    checks_pass = rep.checks_of(rec_ok)
    assert all(c["ok"] is True for c in checks_pass)
    rec_bad = dict(rec_ok, final=dict(rec_ok["final"], kkt_w=1.5 * vl.KKT_W_MAX))
    assert [c["item"] for c in rep.checks_of(rec_bad) if c["ok"] is False] == ["6"]

    record("blur_kernel_max_abs_err", blur_err)
    record("moments_pos_max_err_mas", mom_pos_err)
    record("moments_pa_err_deg", mom_pa_err)
    assert blur_err < 1e-14
    assert mom_pos_err < 1e-10
    assert mom_pa_err < 1e-10


# ------------------------ 16. virgil_bridge (__init__.py)


@pytest.mark.validates(
    "virgil_bridge",
    roots=["mathematics"],
    kind="reference",
    headline="scene_vis_max_abs_err",
)
def test_virgil_bridge_scenes_analytic_visibilities_and_fit_scene_recovery(tmp_path):
    """Every ``virgil_bridge`` scene's ``vis(u, v, wavel)`` matches closed-form Bessel / Fourier /
    Kipping-Hankel integrals computed directly from SciPy, ``data.model(sc.template)`` agrees with
    OIFITS ``V^2`` and closure phases synthesized from ``sc.vis`` across all five scenes, and
    ``fit_scene`` recovers ``flat_truth`` with Laplace covariance matching the central-difference
    Gauss-Newton inverse Fisher matrix ``(J_FD^T W J_FD)^-1`` on a 3-station binary dataset."""
    from scipy.integrate import quad
    from scipy.special import j0, j1

    from crosscheck import array, simulate

    vb = pytest.importorskip("virgil_bridge")

    mas_rad = np.pi / (180.0 * 3600.0 * 1000.0)
    u = np.array([28.0, -64.0, 95.0, -112.0, 41.0])
    v = np.array([-35.0, 77.0, 48.0, -53.0, 108.0])
    wavel = np.full_like(u, 1.65e-6)
    qu = u / wavel
    qv = v / wavel

    # 1. binary(sep, pa, flux)
    sc_bin = vb.binary(sep=6.0, pa=124.0, flux=0.05)
    dra_b = 6.0 * np.sin(np.deg2rad(124.0)) * mas_rad
    ddec_b = 6.0 * np.cos(np.deg2rad(124.0)) * mas_rad
    want_bin = (1.0 + 0.05 * np.exp(-2j * np.pi * (qu * dra_b + qv * ddec_b))) / 1.05
    err_bin = float(np.max(np.abs(sc_bin.vis(u, v, wavel) - want_bin)))

    # 2. resolved_star_companion(diam, dra, ddec, flux)
    sc_ud = vb.resolved_star_companion(diam=1.8, dra=-12.0, ddec=8.0, flux=0.02)
    rho = np.hypot(qu, qv)
    x_ud = np.pi * (1.8 * mas_rad) * rho
    vis_ud = 2.0 * j1(x_ud) / x_ud
    vis_comp = np.exp(-2j * np.pi * (qu * (-12.0 * mas_rad) + qv * (8.0 * mas_rad)))
    want_ud = (vis_ud + 0.02 * vis_comp) / 1.02
    err_ud = float(np.max(np.abs(sc_ud.vis(u, v, wavel) - want_ud)))

    # 3. star_envelope(fwhm, ratio, pa, flux)
    sc_env = vb.star_envelope(fwhm=4.0, ratio=0.5, pa=60.0, flux=0.4)
    pa_e = np.deg2rad(60.0)
    q_maj = qu * np.sin(pa_e) + qv * np.cos(pa_e)
    q_min = qu * np.cos(pa_e) - qv * np.sin(pa_e)
    fwhm_maj = 4.0 * mas_rad
    fwhm_min = 0.5 * fwhm_maj
    vis_env = np.exp(-(np.pi**2) * ((fwhm_maj * q_maj) ** 2 + (fwhm_min * q_min) ** 2) / (4.0 * np.log(2.0)))
    want_env = (1.0 + 0.4 * vis_env) / 1.4
    err_env = float(np.max(np.abs(sc_env.vis(u, v, wavel) - want_env)))

    # 4. star_rim(diam, fwhm, inc, pa, amp, az_pa, flux): Jacobi-Anger J0 - i A cos(psi - phi0) J1 * deprojected blur
    sc_rim = vb.star_rim(diam=6.0, fwhm=1.0, inc=45.0, pa=30.0, amp=0.5, az_pa=120.0, flux=0.8)
    pa_r, inc_r = np.deg2rad(30.0), np.deg2rad(45.0)
    q_par = qu * np.sin(pa_r) + qv * np.cos(pa_r)
    q_perp = (qu * np.cos(pa_r) - qv * np.sin(pa_r)) * np.cos(inc_r)
    q_dep = np.hypot(q_par, q_perp)
    psi = np.arctan2(q_perp, q_par)
    phi0 = np.deg2rad(120.0 - 30.0)
    rad_rim = 3.0 * mas_rad
    fwhm_rim = 1.0 * mas_rad
    blur_rim = np.exp(-(np.pi * fwhm_rim * q_dep) ** 2 / (4.0 * np.log(2.0)))
    vis_ring = (j0(2.0 * np.pi * q_dep * rad_rim) - 1j * 0.5 * np.cos(psi - phi0) * j1(2.0 * np.pi * q_dep * rad_rim)) * blur_rim
    want_rim = (1.0 + 0.8 * vis_ring) / 1.8
    err_rim = float(np.max(np.abs(sc_rim.vis(u, v, wavel) - want_rim)))

    # 5. limb_darkened_star_companion: Kipping (2013) (q1, q2) -> (u1, u2) + continuous Hankel quad integral
    sc_ld = vb.limb_darkened_star_companion(diam=6.0, q1=0.36, q2=0.29, dra=0.7, ddec=-0.4, comp=(-14.0, 9.0), flux=0.03)
    u1 = 2.0 * np.sqrt(0.36) * 0.29
    u2 = np.sqrt(0.36) * (1.0 - 2.0 * 0.29)
    rad_ld = 3.0 * mas_rad
    norm_ld = quad(lambda r: r * (1.0 - u1 * (1.0 - np.sqrt(1.0 - r * r)) - u2 * (1.0 - np.sqrt(1.0 - r * r)) ** 2), 0.0, 1.0, epsabs=1e-14, epsrel=1e-14)[0]
    vis_ld_star = np.array([
        quad(
            lambda r, rq=rq_i: r
            * (1.0 - u1 * (1.0 - np.sqrt(1.0 - r * r)) - u2 * (1.0 - np.sqrt(1.0 - r * r)) ** 2)
            * j0(2.0 * np.pi * rq * rad_ld * r),
            0.0,
            1.0,
            epsabs=1e-14,
            epsrel=1e-14,
        )[0]
        / norm_ld
        for rq_i in rho
    ]) * np.exp(-2j * np.pi * (qu * (0.7 * mas_rad) + qv * (-0.4 * mas_rad)))
    vis_ld_comp = np.exp(-2j * np.pi * (qu * (-14.0 * mas_rad) + qv * (9.0 * mas_rad)))
    want_ld = (vis_ld_star + 0.03 * vis_ld_comp) / 1.03
    err_ld = float(np.max(np.abs(sc_ld.vis(u, v, wavel) - want_ld)))

    max_vis_err = max(err_bin, err_ud, err_env, err_rim, err_ld)

    # 6. Across all five scenes, OIFITS V^2 and closure phases synthesized from sc.vis match data.model(sc.template)
    uts = np.array([[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]])
    ha3 = [-2.0, 0.0, 2.0]
    wl3 = np.array([1.6e-6, 2.2e-6], dtype=np.float32).astype(float)
    sigma_v2 = 0.01
    sigma_cp_deg = 0.5
    tpl_errs = []
    for idx, sc in enumerate((sc_bin, sc_ud, sc_env, sc_rim, sc_ld)):
        sc_path = tmp_path / f"scene_{idx}.fits"
        simulate.observe(
            sc_path,
            sc.vis,
            uts,
            hour_angles_h=ha3,
            wavelengths=wl3,
            dec_deg=-50.0,
            sigma_v2=sigma_v2,
            sigma_cp_deg=sigma_cp_deg,
        )
        sc_data = vb.load(sc_path)
        obs_vec, _ = sc_data.flatten_data()
        mod_vec = sc_data.model(sc.template)
        tpl_errs.append(float(np.max(np.abs(np.asarray(obs_vec) - np.asarray(mod_vec)))))
    max_tpl_err = max(tpl_errs)

    # 7. On a 3-station (single independent closure triangle per snapshot) noiseless binary OIFITS file
    #    synthesized from sc_bin.vis, fit_scene recovers flat_truth and cov matches (J_FD^T W J_FD)^-1
    uts3 = uts[:3]
    bin3_path = tmp_path / "bin3.fits"
    v2_clean, cp_deg_clean = simulate.observe(
        bin3_path,
        sc_bin.vis,
        uts3,
        hour_angles_h=ha3,
        wavelengths=wl3,
        dec_deg=-50.0,
        sigma_v2=sigma_v2,
        sigma_cp_deg=sigma_cp_deg,
    )
    oidata3 = vb.load(bin3_path)
    fit_res, cov = vb.fit_scene(sc_bin, oidata3)
    truth_vec = vb.flat_truth(sc_bin)
    got_vec = vb.flat_values(sc_bin, fit_res.values)
    fit_err = float(np.max(np.abs(got_vec - truth_vec)))

    uv_snaps = [array.snapshot_uv(uts3, h, -50.0, -24.6276) for h in ha3]

    def _binary_obs_vec(theta):
        dra_m, ddec_m, flx = float(theta[0]), float(theta[1]), float(theta[2])
        v2_parts = []
        cp_parts = []
        for u_s, v_s in uv_snaps:
            def _vfn(uu, vv, _dra=dra_m, _ddec=ddec_m, _flx=flx):
                qu_s = uu[:, None] / wl3[None, :]
                qv_s = vv[:, None] / wl3[None, :]
                phase = -2.0 * np.pi * (qu_s * (_dra * mas_rad) + qv_s * (_ddec * mas_rad))
                return (1.0 + _flx * np.exp(1j * phase)) / (1.0 + _flx)

            v2_parts.append((np.abs(_vfn(u_s, v_s)) ** 2).ravel())
            cp_rad, *_ = array.closure_phase(_vfn, u_s, v_s, len(uts3))
            cp_parts.append(cp_rad.ravel())
        return np.concatenate([*v2_parts, *cp_parts])

    want_obs = np.concatenate([v2_clean.ravel(), np.deg2rad(cp_deg_clean).ravel()])
    np.testing.assert_allclose(_binary_obs_vec(truth_vec), want_obs, atol=1e-13)

    eps_fd = 1e-6
    jac = np.empty((want_obs.size, truth_vec.size), dtype=float)
    for j in range(truth_vec.size):
        step = np.zeros_like(truth_vec)
        step[j] = eps_fd
        jac[:, j] = (_binary_obs_vec(truth_vec + step) - _binary_obs_vec(truth_vec - step)) / (2.0 * eps_fd)
    n_v2 = v2_clean.size
    n_cp = cp_deg_clean.size
    weights = np.concatenate([
        np.full(n_v2, 1.0 / sigma_v2**2),
        np.full(n_cp, 1.0 / np.deg2rad(sigma_cp_deg) ** 2),
    ])
    cov_gn = np.linalg.inv(jac.T @ (weights[:, None] * jac))
    cov_rel_err = float(np.max(np.abs(cov - cov_gn)) / np.max(np.abs(cov_gn)))
    assert np.allclose(cov, cov_gn, rtol=1e-4)
    assert not np.allclose(cov, cov_gn[np.ix_([1, 0, 2], [1, 0, 2])], rtol=1e-2)

    record("scene_vis_max_abs_err", max_vis_err)
    record("scene_template_max_abs_err", max_tpl_err)
    record("fit_scene_max_param_err", fit_err)
    record("fit_scene_cov_rel_err", cov_rel_err)
    assert max_vis_err < 1e-12
    assert max_tpl_err < 1e-12
    assert fit_err < 1e-5
    assert cov_rel_err < 1e-4
