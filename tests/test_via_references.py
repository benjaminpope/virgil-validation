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

from crosscheck import (
    informed_starts as ist,
    orbit_population as opop,
    preregistration as prereg,
    radio_innovations as ri,
    radio_trust as rt,
    references as refs,
    skyplane as sp_sky,
    toeplitz_scenes as ts,
)
from evidence.plugin import record
from external_bridge import _subprocess as ext_sp, npz_compare as npzc


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
    roots=["mathematics", "statistics"],
    kind="reference",
    headline="ellipse_quadrature_abs_err",
)
def test_orbit_population_stream_isolation_elliptical_disk_and_band_laws(tmp_path):
    """``system_streams`` isolates parameter draws from noise draws, ``vis_elliptical_disk``
    matches 2D numerical quadrature over the sky ellipse and the circular Airy formula,
    ``band_of`` and ``p_cut`` satisfy their analytical definitions, and ``write_system``
    produces readable OIFITS files."""
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
    roots=["mathematics", "statistics"],
    kind="reference",
    headline="twin_rms_mas",
)
def test_skyplane_twin_invariance_point_reflection_and_verdict_rules():
    """The ``(Omega + 180, omega + 180)`` twin produces identical sky-plane positions
    (``SAME-SKY-ORBIT`` with samples, ``INCOMPLETE`` without), a single ``omega + 180``
    point reflection gives ``UNEXPLAINED-FLIP`` unless ``flipped`` or ``labels_free`` is set,
    and ``good_fit`` separates ``REFERENCE-DISAGREES`` from ``DIFFERENT`` via ``chi2.sf``."""
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

    record("lag1_white_noise_abs_bias", float(bias))
    record("lag1_white_noise_r1", float(r1))
    assert n_lag > 2000
    assert bias < 3.5 * se_r1


# ------------------------------------------------ 11. virgil_bridge.mpol


@pytest.mark.validates(
    "virgil_bridge.mpol",
    roots=["mathematics", "statistics"],
    kind="reference",
    headline="nll_vs_scipy_abs_err",
)
def test_virgil_bridge_mpol_bijection_parameter_map_and_exact_log_likelihood():
    """``pack``/``unpack`` and ``softplus``/``softplus_inverse`` are exact inverses,
    ``virgil_values`` maps MPoL parameters onto virgil sites (``cal.sigma = g * s_M`` vs
    ``s_M`` control), and ``neg_log_likelihood`` matches normalized ``scipy.stats.norm.logpdf``."""
    import jax
    import jax.numpy as jnp

    radio = pytest.importorskip("virgil.radio")
    from virgil_bridge import mpol as vbm

    rng = np.random.default_rng(44)
    sky_im = rng.uniform(0.05, 5.0, (8, 8))
    roundtrip_layout_err = float(np.max(np.abs(vbm.unpack(vbm.pack(sky_im)) - sky_im)))
    base = vbm.softplus_inverse(sky_im)
    softplus_err = float(np.max(np.abs(np.asarray(jax.nn.softplus(jnp.asarray(base))) - sky_im)))
    assert roundtrip_layout_err == 0.0
    assert softplus_err < 1e-13

    obs_map = vbm.obs_of_ddid({0: [0, 1], 1: [2], 2: [3]}, 4)
    assert list(obs_map) == [0, 0, 1, 2]
    pixel_mas = 5.0
    theta = {
        "base_cube": jnp.asarray(vbm.pack(base)),
        "log10_sigma": jnp.log10(jnp.array([1.1, 1.3, 0.9, 1.5])),
        "log10_amp": jnp.log10(jnp.array([1.08, 0.92])),
        "delta_ra": jnp.array([0.002, -0.003]),
        "delta_dec": jnp.array([-0.001, 0.004]),
    }
    vals = vbm.virgil_values(theta, obs_map, pixel_mas, sigma_map="g_times_s")
    vals_ctrl = vbm.virgil_values(theta, obs_map, pixel_mas, sigma_map="s_only")
    assert float(vals["cal.flux"]) == pytest.approx(float(sky_im.sum() * (pixel_mas / 1000.0) ** 2), rel=1e-13)
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
    scene = vbm.imlup_scene(8, pixel_mas)
    term = data.term(scene, nuis)
    got_nll = float(vbm.neg_log_likelihood(term, vals, data))
    mod_vis = np.asarray(data.model(scene(vals), vals, nuis))
    sig = np.asarray(vals["cal.sigma"])[ddid] / np.sqrt(w)
    want_nll = float(-(norm.logpdf(vis.real, mod_vis.real, sig).sum() + norm.logpdf(vis.imag, mod_vis.imag, sig).sum()))
    nll_err = abs(got_nll - want_nll)

    record("nll_vs_scipy_abs_err", nll_err)
    record("softplus_roundtrip_abs_err", softplus_err)
    assert nll_err < 1e-11


# ------------------------------------------------ 12. mpol (external package)


@pytest.mark.skipif(not ext_sp.available("mpol", "mpol"), reason="MPoL not installed (scripts/setup_external.sh mpol)")
@pytest.mark.external
@pytest.mark.validates(
    "mpol",
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
    headline="mpol_entropy_dft_rel_err",
)
def test_virgil_bridge_alma_lambda_scales_ring_geometry_and_mpol_entropy():
    """``lam_r`` and ``lam_f_equivalent`` satisfy their exact analytical scalings,
    ``ring_and_blob`` has unit sum and concentrates the blob in the expected sky quadrant,
    and ``mpol_entropy_jax`` matches an explicit NumPy discrete Fourier Gaussian blur plus
    Shannon relative entropy."""
    import jax.numpy as jnp

    pytest.importorskip("virgil.radio")
    from virgil_bridge import alma as vba

    assert vba.lam_r(50_000, lam_m=1e-9) == pytest.approx(1e-4, rel=1e-15)
    assert vba.lam_f_equivalent(1e-4, flux=0.5, pixel_mas=5.0, f_tot=0.25) == pytest.approx(
        1e-4 * 0.5 / (0.25 * (5.0 / 1000.0) ** 2), rel=1e-15
    )

    scene_im = vba.ring_and_blob(npix=33, pixel_mas=20.0, blob=(0.4, 140.0, -160.0, 40.0))
    assert scene_im.shape == (33, 33)
    assert np.all(scene_im >= 0.0)
    assert float(scene_im.sum()) == pytest.approx(1.0, rel=1e-14)
    # Blob at (+140 mas East -> col < 16, -160 mas North -> row > 16): SE quadrant brighter than NW
    assert float(scene_im[17:, :16].sum()) > float(scene_im[:16, 17:].sum())

    rng = np.random.default_rng(61)
    npix = 16
    pixel_mas, taper_mas, flux, prior, f_tot = 5.0, 20.0, 0.32, 1e-4, 0.25
    eta = rng.normal(0.0, 0.8, (npix, npix))
    got_ent = float(
        vba.mpol_entropy_jax(
            jnp.asarray(eta), flux, pixel_mas=pixel_mas, taper_mas=taper_mas, prior=prior, f_tot=f_tot
        )
    )
    dx_arcsec = pixel_mas / 1000.0
    arcsec_rad = np.pi / (180.0 * 3600.0)
    b = np.exp(eta - eta.max())
    b /= b.sum()
    sb = flux * b / dx_arcsec**2
    freq = np.fft.fftfreq(npix, d=dx_arcsec * arcsec_rad)
    q2 = freq[:, None] ** 2 + freq[None, :] ** 2
    taper_ft = np.exp(-((np.pi * (taper_mas / 1000.0) * arcsec_rad) ** 2) * q2 / (4.0 * np.log(2.0)))
    sb_tap = np.maximum(np.fft.ifft2(np.fft.fft2(sb) * taper_ft).real, 1e-30)
    want_ent = float(np.sum(sb_tap * np.log(sb_tap / prior)) / f_tot)
    ent_rel_err = abs(got_ent / want_ent - 1.0)

    record("mpol_entropy_dft_rel_err", ent_rel_err)
    record("ring_and_blob_sum_abs_err", abs(float(scene_im.sum()) - 1.0))
    assert ent_rel_err < 1e-13
