"""Validation of virgil 0.4.2 synthetic scenes, B3-spline starlet wavelets,
sky centroids, short-baseline field sizing, uv-lattice ``rotation_deg="grid"``,
and KKT optimality certificates against ``crosscheck.starlet_kkt_ref``.
"""

from __future__ import annotations

import dataclasses
import gc
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from scipy import optimize

from crosscheck import simulate, starlet_kkt_ref as ref
from evidence.plugin import record

vc = pytest.importorskip("virgil.coverage")
vf = pytest.importorskip("virgil.fields")
vfit = pytest.importorskip("virgil.fitting")
vi = pytest.importorskip("virgil.imaging")
vl = pytest.importorskip("virgil.likelihood")
vm = pytest.importorskip("virgil.models")
vo = pytest.importorskip("virgil.oidata")
vs = pytest.importorskip("virgil.scenes")

pytestmark = pytest.mark.x64


@pytest.fixture(scope="module", autouse=True)
def _clear_jax_caches():
    yield
    jax.clear_caches()
    gc.collect()


def _simple_v2_oidata(
    u_m: np.ndarray, v_m: np.ndarray, wavel_m: float, v2: np.ndarray, d_v2: np.ndarray
):
    """Build a minimal visibility-squared ``OIData`` without reading any file."""
    n = u_m.size
    return vo.OIData(
        {
            "u": np.asarray(u_m, dtype=float),
            "v": np.asarray(v_m, dtype=float),
            "wavel": np.full(n, float(wavel_m), dtype=float),
            "vis": np.asarray(v2, dtype=float),
            "d_vis": np.asarray(d_v2, dtype=float),
            "vis_mode": "v2",
        }
    )


# ---------------------------------------------------------------------------
# Reference self-checks (kind="reference" -> crosscheck.starlet_kkt_ref)
# Every helper in crosscheck.starlet_kkt_ref is verified here from first principles.
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "crosscheck.starlet_kkt_ref",
    roots=["mathematics"],
    kind="reference",
    headline="synthesis_err",
)
def test_ref_starlet_exact_synthesis_zero_pad_and_dirac_sum():
    rng = np.random.default_rng(42)
    img = rng.normal(size=(20, 20))
    details, coarse = ref.starlet_ref(img, scales=4, mode="constant")
    synth_err = float(np.max(np.abs(details.sum(axis=0) + coarse - img)))

    # Each 1D B3 à-trous filter has unit sum and palindromic symmetry
    max_filter_err = 0.0
    for j in range(5):
        h = ref.starlet_filter_1d(j)
        max_filter_err = max(
            max_filter_err,
            abs(float(np.sum(h)) - 1.0),
            float(np.max(np.abs(h - h[::-1]))),
        )

    # Interior impulse preserves total unit flux across coarse + detail planes
    impulse = np.zeros((33, 33), dtype=float)
    impulse[16, 16] = 1.0
    imp_det, imp_coarse = ref.starlet_ref(impulse, scales=3, mode="constant")
    flux_err = abs(float(np.sum(imp_det) + np.sum(imp_coarse)) - 1.0)

    record("synthesis_err", synth_err)
    record("filter_symmetry_err", max_filter_err)
    record("impulse_flux_err", flux_err)
    assert synth_err < 1e-15
    assert max_filter_err < 1e-15
    assert flux_err < 1e-15


@pytest.mark.validates(
    "crosscheck.starlet_kkt_ref",
    roots=["mathematics"],
    kind="reference",
    headline="laplacian_eig_err",
)
def test_ref_scenes_beams_fwhm_and_regularizer_hessians():
    # 1. Exact eigenvalues of 2D 5-point Dirichlet TSV Hessian + finite-difference MaxEntropy Hessian
    n = 6
    w, f0 = 2.3, 0.8
    hess = ref.tsv_dirichlet_hessian_ref(n, weight=w, reference_flux=f0)
    got_eigs = np.sort(np.linalg.eigvalsh(hess))
    ks = np.arange(1, n + 1, dtype=float)
    eig_1d = 4.0 * np.sin(0.5 * np.pi * ks / (n + 1)) ** 2
    want_eigs = np.sort((2.0 * w / (f0**2)) * (eig_1d[:, None] + eig_1d[None, :]).ravel())
    eig_err = float(np.max(np.abs(got_eigs - want_eigs)))

    rng = np.random.default_rng(12)
    f_me = rng.uniform(0.05, 0.4, size=8)
    m_me = rng.uniform(0.05, 0.4, size=8)

    def skilling_kl(f_vec):
        p = f_vec / f0
        return w * float(np.sum(p * np.log(p / m_me) - (p - m_me)))

    eps = 1e-3
    fd_diag = np.array(
        [
            (
                -skilling_kl(f_me + 2.0 * eps * np.eye(1, 8, k).ravel())
                + 16.0 * skilling_kl(f_me + eps * np.eye(1, 8, k).ravel())
                - 30.0 * skilling_kl(f_me)
                + 16.0 * skilling_kl(f_me - eps * np.eye(1, 8, k).ravel())
                - skilling_kl(f_me - 2.0 * eps * np.eye(1, 8, k).ravel())
            )
            / (12.0 * eps * eps)
            for k in range(8)
        ]
    )
    h_me = ref.maxent_hessian_ref(f_me, weight=w, reference_flux=f0)
    maxent_hess_err = float(np.max(np.abs(np.diag(h_me) - fd_diag)))

    # 2. Gaussian blob, inclined modulated ring, and Euclidean vs radial-cap Archimedean spiral
    blob = ref.gaussian_blob_ref(101, 0.5, sigma_mas=2.5, dra=1.7, ddec=-1.2)
    cent = ref.sky_centroid_ref(blob, 0.5, sigma_mas=2.0)
    centroid_err = float(np.max(np.abs(cent["centroid"] - np.array([1.7, -1.2]))))

    ring = ref.ring_ref(
        81, 0.75, radius_mas=14.0, width_mas=1.2, inc_deg=0.0, asymmetry=0.45, asymmetry_pa_deg=40.0
    )
    east_r, north_r = ref.sky_grid_mas(81, 0.75)
    pa_r = np.arctan2(east_r, north_r)
    dipole_est = 2.0 * float(np.sum(ring * np.cos(pa_r - np.deg2rad(40.0))))
    ring_dipole_err = abs(dipole_est - 0.45)

    sp_euc = ref.spiral_ref(
        81, 1.0, step_mas=10.0, width_mas=1.6, turns=2.5, pa_deg=15.0, distance_mode="euclidean"
    )
    sp_rad = ref.spiral_ref(
        81, 1.0, step_mas=10.0, width_mas=1.6, turns=2.5, pa_deg=15.0, distance_mode="radial_cap"
    )
    east_s, north_s = ref.sky_grid_mas(81, 1.0)
    r_s = np.hypot(east_s, north_s)
    # Outer turn (r > 18 mas, step/(2*pi*r) < 0.09) agrees closely; inner turn (r < 5 mas) differs
    outer_diff = float(np.max(np.abs(sp_euc[r_s > 18.0] - sp_rad[r_s > 18.0])) / np.max(sp_euc))
    inner_diff = float(np.max(np.abs(sp_euc[r_s < 5.0] - sp_rad[r_s < 5.0])) / np.max(sp_euc))

    # 3. convolve_elliptical_beam_ref second moments & v2_point_residual_dirty_map_ref finite-diff check
    imp = np.zeros((51, 51), dtype=float)
    imp[25, 25] = 1.0
    psf = ref.convolve_elliptical_beam_ref(imp, 0.5, 6.0, 3.0, 50.0, rotation_deg=15.0)
    x_l, y_u = ref.sky_grid_mas(51, 0.5, rotation_deg=0.0)
    ang = np.deg2rad(50.0 - 15.0)
    u_maj = x_l * np.sin(ang) + y_u * np.cos(ang)
    u_min = x_l * np.cos(ang) - y_u * np.sin(ang)
    beam_sig_err = max(
        abs(float(np.sqrt(np.sum(psf * u_maj**2))) - 6.0 / ref.FWHM_PER_SIGMA),
        abs(float(np.sqrt(np.sum(psf * u_min**2))) - 3.0 / ref.FWHM_PER_SIGMA),
        abs(float(np.sum(psf)) - 1.0),
    )

    u_v2 = np.array([15.0, -28.0, 42.0, 10.0, -35.0])
    v_v2 = np.array([22.0, 18.0, -12.0, -40.0, -25.0])
    lam = 2.2e-6
    v2_obs = np.array([0.92, 0.88, 0.81, 0.86, 0.79])
    sig_v2 = np.full(5, 0.02)
    rmap = ref.v2_point_residual_dirty_map_ref(u_v2, v_v2, lam, v2_obs, sig_v2, 7, 2.0, rotation_deg=-20.0)
    eg_m, ng_m = ref.sky_grid_mas(7, 2.0, rotation_deg=-20.0)

    def v2_res_vec(eps_f, r_i, c_i):
        ph = -2.0 * np.pi * (
            (u_v2 / lam) * eg_m[r_i, c_i] * ref.MAS_TO_RAD
            + (v_v2 / lam) * ng_m[r_i, c_i] * ref.MAS_TO_RAD
        )
        v_c = (1.0 + eps_f * np.exp(1j * ph)) / (1.0 + eps_f)
        return (np.abs(v_c) ** 2 - v2_obs) / sig_v2

    h_fd = 1e-7
    r_p = v2_res_vec(h_fd, 2, 5)
    r_m = v2_res_vec(-h_fd, 2, 5)
    g_fd = (float(r_p @ r_p) - float(r_m @ r_m)) / (2.0 * h_fd)
    j_fd = np.linalg.norm((r_p - r_m) / (2.0 * h_fd))
    rmap_fd_err = abs(float(rmap[2, 5]) - (-g_fd / j_fd))

    # 4. second_moment_fwhm_ref & image_field_ref exact analytical identities
    rho_test = np.array([4.0e6, 8.0e6, 1.2e7, 1.6e7])
    theta_rad = 4.2 * ref.MAS_TO_RAD
    v2_gauss = np.exp(-0.5 * (np.pi**2 / np.log(2.0)) * (theta_rad**2) * (rho_test**2))
    fwhm_no_int = ref.second_moment_fwhm_ref(rho_test, v2_gauss, np.full(4, 0.01), fit_intercept=False)
    fwhm_int = ref.second_moment_fwhm_ref(
        rho_test, 0.85 * v2_gauss, np.full(4, 0.01), fit_intercept=True
    )
    fwhm_ref_err = max(abs(float(fwhm_no_int) - 4.2), abs(float(fwhm_int) - 4.2))
    assert ref.second_moment_fwhm_ref(rho_test[:1], v2_gauss[:1], np.array([0.01])) is None
    assert ref.second_moment_fwhm_ref(rho_test, np.ones(4), np.full(4, 0.01)) == 0.0

    fov_ceil = ref.image_field_ref(1.0e6, 2.0e7, None, aperture_mas=35.0)
    fov_cross = ref.image_field_ref(1.0e6, 2.0e7, 5.0, inner_elements=500.0, aperture_mas=25.0)
    assert abs(fov_ceil - 35.0) < 1e-12 and abs(fov_cross - 25.0) < 1e-12

    record("laplacian_eig_err", eig_err)
    record("maxent_hess_fd_err", maxent_hess_err)
    record("blob_centroid_err", centroid_err)
    record("beam_sig_err", beam_sig_err)
    assert eig_err < 1e-13
    assert maxent_hess_err < 1e-6
    assert centroid_err < 1e-12
    assert ring_dipole_err < 1e-3
    assert outer_diff < 0.025 and inner_diff > 0.03
    assert beam_sig_err < 5e-3
    assert rmap_fd_err < 1e-7
    assert fwhm_ref_err < 1e-12


@pytest.mark.validates(
    "crosscheck.starlet_kkt_ref",
    roots=["mathematics", "statistics"],
    kind="reference",
    headline="rel_hutchinson_se_err",
)
def test_ref_hutchinson_variance_and_kkt_simplex_and_box_identities():
    rng = np.random.default_rng(7)
    d, m_data, nprobe = 14, 20, 16
    jac = rng.normal(scale=0.6, size=(m_data, d))
    k_mat = jac.T @ jac
    h_reg = np.diag(np.linspace(0.3, 2.1, d))
    exact = ref.ridge_dof_and_hutchinson_se_ref(k_mat, h_reg, nprobe=nprobe)

    estimates = [
        ref.hutchinson_rademacher_estimate(k_mat, h_reg, nprobe=nprobe, seed=1000 + s)[0]
        for s in range(400)
    ]
    emp_mean = float(np.mean(estimates))
    emp_sd = float(np.std(estimates, ddof=1))
    rel_se_err = abs(emp_sd - exact["se_rademacher"]) / exact["se_rademacher"]
    mean_z = abs(emp_mean - exact["g"]) / (exact["se_rademacher"] / np.sqrt(400.0))

    # 1. Simplex KKT residual vanishes at SciPy SLSQP optimum of strictly convex entropy objective
    a_mat = rng.normal(size=(10, 6))
    y_vec = rng.normal(size=10)
    w_ent = 0.25

    def gd_fn(b):
        return a_mat.T @ (a_mat @ b - y_vec)

    def gp_fn(b):
        return w_ent * (np.log(np.clip(b, 1e-30, None)) + 1.0)

    sol = optimize.minimize(
        lambda b: 0.5 * float(np.sum((a_mat @ b - y_vec) ** 2))
        + w_ent * float(np.sum(b * np.log(np.clip(b, 1e-30, None)))),
        np.full(6, 1.0 / 6.0),
        jac=lambda b: gd_fn(b) + gp_fn(b),
        method="SLSQP",
        bounds=[(1e-12, 1.0)] * 6,
        constraints=[{"type": "eq", "fun": lambda b: np.sum(b) - 1.0}],
        options={"ftol": 1e-14, "maxiter": 500},
    )
    assert sol.success
    kkt = ref.kkt_simplex_metrics_ref(sol.x, gd_fn(sol.x), gp_fn(sol.x), flux=0.4, weight=w_ent)
    b0_s = np.full(6, 1.0 / 6.0)
    kkt_with_start = ref.kkt_simplex_metrics_ref(
        sol.x,
        gd_fn(sol.x),
        gp_fn(sol.x),
        flux=0.4,
        weight=w_ent,
        start_brightness=b0_s,
        start_grad_data_b=gd_fn(b0_s) * 8.0,
        start_grad_pen_b=gp_fn(b0_s) * 8.0,
    )
    assert kkt_with_start["scale"] > kkt["scale"]
    assert kkt_with_start["kkt_rel"] < kkt["kkt_rel"]

    # 2. Square-root reparameterized KKT metrics + exact x_i == 0 exclusion + start_sqrt_pixels scale
    x_sq = np.array([0.0, 1e-5, 0.3, 0.5, 0.4, 0.2])
    gd_sq = np.array([-20.0, -20.0, 0.2, -0.1, 0.15, -0.05])
    gp_sq = np.array([0.0, 0.0, -0.2, 0.1, -0.15, 0.05])
    sq_rep = ref.kkt_sqrt_metrics_ref(x_sq, gd_sq, gp_sq, weight=0.5, reach_mode="flux")
    sq_rep_sqrt = ref.kkt_sqrt_metrics_ref(x_sq, gd_sq, gp_sq, weight=0.5, reach_mode="sqrt")
    # x_sq[0] == 0 is excluded from want_grow; x_sq[1] = 1e-5 > 0 violates reach=1 (-> 1/6)
    # while reach=sqrt(0.25 / 1e-10) = 50,000 suppresses it to 0.0
    assert abs(sq_rep["want_grow"] - 1.0 / 6.0) < 1e-15
    assert sq_rep_sqrt["want_grow"] == 0.0
    sq_start = ref.kkt_sqrt_metrics_ref(
        x_sq,
        gd_sq,
        gp_sq,
        weight=0.5,
        start_sqrt_pixels=np.full(6, 0.4),
        start_grad_data_f=np.full(6, 10.0),
        start_grad_pen_f=np.zeros(6),
    )
    assert abs(sq_start["scale"] - 10.0) < 1e-14

    # 3. Box-constrained f >= 0 KKT metrics vanish at SciPy L-BFGS-B optimum with active zeros,
    #    incorporate start_grad_total when supplied, and flag zero_violation > 0 on clamped peak
    q_mat = a_mat.T @ a_mat + 0.5 * np.eye(6)
    c_vec = np.array([-1.2, 0.8, -0.5, 1.5, 2.0, -0.9])
    sol_box = optimize.minimize(
        lambda f: 0.5 * float(f @ q_mat @ f) + float(c_vec @ f),
        np.full(6, 0.2),
        jac=lambda f: q_mat @ f + c_vec,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * 6,
        options={"ftol": 0.0, "gtol": 1e-14, "maxiter": 1000},
    )
    assert sol_box.success and int(np.sum(sol_box.x == 0.0)) >= 2
    box_ok = ref.kkt_nonneg_linear_metrics_ref(
        sol_box.x, q_mat @ sol_box.x, c_vec, weight=0.5
    )
    clamped = sol_box.x.copy()
    clamped[int(np.argmax(clamped))] = 0.0
    box_bad = ref.kkt_nonneg_linear_metrics_ref(
        clamped, q_mat @ clamped, c_vec, weight=0.5
    )
    box_bad_start = ref.kkt_nonneg_linear_metrics_ref(
        clamped, q_mat @ clamped, c_vec, weight=0.5, start_grad_total=np.full(6, 25.0)
    )
    assert abs(box_bad_start["pg_rel"] - box_bad["pg_inf"] / 25.0) < 1e-15

    # 4. Nuisance projected-gradient ratio + MaxEntropy Fourier band_limit reference
    nuis_unbounded = ref.kkt_nuisance_grad_rel_ref(np.array([0.03, -0.04]), np.array([0.2, -0.8]))
    assert abs(nuis_unbounded["grad_rel"] - 0.04 / 0.8) < 1e-15
    nuis_active_bound = ref.kkt_nuisance_grad_rel_ref(
        np.array([2.5, -0.02]),
        np.array([-1.0, 0.5]),
        value_cur=np.array([0.0, 0.4]),
        value_start=np.array([0.2, 0.4]),
        lower=0.0,
        upper=1.0,
    )
    assert abs(nuis_active_bound["grad_norm"] - 0.02) < 1e-15
    nuis_interior_large = ref.kkt_nuisance_grad_rel_ref(
        1e-4, 1200.0, value_cur=0.22, value_start=0.23, lower=0.01, upper=1.0
    )
    assert abs(nuis_interior_large["grad_rel"] - 1e-4 / 1200.0) < 1e-20

    rr, cc = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    eta_band = 0.3 * np.cos(2.0 * np.pi * rr / 16.0) + 0.2 * np.sin(2.0 * np.pi * cc / 16.0)
    bl_clean = ref.maxent_band_limit_ref(
        np.exp(eta_band), np.array([40.0]), np.array([0.0]), 2.2e-6, pixel_scale_mas=0.8
    )
    bl_ripple = ref.maxent_band_limit_ref(
        np.exp(eta_band + 0.8 * ((-1.0) ** (rr + cc))),
        np.array([40.0]),
        np.array([0.0]),
        2.2e-6,
        pixel_scale_mas=0.8,
    )
    bl_coarse = ref.maxent_band_limit_ref(
        np.exp(eta_band), np.array([120.0]), np.array([0.0]), 1.65e-6, pixel_scale_mas=4.0
    )
    assert abs(bl_clean - 1.0) < 1e-13
    assert bl_ripple > 10.0
    assert np.isnan(bl_coarse)

    record("rel_hutchinson_se_err", rel_se_err)
    record("hutchinson_mean_z", mean_z)
    record("simplex_kkt_rel", kkt["kkt_rel"])
    record("box_opt_pg_rel", box_ok["pg_rel"])
    assert rel_se_err < 0.12
    assert mean_z < 3.0
    assert kkt["kkt_rel"] < 1e-7
    assert kkt["want_grow"] == 0.0
    assert box_ok["pg_rel"] < 1e-8
    assert box_ok["kkt_w"] < 1e-8
    assert box_ok["zero_violation"] == 0.0
    assert box_bad["zero_violation"] > 0.0


@pytest.mark.validates(
    "crosscheck.starlet_kkt_ref",
    roots=["mathematics", "statistics"],
    kind="reference",
    headline="laplace_quad_err",
)
def test_ref_laplace_lcurve_classic_maxent_mythra_and_radio_differencing():
    # 1. mackay_laplace_gp_ref against exact 2D numerical Gaussian integral + fixed-point residual
    j_z = np.array([[1.4], [-0.8], [0.5], [1.1]])
    j_ext = np.array([[0.3], [0.7], [-0.4], [0.2]])
    r0 = np.array([0.6, -0.4, 0.9, -0.3])
    z0 = np.array([0.25])
    # At linear MAP (z0, theta0=0): J_joint^T r + [z0, 0] == 0 if we adjust r_map orthogonal to J_joint
    # Even simpler: for linear Gaussian log p(y, z, theta) = -0.5 ||r0 + J_z dz + J_ext dtheta||^2 - 0.5 ||z0 + dz||^2 + lp_ext,
    # when (z0, 0) IS the exact stationary point (J_z^T r0 + z0 == 0 and J_ext^T r0 == 0), Laplace is EXACT!
    # Construct r0 in the nullspace of J_ext^T with J_z^T r0 = -z0:
    # Solve [J_z^T; J_ext^T] r_map = [-z0[0], 0.0] via minimum-norm + nullspace component!
    a_sys = np.vstack([j_z.T, j_ext.T])
    r_map = np.linalg.lstsq(a_sys, np.array([-z0[0], 0.0]), rcond=None)[0]
    u_null = np.eye(4) - a_sys.T @ np.linalg.inv(a_sys @ a_sys.T) @ a_sys
    r_map = r_map + u_null @ np.array([1.2, -0.9, 0.7, -0.5])
    assert abs(float((j_z.T @ r_map)[0] + z0[0])) < 1e-14
    assert abs(float((j_ext.T @ r_map)[0])) < 1e-14

    mk = ref.mackay_laplace_gp_ref(
        r_map,
        0.0,
        z0,
        j_z,
        n_independent=4,
        extra_jac=j_ext,
        extra_log_prior=-1.2,
        observable_blocks={"a": (slice(0, 2), 2), "b": (slice(2, 4), 2)},
    )
    # 2D midpoint quadrature over (dz, dtheta) on [-9, 9]^2 (exact for linear Gaussian!)
    grid_1d = np.linspace(-9.0, 9.0, 361)
    step = float(grid_1d[1] - grid_1d[0])
    dz_g, dt_g = np.meshgrid(grid_1d, grid_1d, indexing="ij")
    res_g = r_map[:, None, None] + j_z[:, 0][:, None, None] * dz_g[None] + j_ext[:, 0][:, None, None] * dt_g[None]
    log_integrand = (
        -0.5 * np.sum(res_g**2, axis=0)
        - 0.5 * (z0[0] + dz_g) ** 2
        - 0.5 * np.log(2.0 * np.pi)
        - 1.2
    )
    m_max = float(np.max(log_integrand))
    logz_quad = m_max + float(np.log(np.sum(np.exp(log_integrand - m_max)) * step * step))
    laplace_quad_err = abs(float(mk["log_evidence"]) - logz_quad)

    # Check scalar and block fixed-point residuals
    s_fp = float(mk["s_fixed_point"])
    sv = np.linalg.svdvals(j_z)
    fp_res = abs(s_fp**2 * (4.0 - float(np.sum((sv / s_fp) ** 2 / (1.0 + (sv / s_fp) ** 2)))) - float(mk["chi2"]))
    s_obs = mk["s_by_observable"]
    assert isinstance(s_obs, dict) and set(s_obs.keys()) == {"a", "b"}

    # 2. lcurve_corner_ref on symmetric log-log hyperbola (chi2 grows with w, penalty falls with w, max curvature at w = 1.0)
    t_grid = np.linspace(3.0, -3.0, 13)
    w_grid = np.exp(t_grid)
    c2_sym = np.exp(t_grid + np.sqrt(t_grid**2 + 0.5))
    pen_sym = np.exp(-t_grid + np.sqrt(t_grid**2 + 0.5))
    assert abs(ref.lcurve_corner_ref(w_grid, c2_sym, pen_sym) - 1.0) < 1e-14

    # 3. classic_maxent_weight_ref on analytical 2-pixel identity model:
    # b = [0.5, 0.5], J_b = sqrt(2) * I_2 => rt_b * J_b = I_2 => eigvals = [1, 1], n_good(w) = 2 / (1 + w).
    # With KL = 1 / (w * (1 + w))? Even better: constant KL = 1/3 => 2 w (1/3) = 2 / (1 + w) => w (1 + w) = 3
    w_me = np.geomspace(10.0, 0.2, 41)
    b_half = [np.array([0.5, 0.5])] * 41
    j_id = [np.sqrt(2.0) * np.eye(2)] * 41
    w_root_true = 0.5 * (-1.0 + np.sqrt(13.0))
    w_got_me = ref.classic_maxent_weight_ref(w_me, np.full(41, 1.0 / 3.0), b_half, j_id)
    classic_err = abs(float(w_got_me) - w_root_true)

    # 4. mythra_filter_indices_ref deterministic rejection cascade
    chi2_mat = np.array(
        [
            [1.00, 1.02],  # 0: kept (best)
            [1.03, 1.01],  # 1: kept
            [1.01, 1.04],  # 2: kept
            [0.98, 0.99],  # 3: outside window -> "window"
            [np.nan, 1.0], # 4: non-finite -> "diverged"
            [2.40, 1.05],  # 5: > 2.0 * best on dataset 0 -> "chi2"
            [1.55, 1.55],  # 6: < 2.0 * best, but > 5 MAD above {1.01, 1.02, 1.025} -> "outlier"
        ]
    )
    in_win = np.array([True, True, True, False, True, True, True])
    alive, reasons = ref.mythra_filter_indices_ref(chi2_mat, in_win, np.array([20.0, 20.0]))
    assert alive == [0, 1, 2]
    assert reasons == [None, None, None, "window", "diverged", "chi2", "outlier"]

    # 5. radio_sigma_scale_group_ref Monte Carlo SE verification
    rng = np.random.default_rng(55)
    t_s = np.arange(80, dtype=float) * 5.0
    w_s = np.full(80, 4.0e4)
    s_true = 1.35
    s_hats, s_ses = [], []
    for _ in range(300):
        v_s = 0.5 + (s_true / np.sqrt(w_s)) * (rng.normal(size=80) + 1j * rng.normal(size=80))
        out = ref.radio_sigma_scale_group_ref(
            t_s, np.zeros(80, int), np.ones(80, int), np.zeros(80, int), np.ones(80, int), np.zeros(80, int), v_s, w_s
        )
        s_hats.append(float(out["s"]))
        s_ses.append(float(out["s_se"]))
    mc_sd = float(np.std(s_hats, ddof=1))
    mean_se = float(np.mean(s_ses))
    radio_se_rel_err = abs(mc_sd - mean_se) / mc_sd

    # 6. skilling_kl_ref, dof_weight_band_ref, and short_baseline_v2_residual_ref exact identities
    assert abs(ref.skilling_kl_ref(np.full((4, 4), 0.25 / 16.0), reference_flux=0.25)) < 1e-15
    kl_two = ref.skilling_kl_ref(np.array([0.3, 0.9]), reference_flux=1.0)
    want_kl_two = (0.3 * np.log(0.3 / 0.5) - (0.3 - 0.5)) + (0.9 * np.log(0.9 / 0.5) - (0.9 - 0.5))
    assert abs(kl_two - want_kl_two) < 1e-15

    w_sweep = np.geomspace(100.0, 0.01, 21)
    # Linear in ln w: m(w; s^2) = (20 + ln w) / s^2 - 20 = 0 => ln w = 20 * (s^2 - 1)
    w_hi, w_lo = ref.dof_weight_band_ref(
        w_sweep,
        20.0 + np.log(w_sweep),
        np.full_like(w_sweep, 4.0),
        24.0,
        noise_scale2=1.0,
        noise_scale_sigma=0.05,
    )
    assert w_hi is not None and w_lo is not None
    assert abs(w_hi - float(np.exp(2.0))) < 1e-13 and abs(w_lo - float(np.exp(-2.0))) < 1e-13

    sb_m, sb_s = ref.short_baseline_v2_residual_ref(
        np.array([10.0, 20.0, 50.0, 80.0]),
        np.zeros(4),
        2.0e-6,
        np.array([0.90, 0.80, 0.50, 0.20]),
        np.full(4, 0.05),
        np.sqrt(np.array([0.95, 0.75, 0.50, 0.20])) + 0j,
        quantile=0.5,
    )
    assert abs(sb_m - 0.0) < 1e-14 and abs(sb_s - 1.0 / np.sqrt(2.0)) < 1e-14

    record("laplace_quad_err", laplace_quad_err)
    record("fixed_point_res", fp_res)
    record("classic_maxent_root_err", classic_err)
    record("radio_se_rel_err", radio_se_rel_err)
    assert laplace_quad_err < 1e-12
    assert fp_res < 1e-12
    assert classic_err < 2e-3
    assert radio_se_rel_err < 0.12


# ---------------------------------------------------------------------------
# virgil.scenes: gaussian_blob, ring, spiral
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.scenes.gaussian_blob",
    roots=["mathematics"],
    headline="max_pixel_err",
)
def test_gaussian_blob_matches_analytical_sky_grid_and_moments():
    npix, scale = 24, 1.5
    got = np.asarray(vs.gaussian_blob(npix, scale, sigma_mas=3.2, dra=2.1, ddec=-1.8))
    want = ref.gaussian_blob_ref(npix, scale, sigma_mas=3.2, dra=2.1, ddec=-1.8)
    max_err = float(np.max(np.abs(got - want)))
    sum_err = abs(float(np.sum(got)) - 1.0)

    peak_row, peak_col = np.unravel_index(int(np.argmax(got)), got.shape)
    cent = ref.sky_centroid_ref(got, scale, sigma_mas=1.0)
    cent_err = float(np.hypot(cent["centroid"][0] - 2.1, cent["centroid"][1] - (-1.8)))

    record("max_pixel_err", max_err)
    record("sum_err", sum_err)
    record("centroid_err_mas", cent_err)
    assert max_err < 1e-14
    assert sum_err < 1e-14
    assert peak_col < (npix - 1) / 2.0
    assert peak_row > (npix - 1) / 2.0
    assert cent_err < 0.05


@pytest.mark.validates(
    "virgil.scenes.gaussian_blob",
    roots=["mathematics"],
    kind="control",
    headline="flipped_dra_err",
)
def test_gaussian_blob_flipped_east_west_control():
    got = np.asarray(vs.gaussian_blob(24, 1.5, sigma_mas=3.2, dra=2.4, ddec=1.5))
    wrong = ref.gaussian_blob_ref(24, 1.5, sigma_mas=3.2, dra=-2.4, ddec=1.5)
    diff = float(np.max(np.abs(got - wrong)))
    record("flipped_dra_err", diff)
    assert diff > 1e-2


@pytest.mark.validates(
    "virgil.scenes.ring",
    roots=["mathematics"],
    headline="max_pixel_err",
)
def test_ring_matches_inclined_modulated_sky_reference():
    npix, scale = 28, 2.0
    got = np.asarray(
        vs.ring(
            npix,
            scale,
            radius_mas=14.0,
            width_mas=2.4,
            inc_deg=55.0,
            pa_deg=35.0,
            asymmetry=0.6,
            asymmetry_pa_deg=110.0,
        )
    )
    want = ref.ring_ref(
        npix,
        scale,
        radius_mas=14.0,
        width_mas=2.4,
        inc_deg=55.0,
        pa_deg=35.0,
        asymmetry=0.6,
        asymmetry_pa_deg=110.0,
    )
    max_err = float(np.max(np.abs(got - want)))

    sym = np.asarray(vs.ring(41, 1.5, radius_mas=15.0, width_mas=2.0, inc_deg=50.0, pa_deg=28.0))
    east, north = ref.sky_grid_mas(41, 1.5)
    cov = np.array(
        [
            [np.sum(sym * east * east), np.sum(sym * east * north)],
            [np.sum(sym * east * north), np.sum(sym * north * north)],
        ]
    )
    evals, evecs = np.linalg.eigh(cov)
    axis_ratio_err = abs(float(np.sqrt(evals[0] / evals[1])) - float(np.cos(np.deg2rad(50.0))))
    pa_est = float(np.rad2deg(np.arctan2(evecs[0, 1], evecs[1, 1])) % 180.0)
    pa_err_deg = abs(pa_est - 28.0)

    record("max_pixel_err", max_err)
    record("axis_ratio_err", axis_ratio_err)
    record("pa_err_deg", pa_err_deg)
    assert max_err < 1e-14
    assert abs(float(np.sum(got)) - 1.0) < 1e-14
    assert axis_ratio_err < 0.015
    assert pa_err_deg < 0.2


@pytest.mark.validates(
    "virgil.scenes.ring",
    roots=["mathematics"],
    kind="control",
    headline="inplane_vs_sky_asym_diff",
)
def test_ring_in_plane_vs_on_sky_asymmetry_control():
    npix, scale = 28, 2.0
    got = np.asarray(
        vs.ring(
            npix,
            scale,
            radius_mas=14.0,
            width_mas=2.4,
            inc_deg=62.0,
            pa_deg=20.0,
            asymmetry=0.7,
            asymmetry_pa_deg=65.0,
        )
    )
    east, north = ref.sky_grid_mas(npix, scale)
    pa = np.deg2rad(20.0)
    along = east * np.sin(pa) + north * np.cos(pa)
    across_deproj = (east * np.cos(pa) - north * np.sin(pa)) / np.cos(np.deg2rad(62.0))
    r_deproj = np.hypot(along, across_deproj)
    phi_inplane = np.arctan2(across_deproj, along)
    wrong = np.exp(-0.5 * ((r_deproj - 14.0) / 2.4) ** 2) * (
        1.0 + 0.7 * np.cos(phi_inplane - np.deg2rad(65.0 - 20.0))
    )
    wrong = wrong / np.sum(wrong)
    diff = float(np.max(np.abs(got - wrong)))
    record("inplane_vs_sky_asym_diff", diff)
    assert diff > 1e-3


@pytest.mark.validates(
    "virgil.scenes.spiral",
    roots=["mathematics"],
    property="arm_cross_section",
    headline="fade_ratio_rel_err",
)
def test_spiral_width_is_sigma_ccw_winding_rounded_cap_and_radial_fade():
    npix, scale = 33, 2.0
    no_fade = np.asarray(
        vs.spiral(npix, scale, step_mas=16.0, width_mas=2.2, turns=1.5, pa_deg=25.0, fade_mas=None)
    )
    with_fade = np.asarray(
        vs.spiral(npix, scale, step_mas=16.0, width_mas=2.2, turns=1.5, pa_deg=25.0, fade_mas=18.0)
    )
    # Compare against BOTH independent arm-distance readings per AGENTS.md:
    # 1. Exact Euclidean distance to the Archimedean curve (agrees to ~2% on outer turns where
    #    step/(2*pi*r) << 1, while inner turns narrow by 1/sqrt(1 + (step/(2*pi*r))^2));
    # 2. Ray-radial cross-section |r - r_arm| with endpoint Euclidean caps (virgil's exact choice).
    want_euclidean = ref.spiral_ref(
        npix, scale, step_mas=16.0, width_mas=2.2, turns=1.5, pa_deg=25.0, fade_mas=18.0, distance_mode="euclidean"
    )
    want_radial = ref.spiral_ref(
        npix, scale, step_mas=16.0, width_mas=2.2, turns=1.5, pa_deg=25.0, fade_mas=18.0, distance_mode="radial_cap"
    )
    max_pixel_err = float(np.max(np.abs(with_fade - want_radial)))

    east, north = ref.sky_grid_mas(npix, scale)
    r_sky = np.hypot(east, north)
    # Renormalize outer annulus (r >= 16 mas) to compare local arm cross-section shape directly
    outer = r_sky >= 16.0
    got_outer = no_fade[outer] / np.sum(no_fade[outer])
    euc_no_fade = ref.spiral_ref(
        npix, scale, step_mas=16.0, width_mas=2.2, turns=1.5, pa_deg=25.0, fade_mas=None, distance_mode="euclidean"
    )
    euc_outer = euc_no_fade[outer] / np.sum(euc_no_fade[outer])
    outer_euclidean_rel_diff = float(np.max(np.abs(got_outer - euc_outer)) / np.max(got_outer))

    predicted_faded = no_fade * np.exp(-r_sky / 18.0)
    predicted_faded /= np.sum(predicted_faded)
    fade_rel_err = float(np.max(np.abs(with_fade - predicted_faded)) / np.max(with_fade))

    # Verify width_mas is Gaussian sigma (not FWHM) on an outer crossing ray
    sp_south = np.asarray(vs.spiral(41, 1.0, step_mas=24.0, width_mas=2.5, turns=1.0, pa_deg=0.0))
    ratio_2mas = float(sp_south[34, 20] / sp_south[32, 20])
    want_sigma_ratio = float(np.exp(-0.5 * (2.0 / 2.5) ** 2))

    # Rounded cap at arm end (turns=1.0, pa_deg=0 -> end at North ddec=+20 mas, row=12, col=32)
    sp_cap = np.asarray(vs.spiral(65, 1.0, step_mas=20.0, width_mas=2.5, turns=1.0, pa_deg=0.0))
    cap_ratio = float(sp_cap[10, 32] / sp_cap[12, 32])

    record("max_pixel_err", max_pixel_err)
    record("outer_euclidean_rel_diff", outer_euclidean_rel_diff)
    record("fade_ratio_rel_err", fade_rel_err)
    record("sigma_profile_err", abs(ratio_2mas - want_sigma_ratio))
    record("cap_profile_err", abs(cap_ratio - want_sigma_ratio))
    assert abs(float(np.sum(with_fade)) - 1.0) < 1e-14
    assert float(np.max(np.abs(with_fade - want_euclidean))) > 1e-3  # inner radial-vs-normal difference is measurable
    assert outer_euclidean_rel_diff < 0.04
    assert fade_rel_err < 1e-13
    assert abs(ratio_2mas - want_sigma_ratio) < 0.03
    assert abs(cap_ratio - want_sigma_ratio) < 0.05
    assert max_pixel_err < 1e-14


@pytest.mark.validates(
    "virgil.scenes.spiral",
    roots=["mathematics"],
    kind="control",
    property="arm_cross_section",
    headline="fwhm_misinterpretation_err",
)
def test_spiral_width_fwhm_and_reversed_winding_controls():
    got = np.asarray(vs.spiral(33, 2.0, step_mas=18.0, width_mas=3.0, turns=1.5, pa_deg=15.0))
    wrong_fwhm = ref.spiral_ref(
        33,
        2.0,
        step_mas=18.0,
        width_mas=3.0 / ref.FWHM_PER_SIGMA,
        turns=1.5,
        pa_deg=15.0,
        distance_mode="radial_cap",
    )
    wrong_euclidean = ref.spiral_ref(
        33,
        2.0,
        step_mas=18.0,
        width_mas=3.0,
        turns=1.5,
        pa_deg=15.0,
        distance_mode="euclidean",
    )
    east, north = ref.sky_grid_mas(33, 2.0)
    inner = (np.hypot(east, north) >= 2.0) & (np.hypot(east, north) <= 8.0)
    inner_euc_diff = float(np.max(np.abs(got[inner] - wrong_euclidean[inner])) / np.max(got[inner]))

    got0 = np.asarray(vs.spiral(33, 2.0, step_mas=18.0, width_mas=3.0, turns=1.25, pa_deg=0.0))
    wrong_cw = got0[:, ::-1]
    fwhm_diff = float(np.max(np.abs(got - wrong_fwhm)) / np.max(got))
    cw_diff = float(np.max(np.abs(got0 - wrong_cw)) / np.max(got0))

    record("fwhm_misinterpretation_err", fwhm_diff)
    record("inner_euclidean_vs_radial_err", inner_euc_diff)
    record("cw_winding_err", cw_diff)
    assert fwhm_diff > 0.25
    assert inner_euc_diff > 0.08
    assert cw_diff > 0.5


# ---------------------------------------------------------------------------
# virgil.imaging F22-free primitives: starlet, Centroid, image_priors,
# second_moment_fwhm, image_field, Beam
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.imaging.starlet",
    roots=["mathematics"],
    headline="max_detail_err",
)
def test_starlet_exact_reconstruction_linearity_and_zero_padded_edges():
    rng = np.random.default_rng(19)
    x = rng.normal(loc=1.5, scale=1.0, size=(20, 24))
    y = rng.normal(loc=-0.5, scale=0.8, size=(20, 24))

    got_det, got_coarse = (np.asarray(a) for a in vi.starlet(x, scales=4))
    want_det, want_coarse = ref.starlet_ref(x, scales=4, mode="constant")
    max_det_err = float(np.max(np.abs(got_det - want_det)))
    max_coarse_err = float(np.max(np.abs(got_coarse - want_coarse)))
    recon_err = float(np.max(np.abs(got_det.sum(axis=0) + got_coarse - x)))

    det_y, coarse_y = (np.asarray(a) for a in vi.starlet(y, scales=4))
    det_lin, coarse_lin = (np.asarray(a) for a in vi.starlet(1.7 * x - 0.9 * y, scales=4))
    lin_err = max(
        float(np.max(np.abs(det_lin - (1.7 * got_det - 0.9 * det_y)))),
        float(np.max(np.abs(coarse_lin - (1.7 * got_coarse - 0.9 * coarse_y)))),
    )

    imp = np.zeros((33, 33))
    imp[16, 16] = 1.0
    imp_det, _ = (np.asarray(a) for a in vi.starlet(imp, scales=4))
    broad = ref.gaussian_blob_ref(33, 1.0, sigma_mas=3.5)
    broad_det, _ = (np.asarray(a) for a in vi.starlet(broad, scales=4))
    imp_energies = np.sum(imp_det**2, axis=(1, 2))
    broad_energies = np.sum(broad_det**2, axis=(1, 2))

    record("max_detail_err", max_det_err)
    record("recon_err", recon_err)
    record("linearity_err", lin_err)
    assert got_det.shape == (4, 20, 24)
    assert max_det_err < 1e-14
    assert max_coarse_err < 1e-14
    assert recon_err < 1e-14
    assert lin_err < 1e-14
    assert int(np.argmax(imp_energies)) == 0
    assert int(np.argmax(broad_energies)) >= 2


@pytest.mark.validates(
    "virgil.imaging.starlet",
    roots=["mathematics"],
    kind="control",
    headline="reflect_boundary_diff",
)
def test_starlet_reflect_boundary_control():
    edge_img = np.ones((16, 16), dtype=float)
    got_det, _ = (np.asarray(a) for a in vi.starlet(edge_img, scales=3))
    wrong_det, _ = ref.starlet_ref(edge_img, scales=3, mode="reflect")
    diff = float(np.max(np.abs(got_det - wrong_det)))
    record("reflect_boundary_diff", diff)
    assert diff > 0.1


@pytest.mark.validates(
    "virgil.imaging.Centroid",
    roots=["mathematics"],
    headline="max_centroid_err_mas",
)
def test_centroid_sky_coordinates_shifts_rotations_and_grid_angle():
    rng = np.random.default_rng(11)
    eta = rng.normal(size=(16, 16))
    eta[3, 4] += 3.0
    scale, sigma = 1.8, 2.5

    img = vm.Image(eta, scale, flux=0.4, dra=1.3, ddec=-0.9, rotation_deg=32.0)
    sys_model = vm.System(star=vm.PointSource(), env=img)
    cent_prior = vi.Centroid(sigma_mas=sigma, path="env")

    got_c = np.asarray(cent_prior.centroid(sys_model))
    got_r = np.asarray(cent_prior.residuals(sys_model))
    got_v = float(cent_prior.value(sys_model))
    want = ref.sky_centroid_ref(
        np.asarray(img.brightness), scale, sigma_mas=sigma, dra=1.3, ddec=-0.9, rotation_deg=32.0
    )

    c_err = float(np.max(np.abs(got_c - want["centroid"])))
    r_err = float(np.max(np.abs(got_r - want["residuals"])))
    v_err = abs(got_v - float(want["value"]))

    grad_ad = np.asarray(
        jax.grad(lambda x: vi.Centroid(sigma).value(vm.Image(x, scale, rotation_deg=32.0)))(jnp.asarray(eta))
    )
    eps = 1e-6
    e00 = np.zeros_like(eta)
    e00[3, 4] = eps
    fd_34 = (
        float(vi.Centroid(sigma).value(vm.Image(eta + e00, scale, rotation_deg=32.0)))
        - float(vi.Centroid(sigma).value(vm.Image(eta - e00, scale, rotation_deg=32.0)))
    ) / (2.0 * eps)
    grad_err = abs(float(grad_ad[3, 4]) - fd_34)

    grid_img = vm.Image(eta, scale, dra=1.3, ddec=-0.9, rotation_deg="grid")
    got_grid_c = np.asarray(vi.Centroid(sigma).centroid(grid_img, grid=32.0))
    grid_c_err = float(np.max(np.abs(got_grid_c - want["centroid"])))

    lin_flux = np.exp(0.3 * eta)
    lin0 = vm.LinearImage(lin_flux, scale, dra=-0.7, ddec=1.1, rotation_deg=0.0, parametrization="linear")
    want_lin0 = ref.sky_centroid_ref(lin_flux, scale, sigma_mas=sigma, dra=-0.7, ddec=1.1, rotation_deg=0.0)
    lin0_err = max(
        float(np.max(np.abs(np.asarray(vi.Centroid(sigma).centroid(lin0)) - want_lin0["centroid"]))),
        abs(float(vi.Centroid(sigma).value(lin0)) - float(want_lin0["value"])),
    )

    lin_m22 = vm.LinearImage(lin_flux, scale, dra=-0.7, ddec=1.1, rotation_deg=-22.0, parametrization="linear")
    want_lin_m22 = ref.sky_centroid_ref(lin_flux, scale, sigma_mas=sigma, dra=-0.7, ddec=1.1, rotation_deg=-22.0)
    lin_m22_err = max(
        float(np.max(np.abs(np.asarray(vi.Centroid(sigma).centroid(lin_m22)) - want_lin_m22["centroid"]))),
        abs(float(vi.Centroid(sigma).value(lin_m22)) - float(want_lin_m22["value"])),
    )

    record("max_centroid_err_mas", max(c_err, grid_c_err, lin0_err, lin_m22_err))
    record("value_err", v_err)
    record("grad_fd_err", grad_err)
    assert c_err < 1e-13
    assert r_err < 1e-13
    assert v_err < 1e-13
    assert grad_err < 1e-9
    assert grid_c_err < 1e-13
    assert lin0_err < 1e-13
    assert lin_m22_err < 1e-13


@pytest.mark.validates(
    "virgil.imaging.Centroid",
    roots=["mathematics"],
    kind="control",
    headline="wrong_rotation_sign_mas",
)
def test_centroid_wrong_rotation_sign_control():
    eta = np.full((15, 15), -20.0)
    eta[2, 7] = 5.0
    img = vm.Image(eta, 2.0, rotation_deg=35.0)
    got_c = np.asarray(vi.Centroid(2.0).centroid(img))
    wrong = ref.sky_centroid_ref(np.asarray(img.brightness), 2.0, sigma_mas=2.0, rotation_deg=-35.0)
    diff = float(np.linalg.norm(got_c - wrong["centroid"]))
    record("wrong_rotation_sign_mas", diff)
    assert diff > 5.0


@pytest.mark.validates(
    "virgil.imaging.image_priors",
    roots=["mathematics"],
    headline="gp_logprob_err",
)
def test_image_priors_plain_gp_and_linear_images():
    gp_field = vf.GaussianField(np.zeros((12, 12)), sigma=1.2, length_mas=4.0)
    scene = vm.System(
        star=vm.PointSource(),
        plain=vm.Image(np.zeros((8, 8)), 2.0, flux=0.2),
        gp=vm.Image(gp_field, 1.5, flux=0.3),
    )
    priors = vi.image_priors(scene)
    assert set(priors.keys()) == {"plain.log_brightness", "gp.log_brightness.latent"}

    rng = np.random.default_rng(3)
    z = rng.normal(size=np.asarray(gp_field.latent).shape)
    lp_got = float(np.sum(priors["gp.log_brightness.latent"].log_prob(z)))
    lp_want = float(np.sum(-0.5 * z**2 - 0.5 * np.log(2.0 * np.pi)))
    gp_lp_err = abs(lp_got - lp_want)

    lp_plain = float(np.sum(priors["plain.log_brightness"].log_prob(rng.normal(size=(8, 8)))))

    lin = vm.LinearImage(np.full((8, 8), 0.02), 2.0)
    lin_priors = vi.image_priors(lin)
    assert set(lin_priors.keys()) == {"pixels"}
    lp_lin = float(np.sum(lin_priors["pixels"].log_prob(np.full((8, 8), 0.05))))

    record("gp_logprob_err", gp_lp_err)
    record("flat_logprob_abs", abs(lp_plain) + abs(lp_lin))
    assert gp_lp_err < 1e-12
    assert abs(lp_plain) == 0.0
    assert abs(lp_lin) == 0.0


@pytest.mark.validates(
    "virgil.imaging.second_moment_fwhm",
    roots=["mathematics"],
    headline="fwhm_err_mas",
)
def test_second_moment_fwhm_wls_unresolved_and_insufficient_samples():
    wavel = 2.2e-6
    baselines = np.array([8.0, 14.0, 22.0, 30.0, 38.0, 55.0])
    angles = np.linspace(0.0, np.pi, baselines.size, endpoint=False)
    u_m = baselines * np.cos(angles)
    v_m = baselines * np.sin(angles)
    rho = baselines / wavel

    true_fwhm_mas = 4.6
    sigma_mas = true_fwhm_mas / ref.FWHM_PER_SIGMA
    disk = vm.GaussianDisk(sigma=sigma_mas)
    v2_exact = np.abs(np.asarray(disk.model(u_m, v_m, wavel))) ** 2
    d_v2 = np.array([0.015, 0.02, 0.018, 0.025, 0.03, 0.02])
    data = _simple_v2_oidata(u_m, v_m, wavel, v2_exact, d_v2)

    got = vi.second_moment_fwhm(data, v2_min=0.3)
    want = ref.second_moment_fwhm_ref(rho, v2_exact, d_v2, v2_min=0.3, fit_intercept=False)
    fwhm_err = abs(float(got) - float(want))

    unresolved = _simple_v2_oidata(u_m, v_m, wavel, np.ones_like(u_m), d_v2)
    got_zero = vi.second_moment_fwhm(unresolved)

    too_few = _simple_v2_oidata(u_m[:1], v_m[:1], wavel, np.array([0.8]), np.array([0.02]))
    disco = vo.OIData(vc.ami_grid_record(wavelength_m=4.8e-6))

    record("fwhm_err_mas", fwhm_err)
    record("unresolved_fwhm_mas", float(got_zero))
    assert fwhm_err < 1e-11
    assert abs(float(got) - true_fwhm_mas) < 1e-11
    assert got_zero == 0.0
    assert vi.second_moment_fwhm(too_few) is None
    assert vi.second_moment_fwhm(disco) is None


@pytest.mark.validates(
    "virgil.imaging.image_field",
    "virgil.imaging.Beam",
    roots=["mathematics"],
    property="field_clipping",
    headline="max_field_err_mas",
)
def test_image_field_and_beam_floor_ceiling_and_restoration():
    wavel = 2.2e-6
    u_m = np.array([5.0, 15.0, 40.0, 80.0])
    v_m = np.zeros_like(u_m)
    b_min, b_max = 5.0 / wavel, 80.0 / wavel
    data = _simple_v2_oidata(u_m, v_m, wavel, np.array([0.95, 0.85, 0.50, 0.15]), np.full(4, 0.02))

    cases = [
        (3.0, dict(field_fwhms=6.0, inner_elements=4.0, aperture_mas=None, largest_mas=500.0)),
        (0.1, dict(field_fwhms=6.0, inner_elements=4.0, aperture_mas=None, largest_mas=500.0)),
        (40.0, dict(field_fwhms=6.0, inner_elements=4.0, aperture_mas=None, largest_mas=500.0)),
        (40.0, dict(field_fwhms=6.0, inner_elements=4.0, aperture_mas=45.0, largest_mas=500.0)),
        (3.0, dict(field_fwhms=6.0, inner_elements=400.0, aperture_mas=30.0, largest_mas=500.0)),
    ]
    max_field_err = 0.0
    for fwhm, kw in cases:
        got = float(vi.image_field(data, fwhm_mas=fwhm, **kw))
        want = ref.image_field_ref(b_min, b_max, fwhm, **kw)
        max_field_err = max(max_field_err, abs(got - want))

    disco = vo.OIData(vc.ami_grid_record(wavelength_m=4.8e-6))
    got_disco = float(vi.image_field(disco, aperture_mas=320.0))
    want_disco = float(min(vi.field_of_view(disco), 320.0))
    max_field_err = max(max_field_err, abs(got_disco - want_disco))

    bm = vi.Beam(major_mas=6.0, minor_mas=3.0, pa_deg=35.0)
    assert (bm.major_mas, bm.minor_mas, bm.pa_deg) == (6.0, 3.0, 35.0)
    delta = np.zeros((41, 41), dtype=float)
    delta[20, 20] = 1.0
    conv = np.asarray(vi.convolve_beam(delta, 0.5, bm))
    east, north = ref.sky_grid_mas(41, 0.5)
    pa_r = np.deg2rad(35.0)
    along = east * np.sin(pa_r) + north * np.cos(pa_r)
    across = east * np.cos(pa_r) - north * np.sin(pa_r)
    sig_maj = float(np.sqrt(np.sum(conv * along**2)))
    sig_min = float(np.sqrt(np.sum(conv * across**2)))
    beam_moment_err = max(
        abs(sig_maj - 6.0 / ref.FWHM_PER_SIGMA),
        abs(sig_min - 3.0 / ref.FWHM_PER_SIGMA),
    )

    record("max_field_err_mas", max_field_err)
    record("beam_moment_err_mas", beam_moment_err)
    assert max_field_err < 1e-9
    assert beam_moment_err < 5e-3


@pytest.mark.validates(
    "virgil.imaging.image_field",
    roots=["mathematics"],
    kind="control",
    property="field_clipping",
    headline="crossover_floor_wins_err_mas",
)
def test_image_field_unclipped_and_floor_over_ceiling_controls():
    wavel = 2.2e-6
    u_m = np.array([5.0, 15.0, 40.0, 80.0])
    data = _simple_v2_oidata(u_m, np.zeros_like(u_m), wavel, np.array([0.95, 0.85, 0.50, 0.15]), np.full(4, 0.02))
    nyquist_mas = float((wavel / (2.0 * 80.0)) / ref.MAS_TO_RAD)

    # 1. Unclipped 6 * fwhm = 0.6 mas misses the Nyquist-element floor (~11.34 mas) by > 10 mas
    got_floor = float(vi.image_field(data, fwhm_mas=0.1, field_fwhms=6.0, inner_elements=4.0))
    unclipped_err = abs(got_floor - 0.6)

    # 2. On crossover (floor = 400 * nyquist ~ 1134 mas > aperture ceiling = 30 mas), floor-wins misses by > 1000 mas
    got_cross = float(
        vi.image_field(data, fwhm_mas=3.0, field_fwhms=6.0, inner_elements=400.0, aperture_mas=30.0)
    )
    wrong_floor_wins = 400.0 * nyquist_mas
    crossover_err = abs(got_cross - wrong_floor_wins)

    record("unclipped_err_mas", unclipped_err)
    record("crossover_floor_wins_err_mas", crossover_err)
    assert unclipped_err > 10.0
    assert crossover_err > 1000.0


# ---------------------------------------------------------------------------
# Lattice rotation_deg="grid" (#705), residual_image, & CleanResult
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.imaging.CleanResult",
    "virgil.imaging.residual_image",
    roots=["mathematics"],
    headline="v2_resid_ref_err",
)
def test_rotation_deg_grid_across_image_dirty_clean_and_residual():
    # 1. Pure-NumPy V^2 Fourier reference check on residual_image (zero virgil calls on reference side)
    rng_v = np.random.default_rng(33)
    wavel_v2 = 2.2e-6
    u_v2 = rng_v.uniform(-80.0, 80.0, size=32)
    v_v2 = rng_v.uniform(-80.0, 80.0, size=32)
    east_v, north_v = ref.sky_grid_mas(11, 1.5, rotation_deg=22.0)
    # Inject a companion at pixel (2, 8) of the rotated 11x11 grid in pure NumPy
    ph_comp = -2.0 * np.pi * (
        (u_v2 / wavel_v2) * east_v[2, 8] * ref.MAS_TO_RAD
        + (v_v2 / wavel_v2) * north_v[2, 8] * ref.MAS_TO_RAD
    )
    v2_comp = np.abs((1.0 + 0.08 * np.exp(1j * ph_comp)) / 1.08) ** 2
    sig_v2 = np.full(32, 0.005)
    v2_data = _simple_v2_oidata(u_v2, v_v2, wavel_v2, v2_comp, sig_v2)
    got_v2_resid = np.asarray(vi.residual_image(vm.PointSource(), v2_data, 11, 1.5, rotation_deg=22.0))
    want_v2_resid = ref.v2_point_residual_dirty_map_ref(
        u_v2, v_v2, wavel_v2, v2_comp, sig_v2, 11, 1.5, rotation_deg=22.0
    )
    v2_resid_ref_err = float(np.max(np.abs(got_v2_resid - want_v2_resid)))
    assert v2_resid_ref_err < 1e-10
    assert float(got_v2_resid[2, 8]) > 10.0  # positive gradient where flux is missing

    # 2. Rotated uv-lattice ("grid") across Image, dirty_image, residual_image, and CleanResult
    rot = -18.5
    template = vo.OIData(vc.ami_grid_record(wavelength_m=4.8e-6, rotation_deg=rot))
    assert abs(float(template.uv_grid.rotation_deg) - rot) < 1e-6

    rng = np.random.default_rng(23)
    eta = rng.normal(scale=0.5, size=(15, 15))
    scene_grid = vm.System(star=vm.PointSource(), env=vm.Image(eta, 25.0, flux=0.08, rotation_deg="grid"))
    scene_num = vm.System(star=vm.PointSource(), env=vm.Image(eta, 25.0, flux=0.08, rotation_deg=rot))
    grid_model_diff = float(np.max(np.abs(np.asarray(template.model(scene_grid) - template.model(scene_num)))))

    ungridded = eqx.tree_at(lambda d: d.uv_grid, template, None, is_leaf=lambda x: x is None)
    scene_zero = vm.System(star=vm.PointSource(), env=vm.Image(eta, 25.0, flux=0.08, rotation_deg=0.0))
    fallback_diff = float(np.max(np.abs(np.asarray(ungridded.model(scene_grid) - ungridded.model(scene_zero)))))

    truth_pix = np.full((15, 15), -30.0)
    truth_pix[4, 10] = 0.0
    truth = vm.System(star=vm.PointSource(), comp=vm.Image(truth_pix, 25.0, flux=0.06, rotation_deg=rot))
    sim = template.with_model(truth, key=jax.random.PRNGKey(5))

    dirty_diff = float(
        np.max(
            np.abs(
                np.asarray(vi.dirty_image(sim, 15, 25.0, rotation_deg="grid"))
                - np.asarray(vi.dirty_image(sim, 15, 25.0, rotation_deg=rot))
            )
        )
    )
    resid_grid = np.asarray(vi.residual_image(vm.PointSource(), sim, 15, 25.0, rotation_deg="grid"))
    resid_num = np.asarray(vi.residual_image(vm.PointSource(), sim, 15, 25.0, rotation_deg=rot))
    resid_diff = float(np.max(np.abs(resid_grid - resid_num)))

    # Verify rotated-lattice residual_image peak lands at injected companion pixel (4, 10)
    # and matches 3rd-order forward-difference -(d chi^2 / d eps) / ||d r / d eps||_2
    assert np.unravel_index(int(np.argmax(resid_grid)), resid_grid.shape) == (4, 10)
    assert float(resid_grid[4, 10]) > 100.0
    assert float(resid_grid[7, 7]) == 0.0  # phase-centre pixel has zero response around PointSource

    east_g, north_g = ref.sky_grid_mas(15, 25.0, rotation_deg=rot)
    eps_fd = 1e-4
    r0_sim = np.asarray(vl.whitened_residuals(vm.PointSource(flux=1.0), sim))
    fd_checks = []
    for r_i, c_i in ((4, 10), (3, 9), (10, 4)):
        dra_i, ddec_i = float(east_g[r_i, c_i]), float(north_g[r_i, c_i])
        r1 = np.asarray(
            vl.whitened_residuals(
                vm.System(star=vm.PointSource(flux=1.0), comp=vm.PointSource(dra=dra_i, ddec=ddec_i, flux=eps_fd)),
                sim,
            )
        )
        r2 = np.asarray(
            vl.whitened_residuals(
                vm.System(star=vm.PointSource(flux=1.0), comp=vm.PointSource(dra=dra_i, ddec=ddec_i, flux=2.0 * eps_fd)),
                sim,
            )
        )
        r3 = np.asarray(
            vl.whitened_residuals(
                vm.System(star=vm.PointSource(flux=1.0), comp=vm.PointSource(dra=dra_i, ddec=ddec_i, flux=3.0 * eps_fd)),
                sim,
            )
        )
        # Third-order 4-point non-negative forward difference at eps = 0 using fluxes (0, eps, 2*eps, 3*eps)
        jac_col = (-11.0 * r0_sim + 18.0 * r1 - 9.0 * r2 + 2.0 * r3) / (6.0 * eps_fd)
        dchi2 = 2.0 * float(r0_sim @ jac_col)
        jnorm = float(np.linalg.norm(jac_col))
        fd_checks.append(abs(float(resid_grid[r_i, c_i]) - (-dchi2 / jnorm)))
    disco_fd_err = max(fd_checks)

    clean_res = vi.clean(
        sim,
        15,
        25.0,
        base=vm.PointSource(),
        max_iterations=60,
        target_chi2_red=1.0,
        rotation_deg="grid",
    )
    assert isinstance(clean_res, vi.CleanResult)
    assert abs(float(clean_res.rotation_deg) - rot) < 1e-6

    # Independent SciPy rotated elliptical Gaussian convolution (and wrong +rot sign control)
    bm = vi.Beam(major_mas=80.0, minor_mas=50.0, pa_deg=25.0)
    restored = np.asarray(clean_res.restored(bm))
    comp_arr = np.asarray(clean_res.components)
    want_restored = ref.convolve_elliptical_beam_ref(
        comp_arr, 25.0, 80.0, 50.0, 25.0, rotation_deg=rot
    )
    wrong_sign_restored = ref.convolve_elliptical_beam_ref(
        comp_arr, 25.0, 80.0, 50.0, 25.0, rotation_deg=-rot
    )
    restored_diff = float(np.max(np.abs(restored - want_restored)))
    wrong_sign_diff = float(np.max(np.abs(restored - wrong_sign_restored)) / max(float(np.max(restored)), 1e-12))
    flux_conserved_err = abs(float(np.sum(restored) - np.sum(comp_arr)))

    record("v2_resid_ref_err", v2_resid_ref_err)
    record("disco_resid_fd_err", disco_fd_err)
    record("grid_model_diff", grid_model_diff)
    record("restored_beam_diff", restored_diff)
    assert disco_fd_err < 1e-8
    assert grid_model_diff < 1e-13
    assert fallback_diff < 1e-13
    assert dirty_diff < 1e-13
    assert resid_diff < 1e-13
    assert restored_diff < 1e-12
    assert wrong_sign_diff > 5e-2
    assert flux_conserved_err < 5e-2


# ---------------------------------------------------------------------------
# virgil.fitting.kkt_report (WF4: non-stationary + converged checks across
# LinearImage("sqrt"), LinearImage("linear"), softmax Image, and nuisance start=)
# ---------------------------------------------------------------------------

_UTS3 = np.array([[-9.925, -20.335], [14.887, 30.502], [103.306, 43.999]])


def _three_ut_oidata(path, scene, seed: int = 1):
    simulate.observe(
        path,
        lambda u, v, w: np.asarray(scene.model(u, v, w)),
        _UTS3,
        hour_angles_h=np.linspace(-2.0, 2.0, 4),
        wavelengths=np.array([1.65e-6, 2.2e-6]),
        dec_deg=-30.0,
        sigma_v2=0.015,
        sigma_cp_deg=0.8,
        rng=np.random.default_rng(seed),
    )
    return vo.OIData(str(path))


@pytest.mark.validates(
    "virgil.fitting.kkt_report",
    roots=["mathematics"],
    headline="nonstat_sqrt_kkt_err",
)
def test_kkt_report_sqrt_linear_image_stationarity_and_exact_zero_pixel(tmp_path):
    """``kkt_report`` on ``LinearImage("sqrt")`` agrees with ``crosscheck.starlet_kkt_ref`` on
    ``kkt_w``, ``kkt_rel`` and ``want_grow`` at a non-stationary point, on a converged MaxEntropy
    fit, on the band-limit fixed point, and on the exact-zero-pixel exclusion.

    Basis: the reference is our numpy re-derivation of the same KKT metrics, written from
    virgil's documented definitions (and from its 0.4.2 sqrt dark-pixel reach, see F31), so
    agreement shows ``kkt_report`` computes what is documented. It is not an independent
    definition of ``kkt_w``, ``kkt_rel`` or the ``want_grow`` deadband. The one independent
    fact used is mathematical: a converged constrained fit is stationary (``kkt_rel`` near 0).
    """
    n, scale, flux = 8, 1.0, 0.3
    shape = ref.gaussian_blob_ref(n, scale, sigma_mas=1.8, dra=0.8, ddec=-0.6) * flux
    truth = vm.System(
        star=vm.PointSource(),
        env=vm.LinearImage(np.sqrt(shape), scale, parametrization="sqrt", reference_flux=flux),
    )
    data = _three_ut_oidata(tmp_path / "sqrt.fits", truth, seed=3)

    # 1. Non-stationary LinearImage("sqrt") check: O(1) kkt_rel & O(100) kkt_w must match starlet_kkt_ref
    rng_ns = np.random.default_rng(71)
    s_ns = np.abs(rng_ns.normal(scale=0.06, size=(n, n))) + 2e-3
    s_ns[0, :] = 5e-5  # positive dark pixels (< 1e-6 * peak flux)
    s_ns[0, 0] = 0.0   # exact x = 0 pixel (stationary under x^2, never counted in want_grow)
    ns_model = vm.System(
        star=vm.PointSource(),
        env=vm.LinearImage(s_ns, scale, parametrization="sqrt", reference_flux=flux),
    )
    priors = vi.image_priors(ns_model)
    tsv_reg = [vi.TSV(15.0, beam_mas=scale, path="env")]
    ns_rep = vfit.kkt_report(
        vfit.FitResult(model=ns_model, values={"env.pixels": jnp.asarray(s_ns)}, info={}),
        priors,
        data,
        tsv_reg,
    )["images"]["env"]

    f_ns = s_ns**2
    gd_ns = np.asarray(
        jax.grad(
            lambda fa: -vl.model_loglike(
                vm.System(star=vm.PointSource(), env=vm.LinearImage(fa, scale, parametrization="linear", reference_flux=flux)),
                data,
            )
        )(jnp.asarray(f_ns))
    )
    gp_ns = np.asarray(
        jax.grad(
            lambda fa: tsv_reg[0].value(
                vm.System(star=vm.PointSource(), env=vm.LinearImage(fa, scale, parametrization="linear", reference_flux=flux))
            )
        )(jnp.asarray(f_ns))
    )
    want_ns_sq = ref.kkt_sqrt_metrics_ref(
        s_ns, gd_ns, gp_ns, weight=15.0, kkt_tol=float(vfit.KKT_TOL), reach_mode="sqrt"
    )
    nonstat_sqrt_err = max(
        abs(float(ns_rep["kkt_w"]) - want_ns_sq["kkt_w"]),
        abs(float(ns_rep["kkt_rel"]) - want_ns_sq["kkt_rel"]),
        abs(float(ns_rep["want_grow"]) - want_ns_sq["want_grow"]),
    )
    assert float(ns_rep["kkt_w"]) > 1.0
    assert nonstat_sqrt_err < 1e-11

    # 2. Converged MaxEntropy fit with start=, band_limit fixed-point / ripple / coarse-NaN checks
    init_sqrt = np.sqrt(np.full((n, n), flux / (n * n)))
    start = vm.System(
        star=vm.PointSource(),
        env=vm.LinearImage(init_sqrt, scale, parametrization="sqrt", reference_flux=flux),
    )
    reg = [vi.MaxEntropy(15.0, path="env")]
    done = vfit.fit(start, priors, data, reg, max_steps=2000, gtol=1e-12)
    rep = vfit.kkt_report(done, priors, data, reg, start={"env.pixels": init_sqrt})
    img_rep = rep["images"]["env"]

    def _me_grads_wrt_flux(f_arr):
        gd = np.asarray(
            jax.grad(
                lambda fa: -vl.model_loglike(
                    vm.System(
                        star=vm.PointSource(),
                        env=vm.LinearImage(jnp.sqrt(fa), scale, parametrization="sqrt", reference_flux=flux),
                    ),
                    data,
                )
            )(jnp.asarray(f_arr))
        )
        gp = np.asarray(
            jax.grad(
                lambda fa: reg[0].value(
                    vm.System(
                        star=vm.PointSource(),
                        env=vm.LinearImage(jnp.sqrt(fa), scale, parametrization="sqrt", reference_flux=flux),
                    )
                )
            )(jnp.asarray(f_arr))
        )
        return gd, gp

    x_opt = np.asarray(done.model.env.pixels)
    f_opt = np.asarray(done.model.env.fluxes)
    gd_opt, gp_opt = _me_grads_wrt_flux(f_opt)
    gd_init, gp_init = _me_grads_wrt_flux(init_sqrt**2)
    want_opt_sq = ref.kkt_sqrt_metrics_ref(
        x_opt,
        gd_opt,
        gp_opt,
        weight=15.0,
        kkt_tol=float(vfit.KKT_TOL),
        reach_mode="sqrt",
        start_sqrt_pixels=init_sqrt,
        start_grad_data_f=gd_init,
        start_grad_pen_f=gp_init,
    )
    kkt_w_diff = abs(float(img_rep["kkt_w"]) - want_opt_sq["kkt_w"])
    kkt_rel_start_diff = abs(float(img_rep["kkt_rel"]) - want_opt_sq["kkt_rel"])
    ref_opt_bl = ref.maxent_band_limit_ref(f_opt, data.u, data.v, data.wavel, scale)

    # Pure band-limited periodic log-brightness (|q| = 0.125 cyc/mas < q_max ~ 0.382 cyc/mas)
    # hits the exact 1.0 band-limited fixed point (low == eta) on both virgil and starlet_kkt_ref;
    # injecting a super-Nyquist checkerboard (-1)^(r+c) drives both > BAND_TOL (converged=False),
    # and a coarse pixel scale (q_max >= sqrt(2) / (2 * pixel_scale_mas)) returns NaN on both.
    rr, cc = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    eta_low = 0.35 * np.cos(2.0 * np.pi * rr / n) + 0.25 * np.sin(2.0 * np.pi * cc / n)
    f_low = flux * np.exp(eta_low) / np.sum(np.exp(eta_low))
    low_model = vm.System(
        star=vm.PointSource(),
        env=vm.LinearImage(np.sqrt(f_low), scale, parametrization="sqrt", reference_flux=flux),
    )
    low_rep = vfit.kkt_report(
        vfit.FitResult(model=low_model, values={"env.pixels": jnp.asarray(np.sqrt(f_low))}, info={}),
        priors,
        data,
        reg,
    )["images"]["env"]
    ref_low_bl = ref.maxent_band_limit_ref(f_low, data.u, data.v, data.wavel, scale)
    band_limit_low_err = abs(float(low_rep["band_limit"]) - ref_low_bl)

    coarse_scale = 4.0  # corner Nyquist sqrt(2) / 8 = 0.177 cyc/mas < q_max ~ 0.382 cyc/mas
    coarse_model = vm.System(
        star=vm.PointSource(),
        env=vm.LinearImage(np.sqrt(f_low), coarse_scale, parametrization="sqrt", reference_flux=flux),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        coarse_rep = vfit.kkt_report(
            vfit.FitResult(model=coarse_model, values={"env.pixels": jnp.asarray(np.sqrt(f_low))}, info={}),
            vi.image_priors(coarse_model),
            data,
            reg,
        )["images"]["env"]
    ref_coarse_bl = ref.maxent_band_limit_ref(f_low, data.u, data.v, data.wavel, coarse_scale)

    x_ripple = np.sqrt(f_opt * np.exp(2.0 * ((-1.0) ** (rr + cc))))
    ripple_res = dataclasses.replace(
        done,
        model=vm.System(
            star=vm.PointSource(),
            env=vm.LinearImage(x_ripple, scale, parametrization="sqrt", reference_flux=flux),
        ),
        values={**done.values, "env.pixels": jnp.asarray(x_ripple)},
    )
    rep_ripple = vfit.kkt_report(ripple_res, priors, data, reg)
    ref_ripple_bl = ref.maxent_band_limit_ref(x_ripple**2, data.u, data.v, data.wavel, scale)

    # 3. Non-vacuous exact x == 0 exclusion at a bright companion peak:
    #    Fit halo with peak_rc masked out so lit halo pixels are stationary (kkt_rel < 1e-5) while
    #    -dJ/df at peak_rc is > 1000. At x[peak_rc] = 8e-4 * x_peak (f = 6.4e-7 * f_peak < 1e-6 * f_peak),
    #    want_grow == 1/n^2 > 0 (converged=False); setting x[peak_rc] = 0.0 flips want_grow to 0.0!
    halo = ref.gaussian_blob_ref(n, scale, sigma_mas=1.8, dra=-1.0, ddec=0.8) * 0.12
    spike = np.zeros((n, n))
    peak_rc = (6, 6)
    spike[peak_rc] = 0.18
    truth_pk = vm.System(
        star=vm.PointSource(),
        env=vm.LinearImage(np.sqrt(halo + spike), scale, parametrization="sqrt", reference_flux=flux),
    )
    data_pk = _three_ut_oidata(tmp_path / "sqrt_pk.fits", truth_pk, seed=21)
    sup_mask = np.ones((n, n), dtype=bool)
    sup_mask[peak_rc] = False
    m_masked = vm.System(
        star=vm.PointSource(),
        env=vm.LinearImage(
            np.where(sup_mask, np.sqrt(flux / (n * n)), 0.0),
            scale,
            parametrization="sqrt",
            reference_flux=flux,
            support=sup_mask,
        ),
    )
    reg_pk = [vi.TSV(2.0, beam_mas=scale, path="env")]
    done_pk = vfit.fit(m_masked, vi.image_priors(m_masked), data_pk, reg_pk, method="lbfgs", max_steps=1000, gtol=1e-12)
    x_base = np.asarray(done_pk.model.env.pixels).copy()
    x_peak = float(np.max(x_base))

    def _eval_pk(frac: float):
        xt = x_base.copy()
        xt[peak_rc] = frac * x_peak
        mt = vm.System(
            star=vm.PointSource(),
            env=vm.LinearImage(xt, scale, parametrization="sqrt", reference_flux=flux),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rt = vfit.kkt_report(
                vfit.FitResult(model=mt, values={"env.pixels": jnp.asarray(xt)}, info={}),
                vi.image_priors(mt),
                data_pk,
                reg_pk,
            )
        ft = xt**2
        gd_t = np.asarray(
            jax.grad(
                lambda fa: -vl.model_loglike(
                    vm.System(star=vm.PointSource(), env=vm.LinearImage(fa, scale, parametrization="linear", reference_flux=flux)),
                    data_pk,
                )
            )(jnp.asarray(ft))
        )
        gp_t = np.asarray(
            jax.grad(
                lambda fa: reg_pk[0].value(
                    vm.System(star=vm.PointSource(), env=vm.LinearImage(fa, scale, parametrization="linear", reference_flux=flux))
                )
            )(jnp.asarray(ft))
        )
        return rt, xt, gd_t, gp_t

    rep_dark_hi, x_hi, gd_hi, gp_hi = _eval_pk(8e-4)
    rep_zero, x_zero, gd_z, gp_z = _eval_pk(0.0)
    rep_dark_lo, x_lo, gd_lo, gp_lo = _eval_pk(1e-6)
    want_hi_sq = ref.kkt_sqrt_metrics_ref(x_hi, gd_hi, gp_hi, weight=2.0, kkt_tol=float(vfit.KKT_TOL), reach_mode="sqrt")
    want_zero_sq = ref.kkt_sqrt_metrics_ref(x_zero, gd_z, gp_z, weight=2.0, kkt_tol=float(vfit.KKT_TOL), reach_mode="sqrt")
    want_lo_sq = ref.kkt_sqrt_metrics_ref(x_lo, gd_lo, gp_lo, weight=2.0, kkt_tol=float(vfit.KKT_TOL), reach_mode="sqrt")
    want_lo_flux = ref.kkt_sqrt_metrics_ref(x_lo, gd_lo, gp_lo, weight=2.0, kkt_tol=float(vfit.KKT_TOL), reach_mode="flux")

    record("nonstat_sqrt_kkt_err", nonstat_sqrt_err)
    record("opt_kkt_w_diff", kkt_w_diff)
    record("opt_kkt_rel_start_diff", kkt_rel_start_diff)
    record("band_limit_low_err", band_limit_low_err)
    record("band_limit", float(img_rep["band_limit"]))
    assert rep["converged"] is True
    assert float(img_rep["band_limit"]) < float(vfit.BAND_TOL)
    assert ref_opt_bl < float(vfit.BAND_TOL)
    assert abs(float(low_rep["band_limit"]) - 1.0) < 1e-12
    assert band_limit_low_err < 1e-12
    assert np.isnan(float(coarse_rep["band_limit"]))
    assert np.isnan(ref_coarse_bl)
    assert rep_ripple["converged"] is False
    assert float(rep_ripple["images"]["env"]["band_limit"]) > float(vfit.BAND_TOL)
    assert ref_ripple_bl > float(vfit.BAND_TOL)
    assert kkt_w_diff < 1e-11
    assert kkt_rel_start_diff < 1e-11
    assert float(img_rep["kkt_rel"]) < float(vfit.KKT_TOL)
    assert float(img_rep["want_grow"]) == 0.0
    # Non-vacuous exact-zero proof: same peak_rc pixel flips want_grow from 1/n^2 > 0 at 8e-4*x_peak to 0.0 at x == 0
    assert abs(float(rep_dark_hi["images"]["env"]["want_grow"]) - 1.0 / (n * n)) < 1e-15
    assert abs(float(rep_dark_hi["images"]["env"]["want_grow"]) - want_hi_sq["want_grow"]) < 1e-15
    assert rep_dark_hi["converged"] is False
    assert float(rep_zero["images"]["env"]["want_grow"]) == 0.0
    assert want_zero_sq["want_grow"] == 0.0
    assert rep_zero["converged"] is True
    # Pin virgil 0.4.2's sqrt(f_peak / f_i) scaling at 1e-6*x_peak (F31 strict xfail below tests documented reach=1)
    assert float(rep_dark_lo["images"]["env"]["want_grow"]) == want_lo_sq["want_grow"] == 0.0
    assert abs(want_lo_flux["want_grow"] - 1.0 / (n * n)) < 1e-15


@pytest.mark.validates(
    "virgil.fitting.kkt_report",
    roots=["mathematics"],
    kind="finding",
    headline="sqrt_dark_want_grow_err",
)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "F31: kkt_report docstring states LinearImage('sqrt') uses the same want_grow rule as "
        "'softmax' (reach = 1), whereas _want_grow multiplies the dark-pixel deadband by "
        "sqrt(f_peak / f_i) > 1000, suppressing >3e4x negative flux gradients to want_grow == 0.0"
    ),
)
def test_f31_kkt_report_sqrt_dark_pixel_want_grow_unscaled_deadband(tmp_path):
    n, scale, flux = 8, 1.0, 0.30
    halo = ref.gaussian_blob_ref(n, scale, sigma_mas=1.8, dra=-1.0, ddec=0.8) * 0.12
    spike = np.zeros((n, n))
    peak_rc = (6, 6)
    spike[peak_rc] = 0.18
    truth_pk = vm.System(
        star=vm.PointSource(),
        env=vm.LinearImage(np.sqrt(halo + spike), scale, parametrization="sqrt", reference_flux=flux),
    )
    data_pk = _three_ut_oidata(tmp_path / "f31_sqrt.fits", truth_pk, seed=21)
    sup_mask = np.ones((n, n), dtype=bool)
    sup_mask[peak_rc] = False
    m_masked = vm.System(
        star=vm.PointSource(),
        env=vm.LinearImage(
            np.where(sup_mask, np.sqrt(flux / (n * n)), 0.0),
            scale,
            parametrization="sqrt",
            reference_flux=flux,
            support=sup_mask,
        ),
    )
    reg_pk = [vi.TSV(2.0, beam_mas=scale, path="env")]
    done_pk = vfit.fit(m_masked, vi.image_priors(m_masked), data_pk, reg_pk, method="lbfgs", max_steps=1000, gtol=1e-12)
    x_lo = np.asarray(done_pk.model.env.pixels).copy()
    x_lo[peak_rc] = 1e-6 * float(np.max(x_lo))  # positive dark pixel: f_i / f_peak = 1e-12 < 1e-6
    m_lo = vm.System(
        star=vm.PointSource(),
        env=vm.LinearImage(x_lo, scale, parametrization="sqrt", reference_flux=flux),
    )
    rep_lo = vfit.kkt_report(
        vfit.FitResult(model=m_lo, values={"env.pixels": jnp.asarray(x_lo)}, info={}),
        vi.image_priors(m_lo),
        data_pk,
        reg_pk,
    )
    f_lo = x_lo**2
    gd_lo = np.asarray(
        jax.grad(
            lambda fa: -vl.model_loglike(
                vm.System(star=vm.PointSource(), env=vm.LinearImage(fa, scale, parametrization="linear", reference_flux=flux)),
                data_pk,
            )
        )(jnp.asarray(f_lo))
    )
    gp_lo = np.asarray(
        jax.grad(
            lambda fa: reg_pk[0].value(
                vm.System(star=vm.PointSource(), env=vm.LinearImage(fa, scale, parametrization="linear", reference_flux=flux))
            )
        )(jnp.asarray(f_lo))
    )
    want_doc = ref.kkt_sqrt_metrics_ref(
        x_lo, gd_lo, gp_lo, weight=2.0, kkt_tol=float(vfit.KKT_TOL), reach_mode="flux"
    )
    doc_deadband = 3.0 * want_doc["scale"] * max(want_doc["kkt_rel"], float(vfit.KKT_TOL))
    assert -(gd_lo + gp_lo)[peak_rc] > 1e4 * doc_deadband
    err = abs(float(rep_lo["images"]["env"]["want_grow"]) - want_doc["want_grow"])
    record("sqrt_dark_want_grow_err", err)
    record("virgil_want_grow", float(rep_lo["images"]["env"]["want_grow"]))
    record("doc_want_grow", want_doc["want_grow"])
    assert err < 1e-12
    assert rep_lo["converged"] is False


@pytest.mark.validates(
    "virgil.fitting.kkt_report",
    roots=["mathematics"],
    headline="nonstat_box_kkt_err",
)
def test_kkt_report_bounded_linear_image_projected_gradient_and_active_zeros(tmp_path):
    n, scale, flux = 8, 1.0, 0.25
    shape = ref.gaussian_blob_ref(n, scale, sigma_mas=0.9, dra=0.8) * flux
    truth = vm.System(
        star=vm.PointSource(),
        env=vm.LinearImage(shape, scale, parametrization="linear", reference_flux=flux),
    )
    data = _three_ut_oidata(tmp_path / "lin.fits", truth, seed=9)

    init_pix = np.full((n, n), flux / (n * n))
    start = vm.System(
        star=vm.PointSource(),
        env=vm.LinearImage(init_pix, scale, parametrization="linear", reference_flux=flux),
    )
    priors = vi.image_priors(start)
    reg = [vi.TSV(20.0, beam_mas=scale, path="env")]

    # 1. Non-stationary LinearImage("linear") with active zeros (both g >= 0 feasible zeros and g < -KKT_TOL*top violations)
    rng_ns = np.random.default_rng(83)
    f_ns = np.abs(rng_ns.normal(scale=0.015, size=(n, n))) + 5e-4
    f_ns[:2, :] = 0.0
    peak_rc = np.unravel_index(int(np.argmax(shape)), shape.shape)
    f_ns[peak_rc] = 0.0  # clamping true peak pixel to zero forces g < -KKT_TOL*top -> zero_violation > 0
    ns_lin = vm.System(
        star=vm.PointSource(),
        env=vm.LinearImage(f_ns, scale, parametrization="linear", reference_flux=flux),
    )
    ns_rep = vfit.kkt_report(
        vfit.FitResult(model=ns_lin, values={"env.pixels": jnp.asarray(f_ns)}, info={}),
        priors,
        data,
        reg,
    )["images"]["env"]

    gd_ns = np.asarray(
        jax.grad(
            lambda fa: -vl.model_loglike(
                vm.System(star=vm.PointSource(), env=vm.LinearImage(fa, scale, parametrization="linear", reference_flux=flux)),
                data,
            )
        )(jnp.asarray(f_ns))
    )
    gp_ns = np.asarray(
        jax.grad(
            lambda fa: reg[0].value(
                vm.System(star=vm.PointSource(), env=vm.LinearImage(fa, scale, parametrization="linear", reference_flux=flux))
            )
        )(jnp.asarray(f_ns))
    )
    want_ns = ref.kkt_nonneg_linear_metrics_ref(
        f_ns,
        gd_ns,
        gp_ns,
        weight=20.0,
        reference_flux=float(np.sum(f_ns)),
        kkt_tol=float(vfit.KKT_TOL),
    )
    nonstat_box_err = max(
        abs(float(ns_rep["kkt_w"]) - want_ns["kkt_w"]),
        abs(float(ns_rep["pg_rel"]) - want_ns["pg_rel"]),
        abs(float(ns_rep["zero_violation"]) - want_ns["zero_violation"]),
    )
    assert float(ns_rep["kkt_w"]) > 1.0
    assert float(ns_rep["zero_violation"]) > 0.0
    assert nonstat_box_err < 1e-11

    # 2. Converged bounded L-BFGS-B optimum with active feasible zeros AND start= normalization
    done = vfit.fit(start, priors, data, reg, method="lbfgsb", max_steps=2000)
    rep = vfit.kkt_report(done, priors, data, reg, start={"env.pixels": init_pix})
    img_rep = rep["images"]["env"]

    def _lin_grads(f_arr):
        gd = np.asarray(
            jax.grad(
                lambda fa: -vl.model_loglike(
                    vm.System(star=vm.PointSource(), env=vm.LinearImage(fa, scale, parametrization="linear", reference_flux=flux)),
                    data,
                )
            )(jnp.asarray(f_arr))
        )
        gp = np.asarray(
            jax.grad(
                lambda fa: reg[0].value(
                    vm.System(star=vm.PointSource(), env=vm.LinearImage(fa, scale, parametrization="linear", reference_flux=flux))
                )
            )(jnp.asarray(f_arr))
        )
        return gd, gp

    f_opt = np.asarray(done.model.env.fluxes)
    gd_opt, gp_opt = _lin_grads(f_opt)
    gd_init, gp_init = _lin_grads(init_pix)
    ref_lin = ref.kkt_nonneg_linear_metrics_ref(
        f_opt,
        gd_opt,
        gp_opt,
        weight=20.0,
        reference_flux=float(np.sum(f_opt)),
        kkt_tol=float(vfit.KKT_TOL),
        start_grad_total=gd_init + gp_init,
    )
    kkt_w_diff = abs(float(img_rep["kkt_w"]) - ref_lin["kkt_w"])
    pg_rel_start_diff = abs(float(img_rep["pg_rel"]) - ref_lin["pg_rel"])

    # Also check non-stationary point WITH start= normalization directly
    ns_rep_start = vfit.kkt_report(
        vfit.FitResult(model=ns_lin, values={"env.pixels": jnp.asarray(f_ns)}, info={}),
        priors,
        data,
        reg,
        start={"env.pixels": init_pix},
    )["images"]["env"]
    want_ns_start = ref.kkt_nonneg_linear_metrics_ref(
        f_ns,
        gd_ns,
        gp_ns,
        weight=20.0,
        reference_flux=float(np.sum(f_ns)),
        kkt_tol=float(vfit.KKT_TOL),
        start_grad_total=gd_init + gp_init,
    )
    ns_start_err = max(
        abs(float(ns_rep_start["pg_rel"]) - want_ns_start["pg_rel"]),
        abs(float(ns_rep_start["zero_violation"]) - want_ns_start["zero_violation"]),
    )

    record("nonstat_box_kkt_err", max(nonstat_box_err, ns_start_err))
    record("opt_kkt_w_diff", kkt_w_diff)
    record("opt_pg_rel_start_diff", pg_rel_start_diff)
    record("pg_rel", float(img_rep["pg_rel"]))
    assert rep["converged"] is True
    assert float(img_rep["zero_violation"]) == 0.0
    assert kkt_w_diff < 1e-11
    assert pg_rel_start_diff < 1e-11
    assert ns_start_err < 1e-11
    assert float(img_rep["pg_rel"]) < float(vfit.KKT_TOL)


@pytest.mark.validates(
    "virgil.fitting.kkt_report",
    roots=["mathematics"],
    headline="nonstat_fixed_flux_kkt_err",
)
def test_kkt_report_softmax_image_multiplier_and_nuisance_start_gate(tmp_path):
    n, scale = 10, 0.65
    east, north = ref.sky_grid_mas(n, scale)
    support = np.hypot(east, north) >= 1.0
    blob = ref.gaussian_blob_ref(n, scale, sigma_mas=1.3, dra=1.5, ddec=-1.0) * support
    blob /= blob.sum()
    init_blob = ref.gaussian_blob_ref(n, scale, sigma_mas=1.8) * support
    init_blob /= init_blob.sum()
    init_eta = np.where(support, np.log(np.clip(init_blob, 1e-12, None)), 0.0)

    truth = vm.System(
        star=vm.PointSource(),
        env=vm.Image.from_brightness(blob, scale, flux=0.22, support=support),
    )
    data = _three_ut_oidata(tmp_path / "softmax.fits", truth, seed=14)

    # 1. Non-stationary FIXED-flux Image check: env.flux = 0.28 is NOT in priors, so
    #    mu = sum_i b_i g_i = F * d(-loglike)/dF != 0, kkt_w = O(10), and want_grow > 0!
    rng_ns = np.random.default_rng(91)
    eta_ns = np.where(support, rng_ns.normal(scale=1.2, size=(n, n)), 0.0)
    # Darken pixels around the true companion at (dra=1.5, ddec=-1.0) so the data strongly want them to grow
    near_comp = support & (np.hypot(east - 1.5, north - (-1.0)) < 1.2)
    eta_ns = np.where(near_comp, -20.0, eta_ns)
    fixed_flux = 0.28
    ns_im = vm.System(
        star=vm.PointSource(),
        env=vm.Image(eta_ns, scale, flux=fixed_flux, support=support),
    )
    reg = [vi.TSV(80.0, beam_mas=scale, path="env")]
    ns_fixed_rep = vfit.kkt_report(
        vfit.FitResult(model=ns_im, values={"env.log_brightness": jnp.asarray(eta_ns)}, info={}),
        vi.image_priors(ns_im),
        data,
        reg,
    )["images"]["env"]

    b_ns = np.asarray(ns_im.env.brightness)
    # Data gradient wrt unit-sum brightness b (fixing star flux=1.0 and scaling env flux = fixed_flux * sum(b)):
    gd_b = np.asarray(
        jax.grad(
            lambda ba: -vl.model_loglike(
                vm.System(
                    star=vm.PointSource(flux=1.0),
                    env=vm.LinearImage(fixed_flux * ba, scale, parametrization="linear", reference_flux=fixed_flux, support=support),
                ),
                data,
            )
        )(jnp.asarray(b_ns))
    )
    # Penalty gradient wrt unit-sum brightness b on Image (homogeneous of degree 0 in F, so mu_pen == 0):
    ge_pen = np.asarray(
        jax.grad(
            lambda e: reg[0].value(
                vm.System(star=vm.PointSource(), env=vm.Image(e, scale, flux=fixed_flux, support=support))
            )
        )(jnp.asarray(eta_ns))
    )
    gp_b = np.where(b_ns > 0.0, ge_pen / np.where(b_ns > 0.0, b_ns, 1.0), 0.0)
    want_ns_sm = ref.kkt_simplex_metrics_ref(
        b_ns,
        gd_b,
        gp_b,
        support=support,
        flux=fixed_flux,
        free_flux=False,
        weight=80.0,
        kkt_tol=float(vfit.KKT_TOL),
    )
    wrong_free_flux = ref.kkt_simplex_metrics_ref(
        b_ns,
        gd_b,
        gp_b,
        support=support,
        flux=fixed_flux,
        free_flux=True,
        weight=80.0,
        kkt_tol=float(vfit.KKT_TOL),
    )
    nonstat_fixed_err = max(
        abs(float(ns_fixed_rep["kkt_w"]) - want_ns_sm["kkt_w"]),
        abs(float(ns_fixed_rep["kkt_rel"]) - want_ns_sm["kkt_rel"]),
        abs(float(ns_fixed_rep["want_grow"]) - want_ns_sm["want_grow"]),
    )
    assert abs(want_ns_sm["mu"]) > 10.0  # genuinely non-zero simplex Lagrange multiplier
    assert abs(float(ns_fixed_rep["kkt_rel"]) - wrong_free_flux["kkt_rel"]) > 0.02
    assert nonstat_fixed_err < 1e-11

    # 2. Converged free-flux Image fit (Uniform prior: flat & value space share zero prior slope) + nuisance start= gate
    start = vm.System(star=vm.PointSource(), env=vm.Image(init_eta, scale, flux=0.28, support=support))
    priors = vi.image_priors(start) | {"env.flux": dist.Uniform(1e-2, 1.0)}
    start_vals = {"env.log_brightness": init_eta, "env.flux": 0.28}
    done = vfit.fit(start, priors, data, reg, init=start_vals, max_steps=2000, gtol=1e-13)
    f_env = float(done.values["env.flux"])
    near_vals = {
        "env.log_brightness": np.asarray(done.values["env.log_brightness"]),
        "env.flux": f_env * 1.002,
    }

    rep_opt_only = vfit.kkt_report(done, priors, data, reg)
    rep_ok = vfit.kkt_report(done, priors, data, reg, start=start_vals)
    rep_near = vfit.kkt_report(done, priors, data, reg, start=near_vals)
    assert rep_ok["converged"] is True
    assert rep_near["converged"] is True
    assert float(rep_ok["nuisances"]["env.flux"]["grad_norm"]) < 1e-3
    assert float(rep_ok["nuisances"]["env.flux"]["grad_rel"]) < float(vfit.NUISANCE_TOL)
    assert float(rep_near["nuisances"]["env.flux"]["grad_rel"]) < float(vfit.NUISANCE_TOL)

    def _simplex_grads(eta_arr, flux_val):
        m_cur = vm.System(star=vm.PointSource(), env=vm.Image(eta_arr, scale, flux=flux_val, support=support))
        b_arr = np.asarray(m_cur.env.brightness)
        gd_arr = np.asarray(
            jax.grad(
                lambda ba: -vl.model_loglike(
                    vm.System(
                        star=vm.PointSource(flux=1.0),
                        env=vm.LinearImage(
                            flux_val * ba,
                            scale,
                            parametrization="linear",
                            reference_flux=flux_val,
                            support=support,
                        ),
                    ),
                    data,
                )
            )(jnp.asarray(b_arr))
        )
        ge_arr = np.asarray(
            jax.grad(
                lambda e: reg[0].value(
                    vm.System(star=vm.PointSource(), env=vm.Image(e, scale, flux=flux_val, support=support))
                )
            )(jnp.asarray(eta_arr))
        )
        gp_arr = np.where(b_arr > 0.0, ge_arr / np.where(b_arr > 0.0, b_arr, 1.0), 0.0)
        return b_arr, gd_arr, gp_arr

    def _flux_neg_loglike_grad(eta_arr, flux_val):
        return float(
            jax.grad(
                lambda fv: -vl.model_loglike(
                    vm.System(star=vm.PointSource(flux=1.0), env=vm.Image(eta_arr, scale, flux=fv, support=support)),
                    data,
                )
            )(jnp.float64(flux_val))
        )

    b_opt, gd_opt, gp_opt = _simplex_grads(done.values["env.log_brightness"], f_env)
    b_init, gd_init_b, gp_init_b = _simplex_grads(init_eta, 0.28)

    ref_sm = ref.kkt_simplex_metrics_ref(
        b_opt, gd_opt, gp_opt, support=support, flux=f_env, free_flux=True, weight=80.0
    )
    ref_sm_start = ref.kkt_simplex_metrics_ref(
        b_opt,
        gd_opt,
        gp_opt,
        support=support,
        flux=f_env,
        free_flux=True,
        weight=80.0,
        start_brightness=b_init,
        start_grad_data_b=gd_init_b,
        start_grad_pen_b=gp_init_b,
    )
    assert abs(float(rep_opt_only["images"]["env"]["kkt_w"]) - ref_sm["kkt_w"]) < 1e-12
    assert abs(float(rep_opt_only["images"]["env"]["kkt_rel"]) - ref_sm["kkt_rel"]) < 1e-12
    assert abs(float(rep_ok["images"]["env"]["kkt_w"]) - ref_sm_start["kkt_w"]) < 1e-12
    kkt_rel_start_err = abs(float(rep_ok["images"]["env"]["kkt_rel"]) - ref_sm_start["kkt_rel"])
    assert kkt_rel_start_err < 1e-11
    assert float(rep_ok["images"]["env"]["kkt_rel"]) < float(rep_opt_only["images"]["env"]["kkt_rel"])

    # Exact nuisance grad_norm and grad_rel check at interior Uniform optimum (both distant & nearby starts)
    g_flux_start = _flux_neg_loglike_grad(init_eta, 0.28)
    g_flux_near = _flux_neg_loglike_grad(near_vals["env.log_brightness"], near_vals["env.flux"])
    g_flux_opt = _flux_neg_loglike_grad(done.values["env.log_brightness"], f_env)
    want_nuis_ok = ref.kkt_nuisance_grad_rel_ref(g_flux_opt, g_flux_start)
    want_nuis_near = ref.kkt_nuisance_grad_rel_ref(g_flux_opt, g_flux_near)
    nuis_ok_err = max(
        abs(float(rep_ok["nuisances"]["env.flux"]["grad_norm"]) - want_nuis_ok["grad_norm"]),
        abs(float(rep_ok["nuisances"]["env.flux"]["grad_rel"]) - want_nuis_ok["grad_rel"]),
        abs(float(rep_near["nuisances"]["env.flux"]["grad_rel"]) - want_nuis_near["grad_rel"]),
    )
    assert nuis_ok_err < 1e-11

    bad_flux = 0.85
    fixed_flux_start = vm.System(
        star=vm.PointSource(),
        env=vm.Image(done.values["env.log_brightness"], scale, flux=bad_flux, support=support),
    )
    img_only = vfit.fit(
        fixed_flux_start, vi.image_priors(fixed_flux_start), data, reg, max_steps=2000, gtol=1e-13
    )
    perturbed = dataclasses.replace(
        done,
        model=img_only.model,
        values={"env.log_brightness": img_only.values["env.log_brightness"], "env.flux": jnp.asarray(bad_flux)},
    )
    rep_no_start = vfit.kkt_report(perturbed, priors, data, reg)
    rep_with_start = vfit.kkt_report(perturbed, priors, data, reg, start=near_vals)
    nuis_grad_rel = float(rep_with_start["nuisances"]["env.flux"]["grad_rel"])
    g_flux_pert = _flux_neg_loglike_grad(img_only.values["env.log_brightness"], bad_flux)
    want_nuis_pert = ref.kkt_nuisance_grad_rel_ref(
        g_flux_pert, g_flux_near, value_cur=bad_flux, lower=1e-2, upper=1.0
    )
    nuis_pert_err = max(
        abs(float(rep_with_start["nuisances"]["env.flux"]["grad_norm"]) - want_nuis_pert["grad_norm"]),
        abs(nuis_grad_rel - want_nuis_pert["grad_rel"]),
    )

    # Also pin virgil 0.4.2's boxed=True +1/F value-space evaluation on LogUniform (F30 xfail below)
    priors_lu = vi.image_priors(start) | {"env.flux": dist.LogUniform(1e-2, 1.0)}
    done_lu = vfit.fit(start, priors_lu, data, reg, init=start_vals, max_steps=2000, gtol=1e-13)
    f_lu = float(done_lu.values["env.flux"])
    g_ll_lu = _flux_neg_loglike_grad(done_lu.values["env.log_brightness"], f_lu)
    rep_lu_far = vfit.kkt_report(done_lu, priors_lu, data, reg, start=start_vals)
    assert abs(g_ll_lu) < 1e-3  # stationary in flat ln F
    assert abs(float(rep_lu_far["nuisances"]["env.flux"]["grad_norm"]) - abs(g_ll_lu + 1.0 / f_lu)) < 1e-11

    record("nonstat_fixed_flux_kkt_err", nonstat_fixed_err)
    record("softmax_start_kkt_rel_err", kkt_rel_start_err)
    record("nuisance_grad_rel_err", max(nuis_ok_err, nuis_pert_err))
    record("nuisance_grad_rel", nuis_grad_rel)
    assert rep_no_start["converged"] is True
    assert rep_with_start["converged"] is False
    assert nuis_pert_err < 1e-11
    assert nuis_grad_rel > float(vfit.NUISANCE_TOL)


@pytest.mark.validates(
    "virgil.fitting.kkt_report",
    roots=["mathematics"],
    kind="finding",
    headline="loguniform_nuis_grad_rel",
)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "F30: kkt_report constructs _Objective(..., boxed=True) on flat-coordinate LogUniform "
        "parameters, adding +1/F to d(-loglike)/dF so a stationary gtol=1e-13 fit() optimum "
        "has grad_norm ~ 1/F and fails NUISANCE_TOL from a nearby start"
    ),
)
def test_f30_kkt_report_loguniform_nuisance_stationary_from_nearby_start(tmp_path):
    n, scale = 10, 0.65
    east, north = ref.sky_grid_mas(n, scale)
    support = np.hypot(east, north) >= 1.0
    blob = ref.gaussian_blob_ref(n, scale, sigma_mas=1.3, dra=1.5, ddec=-1.0) * support
    blob /= blob.sum()
    init_blob = ref.gaussian_blob_ref(n, scale, sigma_mas=1.8) * support
    init_blob /= init_blob.sum()
    init_eta = np.where(support, np.log(np.clip(init_blob, 1e-12, None)), 0.0)

    truth = vm.System(
        star=vm.PointSource(),
        env=vm.Image.from_brightness(blob, scale, flux=0.22, support=support),
    )
    data = _three_ut_oidata(tmp_path / "f30_softmax.fits", truth, seed=14)
    reg = [vi.TSV(80.0, beam_mas=scale, path="env")]
    start = vm.System(star=vm.PointSource(), env=vm.Image(init_eta, scale, flux=0.28, support=support))
    priors_lu = vi.image_priors(start) | {"env.flux": dist.LogUniform(1e-2, 1.0)}
    start_vals = {"env.log_brightness": init_eta, "env.flux": 0.28}
    done_lu = vfit.fit(start, priors_lu, data, reg, init=start_vals, max_steps=2000, gtol=1e-13)
    f_lu = float(done_lu.values["env.flux"])
    near_lu = {
        "env.log_brightness": np.asarray(done_lu.values["env.log_brightness"]),
        "env.flux": f_lu * 1.05,
    }

    g_ll_opt = float(
        jax.grad(
            lambda fv: -vl.model_loglike(
                vm.System(star=vm.PointSource(flux=1.0), env=vm.Image(done_lu.values["env.log_brightness"], scale, flux=fv, support=support)),
                data,
            )
        )(jnp.float64(f_lu))
    )
    g_ll_near = float(
        jax.grad(
            lambda fv: -vl.model_loglike(
                vm.System(star=vm.PointSource(flux=1.0), env=vm.Image(near_lu["env.log_brightness"], scale, flux=fv, support=support)),
                data,
            )
        )(jnp.float64(near_lu["env.flux"]))
    )
    want_flat = ref.kkt_nuisance_grad_rel_ref(
        f_lu * g_ll_opt, near_lu["env.flux"] * g_ll_near
    )
    assert want_flat["grad_rel"] < float(vfit.NUISANCE_TOL)

    rep_lu = vfit.kkt_report(done_lu, priors_lu, data, reg, start=near_lu)
    grad_norm_lu = float(rep_lu["nuisances"]["env.flux"]["grad_norm"])
    grad_rel_lu = float(rep_lu["nuisances"]["env.flux"]["grad_rel"])
    record("loguniform_nuis_grad_rel", grad_rel_lu)
    record("loguniform_nuis_grad_norm", grad_norm_lu)
    record("flat_nuis_grad_rel", want_flat["grad_rel"])
    assert grad_norm_lu < 1e-2
    assert grad_rel_lu < float(vfit.NUISANCE_TOL)
    assert rep_lu["converged"] is True


@pytest.mark.validates(
    "virgil.fitting.kkt_report",
    roots=["mathematics"],
    kind="control",
    headline="short_kkt_rel",
)
def test_kkt_report_two_step_and_zero_violation_controls(tmp_path):
    n, scale, flux = 8, 1.0, 0.3
    shape = ref.gaussian_blob_ref(n, scale, sigma_mas=1.5, dra=1.0) * flux
    truth = vm.System(
        star=vm.PointSource(),
        env=vm.LinearImage(np.sqrt(shape), scale, parametrization="sqrt", reference_flux=flux),
    )
    data = _three_ut_oidata(tmp_path / "ctrl.fits", truth, seed=21)

    start = vm.System(
        star=vm.PointSource(),
        env=vm.LinearImage(np.sqrt(np.full((n, n), flux / (n * n))), scale, parametrization="sqrt", reference_flux=flux),
    )
    priors = vi.image_priors(start)
    reg = [vi.MaxEntropy(10.0, path="env")]
    with pytest.warns(RuntimeWarning):
        short = vfit.fit(start, priors, data, reg, max_steps=2)
    rep_short = vfit.kkt_report(short, priors, data, reg)
    short_kkt_rel = float(rep_short["images"]["env"]["kkt_rel"])

    lin_start = vm.System(
        star=vm.PointSource(),
        env=vm.LinearImage(shape, scale, parametrization="linear", reference_flux=flux),
    )
    lin_priors = vi.image_priors(lin_start)
    lin_reg = [vi.TSV(10.0, beam_mas=scale, path="env")]
    lin_done = vfit.fit(lin_start, lin_priors, data, lin_reg, method="lbfgsb", max_steps=1500)
    clamped = np.asarray(lin_done.model.env.fluxes).copy()
    peak_idx = np.unravel_index(int(np.argmax(clamped)), clamped.shape)
    clamped[peak_idx] = 0.0
    bad_lin = dataclasses.replace(
        lin_done,
        model=vm.System(
            star=vm.PointSource(),
            env=vm.LinearImage(clamped, scale, parametrization="linear", reference_flux=flux),
        ),
        values={**lin_done.values, "env.pixels": jnp.asarray(clamped)},
    )
    rep_clamped = vfit.kkt_report(bad_lin, lin_priors, data, lin_reg)

    record("short_kkt_rel", short_kkt_rel)
    record("clamped_zero_violation", float(rep_clamped["images"]["env"]["zero_violation"]))
    assert rep_short["converged"] is False
    assert short_kkt_rel > float(vfit.KKT_TOL)
    assert rep_clamped["converged"] is False
    assert float(rep_clamped["images"]["env"]["zero_violation"]) > 0.0
