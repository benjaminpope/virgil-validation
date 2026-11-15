"""Validation of virgil 0.4.2 synthetic scenes, B3-spline starlet wavelets,
sky centroids, short-baseline field sizing, uv-lattice ``rotation_deg="grid"``,
and KKT optimality certificates against ``crosscheck.starlet_kkt_ref``.
"""

from __future__ import annotations

import dataclasses
import gc

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


def _simple_v2_oidata(u_m: np.ndarray, v_m: np.ndarray, wavel_m: float, v2: np.ndarray, d_v2: np.ndarray):
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
def test_ref_scenes_moments_and_tsv_dirichlet_eigenvalues():
    # Exact eigenvalues of the 2D 5-point Dirichlet Laplacian on an n x n grid:
    # 4 sin^2(pi k_r / (2(n+1))) + 4 sin^2(pi k_c / (2(n+1)))
    n = 6
    w, f0 = 2.3, 0.8
    hess = ref.tsv_dirichlet_hessian_ref(n, weight=w, reference_flux=f0)
    got_eigs = np.sort(np.linalg.eigvalsh(hess))
    ks = np.arange(1, n + 1, dtype=float)
    eig_1d = 4.0 * np.sin(0.5 * np.pi * ks / (n + 1)) ** 2
    want_eigs = np.sort((2.0 * w / (f0**2)) * (eig_1d[:, None] + eig_1d[None, :]).ravel())
    eig_err = float(np.max(np.abs(got_eigs - want_eigs)))

    # Fine-grid Gaussian blob recovers exact first and second moments
    blob = ref.gaussian_blob_ref(101, 0.5, sigma_mas=2.5, dra=1.7, ddec=-1.2)
    cent = ref.sky_centroid_ref(blob, 0.5, sigma_mas=2.0)
    centroid_err = float(np.max(np.abs(cent["centroid"] - np.array([1.7, -1.2]))))

    record("laplacian_eig_err", eig_err)
    record("blob_centroid_err", centroid_err)
    assert eig_err < 1e-13
    assert centroid_err < 1e-12


@pytest.mark.validates(
    "crosscheck.starlet_kkt_ref",
    roots=["mathematics", "statistics"],
    kind="reference",
    headline="rel_hutchinson_se_err",
)
def test_ref_hutchinson_variance_and_kkt_simplex_identities():
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

    # Simplex KKT residual vanishes at exact SciPy SLSQP optimum of strictly convex entropy objective
    a_mat = rng.normal(size=(10, 6))
    y_vec = rng.normal(size=10)
    w_ent = 0.25

    def obj(b):
        res = a_mat @ b - y_vec
        return 0.5 * float(res @ res) + w_ent * float(np.sum(b * np.log(np.clip(b, 1e-30, None))))

    def grad(b):
        return a_mat.T @ (a_mat @ b - y_vec) + w_ent * (np.log(np.clip(b, 1e-30, None)) + 1.0)

    sol = optimize.minimize(
        obj,
        np.full(6, 1.0 / 6.0),
        jac=grad,
        method="SLSQP",
        bounds=[(1e-12, 1.0)] * 6,
        constraints=[{"type": "eq", "fun": lambda b: np.sum(b) - 1.0}],
        options={"ftol": 1e-14, "maxiter": 500},
    )
    assert sol.success
    kkt = ref.kkt_simplex_metrics_ref(sol.x, grad(sol.x), weight=w_ent)

    record("rel_hutchinson_se_err", rel_se_err)
    record("hutchinson_mean_z", mean_z)
    record("simplex_kkt_rel", kkt["kkt_rel"])
    assert rel_se_err < 0.12
    assert mean_z < 3.0
    assert kkt["kkt_rel"] < 1e-7
    assert kkt["want_grow"] == 0.0


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

    # Positive dra (East) lands in left half (col < 11.5); negative ddec (South) in bottom half (row > 11.5)
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

    # Symmetric inclined ring inertia tensor recovers minor/major ratio cos(inc_deg) and major-axis PA
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
    # Wrong modulation: using deprojected disk-plane azimuth instead of on-sky PA
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
    want_radial = ref.spiral_ref(
        npix, scale, step_mas=16.0, width_mas=2.2, turns=1.5, pa_deg=25.0, fade_mas=18.0, distance_mode="radial_cap"
    )
    max_pixel_err = float(np.max(np.abs(with_fade - want_radial)))

    # Exact pixelwise radial attenuation exp(-r / fade_mas) up to normalization
    east, north = ref.sky_grid_mas(npix, scale)
    r_sky = np.hypot(east, north)
    predicted_faded = no_fade * np.exp(-r_sky / 18.0)
    predicted_faded /= np.sum(predicted_faded)
    fade_rel_err = float(np.max(np.abs(with_fade - predicted_faded)) / np.max(with_fade))

    # Verify width_mas is Gaussian sigma (not FWHM) on an outer straight-crossing ray:
    # For pa_deg=0, turns=1.5, at PA=90 deg (East, row=16, dra > 0), theta = pi/2 -> r_arm = step/4 = 5.0 mas
    # Let's use step_mas=24.0, turns=1.0 -> at PA=180 deg (South, ddec < 0, dra = 0), theta = pi -> r_arm = 12.0 mas
    sp_south = np.asarray(vs.spiral(41, 1.0, step_mas=24.0, width_mas=2.5, turns=1.0, pa_deg=0.0))
    # South axis: col=20 (dra=0), row=32 (ddec=-12 mas, on spine) vs row=34 (ddec=-14 mas, |dr|=2 mas)
    ratio_2mas = float(sp_south[34, 20] / sp_south[32, 20])
    want_sigma_ratio = float(np.exp(-0.5 * (2.0 / 2.5) ** 2))

    # Rounded cap at arm end (turns=1.0, pa_deg=0 -> end at North ddec=+24 mas, row=16-24 on 65x65 grid)
    sp_cap = np.asarray(vs.spiral(65, 1.0, step_mas=20.0, width_mas=2.5, turns=1.0, pa_deg=0.0))
    # End point at (dra=0, ddec=+20) -> row=12, col=32. Beyond end along +North at (dra=0, ddec=+22.5) -> row=9.5:
    # check row=10, col=32 (2 mas past tip) has smooth Gaussian cap exp(-0.5*(2/2.5)^2), not a hard zero cut!
    cap_ratio = float(sp_cap[10, 32] / sp_cap[12, 32])

    record("max_pixel_err", max_pixel_err)
    record("fade_ratio_rel_err", fade_rel_err)
    record("sigma_profile_err", abs(ratio_2mas - want_sigma_ratio))
    record("cap_profile_err", abs(cap_ratio - want_sigma_ratio))
    assert abs(float(np.sum(with_fade)) - 1.0) < 1e-14
    assert fade_rel_err < 1e-13
    assert abs(ratio_2mas - want_sigma_ratio) < 0.03
    assert abs(cap_ratio - want_sigma_ratio) < 0.05
    assert max_pixel_err < 1e-14


@pytest.mark.validates(
    "virgil.scenes.spiral",
    roots=["mathematics"],
    kind="control",
    headline="fwhm_misinterpretation_err",
)
def test_spiral_width_fwhm_and_reversed_winding_controls():
    got = np.asarray(vs.spiral(33, 2.0, step_mas=18.0, width_mas=3.0, turns=1.5, pa_deg=15.0))
    wrong_fwhm = ref.spiral_ref(
        33, 2.0, step_mas=18.0, width_mas=3.0 / ref.FWHM_PER_SIGMA, turns=1.5, pa_deg=15.0
    )
    # Reversed (CW instead of CCW) winding mirrors across pa_deg=0 when pa_deg=0
    got0 = np.asarray(vs.spiral(33, 2.0, step_mas=18.0, width_mas=3.0, turns=1.25, pa_deg=0.0))
    wrong_cw = got0[:, ::-1]
    fwhm_diff = float(np.max(np.abs(got - wrong_fwhm)) / np.max(got))
    cw_diff = float(np.max(np.abs(got0 - wrong_cw)) / np.max(got0))

    record("fwhm_misinterpretation_err", fwhm_diff)
    record("cw_winding_err", cw_diff)
    assert fwhm_diff > 0.25
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
    # Include bright edge pixels so zero-padding beyond boundaries is actively tested
    x = rng.normal(loc=1.5, scale=1.0, size=(20, 24))
    y = rng.normal(loc=-0.5, scale=0.8, size=(20, 24))

    got_det, got_coarse = (np.asarray(a) for a in vi.starlet(x, scales=4))
    want_det, want_coarse = ref.starlet_ref(x, scales=4, mode="constant")
    max_det_err = float(np.max(np.abs(got_det - want_det)))
    max_coarse_err = float(np.max(np.abs(got_coarse - want_coarse)))
    recon_err = float(np.max(np.abs(got_det.sum(axis=0) + got_coarse - x)))

    # Linearity: T(a X + b Y) == a T(X) + b T(Y)
    det_y, coarse_y = (np.asarray(a) for a in vi.starlet(y, scales=4))
    det_lin, coarse_lin = (np.asarray(a) for a in vi.starlet(1.7 * x - 0.9 * y, scales=4))
    lin_err = max(
        float(np.max(np.abs(det_lin - (1.7 * got_det - 0.9 * det_y)))),
        float(np.max(np.abs(coarse_lin - (1.7 * got_coarse - 0.9 * coarse_y)))),
    )

    # Dyadic scale selectivity: fine impulse peaks in scale 0; broad sigma=3.5 px blob peaks in coarser detail scale
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
    # Reflective or periodic boundary has identically zero detail planes on a constant image,
    # whereas zero-padding ("beyond the edges the image is zero") sees the step at the border.
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
    eta[3, 4] += 3.0  # off-centre bright feature
    scale, sigma = 1.8, 2.5

    # Rotated + shifted Image inside a System
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

    # Gradient wrt log_brightness matches finite differences
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

    # rotation_deg="grid" uses the passed grid angle
    grid_img = vm.Image(eta, scale, dra=1.3, ddec=-0.9, rotation_deg="grid")
    got_grid_c = np.asarray(vi.Centroid(sigma).centroid(grid_img, grid=32.0))
    grid_c_err = float(np.max(np.abs(got_grid_c - want["centroid"])))

    # LinearImage at rotation_deg = 0.0 and -22.0 matches sky_centroid_ref
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
    eta[2, 7] = 5.0  # near local "up" axis
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

    # GaussianField latent prior is i.i.d. Normal(0, 1)
    rng = np.random.default_rng(3)
    z = rng.normal(size=np.asarray(gp_field.latent).shape)
    lp_got = float(np.sum(priors["gp.log_brightness.latent"].log_prob(z)))
    lp_want = float(np.sum(-0.5 * z**2 - 0.5 * np.log(2.0 * np.pi)))
    gp_lp_err = abs(lp_got - lp_want)

    # Plain Image prior is flat (constant log_prob = 0)
    lp_plain = float(np.sum(priors["plain.log_brightness"].log_prob(rng.normal(size=(8, 8)))))

    # LinearImage gets a flat prior on "pixels"
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

    # Unresolved source -> 0.0
    unresolved = _simple_v2_oidata(u_m, v_m, wavel, np.ones_like(u_m), d_v2)
    got_zero = vi.second_moment_fwhm(unresolved)

    # Fewer than 2 samples above v2_min -> None; DISCO data -> None
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
    headline="max_field_err_mas",
)
def test_image_field_and_beam_floor_ceiling_and_restoration():
    wavel = 2.2e-6
    u_m = np.array([5.0, 15.0, 40.0, 80.0])
    v_m = np.zeros_like(u_m)
    b_min, b_max = 5.0 / wavel, 80.0 / wavel
    data = _simple_v2_oidata(u_m, v_m, wavel, np.array([0.95, 0.85, 0.50, 0.15]), np.full(4, 0.02))

    # Intermediate FWHM, floor-clipped, ceiling-clipped, aperture-clipped, and crossover (floor > ceiling)
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

    # DISCO data has no V^2 amplitudes -> returns ceiling
    disco = vo.OIData(vc.ami_grid_record(wavelength_m=4.8e-6))
    got_disco = float(vi.image_field(disco, aperture_mas=320.0))
    want_disco = float(min(vi.field_of_view(disco), 320.0))
    max_field_err = max(max_field_err, abs(got_disco - want_disco))

    # Beam dataclass + convolve_beam second moments
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


# ---------------------------------------------------------------------------
# Lattice rotation_deg="grid" (#705) & CleanResult
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.imaging.CleanResult",
    "virgil.imaging.residual_image",
    roots=["mathematics"],
    headline="grid_model_diff",
)
def test_rotation_deg_grid_across_image_dirty_clean_and_residual():
    rot = -18.5
    template = vo.OIData(vc.ami_grid_record(wavelength_m=4.8e-6, rotation_deg=rot))
    assert abs(float(template.uv_grid.rotation_deg) - rot) < 1e-6

    rng = np.random.default_rng(23)
    eta = rng.normal(scale=0.5, size=(15, 15))
    scene_grid = vm.System(star=vm.PointSource(), env=vm.Image(eta, 25.0, flux=0.08, rotation_deg="grid"))
    scene_num = vm.System(star=vm.PointSource(), env=vm.Image(eta, 25.0, flux=0.08, rotation_deg=rot))
    grid_model_diff = float(np.max(np.abs(np.asarray(template.model(scene_grid) - template.model(scene_num)))))

    # Without a uv lattice, rotation_deg="grid" falls back to North up (0.0)
    ungridded = eqx.tree_at(lambda d: d.uv_grid, template, None, is_leaf=lambda x: x is None)
    scene_zero = vm.System(star=vm.PointSource(), env=vm.Image(eta, 25.0, flux=0.08, rotation_deg=0.0))
    fallback_diff = float(np.max(np.abs(np.asarray(ungridded.model(scene_grid) - ungridded.model(scene_zero)))))

    # Simulating a companion aligned with a pixel of the rotated lattice
    truth_pix = np.full((15, 15), -30.0)
    truth_pix[4, 10] = 0.0
    truth = vm.System(star=vm.PointSource(), comp=vm.Image(truth_pix, 25.0, flux=0.06, rotation_deg=rot))
    sim = template.with_model(truth, key=jax.random.PRNGKey(5))

    # dirty_image and residual_image with rotation_deg="grid" match explicit rotation_deg=rot
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

    # clean with rotation_deg="grid" populates CleanResult.rotation_deg andCleanResult.restored(beam)
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

    bm = vi.Beam(major_mas=80.0, minor_mas=50.0, pa_deg=25.0)
    restored = np.asarray(clean_res.restored(bm))
    want_restored = np.asarray(
        vi.convolve_beam(clean_res.components, 25.0, vi.Beam(80.0, 50.0, 25.0 - rot))
    )
    restored_diff = float(np.max(np.abs(restored - want_restored)))
    flux_conserved_err = abs(float(np.sum(restored) - np.sum(clean_res.components)))

    record("grid_model_diff", grid_model_diff)
    record("dirty_and_resid_diff", max(dirty_diff, resid_diff))
    record("restored_beam_diff", restored_diff)
    assert grid_model_diff < 1e-13
    assert fallback_diff < 1e-13
    assert dirty_diff < 1e-13
    assert resid_diff < 1e-13
    assert restored_diff < 1e-13
    assert flux_conserved_err < 5e-2  # small edge leakage only if near border; interior component conserves flux


# ---------------------------------------------------------------------------
# virgil.fitting.kkt_report (WF4: sqrt, linear, softmax, nuisance start= gate)
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
    headline="kkt_rel_diff",
)
def test_kkt_report_sqrt_linear_image_stationarity_and_exact_zero_pixel(tmp_path):
    n, scale, flux = 8, 1.0, 0.3
    shape = ref.gaussian_blob_ref(n, scale, sigma_mas=1.8, dra=0.8, ddec=-0.6) * flux
    truth = vm.System(
        star=vm.PointSource(),
        env=vm.LinearImage(np.sqrt(shape), scale, parametrization="sqrt", reference_flux=flux),
    )
    data = _three_ut_oidata(tmp_path / "sqrt.fits", truth, seed=3)

    init_sqrt = np.sqrt(np.full((n, n), flux / (n * n)))
    start = vm.System(
        star=vm.PointSource(),
        env=vm.LinearImage(init_sqrt, scale, parametrization="sqrt", reference_flux=flux),
    )
    priors = vi.image_priors(start)
    reg = [vi.MaxEntropy(15.0, path="env")]
    done = vfit.fit(start, priors, data, reg, max_steps=2000, gtol=1e-12)
    rep = vfit.kkt_report(done, priors, data, reg, start={"env.pixels": init_sqrt})
    img_rep = rep["images"]["env"]

    # Independent gradient wrt pixel fluxes f = x^2 at the solution
    f_opt = np.asarray(done.model.env.fluxes)

    def loss_of_flux(f_arr):
        m = vm.System(
            star=vm.PointSource(),
            env=vm.LinearImage(jnp.sqrt(f_arr), scale, parametrization="sqrt", reference_flux=flux),
        )
        return -vl.model_loglike(m, data) + reg[0].value(m)

    grad_f = np.asarray(jax.grad(loss_of_flux)(jnp.asarray(f_opt)))
    # For LinearImage("sqrt") mu = 0, and kkt_w normalizes by w / sum(f_opt)
    lit = f_opt > 1e-3 * np.max(f_opt)
    want_kkt_w = float(np.sqrt(np.mean(grad_f[lit] ** 2)) / (15.0 / float(np.sum(f_opt))))
    kkt_w_diff = abs(float(img_rep["kkt_w"]) - want_kkt_w)
    kkt_rel_val = float(img_rep["kkt_rel"])

    # Exact x = 0 pixel is a stationary point of x^2 and is NOT flagged by want_grow
    x_zero = np.asarray(done.model.env.pixels).copy()
    x_zero[0, 0] = 0.0
    zero_res = dataclasses.replace(
        done,
        model=vm.System(
            star=vm.PointSource(),
            env=vm.LinearImage(x_zero, scale, parametrization="sqrt", reference_flux=flux),
        ),
        values={**done.values, "env.pixels": jnp.asarray(x_zero)},
    )
    rep_zero = vfit.kkt_report(zero_res, priors, data, [vi.TSV(15.0, beam_mas=scale, path="env")])

    record("kkt_rel_diff", kkt_w_diff)
    record("kkt_rel", kkt_rel_val)
    record("band_limit", float(img_rep["band_limit"]))
    assert rep["converged"] is True
    assert float(img_rep["band_limit"]) < float(vfit.BAND_TOL)
    assert kkt_w_diff < 1e-11
    assert kkt_rel_val < float(vfit.KKT_TOL)
    assert float(img_rep["want_grow"]) == 0.0
    assert float(rep_zero["images"]["env"]["want_grow"]) == 0.0


@pytest.mark.validates(
    "virgil.fitting.kkt_report",
    roots=["mathematics"],
    headline="kkt_w_diff",
)
def test_kkt_report_bounded_linear_image_projected_gradient_and_active_zeros(tmp_path):
    n, scale, flux = 8, 1.0, 0.25
    # Compact source so outer pixels hit the active f = 0 lower bound under TSV
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
    done = vfit.fit(start, priors, data, reg, method="lbfgsb", max_steps=2000)
    rep = vfit.kkt_report(done, priors, data, reg, start={"env.pixels": init_pix})
    img_rep = rep["images"]["env"]

    f_opt = np.asarray(done.model.env.fluxes)

    def loss_of_flux(f_arr):
        m = vm.System(
            star=vm.PointSource(),
            env=vm.LinearImage(f_arr, scale, parametrization="linear", reference_flux=flux),
        )
        return -vl.model_loglike(m, data) + reg[0].value(m)

    grad_f = np.asarray(jax.grad(loss_of_flux)(jnp.asarray(f_opt)))
    ref_lin = ref.kkt_nonneg_linear_metrics_ref(
        f_opt, grad_f, weight=20.0, reference_flux=float(np.sum(f_opt))
    )
    kkt_w_diff = abs(float(img_rep["kkt_w"]) - ref_lin["kkt_w"])

    record("kkt_w_diff", kkt_w_diff)
    record("pg_rel", float(img_rep["pg_rel"]))
    assert rep["converged"] is True
    assert float(img_rep["zero_violation"]) == 0.0
    assert kkt_w_diff < 1e-6


@pytest.mark.validates(
    "virgil.fitting.kkt_report",
    roots=["mathematics"],
    headline="nuisance_grad_rel",
)
def test_kkt_report_softmax_image_multiplier_and_nuisance_start_gate(tmp_path):
    n, scale = 10, 0.65
    east, north = ref.sky_grid_mas(n, scale)
    # Exclude central r < 1 mas so the envelope cannot mimic the central unresolved star
    support = np.hypot(east, north) >= 1.0
    blob = ref.gaussian_blob_ref(n, scale, sigma_mas=1.3, dra=1.5, ddec=-1.0) * support
    blob /= blob.sum()
    init_blob = ref.gaussian_blob_ref(n, scale, sigma_mas=1.8) * support
    init_blob /= init_blob.sum()
    init_eta = np.where(support, np.log(np.clip(init_blob, 1e-12, None)), 0.0)

    truth = vm.System(star=vm.PointSource(), env=vm.Image.from_brightness(blob, scale, flux=0.22, support=support))
    data = _three_ut_oidata(tmp_path / "softmax.fits", truth, seed=14)

    start = vm.System(star=vm.PointSource(), env=vm.Image(init_eta, scale, flux=0.28, support=support))
    priors = vi.image_priors(start) | {"env.flux": dist.LogUniform(1e-2, 1.0)}
    reg = [vi.TSV(80.0, beam_mas=scale, path="env")]
    start_vals = {"env.log_brightness": init_eta, "env.flux": 0.28}
    done = vfit.fit(start, priors, data, reg, init=start_vals, max_steps=2000, gtol=1e-13)

    rep_ok = vfit.kkt_report(done, priors, data, reg, start=start_vals)
    assert rep_ok["converged"] is True
    assert float(rep_ok["nuisances"]["env.flux"]["grad_rel"]) < float(vfit.NUISANCE_TOL)

    # Verify independent simplex KKT stationarity metric against kkt_report on live support pixels
    b_opt = np.asarray(done.model.env.brightness)
    f_env = float(done.values["env.flux"])

    def loss_of_b(b_arr):
        s_flux = 1.0 + f_env * (1.0 - jnp.sum(b_arr))
        m_vis = vm.System(
            star=vm.PointSource(flux=s_flux),
            env=vm.LinearImage(b_arr * f_env, scale, parametrization="linear", reference_flux=f_env),
        )
        m_reg = vm.System(
            star=vm.PointSource(),
            env=vm.LinearImage(b_arr, scale, parametrization="linear", reference_flux=1.0),
        )
        return -vl.model_loglike(m_vis, data) + reg[0].value(m_reg)

    grad_b = np.asarray(jax.grad(loss_of_b)(jnp.asarray(b_opt)))
    ref_sm = ref.kkt_simplex_metrics_ref(b_opt, grad_b, support=support, flux=f_env, weight=80.0)
    assert abs(float(rep_ok["images"]["env"]["kkt_w"]) - ref_sm["kkt_w"]) < 1e-9

    # Perturb ONLY the nuisance "env.flux" while re-optimizing image pixels at the wrong flux:
    # image KKT stays converged, so without start= converged is True, whereas with start=
    # nuisance grad_rel >= NUISANCE_TOL flips converged to False (WF4)!
    bad_flux = float(done.values["env.flux"]) * 1.6
    fixed_flux_start = vm.System(
        star=vm.PointSource(),
        env=vm.Image(done.values["env.log_brightness"], scale, flux=bad_flux, support=support),
    )
    img_only = vfit.fit(fixed_flux_start, vi.image_priors(fixed_flux_start), data, reg, max_steps=2000, gtol=1e-13)
    perturbed = dataclasses.replace(
        done,
        model=img_only.model,
        values={"env.log_brightness": img_only.values["env.log_brightness"], "env.flux": jnp.asarray(bad_flux)},
    )
    rep_no_start = vfit.kkt_report(perturbed, priors, data, reg)
    rep_with_start = vfit.kkt_report(perturbed, priors, data, reg, start=start_vals)
    nuis_grad_rel = float(rep_with_start["nuisances"]["env.flux"]["grad_rel"])

    record("nuisance_grad_rel", nuis_grad_rel)
    assert rep_no_start["converged"] is True
    assert rep_with_start["converged"] is False
    assert nuis_grad_rel > float(vfit.NUISANCE_TOL)


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

    # LinearImage("linear") with brightest pixel clamped to 0 -> negative gradient -> zero_violation > 0
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
