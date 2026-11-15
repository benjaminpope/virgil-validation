"""Validation of virgil 0.4.2 companion model, field, angle, AMIGO, OIFITS, and gain nodes (Batch 5)."""

import gc
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.special import jv

import virgil.amigo as vamigo
import virgil.angles as vangles
import virgil.coverage as vcov
import virgil.fields as vfields
import virgil.likelihood as vlike
import virgil.models as vm
import virgil.observables as vobs
import virgil.oidata as voidata
import virgil.oifits as voifits
import virgil.spectra as vspec

from crosscheck import sky
from evidence.plugin import record


@pytest.fixture(scope="module", autouse=True)
def _clear_jax_caches():
    yield
    jax.clear_caches()
    gc.collect()


def _matern_corr(r: np.ndarray, ell: float, nu: float, jitter: float = 1e-6) -> np.ndarray:
    """Independent Matern correlation matrix in radius with diagonal jitter."""
    d = np.abs(r[:, None] - r[None, :]) / float(ell)
    if nu == 0.5:
        c = np.exp(-d)
    elif nu == 1.5:
        s = np.sqrt(3.0) * d
        c = (1.0 + s) * np.exp(-s)
    elif nu == 2.5:
        s = np.sqrt(5.0) * d
        c = (1.0 + s + s**2 / 3.0) * np.exp(-s)
    elif np.isinf(nu):
        c = np.exp(-0.5 * d**2)
    else:
        raise ValueError(f"Unsupported nu: {nu}")
    return c + jitter * np.eye(r.size)


# ---------------------------------------------------------------------------
# 1. Component, multipole_disk_cvis, cvis_radial_dirac_delta_modulated,
#    circular_support, RadialGPFunction, gp_priors
# ---------------------------------------------------------------------------


class _ToyGaussianComponent(vm.Component):
    """Minimal Component subclass for testing base-class shift and flux wiring."""

    sigma_mas: float = 1.5

    def _centred_cvis(self, uu, vv):
        q2 = (uu * sky.MAS * self.sigma_mas) ** 2 + (vv * sky.MAS * self.sigma_mas) ** 2
        return jnp.exp(-2.0 * jnp.pi**2 * q2)


@pytest.mark.validates(
    "virgil.models.Component",
    roots=["mathematics"],
    headline="component_system_max_err",
)
def test_component_subclass_shift_chromatic_flux_and_validation():
    """Component applies sky-offset phase shifts to _centred_cvis, evaluates
    scalar or Spectrum flux_at(wavel), mixes cleanly inside System, and rejects
    negative scalar flux at initialization."""
    rng = np.random.default_rng(11)
    u = rng.uniform(-70.0, 70.0, 24)
    v = rng.uniform(-70.0, 70.0, 24)
    wavel = np.linspace(1.9e-6, 2.4e-6, 24)

    spec = vspec.PowerLaw(ratio=0.4, index=-1.5, wavel0=2.2e-6)
    comp = _ToyGaussianComponent(sigma_mas=1.8, flux=spec, dra=7.5, ddec=-4.2)
    cvis = np.asarray(comp.model(u, v, wavel))
    cvis_ref = sky.vis_gaussian(u, v, wavel, sigma=1.8, dra=7.5, ddec=-4.2)
    err_single = float(np.max(np.abs(cvis - cvis_ref)))

    flux_vals = np.asarray(vspec.flux_at(comp.flux, wavel))
    flux_ref = 0.4 * (wavel / 2.2e-6) ** (-1.5)
    err_flux = float(np.max(np.abs(flux_vals - flux_ref)))

    scene = vm.System(star=vm.PointSource(flux=1.0, dra=-1.0, ddec=2.0), comp=comp)
    v_scene = np.asarray(scene.model(u, v, wavel))
    v_star = sky.vis_point(u, v, wavel, dra=-1.0, ddec=2.0)
    v_scene_ref = (1.0 * v_star + flux_ref * cvis_ref) / (1.0 + flux_ref)
    err_sys = float(np.max(np.abs(v_scene - v_scene_ref)))

    with pytest.raises(ValueError):
        _ToyGaussianComponent(sigma_mas=1.0, flux=-0.1, dra=0.0, ddec=0.0)

    record("component_system_max_err", max(err_single, err_flux, err_sys))
    assert err_single < 1e-14
    assert err_flux < 1e-14
    assert err_sys < 1e-14


@pytest.mark.validates(
    "virgil.models.Component",
    roots=["mathematics"],
    kind="control",
    headline="component_wrong_shift_sign_gap",
)
def test_control_component_opposite_shift_sign_fails():
    """Control: reversing the Fourier offset sign (-dra, -ddec) disagrees with Component.model."""
    u = np.array([45.0, -30.0, 60.0])
    v = np.array([-20.0, 50.0, 15.0])
    wavel = np.full(3, 2.2e-6)
    comp = _ToyGaussianComponent(sigma_mas=1.2, flux=1.0, dra=6.0, ddec=-3.5)
    v_true = np.asarray(comp.model(u, v, wavel))
    v_wrong = sky.vis_gaussian(u, v, wavel, sigma=1.2, dra=-6.0, ddec=3.5)
    gap = float(np.max(np.abs(v_true - v_wrong)))
    record("component_wrong_shift_sign_gap", gap)
    assert gap > 0.5


@pytest.mark.validates(
    "virgil.models.multipole_disk_cvis",
    roots=["mathematics"],
    headline="multipole_disk_cvis_max_err",
)
def test_multipole_disk_cvis_jacobi_anger_height_blur_and_stretch():
    """multipole_disk_cvis matches the exact Jacobi-Anger Bessel series with
    uv deprojection, disk-plane Gaussian blur, conical ring height phase shift,
    and negative stretch azimuth reflection."""
    rng = np.random.default_rng(22)
    uu = rng.uniform(-3.5e7, 3.5e7, 30)  # cycles / rad
    vv = rng.uniform(-3.5e7, 3.5e7, 30)

    r = np.array([2.5, 5.5, 9.0, 13.5])
    w_raw = np.array([0.8, 1.7, 1.3, 0.6])
    w = w_raw / np.sum(w_raw)
    inc, pa, blur_fwhm = 48.0, 33.0, 2.4
    c = np.array([[0.22, 0.18, 0.10, 0.05], [-0.12, 0.08, 0.15, -0.06]])  # m = 1, 2
    s = np.array([[-0.14, 0.09, 0.16, 0.04], [0.07, -0.11, 0.05, 0.09]])
    height = np.array([0.4, 1.1, 2.0, 3.2])

    v_vv = np.asarray(
        vm.multipole_disk_cvis(
            uu, vv, r, w_raw, inc=inc, pa=pa, blur_fwhm=blur_fwhm, c=c, s=s, height=height
        )
    )

    # Independent Jacobi-Anger sum from documented notes.md#multipoledisk:
    pa_rad = np.deg2rad(pa)
    inc_rad = np.deg2rad(inc)
    # Major axis along pa (North->East), minor axis along pa + 90 deg (East of North):
    q_maj = (uu * np.sin(pa_rad) + vv * np.cos(pa_rad)) * sky.MAS  # cycles / mas
    q_min_sky = (uu * np.cos(pa_rad) - vv * np.sin(pa_rad)) * sky.MAS
    q_min = q_min_sky * np.cos(inc_rad)
    q = np.hypot(q_maj, q_min)
    phi = np.arctan2(q_min, q_maj)

    # Height shifts ring k by h_k * sin(inc) towards pa + 90 deg on the sky:
    h_phase = np.exp(-2j * np.pi * q_min_sky[:, None] * (height[None, :] * np.sin(inc_rad)))

    # Order 0 + orders +-1, +-2:
    bessel_sum = np.sum(w[None, :] * h_phase * jv(0, 2.0 * np.pi * q[:, None] * r[None, :]), axis=1)
    for m in (1, 2):
        cm = c[m - 1]
        sm = s[m - 1]
        jm = jv(m, 2.0 * np.pi * q[:, None] * r[None, :])
        # (-i)^m [ I_+m e^{+i m phi} + I_-m e^{-i m phi} ] = (-i)^m (c_m cos m phi + s_m sin m phi)
        ang = cm[None, :] * np.cos(m * phi[:, None]) + sm[None, :] * np.sin(m * phi[:, None])
        bessel_sum = bessel_sum + ((-1j) ** m) * np.sum(w[None, :] * h_phase * ang * jm, axis=1)

    sigma_blur = blur_fwhm / sky.FWHM_PER_SIGMA
    env = np.exp(-2.0 * np.pi**2 * sigma_blur**2 * q**2)
    v_ref = bessel_sum * env
    err_main = float(np.max(np.abs(v_vv - v_ref)))

    # Negative stretch mirrors the minor-axis frequency (flips sign of s_m for odd/even sin m phi):
    stretch_neg = -np.cos(inc_rad)
    v_neg = np.asarray(vm.multipole_disk_cvis(uu, vv, r, w_raw, inc=inc, pa=pa, c=c, s=s, stretch=stretch_neg))
    v_mirror = np.asarray(vm.multipole_disk_cvis(uu, vv, r, w_raw, inc=inc, pa=pa, c=c, s=-s, stretch=-stretch_neg))
    err_stretch = float(np.max(np.abs(v_neg - v_mirror)))

    # Normalization V(0, 0) == 1:
    v0 = complex(np.asarray(vm.multipole_disk_cvis(0.0, 0.0, r, w_raw, inc=inc, pa=pa, c=c, s=s, height=height)))
    err_v0 = abs(v0 - 1.0)

    record("multipole_disk_cvis_max_err", max(err_main, err_stretch, err_v0))
    assert err_main < 1e-13
    assert err_stretch < 1e-14
    assert err_v0 < 1e-15


@pytest.mark.validates(
    "virgil.models.cvis_radial_dirac_delta_modulated",
    roots=["mathematics"],
    headline="dirac_ring_modulated_max_err",
)
def test_cvis_radial_dirac_delta_modulated_against_bessel_and_ring_cloud():
    """cvis_radial_dirac_delta_modulated matches both the closed-form Bessel
    expansion and direct azimuthal quadrature of a thin ring (North=0, East=90)."""
    rng = np.random.default_rng(33)
    wavel = 2.2e-6
    u_m = rng.uniform(-60.0, 60.0, 28)
    v_m = rng.uniform(-60.0, 60.0, 28)
    uu = u_m / wavel
    vv = v_m / wavel

    r0 = 6.5
    az_amps = np.array([0.35, -0.18, 0.11])
    az_phis = np.array([25.0, 110.0, -40.0])

    v_vv = np.asarray(vm.cvis_radial_dirac_delta_modulated(uu, vv, r0, az_amps, az_phis))

    # 1. Direct point-cloud quadrature via crosscheck.sky.inclined_ring (inc=0):
    ring_cloud = sky.inclined_ring(
        diam=2.0 * r0, inc=0.0, pa=0.0, az_amps=az_amps, az_pas=az_phis, modulation="sky", n_phi=4096
    )
    v_cloud = sky.visibility(ring_cloud, u_m, v_m, wavel)

    # 2. Analytic Jacobi-Anger series:
    q = np.hypot(uu, vv) * sky.MAS
    psi_uv = np.arctan2(uu, vv)  # North -> East
    v_bessel = jv(0, 2.0 * np.pi * q * r0).astype(complex)
    for m, (amp, phi_deg) in enumerate(zip(az_amps, az_phis), start=1):
        v_bessel = v_bessel + amp * ((-1j) ** m) * jv(m, 2.0 * np.pi * q * r0) * np.cos(
            m * (psi_uv - np.deg2rad(phi_deg))
        )

    err_bessel = float(np.max(np.abs(v_vv - v_bessel)))
    err_cloud = float(np.max(np.abs(v_vv - v_cloud)))

    # Cardinal orientation check: m=1, phi=0 is bright North -> Im V < 0 at small +v;
    # m=1, phi=90 is bright East -> Im V < 0 at small +u:
    v_north = complex(
        np.asarray(vm.cvis_radial_dirac_delta_modulated(0.0, 5.0 / wavel, r0, np.array([0.5]), np.array([0.0])))
    )
    v_east = complex(
        np.asarray(vm.cvis_radial_dirac_delta_modulated(5.0 / wavel, 0.0, r0, np.array([0.5]), np.array([90.0])))
    )
    assert v_north.imag < -0.05
    assert abs(v_north.imag - v_east.imag) < 1e-14

    record("dirac_ring_modulated_max_err", max(err_bessel, err_cloud))
    assert err_bessel < 1e-14
    assert err_cloud < 1e-12


@pytest.mark.validates(
    "virgil.models.cvis_radial_dirac_delta_modulated",
    roots=["mathematics"],
    kind="control",
    headline="dirac_ring_wrong_conjugation_gap",
)
def test_control_dirac_ring_opposite_fourier_sign_fails():
    """Control: replacing (-i)^m with (+i)^m inverts odd-harmonic phases and fails."""
    wavel = 2.2e-6
    uu = np.array([18.0, -25.0, 32.0]) / wavel
    vv = np.array([22.0, 14.0, -19.0]) / wavel
    r0 = 5.0
    v_vv = np.asarray(vm.cvis_radial_dirac_delta_modulated(uu, vv, r0, np.array([0.4]), np.array([35.0])))
    q = np.hypot(uu, vv) * sky.MAS
    psi_uv = np.arctan2(uu, vv)
    v_wrong = jv(0, 2.0 * np.pi * q * r0) + 0.4 * (1j) * jv(1, 2.0 * np.pi * q * r0) * np.cos(
        psi_uv - np.deg2rad(35.0)
    )
    gap = float(np.max(np.abs(v_vv - v_wrong)))
    record("dirac_ring_wrong_conjugation_gap", gap)
    assert gap > 0.2


@pytest.mark.validates(
    "virgil.models.circular_support",
    roots=["mathematics"],
    headline="circular_support_mismatch_pixels",
)
def test_circular_support_solid_and_annular_mask_with_image():
    """circular_support selects pixels with inner_radius_mas <= r <= radius_mas
    about the symmetric image centre ((npix - 1)/2, (npix - 1)/2), and Image
    restricts flux strictly to that support."""
    npix, scale = 32, 0.75
    r_out, r_in = 8.5, 2.5

    mask_solid = np.asarray(vm.circular_support(npix, scale, r_out))
    mask_ann = np.asarray(vm.circular_support(npix, scale, r_out, inner_radius_mas=r_in))

    c = 0.5 * (npix - 1)
    yy, xx = np.meshgrid(np.arange(npix), np.arange(npix), indexing="ij")
    rr = np.hypot((yy - c) * scale, (xx - c) * scale)
    ref_solid = rr <= r_out
    ref_ann = (rr <= r_out) & (rr >= r_in)

    mismatch = int(np.sum(mask_solid != ref_solid) + np.sum(mask_ann != ref_ann))

    img = vm.Image(np.zeros((npix, npix)), pixel_scale_mas=scale, support=mask_ann)
    b = np.asarray(img.brightness)
    outside_flux = float(np.sum(b[~ref_ann]))
    total_flux = float(np.sum(b))

    record("circular_support_mismatch_pixels", float(mismatch))
    record("circular_support_outside_flux", outside_flux)
    assert mismatch == 0
    assert outside_flux == 0.0
    assert np.isclose(total_flux, 1.0, atol=1e-14)


@pytest.mark.validates(
    "virgil.models.RadialGPFunction",
    "virgil.models.gp_priors",
    roots=["mathematics"],
    headline="radial_gp_function_max_err",
)
def test_radial_gp_function_and_gp_priors_across_scene():
    """RadialGPFunction evaluates mu + sigma * L @ z with linear interpolation
    and constant clamping, works as MultipoleDisk.spectral_index, and gp_priors
    assigns Normal(0, 1) to whitened z and bounded LogUniform to hyperparameters."""
    r_in, r_out, n = 1.5, 13.5, 10
    sigma, ell, mu = 0.65, 3.2, -0.4
    rng = np.random.default_rng(44)
    z = rng.normal(size=n)

    max_gp_err = 0.0
    for nu in (0.5, 1.5, 2.5, np.inf):
        gp_fn = vm.RadialGPFunction(r_in=r_in, r_out=r_out, sigma=sigma, ell=ell, mu=mu, z=z, nu=nu)
        nodes = np.linspace(r_in, r_out, n)
        L = cholesky(_matern_corr(nodes, ell=ell, nu=nu, jitter=1e-6), lower=True)
        g_nodes_ref = mu + sigma * (L @ z)

        r_probe = np.array([0.3, 1.5, 2.9, 6.8, 11.2, 13.5, 18.0])
        g_probe = np.asarray(gp_fn(r_probe))
        g_ref = np.interp(r_probe, nodes, g_nodes_ref, left=g_nodes_ref[0], right=g_nodes_ref[-1])
        max_gp_err = max(max_gp_err, float(np.max(np.abs(g_probe - g_ref))))

    draws_signed = np.asarray(
        vm.RadialGPFunction(r_in, r_out, sigma=1.2, ell=2.0, mu=-1.0, n=8).draws(jax.random.PRNGKey(7), n=32)
    )
    draws_pos = np.asarray(
        vm.RadialGPFunction(r_in, r_out, sigma=1.2, ell=2.0, mu=-1.0, n=8).draws(
            jax.random.PRNGKey(7), n=32, positive=True
        )
    )
    assert draws_signed.shape == (32, 8)
    assert np.any(draws_signed < 0.0)
    assert np.all(draws_pos > 0.0)

    # Full MultipoleDisk scene with RadialGP profile, MultipoleKernel harmonics, and RadialGPFunction spectral_index:
    disk = vm.MultipoleDisk(
        profile=vm.RadialGP(1.0, 12.0, sigma=0.8, ell=3.0),
        harmonics=vm.Harmonics(degree=2, prior=vm.MultipoleKernel(ell_r=3.0, ell_theta=2.5, sigma=0.25), form="linear"),
        spectral_index=vm.RadialGPFunction(sigma=0.4, ell=4.0, mu=-0.5, n=8),
        wavel0=2.2e-6,
        n_r=24,
    )
    scene = vm.System(star=vm.PointSource(flux=1.0), disk=disk)
    # Chromatic evaluation exercises spectral_index RadialGPFunction on disk's profile range:
    cvis_20 = np.asarray(scene.model(np.array([35.0]), np.array([-20.0]), np.array([2.0e-6])))
    cvis_24 = np.asarray(scene.model(np.array([35.0]), np.array([-20.0]), np.array([2.4e-6])))
    assert np.all(np.isfinite(cvis_20)) and not np.allclose(cvis_20, cvis_24)

    priors_all = vm.gp_priors(scene, sigma=(0.05, 2.5), ell=(0.5, 10.0), ell_theta=(0.4, 8.0))
    expected_keys = {
        "disk.profile.z",
        "disk.profile.sigma",
        "disk.profile.ell",
        "disk.harmonics.z",
        "disk.harmonics.prior.sigma",
        "disk.harmonics.prior.ell_r",
        "disk.harmonics.prior.ell_theta",
        "disk.spectral_index.z",
        "disk.spectral_index.sigma",
        "disk.spectral_index.ell",
    }
    assert set(priors_all.keys()) == expected_keys

    # LogUniform scale invariance check: log_prob(x1) - log_prob(x2) == -log(x1 / x2)
    lp_diff = float(priors_all["disk.profile.sigma"].log_prob(0.2) - priors_all["disk.profile.sigma"].log_prob(0.8))
    lp_err = abs(lp_diff - (-np.log(0.2 / 0.8)))

    # Leaving bounds None fixes hyperparameters and emits only latent z priors:
    priors_latents_only = vm.gp_priors(scene)
    assert set(priors_latents_only.keys()) == {"disk.profile.z", "disk.harmonics.z", "disk.spectral_index.z"}

    record("radial_gp_function_max_err", max(max_gp_err, lp_err))
    assert max_gp_err < 2e-12  # nu=inf RBF Cholesky has condition ~1/jitter = 1e6
    assert lp_err < 1e-14


# ---------------------------------------------------------------------------
# 2. polar_interpolate, multipole_field_priors, MultipoleField origin taper
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.fields.polar_interpolate",
    roots=["mathematics"],
    headline="polar_interp_refinement_ratio",
)
def test_polar_interpolate_nodes_periodicity_edge_taper_and_second_order():
    """polar_interpolate reproduces grid nodes, wraps periodically in azimuth,
    ramps linearly to fill over one spacing outside [r_min, r_max], and converges
    at second order O(h^2) on a smooth polar function."""
    n_r, n_th = 9, 16
    r_min, spacing, fill = 2.0, 1.5, -7.0
    r_nodes = r_min + np.arange(n_r) * spacing
    th_nodes = 2.0 * np.pi * np.arange(n_th) / n_th
    rr_g, tt_g = np.meshgrid(r_nodes, th_nodes, indexing="ij")
    vals = 1.2 + 0.5 * np.cos(rr_g / 4.0) * np.cos(tt_g - 0.4)

    # Exact at grid nodes and periodic under theta + 2*pi:
    at_nodes = np.asarray(vfields.polar_interpolate(vals, r_min, spacing, rr_g, tt_g, fill=fill))
    at_wrapped = np.asarray(vfields.polar_interpolate(vals, r_min, spacing, rr_g, tt_g + 2.0 * np.pi, fill=fill))
    err_nodes = float(np.max(np.abs(at_nodes - vals)))
    err_wrap = float(np.max(np.abs(at_nodes - at_wrapped)))

    # Linear ramp over one spacing outside [r_min, r_max] and constant fill beyond:
    r_max = r_min + (n_r - 1) * spacing  # 14.0
    th0 = th_nodes[3]
    v_inner_edge = float(vals[0, 3])
    v_outer_edge = float(vals[-1, 3])
    probe_r = np.array([r_min - 1.5 * spacing, r_min - 0.4 * spacing, r_max + 0.25 * spacing, r_max + 1.2 * spacing])
    got_edges = np.asarray(vfields.polar_interpolate(vals, r_min, spacing, probe_r, np.full(4, th0), fill=fill))
    expected_edges = np.array(
        [
            fill,
            0.6 * v_inner_edge + 0.4 * fill,
            0.75 * v_outer_edge + 0.25 * fill,
            fill,
        ]
    )
    err_edges = float(np.max(np.abs(got_edges - expected_edges)))

    # Second-order O(h^2) convergence under 2x refinement in both r and theta:
    def analytic_fn(r, th):
        return np.exp(-0.02 * r**2) * (1.0 + 0.4 * np.cos(2.0 * th + 0.3) * np.sin(0.25 * r))

    rng = np.random.default_rng(55)
    rp = rng.uniform(3.0, 11.0, 200)
    tp = rng.uniform(-np.pi, np.pi, 200)
    truth = analytic_fn(rp, tp)

    errors = []
    for nr, nt in ((17, 32), (33, 64)):
        sp = 10.0 / (nr - 1)
        rn = 2.0 + np.arange(nr) * sp
        tn = 2.0 * np.pi * np.arange(nt) / nt
        rg, tg = np.meshgrid(rn, tn, indexing="ij")
        pred = np.asarray(vfields.polar_interpolate(analytic_fn(rg, tg), 2.0, sp, rp, tp, fill=0.0))
        errors.append(float(np.max(np.abs(pred - truth))))

    ratio = errors[0] / errors[1]
    record("polar_interp_node_err", max(err_nodes, err_wrap, err_edges))
    record("polar_interp_refinement_ratio", ratio)
    assert err_nodes < 1e-14
    assert err_wrap < 1e-14
    assert err_edges < 2e-8  # boundary epsilon clamp in polar_interpolate
    assert 3.7 < ratio < 4.3


@pytest.mark.validates(
    "virgil.fields.multipole_field_priors",
    "virgil.fields.MultipoleField",
    roots=["mathematics", "statistics"],
    headline="origin_non_m0_std",
)
def test_multipole_field_priors_and_042_tanh_origin_regularity():
    """multipole_field_priors returns bounded LogUniform scale priors, and in
    0.4.2 MultipoleField(r_min_mas=0) multiplies azimuthal order m by
    tanh(r / length_r)^|m| so the origin is single-valued (only m=0 free at r=0)
    while an annulus (r_min_mas > 0) keeps a stationary inner edge."""
    pri = vfields.multipole_field_priors(
        path="disk_img.log_brightness", sigma=(0.1, 3.0), length_r=(1.0, 25.0), length_theta=(0.2, 5.0)
    )
    assert set(pri.keys()) == {
        "disk_img.log_brightness.sigma",
        "disk_img.log_brightness.length_r",
        "disk_img.log_brightness.length_theta",
    }
    lp_gap = float(
        pri["disk_img.log_brightness.length_r"].log_prob(2.0)
        - pri["disk_img.log_brightness.length_r"].log_prob(8.0)
    )
    assert np.isclose(lp_gap, -np.log(2.0 / 8.0), atol=1e-14)

    # Probe origin regularity on odd npix=33 (exact centre pixel at (16, 16), r = 0):
    rng = np.random.default_rng(66)
    n_draws = 256
    n_r, n_th = 16, 32
    latents = rng.normal(size=(n_draws, n_r, n_th))
    length_r = 6.0

    # Sample rings at r = 0 (via centre pixel) and r = 3.0 mas = 0.5 * length_r:
    # Build pixel grid of 65x65 with pixel_scale=0.5 mas so r=3.0 mas is 6 pixels out:
    npix, pix = 65, 0.5
    c_idx = (npix - 1) // 2

    # Image(field) uses softmax(log_field); log(brightness) up to an additive constant recovers eta(x, y):
    def log_eta(z_mat, r_min):
        img = vm.Image(
            vfields.MultipoleField(
                z_mat,
                image_shape=(npix, npix),
                r_max_mas=15.0,
                r_min_mas=r_min,
                sigma=1.0,
                length_r=length_r,
                length_theta=0.8,
                outside=-20.0,
            ),
            pixel_scale_mas=pix,
        )
        return jnp.log(img.brightness)

    eta_zero = np.asarray(jax.vmap(lambda z: log_eta(z, 0.0))(jnp.asarray(latents)))
    eta_ann = np.asarray(jax.vmap(lambda z: log_eta(z, 2.0))(jnp.asarray(latents)))

    # 1. At r_min_mas=0, check azimuthal variation on 1-pixel cardinal neighbours
    # (r = 0.5 mas = length_r / 12 -> tanh(1/12) ~ 0.083) vs outer ring at 20 pixels:
    card_1px = np.stack(
        [
            eta_zero[:, c_idx - 1, c_idx],
            eta_zero[:, c_idx, c_idx + 1],
            eta_zero[:, c_idx + 1, c_idx],
            eta_zero[:, c_idx, c_idx - 1],
        ],
        axis=-1,
    )  # shape (n_draws, 4)
    card_20px = np.stack(
        [
            eta_zero[:, c_idx - 20, c_idx],
            eta_zero[:, c_idx, c_idx + 20],
            eta_zero[:, c_idx + 20, c_idx],
            eta_zero[:, c_idx, c_idx - 20],
        ],
        axis=-1,
    )
    # Remove m=0 mean around each 4-point ring to measure m >= 1 azimuthal std:
    std_1px = float(np.sqrt(np.mean((card_1px - card_1px.mean(axis=-1, keepdims=True)) ** 2)))
    std_20px = float(np.sqrt(np.mean((card_20px - card_20px.mean(axis=-1, keepdims=True)) ** 2)))

    # For annulus r_min_mas=2.0 mas (4 pixels from centre), inner ring at 4 px is untapered (~ same azimuthal std as 20 px):
    ann_4px = np.stack(
        [
            eta_ann[:, c_idx - 4, c_idx],
            eta_ann[:, c_idx, c_idx + 4],
            eta_ann[:, c_idx + 4, c_idx],
            eta_ann[:, c_idx, c_idx - 4],
        ],
        axis=-1,
    )
    ann_20px = np.stack(
        [
            eta_ann[:, c_idx - 20, c_idx],
            eta_ann[:, c_idx, c_idx + 20],
            eta_ann[:, c_idx + 20, c_idx],
            eta_ann[:, c_idx, c_idx - 20],
        ],
        axis=-1,
    )
    std_ann_inner = float(np.sqrt(np.mean((ann_4px - ann_4px.mean(axis=-1, keepdims=True)) ** 2)))
    std_ann_outer = float(np.sqrt(np.mean((ann_20px - ann_20px.mean(axis=-1, keepdims=True)) ** 2)))

    # Check exact single-valuedness at r=0: a 90 deg azimuthal rotation of Z
    # (shift = n_th // 4) maps the square Cartesian grid onto itself (preserving
    # softmax normalization), shifts outer pixels by O(1), and leaves the exact
    # origin pixel (c_idx, c_idx) strictly invariant because tanh(0 / length_r)^|m| == 0
    # for all |m| >= 1!
    z0 = latents[0]
    z_rot90 = np.roll(z0, n_th // 4, axis=1)
    b_orig = np.asarray(
        vm.Image(
            vfields.MultipoleField(z0, image_shape=(npix, npix), r_max_mas=15.0, r_min_mas=0.0, length_r=length_r),
            pixel_scale_mas=pix,
        ).brightness
    )
    b_rot90 = np.asarray(
        vm.Image(
            vfields.MultipoleField(z_rot90, image_shape=(npix, npix), r_max_mas=15.0, r_min_mas=0.0, length_r=length_r),
            pixel_scale_mas=pix,
        ).brightness
    )
    origin_m1_diff = float(abs(b_orig[c_idx, c_idx] - b_rot90[c_idx, c_idx]))
    outer_rot_diff = float(np.max(np.abs(b_orig - b_rot90)))
    assert outer_rot_diff > 5e-5

    record("origin_non_m0_std", origin_m1_diff)
    record("taper_inner_to_outer_std_ratio", std_1px / std_20px)
    record("annulus_inner_to_outer_std_ratio", std_ann_inner / std_ann_outer)
    assert origin_m1_diff < 1e-15
    assert std_1px / std_20px < 0.15
    assert 0.80 < std_ann_inner / std_ann_outer < 1.25


# ---------------------------------------------------------------------------
# 3. virgil.angles.{is_angle_vector, vector_angle, vector_site}
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.angles.is_angle_vector",
    "virgil.angles.vector_angle",
    "virgil.angles.vector_site",
    roots=["mathematics"],
    headline="vector_angle_max_deg_err",
)
def test_angle_vector_helpers_quadrants_sites_and_density():
    """vector_angle recovers degrees in [0, 360) across all quadrants and batch
    shapes, vector_site appends _vec, and is_angle_vector distinguishes
    AngleVector from ordinary NumPyro distributions."""
    angles_deg = np.array([0.0, 37.5, 90.0, 142.3, 180.0, 225.7, 270.0, 359.85])
    radii = np.array([0.8, 1.1, 1.3, 0.9, 1.05, 0.75, 1.2, 1.0])
    th = np.deg2rad(angles_deg)
    vecs = np.stack([radii * np.cos(th), radii * np.sin(th)], axis=-1).reshape(2, 4, 2)

    got_deg = np.asarray(vangles.vector_angle(vecs))
    err_deg = float(np.max(np.abs(got_deg - angles_deg.reshape(2, 4))))

    assert vangles.vector_site("pa") == "pa_vec"
    assert vangles.vector_site("orbit.Omega") == "orbit.Omega_vec"
    assert vangles.is_angle_vector(vangles.AngleVector()) is True
    assert vangles.is_angle_vector(vangles.AngleVector(120.0, 10.0, axial=True)) is True
    assert vangles.is_angle_vector(dist.Uniform(0.0, 360.0)) is False
    assert vangles.is_angle_vector(None) is False

    record("vector_angle_max_deg_err", err_deg)
    assert got_deg.shape == (2, 4)
    assert np.all((got_deg >= 0.0) & (got_deg < 360.0))
    assert err_deg < 1e-12


# ---------------------------------------------------------------------------
# 4. AMIGO mixed-DISCO helpers & coverage.ami_grid_record
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.coverage.ami_grid_record",
    "virgil.amigo.is_mixed_disco_record",
    "virgil.amigo.validate_mixed_disco_record",
    "virgil.amigo.mixed_disco_fields",
    roots=["mathematics"],
    headline="mixed_disco_projection_err",
)
def test_ami_grid_record_and_amigo_mixed_disco_validation_and_projection():
    """ami_grid_record builds a valid mixed-DISCO record with SVD nuisance
    nulling; is_mixed_disco_record, validate_mixed_disco_record, and
    mixed_disco_fields enforce schema invariants and match A_logamp Re log V +
    A_phase Im log V to machine precision."""
    rec = vcov.ami_grid_record(wavelength_m=4.3e-6, pitch_m=0.8, rotation_deg=25.0, sigma=1e-4, keep=0.99)
    assert vamigo.is_mixed_disco_record(rec) is True
    assert vamigo.is_mixed_disco_record({"u": np.zeros(4), "v": np.zeros(4)}) is False
    vamigo.validate_mixed_disco_record(rec)

    fields = vamigo.mixed_disco_fields(rec)
    assert fields["vis_encoding"] == "mixed_log_complex"

    oid = voidata.OIData(rec)
    binary = vm.BinaryModelCartesian(dra=-45.0, ddec=160.0, flux=2e-3)
    oid_sim = oid.with_model(binary)

    # AMIGO records store (-u, -v), which OIData negates to physical (+u, +v):
    cvis = np.asarray(binary.model(-rec["u"], -rec["v"], np.full_like(rec["u"], 4.3e-6)))
    y_manual = rec["disco_logamp_model_operator"] @ np.log(np.abs(cvis)) + rec["disco_phase_model_operator"] @ np.angle(
        cvis
    )
    cvis_unnegated = np.asarray(binary.model(rec["u"], rec["v"], np.full_like(rec["u"], 4.3e-6)))
    y_wrong_sign = rec["disco_logamp_model_operator"] @ np.log(np.abs(cvis_unnegated)) + rec[
        "disco_phase_model_operator"
    ] @ np.angle(cvis_unnegated)
    y_oid = np.asarray(oid_sim.model(binary))
    err_proj = float(np.max(np.abs(y_oid - y_manual)))
    assert float(np.max(np.abs(y_oid - y_wrong_sign))) > 5e-3

    # SVD modes project out global flux scaling and first-order centroid shifts:
    tiny_shift = vm.PointSource(dra=1e-3, ddec=-1e-3)
    y_shift = np.asarray(oid.model(tiny_shift))
    null_leak = float(np.max(np.abs(y_shift)))

    # Validation rejects malformed records (missing key, mismatched operator shape, non-positive sigma):
    assert vamigo.is_mixed_disco_record({}) is False
    bad_no_tag = {k: v for k, v in rec.items() if k != "disco_coefficients"}
    assert vamigo.is_mixed_disco_record(bad_no_tag) is False
    bad_missing = {k: v for k, v in rec.items() if k != "disco_phase_model_operator"}
    with pytest.raises((KeyError, ValueError)):
        vamigo.validate_mixed_disco_record(bad_missing)

    bad_shape = {**rec, "disco_phase_model_operator": rec["disco_phase_model_operator"][:, :-1]}
    with pytest.raises(ValueError):
        vamigo.validate_mixed_disco_record(bad_shape)

    bad_sigma = {**rec, "disco_sigma": np.where(np.arange(rec["disco_sigma"].size) == 0, -1.0, rec["disco_sigma"])}
    with pytest.raises(ValueError):
        vamigo.validate_mixed_disco_record(bad_sigma)

    record("mixed_disco_projection_err", err_proj)
    record("ami_grid_centroid_null_leak", null_leak)
    assert err_proj < 1e-14
    assert null_leak < 1e-9


# ---------------------------------------------------------------------------
# 5. oifits.build_hdulist, oidata.{closure_phases, cp_indices}, observables.in_ranges
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.oifits.build_hdulist",
    "virgil.oidata.cp_indices",
    "virgil.oidata.closure_phases",
    "virgil.observables.in_ranges",
    roots=["standards", "mathematics"],
    headline="oifits_hdulist_roundtrip_err",
)
def test_build_hdulist_cp_indices_closure_phases_and_in_ranges(tmp_path):
    """build_hdulist produces OIFITS2 extensions modifiable before writing;
    cp_indices maps triangle legs with orientation checks; closure_phases wraps
    bispectrum phases into [-pi, pi); and in_ranges selects wavelength windows."""
    wls = np.array([2.0e-6, 2.15e-6, 2.3e-6, 2.45e-6], dtype=np.float32)
    mask = np.asarray(vobs.in_ranges(wls, [(1.95e-6, 2.05e-6), (2.28e-6, 2.46e-6)]))
    assert np.array_equal(mask, np.array([True, False, True, True]))
    assert np.array_equal(np.asarray(vobs.in_ranges(wls, [])), np.zeros(4, dtype=bool))

    vis_sta = np.array([[1, 2], [2, 3], [1, 3], [3, 4], [2, 4]], dtype=np.int16)
    cp_sta = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.int16)
    i1, i2, i3 = voidata.cp_indices(vis_sta, cp_sta)
    assert np.array_equal(i1, [0, 1])
    assert np.array_equal(i2, [1, 3])
    assert np.array_equal(i3, [2, 4])

    # Reversed orientation on leg 3 must raise ValueError:
    with pytest.raises(ValueError):
        voidata.cp_indices(np.array([[1, 2], [2, 3], [3, 1]]), np.array([[1, 2, 3]]))

    # closure_phases branch-cut wrap into [-pi, pi):
    phases = np.array([2.4, 2.5, -1.1, -2.2, 0.3])
    cvis = np.array([0.8, 0.7, 0.9, 0.6, 0.5]) * np.exp(1j * phases)
    cp = np.asarray(voidata.closure_phases(cvis, i1, i2, i3))
    cp_ref = np.angle(cvis[i1] * cvis[i2] * np.conj(cvis[i3]))
    # Replace exact +pi with -pi if needed for [-pi, pi):
    cp_ref = np.where(cp_ref >= np.pi, cp_ref - 2.0 * np.pi, cp_ref)
    err_cp = float(np.max(np.abs(cp - cp_ref)))
    assert np.all((cp >= -np.pi) & (cp < np.pi))

    # build_hdulist custom keyword injection + OIFITS2 round-trip:
    tables = {
        "OI_WAVELENGTH": {"EFF_WAVE": wls, "EFF_BAND": np.full(4, 0.05e-6, dtype=np.float32)},
        "OI_VIS2": {
            "UCOORD": np.array([35.0, -20.0, 15.0]),
            "VCOORD": np.array([12.0, 28.0, 40.0]),
            "VIS2DATA": np.full((3, 4), 0.88),
            "VIS2ERR": np.full((3, 4), 0.015),
            "STA_INDEX": np.array([[1, 2], [2, 3], [1, 3]], dtype=np.int16),
            "MJD": np.full(3, 60100.5),
        },
        "OI_T3": {
            "U1COORD": np.array([35.0]),
            "V1COORD": np.array([12.0]),
            "U2COORD": np.array([-20.0]),
            "V2COORD": np.array([28.0]),
            "T3PHI": np.array([[12.5, -8.0, 4.2, -15.0]]),
            "T3PHIERR": np.full((1, 4), 0.8),
            "STA_INDEX": np.array([[1, 2, 3]], dtype=np.int16),
            "MJD": np.array([60100.5]),
        },
    }
    hdul = voifits.build_hdulist(tables)
    assert hdul["OI_VIS2"].header["OI_REVN"] == 2
    hdul[0].header["HIERARCH VIRGIL TESTTAG"] = "BATCH5"
    out_fits = tmp_path / "custom_hdulist.fits"
    hdul.writeto(out_fits)

    oid = voidata.OIData(out_fits)
    err_v2 = float(np.max(np.abs(np.asarray(oid.vis) - 0.88)))
    err_t3 = float(np.max(np.abs(np.rad2deg(np.asarray(oid.phi)) - tables["OI_T3"]["T3PHI"].ravel())))
    err_rt = max(err_cp, err_v2, err_t3)

    record("oifits_hdulist_roundtrip_err", err_rt)
    assert err_cp < 1e-14
    assert err_rt < 1e-12


# ---------------------------------------------------------------------------
# 6. 0.4.2 ClosureOffsets(scale_penalty), OIData.with_gains(include_extras),
#    and OIData vis_mode="logamp" / vis_mode="v2"
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.gains.ClosureOffsets",
    "virgil.gains.closure_offsets",
    "virgil.gains.gain_modes",
    "virgil.oidata.OIData",
    roots=["mathematics"],
    headline="gain_marginal_loglike_err",
)
def test_042_closure_offsets_scale_penalty_and_gains_include_extras(tmp_path):
    """Pin 0.4.2 ClosureOffsets(scale_penalty=True/False) periodic wrap-penalty
    scaling (1 - cos Delta) / hypot(sigma, sigma_off) vs (1 - cos Delta) / sigma,
    and OIData.with_gains(include_extras=True/False) exact Gaussian marginal
    log-likelihood across V^2, VISAMP, and T3AMP."""
    wls = np.array([2.1e-6, 2.3e-6], dtype=np.float32)
    tables = {
        "OI_WAVELENGTH": {"EFF_WAVE": wls, "EFF_BAND": np.full(2, 0.05e-6, dtype=np.float32)},
        "OI_VIS2": {
            "UCOORD": np.array([40.0, -25.0, 15.0]),
            "VCOORD": np.array([10.0, 30.0, 40.0]),
            "VIS2DATA": np.full((3, 2), 0.90),
            "VIS2ERR": np.full((3, 2), 0.02),
            "STA_INDEX": np.array([[1, 2], [2, 3], [1, 3]], dtype=np.int16),
            "MJD": np.full(3, 60000.0),
        },
        "OI_VIS": {
            "UCOORD": np.array([40.0, -25.0, 15.0]),
            "VCOORD": np.array([10.0, 30.0, 40.0]),
            "VISAMP": np.full((3, 2), 0.94),
            "VISAMPERR": np.full((3, 2), 0.015),
            "VISPHI": np.zeros((3, 2)),
            "VISPHIERR": np.ones((3, 2)),
            "STA_INDEX": np.array([[1, 2], [2, 3], [1, 3]], dtype=np.int16),
            "MJD": np.full(3, 60000.0),
        },
        "OI_T3": {
            "U1COORD": np.array([40.0]),
            "V1COORD": np.array([10.0]),
            "U2COORD": np.array([-25.0]),
            "V2COORD": np.array([30.0]),
            "T3PHI": np.array([[175.0, 175.0]]),
            "T3PHIERR": np.array([[2.0, 2.0]]),
            "T3AMP": np.full((1, 2), 0.86),
            "T3AMPERR": np.full((1, 2), 0.025),
            "STA_INDEX": np.array([[1, 2, 3]], dtype=np.int16),
            "MJD": np.array([60000.0]),
        },
    }
    p = tmp_path / "gains_offsets.fits"
    voifits.build_hdulist(tables).writeto(p)

    # 1. ClosureOffsets scale_penalty=True vs False at Delta = 175 deg (cos Delta < 0):
    oid_base = voidata.OIData(voifits.read_oifits(p))
    sigma_rad = np.deg2rad(2.0)
    sigma_off = 0.25
    oid_sp_true = oid_base.with_closure_offsets(triangle=sigma_off, scale_penalty=True)
    oid_sp_false = oid_base.with_closure_offsets(triangle=sigma_off, scale_penalty=False)

    r_true = np.asarray(vlike.whitened_residuals(vm.PointSource(), oid_sp_true))
    r_false = np.asarray(vlike.whitened_residuals(vm.PointSource(), oid_sp_false))

    delta = np.deg2rad(175.0)
    pen_true_ref = (1.0 - np.cos(delta)) / np.hypot(sigma_rad, sigma_off)
    pen_false_ref = (1.0 - np.cos(delta)) / sigma_rad
    err_wrap = max(
        float(np.max(np.abs(r_true[-2:] - pen_true_ref))),
        float(np.max(np.abs(r_false[-2:] - pen_false_ref))),
    )

    # 2. OIData.with_gains(include_extras=True vs False) against dense Cholesky log-likelihood:
    raw_ext = voifits.read_oifits(p, extras=("visamp", "t3amp"))
    # Set T3PHI=0 so no closure-phase wrap occurs when checking amplitude gain marginalization:
    raw_ext["phi"] = np.zeros_like(raw_ext["phi"])
    oid_ext = voidata.OIData(raw_ext)
    tau_tel = 0.04
    g_inc = oid_ext.with_gains(telescope=tau_tel, include_extras=True)
    g_exc = oid_ext.with_gains(telescope=tau_tel, include_extras=False)
    assert g_inc.gains.n_vis == 14  # 6 V^2 + 6 VISAMP + 2 T3AMP
    assert g_exc.gains.n_vis == 6   # 6 V^2 only

    scene = vm.BinaryModelCartesian(dra=3.5, ddec=-2.5, flux=0.2)
    cvis = np.asarray(scene.model(oid_ext.u, oid_ext.v, oid_ext.wavel))
    v_abs = np.abs(cvis)
    v2_mod = v_abs**2
    t3_mod = v_abs[oid_ext.i_cps1] * v_abs[oid_ext.i_cps2] * v_abs[oid_ext.i_cps3]

    # Telescope design matrix B on log|V| for 3 stations (station 1..3 -> columns 0..2):
    # Each baseline (a, b) sums telescope gains g_a + g_b (all channels in 1 frame):
    sta = np.asarray(raw_ext["stations"]) - 1  # shape (6, 2)
    B_vis = np.zeros((6, 3))
    for row_i, (sa, sb) in enumerate(sta):
        B_vis[row_i, sa] += 1.0
        B_vis[row_i, sb] += 1.0
    # T3AMP sums over all three legs (2 * (g_1 + g_2 + g_3) on triangle (1,2,3)):
    B_t3 = B_vis[oid_ext.i_cps1] + B_vis[oid_ext.i_cps2] + B_vis[oid_ext.i_cps3]

    # Independent Gaussian log-likelihood (including -0.5*N*log(2*pi) or matching virgil's convention):
    # Check difference Delta log L = log L(with_gains) - log L(no_gains) so any constant cancels,
    # AND verify absolute log-determinant + Mahalanobis term directly!
    ll_no_gains = float(vlike.model_loglike(scene, oid_ext))
    ll_inc = float(vlike.model_loglike(scene, g_inc))
    ll_exc = float(vlike.model_loglike(scene, g_exc))

    y_all = np.concatenate([np.asarray(oid_ext.vis), np.full(6, 0.94), np.full(2, 0.86)])
    m_all = np.concatenate([v2_mod, v_abs, t3_mod])
    s_all = np.concatenate([np.asarray(oid_ext.d_vis), np.full(6, 0.015), np.full(2, 0.025)])
    res_all = y_all - m_all

    # Linearized Jacobian from log|V| modes to (V^2, |V|, T3AMP): (2*V^2, |V|, T3AMP)
    J_inc = np.vstack([2.0 * v2_mod[:, None] * B_vis, v_abs[:, None] * B_vis, t3_mod[:, None] * B_t3]) * tau_tel
    C_inc = np.diag(s_all**2) + J_inc @ J_inc.T
    cf_inc = cho_factor(C_inc)
    logdet_inc = 2.0 * np.sum(np.log(np.diag(cf_inc[0])))
    dll_inc_ref = -0.5 * (
        (logdet_inc + res_all @ cho_solve(cf_inc, res_all))
        - (np.sum(np.log(s_all**2)) + np.sum((res_all / s_all) ** 2))
    )

    J_exc = (2.0 * v2_mod[:, None] * B_vis) * tau_tel
    C_exc = np.diag(s_all[:6] ** 2) + J_exc @ J_exc.T
    cf_exc = cho_factor(C_exc)
    logdet_exc = 2.0 * np.sum(np.log(np.diag(cf_exc[0])))
    dll_exc_ref = -0.5 * (
        (logdet_exc + res_all[:6] @ cho_solve(cf_exc, res_all[:6]))
        - (np.sum(np.log(s_all[:6] ** 2)) + np.sum((res_all[:6] / s_all[:6]) ** 2))
    )

    err_gain_ll = max(abs((ll_inc - ll_no_gains) - dll_inc_ref), abs((ll_exc - ll_no_gains) - dll_exc_ref))

    record("gain_marginal_loglike_err", err_gain_ll)
    record("closure_wrap_penalty_err", err_wrap)
    assert err_wrap < 1e-13
    assert err_gain_ll < 1e-10
    assert abs(ll_inc - ll_exc) > 10.0


@pytest.mark.validates(
    "virgil.oidata.OIData",
    roots=["mathematics"],
    headline="vis_mode_encoding_err",
)
def test_042_oidata_vis_mode_logamp_and_v2_conversions():
    """Pin 0.4.2 OIData vis_mode='logamp' (drops V^2 <= 0 with warning, transforms
    sigma / (2 V^2), projects log|V_model| via vis_mat, and rejects unknown modes)
    and vis_mode='v2' (exact variance sigma * sqrt(4 V^2 + 2 sigma^2))."""
    rec_v2 = {
        "u": np.array([20.0, 35.0, 50.0, 65.0]),
        "v": np.array([10.0, -15.0, 25.0, -30.0]),
        "wavel": np.full(4, 2.2e-6),
        "vis": np.array([0.81, -0.02, 0.0, 0.49]),
        "d_vis": np.array([0.04, 0.03, 0.02, 0.05]),
        "v2_flag": True,
        "phi": np.zeros(0),
        "d_phi": np.zeros(0),
        "i_cps1": np.zeros(0, dtype=int),
        "i_cps2": np.zeros(0, dtype=int),
        "i_cps3": np.zeros(0, dtype=int),
        "cp_flag": False,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        oid_log = voidata.OIData({**rec_v2, "vis_mode": "logamp"})
    assert any("V² <= 0" in str(w.message) for w in caught)
    assert oid_log.vis.size == 2

    kept_v2 = np.array([0.81, 0.49])
    kept_dv2 = np.array([0.04, 0.05])
    err_logamp = max(
        float(np.max(np.abs(np.asarray(oid_log.vis) - 0.5 * np.log(kept_v2)))),
        float(np.max(np.abs(np.asarray(oid_log.d_vis) - kept_dv2 / (2.0 * kept_v2)))),
    )

    # Projected vis_mat + vis_mode="logamp" projects log|V_model| linearly, and invalid vis_mode raises ValueError:
    vis_mat = np.array([[0.6, 0.4, 0.0, 0.0], [0.0, -0.3, 0.5, 0.8]])
    oid_proj = voidata.OIData(
        {**rec_v2, "vis": np.array([0.05, -0.02]), "d_vis": np.array([0.01, 0.01]), "vis_mat": vis_mat, "vis_mode": "logamp"}
    )
    scene_p = vm.BinaryModelCartesian(dra=12.0, ddec=-8.0, flux=0.3)
    cvis_p = np.asarray(scene_p.model(rec_v2["u"], rec_v2["v"], rec_v2["wavel"]))
    err_proj_logamp = float(np.max(np.abs(np.asarray(oid_proj.model(scene_p)) - (vis_mat @ np.log(np.abs(cvis_p))))))
    assert err_proj_logamp < 1e-15

    with pytest.raises(ValueError, match="Unsupported vis_mode"):
        voidata.OIData({**rec_v2, "vis": np.full(4, 0.8), "vis_mode": "bogus"})

    # Amplitude input (v2_flag=False) converted to V^2 via vis_mode="v2":
    amp = np.array([0.9, 0.5, 0.2])
    sig = np.array([0.03, 0.04, 0.05])
    oid_v2 = voidata.OIData(
        {
            **rec_v2,
            "u": rec_v2["u"][:3],
            "v": rec_v2["v"][:3],
            "wavel": rec_v2["wavel"][:3],
            "vis": amp,
            "d_vis": sig,
            "v2_flag": False,
            "vis_mode": "v2",
        }
    )
    v2_ref = amp**2
    dv2_ref = sig * np.sqrt(4.0 * v2_ref + 2.0 * sig**2)
    err_v2_conv = max(
        float(np.max(np.abs(np.asarray(oid_v2.vis) - v2_ref))),
        float(np.max(np.abs(np.asarray(oid_v2.d_vis) - dv2_ref))),
    )

    record("vis_mode_encoding_err", max(err_logamp, err_v2_conv))
    assert err_logamp < 1e-15
    assert err_v2_conv < 1e-15
