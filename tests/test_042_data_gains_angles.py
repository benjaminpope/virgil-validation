"""Validation of virgil 0.4.2 companion model, field, angle, AMIGO, OIFITS, and gain nodes (Batch 5)."""

import gc
import warnings

from astropy.io import fits
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.special import jv, roots_legendre

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


def _circle_overlap(s: np.ndarray, d: float) -> np.ndarray:
    """Area common to two circular apertures of diameter d separated by s."""
    r = 0.5 * float(d)
    s_clip = np.minimum(np.abs(s), 2.0 * r)
    return 2.0 * r**2 * np.arccos(s_clip / (2.0 * r)) - 0.5 * s_clip * np.sqrt(4.0 * r**2 - s_clip**2)


def _mtf_closed_form(u: np.ndarray, v: np.ndarray, holes: np.ndarray, diameter: float = 0.82) -> np.ndarray:
    """Closed-form pupil autocorrelation of circular mask holes, normalized to 1 at (0, 0)."""
    total = np.zeros_like(u, dtype=float)
    for xi in holes:
        for xj in holes:
            total = total + _circle_overlap(np.hypot(u - (xj[0] - xi[0]), v - (xj[1] - xi[1])), diameter)
    return total / (len(holes) * float(_circle_overlap(np.array(0.0), diameter)))


NIRISS_HOLES = np.array(
    [
        [0.0, -2.64],
        [-2.28631, 0.0],
        [2.28631, -1.32],
        [-2.28631, 1.32],
        [-1.14315, 1.98],
        [2.28631, 1.32],
        [1.14315, 1.98],
    ]
)


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


def _jacobi_anger_disk_ref(
    uu: np.ndarray,
    vv: np.ndarray,
    r: np.ndarray,
    w: np.ndarray,
    inc: float,
    pa: float,
    blur_fwhm: float = 0.0,
    c: np.ndarray | None = None,
    s: np.ndarray | None = None,
    height: np.ndarray | None = None,
    stretch: float | None = None,
) -> np.ndarray:
    """Independent SciPy Jacobi-Anger Bessel sum for multipole_disk_cvis."""
    pa_rad = np.deg2rad(pa)
    inc_rad = np.deg2rad(inc)
    q_maj = (uu * np.sin(pa_rad) + vv * np.cos(pa_rad)) * sky.MAS
    q_min_sky = (uu * np.cos(pa_rad) - vv * np.sin(pa_rad)) * sky.MAS
    st = np.cos(inc_rad) if stretch is None else float(stretch)
    q_min = q_min_sky * st
    q = np.hypot(q_maj, q_min)
    phi = np.arctan2(q_min, q_maj)

    h_phase = (
        np.ones((uu.size, r.size), dtype=complex)
        if height is None
        else np.exp(-2j * np.pi * q_min_sky[:, None] * (height[None, :] * np.sin(inc_rad)))
    )
    bessel_sum = np.sum(w[None, :] * h_phase * jv(0, 2.0 * np.pi * q[:, None] * r[None, :]), axis=1)
    if c is not None and s is not None:
        for m in range(1, c.shape[0] + 1):
            jm = jv(m, 2.0 * np.pi * q[:, None] * r[None, :])
            ang = c[m - 1][None, :] * np.cos(m * phi[:, None]) + s[m - 1][None, :] * np.sin(m * phi[:, None])
            bessel_sum = bessel_sum + ((-1j) ** m) * np.sum(w[None, :] * h_phase * ang * jm, axis=1)

    sigma_blur = float(blur_fwhm) / sky.FWHM_PER_SIGMA
    env = np.exp(-2.0 * np.pi**2 * sigma_blur**2 * q**2)
    return bessel_sum * env


@pytest.mark.validates(
    "virgil.models.multipole_disk_cvis",
    roots=["mathematics"],
    headline="multipole_disk_cvis_max_err",
)
def test_multipole_disk_cvis_jacobi_anger_height_blur_and_stretch():
    """multipole_disk_cvis matches the exact SciPy Jacobi-Anger Bessel series
    with uv deprojection, disk-plane Gaussian blur, conical ring height phase
    shift, and negative stretch minor-axis reflection."""
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
    v_ref = _jacobi_anger_disk_ref(uu, vv, r, w, inc=inc, pa=pa, blur_fwhm=blur_fwhm, c=c, s=s, height=height)
    err_main = float(np.max(np.abs(v_vv - v_ref)))

    # Negative stretch compared directly against closed-form SciPy Bessel sum at q_min = q_min_sky * stretch_neg:
    stretch_neg = -float(np.cos(np.deg2rad(inc)))
    v_neg = np.asarray(vm.multipole_disk_cvis(uu, vv, r, w_raw, inc=inc, pa=pa, c=c, s=s, stretch=stretch_neg))
    v_neg_ref = _jacobi_anger_disk_ref(uu, vv, r, w, inc=inc, pa=pa, c=c, s=s, stretch=stretch_neg)
    err_stretch = float(np.max(np.abs(v_neg - v_neg_ref)))

    # Normalization V(0, 0) == 1:
    v0 = complex(np.asarray(vm.multipole_disk_cvis(0.0, 0.0, r, w_raw, inc=inc, pa=pa, c=c, s=s, height=height)))
    err_v0 = abs(v0 - 1.0)

    record("multipole_disk_cvis_max_err", max(err_main, err_stretch, err_v0))
    assert err_main < 1e-13
    assert err_stretch < 1e-13
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

    z_prof = rng.normal(size=24)
    z_alpha = rng.normal(size=8)
    disk_chrom = vm.MultipoleDisk(
        profile=vm.RadialGP(1.0, 12.0, sigma=0.8, ell=3.0, z=z_prof),
        spectral_index=vm.RadialGPFunction(sigma=0.6, ell=4.0, mu=-0.5, z=z_alpha),
        inc=35.0,
        pa=25.0,
        flux=0.5,
        wavel0=2.2e-6,
        n_r=24,
    )
    sys_chrom = vm.System(star=vm.PointSource(flux=1.0), disk=disk_chrom)

    x_gl, w_gl = roots_legendre(24)
    r_gl = 0.5 * (12.0 - 1.0) * x_gl + 0.5 * (12.0 + 1.0)
    dw_gl = 0.5 * (12.0 - 1.0) * w_gl
    w0_gl = r_gl * dw_gl * np.exp(0.8 * (cholesky(_matern_corr(r_gl, ell=3.0, nu=1.5), lower=True) @ z_prof))
    alpha_nodes = np.linspace(1.0, 12.0, 8)
    alpha_vals = -0.5 + 0.6 * (cholesky(_matern_corr(alpha_nodes, ell=4.0, nu=1.5), lower=True) @ z_alpha)
    alpha_gl = np.interp(r_gl, alpha_nodes, alpha_vals)

    q_u = np.array([1.2e7, -2.1e7, 2.8e7])
    q_v = np.array([-0.9e7, 1.7e7, 1.4e7])
    chrom_err = 0.0
    disk_by_wavel = {}
    for wavel in (1.9e-6, 2.5e-6):
        u_q, v_q = q_u * wavel, q_v * wavel
        w_lam = w0_gl * (wavel / 2.2e-6) ** alpha_gl
        v_disk_ref = _jacobi_anger_disk_ref(q_u, q_v, r_gl, w_lam / np.sum(w_lam), inc=35.0, pa=25.0)
        f_disk_lam = 0.5 * float(np.sum(w_lam) / np.sum(w0_gl))
        v_sys_ref = (1.0 + f_disk_lam * v_disk_ref) / (1.0 + f_disk_lam)
        v_disk_vv = np.asarray(disk_chrom.model(u_q, v_q, np.full(3, wavel)))
        v_sys_vv = np.asarray(sys_chrom.model(u_q, v_q, np.full(3, wavel)))
        disk_by_wavel[wavel] = v_disk_vv
        chrom_err = max(
            chrom_err,
            float(np.max(np.abs(v_disk_vv - v_disk_ref))),
            float(np.max(np.abs(v_sys_vv - v_sys_ref))),
        )
    assert float(np.max(np.abs(disk_by_wavel[1.9e-6] - disk_by_wavel[2.5e-6]))) > 1e-2

    # Control: when z=0, alpha(r) = mu is spatially uniform and cancels on normalizing disk.model at fixed (q_u, q_v):
    disk_flat = vm.MultipoleDisk(
        profile=vm.RadialGP(1.0, 12.0, sigma=0.8, ell=3.0, z=z_prof),
        spectral_index=vm.RadialGPFunction(sigma=0.6, ell=4.0, mu=-0.5, z=np.zeros(8)),
        inc=35.0,
        pa=25.0,
        flux=0.5,
        wavel0=2.2e-6,
        n_r=24,
    )
    flat_diff = float(
        np.max(
            np.abs(
                np.asarray(disk_flat.model(q_u * 1.9e-6, q_v * 1.9e-6, np.full(3, 1.9e-6)))
                - np.asarray(disk_flat.model(q_u * 2.5e-6, q_v * 2.5e-6, np.full(3, 2.5e-6)))
            )
        )
    )
    assert flat_diff < 5e-14

    disk = vm.MultipoleDisk(
        profile=vm.RadialGP(1.0, 12.0, sigma=0.8, ell=3.0),
        harmonics=vm.Harmonics(degree=2, prior=vm.MultipoleKernel(ell_r=3.0, ell_theta=2.5, sigma=0.25), form="linear"),
        spectral_index=vm.RadialGPFunction(sigma=0.4, ell=4.0, mu=-0.5, n=8),
        wavel0=2.2e-6,
        n_r=24,
    )
    scene = vm.System(star=vm.PointSource(flux=1.0), disk=disk)

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

    lp_diff = float(priors_all["disk.profile.sigma"].log_prob(0.2) - priors_all["disk.profile.sigma"].log_prob(0.8))
    lp_err = abs(lp_diff - (-np.log(0.2 / 0.8)))

    priors_latents_only = vm.gp_priors(scene)
    assert set(priors_latents_only.keys()) == {"disk.profile.z", "disk.harmonics.z", "disk.spectral_index.z"}

    record("radial_gp_function_max_err", max(max_gp_err, chrom_err, lp_err))
    assert chrom_err < 5e-14
    assert max_gp_err < 2e-12  # nu=inf RBF Cholesky has condition ~1/jitter = 1e6
    assert lp_err < 1e-14


# ---------------------------------------------------------------------------
# 2. polar_interpolate, multipole_field_priors, MultipoleField origin taper
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.fields.polar_interpolate",
    roots=["mathematics"],
    headline="polar_interp_node_err",
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

    at_nodes = np.asarray(vfields.polar_interpolate(vals, r_min, spacing, rr_g, tt_g, fill=fill))
    at_wrapped = np.asarray(vfields.polar_interpolate(vals, r_min, spacing, rr_g, tt_g + 2.0 * np.pi, fill=fill))
    err_nodes = float(np.max(np.abs(at_nodes - vals)))
    err_wrap = float(np.max(np.abs(at_nodes - at_wrapped)))

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
    record("polar_interp_interior_node_err", max(err_nodes, err_wrap))
    record("polar_interp_refinement_ratio", ratio)
    assert err_nodes < 1e-14
    assert err_wrap < 1e-14
    assert err_edges < 2e-8  # boundary epsilon clamp in polar_interpolate
    assert 3.7 < ratio < 4.3


@pytest.mark.validates(
    "virgil.fields.multipole_field_priors",
    roots=["mathematics"],
    headline="multipole_prior_logprob_err",
)
def test_multipole_field_priors_bounds_and_log_uniform_scaling():
    """multipole_field_priors builds bounded LogUniform scale priors on
    (sigma, length_r, length_theta) with exact d(log p)/d(log x) == -1."""
    pri = vfields.multipole_field_priors(
        path="disk_img.log_brightness", sigma=(0.1, 3.0), length_r=(1.0, 25.0), length_theta=(0.2, 5.0)
    )
    assert set(pri.keys()) == {
        "disk_img.log_brightness.sigma",
        "disk_img.log_brightness.length_r",
        "disk_img.log_brightness.length_theta",
    }
    errs = []
    for key, x1, x2 in [
        ("disk_img.log_brightness.sigma", 0.2, 1.6),
        ("disk_img.log_brightness.length_r", 2.0, 8.0),
        ("disk_img.log_brightness.length_theta", 0.4, 2.0),
    ]:
        lp_gap = float(pri[key].log_prob(x1) - pri[key].log_prob(x2))
        errs.append(abs(lp_gap - (-np.log(x1 / x2))))
    max_err = max(errs)
    record("multipole_prior_logprob_err", max_err)
    assert max_err < 1e-14


@pytest.mark.validates(
    "virgil.fields.MultipoleField",
    roots=["mathematics", "statistics"],
    headline="origin_non_m0_std",
)
def test_multipole_field_042_origin_taper_analytical_variance_and_nr_invariance():
    """In 0.4.2, MultipoleField(r_min_mas=0) multiplies azimuthal order m by
    tanh(r / length_r)^|m| using physical length_r (F25): only m=0 is free at
    r=0, the azimuthal m != 0 variance matches sigma^2 * sum_{m!=0} S(m) tanh(r/ell_r)^{2|m|} / sum S(m)
    both analytically (across n_r = 16 and n_r = 31) and over 512 Monte Carlo
    draws, and an annulus (r_min_mas > 0) stays stationary at its inner edge."""
    length_r, length_theta, nu_th, sigma = 6.0, 0.8, 1.5, 1.2
    n_th = 32
    r_max = 15.0
    npix, pix = 65, 0.5
    c_idx = (npix - 1) // 2

    m_orders = np.abs(np.fft.fftfreq(n_th, d=1.0 / n_th))
    s_m = (1.0 + (m_orders * length_theta) ** 2 / (2.0 * nu_th)) ** (-nu_th - 0.5)

    def _exact_ring_non0_var(n_r: int, r_min: float) -> np.ndarray:
        z_zero = jnp.zeros((n_r, n_th))
        jac = np.asarray(
            jax.jacfwd(
                lambda z: vfields.MultipoleField(
                    z,
                    image_shape=(npix, npix),
                    r_max_mas=r_max,
                    r_min_mas=r_min,
                    sigma=sigma,
                    length_r=length_r,
                    length_theta=length_theta,
                    nu_theta=nu_th,
                ).polar_values()
            )(z_zero)
        ).reshape(n_r, n_th, -1)
        jac_non0 = jac - jac.mean(axis=1, keepdims=True)
        return np.mean(np.sum(jac_non0**2, axis=-1), axis=1)

    # 1. Exact analytical variance profile & radial-grid resolution invariance (n_r=16 vs n_r=31):
    var_16 = _exact_ring_non0_var(16, 0.0)
    var_31 = _exact_ring_non0_var(31, 0.0)
    radii_16 = np.linspace(0.0, r_max, 16)
    taper_16 = np.tanh(radii_16[:, None] / length_r) ** (2.0 * m_orders[None, :])
    var_formula_16 = sigma**2 * np.sum(s_m[None, 1:] * taper_16[:, 1:], axis=1) / np.sum(s_m)

    err_formula_16 = float(np.max(np.abs(var_16 - var_formula_16)))
    # Shared integer radii r = 0, 1, ..., 15 mas sit at indices 0..15 of n_r=16 and 0, 2, ..., 30 of n_r=31:
    err_nr_invariance = float(np.max(np.abs(var_16 - var_31[::2])))
    assert float(var_16[0]) < 1e-30
    assert float(var_31[0]) < 1e-30

    # Annulus (r_min_mas = 2.0) keeps untapered stationary m != 0 variance right to its inner edge:
    var_ann = _exact_ring_non0_var(14, 2.0)
    var_untapered = float(sigma**2 * np.sum(s_m[1:]) / np.sum(s_m))
    err_annulus = float(np.max(np.abs(var_ann - var_untapered)))

    # 2. Monte Carlo verification over 512 standard-normal draws Z ~ N(0, I):
    rng = np.random.default_rng(66)
    n_draws = 512
    latents = rng.normal(size=(n_draws, 16, n_th))
    pv_draws = np.asarray(
        jax.vmap(
            lambda z: vfields.MultipoleField(
                z,
                image_shape=(npix, npix),
                r_max_mas=r_max,
                r_min_mas=0.0,
                sigma=sigma,
                length_r=length_r,
                length_theta=length_theta,
                nu_theta=nu_th,
            ).polar_values()
        )(jnp.asarray(latents))
    )
    pv_non0 = pv_draws - pv_draws.mean(axis=-1, keepdims=True)
    ring_sample_vars = np.mean(pv_non0**2, axis=-1)  # (n_draws, 16)
    mc_mean = ring_sample_vars.mean(axis=0)
    mc_se = ring_sample_vars.std(axis=0, ddof=1) / np.sqrt(n_draws)
    # Check r > 0 rings lie within 3.5 standard errors of the analytical curve:
    max_zscore = float(np.max(np.abs(mc_mean[1:] - var_formula_16[1:]) / mc_se[1:]))

    # 3. Exact single-valuedness at the centre pixel (r = 0) of Image(MultipoleField) under 90 deg rotation:
    z0 = latents[0]
    z_rot90 = np.roll(z0, n_th // 4, axis=1)
    b_orig = np.asarray(
        vm.Image(
            vfields.MultipoleField(z0, image_shape=(npix, npix), r_max_mas=r_max, r_min_mas=0.0, length_r=length_r),
            pixel_scale_mas=pix,
        ).brightness
    )
    b_rot90 = np.asarray(
        vm.Image(
            vfields.MultipoleField(z_rot90, image_shape=(npix, npix), r_max_mas=r_max, r_min_mas=0.0, length_r=length_r),
            pixel_scale_mas=pix,
        ).brightness
    )
    origin_m1_diff = float(abs(b_orig[c_idx, c_idx] - b_rot90[c_idx, c_idx]))
    outer_rot_diff = float(np.max(np.abs(b_orig - b_rot90)))
    assert outer_rot_diff > 5e-5

    record("origin_non_m0_std", origin_m1_diff)
    record("multipole_taper_formula_err", err_formula_16)
    record("multipole_nr_invariance_err", err_nr_invariance)
    record("multipole_mc_max_zscore", max_zscore)
    assert origin_m1_diff < 1e-15
    assert err_formula_16 < 5e-5
    assert err_nr_invariance < 5e-5
    assert err_annulus < 5e-5
    assert max_zscore < 3.5


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
    headline="ami_grid_precision_rel_err",
)
def test_ami_grid_record_and_amigo_mixed_disco_validation_and_projection():
    """ami_grid_record builds the rotated half-plane square lattice where the
    7-hole pupil autocorrelation is positive, projects out global log-amplitude
    flux (A_logamp @ 1 == 0) and phase centroid shifts (A_phase @ [u, v] == 0)
    with exact Fisher precision at keep=1.0, scales disco_sigma linearly with
    sigma, and truncates rank according to keep; AMIGO validators enforce schema
    invariants and match A_logamp log|V| + A_phase arg V at negated (-u, -v)."""
    rec0 = vcov.ami_grid_record(
        wavelength_m=4.3e-6, pitch_m=0.8, rotation_deg=0.0, sigma=1e-4, keep=0.99, holes=NIRISS_HOLES, diameter=0.82
    )
    rec = vcov.ami_grid_record(
        wavelength_m=4.3e-6, pitch_m=0.8, rotation_deg=25.0, sigma=1e-4, keep=0.99, holes=NIRISS_HOLES, diameter=0.82
    )
    rec_2s = vcov.ami_grid_record(
        wavelength_m=4.3e-6, pitch_m=0.8, rotation_deg=25.0, sigma=2e-4, keep=0.99, holes=NIRISS_HOLES, diameter=0.82
    )
    rec_k100 = vcov.ami_grid_record(
        wavelength_m=4.3e-6, pitch_m=0.8, rotation_deg=25.0, sigma=1e-4, keep=1.0, holes=NIRISS_HOLES, diameter=0.82
    )
    rec_k80 = vcov.ami_grid_record(
        wavelength_m=4.3e-6, pitch_m=0.8, rotation_deg=25.0, sigma=1e-4, keep=0.80, holes=NIRISS_HOLES, diameter=0.82
    )

    # 1. Exact half-plane pitch_m lattice where closed-form circular-hole MTF > 0, plus rotation_deg:
    for pitch in (0.8, 0.6):
        r_p = vcov.ami_grid_record(
            wavelength_m=4.3e-6, pitch_m=pitch, rotation_deg=0.0, holes=NIRISS_HOLES, diameter=0.82
        )
        u_phys, v_phys = -r_p["u"], -r_p["v"]
        n_max = int(np.ceil(7.0 / pitch))
        ix, iy = np.meshgrid(np.arange(-n_max, n_max + 1), np.arange(0, n_max + 1))
        hp = (iy > 0) | ((iy == 0) & (ix > 0))
        ug, vg = ix[hp] * pitch, iy[hp] * pitch
        pos_mtf = _mtf_closed_form(ug, vg, NIRISS_HOLES, 0.82) > 0.0
        pts_got = {(round(float(a / pitch), 6), round(float(b / pitch), 6)) for a, b in zip(u_phys, v_phys)}
        pts_ref = {(round(float(a / pitch), 6), round(float(b / pitch), 6)) for a, b in zip(ug[pos_mtf], vg[pos_mtf])}
        assert pts_got == pts_ref

    rot_rad = np.deg2rad(25.0)
    u_rot = rec0["u"] * np.cos(rot_rad) + rec0["v"] * np.sin(rot_rad)
    v_rot = -rec0["u"] * np.sin(rot_rad) + rec0["v"] * np.cos(rot_rad)
    err_rot_uv = float(np.max(np.hypot(rec["u"] - u_rot, rec["v"] - v_rot)))
    assert float(np.max(np.hypot(rec["u"] - rec0["u"], rec["v"] - rec0["v"]))) > 0.5

    # 2. Direct operator nullspace checks (global log-amplitude piston 1 and normalized phase tip/tilt [u, v]):
    a_la = rec["disco_logamp_model_operator"]
    a_ph = rec["disco_phase_model_operator"]
    uv_cols = np.column_stack([rec["u"], rec["v"]])
    null_logamp_ones = float(np.max(np.abs(a_la @ np.ones_like(rec["u"]))))
    null_phase_uv = float(np.max(np.abs(a_ph @ uv_cols)) / np.max(np.abs(uv_cols)))

    # 3. Exact nuisance-projected Fisher precision reconstruction at keep=1.0, sigma scaling, and keep truncation:
    mtf_cells = _mtf_closed_form(-rec0["u"], -rec0["v"], NIRISS_HOLES, 0.82)
    w_cell = (mtf_cells / 1e-4) ** 2
    p_la_ref = np.diag(w_cell) - (w_cell[:, None] @ w_cell[None, :]) / np.sum(w_cell)
    xy = np.column_stack([rec["u"], rec["v"]])
    w_xy = w_cell[:, None] * xy
    p_ph_ref = np.diag(w_cell) - w_xy @ np.linalg.solve(xy.T @ w_xy, w_xy.T)

    w_k100 = 1.0 / (rec_k100["disco_sigma"] ** 2)
    p_la_100 = rec_k100["disco_logamp_model_operator"].T @ (w_k100[:, None] * rec_k100["disco_logamp_model_operator"])
    p_ph_100 = rec_k100["disco_phase_model_operator"].T @ (w_k100[:, None] * rec_k100["disco_phase_model_operator"])
    err_prec_la = float(np.linalg.norm(p_la_100 - p_la_ref) / np.linalg.norm(p_la_ref))
    err_prec_ph = float(np.linalg.norm(p_ph_100 - p_ph_ref) / np.linalg.norm(p_ph_ref))
    err_prec = max(err_prec_la, err_prec_ph)

    err_sigma_scale = float(np.max(np.abs(rec_2s["disco_sigma"] / rec["disco_sigma"] - 2.0)))
    assert rec_k80["disco_sigma"].size < rec["disco_sigma"].size < rec_k100["disco_sigma"].size
    w_k99 = 1.0 / (rec["disco_sigma"] ** 2)
    kept_frac_99 = float(
        (
            np.trace(a_la.T @ (w_k99[:, None] * a_la))
            + np.trace(a_ph.T @ (w_k99[:, None] * a_ph))
        )
        / (np.trace(p_la_ref) + np.trace(p_ph_ref))
    )
    assert 0.99 <= kept_frac_99 < 1.0

    # 4. AMIGO validators & negated (-u, -v) projection through OIData:
    assert vamigo.is_mixed_disco_record(rec) is True
    assert vamigo.is_mixed_disco_record({"u": np.zeros(4), "v": np.zeros(4)}) is False
    vamigo.validate_mixed_disco_record(rec)
    fields = vamigo.mixed_disco_fields(rec)
    assert fields["vis_encoding"] == "mixed_log_complex"

    oid = voidata.OIData(rec)
    binary = vm.BinaryModelCartesian(dra=-45.0, ddec=160.0, flux=2e-3)
    oid_sim = oid.with_model(binary)

    binary_sky = sky.mix([sky.point(0.0, 0.0), sky.point(-45.0, 160.0)], [1.0, 2e-3])
    cvis = sky.visibility(binary_sky, -rec["u"], -rec["v"], np.full_like(rec["u"], 4.3e-6))
    y_manual = a_la @ np.log(np.abs(cvis)) + a_ph @ np.angle(cvis)
    cvis_unnegated = sky.visibility(binary_sky, rec["u"], rec["v"], np.full_like(rec["u"], 4.3e-6))
    y_wrong_sign = a_la @ np.log(np.abs(cvis_unnegated)) + a_ph @ np.angle(cvis_unnegated)
    y_oid = np.asarray(oid_sim.model(binary))
    err_proj = float(np.max(np.abs(y_oid - y_manual)))
    assert float(np.max(np.abs(y_oid - y_wrong_sign))) > 5e-3

    # Sub-mas centroid shift and global constant log|V| shift both vanish:
    null_leak = float(np.max(np.abs(np.asarray(oid.model(vm.PointSource(dra=1e-3, ddec=-1e-3))))))

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

    record("ami_grid_precision_rel_err", err_prec)
    record("mixed_disco_projection_err", err_proj)
    record("ami_grid_nullspace_max_err", max(null_logamp_ones, null_phase_uv))
    record("ami_grid_centroid_null_leak", null_leak)
    assert err_rot_uv < 5e-14
    assert null_logamp_ones < 5e-14
    assert null_phase_uv < 5e-14
    assert err_prec < 2e-14
    assert err_sigma_scale < 1e-14
    assert err_proj < 5e-14
    assert null_leak < 1e-9


# ---------------------------------------------------------------------------
# 5. oifits.build_hdulist, oidata.{closure_phases, cp_indices}, observables.in_ranges
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.oidata.cp_indices",
    "virgil.oidata.closure_phases",
    "virgil.observables.in_ranges",
    roots=["mathematics"],
    headline="closure_phases_wrap_max_err",
)
def test_cp_indices_closure_phases_and_in_ranges():
    """cp_indices maps oriented triangle legs (rejecting reversed legs);
    closure_phases wraps bispectrum phases into [-pi, pi); and in_ranges selects
    inclusive wavelength windows."""
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

    with pytest.raises(ValueError):
        voidata.cp_indices(np.array([[1, 2], [2, 3], [3, 1]]), np.array([[1, 2, 3]]))

    phases = np.array([2.4, 2.5, -1.1, -2.2, 0.3])
    cvis = np.array([0.8, 0.7, 0.9, 0.6, 0.5]) * np.exp(1j * phases)
    cp = np.asarray(voidata.closure_phases(cvis, i1, i2, i3))
    cp_ref = np.angle(cvis[i1] * cvis[i2] * np.conj(cvis[i3]))
    cp_ref = np.where(cp_ref >= np.pi, cp_ref - 2.0 * np.pi, cp_ref)
    err_cp = float(np.max(np.abs(cp - cp_ref)))
    assert np.all((cp >= -np.pi) & (cp < np.pi))

    record("closure_phases_wrap_max_err", err_cp)
    assert err_cp < 1e-14


@pytest.mark.validates(
    "virgil.oifits.build_hdulist",
    roots=["standards"],
    headline="oifits_hdulist_crosscheck_err",
)
def test_build_hdulist_oifits2_headers_units_and_crosscheck_load(tmp_path):
    """build_hdulist produces OIFITS2 extensions with standard OI_REVN=2 headers
    and column units, supports primary-header customization before writing, and
    round-trips cleanly through astropy.io.fits and crosscheck.chi2.load."""
    from crosscheck import chi2 as xchi2

    wls = np.array([2.0e-6, 2.15e-6, 2.3e-6, 2.45e-6], dtype=np.float32)
    tables = {
        "OI_WAVELENGTH": {"EFF_WAVE": wls, "EFF_BAND": np.full(4, 0.05e-6, dtype=np.float32)},
        "OI_VIS2": {
            "UCOORD": np.array([35.0, -20.0, 15.0]),
            "VCOORD": np.array([12.0, 28.0, 40.0]),
            "VIS2DATA": np.array(
                [[0.88, 0.84, 0.79, 0.75], [0.91, 0.89, 0.86, 0.83], [0.76, 0.72, 0.68, 0.64]]
            ),
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
    hdul[0].header["HIERARCH VIRGIL TESTTAG"] = "BATCH5"
    out_fits = tmp_path / "custom_hdulist.fits"
    hdul.writeto(out_fits)

    # 1. Inspect OIFITS2 standard headers and column units directly via astropy.io.fits:
    with fits.open(out_fits) as h:
        assert h[0].header["HIERARCH VIRGIL TESTTAG"] == "BATCH5"
        ext_names = {ext.name for ext in h[1:]}
        assert {"OI_TARGET", "OI_WAVELENGTH", "OI_VIS2", "OI_T3"}.issubset(ext_names)
        for ext_name in ("OI_TARGET", "OI_WAVELENGTH", "OI_VIS2", "OI_T3"):
            assert h[ext_name].header["OI_REVN"] == 2
        wl_cols = {c.name: c.unit for c in h["OI_WAVELENGTH"].columns}
        t3_cols = {c.name: c.unit for c in h["OI_T3"].columns}
        v2_cols = {c.name: c.unit for c in h["OI_VIS2"].columns}
        assert wl_cols["EFF_WAVE"] == "m"
        assert wl_cols["EFF_BAND"] == "m"
        assert v2_cols["UCOORD"] == "m" and v2_cols["VCOORD"] == "m"
        assert t3_cols["T3PHI"] in ("deg", "DEG", "degrees")
        assert t3_cols["T3PHIERR"] in ("deg", "DEG", "degrees")

    # 2. Independent OIFITS read-back via crosscheck.chi2.load (not virgil's reader):
    loaded = xchi2.load(out_fits)
    err_wl = float(np.max(np.abs(loaded["wl"][0] - wls.astype(float))))
    err_uv = max(
        float(np.max(np.abs(loaded["u"][:, 0] - tables["OI_VIS2"]["UCOORD"]))),
        float(np.max(np.abs(loaded["v"][:, 0] - tables["OI_VIS2"]["VCOORD"]))),
    )
    err_v2 = max(
        float(np.max(np.abs(loaded["v2"] - tables["OI_VIS2"]["VIS2DATA"]))),
        float(np.max(np.abs(loaded["dv2"] - tables["OI_VIS2"]["VIS2ERR"]))),
    )
    err_t3 = max(
        float(np.max(np.abs(np.rad2deg(loaded["cp"]) - tables["OI_T3"]["T3PHI"]))),
        float(np.max(np.abs(np.rad2deg(loaded["dcp"]) - tables["OI_T3"]["T3PHIERR"]))),
    )
    err_all = max(err_wl, err_uv, err_v2, err_t3)

    record("oifits_hdulist_crosscheck_err", err_all)
    assert err_all < 1e-7


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
    log-likelihood across V^2, VISAMP, and T3AMP.

    Ledger F26 is open: ``ClosureOffsets(scale_penalty=True)`` divides the wrap
    penalty by the unprojected ``hypot(sigma, sigma_off)`` from ``diag(M M^T)``, so
    on a 4-telescope frame an offset along the left-null vector of T (n^T T = 0)
    changes ``model_loglike`` when Delta != 0. This test does not cover that case:
    it uses a 3-telescope frame (one closure triangle, so T has full column rank
    and there is no left-null mode) and a single per-triangle offset, and its
    ``scale_penalty=True`` reference ``(1 - cos Delta) / hypot(sigma, sigma_off)``
    is the current unprojected formula, so it pins today's behaviour rather than
    confirming it is correct. The F26 null-mode check is
    tests/test_gains_algebra.py::test_f26_closure_offsets_scale_penalty_projects_onto_span_of_T."""
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

    raw_ext = voifits.read_oifits(p, extras=("visamp", "t3amp"))
    raw_ext["phi"] = np.zeros_like(raw_ext["phi"])
    oid_ext = voidata.OIData(raw_ext)
    tau_tel = 0.04
    g_inc = oid_ext.with_gains(telescope=tau_tel, include_extras=True)
    g_exc = oid_ext.with_gains(telescope=tau_tel, include_extras=False)
    assert g_inc.gains.n_vis == 14  # 6 V^2 + 6 VISAMP + 2 T3AMP
    assert g_exc.gains.n_vis == 6   # 6 V^2 only

    scene = vm.BinaryModelCartesian(dra=3.5, ddec=-2.5, flux=0.2)
    scene_sky = sky.mix([sky.point(0.0, 0.0), sky.point(3.5, -2.5)], [1.0, 0.2])
    cvis = sky.visibility(scene_sky, np.asarray(oid_ext.u), np.asarray(oid_ext.v), np.asarray(oid_ext.wavel))
    v_abs = np.abs(cvis)
    v2_mod = v_abs**2
    t3_mod = v_abs[oid_ext.i_cps1] * v_abs[oid_ext.i_cps2] * v_abs[oid_ext.i_cps3]

    sta = np.asarray(raw_ext["stations"]) - 1
    b_vis = np.zeros((6, 3))
    for row_i, (sa, sb) in enumerate(sta):
        b_vis[row_i, sa] += 1.0
        b_vis[row_i, sb] += 1.0
    b_t3 = b_vis[oid_ext.i_cps1] + b_vis[oid_ext.i_cps2] + b_vis[oid_ext.i_cps3]

    ll_no_gains = float(vlike.model_loglike(scene, oid_ext))
    ll_inc = float(vlike.model_loglike(scene, g_inc))
    ll_exc = float(vlike.model_loglike(scene, g_exc))

    y_all = np.concatenate([np.asarray(oid_ext.vis), np.full(6, 0.94), np.full(2, 0.86)])
    m_all = np.concatenate([v2_mod, v_abs, t3_mod])
    s_all = np.concatenate([np.asarray(oid_ext.d_vis), np.full(6, 0.015), np.full(2, 0.025)])
    res_all = y_all - m_all

    j_inc = np.vstack([2.0 * v2_mod[:, None] * b_vis, v_abs[:, None] * b_vis, t3_mod[:, None] * b_t3]) * tau_tel
    c_inc = np.diag(s_all**2) + j_inc @ j_inc.T
    cf_inc = cho_factor(c_inc)
    logdet_inc = 2.0 * np.sum(np.log(np.diag(cf_inc[0])))
    dll_inc_ref = -0.5 * (
        (logdet_inc + res_all @ cho_solve(cf_inc, res_all))
        - (np.sum(np.log(s_all**2)) + np.sum((res_all / s_all) ** 2))
    )

    j_exc = (2.0 * v2_mod[:, None] * b_vis) * tau_tel
    c_exc = np.diag(s_all[:6] ** 2) + j_exc @ j_exc.T
    cf_exc = cho_factor(c_exc)
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
    sigma / (2 V^2), evaluates OIData.model and whitened_residuals on both
    unprojected and vis_mat-projected log|V|, and rejects unknown modes) and
    vis_mode='v2' (exact variance sigma * sqrt(4 V^2 + 2 sigma^2), forward V^2
    model, and whitened residuals)."""
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
    kept_idx = np.array([0, 3])
    scene_p = vm.BinaryModelCartesian(dra=12.0, ddec=-8.0, flux=0.3)
    cvis_all = sky.mix([sky.point(0.0, 0.0), sky.point(12.0, -8.0)], [1.0, 0.3])
    cvis_ref = sky.visibility(cvis_all, rec_v2["u"], rec_v2["v"], rec_v2["wavel"])

    # Unprojected logamp forward model and whitened_residuals:
    log_pred = np.asarray(oid_log.model(scene_p))
    log_mod_ref = np.log(np.abs(cvis_ref[kept_idx]))
    log_dat_ref = 0.5 * np.log(kept_v2)
    log_sig_ref = kept_dv2 / (2.0 * kept_v2)
    r_log = np.asarray(vlike.whitened_residuals(scene_p, oid_log))
    err_logamp_enc = max(
        float(np.max(np.abs(np.asarray(oid_log.vis) - log_dat_ref))),
        float(np.max(np.abs(np.asarray(oid_log.d_vis) - log_sig_ref))),
        float(np.max(np.abs(log_pred - log_mod_ref))),
    )
    err_logamp_white = float(np.max(np.abs(r_log - (log_mod_ref - log_dat_ref) / log_sig_ref)))

    # Projected vis_mat + vis_mode="logamp" projects log|V_model| linearly (both OIData.model and whitened_residuals):
    vis_mat = np.array([[0.6, 0.4, 0.0, 0.0], [0.0, -0.3, 0.5, 0.8]])
    proj_dat = np.array([0.05, -0.02])
    proj_sig = np.array([0.01, 0.01])
    oid_proj = voidata.OIData(
        {**rec_v2, "vis": proj_dat, "d_vis": proj_sig, "vis_mat": vis_mat, "vis_mode": "logamp"}
    )
    proj_mod_ref = vis_mat @ np.log(np.abs(cvis_ref))
    err_proj_logamp = float(np.max(np.abs(np.asarray(oid_proj.model(scene_p)) - proj_mod_ref)))
    r_proj = np.asarray(vlike.whitened_residuals(scene_p, oid_proj))
    err_proj_white = float(np.max(np.abs(r_proj - (proj_mod_ref - proj_dat) / proj_sig)))
    assert err_proj_logamp < 5e-15
    assert err_proj_white < 1e-13

    with pytest.raises(ValueError, match="Unsupported vis_mode"):
        voidata.OIData({**rec_v2, "vis": np.full(4, 0.8), "vis_mode": "bogus"})

    # Amplitude input (v2_flag=False) converted to V^2 via vis_mode="v2", plus forward model & whitened_residuals:
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
    v2_mod_ref = np.abs(cvis_ref[:3]) ** 2
    v2_pred = np.asarray(oid_v2.model(scene_p))
    r_v2 = np.asarray(vlike.whitened_residuals(scene_p, oid_v2))
    err_v2_enc = max(
        float(np.max(np.abs(np.asarray(oid_v2.vis) - v2_ref))),
        float(np.max(np.abs(np.asarray(oid_v2.d_vis) - dv2_ref))),
        float(np.max(np.abs(v2_pred - v2_mod_ref))),
    )
    err_v2_white = float(np.max(np.abs(r_v2 - (v2_mod_ref - v2_ref) / dv2_ref)))

    record("vis_mode_encoding_err", max(err_logamp_enc, err_proj_logamp, err_v2_enc))
    record("vis_mode_whitened_err", max(err_logamp_white, err_proj_white, err_v2_white))
    assert err_logamp_enc < 5e-15
    assert err_v2_enc < 5e-15
    assert err_logamp_white < 1e-13
    assert err_v2_white < 1e-13
