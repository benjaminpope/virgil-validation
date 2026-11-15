"""Independent validation of MultipoleDisk's companion profiles, harmonic conversions,
Gaussian-process kernels, and MultipoleField.

Every reference value here is computed from first principles using SciPy Bessel
functions (``scipy.special.j0``, ``j1``, ``iv``, ``kv``, ``gamma``), adaptive
quadrature (``crosscheck.radial``), or brute-force 2D/3D sky point-cloud Fourier
transforms (``crosscheck.polar_images`` and ``crosscheck.sky``). No code is
shared with virgil.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special

vm = pytest.importorskip("virgil.models")
vh = pytest.importorskip("virgil.harmonics")
vf = pytest.importorskip("virgil.fields")
if not hasattr(vm, "MultipoleDisk"):
    pytest.skip("could not import 'virgil.models.MultipoleDisk'", allow_module_level=True)

from crosscheck import polar_images as P  # noqa: E402
from crosscheck import radial as R  # noqa: E402
from crosscheck import sky  # noqa: E402
from evidence.plugin import record  # noqa: E402

pytestmark = pytest.mark.x64


# -------------------------------------------- independent mathematical helpers
def _matern_corr(dist_arr, ell, nu):
    """Matérn correlation k(d; ell, nu) from the modified Bessel function K_nu
    (Rasmussen & Williams 2006, eq. 4.14), or the squared exponential at nu = inf."""
    d = np.asarray(dist_arr, dtype=float)
    if np.isinf(nu):
        return np.exp(-0.5 * (d / ell) ** 2)
    scaled = np.sqrt(2.0 * nu) * d / ell
    out = np.ones_like(scaled)
    nz = scaled > 0.0
    s = scaled[nz]
    out[nz] = (2.0 ** (1.0 - nu) / special.gamma(nu)) * (s ** nu) * special.kv(nu, s)
    return out


def _periodic_matern_spectrum(orders, ell_theta_rad, nu_theta):
    """Unnormalized azimuthal Matérn power S(m) at integer harmonic orders m."""
    m = np.asarray(orders, dtype=float)
    if np.isinf(nu_theta):
        return np.exp(-0.5 * (m * ell_theta_rad) ** 2)
    return (1.0 + (m * ell_theta_rad) ** 2 / (2.0 * nu_theta)) ** (-nu_theta - 0.5)


def _phasor(u_m, v_m, wl_m, dra_mas, ddec_mas):
    return np.exp(-2j * np.pi * ((u_m / wl_m) * (dra_mas * R.MAS) + (v_m / wl_m) * (ddec_mas * R.MAS)))


def _modulated_cone_cloud(tip, alpha, s0, length, tilt, pa, az_amps, az_pas, n_s=240, n_phi=512):
    """Independent 3D wall point cloud for a circular TruncatedCone with cosine wall modulations.

    The cone axis points towards position angle ``pa`` tilted ``tilt`` out of the sky; the ring
    azimuth runs from the ``pa + 90`` wall (``across``) towards ``in_plane = across x axis``.
    """
    a, b, p = np.deg2rad(alpha), np.deg2rad(tilt), np.deg2rad(pa)
    axis = np.array([np.sin(p) * np.cos(b), np.cos(p) * np.cos(b), np.sin(b)])
    across = np.array([np.cos(p), -np.sin(p), 0.0])
    in_plane = np.cross(across, axis)
    x_gl, w_gl = special.roots_legendre(n_s)
    s = s0 + 2.5 * length * (x_gl + 1.0)
    w_s = w_gl * 2.5 * length
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    s_grid, phi_grid = np.meshgrid(s, phi, indexing="ij")
    rho = s_grid * np.sin(a)
    centre = (s_grid * np.cos(a) - tip)[..., None] * axis
    pts = centre + rho[..., None] * (np.cos(phi_grid)[..., None] * across + np.sin(phi_grid)[..., None] * in_plane)
    phi_wall_deg = (pa + 90.0) + np.rad2deg(phi_grid)
    mod = np.ones_like(phi_grid)
    for m_order, (amp, pa_m) in enumerate(zip(az_amps, az_pas), start=1):
        mod = mod + amp * np.cos(m_order * np.deg2rad(phi_wall_deg - pa_m))
    weight = (w_s * (s * np.sin(a)) * np.exp(-(s - s0) / length))[:, None] * mod
    weight = weight / weight.sum()
    return sky.Cloud(pts[..., 0].ravel(), pts[..., 1].ravel(), weight.ravel())


def _multipole_field_sky_image(
    z, npix, scale, r_min, r_max, sigma, length_r, length_theta, nu_r, nu_theta,
    inc_deg, pa_deg, dra_mas, ddec_mas, outside, east_sign=1.0, img_rot_deg=0.0, img_dra=0.0, img_ddec=0.0,
):
    """Independent latent-to-sky evaluation of MultipoleField from documented math alone:
    1. Polar grid: radial Matérn Cholesky L_r times circulant periodic-Matérn FFT square root,
       with resolution-invariant origin taper tanh(r / length_r)^|m| when r_min == 0.
    2. Sky pixel centres of a (possibly rotated and shifted) Image grid deprojected into
       disk-plane (r, theta) and bilinearly interpolated with a one-spacing linear ramp to `outside`.
    """
    z_arr = np.asarray(z, dtype=float)
    n_r, n_th = z_arr.shape
    radii = np.linspace(r_min, r_max, n_r)
    spacing = (r_max - r_min) / (n_r - 1)
    lr = 5.0 * spacing if length_r is None else length_r
    l_r = np.linalg.cholesky(_matern_corr(np.abs(radii[:, None] - radii[None, :]), lr, nu_r) + 1e-6 * np.eye(n_r))
    freqs = np.fft.fftfreq(n_th, d=1.0 / n_th)
    s_th = _periodic_matern_spectrum(np.abs(freqs), length_theta, nu_theta)
    s_th = s_th / s_th.sum()
    modes = sigma * (l_r @ np.fft.fft(z_arr, axis=1))
    if r_min == 0.0:
        modes = modes * (np.tanh(radii[:, None] / lr) ** np.abs(freqs[None, :]))
    polar_vals = np.fft.ifft(modes * np.sqrt(n_th * s_th)[None, :], axis=1).real

    dy = ((npix - 1) / 2.0 - np.arange(npix)) * scale
    dx = east_sign * ((npix - 1) / 2.0 - np.arange(npix)) * scale
    dxx, dyy = np.meshgrid(dx, dy)
    rot_r = np.deg2rad(img_rot_deg)
    ee_sky = dxx * np.cos(rot_r) + dyy * np.sin(rot_r) + img_dra
    nn_sky = -dxx * np.sin(rot_r) + dyy * np.cos(rot_r) + img_ddec

    ee = ee_sky - dra_mas
    nn = nn_sky - ddec_mas
    p = np.deg2rad(pa_deg)
    a_maj = ee * np.sin(p) + nn * np.cos(p)
    b_min = (ee * np.cos(p) - nn * np.sin(p)) / np.cos(np.deg2rad(inc_deg))
    r_disk = np.hypot(a_maj, b_min)
    th_disk = np.mod(np.arctan2(b_min, a_maj), 2.0 * np.pi)

    u_r = np.clip((r_disk - r_min) / spacing, 0.0, n_r - 1.0)
    k0 = np.minimum(np.floor(u_r).astype(int), n_r - 2)
    fr = u_r - k0
    u_th = th_disk * (n_th / (2.0 * np.pi))
    j0 = np.floor(u_th).astype(int) % n_th
    j1 = (j0 + 1) % n_th
    fth = u_th - np.floor(u_th)
    interp = (
        (1.0 - fr) * ((1.0 - fth) * polar_vals[k0, j0] + fth * polar_vals[k0, j1])
        + fr * ((1.0 - fth) * polar_vals[k0 + 1, j0] + fth * polar_vals[k0 + 1, j1])
    )
    w_out = np.clip((r_disk - r_max) / spacing, 0.0, 1.0)
    w = np.maximum(w_out, np.clip((r_min - r_disk) / spacing, 0.0, 1.0)) if r_min > 0.0 else w_out
    return polar_vals, (1.0 - w) * interp + w * outside, ee_sky, nn_sky


# ------------------------------------------------ parametric radial profiles
@pytest.mark.validates("virgil.models.ThinRing", roots=["mathematics"], headline="max_abs_dV")
def test_thin_ring_matches_deprojected_bessel_j0():
    """An inclined, rotated, offset ThinRing has visibility J_0(2 pi q' r_0) times the sky offset phasor."""
    u, v = R.baselines(250, q_max=8e7, seed=21)
    wl = 1.65e-6
    r0, inc, pa, dra, ddec = 7.5, 52.0, 37.0, 0.85, -0.45
    disk = vm.MultipoleDisk(vm.ThinRing(r0), inc=inc, pa=pa, dra=dra, ddec=ddec)
    got = np.asarray(disk.model(u * wl, v * wl, wl))
    qd, _, _ = R.deproject(u, v, inc, pa)
    truth = special.j0(2.0 * np.pi * qd * r0 * R.MAS) * _phasor(u * wl, v * wl, wl, dra, ddec)
    err = record("max_abs_dV", np.abs(got - truth).max())
    assert err < 1e-13


@pytest.mark.validates("virgil.models.ThinRing", roots=["mathematics"], kind="control")
def test_thin_ring_rejects_diameter_or_pa_confusion():
    """Passing diameter instead of radius, or rotating PA by 90 deg, breaks ThinRing agreement."""
    u, v = R.baselines(120, q_max=6e7, seed=22)
    wl = 1.65e-6
    qd, _, _ = R.deproject(u, v, 50.0, 30.0)
    truth = special.j0(2.0 * np.pi * qd * 7.5 * R.MAS)
    wrong_diam = np.asarray(vm.MultipoleDisk(vm.ThinRing(15.0), inc=50.0, pa=30.0).model(u * wl, v * wl, wl))
    wrong_pa = np.asarray(vm.MultipoleDisk(vm.ThinRing(7.5), inc=50.0, pa=120.0).model(u * wl, v * wl, wl))
    assert np.abs(wrong_diam - truth).max() > 0.2
    assert np.abs(wrong_pa - truth).max() > 0.2


@pytest.mark.validates("virgil.models.GaussianRing", roots=["mathematics"], headline="max_abs_dV")
def test_gaussian_ring_matches_closed_form_centre_and_quadrature_ring():
    """GaussianRing(0, sigma) equals the closed-form 2D Gaussian exp(-2 pi^2 sigma^2 q'^2) up to its 6-sigma tail;
    at r_0 > 0 it matches adaptive Hankel quadrature (including zero imaginary part)."""
    u, v = R.baselines(250, q_max=6e7, seed=23)
    wl = 1.65e-6
    inc, pa = 42.0, -25.0
    qd, _, _ = R.deproject(u, v, inc, pa)
    # 1. r_0 = 0: 2D Gaussian exp(-2 pi^2 sigma^2 q'^2) (6-sigma clip drops exp(-18) ~ 1.5e-8 of tail)
    sigma = 2.4
    disk0 = vm.MultipoleDisk(vm.GaussianRing(0.0, sigma), inc=inc, pa=pa, n_r=64)
    got0 = np.asarray(disk0.model(u * wl, v * wl, wl))
    closed0 = np.exp(-2.0 * (np.pi * sigma * R.MAS * qd) ** 2)
    err0 = float(np.abs(got0 - closed0).max())
    assert err0 < 2e-8
    # 2. r_0 > 0: adaptive quadrature via scipy.integrate.quad
    r0, sig_ring = 9.5, 1.3
    disk_r = vm.MultipoleDisk(vm.GaussianRing(r0, sig_ring), inc=inc, pa=pa, n_r=64)
    got_r = np.asarray(disk_r.model(u * wl, v * wl, wl))
    quad_r = R.hankel_quad(R.gaussian_ring(r0, sig_ring), qd)
    err_r = float(np.abs(got_r - quad_r).max())
    record("max_abs_dV", max(err0, err_r))
    assert err_r < 1e-8


@pytest.mark.validates("virgil.models.GaussianRing", roots=["mathematics"], kind="control")
def test_gaussian_ring_sigma_is_not_fwhm():
    """Passing FWHM (2.355 sigma) in place of sigma produces a much broader GaussianRing."""
    u, v = R.baselines(100, q_max=6e7, seed=24)
    wl = 1.65e-6
    qd, _, _ = R.deproject(u, v, 35.0, 15.0)
    truth = R.hankel_quad(R.gaussian_ring(8.0, 1.5), qd)
    wrong = np.asarray(
        vm.MultipoleDisk(vm.GaussianRing(8.0, 1.5 * 2.0 * np.sqrt(2.0 * np.log(2.0))), inc=35.0, pa=15.0, n_r=64)
        .model(u * wl, v * wl, wl)
    )
    assert np.abs(wrong - truth).max() > 0.1


@pytest.mark.validates("virgil.models.RadialPowerLaw", roots=["mathematics"], headline="max_abs_dV")
def test_radial_power_law_matches_airy_annulus_and_soft_edge_quadrature():
    """Sharp RadialPowerLaw(0, R, 0) is an Airy disk, RadialPowerLaw(r_in, r_out, 0) is the analytic uniform annulus,
    and sloped/soft-edged profiles match adaptive Hankel quadrature (complex visibilities, Im V == 0)."""
    u, v = R.baselines(200, q_max=6e7, seed=25)
    wl = 1.65e-6
    inc, pa = 48.0, 65.0
    qd, _, _ = R.deproject(u, v, inc, pa)

    def airy(radius_mas):
        x = 2.0 * np.pi * radius_mas * R.MAS * qd
        return 2.0 * special.j1(x) / x

    # 1. Sharp uniform disk
    got_ud = np.asarray(vm.MultipoleDisk(vm.RadialPowerLaw(0.0, 6.5, 0.0), inc=inc, pa=pa, n_r=64).model(u * wl, v * wl, wl))
    err_ud = float(np.abs(got_ud - airy(6.5)).max())
    # 2. Sharp uniform annulus: (r_out^2 Airy(r_out) - r_in^2 Airy(r_in)) / (r_out^2 - r_in^2)
    r_in, r_out = 3.0, 8.0
    ann_truth = (r_out**2 * airy(r_out) - r_in**2 * airy(r_in)) / (r_out**2 - r_in**2)
    got_ann = np.asarray(vm.MultipoleDisk(vm.RadialPowerLaw(r_in, r_out, 0.0), inc=inc, pa=pa, n_r=64).model(u * wl, v * wl, wl))
    err_ann = float(np.abs(got_ann - ann_truth).max())
    # 3. Sloped soft-edge power law against adaptive quadrature
    got_soft = np.asarray(
        vm.MultipoleDisk(vm.RadialPowerLaw(4.0, 12.0, -1.5, edge_sigma=0.4), inc=inc, pa=pa, n_r=96)
        .model(u * wl, v * wl, wl)
    )
    soft_truth = R.hankel_quad(R.erf_power(4.0, 12.0, -1.5, 0.4), qd)
    err_soft = float(np.abs(got_soft - soft_truth).max())
    assert err_ud < 1e-12 and err_ann < 1e-12 and err_soft < 1e-8
    record("max_abs_dV", max(err_ud, err_ann, err_soft))


@pytest.mark.validates("virgil.models.RadialPowerLaw", roots=["mathematics"], kind="control")
def test_radial_power_law_distinguishes_sharp_and_soft_edges_and_slope_sign():
    """Softening edges or flipping the power-law index sign measurably alters RadialPowerLaw visibilities."""
    u, v = R.baselines(120, q_max=8e7, seed=26)
    wl = 1.65e-6
    sharp = np.asarray(vm.MultipoleDisk(vm.RadialPowerLaw(4.0, 12.0, -1.5), n_r=96).model(u * wl, v * wl, wl))
    soft = np.asarray(vm.MultipoleDisk(vm.RadialPowerLaw(4.0, 12.0, -1.5, edge_sigma=0.8), n_r=96).model(u * wl, v * wl, wl))
    pos_idx = np.asarray(vm.MultipoleDisk(vm.RadialPowerLaw(4.0, 12.0, +1.5), n_r=96).model(u * wl, v * wl, wl))
    assert np.abs(sharp - soft).max() > 0.02
    assert np.abs(sharp - pos_idx).max() > 0.05


@pytest.mark.validates("virgil.models.Gap", roots=["mathematics"], headline="max_abs_dV")
def test_gap_profile_matches_quadrature_and_reduces_to_base_at_zero_depth():
    """Gap(base, radius, sigma, depth=0) equals base; at depth > 0 it matches adaptive quadrature on GaussianRing."""
    u, v = R.baselines(200, q_max=7e7, seed=27)
    wl = 1.65e-6
    inc, pa = 44.0, 22.0
    qd, _, _ = R.deproject(u, v, inc, pa)
    base = vm.GaussianRing(9.0, 2.2)
    v_base = np.asarray(vm.MultipoleDisk(base, inc=inc, pa=pa, n_r=96).model(u * wl, v * wl, wl))
    v_zero = np.asarray(vm.MultipoleDisk(vm.Gap(base, radius=9.0, sigma=0.7, depth=0.0), inc=inc, pa=pa, n_r=96).model(u * wl, v * wl, wl))
    err_zero = float(np.abs(v_base - v_zero).max())
    assert err_zero < 1e-14

    v_gap = np.asarray(vm.MultipoleDisk(vm.Gap(base, radius=9.3, sigma=0.7, depth=0.85), inc=inc, pa=pa, n_r=96).model(u * wl, v * wl, wl))
    truth = R.hankel_quad(R.with_gap(R.gaussian_ring(9.0, 2.2), 9.3, 0.7, 0.85), qd)
    err_gap = float(np.abs(v_gap - truth).max())
    record("max_abs_dV", max(err_zero, err_gap))
    assert err_gap < 1e-8


@pytest.mark.validates("virgil.models.Gap", roots=["mathematics"], kind="control")
def test_gap_rejects_swapped_radius_and_depth():
    """Swapping radius and depth (the vis-r P10 indexing bug) fails decisively on Gap."""
    u, v = R.baselines(100, q_max=6e7, seed=28)
    wl = 1.65e-6
    base = vm.RadialPowerLaw(2.0, 14.0, 0.0, edge_sigma=0.3)
    right = np.asarray(vm.MultipoleDisk(vm.Gap(base, radius=8.0, sigma=1.0, depth=0.85), n_r=96).model(u * wl, v * wl, wl))
    swapped = np.asarray(vm.MultipoleDisk(vm.Gap(base, radius=0.85, sigma=1.0, depth=8.0), n_r=96).model(u * wl, v * wl, wl))
    assert np.abs(right - swapped).max() > 0.05


@pytest.mark.validates("virgil.models.RadialSum", roots=["mathematics"], headline="max_abs_dV")
def test_radial_sum_mixes_unit_flux_normalized_profiles_by_relative_weights():
    """RadialSum normalizes each component to unit total light before mixing by weights w_k / sum(w)."""
    u, v = R.baselines(220, q_max=7e7, seed=29)
    wl = 1.65e-6
    inc, pa, dra, ddec = 51.0, -35.0, -0.3, 0.6
    w1, w2, w3 = 1.0, 2.5, 0.7
    summed = vm.MultipoleDisk(
        vm.RadialSum(
            inner=vm.ThinRing(2.5),
            mid=(vm.GaussianRing(7.0, 1.1), w2),
            outer=(vm.RadialPowerLaw(10.0, 16.0, -1.0, edge_sigma=0.3), w3),
        ),
        inc=inc, pa=pa, dra=dra, ddec=ddec, n_r=96,
    )
    assert summed.get("profile.mid.weight") == pytest.approx(w2)
    got = np.asarray(summed.model(u * wl, v * wl, wl))

    # Independent component visibilities from SciPy J_0 and adaptive Hankel quadrature (zero virgil calls)
    qd, _, _ = R.deproject(u, v, inc, pa)
    ph = _phasor(u * wl, v * wl, wl, dra, ddec)
    v1 = special.j0(2.0 * np.pi * qd * 2.5 * R.MAS) * ph
    v2 = R.hankel_quad(R.gaussian_ring(7.0, 1.1), qd) * ph
    v3 = R.hankel_quad(R.erf_power(10.0, 16.0, -1.0, 0.3), qd) * ph
    truth = (w1 * v1 + w2 * v2 + w3 * v3) / (w1 + w2 + w3)
    err = record("max_abs_dV", np.abs(got - truth).max())
    assert err < 1e-8


@pytest.mark.validates("virgil.models.RadialSum", roots=["mathematics"], kind="control")
def test_radial_sum_weights_are_not_raw_peak_intensities():
    """Combining unnormalized radial functions I_1(r) + w I_2(r) (without per-part flux normalization) disagrees with RadialSum."""
    u, v = R.baselines(100, q_max=5e7, seed=30)
    wl = 1.65e-6
    # Two Gaussian rings at r = 3 and r = 12 have 2 pi int I(r) r dr differing by ~4x at unit peak
    summed = vm.MultipoleDisk(
        vm.RadialSum(inner=(vm.GaussianRing(3.0, 0.8), 1.0), outer=(vm.GaussianRing(12.0, 0.8), 1.0)), n_r=64
    )
    got = np.asarray(summed.model(u * wl, v * wl, wl))
    raw_profile = (
        lambda r: np.exp(-0.5 * ((r - 3.0) / 0.8) ** 2) + np.exp(-0.5 * ((r - 12.0) / 0.8) ** 2),
        0.0, 18.0, (3.0, 12.0),
    )
    unnorm = R.hankel_quad(raw_profile, np.hypot(u, v))
    assert np.abs(got - unnorm).max() > 0.12


# ------------------------------------------------ RadialLinear & RadialPower
@pytest.mark.validates("virgil.models.RadialLinear", "virgil.models.RadialPower", roots=["mathematics"],
                       headline="max_abs_dV")
def test_radial_linear_and_power_in_harmonics_and_chromatic_disk_match_brute_force_dft():
    """RadialLinear and RadialPower evaluate exact radial curves and drive radially varying harmonics and spectral indices."""
    r_test = np.array([3.0, 6.0, 9.5])
    lin = vm.RadialLinear(value=0.25, slope=0.04, r_ref=6.0)
    pw = vm.RadialPower(value=0.20, index=-1.2, r_ref=6.0)
    spec = vm.RadialPower(value=-0.9, index=0.4, r_ref=6.0)
    assert np.max(np.abs(np.asarray(lin(r_test)) - (0.25 + 0.04 * (r_test - 6.0)))) < 1e-15
    assert np.max(np.abs(np.asarray(pw(r_test)) - (0.20 * (r_test / 6.0) ** (-1.2)))) < 1e-15

    rng = np.random.default_rng(31)
    b = 110.0 * np.sqrt(rng.uniform(0.02, 1.0, 80))
    ang = rng.uniform(0, 2 * np.pi, 80)
    u_m, v_m = b * np.cos(ang), b * np.sin(ang)
    lam0, lam = 2.2e-6, 3.4e-6
    inc, pa = 46.0, 28.0

    disk = vm.MultipoleDisk(
        vm.GaussianRing(6.0, 0.9),
        harmonics=vm.Harmonics(c=(lin,), s=(pw,)),
        inc=inc, pa=pa, n_r=96, flux=0.7, spectral_index=spec, wavel0=lam0,
    )
    scene = vm.System(star=vm.PointSource(1.0), disk=disk)
    got = np.asarray(scene.model(u_m, v_m, lam))

    # Independent brute-force polar point cloud at wavelength lam
    r_nodes, wt0 = P.panelled_nodes(max(6.0 - 8 * 0.9, 1e-6), 6.0 + 8 * 0.9, lambda r: np.exp(-0.5 * ((r - 6.0) / 0.9) ** 2))
    alpha_r = -0.9 * (r_nodes / 6.0) ** 0.4
    wt_lam = wt0 * (lam / lam0) ** alpha_r
    f_disk = 0.7 * wt_lam.sum() / wt0.sum()
    c1_r = 0.25 + 0.04 * (r_nodes - 6.0)
    s1_r = 0.20 * (r_nodes / 6.0) ** (-1.2)
    th = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=False)
    rr, tt = np.meshgrid(r_nodes, th, indexing="ij")
    az = 1.0 + c1_r[:, None] * np.cos(tt) + s1_r[:, None] * np.sin(tt)
    w_grid = (wt_lam[:, None] * az).ravel()
    w_grid = w_grid / w_grid.sum()
    pa_r = np.deg2rad(pa)
    x_maj = rr.ravel() * np.cos(tt.ravel())
    y_min = rr.ravel() * np.sin(tt.ravel()) * np.cos(np.deg2rad(inc))
    dra_pts = x_maj * np.sin(pa_r) + y_min * np.cos(pa_r)
    ddec_pts = x_maj * np.cos(pa_r) - y_min * np.sin(pa_r)
    v_disk_ref = sky.visibility(sky.Cloud(dra_pts, ddec_pts, w_grid), u_m, v_m, lam)
    ref = (1.0 + f_disk * v_disk_ref) / (1.0 + f_disk)

    err = record("max_abs_dV", np.abs(got - ref).max())
    assert err < 1e-6


@pytest.mark.validates("virgil.models.RadialLinear", "virgil.models.RadialPower", roots=["mathematics"], kind="control")
def test_radial_functions_reject_wrong_reference_radius_or_sign():
    """Omitting r_ref or negating the slope/index changes radially modulated MultipoleDisk visibilities."""
    u, v = R.baselines(80, q_max=5e7, seed=32)
    wl = 2.2e-6
    good = vm.MultipoleDisk(
        vm.GaussianRing(6.0, 1.2),
        harmonics=vm.Harmonics(c=(vm.RadialLinear(0.2, 0.08, 6.0),), s=(vm.RadialPower(0.25, -1.5, 6.0),)),
        inc=40.0, pa=20.0, n_r=64,
    )
    wrong_ref = vm.MultipoleDisk(
        vm.GaussianRing(6.0, 1.2),
        harmonics=vm.Harmonics(c=(vm.RadialLinear(0.2, 0.08, 0.0),), s=(vm.RadialPower(0.25, +1.5, 6.0),)),
        inc=40.0, pa=20.0, n_r=64,
    )
    v_good = np.asarray(good.model(u * wl, v * wl, wl))
    v_wrong = np.asarray(wrong_ref.model(u * wl, v * wl, wl))
    assert np.abs(v_good - v_wrong).max() > 0.03


# ------------------------------------------------ RadialGP & MultipoleKernel
@pytest.mark.validates("virgil.models.RadialGP", roots=["mathematics", "statistics"], headline="max_abs_dV")
@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5, float("inf")])
def test_radial_gp_matches_independent_matern_cholesky_and_prior_covariance(nu):
    """RadialGP places n_r Gauss-Legendre nodes on [r_in, r_out], builds exp(mu + sigma L z) from the Matérn-nu Cholesky
    factor, and draws prior realizations with covariance sigma^2 K."""
    r_in, r_out, n_r = 1.5, 13.5, 28
    sigma, ell, mu, jitter = 1.3, 3.2, -0.4, 1e-6
    rng = np.random.default_rng(40 + (99 if np.isinf(nu) else int(10 * nu)))
    z = rng.normal(size=n_r)

    disk = vm.MultipoleDisk(
        vm.RadialGP(r_in, r_out, sigma=sigma, ell=ell, mu=mu, z=z, nu=nu, jitter=jitter),
        inc=47.0, pa=33.0, n_r=n_r,
    )
    # Independent Gauss-Legendre rule on [r_in, r_out]
    gl_x, gl_w = special.roots_legendre(n_r)
    half = 0.5 * (r_out - r_in)
    r_nodes = 0.5 * (r_in + r_out) + half * gl_x
    w_nodes = half * gl_w
    assert np.max(np.abs(np.asarray(disk.profile.nodes()) - r_nodes)) < 1e-13

    corr = _matern_corr(np.abs(r_nodes[:, None] - r_nodes[None, :]), ell, nu) + jitter * np.eye(n_r)
    l_chol = np.linalg.cholesky(corr)
    g = mu + sigma * (l_chol @ z)
    i_nodes = np.exp(g)

    u, v = R.baselines(160, q_max=5e7, seed=41)
    wl = 1.65e-6
    qd, _, _ = R.deproject(u, v, 47.0, 33.0)
    bessel = special.j0(2.0 * np.pi * qd[:, None] * (r_nodes[None, :] * R.MAS))
    weights = w_nodes * r_nodes * i_nodes
    truth_v = (bessel @ weights) / np.sum(weights)
    got_v = np.asarray(disk.model(u * wl, v * wl, wl))
    err_v = record("max_abs_dV", np.abs(got_v - truth_v).max())
    assert err_v < 5e-12

    # Empirical covariance of .draws() on the nodes matches sigma^2 corr
    draws = np.asarray(disk.profile.draws(jax.random.key(7), n=4000, positive=False))
    emp_cov = np.cov(draws, rowvar=False)
    cov_err = record("emp_cov_fro_rel", np.linalg.norm(emp_cov - sigma**2 * corr) / np.linalg.norm(sigma**2 * corr))
    assert np.max(np.abs(draws.mean(axis=0) - mu)) < 5.0 * sigma / np.sqrt(4000)
    assert cov_err < 0.06


@pytest.mark.validates("virgil.models.RadialGP", roots=["mathematics"], kind="control")
def test_radial_gp_distinguishes_kernel_smoothness_and_scale():
    """Changing nu or ell on the same non-zero whitened latent z alters RadialGP visibilities."""
    z = np.linspace(-1.2, 1.1, 24)
    u, v = R.baselines(100, q_max=5e7, seed=42)
    wl = 1.65e-6
    d_rough = vm.MultipoleDisk(vm.RadialGP(1.0, 12.0, sigma=1.5, ell=1.5, z=z, nu=0.5), n_r=24)
    d_smooth = vm.MultipoleDisk(vm.RadialGP(1.0, 12.0, sigma=1.5, ell=5.0, z=z, nu=float("inf")), n_r=24)
    assert np.abs(np.asarray(d_rough.model(u * wl, v * wl, wl)) - np.asarray(d_smooth.model(u * wl, v * wl, wl))).max() > 0.02


@pytest.mark.validates("virgil.models.MultipoleKernel", roots=["mathematics", "statistics"], headline="max_abs_dV")
@pytest.mark.parametrize("nu_theta", [1.5, float("inf")])
def test_multipole_kernel_spectrum_draws_and_disk_visibility(nu_theta):
    """MultipoleKernel matches S(m)/S(1), generates independent radial Matérn draws per harmonic order,
    and produces MultipoleDisk visibilities matching an independent Cholesky + linear-interpolation polar cloud."""
    r_in, r_out, r_ref = 2.0, 14.0, 8.0
    ell_r, ell_theta, sigma, n_nodes, degree = 3.5, 4.0, 0.35, 16, 3
    kern = vm.MultipoleKernel(
        ell_r=ell_r, ell_theta=ell_theta, sigma=sigma,
        r_in=r_in, r_out=r_out, r_ref=r_ref, n=n_nodes, nu_r=1.5, nu_theta=nu_theta,
    )
    m = np.arange(1, degree + 1)
    s_raw = _periodic_matern_spectrum(m, ell_theta / r_ref, nu_theta)
    s_rel = s_raw / s_raw[0]
    spec_got = np.asarray(kern.spectrum(degree))
    assert np.max(np.abs(spec_got - s_rel)) < 1e-14

    # Empirical variance across orders in .draws() decays as sigma^2 * S(m)/S(1)
    draws = np.asarray(kern.draws(jax.random.key(19), degree=degree, n=3000))
    assert draws.shape == (3000, 2, degree, n_nodes)
    var_by_m = draws.var(axis=(0, 1, 3))
    assert np.max(np.abs(var_by_m - sigma**2 * s_rel)) < 0.08 * sigma**2

    # End-to-end MultipoleDisk visibility with deterministic latents z (zero virgil calls in reference)
    rng = np.random.default_rng(43)
    z = rng.normal(size=(2, degree, n_nodes)) * 0.4
    disk = vm.MultipoleDisk(
        vm.GaussianRing(8.0, 1.2),
        harmonics=vm.Harmonics(degree=degree, form="linear", prior=kern),
        inc=45.0, pa=30.0, n_r=192,
    ).set("harmonics.z", jnp.asarray(z))
    u, v = R.baselines(80, q_max=4e7, seed=44)
    wl = 2.2e-6
    got = np.asarray(disk.model(u * wl, v * wl, wl))

    r_q, wt_q = P.panelled_nodes(max(8.0 - 8 * 1.2, 1e-6), 8.0 + 8 * 1.2, lambda r: np.exp(-0.5 * ((r - 8.0) / 1.2) ** 2))
    nodes = np.linspace(r_in, r_out, n_nodes)
    l_chol = np.linalg.cholesky(_matern_corr(np.abs(nodes[:, None] - nodes[None, :]), ell_r, 1.5) + 1e-6 * np.eye(n_nodes))
    scale_m = sigma * np.sqrt(s_rel)[:, None]
    c_nodes = scale_m * (z[0] @ l_chol.T)
    s_nodes = scale_m * (z[1] @ l_chol.T)
    c_r = np.array([np.interp(r_q, nodes, row) for row in c_nodes])
    s_r = np.array([np.interp(r_q, nodes, row) for row in s_nodes])
    c_vir, s_vir = (np.asarray(x) for x in disk.harmonics.coefficients(r_q))
    assert np.max(np.abs(c_r - c_vir)) < 1e-14 and np.max(np.abs(s_r - s_vir)) < 1e-14

    th = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=False)
    rr, tt = np.meshgrid(r_q, th, indexing="ij")
    az = np.ones_like(rr)
    for idx in range(degree):
        order = idx + 1
        az = az + c_r[idx, :, None] * np.cos(order * tt) + s_r[idx, :, None] * np.sin(order * tt)
    w_pts = (wt_q[:, None] * az).ravel()
    w_pts = w_pts / w_pts.sum()
    pa_r = np.deg2rad(30.0)
    x_maj = rr.ravel() * np.cos(tt.ravel())
    y_min = rr.ravel() * np.sin(tt.ravel()) * np.cos(np.deg2rad(45.0))
    cloud = sky.Cloud(x_maj * np.sin(pa_r) + y_min * np.cos(pa_r), x_maj * np.cos(pa_r) - y_min * np.sin(pa_r), w_pts)
    ref = sky.visibility(cloud, u * wl, v * wl, wl)
    err = record("max_abs_dV", np.abs(got - ref).max())
    assert err < 2e-5


@pytest.mark.validates("virgil.models.MultipoleKernel", roots=["mathematics"], kind="control")
def test_multipole_kernel_azimuthal_length_suppresses_high_orders():
    """Increasing ell_theta strongly damps m >= 2 relative to m = 1 in MultipoleKernel.spectrum."""
    short = np.asarray(vm.MultipoleKernel(ell_r=2.0, ell_theta=1.0, r_ref=8.0).spectrum(4))
    long = np.asarray(vm.MultipoleKernel(ell_r=2.0, ell_theta=16.0, r_ref=8.0).spectrum(4))
    assert short[-1] / long[-1] > 50.0


# ------------------------------------------------ virgil.harmonics conversions
@pytest.mark.validates("virgil.harmonics.to_image_harmonics", "virgil.harmonics.from_image_harmonics",
                       roots=["mathematics"], headline="max_abs_diff")
def test_image_harmonics_round_trip_linear_positive_and_von_mises():
    """to_image_harmonics matches exact Fourier integrals for linear, Fejér-Riesz positive, and VonMises forms,
    and from_image_harmonics inverts both linear and positive forms (including non-minimum-phase roots)."""
    # 1. Linear round-trip: I_m = (c_m - i s_m) / 2
    c, s = (0.25, -0.12, 0.07), (-0.18, 0.09, -0.05)
    h_lin = vm.Harmonics(c=c, s=s)
    im_lin = np.asarray(vh.to_image_harmonics(h_lin))
    expected_lin = np.array([1.0] + [0.5 * (cm - 1j * sm) for cm, sm in zip(c, s)])
    err_lin = float(np.max(np.abs(im_lin - expected_lin)))
    h_lin_rt = vh.from_image_harmonics(im_lin, form="linear")
    err_lin_rt = float(np.max(np.abs(np.asarray(vh.to_image_harmonics(h_lin_rt)) - im_lin)))

    # 2. Positive Fejér-Riesz spectral factorization from a non-minimum-phase polynomial (root inside unit circle!)
    # p(z) = (1 - 1.6 z)(1 + 0.5j z) has a root at z = 0.625 inside the unit circle; |p(e^{i theta})|^2 > 0 everywhere.
    th = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    z_circle = np.exp(1j * th)
    poly = (1.0 - 1.6 * z_circle) * (1.0 + 0.5j * z_circle)
    pos_prof = np.abs(poly) ** 2
    pos_prof = pos_prof / pos_prof.mean()
    im_target = np.array([np.mean(pos_prof * np.exp(-1j * m * th)) for m in range(3)])
    h_pos = vh.from_image_harmonics(im_target, form="positive")
    # Minimum-phase factor h(z) has all roots strictly outside the unit circle
    h_coeff = np.asarray(h_pos.polynomial(r=1.0))
    roots = np.roots(h_coeff[::-1])
    assert np.all(np.abs(roots) > 1.0)
    im_pos_got = np.asarray(vh.to_image_harmonics(h_pos))
    err_pos = float(np.max(np.abs(im_pos_got - im_target)))

    # 3. VonMises against modified Bessel ratios [I_m(kappa) / I_0(kappa)] exp(-i m theta0)
    kappa, theta0, deg = 6.5, 42.0, 12
    im_vm = np.asarray(vh.to_image_harmonics(vm.VonMises(kappa=kappa, theta0=theta0, degree=deg)))
    m_arr = np.arange(1, deg + 1)
    expected_vm = np.concatenate([
        [1.0 + 0.0j],
        (special.iv(m_arr, kappa) / special.i0(kappa)) * np.exp(-1j * m_arr * np.deg2rad(theta0)),
    ])
    err_vm = float(np.max(np.abs(im_vm - expected_vm)))

    err = record("max_abs_diff", max(err_lin, err_lin_rt, err_pos, err_vm))
    assert err < 1e-12


@pytest.mark.validates("virgil.harmonics.from_image_harmonics", "virgil.harmonics.to_image_harmonics",
                       roots=["mathematics"], kind="control")
def test_from_image_harmonics_positive_rejects_negative_profile_and_conjugation():
    """from_image_harmonics(..., form='positive') refuses an image that goes negative, and conjugating I_m flips sky phase."""
    # 1 + 1.4 cos(theta) dips to -0.4 at theta = pi -> I_1 = 0.7 has no Fejér-Riesz factorization
    with pytest.raises(ValueError):
        vh.from_image_harmonics(np.array([1.0, 0.7 + 0.0j]), form="positive")

    im = np.array([1.0, 0.15 - 0.2j])
    d1 = vm.MultipoleDisk(vm.ThinRing(6.0), harmonics=vh.from_image_harmonics(im, form="linear"), inc=40.0, pa=25.0)
    d2 = vm.MultipoleDisk(vm.ThinRing(6.0), harmonics=vh.from_image_harmonics(np.conj(im), form="linear"), inc=40.0, pa=25.0)
    u, v = R.baselines(60, q_max=4e7, seed=45)
    assert np.abs(np.asarray(d1.model(u * 2e-6, v * 2e-6, 2e-6)) - np.asarray(d2.model(u * 2e-6, v * 2e-6, 2e-6))).max() > 0.1


@pytest.mark.validates("virgil.harmonics.from_legacy", "virgil.harmonics.to_legacy",
                       "virgil.harmonics.rim_coefficients", "virgil.harmonics.cone_coefficients",
                       roots=["mathematics"], headline="max_abs_dV")
def test_legacy_rim_and_cone_conversions_match_trigonometry_and_point_clouds():
    """rim_coefficients, cone_coefficients, from_legacy, and to_legacy match independent trigonometric identities,
    canonical angle wrapping on [-180/m, 180/m) about the reference axis, and independent crosscheck point clouds."""
    u, v = R.baselines(150, q_max=5e7, seed=46)
    wl = 2.2e-6

    # 1. Rim trigonometry, canonical to_legacy wrapping, and crosscheck.sky.inclined_ring
    amps, pas, pa, inc, diam, fwhm = (0.35, 0.2, 0.12), (75.0, -20.0, 130.0), 32.0, 48.0, 11.0, 1.4
    m_rim = np.arange(1, len(amps) + 1)
    dphi_rim = np.deg2rad(np.asarray(pas) - pa)
    c_rim_ref = np.asarray(amps) * np.cos(m_rim * dphi_rim)
    s_rim_ref = np.asarray(amps) * np.sin(m_rim * dphi_rim)
    c_rim, s_rim = vh.rim_coefficients(np.asarray(amps), np.asarray(pas), pa)
    h_rim = vh.from_legacy(amps, pas, pa=pa, kind="rim")
    coeff_err_rim = float(max(
        np.max(np.abs(np.asarray(c_rim) - c_rim_ref)),
        np.max(np.abs(np.asarray(s_rim) - s_rim_ref)),
        np.max(np.abs(np.asarray(h_rim.c) - c_rim_ref)),
        np.max(np.abs(np.asarray(h_rim.s) - s_rim_ref)),
    ))
    assert coeff_err_rim < 1e-14

    amps_rt, pas_rt = (np.asarray(x) for x in vh.to_legacy(h_rim, pa=pa, kind="rim"))
    periods_rim = 360.0 / m_rim
    pas_canon = pa + ((np.asarray(pas) - pa + 0.5 * periods_rim) % periods_rim - 0.5 * periods_rim)
    assert np.all(amps_rt >= 0.0)
    assert np.max(np.abs(amps_rt - np.asarray(amps))) < 1e-14
    assert np.max(np.abs(pas_rt - pas_canon)) < 1e-12
    # Non-canonical wrapped inputs (shifted by integer multiples of 360/m) map to the identical canonical tuple
    pas_wrapped = tuple(p + 2.0 * per for p, per in zip(pas, periods_rim))
    amps_w, pas_w = (np.asarray(x) for x in vh.to_legacy(vh.from_legacy(amps, pas_wrapped, pa=pa, kind="rim"), pa=pa, kind="rim"))
    assert np.max(np.abs(amps_w - amps_rt)) < 1e-14 and np.max(np.abs(pas_w - pas_rt)) < 1e-12

    disk_rim = vm.MultipoleDisk(vm.ThinRing(diam / 2.0), harmonics=h_rim, inc=inc, pa=pa, blur_fwhm=fwhm)
    cloud_rim = sky.inclined_ring(diam, inc=inc, pa=pa, az_amps=amps, az_pas=pas, modulation="disk", n_phi=4096)
    ref_rim = sky.visibility(cloud_rim, u * wl, v * wl, wl) * sky.in_plane_blur_factor(u * wl, v * wl, wl, fwhm, inc, pa)
    err_rim = float(np.abs(np.asarray(disk_rim.model(u * wl, v * wl, wl)) - ref_rim).max())

    # 2. Cone trigonometry, canonical to_legacy wrapping, and independent 3D modulated wall cloud
    alpha, tip, s0, length, width, tilt, cone_pa = 38.0, 3.5, 4.0, 3.2, 0.8, 40.0, 65.0
    c_amps, c_pas = (0.4, 0.22), (115.0, 25.0)
    m_cone = np.arange(1, len(c_amps) + 1)
    ph_cone = np.deg2rad(cone_pa + 90.0 - np.asarray(c_pas))
    sign_m = (-1.0) ** m_cone
    c_cone_ref = sign_m * np.asarray(c_amps) * np.cos(m_cone * ph_cone)
    s_cone_ref = -sign_m * np.asarray(c_amps) * np.sin(m_cone * ph_cone)
    c_c, s_c = vh.cone_coefficients(np.asarray(c_amps), np.asarray(c_pas), cone_pa)
    h_cone = vh.from_legacy(c_amps, c_pas, pa=cone_pa, kind="cone")
    coeff_err_cone = float(max(
        np.max(np.abs(np.asarray(c_c) - c_cone_ref)),
        np.max(np.abs(np.asarray(s_c) - s_cone_ref)),
        np.max(np.abs(np.asarray(h_cone.c) - c_cone_ref)),
        np.max(np.abs(np.asarray(h_cone.s) - s_cone_ref)),
    ))
    assert coeff_err_cone < 1e-14

    c_amps_rt, c_pas_rt = (np.asarray(x) for x in vh.to_legacy(h_cone, pa=cone_pa, kind="cone"))
    periods_cone = 360.0 / m_cone
    ref_wall = cone_pa + 90.0
    c_pas_canon = ref_wall + ((np.asarray(c_pas) - ref_wall + 0.5 * periods_cone) % periods_cone - 0.5 * periods_cone)
    assert np.all(c_amps_rt >= 0.0)
    assert np.max(np.abs(c_amps_rt - np.asarray(c_amps))) < 1e-14
    assert np.max(np.abs(c_pas_rt - c_pas_canon)) < 1e-12

    sin_a = np.sin(np.deg2rad(alpha))
    disk_cone = vm.MultipoleDisk(
        vm.ExpTail(s0 * sin_a, length * sin_a, cutoff=5.0),
        harmonics=h_cone, height=vm.Cone(alpha=alpha, tip=tip),
        inc=90.0 - tilt, pa=cone_pa - 90.0, n_r=48,
    )
    sky_blur = sky.gaussian_blur_factor(u * wl, v * wl, wl, width)
    v_cone_disk = np.asarray(disk_cone.model(u * wl, v * wl, wl)) * sky_blur
    v_cone_cloud = sky.visibility(
        _modulated_cone_cloud(tip, alpha, s0, length, tilt, cone_pa, c_amps, c_pas), u * wl, v * wl, wl
    ) * sky_blur
    err_cone = float(np.abs(v_cone_disk - v_cone_cloud).max())

    record("max_coeff_err", max(coeff_err_rim, coeff_err_cone))
    err = record("max_abs_dV", max(err_rim, err_cone))
    assert err < 1e-6


@pytest.mark.validates("virgil.harmonics.from_legacy", "virgil.harmonics.to_legacy",
                       "virgil.harmonics.rim_coefficients", "virgil.harmonics.cone_coefficients",
                       roots=["self-consistency"], headline="max_abs_dV")
def test_legacy_conversions_preserve_rim_and_cone_model_equivalence():
    """MultipoleDisk with from_legacy/to_legacy reproduces ModulatedGaussianRim and TruncatedCone to machine precision."""
    u, v = R.baselines(120, q_max=5e7, seed=48)
    wl = 2.2e-6

    amps, pas, pa, inc, diam, fwhm = (0.35, 0.2, 0.12), (75.0, -20.0, 130.0), 32.0, 48.0, 11.0, 1.4
    rim = vm.ModulatedGaussianRim(diam=diam, fwhm=fwhm, inc=inc, pa=pa, az_amps=amps, az_pas=pas)
    h_rim = vh.from_legacy(amps, pas, pa=pa, kind="rim")
    amps_rt, pas_rt = vh.to_legacy(h_rim, pa=pa, kind="rim")
    disk_rim = vm.MultipoleDisk(
        vm.ThinRing(diam / 2.0), harmonics=vh.from_legacy(amps_rt, pas_rt, pa=pa, kind="rim"),
        inc=inc, pa=pa, blur_fwhm=fwhm,
    )
    err_rim = float(np.abs(np.asarray(rim.model(u * wl, v * wl, wl)) - np.asarray(disk_rim.model(u * wl, v * wl, wl))).max())

    alpha, tip, s0, length, width, tilt, cone_pa = 38.0, 3.5, 4.0, 3.2, 0.8, 40.0, 65.0
    c_amps, c_pas = (0.4, 0.22), (115.0, 25.0)
    cone_legacy = vm.TruncatedCone(
        tip=tip, alpha=alpha, s0=s0, length=length, width=width,
        tilt=tilt, pa=cone_pa, ratio=1.0, az_amps=c_amps, az_pas=c_pas, n_rings=48,
    )
    h_cone = vh.from_legacy(c_amps, c_pas, pa=cone_pa, kind="cone")
    c_amps_rt, c_pas_rt = vh.to_legacy(h_cone, pa=cone_pa, kind="cone")
    sin_a = np.sin(np.deg2rad(alpha))
    disk_cone = vm.MultipoleDisk(
        vm.ExpTail(s0 * sin_a, length * sin_a, cutoff=5.0),
        harmonics=vh.from_legacy(c_amps_rt, c_pas_rt, pa=cone_pa, kind="cone"),
        height=vm.Cone(alpha=alpha, tip=tip), inc=90.0 - tilt, pa=cone_pa - 90.0, n_r=48,
    )
    v_cone_disk = np.asarray(disk_cone.model(u * wl, v * wl, wl)) * sky.gaussian_blur_factor(u * wl, v * wl, wl, width)
    err_cone = float(np.abs(np.asarray(cone_legacy.model(u * wl, v * wl, wl)) - v_cone_disk).max())

    err = record("max_abs_dV", max(err_rim, err_cone))
    assert err < 1e-13


@pytest.mark.validates("virgil.harmonics.from_legacy", "virgil.harmonics.to_legacy",
                       "virgil.harmonics.rim_coefficients", "virgil.harmonics.cone_coefficients",
                       roots=["mathematics"], kind="control")
def test_legacy_conversions_distinguish_rim_and_cone_conventions():
    """Mixing up kind='rim' and kind='cone' flips the m = 1 wall phase and breaks visibility agreement."""
    u, v = R.baselines(80, q_max=4e7, seed=47)
    wl = 2.2e-6
    h_cone = vh.from_legacy((0.45,), (115.0,), pa=65.0, kind="cone")
    h_wrong = vh.from_legacy((0.45,), (115.0,), pa=65.0, kind="rim")
    d_right = vm.MultipoleDisk(vm.ThinRing(5.0), harmonics=h_cone, inc=50.0, pa=-25.0)
    d_wrong = vm.MultipoleDisk(vm.ThinRing(5.0), harmonics=h_wrong, inc=50.0, pa=-25.0)
    assert np.abs(np.asarray(d_right.model(u * wl, v * wl, wl)) - np.asarray(d_wrong.model(u * wl, v * wl, wl))).max() > 0.2


# ------------------------------------------------ virgil.fields.MultipoleField
@pytest.mark.validates("virgil.fields.MultipoleField", roots=["mathematics"], headline="max_rel_cov_err")
def test_multipole_field_polar_covariance_and_origin_regularity():
    """On an annulus (r_min_mas > 0) MultipoleField induces the separable Matérn radial x periodic-azimuthal
    covariance; at r = 0 (r_min_mas = 0) all m != 0 modes vanish so the origin is single-valued with variance
    sigma^2 * S(0) / sum_m S(m) < sigma^2."""
    n_r, n_th = 9, 12
    r_min, r_max, sigma, lr, lth = 4.0, 20.0, 1.4, 7.0, 0.9
    nu_r, nu_th = 1.5, 2.5

    freqs = np.fft.fftfreq(n_th, d=1.0 / n_th)  # integer harmonic orders in [-n_th/2, n_th/2)
    s_th = _periodic_matern_spectrum(np.abs(freqs), lth, nu_th)
    s_th = s_th / s_th.sum()
    dtheta = 2.0 * np.pi * (np.arange(n_th)[:, None] - np.arange(n_th)[None, :]) / n_th

    # 1. Annulus (r_min_mas > 0): exact separable covariance sigma^2 * kron(K_r, K_theta)
    cols_ann = []
    for k in range(n_r):
        for j in range(n_th):
            e = np.zeros((n_r, n_th))
            e[k, j] = 1.0
            f = vf.MultipoleField(
                e, image_shape=(16, 16), r_max_mas=r_max, r_min_mas=r_min,
                sigma=sigma, length_r=lr, length_theta=lth, nu=nu_r, nu_theta=nu_th,
            )
            cols_ann.append(np.asarray(f.polar_values()).ravel())
    a_ann = np.column_stack(cols_ann)
    cov_ann = a_ann @ a_ann.T

    radii_ann = np.linspace(r_min, r_max, n_r)
    k_r_ann = _matern_corr(np.abs(radii_ann[:, None] - radii_ann[None, :]), lr, nu_r)
    k_th = np.sum(s_th[None, None, :] * np.cos(freqs[None, None, :] * dtheta[:, :, None]), axis=-1)
    cov_ann_expected = sigma**2 * np.kron(k_r_ann, k_th)
    rel_err_ann = record("max_rel_cov_err", np.max(np.abs(cov_ann - cov_ann_expected)) / sigma**2)
    assert rel_err_ann < 5e-6

    # 2. At r = 0 (r_min_mas = 0, row 0), all m != 0 modes vanish: single-valued with variance sigma^2 * s_th[0] < sigma^2
    origin_var = 0.0
    for k in range(n_r):
        for j in range(n_th):
            e = np.zeros((n_r, n_th))
            e[k, j] = 1.0
            f = vf.MultipoleField(
                e, image_shape=(16, 16), r_max_mas=r_max, r_min_mas=0.0,
                sigma=sigma, length_r=lr, length_theta=lth, nu=nu_r, nu_theta=nu_th,
            )
            pv_e = np.asarray(f.polar_values())
            assert np.ptp(pv_e[0, :]) < 1e-14
            origin_var += float(pv_e[0, 0]) ** 2
    assert origin_var == pytest.approx(sigma**2 * s_th[0], rel=1e-4)
    assert origin_var < 0.5 * sigma**2


@pytest.mark.validates("virgil.fields.MultipoleField", roots=["mathematics"], headline="max_rel_cov_err_lr")
def test_f25_multipole_field_origin_taper_uses_physical_length_r_and_is_grid_invariant():
    """On a full disk (r_min_mas = 0), MultipoleField tapers order m by tanh(r / length_r)^|m| (F25 fixed in
    virgil#673) so the polar prior covariance depends only on physical radius and is invariant when n_r is refined."""
    r_max, sigma, lr, lth, nu_r, nu_th = 20.0, 1.4, 7.0, 0.9, 1.5, 2.5
    n_th = 12
    freqs = np.fft.fftfreq(n_th, d=1.0 / n_th)
    s_th = _periodic_matern_spectrum(np.abs(freqs), lth, nu_th)
    s_th = s_th / s_th.sum()
    dtheta = 2.0 * np.pi * (np.arange(n_th)[:, None] - np.arange(n_th)[None, :]) / n_th

    def induced_polar_cov(n_r_val):
        cols = []
        for k in range(n_r_val):
            for j in range(n_th):
                e = np.zeros((n_r_val, n_th))
                e[k, j] = 1.0
                f = vf.MultipoleField(
                    e, image_shape=(16, 16), r_max_mas=r_max, r_min_mas=0.0,
                    sigma=sigma, length_r=lr, length_theta=lth, nu=nu_r, nu_theta=nu_th,
                )
                cols.append(np.asarray(f.polar_values()).ravel())
        a_mat = np.column_stack(cols)
        return a_mat @ a_mat.T

    cov_9 = induced_polar_cov(9)
    cov_17 = induced_polar_cov(17)
    # Row k of n_r = 9 sits at r_k = 2.5 * k mas, coinciding exactly with row 2 * k of n_r = 17
    shared_idx = np.array([2 * k * n_th + j for k in range(9) for j in range(n_th)])
    cov_17_shared = cov_17[np.ix_(shared_idx, shared_idx)]

    radii_9 = np.linspace(0.0, r_max, 9)
    k_r_9 = _matern_corr(np.abs(radii_9[:, None] - radii_9[None, :]), lr, nu_r)
    taper_lr = np.tanh(radii_9[:, None] / lr) ** np.abs(freqs[None, :])
    weight_m = s_th[None, None, :] * taper_lr[:, None, :] * taper_lr[None, :, :]
    cos_m = np.cos(freqs[None, None, :] * dtheta[:, :, None])
    cov_lr_expected = (
        sigma**2 * k_r_9[:, :, None, None] * np.einsum("abm,ijm->abij", weight_m, cos_m)
    ).transpose(0, 2, 1, 3).reshape(9 * n_th, 9 * n_th)

    rel_err_lr = record("max_rel_cov_err_lr", np.max(np.abs(cov_9 - cov_lr_expected)) / sigma**2)
    refinement_drift = record("max_rel_refinement_drift", np.max(np.abs(cov_9 - cov_17_shared)) / sigma**2)
    assert rel_err_lr < 2e-5
    assert refinement_drift < 1e-12


@pytest.mark.validates("virgil.fields.MultipoleField", roots=["mathematics"], headline="max_abs_dV")
def test_multipole_field_image_visibility_and_tied_system_geometry():
    """MultipoleField.evaluate and Image(MultipoleField) match an independent latent-to-sky polar interpolation
    and 2D pixel DFT, and geometry_from='disk' maps sibling disk coordinates into a rotated, shifted Image frame."""
    npix, scale = 48, 1.0
    rng = np.random.default_rng(52)
    z = rng.normal(size=(12, 16)) * 0.6
    r_min, r_max = 2.5, 15.0
    inc, pa, dra, ddec, outside = 48.0, 35.0, 1.2, -0.8, -25.0
    field = vf.MultipoleField(
        z, image_shape=(npix, npix), r_max_mas=r_max, r_min_mas=r_min, sigma=1.1,
        length_r=4.0, length_theta=0.8, inc=inc, pa=pa, dra=dra, ddec=ddec, outside=outside,
    )
    eta = np.asarray(field.evaluate(scale))
    pv_ref, eta_ref, _, _ = _multipole_field_sky_image(
        z, npix, scale, r_min, r_max, 1.1, 4.0, 0.8, 1.5, 1.5, inc, pa, dra, ddec, outside
    )
    assert np.max(np.abs(np.asarray(field.polar_values()) - pv_ref)) < 1e-13
    pix_err = record("max_abs_d_eta", np.max(np.abs(eta - eta_ref)))
    assert pix_err < 1e-7
    assert eta[0, 0] == pytest.approx(outside, abs=1e-6) and eta[-1, -1] == pytest.approx(outside, abs=1e-6)

    img = vm.Image(field, pixel_scale_mas=scale, flux=1.0)
    u, v = R.baselines(100, q_max=2.5e7, seed=53)
    wl = 2.2e-6
    got = np.asarray(img.model(u * wl, v * wl, wl))

    # Independent pixel-grid DFT built solely from eta_ref (zero virgil calls on the reference side)
    w_pix = np.exp(eta_ref - eta_ref.max())
    ref = sky.visibility(sky.pixel_image(w_pix, scale), u * wl, v * wl, wl)
    err_img = float(np.abs(got - ref).max())
    assert err_img < 1e-9

    # Tied geometry via geometry_from='disk' inside a rotated and shifted Image frame, checked against crosscheck
    disk = vm.MultipoleDisk(vm.GaussianRing(8.0, 1.2), inc=30.0, pa=15.0, dra=0.5, ddec=-0.4, flux=0.6)
    tied = vf.MultipoleField.from_disk(
        disk, z, image_shape=(npix, npix), r_max_mas=r_max, r_min_mas=r_min,
        sigma=1.1, length_r=4.0, length_theta=0.8, outside=outside, geometry_from="disk",
    )
    assert tied.inc is None and tied.pa is None
    img_rot, img_dra, img_ddec = 22.0, 0.7, -0.5
    sys_before = vm.System(
        disk=disk,
        env=vm.Image(tied, pixel_scale_mas=scale, flux=0.4, rotation_deg=img_rot, dra=img_dra, ddec=img_ddec),
    )
    sys_tied = sys_before.set({"disk.inc": 55.0, "disk.pa": 110.0, "disk.dra": -1.1, "disk.ddec": 0.9})
    v_sys_got = np.asarray(sys_tied.model(u * wl, v * wl, wl))
    v_sys_before = np.asarray(sys_before.model(u * wl, v * wl, wl))
    assert np.abs(v_sys_got - v_sys_before).max() > 0.05

    qd_d, _, _ = R.deproject(u, v, 55.0, 110.0)
    v_disk_ref = R.hankel_quad(R.gaussian_ring(8.0, 1.2), qd_d) * _phasor(u * wl, v * wl, wl, -1.1, 0.9)
    _, eta_tied_ref, ee_s, nn_s = _multipole_field_sky_image(
        z, npix, scale, r_min, r_max, 1.1, 4.0, 0.8, 1.5, 1.5,
        55.0, 110.0, -1.1, 0.9, outside, img_rot_deg=img_rot, img_dra=img_dra, img_ddec=img_ddec,
    )
    w_tied = np.exp(eta_tied_ref - eta_tied_ref.max())
    w_tied = w_tied / w_tied.sum()
    v_env_ref = sky.visibility(sky.Cloud(ee_s.ravel(), nn_s.ravel(), w_tied.ravel()), u * wl, v * wl, wl)
    v_sys_ref = 0.6 * v_disk_ref + 0.4 * v_env_ref
    err_tie = float(np.abs(v_sys_got - v_sys_ref).max())
    err = record("max_abs_dV", max(err_img, err_tie))
    assert err < 1e-8


@pytest.mark.validates("virgil.fields.MultipoleField", roots=["mathematics"], kind="control")
def test_multipole_field_rejects_rotated_pa_or_flipped_east_axis():
    """Rotating MultipoleField's PA by 90 deg or flipping the East sky sign breaks visibility agreement."""
    npix, scale = 40, 1.0
    z = np.zeros((10, 16))
    z[5, 0] = 2.5  # strong azimuthal asymmetry off the sky axes
    f_right = vf.MultipoleField(z, image_shape=(npix, npix), r_max_mas=14.0, r_min_mas=2.0, inc=55.0, pa=30.0)
    u, v = R.baselines(80, q_max=2.5e7, seed=54)
    wl = 2.2e-6
    v_got = np.asarray(vm.Image(f_right, pixel_scale_mas=scale).model(u * wl, v * wl, wl))

    def vis_from_ref(pa_deg, east_sign):
        _, eta_r, _, _ = _multipole_field_sky_image(
            z, npix, scale, 2.0, 14.0, 1.0, None, 1.0, 1.5, 1.5, 55.0, pa_deg, 0.0, 0.0, -20.0, east_sign=east_sign
        )
        return sky.visibility(sky.pixel_image(np.exp(eta_r - eta_r.max()), scale), u * wl, v * wl, wl)

    assert np.abs(v_got - vis_from_ref(30.0, +1.0)).max() < 1e-9
    assert np.abs(v_got - vis_from_ref(120.0, +1.0)).max() > 0.05
    assert np.abs(v_got - vis_from_ref(30.0, -1.0)).max() > 0.05
