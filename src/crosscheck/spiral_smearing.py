"""Independent references for Archimedean spiral plumes and bandwidth smearing.

Nothing here imports virgil, drpangloss or jaxbessel.

References
----------
* Archimedean dust-plume geometry: Tuthill, Monnier & Danchi (1999, Nature
  398, 487) and Tuthill et al. (2008, ApJ 675, 698). In polar coordinates
  r(theta) = (|p| / 2 pi) theta in the arm's plane, the differential arc
  length is ds = (|p| / 2 pi) sqrt(1 + theta^2) dtheta.
* Inclined sky projection and OIFITS Fourier sign: Pauls et al. (2005, PASP
  117, 1255) and Berger & Segransan (2007, New Astron. Rev. 51, 576).
* Finite-bandwidth fringe-washing (boxcar in wavenumber sigma = 1 / lambda):
  Thompson, Moran & Swenson (2017, Interferometry and Synthesis in Radio
  Astronomy, 3rd ed., Springer, section 6.3, Eq. 6.36).
"""

import numpy as np
from scipy.special import roots_legendre

from .sky import (
    FWHM_PER_SIGMA,
    MAS,
    Cloud,
    blurred,
    gaussian_blur_factor,
    in_plane_blur_factor,
    shift_phase,
    visibility,
)


def _departure_inplane_azimuth(pa_deg: float, inc_deg: float, node_deg: float) -> float:
    """In-plane azimuth alpha_0 (radians from the major axis at ``node_deg``)
    whose sky projection points along position angle ``pa_deg``."""
    dpsi = np.deg2rad(pa_deg - node_deg)
    cos_i = np.cos(np.deg2rad(inc_deg))
    return float(np.arctan2(np.sin(dpsi), cos_i * np.cos(dpsi)))


def spiral_spine_cloud(
    pitch: float,
    turns: float,
    pa: float = 0.0,
    inc: float = 0.0,
    node: float = 0.0,
    fade: float = 0.0,
    dra: float = 0.0,
    ddec: float = 0.0,
    n_quad: int = 512,
    fade_mode: str = "arc_length",
) -> Cloud:
    """Gauss-Legendre quadrature cloud along an inclined Archimedean spiral spine.

    Parameters
    ----------
    pitch : float
        Radial growth per full turn in the arm plane (mas); positive winds
        counter-clockwise (increasing position angle when face-on).
    turns : float
        Number of turns (> 0); integrates theta over [0, 2 pi turns].
    pa : float
        On-sky departure position angle at the origin (degrees North to East).
    inc, node : float
        Inclination (degrees, 0 = face-on) and sky position angle of the
        projected major axis (degrees North to East).
    fade : float
        Exponential e-foldings lost per turn along the arm (>= 0).
    dra, ddec : float
        Centre offset on the sky (mas, East and North).
    n_quad : int
        Number of Gauss-Legendre nodes over [0, 2 pi turns].
    fade_mode : {"arc_length", "angle_only"}
        ``"arc_length"`` includes the polar arc-length element
        ``sqrt(1 + theta^2)``; ``"angle_only"`` omits it (for negative
        controls).
    """
    if pitch == 0.0:
        raise ValueError("pitch must be non-zero")
    if turns <= 0.0:
        raise ValueError("turns must be positive")

    theta_max = 2.0 * np.pi * float(turns)
    x_gl, w_gl = roots_legendre(int(n_quad))
    theta = 0.5 * theta_max * (x_gl + 1.0)
    decay = np.exp(-float(fade) * theta / (2.0 * np.pi))
    if fade_mode == "arc_length":
        density = decay * np.sqrt(1.0 + theta**2)
    elif fade_mode == "angle_only":
        density = decay
    else:
        raise ValueError(f"Unknown fade_mode: {fade_mode!r}")

    weights = w_gl * density
    weights = weights / np.sum(weights)

    r_plane = abs(float(pitch)) * theta / (2.0 * np.pi)
    alpha0 = _departure_inplane_azimuth(pa, inc, node)
    sign_p = 1.0 if pitch > 0.0 else -1.0
    alpha = alpha0 + sign_p * theta

    cos_i = np.cos(np.deg2rad(inc))
    x_maj = r_plane * np.cos(alpha)
    x_min = r_plane * np.sin(alpha) * cos_i

    node_rad = np.deg2rad(node)
    east = float(dra) + x_maj * np.sin(node_rad) + x_min * np.cos(node_rad)
    north = float(ddec) + x_maj * np.cos(node_rad) - x_min * np.sin(node_rad)
    return Cloud(east=east, north=north, weight=weights)


def spiral_cvis(
    u,
    v,
    wavel,
    pitch: float,
    turns: float,
    width: float,
    pa: float = 0.0,
    inc: float = 0.0,
    node: float = 0.0,
    fade: float = 0.0,
    dra: float = 0.0,
    ddec: float = 0.0,
    n_quad: int = 512,
    fade_mode: str = "arc_length",
    blur_mode: str = "arm_plane",
):
    """Continuous-arc Fourier visibility of a Gaussian-blurred Archimedean spiral."""
    spine = spiral_spine_cloud(
        pitch=pitch,
        turns=turns,
        pa=pa,
        inc=inc,
        node=node,
        fade=fade,
        dra=dra,
        ddec=ddec,
        n_quad=n_quad,
        fade_mode=fade_mode,
    )
    vis_spine = visibility(spine, u, v, wavel)
    if blur_mode == "arm_plane":
        envelope = in_plane_blur_factor(u, v, wavel, width, inc, node)
    elif blur_mode == "sky":
        envelope = gaussian_blur_factor(u, v, wavel, width)
    else:
        raise ValueError(f"Unknown blur_mode: {blur_mode!r}")
    return vis_spine * envelope


def spiral_cloud_2d(
    pitch: float,
    turns: float,
    width: float,
    pa: float = 0.0,
    inc: float = 0.0,
    node: float = 0.0,
    fade: float = 0.0,
    dra: float = 0.0,
    ddec: float = 0.0,
    n_quad: int = 192,
    n_gauss: int = 8,
) -> Cloud:
    """2D point cloud (Gauss-Legendre spine convolved with Gauss-Hermite arm-plane
    Gaussian) for optical wavefront propagation in dLux."""
    spine = spiral_spine_cloud(
        pitch=pitch,
        turns=turns,
        pa=pa,
        inc=inc,
        node=node,
        fade=fade,
        dra=dra,
        ddec=ddec,
        n_quad=n_quad,
    )
    return blurred(spine, width, n=n_gauss, inc=inc, pa=node)


def spiral_render(
    npix: int,
    fov_mas: float,
    pitch: float,
    turns: float,
    width: float,
    pa: float = 0.0,
    inc: float = 0.0,
    node: float = 0.0,
    fade: float = 0.0,
    dra: float = 0.0,
    ddec: float = 0.0,
    n_quad: int = 512,
    floor_px: bool = True,
) -> np.ndarray:
    """Unit-sum 2D image of the Gaussian-blurred Archimedean spiral on an
    East-left, North-up ``npix x npix`` pixel grid spanning ``fov_mas``."""
    spine = spiral_spine_cloud(
        pitch=pitch,
        turns=turns,
        pa=pa,
        inc=inc,
        node=node,
        fade=fade,
        dra=dra,
        ddec=ddec,
        n_quad=n_quad,
    )
    pix = float(fov_mas) / int(npix)
    eff_width = max(float(width), pix) if floor_px else float(width)
    cos_i = abs(float(np.cos(np.deg2rad(inc))))
    sig_maj = eff_width / FWHM_PER_SIGMA
    sig_min = max(eff_width * cos_i, 1e-30) / FWHM_PER_SIGMA

    coords = ((int(npix) - 1) / 2.0 - np.arange(int(npix))) * pix
    ee, nn = np.meshgrid(coords, coords)

    node_rad = np.deg2rad(node)
    sin_n, cos_n = np.sin(node_rad), np.cos(node_rad)

    img = np.zeros((int(npix), int(npix)), dtype=float)
    step = 64
    for start in range(0, spine.weight.size, step):
        sl = slice(start, start + step)
        de = ee[None, :, :] - spine.east[sl, None, None]
        dn = nn[None, :, :] - spine.north[sl, None, None]
        dmaj = de * sin_n + dn * cos_n
        dmin = de * cos_n - dn * sin_n
        blobs = np.exp(-0.5 * ((dmaj / sig_maj) ** 2 + (dmin / sig_min) ** 2))
        img += np.tensordot(spine.weight[sl], blobs, axes=(0, 0))

    return img / np.sum(img)


def sinc_smearing_envelope(u, v, wavel, bandwidth, dra: float, ddec: float):
    """Exact fringe-washing envelope for a uniform boxcar channel in wavenumber
    sigma = 1 / lambda of width Delta_sigma = bandwidth / wavel^2:

        sinc(pi * bandwidth * (u * alpha + v * delta) / wavel^2)

    where ``np.sinc(x) == sin(pi x) / (pi x)``.
    """
    u, v, wavel, bandwidth = np.broadcast_arrays(
        np.asarray(u, float),
        np.asarray(v, float),
        np.asarray(wavel, float),
        np.asarray(bandwidth, float),
    )
    tau_geom = (u * float(dra) + v * float(ddec)) * MAS
    arg = bandwidth * tau_geom / (wavel**2)
    return np.sinc(arg)


def smeared_offset_phasor(u, v, wavel, bandwidth, dra: float, ddec: float):
    """Offset phasor multiplied by the wavenumber-boxcar sinc envelope."""
    return shift_phase(u, v, wavel, dra, ddec) * sinc_smearing_envelope(
        u, v, wavel, bandwidth, dra, ddec
    )


def smeared_binary_wavenumber_quadrature(
    u,
    v,
    wavel,
    bandwidth,
    dra: float,
    ddec: float,
    flux_ratio: float,
    dra_pri: float = 0.0,
    ddec_pri: float = 0.0,
    n_quad: int = 64,
):
    """Complex visibility of a two-point system integrated by Gauss-Legendre
    quadrature over a uniform boxcar in wavenumber sigma = 1 / lambda of width
    Delta_sigma = bandwidth / wavel^2."""
    u, v, wavel, bandwidth = np.broadcast_arrays(
        np.asarray(u, float),
        np.asarray(v, float),
        np.asarray(wavel, float),
        np.asarray(bandwidth, float),
    )
    x_gl, w_gl = roots_legendre(int(n_quad))
    wt = 0.5 * w_gl  # sums to 1 on [-1, 1]

    sigma0 = 1.0 / wavel
    dsigma = bandwidth / (wavel**2)
    sigma_nodes = sigma0[..., None] + 0.5 * dsigma[..., None] * x_gl

    tau_pri = (u * float(dra_pri) + v * float(ddec_pri)) * MAS
    tau_sec = (u * float(dra) + v * float(ddec)) * MAS
    v_pri = np.exp(-2j * np.pi * tau_pri[..., None] * sigma_nodes)
    v_sec = np.exp(-2j * np.pi * tau_sec[..., None] * sigma_nodes)
    cvis_nodes = (v_pri + float(flux_ratio) * v_sec) / (1.0 + float(flux_ratio))
    return np.sum(cvis_nodes * wt, axis=-1)


def smeared_binary_wavelength_quadrature(
    u,
    v,
    wavel,
    bandwidth,
    dra: float,
    ddec: float,
    flux_ratio: float,
    dra_pri: float = 0.0,
    ddec_pri: float = 0.0,
    n_quad: int = 64,
):
    """Complex visibility of a two-point system integrated by Gauss-Legendre
    quadrature over a uniform boxcar in wavelength [wavel - bandwidth/2,
    wavel + bandwidth/2]."""
    u, v, wavel, bandwidth = np.broadcast_arrays(
        np.asarray(u, float),
        np.asarray(v, float),
        np.asarray(wavel, float),
        np.asarray(bandwidth, float),
    )
    x_gl, w_gl = roots_legendre(int(n_quad))
    wt = 0.5 * w_gl

    lam_nodes = wavel[..., None] + 0.5 * bandwidth[..., None] * x_gl
    tau_pri = (u * float(dra_pri) + v * float(ddec_pri)) * MAS
    tau_sec = (u * float(dra) + v * float(ddec)) * MAS
    v_pri = np.exp(-2j * np.pi * tau_pri[..., None] / lam_nodes)
    v_sec = np.exp(-2j * np.pi * tau_sec[..., None] / lam_nodes)
    cvis_nodes = (v_pri + float(flux_ratio) * v_sec) / (1.0 + float(flux_ratio))
    return np.sum(cvis_nodes * wt, axis=-1)


def coherent_endpoint_binary_observables(
    u,
    v,
    wavel,
    bandwidth,
    dra: float,
    ddec: float,
    flux_ratio: float,
    i_cps=None,
    n_smear: int = 3,
):
    """Coherent endpoint average over ``n_smear`` equally spaced wavelengths
    spanning ``[wavel - bandwidth/2, wavel + bandwidth/2]`` (endpoints included,
    equal weights ``1 / n_smear``).

    At ``n_smear = 3`` this is CANDID's default bandwidth-smearing rule:
    ``V_bar = mean_{s in {-1/2, 0, +1/2}} V(lambda + s * dwavel)`` with spectral
    variance ``dwavel^2 / 6`` (twice the ``dwavel^2 / 12`` variance of a
    continuous uniform boxcar), ``V2 = |V_bar|^2``, and
    ``CP = arg(V_bar_12 * V_bar_23 * conj(V_bar_13))``.
    """
    u, v, wavel, bandwidth = np.broadcast_arrays(
        np.asarray(u, float),
        np.asarray(v, float),
        np.asarray(wavel, float),
        np.asarray(bandwidth, float),
    )
    if int(n_smear) <= 2:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-0.5, 0.5, int(n_smear))

    lam_nodes = wavel[..., None] + bandwidth[..., None] * offsets
    tau_sec = (u * float(dra) + v * float(ddec)) * MAS
    cvis_nodes = (1.0 + float(flux_ratio) * np.exp(-2j * np.pi * tau_sec[..., None] / lam_nodes)) / (
        1.0 + float(flux_ratio)
    )
    cvis_mean = np.mean(cvis_nodes, axis=-1)
    v2 = np.abs(cvis_mean) ** 2

    if i_cps is None:
        return v2, None

    i1, i2, i3 = i_cps
    bisp = cvis_mean[i1] * cvis_mean[i2] * np.conj(cvis_mean[i3])
    cp_rad = np.angle(bisp)
    return v2, cp_rad


def incoherent_endpoint_binary_observables(
    u,
    v,
    wavel,
    bandwidth,
    dra: float,
    ddec: float,
    flux_ratio: float,
    i_cps=None,
    n_smear: int = 3,
):
    """Incoherent ``V^2`` average ``mean(|V(lambda)|^2)`` over ``n_smear``
    equally spaced wavelengths spanning ``[wavel - bandwidth/2, wavel + bandwidth/2]``
    (contrasting incoherent squared-visibility averaging, such as PMOIRED's
    ``'wl kernel'`` on ``OI_VIS2``, against coherent fringe washing ``|mean(V)|^2``).

    Returns ``(v2, cp_rad)`` where ``cp_rad`` is ``None`` if ``i_cps`` is ``None``
    and otherwise the argument of the mean complex bispectrum.
    """
    u, v, wavel, bandwidth = np.broadcast_arrays(
        np.asarray(u, float),
        np.asarray(v, float),
        np.asarray(wavel, float),
        np.asarray(bandwidth, float),
    )
    if int(n_smear) <= 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-0.5, 0.5, int(n_smear))

    lam_nodes = wavel[..., None] + bandwidth[..., None] * offsets
    tau_sec = (u * float(dra) + v * float(ddec)) * MAS
    cvis_nodes = (1.0 + float(flux_ratio) * np.exp(-2j * np.pi * tau_sec[..., None] / lam_nodes)) / (
        1.0 + float(flux_ratio)
    )
    v2 = np.mean(np.abs(cvis_nodes) ** 2, axis=-1)

    if i_cps is None:
        return v2, None

    i1, i2, i3 = i_cps
    bisp_nodes = cvis_nodes[i1] * cvis_nodes[i2] * np.conj(cvis_nodes[i3])
    cp_rad = np.angle(np.mean(bisp_nodes, axis=-1))
    return v2, cp_rad


def discrete_gaussian_spectral_kernel(wl_kernel_pix: float) -> np.ndarray:
    """Normalized discrete Gaussian spectral kernel of FWHM ``wl_kernel_pix``
    channels on ``N = 2 * int(2 * wl_kernel_pix) + 3`` taps."""
    n_taps = 2 * int(2.0 * float(wl_kernel_pix)) + 3
    x = np.arange(n_taps, dtype=float)
    sigma_pix = float(wl_kernel_pix) / FWHM_PER_SIGMA
    ker = np.exp(-0.5 * ((x - np.mean(x)) / sigma_pix) ** 2)
    return ker / np.sum(ker)
