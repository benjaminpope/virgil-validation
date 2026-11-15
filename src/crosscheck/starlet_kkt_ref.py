r"""Independent NumPy/SciPy references for Batch 7 (scenes, starlet wavelets,
KKT optimality certificates, DoF-corrected discrepancy, L-curve corner curvature,
Laplace GP evidence, MYTHRA ensemble selection, and radio visibility differencing).

Nothing here imports ``virgil``, ``drpangloss``, or ``jaxbessel``. Every formula
is implemented directly from its published textbook or journal definition:

- Synthetic sky scenes (inclined modulated Gaussian ring, face-on Archimedean
  spiral with rounded arm caps, circular Gaussian blob), rotated sky-grid
  coordinates, elliptical restoring-beam convolution, and sensitivity-normalized
  residual dirty maps: standard interferometric sky geometry (North up, East
  left, position angle North through East; Bracewell 2000, *The Fourier
  Transform and Its Applications*, 3rd ed., McGraw-Hill).
- Isotropic undecimated B3-spline à-trous ("starlet") wavelet decomposition and
  exact synthesis: Starck, Murtagh & Fadili (2010), *Sparse Image and Signal
  Processing*, Cambridge University Press, §3.5.
- Short-baseline Gaussian second-moment FWHM and field-of-view rules: Bracewell
  (2000), $-\ln V^2 = \pi^2 \theta^2 \rho^2 / (2 \ln 2)$.
- Constrained and reparametrized KKT stationarity, projected-gradient displacement
  $x - P_{[l,u]}(x - g)$, and complementary-slackness certificates: Nocedal &
  Wright (2006), *Numerical Optimization*, 2nd ed., Springer, ch. 12 and §16.7
  (Eq. 16.71).
- Regularizer Hessians (Dirichlet-padded Total Squared Variation, Kuramochi et
  al. 2018, *ApJ* 858, 56; unnormalized Skilling entropy, Skilling 1989),
  degrees-of-freedom-corrected Morozov discrepancy $m(w) = \chi^2/s^2 - (n - G)$
  with $G = \operatorname{tr}[K(K + H_R)^{-1}]$ (Morozov 1966; Craven & Wahba
  1979), and exact Rademacher trace-estimator variance (Hutchinson 1989,
  *Commun. Stat. Simul. Comput.* 18, 1059; Bekas, Kokiopoulou & Saad 2007,
  *Appl. Numer. Math.* 57, 1214).
- Laplace log-evidence (including joint latent-plus-nuisance marginalization),
  posterior covariance, effective parameter count $\gamma = \sum_k s_k^2 / (1 + s_k^2)$,
  and self-consistent empirical-Bayes error scale $s^2 = \chi^2 / (N - \gamma(s))$:
  MacKay (1992, *Neural Comput.* 4, 415); classic MaxEnt balance (Gull 1989) and
  parametric L-curve curvature (Hansen & O'Leary 1993, *SIAM J. Sci. Comput.* 14, 1487).
- Randomized multi-family ensemble selection and non-compounding running-mean
  $\chi^2$ gate: Drevon et al. (2026, MYTHRA).
- Model-free radio visibility noise scale from non-overlapping time-adjacent
  baseline-channel-scan pairs and far-pair lag ratio: Thompson, Moran & Swenson
  (2017, *Interferometry and Synthesis in Radio Astronomy*, 3rd ed., Springer,
  ch. 6) and Cornwell & Wilkinson (1981, *MNRAS* 196, 1067).
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, ndimage, signal

MAS_TO_RAD = np.pi / (180.0 * 3600.0 * 1000.0)
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
B3_KERNEL_1D = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=float) / 16.0


# ---------------------------------------------------------------------------
# Sky-grid coordinates, synthetic scenes, beams, residual maps, and centroids
# ---------------------------------------------------------------------------


def sky_grid_mas(
    npix: int,
    pixel_scale_mas: float,
    dra: float = 0.0,
    ddec: float = 0.0,
    rotation_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sky coordinates ``(east_mas, north_mas)`` of an ``npix x npix`` pixel grid.

    Row 0 is the local "up" edge and column 0 is the local "left" edge, centred
    on index ``(npix - 1) / 2``. When ``rotation_deg == 0``, up points North
    (``+ddec``) and left points East (``+dra``). Turning the grid's "up" axis
    to position angle ``rotation_deg`` (North through East, counter-clockwise on
    the sky) maps local ``(x_left, y_up)`` to sky ``(dra, ddec)`` via

    .. math::
        \\Delta\\alpha = \\Delta\\alpha_0 + x_{\\text{loc}}\\cos\\alpha + y_{\\text{loc}}\\sin\\alpha,
        \\qquad
        \\Delta\\delta = \\Delta\\delta_0 - x_{\\text{loc}}\\sin\\alpha + y_{\\text{loc}}\\cos\\alpha.
    """
    offsets_1d = ((npix - 1) / 2.0 - np.arange(npix, dtype=float)) * float(pixel_scale_mas)
    x_loc = np.broadcast_to(offsets_1d[None, :], (npix, npix))
    y_loc = np.broadcast_to(offsets_1d[:, None], (npix, npix))
    alpha = np.deg2rad(float(rotation_deg))
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    east = float(dra) + x_loc * cos_a + y_loc * sin_a
    north = float(ddec) - x_loc * sin_a + y_loc * cos_a
    return east, north


def gaussian_blob_ref(
    npix: int,
    pixel_scale_mas: float,
    sigma_mas: float,
    dra: float = 0.0,
    ddec: float = 0.0,
) -> np.ndarray:
    """Unit-sum circular Gaussian blob centred at ``(dra, ddec)`` mas (East+, North+)."""
    east, north = sky_grid_mas(npix, pixel_scale_mas)
    r2 = (east - float(dra)) ** 2 + (north - float(ddec)) ** 2
    img = np.exp(-0.5 * r2 / (float(sigma_mas) ** 2))
    return img / np.sum(img)


def ring_ref(
    npix: int,
    pixel_scale_mas: float,
    radius_mas: float,
    width_mas: float,
    inc_deg: float = 0.0,
    pa_deg: float = 0.0,
    asymmetry: float = 0.0,
    asymmetry_pa_deg: float = 0.0,
) -> np.ndarray:
    """Unit-sum inclined Gaussian-profile ring with on-sky azimuthal modulation.

    - Semi-major axis at ``pa_deg`` (North through East), foreshortened by
      ``cos(inc_deg)`` along the minor axis.
    - Radial Gaussian profile ``exp(-(r_deproj - radius_mas)^2 / (2 width_mas^2))``.
    - On-sky azimuthal factor ``1 + asymmetry * cos(PA_sky - asymmetry_pa_deg)``
      where ``PA_sky = arctan2(east, north)``.
    """
    east, north = sky_grid_mas(npix, pixel_scale_mas)
    pa = np.deg2rad(float(pa_deg))
    along_major = east * np.sin(pa) + north * np.cos(pa)
    across_minor = east * np.cos(pa) - north * np.sin(pa)
    cos_i = np.cos(np.deg2rad(float(inc_deg)))
    r_deproj = np.hypot(along_major, across_minor / cos_i)
    radial = np.exp(-0.5 * ((r_deproj - float(radius_mas)) / float(width_mas)) ** 2)
    if asymmetry != 0.0:
        pa_sky = np.arctan2(east, north)
        asym_pa = np.deg2rad(float(asymmetry_pa_deg))
        mod = 1.0 + float(asymmetry) * np.cos(pa_sky - asym_pa)
    else:
        mod = 1.0
    img = radial * mod
    return img / np.sum(img)


def spiral_ref(
    npix: int,
    pixel_scale_mas: float,
    step_mas: float,
    width_mas: float,
    turns: float = 2.0,
    pa_deg: float = 0.0,
    fade_mas: float | None = None,
    *,
    distance_mode: str = "euclidean",
) -> np.ndarray:
    """Unit-sum face-on Archimedean spiral ``r = step_mas * theta / (2 pi)``.

    Winds counter-clockwise (North through East) from ``pa_deg`` over
    ``theta in [0, 2 * pi * turns]``, with Gaussian standard deviation
    ``width_mas`` (NOT FWHM), rounded arm-end caps, and optional radial decay
    ``exp(-r / fade_mas)``.

    Two physically sensible readings of the arm-distance specification are
    provided so tests can compare both against ``virgil.scenes.spiral``:

    - ``distance_mode="euclidean"`` (default): exact minimum Euclidean distance in
      the sky plane to the Archimedean curve ``(r(theta) sin psi, r(theta) cos psi)``,
      which inherently has uniform normal width ``width_mas`` everywhere along the
      spine and exact circular Euclidean caps at both ``theta = 0`` and
      ``theta = 2 * pi * turns``.
    - ``distance_mode="radial_cap"``: ray-radial cross-section distance
      ``|r - r(theta_k)|`` along each ray ``psi`` plus circular Euclidean caps at
      the endpoints ``theta = 0`` and ``theta = 2 * pi * turns`` (whose normal arm
      width narrows by ``1 / sqrt(1 + (step_mas / (2 pi r))^2)`` near the origin
      and converges to ``"euclidean"`` at large ``2 pi r / step_mas``).
    """
    east, north = sky_grid_mas(npix, pixel_scale_mas)
    r = np.hypot(east, north)
    theta_max = 2.0 * np.pi * float(turns)
    pa0 = np.deg2rad(float(pa_deg))

    if distance_mode == "euclidean":
        n_curve = max(2048, int(np.ceil(1024 * float(turns))))
        theta = np.linspace(0.0, theta_max, n_curve)
        r_c = float(step_mas) * theta / (2.0 * np.pi)
        psi_c = pa0 + theta
        cx = r_c * np.sin(psi_c)
        cy = r_c * np.cos(psi_c)
        dx = east[..., None] - cx[None, None, :]
        dy = north[..., None] - cy[None, None, :]
        dist = np.sqrt(np.min(dx * dx + dy * dy, axis=-1))
    elif distance_mode == "radial_cap":
        dpsi = np.mod(np.arctan2(east, north) - pa0, 2.0 * np.pi)
        r_end = float(step_mas) * float(turns)
        psi_end = pa0 + theta_max
        d_start = r
        d_end = np.hypot(east - r_end * np.sin(psi_end), north - r_end * np.cos(psi_end))
        dist = np.minimum(d_start, d_end)
        max_k = int(np.floor(float(turns))) + 1
        for k in range(max_k + 1):
            theta_k = dpsi + 2.0 * np.pi * k
            valid = theta_k <= theta_max
            r_arm = float(step_mas) * theta_k / (2.0 * np.pi)
            d_k = np.where(valid, np.abs(r - r_arm), np.inf)
            dist = np.minimum(dist, d_k)
    else:
        raise ValueError(f"Unknown distance_mode: {distance_mode!r}")

    img = np.exp(-0.5 * (dist / float(width_mas)) ** 2)
    if fade_mas is not None:
        img = img * np.exp(-r / float(fade_mas))
    return img / np.sum(img)


def sky_centroid_ref(
    brightness: np.ndarray,
    pixel_scale_mas: float,
    sigma_mas: float,
    dra: float = 0.0,
    ddec: float = 0.0,
    rotation_deg: float = 0.0,
) -> dict[str, np.ndarray | float]:
    """Flux-weighted sky centroid ``(c_dra, c_ddec)``, residuals, and Gaussian prior penalty."""
    b = np.asarray(brightness, dtype=float)
    b = b / np.sum(b)
    east, north = sky_grid_mas(
        b.shape[0], pixel_scale_mas, dra=dra, ddec=ddec, rotation_deg=rotation_deg
    )
    c_dra = float(np.sum(b * east))
    c_ddec = float(np.sum(b * north))
    residuals = np.array([c_dra / float(sigma_mas), c_ddec / float(sigma_mas)], dtype=float)
    value = 0.5 * float(np.sum(residuals**2))
    return {
        "centroid": np.array([c_dra, c_ddec], dtype=float),
        "residuals": residuals,
        "value": value,
    }


def convolve_elliptical_beam_ref(
    components: np.ndarray,
    pixel_scale_mas: float,
    major_mas: float,
    minor_mas: float,
    pa_deg: float,
    *,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """Convolve a pixel-grid image with a sky-frame elliptical Gaussian restoring beam.

    In a grid rotated by ``rotation_deg`` (North through East), a sky beam with
    major-axis position angle ``pa_deg`` sits at local grid position angle
    ``pa_deg - rotation_deg``. The discrete PSF kernel is evaluated on an odd
    grid wide enough to enclose ``>= 6 sigma_maj``, normalized to unit sum, and
    convolved with zero-padded boundary conditions.
    """
    comp = np.asarray(components, dtype=float)
    sig_maj = float(major_mas) / FWHM_PER_SIGMA
    sig_min = float(minor_mas) / FWHM_PER_SIGMA
    half = max(comp.shape[0], int(np.ceil(6.0 * sig_maj / float(pixel_scale_mas))))
    k_npix = 2 * half + 1
    # Kernel local coordinates: when rotation_deg == 0, sky_grid_mas gives local (x_left, y_up)
    x_left, y_up = sky_grid_mas(k_npix, pixel_scale_mas, rotation_deg=0.0)
    pa_local = np.deg2rad(float(pa_deg) - float(rotation_deg))
    along = x_left * np.sin(pa_local) + y_up * np.cos(pa_local)
    across = x_left * np.cos(pa_local) - y_up * np.sin(pa_local)
    psf = np.exp(-0.5 * ((along / sig_maj) ** 2 + (across / sig_min) ** 2))
    psf = psf / np.sum(psf)
    return signal.fftconvolve(comp, psf, mode="same")


def v2_point_residual_dirty_map_ref(
    u_m: np.ndarray,
    v_m: np.ndarray,
    wavel_m: np.ndarray | float,
    v2: np.ndarray,
    sigma_v2: np.ndarray,
    npix: int,
    pixel_scale_mas: float,
    *,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """First-principles sensitivity-normalized ``-grad(chi^2) / ||J e_p||_2`` residual map for ``V^2``.

    Around an unresolved unit-flux point source at the phase centre (``V_0 = 1``),
    adding infinitesimal flux ``eps`` at sky offset ``(dra_p, ddec_p)`` yields
    ``V(u, v; eps) = (1 + eps exp(-2 pi i (u dra_p + v ddec_p) / lambda)) / (1 + eps)``,
    so ``d(|V|^2)/d(eps)|_{eps=0} = 2 [cos(2 pi (u dra_p + v ddec_p) / lambda) - 1]``.
    With whitened residual ``r_k = (1 - V^2_k) / sigma_k`` and column
    ``J_{k,p} = (2 / sigma_k) [cos(phi_{k,p}) - 1]``, the gradient of ``chi^2 = ||r||_2^2``
    is ``g_p = 2 r^T J_{.,p}`` and the normalized residual map is ``-g_p / ||J_{.,p}||_2``
    (zero where ``||J_{.,p}||_2 == 0``, such as the phase-centre pixel).
    """
    u = np.asarray(u_m, dtype=float).ravel()
    v = np.asarray(v_m, dtype=float).ravel()
    lam = np.broadcast_to(np.asarray(wavel_m, dtype=float), u.shape).ravel()
    v2_arr = np.asarray(v2, dtype=float).ravel()
    err = np.asarray(sigma_v2, dtype=float).ravel()

    east, north = sky_grid_mas(npix, pixel_scale_mas, rotation_deg=rotation_deg)
    u_lam = (u / lam)[:, None, None]
    v_lam = (v / lam)[:, None, None]
    phase = 2.0 * np.pi * (u_lam * (east * MAS_TO_RAD)[None, :, :] + v_lam * (north * MAS_TO_RAD)[None, :, :])
    jac = (2.0 / err[:, None, None]) * (np.cos(phase) - 1.0)
    r0 = (1.0 - v2_arr) / err
    norms = np.linalg.norm(jac, axis=0)
    grad_chi2 = 2.0 * np.tensordot(r0, jac, axes=(0, 0))
    out = np.zeros((npix, npix), dtype=float)
    live = norms > 1e-12 * max(float(np.max(norms)), 1.0)
    out[live] = -grad_chi2[live] / norms[live]
    return out


# ---------------------------------------------------------------------------
# Isotropic undecimated B3-spline à-trous ("starlet") wavelet transform
# ---------------------------------------------------------------------------


def starlet_filter_1d(scale_index: int) -> np.ndarray:
    """1D à-trous B3-spline filter at dyadic scale ``scale_index >= 0`` (step ``2**j``).

    Starck, Murtagh & Fadili (2010, §3.5): ``h_0 = [1, 4, 6, 4, 1] / 16`` with
    ``2**j - 1`` zeros interleaved between adjacent taps.
    """
    if scale_index < 0:
        raise ValueError("scale_index must be non-negative")
    step = 1 << int(scale_index)
    filt = np.zeros(4 * step + 1, dtype=float)
    filt[::step] = B3_KERNEL_1D
    return filt


def starlet_ref(
    image: np.ndarray,
    scales: int = 4,
    *,
    mode: str = "constant",
) -> tuple[np.ndarray, np.ndarray]:
    """Undecimated 2D isotropic B3-spline starlet decomposition.

    Uses separable 1D à-trous convolution with zero-padding beyond image edges
    (``mode="constant", cval=0.0`` by default). Returns ``(details, coarse)``
    where ``details`` has shape ``(scales, ny, nx)`` (finest scale first) and
    ``details.sum(axis=0) + coarse == image`` identically for any boundary mode.
    """
    smooth = np.asarray(image, dtype=float)
    details = []
    for j in range(int(scales)):
        h = starlet_filter_1d(j)
        nxt = ndimage.convolve1d(smooth, h, axis=0, mode=mode, cval=0.0)
        nxt = ndimage.convolve1d(nxt, h, axis=1, mode=mode, cval=0.0)
        details.append(smooth - nxt)
        smooth = nxt
    return np.asarray(details, dtype=float), smooth


# ---------------------------------------------------------------------------
# Short-baseline second-moment FWHM & field sizing
# ---------------------------------------------------------------------------


def second_moment_fwhm_ref(
    rho_rad_inv: np.ndarray,
    v2: np.ndarray,
    sigma_v2: np.ndarray,
    v2_min: float = 0.3,
    *,
    fit_intercept: bool = False,
) -> float | None:
    """Short-baseline Gaussian FWHM (mas) from WLS of ``-ln V^2 = pi^2 theta^2 rho^2 / (2 ln 2)``.

    Bracewell (2000): a circular Gaussian of FWHM ``theta`` (radians) has
    ``V^2(rho) = exp(-pi^2 theta^2 rho^2 / (2 ln 2))``. Propagating Gaussian errors
    on ``V^2`` to ``y = -ln V^2`` gives ``sigma_y = sigma_{V^2} / V^2``, i.e. inverse
    standard-deviation weights ``w = V^2 / sigma_{V^2}``. Returns ``None`` when
    fewer than two samples satisfy ``v2 >= v2_min``, and ``0.0`` if the fitted
    slope is non-positive.
    """
    rho = np.asarray(rho_rad_inv, dtype=float).ravel()
    v2_arr = np.asarray(v2, dtype=float).ravel()
    err = np.asarray(sigma_v2, dtype=float).ravel()
    mask = (
        np.isfinite(rho)
        & np.isfinite(v2_arr)
        & np.isfinite(err)
        & (err > 0.0)
        & (v2_arr >= float(v2_min))
    )
    if int(np.sum(mask)) < 2:
        return None
    x = rho[mask] ** 2
    y = -np.log(v2_arr[mask])
    w = v2_arr[mask] / err[mask]
    if fit_intercept:
        design = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)
        slope = float(beta[1])
    else:
        denom = float(np.sum((w * x) ** 2))
        if denom <= 0.0:
            return 0.0
        slope = float(np.sum((w**2) * x * y) / denom)
    if slope <= 0.0:
        return 0.0
    theta_rad = np.sqrt(2.0 * np.log(2.0) * slope) / np.pi
    return float(theta_rad / MAS_TO_RAD)


def image_field_ref(
    b_over_lambda_min: float,
    b_over_lambda_max: float,
    fwhm_mas: float | None,
    *,
    field_fwhms: float = 6.0,
    inner_elements: float = 4.0,
    aperture_mas: float | None = None,
    largest_mas: float = 500.0,
) -> float:
    """Field size (mas): ``field_fwhms * fwhm_mas`` clipped to ``[floor, ceiling]`` (ceiling wins)."""
    res_element_mas = (1.0 / (2.0 * float(b_over_lambda_max))) / MAS_TO_RAD
    floor_mas = float(inner_elements) * res_element_mas
    fov_mas = min((1.0 / float(b_over_lambda_min)) / MAS_TO_RAD, float(largest_mas))
    ceiling_mas = fov_mas if aperture_mas is None else min(fov_mas, float(aperture_mas))
    if fwhm_mas is None:
        raw = ceiling_mas
    else:
        raw = float(field_fwhms) * float(fwhm_mas)
    if floor_mas > ceiling_mas:
        return float(ceiling_mas)
    return float(np.clip(raw, floor_mas, ceiling_mas))


# ---------------------------------------------------------------------------
# Constrained & reparametrized KKT optimality certificates
# ---------------------------------------------------------------------------


def kkt_simplex_metrics_ref(
    brightness: np.ndarray,
    grad_data_b: np.ndarray,
    grad_pen_b: np.ndarray | None = None,
    *,
    support: np.ndarray | None = None,
    flux: float = 1.0,
    free_flux: bool = False,
    weight: float | None = None,
    kkt_tol: float = 1e-3,
    peak_lit_frac: float = 1e-3,
    dark_frac: float = 1e-6,
    start_brightness: np.ndarray | None = None,
    start_grad_data_b: np.ndarray | None = None,
    start_grad_pen_b: np.ndarray | None = None,
) -> dict[str, float]:
    """Simplex (softmax ``Image``) KKT stationarity & complementary-slackness metrics.

    Nocedal & Wright (2006, ch. 12): minimizing ``J(b) = J_data(b) + J_pen(b)`` on
    ``{b >= 0, sum b = 1}`` has Lagrangian ``J(b) - mu (sum b_i - 1) - sum lambda_i b_i``
    with ``lambda_i >= 0`` and ``lambda_i b_i = 0``. Multiplying ``g_i - mu - lambda_i = 0``
    by ``b_i`` and summing gives ``mu = sum_i b_i g_i`` when image flux ``F`` is fixed
    (or ``mu = 0`` when total image flux ``F`` is itself a free parameter).

    - Centered simplex components are ``d_data = g_data - sum(b * g_data)`` and
      ``d_pen = g_pen - sum(b * g_pen)``, with simplex residual ``res_b = d_data + d_pen``
      and gradient scale ``scale = sqrt(mean(d_data[lit]^2 + d_pen[lit]^2))`` (replaced by
      ``max(scale, scale_start)`` when starting brightness and gradients are supplied).
    - Per-flux residual on lit pixels (``b_i > peak_lit_frac * max(b)``) is
      ``g_I - mu_I = res_b / flux``, normalized by ``weight / flux`` to give ``kkt_w``.
    - Relative stationarity is ``kkt_rel = rms(res_b[lit]) / ((0 if free_flux else |mu|) + scale)``.
    - ``want_grow`` is the fraction of support pixels that are dark (``b_i < dark_frac * max(b)``)
      and violate dual feasibility beyond the deadband ``res_b < -3 * scale * max(kkt_rel, kkt_tol)``.
    """
    b = np.asarray(brightness, dtype=float)
    gd = np.asarray(grad_data_b, dtype=float)
    gp = np.zeros_like(gd) if grad_pen_b is None else np.asarray(grad_pen_b, dtype=float)
    sup = np.ones_like(b, dtype=bool) if support is None else np.asarray(support, dtype=bool)
    f_scale = float(flux)
    if f_scale <= 0.0:
        raise ValueError("flux must be positive")

    mu_d = float(np.sum(b[sup] * gd[sup]))
    mu_p = float(np.sum(b[sup] * gp[sup]))
    d_data = gd - mu_d
    d_pen = gp - mu_p
    mu = mu_d + mu_p

    peak = float(np.max(b[sup]))
    lit = sup & (b > float(peak_lit_frac) * peak)
    dark = sup & (b < float(dark_frac) * peak)

    res_b = d_data + d_pen
    res_flux = res_b / f_scale
    rms_res_b = float(np.sqrt(np.mean(res_b[lit] ** 2))) if np.any(lit) else 0.0
    rms_res_flux = float(np.sqrt(np.mean(res_flux[lit] ** 2))) if np.any(lit) else 0.0
    scale = (
        float(np.sqrt(np.mean(d_data[lit] ** 2 + d_pen[lit] ** 2))) if np.any(lit) else 0.0
    )
    if start_brightness is not None and start_grad_data_b is not None:
        bs = np.asarray(start_brightness, dtype=float)
        gds = np.asarray(start_grad_data_b, dtype=float)
        gps = (
            np.zeros_like(gds)
            if start_grad_pen_b is None
            else np.asarray(start_grad_pen_b, dtype=float)
        )
        dds = gds - float(np.sum(bs[sup] * gds[sup]))
        dps = gps - float(np.sum(bs[sup] * gps[sup]))
        peaks = float(np.max(bs[sup])) if np.any(sup) else 0.0
        lits = sup & (bs > float(peak_lit_frac) * peaks)
        scale_start = (
            float(np.sqrt(np.mean(dds[lits] ** 2 + dps[lits] ** 2))) if np.any(lits) else 0.0
        )
        scale = max(scale, scale_start)

    denom = (0.0 if free_flux else abs(mu)) + scale
    kkt_rel = rms_res_b / denom if denom > 0.0 else 0.0
    kkt_w = (
        rms_res_flux / (float(weight) / f_scale)
        if (weight is not None and weight > 0.0)
        else np.nan
    )
    n_sup = int(np.sum(sup))
    thresh = -3.0 * scale * max(float(kkt_rel), float(kkt_tol))
    want_grow = float(np.sum(res_b[dark] < thresh) / max(n_sup, 1)) if np.any(dark) else 0.0
    return {
        "mu": float(mu),
        "scale": float(scale),
        "kkt_rel": float(kkt_rel),
        "kkt_w": float(kkt_w),
        "want_grow": float(want_grow),
    }


def kkt_sqrt_metrics_ref(
    sqrt_pixels: np.ndarray,
    grad_data_f: np.ndarray,
    grad_pen_f: np.ndarray | None = None,
    *,
    support: np.ndarray | None = None,
    weight: float | None = None,
    reference_flux: float | None = None,
    kkt_tol: float = 1e-3,
    peak_lit_frac: float = 1e-3,
    dark_frac: float = 1e-6,
    start_sqrt_pixels: np.ndarray | None = None,
    start_grad_data_f: np.ndarray | None = None,
    start_grad_pen_f: np.ndarray | None = None,
) -> dict[str, float]:
    """Square-root reparameterized (``f_i = x_i^2``, ``LinearImage("sqrt")``) KKT metrics.

    Nocedal & Wright (2006, ch. 12): because ``f_i = x_i^2 >= 0`` is unconstrained in ``x``
    with no simplex sum constraint, the equality multiplier vanishes (``mu = 0``),
    ``dJ / dx_i = 2 x_i (dJ / df_i)`` vanishes identically at ``x_i = 0`` (excluding exact
    zeros from ``want_grow``), and ``scale`` on lit pixels (``f_i > peak_lit_frac * max(f)``)
    is ``max(scale_cur, scale_start)`` when starting values are given.
    """
    x = np.asarray(sqrt_pixels, dtype=float)
    f = x * x
    gd = np.asarray(grad_data_f, dtype=float)
    gp = np.zeros_like(gd) if grad_pen_f is None else np.asarray(grad_pen_f, dtype=float)
    g = gd + gp
    sup = np.ones_like(f, dtype=bool) if support is None else np.asarray(support, dtype=bool)
    f0 = float(np.sum(f[sup])) if reference_flux is None else float(reference_flux)
    peak = float(np.max(f[sup])) if np.any(sup) else 0.0
    lit = sup & (f > float(peak_lit_frac) * peak)
    dark = sup & (f < float(dark_frac) * peak) & (x != 0.0)

    rms_g = float(np.sqrt(np.mean(g[lit] ** 2))) if np.any(lit) else 0.0
    scale = float(np.sqrt(np.mean(gd[lit] ** 2 + gp[lit] ** 2))) if np.any(lit) else 0.0
    if start_sqrt_pixels is not None and start_grad_data_f is not None:
        xs = np.asarray(start_sqrt_pixels, dtype=float)
        fs = xs * xs
        gds = np.asarray(start_grad_data_f, dtype=float)
        gps = (
            np.zeros_like(gds)
            if start_grad_pen_f is None
            else np.asarray(start_grad_pen_f, dtype=float)
        )
        peaks = float(np.max(fs[sup])) if np.any(sup) else 0.0
        lits = sup & (fs > float(peak_lit_frac) * peaks)
        scale_start = (
            float(np.sqrt(np.mean(gds[lits] ** 2 + gps[lits] ** 2))) if np.any(lits) else 0.0
        )
        scale = max(scale, scale_start)

    kkt_rel = rms_g / scale if scale > 0.0 else 0.0
    kkt_w = (
        rms_g / (float(weight) / f0)
        if (weight is not None and weight > 0.0 and f0 > 0.0)
        else np.nan
    )
    n_sup = int(np.sum(sup))
    thresh = -3.0 * scale * max(float(kkt_rel), float(kkt_tol))
    want_grow = float(np.sum(g[dark] < thresh) / max(n_sup, 1)) if np.any(dark) else 0.0
    return {
        "scale": float(scale),
        "kkt_rel": float(kkt_rel),
        "kkt_w": float(kkt_w),
        "want_grow": float(want_grow),
    }


def kkt_nonneg_linear_metrics_ref(
    fluxes: np.ndarray,
    grad_data: np.ndarray,
    grad_pen: np.ndarray | None = None,
    *,
    support: np.ndarray | None = None,
    weight: float | None = None,
    reference_flux: float | None = None,
    kkt_tol: float = 1e-3,
    peak_lit_frac: float = 1e-3,
    zero_tol: float = 0.0,
    lower: float = 0.0,
    upper: float = np.inf,
    start_grad_total: np.ndarray | None = None,
) -> dict[str, float]:
    """Box-constrained ``lower <= f <= upper`` KKT metrics (Nocedal & Wright 2006, §16.7 Eq. 16.71).

    For ``min_{l <= f <= u} J(f)`` with ``g = g_data + g_pen``:
    - Projected-gradient displacement on support is ``pg = f - clip(f - g, lower, upper)``
      (for ``[0, inf)``, ``pg_i = min(g_i, f_i)``, which equals ``min(g_i, 0)`` at ``f_i = 0``
      and equals ``g_i`` whenever a unit gradient step stays feasible, ``f_i - g_i >= 0``).
    - Reference infinity-norm scale on support is ``max(||g_data||_inf, ||g_pen||_inf)`` when
      ``start_grad_total`` is omitted, or ``||g_start||_inf`` when ``start_grad_total`` is
      supplied, giving ``pg_rel = ||pg||_inf / top``.
    - ``kkt_w`` is the RMS of ``pg`` over active-or-lit pixels
      (``f_i > peak_lit_frac * max(f)`` or ``f_i <= zero_tol`` with ``g_i < 0``) divided by
      ``weight / F_0`` with ``F_0 = sum(f)`` (or ``reference_flux`` if explicitly given).
    - ``zero_violation`` is the fraction of zero pixels (``f_i <= zero_tol`` in support)
      whose gradient violates dual feasibility beyond the deadband ``g_i < -kkt_tol * top``.
    """
    f = np.asarray(fluxes, dtype=float)
    gd = np.asarray(grad_data, dtype=float)
    gp = np.zeros_like(gd) if grad_pen is None else np.asarray(grad_pen, dtype=float)
    g = gd + gp
    sup = np.ones_like(f, dtype=bool) if support is None else np.asarray(support, dtype=bool)
    f0 = float(np.sum(f[sup])) if reference_flux is None else float(reference_flux)
    peak = float(np.max(f[sup])) if np.any(sup) else 0.0
    at_zero = sup & (f <= float(zero_tol))
    pg = np.where(sup, f - np.clip(f - g, float(lower), float(upper)), 0.0)
    active_or_lit = sup & ((f > float(peak_lit_frac) * peak) | (at_zero & (g < 0.0)))
    rms_pg = float(np.sqrt(np.mean(pg[active_or_lit] ** 2))) if np.any(active_or_lit) else 0.0
    kkt_w = (
        rms_pg / (float(weight) / f0)
        if (weight is not None and weight > 0.0 and f0 > 0.0)
        else np.nan
    )
    if start_grad_total is not None:
        gs = np.asarray(start_grad_total, dtype=float)
        top = float(np.max(np.abs(gs[sup]))) if np.any(sup) else 0.0
    else:
        top = (
            max(float(np.max(np.abs(gd[sup]))), float(np.max(np.abs(gp[sup]))))
            if np.any(sup)
            else 0.0
        )
    pg_inf = float(np.max(np.abs(pg[sup]))) if np.any(sup) else 0.0
    pg_rel = pg_inf / top if top > 0.0 else 0.0
    n_zero = int(np.sum(at_zero))
    zero_violation = (
        float(np.sum(g[at_zero] < -float(kkt_tol) * top) / n_zero) if n_zero > 0 else 0.0
    )
    return {
        "pg_inf": pg_inf,
        "pg_rel": float(pg_rel),
        "rms_pg": rms_pg,
        "kkt_w": float(kkt_w),
        "zero_violation": zero_violation,
    }


def kkt_nuisance_grad_rel_ref(
    grad_cur: np.ndarray | float,
    grad_start: np.ndarray | float,
    *,
    value_cur: np.ndarray | float | None = None,
    value_start: np.ndarray | float | None = None,
    lower: np.ndarray | float = -np.inf,
    upper: np.ndarray | float = np.inf,
) -> dict[str, float]:
    """Projected-gradient infinity-norm ratio ``grad_rel = ||pg_cur||_inf / ||pg_start||_inf``
    for non-image parameters in their fitted coordinate (Nocedal & Wright 2006, §16.7).
    """
    gc = np.asarray(grad_cur, dtype=float)
    gs = np.asarray(grad_start, dtype=float)
    if value_cur is not None:
        vc_arr = np.asarray(value_cur, dtype=float)
        pg_cur = vc_arr - np.clip(vc_arr - gc, lower, upper)
    else:
        pg_cur = gc
    if value_start is not None:
        vs_arr = np.asarray(value_start, dtype=float)
        pg_start = vs_arr - np.clip(vs_arr - gs, lower, upper)
    else:
        pg_start = gs
    norm_cur = float(np.max(np.abs(pg_cur)))
    norm_start = float(np.max(np.abs(pg_start)))
    grad_rel = norm_cur / norm_start if norm_start > 0.0 else np.nan
    return {
        "grad_norm": norm_cur,
        "start_grad_norm": norm_start,
        "grad_rel": float(grad_rel),
    }


def maxent_band_limit_ref(
    fluxes: np.ndarray,
    u_m: np.ndarray,
    v_m: np.ndarray,
    wavel_m: np.ndarray | float,
    pixel_scale_mas: float,
) -> float:
    """Windowed super-cutoff Fourier power ratio of ``ln b`` over its hard low-pass.

    Skilling & Bryan (1984, *MNRAS* 211, 111): at a first-order MaxEnt stationary point
    ``w ln(p / m) = -F_0 (dJ_data / df)``, the log-brightness ``ln b`` is a linear
    combination of measured Fourier modes ``|q| <= q_max = max(hypot(u, v) / wavel)``
    (or ``2 q_max`` through squared visibilities). Comparing the 2D Hann-windowed high-frequency
    power of ``ln b`` strictly above ``q_max`` to that of its hard low-pass projection onto
    ``|q| <= q_max`` gives ``1.0`` identically for any band-limited ``ln b`` and grows well
    above unity when unconstrained super-Nyquist structure (such as a checkerboard ripple) is
    present.
    """
    arr = np.asarray(fluxes, dtype=float)
    b = arr / np.sum(arr)
    lnb = np.log(np.clip(b, 1e-300, None)) - float(np.mean(np.log(np.clip(b, 1e-300, None))))
    ny, nx = lnb.shape
    q_max_cyc_per_mas = float(
        np.max(np.hypot(np.asarray(u_m, dtype=float), np.asarray(v_m, dtype=float)) / np.asarray(wavel_m, dtype=float))
        * MAS_TO_RAD
    )
    fx = np.fft.fftfreq(nx, d=float(pixel_scale_mas))
    fy = np.fft.fftfreq(ny, d=float(pixel_scale_mas))
    qr = np.hypot(fx[None, :], fy[:, None])
    beyond = qr > q_max_cyc_per_mas
    if not np.any(beyond):
        return np.nan
    wx = np.hanning(nx)
    wy = np.hanning(ny)
    w2 = np.outer(wy, wx)
    lnb_lp = np.fft.ifft2(np.where(~beyond, np.fft.fft2(lnb), 0.0)).real
    pow_hi = float(np.sum(np.abs(np.fft.fft2(w2 * lnb))[beyond] ** 2))
    pow_lp = float(np.sum(np.abs(np.fft.fft2(w2 * lnb_lp))[beyond] ** 2))
    return pow_hi / pow_lp if pow_lp > 0.0 else (1.0 if pow_hi == 0.0 else np.inf)


# ---------------------------------------------------------------------------
# Regularizer Hessians, DoF-corrected discrepancy & Hutchinson trace variance
# ---------------------------------------------------------------------------


def tsv_dirichlet_hessian_ref(
    npix: int,
    weight: float,
    reference_flux: float,
    *,
    pixel_scale_mas: float = 1.0,
    beam_mas: float = 1.0,
    free_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Exact Hessian ``d^2 R / df^2`` of zero-padded Total Squared Variation on free pixels.

    Kuramochi et al. (2018, *ApJ* 858, 56): with ``p = f / reference_flux`` and a
    homogeneous Dirichlet zero ring outside the ``npix x npix`` grid,
    ``R(f) = w * (beam_mas / pixel_scale_mas)^4 * sum [(Delta_x p)^2 + (Delta_y p)^2]``,
    so ``H_R = (2 * w * scale / F_0^2) * (-L_Dirichlet)`` where ``L_Dirichlet`` is
    the 5-point discrete Laplacian.
    """
    scale = (float(beam_mas) / float(pixel_scale_mas)) ** 4 if beam_mas != pixel_scale_mas else 1.0
    n2 = npix * npix
    lap = np.zeros((n2, n2), dtype=float)
    for r in range(npix):
        for c in range(npix):
            idx = r * npix + c
            lap[idx, idx] = 4.0
            if r > 0:
                lap[idx, (r - 1) * npix + c] = -1.0
            if r + 1 < npix:
                lap[idx, (r + 1) * npix + c] = -1.0
            if c > 0:
                lap[idx, r * npix + (c - 1)] = -1.0
            if c + 1 < npix:
                lap[idx, r * npix + (c + 1)] = -1.0
    hess = (2.0 * float(weight) * scale / (float(reference_flux) ** 2)) * lap
    if free_mask is not None:
        m = np.asarray(free_mask, dtype=bool).ravel()
        hess = hess[np.ix_(m, m)]
    return hess


def maxent_hessian_ref(
    fluxes: np.ndarray,
    weight: float,
    reference_flux: float,
    *,
    free_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Exact diagonal Hessian ``d^2 R / df^2 = diag(w / (F_0 * f_i))`` of Skilling MaxEntropy.

    Skilling (1989): ``R(f) = w sum_i [p_i ln(p_i / m_i) - (p_i - m_i)]`` with
    ``p_i = f_i / F_0`` has ``dR/df_i = (w / F_0) ln(p_i / m_i)`` and
    ``d^2 R / df_i^2 = w / (F_0 * f_i)`` independent of ``m_i``.
    """
    f = np.asarray(fluxes, dtype=float).ravel()
    if free_mask is not None:
        f = f[np.asarray(free_mask, dtype=bool).ravel()]
    return np.diag(float(weight) / (float(reference_flux) * f))


def ridge_dof_and_hutchinson_se_ref(
    k_mat: np.ndarray,
    h_reg: np.ndarray,
    nprobe: int = 16,
) -> dict[str, float]:
    """Closed-form effective DoF ``G = Tr[K (K + H_R)^{-1}]`` and exact Rademacher SE.

    Morozov (1966) / Craven & Wahba (1979): ``G = Tr[K (K + H_R)^{-1}]``.
    Hutchinson (1989) & Bekas, Kokiopoulou & Saad (2007): for ``z in {-1, +1}^d``
    with independent symmetric Bernoulli components, ``z^T A z = z^T S z`` where
    ``S = (A + A^T) / 2``, ``E[z^T A z] = Tr(A)``, and
    ``Var(z^T A z) = 2 * (Tr(S^2) - sum_i S_{ii}^2)``.
    """
    k = np.asarray(k_mat, dtype=float)
    hr = np.asarray(h_reg, dtype=float)
    a_mat = linalg.solve(k + hr, k, assume_a="pos")
    g_exact = float(np.trace(a_mat))
    s_mat = 0.5 * (a_mat + a_mat.T)
    var_single = 2.0 * float(np.sum(s_mat * s_mat) - np.sum(np.diag(s_mat) ** 2))
    var_single = max(0.0, var_single)
    se_rademacher = float(np.sqrt(var_single / float(nprobe)))
    return {
        "g": g_exact,
        "se_rademacher": se_rademacher,
        "var_single": var_single,
    }


def hutchinson_rademacher_estimate(
    k_mat: np.ndarray,
    h_reg: np.ndarray,
    nprobe: int = 16,
    seed: int = 0,
) -> tuple[float, float]:
    """Direct Monte Carlo Rademacher estimate ``(mean, sample_se)`` of ``Tr[(K + H_R)^{-1} K]``."""
    rng = np.random.default_rng(seed)
    d = k_mat.shape[0]
    z = rng.choice([-1.0, 1.0], size=(d, nprobe))
    kz = k_mat @ z
    sol = linalg.solve(k_mat + h_reg, kz, assume_a="pos")
    probes = np.sum(z * sol, axis=0)
    mean = float(np.mean(probes))
    se = float(np.std(probes, ddof=1) / np.sqrt(nprobe)) if nprobe > 1 else 0.0
    return mean, se


def skilling_kl_ref(
    fluxes: np.ndarray,
    *,
    reference_flux: float | None = None,
    default_prob: np.ndarray | None = None,
    support: np.ndarray | None = None,
) -> float:
    """Generalized Kullback–Leibler divergence ``sum [p_i ln(p_i / m_i) - (p_i - m_i)]``
    (Skilling 1989) for positive measure ``p_i = f_i / F_0`` against default ``m_i``.
    """
    f = np.asarray(fluxes, dtype=float)
    sup = np.ones_like(f, dtype=bool) if support is None else np.asarray(support, dtype=bool)
    f_sup = f[sup]
    f0 = float(np.sum(f_sup)) if reference_flux is None else float(reference_flux)
    p = f_sup / f0
    m = (
        np.full_like(p, 1.0 / max(p.size, 1))
        if default_prob is None
        else np.asarray(default_prob, dtype=float)[sup]
    )
    pos = p > 0.0
    term = np.where(pos, p * np.log(np.where(pos, p, 1.0) / m) - (p - m), m)
    return float(np.sum(term))


def dof_weight_band_ref(
    weights: np.ndarray,
    chi2: np.ndarray,
    g_sweep: np.ndarray,
    n_real: float,
    *,
    noise_scale2: float = 1.0,
    noise_scale_sigma: float = 0.05,
) -> tuple[float | None, float | None]:
    """Interpolate ``±2 * noise_scale_sigma`` Morozov discrepancy crossings ``(w_plus, w_minus)``
    linearly in ``ln w`` along the sweep (Morozov 1966; Craven & Wahba 1979).

    At shifted noise variance ``s_pm^2 = noise_scale2 +/- 2 * noise_scale_sigma``, finds
    all sign-change crossings of ``m(w; s_pm^2) = chi2(w) / s_pm^2 - (n_real - G(w))``
    linearly in ``ln w`` and selects the largest-weight crossing (or ``None`` if ``s_pm^2 <= 0``
    or no crossing exists). Because larger ``s^2`` lowers ``chi2 / s^2``, ``w_plus > w_minus``
    on a monotone single-crossing sweep.
    """
    w = np.asarray(weights, dtype=float).ravel()
    c2 = np.asarray(chi2, dtype=float).ravel()
    g = np.asarray(g_sweep, dtype=float).ravel()
    order = np.argsort(w)[::-1]  # largest weight first
    lw = np.log(w[order])
    c2 = c2[order]
    g = g[order]

    def _strongest_root(s2: float) -> float | None:
        if s2 <= 0.0:
            return None
        m = c2 / s2 - (float(n_real) - g)
        roots: list[float] = []
        for i in range(m.size - 1):
            m0, m1 = float(m[i]), float(m[i + 1])
            if not (np.isfinite(m0) and np.isfinite(m1)):
                continue
            if m0 == 0.0:
                roots.append(float(np.exp(lw[i])))
            elif (m0 > 0.0 and m1 <= 0.0) or (m0 < 0.0 and m1 >= 0.0):
                frac = -m0 / (m1 - m0)
                roots.append(float(np.exp(lw[i] + frac * (lw[i + 1] - lw[i]))))
        return max(roots) if roots else None

    w_plus = _strongest_root(float(noise_scale2) + 2.0 * float(noise_scale_sigma))
    w_minus = _strongest_root(float(noise_scale2) - 2.0 * float(noise_scale_sigma))
    return w_plus, w_minus


def short_baseline_v2_residual_ref(
    u_m: np.ndarray,
    v_m: np.ndarray,
    wavel_m: np.ndarray | float,
    vis2_obs: np.ndarray,
    sigma_vis2: np.ndarray,
    model_cvis: np.ndarray,
    *,
    quantile: float = 0.25,
) -> tuple[float, float]:
    """Unweighted mean and unit-normal standard error ``1 / sqrt(n_short)`` of shortest-quantile
    whitened ``V^2`` residuals ``(|V_model|^2 - V^2_obs) / sigma_{V^2}`` (Ireland, Monnier & Thureau 2006).
    """
    q = np.hypot(np.asarray(u_m, dtype=float), np.asarray(v_m, dtype=float)) / np.asarray(
        wavel_m, dtype=float
    )
    q_cut = float(np.quantile(q, float(quantile)))
    mask = q <= q_cut
    n_short = int(np.sum(mask))
    if n_short == 0:
        raise ValueError("no baselines within quantile")
    v2_mod = np.abs(np.asarray(model_cvis, dtype=complex)) ** 2
    r = (v2_mod - np.asarray(vis2_obs, dtype=float)) / np.asarray(sigma_vis2, dtype=float)
    return float(np.mean(r[mask])), float(1.0 / np.sqrt(n_short))


def lcurve_corner_ref(
    weights: np.ndarray,
    chi2: np.ndarray,
    penalty: np.ndarray,
) -> float:
    """Hansen & O'Leary (1993) interior L-curve corner maximizing parametric log-log curvature.

    Filters out non-finite ``(chi2, penalty)`` entries, orders points by ascending
    ``t = ln w`` (so ``x = ln chi2`` increases with ``t`` and ``y = ln penalty``
    decreases with ``t``, giving positive ``x' y'' - y' x''`` at the convex L-curve
    elbow), evaluates first and second derivatives on each interior 3-point Lagrange
    stencil ``(t_{i-1}, t_i, t_{i+1})``, and returns the interior weight ``w_i``
    (``1 <= i <= m-2``) maximizing signed curvature
    ``(x' y'' - y' x'') / (x'^2 + y'^2)^{3/2}``.
    """
    w = np.asarray(weights, dtype=float).ravel()
    c2 = np.asarray(chi2, dtype=float).ravel()
    pen = np.asarray(penalty, dtype=float).ravel()
    ok = np.isfinite(w) & (w > 0.0) & np.isfinite(c2) & (c2 > 0.0) & np.isfinite(pen) & (pen > 0.0)
    w, c2, pen = w[ok], c2[ok], pen[ok]
    if w.size < 3:
        raise ValueError("lcurve_corner_ref requires at least 3 finite positive sweep points")
    order = np.argsort(np.log(w))
    w, c2, pen = w[order], c2[order], pen[order]
    t = np.log(w)
    lx = np.log(c2)
    ly = np.log(pen)
    h0 = t[1:-1] - t[:-2]
    h1 = t[2:] - t[1:-1]

    def _d1(f: np.ndarray) -> np.ndarray:
        return (
            (-h1 / (h0 * (h0 + h1))) * f[:-2]
            + ((h1 - h0) / (h0 * h1)) * f[1:-1]
            + (h0 / (h1 * (h0 + h1))) * f[2:]
        )

    def _d2(f: np.ndarray) -> np.ndarray:
        return 2.0 * (
            f[:-2] / (h0 * (h0 + h1))
            - f[1:-1] / (h0 * h1)
            + f[2:] / (h1 * (h0 + h1))
        )

    dx1 = _d1(lx)
    dy1 = _d1(ly)
    dx2 = _d2(lx)
    dy2 = _d2(ly)
    denom = np.power(dx1 * dx1 + dy1 * dy1, 1.5)
    kappa = (dx1 * dy2 - dy1 * dx2) / denom
    best_interior = 1 + int(np.argmax(kappa))
    return float(w[best_interior])


# ---------------------------------------------------------------------------
# MacKay (1992) Laplace evidence, posterior covariance, gamma, & L-curve rules
# ---------------------------------------------------------------------------


def mackay_laplace_gp_ref(
    residuals: np.ndarray,
    log_norm: float,
    z_map: np.ndarray,
    jac_z: np.ndarray,
    *,
    n_independent: int | None = None,
    extra_jac: np.ndarray | None = None,
    extra_log_prior: float = 0.0,
    observable_blocks: dict[str, tuple[slice, int]] | None = None,
) -> dict[str, np.ndarray | float | dict[str, float]]:
    """MacKay (1992) Laplace GP quantities at MAP ``z_map`` with whitened-residual Jacobian ``J``.

    - ``chi2 = sum(r^2)``
    - ``log_evidence``: for pure standard-normal latents ``z ~ N(0, I_d)``,
      ``-0.5 * chi2 - log_norm - 0.5 * ||z_map||^2 - 0.5 * ln det(I_d + J_z^T J_z)``.
      When ``m`` extra non-latent parameters (such as ``ln flux`` under a ``LogUniform``
      prior) are marginalized jointly via ``extra_jac`` (shape ``(n_rows, m)``) and
      local log-prior density ``extra_log_prior``, the joint Laplace integral adds
      ``extra_log_prior + 0.5 * m * ln(2 pi)`` and replaces ``I_d + J_z^T J_z`` with
      ``diag(1_d, 0_m) + J_joint^T J_joint`` (equivalently the Schur complement of the
      ``(z, theta)`` Gauss-Newton precision).
    - ``cov = (I_d + J_z^T J_z)^{-1}``
    - ``gamma = sum_k s_k^2 / (1 + s_k^2)`` over singular values ``s_k`` of ``J_z``
    - Self-consistent fixed-point empirical-Bayes error scale ``s_fixed_point`` solving
      ``s^2 = chi2 / (N_ind - sum_k (s_k / s)^2 / (1 + (s_k / s)^2))``, plus ``s_plugin``
      (the unrescaled one-step approximation freezing ``s = 1`` inside ``gamma``) and,
      when ``observable_blocks`` is provided as ``{name: (row_slice, n_ind_block)}``,
      coupled per-observable fixed-point scales ``s_by_observable``.
    """
    r = np.asarray(residuals, dtype=float).ravel()
    z = np.asarray(z_map, dtype=float).ravel()
    j = np.asarray(jac_z, dtype=float)
    chi2 = float(np.sum(r * r))
    n_data = int(r.size) if n_independent is None else int(n_independent)
    d = z.size
    jtj = j.T @ j
    eye = np.eye(d, dtype=float)
    post_prec = eye + jtj
    chol = linalg.cholesky(post_prec, lower=True)
    cov = linalg.cho_solve((chol, True), eye)

    if extra_jac is None:
        logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
        log_ev = -0.5 * chi2 - float(log_norm) - 0.5 * float(np.sum(z * z)) - 0.5 * logdet
    else:
        j_ext = np.asarray(extra_jac, dtype=float)
        if j_ext.ndim == 1:
            j_ext = j_ext[:, None]
        m_ext = j_ext.shape[1]
        j_joint = np.hstack([j, j_ext])
        prior_diag = np.concatenate([np.ones(d, dtype=float), np.zeros(m_ext, dtype=float)])
        joint_prec = np.diag(prior_diag) + j_joint.T @ j_joint
        chol_joint = linalg.cholesky(joint_prec, lower=True)
        logdet_joint = 2.0 * float(np.sum(np.log(np.diag(chol_joint))))
        log_ev = (
            -0.5 * chi2
            - float(log_norm)
            - 0.5 * float(np.sum(z * z))
            + float(extra_log_prior)
            + 0.5 * float(m_ext) * float(np.log(2.0 * np.pi))
            - 0.5 * logdet_joint
        )

    svals = linalg.svdvals(j)
    gamma = float(np.sum((svals**2) / (1.0 + svals**2)))
    s_plugin = float(np.sqrt(chi2 / max(n_data - gamma, 1e-12)))

    s_fp = s_plugin
    for _ in range(200):
        gamma_s = float(np.sum(((svals / s_fp) ** 2) / (1.0 + (svals / s_fp) ** 2)))
        s_next = float(np.sqrt(chi2 / max(n_data - gamma_s, 1e-12)))
        if abs(s_next - s_fp) <= 1e-13 * max(1.0, s_fp):
            s_fp = s_next
            break
        s_fp = s_next

    out: dict[str, np.ndarray | float | dict[str, float]] = {
        "chi2": chi2,
        "log_evidence": float(log_ev),
        "cov": cov,
        "gamma": gamma,
        "s_plugin": s_plugin,
        "s_fixed_point": s_fp,
    }

    if observable_blocks is not None:
        names = list(observable_blocks.keys())
        chi2_b = {k: float(np.sum(r[sl] ** 2)) for k, (sl, _) in observable_blocks.items()}
        k_raw = {k: j[sl].T @ j[sl] for k, (sl, _) in observable_blocks.items()}
        n_b = {k: float(nb) for k, (_, nb) in observable_blocks.items()}
        scales = {k: 1.0 for k in names}
        for _ in range(300):
            scaled_k = {k: k_raw[k] / (scales[k] ** 2) for k in names}
            prec = eye + sum(scaled_k.values())
            chol_b = linalg.cholesky(prec, lower=True)
            cov_b = linalg.cho_solve((chol_b, True), eye)
            nxt = {}
            max_step = 0.0
            for k in names:
                gamma_k = float(np.sum(cov_b * scaled_k[k]))
                nxt[k] = float(np.sqrt(chi2_b[k] / max(n_b[k] - gamma_k, 1e-12)))
                max_step = max(max_step, abs(nxt[k] - scales[k]))
            scales = nxt
            if max_step <= 1e-13:
                break
        out["s_by_observable"] = scales

    return out


def classic_maxent_weight_ref(
    weights: np.ndarray,
    penalties_unweighted: np.ndarray,
    brightnesses: list[np.ndarray],
    jacobians_b: list[np.ndarray],
) -> float | None:
    """Gull (1989) & Skilling (1989) classic MaxEnt weight ``2 w KL = sum_k lam_k / (lam_k + w)``.

    Here ``penalty`` is unweighted KL divergence ``-S >= 0``, so ``-2 w S = 2 w * penalty``.
    ``lam_k`` are the eigenvalues of ``diag(sqrt(b)) J_b^T J_b diag(sqrt(b))``.
    Linearly interpolates the root in ``ln w`` across the sweep.
    """
    w_arr = np.asarray(weights, dtype=float)
    pen_arr = np.asarray(penalties_unweighted, dtype=float)
    diff = np.empty_like(w_arr)
    for idx, (w, pen, b, jb) in enumerate(zip(w_arr, pen_arr, brightnesses, jacobians_b)):
        rt_b = np.sqrt(np.clip(np.asarray(b, dtype=float).ravel(), 0.0, None))
        jb_scaled = np.asarray(jb, dtype=float) * rt_b[None, :]
        eigvals = np.linalg.eigvalsh(jb_scaled.T @ jb_scaled)
        eigvals = np.clip(eigvals, 0.0, None)
        n_good = float(np.sum(eigvals / (eigvals + w)))
        diff[idx] = 2.0 * w * pen - n_good
    for i in range(len(w_arr) - 1):
        if diff[i] == 0.0:
            return float(w_arr[i])
        if diff[i] * diff[i + 1] < 0.0:
            frac = -diff[i] / (diff[i + 1] - diff[i])
            ln_w = np.log(w_arr[i]) + frac * (np.log(w_arr[i + 1]) - np.log(w_arr[i]))
            return float(np.exp(ln_w))
    return None


# ---------------------------------------------------------------------------
# MYTHRA ensemble selection (Drevon et al. 2026)
# ---------------------------------------------------------------------------


def mythra_filter_indices_ref(
    chi2_red_matrix: np.ndarray,
    in_window: np.ndarray,
    dataset_sizes: np.ndarray,
    *,
    chi2_ratio: float = 2.0,
    mad_cut: float = 5.0,
) -> tuple[list[int], list[str | None]]:
    """Pre-mean MYTHRA member filtering: window -> diverged -> per-dataset chi2_ratio -> MAD outlier.

    - ``chi2_red_matrix``: shape ``(n_members, n_datasets)`` raw ``chi2 / n_d`` per dataset.
    - ``in_window``: boolean mask of length ``n_members`` (True for GP members or L-curve
      weights in ``[corner / 10**window_dex, corner]``).
    - ``dataset_sizes``: length ``n_datasets`` sample counts ``n_d`` so total reduced chi2 is
      ``sum_d (n_d * chi2_red_d) / sum_d n_d``.
    Returns ``(survivor_indices_sorted_by_total_chi2, reasons)``.
    """
    chi2_red = np.asarray(chi2_red_matrix, dtype=float)
    win = np.asarray(in_window, dtype=bool)
    sizes = np.asarray(dataset_sizes, dtype=float)
    n_members = chi2_red.shape[0]
    reasons: list[str | None] = [None] * n_members

    for i in range(n_members):
        if not win[i]:
            reasons[i] = "window"
        elif not np.all(np.isfinite(chi2_red[i])):
            reasons[i] = "diverged"

    alive = [i for i in range(n_members) if reasons[i] is None]
    if not alive:
        return [], reasons

    best_per_ds = np.min(chi2_red[alive], axis=0)
    for i in list(alive):
        if np.any(chi2_red[i] > float(chi2_ratio) * best_per_ds):
            reasons[i] = "chi2"
    alive = [i for i in range(n_members) if reasons[i] is None]
    if not alive:
        return [], reasons

    total_red = (chi2_red @ sizes) / float(np.sum(sizes))
    vals = total_red[alive]
    med = float(np.median(vals))
    mad_sigma = 1.4826 * float(np.median(np.abs(vals - med)))
    if mad_sigma > 0.0:
        for i in list(alive):
            if total_red[i] - med > float(mad_cut) * mad_sigma:
                reasons[i] = "outlier"
    alive = [i for i in range(n_members) if reasons[i] is None]
    alive.sort(key=lambda idx: float(total_red[idx]))
    return alive, reasons


# ---------------------------------------------------------------------------
# Radio time-adjacent visibility differencing noise scale
# ---------------------------------------------------------------------------


def radio_sigma_scale_group_ref(
    time: np.ndarray,
    ant1: np.ndarray,
    ant2: np.ndarray,
    chan: np.ndarray | None,
    scan: np.ndarray,
    ddid: np.ndarray,
    vis: np.ndarray,
    weight: np.ndarray,
) -> dict[str, float | int]:
    """Model-free error scale ``s`` from non-overlapping time-adjacent pairs.

    Within each ``(ddid, scan, ant1, ant2, chan)`` stream sorted by ``time``:
    - Non-overlapping adjacent pairs ``(0, 1), (2, 3), ...`` have variance
      ``Var(Delta Re) = s^2 (1/w_i + 1/w_j)`` for constant sky signal.
    - Normalized squared differences are ``u_re = (Delta Re)^2 / (1/w_i + 1/w_j)``
      and ``u_im = (Delta Im)^2 / (1/w_i + 1/w_j)``, with pair average
      ``u_comb = 0.5 * (u_re + u_im)``.
    - ``s_re = sqrt(mean(u_re))``, ``s_im = sqrt(mean(u_im))``, and
      ``s = sqrt(mean(u_comb))``, with plug-in delta-method standard errors
      ``se(sqrt(m)) = std(u, ddof=0) / (2 sqrt(m * n_pairs))`` across the ``n_pairs``
      pair estimates (capturing any shared amplitude/phase correlation between
      the real and imaginary quadratures of the same pair).
    - ``lag_ratio`` compares far-apart pairs (separated by half a stream's length)
      against adjacent pairs: ``~1`` for white noise on constant signal, ``>> 1``
      when smooth temporal sky/phase drift dominates.
    """
    t = np.asarray(time, dtype=float)
    a1 = np.asarray(ant1, dtype=int)
    a2 = np.asarray(ant2, dtype=int)
    ch = np.zeros_like(a1) if chan is None else np.asarray(chan, dtype=int)
    sc = np.asarray(scan, dtype=int)
    dd = np.asarray(ddid, dtype=int)
    v = np.asarray(vis, dtype=np.complex128)
    w = np.asarray(weight, dtype=float)

    keys = np.column_stack([dd, sc, a1, a2, ch])
    unique_keys = np.unique(keys, axis=0)

    u_re_list = []
    u_im_list = []
    far_sq_list = []

    for key in unique_keys:
        mask = np.all(keys == key, axis=1)
        idx = np.flatnonzero(mask)
        if idx.size < 2:
            continue
        order = idx[np.argsort(t[idx], kind="mergesort")]
        n_even = (order.size // 2) * 2
        i0 = order[0:n_even:2]
        i1 = order[1:n_even:2]
        denom = (1.0 / w[i0]) + (1.0 / w[i1])
        dv = v[i1] - v[i0]
        u_re_list.append((dv.real**2) / denom)
        u_im_list.append((dv.imag**2) / denom)

        if order.size >= 4:
            half = order.size // 2
            f0 = order[:half]
            f1 = order[half : 2 * half]
            f_denom = (1.0 / w[f0]) + (1.0 / w[f1])
            fdv = v[f1] - v[f0]
            far_sq_list.append(0.5 * ((fdv.real**2 + fdv.imag**2) / f_denom))

    u_re = np.concatenate(u_re_list) if u_re_list else np.empty(0, dtype=float)
    u_im = np.concatenate(u_im_list) if u_im_list else np.empty(0, dtype=float)
    n_pairs = int(u_re.size)
    if n_pairs == 0:
        return {
            "n_pairs": 0,
            "s": np.nan,
            "s_se": np.nan,
            "s_re": np.nan,
            "s_re_se": np.nan,
            "s_im": np.nan,
            "s_im_se": np.nan,
            "lag_ratio": np.nan,
        }

    m_re = float(np.mean(u_re))
    m_im = float(np.mean(u_im))
    u_comb = 0.5 * (u_re + u_im)
    m_comb = float(np.mean(u_comb))

    s_re = float(np.sqrt(m_re))
    s_im = float(np.sqrt(m_im))
    s_val = float(np.sqrt(m_comb))

    se_m_re = float(np.std(u_re, ddof=0) / np.sqrt(n_pairs)) if n_pairs > 1 else np.nan
    se_m_im = float(np.std(u_im, ddof=0) / np.sqrt(n_pairs)) if n_pairs > 1 else np.nan
    se_m_pair = float(np.std(u_comb, ddof=0) / np.sqrt(n_pairs)) if n_pairs > 1 else np.nan

    s_re_se = se_m_re / (2.0 * s_re) if s_re > 0.0 else np.nan
    s_im_se = se_m_im / (2.0 * s_im) if s_im > 0.0 else np.nan
    s_se_pair = se_m_pair / (2.0 * s_val) if s_val > 0.0 else np.nan

    far_arr = np.concatenate(far_sq_list) if far_sq_list else np.empty(0, dtype=float)
    lag_ratio = float(np.mean(far_arr) / m_comb) if far_arr.size > 0 and m_comb > 0.0 else np.nan

    return {
        "n_pairs": n_pairs,
        "s": s_val,
        "s_se": s_se_pair,
        "s_re": s_re,
        "s_re_se": s_re_se,
        "s_im": s_im,
        "s_im_se": s_im_se,
        "lag_ratio": lag_ratio,
    }
