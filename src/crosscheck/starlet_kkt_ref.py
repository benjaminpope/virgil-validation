r"""Independent NumPy/SciPy references for Batch 7 (scenes, starlet wavelets,
KKT optimality certificates, DoF-corrected discrepancy, Laplace GP evidence,
MYTHRA ensemble selection, and time-differenced radio error scales).

Nothing here imports ``virgil``, ``drpangloss``, or ``jaxbessel``. Every formula
is implemented directly from its published textbook or journal definition:

- Synthetic sky scenes (inclined modulated Gaussian ring, face-on Archimedean
  spiral with rounded arm caps, circular Gaussian blob) and rotated sky-grid
  coordinates: standard visual-binary / interferometric sky geometry (North up,
  East left, position angle North through East; Bracewell 2000, *The Fourier
  Transform and Its Applications*, 3rd ed., McGraw-Hill).
- Isotropic undecimated B3-spline à-trous ("starlet") wavelet decomposition and
  exact synthesis: Starck, Murtagh & Fadili (2010), *Sparse Image and Signal
  Processing*, Cambridge University Press, §3.5.
- Short-baseline Gaussian second-moment FWHM and field-of-view rules: Bracewell
  (2000), $-\ln V^2 = \pi^2 \theta^2 \rho^2 / (2 \ln 2)$.
- Constrained and reparametrized KKT stationarity and complementary-slackness
  certificates: Nocedal & Wright (2006), *Numerical Optimization*, 2nd ed.,
  Springer, ch. 12 and §16.7.
- Regularizer Hessians (Dirichlet-padded Total Squared Variation, Kuramochi et
  al. 2018, *ApJ* 858, 56; unnormalized Skilling entropy, Skilling 1989),
  degrees-of-freedom-corrected Morozov discrepancy $m(w) = \chi^2/s^2 - (n - G)$
  with $G = \operatorname{tr}[K(K + H_R)^{-1}]$ (Morozov 1966; Craven & Wahba
  1979), and exact Rademacher trace-estimator variance (Hutchinson 1989,
  *Commun. Stat. Simul. Comput.* 18, 1059; Bekas, Kokiopoulou & Saad 2007,
  *Appl. Numer. Math.* 57, 1214).
- Laplace log-evidence, posterior covariance, effective parameter count
  $\gamma = \sum_k s_k^2 / (1 + s_k^2)$, and empirical Bayes error scale
  $s^2 = \chi^2 / (N - \gamma)$: MacKay (1992, *Neural Comput.* 4, 415);
  classic MaxEnt balance (Gull 1989) and L-curve curvature (Hansen & O'Leary
  1993, *SIAM J. Sci. Comput.* 14, 1487).
- Randomized multi-family ensemble selection and non-compounding running-mean
  $\chi^2$ gate: Drevon et al. (2026, MYTHRA).
- Model-free radio visibility noise scale from non-overlapping time-adjacent
  baseline-channel-scan pairs and far-pair lag ratio: standard radio
  interferometric visibility differencing (Thompson, Moran & Swenson 2017,
  *Interferometry and Synthesis in Radio Astronomy*, 3rd ed., Springer, ch. 6)
  and Cornwell & Wilkinson (1981, *MNRAS* 196, 1067) reference-gain gauge.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg, ndimage

MAS_TO_RAD = np.pi / (180.0 * 3600.0 * 1000.0)
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
B3_KERNEL_1D = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=float) / 16.0


# ---------------------------------------------------------------------------
# Sky-grid coordinates, synthetic scenes, and centroids
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

    Parameters match the physical specification:
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
    distance_mode: str = "radial_cap",
) -> np.ndarray:
    """Unit-sum face-on Archimedean spiral ``r = step_mas * theta / (2 pi)``.

    Winds towards increasing position angle (North through East) from ``pa_deg``
    over ``theta in [0, 2 * pi * turns]``, with Gaussian standard deviation
    ``width_mas`` (NOT FWHM), rounded arm-end caps, and optional radial decay
    ``exp(-r / fade_mas)``.

    ``distance_mode="radial_cap"`` measures radial cross-section distance
    ``|r - r(theta_k)|`` along the ray at interior turns plus circular Euclidean
    caps at ``theta = 0`` and ``theta = 2 * pi * turns`` (directly implementing
    "Gaussian cross-section along the radial direction ... ends in a rounded cap").
    ``distance_mode="euclidean"`` computes minimum Euclidean distance in the
    sky plane to the densely sampled spine curve ``(r(theta) sin psi, r(theta) cos psi)``.
    """
    east, north = sky_grid_mas(npix, pixel_scale_mas)
    r = np.hypot(east, north)
    theta_max = 2.0 * np.pi * float(turns)
    pa0 = np.deg2rad(float(pa_deg))

    if distance_mode == "radial_cap":
        # Ray angle relative to pa0 in [0, 2*pi)
        dpsi = np.mod(np.arctan2(east, north) - pa0, 2.0 * np.pi)
        # Endpoint cap centres at theta = 0 (origin) and theta = theta_max
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
    elif distance_mode == "euclidean":
        n_curve = max(2048, int(np.ceil(1024 * float(turns))))
        theta = np.linspace(0.0, theta_max, n_curve)
        r_c = float(step_mas) * theta / (2.0 * np.pi)
        psi_c = pa0 + theta
        cx = r_c * np.sin(psi_c)
        cy = r_c * np.cos(psi_c)
        dx = east[..., None] - cx[None, None, :]
        dy = north[..., None] - cy[None, None, :]
        dist = np.sqrt(np.min(dx * dx + dy * dy, axis=-1))
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
    grad_b: np.ndarray,
    *,
    support: np.ndarray | None = None,
    flux: float = 1.0,
    weight: float | None = None,
    peak_lit_frac: float = 1e-3,
    dark_frac: float = 1e-6,
) -> dict[str, float]:
    """Simplex (softmax ``Image``) KKT stationarity & complementary-slackness metrics.

    Nocedal & Wright (2006, ch. 12): minimizing ``J(b)`` on ``{b >= 0, sum b = 1}``
    has Lagrangian ``J(b) - mu (sum b_i - 1) - sum lambda_i b_i`` with ``lambda_i >= 0``
    and ``lambda_i b_i = 0``. Multiplying ``g_i - mu - lambda_i = 0`` by ``b_i`` and
    summing gives ``mu = sum_i b_i g_i`` where ``g_i = dJ / db_i``.
    - On lit pixels (``b_i > peak_lit_frac * max(b)``), ``g_i - mu`` must vanish;
      per-flux gradient residual is ``(g_i - mu) / flux``, normalized by ``weight / flux``
      (giving ``rms(g_i - mu) / weight`` when ``weight`` is positive).
    - On dark support pixels (``b_i < dark_frac * max(b)``), ``g_i - mu < 0`` means
      increasing ``b_i`` lowers ``J`` (violating dual feasibility ``lambda_i >= 0``).
    """
    b = np.asarray(brightness, dtype=float)
    g = np.asarray(grad_b, dtype=float)
    sup = np.ones_like(b, dtype=bool) if support is None else np.asarray(support, dtype=bool)
    mu = float(np.sum(b[sup] * g[sup]))
    peak = float(np.max(b[sup]))
    lit = sup & (b > float(peak_lit_frac) * peak)
    dark = sup & (b < float(dark_frac) * peak)
    residual = g[lit] - mu
    rms_res = float(np.sqrt(np.mean(residual**2))) if np.any(lit) else 0.0
    g_scale = abs(mu) + (float(np.sqrt(np.mean(g[lit] ** 2))) if np.any(lit) else 0.0)
    kkt_rel = rms_res / g_scale if g_scale > 0.0 else 0.0
    kkt_w = (rms_res / float(weight)) if (weight is not None and weight > 0.0) else np.nan
    n_sup = int(np.sum(sup))
    want_grow = float(np.sum((g[dark] - mu) < 0.0) / max(n_sup, 1)) if np.any(dark) else 0.0
    return {
        "mu": mu,
        "kkt_rel": float(kkt_rel),
        "kkt_w": float(kkt_w),
        "want_grow": float(want_grow),
    }


def kkt_nonneg_linear_metrics_ref(
    fluxes: np.ndarray,
    grad_flux: np.ndarray,
    *,
    support: np.ndarray | None = None,
    weight: float | None = None,
    reference_flux: float | None = None,
    peak_lit_frac: float = 1e-3,
    zero_tol: float = 0.0,
) -> dict[str, float]:
    """Box-constrained ``f >= 0`` KKT metrics (Nocedal & Wright 2006, §16.7).

    For ``min_{f >= 0} J(f)``:
    - Projected gradient on support is ``pg_i = g_i`` if ``f_i > zero_tol`` else ``min(g_i, 0)``.
    - ``zero_violation`` is the fraction of zero pixels (``f_i <= zero_tol`` in support)
      whose gradient is strictly negative (``g_i < 0``), meaning the objective wants
      the clamped zero pixel to become positive. Active bounds with ``g_i >= 0``
      satisfy dual feasibility and have ``pg_i = 0``.
    """
    f = np.asarray(fluxes, dtype=float)
    g = np.asarray(grad_flux, dtype=float)
    sup = np.ones_like(f, dtype=bool) if support is None else np.asarray(support, dtype=bool)
    f0 = float(np.sum(f[sup])) if reference_flux is None else float(reference_flux)
    peak = float(np.max(f[sup])) if np.any(sup) else 0.0
    at_zero = sup & (f <= float(zero_tol))
    pg = np.where(~sup, 0.0, np.where(at_zero, np.minimum(g, 0.0), g))
    active_or_lit = sup & ((f > float(peak_lit_frac) * peak) | (at_zero & (g < 0.0)))
    rms_pg = float(np.sqrt(np.mean(pg[active_or_lit] ** 2))) if np.any(active_or_lit) else 0.0
    kkt_w = (
        rms_pg / (float(weight) / f0)
        if (weight is not None and weight > 0.0 and f0 > 0.0)
        else np.nan
    )
    n_zero = int(np.sum(at_zero))
    zero_violation = float(np.sum(g[at_zero] < 0.0) / n_zero) if n_zero > 0 else 0.0
    pg_inf = float(np.max(np.abs(pg[sup]))) if np.any(sup) else 0.0
    return {
        "pg_inf": pg_inf,
        "rms_pg": rms_pg,
        "kkt_w": float(kkt_w),
        "zero_violation": zero_violation,
    }


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

    Kuramochi et al. (2018, *ApJ* 858, 56): with zero-padded boundary,
    ``TSV(p) = w * (p_scale / beam)^0`` ... more generally in virgil/eht-imaging
    ``norm_reg=True`` units, ``R(f) = w * (pixel_scale_mas / beam_mas)^2 ...`` when
    ``beam_mas == pixel_scale_mas``, ``R(f) = w * sum [(p_{i+1,j}-p_{i,j})^2 + (p_{i,j+1}-p_{i,j})^2]``
    with ``p = f / reference_flux`` and a zero ring outside the grid. Differentiating
    twice gives ``H_R = (2 * w * scale / F_0^2) * (-L_Dirichlet)``, where ``L_Dirichlet``
    is the 5-point discrete Laplacian with homogeneous Dirichlet boundaries.
    """
    # Scale factor matching eht-imaging norm_reg=True (Chael et al. 2018):
    # R(f) = w * sum [(Delta_x p)^2 + (Delta_y p)^2] / (p_scale / beam)^4 with p = f / F_0.
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
    ``S = (A + A^T) / 2`` (or for symmetric ``M = K^{1/2} (K + H_R)^{-1} K^{1/2}``),
    ``E[z^T A z] = Tr(A)`` and ``Var(z^T A z) = 2 * (Tr(S^2) - sum_i S_{ii}^2)``.
    Note: if the implementation probes ``z^T (K + H_R)^{-1} (K z)`` directly with
    non-symmetric ``A = (K + H_R)^{-1} K``, ``z^T A z = z^T S z`` has variance
    ``2 * (Tr(S^2) - sum_i S_{ii}^2)``; if it probes ``|| J (K + H_R)^{-1} z ||^2``
    or ``(J^T z)...``, we return both the ``A = (K + H_R)^{-1} K`` Rademacher SE
    and the eigenvalue spectrum ``lambda_i / (lambda_i + 1)``.
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
) -> dict[str, np.ndarray | float]:
    """MacKay (1992) Laplace GP quantities at MAP ``z_map`` with whitened-residual Jacobian ``J``.

    - ``chi2 = sum(r^2)``
    - ``log_evidence = -0.5 * chi2 - log_norm - 0.5 * ||z_map||^2 - 0.5 * ln det(I + J^T J)``
      where ``log_norm = sum ln sigma_eff + 0.5 * N * ln(2 pi)`` (or ``0`` when data-only
      constants are omitted).
    - ``cov = (I + J^T J)^{-1}``
    - ``gamma = sum_k s_k^2 / (1 + s_k^2)`` over singular values ``s_k`` of ``J``
    - Fixed-point empirical Bayes error scale ``s`` solving
      ``s^2 = chi2 / (N - sum_k (s_k / s)^2 / (1 + (s_k / s)^2))`` with ``N = n_independent``
      (important when rank-deficient closure-phase triangles have more rows than independent
      closure phases), as well as the single-step plug-in ``sqrt(chi2 / (N - gamma))``.
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
    logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
    log_ev = -0.5 * chi2 - float(log_norm) - 0.5 * float(np.sum(z * z)) - 0.5 * logdet
    cov = linalg.cho_solve((chol, True), eye)
    svals = linalg.svdvals(j)
    gamma = float(np.sum((svals**2) / (1.0 + svals**2)))
    s_plugin = float(np.sqrt(chi2 / max(n_data - gamma, 1e-12)))

    # Fixed-point iteration s^2 = chi2 / (N - gamma(s)) with J(s) = J / s
    s_fp = s_plugin
    for _ in range(200):
        gamma_s = float(np.sum(((svals / s_fp) ** 2) / (1.0 + (svals / s_fp) ** 2)))
        s_next = float(np.sqrt(chi2 / max(n_data - gamma_s, 1e-12)))
        if abs(s_next - s_fp) <= 1e-13 * max(1.0, s_fp):
            s_fp = s_next
            break
        s_fp = s_next

    return {
        "chi2": chi2,
        "log_evidence": float(log_ev),
        "cov": cov,
        "gamma": gamma,
        "s_plugin": s_plugin,
        "s_fixed_point": s_fp,
    }


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
      and ``u_im = (Delta Im)^2 / (1/w_i + 1/w_j)``.
    - ``s_re = sqrt(mean(u_re))``, ``s_im = sqrt(mean(u_im))``, and
      ``s = sqrt(0.5 * (mean(u_re) + mean(u_im)))`` with delta-method standard
      errors ``se(sqrt(m)) = se(m) / (2 sqrt(m))`` from the sample variance of
      the pair values.
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

    # Plug-in delta-method standard errors (empirical variance ddof=0 over n_pairs):
    se_m_re = float(np.std(u_re, ddof=0) / np.sqrt(n_pairs)) if n_pairs > 1 else np.nan
    se_m_im = float(np.std(u_im, ddof=0) / np.sqrt(n_pairs)) if n_pairs > 1 else np.nan
    all_norm = np.concatenate([u_re, u_im])
    se_m_all = (
        float(np.std(all_norm, ddof=0) / np.sqrt(all_norm.size)) if all_norm.size > 1 else np.nan
    )
    se_m_pair = float(np.std(u_comb, ddof=0) / np.sqrt(n_pairs)) if n_pairs > 1 else np.nan

    s_re_se = se_m_re / (2.0 * s_re) if s_re > 0.0 else np.nan
    s_im_se = se_m_im / (2.0 * s_im) if s_im > 0.0 else np.nan
    s_se_pair = se_m_pair / (2.0 * s_val) if s_val > 0.0 else np.nan
    s_se_all = se_m_all / (2.0 * s_val) if s_val > 0.0 else np.nan

    far_arr = np.concatenate(far_sq_list) if far_sq_list else np.empty(0, dtype=float)
    lag_ratio = float(np.mean(far_arr) / m_comb) if far_arr.size > 0 and m_comb > 0.0 else np.nan

    return {
        "n_pairs": n_pairs,
        "s": s_val,
        "s_se": s_se_pair,
        "s_se_2n": s_se_all,
        "s_re": s_re,
        "s_re_se": s_re_se,
        "s_im": s_im,
        "s_im_se": s_im_se,
        "lag_ratio": lag_ratio,
    }
