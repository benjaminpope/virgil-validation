"""Independent SciPy/NumPy reference primitives for multi-epoch orbit cells, short arcs and summaries.

Every routine here is derived from textbook celestial mechanics or statistical
quadrature and shares no code or imports with ``virgil``:

* Sky-plane positions and velocities from Kepler's equation ``M = E - e sin E``
  and the time derivative ``dE/dt = (2 pi / P) / (1 - e cos E)`` via the
  Thiele-Innes constants of ``crosscheck.orbits`` (Heintz 1978, *Double Stars*,
  sec. 15; Danby 1988, *Fundamentals of Celestial Mechanics*, 2nd ed., Ch. 6).
* Generalized linear least-squares variable projection of ``(A, B, F, G)`` on
  sky positions and optional sky velocities at fixed ``(P, t_p, e)``
  (Golub & Pereyra 1973, *SIAM J. Numer. Anal.* 10, 413; Lucy 2014, *A&A* 563, A126).
* Cycle-count alias bands ``T / (N + 1/2) <= P <= T / (N - 1/2)`` and
  log-uniform Jeffreys cell masses (Jeffreys 1946; Price-Whelan et al. 2017,
  *ApJ* 837, 20; VanderPlas 2018, *ApJS* 236, 16).
* Exact Gaussian-convolved uniform log-flux box prior via ``scipy.special.ndtr`` /
  ``log_ndtr``, Gauss-Jacobi quadrature nodes for the ``Beta(1, 9)`` outlier-rate
  integral (Golub & Welsch 1969, *Math. Comp.* 23, 221; NIST DLMF sec. 3.5(v)),
  Aitken (1935) weighted linear-motion projection, and Laplace outlier integrals.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import special

from crosscheck.orbits import eccentric_anomaly, thiele_innes


def sky_state(t, period, t_peri, ecc, inc, omega, Omega, a):
    """Exact relative sky position ``(dra, ddec)`` and velocity ``(vra, vdec)``.

    Differentiating ``x = cos E - e`` and ``y = sqrt(1 - e^2) sin E`` with
    ``dE/dt = (2 pi / P) / (1 - e cos E)`` (Danby 1988, Ch. 6; Heintz 1978,
    sec. 15) gives sky offsets in ``a``'s units and sky velocities in ``a / day``.

    Parameters
    ----------
    t : array_like
        Observation times (days).
    period, t_peri, ecc, inc, omega, Omega, a : float
        Period (days), periastron epoch (days), eccentricity, inclination (deg),
        companion argument of periastron (deg), ascending node (deg), and
        semi-major axis (mas).

    Returns
    -------
    dra, ddec, vra, vdec : numpy.ndarray
        East/North offsets (mas) and East/North angular velocities (mas/day).
    """
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    mean_motion = 2.0 * np.pi / float(period)
    mean_anom = np.mod(mean_motion * (t_arr - float(t_peri)) + np.pi, 2.0 * np.pi) - np.pi
    E = eccentric_anomaly(mean_anom, float(ecc))

    one_minus_e2 = math.sqrt(max(0.0, 1.0 - float(ecc) ** 2))
    denom = 1.0 - float(ecc) * np.cos(E)
    Edot = mean_motion / denom

    x = np.cos(E) - float(ecc)
    y = one_minus_e2 * np.sin(E)
    xdot = -Edot * np.sin(E)
    ydot = one_minus_e2 * Edot * np.cos(E)

    A, B, F, G = thiele_innes(a, inc, omega, Omega)
    dra = B * x + G * y
    ddec = A * x + F * y
    vra = B * xdot + G * ydot
    vdec = A * xdot + F * ydot
    return dra, ddec, vra, vdec


def fit_thiele_innes_state(
    t,
    dra,
    ddec,
    cov_pos,
    period,
    t_peri,
    ecc,
    *,
    vra=None,
    vdec=None,
    cov_vel=None,
    measured=None,
):
    """Generalized linear least-squares fit of ``(A, B, F, G)`` at fixed ``(P, t_p, e)``.

    At fixed nonlinear phase parameters ``(P, t_p, e)``, both sky positions and
    sky velocities are exact linear combinations of ``theta = (A, B, F, G)``:
    ``[dra, ddec] = [[0, x, 0, y], [x, 0, y, 0]] theta`` and
    ``[vra, vdec] = [[0, xdot, 0, ydot], [xdot, 0, ydot, 0]] theta``
    (Heintz 1978; Golub & Pereyra 1973).

    Returns
    -------
    theta : numpy.ndarray, shape (4,)
        Fitted ``(A, B, F, G)``.
    chi2 : float
        Weighted residual sum of squares at ``theta``.
    """
    t_arr = np.asarray(t, dtype=float)
    dra_arr = np.asarray(dra, dtype=float)
    ddec_arr = np.asarray(ddec, dtype=float)
    cov_p = np.asarray(cov_pos, dtype=float)

    mean_motion = 2.0 * np.pi / float(period)
    mean_anom = np.mod(mean_motion * (t_arr - float(t_peri)) + np.pi, 2.0 * np.pi) - np.pi
    E = eccentric_anomaly(mean_anom, float(ecc))
    one_minus_e2 = math.sqrt(max(0.0, 1.0 - float(ecc) ** 2))
    Edot = mean_motion / (1.0 - float(ecc) * np.cos(E))

    x = np.cos(E) - float(ecc)
    y = one_minus_e2 * np.sin(E)
    xdot = -Edot * np.sin(E)
    ydot = one_minus_e2 * Edot * np.cos(E)

    normal = np.zeros((4, 4), dtype=float)
    rhs = np.zeros(4, dtype=float)
    design_rows = []
    obs_blocks = []
    weight_blocks = []

    for k in range(len(t_arr)):
        M_p = np.array(
            [
                [0.0, x[k], 0.0, y[k]],
                [x[k], 0.0, y[k], 0.0],
            ],
            dtype=float,
        )
        y_p = np.array([dra_arr[k], ddec_arr[k]], dtype=float)
        W_p = np.linalg.inv(cov_p[k])
        normal += M_p.T @ W_p @ M_p
        rhs += M_p.T @ (W_p @ y_p)
        design_rows.append(M_p)
        obs_blocks.append(y_p)
        weight_blocks.append(W_p)

    if vra is not None and vdec is not None and cov_vel is not None:
        vra_arr = np.asarray(vra, dtype=float)
        vdec_arr = np.asarray(vdec, dtype=float)
        cov_v = np.asarray(cov_vel, dtype=float)
        mask = np.ones(len(t_arr), dtype=bool) if measured is None else np.asarray(measured, dtype=bool)
        for k in range(len(t_arr)):
            if not mask[k]:
                continue
            M_v = np.array(
                [
                    [0.0, xdot[k], 0.0, ydot[k]],
                    [xdot[k], 0.0, ydot[k], 0.0],
                ],
                dtype=float,
            )
            y_v = np.array([vra_arr[k], vdec_arr[k]], dtype=float)
            W_v = np.linalg.inv(cov_v[k])
            normal += M_v.T @ W_v @ M_v
            rhs += M_v.T @ (W_v @ y_v)
            design_rows.append(M_v)
            obs_blocks.append(y_v)
            weight_blocks.append(W_v)

    theta = np.linalg.solve(normal, rhs)
    chi2 = 0.0
    for M_b, y_b, W_b in zip(design_rows, obs_blocks, weight_blocks):
        r_b = y_b - M_b @ theta
        chi2 += float(r_b @ (W_b @ r_b))
    return theta, chi2


def alias_band_edges(times, p_range):
    """Frequency-tiling alias bands ``N = round(T / P)`` clipped to ``p_range``.

    For span ``T = max(times) - min(times)``, band ``N >= 1`` covers
    ``[T / (N + 1/2), T / (N - 1/2)]`` and band ``N = 0`` covers ``[2 T, inf)``
    (VanderPlas 2018; Price-Whelan et al. 2017). Ordered by integer cycle count
    ``N = 1, 2, ..., n_max``, followed by ``N = 0`` when ``p_max > 2 T``.
    """
    t = np.asarray(times, dtype=float)
    span = float(np.max(t) - np.min(t))
    p_min, p_max = float(p_range[0]), float(p_range[1])
    if span <= 0.0 or p_min <= 0.0 or p_max <= p_min:
        raise ValueError("Require positive span and 0 < p_min < p_max.")

    n_max = max(1, int(math.floor(span / p_min + 0.5)))
    n_min = max(1, int(math.ceil(span / p_max - 0.5)))
    bands = []
    for n in range(1, n_max + 1):
        if n < n_min:
            continue
        lo = max(p_min, span / (n + 0.5))
        hi = min(p_max, span / (n - 0.5))
        if hi > lo:
            bands.append((n, lo, hi))
    if p_max > 2.0 * span:
        lo0 = max(p_min, 2.0 * span)
        if p_max > lo0:
            bands.append((0, lo0, p_max))
    return bands


def cell_log_masses(cells, p_cut, p_max):
    """Log-uniform Jeffreys prior share ``ln[ln(p_hi / p_lo) / ln(p_max / p_cut)]``."""
    denom = math.log(float(p_max) / float(p_cut))
    return np.array(
        [math.log(math.log(float(p_hi) / float(p_lo)) / denom) for _, p_lo, p_hi in cells],
        dtype=float,
    )


def period_cut_bound(
    times,
    field,
    night_length,
    *,
    resolution,
    motion_fraction=0.5,
    p_floor=0.2,
):
    """Declared shortest period ``max(2 * dt_min, P_night, p_floor)`` from circular-speed bound."""
    p_night = max(
        float(p_floor),
        2.0 * math.pi * float(field) * float(night_length) / (float(motion_fraction) * float(resolution)),
    )
    t_unique = np.unique(np.asarray(times, dtype=float))
    if len(t_unique) < 2:
        return p_night
    dt_min = float(np.min(np.diff(t_unique)))
    return max(2.0 * dt_min, p_night)


def period_window_bounds(
    times,
    field,
    night_length,
    *,
    resolution,
    motion_fraction=0.5,
    span_factor=4.0,
    p_floor=0.2,
):
    """Data-only period window ``(p_min, p_max)`` from within-night smearing and total span."""
    t_arr = np.asarray(times, dtype=float)
    span = float(np.max(t_arr) - np.min(t_arr))
    p_min = max(
        float(p_floor),
        2.0 * math.pi * float(field) * float(night_length) / (float(motion_fraction) * float(resolution)),
    )
    p_max = float(span_factor) * span
    return p_min, p_max


def log_flux_box_exact(m_f, sigma_f, ln_f_floor):
    """Exact ``ln E[pi(ln f)]`` for ``ln f ~ N(m_f, sigma_f^2)`` and uniform box ``[ell, 0]``.

    Integrating ``1 / |ell|`` over ``[ell, 0]`` against a Gaussian of mean
    ``m_f`` and standard deviation ``sigma_f`` yields
    ``ln([Phi(-m_f / sigma_f) - Phi((ell - m_f) / sigma_f)] / |ell|)``.
    Evaluated stably in the tails via ``scipy.special.log_ndtr``:
    ``log(Phi(b) - Phi(a)) = log_ndtr(b) + log1p(-exp(log_ndtr(a) - log_ndtr(b)))``
    for ``b <= 0``, and by ``1 - Phi(z) = Phi(-z)`` when ``a > 0``.
    """
    ell = float(ln_f_floor)
    if ell >= 0.0:
        raise ValueError("ln_f_floor must be strictly negative.")
    m = np.asarray(m_f, dtype=float)
    s = np.asarray(sigma_f, dtype=float)
    b = -m / s
    a = (ell - m) / s

    # Use symmetry Phi(b) - Phi(a) = Phi(-a) - Phi(-b) when b > 0 and a > 0 so
    # the dominant log_ndtr argument stays in the accurate negative half-line.
    flip = (a > 0.0) & (b > 0.0)
    hi = np.where(flip, -a, b)
    lo = np.where(flip, -b, a)
    log_hi = special.log_ndtr(hi)
    log_lo = special.log_ndtr(lo)
    log_diff = log_hi + np.log1p(-np.exp(log_lo - log_hi))
    return log_diff - math.log(abs(ell))


def beta_eps_lo_node(n_nights, alpha_prior=1.0, beta_prior=9.0):
    """Smallest node ``eps_lo`` of the ``ceil((K + 1) / 2)``-point Gauss-Jacobi rule on ``[0, 1]``.

    The Beta density on ``[0, 1]`` is proportional to ``eps^(alpha - 1) (1 - eps)^(beta - 1)``.
    Mapping ``x in [-1, 1]`` via ``eps = (x + 1) / 2`` converts ``(1 - x)^a (1 + x)^b``
    with ``a = beta - 1`` and ``b = alpha - 1`` (Golub & Welsch 1969; NIST DLMF sec. 3.5(v)).
    """
    n_nodes = int(math.ceil((int(n_nights) + 1) / 2.0))
    roots, _ = special.roots_jacobi(n_nodes, float(beta_prior) - 1.0, float(alpha_prior) - 1.0)
    eps_nodes = 0.5 * (roots + 1.0)
    return float(np.min(eps_nodes))


def keep_peak_mask(peak_loglikes, log_outlier_k, n_nights, *, delta=0.001, alpha_prior=1.0, beta_prior=9.0):
    """Boolean mask of peaks satisfying ``(1 - eps_lo) L_j >= delta * eps_lo * exp(b_k)``."""
    eps_lo = beta_eps_lo_node(n_nights, alpha_prior=alpha_prior, beta_prior=beta_prior)
    ll = np.asarray(peak_loglikes, dtype=float)
    lhs = math.log1p(-eps_lo) + ll
    rhs = math.log(float(delta)) + math.log(eps_lo) + float(log_outlier_k)
    return lhs >= rhs


def wls_line_projection(offsets, weights=None):
    """``(2, n_frames)`` Aitken (1935) weighted least-squares map from frame positions to ``(p, v)``."""
    dt = np.asarray(offsets, dtype=float)
    n_frames = len(dt)
    if n_frames == 1 or np.allclose(dt, 0.0):
        proj = np.zeros((2, n_frames), dtype=float)
        if weights is None:
            proj[0, :] = 1.0 / n_frames
        else:
            w = np.asarray(weights, dtype=float)
            proj[0, :] = w / np.sum(w)
        return proj

    w = np.ones(n_frames, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    X = np.column_stack([np.ones(n_frames, dtype=float), dt])
    XW = X.T * w[None, :]
    return np.linalg.solve(XW @ X, XW)


def rescaled_peak_precision(hessians, scale=1.0):
    """Precision ``sum_g s_g^-2 H_g`` for per-block ``1/2 chi^2`` Hessians ``H_g``."""
    H = np.asarray(hessians, dtype=float)
    s = np.broadcast_to(np.asarray(scale, dtype=float), (H.shape[0],))
    return np.sum(H / (s[:, None, None] ** 2), axis=0)


def rescaled_peak_log_scale(n_blocks, chi2_blocks, scale):
    """Exact Gaussian block-scale log-likelihood shift ``-sum_g [n_g ln s_g + 0.5 chi2_g (s_g^-2 - 1)]``."""
    n_b = np.asarray(n_blocks, dtype=float)
    c_b = np.asarray(chi2_blocks, dtype=float)
    s_b = np.broadcast_to(np.asarray(scale, dtype=float), n_b.shape)
    return float(-np.sum(n_b * np.log(s_b) + 0.5 * c_b * (s_b**-2 - 1.0)))


def night_summary_jitter_cov(peak_cov_scaled, jitter, n_frames, t_ss):
    """Add independent per-frame position jitter ``sigma_J`` in quadrature to a night summary."""
    out = np.array(peak_cov_scaled, dtype=float, copy=True)
    j2 = float(jitter) ** 2
    if j2 == 0.0:
        return out
    out[0, 0] += j2 / float(n_frames)
    out[1, 1] += j2 / float(n_frames)
    if int(n_frames) <= 1 or float(t_ss) <= 0.0:
        out[2, 2] = np.inf
        out[3, 3] = np.inf
    else:
        out[2, 2] += j2 / float(t_ss)
        out[3, 3] += j2 / float(t_ss)
    return out


def laplace_outlier_log_density_fixed_scale(peak_loglikes, peak_covs, log_prior_vals, *, velocity=True):
    """Fixed-scale Laplace integral ``ln sum_j L_j (2 pi)^(d/2) |Sigma_j|^(1/2) pi_out(m_j)``."""
    terms = []
    idx = [0, 1, 2, 3, 4] if velocity else [0, 1, 4]
    dim = len(idx)
    for ll, cov, lp in zip(peak_loglikes, peak_covs, log_prior_vals):
        sub_cov = np.asarray(cov, dtype=float)[np.ix_(idx, idx)]
        sign, logdet = np.linalg.slogdet(sub_cov)
        if sign <= 0 or not np.isfinite(lp):
            continue
        terms.append(float(ll) + 0.5 * dim * math.log(2.0 * math.pi) + 0.5 * float(logdet) + float(lp))
    if not terms:
        return -np.inf
    m = max(terms)
    return float(m + math.log(sum(math.exp(t - m) for t in terms)))


def quantum_rank_scores(scores, quantum=1e-3):
    """Sort candidate indices best-first after rounding to ``quantum`` nats; non-finite last."""
    arr = np.asarray(scores, dtype=float)
    n = len(arr)
    finite = np.isfinite(arr)
    q = np.where(finite, np.round(arr / float(quantum)), -np.inf)
    # Sort ascending by (-is_finite, -rounded_score, candidate_index)
    keys = [
        (0 if finite[i] else 1, -float(q[i]) if finite[i] else 0.0, i)
        for i in range(n)
    ]
    keys.sort()
    return np.array([k[2] for k in keys], dtype=int)
