"""Brute-force references for the derived radio-imaging methods of virgil's innovation register.

numpy only; nothing is taken from virgil. Each function is written from the formula named in its
docstring, at sizes where the dense answer is cheap:

* ``gn_diagonal``, ``entropy_metric_diagonal``: the Gauss-Newton diagonal of a softmax image fit and the
  Skilling-Bryan entropy metric pulled back to the log-brightness (register B2);
* ``direct_chi2_by_ddid``: the chi-squared of a one-image scene with per-observation gains and phase-centre
  offsets, by a direct DFT (register B5);
* ``DeltaProblem``: chi-squared differences about a float64 reference, in float64, in float32 and by the
  linear-delta-V formula (register B6).
"""

import numpy as np

MAS = np.pi / 180.0 / 3600.0 / 1000.0  # radians per milliarcsecond


def softmax_image(eta):
    """Unit-sum brightness ``b = exp(eta) / sum(exp(eta))`` of an array of log-brightnesses."""
    e = np.exp(np.asarray(eta, float) - np.max(eta))
    return e / e.sum()


def pixel_positions(npix, pixel_mas):
    """East and North (mas) of the pixel centres, rows North first and columns East first, flattened."""
    axis = ((npix - 1) / 2.0 - np.arange(npix)) * pixel_mas
    east, north = np.meshgrid(axis, axis)
    return east.ravel(), north.ravel()


def dft_matrix(u, v, east, north):
    """``E[j, p] = exp(-2 pi i (u_j East_p + v_j North_p))`` for u, v in wavelengths and positions in mas."""
    return np.exp(-2j * np.pi * MAS * (np.outer(u, east) + np.outer(v, north)))


# ------------------------------------------------------------------ B2


def gn_diagonal(u, v, weight, eta, pixel_mas, flux):
    """Diagonal of ``J^T J`` of the whitened residuals ``sqrt(w) (V - M)`` (Re and Im) with respect to ``eta``.

    ``M = flux * E b`` and ``b = softmax(eta)``, so ``dM/d eta_q = b_q (flux E[:, q] - M)``: one dense
    column per pixel, summed in squares over the rows. Source: Schraudolph 2002,
    doi:10.1162/08997660260028683 (the Gauss-Newton matrix of a least-squares term)."""
    east, north = pixel_positions(eta.shape[0], pixel_mas)
    b = softmax_image(eta).ravel()
    vis = flux * dft_matrix(u, v, east, north)
    model = vis @ b
    dm = b[None, :] * (vis - model[:, None])
    return (np.asarray(weight)[:, None] * np.abs(dm) ** 2).sum(axis=0).reshape(eta.shape)


def entropy_metric_diagonal(weight, eta):
    """Diagonal of ``G = weight * sum_p (db_p)^2 / b_p`` pulled back to ``eta`` through the softmax.

    ``db_p / d eta_q = b_p (delta_pq - b_q)`` is built as a dense matrix, so
    ``G_qq = weight * sum_p (db_p / d eta_q)^2 / b_p``. Source: Skilling & Bryan 1984, 1984MNRAS.211..111S."""
    b = softmax_image(eta).ravel()
    jac = np.diag(b) - np.outer(b, b)  # (p, q)
    return (weight * (jac**2 / b[:, None]).sum(axis=0)).reshape(np.shape(eta))


# ------------------------------------------------------------------ B5


def direct_chi2_by_ddid(u, v, vis, weight, ddid, obs, brightness, pixel_mas, flux, gain, offset_mas):
    """``sum_j w_j |V_j - M_j|^2`` per ddid, by the pixel DFT, for a one-image scene.

    ``M_j = flux g_e exp(-2 pi i (u_j dRA_e + v_j dDec_e)) sum_p b_p exp(-2 pi i (u_j East_p + v_j North_p))``
    with the gain and offset of the observation ``e`` of point ``j`` (``gain`` and ``offset_mas`` list all
    observations, the reference first). Source: Barnett et al. 2019,
    2019SJSC...41C.479B (the transform the offsets evaluate); the DFT is its definition."""
    east, north = pixel_positions(brightness.shape[0], pixel_mas)
    scene = dft_matrix(u, v, east, north) @ (brightness.ravel() / brightness.sum())
    offset = np.asarray(offset_mas, float)[obs]
    shift = np.exp(-2j * np.pi * MAS * (u * offset[:, 0] + v * offset[:, 1]))
    model = flux * np.asarray(gain, float)[obs] * shift * scene
    resid = weight * np.abs(vis - model) ** 2
    return np.array([resid[ddid == d].sum() for d in range(int(ddid.max()) + 1)])


# ------------------------------------------------------------------ B6


class DeltaProblem:
    """A small unnormalised-image fit, ``V(z) = E exp(z)``, with ``chi2 = sum w |V_data - V|^2``.

    The reference ``z_ref`` and its residual are held in float64. ``delta_chi2_float32`` forms
    ``dV = E exp(z_ref) expm1(z - z_ref)`` in float32 (the linearity of the transform: ``V = V_ref + dV``)
    and the difference ``chi2(z) - chi2(z_ref) = sum w (|dV|^2 - 2 Re(conj(r_ref) dV))`` in float64;
    ``naive_chi2_float32`` subtracts two float32 chi-squared values. Source: residual-based mixed-precision
    iterative refinement, Higham 2002, doi:10.1137/1.9780898718027."""

    def __init__(self, dft, z_ref, vis_data, weight):
        self.dft = np.asarray(dft)
        self.z_ref = np.asarray(z_ref, float)
        self.vis_data = np.asarray(vis_data)
        self.weight = np.asarray(weight, float)
        self.resid_ref = self.vis_data - self.dft @ np.exp(self.z_ref)
        self.dft32 = self.dft.astype(np.complex64)
        self.vis_data32 = self.vis_data.astype(np.complex64)
        self.weight32 = self.weight.astype(np.float32)

    def chi2_float64(self, z):
        return float(np.sum(self.weight * np.abs(self.vis_data - self.dft @ np.exp(z)) ** 2))

    def chi2_float32(self, z):
        r = self.vis_data32 - self.dft32 @ np.exp(np.asarray(z).astype(np.float32))
        return float(np.sum(self.weight32 * (r.real**2 + r.imag**2), dtype=np.float32))

    def exact_delta(self, z):
        return self.chi2_float64(z) - self.chi2_float64(self.z_ref)

    def naive_delta_float32(self, z):
        return self.chi2_float32(z) - self.chi2_float32(self.z_ref)

    def delta_vis(self, z):
        """``dV`` in float32, returned as complex128."""
        dz = (np.asarray(z) - self.z_ref).astype(np.float32)
        return (self.dft32 @ (np.exp(self.z_ref).astype(np.float32) * np.expm1(dz))).astype(np.complex128)

    def delta_chi2_float32(self, z):
        dv = self.delta_vis(z)
        return float(np.sum(self.weight * (np.abs(dv) ** 2 - 2.0 * (self.resid_ref.conj() * dv).real)))

    def absolute_scale(self, z):
        """``sum w (|dV|^2 + 2 |r_ref| |dV|)``: the sum of the absolute values of the terms of the difference."""
        dv = np.abs(self.delta_vis(z))
        return float(np.sum(self.weight * (dv**2 + 2.0 * np.abs(self.resid_ref) * dv)))
