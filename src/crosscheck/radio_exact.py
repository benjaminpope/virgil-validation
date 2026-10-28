"""Exact reference for the one-image radio loss, its gradient and its float64 error budget (Toeplitz re-examination, G2).

numpy only; nothing is taken from virgil (``design/toeplitz_reexam_042.md``). The model of point ``j`` (ddid ``d``,
observation ``e``) is

    M_j = F g_e exp(-2 pi i (u_j dRA_e + v_j dDec_e)) sum_p b_p exp(-2 pi i (u_j East_p + v_j North_p)),

``b = softmax(eta)`` (unit sum), with ``chi2_d = sum_{j in d} w_j |V_j - M_j|^2``. The phase ramp is the shift theorem
(Bracewell, *The Fourier Transform and its Applications*, 3rd ed., 2000, ch. 6) and the pixel sum is the definition
of the pixel transform; the conventions are ``crosscheck.radio_innovations``'s (``docs/alma_conventions.md``).
The loss in each error-scale mode, with ``N_d`` points in ddid ``d`` (two real numbers each):

* fixed: ``L = chi2 / 2``;
* fitted ``s_d``: ``L = sum_d [chi2_d / (2 s_d^2) + 2 N_d ln s_d]`` (Gaussian, variance ``s_d^2 / w_j`` in each of Re, Im);
* profiled: ``L = sum_d N_d [1 + ln(chi2_d / 2 N_d)]``, the fitted loss at ``s_d^2 = chi2_d / 2 N_d``.

Gradients are analytic (``d chi2_d / d theta = -2 sum_j w_j Re(conj(r_j) dM_j / d theta)``, as in
``crosscheck.radio_offsets``), computed in chunks of points so that no ``(points, pixels)`` matrix is held; the
per-ddid sums of non-negative terms use ``math.fsum`` (exactly rounded, Shewchuk 1997).

``budget`` is the error allowed between a float64 implementation and this reference, from first principles (no
safety factor; ``design/toeplitz_reexam_042.md`` section 3 gives the derivation). Unit roundoff ``u = 2^-53``.
"""

import math

import numpy as np

from crosscheck.radio_innovations import MAS, softmax_image

U = 2.0**-53  # unit roundoff of float64 (Higham 2002, section 2.1)
ETA_FFT = U + 4 * U / (1 - 4 * U) * (math.sqrt(2) + U)  # Higham 2002, Thm 24.2, with twiddle-factor error mu = u


def gamma(n):
    """``gamma_n = n u / (1 - n u)``: the worst-case relative error of a sum of ``n`` terms in any order, against the sum of
    their absolute values (Higham 2002, Lemma 3.1 and eq. 4.4)."""
    return n * U / (1.0 - n * U)


def gamma_probabilistic(n, delta=1e-6):
    """``lambda sqrt(n) u`` with ``lambda = sqrt(2 ln(2 n / delta))``: the bound that holds with probability ``1 - delta`` if the
    rounding errors are independent with mean zero (Higham & Mary 2019, Thm 2.4). Reported only: an order-free reduction
    (a GPU scatter-add of same-sign terms) need not satisfy the model."""
    return math.sqrt(2.0 * math.log(2.0 * n / delta)) * math.sqrt(n) * U


def fft_factor(n_fft):
    """Relative l2 error of a radix-2 FFT of ``n_fft`` points, ``log2(n) eta / (1 - log2(n) eta)`` (Higham 2002, Thm 24.2)."""
    t = math.log2(n_fft) * ETA_FFT
    return t / (1.0 - t)


class Case:
    """One data set: points, weights, data, the ddid and observation of each point, and the image layout."""

    def __init__(self, u, v, vis, weight, ddid, obs, npix, pixel_mas):
        self.u, self.v = np.asarray(u, float), np.asarray(v, float)
        self.vis, self.weight = np.asarray(vis, complex), np.asarray(weight, float)
        self.ddid, self.obs = np.asarray(ddid, int), np.asarray(obs, int)
        self.npix, self.pixel_mas = int(npix), float(pixel_mas)
        self.n_ddid, self.n_obs = int(self.ddid.max()) + 1, int(self.obs.max()) + 1
        self.n_d = np.bincount(self.ddid, minlength=self.n_ddid)
        self.obs_of_ddid = np.array([self.obs[self.ddid == d][0] for d in range(self.n_ddid)])
        assert all(np.all(self.obs[self.ddid == d] == self.obs_of_ddid[d]) for d in range(self.n_ddid)), "a ddid spans observations"


def _axis(case):
    """Pixel-centre coordinates (mas) along one axis: column c is East ``axis[c]``, row r is North ``axis[r]``
    (``radio_innovations.pixel_positions``)."""
    return ((case.npix - 1) / 2.0 - np.arange(case.npix)) * case.pixel_mas


def _factors(case, s, offset):
    """For the points ``s``: ``ex[j, c] = exp(-2 pi i u_j East_c)``, ``ey[j, r] = exp(-2 pi i v_j North_r)`` and the ramp. The pixel
    phase factorises, ``E_jp = ey[j, r] ex[j, c]``, so the pixel sum is ``sum_c ex[j, c] (ey @ b)[j, c]``: the same sum, by
    matrix products."""
    axis = _axis(case)
    ex = np.exp(-2j * np.pi * MAS * np.outer(case.u[s], axis))
    ey = np.exp(-2j * np.pi * MAS * np.outer(case.v[s], axis))
    off = np.asarray(offset, float)[case.obs[s]]
    ramp = np.exp(-2j * np.pi * MAS * (case.u[s] * off[:, 0] + case.v[s] * off[:, 1]))
    return ex, ey, ramp


def _by_ddid(case, chunk):
    for d in range(case.n_ddid):
        idx = np.flatnonzero(case.ddid == d)
        for i in range(0, len(idx), chunk):
            yield d, idx[i:i + chunk]


def model(case, b, flux, gain, offset_mas, chunk=8192):
    """``M_j`` for every point; ``gain`` and ``offset_mas`` list every observation, the reference first."""
    b = np.asarray(b, float).reshape(case.npix, case.npix)
    out = np.empty(len(case.u), complex)
    for _, s in _by_ddid(case, chunk):
        ex, ey, ramp = _factors(case, s, offset_mas)
        out[s] = flux * np.asarray(gain, float)[case.obs[s]] * ramp * np.sum(ex * (ey @ b), axis=1)
    return out


def chi2_by_ddid(case, m):
    """``chi2_d`` for every ddid, each an exactly rounded sum (``math.fsum``)."""
    terms = case.weight * np.abs(case.vis - m) ** 2
    return np.array([math.fsum(terms[case.ddid == d]) for d in range(case.n_ddid)])


def partials(case, values, chunk=8192):
    """One pass over the points at ``values`` (keys ``eta``, ``flux``, and ``gain``, ``offset`` of every observation): ``chi2_d``
    and, per ddid, ``d chi2_d / d theta`` for ``b`` (pixels), flux, gain and offset of every observation. Every error-scale mode's
    loss and gradient follow from these (``combine``)."""
    eta, flux = np.asarray(values["eta"], float), float(values["flux"])
    gain, offset = np.asarray(values["gain"], float), np.asarray(values["offset"], float)
    b = softmax_image(eta)
    n_dd, n_obs = case.n_ddid, case.n_obs
    terms = np.empty(len(case.u))
    d_b = np.zeros((n_dd, case.npix, case.npix))
    re_m = np.empty(len(case.u))
    re_u, re_v = np.empty(len(case.u)), np.empty(len(case.u))
    for d, s in _by_ddid(case, chunk):
        ex, ey, ramp = _factors(case, s, offset)
        scale = flux * gain[case.obs[s]] * ramp
        m = scale * np.sum(ex * (ey @ b), axis=1)
        r = case.vis[s] - m
        terms[s] = case.weight[s] * np.abs(r) ** 2
        a = case.weight[s] * np.conj(r) * scale  # d chi2 / d b_p = -2 Re sum_j a_j E_jp
        d_b[d] += -2.0 * np.real(ey.T @ (a[:, None] * ex))
        wm = case.weight[s] * np.conj(r) * m
        re_m[s] = np.real(wm)
        re_u[s] = np.real(wm * (-2j * np.pi * MAS * case.u[s]))
        re_v[s] = np.real(wm * (-2j * np.pi * MAS * case.v[s]))
    chi2 = np.array([math.fsum(terms[case.ddid == d]) for d in range(n_dd)])
    f_flux = np.array([-2.0 * math.fsum(re_m[case.ddid == d]) / flux for d in range(n_dd)])
    f_gain = np.zeros((n_dd, n_obs))
    f_off = np.zeros((n_dd, n_obs, 2))
    for d in range(n_dd):
        e, sel = case.obs_of_ddid[d], case.ddid == d
        f_gain[d, e] = -2.0 * math.fsum(re_m[sel]) / gain[e]
        f_off[d, e] = -2.0 * math.fsum(re_u[sel]), -2.0 * math.fsum(re_v[sel])
    return dict(chi2_d=chi2, b=b, d_b=d_b, flux=f_flux, gain=f_gain, offset=f_off)


def loss_from_chi2(chi2_d, n_d, mode, sigma=None):
    """The loss in an error-scale mode (module docstring) and ``dL/d chi2_d``."""
    if mode == "fixed":
        return 0.5 * math.fsum(chi2_d), np.full(len(chi2_d), 0.5)
    if mode == "fitted":
        s = np.asarray(sigma, float)
        return math.fsum(np.concatenate([chi2_d / (2 * s**2), 2 * n_d * np.log(s)])), 1.0 / (2 * s**2)
    if mode == "profiled":
        return math.fsum(n_d * (1.0 + np.log(chi2_d / (2.0 * n_d)))), n_d / chi2_d
    raise ValueError(mode)


def combine(case, part, mode, sigma=None):
    """Loss and gradient in an error-scale mode from ``partials``. ``grad`` has ``eta`` (image shape), ``flux``, ``gain`` and
    ``offset`` of every observation, and ``sigma`` in the fitted mode."""
    chi2 = part["chi2_d"]
    loss, dl = loss_from_chi2(chi2, case.n_d, mode, sigma)
    b = part["b"]
    d_b = np.tensordot(dl, part["d_b"], axes=1)
    grad = {"eta": b * (d_b - np.sum(b * d_b)),  # softmax Jacobian diag(b) - b b^T
            "flux": float(dl @ part["flux"]), "gain": dl @ part["gain"], "offset": np.tensordot(dl, part["offset"], axes=1)}
    if mode == "fitted":
        s = np.asarray(sigma, float)
        grad["sigma"] = -chi2 / s**3 + 2.0 * case.n_d / s
    return dict(loss=loss, chi2_d=chi2, grad=grad)


def evaluate(case, values, mode, chunk=8192):
    """``combine(partials(...))`` at ``values`` (``sigma`` (n_ddid) in the fitted mode)."""
    return combine(case, partials(case, values, chunk), mode, values.get("sigma"))


def sigma_hat(chi2_d, n_d):
    """The error scales at which ``dL/ds_d = 0`` for fixed everything else: ``s_d^2 = chi2_d / 2 N_d``."""
    return np.sqrt(chi2_d / (2.0 * n_d))


def strengths(case, flux, gain):
    """Per ddid: ``S_d = 2 g_d sum w|V| + g_d^2 sum w`` (``g_d`` = flux x gain of its observation) and ``C_d = sum w |V|^2``.
    ``S_d`` bounds every contraction of a unit-sum image with the dirty image and the dirty beam (``|D_p| <= sum w|V|``,
    ``|B_m| <= sum w``); ``C_d + S_d`` bounds ``sum w |r|^2`` and every partial sum of it (``|r| <= |V| + g``)."""
    g = flux * np.asarray(gain, float)[case.obs_of_ddid]
    sv = np.array([math.fsum((case.weight * np.abs(case.vis))[case.ddid == d]) for d in range(case.n_ddid)])
    sw = np.array([math.fsum(case.weight[case.ddid == d]) for d in range(case.n_ddid)])
    cc = np.array([math.fsum((case.weight * np.abs(case.vis) ** 2)[case.ddid == d]) for d in range(case.n_ddid)])
    return 2 * g * sv + g**2 * sw, cc, g


def phase_error(case, offset_mas):
    """Relative error of one evaluated ``exp(-i phi)`` term (with its ramp), ``(3 phi_max + 2) u``: ``phi`` carries a few roundings
    of relative size ``u``, each an absolute phase error ``u |phi|``, and the exponential one more (Higham 2002, section 1.12)."""
    r = 2 * np.max(np.abs(_axis(case))) + np.max(np.abs(np.asarray(offset_mas, float)).sum(axis=1))
    phi = 2 * np.pi * MAS * np.max(np.hypot(case.u, case.v)) * r  # |u x| + |v y| over both factors and the ramp
    return (3 * phi + 2) * U


def budget(case, values, mode, eps_impl, n_fft, eps_ref=None, worst_case=True, ref=None):
    """Allowed |implementation - reference| per quantity at ``values`` (design section 3).

    ``eps_impl``: the implementation's documented NUFFT tolerance against the l1 norm of its strengths (Toeplitz: kernel eps plus
    the psi-hat quadrature error; direct: its NUFFT eps). ``n_fft``: the size of the largest FFT the implementation convolves the
    image with (``(2 N)^2`` for the Toeplitz beam; 0 for a direct sum). ``eps_ref``: the reference's own per-term error (default
    this module's exact sum: ``phase_error`` plus ``gamma(P)``; a NUFFT reference passes its eps). ``worst_case=False`` uses the
    probabilistic accumulation bound instead of ``gamma_n`` (reported, never gated). ``ref``: this module's ``combine`` output at
    ``values`` (computed if not given). Returns ``tolerances(...)``."""
    flux, gain = float(values["flux"]), np.asarray(values["gain"], float)
    if eps_ref is None:
        eps_ref = phase_error(case, values["offset"]) + gamma(case.npix**2)
    ref = ref or evaluate(case, values, mode)
    return tolerances(stats(case, flux, gain), ref["chi2_d"], ref["loss"], mode, values.get("sigma"), flux, gain, eps_impl, n_fft,
                      eps_ref, worst_case)


def stats(case, flux, gain):
    """What a budget needs of the data: ``strengths``, the points per ddid, the observation of each ddid, ``q_max``, pixels."""
    s_d, c_d, _ = strengths(case, flux, gain)
    return dict(s_d=s_d, c_d=c_d, n_d=case.n_d, obs_of_ddid=case.obs_of_ddid, n_obs=case.n_obs, p=case.npix**2,
                q_max=float(np.max(np.hypot(case.u, case.v))))


def tolerances(st, chi2_d, loss, mode, sigma, flux, gain, eps_impl, n_fft, eps_ref, worst_case=True):
    """The budget from ``stats`` ``st`` and the reference's ``chi2_d`` (at unit error scales) and ``loss``.

    Per ddid: ``tau_d = (eps_impl + 2 eps_ref + 2 g_acc + rho) S_d + 2 g_acc C_d``, with ``g_acc = gamma(n)`` for the longest
    reduction ``n = max(N_d, P, n_fft)`` (both sides: hence 2) and ``rho = 3 fft_factor(n_fft) sqrt(n_fft)`` (a convolution by
    three FFTs; the l2 error of the convolved beam, at most ``sqrt(n_fft)`` times its l-infinity norm ``sum w``, contracted with
    a unit-sum image, l2 norm <= 1). The reference enters twice (``|delta chi2| <= 2 sum w |r| |delta M|`` and
    ``sum w|r| <= S_d / g``). Gradients: ``d chi2_d / d theta`` has the same structure times the derivative's factor: ``2 / F``
    (flux), ``2 / g_e`` (gain), ``2 x 2 pi MAS q_max`` (offset, rad per mas), ``2`` per pixel of ``b`` (l2 over pixels:
    ``sqrt(P)``; the softmax Jacobian has norm <= 1). The loss and gradient bounds weight each ddid by ``|dL/d chi2_d|`` (to first
    order in ``tau`` where it moves with chi2, the profiled mode); the fitted ``dL/ds_d = -chi2_d/s^3 + 2N/s`` has error
    ``tau_d / s^3 + u (chi2_d / s^3 + 2 N_d / s)``. Returns ``dict(chi2_d, loss, grad)``, the gradient ones as l2 norms per site
    (per element for ``sigma``)."""
    s_d, c_d, n_d, p = st["s_d"], st["c_d"], np.asarray(st["n_d"]), st["p"]
    n = max(int(n_d.max()), p, int(n_fft))
    acc = gamma(n) if worst_case else gamma_probabilistic(n)
    rho = 3 * fft_factor(n_fft) * math.sqrt(n_fft) if n_fft else 0.0
    tau = (eps_impl + 2 * eps_ref + 2 * acc + rho) * s_d + 2 * acc * c_d
    _, dl = loss_from_chi2(np.asarray(chi2_d, float), n_d, mode, sigma)
    if mode == "profiled":
        dl = dl * (1 + tau / chi2_d)
    wt = np.abs(dl) * tau
    obs_of_ddid, n_obs = np.asarray(st["obs_of_ddid"]), st["n_obs"]
    per_obs = np.array([np.sum(wt[obs_of_ddid == e]) for e in range(n_obs)])
    gain = np.asarray(gain, float)
    tol = {"chi2_d": tau, "loss": float(np.sum(wt)) + U * abs(loss)}
    tol["grad"] = {"eta": math.sqrt(p) * 2 * float(np.sum(wt)), "flux": 2 * float(np.sum(wt)) / flux,
                   "gain": float(np.linalg.norm(2 * per_obs[1:] / gain[1:])),
                   "offset": float(np.linalg.norm(2 * 2 * np.pi * MAS * st["q_max"] * np.repeat(per_obs[1:], 2)))}
    if mode == "fitted":
        s = np.asarray(sigma, float)
        tol["grad"]["sigma"] = tau / s**3 + U * (np.asarray(chi2_d) / s**3 + 2 * n_d / s)
    return tol
