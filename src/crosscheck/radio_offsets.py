"""Brute-force reference for the free phase-centre offsets of a one-image radio chi-squared (register B5).

numpy only; nothing is taken from virgil. The model of point ``j`` in observation ``e`` is

    M_j = F g_e exp(-2 pi i (u_j dRA_e + v_j dDec_e)) sum_p b_p exp(-2 pi i (u_j East_p + v_j North_p)),

``b = softmax(eta)``, with ``chi2 = sum_j w_j |V_j - M_j|^2``. The phase ramp is the exact shift theorem
(Bracewell, *The Fourier Transform and its Applications*, 3rd ed., 2000, ch. 6) and the sum over pixels is the
definition of the pixel transform, so nothing here is approximate. Written for this repository from those
formulae. Provided here: the chi-squared and its analytic gradient (checked against central differences in the
tests), the Fisher information of the nuisance parameters, and a simulator.
"""

import numpy as np

from crosscheck.radio_innovations import MAS, dft_matrix, pixel_positions, softmax_image


def model_visibilities(u, v, obs, eta, pixel_mas, flux, gain, offset_mas):
    """``M_j`` of the formula above; ``gain`` and ``offset_mas`` list every observation, the reference first.

    Returns ``(M, E b)``: the model and the unshifted, unit-flux transform of the normalised image."""
    east, north = pixel_positions(np.shape(eta)[0], pixel_mas)
    scene = dft_matrix(u, v, east, north) @ softmax_image(eta).ravel()
    off = np.asarray(offset_mas, float)[obs]
    ramp = np.exp(-2j * np.pi * MAS * (u * off[:, 0] + v * off[:, 1]))
    return flux * np.asarray(gain, float)[obs] * ramp * scene, scene


def chi2_and_gradient(u, v, vis, weight, obs, eta, pixel_mas, flux, gain, offset_mas):
    """``(chi2, grads)``: the total chi-squared and its derivatives with respect to ``eta`` (image shape),
    ``flux`` (scalar), ``gain`` (every observation) and ``offset`` (every observation, mas; shape ``(n_obs, 2)``).

    With ``r = V - M``: ``d chi2 / d theta = -2 sum_j w_j Re(conj(r_j) dM_j/d theta)``, where
    ``dM/db_p = F g s E_jp``, ``dM/dg_e = M/g_e``, ``dM/dF = M/F`` and ``dM/d dRA_e = -2 pi i MAS u M``
    (``dDec`` likewise with ``v``); ``d b / d eta = diag(b) - b b^T``."""
    gain = np.asarray(gain, float)
    offset = np.asarray(offset_mas, float)
    n_obs = len(gain)
    east, north = pixel_positions(np.shape(eta)[0], pixel_mas)
    transform = dft_matrix(u, v, east, north)
    b = softmax_image(eta).ravel()
    scene = transform @ b
    ramp = np.exp(-2j * np.pi * MAS * (u * offset[obs, 0] + v * offset[obs, 1]))
    scale = flux * gain[obs] * ramp
    model = scale * scene
    resid = vis - model
    chi2 = float(np.sum(weight * np.abs(resid) ** 2))
    wr = weight * resid.conj()  # d chi2 = -2 Re(wr dM)
    # Re(wr s E) = Re(wr s) E_re - Im(wr s) E_im:
    d_b = -2.0 * (np.real(wr * scale) @ transform.real - np.imag(wr * scale) @ transform.imag)
    d_eta = b * (d_b - b @ d_b)
    grads = {"eta": d_eta.reshape(np.shape(eta)),
             "flux": float(-2.0 * np.sum(np.real(wr * model)) / flux),
             "gain": np.array([-2.0 * np.sum(np.real(wr * model)[obs == e]) / gain[e] for e in range(n_obs)]),
             "offset": np.zeros((n_obs, 2))}
    for axis, freq in enumerate((u, v)):
        dm = -2j * np.pi * MAS * freq * model
        for e in range(n_obs):
            grads["offset"][e, axis] = -2.0 * np.sum(np.real(wr * dm)[obs == e])
    return chi2, grads


def nuisance_fisher(u, v, weight, obs, eta, pixel_mas, flux, gain, offset_mas, ref_obs=0):
    """Fisher information of ``(flux, gain_e, dRA_e, dDec_e)`` for the non-reference observations ``e`` at the
    truth, for ``chi2 = sum w |V - M|^2`` with exp(-chi2/2) the likelihood (each of Re and Im has variance
    ``1/w``): ``F_ab = sum_j w_j Re(conj(dM_j/da) dM_j/db)``, from the derivatives of ``M`` listed in
    ``chi2_and_gradient``. Parameter order: flux, gains of the other observations, then (dRA, dDec) of each."""
    gain = np.asarray(gain, float)
    offset = np.asarray(offset_mas, float)
    model, _ = model_visibilities(u, v, obs, eta, pixel_mas, flux, gain, offset)
    others = [e for e in range(len(gain)) if e != ref_obs]
    columns = [model / flux]
    for e in others:
        columns.append(np.where(obs == e, model / gain[e], 0.0))
    for e in others:
        for freq in (u, v):
            columns.append(np.where(obs == e, -2j * np.pi * MAS * freq * model, 0.0))
    jac = np.array(columns).T  # (n_points, n_params), complex
    return np.real((jac.conj().T * weight) @ jac)
