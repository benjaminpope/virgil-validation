"""NumPy-only references for the virgil.radio trust checks (``tests/test_radio_trust_*.py``).

Nothing here imports virgil. Each function is written from the formula in its docstring, not from
virgil's implementation.

Conventions (docs/alma_conventions.md): ``V(u, v) = sum I(x, y) exp(-2 pi i (u x + v y))``, x East and
y North, u and v in wavelengths. A pixel image has row 0 North and column 0 East, and the phase centre
sits at index ``(npix - 1) / 2`` on both axes, so on an even grid it lies between four pixels (the
convention virgil documents for ``Image``): the East offset of column c is ``((npix - 1) / 2 - c) * pixel``
and the North offset of row r is ``((npix - 1) / 2 - r) * pixel``.
"""

import numpy as np

MAS = np.pi / 180.0 / 3600.0 / 1000.0  # radians per milliarcsecond


def offsets_rad(npix, pixel_mas):
    """East (columns) and North (rows) offsets of the pixel centres, radians."""
    k = ((npix - 1) / 2.0 - np.arange(npix)) * pixel_mas * MAS
    return k, k


def dirty(u, v, a, w, npix, pixel_mas):
    """``sum_k w_k Re[a_k exp(+2 pi i (u_k x + v_k y))] / sum_k w_k`` on the pixel grid (Thompson, Moran
    and Swenson, ch. 5); ``a = V`` gives the dirty image, ``a = 1`` the dirty beam."""
    x, y = offsets_rad(npix, pixel_mas)
    u, v, w = (np.asarray(t, float) for t in (u, v, w))
    a = np.asarray(a, complex)
    out = np.zeros((npix, npix))
    for lo in range(0, len(u), 2000):
        s = slice(lo, lo + 2000)
        fx = np.exp(2j * np.pi * np.outer(u[s], x))
        fy = np.exp(2j * np.pi * np.outer(v[s], y))
        out += ((fy * (w[s] * a[s])[:, None]).T @ fx).real
    return out / w.sum()


def density_weights(u, v, w, cell, kind, robust=0.5):
    """Imaging weights of Briggs (1995, thesis ch. 3), by explicit loops over uv cells.

    The uv plane is cut into square cells of side ``cell`` (wavelengths) and Hermitian partners share a
    cell (the point is reflected to v > 0, or u > 0 on the u axis). With ``W_c`` the summed natural weight
    of cell c: uniform ``w_k / W_c``; Briggs ``w_k / (1 + W_c f^2)``, ``f^2 = (5 10^-R)^2 / (sum_c W_c^2 /
    sum_k w_k)``."""
    u, v, w = (np.asarray(t, float) for t in (u, v, w))
    keys = []
    for uk, vk in zip(u, v):
        if vk < 0 or (vk == 0 and uk < 0):
            uk, vk = -uk, -vk
        keys.append((int(np.floor(uk / cell)), int(np.floor(vk / cell))))
    total = {}
    for key, wk in zip(keys, w):
        total[key] = total.get(key, 0.0) + wk
    cw = np.array([total[k] for k in keys])
    if kind == "uniform":
        return w / cw
    f2 = (5.0 * 10.0 ** (-robust)) ** 2 / (sum(t * t for t in total.values()) / w.sum())
    return w / (1.0 + cw * f2)


def uv_taper(u, v, bmaj_mas, bmin_mas, pa_deg):
    """Fourier transform of an elliptical Gaussian of FWHM ``bmaj`` along the position angle ``pa`` (East of
    North) and ``bmin`` across it: ``exp(-pi^2 (bmaj^2 q_par^2 + bmin^2 q_perp^2) / (4 ln 2))``, where ``q_par``
    is the component of (u, v) along the major axis direction (sin pa, cos pa) in (East, North)."""
    pa = np.deg2rad(pa_deg)
    par = u * np.sin(pa) + v * np.cos(pa)
    perp = u * np.cos(pa) - v * np.sin(pa)
    a, b = bmaj_mas * MAS, bmin_mas * MAS
    return np.exp(-(np.pi**2) * ((a * par) ** 2 + (b * perp) ** 2) / (4.0 * np.log(2.0)))


def softmax(eta):
    e = np.exp(eta - np.max(eta))
    return e / e.sum()


def image_model(eta, pixel_mas, u, v, flux=1.0):
    """``flux * sum_pix softmax(eta) exp(-2 pi i (u x + v y))`` for a pixel image (unit-sum brightness)."""
    b = softmax(np.asarray(eta, float))
    x, y = offsets_rad(b.shape[1], pixel_mas)
    fx = np.exp(-2j * np.pi * np.outer(u, x))
    fy = np.exp(-2j * np.pi * np.outer(v, y))
    return flux * np.einsum("kr,rc,kc->k", fy, b, fx)


def gain_marginal_brute_force(model, vis, w, a1, a2, n_ant, tau_a, tau_p, nodes=11, prior_scale=1.0):
    """``-log`` of the gain-marginalised likelihood of one solution interval, by quadrature.

    Model: ``V_k ~ g_i conj(g_j) M_k`` with ``g = exp(lam + i phi)`` per antenna, ``lam ~ N(0, tau_a^2)`` and
    ``phi ~ N(0, tau_p^2)`` independent, and each of Re and Im Gaussian with weight w (variance 1/w). The
    quantity is ``-log integral exp(-chi2/2) N(lam) N(phi) d lam d phi`` with ``chi2 = sum w |V - model|^2``
    and the constant ``-sum log(w / 2 pi)`` of the data left out (it does not depend on the gains).

    The integral is a tensor-product Gauss-Hermite rule in coordinates whitened by the numerically
    differentiated Hessian at the numerically found mode, so it is exact to the quadrature order and shares
    nothing with the Laplace formula it is compared with. ``prior_scale`` multiplies both prior widths (a
    deliberately wrong reference for the control)."""
    from scipy.optimize import minimize

    p = 2 * n_ant
    ta, tp = tau_a * prior_scale, tau_p * prior_scale
    prior_prec = np.concatenate([np.full(n_ant, 1 / ta**2), np.full(n_ant, 1 / tp**2)])

    def f(x):
        """-log of the unnormalised integrand; x of shape (M, p)."""
        lam, phi = x[:, :n_ant], x[:, n_ant:]
        g = np.exp(lam[:, a1] + lam[:, a2] + 1j * (phi[:, a1] - phi[:, a2])) * model
        chi2 = (w * np.abs(vis - g) ** 2).sum(axis=1)
        return 0.5 * chi2 + 0.5 * (x**2 * prior_prec).sum(axis=1)

    mode = minimize(lambda x: f(x[None])[0], np.zeros(p), method="BFGS", options=dict(gtol=1e-10)).x
    h = 1e-4
    hess = np.empty((p, p))
    for i in range(p):
        for j in range(p):
            d = np.zeros((4, p))
            d[0, i] += h
            d[0, j] += h
            d[1, i] += h
            d[1, j] -= h
            d[2, i] -= h
            d[2, j] += h
            d[3, i] -= h
            d[3, j] -= h
            fv = f(mode + d)
            hess[i, j] = (fv[0] - fv[1] - fv[2] + fv[3]) / (4 * h * h)
    chol = np.linalg.cholesky(0.5 * (hess + hess.T))
    t, wt = np.polynomial.hermite.hermgauss(nodes)
    grid = np.stack(np.meshgrid(*[t] * p, indexing="ij"), axis=-1).reshape(-1, p)
    weights = np.prod(np.stack(np.meshgrid(*[wt] * p, indexing="ij"), axis=-1), axis=-1).reshape(-1)
    total = 0.0
    f_mode = f(mode[None])[0]
    for lo in range(0, len(grid), 200_000):
        z = grid[lo:lo + 200_000] * np.sqrt(2.0)
        x = mode + np.linalg.solve(chol.T, z.T).T  # x = mode + L^-T y, y = sqrt(2) t
        # integrand exp(-f) divided by the Gaussian weight exp(-|y|^2 / 2) that the Hermite rule carries
        g = np.exp(-(f(x) - f_mode) + 0.5 * (z**2).sum(axis=1))
        total += (weights[lo:lo + 200_000] * g).sum()
    log_int = -f_mode - np.log(np.prod(np.diag(chol))) + 0.5 * p * np.log(2.0) + np.log(total)
    # prior normalisation: N(lam) N(phi) = exp(-prior quad) / ((2 pi)^(p/2) prod tau)
    log_norm = -0.5 * p * np.log(2 * np.pi) - n_ant * (np.log(ta) + np.log(tp))
    return -(log_int + log_norm), mode
