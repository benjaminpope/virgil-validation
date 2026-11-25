"""Neutral NumPy reference scorers for the paper-performance benchmarks.

Imports neither virgil nor PMOIRED / Octofitter / eht-imaging: NumPy (and SciPy for J1 only). Every function is written
from the definitions quoted in its docstring, so each code's likelihood or objective can be compared against the same
independent implementation (plan section 1, "Gates use a neutral reference").

Functions
---------
``binary_vis``         complex visibility of a point (or uniform-disk) primary plus a point companion.
``chi2_v2_cp``         independent-Gaussian chi-squared of V2 and closure phases, wrapped closure-phase residual.
``astrometry_loglike`` bivariate-Gaussian log-likelihood of relative astrometry (Level A of the orbit comparison).
``image_objective``    ``J = 1/2 sum r_amp^2 + 1/2 sum r_cp^2 + beta * sum (db)^2`` on the unit-flux image.

Conventions (as in ``docs/method/ehtim.md`` and ``crosscheck``): x = East, y = North, row 0 of an image is North and
column 0 is East; the model visibility is exp(-2 pi i (u x + v y)) (``sign=-1``); eht-imaging's is the conjugate
(``sign=+1``), and its pixel centres are those of ``ehtim.image``: ``x_j = (n/2 - j - 1/2) * pixel``.

Self-test: ``~/.claude/bin/pyquick $BENCH_PYTHON benchmarks/paper_performance/neutral.py --selftest``.
"""

from __future__ import annotations

import sys

import numpy as np

MAS = np.pi / 180.0 / 3600.0 / 1000.0  # milliarcsecond in radians

# ----------------------------------------------------------------------------------------------------------------
# visibilities and interferometric chi-squared
# ----------------------------------------------------------------------------------------------------------------


def binary_vis(u, v, wavel, dra_mas, ddec_mas, flux_ratio, ud_mas=0.0, sign=-1):
    """Complex visibility of a primary at the origin and a point companion at (dra, ddec) [mas, East/North].

    ``V = (V_1 + f exp(sign * 2 pi i (u x + v y) / lambda)) / (1 + f)``, with ``f`` the companion/primary flux ratio and
    ``V_1 = 2 J_1(pi d rho) / (pi d rho)`` for a uniform-disk primary of diameter ``d`` (1 for a point). ``u``, ``v`` in
    metres, ``wavel`` in metres (broadcast together).
    """
    u, v, wavel = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float), np.asarray(wavel, float))
    x, y = dra_mas * MAS, ddec_mas * MAS
    v1 = np.ones(u.shape, complex)
    if ud_mas:
        from scipy.special import j1

        arg = np.pi * ud_mas * MAS * np.hypot(u, v) / wavel
        with np.errstate(invalid="ignore", divide="ignore"):
            v1 = np.where(arg == 0, 1.0, 2.0 * j1(arg) / np.where(arg == 0, 1.0, arg)).astype(complex)
    phase = sign * 2j * np.pi * (u * x + v * y) / wavel
    return (v1 + flux_ratio * np.exp(phase)) / (1.0 + flux_ratio)


def closure_phase(vis1, vis2, vis3):
    """Closure phase (radians) as the sum of the three baseline phases, ``arg(V1 V2 V3)`` in (-pi, pi]."""
    return np.angle(vis1 * vis2 * vis3)


def wrap(x):
    """Wrap angles to (-pi, pi]."""
    return np.pi - np.mod(np.pi - np.asarray(x, float), 2.0 * np.pi)


def chi2_v2_cp(v2_data, v2_err, v2_model, cp_data=None, cp_err=None, cp_model=None):
    """Independent-Gaussian chi-squared with a wrapped closure-phase residual.

    ``chi2_v2 = sum ((V2_model - V2_data) / sigma)^2``;
    ``chi2_cp = sum (wrap(CP_model - CP_data) / sigma_cp)^2`` with angles in **radians**. Returns a dict with
    ``chi2``, ``chi2_v2``, ``chi2_cp``, ``n_v2``, ``n_cp``. This is PMOIRED's form (wrapped difference) and, for small
    residuals, Octofitter's and virgil's with independent closure-phase errors; the sine-chord and baseline-noise
    differences are the documented mismatches (plan R1).
    """
    v2_data, v2_err, v2_model = (np.asarray(a, float) for a in (v2_data, v2_err, v2_model))
    c_v2 = float(np.sum(((v2_model - v2_data) / v2_err) ** 2))
    c_cp, n_cp = 0.0, 0
    if cp_data is not None:
        cp_data, cp_err, cp_model = (np.asarray(a, float) for a in (cp_data, cp_err, cp_model))
        c_cp = float(np.sum((wrap(cp_model - cp_data) / cp_err) ** 2))
        n_cp = cp_data.size
    return {"chi2": c_v2 + c_cp, "chi2_v2": c_v2, "chi2_cp": c_cp, "n_v2": v2_data.size, "n_cp": n_cp}


# ----------------------------------------------------------------------------------------------------------------
# astrometry
# ----------------------------------------------------------------------------------------------------------------


def astrometry_loglike(dra, ddec, model_dra, model_ddec, sig_ra, sig_dec, cor=0.0):
    """Log-likelihood of relative astrometry, a bivariate Gaussian per epoch (full normalisation).

    ``-1/2 [ (z_ra^2 - 2 rho z_ra z_dec + z_dec^2) / (1 - rho^2) ] - ln(2 pi sig_ra sig_dec sqrt(1 - rho^2))`` summed
    over epochs, with ``z = (data - model) / sigma``. All arrays are per epoch; ``cor`` is the correlation coefficient.
    """
    dra, ddec, mra, mdec, sa, sd = (np.asarray(a, float) for a in (dra, ddec, model_dra, model_ddec, sig_ra, sig_dec))
    rho = np.broadcast_to(np.asarray(cor, float), sa.shape)
    za, zd = (dra - mra) / sa, (ddec - mdec) / sd
    q = (za**2 - 2.0 * rho * za * zd + zd**2) / (1.0 - rho**2)
    ll = -0.5 * q - np.log(2.0 * np.pi * sa * sd * np.sqrt(1.0 - rho**2))
    return float(np.sum(ll))


# ----------------------------------------------------------------------------------------------------------------
# imaging
# ----------------------------------------------------------------------------------------------------------------


def pixel_grid(npix, pixel_mas):
    """Pixel-centre offsets (x East, y North) in radians for an ``npix`` x ``npix`` image, flattened row-major.

    Row 0 is North and column 0 is East (eht-imaging's ``xlist``/``ylist``): ``x_j = (n/2 - j - 1/2) * pixel``.
    """
    c = (npix / 2.0 - np.arange(npix) - 0.5) * pixel_mas * MAS
    xx, yy = np.meshgrid(c, c)  # xx varies along columns, yy along rows
    return xx.ravel(), yy.ravel()


def dft_matrix(u, v, wavel, x, y, sign=+1):
    """Direct Fourier matrix ``A[k, p] = exp(sign * 2 pi i (u_k x_p + v_k y_p) / lambda_k)`` (``sign=+1`` is
    eht-imaging's ``ftmatrix``, ``-1`` virgil's)."""
    u, v, wavel = (np.asarray(a, float).ravel() for a in np.broadcast_arrays(u, v, wavel))
    return np.exp(sign * 2j * np.pi * (np.outer(u / wavel, x) + np.outer(v / wavel, y)))


def tv2_cost(image, edges="ehtim"):
    """``sum (db)^2`` over neighbouring pixels, ``image`` a 2-D array.

    ``edges="ehtim"``: eht-imaging's ``-stv2`` with ``norm_reg=False``: the image is padded with zeros and each pixel
    contributes ``(b[i+1, j] - b[i, j])^2 + (b[i, j+1] - b[i, j])^2``, so the steps out of the last row and column into
    the padding count and the steps from the padding into the first row and column do not.
    ``edges="virgil"``: the same plus ``sum b[0, :]^2 + sum b[:, 0]^2`` (``TSV``'s extra edge terms, vv ledger D9).
    """
    b = np.asarray(image, float)
    pad = np.pad(b, ((0, 1), (0, 1)))
    out = np.sum((pad[1:, :-1] - b) ** 2 + (pad[:-1, 1:] - b) ** 2)
    if edges == "virgil":
        out += np.sum(b[0, :] ** 2) + np.sum(b[:, 0] ** 2)
    elif edges != "ehtim":
        raise ValueError("edges must be 'ehtim' or 'virgil'")
    return float(out)


def image_terms(image, data, sign=+1):
    """Data terms of ``J`` for a 2-D image (any total flux; it is normalised to 1 first).

    ``data`` keys: ``npix``, ``pixel_mas``, ``wavel_amp``/``u_amp``/``v_amp`` (baselines in metres) with ``amp`` and
    ``sigma_amp``; for closure phases ``u1,v1,u2,v2,u3,v3,wavel_cp`` (three baselines of each triangle, summing to
    zero) with ``cphase_deg`` and ``sigma_cp_deg``. Returns ``half_chi2_amp``, ``half_chi2_cp``, ``tv2`` and ``flux``.
    """
    b = np.asarray(image, float)
    flux = float(b.sum())
    b = b / flux
    x, y = pixel_grid(data["npix"], data["pixel_mas"])
    flat = b.ravel()
    out = {"flux": flux, "half_chi2_amp": 0.0, "half_chi2_cp": 0.0}
    if "amp" in data:
        A = dft_matrix(data["u_amp"], data["v_amp"], data["wavel_amp"], x, y, sign)
        r = (data["amp"] - np.abs(A @ flat)) / data["sigma_amp"]
        out["half_chi2_amp"] = float(0.5 * np.sum(r**2))
    if "cphase_deg" in data:
        w = data["wavel_cp"]
        vis = [dft_matrix(data[f"u{i}"], data[f"v{i}"], w, x, y, sign) @ flat for i in (1, 2, 3)]
        delta = np.deg2rad(data["cphase_deg"]) - np.angle(vis[0] * vis[1] * vis[2])
        out["half_chi2_cp"] = float(np.sum((1.0 - np.cos(delta)) / np.deg2rad(data["sigma_cp_deg"]) ** 2))
    out["tv2"] = tv2_cost(b)
    return out


def image_objective(image, data, beta, sign=+1, edges="ehtim"):
    """The common score ``J = 1/2 sum r_amp^2 + 1/2 sum r_cp^2 + beta * sum (db)^2`` on the unit-flux image.

    ``1/2 r_cp^2 = (1 - cos(Delta)) / sigma^2`` is the sine-chord form; with ``alpha_d = N_d / 2`` it equals
    eht-imaging's ``alpha_d (chi2_d - 1)`` up to the constant. ``edges`` selects the difference-term convention
    (``tv2_cost``); the benchmark scores both codes' images with ``"ehtim"``.
    """
    t = image_terms(image, data, sign)
    tv = tv2_cost(np.asarray(image, float) / t["flux"], edges)
    return t["half_chi2_amp"] + t["half_chi2_cp"] + beta * tv


# ----------------------------------------------------------------------------------------------------------------
# self-test
# ----------------------------------------------------------------------------------------------------------------


def _selftest() -> int:
    rng = np.random.default_rng(0)
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"{'ok  ' if cond else 'FAIL'} {name}")
        ok &= bool(cond)

    # binary_vis: zero baseline, V^2 depends on separation, ud limit
    check("binary_vis V(0)=1", np.allclose(binary_vis(0.0, 0.0, 2e-6, 6.0, -4.0, 0.1), 1.0))
    u, v, wl = 40.0, -25.0, 2e-6
    f, x, y = 0.1, 6.0, -4.0
    vis = binary_vis(u, v, wl, x, y, f)
    expect = (1 + f * np.exp(-2j * np.pi * (u * x + v * y) * MAS / wl)) / (1 + f)
    check("binary_vis closed form", abs(vis - expect) < 1e-14)
    check("binary_vis conjugate sign", abs(binary_vis(u, v, wl, x, y, f, sign=+1) - np.conj(vis)) < 1e-14)
    check("binary_vis ud small-disk limit", abs(binary_vis(u, v, wl, x, y, f, ud_mas=1e-6) - vis) < 1e-12)
    ud = binary_vis(30.0, 0.0, 2e-6, x, y, 0.0, ud_mas=1.0)
    arg = np.pi * 1.0 * MAS * 30.0 / 2e-6
    from scipy.special import j1

    check("binary_vis uniform disk", abs(ud - 2 * j1(arg) / arg) < 1e-14)

    # chi2_v2_cp: zero at truth, known value, wrapping
    v2 = rng.uniform(0.2, 1, 7)
    cp = rng.uniform(-3, 3, 5)
    r = chi2_v2_cp(v2, 0.03 * np.ones(7), v2, cp, np.full(5, 0.02), cp)
    check("chi2 zero at truth", r["chi2"] == 0.0 and r["n_v2"] == 7 and r["n_cp"] == 5)
    r = chi2_v2_cp([0.5], [0.1], [0.6], [3.1], [0.1], [-3.1])  # CP residual is -6.2 -> wraps to 2pi - 6.2
    want = 1.0 + ((2 * np.pi - 6.2) / 0.1) ** 2
    check("chi2 wrapped closure phase", abs(r["chi2"] - want) < 1e-9)
    check("wrap range", np.all(np.abs(wrap(rng.uniform(-30, 30, 100))) <= np.pi + 1e-12))

    # astrometry_loglike vs the analytic bivariate normal
    d = rng.normal(size=(3, 2))
    sa, sd, rho = np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.1, 0.4]), np.array([0.0, 0.5, -0.3])
    ll = astrometry_loglike(d[:, 0], d[:, 1], 0 * sa, 0 * sa, sa, sd, rho)
    ref = 0.0
    for k in range(3):
        cov = np.array([[sa[k] ** 2, rho[k] * sa[k] * sd[k]], [rho[k] * sa[k] * sd[k], sd[k] ** 2]])
        ref += -0.5 * d[k] @ np.linalg.solve(cov, d[k]) - 0.5 * np.log(np.linalg.det(2 * np.pi * cov))
    check("astrometry_loglike bivariate normal", abs(ll - ref) < 1e-12)
    check("astrometry_loglike zero correlation scalar", np.isfinite(astrometry_loglike([0.1], [0.2], [0], [0], [0.1], [0.1])))

    # imaging: delta pixel at the grid centre-ish, DFT sign, tv2 conventions, J consistency
    n, pix = 8, 0.5
    x, y = pixel_grid(n, pix)
    check("pixel_grid row0 North col0 East", y[0] > 0 and x[0] > 0 and x[n - 1] < 0 and y[-1] < 0)
    A = dft_matrix([10.0], [5.0], [2e-6], x, y, +1)
    check("dft_matrix sign", np.allclose(A, np.conj(dft_matrix([10.0], [5.0], [2e-6], x, y, -1))))
    b = rng.uniform(0.1, 1, (n, n))
    loop = 0.0
    for i in range(n):
        for j in range(n):
            down = b[i + 1, j] if i + 1 < n else 0.0
            right = b[i, j + 1] if j + 1 < n else 0.0
            loop += (down - b[i, j]) ** 2 + (right - b[i, j]) ** 2
    check("tv2_cost ehtim loop", abs(tv2_cost(b) - loop) < 1e-12)
    check("tv2_cost virgil edges", abs(tv2_cost(b, "virgil") - loop - np.sum(b[0] ** 2) - np.sum(b[:, 0] ** 2)) < 1e-12)

    img = rng.uniform(0.1, 1, (n, n))
    unit = img / img.sum()
    nb = 12
    uu, vv = rng.uniform(-40, 40, (2, nb))
    data = {"npix": n, "pixel_mas": pix, "u_amp": uu, "v_amp": vv, "wavel_amp": np.full(nb, 2e-6)}
    data["amp"] = np.abs(dft_matrix(uu, vv, data["wavel_amp"], *pixel_grid(n, pix)) @ unit.ravel())
    data["sigma_amp"] = np.full(nb, 0.01)
    u1, v1, u2, v2 = rng.uniform(-30, 30, (4, 6))
    data.update(u1=u1, v1=v1, u2=u2, v2=v2, u3=-(u1 + u2), v3=-(v1 + v2), wavel_cp=np.full(6, 2e-6))
    vis = [dft_matrix(data[f"u{i}"], data[f"v{i}"], data["wavel_cp"], x, y) @ unit.ravel() for i in (1, 2, 3)]
    data["cphase_deg"] = np.rad2deg(np.angle(vis[0] * vis[1] * vis[2]))
    data["sigma_cp_deg"] = np.full(6, 1.0)
    check("image_objective data terms zero at truth",
          abs(image_objective(img, data, beta=0.0)) < 1e-9 and abs(image_objective(3.0 * img, data, 0.0)) < 1e-9)
    beta = 7.5e5
    check("image_objective adds beta * tv2",
          abs(image_objective(img, data, beta) - beta * tv2_cost(unit)) < 1e-6 * beta * tv2_cost(unit))
    pert = unit.copy()
    pert[3, 3] *= 1.5
    t = image_terms(pert, data)
    check("image_terms positive for a perturbed image", t["half_chi2_amp"] > 0 and t["half_chi2_cp"] > 0)
    # half chi2 cp equals 1/2 of the sine-chord chi2: 2(1-cos d)/s^2 / 2
    vis = [dft_matrix(data[f"u{i}"], data[f"v{i}"], data["wavel_cp"], x, y) @ (pert / pert.sum()).ravel() for i in (1, 2, 3)]
    delta = np.deg2rad(data["cphase_deg"]) - np.angle(vis[0] * vis[1] * vis[2])
    chord = 2.0 * np.sin(delta / 2.0) / np.deg2rad(1.0)
    check("image half_chi2_cp equals half the chord chi2", abs(t["half_chi2_cp"] - 0.5 * np.sum(chord**2)) < 1e-8)

    print("selftest passed" if ok else "selftest FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print(__doc__)
