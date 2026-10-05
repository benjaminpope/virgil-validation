"""A rapidly rotating star, written from Espinosa Lara & Rieutord (2011,
A&A 533, A43; ELR11), not from virgil.

Units: the equatorial radius R_eq = 1 and GM = 1, so the Keplerian rate at
the equator is 1 and omega = Omega / Omega_K.

* Shape (Roche, a point mass): the surface is the equipotential through the
  equator, 1/r + omega² r² sin²θ / 2 = 1 + omega² / 2, a cubic in r solved
  here for its root in (0, 1].
* Effective gravity: g = -grad(Phi), Phi = -1/r - omega² r² sin²θ / 2, so
  g_r = -1/r² + omega² r sin²θ and g_θ = omega² r sinθ cosθ.
* Flux (ELR11, their eqs. 24 and 26): F = |g| tan²ϑ / tan²θ (up to a
  constant), with ϑ the root of
  cos ϑ + ln tan(ϑ/2) = omega² r³ cos³θ / 3 + cos θ + ln tan(θ/2).
* Brightness: proportional to F, the same in every direction (no limb
  darkening), as virgil's GravityDarkenedStar documents.

The image is a weighted cloud of surface points: a midpoint grid in θ and
φ, each weighted by F times its projected area, the outward area vector
dX/dθ × dX/dφ dotted with the line of sight, where positive.
"""

import numpy as np
from scipy.optimize import brentq

from .sky import Cloud


def radius(theta, omega):
    """r(θ) on the Roche surface through the equator (r = 1 there)."""
    theta = np.asarray(theta, float)
    s2 = np.sin(theta) ** 2
    out = np.empty_like(theta)
    for k, (a) in enumerate(0.5 * omega**2 * s2.ravel()):
        # a r³ - (1 + omega²/2) r + 1 = 0
        if a == 0.0:
            out.flat[k] = 1.0 / (1.0 + 0.5 * omega**2)
            continue
        roots = np.roots([a, 0.0, -(1.0 + 0.5 * omega**2), 1.0])
        real = roots[np.abs(roots.imag) < 1e-12].real
        out.flat[k] = real[(real > 0) & (real <= 1.0 + 1e-12)].min()
    return out


def dradius(theta, r, omega):
    """dr/dθ, by implicit differentiation of the Roche surface."""
    s, c = np.sin(theta), np.cos(theta)
    return -(omega**2 * r**2 * s * c) / (-1.0 / r**2 + omega**2 * r * s**2)


def gravity(theta, r, omega):
    s, c = np.sin(theta), np.cos(theta)
    return -1.0 / r**2 + omega**2 * r * s**2, omega**2 * r * s * c


def vartheta(theta, r, omega):
    """ϑ of ELR11 for θ in (0, π/2]; the star is symmetric about its equator."""
    out = np.empty_like(np.asarray(theta, float))
    for k, (t, rr) in enumerate(zip(np.ravel(theta), np.ravel(r))):
        if abs(t - np.pi / 2) < 1e-12:
            out.flat[k] = np.pi / 2
            continue
        rhs = omega**2 * rr**3 * np.cos(t) ** 3 / 3.0 + np.cos(t) + np.log(np.tan(t / 2))
        out.flat[k] = brentq(lambda v: np.cos(v) + np.log(np.tan(v / 2)) - rhs, 1e-15, np.pi / 2, xtol=1e-15)
    return out


def flux(theta, r, omega):
    """ELR11's local bolometric flux, up to a constant, for θ in (0, π)."""
    north = np.where(theta <= np.pi / 2, theta, np.pi - theta)  # equatorial symmetry
    g_r, g_t = gravity(theta, r, omega)
    g = np.hypot(g_r, g_t)
    if omega == 0.0:
        return g
    v = vartheta(north, r, omega)
    return g * np.tan(v) ** 2 / np.tan(north) ** 2


def surface(omega, n_theta=400, n_phi=800):
    """Midpoint grid: positions X (star frame, z the rotation axis), outward
    area vectors dA (dX/dθ × dX/dφ dθ dφ) and fluxes F."""
    theta = (np.arange(n_theta) + 0.5) * np.pi / n_theta
    phi = (np.arange(n_phi) + 0.5) * 2 * np.pi / n_phi
    r = radius(theta, omega)
    dr = dradius(theta, r, omega)
    F = flux(theta, r, omega)
    T, P = np.meshgrid(theta, phi, indexing="ij")
    R, DR = r[:, None], dr[:, None]
    st, ct, sp, cp = np.sin(T), np.cos(T), np.sin(P), np.cos(P)
    X = np.stack([R * st * cp, R * st * sp, R * ct], -1)
    d_theta = np.stack([DR * st * cp + R * ct * cp, DR * st * sp + R * ct * sp, DR * ct - R * st], -1)
    d_phi = np.stack([-R * st * sp, R * st * cp, np.zeros_like(R * st)], -1)
    dA = np.cross(d_theta, d_phi) * (np.pi / n_theta) * (2 * np.pi / n_phi)
    return X, dA, np.broadcast_to(F[:, None], T.shape), (T, R, DR)


def planck(wavel, temperature):
    """B_lambda(T), up to a constant (SI h, c, k)."""
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    return 1.0 / (wavel**5 * np.expm1(h * c / (wavel * k * temperature)))


def cloud(diam_eq, omega, inc_deg, pa_deg, n_theta=400, n_phi=800, t_pole=None, wavel=None, t_exponent=0.25):
    """The star's image as a weighted point cloud (East, North in mas).

    The observer is at inclination inc from the rotation pole (0: pole-on);
    the visible pole projects onto the sky at position angle pa (North
    through East). The image is symmetric about the plane of the pole and
    the line of sight, so the handedness of the sky axes does not matter.

    With t_pole, each point radiates B_lambda(T) at ``wavel`` instead of the
    bolometric flux, with T = t_pole (F / F_pole)^(1/4) (T_eff^4 is
    proportional to F). ``t_exponent`` other than 1/4 is a wrong law, for
    controls."""
    i, p = np.deg2rad(inc_deg), np.deg2rad(pa_deg)
    X, dA, F, _ = surface(omega, n_theta, n_phi)
    los = np.array([np.sin(i), 0.0, np.cos(i)])
    up = np.array([-np.cos(i), 0.0, np.sin(i)])  # the visible pole's direction on the sky
    side = np.cross(los, up)
    proj = dA @ los
    if t_pole is not None:
        r_p = radius(np.array([1e-9]), omega)[0]
        f_pole = np.exp(2 / 3 * omega**2 * r_p**3) / r_p**2  # ELR11's polar limit (1/r_p² at omega = 0)
        F = planck(wavel, t_pole * (F / f_pole) ** t_exponent)
    w = np.where(proj > 0, F * proj, 0.0).ravel()
    a, b = (X @ up).ravel(), (X @ side).ravel()
    scale = diam_eq / 2.0
    east = scale * (a * np.sin(p) + b * np.cos(p))
    north = scale * (a * np.cos(p) - b * np.sin(p))
    keep = w > 0
    return Cloud(east[keep], north[keep], w[keep] / w[keep].sum())
