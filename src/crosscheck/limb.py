"""Limb-darkened stellar disks, written from first principles.

Nothing here imports virgil. The brightness of a disk of radius R is a
function of mu = cos(angle between the line of sight and the surface
normal) = sqrt(1 - (r / R)^2). Its visibility is computed by three routes:

* a point cloud (``disk``): Gauss-Legendre quadrature over the disk in the
  variable t = sqrt(mu), with the trapezoid rule in azimuth. Since
  r dr = -R^2 mu dmu and mu = t^2, the area element is 2 R^2 t^3 dt, and
  every power of mu, including sqrt(mu), becomes a polynomial in t, so the
  radial rule is exact for the brightness and spectrally accurate for the
  Fourier kernel. The same cloud feeds the OIFITS simulator.
* the Hankel transform (``vis_hankel``): the visibility of a circularly
  symmetric brightness I(r) is

      V(q) = int_0^R I(r) J0(2 pi q r) r dr / int_0^R I(r) r dr

  (Bracewell, *The Fourier Transform and its Applications*, ch. 12),
  evaluated by adaptive quadrature (``scipy.integrate.quad``) in t.
* for the linear law, the closed form of Hanbury Brown, Davis, Lake &
  Thompson (1974, MNRAS 167, 475), with SciPy Bessel functions.

The laws and their parametrizations follow the papers that define them:

* polynomial: I(mu) / I(1) = 1 - sum_n u_n (1 - mu)^n (the convention of
  starry; Luger et al. 2019, AJ 157, 64);
* square root: I(mu) / I(1) = 1 - c (1 - mu) - d (1 - sqrt(mu))
  (Diaz-Cordoves & Gimenez 1992, A&A 259, 227);
* Kipping (2013, MNRAS 435, 2152): q1, q2 in the unit square, eqs. 15-18
  for the quadratic law and 23-24 for the square-root law, and the physical
  constraints of eqs. 8 and 20-22.
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import j0, jv, roots_legendre

from .sky import MAS, Cloud, shift_phase

# ------------------------------------------------------------------ laws


def polynomial(u):
    """I(mu) / I(1) = 1 - sum_n u_n (1 - mu)^n, n = 1 .. len(u)."""
    u = np.atleast_1d(np.asarray(u, float))

    def profile(mu):
        mu = np.asarray(mu, float)
        return 1.0 - sum(un * (1.0 - mu) ** (n + 1) for n, un in enumerate(u))

    return profile


def square_root(c, d):
    """I(mu) / I(1) = 1 - c (1 - mu) - d (1 - sqrt(mu))."""

    def profile(mu):
        mu = np.asarray(mu, float)
        return 1.0 - c * (1.0 - mu) - d * (1.0 - np.sqrt(mu))

    return profile


# ------------------------------------------------- Kipping (2013) maps


def kipping_quadratic_u(q1, q2):
    """Kipping (2013) eqs. 15-16: (q1, q2) -> (u1, u2)."""
    return 2.0 * np.sqrt(q1) * q2, np.sqrt(q1) * (1.0 - 2.0 * q2)


def kipping_quadratic_q(u1, u2):
    """Kipping (2013) eqs. 17-18: (u1, u2) -> (q1, q2)."""
    return (u1 + u2) ** 2, u1 / (2.0 * (u1 + u2))


def kipping_square_root_cd(q1, q2):
    """Inverse of Kipping (2013) eqs. 23-24: (q1, q2) -> (c, d)."""
    return np.sqrt(q1) * (1.0 - 2.0 * q2), 2.0 * np.sqrt(q1) * q2


def kipping_square_root_q(c, d):
    """Kipping (2013) eqs. 23-24: (c, d) -> (q1, q2)."""
    return (c + d) ** 2, d / (2.0 * (c + d))


def quadratic_is_physical(u1, u2):
    """Kipping (2013) eq. 8: positive everywhere and decreasing to the limb."""
    return (u1 + u2 < 1.0) & (u1 > 0.0) & (u1 + 2.0 * u2 > 0.0)


def square_root_is_physical(c, d):
    """Kipping (2013) eqs. 20-22."""
    return (c + d < 1.0) & (d > 0.0) & (2.0 * c + d > 0.0)


# ------------------------------------------------------------ visibility


def disk(diam, profile, dra=0.0, ddec=0.0, n_t=96, n_theta=256):
    """Point cloud of a disk of diameter ``diam`` (mas) with brightness
    ``profile(mu)``, centred at (``dra``, ``ddec``)."""
    radius = diam / 2.0
    x, w = roots_legendre(n_t)
    t = (x + 1.0) / 2.0
    # r dr = 2 R^2 t^3 dt; the constant cancels in the normalisation
    wt = w * profile(t**2) * t**3
    r = radius * np.sqrt(1.0 - t**4)
    theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
    rr, tt = np.meshgrid(r, theta)
    ww = np.broadcast_to(wt, rr.shape)
    return Cloud(
        (rr * np.sin(tt)).ravel() + dra,
        (rr * np.cos(tt)).ravel() + ddec,
        (ww / ww.sum()).ravel(),
    )


def vis_hankel(u, v, wavel, diam, profile, dra=0.0, ddec=0.0):
    """Visibility by adaptive quadrature of the Hankel transform."""
    u, v, wavel = np.broadcast_arrays(
        np.asarray(u, float), np.asarray(v, float), np.asarray(wavel, float)
    )
    q = np.hypot(u, v) / wavel * MAS  # cycles per mas
    radius = diam / 2.0

    def integral(kernel):
        # in t = sqrt(mu): r = R sqrt(1 - t^4), r dr = 2 R^2 t^3 dt
        f = lambda t: profile(t * t) * kernel(radius * np.sqrt(1.0 - t**4)) * t**3  # noqa: E731
        return quad(f, 0.0, 1.0, epsabs=1e-15, epsrel=1e-13, limit=500)[0]

    norm = integral(lambda r: 1.0)
    amp = np.array(
        [integral(lambda r, qi=qi: j0(2.0 * np.pi * qi * r)) for qi in q.ravel()]
    ).reshape(q.shape)
    return amp / norm * shift_phase(u, v, wavel, dra, ddec)


def vis_linear_closed_form(u, v, wavel, diam, coeff, dra=0.0, ddec=0.0):
    """Linear law I = 1 - coeff (1 - mu): Hanbury Brown et al. (1974),

    V(x) = [(1 - a) J1(x) / x + a sqrt(pi / 2) J_{3/2}(x) / x^{3/2}]
           / [(1 - a) / 2 + a / 3],   x = pi diam q.
    """
    q = np.hypot(np.asarray(u, float), np.asarray(v, float)) / wavel * MAS
    x = np.pi * diam * q
    safe = np.where(x > 0, x, 1.0)
    a = coeff
    num = (1 - a) * jv(1, safe) / safe + a * np.sqrt(np.pi / 2) * jv(
        1.5, safe
    ) / safe**1.5
    amp = np.where(x > 0, num / ((1 - a) / 2 + a / 3), 1.0)
    return amp * shift_phase(u, v, wavel, dra, ddec)


def image(diam, profile, npix, fov, dra=0.0, ddec=0.0, oversample=1):
    """Brightness on a pixel grid (row 0 North, column 0 East, as
    ``sky.pixel_image`` reads it), averaged over ``oversample``^2 points per
    pixel."""
    pixel = fov / npix
    offsets = ((np.arange(oversample) + 0.5) / oversample - 0.5) * pixel
    centres = ((npix - 1) / 2.0 - np.arange(npix)) * pixel
    north = (centres[:, None] + offsets[None, :]).ravel()
    east = (centres[:, None] + offsets[None, :]).ravel()
    ee, nn = np.meshgrid(east - dra, north - ddec)
    r2 = (ee**2 + nn**2) / (diam / 2.0) ** 2
    inside = r2 <= 1.0
    bright = np.where(inside, profile(np.sqrt(np.where(inside, 1.0 - r2, 0.0))), 0.0)
    return bright.reshape(npix, oversample, npix, oversample).mean(axis=(1, 3))
