"""Rendered scenes for PolarDisk, summed by brute force (stage V, V2).

Nothing here imports virgil. A scene is rendered *in the plane of the disk*
at fine resolution (polar nodes in radius and azimuth, with their area
weights), carried to the sky by the documented geometry, and its Fourier
transform is the direct sum of ``crosscheck.sky.visibility``: no Bessel
functions and none of the per-order structure of the Hankel route.

Geometry (virgil's documentation of ``PolarDisk``, ``Harmonics``, ``Cone``;
the same sky conventions as ``crosscheck.sky.inclined_ring``):

* a disk-plane point (r, theta) has coordinates ``a = r cos(theta)`` along the
  major axis (position angle ``pa``) and ``b = r sin(theta)`` along the
  direction ``pa + 90``; theta is the in-plane azimuth from the major axis
  towards ``pa + 90``;
* on the sky the minor-axis coordinate is compressed, ``b -> b cos(inc)``;
* a ring raised by height ``h(r)`` along the disk normal moves by
  ``h sin(inc)`` towards position angle ``pa + 90`` (``h > 0``) ("the
  emitting surface is thin and nothing is occulted").

Azimuthal factors (mean one over theta), all from their documented
definitions:

* linear: ``1 + sum_m c_m cos(m theta) + s_m sin(m theta)``;
* positive: ``|h(e^{i theta})|^2 / sum_j |h_j|^2`` with ``h_0 = 1`` and
  ``h_1 ... h_M`` from the complex reflection coefficients by the step-up
  recursion ``h^(j)_i = h^(j-1)_i + k_j conj(h^(j-1)_{j-i})``,
  ``k_j = w_j tanh|w_j| / |w_j|`` (Dumitrescu 2007; Barndorff-Nielsen &
  Schou 1973);
* von Mises: ``exp[kappa cos(theta - theta_0)] / I_0(kappa)``, the exact
  function (virgil truncates its harmonic series at ``degree``).
"""

import numpy as np
from scipy import special

from crosscheck.sky import Cloud


# ------------------------------------------------------ azimuthal factors
def az_linear(theta, c=(), s=()):
    out = np.ones_like(theta)
    for m, (cm, sm) in enumerate(zip(c, s), start=1):
        out = out + cm * np.cos(m * theta) + sm * np.sin(m * theta)
    # cosine-only or sine-only lists may differ in length
    for m, cm in enumerate(c[len(s):], start=len(s) + 1):
        out = out + cm * np.cos(m * theta)
    for m, sm in enumerate(s[len(c):], start=len(c) + 1):
        out = out + sm * np.sin(m * theta)
    return out


def reflection_to_h(k):
    """Step-up recursion: reflection coefficients k_1..k_M -> h_0..h_M (h_0 = 1)."""
    h = np.array([1.0 + 0j])
    for j, kj in enumerate(k, start=1):
        ext = np.concatenate([h, [0j]])
        h = ext + kj * np.conj(ext[::-1])
    return h


def latents_to_k(w):
    w = np.asarray(w, complex)
    a = np.abs(w)
    safe = np.where(a > 0, a, 1.0)
    return np.where(a > 0, w * np.tanh(a) / safe, w)


def az_positive(theta, w):
    h = reflection_to_h(latents_to_k(w))
    z = np.exp(1j * theta)
    poly = sum(hj * z ** j for j, hj in enumerate(h))
    return np.abs(poly) ** 2 / np.sum(np.abs(h) ** 2)


def az_vonmises(theta, kappa, theta0_deg):
    """exp[kappa cos(theta - theta0)] / I0(kappa), by the scaled Bessel function (no overflow)."""
    return np.exp(kappa * (np.cos(theta - np.deg2rad(theta0_deg)) - 1.0)) / special.i0e(kappa)


# ------------------------------------------------------------- rendering
def render(radii, weights, azimuth, inc=0.0, pa=0.0, height=None, n_theta=1024, dra=0.0, ddec=0.0):
    """A disk as a point cloud on the sky.

    ``radii`` and ``weights``: nodes in radius (mas) and the weight
    ``I0(r) r dr`` of each (a thin ring: one node, weight 1); ``azimuth``:
    function of theta (mean one); ``height``: function of r (mas) or None.
    The azimuth is sampled by the trapezoid rule on ``n_theta`` points
    (spectrally accurate for a periodic function of band limit below
    ``n_theta / 2``).
    """
    theta = 2 * np.pi * np.arange(n_theta) / n_theta
    az = azimuth(theta) if azimuth is not None else np.ones_like(theta)
    r = np.asarray(radii, float)[:, None]
    a = r * np.cos(theta)[None, :]
    b = r * np.sin(theta)[None, :] * np.cos(np.deg2rad(inc))
    if height is not None:
        b = b + (np.asarray(height(np.asarray(radii, float)))[:, None]) * np.sin(np.deg2rad(inc))
    p = np.deg2rad(pa)
    east = a * np.sin(p) + b * np.cos(p)
    north = a * np.cos(p) - b * np.sin(p)
    w = np.asarray(weights, float)[:, None] * az[None, :]
    return Cloud(east.ravel() + dra, north.ravel() + ddec, (w / w.sum()).ravel())


def gauss_legendre_nodes(r_lo, r_hi, n, weight=None):
    """Gauss-Legendre nodes on [r_lo, r_hi] with weights x r (the area element), times ``weight(r)``."""
    x, w = np.polynomial.legendre.leggauss(n)
    r = 0.5 * (r_hi - r_lo) * x + 0.5 * (r_hi + r_lo)
    w = 0.5 * (r_hi - r_lo) * w * r
    if weight is not None:
        w = w * weight(r)
    return r, w


def panelled_nodes(r_lo, r_hi, weight, n_panels=60, order=12):
    """Composite Gauss-Legendre over many panels: many nodes, to resolve narrow structure."""
    edges = np.linspace(r_lo, r_hi, n_panels + 1)
    x, w = np.polynomial.legendre.leggauss(order)
    mid, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * np.diff(edges)
    r = (mid[:, None] + half[:, None] * x).ravel()
    ww = (half[:, None] * w).ravel() * r
    return r, ww * weight(r)
