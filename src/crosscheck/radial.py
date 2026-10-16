"""Radial profiles and their Hankel transforms, from first principles.

Written for the PolarDisk benchmarks (stage V). Nothing here imports virgil.

Conventions (all lengths in milliarcseconds on the sky, spatial frequencies
in cycles per radian, ``q = baseline / wavelength``):

* A face-on, centred, axisymmetric disk of brightness ``f(r)`` has the
  normalized visibility (Thompson, Moran & Swenson 2017, eq. 3.xx; the
  Fourier-Bessel / Hankel transform of order 0)

      V(q) = int f(r) J0(2 pi q r) r dr / int f(r) r dr.

* Inclination ``inc`` and position angle ``pa`` of the major axis (North
  through East): the disk plane is compressed by ``cos(inc)`` along the
  minor axis, so the visibility at ``(u, v)`` (East, North) is the face-on
  one at the deprojected frequency

      q' = hypot(u_major, u_minor cos(inc)),

  with ``u_major = u sin(pa) + v cos(pa)`` and ``u_minor = u cos(pa) -
  v sin(pa)`` (the components along and across the major axis, from the
  sky geometry alone).

The transform is done by composite Gauss-Legendre panels, fine enough to
resolve both the profile and the oscillation of ``J0`` (SciPy's ``j0``), and
checked against ``scipy.integrate.quad`` in the tests. It shares nothing
with virgil's per-profile quadrature rules.
"""

import numpy as np
from scipy import special

MAS = np.pi / 180.0 / 3600.0 / 1000.0  # radians per milliarcsecond


# ------------------------------------------------------------- profiles
def gaussian_ring(r0, sigma):
    """exp[-(r - r0)^2 / 2 sigma^2], r >= 0. Returns (f, r_lo, r_hi, width)."""
    r_lo = max(r0 - 8 * sigma, 0.0)
    return (lambda r: np.exp(-0.5 * ((r - r0) / sigma) ** 2)), r_lo, r0 + 8 * sigma, sigma


def erf_power(r_in, r_out, index, edge):
    """vis-r's ``r_prof_erf2_power`` written out (Kennedy 2025): a power law
    r^index with error-function edges of standard deviation ``edge`` at
    ``r_in`` and ``r_out``,
    0.5 erfc((r_in - r)/(sqrt2 edge)) * 0.5 erfc((r - r_out)/(sqrt2 edge)) * r^index."""
    def f(r):
        r = np.asarray(r, float)
        inner = 0.5 * special.erfc((r_in - r) / (np.sqrt(2.0) * edge))
        outer = 0.5 * special.erfc((r - r_out) / (np.sqrt(2.0) * edge))
        return inner * outer * np.where(r > 0, r, 1e-300) ** index

    return f, max(r_in - 8 * edge, 1e-9), r_out + 8 * edge, edge


def with_gap(base, r_gap, width, depth):
    """base(r) * (1 - depth exp[-(r - r_gap)^2 / 2 width^2]): a multiplicative Gaussian gap."""
    f, lo, hi, w = base
    return (lambda r: f(r) * (1.0 - depth * np.exp(-0.5 * ((r - r_gap) / width) ** 2))), lo, hi, min(w, width)


def exp_tail(r0, scale, n_scales=80.0):
    """exp[-(r - r0)/scale] for r >= r0, summed to ``n_scales`` scales (e^-80)."""
    return (lambda r: np.where(r >= r0, np.exp(-(r - r0) / scale), 0.0)), r0, r0 + n_scales * scale, scale


# ------------------------------------------------------------ transform
def _panels(r_lo, r_hi, width, q_max, order=16, per_scale=2.0, per_cycle=0.5):
    """Gauss-Legendre nodes and weights over [r_lo, r_hi], panels no wider than
    ``width``/``per_scale`` or ``per_cycle`` of the fastest J0 cycle."""
    cycle = 1.0 / (q_max * MAS) if q_max > 0 else np.inf  # mas
    step = min(width / per_scale, per_cycle * cycle, r_hi - r_lo)
    n_pan = max(int(np.ceil((r_hi - r_lo) / step)), 1)
    edges = np.linspace(r_lo, r_hi, n_pan + 1)
    x, w = np.polynomial.legendre.leggauss(order)
    mid, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * np.diff(edges)
    return (mid[:, None] + half[:, None] * x).ravel(), (half[:, None] * w).ravel()


def hankel(profile, q, q_max=None):
    """Normalized visibility of a face-on centred disk at frequencies ``q``
    (cycles per radian), by composite Gauss-Legendre quadrature."""
    f, r_lo, r_hi, width = profile
    q = np.atleast_1d(np.asarray(q, float))
    r, w = _panels(r_lo, r_hi, width, float(np.max(q)) if q_max is None else q_max)
    fr = f(r) * r * w
    total = fr.sum()
    out = np.empty(q.shape)
    for i in range(0, q.size, 2000):  # chunked: N_vis x N_r matrices stay small
        out[i:i + 2000] = special.j0(2 * np.pi * np.outer(q[i:i + 2000], r * MAS)) @ fr
    return out / total


def hankel_quad(profile, q):
    """The same by ``scipy.integrate.quad`` (slow; for checking ``hankel``)."""
    from scipy.integrate import quad

    f, r_lo, r_hi, _ = profile
    norm = quad(lambda r: f(r) * r, r_lo, r_hi, limit=2000)[0]
    return np.array([
        quad(lambda r, qq=qq: f(r) * special.j0(2 * np.pi * qq * r * MAS) * r, r_lo, r_hi, limit=4000)[0] / norm
        for qq in np.atleast_1d(q)
    ])


# ------------------------------------------------------------- geometry
def deproject(u, v, inc_deg, pa_deg):
    """Deprojected frequency and the along/across-major-axis components.

    ``u``, ``v`` (East, North). Returns ``(q', u_major, u_minor)``."""
    pa, inc = np.deg2rad(pa_deg), np.deg2rad(inc_deg)
    u_major = u * np.sin(pa) + v * np.cos(pa)
    u_minor = u * np.cos(pa) - v * np.sin(pa)
    return np.hypot(u_major, u_minor * np.cos(inc)), u_major, u_minor


def disk_visibility(profile, u, v, inc_deg=0.0, pa_deg=0.0):
    """Complex (real) visibility of an inclined, rotated, centred disk at (u, v) in cycles/rad."""
    qd, _, _ = deproject(u, v, inc_deg, pa_deg)
    return hankel(profile, qd, q_max=float(np.max(qd)))


def baselines(n, q_max=8.0e7, seed=0):
    """``n`` random (u, v) with |q| <= q_max cycles/rad (uniform over the disk, like a filled uv plane)."""
    rng = np.random.default_rng(seed)
    r = q_max * np.sqrt(rng.uniform(1e-4, 1.0, n))
    a = rng.uniform(0, 2 * np.pi, n)
    return r * np.cos(a), r * np.sin(a)
