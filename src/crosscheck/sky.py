"""Sky brightness primitives, written from first principles.

Nothing here imports virgil. Every primitive is described twice, by
independent routes:

* a *point cloud*: positions (East, North offsets in mas) and weights that
  sum to one, from a quadrature rule over the brightness distribution. Its
  Fourier transform is a direct sum, and the same cloud can be pushed through
  an optical simulator (dLux) source by source;
* where a textbook closed form exists (Berger & Segransan 2007, New Astron.
  Rev. 51, 576), that closed form, evaluated with SciPy special functions.

Fourier convention (OIFITS, Pauls et al. 2005, PASP 117, 1255):

    V(u, v) = sum_k w_k exp(-2 pi i (u alpha_k + v delta_k) / lambda)

with u, v the baseline projected East and North in metres, alpha the offset
towards East and delta towards North, in radians. Position angles are
measured from North towards East.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import j1, roots_hermite, roots_legendre

MAS = np.pi / 180.0 / 3600.0 / 1000.0  # radians per milliarcsecond
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass
class Cloud:
    """Weighted points: East and North offsets in mas, weights sum to 1."""

    east: np.ndarray
    north: np.ndarray
    weight: np.ndarray

    def shifted(self, dra, ddec):
        return Cloud(self.east + dra, self.north + ddec, self.weight)

    def rotated(self, angle_deg):
        """Rotate the scene about the origin by ``angle_deg`` North->East.

        A point at position angle p moves to position angle p + angle.
        """
        radius = np.hypot(self.east, self.north)
        pa = np.arctan2(self.east, self.north) + np.deg2rad(angle_deg)
        return Cloud(radius * np.sin(pa), radius * np.cos(pa), self.weight)


def mix(clouds, fluxes, resolved_flux=0.0):
    """Flux-weighted union of clouds; ``resolved_flux`` adds only to the
    normalisation (light with zero visibility on every baseline)."""
    total = float(np.sum(fluxes)) + resolved_flux
    east = np.concatenate([c.east for c in clouds])
    north = np.concatenate([c.north for c in clouds])
    weight = np.concatenate(
        [c.weight * f / total for c, f in zip(clouds, fluxes)]
    )
    return Cloud(east, north, weight)


def visibility(cloud, u, v, wavel):
    """Complex visibility of a cloud: a direct sum, float64, in chunks."""
    u, v, wavel = np.broadcast_arrays(
        np.asarray(u, float), np.asarray(v, float), np.asarray(wavel, float)
    )
    shape = u.shape
    fu = (u / wavel * MAS).ravel()
    fv = (v / wavel * MAS).ravel()
    out = np.empty(fu.shape, complex)
    step = max(1, int(2e7 // max(cloud.weight.size, 1)))
    for start in range(0, fu.size, step):
        sl = slice(start, start + step)
        phase = -2j * np.pi * (
            np.outer(fu[sl], cloud.east) + np.outer(fv[sl], cloud.north)
        )
        out[sl] = np.exp(phase) @ cloud.weight
    return out.reshape(shape)


# ----------------------------------------------------------------- clouds


def point(dra=0.0, ddec=0.0):
    return Cloud(np.array([dra]), np.array([ddec]), np.array([1.0]))


def _hermite_2d(n):
    """Gauss-Hermite nodes and weights for a unit-variance 2D Gaussian."""
    x, w = roots_hermite(n)
    x = x * np.sqrt(2.0)
    w = w / np.sqrt(np.pi)
    xx, yy = np.meshgrid(x, x)
    ww = np.outer(w, w)
    return xx.ravel(), yy.ravel(), ww.ravel()


def elliptical_gaussian(fwhm, ratio=1.0, pa=0.0, dra=0.0, ddec=0.0, n=48):
    """Gaussian with major-axis FWHM ``fwhm`` along position angle ``pa``."""
    sig_major = fwhm / FWHM_PER_SIGMA
    sig_minor = sig_major * ratio
    a, b, w = _hermite_2d(n)
    major = np.array([np.sin(np.deg2rad(pa)), np.cos(np.deg2rad(pa))])
    minor = np.array([np.cos(np.deg2rad(pa)), -np.sin(np.deg2rad(pa))])
    pos = np.outer(a * sig_major, major) + np.outer(b * sig_minor, minor)
    return Cloud(pos[:, 0] + dra, pos[:, 1] + ddec, w)


def gaussian(sigma, dra=0.0, ddec=0.0, n=48):
    return elliptical_gaussian(sigma * FWHM_PER_SIGMA, 1.0, 0.0, dra, ddec, n)


def uniform_disk(diam, dra=0.0, ddec=0.0, n_r=96, n_theta=256):
    """Tophat disk by Gauss-Legendre in r (area element r dr) and the
    trapezoid rule in azimuth (spectrally accurate for a periodic integrand).
    """
    radius = diam / 2.0
    x, w = roots_legendre(n_r)
    r = radius * (x + 1.0) / 2.0
    wr = w * r  # integrand weight r dr, up to a constant
    theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
    rr, tt = np.meshgrid(r, theta)
    ww = np.broadcast_to(wr, rr.shape)
    weight = (ww / ww.sum()).ravel()
    return Cloud(
        (rr * np.sin(tt)).ravel() + dra,
        (rr * np.cos(tt)).ravel() + ddec,
        weight,
    )


def inclined_ring(
    diam,
    inc=0.0,
    pa=0.0,
    az_amps=(),
    az_pas=(),
    modulation="sky",
    dra=0.0,
    ddec=0.0,
    n_phi=2048,
):
    """An infinitely thin circular ring, inclined and rotated on the sky.

    The ring is uniform per unit length in its own plane (uniform in the
    in-plane azimuth phi). Seen at inclination ``inc``, it is an ellipse
    whose major axis (length ``diam``) lies along position angle ``pa`` and
    whose minor axis is shortened by cos(inc).

    The optional brightness modulation 1 + sum_m A_m cos(m (theta - pa_m))
    is defined in one of two ways, because the phrase "azimuth" is
    ambiguous for an inclined ring:

    * ``modulation="sky"``: theta is the on-sky position angle of each
      point of the ellipse, seen from its centre;
    * ``modulation="disk"``: theta is the in-plane azimuth, counted from
      the major axis at ``pa`` in the same sense as position angle, so
      that a pole-on ring gives the same answer either way.
    """
    radius = diam / 2.0
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    a = radius * np.cos(phi)  # along the major axis
    b = radius * np.sin(phi) * np.cos(np.deg2rad(inc))  # along the minor axis
    p = np.deg2rad(pa)
    # unit vectors (East, North): major at PA, minor at PA + 90 deg
    east = a * np.sin(p) + b * np.cos(p)
    north = a * np.cos(p) - b * np.sin(p)
    if modulation == "sky":
        theta = np.arctan2(east, north)
    elif modulation == "disk":
        theta = phi + p
    else:
        raise ValueError(modulation)
    bright = np.ones_like(phi)
    for m, (amp, pa_m) in enumerate(zip(az_amps, az_pas), start=1):
        bright = bright + amp * np.cos(m * (theta - np.deg2rad(pa_m)))
    weight = bright / bright.sum()
    return Cloud(east + dra, north + ddec, weight)


def blurred(cloud, fwhm, n=12):
    """Convolve a cloud with an isotropic Gaussian (as a product cloud)."""
    g = gaussian(fwhm / FWHM_PER_SIGMA, n=n)
    east = (cloud.east[:, None] + g.east[None, :]).ravel()
    north = (cloud.north[:, None] + g.north[None, :]).ravel()
    weight = (cloud.weight[:, None] * g.weight[None, :]).ravel()
    return Cloud(east, north, weight)


def arc_curve(radius, length, pa=0.0, dra=0.0, ddec=0.0, n=8192):
    """A circle of ``radius`` about (dra, ddec), weighted by a Gaussian in arc
    length (FWHM ``length``) peaking at position angle ``pa``. The weight
    runs round the whole circle (arc length wrapped to [-pi R, pi R))."""
    psi = 2.0 * np.pi * np.arange(n) / n
    offset = np.angle(np.exp(1j * (psi - np.deg2rad(pa))))  # wrapped
    s = radius * offset
    sigma = length / FWHM_PER_SIGMA
    weight = np.exp(-0.5 * (s / sigma) ** 2)
    weight /= weight.sum()
    return Cloud(
        radius * np.sin(psi) + dra, radius * np.cos(psi) + ddec, weight
    )


def pixel_image(image, pixel_scale, dra=0.0, ddec=0.0):
    """Pixels as points at their centres. Row 0 is North (top), column 0 is
    East (left), as an astronomer displays the sky; the image centre is at
    index ((nrow - 1) / 2, (ncol - 1) / 2)."""
    image = np.asarray(image, float)
    nrow, ncol = image.shape
    north = ((nrow - 1) / 2.0 - np.arange(nrow)) * pixel_scale
    east = ((ncol - 1) / 2.0 - np.arange(ncol)) * pixel_scale
    ee, nn = np.meshgrid(east, north)
    return Cloud(ee.ravel() + dra, nn.ravel() + ddec, (image / image.sum()).ravel())


# ----------------------------------------------------- textbook closed forms


def _q(u, v, wavel):
    """Spatial frequency (cycles per mas) East and North."""
    return np.asarray(u) / wavel * MAS, np.asarray(v) / wavel * MAS


def shift_phase(u, v, wavel, dra, ddec):
    fu, fv = _q(u, v, wavel)
    return np.exp(-2j * np.pi * (fu * dra + fv * ddec))


def vis_point(u, v, wavel, dra=0.0, ddec=0.0):
    return shift_phase(u, v, wavel, dra, ddec)


def vis_gaussian(u, v, wavel, sigma, dra=0.0, ddec=0.0):
    fu, fv = _q(u, v, wavel)
    return np.exp(-2.0 * np.pi**2 * sigma**2 * (fu**2 + fv**2)) * shift_phase(
        u, v, wavel, dra, ddec
    )


def vis_elliptical_gaussian(u, v, wavel, fwhm, ratio, pa, dra=0.0, ddec=0.0):
    fu, fv = _q(u, v, wavel)
    p = np.deg2rad(pa)
    q_major = fu * np.sin(p) + fv * np.cos(p)
    q_minor = fu * np.cos(p) - fv * np.sin(p)
    s = fwhm / FWHM_PER_SIGMA
    amp = np.exp(
        -2.0 * np.pi**2 * s**2 * (q_major**2 + (ratio * q_minor) ** 2)
    )
    return amp * shift_phase(u, v, wavel, dra, ddec)


def vis_uniform_disk(u, v, wavel, diam, dra=0.0, ddec=0.0):
    fu, fv = _q(u, v, wavel)
    x = np.pi * diam * np.hypot(fu, fv)
    with np.errstate(invalid="ignore", divide="ignore"):
        amp = np.where(x > 0, 2.0 * j1(x) / np.where(x > 0, x, 1.0), 1.0)
    return amp * shift_phase(u, v, wavel, dra, ddec)


def gaussian_blur_factor(u, v, wavel, fwhm):
    """Fourier transform of a unit isotropic Gaussian of FWHM ``fwhm``."""
    return vis_gaussian(u, v, wavel, fwhm / FWHM_PER_SIGMA)
