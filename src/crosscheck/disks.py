"""Brightness of a flared scattered-light disk (Blakely et al. 2024, Eqs. 2-9),
written from the published model as virgil's FlaredDisk documents it:
mid-plane coordinates (x along the major axis at position angle pa, y towards
the near side at pa + 90, divided by cos i), a flared surface raising the
apparent radius, a skewed Gaussian ring and an azimuthal phase function.
"""

import numpy as np
from scipy.special import erf

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def phase_hg(theta, g):
    return (1 - g**2) / (4 * np.pi * (1 + g**2 - 2 * g * np.cos(theta)) ** 1.5)


def phase_gaussian(theta, sigma_theta_deg):
    return np.exp(-0.5 * (theta / np.deg2rad(sigma_theta_deg)) ** 2)


def phase_power(theta, n):
    return np.cos(theta / 2.0) ** n


def brightness(east, north, radius, fwhm, inc, pa, phase, skew=0.0,
               aspect=0.0, flaring=1.25, symmetric=0.0):
    """Surface brightness at sky offsets (mas) from the disk centre."""
    p, i = np.deg2rad(pa), np.deg2rad(inc)
    major = np.array([np.sin(p), np.cos(p)])
    near = np.array([np.cos(p), -np.sin(p)])  # position angle pa + 90
    x = east * major[0] + north * major[1]
    y = (east * near[0] + north * near[1]) / np.cos(i)
    rho = np.hypot(x, y)
    z = aspect * radius * (rho / radius) ** flaring
    r = np.sqrt(x**2 + (y + z * np.sin(i)) ** 2 + z**2)
    theta = np.arctan2(x, y)  # mid-plane azimuth from the near-side minor axis
    sigma_r = fwhm / FWHM_PER_SIGMA
    ring = np.exp(-((r - radius) ** 2) / (2 * sigma_r**2))
    edge = 0.5 * (1 + erf(skew * (r - radius) / (np.sqrt(2) * sigma_r)))
    return (phase(theta) + symmetric) * ring * edge


def pixel_centres(npix, pixel_scale):
    """Offsets of the centres of a centred npix x npix grid (npix even)."""
    c = (np.arange(npix) - (npix - 1) / 2.0) * pixel_scale
    east, north = np.meshgrid(c, c)
    return east.ravel(), north.ravel()


def scattered_ring(radius, fwhm, inc, pa, phase, n_r=300, n_phi=1440):
    """A flat ring of dust lit by the central star, rendered in 3-D from
    first principles rather than from Blakely et al.'s azimuthal form: each
    point of the mid-plane (Gaussian in radius) scatters starlight towards
    the observer with ``phase`` of the true scattering angle, the angle
    between the star-to-point and point-to-observer directions. The
    mid-plane's near side (towards the observer) is labelled by position
    angle pa + 90, as virgil labels it. Returns (east, north, weight)."""
    sigma = fwhm / FWHM_PER_SIGMA
    r = np.linspace(max(radius - 4 * sigma, 1e-6), radius + 4 * sigma, n_r)
    phi = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    R, P = np.meshgrid(r, phi)
    x, y = R * np.sin(P), R * np.cos(P)  # along the major axis, towards the near side
    i, p = np.deg2rad(inc), np.deg2rad(pa)
    major, near = np.array([np.sin(p), np.cos(p)]), np.array([np.cos(p), -np.sin(p)])
    east = x * major[0] + y * np.cos(i) * near[0]
    north = x * major[1] + y * np.cos(i) * near[1]
    towards_observer = y * np.sin(i)
    scattering = np.arccos(np.clip(towards_observer / R, -1.0, 1.0))
    w = np.exp(-((R - radius) ** 2) / (2 * sigma**2)) * phase(scattering) * R  # R: the area element
    return east.ravel(), north.ravel(), (w / w.sum()).ravel()
