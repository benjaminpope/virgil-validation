"""Aperture-masking interferograms, by dLux and in closed form.

The scene is a point cloud (crosscheck.sky.Cloud). Each point is imaged
through the mask and the images are summed with the cloud's weights, which is
exactly how an incoherent extended source forms an image.

Two imagers, independent of each other:

* ``image_dlux``: dLux (Desdoigts et al. 2023) Fraunhofer propagation of a
  sampled, soft-edged pupil with one wavefront per point;
* ``image_analytic``: identical circular holes of diameter d at x_j give the
  point-spread function |A(theta)|^2 |sum_j exp(-2 pi i x_j . theta / lambda)|^2,
  where A(theta) = 2 J1(pi d |theta| / lambda) / (pi d |theta| / lambda) is
  the Airy amplitude of one hole.

Observables come from the image's Fourier transform at the baselines
u = (x_j - x_i) / lambda, divided by that of a point-source (calibrator)
image: the optical transfer function cancels, leaving the scene's
visibility. This is exact for a monochromatic image sampled finer than
Nyquist, up to the light lost off the edge of the detector.

Pixel (row r, column c) of an n x n image sits at sky offset
East = (c - (n - 1) / 2) * scale and North = (r - (n - 1) / 2) * scale,
which is how dLux places a source offset by (x, y) = (East, North) (found by
experiment in tests/test_nrm.py). The orientation is a choice; using it
consistently for the sources and the Fourier transform is what matters.
"""

from itertools import combinations

import numpy as np
from scipy.special import j1

from .sky import MAS

# NIRISS-AMI-like hole centres (m) on the hexagonal 1.32 m grid of a 6.5 m
# pupil, with circular holes in place of hexagons.
HOLES = np.array(
    [
        [0.0, -2.64],
        [-2.28631, 0.0],
        [2.28631, -1.32],
        [-2.28631, 1.32],
        [-1.143155, 1.98],
        [2.28631, 1.32],
        [1.143155, 1.98],
    ]
)
HOLE_DIAM = 0.8


def is_non_redundant(holes, hole_diam):
    """No two baselines (as vectors, either sign) closer than a hole."""
    b = [holes[j] - holes[i] for i, j in combinations(range(len(holes)), 2)]
    for (k, p), (l, q) in combinations(list(enumerate(b)), 2):
        if min(np.hypot(*(p - q)), np.hypot(*(p + q))) < hole_diam:
            return False
    return True


def pixel_grid(npix, pixel_mas):
    c = (np.arange(npix) - (npix - 1) / 2.0) * pixel_mas
    east, north = np.meshgrid(c, c)  # east varies along columns
    return east, north


def image_analytic(cloud, wavel, npix=256, pixel_mas=30.0, holes=HOLES, hole_diam=HOLE_DIAM, chunk=64):
    east, north = pixel_grid(npix, pixel_mas)
    img = np.zeros_like(east)
    for start in range(0, cloud.weight.size, chunk):
        sl = slice(start, start + chunk)
        de = (east[None] - cloud.east[sl, None, None]) * MAS
        dn = (north[None] - cloud.north[sl, None, None]) * MAS
        x = np.pi * hole_diam * np.hypot(de, dn) / wavel
        with np.errstate(invalid="ignore", divide="ignore"):
            amp = np.where(x > 0, 2 * j1(x) / np.where(x > 0, x, 1), 1.0)
        fringe = np.zeros(de.shape, complex)
        for hx, hy in holes:
            fringe += np.exp(-2j * np.pi * (hx * de + hy * dn) / wavel)
        psf = amp**2 * np.abs(fringe) ** 2
        img += np.tensordot(cloud.weight[sl], psf, axes=1)
    return img


def dlux_optics(npix=256, pixel_mas=30.0, holes=HOLES, hole_diam=HOLE_DIAM, wf_npix=256, oversample=1):
    import dLux as dl
    import jax.numpy as jnp

    apertures = [
        dl.CircularAperture(
            hole_diam / 2,
            transformation=dl.CoordTransform(translation=jnp.array(h)),
        )
        for h in holes
    ]
    return dl.AngularOpticalSystem(
        wf_npix, 6.5, [("mask", dl.MultiAperture(apertures))], npix,
        pixel_mas / 1000.0, oversample,
    )


def image_dlux(cloud, wavel, optics, chunk=256):
    import jax
    import jax.numpy as jnp

    def one(pos):
        return optics.propagate_mono(wavel, pos)

    batched = jax.jit(jax.vmap(one))
    pos = np.column_stack([cloud.east, cloud.north]) * MAS
    img = 0.0
    for start in range(0, len(pos), chunk):
        sl = slice(start, start + chunk)
        psfs = batched(jnp.asarray(pos[sl]))
        img = img + np.tensordot(cloud.weight[sl], np.asarray(psfs), axes=1)
    if optics.oversample > 1:
        n = img.shape[0] // optics.oversample
        img = img.reshape(n, optics.oversample, n, optics.oversample).sum((1, 3))
    return img


def fourier(image, u, v, wavel, pixel_mas):
    """Image Fourier transform at baselines u, v (m), OIFITS sign."""
    east, north = pixel_grid(image.shape[0], pixel_mas)
    u = np.atleast_1d(u)
    v = np.atleast_1d(v)
    shape = np.broadcast(u, v).shape
    fu = (np.broadcast_to(u, shape) / wavel * MAS).ravel()
    fv = (np.broadcast_to(v, shape) / wavel * MAS).ravel()
    phase = -2j * np.pi * (np.outer(fu, east.ravel()) + np.outer(fv, north.ravel()))
    return (np.exp(phase) @ image.ravel()).reshape(shape)


def calibrated_vis_fn(science, calibrator, wavel, pixel_mas):
    """vis(u, v, wavel) from the ratio of two images' Fourier transforms."""

    def vis(u, v, w):
        assert np.allclose(w, wavel)
        return fourier(science, u, v, wavel, pixel_mas) / fourier(calibrator, u, v, wavel, pixel_mas)

    return vis
