"""Image-recovery figures of merit, written from their published definitions.

NumPy only. Each function says where its formula comes from; none is taken
from virgil's code. Images are 2-D arrays on a common grid; where the
orientation matters (``shift``) row 0 is North and column 0 East, with the
centre at the middle of the array, as in the OIFITS image convention that
virgil documents for its ``Image``.

* ``ncc``: the zero-mean normalized cross-correlation (Pearson's r of the
  pixel values).
* ``l1_score``: the 2024 imaging contest's L1 score, 1 - min_a Σ|a e - r| /
  Σ r over a >= 0. The objective is convex and piecewise linear in a, so
  its minimum is at a = 0 or at one of the breakpoints r_i / e_i; we
  evaluate every one (brute force, no sorting trick).
* ``lawson_sigma_over_peak``: Lawson et al. (2004, Proc. SPIE 5491, 886),
  Eq. 2: σ² = Σ r (e - r)² / Σ r with both images normalized to unit sum,
  over the peak of the normalized truth.
* ``rms``: the rms pixel difference of the unit-sum images (Cotton et al.
  2008; Malbet et al. 2010), optionally after both are convolved with a
  Gaussian beam sampled on the pixels.
* ``resample``: exact flux-conserving rebinning, treating each pixel as
  uniform: the flux of a new pixel is the sum over old pixels of their flux
  times the fraction of their area inside it (separable 1-D overlap
  matrices).
* ``shift``: an integer-pixel translation with zeros brought in.
"""

import numpy as np

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def unit(image):
    image = np.asarray(image, float)
    return image / image.sum()


def ncc(image, truth):
    e = np.asarray(image, float).ravel()
    r = np.asarray(truth, float).ravel()
    e, r = e - e.mean(), r - r.mean()
    return float(e @ r / np.sqrt((e @ e) * (r @ r)))


def l1_score(image, truth):
    e = np.clip(np.asarray(image, float).ravel(), 0.0, None)
    r = np.asarray(truth, float).ravel()
    candidates = np.concatenate([[0.0], r[e > 0] / e[e > 0]])
    candidates = candidates[candidates >= 0]
    cost = np.abs(candidates[:, None] * e[None, :] - r[None, :]).sum(1)
    return float(1.0 - cost.min() / r.sum())


def lawson_sigma_over_peak(image, truth):
    e, r = unit(image), unit(truth)
    sigma = np.sqrt(np.sum(r * (e - r) ** 2) / np.sum(r))
    return float(sigma / r.max())


def gaussian_kernel(npix, pixel_scale, major, minor, pa_deg):
    """A unit-sum elliptical Gaussian of FWHMs ``major`` and ``minor`` (mas),
    major axis at ``pa_deg`` North through East, sampled on an ``npix``
    square grid (row 0 North, column 0 East)."""
    c = (np.arange(npix) - (npix - 1) / 2) * pixel_scale
    east = -c[None, :] * np.ones((npix, 1))
    north = -c[:, None] * np.ones((1, npix))
    p = np.deg2rad(pa_deg)
    along = east * np.sin(p) + north * np.cos(p)
    across = east * np.cos(p) - north * np.sin(p)
    k = np.exp(-0.5 * ((along / (major / FWHM_PER_SIGMA)) ** 2 + (across / (minor / FWHM_PER_SIGMA)) ** 2))
    return k / k.sum()


def convolve(image, kernel):
    """Linear (zero-padded) convolution with a kernel of odd size centred on
    its middle pixel; the output has the input's shape."""
    image = np.asarray(image, float)
    ny, nx = image.shape
    ky, kx = kernel.shape
    if ky % 2 == 0 or kx % 2 == 0:
        raise ValueError("the kernel needs an odd size, to have a centre pixel")
    shape = (ny + ky, nx + kx)
    out = np.fft.irfft2(np.fft.rfft2(image, shape) * np.fft.rfft2(kernel, shape), shape)
    return out[(ky - 1) // 2:(ky - 1) // 2 + ny, (kx - 1) // 2:(kx - 1) // 2 + nx]


def rms(image, truth, kernel=None, relative=False):
    e, r = unit(image), unit(truth)
    if kernel is not None:
        e, r = convolve(e, kernel), convolve(r, kernel)
    value = np.sqrt(np.mean((e - r) ** 2))
    return float(value / r.max() if relative else value)


def overlap_matrix(n_old, s_old, n_new, s_new):
    """W[j, i]: the fraction of old pixel i inside new pixel j, both grids
    centred on the same point."""
    old = (np.arange(n_old + 1) - n_old / 2) * s_old
    new = (np.arange(n_new + 1) - n_new / 2) * s_new
    lo = np.maximum(new[:-1, None], old[None, :-1])
    hi = np.minimum(new[1:, None], old[None, 1:])
    return np.clip(hi - lo, 0.0, None) / s_old


def resample(image, pixel_scale, npix, new_pixel_scale):
    image = np.asarray(image, float)
    ny, nx = (npix, npix) if np.isscalar(npix) else npix
    wy = overlap_matrix(image.shape[0], pixel_scale, ny, new_pixel_scale)
    wx = overlap_matrix(image.shape[1], pixel_scale, nx, new_pixel_scale)
    return wy @ image @ wx.T


def shift(image, rows, cols):
    """Move the content by whole pixels, +rows down (South) and +cols right
    (West), filling with zeros."""
    image = np.asarray(image, float)
    out = np.zeros_like(image)
    ny, nx = image.shape
    src_r = slice(max(0, -rows), min(ny, ny - rows))
    dst_r = slice(max(0, rows), min(ny, ny + rows))
    src_c = slice(max(0, -cols), min(nx, nx - cols))
    dst_c = slice(max(0, cols), min(nx, nx + cols))
    out[dst_r, dst_c] = image[src_r, src_c]
    return out
