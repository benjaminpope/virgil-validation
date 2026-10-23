"""NumPy side of the IM Lup CLEAN image: multiscale minor cycles, the restoring beam and image statistics. Nothing here imports virgil.

Images follow ``crosscheck.alma``: ``image[row, col]`` with row 0 North and column 0 East; a grid of odd side
``n`` has its phase centre in the pixel ``((n - 1) / 2, (n - 1) / 2)``, so that the dirty beam peaks on a pixel.

* The dirty image is the image of the weighted visibilities, in Jy/beam; the dirty beam (PSF) peaks at 1. A point
  source of flux ``F`` makes the dirty image ``F`` times the PSF, so a component of amplitude ``a`` is ``a`` Jy.
* The minor cycles are the multi-scale CLEAN of Cornwell (2008, IEEE J. Sel. Topics Signal Process. 2, 793)
  with Gaussian scales (``B_s``, peak 1, FWHM ``s``; scale 0 is a point), the loop gain of Högbom (1974,
  A&AS 15, 417) and CASA's small-scale bias ``1 - bias * s / s_max``. At each iteration the pixel and scale
  with the largest bias-weighted ``D_s / sqrt(H_ss)`` is found (the decrease in chi^2 of a component is
  ``D_s^2 / H_ss``; ``D_s = B_s * D`` the smoothed residual, ``H_ss = (B_s * B_s * PSF)[0]``), ``gain * D_s / H_ss`` is
  added, and ``a (B_s' * B_s * PSF)`` is subtracted from every smoothed residual ``D_s'``. For Gaussians
  ``B_s' * B_s * PSF`` is the PSF convolved with a Gaussian of variance ``sigma_s^2 + sigma_s'^2``, so every cross
  term is one convolution of the PSF. The subtraction uses a window of the PSF, so the residual is approximate
  until the major cycle (the caller) recomputes it from the visibilities.
* The restoring beam is the elliptical Gaussian fitted to the PSF's main lobe above ``cutoff`` (CASA's ``psfcutoff``,
  0.35), which is what ``tclean`` fits. The restored image is the model convolved with the beam (peak 1) plus the
  residual.

Position angles are degrees East of North (the convention of ``crosscheck.alma_maxent.beam_transfer``).
"""

import numpy as np
from scipy import fft as sfft
from scipy.ndimage import label
from scipy.optimize import least_squares

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
MAS = np.pi / 180.0 / 3600.0 / 1000.0


# --------------------------------------------------------------- beams


def gaussian_image(shape, centre, bmaj, bmin, pa_deg):
    """Peak-1 elliptical Gaussian on pixels; ``bmaj``, ``bmin`` FWHM in pixels, ``pa_deg`` East of North, ``centre`` (row, col)."""
    rr, cc = np.indices(shape, dtype=float)
    east, north = -(cc - centre[1]), -(rr - centre[0])  # column 0 is East, row 0 is North
    pa = np.deg2rad(pa_deg)
    along = east * np.sin(pa) + north * np.cos(pa)
    across = east * np.cos(pa) - north * np.sin(pa)
    return np.exp(-4.0 * np.log(2.0) * ((along / bmaj) ** 2 + (across / bmin) ** 2))


def fit_beam(psf, pixel_mas, cutoff=0.35):
    """(bmaj_mas, bmin_mas, pa_deg, centre) of the Gaussian fitted to the PSF's main lobe.

    The lobe is the connected region around the peak where ``psf >= cutoff`` (plus a one-pixel margin); the fit is
    least squares over the lobe's pixels only, as CASA's ``psfcutoff`` fit does (virgil 0.4.1 restricts its fit the same way),
    with the peak fixed at 1 and the centre free (so even grids work).
    """
    psf = np.asarray(psf, float)
    peak = np.unravel_index(np.argmax(psf), psf.shape)
    lab, _ = label(psf >= cutoff)
    lobe = lab == lab[peak]
    rows, cols = np.where(lobe)
    r0, r1 = max(rows.min() - 1, 0), min(rows.max() + 2, psf.shape[0])
    c0, c1 = max(cols.min() - 1, 0), min(cols.max() + 2, psf.shape[1])
    sub = psf[r0:r1, c0:c1]
    lobe_sub = lobe[r0:r1, c0:c1]
    area = lobe.sum()
    r_eq = np.sqrt(area / np.pi)  # a circular start from the lobe's area at the cutoff
    start = [peak[0] - r0, peak[1] - c0, 2.0 * r_eq * np.sqrt(np.log(2) / np.log(1 / cutoff)), 2.0 * r_eq * np.sqrt(np.log(2) / np.log(1 / cutoff)) * 0.9, 0.0]

    def resid(p):
        return ((gaussian_image(sub.shape, (p[0], p[1]), p[2], p[3], p[4]) - sub) * lobe_sub).ravel()

    best = None
    for pa0 in (0.0, 45.0, 90.0, 135.0):  # the position angle is periodic and poorly conditioned for a round start
        s = list(start)
        s[4] = pa0
        fit = least_squares(resid, s, x_scale=[1, 1, s[2], s[2], 30.0])
        if best is None or fit.cost < best.cost:
            best = fit
    r, c, a, b, pa = best.x
    if b > a:
        a, b, pa = b, a, pa + 90.0
    return a * pixel_mas, b * pixel_mas, float(pa % 180.0), (r + r0, c + c0)


def sidelobe_level(psf):
    """Largest |PSF| outside the main lobe, the connected region around the peak down to the first null (CASA's
    ``maxpsfsidelobe``, used for the cycle threshold)."""
    psf = np.asarray(psf, float)
    peak = np.unravel_index(np.argmax(psf), psf.shape)
    lab, _ = label(psf > 0)
    out = np.where(lab == lab[peak], 0.0, np.abs(psf))
    return float(out.max())


# ------------------------------------------------------------- masks


def ellipse_mask(shape, centre, semimajor_pix, semiminor_pix, pa_deg):
    """Boolean ellipse (CASA's ``ellipse[[x, y], [maj, min], pa]``: semi-axes, major axis at ``pa`` East of North)."""
    rr, cc = np.indices(shape, dtype=float)
    east, north = -(cc - centre[1]), -(rr - centre[0])
    pa = np.deg2rad(pa_deg)
    along = east * np.sin(pa) + north * np.cos(pa)
    across = east * np.cos(pa) - north * np.sin(pa)
    return (along / semimajor_pix) ** 2 + (across / semiminor_pix) ** 2 <= 1.0


# ------------------------------------------------------- convolutions


class Smoother:
    """Convolution of images of one shape with unit-area Gaussians (zero-padded FFT, analytic transfer function)."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.pad = tuple(sfft.next_fast_len(2 * s, real=True) for s in shape)
        self._fy = np.fft.fftfreq(self.pad[0])[:, None]
        self._fx = np.fft.rfftfreq(self.pad[1])[None, :]

    def transfer(self, sigma_pix):
        return np.exp(-2.0 * np.pi**2 * sigma_pix**2 * (self._fx**2 + self._fy**2))

    def __call__(self, image, sigma_pix):
        """``image`` convolved with the unit-area Gaussian of standard deviation ``sigma_pix`` (0: unchanged)."""
        if sigma_pix == 0:
            return np.array(image, float)
        spec = sfft.rfft2(image, s=self.pad)
        out = sfft.irfft2(spec * self.transfer(sigma_pix), s=self.pad)
        return out[: self.shape[0], : self.shape[1]]


def blob_area(sigma_pix):
    """Area in pixels of the peak-1 Gaussian of deviation ``sigma_pix`` (a point is 1 pixel)."""
    return 1.0 if sigma_pix == 0 else 2.0 * np.pi * sigma_pix**2


def model_image(components, scales_pix):
    """The model in Jy per pixel: ``sum_s B_s * components_s`` (``B_s`` peak 1, FWHM ``scales_pix[s]``)."""
    sm = Smoother(components.shape[1:])
    return sum(blob_area(s * FWHM_TO_SIGMA) * sm(c, s * FWHM_TO_SIGMA) for c, s in zip(components, scales_pix) if np.any(c))


def restore(model, residual, beam_pix, centre_for_beam=None):
    """Restored image in Jy/beam: the model (Jy/pixel) convolved with the peak-1 elliptical Gaussian ``beam_pix`` =
    (bmaj, bmin, pa) plus the residual."""
    n0, n1 = model.shape
    kernel = gaussian_image(model.shape, ((n0 - 1) / 2, (n1 - 1) / 2), *beam_pix)
    pad = tuple(sfft.next_fast_len(2 * s, real=True) for s in model.shape)
    conv = sfft.irfft2(sfft.rfft2(model, s=pad) * sfft.rfft2(kernel, s=pad), s=pad)
    r, c = (n0 - 1) // 2, (n1 - 1) // 2  # the kernel's centre pixel is the origin of the convolution
    return conv[r : r + n0, c : c + n1] + residual


# ---------------------------------------------------- minor cycles


def multiscale_minor(
    residual, psf, scales_pix, mask, *, gain=0.1, niter=300, threshold=0.0, bias=0.6, patch=256, max_scale_patch=None
):
    """One minor cycle of multi-scale CLEAN on ``residual`` (Jy/beam) with the PSF ``psf`` (peak 1 at the centre pixel).

    Parameters
    ----------
    residual, psf : (n, n) arrays on the same grid (odd ``n`` for ``psf``).
    scales_pix : sequence of float
        FWHM of the Gaussian scales in pixels (0 is a point).
    mask : (n, n) bool
        Where components may be placed.
    gain, niter, threshold, bias : the loop gain, the iteration limit, the peak (Jy/beam) of the dirty response of the next
        component below which to stop, and CASA's small-scale bias.
    patch : int
        Half-width in pixels of the window of the PSF subtracted at each iteration (widened for broad scales to 2.5 sigma).

    Returns
    -------
    components : (n_scales, n, n) array of amplitudes added (Jy; see the module docstring)
    n_iter : int
    peaks : list of the peak (Jy/beam, signed) of the dirty response of the component chosen at each iteration
    """
    residual = np.asarray(residual, float)
    n0, n1 = residual.shape
    scales = [float(s) for s in scales_pix]
    sig = [s * FWHM_TO_SIGMA for s in scales]
    smax = max(scales)
    wbias = np.array([1.0 - (bias * s / smax if smax > 0 else 0.0) for s in scales])
    # crop to the mask's bounding box: nothing outside it is ever chosen
    rows, cols = np.where(mask)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    cm = mask[r0:r1, c0:c1]
    ny, nx = cm.shape
    sm = Smoother((ny, nx))
    ctr = (psf.shape[0] // 2, psf.shape[1] // 2)
    nsc = len(scales)
    # windows of K_{s's} = PSF * G(sigma_s'^2 + sigma_s^2) times the blob areas
    w_pair = {}
    kern = {}
    for i in range(nsc):
        for j in range(i, nsc):
            seff = np.hypot(sig[i], sig[j])
            w = int(min(max(patch, np.ceil(2.5 * seff)), ctr[0], ctr[1]))
            ext = int(min(w + np.ceil(4 * seff), ctr[0], ctr[1]))  # extra margin so the Gaussian smoothing is accurate at the window's edge
            sub = psf[ctr[0] - ext : ctr[0] + ext + 1, ctr[1] - ext : ctr[1] + ext + 1]
            k = Smoother(sub.shape)(sub, seff)[ext - w : ext + w + 1, ext - w : ext + w + 1]
            k = blob_area(sig[i]) * blob_area(sig[j]) * k
            kern[(i, j)] = kern[(j, i)] = k
            w_pair[(i, j)] = w_pair[(j, i)] = w
    hdiag = np.array([kern[(i, i)][w_pair[(i, i)], w_pair[(i, i)]] for i in range(nsc)])
    # Jy/beam at its centre of the dirty response of a unit component of each scale: the amplitude the stopping test uses
    pk = np.array([blob_area(g) * (Smoother(psf.shape)(psf, g)[ctr] if g else psf[ctr]) for g in sig])
    norm = wbias / np.sqrt(hdiag)  # E_s = D_s * norm_s: the score the peak search compares
    es = np.stack([blob_area(sig[i]) * sm(residual[r0:r1, c0:c1], sig[i]) * norm[i] for i in range(nsc)])
    es *= cm[None]
    # tile maxima for a fast peak search
    tile = 64
    py, px = -(-ny // tile) * tile, -(-nx // tile) * tile
    big = np.zeros((nsc, py, px))
    big[:, :ny, :nx] = es
    tmax = np.abs(big).reshape(nsc, py // tile, tile, px // tile, tile).max(axis=(2, 4))
    comps = np.zeros((nsc, ny, nx))
    peaks = []
    it = 0
    for it in range(1, niter + 1):
        s, ty, tx = np.unravel_index(np.argmax(tmax), tmax.shape)
        blk = big[s, ty * tile : (ty + 1) * tile, tx * tile : (tx + 1) * tile]
        iy, ix = np.unravel_index(np.argmax(np.abs(blk)), blk.shape)
        y, x = ty * tile + iy, tx * tile + ix
        amp = big[s, y, x] / (norm[s] * hdiag[s])  # D_s / H_ss
        amp_jy = amp * pk[s]
        if abs(amp_jy) < threshold:
            it -= 1
            break
        peaks.append(float(amp_jy))
        a = gain * amp
        comps[s, y, x] += a
        for t in range(nsc):
            w = w_pair[(s, t)]
            k = kern[(t, s)]
            ya, yb = max(y - w, 0), min(y + w + 1, ny)
            xa, xb = max(x - w, 0), min(x + w + 1, nx)
            kk = k[ya - (y - w) : yb - (y - w), xa - (x - w) : xb - (x - w)]
            reg = big[t, ya:yb, xa:xb]
            reg -= a * norm[t] * kk
            reg *= cm[ya:yb, xa:xb]  # outside the mask the score is never used
            ta, tb, ua, ub = ya // tile, -(-yb // tile), xa // tile, -(-xb // tile)
            tmax[t, ta:tb, ua:ub] = np.abs(big[t, ta * tile : tb * tile, ua * tile : ub * tile]).reshape(tb - ta, tile, ub - ua, tile).max(axis=(1, 3))
    out = np.zeros((nsc, n0, n1))
    out[:, r0:r1, c0:c1] = comps
    return out, it, peaks


# --------------------------------------------------- image statistics


def annulus_rms(image, centre, r_in_pix, r_out_pix, exclude=None):
    """rms of ``image`` over the annulus ``r_in <= r <= r_out`` around ``centre`` (row, col), optionally excluding a mask."""
    rr, cc = np.indices(image.shape, dtype=float)
    r = np.hypot(rr - centre[0], cc - centre[1])
    sel = (r >= r_in_pix) & (r <= r_out_pix)
    if exclude is not None:
        sel &= ~exclude
    return float(np.sqrt(np.mean(image[sel] ** 2))), int(sel.sum())


def radial_profile(image, centre, pixel_arcsec, incl_deg, pa_deg, edges_arcsec):
    """Mean of ``image`` in annuli of the disc plane (deprojected radius; ``pa_deg`` the major axis East of North,
    ``incl_deg`` the inclination). Returns (bin centres, means, counts)."""
    rr, cc = np.indices(image.shape, dtype=float)
    east, north = -(cc - centre[1]) * pixel_arcsec, -(rr - centre[0]) * pixel_arcsec
    pa = np.deg2rad(pa_deg)
    major = east * np.sin(pa) + north * np.cos(pa)
    minor = (east * np.cos(pa) - north * np.sin(pa)) / np.cos(np.deg2rad(incl_deg))
    r = np.hypot(major, minor)
    which = np.digitize(r.ravel(), edges_arcsec) - 1
    ok = (which >= 0) & (which < len(edges_arcsec) - 1) & np.isfinite(image.ravel())
    cnt = np.bincount(which[ok], minlength=len(edges_arcsec) - 1)
    tot = np.bincount(which[ok], weights=image.ravel()[ok], minlength=len(edges_arcsec) - 1)
    mid = 0.5 * (np.asarray(edges_arcsec)[1:] + np.asarray(edges_arcsec)[:-1])
    return mid, np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan), cnt


def uv_taper(u, v, bmaj_mas, bmin_mas, pa_deg):
    """The Gaussian ``uvtaper`` of CASA's ``tclean``: the Fourier transform of an elliptical Gaussian of FWHM ``bmaj`` x
    ``bmin`` (image plane, major axis at ``pa_deg`` East of North), which is the factor the weights are multiplied
    by; ``u`` East and ``v`` North in wavelengths. The tapered PSF is the PSF convolved with that Gaussian
    (Bracewell, the Gaussian pair)."""
    pa = np.deg2rad(pa_deg)
    along = u * np.sin(pa) + v * np.cos(pa)
    across = u * np.cos(pa) - v * np.sin(pa)
    return np.exp(-(np.pi**2) * ((bmaj_mas * MAS * along) ** 2 + (bmin_mas * MAS * across) ** 2) / (4.0 * np.log(2.0)))
