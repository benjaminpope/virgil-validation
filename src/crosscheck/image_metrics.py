"""Image-quality metrics of the interferometric imaging contests.

Nothing here imports virgil. Images are 2-D arrays in the sky convention of
``sky.pixel_image``: row 0 North, column 0 East, centre at
((n - 1) / 2, (n - 1) / 2). The metrics follow each contest's published
definition (docs/contest_scoring_sources.md):

- ``lawson_sigma_over_peak``: 2004 (Lawson et al. 2004, Eq. 2), a
  reference-weighted RMS divided by the reference peak; lower is better.
- ``rms_convolved``: 2008/2010/2012 style, the RMS of the difference of
  unit-sum images after convolving both to a common resolution, times 10⁶;
  lower is better.
- ``l1_score``: 2024 (ImageMetrics), 1 − Σ|a e − r| / Σ r with the best
  flux scale a ≥ 0; higher is better (1 is perfect).
- ``ncc``: normalised cross-correlation, a scale-free check; 1 is perfect.

``score`` aligns an entry on the reference by the whole-pixel shift that
maximises their cross-correlation (the contests aligned entries by hand or
by fitted shifts), and for V²-only data also scores the inversion through
the origin and keeps the better (V² cannot tell them apart).
"""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import fftconvolve

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def offsets(npix, pixel):
    """East and North offsets (mas) of the pixel centres: row 0 North,
    column 0 East."""
    c = (npix - 1) / 2.0
    east = (c - np.arange(npix)) * pixel
    north = (c - np.arange(npix)) * pixel
    return np.meshgrid(east, north)


def resample(image, fov, npix, pixel):
    """An image ``fov`` mas across onto an ``npix`` grid of ``pixel`` mas,
    bilinearly, renormalised to unit sum. Coarser targets are smoothed first
    (about one target pixel), so that sampling cannot skip compact flux."""
    image = np.asarray(image, float)
    n = image.shape[0]
    ratio = pixel / (fov / n)
    if ratio > 1:
        image = gaussian_filter(image, sigma=0.5 * ratio, mode="constant")
    east, north = offsets(npix, pixel)
    c = (n - 1) / 2.0
    col = c - east / (fov / n)
    row = c - north / (fov / n)
    out = map_coordinates(image, [row, col], order=1, mode="constant", cval=0.0)
    total = out.sum()
    return out / total if total > 0 else out


def shifted(image, shift):
    """``image`` translated by whole pixels (rows, columns), zero-filled: flux
    leaving one edge does not wrap onto the other."""
    out = np.zeros_like(image)
    (di, dj), (n, m) = shift, image.shape
    out[max(di, 0) : n + min(di, 0), max(dj, 0) : m + min(dj, 0)] = image[
        max(-di, 0) : n + min(-di, 0), max(-dj, 0) : m + min(-dj, 0)
    ]
    return out


def align(entry, ref, max_shift):
    """``entry`` (on a grid ``max_shift`` pixels wider than ``ref`` on every
    side) translated by the whole-pixel offset that maximises its
    cross-correlation with ``ref``, cropped to ``ref``'s grid, and that shift
    (rows, columns). The padding keeps emission a shift brings into view;
    translation is zero-filled, so nothing wraps."""
    padded = np.pad(ref, max_shift)
    corr = fftconvolve(padded, entry[::-1, ::-1], mode="same")
    c = np.array(corr.shape) // 2
    window = corr[c[0] - max_shift : c[0] + max_shift + 1, c[1] - max_shift : c[1] + max_shift + 1]
    di, dj = np.unravel_index(np.argmax(window), window.shape)
    shift = (int(di - max_shift), int(dj - max_shift))
    if max_shift == 0:
        return entry, shift
    return shifted(entry, shift)[max_shift:-max_shift, max_shift:-max_shift], shift


def _unit(image, box=None):
    image = np.asarray(image, float) if box is None else np.where(box, image, 0.0)
    total = image.sum()
    return image / total if total > 0 else image


def lawson_sigma_over_peak(entry, ref, box=None):
    """2004: σ = [Σ r (e − r)² / Σ r]^½ over the box, both unit sum in it,
    divided by the reference's peak. Returns (σ/peak, σ, peak)."""
    e, r = _unit(entry, box), _unit(ref, box)
    sigma = float(np.sqrt(np.sum(r * (e - r) ** 2) / np.sum(r)))
    peak = float(r.max())
    return sigma / peak, sigma, peak


def rms_convolved(entry, ref, fwhm_pix, box=None):
    """10⁶ × RMS of (e − r) after convolving both with a Gaussian of FWHM
    ``fwhm_pix`` pixels and normalising each to unit sum in the box."""
    s = fwhm_pix / FWHM_PER_SIGMA
    e = _unit(gaussian_filter(np.asarray(entry, float), s, mode="constant"), box)
    r = _unit(gaussian_filter(np.asarray(ref, float), s, mode="constant"), box)
    if box is not None:
        e, r = e[box], r[box]
    return float(1e6 * np.sqrt(np.mean((e - r) ** 2)))


def _weighted_median(values, weights):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w)
    return float(v[np.searchsorted(cum, 0.5 * cum[-1])])


def l1_score(entry, ref, fwhm_pix=None):
    """2024: 1 − min_a Σ|a e − r| / Σ r, with a ≥ 0 the L1-optimal flux scale
    (the e-weighted median of r/e), after optional convolution of both to
    ``fwhm_pix``."""
    e, r = np.asarray(entry, float), np.asarray(ref, float)
    if fwhm_pix:
        s = fwhm_pix / FWHM_PER_SIGMA
        e = gaussian_filter(e, s, mode="constant")
        r = gaussian_filter(r, s, mode="constant")
    # Pixels with no entry flux carry no weight in the median; a relative
    # floor also drops the denormal tails of a blurred entry, whose ratios
    # r/e overflowed (C2g).
    keep = e > 1e-12 * e.max() if e.size and e.max() > 0 else np.zeros(e.shape, bool)
    a = max(_weighted_median(r[keep] / e[keep], e[keep]), 0.0) if keep.any() else 0.0
    return float(1.0 - np.sum(np.abs(a * e - r)) / np.sum(r))


def ncc(entry, ref):
    """Normalised (Pearson) cross-correlation of the two images."""
    e = np.ravel(entry) - np.mean(entry)
    r = np.ravel(ref) - np.mean(ref)
    return float(np.dot(e, r) / np.sqrt(np.dot(e, e) * np.dot(r, r)))


def score(entry, fov, ref, pixel, beam_mas, v2_only=False, box_margin_mas=None, max_shift=None):
    """Every metric for ``entry`` (``fov`` mas across) against ``ref`` (on a
    grid of ``pixel`` mas). Resolution for the convolved metrics: the beam,
    λ/B_max in mas. Alignment searches ``max_shift`` pixels (default n/4) on a
    grid padded by as much. Returns a dict; with ``v2_only`` the better of the image
    and its inversion (by NCC) is kept and ``inverted`` says which."""
    n = ref.shape[0]
    ref = _unit(ref)
    margin = beam_mas if box_margin_mas is None else box_margin_mas
    rows, cols = np.nonzero(ref > 1e-6 * ref.max())
    m = int(np.ceil(margin / pixel))
    box = np.zeros_like(ref, bool)
    box[max(rows.min() - m, 0) : rows.max() + m + 1, max(cols.min() - m, 0) : cols.max() + m + 1] = True
    max_shift = n // 4 if max_shift is None else max_shift
    candidates = [("as imaged", np.asarray(entry, float))]
    if v2_only:
        candidates.append(("inverted", np.asarray(entry, float)[::-1, ::-1]))
    results = []
    for name, img in candidates:
        moved, shift = align(resample(img, fov, n + 2 * max_shift, pixel), ref, max_shift)
        fwhm_pix = beam_mas / pixel
        results.append({
            "orientation": name,
            "shift": shift,
            "lawson": lawson_sigma_over_peak(moved, ref, box)[0],
            "rms_e6": rms_convolved(moved, ref, fwhm_pix, box),
            "l1": l1_score(moved, ref, 0.5 * fwhm_pix),
            "ncc": ncc(moved, ref),
        })
    best = max(results, key=lambda d: d["ncc"])
    best["alternatives"] = results
    return best
