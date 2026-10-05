"""Score 2004 data2 images with the contest's own metric (Lawson et al. 2004, Eq. 2).

    python scripts/score_2004.py ENSEMBLE_NPZ [--data DIR] [--out DIR]

The truth is rebuilt from the model published *after* the contest
(docs/contest_scoring_sources.md, "data2 truth model"). It is used here for
scoring only, never for imaging. The model:

- A: an elliptical limb-darkened disk, 5.5 mas major axis, axis ratio 0.7,
  major axis at PA 120°. Its limb darkening came from a 7000 K, log g 4 model
  atmosphere, which the paper does not tabulate: we use a linear law with
  u = ``U_LD`` at 550 nm, and report how the score changes with u.
- A spot on A, 1 mas across, 2.5 mas from A's centre at PA 150°, at 1.5 x the
  star's temperature. Its surface brightness is the Planck ratio at 550 nm
  (about 3.7) times the limb-darkened profile beneath it, and only the part
  on A's disk counts.
- B: a 0.5 mas uniform disk, 10 mas from A at PA 90°.

The paper does not state the A:B flux ratio, so it is fitted to the data's V²
(a one-parameter scan, NumPy DFT; V² does not depend on the sign convention of
the phases). The paper also leaves out the reference grid, the comparison box
and the flux normalisation. We use 0.08 mas pixels, which reproduce the
published reference peak, and normalise the truth and the entry to unit sum inside a box
holding all of the truth's emission. The published data2 reference peak
(3.07e-3 after correcting the exponent typo) gives a check on the grid.

Metric: σ = [Σ p_ref (p − p_ref)² / Σ p_ref]^½ over the box, reported as
σ / peak(ref), lower is better. Entries were aligned on a feature by hand in
the contest; we align by the integer shift that maximises the
cross-correlation, and also score the inversion through the origin (the
closure phases fix the orientation, so the unflipped score is the one that
counts; the flipped one is reported as a check).

The unpublished details, fitted to the data (V² and closure phases, a
Nelder-Mead fit, for scoring only), give u = 0.78, a spot contrast of 7.1 and
B/A = 0.096 (``FITTED``). Even then the truth fits at V² χ²/N ≈ 50 and closure
χ²/N ≈ 9, so the data carry errors beyond their quoted noise: the 2004 rules
allowed calibration errors. Other readings of the geometry (PA 120° as the
minor axis, 5.5 mas as the minor diameter, the spot at PA −30°) fit far worse
(V² χ²/N 100-1100). Our score moves little between the two truths (campaign
1: 0.250 published, 0.260 fitted); both are reported.

Published data2 σ/peak (Table 2): BSMEM 0.116, WISARD 0.163, MIRA 0.532,
VLBMEM 0.798.
"""

import argparse
import pathlib

import numpy as np

# The reference grid is not published. 0.08 mas pixels reproduce the published
# data2 reference peak (3.16e-3 against 3.07e-3, unit sum); 0.05 mas (data1's)
# gives 1.2e-3. The score itself moves by 0.005 between the two.
PIXEL = 0.08  # mas
NPIX = 301  # 24 mas across: A at the centre and B 10 mas east both inside
U_LD = 0.5  # linear limb darkening at 550 nm for a 7000 K, log g 4 star (assumed)
WAVEL = 5.5e-7
FITTED = {"u_ld": 0.78, "spot_contrast": 7.06, "ratio": 0.096}  # see the module docstring
PUBLISHED = {"BSMEM": 0.116, "WISARD": 0.163, "MIRA": 0.532, "VLBMEM": 0.798}
MAS = np.pi / 180 / 3600e3


def planck_ratio(t_hot, t_cool, wavel=WAVEL):
    x = 6.62607015e-34 * 2.99792458e8 / (wavel * 1.380649e-23)
    return np.expm1(x / t_cool) / np.expm1(x / t_hot)


def grid(npix=NPIX, pixel=PIXEL):
    """Sky offsets in mas, East (+RA) towards column 0 and North towards row 0,
    as the campaign's images are stored (rendered by virgil, plotted with
    East to the left and North up)."""
    c = (npix - 1) / 2
    ra = (c - np.arange(npix)) * pixel
    dec = (c - np.arange(npix)) * pixel
    return np.meshgrid(ra, dec)  # x = RA offset (columns), y = Dec offset (rows)


def offset(pa_deg, sep):
    """(ΔRA, ΔDec) of a point at separation sep and PA (East of North)."""
    pa = np.radians(pa_deg)
    return sep * np.sin(pa), sep * np.cos(pa)


def truth_parts(u_ld=U_LD, npix=NPIX, pixel=PIXEL, spot_contrast=None):
    """A (with its spot) and B, each normalised to unit sum, on the grid."""
    x, y = grid(npix, pixel)
    # A's frame: along the major axis (PA 120°) and across it.
    pa = np.radians(120.0)
    along = x * np.sin(pa) + y * np.cos(pa)
    across = x * np.cos(pa) - y * np.sin(pa)
    a, b = 5.5 / 2, 0.7 * 5.5 / 2
    r2 = (along / a) ** 2 + (across / b) ** 2
    inside = r2 < 1.0
    mu = np.sqrt(np.clip(1.0 - r2, 0.0, None))
    star = np.where(inside, 1.0 - u_ld * (1.0 - mu), 0.0)
    sx, sy = offset(150.0, 2.5)
    spot = inside & ((x - sx) ** 2 + (y - sy) ** 2 < 0.5**2)
    contrast = planck_ratio(1.5 * 7000.0, 7000.0) if spot_contrast is None else spot_contrast
    star = np.where(spot, star * contrast, star)
    bx, by = offset(90.0, 10.0)
    companion = (((x - bx) ** 2 + (y - by) ** 2) < 0.25**2).astype(float)
    return star / star.sum(), companion / companion.sum()


def visibilities(image, u, v, npix=NPIX, pixel=PIXEL):
    """Complex visibilities of a unit-sum image at (u, v) in wavelengths."""
    x, y = grid(npix, pixel)
    keep = image > 0
    phase = -2j * np.pi * (np.outer(u, x[keep]) + np.outer(v, y[keep])) * MAS
    return np.exp(phase) @ image[keep]


def read_v2(path):
    from astropy.io import fits

    with fits.open(path) as h:
        d = h["OI_VIS2"].data
        ok = ~d["FLAG"].ravel() if "FLAG" in d.columns.names else np.ones(len(d), bool)
        u, v = d["UCOORD"].ravel() / WAVEL, d["VCOORD"].ravel() / WAVEL
        return u[ok], v[ok], d["VIS2DATA"].ravel()[ok], d["VIS2ERR"].ravel()[ok]


def fit_ratio(star, companion, u, v, v2, err, ratios=np.geomspace(0.01, 2.0, 400)):
    """B/A flux ratio minimising the V² χ² (a scan: one parameter)."""
    va, vb = visibilities(star, u, v), visibilities(companion, u, v)
    chi2 = [np.sum(((np.abs(va + r * vb) / (1 + r)) ** 2 - v2) ** 2 / err**2) for r in ratios]
    i = int(np.argmin(chi2))
    return float(ratios[i]), float(chi2[i]) / len(v2)


def read_t3(path):
    from astropy.io import fits

    with fits.open(path) as h:
        d = h["OI_T3"].data
        ok = ~d["FLAG"].ravel() if "FLAG" in d.columns.names else np.ones(len(d), bool)
        u1, v1 = d["U1COORD"].ravel() / WAVEL, d["V1COORD"].ravel() / WAVEL
        u2, v2 = d["U2COORD"].ravel() / WAVEL, d["V2COORD"].ravel() / WAVEL
        return u1[ok], v1[ok], u2[ok], v2[ok], np.radians(d["T3PHI"].ravel()[ok]), np.radians(d["T3PHIERR"].ravel()[ok])


def closure_chi2(image, t3):
    """χ²/N of the closure phases for ``image`` and for its inversion: checks
    that the truth's orientation on our grid agrees with the data."""
    u1, v1, u2, v2, phi, err = t3
    out = []
    for img in (image, image[::-1, ::-1]):
        bis = visibilities(img, u1, v1) * visibilities(img, u2, v2) * np.conj(visibilities(img, u1 + u2, v1 + v2))
        d = np.angle(np.exp(1j * (np.angle(bis) - phi)))
        out.append(float(np.mean((d / err) ** 2)))
    return out


def truth_image(ratio, u_ld=U_LD, spot_contrast=None):
    star, companion = truth_parts(u_ld, spot_contrast=spot_contrast)
    img = star + ratio * companion
    return img / img.sum()


def resample(image, fov, npix=NPIX):
    """An image ``fov`` mas across onto an ``npix`` grid of PIXEL mas
    (bilinear), unit sum."""
    from scipy.ndimage import map_coordinates

    n = image.shape[0]
    x, y = grid(npix)
    # Pixel (row, col) of the entry at each truth pixel; same conventions.
    c = (n - 1) / 2
    col = c - x / (fov / n)
    row = c - y / (fov / n)
    out = map_coordinates(image, [row, col], order=1, mode="constant", cval=0.0)
    return out / out.sum()


def sigma_over_peak(entry, ref, box):
    e = entry * box
    r = ref * box
    e, r = e / e.sum(), r / r.sum()
    sigma = np.sqrt(np.sum(r * (e - r) ** 2) / np.sum(r))
    return float(sigma / r.max()), float(sigma), float(r.max())


def shifted(image, shift):
    """``image`` translated by whole pixels (rows, columns), zero-filled: flux
    leaving one edge does not wrap onto the other."""
    out = np.zeros_like(image)
    (di, dj), (n, m) = shift, image.shape
    out[max(di, 0) : n + min(di, 0), max(dj, 0) : m + min(dj, 0)] = image[
        max(-di, 0) : n + min(-di, 0), max(-dj, 0) : m + min(-dj, 0)
    ]
    return out


MAX_SHIFT = 40  # pixels (3.2 mas): the alignment search's range


def aligned(entry, ref, max_shift=MAX_SHIFT):
    """``entry``, on a grid ``max_shift`` pixels wider than ``ref`` on every
    side, translated by the whole-pixel offset that best correlates it with
    ``ref`` and cropped to ``ref``'s grid (the contest aligned entries on a
    feature by hand). The padding keeps emission that the shift brings into
    view; nothing wraps."""
    from scipy.signal import fftconvolve

    padded = np.pad(ref, max_shift)
    corr = fftconvolve(padded, entry[::-1, ::-1], mode="same")
    c = np.array(corr.shape) // 2
    window = corr[c[0] - max_shift : c[0] + max_shift + 1, c[1] - max_shift : c[1] + max_shift + 1]
    di, dj = np.unravel_index(np.argmax(window), window.shape)
    shift = (int(di - max_shift), int(dj - max_shift))
    moved = shifted(entry, shift)[max_shift:-max_shift, max_shift:-max_shift]
    return moved, shift


def score(entry, fov, ratio, u_ld=U_LD):
    ref = truth_image(ratio, u_ld)
    # A box around all of the truth's emission, with a 0.5 mas margin (the
    # published box is not given).
    box = np.zeros_like(ref, bool)
    rows, cols = np.nonzero(ref > 0)
    box[rows.min() - 10 : rows.max() + 11, cols.min() - 10 : cols.max() + 11] = True
    out = {}
    for name, img in (("as imaged", entry), ("inverted", entry[::-1, ::-1])):
        moved, shift = aligned(resample(img, fov, NPIX + 2 * MAX_SHIFT), ref)
        out[name] = (*sigma_over_peak(moved, ref, box), shift)
    return out, ref


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ensemble", help="<label>_ensemble.npz from combine_ensemble.py, or a member npz")
    parser.add_argument("--data", default="~/data/imaging_contests")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    data_path = pathlib.Path(args.data).expanduser() / "2004" / "2004-data2.fits"
    d = np.load(args.ensemble)
    entry = np.asarray(d["mean"] if "mean" in d else d["ref_image"], float)
    fov = float(d["fov"] if "fov" in d else d["ref_fov"])
    u, v, v2, err = read_v2(data_path)
    t3 = read_t3(data_path)
    lines = []
    for u_ld in (0.3, U_LD, 0.7):
        star, companion = truth_parts(u_ld)
        ratio, chi2 = fit_ratio(star, companion, u, v, v2, err)
        scores, ref = score(entry, fov, ratio, u_ld)
        cp, cp_inv = closure_chi2(ref, t3)
        lines.append(f"u_ld={u_ld}: B/A={ratio:.3f} truth V² χ²/N={chi2:.2f}, closure χ²/N {cp:.2f} "
                     f"(inverted truth {cp_inv:.2f}), ref peak={ref.max():.3e} (published 3.07e-3)")
        for name, (sp, s, peak, shift) in scores.items():
            lines.append(f"  {name:9s} σ/peak={sp:.3f} σ={s:.2e} shift={shift}")
    ref = truth_image(FITTED["ratio"], FITTED["u_ld"], FITTED["spot_contrast"])
    box = np.zeros_like(ref, bool)
    rows, cols = np.nonzero(ref > 0)
    box[rows.min() - 10 : rows.max() + 11, cols.min() - 10 : cols.max() + 11] = True
    moved, shift = aligned(resample(entry, fov, NPIX + 2 * MAX_SHIFT), ref)
    lines.append(f"fitted truth (u_ld={FITTED['u_ld']}, spot contrast {FITTED['spot_contrast']}, B/A={FITTED['ratio']}): "
                 f"σ/peak={sigma_over_peak(moved, ref, box)[0]:.3f} shift={shift}")
    lines.append("published data2 σ/peak: " + ", ".join(f"{k} {v}" for k, v in PUBLISHED.items()))
    text = "\n".join(lines)
    print(text)
    if args.out:
        out = pathlib.Path(args.out).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        (out / "2004_data2_score.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
