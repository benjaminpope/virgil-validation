"""Score 2004 data2 images with the contest's own metric (Lawson et al. 2004, Eq. 2).

    python scripts/score_2004.py ENSEMBLE_NPZ [--data DIR] [--out DIR]

The truth is rebuilt from the model published *after* the contest
(docs/contest_scoring_sources.md, "data2 truth model"). It is used here for
scoring only, never for imaging. The model:

- A: an elliptical limb-darkened disk, 5.5 mas major axis, axis ratio 0.7,
  major axis at PA 120°. Its limb darkening came from a 7000 K, log g 4 model
  atmosphere. The paper does not tabulate it; OYSTER's linear coefficient is
  taken to be Van Hamme (1993, AJ 106, 2096) Table 1's bolometric one for
  7000 K, log g 4, u = 0.469 (``U_LD``), which Hummel's papers cite as the
  source of his coefficients. The data pin u to 0.48 ± 0.02 (V² χ²/N 1.74 at
  0.44, 2.6 at 0.55), so a 550 nm monochromatic value (about 0.55) is
  rejected. The score is reported for u = 0.3, ``U_LD`` and 0.7.
- A spot, 1 mas across, 2.5 mas from A's centre at PA 150°, at 1.5 x the
  star's temperature. It is a uniform disk added to the star, not clipped at
  the limb (the limb is at 2.45 mas on that PA, so half the spot hangs off the
  star), with a surface brightness of (contrast - 1) x the limb brightness
  1 - u above the photosphere's; the contrast is the Planck ratio at 550 nm,
  about 3.7. Of the spot variants tried (scripts/diagnose_2004.py; plan C4b,
  2026-10-14) only this one, with no free spot parameter, fits the data (V² χ²/N
  1.35 against 2.84 for the spot clipped to the disk).
- B: a 0.5 mas uniform disk, 10 mas from A at PA 90°.

The paper does not state the A:B flux ratio, so it is fitted to the data's V²
(a one-parameter scan, NumPy DFT; V² does not depend on the sign convention of
the phases). The paper also leaves out the reference grid and the comparison
box. We use 0.08 mas pixels, which reproduce the published reference peak,
and normalise the truth image and the entry to unit sum inside a box holding
all of the truth's emission.

Two normalisations, which differ by the spot's excess flux. The *scoring*
image (``truth_image``) has unit sum, as the paper's comparison does. The
*visibilities* the data were normalised with do not include the spot's
brightening: V(0) = (1 + spot excess + B/A) / (1 + B/A) ≈ 1.07, so V²(0) ≈
1.145, not 1 (``truth_vis_image``; the short-baseline V² of data2 sit 7-10 %
above a unit-sum model on every baseline). The spot's excess flux is therefore
left out of the star's normalisation: ``truth_parts`` normalises the *unspotted*
star to unit sum. The published data2 reference peak
(3.07e-3 after correcting the exponent typo) gives a check on the grid.

Metric: σ = [Σ p_ref (p − p_ref)² / Σ p_ref]^½ over the box, reported as
σ / peak(ref), lower is better. Entries were aligned on a feature by hand in
the contest; we align by the integer shift that maximises the
cross-correlation, and also score the inversion through the origin (the
closure phases fix the orientation, so the unflipped score is the one that
counts; the flipped one is reported as a check).

The unpublished details, fitted to the data (V² and closure phases, a
Nelder-Mead fit, for scoring only; ``fit_nuisances``), give the values in
``FITTED``. With the normalisation and the spot above the truth fits at V² χ²/N
1.3, closure χ²/N 0.55 and T3AMP χ²/N 0.6-0.8 (B/A 0.112, u = 0.469; the free
spot contrast comes out at 3.68, the Planck ratio 3.70, which is what ``FITTED``
uses). With a unit-sum normalisation it fitted only at V² χ²/N ≈ 50, which was
once read as calibration error in the data (the 2004 rules allowed it) but is
the normalisation: the BSC1948 test binary from the same chain fits at V² scale
0.997. With the spot clipped to the disk the fit stalled at V² χ²/N 2.8-3.0: it
was the spot, not bandwidth smearing (20 nm worsens the fit), a quadratic
limb-darkening law (V² 2.4) or pixelization (3.05 against 3.02 at 0.02 mas).
A scan of the baseline scale pins it at 1.000 within 0.3 %, so the transform
and uv sampling are right. Other readings of the geometry (PA 120° as the
minor axis, 5.5 mas as the minor diameter, the spot at PA −30°) fit far worse
(V² χ²/N 100-1100), and the closure phases reject the inverted truth (χ²/N ~900),
so there is no coordinate or orientation error. Our score moves little between
the two truths; both are reported.

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
U_LD = 0.469  # linear limb darkening, Van Hamme (1993) Table 1 (bolometric), Kurucz model 7000 K, log g 4
WAVEL = 5.5e-7
FITTED = {
    "u_ld": U_LD,
    "spot_contrast": 3.7,
    "ratio": 0.112,
}  # fit_nuisances; see the module docstring
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
    """A (with its spot) and B on the grid. B is normalised to unit sum; A is
    normalised so that the star *without* its spot has unit sum, so the spot's
    excess flux is left out of the visibility normalisation (see the module
    docstring): ``A.sum()`` is 1 plus the excess."""
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
    # The spot is a uniform disk, not clipped at the limb (it sits at 2.5 mas,
    # on the limb's 2.45 mas, so half of it is off the star), added to the
    # star: its surface brightness exceeds the photosphere's limb brightness
    # (1 - u) by (contrast - 1) times it. See the module docstring.
    sx, sy = offset(150.0, 2.5)
    spot = (x - sx) ** 2 + (y - sy) ** 2 < 0.5**2
    contrast = planck_ratio(1.5 * 7000.0, 7000.0) if spot_contrast is None else spot_contrast
    spotted = star + (contrast - 1.0) * (1.0 - u_ld) * spot
    bx, by = offset(90.0, 10.0)
    companion = (((x - bx) ** 2 + (y - by) ** 2) < 0.25**2).astype(float)
    return spotted / star.sum(), companion / companion.sum()


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
    """B/A flux ratio minimising the V² χ² (a scan: one parameter). ``star`` and
    ``companion`` are ``truth_parts``, so V(0) = (star.sum() + r) / (1 + r)."""
    va, vb = visibilities(star, u, v), visibilities(companion, u, v)
    chi2 = [np.sum(((np.abs(va + r * vb) / (1 + r)) ** 2 - v2) ** 2 / err**2) for r in ratios]
    i = int(np.argmin(chi2))
    return float(ratios[i]), float(chi2[i]) / len(v2)


def model_vis(u, v, ratio, u_ld=U_LD, spot_contrast=None):
    """Analytic complex visibilities of the contest model, normalised as
    ``truth_vis_image`` (V(0) = 1 + spot excess + B/A over 1 + B/A), with no
    pixel grid. OYSTER adds the Fourier transform of each component: A is a
    linear limb-darkened elliptical disk (the baselines stretched along its
    axes), the spot a uniform 1 mas disk of flux (contrast - 1)(1 - u) times
    its area, over A's, and B a uniform disk. At u = 0.469 the 0.08 mas pixel
    grid reads V² χ²/N 1.45 where this reads 1.33."""
    from scipy.special import j1, jv

    def uniform(q):
        x = np.maximum(2 * np.pi * q, 1e-12)
        return 2 * j1(x) / x

    uu, vv = np.asarray(u) * MAS, np.asarray(v) * MAS  # cycles per mas
    pa = np.radians(120.0)
    along, across = uu * np.sin(pa) + vv * np.cos(pa), uu * np.cos(pa) - vv * np.sin(pa)
    x = np.maximum(2 * np.pi * np.hypot(2.75 * along, 0.7 * 2.75 * across), 1e-12)
    # Hanbury Brown et al. (1974): the linear limb-darkened disk
    va = ((1 - u_ld) * j1(x) / x + u_ld * np.sqrt(np.pi / 2) * jv(1.5, x) / x**1.5) / ((1 - u_ld) / 2 + u_ld / 3)
    contrast = planck_ratio(1.5 * 7000.0, 7000.0) if spot_contrast is None else spot_contrast
    star_area = np.pi * 2.75 * 0.7 * 2.75 * (1 - u_ld / 3)
    flux = (contrast - 1.0) * (1.0 - u_ld) * np.pi * 0.25 / star_area
    sx, sy = offset(150.0, 2.5)
    bx, by = offset(90.0, 10.0)
    rho = np.hypot(uu, vv)
    vs = flux * uniform(0.5 * rho) * np.exp(-2j * np.pi * (uu * sx + vv * sy))
    vb = ratio * uniform(0.25 * rho) * np.exp(-2j * np.pi * (uu * bx + vv * by))
    return (va + vs + vb) / (1 + ratio)


def read_t3(path):
    from astropy.io import fits

    with fits.open(path) as h:
        d = h["OI_T3"].data
        ok = ~d["FLAG"].ravel() if "FLAG" in d.columns.names else np.ones(len(d), bool)
        u1, v1 = d["U1COORD"].ravel() / WAVEL, d["V1COORD"].ravel() / WAVEL
        u2, v2 = d["U2COORD"].ravel() / WAVEL, d["V2COORD"].ravel() / WAVEL
        return (
            u1[ok],
            v1[ok],
            u2[ok],
            v2[ok],
            np.radians(d["T3PHI"].ravel()[ok]),
            np.radians(d["T3PHIERR"].ravel()[ok]),
        )


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


def truth_vis_image(ratio, u_ld=U_LD, spot_contrast=None):
    """The image the data's visibilities are normalised against: V(0) =
    (A.sum() + B/A) / (1 + B/A) > 1, the spot's excess flux outside the
    normalisation."""
    star, companion = truth_parts(u_ld, spot_contrast=spot_contrast)
    return (star + ratio * companion) / (1 + ratio)


def truth_image(ratio, u_ld=U_LD, spot_contrast=None):
    """The scoring image: ``truth_vis_image`` normalised to unit sum."""
    img = truth_vis_image(ratio, u_ld, spot_contrast)
    return img / img.sum()


def v2_chi2(image, u, v, v2, err):
    return float(np.mean(((np.abs(visibilities(image, u, v)) ** 2 - v2) / err) ** 2))


def fit_nuisances(v2_data, t3, start=None):
    """Nelder-Mead fit of (u_ld, spot contrast, B/A) to the V² and closure
    phases (sum of the two χ²/N), the visibilities normalised as in
    ``truth_vis_image``. Used once to set ``FITTED``; scoring only."""
    from scipy.optimize import minimize

    u, v, v2, err = v2_data
    x0 = start or (FITTED["u_ld"], FITTED["spot_contrast"], np.log(FITTED["ratio"]))

    def objective(p):
        u_ld, contrast, log_r = p
        if not (0.0 <= u_ld <= 1.0 and contrast > 1.0):
            return 1e6
        img = truth_vis_image(np.exp(log_r), u_ld, contrast)
        return v2_chi2(img, u, v, v2, err) + closure_chi2(img, t3)[0]

    best = None
    for guess in (x0, (0.45, 5.6, np.log(0.11)), (0.8, 3.0, np.log(0.1))):
        r = minimize(
            objective,
            guess,
            method="Nelder-Mead",
            options=dict(xatol=1e-4, fatol=1e-4, maxiter=1000, adaptive=True),
        )
        if best is None or r.fun < best.fun:
            best = r
    return {
        "u_ld": float(best.x[0]),
        "spot_contrast": float(best.x[1]),
        "ratio": float(np.exp(best.x[2])),
    }


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
    parser.add_argument(
        "ensemble",
        help="<label>_ensemble.npz from combine_ensemble.py, or a member npz",
    )
    parser.add_argument("--data", default="~/data/imaging_contests")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--refit",
        action="store_true",
        help="refit FITTED's nuisances to the data (about 10 s)",
    )
    args = parser.parse_args()
    data_path = pathlib.Path(args.data).expanduser() / "2004" / "2004-data2.fits"
    d = np.load(args.ensemble)
    entry = np.asarray(d["mean"] if "mean" in d else d["ref_image"], float)
    fov = float(d["fov"] if "fov" in d else d["ref_fov"])
    u, v, v2, err = read_v2(data_path)
    t3 = read_t3(data_path)
    lines = []
    if args.refit:
        lines.append(f"fit_nuisances: {fit_nuisances((u, v, v2, err), t3)}")
    for u_ld in (0.3, U_LD, 0.7):
        star, companion = truth_parts(u_ld)
        ratio, chi2 = fit_ratio(star, companion, u, v, v2, err)
        scores, ref = score(entry, fov, ratio, u_ld)
        cp, cp_inv = closure_chi2(ref, t3)
        lines.append(
            f"u_ld={u_ld}: B/A={ratio:.3f} truth V² χ²/N={chi2:.2f}, closure χ²/N {cp:.2f} "
            f"(inverted truth {cp_inv:.2f}), ref peak={ref.max():.3e} (published 3.07e-3)"
        )
        for name, (sp, s, peak, shift) in scores.items():
            lines.append(f"  {name:9s} σ/peak={sp:.3f} σ={s:.2e} shift={shift}")
    ref = truth_image(FITTED["ratio"], FITTED["u_ld"], FITTED["spot_contrast"])
    box = np.zeros_like(ref, bool)
    rows, cols = np.nonzero(ref > 0)
    box[rows.min() - 10 : rows.max() + 11, cols.min() - 10 : cols.max() + 11] = True
    moved, shift = aligned(resample(entry, fov, NPIX + 2 * MAX_SHIFT), ref)
    vis_img = truth_vis_image(FITTED["ratio"], FITTED["u_ld"], FITTED["spot_contrast"])
    cp, cp_inv = closure_chi2(vis_img, t3)
    an = np.abs(model_vis(u, v, FITTED["ratio"], FITTED["u_ld"], FITTED["spot_contrast"])) ** 2
    lines.append(f"  analytic visibilities (no pixel grid): V² χ²/N={np.mean(((an - v2) / err) ** 2):.2f}")
    lines.append(
        f"fitted truth (u_ld={FITTED['u_ld']}, spot contrast {FITTED['spot_contrast']}, B/A={FITTED['ratio']}): "
        f"V² χ²/N={v2_chi2(vis_img, u, v, v2, err):.2f}, closure χ²/N {cp:.2f} (inverted {cp_inv:.2f}), "
        f"V(0)={vis_img.sum():.3f}, σ/peak={sigma_over_peak(moved, ref, box)[0]:.3f} shift={shift}"
    )
    lines.append("published data2 σ/peak: " + ", ".join(f"{k} {v}" for k, v in PUBLISHED.items()))
    text = "\n".join(lines)
    print(text)
    if args.out:
        out = pathlib.Path(args.out).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        (out / "2004_data2_score.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
