"""Independent checks for radio-interferometric (ALMA) visibilities, NumPy only.

Nothing here imports virgil. Conventions (docs/alma_conventions.md):

    V(u, v) = sum I(x, y) exp(-2 pi i (u x + v y)),

x towards East and y towards North, u and v in wavelengths. Images are
returned as ``image[row, col]`` with row 0 North and column 0 East (as
``crosscheck.sky`` and eht-imaging); the pixel at offset zero is row and
column ``npix // 2`` counted from the far (West or South) side, i.e. the
East offset of column c is ``(npix//2 - c) * pixel`` and the North offset of
row r is ``(npix//2 - r) * pixel``. The dirty image is the natural-weighted
inverse transform (Thompson, Moran and Swenson, *Interferometry and
Synthesis in Radio Astronomy*, ch. 5).
"""

import numpy as np

MAS = np.pi / 180.0 / 3600.0 / 1000.0  # radians per milliarcsecond


def pixel_offsets(npix, pixel_scale_mas):
    """(East, North) offsets in radians of the columns and rows."""
    k = (npix // 2 - np.arange(npix)) * pixel_scale_mas * MAS
    return k, k


def dft_dirty_image(u, v, vis, weight, npix, pixel_scale_mas, chunk=4096):
    """Float64 natural-weighted dirty image, a direct sum over points.

    I(x, y) = sum_k w_k Re(V_k exp(+2 pi i (u_k x + v_k y))) / sum_k w_k.
    The exponential separates into a column and a row factor, so the sum is
    a pair of matrix products per chunk of points.
    """
    u, v, w = (np.asarray(a, float) for a in (u, v, weight))
    vis = np.asarray(vis, complex)
    x, y = pixel_offsets(npix, pixel_scale_mas)
    out = np.zeros((npix, npix))
    for i in range(0, len(u), chunk):
        s = slice(i, i + chunk)
        wv = w[s] * vis[s]
        fx = np.exp(2j * np.pi * np.outer(u[s], x))  # (M, ncol)
        fy = np.exp(2j * np.pi * np.outer(v[s], y))  # (M, nrow)
        out += ((fy * wv[:, None]).T @ fx).real
    return out / w.sum()


def gridded_dirty_image(u, v, vis, weight, npix, pixel_scale_mas):
    """Nearest-cell gridding of V and its Hermitian partner, then ``numpy.fft``.

    A diagnostic for data too big for the DFT: cell size 1 / (npix * pixel),
    so the image is periodic and is tapered by sinc(x du) from the box
    gridding kernel. Points beyond the grid are dropped.
    """
    u, v, w = (np.asarray(a, float) for a in (u, v, weight))
    vis = np.asarray(vis, complex)
    du = 1.0 / (npix * pixel_scale_mas * MAS)
    iu, iv = np.rint(u / du).astype(int), np.rint(v / du).astype(int)
    half = npix // 2
    keep = (np.abs(iu) < half) & (np.abs(iv) < half)
    iu, iv, wv = iu[keep], iv[keep], (w * vis)[keep]
    grid = np.zeros((npix, npix), complex)  # [iv mod n, iu mod n]
    for a, b, val in ((iu, iv, wv), (-iu, -iv, np.conj(wv))):
        np.add.at(grid, (b % npix, a % npix), val)
    # image(x, y) = sum G exp(+2 pi i (u x + v y)); numpy's ifft2 has that sign
    # and a 1/N^2 factor; the pixel at offset (i, j) * pixel is index (j, i) mod n
    img = (np.fft.ifft2(grid) * npix * npix).real / (2.0 * w.sum())
    # FFT index j holds offset j (mod n) pixels; row r is North offset n//2 - r
    # and column c East offset n//2 - c
    idx = (npix // 2 - np.arange(npix)) % npix
    return img[np.ix_(idx, idx)]


def peak_offset_mas(image, pixel_scale_mas, box=None):
    """(East, North) offset in mas of the largest pixel, optionally within a
    (row0, row1, col0, col1) box."""
    npix = image.shape[0]
    r0, r1, c0, c1 = box or (0, npix, 0, npix)
    sub = image[r0:r1, c0:c1]
    r, c = np.unravel_index(np.argmax(sub), sub.shape)
    r, c = r + r0, c + c0
    return (npix // 2 - c) * pixel_scale_mas, (npix // 2 - r) * pixel_scale_mas


def successive_difference_sigma(npz):
    """Model-free check of the weights from time-adjacent samples.

    ``npz`` holds one ddid (a dict or ``numpy.load`` result) with the keys
    ``re, im, weight, ant1, ant2, chan, scan, time, t_int_s``. Samples are
    sorted by (scan, ant1, ant2, chan, time); a pair (i, i+1) is two
    consecutive integrations on the same (scan, ant1, ant2, chan) with
    0 < dt <= 1.5 t_int, and pairs do not overlap (a run of n consecutive
    adjacent samples gives n // 2 pairs). With weight = 1/sigma^2 for each of
    Re and Im, E|V_i - V_j|^2 = 2 (sigma_i^2 + sigma_j^2), so

        s^2 = mean(|V_i - V_j|^2 / (2 (1/w_i + 1/w_j)))

    is 1 when the weights are right (and larger if the sky changes between
    integrations). Returns ``dict(s2, s2_err, n_pairs, s)``; the error is the
    standard error of the mean over pairs.
    """
    g = {k: np.asarray(npz[k]) for k in ("re", "im", "weight", "ant1", "ant2", "chan", "scan", "time")}
    t_int = float(npz["t_int_s"])
    order = np.lexsort((g["time"], g["chan"], g["ant2"], g["ant1"], g["scan"]))
    g = {k: a[order] for k, a in g.items()}
    same = np.ones(len(g["time"]) - 1, bool)
    for k in ("scan", "ant1", "ant2", "chan"):
        same &= g[k][1:] == g[k][:-1]
    dt = np.diff(g["time"])
    adjacent = same & (dt > 0) & (dt <= 1.5 * t_int)
    # position within each run of True: take positions 0, 2, 4, ...
    idx = np.arange(len(adjacent))
    starts = np.where(adjacent & ~np.r_[False, adjacent[:-1]], idx, 0)
    run_start = np.maximum.accumulate(starts)
    first = adjacent & (((idx - run_start) % 2) == 0)
    i = np.nonzero(first)[0]
    if len(i) == 0:
        return {"s2": np.nan, "s2_err": np.nan, "n_pairs": 0, "s": np.nan}
    dv = (g["re"][i] - g["re"][i + 1]).astype(float) ** 2 + (g["im"][i] - g["im"][i + 1]).astype(float) ** 2
    var = 2.0 * (1.0 / g["weight"][i].astype(float) + 1.0 / g["weight"][i + 1].astype(float))
    r = dv / var
    s2, err = float(r.mean()), float(r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else float("nan")
    return {"s2": s2, "s2_err": err, "n_pairs": int(len(r)), "s": float(np.sqrt(s2))}
