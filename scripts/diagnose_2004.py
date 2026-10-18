"""Where does the rebuilt 2004 data2 truth's V² excess sit, and which untested
ingredient (bandwidth smearing, limb-darkening law, spot profile) closes it?

    python scripts/diagnose_2004.py [--data DIR] [--out DIR]

Each variant fits only the unpublished nuisances (limb darkening, spot
contrast, B/A) to the data's V² and closure phases, with the published
geometry fixed; NumPy and SciPy only. Scoring only, never imaging. The data
are one 550 nm channel, 20 nm wide (``OI_WAVELENGTH``), so "per channel"
means the band is sub-sampled for smearing, not that there are channels.
"""

import argparse
import pathlib

import numpy as np

import score_2004 as s

PIX = 0.05  # finer than score_2004's 0.08: V² moves by 1 % between the two
NPIX = 481  # 24 mas
BAND = 20e-9  # OI_WAVELENGTH EFF_BAND
SPOT_PA, SPOT_SEP, SPOT_DIAM = 150.0, 2.5, 1.0


def build(u_ld, u2, contrast, spot, pix=PIX, npix=NPIX):
    """(A with its spot, B): the unspotted star has unit sum, the spot's
    excess is left out of the normalisation (see score_2004). ``spot``:
    ``clip-ld`` (score_2004's old spot, superseded by ``add-planck``: contrast x the star's profile, only on the
    disk), ``clip-flat`` (uniform brightness ``contrast``, only on the disk),
    ``free-flat`` (the whole 1 mas disk at ``contrast``, also beyond the
    limb), ``free-add`` (the disk added to the star), ``add-planck`` (the Planck excess over the limb brightness, added),
    ``free-planck`` (``free-flat``
    with the contrast tied to Planck(1.5 Teff)/Planck(Teff) times the limb
    brightness 1 - u1 - u2), ``gauss-ld`` (FWHM 1 mas Gaussian brightening of the star's own
    profile, on the disk) and ``gauss-free`` (the Gaussian added, also beyond
    the limb). ``u2`` is the quadratic term (0 for a linear law)."""
    x, y = s.grid(npix, pix)
    pa = np.radians(120.0)
    along, across = x * np.sin(pa) + y * np.cos(pa), x * np.cos(pa) - y * np.sin(pa)
    a, b = 5.5 / 2, 0.7 * 5.5 / 2
    r2 = (along / a) ** 2 + (across / b) ** 2
    inside = r2 < 1.0
    one_mu = 1.0 - np.sqrt(np.clip(1.0 - r2, 0.0, None))
    star = np.where(inside, 1.0 - u_ld * one_mu - u2 * one_mu**2, 0.0)
    norm = star.sum()
    sx, sy = s.offset(SPOT_PA, SPOT_SEP)
    d2 = (x - sx) ** 2 + (y - sy) ** 2
    if spot == "clip-ld":
        img = np.where(inside & (d2 < (SPOT_DIAM / 2) ** 2), star * contrast, star)
    elif spot == "clip-flat":
        img = np.where(inside & (d2 < (SPOT_DIAM / 2) ** 2), contrast, star)
    elif spot in ("free-flat", "free-add", "free-planck", "add-planck"):
        disk = d2 < (SPOT_DIAM / 2) ** 2
        if spot == "free-planck":
            # contrast tied to the physics: a uniform disk at 1.5 Teff, the Planck
            # ratio at 550 nm times the star's limb brightness, 1 - u1 - u2
            contrast = s.planck_ratio(1.5 * 7000.0, 7000.0) * (1.0 - u_ld - u2)
        if spot == "add-planck":
            # the Planck excess over the limb brightness, added as a uniform disk
            contrast = (s.planck_ratio(1.5 * 7000.0, 7000.0) - 1.0) * (1.0 - u_ld - u2)
        img = star + contrast * disk if spot in ("free-add", "add-planck") else np.where(disk, contrast, star)
    else:
        g = (contrast - 1.0) * np.exp(-4 * np.log(2) * d2 / SPOT_DIAM**2)
        img = star * (1.0 + g) if spot == "gauss-ld" else star + g
    bx, by = s.offset(90.0, 10.0)
    comp = (((x - bx) ** 2 + (y - by) ** 2) < 0.25**2).astype(float)
    return img / norm, comp / comp.sum(), x, y


def vis(img, x, y, u, v, band=0.0, nsub=7):
    """Complex V at (u, v) given in wavelengths at 550 nm; with ``band`` > 0
    the *power* is averaged over a top-hat of that width (returns |V|² via
    ``power``)."""
    keep = img != 0
    xs, ys, w = x[keep], y[keep], img[keep]
    if band <= 0:
        return np.exp(-2j * np.pi * (np.outer(u, xs) + np.outer(v, ys)) * s.MAS) @ w
    lams = s.WAVEL + band * (np.arange(nsub) + 0.5) / nsub - band / 2
    return [
        np.exp(-2j * np.pi * (np.outer(u * s.WAVEL / lam, xs) + np.outer(v * s.WAVEL / lam, ys)) * s.MAS) @ w
        for lam in lams
    ]


def model_obs(p, spot, band, data):
    """(V², CP, T3AMP) predictions; ``p`` = (u1, u2, contrast, log ratio)."""
    u1, u2, c, lr = p
    st, co, x, y = build(u1, u2, c, spot)
    r = np.exp(lr)
    img = (st + r * co) / (1 + r)
    u, v, _, _, (u1c, v1c, u2c, v2c, _, _), _ = data

    def vv(uu, vv_):
        return vis(img, x, y, uu, vv_, band)

    if band <= 0:
        v2 = np.abs(vv(u, v)) ** 2
        bis = vv(u1c, v1c) * vv(u2c, v2c) * np.conj(vv(u1c + u2c, v1c + v2c))
        return v2, np.angle(bis), np.abs(bis)
    sub = [np.abs(z) ** 2 for z in vv(u, v)]
    b1, b2, b3 = vv(u1c, v1c), vv(u2c, v2c), vv(u1c + u2c, v1c + v2c)
    bis = [a * b * np.conj(c_) for a, b, c_ in zip(b1, b2, b3)]
    # closure phase and amplitude of the band-averaged bispectrum
    m = np.mean(bis, axis=0)
    return np.mean(sub, axis=0), np.angle(m), np.abs(m)


def chis(p, spot, band, data):
    _, _, v2, err, (_, _, _, _, phi, perr), (amp, aerr) = data
    m_v2, m_phi, m_amp = model_obs(p, spot, band, data)
    c_v2 = np.mean(((m_v2 - v2) / err) ** 2)
    c_cp = np.mean((np.angle(np.exp(1j * (m_phi - phi))) / perr) ** 2)
    c_amp = np.mean(((m_amp - amp) / aerr) ** 2)
    return c_v2, c_cp, c_amp


def fit(spot, band, data, free_u2=False, free_scale=False, start=None):
    from scipy.optimize import minimize

    F = s.FITTED
    x0 = start or [F["u_ld"], 0.0, F["spot_contrast"], np.log(F["ratio"])]

    def obj(q):
        p = [q[0], q[1] if free_u2 else 0.0, q[2], q[3]]
        if not (-0.5 <= p[0] <= 1.5 and -1.0 <= p[1] <= 1.0 and p[2] > 0.5):
            return 1e6
        c = chis(p, spot, band, data)
        return c[0] + c[1]

    best = min(
        (
            minimize(
                obj,
                g,
                method="Nelder-Mead",
                options=dict(xatol=1e-4, fatol=1e-4, maxiter=800, adaptive=True),
            )
            for g in (x0, [0.45, 0.1, 5.6, np.log(0.11)])
        ),
        key=lambda r: r.fun,
    )
    q = best.x
    p = [q[0], q[1] if free_u2 else 0.0, q[2], q[3]]
    return p, chis(p, spot, band, data)


def load(path):
    from astropy.io import fits

    u, v, v2, err = s.read_v2(path)
    t3 = s.read_t3(path)
    with fits.open(path) as h:
        d = h["OI_T3"].data
        ok = ~d["FLAG"].ravel()
        amp, aerr = d["T3AMP"].ravel()[ok], d["T3AMPERR"].ravel()[ok]
    return u, v, v2, err, t3, (amp, aerr)


def residual_tables(p, data, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits

    u, v, v2, err, *_ = data
    m = model_obs(p, "clip-ld", 0.0, data)[0]  # the old clipped-spot truth, not the current scorer's
    r = (v2 - m) / err
    b = np.hypot(u, v) * s.WAVEL
    with fits.open(DATA_PATH) as h:
        sta = h["OI_VIS2"].data["STA_INDEX"]
        t = h["OI_VIS2"].data["TIME"]
    ok = ~fits.open(DATA_PATH)["OI_VIS2"].data["FLAG"].ravel()
    sta, t = sta[ok], t[ok]
    lines = ["baseline range (m): N, χ²/N, mean residual (σ), mean V²"]
    for lo, hi in [(0, 15), (15, 25), (25, 40), (40, 60)]:
        k = (b >= lo) & (b < hi)
        lines.append(f"  {lo}-{hi}: {k.sum()}, {np.mean(r[k] ** 2):.2f}, {r[k].mean():+.2f}, {v2[k].mean():.3f}")
    lines.append("per baseline (stations): N, mean B (m), χ²/N, mean residual (σ)")
    keys = sorted({tuple(x) for x in sta}, key=lambda k: b[(sta == k).all(1)].mean())
    for k in keys:
        i = (sta == k).all(1)
        lines.append(f"  {k}: {i.sum()}, {b[i].mean():.1f}, {np.mean(r[i] ** 2):.2f}, {r[i].mean():+.2f}")
    lines.append("per time block: N, χ²/N, mean residual (σ)")
    for tt in np.unique(t):
        i = t == tt
        lines.append(f"  t={tt:.0f}: {i.sum()}, {np.mean(r[i] ** 2):.2f}, {r[i].mean():+.2f}")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].scatter(b, r, s=8)
    ax[0].axhline(0, c="k", lw=0.5)
    ax[0].set(
        xlabel="baseline (m)",
        ylabel="(data - model)/σ",
        title="V² residuals vs baseline",
    )
    ax[1].scatter(b, v2, s=8, label="data")
    ax[1].scatter(b, m, s=8, label="model")
    ax[1].set(yscale="log", xlabel="baseline (m)", ylabel="V²")
    ax[1].legend()
    ax[2].scatter(t, r, s=8, c=b)
    ax[2].set(xlabel="TIME", title="residual vs time (colour = baseline)")
    fig.tight_layout()
    if out:
        fig.savefig(out / "2004_v2_residuals.png", dpi=110)
    return "\n".join(lines)


DATA_PATH = None


def per_baseline_scale(p, spot, data):
    """V² χ²/N after fitting a free multiplicative scale to each of the 15
    station pairs (diagnostic only: it shows which baselines the model
    gets wrong; the data are a simulation, so it is not a calibration)."""
    from astropy.io import fits

    u, v, v2, err, *_ = data
    m = model_obs(p, spot, 0.0, data)[0]
    with fits.open(DATA_PATH) as h:
        d = h["OI_VIS2"].data
        sta = d["STA_INDEX"][~d["FLAG"].ravel()]
    total, scales = 0.0, []
    for k in {tuple(x) for x in sta}:
        i = (sta == k).all(1)
        w = 1 / err[i] ** 2
        sc = np.sum(w * v2[i] * m[i]) / np.sum(w * m[i] ** 2)
        total += np.sum(w * (v2[i] - sc * m[i]) ** 2)
        scales.append(sc)
    return total / len(v2), sorted(scales)


def main():
    global DATA_PATH
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="~/data/imaging_contests")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--skip-slow",
        action="store_true",
        help="skip the bandwidth-smeared fits (about 30 min each)",
    )
    args = ap.parse_args()
    DATA_PATH = pathlib.Path(args.data).expanduser() / "2004" / "2004-data2.fits"
    out = pathlib.Path(args.out).expanduser() if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    data = load(DATA_PATH)
    # the first fit of score_2004 (spot clipped to the disk), for the residual tables
    base = [0.462, 0.0, 5.575, np.log(0.1129)]
    rows = []
    lines = []

    def add(name, spot, band, **kw):
        p, c = fit(spot, band, data, **kw)
        rows.append((name, *c, p))
        lines.append(
            f"{name:38s} V² {c[0]:.2f}  CP {c[1]:.2f}  T3AMP {c[2]:.2f}  u={p[0]:.3f} u2={p[1]:.3f} contrast={p[2]:.2f} B/A={np.exp(p[3]):.3f}"
        )

    lines.append(residual_tables(base, data, out))
    add("old scorer truth (linear LD, clipped spot; the scorer now uses add-planck)", "clip-ld", 0.0)
    if not args.skip_slow:
        add("(a) bandwidth 20 nm", "clip-ld", BAND)
    add("(b) quadratic LD", "clip-ld", 0.0, free_u2=True)
    for sp in (
        "clip-flat",
        "free-flat",
        "free-add",
        "free-planck",
        "add-planck",
        "gauss-ld",
        "gauss-free",
    ):
        add(f"(c) spot {sp}", sp, 0.0)
    add("(b)+(c: free-flat)", "free-flat", 0.0, free_u2=True)
    add("(b)+(c: add-planck)", "add-planck", 0.0, free_u2=True)
    if not args.skip_slow:
        add("(a)+(c: free-flat)", "free-flat", BAND)
    for name, spot, p in (
        ("baseline", "clip-ld", base),
        ("adopted spot", "add-planck", fit("add-planck", 0.0, data)[0]),
    ):
        lines.append(
            f"(d) {name}, free V² scale per baseline (diagnostic): V² {per_baseline_scale(p, spot, data)[0]:.2f}"
        )
    text = "\n".join(lines)
    print(text)
    if out:
        (out / "2004_diagnose.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
