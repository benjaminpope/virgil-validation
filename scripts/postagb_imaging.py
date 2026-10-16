"""virgil against published VLTI/PIONIER post-AGB disc images (De Prins et al. 2026).

    python scripts/postagb_imaging.py {inspect,parametric,image,compare,replot,table} OUTDIR
        [--target IW_Car] [--smoke] [--files N]

Stages (each writes into OUTDIR and the later ones read the earlier ones):

* ``inspect``    data summary, uv coverage, V^2 and closure phases; checks the files
                 against the paper's observation log (Zenodo table_d1.dat).
* ``parametric`` refit the PMOIRED-style geometry (UD primary + point secondary +
                 power-law rim with an m=1 azimuthal modulation + resolved
                 background) with virgil components, and tabulate it beside the paper.
* ``image``      SPARCO-like image reconstruction: UD primary + point secondary +
                 virgil.models.Image environment, power-law spectra, MaxEntropy
                 regulariser, the weight chosen on an L-curve, starting image from
                 ``starting_image``.
* ``compare``    our image and the published ORGANIC image on one grid, convolved
                 to the PIONIER beam: normalised cross-correlation, rim radius,
                 position angle and azimuthal asymmetry, and the reduced chi2 of the
                 published image on our data.
* ``replot``     redraw every figure from the saved results (no fitting).
* ``table``      markdown parameter table: De Prins et al. PMOIRED fit | virgil | difference in sigma.

``--smoke`` shrinks everything (three files, a coarse grid, two L-curve weights,
a handful of optimiser steps) so that it runs on a laptop in a few minutes; its
outputs only prove the pipeline runs. A full IW Car run (23 files, 128x128 pixels at 0.35 mas) takes
minutes on two CPU threads; larger targets should go to OzSTAR. Data: ``postagb_data.py fetch``;
published images: ``postagb_literature.py``. Add targets to TARGETS.
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from astropy.io import fits  # noqa: E402
from scipy import ndimage  # noqa: E402
from virgil import OIData, read_oifits
from virgil.oidata import closure_phases  # noqa: E402
from virgil.fitting import fit  # noqa: E402
from virgil.imaging import (  # noqa: E402
    MaxEntropy, beam, convolve_beam, image_priors, l_curve, nyquist_pixel_scale,
    starting_image, whitened_residuals,
)
from virgil.models import (  # noqa: E402
    Image, ModulatedGaussianRim, PointSource, Resolved, System, UniformDisk,
)
from virgil.spectra import PowerLaw  # noqa: E402

DATA = Path(os.environ.get("POSTAGB_DATA", "~/data/postagb_pionier")).expanduser()  # POSTAGB_DATA: the OzSTAR copy
ZENODO = DATA / "literature/obs_logs_and_images"
L0 = 1.65e-6  # SPARCO reference wavelength (m), as in the published FITS headers

# Per-target configuration. ``paper`` holds numbers transcribed from De Prins et
# al. (2026, arXiv:2605.13445v2) Table 2 (image-derived and SPARCO values); the
# SPARCO values are also in the Zenodo FITS headers, which compare/inspect read.
TARGETS = {
    "IW_Car": dict(
        name="IW Car",
        files=DATA / "IW_Car/INSPIRING/*.fits",  # the 23 files of the paper's observation log
        log_name="IW Car",
        fits="IW_Car_image.fits",
        fov_mas=44.8,
        pixel_mas=0.35,
        sec0=(1.12, -1.90),  # secondary from the paper's Gaussian fit to the image (Table 2)
        half_mas=18.0, mask_mas=20.0, half_cmp_mas=15.0, ud0=0.5,
        inner_rim_max_mas=5.0,  # inner arcs lie inside this radius, the outer arc beyond (paper Sect. 5.3.7)
        regions=dict(inner=(0.0, 5.0), outer=(5.0, 20.0)),
        paper=dict(
            f_prim_pct=63.6, ud_prim_mas=0.50, f_sec_pct=1.99, sec_dra=1.12, sec_ddec=-1.90,
            image_index=0.89, rim_diam_mas=6.9, rim_inc_deg=48.0, rim_pa_deg=144.0,
            rim_dra=0.15, rim_ddec=-0.1, r_hl_mas=11.7,
            outer_inc_deg=57.0, outer_pa_deg=176.0, sphere_inc_deg=41.7, sphere_pa_deg=161.0,
        ),
    ),
}


# The other seven targets of the paper share one recipe; everything geometric (grid, SPARCO
# secondary, plot field of view) comes from the Zenodo FITS header at run time (see ``setup``).
# ``paper`` holds only the numbers the paper reports for that target (Table 2 / PMOIRED tables).
OTHERS = {
    "HR4049": "HR 4049", "HD108015": "HD 108015", "AI_Sco": "AI Sco", "EN_TrA": "EN TrA",
    "HD95767": "HD 95767", "IRAS15469-5311": "IRAS 15469-5311", "PS_Gem": "PS Gem",
}
for _k, _name in OTHERS.items():
    TARGETS[_k] = dict(name=_name, log_name=_name, fits=f"{_k}_image.fits", paper={})

# PS Gem's primary carries 87 % of the flux, so the default weights (1e4 to 1e-2) leave the first fit unconverged
# and the sweep goes to NaN; shift the sweep down two decades.
TARGETS["PS_Gem"]["log_weights"] = (2, -4)
TARGETS["HR4049"]["half_mas"] = 25.0  # the extent of the paper's Fig. 4 panel

for _k, _v in TARGETS.items():
    _v["key"] = _k
    _v.setdefault("files", None)  # None: the files of the paper's observation log, from the archive


def path(out, cfg, name):
    """Output file for this target: <outdir>/<target>_<name>."""
    return out / f"{cfg['key']}_{name}"


# ----------------------------------------------------------------------------
# data


def cfg_of(args):
    cfg = TARGETS[args.target]
    if "fov_mas" not in cfg:
        setup(cfg)
    return cfg


def setup(cfg):
    """Fill grid, SPARCO secondary and plot extents from the published FITS header."""
    hdr, im, _ = zenodo_header(cfg)
    n, scale = im.shape[0], abs(hdr["CDELT1"])
    yy, xx = np.mgrid[:n, :n]
    rad = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2) * scale
    o = np.argsort(rad.ravel())
    r90 = float(rad.ravel()[o][np.searchsorted(np.cumsum(im.ravel()[o]) / im.sum(), 0.9)])
    half = cfg.get("half_mas") or float(np.clip(1.5 * r90, 8.0, 0.95 * n * scale / 2))  # per-target override: the paper's figure extent
    has_sec = hdr["SFLU2"] > 0
    cfg.update(
        fov_mas=n * scale, pixel_mas=scale, sec0=(hdr["SDEX2"], hdr["SDEY2"]) if has_sec else None,
        ud0=max(hdr["SUD1"], 0.3), half_mas=half, mask_mas=half, half_cmp_mas=0.85 * half,
        regions=dict(disc=(0.0, half)), inner_rim_max_mas=None, paper_header=dict(
            ud=hdr["SUD1"], f_prim_pct=100 * hdr["SFLU1"], f_sec_pct=100 * hdr["SFLU2"],
            sec=(hdr["SDEX2"], hdr["SDEY2"]), image_index=hdr["SIND0"], star_index=hdr["SIND1"]),
    )


def load_files(cfg, n=None):
    # the INSPIRING release holds exactly the exposures of the paper's observation log (counts checked per target)
    pattern = cfg["files"] if cfg["files"] is not None else DATA / cfg["key"] / "INSPIRING/*.fits"
    files = sorted(glob.glob(str(pattern)))
    if not files:
        raise SystemExit(f"no files for {cfg['key']}; run scripts/postagb_data.py fetch")
    return files[:n] if n else files


def load(cfg, n=None):
    files = load_files(cfg, n)
    return OIData(read_oifits(files)), files


def paper_log_mjds(cfg):
    """MJDs the paper used for this target, from the Zenodo observation log."""
    out = []
    for line in (ZENODO / "table_d1.dat").read_text().splitlines():
        if line.startswith(cfg["log_name"]):
            out.append(float(line[27:35]))
    return out


def observables(model, data):
    """Model V^2 (at the unflagged V^2 samples) and closure phases (degrees)."""
    cv = model.model(data.u, data.v, data.wavel)
    t3 = closure_phases(cv, data.i_cps1, data.i_cps2, data.i_cps3)
    v2 = np.asarray(jnp.abs(cv) ** 2)
    vi = data.vis_index
    return (v2 if vi is None or np.ndim(vi) == 0 else v2[np.asarray(vi)]), np.degrees(np.asarray(t3))


def chi2r(model, data):
    r = np.asarray(whitened_residuals(model, data))
    return float(np.sum(r**2) / r.size), int(r.size)


def zenodo_header(cfg):
    with fits.open(ZENODO / "fits" / cfg["fits"]) as h:
        return dict(h["IMAGE"].header), np.asarray(h["IMAGE"].data, float), np.asarray(h["NOISE"].data, float)


# ----------------------------------------------------------------------------
# inspect


def inspect_stage(args, out):
    cfg = cfg_of(args)
    data, files = load(cfg, args.files)
    mjd = [float(fits.getheader(f)["MJD-OBS"]) for f in files]
    log = paper_log_mjds(cfg)
    matched = sum(any(abs(m - x) < 0.01 for x in log) for m in mjd)
    b = beam(data)
    info = dict(
        target=cfg["name"], n_files=len(files), n_vis2=int(data.vis.shape[0]), n_t3=int(data.phi.shape[0]),
        wavel_um=[float(jnp.min(data.wavel)) * 1e6, float(jnp.max(data.wavel)) * 1e6],
        beam_major_mas=b.major_mas, beam_minor_mas=b.minor_mas, beam_pa_deg=b.pa_deg,
        paper_log_rows=len(log), files_in_paper_log=matched,
        mjd_range=[min(mjd), max(mjd)],
    )
    print(info)
    json.dump(info, open(path(out, cfg, "inspect.json"), "w"), indent=1)
    plot_inspect(data, info, out, cfg)


def plot_inspect(data, info, out, cfg):
    vi = data.vis_index  # None when no sample is flagged
    u, v, wl = (np.asarray(x) for x in (data.u, data.v, data.wavel))
    vi = np.arange(u.size) if vi is None or np.ndim(vi) == 0 else np.asarray(vi)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    sc = ax[0].scatter(u[vi] / wl[vi] / 1e6, v[vi] / wl[vi] / 1e6, c=wl[vi] * 1e6, s=6, cmap="viridis")
    ax[0].scatter(-u[vi] / wl[vi] / 1e6, -v[vi] / wl[vi] / 1e6, c=wl[vi] * 1e6, s=6, cmap="viridis")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("u (Mλ)")
    ax[0].set_ylabel("v (Mλ)")
    ax[0].set_title(f"uv coverage, {info['n_files']} files")
    plt.colorbar(sc, ax=ax[0], label="wavelength (µm)")
    B = np.hypot(u, v)[vi] / wl[vi] / 1e6
    ax[1].errorbar(B, np.asarray(data.vis), np.asarray(data.d_vis), fmt=".", ms=2, alpha=0.4)
    ax[1].set_xlabel("B/λ (Mλ)")
    ax[1].set_ylabel("V²")
    ax[1].set_title("calibrated squared visibilities")
    phi, dphi = np.degrees(np.asarray(data.phi)), np.degrees(np.asarray(data.d_phi))
    ax[2].errorbar(np.arange(phi.size), phi, dphi, fmt=".", ms=2, alpha=0.4)
    ax[2].set_xlabel("closure-phase sample")
    ax[2].set_ylabel("T3 phase (deg)")
    ax[2].set_title(f"closure phases (rms {phi.std():.1f}°, median error {np.median(dphi):.1f}°)")
    fig.tight_layout()
    fig.savefig(path(out, cfg, "inspect.png"), dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# parametric


def parametric_start(cfg, inc=50.0, pa=150.0):
    """The PMOIRED geometry of De Prins et al. (2026, Sect. 4.2) in virgil components."""
    sec = cfg["sec0"]
    parts = dict(
        primary=UniformDisk(diam=cfg["ud0"], flux=PowerLaw(1.0, -3.165, L0)),
        rim=ModulatedGaussianRim(
            12.0, 4.0, inc, pa, az_amps=(0.3,), az_pas=(0.0,), flux=PowerLaw(0.4, 0.9, L0)),
        bg=Resolved(flux=PowerLaw(0.1, 0.0, L0)),
    )
    if sec is not None:  # targets without a SPARCO secondary in the paper have none here
        parts["secondary"] = PointSource(flux=PowerLaw(0.03, 0.0, L0), dra=sec[0], ddec=sec[1])
    return System(**parts)


PARAM_PRIORS = {
    "primary.diam": dist.Uniform(0.05, 1.5),
    "secondary.flux.ratio": dist.Uniform(0.0, 0.5),
    "secondary.dra": dist.Uniform(-5.0, 5.0),
    "secondary.ddec": dist.Uniform(-5.0, 5.0),
    "rim.diam": dist.Uniform(2.0, 60.0),
    "rim.fwhm": dist.Uniform(0.1, 30.0),
    "rim.inc": dist.Uniform(0.0, 89.0),
    "rim.pa": dist.Uniform(-90.0, 270.0),
    "rim.az_amps": dist.Uniform(0.0, 1.0),
    "rim.az_pas": dist.Uniform(-180.0, 540.0),
    "rim.dra": dist.Uniform(-5.0, 5.0),
    "rim.ddec": dist.Uniform(-5.0, 5.0),
    "rim.flux.ratio": dist.Uniform(0.0, 5.0),
    "rim.flux.index": dist.Uniform(-10.0, 10.0),
    "bg.flux.ratio": dist.Uniform(0.0, 5.0),
    "bg.flux.index": dist.Uniform(-10.0, 10.0),
}


def parametric_stage(args, out):
    cfg = cfg_of(args)
    data, files = load(cfg, args.files)
    t0 = time.time()
    priors = {k: v for k, v in PARAM_PRIORS.items() if cfg["sec0"] is not None or not k.startswith("secondary.")}
    # blind multi-start over the rim orientation (a single start can stick face-on: IRAS 15469-5311); keep the lowest chi2
    starts = [(50.0, 150.0)] if args.smoke else [(50.0, 150.0), (25.0, 30.0), (25.0, 120.0), (60.0, 30.0), (60.0, 100.0), (40.0, 170.0)]
    best, chi2_starts = None, []
    for inc, pa in starts:
        res = fit(parametric_start(cfg, inc, pa), priors, data, method="lbfgs", **({"max_steps": 25} if args.smoke else {}))
        c = chi2r(res.model, data)[0]
        chi2_starts.append(c)
        if best is None or c < best[0]:
            best = (c, res)
    result = best[1]
    model = result.model
    c2, n = chi2r(model, data)
    v = {k: np.asarray(x).tolist() for k, x in result.values.items()}
    has_sec = cfg["sec0"] is not None
    ratio = lambda name: float(np.asarray(getattr(model, name).flux(L0))) if name != "secondary" or has_sec else 0.0  # noqa: E731
    total = 1.0 + ratio("secondary") + ratio("rim") + ratio("bg")
    summary = dict(
        values=v, chi2_per_point=c2, n_points=n, chi2_starts=chi2_starts, seconds=time.time() - t0, smoke=args.smoke,
        f_prim_pct=100.0 / total, f_sec_pct=100 * ratio("secondary") / total,
        f_rim_pct=100 * ratio("rim") / total, f_bg_pct=100 * ratio("bg") / total,
        ud_prim_mas=float(model.primary.diam), rim_diam_mas=float(model.rim.diam),
        rim_fwhm_mas=float(model.rim.fwhm), rim_inc_deg=float(model.rim.inc),
        rim_pa_deg=float(model.rim.pa) % 180.0, rim_index=float(np.asarray(model.rim.flux.index)),
        az_amp=float(np.ravel(model.rim.az_amps)[0]), az_pa_deg=float(np.ravel(model.rim.az_pas)[0]) % 360.0,
        sec_dra=float(model.secondary.dra) if has_sec else None,
        sec_ddec=float(model.secondary.ddec) if has_sec else None,
    )
    print(json.dumps(summary, indent=1))
    json.dump(summary, open(path(out, cfg, "parametric.json"), "w"), indent=1)


# ----------------------------------------------------------------------------
# image


def sparco_scene(cfg, env, f_env=0.5, index=0.9, f_sec=0.03, ud=None):
    """UD primary + (optional) point secondary + Image environment, power-law spectra (SPARCO)."""
    sec = cfg["sec0"]
    parts = dict(
        star=UniformDisk(diam=cfg["ud0"] if ud is None else ud, flux=PowerLaw(1.0, -3.165, L0)),
        env=env.set("flux", PowerLaw(f_env, index, L0)) if hasattr(env, "set") else env,
    )
    if sec is not None:
        parts["sec"] = PointSource(flux=PowerLaw(f_sec, 0.0, L0), dra=sec[0], ddec=sec[1])
    return System(**parts)


def image_stage(args, out):
    cfg = cfg_of(args)
    data, files = load(cfg, args.files)
    t0 = time.time()
    # grid: the paper's pixel scale and field of view (parity with ORGANIC); coarser in --smoke
    scale = cfg["pixel_mas"] * (4.0 if args.smoke else 1.0)
    fov = cfg["fov_mas"] * (0.5 if args.smoke else 1.0)
    start = starting_image(
        data, star=True, oversample=nyquist_pixel_scale(data) / scale, largest_mas=fov,
        hole_mas=0.5 * beam(data).minor_mas)
    env = start.env
    print("grid", env.log_brightness.shape, "pixel", env.pixel_scale_mas, "mas; start flux", float(env.flux))
    scene = sparco_scene(cfg, env, f_env=float(np.asarray(env.flux)))
    priors = image_priors(scene) | {
        "star.diam": dist.Uniform(0.05, 1.5),
        "env.flux.ratio": dist.LogUniform(0.01, 20.0),
        "env.flux.index": dist.Uniform(-3.0, 5.0),
    }
    if cfg["sec0"] is not None:
        priors |= {
            "sec.flux.ratio": dist.Uniform(0.0, 0.3),
            "sec.dra": dist.Normal(cfg["sec0"][0], 1.0),
            "sec.ddec": dist.Normal(cfg["sec0"][1], 1.0),
        }
    weights = np.array([1e3, 1e1]) if args.smoke else np.logspace(*cfg.get("log_weights", (4, -2)), 9)
    opts = dict(max_steps=15) if args.smoke else dict(max_steps=800)
    curve = l_curve(scene, priors, data, MaxEntropy(1.0, path="env"), weights, **opts)
    corner = curve.corner() if len(weights) > 3 else None
    w_star = float(corner) if corner is not None else float(weights[len(weights) // 2])
    k = int(np.argmin(np.abs(np.log(np.asarray(curve.weights)) - np.log(w_star))))
    model = curve.results[k].model
    has_sec = cfg["sec0"] is not None
    c2, n = chi2r(model, data)
    bright = np.asarray(model.env.brightness)
    mv2, mt3 = observables(model, data)
    np.savez(
        path(out, cfg, "image.npz"), brightness=bright, pixel_mas=float(model.env.pixel_scale_mas),
        weights=np.asarray(curve.weights), chi2_red=np.asarray(curve.chi2_red).tolist(),
        penalty=np.asarray(curve.penalty), chosen=k, model_v2=mv2, model_t3=mt3,
        stars=json.dumps(dict(
            ud=float(model.star.diam), f_env=float(np.asarray(model.env.flux.ratio)),
            env_index=float(np.asarray(model.env.flux.index)),
            f_sec=float(np.asarray(model.sec.flux.ratio)) if has_sec else 0.0,
            sec_dra=float(model.sec.dra) if has_sec else None,
            sec_ddec=float(model.sec.ddec) if has_sec else None)),
    )
    p = json.loads(str(np.load(path(out, cfg, "image.npz"))["stars"]))
    tot = 1.0 + p["f_env"] + p["f_sec"]
    summary = dict(
        chi2_per_point=c2, n_points=n, weight=float(curve.weights[k]), corner_weight=w_star,
        grid=list(bright.shape), pixel_mas=float(model.env.pixel_scale_mas), smoke=args.smoke,
        seconds=time.time() - t0, f_prim_pct=100 / tot, f_sec_pct=100 * p["f_sec"] / tot,
        f_env_pct=100 * p["f_env"] / tot, **{k_: v for k_, v in p.items() if k_ in ("ud", "env_index", "sec_dra", "sec_ddec")},
    )
    print(json.dumps(summary, indent=1))
    json.dump(summary, open(path(out, cfg, "image.json"), "w"), indent=1)


# ----------------------------------------------------------------------------
# compare


def to_grid(img, scale_in, scale_out, npix_out):
    """Flux-conserving resample of a (North up, East left) image, centred on the star, to a new grid."""
    ny, nx = img.shape
    yy, xx = np.meshgrid(np.arange(npix_out), np.arange(npix_out), indexing="ij")
    c_out = (npix_out - 1) / 2.0
    c_in_y, c_in_x = (ny - 1) / 2.0, (nx - 1) / 2.0
    ys = c_in_y + (yy - c_out) * scale_out / scale_in
    xs = c_in_x + (xx - c_out) * scale_out / scale_in
    r = ndimage.map_coordinates(img, [ys, xs], order=1, mode="constant", cval=0.0)
    return r * (scale_out / scale_in) ** 2  # per-pixel flux scales with pixel area


def published_image(cfg):
    """The ORGANIC environment image in virgil's orientation (row 0 North, column 0 East), unit sum."""
    hdr, im, noise = zenodo_header(cfg)
    # FITS: CDELT1 < 0 so x runs East to West (column 0 is East already); CDELT2 > 0 so y runs
    # South to North, i.e. row 0 is South: flip the rows.
    assert hdr["CDELT1"] < 0 < hdr["CDELT2"]
    return im[::-1, :] / im.sum(), noise[::-1, :], abs(hdr["CDELT1"]), hdr


def published_scene(cfg):
    """The published SPARCO + ORGANIC model as a virgil scene (for chi2 on our data)."""
    img, _, scale, h = published_image(cfg)
    f1, f2 = h["SFLU1"], h["SFLU2"]
    f_env = 1.0 - f1 - f2 - h.get("SFLU3", 0.0)
    env = Image.from_brightness(img, scale, floor=0.0, flux=PowerLaw(f_env / f1, h["SIND0"], h["SWAVE0"]))
    parts = dict(star=UniformDisk(diam=h["SUD1"], flux=PowerLaw(1.0, h["SIND1"], h["SWAVE0"])), env=env)
    if f2 > 0:
        parts["sec"] = PointSource(flux=PowerLaw(f2 / f1, h["SIND2"], h["SWAVE0"]), dra=h["SDEX2"], ddec=h["SDEY2"])
    return System(**parts)


def region_metrics(img, scale, r_lo, r_hi):
    """Moments of the flux in the annulus r_lo <= r < r_hi mas around the star.

    ``img`` is North up, East left, centred on the star. The paper's own rim fit (an ellipse
    through radial peaks above 3 sigma, Sect. 5.1) needs the per-pixel significance and does not
    survive IW Car's arcs on a hand-off, so both images get the same moment-based numbers:
    flux fraction, flux-weighted radius, the position angle (East of North) and axis ratio of
    the second-moment ellipse, and the "asymmetry angle", the position angle of the flux
    centroid (the m=1 azimuthal maximum) with its offset from the star.
    """
    ny, nx = img.shape
    yy, xx = np.mgrid[:ny, :nx]
    dx = -(xx - (nx - 1) / 2.0) * scale  # East
    dy = -(yy - (ny - 1) / 2.0) * scale  # North (row 0 is North)
    rad = np.hypot(dx, dy)
    w = np.where((rad >= r_lo) & (rad < r_hi), img, 0.0)
    tot = w.sum()
    out = dict(flux_fraction=float(tot / img.sum()), mean_radius_mas=np.nan, major_pa_deg=np.nan,
               axis_ratio=np.nan, centroid_pa_deg=np.nan, centroid_offset_mas=np.nan)
    if tot <= 0:
        return out
    cx, cy = (w * dx).sum() / tot, (w * dy).sum() / tot
    cov = np.array([[(w * (dx - cx) ** 2).sum(), (w * (dx - cx) * (dy - cy)).sum()],
                    [(w * (dx - cx) * (dy - cy)).sum(), (w * (dy - cy) ** 2).sum()]]) / tot
    ev, vec = np.linalg.eigh(cov)
    major = vec[:, 1]  # (East, North) components
    out.update(
        mean_radius_mas=float((w * rad).sum() / tot),
        major_pa_deg=float(np.degrees(np.arctan2(major[0], major[1])) % 180.0),
        axis_ratio=float(np.sqrt(max(ev[0], 0) / ev[1])),
        centroid_pa_deg=float(np.degrees(np.arctan2(cx, cy)) % 360.0),
        centroid_offset_mas=float(np.hypot(cx, cy)),
    )
    return out


def ncc(a, b, mask):
    a, b = a[mask] - a[mask].mean(), b[mask] - b[mask].mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def compare_stage(args, out):
    cfg = cfg_of(args)
    data, _ = load(cfg, args.files)
    ours = np.load(path(out, cfg, "image.npz"))
    pub, noise, pub_scale, hdr = published_image(cfg)
    n = pub.shape[0]
    grid = to_grid(ours["brightness"], float(ours["pixel_mas"]), pub_scale, n)
    grid = grid / grid.sum()
    bm = beam(data)
    ours_c = np.asarray(convolve_beam(grid, pub_scale, bm))
    pub_c = np.asarray(convolve_beam(pub, pub_scale, bm))
    yy, xx = np.mgrid[:n, :n]
    rad = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2) * pub_scale
    mask = rad < cfg["mask_mas"]
    shifts = {}
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            shifts[(dy, dx)] = ncc(np.roll(ours_c, (dy, dx), (0, 1)), pub_c, mask)
    best = max(shifts, key=shifts.get)
    # controls: what the same statistic gives for a wrong azimuthal structure
    yy2, xx2 = np.mgrid[:n, :n]
    rbin = np.round(np.hypot(xx2 - (n - 1) / 2, yy2 - (n - 1) / 2)).astype(int)
    prof = np.bincount(rbin.ravel(), pub_c.ravel()) / np.bincount(rbin.ravel())
    ncc_controls = {
        "mirrored_East-West": ncc(ours_c, pub_c[:, ::-1], mask),
        "rotated_180": ncc(ours_c, pub_c[::-1, ::-1], mask),
        "azimuthal_average_of_published": ncc(ours_c, prof[rbin], mask),
    }
    regions = cfg["regions"]
    m_ours = {k: region_metrics(ours_c, pub_scale, *v) for k, v in regions.items()}
    m_pub = {k: region_metrics(pub_c, pub_scale, *v) for k, v in regions.items()}
    pub_model = published_scene(cfg)
    c2_pub, n_pts = chi2r(pub_model, data)
    flips = {}
    for name, im in {"as_published": pub, "mirrored_East-West": pub[:, ::-1], "mirrored_North-South": pub[::-1, :]}.items():
        env = Image.from_brightness(im, pub_scale, floor=0.0, flux=pub_model.env.flux)
        flips[name] = chi2r(pub_model.set("env", env), data)[0]
    res = dict(
        ncc_zero_shift=ncc(ours_c, pub_c, mask), ncc_controls=ncc_controls, ncc_best_shift=shifts[best],
        best_shift_pixels=dict(dy=best[0], dx=best[1]), pixel_mas=pub_scale, beam=dict(
            major_mas=bm.major_mas, minor_mas=bm.minor_mas, pa_deg=bm.pa_deg),
        ours=m_ours, published=m_pub, chi2_per_point_published_on_our_data=c2_pub,
        chi2_per_point_ours=json.load(open(path(out, cfg, "image.json")))["chi2_per_point"], n_points=n_pts,
        published_orientation_chi2=flips, smoke=json.load(open(path(out, cfg, "image.json")))["smoke"],
    )
    print(json.dumps(res, indent=1))
    json.dump(res, open(path(out, cfg, "compare.json"), "w"), indent=1)
    pv2, pt3 = observables(pub_model, data)
    np.savez(path(out, cfg, "compare_images.npz"), pub_v2=pv2, pub_t3=pt3, ours=grid, published=pub, ours_conv=ours_c, published_conv=pub_c,
             pixel_mas=pub_scale)
    plot_compare(out, cfg)
    plot_images(out, cfg)
    plot_fit(out, cfg, args.files)


def plot_compare(out, cfg):
    r = json.load(open(path(out, cfg, "compare.json")))
    z = np.load(path(out, cfg, "compare_images.npz"))
    s = float(z["pixel_mas"])
    n = z["ours"].shape[0]
    half = cfg["half_cmp_mas"]
    ext = [n * s / 2, -n * s / 2, -n * s / 2, n * s / 2]
    panels = [("virgil, native", z["ours"]), ("virgil, PIONIER beam", z["ours_conv"]),
              ("ORGANIC (Zenodo), PIONIER beam", z["published_conv"])]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.8))
    vmax = max(z["ours_conv"].max(), z["published_conv"].max())
    for a, (title, im) in zip(ax, panels):
        a.imshow(im / (im.max() if "native" in title else vmax), origin="upper", extent=ext, cmap="magma", vmin=0, vmax=1)
        a.set_title(title)
    th = np.linspace(0, 2 * np.pi, 200)
    rin = cfg.get("inner_rim_max_mas")
    for a in ax[:3] if rin else []:
        a.plot(rin * np.sin(th), rin * np.cos(th), "c-.", lw=0.8)
    diff = z["ours_conv"] / z["ours_conv"].max() - z["published_conv"] / z["published_conv"].max()
    im = ax[3].imshow(diff, origin="upper", extent=ext, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    ax[3].set_title("difference of peak-normalised\nconvolved images")
    plt.colorbar(im, ax=ax[3], fraction=0.046)
    for a in ax:
        a.set_xlim(half, -half)
        a.set_ylim(-half, half)
        a.plot(0, 0, "*", c="gold", ms=9, mec="k")
        a.set_xlabel("Δα (mas, East left)")
    ax[0].set_ylabel("Δδ (mas)")
    ax[0].set_title("virgil, native pixels")
    fig.suptitle(
        f"{cfg['name']}: NCC {r['ncc_zero_shift']:.2f} (best shift {r['ncc_best_shift']:.2f}); "
        f"χ²/pt virgil {r['chi2_per_point_ours']:.2f}, ORGANIC on our data {r['chi2_per_point_published_on_our_data']:.2f}"
        + ("  [SMOKE]" if r["smoke"] else ""), fontsize=11)
    fig.tight_layout()
    fig.savefig(path(out, cfg, "compare.png"), dpi=130)
    plt.close(fig)


def plot_images(out, cfg):
    """The page's headline figure: published (Zenodo FITS) and virgil, one style, one beam."""
    from matplotlib.patches import Ellipse

    r = json.load(open(path(out, cfg, "compare.json")))
    z = np.load(path(out, cfg, "compare_images.npz"))
    s, n, half = float(z["pixel_mas"]), z["ours"].shape[0], cfg["half_mas"]
    ext = [n * s / 2, -n * s / 2, -n * s / 2, n * s / 2]
    b = r["beam"]
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 5.2), sharey=True)
    for a, key, title in ((ax[0], "published_conv", "ORGANIC (De Prins et al. 2026)"),
                          (ax[1], "ours_conv", "virgil (MaxEntropy, L-curve)")):
        im = z[key]
        a.imshow(im / im.max(), origin="upper", extent=ext, cmap="magma", vmin=0, vmax=1)
        a.set_xlim(half, -half)
        a.set_ylim(-half, half)
        a.plot(0, 0, "*", c="gold", ms=11, mec="k")
        a.add_patch(Ellipse((half - 0.22 * half, -half + 0.2 * half), b["major_mas"], b["minor_mas"], angle=90 - b["pa_deg"],
                            fc="0.85", ec="w", alpha=0.9))
        a.set_xlabel("Δα (mas, East left)")
        a.set_title(title)
    ax[0].set_ylabel("Δδ (mas, North up)")
    if r["smoke"]:
        fig.suptitle("SMOKE RUN: pipeline check only", color="r")
    fig.tight_layout()
    fig.savefig(path(out, cfg, "images.png"), dpi=140)
    plt.close(fig)


def plot_fit(out, cfg, files=None):
    """V^2 and closure-phase fits of virgil and the published image on the same data."""
    data, _ = load(cfg, files)
    ours, z = np.load(path(out, cfg, "image.npz")), np.load(path(out, cfg, "compare_images.npz"))
    B = np.hypot(np.asarray(data.u), np.asarray(data.v)) / np.asarray(data.wavel) / 1e6
    vi = data.vis_index
    if vi is not None and np.ndim(vi) > 0:
        B = B[np.asarray(vi)]
    v2, e2 = np.asarray(data.vis), np.asarray(data.d_vis)
    t3, et3 = np.degrees(np.asarray(data.phi)), np.degrees(np.asarray(data.d_phi))
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].errorbar(B, v2, e2, fmt=".", c="0.6", ms=2, alpha=0.5, label="data")
    ax[0, 0].plot(B, ours["model_v2"], ".", c="C1", ms=3, label="virgil")
    ax[0, 0].plot(B, z["pub_v2"], ".", c="C0", ms=3, label="ORGANIC image")
    ax[0, 0].set_xlabel("B/λ (Mλ)")
    ax[0, 0].set_ylabel("V²")
    ax[0, 0].legend()
    ax[0, 1].plot(np.arange(t3.size), t3, ".", c="0.6", label="data")
    ax[0, 1].plot(ours["model_t3"], ".", c="C1", ms=3, label="virgil")
    ax[0, 1].plot(z["pub_t3"], ".", c="C0", ms=3, label="ORGANIC image")
    ax[0, 1].set_xlabel("closure-phase sample")
    ax[0, 1].set_ylabel("T3 (deg)")
    for a, obs, err, key in ((ax[1, 0], v2, e2, "v2"), (ax[1, 1], t3, et3, "t3")):
        for lab, c, mod in (("virgil", "C1", ours["model_" + key]), ("ORGANIC image", "C0", z["pub_" + key])):
            res = (obs - mod) / err
            a.hist(res, bins=np.linspace(-6, 6, 49), histtype="step", color=c, label=f"{lab}: rms {np.sqrt(np.mean(res**2)):.2f}")
        a.set_xlabel(f"({'V²' if key == 'v2' else 'T3'} data - model) / error")
        a.legend()
    fig.tight_layout()
    fig.savefig(path(out, cfg, "fit.png"), dpi=130)
    plt.close(fig)


def plot_lcurve(out, cfg):
    z = np.load(path(out, cfg, "image.npz"))
    chi2 = np.asarray(z["chi2_red"]).mean(axis=-1) if np.ndim(z["chi2_red"]) > 1 else np.asarray(z["chi2_red"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(chi2, z["penalty"], "o-")
    k = int(z["chosen"])
    ax.loglog(chi2[k], np.asarray(z["penalty"])[k], "r*", ms=14, label=f"chosen, weight {z['weights'][k]:.3g}")
    ax.set_xlabel("χ² per point")
    ax.set_ylabel("MaxEntropy penalty")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path(out, cfg, "lcurve.png"), dpi=130)
    plt.close(fig)


def replot_stage(args, out):
    cfg = cfg_of(args)
    if (path(out, cfg, "image.npz")).exists():
        plot_lcurve(out, cfg)
    if (path(out, cfg, "compare.json")).exists():
        plot_compare(out, cfg)
        plot_images(out, cfg)
        plot_fit(out, cfg, args.files)
    if (path(out, cfg, "inspect.json")).exists():
        data, _ = load(cfg, args.files)
        plot_inspect(data, json.load(open(path(out, cfg, "inspect.json"))), out, cfg)


# ----------------------------------------------------------------------------
# table: the page's parameter table (paper PMOIRED fit | virgil parametric fit | difference in sigma)


def _fmt(v, e=None):
    return f"{v:.3g}" if e is None else f"{v:.3g} ± {e:.2g}"


def table_stage(args, out):
    """Markdown rows of the parametric comparison, only for what the paper reports for this target."""
    from postagb_paper import IMAGE, PMOIRED  # noqa: PLC0415

    cfg = cfg_of(args)
    par = json.load(open(path(out, cfg, "parametric.json")))
    img = json.load(open(path(out, cfg, "image.json")))
    paper = PMOIRED[cfg["key"]]
    rows = [  # label, key in paper, virgil value, wrap period
        ("Primary flux fraction, H (%)", "f_prim", par["f_prim_pct"], None),
        ("Primary UD diameter (mas)", "ud", par["ud_prim_mas"], None),
        ("Secondary flux fraction, H (%)", "f_sec", par["f_sec_pct"], None),
        ("Resolved-background flux fraction, H (%)", "f_bg", par["f_bg_pct"], None),
        ("Secondary offset Δα (mas)", "sec_dra", par["sec_dra"], None),
        ("Secondary offset Δδ (mas)", "sec_ddec", par["sec_ddec"], None),
        ("Rim diameter (mas)", "theta_rim", par["rim_diam_mas"], None),
        ("Rim inclination (°)", "i_rim", par["rim_inc_deg"], None),
        ("Rim position angle (°)", "pa_rim", par["rim_pa_deg"], 180.0),
        ("Rim modulation amplitude (m = 1)", "a_mod", par["az_amp"], None),
        ("Rim modulation phase (°)", "psi_mod", par["az_pa_deg"], 360.0),
    ]
    lines = ["| Parameter | De Prins+2026 (PMOIRED) | virgil (parametric) | Difference |", "| --- | --- | --- | --- |"]
    for label, key, mine, period in rows:
        if key not in paper or mine is None:
            continue
        val, lo, hi = paper[key]
        d = mine - val
        if period:
            d = (d + period / 2) % period - period / 2
        sig = d / (lo if d < 0 else hi)
        lines.append(f"| {label} | {_fmt(val, max(lo, hi))} | {_fmt(mine)} | {d:+.3g} ({sig:+.1f}σ) |")
    lines.append(f"| χ² per point | {paper['chi2']:.2f} | {par['chi2_per_point']:.2f} | |")
    lines += ["", "ORGANIC/SPARCO parameters (Table 2 and FITS header) against the virgil image fit:", "",
              "| Parameter | De Prins+2026 (ORGANIC) | virgil (image fit) | Difference |", "| --- | --- | --- | --- |"]
    ip = IMAGE[cfg["key"]]
    for label, key, mine in (("Primary flux fraction, H (%)", "f_prim", img["f_prim_pct"]),
                             ("Primary UD diameter (mas)", "ud", img["ud"]),
                             ("Secondary flux fraction, H (%)", "f_sec", img["f_sec_pct"]),
                             ("Image spectral index", "image_index", img["env_index"])):
        if key in ip and not (key == "f_sec" and cfg["sec0"] is None):
            lines.append(f"| {label} | {_fmt(ip[key])} | {_fmt(mine)} | {mine - ip[key]:+.3g} |")
    text = "\n".join(lines)
    print(text)
    path(out, cfg, "table.md").write_text(text + "\n")


STAGES = dict(inspect=inspect_stage, parametric=parametric_stage, image=image_stage,
              compare=compare_stage, replot=replot_stage, table=table_stage)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stage", choices=list(STAGES))
    p.add_argument("outdir", type=Path)
    p.add_argument("--target", default="IW_Car", choices=list(TARGETS))
    p.add_argument("--smoke", action="store_true", help="tiny run: three files, coarse grid, few steps")
    p.add_argument("--files", type=int, default=None, help="use only the first N files")
    args = p.parse_args()
    if args.smoke and args.files is None:
        args.files = 3
    args.outdir.mkdir(parents=True, exist_ok=True)
    STAGES[args.stage](args, args.outdir)
    if args.stage == "image":
        plot_lcurve(args.outdir, cfg_of(args))


if __name__ == "__main__":
    sys.exit(main())
