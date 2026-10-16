"""virgil against published VLTI/PIONIER post-AGB disc images (De Prins et al. 2026; Hillen et al. 2016).

    python scripts/postagb_imaging.py {inspect,parametric,image,compare,replot} OUTDIR
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

``--smoke`` shrinks everything (three files, a coarse grid, two L-curve weights,
a handful of optimiser steps) so that it runs on a laptop in a few minutes; its
outputs only prove the pipeline runs. A full IW Car run (23 files, 128x128 pixels at 0.35 mas) takes
minutes on two CPU threads; larger targets should go to OzSTAR. Data: ``postagb_data.py fetch``;
published images: ``postagb_literature.py``. Add targets to TARGETS.

A target whose published image is only a bitmap (IRAS08544: Hillen et al. 2016, Fig. 4) sets
``bitmap`` in its TARGETS entry: ``compare`` then maps the bitmap's colours back to values
through its colour bar (morphology only; the stretch cannot be matched exactly), convolves virgil
with the beam ellipse drawn on the figure, and has no ORGANIC-style chi2 for the published image.
``secondary=False`` fits a single-star-subtracted scene (UD primary + image, the secondary stays
in the image), as that paper does.
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

DATA = Path("~/data/postagb_pionier").expanduser()
ZENODO = DATA / "literature/obs_logs_and_images"
LITERATURE = Path(__file__).resolve().parents[1] / "docs/imaging/postagb/literature"
NUHOR = Path(os.environ.get("POSTAGB_NUHOR", "~/code/nuHor/data")).expanduser()  # authors' reduced copies (IRAS08544)
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
        inner_rim_max_mas=5.0,  # inner arcs lie inside this radius, the outer arc beyond (paper Sect. 5.3.7)
        paper=dict(
            f_prim_pct=63.6, ud_prim_mas=0.50, f_sec_pct=1.99, sec_dra=1.12, sec_ddec=-1.90,
            image_index=0.89, rim_diam_mas=6.9, rim_inc_deg=48.0, rim_pa_deg=144.0,
            rim_dra=0.15, rim_ddec=-0.1, r_hl_mas=11.7,
            outer_inc_deg=57.0, outer_pa_deg=176.0, sphere_inc_deg=41.7, sphere_pa_deg=161.0,
        ),
    ),
    # Hillen et al. (2016, A&A 588, L1, arXiv:1603.03023). The 2015 data (ESO 094.D-0865, PI Hillen) have no
    # Phase 3 products: the authors' reduced copy has the nights of 2015-01-21 and -24 in A/ and 2015-02-23
    # in B/ (UT dates one day later in the file names), all three used by the paper (27 OBs).
    "IRAS08544": dict(
        name="IRAS 08544-4431",
        files=[NUHOR / "ep1/A/*.fits", NUHOR / "ep1/B/*.fits"],
        log_name=None,
        bitmap="IRAS08544_hillen_image.npz",
        pub_label="Hillen et al. 2016, Fig. 4",
        secondary=False,  # single-star-subtracted image: the secondary stays in the image
        star_index=-4.0,  # paper: primary flux ~ lambda^-4
        env_index0=0.42,  # paper's grid maximum: f_pri = 0.61, environment index 0.42
        ud0=0.5,
        fit_ud=False,  # the paper fixes the primary at a 0.5 mas uniform disc
        fov_mas=38.4,  # 256 pixels of 0.15 mas
        pixel_mas=0.15,
        sec0=(-0.67, -0.45),  # parametric start: the image puts the secondary SW of the primary
        regions=dict(inner=(0.0, 3.5), outer=(3.5, 16.0)),  # central point source and the ~15 mas ring
        inner_rim_max_mas=3.5,
        rim0=(14.15, 6.5, 19.0, 6.0), az_amps0=(0.43, 0.27), az_pas0=(0.0, 0.0),  # Hillen's best model as the start
        paper=dict(  # Table 1 (best model) of Hillen et al. 2016
            f_prim_pct=59.7, ud_prim_mas=0.5, f_sec_pct=3.9, f_rim_pct=20.9, f_bg_pct=15.5,
            bin_sep_mas=0.81, bin_pa_deg=56.0, rim_diam_mas=14.15, rim_width_frac=0.457,
            rim_inc_deg=19.0, rim_pa_deg=6.0, c1=0.35, s1=-0.25, c2=-0.05, s2=-0.27,
            image_f_prim=0.61, image_index=0.42,
        ),
    ),
}


for _k, _v in TARGETS.items():
    _v["key"] = _k


def path(out, cfg, name):
    """Output file for this target: <outdir>/<target>_<name>."""
    return out / f"{cfg['key']}_{name}"


# ----------------------------------------------------------------------------
# data


def cfg_of(args):
    return TARGETS[args.target]


def load_files(cfg, n=None):
    pats = cfg["files"] if isinstance(cfg["files"], (list, tuple)) else [cfg["files"]]
    files = sorted(f for pat in pats for f in glob.glob(str(pat)))
    if not files:
        raise SystemExit(f"no files match {cfg['files']}; run scripts/postagb_data.py fetch")
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
    """Model V^2 and closure phases (degrees) at the data's samples (no flagged samples assumed)."""
    cv = model.model(data.u, data.v, data.wavel)
    t3 = closure_phases(cv, data.i_cps1, data.i_cps2, data.i_cps3)
    return np.asarray(jnp.abs(cv) ** 2), np.degrees(np.asarray(t3))


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
    log = paper_log_mjds(cfg) if cfg["log_name"] else []
    matched = sum(any(abs(m - x) < 0.01 for x in log) for m in mjd) if log else None
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


def parametric_start(cfg):
    """The PMOIRED geometry of De Prins et al. (2026, Sect. 4.2) in virgil components."""
    sec = cfg["sec0"]
    return System(
        primary=UniformDisk(diam=cfg.get("ud0", 0.5), flux=PowerLaw(1.0, cfg.get("star_index", -3.165), L0)),
        secondary=PointSource(flux=PowerLaw(0.03, 0.0, L0), dra=sec[0], ddec=sec[1]),
        rim=ModulatedGaussianRim(*cfg.get("rim0", (12.0, 4.0, 50.0, 150.0)), az_amps=cfg.get("az_amps0", (0.3,)),
                                 az_pas=cfg.get("az_pas0", (0.0,)), flux=PowerLaw(0.4, 0.9, L0)),
        bg=Resolved(flux=PowerLaw(0.1, 0.0, L0)),
    )


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
    start = parametric_start(cfg)
    t0 = time.time()
    result = fit(start, PARAM_PRIORS, data, method="lbfgs", **({"max_steps": 25} if args.smoke else {}))
    model = result.model
    c2, n = chi2r(model, data)
    v = {k: np.asarray(x).tolist() for k, x in result.values.items()}
    ratio = lambda name: float(np.asarray(getattr(model, name).flux(L0)))  # noqa: E731
    total = 1.0 + ratio("secondary") + ratio("rim") + ratio("bg")
    summary = dict(
        values=v, chi2_per_point=c2, n_points=n, seconds=time.time() - t0, smoke=args.smoke,
        f_prim_pct=100.0 / total, f_sec_pct=100 * ratio("secondary") / total,
        f_rim_pct=100 * ratio("rim") / total, f_bg_pct=100 * ratio("bg") / total,
        ud_prim_mas=float(model.primary.diam), rim_diam_mas=float(model.rim.diam),
        rim_fwhm_mas=float(model.rim.fwhm), rim_inc_deg=float(model.rim.inc),
        rim_pa_deg=float(model.rim.pa) % 180.0, rim_index=float(np.asarray(model.rim.flux.index)),
        az_amp=float(np.ravel(model.rim.az_amps)[0]), az_pa_deg=float(np.ravel(model.rim.az_pas)[0]) % 360.0,
        az_amps=[float(a) for a in np.ravel(model.rim.az_amps)],
        az_pas_deg=[float(a) % 360.0 for a in np.ravel(model.rim.az_pas)],
        sec_dra=float(model.secondary.dra), sec_ddec=float(model.secondary.ddec),
        sec_sep_mas=float(np.hypot(model.secondary.dra, model.secondary.ddec)),
        sec_pa_deg=float(np.degrees(np.arctan2(model.secondary.dra, model.secondary.ddec)) % 360.0),
        rim_dra=float(model.rim.dra), rim_ddec=float(model.rim.ddec),
    )
    print(json.dumps(summary, indent=1))
    json.dump(summary, open(path(out, cfg, "parametric.json"), "w"), indent=1)


# ----------------------------------------------------------------------------
# image


def sparco_scene(cfg, env, f_env=0.5, index=None, f_sec=0.03, ud=None):
    """UD primary + point secondary + Image environment, power-law spectra (SPARCO).

    With ``secondary=False`` in the target's config the secondary is left out (it stays in the image).
    """
    sec = cfg["sec0"]
    index = cfg.get("env_index0", 0.9) if index is None else index
    comps = dict(
        star=UniformDisk(diam=cfg.get("ud0", 0.5) if ud is None else ud,
                         flux=PowerLaw(1.0, cfg.get("star_index", -3.165), L0)),
        env=env.set("flux", PowerLaw(f_env, index, L0)) if hasattr(env, "set") else env,
    )
    if cfg.get("secondary", True):
        comps = dict(star=comps["star"], sec=PointSource(flux=PowerLaw(f_sec, 0.0, L0), dra=sec[0], ddec=sec[1]),
                     env=comps["env"])
    return System(**comps)


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
    scene = sparco_scene(cfg, env, f_env=float(np.asarray(env.flux)))  # index: cfg env_index0
    priors = image_priors(scene) | {
        "env.flux.ratio": dist.LogUniform(0.01, 20.0),
        "env.flux.index": dist.Uniform(-3.0, 5.0),
    }
    if cfg.get("fit_ud", True):  # fixed in the paper for IRAS08544 (0.5 mas); a free UD collapses onto its bound there
        priors["star.diam"] = dist.Uniform(0.05, 1.5)
    if cfg.get("secondary", True):
        priors |= {
            "sec.flux.ratio": dist.Uniform(0.0, 0.3),
            "sec.dra": dist.Normal(cfg["sec0"][0], 1.0),
            "sec.ddec": dist.Normal(cfg["sec0"][1], 1.0),
        }
    weights = np.array([1e3, 1e1]) if args.smoke else np.logspace(4, -2, 9)
    opts = dict(max_steps=15) if args.smoke else dict(max_steps=800)
    curve = l_curve(scene, priors, data, MaxEntropy(1.0, path="env"), weights, **opts)
    corner = curve.corner() if len(weights) > 3 else None
    w_star = float(corner) if corner is not None else float(weights[len(weights) // 2])
    k = int(np.argmin(np.abs(np.log(np.asarray(curve.weights)) - np.log(w_star))))
    model = curve.results[k].model
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
            **(dict(f_sec=float(np.asarray(model.sec.flux.ratio)), sec_dra=float(model.sec.dra),
                    sec_ddec=float(model.sec.ddec)) if cfg.get("secondary", True) else dict(f_sec=0.0)))),
    )
    p = json.loads(str(np.load(path(out, cfg, "image.npz"))["stars"]))
    tot = 1.0 + p["f_env"] + p["f_sec"]
    summary = dict(
        chi2_per_point=c2, n_points=n, weight=float(curve.weights[k]), corner_weight=w_star,
        grid=list(bright.shape), pixel_mas=float(model.env.pixel_scale_mas), smoke=args.smoke,
        seconds=time.time() - t0, f_prim_pct=100 / tot, f_sec_pct=100 * p["f_sec"] / tot if p["f_sec"] else None,
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
    return System(
        star=UniformDisk(diam=h["SUD1"], flux=PowerLaw(1.0, h["SIND1"], h["SWAVE0"])),
        sec=PointSource(flux=PowerLaw(f2 / f1, h["SIND2"], h["SWAVE0"]), dra=h["SDEX2"], ddec=h["SDEY2"]),
        env=env,
    )


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


def bitmap_published(cfg):
    """The published bitmap image (peak-normalised, North up East left), its pixel scale (mas) and drawn beam."""
    z = np.load(LITERATURE / cfg["bitmap"])
    return np.asarray(z["image"], float), float(z["pixel_mas"]), json.loads(str(z["beam"])), z


def compare_bitmap_stage(args, out):
    """Compare with an image that exists only as a bitmap: morphology at the paper's drawn beam."""
    from virgil.imaging import Beam

    cfg = cfg_of(args)
    data, _ = load(cfg, args.files)
    ours = np.load(path(out, cfg, "image.npz"))
    pub, scale, pb, _ = bitmap_published(cfg)
    n = pub.shape[0]
    grid = to_grid(ours["brightness"], float(ours["pixel_mas"]), scale, n)
    grid = grid / grid.sum()
    bm = Beam(pb["major_mas"], pb["minor_mas"], pb["pa_deg"])  # the ellipse drawn on the published figure
    ours_c = np.asarray(convolve_beam(grid, scale, bm))
    # the published image is already a restored (beam-convolved) image; do not convolve it again
    pub = pub / pub.sum()
    yy, xx = np.mgrid[:n, :n]
    rad = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2) * scale
    mask = rad < 15.0
    step = int(round(0.15 / scale))  # shifts of up to 0.45 mas
    shifts = {(dy, dx): ncc(np.roll(ours_c, (dy * step, dx * step), (0, 1)), pub, mask)
              for dy in range(-3, 4) for dx in range(-3, 4)}
    best = max(shifts, key=shifts.get)
    m_ours = {k: region_metrics(ours_c, scale, *v) for k, v in cfg["regions"].items()}
    m_pub = {k: region_metrics(pub, scale, *v) for k, v in cfg["regions"].items()}
    flips = {name: ncc(ours_c, im, mask) for name, im in
             {"as_published": pub, "mirrored_East-West": pub[:, ::-1], "mirrored_North-South": pub[::-1, :]}.items()}
    img_json = json.load(open(path(out, cfg, "image.json")))
    res = dict(
        ncc_zero_shift=ncc(ours_c, pub, mask), ncc_best_shift=shifts[best],
        best_shift_mas=dict(dy=best[0] * step * scale, dx=best[1] * step * scale), pixel_mas=scale,
        beam=pb, virgil_data_beam=dict(major_mas=beam(data).major_mas, minor_mas=beam(data).minor_mas,
                                       pa_deg=beam(data).pa_deg),
        ours=m_ours, published=m_pub, ncc_orientation=flips,
        chi2_per_point_ours=img_json["chi2_per_point"], n_points=int(img_json["n_points"]), smoke=img_json["smoke"],
    )
    print(json.dumps(res, indent=1))
    json.dump(res, open(path(out, cfg, "compare.json"), "w"), indent=1)
    np.savez_compressed(path(out, cfg, "compare_images.npz"), ours=grid.astype("f4"), ours_conv=ours_c.astype("f4"),
                        published_conv=pub.astype("f4"), pixel_mas=scale)
    plot_images(out, cfg)
    plot_bitmap_compare(out, cfg)
    plot_fit_ours(out, cfg, args.files)


def plot_bitmap_compare(out, cfg):
    """Difference and metrics panel (published image as recovered from its bitmap)."""
    r = json.load(open(path(out, cfg, "compare.json")))
    z = np.load(path(out, cfg, "compare_images.npz"))
    s, n, half = float(z["pixel_mas"]), z["ours"].shape[0], 16.0
    ext = [n * s / 2, -n * s / 2, -n * s / 2, n * s / 2]
    a, b = z["ours_conv"] / z["ours_conv"].max(), z["published_conv"] / z["published_conv"].max()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    for q, im, t in ((ax[0], a, "virgil at the published beam"), (ax[1], b, "Hillen+2016, recovered from the bitmap")):
        q.imshow(im, origin="upper", extent=ext, cmap="magma", vmin=0, vmax=1)
        q.set_title(t)
    d = ax[2].imshow(a - b, origin="upper", extent=ext, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    ax[2].set_title("difference of peak-normalised images")
    plt.colorbar(d, ax=ax[2], fraction=0.046)
    for q in ax:
        q.set_xlim(half, -half)
        q.set_ylim(-half, half)
        q.plot(0, 0, "*", c="gold", ms=9, mec="k")
        q.set_xlabel("Δα (mas, East left)")
    ax[0].set_ylabel("Δδ (mas)")
    fig.suptitle(f"{cfg['name']}: NCC {r['ncc_zero_shift']:.2f} (best shift {r['ncc_best_shift']:.2f})"
                 + ("  [SMOKE]" if r["smoke"] else ""), fontsize=11)
    fig.tight_layout()
    fig.savefig(path(out, cfg, "compare.png"), dpi=130)
    plt.close(fig)


def compare_stage(args, out):
    cfg = cfg_of(args)
    if cfg.get("bitmap"):
        return compare_bitmap_stage(args, out)
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
    mask = rad < 20.0
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
    rin = cfg["inner_rim_max_mas"]
    regions = dict(inner=(0.0, rin), outer=(rin, 20.0))
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
    half = 15.0
    ext = [n * s / 2, -n * s / 2, -n * s / 2, n * s / 2]
    panels = [("virgil, native", z["ours"]), ("virgil, PIONIER beam", z["ours_conv"]),
              ("ORGANIC (Zenodo), PIONIER beam", z["published_conv"])]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.8))
    vmax = max(z["ours_conv"].max(), z["published_conv"].max())
    for a, (title, im) in zip(ax, panels):
        a.imshow(im / (im.max() if "native" in title else vmax), origin="upper", extent=ext, cmap="magma", vmin=0, vmax=1)
        a.set_title(title)
    th = np.linspace(0, 2 * np.pi, 200)
    for a in ax[:3]:
        a.plot(cfg["inner_rim_max_mas"] * np.sin(th), cfg["inner_rim_max_mas"] * np.cos(th), "c-.", lw=0.8)
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

    if cfg.get("bitmap"):
        return plot_images_bitmap(out, cfg)
    r = json.load(open(path(out, cfg, "compare.json")))
    z = np.load(path(out, cfg, "compare_images.npz"))
    s, n, half = float(z["pixel_mas"]), z["ours"].shape[0], 18.0
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
        a.add_patch(Ellipse((half - 4, -half + 3.5), b["major_mas"], b["minor_mas"], angle=90 - b["pa_deg"],
                            fc="0.85", ec="w", alpha=0.9))
        a.set_xlabel("Δα (mas, East left)")
        a.set_title(title)
    ax[0].set_ylabel("Δδ (mas, North up)")
    if r["smoke"]:
        fig.suptitle("SMOKE RUN: pipeline check only", color="r")
    fig.tight_layout()
    fig.savefig(path(out, cfg, "images.png"), dpi=140)
    plt.close(fig)


def plot_images_bitmap(out, cfg):
    """Headline figure for a bitmap-only publication: the bitmap as published (left), virgil in the
    same colour map, field of view, orientation, linear 0-1 stretch and beam (right)."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Ellipse

    r = json.load(open(path(out, cfg, "compare.json")))
    z = np.load(path(out, cfg, "compare_images.npz"))
    _, _, pb, lit = bitmap_published(cfg)
    s, n, half = float(z["pixel_mas"]), z["ours"].shape[0], 19.2
    ext = [n * s / 2, -n * s / 2, -n * s / 2, n * s / 2]
    cmap = ListedColormap(np.clip(lit["cmap"], 0, 1))  # the published figure's own colour bar
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 5.2), sharey=True)
    ax[0].imshow(lit["crop"], extent=[half, -half, -half, half], interpolation="nearest")
    im = z["ours_conv"]
    ax[1].imshow(im / im.max(), origin="upper", extent=ext, cmap=cmap, vmin=0, vmax=1)
    ax[1].plot(0, 0, "*", c="#2a52be", ms=9, mec="k")
    ax[1].add_patch(Ellipse((half - 2.5, half - 2.2), pb["minor_mas"], pb["major_mas"], angle=-pb["pa_deg"],
                            fc="#8080ff", ec="#4040c0", alpha=0.9))
    for a, title in zip(ax, (f"as published ({cfg['pub_label']})", "virgil (MaxEntropy, L-curve)")):
        a.set_xlim(half, -half)
        a.set_ylim(-half, half)
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


def plot_fit_ours(out, cfg, files=None):
    """V^2 and closure-phase fits of virgil alone (no published model image to overplot)."""
    data, _ = load(cfg, files)
    ours = np.load(path(out, cfg, "image.npz"))
    B = np.hypot(np.asarray(data.u), np.asarray(data.v)) / np.asarray(data.wavel) / 1e6
    v2, e2 = np.asarray(data.vis), np.asarray(data.d_vis)
    t3, et3 = np.degrees(np.asarray(data.phi)), np.degrees(np.asarray(data.d_phi))
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].errorbar(B, v2, e2, fmt=".", c="0.6", ms=2, alpha=0.5, label="data")
    ax[0, 0].plot(B, ours["model_v2"], ".", c="C1", ms=3, label="virgil")
    ax[0, 0].set_xlabel("B/λ (Mλ)")
    ax[0, 0].set_ylabel("V²")
    ax[0, 0].legend()
    ax[0, 1].plot(t3, ".", c="0.6", label="data")
    ax[0, 1].plot(ours["model_t3"], ".", c="C1", ms=3, label="virgil")
    ax[0, 1].set_xlabel("closure-phase sample")
    ax[0, 1].set_ylabel("T3 (deg)")
    for a, obs, err, key in ((ax[1, 0], v2, e2, "v2"), (ax[1, 1], t3, et3, "t3")):
        res = (obs - ours["model_" + key]) / err
        a.hist(res, bins=np.linspace(-6, 6, 49), histtype="step", color="C1", label=f"virgil: rms {np.sqrt(np.mean(res**2)):.2f}")
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
    if (path(out, cfg, "compare.json")).exists() and cfg.get("bitmap"):
        plot_images(out, cfg)
        plot_bitmap_compare(out, cfg)
        plot_fit_ours(out, cfg, args.files)
    elif (path(out, cfg, "compare.json")).exists():
        plot_compare(out, cfg)
        plot_images(out, cfg)
        plot_fit(out, cfg, args.files)
    if (path(out, cfg, "inspect.json")).exists():
        data, _ = load(cfg, args.files)
        plot_inspect(data, json.load(open(path(out, cfg, "inspect.json"))), out, cfg)


STAGES = dict(inspect=inspect_stage, parametric=parametric_stage, image=image_stage,
              compare=compare_stage, replot=replot_stage)


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
