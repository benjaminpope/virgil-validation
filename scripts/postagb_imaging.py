"""virgil against published VLTI/PIONIER post-AGB disc images (De Prins et al. 2026; Hillen et al. 2016).

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
* ``calibrate``  does the gp-polar hyperparameter choice recover a smooth known disc on this target's own (u,v)?
                 ``--arm {white,corr,smear,geo,night}``, ``--m-max``, ``--seed`` (design/gp_polar_hyperparameters.md).
* ``replot``     redraw every figure from the saved results (no fitting).
* ``table``      markdown parameter table: De Prins et al. PMOIRED fit | virgil | difference in sigma.

``--smoke`` shrinks everything (three files, a coarse grid, two L-curve weights,
a handful of optimiser steps) so that it runs on a laptop in a few minutes; its
outputs only prove the pipeline runs. A full IW Car run (23 files, 128x128 pixels at 0.35 mas) takes
minutes on two CPU threads; larger targets should go to OzSTAR. Data: ``postagb_data.py fetch``;
published images: ``postagb_literature.py``. Add targets to TARGETS.

A target whose published image is only a bitmap (IRAS08544: Hillen et al. 2016, Fig. 4) sets
``bitmap`` in its TARGETS entry: ``compare`` then maps the bitmap's colours back to values
through its colour bar (morphology only; the stretch cannot be matched exactly), convolves virgil
with the beam ellipse drawn on the figure, and has no ORGANIC-style chi2 for the published image.
IRAS08544 is compared with the binary-subtracted panel (Fig. 4, lower): the scene has a free secondary
(``sec_index0`` frees its power-law index) and both stars are blanked in the image support.

Per-target options: the window blanked in the image support around each central star has radius
max(lambda_min / (2 B_max), the starting UD diameter) (``hole_mas`` overrides), so the pixels cannot trade flux with
the stellar point sources or UD fluxes; ``bg`` adds a grey over-resolved background (SPARCO ``f_bg``, free) to the
image scene; the regularisation weight is set by the discrepancy principle (chi2 per point = 1 on the errors as
given) for MaxEntropy, and for ``gp=dict(sigmas, length_beams)`` (a ``GaussianField`` prior) the grid cell that is
whose chi2 per point is closest to 1, ties broken by the Laplace evidence;
``secondary=False`` fits a single-star-subtracted scene (the secondary stays in the image).

``--noise vlti`` swaps the independent errors and MacKay ``error_scale`` for virgil-vlti's PIONIER error model (``VLTI_NOISE``:
V^2 and closure-phase errors scaled by kind, gains per telescope and baseline, closure-phase offsets; no bandwidth smearing, which virgil does not support for pixel Images) and
``--mean flat`` gives the GaussianField / MultipoleField a flat template inside the support instead of the Gaussian start blob;
outputs carry ``_vlti`` / ``_flat`` in their names. ``--noise night`` adds calibration terms shared by the frames of a night
(``night_noise``; outputs ``_night``), and ``--unbounded`` lifts gp-polar's prior bounds (outputs ``_unbounded``). ``--chrom`` adds a chromatic ring (index linear in radius) and frees the star index
(design/gp_polar_hyperparameters.md §11; outputs ``_chrom``). ``--binary free|motion|epoch`` frees the secondary and moves the stars against the disc (design §12; outputs ``_bin<mode>``);
``--noise nightw`` keeps the MacKay white errors and adds night terms of 3 % and 2 deg (``NIGHTW``; outputs ``_nightw``), and ``--wide-field`` sets the field half-widths
to 15 mas (outputs ``_wide``; PS Gem's field was 5 mas until 18 Nov 2026, now the paper's, 19 mas). A support below the data's field (``check_support``) is refused, and an image that is flat or piled at the support edge (``image_degeneracy``) is flagged and not scored. The geometry of the polar GP still comes from the ``scale`` stage (current noise).
``--companion {fixed,free} --companion-start E,N`` adds a point companion to gp-polar (flux ratio and index free; position held or free, the chosen cell also
refitted from the mirror start; outputs ``_comp<arm>``), ``--hole-scale F`` scales the hole (``_holex<F>``), ``--geometry INC,PA`` replaces the scale stage's polar inclination
and position angle (``_geo<inc>-<pa>``) and ``--white V2,T3`` the pair-test white scales of ``--noise vlti|night`` (``_white<v2>-<t3>``); see reports/postagb-binaries-psgem-review.md in virgil-paper.
``scripts/postagb_noise_check.py`` measures the residuals and runs the synthetic evidence check (design/postagb_noise_model.md).
"""

import argparse
import datetime
import hashlib
from scipy.ndimage import rotate as rotate_img
import glob
import json
import os
import pathlib
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import equinox as eqx  # noqa: E402
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
from virgil import OIData  # noqa: F401
sys.path.insert(0, str(Path(__file__).resolve().parent))
from postagb_noise import (  # noqa: E402  (the PIONIER data layer and noise model, on virgil-vlti)
    NIGHT_KEYS, NIGHT_SPLIT_MJD, NIGHT_WIDTHS, NIGHTW, NOISE_WIDTHS, VLTI_NOISE, load_pionier, night_labels, night_noise, night_widths,
    noise_widths, pionier_noise, vlti_noise,
)
from virgil.oidata import closure_phases  # noqa: E402
from virgil.fitting import fit  # noqa: E402
from virgil import gp_priors  # noqa: E402
from virgil.fields import GaussianField, MultipoleField  # noqa: E402
from virgil.imaging import (  # noqa: E402
    MaxEntropy, beam, convolve_beam, error_scale, image_priors, l_curve, log_evidence, nyquist_pixel_scale,
    starting_image, whitened_residuals,
)
from virgil.imaging import _evidence_latents, _log_norm, _residual_jacobian  # noqa: E402  (scene_evidence; private, so pinned)
from virgil.models import (  # noqa: E402
    Component, GaussianRing, Harmonics, Image, ModulatedGaussianRim, MultipoleDisk, MultipoleKernel, PointSource, RadialGP, RadialLinear, Resolved,
    System, UniformDisk,
)
from virgil.spectra import PowerLaw  # noqa: E402

DATA = Path(os.environ.get("POSTAGB_DATA", "~/data/postagb_pionier")).expanduser()  # POSTAGB_DATA: the OzSTAR copy
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
        half_mas=18.0, mask_mas=20.0, half_cmp_mas=15.0, ud0=0.5, ud_fix=0.5,
        inner_rim_max_mas=5.0,  # inner arcs lie inside this radius, the outer arc beyond (paper Sect. 5.3.7)
        regions=dict(inner=(0.0, 5.0), outer=(5.0, 20.0)),
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
        bitmap="IRAS08544_hillen_binsub.npz",
        pub_label="Hillen et al. 2016, Fig. 4 lower panel",
        # The core is a binary: the scene has a point-source secondary, free in position, flux and power-law index,
        # started at the published values (Table 1: separation 0.81 mas, PA of the primary from the secondary 56 deg,
        # f_sec 3.9 % against f_pri 59.7 %; T_sec = 4000 K, a local index of -2.5 at 1.65 um). Both stars get a
        # window in the image support, as in the paper's binary-subtracted reconstruction.
        sec_index0=-2.5, f_sec0=0.065,
        star_index=-4.0,  # paper: primary flux ~ lambda^-4
        env_index0=0.42,  # paper's grid maximum: f_pri = 0.61, environment index 0.42
        ud0=0.5, ud_fix=0.5,
        fit_ud=False,  # the paper fixes the primary at a 0.5 mas uniform disc
        fov_mas=38.4,  # 256 pixels of 0.15 mas
        pixel_mas=0.15,
        half_mas=16.0, half_cmp_mas=14.0,  # the ~15 mas ring and a little beyond
        sec0=(-0.671, -0.453),  # 0.81 mas at PA 236 deg (56 + 180) from the primary
        regions=dict(inner=(0.0, 3.5), outer=(3.5, 16.0)),  # central point source and the ~15 mas ring
        inner_rim_max_mas=3.5,
        # Blind starts for the parametric fit (none uses the paper's values; the best chi2 is kept).
        param_starts=[
            dict(rim0=(10.0, 3.0, 40.0, 90.0), az_amps0=(0.3, 0.1), az_pas0=(0.0, 0.0), sec0=(0.8, 0.8)),
            dict(rim0=(18.0, 5.0, 10.0, 0.0), az_amps0=(0.3, 0.1), az_pas0=(180.0, 90.0), sec0=(-0.8, 0.8)),
            dict(rim0=(12.0, 4.0, 30.0, 150.0), az_amps0=(0.3, 0.1), az_pas0=(90.0, 0.0), sec0=(0.8, -0.8)),
            dict(rim0=(16.0, 8.0, 25.0, 45.0), az_amps0=(0.3, 0.1), az_pas0=(270.0, 45.0), sec0=(-0.8, -0.8)),
        ],
        paper=dict(  # Table 1 (best model) of Hillen et al. 2016
            f_prim_pct=59.7, ud_prim_mas=0.5, f_sec_pct=3.9, f_rim_pct=20.9, f_bg_pct=15.5,
            bin_sep_mas=0.81, bin_pa_deg=56.0, rim_diam_mas=14.15, rim_width_frac=0.457,
            rim_inc_deg=19.0, rim_pa_deg=6.0, c1=0.35, s1=-0.25, c2=-0.05, s2=-0.27,
            image_f_prim=0.61, image_index=0.42,
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
TARGETS["IRAS08544"]["log_weights"] = (3, 0)  # the full (4, -2) sweep reaches a non-finite fit at its low-weight end here
TARGETS["PS_Gem"]["log_weights"] = (2, -4)  # the MaxEntropy sweep, kept for the comparison
# ... and the MaxEntropy image is arbitrary on a flat L-curve (NCC 0.63), so PS Gem is imaged with a GP prior instead:
# a (sigma, length) grid, lengths in beams, the cell with the largest Laplace evidence kept.
TARGETS["PS_Gem"]["gp"] = dict(sigmas=(0.25, 0.5, 1.0, 2.0, 4.0), length_beams=(0.25, 0.5, 1.0, 2.0))
TARGETS["HR4049"]["gp"] = dict(sigmas=(0.25, 0.5, 1.0, 2.0, 4.0), length_beams=(0.25, 0.5, 1.0, 2.0))  # MaxEntropy came out too compact
TARGETS["HR4049"]["half_mas"] = 25.0  # the extent of the paper's Fig. 4 panel
# Field sizes (Ben): the support is a disc (mask_mas) in every target. HR 4049's ORGANIC halo extends past 20 mas and was clipped by the
# old +-18 mas square, so its grid grows to +-25.5 mas (146 px of 0.35 mas) with a disc of 25 mas. Other pixel scales are the paper's.
# PS Gem had a hand-set 5 mas support (fov_override=11, half/half_cmp/mask 5 mas) until 18 Nov 2026. That cut off emission that the shortest
# baselines (9.9 m) still see: up to 47 % of the disc's flux sat in the outer 15 % of the support radius, the over-resolved background took
# 6-10 % of the flux (0-0.7 % on a 15 mas field) and under night terms the disc fell to 1.6 % and a flat image. It now takes the paper's
# field like the other targets, and `check_support` refuses a support smaller than the data's own field.
TARGETS["PS_Gem"].update(hole_mas=0.37)  # hole at the UD only: ORGANIC puts compact emission inside 1.5 mas, the 1.2 mas beam hole blanked it
# The grid must contain the 25 mas disc: 0.35 mas pixels (about 4 per lambda/2B beam) and a 51 mas field, 146 pixels.
TARGETS["HR4049"].update(polar_lr_beams=(0.5, 1.0, 2.0, 4.0, 8.0), gp_mults=(0.5, 1.0, 2.0, 4.0, 8.0), fov_override=51.0, pixel_override=0.35, half_cmp_mas=25.0, mask_mas=25.0)

for _k, _v in TARGETS.items():
    _v["bg"] = _k != "IRAS08544"  # De Prins+2026 images account for an over-resolved flux; Hillen's image scene has none
    _v["key"] = _k
    _v.setdefault("files", None)  # None: the files of the paper's observation log, from the archive


def path(out, cfg, name):
    """Output file for this target: <outdir>/<target>_<name>."""
    return out / f"{cfg['key']}_{name}"


# ----------------------------------------------------------------------------
# data


def paper_indices(cfg):
    """(star, companion) spectral indices of the paper's ORGANIC model: SIND1 and SIND2 of its FITS header."""
    ph = cfg.get("paper_header")
    if ph and "sec_index" in ph:
        return float(ph["star_index"]), float(ph["sec_index"])
    hdr, _, _ = zenodo_header(cfg)
    return float(hdr["SIND1"]), float(hdr["SIND2"])


def secondary_defaults(cfg, target):
    """The default secondary treatment of every target with one (except IRAS 08544, which has its own, from Hillen+2016): position and flux
    ratio from De Prins's PMOIRED fit as informative priors where it has them (position sigma 0.05 mas, flux sigma 3 % of the total), the
    index fixed at the paper's SIND2 (0), no window around it when it lies inside the primary's hole. For HD95767 this took the inner knots
    out of the image (NCC 0.90 -> 0.95, log Z +342) and the separation from 1.04 to 0.89 mas (paper 0.88)."""
    if cfg.get("_sec_done") or cfg.get("sec0") is None or "sec_index0" in cfg or not cfg.get("secondary", True):
        return
    from postagb_paper import PMOIRED  # noqa: PLC0415

    cfg["_sec_done"] = True
    cfg["sec_index_fixed"] = paper_indices(cfg)[1]  # the paper's own value (ORGANIC SIND2, 0), fixed: indices are only comparable that way
    cfg["sec_index_source"] = "ORGANIC SIND2"
    pm = PMOIRED.get(target, {})
    if pm.get("sec_dra"):
        cfg["sec0"] = (pm["sec_dra"][0], pm["sec_ddec"][0])
        cfg["sec_prior"] = dict(dra=pm["sec_dra"][0], ddec=pm["sec_ddec"][0], pos_sigma=0.05)
        if pm.get("f_sec") and pm.get("f_prim"):
            cfg["f_sec0"] = pm["f_sec"][0] / pm["f_prim"][0]
            cfg["sec_prior"] |= dict(ratio=cfg["f_sec0"], ratio_sigma=0.03 / (pm["f_prim"][0] / 100.0))
    else:
        cfg["sec_prior"] = dict(dra=cfg["sec0"][0], ddec=cfg["sec0"][1], pos_sigma=0.1)


WIDE_FIELD_MAS = 15.0  # --wide-field: image, support and comparison half-widths (design §12, when PS Gem's stock field was 5 mas)

# check_support: emission at radius r keeps a ring visibility J0(2 pi B r / lambda) on a baseline B. Outside the support it can only go to the
# over-resolved background (visibility 0), which is wrong where the shortest baseline still sees it at more than a half: J0(x) = 0.5 at x = 1.52.
SUPPORT_J0_HALF_X = 1.52


def support_floor_mas(data):
    """The smallest support radius (mas) the data allow: a ring just outside it is at visibility <= 0.5 on the shortest baseline, at the
    longest wavelength."""
    b = np.hypot(np.asarray(data.u), np.asarray(data.v))
    b_min = float(b[b > 0].min())
    return SUPPORT_J0_HALF_X / (2 * np.pi) * float(np.max(np.asarray(data.wavel))) / b_min * 180 / np.pi * 3.6e6


def check_support(cfg, data):
    """Refuse a support disc smaller than ``support_floor_mas``: the disc would be truncated and its outer flux pushed into the background
    (PS Gem's 5 mas support, 18 Nov 2026). Returns the floor."""
    floor = support_floor_mas(data)
    if cfg.get("mask_mas") is not None and cfg["mask_mas"] < floor:
        raise SystemExit(f"{cfg['key']}: support radius {cfg['mask_mas']:.1f} mas is below the data's field ({floor:.1f} mas: a ring there "
                         "is still at visibility > 0.5 on the shortest baseline); widen mask_mas")
    return floor


# image_degeneracy: an image that is the prior's shape, not the data's, is flagged and not scored. The edge is the image's own field
# boundary (gp-polar's field ends on an inclined ellipse inside the circular support), and the band is 0.15 of the field's half-width
# (22-27 % of its area). Calibrated on the 40 gp-polar BN images (job 21123993): outside PS Gem's 5 mas support, flat <= 0.04 and
# edge <= 0.20 (AI Sco's highest); PS Gem's 5 mas support gave edge 0.28-0.59 and, under night terms, flat 0.97.
DEGENERATE_FLAT_FRAC = 0.5  # flat: more than this fraction of the field (outside the hole) above half the image's peak
DEGENERATE_EDGE_FRAC = 0.25  # edge-piled: more than this fraction of the disc's flux in the band along the field's boundary


def image_degeneracy(bright, pix, support_mas, hole_mas):
    """Flatness and edge-piling of a rendered disc image (``bright``, square, centred, ``pix`` mas): ``dict(img_flat_frac,
    img_edge_frac, image_degenerate)``, the last a list of reasons (empty when the image is usable). The field is where the image is
    non-zero inside the support disc, with the stellar hole filled in."""
    b = np.clip(np.asarray(bright, float), 0.0, None)
    if b.max() <= 0:
        return dict(img_flat_frac=None, img_edge_frac=None, image_degenerate=["empty"])
    n = b.shape[0]
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2) * pix
    field = (r <= support_mas) & ndimage.binary_fill_holes(b > 1e-6 * b.max())
    depth = ndimage.distance_transform_edt(field) * pix  # distance to the field's boundary
    use = field & (r >= hole_mas)
    bs = b[use]
    flat = float(np.mean(bs > 0.5 * bs.max()))
    edge = float(bs[depth[use] < 0.15 * depth.max()].sum() / bs.sum())
    why = ([f"flat ({flat:.2f} of the field above half the peak)"] if flat > DEGENERATE_FLAT_FRAC else []) + (
        [f"edge-piled ({edge:.2f} of the flux along the field's boundary)"] if edge > DEGENERATE_EDGE_FRAC else [])
    return dict(img_flat_frac=flat, img_edge_frac=edge, image_degenerate=why)


def cfg_of(args):
    cfg = TARGETS[args.target]
    if getattr(args, "wide_field", False):  # design §12: PS Gem at 15 mas, when its stock field was 5 (now the paper's, 19)
        cfg = {k: v for k, v in cfg.items() if k != "fov_mas"} | dict(half_mas=WIDE_FIELD_MAS, half_cmp_mas=WIDE_FIELD_MAS, mask_mas=WIDE_FIELD_MAS,
                                                                fov_override=2.0 * WIDE_FIELD_MAS / 0.95)
    if "fov_mas" not in cfg:
        setup(cfg)
    if "sec_index0" not in cfg and "img_star_index" not in cfg:
        # the paper's own stellar index per target (ORGANIC SIND1, -2.1 ... -3.2), fixed: the environment's index and flux fraction are only
        # defined relative to the stellar spectrum, so they compare with De Prins's only if the star is the same (IRAS 08544 keeps -4, Hillen)
        cfg["img_star_index"] = paper_indices(cfg)[0]
        cfg["star_index_source"] = "ORGANIC SIND1"
    if getattr(args, "bg_index", None) is not None:
        cfg = cfg | dict(bg_index_fixed=args.bg_index)
    if getattr(args, "hole_scale", None) is not None:
        cfg = cfg | dict(hole_scale=args.hole_scale)
    if getattr(args, "star_index", None) is not None:
        cfg = cfg | dict(img_star_index=args.star_index, star_index_source=f"one-off override ({args.star_index:g})")
    secondary_defaults(cfg, args.target)
    return cfg


# ----------------------------------------------------------------------------
# options of the post-AGB binary / PS Gem review (reports/postagb-binaries-psgem-review.md in virgil-paper): a point companion on top of
# gp-polar, a finer hole, a fixed polar geometry, and white scales set by hand. Each is off by default and leaves the default outputs unchanged.

COMP_FLUX0 = 0.02  # --companion: starting flux ratio to the primary (the HD 95767 / HR 4049 dots carry about 2 %)
COMP_RATIO_RANGE = (1e-3, 0.3)  # LogUniform prior of its flux ratio
COMP_INDEX_RANGE = (-5.0, 4.0)  # Uniform prior of its power-law index: photospheric (-3.8 to -2) and dust-like both allowed
COMP_FREE_RADIUS_MAS = 3.0  # --companion free: Uniform(-3, 3) mas on each offset about the primary
COMP_NEAR_MAS = 0.75  # image flux within this of the companion counts as the point's flux left in the image


def pair_arg(text):
    """argparse ``type`` of ``a,b``: two finite floats."""
    try:
        a, b = (float(x) for x in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected two numbers 'a,b', got {text!r}") from None
    if not (np.isfinite(a) and np.isfinite(b)):
        raise argparse.ArgumentTypeError(f"expected finite numbers, got {text!r}")
    return a, b


def check_options(args):
    """Refuse an option combination that would silently do nothing or the wrong thing (SystemExit with the reason)."""
    white, geo, comp = getattr(args, "white", None), getattr(args, "geometry", None), getattr(args, "companion", None)
    if white is not None:
        if min(white) <= 0:
            raise SystemExit("--white: the scales must be positive")
        if getattr(args, "noise", "current") not in ("vlti", "night"):
            raise SystemExit("--white replaces the pair-test white scales of --noise vlti or night; it does nothing under --noise " + str(args.noise))
    if geo is not None and not 0.0 <= geo[0] <= 90.0:
        raise SystemExit("--geometry: the inclination must lie in [0, 90] degrees")
    if getattr(args, "hole_scale", None) is not None and args.hole_scale <= 0:
        raise SystemExit("--hole-scale must be positive")
    if (comp is None) != (getattr(args, "companion_start", None) is None):
        raise SystemExit("--companion and --companion-start go together (the start, or the fixed position, is E,N in mas from the primary)")
    if getattr(args, "companion_box", None) is not None and (comp != "free" or args.companion_box <= 0):
        raise SystemExit("--companion-box is the positive half-width of the free companion's position box: it needs --companion free")
    if getattr(args, "min_length_r", None) is not None and args.min_length_r <= 0:
        raise SystemExit("--min-length-r must be positive (beams)")
    if (comp or geo is not None or getattr(args, "min_length_r", None) is not None) and getattr(args, "strategy", "gp-polar") != "gp-polar":
        raise SystemExit("--companion and --geometry are for --strategy gp-polar only")


def option_suffix(args):
    """The output-name suffix of the review options (empty when none is set, so default names are unchanged)."""
    sfx = ""
    if getattr(args, "companion", None):
        sfx += f"_comp{args.companion}"
    if getattr(args, "companion_box", None) is not None:
        sfx += f"_box{args.companion_box:g}"
    if getattr(args, "min_length_r", None) is not None:
        sfx += f"_minlr{args.min_length_r:g}"
    if getattr(args, "hole_scale", None) is not None:
        sfx += f"_holex{args.hole_scale:g}"
    if getattr(args, "geometry", None) is not None:
        sfx += f"_geo{args.geometry[0]:g}-{args.geometry[1]:g}"
    if getattr(args, "white", None) is not None:
        sfx += f"_white{args.white[0]:g}-{args.white[1]:g}"
    return sfx


def resolve_geometry(args, out, cfg, strat):
    """The (inc_deg, pa_deg, ...) the image stage deprojects with: ``--geometry`` when given, else the ``scale`` stage's fit for the gp
    strategies that need one, else a placeholder (the other strategies and ``--smoke``)."""
    over = getattr(args, "geometry", None)
    if over is not None:
        print(f"geometry fixed by --geometry: inc {over[0]:g}, PA {over[1]:g} deg", flush=True)
        return dict(inc_deg=float(over[0]), pa_deg=float(over[1]) % 180.0)
    geo_file = scale_file(out, cfg, args.noise)
    if not geo_file.exists() and not args.smoke and strat in ("gp-aniso", "gp-polar", "disc+res"):
        geo_file = scale_file(out, cfg, DEFAULT_NOISE)  # a noise model without its own geometry fit uses the default's
        print("geometry from", geo_file.name, flush=True)
    return json.load(open(geo_file))["geometry"] if strat in ("gp-aniso", "gp-polar", "disc+res") and not args.smoke else dict(
        inc_deg=45.0, pa_deg=90.0)


def companion_component(args):
    """The ``--companion`` point source: flux ratio to the primary and index free, at ``--companion-start`` (E, N mas from the primary)."""
    return PointSource(flux=PowerLaw(COMP_FLUX0, 0.0, L0), dra=float(args.companion_start[0]), ddec=float(args.companion_start[1]))


def companion_priors(args):
    """Priors of the companion: flux ratio and index always free; the position only for ``--companion free`` (Uniform over +-3 mas, a box
    around the primary)."""
    pri = {"comp.flux.ratio": dist.LogUniform(*COMP_RATIO_RANGE), "comp.flux.index": dist.Uniform(*COMP_INDEX_RANGE)}
    if args.companion == "free":
        box = getattr(args, "companion_box", None)
        if box:  # --companion-box: a box of half-width ``box`` about --companion-start, so the fit can say whether the source stays there
            (e0, n0) = args.companion_start
            pri |= {"comp.dra": dist.Uniform(e0 - box, e0 + box), "comp.ddec": dist.Uniform(n0 - box, n0 + box)}
        else:
            pri |= {"comp.dra": dist.Uniform(-COMP_FREE_RADIUS_MAS, COMP_FREE_RADIUS_MAS), "comp.ddec": dist.Uniform(-COMP_FREE_RADIUS_MAS, COMP_FREE_RADIUS_MAS)}
    return pri


def index_colour(d, env, d_sd):
    """How a secondary's colour reads (design §12): photospheric, dust-like, neither, or unconstrained when its posterior sd is too large."""
    if d_sd is None or d_sd > BINARY_INDEX_SD_MAX:
        return "unconstrained"
    if BINARY_STELLAR_INDEX[0] <= d <= BINARY_STELLAR_INDEX[1]:
        return "photospheric"
    return "dust-like" if min(abs(d), abs(d - env)) <= 1.0 else "neither"


def companion_summary(model, bright_grid, pix, f_env, sd, tot):
    """The companion's fitted terms for the JSON: position, flux, index and its reading, and the share of its flux left in the image
    within ``COMP_NEAR_MAS`` of it (rule 4 of the review's selection). ``bright_grid`` is the environment on the pixel grid (North up,
    East left, rotation 0); ``sd`` the Laplace sds of ``scene_evidence``; ``tot`` the total flux in units of the primary's."""
    c, sd = model.comp, sd or {}
    ratio, index = float(np.asarray(c.flux.ratio)), float(np.asarray(c.flux.index))
    e, n_ = float(c.dra), float(c.ddec)
    ny, nx = bright_grid.shape
    yy, xx = np.mgrid[:ny, :nx]
    east, north = ((nx - 1) / 2.0 - xx) * pix, ((ny - 1) / 2.0 - yy) * pix
    near = np.hypot(east - e, north - n_) <= COMP_NEAR_MAS
    left = f_env * float(bright_grid[near].sum() / bright_grid.sum()) if bright_grid.sum() > 0 else 0.0
    pos_sd = [sd.get("comp.dra"), sd.get("comp.ddec")]
    return dict(comp_dra=e, comp_ddec=n_, comp_sep_mas=float(np.hypot(e, n_)), comp_pa_deg=float(np.degrees(np.arctan2(e, n_)) % 360.0),
                comp_pos_sd_mas=None if None in pos_sd else float(np.hypot(*pos_sd)), comp_flux_ratio=ratio, comp_flux_pct=100 * ratio / tot,
                comp_flux_ratio_sd=sd.get("comp.flux.ratio"), comp_index=index, comp_index_sd=sd.get("comp.flux.index"),
                comp_colour=index_colour(index, float(np.asarray(model.env.flux.index)), sd.get("comp.flux.index")),
                comp_image_near_ratio=left, comp_image_left_frac=left / ratio if ratio > 0 else None, comp_near_mas=COMP_NEAR_MAS)


def companion_mirror(cfg, scene, start_env, support, priors, data, args, geo, r_min, m_max, cell, best):
    """``--companion free``: refit the chosen grid cell with the companion started at the mirror image of ``--companion-start`` through the
    primary (the closure phases cannot tell a point from its mirror image until the disc is in the model), and return ``(fit, ln Z_scene,
    record)`` of whichever of the two scene-evidence values is higher. Only the chosen cell is refitted, not the grid."""
    lr, lt, sg = cell
    start = polar_scene_fn(cfg, scene, start_env, support, args, geo, r_min, m_max)(sg, lr, lt)
    start = start.set("comp.dra", -float(args.companion_start[0])).set("comp.ddec", -float(args.companion_start[1]))
    pri = {k: v for k, v in priors.items() if not k.startswith("env.log_brightness")}
    f_, rec = fit_cell(start, image_priors(start) | pri, data, dict(max_steps=15) if args.smoke else dict(max_steps=POLAR_MAX_STEPS))
    lz_m, lz_s = scene_evidence(f_, data, pri)[0], scene_evidence(best, data, pri)[0]
    rec = dict(comp_mirror_log_z_scene=lz_m, comp_start_log_z_scene=lz_s, comp_mirror_dra=float(f_.model.comp.dra), comp_mirror_ddec=float(f_.model.comp.ddec),
               comp_mirror_chi2=float(f_.info["chi2_red"]), comp_mirror_better=bool(lz_m > lz_s), comp_mirror_converged=bool(f_.info.get("converged", True)))
    print(f"companion mirror start: ln Z_scene {lz_m:.1f} against {lz_s:.1f} from the start; kept the {'mirror' if lz_m > lz_s else 'start'}", flush=True)
    return (f_ if lz_m > lz_s else best), rec


def setup(cfg):
    """Fill grid, SPARCO secondary and plot extents from the published FITS header."""
    hdr, im, _ = zenodo_header(cfg)
    n, scale = im.shape[0], abs(hdr["CDELT1"])
    yy, xx = np.mgrid[:n, :n]
    rad = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2) * scale
    o = np.argsort(rad.ravel())
    r90 = float(rad.ravel()[o][np.searchsorted(np.cumsum(im.ravel()[o]) / im.sum(), 0.9)])
    scale = cfg.get("pixel_override") or scale  # per-target pixel (HR 4049: finer than the paper's 0.635 mas)
    fov = cfg.get("fov_override") or n * scale  # per-target field override (HR 4049 larger); grid_setup pads the start image to it
    half = cfg.get("half_mas") or float(np.clip(1.5 * r90, 8.0, 0.95 * fov / 2))  # per-target override: the paper's figure extent
    has_sec = hdr["SFLU2"] > 0
    cfg.update(
        fov_mas=fov, pixel_mas=scale, sec0=(hdr["SDEX2"], hdr["SDEY2"]) if has_sec else None,
        ud_fix=hdr["SUD1"],  # the paper's fixed stellar diameter
        ud0=max(hdr["SUD1"], 0.3), half_mas=half, mask_mas=cfg.get("mask_mas") or 0.95 * fov / 2, half_cmp_mas=cfg.get("half_cmp_mas") or 0.85 * half,
        regions=dict(disc=(0.0, cfg.get("mask_mas") or 0.95 * fov / 2)), inner_rim_max_mas=None, paper_header=dict(
            ud=hdr["SUD1"], f_prim_pct=100 * hdr["SFLU1"], f_sec_pct=100 * hdr["SFLU2"],
            sec=(hdr["SDEX2"], hdr["SDEY2"]), image_index=hdr["SIND0"], star_index=hdr["SIND1"], sec_index=hdr["SIND2"]),
    )


def load_files(cfg, n=None):
    # the INSPIRING release holds exactly the exposures of the paper's observation log (counts checked per target)
    pats = cfg["files"] if isinstance(cfg["files"], (list, tuple)) else [cfg["files"]]
    pats = [DATA / cfg.get("data_key", cfg["key"]) / "INSPIRING/*.fits" if pat is None else pat for pat in pats]
    files = sorted(f for pat in pats for f in glob.glob(str(pat)))
    if not files:
        raise SystemExit(f"no files for {cfg['key']}; run scripts/postagb_data.py fetch")
    return files[:n] if n else files


def load(cfg, n=None, bandwidth=False):
    files = load_files(cfg, n)
    return load_pionier(files, bandwidth=bandwidth), files


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


def rows_of(data):
    """Indices into the V^2 arrays (``data.vis``) of each (frame, station pair), channels in wavelength order: an (n_group, n_channel) array.

    The per-sample arrays (frame, stations, wavel) hold every V^2 sample, flagged ones included, and ``vis_index`` (None if nothing is
    flagged) picks those that are kept; a (frame, baseline) that lost a channel to flagging has fewer than the usual number and is left out."""
    sel = slice(None) if data.vis_index is None else np.asarray(data.vis_index)  # None: nothing flagged
    frame, st, wl = np.asarray(data.frame)[sel], np.asarray(data.stations)[sel], np.asarray(data.wavel)[sel]
    key = {}
    for i, (f, s_, w) in enumerate(zip(frame, st, wl)):
        key.setdefault((int(f), int(s_[0]), int(s_[1])), []).append((float(w), i))
    groups = [[i for _, i in sorted(v)] for v in key.values()]
    n_ch = max(len(g) for g in groups)
    kept = [g for g in groups if len(g) == n_ch]
    if len(kept) < len(groups):
        print(f"rows_of: {len(groups) - len(kept)} of {len(groups)} (frame, baseline) groups dropped for having fewer than {n_ch} channels", flush=True)
    return np.array(kept)


def residual_stats(data, model_v2, model_t3):
    """What a fit leaves behind, on the *quoted* errors of ``data`` (as read, before any rescaling or gains): chi2 per point of the V^2
    and of the closure phases, the common-mode ratio of the V^2 residuals and the adjacent-channel correlation.

    The six channels of one (frame, baseline) share calibration gains, so the per-(frame, baseline) channel mean m_g of the
    standardised residual z = residual / quoted error has variance ``n_channels * mean(m_g^2) / mean(z^2)`` times what independent
    errors give (ratio 1 for independent errors; its inverse share of the total variance is mean(m_g^2) / mean(z^2) = ratio / n_channels,
    17 % for six channels); the correlation is that of adjacent channels' z within a (frame, baseline), pooled."""
    z = (np.asarray(data.vis) - np.asarray(model_v2)) / np.asarray(data.d_vis)
    dt = np.degrees(np.asarray(data.phi)) - np.asarray(model_t3)
    zt = (((dt + 180.0) % 360.0) - 180.0) / np.degrees(np.asarray(data.d_phi))
    zg = z[rows_of(data)]  # (n_group, n_channel)
    n_ch = zg.shape[1]
    m = zg.mean(axis=1)
    return dict(chi2_v2=float(np.mean(z**2)), chi2_t3=float(np.mean(zt**2)), n_v2=int(z.size), n_t3=int(zt.size),
                common_mode_v2=float(n_ch * np.mean(m**2) / np.mean(z**2)), white_expectation=1.0 / n_ch,
                adjacent_corr_v2=float(np.corrcoef(zg[:, :-1].ravel(), zg[:, 1:].ravel())[0, 1]))


def map_gain_stats(data, data_g, model_v2, model_t3):
    """Residual statistics after the MAP gains of the fitted error model are subtracted (``--noise vlti``).

    ``data_g`` is ``data`` under the vlti error model, so its V^2 errors e are the scaled ones and its gains cover the (frame, telescope)
    and (frame, baseline) modes. In units of e the residual x has covariance ``I + L L^T`` per block of linked modes (L the modes' columns,
    ``2 V^2 / e`` times their shapes times their widths); the conditional (MAP) gain estimate is ``L L^T C^-1 x`` and what is left is
    ``C^-1 x``. The statistics are those of ``residual_stats`` on the quoted errors with that gain removed from the model, plus
    ``chi2_v2_own`` = x^T C^-1 x per point: chi2 under the fitted covariance (gains marginalised), which is the fit's own V^2 chi2."""
    g = data_g.gains
    assert g.spanning is None, "modes that span frames are not used by the vlti error model"
    e = np.asarray(data_g.d_vis, float)
    x = (np.asarray(data.vis, float) - np.asarray(model_v2, float)) / e
    jw = 2.0 * np.asarray(model_v2, float) / e
    widths = np.asarray(g.widths_for(None), float)
    rows, shapes, grp = np.asarray(g.rows), np.asarray(g.shapes, float), np.asarray(g.group)
    xp = x.copy()
    for b in range(rows.shape[0]):
        ok = rows[b] < e.size
        idx = rows[b][ok]
        if idx.size == 0:
            continue
        local = shapes[b][ok] * jw[idx][:, None] * widths[grp[b]][None, :]
        xp[idx] = np.linalg.solve(np.eye(idx.size) + local @ local.T, x[idx])
    stats = residual_stats(data, np.asarray(model_v2, float) + e * (x - xp), model_t3)
    stats["chi2_v2_own"] = float(np.mean(x * xp))
    return stats


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


def parametric_start(cfg, inc=50.0, pa=150.0, start=None):
    """The PMOIRED geometry of De Prins et al. (2026, Sect. 4.2) in virgil components.

    ``start`` overrides ``rim0``, ``az_amps0``, ``az_pas0`` and ``sec0`` (see ``param_starts``).
    """
    cfg = cfg | (start or {})
    sec = cfg["sec0"]
    parts = dict(
        primary=UniformDisk(diam=cfg["ud0"], flux=PowerLaw(1.0, cfg.get("star_index", -3.165), L0)),
        rim=ModulatedGaussianRim(*cfg.get("rim0", (12.0, 4.0, inc, pa)), az_amps=cfg.get("az_amps0", (0.3,)),
                                 az_pas=cfg.get("az_pas0", (0.0,)), flux=PowerLaw(0.4, 0.9, L0)),
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
    priors = {k: v for k, v in PARAM_PRIORS.items()
              if (k != "primary.diam" or cfg.get("fit_ud", True)) and (cfg["sec0"] is not None or not k.startswith("secondary."))}
    # blind multi-start (a single start can stick face-on: IRAS 15469-5311); keep the lowest chi2 and report them all
    starts = cfg.get("param_starts") or [dict(inc=i, pa=p_) for i, p_ in
                                          [(50.0, 150.0), (25.0, 30.0), (25.0, 120.0), (60.0, 30.0), (60.0, 100.0), (40.0, 170.0)]]
    if args.smoke:
        starts = starts[:1]
    tried = []
    for st in starts:
        kw = dict(st) if "inc" in st else dict(start=st)
        r = fit(parametric_start(cfg, **kw), priors, data, method="lbfgs", **({"max_steps": 25} if args.smoke else {}))
        tried.append((chi2r(r.model, data)[0], r))
        print("start", st, "chi2/pt", tried[-1][0])
    result = min(tried, key=lambda t: t[0])[1]
    chi2_starts = [float(c) for c, _ in tried]
    model = result.model
    c2, n = chi2r(model, data)
    v = {k: np.asarray(x).tolist() for k, x in result.values.items()}
    has_sec = cfg["sec0"] is not None
    ratio = lambda name: float(np.asarray(getattr(model, name).flux(L0))) if name != "secondary" or has_sec else 0.0  # noqa: E731
    total = 1.0 + ratio("secondary") + ratio("rim") + ratio("bg")
    summary = dict(
        values=v, chi2_per_point=c2, n_points=n, chi2_starts=chi2_starts, start_chi2_per_point=chi2_starts,
        seconds=time.time() - t0, smoke=args.smoke,
        f_prim_pct=100.0 / total, f_sec_pct=100 * ratio("secondary") / total,
        f_rim_pct=100 * ratio("rim") / total, f_bg_pct=100 * ratio("bg") / total,
        ud_prim_mas=float(model.primary.diam), rim_diam_mas=float(model.rim.diam),
        rim_fwhm_mas=float(model.rim.fwhm), rim_inc_deg=float(model.rim.inc),
        rim_pa_deg=float(model.rim.pa) % 180.0, rim_index=float(np.asarray(model.rim.flux.index)),
        az_amp=float(np.ravel(model.rim.az_amps)[0]), az_pa_deg=float(np.ravel(model.rim.az_pas)[0]) % 360.0,
        az_amps=[float(a) for a in np.ravel(model.rim.az_amps)],
        az_pas_deg=[float(a) % 360.0 for a in np.ravel(model.rim.az_pas)],
        sec_dra=float(model.secondary.dra) if has_sec else None,
        sec_ddec=float(model.secondary.ddec) if has_sec else None,
        sec_sep_mas=float(np.hypot(model.secondary.dra, model.secondary.ddec)) if has_sec else None,
        sec_pa_deg=float(np.degrees(np.arctan2(model.secondary.dra, model.secondary.ddec)) % 360.0) if has_sec else None,
        rim_dra=float(model.rim.dra), rim_ddec=float(model.rim.ddec),
    )
    print(json.dumps(summary, indent=1))
    json.dump(summary, open(path(out, cfg, "parametric.json"), "w"), indent=1)


# ----------------------------------------------------------------------------
# image


def sparco_scene(cfg, env, f_env=0.5, index=None, f_sec=None, extra=None):
    """UD primary (diameter fixed, flux ~ lambda^-4) + point secondary + environment + grey over-resolved background (SPARCO).

    Every strategy uses this one scene; only the environment component differs. ``extra`` adds components (the disc strategy's
    residual field). A target with ``sec0=None`` or ``secondary=False`` has no secondary.
    """
    sec = cfg["sec0"] if cfg.get("secondary", True) else None
    index = cfg.get("env_index0", 0.9) if index is None else index
    parts = dict(
        star=UniformDisk(diam=cfg["ud_fix"], flux=PowerLaw(1.0, cfg.get("img_star_index", -4.0), L0)),
        env=env.set("flux", PowerLaw(f_env, index, L0)) if hasattr(env, "set") else env,
    )
    if cfg.get("bg"):  # grey over-resolved background, free flux fraction and index
        parts["bg"] = Resolved(flux=PowerLaw(cfg.get("f_bg0", 0.1), cfg.get("bg_index_fixed") or 0.0, L0))
    if sec is not None:
        parts["sec"] = PointSource(flux=PowerLaw(cfg.get("f_sec0", 0.03) if f_sec is None else f_sec, cfg.get("sec_index_fixed", cfg.get("sec_index0", 0.0)), L0),
                                   dra=sec[0], ddec=sec[1])
    return System(**parts, **(extra or {}))


def hole_mask(cfg, env, data, rot_deg=0.0):
    """Pixels of the environment image allowed to carry flux: the image's own support minus a window around
    each central star. Extended flux within a fraction of a beam of a star is nearly indistinguishable from the
    star's own, so without the window the pixels and the UD / point-source fluxes are degenerate.

    The radius is the larger of lambda_min / (2 B_max) (one resolution element, the comparison beam's FWHM) and the
    starting UD diameter ``ud0`` (the diameter's number used as a radius), unless ``hole_mas`` sets it. The image is also
    limited to a disc of radius ``mask_mas`` (half the field by default), so that the corners of the square grid, which
    the data barely constrain, carry no flux. The primary is at the origin,
    the window follows ``sec0`` too when the secondary is a point source of the scene (``secondary`` true), so that
    only a secondary left in the image (``secondary=False``) is not masked.
    """
    radius = (cfg.get("hole_mas") or max(compare_beam(data).minor_mas, cfg["ud_fix"])) * cfg.get("hole_scale", 1.0)  # --hole-scale
    ny, nx = np.shape(env.log_brightness)
    scale = float(env.pixel_scale_mas)
    mask = np.ones((ny, nx), bool) if env.support is None else np.asarray(env.support, bool).copy()
    yy, xx = np.mgrid[:ny, :nx]
    left, up = ((nx - 1) / 2.0 - xx) * scale, ((ny - 1) / 2.0 - yy) * scale  # row 0 up, column 0 left of the grid
    th = np.radians(rot_deg)  # the grid's up axis points at position angle ``rot_deg`` (North through East)
    east, north = left * np.cos(th) + up * np.sin(th), up * np.cos(th) - left * np.sin(th)
    mask &= (np.hypot(east, north) > radius) & (np.hypot(left, up) <= cfg.get("mask_mas", nx * scale / 2))
    # no window around a secondary that lies inside the primary's own hole: the two would merge into one blanked region
    if (cfg.get("sec0") is not None and cfg.get("secondary", True) and cfg.get("sec_hole", True)
            and np.hypot(*cfg["sec0"]) >= radius):
        mask &= np.hypot(east - cfg["sec0"][0], north - cfg["sec0"][1]) > radius
    return mask, radius


STRATEGIES = ("mem", "gp-iso", "gp-aniso", "gp-polar", "disc+res")
HEADLINE = "gp-polar"
GP_FRACS = (0.25, 0.5, 1.0, 2.0)  # GP lengths in units of the comparison beam's FWHM, lambda_min / (2 B_max)


CHROM_DALPHA_MAX = 4.0  # --chrom: the ring's index may change by up to this across +-w0 (twice the chrom arm's CAL_CHROM_DALPHA)
CHROM_STAR_WINDOW = 2.0  # --chrom: the star's index is free within SIND1 +- this (twice the chrom arm's offset)


def chromatic_ring(cfg, geo, unit):
    """``--chrom``'s extra component (design §11): a Gaussian ring with no harmonics whose spectral index varies linearly with
    radius, inclination and position angle fixed at the scale stage's ``geo``, radius and width started there, and its priors (with
    a free star index). Returns ``(rim, priors)``; the grey image keeps everything else."""
    r0 = float(np.clip(geo.get("radius_mas", 2.0), 0.5, float(cfg["half_mas"])))
    w0 = float(np.clip(geo.get("width_mas", 0.4 * r0), 0.1, 12.0))
    rim = MultipoleDisk(GaussianRing(r0, w0), inc=float(geo["inc_deg"]), pa=float(geo["pa_deg"]), n_r=48, flux=0.1,
                        spectral_index=RadialLinear(float(cfg.get("env_index0", 0.9)), 0.0, r0), wavel0=L0)
    smax = CHROM_DALPHA_MAX / (2.0 * max(w0, 0.5 * unit))
    s0 = float(cfg.get("img_star_index", -4.0))
    priors = {"rim.flux": dist.LogUniform(1e-3, 5.0), "rim.profile.radius": dist.LogUniform(0.5, float(cfg["half_mas"])),
              "rim.profile.sigma": dist.LogUniform(0.1, 12.0), "rim.spectral_index.value": dist.Uniform(-6.0, 6.0),
              "rim.spectral_index.slope": dist.Uniform(-smax, smax),
              "star.flux.index": dist.Uniform(s0 - CHROM_STAR_WINDOW, s0 + CHROM_STAR_WINDOW)}
    return rim, priors


def chromatic_summary(model, cfg):
    """The fitted chromatic terms of a ``--chrom`` scene, and whether the fit uses them (rule 3 of design §11)."""
    w = float(np.asarray(model.rim.profile.sigma))
    d_alpha = -2.0 * float(np.asarray(model.rim.spectral_index.slope)) * w
    f_rim = float(np.asarray(model.rim.flux))
    star = float(np.asarray(model.star.flux.index))
    s0 = float(cfg.get("img_star_index", -4.0))
    tot = 1.0 + f_rim + sum(float(np.asarray(getattr(model, k).flux.ratio)) for k in ("env", "sec", "bg") if hasattr(model, k))
    used = bool((f_rim / tot >= 0.01 and abs(d_alpha) >= 0.5) or abs(star - s0) >= 0.5)
    return dict(chrom_rim_flux_pct=100.0 * f_rim / tot, chrom_rim_radius_mas=float(np.asarray(model.rim.profile.radius)), chrom_rim_width_mas=w,
                chrom_rim_index=float(np.asarray(model.rim.spectral_index.value)), chrom_rim_d_alpha=d_alpha, chrom_star_index=star,
                chrom_star_shift=star - s0, chrom_used=used)


BINARY_MODES = ("paper", "free", "motion", "epoch")
BINARY_POS_SIGMA_MAS = 1.0  # --binary free: the secondary's position prior about sec0 (design §12)
BINARY_V_SIGMA = 0.02  # mas/day per axis: twice the fastest companion expected, 2 pi sep / P = 0.011 (AI Sco)
BINARY_OFFSET_SIGMA_MAS = 0.2  # per-night position offsets: about the largest orbital displacement from the mid-time (AI Sco +-0.18 mas)
MAS2RAD = np.pi / 180.0 / 3600.0 / 1000.0


class MovingComponent(Component):
    """A star whose sky position changes with time against the disc (design §12): ``(dra, ddec)`` at the data's mid-time, plus
    a linear motion ``(vra, vdec)`` in mas/day and an offset per night ``(off_ra, off_dec)``. ``shape`` gives the visibility
    (its own flux and offset are not used). ``dt`` (days from the mid-time) and ``group`` (night index) are per sample of the
    data's ``u``, so the component evaluates only on that data's (u, v): any other sampling raises."""

    shape: Component
    vra: jax.Array
    vdec: jax.Array
    off_ra: jax.Array
    off_dec: jax.Array
    dt: tuple = eqx.field(static=True)
    group: tuple = eqx.field(static=True)

    def __init__(self, shape, dt, group, flux=1.0, dra=0.0, ddec=0.0, vra=0.0, vdec=0.0, off_ra=None, off_dec=None):
        self._place(flux, dra, ddec)
        self.shape = shape
        self.dt, self.group = tuple(float(x) for x in dt), tuple(int(g) for g in group)
        k = max(self.group) + 1
        self.vra, self.vdec = np.asarray(vra, float), np.asarray(vdec, float)
        self.off_ra = np.zeros(k) if off_ra is None else np.asarray(off_ra, float)
        self.off_dec = np.zeros(k) if off_dec is None else np.asarray(off_dec, float)

    def positions(self):
        """``(dra, ddec)`` per sample, in mas."""
        dt, g = jnp.asarray(self.dt), jnp.asarray(self.group)
        return (self.dra + self.vra * dt + jnp.asarray(self.off_ra)[g], self.ddec + self.vdec * dt + jnp.asarray(self.off_dec)[g])

    def model(self, u, v, wavel):
        if np.shape(u) != (len(self.dt),):
            raise ValueError(f"MovingComponent is labelled for {len(self.dt)} samples, evaluated on {np.shape(u)}: it only "
                             "evaluates on the data it was built for")
        uu, vv = u / wavel, v / wavel
        ra, dec = self.positions()
        arg = 2.0 * jnp.pi * MAS2RAD * (uu * ra + vv * dec)
        return self.shape._centred_cvis(uu, vv) * jax.lax.complex(jnp.cos(arg), -jnp.sin(arg))

    def _centred_cvis(self, uu, vv):
        return self.shape._centred_cvis(uu, vv)

    def _centred_image(self, xx, yy, pixel_scale_mas):
        return self.shape._centred_image(xx, yy, pixel_scale_mas)

    def _shape_checks(self):
        return self.shape._shape_checks()


def binary_scene(scene, cfg, data, mode):
    """``scene`` with the binary of ``--binary mode`` (design §12) and the priors to add. ``paper`` changes nothing. ``free``
    frees the secondary's position (Normal about sec0), flux and index. ``motion`` and ``epoch`` also move both stars against the
    disc, linearly or by an offset per night (a star absent from the scene is skipped). The primary's index stays at SIND1."""
    if mode == "paper":
        return scene, {}
    has_sec = hasattr(scene, "sec")
    pri = {}
    if has_sec:
        s0 = cfg.get("sec0") or (float(scene.sec.dra), float(scene.sec.ddec))
        pri |= {"sec.dra": dist.Normal(float(s0[0]), BINARY_POS_SIGMA_MAS), "sec.ddec": dist.Normal(float(s0[1]), BINARY_POS_SIGMA_MAS),
                "sec.flux.ratio": dist.LogUniform(1e-3, 1.0), "sec.flux.index": dist.Uniform(-5.0, 2.0)}
    if mode == "free":
        return scene, pri
    mjd = float(data.t_ref) + np.asarray(data.dt, float)
    dt = mjd - 0.5 * (mjd.min() + mjd.max())
    group = np.unique(night_labels(data)["u_night"], return_inverse=True)[1].ravel()
    k = int(group.max()) + 1
    for name in ("star", "sec") if has_sec else ("star",):
        c = getattr(scene, name)
        scene = scene.set(name, MovingComponent(c, dt, group, flux=c.flux, dra=float(c.dra), ddec=float(c.ddec)))
        if mode == "motion":
            pri |= {f"{name}.vra": dist.Normal(0.0, BINARY_V_SIGMA), f"{name}.vdec": dist.Normal(0.0, BINARY_V_SIGMA)}
        else:
            pri |= {f"{name}.off_ra": dist.Normal(jnp.zeros(k), BINARY_OFFSET_SIGMA_MAS), f"{name}.off_dec": dist.Normal(jnp.zeros(k), BINARY_OFFSET_SIGMA_MAS)}
    return scene, pri


BINARY_STELLAR_INDEX = (-3.8, -2.0)  # a stellar companion at 1.65 um: F_lambda index -2.5 (cool) to -3.3 (hot), +- 0.5
BINARY_INDEX_SD_MAX = 0.5  # a secondary index with a larger posterior sd is not read at all


def binary_summary(model, mode, cfg, sd=None, data=None):
    """The fitted binary terms of a ``--binary`` scene, and how the secondary's colour reads (design §12). ``sd`` holds the
    Laplace posterior sds of ``scene_evidence``; the colour is read against absolute windows (a stellar companion's index does not
    depend on the primary's SIND1), and only when the index is constrained."""
    out, sd = dict(binary=mode), sd or {}
    if hasattr(model, "sec"):
        d = float(np.asarray(model.sec.flux.index))
        env = float(np.asarray(model.env.flux.index))
        d_sd = sd.get("sec.flux.index")
        if mode == "paper":
            colour = "fixed"
        else:
            colour = index_colour(d, env, d_sd)
        out |= dict(bin_sec_index=d, bin_sec_index_sd=d_sd, bin_sec_dra=float(model.sec.dra), bin_sec_ddec=float(model.sec.ddec),
                    bin_sec_colour=colour)
    for name in ("star", "sec"):
        c = getattr(model, name, None)
        if isinstance(c, MovingComponent):
            out |= {f"bin_{name}_v_mas_per_day": [float(c.vra), float(c.vdec)],
                    f"bin_{name}_v_sd": [sd.get(f"{name}.vra"), sd.get(f"{name}.vdec")],
                    f"bin_{name}_offsets_mas": np.stack([np.asarray(c.off_ra), np.asarray(c.off_dec)], 1).round(4).tolist(),
                    f"bin_{name}_offsets_sd": [[sd.get(f"{name}.off_ra[{i}]"), sd.get(f"{name}.off_dec[{i}]")] for i in range(np.size(c.off_ra))]}
    if isinstance(getattr(model, "sec", None), MovingComponent) and isinstance(model.star, MovingComponent):
        out |= dict(bin_rel_v_mas_per_day=[float(model.sec.vra - model.star.vra), float(model.sec.vdec - model.star.vdec)],
                    bin_rel_v_sd=[sd.get("sec.vra-star.vra"), sd.get("sec.vdec-star.vdec")])
    if data is not None:  # the night of each offset, so that offsets can be regressed on time
        mjd = float(data.t_ref) + np.asarray(data.dt, float)
        night = night_labels(data)["u_night"]
        out |= dict(bin_night_mjd=[round(float(mjd[night == n].mean()), 3) for n in np.unique(night)], bin_mid_mjd=float(0.5 * (mjd.min() + mjd.max())))
    return out


def scene_evidence(fit_result, data, priors):
    """Laplace evidence with the scene parameters integrated as well as the field latents, and their posterior sds.

    ``log_evidence(fit, data)`` (the grid's selector) conditions on every non-latent parameter at its MAP, so it cannot compare
    scenes with different numbers of parameters: a per-night offset lowers chi2 and pays nothing. Here every path of ``priors``
    that is not a latent is integrated too: ln Z = -chi2/2 - L - |z|^2/2 + sum_e [ln p(theta_e) + ln(2 pi)/2] - ln det(J^T J + Lambda)/2,
    with Lambda = 1 for the latents, 1/sigma^2 for a Normal prior and 0 for the flat ones. (virgil's ``log_evidence(priors=)``
    leaves Lambda out for the scene parameters, which overstates the evidence of parameters the prior, not the data, constrains.)
    Returns ``(ln Z, {path or path[i]: sd})``; the sds include ``sec.vra-star.vra`` and ``sec.vdec-star.vdec`` when both stars move."""
    model = fit_result.model
    lat = tuple(_evidence_latents(model, "env", "scene_evidence"))
    extra = tuple(p for p in priors if p not in lat and not p.endswith("_vec") and not p.startswith("env.log_brightness"))
    r, jac = _residual_jacobian(model, data, lat + extra, on_device=True)
    r, jac = np.asarray(r, float), np.asarray(jac, float)
    n_z = sum(int(np.size(model.get(p))) for p in lat)
    lam, names, log_p = [np.ones(n_z)], [], 0.0
    for p in extra:
        val = np.asarray(model.get(p), float)
        pr = priors[p]
        normal = type(pr).__name__ == "Normal"
        lam.append(np.broadcast_to(1.0 / np.asarray(pr.scale, float) ** 2, val.shape).ravel() if normal else np.zeros(val.size))
        names += [p] if val.ndim == 0 else [f"{p}[{i}]" for i in range(val.size)]
        log_p += float(np.sum(np.asarray(pr.log_prob(val), float))) + 0.5 * val.size * np.log(2.0 * np.pi)
    lam = np.concatenate(lam)
    if jac.shape[1] != lam.size:
        raise ValueError(f"scene_evidence: the Jacobian has {jac.shape[1]} columns for {lam.size} parameters")
    hess = jac.T @ jac + np.diag(lam)
    sign, logdet = np.linalg.slogdet(hess)
    if sign <= 0:
        return float("-inf"), {}
    z2 = sum(float(np.sum(np.asarray(model.get(p), float) ** 2)) for p in lat)
    cov = np.linalg.inv(hess)[n_z:, n_z:]
    sd = {k: float(v) for k, v in zip(names, np.sqrt(np.clip(np.diag(cov), 0.0, None)))}
    for ax in ("vra", "vdec"):  # the relative (orbital) velocity is far better measured than either star's against the disc
        if f"sec.{ax}" in names and f"star.{ax}" in names:
            i, j = names.index(f"sec.{ax}"), names.index(f"star.{ax}")
            sd[f"sec.{ax}-star.{ax}"] = float(np.sqrt(max(cov[i, i] + cov[j, j] - 2.0 * cov[i, j], 0.0)))
    return float(-0.5 * (r @ r) - _log_norm(model, data) - 0.5 * z2 + log_p - 0.5 * logdet), sd


def scene_priors(cfg, scene, env_fixed_geometry=False):
    """Priors shared by every strategy: the SPARCO fluxes and indices of the environment, the grey background and the secondary."""
    priors = {"env.flux.ratio": dist.LogUniform(0.01, 20.0), "env.flux.index": dist.Uniform(-6.0, 6.0)}
    if cfg.get("bg"):
        priors |= {"bg.flux.ratio": dist.Uniform(0.0, 2.0)}
        if cfg.get("bg_index_fixed") is None:  # the index is unconstrained (a few per cent of the flux): one-off ``--bg-index`` fixes it
            priors["bg.flux.index"] = dist.Uniform(-6.0, 6.0)
    if cfg["sec0"] is not None and cfg.get("secondary", True):
        priors |= {
            "sec.flux.ratio": dist.LogUniform(1e-3, 1.0),  # was Uniform(0, 0.3), which HD95767 (paper: 0.43) pinned and IW Car mem soaked up
            "sec.dra": dist.Normal(cfg["sec0"][0], 1.0),
            "sec.ddec": dist.Normal(cfg["sec0"][1], 1.0),
        }
        if "sec_prior" in cfg:  # the PMOIRED fit as an informative prior (see ``secondary_defaults``); the index is fixed, not fitted
            sp = cfg["sec_prior"]
            priors |= {"sec.dra": dist.Normal(sp["dra"], sp["pos_sigma"]), "sec.ddec": dist.Normal(sp["ddec"], sp["pos_sigma"])}
            if "ratio" in sp:
                priors["sec.flux.ratio"] = dist.TruncatedNormal(sp["ratio"], sp["ratio_sigma"], low=0.0)
            if cfg.get("sec_index_fixed") is None:
                priors["sec.flux.index"] = dist.Uniform(-5.0, 2.0)
        elif "sec_index0" in cfg:  # free power-law index of the secondary (IRAS08544: published black body of 4000 K)
            priors["sec.flux.index"] = dist.Uniform(-5.0, 2.0)  # a star is not steeper than the primary (-4) by much, nor rising faster than lambda^2
    return priors


def grid_setup(cfg, data, args, rot=0.0):
    """The paper's pixel grid, the window around the star(s), and the Gaussian start image."""
    # the paper's pixel, or the finer of that and a quarter of the comparison beam (lambda_min / 2 B_max), so that the beam spans four pixels or more
    scale = min(cfg["pixel_mas"], compare_beam(data).minor_mas / 4.0) * (4.0 if args.smoke else 1.0)
    fov = cfg["fov_mas"] * (0.5 if args.smoke else 1.0)
    start = starting_image(data, star=True, oversample=nyquist_pixel_scale(data) / scale, largest_mas=fov)
    # starting_image never exceeds the interferometric field of view (lambda / B_min), whatever ``largest_mas`` says, so a larger
    # requested field (HR 4049: 36 mas against 51 mas) is padded here with the start image's own floor, keeping the star at the centre.
    n0, n_want = start.env.log_brightness.shape[0], int(np.ceil(fov / scale))
    n_want += (n_want - n0) % 2
    if n_want > n0:
        b0 = np.exp(np.asarray(start.env.log_brightness))
        pad = (n_want - n0) // 2
        env = Image.from_brightness(np.pad(b0, pad, constant_values=0.0), start.env.pixel_scale_mas, floor=1e-3, flux=start.env.flux)
        start = System(star=start.star, env=env)
    half = start.env.log_brightness.shape[0] * float(start.env.pixel_scale_mas) / 2
    if not args.smoke and half < cfg.get("mask_mas", half) - float(start.env.pixel_scale_mas):  # smoke halves the field on purpose; the saved field and support radii are config values, so check the grid itself
        raise SystemExit(f"{cfg['key']}: the grid reaches only +-{half:.1f} mas but the support radius is {cfg['mask_mas']:.1f} mas; raise fov_override")
    support, hole_mas = hole_mask(cfg, start.env, data, rot)
    return start, support, hole_mas


DEFAULT_NOISE, DEFAULT_MEAN = "current", "blob"  # what a run without --noise / --mean uses; outputs of anything else carry the option in their names


def noise_tag(noise):
    return "" if noise == DEFAULT_NOISE else f"_{noise}"


def scale_file(out, cfg, noise=DEFAULT_NOISE):
    return out / f"{cfg.get('data_key', cfg['key'])}_scale{noise_tag(noise)}.json"


# The noise models live in postagb_noise.py (virgil-vlti): ``--noise vlti`` is the pair-test error model of design/pionier_systematics.md
# (the MacKay ``error_scale`` is skipped: virgil refuses it with gains), ``--noise night`` adds a gain per (night, baseline) and a
# closure-phase offset per (night, triangle) at the measured widths of postagb_night_widths.json, ``--noise nightw`` keeps the MacKay
# white errors and adds night terms of ``NIGHTW`` (3 % and 2 deg, the 18 Nov review of design section 12).


def effective_noise_widths(cfg, white=None):
    """The pair-test widths of this target, with ``--white``'s (V^2, closure) white scales in place of the pair test's when given."""
    w = noise_widths(cfg)
    return w if white is None else w | dict(v2_white=float(white[0]), t3_white=float(white[1]))


def modelled_noise(data, cfg, noise, white=None):
    """``data`` under a correlated error model: ``vlti`` or ``night``. ``white`` = ``--white``: (V^2, closure) white scales that replace
    the pair test's (everything else, the frame gains and the night terms, stays at the measured widths)."""
    widths = effective_noise_widths(cfg, white)
    return night_noise(data, widths, night_widths(cfg)) if noise == "night" else vlti_noise(data, widths)


def load_scaled(cfg, args, out):
    """The target's data under the chosen noise model: ``--noise current`` rescales the errors by the MacKay factor stored by the
    ``scale`` stage (1 if absent); ``--noise vlti`` applies ``vlti_noise`` (no bandwidth smearing, which virgil does not support for pixel Images; factor reported as 1)."""
    if getattr(args, "noise", "current") in ("vlti", "night"):
        data, files = load(cfg, args.files)  # no bandwidth smearing: virgil does not support it for pixel Images
        return modelled_noise(data, cfg, args.noise, getattr(args, "white", None)), data, 1.0, files
    data, files = load(cfg, args.files)
    f = scale_file(out, cfg, "current")
    s = json.load(open(f))["error_scale"] if f.exists() and not args.smoke else 1.0
    if getattr(args, "noise", "current") == "nightw":  # MacKay white errors, then the night terms on top
        return pionier_noise(data, white=s, night=NIGHTW), data, s, files
    return data.with_error_scale(s) if s != 1.0 else data, data, s, files


def pilot_error_scale(scene, start, support, priors, data, args):
    """MacKay's error scale from an isotropic GP pilot at the middle of the (sigma, length) grid, on ``data`` with its errors as
    given (``error_scale`` needs a GaussianField). Returns ``(s, the pilot's chi2/pt on those errors)``."""
    unit = compare_beam(data).minor_mas
    mean = np.asarray(start.env.brightness)
    pilot_sc = scene.set("env", Image(GaussianField(np.zeros(mean.shape), 1.0, unit, mean=mean), float(start.env.pixel_scale_mas),
                                      support=support, flux=scene.env.flux))
    pilot_pri = {k: v for k, v in priors.items() if not k.startswith("env.log_brightness")}
    pilot = fit(pilot_sc, image_priors(pilot_sc) | pilot_pri, data, **(dict(max_steps=15) if args.smoke else {})).model
    return float(error_scale(pilot, data)), float(chi2r(pilot, data)[0])


def geometry_fit(cfg, data_s, out, hole_mas, args):
    """A parametric ``MultipoleDisk`` (Gaussian ring + first harmonic) with the star(s), started from virgil's own blind rim fit
    (parametric stage), never the paper's values. Its inclination and position angle are the polar GP's geometry (``geometry_of``);
    the whole fitted model is the smooth truth of the ``calibrate`` stage."""
    from virgil.priors import IsotropicInclination  # noqa: PLC0415

    par = json.load(open(path(out, cfg, "parametric.json")))
    r0 = max(0.5 * par["rim_diam_mas"], 1.5 * hole_mas)
    disk = MultipoleDisk(GaussianRing(r0, 0.4 * r0), harmonics=Harmonics(re=0.1, im=0.1), inc=min(max(par["rim_inc_deg"], 5.0), 80.0),
                         pa=par["rim_pa_deg"] % 180.0, n_r=48, flux=PowerLaw(0.3, 0.0, L0))
    sc = sparco_scene(cfg, disk, f_env=0.3)
    gp = scene_priors(cfg, sc) | {
        "env.inc": IsotropicInclination(0, 85), "env.pa": dist.Uniform(-90.0, 180.0), "env.profile.radius": dist.LogUniform(0.5, 60.0),
        "env.profile.sigma": dist.LogUniform(0.1, 12.0), "env.harmonics.re": dist.Uniform(-1.0, 1.0), "env.harmonics.im": dist.Uniform(-1.0, 1.0)}
    return fit(sc, gp, data_s, **(dict(max_steps=15) if args.smoke else {}))


def geometry_of(g, data_s):
    """The disc geometry of a ``geometry_fit`` result, as stored in the scale file."""
    v = g.values
    return dict(inc_deg=float(v["env.inc"]), pa_deg=float(v["env.pa"]) % 180.0, radius_mas=float(v["env.profile.radius"]),
                width_mas=float(v["env.profile.sigma"]), h_re=float(np.ravel(v["env.harmonics.re"])[0]), h_im=float(np.ravel(v["env.harmonics.im"])[0]),
                chi2_per_point=float(chi2r(g.model, data_s)[0]), converged=bool(g.info.get("converged", True)))


def scale_stage(args, out):
    """Error rescaling and geometry, once per target: an isotropic GP pilot on the errors as given gives
    ``error_scale`` (MacKay), then a parametric ``MultipoleDisk`` (Gaussian ring + first harmonic) on the rescaled data gives the
    disc's inclination and position angle for the anisotropic and polar GPs and the disc strategy."""
    cfg = cfg_of(args)
    vlti = args.noise in ("vlti", "night")
    data, files = load(cfg, args.files)
    start, support, hole_mas = grid_setup(cfg, data, args)
    env = Image(start.env.log_brightness, start.env.pixel_scale_mas, support=support, flux=start.env.flux)
    scene = sparco_scene(cfg, env, f_env=float(np.asarray(env.flux)))
    priors = image_priors(scene) | scene_priors(cfg, scene)
    if vlti:  # no MacKay rescaling: virgil's error_scale refuses gains; the noise model sets the errors
        s, pilot_chi2, data_s = 1.0, None, modelled_noise(data, cfg, args.noise)
    else:
        s, pilot_chi2 = pilot_error_scale(scene, start, support, priors, data, args)
        data_s = data.with_error_scale(s)
        print(f"pilot GP: chi2/pt {pilot_chi2:.3f}; error scale s = {s:.3f}", flush=True)
    geo = geometry_of(geometry_fit(cfg, data_s, out, hole_mas, args), data_s)
    print("geometry", json.dumps(geo), flush=True)
    json.dump(dict(error_scale=s, noise=args.noise, pilot_chi2_per_point=pilot_chi2, hole_mas=hole_mas,
                   geometry=geo, smoke=args.smoke), open(scale_file(out, cfg, args.noise), "w"), indent=1)


def safe_l_curve(scene, priors, data, regulariser, weights, **opts):
    """``l_curve``, but a weight whose fit goes non-finite is skipped instead of fatal.

    virgil starts each weight from the previous fit, so one NaN fit poisons every later weight and ``l_curve`` raises. On that error the
    sweep is redone one weight at a time (largest first), keeping the last finite fit as the start; the skipped weights are printed and
    left out of the returned ``LCurve``."""
    from virgil.imaging import LCurve  # noqa: PLC0415

    try:
        curve = l_curve(scene, priors, data, regulariser, weights, **opts)
        if np.all(np.isfinite(curve.chi2)) and np.all(np.isfinite(curve.penalty)):
            return curve
        print("l_curve has non-finite fits -> sweeping one weight at a time and skipping them", flush=True)
    except ValueError as e:  # virgil raises ValueError for a NaN fit; the weights are then retried one at a time, and a sweep with no finite fit raises below
        print("l_curve raised", repr(str(e)[:120]), "-> sweeping one weight at a time and skipping non-finite ones", flush=True)
    kept, skipped, init = [], [], None

    def attempt(w, start):  # one weight from ``start``; (curve, finite)
        try:
            c = l_curve(scene, priors, data, regulariser, [w], init=start, **opts)
            r = c.results[0]
            return c, bool(np.isfinite(c.chi2[0]) and np.isfinite(c.penalty[0]) and all(np.all(np.isfinite(np.asarray(v))) for v in r.values.values()))
        except ValueError:
            return None, False

    for w in sorted((float(w) for w in weights), reverse=True):
        c, finite = attempt(w, init)
        if not finite and init is not None:  # the warm start may be what diverged: try from the scene itself
            c, finite = attempt(w, None)
        if finite:
            kept.append((w, c))
            init = c.results[0].values
        else:
            skipped.append(w)
    print("weights skipped (non-finite fit):", skipped, flush=True)
    if not kept:
        raise ValueError("every weight of the MaxEntropy sweep gave a non-finite fit")
    return LCurve(np.asarray([w for w, _ in kept]), np.asarray([float(c.chi2[0]) for _, c in kept]), np.asarray([np.asarray(c.chi2_red[0]) for _, c in kept]),
                  np.asarray([float(c.penalty[0]) for _, c in kept]), [c.results[0] for _, c in kept])


def fit_mem(cfg, scene, priors, data, args):
    """MaxEntropy image: the weight where chi2 per point, on the (rescaled) errors, is 1 (Morozov's discrepancy principle)."""
    weights = np.array([1e3, 1e1]) if args.smoke else np.logspace(*cfg.get("log_weights", (4, -2)), 9)
    opts = dict(max_steps=15) if args.smoke else dict(max_steps=1500)
    curve = safe_l_curve(scene, priors, data, MaxEntropy(1.0, path="env"), weights, **opts)
    try:
        corner = curve.corner()
    except ValueError:  # fewer than three swept weights converged (virgil raises); fall back to the middle weight
        corner = None
    corner_w = float(corner) if corner is not None else float(weights[len(weights) // 2])
    ws = np.asarray(curve.weights, float)
    chi = np.array([chi2r(r.model, data)[0] for r in curve.results])
    k = int(np.argmin(np.abs(np.log(chi))))
    model, w_star, how = curve.results[k].model, float(ws[k]), "closest swept weight to chi2 = 1 (the sweep does not bracket it)"
    cross = [i for i in range(len(ws) - 1) if (chi[i] - 1.0) * (chi[i + 1] - 1.0) <= 0 and chi[i] != chi[i + 1]]
    if cross and not args.smoke:
        i = cross[0]
        lo, hi = np.log(ws[i]), np.log(ws[i + 1])
        c_lo, c_hi = chi[i], chi[i + 1]
        start = curve.results[i].model
        for _ in range(4):  # interpolate in log w, refit from the nearest stronger fit, secant-correct
            w_try = float(np.exp(lo + (c_lo - 1.0) / (c_lo - c_hi) * (hi - lo)))
            try:
                model = safe_l_curve(start, priors, data, MaxEntropy(1.0, path="env"), [w_try], max_steps=2500).results[0].model
            except ValueError as e:  # the refinement went non-finite: keep the closest swept fit
                print("discrepancy refinement skipped:", e, flush=True)
                break
            c_try = chi2r(model, data)[0]
            w_star, how = w_try, "discrepancy"
            if abs(c_try - 1.0) < 0.03:
                break
            if c_try > 1.0:
                lo, c_lo = np.log(w_try), c_try
            else:
                hi, c_hi = np.log(w_try), c_try
    arrays = dict(weights=ws, chi2_red=np.asarray(curve.chi2_red).tolist(), chi2_all=chi,
                  penalty=np.asarray(curve.penalty), chosen=k, chosen_weight=w_star)
    return model, arrays, dict(regulariser="MaxEntropy", weight=w_star, weight_rule=how, corner_weight=corner_w)


def pick_cell(log_z, chi2):
    """The grid cell with the largest Laplace evidence, and whether it lies on the edge of the grid."""
    cell = np.unravel_index(int(np.argmax(log_z)), log_z.shape)
    edge = any(c in (0, n - 1) for c, n in zip(cell, log_z.shape) if n > 1)
    return cell, bool(edge)


def grid_search(axes, run, extenders, smoke=False, eligible=None):
    """Evidence maximisation over a grid of hyperparameters, extended once in the direction of any edge pick.

    ``axes`` is a list of ascending value lists; ``run(values)`` fits one cell and returns ``(fit, log_z, chi2)``. After the first pass
    the best cell is looked at: for each axis it sits at the end of, ``extenders[axis]`` is a ``(lower, upper)`` pair of functions of
    the axis' current end value giving one more value beyond it (or ``None`` for an end that cannot move, such as a correlation length
    below the pixel, where a GP is white noise). The new slice is fitted and the best cell picked again, each axis-direction extended
    at most once, so an edge pick that survives is reported (``gp_on_edge``) rather than chased. ``eligible(fit, values)`` can bar
    cells from the pick (the disc's render guard). Returns ``(cells, axes, best_values, on_edge, extended)``; ``cells`` maps value
    tuples to ``(fit, log_z, chi2)``."""
    import itertools  # noqa: PLC0415

    axes = [list(a) for a in axes]
    cells, used, extended = {}, set(), []

    def fill():
        for vals in itertools.product(*axes):
            if vals not in cells:
                cells[vals] = run(vals)

    def best():
        score = {v: (c[1] if eligible is None or eligible(c[0], v) else -np.inf) for v, c in cells.items()}
        if not np.isfinite(max(score.values())):  # no cell is eligible: take the evidence's choice
            score = {v: c[1] for v, c in cells.items()}
        return max(score, key=score.get)

    fill()
    while not smoke:
        b = best()
        new = False
        for k, ext in enumerate(extenders):
            for side, name in ((0, "lower"), (1, "upper")):
                at_edge = b[k] == (axes[k][0] if side == 0 else axes[k][-1]) and len(axes[k]) > 1
                if at_edge and ext[side] is not None and (k, side) not in used:
                    v = ext[side](axes[k][0] if side == 0 else axes[k][-1])
                    if v is not None:
                        used.add((k, side))
                        axes[k] = sorted(axes[k] + [v])
                        extended.append((k, name, float(v)))
                        new = True
        if not new:
            break
        fill()
    b = best()
    on_edge = any(len(a) > 1 and b[k] in (a[0], a[-1]) for k, a in enumerate(axes))
    return cells, axes, b, on_edge, extended


def cube(cells, axes, which):
    """Array of ``cells[...][which]`` over the product of ``axes`` (``which``: 1 log Z, 2 chi2)."""
    out = np.zeros([len(a) for a in axes])
    for idx in np.ndindex(*out.shape):
        out[idx] = cells[tuple(a[i] for a, i in zip(axes, idx))][which]
    return out


def pixel_floor_lengths(pix, unit, mults=(0.5, 1.0, 2.0)):
    """GP correlation lengths from one pixel (the floor: shorter is white noise) up through ``mults`` x the comparison beam, with one
    geometric intermediate step between the pixel and the first beam multiple."""
    top = [m * unit for m in mults if m * unit > 1.05 * pix]
    first = top[0] if top else 2.0 * pix
    return sorted({float(pix), float(np.sqrt(pix * first)), *top}) if first > 1.2 * pix else sorted({float(pix), *top})


def field_mean(start_env, support, args):
    """The GP's template image: the Gaussian start blob (``--mean blob``, the default), or ``--mean flat``, one inside the support (the
    field then models the whole log-brightness about a uniform disc; outside the support the template is the 1e-3 floor, and masked)."""
    if getattr(args, "mean", "blob") == "flat":
        return np.where(np.asarray(support, bool), 1.0, 0.0)
    return np.asarray(start_env.brightness)


def fit_gp(cfg, scene, start_env, support, priors, data, args, geo=None, aniso=False):
    """Gaussian-process image: a GaussianField log-brightness about the Gaussian start, (sigma, length) from the Laplace evidence.

    ``data`` already carries the MacKay error rescaling. With ``aniso`` the grid's up axis points along the disc's major axis
    (``geo['pa_deg']``) and the correlation lengths are (3 l, l) along and across it. Lengths are fractions of the comparison beam.
    """
    sigmas = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    unit = compare_beam(data).minor_mas
    pix_ = float(start_env.pixel_scale_mas)
    lengths = pixel_floor_lengths(pix_, unit, cfg.get("gp_mults", (0.5, 1.0, 2.0)))
    if args.smoke:
        sigmas, lengths = sigmas[:1], lengths[:1]
    ratio = 3.0 if aniso else None
    rot = float(geo["pa_deg"]) if aniso else 0.0
    mean = field_mean(start_env, support, args)
    pix = float(start_env.pixel_scale_mas)
    opts = dict(max_steps=15) if args.smoke else {}

    def gp_scene(sigma, length):
        ell = (ratio * length, length) if ratio else length
        env = Image(GaussianField(np.zeros(mean.shape), sigma, ell, mean=mean), pix, support=support, flux=scene.env.flux, rotation_deg=rot)
        return scene.set("env", env)

    priors = {k: v for k, v in priors.items() if not k.startswith("env.log_brightness")}
    state = dict(previous=None)

    def run(vals):
        length, sigma = vals
        sc = gp_scene(sigma, length)
        prev = state["previous"]
        f_ = state["previous"] = fit(sc, image_priors(sc) | priors, data, init=None if prev is None else prev.values, **opts)
        lz, c2 = float(log_evidence(f_, data)), float(f_.info["chi2_red"])
        print(f"length {length:.2f} mas, sigma {sigma}: log Z {lz:.1f}, chi2/pt {c2:.3f}", flush=True)
        return f_, lz, c2

    cells, (lengths, sigmas), (bl, bs), edge, extended = grid_search(
        [lengths, sigmas], run, [(None, lambda v: 2.0 * v), (lambda v: v / 2.0, lambda v: 2.0 * v)], smoke=args.smoke)
    log_z, chi2 = cube(cells, (lengths, sigmas), 1), cube(cells, (lengths, sigmas), 2)
    bi, bj = lengths.index(bl), sigmas.index(bs)
    best = cells[bl, bs][0]
    arrays = dict(gp_log_z=log_z, gp_chi2=chi2, gp_sigmas=np.asarray(sigmas), gp_lengths_mas=np.asarray(lengths), chosen=np.array([bi, bj]))
    return best.model, arrays, dict(
        regulariser="GaussianField", rotation_deg=rot, weight_rule="Laplace evidence", gp_chi2=float(chi2[bi, bj]), gp_sigma=float(bs),
        gp_length_mas=float(bl), gp_ratio=ratio, gp_length_beams=float(bl / unit), log_evidence=float(log_z[bi, bj]),
        gp_on_edge=edge, gp_grid_extended=extended, gp_converged=bool(best.info.get("converged", True)))


def azimuthal_log_spectrum(m, length_theta, nu):
    """Log variance of azimuthal order ``m`` under the periodic Matérn of ``MultipoleKernel`` (virgil's documented spectrum,
    ``[1 + (m l)^2 / (2 nu)]^(-nu - 1/2)``, Gaussian for ``nu`` = inf), up to a constant."""
    x = (jnp.asarray(m, float) * length_theta) ** 2
    return -0.5 * x if np.isinf(nu) else -(nu + 0.5) * jnp.log1p(x / (2.0 * nu))


class LowOrderField(MultipoleField):
    """A ``MultipoleField`` whose azimuthal orders above ``m_max`` are removed (a low-pass along the azimuth of the polar grid).

    Within the cap the orders keep the periodic Matérn spectrum in ``m * length_theta``; the kept orders are rescaled so that the
    field's variance per node is still ``sigma^2``, so ``sigma`` means the same rms log-brightness contrast with or without the
    cap. The latents of the removed orders do not reach the image: their Jacobian is zero, the fit leaves them at zero and they add
    nothing to the Laplace evidence, so cells of one cap compare fairly. Why a cap: ``design/gp_polar_hyperparameters.md``. With
    ``n_theta`` = 32 nodes a field of order 3 or less is sampled at ten or more nodes per period, so the bilinear interpolation onto
    the pixels is smooth (a cap at ``n_theta`` = 8 nodes would draw polygons instead)."""

    m_max: int = eqx.field(static=True)

    def __init__(self, latent, image_shape, r_max_mas, m_max=3, **kwargs):
        super().__init__(latent, image_shape, r_max_mas, **kwargs)
        n_theta = int(np.shape(latent)[1])
        if not 0 <= int(m_max) < n_theta // 2:
            raise ValueError(f"m_max must be in [0, n_theta/2) = [0, {n_theta // 2}), not {m_max}")
        self.m_max = int(m_max)

    def with_geometry(self, inc, pa, dra, ddec):
        """This field untied, with the given geometry (as ``MultipoleField.with_geometry``), keeping its cap."""
        return LowOrderField(self.latent, self.image_shape, self.r_max_mas, m_max=self.m_max, r_min_mas=self.r_min_mas, sigma=self.sigma,
                             length_r=self.length_r, length_theta=self.length_theta, inc=inc, pa=pa, dra=dra, ddec=ddec, nu=self.nu,
                             mean=self.mean, mean_floor=self.mean_floor, outside=self.outside, nu_theta=self.nu_theta)

    def polar_values(self):
        field = super().polar_values()
        n_theta = field.shape[1]
        j = jnp.arange(n_theta)
        m = jnp.minimum(j, n_theta - j)
        log_s = azimuthal_log_spectrum(m, self.length_theta, self.nu_theta)
        keep = m <= self.m_max
        gain = jnp.exp(0.5 * (jax.scipy.special.logsumexp(log_s) - jax.scipy.special.logsumexp(jnp.where(keep, log_s, -jnp.inf))))
        return jnp.real(jnp.fft.ifft(jnp.where(keep, gain, 0.0) * jnp.fft.fft(field, axis=1), axis=1))


# gp-polar grid. Azimuthal orders are capped at POLAR_M_MAX (Ben: none of these discs should go to high azimuthal order); within
# the cap l_theta = 0.25 rad already gives the orders 0-3 nearly equal variance, so shorter lengths add nothing and are not searched.
POLAR_M_MAX = 3
POLAR_SIGMAS = (0.125, 0.25, 0.5, 1.0, 2.0)
POLAR_LT = (0.25, 0.5, 1.0, 2.0)
POLAR_LT_UNCAPPED = (0.125, 0.25, 0.5, 1.0, 2.0)  # the grid before the cap (m_max None), kept for the calibration's uncapped arm
POLAR_LR_BEAMS = (1.0, 2.0, 4.0)
# Prior bounds on the evidence's choice (Ben: "bound by prior"), fixed from the stage-C calibration before they were applied to the
# real targets (design/gp_polar_hyperparameters.md §8). l_r >= 1 beam: the data do not constrain radial structure below the beam, and
# the truth's best cell was 2 beams in 65 of 72 calibration runs. sigma <= 2 (rms log-contrast per node, e^2 ~ 7x): the truth needed
# sigma = 4 in 5 of 72. The grid is never extended past them; a pick on a bound is reported (gp_at_bound).
POLAR_LR_MIN_BEAMS = 1.0
POLAR_SIGMA_MAX = 2.0
POLAR_MAX_STEPS = 3000  # every cell starts cold, from the same point, so it needs more LM steps than a warm-started chain


def fit_cell(scene, priors, data, opts):
    """One fit with its loss history (from virgil's ``progress`` reports, one per LM chunk) and stopping record."""
    hist, t0 = [], time.time()
    f_ = fit(scene, priors, data, progress=lambda r: hist.append([int(r["steps"]), float(r["loss"]), float(r["grad_norm"])]), **opts)
    i = f_.info
    return f_, dict(steps=int(i.get("steps", -1)), stop=str(i.get("stop")), converged=bool(i.get("converged", True)),
                    loss=float(i.get("loss", np.nan)), grad_norm=float(i.get("grad_norm", np.nan)), seconds=round(time.time() - t0, 1),
                    history=hist)


def polar_scene_fn(cfg, scene, start_env, support, args, geo, r_min, m_max):
    """``(sigma, length_r, length_theta) -> scene`` with the polar field as its environment (``LowOrderField`` unless ``m_max`` is None)."""
    mean = field_mean(start_env, support, args)
    pix = float(start_env.pixel_scale_mas)
    r_max = float(cfg["half_mas"])
    n_r, n_th = 40, 32

    def mp_scene(sigma, lr, lt):
        kw = dict(image_shape=mean.shape, r_max_mas=r_max, r_min_mas=r_min, sigma=sigma, length_r=lr, length_theta=lt,
                  inc=geo["inc_deg"], pa=geo["pa_deg"], mean=mean)
        field = MultipoleField(np.zeros((n_r, n_th)), **kw) if m_max is None else LowOrderField(np.zeros((n_r, n_th)), m_max=m_max, **kw)
        return scene.set("env", Image(field, pix, support=support, flux=scene.env.flux))

    return mp_scene


def polar_bounds(unit, pix):
    """The prior bounds of the gp-polar choice: (smallest length_r in mas, largest sigma)."""
    return max(POLAR_LR_MIN_BEAMS * unit, pix), POLAR_SIGMA_MAX


def polar_cells(cfg, scene, start_env, support, priors, data, args, geo, r_min, m_max=POLAR_M_MAX, axes=None, extend=True):
    """Fit the gp-polar grid: every cell from the same cold start (the scene's own parameters, zero latents), so that no cell's
    evidence depends on the order the grid is visited in. ``axes`` = (length_r mas, length_theta rad, sigma) lists, by default the
    POLAR_* grid within the prior bounds (``polar_bounds``); with ``extend`` an edge pick is extended once (``grid_search``), never
    past a bound. Explicit ``axes`` (the calibration's fixed grid) are used as given. Returns ``grid_search``'s tuple; each cell is
    ``(fit, log Z, chi2/pt, record)`` with the record of ``fit_cell``."""
    unit_, pix_ = compare_beam(data).minor_mas, float(start_env.pixel_scale_mas)
    lr_min, sg_max = (pix_, np.inf) if getattr(args, "unbounded", False) else polar_bounds(unit_, pix_)
    if getattr(args, "min_length_r", None) is not None:  # --min-length-r: no radial length below this many beams (a smooth image cannot hold a point)
        lr_min = max(lr_min, args.min_length_r * unit_)
    if axes is None:
        lr_beams = (0.5, *POLAR_LR_BEAMS) if getattr(args, "unbounded", False) else POLAR_LR_BEAMS  # unbounded: the grid before §8
        lens_r = [m * unit_ for m in cfg.get("polar_lr_beams", lr_beams) if m * unit_ >= lr_min * (1 - 1e-9)]
        axes = [lens_r, list(POLAR_LT if m_max is not None else POLAR_LT_UNCAPPED), [s for s in POLAR_SIGMAS if s <= sg_max]]
    if args.smoke:
        axes = [a[:1] for a in axes]
    mp_scene = polar_scene_fn(cfg, scene, start_env, support, args, geo, r_min, m_max)
    priors = {k: v for k, v in priors.items() if not k.startswith("env.log_brightness")}
    opts = dict(max_steps=15) if args.smoke else dict(max_steps=POLAR_MAX_STEPS)

    def run(vals):
        lr, lt, sg = vals
        sc = mp_scene(sg, lr, lt)
        f_, rec = fit_cell(sc, image_priors(sc) | priors, data, opts)
        lz, c2 = float(log_evidence(f_, data)), float(f_.info["chi2_red"])
        print(f"length_r {lr:.2f} mas, length_theta {lt:g} rad, sigma {sg}: log Z {lz:.1f}, chi2/pt {c2:.3f}, {rec['steps']} steps "
              f"({rec['stop']}), {rec['seconds']:.0f} s", flush=True)
        return f_, lz, c2, rec

    # lengths in radius go up by doubles and never below the bound; in azimuth, 2 rad is nearly one lobe, so an upper edge is a
    # constant ring, and under the cap a shorter length than the grid's adds nothing; sigma goes down by halves, never above its bound
    lt_lower = None if m_max is not None else (lambda v: v / 2.0)
    extenders = [(lambda v: v / 2.0 if v / 2.0 >= lr_min * (1 - 1e-9) else None, lambda v: 2.0 * v), (lt_lower, lambda v: 2.0 * v),
                 (lambda v: v / 2.0, lambda v: 2.0 * v if 2.0 * v <= sg_max else None)]
    return grid_search(axes, run, extenders, smoke=args.smoke or not extend)


def cell_records(cells, axes):
    """JSON-ready list of every cell: its values, log Z, chi2/pt and fit record (loss history included)."""
    import itertools  # noqa: PLC0415

    return [dict(length_r_mas=float(v[0]), length_theta=float(v[1]), sigma=float(v[2]), log_z=cells[v][1], chi2=cells[v][2], **cells[v][3])
            for v in itertools.product(*axes)]


def fit_polar(cfg, scene, start_env, support, priors, data, args, geo, r_min):
    """Polar-GP image (``virgil.fields.MultipoleField``, azimuthal orders capped by ``LowOrderField``): a Matern kernel in the disc's
    radius times a periodic one in azimuth, deprojected by the inclination and position angle of our own parametric disc fit
    (``geo``); the polar grid starts at the hole radius. (sigma, length_r, length_theta) from the Laplace evidence on the rescaled
    data, every cell from the same cold start (``polar_cells``), within the prior bounds ``POLAR_LR_MIN_BEAMS`` and
    ``POLAR_SIGMA_MAX``."""
    m_max = cfg.get("polar_m_max", POLAR_M_MAX)
    cells, (lens_r, lens_t, sigmas), (blr, blt, bsg), edge, extended = polar_cells(cfg, scene, start_env, support, priors, data, args,
                                                                                   geo, r_min, m_max=m_max)
    log_z, chi2 = cube(cells, (lens_r, lens_t, sigmas), 1), cube(cells, (lens_r, lens_t, sigmas), 2)
    bi, bj, bk = lens_r.index(blr), lens_t.index(blt), sigmas.index(bsg)
    best = cells[blr, blt, bsg][0]
    arrays = dict(gp_log_z=log_z[:, :, bk], gp_chi2=chi2[:, :, bk], gp_sigmas=np.asarray(lens_t), gp_lengths_mas=np.asarray(lens_r),
                  gp_xlabel="ℓ_θ (rad)", gp_ylabel="ℓ_r (mas)", chosen=np.array([bi, bj]), mp_log_z=log_z, mp_chi2=chi2,
                  mp_sigmas=np.asarray(sigmas), mp_cells=json.dumps(cell_records(cells, (lens_r, lens_t, sigmas))))
    rec = cells[blr, blt, bsg][3]
    mirror_rec = {}
    if getattr(args, "companion", None) == "free" and getattr(args, "companion_box", None) is None:  # a box about the start has no mirror in it
        best, mirror_rec = companion_mirror(cfg, scene, start_env, support, priors, data, args, geo, r_min, m_max, (blr, blt, bsg), best)
    unbounded = getattr(args, "unbounded", False)
    lr_min, sg_max = (float(start_env.pixel_scale_mas), np.inf) if unbounded else polar_bounds(compare_beam(data).minor_mas,
                                                                                               float(start_env.pixel_scale_mas))
    at_bound = [] if unbounded else [n for n, hit in (("length_r", np.isclose(blr, lr_min)), ("sigma", np.isclose(bsg, sg_max))) if hit]
    return best.model, arrays, dict(
        regulariser="MultipoleField" if m_max is None else f"MultipoleField, azimuthal orders <= {m_max}", mp_m_max=m_max,
        weight_rule="Laplace evidence, unbounded (diagnostic)" if unbounded else
        f"Laplace evidence within prior bounds (length_r >= {POLAR_LR_MIN_BEAMS:g} beam, sigma <= {POLAR_SIGMA_MAX:g})",
        gp_lr_min_mas=None if unbounded else float(lr_min), gp_sigma_max=None if unbounded else float(sg_max), gp_at_bound=at_bound, cell_start="cold", gp_chi2=float(chi2[bi, bj, bk]), gp_sigma=float(bsg),
        gp_length_r_mas=float(blr), gp_length_theta=float(blt), mp_inc_deg=geo["inc_deg"], mp_pa_deg=geo["pa_deg"],
        mp_r_min_mas=r_min, mp_r_max_mas=float(cfg["half_mas"]), log_evidence=float(log_z[bi, bj, bk]), gp_on_edge=edge,
        gp_grid_extended=extended, gp_converged=bool(best.info.get("converged", True)), gp_steps=rec["steps"], gp_stop=rec["stop"],
        gp_cells_unconverged=int(sum(not c[3]["converged"] for c in cells.values()))) | mirror_rec | dict(
        zip(("log_evidence_scene", "scene_sd"), scene_evidence(best, data, {k: v for k, v in priors.items() if not k.startswith("env.log_brightness")})))


def rendered_chi2_disc(model, shape, pix, data):
    """chi2/pt of the disc+residual model's rendered pixel image, put back into the fitted scene (the render guard of ``image_one``)."""
    b = render_env(model, "disc+res", shape, pix)
    if not (np.all(np.isfinite(b)) and b.sum() > 0):  # a fit that diverged has no image: never eligible
        return float("inf")
    m = model.resolved()
    env_r = Image(np.log(np.clip(b / b.sum(), 1e-14, None)), pix,
                  flux=PowerLaw(float(np.asarray(m.env.flux.ratio)) + float(np.asarray(m.res.flux.ratio)), float(np.asarray(m.env.flux.index)), L0))
    return chi2r(System(**{k: getattr(m, k) for k in ("star", "sec", "bg") if hasattr(m, k)}, env=env_r), data)[0]


def fit_disc_at(cfg, data, args, geo, r_in, f_env0, d_env, pix, npix):
    """Disc + residual: a ``MultipoleDisk`` with a ``RadialGP`` profile and degree-2 harmonics (``MultipoleKernel``), geometry
    frozen at our parametric fit (``geo``), plus a coarse ``MultipoleField`` residual that follows the disc's geometry.
    (sigma, length) of the profile from the Laplace evidence on the rescaled data; the inner edge is the hole.
    Disc and residual share one spectral index ``d_env`` (fixed here, chosen by ``fit_disc``): SPARCO's environment is one grey image
    with one index. Separate indices need no single image and make the rendered one meaningless (chi2/pt 76 against the model's 1.05)."""
    unit = compare_beam(data).minor_mas
    sigmas = [1.0, 2.0, 4.0, 8.0, 16.0]
    lens = pixel_floor_lengths(float(pix), unit, mults=(0.5, 1.0, 2.0, 4.0))
    if args.smoke:
        sigmas, lens = sigmas[:1], lens[:1]
    r_out = float(cfg["half_mas"])
    opts = dict(max_steps=15) if args.smoke else {}

    probe = MultipoleDisk(RadialGP(r_in, r_out, sigma=1.0, ell=unit), harmonics=Harmonics(re=0.0, im=0.0), inc=geo["inc_deg"], pa=geo["pa_deg"], n_r=48)
    # 192 nodes: the model's chi2 does not depend on n_r, but the rendered image only converges there (IW Car: rendered chi2/pt 2.44 at 48,
    # 2.02 at 96, 1.92 at 192 against the model's 1.86); suggest_n_r says 48 and is a starting point only.
    n_r = int(os.environ.get("POSTAGB_NR", 0)) or (192 if not args.smoke else 48)
    print("disc quadrature nodes n_r =", n_r, flush=True)

    def disc_scene(sigma, ell):
        disk = MultipoleDisk(
            RadialGP(r_in, r_out, sigma=sigma, ell=ell),
            harmonics=Harmonics(degree=2, prior=MultipoleKernel(ell_r=ell, ell_theta=2.0, sigma=0.5, n=10, r_in=r_in, r_out=r_out)),
            inc=geo["inc_deg"], pa=geo["pa_deg"], n_r=n_r, flux=PowerLaw(f_env0, d_env, L0))
        # the residual lives on the main image's pixel grid (its polar latent grid stays coarse), so rendering needs no resampling
        res = Image(MultipoleField(np.zeros((16, 16)), image_shape=(npix, npix), r_max_mas=r_out, r_min_mas=r_in, sigma=1.0, length_r=4 * unit,
                                   length_theta=1.0, geometry_from="env"),
                    pix, flux=PowerLaw(0.02, d_env, L0))
        sc = sparco_scene(cfg, disk, f_env=f_env0, extra=dict(res=res))
        return sc.set("env", disk.freeze_geometry(data))

    state, side = dict(previous=None), {}

    def run(vals):
        ell, sigma = vals
        sc = disc_scene(sigma, ell)
        pri = gp_priors(sc) | image_priors(sc) | scene_priors(cfg, sc) | {"res.flux.ratio": dist.LogUniform(1e-3, 0.3 * f_env0)}  # a residual, not the model: at most 30 % of the starting environment flux
        pri.pop("env.flux.index", None)
        pri.pop("res.flux.index", None)
        prev = state["previous"]
        f_ = state["previous"] = fit(sc, pri, data, init=None if prev is None else prev.values, **opts)
        lz, c2 = float(log_evidence(f_, data)), float(f_.info["chi2_red"])
        rend = rendered_chi2_disc(f_.model, (npix, npix), pix, data)
        side[vals] = (rend, bool(np.isfinite(rend) and np.isfinite(lz) and abs(rend - c2) <= 0.1 * c2 + 0.02))
        print(f"profile length {ell:.2f} mas, sigma {sigma}: log Z {lz:.1f}, chi2/pt {c2:.3f}, rendered {rend:.3f}"
              f"{'' if side[vals][1] else ' (render mismatch: not eligible)'}", flush=True)
        return f_, lz, c2

    # a profile rougher than the pixel grid can show has no image: such cells are not eligible (all of them, if none passes)
    cells, (lens, sigmas), (bl, bs), edge, extended = grid_search(
        [lens, sigmas], run, [(None, lambda v: 2.0 * v), (lambda v: v / 2.0, lambda v: 2.0 * v)], smoke=args.smoke,
        eligible=lambda f_, vals: side[vals][1])
    log_z, chi2 = cube(cells, (lens, sigmas), 1), cube(cells, (lens, sigmas), 2)
    rendered = np.array([[side[l_, s_][0] for s_ in sigmas] for l_ in lens])
    ok = np.array([[side[l_, s_][1] for s_ in sigmas] for l_ in lens])
    bi, bj = lens.index(bl), sigmas.index(bs)
    best = cells[bl, bs][0]
    arrays = dict(gp_log_z=log_z, gp_chi2=chi2, gp_rendered_chi2=rendered, gp_render_ok=ok, gp_sigmas=np.asarray(sigmas), gp_lengths_mas=np.asarray(lens), chosen=np.array([bi, bj]))
    return best.model, arrays, dict(
        regulariser="MultipoleDisk", n_r=n_r, weight_rule="Laplace evidence", gp_chi2=float(chi2[bi, bj]), gp_sigma=float(bs),
        gp_length_mas=float(bl), mp_inc_deg=geo["inc_deg"], mp_pa_deg=geo["pa_deg"], mp_r_min_mas=r_in, mp_r_max_mas=r_out,
        log_evidence=float(log_z[bi, bj]), gp_on_edge=edge, gp_grid_extended=extended, gp_converged=bool(best.info.get("converged", True)))


def fit_disc(cfg, data, args, geo, r_in, f_env0, d_mem, pix, npix):
    """``fit_disc_at`` over a grid of the shared environment index (the MEM fit's value ``d_mem`` and one unit either side), the largest Laplace
    evidence choosing it: virgil cannot tie two components' indices, so the one free SPARCO index is scanned."""
    ds = sorted({round(float(d_mem) + k, 2) for k in (-1.0, 0.0, 1.0)})  # the evidence peaks within +-1 of the MEM value (IW Car: 1.0 against 0.82)
    if args.smoke:
        ds = ds[:1]
    fits = {d: fit_disc_at(cfg, data, args, geo, r_in, f_env0, d, pix, npix) for d in ds}
    best = max(fits, key=lambda d: fits[d][2]["log_evidence"])
    model, arrays, extra = fits[best]
    arrays["d_env_grid"] = np.asarray(ds)
    arrays["d_env_log_z"] = np.asarray([fits[d][2]["log_evidence"] for d in ds])
    extra["d_env_shared"] = float(best)
    extra["d_env_on_edge"] = bool(best in (ds[0], ds[-1]))
    print("shared index grid:", {d: round(fits[d][2]["log_evidence"], 1) for d in ds}, "-> chosen", best, flush=True)
    return model, arrays, extra


def render_env(model, strategy, bright_shape, pix):
    """The environment's brightness on the pixel grid (North up, East left), flux-weighted for the disc strategy."""
    if strategy != "disc+res":
        return np.asarray(model.env.brightness)
    n = bright_shape[0]
    model = model.resolved()  # the residual's geometry_from="env" tie is only applied here, not when its pixels are read off the component
    disk = np.asarray(model.env.render(npix=n, fov_mas=n * pix))
    f_d = float(np.asarray(model.env.flux.ratio))
    res = np.asarray(model.res.brightness)  # coarse residual on its own grid, resampled to ours
    res_g = res / res.sum() if res.shape == (n, n) and abs(float(model.res.pixel_scale_mas) - pix) < 1e-9 else to_grid(
        res / res.sum(), float(model.res.pixel_scale_mas), pix, n)
    return f_d * disk / disk.sum() + float(np.asarray(model.res.flux.ratio)) * res_g / res_g.sum()


def per_strategy(fn, args, out, *extra):
    """Run ``fn`` for each requested strategy, with the target's outputs named <target>_<strategy>_*."""
    base = TARGETS[args.target]
    for strat in (STRATEGIES if args.strategy == "all" else (args.strategy,)):
        # a one-off star-index or background-index run, and every non-default noise or mean, is named apart and never overwrites the headline files
        sfx = ("" if args.star_index is None else f"_si{abs(args.star_index):g}") + ("" if args.bg_index is None else f"_bg{args.bg_index:g}")
        sfx += noise_tag(args.noise) + ("" if args.mean == DEFAULT_MEAN else f"_{args.mean}")
        sfx += "_chrom" if getattr(args, "chrom", False) else ""
        sfx += "" if getattr(args, "binary", "paper") == "paper" else f"_bin{args.binary}"
        sfx += "_unbounded" if getattr(args, "unbounded", False) else ""
        sfx += "_wide" if getattr(args, "wide_field", False) else ""
        sfx += option_suffix(args)
        TARGETS[args.target] = dict(base, key=f"{base['key']}_{strat}{sfx}", data_key=base.get("data_key", base["key"]))
        try:
            fn(args, out, strat, *extra)
        except FileNotFoundError as e:  # a strategy not run for this target
            print("skipped", strat, e)
        finally:
            TARGETS[args.target] = base


def image_stage(args, out):
    """The imaging strategies (``--strategy``): the common SPARCO scene, each strategy's environment, hyperparameters by evidence."""
    per_strategy(image_one, args, out)


def bound_warnings(model, priors, tol=0.01):
    """Fitted scene parameters that sit within ``tol`` of the range of their uniform / log-uniform prior: the prior, not the data, then
    chose them. (Field latents and Normal priors have no bound.)"""
    out = []
    for key, pr in priors.items():
        lo, hi = getattr(pr, "low", None), getattr(pr, "high", None)
        if lo is None or hi is None or key.startswith(("env.log_brightness", "env.field")) or ".z" in key:
            continue
        try:
            v = model
            for part in key.split("."):
                v = getattr(v, part)
            v = float(np.ravel(np.asarray(v))[0])
        except Exception:  # noqa: BLE001 - a prior on something that is not a plain scalar of the model
            continue
        lo, hi = float(np.ravel(np.asarray(lo))[0]), float(np.ravel(np.asarray(hi))[0])
        if "LogUniform" in type(pr).__name__ and v > 0 and lo > 0:
            v, lo, hi = np.log(v), np.log(lo), np.log(hi)
        span = hi - lo
        if v - lo < tol * span:
            out.append(f"{key} at its lower bound")
        elif hi - v < tol * span:
            out.append(f"{key} at its upper bound")
    return out


def warm_params(out, cfg, strat, args):
    """Scene parameters of the previous run of this target and strategy, if it converged cleanly: ``(values by path, source)``.
    Only the scene (flux ratios, indices, secondary position) is reused: field latents are not saved, and a changed field size
    (PS Gem, HR 4049) would not fit them anyway. Disc runs need the render guard to have passed."""
    f = out / f"{cfg['key']}_image.json"
    if args.smoke or args.cold or not f.exists():
        return None, None
    j = json.load(open(f))
    clean = j.get("gp_converged", True) is not False and 0.2 < j.get("chi2_per_point", 9.0) < 3.0 and not j.get("smoke") and not j.get("image_degenerate")
    if strat == "disc+res":
        clean = clean and bool(j.get("render_ok")) and j.get("d_env_shared") is not None
    if not clean or not j.get("f_prim_pct"):
        return None, None
    fp = j["f_prim_pct"]
    w = {"env.flux.ratio": float(np.clip(j["f_env_pct"] / fp, 0.011, 19.0)), "env.flux.index": float(np.clip(j["env_index"], -5.9, 5.9))}
    if cfg.get("bg") and j.get("f_bg_pct") is not None and j.get("bg_index") is not None:
        w |= {"bg.flux.ratio": float(np.clip(j["f_bg_pct"] / fp, 1e-4, 1.9)), "bg.flux.index": float(np.clip(j["bg_index"], -5.9, 5.9))}
        if cfg.get("bg_index_fixed") is not None:
            w.pop("bg.flux.index")
    if j.get("sec_dra") is not None and cfg.get("sec0") is not None and cfg.get("secondary", True):
        w |= {"sec.flux.ratio": float(np.clip(j["f_sec_pct"] / fp, 1e-4, 0.97)), "sec.dra": float(j["sec_dra"]), "sec.ddec": float(j["sec_ddec"])}
        if j.get("sec_index") is not None and "sec_index0" in cfg:
            w["sec.flux.index"] = float(np.clip(j["sec_index"], -4.9, 1.9))
    return w, f.name


def skip_clean_reason(out, cfg, strat, args):
    """Why this target x strategy need not be rerun (its previous result converged, was not on a grid edge, and for the disc
    passed the render guard), or None. Only with ``--reuse-converged``; the reuse is recorded by `image_one`. PS Gem and HR 4049 changed field size, so they always rerun; MEM does not use the GP grids."""
    f = out / f"{cfg['key']}_image.json"
    if args.smoke or not args.reuse_converged or not f.exists() or cfg.get("fov_override"):
        return None
    j = json.load(open(f))
    if j.get("smoke") or not (0.2 < j.get("chi2_per_point", 9.0) < 3.0) or j.get("gp_converged", True) is False or j.get("image_degenerate"):
        return None
    if j.get("f_sec_pct") and j["f_sec_pct"] / j["f_prim_pct"] > 0.28:
        return None  # pinned at the old 0.3 bound of the secondary's flux ratio
    if ("sec_index0" not in cfg and not j.get("star_index_source", "").startswith("ORGANIC")) or (
            cfg.get("sec_index_fixed") is not None and not j.get("sec_treatment")):
        return None  # fitted before the default secondary treatment (PMOIRED prior, fixed index, no window inside the primary hole)
    if strat == "mem":
        return "mem does not use the GP grids"
    if strat == "gp-polar" and (j.get("cell_start") != "cold" or j.get("mp_m_max") != cfg.get("polar_m_max", POLAR_M_MAX)):
        return None  # fitted before the azimuthal cap, or with warm-started cells: rerun
    if j.get("gp_on_edge") is not False:
        return None
    if strat == "disc+res" and not (j.get("render_ok") and j.get("d_env_shared") is not None and j.get("n_r") == 192):
        return None
    return "converged, interior of its grid" + (", render guard passed" if strat == "disc+res" else "")


def image_one(args, out, strat):
    cfg = cfg_of(args)
    why = skip_clean_reason(out, cfg, strat, args)
    if why:
        print("skipped", strat, "-", why, flush=True)
        prev = out / f"{cfg['key']}_image.json"
        json.dump(dict(strategy=strat, reuse_converged=True, reason=why, kept_file=prev.name,
                       kept_sha256=hashlib.sha256(prev.read_bytes()).hexdigest(),
                       utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")),
                  open(path(out, cfg, f"{strat}_reused.json"), "w"), indent=1)
        return
    data_s, data0, s, files = load_scaled(cfg, args, out)
    support_floor = check_support(cfg, data0) if not args.smoke else None
    geo = resolve_geometry(args, out, cfg, strat)
    t0 = time.time()
    rot = float(geo["pa_deg"]) if strat == "gp-aniso" else 0.0
    start, support, hole_mas = grid_setup(cfg, data_s, args, rot)
    env = Image(start.env.log_brightness, start.env.pixel_scale_mas, support=support, flux=start.env.flux, rotation_deg=rot)
    pix_ = float(env.pixel_scale_mas)  # the pixel actually used (grid_setup may take a finer one than the paper's), not cfg["pixel_mas"]
    print(strat, "grid", env.log_brightness.shape, "pixel", env.pixel_scale_mas, "mas; hole radius", f"{hole_mas:.3f}", "mas;",
          f"error scale {s:.3f}", flush=True)
    f_env0 = float(np.asarray(env.flux))
    chrom = getattr(args, "chrom", False)
    if chrom and strat != "gp-polar":
        raise SystemExit("--chrom is for gp-polar only (design §11)")
    rim, chrom_pri = chromatic_ring(cfg, geo, compare_beam(data_s).minor_mas) if chrom else (None, {})
    companion = getattr(args, "companion", None)
    if companion and strat != "gp-polar":
        raise SystemExit("--companion is for gp-polar only")
    scene = sparco_scene(cfg, env, f_env=f_env0, extra=(dict(rim=rim) if chrom else {}) | (dict(comp=companion_component(args)) if companion else {}) or None)
    warm, warm_src = warm_params(out, cfg, strat, args)
    if warm and strat != "disc+res":
        for k_, v_ in warm.items():
            scene = scene.set(k_, v_)
    if warm:
        print("warm start from", warm_src, {k_: round(v_, 3) for k_, v_ in warm.items()}, flush=True)
    binary = getattr(args, "binary", "paper")
    if binary != "paper" and strat != "gp-polar":
        raise SystemExit("--binary is for gp-polar only (design §12)")
    scene, bin_pri = binary_scene(scene, cfg, data_s, binary)
    priors = image_priors(scene) | scene_priors(cfg, scene) | chrom_pri | bin_pri | (companion_priors(args) if companion else {})
    has_sec = cfg["sec0"] is not None and cfg.get("secondary", True)
    if strat == "mem":
        model, curve_arrays, extra_gp = fit_mem(cfg, scene, priors, data_s, args)
    elif strat in ("gp-iso", "gp-aniso"):
        model, curve_arrays, extra_gp = fit_gp(cfg, scene, start.env, support, priors, data_s, args, geo, aniso=strat == "gp-aniso")
    elif strat == "gp-polar":
        model, curve_arrays, extra_gp = fit_polar(cfg, scene, start.env, support, priors, data_s, args, geo, hole_mas)
    else:
        mem_json = out / f"{cfg['key'].split('_disc+res')[0]}_mem{noise_tag(args.noise)}_image.json"  # the mem fit under the same noise model
        d_env = json.load(open(mem_json))["env_index"] if mem_json.exists() else 0.0
        model, curve_arrays, extra_gp = fit_disc(cfg, data_s, args, geo, hole_mas, f_env0, d_env, float(env.pixel_scale_mas), env.log_brightness.shape[0])

    extra = dict(strategy=strat, noise=args.noise, gp_mean=args.mean, noise_model=None if args.noise == "current" else (NIGHTW if args.noise == "nightw" else effective_noise_widths(cfg, getattr(args, "white", None)) | (night_widths(cfg) if args.noise == "night" else {})), error_scale=s, rotation_deg=rot, hole_mas=hole_mas, hole_rule="max(lambda_min/(2 B_max), UD diameter)",
                 n_blanked=int((~support).sum()), star_index=cfg.get("img_star_index", -4.0), ud_fixed_mas=cfg["ud_fix"],
                 bound_warnings=bound_warnings(model, priors | ({"res.flux.ratio": dist.LogUniform(1e-3, 0.3 * f_env0)} if strat == "disc+res" else {})),
                 sec_treatment=("PMOIRED position/flux prior, index fixed at the paper's SIND2, no window inside the primary hole"
                                if cfg.get("sec_index_fixed") is not None else None),
                 star_index_source=cfg.get("star_index_source", "SPARCO convention (-4)"), sec_index_source=cfg.get("sec_index_source"),
                 warm_start=bool(warm) and strat != "disc+res", warm_source=warm_src if strat != "disc+res" else None,
                 field_radius_mas=float(cfg["half_mas"]), grid_half_mas=float(np.shape(env.log_brightness)[0] * pix_ / 2), support_radius_mas=float(cfg.get("mask_mas", np.shape(env.log_brightness)[0] * pix_ / 2))) | extra_gp
    if chrom:
        extra |= dict(chrom=True) | chromatic_summary(model, cfg)
        extra["star_index"] = extra["chrom_star_index"]  # fitted under --chrom (star_index_source keeps the window's centre)
    if binary != "paper":
        extra |= binary_summary(model, binary, cfg, extra.get("scene_sd"), data_s)
        extra["sec_treatment"] = (f"--binary {binary}: position Normal(sec0, {BINARY_POS_SIGMA_MAS:g} mas), flux and index free"
                                  + ("" if binary == "free" else f", both stars moving against the disc ({binary})")) if has_sec else f"--binary {binary}: no secondary"
        extra["sec_index_source"] = "fitted" if has_sec else None
    c2, n = chi2r(model, data_s)
    pix = float(env.pixel_scale_mas)
    bright_grid = render_env(model, strat, np.shape(env.log_brightness), pix)
    bright = bright_grid if not rot else np.clip(rotate_img(bright_grid, rot, order=1, reshape=False, mode="constant", cval=0.0), 0.0, None)
    mv2, mt3 = observables(model, data_s)
    try:  # a diagnostic never discards a finished fit
        resid = residual_stats(data0, mv2, mt3)  # on the data as quoted, whichever noise model the fit used
    except Exception as e:  # noqa: BLE001
        resid = dict(failed=f"{type(e).__name__}: {e}")
        print("residual statistics failed:", resid["failed"], flush=True)
    # guard: the rendered environment, put back into the fitted scene as a pixel image, must fit as well as the model does
    b_ = bright_grid / bright_grid.sum()
    env_r = Image(np.log(np.clip(b_, 1e-14, None)), pix, rotation_deg=rot, flux=PowerLaw(float(np.asarray(model.env.flux.ratio)) + (float(np.asarray(model.res.flux.ratio)) if strat == "disc+res" else 0.0),
                                                                      float(np.asarray(model.env.flux.index)), L0))
    chi2_rendered = chi2r(model.set("env", env_r) if strat != "disc+res" else System(**{k: getattr(model, k) for k in ("star", "sec", "bg") if hasattr(model, k)}, env=env_r), data_s)[0]
    render_ok = bool(abs(chi2_rendered - c2) <= 0.1 * c2 + 0.02)
    print(f"render guard: chi2/pt model {c2:.3f}, rendered image {chi2_rendered:.3f}", "ok" if render_ok else "RENDER MISMATCH", flush=True)
    f_env = float(np.asarray(model.env.flux.ratio))
    f_sec = float(np.asarray(model.sec.flux.ratio)) if has_sec else 0.0
    f_bg = float(np.asarray(model.bg.flux.ratio)) if cfg.get("bg") else 0.0
    f_res = float(np.asarray(model.res.flux.ratio)) if strat == "disc+res" else 0.0
    f_disk = f_env if strat == "disc+res" else None
    n_ = bright.shape[0]
    yy, xx = np.mgrid[:n_, :n_]
    outer = np.hypot(xx - (n_ - 1) / 2, yy - (n_ - 1) / 2) * pix > 0.8 * cfg["half_mas"]  # env flux near the field edge
    f_outer = (f_env + f_res) * float(bright[outer].sum() / bright.sum()) if bright.sum() > 0 else 0.0
    stars = dict(ud=float(cfg["ud_fix"]), f_env=f_env + f_res, env_index=float(np.asarray(model.env.flux.index)), f_sec=f_sec,
                 f_bg=f_bg, bg_index=float(np.asarray(model.bg.flux.index)) if cfg.get("bg") else None,
                 sec_dra=float(model.sec.dra) if has_sec else None, sec_ddec=float(model.sec.ddec) if has_sec else None,
                 sec_index=float(np.asarray(model.sec.flux.index)) if has_sec else None)
    degen = image_degeneracy(bright_grid, pix, float(cfg.get("mask_mas", n_ * pix / 2)), hole_mas)
    if degen["image_degenerate"]:
        print("DEGENERATE IMAGE (flagged, not scored):", "; ".join(degen["image_degenerate"]), flush=True)
    np.savez(path(out, cfg, "image.npz"), brightness=bright, brightness_grid=bright_grid, rotation_deg=rot, pixel_mas=pix, model_v2=mv2, model_t3=mt3,
             **curve_arrays, stars=json.dumps(stars))
    f_comp = float(np.asarray(model.comp.flux.ratio)) if companion else 0.0
    tot = 1.0 + stars["f_env"] + f_sec + f_bg + f_comp + (float(np.asarray(model.rim.flux)) if chrom else 0.0)  # the --chrom ring, as chromatic_summary
    summary = dict(chi2_per_point=c2, chi2_rendered=chi2_rendered, render_ok=render_ok, f_disk=f_disk, f_res=f_res if strat == "disc+res" else None, n_points=n, grid=list(bright.shape), pixel_mas=pix, smoke=args.smoke, seconds=time.time() - t0,
                   f_prim_pct=100 / tot, f_sec_pct=100 * f_sec / tot if has_sec else None, f_env_pct=100 * stars["f_env"] / tot,
                   f_bg_pct=100 * f_bg / tot if cfg.get("bg") else None, f_outer_pct=100 * f_outer / tot,
                   f_overresolved_pct=100 * (f_bg + f_outer) / tot, resid=resid, support_floor_mas=support_floor, **degen, **extra,
                   **{k_: v for k_, v in stars.items() if k_ in ("ud", "env_index", "bg_index", "sec_dra", "sec_ddec", "sec_index")})
    if companion:
        summary |= dict(companion=companion, companion_start=list(args.companion_start),
                        **companion_summary(model, bright_grid, pix, f_env + f_res, extra.get("scene_sd"), tot))
    if getattr(args, "geometry", None) is not None:
        summary |= dict(geometry_source="--geometry")
    if getattr(args, "white", None) is not None:
        summary |= dict(white_override=list(args.white))
    if getattr(args, "hole_scale", None) is not None:
        summary |= dict(hole_scale=args.hole_scale)
    if getattr(args, "companion_box", None) is not None:
        summary |= dict(companion_box_mas=args.companion_box)
    if getattr(args, "min_length_r", None) is not None:
        summary |= dict(min_length_r_beams=args.min_length_r)
    if has_sec:
        summary |= dict(sec_sep_mas=float(np.hypot(stars["sec_dra"], stars["sec_ddec"])),
                        sec_pa_deg=float(np.degrees(np.arctan2(stars["sec_dra"], stars["sec_ddec"])) % 360.0))
    print(json.dumps(summary, indent=1))
    json.dump(summary, open(path(out, cfg, "image.json"), "w"), indent=1)
    plot_lcurve(out, cfg)




# ----------------------------------------------------------------------------
# calibrate: does the gp-polar hyperparameter selection recover a known smooth disc? (design/gp_polar_hyperparameters.md)

CAL_LR_BEAMS = (0.5, 1.0, 2.0)
CAL_LT = (0.0625, 0.25, 1.0)
CAL_SIGMAS = (0.5, 1.0, 2.0, 4.0)
CAL_ARMS = ("white", "corr", "smear", "geo", "night", "chrom")
CAL_GEO_ERROR_DEG = (8.0, 8.0)  # geo arm: the polar GP deprojects with (inc, PA) this far off the truth's, a typical error of the real ring fits
CAL_CHROM_DALPHA = 2.0  # chrom arm: the ring's index is this much higher one width inside its centre than one width outside (§11)
CAL_CHROM_STAR = -1.0  # chrom arm: the truth's star index is SIND1 plus this (towards Rayleigh-Jeans); the fit keeps SIND1


def calib_tag(arm, m_max, seed=0):
    return f"calib_{arm}_m{'all' if m_max is None else m_max}_s{seed}"


def image_errors(img, truth, pix, bm, mask):
    """Shape errors of a reconstruction against the truth, both normalised to unit sum on ``mask``: relative rms error of the raw
    images and of both convolved with the comparison beam ``bm``, and the normalised cross-correlation of the convolved ones."""
    a, b = img / img[mask].sum(), truth / truth[mask].sum()
    ac, bc = np.asarray(convolve_beam(a, pix, bm)), np.asarray(convolve_beam(b, pix, bm))

    def rel(x, y):
        return float(np.linalg.norm((x - y)[mask]) / np.linalg.norm(y[mask]))

    return dict(nrmse_raw=rel(a, b), nrmse_beam=rel(ac, bc), ncc_beam=ncc(ac, bc, mask))


def simulate(cfg, files, s, truth, arm, key):
    """Simulated data from ``truth`` on ``files``' (u,v), errors and closure triangles, and the error scale to fit them with.

    * ``white``: Gaussian noise at the quoted errors times the target's MacKay factor ``s``, fitted with those errors (right by
      construction).
    * ``corr``: the target's virgil-vlti error model (white parts scaled by kind, per-frame gains on log|V| per telescope and
      baseline, closure-phase offsets; widths from its own pair test), fitted as the real data are: independent quoted errors with
      the MacKay factor of a pilot GP (``pilot_error_scale``, run by the caller: returned scale None).
    * ``smear``: white noise as for ``white``, but the truth seen through PIONIER's spectral channels (bandwidth smearing), which
      the pixel-image fits leave out.
    * ``geo``: noise as for ``white``; the caller fits with a wrong disc geometry (``CAL_GEO_ERROR_DEG``).
    * ``chrom``: noise as for ``white``; the caller makes the truth chromatic (``chromatic_truth``) and fits it grey.
    * ``night``: as ``corr``, plus the night terms of ``night_noise`` (a gain per (night, baseline), a closure offset per
      (night, triangle)), fitted the same way."""
    data0 = load_pionier(files)
    if arm in ("white", "geo", "chrom"):
        return data0.with_error_scale(s).with_model(truth, key=key), s
    if arm == "smear":
        sim = load_pionier(files, bandwidth=True).with_error_scale(s).with_model(truth, key=key)
        return data0.with_error_scale(s).set(["vis", "phi"], [sim.vis, sim.phi]), s
    sim = modelled_noise(data0, cfg, "night" if arm == "night" else "vlti").with_model(truth, key=key)
    return data0.set(["vis", "phi"], [sim.vis, sim.phi]), None


def representable_truth(g, geo, r_min, r_max, unit):
    """The smooth truth: the ``geometry_fit`` scene (stars, fluxes, spectral indices, first harmonic), its Gaussian ring moved
    where the polar GP can draw it. The real fits put some rings at their prior floor (AI Sco, EN TrA: radius 0.5 mas, a central
    blob), with flux inside the blanked hole, which no image of the pipeline can hold; a truth like that would test the model
    mismatch, not the hyperparameter choice. So the width is between half a beam and 0.15 of the polar grid's radial span, and the
    radius at least 2.5 widths from the hole and from ``r_max``; inclination and position angle are the polar GP's (``geo``)."""
    w = min(max(float(g.values["env.profile.sigma"]), 0.5 * unit), 0.15 * (r_max - r_min))
    r = float(np.clip(float(g.values["env.profile.radius"]), r_min + 2.5 * w, r_max - 2.5 * w))
    truth = g.model.set("env.profile.radius", r).set("env.profile.sigma", w).set("env.inc", geo["inc_deg"]).set("env.pa", geo["pa_deg"])
    return truth, dict(radius_mas=r, width_mas=w, inc_deg=geo["inc_deg"], pa_deg=geo["pa_deg"],
                       fit_radius_mas=float(g.values["env.profile.radius"]), fit_width_mas=float(g.values["env.profile.sigma"]),
                       h_re=float(np.ravel(g.values["env.harmonics.re"])[0]), h_im=float(np.ravel(g.values["env.harmonics.im"])[0]),
                       f_env=float(np.asarray(truth.env.flux.ratio)))


def chromatic_truth(truth, info, star_index, d_alpha=CAL_CHROM_DALPHA, d_star=CAL_CHROM_STAR):
    """The ``chrom`` arm's truth (design §11): ``truth``'s ring with an index falling linearly with radius, ``d_alpha`` higher one
    ring width inside its centre than one width outside (virgil's chromatic ``MultipoleDisk``, same F_lambda convention as
    ``PowerLaw``), about the ring's own index at its centre; and the star's index ``star_index + d_star``."""
    env, r, w = truth.env, float(info["radius_mas"]), float(info["width_mas"])
    disk = MultipoleDisk(env.profile, harmonics=env.harmonics, inc=env.inc, pa=env.pa, n_r=env.n_r, flux=float(np.asarray(env.flux.ratio)),
                         spectral_index=RadialLinear(float(np.asarray(env.flux.index)), -d_alpha / (2.0 * w), r), wavel0=L0)
    return truth.set("env", disk).set("star.flux.index", float(star_index) + d_star), dict(
        d_alpha=d_alpha, index_slope_per_mas=-d_alpha / (2.0 * w), star_index_truth=float(star_index) + d_star, star_index_fit=float(star_index))


def calibrate_stage(args, out):
    """Hyperparameter calibration of gp-polar on a smooth known disc with this target's own (u,v), errors and scene.

    The truth is ``representable_truth``: the stars of ``geometry_fit`` on the real (rescaled) data and a Gaussian ring with a first
    harmonic, so nothing in it
    is above azimuthal order 1. ``--arm`` sets the noise (``simulate``); ``--m-max`` the azimuthal cap (negative: no cap, the
    pre-cap grid). Every cell of a fixed grid (CAL_*: lengths from half a beam, l_theta from below the node spacing, sigma to 4) is
    fitted cold, on all the files and, for a cross-validation score, on the even-numbered files alone with chi2/pt on the odd ones.
    Written: ``<target>_calib_<arm>_m<cap>_s<seed>.json`` (every cell: log Z, chi2, CV chi2, image errors against the truth, fit record and
    loss history; the picks of the evidence, of CV and of the truth itself), an ``.npz`` of the images and a figure."""
    cfg = cfg_of(args)
    m_max = None if args.m_max is not None and args.m_max < 0 else (POLAR_M_MAX if args.m_max is None else args.m_max)
    tag = calib_tag(args.arm, m_max, args.seed)
    sc_file = scale_file(out, cfg, "current")
    if not sc_file.exists() and not args.smoke:  # never a made-up error scale or geometry in a real run
        raise SystemExit(f"calibrate needs the scale stage's output ({sc_file.name}) in OUTDIR")
    sc = json.load(open(sc_file)) if not args.smoke else dict(error_scale=1.0, geometry=dict(inc_deg=45.0, pa_deg=90.0))
    s, geo = float(sc["error_scale"]), sc["geometry"]
    files = load_files(cfg, args.files)
    real = load_pionier(files).with_error_scale(s)
    start, support, hole_mas = grid_setup(cfg, real, args)
    if not path(out, cfg, "parametric.json").exists():
        raise SystemExit("calibrate needs the parametric stage's output (parametric.json) in OUTDIR")
    g = geometry_fit(cfg, real, out, hole_mas, args)
    unit, pix = compare_beam(real).minor_mas, float(start.env.pixel_scale_mas)
    truth, truth_info = representable_truth(g, geo, hole_mas, float(cfg["half_mas"]), unit)
    if args.arm == "chrom":  # the fit below stays grey, with the star at cfg's index
        truth, chrom = chromatic_truth(truth, truth_info, cfg.get("img_star_index", -4.0))
        truth_info |= dict(chromatic=chrom)
    print("truth:", json.dumps(truth_info), flush=True)
    geo_fit = dict(geo)  # the geometry the polar GP deprojects with: the truth's, except in the geo arm
    if args.arm == "geo":
        geo_fit = dict(geo, inc_deg=float(min(geo["inc_deg"] + CAL_GEO_ERROR_DEG[0], 85.0)), pa_deg=float(geo["pa_deg"] + CAL_GEO_ERROR_DEG[1]))

    keys = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    sets = {}
    for name, fs, k in (("all", files, keys[0]), ("even", files[0::2], keys[1]), ("odd", files[1::2], keys[2])):
        sets[name] = simulate(cfg, fs, s, truth, args.arm, k)
    env = Image(start.env.log_brightness, start.env.pixel_scale_mas, support=support, flux=start.env.flux)
    scene = sparco_scene(cfg, env, f_env=float(np.asarray(env.flux)))
    priors = image_priors(scene) | scene_priors(cfg, scene)
    if sets["all"][1] is None:  # corr, night: the MacKay factor of a pilot GP on the simulated data, as the real pipeline does
        s_c, pilot_chi2 = pilot_error_scale(scene, start, support, priors, sets["all"][0], args)
        print(f"{args.arm} arm: pilot chi2/pt {pilot_chi2:.3f}, error scale {s_c:.3f}", flush=True)
        sets = {k: (d.with_error_scale(s_c), s_c) for k, (d, _) in sets.items()}
    data, s_fit = sets["all"]

    axes = [[m * unit for m in CAL_LR_BEAMS if m * unit >= pix], list(CAL_LT), list(CAL_SIGMAS)]
    common = (cfg, scene, start.env, support, priors)
    cells, axes = polar_cells(*common, data, args, geo_fit, hole_mas, m_max=m_max, axes=axes, extend=False)[:2]  # smoke trims the axes
    cells_cv = polar_cells(*common, sets["even"][0], args, geo_fit, hole_mas, m_max=m_max, axes=axes, extend=False)[0]

    n = np.shape(env.log_brightness)[0]
    truth_img = np.asarray(truth.env.render(npix=n, fov_mas=n * pix))
    mask = np.asarray(support, bool)
    bm = compare_beam(real)
    import itertools  # noqa: PLC0415

    keys_ = list(itertools.product(*axes))
    rows = cell_records(cells, axes)
    for row, v in zip(rows, keys_):
        row |= image_errors(render_env(cells[v][0].model, "gp-polar", (n, n), pix), truth_img, pix, bm, mask)
        row["cv_chi2"] = float(chi2r(cells_cv[v][0].model, sets["odd"][0])[0])
        row["cv_record"] = {k: cells_cv[v][3][k] for k in ("steps", "stop", "converged", "seconds")}
    picks = dict(evidence=int(np.argmax([r["log_z"] for r in rows])), cv=int(np.argmin([r["cv_chi2"] for r in rows])),
                 truth=int(np.argmin([r["nrmse_beam"] for r in rows])))
    order = np.argsort([r["nrmse_beam"] for r in rows])
    rank = {int(i): k + 1 for k, i in enumerate(order)}  # 1 = closest to the truth
    corner = (min(axes[0]), max(axes[2]))  # the failure corner of the real fits: the smallest l_r with the largest sigma
    summary = {k: dict(cell=i, rank_vs_truth=rank[i], rank_fraction=rank[i] / len(rows),
                       at_corner=bool((rows[i]["length_r_mas"], rows[i]["sigma"]) == corner), **{q: rows[i][q] for q in ("length_r_mas", "length_theta", "sigma", "log_z", "chi2",
                                                                                   "cv_chi2", "nrmse_beam", "nrmse_raw", "ncc_beam")})
               for k, i in picks.items()}
    res = dict(target=cfg["key"], arm=args.arm, m_max=m_max, seed=args.seed, error_scale_real=s, error_scale_fit=s_fit,
               beam_mas=unit, pixel_mas=pix, axes=dict(length_r_mas=axes[0], length_theta=axes[1], sigma=axes[2]),
               truth=truth_info, truth_fit=geometry_of(g, real), geometry_fitted_with=geo_fit, n_cells=len(rows), n_unconverged=int(sum(not r["converged"] for r in rows)),
               picks=summary, cells=rows, smoke=args.smoke)
    json.dump(res, open(path(out, cfg, f"{tag}.json"), "w"), indent=1)
    np.savez(path(out, cfg, f"{tag}.npz"), truth=truth_img, pixel_mas=pix,
             **{f"pick_{k}": render_env(cells[keys_[i]][0].model, "gp-polar", (n, n), pix) for k, i in picks.items()})
    print(json.dumps(dict(picks=summary, n_unconverged=res["n_unconverged"]), indent=1), flush=True)
    plot_calibration(out, cfg, tag)


def plot_calibration(out, cfg, tag):
    """Truth | the evidence's pick | CV's pick | the cell closest to the truth (raw images), and log Z against the image error."""
    res, z = json.load(open(path(out, cfg, f"{tag}.json"))), np.load(path(out, cfg, f"{tag}.npz"))
    pix, n = float(z["pixel_mas"]), z["truth"].shape[0]
    ext = [n * pix / 2, -n * pix / 2, -n * pix / 2, n * pix / 2]
    fig, ax = plt.subplots(1, 5, figsize=(20, 4))
    for a, (k, title) in zip(ax, (("truth", "truth"), ("pick_evidence", "evidence"), ("pick_cv", "cross-validation"), ("pick_truth", "closest to truth"))):
        a.imshow(z[k] / z[k].max(), origin="upper", extent=ext, cmap="inferno")
        p = res["picks"].get(k.removeprefix("pick_")) if k != "truth" else None
        a.set_title(title if p is None else f"{title}: σ {p['sigma']:g}, ℓ_r {p['length_r_mas']:.2f}, ℓ_θ {p['length_theta']:g}\n"
                                              f"error {p['nrmse_beam']:.2f} (rank {p['rank_vs_truth']})", fontsize=9)
        a.set_xlim(cfg["half_mas"], -cfg["half_mas"])
        a.set_ylim(-cfg["half_mas"], cfg["half_mas"])
    cells = res["cells"]
    lz = np.array([c["log_z"] for c in cells])
    sc = ax[4].scatter([c["nrmse_beam"] for c in cells], lz - lz.max(), c=np.log2([c["sigma"] for c in cells]), cmap="viridis")
    ax[4].set_xlabel("image error (beam-convolved)")
    ax[4].set_ylabel("ln Z − max")
    ax[4].set_yscale("symlog")
    fig.colorbar(sc, ax=ax[4], label="log2 σ")
    orders = "all" if res["m_max"] is None else f"<= {res['m_max']}"
    fig.suptitle(f"{cfg['name']}: {res['arm']} noise, azimuthal orders {orders}")
    fig.tight_layout()
    fig.savefig(path(out, cfg, f"{tag}.png"), dpi=110)
    plt.close(fig)

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


def central_peak(img, scale, rmax=2.5, rfit=0.8):
    """Position of the point-like source near the star: the brightest pixel within ``rmax`` mas, refined by the
    flux-weighted centroid within ``rfit`` mas of it. Returns (separation mas, PA East of North in degrees)."""
    ny, nx = img.shape
    yy, xx = np.mgrid[:ny, :nx]
    dx, dy = -(xx - (nx - 1) / 2.0) * scale, ((ny - 1) / 2.0 - yy) * scale  # East (column 0), North (row 0)
    r = np.hypot(dx, dy)
    k = np.argmax(np.where(r < rmax, img, -np.inf))
    x0, y0 = dx.ravel()[k], dy.ravel()[k]
    w = np.where(np.hypot(dx - x0, dy - y0) < rfit, img, 0.0)
    cx, cy = (w * dx).sum() / w.sum(), (w * dy).sum() / w.sum()
    return float(np.hypot(cx, cy)), float(np.degrees(np.arctan2(cx, cy)) % 360.0)


def ncc(a, b, mask):
    a, b = a[mask] - a[mask].mean(), b[mask] - b[mask].mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def bitmap_published(cfg):
    """The published bitmap image (peak-normalised, North up East left), its pixel scale (mas) and drawn beam."""
    z = np.load(LITERATURE / cfg["bitmap"])
    return np.asarray(z["image"], float), float(z["pixel_mas"]), json.loads(str(z["beam"])), z


def compare_beam(data):
    """The comparison beam: a circular Gaussian of FWHM lambda_min / (2 B_max), from this target's data.

    Both images are convolved with it (the virgil data beam, ``beam(data)``, only sets the hole radius).
    """
    from virgil.imaging import Beam

    b_max = float(np.max(np.hypot(np.asarray(data.u), np.asarray(data.v))))
    fwhm = float(np.min(np.asarray(data.wavel))) / (2.0 * b_max) * 180.0 / np.pi * 3.6e6
    return Beam(fwhm, fwhm, 0.0)


def compare_bitmap_stage(args, out):
    """Compare with an image that exists only as a bitmap: morphology at the paper's drawn beam."""
    from virgil.imaging import Beam

    cfg = cfg_of(args)
    data, _, _, _ = load_scaled(cfg, args, out)
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
    cp_ours, cp_pub = central_peak(ours_c, scale), central_peak(pub, scale)
    flips = {name: ncc(ours_c, im, mask) for name, im in
             {"as_published": pub, "mirrored_East-West": pub[:, ::-1], "mirrored_North-South": pub[::-1, :]}.items()}
    rbin = np.round(np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2)).astype(int)  # azimuthal-average control
    prof = np.bincount(rbin.ravel(), pub.ravel()) / np.bincount(rbin.ravel())
    ncc_controls = {"mirrored_East-West": flips["mirrored_East-West"], "rotated_180": ncc(ours_c, pub[::-1, ::-1], mask),
                    "azimuthal_average_of_published": ncc(ours_c, prof[rbin], mask)}
    img_json = json.load(open(path(out, cfg, "image.json")))
    res = dict(
        ncc_zero_shift=ncc(ours_c, pub, mask), ncc_controls=ncc_controls, ncc_best_shift=shifts[best],
        best_shift_mas=dict(dy=best[0] * step * scale, dx=best[1] * step * scale), pixel_mas=scale,
        beam=pb, virgil_data_beam=dict(major_mas=beam(data).major_mas, minor_mas=beam(data).minor_mas,
                                       pa_deg=beam(data).pa_deg),
        ours=m_ours, published=m_pub, ncc_orientation=flips,
        central_source=dict(ours=dict(sep_mas=cp_ours[0], pa_deg=cp_ours[1]),
                            published=dict(sep_mas=cp_pub[0], pa_deg=cp_pub[1])),
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
    per_strategy(lambda a, o, strat: compare_one(a, o), args, out)


def compare_one(args, out):
    cfg = cfg_of(args)
    if cfg.get("bitmap"):
        return compare_bitmap_stage(args, out)
    data, _, _, _ = load_scaled(cfg, args, out)  # the errors as rescaled for the fits
    ours = np.load(path(out, cfg, "image.npz"))
    pub, noise, pub_scale, hdr = published_image(cfg)
    n = pub.shape[0]
    grid = to_grid(ours["brightness"], float(ours["pixel_mas"]), pub_scale, n)
    grid = grid / grid.sum()
    bm = compare_beam(data)
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
        beam_convention="circular Gaussian, FWHM lambda_min / (2 B_max)",
        virgil_data_beam=dict(major_mas=beam(data).major_mas, minor_mas=beam(data).minor_mas, pa_deg=beam(data).pa_deg),
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


def reg_label(out, cfg):
    j = json.load(open(path(out, cfg, "image.json")))
    if j.get("regulariser") == "GaussianField":
        return f"GP, σ = {j['gp_sigma']:g}, ℓ = {j['gp_length_mas']:.2f} mas"
    if j.get("regulariser") == "MultipoleDisk":
        return f"disc + residual GP, σ = {j['gp_sigma']:g}, ℓ = {j['gp_length_mas']:.2f} mas"
    if str(j.get("regulariser", "")).startswith("MultipoleField"):  # capped fields read "MultipoleField, azimuthal orders <= m"
        return f"polar GP, σ = {j['gp_sigma']:g}, ℓ_r = {j['gp_length_r_mas']:g} mas, ℓ_θ = {j['gp_length_theta']:g} rad"
    return "MaxEntropy, discrepancy"


def plot_images(out, cfg):
    """The page's headline figure: published (Zenodo FITS) and virgil, one style, one beam."""
    from matplotlib.patches import Ellipse

    if cfg.get("bitmap"):
        return plot_images_bitmap(out, cfg)
    r = json.load(open(path(out, cfg, "compare.json")))
    z = np.load(path(out, cfg, "compare_images.npz"))
    s, n, half = float(z["pixel_mas"]), z["ours"].shape[0], cfg["half_mas"]
    ext = [n * s / 2, -n * s / 2, -n * s / 2, n * s / 2]
    b = r["beam"]
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 5.2), sharey=True)
    for a, key, title in ((ax[0], "published_conv", "ORGANIC (De Prins et al. 2026)"),
                          (ax[1], "ours_conv", f"virgil ({reg_label(out, cfg)})")):
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
    for a, title in zip(ax, (f"as published ({cfg['pub_label']})", f"virgil ({reg_label(out, cfg)})")):
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


def plot_gp_grid(out, cfg, z):
    """The (sigma, length) grid of a GP image: Laplace log evidence and chi2 per point, the chosen cell crossed."""
    log_z, chi2 = z["gp_log_z"], z["gp_chi2"]
    sig, ln, (bi, bj) = z["gp_sigmas"], z["gp_lengths_mas"], z["chosen"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for a, vals, title, fmt in ((ax[0], log_z - log_z.max(), "log evidence - max", "{:.0f}"), (ax[1], chi2, "χ² per point (errors as given)", "{:.2f}")):
        im = a.imshow(vals, origin="lower", cmap="viridis", aspect="auto")
        a.set(xticks=range(len(sig)), xticklabels=[f"{x:g}" for x in sig], yticks=range(len(ln)),
              yticklabels=[f"{x:.2f}" for x in ln], xlabel=str(z["gp_xlabel"]) if "gp_xlabel" in z else "σ", ylabel=str(z["gp_ylabel"]) if "gp_ylabel" in z else "ℓ (mas)", title=title)
        a.plot(bj, bi, "wx", ms=12, mew=2)
        mid = 0.5 * (vals.min() + vals.max())
        for (i, j), v in np.ndenumerate(vals):
            a.text(j, i, fmt.format(v), ha="center", va="center", color="k" if v > mid else "w", fontsize=8)
        plt.colorbar(im, ax=a)
    fig.tight_layout()
    fig.savefig(path(out, cfg, "lcurve.png"), dpi=130)
    plt.close(fig)


def plot_lcurve(out, cfg):
    z = np.load(path(out, cfg, "image.npz"))
    if "gp_log_z" in z:
        return plot_gp_grid(out, cfg, z)
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


# ----------------------------------------------------------------------------
# table: the page's parameter table (paper PMOIRED fit | virgil parametric fit | difference in sigma)


def _fmt(v, e=None):
    return f"{v:.3g}" if e is None else f"{v:.3g} ± {e:.2g}"


STRATEGY_TITLES = {"mem": "S0 MaxEntropy", "gp-iso": "S1 GP isotropic", "gp-aniso": "S2 GP anisotropic", "gp-polar": "S3 GP polar",
                   "disc+res": "S4 disc + residual"}
CANONICAL = ("image.json", "image.npz", "compare.json", "compare_images.npz", "compare.png", "images.png", "fit.png", "lcurve.png")


def select_stage(args, out):
    """Collect the strategies of a target: the table (``<t>_strategies.json``), the strip ``<t>_strategies.png`` (published | S0..S4 at the
    comparison beam), and the evidence-preferred strategy's files copied to the canonical ``<t>_*`` names the page links to.

    The headline is HEADLINE (the polar GP, chosen by eye by Ben on IW Car: the strategies differ a lot in latent dimension and Occam
    factor and chi2/pt sits near 0.5-0.6, so the Laplace evidence is not followed blindly); ``evidence_preferred`` records the strategy
    with the largest Laplace log evidence among the GP and disc strategies (MaxEntropy has none), and dlogz is relative to it."""
    import shutil  # noqa: PLC0415

    cfg = cfg_of(args)
    rows = {}
    for st in STRATEGIES:
        c = cfg | dict(key=f"{cfg['key']}_{st}")
        if not (path(out, c, "image.json").exists() and path(out, c, "compare.json").exists()):
            continue
        i, r = json.load(open(path(out, c, "image.json"))), json.load(open(path(out, c, "compare.json")))
        rows[st] = dict(strategy=st, ncc=r["ncc_zero_shift"], ncc_best_control=max(r["ncc_controls"].values()), chi2_per_point=i["chi2_per_point"],
                        log_evidence=i.get("log_evidence"), f_prim_pct=i["f_prim_pct"], f_sec_pct=i.get("f_sec_pct"), f_env_pct=i["f_env_pct"],
                        f_bg_pct=i.get("f_bg_pct"), f_outer_pct=i.get("f_outer_pct"), f_overresolved_pct=i.get("f_overresolved_pct"),
                        env_index=i.get("env_index"), error_scale=i["error_scale"], gp_on_edge=i.get("gp_on_edge"), regulariser=i.get("regulariser"),
                        hole_mas=i["hole_mas"], seconds=i["seconds"])
    if not rows:
        raise FileNotFoundError("no strategy outputs")
    evid = {k: v["log_evidence"] for k, v in rows.items() if v["log_evidence"] is not None}
    ev_best = max(evid, key=evid.get) if evid else "mem"
    best = HEADLINE if HEADLINE in rows else ev_best
    for k, v in rows.items():
        v["dlogz"] = (v["log_evidence"] - evid[ev_best]) if v["log_evidence"] is not None else None
    json.dump(dict(preferred=best, evidence_preferred=ev_best, strategies=rows), open(out / f"{cfg['key']}_strategies.json", "w"), indent=1)
    for suf in CANONICAL:
        shutil.copyfile(path(out, cfg | dict(key=f"{cfg['key']}_{best}"), suf), path(out, cfg, suf))
    # the strip
    names = list(rows)
    zs = {k: np.load(path(out, cfg | dict(key=f"{cfg['key']}_{k}"), "compare_images.npz")) for k in names}
    first = zs[names[0]]
    s, n, half = float(first["pixel_mas"]), first["ours"].shape[0], cfg["half_mas"]
    ext = [n * s / 2, -n * s / 2, -n * s / 2, n * s / 2]
    fig, ax = plt.subplots(1, len(names) + 1, figsize=(3.1 * (len(names) + 1), 3.6), sharey=True)
    panels = [("published_conv", "published", first)] + [("ours_conv", f"{STRATEGY_TITLES[k]}\nNCC {rows[k]['ncc']:.2f}, χ²/pt {rows[k]['chi2_per_point']:.2f}"
                                                           + (f", ΔlnZ {rows[k]['dlogz']:.0f}" if rows[k]["dlogz"] is not None else ""), zs[k]) for k in names]
    for a, (key, title, z) in zip(ax, panels):
        im = z[key]
        a.imshow(im / im.max(), origin="upper", extent=ext, cmap="magma", vmin=0, vmax=1)
        a.set_xlim(half, -half)
        a.set_ylim(-half, half)
        a.plot(0, 0, "*", c="gold", ms=8, mec="k")
        a.set_title(title + (" (preferred)" if key == "ours_conv" and title.startswith(STRATEGY_TITLES[best]) else ""), fontsize=8)
        a.set_xlabel("Δα (mas)", fontsize=8)
    ax[0].set_ylabel("Δδ (mas)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{cfg['key']}_strategies.png", dpi=130)
    plt.close(fig)
    print("preferred strategy:", best, json.dumps({k: (round(v["ncc"], 3), round(v["chi2_per_point"], 3), v["dlogz"]) for k, v in rows.items()}))


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
        ("Rim modulation phase (°)", "psi_mod", par["az_pa_deg"] - par["rim_pa_deg"], 360.0),  # PMOIRED az projang = az_pas - pa
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


STAGES = dict(inspect=inspect_stage, parametric=parametric_stage, scale=scale_stage, image=image_stage, calibrate=calibrate_stage,
              compare=compare_stage, select=select_stage, replot=replot_stage, table=table_stage)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stage", choices=list(STAGES))
    p.add_argument("outdir", type=Path)
    p.add_argument("--target", default="IW_Car", choices=list(TARGETS))
    p.add_argument("--smoke", action="store_true", help="tiny run: three files, coarse grid, few steps")
    p.add_argument("--reuse-converged", action="store_true",
                   help="skip a target x strategy whose earlier image.json converged (recorded in <key>_<strategy>_reused.json); "
                        "off by default, so a gate run always refits")
    p.add_argument("--strategy", default="all", choices=("all",) + STRATEGIES, help="imaging strategy for the image and compare stages")
    p.add_argument("--star-index", type=float, default=None, help="one-off: stellar power-law index of the image scene (default -4), written as <target>_si<n>_*")
    p.add_argument("--bg-index", type=float, default=None, help="one-off: fix the over-resolved background's index (default free), written as <target>_<strategy>_bg<n>_*")
    p.add_argument("--noise", default="current", choices=("current", "vlti", "night", "nightw"),
                   help="error model: current (independent errors, MacKay error_scale), vlti (virgil-vlti PIONIER: V2 x 0.32, T3 x 0.65, "
                        "gains per telescope and baseline, closure offsets) or night (vlti plus a gain per (night, baseline) and a closure "
                        "offset per (night, triangle)); written as <target>_<strategy>_<noise>_*")
    p.add_argument("--binary", default="paper", choices=BINARY_MODES,
                   help="gp-polar: the binary model (design §12): paper (B1), free (secondary position, flux and index), motion (plus linear motion of both stars against the disc), epoch (plus an offset per night); written as ..._bin<mode>_*")
    p.add_argument("--chrom", action="store_true",
                   help="gp-polar: add a chromatic ring (index linear in radius) and free the star index (design §11); written as ..._chrom_*")
    p.add_argument("--wide-field", action="store_true",
                   help="image, support and comparison half-widths of WIDE_FIELD_MAS (15 mas) in place of the target's own "
                        "(design §12, when PS Gem's was 5 mas); written as <target>_<strategy>_..._wide_*")
    p.add_argument("--unbounded", action="store_true",
                   help="gp-polar: the evidence over the whole grid, without the prior bounds (diagnostic); written as ..._unbounded_*")
    p.add_argument("--companion", default=None, choices=("fixed", "free"),
                   help="gp-polar: add a point companion (flux ratio and index free) at --companion-start: held there (fixed) or free within "
                        f"+-{COMP_FREE_RADIUS_MAS:g} mas, the better of the start and its mirror image kept by scene evidence (free); written as ..._comp<arm>_*")
    p.add_argument("--companion-start", type=pair_arg, default=None, metavar="E,N",
                   help="--companion: the position (East, North mas from the primary), fixed or the start")
    p.add_argument("--companion-box", type=float, default=None, metavar="MAS",
                   help="--companion free: confine the position to +-MAS about --companion-start (no mirror refit: start a second run at the mirror "
                        "position); written as ..._box<mas>_*")
    p.add_argument("--min-length-r", type=float, default=None, metavar="BEAMS",
                   help="gp-polar: no radial length below this many beams, so the image is smooth and a compact source needs a companion; "
                        "written as ..._minlr<beams>_*")
    p.add_argument("--hole-scale", type=float, default=None, help="multiply the hole around the star(s) by this (0.5: half); written as ..._holex<f>_*")
    p.add_argument("--geometry", type=pair_arg, default=None, metavar="INC,PA",
                   help="gp-polar: deproject with this inclination and position angle (deg) in place of the scale stage's fit (no scale stage "
                        "needed); written as ..._geo<inc>-<pa>_*")
    p.add_argument("--white", type=pair_arg, default=None, metavar="V2,T3",
                   help="--noise vlti|night: the V^2 and closure white error scales, in place of the pair test's lower bounds (measure them from "
                        "the channel scatter, review section 5); written as ..._white<v2>-<t3>_*")
    p.add_argument("--mean", default="blob", choices=("blob", "flat"),
                   help="GaussianField / MultipoleField mean: the Gaussian start blob, or flat inside the support; written as ..._flat_*")
    p.add_argument("--cold", action="store_true", help="never warm-start the scene parameters from an earlier <target>_image.json")
    p.add_argument("--files", type=int, default=None, help="use only the first N files")
    p.add_argument("--arm", default="white", choices=CAL_ARMS, help="calibrate: noise of the simulated data (see simulate)")
    p.add_argument("--m-max", type=int, default=None, help=f"calibrate: azimuthal cap of the polar GP (default {POLAR_M_MAX}; negative: none)")
    p.add_argument("--seed", type=int, default=0, help="calibrate: noise seed")
    args = p.parse_args()
    check_options(args)
    if args.smoke and args.files is None:
        args.files = 3
    args.outdir.mkdir(parents=True, exist_ok=True)
    STAGES[args.stage](args, args.outdir)


if __name__ == "__main__":
    sys.exit(main())
