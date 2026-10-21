"""Stage C2: blind maximum-entropy images of the imaging contests' data.

    python scripts/contest_images.py --list
    python scripts/contest_images.py --task N [--data DIR] [--out DIR] [--smoke]

One task is one dataset (TASKS below). It follows virgil's documented RML
recipe (imaging tutorial, part 2): a starting image sized from the data
(`starting_image`), then an L-curve of maximum-entropy fits from strong to
weak regularisation (`l_curve`), with the weight chosen by the discrepancy
principle (χ² = N) and the L-curve corner. It writes, into --out:

- <label>.npz: weights, χ² (total and per point), penalties, every image
  on its native grid, the pixel scale, the beam, and the chosen weights;
- <label>.png: the L-curve, the starting image and the images at the corner
  and discrepancy weights, East left and North up, beam shown;
- <label>.txt: the run summary and virgil's `diagnose` report.

The settings use only what contestants had before the deadline (TASKS;
manifest "presubmission"), never the truths or parameters published
afterwards. A field given by the contest is used as given; otherwise it is
chosen from the data and grown while that lowers χ² (run()). An analytic
central star is used where the contest described a star, or where the data
show an unresolved source dominates (each task's "star_source").
The images are compared with the published entries in
docs/plan_imaging_contests.md (Stage C2). This is heavy: run it on OzSTAR
(ozstar_scripts job `contest_imaging`), not on a laptop; --smoke is a
seconds-long check that the script runs.
"""

import argparse
import dataclasses
import json
import pathlib
import sys
import tempfile
import time

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

import virgil  # noqa: E402
import virgil.models as vm  # noqa: E402
from virgil.fields import GaussianField  # noqa: E402
from virgil.imaging import Beam as vm_beam  # noqa: E402
from virgil.imaging import (  # noqa: E402
    clean,
    error_scale,
    log_evidence,
    Centroid,
    LCurve,
    MaxEntropy,
    StarletL1,
    LogSum,
    TSV,
    TV,
    beam,
    diagnose,
    image_priors,
    l_curve,
    starting_image,
)
# Private virgil helpers, deliberately: this script is pinned to a virgil commit
# (the summary line records it), so a rename upstream shows up as an import error
# here, not as a silent change. _reference is how virgil's l_curve evaluates a
# penalty on a fitted model (used to put Adam refits on the same footing);
# _default_weight_ranges is ensemble's per-data-point weight ranges.
from virgil._utils import _reference  # noqa: E402
from virgil.ensemble import _default_weight_ranges  # noqa: E402
from virgil.fitting import fit  # noqa: E402
from virgil.likelihood import whitened_residuals  # noqa: E402
from virgil.oidata import OIData  # noqa: E402
from virgil.plotting import plot_model  # noqa: E402
from virgil.spectra import PowerLaw  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from crosscheck import starts as linear_starts  # noqa: E402

# One task per dataset. Everything here is what contestants had before they
# submitted (contests/manifest.yml, "presubmission"); nothing revealed in the
# papers afterwards is used (docs/plan_imaging_contests.md, "Initialisation").
#   field:   field of view in mas, when the contest gave one (with its source);
#            None: chosen from the data, enlarged while flux reaches the edge
#   prior:   ("gauss", FWHM mas): the MaxEnt default image and the start
#   star:    an analytic star at the centre, with star_source saying why: the
#            contest's description, or the data (an unresolved source dominates)
#   wavel:   (min, max) in µm: keep only these channels (flag the rest)
TASKS = [
    # 2004: released blind; nothing about field or target.
    dict(label="2004_data1", files=["2004/2004-data1.fits"]),
    # The contest page listed "compact source with extended envelope" among the
    # possible targets, so data2's members also try a halo.
    dict(label="2004_data2", files=["2004/2004-data2.fits"], halo_members=True),
    # 2006: the clue image (the model at 10 mas resolution) spans 106 mas; the
    # rules allow it only for the field of view, not as a prior.
    dict(label="2006_disk", files=[f"2006/2006-03-0{n}.fits" for n in (3, 4, 5, 6)],
         field=106.0, field_source="2006 clue.10.fits"),
    # 2008: "tapered with a 15 mas FWHM Gaussian" (readme); grey per band.
    *[dict(label=f"2008_{o}_{b}", files=[f"2008/2008-Contest{k}_{b}.oifits"], field=30.0,
           field_source="2008 readme: 15 mas FWHM taper", prior=("gauss", 15.0))
      for k, o in ((1, "agb"), (2, "agn")) for b in "JHK"],
    # 2010: a bright source; the grey category was judged as separate images of
    # Low HK channels 1-10 (H) and 11-20 (K) (Contest10.html).
    # Challenge.txt gave each channel's SED, so a star and an image with
    # different spectra (SPARCO) is information contestants had.
    dict(label="2010_lowH", files=["2010/Mystery-Low_HK.oifits"], wavel=(1.4, 1.9), sparco=True, halo_members=True),
    # 2022: no description found, so the star is inferred from the data, as a
    # contestant could: both are dominated by an unresolved source.
    dict(label="2022_gravity", files=["2022/c_imaging_contest1.fits"], star=True,
         star_source="data: |V| ~0.7 on the shortest baselines (V² median 0.48), so an unresolved source carries most of the flux"),
    dict(label="2022_ami", files=["2022/c_imaging_contest2.fits"], star=True,
         star_source="data: V² ~0.9 on every baseline"),
    # 2024: Obj1 "a hot star with an environment", Obj2 "a young star" with a
    # suspected companion (contest page). Grey per instrument is only a first
    # look: the rules ask for cubes (Stage C3).
    dict(label="2024_obj1_pionier", files=["2024/Obj1_PIONIER_1.5-1.8.fits"], star=True,
         star_source="contest page: a hot star with an environment"),
    dict(label="2024_obj1_gravity", files=["2024/Obj1_GRAVITY_2.0-2.5.fits"], star=True,
         star_source="contest page: a hot star with an environment"),
    dict(label="2024_obj2_pionier", sparco=True, halo_members=True, files=["2024/Obj2_PIONIER_1.5-1.8.fits"], star=True,
         star_source="contest page: a young star"),
    dict(label="2024_obj2_gravity", sparco=True, halo_members=True, files=["2024/Obj2_GRAVITY_2.0-2.5.fits"], star=True,
         star_source="contest page: a young star"),
    # New tasks go at the end, so that earlier indices keep their meaning.
    dict(label="2010_lowK", files=["2010/Mystery-Low_HK.oifits"], wavel=(1.9, 2.6), sparco=True, halo_members=True),
    # 2018: "a young star's disk, with a planet" (readme): a star plus an image.
    dict(label="2018_disk", star=True, star_source="readme: a young star's disk", files=[
        "2018/Aspro2_Altair_MIRC_6T_1_47493-1_75256-8ch_S1-S2-E1-E2-W1-W2_2018-08-02_FAKE.fits",
        "2018/Aspro2_Altair_PIONIER_1_533-1_772-6ch_A0-B2-C1-D0_2018-08-02_FAKE.fits",
        "2018/Aspro2_Altair_PIONIER_1_533-1_772-6ch_A0-G1-J2-J3_2018-08-02_FAKE.fits",
        "2018/Aspro2_Altair_PIONIER_1_533-1_772-6ch_D0-G2-J3-K0_2018-08-02_FAKE.fits"]),
    # 2024 MATISSE L and N bands (the files contestants had; grey per band).
    dict(label="2024_obj1_matisseL", files=["2024/Obj1_MATISSE_2.9-4.2.fits"], star=True,
         star_source="contest page: a hot star with an environment"),
    dict(label="2024_obj1_matisseN", files=["2024/Obj1_MATISSE_8-13.fits"], star=True,
         star_source="contest page: a hot star with an environment"),
    dict(label="2024_obj2_matisseL", files=["2024/Obj2_MATISSE_2.9-4.0.fits"], star=True,
         star_source="contest page: a young star"),
    dict(label="2024_obj2_matisseN", files=["2024/Obj2_MATISSE_8-13.fits"], star=True,
         star_source="contest page: a young star"),
    # John Young's data (docs/plan_imaging_contests.md, C4b). 2012: CHARA/MIRC-6T,
    # H band, 8 channels. The rules page described no target and the readme was
    # never archived, so field and star come from the data.
    dict(label="2012_alp", files=["2012/2012_Alp_Fak_MIRC6T_LowH.oifits"]),
    dict(label="2012_bet", files=["2012/2012_Bet_Fak_MIRC6T_LowH.oifits"]),
    # 2014: real PIONIER data on VY CMa, a red supergiant (named in OI_TARGET).
    # R Car (2014/2014_R_CAR_all.fits) is left out for now: virgil 0.3.0's
    # reader refuses it (a closure triangle whose baseline is in no V² table
    # at the same time).
    dict(label="2014_vycma", files=["2014/2014_VY_CMA_all.fits"]),
    # 2016: the contest page's 9 May 2016 update suggested the fields (FOV
    # 50-60 mas for Object 1, GRAVITY K; 0.4 arcsec for Object 2, MATISSE N).
    # Grey first looks; the rules ask for chromatic images (Stage C3).
    dict(label="2016_obj1", files=["2016/2016_Object1-v3.oifits"], field=60.0,
         field_source="2016 contest page, 9 May 2016: FOV 50-60 mas"),
    dict(label="2016_obj2", files=["2016/2016_Object2-v3.fits"], field=400.0,
         field_source="2016 contest page, 9 May 2016: FOV 0.4 arcsec"),
]

# 2010 Low HK, one task per channel (the winners' per-channel imaging): channel k is
# centred 1.521375 + 0.0475 (k - 1) µm, window ±0.019 (OI_WAVELENGTH; width 0.0475); labels 2010_ch01..2010_ch20.
# Run with --prior gp --init clean --systematics block_scales (docs/contest_2010_calibration.md).
TASKS += [dict(label=f"2010_ch{k:02d}", files=["2010/Mystery-Low_HK.oifits"],
               wavel=(1.502375 + 0.0475 * (k - 1), 1.540375 + 0.0475 * (k - 1))) for k in range(1, 21)]

WEIGHTS = np.logspace(4.0, 0.0, 13)  # strong to weak
MAX_PIX = 256  # pixels on a side; larger fields get coarser pixels
JACOBIAN_BUDGET = 2.0e8  # data points x pixels, about 3 GB per complex128 Jacobian
CLEAN_MAX_PIX = 65  # CLEAN runs on at most this grid (see setup())
FLUX_FLOOR = 1e-4  # lower bound of the log-uniform (Jeffreys) flux priors, relative to the star
GROWTHS = 3  # data-chosen fields grow at most three times (see run())


def image_of(model, star=None, halo=False):
    """The Image in ``model``: ``env`` in a System (a star, halo or point
    sources beside it), else the model itself."""
    return model.env if isinstance(model, vm.System) else model


# Round 2 (contests/fit_plan.yml): regularisers beside MaxEnt, chosen by the
# discrepancy principle on the quoted errors as MaxEnt is. Their penalties
# differ in scale by orders of magnitude (TSV is quadratic in unit-sum pixel
# steps), so MaxEnt's fixed WEIGHTS do not serve. The sweep follows virgil's
# design/regulariser_weight_selection.md: a weight per data point, over the
# ranges of virgil.ensemble's defaults (fitted to the corners of the 60
# contest-bench datasets), times the number of independent data N, 12 log-spaced
# points from strong to weak.
REGULARISERS = {"mem": None, "tv": TV, "tsv": TSV, "starlet": StarletL1, "logsum": LogSum}
REG_POINTS = 12
# virgil.ensemble._default_weight_ranges(): per data point. LogSum has no entry
# there; it takes StarletL1's range (both are sparsity penalties of O(1) on a
# unit-sum image).
WEIGHT_RANGES = {**_default_weight_ranges(), "logsum": _default_weight_ranges()["starlet"]}


def regulariser(name, weight, path, q=None):
    """The image penalty ``name`` (REGULARISERS) at ``weight``."""
    if name == "mem":
        return MaxEntropy(weight, prior=q, path=path)
    return REGULARISERS[name](weight, path=path)


def off_bounds(values, priors, edge=1e-3):
    """``values`` with each scalar parameter that has a Uniform or LogUniform
    prior moved to at least ``edge`` of its (log) range from the bounds.
    A fit can end on a bound (a faint Gaussian halo's flux, a width at its
    floor); a warm start from there is infinite in virgil's unconstrained
    coordinates and the next fit is NaN from step 0."""
    out = dict(values)
    for k, prior in priors.items():
        if isinstance(prior, (dist.Uniform, dist.LogUniform)) and k in out and np.ndim(out[k]) == 0:
            out[k] = jnp.asarray(inside(prior, out[k], edge))
    return out


def finite_points(curve):
    """Indices of the L-curve points whose fit is finite (χ² and every value)."""
    return [i for i, r in enumerate(curve.results)
            if np.isfinite(float(curve.chi2[i])) and all(np.all(np.isfinite(np.asarray(v))) for v in r.values.values())]


def safe_render(model, npix, fov):
    """``model.render``, or an all-NaN image when the model is non-finite or
    has no positive flux (virgil raises then)."""
    try:
        return np.asarray(model.render(npix, fov))
    except ValueError:
        return np.full((npix, npix), np.nan)


def safe_clean(*args, **kwargs):
    """virgil's ``clean``, and when its non-negative refit meets non-finite
    numbers (a runaway component: scipy's nnls raises "array must not contain
    infs or NaNs", 2004 data2 TV, job 18579912) once more at a quarter of the
    gain, which cannot take as large a step. A second failure is raised."""
    try:
        return clean(*args, **kwargs)
    except ValueError as e:
        if "infs or NaNs" not in str(e):
            raise
        gain = float(kwargs.get("gain", 0.1)) / 4
        print(f"CLEAN met non-finite values in its refit ({e}); retrying at gain {gain:g}", flush=True)
        return clean(*args, **{**kwargs, "gain": gain})


def sweep(start, priors, data, penalty, weights, others, warm_start=True, on_point=None, **options):
    """virgil's ``l_curve`` (the same fits, weights strongest first, each warm
    started from the previous unless ``warm_start`` is False) except that a
    warm start is moved off the prior bounds (``off_bounds``) and a fit that
    went non-finite does not seed the next weight (it is refitted from the last
    finite values instead). ``on_point(curve)`` is called with the L-curve so
    far after every weight but the last (a checkpoint: a job that times out
    keeps what was fitted)."""
    weights = sorted((float(w) for w in weights), reverse=True)
    parts, init = [], None

    def so_far():
        return LCurve(np.asarray(weights[:len(parts)]), np.concatenate([np.atleast_1d(c.chi2) for c in parts]),
                      np.concatenate([np.atleast_2d(c.chi2_red) for c in parts]),
                      np.concatenate([np.atleast_1d(c.penalty) for c in parts]), [c.results[0] for c in parts])

    for w in weights:
        piece = l_curve(start, priors, data, penalty, [w], others, warm_start=False, init=init, **options)
        parts.append(piece)
        values = piece.results[0].values
        if warm_start and all(np.all(np.isfinite(np.asarray(v))) for v in values.values()):
            init = off_bounds(values, priors)
        if on_point is not None and len(parts) < len(weights):
            on_point(so_far())
    return so_far()


def choose_weights(curve, reg, strict=True):
    """The L-curve's corner and discrepancy weights among its finite fits.
    Returns (finite indices, the curve of those, chosen {name: weight}, index
    {name: curve index}, discrepancy reached). Non-finite fits take no part;
    with none finite it raises. A corner needs three finite points: without
    them it is the weakest weight, except that ``strict`` MaxEnt with every fit
    finite keeps round 1's error."""
    ok = finite_points(curve)
    if not ok:
        raise RuntimeError("every L-curve fit is non-finite")
    if len(ok) < len(curve.results):
        print("non-finite fits at weights", [float(curve.weights[i]) for i in range(len(curve.results)) if i not in ok])
    good = curve if len(ok) == len(curve.results) else LCurve(
        np.asarray(curve.weights)[ok], np.asarray(curve.chi2)[ok], np.asarray(curve.chi2_red)[ok],
        np.asarray(curve.penalty)[ok], [curve.results[i] for i in ok])
    weakest = float(good.weights[-1])
    try:
        corner = good.corner()
    except ValueError:
        if strict and reg == "mem" and len(ok) == len(curve.results):
            raise  # round-1 behaviour
        corner = None
    corner = weakest if corner is None else float(corner)
    discrepancy = good.discrepancy()
    chosen = {"discrepancy": corner if discrepancy is None else float(discrepancy), "corner": corner}
    index = {k: ok[int(np.argmin(np.abs(np.log(good.weights) - np.log(w))))] for k, w in chosen.items()}
    return ok, good, chosen, index, discrepancy is not None


def regulariser_weights(name, data, smoke=False):
    """The weights of an L-curve for ``name``: MaxEnt's WEIGHTS, or
    WEIGHT_RANGES[name] x N (``n_independent(data)``), REG_POINTS log-spaced
    from strong to weak."""
    if name == "mem":
        weights = WEIGHTS
    else:
        low, high = WEIGHT_RANGES[name]
        weights = n_independent(data) * np.logspace(np.log10(high), np.log10(low), REG_POINTS)
    return weights[[0, len(weights) // 2, -1]] if smoke else weights


# Point sources (round 2, point-sources): up to K PointSources beside the
# image, started at CLEAN's brightest compact clusters outside the main
# source, each kept only if it lowers χ² by COMPANION_DCHI2 with the image
# fixed. The main source is the connected region of CLEAN's restored image
# above MAIN_LEVEL of its peak that holds the peak, grown by half a beam.
MAIN_LEVEL = 0.1


def pixel_offset(row, col, n, pixel):
    """(dra, ddec) mas of pixel (row, col) of an n x n virgil Image: East is
    towards column 0 and North towards row 0 (tests/test_contest_round2.py)."""
    c = (n - 1) / 2
    return (c - col) * pixel, (c - row) * pixel


def clean_point_candidates(components, restored, pixel, resolution, k):
    """Up to ``k`` compact CLEAN clusters outside the main source, brightest
    first: dicts of dra, ddec (the cluster's flux-weighted centre, mas) and
    flux (its components' total, positive part)."""
    from scipy import ndimage

    c = np.clip(np.asarray(components, float), 0.0, None)
    restored = np.asarray(restored, float)
    n = c.shape[0]
    labels, _ = ndimage.label(restored > MAIN_LEVEL * restored.max())
    main = labels == labels[np.unravel_index(np.argmax(restored), restored.shape)]
    main = ndimage.binary_dilation(main, iterations=max(1, int(round(0.5 * resolution.minor_mas / pixel))))
    rows, cols = np.mgrid[:n, :n]
    left = np.where(main, 0.0, c)
    found = []
    while len(found) < k and left.max() > 0:
        i, j = np.unravel_index(np.argmax(left), left.shape)
        near = np.hypot(rows - i, cols - j) * pixel < resolution.minor_mas
        w = left * near
        flux = float(w.sum())
        dra, ddec = pixel_offset(float((w * rows).sum() / flux), float((w * cols).sum() / flux), n, pixel)
        found.append({"dra": float(dra), "ddec": float(ddec), "flux": flux})
        left = np.where(near, 0.0, left)
    return found


def add_point_sources(s, k):
    """Setup ``s`` with up to ``k`` PointSources ``p0``, ``p1``... added (see
    MAIN_LEVEL), each tested against the scene so far with only its own
    parameters free and kept if Δχ² >= COMPANION_DCHI2. Positions are uniform
    within a beam of the start, fluxes log-uniform. The image is cut under a
    kept source (half a beam) so that it does not start with the same flux
    twice. Needs a CLEAN start (init="clean")."""
    if s.get("cleaned") is None:
        raise ValueError("point sources need a CLEAN start (init='clean')")
    cleaned, data, res = s["cleaned"], s["data"], s["resolution"]
    pix_c = float(cleaned.pixel_scale_mas)
    candidates = clean_point_candidates(cleaned.components, cleaned.restored(s["restore"]), pix_c, res, k)
    start = s["start"] if isinstance(s["start"], vm.System) else vm.System(env=s["start"])
    parts, priors, records = dict(start.components), {}, []
    flux_scale = flux_value(start.env.flux) / max(float(np.sum(np.clip(np.asarray(cleaned.components), 0, None))), 1e-12)
    n = n_independent(data)
    for cand in candidates:
        name = f"p{sum(1 for r in records if r['kept'])}"
        beam_ = res.major_mas
        pp = {f"{name}.dra": dist.Uniform(cand["dra"] - beam_, cand["dra"] + beam_),
              f"{name}.ddec": dist.Uniform(cand["ddec"] - beam_, cand["ddec"] + beam_),
              f"{name}.flux": dist.LogUniform(FLUX_FLOOR, FLUX_CAP)}
        point = vm.PointSource(inside(pp[f"{name}.flux"], cand["flux"] * flux_scale), dra=cand["dra"], ddec=cand["ddec"])
        trial = fit(vm.System(**parts, **{name: point}), pp, data).model
        dchi2 = (chi2_red(vm.System(**parts), data) - chi2_red(trial, data)) * n
        kept = bool(dchi2 >= COMPANION_DCHI2)
        records.append(cand | {"dchi2": float(dchi2), "kept": kept})
        print(f"point source at ({cand['dra']:.3g}, {cand['ddec']:.3g}) mas, CLEAN flux {cand['flux']:.3g}: "
              f"Δχ² {dchi2:.3g}, {'kept as ' + name if kept else 'dropped'}", flush=True)
        if kept:
            # The trial fit may end on a prior bound (its box is a beam wide); a
            # start on a bound is infinite in the unconstrained coordinates and
            # makes the joint L-BFGS fit NaN, so move it into the interior.
            fitted = trial.components[name]
            parts[name] = eqx.tree_at(lambda m: (m.dra, m.ddec, m.flux), fitted,
                                      tuple(jnp.asarray(inside(pp[f"{name}.{a_}"], getattr(fitted, a_)))
                                            for a_ in ("dra", "ddec", "flux")))
            priors |= pp
    env = parts["env"]
    if priors:
        nn, pix = int(np.shape(env.log_brightness)[0]), float(env.pixel_scale_mas)
        rows, cols = np.mgrid[:nn, :nn]
        lb = np.asarray(env.log_brightness).copy()
        floor = float(lb.max()) - np.log(1e3)
        for name in (k_ for k_ in parts if k_.startswith("p")):
            ra, de = float(parts[name].dra), float(parts[name].ddec)
            dra, ddec = pixel_offset(rows, cols, nn, pix)
            under = np.hypot(dra - ra, ddec - de) < 0.5 * res.minor_mas
            lb[under] = np.minimum(lb[under], floor)
        parts["env"] = eqx.tree_at(lambda m: m.log_brightness, env, jnp.asarray(lb))
    s = dict(s)
    s["start"] = vm.System(**parts)
    s["img0"] = parts["env"]
    s["path"] = "env"
    s["points"] = {k_: v for k_, v in parts.items() if k_.startswith("p") and k_[1:].isdigit()}
    s["point_priors"] = priors
    s["point_record"] = records
    # The other parameters' priors already carry their paths when the start
    # was a System; a bare Image's were only its pixels.
    rest = {k_: v for k_, v in s["priors"].items() if "log_brightness" not in k_}
    s["priors"] = image_priors(s["start"]) | rest | priors
    if not s["star"]:
        s["others"] = (Centroid(0.1 * res.minor_mas, path="env"),)
    return s


# Systematics arms (round 2): calibration nuisances marginalised by virgil
# (OIData.with_gains / with_closure_offsets), their widths chosen by an
# outer evidence scan, and per-observable error scales (error_scale's
# by_observable). Raw χ²/N on the quoted errors is always reported first.
def with_systematics(data, systematics):
    """``data`` with gains per baseline and/or closure offsets per triangle
    of the widths in ``systematics`` ({"gains": w, "closure": w})."""
    if not systematics:
        return data
    datasets = data if isinstance(data, list) else [data]
    out = []
    for d in datasets:
        if systematics.get("gains"):
            d = d.with_gains(baseline=float(systematics["gains"]))
        if systematics.get("closure"):
            d = d.with_closure_offsets(triangle=float(systematics["closure"]))
        out.append(d)
    return out if isinstance(data, list) else out[0]


# --systematics: one switch over the calibration nuisances and error scales.
# block_scales: one error scale per observable kind (error_scale's
# by_observable), estimated on the gains-free model and applied before any
# gains or closure offsets are added (error_scale refuses data that has them).
# The nuisances' widths come from the outer evidence scan (systematics_scan) on
# --prior gp; on the evidence-free path (MaxEnt and the other regularisers) the
# scan's median is used, fixed.
SYSTEMATICS = {"none": (), "block_scales": ("block_scales",), "gains+block_scales": ("gains", "block_scales"),
               "gains+closure_offsets": ("gains", "closure")}
SYSTEMATIC_WIDTHS = {"gains": (0.01, 0.03, 0.1), "closure": (0.01, 0.03, 0.1)}  # log|V| per baseline; rad per triangle
SYSTEMATIC_SUFFIX = {"none": "", "block_scales": "_bscale", "gains+block_scales": "_gain_bscale",
                     "gains+closure_offsets": "_gain_cloff"}


def systematic_widths(name, scan=True):
    """{"gains": [w...], "closure": [w...]} for the nuisances in ``name``
    (SYSTEMATICS): the whole scan, or with ``scan=False`` the median alone."""
    out = {}
    for kind in SYSTEMATICS[name]:
        if kind in SYSTEMATIC_WIDTHS:
            ws = SYSTEMATIC_WIDTHS[kind]
            out[kind] = list(ws) if scan else [ws[len(ws) // 2]]
    return out


def rescale(data, scales):
    """``data`` (one OIData or a list) with ``scales`` (error_scale's dict) applied."""
    if isinstance(data, list):
        return [d.with_error_scale(scales) for d in data]
    return data.with_error_scale(scales)


def informed_line(informed):
    """The provenance line of a summary: round-2 fits chose their method after
    the truths were seen (contests/fit_plan.yml), so they are "informed"."""
    return ("provenance=informed (method chosen after seeing the truth; values from pre-submission information "
            "and the data only)") if informed else "provenance=blind"


def task_index(x):
    """TASKS index of ``x``, an index or a label."""
    return int(x) if str(x).isdigit() else [t["label"] for t in TASKS].index(x)


def joint_spec(selection):
    """One spec from the TASKS named in ``selection`` (indices or labels):
    their files in one dataset, the rest (field, prior, star, ...) from the
    first, which the others must agree with on field, prior and wavel
    (--joint: the bands of one target fitted together)."""
    specs = [TASKS[task_index(x)] for x in selection]
    for key in ("field", "prior", "wavel"):
        if len({repr(sp.get(key)) for sp in specs}) > 1:
            raise ValueError(f"--joint tasks disagree on {key}: {[sp.get(key) for sp in specs]}")
    return dict(specs[0], label="+".join(sp["label"] for sp in specs), files=[f for sp in specs for f in sp["files"]])


def spec_of(task):
    """A task's spec: ``TASKS[task]`` for an index, or a spec dict as given
    (scripts/contest_bench.py passes simulated datasets this way, with
    absolute ``files``)."""
    return task if isinstance(task, dict) else TASKS[task]


COMPANION_SNR = 5.0  # a companion is kept when the linear flux map's peak SNR reaches this
COMPANION_DCHI2 = 25.0  # ... and its refit lowers χ² by this much (3 extra parameters)
COMPANION_GUARD = 10.0  # search only when the primary's χ²/N is within this factor of the no-star ellipse's
COMPANION_MAX_AXIS = 201  # samples per axis of the companion map (coarser steps beyond)
# Star fraction f = 1 / (1 + env/star). Below F_MIN the star is absent; the
# env/star flux ratio's log-uniform prior has the fixed upper bound
# 1/F_MIN - 1, not one that moves with the start (C2g: CLEAN's runaway totals
# of 1e163 moved a cap of 10x the start along with them).
F_MIN = 0.01
FLUX_CAP = 1.0 / F_MIN - 1.0
STAR_GUARD = 2.0  # a star start whose χ²/N exceeds this x the no-star ellipse's is dropped
EDGE = 0.05  # starts keep this fraction of a prior's range (log range for scales) from its bounds
N_STARTS, N_STARTS_SMOKE = 16, 4  # multistart fits of the parametric models
# Position angles: a window two half-turns wide, so that no plausible PA
# (0-180) starts or ends near a bound; the models are symmetric under
# PA + 180, so results are wrapped into [0, 180).
PA_PRIOR = (-90.0, 270.0)
# The disk and ellipse arms model a resolved star, so their diameters'
# log-uniform priors start at this fraction of the beam's minor axis (C2g).
RESOLVED_FLOOR = 0.5


def v2_table(data):
    """Every dataset's V² as flat arrays: u, v and wavelength (m), V² and its
    error (from |V| and its error for amplitude data)."""
    cols = []
    for d in data if isinstance(data, list) else [data]:
        u, v = np.ravel(np.asarray(d.u, float)), np.ravel(np.asarray(d.v, float))
        wavel = np.broadcast_to(np.ravel(np.asarray(d.wavel, float)), u.shape) if np.size(d.wavel) in (1, u.size) \
            else np.ravel(np.asarray(d.wavel, float))[: u.size]
        vis, err = np.ravel(np.asarray(d.vis, float)), np.ravel(np.asarray(d.d_vis, float))
        index = np.arange(vis.size) if d.vis_index is None else np.ravel(np.asarray(d.vis_index))
        if not d.v2_flag:
            vis, err = vis**2, 2 * np.abs(vis) * err
        cols.append((u[index], v[index], wavel[index], vis, err))
    return tuple(np.concatenate(c) for c in zip(*cols))


def moments_of(data):
    """``crosscheck.starts.second_moments`` of the data's V²."""
    u, v, wavel, v2, err = v2_table(data)
    return linear_starts.second_moments(u / wavel, v / wavel, v2, err)


def chi2_red(model, data):
    """χ²/N of ``model`` against the quoted errors, over all datasets."""
    datasets = data if isinstance(data, list) else [data]
    chi2 = sum(float(np.sum(np.asarray(whitened_residuals(model, d)) ** 2)) for d in datasets)
    return chi2 / sum(int(d.n_independent) for d in datasets)


def n_independent(data):
    return sum(int(d.n_independent) for d in (data if isinstance(data, list) else [data]))


def _interior(prior, edge=EDGE):
    """A prior's range shrunk by ``edge`` of it at each end (in log for a
    log-uniform prior): (low, high, log)."""
    log = isinstance(prior, dist.LogUniform)
    lo, hi = float(prior.low), float(prior.high)
    if log:
        lo, hi = np.log(lo), np.log(hi)
    pad = edge * (hi - lo)
    return lo + pad, hi - pad, log


def inside(prior, x, edge=EDGE):
    """``x`` moved into the prior's interior (see ``_interior``)."""
    lo, hi, log = _interior(prior, edge)
    y = np.log(max(float(x), 1e-300)) if log else float(x)
    y = float(np.clip(y if np.isfinite(y) else hi, lo, hi))
    return float(np.exp(y)) if log else y


def jittered_starts(centre, priors, n, rng, jitter):
    """``centre`` (a dict of parameter values) and ``n - 1`` draws about it:
    ``jitter[k] = ("log", s)`` multiplies by exp(s·N(0, 1)) (a scale) and
    ``("add", s)`` adds U(-s, s) (a location or angle). Every value is then
    moved into its prior's interior, so no start sits on a bound."""
    out = []
    for i in range(n):
        start = {}
        for k, x in centre.items():
            if i and k in jitter:
                kind, size = jitter[k]
                x = x * np.exp(size * rng.standard_normal()) if kind == "log" else x + rng.uniform(-size, size)
            start[k] = inside(priors[k], x) if k in priors else float(x)
        out.append(start)
    return out


def multistart_fit(model_fn, priors, data, starts, **options):
    """Fit ``model_fn(start)`` from each start and keep the lowest χ²/N.
    Returns the best FitResult and a record: χ²/N of the best and of every
    start, their spread and whether the best fit converged."""
    best, chi2s = None, []
    for start in starts:
        r = fit(model_fn(start), priors, data, **options)
        c = chi2_red(r.model, data)
        chi2s.append(c)
        if best is None or (np.isfinite(c) and not c >= best[0]):
            best = (c, r)
    finite = [c for c in chi2s if np.isfinite(c)]
    return best[1], {"chi2_red": best[0], "chi2_starts": chi2s,
                     "spread": float(max(finite) - min(finite)) if finite else np.nan,
                     "converged": best[1].info.get("converged") is not False}


def fit_primary(data, resolution, star_model, start, n_starts=N_STARTS, seed=0):
    """The primary alone, as a parametric model, its priors (paths relative
    to the primary) and a fit record. "ellipse": an elliptical limb-darkened
    disk (linear law, u = 0.5 held fixed: a starting model, the image takes
    the rest), diameter log-uniform from RESOLVED_FLOOR x the beam's minor
    axis (a resolved star: an unresolved one is the point arm's) to half the
    starting field, axis ratio uniform, PA uniform over PA_PRIOR. Seeded by the
    data's second moments (crosscheck.starts), then a diameter scan from 0.5x
    to 2x that picks the visibility null's branch, then ``n_starts`` fits
    jittered about it (C2g). "point_companion": a point."""
    if star_model == "point_companion":
        point = vm.PointSource()
        return point, {}, {"chi2_red": chi2_red(point, data), "converged": True}
    if not hasattr(vm, "EllipticalLimbDarkenedDisk"):
        raise RuntimeError("star_model='ellipse' needs virgil with EllipticalLimbDarkenedDisk (virgil#250)")
    field = float(start.env.pixel_scale_mas * np.shape(start.env.log_brightness)[0])
    priors = {"diam": dist.LogUniform(RESOLVED_FLOOR * resolution.minor_mas, 0.5 * field),
              "ratio": dist.Uniform(0.2, 1.0), "pa": dist.Uniform(*PA_PRIOR)}
    m = moments_of(data)
    diam0 = float(linear_starts.disk_diameter_from_sigma(m["sigma_major"], 0.5))
    diam0 = inside(priors["diam"], diam0 if np.isfinite(diam0) and diam0 > 0 else resolution.major_mas)
    ratio0 = float(np.clip(m["ratio"] if np.isfinite(m["ratio"]) else 0.8, 0.25, 0.95))
    pa0 = float(m["pa"]) if np.isfinite(m["pa"]) else 90.0

    def disk(p):
        return vm.EllipticalLimbDarkenedDisk(p["diam"], ratio=p["ratio"], pa=p["pa"], u=(0.5,))

    scan = [inside(priors["diam"], diam0 * f) for f in np.logspace(np.log10(0.5), np.log10(2.0), 9)]
    scan_chi2 = [chi2_red(disk({"diam": d, "ratio": ratio0, "pa": pa0}), data) for d in scan]
    centre = {"diam": scan[int(np.nanargmin(scan_chi2))], "ratio": ratio0, "pa": pa0}
    starts = jittered_starts(centre, priors, n_starts, np.random.default_rng(seed),
                             {"diam": ("log", 0.3), "ratio": ("add", 0.1), "pa": ("add", 30.0)})
    for s in starts:
        s["ratio"] = float(np.clip(s["ratio"], 0.25, 0.95))
    best, record = multistart_fit(disk, priors, data, starts)
    model = eqx.tree_at(lambda d: d.pa, best.model, jnp.asarray(float(best.model.pa) % 180.0))
    record |= {"moments": m, "scan_chi2": scan_chi2}
    print(f"primary: χ²/N {record['chi2_red']:.4g} (spread {record['spread']:.3g} over {len(starts)} starts, "
          f"converged {record['converged']}); diam {float(model.diam):.3g} mas ratio {float(model.ratio):.3f} "
          f"pa {float(model.pa):.1f}; moments start diam {diam0:.3g} ratio {ratio0:.3f} pa {pa0:.1f}", flush=True)
    return model, priors, record


def companion_map(data, resolution, primary):
    """A linear flux map (virgil.grid_fit.linear_flux_grid) for a point
    companion to ``primary``, emission only, over ±λ/B_min (the largest scale
    the data probe; not the starting image's field, which for 2004 data2
    reaches only 6.9 mas) a third of a beam per step (coarser beyond
    COMPANION_MAX_AXIS samples), with positions on the primary's disk (its
    major radius plus half a beam) left out: there the map's peak is a spot,
    not a companion (vv#70). Datasets are combined by inverse variance.
    Returns the peak: offsets (mas), flux, SNR, the half-width searched and
    the radius left out."""
    from virgil.grid_fit import linear_flux_grid

    u, v, wavel, _, _ = v2_table(data)
    half = 1.0 / float(np.min(np.hypot(u, v) / wavel)) / linear_starts.MAS
    step = max(resolution.minor_mas / 3, 2 * half / (COMPANION_MAX_AXIS - 1))
    axis = np.arange(-half, half + 1e-9, step)
    template = vm.System(star=primary, comp=vm.PointSource(1e-3))
    samples = {"comp.dra": axis, "comp.ddec": axis, "comp.flux": np.array([1e-3])}
    num = den = 0.0
    for d in data if isinstance(data, list) else [data]:
        g = linear_flux_grid(template, d, samples)  # (model, data, grid); .flux, .flux_error
        w = 1.0 / np.asarray(g.flux_error) ** 2
        num, den = num + np.nan_to_num(np.asarray(g.flux) * w), den + np.nan_to_num(w)
    flux = num / np.where(den > 0, den, np.inf)
    snr = np.where(flux > 0, num / np.sqrt(np.where(den > 0, den, np.inf)), 0.0)
    dra, ddec = np.meshgrid(axis, axis, indexing="ij")  # as linear_flux_grid's axes: dra first
    hole = 0.5 * float(np.ravel(getattr(primary, "diam", 0.0))[0]) + 0.5 * resolution.minor_mas
    snr = np.nan_to_num(np.where(np.hypot(dra, ddec) < hole, 0.0, snr))
    i, j = np.unravel_index(np.argmax(snr), snr.shape)
    return {"dra": float(axis[i]), "ddec": float(axis[j]), "flux": float(flux[i, j]), "snr": float(snr[i, j]),
            "half": float(half), "hole": float(hole)}


def companion_search(data, resolution, primary, primary_priors, primary_chi2=None, nostar_chi2=None):
    """A companion to ``primary`` from ``companion_map``, with guards (C2g):
    the search runs only when the primary's χ²/N (``primary_chi2``) is within
    COMPANION_GUARD of the no-star ellipse's (``nostar_chi2``), since against
    a broken primary any residual looks like a companion (2006: SNR 35,000);
    a companion is kept only at SNR >= COMPANION_SNR and when the refitted
    System lowers χ² by COMPANION_DCHI2. Returns the base (the primary, or
    System(star=primary, comp=...) refitted), its priors with paths relative
    to the scene ("star.<...>") and a record. Positions are uniform within a
    beam of the peak, the flux ratio log-uniform."""
    priors = {f"star.{k}": v for k, v in primary_priors.items()}
    record = {"searched": False, "kept": False, "primary_chi2": primary_chi2, "nostar_chi2": nostar_chi2}
    if primary_chi2 is not None and nostar_chi2 is not None and not primary_chi2 <= COMPANION_GUARD * nostar_chi2:
        print(f"companion search skipped: primary χ²/N {primary_chi2:.4g} > {COMPANION_GUARD:g} x no-star "
              f"ellipse χ²/N {nostar_chi2:.4g}", flush=True)
        return primary, priors, record
    peak = companion_map(data, resolution, primary)
    record |= {"searched": True, "peak": peak}
    line = (f"companion search (±{peak['half']:.3g} mas, < {peak['hole']:.3g} mas left out): peak SNR {peak['snr']:.2f} "
            f"at ({peak['dra']:.3g}, {peak['ddec']:.3g}) mas, flux {peak['flux']:.3g}")
    if peak["snr"] < COMPANION_SNR:
        print(f"{line}: below threshold", flush=True)
        return primary, priors, record
    dra, ddec, beam_ = peak["dra"], peak["ddec"], resolution.major_mas
    pair_priors = {f"star.{k}": v for k, v in primary_priors.items()} | {
        "comp.dra": dist.Uniform(dra - beam_, dra + beam_), "comp.ddec": dist.Uniform(ddec - beam_, ddec + beam_),
        "comp.flux": dist.LogUniform(FLUX_FLOOR, 1.0)}
    pair = vm.System(star=primary, comp=vm.PointSource(inside(pair_priors["comp.flux"], peak["flux"]), dra=dra, ddec=ddec))
    pair = fit(pair, pair_priors, data).model
    n = n_independent(data)
    chi2_one, chi2_two = chi2_red(primary, data), chi2_red(pair, data)
    dchi2 = (chi2_one - chi2_two) * n
    kept = dchi2 >= COMPANION_DCHI2
    record |= {"kept": bool(kept), "chi2_single": chi2_one, "chi2_pair": chi2_two, "dchi2": float(dchi2)}
    print(f"{line}; χ²/N {chi2_one:.4g} alone, {chi2_two:.4g} with it (Δχ² {dchi2:.3g}): "
          f"{'kept' if kept else 'dropped (Δχ² < %g)' % COMPANION_DCHI2}", flush=True)
    if not kept:
        return primary, priors, record
    return pair, {f"star.{k}": v for k, v in pair_priors.items()}, record


def clean_companion(components, pixel, resolution):
    """The brightest CLEAN component cluster (flux within a beam of the
    brightest component) more than 1.5 beams from the centre, for the log:
    a cross-check of the companion search, not used to decide."""
    c = np.asarray(components, float)
    n = c.shape[0]
    offsets = (np.arange(n) - (n - 1) / 2) * pixel
    yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
    r = np.hypot(xx, yy)
    outer = np.where(r > 1.5 * resolution.major_mas, c, -np.inf)
    if not np.any(np.isfinite(outer)) or np.max(outer) <= 0:
        return None
    i, j = np.unravel_index(np.argmax(outer), c.shape)
    near = np.hypot(xx - xx[i, j], yy - yy[i, j]) < resolution.major_mas
    return {"r_mas": float(r[i, j]), "row": int(i), "col": int(j), "flux": float(np.sum(np.clip(c[near], 0, None)))}


def flux_priors(flux_cap, sparco=False):
    """Priors on the image's flux relative to the star: log-uniform (a scale),
    and with SPARCO a uniform power-law index (a location on log flux against
    log wavelength), relative to the star's: -8 to 8 spans a cool disk against
    a hot star's Rayleigh-Jeans slope and the reverse."""
    if sparco:
        return {"env.flux.ratio": dist.LogUniform(FLUX_FLOOR, flux_cap), "env.flux.index": dist.Uniform(-8.0, 8.0)}
    return {"env.flux": dist.LogUniform(FLUX_FLOOR, flux_cap)}


def flux_value(flux):
    """A component's flux as a number: a spectrum's ratio at its wavel0."""
    return float(np.ravel(getattr(flux, "ratio", flux))[0])


# The images saved for scoring are rendered from the fitted model on the
# reference grid, not resampled from the fit grid (C2g, smoke job 18207630:
# SourceModel.render resamples an Image bilinearly, which smooths it, and
# the saved images' own χ²/N was 3-11 against the fits' 1.2).
def _shift(fu, fv, dra, ddec):
    """The Fourier shift factor of an offset (dra, ddec) mas at spatial
    frequencies fu, fv (cycles per mas, East and North)."""
    return np.exp(-2j * np.pi * (fu * float(np.ravel(dra)[0]) + fv * float(np.ravel(ddec)[0])))


def _taper(f, q_pass, q_stop):
    """1 up to ``q_pass``, a raised cosine down to 0 at ``q_stop``, then 0,
    in |f| (cycles per mas); a sharp cut at ``q_pass`` when there is no room
    for the roll-off."""
    f = np.abs(f)
    if not q_stop > q_pass:
        return (f <= q_pass).astype(float)
    x = np.clip((f - q_pass) / (q_stop - q_pass), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * x))


def _image_visibility(image, fu, fv, q_max, chunk=16384):
    """An Image's centred visibilities at fu, fv (cycles per mas): the
    transform of its pixels, each a point at its centre (as virgil's
    ``Image.model``), kept as they are up to ``q_max`` (the data's largest
    spatial frequency) and rolled off to zero at its own grid's Nyquist
    frequency (``_taper`` along each axis of the grid's rotated frame),
    beyond which a pixel grid's transform only repeats itself. Rendered on
    a finer grid, that is a smooth band-limited interpolation of the pixels:
    flux-conserving, with the pixels' visibilities wherever the data reach,
    and without the ringing of a sharp cut."""
    b = np.asarray(image.brightness, float)
    nrow, ncol = b.shape
    pixel = float(image.pixel_scale_mas)
    shape = np.shape(fu)
    fr, gr = (np.ravel(np.asarray(x, float)) for x in vm.rotate(fu, fv, -image.rotation_deg))
    x = (0.5 * (ncol - 1) - np.arange(ncol)) * pixel  # column j's offset East, as Image._centred_image
    y = (0.5 * (nrow - 1) - np.arange(nrow)) * pixel  # row i's offset North
    nyquist = 0.5 / pixel
    weight = _taper(fr, q_max, nyquist) * _taper(gr, q_max, nyquist)
    vis = np.zeros(fr.shape, complex)
    for start in range(0, fr.size, chunk):
        sl = slice(start, start + chunk)
        keep = np.flatnonzero(weight[sl] > 0) + start
        rows = np.exp(-2j * np.pi * np.outer(gr[keep], y)) @ b
        vis[keep] = np.sum(rows * np.exp(-2j * np.pi * np.outer(fr[keep], x)), axis=1)
    return (vis * weight).reshape(shape)


def scene_visibility(model, fu, fv, q_max):
    """Unit-sum visibilities of ``model`` at spatial frequencies ``fu``,
    ``fv`` (cycles per mas, East and North), as it is drawn: a System's
    Resolved parts are left out and the rest weighted by their reference
    fluxes (as ``SourceModel.render`` does); analytic components are their
    own visibilities; an Image's are rolled off beyond ``q_max``
    (``_image_visibility``)."""
    fu, fv = np.asarray(fu, float), np.asarray(fv, float)
    if isinstance(model, vm.System):
        parts = [c for c in model.parts if not isinstance(c, vm.Resolved)]
        weights = [flux_value(getattr(c, "flux", 1.0)) for c in parts]
        vis = sum(w * scene_visibility(c, fu, fv, q_max) for w, c in zip(weights, parts)) / sum(weights)
    elif isinstance(model, vm.Image):
        vis = _image_visibility(model, fu, fv, q_max)
    else:  # an analytic component: its model() includes its offset
        mas = linear_starts.MAS
        return np.asarray(model.model(fu / mas, fv / mas, 1.0), complex)
    return vis * _shift(fu, fv, model.dra, model.ddec)


def data_q_max(data):
    """The data's largest spatial frequency (cycles per mas), over V² and
    closure triangles alike (each triangle's third baseline too)."""
    best = 0.0
    for d in data if isinstance(data, list) else [data]:
        u, v = np.ravel(np.asarray(d.u, float)), np.ravel(np.asarray(d.v, float))
        wavel = np.ravel(np.asarray(d.wavel, float))
        wavel = wavel if wavel.size == u.size else np.min(wavel)
        best = max(best, float(np.max(np.hypot(u, v) / wavel)) * linear_starts.MAS)
    return best


def reference_npix(fov_mas, q_max, base=None):
    """The reference grid's side: ``REF_NPIX``, or the smallest odd side whose
    Nyquist frequency is at least 1.1 x ``q_max`` (cycles per mas) if the data
    reach beyond what ``REF_NPIX`` samples, so the saved image always samples
    every baseline."""
    base = REF_NPIX if base is None else base
    n = int(np.ceil(2.2 * q_max * fov_mas))
    return max(base, n + 1 - n % 2)


def render_scene(model, npix, fov_mas, q_max):
    """``model`` drawn on an ``npix`` x ``npix`` grid spanning ``fov_mas``,
    in ``SourceModel.render``'s orientation (East left, North up) and unit
    sum, by Fourier synthesis: the inverse DFT of ``scene_visibility`` on
    the grid's own frequency lattice, rolled off (``_taper``) from
    ``q_max``, the data's largest spatial frequency (``data_q_max``), to
    the grid's Nyquist frequency. So the image's DFT reproduces the model's
    visibilities wherever the data reach (exactly at the lattice points;
    between them, to the leakage of what lies outside the field), with no
    resampling of the fit's pixels and no blurring of point sources, while
    what the data cannot see is smoothed rather than left to ring. A few
    pixels can still be slightly negative."""
    pixel = float(fov_mas) / int(npix)
    if 0.5 / pixel < q_max:
        raise ValueError(f"a {npix}-pixel grid over {fov_mas:.4g} mas (Nyquist {0.5 / pixel:.3g} cycles/mas) "
                         f"undersamples the data (q_max {q_max:.3g}); use reference_npix")
    freq = (np.arange(npix) - (npix - 1) / 2) / (npix * pixel)
    fu, fv = np.meshgrid(freq, freq)  # [row l, column k] = (freq[k] East, freq[l] North)
    nyquist = 0.5 / pixel
    vis = scene_visibility(model, fu, fv, q_max) * _taper(fu, q_max, nyquist) * _taper(fv, q_max, nyquist)
    offsets = ((npix - 1) / 2 - np.arange(npix)) * pixel  # column j's dra, row i's ddec
    e = np.exp(2j * np.pi * np.outer(offsets, freq))
    image = np.real(e @ vis @ e.T)
    return image / np.sum(image)


def star_record(model):
    """The fitted star, for the npz (NaN where the model has no such
    parameter): ``star_frac``, the star's share of the total flux (with a
    companion, of the pair's); the primary's ``star_diam``, ``star_ratio``
    and ``star_pa``; and a companion's ``comp_dra``, ``comp_ddec`` and
    ``comp_flux`` (relative to the primary)."""
    nan = float("nan")
    out = dict(star_frac=nan, star_diam=nan, star_ratio=nan, star_pa=nan, comp_dra=nan, comp_ddec=nan, comp_flux=nan)
    if not (isinstance(model, vm.System) and "star" in model.names):
        return out
    total = sum(flux_value(getattr(c, "flux", 1.0)) for c in model.parts)
    star = model.star
    out["star_frac"] = flux_value(getattr(star, "flux", 1.0)) / total
    primary = star.star if isinstance(star, vm.System) else star
    for key in ("diam", "ratio", "pa"):
        if hasattr(primary, key):
            out[f"star_{key}"] = float(np.ravel(getattr(primary, key))[0])
    if isinstance(star, vm.System) and "comp" in star.names:
        weights = {k: flux_value(getattr(c, "flux", 1.0)) for k, c in star.components.items()}
        out |= {"comp_dra": float(np.ravel(star.comp.dra)[0]), "comp_ddec": float(np.ravel(star.comp.ddec)[0]),
                "comp_flux": weights["comp"] / weights["star"]}
    return out


# Star or no star (C2g): the star arm's GP fit against the same fit with no
# star, by Laplace evidence over every fitted parameter. A star is kept only
# when it raises the evidence by ln 100 (Jeffreys' "decisive", odds 100:1).
STAR_LOG_BAYES = float(np.log(100.0))


def log_evidence_full(result, data, priors, path="env"):
    """``log_evidence`` (the image's latents integrated out, every other
    parameter at its MAP), plus the Laplace integral over those other
    parameters (``priors``: the star, the fluxes, a halo), so that a model
    with more of them pays its Occam factor. Each is integrated in the
    coordinate in which its Jeffreys prior is flat (log for a log-uniform
    scale, linear for a uniform location or angle), with the curvature
    ``Jᵀ(I + J_z J_zᵀ)⁻¹J`` of the likelihood marginalised over the latents
    (J_z: the Jacobian with respect to the latents) plus the prior's
    variance (R²/12 for a range R), so that a parameter the data do not
    constrain costs ½ ln(π/6) ≈ −0.3 rather than a divergent log
    determinant. Returns (log Z, the Occam term)."""
    from virgil.imaging import _residual_jacobian  # private in virgil 0.3.0: the Jacobian log_evidence uses

    log_z = float(log_evidence(result, data, path))
    if not priors:
        return log_z, 0.0
    model = result.model
    _, jac_z = _residual_jacobian(model, data, f"{path}.log_brightness.latent")
    u, sv, _ = np.linalg.svd(jac_z, full_matrices=False)
    columns, ranges = [], []
    for name, prior in priors.items():
        _, jac = _residual_jacobian(model, data, name)
        lo, hi, log = _interior(prior, 0.0)
        if log:  # d/d ln θ = θ d/dθ
            jac = jac * float(np.ravel(model.get(name))[0])
        columns.append(jac.reshape(jac.shape[0], -1))
        ranges.append(hi - lo)
    jac = np.concatenate(columns, axis=1)
    projected = u.T @ jac
    curvature = jac.T @ jac - projected.T @ ((sv**2 / (1.0 + sv**2))[:, None] * projected)
    curvature += np.diag(12.0 / np.square(ranges))
    _, logdet = np.linalg.slogdet(curvature)
    occam = float(-np.sum(np.log(ranges)) + 0.5 * len(ranges) * np.log(2 * np.pi) - 0.5 * logdet)
    return log_z + occam, occam


def select_channels(path, wavel, tmp_dir):
    """A copy of an OIFITS file with every channel outside ``wavel`` (µm)
    flagged, since OIData reads FLAG but cannot select channels itself."""
    from astropy.io import fits

    out = pathlib.Path(tmp_dir) / f"{pathlib.Path(path).stem}_{wavel[0]}-{wavel[1]}um.fits"
    with fits.open(path) as h:
        keep = {x.header["INSNAME"]: (x.data["EFF_WAVE"] >= wavel[0] * 1e-6) & (x.data["EFF_WAVE"] <= wavel[1] * 1e-6)
                for x in h if x.name == "OI_WAVELENGTH"}
        for x in h:
            if x.name in ("OI_VIS2", "OI_VIS", "OI_T3", "OI_FLUX") and "FLAG" in x.columns.names:
                x.data["FLAG"] |= ~keep[x.header["INSNAME"]][None, :]
        h.writeto(out, overwrite=True)
    return str(out)


def hole_support(x, hole_mas):
    """The support of an image on the axis ``x`` (mas) with a hole of radius
    ``hole_mas`` at the centre (None: no hole, no mask)."""
    return None if hole_mas is None else np.hypot(x[None, :], x[:, None]) >= hole_mas


SCENE_PARTS = ("point", "ud", "ld_ellipse", "gauss_halo", "image")


def scene_options(text, hole_mas=None):
    """--scene's comma list of SCENE_PARTS as setup() options. The central
    source is "point" (an unresolved star), "ud" (a uniform disk, diameter
    fitted) or "ld_ellipse" (an elliptical limb-darkened disk, fit_primary;
    its companion search covers a "point" beside it); "gauss_halo" adds a
    circular Gaussian (FWHM and flux fitted); "image" is required: the pixel
    image every arm fits. ``hole_mas``: --hole."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    bad = [p for p in parts if p not in SCENE_PARTS]
    if bad or len(set(parts)) != len(parts):
        raise ValueError(f"--scene takes distinct parts of {SCENE_PARTS}, not {parts}")
    if "image" not in parts:
        raise ValueError("--scene needs an image")
    if "ud" in parts and ("ld_ellipse" in parts or "point" in parts):
        raise ValueError("--scene: ud cannot be combined with ld_ellipse or point")
    model = "ellipse" if "ld_ellipse" in parts else "disk" if "ud" in parts else "point" if "point" in parts else None
    return dict(star=model is not None, star_model=model or "point", gauss_halo="gauss_halo" in parts,
                hole_mas=hole_mas, tag="_" + "-".join(parts))


def load_data(spec, data_dir):
    """The OIData of ``spec``'s files (a list if there are several), with
    channels outside its "wavel" flagged."""
    paths = [str(data_dir / f) for f in spec["files"]]
    if spec.get("wavel"):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="contest_"))
        paths = [select_channels(p, spec["wavel"], tmp) for p in paths]
    return OIData(paths[0]) if len(paths) == 1 else [OIData(p) for p in paths]


def setup(task, data_dir, halo=False, grow=1.0, star=None, init="moments", clean_iters=3000,
          oversample=4.0, clean_gain=0.1, star_model="point", sparco=False, mean_blur=1.0,
          n_starts=N_STARTS, guard=True, field_override=None, clean_scales=(0.0,), gauss_halo=False,
          hole_mas=None, tag=None, max_pix=None):
    """The data, starting model, priors and fixed regularisers for one task
    (shared with scripts/diagnose_stall.py). ``grow`` enlarges a field chosen
    from the data. With a star, ``star_model`` is "point" (unresolved) or
    "disk" (a uniform disk whose diameter is fitted); ``sparco`` gives the
    image a power-law spectrum relative to the star's. ``mean_blur`` scales
    the beam that restores CLEAN's components into the starting image (1:
    the beam; 0.5 keeps detail finer than a beam). ``n_starts``: the
    parametric fits' multistart size. With ``guard``, a star whose start
    fails (CLEAN's star fraction below F_MIN, or a parametric χ²/N worse
    than STAR_GUARD x the no-star ellipse's, or an unconverged fit) is
    dropped and the setup redone without it; ``star_rejected`` says why
    (C2g). ``field_override`` (mas) fixes the field instead of the task's
    (round 2, --field: only where pre-submission information gives one).
    ``clean_scales`` (in beam minor axes; (0,) is point components) makes CLEAN multi-scale.
    ``gauss_halo`` adds a circular Gaussian beside the image (``ghalo``, FWHM
    and flux fitted); ``hole_mas`` is the radius of the hole in the image's
    support (default with a star: half the beam's minor axis; the image's
    flux is then a fitted parameter, relative to the star's); ``tag`` is the
    label suffix of an explicit scene (--scene), in place of _star/_nostar.
    ``max_pix`` caps the pixels on a side (default MAX_PIX; --max-pix): the
    evidence's SVD (GP runs) costs data points x pixels², so a large field on a
    fine grid takes hours (round 2: 2016 Obj2 at 400 mas, 2024 MATISSE)."""
    spec = spec_of(task)
    args = dict(halo=halo, grow=grow, init=init, clean_iters=clean_iters, oversample=oversample,
                clean_gain=clean_gain, sparco=sparco, mean_blur=mean_blur, n_starts=n_starts,
                field_override=field_override, clean_scales=clean_scales, gauss_halo=gauss_halo,
                hole_mas=hole_mas, tag=tag, max_pix=max_pix)
    # star=None keeps the task's default; True/False override it, so any
    # dataset can be imaged with and without a central star and compared
    # (labels get _star / _nostar).
    label, files = spec["label"], spec["files"]
    if star is None:
        star = spec.get("star", False)
    else:
        label += tag or ("_star" if star else "_nostar")
    field, prior_spec = spec.get("field"), spec.get("prior")
    field_source = spec.get("field_source")
    if field_override is not None:
        field, field_source = float(field_override), f"--field {float(field_override):g} mas"
    if halo:
        label += "_halo"
    if hole_mas is not None:
        label += f"_hole{hole_mas:g}"
    if sparco:
        label += "_sparco"
    if init == "clean":
        label += "_clean"
    clean_scales = tuple(float(f) for f in clean_scales)
    data = load_data(spec, data_dir)
    datasets = data if isinstance(data, list) else [data]
    npts = sum(np.asarray(d.flatten_data()[0]).size for d in datasets)
    resolution = beam(data)

    kwargs = {"star": star, "oversample": oversample}
    if hole_mas is None and star:
        hole_mas = 0.5 * resolution.minor_mas
    if hole_mas is not None:
        kwargs["hole_mas"] = hole_mas
    start = starting_image(data, **kwargs)
    star_priors = {}
    star_fit = {"converged": True}  # the parametric star fit's record, for the guard
    nostar = None  # the no-star ellipse (fit_ellipse's tuple), the guards' reference
    if star and guard:
        field0 = float(start.env.pixel_scale_mas * np.shape(start.env.log_brightness)[0])
        nostar = fit_ellipse(dict(fov=field0, star=False, data=data, resolution=resolution, start=start,
                                  img0=start.env), n_starts=n_starts)

    def no_star_setup():
        """This setup with no star (the guards' and the evidence test's
        alternative), under this label."""
        s_ = setup(task, data_dir, star=False, star_model="none", guard=False, **args)
        s_.update(label=label, ellipse=nostar if nostar is not None else s_.get("ellipse"))
        return s_

    def without_star(reason, detail=""):
        print(f"{label}: star rejected ({reason}{': ' + detail if detail else ''}); imaged without it", flush=True)
        s_ = no_star_setup()
        s_.update(star_rejected=reason)
        return s_

    if star and star_model == "disk":
        # Diameter: a scale parameter, so log-uniform, from half the beam's
        # minor axis (this arm is the resolved star: an unresolved one is the
        # point_star arm's, and below that floor the GP took the star down to
        # 0.075 mas, C2g) to the beam's major axis (beyond that, flux belongs
        # in the image). Started from a fit of the disk alone, from the
        # second-moment diameter (C2g) and from half a beam.
        diam_prior = dist.LogUniform(RESOLVED_FLOOR * resolution.minor_mas, resolution.major_mas)
        diam0 = float(linear_starts.disk_diameter_from_sigma(moments_of(data)["sigma_major"], 0.0))
        starts = [{"diam": inside(diam_prior, d)} for d in (diam0, 0.5 * resolution.minor_mas) if np.isfinite(d) and d > 0]
        alone, star_fit = multistart_fit(lambda p: vm.UniformDisk(p["diam"]), {"diam": diam_prior}, data, starts)
        start = vm.System(star=alone.model, env=start.env)
        star_priors = {"star.diam": diam_prior}
    elif star and star_model in ("ellipse", "point_companion"):
        # A parametric start from what contestants were told (2004: "a limb
        # darkened star with one or more spots"; 2024 Obj2: "a young star with
        # a suspected companion"): the primary alone, then a companion search,
        # with the GP image left for the spots and the rest.
        primary, primary_priors, star_fit = fit_primary(data, resolution, star_model, start, n_starts)
        base, star_priors, star_fit["companion"] = companion_search(
            data, resolution, primary, primary_priors, star_fit["chi2_red"], None if nostar is None else nostar[0])
        start = vm.System(star=base, env=start.env)
    elif star and star_model != "point":
        raise ValueError(f"star_model must be 'point', 'disk', 'ellipse' or 'point_companion', not {star_model!r}")
    wavel0 = float(np.median(np.concatenate([np.ravel(np.asarray(d.wavel)) for d in datasets])))
    img0 = image_of(start, star)
    pixel0 = float(img0.pixel_scale_mas)
    n0 = int(np.shape(img0.log_brightness)[0])
    # The evidence Jacobian (and CLEAN's columns) are data points x pixels:
    # cap the grid so that product stays within JACOBIAN_BUDGET. MATISSE's
    # 8-26k points on 255^2 pixels needed 432 GB on an 80 GB A100 (job
    # 18073823). The cap only bites for the largest datasets.
    max_pix = min(max_pix or MAX_PIX, int(np.sqrt(JACOBIAN_BUDGET / max(npts, 1))))
    want = field if field is not None else (n0 * pixel0 * grow if grow > 1 else None)
    if want is None and n0 > max_pix:  # starting_image's own grid can exceed the cap
        want = n0 * pixel0
    q = None
    if want is not None:
        n = int(np.ceil(want / pixel0)) | 1
        if n > max_pix:  # the largest odd size within the cap, coarser pixels
            n = max_pix - 1 if max_pix % 2 == 0 else max_pix
        pixel = want / n if n * pixel0 < want else pixel0
        x = (np.arange(n) - (n - 1) / 2) * pixel
        if prior_spec:
            sigma = prior_spec[1] / 2.3548
            q = np.exp(-0.5 * (x[None, :] ** 2 + x[:, None] ** 2) / sigma**2)
            log_start = jnp.asarray(np.log(q))
        else:
            log_start = jnp.zeros((n, n))
        # The hole under the star (starting_image's hole_mas) is kept on the
        # new grid: without it the star and the envelope next to it trade flux.
        hole = hole_support(x, hole_mas)
        new = vm.Image(log_start, pixel, flux=img0.flux, support=None if hole is None else jnp.asarray(hole))
        start = vm.System(star=start.star, env=new) if star else new
        img0 = new
    if sparco and init != "clean":
        raise ValueError("sparco needs init='clean' (the image's spectrum is set on the CLEAN start)")
    if init == "clean":
        # Gradient CLEAN (virgil.imaging.clean) on the same grid, relative to
        # the analytic star if there is one; its components convolved with
        # the beam become the starting image. A start with the right layout
        # keeps closure-phase residuals away from ±π.
        n_img, pix_img = int(np.shape(img0.log_brightness)[0]), float(img0.pixel_scale_mas)
        support = getattr(img0, "support", None)
        # CLEAN's setup costs one Jacobian-vector product per pixel, each a
        # full transform: run it on a coarser grid over the same field (it is
        # only a starting image), then resample.
        n_c = min(n_img, CLEAN_MAX_PIX)
        pix_c = n_img * pix_img / n_c
        support_c = support
        if support is not None and n_c != n_img:
            from scipy import ndimage

            support_c = ndimage.zoom(np.asarray(support, float), n_c / n_img, order=0) > 0.5
        cleaned = safe_clean(data, n_c, pix_c, base=start.star if star else None, support=support_c,
                        max_iterations=clean_iters, target_chi2_red=1.0, gain=clean_gain,
                        scales_mas=tuple(f * resolution.minor_mas for f in clean_scales),
                        base_priors={k.removeprefix("star."): v for k, v in star_priors.items()} or None)
        if star_priors:  # the base's parameters CLEAN fitted along with the components
            fitted = cleaned.model
            if isinstance(start.star, vm.System):  # a System base: its parts are siblings of "clean"
                fitted_base = vm.System(**{k: v for k, v in fitted.components.items() if k != "clean"})
            else:
                fitted_base = fitted.base
            start = vm.System(star=fitted_base, env=start.env)
        restore = vm_beam(resolution.major_mas * mean_blur, resolution.minor_mas * mean_blur, resolution.pa_deg)
        restored = np.clip(np.asarray(cleaned.restored(restore)), 0.0, None)
        if n_c != n_img:
            from scipy import ndimage

            restored = np.clip(ndimage.zoom(restored, n_img / n_c, order=1), 0.0, None)[:n_img, :n_img]
            if restored.shape != (n_img, n_img):  # zoom can come up a pixel short
                restored = np.pad(restored, [(0, n_img - restored.shape[0]), (0, n_img - restored.shape[1])], mode="edge")
        restored = restored / restored.max() + 1e-3  # a floor, so no pixel starts switched off
        new = vm.Image(jnp.asarray(np.log(restored)), pix_img, support=support, flux=1.0)
        clean_comp = None
        if star:
            # CLEAN fits normalised visibilities with the base's flux fixed at
            # 1, so when the data want little or no compact star its
            # components' total runs away (1e163 in campaign 18107553). As a
            # star fraction it is bounded: below F_MIN the star is absent.
            total = float(np.sum(cleaned.components))
            f_clean = 1.0 / (1.0 + total) if np.isfinite(total) and total >= 0 else 0.0
            if guard and f_clean < F_MIN:
                return without_star("clean_flux", f"star fraction {f_clean:.3g} < {F_MIN:g} (components total {total:.3g})")
            # The start's flux ratio: the exact star fraction for this star
            # and the CLEAN image's shape (crosscheck.starts.star_fraction).
            u_, v_, w_, v2_, err_ = v2_table(data)
            f_star = linear_starts.star_fraction(np.asarray(start.star.model(u_, v_, w_)), np.asarray(new.model(u_, v_, w_)),
                                                 v2_, err_)
            flux = (1.0 - f_star) / f_star if f_star > 0 else np.inf
            flux = inside(dist.LogUniform(FLUX_FLOOR, FLUX_CAP), flux)
            clean_comp = clean_companion(cleaned.components, pix_c, resolution)
            print(f"CLEAN with the star: star fraction {f_clean:.3g} from the components, {f_star:.3g} fitted; "
                  f"env/star start {flux:.3g}; brightest cluster beyond 1.5 beams: {clean_comp}", flush=True)
        else:
            flux = float(img0.flux)
        new = eqx.tree_at(lambda m: m.flux, new, max(flux, 1e-3))
        if sparco and star:
            new = eqx.tree_at(lambda m: m.flux, new, PowerLaw(max(flux, 1e-3), 0.0, wavel0))
        start = vm.System(star=start.star, env=new) if star else new
        img0 = new
        clean_info = (cleaned.stop, float(np.ravel(cleaned.chi2_red)[-1]))
    else:
        clean_info = clean_comp = cleaned = None
        if star:  # starting_image's flux, within the fixed prior
            flux = inside(dist.LogUniform(FLUX_FLOOR, FLUX_CAP), flux_value(img0.flux))
            img0 = eqx.tree_at(lambda m: m.flux, img0, flux)
            start = vm.System(star=start.star, env=img0)
    if halo:
        parts = {"star": start.star, "env": start.env} if star else {"env": start}
        start = vm.System(**parts, halo=vm.Resolved(1.0))
    npix = int(np.shape(img0.log_brightness)[0])
    pixel = float(img0.pixel_scale_mas)
    fov = npix * pixel

    ellipse = None
    if star and guard:
        # The sanity guard (C2g): the star start with an elliptical Gaussian
        # environment must fit about as well as that environment alone (it
        # can, with a faint star), or the star is dropped.
        ellipse, record = fit_ellipse(dict(fov=fov, star=True, data=data, resolution=resolution, start=start,
                                           img0=img0), n_starts=n_starts, record=True)
        worse = not ellipse[0] <= STAR_GUARD * nostar[0]
        print(f"star guard: χ²/N {ellipse[0]:.4g} with the star (converged {record['converged']}, parametric fit "
              f"converged {star_fit['converged']}), {nostar[0]:.4g} without", flush=True)
        if worse:
            return without_star("chi2", f"χ²/N {ellipse[0]:.4g} > {STAR_GUARD:g} x {nostar[0]:.4g}")
        # Only the star's own fit must converge: the environment's ellipse is
        # often degenerate next to a star (a faint or bare one), so LM runs
        # out of steps without that meaning anything is wrong.
        if not star_fit["converged"]:
            return without_star("not_converged", "the parametric star fit")

    extras, extra_priors = {}, {}
    if gauss_halo:
        extra_priors = {"ghalo.fwhm": dist.LogUniform(resolution.minor_mas, fov),
                        "ghalo.flux": dist.LogUniform(FLUX_FLOOR, 1000.0)}
        extras["ghalo"] = vm.EllipticalGaussian(inside(extra_priors["ghalo.fwhm"], 0.3 * fov), 1.0, 0.0,
                                                flux=inside(extra_priors["ghalo.flux"], 0.1))
        start = vm.System(**(start.components if isinstance(start, vm.System) else {"env": start}), **extras)
    path = "env" if (star or halo or gauss_halo) else None
    # With a star, the image flux is relative to it, with the fixed bounds
    # [FLUX_FLOOR, FLUX_CAP] (C2g). Jeffreys priors (Ben, 2026-10-05): fluxes
    # are scale parameters, so log-uniform, with stated bounds.
    priors = image_priors(start) | (flux_priors(FLUX_CAP, sparco) if star else {}) | star_priors
    if halo:
        priors |= {"halo.flux": dist.LogUniform(FLUX_FLOOR, 1000.0)}
    priors |= extra_priors
    others = () if star else (Centroid(0.1 * resolution.minor_mas, path=path),)
    return dict(label=label, files=files, star=star, halo=halo, data=data, npts=npts,
                star_model=star_model if star else "none", star_priors=star_priors, sparco=bool(sparco and star),
                wavel0=wavel0, extras=extras, extra_priors=extra_priors,
                resolution=resolution, start=start, img0=img0, npix=npix, pixel=pixel, fov=fov,
                path=path, priors=priors, others=others, q=q, adaptive=field is None,
                field_source=field_source or "data" + (f" x{grow:g}" if grow > 1 else ""),
                cleaned=cleaned, restore=restore if cleaned is not None else None,
                clean_info=clean_info, star_rejected="", star_fit=star_fit, ellipse=ellipse,
                clean_companion=clean_comp, n_starts=n_starts, no_star_setup=no_star_setup if star else None)


def residual_diagnostics(model, data):
    """Per dataset: χ² per point for V² and for closure phases separately,
    and the mean whitened V² residual on the shortest 20% of spatial
    frequencies (near 0 when short baselines are not traded for long ones;
    MACIM's selection test in the 2012 contest). Residuals come in the order
    V², whitened closure phases, then (for correlated closure phases) one
    periodic penalty per phase; both closure-phase components contribute to χ²."""
    out = []
    for d in data if isinstance(data, list) else [data]:
        r = np.asarray(whitened_residuals(model, d))
        nv = int(np.asarray(d.vis).size)
        n_cp = int(d.n_independent) - nv
        freq = np.hypot(np.asarray(d.u)[:nv], np.asarray(d.v)[:nv]) / np.asarray(d.wavel).ravel()[:nv] if np.asarray(d.wavel).size >= nv \
            else np.hypot(np.asarray(d.u)[:nv], np.asarray(d.v)[:nv]) / float(np.ravel(d.wavel)[0])
        short = freq <= np.quantile(freq, 0.2)
        out.append({
            "chi2_v2": float(np.sum(r[:nv] ** 2) / max(nv, 1)),
            "chi2_cp": float(np.sum(r[nv:] ** 2) / n_cp) if n_cp > 0 else None,
            "short_mean": float(np.mean(r[:nv][short])),
            "n_v2": nv, "n_cp_indep": n_cp,
        })
    return out


def grown_setup(task, data_dir, halo=False, star=None, init="moments", smoke=False, field=None, clean_scales=(0.0,),
                **scene):
    """setup(), with a field chosen from the data grown (x1.5) while doing so
    lowers χ² by more than 10% in a probe fit at a middle weight: what a
    contestant would try. (Flux at the image edge is not a usable trigger:
    the support and centroid prior keep it off the edge even when the field
    is too small.) Returns the setup and the growth record."""
    opts = dict(star=star, init=init, clean_iters=50 if smoke else 3000, n_starts=N_STARTS_SMOKE if smoke else N_STARTS,
                field_override=field, clean_scales=clean_scales, **scene)
    s = setup(task, data_dir, halo, **opts)
    growth = []
    if s["adaptive"]:
        def probe(setup_dict):
            r = fit(setup_dict["start"], setup_dict["priors"], setup_dict["data"],
                    [MaxEntropy(float(WEIGHTS[6]), prior=setup_dict["q"], path=setup_dict["path"]), *setup_dict["others"]],
                    max_steps=50 if smoke else 3000)
            return float(np.sum(r.info["chi2_red"]))

        grow, chi2 = 1.0, probe(s)
        growth.append((round(s["fov"], 3), round(chi2, 3)))
        for _ in range(GROWTHS):
            try:
                bigger = setup(task, data_dir, halo, grow * 1.5, **opts)
            except ValueError as e:  # a probe of a larger field that cannot be set up ends the growth, it does not stop the run
                print(f"field x{grow * 1.5:g} could not be set up ({e}); keeping the field x{grow:g}", flush=True)
                break
            chi2_big = probe(bigger)
            growth.append((round(bigger["fov"], 3), round(chi2_big, 3)))
            if chi2_big > 0.9 * chi2:
                break
            grow, chi2, s = grow * 1.5, chi2_big, bigger
    return s, growth


ADAM_STEPS, ADAM_STEPS_SMOKE = 2000, 50


def adam_fallback(curve, start, priors, data, reg, path, q, others, smoke=False):
    """Refit, with Adam from the L-BFGS values, each weight whose fit did not
    converge or stopped "stalled"/"limit"; keep the refit only if its objective
    χ² + w·P (what the fit minimises at that weight) is lower than L-BFGS's,
    both penalties evaluated the same way. A refit with only a lower χ² may
    just be less regularised, and would bend the χ²(w) curve that the
    discrepancy principle reads. Returns (a patched LCurve, the method kept per
    weight, and per weight a (lbfgs, adam) χ²/N pair, or None where no refit
    was made).

    Not used for MaxEnt (``reg == "mem"``): round-1 runs and the bench's mem arm
    keep L-BFGS's fits, unchanged."""
    n = len(curve.weights)
    if reg == "mem":
        return curve, ["lbfgs"] * n, [None] * n
    results, chi2, chi2_red, penalty, methods, both = [], [], [], [], [], []
    n_total = n_independent(data)
    for w, r, c, cr, pen in zip(curve.weights, curve.results, curve.chi2, curve.chi2_red, curve.penalty):
        method, pair = "lbfgs", None
        if not r.info.get("converged", False) or r.info.get("stop") in ("stalled", "limit"):
            weighted = regulariser(reg, np.asarray(w, dtype=float), path, q)
            value_l = float(weighted.value(_reference(r.model)))  # w·P of the L-BFGS fit, as for Adam's below
            if abs(value_l / float(w) - float(pen)) > 1e-3 * max(abs(float(pen)), 1e-12):
                print(f"warning: penalty recomputed for the L-BFGS fit ({value_l / float(w):.6g}) differs from "
                      f"l_curve's ({float(pen):.6g}) at weight {float(w):.4g}", flush=True)
            a = fit(start, priors, data, [weighted, *others], init=r.values, method="adam",
                    max_steps=ADAM_STEPS_SMOKE if smoke else ADAM_STEPS)
            c_a = float(sum(a.info["chi2"]))
            pair = (round(float(c) / n_total, 4), round(c_a / n_total, 4))
            value_a = float(weighted.value(_reference(a.model)))
            if np.isfinite(c_a) and np.isfinite(value_a) and c_a + value_a < float(c) + value_l:
                r, c, method = a, c_a, "adam"
                cr = [x / m for x, m in zip(a.info["chi2"], a.info["ndata"])]
                pen = value_a / float(w)
        results.append(r), chi2.append(c), chi2_red.append(cr), penalty.append(pen)
        methods.append(method), both.append(pair)
    patched = dataclasses.replace(curve, chi2=np.asarray(chi2), chi2_red=np.asarray(chi2_red),
                                  penalty=np.asarray(penalty), results=results)
    return patched, methods, both


CLASSIC_MAXENT_BUDGET = 1.0e8  # Jacobian elements (data points x pixels): above it classic_maxent's SVD per weight is skipped


def weight_report(curve, data, reg, path, index, jacobian_size=0):
    """The weight-selection rule's numbers (virgil design/regulariser_weight_selection.md):
    the discrepancy weight (None if no weight reaches χ²/N = 1), the L-curve
    corner (None if the sweep is too short), for MaxEnt Gull and Skilling's
    classic weight, and the dataset that binds at the chosen weight."""
    def guarded(f, *a):
        try:
            v = f(*a)
            return None if v is None else float(v)
        except Exception as e:  # report-only numbers must not stop a fit (classic_maxent needs an image path in a System)
            return f"unavailable ({type(e).__name__}: {e})"

    chi2_here = np.atleast_1d(np.asarray(curve.chi2_red, dtype=float)[index])  # (weights, datasets), or (weights,) for one OIData
    datasets = data if isinstance(data, list) else [data]
    report = {"discrepancy": guarded(curve.discrepancy), "corner": guarded(curve.corner),
              "binding_dataset": int(np.argmax(chi2_here)), "chi2_red_by_dataset": [round(float(x), 4) for x in chi2_here],
              "n_datasets": len(datasets)}
    if reg == "mem":
        # One SVD of the data x pixels Jacobian per weight: hours at 400 mas (2016 Obj2 MEM, job 18593342).
        report["classic_maxent"] = (f"skipped (Jacobian of {jacobian_size:.3g} elements > {CLASSIC_MAXENT_BUDGET:g})"
                                    if jacobian_size > CLASSIC_MAXENT_BUDGET else guarded(curve.classic_maxent, data, path))
    return report


def run(task, data_dir, out_dir, smoke=False, halo=False, star=None, init="moments", settings=None, label=None,
        reg="mem", field=None, informed=False, points=0, prune_blobs=False, systematics="none",
        scene=None):
    """MaxEnt L-curve. With ``settings`` (scripts/contest_bench.py's "mem"
    arm) the field, pixels and CLEAN start follow those settings, as for a GP
    member, and the default model of the entropy is the CLEAN starting image
    (as BSMEM and MiRA use a prior image); the chosen image is also rendered
    on the common reference grid (``ref_image``) for scoring. ``reg`` names
    the penalty (REGULARISERS; labels get _<reg> unless MaxEnt), ``field``
    fixes the field (mas), ``informed`` marks a round-2 fit, and ``points``
    adds up to that many point sources (add_point_sources; a CLEAN start).
    ``systematics`` (SYSTEMATICS; no block scales here, which need a Gaussian
    field) adds gains / closure offsets at the median width of the scan: this
    path has no evidence to choose it. ``scene``: more setup() options (scene_options). The field growth
    (grown_setup) is a MaxEnt probe fit at WEIGHTS[6] whatever ``reg`` is, as
    a contestant would size the field before choosing a penalty."""
    t0 = time.time()
    if "block_scales" in SYSTEMATICS[systematics]:
        raise ValueError("block_scales are estimated from a Gaussian-field fit: use --prior gp")
    if reg != "mem":
        init = "clean"  # pixel-basis penalties stall from flat or Gaussian starts (docs/contest_convergence_diagnosis.md)
    if settings is None:
        s, growth = grown_setup(task, data_dir, halo, star, "clean" if points else init, smoke, field, **(scene or {}))
        if points:
            s = add_point_sources(s, points)
    else:
        s, settings = member_setup(task, data_dir, None, smoke, settings)
        growth, init, halo = [], "clean", s["halo"]
        template = np.asarray(s["img0"].brightness).reshape(s["npix"], s["npix"])
        s["q"] = template / template.sum()
    label_ = label
    label, files, star, data, npts = s["label"], s["files"], s["star"], s["data"], s["npts"]
    label = label_ or label + suffix(points, systematics, reg=reg) + ("_prune" if prune_blobs else "")
    fixed = {k: v[0] for k, v in systematic_widths(systematics, scan=False).items()}
    quoted, data = data, with_systematics(data, fixed)
    resolution, start, img0, npix, pixel, fov = (s[k] for k in ("resolution", "start", "img0", "npix", "pixel", "fov"))
    path, priors, others = s["path"], s["priors"], s["others"]
    weights = regulariser_weights(reg, data, smoke)
    options = {"max_steps": 50} if smoke else {"max_steps": 200_000}
    out_dir.mkdir(parents=True, exist_ok=True)
    first = np.asarray(img0.render(npix, fov))  # V² and closure phases have no absolute phases, so no dirty image: show the starting image instead

    def save(curve, methods, chosen, index, report, partial):
        """The npz (and, partial, a note): ``partial`` marks a checkpoint of a sweep that has not finished
        (a job that times out keeps its output); the final call overwrites it."""
        images = np.stack([safe_render(image_of(r.model, star, halo), npix, fov) for r in curve.results])
        reference = {}
        if settings is not None:
            chosen_model = curve.results[index["discrepancy"]].model
            reference = dict(ref_image=render_scene(chosen_model, reference_npix(s["ref_fov"], data_q_max(data)), s["ref_fov"], data_q_max(data)), ref_fov=s["ref_fov"],
                             best_chi2_red=float(np.sum(np.asarray(curve.chi2_red)[index["discrepancy"]])), error_scale=np.nan,
                             star=bool(star), star_rejected=s["star_rejected"], best_log_z=np.nan, flip_dchi2=np.nan)
        np.savez_compressed(
            out_dir / f"{label}.npz", **reference,
            weights=np.asarray(curve.weights), chi2=np.asarray(curve.chi2),
            chi2_red=np.asarray(curve.chi2_red), penalty=np.asarray(curve.penalty),
            images=images, start=first, pixel_scale_mas=pixel, npix=npix,
            beam=np.array([resolution.major_mas, resolution.minor_mas, resolution.pa_deg]),
            halo_flux=np.array([float(r.model.halo.flux) if halo else 0.0 for r in curve.results]),
            converged=np.array([bool(r.info.get("converged", False)) for r in curve.results]),
            methods=np.array(methods), adam_used=np.array([m == "adam" for m in methods]),
            chosen=json.dumps(chosen), index=json.dumps(index), regulariser=reg, informed=bool(informed),
            systematics_mode=systematics, systematics_widths=json.dumps(fixed),
            point_sources=json.dumps(s.get("point_record"), default=float),
            weight_report=json.dumps(report), partial=partial, n_weights_planned=len(weights),
        )
        if partial:
            (out_dir / f"{label}.txt").write_text(
                f"PARTIAL: {len(curve.weights)} of {len(weights)} weights fitted at {time.time() - t0:.0f}s; "
                f"chosen={chosen}\n")

    def checkpoint(curve):
        try:
            _, _, chosen_, index_, _ = choose_weights(curve, reg, strict=False)
        except RuntimeError:  # nothing finite yet
            return
        save(curve, ["lbfgs"] * len(curve.results), chosen_, index_, {}, True)

    # A warm-started LogSum sweep collapses to one pixel (virgil design/sparse_imaging.md).
    curve = sweep(start, priors, data, regulariser(reg, 1.0, path, s["q"]), jnp.asarray(weights), others,
                  warm_start=reg != "logsum", on_point=checkpoint, **options)
    curve, methods, adam_chi2 = adam_fallback(curve, start, priors, data, reg, path, s["q"], others, smoke)

    # discrepancy() is None when no weight reaches χ² = N (the data are then
    # under-fitted at every weight: worth a note); corner() needs >= 3 points.
    # Fall back to the L-curve corner, then to the weakest weight.
    # Non-finite fits (logged) take no part in choosing the weight.
    ok, good, chosen, index, reached = choose_weights(curve, reg)
    # The output first (the report's classic_maxent is an SVD per weight, hours at 400 mas, job 18593342).
    save(curve, methods, chosen, index, {}, False)
    report_w = weight_report(curve, data, reg, path, index["discrepancy"], npts * npix**2)
    save(curve, methods, chosen, index, report_w, False)

    best = curve.results[index["discrepancy"]].model
    pruned = None
    if prune_blobs:
        result, pruned = prune(curve.results[index["discrepancy"]], priors, data,
                               regulariser(reg, chosen["discrepancy"], path, s["q"]), others, path, npix, fov,
                               resolution, options)
        best = result.model
        np.savez_compressed(out_dir / f"{label}_pruned.npz", image=np.asarray(image_of(best).render(npix, fov)),
                            pixel_scale_mas=pixel, npix=npix, prune=json.dumps(pruned), informed=bool(informed),
                            raw_chi2_red=chi2_red(best, quoted))
    report = diagnose(best, data, [regulariser(reg, chosen["discrepancy"], path, s["q"]), *others])
    summary = (
        f"task={task} label={label} files={files} star={star} star_rejected={s['star_rejected'] or None} halo={halo}\n"
        f"raw chi2/N={chi2_red(best, quoted):.4g} (quoted errors, no nuisances)\n"
        f"{informed_line(informed)} regulariser={reg} point_sources={s.get('point_record')}\n"
        f"systematics={systematics}"
        f"{' (widths ' + str(fixed) + ': fixed at the scan median, no evidence on this path)' if fixed else ''}\n"
        f"field_source={s['field_source']} clean={s['clean_info']} star_source={spec_of(task).get('star_source')} prior={spec_of(task).get('prior')} wavel={spec_of(task).get('wavel')} growth(fov, chi2/N)={growth}\n"
        f"virgil={virgil.__version__} from {virgil.__file__}\n"
        f"points={npts} npix={npix} pixel={pixel:.4g} mas fov={fov:.4g} mas "
        f"beam={resolution.major_mas:.3g}x{resolution.minor_mas:.3g} mas\n"
        f"weights={np.asarray(curve.weights).round(4).tolist()}\n"
        f"chi2_red={np.asarray(curve.chi2_red).round(3).tolist()}\n"
        f"halo_flux={[round(float(r.model.halo.flux), 4) for r in curve.results] if halo else None}\n"
        f"chi2/N at the grid point nearest the discrepancy weight={chi2_red(best, quoted):.4g} (quoted errors, no nuisances; "
        f"the corner weight when discrepancy_reached=False)"
        f"{' after pruning ' + str(pruned) if prune_blobs else ''}\n"
        f"growth probe: MaxEnt at WEIGHTS[6]={float(WEIGHTS[6]):.4g} from the setup's start, whatever the regulariser\n"
        f"chosen={chosen} discrepancy_reached={reached}\nconverged={[bool(r.info.get('converged', False)) for r in curve.results]}\n"
        f"methods (lbfgs/adam)={methods}\nadam_used={[m == 'adam' for m in methods]} "
        f"lbfgs vs adam chi2/N per weight (None: not refitted)={adam_chi2}\n"
        f"weight report={report_w}\n"
        f"residuals at the discrepancy weight={residual_diagnostics(best, quoted)}\n"
        f"residuals at the corner weight={residual_diagnostics(curve.results[index['corner']].model, quoted)}\n"
        f"elapsed={time.time() - t0:.0f}s\n\n{report}\n"
    )
    (out_dir / f"{label}.txt").write_text(summary)
    print(summary)
    plot(label, curve, index, chosen, npts, img0, star, halo, fov, npix, resolution, out_dir)


def plot(label, curve, index, chosen, npts, img0, star, halo, fov, npix, resolution, out_dir):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    axes[0].plot(curve.penalty, curve.chi2, "o-", ms=4)
    for (name, k), m in zip(index.items(), ("s", "^")):
        axes[0].plot(curve.penalty[k], curve.chi2[k], m, ms=11, mfc="none", mew=2, label=f"{name}: w = {chosen[name]:.3g}")
    axes[0].axhline(npts, color="k", ls=":", lw=1, label="χ² = N")
    positive = np.all(np.asarray(curve.penalty) > 0) and np.all(np.asarray(curve.chi2) > 0)
    scale = "log" if positive else "linear"  # NaN or zero values cannot go on log axes
    axes[0].set(xscale=scale, yscale=scale, xlabel="negative entropy", ylabel="χ²", title=label)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    plot_model(img0, fov_mas=fov, npix=npix, ax=axes[1], beam=resolution, title="starting image")
    for ax, name in zip(axes[2:], ("corner", "discrepancy")):
        r = curve.results[index[name]]
        plot_model(image_of(r.model, star, halo), fov_mas=fov, npix=npix, ax=ax, beam=resolution,
                   title=f"{name}: w = {float(curve.weights[index[name]]):.3g}, χ²/N = {float(np.sum(r.info['chi2_red'])):.2f}")
    plt.tight_layout()
    fig.savefig(out_dir / f"{label}.png", dpi=110)


# Gaussian-process prior (--prior gp): σ and ℓ (in beam minor axes) on a grid,
# chosen by virgil's Laplace evidence, which also compares isotropic against
# anisotropic fields, and runs with against without a star (same data).
GP_SIGMAS = (1.0, 2.0, 4.0)
GP_LENGTHS = (0.5, 1.0, 2.0)
# When the evidence's best point lies on an edge of the grid, the grid grows by
# a factor of 2 in that direction (log-spaced: σ and ℓ are scale parameters),
# up to GP_GROWTHS times and within these limits (ℓ in beam minor axes). In the
# first campaign 2004 data2 and 2024 Obj2 GRAVITY chose σ = 4 and ℓ = 0.5 beam,
# both edges.
GP_SIGMA_LIMITS = (0.25, 16.0)
GP_LENGTH_LIMITS = (0.125, 8.0)
GP_GROWTHS = 3


def fit_ellipse(s, n_starts=None, record=False, seed=0):
    """Elongation and position angle from the data: an elliptical Gaussian
    (with the analytic star, if ``s["star"]``), from ``n_starts`` starts
    jittered about the data's second moments (crosscheck.starts), never on a
    prior bound (C2g; the old fixed start, ratio 0.7 and PA 0, stalled there).
    Returns (χ²/N, ratio, pa, fwhm), and with ``record`` the multistart's
    record too."""
    fov, star = s["fov"], s["star"]
    n_starts = s.get("n_starts", N_STARTS) if n_starts is None else n_starts
    priors = {"env.fwhm": dist.LogUniform(0.01, fov), "env.ratio": dist.LogUniform(0.05, 1.0),
              "env.pa": dist.Uniform(*PA_PRIOR)}
    m = moments_of(s["data"])
    # Without moments, a few beams, not a fraction of the field: in a 4x field,
    # 0.3 fov is so broad that V² ≈ 0 on every baseline, and LM crawled on 2010
    # low-H past the 30-min watchdog (campaign 18098742 tasks 75/77/79).
    fwhm0 = 2.3548 * m["sigma_major"] if np.isfinite(m["sigma_major"]) and m["sigma_major"] > 0 \
        else min(0.3 * fov, 3.0 * s["resolution"].major_mas)
    centre = {"env.fwhm": fwhm0, "env.ratio": float(np.clip(m["ratio"] if np.isfinite(m["ratio"]) else 0.7, 0.25, 0.95)),
              "env.pa": float(m["pa"]) if np.isfinite(m["pa"]) else 90.0}
    if star:
        priors["env.flux"] = dist.LogUniform(FLUX_FLOOR, FLUX_CAP)
        centre["env.flux"] = flux_value(s["img0"].flux)
    starts = jittered_starts(centre, priors, n_starts, np.random.default_rng(seed),
                             {"env.fwhm": ("log", 0.5), "env.ratio": ("add", 0.1), "env.pa": ("add", 30.0),
                              "env.flux": ("log", 1.0)})
    for p in starts:
        p["env.ratio"] = inside(priors["env.ratio"], float(np.clip(p["env.ratio"], 0.25, 0.95)))

    def scene(p):
        env = vm.EllipticalGaussian(p["env.fwhm"], p["env.ratio"], p["env.pa"], flux=p.get("env.flux", 1.0))
        return vm.System(star=s["start"].star, env=env) if star else vm.System(env=env)

    best, rec = multistart_fit(scene, priors, s["data"], starts, max_steps=300)
    out = (rec["chi2_red"], float(best.values["env.ratio"]), float(best.values["env.pa"]) % 180.0,
           float(best.values["env.fwhm"]))
    return (out, rec) if record else out


def oriented_template(template, pa_deg, pixel, n):
    """The template resampled onto a grid whose "up" axis points at pa_deg.
    The rotation's sign is checked numerically against virgil's rendering,
    rather than assumed."""
    from scipy import ndimage

    target = template / template.sum()
    best = None
    for sign in (1.0, -1.0):
        rotated = np.clip(ndimage.rotate(template, sign * pa_deg, reshape=False, order=1), 0, None) + 1e-6
        img = vm.Image(jnp.asarray(np.log(rotated)), pixel, rotation_deg=pa_deg)
        sky = np.asarray(img.render(n, n * pixel))
        corr = float(np.sum((sky - sky.mean()) * (target - target.mean())))
        if best is None or corr > best[0]:
            best = (corr, rotated)
    return best[1]


# Ensemble members (--member m, one per array task): one CLEAN-started GP fit each,
# with a randomised start, as the 2012 random-start entries, the 2018 winner's
# chains and PYRA/MYTHRA (2022-24) did. Members cycle the central source through
# none, an unresolved star and a resolved (uniform-disk) star, so the evidence
# compares them on the same data: the 2004 contest page listed "a limb darkened
# star with one or more spots" among the possible targets, and a point star fits
# a resolved one badly (2004 data2, first campaign: χ²/N 1300 against 21).
# Tasks with halo_members=True also try an over-resolved halo (members 4-7);
# tasks with sparco=True give the image its own spectral index.
MEMBER_STARS = ("point", "none", "disk", "none")
MEMBER_FIELDS = (1.0, 2.0, 4.0)  # x the data-chosen field (contest-given fields stay fixed)
MEMBER_OVERSAMPLE = (2.0, 3.0, 4.0)  # pixels per Nyquist pixel
MEMBER_GAINS = (0.05, 0.1, 0.2)  # CLEAN loop gain
REF_NPIX = 257  # the common grid members are rendered onto, max(MEMBER_FIELDS) x the base field


def member_settings(task, member):
    spec = spec_of(task)
    rng = np.random.default_rng(1000 * task + member)
    star_model = MEMBER_STARS[member % len(MEMBER_STARS)]
    return {
        "star": star_model != "none",
        "star_model": star_model,
        "halo": bool(spec.get("halo_members", False)) and member % 8 >= 4,
        "sparco": bool(spec.get("sparco", False)),
        "field": float(rng.choice(MEMBER_FIELDS)),
        "oversample": float(rng.choice(MEMBER_OVERSAMPLE)),
        "clean_gain": float(rng.choice(MEMBER_GAINS)),
    }


def member_setup(task, data_dir, member, smoke=False, settings=None):
    """One member's setup. ``settings`` (a dict like member_settings's) fixes
    the configuration instead of drawing it from the member's seed."""
    settings = dict(settings) if settings is not None else member_settings(task, member)
    opts = dict(star=settings["star"], init="clean", clean_iters=50 if smoke else 3000,
                oversample=settings["oversample"], clean_gain=settings["clean_gain"],
                star_model=settings["star_model"], sparco=settings["sparco"],
                mean_blur=settings.get("mean_blur", 1.0), n_starts=N_STARTS_SMOKE if smoke else N_STARTS)
    if spec_of(task).get("field") is not None:
        settings["field"] = 1.0  # a contest-given field is used as given
    s = setup(task, data_dir, settings["halo"], settings["field"], **opts)
    s["ref_fov"] = reference_fov(task, data_dir)
    return s, settings


def reference_fov(task, data_dir):
    """The common grid's field for all members of a task: max(MEMBER_FIELDS) x the base field
    of the task's own configuration (its default star, a moments start), so it
    does not depend on any member's random settings or star choice."""
    base = setup(task, data_dir, False, 1.0, star=None, init="moments", guard=False)
    return max(MEMBER_FIELDS) * base["fov"]


def grow_grid(evaluate, lengths, sigmas, length_limits, sigma_limits, growths):
    """Evaluate ``evaluate(ℓ, σ)`` (a log evidence) on the grid ``lengths`` x
    ``sigmas``. While the best point lies on an edge that can move (within the
    limits), add a row or column a factor of 2 beyond it, up to ``growths``
    times. Returns the sorted axes and a dict of values keyed by (ℓ, σ)."""
    ls, ss = sorted(lengths), sorted(sigmas)
    values = {}

    def fill():
        for ell_ in ls:
            for sig in ss:
                if (ell_, sig) not in values:
                    values[ell_, sig] = evaluate(ell_, sig)

    fill()
    for _ in range(growths):
        ell_, sig = max(values, key=lambda k: values[k] if np.isfinite(values[k]) else -np.inf)
        grew = False
        if ell_ == ls[0] and ell_ / 2 >= length_limits[0]:
            ls.insert(0, ell_ / 2)
            grew = True
        elif ell_ == ls[-1] and ell_ * 2 <= length_limits[1]:
            ls.append(ell_ * 2)
            grew = True
        if sig == ss[0] and sig / 2 >= sigma_limits[0]:
            ss.insert(0, sig / 2)
            grew = True
        elif sig == ss[-1] and sig * 2 <= sigma_limits[1]:
            ss.append(sig * 2)
            grew = True
        if not grew:
            break
        fill()
    return ls, ss, values


def gp_fit(s, smoke=False):
    """The GP fits of one setup ``s``: σ and ℓ on a grid grown by evidence,
    isotropic and (for an elongated source) anisotropic fields, then
    MacKay's error-scale check, with a refit on rescaled errors when it is
    far from 1. Returns a dict of the grid, the winner and the final fit."""
    star, halo, res = s["star"], s["halo"], s["resolution"]
    data = with_systematics(s["data"], s.get("systematics"))
    img0, n, pixel = s["img0"], s["npix"], s["pixel"]
    template = np.asarray(img0.brightness).reshape(n, n)
    support = getattr(img0, "support", None)
    flux0 = img0.flux  # a number, or with SPARCO a PowerLaw
    flux_cap = FLUX_CAP  # fixed (C2g), not 10x a start that CLEAN can run away with
    ell = s.get("ellipse") or fit_ellipse(s)  # (chi2/N, ratio, pa, fwhm); setup's guard may have fitted it
    sigmas0 = GP_SIGMAS[1:2] if smoke else GP_SIGMAS
    # ℓ in beam minor axes, capped at the field: a correlation length beyond
    # the field is unconstrained by the image, so the evidence's grid stops
    # there. This is a guard; it did not bind in the 2022 AMI smoke (ℓ = 103 mas,
    # one beam minor axis, inside the 618 mas field).
    fov = n * pixel
    lengths0 = sorted({min(f * res.minor_mas, fov) for f in (GP_LENGTHS[1:2] if smoke else GP_LENGTHS)})
    length_limits = [min(f * res.minor_mas, fov) for f in GP_LENGTH_LIMITS]
    options = {"max_steps": 50} if smoke else {}

    def scene(field, rotation):
        env = vm.Image(field, pixel, support=support, flux=flux0, rotation_deg=rotation)
        parts = {"star": s["start"].star, "env": env} if star else {"env": env}
        if halo:
            parts["halo"] = vm.Resolved(1.0)
        return vm.System(**parts, **s.get("points", {}), **s.get("extras", {}))

    others = () if star else (Centroid(0.1 * res.minor_mas, path="env"),)
    # Every fitted parameter other than the image's latents (the evidence's
    # Occam factor integrates over these: log_evidence_full).
    extra = ((flux_priors(flux_cap, s["sparco"]) | s["star_priors"]) if star else {}) | (
        {"halo.flux": dist.LogUniform(FLUX_FLOOR, 1000.0)} if halo else {}) | s.get("point_priors", {}) | s.get("extra_priors", {})
    variants = {"iso": (template, 0.0, lambda length: length)}
    if ell[1] < 0.9:  # elongated enough for an anisotropic field to differ
        r = ell[1]
        variants["aniso"] = (oriented_template(template, ell[2], pixel, n), ell[2],
                             lambda length: (length / np.sqrt(r), length * np.sqrt(r)))
    results, log_z, axes = {}, {}, {}  # results[name, ℓ, σ]; log_z[name][i, j] over axes[name] = (ℓs, σs)
    for name, (mean, rotation, lens) in variants.items():
        previous = None

        def evaluate(length, sigma, name=name, mean=mean, rotation=rotation, lens=lens):
            nonlocal previous
            field = GaussianField(np.zeros((n, n)), sigma, lens(length), mean=mean)
            sc = scene(field, rotation)
            r_ = fit(sc, image_priors(sc) | extra, data, others,
                     init=None if previous is None else previous.values, **options)
            previous = r_
            results[name, length, sigma] = r_
            log_z_ = float(log_evidence(r_, data))
            if s.get("on_fit"):  # checkpoint: a job that times out keeps the best fit so far
                s["on_fit"](name, length, sigma, log_z_, r_)
            return log_z_

        ls, ss, values = grow_grid(evaluate, lengths0, sigmas0, length_limits, GP_SIGMA_LIMITS,
                                   0 if smoke else GP_GROWTHS)
        axes[name] = (ls, ss)
        log_z[name] = np.array([[values[ell_, sig] for sig in ss] for ell_ in ls])
    best_of = {name: np.unravel_index(np.nanargmax(g), g.shape) for name, g in log_z.items()}
    best_name = max(log_z, key=lambda k: np.nanmax(log_z[k]))
    i_b, j_b = best_of[best_name]
    best_length, best_sigma = axes[best_name][0][i_b], axes[best_name][1][j_b]
    winner = results[best_name, best_length, best_sigma]
    best_log_z = float(log_z[best_name][i_b, j_b])

    # Error-scale check (MacKay's fixed point, virgil.imaging.error_scale): the
    # evidence assumes correct error bars, which several contest datasets
    # break on purpose. Far from 1, refit the winner with rescaled errors.
    datasets = data if isinstance(data, list) else [data]
    # error_scale refuses data with gains or closure offsets, so with nuisances
    # the single scale is left at 1 (block scales are estimated before they are added).
    nuisances = bool(s.get("systematics"))
    scale = 1.0 if nuisances else float(error_scale(winner.model, data))
    # MacKay's effective number of parameters, from s² = χ²/(N − γ).
    n_indep = sum(d.n_independent for d in datasets)
    chi2_winner = float(np.sum(np.asarray(winner.info["chi2_red"]) * np.asarray([d.n_independent for d in datasets])))
    gamma = n_indep - chi2_winner / scale**2
    rescaled, data_s = None, data
    if not 1 / 1.3 < scale < 1.3:
        data_s = [d.with_error_scale(scale) for d in datasets]
        data_s = data_s if isinstance(data, list) else data_s[0]
        mean, rotation, lens = variants[best_name]
        field = GaussianField(np.zeros((n, n)), best_sigma, lens(best_length), mean=mean)
        sc = scene(field, rotation)
        rr = fit(sc, image_priors(sc) | extra, data_s, others, init=winner.values, **options)
        rescaled = {"scale": scale, "log_z": float(log_evidence(rr, data_s)),
                    "chi2_red": float(np.sum(rr.info["chi2_red"])), "result": rr}

    def best_result(name):
        i, j = best_of[name]
        return results[name, axes[name][0][i], axes[name][1][j]]

    return dict(template=template, ell=ell, others=others, extra=extra, results=results, log_z=log_z, axes=axes,
                best_of=best_of, best_name=best_name, best_length=best_length, best_sigma=best_sigma, winner=winner,
                best_log_z=best_log_z, scale=scale, n_indep=n_indep, chi2_winner=chi2_winner, gamma=gamma,
                rescaled=rescaled, data_s=data_s, final=rescaled["result"] if rescaled else winner,
                best_result=best_result)


def suffix(points=0, systematics="none", reg="mem"):
    """Label suffixes of the round-2 options."""
    return (f"_{reg}" if reg != "mem" else "") + (f"_ps{points}" if points else "") + SYSTEMATIC_SUFFIX[systematics]


def point_scene(s, points, point_priors):
    """Setup ``s`` with the given point sources added unchanged (no test)."""
    start = s["start"] if isinstance(s["start"], vm.System) else vm.System(env=s["start"])
    s = dict(s, start=vm.System(**start.components, **points), path="env", points=points, point_priors=point_priors)
    s["img0"] = s["start"].env
    if not s["star"]:
        s["others"] = (Centroid(0.1 * s["resolution"].minor_mas, path="env"),)
    return s


def systematics_scan(s, smoke, gains=None, closure=None):
    """gp_fit over every combination of the ``gains`` and ``closure`` widths
    (lists; None leaves a nuisance out), keeping the highest evidence: the
    outer evidence scan of the widths. Sets s["systematics"] to the winner's
    and returns its fit and the scan [(widths, best log Z, raw χ²/N)]. The
    raw χ²/N is on the quoted errors (s["quoted"]), with no nuisances."""
    import itertools

    scan, best = [], None
    for gw, cw in itertools.product(gains or [None], closure or [None]):
        s["systematics"] = {"gains": gw, "closure": cw} if (gw or cw) else None
        g = gp_fit(s, smoke)
        raw = chi2_red(g["winner"].model, s.get("quoted", s["data"]))
        scan.append(({"gains": gw, "closure": cw}, g["best_log_z"], raw))
        if gains or closure:
            print(f"raw χ²/N {raw:.4g} (quoted errors, no nuisances); systematics widths gains={gw} closure={cw}: "
                  f"log Z {g['best_log_z']:.2f}", flush=True)
        if best is None or g["best_log_z"] > best[1]["best_log_z"]:
            best = (dict(s["systematics"] or {}), g)
    s["systematics"] = best[0] or None
    return best[1], scan


def quadrature_floor(r, sigma):
    """The f >= 0 with sum r²σ²/(σ²+f²) = N for whitened residuals ``r`` of errors ``sigma``
    (the data's units): error bars σ -> √(σ²+f²) that make χ²/N = 1; 0 if it is already <= 1."""
    from scipy import optimize

    r, sigma = np.ravel(r), np.ravel(sigma)
    excess = lambda log_f: np.sum((r * sigma) ** 2 / (sigma**2 + np.exp(2 * log_f))) - r.size  # noqa: E731
    if excess(-np.inf) <= 0:
        return 0.0
    return float(np.exp(optimize.brentq(excess, np.log(np.min(sigma)) - 20.0, np.log(np.max(np.abs(r * sigma)) + np.max(sigma)))))


def calibration_report(model, quoted, scales):
    """The calibration estimated from one fit, for the summary: RAW χ²/N on the quoted errors (V² and
    closure phases) first, the block ``scales`` and a V² quadrature floor next to them, and the χ²/N
    with the rescaled errors, which is tautological (about 1 by construction, never a success criterion)."""
    out = {"scales": {k: float(v) for k, v in scales.items()}, "raw": residual_diagnostics(model, quoted),
           "rescaled": residual_diagnostics(model, rescale(quoted, scales)), "v2_floor": []}
    for d in quoted if isinstance(quoted, list) else [quoted]:
        nv = int(np.asarray(d.vis).size)
        out["v2_floor"].append(quadrature_floor(np.asarray(whitened_residuals(model, d))[:nv], np.asarray(d.d_vis)))
    return out


def estimate_block_scales(s, smoke):
    """Per-observable error scales from a GP fit of ``s`` on the gains-free
    data (printing its raw χ²/N first); s["data"] becomes the rescaled data,
    s["quoted"] keeps the quoted errors', s["error_scales"] the scales."""
    s["systematics"] = None
    g = gp_fit(s, smoke)
    quoted = s["data"]
    print(f"raw χ²/N {chi2_red(g['winner'].model, quoted):.4g} (quoted errors, no nuisances)", flush=True)
    scales = error_scale(g["winner"].model, quoted, by_observable=True)
    print(f"block error scales {scales}", flush=True)
    s.update(quoted=quoted, error_scales=scales, data=rescale(quoted, scales))
    s["calibration"] = calibration_report(g["winner"].model, quoted, scales)
    print(f"calibration (raw V²/CP χ²/N first; the rescaled χ²/N is tautological, not a success criterion): "
          f"{s['calibration']}", flush=True)
    return scales


def star_evidence(s, g, smoke=False):
    """The evidence test (C2g): the star arm's GP winner ``g`` against the
    same setup's GP fit with no star, both by ``log_evidence_full`` on the
    quoted errors. Returns the record and the no-star setup and fit."""
    s0 = s["no_star_setup"]()
    s0["on_fit"] = s.get("on_fit")
    if s.get("points"):  # the same point sources, refitted with the image
        s0 = point_scene(s0, s["points"], s["point_priors"])
    s0["systematics"] = s.get("systematics")
    if s.get("error_scales"):
        s0.update(quoted=s0["data"], error_scales=s["error_scales"], data=rescale(s0["data"], s["error_scales"]))
    g0 = gp_fit(s0, smoke)
    data, data0 = with_systematics(s["data"], s.get("systematics")), with_systematics(s0["data"], s0.get("systematics"))
    log_z_star, occam_star = log_evidence_full(g["winner"], data, g["extra"])
    log_z_none, occam_none = log_evidence_full(g0["winner"], data0, g0["extra"])
    log_bayes = log_z_star - log_z_none
    record = {"log_z_star": log_z_star, "occam_star": occam_star, "log_z_nostar": log_z_none,
              "occam_nostar": occam_none, "log_bayes": log_bayes, "kept": bool(log_bayes >= STAR_LOG_BAYES),
              "chi2_red_star": g["chi2_winner"] / g["n_indep"], "chi2_red_nostar": g0["chi2_winner"] / g0["n_indep"]}
    print(f"evidence test: χ²/N {record['chi2_red_star']:.4g} with the star, {record['chi2_red_nostar']:.4g} without; "
          f"log Z {log_z_star:.2f} (Occam {occam_star:.2f}) against {log_z_none:.2f} (Occam {occam_none:.2f}): "
          f"ln B = {log_bayes:.2f}, star {'kept' if record['kept'] else 'rejected (ln B < %.2f)' % STAR_LOG_BAYES}",
          flush=True)
    return record, s0, g0


def gp_checkpoint(s, out_dir, label, t0):
    """A callback for gp_fit's ``s["on_fit"]``: after every fitted (ℓ, σ) it
    rewrites ``<label>.npz`` (``partial=True``) with the best-evidence image so
    far and the log Z of every fit so far, and ``<label>.txt`` with a note. The
    finished run overwrites both. GP runs take hours at large fields and a
    timeout used to leave nothing (round 2, 2016 Obj2 and 2024 MATISSE)."""
    best, fits = {}, {}

    def on_fit(name, length, sigma, log_z, result):
        fits[f"{name}/{length:.4g}/{sigma:g}"] = log_z
        if np.isfinite(log_z) and (name not in best or log_z > best[name][0]):
            best[name] = (log_z, length, sigma, np.asarray(image_of(result.model).render(s["npix"], s["fov"])))
        if not best:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"{label}.npz", partial=True, log_z_fits=json.dumps(fits),
                            **{f"image_{k}": v[3] for k, v in best.items()},
                            best_so_far=json.dumps({k: v[:3] for k, v in best.items()}),
                            pixel_scale_mas=s["pixel"], npix=s["npix"], informed=True)
        (out_dir / f"{label}.txt").write_text(f"PARTIAL: {len(fits)} GP fits at {time.time() - t0:.0f}s; "
                                              f"best so far (log Z, ℓ mas, σ)={ {k: v[:3] for k, v in best.items()} }\n")

    return on_fit


def run_gp(task, data_dir, out_dir, smoke=False, halo=False, star=None, init="moments", member=None,
           settings=None, label=None, field=None, informed=False, points=0, systematics="none", scene=None):
    """GP fits chosen by evidence. With ``member`` (or ``settings``, a fixed
    configuration as used by scripts/contest_bench.py) a CLEAN-started
    ensemble member; otherwise a grown-field run. ``label`` names the
    outputs. With a star, the winner is compared with the same fit without
    it (``star_evidence``); unless the star wins decisively the no-star fit
    is kept, with ``star_rejected="evidence"``. ``field`` (mas) fixes the
    field of a grown-field run; ``informed`` marks a round-2 fit. Round 2:
    ``points`` adds up to that many point sources (add_point_sources; a
    CLEAN start), ``systematics`` is one of SYSTEMATICS: block error scales
    first (estimate_block_scales), then gains / closure offsets with their
    widths chosen by the evidence scan (systematics_scan)."""
    t0 = time.time()
    if member is None and settings is None:
        s, growth = grown_setup(task, data_dir, halo, star, "clean" if points else init, smoke, field, **(scene or {}))
        label = label or s["label"] + "_gp" + suffix(points, systematics)
    else:
        s, settings = member_setup(task, data_dir, member, smoke, settings)
        growth, init = [], "clean"
        label = label or f"{spec_of(task)['label']}_m{member}"
    if points:
        s = add_point_sources(s, points)
    s["on_fit"] = gp_checkpoint(s, out_dir, label, t0)
    if "block_scales" in SYSTEMATICS[systematics]:
        estimate_block_scales(s, smoke)
    # In smoke runs the scan is cut to its median width.
    widths = systematic_widths(systematics, scan=not smoke)
    g, scan = systematics_scan(s, smoke, widths.get("gains"), widths.get("closure"))
    evidence, star_arm = {}, {}
    if s["star"]:
        evidence, s0, g0 = star_evidence(s, g, smoke)
        star_arm = star_record(g["final"].model)  # the star as fitted, kept even when rejected
        if not evidence["kept"]:
            print(f"{label}: star rejected (evidence); imaged without it", flush=True)
            s0.update(star_rejected="evidence", star_fit=s["star_fit"], ref_fov=s.get("ref_fov"))
            s, g = s0, g0
    # A member's halo comes from its settings (setup's flag), not this
    # function's argument.
    halo = s["halo"]
    data, star, res = s.get("quoted", s["data"]), s["star"], s["resolution"]
    n, pixel = s["npix"], s["pixel"]
    fov = n * pixel
    template, ell, others, log_z, axes, best_of = (g[k] for k in ("template", "ell", "others", "log_z", "axes", "best_of"))
    best_name, best_length, best_sigma, winner = g["best_name"], g["best_length"], g["best_sigma"], g["winner"]
    best_log_z, scale, n_indep, chi2_winner, gamma = (g[k] for k in ("best_log_z", "scale", "n_indep", "chi2_winner", "gamma"))
    rescaled, data_s, final, best_result = g["rescaled"], g["data_s"], g["final"], g["best_result"]

    out_dir.mkdir(parents=True, exist_ok=True)
    images = {name: np.asarray(best_result(name).model.env.render(n, fov)) for name in log_z}
    member_extra = {}
    if settings is not None:
        ref_fov = s["ref_fov"]
        n_ref = reference_npix(ref_fov, data_q_max(data))
        companion = s["star_fit"].get("companion") if isinstance(s.get("star_fit"), dict) else None
        member_extra = dict(
            # The whole scene (star and companion included) rendered on the
            # common grid (render_scene, not a resampling of the fit grid), so
            # that star and no-star members compare like with like; the
            # environment alone too.
            ref_image=render_scene(final.model, n_ref, ref_fov, data_q_max(data)),
            ref_env=render_scene(final.model.env, n_ref, ref_fov, data_q_max(data)), ref_fov=ref_fov,
            star=bool(star), star_rejected=s["star_rejected"], star_model=s["star_model"], halo=settings["halo"],
            sparco=settings["sparco"], field_factor=settings["field"], oversample=settings["oversample"],
            clean_gain=settings["clean_gain"], env_flux=flux_value(final.model.env.flux),
            env_index=float(np.ravel(getattr(final.model.env.flux, "index", np.nan))[0]),
            # The fitted star (NaN without one) and, for a rejected star, the
            # star arm's own fit (arm_*), with the evidence test's record.
            **star_record(final.model), **{f"arm_{k}": v for k, v in star_arm.items()},
            evidence=json.dumps(evidence), companion=json.dumps(companion, default=float),
            n_independent=n_indep, gamma=gamma,
            best_log_z=best_log_z,
            best_chi2_red=float(np.sum(winner.info["chi2_red"])),
            error_scale=scale,
            rescaled_log_z=rescaled["log_z"] if rescaled else np.nan,
            rescaled_chi2_red=rescaled["chi2_red"] if rescaled else np.nan,
            flip_dchi2=float(np.ravel(diagnose(final.model, data_s, list(others)).checks["flip_dchi2"])[0]),
        )
    np.savez_compressed(out_dir / f"{label}.npz", **member_extra, **{f"log_z_{k}": v for k, v in log_z.items()},
                        **{f"image_{k}": v for k, v in images.items()},
                        **{f"sigmas_{k}": np.asarray(a[1]) for k, a in axes.items()},
                        **{f"lengths_mas_{k}": np.asarray(a[0]) for k, a in axes.items()},
                        ellipse=np.asarray(ell), pixel_scale_mas=pixel, npix=n,
                        template=template, informed=bool(informed), partial=False,
                        systematics=json.dumps(s.get("systematics")), systematics_scan=json.dumps(scan, default=float),
                        systematics_mode=systematics, error_scales=json.dumps(s.get("error_scales")),
                        calibration=json.dumps(s.get("calibration")),
                        point_sources=json.dumps(s.get("point_record"), default=float),
                        raw_chi2_red=chi2_red(winner.model, data))
    lines = [f"task={task} label={label} star={star} star_rejected={s['star_rejected'] or None} halo={halo} init={init} "
             f"field_source={s['field_source']} growth={growth}", informed_line(informed),
             f"raw chi2/N={chi2_red(winner.model, data):.4g} (quoted errors, no nuisances) mode={systematics} "
             f"widths={s.get('systematics')} error_scales={s.get('error_scales')} calibration={s.get('calibration')} "
             f"scan={scan} point_sources={s.get('point_record')}",
             f"points={s['npts']} npix={n} pixel={pixel:.4g} mas fov={fov:.4g} mas beam={res.major_mas:.3g}x{res.minor_mas:.3g} mas",
             f"ellipse fit: chi2/N={ell[0]:.4g} ratio={ell[1]:.3f} pa={ell[2]:.1f} deg fwhm={ell[3]:.3g} mas",
             f"star_model={s['star_model']} sparco={s['sparco']} wavel0={s['wavel0']:.4g}"]
    if evidence:
        lines.append(f"evidence test (star arm against no star): {evidence}; star arm's star: {star_arm}")
    for name, g_ in log_z.items():
        ls, ss = axes[name]
        lines.append(f"{name}: sigmas={list(ss)} lengths_mas={[round(x, 4) for x in ls]}")
        lines.append(f"log_z[{name}] (rows ℓ, cols σ):\n{np.array2string(g_, precision=2)}")
        i, j = best_of[name]
        rr = best_result(name)
        lines.append(f"best {name}: ℓ={ls[i]:.4g} σ={ss[j]} log_z={g_[i, j]:.2f} chi2/N={float(np.sum(rr.info['chi2_red'])):.3f} converged={rr.info.get('converged')}")
    lines.append(f"residuals of the winner={residual_diagnostics(winner.model, data)}")
    lines.append(f"N={n_indep} gamma={gamma:.1f} N-gamma={n_indep - gamma:.1f}")
    lines.append(f"chi2/N={chi2_winner / n_indep:.4g} ({'RESCALED errors: tautological' if s.get('error_scales') else 'quoted errors'}), error_scale={scale:.3f}"
                 + (f" -> refit with rescaled errors: log_z={rescaled['log_z']:.2f} chi2/N={rescaled['chi2_red']:.3f}" if rescaled else " (within 1.3x: kept)"))
    if settings is not None:
        lines.append(f"member settings={settings}")
    lines.append(f"WINNER: chi2/N={chi2_winner / n_indep:.4g} error_scale={scale:.3f} {best_name} ℓ={best_length:.4g} "
                 f"σ={best_sigma} log_z={best_log_z:.2f}")
    lines.append(f"elapsed={time.time() - t0:.0f}s\n\n{diagnose(winner.model, data, list(others))}")
    text = "\n".join(lines) + "\n"
    (out_dir / f"{label}.txt").write_text(text)
    print(text)

    fig, panels = plt.subplots(1, 1 + len(images) + len(log_z), figsize=(4.6 * (1 + len(images) + len(log_z)), 4.4))
    plot_model(vm.Image(jnp.asarray(np.log(template + 1e-12)), pixel), fov_mas=fov, npix=n, ax=panels[0], beam=res,
               title=f"template ({init})")
    for ax, (name, im) in zip(panels[1:], images.items()):
        i, j = best_of[name]
        plot_model(best_result(name).model.env, fov_mas=fov, npix=n, ax=ax, beam=res,
                   title=f"GP {name}: ℓ={axes[name][0][i]:.2g} mas, σ={axes[name][1][j]:g}, logZ={log_z[name][i, j]:.0f}")
    for ax, (name, g_) in zip(panels[1 + len(images):], log_z.items()):
        ls, ss = axes[name]
        ax.imshow(g_ - np.nanmax(g_), origin="lower", cmap="viridis", aspect="auto")
        ax.set(xticks=range(len(ss)), xticklabels=ss, yticks=range(len(ls)),
               yticklabels=[f"{x:.2g}" for x in ls], xlabel="σ", ylabel="ℓ (mas)", title=f"log Z − max ({name})")
    plt.tight_layout()
    fig.savefig(out_dir / f"{label}.png", dpi=100)


# Two-scale GP (round 2, two-scale): a compact and an extended Gaussian field
# on the same grid, summed as two Images in one System (virgil has no sum of
# fields). Each ℓ (in beam minor axes, capped at the field) is chosen by the
# evidence with both fields' latents integrated out (log_evidence_two).
TWO_SCALE_COMPACT = (0.5, 1.0)
TWO_SCALE_EXTENDED = (2.0, 4.0, 8.0)
TWO_SCALE_SIGMAS = (1.0, 2.0)


def log_evidence_two(result, data, paths=("env", "ext")):
    """log_evidence with the latents of several Gaussian fields integrated
    out together: -½χ² - log_norm - ½Σ|z|² - ½ Σ log(1 + s²) over the
    singular values s of the joint latent Jacobian (virgil's log_evidence,
    one field, generalised)."""
    from virgil.imaging import _log_norm, _residual_jacobian  # private in virgil: as log_evidence

    model = getattr(result, "model", result)
    jacs, z2 = [], 0.0
    for p in paths:
        r, jac = _residual_jacobian(model, data, f"{p}.log_brightness.latent")
        jacs.append(jac.reshape(jac.shape[0], -1))
        z2 += float(np.sum(np.square(np.asarray(model.get(f"{p}.log_brightness.latent"), float))))
    singular = np.linalg.svd(np.concatenate(jacs, axis=1), compute_uv=False)
    return float(-0.5 * float(r @ r) - float(_log_norm(model, data)) - 0.5 * z2 - 0.5 * np.sum(np.log1p(singular**2)))


def two_scale_image(env, ext, f_env, f_ext):
    """The sum of the compact and extended fields as one unit-sum image.
    ``Image.render`` is unit-sum whatever the flux, so the components are
    weighted by their fitted fluxes (``image_env`` and ``image_ext`` stay
    unit-sum each). Summing them unweighted gave an image of total 2 whose
    extended part counted as much as the compact one (round 2)."""
    total = f_env + f_ext
    return (f_env * np.asarray(env) + f_ext * np.asarray(ext)) / total


def run_gp2(task, data_dir, out_dir, smoke=False, halo=False, star=None, init="moments", label=None, field=None,
            informed=False, scene=None):
    """Two-scale GP: compact (``env``, mean the starting image) plus extended
    (``ext``, flat mean, its flux relative to the compact one log-uniform)
    fields, ℓ for each and a common σ chosen by log_evidence_two."""
    import itertools

    t0 = time.time()
    s, growth = grown_setup(task, data_dir, halo, star, init, smoke, field, **(scene or {}))
    label = label or s["label"] + "_gp2"
    data, star, res, n, pixel = s["data"], s["star"], s["resolution"], s["npix"], s["pixel"]
    fov = n * pixel
    img0 = s["img0"]
    template = np.asarray(img0.brightness).reshape(n, n)
    support = getattr(img0, "support", None)
    cap = lambda fs: sorted({min(f * res.minor_mas, fov) for f in fs})  # noqa: E731
    compact = cap(TWO_SCALE_COMPACT[1:] if smoke else TWO_SCALE_COMPACT)
    extended = cap(TWO_SCALE_EXTENDED[1:2] if smoke else TWO_SCALE_EXTENDED)
    sigmas = TWO_SCALE_SIGMAS[1:] if smoke else TWO_SCALE_SIGMAS
    extra = (flux_priors(FLUX_CAP, s["sparco"]) | s["star_priors"]) if star else {}
    extra = extra | {"ext.flux": dist.LogUniform(FLUX_FLOOR, FLUX_CAP)} | (
        {"halo.flux": dist.LogUniform(FLUX_FLOOR, 1000.0)} if halo else {})
    others = () if star else (Centroid(0.1 * res.minor_mas, path="env"),)
    options = {"max_steps": 50} if smoke else {}
    grid, previous = {}, None
    for l1, l2, sig in itertools.product(compact, extended, sigmas):
        if l2 <= l1:
            continue
        env = vm.Image(GaussianField(np.zeros((n, n)), sig, l1, mean=template), pixel, support=support, flux=img0.flux)
        ext = vm.Image(GaussianField(np.zeros((n, n)), sig, l2), pixel, support=support, flux=0.3)
        parts = ({"star": s["start"].star} if star else {}) | {"env": env, "ext": ext} | (
            {"halo": vm.Resolved(1.0)} if halo else {})
        sc = vm.System(**parts)
        r_ = fit(sc, image_priors(sc) | extra, data, others,
                 init=None if previous is None else previous.values, **options)
        previous = r_
        grid[l1, l2, sig] = (log_evidence_two(r_, data), r_)
        if np.isfinite(grid[l1, l2, sig][0]):  # checkpoint: the best fit so far, partial=True (the final npz overwrites it)
            k_ = max((k for k in grid if np.isfinite(grid[k][0])), key=lambda k: grid[k][0])
            m_ = grid[k_][1].model
            e_, x_ = (np.asarray(getattr(m_, c).render(n, fov)) for c in ("env", "ext"))
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_dir / f"{label}.npz", partial=True, image_env=e_, image_ext=x_,
                                image=two_scale_image(e_, x_, flux_value(m_.env.flux), flux_value(m_.ext.flux)),
                                pixel_scale_mas=pixel, npix=n, best=np.array(k_), informed=bool(informed),
                                log_z=json.dumps({f"{a:.4g}/{b:.4g}/{c:g}": v for (a, b, c), (v, _) in grid.items()}))
            (out_dir / f"{label}.txt").write_text(f"PARTIAL: {len(grid)} two-scale fits at {time.time() - t0:.0f}s\n")
        print(f"two-scale ℓ={l1:.3g}/{l2:.3g} mas σ={sig:g}: log Z {grid[l1, l2, sig][0]:.2f} "
              f"χ²/N {chi2_red(r_.model, data):.4g}", flush=True)
    key = max(grid, key=lambda k: grid[k][0])
    log_z, best = grid[key]
    m = best.model
    raw = chi2_red(m, data)
    images = {k: np.asarray(getattr(m, k).render(n, fov)) for k in ("env", "ext")}
    f_env, f_ext = flux_value(m.env.flux), flux_value(m.ext.flux)
    combined = two_scale_image(images["env"], images["ext"], f_env, f_ext)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{label}.npz", image_env=images["env"], image_ext=images["ext"],
                        image=combined, ext_fraction=f_ext / (f_env + f_ext), pixel_scale_mas=pixel, npix=n, raw_chi2_red=raw,
                        log_z=json.dumps({f"{a:.4g}/{b:.4g}/{c:g}": v for (a, b, c), (v, _) in grid.items()}),
                        best=np.array(key), ext_flux=flux_value(m.ext.flux), informed=bool(informed), partial=False)
    text = (f"task={task} label={label} star={star} halo={halo} init={init} field_source={s['field_source']} "
            f"growth={growth}\n{informed_line(informed)}\n"
            f"points={s['npts']} npix={n} pixel={pixel:.4g} mas fov={fov:.4g} mas "
            f"beam={res.major_mas:.3g}x{res.minor_mas:.3g} mas\n"
            f"grid (ℓ compact / ℓ extended / σ: log Z)={ {k: round(v[0], 2) for k, v in grid.items()} }\n"
            f"WINNER: chi2/N={raw:.4g} (quoted errors) ℓ={key[0]:.4g}/{key[1]:.4g} mas σ={key[2]:g} log_z={log_z:.2f} "
            f"ext/env flux={flux_value(m.ext.flux) / max(flux_value(m.env.flux), 1e-300):.3g}\n"
            f"residuals={residual_diagnostics(m, data)}\nelapsed={time.time() - t0:.0f}s\n")
    (out_dir / f"{label}.txt").write_text(text)
    print(text)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (name, model_) in zip(axes, (("compact", m.env), ("extended", m.ext))):
        plot_model(model_, fov_mas=fov, npix=n, ax=ax, beam=res, title=f"{label}: {name}")
    axes[2].imshow(combined, origin="lower", cmap="inferno")
    axes[2].set_title(f"flux-weighted sum, χ²/N={raw:.3g}")
    plt.tight_layout()
    fig.savefig(out_dir / f"{label}.png", dpi=100)


# Artefact pruning (round 2, artefact-prune): isolated compact blobs outside
# the main source (connected regions above PRUNE_LEVEL of the peak, smaller
# than a beam's area, not touching the main source) are set to the image's
# floor and the fit is redone at the same weight, up to PRUNE_ROUNDS times.
PRUNE_LEVEL = 0.02
PRUNE_ROUNDS = 5


def isolated_blobs(image, pixel, resolution):
    """Masks of compact regions of ``image`` above PRUNE_LEVEL of its peak
    outside the main source (see MAIN_LEVEL)."""
    from scipy import ndimage

    im = np.asarray(image, float)
    labels, _ = ndimage.label(im > MAIN_LEVEL * im.max())
    main = labels == labels[np.unravel_index(np.argmax(im), im.shape)]
    main = ndimage.binary_dilation(main, iterations=max(1, int(round(0.5 * resolution.minor_mas / pixel))))
    blobs, count = ndimage.label(im > PRUNE_LEVEL * im.max())
    beam_area = np.pi / 4 * resolution.major_mas * resolution.minor_mas / pixel**2
    out = []
    for i in range(1, count + 1):
        b = blobs == i
        if not np.any(b & main) and b.sum() <= beam_area:
            out.append(b)
    return out


def prune(result, priors, data, reg_, others, path, n, fov, resolution, options):
    """Remove isolated blobs from ``result``'s image and refit, repeatedly.
    Returns the final fit and a record per round."""
    record = []
    for _ in range(PRUNE_ROUNDS):
        img = image_of(result.model)
        blobs = isolated_blobs(np.asarray(img.render(n, fov)), fov / n, resolution)
        if not blobs:
            break
        lb = np.asarray(img.log_brightness).copy()
        floor = float(lb.max()) - np.log(1e4)
        for b in blobs:
            lb[b] = floor
        model = result.model
        new = eqx.tree_at(lambda m: m.log_brightness, img, jnp.asarray(lb))
        model = vm.System(**(model.components | {"env": new})) if path else new
        before = chi2_red(result.model, data)
        result = fit(model, priors, data, [reg_, *others], **options)
        record.append({"removed": len(blobs), "chi2_before": before, "chi2_after": chi2_red(result.model, data)})
        print(f"prune: removed {len(blobs)} blob(s), χ²/N {before:.4g} -> {record[-1]['chi2_after']:.4g}", flush=True)
    return result, record


def clean_label(s, clean_scales):
    """run_clean's output label: setup's (task, _star/_nostar, _halo suffixes), without
    the ``_clean`` init suffix, then _cleanfinal (and _cleanms for multi-scale CLEAN)."""
    multiscale = tuple(float(f) for f in clean_scales) != (0.0,)
    base = s["label"]
    base = base[: -len("_clean")] if base.endswith("_clean") else base
    return base + "_cleanfinal" + ("_cleanms" if multiscale else "")


def run_clean(task, data_dir, out_dir, smoke=False, halo=False, star=None, label=None, field=None,
              informed=False, clean_scales=(0.0,), scene=None):
    """CLEAN's own result (round 2, clean-final): the restored image and the
    components of setup's CLEAN run, with χ²/N of the components' model on
    the quoted errors. CLEAN stops at χ²/N = 1, so no weight is chosen."""
    t0 = time.time()
    s, growth = grown_setup(task, data_dir, halo, star, "clean", smoke, field, clean_scales, **(scene or {}))
    cleaned, data, res = s["cleaned"], s["data"], s["resolution"]
    label = label or clean_label(s, clean_scales)
    restored = np.clip(np.asarray(cleaned.restored(s["restore"])), 0.0, None)
    comps = np.asarray(cleaned.components)
    out_dir.mkdir(parents=True, exist_ok=True)
    chi2 = chi2_red(cleaned.model, data)
    np.savez_compressed(out_dir / f"{label}.npz", restored=restored, components=comps,
                        pixel_scale_mas=float(cleaned.pixel_scale_mas), npix=restored.shape[0],
                        beam=np.array([res.major_mas, res.minor_mas, res.pa_deg]), chi2_red=chi2,
                        stop=str(cleaned.stop), informed=bool(informed),
                        components_by_scale=np.asarray(cleaned.components_by_scale),
                        scales_mas=np.asarray(cleaned.scales_mas), clean_scales=np.asarray(clean_scales, float))
    text = (f"task={task} label={label} files={s['files']} star={s['star']} halo={halo}\n{informed_line(informed)}\n"
            f"field_source={s['field_source']} growth(fov, chi2/N)={growth}\n"
            f"virgil={virgil.__version__} from {virgil.__file__}\n"
            f"points={s['npts']} npix={restored.shape[0]} pixel={float(cleaned.pixel_scale_mas):.4g} mas "
            f"beam={res.major_mas:.3g}x{res.minor_mas:.3g} mas components={int(np.count_nonzero(comps))}\n"
            f"clean_scales(beams)={tuple(clean_scales)} scales_mas={tuple(round(float(x), 4) for x in cleaned.scales_mas)}\n"
            f"CLEAN stop={cleaned.stop} CLEAN's chi2/N={float(np.ravel(cleaned.chi2_red)[-1]):.4g}\nchi2/N={chi2:.4g} (quoted errors; the components' model)\n"
            f"residuals={residual_diagnostics(cleaned.model, data)}\nelapsed={time.time() - t0:.0f}s\n")
    (out_dir / f"{label}.txt").write_text(text)
    print(text)
    fov = restored.shape[0] * float(cleaned.pixel_scale_mas)
    fig, ax = plt.subplots(figsize=(5, 4.6))
    plot_model(vm.Image(jnp.asarray(np.log(restored / max(float(restored.max()), 1e-300) + 1e-6)), float(cleaned.pixel_scale_mas)),
               fov_mas=fov, npix=restored.shape[0], ax=ax, beam=res, title=f"{label}: CLEAN restored, χ²/N={chi2:.3g}")
    fig.savefig(out_dir / f"{label}.png", dpi=100)


# Ensemble sweep (round 2, --sweep): virgil.ensemble's randomised L-curves
# (regulariser family, pixel scale, field, start), the kept members' mean and
# spread. The spec is virgil's EnsembleSpec() default, apart from --sweep-families
# (and the smoke cut). Its weight ranges are virgil's untuned defaults (fitted
# to the contest bench's corners, not to this data), and the 180° ambiguity of
# closure-phase data (an image and its point reflection fit alike) is not handled:
# members are not flipped onto each other, so the mean can blur a flipped member.
SWEEP_SMOKE = dict(n_weights=3, max_npix=32)


def nearest_discrepancy(members):
    """The member whose total χ²/N is nearest 1 (in log) among those whose fit
    is finite, kept or not, and how many kept members fit better than χ²/N =
    0.5. virgil.ensemble keeps the weights up to the L-curve's corner, whose
    window and "within 2x the best member" cut favour the lowest χ²: with
    errors that are too large (2008 AGB K, MATISSE N: winners at χ²/N 0.27 and
    0.12) that is an over-fit. The discrepancy principle asks for χ²/N ≈ 1; this
    reports the member that does, beside virgil's mean (which is not
    changed). Returns (member or None, n_overfit)."""
    finite = [m for m in members if np.isfinite(m.total_chi2_red) and m.total_chi2_red > 0]
    best = min(finite, key=lambda m: abs(np.log(m.total_chi2_red)), default=None)
    return best, sum(1 for m in finite if m.kept and m.total_chi2_red < 0.5)


def run_sweep(task, data_dir, out_dir, n_groups, smoke=False, star=None, families=None, label=None, informed=False):
    """``n_groups`` groups of virgil.ensemble.ensemble on the task's data,
    with the task's star unless ``star`` says otherwise. Saves the mean
    image, its pixel-wise std, the kept members, and the ensemble's summary."""
    from virgil.ensemble import EnsembleSpec, ensemble

    t0 = time.time()
    spec = spec_of(task)
    star = spec.get("star", False) if star is None else star
    data = load_data(spec, data_dir)
    options = dict(SWEEP_SMOKE) if smoke else {}
    if families:
        options["families"] = tuple(families)
    ens_spec = EnsembleSpec(**options)
    ens = ensemble(data, n_groups, jax.random.PRNGKey(0), spec=ens_spec, star=star, **({"max_steps": 50} if smoke else {}))
    label = label or spec["label"] + ("_star" if star and not spec.get("star") else "") + "_sweep"
    kept = [{"group": m.draw.index, "family": m.draw.family, "weight": m.weight, "npix": m.draw.npix,
             "pixel_scale_mas": m.draw.pixel_scale_mas, "start": m.draw.start, "chi2_red": list(m.chi2_red)}
            for m in ens.kept]
    mean, std = np.asarray(ens.mean.brightness), np.asarray(ens.std)
    pixel = float(ens.mean.pixel_scale_mas)
    raw = chi2_red(ens.model, data)
    summary = ens.summary()
    near, n_overfit = nearest_discrepancy(ens.members)
    near_record, near_image = None, np.zeros((0, 0))
    if near is not None:
        near_record = {"group": near.draw.index, "family": near.draw.family, "weight": near.weight, "kept": near.kept,
                       "reason": near.reason, "chi2_red": list(near.chi2_red), "total_chi2_red": near.total_chi2_red,
                       "npix": near.draw.npix, "pixel_scale_mas": near.draw.pixel_scale_mas}
        near_image = safe_render(image_of(near.result.model), near.draw.npix, near.draw.npix * near.draw.pixel_scale_mas)
    if len(data if isinstance(data, list) else [0]) > 1:
        summary += ("\nnote: the datasets are fitted jointly with one grey image (no SPARCO in virgil.ensemble); "
                    "per-dataset χ²/N above is not comparable with a single-band fit")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{label}.npz", mean=mean, std=std, pixel_scale_mas=pixel, npix=mean.shape[0],
                        mean_flux=flux_value(ens.mean.flux), kept=json.dumps(kept, default=float), summary=summary,
                        ensemble_chi2_red=np.asarray(ens.chi2_red), raw_chi2_red=raw, n_groups=n_groups, star=bool(star),
                        spec=json.dumps(dataclasses.asdict(ens_spec), default=float), informed=bool(informed),
                        nearest_discrepancy=json.dumps(near_record, default=float), nearest_discrepancy_image=near_image,
                        n_kept_below_half=n_overfit)
    text = (f"task={task if isinstance(task, int) else spec['label']} label={label} files={spec['files']} star={star}\n"
            f"{informed_line(informed)}\nraw chi2/N={raw:.4g} (quoted errors, no nuisances; the mean scene) "
            f"per dataset={tuple(round(float(c), 4) for c in ens.chi2_red)}\n"
            f"sweep: {n_groups} groups, EnsembleSpec {dataclasses.asdict(ens_spec)}\n"
            f"the weight ranges are virgil's untuned defaults; the 180° ambiguity is not handled\n"
            f"virgil={virgil.__version__} from {virgil.__file__}\nnpix={mean.shape[0]} pixel={pixel:.4g} mas "
            f"kept={len(kept)} members\n{summary}\n"
            f"member nearest χ²/N = 1 (the discrepancy principle on the quoted errors; virgil's mean keeps the lowest-χ² "
            f"members, {n_overfit} kept below χ²/N 0.5, an over-fit when errors are too large): {near_record}\nelapsed={time.time() - t0:.0f}s\n")
    (out_dir / f"{label}.txt").write_text(text)
    print(text)
    fov = mean.shape[0] * pixel
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    plot_model(ens.mean, fov_mas=fov, npix=mean.shape[0], ax=axes[0], title=f"{label}: mean of {len(kept)}")
    im = axes[1].imshow(std, origin="lower", cmap="magma")
    axes[1].set_title("std across kept members")
    fig.colorbar(im, ax=axes[1])
    plt.tight_layout()
    fig.savefig(out_dir / f"{label}.png", dpi=100)


def build_parser():
    """The command line's parser (scripts/contest_fit_plan.py's tests parse the round-2 lines with it)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=int)
    parser.add_argument("--joint", type=lambda x: x.split(","),
                        help="TASKS (indices or labels, comma-separated) merged into one dataset, e.g. the bands of one "
                             "target; the first fixes field, prior and star (--task, if given, must be that first one)")
    parser.add_argument("--sparco", action="store_true",
                        help="with a star (--scene or the task's): the image has its own power-law spectrum relative to "
                             "the star's (a CLEAN start; labels get _sparco)")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--data", default="~/data/imaging_contests")
    parser.add_argument("--out", default="contest_images")
    parser.add_argument("--halo", action="store_true", help="add a fully resolved component (labels get _halo)")
    parser.add_argument("--star", choices=("on", "off"), help="override the task's central star (labels get _star/_nostar)")
    parser.add_argument("--init", choices=("moments", "clean"), default="moments", help="starting image (labels get _clean)")
    parser.add_argument("--prior", choices=("mem", "gp", "gp2", "clean"), default="mem",
                        help="regularised L-curve, Gaussian-process fits chosen by evidence (labels get _gp), "
                             "a compact plus an extended GP (labels get _gp2), "
                             "or CLEAN's own restored image and components (labels get _cleanfinal)")
    parser.add_argument("--prune", action="store_true",
                        help="--prior mem: remove isolated compact blobs outside the main source and refit (_prune)")
    parser.add_argument("--regulariser", choices=tuple(REGULARISERS), default="mem",
                        help="--prior mem's penalty, weight by the discrepancy principle (labels get _<name> unless mem)")
    parser.add_argument("--scene", help="a composite scene, NAME=KIND[...] + ... (scripts/contest_composite.py), or a "
                        "comma list of parts of the scene (point, ud, ld_ellipse, gauss_halo, image; "
                        "image is required): a central source, a Gaussian halo with fitted FWHM and flux, and the pixel image "
                        "(labels get _<parts>); in place of --star")
    parser.add_argument("--max-pix", type=int, help="cap the image's pixels on a side (default %d; coarser pixels over the same field). "
                        "The GP evidence costs data points x pixels², so heavy lines (400 mas fields, MATISSE) take this" % MAX_PIX)
    parser.add_argument("--hole", type=float, help="radius (mas) of the hole in the image's support (default with a star: "
                        "half the beam's minor axis); the image's flux is then fitted relative to the star's")
    parser.add_argument("--field", type=float, help="field of view (mas), only where pre-submission information gives one")
    parser.add_argument("--points", type=int, default=0,
                        help="up to K point sources from CLEAN's compact clusters outside the main source, each kept "
                             "if Δχ² >= 25 (a CLEAN start; labels get _psK)")
    parser.add_argument("--systematics", choices=tuple(SYSTEMATICS), default="none",
                        help="calibration nuisances and error scales: per-observable error scales (block_scales, "
                             "--prior gp), gains per baseline, closure offsets per triangle; widths by evidence with "
                             "--prior gp, the scan's median otherwise (labels get _bscale, _gain_bscale, _gain_cloff)")
    parser.add_argument("--clean-scales", default="0",
                        help="CLEAN component widths in beam minor axes, comma-separated (multi-scale CLEAN, e.g. 0,1,2; "
                             "labels get _cleanms); only with --prior clean")
    parser.add_argument("--sweep", type=int, metavar="N",
                        help="N groups of virgil.ensemble (randomised L-curves, mean and std of the kept members; labels "
                             "get _sweep); virgil's default EnsembleSpec, untuned weight ranges, 180° ambiguity not handled")
    parser.add_argument("--sweep-families", type=lambda x: x.split(","),
                        help="--sweep: regulariser families (tv, tsv, maxent, starlet; default all four)")
    parser.add_argument("--informed", action="store_true",
                        help="mark the outputs as round-2 fits whose method was chosen after seeing the truth")
    parser.add_argument("--member", type=int, help="ensemble member (CLEAN-started GP with a randomised start)")
    parser.add_argument("--settings", help="a fixed ensemble configuration (JSON, keys as member_settings), e.g. from contests/fit_plan.yml")
    parser.add_argument("--label", help="output label (default: the task's, with suffixes)")
    parser.add_argument("--smoke", action="store_true", help="three weights, 50 steps: check that it runs")
    return parser


def validate(parser, args):
    """Check the combination of options (``parser.error`` on a bad one) and
    resolve the scene: (task, star, scene), ``args.init`` set for --sparco.
    No data are read."""
    star = None if args.star is None else args.star == "on"
    task = args.task
    if args.joint:
        try:
            task = joint_spec(args.joint)
            first = task_index(args.joint[0])
        except (ValueError, IndexError) as e:
            parser.error(f"--joint: {e}")
        if args.task is not None and args.task != first:
            parser.error("--task must be the first of --joint's tasks (or omitted)")
    elif task is None:
        parser.error("give --task or --joint")
    if args.regulariser != "mem" and args.prior != "mem":
        parser.error(f"--regulariser {args.regulariser} has no effect with --prior {args.prior}")
    if args.clean_scales != "0" and args.prior != "clean":
        parser.error("--clean-scales only applies with --prior clean")
    if args.member is not None:
        for flag in ("field", "label", "informed"):
            if getattr(args, flag) not in (None, False):
                parser.error(f"--{flag} has no effect with --member")
    if args.settings is not None and args.field is not None:
        parser.error("--field has no effect with --settings")
    scene = {} if args.hole is None else {"hole_mas": args.hole}
    if args.max_pix is not None:
        if args.max_pix < 9 or args.settings is not None or args.member is not None or args.sweep or (
                args.scene and "=" in args.scene):
            parser.error("--max-pix is at least 9 and needs a grown-field run (not --settings, --member, --sweep or a composite --scene)")
        scene["max_pix"] = args.max_pix
    if args.scene and "=" in args.scene:  # a composite scene (scripts/contest_composite.py)
        import contest_composite

        if (args.star or args.settings is not None or args.member is not None or args.prior != "mem"
                or args.hole is not None or args.sparco or args.points or args.systematics != "none" or args.sweep):
            parser.error("a composite --scene sets its own priors, holes and stars: not with --star, --prior, --hole, "
                         "--sparco, --points, --systematics, --sweep, --settings or --member")
        try:
            return task, None, {"composite": contest_composite.parse_scene(args.scene)}
        except ValueError as e:
            parser.error(str(e))
    if args.scene:
        if args.star or args.settings is not None or args.member is not None:
            parser.error("--scene replaces --star, and --settings / --member fix their own scene")
        try:
            scene = scene_options(args.scene, args.hole) | ({} if args.max_pix is None else {"max_pix": args.max_pix})
        except ValueError as e:
            parser.error(str(e))
        star = scene.pop("star")
        if args.prior in ("gp2", "clean") and scene["gauss_halo"]:
            parser.error("gauss_halo needs --prior gp or mem")
    if args.sparco:
        spec = spec_of(task)
        if not (star if star is not None else spec.get("star", False)):
            parser.error("--sparco needs a star (--scene with point, ud or ld_ellipse, or a task with one)")
        if args.settings is not None or args.member is not None or args.prior in ("gp2", "clean"):
            parser.error("--sparco needs --prior gp or mem in a grown-field run")
        scene["sparco"] = True
        args.init = "clean"
    if args.hole is not None and (args.settings is not None or args.member is not None):
        parser.error("--hole needs a grown-field run (not --settings / --member)")
    if args.systematics != "none" and args.settings is None and (
            args.prior in ("gp2", "clean") or "block_scales" in args.systematics and args.prior != "gp"):
        parser.error("--systematics needs --prior gp or mem (block_scales: gp only)")
    if args.sweep:
        if args.settings is not None or args.member is not None or args.prior != "mem" or args.regulariser != "mem":
            parser.error("--sweep takes its own regularisers (--sweep-families): not --prior, --regulariser, --settings or --member")
        if (args.systematics != "none" or args.points or args.sparco or args.hole is not None or args.prune
                or scene.get("star_model", "point") != "point" or scene.get("gauss_halo")):
            parser.error("--sweep fits virgil's ensemble scene: at most --scene point,image (or --star), "
                         "no --systematics, --points, --sparco, --hole or --prune")
    return task, star, scene


def main():
    # A run that stalls prints where it is: every 30 minutes, every thread's
    # traceback goes to stderr (campaign-2 smoke tasks 75, 77 and 79 hung for
    # 45 min without output, with low CPU and GPU use).
    import faulthandler
    import sys

    faulthandler.dump_traceback_later(1800, repeat=True, file=sys.stderr)
    parser = build_parser()
    args = parser.parse_args()
    if args.list:
        for i, spec in enumerate(TASKS):
            print(f"{i:2d}  {spec['label']:20s} field={spec.get('field')}  prior={spec.get('prior')}  "
                  f"star={spec.get('star', False)}  wavel={spec.get('wavel')}  {len(spec['files'])} file(s)")
        return
    task, star, scene = validate(parser, args)
    if "composite" in scene:
        import contest_composite

        contest_composite.run_composite(task, pathlib.Path(args.data).expanduser(), pathlib.Path(args.out).expanduser(),
                                        scene["composite"], args.smoke, args.label, args.field, args.informed)
        return
    if args.sweep:
        run_sweep(task, pathlib.Path(args.data).expanduser(), pathlib.Path(args.out).expanduser(), args.sweep, args.smoke,
                  star, args.sweep_families, args.label, args.informed)
        return
    if args.settings is not None:
        run_gp(task, pathlib.Path(args.data).expanduser(), pathlib.Path(args.out).expanduser(), args.smoke,
               settings=json.loads(args.settings), label=args.label, informed=args.informed, points=args.points,
               systematics=args.systematics)
        return
    if args.member is not None:
        run_gp(task, pathlib.Path(args.data).expanduser(), pathlib.Path(args.out).expanduser(), args.smoke,
               member=args.member)
        return
    paths = pathlib.Path(args.data).expanduser(), pathlib.Path(args.out).expanduser()
    scales = tuple(float(x) for x in args.clean_scales.split(","))
    common = dict(label=args.label, field=args.field, informed=args.informed)
    if args.prior == "gp":
        run_gp(task, *paths, args.smoke, args.halo, star, args.init, points=args.points,
               systematics=args.systematics, scene=scene, **common)
    elif args.prior == "gp2":
        run_gp2(task, *paths, args.smoke, args.halo, star, args.init, scene=scene, **common)
    elif args.prior == "clean":
        run_clean(task, *paths, args.smoke, args.halo, star, clean_scales=scales, scene=scene, **common)
    else:
        run(task, *paths, args.smoke, args.halo, star, args.init, reg=args.regulariser, points=args.points,
            prune_blobs=args.prune, systematics=args.systematics, scene=scene, **common)


if __name__ == "__main__":
    main()
