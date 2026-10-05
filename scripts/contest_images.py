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
import json
import pathlib
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
    MaxEntropy,
    beam,
    diagnose,
    image_priors,
    l_curve,
    starting_image,
)
from virgil.fitting import fit  # noqa: E402
from virgil.likelihood import whitened_residuals  # noqa: E402
from virgil.oidata import OIData  # noqa: E402
from virgil.plotting import plot_model  # noqa: E402
from virgil.spectra import PowerLaw  # noqa: E402

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
]

WEIGHTS = np.logspace(4.0, 0.0, 13)  # strong to weak
MAX_PIX = 256  # pixels on a side; larger fields get coarser pixels
JACOBIAN_BUDGET = 2.0e8  # data points x pixels, about 3 GB per complex128 Jacobian
CLEAN_MAX_PIX = 65  # CLEAN runs on at most this grid (see setup())
FLUX_FLOOR = 1e-4  # lower bound of the log-uniform (Jeffreys) flux priors, relative to the star
GROWTHS = 3  # data-chosen fields grow at most three times (see run())


def image_of(model, star, halo=False):
    return model.env if (star or halo) else model


def spec_of(task):
    """A task's spec: ``TASKS[task]`` for an index, or a spec dict as given
    (scripts/contest_bench.py passes simulated datasets this way, with
    absolute ``files``)."""
    return task if isinstance(task, dict) else TASKS[task]


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


def setup(task, data_dir, halo=False, grow=1.0, star=None, init="moments", clean_iters=3000,
          oversample=4.0, clean_gain=0.1, star_model="point", sparco=False, mean_blur=1.0):
    """The data, starting model, priors and fixed regularisers for one task
    (shared with scripts/diagnose_stall.py). ``grow`` enlarges a field chosen
    from the data. With a star, ``star_model`` is "point" (unresolved) or
    "disk" (a uniform disk whose diameter is fitted); ``sparco`` gives the
    image a power-law spectrum relative to the star's. ``mean_blur`` scales
    the beam that restores CLEAN's components into the starting image (1:
    the beam; 0.5 keeps detail finer than a beam)."""
    spec = spec_of(task)
    # star=None keeps the task's default; True/False override it, so any
    # dataset can be imaged with and without a central star and compared
    # (labels get _star / _nostar).
    label, files = spec["label"], spec["files"]
    if star is None:
        star = spec.get("star", False)
    else:
        label += "_star" if star else "_nostar"
    field, prior_spec = spec.get("field"), spec.get("prior")
    if halo:
        label += "_halo"
    if init == "clean":
        label += "_clean"
    paths = [str(data_dir / f) for f in files]
    if spec.get("wavel"):
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="contest_"))
        paths = [select_channels(p, spec["wavel"], tmp) for p in paths]
    data = OIData(paths[0]) if len(paths) == 1 else [OIData(p) for p in paths]
    datasets = data if isinstance(data, list) else [data]
    npts = sum(np.asarray(d.flatten_data()[0]).size for d in datasets)
    resolution = beam(data)

    kwargs = {"star": star, "oversample": oversample}
    if star:
        kwargs["hole_mas"] = 0.5 * resolution.minor_mas
    start = starting_image(data, **kwargs)
    star_priors = {}
    if star and star_model == "disk":
        # Diameter: a scale parameter, so log-uniform, from a twentieth of the
        # beam (unresolved) to the beam's major axis (beyond that, flux belongs
        # in the image). Started from a fit of the disk alone.
        diam_prior = dist.LogUniform(0.05 * resolution.minor_mas, resolution.major_mas)
        alone = fit(vm.UniformDisk(0.5 * resolution.minor_mas), {"diam": diam_prior}, data)
        start = vm.System(star=alone.model, env=start.env)
        star_priors = {"star.diam": diam_prior}
    elif star and star_model != "point":
        raise ValueError(f"star_model must be 'point' or 'disk', not {star_model!r}")
    wavel0 = float(np.median(np.concatenate([np.ravel(np.asarray(d.wavel)) for d in datasets])))
    img0 = image_of(start, star)
    pixel0 = float(img0.pixel_scale_mas)
    n0 = int(np.shape(img0.log_brightness)[0])
    # The evidence Jacobian (and CLEAN's columns) are data points x pixels:
    # cap the grid so that product stays within JACOBIAN_BUDGET. MATISSE's
    # 8-26k points on 255^2 pixels needed 432 GB on an 80 GB A100 (job
    # 18073823). The cap only bites for the largest datasets.
    max_pix = min(MAX_PIX, int(np.sqrt(JACOBIAN_BUDGET / max(npts, 1))))
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
        new = vm.Image(log_start, pixel, flux=img0.flux)
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
        cleaned = clean(data, n_c, pix_c, base=start.star if star else None, support=support_c,
                        max_iterations=clean_iters, target_chi2_red=1.0, gain=clean_gain,
                        base_priors={k.removeprefix("star."): v for k, v in star_priors.items()} or None)
        if star_priors:  # the diameter CLEAN fitted along with the components
            start = vm.System(star=cleaned.model.base, env=start.env)
        restore = vm_beam(resolution.major_mas * mean_blur, resolution.minor_mas * mean_blur, resolution.pa_deg)
        restored = np.clip(np.asarray(cleaned.restored(restore)), 0.0, None)
        if n_c != n_img:
            from scipy import ndimage

            restored = np.clip(ndimage.zoom(restored, n_img / n_c, order=1), 0.0, None)[:n_img, :n_img]
            if restored.shape != (n_img, n_img):  # zoom can come up a pixel short
                restored = np.pad(restored, [(0, n_img - restored.shape[0]), (0, n_img - restored.shape[1])], mode="edge")
        restored = restored / restored.max() + 1e-3  # a floor, so no pixel starts switched off
        flux = float(np.sum(cleaned.components)) if star else float(img0.flux)
        new = vm.Image(jnp.asarray(np.log(restored)), pix_img, support=support, flux=max(flux, 1e-3))
        if sparco and star:
            new = eqx.tree_at(lambda m: m.flux, new, PowerLaw(max(flux, 1e-3), 0.0, wavel0))
        start = vm.System(star=start.star, env=new) if star else new
        img0 = new
        clean_info = (cleaned.stop, float(np.ravel(cleaned.chi2_red)[-1]))
    else:
        clean_info = None
    if halo:
        parts = {"star": start.star, "env": start.env} if star else {"env": start}
        start = vm.System(**parts, halo=vm.Resolved(1.0))
    npix = int(np.shape(img0.log_brightness)[0])
    pixel = float(img0.pixel_scale_mas)
    fov = npix * pixel

    path = "env" if (star or halo) else None
    # With a star, the image flux is relative to it; starting_image can start
    # it well above 1 (5.8 for 2022 GRAVITY), so the prior must reach beyond.
    flux_cap = max(100.0, 10 * flux_value(img0.flux))
    # Jeffreys priors (Ben, 2026-10-05): fluxes are scale parameters, so
    # log-uniform, with stated bounds.
    priors = image_priors(start) | (flux_priors(flux_cap, sparco) if star else {}) | star_priors
    if halo:
        priors |= {"halo.flux": dist.LogUniform(FLUX_FLOOR, 1000.0)}
    others = () if star else (Centroid(0.1 * resolution.minor_mas, path=path),)
    return dict(label=label, files=files, star=star, halo=halo, data=data, npts=npts,
                star_model=star_model if star else "none", star_priors=star_priors, sparco=bool(sparco and star),
                wavel0=wavel0,
                resolution=resolution, start=start, img0=img0, npix=npix, pixel=pixel, fov=fov,
                path=path, priors=priors, others=others, q=q, adaptive=field is None,
                field_source=spec.get("field_source", "data" + (f" x{grow:g}" if grow > 1 else "")),
                clean_info=clean_info)


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


def grown_setup(task, data_dir, halo=False, star=None, init="moments", smoke=False):
    """setup(), with a field chosen from the data grown (x1.5) while doing so
    lowers χ² by more than 10% in a probe fit at a middle weight: what a
    contestant would try. (Flux at the image edge is not a usable trigger:
    the support and centroid prior keep it off the edge even when the field
    is too small.) Returns the setup and the growth record."""
    opts = dict(star=star, init=init, clean_iters=50 if smoke else 3000)
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
            bigger = setup(task, data_dir, halo, grow * 1.5, **opts)
            chi2_big = probe(bigger)
            growth.append((round(bigger["fov"], 3), round(chi2_big, 3)))
            if chi2_big > 0.9 * chi2:
                break
            grow, chi2, s = grow * 1.5, chi2_big, bigger
    return s, growth


def run(task, data_dir, out_dir, smoke=False, halo=False, star=None, init="moments", settings=None, label=None):
    """MaxEnt L-curve. With ``settings`` (scripts/contest_bench.py's "mem"
    arm) the field, pixels and CLEAN start follow those settings, as for a GP
    member, and the default model of the entropy is the CLEAN starting image
    (as BSMEM and MiRA use a prior image); the chosen image is also rendered
    on the common reference grid (``ref_image``) for scoring."""
    t0 = time.time()
    if settings is None:
        s, growth = grown_setup(task, data_dir, halo, star, init, smoke)
    else:
        s, settings = member_setup(task, data_dir, None, smoke, settings)
        growth, init, halo = [], "clean", s["halo"]
        template = np.asarray(s["img0"].brightness).reshape(s["npix"], s["npix"])
        s["q"] = template / template.sum()
    label_ = label
    label, files, star, data, npts = s["label"], s["files"], s["star"], s["data"], s["npts"]
    label = label_ or label
    resolution, start, img0, npix, pixel, fov = (s[k] for k in ("resolution", "start", "img0", "npix", "pixel", "fov"))
    path, priors, others = s["path"], s["priors"], s["others"]
    weights = WEIGHTS[[0, 6, -1]] if smoke else WEIGHTS
    options = {"max_steps": 50} if smoke else {"max_steps": 200_000}
    curve = l_curve(start, priors, data, MaxEntropy(1.0, prior=s["q"], path=path), jnp.asarray(weights), others, **options)

    # discrepancy() is None when no weight reaches χ² = N (the data are then
    # under-fitted at every weight: worth a note); corner() needs >= 3 points.
    # Fall back to the L-curve corner, then to the weakest weight.
    weakest = float(curve.weights[-1])
    corner = curve.corner()
    corner = weakest if corner is None else float(corner)
    discrepancy = curve.discrepancy()
    chosen = {"discrepancy": corner if discrepancy is None else float(discrepancy), "corner": corner}
    reached = discrepancy is not None
    index = {k: int(np.argmin(np.abs(np.log(curve.weights) - np.log(w)))) for k, w in chosen.items()}
    images = np.stack([np.asarray(image_of(r.model, star, halo).render(npix, fov)) for r in curve.results])
    # V² and closure phases have no absolute phases, so no dirty image:
    # show the starting image instead.
    first = np.asarray(img0.render(npix, fov))

    out_dir.mkdir(parents=True, exist_ok=True)
    reference = {}
    if settings is not None:
        chosen_model = curve.results[index["discrepancy"]].model
        reference = dict(ref_image=np.asarray(chosen_model.render(REF_NPIX, s["ref_fov"])), ref_fov=s["ref_fov"],
                         best_chi2_red=float(np.asarray(curve.chi2_red)[index["discrepancy"]]), error_scale=np.nan,
                         star=bool(star), best_log_z=np.nan, flip_dchi2=np.nan)
    np.savez_compressed(
        out_dir / f"{label}.npz", **reference,
        weights=np.asarray(curve.weights), chi2=np.asarray(curve.chi2),
        chi2_red=np.asarray(curve.chi2_red), penalty=np.asarray(curve.penalty),
        images=images, start=first, pixel_scale_mas=pixel, npix=npix,
        beam=np.array([resolution.major_mas, resolution.minor_mas, resolution.pa_deg]),
        halo_flux=np.array([float(r.model.halo.flux) if halo else 0.0 for r in curve.results]),
        converged=np.array([bool(r.info.get("converged", False)) for r in curve.results]),
        chosen=json.dumps(chosen), index=json.dumps(index),
    )


    best = curve.results[index["discrepancy"]].model
    report = diagnose(best, data, [MaxEntropy(chosen["discrepancy"], prior=s["q"], path=path), *others])
    summary = (
        f"task={task} label={label} files={files} star={star} halo={halo}\n"
        f"field_source={s['field_source']} clean={s['clean_info']} star_source={spec_of(task).get('star_source')} prior={spec_of(task).get('prior')} wavel={spec_of(task).get('wavel')} growth(fov, chi2/N)={growth}\n"
        f"virgil={virgil.__version__} from {virgil.__file__}\n"
        f"points={npts} npix={npix} pixel={pixel:.4g} mas fov={fov:.4g} mas "
        f"beam={resolution.major_mas:.3g}x{resolution.minor_mas:.3g} mas\n"
        f"weights={np.asarray(curve.weights).round(4).tolist()}\n"
        f"chi2_red={np.asarray(curve.chi2_red).round(3).tolist()}\n"
        f"halo_flux={[round(float(r.model.halo.flux), 4) for r in curve.results] if halo else None}\n"
        f"chosen={chosen} discrepancy_reached={reached}\nconverged={[bool(r.info.get('converged', False)) for r in curve.results]}\n"
        f"residuals at the discrepancy weight={residual_diagnostics(best, data)}\n"
        f"residuals at the corner weight={residual_diagnostics(curve.results[index['corner']].model, data)}\n"
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


def fit_ellipse(s):
    """Elongation and position angle from the data: an elliptical Gaussian
    (with the analytic star, if any), from a few starting angles."""
    fov, star = s["fov"], s["star"]
    best = None
    for pa0 in (0.0, 45.0, 90.0, 135.0):
        env = vm.EllipticalGaussian(0.3 * fov, 0.7, pa0, flux=flux_value(s["img0"].flux) if star else 1.0)
        scene = vm.System(star=s["start"].star, env=env) if star else vm.System(env=env)
        priors = {"env.fwhm": dist.LogUniform(0.01, fov), "env.ratio": dist.LogUniform(0.05, 1.0),
                  "env.pa": dist.Uniform(pa0 - 90.0, pa0 + 90.0)}
        if star:
            priors["env.flux"] = dist.LogUniform(FLUX_FLOOR, max(100.0, 10 * flux_value(s["img0"].flux)))
        r = fit(scene, priors, s["data"])
        chi2 = float(np.sum(r.info["chi2_red"]))
        if best is None or chi2 < best[0]:
            best = (chi2, float(r.values["env.ratio"]), float(r.values["env.pa"]) % 180.0, float(r.values["env.fwhm"]))
    return best


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
                mean_blur=settings.get("mean_blur", 1.0))
    if spec_of(task).get("field") is not None:
        settings["field"] = 1.0  # a contest-given field is used as given
    s = setup(task, data_dir, settings["halo"], settings["field"], **opts)
    s["ref_fov"] = reference_fov(task, data_dir)
    return s, settings


def reference_fov(task, data_dir):
    """The common grid's field for all members of a task: max(MEMBER_FIELDS) x the base field
    of the task's own configuration (its default star, a moments start), so it
    does not depend on any member's random settings or star choice."""
    base = setup(task, data_dir, False, 1.0, star=None, init="moments")
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


def run_gp(task, data_dir, out_dir, smoke=False, halo=False, star=None, init="moments", member=None,
           settings=None, label=None):
    """GP fits chosen by evidence. With ``member`` (or ``settings``, a fixed
    configuration as used by scripts/contest_bench.py) a CLEAN-started
    ensemble member; otherwise a grown-field run. ``label`` names the
    outputs."""
    t0 = time.time()
    if member is None and settings is None:
        s, growth = grown_setup(task, data_dir, halo, star, init, smoke)
        label = label or s["label"] + "_gp"
    else:
        s, settings = member_setup(task, data_dir, member, smoke, settings)
        growth, init = [], "clean"
        label = label or f"{spec_of(task)['label']}_m{member}"
    # A member's halo comes from its settings (setup's flag), not this
    # function's argument.
    halo = s["halo"]
    data, star, res = s["data"], s["star"], s["resolution"]
    img0, n, pixel = s["img0"], s["npix"], s["pixel"]
    fov = n * pixel
    template = np.asarray(img0.brightness).reshape(n, n)
    support = getattr(img0, "support", None)
    flux0 = img0.flux  # a number, or with SPARCO a PowerLaw
    flux_cap = max(100.0, 10 * flux_value(flux0))
    ell = fit_ellipse(s)  # (chi2, ratio, pa, fwhm)
    sigmas0 = GP_SIGMAS[1:2] if smoke else GP_SIGMAS
    lengths0 = [f * res.minor_mas for f in (GP_LENGTHS[1:2] if smoke else GP_LENGTHS)]
    length_limits = [f * res.minor_mas for f in GP_LENGTH_LIMITS]
    options = {"max_steps": 50} if smoke else {}

    def scene(field, rotation):
        env = vm.Image(field, pixel, support=support, flux=flux0, rotation_deg=rotation)
        parts = {"star": s["start"].star, "env": env} if star else {"env": env}
        if halo:
            parts["halo"] = vm.Resolved(1.0)
        return vm.System(**parts)

    others = () if star else (Centroid(0.1 * res.minor_mas, path="env"),)
    extra = ((flux_priors(flux_cap, s["sparco"]) | s["star_priors"]) if star else {}) | (
        {"halo.flux": dist.LogUniform(FLUX_FLOOR, 1000.0)} if halo else {})
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
            return float(log_evidence(r_, data))

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
    scale = float(error_scale(winner.model, data))
    # MacKay's effective number of parameters, from s² = χ²/(N − γ).
    n_indep = sum(d.n_independent for d in datasets)
    chi2_winner = float(np.sum(np.asarray(winner.info["chi2_red"]) * np.asarray([d.n_independent for d in datasets])))
    gamma = n_indep - chi2_winner / scale**2
    rescaled = None
    if not 1 / 1.3 < scale < 1.3:
        data_s = [d.with_error_scale(scale) for d in datasets]
        data_s = data_s if isinstance(data, list) else data_s[0]
        mean, rotation, lens = variants[best_name]
        field = GaussianField(np.zeros((n, n)), best_sigma, lens(best_length), mean=mean)
        sc = scene(field, rotation)
        rr = fit(sc, image_priors(sc) | extra, data_s, others, init=winner.values, **options)
        rescaled = {"scale": scale, "log_z": float(log_evidence(rr, data_s)),
                    "chi2_red": float(np.sum(rr.info["chi2_red"])), "result": rr}

    out_dir.mkdir(parents=True, exist_ok=True)
    def best_result(name):
        i, j = best_of[name]
        return results[name, axes[name][0][i], axes[name][1][j]]

    images = {name: np.asarray(best_result(name).model.env.render(n, fov)) for name in log_z}
    final = rescaled["result"] if rescaled else winner
    member_extra = {}
    if settings is not None:
        ref_fov = s["ref_fov"]
        member_extra = dict(
            # The whole scene (star included) on the common grid, so that star and
            # no-star members compare like with like; the environment alone too.
            ref_image=np.asarray(final.model.render(REF_NPIX, ref_fov)),
            ref_env=np.asarray(final.model.env.render(REF_NPIX, ref_fov)), ref_fov=ref_fov,
            star=settings["star"], star_model=settings["star_model"], halo=settings["halo"],
            sparco=settings["sparco"], field_factor=settings["field"], oversample=settings["oversample"],
            clean_gain=settings["clean_gain"], env_flux=flux_value(final.model.env.flux),
            env_index=float(np.ravel(getattr(final.model.env.flux, "index", np.nan))[0]),
            star_diam=float(np.ravel(getattr(final.model.star, "diam", np.nan))[0]) if settings["star"] else np.nan,
            n_independent=n_indep, gamma=gamma,
            best_log_z=best_log_z,
            best_chi2_red=float(np.sum(winner.info["chi2_red"])),
            error_scale=scale,
            rescaled_log_z=rescaled["log_z"] if rescaled else np.nan,
            rescaled_chi2_red=rescaled["chi2_red"] if rescaled else np.nan,
            flip_dchi2=float(np.ravel(diagnose(final.model, data if not rescaled else data_s,
                                                list(others)).checks["flip_dchi2"])[0]),
        )
    np.savez_compressed(out_dir / f"{label}.npz", **member_extra, **{f"log_z_{k}": v for k, v in log_z.items()},
                        **{f"image_{k}": v for k, v in images.items()},
                        **{f"sigmas_{k}": np.asarray(a[1]) for k, a in axes.items()},
                        **{f"lengths_mas_{k}": np.asarray(a[0]) for k, a in axes.items()},
                        ellipse=np.asarray(ell), pixel_scale_mas=pixel, npix=n,
                        template=template)
    lines = [f"task={task} label={label} star={star} halo={halo} init={init} field_source={s['field_source']} growth={growth}",
             f"points={s['npts']} npix={n} pixel={pixel:.4g} mas fov={fov:.4g} mas beam={res.major_mas:.3g}x{res.minor_mas:.3g} mas",
             f"ellipse fit: chi2={ell[0]:.4g} ratio={ell[1]:.3f} pa={ell[2]:.1f} deg fwhm={ell[3]:.3g} mas",
             f"star_model={s['star_model']} sparco={s['sparco']} wavel0={s['wavel0']:.4g}"]
    for name, g in log_z.items():
        ls, ss = axes[name]
        lines.append(f"{name}: sigmas={list(ss)} lengths_mas={[round(x, 4) for x in ls]}")
        lines.append(f"log_z[{name}] (rows ℓ, cols σ):\n{np.array2string(g, precision=2)}")
        i, j = best_of[name]
        rr = best_result(name)
        lines.append(f"best {name}: ℓ={ls[i]:.4g} σ={ss[j]} log_z={g[i, j]:.2f} chi2/N={float(np.sum(rr.info['chi2_red'])):.3f} converged={rr.info.get('converged')}")
    lines.append(f"residuals of the winner={residual_diagnostics(winner.model, data)}")
    lines.append(f"N={n_indep} gamma={gamma:.1f} N-gamma={n_indep - gamma:.1f}")
    lines.append(f"error_scale={scale:.3f}" + (f" -> refit with rescaled errors: log_z={rescaled['log_z']:.2f} chi2/N={rescaled['chi2_red']:.3f}" if rescaled else " (within 1.3x: kept)"))
    if settings is not None:
        lines.append(f"member settings={settings}")
    lines.append(f"WINNER: {best_name} ℓ={best_length:.4g} σ={best_sigma} log_z={best_log_z:.2f}")
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
    for ax, (name, g) in zip(panels[1 + len(images):], log_z.items()):
        ls, ss = axes[name]
        ax.imshow(g - np.nanmax(g), origin="lower", cmap="viridis", aspect="auto")
        ax.set(xticks=range(len(ss)), xticklabels=ss, yticks=range(len(ls)),
               yticklabels=[f"{x:.2g}" for x in ls], xlabel="σ", ylabel="ℓ (mas)", title=f"log Z − max ({name})")
    plt.tight_layout()
    fig.savefig(out_dir / f"{label}.png", dpi=100)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=int)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--data", default="~/data/imaging_contests")
    parser.add_argument("--out", default="contest_images")
    parser.add_argument("--halo", action="store_true", help="add a fully resolved component (labels get _halo)")
    parser.add_argument("--star", choices=("on", "off"), help="override the task's central star (labels get _star/_nostar)")
    parser.add_argument("--init", choices=("moments", "clean"), default="moments", help="starting image (labels get _clean)")
    parser.add_argument("--prior", choices=("mem", "gp"), default="mem", help="MaxEnt L-curve, or Gaussian-process fits chosen by evidence (labels get _gp)")
    parser.add_argument("--member", type=int, help="ensemble member (CLEAN-started GP with a randomised start)")
    parser.add_argument("--smoke", action="store_true", help="three weights, 50 steps: check that it runs")
    args = parser.parse_args()
    if args.list:
        for i, spec in enumerate(TASKS):
            print(f"{i:2d}  {spec['label']:20s} field={spec.get('field')}  prior={spec.get('prior')}  "
                  f"star={spec.get('star', False)}  wavel={spec.get('wavel')}  {len(spec['files'])} file(s)")
        return
    star = None if args.star is None else args.star == "on"
    if args.member is not None:
        run_gp(args.task, pathlib.Path(args.data).expanduser(), pathlib.Path(args.out).expanduser(), args.smoke,
               member=args.member)
        return
    runner = run_gp if args.prior == "gp" else run
    runner(args.task, pathlib.Path(args.data).expanduser(), pathlib.Path(args.out).expanduser(), args.smoke,
           args.halo, star, args.init)


if __name__ == "__main__":
    main()
