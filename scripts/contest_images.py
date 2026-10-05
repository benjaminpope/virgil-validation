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
from virgil.imaging import (  # noqa: E402
    clean,
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
from virgil.oidata import OIData  # noqa: E402
from virgil.plotting import plot_model  # noqa: E402

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
    dict(label="2004_data2", files=["2004/2004-data2.fits"]),
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
    dict(label="2010_lowH", files=["2010/Mystery-Low_HK.oifits"], wavel=(1.4, 1.9)),
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
    dict(label="2024_obj2_pionier", files=["2024/Obj2_PIONIER_1.5-1.8.fits"], star=True,
         star_source="contest page: a young star"),
    dict(label="2024_obj2_gravity", files=["2024/Obj2_GRAVITY_2.0-2.5.fits"], star=True,
         star_source="contest page: a young star"),
    # New tasks go at the end, so that earlier indices keep their meaning.
    dict(label="2010_lowK", files=["2010/Mystery-Low_HK.oifits"], wavel=(1.9, 2.6)),
    # 2018: "a young star's disk, with a planet" (readme): a star plus an image.
    dict(label="2018_disk", star=True, star_source="readme: a young star's disk", files=[
        "2018/Aspro2_Altair_MIRC_6T_1_47493-1_75256-8ch_S1-S2-E1-E2-W1-W2_2018-08-02_FAKE.fits",
        "2018/Aspro2_Altair_PIONIER_1_533-1_772-6ch_A0-B2-C1-D0_2018-08-02_FAKE.fits",
        "2018/Aspro2_Altair_PIONIER_1_533-1_772-6ch_A0-G1-J2-J3_2018-08-02_FAKE.fits",
        "2018/Aspro2_Altair_PIONIER_1_533-1_772-6ch_D0-G2-J3-K0_2018-08-02_FAKE.fits"]),
]

WEIGHTS = np.logspace(4.0, 0.0, 13)  # strong to weak
MAX_PIX = 256  # pixels on a side; larger fields get coarser pixels
CLEAN_MAX_PIX = 65  # CLEAN runs on at most this grid (see setup())
GROWTHS = 3  # data-chosen fields grow at most three times (see run())


def image_of(model, star, halo=False):
    return model.env if (star or halo) else model


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


def setup(task, data_dir, halo=False, grow=1.0, star=None, init="moments", clean_iters=3000):
    """The data, starting model, priors and fixed regularisers for one task
    (shared with scripts/diagnose_stall.py). ``grow`` enlarges a field chosen
    from the data."""
    spec = TASKS[task]
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

    kwargs = {"star": star}
    if star:
        kwargs["hole_mas"] = 0.5 * resolution.minor_mas
    start = starting_image(data, **kwargs)
    img0 = image_of(start, star)
    pixel0 = float(img0.pixel_scale_mas)
    n0 = int(np.shape(img0.log_brightness)[0])
    want = field if field is not None else (n0 * pixel0 * grow if grow > 1 else None)
    if want is None and n0 > MAX_PIX:  # starting_image's own grid can exceed the cap
        want = n0 * pixel0
    q = None
    if want is not None:
        n = int(np.ceil(want / pixel0)) | 1
        if n > MAX_PIX:  # the largest odd size within the cap, coarser pixels
            n = MAX_PIX - 1 if MAX_PIX % 2 == 0 else MAX_PIX
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
                        max_iterations=clean_iters, target_chi2_red=1.0)
        restored = np.clip(np.asarray(cleaned.restored(resolution)), 0.0, None)
        if n_c != n_img:
            from scipy import ndimage

            restored = np.clip(ndimage.zoom(restored, n_img / n_c, order=1), 0.0, None)[:n_img, :n_img]
            if restored.shape != (n_img, n_img):  # zoom can come up a pixel short
                restored = np.pad(restored, [(0, n_img - restored.shape[0]), (0, n_img - restored.shape[1])], mode="edge")
        restored = restored / restored.max() + 1e-3  # a floor, so no pixel starts switched off
        flux = float(np.sum(cleaned.components)) if star else float(img0.flux)
        new = vm.Image(jnp.asarray(np.log(restored)), pix_img, support=support, flux=max(flux, 1e-3))
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
    flux_cap = max(100.0, 10 * float(img0.flux))
    priors = image_priors(start) | ({"env.flux": dist.Uniform(0.0, flux_cap)} if star else {})
    if halo:
        priors |= {"halo.flux": dist.Uniform(0.0, 1000.0)}
    others = () if star else (Centroid(0.1 * resolution.minor_mas, path=path),)
    return dict(label=label, files=files, star=star, halo=halo, data=data, npts=npts,
                resolution=resolution, start=start, img0=img0, npix=npix, pixel=pixel, fov=fov,
                path=path, priors=priors, others=others, q=q, adaptive=field is None,
                field_source=spec.get("field_source", "data" + (f" x{grow:g}" if grow > 1 else "")),
                clean_info=clean_info)


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


def run(task, data_dir, out_dir, smoke=False, halo=False, star=None, init="moments"):
    t0 = time.time()
    s, growth = grown_setup(task, data_dir, halo, star, init, smoke)
    label, files, star, data, npts = s["label"], s["files"], s["star"], s["data"], s["npts"]
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
    np.savez_compressed(
        out_dir / f"{label}.npz",
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
        f"field_source={s['field_source']} clean={s['clean_info']} star_source={TASKS[task].get('star_source')} prior={TASKS[task].get('prior')} wavel={TASKS[task].get('wavel')} growth(fov, chi2/N)={growth}\n"
        f"virgil={virgil.__version__} from {virgil.__file__}\n"
        f"points={npts} npix={npix} pixel={pixel:.4g} mas fov={fov:.4g} mas "
        f"beam={resolution.major_mas:.3g}x{resolution.minor_mas:.3g} mas\n"
        f"weights={np.asarray(curve.weights).round(4).tolist()}\n"
        f"chi2_red={np.asarray(curve.chi2_red).round(3).tolist()}\n"
        f"halo_flux={[round(float(r.model.halo.flux), 4) for r in curve.results] if halo else None}\n"
        f"chosen={chosen} discrepancy_reached={reached}\nconverged={[bool(r.info.get('converged', False)) for r in curve.results]}\n"
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


def fit_ellipse(s):
    """Elongation and position angle from the data: an elliptical Gaussian
    (with the analytic star, if any), from a few starting angles."""
    fov, star = s["fov"], s["star"]
    best = None
    for pa0 in (0.0, 45.0, 90.0, 135.0):
        env = vm.EllipticalGaussian(0.3 * fov, 0.7, pa0, flux=float(s["img0"].flux) if star else 1.0)
        scene = vm.System(star=s["start"].star, env=env) if star else vm.System(env=env)
        priors = {"env.fwhm": dist.Uniform(0.01, fov), "env.ratio": dist.Uniform(0.05, 1.0),
                  "env.pa": dist.Uniform(pa0 - 90.0, pa0 + 90.0)}
        if star:
            priors["env.flux"] = dist.Uniform(0.0, max(100.0, 10 * float(s["img0"].flux)))
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


def run_gp(task, data_dir, out_dir, smoke=False, halo=False, star=None, init="moments"):
    t0 = time.time()
    s, growth = grown_setup(task, data_dir, halo, star, init, smoke)
    label = s["label"] + "_gp"
    data, star, res = s["data"], s["star"], s["resolution"]
    img0, n, pixel = s["img0"], s["npix"], s["pixel"]
    fov = n * pixel
    template = np.asarray(img0.brightness).reshape(n, n)
    support = getattr(img0, "support", None)
    flux0 = float(img0.flux)
    flux_cap = max(100.0, 10 * flux0)
    ell = fit_ellipse(s)  # (chi2, ratio, pa, fwhm)
    sigmas = GP_SIGMAS[1:2] if smoke else GP_SIGMAS
    lengths = [f * res.minor_mas for f in (GP_LENGTHS[1:2] if smoke else GP_LENGTHS)]
    options = {"max_steps": 50} if smoke else {}

    def scene(field, rotation):
        env = vm.Image(field, pixel, support=support, flux=flux0, rotation_deg=rotation)
        parts = {"star": s["start"].star, "env": env} if star else {"env": env}
        if halo:
            parts["halo"] = vm.Resolved(1.0)
        return vm.System(**parts)

    others = () if star else (Centroid(0.1 * res.minor_mas, path="env"),)
    extra = ({"env.flux": dist.Uniform(0.0, flux_cap)} if star else {}) | ({"halo.flux": dist.Uniform(0.0, 1000.0)} if halo else {})
    variants = {"iso": (template, 0.0, lambda length: length)}
    if ell[1] < 0.9:  # elongated enough for an anisotropic field to differ
        r = ell[1]
        variants["aniso"] = (oriented_template(template, ell[2], pixel, n), ell[2],
                             lambda length: (length / np.sqrt(r), length * np.sqrt(r)))
    results, log_z = {}, {}
    for name, (mean, rotation, lens) in variants.items():
        grid = np.full((len(lengths), len(sigmas)), np.nan)
        previous = None
        for i, length in enumerate(lengths):
            for j, sigma in enumerate(sigmas):
                field = GaussianField(np.zeros((n, n)), sigma, lens(length), mean=mean)
                sc = scene(field, rotation)
                r_ = fit(sc, image_priors(sc) | extra, data, others,
                         init=None if previous is None else previous.values, **options)
                previous = r_
                grid[i, j] = float(log_evidence(r_, data))
                results[name, i, j] = r_
        log_z[name] = grid
    best = max(((name, i, j) for name, g in log_z.items() for (i, j), _ in np.ndenumerate(g)),
               key=lambda k: log_z[k[0]][k[1], k[2]])
    winner = results[best]

    out_dir.mkdir(parents=True, exist_ok=True)
    best_of = {name: max(((i, j) for (i, j), _ in np.ndenumerate(g)), key=lambda k: g[k]) for name, g in log_z.items()}
    images = {name: np.asarray(results[(name, *ij)].model.env.render(n, fov)) for name, ij in best_of.items()}
    np.savez_compressed(out_dir / f"{label}.npz", **{f"log_z_{k}": v for k, v in log_z.items()},
                        **{f"image_{k}": v for k, v in images.items()}, sigmas=np.asarray(sigmas),
                        lengths_mas=np.asarray(lengths), ellipse=np.asarray(ell), pixel_scale_mas=pixel, npix=n,
                        template=template)
    lines = [f"task={task} label={label} star={star} halo={halo} init={init} field_source={s['field_source']} growth={growth}",
             f"points={s['npts']} npix={n} pixel={pixel:.4g} mas fov={fov:.4g} mas beam={res.major_mas:.3g}x{res.minor_mas:.3g} mas",
             f"ellipse fit: chi2={ell[0]:.4g} ratio={ell[1]:.3f} pa={ell[2]:.1f} deg fwhm={ell[3]:.3g} mas",
             f"sigmas={list(sigmas)} lengths_mas={[round(x, 4) for x in lengths]}"]
    for name, g in log_z.items():
        lines.append(f"log_z[{name}] (rows ℓ, cols σ):\n{np.array2string(g, precision=2)}")
        i, j = best_of[name]
        rr = results[(name, i, j)]
        lines.append(f"best {name}: ℓ={lengths[i]:.4g} σ={sigmas[j]} log_z={g[i, j]:.2f} chi2/N={float(np.sum(rr.info['chi2_red'])):.3f} converged={rr.info.get('converged')}")
    lines.append(f"WINNER: {best[0]} ℓ={lengths[best[1]]:.4g} σ={sigmas[best[2]]} log_z={log_z[best[0]][best[1], best[2]]:.2f}")
    lines.append(f"elapsed={time.time() - t0:.0f}s\n\n{diagnose(winner.model, data, list(others))}")
    text = "\n".join(lines) + "\n"
    (out_dir / f"{label}.txt").write_text(text)
    print(text)

    fig, axes = plt.subplots(1, 1 + len(images) + len(log_z), figsize=(4.6 * (1 + len(images) + len(log_z)), 4.4))
    plot_model(vm.Image(jnp.asarray(np.log(template + 1e-12)), pixel), fov_mas=fov, npix=n, ax=axes[0], beam=res,
               title=f"template ({init})")
    for ax, (name, im) in zip(axes[1:], images.items()):
        i, j = best_of[name]
        plot_model(results[(name, i, j)].model.env, fov_mas=fov, npix=n, ax=ax, beam=res,
                   title=f"GP {name}: ℓ={lengths[i]:.2g} mas, σ={sigmas[j]:g}, logZ={log_z[name][i, j]:.0f}")
    for ax, (name, g) in zip(axes[1 + len(images):], log_z.items()):
        ax.imshow(g - np.nanmax(g), origin="lower", cmap="viridis", aspect="auto")
        ax.set(xticks=range(len(sigmas)), xticklabels=sigmas, yticks=range(len(lengths)),
               yticklabels=[f"{x:.2g}" for x in lengths], xlabel="σ", ylabel="ℓ (mas)", title=f"log Z − max ({name})")
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
    parser.add_argument("--smoke", action="store_true", help="three weights, 50 steps: check that it runs")
    args = parser.parse_args()
    if args.list:
        for i, spec in enumerate(TASKS):
            print(f"{i:2d}  {spec['label']:20s} field={spec.get('field')}  prior={spec.get('prior')}  "
                  f"star={spec.get('star', False)}  wavel={spec.get('wavel')}  {len(spec['files'])} file(s)")
        return
    star = None if args.star is None else args.star == "on"
    runner = run_gp if args.prior == "gp" else run
    runner(args.task, pathlib.Path(args.data).expanduser(), pathlib.Path(args.out).expanduser(), args.smoke,
           args.halo, star, args.init)


if __name__ == "__main__":
    main()
