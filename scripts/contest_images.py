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
from virgil.imaging import (  # noqa: E402
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


def setup(task, data_dir, halo=False, grow=1.0):
    """The data, starting model, priors and fixed regularisers for one task
    (shared with scripts/diagnose_stall.py). ``grow`` enlarges a field chosen
    from the data."""
    spec = TASKS[task]
    label, files, star = spec["label"], spec["files"], spec.get("star", False)
    field, prior_spec = spec.get("field"), spec.get("prior")
    if halo:
        label += "_halo"
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
                field_source=spec.get("field_source", "data" + (f" x{grow:g}" if grow > 1 else "")))


def run(task, data_dir, out_dir, smoke=False, halo=False):
    t0 = time.time()
    s = setup(task, data_dir, halo)
    # A field chosen from the data grows (x1.5) while doing so lowers χ² by
    # more than 10% in a probe fit at a middle weight: what a contestant would
    # try. (Flux at the image edge is not a usable trigger here: the support
    # and centroid prior keep it off the edge even when the field is too small.)
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
            bigger = setup(task, data_dir, halo, grow * 1.5)
            chi2_big = probe(bigger)
            growth.append((round(bigger["fov"], 3), round(chi2_big, 3)))
            if chi2_big > 0.9 * chi2:
                break
            grow, chi2, s = grow * 1.5, chi2_big, bigger
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
        f"field_source={s['field_source']} star_source={TASKS[task].get('star_source')} prior={TASKS[task].get('prior')} wavel={TASKS[task].get('wavel')} growth(fov, chi2/N)={growth}\n"
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=int)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--data", default="~/data/imaging_contests")
    parser.add_argument("--out", default="contest_images")
    parser.add_argument("--halo", action="store_true", help="add a fully resolved component (labels get _halo)")
    parser.add_argument("--smoke", action="store_true", help="three weights, 50 steps: check that it runs")
    args = parser.parse_args()
    if args.list:
        for i, spec in enumerate(TASKS):
            print(f"{i:2d}  {spec['label']:20s} field={spec.get('field')}  prior={spec.get('prior')}  "
                  f"star={spec.get('star', False)}  wavel={spec.get('wavel')}  {len(spec['files'])} file(s)")
        return
    run(args.task, pathlib.Path(args.data).expanduser(), pathlib.Path(args.out).expanduser(), args.smoke, args.halo)


if __name__ == "__main__":
    main()
