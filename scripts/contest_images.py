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

The settings come from the contest papers, not from looking at the
answers: the field of view is the published truth image's when known, and a
central star is put in analytically only where the organisers describe one.
The images are compared with the published entries in
docs/plan_imaging_contests.md (Stage C2). This is heavy: run it on OzSTAR
(ozstar_scripts job `contest_imaging`), not on a laptop; --smoke is a
seconds-long check that the script runs.
"""

import argparse
import json
import pathlib
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
from virgil.oidata import OIData  # noqa: E402
from virgil.plotting import plot_model  # noqa: E402

# label, files (relative to the data directory), largest field of view (mas;
# None: virgil's default from the data), analytic star at the centre.
TASKS = [
    # 2004: NPOI at 550 nm. data1 is Tuthill's LkHa 101 model (242 x 0.05 mas);
    # data2 a spotted elliptical star and a companion 10 mas East.
    ("2004_data1", ["2004/2004-data1.fits"], 12.0, False),
    ("2004_data2", ["2004/2004-data2.fits"], 24.0, False),
    # 2006: AMBER on the UTs, four nights; Chesneau's disk (301 x 0.35 mas).
    # Needs virgil#167 (merged 2026-10-04) to read.
    ("2006_disk", [f"2006/2006-03-0{n}.fits" for n in (3, 4, 5, 6)], 105.0, False),
    # 2008: CHARA; the field is tapered by a 15 mas FWHM Gaussian; each band
    # is a different model, so each is imaged alone.
    ("2008_agb_J", ["2008/2008-Contest1_J.oifits"], 30.0, False),
    ("2008_agb_H", ["2008/2008-Contest1_H.oifits"], 30.0, False),
    ("2008_agb_K", ["2008/2008-Contest1_K.oifits"], 30.0, False),
    ("2008_agn_J", ["2008/2008-Contest2_J.oifits"], 30.0, False),
    ("2008_agn_H", ["2008/2008-Contest2_H.oifits"], 30.0, False),
    ("2008_agn_K", ["2008/2008-Contest2_K.oifits"], 30.0, False),
    # 2010: AMBER low resolution, H and K treated as grey (a first pass; the
    # supergiant is about 20 mas, its companion at 84 mas).
    ("2010_lowHK", ["2010/Mystery-Low_HK.oifits"], None, False),
    # 2022: high-contrast tests; a star with faint structure around it.
    ("2022_gravity", ["2022/c_imaging_contest1.fits"], None, True),
    ("2022_ami", ["2022/c_imaging_contest2.fits"], None, True),
    # 2024: chromatic; grey images per instrument are only a first look
    # (Stage C3 does them properly). Obj1 is a WR pinwheel, Obj2 a Herbig disk.
    ("2024_obj1_pionier", ["2024/Obj1_PIONIER_1.5-1.8.fits"], None, False),
    ("2024_obj1_gravity", ["2024/Obj1_GRAVITY_2.0-2.5.fits"], None, False),
    ("2024_obj2_pionier", ["2024/Obj2_PIONIER_1.5-1.8.fits"], None, True),
    ("2024_obj2_gravity", ["2024/Obj2_GRAVITY_2.0-2.5.fits"], None, True),
]

WEIGHTS = np.logspace(4.0, 0.0, 13)  # strong to weak


def image_of(model, star, halo=False):
    return model.env if (star or halo) else model


def run(task, data_dir, out_dir, smoke=False, halo=False):
    label, files, largest, star = TASKS[task]
    if halo:
        label += "_halo"
    t0 = time.time()
    paths = [str(data_dir / f) for f in files]
    data = OIData(paths[0]) if len(paths) == 1 else [OIData(p) for p in paths]
    datasets = data if isinstance(data, list) else [data]
    npts = sum(np.asarray(d.flatten_data()[0]).size for d in datasets)
    resolution = beam(data)

    kwargs = {"star": star, "largest_mas": largest}
    if star:
        kwargs["hole_mas"] = 0.5 * resolution.minor_mas
    start = starting_image(data, **kwargs)
    img0 = image_of(start, star)
    if halo:
        # A fully resolved component (zero visibility on every baseline) for
        # flux in structure larger than the shortest baseline sees: e.g. the
        # 2006 disk, whose V² is ~1e-3 on every UT baseline.
        parts = {"star": start.star, "env": start.env} if star else {"env": start}
        start = vm.System(**parts, halo=vm.Resolved(1.0))
    npix = int(np.shape(img0.log_brightness)[0])
    pixel = float(img0.pixel_scale_mas)
    fov = npix * pixel

    path = "env" if (star or halo) else None
    priors = image_priors(start) | ({"env.flux": dist.Uniform(0.0, 1.0)} if star else {})
    if halo:
        priors |= {"halo.flux": dist.Uniform(0.0, 1000.0)}
    others = () if star else (Centroid(0.1 * resolution.minor_mas, path=path),)
    weights = WEIGHTS[[0, 6, -1]] if smoke else WEIGHTS
    options = {"max_steps": 50} if smoke else {"max_steps": 200_000}
    curve = l_curve(start, priors, data, MaxEntropy(1.0, path=path), jnp.asarray(weights), others, **options)

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

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    axes[0].plot(curve.penalty, curve.chi2, "o-", ms=4)
    for (name, k), m in zip(index.items(), ("s", "^")):
        axes[0].plot(curve.penalty[k], curve.chi2[k], m, ms=11, mfc="none", mew=2, label=f"{name}: w = {chosen[name]:.3g}")
    axes[0].axhline(npts, color="k", ls=":", lw=1, label="χ² = N")
    axes[0].set(xscale="log", yscale="log", xlabel="negative entropy", ylabel="χ²", title=label)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    plot_model(img0, fov_mas=fov, npix=npix, ax=axes[1], beam=resolution, title="starting image")
    for ax, name in zip(axes[2:], ("corner", "discrepancy")):
        r = curve.results[index[name]]
        plot_model(image_of(r.model, star, halo), fov_mas=fov, npix=npix, ax=ax, beam=resolution,
                   title=f"{name}: w = {float(curve.weights[index[name]]):.3g}, χ²/N = {float(np.sum(r.info['chi2_red'])):.2f}")
    plt.tight_layout()
    fig.savefig(out_dir / f"{label}.png", dpi=110)

    best = curve.results[index["discrepancy"]].model
    report = diagnose(best, data, [MaxEntropy(chosen["discrepancy"], path=path), *others])
    summary = (
        f"task={task} label={label} files={files} star={star} halo={halo}\n"
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
        for i, (label, files, largest, star) in enumerate(TASKS):
            print(f"{i:2d}  {label:20s} fov<={largest}  star={star}  {files}")
        return
    run(args.task, pathlib.Path(args.data).expanduser(), pathlib.Path(args.out).expanduser(), args.smoke, args.halo)


if __name__ == "__main__":
    main()
