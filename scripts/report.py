"""Regenerate docs/results.md and the figures quoted in the README.

    .venv/bin/python scripts/report.py [--pulls 200]

Runs one process at a time (dLux and virgil are both JAX): about 10 minutes
on a laptop.
"""

import argparse
import pathlib
import tempfile
import time
import warnings

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
jax.config.update("jax_enable_x64", True)

import virgil.models as vm  # noqa: E402

import virgil_bridge as vb  # noqa: E402
from crosscheck import array, nrm, simulate, sky  # noqa: E402

DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"
UTS = np.array(
    [[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]]
)
HA = np.linspace(-3, 3, 7)
WL = np.linspace(1.5e-6, 2.4e-6, 6)
DEC = -50.0
TMP = pathlib.Path(tempfile.mkdtemp())
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(s)


def parity():
    rng = np.random.default_rng(2026)
    u, v = rng.uniform(-130, 130, (2, 500))
    w = rng.uniform(1.5e-6, 2.4e-6, 500)

    def V(m):
        return np.asarray(m.model(u, v, w))

    blur = sky.gaussian_blur_factor
    ring = sky.inclined_ring(4.0, 50.0, 30.0, (0.4, 0.25), (40.0, 110.0), "disk", 0.3, -0.2)
    ring_sky = sky.inclined_ring(4.0, 50.0, 30.0, (0.4, 0.25), (40.0, 110.0), "sky", 0.3, -0.2)
    rim = vm.ModulatedGaussianRim(4.0, 0.8, 50.0, 30.0, np.array([0.4, 0.25]), np.array([40.0, 110.0]), dra=0.3, ddec=-0.2)
    img = rng.random((7, 10))
    rows = [
        ("PointSource", sky.vis_point(u, v, w, 3.1, -2.2), V(vm.PointSource(dra=3.1, ddec=-2.2)), "exact phase"),
        ("GaussianDisk", sky.vis_gaussian(u, v, w, 1.2, 0.4, -0.7), V(vm.GaussianDisk(1.2, dra=0.4, ddec=-0.7)), "closed form"),
        ("EllipticalGaussian", sky.vis_elliptical_gaussian(u, v, w, 3, 0.4, 35, 1, 2), V(vm.EllipticalGaussian(3, 0.4, 35, dra=1, ddec=2)), "closed form"),
        ("UniformDisk (12 mas, 4 nulls)", sky.vis_uniform_disk(u, v, w, 12.0, 0.4, -0.7), V(vm.UniformDisk(12.0, dra=0.4, ddec=-0.7)), "SciPy J1"),
        ("BinaryModelAngular", (sky.vis_point(u, v, w) + 0.1 * sky.vis_point(u, v, w, 5 * np.sin(np.deg2rad(70)), 5 * np.cos(np.deg2rad(70)))) / 1.1, V(vm.BinaryModelAngular(5.0, 70.0, 0.1)), "exact"),
        ("ModulatedGaussianRim, in-plane azimuth and blur", sky.visibility(ring, u, v, w) * sky.in_plane_blur_factor(u, v, w, 0.8, 50.0, 30.0), V(rim), "ring quadrature x in-plane Gaussian"),
        ("ModulatedGaussianRim, sky azimuth", sky.visibility(ring_sky, u, v, w) * sky.in_plane_blur_factor(u, v, w, 0.8, 50.0, 30.0), V(rim), "(other reading of the docs)"),
        ("ModulatedGaussianRim, blur round on the sky (before virgil#139)", sky.visibility(ring, u, v, w) * blur(u, v, w, 0.8), V(rim), "(old definition)"),
        ("GaussianArc (default nodes)", sky.visibility(sky.arc_curve(5.0, 4.0, 250.0, 0.5, 0.3), u, v, w) * blur(u, v, w, 0.6), V(vm.GaussianArc(5.0, 0.6, 4.0, 250.0, dra=0.5, ddec=0.3)), "full-circle quadrature"),
        ("Image (7x10 random, orientation)", sky.visibility(sky.pixel_image(img, 0.5, 0.2, 0.1), u, v, w), V(vm.Image(np.log(img), 0.5, dra=0.2, ddec=0.1)), "pixel DFT"),
        ("Resolved dilution", sky.visibility(sky.mix([sky.uniform_disk(1.0)], [1.0], 0.3), u, v, w), V(vm.System(s=vm.UniformDisk(1.0), r=vm.Resolved(0.3))), "quadrature"),
    ]
    out("## Model visibilities (float64, 500 random VLTI-like samples)\n")
    out("| virgil model | our route | max abs(ΔV) |")
    out("| --- | --- | --- |")
    for name, ours, theirs, how in rows:
        out(f"| {name} | {how} | {np.max(np.abs(ours - theirs)):.1e} |")
    out()


def vlti(n_pulls):
    out("## Long-baseline injection and recovery (4 UTs, 7 hour angles, 6 channels)\n")
    out("| scene | file vs virgil model | max relative error, noise-free fit |")
    out("| --- | --- | --- |")
    path = TMP / "v.fits"
    for make in vb.SCENES:
        s = make()
        simulate.observe(path, s.vis, UTS, hour_angles_h=HA, wavelengths=WL, dec_deg=DEC, sigma_v2=0.01, sigma_cp_deg=0.5)
        d = vb.load(path)
        obs, _ = d.flatten_data()
        diff = np.max(np.abs(np.asarray(obs) - np.asarray(d.model(s.template))))
        res, _ = vb.fit_scene(s, d)
        tr, got = vb.flat_truth(s), vb.flat_values(s, res.values)
        rel = np.max(np.abs(got - tr) / np.maximum(np.abs(tr), 1e-3))
        out(f"| {s.name} | {diff:.1e} | {rel:.1e} |")
    out()

    out(f"### Pulls over {n_pulls} noisy realisations (σ(V²) = 0.02, σ(CP) = 1°)\n")
    out("(fit − truth) / σ, with σ from virgil's Laplace covariance. Closure-phase noise is either drawn per baseline (so triangles sharing a baseline are correlated, as in real data) or independently per triangle.\n")
    out("With 200 draws the sampling sd of a mean is 0.07 and of an sd 0.05. virgil whitens closure phases as a correlated group, as they are when formed from baseline phases, so the `baseline` rows test the realistic case; with independent noise per triangle (`triangle`) its errors are slightly conservative.\n")
    out("| scene | CP noise | parameter | mean | sd |")
    out("| --- | --- | --- | --- | --- |")
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True)
    for col, make in enumerate(vb.SCENES):
        s = make()
        for row, mode in enumerate(["baseline", "triangle"]):
            rng = np.random.default_rng(100 + col)
            z = []
            for _ in range(n_pulls):
                simulate.observe(path, s.vis, UTS, hour_angles_h=HA, wavelengths=WL, dec_deg=DEC, sigma_v2=0.02, sigma_cp_deg=1.0, rng=rng, phase_noise=mode)
                res, cov = vb.fit_scene(s, vb.load(path), start=s.truth)
                z.append((vb.flat_values(s, res.values) - vb.flat_truth(s)) / np.sqrt(np.diag(cov)))
            z = np.array(z)
            for k, p in enumerate(s.truth):
                out(f"| {s.name} | {mode} | `{p}` | {z[:, k].mean():+.2f} | {z[:, k].std():.2f} |")
            ax = axes[row, col]
            grid = np.linspace(-4, 4, 200)
            for k, p in enumerate(s.truth):
                ax.hist(z[:, k], bins=np.linspace(-4, 4, 25), density=True, histtype="step", label=p)
            ax.plot(grid, np.exp(-grid**2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1)
            ax.set_title(f"{s.name}\n{mode} CP noise", fontsize=9)
            ax.legend(fontsize=6)
    for ax in axes[1]:
        ax.set_xlabel("pull (fit − truth) / σ")
    fig.tight_layout()
    fig.savefig(DOCS / "vlti_pulls.png", dpi=120)
    out("\n![pulls](vlti_pulls.png)\n")


def masking():
    wl = float(np.float32(4.8e-6))
    uv = array.pupil_uv(nrm.HOLES)
    out("## Aperture masking with dLux (7-hole NIRISS-like mask, 4.8 µm, 30 mas pixels)\n")
    out("max abs(ΔV) between the calibrated visibilities (image FT / point-source image FT at the 21 baselines) and the exact scene, for two detector sizes. The two imagers share nothing but the point cloud.\n")
    out("| scene | points | dLux 128 px | analytic 128 px | dLux 256 px | analytic 256 px | dLux vs analytic, 256 px |")
    out("| --- | --- | --- | --- | --- | --- | --- |")
    scenes = vb.masking_scenes()
    res = {}
    for npix in (128, 256):
        optics = nrm.dlux_optics(npix=npix)
        cal = (nrm.image_dlux(sky.point(), wl, optics), nrm.image_analytic(sky.point(), wl, npix=npix))
        for s in scenes:
            sci = (nrm.image_dlux(s.cloud, wl, optics), nrm.image_analytic(s.cloud, wl, npix=npix))
            vd = nrm.calibrated_vis_fn(sci[0], cal[0], wl, 30.0)(*uv, wl)
            va = nrm.calibrated_vis_fn(sci[1], cal[1], wl, 30.0)(*uv, wl)
            res[(s.name, npix)] = (vd, va, sci, cal)
    for s in scenes:
        ex = s.vis(*uv, wl)
        e = [np.max(np.abs(res[(s.name, n)][i] - ex)) for n in (128, 256) for i in (0, 1)]
        dd = np.max(np.abs(res[(s.name, 256)][0] - res[(s.name, 256)][1]))
        out(f"| {s.name} | {s.cloud.weight.size} | " + " | ".join(f"{x:.1e}" for x in e) + f" | {dd:.1e} |")
    out()

    out("### virgil fits to the noise-free dLux observables (256 px)\n")
    out("Bias in units of the Laplace σ for realistic errors (σ(CP) = 1°, σ(V²) = 0.02).\n")
    out("| scene | parameter | truth | fit | bias / σ |")
    out("| --- | --- | --- | --- | --- |")
    path = TMP / "n.fits"
    for s in scenes:
        vd, _, sci, cal = res[(s.name, 256)]
        vis = nrm.calibrated_vis_fn(sci[0], cal[0], wl, 30.0)
        simulate.observe(path, vis, nrm.HOLES, hour_angles_h=[0.0], wavelengths=[wl], fixed_uv=uv, sigma_v2=0.02, sigma_cp_deg=1.0)
        fitres, cov = vb.fit_scene(s, vb.load(path))
        got, tr = vb.flat_values(s, fitres.values), vb.flat_truth(s)
        sig = np.sqrt(np.diag(cov)) if cov is not None else np.full(tr.size, np.nan)
        names = [p for p in s.truth for _ in np.atleast_1d(s.truth[p])]
        for p, t, g, e in zip(names, tr, got, sig):
            b = "— (see finding 5)" if np.isnan(e) else f"{(g - t) / e:+.3f}"
            out(f"| {s.name} | `{p}` | {t:.4g} | {g:.4g} | {b} |")
    out()

    s = scenes[3]
    _, _, sci, cal = res[(s.name, 256)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ext = np.array([-1, 1, -1, 1]) * 128 * 30.0 / 1000
    for ax, im, title in [(axes[0], sci[0], "dLux"), (axes[1], sci[1], "closed form (Airy × fringes)")]:
        ax.imshow(np.sqrt(im / im.max()), origin="lower", extent=ext, cmap="magma")
        ax.set_title(f"star + modulated rim: {title}", fontsize=9)
        ax.set_xlabel("x (arcsec)")
    d = sci[0] / sci[0].sum() - sci[1] / sci[1].sum()
    m = axes[2].imshow(d / (sci[1] / sci[1].sum()).max(), origin="lower", extent=ext, cmap="RdBu_r")
    axes[2].set_title("difference / peak", fontsize=9)
    fig.colorbar(m, ax=axes[2])
    fig.tight_layout()
    fig.savefig(DOCS / "nrm_images.png", dpi=120)
    out("![masking images](nrm_images.png)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pulls", type=int, default=200)
    args = ap.parse_args()
    t = time.time()
    out("# Results\n")
    out("Generated by `scripts/report.py`; all virgil evaluations in float64.")
    try:
        from importlib.metadata import version

        out(f"External packages: PMOIRED {version('pmoired')} (conventions in")
        out("[the PMOIRED page](method/pmoired.md)).\n")
    except Exception:
        out("External packages: PMOIRED not installed.\n")
    parity()
    vlti(args.pulls)
    masking()
    out(f"_Run time {time.time() - t:.0f} s._")
    (DOCS / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
