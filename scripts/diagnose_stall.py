"""Why do some contest image fits stall? (Stage C2 diagnostic.)

    python scripts/diagnose_stall.py --case N [--data DIR] [--out DIR]
    python scripts/diagnose_stall.py --list

On OzSTAR (JAX 0.11.2, the latest, as on the laptop), several image fits in
scripts/contest_images.py stopped with "line search ran out of float64
precision". Their χ² then stayed the same at every regularisation weight,
and the strongly regularised image was a scatter of single pixels: for
2022 GRAVITY, χ²/N = 462 after 2560 L-BFGS steps at w = 1e4. Each case
here takes one such fit, at the first (strongest) weight of its L-curve,
and asks three questions:

1. How does it fail? It refits from the start with max_steps = 10, 30, ...,
   recording χ²/N, the image's effective number of pixels (exp of its
   entropy) and its "dead" pixels (brightness < 1e-12 of the peak, whose
   log-brightness gradient is then ~0).
2. Are the gradients right? At the stalled point, the gradient of the
   loss the fitter minimises (½χ² + the regulariser) with respect to the
   log-brightness pixels, from JAX, is compared with central finite
   differences along random directions. A scan of the loss along the
   negative gradient shows whether any descent is left.
3. What fixes it? It refits with a smaller step cap (max_step_size 0.2 and
   0.05), with Adam, and, with a star, with the image flux held at its
   start.

Outputs: <label>_diag.txt (all numbers) and <label>_diag.png (images).
Heavy: run on OzSTAR (ozstar_scripts job contest_imaging, diagnose.sbatch).
"""

import argparse
import pathlib
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from contest_images import WEIGHTS, image_of, setup  # noqa: E402

from virgil.fitting import fit  # noqa: E402
from virgil.imaging import MaxEntropy  # noqa: E402
from virgil.likelihood import whitened_residuals  # noqa: E402
from virgil.plotting import plot_model  # noqa: E402

# (task in contest_images.TASKS, halo): the fits that stalled.
CASES = [(10, False), (15, False), (9, False), (2, True)]
SNAPSHOTS = (10, 30, 100, 300, 1000, 3000, 20000)


def image_stats(model, star, halo):
    b = np.asarray(_brightness(model, star, halo))
    b = b / b.sum()
    nz = b[b > 0]
    return {
        "eff_pixels": float(np.exp(-np.sum(nz * np.log(nz)))),
        "dead_frac": float(np.mean(b < 1e-12 * b.max())),
        "peak_frac": float(b.max()),
    }


def _brightness(model, star, halo):
    img = image_of(model, star, halo)
    return jax.nn.softmax(jnp.ravel(img.log_brightness))


def pixel_loss(model, data, regs, star, halo):
    """The fitter's loss (½χ² + regularisers; the priors are flat) as a
    function of the log-brightness pixels alone."""
    where = (lambda m: m.env.log_brightness) if (star or halo) else (lambda m: m.log_brightness)

    def loss(x):
        m = eqx.tree_at(where, model, x)
        r = whitened_residuals(m, data)
        return 0.5 * jnp.sum(r**2) + sum(reg.value(m) for reg in regs)

    return loss, where(model)


def summary(r, s):
    chi2 = float(np.sum(r.info["chi2_red"]))
    return {"chi2_red": chi2, "steps": int(r.info.get("steps", -1)), "converged": bool(r.info.get("converged", False)),
            **image_stats(r.model, s["star"], s["halo"])}


def run(case, data_dir, out_dir, smoke=False):
    task, halo = CASES[case]
    t0 = time.time()
    s = setup(task, data_dir, halo)
    w = float(WEIGHTS[0])
    regs = [MaxEntropy(w, path=s["path"]), *s["others"]]
    lines = [f"case={case} task={task} label={s['label']} weight={w:g} jax={jax.__version__} points={s['npts']} npix={s['npix']}"]
    shown = {}

    # 1. How it fails.
    lines.append("\n# 1. snapshots: max_steps -> chi2/N, steps, converged, effective pixels, dead fraction, peak fraction")
    last = None
    for k in (5, 10) if smoke else SNAPSHOTS:
        r = fit(s["start"], s["priors"], s["data"], regs, max_steps=k)
        st = summary(r, s)
        lines.append(f"max_steps={k:6d}  " + "  ".join(f"{a}={b:.4g}" if isinstance(b, float) else f"{a}={b}" for a, b in st.items()))
        shown[f"{k} steps"] = (r.model, st)
        last = r

    # 2. Gradients at the stalled point.
    loss, x = pixel_loss(last.model, s["data"], regs, s["star"], s["halo"])
    g = np.asarray(jax.grad(loss)(x)).ravel()
    lines.append(f"\n# 2. at the stalled point: loss={float(loss(x)):.10g}  |grad|={np.linalg.norm(g):.4g}  max|grad|={np.abs(g).max():.4g}")
    b = np.asarray(_brightness(last.model, s["star"], s["halo"]))
    dead = b < 1e-12 * b.max()
    lines.append(f"gradient norm on dead pixels / total: {np.linalg.norm(g[dead]) / np.linalg.norm(g):.3g} ({dead.sum()} dead of {dead.size})")
    rng = np.random.default_rng(1)
    for i in range(3):
        d = rng.normal(size=x.shape)
        d /= np.linalg.norm(d)
        ad = float(np.dot(g, d.ravel()))
        for eps in (1e-3, 1e-5):
            fd = float((loss(x + eps * d) - loss(x - eps * d)) / (2 * eps))
            lines.append(f"direction {i} eps={eps:g}: AD {ad:.8g}  FD {fd:.8g}  rel.diff {abs(ad - fd) / max(abs(fd), 1e-300):.2e}")
    gdir = -g.reshape(x.shape) / np.linalg.norm(g)
    l0 = float(loss(x))
    scan = [(t, float(loss(x + t * gdir)) - l0) for t in np.logspace(-8, 1, 10)]
    lines.append("loss change along -grad (step, Δloss): " + "  ".join(f"({t:.0e}, {dl:.4g})" for t, dl in scan))

    # 3. Remedies, each a full fit at the same weight from the same start.
    lines.append("\n# 3. remedies (full fits at the same weight)")
    remedies = {
        "max_step_size=0.2": dict(max_step_size=0.2),
        "max_step_size=0.05": dict(max_step_size=0.05),
        "adam lr=0.01, 20000": dict(method="adam", learning_rate=0.01, max_steps=20000),
    }
    cap = {"max_steps": 10} if smoke else {}
    for name, opts in remedies.items():
        r = fit(s["start"], s["priors"], s["data"], regs, **(opts | cap))
        st = summary(r, s)
        lines.append(f"{name:22s} " + "  ".join(f"{a}={b:.4g}" if isinstance(b, float) else f"{a}={b}" for a, b in st.items()))
        shown[name] = (r.model, st)
    if s["star"]:
        priors = {k: v for k, v in s["priors"].items() if k != "env.flux"}
        r = fit(s["start"], priors, s["data"], regs, **cap)
        st = summary(r, s)
        lines.append(f"{'env.flux fixed':22s} " + "  ".join(f"{a}={b:.4g}" if isinstance(b, float) else f"{a}={b}" for a, b in st.items()))
        shown["env.flux fixed"] = (r.model, st)
    lines.append(f"\nelapsed={time.time() - t0:.0f}s")

    out_dir.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    (out_dir / f"{s['label']}_diag.txt").write_text(text)
    print(text)

    n = len(shown)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4.2 * ((n + 1) // 2), 8.4))
    for ax, (name, (m, st)) in zip(axes.ravel(), shown.items()):
        plot_model(image_of(m, s["star"], s["halo"]), fov_mas=s["fov"], npix=s["npix"], ax=ax,
                   title=f"{name}\nχ²/N {st['chi2_red']:.3g}, {st['eff_pixels']:.0f} eff. px")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_dir / f"{s['label']}_diag.png", dpi=100)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", type=int)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--data", default="~/data/imaging_contests")
    parser.add_argument("--out", default="contest_images")
    parser.add_argument("--smoke", action="store_true", help="a few steps only: check that it runs")
    args = parser.parse_args()
    if args.list:
        for i, (task, halo) in enumerate(CASES):
            print(f"{i}  task {task}  halo={halo}")
        return
    run(args.case, pathlib.Path(args.data).expanduser(), pathlib.Path(args.out).expanduser(), args.smoke)


if __name__ == "__main__":
    main()
