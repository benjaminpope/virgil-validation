"""virgil side of the imaging comparison. Never imports eht-imaging (plan section 1, independence).

Driver (BENCH_PYTHON; launches one fresh worker process per row and scores its images with the neutral scorer):

    python run_virgil.py gate   --case I-sim-24
    python run_virgil.py run    --case I-sim-24 --task ladder --config matched [--size tiny] [--threads 1]
    tasks: ladder | per_eval | stop_default | stop_tight | native      (see imaging/cases.yml)

Worker (a fresh process under ``/usr/bin/time -l``): ``python run_virgil.py --worker '<json>'``. The conventions are
those of ``docs/method/ehtim.md``: the same data term and the same ``TSV`` weight mapping as the vv test
``tests/test_ehtim_reconstruction.py`` (read, not imported). Exact direct Fourier sum (``ft="dft"``), float64, CPU.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import make_data  # noqa: E402

BEAM_MAS = 3.0     # the arbitrary beam of the TSV mapping (p / B only rescales the weight; the product is exact)
BETA = 7.5e5       # the validated WEIGHT_HI, the same value on every grid (plan section 6)


def _tsv(vi, weight, pixel_mas):
    """virgil's ``TSV`` equal to eht-imaging's unnormalised ``tv2`` (+ edge terms) at beta = ``weight``: virgil divides by
    F^2 (p/B)^4 with F = 1 (softmax), so multiply by it."""
    return vi.TSV(weight * (pixel_mas / BEAM_MAS) ** 4, beam_mas=BEAM_MAS)


class Problem:
    """Everything virgil needs for one case; built inside the worker (its cost is ``t_load``)."""

    def __init__(self, case):
        import jax

        jax.config.update("jax_enable_x64", True)  # float64 everywhere (plan section 3)
        import numpy as np
        import virgil.imaging as vi
        import virgil.models as vm
        from virgil.likelihood import whitened_residuals
        from virgil.oidata import OIData

        self.np, self.vi, self.vm, self.whitened_residuals = np, vi, vm, whitened_residuals
        dataset, self.npix = make_data.CASES[case]
        self.pixel = make_data.pixel_mas(self.npix)
        self.data = OIData(str(make_data.fits_path(dataset)))
        self.start = make_data.start_image(self.npix, self.pixel)
        self.template = self.image(self.start)
        self.priors = vi.image_priors(self.template)
        self.tsv = _tsv(vi, BETA, self.pixel)

    def image(self, brightness):
        np = self.np
        return self.vm.Image(np.log(np.maximum(brightness, 1e-300)), self.pixel, ft="dft")

    def loss_at(self, brightness):
        """virgil's loss ``1/2 sum r^2 + TSV`` at an image (any flux: normalised first)."""
        np = self.np
        b = np.asarray(brightness, float)
        image = self.image(b / b.sum())
        r = np.asarray(self.whitened_residuals(image, self.data))
        return {"loss": 0.5 * float(np.sum(r**2)) + float(self.tsv.value(image)), "n": int(r.size)}

    def fit(self, budget, method, gtol, memory):
        kw = {} if memory is None else {"lbfgs_memory": memory}
        method = None if method == "default" else method  # fit's own default method
        res = _fit(self.template, self.priors, self.data, [self.tsv], method=method, max_steps=int(budget), gtol=gtol, **kw)
        return res


def _fit(*a, **k):
    from virgil.fitting import fit

    return fit(*a, **k)


def gate_worker(args):
    """Loss values at the common start and at the rendered truth (the neutral scorer compares them)."""
    import numpy as np

    pb = Problem(args["case"])
    dataset, npix = make_data.CASES[args["case"]]
    scene = make_data.DATASETS[dataset]["scene"]
    out = {"case": args["case"], "points": {}}
    for name, img in (("start", pb.start), ("truth", make_data.truth_image(scene, npix, pb.pixel))):
        out["points"][name] = pb.loss_at(img)
    return out


def bench_worker(args):
    """One timed process: calibration, import, load, cold call, warm repeats; images go to an npz for scoring."""
    sw = common.Stopwatch()
    sys.path.insert(0, str(HERE.parent))
    from calib import calibrate

    with sw("t_calib"):
        calibrate()
    with sw("t_import"):
        import jax

        jax.config.update("jax_enable_x64", True)
        import virgil.fitting  # noqa: F401
        import virgil.imaging  # noqa: F401
        import virgil.models  # noqa: F401
        import numpy as np

        common.enable_jax_cache()  # after the virgil import: uses virgil.enable_compilation_cache
    with sw("t_load"):
        pb = Problem(args["case"])
    task, reps = args["task"], int(args["reps"])
    images, accuracy, iterations = [], {}, {}
    method, memory = args.get("method", "lbfgsb"), args.get("memory")

    if task == "ladder":
        budgets = args["budgets"]
        ladder = []
        gtol = float(args["gtol"])
        # cold: the first call of the first budget in this process (includes JIT)
        with sw("t_cold"):
            res = pb.fit(budgets[0], method, gtol, memory)
            common.block(res.model.brightness)
        for budget in budgets:
            times = []
            for _ in range(reps):
                t0 = time.perf_counter()
                res = pb.fit(budget, method, gtol, memory)
                common.block(res.model.brightness)
                times.append(time.perf_counter() - t0)
            images.append(np.asarray(res.model.brightness))
            ladder.append({"budget": budget, "t_warm": times, "steps": int(res.info["steps"]),
                           "converged": bool(res.info["converged"]), "stop": res.info.get("stop")})
            if budget == budgets[0]:
                sw.lists["t_warm"] = times
        accuracy["ladder"] = ladder
        iterations = {"steps_by_budget": [x["steps"] for x in ladder]}
    elif task in ("stop_default", "stop_tight", "native"):
        if task == "native":
            res, extra = native_fit(pb, args, sw, reps, images)
            accuracy.update(extra)
            iterations = {"steps": extra.get("steps")}
        else:
            kw = {"method": method, "gtol": float(args["gtol"]), "memory": memory}
            max_steps = args.get("max_steps") or 20000  # virgil's L-BFGS-B default limit
            with sw("t_cold"):
                res = pb.fit(max_steps, kw["method"], kw["gtol"], memory)
                common.block(res.model.brightness)
            for _ in range(reps):
                with sw("t_warm", repeat=True):
                    res = pb.fit(max_steps, kw["method"], kw["gtol"], memory)
                    common.block(res.model.brightness)
            images.append(np.asarray(res.model.brightness))
            iterations = {"steps": int(res.info["steps"])}
            accuracy.update(converged=bool(res.info["converged"]), stop=res.info.get("stop"),
                            loss=float(res.info["loss"]))
    elif task == "per_eval":
        import jax
        import jax.numpy as jnp

        calls = int(args.get("calls", 50))
        z0 = jnp.log(jnp.maximum(jnp.asarray(pb.start), 1e-300))

        def loss(z):
            image = pb.vm.Image(z, pb.pixel, ft="dft")
            return 0.5 * jnp.sum(pb.whitened_residuals(image, pb.data) ** 2) + pb.tsv.value(image)

        vg = jax.jit(jax.value_and_grad(loss))
        with sw("t_cold"):
            common.block(vg(z0))
        for _ in range(reps):
            t0 = time.perf_counter()
            for _ in range(calls):
                out = vg(z0)
            common.block(out)
            sw.lists.setdefault("t_warm", []).append((time.perf_counter() - t0) / calls)
        accuracy.update(per_call_calls=calls, value_at_start=float(out[0]))
        iterations = {"calls": calls}
    else:
        raise ValueError(task)

    n_data = int(pb.data.vis.size + pb.data.phi.size)
    n_param = int(pb.npix**2)
    out_npz = None
    if images:
        out_npz = args["images_file"]
        np.savez(out_npz, **{f"img_{i}": im for i, im in enumerate(images)})
    common.emit_inner(sw, n_data=n_data, n_param=n_param, iterations=iterations, accuracy=accuracy,
                      settings={k: v for k, v in args.items() if k != "images_file"},
                      provenance=common.provenance(data_manifest=common.manifest_hash(make_data.MANIFEST)),
                      images_file=out_npz)


def native_fit(pb, args, sw, reps, images):
    """virgil's documented RML route of ``docs/imaging_rml.md``: ``starting_image``, ``image_priors``, ``fit`` with a
    maximum-entropy regulariser, default method and tolerances, no weight tuning (see cases.yml)."""
    import numpy as np

    vi, vm = pb.vi, pb.vm
    spec = args["native"]

    def once():
        start = vi.starting_image(pb.data, star=False, largest_mas=make_data.FIELD_MAS, start=spec["start"])
        res = _fit(start, vi.image_priors(start), pb.data, [vi.MaxEntropy(float(spec["weight"]))])
        return start, res

    with sw("t_cold"):
        start, res = once()
        common.block(res.model.brightness)
    for _ in range(reps):
        with sw("t_warm", repeat=True):
            start, res = once()
            common.block(res.model.brightness)
    images.append(np.asarray(res.model.brightness))
    return res, {"steps": int(res.info["steps"]), "converged": bool(res.info["converged"]),
                 "grid_npix": list(np.shape(res.model.brightness)), "grid_pixel_mas": float(start.pixel_scale_mas)}


# ----------------------------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        a = json.loads(sys.argv[2])
        if a["task"] == "gate":
            print(common.INNER_TAG + json.dumps({"gate": gate_worker(a)}))
        else:
            bench_worker(a)
    else:
        import compare

        compare.main("virgil", sys.argv[1:])
