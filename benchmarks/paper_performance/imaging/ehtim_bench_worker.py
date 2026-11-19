"""eht-imaging side of the imaging comparison. Runs in eht-imaging's own environment (``EHTIM_PYTHON``); never imports virgil.

It imports ``_imager`` and ``objective`` from the vv bridge worker (``src/external_bridge/ehtim_worker.py``) by path and
adds only timing, budget (``maxit``) control and the documented recipe of ``native``. The bridge worker is not edited.

    python ehtim_bench_worker.py --task-file task.json result.json    # utility tasks: ``triangles``, ``gate``
    python ehtim_bench_worker.py --worker '<json>'                     # one timed process, one BENCH_INNER line

Matched settings (plan section 6): ``ttype="direct"``, ``norm_reg=False``, ``transform=["log"]``, data terms amplitude +
closure phase with ``alpha_d = N_d / 2``, regulariser ``tv2`` (beta = 7.5e5, the same on every grid) and a soft ``flux``
term at weight 1e6, ``debias=False``, SciPy L-BFGS-B with ehtim's own ``maxcor=50``, ``maxls=40``.
Timing note: an eht-imaging run is timed from building the Obsdata and Imager from the rows to the returned image (the
Imager's own set-up, ``init_imager``, is inside ``make_image``); virgil's run is timed from ``fit`` to the returned model.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import make_data  # noqa: E402

BETA = 7.5e5
FLUX_WEIGHT = 1e6
BRIDGE = HERE.parents[2] / "src" / "external_bridge" / "ehtim_worker.py"
MAS = make_data.MAS


def _bridge():
    spec = importlib.util.spec_from_file_location("ehtim_bridge_worker", BRIDGE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _task(d, pb_npix, init, **kw):
    """The bridge worker's task dict for a dataset ``d`` (npz arrays), grid ``pb_npix`` and init image."""
    rows = d.get("_rows") or make_data.ehtim_rows(d)
    return {"rows": rows, "pdim": make_data.pixel_mas(pb_npix) * MAS, "init": np.asarray(init).tolist(),
            "alpha_amp": d["amp"].size / 2, "alpha_cphase": d["cphase_deg"].size / 2, "beta_tv2": BETA,
            "maxit": 200, "stop": 1e-6, **kw}


def triangles(task):
    """Station triples of the closure triangles eht-imaging forms (``c_phases(count="min")``) at the first time."""
    br = _bridge()
    rows = task["rows"]
    t0 = rows[0][0]
    sub = [r for r in rows if r[0] == t0]
    task2 = {"rows": sub, "pdim": 1e-9, "init": [[1.0]], "alpha_amp": 1.0, "alpha_cphase": 1.0, "beta_tv2": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = br._imager({**task2, "init": np.ones((4, 4)).tolist()}, np.ones((4, 4)), 0.0)
        cp = obs.c_phases(count="min")
    tri = {tuple(sorted(int(str(s)[1:]) for s in (r["t1"], r["t2"], r["t3"]))) for r in cp}
    return {"triangles": [list(t) for t in sorted(tri)]}


def gate(task):
    """eht-imaging's objective (flux term excluded) at the common start and at the rendered truth, on one grid."""
    br = _bridge()
    case = task["case"]
    dataset, npix = make_data.CASES[case]
    d = make_data.load(dataset)
    pixel = make_data.pixel_mas(npix)
    scene = make_data.DATASETS[dataset]["scene"]
    images = [make_data.start_image(npix, pixel), make_data.truth_image(scene, npix, pixel)]
    t = _task(d, npix, images[0], flux_weight=0.0)
    t["images"] = [im.tolist() for im in images]
    res = br.objective(t)
    out = {"case": case, "n_amp": res["n_amp"], "n_cphase": res["n_cphase"], "points": {}}
    na, nc = d["amp"].size, d["cphase_deg"].size
    for name, r in zip(("start", "truth"), res["results"]):
        # cost = sum alpha_d (chi2_d - 1) + beta tv2: add the constants back to compare with J
        out["points"][name] = {"cost": r["cost"], "J": r["cost"] + na / 2 + nc / 2, "tv2": r["tv2"]}
    out["n_amp_file"], out["n_cphase_file"] = int(na), int(nc)
    return out


class _Counter:
    def __init__(self):
        self.f = self.g = 0


def _make_runner(br, d, npix, start, budget, stop, counter, flux_weight=FLUX_WEIGHT):
    """Build the Imager (timed with the run) and return it ready for ``make_image_I``."""
    task = _task(d, npix, start, maxit=int(budget), stop=float(stop))
    with contextlib.redirect_stdout(io.StringIO()):
        obs, imager = br._imager(task, start, flux_weight)
    if counter is not None:
        f, g = imager.objfunc, imager.objgrad

        def cf(x):
            counter.f += 1
            return f(x)

        def cg(x):
            counter.g += 1
            return g(x)

        imager.objfunc, imager.objgrad = cf, cg
    return imager


def _one_run(br, d, npix, start, budget, stop):
    counter = _Counter()
    t0 = time.perf_counter()
    imager = _make_runner(br, d, npix, start, budget, stop, counter)
    with contextlib.redirect_stdout(io.StringIO()):
        imager.make_image_I(grads=True, show_updates=False)
    img = np.array(imager.out_last().imarr())
    # drop the counting wrappers: they reference the bound methods of ``imager`` (a reference cycle that would keep the
    # Imager and its Fourier matrices alive until the cyclic collector runs)
    imager.__dict__.pop("objfunc", None)
    imager.__dict__.pop("objgrad", None)
    dt = time.perf_counter() - t0
    return img, dt, counter


def native_recipe(br, d, npix, start, spec):
    """The documented recipe of ``examples/example_im_closure.py`` (eht-imaging v1.3.2, c8c7a89): three rounds, each
    re-initialised from the previous result blurred by the nominal resolution, data weights amp 50 / cphase 100 (then 10
    / 20), regularisers ``simple`` (round 1) then ``tv`` (rounds 2-3), ``flux`` 100 and ``cm`` 50, ``maxit=500`` (the
    example's ``maxi_next = 1000`` is a misspelt attribute and has no effect); ``ttype="direct"``, Gaussian prior = start."""
    task = _task(d, npix, start, maxit=int(spec["maxit"]), stop=1e-12)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, imgr = br._imager(task, start, spec["flux"])
        res = obs.res()  # nominal resolution 1 / longest baseline (radians), as in the example
        imgr.dat_term_next = {"amp": spec["amp"][0], "cphase": spec["cphase"][0]}
        imgr.reg_term_next = {"simple": spec["simple"], "flux": spec["flux"], "cm": spec["cm"]}
        imgr.make_image(show_updates=False)
        out = imgr.out_last()
        imgr.init_next = out.blur_circ(res)
        imgr.prior_next = imgr.init_next
        imgr.dat_term_next = {"amp": spec["amp"][0], "cphase": spec["cphase"][0]}
        imgr.reg_term_next = {"tv": spec["tv"], "flux": spec["flux"], "cm": spec["cm"]}
        imgr.make_image(show_updates=False)
        out = imgr.out_last()
        imgr.init_next = out.blur_circ(res / 2)
        imgr.prior_next = imgr.init_next
        imgr.dat_term_next = {"amp": spec["amp"][1], "cphase": spec["cphase"][1]}
        imgr.reg_term_next = {"tv": spec["tv"], "flux": spec["flux"], "cm": spec["cm"]}
        imgr.make_image(show_updates=False)
        out = imgr.out_last()
    return np.array(out.imarr())


def bench(args):
    sw = common.Stopwatch()
    from calib import calibrate

    with sw("t_calib"):
        calibrate()
    with sw("t_import"):
        import ehtim  # noqa: F401

        br = _bridge()
    case, task_name, reps = args["case"], args["task"], int(args["reps"])
    dataset, npix = make_data.CASES[case]
    pixel = make_data.pixel_mas(npix)
    with sw("t_load"):
        d = make_data.load(dataset)
        d["_rows"] = make_data.ehtim_rows(d)
        start = make_data.start_image(npix, pixel)
    images, accuracy, iterations = [], {}, {}

    if task_name == "ladder":
        budgets, stop = args["budgets"], args["stop"]
        ladder = []
        with sw("t_cold"):
            img, _, _ = _one_run(br, d, npix, start, budgets[0], stop)
        for budget in budgets:
            times = []
            for _ in range(reps):
                img, dt, counter = _one_run(br, d, npix, start, budget, stop)
                times.append(dt)
            images.append(img)
            ladder.append({"budget": budget, "t_warm": times, "n_objfunc": counter.f, "n_objgrad": counter.g,
                           "flux": float(img.sum())})
            if budget == budgets[0]:
                sw.lists["t_warm"] = times
        accuracy["ladder"] = ladder
        iterations = {"objfunc_by_budget": [x["n_objfunc"] for x in ladder]}
    elif task_name in ("stop_default", "stop_tight"):
        with sw("t_cold"):
            img, _, counter = _one_run(br, d, npix, start, args["maxit"], args["stop"])
        for _ in range(reps):
            t0 = time.perf_counter()
            img, _, counter = _one_run(br, d, npix, start, args["maxit"], args["stop"])
            sw.lists.setdefault("t_warm", []).append(time.perf_counter() - t0)
        images.append(img)
        iterations = {"objfunc": counter.f, "objgrad": counter.g}
        accuracy["flux"] = float(img.sum())
    elif task_name == "native":
        spec = args["native"]
        with sw("t_cold"):
            img = native_recipe(br, d, npix, start, spec)
        for _ in range(reps):
            with sw("t_warm", repeat=True):
                img = native_recipe(br, d, npix, start, spec)
        images.append(img)
        accuracy["flux"] = float(img.sum())
    elif task_name == "per_eval":
        calls = int(args.get("calls", 50))
        task = _task(d, npix, start, flux_weight=FLUX_WEIGHT)
        with contextlib.redirect_stdout(io.StringIO()):
            obs, imager = br._imager(task, start, FLUX_WEIGHT)
            imager.check_params()
            imager.check_limits()
            imager.init_imager()
        x = np.log(start.ravel())
        with sw("t_cold"):
            imager.objfunc(x)
            imager.objgrad(x)
        for _ in range(reps):
            t0 = time.perf_counter()
            for _ in range(calls):
                f = imager.objfunc(x)
                imager.objgrad(x)
            sw.lists.setdefault("t_warm", []).append((time.perf_counter() - t0) / calls)
        accuracy.update(per_call_calls=calls, value_at_start=float(f))
        iterations = {"calls": calls}
    else:
        raise ValueError(task_name)

    out_npz = None
    if images:
        out_npz = args["images_file"]
        np.savez(out_npz, **{f"img_{i}": im for i, im in enumerate(images)})
    common.emit_inner(sw, n_data=int(d["amp"].size + d["cphase_deg"].size), n_param=int(npix * npix),
                      iterations=iterations, accuracy=accuracy,
                      settings={k: v for k, v in args.items() if k != "images_file"},
                      provenance=common.provenance(data_manifest=common.manifest_hash(make_data.MANIFEST)),
                      images_file=out_npz)


if __name__ == "__main__":
    if sys.argv[1] == "--task-file":
        task = json.loads(Path(sys.argv[2]).read_text())
        fn = {"triangles": triangles, "gate": gate}[task["task"]]
        Path(sys.argv[3]).write_text(json.dumps(fn(task)))
    elif sys.argv[1] == "--worker":
        a = json.loads(sys.argv[2])
        if a["task"] == "gate":
            print(common.INNER_TAG + json.dumps({"gate": gate(a)}))
        else:
            bench(a)
    else:
        raise SystemExit(__doc__)
