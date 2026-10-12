"""False-alarm and contrast calibration of virgil.detection (Stage 8).

virgil's detection statistics (detection_statistics: delta_chi2,
log_bayes_factor, max_snr) and its Monte Carlo (injection_recovery,
DetectionMC) are checked against simulations that are ours, not virgil's:
crosscheck.simulate.observe writes a four-UT VLTI file (correlated closure
phases), with or without a companion, and virgil searches it.

Parts, each an OzSTAR array (``run --part P --task T --draws N``):

* ``null``: companion-free files from our simulator. Records the grid
  statistics, and delta_chi2 at one fixed position.
* ``inject``: a companion at a random position angle, at separations SEPS
  and fluxes FLUXES. Records the grid statistics, delta_chi2 at the true
  position, and lambda, our own noiseless chi-squared of the injected data
  under the null model (the noncentrality).
* ``virgil``: virgil's own null simulations (injection_recovery with
  gaussian_null) of the same template, saved as a DetectionMC.

Version 3 (design/detection_v3.yml, rationale design/detection_v3.md) adds:

* ``proxy``: the linear reference statistic S (crosscheck.tails) of draws of
  either simulator (``--sim ours|virgil``), millisecond-cheap, for the tail of
  the noise itself at 1e-2 and 1e-3 with 10^6 draws;
* ``null_is`` and ``virgil_is`` (``--threshold NAME``): importance-sampling
  draws of delta_chi2 for the registered threshold, from our simulator and
  from virgil's gaussian_null, each at the mean of a companion at a random
  search position, with the exact Gaussian weight (crosscheck.tails.Mixture);
* ``check``: virgil's detection_statistics on gaussian_null draws against S,
  on the device run (``--device cpu|default``); ``virgil`` also takes
  ``--device``, and a run on another device with the same task and draws
  repeats the same keys, so that devices are compared draw by draw.

``aggregate`` applies the criteria registered in design/detection_v3.yml (its
SHA-256 is recorded in the summary): the tail in its own family (equivalence
of the exceedance-rate ratio within a margin), the bulk and the boundary in a
secondary family, and must-hold checks. Version 2's criteria (design/
detection_v2.yml, summary trust/campaigns/detection_mc_v2.json) are kept and
superseded.
"""

import argparse
import hashlib
import json
import pathlib
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

UTS4 = np.array([[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]])
OBS = dict(hour_angles_h=np.linspace(-3, 3, 5), wavelengths=np.linspace(1.5e-6, 2.4e-6, 4), dec_deg=-50.0,
           sigma_v2=0.01, sigma_cp_deg=0.5)
GRID = {"dra": np.linspace(-20.0, 20.0, 17), "ddec": np.linspace(-20.0, 20.0, 17),
        "flux": np.geomspace(1e-5, 0.1, 40)}
FIXED = (6.0, -4.0)
SEPS = np.array([5.0, 10.0, 15.0])
FLUXES = np.geomspace(3e-4, 3e-3, 4)  # lambda ~ 1-80: near the detection threshold

CRITERIA_V2 = ROOT / "design" / "detection_v2.yml"
CRITERIA_FILE = ROOT / "design" / "detection_v3.yml"
VIRGIL_CHUNK = 1000  # virgil's draws per saved chunk, so a stopped task resumes


def criteria(path=CRITERIA_FILE):
    import yaml

    return yaml.safe_load(pathlib.Path(path).read_text())


def criteria_hash(path=CRITERIA_FILE):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def design_hash():
    """The experiment, apart from the criteria: the array, grid, injections
    and draw code here, and the sources of the simulator and reference
    chi-squared they use. Rows are combined only when it matches."""
    import inspect

    config = {"UTS4": UTS4.tolist(), "OBS": {k: np.asarray(v).tolist() for k, v in OBS.items()},
              "GRID": {k: v.tolist() for k, v in GRID.items()}, "FIXED": FIXED, "SEPS": SEPS.tolist(),
              "FLUXES": FLUXES.tolist(), "VIRGIL_CHUNK": VIRGIL_CHUNK}
    h = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for fn in (vis_fn, observe, stats_of, one_position, draw, reference, residual, draw_is, draw_virgil_is,
               draw_check, draw_proxy):
        h.update(inspect.getsource(fn).encode())
    for name in ("simulate.py", "sky.py", "array.py", "chi2.py", "tails.py"):
        h.update((ROOT / "src" / "crosscheck" / name).read_bytes())
    return h.hexdigest()[:16]


def vis_fn(f, x, y):
    from crosscheck import sky

    def vis(u, v, w):
        return (sky.vis_point(u, v, w) + f * sky.vis_point(u, v, w, x, y)) / (1 + f)

    return vis


def observe(path, f, x, y, rng):
    from crosscheck import simulate

    simulate.observe(path, vis_fn(f, x, y), UTS4, rng=rng, **OBS)


def virgil_commit():
    import os

    if os.environ.get("PIN_COMMIT"):
        return os.environ["PIN_COMMIT"]
    from evidence.meta import virgil_source

    return virgil_source()["commit"]


def stats_of(data, grid):
    import warnings

    from virgil import detection, models

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # flux-axis diagnostics on noise draws
        s = detection.detection_statistics(models.BinaryModelCartesian, data, grid)
    return {k: float(np.asarray(v)) for k, v in s.items()}


def one_position(x, y):
    return {"dra": np.array([x]), "ddec": np.array([y]), "flux": GRID["flux"]}


_REF = {}


def reference():
    """Our linear reference statistic on the template (crosscheck.tails): the
    noiseless data vector ``m0``, the precision matrix from our own covariance
    of the file's errors, the response of the data to a companion's flux at
    every search position, and the importance-sampling mixtures built so far.
    Built once per process (about 10 s)."""
    if not _REF:
        from crosscheck import chi2 as ours, simulate, tails

        tmp = pathlib.Path(tempfile.mkdtemp(prefix="detection_reference_"))

        def clean(f, x, y):
            v2, cp = simulate.observe(tmp / "clean.fits", vis_fn(f, x, y), UTS4, rng=None, **OBS)
            return np.concatenate([np.asarray(v2).ravel(), np.deg2rad(np.asarray(cp)).ravel()])

        observe(tmp / "template.fits", 0.0, 0.0, 0.0, None)
        w = tails.precision(ours.load(tmp / "template.fits"))
        positions = [(a, b) for a in GRID["dra"] for b in GRID["ddec"]]
        m0, g = tails.response(clean, positions)
        _REF.update(clean=clean, positions=positions, m0=m0, search=tails.Search(w, g), mixtures={})
    return _REF


def residual(d):
    """The data minus the noiseless null model, as one vector (V² then closure
    phases in radians): from a file loaded by crosscheck.chi2.load, or from a
    virgil OIData."""
    if isinstance(d, dict):
        v = np.concatenate([d["v2"].ravel(), d["cp"].ravel()])
    else:
        v = np.concatenate([np.asarray(d.vis), np.asarray(d.phi)])
    return v - reference()["m0"]


def mixture(name):
    from crosscheck import tails

    ref = reference()
    if name not in ref["mixtures"]:
        ref["mixtures"][name] = tails.Mixture(ref["search"], ref["positions"], ref["clean"],
                                              criteria()["thresholds"][name])
    return ref["mixtures"][name]


def draw(part, seed, tmp):
    from crosscheck import chi2 as ours
    from virgil.oidata import OIData

    rng = np.random.default_rng(seed)
    path = pathlib.Path(tmp) / f"{seed}.fits"
    out = {"seed": int(seed)}
    if part == "null":
        observe(path, 0.0, 0.0, 0.0, rng)
        data = OIData(str(path))
        out["grid"] = stats_of(data, GRID)
        out["fixed"] = stats_of(data, one_position(*FIXED))["delta_chi2"]
        out["proxy"] = float(reference()["search"].statistic(residual(ours.load(path)))[0])
    else:
        sep = float(rng.choice(SEPS))
        f = float(rng.choice(FLUXES))
        pa = float(rng.uniform(0.0, 360.0))
        x, y = sep * np.sin(np.deg2rad(pa)), sep * np.cos(np.deg2rad(pa))
        clean = pathlib.Path(tmp) / f"{seed}_clean.fits"
        observe(clean, f, x, y, None)  # noiseless: lambda is chi2 of the null model to these data
        lam = ours.chi2(ours.load(clean), vis_fn(0.0, 0.0, 0.0), correlated=True, sine=True)
        observe(path, f, x, y, rng)
        data = OIData(str(path))
        out.update(sep=sep, flux=f, pa=pa, dra=x, ddec=y, lam=float(lam))
        out["grid"] = stats_of(data, GRID)
        out["true"] = stats_of(data, one_position(x, y))["delta_chi2"]
    return out


def draw_is(seed, name, tmp):
    """One importance-sampling draw of our simulator for threshold ``name``:
    a component of the mixture, our file at that mean, virgil's delta_chi2,
    and the log weight."""
    from crosscheck import chi2 as ours
    from virgil.oidata import OIData

    rng = np.random.default_rng(seed)
    mix = mixture(name)
    c = mix.component(rng)
    x, y = mix.pos[c]
    path = pathlib.Path(tmp) / f"{seed}.fits"
    observe(path, float(mix.flux[c]), float(x), float(y), rng)
    r = residual(ours.load(path))
    return {"seed": int(seed), "comp": c, "logw": float(mix.log_weight(r)[0]),
            "delta_chi2": stats_of(OIData(str(path)), GRID)["delta_chi2"],
            "proxy": float(reference()["search"].statistic(r)[0])}


def draw_virgil_is(seed, name, simulator):
    """The same for virgil's gaussian_null: the scene is the mixture's
    component, the noise virgil's."""
    import jax

    from virgil import models

    rng = np.random.default_rng(seed)
    mix = mixture(name)
    c = mix.component(rng)
    x, y = mix.pos[c]
    data = simulator(jax.random.PRNGKey(int(seed)), models.BinaryModelCartesian(float(x), float(y), float(mix.flux[c])))
    r = residual(data)
    return {"seed": int(seed), "comp": c, "logw": float(mix.log_weight(r)[0]),
            "delta_chi2": stats_of(data, GRID)["delta_chi2"],
            "proxy": float(reference()["search"].statistic(r)[0])}


def draw_check(seed, simulator):
    """virgil's delta_chi2 on a draw of gaussian_null, and the linear statistic of the same draw."""
    import jax

    data = simulator(jax.random.PRNGKey(int(seed)))
    return {"seed": int(seed), "real": stats_of(data, GRID)["delta_chi2"],
            "proxy": float(reference()["search"].statistic(residual(data))[0])}


def draw_proxy(sim, seed0, n, batch=5000):
    """The linear statistic S of ``n`` draws of a simulator, ``ours`` (crosscheck.simulate.observe
    with its writer replaced by a capture, seeds seed0 + k) or ``virgil`` (gaussian_null, jitted, vmapped,
    keys PRNGKey(seed0 + k)). Returns a list of floats."""
    import jax

    from crosscheck import simulate

    search, m0 = reference()["search"], reference()["m0"]
    out = []
    if sim == "ours":
        grabbed = {}

        def capture(path, *, wavelengths, stations_xyz, vis2, t3):
            grabbed["v"] = np.concatenate([vis2["vis2"].ravel(), np.deg2rad(t3["phi"]).ravel()])

        real, simulate.oifits_writer.write = simulate.oifits_writer.write, capture
        try:
            for start in range(0, n, batch):
                rows = []
                for k in range(start, min(start + batch, n)):
                    simulate.observe("unused.fits", vis_fn(0.0, 0.0, 0.0), UTS4, rng=np.random.default_rng(seed0 + k), **OBS)
                    rows.append(grabbed["v"] - m0)
                out += search.statistic(np.array(rows)).tolist()
        finally:
            simulate.oifits_writer.write = real
        return out
    import jax.numpy as jnp

    from virgil import detection, models
    from virgil.oidata import OIData

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "template.fits"
        observe(path, 0.0, 0.0, 0.0, None)
        template = OIData(str(path))
    simulator = detection.gaussian_null(models.BinaryModelCartesian(0.0, 0.0, 0.0), template)
    a, m0_ = jnp.asarray(search.a), jnp.asarray(m0)

    @jax.jit
    def stat(keys):
        r = jax.vmap(lambda k: (lambda d: jnp.concatenate([d.vis, d.phi]))(simulator(k)))(keys) - m0_
        return (jnp.maximum(r @ a.T, 0.0) ** 2).max(axis=1)

    for start in range(0, n, batch):
        keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(seed0 + start, seed0 + min(start + batch, n)))
        out += np.asarray(stat(keys)).tolist()
    return out


SEED_OFFSET = {"null": 0, "inject": 10_000_000, "virgil": 20_000_000, "proxy_ours": 30_000_000,
               "proxy_virgil": 40_000_000, "check": 70_000_000, "null_is": 100_000_000, "virgil_is": 200_000_000}


def run(args):
    import jax

    jax.config.update("jax_enable_x64", True)
    cpu = jax.devices("cpu")[0]
    device = cpu if args.device == "cpu" else jax.devices()[0]
    with jax.default_device(device):
        _run(args, device.platform)


def _run(args, platform):
    import jax

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = virgil_commit()
    if not commit:
        raise SystemExit("cannot tell which virgil commit is running: set PIN_COMMIT")
    spec_ = criteria()
    names = list(spec_["thresholds"])
    if args.part in ("null_is", "virgil_is") and args.threshold not in names:
        raise SystemExit(f"--threshold must be one of {names}")
    offset = SEED_OFFSET[args.part] + (1_000_000 * names.index(args.threshold) if args.threshold else 0)
    # the other device repeats the cpu run's keys, so that the two are compared draw by draw
    seed0 = args.seed_base + offset + args.task * args.draws
    head = {"part": args.part, "task": args.task, "draws": args.draws, "seed0": seed0, "virgil_commit": commit,
            "criteria_hash": criteria_hash(), "design_hash": design_hash(), "device": platform,
            "threshold": args.threshold}
    tag = f"{args.part}_{args.threshold}_" if args.threshold else f"{args.part}_"
    if args.part == "check":
        tag = f"check_{platform}_"
    start = time.time()
    if args.part == "virgil":
        from virgil import detection, models
        from virgil.oidata import OIData

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "template.fits"
            observe(path, 0.0, 0.0, 0.0, None)
            template = OIData(str(path))
        meta = out_dir / f"virgil_{platform}_{args.task:04d}.json"
        chunks = []
        if meta.exists():  # resume: keep the chunks already saved
            old = json.loads(meta.read_text())
            if {k: old.get(k) for k in head} != head:
                raise SystemExit(f"{meta} was made differently: move it before rerunning")
            chunks = old["chunks"]
        n_chunks = -(-args.draws // VIRGIL_CHUNK)
        for c in range(n_chunks):
            name = f"virgil_{platform}_{args.task:04d}_{c:03d}.npz"
            if name in chunks:
                continue
            n = min(VIRGIL_CHUNK, args.draws - c * VIRGIL_CHUNK)
            # chunk c's key is seed0 + c * VIRGIL_CHUNK: distinct from every other chunk and task
            mc = detection.injection_recovery(models.BinaryModelCartesian, models.BinaryModelCartesian(0.0, 0.0, 0.0),
                                              template, GRID, seed0 + c * VIRGIL_CHUNK, n_null=n, progress=False)
            mc.save(out_dir / name)
            chunks.append(name)
            tmp_meta = meta.with_suffix(".tmp")
            tmp_meta.write_text(json.dumps({**head, "complete": len(chunks) == n_chunks, "chunks": chunks,
                                            "elapsed_s": time.time() - start}))
            tmp_meta.rename(meta)
            print(f"virgil task={args.task} chunk={c} draws={n} device={platform} elapsed={time.time() - start:.0f}s",
                  flush=True)
        return
    if args.part in ("proxy_ours", "proxy_virgil"):
        path = out_dir / f"{args.part}_{args.task:04d}.json"
        values = draw_proxy(args.part.split("_")[1], seed0, args.draws)
        path.write_text(json.dumps({**head, "complete": True, "values": values}))
        print(f"wrote {path} elapsed={time.time() - start:.0f}s")
        return
    path = out_dir / f"{tag}{args.task:04d}.json"
    rows = []
    if path.exists():  # resume after a time limit
        old = json.loads(path.read_text())
        if {k: old.get(k) for k in head} != head:
            raise SystemExit(f"{path} was made differently: move it before rerunning")
        rows = old["rows"]
    done = {r["seed"] for r in rows}
    simulator = None
    if args.part in ("virgil_is", "check"):
        from virgil import detection, models
        from virgil.oidata import OIData

        with tempfile.TemporaryDirectory() as tmp:
            template_path = pathlib.Path(tmp) / "template.fits"
            observe(template_path, 0.0, 0.0, 0.0, None)
            template = OIData(str(template_path))
        simulator = detection.gaussian_null(models.BinaryModelCartesian(0.0, 0.0, 0.0), template)
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(args.draws):
            seed = seed0 + k
            if seed in done:
                continue
            t0 = time.time()
            if args.part == "null_is":
                row = draw_is(seed, args.threshold, tmp)
            elif args.part == "virgil_is":
                row = draw_virgil_is(seed, args.threshold, simulator)
            elif args.part == "check":
                row = draw_check(seed, simulator)
            else:
                row = draw(args.part, seed, tmp)
            rows.append(row)
            tmp_path = path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps({**head, "complete": len(rows) == args.draws, "rows": rows}))
            tmp_path.rename(path)
            print(f"{args.part} seed={seed} elapsed={time.time() - t0:.2f}s", flush=True)
    print(f"wrote {path} elapsed={time.time() - start:.0f}s")


NOMINAL = {"1e-2": 1e-2, "1e-3": 1e-3, "1e-4": 1e-4, "3sigma": 1.349898031630093e-3, "5sigma": 2.866515718791939e-7}


def holm(pvalues, alpha):
    """Holm's step-down procedure: {name: rejected}."""
    names = sorted(pvalues, key=pvalues.get)
    reject, k = {}, len(names)
    for rank, name in enumerate(names):
        reject[name] = pvalues[name] <= alpha / (k - rank)
        if not reject[name]:
            for rest in names[rank + 1:]:
                reject[rest] = False
            break
    return reject


def ratio_test(ratio, log_se, margin):
    """The equivalence test of ln(ratio) within +-ln(margin), and the 95% interval."""
    from crosscheck import tails

    if not (np.isfinite(ratio) and ratio > 0 and np.isfinite(log_se)):
        return {"ratio": None, "log_se": None, "ci95": None, "p": 1.0}
    half = 1.959964 * log_se
    return {"ratio": float(ratio), "log_se": float(log_se), "ci95": [float(ratio * np.exp(-half)), float(ratio * np.exp(half))],
            "p": tails.equivalence_pvalue(ratio, log_se, margin)}


def plain_ratio(x_ours, x_virgil, t, margin):
    """Exceedance-rate ratio of two samples above t, plain Monte Carlo."""
    k1, k2 = int(np.sum(np.asarray(x_ours) >= t)), int(np.sum(np.asarray(x_virgil) >= t))
    n1, n2 = len(x_ours), len(x_virgil)
    out = {"ours": k1, "virgil": k2, "n_ours": n1, "n_virgil": n2}
    if k1 and k2:
        out.update(ratio_test((k1 / n1) / (k2 / n2), np.sqrt(1 / k1 + 1 / k2), margin))
    else:
        out.update(ratio=None, log_se=None, ci95=None, p=1.0)
    return out


def is_estimate(rows, t):
    from crosscheck import tails

    if len(rows) < 2:
        return None
    p, se, ess, mw, smw = tails.estimate([r["delta_chi2"] for r in rows], [r["logw"] for r in rows], t)
    return {"n": len(rows), "p": p, "se": se, "rel_se": se / p if p > 0 else None, "ess": ess, "mean_weight": mw,
            "mean_weight_se": smw}


def is_ratio(a, b, margin):
    if not a or not b or a["p"] <= 0 or b["p"] <= 0:
        return {"ratio": None, "log_se": None, "ci95": None, "p": 1.0}
    return ratio_test(a["p"] / b["p"], np.hypot(a["rel_se"], b["rel_se"]), margin)


def aggregate(args):
    from collections import defaultdict

    from scipy import stats

    from crosscheck import tails
    from virgil.detection import DetectionMC

    files = [f for d in args.dirs for f in sorted(pathlib.Path(d).glob("*_*.json"))]
    if len({f.resolve() for f in files}) != len(files):
        raise SystemExit("a results directory is given twice")
    if not files:
        raise SystemExit(f"no results found in {', '.join(map(str, args.dirs))}")
    rows = defaultdict(list)  # (part, threshold, device) -> rows
    values = {"proxy_ours": [], "proxy_virgil": []}
    mcs = defaultdict(dict)  # device -> {key: DetectionMC}
    commits, hashes, designs, seeds = set(), set(), set(), defaultdict(list)
    for f in files:
        r = json.loads(f.read_text())
        commits.add(r["virgil_commit"])
        hashes.add(r["criteria_hash"])
        designs.add(r.get("design_hash"))
        part, device = r["part"], r.get("device")
        if part == "virgil":
            for c in r["chunks"]:
                key = r["seed0"] + int(c.rsplit("_", 1)[1].split(".")[0]) * VIRGIL_CHUNK
                if key in mcs[device]:
                    raise SystemExit("virgil: repeated chunk keys")
                mcs[device][key] = DetectionMC.load(f.parent / c)
        elif part in values:
            values[part] += r["values"]
            seeds[(part, None)] += list(range(r["seed0"], r["seed0"] + len(r["values"])))
        else:
            rows[(part, r.get("threshold"), device if part == "check" else None)] += r["rows"]
            seeds[(part, r.get("threshold"))] += [x["seed"] for x in r["rows"]]
    if hashes != {criteria_hash()}:
        raise SystemExit(f"criteria changed since the runs: {hashes} vs {criteria_hash()}")
    if designs != {design_hash()}:
        raise SystemExit(f"the experiment changed since the runs (design {designs} vs {design_hash()})")
    if len(commits) != 1:
        raise SystemExit(f"the parts ran on different virgil commits: {commits}")
    for key, s_ in seeds.items():
        if len(set(s_)) != len(s_):
            raise SystemExit(f"{key[0]}: repeated seeds")
    spec = criteria()
    thr, counts = spec["thresholds"], spec["counts"]
    margin, prim = spec["primary"]["margin"], spec["primary"]
    out = {"version": 3, "virgil_commit": commits.pop(), "criteria_hash": criteria_hash(), "design_hash": design_hash(),
           "primary": {}, "pvalues": {}, "must_hold": {}, "reported": {}}
    null, inj = rows[("null", None, None)], rows[("inject", None, None)]
    mc = None
    if mcs.get("cpu"):
        mc = DetectionMC.concatenate([mcs["cpu"][k] for k in sorted(mcs["cpu"])])
    statistics = ["delta_chi2", "log_bayes_factor", "max_snr"]

    # --- primary family: the tail, equivalence of the exceedance-rate ratio, ours / virgil's
    tests = out["primary"]
    for name in ("1e-2", "1e-3"):
        a, b = values["proxy_ours"], values["proxy_virgil"]
        tests[f"noise_tail_{name}"] = plain_ratio(a, b, thr[name], margin) if a and b else {"p": 1.0}
    mine = [r["grid"]["delta_chi2"] for r in null]
    theirs = np.asarray(mc.null["delta_chi2"]) if mc is not None else np.array([])
    tests["e2e_tail_1e-2"] = plain_ratio(mine, theirs, thr["1e-2"], margin) if mine and theirs.size else {"p": 1.0}
    est = {}
    for sim in ("null_is", "virgil_is"):
        for name in thr:
            est[(sim, name)] = is_estimate(rows[(sim, name, None)], thr[name])
    tests["e2e_tail_1e-3"] = is_ratio(est[("null_is", "1e-3")], est[("virgil_is", "1e-3")], margin)
    p_primary = {k: v["p"] for k, v in tests.items()}
    primary_reject = holm(p_primary, prim["family_alpha"])
    out["rejected_primary"] = sorted(n for n, r in primary_reject.items() if not r)  # equivalence NOT shown
    out["primary_equivalent"] = {n: bool(r) for n, r in primary_reject.items()}

    # --- secondary family: the bulk and the boundary
    pv = out["pvalues"]
    fixed = np.array([r["fixed"] for r in null])
    zero = fixed <= 1e-9
    if fixed.size:
        pv["chernoff_zero_fraction"] = float(stats.binomtest(int(zero.sum()), fixed.size, 0.5).pvalue)
        pv["chernoff_positive"] = float(stats.kstest(fixed[~zero], stats.chi2(1).cdf).pvalue) if (~zero).any() else 0.0
        z = (fixed.mean() - 0.5) / np.sqrt(1.25 / fixed.size)
        pv["chernoff_mean"] = float(2 * stats.norm.sf(abs(z)))
    else:
        pv.update(chernoff_zero_fraction=0.0, chernoff_positive=0.0, chernoff_mean=0.0)
    ratio = {}
    for s_ in statistics:
        mine_s = np.array([r["grid"][s_] for r in null])
        theirs_s = np.asarray(mc.null[s_]) if mc is not None else np.array([])
        if not (theirs_s.size and mine_s.size):
            pv[f"simulators_ad_{s_}"] = 0.0
            continue
        pv[f"simulators_ad_{s_}"] = float(stats.anderson_ksamp([mine_s, theirs_s], method=stats.PermutationMethod(
            n_resamples=999, random_state=0)).pvalue)
        for fap in (1e-2, 1e-3):  # reported: virgil's own empirical thresholds, a 95% interval on the rate ratio
            t = np.quantile(theirs_s, 1 - fap)
            k1, k2 = int(np.sum(mine_s >= t)), int(np.sum(theirs_s >= t))
            lo, hi = stats.binomtest(k1, k1 + k2).proportion_ci(0.95, method="exact") if k1 + k2 else (0.0, 1.0)
            scale = theirs_s.size / mine_s.size  # odds k1/k2 -> rate ratio
            ratio[f"{s_}@{fap:g}"] = {"ours": k1, "virgil": k2,
                                      "rate_ratio_ci": [lo / (1 - lo) * scale if lo < 1 else float("inf"),
                                                        hi / (1 - hi) * scale if hi < 1 else float("inf")]}
    out["reported"]["exceedance_ratio_ci"] = ratio
    lam = np.array([r["lam"] for r in inj])
    true = np.array([r["true"] for r in inj])
    root = np.sqrt(lam)
    atom = stats.norm.cdf(-root)
    rng = np.random.default_rng(0)
    pit = np.where(true <= 1e-9, rng.uniform(0, 1, true.size) * atom, stats.norm.cdf(np.sqrt(np.maximum(true, 0)) - root))
    pv["boundary_pit"] = float(stats.kstest(pit, "uniform").pvalue) if inj else 0.0
    # moments of X = max(0, m + Z)^2: E = (m^2 + 1) Phi(m) + m phi(m), E[X^2] from the truncated normal
    m = root
    e1 = (m**2 + 1) * stats.norm.cdf(m) + m * stats.norm.pdf(m)
    e2 = (m**4 + 6 * m**2 + 3) * stats.norm.cdf(m) + (m**3 + 5 * m) * stats.norm.pdf(m)
    zb = np.sum(true - e1) / np.sqrt(np.sum(e2 - e1**2)) if inj else np.inf
    pv["boundary_mean"] = float(2 * stats.norm.sf(abs(zb)))
    out["reported"]["lambda_range"] = [float(lam.min()), float(lam.max())] if inj else None
    secondary_reject = holm(pv, spec["secondary"]["family_alpha"])
    out["rejected"] = sorted(n for n, r in secondary_reject.items() if r)

    # --- must hold
    book = {}
    if mc is not None:
        for s_ in statistics:
            v = np.sort(np.asarray(mc.null[s_]))
            n = v.size
            for fap in (1e-2, 3e-3, 1e-3):
                value = float(np.quantile(v, 1 - fap))
                k = int(np.sum(v >= value))
                ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
                got, lo, hi = (float(np.asarray(x)) for x in mc.false_alarm_probability(s_, value))
                thr_, _ = mc.threshold(s_, fap, n_boot=0)
                book[f"{s_}@{fap}"] = bool(abs(got - (k + 1) / (n + 1)) < 1e-12 and abs(lo - ci.low) < 1e-9
                                           and abs(hi - ci.high) < 1e-9 and abs(thr_ - value) < 1e-12)
    mh = out["must_hold"]
    mh["bookkeeping"] = {"cases": book, "pass": bool(book) and all(book.values())}
    ident = {}
    for (part, _, device), r in rows.items():
        if part == "check":
            d_ = np.abs(np.array([x["real"] - x["proxy"] for x in r]))
            ident[device] = {"n": len(r), "max_abs_diff": float(d_.max()), "ok": bool(len(r) >= counts["min_check"] and d_.max() <= 0.05)}
    mh["statistic_identity"] = {"devices": ident, "pass": bool(ident) and all(v["ok"] for v in ident.values())}
    pairs = {}
    for device, chunks in mcs.items():
        if device == "cpu":
            continue
        common = sorted(set(chunks) & set(mcs.get("cpu", {})))
        res = {"n_paired": 0, "frac_off": {}, "exceed_cpu": {}, "exceed_other": {}}
        if common:
            for s_ in statistics:
                a = np.concatenate([np.asarray(mcs["cpu"][k].null[s_]) for k in common])
                b = np.concatenate([np.asarray(chunks[k].null[s_]) for k in common])
                res["n_paired"] = int(a.size)
                res["frac_off"][s_] = float(np.mean(np.abs(a - b) > 0.05))
                if s_ == "delta_chi2":
                    for name, t in thr.items():
                        res["exceed_cpu"][name], res["exceed_other"][name] = int(np.sum(a >= t)), int(np.sum(b >= t))
        res["ok"] = bool(res["n_paired"] >= counts["min_virgil_other"]
                         and all(f <= 1e-3 for f in res["frac_off"].values())
                         and all(abs(res["exceed_cpu"][k] - res["exceed_other"][k]) <= max(3, 0.03 * res["exceed_cpu"][k])
                                 for k in res["exceed_cpu"]))
        pairs[device] = res
    mh["device_agreement"] = {"devices": pairs, "pass": bool(pairs) and all(v["ok"] for v in pairs.values())}
    isw = {f"{sim}@{name}": {**e, "ok": bool(abs(e["mean_weight"] - 1) <= 4 * e["mean_weight_se"])}
           for (sim, name), e in est.items() if e}
    mh["is_weights"] = {"cases": isw, "pass": bool(isw) and all(v["ok"] for v in isw.values())}
    isp = {f"{sim}@{name}": {"rel_se": e["rel_se"], "ess": e["ess"], "n": e["n"],
                             "ok": bool(e["n"] >= counts["min_is"] and e["rel_se"] is not None and e["rel_se"] <= 0.05 and e["ess"] >= 400)}
           for (sim, name), e in est.items() if e and name in ("1e-3",)}
    mh["is_precision"] = {"cases": isp, "pass": bool(isp) and len(isp) == 2 and all(v["ok"] for v in isp.values())}
    target = prim["precision_target"]
    mh["primary_precision"] = {"log_se": {k: v.get("log_se") for k, v in tests.items()},
                               "pass": all(v.get("log_se") is not None and v["log_se"] <= target for v in tests.values())}

    # --- reported
    rep = out["reported"]
    rep["far_tail"] = {name: {"ours": est[("null_is", name)], "virgil": est[("virgil_is", name)],
                              **{k: v for k, v in is_ratio(est[("null_is", name)], est[("virgil_is", name)], margin).items()
                                 if k in ("ratio", "ci95")}} for name in thr}
    nominal = {}
    for name, alpha in NOMINAL.items():
        for sim, key in (("ours", "proxy_ours"), ("virgil", "proxy_virgil")):
            v = np.asarray(values[key])
            if v.size and alpha * v.size >= 100:
                k = int(np.sum(v >= thr[name]))
                nominal[f"proxy_{sim}@{name}"] = {"rate": k / v.size, "nominal": alpha, "k": k}
        for sim in ("null_is", "virgil_is"):
            e = est[(sim, name)]
            if e:
                nominal[f"{sim}@{name}"] = {"rate": e["p"], "rel_se": e["rel_se"], "nominal": alpha}
    rep["nominal"] = nominal
    gpd = {}
    for key in values:
        if len(values[key]) >= 100000:
            for name in ("1e-3", "1e-4"):
                p, (lo, hi) = tails.gpd_tail(values[key], thr[name])
                gpd[f"{key}@{name}"] = {"p": p, "interval90": [lo, hi], "nominal": NOMINAL[name]}
    rep["gpd_cross_check"] = gpd
    if mc is not None and inj:  # reported: completeness at virgil's 1e-3 threshold
        thr3 = float(np.quantile(np.asarray(mc.null["delta_chi2"]), 1 - 1e-3))
        comp = {}
        for sep in SEPS:
            for f in FLUXES:
                sel = [r for r in inj if r["sep"] == sep and np.isclose(r["flux"], f)]
                if sel:
                    comp[f"{sep:g}mas_{f:.2g}"] = float(np.mean([r["grid"]["delta_chi2"] > thr3 for r in sel]))
        rep["completeness"] = {"threshold_delta_chi2": thr3, "by_sep_flux": comp}

    n_virgil = {d: int(sum(m_.n_null for m_ in c.values())) for d, c in mcs.items()}
    n_other = max([n for d, n in n_virgil.items() if d != "cpu"] or [0])
    out["counts"] = {"null": len(null), "inject": len(inj), "virgil": n_virgil,
                     "proxy": {k: len(v) for k, v in values.items()},
                     "is": {f"{sim}@{name}": len(rows[(sim, name, None)]) for sim in ("null_is", "virgil_is") for name in thr
                            if rows[(sim, name, None)]},
                     "check": {d: len(r) for (p_, _, d), r in rows.items() if p_ == "check"}}
    enough = (len(null) >= counts["min_null"] and len(inj) >= counts["min_injected"]
              and n_virgil.get("cpu", 0) >= counts["min_virgil_cpu"] and n_other >= counts["min_virgil_other"]
              and all(len(v) >= counts["min_proxy"] for v in values.values())
              and all(len(rows[(sim, "1e-3", None)]) >= counts["min_is"] for sim in ("null_is", "virgil_is")))
    out["enough"] = bool(enough)
    out["pass"] = bool(enough and not out["rejected_primary"] and not out["rejected"]
                       and all(v["pass"] for v in mh.values()))
    text = json.dumps(out, indent=1)
    if args.summary:
        pathlib.Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.summary).write_text(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--part", choices=list(SEED_OFFSET), required=True)
    r.add_argument("--threshold", help="registered threshold name (null_is, virgil_is)")
    r.add_argument("--device", choices=["cpu", "default"], default="cpu", help="cpu, or the default device (a GPU)")
    r.add_argument("--task", type=int, required=True)
    r.add_argument("--draws", type=int, default=100)
    r.add_argument("--seed-base", type=int, default=20261006)
    r.add_argument("--out", required=True)
    g = sub.add_parser("aggregate")
    g.add_argument("dirs", nargs="+")
    g.add_argument("--summary")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else aggregate(args)


if __name__ == "__main__":
    main()
