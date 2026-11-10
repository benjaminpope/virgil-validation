"""Informed fits of the imaging contests' data: the model family the contest paper describes, fitted from the data.

    python scripts/contest_informed.py --list
    python scripts/contest_informed.py --task N --model FAMILY [--data DIR] [--out DIR] [--starts 32] [--smoke]

This is **not** a blind reconstruction, and its results must never be quoted as one. It uses what a contest paper
said the object *is* (a spotted star with a companion; a thin disk; a star, a disk rim, a planet and an outer disk;
a ringed disk with companions) to choose a parametric model. It uses nothing the paper said about the *values*: no diameter, position, flux ratio or angle
from a paper is a start, a bound or a constant here. Every start is drawn from priors whose scales come from the
data's own baselines, so the published parameters stay a test of the fit (they are compared afterwards, by the
campaign's sealed stage).

A family is a ladder of nested models, simplest first (FAMILIES). Each rung is fitted from many random starts drawn
from its priors, plus the best fit of the rung below; the best start of each rung is kept and the rungs are compared
by BIC on the quoted errors. Writes into --out:

- <label>_<family>.json: per rung, the best parameters, chi2, N, k, BIC, how many starts reached the best chi2
  (a small number means the optimum is hard to find or not unique), and the chosen rung;
- result.json: status and checks for the campaign (`converged`, `starts_agree`), with chi2/N of the chosen rung.

Task numbers and files are those of scripts/contest_images.py (TASKS).
"""

import argparse
import ast
import json
import pathlib
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

import virgil  # noqa: E402
import virgil.models as vm  # noqa: E402
from virgil.fitting import fit  # noqa: E402
from virgil.grid_fit import likelihood_grid  # noqa: E402
from virgil.oidata import OIData  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
REACH = 1.5  # positions are searched out to this many fringe periods of the shortest baseline (see prior_of)
PEAKS = 6  # likelihood-map peaks refined per new component
TIME_LIMIT = 300.0  # seconds per start; a start that runs out is recorded as not converged
AGREE = 1.01  # a start "reached the best" when its chi2 is within 1% of it
MIN_AGREE = 2  # fewer starts than this at the best chi2: the optimum was found once, by luck


def tasks():
    """TASKS of contest_images.py, read without importing it (that module sets up a whole imaging run)."""
    tree = ast.parse((REPO / "scripts" / "contest_images.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "TASKS" for t in node.targets):
            return eval(compile(ast.Expression(node.value), "contest_images.py", "eval"), {"dict": dict})
    raise KeyError("TASKS")


def scales(datasets):
    """(resolution, field) in mas from the data's own baselines: lambda / 2 B_max and lambda / B_min."""
    mas = 180.0 / np.pi * 3600e3
    longest, shortest = 0.0, np.inf
    for d in datasets:
        b = np.hypot(np.asarray(d.u, float), np.asarray(d.v, float)).ravel()
        wavel = np.asarray(d.wavel, float).ravel()
        b = b[b > 0]
        longest = max(longest, float(b.max() / wavel.min()))
        shortest = min(shortest, float(b.min() / wavel.max()))
    return mas / (2.0 * longest), mas / shortest


# Each family is a ladder of rungs, simplest first: (name, {component: (maker, {parameter path: prior kind})}).
# A maker is a virgil class, or "RingedDisk" (make_component). Prior kinds are scale-free words that `prior_of`
# turns into distributions from the data's resolution and field; no number here comes from a paper. The first
# component of a rung is the flux reference (flux 1); every other component's flux is relative to it.
POINT = ("PointSource", {})
SMALL_STAR = ("UniformDisk", {"diam": "small"})
LD_STAR = ("LimbDarkenedDisk", {"diam": "size", "u": "limb"})
ELLIPTICAL_STAR = ("EllipticalLimbDarkenedDisk", {"diam": "size", "ratio": "ratio", "pa": "angle", "u": "limb"})
COMPANION = ("PointSource", {"flux": "faint", "dra": "far", "ddec": "far"})
SMALL_DISK = ("UniformDisk", {"diam": "small", "flux": "faint", "dra": "far", "ddec": "far"})
SPOT = ("UniformDisk", {"diam": "small", "flux": "faint", "dra": "near", "ddec": "near"})
SURFACE_SPOT = ("UniformDisk", {"diam": "small", "flux": "faint", "dra": "far", "ddec": "far"})
HALO = ("GaussianDisk", {"sigma": "size", "flux": "bright"})
ELLIPSE = ("EllipticalGaussian", {"fwhm": "size", "ratio": "ratio", "pa": "angle", "flux": "bright"})
RIM = ("ModulatedGaussianRim", {"diam": "size", "fwhm": "size", "inc": "inclination", "pa": "angle",
                                "az_amps": "modulation", "az_pas": "full_angle", "flux": "bright"})
ARC = ("GaussianArc", {"radius": "size", "width": "size", "length": "size", "pa": "full_angle", "flux": "bright"})


def rings(*names):
    """An inclined disk of Gaussian rings; each ring after the first has a weight relative to the first."""
    params = {"inc": "inclination", "pa": "angle", "flux": "bright"}
    for i, name in enumerate(names):
        params[f"profile.{name}.radius"] = "size"
        params[f"profile.{name}.sigma"] = "size"
        if i:
            params[f"profile.{name}.weight"] = "bright"
    return ("RingedDisk", params)


FAMILIES = {
    # Lawson et al. 2004, data1: a star inside an asymmetric shell.
    "star_in_shell": [
        ("star+halo", {"star": SMALL_STAR, "halo": HALO}),
        ("star+shell", {"star": SMALL_STAR, "shell": RIM}),
        ("star+shell+halo", {"star": SMALL_STAR, "shell": RIM, "halo": HALO}),
    ],
    # Lawson et al. 2004, data2: a binary whose primary is a limb-darkened elliptical disk with a bright spot.
    "spotted_star_binary": [
        ("star", {"star": ELLIPTICAL_STAR}),
        ("star+companion", {"star": ELLIPTICAL_STAR, "companion": SMALL_DISK}),
        ("star+companion+spot", {"star": ELLIPTICAL_STAR, "companion": SMALL_DISK, "spot": SPOT}),
    ],
    # Lawson et al. 2006: a geometrically thin disk.
    "thin_disk": [
        ("disk", {"disk": ("EllipticalGaussian", {"fwhm": "size", "ratio": "ratio", "pa": "angle"})}),
        ("star+disk", {"star": POINT, "disk": ELLIPSE}),
        ("star+ring", {"star": POINT, "ring": RIM}),
        ("star+ring+disk", {"star": POINT, "ring": RIM, "disk": ELLIPSE}),
    ],
    # Cotton et al. 2008: an AGB star. A limb-darkened disk, an envelope, and structure on the surface.
    "evolved_star_with_surface_structure": [
        ("star", {"star": LD_STAR}),
        ("star+envelope", {"star": LD_STAR, "envelope": HALO}),
        ("star+envelope+spot", {"star": LD_STAR, "envelope": HALO, "spot": SURFACE_SPOT}),
        ("star+envelope+two spots", {"star": LD_STAR, "envelope": HALO, "spot": SURFACE_SPOT, "spot_b": SURFACE_SPOT}),
    ],
    # Cotton et al. 2008: an AGN. An unresolved core and elongated extended emission.
    "agn": [
        ("core+dust", {"core": POINT, "dust": ELLIPSE}),
        ("core+dust+extended", {"core": POINT, "dust": ELLIPSE, "extended": ELLIPSE}),
    ],
    # Malbet et al. 2010: a red supergiant's structured surface, and a much fainter companion far outside the disk.
    "structured_supergiant_with_companion": [
        ("star", {"star": LD_STAR}),
        ("star+companion", {"star": LD_STAR, "companion": COMPANION}),
        ("star+companion+spot", {"star": LD_STAR, "companion": COMPANION, "spot": SURFACE_SPOT}),
        ("star+companion+two spots", {"star": LD_STAR, "companion": COMPANION, "spot": SURFACE_SPOT, "spot_b": SURFACE_SPOT}),
    ],
    # Sanchez-Bermudez et al. 2018: a star, an inner disk with a bright rim, a planet, and an outer disk.
    "star_rim_planet_disk": [
        ("star+outer", {"star": SMALL_STAR, "outer": ELLIPSE}),
        ("star+rim+outer", {"star": SMALL_STAR, "rim": RIM, "outer": ELLIPSE}),
        ("star+rim+outer+planet", {"star": SMALL_STAR, "rim": RIM, "outer": ELLIPSE, "planet": COMPANION}),
    ],
    # The 2022 contest: high-contrast tests. A star, possibly extended light, and a faint companion.
    "star_with_faint_companion": [
        ("star", {"star": SMALL_STAR}),
        ("star+halo", {"star": SMALL_STAR, "halo": HALO}),
        ("star+halo+companion", {"star": SMALL_STAR, "halo": HALO, "companion": COMPANION}),
    ],
    # The 2024 contest, Obj1: a Wolf-Rayet pinwheel of several turns. virgil has no spiral component, so each turn is
    # approximated by one arc; the arcs are not tied to a common pitch. A stand-in until a spiral exists in virgil.
    "pinwheel_arcs": [
        ("star+envelope", {"star": POINT, "envelope": HALO}),
        ("star+envelope+arc", {"star": POINT, "envelope": HALO, "arc": ARC}),
        ("star+envelope+two arcs", {"star": POINT, "envelope": HALO, "arc": ARC, "arc_b": ARC}),
        ("star+envelope+three arcs", {"star": POINT, "envelope": HALO, "arc": ARC, "arc_b": ARC, "arc_c": ARC}),
    ],
    # The 2024 contest, Obj2: an inclined disk with several rings and gaps, and point-like companions.
    "ringed_disk_with_companions": [
        ("star+one ring", {"star": POINT, "disk": rings("a")}),
        ("star+two rings", {"star": POINT, "disk": rings("a", "b")}),
        ("star+three rings", {"star": POINT, "disk": rings("a", "b", "c")}),
        ("star+three rings+companion", {"star": POINT, "disk": rings("a", "b", "c"), "companion": COMPANION}),
    ],
}
# A value for the constructor before fitting; every free parameter is then set from a prior draw.
PLACEHOLDER = {"diam": 1.0, "fwhm": 1.0, "sigma": 1.0, "radius": 1.0, "width": 1.0, "length": 1.0, "ratio": 0.9, "pa": 0.0,
               "u": 0.5, "inc": 30.0, "az_amps": 0.1, "az_pas": 0.0, "flux": 0.1, "weight": 1.0, "dra": 0.0, "ddec": 0.0}


def prior_of(kind, res, field):
    return {
        "size": dist.LogUniform(res / 5.0, field),  # anything from well under the resolution to the whole field
        "small": dist.LogUniform(res / 20.0, 4.0 * res),  # barely resolved or unresolved
        "ratio": dist.Uniform(0.3, 1.0),
        "angle": dist.Uniform(0.0, 180.0),
        "full_angle": dist.Uniform(0.0, 360.0),
        "inclination": dist.Uniform(0.0, 80.0),
        "limb": dist.Uniform(0.0, 1.0),
        "modulation": dist.Uniform(0.0, 1.0),
        "faint": dist.LogUniform(1e-4, 1.0),  # relative to the star, which is fixed at flux 1
        "bright": dist.LogUniform(1e-3, 10.0),
        # Anywhere the data can place a source. lambda / B_min is the fringe period of the shortest baseline, not an
        # edge: longer baselines still locate a source beyond it. The blind images' field rule also grows to 1.5x.
        "far": dist.Uniform(-REACH * field, REACH * field),
        "near": dist.Uniform(-3.0 * res, 3.0 * res),  # a feature on or beside the star (a few resolution elements)
    }[kind]


VECTORS = ("u", "az_amps", "az_pas")  # parameters that are arrays of one coefficient here


def make_component(maker, values):
    """A virgil component from ``{parameter path: number}``."""
    if maker != "RingedDisk":
        return getattr(vm, maker)(**values)
    parts = {}
    for path, value in values.items():
        if path.startswith("profile."):
            _, ring, what = path.split(".")
            parts.setdefault(ring, {})[what] = value
    profile = vm.RadialSum(**{ring: (vm.GaussianRing(radius=v["radius"], sigma=v["sigma"]), v.get("weight", 1.0))
                              for ring, v in parts.items()})
    return vm.MultipoleDisk(profile, inc=values["inc"], pa=values["pa"], flux=values["flux"])


def build(rung, res, field, values=None):
    """(template System, {path: prior}) of one rung, with any given ``values`` ({path: number}) set on the template."""
    parts, priors = {}, {}
    for name, (maker, params) in rung.items():
        kwargs = {}
        for p in params:
            leaf = p.rsplit(".", 1)[-1]
            value = (values or {}).get(f"{name}.{p}", PLACEHOLDER[leaf])
            kwargs[p] = np.array([value]) if leaf in VECTORS else value
        parts[name] = make_component(maker, kwargs)
        for p, kind in params.items():
            priors[f"{name}.{p}"] = prior_of(kind, res, field)
    return vm.System(**parts), priors


def seeds_from_map(name, rung, datasets, res, field, carry):
    """Starts for a newly added component, from a likelihood map over its position and flux with the rest of the
    scene held at the best fit of the rung below. The map's highest, well separated peaks are returned as
    ``{path: value}`` (virgil's documented companion search; nothing is assumed about where the component is)."""
    kinds = rung[name][1]
    half = (REACH * field) if kinds.get("dra") == "far" else 3.0 * res
    n = int(min(121, 2 * round(2.0 * half / res) + 1))  # about half a resolution element per step
    axis = np.linspace(-half, half, n)
    fluxes = 10.0 ** np.linspace(-3.5, 0.0, 12)
    template, _ = build(rung, res, field, carry)
    grid = {f"{name}.dra": axis, f"{name}.ddec": axis, f"{name}.flux": fluxes}
    data = datasets[0] if len(datasets) == 1 else datasets
    loglike = np.asarray(likelihood_grid(template, data=data, grid=grid))
    best_flux = np.argmax(loglike, axis=2)
    surface = np.max(loglike, axis=2)
    seeds = []
    for flat in np.argsort(surface, axis=None)[::-1]:
        i, j = np.unravel_index(flat, surface.shape)
        if all(np.hypot(axis[i] - q[f"{name}.dra"], axis[j] - q[f"{name}.ddec"]) > 2.0 * res for q in seeds):
            seeds.append({f"{name}.dra": float(axis[i]), f"{name}.ddec": float(axis[j]),
                          f"{name}.flux": float(fluxes[best_flux[i, j]])})
        if len(seeds) == PEAKS:
            break
    return seeds


def draw(priors, rng):
    """One start: each parameter drawn from its own prior."""
    out = {}
    for path, p in priors.items():
        if isinstance(p, dist.LogUniform):
            value = float(np.exp(rng.uniform(np.log(float(p.low)), np.log(float(p.high)))))
        else:
            value = float(rng.uniform(float(p.low), float(p.high)))
        out[path] = np.array([value]) if path.rsplit(".", 1)[-1] in VECTORS else value
    return out


def fit_rung(name, rung, datasets, res, field, rng, starts, carry, max_steps, below=()):
    template, priors = build(rung, res, field)
    data = datasets[0] if len(datasets) == 1 else datasets
    # Starts: for each component this rung adds that has a position, the peaks of its likelihood map with the rest
    # of the scene at the best fit below; then random draws from the priors.
    inits = []
    for comp in [c for c in rung if c not in below and "dra" in rung[c][1]]:
        for seed in seeds_from_map(comp, rung, datasets, res, field, carry):
            inits.append({**draw(priors, rng), **carry, **seed})
    if carry and not inits:
        inits.append({**draw(priors, rng), **carry})
    inits += [draw(priors, rng) for _ in range(starts)]
    runs, n = [], 0
    for init in inits:
        init = {k: (np.array([float(np.ravel(v)[0])]) if k.rsplit(".", 1)[-1] in VECTORS else float(np.ravel(v)[0]))
                for k, v in init.items() if k in priors}
        try:
            r = fit(template, priors, data, init=init, max_steps=max_steps, time_limit=TIME_LIMIT)
        except Exception as err:  # noqa: BLE001 a start that fails is recorded, not hidden
            runs.append({"chi2": float("inf"), "error": f"{type(err).__name__}: {err}"})
            continue
        chi2, n = float(np.sum(r.info["chi2"])), int(np.sum(r.info["ndata"]))
        runs.append({"chi2": chi2 if np.isfinite(chi2) else float("inf"), "converged": bool(r.info.get("converged")),
                     "values": {k: [float(x) for x in np.ravel(v)] for k, v in r.values.items()}})
    best = min(runs, key=lambda x: x["chi2"])
    k = len(priors)
    agree = sum(x["chi2"] <= AGREE * best["chi2"] for x in runs)
    return {"rung": name, "k": k, "n": n, "chi2": best["chi2"], "chi2_red": best["chi2"] / max(n, 1),
            "bic": best["chi2"] + k * float(np.log(max(n, 1))), "converged": best.get("converged", False),
            "starts": len(inits), "starts_at_best": int(agree), "failed_starts": sum("error" in x for x in runs),
            "values": best.get("values", {})}


def run(task, family, data_dir, out_dir, starts=32, smoke=False):
    spec = tasks()[task]
    label = f"{spec['label']}_{family}"
    paths = [pathlib.Path(data_dir) / f for f in spec["files"]]
    if spec.get("wavel"):  # the same channel selection as the blind images of this task
        import tempfile

        sys.path.insert(0, str(REPO / "scripts"))
        import contest_images

        tmp = pathlib.Path(tempfile.mkdtemp())
        paths = [contest_images.select_channels(p, spec["wavel"], tmp) for p in paths]
    datasets = [OIData(str(p)) for p in paths]
    res, field = scales(datasets)
    rng = np.random.default_rng(7000 + task)
    t0, rungs, carry, below = time.time(), [], {}, ()
    for name, rung in FAMILIES[family]:
        got = fit_rung(name, rung, datasets, res, field, rng, 3 if smoke else starts, carry, 50 if smoke else None, below)
        carry, below = {k: v[0] for k, v in got["values"].items() if len(v) == 1}, tuple(rung)
        rungs.append(got)
        print(f"{label} {name}: chi2/N={got['chi2_red']:.3f} k={got['k']} BIC={got['bic']:.1f} "
              f"starts at best {got['starts_at_best']}/{got['starts']}")
    chosen = min(rungs, key=lambda x: x["bic"])
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"label": label, "task": task, "family": family, "knowledge": "informed: model family from the contest paper; no value",
              "files": spec["files"], "resolution_mas": res, "field_mas": field, "virgil": getattr(virgil, "__version__", None),
              "rungs": rungs, "chosen": chosen["rung"], "elapsed_s": time.time() - t0, "smoke": smoke}
    (out_dir / f"{label}.json").write_text(json.dumps(report, indent=1) + "\n")
    agree = chosen["starts_at_best"] >= MIN_AGREE
    checks = {"converged": "pass" if chosen["converged"] else "fail", "starts_agree": "pass" if agree else "warn"}
    status = "fail" if not chosen["converged"] else ("warn" if not agree or chosen["chi2_red"] > 1.5 else "pass")
    (out_dir / "result.json").write_text(json.dumps(
        {"status": status, "checks": checks, "chosen": chosen["rung"], "chi2_red": chosen["chi2_red"],
         "starts_at_best": chosen["starts_at_best"], "knowledge": "informed"}, indent=1) + "\n")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--task", type=int)
    ap.add_argument("--model", choices=sorted(FAMILIES))
    ap.add_argument("--data", default="~/data/imaging_contests")
    ap.add_argument("--out", default="contest_informed")
    ap.add_argument("--starts", type=int, default=32, help="random starts per rung")
    ap.add_argument("--smoke", action="store_true", help="three starts of 50 steps: check that it runs")
    a = ap.parse_args(argv)
    if a.list:
        for name, rungs in FAMILIES.items():
            print(name + ": " + " -> ".join(r[0] for r in rungs))
        return 0
    if a.task is None or a.model is None:
        ap.error("--task and --model are required")
    run(a.task, a.model, pathlib.Path(a.data).expanduser(), pathlib.Path(a.out).expanduser(), a.starts, a.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
