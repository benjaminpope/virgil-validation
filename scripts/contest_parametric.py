"""Stage C2e: parametric (SPARCO-style) and combined routes for the contest datasets
that image-only fits do not reach (2004 data2, 2024 Obj2).

    python scripts/contest_parametric.py --list
    python scripts/contest_parametric.py --target 2024_obj2 [--data DIR] [--out DIR] [--smoke]

One target is a dataset, or the datasets fitted jointly (2024 Obj2: PIONIER and
GRAVITY, one flux-ratio spectrum across both). Three routes are fitted to it and
compared:

- ``parametric``: a uniform-disk star plus a point companion (the contest
  page's "suspected companion", or what the 2004 data2 residuals showed),
  fitted by virgil.fitting.fit. With several wavelengths the companion's flux
  ratio is a power law (virgil.spectra.PowerLaw, SPARCO), a free index;
- ``image``: an unresolved star plus a CLEAN-started Gaussian-process image
  (the earlier members' recipe), hyperparameters chosen by log evidence;
- ``combined``: the parametric model plus the image of what it leaves, CLEAN
  started from the parametric fit (virgil.imaging.clean with it as the base),
  the parametric parameters free in the final fit.

The route with the highest log evidence is the choice (a difference under
EVIDENCE_TIE prefers the route with fewer parameters). Every χ²/N reported is
raw, on the quoted errors, split into V² and closure phases per dataset; the
error scale (MacKay) is information, never a replacement for them. The
companion's presence is ``informed`` for 2004 data2 (found in our own residuals,
design/contest_convergence_diagnosis.md) and ``blind`` for Obj2 (the contest
page). Nothing published after the deadline is used.

It writes <label>.json (all numbers) and <label>.txt into --out. Heavy: run on
OzSTAR (design/plan_imaging_contests.md C2e); --smoke runs in seconds.
"""

import argparse
import json
import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import contest_images as ci  # noqa: E402
import virgil.models as vm  # noqa: E402
from virgil.fields import GaussianField  # noqa: E402
from virgil.fitting import fit  # noqa: E402
from virgil.imaging import beam, chi2_by_observable, clean, error_scale, image_priors, log_evidence, starting_image  # noqa: E402
from virgil.oidata import OIData  # noqa: E402
from virgil.spectra import PowerLaw  # noqa: E402

# tasks: contest_images.TASKS labels whose files are fitted together.
TARGETS = {
    "2004_data2": dict(tasks=["2004_data2"], companion="point", status="informed"),
    "2024_obj2": dict(tasks=["2024_obj2_pionier", "2024_obj2_gravity"], companion="point", status="blind"),
}
ROUTES = ("parametric", "image", "combined")
BIC_TIE = 10.0  # BIC difference below which the simpler route is kept ("strong" evidence, Kass & Raftery 1995)
MAX_PIX = ci.CLEAN_MAX_PIX  # one grid for CLEAN and the GP: the parametric part carries the compact structure
SPARCO_BANDWIDTH = 1.1  # max/min wavelength above which the flux ratio gets a free index
COMPANION_FLUX = (1e-3, 10.0)  # log-uniform flux ratio to the star


def load(label, data_dir):
    """The OIData list of a target (channels outside a task's ``wavel`` flagged)."""
    import tempfile

    by_label = {t["label"]: t for t in ci.TASKS}
    out = []
    for name in TARGETS[label]["tasks"]:
        spec = by_label[name]
        paths = [str(pathlib.Path(data_dir) / f) for f in spec["files"]]
        if spec.get("wavel"):
            tmp = tempfile.mkdtemp(prefix="contest_")
            paths = [ci.select_channels(p, spec["wavel"], tmp) for p in paths]
        out += [OIData(p) for p in paths]
    return out


def wavelengths(data):
    w = np.concatenate([np.ravel(np.asarray(d.wavel)) for d in data])
    return float(np.median(w)), float(w.max() / w.min())


def parametric_model(companion, diam, flux, dra, ddec, sparco, wavel0):
    """A uniform-disk star (flux 1) and, if ``companion``, a point source (or
    Gaussian, ``companion="gauss"``, FWHM-sigma = ``diam``) at (dra, ddec) mas
    with flux ratio ``flux``, a power law in wavelength when ``sparco``."""
    parts = {"star": vm.UniformDisk(diam)}
    if companion:
        f = PowerLaw(flux, 0.0, wavel0) if sparco else flux
        parts["comp"] = vm.PointSource(f, dra, ddec) if companion == "point" else vm.GaussianDisk(diam, f, dra, ddec)
    return vm.System(**parts)


def parametric_priors(companion, res, fov, sparco):
    """Log-uniform scales (diameter from a twentieth of the beam to its major
    axis, flux ratio), uniform locations (offsets within the field, index)."""
    priors = {"star.diam": dist.LogUniform(0.05 * res.minor_mas, res.major_mas)}
    if companion:
        flux = {"comp.flux.ratio": dist.LogUniform(*COMPANION_FLUX), "comp.flux.index": dist.Uniform(-8.0, 8.0)} if sparco \
            else {"comp.flux": dist.LogUniform(*COMPANION_FLUX)}
        half = dist.Uniform(-fov / 2, fov / 2)
        priors |= flux | {"comp.dra": half, "comp.ddec": half}
    return priors


def chi2_report(model, data):
    """Raw χ² and χ²/N over all datasets, and per dataset V² and closure phases apart."""
    total = chi2_by_observable(model, data)
    diag = ci.residual_diagnostics(model, data)
    return {"datasets": [{k: d[k] for k in ("chi2_v2", "chi2_cp", "short_mean")} for d in diag],
            "chi2": float(total["chi2"]["total"]), "chi2_red": float(total["chi2_red"]["total"])}


def companion_start(data, star, res, fov, n):
    """Starting offsets for the companion: the brightest CLEAN component of the
    residual of the star alone outside one beam (inside it, flux is the star's),
    and its mirror through the centre (closure phases alone cannot tell them
    apart; the fit chooses)."""
    cleaned = clean(data, n, fov / n, base=star, max_iterations=200, target_chi2_red=1.0)
    c = (n - 1) / 2
    rows, cols = np.indices((n, n))
    comp = np.where(np.hypot(rows - c, cols - c) * fov / n > res.major_mas, np.asarray(cleaned.components), 0.0)
    row, col = np.unravel_index(np.argmax(comp), comp.shape)
    dra, ddec = -(col - c) * fov / n, -(row - c) * fov / n  # column 0 is East, row 0 is North
    return [(dra, ddec), (-dra, -ddec)]


def fit_parametric(data, companion, res, fov, n, sparco, wavel0, max_steps=None):
    """Star plus companion from each of the companion's starts, best χ² kept (the star alone if no companion).
    Returns the FitResult and its priors."""
    opts = {} if max_steps is None else {"max_steps": max_steps}
    diam0 = 0.5 * res.minor_mas  # not the star-alone fit's: the companion's flux would inflate it
    if not companion:
        priors = parametric_priors(None, res, fov, False)
        return fit(parametric_model(None, diam0, 0, 0, 0, False, wavel0), priors, data, **opts), priors
    priors = parametric_priors(companion, res, fov, sparco)
    fits = [fit(parametric_model(companion, diam0, 0.1, dra, ddec, sparco, wavel0), priors, data, **opts)
            for dra, ddec in companion_start(data, vm.UniformDisk(diam0), res, fov, n)]
    return min(fits, key=lambda r: float(np.sum(r.info["chi2_red"]))), priors


def parts_of(model):
    """The parametric components (star, companion) of a fitted System."""
    return {k: getattr(model, k) for k in ("star", "comp") if hasattr(model, k)}


def with_image(parts, restored, pixel, wavel0, sparco, flux, sigma, length):
    """The parametric ``parts`` plus a GP image whose mean is the CLEAN start."""
    n = restored.shape[0]
    field = GaussianField(np.zeros((n, n)), sigma, length, mean=restored)
    return vm.System(**parts, env=vm.Image(field, pixel, flux=PowerLaw(flux, 0.0, wavel0) if sparco else flux))


def fit_image(data, parts, base_priors, res, fov, n, sparco, wavel0, smoke):
    """CLEAN from the parametric ``parts``, then GP images on a (ℓ, σ) grid
    grown by ci.grow_grid; the best by log evidence. Returns (result, log_z, hyperparameters, γ)."""
    pixel = fov / n
    cleaned = clean(data, n, pixel, base=vm.System(**parts), base_priors=base_priors or None,
                    max_iterations=50 if smoke else 3000, target_chi2_red=1.0)
    # CLEAN's model is the base alone when the base already reaches the target χ².
    parts = parts_of(getattr(cleaned.model, "base", cleaned.model))
    restored = np.clip(np.asarray(cleaned.restored(res)), 0.0, None)
    restored = restored / max(restored.max(), 1e-12) + 1e-3  # a floor; flat when CLEAN found nothing
    flux = max(float(np.sum(cleaned.components)), 1e-3)
    cap = max(100.0, 10 * flux)
    options = {"max_steps": 50} if smoke else {}
    best = {}

    def evaluate(length, sigma):
        sc = with_image(parts, restored, pixel, wavel0, sparco, flux, sigma, length)
        extra = ci.flux_priors(cap, sparco) | base_priors
        r = fit(sc, image_priors(sc) | extra, data, **options)
        z = float(log_evidence(r, data))
        if not best or z > best["z"]:
            best.update(z=z, result=r, length=length, sigma=sigma)
        return z

    lengths = [f * res.minor_mas for f in (ci.GP_LENGTHS[1:2] if smoke else ci.GP_LENGTHS)]
    ci.grow_grid(evaluate, lengths, ci.GP_SIGMAS[1:2] if smoke else ci.GP_SIGMAS,
                 [f * res.minor_mas for f in ci.GP_LENGTH_LIMITS], ci.GP_SIGMA_LIMITS, 0 if smoke else ci.GP_GROWTHS)
    r = best["result"]
    return r, best["z"], {"length_mas": best["length"], "sigma": best["sigma"]}, float(chi2_by_observable(r.model, data, "env")["n_gamma"])


def choose(bic, tie=BIC_TIE):
    """The route with the lowest BIC; of the routes within ``tie`` of it, the
    simplest (parametric, then image, then combined)."""
    low = min(bic.values())
    return next(r for r in ROUTES if bic.get(r, np.inf) <= low + tie)


def run(label, data_dir, out_dir, smoke=False, field_factor=2.0):
    spec = TARGETS[label]
    data = load(label, data_dir)
    res = beam(data)
    wavel0, bandwidth = wavelengths(data)
    sparco = bandwidth > SPARCO_BANDWIDTH
    start = starting_image(data, star=True, oversample=3.0)
    fov = field_factor * float(start.env.pixel_scale_mas) * int(np.shape(start.env.log_brightness)[0])
    n = min(MAX_PIX, int(np.ceil(fov / float(start.env.pixel_scale_mas))) | 1)
    companion = spec["companion"]
    steps = 50 if smoke else None
    out = {"label": label, "status": spec["status"], "companion": companion, "sparco": bool(sparco),
           "wavel0_m": wavel0, "fov_mas": fov, "npix": n, "routes": {}}

    par, par_priors = fit_parametric(data, companion, res, fov, n, sparco, wavel0, steps)
    results = {"parametric": (par, None, {}, len(par_priors))}
    results["image"] = fit_image(data, {"star": vm.PointSource(1.0)}, {}, res, fov, n, sparco, wavel0, smoke)
    results["combined"] = fit_image(data, parts_of(par.model), par_priors, res, fov, n, sparco, wavel0, smoke)
    n_ind = sum(d.n_independent for d in data)
    for route, (r, z, hyper, k) in results.items():
        rep = chi2_report(r.model, data)
        entry = {"log_z": z, "k": k, "bic": rep["chi2"] + k * float(np.log(n_ind)), "hyper": hyper,
                 "error_scale": None if z is None else float(error_scale(r.model, data)),  # defined for the GP routes only
                  **rep}
        if route != "image" and companion:
            comp = r.model.comp
            entry["companion"] = {"dra": float(comp.dra), "ddec": float(comp.ddec),
                                  "flux": ci.flux_value(comp.flux), "index": float(np.ravel(getattr(comp.flux, "index", np.nan))[0])}
        out["routes"][route] = entry
    out["choice"] = choose({k: v["bic"] for k, v in out["routes"].items()})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{label}_parametric.json").write_text(json.dumps(out, indent=1, default=float))
    text = report(out)
    (out_dir / f"{label}_parametric.txt").write_text(text)
    print(text)
    return out


def report(out):
    lines = [f"{out['label']} ({out['status']}): sparco={out['sparco']} fov={out['fov_mas']:.3g} mas npix={out['npix']}"]
    for route, e in out["routes"].items():
        parts = ", ".join(f"V2 {d['chi2_v2']:.3g} CP {d['chi2_cp'] if d['chi2_cp'] is None else round(d['chi2_cp'], 3)}"
                          for d in e["datasets"])
        lines.append(f"{route:10s} BIC={e['bic']:.0f} k={e['k']:.1f} logZ(GP)={e['log_z']} chi2/N={e['chi2_red']:.3g} ({parts}) error_scale={e['error_scale']}"
                     + (f" companion={e['companion']}" if "companion" in e else ""))
    lines.append(f"CHOICE (lowest BIC, ties to the simpler route): {out['choice']}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=sorted(TARGETS))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--data", default="~/data/imaging_contests")
    parser.add_argument("--out", default="contest_parametric")
    parser.add_argument("--field", type=float, default=2.0, help="x the data-chosen field")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.list or not args.target:
        for k, v in TARGETS.items():
            print(k, v)
        return
    run(args.target, pathlib.Path(args.data).expanduser(), pathlib.Path(args.out).expanduser(), args.smoke, args.field)


if __name__ == "__main__":
    main()
