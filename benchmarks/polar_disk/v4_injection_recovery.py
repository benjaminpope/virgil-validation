"""V4: optical/IR injection-recovery of PolarDisk scenes (no reference code covers this).

Each injection draws a ring + gap + planet scene with POSITIVE azimuthal harmonics
(Fejer-Riesz form, M = 1..3) from the prior below, observes it on synthetic NRM
(squared visibilities and closure phases of the 7-hole NIRISS mask at 4.3 um) or VLTI
(four UTs, L band, V^2 and closure phases) coverage with Gaussian noise, fits it with
``virgil.fit`` (MAP) and takes the Laplace covariance, then asks whether the posterior
bands are calibrated:

  * z-scores of the scalar parameters (ring radius and width, gap radius, width and depth,
    inclination, position angle, disk flux, planet flux and offsets): (MAP - truth) / sigma_Laplace;
  * the IMAGE harmonics I_m (complex, m = 1..M), which are what the data constrain: the
    latents w are not (reflecting a root of the polynomial leaves the image unchanged), so the
    Laplace samples are mapped to I_m and the truth is checked against the marginal central
    68 and 95 percent intervals of Re I_m and Im I_m;
  * coverage of the true radial profile in the pointwise 68 / 95 percent bands of the
    Laplace samples.

PASS CRITERIA (fixed before any run; N = number of (injection, quantity) pairs)
  K1  coverage of the 68 percent interval within 3 binomial sigma of 0.68 and of the 95 percent
      interval at least 0.95 - 3 sqrt(0.95 * 0.05 / N);
  K2  the root-mean-square z-score within 3 sigma of 1 (1 +- 3 / sqrt(2 N));
  K3  at least 90 percent of the fits converge (no warning, ``info['stop']`` not a limit);
  K4  the MAP's loss is no worse than at the truth for every converged fit (the optimizer found
      the global mode that the injection defines; a failure is a multi-modality finding, not a
      calibration one).
The laptop configuration (``--config laptop``) has few injections: its criteria are
statistically weak (N is small), so it checks that the pipeline runs and is not grossly wrong.
Its starting point is the truth plus a small perturbation (local convergence). The full
configuration (``--config full``, for OzSTAR; see benchmarks/polar_disk/README_v4.md) uses
many injections, both instruments and multi-start initialization. NOTHING here is launched
by the repository's tests beyond the laptop configuration, and none of it needs a cluster.

    .venv/bin/python benchmarks/polar_disk/v4_injection_recovery.py --config laptop
"""

import argparse
import time
import warnings

import equinox as eqx
import jax
import numpy as np
import numpyro.distributions as dist

from common import save
from virgil import (Gap, GaussianRing, Harmonics, PointSource, PolarDisk, System, coverage, fit, laplace_cov,
                    model_loglike)

CONFIGS = {
    # tiny: laptop, MAP + Laplace, a few injections
    "laptop": dict(n_inj=5, instruments=("vlti",), n_r=24, degrees=(2,), seed=7, perturb=0.03, multistart=1),
    # the real run: OzSTAR, Ben launches it (not run here)
    "full": dict(n_inj=200, instruments=("nrm", "vlti"), n_r=64, degrees=(1, 2, 3), seed=7, perturb=0.3, multistart=8),
}
# nominal scene per instrument (mas): ring radius, ring sigma, gap radius, gap width, planet separation
SCALE = {"nrm": dict(r0=45.0, sigma=8.0, rgap=45.0, wgap=6.0, planet=70.0),
         "vlti": dict(r0=9.0, sigma=1.5, rgap=9.0, wgap=1.2, planet=14.0)}


def make_scene(par, w, n_r):
    """par: dict of scalars; w: complex latents (length M)."""
    disk = PolarDisk(
        Gap(GaussianRing(par["r0"], par["sigma"]), r0=par["rgap"], width=par["wgap"], depth=par["depth"]),
        harmonics=Harmonics(w=tuple(complex(x) for x in w)), inc=par["inc"], pa=par["pa"], n_r=n_r, flux=par["fdisk"])
    return System(star=PointSource(1.0), disk=disk,
                  planet=PointSource(par["fplanet"], dra=par["pdra"], ddec=par["pddec"]))


def draw_truth(rng, instrument, degree):
    s = SCALE[instrument]
    par = dict(
        r0=s["r0"] * rng.uniform(0.8, 1.2), sigma=s["sigma"] * rng.uniform(0.7, 1.3),
        rgap=s["rgap"] * rng.uniform(0.9, 1.15), wgap=s["wgap"] * rng.uniform(0.7, 1.3),
        depth=rng.uniform(0.3, 0.9), inc=rng.uniform(20.0, 65.0), pa=rng.uniform(0.0, 180.0),
        fdisk=rng.uniform(0.3, 1.0), fplanet=10 ** rng.uniform(-2.3, -1.3))
    ang = rng.uniform(0, 2 * np.pi)
    par["pdra"], par["pddec"] = s["planet"] * np.sin(ang) * rng.uniform(0.8, 1.2), s["planet"] * np.cos(ang) * rng.uniform(0.8, 1.2)
    w = (rng.normal(size=degree) + 1j * rng.normal(size=degree)) * 0.4 / np.arange(1, degree + 1)
    return par, w


def priors(instrument, degree):
    s = SCALE[instrument]
    p = {
        "disk.profile.base.r0": dist.LogUniform(0.3 * s["r0"], 3 * s["r0"]),
        "disk.profile.base.sigma": dist.LogUniform(0.2 * s["sigma"], 4 * s["sigma"]),
        "disk.profile.r0": dist.LogUniform(0.3 * s["r0"], 3 * s["r0"]),
        "disk.profile.width": dist.LogUniform(0.2 * s["wgap"], 4 * s["wgap"]),
        "disk.profile.depth": dist.Uniform(0.0, 1.0),
        "disk.inc": dist.Uniform(5.0, 80.0), "disk.pa": dist.Uniform(-10.0, 190.0),
        "disk.flux": dist.LogUniform(0.05, 5.0),
        "planet.flux": dist.LogUniform(1e-3, 0.3),
        "planet.dra": dist.Uniform(-3 * s["planet"], 3 * s["planet"]),
        "planet.ddec": dist.Uniform(-3 * s["planet"], 3 * s["planet"]),
    }
    # array-valued paths (``disk.harmonics.re``) fail in this virgil commit: see the README findings
    for k in ("re", "im"):
        for j in range(degree):
            p[f"disk.harmonics.{k}.{j}"] = dist.Normal(0.0, 1.0)
    return p


SCALARS = {"disk.profile.base.r0": "r0", "disk.profile.base.sigma": "sigma", "disk.profile.r0": "rgap",
           "disk.profile.width": "wgap", "disk.profile.depth": "depth", "disk.inc": "inc", "disk.pa": "pa",
           "disk.flux": "fdisk", "planet.flux": "fplanet", "planet.dra": "pdra", "planet.ddec": "pddec"}


def observe(instrument, truth, key):
    if instrument == "nrm":
        template = coverage.nrm_oidata(wavelength_m=4.3e-6, sigma_v2=0.01, sigma_cp_deg=0.5)
    else:
        template = coverage.vlti_oidata(sigma_v2=0.03, sigma_cp_deg=1.0, hour_angles_h=(-2.0, 0.0, 2.0))
    return template.with_model(truth, key=key)


def image_h(scene, degree):
    return np.asarray(scene.disk.harmonics.image_harmonics())[1:degree + 1]


def band(samples, truth, lo, hi):
    """Truth inside the central [lo, hi] percentile interval of the samples, per column."""
    a, b = np.percentile(samples, [lo, hi], axis=0)
    return (truth >= a) & (truth <= b)


def one_injection(i, cfg, instrument, degree, rng):
    par, w = draw_truth(rng, instrument, degree)
    truth = make_scene(par, w, cfg["n_r"])
    data = observe(instrument, truth, jax.random.key(1000 + i))
    pri = priors(instrument, degree)
    names = list(pri)
    # start: the truth perturbed (laptop), or in the full run a prior-scale spread over several starts
    best = None
    for k in range(cfg["multistart"]):
        par0 = {key: val * (1 + cfg["perturb"] * rng.normal()) if key not in ("pa",) else val + 20 * cfg["perturb"] * rng.normal()
                for key, val in par.items()}
        par0["depth"] = float(np.clip(par0["depth"], 0.05, 0.95))
        w0 = w + cfg["perturb"] * (rng.normal(size=degree) + 1j * rng.normal(size=degree))
        start = make_scene(par0, w0, cfg["n_r"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = fit(start, pri, data)
        warned = len(caught)
        info = res.info
        stop, converged, loss = str(info.get("stop")), bool(info.get("converged")), float(info.get("loss", np.nan))
        if best is None or loss < best["loss"]:
            best = dict(res=res, loss=loss, converged=converged, stop=stop)
    res = best["res"]
    values = np.asarray([res.model.get(n) for n in names], float)
    cov = np.asarray(laplace_cov(values, names, res.model, data))
    sig = np.sqrt(np.diag(cov))
    z = {}
    for n, v, s in zip(names, values, sig):
        if n in SCALARS:
            z[SCALARS[n]] = float((v - par[SCALARS[n]]) / s)
    # image harmonics and the radial profile from Laplace samples
    draws = np.random.default_rng(10_000 + i).multivariate_normal(values, cov + 1e-12 * np.eye(len(values)), size=200)
    ih_t = image_h(truth, degree)
    ih = []
    prof = []
    rg = np.linspace(0.3 * SCALE[instrument]["r0"], 2.0 * SCALE[instrument]["r0"], 40)
    prof_t = np.asarray(truth.disk.profile.intensity(rg, 1.0)) if hasattr(truth.disk.profile, "intensity") else None
    for d in draws:
        m = res.model
        for n, x in zip(names, d):
            m = m.set(n, x)
        ih.append(image_h(m, degree))
        if prof_t is not None:
            prof.append(np.asarray(m.disk.profile.intensity(rg, 1.0)))
    ih = np.asarray(ih)
    cover = {"ih68": [], "ih95": []}
    for lo, hi, key in ((16, 84, "ih68"), (2.5, 97.5, "ih95")):
        for part in (np.real, np.imag):
            cover[key] += band(part(ih), part(ih_t), lo, hi).tolist()
    out = {"i": i, "instrument": instrument, "degree": degree, "z": z, "converged": best["converged"], "stop": best["stop"],
           "loss_map": best["loss"], "image_harmonics_coverage": cover,
           "loglike_map_minus_truth": float(model_loglike(res.model, data) - model_loglike(truth, data))}
    if prof:
        prof = np.asarray(prof)
        out["profile_coverage"] = {"p68": float(band(prof, prof_t, 16, 84).mean()), "p95": float(band(prof, prof_t, 2.5, 97.5).mean())}
    return out


def summarize(rows):
    z = np.array([v for r in rows for v in r["z"].values()])
    n = len(z)
    c68 = float(np.mean(np.abs(z) <= 1.0))  # Gaussian z: |z| <= 1 is the central 68 percent
    c95 = float(np.mean(np.abs(z) <= 1.96))
    ih68 = np.array([x for r in rows for x in r["image_harmonics_coverage"]["ih68"]])
    ih95 = np.array([x for r in rows for x in r["image_harmonics_coverage"]["ih95"]])
    rms = float(np.sqrt(np.mean(z**2)))
    tol68 = 3 * np.sqrt(0.68 * 0.32 / n)
    tol95 = 3 * np.sqrt(0.95 * 0.05 / n)
    return {
        "n_pairs": n, "n_injections": len(rows),
        "scalar_coverage_68": c68, "scalar_coverage_95": c95, "z_rms": rms,
        "image_harmonic_coverage_68": float(ih68.mean()), "image_harmonic_coverage_95": float(ih95.mean()),
        "K1_pass": bool(abs(c68 - 0.68) <= tol68 and c95 >= 0.95 - tol95),
        "K2_pass": bool(abs(rms - 1.0) <= 3 / np.sqrt(2 * n)),
        "K3_converged_fraction": float(np.mean([r["converged"] for r in rows])),
        "K3_pass": bool(np.mean([r["converged"] for r in rows]) >= 0.9),
        "K4_pass": bool(all(r["loglike_map_minus_truth"] >= -1e-6 for r in rows if r["converged"])),
        "K4_worst_loglike_map_minus_truth": float(min(r["loglike_map_minus_truth"] for r in rows)),
        "tolerances": {"cov68": tol68, "cov95_floor": 0.95 - tol95, "z_rms": 3 / np.sqrt(2 * n)},
    }


def run(config="laptop", instrument=None, degree_only=None, seed_offset=0):
    cfg = dict(CONFIGS[config], seed=CONFIGS[config]["seed"] + seed_offset)
    if instrument:
        cfg["instruments"] = (instrument,)
    if degree_only:
        cfg["degrees"] = (degree_only,)
    rng = np.random.default_rng(cfg["seed"])
    rows, t0 = [], time.time()
    for inst in cfg["instruments"]:
        for degree in cfg["degrees"]:
            for i in range(cfg["n_inj"]):
                rows.append(one_injection(i, cfg, inst, degree, rng))
                print(inst, degree, i, {k: round(v, 2) for k, v in rows[-1]["z"].items()}, flush=True)
    summary = summarize(rows)
    summary["seconds"] = time.time() - t0
    return {"config_name": config, "config": cfg, "summary": summary, "rows": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=sorted(CONFIGS), default="laptop")
    ap.add_argument("--instrument", choices=("nrm", "vlti"), help="restrict to one instrument (array tasks)")
    ap.add_argument("--degree", type=int, help="restrict to one harmonic degree M")
    ap.add_argument("--seed-offset", type=int, default=0, help="added to the config seed")
    a = ap.parse_args()
    out = run(a.config, a.instrument, a.degree, a.seed_offset)
    tag = "".join([f"_{a.instrument}" if a.instrument else "", f"_M{a.degree}" if a.degree else "",
                   f"_s{a.seed_offset}" if a.seed_offset else ""])
    print(save(f"v4_{out['config_name']}{tag}", out))
    print(out["summary"])
