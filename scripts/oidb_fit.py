"""Stage O1 of design/plan_oidb.md: fit the authors' own OiDB OIFITS with virgil.

One collection per call (one OzSTAR array task per collection, ozstar_scripts job
``oidb_o1_fits``). Each recipe uses the model the reference publication used
(``oidb/references/<id>.json``) and writes ``<out>/fit_<id>.json``, which
``scripts/oidb_compare.py`` turns into the comparison table. The data are never
in git: they are on OzSTAR under ``/fred/oz440/bpope/oidb/<id>/``.

Models, by collection (the reasons are in RECIPES' docstrings):

* ``782185b2`` Gl 229 Ba-Bb, GRAVITY: two point stars, closure phases only
  (2.05-2.18 um, as Xuan et al. 2024); per-night positions, then a Keplerian
  orbit started with ``start_from_positions(..., scales="marginal")``.
* ``696baf06`` HR 6819, GRAVITY HR: **not clean** (Be decretion disk). Two point
  stars on the K-band continuum (Br-gamma and He I windows left out, where the
  paper's Keplerian disk contributes), per epoch and as an orbit. Reported, not
  scored.
* ``647a22a9`` CHARA workshop 2023, **L2**: iota Peg (the same seven files as
  fac164e1's 2018-10-22 night) and sigma Ori Aa-Ab as a "scaled binary" (Aa, Ab
  and the incoherent light of B, Schaefer et al. 2016). Fitted, but its results
  are not published until the dataPI has been contacted.
* ``fac164e1`` iota Peg, MIRC-X: a binary of two uniform disks fixed at 1.05 and
  0.6 mas (Anugu et al. 2020), per night and as an orbit.
* ``bda75673`` A-star companions, MIRC-X: the five detections of De Furio et al.
  (2022) with CANDID's model (uniform-disk primary, companion, resolved flux,
  bandwidth smearing at the file's resolving power). Detection limits are not
  part of O1.
* ``f4afc4cd`` HD 45166, GRAVITY: two point stars (qWR primary, B7 V companion)
  on the continuum, emission-line windows left out.
* ``19f7e2cf`` pi1 Gru, PIONIER: a uniform disk fitted to V^2 (LitPro's model in
  Paladini et al. 2018).

Priors are Jeffreys priors under the relevant group: log-uniform scales (fluxes,
diameters, periods, semimajor axes, error scales), uniform locations (positions)
and angles (``AngleVector``: omega, Omega and the mean anomaly at t_ref), uniform
cos i; e uniform on [0, 0.95]. Every fit reports the raw chi2/N on the quoted
errors first; the fitted error scales follow and are never a pass criterion.

Usage: python scripts/oidb_fit.py <collection id> <data dir> <out dir> [--quick] [--no-nuts]
"""

import argparse
import importlib.metadata as md
import json
import os
import re
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from astropy.io import fits  # noqa: E402
from virgil.angles import AngleVector  # noqa: E402
from virgil.epochs import Epochs, epoch_positions, start_from_positions  # noqa: E402
from virgil.fitting import fit  # noqa: E402
from virgil.inference import laplace_cov  # noqa: E402
from virgil.likelihood import chain_init_params, numpyro_model, whitened_residuals  # noqa: E402
from virgil.models import (  # noqa: E402
    Attached, BinaryModelCartesian, OrbitalBinary, SourceModel, System, UniformDisk)
from virgil.oidata import OIData  # noqa: E402
from virgil.oifits import read_oifits  # noqa: E402
from virgil.orbits import KeplerOrbit  # noqa: E402

UM = 1e-6
SCALE_PRIOR = dist.LogUniform(0.1, 100.0)  # error scales: Jeffreys, bounds wide enough to show a failed fit


# ----------------------------------------------------------------------------- reading

def oifits_info(path):
    """INSNAMEs, targets and median resolving power of an OIFITS file (astropy only)."""
    with fits.open(path) as hdul:
        ins, resolving, targets, rows = set(), {}, {}, {}
        for h in hdul:
            name = h.header.get("EXTNAME")
            if name == "OI_WAVELENGTH":
                key = h.header.get("INSNAME")
                ins.add(key)
                wl, band = np.asarray(h.data["EFF_WAVE"], float), np.asarray(h.data["EFF_BAND"], float)
                good = band > 0
                resolving[key] = float(np.median(wl[good] / band[good])) if good.any() else float("inf")
            elif name == "OI_TARGET":
                targets = {int(i): str(t).strip() for i, t in zip(h.data["TARGET_ID"], h.data["TARGET"])}
            elif name in ("OI_VIS2", "OI_T3"):
                for i in np.asarray(h.data["TARGET_ID"]):
                    rows[int(i)] = rows.get(int(i), 0) + 1
    return dict(insnames=sorted(ins), resolving=resolving, targets=targets, rows=rows)


def pick_target(path, pattern=None):
    """The target to read: None when the file has one; else the first name matching
    ``pattern``, or the target with the most V^2/T3 rows."""
    info = oifits_info(path)
    names = info["targets"]
    if len(names) <= 1:
        return None
    if pattern:
        hits = [n for n in names.values() if re.search(pattern, n)]
        if not hits:
            raise ValueError(f"{path}: no target matches {pattern!r} in {sorted(names.values())}")
        return hits[0]
    return names[max(info["rows"], key=info["rows"].get)]


def load(paths, *, insname_prefix=None, target=None, select=None):
    """One OIData from one or more files: the tables whose INSNAME starts with
    ``insname_prefix`` (GRAVITY: the science channel, both polarisations), the target
    matching ``target`` (a regex), then ``OIData.select(**select)``."""
    paths = list(paths)
    insname = None
    if insname_prefix:
        insname = sorted({n for p in paths for n in oifits_info(p)["insnames"] if n.startswith(insname_prefix)})
        if not insname:
            raise ValueError(f"no INSNAME starting {insname_prefix!r} in {paths}")
    name = pick_target(paths[0], target)
    data = OIData(read_oifits(paths if len(paths) > 1 else paths[0], target=name, insname=insname))
    return data.select(**select) if select else data


def resolving_power(paths, insname_prefix=None):
    rs = [r for p in paths for k, r in oifits_info(p)["resolving"].items()
          if insname_prefix is None or k.startswith(insname_prefix)]
    return float(np.median(rs))


def mean_mjd(data):
    return float(np.mean(np.asarray(data.mjd)))


# ----------------------------------------------------------------------------- models

class Smeared(SourceModel):
    """Bandwidth smearing: a scene's complex visibility averaged over a top-hat channel
    of width lambda/R, as ``n_sub`` sub-channels (u, v in metres, so the spatial
    frequency of each sub-channel is u/lambda_k). Flat spectrum across the channel."""

    scene: SourceModel
    resolving: float = eqx.field(static=True)
    n_sub: int = eqx.field(static=True)

    def __init__(self, scene, resolving, n_sub=7):
        self.scene, self.resolving, self.n_sub = scene, float(resolving), int(n_sub)

    def model(self, u, v, wavel):
        offsets = (np.arange(self.n_sub) + 0.5) / self.n_sub - 0.5
        return sum(self.scene.model(u, v, wavel * (1.0 + o / self.resolving)) for o in offsets) / self.n_sub


class CandidBinary(SourceModel):
    """CANDID's binary (Gallenne et al. 2015, as De Furio et al. 2022 describe it): a
    uniform-disk primary (flux 1), a uniform-disk companion of flux ``flux`` at (dra, ddec),
    and a fully resolved flux ``resolved`` (relative to the primary; may be negative, as
    De Furio et al. report for HD 5448) that only adds to the normalisation:
    V = (V1 + f V2) / (1 + f + r)."""

    ud1: jax.Array
    ud2: jax.Array
    dra: jax.Array
    ddec: jax.Array
    flux: jax.Array
    resolved: jax.Array

    def __init__(self, ud1, ud2, dra, ddec, flux, resolved=0.0):
        self.ud1, self.ud2 = jnp.asarray(ud1, float), jnp.asarray(ud2, float)
        self.dra, self.ddec = jnp.asarray(dra, float), jnp.asarray(ddec, float)
        self.flux, self.resolved = jnp.asarray(flux, float), jnp.asarray(resolved, float)

    def model(self, u, v, wavel):
        scene = System(primary=UniformDisk(self.ud1),
                       companion=UniformDisk(self.ud2, flux=self.flux, dra=self.dra, ddec=self.ddec))
        return scene.model(u, v, wavel) * (1.0 + self.flux) / (1.0 + self.flux + self.resolved)


def candid_scene(resolving=None, ud1=None, ud2=None, resolved=None):
    """A scene function of the free parameters; ``None`` arguments are fitted."""
    fixed = dict(ud1=ud1, ud2=ud2, resolved=resolved)

    def scene(**v):
        kw = {k: (v[k] if fixed[k] is None else fixed[k]) for k in fixed}
        m = CandidBinary(kw["ud1"], kw["ud2"], v["dra"], v["ddec"], v["flux"], kw["resolved"])
        return Smeared(m, resolving) if resolving else m
    return scene


def point_binary(**v):
    return BinaryModelCartesian(v["dra"], v["ddec"], v["flux"])


# ----------------------------------------------------------------------------- statistics

def noise_terms(data):
    """Error scales for the blocks the data have (Jeffreys, log-uniform)."""
    out = {}
    if np.asarray(data.vis).size:
        out["vis_scale"] = SCALE_PRIOR
    if np.asarray(data.phi).size:
        out["phi_scale"] = SCALE_PRIOR
    return out


def raw_chi2(model, data):
    """chi2/N on the quoted errors, per block and overall (N = independent observables)."""
    out = {}
    total, n_total = 0.0, 0
    for block in ("vis", "phi"):
        if not np.asarray(getattr(data, block)).size:
            continue
        d = data.select(observables=block)
        chi2 = float(np.sum(np.asarray(whitened_residuals(model, d)) ** 2))
        n = int(d.n_independent)
        out[block] = dict(chi2=chi2, n=n, chi2_red=chi2 / n)
        total, n_total = total + chi2, n_total + n
    out["all"] = dict(chi2=total, n=n_total, chi2_red=total / max(n_total, 1))
    return out


def scales_of(values, prefix="noise."):
    return {k[len(prefix):]: float(np.asarray(v)) for k, v in values.items() if k.startswith(prefix)}


def polar(dra, ddec, cov):
    """(rho, PA) with their 1-sigma errors from (dra, ddec) and its 2x2 covariance."""
    rho = float(np.hypot(dra, ddec))
    pa = float(np.degrees(np.arctan2(dra, ddec)) % 360.0)
    # d rho/d(dra, ddec) and d PA/d(dra, ddec) (PA = atan2(dra, ddec), radians)
    jr = np.array([dra, ddec]) / rho
    jp = np.array([ddec, -dra]) / rho ** 2
    cov = np.asarray(cov, float)
    return dict(rho_mas=[rho, float(np.sqrt(jr @ cov @ jr))],
                pa_deg=[pa, float(np.degrees(np.sqrt(jp @ cov @ jp)))])


# ----------------------------------------------------------------------------- one epoch

def fit_epoch(label, data, scene, priors, grid, *, extra=None):
    """Static fit of one epoch: grid start from ``epoch_positions`` (point binary,
    scale-marginalised), then ``fit`` of ``scene`` with free error scales, Laplace
    covariance on the data with the fitted scales, raw chi2/N on the quoted errors."""
    t0 = time.time()
    pos = epoch_positions(Epochs({label: data}), grid)
    start = dict(dra=float(pos.dra[0]), ddec=float(pos.ddec[0]), flux=float(pos.flux[0]))
    init = {}
    for k, prior in priors.items():
        x = start.get(k, (extra or {}).get(k))
        x = float(np.asarray(prior.mean)) if x is None else x
        lo, hi = float(np.asarray(prior.support.lower_bound)), float(np.asarray(prior.support.upper_bound))
        init[k] = float(np.clip(x, lo + 1e-3 * (hi - lo), hi - 1e-3 * (hi - lo)))  # strictly inside the prior
    noise = noise_terms(data)
    res = fit(scene, priors, data, noise=noise, init=init)
    values = {k: float(np.asarray(res.values[k])) for k in priors}
    scales = scales_of(res.values)
    scaled = data.with_error_scale({k.replace("_scale", ""): s for k, s in scales.items()})
    names = list(priors)
    cov = np.asarray(laplace_cov(np.array([values[k] for k in names]), names, scene, scaled))
    err = {k: float(np.sqrt(max(cov[i, i], 0.0))) for i, k in enumerate(names)}
    model = scene(**values)
    out = dict(label=label, mjd=mean_mjd(data), n_vis=int(np.asarray(data.vis).size),
               n_phi=int(np.asarray(data.phi).size), chi2_raw=raw_chi2(model, data), scales=scales,
               converged=bool(res.info.get("converged")), loss=float(np.asarray(res.info["loss"])),
               values={k: [values[k], err[k]] for k in names}, cov_names=names, cov=cov.tolist(),
               grid_start=dict(start, gap_marginal=float(pos.gap_marginal[0]),
                               chi2_raw=pos.chi2_raw[0], scale=pos.scale[0]),
               seconds=time.time() - t0)
    if "dra" in names and "ddec" in names:
        i, j = names.index("dra"), names.index("ddec")
        out["values"].update(polar(values["dra"], values["ddec"], cov[np.ix_([i, j], [i, j])]))
    return out


# ----------------------------------------------------------------------------- orbits

ORBIT_SITES = ("log_period", "phase", "ecc", "cos_inc", "omega", "Omega", "log_a_mas", "log_flux")


def orbit_priors(period, a_mas, flux=(0.02, 1.0)):
    """Jeffreys priors (bounds stated): log-uniform P, a and flux; uniform cos i, e on
    [0, 0.95]; uniform angles omega, Omega and the mean anomaly at t_ref (``phase``)."""
    return {"log_period": dist.Uniform(*np.log(period)), "phase": AngleVector(),
            "ecc": dist.Uniform(0.0, 0.95), "cos_inc": dist.Uniform(-1.0, 1.0),
            "omega": AngleVector(), "Omega": AngleVector(),
            "log_a_mas": dist.Uniform(*np.log(a_mas)), "log_flux": dist.Uniform(*np.log(flux))}


def kepler(v, t_ref):
    period = jnp.exp(v["log_period"])
    inc = jnp.degrees(jnp.arccos(jnp.clip(v["cos_inc"], -1.0, 1.0 - 1e-12)))
    return KeplerOrbit(period, -v["phase"] / 360.0 * period, v["ecc"], inc, v["omega"], v["Omega"],
                       jnp.exp(v["log_a_mas"]), t_ref=t_ref)


def orbital_point_binary(t_ref):
    def scene(**v):
        return OrbitalBinary(kepler(v, t_ref), jnp.exp(v["log_flux"]))
    return scene


def orbital_disk_binary(t_ref, ud1, ud2):
    """Two uniform disks of fixed diameters, the companion on the orbit."""
    def scene(**v):
        return System(primary=UniformDisk(ud1),
                      companion=Attached(UniformDisk(ud2, flux=jnp.exp(v["log_flux"])), kepler(v, t_ref)))
    return scene


def start_values_for(priors):
    lo_p, hi_p = (float(np.exp(priors["log_period"].low)), float(np.exp(priors["log_period"].high)))
    lo_a, hi_a = (float(np.exp(priors["log_a_mas"].low)), float(np.exp(priors["log_a_mas"].high)))
    lo_f, hi_f = (float(np.exp(priors["log_flux"].low)), float(np.exp(priors["log_flux"].high)))

    def start_values(orbit, flux):
        P = float(np.clip(float(orbit.period), lo_p * 1.0001, hi_p * 0.9999))
        return dict(log_period=float(np.log(P)), phase=float((-float(orbit.dt_peri) / P * 360.0) % 360.0),
                    ecc=float(np.clip(float(orbit.ecc), 0.005, 0.94)),
                    cos_inc=float(np.clip(np.cos(np.radians(float(orbit.inc))), -0.999, 0.999)),
                    omega=float(float(orbit.omega) % 360.0), Omega=float(float(orbit.Omega) % 360.0),
                    log_a_mas=float(np.log(np.clip(float(orbit.a_mas), lo_a * 1.001, hi_a * 0.999))),
                    log_flux=float(np.log(np.clip(flux, lo_f * 1.001, hi_f * 0.999))))
    return start_values


def elements(v, t_ref):
    """Physical elements from the sampled parameters (numpy, works on arrays)."""
    P = np.exp(np.asarray(v["log_period"]))
    return dict(period_day=P, t_peri_mjd=t_ref - np.asarray(v["phase"]) / 360.0 * P, ecc=np.asarray(v["ecc"]),
                inc_deg=np.degrees(np.arccos(np.clip(np.asarray(v["cos_inc"]), -1, 1))),
                omega_deg=np.asarray(v["omega"]) % 360.0, Omega_deg=np.asarray(v["Omega"]) % 360.0,
                a_mas=np.exp(np.asarray(v["log_a_mas"])), flux_ratio=np.exp(np.asarray(v["log_flux"])))


def _summary(x, angle=False):
    x = np.asarray(x, float).ravel()
    if angle:  # circular mean and spread, degrees
        z = np.mean(np.exp(1j * np.radians(x)))
        c = float(np.degrees(np.angle(z)) % 360.0)
        d = (x - c + 180.0) % 360.0 - 180.0
        return dict(median=float((c + np.median(d)) % 360.0), sd=float(np.std(d)),
                    p16=float((c + np.percentile(d, 16)) % 360.0), p84=float((c + np.percentile(d, 84)) % 360.0))
    return dict(median=float(np.median(x)), sd=float(np.std(x)), p16=float(np.percentile(x, 16)),
                p84=float(np.percentile(x, 84)))


def fit_orbit(start_data, fit_data, scene_factory, priors, *, grid, periods, eccs=None, n_candidates=200,
              n_refine=4, min_gap=5.0, nuts=None, seed=0):
    """Orbit from multi-epoch visibilities.

    1. ``start_from_positions(..., scales="marginal")`` on ``start_data`` (an Epochs, one
       dataset per night: positions are decisive there), error scales per dataset.
    2. A refit on ``fit_data`` (an Epochs with every file a dataset, each a snapshot at
       its own mean time) from each distinct start; the lowest loss is the answer.
    3. Optionally NUTS (``chain_method="vectorized"``) from the distinct modes.
    Reports the raw chi2/N of every dataset on its quoted errors."""
    t0 = time.time()
    t_ref = float(np.round(np.mean(start_data.times), 1))
    scene = scene_factory(t_ref)
    start = start_from_positions(
        scene, priors, start_data, start_values_for(priors), grid=grid, periods=periods, t_ref=t_ref,
        scales="marginal", noise=[noise_terms(d) for d in start_data.data], eccs=eccs,
        n_candidates=n_candidates, n_refine=n_refine, min_gap=min_gap)
    pos = start.positions
    positions = [dict(name=n, mjd=float(m), dra=float(x), ddec=float(y), cov=np.asarray(c).tolist(),
                      flux=float(f), gap_marginal=float(g), chi2_raw=cr, scale=sc,
                      **polar(float(x), float(y), c))
                 for n, m, x, y, c, f, g, cr, sc in zip(pos.names, pos.mjd, pos.dra, pos.ddec, pos.cov, pos.flux,
                                                        pos.gap_marginal, pos.chi2_raw, pos.scale)]
    model_fn = fit_data.model_fn(scene)
    noise = [noise_terms(d) for d in fit_data.data]
    refits = []
    for mode in start.modes():
        init = {k: mode.values[k] for k in priors}
        r = fit(model_fn, priors, list(fit_data.data), noise=noise, init=init)
        refits.append(r)
    refits.sort(key=lambda r: float(np.asarray(r.info["loss"])))
    best = refits[0]
    v = {k: float(np.asarray(best.values[k])) for k in priors}
    models = model_fn(**v)
    per_dataset = [dict(name=n, epoch=e, mjd=float(t), chi2_raw=raw_chi2(m, d),
                        scales=scales_of(best.values, f"noise[{i}]."))
                   for i, (n, e, t, m, d) in enumerate(zip(fit_data.dataset_names, fit_data.epoch_of,
                                                           fit_data.times, models, fit_data.data))]
    chi2 = sum(p["chi2_raw"]["all"]["chi2"] for p in per_dataset)
    n = sum(p["chi2_raw"]["all"]["n"] for p in per_dataset)
    out = dict(t_ref=t_ref, map={k: float(np.asarray(x)) for k, x in elements(v, t_ref).items()},
               map_sampled=v, loss=float(np.asarray(best.info["loss"])),
               modes=[dict(loss=float(np.asarray(r.info["loss"])),
                           elements={k: float(np.asarray(x)) for k, x in elements(
                               {k: float(np.asarray(r.values[k])) for k in priors}, t_ref).items()})
                      for r in refits],
               chi2_raw=dict(chi2=chi2, n=n, chi2_red=chi2 / n), datasets=per_dataset, positions=positions,
               n_candidates=n_candidates, seconds_map=time.time() - t0)
    if nuts:
        starts = [refits[k % len(refits)].values for k in range(nuts["chains"])]  # the distinct modes, best first
        out["nuts"] = run_nuts(model_fn, priors, fit_data, noise, starts, t_ref, nuts, seed)
    return out


def run_nuts(model_fn, priors, fit_data, noise, starts, t_ref, cfg, seed):
    from numpyro.diagnostics import summary
    from numpyro.infer import MCMC, NUTS

    t0 = time.time()
    post = numpyro_model(model_fn, priors, list(fit_data.data), noise=noise)
    init = chain_init_params(post, [dict(s) for s in starts], key=jax.random.PRNGKey(seed + 1))
    mcmc = MCMC(NUTS(post), num_warmup=cfg["warmup"], num_samples=cfg["samples"], num_chains=cfg["chains"],
                chain_method="vectorized", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), init_params=init, extra_fields=("diverging",))
    samples = {k: np.asarray(x) for k, x in mcmc.get_samples().items()}
    el = elements(samples, t_ref)
    grouped = mcmc.get_samples(group_by_chain=True)
    diag = summary({k: grouped[k] for k in ("log_period", "ecc", "cos_inc", "log_a_mas", "log_flux")})
    return dict(config=cfg, seconds=time.time() - t0,
                divergences=int(np.sum(np.asarray(mcmc.get_extra_fields()["diverging"]))),
                r_hat={k: float(np.max(d["r_hat"])) for k, d in diag.items()},
                n_eff={k: float(np.min(d["n_eff"])) for k, d in diag.items()},
                elements={k: _summary(x, angle=k in ("omega_deg", "Omega_deg")) for k, x in el.items()})


# ----------------------------------------------------------------------------- recipes

def _files(data_dir, pattern):
    names = sorted(f for f in os.listdir(data_dir) if re.search(pattern, f))
    if not names:
        raise FileNotFoundError(f"no file matching {pattern!r} in {data_dir}")
    return [os.path.join(data_dir, f) for f in names]


def _group_by(paths, key):
    out = {}
    for p in paths:
        out.setdefault(key(os.path.basename(p)), []).append(p)
    return dict(sorted(out.items()))


def _date_iso(s):
    return time.strftime("%Y-%m-%d", time.strptime(s, "%Y%b%d"))


def _grid(half, step, fluxes):
    axis = np.arange(-half, half + step / 2, step)
    return {"dra": axis, "ddec": axis, "flux": np.asarray(fluxes, float)}


def recipe_gl229(data_dir, cfg):
    """Gl 229 Ba-Bb (Xuan et al. 2024): closure phases only, 2.05-2.18 um, science
    channel; per-night positions and the orbit. Both are point sources (the paper's model)."""
    sel = dict(wavel_min=2.05 * UM, wavel_max=2.18 * UM, observables="phi")
    nights = _group_by(_files(data_dir, r"dualscivis\.fits$"), lambda f: f[6:16])
    per_file, per_night = {}, {}
    for night, paths in nights.items():
        per_night[night] = load(paths, insname_prefix="GRAVITY_SC", target=r"B$|_B\b", select=sel)
        per_file[night] = {os.path.basename(p): load([p], insname_prefix="GRAVITY_SC", target=r"B$|_B\b", select=sel)
                           for p in paths}
    grid = _grid(12.0, cfg.get("step", 0.2), np.linspace(0.2, 1.0, 9))
    priors = {"dra": dist.Uniform(-15.0, 15.0), "ddec": dist.Uniform(-15.0, 15.0), "flux": dist.LogUniform(0.05, 1.0)}
    epochs = [fit_epoch(n, d, point_binary, priors, grid) for n, d in per_night.items()]
    orbit = fit_orbit(Epochs(per_night), Epochs(per_file), orbital_point_binary,
                      orbit_priors((11.5, 12.8), (2.0, 20.0), (0.05, 1.0)), grid=grid,
                      periods=np.arange(11.8, 12.5, cfg.get("dp", 0.0005)), eccs=np.arange(0.0, 0.55, 0.05),
                      n_candidates=cfg.get("n_candidates", 400), n_refine=cfg.get("n_refine", 6),
                      nuts=cfg.get("nuts"))
    return dict(model="two point sources; closure phases 2.05-2.18 um; Keplerian orbit (OrbitalBinary)",
                epochs=epochs, orbit=orbit)


HR6819_LINES = [(2.050 * UM, 2.066 * UM), (2.105 * UM, 2.120 * UM), (2.150 * UM, 2.182 * UM)]


def recipe_hr6819(data_dir, cfg):
    """HR 6819 (Klement et al. 2025): NOT CLEAN. Two point stars (the paper fixed both
    at 0.15 mas UD, unresolved at 3.4 mas resolution) on the K-band continuum, V^2 and
    closure phases; Br-gamma and He I windows left out, where the paper's Keplerian
    Be disk contributes. Positions per epoch and the orbit; reported, not scored."""
    sel = dict(ranges=[(2.02 * UM, 2.40 * UM)], exclude=HR6819_LINES)
    paths = _files(data_dir, r"SCI_VIS_CALIBRATED.*\.fits$")
    per_epoch = {_date_iso(os.path.basename(p)[:9]): load([p], insname_prefix="GRAVITY_SC", select=sel) for p in paths}
    per_epoch = dict(sorted(per_epoch.items()))
    grid = _grid(2.5, cfg.get("step", 0.05), np.linspace(0.3, 1.0, 8))
    priors = {"dra": dist.Uniform(-3.0, 3.0), "ddec": dist.Uniform(-3.0, 3.0), "flux": dist.LogUniform(0.05, 1.0)}
    epochs = [fit_epoch(n, d, point_binary, priors, grid) for n, d in per_epoch.items()]
    orbit = fit_orbit(Epochs(per_epoch), Epochs(per_epoch), orbital_point_binary,
                      orbit_priors((30.0, 50.0), (0.3, 5.0), (0.05, 1.0)), grid=grid,
                      periods=np.arange(39.5, 41.5, cfg.get("dp", 0.001)), eccs=np.arange(0.0, 0.35, 0.05),
                      n_candidates=cfg.get("n_candidates", 400), n_refine=cfg.get("n_refine", 6),
                      nuts=cfg.get("nuts"))
    return dict(model="two point stars on the K continuum (lines excluded); not clean: Be decretion disk",
                flags=["not-clean"], epochs=epochs, orbit=orbit)


IOTA_PEG_UD = (1.05, 0.6)  # mas, fixed by Anugu et al. 2020 (Sect. 5.6)


def _iota_peg_epochs(paths_by_night, cfg):
    scene = candid_scene(ud1=IOTA_PEG_UD[0], ud2=IOTA_PEG_UD[1], resolved=0.0)
    priors = {"dra": dist.Uniform(-15.0, 15.0), "ddec": dist.Uniform(-15.0, 15.0), "flux": dist.LogUniform(0.02, 1.0)}
    grid = _grid(12.0, cfg.get("step", 0.1), np.geomspace(0.05, 1.0, 8))
    data = {n: load(p) for n, p in paths_by_night.items()}
    return [fit_epoch(n, d, scene, priors, grid) for n, d in data.items()], data, grid


def recipe_iota_peg(data_dir, cfg):
    """iota Peg (Anugu et al. 2020): two uniform disks fixed at 1.05 and 0.6 mas, V^2 and
    closure phases, per night; then the orbit with the same disks. Bandwidth smearing at
    10 mas is below 0.5% at R ~ 190 and is not modelled."""
    nights = _group_by(_files(data_dir, r"mircx\d+_oifits_viscal\.fits$"), lambda f: f[:10])
    epochs, per_night, grid = _iota_peg_epochs(nights, cfg)
    per_file = {n: {os.path.basename(p): load([p]) for p in ps} for n, ps in nights.items()}
    orbit = fit_orbit(Epochs(per_night), Epochs(per_file),
                      lambda t: orbital_disk_binary(t, *IOTA_PEG_UD),
                      orbit_priors((9.0, 11.5), (3.0, 30.0), (0.02, 1.0)), grid=grid,
                      periods=np.arange(10.0, 10.4, cfg.get("dp", 0.0002)), eccs=np.arange(0.0, 0.3, 0.05),
                      n_candidates=cfg.get("n_candidates", 400), n_refine=cfg.get("n_refine", 6),
                      nuts=cfg.get("nuts"))
    return dict(model="two uniform disks fixed at 1.05 and 0.6 mas; V^2 + closure phases; Keplerian orbit",
                epochs=epochs, orbit=orbit)


def recipe_workshop(data_dir, cfg):
    """CHARA imaging workshop 2023 (L2). iota Peg 2018-10-22: as fac164e1 (the files are
    byte-identical, sha256 in the manifest). sigma Ori Aa-Ab (Schaefer et al. 2016):
    the "scaled binary": point-like Aa and Ab (O stars, ~0.1 mas) and B's light,
    incoherent in MIRC, as resolved flux; without smearing (the reported solution) and
    with smearing at the file's resolving power (a variant)."""
    nights = _group_by(_files(data_dir, r"mircx\d+_oifits_viscal\.fits$"), lambda f: f[:10])
    iota, _, _ = _iota_peg_epochs(nights, cfg)
    (sig,) = _files(data_dir, r"sig_Ori.*oifits$")
    data = load([sig])
    priors = {"dra": dist.Uniform(-10.0, 10.0), "ddec": dist.Uniform(-10.0, 10.0), "flux": dist.LogUniform(0.05, 1.0),
              "resolved": dist.LogUniform(0.01, 3.0)}
    grid = _grid(9.0, cfg.get("step", 0.1), np.geomspace(0.1, 1.0, 8))
    sigma = fit_epoch("2011-09-29", data, candid_scene(ud1=0.01, ud2=0.01), priors, grid, extra={"resolved": 0.5})
    smeared = fit_epoch("2011-09-29", data, candid_scene(resolving_power([sig]), ud1=0.01, ud2=0.01), priors, grid,
                        extra={"resolved": 0.5})
    for e in (sigma, smeared):  # fractions of the total light, as Schaefer et al.'s Table 3
        f, r = e["values"]["flux"][0], e["values"]["resolved"][0]
        e["fractions"] = dict(fAa=1 / (1 + f + r), fAb=f / (1 + f + r), fB=r / (1 + f + r))
    smeared["variant"] = "bandwidth smearing at R = %.0f" % resolving_power([sig])
    return dict(model="iota Peg: two UDs fixed at 1.05/0.6 mas; sigma Ori: scaled binary (Aa, Ab, B incoherent)",
                flags=["L2: dataPI (Gail Schaefer) to be contacted before results are presented"],
                targets={"iot Peg": dict(epochs=iota), "sig Ori": dict(epochs=[sigma], variants=[smeared])})


# De Furio et al. 2022 Table 3: the five detections. UD2 free where the paper fitted it.
ASTAR_DETECTIONS = {"HD 5448": False, "HD 11636": True, "HD 28910": True, "HD 29388": False, "HD 48097": False}


def recipe_astars(data_dir, cfg):
    """A-star companions (De Furio et al. 2022): CANDID's model on V^2 + closure phases
    (UD primary, companion UD where the paper fitted one, else 0.01 mas; resolved flux,
    signed; bandwidth smearing at the file's resolving power). Grid +-100 mas for the
    start (CANDID searched closure phases out to 300 mas; all detections are within 65)."""
    out = {}
    half = cfg.get("astar_half", 100.0)
    for star, ud2_free in ASTAR_DETECTIONS.items():
        path = os.path.join(data_dir, star.replace(" ", "_") + ".oifits")
        data = load([path])
        priors = {"dra": dist.Uniform(-half - 5, half + 5), "ddec": dist.Uniform(-half - 5, half + 5),
                  "flux": dist.LogUniform(1e-3, 1.0), "ud1": dist.LogUniform(0.05, 3.0),
                  "resolved": dist.Uniform(-0.3, 0.3)}
        if ud2_free:
            priors["ud2"] = dist.LogUniform(0.01, 3.0)
        scene = candid_scene(resolving_power([path]), ud2=None if ud2_free else 0.01)
        grid = _grid(half, cfg.get("step", 0.5), np.geomspace(0.005, 1.0, 8))
        e = fit_epoch(star, data, scene, priors, grid, extra={"ud1": 0.5, "ud2": 0.3, "resolved": 0.0})
        e["resolving_power"] = resolving_power([path])
        out[star] = dict(epochs=[e])
    return dict(model="CANDID binary: UD primary, UD/point companion, resolved flux, bandwidth smearing",
                targets=out, not_fitted="the 22 non-detections (detection limits are not part of O1)")


HD45166_LINES = [(2.030 * UM, 2.045 * UM), (2.050 * UM, 2.066 * UM), (2.068 * UM, 2.090 * UM),
                 (2.100 * UM, 2.120 * UM), (2.155 * UM, 2.195 * UM), (2.335 * UM, 2.355 * UM)]


def recipe_hd45166(data_dir, cfg):
    """HD 45166 (Deshmukh et al. 2025): two point stars (qWR primary with emission lines,
    B7 V companion), V^2 and closure phases on the continuum with the qWR's He, H, C and
    N windows left out. The scored fit uses the pipeline-calibrated science files; the
    paper's adopted values are the mean of four calibrations (its calibrator is a binary);
    ``output_zpcal.fits`` is fitted as a variant."""
    sel = dict(ranges=[(2.02 * UM, 2.40 * UM)], exclude=HD45166_LINES)
    priors = {"dra": dist.Uniform(-20.0, 20.0), "ddec": dist.Uniform(-20.0, 20.0), "flux": dist.LogUniform(0.05, 1.0)}
    grid = _grid(20.0, cfg.get("step", 0.2), np.linspace(0.3, 1.0, 8))
    sci = load(_files(data_dir, r"singlesciviscalibrated\.fits$"), insname_prefix="GRAVITY_SC",
               target="HD.?45166", select=sel)
    main = fit_epoch("2023-11-26", sci, point_binary, priors, grid)
    variants = []
    zp = [p for p in _files(data_dir, r"\.fits$") if os.path.basename(p) == "output_zpcal.fits"]
    if zp:
        try:
            d = load(zp, insname_prefix="GRAVITY_SC", target="HD.?45166", select=sel)
            variants.append(dict(fit_epoch("2023-11-26", d, point_binary, priors, grid), variant="output_zpcal.fits"))
        except Exception as exc:  # a variant must not lose the main fit
            variants.append(dict(variant="output_zpcal.fits", error=f"{type(exc).__name__}: {exc}"))
    return dict(model="two point stars on the K continuum (qWR lines excluded)", epochs=[main], variants=variants)


def recipe_pi1gru(data_dir, cfg):
    """pi1 Gru (Paladini et al. 2018): a uniform disk on all V^2 (LitPro's fit), started
    from a 1-D chi2 scan (the V^2 of a UD has several minima). Variant: V^2 below the
    first null of the fitted disk only."""
    from virgil.likelihood import model_loglike

    (path,) = _files(data_dir, r"\.fits$")
    rec = dict(read_oifits(path))
    rec["phi_flag"] = np.ones_like(np.asarray(rec["phi_flag"]), dtype=bool)  # V^2 only, as LitPro's UD fit
    data = OIData(rec)
    scan = np.arange(8.0, 30.0, 0.05)
    start = float(scan[int(np.argmax([float(model_loglike(UniformDisk(d), data)) for d in scan]))])

    def ud_fit(d, label):
        priors = {"diam": dist.LogUniform(5.0, 40.0)}
        scene = lambda **v: UniformDisk(v["diam"])  # noqa: E731
        res = fit(scene, priors, d, noise={"vis_scale": SCALE_PRIOR}, init={"diam": start})
        diam = float(np.asarray(res.values["diam"]))
        s = scales_of(res.values)
        cov = np.asarray(laplace_cov(np.array([diam]), ["diam"], scene, d.with_error_scale({"vis": s["vis_scale"]})))
        return dict(label=label, mjd=mean_mjd(d), n_vis=int(np.asarray(d.vis).size), scales=s,
                    chi2_raw=raw_chi2(UniformDisk(diam), d), values={"ud_mas": [diam, float(np.sqrt(cov[0, 0]))]},
                    converged=bool(res.info.get("converged")), scan_start=start)

    main = ud_fit(data, "all V2")
    # spatial frequency of every sample (rad^-1); the first null of a UD is at 1.2197 / theta
    wl = np.broadcast_to(np.asarray(rec["wavel"], float), np.shape(rec["u"]))
    freq = np.hypot(np.asarray(rec["u"], float), np.asarray(rec["v"], float)) / wl
    first = dict(rec, vis_flag=np.asarray(rec["vis_flag"], bool) | (freq >= 1.2197 / np.radians(
        main["values"]["ud_mas"][0] / 3.6e6)))
    variants = [dict(ud_fit(OIData(first), "first lobe"), variant="V2 below the first null only")]
    return dict(model="uniform disk on V^2", parametric=main, variants=variants)


RECIPES = {
    "782185b2-0727-42b0-a185-b2072732b047": recipe_gl229,
    "696baf06-6c3c-424d-abaf-066c3c324d99": recipe_hr6819,
    "647a22a9-5047-4220-ba22-a95047022072": recipe_workshop,
    "fac164e1-d9d0-4500-8164-e1d9d0450099": recipe_iota_peg,
    "bda75673-61c6-49f0-a756-7361c699f0c4": recipe_astars,
    "f4afc4cd-fd31-40d3-afc4-cdfd3150d340": recipe_hd45166,
    "19f7e2cf-2a03-4bb2-b7e2-cf2a03bbb245": recipe_pi1gru,
}
NUTS = dict(warmup=500, samples=1000, chains=4)


def provenance():
    out = {"python": sys.version.split()[0]}
    for p in ("virgil-astro", "jax", "jaxlib", "numpyro", "numpy"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    out["virgil_commit"] = os.environ.get("VIRGIL_COMMIT")
    out["validation_commit"] = os.environ.get("VALIDATION_COMMIT")
    return out


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.ndarray, jax.Array)):
        return _jsonable(np.asarray(x).tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def run(collection, data_dir, out_dir, *, quick=False, nuts=True):
    if collection not in RECIPES:
        raise SystemExit(f"unknown collection {collection}; known: {', '.join(RECIPES)}")
    cfg = dict(nuts=NUTS if nuts else None)
    if quick:  # a smoke run: coarse grids, few candidates, no NUTS
        cfg = dict(nuts=None, n_candidates=50, n_refine=2, dp=0.005)
    t0 = time.time()
    result = RECIPES[collection](data_dir, cfg)
    result = dict(collection=collection, recipe=RECIPES[collection].__name__, quick=quick,
                  provenance=provenance(), seconds=time.time() - t0, **result)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"fit_{collection}.json")
    with open(path + ".part", "w") as f:
        json.dump(_jsonable(result), f, indent=1)
    os.replace(path + ".part", path)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("collection")
    p.add_argument("data_dir")
    p.add_argument("out_dir")
    p.add_argument("--quick", action="store_true", help="coarse grids, no NUTS (smoke test)")
    p.add_argument("--no-nuts", action="store_true", help="orbits: maximum a posteriori only")
    a = p.parse_args(argv)
    print(run(a.collection, a.data_dir, a.out_dir, quick=a.quick, nuts=not a.no_nuts), flush=True)


if __name__ == "__main__":
    main()
