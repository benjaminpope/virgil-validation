"""Probe the posterior geometry behind SBC v1's divergences (an investigation,
not a registered campaign: its numbers are diagnostics on a few replicates).

SBC v1 (scripts/sbc_numpyro.py) found divergences in 978 of 1000 replicates.
Here the same replicates (v1's truths and data, by seed) are sampled under
variants of the model and the sampler, and the divergence rate, worst R-hat,
smallest bulk ESS and the mean tree size are compared:

  v1            as v1: diag mass, AngleVector(0.25) for the position angle
  v1_ta95       v1 with target_accept_prob 0.95 (is it a step-size artefact?)
  v1_fixnoise   v1 with vis_scale fixed at the truth (does the noise site matter?)
  ring01_dense  AngleVector(0.1), dense mass on (pa_vec, sep, flux), as sbc_v2
  gn            v1 with fixed noise, fit then virgil's gauss_newton_mass (virgil's recipe)
  cart          companion as Cartesian offsets (dra, ddec) ~ U(-25, 25), diag mass
  cart_dense    cart with dense_mass=True
  cart_gn       cart with fixed noise and gauss_newton_mass
  cart_astrom   cart with an astrometric prior: dra, ddec ~ N(measured position, ASTROM_SIGMA), the measured
                position being the truth plus N(0, ASTROM_SIGMA) noise (valid SBC for a flat truth prior);
                chains start at the measured position, nuisance parameters at their prior medians
  plain_angle   (diagnostic only) pa as a Uniform angle within 180 deg of the truth: no ring

    python scripts/sbc_geometry_probe.py --variants v1 cart --seeds 20261005 20261006 --out FILE.json
"""

import argparse
import importlib.util
import json
import pathlib
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("sbc_numpyro", ROOT / "scripts" / "sbc_numpyro.py")
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)

FIXED_NOISE = ("gn", "cart_gn", "v1_fixnoise")
ASTROM_SIGMA = 0.5  # mas
VARIANTS = ("cart_astrom", "v1", "v1_ta95", "v1_fixnoise", "ring01_dense", "gn", "cart", "cart_dense", "cart_gn", "plain_angle")


def cart_scene(diam, flux, dra, ddec):
    from virgil import models as vm

    return vm.System(star=vm.UniformDisk(diam), comp=vm.PointSource(flux, dra, ddec))


def polar_scene(diam, flux, sep, pa):
    return v1.scene(diam, flux, sep, pa)


def build(variant, truth, data):
    """(scene function, priors, noise priors, init values for the model's
    sites, NUTS keyword arguments)."""
    import numpyro.distributions as dist
    from virgil.angles import AngleVector

    b = v1.BOUNDS
    noise = {"vis_scale": dist.LogUniform(*b["vis_scale"])}
    p = np.deg2rad(truth["pa"])
    dra, ddec = truth["sep"] * np.sin(p), truth["sep"] * np.cos(p)
    common = {"diam": dist.LogUniform(*b["diam"]), "flux": dist.LogUniform(*b["flux"])}
    cartesian = variant.startswith("cart")
    if variant == "cart_astrom":  # the measured position: truth + noise, from a stream independent of the data's
        meas = np.random.default_rng([int(truth["sep"] * 1e6), 7]).normal(size=2) * ASTROM_SIGMA + [dra, ddec]
    if cartesian:
        scene = cart_scene
        priors = common | {"dra": dist.Uniform(-25.0, 25.0), "ddec": dist.Uniform(-25.0, 25.0)}
        init = {"diam": truth["diam"], "flux": truth["flux"], "dra": dra, "ddec": ddec}
        if variant == "cart_astrom":
            priors = common | {"dra": dist.Normal(meas[0], ASTROM_SIGMA), "ddec": dist.Normal(meas[1], ASTROM_SIGMA)}
            init = {"dra": meas[0], "ddec": meas[1], "diam": float(np.sqrt(b["diam"][0] * b["diam"][1])),
                    "flux": float(np.sqrt(b["flux"][0] * b["flux"][1]))}  # geometric medians of the priors
    else:
        scene = polar_scene
        ring = 0.1 if variant == "ring01_dense" else 0.25
        pa_prior = (dist.Uniform(truth["pa"] - 180.0, truth["pa"] + 180.0) if variant == "plain_angle"
                    else AngleVector(ring_width=ring))
        priors = common | {"sep": dist.LogUniform(*b["sep"]), "pa": pa_prior}
        init = {"diam": truth["diam"], "flux": truth["flux"], "sep": truth["sep"]}
        init |= {"pa": truth["pa"]} if variant == "plain_angle" else {"pa_vec": np.array([np.cos(p), np.sin(p)])}
    if variant in FIXED_NOISE:  # the file's errors are made the true ones (see run_variant): no noise site
        noise = None
    else:
        init["noise.vis_scale"] = truth["vis_scale"]
    nuts = {}
    if variant == "v1_ta95":
        nuts["target_accept_prob"] = 0.95
    if variant == "ring01_dense":
        nuts["dense_mass"] = [("pa_vec", "sep", "flux")]
    if variant == "cart_dense":
        nuts["dense_mass"] = True
    return scene, priors, noise, init, nuts


def run_variant(variant, seed, warmup, samples, chains, init_from="truth"):
    import jax
    import numpyro
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    from numpyro.infer import MCMC, NUTS, init_to_value
    from virgil.fitting import fit, gauss_newton_mass
    from virgil.likelihood import numpyro_model
    from virgil.oidata import OIData

    rng = np.random.default_rng(seed)
    truth = v1.draw_truth(rng)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "sbc.fits"
        v1.observe(path, truth, rng)
        if variant in FIXED_NOISE:  # state the true errors in the file instead of sampling vis_scale
            from astropy.io import fits

            with fits.open(path, mode="update") as h:
                h["OI_VIS2"].data["VIS2ERR"][:] = v1.SIGMA_V2 * truth["vis_scale"]
        data = OIData(str(path))
    scene, priors, noise, init, nuts = build(variant, truth, data)
    model = numpyro_model(scene, priors, data, noise=noise)
    t0 = time.time()
    if variant in ("gn", "cart_gn"):
        start = {k: v for k, v in init.items() if k in ("diam", "flux", "sep", "dra", "ddec")}
        if "pa_vec" in init:
            start["pa"] = truth["pa"]
        result = fit(scene, priors, data, init=start)
        mass = gauss_newton_mass(scene, priors, data, result.values)
        nuts = dict(mass)
        init = {k: np.asarray(v) for k, v in result.values.items() if k in ("diam", "flux", "sep", "dra", "ddec", "pa_vec")}
    if init_from == "prior":  # each chain starts at its own draw from the prior: no truth, no fit
        from numpyro.infer import init_to_sample

        kernel = NUTS(model, init_strategy=init_to_sample, **nuts)
    else:
        kernel = NUTS(model, init_strategy=init_to_value(values=init), **nuts)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=chains, chain_method="sequential",
                progress_bar=False)
    mcmc.run(jax.random.key(seed), extra_fields=("diverging", "num_steps", "adapt_state.step_size"))
    extra = mcmc.get_extra_fields(group_by_chain=True)
    grouped = mcmc.get_samples(group_by_chain=True)
    div = np.asarray(extra["diverging"])
    out = {"variant": variant, "init": init_from, "seed": seed, "truth": truth, "divergent": int(div.sum()), "draws": int(div.size),
           "mean_steps": float(np.mean(extra["num_steps"])), "seconds": time.time() - t0, "ess": {}, "rhat": {},
           "divergent_where": {}}
    scalar = [k for k, v in grouped.items() if np.asarray(v).ndim == 2]
    # chains that ended in the truth's mode: the chain's median companion position within 2 mas of the truth's
    p = np.deg2rad(truth["pa"])
    tx, ty = truth["sep"] * np.sin(p), truth["sep"] * np.cos(p)
    if "dra" in grouped:
        xs, ys = np.asarray(grouped["dra"]), np.asarray(grouped["ddec"])
    else:
        sep, pa = np.asarray(grouped["sep"]), np.deg2rad(np.asarray(grouped["pa"]))
        xs, ys = sep * np.sin(pa), sep * np.cos(pa)
    out["chains_in_truth_mode"] = int(np.sum(np.hypot(np.median(xs, axis=1) - tx, np.median(ys, axis=1) - ty) < 2.0))
    out["chains"] = int(xs.shape[0])
    for k in scalar:
        out["ess"][k] = float(effective_sample_size(np.asarray(grouped[k])))
        out["rhat"][k] = float(split_gelman_rubin(np.asarray(grouped[k])))
    if "pa_vec" in grouped:  # radius of the angle vector, where the divergences sit
        r = np.linalg.norm(np.asarray(grouped["pa_vec"]), axis=-1)
        out["pa_radius"] = {"mean": float(r.mean()), "sd": float(r.std()),
                            "divergent_mean": float(r[div].mean()) if div.any() else None}
    for k in scalar:
        x = np.asarray(grouped[k])
        if div.any() and x.std() > 0:
            out["divergent_where"][k] = float((x[div].mean() - x.mean()) / x.std())  # in posterior sd
    return out


def model_sites(model):
    from numpyro import handlers
    import jax

    return set(handlers.trace(handlers.seed(model, jax.random.key(0))).get_trace())


def main():
    import jax

    jax.config.update("jax_enable_x64", True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=["v1"])
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--init", choices=("truth", "prior"), default="truth")
    ap.add_argument("--regime", choices=("bright", "low"), default="bright",
                    help="low: flux log-uniform on 1e-3 to 2e-2 (SBC v2's low regime), else v1's 0.02 to 0.1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.regime == "low":
        v1.BOUNDS["flux"] = (1e-3, 2e-2)  # truths and priors both read it
    out = pathlib.Path(a.out)
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(r["variant"], r["seed"], r.get("init", "truth")) for r in rows}  # one regime per file
    for seed in a.seeds:
        for variant in a.variants:
            if (variant, seed, a.init) in done:
                continue
            try:
                r = run_variant(variant, seed, a.warmup, a.samples, a.chains, a.init)
            except Exception as e:  # noqa: BLE001 - a variant that cannot run is a result
                r = {"variant": variant, "seed": seed, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            rows.append(r)
            out.write_text(json.dumps(rows, indent=1))
            if "error" in r:
                print(variant, seed, r["error"], flush=True)
            else:
                print(f"{variant:13s} {a.init:5s} {seed} modes {r['chains_in_truth_mode']}/{r['chains']} div {r['divergent']:4d}/{r['draws']} minESS {min(r['ess'].values()):7.0f} "
                      f"maxRhat {max(r['rhat'].values()):.3f} steps {r['mean_steps']:6.1f} {r['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
