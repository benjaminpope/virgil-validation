"""Why do the oidb_fit orbits of Gl 229 and iota Peg miss the published ones?

Four questions, on the same data and the same loss as ``oidb_fit.fit_orbit``:

1. Per night: the catalogue of grid peaks (``epoch_positions``, 8 peaks) and the
   marginal score of each, against the published position at that night.
2. Per night: the loss and chi2 of the static fit started at the published position
   (``fit_epoch(start=...)``), against the loss of the position the start search chose.
3. The orbit refit started at the published elements: its loss and chi2 against the
   MAP of the committed ``fit_<id>.json`` (lower loss = the search missed a better mode).
4. The same refit started at the committed MAP (does the refit reproduce its loss?).

Published elements are converted to virgil's convention (omega of the secondary, Omega
of the receding node); the conversion is checked against the published astrometry
(``astrometry_check`` in the output).

    python scripts/orbit_diagnose.py gl229|iotapeg DATA_DIR FIT_JSON OUT_JSON
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oidb_fit as of  # noqa: E402

np, jnp, dist = of.np, of.jnp, of.dist
from virgil.epochs import Epochs, epoch_positions  # noqa: E402
from virgil.fitting import fit  # noqa: E402
from virgil.orbits import KeplerOrbit  # noqa: E402

PLX_GL229 = 173.8  # mas, Gaia; a = 0.0424 au -> 7.37 mas (Xuan et al. quote ~7.3)

# collection -> (published orbit as KeplerOrbit arguments at t_ref, published positions per night
# (rho, PA) if the paper gives them, else None: then the orbit predicts them)
REF = {
    "gl229": dict(P=12.134, T=60377.88, ecc=0.234, inc=31.4, omega=(180.7 + 180.0) % 360.0, Omega=213.0,
                  a_mas=0.0424 * PLX_GL229),
    "iotapeg": dict(P=10.21277, T=52997.4686, ecc=0.0052, inc=95.865, omega=(272.45 + 180.0) % 360.0,
                    Omega=176.194, a_mas=10.3609),
}
IOTA_TABLE2 = {"2018-10-22": (10.0352, 354.7005), "2018-10-23": (6.5705, 348.914),
               "2018-11-21": (10.1793, 357.2094), "2019-07-31": (4.979, 165.5565),
               "2019-08-06": (1.1297, 244.4933)}


def gl229(data_dir, cfg):
    sel = dict(wavel_min=2.05 * of.UM, wavel_max=2.18 * of.UM, observables="phi")
    nights = of._group_by(of._files(data_dir, r"dualscivis\.fits$"), lambda f: f[6:16])
    per_night = {n: of.load(p, insname_prefix="GRAVITY_SC", target=r"B$|_B\b", select=sel) for n, p in nights.items()}
    per_file = {n: {os.path.basename(f): of.load([f], insname_prefix="GRAVITY_SC", target=r"B$|_B\b", select=sel)
                    for f in p} for n, p in nights.items()}
    grid = of._grid(12.0, 0.2, np.linspace(0.2, 1.0, 9))
    priors = {"dra": dist.Uniform(-15.0, 15.0), "ddec": dist.Uniform(-15.0, 15.0), "flux": dist.LogUniform(0.05, 1.0)}
    opri = of.orbit_priors((11.5, 12.8), (2.0, 20.0), (0.05, 1.0))
    return per_night, per_file, of.point_binary, priors, grid, of.orbital_point_binary, opri


def iotapeg(data_dir, cfg):
    nights = of._group_by(of._files(data_dir, r"mircx\d+_oifits_viscal\.fits$"), lambda f: f[:10])
    per_night = {n: of.load(p) for n, p in nights.items()}
    per_file = {n: {os.path.basename(f): of.load([f]) for f in p} for n, p in nights.items()}
    scene = of.candid_scene(ud1=of.IOTA_PEG_UD[0], ud2=of.IOTA_PEG_UD[1], resolved=0.0)
    priors = {"dra": dist.Uniform(-15.0, 15.0), "ddec": dist.Uniform(-15.0, 15.0), "flux": dist.LogUniform(0.02, 1.0)}
    grid = of._grid(12.0, 0.1, np.geomspace(0.05, 1.0, 8))
    opri = of.orbit_priors((9.0, 11.5), (3.0, 30.0), (0.02, 1.0))
    return (per_night, per_file, scene, priors, grid,
            lambda t: of.orbital_disk_binary(t, *of.IOTA_PEG_UD), opri)


def predicted(ref, t_ref, mjd):
    o = KeplerOrbit(ref["P"], ref["T"] - t_ref, ref["ecc"], ref["inc"], ref["omega"], ref["Omega"], ref["a_mas"],
                    t_ref=t_ref)
    rho, pa = (float(x) for x in o.separation_pa(mjd))
    return rho, pa, o


def xy(rho, pa):
    return rho * np.sin(np.radians(pa)), rho * np.cos(np.radians(pa))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=list(REF))
    ap.add_argument("data_dir")
    ap.add_argument("fit_json", help="the committed fit_<id>.json (its orbit.t_ref, map_sampled, loss)")
    ap.add_argument("out_json")
    a = ap.parse_args()
    per_night, per_file, scene, priors, grid, orbit_scene, opri = {"gl229": gl229, "iotapeg": iotapeg}[a.which](
        a.data_dir, {})
    prev = json.load(open(a.fit_json))
    prev_epochs = {e["label"]: e for e in prev["epochs"]}
    t_ref = prev["orbit"]["t_ref"]
    ref = REF[a.which]
    out = dict(which=a.which, t_ref=t_ref, ref=ref, committed_loss=prev["orbit"]["loss"], nights=[])

    # 1-2: per night
    pos = epoch_positions(Epochs(per_night), grid, n_peaks=8)
    for k, (night, data) in enumerate(per_night.items()):
        mjd = of.mean_mjd(data)
        rho, pa, _ = predicted(ref, t_ref, mjd)
        if a.which == "iotapeg":
            rho, pa = IOTA_TABLE2[night]  # the paper's own astrometry
        px, py = xy(rho, pa)
        pk = pos.peaks[k]
        peaks = [dict(rho=float(np.hypot(x, y)), pa=float(np.degrees(np.arctan2(x, y)) % 360), flux=float(f),
                      marginal=float(m), loglike=float(l), dist_to_published=float(np.hypot(x - px, y - py)))
                 for x, y, f, m, l in zip(pk.dra, pk.ddec, pk.flux, pk.marginal, pk.loglike)]
        flux0 = peaks[0]["flux"] if a.which == "gl229" else 1 / 4.6
        at_pub = of.fit_epoch(night, data, scene, priors, grid, start=dict(dra=px, ddec=py, flux=flux0))
        out["nights"].append(dict(
            night=night, mjd=mjd, published=dict(rho=rho, pa=pa), peaks=peaks,
            gap_marginal=float(pos.gap_marginal[k]),
            committed=dict(loss=prev_epochs[night]["loss"], chi2=prev_epochs[night]["chi2_raw"]["all"],
                           values=prev_epochs[night]["values"]),
            from_published=dict(loss=at_pub["loss"], chi2=at_pub["chi2_raw"]["all"], values=at_pub["values"],
                                scales=at_pub["scales"])))
        n = out["nights"][-1]
        of.log(f"{night}: published rho={rho:.2f} PA={pa:.1f}; loss from published {at_pub['loss']:.1f} "
               f"vs committed {n['committed']['loss']:.1f}; best peak marginal {peaks[0]['marginal']:.1f}, "
               f"nearest peak to published {min(p['dist_to_published'] for p in peaks):.2f} mas")

    # 3-4: the orbit refit from the published elements and from the committed MAP
    fit_data = Epochs(per_file)
    model_fn = fit_data.model_fn(orbit_scene(t_ref))
    noise = [of.noise_terms(d) for d in fit_data.data]
    flux = prev["orbit"]["map"]["flux_ratio"] if a.which == "gl229" else 1 / 4.6
    _, _, o = predicted(ref, t_ref, t_ref)
    out["astrometry_check"] = [dict(night=n["night"], rho_pa_orbit=predicted(ref, t_ref, n["mjd"])[:2],
                                    rho_pa_published=(n["published"]["rho"], n["published"]["pa"]))
                               for n in out["nights"]]
    for label, init in (("published", of.start_values_for(opri)(o, flux)), ("committed_map", prev["orbit"]["map_sampled"])):
        r = fit(model_fn, opri, list(fit_data.data), noise=noise, init={k: init[k] for k in opri})
        v = {k: float(np.asarray(r.values[k])) for k in opri}
        models = model_fn(**v)
        chi2 = sum(of.raw_chi2(m, d)["all"]["chi2"] for m, d in zip(models, fit_data.data))
        n_obs = sum(of.raw_chi2(m, d)["all"]["n"] for m, d in zip(models, fit_data.data))
        out[label] = dict(loss=float(np.asarray(r.info["loss"])), converged=bool(r.info.get("converged")),
                          elements={k: float(x) for k, x in of.elements(v, t_ref).items()},
                          chi2=chi2, n=n_obs, chi2_red=chi2 / n_obs)
        of.log(f"orbit from {label}: loss {out[label]['loss']:.1f} (committed {prev['orbit']['loss']:.1f}), "
               f"chi2/N {chi2 / n_obs:.3g}, elements {out[label]['elements']}")
    with open(a.out_json, "w") as f:
        json.dump(of._jsonable(out), f, indent=1)
    print(a.out_json)


if __name__ == "__main__":
    main()
