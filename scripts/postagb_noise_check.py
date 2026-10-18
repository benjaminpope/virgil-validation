"""Does the virgil-vlti PIONIER noise model fix the post-AGB GP fits?  Measurements for design/postagb_noise_model.md.

    python scripts/postagb_noise_check.py table  OUTDIR   # the 2 x 2 table (noise x GP mean) on IW Car, from saved fits
    python scripts/postagb_noise_check.py strips OUTDIR   # one strip figure per strategy: published | the four cells
    python scripts/postagb_noise_check.py synth  OUTDIR --case {white,gains-indep,gains-model}   # synthetic evidence check

The fits themselves are ``scripts/postagb_imaging.py image / compare`` with ``--noise {current,vlti}`` and ``--mean {blob,flat}``,
all into OUTDIR (files ``IW_Car_<strategy>[_vlti][_flat]_*``). ``table`` reads them and measures, on the *quoted* errors of the data,
what the fits leave behind:

* chi2 per point of the V^2 and of the closure phases, separately;
* the common-mode share of the V^2 residual variance: the six channels of one (frame, baseline) share a calibration gain, so the
  per-(frame, baseline) channel mean carries ``n_channels * mean(m_g^2) / mean(z^2)`` of the standardised residual variance
  (z = residual / quoted error); independent errors give 1 / n_channels = 17 %;
* the correlation of adjacent channels' standardised V^2 residuals within a (frame, baseline).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import postagb_imaging as pi  # noqa: E402  (sets the JAX environment before jax is imported)

CELLS = {  # label: (noise, mean) -> file-name suffix of postagb_imaging
    "current / blob": "", "current / flat": "_flat", "vlti / blob": "_vlti", "vlti / flat": "_vlti_flat"}
STRATS = ("mem", "gp-iso", "gp-polar")


def rows_of(data):
    """Sample indices of each (frame, station pair), channels in wavelength order: an (n_group, n_channel) array."""
    frame, st, wl = np.asarray(data.frame), np.asarray(data.stations), np.asarray(data.wavel)
    key = {}
    for i, (f, s, w) in enumerate(zip(frame, st, wl)):
        key.setdefault((int(f), int(s[0]), int(s[1])), []).append((float(w), i))
    return np.array([[i for _, i in sorted(v)] for v in key.values()])


def residual_stats(data, model_v2, model_t3):
    """chi2 per point (V^2, T3), common-mode share of the V^2 residual variance and adjacent-channel correlation, on quoted errors."""
    z = (np.asarray(data.vis) - np.asarray(model_v2)) / np.asarray(data.d_vis)
    dt = np.degrees(np.asarray(data.phi)) - np.asarray(model_t3)
    zt = (((dt + 180.0) % 360.0) - 180.0) / np.degrees(np.asarray(data.d_phi))
    g = rows_of(data)
    zg = z[g]  # (n_group, n_channel)
    m = zg.mean(axis=1)
    n_ch = zg.shape[1]
    a, b = zg[:, :-1].ravel(), zg[:, 1:].ravel()
    return dict(chi2_v2=float(np.mean(z**2)), chi2_t3=float(np.mean(zt**2)), n_v2=int(z.size), n_t3=int(zt.size),
                common_mode_v2=float(n_ch * np.mean(m**2) / np.mean(z**2)), white_expectation=1.0 / n_ch,
                adjacent_corr_v2=float(np.corrcoef(a, b)[0, 1]))


def cell_row(out, strat, suffix, quoted):
    """One cell of the table for one strategy, from the saved image / compare files; None if the fit has not been run."""
    key = f"IW_Car_{strat}{suffix}"
    img, cmp_, npz = out / f"{key}_image.json", out / f"{key}_compare.json", out / f"{key}_image.npz"
    if not (img.exists() and cmp_.exists()):
        return None
    i, c, z = json.load(open(img)), json.load(open(cmp_)), np.load(npz)
    r = residual_stats(quoted, z["model_v2"], z["model_t3"])
    hyper = {k: i[k] for k in ("gp_sigma", "gp_length_mas", "gp_length_r_mas", "gp_length_theta") if k in i}
    return dict(strategy=strat, hyper=hyper, on_edge=i.get("gp_on_edge"), log_evidence=i.get("log_evidence"), ncc=c["ncc_zero_shift"],
                ncc_best_control=max(c["ncc_controls"].values()), f_env_pct=i["f_env_pct"], f_bg_pct=i.get("f_bg_pct"),
                bg_index=i.get("bg_index"), bound_warnings=i.get("bound_warnings"), env_index=i.get("env_index"),
                chi2_fit=i["chi2_per_point"], seconds=i["seconds"], **r)


def table_stage(args):
    from postagb_imaging import TARGETS, cfg_of  # noqa: PLC0415

    cfg = cfg_of(argparse.Namespace(target="IW_Car", bg_index=None, star_index=None))
    quoted, _ = pi.load(cfg)
    res = {lab: {s: cell_row(args.outdir, s, suf, quoted) for s in STRATS} for lab, suf in CELLS.items()}
    # the grid edge and evidence differences between strategies are within a cell (same data and same error model)
    for lab, row in res.items():
        ev = {s: v["log_evidence"] for s, v in row.items() if v and v["log_evidence"] is not None}
        for s, v in row.items():
            if v and v["log_evidence"] is not None:
                v["dlogz_vs_best_strategy"] = v["log_evidence"] - max(ev.values())
    json.dump(res, open(args.outdir / "IW_Car_noise_table.json", "w"), indent=1)
    lines = ["| strategy | noise / mean | hyperparameters (edge?) | χ²/pt V² | χ²/pt T3 | common-mode V² | adjacent corr | NCC (best control) | ΔlnZ | f_env % | f_bg % | bg index |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in STRATS:
        for lab in CELLS:
            v = res[lab][s]
            if v is None:
                lines.append(f"| {s} | {lab} | not run | | | | | | | | | |")
                continue
            h = ", ".join(f"{k.replace('gp_', '')} {x:.3g}" for k, x in v["hyper"].items()) or "–"
            edge = {True: " **edge**", False: "", None: ""}[v["on_edge"]]
            dz = v.get("dlogz_vs_best_strategy")
            bg = "–" if v["bg_index"] is None else f"{v['bg_index']:.2f}" + (" (bound)" if any("bg.flux.index" in w for w in v["bound_warnings"] or []) else "")
            lines.append(f"| {s} | {lab} | {h}{edge} | {v['chi2_v2']:.2f} | {v['chi2_t3']:.2f} | {100 * v['common_mode_v2']:.0f} % | {v['adjacent_corr_v2']:.2f} | "
                         f"{v['ncc']:.2f} ({v['ncc_best_control']:.2f}) | {'–' if dz is None else f'{dz:.0f}'} | {v['f_env_pct']:.1f} | {v['f_bg_pct']:.1f} | {bg} |")
    text = "\n".join(lines)
    (args.outdir / "IW_Car_noise_table.md").write_text(text + "\n")
    print(text)


def strips_stage(args):
    import matplotlib.pyplot as plt  # noqa: PLC0415

    cfg = pi.TARGETS["IW_Car"]
    for s in STRATS:
        have = {lab: np.load(args.outdir / f"IW_Car_{s}{suf}_compare_images.npz") for lab, suf in CELLS.items()
                if (args.outdir / f"IW_Car_{s}{suf}_compare_images.npz").exists()}
        if not have:
            continue
        first = next(iter(have.values()))
        sc, n, half = float(first["pixel_mas"]), first["ours"].shape[0], cfg["half_mas"]
        ext = [n * sc / 2, -n * sc / 2, -n * sc / 2, n * sc / 2]
        panels = [("ORGANIC", first["published_conv"])] + [(lab, z["ours_conv"]) for lab, z in have.items()]
        fig, ax = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.5), sharey=True)
        for a, (lab, im) in zip(ax, panels):
            a.imshow(im / im.max(), origin="upper", extent=ext, cmap="magma", vmin=0, vmax=1)
            a.set_xlim(half, -half)
            a.set_ylim(-half, half)
            a.plot(0, 0, "*", c="gold", ms=8, mec="k")
            a.set_title(lab, fontsize=9)
            a.set_xlabel("Δα (mas)", fontsize=8)
        ax[0].set_ylabel("Δδ (mas)", fontsize=8)
        fig.suptitle(f"IW Car, {s}: noise model / GP mean", fontsize=10)
        fig.tight_layout()
        fig.savefig(args.outdir / f"IW_Car_{s}_noise_strip.png", dpi=130)
        plt.close(fig)


# ----------------------------------------------------------------------------
# synthetic evidence check


def synth_truth(cfg, args, data):
    """A smooth known disc on IW Car's uv coverage: our own parametric ring (a Gaussian ring with a first harmonic, the ``scale`` stage's
    geometry) with the fluxes of the isotropic-GP image fit, in the SPARCO scene."""
    from virgil.models import GaussianRing, Harmonics, MultipoleDisk  # noqa: PLC0415
    from virgil.spectra import PowerLaw  # noqa: PLC0415

    geo = json.load(open(args.outdir / "IW_Car_scale.json"))["geometry"]
    img = json.load(open(args.outdir / "IW_Car_gp-iso_image.json"))
    disk = MultipoleDisk(GaussianRing(geo["radius_mas"], geo["width_mas"]), harmonics=Harmonics(re=0.3, im=0.3), inc=geo["inc_deg"],
                         pa=geo["pa_deg"], n_r=48, flux=PowerLaw(img["f_env_pct"] / img["f_prim_pct"], img["env_index"], pi.L0))
    scene = pi.sparco_scene(cfg, disk, f_env=img["f_env_pct"] / img["f_prim_pct"], f_sec=img["f_sec_pct"] / img["f_prim_pct"])
    return scene.set("sec.dra", img["sec_dra"]).set("sec.ddec", img["sec_ddec"]) if hasattr(scene, "sec") else scene


def synth_stage(args):
    import equinox as eqx  # noqa: PLC0415
    import jax  # noqa: PLC0415

    cfg = pi.cfg_of(argparse.Namespace(target="IW_Car", bg_index=None, star_index=None))
    base, _ = pi.load(cfg, bandwidth=True)
    truth = synth_truth(cfg, args, base)
    key = jax.random.PRNGKey(args.seed)
    if args.case == "white":
        data = base.with_model(truth, key)  # white noise at the quoted errors, nothing else
    else:
        sim = pi.vlti_noise(base).with_model(truth, key)  # white at the vlti scale + gains + closure offsets, drawn at the vlti widths
        if args.case == "gains-model":
            data = sim
        else:  # the same numbers fitted with independent errors: the quoted errors back, no gains, no offsets
            n = pi.VLTI_NOISE
            data = eqx.tree_at(lambda d: (d.gains, d.phase_offsets), sim, (None, None), is_leaf=lambda x: x is None)
            data = data.with_error_scale({"vis": 1.0 / n["v2_white"], "phi": 1.0 / n["t3_white"]})
    ns = argparse.Namespace(smoke=False, mean=args.mean, noise="vlti" if args.case == "gains-model" else "current")
    start, support, hole = pi.grid_setup(cfg, data, ns)
    env = pi.Image(start.env.log_brightness, start.env.pixel_scale_mas, support=support, flux=start.env.flux)
    scene = pi.sparco_scene(cfg, env, f_env=float(np.asarray(env.flux)))
    priors = pi.image_priors(scene) | pi.scene_priors(cfg, scene)
    model, arrays, extra = pi.fit_gp(cfg, scene, start.env, support, priors, data, ns)
    log_z = arrays["gp_log_z"]
    res = dict(case=args.case, mean=args.mean, seed=args.seed, hyper={k: extra[k] for k in ("gp_sigma", "gp_length_mas")},
               on_edge=extra["gp_on_edge"], log_evidence=extra["log_evidence"], chi2_chosen=extra["gp_chi2"],
               sigmas=arrays["gp_sigmas"].tolist(), lengths_mas=arrays["gp_lengths_mas"].tolist(),
               log_z=(log_z - log_z.max()).tolist(), chi2=arrays["gp_chi2"].tolist())
    json.dump(res, open(args.outdir / f"synth_{args.case}_{args.mean}_s{args.seed}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("stage", choices=("table", "strips", "synth"))
    p.add_argument("outdir", type=Path)
    p.add_argument("--case", default="white", choices=("white", "gains-indep", "gains-model"))
    p.add_argument("--mean", default="blob", choices=("blob", "flat"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    {"table": table_stage, "strips": strips_stage, "synth": synth_stage}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
