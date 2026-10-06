"""Synthetic benchmark for contest imaging: known phantoms on the real contest
uv coverages, so that each method choice is judged against a truth.

    python scripts/contest_bench.py simulate [--data DIR] [--out DIR] [--phantoms 6] [--draws 2]
    python scripts/contest_bench.py list [--out DIR]
    python scripts/contest_bench.py run --id N [--bench DIR] [--out DIR] [--configs a,b,...] [--smoke] [--seed S]
    python scripts/contest_bench.py score [--bench DIR] [--out DIR]

``simulate`` takes each coverage's contest OIFITS as a template (uv points,
wavelengths, error bars, flags) and replaces V² and closure phases with those
of a phantom (``crosscheck.phantoms``, a family matching that contest's
pre-submission description, never its published truth), plus Gaussian noise
drawn from the quoted errors. The visibilities are a direct NumPy sum
(``crosscheck.sky``): nothing here imports virgil. Tables virgil would also
read but that the phantom does not define (OI_VIS, OI_FLUX) are dropped.
Each dataset is written as ``<coverage>_p<k>_d<j>/`` holding the FITS files,
``truth.npz`` (image, pixel scale, parameters) and ``meta.json``.

``run`` images dataset ``N // len(configs)`` with configuration
``N % len(configs)`` (one OzSTAR array task each) through the contest code
path (``contest_images.run_gp`` with fixed settings), writing
``<dataset>__<config>.npz``. The opt-in ``ensemble`` arm (``--configs
ensemble``) runs ``virgil.ensemble`` instead and also saves its standard
deviation map (``std``, and ``ref_std`` on the reference grid). ``score`` compares every result with its truth
(``crosscheck.image_metrics.score``) and writes ``bench_scores.md`` (median
per configuration, overall and per family) and ``bench_scores.json``.

Optional calibration errors (``--cal``) follow the 2018 readme: a random
multiplicative V² offset per baseline and pointing, and a random closure
phase offset per triangle and pointing, on top of the noise.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from crosscheck import phantoms, sky  # noqa: E402

# coverage -> (contest files, phantom family). Grey datasets that the first
# campaign fitted, each with the family its pre-submission information named.
COVERAGES = {
    "2004_data2": (["2004/2004-data2.fits"], "spotted_star"),
    "2006": ([f"2006/2006-03-0{n}.fits" for n in (3, 4, 5, 6)], "thin_disk"),
    "2008_agb_H": (["2008/2008-Contest1_H.oifits"], "envelope"),
    "2018": (["2018/Aspro2_Altair_MIRC_6T_1_47493-1_75256-8ch_S1-S2-E1-E2-W1-W2_2018-08-02_FAKE.fits",
              "2018/Aspro2_Altair_PIONIER_1_533-1_772-6ch_A0-B2-C1-D0_2018-08-02_FAKE.fits",
              "2018/Aspro2_Altair_PIONIER_1_533-1_772-6ch_A0-G1-J2-J3_2018-08-02_FAKE.fits",
              "2018/Aspro2_Altair_PIONIER_1_533-1_772-6ch_D0-G2-J3-K0_2018-08-02_FAKE.fits"], "star_disk_planet"),
    "2024_obj1_pionier": (["2024/Obj1_PIONIER_1.5-1.8.fits"], "spiral"),
}
PIXELS_PER_BEAM = 10  # truth grid resolution

# Pipeline configurations (benchmark arms). Each is the settings dict that
# contest_images.member_setup takes; change one factor at a time against
# "baseline" (the first campaign's CLEAN-started GP with no star).
BASE = {"method": "gp", "star": False, "star_model": "none", "halo": False, "sparco": False,
        "field": 2.0, "oversample": 3.0, "clean_gain": 0.1, "mean_blur": 1.0}
CONFIGS = {
    "baseline": BASE,
    "point_star": BASE | {"star": True, "star_model": "point"},
    "disk_star": BASE | {"star": True, "star_model": "disk"},
    "halo": BASE | {"halo": True},
    "field_1x": BASE | {"field": 1.0},
    "field_4x": BASE | {"field": 4.0},
    # GP mean from CLEAN components restored with half a beam (detail finer
    # than a beam survives into the start).
    "half_beam_mean": BASE | {"mean_blur": 0.5},
    # The winners' own method: MaxEnt with the CLEAN image as default model,
    # weight by the discrepancy principle (contest_images.run).
    "mem": BASE | {"method": "mem"},
    # Parametric starts from pre-submission information: an elliptical
    # limb-darkened star (needs virgil#250) and a point star, each with a
    # companion search (linear flux map) before CLEAN and the GP.
    "ellipse_star": BASE | {"star": True, "star_model": "ellipse"},
    "companion": BASE | {"star": True, "star_model": "point_companion"},
    # Not one pipeline but many: virgil.ensemble (Drevon et al. 2025's
    # PYRA/MYTHRA) draws regulariser families (TV, TSV, MaxEnt, starlet L1),
    # weights, pixel sizes, fields and starts at random, keeps the members
    # that fit and averages them while the mean still fits. No star, as in
    # the baseline. Opt-in (not in DEFAULT_CONFIGS), so that the ten-arm
    # task numbering of earlier runs is unchanged.
    "ensemble": BASE | {"method": "ensemble"},
}
OPT_IN = ("ensemble",)
DEFAULT_CONFIGS = [c for c in CONFIGS if c not in OPT_IN]
# The ensemble arm: groups (each one L-curve of ENSEMBLE_WEIGHTS weights, one
# family, geometry and start) and virgil.ensemble.EnsembleSpec's defaults
# otherwise (fields 0.5-1 x field_of_view, moments/flat starts, members within
# 2x the best χ² per dataset, a window 1 dex above each L-curve's corner).
ENSEMBLE_GROUPS = 12
ENSEMBLE_WEIGHTS = 6
DROP = ("OI_VIS", "OI_FLUX")


def wavelengths(hdul):
    """INSNAME -> effective wavelengths (m)."""
    return {h.header.get("INSNAME"): np.asarray(h.data["EFF_WAVE"], float)
            for h in hdul if h.name == "OI_WAVELENGTH"}


def beam_mas(paths):
    """λ_min / B_max over all files, in mas."""
    best = np.inf
    for path in paths:
        with fits.open(path) as h:
            waves = wavelengths(h)
            for t in h:
                if t.name == "OI_VIS2":
                    b = np.hypot(t.data["UCOORD"], t.data["VCOORD"]).max()
                    lam = waves[t.header.get("INSNAME")].min()
                    best = min(best, lam / b / sky.MAS)
    return float(best)


def _vis(cloud, u, v, lam):
    """Complex visibilities at baselines (n,) for wavelengths (m,): (n, m)."""
    return sky.visibility(cloud, u[:, None], v[:, None], lam[None, :])


def closure_noise(t3, err, rng):
    """Closure-phase noise (deg) formed from noisy baseline phases, as in real
    data: within a frame (same TIME/MJD), triangles that share a baseline
    share its phase error, so closure phases from four or more telescopes are
    correlated. Each baseline's phase σ is the smallest σ_cp/√3 among its
    triangles in the frame, and each closure phase gets independent noise on
    top to bring its variance to exactly its quoted σ_cp² (quoted errors
    range over 1-180° within a frame, so a shared average would swamp the
    precise triangles)."""
    # A frame is a (MJD, TIME) pair: OIFITS v1 files such as 2004's give every
    # row one MJD and tell snapshots apart only by TIME (finding F13).
    names = t3.columns.names
    frame_key = list(zip(*(np.asarray(t3[c]).tolist() for c in ("MJD", "TIME") if c in names)))
    sta = np.asarray(t3["STA_INDEX"])
    sigma = {}
    for row, (a, b, c) in enumerate(sta):
        for pair in ((a, b), (b, c), (a, c)):
            sigma.setdefault((frame_key[row], *sorted(pair)), []).append(err[row] / np.sqrt(3))
    sigma = {key: np.min(s, axis=0) for key, s in sigma.items()}
    phase = {key: s * rng.standard_normal(np.shape(s)) for key, s in sigma.items()}

    def baseline(row, i, j):
        sign = 1.0 if i < j else -1.0
        return sign * phase[(frame_key[row], *sorted((i, j)))]

    out = np.array([baseline(r, a, b) + baseline(r, b, c) + baseline(r, c, a) for r, (a, b, c) in enumerate(sta)])
    shared = np.array([sum(sigma[(frame_key[r], *sorted(p))] ** 2 for p in ((a, b), (b, c), (a, c)))
                       for r, (a, b, c) in enumerate(sta)])
    extra = np.sqrt(np.clip(err**2 - shared, 0.0, None))
    return out + extra * rng.standard_normal(np.shape(err))


def simulate_file(template, out, cloud, rng, noise=True, cal=None):
    """Write ``template`` with V² and closure phases replaced by the cloud's."""
    cal = cal or {}
    with fits.open(template) as h:
        waves = wavelengths(h)
        keep = [t for t in h if t.name not in DROP]
        for t in keep:
            if t.name == "OI_VIS2":
                d = t.data
                lam = waves[t.header.get("INSNAME")]
                v2 = np.abs(_vis(cloud, d["UCOORD"], d["VCOORD"], lam)) ** 2
                if cal.get("v2_rel"):  # one offset per row (baseline and pointing)
                    v2 = v2 * (1 + cal["v2_rel"] * rng.standard_normal((len(d), 1)))
                err = np.asarray(d["VIS2ERR"], float).reshape(v2.shape)
                if noise:
                    v2 = v2 + err * rng.standard_normal(v2.shape)
                d["VIS2DATA"] = v2.reshape(np.shape(d["VIS2DATA"]))
            elif t.name == "OI_T3":
                d = t.data
                lam = waves[t.header.get("INSNAME")]
                u1, v1, u2, v2_ = d["U1COORD"], d["V1COORD"], d["U2COORD"], d["V2COORD"]
                bis = _vis(cloud, u1, v1, lam) * _vis(cloud, u2, v2_, lam) * np.conj(_vis(cloud, u1 + u2, v1 + v2_, lam))
                phi = np.degrees(np.angle(bis))
                if cal.get("cp_deg"):  # one offset per row (triangle and pointing)
                    phi = phi + cal["cp_deg"] * rng.standard_normal((len(d), 1))
                err = np.asarray(d["T3PHIERR"], float).reshape(phi.shape)
                if noise:
                    phi = phi + closure_noise(d, err, rng)
                d["T3PHI"] = (((phi + 180.0) % 360.0) - 180.0).reshape(np.shape(d["T3PHI"]))
                d["T3AMP"] = np.abs(bis).reshape(np.shape(d["T3AMP"]))
        fits.HDUList([t.copy() for t in keep]).writeto(out, overwrite=True)


def truth_grid(params, beam):
    pixel = beam / PIXELS_PER_BEAM
    npix = int(np.ceil(2 * phantoms.extent(params) / pixel)) | 1
    return npix, pixel


def simulate(data_dir, out_dir, n_phantoms, n_draws, seed=0, cal=None, coverages=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for ci, (cov, (files, family)) in enumerate(COVERAGES.items()):
        if coverages and cov not in coverages:
            continue
        paths = [data_dir / f for f in files]
        beam = beam_mas(paths)
        for k in range(n_phantoms):
            prng = np.random.default_rng([seed, ci, k])
            params = phantoms.sample(family, prng, beam)
            npix, pixel = truth_grid(params, beam)
            image = phantoms.render(params, npix, pixel)
            cloud = sky.pixel_image(image, pixel)
            for j in range(n_draws):
                name = f"{cov}_p{k}_d{j}"
                d = out_dir / name
                d.mkdir(exist_ok=True)
                rng = np.random.default_rng([seed, ci, k, j, 1])
                for p in paths:
                    simulate_file(p, d / p.name, cloud, rng, cal=cal)
                np.savez_compressed(d / "truth.npz", image=image, pixel=pixel, npix=npix, beam=beam)
                (d / "meta.json").write_text(json.dumps(
                    {"coverage": cov, "family": family, "phantom": k, "draw": j, "seed": seed, "cal": cal or {},
                     "files": [p.name for p in paths], "beam_mas": beam, "pixel_mas": pixel, "npix": npix,
                     "params": params}, indent=1, default=float))
                made.append(name)
    (out_dir / "index.json").write_text(json.dumps(made, indent=1))
    return made


def run_ensemble(spec, out, label, seed=0, smoke=False):
    """The ``ensemble`` arm: ``virgil.ensemble.ensemble`` on the dataset's
    files, written in the other arms' format (``ref_image`` on the common
    reference grid, ``ref_fov``, ``best_chi2_red``, ``error_scale``) plus the
    ensemble's standard deviation map. Reads only the OIFITS files (blind:
    never ``truth.npz``)."""
    import time

    import contest_images  # sets jax_enable_x64, as for the other arms
    import jax
    import matplotlib.pyplot as plt
    from virgil.ensemble import EnsembleSpec, ensemble
    from virgil.imaging import beam
    from virgil.metrics import resample
    from virgil.oidata import OIData

    t0 = time.time()
    data = [OIData(f) for f in spec["files"]]
    data = data[0] if len(data) == 1 else data
    settings = CONFIGS["ensemble"]
    n_groups, n_weights = (2, 3) if smoke else (ENSEMBLE_GROUPS, ENSEMBLE_WEIGHTS)
    options = {"max_steps": 50} if smoke else {}
    ens_spec = EnsembleSpec(n_weights=n_weights)
    result = ensemble(data, n_groups, jax.random.PRNGKey(seed), spec=ens_spec, star=settings["star"], **options)

    # The other arms' common grid: max(MEMBER_FIELDS) x the data-chosen field.
    ref_fov = contest_images.reference_fov(spec, pathlib.Path("/"))
    n_ref = contest_images.REF_NPIX
    scale = float(result.mean.pixel_scale_mas)
    std = np.asarray(result.std, float)
    mean = np.asarray(result.mean.brightness, float)
    # Flux-conserving resampling of the std map onto the reference grid: exact
    # for the mean, an area-weighted approximation for a spread.
    ref_std = np.asarray(resample(std, scale, n_ref, ref_fov / n_ref))
    kept = result.kept
    reasons = {}
    for m in result.members:
        if not m.kept:
            reasons[m.reason] = reasons.get(m.reason, 0) + 1
    resolution = beam(data)
    out.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        ref_image=np.asarray(result.model.render(n_ref, ref_fov)), ref_fov=ref_fov, ref_std=ref_std,
        best_chi2_red=float(np.sum(result.chi2_red)), error_scale=np.nan, best_log_z=np.nan, flip_dchi2=np.nan,
        star=bool(settings["star"]), mean=mean, std=std, pixel_scale_mas=scale, npix=mean.shape[0],
        chi2_red=np.asarray(result.chi2_red), trace=np.asarray(result.trace),
        best_member_chi2_red=np.asarray(result.trace[0]), n_members=len(result.members), n_kept=len(kept),
        n_groups=n_groups, n_weights=n_weights, seed=seed,
        beam=np.array([resolution.major_mas, resolution.minor_mas, resolution.pa_deg]),
        dropped=json.dumps(reasons),
    )
    # Atomic: a reader (score, a download) never sees half a file.
    tmp = out / f".{label}.npz.part"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(out / f"{label}.npz")

    summary = (f"label={label} files={spec['files']} seed={seed} groups={n_groups} weights={n_weights} "
               f"star={settings['star']} smoke={smoke}\nref_fov={ref_fov:.4g} mas npix={mean.shape[0]} "
               f"pixel={scale:.4g} mas beam={resolution.major_mas:.3g}x{resolution.minor_mas:.3g} mas\n"
               f"{result.summary()}\nelapsed={time.time() - t0:.0f}s\n")
    (out / f"{label}.txt").write_text(summary)
    print(summary)

    fov = mean.shape[0] * scale
    extent = [fov / 2, -fov / 2, -fov / 2, fov / 2]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, img, title in zip(axes, (mean, std), ("ensemble mean", "ensemble std")):
        im = ax.imshow(img, origin="upper", extent=extent, cmap="inferno")
        ax.set(title=f"{label}: {title}", xlabel="ΔRA (mas)", ylabel="ΔDec (mas)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / f"{label}.png", dpi=90)
    plt.close(fig)
    return result


def run(bench, out, dataset_id, config_names, smoke=False, seed=0):
    """Image one simulated dataset with one configuration (array task id)."""
    import faulthandler

    faulthandler.dump_traceback_later(1800, repeat=True, file=sys.stderr)  # see contest_images.main
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import contest_images

    index = json.loads((bench / "index.json").read_text())
    name = index[dataset_id // len(config_names)]
    config = config_names[dataset_id % len(config_names)]
    meta = json.loads((bench / name / "meta.json").read_text())
    spec = {"label": name, "files": [str(bench / name / f) for f in meta["files"]]}
    settings = CONFIGS[config]
    if settings["method"] == "ensemble":
        run_ensemble(spec, out, f"{name}__{config}", seed=seed, smoke=smoke)
        return name, config
    runner = contest_images.run if settings["method"] == "mem" else contest_images.run_gp
    runner(spec, pathlib.Path("/"), out, smoke=smoke, settings=settings, label=f"{name}__{config}")
    return name, config


def score(bench, out):
    """Every result against its truth; medians per configuration and family."""
    from crosscheck import image_metrics

    rows = []
    for path in sorted(out.glob("*__*.npz")):
        name, config = path.stem.split("__")
        d = np.load(path)
        if "ref_image" not in d:
            continue
        truth = np.load(bench / name / "truth.npz")
        meta = json.loads((bench / name / "meta.json").read_text())
        s = image_metrics.score(d["ref_image"], float(d["ref_fov"]), truth["image"], float(truth["pixel"]),
                                float(truth["beam"]))
        rows.append({"dataset": name, "config": config, "family": meta["family"], "coverage": meta["coverage"],
                     "chi2_red": float(d["best_chi2_red"]), "error_scale": float(d["error_scale"]),
                     # Why a star arm's star was dropped (contest_images.setup's guards), or "".
                     "star_rejected": str(d["star_rejected"]) if "star_rejected" in d else "",
                     **{k: s[k] for k in ("lawson", "rms_e6", "l1", "ncc")}})
    (out / "bench_scores.json").write_text(json.dumps(rows, indent=1))
    metrics = ("lawson", "rms_e6", "l1", "ncc", "chi2_red")
    lines = ["# Benchmark scores (medians; lawson and rms lower is better, l1 and ncc higher)", "",
             "star_rejected: results whose star the guards dropped (imaged without it).", "",
             "| config | n | " + " | ".join(metrics) + " | star_rejected |", "|---|---|" + "---|" * (len(metrics) + 1)]
    configs = sorted({r["config"] for r in rows}, key=lambda c: list(CONFIGS).index(c) if c in CONFIGS else 99)
    for c in configs:
        sel = [r for r in rows if r["config"] == c]
        lines.append(f"| {c} | {len(sel)} | " + " | ".join(f"{np.median([r[m] for r in sel]):.4g}" for m in metrics)
                     + f" | {sum(bool(r['star_rejected']) for r in sel)} |")
    for fam in sorted({r["family"] for r in rows}):
        lines += ["", f"## {fam}", "", "| config | n | " + " | ".join(metrics) + " | star_rejected |",
                  "|---|---|" + "---|" * (len(metrics) + 1)]
        for c in configs:
            sel = [r for r in rows if r["config"] == c and r["family"] == fam]
            if sel:
                lines.append(f"| {c} | {len(sel)} | " + " | ".join(f"{np.median([r[m] for r in sel]):.4g}" for m in metrics)
                             + f" | {sum(bool(r['star_rejected']) for r in sel)} |")
    (out / "bench_scores.md").write_text("\n".join(lines) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--data", default="~/data/imaging_contests")
    s.add_argument("--out", default="~/data/imaging_contests/bench")
    s.add_argument("--phantoms", type=int, default=6)
    s.add_argument("--draws", type=int, default=2)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--coverage", action="append", help="only these coverages (repeatable)")
    s.add_argument("--cal", default=None, help="calibration offsets 'v2_rel,cp_deg', e.g. 0.02,1.0")
    ls = sub.add_parser("list")
    ls.add_argument("--out", default="~/data/imaging_contests/bench")
    r = sub.add_parser("run")
    r.add_argument("--id", type=int, required=True, help="dataset index x len(configs) + config index")
    r.add_argument("--bench", default="~/data/imaging_contests/bench")
    r.add_argument("--out", default="bench_results")
    r.add_argument("--configs", default=",".join(DEFAULT_CONFIGS),
                   help=f"comma-separated configuration names (opt-in: {', '.join(OPT_IN)})")
    r.add_argument("--smoke", action="store_true")
    r.add_argument("--seed", type=int, default=0, help="random key of the ensemble arm")
    sc = sub.add_parser("score")
    sc.add_argument("--bench", default="~/data/imaging_contests/bench")
    sc.add_argument("--out", default="bench_results")
    args = parser.parse_args()
    if args.cmd == "run":
        print(run(pathlib.Path(args.bench).expanduser(), pathlib.Path(args.out).expanduser(), args.id,
                  args.configs.split(","), args.smoke, args.seed))
        return
    if args.cmd == "score":
        rows = score(pathlib.Path(args.bench).expanduser(), pathlib.Path(args.out).expanduser())
        print((pathlib.Path(args.out).expanduser() / "bench_scores.md").read_text(), f"{len(rows)} results")
        return
    if args.cmd == "simulate":
        cal = None
        if args.cal:
            a, b = (float(x) for x in args.cal.split(","))
            cal = {"v2_rel": a, "cp_deg": b}
        made = simulate(pathlib.Path(args.data).expanduser(), pathlib.Path(args.out).expanduser(),
                        args.phantoms, args.draws, args.seed, cal, args.coverage)
        print(f"{len(made)} datasets")
    else:
        index = json.loads((pathlib.Path(args.out).expanduser() / "index.json").read_text())
        for i, name in enumerate(index):
            print(i, name)


if __name__ == "__main__":
    main()
