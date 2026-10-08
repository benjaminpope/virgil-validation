"""Re-analyse undersampled multi-epoch binaries with virgil.aliases.fit_orbit_aliases.

For each system the epochs are too few or too far apart to count orbital cycles by eye. The
closure phases (and, for PIONIER, the V2) of every epoch are fitted directly by a Keplerian
binary in each period-alias band N = round(T/P), and the bands are weighed by their marginal
likelihoods (virgil PR #319, design/orbit_aliases.md in virgil). Positions per epoch seed the
fits and are plotted as diagnostics only.

Install virgil from the Stage A branch (until the PR merges):

    pip install git+https://github.com/benjaminpope/virgil@claude/orbit-aliases-d040f2
    pip install -e ".[orbit-aliases]"   # the extra pins virgil (Stage A commit) and corner

Usage (the fits are heavy JAX: OzSTAR, see ozstar_scripts/scripts/orbit_aliases; on a laptop
only --list is safe):

    python scripts/reanalyse_orbits.py --list
    python scripts/reanalyse_orbits.py --system gl229 --out DIR [--n-is N] [--n-candidates N]
                                       [--n-refine N] [--p-range LO,HI] [--dry-run]

Data root: $ESO_BIN_ROOT (default /fred/oz440/bpope/eso_binaries), holding gravity/oifits/<night>/
(Gl 229) and trackb/<system>/{system.json,oifits/} (Track B, see design/plan_eso_binaries.md).

Writes DIR/<system>/{bands.json, meta.json, sky.png, corner.png, residuals.png} and prints the
band table with the raw chi2/N on the quoted errors first.

Independence rule (AGENTS.md): reference orbits and every orbit track drawn here come from
src/crosscheck/orbits.py (our own Kepler solver), never from virgil. The fit itself is virgil's.
Systems skipped for want of a reusable loader or enough epochs are listed in
docs/orbits/aliases.md.
"""

import argparse
import dataclasses
import json
import os
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

ROOT_DEFAULT = "/fred/oz440/bpope/eso_binaries"
MJD_JD = 2400000.5
P_WINDOW = 1.25  # Track B: prior window P_ref / 1.25 .. 1.25 P_ref (a prior window, not a blind search)
GL229_BAND = (2.05e-6, 2.18e-6)
# Exposure left out of the Gl 229 night fits (fit_nights.py in the eso_binaries scripts), with the reason:
GL229_EXCLUDE = {
    "GRAVI.2023-12-26T05_51_29.891": "closure phases +38 to +57 deg on all triangles, errors 5-10x the night's",
}
# Xuan et al. 2024, Nature 634, 1070, Table 1 (PMOIRED fit). omega is the primary's: the
# secondary's, as in the Kepler convention of crosscheck.orbits, is +180 deg.
XUAN_2024 = dict(P=12.134, t_peri=60377.88, ecc=0.234, inc=31.4, omega=180.7 + 180.0, Omega=213.0, a_mas=0.0424 * 173.574,
                 source="Xuan+2024 Table 1 (PMOIRED); Omega node convention unverified: compare mod 180 deg")


@dataclasses.dataclass
class Loaded:
    """One system, ready to fit."""

    names: list  # epoch labels
    data: list  # OIData per epoch (all observables used by the fit)
    grid_data: list  # the same, thinned for the position grid (a seed and a diagnostic)
    grid_half: float  # position grid half-width (mas)
    a_range: tuple
    reference: dict = None  # orbit elements in crosscheck.orbits convention, or None
    ref_positions: list = None  # published (mjd, dra, ddec) per epoch, or None
    notes: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class System:
    label: str
    instrument: str
    reference_source: str
    p_range: tuple
    loader: object
    n_epochs: str  # for --list


def _thin(data, max_ch=24):
    """Every k-th spectral channel of a GRAVITY epoch, for the position grid."""
    ch = np.unique(np.asarray(data.wavel, float))
    if ch.size <= max_ch:
        return data
    keep = ch[:: int(np.ceil(ch.size / max_ch))]
    eps = 0.25 * float(np.min(np.diff(ch)))
    return data.select(ranges=[(w - eps, w + eps) for w in keep])


def _grid_half(bmax, wmin, half):
    step = wmin / (4 * bmax) * 180.0 / np.pi * 3600e3
    return half, float(step)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_gl229(root):
    """Gl 229 Ba-Bb, GRAVITY SC closure phases, one OIData per night (as fit_nights.py in the
    eso_binaries scripts: V2 flagged, closure phases outside 2.05-2.18 um flagged)."""
    import glob

    from virgil.oidata import OIData
    from virgil.oifits import read_oifits

    base = pathlib.Path(root) / "gravity" / "oifits"
    nights = sorted(p for p in base.iterdir() if p.is_dir())
    if not nights:
        raise FileNotFoundError(f"no night directories in {base}")
    names, data, notes = [], [], []
    for n in nights:
        paths = sorted(glob.glob(str(n / "*_DUAL_SCI_VIS.fits")))
        kept = [p for p in paths if os.path.basename(p).split("_DUAL")[0] not in GL229_EXCLUDE]
        notes += [f"{n.name}: left out {os.path.basename(p)}" for p in paths if p not in kept]
        rec = dict(read_oifits(kept, insname="GRAVITY_SC"))
        wl = np.asarray(rec["wavel"], float)
        i1 = np.asarray(rec["i_cps1"])
        lam = wl[i1] if wl.size > 1 else np.full(i1.shape, wl.item())
        rec["vis_flag"] = np.ones_like(np.asarray(rec["vis_flag"]), dtype=bool)
        rec["phi_flag"] = np.asarray(rec["phi_flag"], bool) | (lam < GL229_BAND[0]) | (lam > GL229_BAND[1])
        names.append(n.name)
        data.append(OIData(rec))
    return Loaded(names, data, data, 12.0, (1.0, 15.0), reference=XUAN_2024, notes=notes)


def _trackb_system(root, name):
    p = pathlib.Path(root) / "trackb" / name / "system.json"
    return json.load(open(p)), p.parent


def _trackb_reference(s):
    ro = s.get("reference_orbit") or {}
    need = ("P_day", "T0", "e", "omega_deg", "Omega_deg", "i_deg", "a_mas")
    if not all(k in ro for k in need):
        return None
    return dict(P=ro["P_day"][0], t_peri=ro["T0"][0] - MJD_JD, ecc=ro["e"][0], inc=ro["i_deg"][0],
                omega=ro["omega_deg"][0] + 180.0,  # RV omega is the primary's
                Omega=ro["Omega_deg"][0], a_mas=ro["a_mas"][0], source=s["reference"].get("positions_source", "")[:60]
                or s["reference"].get("bibcode", ""))


def make_trackb_loader(name):
    def load(root):
        import trackb_fit as T

        s, _ = _trackb_system(root, name)
        p_json = s["reference_orbit"]["P_day"][0]  # one source for P_ref: TRACKB; a mismatch is an error
        if abs(p_json / TRACKB[name][2] - 1) > 1e-4:
            raise ValueError(f"{name}: TRACKB period {TRACKB[name][2]} != system.json {p_json}")
        eps = [e for e in T.epochs_of(name, s, pathlib.Path(root) / "trackb") if e["counted"]]
        names, data, gdata, notes, refpos, seps = [], [], [], [], [], []
        for e in eps:
            d, inst, dropped = T.load(e["files"])
            notes += [f"{e['date']}: dropped {x['file']}" for x in dropped]
            names.append(e["date"])
            data.append(d)
            gdata.append(T.grid_data(d, inst, T.geometry(d)[3]))
            if e["published"]:
                dra, ddec = e["published"]["pos"]
                refpos.append((e["mjd"], dra, ddec))
                seps.append(float(np.hypot(dra, ddec)))
        sep = max(seps) if seps else 20.0
        lo = max(0.3 * min(seps), 0.2) if seps else 0.5
        return Loaded(names, data, gdata, 1.5 * sep, (lo, 3.0 * sep), reference=_trackb_reference(s),
                      ref_positions=refpos or None, notes=notes)
    return load


# Track B systems with a tabulated reference period and at least four counted epochs on disk.
# Others are skipped (docs/orbits/aliases.md): span shorter than the period (kap_vel, nn_del), 1-2 epochs (9sgr, hd152314, hd168137, kq_vel, cpd-71_172,
# tyc1703-394-1), no tabulated period (zet_boo, eta_oph), mixed instruments or a manifest conflict
# (del_cir, tz_for). Gl 229's loader is the one outside Track B.
TRACKB = {
    "al_dor": ("AL Dor", "PIONIER", 14.90537),
    "alf_equ": ("alpha Equ", "PIONIER", 98.8045),
    "hd188088": ("HD 188088", "GRAVITY", 46.81614),
    "hd210763": ("HD 210763", "GRAVITY", 42.38113),
    "hd41255": ("HD 41255", "GRAVITY", 148.329),
    "hd70937": ("HD 70937", "GRAVITY", 27.8858),
    "omi_leo": ("o Leo", "GRAVITY", 14.498068),
    "psi_cen": ("psi Cen", "PIONIER", 38.8121),
}

SYSTEMS = {
    "gl229": System("Gl 229 Ba-Bb", "GRAVITY", XUAN_2024["source"], (10.0, 14.0), load_gl229, "7 nights"),
}
for _k, (_label, _inst, _p) in TRACKB.items():
    SYSTEMS[_k] = System(_label, _inst, "Track B reference orbit (system.json, Gallenne+ tables)", (_p / P_WINDOW, _p * P_WINDOW),
                         make_trackb_loader(_k), "Track B")


# ---------------------------------------------------------------------------
# Fit and outputs
# ---------------------------------------------------------------------------
def positions_for(loaded, t_ref):
    """Per-epoch positions (seeds and diagnostic): epoch_positions on a grid at lambda_min / 4 B_max."""
    from virgil.epochs import Epochs, epoch_positions

    wmin = min(float(np.min(np.asarray(d.wavel, float))) for d in loaded.grid_data)
    bmax = max(float(np.max(np.hypot(np.asarray(d.u, float), np.asarray(d.v, float)))) for d in loaded.grid_data)
    half, step = _grid_half(bmax, wmin, loaded.grid_half)
    n = 2 * int(np.ceil(half / step)) + 1
    ax = np.linspace(-half, half, n)
    grid = dict(dra=ax, ddec=ax, flux=np.geomspace(0.02, 1.0, 13))
    ep = epoch_positions(Epochs(dict(zip(loaded.names, loaded.grid_data))), grid, n_peaks=5)
    return ep.positions(t_ref=t_ref), ep


def ref_track(ref, t):
    """Reference relative position at times t from crosscheck.orbits (the independent Kepler solver)."""
    from crosscheck import orbits as O

    return O.position(t, ref["P"], ref["t_peri"], ref["ecc"], ref["inc"], ref["omega"], ref["Omega"], ref["a_mas"])


def sample_track(s, i, t_ref, t):
    from crosscheck import orbits as O

    return O.position(t, s["period"][i], t_ref + s["dt_peri"][i], s["ecc"][i], s["inc"][i], s["omega"][i], s["Omega"][i],
                      s["a_mas"][i])


def element_comparison(result, ref, name):
    """Winning-band posterior against the reference elements (our Kepler convention). Omega is folded mod 180 deg
    with omega shifted (crosscheck.orbits.fold_samples); for the near-equal-mass systems (crosscheck.astrometry.NEAR_EQUAL:
    al_dor, hd41255, hd188088), where closure phases barely tell the twins apart, omega is also compared mod 180 deg."""
    from crosscheck import astrometry as A
    from crosscheck import orbits as O

    s = result.samples[result.best.n]
    om, Om, mirror = O.fold_samples(s["omega"], s["Omega"], ref["Omega"])
    twin = name in A.NEAR_EQUAL
    out = {}
    for k, v, r in (("period", s["period"], ref["P"]), ("ecc", s["ecc"], ref["ecc"]), ("inc", s["inc"], ref["inc"]),
                    ("Omega", Om, ref["Omega"]), ("omega", om, ref["omega"]), ("a_mas", s["a_mas"], ref["a_mas"])):
        v = np.asarray(v, float)
        if k == "omega":
            d = (v - r + (90.0 if twin else 180.0)) % (180.0 if twin else 360.0) - (90.0 if twin else 180.0)
        else:
            d = v - r
        sd = float(np.std(d))
        out[k] = dict(ref=r, mean_offset=float(np.mean(d)), sd=sd, pull=float(np.mean(d) / sd) if sd > 0 else None)
    out["twin_fold"] = twin
    out["mirror_fraction"] = float(np.mean(mirror))
    return out


def plot_sky(path, result, loaded, epos, ref, n_draw=60):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    s = result.samples[result.best.n]
    n = len(s["period"])
    rng = np.random.default_rng(0)
    pbest = result.best.best["period"]
    tt = result.t_ref + np.linspace(0, pbest, 400)
    for i in rng.choice(n, size=min(n_draw, n), replace=False):
        x, y = sample_track(s, i, result.t_ref, tt)
        ax.plot(x, y, color="C0", alpha=0.12, lw=0.8)
    if ref is not None:
        tr = result.t_ref + np.linspace(0, ref["P"], 400)
        x, y = ref_track(ref, tr)
        ax.plot(x, y, color="k", lw=1.5, ls="--", label=f"reference orbit ({ref['source'][:40]})")
    ax.errorbar(epos.dra, epos.ddec, xerr=np.sqrt(epos.cov[:, 0, 0]), yerr=np.sqrt(epos.cov[:, 1, 1]), fmt="o", color="C3",
                ms=4, label="per-epoch positions (diagnostic)")
    if loaded.ref_positions:
        rp = np.array(loaded.ref_positions)
        ax.plot(rp[:, 1], rp[:, 2], "x", color="k", label="published positions")
    ax.plot([0], [0], "*", color="gold", ms=12, mec="k")
    ax.set_xlabel("dRA (mas, East left)")
    ax.set_ylabel("dDec (mas)")
    ax.invert_xaxis()
    ax.set_aspect("equal")
    ax.set_title(f"N={result.best.n}, P={pbest:.3f} d, p={result.best.p:.2f}")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_corner(path, result):
    import matplotlib

    matplotlib.use("Agg")
    import corner
    import matplotlib.pyplot as plt

    s = result.samples[result.best.n]
    keys = ["period", "dt_peri", "ecc", "inc", "omega", "Omega", "a_mas", "flux"]
    arr = np.column_stack([np.asarray(s[k], float) for k in keys])
    ok = np.ptp(arr, axis=0) > 0
    fig = corner.corner(arr[:, ok], labels=[k for k, o in zip(keys, ok) if o], show_titles=True, title_fmt=".3g")
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_residuals(path, result, loaded):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from virgil.likelihood import whitened_residuals
    from virgil.models import OrbitalBinary
    from virgil.orbits import KeplerOrbit

    b = result.best.best
    orbit = KeplerOrbit(b["period"], b["dt_peri"], b["ecc"], b["inc"], b["omega"], b["Omega"], b["a_mas"], t_ref=result.t_ref)
    model = OrbitalBinary(orbit, b["flux"])
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(loaded.data) + 3), 4))
    rms, chi2, n_ind = [], 0.0, 0
    for k, d in enumerate(loaded.data):
        r = np.asarray(whitened_residuals(model, d))  # the full vector, as trackb_fit.py sums it
        ax.plot(k + 0.15 * rng.standard_normal(r.size), r, ".", ms=2, alpha=0.4, color="C0")
        rms.append(float(np.sqrt(np.mean(r**2))))
        chi2 += float(np.sum(r**2))
        n_ind += int(d.n_independent)
    ax.plot(range(len(rms)), rms, "o", color="C3", label="rms (raw chi/N^0.5)")
    ax.plot(range(len(rms)), -np.array(rms), "o", color="C3")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(loaded.names)), loaded.names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("residual / quoted error")
    ax.set_title(f"residuals / quoted error at the best orbit (N={result.best.n})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return rms, chi2, n_ind


def print_table(result):
    print(f"{'N':>4} {'P (d)':>9} {'chi2/N':>7} {'s_min':>6} {'s_max':>6} {'logZ':>9} {'logZ_IS':>9} {'p':>7} {'ESS':>7}  flag")
    for r in result.table():
        sc = r["scales"] or [float("nan")]
        print(f"{r['n']:>4d} {r['period'] or float('nan'):>9.4f} {r['chi2_red']:>7.2f} {min(sc):>6.2f} {max(sc):>6.2f} "
              f"{r['log_z']:>9.2f} {r['log_z_is']:>9.2f} {r['p']:>7.3f} {r['ess']:>7.0f}  {'FLAG' if r['flagged'] else ''}")


def run(name, args):
    from virgil.aliases import fit_orbit_aliases

    sysd = SYSTEMS[name]
    root = os.environ.get("ESO_BIN_ROOT", ROOT_DEFAULT)
    outdir = pathlib.Path(args.out) / name
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    loaded = sysd.loader(root)
    p_range = tuple(float(x) for x in args.p_range.split(",")) if args.p_range else sysd.p_range
    times = np.array([float(np.mean(np.asarray(d.mjd, float))) for d in loaded.data])
    t_ref = float(np.round(times.mean(), 1))
    print(f"{name}: {len(loaded.data)} epochs, MJD {times.min():.1f}-{times.max():.1f}, p_range {p_range}, "
          f"{sum(int(d.n_independent) for d in loaded.data)} independent observables")
    for n in loaded.notes:
        print("note:", n)
    if args.dry_run:
        return
    positions, epos = positions_for(loaded, t_ref)
    result = fit_orbit_aliases(loaded.data, p_range, positions=positions, times=times, t_ref=t_ref, a_range=loaded.a_range,
                               ecc_max=args.ecc_max, n_candidates=args.n_candidates, n_refine=args.n_refine, n_is=args.n_is,
                               n_samples=args.n_samples)
    print_table(result)
    result.to_json(outdir / "bands.json", n_samples=args.n_samples)
    ref = loaded.reference
    meta = dict(system=name, label=sysd.label, instrument=sysd.instrument, epochs=loaded.names, mjd=times.tolist(),
                p_range=p_range, args={k: v for k, v in vars(args).items() if k != "list"}, elapsed_s=time.time() - t0,
                notes=loaded.notes, reference=ref,
                virgil=_virgil_version())
    if ref is not None:
        hit = next((b for b in result.bands if b.p_lo <= ref["P"] <= b.p_hi), None)  # virgil's own band edges
        meta["reference_band"] = dict(n=None if hit is None else hit.n, period=ref["P"],
                                      p=None if hit is None else hit.p, is_winner=hit is not None and hit is result.best)
        meta["elements"] = element_comparison(result, ref, name)
        print(f"reference (P={ref['P']:.4f} d) lies in band N={meta['reference_band']['n']}; winner N={result.best.n}")
    json.dump(meta, open(outdir / "meta.json", "w"), indent=1, default=str)
    plot_sky(outdir / "sky.png", result, loaded, epos, ref)
    plot_corner(outdir / "corner.png", result)
    rms, chi2, n_ind = plot_residuals(outdir / "residuals.png", result, loaded)
    meta["rms_residual"] = rms
    meta["chi2_raw"], meta["n_independent"] = chi2, n_ind
    meta["chi2_raw_per_n"] = chi2 / n_ind
    meta["chi2_check"] = dict(ours=chi2 / n_ind, virgil=result.best.chi2_red,
                              agrees=bool(abs(chi2 / n_ind / result.best.chi2_red - 1) < 0.05))
    if not meta["chi2_check"]["agrees"]:
        print(f"WARNING: raw chi2/N here {chi2 / n_ind:.3f} differs from the band table's {result.best.chi2_red:.3f}")
    json.dump(meta, open(outdir / "meta.json", "w"), indent=1, default=str)
    (outdir / f"{name}.done").write_text("ok\n")
    print(f"wrote {outdir} in {time.time() - t0:.0f} s")


def _virgil_version():
    try:
        import importlib.metadata as im

        v = im.version("virgil-astro")
    except Exception:
        return None
    try:  # the commit a pip install from git recorded
        du = json.loads(im.distribution("virgil-astro").read_text("direct_url.json") or "{}")
        commit = du.get("vcs_info", {}).get("commit_id")
    except Exception:
        commit = None
    return dict(version=v, commit_id=commit or os.environ.get("VIRGIL_COMMIT"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the system names, one per line, and exit")
    ap.add_argument("--system", help="system name (see --list)")
    ap.add_argument("--out", default="orbit_aliases_out")
    ap.add_argument("--n-is", type=int, default=2000, dest="n_is")
    ap.add_argument("--n-candidates", type=int, default=100, dest="n_candidates")
    ap.add_argument("--n-refine", type=int, default=6, dest="n_refine")
    ap.add_argument("--n-samples", type=int, default=1000, dest="n_samples")
    ap.add_argument("--ecc-max", type=float, default=0.9, dest="ecc_max")
    ap.add_argument("--p-range", default=None, dest="p_range", help="LO,HI days (default: the registry's)")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run", help="load the data and print its shape, no fit")
    args = ap.parse_args(argv)
    if args.list:
        for k in SYSTEMS:
            print(k)
        return 0
    if args.system not in SYSTEMS:
        ap.error(f"--system must be one of {', '.join(SYSTEMS)}")
    run(args.system, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
