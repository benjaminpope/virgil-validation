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

Writes DIR/<system>/{bands/band_<N>.{pkl,json} (per band, as fitted; a rerun resumes), bands.json, meta.json, sky.png, corner.png, residuals.png} and prints the
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
import traceback

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
XUAN_2024 = dict(
    P=12.134,
    t_peri=60377.88,
    ecc=0.234,
    inc=31.4,
    omega=180.7 + 180.0,
    Omega=213.0,
    a_mas=0.0424 * 173.574,
    source="Xuan+2024 Table 1 (PMOIRED); Omega node convention unverified: compare mod 180 deg",
)


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
    sign_free: bool = False  # near-equal fluxes: position seeds known only up to a sign (virgil#324)
    primary_diam: object = None  # the primary's uniform-disk diameter: None (point), mas, or "fit"


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
    """Gl 229 Ba-Bb, GRAVITY SC V2 and closure phases, one OIData per night (as fit_nights.py in the
    eso_binaries scripts, but with V2 kept: V2 and closure phases outside 2.05-2.18 um flagged)."""
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
        kept = [
            p
            for p in paths
            if os.path.basename(p).split("_DUAL")[0] not in GL229_EXCLUDE
        ]
        notes += [
            f"{n.name}: left out {os.path.basename(p)}" for p in paths if p not in kept
        ]
        rec = dict(read_oifits(kept, insname="GRAVITY_SC"))
        wl = np.asarray(rec["wavel"], float)
        i1 = np.asarray(rec["i_cps1"])
        lam = wl[i1] if wl.size > 1 else np.full(i1.shape, wl.item())
        rec["vis_flag"] = (  # V2 over the same band as the closure phases: both are fitted
            np.asarray(rec["vis_flag"], bool)
            | (wl < GL229_BAND[0])
            | (wl > GL229_BAND[1])
        )
        rec["phi_flag"] = (
            np.asarray(rec["phi_flag"], bool)
            | (lam < GL229_BAND[0])
            | (lam > GL229_BAND[1])
        )
        names.append(n.name)
        data.append(OIData(rec))
    return Loaded(
        names, data, data, 12.0, (1.0, 15.0), reference=XUAN_2024, notes=notes
    )


def _trackb_system(root, name):
    p = pathlib.Path(root) / "trackb" / name / "system.json"
    return json.load(open(p)), p.parent


def _trackb_reference(s, primary_omega=False):
    """Track B reference orbit in crosscheck.orbits convention, via `crosscheck.references.trackb_reference`.

    Solved problem, primary versus secondary omega: the tabulated omega reproduces the published companion positions
    as it stands for 7 of the 8 Track B systems (the old +180 deg gave their point reflection); alpha Equ's is
    omega_1, the primary's. ``primary_omega`` is that ``omega_of`` choice made explicit (``PRIMARY_OMEGA``, to be
    removed when the Track B files carry ``reference_orbit.omega_of``).
    """
    from crosscheck import references as REF

    return REF.trackb_reference(s, omega_of="primary" if primary_omega else "secondary")


def _check_published(ref, refpos, limit_mas=0.5):
    """Median miss (mas) of the reference orbit against the published positions; raises if it exceeds limit_mas.

    A check, not an adjustment: a large miss means a wrong reference (convention, T0 epoch), not a sign to choose.
    It is an error, not a note: a wrong reference (alpha Equ, 7.4 mas) once went through as a warning."""
    from crosscheck import orbits as O

    rp = np.array(refpos)
    x, y = O.position(
        rp[:, 0], ref["P"], ref["t_peri"], ref["ecc"], ref["inc"], ref["omega"], ref["Omega"], ref["a_mas"]
    )
    miss = float(np.median(np.hypot(x - rp[:, 1], y - rp[:, 2])))
    if miss > limit_mas:
        raise ValueError(
            f"reference orbit misses the published positions by a median {miss:.2f} mas (limit {limit_mas} mas): "
            "the reference convention or epoch is wrong"
        )
    return miss, None


def _read_with_vis(T):
    """trackb_fit._read, but GRAVITY keeps V2 beside the closure phases (the Track B criteria are closure phases only)."""
    read_tb = T._read

    def read(files, inst):
        if inst != "GRAVITY":
            return read_tb(files, inst)
        from virgil.oidata import OIData
        from virgil.oifits import read_oifits

        return OIData(read_oifits([str(f) for f in files], insname="GRAVITY_SC")).select(
            ranges=T.GRAVITY_RANGES, observables=("vis", "phi")
        )

    return read


def make_trackb_loader(name, primary_omega=False):
    def load(root):
        import trackb_fit as T

        T._read = _read_with_vis(T) if not getattr(T._read, "_vis", False) else T._read
        T._read._vis = True

        s, _ = _trackb_system(root, name)
        p_json = s["reference_orbit"]["P_day"][
            0
        ]  # one source for P_ref: TRACKB; a mismatch is an error
        if abs(p_json / TRACKB[name][2] - 1) > 1e-4:
            raise ValueError(
                f"{name}: TRACKB period {TRACKB[name][2]} != system.json {p_json}"
            )
        eps = [
            e
            for e in T.epochs_of(name, s, pathlib.Path(root) / "trackb")
            if e["counted"]
        ]
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
        ref = _trackb_reference(s, primary_omega)
        if ref is not None and refpos:
            miss, note = _check_published(ref, refpos)
            ref["median_miss_published_mas"] = miss
            if note:
                notes.append(note)
        sep = max(seps) if seps else 20.0
        lo = max(0.3 * min(seps), 0.2) if seps else 0.5
        return Loaded(
            names,
            data,
            gdata,
            1.5 * sep,
            (lo, 3.0 * sep),
            reference=ref,
            ref_positions=refpos or None,
            notes=notes,
        )

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

# The primary's angular diameter (mas) for the uniform-disk model: "fit" where the primary is resolved on the
# longest baselines (UD >= ~0.4 mas, i.e. V^2(B_max) < 0.98 in K), else the literature value fixed (JSDC, VizieR
# II/346, which agrees with the Track B system.json "diameters_mas" wherever both exist), else None (unresolved).
PRIMARY_DIAM = {
    "gl229": None,  # Ba: no catalogue value; R ~0.08 Rsun at 5.76 pc is ~0.13 mas, V^2 > 0.99: a point
    "omi_leo": "fit",  # 1.30 +/- 0.08 mas LD (Gallenne+2023, 2023A&A...672A.119G); strongly resolved
    "al_dor": 0.156,  # LD, H (Graczyk+2019, adopted in Gallenne+2019, 2019A&A...632A..31G); unresolved
    "alf_equ": "fit",  # JSDC UD ~1.48 mas (no paper value; Halbwachs+2020 assume unresolved); strongly resolved
    "hd188088": "fit",  # 0.746 +/- 0.007 mas LD, K (Gallenne+2023 Table 3)
    "hd210763": 0.367,  # 0.367 +/- 0.009 mas LD, K (Gallenne+2023 Table 3); marginal: V^2 ~0.98, so fixed
    "hd41255": 0.245,  # 0.245 +/- 0.002 mas LD, K (Gallenne+2023 Table 3); unresolved
    "hd70937": "fit",  # 0.413 +/- 0.005 mas LD, K (Gallenne+2023 Table 3); just resolved
    "psi_cen": "fit",  # 0.424 +/- 0.007 mas LD, H (Mantegazza+2010, as used in Gallenne+2019); resolved at H
}
# alpha Equ: the tabulated omega is "omega_1, refers to the motion of the primary" (system.json): +180 deg
PRIMARY_OMEGA = ("alf_equ",)

SYSTEMS = {
    "gl229": System(
        "Gl 229 Ba-Bb",
        "GRAVITY",
        XUAN_2024["source"],
        (10.0, 14.0),
        load_gl229,
        "7 nights",
        primary_diam=PRIMARY_DIAM["gl229"],
    ),
}
TWINS = ("al_dor", "hd41255", "hd188088")  # flux ratios 0.9-1.0
for _k, (_label, _inst, _p) in TRACKB.items():
    SYSTEMS[_k] = System(
        _label,
        _inst,
        "Track B reference orbit (system.json, Gallenne+ tables)",
        (_p / P_WINDOW, _p * P_WINDOW),
        make_trackb_loader(_k, primary_omega=_k in PRIMARY_OMEGA),
        "Track B",
        sign_free=_k in TWINS,
        primary_diam=PRIMARY_DIAM[_k],
    )


# ---------------------------------------------------------------------------
# Fit and outputs
# ---------------------------------------------------------------------------
def positions_for(loaded, t_ref, max_widen=5):
    """Per-epoch positions (seeds and diagnostic): epoch_positions on a grid at lambda_min / 4 B_max.

    A best peak at the grid edge may stand for a companion outside it, so the half-width is widened by
    1.5 (up to ``max_widen`` times) until no epoch's best peak is at the edge. Returns (positions, epochs,
    info) with info = dict(half_widths, edge_epochs) recording what happened; an epoch still at the edge
    after the last widening is listed in ``edge_epochs`` and the run goes on (the positions only seed the fits).
    """
    from virgil.epochs import Epochs, epoch_positions

    wmin = min(float(np.min(np.asarray(d.wavel, float))) for d in loaded.grid_data)
    bmax = max(
        float(np.max(np.hypot(np.asarray(d.u, float), np.asarray(d.v, float))))
        for d in loaded.grid_data
    )
    half = loaded.grid_half
    halves = []
    for _ in range(max_widen + 1):
        half, step = _grid_half(bmax, wmin, half)
        n = 2 * int(np.ceil(half / step)) + 1
        ax = np.linspace(-half, half, n)
        grid = dict(dra=ax, ddec=ax, flux=np.geomspace(0.02, 1.0, 13))
        ep = epoch_positions(
            Epochs(dict(zip(loaded.names, loaded.grid_data))), grid, n_peaks=5
        )
        halves.append(float(half))
        edge = np.asarray(ep.edge, bool)
        if not edge.any():
            break
        half = 1.5 * half
    info = dict(
        half_widths=halves, edge_epochs=[nm for nm, e in zip(loaded.names, edge) if e]
    )
    # The EpochPositions itself, not ep.positions(t_ref): fit_orbit_aliases then drops the flagged alias-peak
    # nights from the seeds (skip_flagged). run() passes primary_diam explicitly, so no 'auto' resolution here.
    return ep, ep, info


def ref_track(ref, t):
    """Reference relative position at times t from crosscheck.orbits (the independent Kepler solver)."""
    from crosscheck import orbits as O

    return O.position(
        t,
        ref["P"],
        ref["t_peri"],
        ref["ecc"],
        ref["inc"],
        ref["omega"],
        ref["Omega"],
        ref["a_mas"],
    )


def sample_track(s, i, t_ref, t):
    from crosscheck import orbits as O

    return O.position(
        t,
        s["period"][i],
        t_ref + s["dt_peri"][i],
        s["ecc"][i],
        s["inc"][i],
        s["omega"][i],
        s["Omega"][i],
        s["a_mas"][i],
    )


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
    for k, v, r in (
        ("period", s["period"], ref["P"]),
        ("ecc", s["ecc"], ref["ecc"]),
        ("inc", s["inc"], ref["inc"]),
        ("Omega", Om, ref["Omega"]),
        ("omega", om, ref["omega"]),
        ("a_mas", s["a_mas"], ref["a_mas"]),
    ):
        v = np.asarray(v, float)
        if k == "omega":
            d = (v - r + (90.0 if twin else 180.0)) % (180.0 if twin else 360.0) - (
                90.0 if twin else 180.0
            )
        else:
            d = v - r
        sd = float(np.std(d))
        out[k] = dict(
            ref=r,
            mean_offset=float(np.mean(d)),
            sd=sd,
            pull=float(np.mean(d) / sd) if sd > 0 else None,
        )
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
        ax.plot(
            x,
            y,
            color="k",
            lw=1.5,
            ls="--",
            label=f"reference orbit ({ref['source'][:40]})",
        )
    ax.errorbar(
        epos.dra,
        epos.ddec,
        xerr=np.sqrt(epos.cov[:, 0, 0]),
        yerr=np.sqrt(epos.cov[:, 1, 1]),
        fmt="o",
        color="C3",
        ms=4,
        label="per-epoch positions (diagnostic)",
    )
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
    if (
        not ok.any()
    ):  # every draw identical (a degenerate importance sample): nothing to plot
        raise ValueError(
            "no parameter varies across the posterior draws (importance sample has collapsed)"
        )
    fig = corner.corner(
        arr[:, ok],
        labels=[k for k, o in zip(keys, ok) if o],
        show_titles=True,
        title_fmt=".3g",
    )
    fig.savefig(path, dpi=110)
    plt.close(fig)


VIS_AMP = (0.01, 1.5)  # per-epoch V^2 amplitude, log-uniform (virgil vis_amp): a contrast loss the same on all baselines


def best_scene(result, primary_diam, params=None):
    """The best band's scene (or that of ``params``), as the fit built it: point or uniform-disk primary, one
    orbiting companion."""
    from virgil.aliases import _UDBinary
    from virgil.models import OrbitalBinary
    from virgil.orbits import KeplerOrbit

    b = result.best.best if params is None else params
    orbit = KeplerOrbit(
        b["period"], b["dt_peri"], b["ecc"], b["inc"], b["omega"], b["Omega"], b["a_mas"], t_ref=result.t_ref
    )
    if primary_diam == "fit" and "diam" not in result.best.best:
        raise ValueError("primary_diam is 'fit' but the best band has no fitted 'diam': cannot rebuild the fit's scene")
    diam = (b.get("diam") or result.best.best["diam"]) if primary_diam == "fit" else primary_diam
    return _UDBinary(orbit, b["flux"], diam) if diam else OrbitalBinary(orbit, b["flux"])


def fit_vis_amps(result):
    """The fit's own per-epoch V^2 amplitudes (virgil vis_amp, sampled log-uniform) from the best band, or None."""
    amps = [e.get("vis_amp") for e in result.best.epochs]
    return amps if amps and all(a is not None for a in amps) else None


def epoch_residuals(model, d, amp=None):
    """Whitened residuals of one epoch (virgil's whitened_residuals, assumed (model - obs) / d_vis, diagonal), with
    the squared visibilities scaled by the fit's own amplitude ``amp`` for this epoch, if given."""
    from virgil.epochs import _blocks
    from virgil.likelihood import whitened_residuals

    r = np.array(whitened_residuals(model, d))
    blocks = _blocks(d)
    if amp is None or blocks[0][0] != "vis" or not d.v2_flag:
        return r
    n = blocks[0][2]
    err = np.asarray(d.d_vis, float).ravel()
    obs = np.asarray(d.vis, float).ravel()
    r[:n] = (amp * (obs + r[:n] * err) - obs) / err
    return r


def block_chi2(result, loaded, primary_diam=None):
    """Raw chi2 and independent-observable count of the V2 and the closure-phase blocks at the best orbit,
    summed over epochs ("vis" and "phi"), on the quoted errors, with the fit's V2 amplitudes if it fitted them.
    This is virgil's own model: it checks the bookkeeping, not the model (see residual_stats)."""
    from virgil.epochs import _blocks

    model = best_scene(result, primary_diam)
    amps = fit_vis_amps(result) or [None] * len(loaded.data)
    out = {}
    for d, amp in zip(loaded.data, amps):
        r = epoch_residuals(model, d, amp)
        for name, a, z, nu, _ in _blocks(d):
            c, n = out.get(name, (0.0, 0))
            out[name] = (c + float(np.sum(r[a:z] ** 2)), n + int(nu))
    return {k: dict(chi2=c, n=n, chi2_per_n=c / n if n else None) for k, (c, n) in out.items()}


def residual_stats(result, loaded, primary_diam=None):
    """Raw residuals of every epoch at the best orbit, whitened by the quoted errors, as trackb_fit.py sums them.

    Returns (residuals per epoch, rms per epoch, chi2, n_independent). This is NOT an independent test of the
    model: the model (_UDBinary / OrbitalBinary, whitened_residuals) and the V2 amplitudes are virgil's own, so
    agreement with chi2_red checks only the summation, the observable count and the amplitudes. It is kept apart
    from the plotting so a plotting failure cannot remove it. A model-independent chi2 (V2 and CP from
    crosscheck's UD + point-companion visibility, against d.vis / d.d_vis) is not yet implemented.
    """
    model = best_scene(result, primary_diam)
    amps = fit_vis_amps(result) or [None] * len(loaded.data)
    res = [epoch_residuals(model, d, a) for d, a in zip(loaded.data, amps)]
    rms = [float(np.sqrt(np.mean(r**2))) for r in res]
    return (
        res,
        rms,
        sum(float(np.sum(r**2)) for r in res),
        sum(int(d.n_independent) for d in loaded.data),
    )


def init_check(result, loaded, init, primary_diam, mjd):
    """How far the best orbit of the warm-started band moved from ``init`` (the reference orbit), and the chi2 gain
    over the init point (V2 amplitudes unscaled at both): a fit that never moved passes any comparison with the
    reference it started from."""
    from crosscheck import skyplane as SP

    mjd = np.asarray(mjd, float)
    b = result.best.best
    moved = SP.fit_pos(b, mjd, result.t_ref) - SP.fit_pos(init, mjd, result.t_ref)
    chi2 = {}
    for k, params in (("init", dict(init, **({"diam": b["diam"]} if "diam" in b else {}))), ("best", b)):
        model = best_scene(result, primary_diam, params)
        chi2[k] = sum(float(np.sum(epoch_residuals(model, d) ** 2)) for d in loaded.data)
    return dict(
        offset_rms_mas=SP._rms(moved),
        offset_max_mas=float(np.hypot(*moved.T).max()),
        chi2_init=chi2["init"],
        chi2_best=chi2["best"],
        chi2_gain=chi2["init"] - chi2["best"],
    )


def plot_residuals(path, loaded, res, rms, n_best):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(loaded.data) + 3), 4))
    for k, r in enumerate(res):
        ax.plot(
            k + 0.15 * rng.standard_normal(r.size), r, ".", ms=2, alpha=0.4, color="C0"
        )
    ax.plot(range(len(rms)), rms, "o", color="C3", label="rms (raw chi/N^0.5)")
    ax.plot(range(len(rms)), -np.array(rms), "o", color="C3")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(
        range(len(loaded.names)), loaded.names, rotation=60, ha="right", fontsize=7
    )
    ax.set_ylabel("residual / quoted error")
    ax.set_title(f"residuals / quoted error at the best orbit (N={n_best})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _flatten(v):
    """Scalars of a nested list (virgil's per-block scales were lists of lists before PR #321)."""
    for x in v:
        if isinstance(x, (list, tuple)):
            yield from _flatten(x)
        else:
            yield x


def print_table(result):
    print(
        f"{'N':>4} {'P (d)':>9} {'chi2/N':>7} {'s_min':>6} {'s_max':>6} {'logZ':>9} {'logZ_IS':>9} {'p':>7} {'ESS':>7}  flags"
    )
    for r in result.table():
        sc = [
            x for x in _flatten(r["scales"] or []) if x is not None and np.isfinite(x)
        ]
        sc = sc or [float("nan")]
        flags = r.get("flags")
        flags = ("FLAG" if r["flagged"] else "") if flags is None else ", ".join(flags)
        nan = float("nan")
        v = {
            k: nan if r[k] is None else r[k]
            for k in ("period", "chi2_red", "log_z", "log_z_is", "p", "ess")
        }
        print(
            f"{r['n']:>4d} {v['period']:>9.4f} {v['chi2_red']:>7.2f} {min(sc):>6.2f} {max(sc):>6.2f} "
            f"{v['log_z']:>9.2f} {v['log_z_is']:>9.2f} {v['p']:>7.3f} {v['ess']:>7.0f}  {flags}"
        )


def run(name, args):
    from virgil.aliases import fit_orbit_aliases

    sysd = SYSTEMS[name]
    root = os.environ.get("ESO_BIN_ROOT", ROOT_DEFAULT)
    outdir = pathlib.Path(args.out) / name
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    loaded = sysd.loader(root)
    p_range = (
        tuple(float(x) for x in args.p_range.split(","))
        if args.p_range
        else sysd.p_range
    )
    times = np.array([float(np.mean(np.asarray(d.mjd, float))) for d in loaded.data])
    t_ref = float(np.round(times.mean(), 1))
    print(
        f"{name}: {len(loaded.data)} epochs, MJD {times.min():.1f}-{times.max():.1f}, p_range {p_range}, "
        f"{sum(int(d.n_independent) for d in loaded.data)} independent observables"
    )
    for n in loaded.notes:
        print("note:", n)
    if args.dry_run:
        return
    if args.closure_only:  # the old fit: closure phases alone
        loaded.data = [d.select(observables="phi") for d in loaded.data]
        loaded.grid_data = [d.select(observables="phi") for d in loaded.grid_data]
    errors = []
    vis_amp = None if args.no_vis_amp or args.closure_only else VIS_AMP
    primary_diam = None if args.no_ud else sysd.primary_diam
    ref0 = loaded.reference

    def attempt(label, fn):
        """Run a post-processing step; a failure is recorded in meta.json and printed, never fatal."""
        try:
            return fn()
        except Exception as err:  # noqa: BLE001
            errors.append(f"{label}: {type(err).__name__}: {err}")
            print(f"WARNING: {errors[-1]}", file=sys.stderr)
            traceback.print_exc()
            return None

    meta = dict(
        system=name,
        label=sysd.label,
        instrument=sysd.instrument,
        epochs=loaded.names,
        mjd=times.tolist(),
        p_range=p_range,
        args={k: v for k, v in vars(args).items() if k != "list"},
        primary_diam=primary_diam,
        vis_amp=vis_amp,
        notes=loaded.notes,
        reference=loaded.reference,
        virgil=_virgil_version(),
        errors=errors,
    )

    def save_meta():
        meta["elapsed_s"] = time.time() - t0
        json.dump(meta, open(outdir / "meta.json", "w"), indent=1, default=str)

    save_meta()
    positions, epos, pinfo = positions_for(loaded, t_ref)
    meta["position_grid"] = pinfo
    if pinfo["edge_epochs"]:
        # Loud: a best position still at the grid edge may be a bad night or a companion beyond the grid.
        pinfo["edge_detail"] = {}
        for nm in pinfo["edge_epochs"]:
            i = loaded.names.index(nm)
            pinfo["edge_detail"][nm] = dict(
                dra=float(epos.dra[i]),
                ddec=float(epos.ddec[i]),
                chi2_raw=epos.chi2_raw[i] if getattr(epos, "chi2_raw", None) is not None else None,
            )
        msg = (
            f"position grid: best peak still at the edge after widening to {pinfo['half_widths'][-1]:.1f} mas "
            f"for {pinfo['edge_epochs']}: {pinfo['edge_detail']}"
        )
        errors.append(msg)
        print(f"WARNING: {msg}", file=sys.stderr)
    save_meta()
    # Each band is written to outdir/bands/ as soon as it is fitted; a rerun skips the finished ones, but only
    # if the settings, data and virgil version match: otherwise the old bands move aside and the run starts fresh.
    _check_resume(outdir, _bands_fingerprint(name, p_range, loaded, times, args))
    init = None
    meta["init"] = None
    if ref0 is not None and not args.no_init:
        # Warm start in the reference band, so "no valid mode" cannot happen there: the fit still decides.
        init = dict(
            period=ref0["P"],
            dt_peri=ref0["t_peri"] - t_ref,
            ecc=ref0["ecc"],
            inc=ref0["inc"],
            omega=ref0["omega"],
            Omega=ref0["Omega"],
            a_mas=ref0["a_mas"],
            flux=0.9 if sysd.sign_free else 0.3,
        )
        meta["init"] = dict(init, source="reference orbit")
        save_meta()
    result = fit_orbit_aliases(
        loaded.data,
        p_range,
        positions=positions,
        times=times,
        t_ref=t_ref,
        a_range=loaded.a_range,
        ecc_max=args.ecc_max,
        sign_free=sysd.sign_free,
        n_candidates=args.n_candidates,
        n_refine=args.n_refine,
        n_is=args.n_is,
        n_samples=args.n_samples,
        max_steps=args.max_steps,
        primary_diam=primary_diam,
        vis_amp=vis_amp,
        flip_flux=not args.no_flip,
        init=init,
        checkpoint_dir=outdir / "bands",
        on_band=lambda b: print(
            f"band N={b.n}: log_z {b.log_z:.2f} flags {b.flags}{' FLIPPED' if getattr(b, 'flipped', False) else ''} "
            f"({time.time() - t0:.0f} s)",
            flush=True,
        ),
    )
    result.to_json(
        outdir / "bands.json", n_samples=args.n_samples
    )  # results first: nothing below can lose them
    attempt("print_table", lambda: print_table(result))
    ref = loaded.reference
    if ref is not None:
        hit = next(
            (b for b in result.bands if b.p_lo <= ref["P"] <= b.p_hi), None
        )  # virgil's own band edges
        meta["reference_band"] = dict(
            n=None if hit is None else hit.n,
            period=ref["P"],
            p=None if hit is None else hit.p,
            is_winner=hit is not None and hit is result.best,
        )
        meta["elements"] = attempt(
            "element_comparison", lambda: element_comparison(result, ref, name)
        )
        print(
            f"reference (P={ref['P']:.4f} d) lies in band N={meta['reference_band']['n']}; winner N={result.best.n}"
        )
        # The reflection is accepted for a near-equal pair only when the fit reports its labels free. virgil's result
        # does not report it yet (virgil#324 follow-up, OrbitPipeline), so until it does this is the fit setting
        # the run was given; a result that carries ``labels_free`` wins.
        labels_free = bool(getattr(result, "labels_free", sysd.sign_free))
        meta["sky_plane"] = attempt("sky_plane", lambda: sky_plane(result, meta, ref, labels_free, init is not None))
        if init is not None and result.best.best:
            meta["init_check"] = attempt(
                "init_check", lambda: init_check(result, loaded, init, primary_diam, meta["mjd"])
            )
            ic = meta["init_check"]
            if ic:
                print(
                    f"warm start: best orbit is {ic['offset_rms_mas']:.3f} mas rms from the init (the reference), "
                    f"chi2 gain over init {ic['chi2_gain']:.1f}"
                )
        print(
            "sky plane vs reference: "
            + ", ".join(
                f"{k} band {v['band']}: rms {v['rms']:.2f} mas ({v['frac_sep']:.0%} of sep) {v['verdict']}"
                for k, v in (meta["sky_plane"] or {}).items()
            )
        )
    meta["fit_summary"] = attempt("fit_summary", lambda: fit_summary(result))
    save_meta()
    vr = result.best.chi2_red
    stats = (
        attempt("residual_stats", lambda: residual_stats(result, loaded, primary_diam))
        if result.best.best
        else None
    )
    blocks = attempt("block_chi2", lambda: block_chi2(result, loaded, primary_diam)) if result.best.best else None
    if blocks is not None:
        meta["chi2_raw_by_block"] = blocks
        print(
            "raw chi2/N by observable: "
            + ", ".join(f"{k} {v['chi2_per_n']:.2f} (N={v['n']})" for k, v in blocks.items())
        )
    if stats is not None:  # the chi2 bookkeeping check does not depend on any plot
        res, rms, chi2, n_ind = stats
        meta["rms_residual"] = rms
        meta["chi2_raw"], meta["n_independent"] = chi2, n_ind
        meta["chi2_raw_per_n"] = chi2 / n_ind
        meta["chi2_check"] = dict(
            ours=chi2 / n_ind,
            virgil=vr,
            agrees=bool(
                np.isfinite(vr) and vr > 0 and abs(chi2 / n_ind / vr - 1) < 0.05
            ),
        )
        if not meta["chi2_check"]["agrees"]:
            print(
                f"WARNING: raw chi2/N here {chi2 / n_ind:.3f} differs from the band table's {vr:.3f}"
            )
    else:  # absence must not read as a pass
        meta["chi2_check"] = dict(
            ours=None, virgil=vr, agrees=False, reason="not computed: see errors"
        )
        errors.append("chi2_check: not computed")
    if result.samples.get(result.best.n) is not None and result.best.best:
        attempt(
            "plot_sky", lambda: plot_sky(outdir / "sky.png", result, loaded, epos, ref)
        )
        attempt("plot_corner", lambda: plot_corner(outdir / "corner.png", result))
        if stats is not None:
            attempt(
                "plot_residuals",
                lambda: plot_residuals(
                    outdir / "residuals.png", loaded, res, rms, result.best.n
                ),
            )
    else:
        errors.append("no posterior samples in the best band: plots skipped")
    save_meta()
    for stale in (outdir / f"{name}.done", outdir / f"{name}.errors"):
        stale.unlink(missing_ok=True)
    if errors:  # .done means complete: a run with failed steps is marked apart, so batch tooling reruns or looks
        (outdir / f"{name}.errors").write_text("\n".join(errors) + "\n")
    else:
        (outdir / f"{name}.done").write_text("ok\n")
    print(f"wrote {outdir} in {time.time() - t0:.0f} s")


def sky_plane(result, meta, ref, labels_free=False, warm_started=False):
    """Sky-plane agreement with the reference (crosscheck.skyplane) for the winning band and the reference's
    band: rms and max offset at the epochs, the point-reflection (flux flip) used, and the verdict. A reflection
    is accepted only if the band's fit swapped the labels (band.flipped or fitted flux > 1) or the fit reported its
    labels free (``labels_free``). Each band's raw chi2/N goes to the score so that a good fit far from the reference
    reads REFERENCE-DISAGREES. ``warm_started``: the reference band's fit began at the reference orbit."""
    from crosscheck import skyplane as SP

    m = dict(mjd=meta["mjd"], reference=ref)
    bands = {"top": result.best}
    hit = next((b for b in result.bands if b.p_lo <= ref["P"] <= b.p_hi), None)
    if hit is not None:
        bands["reference"] = hit
    out = {}
    for k, b in bands.items():
        if not b.best:
            continue
        flipped = bool(getattr(b, "flipped", False)) or (b.best.get("flux") or 0.0) > 1.0
        sc = SP.score(m, b.best, result.t_ref, result.samples.get(b.n), flipped=flipped, labels_free=labels_free,
                      chi2_n=getattr(b, "chi2_red", None))
        warm = warm_started and b is hit
        out[k] = dict(sc, band=b.n, period=b.best["period"], p=b.p, flipped=flipped, warm_started=warm)
    return out


def fit_summary(result):
    """What the fit added beyond the orbit: flux ratio and whether the labels were swapped, the primary's diameter
    and each epoch's V2 amplitude, for the winning band."""
    b = result.best
    return dict(
        band=b.n,
        flux=b.best.get("flux"),
        flipped=bool(getattr(b, "flipped", False)),
        diam=b.best.get("diam"),
        vis_amp=[e.get("vis_amp") for e in b.epochs],
    )


SEEDING = "EpochPositions"  # what fit_orbit_aliases is seeded with; bump if the seeding changes


def _bands_fingerprint(name, p_range, loaded, times, args):
    """Everything that decides whether bands fitted earlier can be resumed, seeding included."""
    return dict(
        system=name,
        p_range=list(p_range),
        a_range=list(loaded.a_range),
        epochs=loaded.names,
        mjd=times.tolist(),
        virgil=_virgil_version(),
        seeding=SEEDING,
        args={k: v for k, v in vars(args).items() if k not in ("list", "out", "dry_run")},
    )


def _check_resume(outdir, fingerprint):
    """Keep outdir/bands for resuming only if it was fitted under this fingerprint; else move it aside."""
    fp = outdir / "bands_fingerprint.json"
    text = json.dumps(fingerprint, sort_keys=True, default=str)
    if (outdir / "bands").exists() and (not fp.exists() or fp.read_text() != text):
        aside = outdir / f"bands.stale-{int(time.time())}"
        (outdir / "bands").rename(aside)
        print(
            f"NOTE: earlier bands were fitted under other settings or code; moved to {aside.name}, starting fresh"
        )
    fp.write_text(text)


def _virgil_version():
    try:
        import importlib.metadata as im

        v = im.version("virgil-astro")
    except Exception:
        return None
    try:  # the commit a pip install from git recorded
        du = json.loads(
            im.distribution("virgil-astro").read_text("direct_url.json") or "{}"
        )
        commit = du.get("vcs_info", {}).get("commit_id")
    except Exception:
        commit = None
    return dict(version=v, commit_id=commit or os.environ.get("VIRGIL_COMMIT"))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="print the system names, one per line, and exit",
    )
    ap.add_argument("--system", help="system name (see --list)")
    ap.add_argument("--out", default="orbit_aliases_out")
    ap.add_argument("--n-is", type=int, default=2000, dest="n_is")
    ap.add_argument("--n-candidates", type=int, default=100, dest="n_candidates")
    ap.add_argument("--n-refine", type=int, default=6, dest="n_refine")
    ap.add_argument("--n-samples", type=int, default=1000, dest="n_samples")
    ap.add_argument(
        "--max-steps",
        type=int,
        default=200,
        dest="max_steps",
        help="L-BFGS-B iterations per start",
    )
    ap.add_argument(
        "--closure-only",
        action="store_true",
        dest="closure_only",
        help="fit closure phases only (default: V2 and closure phases, per-epoch error scales for each)",
    )
    ap.add_argument("--no-vis-amp", action="store_true", dest="no_vis_amp", help="no per-epoch V2 amplitude")
    ap.add_argument("--no-ud", action="store_true", dest="no_ud", help="point primary (default: the registry's diameter)")
    ap.add_argument("--no-flip", action="store_true", dest="no_flip", help="no automatic flux-ratio flip")
    ap.add_argument("--no-init", action="store_true", dest="no_init", help="no reference-orbit warm start")
    ap.add_argument("--ecc-max", type=float, default=0.9, dest="ecc_max")
    ap.add_argument(
        "--p-range",
        default=None,
        dest="p_range",
        help="LO,HI days (default: the registry's)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="load the data and print its shape, no fit",
    )
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
