"""Track B: virgil per-epoch binary fits compared with published positions.

Implements design/trackb_criteria.md (registered at commit CRITERIA_COMMIT; its
SHA-256 is checked at start-up and written into every result). One system per
invocation, all its epochs:

  1. per epoch, the Phase 3 files named by its phase3_dp_ids: GRAVITY closure
     phases only (GRAVITY_SC, 2.05-2.38 um without 2.155-2.175 um), PIONIER V2
     and closure phases;
  2. likelihood_grid(model, data, grid) of BinaryModelCartesian over
     +-max(3 sep_published, 30) mas at lambda_min / (4 B_max), then fit() of the
     full model (fixed uniform disks where the paper gives diameters) from the 5
     best separate peaks, with Jeffreys priors and free error scales;
  3. raw chi2/N on the quoted errors, the Laplace covariance (laplace_cov) on
     the scaled errors, and d2 against the published 1-sigma ellipse
     (crosscheck.astrometry, which does not import virgil);
  4. a diagnostic refit with the position fixed at the published one.

Writes <out>/<system>.json, then <out>/<system>.done.

  python scripts/trackb_fit.py <system> --out <dir> [--root <trackb dir>] [--dry-run]

--dry-run loads every epoch's data and prints shapes and the search set-up,
with no grid or fit (safe on a laptop).
"""

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
import traceback

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from crosscheck import astrometry as A  # noqa: E402

CRITERIA_FILE = "design/trackb_criteria.md"
CRITERIA_COMMIT = "1d2b3e6eca2002bb57310dcc27995730adb894ae"
CRITERIA_SHA256 = "d763e4c19b85f4f38fe0d917adea0de0bc636c72e09e018cf74d1c46c18c9f0e"

MAS_PER_RAD = 180.0 / np.pi * 3600e3
GRAVITY_RANGES = [(2.05e-6, 2.155e-6), (2.175e-6, 2.38e-6)]
GRID_MAX_CHANNELS = 24
N_PEAKS = 5
N_FLUX_GRID = 13
FLUX_MIN = 1e-3
SCALE_PRIOR = (0.1, 10.0)
# o Leo: the primary's mean UD diameter of Gallenne+2023 (fitted per epoch there), and the secondary's.
DIAMETER_OVERRIDE = {"omi_leo": (1.285, 0.49)}
SENSITIVITY_FREE_PRIMARY = ("omi_leo",)


def criteria_block():
    path = REPO / CRITERIA_FILE
    sha = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    if sha is not None and sha != CRITERIA_SHA256:
        raise SystemExit(f"{CRITERIA_FILE} has changed since registration (sha256 {sha}): a new campaign version")
    return dict(file=CRITERIA_FILE, commit=CRITERIA_COMMIT, sha256=CRITERIA_SHA256, checked=sha is not None)


def diameters(system, s):
    if system in DIAMETER_OVERRIDE:
        return DIAMETER_OVERRIDE[system]
    d = s.get("diameters_mas") or {}

    def one(x):
        if isinstance(x, (list, tuple)) and x and isinstance(x[0], (int, float)):
            return float(x[0])
        return None
    return one(d.get("primary")), one(d.get("secondary"))


def instrument(path):
    from astropy.io import fits
    with fits.open(path) as h:
        return h[0].header.get("INSTRUME", "").strip().upper()


def mjd_of(path):
    from astropy.io import fits
    with fits.open(path) as h:
        for ext in h:
            if ext.name == "OI_VIS2":
                return float(np.mean(ext.data["MJD"]))
    return float("nan")


def epochs_of(system, s, root):
    """Published epochs with Phase 3 files, then any unreferenced files grouped into nights (reported only)."""
    sysdir = root / system
    out, used = [], set()
    paper = s["reference"]["bibcode"]
    for i, rec in enumerate(s["reference_positions"]):
        ids = rec.get("phase3_dp_ids") or []
        if not ids:
            continue
        files = [sysdir / "oifits" / (d.replace(":", "_") + ".fits") for d in ids]
        used |= {f.name for f in files}
        pub = A.published_for_system(system, rec, paper)
        reason = None
        if rec.get("below_lambda_2B"):
            reason = "below lambda/2B"
        elif rec.get("flag"):
            reason = rec["flag"]
        out.append(dict(index=i, date=rec["date"], mjd=rec.get("mjd"), files=files, rec=rec, published=pub,
                        counted=reason is None, reason=reason))
    rest = sorted(p for p in (sysdir / "oifits").glob("*.fits") if p.name not in used)
    nights = {}
    for p in rest:  # one group per night (MJD integer boundary at UT 12h) and instrument
        nights.setdefault((int(np.floor(mjd_of(p) - 0.5)), instrument(p)), []).append(p)
    for (n, inst), files in sorted(nights.items()):
        out.append(dict(index=None, date=f"night MJD {n} {inst}", mjd=None, files=files, rec=None, published=None,
                        counted=False, reason="no published epoch with Phase 3 data (fit reported only)"))
    return out


def _read(files, inst):
    from virgil.oidata import OIData
    from virgil.oifits import read_oifits
    if inst == "GRAVITY":
        return OIData(read_oifits([str(f) for f in files], insname="GRAVITY_SC")).select(
            ranges=GRAVITY_RANGES, observables="phi")
    if inst == "PIONIER":
        return OIData([str(f) for f in files])
    raise ValueError(f"unknown instrument {inst}")


def load(files):
    """OIData of one epoch per design/trackb_criteria.md, the instrument, and the files left out.

    A file virgil cannot read is left out of its epoch and listed with the reason (criteria,
    "Data per instrument"); e.g. a PIONIER product whose closure triangle names a baseline
    with no V2 row (AL Dor 2017-11-24)."""
    inst = {instrument(f) for f in files}
    if len(inst) != 1:
        raise ValueError(f"mixed instruments in one epoch: {inst}")
    inst = inst.pop()
    try:
        return _read(files, inst), inst, []
    except ValueError:
        if len(files) == 1:
            raise
    good, dropped = [], []
    for f in files:
        try:
            _read([f], inst)
            good.append(f)
        except ValueError as e:
            dropped.append(dict(file=f.name, error=str(e)[:300]))
    if not good:
        raise ValueError(f"no readable file: {dropped}")
    return _read(good, inst), inst, dropped


def geometry(data):
    u, v = np.asarray(data.u, float), np.asarray(data.v, float)
    wl = np.asarray(data.wavel, float)
    bmax = float(np.max(np.hypot(u, v)))
    return bmax, float(wl.min()), float(wl.max()), np.unique(wl)


def grid_data(data, inst, channels):
    """The grid uses at most GRID_MAX_CHANNELS of GRAVITY's channels (every k-th)."""
    if inst != "GRAVITY" or channels.size <= GRID_MAX_CHANNELS:
        return data
    k = int(np.ceil(channels.size / GRID_MAX_CHANNELS))
    keep = channels[::k]
    eps = 0.25 * float(np.min(np.diff(channels)))
    return data.select(ranges=[(w - eps, w + eps) for w in keep])


def build(diam, dra, ddec, flux):
    """Template model and the paths of (dra, ddec, flux)."""
    from virgil.models import BinaryModelCartesian, PointSource, System, UniformDisk
    d1, d2 = diam
    if d1 is None and d2 is None:
        return BinaryModelCartesian(dra=dra, ddec=ddec, flux=flux), ["dra", "ddec", "flux"]
    star = UniformDisk(diam=d1) if d1 is not None else PointSource()
    comp = (UniformDisk(diam=d2, flux=flux, dra=dra, ddec=ddec) if d2 is not None
            else PointSource(flux=flux, dra=dra, ddec=ddec))
    return System(star=star, comp=comp), ["comp.dra", "comp.ddec", "comp.flux"]


def peaks(ll, axes, n, min_sep):
    m, fb = np.nanmax(ll, axis=2), np.nanargmax(ll, axis=2)
    out = []
    for k in np.argsort(m, axis=None)[::-1]:
        i, j = np.unravel_index(k, m.shape)
        x, y = float(axes[0][i]), float(axes[1][j])
        if all(np.hypot(x - p["dra"], y - p["ddec"]) >= min_sep for p in out):
            out.append(dict(dra=x, ddec=y, flux=float(axes[2][fb[i, j]]), ll=float(m[i, j])))
        if len(out) == n:
            break
    return out


def f(x):
    return float(np.asarray(x))


def fit_epoch(system, ep, diam, fmax, dry_run):
    import numpyro.distributions as dist
    t0 = time.time()
    data, inst, dropped = load(ep["files"])
    bmax, wmin, wmax, channels = geometry(data)
    pub = ep["published"]
    sep_pub = float(np.hypot(*pub["pos"])) if pub else 10.0
    half = max(3.0 * sep_pub, 30.0)
    step = wmin / (4 * bmax) * MAS_PER_RAD
    min_sep = wmin / bmax * MAS_PER_RAD
    n_axis = 2 * int(np.ceil(half / step)) + 1
    info = dict(instrument=inst, files=[p.name for p in ep["files"]], dropped_files=dropped, n_vis=int(np.size(data.vis)),
                n_phi=int(np.size(data.phi)), n_independent=int(data.n_independent), n_channels=int(channels.size),
                wavel_um=[wmin * 1e6, wmax * 1e6], b_max_m=bmax, grid_half_mas=half, grid_step_mas=step,
                grid_shape=[n_axis, n_axis, N_FLUX_GRID],
                mjd_data=float(np.nanmean(np.asarray(data.mjd))) if data.mjd is not None else None)
    if dry_run:
        return info

    from virgil.fitting import fit
    from virgil.grid_fit import likelihood_grid
    from virgil.inference import laplace_cov
    from virgil.likelihood import whitened_residuals
    from virgil.models import BinaryModelCartesian

    axes = [np.linspace(-half, half, n_axis), np.linspace(-half, half, n_axis), np.geomspace(FLUX_MIN, fmax, N_FLUX_GRID)]
    gdata = grid_data(data, inst, channels)
    ll = np.asarray(likelihood_grid(BinaryModelCartesian, gdata, {"dra": axes[0], "ddec": axes[1], "flux": axes[2]}))
    info["grid_seconds"] = time.time() - t0
    noise = {"phi_scale": dist.LogUniform(*SCALE_PRIOR)}
    if inst == "PIONIER":
        noise["vis_scale"] = dist.LogUniform(*SCALE_PRIOR)
    _, paths = build(diam, 0.0, 0.0, 0.5)
    priors = {paths[0]: dist.Uniform(-half, half), paths[1]: dist.Uniform(-half, half),
              paths[2]: dist.LogUniform(FLUX_MIN, fmax)}

    def summarise(res):
        v = {p: f(res.values[p]) for p in paths}
        sc = {k: f(res.values[f"noise.{k}"]) for k in noise if f"noise.{k}" in res.values}
        return dict(dra=v[paths[0]], ddec=v[paths[1]], flux=v[paths[2]], loss=f(res.info["loss"]),
                    chi2_red_scaled=f(res.info["chi2_red"]),
                    converged=None if res.info.get("converged") is None else bool(np.asarray(res.info["converged"])),
                    scales=sc)

    fits = []
    for g in peaks(ll, axes, N_PEAKS, min_sep):
        start = dict(dra=float(np.clip(g["dra"], -0.98 * half, 0.98 * half)),
                     ddec=float(np.clip(g["ddec"], -0.98 * half, 0.98 * half)),
                     flux=float(np.clip(g["flux"], 1.2 * FLUX_MIN, 0.95 * fmax)))
        tmpl, _ = build(diam, **start)
        res = fit(tmpl, priors, data, noise=noise)
        fits.append((f(res.info["loss"]), dict(summarise(res), start=start, grid_ll=g["ll"]), res))
    fits.sort(key=lambda x: x[0])
    _, best, res = fits[0]

    # raw chi2 on the quoted errors at the best fit
    r = np.asarray(whitened_residuals(res.model, data))
    chi2_raw = float(np.sum(r**2))
    scales = best["scales"]
    scale_dict = {"phi": scales.get("phi_scale", 1.0)}
    if "vis_scale" in scales:
        scale_dict["vis"] = scales["vis_scale"]
    x = np.array([best["dra"], best["ddec"], best["flux"]])
    cov = np.asarray(laplace_cov(x, paths, res.model, data.with_error_scale(scale_dict)))
    cov_raw = np.asarray(laplace_cov(x, paths, res.model, data))
    sep, pa = A.offsets_to_sep_pa(best["dra"], best["ddec"])
    out = dict(info, chi2_raw=chi2_raw, chi2_raw_per_n=chi2_raw / info["n_independent"], scales=scales,
               scale_failure=any(s > A.SCALE_FAILURE for s in scales.values()),
               dra=best["dra"], ddec=best["ddec"], sep=sep, pa=pa, flux=best["flux"],
               loss=best["loss"], converged=best["converged"], cov_full=cov.tolist(), cov_full_unscaled=cov_raw.tolist(),
               cov_virgil=cov[:2, :2].tolist(), cov_virgil_unscaled=cov_raw[:2, :2].tolist(),
               peaks=[dict(p, loss=L) for L, p, _ in fits])

    if pub:
        fold = system in A.NEAR_EQUAL
        d2, sign, delta = A.d2_compare([best["dra"], best["ddec"]], cov[:2, :2], pub["pos"], pub["cov"], fold180=fold)
        out.update(published=pub["pos"], cov_published=np.asarray(pub["cov"]).tolist(), convention=pub["convention"],
                   fold180=fold, sign=sign, delta=delta, d2=d2, d2_flag=d2 > A.D2_FLAG)
        if pub["cov_alt"] is not None:
            out.update(d2_alt=A.d2_compare([best["dra"], best["ddec"]], cov[:2, :2], pub["pos"], pub["cov_alt"],
                                           fold180=fold)[0], d2_alt_note=pub["alt_note"])
        # diagnostic: the position fixed at the published one (nearer sign), flux and scales refitted
        try:
            rx, ry = sign * np.asarray(pub["pos"])
            tmpl, _ = build(diam, float(rx), float(ry), float(np.clip(best["flux"], 1.2 * FLUX_MIN, 0.95 * fmax)))
            rr = fit(tmpl, {paths[2]: priors[paths[2]]}, data, noise=noise)
            out["at_reference"] = dict(loss=f(rr.info["loss"]), two_dloss=2 * (f(rr.info["loss"]) - best["loss"]),
                                       flux=f(rr.values[paths[2]]),
                                       scales={k: f(rr.values[f"noise.{k}"]) for k in noise})
        except Exception as e:  # a diagnostic: never lose the epoch's fit
            out["at_reference"] = dict(error=f"{type(e).__name__}: {e}")

    if system in SENSITIVITY_FREE_PRIMARY:
        try:
            tmpl, _ = build(diam, best["dra"], best["ddec"], best["flux"])
            p2 = dict(priors, **{"star.diam": dist.LogUniform(0.1, 10.0)})
            rs = fit(tmpl, p2, data, noise=noise)
            sv = dict(summarise(rs), star_diam=f(rs.values["star.diam"]))
            if pub:
                cs = np.asarray(laplace_cov(np.array([sv["dra"], sv["ddec"], sv["flux"], sv["star_diam"]]),
                                            paths + ["star.diam"], rs.model, data.with_error_scale(scale_dict)))
                sv["d2"] = A.d2_compare([sv["dra"], sv["ddec"]], cs[:2, :2], pub["pos"], pub["cov"])[0]
            out["sensitivity_free_primary"] = sv
        except Exception as e:
            out["sensitivity_free_primary"] = dict(error=f"{type(e).__name__}: {e}")
    out["seconds"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("system")
    ap.add_argument("--root", default=os.environ.get("TRACKB_ROOT", str(pathlib.Path.home() / "data/eso_binaries/trackb")))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root, out = pathlib.Path(a.root), pathlib.Path(a.out)
    s = json.loads((root / a.system / "system.json").read_text())
    crit = criteria_block()
    if not a.dry_run:
        import jax
        jax.config.update("jax_enable_x64", True)
    diam = diameters(a.system, s)
    fmax = 2.0 if a.system in A.NEAR_EQUAL else 1.0
    eps = epochs_of(a.system, s, root)
    print(f"{a.system}: {len(eps)} epochs ({sum(e['counted'] for e in eps)} counted); diameters {diam}; "
          f"flux prior LogUniform({FLUX_MIN}, {fmax}); criteria {crit['commit'][:12]} (sha256 checked: {crit['checked']})",
          flush=True)
    rows = []
    for ep in eps:
        row = dict(date=ep["date"], mjd=ep["mjd"], counted=ep["counted"], reason=ep["reason"],
                   published_record=ep["rec"])
        try:
            row.update(fit_epoch(a.system, ep, diam, fmax, a.dry_run))
        except Exception as e:  # report and go on; a counted epoch without a fit is listed as such
            row.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
            print(f"  {ep['date']}: FAILED {row['error']}", flush=True)
            rows.append(row)
            continue
        if a.dry_run:
            pub = ep["published"]
            print(f"  {ep['date']:<16} {row['instrument']:<8} {len(row['files'])} file(s)  vis {row['n_vis']:6d}  "
                  f"phi {row['n_phi']:6d}  n_ind {row['n_independent']:6d}  ch {row['n_channels']:5d}  "
                  f"{row['wavel_um'][0]:.3f}-{row['wavel_um'][1]:.3f} um  Bmax {row['b_max_m']:6.1f} m  "
                  f"grid +-{row['grid_half_mas']:.1f} mas @ {row['grid_step_mas']:.3f} -> {row['grid_shape']}  "
                  f"pub {np.round(pub['pos'], 3).tolist() if pub else None} ({pub['convention'] if pub else '-'})  "
                  f"{'counted' if ep['counted'] else 'not counted: ' + str(ep['reason'])}"
                  f"{'  DROPPED ' + str([d['file'] for d in row['dropped_files']]) if row['dropped_files'] else ''}", flush=True)
        else:
            print(f"  {ep['date']}: chi2_raw/N {row['chi2_raw_per_n']:.2f}  scales {row['scales']}  "
                  f"({row['dra']:+.3f}, {row['ddec']:+.3f}) mas  flux {row['flux']:.4g}  d2 {row.get('d2', float('nan')):.2f}  "
                  f"{'counted' if ep['counted'] else 'not counted'}  [{row['seconds']:.0f} s]", flush=True)
        rows.append(row)
    if a.dry_run:
        return
    import virgil
    counted = [r["d2"] for r in rows if r["counted"] and "d2" in r]
    result = dict(system=a.system, target=s.get("target"), reference=s["reference"]["bibcode"], criteria=crit,
                  virgil=dict(version=getattr(virgil, "__version__", None), commit=os.environ.get("VIRGIL_COMMIT")),
                  validation_commit=os.environ.get("VALIDATION_COMMIT"), diameters_mas=diam, flux_max=fmax,
                  epochs=rows, system_stat=A.system_statistic(counted),
                  n_counted_without_fit=sum(1 for r in rows if r["counted"] and "d2" not in r))
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"{a.system}.json.part"
    tmp.write_text(json.dumps(result, indent=1, default=float))
    os.replace(tmp, out / f"{a.system}.json")
    (out / f"{a.system}.done").write_text(time.strftime("%Y-%m-%dT%H:%M:%S\n"))
    print(f"{a.system}: system {result['system_stat']} -> {out / (a.system + '.json')}", flush=True)


if __name__ == "__main__":
    main()
