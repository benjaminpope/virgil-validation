"""Blind orbit fits for the ESO (Track B) targets of the orbit gallery: a thin caller of virgil's ``OrbitPipeline``.

    python scripts/gallery_orbits.py --list
    python scripts/gallery_orbits.py --system hd_9312 --out DIR [--dry-run] [--include-extra] [--quick]
    python scripts/gallery_orbits.py --system iota_peg --out DIR [--gains TEL,BASE,OFFSET_DEG]   (MIRC-X, from $OIDB_ROOT)

One system per call (the OzSTAR array job `gallery_orbits` runs one per task). The orbit work is all virgil's:
``OrbitPipeline`` runs the per-night position search with automatic night flags, the alias-band scan over a period
window computed from the data (mode keeping, tempered evidences, label swap, flux flip, per-night V^2 amplitudes,
automatic UD primary), the mode table with the ``no-orbit`` comparison, and NUTS per significant mode. This file
holds only what the pipeline expects its caller to do (design/orbit_pipeline.md in virgil):

- the data preparation (``calibrated_load``, virgil-vlti's ``to_oidata``): GRAVITY from the calibrated files under
  ``$ESO_BIN_ROOT/trackb/<slug>/calibrated/`` on the Track B continuum channels (Brγ excluded), V^2 with gains and
  closure phases with offsets, or closure phases only on a night with no calibrator; PIONIER (already calibrated)
  with pair-test gains. The uncalibrated SCI_VIS V^2 is refused. Nothing from the published solution is read here;
- the run folder ``DIR/<slug>/`` and ``DIR/<slug>/gallery.json``, the provenance (file hashes, pins) that
  ``gallery_pull_back.py`` turns into a gallery tile (``data_path`` says how the data were reduced) together with the published orbit of
  ``oidb/trackb/<slug>.json``, which is used only there, after the fit, in the sky plane.

No period, window, eccentricity cap, diameter or starting orbit is passed: every setting is the pipeline's default.
``--quick`` shrinks the search effort and the NUTS run for a smoke test.
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

RECORDS = REPO / "oidb" / "trackb"
OIDB_ROOT_DEFAULT = "/fred/oz440/bpope/oidb"
# CHARA/MIRC-X systems read from an OiDB collection (the authors' calibrated viscal files, one night per UT date in the
# file name); no trackb record. Gains are on log|V| per frame, closure offsets in degrees: MIRC-X has no measured
# virgil-vlti template, so these start values are placeholders and the per-night chi2/N says whether they suffice.
MIRCX = {"iota_peg": dict(collection="fac164e1-d9d0-4500-8164-e1d9d0450099", pattern=r"mircx\d+_oifits_viscal\.fits$",
                          instrument="MIRC-X", ud_note="primary UD fitted by the pipeline")}
MIRCX_GAINS = dict(telescope=0.02, baseline=0.02, offset_deg=0.2)
ROOT_DEFAULT = "/fred/oz440/bpope/eso_binaries"
MIN_NIGHTS = 3  # fewer nights cannot say anything about an orbit
QUICK = dict(n_candidates=10, n_refine=2, max_steps=50, n_is=200, num_warmup=100, num_samples=100, num_chains=2)


GRID_BATCH_VISIBILITIES = 4096  # model visibilities per grid batch with gains (virgil default 2**20 on a CPU)


def limit_grid_batches(n_vis=GRID_BATCH_VISIBILITIES):
    """Stop-gap: grid batches that fit in memory for data with gains.

    virgil sizes a batch of grid points as ``batch_visibilities() // n_vis`` (at least 256), counting one model
    visibility per point. With gains each point also whitens an N x N covariance at 32 scale nodes, so the default
    (about 1200 points for a 900-sample night) needs hundreds of GB: HD 9312 night MJD 59191 on a 16 GB laptop was killed.
    ``OrbitPipeline`` has no ``batch_size`` setting, so this sets virgil's private defaults (4 points at N = 900,
    never fewer than 2). To go once virgil sizes batches by the gains (design/orbit_noise_models.md 3.2). Returns
    what was set, for the provenance."""
    from virgil import _grid

    _grid.CPU_BATCH_VISIBILITIES = _grid.ACCELERATOR_BATCH_VISIBILITIES = int(n_vis)
    _grid.MIN_BATCH_SIZE = 2
    return dict(batch_visibilities=int(n_vis), min_batch_size=2)


def slugs():
    return sorted(p.stem for p in RECORDS.glob("*.json"))


def record_of(slug):
    return json.loads((RECORDS / f"{slug}.json").read_text())


def group_nights(items):
    """[(night, [path, ...])] from (path, mjd) pairs: one night per UT-noon-to-noon day, as trackb_fit.epochs_of."""
    nights = {}
    for path, mjd in items:
        nights.setdefault(int(np.floor(mjd - 0.5)), []).append(path)
    return sorted(nights.items())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_nights(slug, root, include_extra=False):
    """``({night label: OIData}, provenance)``: the prepared data of one system, blind to its published solution.

    GRAVITY nights come from the calibrated files (V² with gains, or closure phases only where the night has no
    calibrator), PIONIER from the already calibrated files with pair-test gains: all through ``to_oidata``
    (``calibrated_load``). ``provenance["data_path"]`` is ``calibrated``, ``cp-only``, ``pair-test gains`` or
    ``mixed``, and ``provenance["nights"][label]`` has each night's path, mode and files."""
    import calibrated_load as C
    import trackb_fit as T

    inst = record_of(slug)["instrument"].upper()
    sysdir = pathlib.Path(root) / "trackb" / slug
    notes, nights, used = [], {}, {}
    if inst == "GRAVITY":
        if include_extra:
            raise ValueError("oifits_extra/ nights are not calibrated; GRAVITY systems read only <system>/calibrated/")
        found = C.gravity_nights(sysdir)
        multi = len({n["mode"] for n in found}) > 1
        for n in found:
            label = C.night_label(n["mjd"], n["mode"] if multi else None)
            try:
                nights[label] = C.load_gravity_night(n, template=record_of(slug).get("template"))
            except ValueError as err:  # a night with no readable file is left out; MIN_NIGHTS decides
                notes += [f"{label}: {x}" for x in n["notes"]]
                notes.append(f"{label}: left out: {err}")
                continue
            notes += [f"{label}: {x}" for x in n["notes"]]  # after the load: it may leave a file out
            used[label] = dict(data_path=n["path"], mode=n["mode"], files=[dict(file=f.name, sha256=sha256(f)) for f in n["files"]])
    elif inst == "PIONIER":
        dirs = ["oifits"] + (["oifits_extra"] if include_extra else [])
        files = sorted(f for d in dirs for f in (sysdir / d).glob("*.fits"))
        if not files:
            raise FileNotFoundError(f"no OIFITS under {sysdir}/{dirs}")
        items = []
        for f in files:
            if T.instrument(f) != inst:
                notes.append(f"{f.name}: {T.instrument(f)}, not the record's {inst}; left out")
            elif np.isfinite(T.mjd_of(f)):
                items.append((f, T.mjd_of(f)))
            else:
                notes.append(f"{f.name}: no OI_VIS2 time stamp; left out")
        for night, fs in group_nights(items):
            label = C.night_label(night)
            try:
                nights[label], dropped = C.load_pionier_night_robust(fs)
            except ValueError as err:
                notes.append(f"{label}: left out: {err}")
                continue
            notes += [f"{label}: dropped {x['file']}: {x['error']}" for x in dropped]
            left = {x["file"] for x in dropped}
            used[label] = dict(data_path=C.PAIR_GAINS, mode=None,
                               files=[dict(file=f.name, sha256=sha256(f)) for f in fs if f.name not in left])
    else:
        raise ValueError(f"{slug}: no calibrated loader for {inst}")
    if len(nights) < MIN_NIGHTS:
        raise ValueError(f"{slug}: {len(nights)} nights of {inst} data; at least {MIN_NIGHTS} are needed")
    return nights, dict(instrument=inst, include_extra=include_extra, notes=notes, nights=used,
                        data_path=C.system_data_path(v["data_path"] for v in used.values()))


def load_mircx_nights(slug, oidb_root, gains=None):
    """``({night: OIData}, provenance)`` of a MIRC-X system: each UT night's viscal files read together (a closure
    triangle whose baseline has no V^2 row is placed from the T3 table's own u, v), V^2 with per-frame gains on log|V|
    and closure phases with per-frame offsets. ``gains`` = dict(telescope, baseline, offset_deg), ``MIRCX_GAINS`` by default."""
    import oidb_fit as O

    cfg, w = MIRCX[slug], dict(MIRCX_GAINS, **(gains or {}))
    data_dir = pathlib.Path(oidb_root) / cfg["collection"]
    nights_files = O._group_by(O._files(str(data_dir), cfg["pattern"]), lambda f: (re.match(r"\d{4}(?:[A-Za-z]{3}|-\d{2}-)\d{2}", f) or re.match(".{10}", f)).group())
    if not nights_files:
        raise FileNotFoundError(f"no files matching {cfg['pattern']} under {data_dir}")
    nights, used, notes = {}, {}, []
    for night, paths in nights_files.items():
        try:
            d = O.load(paths)
        except ValueError as err:
            notes.append(f"{night}: left out: {err}")
            continue
        if np.asarray(d.vis).size:
            d = d.with_gains(telescope=w["telescope"], baseline=w["baseline"])
        else:
            notes.append(f"{night}: no V^2 rows; closure phases only")
        try:
            d = d.with_closure_offsets(baseline=np.radians(w["offset_deg"]))
        except ValueError as err:  # fewer than four telescopes
            notes.append(f"{night}: no closure offsets: {err}")
        nights[night] = d
        used[night] = dict(data_path="viscal+gains", files=[dict(file=os.path.basename(p), sha256=sha256(p)) for p in paths])
    if len(nights) < MIN_NIGHTS:
        raise ValueError(f"{slug}: {len(nights)} nights; at least {MIN_NIGHTS} are needed")
    return nights, dict(instrument=cfg["instrument"], collection=cfg["collection"], include_extra=False, notes=notes + [str(x) for x in O.LOAD_NOTES], nights=used, data_path="viscal+gains", gain_widths=w,
                        gain_widths_note="placeholders: no measured MIRC-X template (virgil-vlti has none)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the system slugs and exit")
    ap.add_argument("--system", choices=slugs() + sorted(MIRCX), help="a Track B system, or a MIRC-X system (read from OiDB)")
    ap.add_argument("--out", default="gallery_orbits_out")
    ap.add_argument("--include-extra", action="store_true", dest="include_extra", help="also use oifits_extra/ (nights the paper has no epoch for)")
    ap.add_argument("--quick", action="store_true", help="smoke settings: tiny search and NUTS")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run", help="load and group the data, no fit")
    ap.add_argument("--grid-batch-vis", type=int, default=GRID_BATCH_VISIBILITIES, dest="grid_batch_vis",
                    help="model visibilities per grid batch (memory; a GPU can take more than the laptop default)")
    ap.add_argument("--gains", metavar="TEL,BASE,OFFSET_DEG", help="MIRC-X systems: gain widths on log|V| and closure offset in degrees (default %s)" % ",".join(str(v) for v in MIRCX_GAINS.values()))
    ap.add_argument("--no-quicklook-execute", action="store_true", dest="no_qe", help="write the quicklook notebook without running it")
    a = ap.parse_args(argv)
    if a.list:
        print("\n".join(slugs()))
        return 0
    if not a.system:
        ap.error("--system is required")
    root = os.environ.get("ESO_BIN_ROOT", ROOT_DEFAULT)
    t0 = time.time()
    if a.system in MIRCX:
        g = dict(zip(("telescope", "baseline", "offset_deg"), (float(x) for x in a.gains.split(",")))) if a.gains else None
        nights, prov = load_mircx_nights(a.system, os.environ.get("OIDB_ROOT", OIDB_ROOT_DEFAULT), g)
    else:
        nights, prov = load_nights(a.system, root, a.include_extra)
    span = [float(np.mean(np.asarray(d.mjd, float))) for d in nights.values()]
    print(f"{a.system}: {len(nights)} nights, MJD {min(span):.1f}-{max(span):.1f}, {prov['instrument']}, data path {prov['data_path']}")
    for n in prov["notes"]:
        print("note:", n)
    if a.dry_run:
        return 0
    from virgil.pipeline import OrbitPipeline

    prov["grid_batches"] = limit_grid_batches(a.grid_batch_vis)
    settings = dict(QUICK) if a.quick else {}
    if a.no_qe:
        settings["quicklook_execute"] = False
    out = pathlib.Path(a.out) / a.system
    out.mkdir(parents=True, exist_ok=True)
    import virgil

    meta = dict(slug=a.system, virgil=getattr(virgil, "__version__", None), virgil_commit=os.environ.get("VIRGIL_COMMIT"),
                validation_commit=os.environ.get("VALIDATION_COMMIT"),
                virgil_vlti_commit=os.environ.get("VIRGIL_VLTI_COMMIT"), settings=settings, **prov)
    (out / "gallery.json").write_text(json.dumps(meta, indent=1) + "\n")
    result = OrbitPipeline(nights, output=out / "run", **settings).run()
    meta["elapsed_s"] = time.time() - t0
    (out / "gallery.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {out} in {meta['elapsed_s']:.0f} s")
    return 0 if result is not None else 1


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.exit(main())
