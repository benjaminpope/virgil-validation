"""Blind orbit fits for the ESO (Track B) targets of the orbit gallery: a thin caller of virgil's ``OrbitPipeline``.

    python scripts/gallery_orbits.py --list
    python scripts/gallery_orbits.py --system hd_9312 --out DIR [--dry-run] [--include-extra] [--quick]

One system per call (the OzSTAR array job `gallery_orbits` runs one per task). The orbit work is all virgil's:
``OrbitPipeline`` runs the per-night position search with automatic night flags, the alias-band scan over a period
window computed from the data (mode keeping, tempered evidences, label swap, flux flip, per-night V^2 amplitudes,
automatic UD primary), the mode table with the ``no-orbit`` comparison, and NUTS per significant mode. This file
holds only what the pipeline expects its caller to do (design/orbit_pipeline.md in virgil):

- the data preparation: every OIFITS file under ``$ESO_BIN_ROOT/trackb/<slug>/oifits/`` of the record's instrument,
  GRAVITY on its continuum channels with V^2 and closure phases (``trackb_fit.GRAVITY_RANGES``), grouped into nights
  by the files' own times. Nothing from the published solution is read here;
- the run folder ``DIR/<slug>/`` and ``DIR/<slug>/gallery.json``, the provenance (file hashes, pins) that
  ``gallery_pull_back.py`` turns into a gallery tile together with the published orbit of
  ``oidb/trackb/<slug>.json``, which is used only there, after the fit, in the sky plane.

No period, window, eccentricity cap, diameter or starting orbit is passed: every setting is the pipeline's default.
``--quick`` shrinks the search effort and the NUTS run for a smoke test.
"""

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

RECORDS = REPO / "oidb" / "trackb"
ROOT_DEFAULT = "/fred/oz440/bpope/eso_binaries"
MIN_NIGHTS = 3  # fewer nights cannot say anything about an orbit
QUICK = dict(n_candidates=10, n_refine=2, max_steps=50, n_is=200, num_warmup=100, num_samples=100, num_chains=2)


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
    """``({night label: OIData}, provenance)``: the prepared data of one system, blind to its published solution."""
    import trackb_fit as T
    from reanalyse_orbits import _read_with_vis

    T._read = _read_with_vis(T)  # GRAVITY keeps V2 beside the closure phases
    inst = record_of(slug)["instrument"].upper()
    sysdir = pathlib.Path(root) / "trackb" / slug
    dirs = ["oifits"] + (["oifits_extra"] if include_extra else [])
    files = sorted(f for d in dirs for f in (sysdir / d).glob("*.fits"))
    if not files:
        raise FileNotFoundError(f"no OIFITS under {sysdir}/{dirs}")
    notes, items = [], []
    for f in files:
        if T.instrument(f) != inst:
            notes.append(f"{f.name}: {T.instrument(f)}, not the record's {inst}; left out")
        else:
            items.append((f, T.mjd_of(f)))
    nights, used = {}, {}
    for night, fs in group_nights(items):
        d, _, dropped = T.load(fs)
        notes += [f"MJD {night}: dropped {x['file']}: {x.get('error', '')}" for x in dropped]
        nights[f"MJD {night}"] = d
        used[f"MJD {night}"] = [dict(file=f.name, sha256=sha256(f)) for f in fs if f.name not in {x["file"] for x in dropped}]
    if len(nights) < MIN_NIGHTS:
        raise ValueError(f"{slug}: {len(nights)} nights of {inst} data; at least {MIN_NIGHTS} are needed")
    return nights, dict(instrument=inst, include_extra=include_extra, notes=notes, nights=used)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the system slugs and exit")
    ap.add_argument("--system", choices=slugs())
    ap.add_argument("--out", default="gallery_orbits_out")
    ap.add_argument("--include-extra", action="store_true", dest="include_extra", help="also use oifits_extra/ (nights the paper has no epoch for)")
    ap.add_argument("--quick", action="store_true", help="smoke settings: tiny search and NUTS")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run", help="load and group the data, no fit")
    ap.add_argument("--no-quicklook-execute", action="store_true", dest="no_qe", help="write the quicklook notebook without running it")
    a = ap.parse_args(argv)
    if a.list:
        print("\n".join(slugs()))
        return 0
    if not a.system:
        ap.error("--system is required")
    root = os.environ.get("ESO_BIN_ROOT", ROOT_DEFAULT)
    t0 = time.time()
    nights, prov = load_nights(a.system, root, a.include_extra)
    span = [float(np.mean(np.asarray(d.mjd, float))) for d in nights.values()]
    print(f"{a.system}: {len(nights)} nights, MJD {min(span):.1f}-{max(span):.1f}, {prov['instrument']}")
    for n in prov["notes"]:
        print("note:", n)
    if a.dry_run:
        return 0
    from virgil.pipeline import OrbitPipeline

    settings = dict(QUICK) if a.quick else {}
    if a.no_qe:
        settings["quicklook_execute"] = False
    out = pathlib.Path(a.out) / a.system
    out.mkdir(parents=True, exist_ok=True)
    import virgil

    meta = dict(slug=a.system, virgil=getattr(virgil, "__version__", None), virgil_commit=os.environ.get("VIRGIL_COMMIT"),
                validation_commit=os.environ.get("VALIDATION_COMMIT"), settings=settings, **prov)
    (out / "gallery.json").write_text(json.dumps(meta, indent=1) + "\n")
    result = OrbitPipeline(nights, output=out / "run", **settings).run()
    meta["elapsed_s"] = time.time() - t0
    (out / "gallery.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {out} in {meta['elapsed_s']:.0f} s")
    return 0 if result is not None else 1


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.exit(main())
