"""Laptop smoke test of the Track B orbit targets (``oidb/trackb/<slug>.json``): no orbit fit, no published start.

    python scripts/trackb_smoke.py [slug ...] [--nights 3] [--root ~/data/eso_binaries/trackb] [--out smoke.json]

1. **Every file loads.** Each OIFITS in the record is read alone with virgil (the Track B recipe of
   ``scripts/trackb_fit.py``: GRAVITY science-channel closure phases outside Brgamma, PIONIER V2 and closure phases).
2. **Single-night binary fits.** For ``--nights`` published nights, spread over the paper's epochs, the files of that
   night are fitted blind (a grid search over (dra, ddec, flux) from the data alone, then ``fit`` from the best peaks; the
   published position is never used to start or bound it). The separation must fall in a sensible range: at most
   ``1.5 a (1 + e) + 1 mas`` of the published orbit, since the projected separation of an orbit cannot exceed its
   apastron distance. The published position of the night is printed beside it, for reading only.

This is a sanity check that the data are usable. It tests nothing about orbits, and the sub-resolution systems
(separations below λ/2B) may fall outside the range on a night without that being a fault: the line says so.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
RECORDS = ROOT / "oidb" / "trackb"
TRACKB = pathlib.Path.home() / "data" / "eso_binaries" / "trackb"


def record(slug):
    return json.loads((RECORDS / f"{slug}.json").read_text())


def path_of(root, slug, f):
    """Where a record's file is on disk: ``oifits`` (read by Track B) or ``oifits_extra`` (the rest)."""
    for sub in (f.get("dir"), "oifits", "oifits_extra"):
        if sub and (pathlib.Path(root) / slug / sub / f["file"]).exists():
            return pathlib.Path(root) / slug / sub / f["file"]
    return None


def diameters(rec):
    d = rec.get("diameters_mas") or {}

    def one(x):
        return float(x[0]) if isinstance(x, (list, tuple)) and x and isinstance(x[0], (int, float)) else None
    return one(d.get("primary")), one(d.get("secondary"))


def max_sep(rec):
    o = rec["orbit"]
    return 1.5 * o["a_mas"][0] * (1 + o["e"][0]) + 1.0


def load_all(rec, root):
    import trackb_fit as T
    out = []
    for f in rec["files"]:
        p = path_of(root, rec["slug"], f)
        row = dict(dp_id=f["dp_id"], role=f["role"], ok=False)
        if p is None:
            row["error"] = "file not on disk"
        else:
            try:
                data, inst, _ = T.load([p])
                row.update(ok=True, instrument=inst, n_vis=int(np.size(data.vis)), n_phi=int(np.size(data.phi)),
                           n_independent=int(data.n_independent))
            except Exception as e:  # report, do not stop: the point is the list
                row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out.append(row)
    return out


def pick(rec, n):
    """n epochs spread over the paper's, among those with a file on disk."""
    ok = [i for i, e in enumerate(rec["epochs"]) if e["dp_ids"]]
    if len(ok) <= n:
        return ok
    return [ok[round(k * (len(ok) - 1) / (n - 1))] for k in range(n)] if n > 1 else [ok[len(ok) // 2]]


def fit_nights(rec, root, n):
    import trackb_fit as T
    out = []
    files = {f["dp_id"]: f for f in rec["files"]}
    for i in pick(rec, n):
        e = rec["epochs"][i]
        paths = [path_of(root, rec["slug"], files[d]) for d in e["dp_ids"]]
        fmax = 2.0 if rec["slug"] in ("hd41255", "hd188088") else 1.0
        t0 = time.time()
        try:
            r = T.fit_epoch(rec["slug"], dict(files=paths, published=None), diameters(rec), fmax, False)
            sep = float(r["sep"])
            row = dict(date=e["date"], mjd=e["mjd"], sep_mas=sep, pa_deg=float(r["pa"]), flux=float(r["flux"]),
                       paper_sep_mas=e["sep_mas"], below_lambda_2B=e["below_lambda_2B"],
                       chi2_raw_per_n=float(r["chi2_raw_per_n"]), scales=r["scales"], seconds=time.time() - t0,
                       sensible=bool(sep <= max_sep(rec)))
        except Exception as ex:
            row = dict(date=e["date"], mjd=e["mjd"], error=f"{type(ex).__name__}: {str(ex)[:300]}", sensible=False)
        out.append(row)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("slugs", nargs="*")
    ap.add_argument("--nights", type=int, default=3)
    ap.add_argument("--root", default=str(TRACKB))
    ap.add_argument("--out")
    ap.add_argument("--load-only", action="store_true")
    a = ap.parse_args(argv)
    slugs = a.slugs or sorted(p.stem for p in RECORDS.glob("*.json"))
    report = {}
    for s in slugs:
        rec = record(s)
        loads = load_all(rec, a.root)
        bad = [x for x in loads if not x["ok"]]
        print(f"{s}: {len(loads) - len(bad)}/{len(loads)} files load" + (f"; FAILED {[b['dp_id'] for b in bad]}" if bad else ""))
        fits = [] if a.load_only else fit_nights(rec, a.root, a.nights)
        for f in fits:
            if "error" in f:
                print(f"   {f['date']}: ERROR {f['error']}")
            else:
                print(f"   {f['date']}: sep {f['sep_mas']:.2f} mas (paper {f['paper_sep_mas']:.2f}; limit {max_sep(rec):.2f}), "
                      f"PA {f['pa_deg']:.0f}, flux {f['flux']:.3f}, raw chi2/N {f['chi2_raw_per_n']:.2f}, "
                      f"{'sensible' if f['sensible'] else 'OUT OF RANGE'}{' (below lambda/2B)' if f['below_lambda_2B'] else ''}")
        report[s] = dict(loads=loads, fits=fits, max_sep_mas=max_sep(rec))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
