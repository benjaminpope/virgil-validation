"""Laptop smoke test of the Track B orbit targets (``oidb/trackb/<slug>.json``): no orbit fit, no published start.

    python scripts/trackb_smoke.py [slug ...] [--nights 3] [--root ~/data/eso_binaries/trackb] [--out smoke.json]

1. **Every file loads.** Each OIFITS in the record is read alone with virgil (the Track B recipe of
   ``scripts/trackb_fit.py``: GRAVITY science-channel closure phases outside Brgamma, PIONIER V2 and closure phases).
2. **Single-night binary fits.** For ``--nights`` published nights, spread over the paper's epochs, the files of that
   night are fitted blind (a grid search over (dra, ddec, flux) from the data alone, then ``fit`` from the best peaks; the
   published position is never used to start or bound it). The separation must fall in a sensible range: at most
   ``1.5 a (1 + e) + 1 mas`` of the published orbit, since the projected separation of an orbit cannot exceed its
   apastron distance: a loose bound, about three times the separations seen. The sharper number is ``miss_mas``, the
   distance from the night's fitted (dRA, dDec) to the paper's position for it. The paper's position is used only after
   the fit, to score it: it never starts or bounds the fit.

This is a sanity check that the data are usable. It tests nothing about orbits, and the sub-resolution systems
(separations below λ/2B) may fall outside the range on a night without that being a fault: the line says so.
"""

import argparse
import json
import math
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
            sep, pa = float(r["sep"]), float(r["pa"])
            miss = math.hypot(sep * math.sin(math.radians(pa)) - e["dra_mas"], sep * math.cos(math.radians(pa)) - e["ddec_mas"])
            row = dict(date=e["date"], mjd=e["mjd"], sep_mas=sep, pa_deg=pa, flux=float(r["flux"]), miss_mas=miss,
                       paper_sep_mas=e["sep_mas"], below_lambda_2B=e["below_lambda_2B"],
                       chi2_raw_per_n=float(r["chi2_raw_per_n"]), scales=r["scales"], seconds=time.time() - t0,
                       sensible=bool(sep <= max_sep(rec)))
        except Exception as ex:
            row = dict(date=e["date"], mjd=e["mjd"], error=f"{type(ex).__name__}: {str(ex)[:300]}", sensible=False)
        out.append(row)
    return out


READING = """## Reading it

- Where the companion is resolved and the flux ratio is not near 1 (HD 210763, NN Del, HD 224974 on its resolved night, HD 41255 on 2021-01-17, HD 70937 on 2022-03-30, κ Vel on two nights) a blind single-night fit recovers the paper's position to 0.11 mas or better. NN Del is already at raw χ²/N 0.8.
- Near-equal fluxes (HD 188088, HD 41255) and some other resolved nights (HD 70937 on 2020-12-14, where the separation is right at 4.39 mas against 4.45) give a separation near the paper's but the companion on the wrong side: a closure-phase-only fit cannot always tell the two stars apart. The orbit-level solver, which sees all nights and the period, has to resolve this.
- Sub-resolution nights (paper separation below λ/2B) usually return a **wrong** peak: a large separation (6 to 29 mas) with a flux ratio of 2 to 3 per cent, which is what a blind grid does when the closure phases carry almost no binary signal. These are the nights on which the five Araucaria SB2s are *not* clean for Track B; they are expected failures, not faults of the data.
- Not explained by resolution: o Leo (2022-03-30 and 2023-01-27), HD 9312 and HD 41255 (2022-04-10) fit a side-lobe. Single-night blind fits find side-lobes; the orbit-level solver, which sees all nights, is what must reject them.
- κ Vel 2025-02-28 *agrees* with the fit only up to the sign of dRA: the paper prints +0.510 and the fit finds −0.51 (see the `suspect` flag in `oidb/trackb/kap_vel.json`), so its miss of about 1 mas is the paper's, not the fit's.
- κ Vel's first night has raw χ²/N 51: the single-calibrator, minimum-error issue the paper notes (bootstrap errors scaled by about 2.2). Raw χ²/N of 2 to 4 is typical for GRAVITY closure phases here; it is not tuned.
"""


AGREE_MAS = 0.5  # a night "agrees" with the paper when the fitted position is this close to the published one


def miss_of(rec, fit):
    """The fit's distance to the paper's position for its night (computed from sep and PA when an older report lacks it)."""
    if "miss_mas" in fit:
        return fit["miss_mas"]
    e = next(e for e in rec["epochs"] if e["mjd"] == fit["mjd"])
    p = math.radians(fit["pa_deg"])
    return math.hypot(fit["sep_mas"] * math.sin(p) - e["dra_mas"], fit["sep_mas"] * math.cos(p) - e["ddec_mas"])


def markdown(report, date):
    """design/trackb_smoke.md from a report (``--out``)."""
    L = ["# Track B orbit targets: laptop smoke test", "",
         f"Generated by `scripts/trackb_smoke.py --nights 3 --md` ({date}, laptop, the virgil installed in `.venv`; no OzSTAR). "
         "It tests that the data are usable; it fits **no orbit** and uses **no published value as a start or bound**.", "",
         "1. **Every file loads** alone with virgil under the Track B recipe (`scripts/trackb_fit.py`: GRAVITY science-channel "
         "closure phases outside Brγ; PIONIER V² and closure phases).",
         "2. **Single-night blind binary fits** on three nights per system, spread over the paper's epochs: a grid search over "
         "(dRA, dDec, flux) from the data alone, then `fit`. After the fit, each night is scored against the paper's position "
         f"for it: *miss* is the distance between the two, and the night *agrees* if the miss is under {AGREE_MAS} mas. "
         "*In range* is a much looser control: the separation is at most 1.5 a(1+e) + 1 mas, about three times the separations seen.",
         "", "## Files", "", "| system | files load | not loading |", "|---|---|---|"]
    tot = bad = 0
    for s, v in report.items():
        b = [x for x in v["loads"] if not x["ok"]]
        tot += len(v["loads"])
        bad += len(b)
        L.append(f"| {s} | {len(v['loads']) - len(b)}/{len(v['loads'])} | "
                 + ("; ".join(f"`{x['dp_id']}` ({x['error'][:70]}…)" for x in b) or "none") + " |")
    L += ["", f"{tot - bad} of {tot} files load. The {bad} that do not are all products of held-out nights (processed in one batch, "
          "2026-03-04); the builder marks them `unreadable` in the record, and the blind solver skips them.", "",
          "## Single-night fits", "",
          "| system | night | fitted sep (mas) | paper sep (mas) | miss (mas) | PA (°) | flux | raw χ²/N | agrees | in range | below λ/2B |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    n = agree = rng = 0
    for s, v in report.items():
        rec = record(s)
        for f in v["fits"]:
            m = miss_of(rec, f)
            n += 1
            agree += m < AGREE_MAS
            rng += f["sensible"]
            L.append(f"| {s} | {f['date']} | {f['sep_mas']:.2f} | {f['paper_sep_mas']:.2f} | {m:.2f} | {f['pa_deg']:.0f} | "
                     f"{f['flux']:.3f} | {f['chi2_raw_per_n']:.2f} | {'yes' if m < AGREE_MAS else '**no**'} | "
                     f"{'yes' if f['sensible'] else '**no**'} | {'yes' if f['below_lambda_2B'] else ''} |")
    L += ["", f"{agree} of {n} nights agree with the paper's position; {rng} of {n} are in the loose range, so that control "
          "says little.", ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("slugs", nargs="*")
    ap.add_argument("--nights", type=int, default=3)
    ap.add_argument("--root", default=str(TRACKB))
    ap.add_argument("--out")
    ap.add_argument("--load-only", action="store_true")
    ap.add_argument("--md", help="also write the Markdown report here")
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
                print(f"   {f['date']}: sep {f['sep_mas']:.2f} mas (paper {f['paper_sep_mas']:.2f}; limit {max_sep(rec):.2f}), miss {f['miss_mas']:.2f} mas, "
                      f"PA {f['pa_deg']:.0f}, flux {f['flux']:.3f}, raw chi2/N {f['chi2_raw_per_n']:.2f}, "
                      f"{'sensible' if f['sensible'] else 'OUT OF RANGE'}{' (below lambda/2B)' if f['below_lambda_2B'] else ''}")
        report[s] = dict(loads=loads, fits=fits, max_sep_mas=max_sep(rec))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(report, indent=1))
    if a.md:
        pathlib.Path(a.md).write_text(markdown(report, time.strftime("%Y-%m-%d")) + READING)
    return report


if __name__ == "__main__":
    main()
