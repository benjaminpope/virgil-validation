"""UVFITS to measurement set to virgil input for the M87* replication (design/m87_replication.md, stage 3).

    .venv/bin/python scripts/m87_uvfits_to_ms.py --day 2017-04-11 [--out DIR] [--weights ms|eht]

Per fiducial day:
1. ``casatasks.importuvfits`` in ``.venv-alma`` (``external_bridge/m87_casa_worker.py``, task ``import``) makes
   ``<out>/<day>.ms`` and reads back what virgil needs: times, scans, antenna names in table order, spectral windows,
   polarizations. These are printed and saved in ``report.json``.
2. The raw cross-correlation samples are exported (task ``export``) and compared row by row (time, baseline) with
   eht-imaging's own reading of the UVFITS in ``.venv-basis``. The conjugation and the signs of u and v that make
   them agree are found here, not assumed (CASA's DATA column and UVW against the UVFITS; virgil's pinned rule for a
   simulated MS is ``V = conj(DATA)``, ``u = +UVW[0] nu / c``, which need not hold for an imported UVFITS).
3. ``<out>/ddid_NN.npz`` is written in that convention (keys ``u, v, re, im, weight, freq, ddid, obsid, time, ant1,
   ant2, chan, scan``), which ``virgil_bridge.m87_fit.load_vis`` reads. ``virgil``'s ``read_ms`` refuses a
   Stokes I MS (it wants a parallel-hand pair), so the fits read these files, not the MS.

``--weights ms`` (default) keeps CASA's WEIGHT; ``eht`` sets ``1 / sigma^2`` from eht-imaging's sigma on the matched
rows. The report gives the ratio of the two. Untested until ``.venv-alma`` exists (``scripts/setup_external.sh alma``)
and ``.venv-basis`` (``scripts/setup_m87.sh basis``); inputs from ``$M87_DATA`` (``scripts/setup_m87.sh data``).
"""

import argparse
import glob
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from external_bridge import _subprocess, basis_bridge  # noqa: E402

DATA = pathlib.Path(os.environ.get("M87_DATA", pathlib.Path.home() / "data" / "m87_eht"))
DAYS = {"2017-04-11": "M87_2017_101", "2018-04-21": "M87_2018_111", "2021-04-18": "M87_2021_108"}
C = 299792458.0


def find_uvfits(day):
    hits = sorted(glob.glob(str(DATA / "zenodo" / "gendirect" / "x" / "*" / "uvfits" / f"*{DAYS[day]}*_hops_netcal_StokesI.uvfits")))
    if not hits:
        raise FileNotFoundError(f"no {DAYS[day]} StokesI UVFITS under {DATA}/zenodo/gendirect (scripts/setup_m87.sh data)")
    return hits[0]


def key(time, a, b):
    return (round(float(time), 0), min(int(a), int(b)), max(int(a), int(b)))


def compare(raw, eht, ddid_first=True):
    """Match the MS samples to eht-imaging's rows and find the convention. Returns a dict (and the matched mask)."""
    index = {}
    for i, (t, a, b) in enumerate(zip(eht["time"], eht["t1"], eht["t2"])):
        index.setdefault(key(t, a, b), []).append(i)
    rows, partner, swapped = [], [], []
    for r, (t, a, b) in enumerate(zip(raw["time"], raw["ant1"], raw["ant2"])):
        cand = index.get(key(t, a, b), [])
        if len(cand) == 1:
            rows.append(r)
            partner.append(cand[0])
            swapped.append(int(a) != int(eht["t1"][cand[0]]))
    rows, partner, swapped = np.array(rows), np.array(partner), np.array(swapped)
    info = {"n_ms": int(len(raw["time"])), "n_eht": int(len(eht["time"])), "n_matched": int(len(rows)),
            "n_order_swapped": int(swapped.sum())}
    if len(rows) == 0:
        info["error"] = "no rows matched by (time, baseline)"
        return info, rows, partner
    u = raw["u_m"][rows] * raw["freq"][rows] / C
    v = raw["v_m"][rows] * raw["freq"][rows] / C
    d = raw["data"][rows].astype(complex)
    # bring the MS baseline into eht-imaging's order: a reversed baseline has conjugate data and negated u, v
    flip = np.where(swapped, -1.0, 1.0)
    u, v, d = u * flip, v * flip, np.where(swapped, np.conj(d), d)
    ue, ve, ve_vis = eht["u"][partner], eht["v"][partner], eht["vis"][partner]
    best = None
    scores = {}
    for su in (1.0, -1.0):
        for sv in (1.0, -1.0):
            e = np.max(np.hypot(su * u - ue, sv * v - ve)) / np.max(np.hypot(ue, ve))
            scores[f"u{su:+.0f}_v{sv:+.0f}"] = float(e)
            if best is None or e < best[0]:
                best = (e, su, sv)
    vis_scores = {"data": float(np.max(np.abs(d - ve_vis)) / np.max(np.abs(ve_vis))),
                  "conj(data)": float(np.max(np.abs(np.conj(d) - ve_vis)) / np.max(np.abs(ve_vis)))}
    conjugate = vis_scores["conj(data)"] < vis_scores["data"]
    info.update({"uv_relative_error_by_sign": scores, "u_sign": best[1], "v_sign": best[2],
                 "vis_relative_error": vis_scores, "conjugate": bool(conjugate),
                 "virgil_pinned_convention_holds": bool(conjugate and best[1] == 1.0 and best[2] == 1.0),
                 "median_weight_x_sigma2": float(np.median(raw["weight"][rows] * eht["sigma"][partner] ** 2))})
    info["agree"] = bool(min(vis_scores.values()) < 1e-3 and best[0] < 1e-3)
    info["_sigma"] = (rows, eht["sigma"][partner])
    return info, rows, partner


def convert(day, out, weights, uvfits=None):
    uvfits = uvfits or find_uvfits(day)
    out = pathlib.Path(out or DATA / "ms" / day)
    out.mkdir(parents=True, exist_ok=True)
    ms = out / f"{day}.ms"
    print(f"{day}: {uvfits}")
    imp = _subprocess.run("alma", "m87_casa_worker.py", {"task": "import", "uvfits": uvfits, "ms": str(ms)})
    print(f"  MS: {imp['n_rows']} rows, {imp['n_times']} times, {imp['n_scans']} scans "
          f"(SCAN_NUMBER {imp['scan_numbers'][:6]}...), TIME {imp['time_min_s']:.1f}..{imp['time_max_s']:.1f} s")
    print(f"  antennas (table order): {imp['antennas']}")
    print(f"  spw channels: {[len(f) for f in imp['spw_freq_hz']]}, CORR_TYPE {imp['corr_type']}, DATA columns {imp['datacolumns']}")
    raw_dir = out / "raw"
    exp = _subprocess.run("alma", "m87_casa_worker.py", {"task": "export", "ms": str(ms), "out": str(raw_dir)})
    eht_npz = out / "eht_reading.npz"
    dump = basis_bridge.dump_uvfits(uvfits, eht_npz)
    eht = dict(np.load(eht_npz))
    names_match = [str(s) for s in eht["stations"]] == imp["antennas"]
    print(f"  eht-imaging stations {dump['stations']}; same order as the MS ANTENNA table: {names_match}")
    report = {"day": day, "uvfits": uvfits, "ms": str(ms), "import": imp, "eht": {k: v for k, v in dump.items() if k != "out"},
              "station_order_matches": names_match, "ddids": []}
    final = []
    for f in exp["files"]:
        raw = dict(np.load(f["file"]))
        info, rows, partner = compare(raw, eht)
        sig = info.pop("_sigma", None)
        report["ddids"].append({**{k: v for k, v in f.items() if k != "file"}, "compare": info})
        print(f"  ddid {f['ddid']}: {info}")
        if "error" in info or not info["agree"]:
            print("  !! conversion does not reproduce eht-imaging's reading; no npz written for this ddid")
            continue
        su, sv, conj = info["u_sign"], info["v_sign"], info["conjugate"]
        vis = np.conj(raw["data"]) if conj else raw["data"]
        weight = raw["weight"].astype(float)
        if weights == "eht":
            weight = np.full(len(weight), np.nan)
            weight[rows] = 1.0 / sig[1] ** 2
        keep = np.isfinite(weight)
        np.savez(out / f"ddid_{f['ddid']:02d}.npz", u=(su * raw["u_m"] * raw["freq"] / C)[keep],
                 v=(sv * raw["v_m"] * raw["freq"] / C)[keep], re=vis.real[keep].astype(np.float64),
                 im=vis.imag[keep].astype(np.float64), weight=weight[keep], freq=raw["freq"][keep],
                 ddid=int(f["ddid"]), obsid=0, time=raw["time"][keep], ant1=raw["ant1"][keep], ant2=raw["ant2"][keep],
                 chan=raw["chan"][keep], scan=raw["scan"][keep])
        final.append(f["ddid"])
    report["npz_ddids"] = final
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"  wrote {out}/report.json and ddid npz for {final}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--day", choices=sorted(DAYS), action="append", help="fiducial day (repeat; default all three)")
    ap.add_argument("--uvfits", help="a UVFITS to convert instead of the fiducial day's (needs one --day for the name)")
    ap.add_argument("--out", help="output directory (default $M87_DATA/ms/<day>)")
    ap.add_argument("--weights", choices=["ms", "eht"], default="ms")
    a = ap.parse_args()
    for day in a.day or sorted(DAYS):
        convert(day, a.out, a.weights, a.uvfits)
