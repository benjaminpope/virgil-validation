"""Run inside the CASA environment (.venv-alma), not ours: UVFITS to a measurement set for the M87* replication.

    python m87_casa_worker.py task.json result.json

Tasks
-----
``import``
    ``casatasks.importuvfits(fitsfile, vis, antnamescheme="old")``, then read back the fields virgil needs: row,
    scan and distinct time counts, the TIME range (MJD seconds), SCAN_NUMBER values, the antenna names in table
    order (the order BASIS's reference-station rule uses), the spectral windows, the polarizations (CORR_TYPE)
    and the first rows of ANTENNA1/ANTENNA2/TIME/UVW. Untested until ``.venv-alma`` exists: run
    ``scripts/m87_uvfits_to_ms.py`` and read its report before trusting the conversion.
``export``
    Raw cross-correlation samples of an MS per ddid, as npz files in `out` (see `export`), for the comparison with
    eht-imaging's own reading (and the final ddid_NN.npz). A Stokes I MS has one correlation, which
    the virgil and alma readers (parallel-hand pairs only) refuse, so this reader takes it.
"""

import json
import pathlib
import shutil
import sys

import numpy as np


def _tables(ct, path):
    tb = ct.table()
    out = {}
    tb.open(f"{path}/ANTENNA")
    out["antennas"] = [str(x) for x in tb.getcol("NAME")]
    tb.close()
    tb.open(f"{path}/SPECTRAL_WINDOW")
    out["spw_freq_hz"] = [[float(f) for f in tb.getcell("CHAN_FREQ", i)] for i in range(tb.nrows())]
    tb.close()
    tb.open(f"{path}/POLARIZATION")
    out["corr_type"] = [[int(c) for c in tb.getcell("CORR_TYPE", i)] for i in range(tb.nrows())]
    tb.close()
    tb.open(f"{path}/DATA_DESCRIPTION")
    out["ddid_to_spw"] = [int(x) for x in tb.getcol("SPECTRAL_WINDOW_ID")]
    tb.close()
    return out


def import_uvfits(task):
    import casatools as ct
    from casatasks import importuvfits

    ms = pathlib.Path(task["ms"])
    if ms.exists():
        shutil.rmtree(ms)
    importuvfits(fitsfile=task["uvfits"], vis=str(ms), antnamescheme=task.get("antnamescheme", "old"))
    tb = ct.table()
    tb.open(str(ms))
    time, scan = tb.getcol("TIME"), tb.getcol("SCAN_NUMBER")
    a1, a2 = tb.getcol("ANTENNA1"), tb.getcol("ANTENNA2")
    uvw = tb.getcol("UVW")
    info = {
        "ms": str(ms), "casatools": ct.version_string(), "n_rows": int(len(time)),
        "n_times": int(len(np.unique(time))), "n_scans": int(len(np.unique(scan))),
        "scan_numbers": [int(s) for s in np.unique(scan)][:200],
        "time_min_s": float(time.min()), "time_max_s": float(time.max()),
        "n_cross": int(np.sum(a1 != a2)), "datacolumns": [c for c in tb.colnames() if c.endswith("DATA")],
        "first_rows": {"time": time[:5].tolist(), "ant1": a1[:5].tolist(), "ant2": a2[:5].tolist(),
                       "uvw": uvw[:, :5].tolist()},
    }
    tb.close()
    info.update(_tables(ct, str(ms)))
    return info


def export(task):
    """Raw per-ddid arrays of the cross-correlations (one sample per unflagged row and channel), no convention
    applied: u_m, v_m (metres), data (complex DATA; the weighted mean of the two parallel hands when the MS has
    them, the single correlation of a Stokes I MS), weight (WEIGHT, summed over the hands used), freq, chan, ant1,
    ant2, time (MJD s), scan. ``scripts/m87_uvfits_to_ms.py`` applies the conjugation and u sign that make these
    agree with eht-imaging's reading, then writes the ``ddid_NN.npz`` that ``virgil_bridge.m87_fit.load_vis`` reads."""
    import casatools as ct

    ms, out = task["ms"], pathlib.Path(task["out"])
    out.mkdir(parents=True, exist_ok=True)
    info = _tables(ct, ms)
    tb = ct.table()
    tb.open(ms)
    ddids = sorted({int(d) for d in np.unique(tb.getcol("DATA_DESC_ID"))})
    tb.close()
    files = []
    for d in ddids:
        corr = info["corr_type"][d] if d < len(info["corr_type"]) else info["corr_type"][0]
        tb.open(ms)
        q = tb.query(f"DATA_DESC_ID=={d} && ANTENNA1!=ANTENNA2")
        data, flag, uvw = q.getcol("DATA"), q.getcol("FLAG"), q.getcol("UVW")  # (npol, nchan, nrow)
        weight, flag_row = q.getcol("WEIGHT"), q.getcol("FLAG_ROW")
        cols = {k: q.getcol(k) for k in ("ANTENNA1", "ANTENNA2", "TIME", "SCAN_NUMBER")}
        q.close()
        tb.close()
        pol = [i for i, c in enumerate(corr) if c in (1, 5, 8, 9, 12)]  # I, RR, LL, XX, YY
        if not pol:
            raise ValueError(f"ddid {d}: CORR_TYPE {corr} holds no I/RR/LL/XX/YY")
        w = weight[pol]  # (npol_used, nrow)
        ok = ~flag[pol].any(axis=0) & ~flag_row[None, :] & (w.min(axis=0)[None, :] > 0)  # (nchan, nrow)
        vis = (w[:, None, :] * data[pol]).sum(axis=0) / w.sum(axis=0)[None, :]
        freq = np.asarray(info["spw_freq_hz"][info["ddid_to_spw"][d]])
        ch, rw = np.nonzero(ok)
        f = out / f"ddid_{d:02d}.npz"
        np.savez(f, u_m=uvw[0, rw], v_m=uvw[1, rw], data=vis[ch, rw], weight=w.sum(axis=0)[rw], freq=freq[ch],
                 chan=ch.astype(np.int16), ant1=cols["ANTENNA1"][rw].astype(np.int32),
                 ant2=cols["ANTENNA2"][rw].astype(np.int32), time=cols["TIME"][rw],
                 scan=cols["SCAN_NUMBER"][rw].astype(np.int32), ddid=d, obsid=0)
        files.append({"file": str(f), "n": int(len(rw)), "ddid": d, "corr_type": [int(c) for c in corr], "pols_used": pol})
    return {"files": files, "antennas": info["antennas"], "casatools": ct.version_string()}


TASKS = {"import": import_uvfits, "export": export}

if __name__ == "__main__":
    task = json.load(open(sys.argv[1]))
    json.dump(TASKS[task["task"]](task), open(sys.argv[2], "w"))
