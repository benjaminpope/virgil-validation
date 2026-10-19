"""Run inside CASA's own environment (.venv-alma), not ours.

    python alma_worker.py task.json result.json

Tasks (``task["task"]``):

``simulate_ms``  make a small measurement set from scratch with
                 ``casatools.simulator`` and put two point sources into it
                 with CASA's own predict (the "casa" root of trust);
``read``         our reader of a measurement set, written from CASA's
                 MS definition (casacore MS v2 specification, tables
                 MAIN, DATA_DESCRIPTION, SPECTRAL_WINDOW, POLARIZATION)
                 through ``casatools.table``; one .npz per ddid;
``visread_read`` the same output through visread, for comparison only;
``add_weight_spectrum`` add a WEIGHT_SPECTRUM column (for its test);
``flag_ms``      mark some samples of a synthetic MS bad (for the flag test);
``info``         per ddid: N_vis, channels, channel width, integration time.

The conventions (docs/alma_conventions.md) are applied here exactly as the
spec lists them; ``conjugate`` is the one choice the synthetic-MS test makes.
Nothing here imports virgil.
"""

import json
import os
import pathlib
import sys

import numpy as np

C = 299792458.0
MAS = np.pi / 180 / 3600 / 1000

# Measures data live inside the venv (scripts/setup_external.sh writes this file)
_site = pathlib.Path(sys.executable).parents[1] / "casasiteconfig.py"
if "CASASITECONFIG" not in os.environ and _site.exists():
    os.environ["CASASITECONFIG"] = str(_site)

PHASE_CENTRE = "J2000 15h56m09.20s -37d56m06.42s"
RA0_DEG = (15 + 56 / 60 + 9.20 / 3600) * 15.0
DEC0_DEG = -(37 + 56 / 60 + 6.42 / 3600)
# 8 antennas, local East, North, Up (m): an irregular layout chosen so that the
# dirty beam of one hour at hour angle -1.5 h to -0.5 h has sidelobes low enough
# for each source's peak to sit within a pixel (longest baseline about 1.6 km)
ANTENNAS = [
    [785.0, 132.0, 0.0], [-304.0, 127.0, 0.0], [-580.0, 55.0, 0.0], [662.0, 287.0, 0.0],
    [672.0, 24.0, 0.0], [404.0, -728.0, 0.0], [784.0, -83.0, 0.0], [180.0, -815.0, 0.0],
]
# (flux Jy, East mas, North mas): the brighter source is the first
COMPONENTS = [(0.010, 300.0, 100.0), (0.005, -200.0, -400.0)]
SPWS = [(230e9, "spw0"), (245e9, "spw1")]
NCHAN, CHAN_WIDTH = 4, 125e6
# CORR_TYPE codes of POLARIZATION: XX, YY (linear), RR, LL (circular)
PARALLEL = {(9, 12), (5, 8)}


def _quiet_import():
    import casatools
    return casatools


def simulate_ms(task):
    ct = _quiet_import()
    path = pathlib.Path(task["path"])
    noise = float(task.get("noise_jy") or 0.0)
    if path.exists():
        import shutil
        shutil.rmtree(path)
    me, qa = ct.measures(), ct.quanta()
    sm = ct.simulator()
    sm.open(str(path))
    xyz = np.array(ANTENNAS)
    sm.setconfig(
        telescopename="ALMA", x=xyz[:, 0].tolist(), y=xyz[:, 1].tolist(), z=xyz[:, 2].tolist(),
        dishdiameter=[12.0] * len(xyz), mount=["alt-az"] * len(xyz),
        antname=[f"A{i:02d}" for i in range(len(xyz))], coordsystem="local",
        referencelocation=me.observatory("ALMA"),
    )
    for freq, name in SPWS:
        sm.setspwindow(spwname=name, freq=f"{freq}Hz", deltafreq=f"{CHAN_WIDTH}Hz",
                       freqresolution=f"{CHAN_WIDTH}Hz", nchannels=NCHAN, stokes="XX YY")
    sm.setfeed(mode="perfect X Y")
    sm.setfield(sourcename="IMLUP", sourcedirection=PHASE_CENTRE)
    two = bool(task.get("two_fields"))
    if two:  # a second pointing (for the mosaic guard), 1 arcmin North
        sm.setfield(sourcename="OTHER", sourcedirection=f"J2000 15h56m09.20s -37d55m06.42s")
    sm.setlimits(shadowlimit=0.0, elevationlimit="5deg")
    sm.setauto(autocorrwt=0.0)
    sm.settimes(integrationtime="6s", usehourangle=True,
                referencetime=me.epoch("utc", "2025-04-01T00:00:00"))
    for _, name in SPWS:
        sm.observe(sourcename="IMLUP", spwname=name, starttime="-1.5h", stoptime="-1.0h" if two else "-0.5h")
        if two:
            sm.observe(sourcename="OTHER", spwname=name, starttime="-1.0h", stoptime="-0.5h")
    sm.setdata(spwid=list(range(len(SPWS))), fieldid=[0, 1] if two else [0])
    cl = ct.componentlist()
    dec0 = np.deg2rad(DEC0_DEG)
    clpath = str(path) + ".cl"
    for flux, east, north in COMPONENTS:
        ra = RA0_DEG + east * MAS / np.cos(dec0) * 180 / np.pi
        dec = DEC0_DEG + north * MAS * 180 / np.pi
        cl.addcomponent(flux=flux, fluxunit="Jy", shape="point", polarization="Stokes",
                        dir=f"J2000 {ra}deg {dec}deg", freq="237.5GHz", spectrumtype="constant")
    cl.rename(clpath)
    cl.done()
    sm.predict(complist=clpath)
    if noise > 0:
        sm.setnoise(mode="simplenoise", simplenoise=f"{noise}Jy")
        sm.corrupt()
    sm.done()
    import shutil
    shutil.rmtree(clpath, ignore_errors=True)
    # WEIGHT: 1/sigma^2 of each of Re and Im of a correlation product, set
    # explicitly (the simulator leaves WEIGHT at 1); the test checks it
    tb = ct.table()
    tb.open(str(path), nomodify=False)
    nrow = tb.nrows()
    w = np.full((2, nrow), 1.0 / noise**2 if noise > 0 else 1.0)
    tb.putcol("WEIGHT", w)
    tb.putcol("SIGMA", 1.0 / np.sqrt(w))
    tb.close()
    return {"path": str(path), "casatools": ct.version_string(), "noise_jy": noise,
            "components": COMPONENTS, "n_ddid": len(SPWS)}


def _ddid_tables(tb, path):
    """(SPECTRAL_WINDOW_ID, POLARIZATION_ID) for every DATA_DESC_ID."""
    tb.open(f"{path}/DATA_DESCRIPTION")
    spw, pol = tb.getcol("SPECTRAL_WINDOW_ID"), tb.getcol("POLARIZATION_ID")
    tb.close()
    return spw, pol


def _spw(tb, path, spw_id):
    tb.open(f"{path}/SPECTRAL_WINDOW")
    freq = tb.getcell("CHAN_FREQ", spw_id)
    width = tb.getcell("CHAN_WIDTH", spw_id)
    tb.close()
    return np.asarray(freq, float), np.asarray(width, float)


def _corr_type(tb, path, pol_id):
    tb.open(f"{path}/POLARIZATION")
    ct = np.asarray(tb.getcell("CORR_TYPE", pol_id)).ravel()
    tb.close()
    return ct


def read_ddid(ct, path, ddid, stride=1, conjugate=False, datacolumn="DATA", field=None):
    """One ddid as a dict of arrays, with the conventions of the spec.

    Samples: cross-correlations, not FLAG_ROW, neither parallel-hand
    correlation flagged, both weights positive; polarizations combined as
    V = (w1 V1 + w2 V2) / (w1 + w2), w = w1 + w2; weight from
    WEIGHT_SPECTRUM when present and defined, otherwise WEIGHT per channel.
    u = UVW[0] nu / c, v = UVW[1] nu / c per channel.
    """
    tb = ct.table()
    spw_of, pol_of = _ddid_tables(tb, path)
    freq, width = _spw(tb, path, int(spw_of[ddid]))
    corr = _corr_type(tb, path, int(pol_of[ddid]))
    pair = [i for i, c in enumerate(corr) if c in (9, 12, 5, 8)]
    if len(pair) != 2 or tuple(sorted(int(corr[i]) for i in pair)) not in PARALLEL:
        raise ValueError(f"ddid {ddid}: no pair of parallel-hand correlations in CORR_TYPE {corr}")
    tb.open(path)
    has_ws = "WEIGHT_SPECTRUM" in tb.colnames()
    sel = f"DATA_DESC_ID=={ddid} && ANTENNA1!=ANTENNA2" + ("" if field is None else f" && FIELD_ID=={int(field)}")
    q = tb.query(sel)
    tb.close()
    n = q.nrows()
    if n == 0:
        q.close()
        return None
    fields = np.unique(q.getcol("FIELD_ID"))
    if len(fields) > 1:
        q.close()
        raise ValueError(f"ddid {ddid}: the selection holds fields {fields.tolist()} (a mosaic?); give one with `field`")
    step = max(int(stride), 1)
    nsel = (n + step - 1) // step

    def col(name):
        return q.getcol(name, 0, nsel, step)

    data, flag = col(datacolumn), col("FLAG")  # (npol, nchan, nrow)
    uvw, flag_row = col("UVW"), col("FLAG_ROW")
    weight = col("WEIGHT")  # (npol, nrow)
    ws = None
    if has_ws:
        first, last = q.iscelldefined("WEIGHT_SPECTRUM", 0), q.iscelldefined("WEIGHT_SPECTRUM", n - 1)
        if first != last:
            q.close()
            raise ValueError(f"ddid {ddid}: WEIGHT_SPECTRUM is defined in some rows only; refusing to mix it with WEIGHT")
        if first:
            ws = col("WEIGHT_SPECTRUM")
            if ws.shape != data.shape or not np.all(np.isfinite(ws)):
                q.close()
                raise ValueError(f"ddid {ddid}: WEIGHT_SPECTRUM has shape {ws.shape}, DATA {data.shape}, or is not finite")
    ant1, ant2, time = col("ANTENNA1"), col("ANTENNA2"), col("TIME")
    scan, obs, interval = col("SCAN_NUMBER"), col("OBSERVATION_ID"), col("INTERVAL")
    q.close()
    i1, i2 = pair
    nchan, nrow = data.shape[1], data.shape[2]
    if ws is None:
        w1 = np.broadcast_to(weight[i1][None, :], (nchan, nrow))
        w2 = np.broadcast_to(weight[i2][None, :], (nchan, nrow))
    else:
        w1, w2 = ws[i1], ws[i2]
    good = ~flag[i1] & ~flag[i2] & ~flag_row[None, :] & (w1 > 0) & (w2 > 0)
    wsum = w1 + w2
    with np.errstate(invalid="ignore", divide="ignore"):
        vis = (w1 * data[i1] + w2 * data[i2]) / wsum
    if conjugate:
        vis = np.conj(vis)
    ch, rw = np.nonzero(good)
    nu = freq[ch]
    return {
        "u": uvw[0, rw] * nu / C, "v": uvw[1, rw] * nu / C,
        "re": vis[ch, rw].real.astype(np.float32), "im": vis[ch, rw].imag.astype(np.float32),
        "weight": wsum[ch, rw].astype(np.float32), "freq": nu,
        "ant1": ant1[rw].astype(np.int32), "ant2": ant2[rw].astype(np.int32),
        "chan": ch.astype(np.int16), "time": time[rw], "scan": scan[rw].astype(np.int32),
        "obsid": int(obs[0]), "ddid": int(ddid),
        "weight_source": "WEIGHT_SPECTRUM" if ws is not None else "WEIGHT",
        "chan_width_hz": float(np.median(np.abs(width))), "t_int_s": float(np.median(interval)),
    }


def _ddids(ct, path, wanted):
    tb = ct.table()
    tb.open(f"{path}/DATA_DESCRIPTION")
    n = tb.nrows()
    tb.close()
    return list(range(n)) if wanted is None else [int(d) for d in wanted]


def read(task):
    ct = _quiet_import()
    path, out = task["path"], pathlib.Path(task["out"])
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for d in _ddids(ct, path, task.get("ddids")):
        rec = read_ddid(ct, path, d, task.get("stride", 1), task.get("conjugate", True),
                        task.get("datacolumn", "DATA"), task.get("field"))
        if rec is None:
            continue
        f = out / f"ddid_{d:02d}.npz"
        np.savez(f, **rec)
        files.append({"file": str(f), "n": int(len(rec["u"])), "ddid": d, "obsid": rec["obsid"]})
    return {"files": files, "casatools": ct.version_string(), "conjugate": task.get("conjugate", True)}


def visread_read(task):
    """The same output through visread (a comparison, not our reader).

    visread (0.0.5) returns channels in decreasing frequency, the data
    already conjugated and averaged over polarizations, and the baselines in
    metres; they are put back in MS channel order and turned into the same
    arrays as ``read``, with visread's flags and no further weight cuts.
    """
    from visread import process_casa
    ct = _quiet_import()
    path, out = task["path"], pathlib.Path(task["out"])
    out.mkdir(parents=True, exist_ok=True)
    tb = ct.table()
    spw_of, _ = _ddid_tables(tb, path)
    files = []
    for d in _ddids(ct, path, task.get("ddids")):
        r = process_casa.get_processed_visibilities(path, d, datacolumn=task.get("datacolumn", "data"))
        ms_freq, _ = _spw(tb, path, int(spw_of[d]))
        data, flag, freq = r["data"], r["flag"], np.asarray(r["frequencies"], float)
        if len(freq) > 1 and (freq[1] < freq[0]) != (ms_freq[1] < ms_freq[0]):
            data, flag, freq = data[::-1], flag[::-1], freq[::-1]
        nchan, nrow = data.shape
        w = np.broadcast_to(np.asarray(r["weight"])[None, :], (nchan, nrow)) if np.ndim(r["weight"]) == 1 else r["weight"]
        ch, rw = np.nonzero(~flag)
        nu = freq[ch]
        rec = {
            "u": r["uu"][rw] * nu / C, "v": r["vv"][rw] * nu / C,
            "re": data[ch, rw].real.astype(np.float32), "im": data[ch, rw].imag.astype(np.float32),
            "weight": w[ch, rw].astype(np.float32), "freq": nu,
            "ant1": np.asarray(r["antenna1"])[rw].astype(np.int32),
            "ant2": np.asarray(r["antenna2"])[rw].astype(np.int32),
            "chan": ch.astype(np.int16), "time": np.asarray(r["time"])[rw],
        }
        f = out / f"ddid_{d:02d}.npz"
        np.savez(f, **rec)
        files.append({"file": str(f), "n": int(len(rec["u"])), "ddid": d})
    import importlib.metadata as m
    return {"files": files, "visread_version": m.version("visread")}


def info(task):
    ct = _quiet_import()
    path = task["path"]
    tb = ct.table()
    spw_of, pol_of = _ddid_tables(tb, path)
    rows = []
    for d in _ddids(ct, path, task.get("ddids")):
        freq, width = _spw(tb, path, int(spw_of[d]))
        tb.open(path)
        q = tb.query(f"DATA_DESC_ID=={d} && ANTENNA1!=ANTENNA2")
        tb.close()
        n = q.nrows()
        interval = float(np.median(q.getcol("INTERVAL", 0, min(n, 2000), max(n // 2000, 1)))) if n else 0.0
        obs = int(q.getcell("OBSERVATION_ID", 0)) if n else -1
        has_ws = bool(n) and "WEIGHT_SPECTRUM" in q.colnames() and q.iscelldefined("WEIGHT_SPECTRUM", 0)
        q.close()
        rows.append({"ddid": d, "obsid": obs, "n_rows": n, "weight_source": "WEIGHT_SPECTRUM" if has_ws else "WEIGHT", "nchan": len(freq),
                     "chan_width_hz": float(np.median(np.abs(width))), "t_int_s": interval,
                     "freq_min_hz": float(freq.min()), "freq_max_hz": float(freq.max()),
                     "corr_type": [int(c) for c in _corr_type(tb, path, int(pol_of[d]))]})
    return {"ddids": rows, "casatools": ct.version_string()}


def flag_ms(task):
    """For the flag test: in rows 0-99 flag XX, in 100-199 set FLAG_ROW, in
    200-299 zero the YY weight, in 300-399 flag channel 1 only (all of one
    ddid; the other is untouched)."""
    ct = _quiet_import()
    tb = ct.table()
    tb.open(task["path"], nomodify=False)
    q = tb.query(f"DATA_DESC_ID=={int(task.get('ddid', 0))}")
    rows = np.asarray(q.rownumbers())
    q.close()
    r = rows[:400]
    flag = tb.getcol("FLAG")  # (npol, nchan, nrow)
    flag[0, :, r[:100]] = True
    flag[:, 1, r[300:400]] = True
    tb.putcol("FLAG", flag)
    fr = tb.getcol("FLAG_ROW")
    fr[r[100:200]] = True
    tb.putcol("FLAG_ROW", fr)
    w = tb.getcol("WEIGHT")
    w[1, r[200:300]] = 0.0
    tb.putcol("WEIGHT", w)
    tb.close()
    return {"rows": [int(x) for x in r[:3]]}


def add_weight_spectrum(task):
    """For the weight-branch test: add WEIGHT_SPECTRUM = WEIGHT x (channel + 1)
    to every row, so that it differs from WEIGHT in value and by channel."""
    ct = _quiet_import()
    tb = ct.table()
    tb.open(task["path"], nomodify=False)
    desc = tb.getcoldesc("DATA")
    desc.update(valueType="float", comment="weight spectrum")
    for k in ("keywords", "dataManagerType", "dataManagerGroup"):
        desc.pop(k, None)
    tb.addcols({"WEIGHT_SPECTRUM": desc})
    nchan = tb.getcol("DATA", 0, 1).shape[1]
    w = tb.getcol("WEIGHT")  # (npol, nrow)
    tb.putcol("WEIGHT_SPECTRUM", w[:, None, :] * (np.arange(nchan) + 1.0)[None, :, None])
    tb.close()
    return {}


TASKS = {"add_weight_spectrum": add_weight_spectrum, "flag_ms": flag_ms, "simulate_ms": simulate_ms, "read": read, "visread_read": visread_read, "info": info}

if __name__ == "__main__":
    task = json.load(open(sys.argv[1]))
    result = TASKS[task["task"]](task)
    json.dump(result, open(sys.argv[2], "w"), default=str)
