"""Bin the science channels of already calibrated GRAVITY files (a PI's calibrated OIFITS, e.g. OiDB) for the gallery path.

virgil-vlti calibrates raw products itself (``gravity_calibrate_gallery.py``). Some gallery targets only have the data
provider's calibrated files (HR 6819, Klement+25: ``*_SINGLE_SCI_VIS_CALIBRATED*.fits``, HIGH resolution, 1628 channels).
Read unbinned, one night has about 14000 samples, ten times a binned HIGH night of the Track B systems, and the orbit
pipeline's cost grows as N². This writes, per night, one file of the same layout the loader reads: the science table
(``GRAVITY_SC``) binned by ``--bin`` channels (default 8, as the HIGH setting of ``gravity_calibrate_gallery.py``), V² and
closure phases only, the two polarisation tables of a split-polarisation file averaged into one.

Binning: V² and its error are the mean of the unflagged channels of a bin and ``sqrt(sum err²)/n`` (independent channels:
an optimistic error, which the gains and the fitted error scales of the pipeline are there to absorb); closure phases are
the circular mean, with the same error rule. A bin with fewer than half its channels unflagged is flagged.

    python scripts/gravity_bin_calibrated.py ~/data/eso_binaries/trackb/hr6819 [--bin 8]

Reads ``<root>/oifits/*.fits``, writes ``<root>/calibrated/HIGH/<night>/<file>.fits`` (the night folder is the
UT-noon-to-noon night, as virgil-vlti names them).
"""

import argparse
import datetime
import pathlib

import numpy as np
from astropy.io import fits

MJD_ZERO = datetime.date(1858, 11, 17)
SC = "GRAVITY_SC"


def night_of(mjd):
    return (MJD_ZERO + datetime.timedelta(days=int(np.floor(mjd - 0.5)))).isoformat()


def _pol_tables(hdul, name):
    """Rows of every science-channel table ``name`` (one per polarisation), as a list of HDUs."""
    return [h for h in hdul if h.name == name and str(h.header.get("INSNAME", "")).startswith(SC)]


def _wavelengths(hdul):
    waves = [h for h in hdul if h.name == "OI_WAVELENGTH" and str(h.header.get("INSNAME", "")).startswith(SC)]
    return np.asarray(waves[0].data["EFF_WAVE"], float), np.asarray(waves[0].data["EFF_BAND"], float)


def _blocks(n_chan, nbin):
    n = n_chan // nbin
    return n, np.arange(n * nbin).reshape(n, nbin)


def _bin_real(val, err, flag, idx):
    """Mean and error of ``val`` ``(rows, chan)`` over the channel blocks ``idx``; flagged or NaN channels are dropped."""
    v, e, f = val[:, idx], err[:, idx], flag[:, idx] | ~np.isfinite(val[:, idx]) | ~np.isfinite(err[:, idx]) | (err[:, idx] <= 0)
    good = ~f
    n = good.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, np.where(good, v, 0).sum(-1) / n, 0.0)
        sig = np.where(n > 0, np.sqrt(np.where(good, e**2, 0).sum(-1)) / n, 1.0)
    return mean, sig, n < idx.shape[1] / 2


def _bin_phase(phi_deg, err_deg, flag, idx):
    v, e = phi_deg[:, idx], err_deg[:, idx]
    f = flag[:, idx] | ~np.isfinite(v) | ~np.isfinite(e) | (e <= 0)
    good = ~f
    n = good.sum(-1)
    z = np.where(good, np.exp(1j * np.radians(v)), 0).sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, np.degrees(np.angle(z)), 0.0)
        sig = np.where(n > 0, np.sqrt(np.where(good, e**2, 0).sum(-1)) / n, 1.0)
    return mean, sig, n < idx.shape[1] / 2


def _combine(tables, cols, binner, idx, errcol, flagcol):
    """Bin each polarisation table, then average the polarisations (inverse-variance) where both are good."""
    outs = []
    for t in tables:
        d = t.data
        m, s, f = binner(np.asarray(d[cols], float), np.asarray(d[errcol], float), np.asarray(d[flagcol], bool), idx)
        outs.append((m, s, f))
    if len(outs) == 1:
        return outs[0]
    w = np.stack([np.where(f, 0.0, 1 / s**2) for _, s, f in outs])
    m = np.stack([m for m, _, _ in outs])
    wsum = w.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(wsum > 0, (w * m).sum(0) / wsum, 0.0)
        sig = np.where(wsum > 0, 1 / np.sqrt(wsum), 1.0)
    return mean, sig, wsum == 0


def bin_file(path, nbin):
    with fits.open(path) as hdul:
        wave, band = _wavelengths(hdul)
        nb, idx = _blocks(len(wave), nbin)
        new_wave = wave[idx].mean(-1)
        new_band = band[idx].sum(-1) / 1.0
        out = [hdul[0].copy()]
        out += [h.copy() for h in hdul if h.name in ("OI_ARRAY", "OI_TARGET")]
        wl = fits.BinTableHDU.from_columns(
            [fits.Column(name="EFF_WAVE", format="E", array=new_wave.astype("f4")),
             fits.Column(name="EFF_BAND", format="E", array=new_band.astype("f4"))], name="OI_WAVELENGTH")
        wl.header["EXTVER"] = 1
        wl.header["INSNAME"] = SC
        out.append(wl)
        v2 = _pol_tables(hdul, "OI_VIS2")
        t3 = _pol_tables(hdul, "OI_T3")
        for tabs, kind in ((v2, "v2"), (t3, "t3")):
            base = tabs[0]
            data = base.data.copy()
            if kind == "v2":
                m, s, f = _combine(tabs, "VIS2DATA", _bin_real, idx, "VIS2ERR", "FLAG")
                cols = dict(VIS2DATA=m, VIS2ERR=s, FLAG=f)
            else:
                m, s, f = _combine(tabs, "T3PHI", _bin_phase, idx, "T3PHIERR", "FLAG")
                am, ae, _ = _combine(tabs, "T3AMP", _bin_real, idx, "T3AMPERR", "FLAG")
                cols = dict(T3PHI=m, T3PHIERR=s, T3AMP=am, T3AMPERR=ae, FLAG=f)
            newcols = []
            for c in base.columns:
                if c.name in cols:
                    arr = cols[c.name]
                    fmt = f"{nb}L" if c.name == "FLAG" else f"{nb}D"
                    newcols.append(fits.Column(name=c.name, format=fmt, unit=c.unit, array=arr))
                else:
                    newcols.append(fits.Column(name=c.name, format=c.format, unit=c.unit, array=data[c.name]))
            hdu = fits.BinTableHDU.from_columns(newcols, name=base.name)
            hdu.header.update({k: v for k, v in base.header.items() if k in ("OI_REVN", "DATE-OBS", "ARRNAME")})
            hdu.header["EXTVER"] = 1
            hdu.header["INSNAME"] = SC
            out.append(hdu)
        mjd = float(hdul[0].header["MJD-OBS"])
        return fits.HDUList(out), mjd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=pathlib.Path)
    ap.add_argument("--bin", type=int, default=8)
    ap.add_argument("--mode", default="HIGH")
    a = ap.parse_args(argv)
    for f in sorted((a.root / "oifits").glob("*.fits")):
        hdul, mjd = bin_file(f, a.bin)
        dest = a.root / "calibrated" / a.mode / night_of(mjd)
        dest.mkdir(parents=True, exist_ok=True)
        hdul.writeto(dest / f.name, overwrite=True)
        print(f"{f.name} -> {dest.relative_to(a.root)}/ ({hdul[3].data.shape if False else len(hdul)} HDUs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
