"""A minimal OIFITS v2 writer (Duvert et al. 2017, A&A 597, A8), astropy only.

Tables: OI_TARGET, OI_ARRAY, OI_WAVELENGTH, OI_VIS2, OI_T3. Angles in
degrees. For a baseline (STA_INDEX = [s1, s2]) UCOORD, VCOORD are those of
x(s2) - x(s1); for a triangle (s1, s2, s3), (U1, V1) is s1->s2 and (U2, V2)
is s2->s3, and T3PHI = arg V(U1, V1) V(U2, V2) V(-U1-U2, -V1-V2).
"""

import numpy as np
from astropy.io import fits


def _col(name, fmt, data, unit=None):
    return fits.Column(name=name, format=fmt, array=data, unit=unit)


def write(
    path,
    *,
    wavelengths,
    stations_xyz,
    vis2,
    t3,
    target="SIM",
    insname="SIM",
    arrname="SIM",
    mjd=60000.0,
):
    """Write an OIFITS file.

    vis2 : dict of arrays per record (n_rec, n_wl): ``vis2``, ``err``,
        ``u``, ``v`` (n_rec,), ``sta`` (n_rec, 2), optional ``mjd``.
    t3 : dict: ``phi`` and ``err`` in degrees (n_rec, n_wl), ``u1``,
        ``v1``, ``u2``, ``v2`` (n_rec,), ``sta`` (n_rec, 3), optional ``mjd``;
        or None for a file of visibilities alone (no OI_T3 table).
    """
    from astropy.time import Time

    # OIFITS v2 requires DATE-OBS (UTC date of the first observation)
    date_obs = Time(float(np.min(np.atleast_1d(vis2.get("mjd", mjd)))), format="mjd").isot[:10]
    wavelengths = np.atleast_1d(np.asarray(wavelengths, float))
    n_wl = wavelengths.size
    n_sta = len(stations_xyz)

    primary = fits.PrimaryHDU()
    primary.header["CONTENT"] = "OIFITS2"

    target_hdu = fits.BinTableHDU.from_columns(
        [
            _col("TARGET_ID", "I", np.array([1])),
            _col("TARGET", "16A", np.array([target])),
            _col("RAEP0", "D", np.array([0.0]), "deg"),
            _col("DECEP0", "D", np.array([0.0]), "deg"),
            _col("EQUINOX", "E", np.array([2000.0]), "yr"),
            _col("RA_ERR", "D", np.array([0.0]), "deg"),
            _col("DEC_ERR", "D", np.array([0.0]), "deg"),
            _col("SYSVEL", "D", np.array([0.0]), "m/s"),
            _col("VELTYP", "8A", np.array(["LSR"])),
            _col("VELDEF", "8A", np.array(["OPTICAL"])),
            _col("PMRA", "D", np.array([0.0]), "deg/yr"),
            _col("PMDEC", "D", np.array([0.0]), "deg/yr"),
            _col("PMRA_ERR", "D", np.array([0.0]), "deg/yr"),
            _col("PMDEC_ERR", "D", np.array([0.0]), "deg/yr"),
            _col("PARALLAX", "E", np.array([0.0]), "deg"),
            _col("PARA_ERR", "E", np.array([0.0]), "deg"),
            _col("SPECTYP", "16A", np.array(["UNKNOWN"])),
        ],
        name="OI_TARGET",
    )
    target_hdu.header["OI_REVN"] = 2

    array_hdu = fits.BinTableHDU.from_columns(
        [
            _col("TEL_NAME", "16A", np.array([f"T{i}" for i in range(n_sta)])),
            _col("STA_NAME", "16A", np.array([f"S{i}" for i in range(n_sta)])),
            _col("STA_INDEX", "I", np.arange(1, n_sta + 1)),
            _col("DIAMETER", "E", np.full(n_sta, 1.0), "m"),
            _col("STAXYZ", "3D", np.asarray(stations_xyz, float), "m"),
            _col("FOV", "D", np.full(n_sta, 1.0), "arcsec"),
            _col("FOVTYPE", "6A", np.array(["FWHM"] * n_sta)),
        ],
        name="OI_ARRAY",
    )
    array_hdu.header.update(
        OI_REVN=2, ARRNAME=arrname, FRAME="GEOCENTRIC",
        ARRAYX=0.0, ARRAYY=0.0, ARRAYZ=0.0,
    )

    wl_hdu = fits.BinTableHDU.from_columns(
        [
            _col("EFF_WAVE", "E", wavelengths, "m"),
            _col("EFF_BAND", "E", np.full(n_wl, 1e-9), "m"),
        ],
        name="OI_WAVELENGTH",
    )
    wl_hdu.header.update(OI_REVN=2, INSNAME=insname)

    def common(rec, n):
        mjd_arr = np.asarray(rec.get("mjd", np.full(n, mjd)), float)
        return [
            _col("TARGET_ID", "I", np.ones(n, int)),
            _col("TIME", "D", np.zeros(n), "s"),
            _col("MJD", "D", mjd_arr, "day"),
            _col("INT_TIME", "D", np.ones(n), "s"),
        ]

    n = len(vis2["u"])
    fmt = f"{n_wl}D"
    v2_hdu = fits.BinTableHDU.from_columns(
        common(vis2, n)
        + [
            _col("VIS2DATA", fmt, np.asarray(vis2["vis2"]).reshape(n, n_wl)),
            _col("VIS2ERR", fmt, np.asarray(vis2["err"]).reshape(n, n_wl)),
            _col("UCOORD", "D", vis2["u"], "m"),
            _col("VCOORD", "D", vis2["v"], "m"),
            _col("STA_INDEX", "2I", np.asarray(vis2["sta"]) + 1),
            _col("FLAG", f"{n_wl}L", np.zeros((n, n_wl), bool)),
        ],
        name="OI_VIS2",
    )
    v2_hdu.header.update(OI_REVN=2, ARRNAME=arrname, INSNAME=insname)
    v2_hdu.header["DATE-OBS"] = date_obs  # a hyphen: not a Python keyword argument

    tables = [primary, target_hdu, array_hdu, wl_hdu, v2_hdu]
    if t3 is None:
        fits.HDUList(tables).writeto(path, overwrite=True)
        return

    n = len(t3["u1"])
    t3_hdu = fits.BinTableHDU.from_columns(
        common(t3, n)
        + [
            _col("T3AMP", fmt, np.ones((n, n_wl))),
            _col("T3AMPERR", fmt, np.zeros((n, n_wl))),
            _col("T3PHI", fmt, np.asarray(t3["phi"]).reshape(n, n_wl), "deg"),
            _col("T3PHIERR", fmt, np.asarray(t3["err"]).reshape(n, n_wl), "deg"),
            _col("U1COORD", "D", t3["u1"], "m"),
            _col("V1COORD", "D", t3["v1"], "m"),
            _col("U2COORD", "D", t3["u2"], "m"),
            _col("V2COORD", "D", t3["v2"], "m"),
            _col("STA_INDEX", "3I", np.asarray(t3["sta"]) + 1),
            _col("FLAG", f"{n_wl}L", np.zeros((n, n_wl), bool)),
        ],
        name="OI_T3",
    )
    t3_hdu.header.update(OI_REVN=2, ARRNAME=arrname, INSNAME=insname)
    t3_hdu.header["DATE-OBS"] = date_obs

    fits.HDUList(tables + [t3_hdu]).writeto(path, overwrite=True)
