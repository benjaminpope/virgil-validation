"""crosscheck.oifits_writer called directly, on hand-written arrays: the file has the columns,
keywords and units that the OIFITS v2 standard (Duvert et al. 2017, Table 1-7) requires, and
carries the numbers back that were put in. Imports no other crosscheck module, so this check does
not go through crosscheck.simulate (which depends on the writer); the simulate -> writer path is
checked in tests/test_reference_oifits_writer.py."""

import numpy as np
import pytest
from astropy.io import fits

from crosscheck import oifits_writer

REQUIRED = {  # OIFITS v2 (Duvert et al. 2017): table -> (columns, keywords); as in test_reference_oifits_writer.py
    "OI_TARGET": ({"TARGET_ID", "TARGET", "RAEP0", "DECEP0", "EQUINOX"}, {"OI_REVN"}),
    "OI_ARRAY": ({"TEL_NAME", "STA_NAME", "STA_INDEX", "DIAMETER", "STAXYZ"}, {"OI_REVN", "ARRNAME", "FRAME"}),
    "OI_WAVELENGTH": ({"EFF_WAVE", "EFF_BAND"}, {"OI_REVN", "INSNAME"}),
    "OI_VIS2": ({"TARGET_ID", "TIME", "MJD", "INT_TIME", "VIS2DATA", "VIS2ERR", "UCOORD", "VCOORD", "STA_INDEX", "FLAG"},
                {"OI_REVN", "DATE-OBS", "ARRNAME", "INSNAME"}),
    "OI_T3": ({"TARGET_ID", "TIME", "MJD", "INT_TIME", "T3AMP", "T3AMPERR", "T3PHI", "T3PHIERR", "U1COORD", "V1COORD",
               "U2COORD", "V2COORD", "STA_INDEX", "FLAG"}, {"OI_REVN", "DATE-OBS", "ARRNAME", "INSNAME"}),
}
UNITS = {("OI_WAVELENGTH", "EFF_WAVE"): "m", ("OI_VIS2", "UCOORD"): "m", ("OI_VIS2", "VCOORD"): "m",
         ("OI_T3", "T3PHI"): "deg", ("OI_T3", "T3PHIERR"): "deg", ("OI_T3", "U1COORD"): "m"}

STATIONS = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 20.0, 0.0]])  # three stations, metres
WAVELENGTHS = [2.0e-6, 2.2e-6]
VIS2 = {  # three baselines (0-based pairs), two channels
    "vis2": np.array([[0.9, 0.8], [0.7, 0.6], [0.5, 0.4]]),
    "err": np.array([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]]),
    "u": np.array([10.0, 0.0, -10.0]),
    "v": np.array([0.0, 20.0, 20.0]),
    "sta": np.array([[0, 1], [0, 2], [1, 2]]),
}
T3 = {  # the one triangle (0, 1, 2): (U1, V1) is 0->1, (U2, V2) is 1->2
    "phi": np.array([[5.0, -5.0]]),
    "err": np.array([[1.0, 2.0]]),
    "u1": np.array([10.0]), "v1": np.array([0.0]),
    "u2": np.array([-10.0]), "v2": np.array([20.0]),
    "sta": np.array([[0, 1, 2]]),
}


@pytest.mark.validates("crosscheck.oifits_writer", roots=["standards"], kind="reference")
def test_oifits_writer_meets_the_v2_standard_on_hand_written_arrays(tmp_path):
    path = tmp_path / "direct.fits"
    oifits_writer.write(path, wavelengths=WAVELENGTHS, stations_xyz=STATIONS, vis2=VIS2, t3=T3)
    missing = []
    with fits.open(path) as h:
        assert h[0].header.get("CONTENT") == "OIFITS2"
        for table, (cols, keys) in REQUIRED.items():
            names = set(h[table].columns.names)
            missing += [f"{table}.{c}" for c in cols - names]
            missing += [f"{table}:{k}" for k in keys if k not in h[table].header]
            if "OI_REVN" in h[table].header:
                assert int(h[table].header["OI_REVN"]) == 2, table
        for (table, col), unit in UNITS.items():
            assert (h[table].columns[col].unit or "").strip() == unit, (table, col)
        # the numbers come back as written, with 1-based station indices
        assert np.allclose(h["OI_VIS2"].data["VIS2DATA"], VIS2["vis2"])
        assert np.allclose(h["OI_VIS2"].data["VIS2ERR"], VIS2["err"])
        assert np.array_equal(h["OI_VIS2"].data["STA_INDEX"], VIS2["sta"] + 1)
        assert np.allclose(h["OI_T3"].data["T3PHI"], T3["phi"])
        assert np.array_equal(h["OI_T3"].data["STA_INDEX"], T3["sta"] + 1)
        assert np.allclose(h["OI_WAVELENGTH"].data["EFF_WAVE"], WAVELENGTHS)
    assert not missing, missing
