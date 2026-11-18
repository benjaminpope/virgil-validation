"""crosscheck.oifits_writer, checked before it judges virgil: the file has the columns,
keywords and units that the OIFITS v2 standard (Duvert et al. 2017, Table 1-7) requires.
The file is written through crosscheck.simulate, the only caller of the writer: so this check
goes through simulate. tests/test_reference_oifits_writer_direct.py checks the writer called
directly, so that the writer does not rest on simulate, which depends on it."""

import numpy as np
import pytest
from astropy.io import fits

from crosscheck import simulate

UTS4 = np.array([[-9.925, -20.335, 0.0], [14.887, 30.502, 0.0], [44.915, 66.183, 0.0], [103.306, 43.999, 0.0]])

REQUIRED = {  # OIFITS v2 (Duvert et al. 2017): table -> (columns, keywords)
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


@pytest.mark.validates("crosscheck.oifits_writer", roots=["standards"], kind="reference")
def test_oifits_writer_meets_the_v2_standard(tmp_path):
    path = tmp_path / "s.fits"
    simulate.observe(path, lambda u, v, w: np.ones(np.broadcast(u, v, w).shape, complex), UTS4,
                     hour_angles_h=[0.0], wavelengths=[2.2e-6], sigma_v2=0.01, sigma_cp_deg=1.0)
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
    assert not missing, missing
