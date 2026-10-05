"""OIFITS v1 snapshots told apart by TIME, not MJD (finding F13, fixed in virgil#203).

OIFITS v1 (Pauls et al. 2005) gives each row a TIME (UTC seconds) and an
MJD. Some writers set MJD to the night's date only and put the snapshot in
TIME: the 2004 imaging-contest files (OYSTER) give 13 snapshots one MJD.
Closure phases are correlated only within a snapshot, through the
baselines they share at that instant. Grouping frames by MJD alone merges
all snapshots into one: the 2004 data2 file's 130 closure phases then whiten
to 10 independent combinations instead of 130, throwing most of them away.

Our simulator writes four VLTI UTs at seven hour angles with a distinct MJD
per snapshot. The same file with one MJD and the snapshots in TIME must give
the same number of independent observables.
"""

import numpy as np
import pytest
from astropy.io import fits

from crosscheck import simulate, sky

pytest.importorskip("virgil.oidata")
from virgil.oidata import OIData  # noqa: E402

pytestmark = pytest.mark.x64

UTS = np.array(
    [[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]]
)


def _binary(u, v, w):
    return (sky.vis_point(u, v, w) + 0.3 * sky.vis_point(u, v, w, 4.0, 1.0)) / 1.3


def _files(tmp_path):
    generated = tmp_path / "generated.fits"
    distinct = tmp_path / "distinct_mjd.fits"
    simulate.observe(
        generated, _binary, UTS, hour_angles_h=np.linspace(-3, 3, 7),
        wavelengths=np.array([1.65e-6]), dec_deg=-30.0, sigma_v2=0.01, sigma_cp_deg=1.0,
    )
    shared = tmp_path / "shared_mjd.fits"
    with fits.open(generated) as h:
        h[0].header["CONTENT"] = "OIFITS"
        for x in h:
            if "OI_REVN" in x.header:
                x.header["OI_REVN"] = 1
            if x.name == "OI_ARRAY":
                h[h.index_of(x.name)] = fits.BinTableHDU.from_columns(
                    [column for column in x.columns if column.name not in {"FOV", "FOVTYPE"}],
                    header=x.header,
                )
        array = h["OI_ARRAY"]
        assert h[0].header["CONTENT"] == "OIFITS"
        assert all(x.header["OI_REVN"] == 1 for x in h if "OI_REVN" in x.header)
        assert not {"FOV", "FOVTYPE"} & set(array.columns.names)
        h.writeto(distinct)
        for x in h:
            if x.name in ("OI_VIS2", "OI_VIS", "OI_T3"):
                mjd = np.array(x.data["MJD"], dtype=float)
                x.data["TIME"] = (mjd - np.floor(mjd.min())) * 86400.0
                x.data["MJD"] = np.floor(mjd.min())
        h.writeto(shared)
    return OIData(str(distinct)), OIData(str(shared))


@pytest.mark.validates("virgil.oidata.OIData", roots=["standards"])
def test_simulated_snapshots_have_distinct_mjd(tmp_path):
    distinct, _ = _files(tmp_path)
    n_cp = np.asarray(distinct.phi).size
    assert n_cp == 7 * 4  # four triangles per snapshot
    assert distinct.n_independent - np.asarray(distinct.vis).size == 7 * 3  # three independent per snapshot


@pytest.mark.validates("virgil.oidata.OIData", roots=["standards"])
def test_snapshots_in_time_column_are_separate_frames(tmp_path):
    distinct, shared = _files(tmp_path)
    assert shared.n_independent == distinct.n_independent
