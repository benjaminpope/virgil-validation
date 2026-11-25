"""Findings from the M87* replication (Lai et al. 2026) kept as strict xfails (trust/ledger.yml F33, P15)."""

import sys
import types

import pytest

from external_bridge import basis_bridge


class _FakeTable:
    """Just enough of ``casatools.table`` to reach read_ms's polarization check on a Stokes-I-only MS."""

    def __init__(self):
        self.path = None

    def open(self, path):
        self.path = path

    def close(self):
        pass

    done = close

    def nrows(self):
        return 1

    def getcol(self, name):
        return {"SPECTRAL_WINDOW_ID": [0], "POLARIZATION_ID": [0], "FIELD_ID": [0], "DATA_DESC_ID": [0]}[name]

    def getcell(self, name, row):
        return {"CHAN_FREQ": [2.3e11], "CORR_TYPE": [1]}[name]  # CORR_TYPE 1 = Stokes I (casacore Stokes enum)


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="F33: ComplexVisData.read_ms needs a parallel-hand pair; a Stokes-I-only MS is refused")
@pytest.mark.validates("virgil.radio.ComplexVisData.read_ms", roots=["mathematics"], kind="finding")
def test_f33_read_ms_accepts_a_stokes_i_measurement_set(monkeypatch):
    """F33: the EHT UVFITS give a Stokes-I-only MS after importuvfits; read_ms raises before reading any row."""
    from virgil.radio import ComplexVisData

    monkeypatch.setitem(sys.modules, "casatools", types.SimpleNamespace(table=_FakeTable))
    try:
        ComplexVisData.read_ms("stokes_i.ms")  # today: ValueError "no XX/YY or RR/LL polarization pair"
    except AttributeError as err:
        assert "query" in str(err), err  # got past the polarization check to the fake's missing query(): Stokes I accepted


@pytest.mark.skipif(not basis_bridge.available(), reason="needs .venv-basis or BASIS_PYTHON (basis.sif)")
@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="P15: BASIS sky_map renders the xsringauss floor Gaussian on the side opposite its visibilities")
@pytest.mark.validates("external_bridge.basis_worker", roots=["mathematics"], tier="B", kind="upstream")
def test_p15_basis_floor_gaussian_render_matches_its_visibilities():
    """P15: centroid of the rendered floor-only model against the centroid implied by BASIS's analytic phases."""
    chk = basis_bridge.render_check()
    ps = 225.0 / 127
    (rx, ry), (ax, ay) = chk["floor_centroid_render_east_north_uas"], chk["floor_centroid_analytic_east_north_uas"]
    assert abs(rx - ax) <= ps and abs(ry - ay) <= ps, chk
