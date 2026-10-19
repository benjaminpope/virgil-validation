"""Closure-phase legs are matched to visibility rows (mutation-survivor tests).

Each closure-phase triangle leg is a baseline; virgil reuses the visibility
row of the same baseline, so a triangle adds no new (u, v) sample. Two corner
cases of that matching survived the 0.4.0 mutation run (M108, M109), so they
are pinned here against a file we write ourselves, with the baseline
coordinates computed from the station positions (x_b - x_a), not from virgil.

* Legs may sit under a different INSNAME than the visibilities when the two
  OI_WAVELENGTH tables hold identical wavelengths (some pipelines write one
  table per table type); the legs are then still matched.
* A visibility stored as baseline (2, 0) matches the triangle leg (0, 2).
"""

import warnings

import numpy as np
import pytest
from astropy.io import fits

from crosscheck import oifits_writer

pytest.importorskip("virgil.oifits")
from virgil.oidata import OIData  # noqa: E402

STATIONS = np.array([[0, 0, 0], [10, 0, 0], [3, 8, 0]], float)
WAVELENGTH = 2.2e-6


def _uv(a, b):
    return STATIONS[b][:2] - STATIONS[a][:2]


def _write(path, t3_insname=None, t3_wavelength=None, vis_pairs=((0, 1), (1, 2), (0, 2))):
    pairs = list(vis_pairs)
    u = np.array([_uv(*p)[0] for p in pairs])
    v = np.array([_uv(*p)[1] for p in pairs])
    n = len(pairs)
    vis2 = dict(vis2=np.full((n, 1), 0.9), err=np.full((n, 1), 0.01), u=u, v=v, sta=np.array(pairs))
    u1, v1 = _uv(0, 1)
    u2, v2 = _uv(1, 2)
    t3 = dict(phi=np.array([[5.0]]), err=np.array([[1.0]]), u1=[u1], v1=[v1], u2=[u2], v2=[v2],
              sta=np.array([[0, 1, 2]]))
    oifits_writer.write(path, wavelengths=[WAVELENGTH], stations_xyz=STATIONS, vis2=vis2, t3=t3)
    if t3_insname:
        with fits.open(path, mode="update") as hdus:
            table = hdus["OI_WAVELENGTH"].copy()
            table.header["INSNAME"] = t3_insname
            if t3_wavelength is not None:
                table.data["EFF_WAVE"] = t3_wavelength
            hdus.append(table)
            hdus["OI_T3"].header["INSNAME"] = t3_insname
            hdus.flush()


def _read(path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = OIData(str(path))
    return data, [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


def _baselines(data):
    return sorted(zip(np.round(np.asarray(data.u), 6), np.round(np.asarray(data.v), 6)))


EXPECTED = sorted((round(float(_uv(a, b)[0]), 6), round(float(_uv(a, b)[1]), 6)) for a, b in [(0, 1), (1, 2), (0, 2)])


@pytest.mark.validates("virgil.oifits.read_oifits", "virgil.oidata.OIData", roots=["standards"], kind="regression")  # M108
def test_legs_under_another_insname_with_equal_wavelengths_reuse_the_visibility_rows(tmp_path):
    _write(tmp_path / "a.fits", t3_insname="OTHER")
    data, warned = _read(tmp_path / "a.fits")
    assert _baselines(data) == EXPECTED  # three baselines, not three plus three duplicate legs
    assert not warned


@pytest.mark.validates("virgil.oifits.read_oifits", "virgil.oidata.OIData", roots=["standards"], kind="regression")  # M108 control
def test_legs_at_different_wavelengths_are_not_matched(tmp_path):
    _write(tmp_path / "a.fits", t3_insname="OTHER", t3_wavelength=[2.3e-6])
    data, _ = _read(tmp_path / "a.fits")
    assert len(np.asarray(data.u)) > 3  # legs at another wavelength are new samples


@pytest.mark.validates("virgil.oifits.read_oifits", "virgil.oidata.OIData", roots=["standards"], kind="regression")  # M109
def test_a_reversed_visibility_baseline_matches_its_triangle_leg_silently(tmp_path):
    _write(tmp_path / "a.fits", vis_pairs=((0, 1), (1, 2), (2, 0)))
    _, warned = _read(tmp_path / "a.fits")
    assert not warned


@pytest.mark.validates("virgil.oifits.read_oifits", "virgil.oidata.OIData", roots=["standards"], kind="regression")  # M109 control
def test_a_missing_visibility_baseline_is_reported_once(tmp_path):
    _write(tmp_path / "a.fits", vis_pairs=((0, 1), (1, 2)))
    _, warned = _read(tmp_path / "a.fits")
    assert len(warned) == 1 and "1 closure-phase triangle leg" in warned[0]


