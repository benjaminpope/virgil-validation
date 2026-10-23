"""The cold-start floor and the dead-pixel diagnostic of the IM Lup MaxEnt fit (R ob100, job 18968246_0, ended with 64 % of its pixels dead).
Synthetic data on the CPU, seconds."""

import json
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("virgil.radio")

import virgil_bridge.alma as va  # noqa: E402
from evidence.plugin import record  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import alma_imlup_maxent as script  # noqa: E402
from test_alma_maxent import synthetic_dir  # noqa: E402


def _call(*argv):
    script.main([str(a) for a in argv])


class _Prob:
    init = {}

    def __init__(self, npix, pixel_mas):
        self.npix, self.pixel_mas = npix, pixel_mas


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
@pytest.mark.parametrize("f_config", [False, True])
def test_floor_start_raises_every_pixel_and_reads_back_through_image_of(f_config):
    """The 1" Gaussian start has pixels at 1e-8 of the peak and below; floored, every pixel is at least the floor and ``image_of`` returns
    exactly the floored image (in config F as well, where ``cal.flux`` is the apparent flux)."""
    n, px, floor = 512, 10.0, 1e-4  # a 5.1 arcsec field, as IM Lup's 5.1 arcsec at 1028 x 5 mas
    v = va.start_values(_Prob(n, px), flux=0.25, fwhm_mas=1000.0)
    log_pb = va.beam_log(n, px) if f_config else None
    before = va.image_of(v, n, px, log_pb)
    assert before.min() < 1e-6 * before.max()
    eta, flux, info = va.floor_start(v, n, px, floor, log_pb=log_pb)
    after = va.image_of({"log_brightness": eta, "cal.flux": flux}, n, px, log_pb)
    assert after.min() >= floor * (1 - 1e-12)
    assert np.allclose(after, np.maximum(before, floor), rtol=1e-10, atol=0)
    assert info["n_raised"] == int((before < floor).sum()) and info["flux_added_jy"] > 0
    record("flux_added_jy" + ("_F" if f_config else "_R"), info["flux_added_jy"])


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_dead_pixel_report_flags_the_bare_gaussian_and_passes_the_floored_start():
    n, px = 512, 10.0
    v = va.start_values(_Prob(n, px), flux=0.25, fwhm_mas=1000.0)
    bare = va.dead_pixel_report(va.image_of(v, n, px), px)
    eta, flux, _ = va.floor_start(v, n, px, 1e-4)
    live = va.dead_pixel_report(va.image_of({"log_brightness": eta, "cal.flux": flux}, n, px), px)
    assert bare["warning"] and bare["frac_below_1e-6_peak"] > 0.3
    assert live["warning"] is None and live["frac_below_1e-6_peak"] == 0.0 and live["frac_below_1e-3_floor"] == 0.0
    assert live["largest_live_radius_arcsec"] > bare["largest_live_radius_arcsec"]
    record("dead_fraction_bare_gaussian", bare["frac_below_1e-6_peak"])


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
@pytest.mark.parametrize("floor", [None, 0.0])
def test_fit_p3_records_the_floor_start_by_default_and_the_dead_pixels(tmp_path, floor):
    """``fit`` raises the p3 start to the entropy floor by default (recorded in result.json and the first stage) and reports the dead pixels;
    ``--start-floor 0`` is the bare Gaussian and records no floor."""
    syn = synthetic_dir(tmp_path / "s")
    extra = [] if floor is None else ["--start-floor", floor]
    out = tmp_path / "o"
    _call("fit", "--config", "R", "--data", syn, "--out", out, "--npix", 128, "--max-steps", 3, "--no-dirty", *extra)
    r = json.loads((out / "result.json").read_text())
    if floor is None:
        assert r["args"]["start_floor"] == va.MPOL["prior"]
        assert r["start_floor"]["floor_jy_arcsec2"] == va.MPOL["prior"] and r["stages"][0]["start_floor"] == r["start_floor"]
        assert r["dead_pixels"]["warning"] is None and r["dead_pixels"]["frac_below_1e-6_peak"] == 0.0
    else:
        assert r["start_floor"] is None and "start_floor" not in r["stages"][0]  # (a 0.64 arcsec test field is inside the Gaussian: nothing dead; the report is tested above)
    assert set(r["dead_pixels"]) >= {"frac_below_1e-6_peak", "frac_below_1e-3_floor", "largest_live_radius_arcsec", "median_beyond_2p4_arcsec_over_peak"}
