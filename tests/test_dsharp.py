"""The DSHARP / HL Tau campaign code that needs no GPU and no download: target table, checksummed fetch (a local file:// server stands in
for ALMA), safe extraction, the beam-convolved comparison with a released image (analytic Gaussian), the start flux, and the scripts' CLIs."""

import importlib.util
import io
import json
import pathlib
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from crosscheck import alma_maxent as cx
from crosscheck import dsharp as ds

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.validates("crosscheck.dsharp", roots=["self-consistency"], kind="guard")
def test_campaign_covers_the_five_discs_and_both_hl_tau_bands():
    assert set(ds.CAMPAIGN) == {"imlup", "hltau_b6", "hltau_b7", "hd163296", "rulup", "as209"}
    for k in ds.CAMPAIGN:
        t = ds.target(k)
        assert t["ms"]["url"].startswith("https://almascience.eso.org/almadata/") and t["image"]["url"].startswith("https://almascience.eso.org/almadata/")
        assert t["ms"]["bytes"] > 1e8 and t["image"]["bytes"] > 1e6
    assert ds.target("imlup")["ms"]["url"].endswith("/MSfiles/IMLup_continuum.ms.tgz")
    assert ds.target("rulup")["image"]["url"].endswith("/images/RULup_continuum.fits")
    with pytest.raises(KeyError, match="unknown target"):
        ds.target("hd142666")


@pytest.mark.validates("crosscheck.dsharp", roots=["self-consistency"], kind="guard")
def test_hl_tau_uses_the_released_selfcal_recipe_and_dsharp_does_not():
    for k, tables in (("hltau_b6", ["pcal2", "apcal"]), ("hltau_b7", ["pcal2", "apcal"])):
        t = ds.target(k)
        assert t["gaintables"] == tables and t["spwmap"] == [[0, 0, 0, 0], [0, 0, 0, 0]] and not t["selfcal_released"]
        assert t["datacolumn"] == "DATA"  # the split of CORRECTED_DATA
    assert ds.target("hltau_b7")["flags_before"] == [dict(scan="271")]
    assert ds.target("hltau_b6")["image"]["pbcor"] is False and ds.target("hltau_b7")["image"]["pbcor"] is True
    for k in ("imlup", "hd163296", "rulup", "as209"):
        assert ds.target(k)["selfcal_released"] and ds.target(k)["datacolumn"] == "CORRECTED_DATA"
    assert ds.HL_TAU_HEADLINE == ("hltau_b6", "hltau_b7")


@pytest.mark.validates("crosscheck.dsharp", roots=["mathematics"], kind="reference")
def test_checksum_is_recorded_then_verified_then_refuses_a_changed_file(tmp_path):
    f, table = tmp_path / "a.bin", tmp_path / "sums.json"
    f.write_bytes(b"abc" * 100)
    assert ds.record_checksum("u", f, 300, table) == "recorded"
    assert json.loads(table.read_text())["u"]["sha256"] == ds.sha256_file(f)
    assert ds.record_checksum("u", f, 300, table) == "verified"
    f.write_bytes(b"abd" * 100)
    with pytest.raises(ds.ChecksumMismatch, match="differs"):
        ds.record_checksum("u", f, 300, table)
    with pytest.raises(ds.ChecksumMismatch, match="bytes"):
        ds.record_checksum("v", f, 299, table)


@pytest.mark.validates("crosscheck.dsharp", roots=["mathematics"], kind="reference")
def test_sha256_of_a_known_string(tmp_path):
    f = tmp_path / "x"
    f.write_bytes(b"abc")
    assert ds.sha256_file(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"  # FIPS 180-2 test vector


def _tgz(path, names):
    with tarfile.open(path, "w:gz") as tf:
        for n, data in names.items():
            ti = tarfile.TarInfo(n)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return path


@pytest.mark.validates("crosscheck.dsharp", roots=["standards"], kind="guard")
def test_safe_extract_keeps_requested_members_and_refuses_escapes(tmp_path):
    good = _tgz(tmp_path / "g.tgz", {"keep/a.txt": b"1", "skip/b.txt": b"2"})
    ds.safe_extract(good, tmp_path / "out", members=["keep"])
    assert (tmp_path / "out/keep/a.txt").exists() and not (tmp_path / "out/skip").exists()
    bad = _tgz(tmp_path / "b.tgz", {"../evil.txt": b"x"})
    with pytest.raises(ValueError, match="leave"):
        ds.safe_extract(bad, tmp_path / "out2")
    assert not (tmp_path / "evil.txt").exists()
    link = tmp_path / "l.tgz"
    with tarfile.open(link, "w:gz") as tf:
        ti = tarfile.TarInfo("ln")
        ti.type, ti.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tf.addfile(ti)
    with pytest.raises(ValueError, match="regular file"):
        ds.safe_extract(link, tmp_path / "out3")


@pytest.mark.validates("crosscheck.dsharp", roots=["standards"], kind="guard")
def test_fetch_image_downloads_checks_size_extracts_and_records(tmp_path, monkeypatch):
    (tmp_path / "srv").mkdir()
    tgz = _tgz(tmp_path / "srv" / "ref.tgz", {"ref.image.fits": b"FITS" * 10, "other": b"z"})
    size = tgz.stat().st_size
    t = dict(ds.TARGETS["hltau_b6"], image=dict(url=tgz.as_uri(), bytes=size, etag="", member="ref.image.fits", pbcor=False))
    monkeypatch.setitem(ds.TARGETS, "hltau_b6", t)
    sums = tmp_path / "sums.json"
    path, status = ds.fetch_image("hltau_b6", tmp_path / "data", sums)
    assert status == "recorded" and path.read_bytes() == b"FITS" * 10 and not (path.parent / "ref.tgz").exists()
    assert ds.fetch_image("hltau_b6", tmp_path / "data", sums)[1] == "verified"
    t["image"]["bytes"] = size + 1
    with pytest.raises(ds.ChecksumMismatch):
        ds.fetch_image("hltau_b6", tmp_path / "data", sums)
    assert not list((tmp_path / "data").rglob("*.part"))


@pytest.mark.validates("crosscheck.dsharp", roots=["mathematics"], kind="reference")
def test_auto_flux_is_the_weighted_short_baseline_mean():
    rng = np.random.default_rng(0)
    q = rng.uniform(1e3, 1e6, 5000)
    ph = rng.uniform(0, 2 * np.pi, 5000)
    u, v = q * np.cos(ph), q * np.sin(ph)
    vis = 0.7 * np.exp(-((q / 3e5) ** 2)) + 0j
    assert ds.auto_flux(u, v, vis, np.ones_like(q)) == pytest.approx(0.7, rel=2e-3)


def _gauss_fits(path, flux, sigma_mas, px, npix, beam=None, bunit="Jy/arcsec2", restfrq=None):
    """A circular Gaussian of total ``flux`` (Jy) in Jy/arcsec2 (or Jy/beam with ``beam`` = FWHM in mas), written with the repository's writer."""
    c = npix // 2 - 1  # the phase-centre cell of the repository's layout
    yy, xx = np.indices((npix, npix))
    r2 = ((xx - c) ** 2 + (yy - c) ** 2) * px**2
    dens = flux / (2 * np.pi * (sigma_mas / 1e3) ** 2) * np.exp(-r2 / (2 * sigma_mas**2))  # Jy/arcsec2
    hdr = {}
    if beam is not None:
        dens = dens * ds.beam_area_arcsec2(beam, beam)
        hdr = dict(BMAJ=beam / 3.6e6, BMIN=beam / 3.6e6, BPA=0.0)
    if restfrq is not None:
        hdr["RESTFRQ"] = restfrq
    cx.write_fits(path, dens, px, 100.0, -30.0, bunit=bunit if beam is None else "Jy/beam", **hdr)


def _ellipse_ref_fits(path, flux, sigma_src, beam, px, npix, mirror=False):
    """The analytic answer for a circular Gaussian source (sigma ``sigma_src`` mas) seen with an elliptical beam ``(bmaj, bmin, bpa)``: a Gaussian
    of covariance sigma_src^2 I + sigma_maj^2 u u^T + sigma_min^2 v v^T in (East, North), u the unit vector at position angle bpa East of North.
    Written by hand in FITS orientation (East to the left: x_East = -(column - c) px; North up the rows), so the beam convention of the comparison is
    checked independently. ``mirror`` flips the position angle."""
    bmaj, bmin, bpa = beam
    th = np.radians(-bpa if mirror else bpa)
    u, v = np.array([np.sin(th), np.cos(th)]), np.array([np.cos(th), -np.sin(th)])
    f = 2 * np.sqrt(2 * np.log(2))
    cov = sigma_src**2 * np.eye(2) + (bmaj / f) ** 2 * np.outer(u, u) + (bmin / f) ** 2 * np.outer(v, v)  # mas^2
    c = npix // 2 - 1
    rows, cols = np.indices((npix, npix))
    east, north = -(cols - c) * px, (c - rows) * px  # the layout has row 0 North
    q = np.stack([east.ravel(), north.ravel()], axis=1)
    quad = np.einsum("ni,ij,nj->n", q, np.linalg.inv(cov), q).reshape(npix, npix)
    dens = flux / (2 * np.pi * np.sqrt(np.linalg.det(cov)) / 1e6) * np.exp(-0.5 * quad)  # Jy/arcsec2
    cx.write_fits(path, dens * ds.beam_area_arcsec2(bmaj, bmin), px, 100.0, -30.0, bunit="Jy/beam", BMAJ=bmaj / 3.6e6, BMIN=bmin / 3.6e6, BPA=float(bpa))


@pytest.mark.validates("crosscheck.dsharp", roots=["mathematics"], kind="reference")
def test_our_image_in_the_released_beam_matches_the_analytic_gaussian(tmp_path):
    """A Gaussian source of FWHM-sigma 30 mas seen with a 40 mas circular beam is a Gaussian of sigma sqrt(30^2 + (40/2.355)^2) mas;
    the comparison's convolution, unit change (Jy/arcsec2 to Jy/beam) and regridding onto a different pixel grid reproduce it."""
    flux, s, bm = 0.5, 30.0, 40.0
    _gauss_fits(tmp_path / "ours.fits", flux, s, 5.0, 128)
    _gauss_fits(tmp_path / "ref.fits", flux, np.hypot(s, bm / 2.3548200450309493), 4.0, 160, beam=bm)  # the analytic answer, on another grid
    ours, ref, wcs, beam = ds.our_image_in_beam(tmp_path / "ours.fits", tmp_path / "ref.fits")
    assert beam == pytest.approx((bm, bm, 0.0), rel=1e-6)
    assert ours.max() == pytest.approx(ref.max(), rel=2e-3)
    st = ds.compare_stats(ours, ref, radius_pix=40)
    assert st["correlation"] > 0.999 and st["flux_ratio"] == pytest.approx(1.0, abs=2e-3) and st["rms_over_peak"] < 6e-3  # bilinear regridding of 5 mas pixels


@pytest.mark.validates("crosscheck.dsharp", roots=["mathematics"], kind="control")
def test_a_shifted_image_is_seen_as_different(tmp_path):
    """Control: the matched comparison passes, and only our source being 40 mas off the released one (the WCS) makes it fail."""
    flux, s, bm = 0.5, 30.0, 40.0
    _gauss_fits(tmp_path / "ours.fits", flux, s, 5.0, 128)
    _gauss_fits(tmp_path / "ref.fits", flux, np.hypot(s, bm / 2.3548200450309493), 5.0, 128, beam=bm)
    ours, ref, *_ = ds.our_image_in_beam(tmp_path / "ours.fits", tmp_path / "ref.fits")
    assert ds.compare_stats(ours, ref, radius_pix=40)["rms_over_peak"] < 6e-3
    from astropy.io import fits

    with fits.open(tmp_path / "ours.fits", mode="update") as h:
        h[0].header["CRPIX1"] += 8  # 8 pixels of 5 mas
    ours, ref, *_ = ds.our_image_in_beam(tmp_path / "ours.fits", tmp_path / "ref.fits")
    assert ds.compare_stats(ours, ref, radius_pix=40)["rms_over_peak"] > 0.1


@pytest.mark.validates("crosscheck.dsharp", roots=["mathematics"], kind="reference")
def test_an_elliptical_beam_at_a_position_angle_matches_the_hand_written_ellipse(tmp_path):
    """A 60 x 30 mas beam at PA 30 deg East of North: the analytic ellipse written by hand in FITS orientation is reproduced (row flip, beam
    angle convention and regridding); the same ellipse at PA -30 deg (the mirror) is not."""
    flux, s, beam = 0.5, 30.0, (60.0, 30.0, 30.0)
    _gauss_fits(tmp_path / "ours.fits", flux, s, 5.0, 128)
    _ellipse_ref_fits(tmp_path / "ref.fits", flux, s, beam, 5.0, 128)
    ours, ref, _, b = ds.our_image_in_beam(tmp_path / "ours.fits", tmp_path / "ref.fits")
    assert b == pytest.approx(beam, rel=1e-6)
    st = ds.compare_stats(ours, ref, radius_pix=40)
    assert st["rms_over_peak"] < 6e-3 and st["correlation"] > 0.999 and st["peak_ratio"] == pytest.approx(1.0, abs=2e-3)
    _ellipse_ref_fits(tmp_path / "mirror.fits", flux, s, beam, 5.0, 128, mirror=True)
    assert ds.compare_stats(ours, _read(tmp_path / "mirror.fits"), radius_pix=40)["rms_over_peak"] > 1.5e-2


def _read(path):
    return cx.read_fits(path)[0]


@pytest.mark.validates("crosscheck.dsharp", roots=["mathematics"], kind="guard")
def test_statistics_use_only_our_footprint_and_finite_pixels(tmp_path):
    """The released image is larger than ours: pixels outside our image hold zeros, not a model, and are left out; NaNs (outside a primary-beam
    limit) are skipped, not propagated."""
    flux, s, bm = 0.5, 30.0, 40.0
    _gauss_fits(tmp_path / "ours.fits", flux, s, 5.0, 64)  # 320 mas across
    _gauss_fits(tmp_path / "ref.fits", flux, np.hypot(s, bm / 2.3548200450309493), 5.0, 128, beam=bm)  # 640 mas across
    ours, ref, *_ = ds.our_image_in_beam(tmp_path / "ours.fits", tmp_path / "ref.fits")
    foot = ds.our_footprint(tmp_path / "ours.fits", tmp_path / "ref.fits")
    assert 0.2 < foot.mean() < 0.3 and foot[64, 64] and not foot[0, 0]
    whole = ds.compare_stats(ours, ref, footprint=foot)
    assert whole["footprint_radius_pix"] < 0.6 * 64 * np.sqrt(2)  # the half-size image's corners, not the released field's
    assert whole["ours_noise_rms"] > 0 or whole["ref_noise_rms"] >= 0
    ref_nan = ref.copy()
    ref_nan[:10] = np.nan
    st = ds.compare_stats(ours, ref_nan, footprint=foot)
    assert all(np.isfinite(v) for v in st.values())


@pytest.mark.validates("crosscheck.dsharp", roots=["mathematics"], kind="reference")
def test_primary_beam_correction_is_undone_for_a_pbcor_reference(tmp_path):
    """A primary-beam-corrected reference is multiplied by the ALMA 12 m beam (a Gaussian of FWHM 1.13 lambda/D) before the comparison. The beam
    is 1 at the pointing and 1/2 at FWHM/2 from it, and a 0.6 arcsec field (about 17 arcsec FWHM) is multiplied by almost 1."""
    flux, s, bm = 0.5, 30.0, 40.0
    _gauss_fits(tmp_path / "ours.fits", flux, s, 5.0, 128)
    _gauss_fits(tmp_path / "ref.fits", flux, np.hypot(s, bm / 2.3548200450309493), 5.0, 128, beam=bm, restfrq=344e9)
    ours, ref, wcs, _ = ds.our_image_in_beam(tmp_path / "ours.fits", tmp_path / "ref.fits", pbcor=True)
    raw = _read(tmp_path / "ref.fits")
    assert np.all(ref <= raw * (1 + 1e-12)) and ref.max() / raw.max() > 0.9999
    fwhm = np.degrees(1.13 * 2.99792458e8 / 344e9 / 12.0) * 3600  # arcsec, about 17
    ra0, dec0 = (float(x) for x in wcs.all_pix2world(0, 0, 0))
    assert ds.primary_beam(wcs, (1, 1), (ra0, dec0), 344e9)[0, 0] == pytest.approx(1.0)
    assert ds.primary_beam(wcs, (1, 1), (ra0, dec0 + fwhm / 2 / 3600), 344e9)[0, 0] == pytest.approx(0.5, rel=1e-6)
    _gauss_fits(tmp_path / "nofreq.fits", flux, 40.0, 5.0, 128, beam=bm)
    with pytest.raises(ValueError, match="RESTFRQ"):
        ds.our_image_in_beam(tmp_path / "ours.fits", tmp_path / "nofreq.fits", pbcor=True)


@pytest.mark.validates("crosscheck.dsharp", roots=["literature"], kind="guard")
def test_a_released_image_without_a_beam_is_refused(tmp_path):
    _gauss_fits(tmp_path / "nobeam.fits", 0.1, 30.0, 5.0, 64)
    with pytest.raises(ValueError, match="BMAJ"):
        ds.alma_image(tmp_path / "nobeam.fits")


@pytest.mark.validates("crosscheck.dsharp", roots=["mathematics"], kind="reference")
def test_compare_and_hl_tau_preview_write_their_files(tmp_path):
    pytest.importorskip("matplotlib")
    comp = _script("dsharp_compare")
    for b in ("b6", "b7"):
        (tmp_path / b).mkdir()
        _gauss_fits(tmp_path / b / "image.fits", 0.4, 40.0, 5.0, 128)
        (tmp_path / b / "result.json").write_text(json.dumps(dict(flux_jy=0.4, chi2n_raw=1.2, converged=True, n=10)))
        _gauss_fits(tmp_path / f"ref_{b}.fits", 0.4, 45.0, 5.0, 128, beam=35.0, restfrq=344e9)
    comp.main(["compare", "--target", "as209", "--fit", str(tmp_path / "b6"), "--ref", str(tmp_path / "ref_b6.fits"), "--out", str(tmp_path / "c")])
    j = json.loads((tmp_path / "c/compare.json").read_text())
    assert (tmp_path / "c/compare.png").stat().st_size > 1000 and j["central"]["correlation"] > 0.9 and j["fit"]["converged"]
    comp.main(["preview", "--out", str(tmp_path / "p/hltau.png"), "--b6", str(tmp_path / "b6"), str(tmp_path / "ref_b6.fits"),
               "--b7", str(tmp_path / "b7"), str(tmp_path / "ref_b7.fits")])
    assert (tmp_path / "p/hltau.png").stat().st_size > 1000
    assert set(json.loads((tmp_path / "p/hltau.json").read_text())) == {"hltau_b6", "hltau_b7"}


@pytest.mark.validates("crosscheck.dsharp", roots=["mathematics"], kind="reference")
def test_the_fit_driver_gauges_on_the_heaviest_observation():
    fit = _script("dsharp_fit")
    obs = np.array([0, 0, 1, 1, 1, 2])
    w = np.array([1.0, 1.0, 0.5, 0.5, 0.5, 5.0])
    assert fit.heaviest_obs(obs, w) == 2
    assert fit.jsonable({"a": np.arange(2), "b": np.float64(1.5)}) == {"a": [0, 1], "b": 1.5}


@pytest.mark.validates("crosscheck.dsharp", roots=["mathematics"], kind="guard")
@pytest.mark.parametrize("name", ["dsharp_fetch", "dsharp_compare", "dsharp_fit", "dsharp_ms_info", "dsharp_hltau_applycal"])
def test_help_renders_without_casa_or_a_gpu(name):
    r = subprocess.run([sys.executable, str(SCRIPTS / f"{name}.py"), "--help"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "usage" in r.stdout.lower()
