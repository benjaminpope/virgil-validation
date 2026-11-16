"""The DSHARP / HL Tau campaign: targets, public-data provenance and the comparison with the released ALMA images.

Nothing here imports virgil (the independent side). The campaign (design: ``reports/dsharp-plan.md`` of virgil-paper;
Ben's rulings of 2026-11-17) fits five discs, IM Lup, HL Tau, HD 163296, RU Lup and AS 209, with virgil's MaxEnt model and
compares each with the *public* ALMA image: the DSHARP fiducial continuum image (Andrews et al. 2018, ApJ 869, L41) or the
ALMA 2014 Long Baseline Campaign science-verification image of HL Tau (ALMA Partnership 2015, ApJ 808, L3). HL Tau is run in
Band 6 and Band 7 (``hltau_b6``, ``hltau_b7``) so that Ben can choose between them.

Sizes and ETags below were read from the servers' HTTP headers on 2026-11-17. A SHA-256 is recorded the first time a file is
fetched (``record_checksum``; ``data/dsharp/checksums.json``) and checked on every later fetch: the archive publishes none,
so the first fetch is trusted on first use, and the table says so (``DATA_ATTRIBUTION.md``).
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import tarfile
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
CHECKSUMS = ROOT / "data" / "dsharp" / "checksums.json"
DSHARP_BASE = "https://almascience.eso.org/almadata/lp/DSHARP"
SV_BASE = "https://almascience.eso.org/almadata/sciver"

CAMPAIGN = ("imlup", "hltau_b6", "hltau_b7", "hd163296", "rulup", "as209")
HL_TAU_HEADLINE = ("hltau_b6", "hltau_b7")  # one disc, two bands: Ben picks one from the previews


def _dsharp(key, name, ms_bytes, img_bytes, img_etag, codes):
    return dict(
        key=key, name=name, kind="dsharp", band=6, reference="Andrews et al. 2018, ApJ 869, L41 (DSHARP I)", programmes=codes,
        ms=dict(url=f"{DSHARP_BASE}/MSfiles/{key_file(name)}_continuum.ms.tgz", bytes=ms_bytes, member=f"{key_file(name)}_continuum.ms"),
        image=dict(url=f"{DSHARP_BASE}/images/{key_file(name)}_continuum.fits", bytes=img_bytes, etag=img_etag, member=None, pbcor=False),
        npix=1028, pixel_mas=5.0, selfcal_released=True, datacolumn="CORRECTED_DATA",
    )


def key_file(name):
    """The file stem the DSHARP release uses: the name without spaces (``IM Lup`` gives ``IMLup``)."""
    return name.replace(" ", "")


TARGETS = {
    "imlup": _dsharp("imlup", "IM Lup", 1908078667, 64010880, "3d0ba80-57a3e3d350840",
                     ["2013.1.00226.S", "2013.1.00694.S", "2013.1.00798.S", "2016.1.00484.L"]),
    "hd163296": _dsharp("hd163296", "HD 163296", 721682410, 36008640, "22572c0-57aa3e0d70880",
                        ["2013.1.00366.S", "2013.1.00601.S", "2016.1.00484.L"]),
    "rulup": _dsharp("rulup", "RU Lup", 986669197, 36008640, "22572c0-57a3eca3b53c0", ["2016.1.00484.L"]),
    "as209": _dsharp("as209", "AS 209", 944595260, 36008640, "22572c0-57a8c6770a940",
                     ["2013.1.00226.S", "2015.1.00486.S", "2016.1.00484.L"]),
}
# HL Tau, ALMA 2014 Long Baseline Campaign science verification (project 2011.0.00015.SV; CASA 4.2.2 products). The calibrated data are
# the channel-averaged measurement set *without* self-calibration, plus the self-calibration tables in ``calibration/``. The reference
# images are the released CLEAN images (Band 6: ``.image.fits``; Band 7: ``.image.pbcor.fits``).
TARGETS["hltau_b6"] = dict(
    key="hltau_b6", name="HL Tau (Band 6, 233 GHz)", kind="hltau", band=6, reference="ALMA Partnership et al. 2015, ApJ 808, L3",
    programmes=["2011.0.00015.SV"],
    ms=dict(url=f"{SV_BASE}/HLTauBand6/HLTau_Band6_CalibratedData.tgz", bytes=1592853462, member="HLTau_B6cont.calavg", calibration="calibration"),
    image=dict(url=f"{SV_BASE}/HLTauBand6/HLTau_Band6_ReferenceImages.tgz", bytes=9511237, etag="912145-50ed9a21c2980",
               member="HLTau_B6cont_mscale_ap.image.fits", pbcor=False),
    npix=1028, pixel_mas=5.0, selfcal_released=False, datacolumn="DATA",
    # imaging script HLTau_Band6_Cont_Imaging_5.1.py (casaguides.nrao.edu/images/a/ad/): the flags it sets before the final applycal,
    # then applycal(gaintable=['pcal2','apcal'], spwmap=[[0,0,0,0],[0,0,0,0]], calwt=False), then one more flag
    flags_before=[dict(antenna="DV01", spw="2"), dict(antenna="DA64&DV14", scan="275,315")], flags_after=[dict(scan="263")],
    gaintables=["pcal2", "apcal"], spwmap=[[0, 0, 0, 0], [0, 0, 0, 0]],
)
TARGETS["hltau_b7"] = dict(
    key="hltau_b7", name="HL Tau (Band 7, 344 GHz)", kind="hltau", band=7, reference="ALMA Partnership et al. 2015, ApJ 808, L3",
    programmes=["2011.0.00015.SV"],
    ms=dict(url=f"{SV_BASE}/HLTauBand7/HLTau_Band7_CalibratedData.tgz", bytes=1863049392, member="HLTau_B7cont.calavg", calibration="calibration"),
    image=dict(url=f"{SV_BASE}/HLTauBand7/HLTau_Band7_ReferenceImages.tgz", bytes=15609040, etag="ee2cd0-50ed9a2c40240",
               member="HLTau_B7cont_mscale_ap.image.pbcor.fits", pbcor=True),
    npix=1028, pixel_mas=5.0, selfcal_released=False, datacolumn="DATA",
    # HLTau_Band7_Cont_Imaging_5.1.py (casaguides.nrao.edu/images/2/21/): one flag, then the same two tables
    flags_before=[dict(scan="271")], flags_after=[],
    gaintables=["pcal2", "apcal"], spwmap=[[0, 0, 0, 0], [0, 0, 0, 0]],
)


def target(key):
    try:
        return TARGETS[key]
    except KeyError:
        raise KeyError(f"unknown target {key!r}; the campaign is {', '.join(CAMPAIGN)}") from None


# ------------------------------------------------------------------ provenance


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def read_checksums(path=CHECKSUMS):
    path = pathlib.Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


class ChecksumMismatch(RuntimeError):
    pass


def record_checksum(url, file, expect_bytes=None, path=CHECKSUMS):
    """Check ``file`` against the recorded SHA-256 of ``url`` (and the expected size), or record it on the first fetch.

    Returns ``"recorded"`` or ``"verified"``; raises ``ChecksumMismatch`` when the size or the hash differs from what was recorded.
    """
    file, path = pathlib.Path(file), pathlib.Path(path)
    size, digest = file.stat().st_size, sha256_file(file)
    if expect_bytes is not None and size != expect_bytes:
        raise ChecksumMismatch(f"{file.name}: {size} bytes, the server's Content-Length on 2026-11-17 was {expect_bytes}")
    table = read_checksums(path)
    if url in table:
        if table[url]["sha256"] != digest:
            raise ChecksumMismatch(f"{file.name}: sha256 {digest} differs from the recorded {table[url]['sha256']}")
        return "verified"
    table[url] = dict(sha256=digest, bytes=size, file=file.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    return "recorded"


def download(url, dest, expect_bytes=None, chunk=1 << 20):
    """Stream ``url`` to ``dest`` through a ``.part`` file (no resume), refusing a wrong size. Returns ``dest``."""
    dest = pathlib.Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as r, open(part, "wb") as f:  # noqa: S310  (the URLs are the constants above, or file:// in tests)
        while block := r.read(chunk):
            f.write(block)
    if expect_bytes is not None and part.stat().st_size != expect_bytes:
        got = part.stat().st_size
        part.unlink()
        raise ChecksumMismatch(f"{url}: {got} bytes, expected {expect_bytes}")
    part.replace(dest)
    return dest


def safe_extract(tgz, dest, members=None):
    """Unpack a downloaded archive (untrusted) into the new directory ``dest``: regular files and directories only, nothing
    outside ``dest`` (no absolute path, no ``..``, no link). ``members`` keeps only the members whose path is one of those, or inside one."""
    dest = pathlib.Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    keep = []
    with tarfile.open(tgz) as tf:
        for m in tf.getmembers():
            p = (dest / m.name).resolve()
            if not (m.isfile() or m.isdir()):
                raise ValueError(f"{tgz}: member {m.name!r} is not a regular file or directory")
            if dest != p and dest not in p.parents:
                raise ValueError(f"{tgz}: member {m.name!r} would leave {dest}")
            if members is None or any(m.name.rstrip("/").lstrip("./") == s or m.name.lstrip("./").startswith(s.rstrip("/") + "/") for s in members):
                keep.append(m)
        tf.extractall(dest, members=keep, filter="data")  # noqa: S202  (members checked above; filter refuses the rest)
    return [m.name for m in keep]


def image_path(key, root):
    """Where ``fetch_image`` puts the target's released image under ``root``."""
    t = target(key)
    return pathlib.Path(root) / key / (t["image"]["member"] or pathlib.PurePosixPath(t["image"]["url"]).name)


def fetch_image(key, root, checksums=CHECKSUMS):
    """Download the target's released ALMA image under ``root/<key>/`` (a FITS file; for HL Tau out of its small tarball), check or
    record its SHA-256, and return ``(path, status)``."""
    t = target(key)
    img, root = t["image"], pathlib.Path(root) / key
    root.mkdir(parents=True, exist_ok=True)
    name = pathlib.PurePosixPath(img["url"]).name
    tmp = root / name
    download(img["url"], tmp, img["bytes"])
    status = record_checksum(img["url"], tmp, img["bytes"], checksums)
    if img["member"] is None:
        return tmp, status
    safe_extract(tmp, root, members=[img["member"]])
    tmp.unlink()
    out = root / img["member"]
    if not out.exists():
        raise FileNotFoundError(f"{img['member']} not in {name}")
    return out, status


# ------------------------------------------------------------------ images and comparison


def alma_image(path):
    """``(image, wcs, header, beam)`` of a released ALMA CLEAN image: Jy/beam, 2-d, FITS orientation (row 0 South);
    ``beam`` is ``(bmaj_mas, bmin_mas, bpa_deg)`` from the header (``BMAJ``/``BMIN``/``BPA``, or the first ``CASAMBM`` row is refused)."""
    from crosscheck import alma_maxent as cx

    img, wcs, hdr = cx.read_fits(path)
    if "BMAJ" not in hdr:
        raise ValueError(f"{path}: no BMAJ in the header (a per-channel beam table?); the released continuum images have one beam")
    return img, wcs, hdr, cx.beam_of(hdr)


def beam_area_arcsec2(bmaj_mas, bmin_mas):
    return float(np.pi * bmaj_mas * bmin_mas / (4 * np.log(2))) / 1e6


def our_image_in_beam(ours_path, ref_path):
    """Our image (Jy/arcsec2, from ``virgil.radio``'s fit, written by ``scripts/dsharp_fit.py``) convolved with the beam of the released image,
    in Jy/beam, on the released image's pixel grid. Returns ``(ours, ref, wcs_ref, beam)``, both in FITS orientation."""
    from crosscheck import alma_maxent as cx

    ours, w_o, h_o = cx.read_fits(ours_path)
    ref, w_r, h_r, beam = alma_image(ref_path)
    px = abs(h_o["CDELT1"]) * 3.6e6
    conv = cx.convolve(ours[::-1], px, *beam)[::-1]  # crosscheck.alma_maxent works with row 0 North
    conv = conv * beam_area_arcsec2(beam[0], beam[1])  # Jy/arcsec2 -> Jy/beam
    on_ref = cx.resample(conv, w_o, w_r, ref.shape)
    return on_ref, ref, w_r, beam


def compare_stats(ours, ref, radius_pix=None):
    """Differences of two images on one grid (``crosscheck.alma_maxent.difference_stats``), within a centred disc if ``radius_pix`` is given,
    plus the rms of ``ref`` in the outer 10 per cent of the field (a noise level)."""
    from crosscheck import alma_maxent as cx

    ny, nx = ref.shape
    rr, cc = np.indices(ref.shape)
    r = np.hypot(rr - (ny - 1) / 2, cc - (nx - 1) / 2)
    region = None if radius_pix is None else r <= radius_pix
    out = cx.difference_stats(ours, ref, region=region)
    out["ref_noise_rms"] = float(np.std(ref[r > 0.9 * r.max()]))
    out["ours_noise_rms"] = float(np.std(ours[r > 0.9 * r.max()]))
    return out


def auto_flux(u, v, vis, weight, fraction=0.01):
    """The start flux: the weighted mean real part of the visibilities on the shortest ``fraction`` of baselines (Jy)."""
    q = np.hypot(u, v)
    sel = q <= np.quantile(q, fraction)
    return float(np.sum(weight[sel] * np.real(vis[sel])) / np.sum(weight[sel]))
