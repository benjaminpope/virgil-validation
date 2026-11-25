"""Deterministic simulated data for the imaging comparison (virgil vs eht-imaging), plus the helpers every runner shares.

Run (BENCH_PYTHON, under pyquick):  ``python make_data.py [--cases sim-well sim-ring sim-large] [--check]``

Writes under ``~/.cache/virgil-bench/data/imaging/`` (never committed; ``*.fits`` is git-ignored):

    sim-well.fits / .npz    the ``well_posed`` problem of tests/test_ehtim_reconstruction.py: 3 stations, 13 hour angles,
                            6 channels 1.5-2.4 um, dec -50, sigma 0.002, seed 1, two 0.8 mas Gaussians (0.3 relative
                            flux at (3, -2) mas). Shared by I-sim-24, I-sim-48 and I-sim-96 (only the image grid differs).
    sim-ring.fits / .npz    same coverage and sigma; a ring of diameter 5 mas, width (FWHM) 0.8 mas, 20 % azimuthal
                            asymmetry, total flux 1 (I-sim-ring, 48 pixels).
    sim-large.fits / .npz   4 stations (VLTI UT1-UT4), 25 hour angles, 30 channels, the Gaussian scene (I-sim-large); the
                            written closure triangles are exactly the minimal set eht-imaging forms for that array.

and ``imaging/manifest.json`` (sha256 of every file; committed). Runners verify it before use.

Simulated data and truth come from ``crosscheck`` (plan section 1, independence). This module imports neither virgil nor
eht-imaging. The only call into eht-imaging's environment is the triangle query for sim-large, a subprocess of
``ehtim_bench_worker.py`` (it returns station triples; nothing else crosses).

Shared helpers (used by run_virgil.py, ehtim_bench_worker.py, compare.py): ``load``, ``ehtim_rows``, ``neutral_data``,
``start_image``, ``truth_image``, ``CASES``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = HERE.parents[2]
DATA_DIR = Path(os.environ.get("VIRGIL_BENCH_HOME", Path.home() / ".cache" / "virgil-bench")) / "data" / "imaging"
MANIFEST = HERE / "manifest.json"

MAS = np.pi / 180.0 / 3600.0 / 1000.0
FIELD_MAS = 12.0
SIGMA = 0.002          # visibility amplitude error (each of Re and Im)
SEED = 1
FWHM_START_MAS = 4.0   # common start: circular Gaussian

# stations, metres East / North. UTS3 is the three-station array of the vv test; UT4 adds the fourth VLTI UT
# (ESO station table, as src/crosscheck/orbit_population.py T1)
UTS3 = np.array([[-9.925, -20.335], [14.887, 30.502], [103.306, 43.999]])
UTS4 = np.array([[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]])

#: dataset -> description. ``grids``: image sizes (pixels over the 12 mas field).
DATASETS = {
    "sim-well": {"stations": "UTS3", "n_ha": 13, "n_ch": 6, "scene": "gauss", "ha": (-4.0, 4.0)},
    "sim-ring": {"stations": "UTS3", "n_ha": 13, "n_ch": 6, "scene": "ring", "ha": (-4.0, 4.0)},
    "sim-large": {"stations": "UTS4", "n_ha": 25, "n_ch": 30, "scene": "gauss", "ha": (-4.0, 4.0)},
}
#: case id -> (dataset, npix)
CASES = {
    "I-sim-24": ("sim-well", 24), "I-sim-48": ("sim-well", 48), "I-sim-96": ("sim-well", 96),
    "I-sim-ring": ("sim-ring", 48), "I-sim-large": ("sim-large", 48),
}

# ----------------------------------------------------------------------------------------------------------------
# scenes (crosscheck side)
# ----------------------------------------------------------------------------------------------------------------

RING = {"diam": 5.0, "fwhm": 0.8, "amp": 0.2, "pa": 90.0}  # mas; asymmetry cos(theta - 90 deg), theta from North
GAUSS = [(1.0, 0.8, 0.0, 0.0), (0.3, 0.8, 3.0, -2.0)]      # (flux, sigma mas, dra, ddec)


def vis_function(scene):
    from crosscheck import sky

    if scene == "gauss":
        def vis(u, v, w):
            return (sky.vis_gaussian(u, v, w, 0.8) + 0.3 * sky.vis_gaussian(u, v, w, 0.8, dra=3.0, ddec=-2.0)) / 1.3
        return vis
    cloud = sky.inclined_ring(RING["diam"], az_amps=(RING["amp"],), az_pas=(RING["pa"],), modulation="sky")

    def vis(u, v, w):
        return sky.visibility(cloud, u, v, w) * sky.gaussian_blur_factor(u, v, w, RING["fwhm"])
    return vis


def _pixel_centres(npix, pixel_mas):
    """East and North offsets (mas) of pixel centres, row 0 North and column 0 East (the Image convention)."""
    c = (np.arange(npix) - (npix - 1) / 2.0) * pixel_mas
    east, north = np.meshgrid(-c, c[::-1])
    return east, north


def truth_image(scene, npix, pixel_mas):
    """The analytic scene sampled at pixel centres, unit flux (row 0 North, column 0 East)."""
    east, north = _pixel_centres(npix, pixel_mas)
    out = np.zeros((npix, npix))
    if scene == "gauss":
        for flux, sig, dra, ddec in GAUSS:
            out += flux * np.exp(-((east - dra) ** 2 + (north - ddec) ** 2) / (2 * sig**2)) / (2 * np.pi * sig**2)
    else:
        from crosscheck import sky

        cloud = sky.inclined_ring(RING["diam"], az_amps=(RING["amp"],), az_pas=(RING["pa"],), modulation="sky")
        sig = RING["fwhm"] / (2 * np.sqrt(2 * np.log(2)))
        e, n = east.ravel(), north.ravel()
        flat = np.zeros(e.size)
        for k in range(0, cloud.weight.size, 256):  # chunks bound the memory
            sl = slice(k, k + 256)
            d2 = (e[:, None] - cloud.east[None, sl]) ** 2 + (n[:, None] - cloud.north[None, sl]) ** 2
            flat += np.exp(-d2 / (2 * sig**2)) @ cloud.weight[sl]
        out = flat.reshape(npix, npix)
    return out / out.sum()


def start_image(npix, pixel_mas, fwhm_mas=FWHM_START_MAS):
    """The common start: a circular Gaussian of FWHM ``fwhm_mas`` at the field centre, unit flux."""
    east, north = _pixel_centres(npix, pixel_mas)
    sig = fwhm_mas / (2 * np.sqrt(2 * np.log(2)))
    b = np.exp(-(east**2 + north**2) / (2 * sig**2))
    return b / b.sum()


# ----------------------------------------------------------------------------------------------------------------
# building and loading
# ----------------------------------------------------------------------------------------------------------------


def _read_fits(path):
    """Arrays the neutral scorer needs, read from the OIFITS file exactly as virgil reads it."""
    from astropy.io import fits

    with fits.open(path) as h:
        wl = np.asarray(h["OI_WAVELENGTH"].data["EFF_WAVE"], float)
        v, t = h["OI_VIS"].data, h["OI_T3"].data
        nw = wl.size
        out = {
            "amp": np.asarray(v["VISAMP"], float).ravel(), "sigma_amp": np.asarray(v["VISAMPERR"], float).ravel(),
            "u_amp": np.repeat(np.asarray(v["UCOORD"], float), nw), "v_amp": np.repeat(np.asarray(v["VCOORD"], float), nw),
            "wavel_amp": np.tile(wl, len(v)),
            "cphase_deg": np.asarray(t["T3PHI"], float).ravel(), "sigma_cp_deg": np.asarray(t["T3PHIERR"], float).ravel(),
            "wavel_cp": np.tile(wl, len(t)),
        }
        u1, v1, u2, v2 = (np.repeat(np.asarray(t[c], float), nw) for c in ("U1COORD", "V1COORD", "U2COORD", "V2COORD"))
        out.update(u1=u1, v1=v1, u2=u2, v2=v2, u3=-(u1 + u2), v3=-(v1 + v2))
        out["t3_sta"] = np.asarray(t["STA_INDEX"], int)
        out["t3_mjd"] = np.asarray(t["MJD"], float)
    return out


def _filter_t3(path, keep_triples):
    """Rewrite the OI_T3 table of ``path`` keeping only triangles whose station set is in ``keep_triples``."""
    from astropy.io import fits

    with fits.open(path) as h:
        t3 = h["OI_T3"]
        mask = np.array([tuple(sorted(int(s) for s in row)) in keep_triples for row in t3.data["STA_INDEX"]])
        t3.data = t3.data[mask]
        h.writeto(path, overwrite=True)


def _ehtim_triangles(rows):
    """Station triples of eht-imaging's minimal closure-triangle set, one snapshot (a subprocess of the worker)."""
    import tempfile

    env = os.environ.get("EHTIM_PYTHON")
    if not env:
        raise RuntimeError("EHTIM_PYTHON is not set (source ~/.cache/virgil-bench/activate.sh)")
    with tempfile.TemporaryDirectory() as d:
        task, res = Path(d) / "task.json", Path(d) / "res.json"
        task.write_text(json.dumps({"task": "triangles", "rows": rows}))
        subprocess.run([env, str(HERE / "ehtim_bench_worker.py"), "--task-file", str(task), str(res)], check=True,
                       capture_output=True, text=True)
        return json.loads(res.read_text())["triangles"]


def build(name):
    """Simulate one dataset and write its OIFITS and npz; returns the paths."""
    from crosscheck import simulate

    spec = DATASETS[name]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.fits"
    stations = {"UTS3": UTS3, "UTS4": UTS4}[spec["stations"]]
    sim = simulate.observe_visibilities(
        path, vis_function(spec["scene"]), stations, hour_angles_h=np.linspace(*spec["ha"], spec["n_ha"]),
        wavelengths=np.linspace(1.5e-6, 2.4e-6, spec["n_ch"]), dec_deg=-50.0, sigma=SIGMA,
        rng=np.random.default_rng(SEED),
    )
    if len(stations) > 3:
        # eht-imaging forms the minimal triangle set of the array; write exactly those (plan R3)
        rows = ehtim_rows({k: sim[k] for k in ("mjd", "sta", "u", "v", "wl", "vis")})
        keep = {tuple(sorted(t)) for t in _ehtim_triangles(rows)}
        _filter_t3(path, keep)
    arrays = _read_fits(path)
    np.savez_compressed(DATA_DIR / f"{name}.npz", **{f"sim_{k}": np.asarray(sim[k]) for k in sim}, **arrays)
    return path


def load(name):
    """The dataset's arrays (``sim_*`` for the complex-visibility table, the rest for the neutral scorer)."""
    z = np.load(DATA_DIR / f"{name}.npz")
    return {k: z[k] for k in z.files}


def fits_path(name):
    return DATA_DIR / f"{name}.fits"


def ehtim_rows(sim):
    """Rows ``[time, t1, t2, u, v, re, im, sigma]`` for eht-imaging's Obsdata (as the vv test builds them).

    One time per snapshot and channel (so triangles close within a channel); eht-imaging's Fourier sign is the opposite
    of the OIFITS convention, so (u, v) are passed negated, in wavelengths."""
    g = (lambda k: sim[f"sim_{k}"]) if "sim_mjd" in sim else (lambda k: sim[k])
    mjd, sta, u, v, wl, vis = g("mjd"), g("sta"), g("u"), g("v"), g("wl"), g("vis")
    epochs = list(np.unique(mjd))
    nw = vis.shape[1]
    rows = []
    for k in range(vis.shape[0]):
        for c in range(nw):
            rows.append([
                float(epochs.index(mjd[k, c]) * nw + c), f"S{int(sta[k, 0])}", f"S{int(sta[k, 1])}",
                float(-u[k, c] / wl[k, c]), float(-v[k, c] / wl[k, c]), float(vis[k, c].real), float(vis[k, c].imag),
                SIGMA,
            ])
    return rows


def neutral_data(d, npix, pixel_mas):
    """The ``data`` dict of ``neutral.image_terms`` (OIFITS convention: score with ``sign=-1``)."""
    keys = ("amp", "sigma_amp", "u_amp", "v_amp", "wavel_amp", "cphase_deg", "sigma_cp_deg", "wavel_cp",
            "u1", "v1", "u2", "v2", "u3", "v3")
    out = {k: d[k] for k in keys}
    out.update(npix=npix, pixel_mas=pixel_mas)
    return out


def pixel_mas(npix):
    return FIELD_MAS / npix


# ----------------------------------------------------------------------------------------------------------------
# manifest
# ----------------------------------------------------------------------------------------------------------------


def write_manifest(names):
    sys.path.insert(0, str(BENCH))
    import common

    files = {}
    for n in names:
        for ext in ("fits", "npz"):
            files[f"{n}.{ext}"] = common.file_sha256(DATA_DIR / f"{n}.{ext}")
    MANIFEST.write_text(json.dumps({"comparison": "imaging", "seed": SEED, "files": files}, indent=1) + "\n")


def check_manifest():
    sys.path.insert(0, str(BENCH))
    import common

    return common.verify_manifest(MANIFEST, DATA_DIR)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--check" in args:
        bad = check_manifest()
        print("manifest ok" if not bad else f"MISMATCH: {bad}")
        sys.exit(1 if bad else 0)
    names = args[args.index("--cases") + 1:] if "--cases" in args else list(DATASETS)
    for n in names:
        p = build(n)
        print("wrote", p)
    write_manifest(sorted(DATASETS))
    print("manifest", MANIFEST)
