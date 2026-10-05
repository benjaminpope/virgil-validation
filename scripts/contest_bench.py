"""Synthetic benchmark for contest imaging: known phantoms on the real contest
uv coverages, so that each method choice is judged against a truth.

    python scripts/contest_bench.py simulate [--data DIR] [--out DIR] [--phantoms 6] [--draws 2]
    python scripts/contest_bench.py list [--out DIR]

``simulate`` takes each coverage's contest OIFITS as a template (uv points,
wavelengths, error bars, flags) and replaces V² and closure phases with those
of a phantom (``crosscheck.phantoms``, a family matching that contest's
pre-submission description, never its published truth), plus Gaussian noise
drawn from the quoted errors. The visibilities are a direct NumPy sum
(``crosscheck.sky``): nothing here imports virgil. Tables virgil would also
read but that the phantom does not define (OI_VIS, OI_FLUX) are dropped.
Each dataset is written as ``<coverage>_p<k>_d<j>/`` holding the FITS files,
``truth.npz`` (image, pixel scale, parameters) and ``meta.json``.

Optional calibration errors (``--cal``) follow the 2018 readme: a random
multiplicative V² offset per baseline and pointing, and a random closure
phase offset per triangle and pointing, on top of the noise.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from crosscheck import phantoms, sky  # noqa: E402

# coverage -> (contest files, phantom family). Grey datasets that the first
# campaign fitted, each with the family its pre-submission information named.
COVERAGES = {
    "2004_data2": (["2004/2004-data2.fits"], "spotted_star"),
    "2006": ([f"2006/2006-03-0{n}.fits" for n in (3, 4, 5, 6)], "thin_disk"),
    "2008_agb_H": (["2008/2008-Contest1_H.oifits"], "envelope"),
    "2018": (["2018/Aspro2_Altair_MIRC_6T_1_47493-1_75256-8ch_S1-S2-E1-E2-W1-W2_2018-08-02_FAKE.fits",
              "2018/Aspro2_Altair_PIONIER_1_533-1_772-6ch_A0-B2-C1-D0_2018-08-02_FAKE.fits",
              "2018/Aspro2_Altair_PIONIER_1_533-1_772-6ch_A0-G1-J2-J3_2018-08-02_FAKE.fits",
              "2018/Aspro2_Altair_PIONIER_1_533-1_772-6ch_D0-G2-J3-K0_2018-08-02_FAKE.fits"], "star_disk_planet"),
    "2024_obj1_pionier": (["2024/Obj1_PIONIER_1.5-1.8.fits"], "spiral"),
}
PIXELS_PER_BEAM = 10  # truth grid resolution
DROP = ("OI_VIS", "OI_FLUX")


def wavelengths(hdul):
    """INSNAME -> effective wavelengths (m)."""
    return {h.header.get("INSNAME"): np.asarray(h.data["EFF_WAVE"], float)
            for h in hdul if h.name == "OI_WAVELENGTH"}


def beam_mas(paths):
    """λ_min / B_max over all files, in mas."""
    best = np.inf
    for path in paths:
        with fits.open(path) as h:
            waves = wavelengths(h)
            for t in h:
                if t.name == "OI_VIS2":
                    b = np.hypot(t.data["UCOORD"], t.data["VCOORD"]).max()
                    lam = waves[t.header.get("INSNAME")].min()
                    best = min(best, lam / b / sky.MAS)
    return float(best)


def _vis(cloud, u, v, lam):
    """Complex visibilities at baselines (n,) for wavelengths (m,): (n, m)."""
    return sky.visibility(cloud, u[:, None], v[:, None], lam[None, :])


def closure_noise(t3, err, rng):
    """Closure-phase noise (deg) formed from noisy baseline phases, as in real
    data: within a frame (same TIME/MJD), triangles that share a baseline
    share its phase error, so closure phases from four or more telescopes are
    correlated. Each baseline's phase σ is the smallest σ_cp/√3 among its
    triangles in the frame, and each closure phase gets independent noise on
    top to bring its variance to exactly its quoted σ_cp² (quoted errors
    range over 1-180° within a frame, so a shared average would swamp the
    precise triangles)."""
    # A frame is a (MJD, TIME) pair: OIFITS v1 files such as 2004's give every
    # row one MJD and tell snapshots apart only by TIME (finding F13).
    names = t3.columns.names
    frame_key = list(zip(*(np.asarray(t3[c]).tolist() for c in ("MJD", "TIME") if c in names)))
    sta = np.asarray(t3["STA_INDEX"])
    sigma = {}
    for row, (a, b, c) in enumerate(sta):
        for pair in ((a, b), (b, c), (a, c)):
            sigma.setdefault((frame_key[row], *sorted(pair)), []).append(err[row] / np.sqrt(3))
    sigma = {key: np.min(s, axis=0) for key, s in sigma.items()}
    phase = {key: s * rng.standard_normal(np.shape(s)) for key, s in sigma.items()}

    def baseline(row, i, j):
        sign = 1.0 if i < j else -1.0
        return sign * phase[(frame_key[row], *sorted((i, j)))]

    out = np.array([baseline(r, a, b) + baseline(r, b, c) + baseline(r, c, a) for r, (a, b, c) in enumerate(sta)])
    shared = np.array([sum(sigma[(frame_key[r], *sorted(p))] ** 2 for p in ((a, b), (b, c), (a, c)))
                       for r, (a, b, c) in enumerate(sta)])
    extra = np.sqrt(np.clip(err**2 - shared, 0.0, None))
    return out + extra * rng.standard_normal(np.shape(err))


def simulate_file(template, out, cloud, rng, noise=True, cal=None):
    """Write ``template`` with V² and closure phases replaced by the cloud's."""
    cal = cal or {}
    with fits.open(template) as h:
        waves = wavelengths(h)
        keep = [t for t in h if t.name not in DROP]
        for t in keep:
            if t.name == "OI_VIS2":
                d = t.data
                lam = waves[t.header.get("INSNAME")]
                v2 = np.abs(_vis(cloud, d["UCOORD"], d["VCOORD"], lam)) ** 2
                if cal.get("v2_rel"):  # one offset per row (baseline and pointing)
                    v2 = v2 * (1 + cal["v2_rel"] * rng.standard_normal((len(d), 1)))
                err = np.asarray(d["VIS2ERR"], float).reshape(v2.shape)
                if noise:
                    v2 = v2 + err * rng.standard_normal(v2.shape)
                d["VIS2DATA"] = v2.reshape(np.shape(d["VIS2DATA"]))
            elif t.name == "OI_T3":
                d = t.data
                lam = waves[t.header.get("INSNAME")]
                u1, v1, u2, v2_ = d["U1COORD"], d["V1COORD"], d["U2COORD"], d["V2COORD"]
                bis = _vis(cloud, u1, v1, lam) * _vis(cloud, u2, v2_, lam) * np.conj(_vis(cloud, u1 + u2, v1 + v2_, lam))
                phi = np.degrees(np.angle(bis))
                if cal.get("cp_deg"):  # one offset per row (triangle and pointing)
                    phi = phi + cal["cp_deg"] * rng.standard_normal((len(d), 1))
                err = np.asarray(d["T3PHIERR"], float).reshape(phi.shape)
                if noise:
                    phi = phi + closure_noise(d, err, rng)
                d["T3PHI"] = (((phi + 180.0) % 360.0) - 180.0).reshape(np.shape(d["T3PHI"]))
                d["T3AMP"] = np.abs(bis).reshape(np.shape(d["T3AMP"]))
        fits.HDUList([t.copy() for t in keep]).writeto(out, overwrite=True)


def truth_grid(params, beam):
    pixel = beam / PIXELS_PER_BEAM
    npix = int(np.ceil(2 * phantoms.extent(params) / pixel)) | 1
    return npix, pixel


def simulate(data_dir, out_dir, n_phantoms, n_draws, seed=0, cal=None, coverages=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for ci, (cov, (files, family)) in enumerate(COVERAGES.items()):
        if coverages and cov not in coverages:
            continue
        paths = [data_dir / f for f in files]
        beam = beam_mas(paths)
        for k in range(n_phantoms):
            prng = np.random.default_rng([seed, ci, k])
            params = phantoms.sample(family, prng, beam)
            npix, pixel = truth_grid(params, beam)
            image = phantoms.render(params, npix, pixel)
            cloud = sky.pixel_image(image, pixel)
            for j in range(n_draws):
                name = f"{cov}_p{k}_d{j}"
                d = out_dir / name
                d.mkdir(exist_ok=True)
                rng = np.random.default_rng([seed, ci, k, j, 1])
                for p in paths:
                    simulate_file(p, d / p.name, cloud, rng, cal=cal)
                np.savez_compressed(d / "truth.npz", image=image, pixel=pixel, npix=npix, beam=beam)
                (d / "meta.json").write_text(json.dumps(
                    {"coverage": cov, "family": family, "phantom": k, "draw": j, "seed": seed, "cal": cal or {},
                     "files": [p.name for p in paths], "beam_mas": beam, "pixel_mas": pixel, "npix": npix,
                     "params": params}, indent=1, default=float))
                made.append(name)
    (out_dir / "index.json").write_text(json.dumps(made, indent=1))
    return made


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--data", default="~/data/imaging_contests")
    s.add_argument("--out", default="~/data/imaging_contests/bench")
    s.add_argument("--phantoms", type=int, default=6)
    s.add_argument("--draws", type=int, default=2)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--coverage", action="append", help="only these coverages (repeatable)")
    s.add_argument("--cal", default=None, help="calibration offsets 'v2_rel,cp_deg', e.g. 0.02,1.0")
    ls = sub.add_parser("list")
    ls.add_argument("--out", default="~/data/imaging_contests/bench")
    args = parser.parse_args()
    if args.cmd == "simulate":
        cal = None
        if args.cal:
            a, b = (float(x) for x in args.cal.split(","))
            cal = {"v2_rel": a, "cp_deg": b}
        made = simulate(pathlib.Path(args.data).expanduser(), pathlib.Path(args.out).expanduser(),
                        args.phantoms, args.draws, args.seed, cal, args.coverage)
        print(f"{len(made)} datasets")
    else:
        index = json.loads((pathlib.Path(args.out).expanduser() / "index.json").read_text())
        for i, name in enumerate(index):
            print(i, name)


if __name__ == "__main__":
    main()
