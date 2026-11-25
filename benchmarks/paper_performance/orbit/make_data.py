"""Deterministic data for the orbit comparison (virgil vs Octofitter.jl). Plan: reports/benchmarks-plan.md section 5.

Writes under ``~/.cache/virgil-bench/data/orbit/`` and ``orbit/manifest.json`` (sha256 of every file; committed).

Everything is simulated on the crosscheck side (``crosscheck.orbits`` for positions, ``crosscheck.simulate.observe``
for OIFITS); neither virgil nor Octofitter is imported. Cases:

* ``O-A-sim``       10 epochs over 300 d, 0.05 mas per axis, ``astrometry_O-A-sim.csv``.
* ``O-A-sim-pulls`` the same orbit and epochs, 20 noise seeds, ``astrometry_O-A-sim-pulls_seed<k>.csv``.
* ``O-A-real``      HD 224974, the 12 ``role: paper`` epochs of ``oidb/trackb/hd_224974.json``; error ellipse to
                    (sigma_ra, sigma_dec, cor) by one neutral conversion here; ``astrometry_O-A-real.csv``.
* ``O-B-sim``       the O-A-sim orbit as 8 epochs of OIFITS (4 ATs, 3 hour angles, 6 H-band channels, point + point
                    at flux ratio 0.25, sigma_V2 = 0.03, sigma_CP = 1 deg, triangle noise), one file per epoch,
                    ``oifits_O-B-sim/epoch<k>.fits`` plus ``epochs_O-B-sim.csv`` (the epoch MJD of each file).

Run: ``~/.claude/bin/pyquick $BENCH_PYTHON benchmarks/paper_performance/orbit/make_data.py [--check]``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent))

import neutral  # noqa: E402
from crosscheck import astrometry, orbits, simulate  # noqa: E402

DATA = Path.home() / ".cache" / "virgil-bench" / "data" / "orbit"
MANIFEST = HERE / "manifest.json"

# ---- the simulated truth (plan section 5, Data) ------------------------------------------------------------------
T_REF = 60000.0                     # reference epoch of the orbit (MJD); phase is defined here
TRUTH = dict(P=200.0, M_tot=2.0, plx=20.0, e=0.34, inc=53.4, omega=76.9, Omega=28.2, M0=2.1)  # M0 [rad] at T_REF
N_EPOCH_A, N_EPOCH_B, SPAN_D = 10, 8, 300.0
SIGMA_ASTROM = 0.05                 # mas per axis
N_PULL_SEEDS = 20
SEED_A, SEED_PULLS, SEED_B, SEED_EPOCHS = 20261118, 20261200, 20261300, 20261117

# O-B-sim
FLUX_RATIO = 0.25
STATIONS = np.array([[0.0, 0.0], [32.0, -18.0], [-24.0, 52.0], [88.0, 41.0]])   # illustrative 4-AT layout (E, N) m
HOUR_ANGLES = (-1.0, 0.0, 1.0)
WAVELENGTHS = np.linspace(1.5e-6, 1.8e-6, 6)                                    # H band
DEC_DEG = -30.0
SIGMA_V2, SIGMA_CP = 0.03, 1.0


GM_SUN, AU_M, DAY_S = 1.3271244e20, 149597870700.0, 86400.0     # IAU 2015 nominal values (as virgil's docs)
KEPLER_K = 4 * np.pi**2 * AU_M**3 / (GM_SUN * DAY_S**2)         # M [Msun] = K a [au]^3 / P [d]^2


def a_mas(P=None, M_tot=None, plx=None, **_):
    """Semi-major axis in mas from Kepler's third law with the IAU nominal constants: a [au] = (M_tot P^2 / K)^(1/3)
    (P in days), times the parallax in mas."""
    P = TRUTH["P"] if P is None else P
    return plx * (M_tot * P**2 / KEPLER_K) ** (1.0 / 3.0)


def t_peri(P, M0, t_ref=T_REF):
    """Time of periastron (MJD) from the mean anomaly ``M0`` [rad] at ``t_ref``: M = 2 pi (t - t_peri) / P."""
    return t_ref - M0 * P / (2.0 * np.pi)


def sky_position(t, **p):
    p = {**TRUTH, **p}
    return orbits.position(t, p["P"], t_peri(p["P"], p["M0"]), p["e"], p["inc"], p["omega"], p["Omega"],
                           a_mas(**p))


def epoch_times(n, seed):
    rng = np.random.default_rng(seed)
    return T_REF + np.sort(rng.uniform(0.0, SPAN_D, n))


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


ASTROM_HEADER = ["epoch", "ra", "dec", "sigma_ra", "sigma_dec", "cor"]   # Octofitter RelAstromObs columns (mas, MJD)


def make_astrometry(out):
    t = epoch_times(N_EPOCH_A, SEED_EPOCHS)
    ra0, dec0 = sky_position(t)
    rng = np.random.default_rng(SEED_A)
    ra = ra0 + SIGMA_ASTROM * rng.standard_normal(t.size)
    dec = dec0 + SIGMA_ASTROM * rng.standard_normal(t.size)
    _write_csv(out / "astrometry_O-A-sim.csv", ASTROM_HEADER,
               [(f"{a:.10f}", f"{b:.10f}", f"{c:.10f}", SIGMA_ASTROM, SIGMA_ASTROM, 0.0)
                for a, b, c in zip(t, ra, dec)])
    _write_csv(out / "truth_O-A-sim.csv", ["epoch", "ra", "dec"],
               [(f"{a:.10f}", f"{b:.12f}", f"{c:.12f}") for a, b, c in zip(t, ra0, dec0)])
    for k in range(N_PULL_SEEDS):
        r = np.random.default_rng(SEED_PULLS + k)
        ra = ra0 + SIGMA_ASTROM * r.standard_normal(t.size)
        dec = dec0 + SIGMA_ASTROM * r.standard_normal(t.size)
        _write_csv(out / f"astrometry_O-A-sim-pulls_seed{k:02d}.csv", ASTROM_HEADER,
                   [(f"{a:.10f}", f"{b:.10f}", f"{c:.10f}", SIGMA_ASTROM, SIGMA_ASTROM, 0.0)
                    for a, b, c in zip(t, ra, dec)])


def ellipse_to_sigmas(sigma_maj, sigma_min, pa_deg):
    """(sigma_ra, sigma_dec, cor) of the 1-sigma ellipse (major axis at PA East of North): the same bivariate Gaussian."""
    cov = astrometry.ellipse_covariance(sigma_maj, sigma_min, pa_deg)
    sa, sd = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))
    return sa, sd, float(cov[0, 1] / (sa * sd))


def make_real(out):
    rec = json.loads((ROOT / "oidb" / "trackb" / "hd_224974.json").read_text())
    rows = []
    for ep in rec["epochs"]:
        if ep.get("role") != "paper":
            continue
        sa, sd, cor = ellipse_to_sigmas(ep["sigma_major_mas"], ep["sigma_minor_mas"], ep["sigma_pa_deg"])
        rows.append((f"{ep['mjd']:.6f}", ep["dra_mas"], ep["ddec_mas"], f"{sa:.10f}", f"{sd:.10f}", f"{cor:.10f}"))
    assert len(rows) == 12, len(rows)
    _write_csv(out / "astrometry_O-A-real.csv", ASTROM_HEADER, rows)


def make_oifits(out):
    t = epoch_times(N_EPOCH_B, SEED_EPOCHS + 1)
    flux = FLUX_RATIO
    rows = []
    for k, tk in enumerate(t):
        dra, ddec = (float(x[0]) for x in sky_position(np.array([tk])))

        def vis(u, v, wl, dra=dra, ddec=ddec):
            return neutral.binary_vis(u, v, wl, dra, ddec, flux)   # the neutral convention (sign -1), as crosscheck

        path = out / "oifits_O-B-sim" / f"epoch{k:02d}.fits"
        path.parent.mkdir(parents=True, exist_ok=True)
        simulate.observe(path, vis, STATIONS, hour_angles_h=HOUR_ANGLES, wavelengths=WAVELENGTHS, dec_deg=DEC_DEG,
                         sigma_v2=SIGMA_V2, sigma_cp_deg=SIGMA_CP, rng=np.random.default_rng(SEED_B + k),
                         phase_noise="triangle")
        rows.append((f"{tk:.10f}", f"{dra:.10f}", f"{ddec:.10f}", path.name))
    _write_csv(out / "epochs_O-B-sim.csv", ["epoch", "true_ra", "true_dec", "file"], rows)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(out=DATA):
    make_astrometry(out)
    make_real(out)
    make_oifits(out)
    files = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    # FITS headers carry no timestamps from the writer (DATE-OBS is the simulated date), so the hash is stable
    MANIFEST.write_text(json.dumps({"truth": TRUTH, "t_ref": T_REF, "files": files}, indent=1) + "\n")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="rebuild in a scratch directory and compare with manifest.json")
    ap.add_argument("--out", default=str(DATA))
    a = ap.parse_args()
    if a.check:
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="orbit-data-"))
        old = MANIFEST.read_text()
        files = build(tmp)
        MANIFEST.write_text(old)
        want = json.loads(old)["files"]
        bad = [k for k in want if want[k] != files.get(k)]
        print("manifest check:", "OK" if not bad else f"MISMATCH {bad}", f"({len(want)} files)")
        sys.exit(1 if bad else 0)
    files = build(Path(a.out))
    print(f"wrote {len(files)} files to {a.out}; manifest {MANIFEST}")


if __name__ == "__main__":
    main()
