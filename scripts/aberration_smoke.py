"""Single-night smoke fits of the wide single-fibre GRAVITY binaries (zeta Boo, eta Oph).

    python scripts/aberration_smoke.py zet_boo --epochs 1 3 --out <dir>
    python scripts/aberration_smoke.py eta_oph --epochs 2 4 --out <dir> --dry-run

A laptop check for design/gravity_aberration_test.md, not a campaign. It reuses the registered Track B
per-epoch recipe of ``scripts/trackb_fit.py`` unchanged (GRAVITY_SC closure phases, 2.05-2.38 um without
Br-gamma, a likelihood grid, then ``fit`` from the five best peaks, fixed uniform-disk diameters from the
paper) on the chosen rows of ``reference_positions`` (0-based), and prints the fitted position against the
published one (Waisberg+2025, Tables 3 and 4) in micro-arcseconds, with the file headers' per-telescope
fibre offsets (ESO QC MET SOBJ DRA/DDEC) beside it. No orbit is fitted and no orbit is used.

The published error is the paper's scatter-based 40 uas (zeta Boo) or 20 uas (eta Oph) per axis, so a
residual several times that, nights apart, is the signature the note looks for.
"""

import argparse
import json
import os
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import trackb_fit as tb  # noqa: E402
from crosscheck import astrometry as A  # noqa: E402

FIBRE_KEYS = [f"ESO QC MET SOBJ {k}{i}" for i in (1, 2, 3, 4) for k in ("DRA", "DDEC")]


def fibre_offsets(path):
    """Per-telescope (DRA, DDEC) fibre-position QC values of one product, as the header carries them."""
    from astropy.io import fits
    with fits.open(path) as h:
        hd = h[0].header
        return [[float(hd.get(f"ESO QC MET SOBJ DRA{i}", np.nan)), float(hd.get(f"ESO QC MET SOBJ DDEC{i}", np.nan))]
                for i in (1, 2, 3, 4)]


def residual(row, sigma_mas):
    """Fitted minus published position in uas, at the sign (180 degree fold for a near-equal pair) nearer
    the paper's, with the separation and PA parts and the residual in units of the published sigma."""
    pos = np.array([row["dra"], row["ddec"]])
    pub = np.array(row["published"])
    sgn = row.get("sign", 1.0)
    d = pos - sgn * pub
    sep_p, pa_p = A.offsets_to_sep_pa(*(sgn * pub))
    u = np.array([np.sin(np.radians(pa_p)), np.cos(np.radians(pa_p))])  # radial (along the separation)
    t = np.array([np.cos(np.radians(pa_p)), -np.sin(np.radians(pa_p))])  # tangential (along increasing PA)
    return dict(dra_uas=1e3 * d[0], ddec_uas=1e3 * d[1], radial_uas=1e3 * float(d @ u), tangential_uas=1e3 * float(d @ t),
                total_uas=1e3 * float(np.hypot(*d)), total_sigma=float(np.hypot(*d) / sigma_mas))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("system", choices=["zet_boo", "eta_oph"])
    ap.add_argument("--epochs", type=int, nargs="+", required=True, help="0-based rows of reference_positions")
    ap.add_argument("--root", default=os.environ.get("TRACKB_ROOT", str(pathlib.Path.home() / "data/eso_binaries/trackb")))
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root, out = pathlib.Path(a.root), pathlib.Path(a.out)
    s = json.loads((root / a.system / "system.json").read_text())
    if not a.dry_run:
        import jax
        jax.config.update("jax_enable_x64", True)
    diam = tb.diameters(a.system, s)
    fmax = 2.0 if a.system in A.NEAR_EQUAL else 1.0
    sigma = A.WAISBERG_SIGMA[a.system]
    eps = [e for e in tb.epochs_of(a.system, s, root) if e["index"] in a.epochs]
    rows = []
    for ep in eps:
        row = dict(date=ep["date"], mjd=ep["mjd"], published_record=ep["rec"])
        row.update(tb.fit_epoch(a.system, ep, diam, fmax, a.dry_run))
        row["fibre_offsets_dra_ddec"] = fibre_offsets(ep["files"][0])
        if not a.dry_run:
            row["residual"] = residual(row, sigma)
            sep, pa = row["sep"], row["pa"]
            ps, pp = A.offsets_to_sep_pa(*(row["sign"] * np.array(row["published"])))
            print(f"{a.system} {ep['date']}: fitted ({row['dra']:+.4f}, {row['ddec']:+.4f}) mas  sep {sep:.4f} PA {pa:.3f}  "
                  f"published sep {ps:.4f} PA {pp:.3f}  residual {row['residual']['dra_uas']:+.0f}, {row['residual']['ddec_uas']:+.0f} uas "
                  f"(radial {row['residual']['radial_uas']:+.0f}, tangential {row['residual']['tangential_uas']:+.0f}; "
                  f"{row['residual']['total_sigma']:.1f} sigma_pub)  flux {row['flux']:.3f}  chi2_raw/N {row['chi2_raw_per_n']:.2f}  "
                  f"scale {row['scales']}  [{row['seconds']:.0f} s]", flush=True)
        else:
            print(f"{a.system} {ep['date']}: n_vis {row['n_vis']} n_phi {row['n_phi']} ch {row['n_channels']} "
                  f"Bmax {row['b_max_m']:.1f} m grid {row['grid_shape']} step {row['grid_step_mas']:.3f} mas", flush=True)
        rows.append(row)
    if not a.dry_run:
        import virgil
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{a.system}_smoke.json").write_text(json.dumps(
            dict(system=a.system, sigma_published_mas=sigma, diameters_mas=diam, virgil=getattr(virgil, "__version__", None),
                 epochs=rows), indent=1, default=float))


if __name__ == "__main__":
    main()
