"""Quicklook figures for the informed contest fits: one PNG per dataset, from the reports of contest_informed.py.

    python scripts/contest_informed_quicklook.py REPORT_DIR [--data DIR] [--out DIR]

For each <label>_<family>.json under REPORT_DIR (searched recursively) it rebuilds the chosen rung's model from the
fitted values and draws three panels: the model image at the reference wavelength, the model against the data for
the visibilities, and the same for the phases (closure phases where the data have them). Nothing is fitted here.
Writes <label>_<family>.png and quicklook.json (one row per dataset: the numbers shown on the page).
"""

import argparse
import json
import pathlib
import sys

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import contest_informed as C  # noqa: E402
from virgil.oidata import OIData  # noqa: E402
from virgil.plotting import plot_model  # noqa: E402


def datasets_of(spec, data_dir):
    """The data as the fit read them, without the flux spectra (the panels show visibilities and phases)."""
    paths = [pathlib.Path(data_dir) / f for f in spec["files"]]
    if spec.get("wavel"):
        import tempfile

        import contest_images

        tmp = pathlib.Path(tempfile.mkdtemp())
        paths = [contest_images.select_channels(p, spec["wavel"], tmp) for p in paths]
    return [OIData(str(p)) for p in paths]


def extent_of(values, field):
    """A field of view that holds every fitted component: twice the largest offset or size, at least a third of the field."""
    reach = 0.0
    for path, v in values.items():
        leaf = path.rsplit(".", 1)[-1]
        if leaf in ("dra", "ddec", "diam", "fwhm", "radius", "sigma"):
            reach = max(reach, abs(v[0]) * (1.0 if leaf in ("dra", "ddec", "radius") else 0.75))
    return float(min(max(2.6 * reach, field / 3.0), 2.0 * C.REACH * field))


def figure(report, data_dir, out_dir):
    spec = C.tasks()[report["task"]]
    rungs = dict(C.FAMILIES[report["family"]])
    chosen = next(r for r in report["rungs"] if r["rung"] == report["chosen"])
    rung = rungs[report["chosen"]]
    wavel0 = report.get("reference_wavelength_m") if report.get("chromatic") else None
    if wavel0:
        rung = C.chromatic(rung)
    values = {k: v[0] for k, v in chosen["values"].items()}
    model, _ = C.build(rung, report["resolution_mas"], report["field_mas"], values, wavel0)
    datasets = datasets_of(spec, data_dir)
    fov = extent_of(chosen["values"], report["field_mas"])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    # The colour scale saturates at the 99th percentile, or an unresolved component would be the only thing visible.
    plot_model(model, fov_mas=fov, npix=256, ax=axes[0], saturate=0.99, title=f"{report['chosen']}  (field {fov:.3g} mas)")
    seen_vis, model_vis, seen_phi, model_phi = [], [], [], []
    for d in datasets:
        ov, op = np.ravel(np.asarray(d.vis, float)), np.ravel(np.asarray(d.phi, float))
        got = np.ravel(np.asarray(d.model(model), float))  # the visibilities, then the phases, as the data hold them
        if got.size == ov.size + op.size:
            # Flagged points (channels a task leaves out, bad data) are not fitted and are not shown.
            keep_v = ~np.ravel(np.asarray(d.v2_flag, bool)) if np.size(d.v2_flag) == ov.size else np.ones(ov.size, bool)
            keep_p = ~np.ravel(np.asarray(d.cp_flag, bool)) if np.size(d.cp_flag) == op.size else np.ones(op.size, bool)
            seen_vis.append(ov[keep_v])
            model_vis.append(got[: ov.size][keep_v])
            if op.size:
                mp = got[ov.size:]
                seen_phi.append(op[keep_p])
                model_phi.append((op + (mp - op + np.pi) % (2.0 * np.pi) - np.pi)[keep_p])  # the model on the data's branch
    for ax, seen, pred, name in ((axes[1], seen_vis, model_vis, "visibility"), (axes[2], seen_phi, model_phi, "phase")):
        if not seen:
            ax.text(0.5, 0.5, f"no {name} data to show", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        x, y = np.concatenate(seen), np.concatenate(pred)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], color="0.6", lw=1, zorder=0)
        ax.scatter(x, y, s=4, alpha=0.5, color="#1f6f8b", linewidths=0)
        ax.set(xlabel=f"data: {name}", ylabel=f"model: {name}", title=f"{name}: rms residual {np.sqrt(np.mean((x - y) ** 2)):.3g}")
    fig.suptitle(f"{report['label']}   chi2/N {chosen['chi2_red']:.3g}   {chosen['starts_at_best']}/{chosen['starts']} starts at best"
                 f"   {'chromatic' if wavel0 else 'grey'}", fontsize=11)
    fig.tight_layout()
    path = pathlib.Path(out_dir) / f"{report['label']}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    rms = {}
    if seen_vis:
        rms["vis_rms"] = float(np.sqrt(np.nanmean((np.concatenate(seen_vis) - np.concatenate(model_vis)) ** 2)))
    if seen_phi:
        rms["phase_rms"] = float(np.sqrt(np.nanmean((np.concatenate(seen_phi) - np.concatenate(model_phi)) ** 2)))
    return {"label": report["label"], "task": report["task"], "family": report["family"], "chosen": report["chosen"],
            "rungs": [{"rung": r["rung"], "chi2_red": r["chi2_red"], "bic": r["bic"], "k": r["k"]} for r in report["rungs"]],
            "chi2_red": chosen["chi2_red"], "converged": chosen["converged"], "starts_at_best": chosen["starts_at_best"],
            "starts": chosen["starts"], "chromatic": bool(wavel0), "channels": report.get("channels"),
            "flux_spectra_fitted": report.get("flux_spectra_fitted", 0), "quoted_errors": report.get("quoted_errors"),
            "values": values, "png": path.name, **rms}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reports")
    ap.add_argument("--data", default="~/data/imaging_contests")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out or a.reports)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(pathlib.Path(a.reports).rglob("*.json")):
        if path.name in ("result.json", "quicklook.json"):
            continue
        report = json.loads(path.read_text())
        if "rungs" not in report:
            continue
        try:
            rows.append(figure(report, pathlib.Path(a.data).expanduser(), out))
            print("drew", rows[-1]["png"])
        except Exception as err:  # noqa: BLE001 one bad figure must not hide the others
            print(f"FAILED {path.name}: {type(err).__name__}: {err}")
    (out / "quicklook.json").write_text(json.dumps(sorted(rows, key=lambda r: r["task"]), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
