"""Run inside CASA's own environment (.venv-alma), not ours.

    python casa_clean_worker.py task.json result.json

Tasks (``task["task"]``):

``tclean``   run CASA's ``tclean`` (casatasks) on a measurement set and write the PSF, the restored image, the
             residual and the model as one .npz in this repository's orientation (row 0 North, column 0
             East), together with the restoring beam CASA fitted to its own PSF (``psfcutoff`` 0.35 by default);
``columns``  read DATA, WEIGHT, FLAG and UVW of the measurement set and the ddid frequencies straight from the
             tables with ``casatools.table`` (raw columns, no selection logic), for the reader check.

Nothing here imports virgil.
"""

import json
import os
import pathlib
import shutil
import sys

import numpy as np

_site = pathlib.Path(sys.executable).parents[1] / "casasiteconfig.py"
if "CASASITECONFIG" not in os.environ and _site.exists():
    os.environ["CASASITECONFIG"] = str(_site)

sys.path.insert(0, str(pathlib.Path(__file__).parent))


def _orient(arr):
    """CASA image array [x, y, ...] -> [row (North first), col (East first)]. The RA axis runs East to West with
    increasing x (CDELT1 < 0), so column 0 East is x = 0; y runs South to North, so row 0 North is y = max."""
    return np.asarray(arr)[:, :, 0, 0][:, ::-1].T


def tclean(task):
    import casatasks
    import casatools

    out = pathlib.Path(task["outdir"])
    out.mkdir(parents=True, exist_ok=True)
    name = str(out / "im")
    for p in out.glob("im.*"):
        shutil.rmtree(p, ignore_errors=True)
    n, cell = int(task["imsize"]), float(task["cell_mas"])
    casatasks.tclean(
        vis=task["ms"], imagename=name, imsize=[n, n], cell=f"{cell}mas", specmode="mfs", gridder="standard",
        weighting=task.get("weighting", "natural"), robust=float(task.get("robust", 0.5)),
        deconvolver="hogbom", niter=int(task["niter"]), gain=float(task.get("gain", 0.1)),
        threshold=task.get("threshold", "0Jy"), psfcutoff=float(task.get("psfcutoff", 0.35)), stokes="I",
        interactive=False, pbcor=False, datacolumn="data", savemodel="none", parallel=False, verbose=False,
        restart=True, calcres=True, calcpsf=True,
    )
    ia = casatools.image()
    arrays = {}
    for key, suffix in (("psf", "psf"), ("image", "image"), ("residual", "residual"), ("model", "model")):
        ia.open(f"{name}.{suffix}")
        arrays[key] = _orient(ia.getchunk())
        if suffix == "image":
            beam = ia.restoringbeam()
        ia.close()
    ia.open(f"{name}.psf")
    psf_beam = ia.restoringbeam()
    ia.close()
    ia.done()
    np.savez(out / "products.npz", **arrays)
    as_mas = lambda q: float(q["value"]) * (1000.0 if q["unit"] == "arcsec" else 1.0)  # noqa: E731
    pa = beam["positionangle"]
    return {"npz": str(out / "products.npz"), "casatasks": ".".join(str(x) for x in casatasks.version()[:4]),
            "beam_mas_deg": [as_mas(beam["major"]), as_mas(beam["minor"]), float(pa["value"])],
            "psf_beam_mas_deg": [as_mas(psf_beam["major"]), as_mas(psf_beam["minor"]), float(psf_beam["positionangle"]["value"])]}


def columns(task):
    import casatools

    tb = casatools.table()
    path = task["ms"]
    tb.open(path + "/SPECTRAL_WINDOW")
    freqs = [np.asarray(tb.getcell("CHAN_FREQ", i)) for i in range(tb.nrows())]
    tb.close()
    tb.open(path + "/DATA_DESCRIPTION")
    spw = tb.getcol("SPECTRAL_WINDOW_ID")
    tb.close()
    tb.open(path)
    cols = {k: tb.getcol(k) for k in ("DATA", "WEIGHT", "FLAG", "FLAG_ROW", "UVW", "ANTENNA1", "ANTENNA2", "DATA_DESC_ID", "TIME")}
    tb.close()
    out = pathlib.Path(task["out"])
    np.savez(out, ddid_freqs=np.array([freqs[s] for s in spw]), **cols)
    return {"npz": str(out), "casatools": casatools.version_string()}


TASKS = {"tclean": tclean, "columns": columns}

if __name__ == "__main__":
    task = json.load(open(sys.argv[1]))
    result = TASKS[task["task"]](task)
    json.dump(result, open(sys.argv[2], "w"), default=str)
