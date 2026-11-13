"""Run inside the BASIS environment (.venv-basis), not ours: Lai et al.'s own ring extraction.

    python m87_worker.py task.json result.json

Task ``ring_features``: for each image (ny, nx, row 0 North, column 0 East, as eht-imaging's
``imarr``), wrap it in an eht-imaging image of pixel size ``psize_uas`` and run ``RingFit`` from
``ringfitting_path`` (GenDIReCT's ``data/ringFitting.py`` for Table 1, or BASIS's
``BASIS/modules/ringFitting.py`` for ED Table 1) with the settings of GenDIReCT's notebook
(``interp_factor=2, blur_uas=0, total_flux=1``, ``run_all`` defaults). Their file is loaded by path
from the unpacked Zenodo archive; it is not copied here (MIT, Lai et al.).
"""

import importlib.util
import json
import sys

import ehtim as eh
import numpy as np


def _load(path):
    spec = importlib.util.spec_from_file_location("lai_ringfitting", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plain(v):
    """ufloat -> [value, std]; arrays -> lists; numbers -> float."""
    if hasattr(v, "nominal_value"):
        return [float(v.nominal_value), float(v.std_dev)]
    if isinstance(v, (tuple, list, np.ndarray)):
        return [_plain(x) for x in np.asarray(v, dtype=object).ravel()]
    return float(v)


def ring_features(task):
    rf = _load(task["ringfitting_path"])
    psize = float(task["psize_uas"]) * eh.RADPERUAS
    out = []
    for arr in task["images"]:
        arr = np.asarray(arr, float)
        n = arr.shape[0]
        img = eh.image.make_empty(n, n * psize, 187.70593075, 12.39112330, rf=230e9, source="M87")
        img.imvec = arr.ravel()
        fit = rf.RingFit(img, interp_factor=int(task.get("interp_factor", 2)), blur_uas=0, total_flux=1)
        fit.run_all(verbose=False)
        out.append({k: _plain(v) for k, v in fit.get_results().items()})
    return {"results": out}


if __name__ == "__main__":
    task = json.load(open(sys.argv[1]))
    result = {"ring_features": ring_features}[task["task"]](task)
    import importlib.metadata as m

    result["ehtim_version"] = m.version("ehtim")
    json.dump(result, open(sys.argv[2], "w"))
