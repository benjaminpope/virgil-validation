"""Data source for the prototypes: the real Gl 229 nights, or (env INJ=path.json) an injection made by inject.py."""
import os, sys, json, pickle, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]  # the virgil-validation checkout
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src"), str(pathlib.Path(__file__).resolve().parent)]
import reanalyse_orbits as R

TAG = os.environ.get("TAG", "gl229")


def load():
    inj = os.environ.get("INJ")
    L = R.load_gl229(os.environ.get("ESO_BIN_ROOT", os.path.expanduser("~/data/eso_binaries")))
    if inj:
        L.data = pickle.load(open(inj.replace(".json", ".pkl"), "rb"))
        L.grid_data = L.data
    if os.environ.get("CPONLY"):
        L.data = [d.select(observables="phi") for d in L.data]
        L.grid_data = L.data
    return L


def reference():
    inj = os.environ.get("INJ")
    if not inj or json.load(open(inj))["truth"]["flux"] == 0:  # null injection: compare with the real orbit (ghost check)
        return R.XUAN_2024
    return json.load(open(inj))["truth"]
