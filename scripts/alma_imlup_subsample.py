"""Read a strided subsample of the IM Lup continuum MS, and check its weights.

    python scripts/alma_imlup_subsample.py [--ms PATH] [--stride 1000] [--out DIR] [--sigma]

Uses our own reader (src/external_bridge/alma_worker.py, .venv-alma): one
.npz per ddid at stride ``--stride`` into ``--out`` (default
~/data/alma_imlup/subsample_1000/, never in git) and ``info.json`` (N_vis,
channels, channel width, integration time of every ddid).

``--sigma`` also computes the model-free weight check (crosscheck.alma.
successive_difference_sigma) for each ddid. Adjacent integrations are rows
apart, so this needs every row, not the strided subsample: each ddid is read
whole into a temporary directory, reduced to its statistics and deleted;
one ddid is in memory at a time. Writes ``sigma.json``.
"""

import argparse
import json
import pathlib
import shutil
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from crosscheck import alma  # noqa: E402
from external_bridge import _subprocess as sp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="~/data/alma_imlup/ms/IMLup_continuum.ms")
    ap.add_argument("--stride", type=int, default=1000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sigma", action="store_true")
    ap.add_argument("--ddids", type=int, nargs="*", default=None)
    a = ap.parse_args()
    ms = str(pathlib.Path(a.ms).expanduser())
    out = pathlib.Path(a.out or f"~/data/alma_imlup/subsample_{a.stride}").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    info = sp.run("alma", "alma_worker.py", {"task": "info", "path": ms, "ddids": a.ddids})
    (out / "info.json").write_text(json.dumps(info, indent=1))
    res = sp.run("alma", "alma_worker.py", {"task": "read", "path": ms, "out": str(out), "stride": a.stride,
                                             "ddids": a.ddids})
    (out / "read.json").write_text(json.dumps(res, indent=1))
    n = sum(f["n"] for f in res["files"])
    print(f"stride {a.stride}: {n} visibilities in {len(res['files'])} ddids -> {out}")
    if a.sigma:
        stats = []
        for row in info["ddids"]:
            d = row["ddid"]
            with tempfile.TemporaryDirectory() as tmp:
                r = sp.run("alma", "alma_worker.py", {"task": "read", "path": ms, "out": tmp, "stride": 1,
                                                       "ddids": [d]})
                if not r["files"]:
                    continue
                z = dict(np.load(r["files"][0]["file"]))
                s = alma.successive_difference_sigma(z)
                s.update(ddid=d, obsid=int(z["obsid"]), n_vis=int(len(z["u"])), t_int_s=float(z["t_int_s"]))
                stats.append(s)
                print(f"ddid {d:2d} obs {s['obsid']} n={s['n_vis']:>9d} pairs={s['n_pairs']:>8d} s={s['s']:.4f} +- {s['s2_err'] / (2 * s['s']):.4f}", flush=True)
                shutil.rmtree(tmp, ignore_errors=True)
        (out / "sigma.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
