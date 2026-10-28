"""Dump numbers of the existing R and F paths for a before/after comparison (PR 197): run once per source tree, then ``--compare A.json B.json``.

    PYTHONPATH=<tree>/src:<tree>/scripts python scripts/alma_equiv_dump.py --out DIR --json FILE     # run from <tree> = origin/main's checkout, then this branch's (tests/ and scripts/ come from the cwd tree)
    python scripts/alma_equiv_dump.py --compare MAIN.json BRANCH.json

Toy data (``tests/test_alma_maxent.synthetic_dir``, which both trees have), fixed random parameter vectors. Dumps: config R's ``objective_R`` and its parts at eps 1e-12 (``r_objective``), config F's
-ln L (``score``) and MaxEntropy value (``reg.value``) with direct chi^2, and ``cv-task`` (R and F, plain ``--lam``) cv.json values from the same folds and ``--init``.
"""

import argparse
import json
import pathlib
import sys

import numpy as np


def dump(out):
    sys.path.insert(0, str(pathlib.Path.cwd() / "tests"))
    import jax.numpy as jnp
    from test_alma_maxent import _call, synthetic_dir

    import virgil_bridge.alma as va

    out = pathlib.Path(out)
    syn = synthetic_dir(out / "syn")
    data = va.load_dir(syn)
    n, res = len(data.u), {}
    rng = np.random.default_rng(5)
    for cfg, npix in (("R", 48), ("F", 48)):
        prob = va.build_problem(data, cfg, lam=va.lam_r(n) if cfg == "R" else 3e3, npix=npix, pixel_mas=5.0)
        v = {k: jnp.asarray(x) * (1 + 0.01 * rng.normal(size=np.shape(x))) if k in ("cal.gain", "cal.sigma") else jnp.asarray(x) for k, x in prob.init.items()}
        v["cal.offset"] = jnp.asarray(0.3 * rng.normal(size=np.shape(prob.init["cal.offset"])))
        v["log_brightness"] = jnp.asarray(rng.normal(size=(npix, npix)))
        with va.ft_eps(1e-12):
            sc = va.score(data, prob.scene, v, prob.nuis)
        r = dict(nll=sc["nll"], chi2_raw=sc["chi2_raw"], chi2_fit=sc["chi2_fit"])
        if cfg == "R":
            o = va.r_objective(data, v, lam=va.lam_r(n), pixel_mas=5.0)
            r.update(objective_R=o["objective"], **{k: o[k] for k in o if isinstance(o[k], float)})
            r["entropy_term"] = float(prob.terms[1].entropy(v))
        else:
            r["reg_value"] = float(prob.regs[0].value(prob.scene(v)))
        res[cfg] = r
    folds = out / "folds.npz"
    _call("folds", "--data", syn, "--out", folds, "--k", 2)
    for cfg, lam in (("R", None), ("F", 3e3)):
        init = out / f"init_{cfg}"
        _call("fit", "--config", cfg, *(["--lam", lam] if lam else []), "--data", syn, "--out", init, "--npix", 48, "--max-steps", 8, "--no-dirty")
        for lam_cv in ((None,) if cfg == "R" else (1e2, 1e4)):
            o = out / f"cv_{cfg}_{lam_cv}"
            _call("cv-task", "--config", cfg, *(["--lam", lam_cv] if lam_cv else []), "--data", syn, "--folds", folds, "--fold", 0, "--init", init, "--npix", 48, "--max-steps", 8, "--out", o)
            c = json.loads((o / "cv.json").read_text())
            res[f"cv_{cfg}_{lam_cv}"] = dict(lam=c["lam"], lam_train=c["lam_train"], n_train=c["n_train"], n_test=c["n_test"], steps=c["steps"], heldout=c["heldout"], train={k: c["train"][k] for k in ("nll", "chi2_raw", "chi2_fit") if k in c["train"]})
    return res


def flat(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, f"{prefix}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flat(v, f"{prefix}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield prefix, float(x)


def compare(a, b):
    fa, fb = dict(flat(a)), dict(flat(b))
    only = sorted(set(fa) ^ set(fb))
    worst = []
    for k in sorted(set(fa) & set(fb)):
        if np.isnan(fa[k]) and np.isnan(fb[k]):
            continue
        worst.append((abs(fa[k] - fb[k]) / max(abs(fa[k]), abs(fb[k]), 1e-300), k, fa[k], fb[k]))
    worst.sort(reverse=True)
    return only, worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--json")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        only, worst = compare(*(json.loads(pathlib.Path(p).read_text()) for p in a.compare))
        print(f"{len(worst)} numbers compared; keys in one file only: {only}")
        for r, k, x, y in worst[:8]:
            print(f"  max rel diff {r:.3e}  {k}: {x!r} vs {y!r}")
        sys.exit(1 if only or (worst and worst[0][0] > 1e-9) else 0)
    pathlib.Path(a.json).write_text(json.dumps(dump(a.out), indent=1, default=float))


if __name__ == "__main__":
    main()
