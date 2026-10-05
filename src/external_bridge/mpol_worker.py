"""Run inside MPoL's own environment (.venv-mpol), not ours.

    python mpol_worker.py task.json result.json

Evaluates MPoL's image losses (mpol.losses) and their autograd gradients on
given images, in float64.
"""

import json
import sys

import torch
from mpol import losses

torch.set_default_dtype(torch.float64)


def regularisers(task):
    out = []
    for im, prior in zip(task["images"], task["priors"]):
        cube = torch.tensor([im], requires_grad=True)  # (1, ny, nx)
        # MPoL's entropy takes a positive scalar prior and positive pixels
        p = torch.tensor(float(prior)) if prior is not None else None
        res = {}
        for name, fn in {
            **({"entropy": lambda c: losses.entropy(c, p, tot_flux=1.0)} if p is not None else {}),
            "tv": lambda c: losses.TV_image(c, epsilon=float(task["tv_epsilon"])),
            "tsv": losses.TSV,
        }.items():
            if cube.grad is not None:
                cube.grad = None
            value = fn(cube)
            value.backward()
            res[name] = float(value)
            res[name + "_grad"] = cube.grad[0].tolist()
        out.append(res)
    return {"results": out}


TASKS = {"regularisers": regularisers}

if __name__ == "__main__":
    task = json.load(open(sys.argv[1]))
    result = TASKS[task["task"]](task)
    import importlib.metadata as m

    result["mpol_version"] = m.version("mpol")
    json.dump(result, open(sys.argv[2], "w"))
