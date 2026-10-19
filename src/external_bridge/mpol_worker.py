"""Run inside MPoL's own environment (.venv-mpol), not ours.

    python mpol_worker.py task.json result.json

Evaluates, in float64:

* ``regularisers``: MPoL's image regularisers (mpol.losses) and their autograd
  gradients on given images;
* ``loglike``: the visibility log likelihood (``mpol.losses.log_likelihood``
  and ``neg_log_likelihood_avg``) after the IM Lup example's own weight and
  amplitude adjustments;
* ``objective``: the IM Lup multi-EB example's forward model and loss, with
  autograd gradients. The model is the example's own ``Net``
  (MPoL-dev/examples, 04-IMLup-multi-EB, ``mpol_analysis/src/sgd.py``),
  imported unmodified (``forward``, lines 141-158); the loss is assembled as
  in its ``train`` (lines 286-302 at the pinned commit), without the entropy
  term.

Large arrays travel as ``.npz`` files named in the task.
"""

import importlib.util
import json
import sys
from unittest import mock

import numpy as np
import torch
from mpol import coordinates, losses

torch.set_default_dtype(torch.float64)


def load_net(examples_dir):
    """The example's ``Net`` class, imported unmodified from ``sgd.py``.

    sgd.py imports plotting, tensorboard, asdf and visread at module level.
    The model uses none of them and none is installed here, so each missing
    one is replaced by a mock and the file imports as written. Returns the
    class and the names that were mocked.
    """
    src = f"{examples_dir}/04-IMLup-multi-EB/mpol_analysis/src"
    sys.path.insert(0, src)
    # plotting and logging only (mpol.plot needs matplotlib; torch's tensorboard
    # raises ImportError, not ModuleNotFoundError, when tensorboard is absent)
    mocked = ["mpol.plot", "torch.utils.tensorboard"]
    for name in mocked:
        sys.modules[name] = mock.MagicMock()
    for _ in range(40):
        try:
            spec = importlib.util.spec_from_file_location("imlup_sgd", f"{src}/sgd.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.Net, mocked
        except ModuleNotFoundError as err:
            mocked.append(err.name)
            parts = err.name.split(".")
            for i in range(1, len(parts) + 1):
                sys.modules.setdefault(".".join(parts[:i]), mock.MagicMock())
    raise RuntimeError("could not import sgd.py")


def _net(task):
    Net, mocked = load_net(task["examples_dir"])
    coords = coordinates.GridCoords(cell_size=task["cell_size"], npix=task["npix"])
    obsid_dict = {int(k): v for k, v in task["obsid_dict"].items()}
    return Net(coords, FWHM=task["fwhm"], obsid_dict=obsid_dict), mocked


def _complex(z, name):
    return torch.tensor(z[name + "_re"] + 1j * z[name + "_im"])


def _set(net, z):
    with torch.no_grad():
        net.log10_sigma_factors.copy_(torch.tensor(z["log10_sigma"]))
        net.log10_amp_factors.copy_(torch.tensor(z["log10_amp"]))
        if "delta_ra" in z:
            net.delta_ra.copy_(torch.tensor(z["delta_ra"]))  # arcsec
            net.delta_dec.copy_(torch.tensor(z["delta_dec"]))
        if "base_cube" in z:
            net.bcube.base_cube.copy_(torch.tensor(z["base_cube"]))  # packed layout, softplus domain


def loglike(task):
    """``log_likelihood`` and ``neg_log_likelihood_avg`` of data against a
    model, after the example's per-ddid sigma factors and per-observation
    amplitude factors (``Net.adjust_weights``, ``Net.adjust_amplitudes``)."""
    z = np.load(task["npz"])
    net, mocked = _net(task)
    _set(net, z)
    ddid, weight = torch.tensor(z["ddid"]), torch.tensor(z["weight"])
    model, data = _complex(z, "model"), _complex(z, "data")
    model_s, w_s = net.adjust_amplitudes(ddid, model, net.adjust_weights(ddid, weight))
    return {
        "log_likelihood": float(losses.log_likelihood(model_s, data, w_s)),
        "neg_log_likelihood_avg": float(losses.neg_log_likelihood_avg(model_s, data, w_s)),
        "chi2": float(losses._chi_squared(model_s, data, w_s)),
        "n": int(data.numel()),
        "mocked": mocked,
    }


def objective(task):
    """The example's forward model and loss at given parameters.

    The steps of ``train`` in sgd.py: ``model(uu, vv)[0]``, ``adjust_weights``,
    ``adjust_amplitudes``, ``adjust_positions``, then
    ``neg_log_likelihood_avg``. Returns -ln L = 2N * loss and, unless the task
    says ``"grad": false``, its autograd gradients, in ``out_npz``.
    """
    z = np.load(task["npz"])
    net, mocked = _net(task)
    _set(net, z)
    uu, vv, ddid = torch.tensor(z["u"]), torch.tensor(z["v"]), torch.tensor(z["ddid"])
    data, weight = _complex(z, "data"), torch.tensor(z["weight"])
    vis = net(uu, vv)[0]
    vis_scaled, w_scaled = net.adjust_amplitudes(ddid, vis, net.adjust_weights(ddid, weight))
    vis_shifted = net.adjust_positions(ddid, uu, vv, vis_scaled)
    nll = losses.neg_log_likelihood_avg(vis_shifted, data, w_scaled) * (2 * data.numel())
    out = {
        "vis_raw_re": vis.detach().numpy().real, "vis_raw_im": vis.detach().numpy().imag,
        "vis_model_re": vis_shifted.detach().numpy().real, "vis_model_im": vis_shifted.detach().numpy().imag,
        "weight_used": w_scaled.detach().numpy(), "neg_log_likelihood": np.array(float(nll)),
        "image_packed": net.icube.packed_cube.detach().numpy(),
    }
    with torch.no_grad():
        # the same model through the NuFFT's own more accurate option (the
        # example uses the default table interpolation): attributes the
        # difference from an exact transform to the interpolation
        raw_sparse = net.nufft(net.icube.packed_cube, uu, vv, sparse_matrices=True)[0]
        scaled_sparse, _ = net.adjust_amplitudes(ddid, raw_sparse, net.adjust_weights(ddid, weight))
        shifted_sparse = net.adjust_positions(ddid, uu, vv, scaled_sparse)
    out["vis_model_sparse_re"], out["vis_model_sparse_im"] = shifted_sparse.numpy().real, shifted_sparse.numpy().imag
    if task.get("grad", True):
        nll.backward()
        out.update(
            grad_base_cube=net.bcube.base_cube.grad.numpy(), grad_log10_sigma=net.log10_sigma_factors.grad.numpy(),
            grad_log10_amp=net.log10_amp_factors.grad.numpy(), grad_delta_ra=net.delta_ra.grad.numpy(),
            grad_delta_dec=net.delta_dec.grad.numpy(),
        )
    np.savez(task["out_npz"], **out)
    return {"neg_log_likelihood": float(nll), "n": int(data.numel()), "mocked": mocked, "taper_dtype": str(net.conv_layer.taper_2D.dtype)}


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


TASKS = {"regularisers": regularisers, "loglike": loglike, "objective": objective}

if __name__ == "__main__":
    task = json.load(open(sys.argv[1]))
    result = TASKS[task["task"]](task)
    import importlib.metadata as m

    result["mpol_version"] = m.version("mpol")
    result["torch_version"] = torch.__version__
    result["torchkbnufft_version"] = m.version("torchkbnufft")
    json.dump(result, open(sys.argv[2], "w"))
