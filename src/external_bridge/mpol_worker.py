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
  term;
* ``export_checkpoint``, ``objective_full``, ``asdf_vs_npz``: the reference run of the IM Lup example
  (``sgd.py`` run unmodified, ``docs/alma_conventions.md`` Part 4). They load a checkpoint written by
  ``sgd.py`` or an asdf written by ``casa_analysis/src/export_asdf.py`` and use the example's own
  ``Net``, ``loaddata`` and MPoL's losses from the pinned clone; nothing of their maths is rewritten here;
* ``export_fold_asdf``, ``score_fold``: MPoL's side of the held-out cross-validation on virgil's dartboard folds
  (``docs/alma_conventions.md`` Part 5). The first writes the asdf ``sgd.py`` is trained on for fold J (the points of the
  other folds, in the same visread layout, by flagging the held-out ones); the second scores a trained checkpoint on the
  held-out points.

Large arrays travel as ``.npz`` files named in the task.
"""

import importlib.util
import json
import pathlib
import sys
from unittest import mock

import numpy as np
import torch
from mpol import coordinates, losses, utils

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cv_folds  # noqa: E402  (numpy only, as npz_compare)
import npz_compare  # noqa: E402  (numpy only, next to this file, so a test can import it without MPoL)

torch.set_default_dtype(torch.float64)

# the example's own constants (sgd.py: `obsid_dict`, `GridCoords(cell_size=0.005, npix=1028)`, `--FWHM 0.03` in the
# paper run, entropy `prior_intensity=1e-4, tot_flux=0.253`, `--lam-ent 1e-9`)
OBSID_DICT = {0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9, 10], 2: [11, 12], 3: [13, 14], 4: [15, 16, 17], 5: [18, 19, 20, 21], 6: [22, 23, 24, 25]}
CELL_SIZE, NPIX, FWHM = 0.005, 1028, 0.03
ENTROPY_PRIOR, ENTROPY_FLUX, LAM_ENT = 1e-4, 0.253, 1e-9


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
    coords = coordinates.GridCoords(cell_size=task.get("cell_size", CELL_SIZE), npix=task.get("npix", NPIX))
    obsid_dict = {int(k): v for k, v in task.get("obsid_dict", OBSID_DICT).items()}
    return Net(coords, FWHM=task.get("fwhm", FWHM), obsid_dict=obsid_dict), mocked


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


# ----------------------------------------------------------------------------------------------
# the reference run: checkpoint export, full-data objective, asdf against our reader
# ----------------------------------------------------------------------------------------------


def _load_checkpoint(net, path):
    """Load ``sgd.py``'s checkpoint into ``net`` the way ``sgd.py`` does (``model_state_dict``,
    ``strict=False``), and return the raw state dict and the keys that did not match."""
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt  # a bare state dict (tests)
    res = net.load_state_dict(state, strict=False)
    return state, {"missing_keys": list(res.missing_keys), "unexpected_keys": list(res.unexpected_keys)}


def _image(net):
    """The example's forward chain without the NuFFT: base cube, Gaussian taper, image cube
    (``Net.forward`` lines 141-158). Returns the packed cube (1, npix, npix), Jy/arcsec^2."""
    return net.icube(net.conv_layer(net.bcube()))


def _north_up(packed):
    """MPoL packed layout -> rows North to South, columns East to West (what ``imshow`` shows with
    East left and North up). ``sky_cube`` has rows South to North, so its rows are reversed."""
    sky = utils.packed_cube_to_sky_cube(torch.as_tensor(packed))[0]
    return sky.flip(0).detach().cpu().numpy()


def _nuisances(net):
    """The checkpoint's nuisance parameters with the fixed observation put back in (value 0 in log10
    or arcsec), as ``adjust_amplitudes`` and ``adjust_positions`` do."""
    def insert(par, i):
        v = par.detach().cpu().numpy().astype(float)
        return np.insert(v, i, 0.0)

    log10_amp = insert(net.log10_amp_factors, net.fixed_amp_index)
    return dict(
        log10_amp=log10_amp,
        amp=10.0**log10_amp,
        log10_sigma=net.log10_sigma_factors.detach().cpu().numpy().astype(float),
        sigma=10.0 ** net.log10_sigma_factors.detach().cpu().numpy().astype(float),
        delta_ra_arcsec=insert(net.delta_ra, net.fixed_position_index),
        delta_dec_arcsec=insert(net.delta_dec, net.fixed_position_index),
    )


def export_checkpoint(task):
    """Checkpoint -> npz (arrays) and JSON (scalars).

    Rebuilds ``Net`` as the example does, loads ``model_state_dict``, recomputes the image from the
    base cube (``icube.packed_cube`` is stored in the checkpoint only after a forward pass, and in
    float32) and writes ``out_npz`` (keys in ``docs/alma_conventions.md`` Part 4). The JSON carries
    the scalars and ``analyze_scale_factors.py``'s per-observation mean of the sigma factors
    (``np.average`` of ``10**log10_sigma_factors`` over each observation's ddids, here in float64 from
    the checkpoint's float32 values), the image's flux and peak, and, when the checkpoint has its own
    ``icube.packed_cube``, how far the recomputed image is from it."""
    net, mocked = _net(task)
    state, load = _load_checkpoint(net, task["checkpoint"])
    obsid_dict = {int(k): v for k, v in task.get("obsid_dict", OBSID_DICT).items()}
    with torch.no_grad():
        packed = _image(net)
        base_image = net.bcube.pixel_mapping(net.bcube.base_cube)
    nuis = _nuisances(net)
    npix, cell = net.coords.npix, net.coords.cell_size
    # pixel centres of the north-up, East-left image: unpacked column c is l = (c - N/2) dx towards
    # East (mpol GridCoords.l_centers), and the output columns run East to West
    centres = (np.arange(npix) - npix // 2) * cell
    l_east, m_north = centres[::-1].copy(), centres[::-1].copy()
    image = _north_up(packed)
    arrays = dict(
        image=image, base_image=_north_up(base_image), base_cube=net.bcube.base_cube.detach().cpu().numpy()[0],
        l_east_arcsec=l_east, m_north_arcsec=m_north, cell_size_arcsec=np.array(cell), **nuis,
    )
    np.savez_compressed(task["out_npz"], **arrays)
    sigma = nuis["sigma"]
    stored = state.get("icube.packed_cube")
    out = {
        "npix": int(npix), "cell_size_arcsec": float(cell), "fwhm_arcsec": float(net.conv_layer.FWHM_maj),
        "flux_jy": float(image.sum() * cell**2), "peak_jy_arcsec2": float(image.max()), "min_jy_arcsec2": float(image.min()),
        "amp": nuis["amp"].tolist(), "log10_amp": nuis["log10_amp"].tolist(), "sigma": sigma.tolist(), "log10_sigma": nuis["log10_sigma"].tolist(),
        "delta_ra_arcsec": nuis["delta_ra_arcsec"].tolist(), "delta_dec_arcsec": nuis["delta_dec_arcsec"].tolist(),
        "delta_ra_mas": (1e3 * nuis["delta_ra_arcsec"]).tolist(), "delta_dec_mas": (1e3 * nuis["delta_dec_arcsec"]).tolist(),
        "sigma_eb_mean": {str(k): float(np.average(sigma[v])) for k, v in obsid_dict.items()},
        "fixed_amp_index": int(net.fixed_amp_index), "fixed_position_index": int(net.fixed_position_index),
        "stored_packed_cube_max_rel_diff": None if stored is None else float(np.max(np.abs(stored.numpy() - packed.numpy())) / np.max(np.abs(packed.numpy()))),
        "mocked": mocked, **load,
    }
    return out


def _asdf_points(path, examples_dir):
    """Everything the example's ``loaddata.get_ddid_data`` returns (its flag handling and its unit
    conversion, called as it is), plus, from a second pass over the same tree in the same order, the
    channel index, antennas and time of each kept point, which it discards."""
    import asdf

    sys.path.insert(0, f"{examples_dir}/04-IMLup-multi-EB/mpol_analysis/src")
    import loaddata

    uu, vv, data, weight, ddid = loaddata.get_ddid_data(path)
    chan, freq, ant1, ant2, time = [], [], [], [], []
    with asdf.open(path) as af:
        for obsid in af["obsids"].keys():
            for d_id in af["obsids"][obsid]["ddids"].keys():
                d = af["obsids"][obsid]["ddids"][d_id]
                ch, row = np.nonzero(~np.asarray(d["flag"]))  # row-major over (nchan, nvis): the order `[~flag]` returns
                chan.append(ch)
                freq.append(np.asarray(d["frequencies"], float)[ch])
                ant1.append(np.asarray(d["antenna1"])[row] if "antenna1" in d else np.full(len(ch), -1))
                ant2.append(np.asarray(d["antenna2"])[row] if "antenna2" in d else np.full(len(ch), -1))
                time.append(np.asarray(d["time"])[row] if "time" in d else np.full(len(ch), np.nan))
    cat = np.concatenate
    return dict(u=uu, v=vv, vis=data, weight=weight, ddid=ddid, chan=cat(chan), freq=cat(freq), ant1=cat(ant1), ant2=cat(ant2), time=cat(time))


def objective_full(task):
    """The example's objective summed over every visibility of an asdf, in float64 and batches.

    The model is the example's ``Net`` at the checkpoint's parameters; the image is computed once and
    the NuFFT (``sparse_matrices`` as the task says, True by default: MPoL's accurate option, the
    example's default interpolation table is 6e-4 off, P12) is evaluated per batch. Per batch the
    steps are those of ``sgd.py``'s ``train``: ``adjust_weights``, ``adjust_amplitudes``,
    ``adjust_positions``. -ln L is the sum over batches of ``-mpol.losses.log_likelihood`` (a sum over
    visibilities, so batches add); S_M is ``mpol.losses.entropy`` of the *tapered* image with the
    example's constants. chi2 is ``mpol.losses._chi_squared`` per ddid, with the fitted weights
    (``chi2_fit``, s_M and g applied) and with the quoted weights (``chi2_raw``, s = 1), the model
    (amplitude-scaled and shifted) the same in both."""
    batch = int(task.get("batch", 10**6))
    sparse = bool(task.get("sparse_matrices", True))
    device = torch.device(task.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    net, mocked = _net(task)
    _, load = _load_checkpoint(net, task["checkpoint"])
    net = net.to(device).eval()
    sys.path.insert(0, f"{task['examples_dir']}/04-IMLup-multi-EB/mpol_analysis/src")
    import loaddata

    uu, vv, data, weight, ddid = loaddata.get_ddid_data(task["asdf"], bool(task.get("sb_only", False)))
    n = len(uu)
    n_ddid = len(net.log10_sigma_factors)
    nll = torch.zeros((), dtype=torch.float64, device=device)
    chi2_fit = torch.zeros(n_ddid, dtype=torch.float64, device=device)
    chi2_raw = torch.zeros(n_ddid, dtype=torch.float64, device=device)
    count = torch.zeros(n_ddid, dtype=torch.float64, device=device)
    sum_ln_w = torch.zeros(n_ddid, dtype=torch.float64, device=device)
    with torch.no_grad():
        x = _image(net)
        entropy = losses.entropy(net.icube.packed_cube, prior_intensity=ENTROPY_PRIOR, tot_flux=ENTROPY_FLUX)
        for i in range(0, n, batch):
            sl = slice(i, i + batch)
            u_b, v_b = torch.as_tensor(uu[sl], dtype=torch.float64, device=device), torch.as_tensor(vv[sl], dtype=torch.float64, device=device)
            d_b = torch.as_tensor(data[sl], dtype=torch.complex128, device=device)
            w_b, dd_b = torch.as_tensor(weight[sl], dtype=torch.float64, device=device), torch.as_tensor(ddid[sl], device=device)
            vis = net.nufft(x, u_b, v_b, sparse_matrices=sparse)[0]
            w_adj = net.adjust_weights(dd_b, w_b)
            vis_s, w_s = net.adjust_amplitudes(dd_b, vis, w_adj)
            vis_sh = net.adjust_positions(dd_b, u_b, v_b, vis_s)
            nll = nll - losses.log_likelihood(vis_sh, d_b, w_s)
            r2 = (vis_sh - d_b).abs() ** 2
            chi2_fit.index_add_(0, dd_b, w_s * r2)
            chi2_raw.index_add_(0, dd_b, w_b * r2)
            sum_ln_w.index_add_(0, dd_b, torch.log(w_s))
            count.index_add_(0, dd_b, torch.ones_like(w_b))
    nll, entropy = float(nll), float(entropy)
    lam = float(task.get("lam_ent", LAM_ENT))
    cnt = count.cpu().numpy()
    c_fit, c_raw = chi2_fit.cpu().numpy(), chi2_raw.cpu().numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        per = lambda c: (c / (2 * cnt)).tolist()  # noqa: E731
    return {
        "n": int(n), "neg_log_likelihood": nll, "neg_log_likelihood_avg": nll / (2 * n), "entropy": entropy,
        "lam_ent": lam, "loss_mpol": nll / (2 * n) + lam * entropy,  # the example's loss, averaged over the data
        "objective_R": nll + 2 * n * lam * entropy,  # -ln L + 2N lam_M S_M (C4)
        "n_ddid": cnt.astype(int).tolist(), "chi2_fit": c_fit.tolist(), "chi2_raw": c_raw.tolist(),
        "chi2_fit_per_dof": per(c_fit), "chi2_raw_per_dof": per(c_raw),
        "chi2_fit_total": float(c_fit.sum()), "chi2_raw_total": float(c_raw.sum()),
        "sum_ln_weight_fit": sum_ln_w.cpu().numpy().tolist(),
        "batch": batch, "sparse_matrices": sparse, "device": str(device), "dtype": "float64", "mocked": mocked, **load,
    }


def _npz_ddid(npz_dir, d):
    z = np.load(f"{npz_dir}/ddid_{d:02d}.npz")
    vis = z["vis"] if "vis" in z else z["re"].astype(float) + 1j * z["im"].astype(float)
    return {k: np.asarray(z[k]) for k in z if k not in ("vis", "re", "im")} | {"vis": np.asarray(vis)}


def asdf_vs_npz(task):
    """MPoL's reading of an asdf (``loaddata.get_ddid_data``: flags, kilolambda-or-lambda conversion,
    channel flattening) against our per-ddid ``ddid_NN.npz`` (vv's reader). Per ddid: N, and, after
    matching the points, the maximum absolute and relative differences of u, v, vis and weight, and
    what a conjugation or a unit factor would look like (``vis_conj_max_abs``: the difference to
    conj(vis); ``u_flip_max_abs``: u against -u; ``u_ratio_median``). Reports only: nothing is
    changed to make the two agree.

    Matching: when both sides have channel, antennas and time (the stride-1000 subsample does; the
    full arrays may not), points are sorted by (channel rank, time, ant1, ant2) on both sides and
    ``key_mismatch`` counts the points whose keys then differ. The channel rank is the rank of the
    channel's own frequency (the asdf's ``frequencies``, our npz's ``freq``), ascending. visread 0.0.5 orders
    channels by decreasing frequency (``process_casa.get_channel_sorted_data`` flips an ascending MS), while our
    npz keeps the MS order, which is increasing for some ddids (4, 17, 19, 20, 23, 24 broke the old index match);
    no direction is assumed on either side, and our npz files stay as they are. Otherwise the order is assumed equal
    (``matching: "order"``); a different order shows as a large u difference.

    Test: per ddid ``pass`` is true when N agrees, every key matches, |dV| < ``tol_vis`` (default 1e-5 Jy),
    the per-point relative differences of u, v (against max(|u|, 1 lambda)) and of the weight are < ``tol_rel`` (default 1e-6).
    ``worst.all_pass`` is true when every ddid passes; a missing npz or a different N is a FAIL."""
    tol_vis, tol_rel = float(task.get("tol_vis", 1e-5)), float(task.get("tol_rel", 1e-6))
    pts = _asdf_points(task["asdf"], task["examples_dir"])
    out = {"ddids": {}}
    worst = {k: 0.0 for k in ("u_max_abs", "v_max_abs", "vis_max_abs", "weight_max_rel")}
    for d in sorted(int(x) for x in np.unique(pts["ddid"])):
        sel = pts["ddid"] == d
        a = {k: v[sel] for k, v in pts.items()}
        try:
            b = _npz_ddid(task["npz_dir"], d)
        except FileNotFoundError:
            out["ddids"][str(d)] = {"n_asdf": int(sel.sum()), "n_npz": None}
            continue
        res = {"n_asdf": int(sel.sum()), "n_npz": int(len(b["u"]))}
        if res["n_asdf"] != res["n_npz"]:
            out["ddids"][str(d)] = res
            continue
        res.update(npz_compare.compare_ddid(a, b, tol_vis, tol_rel))
        for k in worst:
            worst[k] = max(worst[k], res[k] if k in res else 0.0)
        out["ddids"][str(d)] = res
    # a ddid whose counts disagree (or with no npz) is skipped above, so say so in the summary
    worst["n_mismatch_ddids"] = [int(d) for d, r in out["ddids"].items() if r["n_npz"] != r["n_asdf"]]
    worst["key_mismatch_max"] = max([r.get("key_mismatch", 0) for r in out["ddids"].values()] + [0])
    for r in out["ddids"].values():
        r.setdefault("pass", False)
    worst["all_pass"] = all(r["pass"] for r in out["ddids"].values())
    out["tolerances"] = {"vis_max_abs_jy": tol_vis, "rel": tol_rel}
    out["worst"] = worst
    out["n_asdf_total"], out["n_npz_total"] = int(len(pts["u"])), int(sum(r["n_npz"] or 0 for r in out["ddids"].values()))
    return out


# ----------------------------------------------------------------------------------------------
# cross-validation on virgil's dartboard folds
# ----------------------------------------------------------------------------------------------


def _asdf_ddids(af):
    """``(obsid, ddid, node)`` of the asdf in the order ``loaddata`` reads it."""
    for obsid in af["obsids"].keys():
        for d_id in af["obsids"][obsid]["ddids"].keys():
            yield obsid, d_id, af["obsids"][obsid]["ddids"][d_id]


def export_fold_asdf(task):
    """Write the training asdf of one cross-validation fold: ``asdf`` (the example's export) with the points of fold
    ``fold`` flagged, so that ``loaddata`` (``uu[~flag]``) and hence the unmodified ``sgd.py`` see only the other folds.

    Task fields: ``asdf``, ``npz_dir`` (our per-ddid arrays, the order of virgil's fold file), ``folds`` (the file of
    ``alma_imlup_maxent.py folds``), ``fold`` (J), ``out_asdf``. Nothing else of the asdf changes (same tree, same arrays, same
    point order), so the points left are exactly those ``cv-task`` trains on. Each ddid's unflagged points are matched to the
    npz by (channel frequency rank, time, ant1, ant2) (``cv_folds.match_index``), then take the fold of the matched point; the
    match is checked (keys equal, u and v within ``tol_rel`` of the npz's, from the asdf's metres and the channel frequency, and
    the weight), so a wrong pairing raises instead of writing a wrong mask. Hermitian partners share a fold because the
    dartboard cell is symmetric under (u, v) -> (-u, -v): the fold file already has that.

    Returns the counts (``n_total``, ``n_train``, ``n_held``, per ddid), the fold file's sha256 and parameters, and the worst
    match differences. Memory is one ddid at a time plus the asdf tree (about 1.6 GB at full size)."""
    import asdf

    c_light = 299792458.0
    folds, meta = cv_folds.load_folds(task["folds"])
    j = int(task["fold"])
    if not 0 <= j < int(meta["k"]):
        raise ValueError(f"fold {j} is not in 0..{int(meta['k']) - 1}")
    tol_rel = float(task.get("tol_rel", 1e-6))
    per, worst = {}, dict(u_max_rel=0.0, v_max_rel=0.0, weight_max_rel=0.0)
    with asdf.open(task["asdf"], lazy_load=False) as af:
        for _, d_id, d in _asdf_ddids(af):
            flag = np.array(d["flag"], bool)
            ch, row = np.nonzero(~flag)
            freq = np.asarray(d["frequencies"], float)[ch]
            a = dict(freq=freq, ant1=np.asarray(d["antenna1"])[row], ant2=np.asarray(d["antenna2"])[row], time=np.asarray(d["time"], float)[row],
                     u=np.asarray(d["uu"], float)[row] * freq / c_light, v=np.asarray(d["vv"], float)[row] * freq / c_light,
                     weight=np.asarray(d["weight"], float)[row] if np.ndim(d["weight"]) == 1 else np.asarray(d["weight"], float)[ch, row])
            b = _npz_ddid(task["npz_dir"], int(d_id))
            fold, _, rep = cv_folds.labels_for_ddid(a, folds, int(d_id), b, tol_rel)
            held = fold == j
            flag[ch[held], row[held]] = True
            d["flag"] = flag
            per[str(d_id)] = dict(n=int(len(fold)), n_held=int(held.sum()), n_train=int((~held).sum()))
            for k in worst:
                worst[k] = max(worst[k], rep.get(k, 0.0))
        af.write_to(task["out_asdf"])
    out = dict(fold=j, n_total=sum(r["n"] for r in per.values()), n_held=sum(r["n_held"] for r in per.values()),
               n_train=sum(r["n_train"] for r in per.values()), by_ddid=per, folds_sha256=meta["sha256"], folds_params={k: meta[k] for k in ("k", "n_q", "n_phi", "seed")},
               worst_match=worst, tol_rel=tol_rel)
    if out["n_train"] + out["n_held"] != out["n_total"] or out["n_held"] == 0:
        raise ValueError(f"fold {j}: {out['n_held']} held out of {out['n_total']}")
    return out


def score_fold(task):
    """Held-out score of a trained checkpoint on fold ``fold`` of virgil's dartboard, in the convention of
    ``alma_imlup_maxent.py cv-task`` (``virgil_bridge.alma.score``).

    Task fields: ``checkpoint``, ``asdf`` (the *full* export, not the training one), ``npz_dir``, ``folds``, ``fold``, and as
    ``objective_full``: ``batch`` (10^6), ``sparse_matrices`` (true: MPoL's accurate NuFFT; the example's default table is
    6e-4 off, P12), ``device``. The held-out points are those of the asdf that match (``cv_folds``) points of the fold file with
    fold J. The model is the example's: image once, ``nufft`` per batch, ``adjust_amplitudes`` (V -> g V), ``adjust_positions``.
    With per-ddid ``s = g s_M`` (the ``w / (s_M g)^2`` of ``adjust_weights``/``adjust_amplitudes``) the per-point
    -ln L is ``w |V - M|^2 / (2 s^2) + 2 ln s + ln 2 pi - ln w``, as virgil's; the "s = 1" variant (``*_s1``) puts s = 1 with M unchanged.

    Result: ``heldout`` with the keys of ``cv.json``'s: ``nll`` (at the fitted s), ``chi2_s1`` (raw, quoted weights), ``chi2_fit``,
    ``by_ddid``, ``by_annulus`` (12 dartboard annuli: n, nll, chi2_raw, chi2_fit) and the same at s = 1 (``by_ddid_s1``,
    ``by_annulus_s1``); ``chi2_raw_per_dof`` (chi2_s1 / 2n) first; ``nll_mpol`` is the same sum from ``mpol.losses.log_likelihood`` on
    the adjusted weights, an independent check of ``nll``."""
    batch = int(task.get("batch", 10**6))
    sparse = bool(task.get("sparse_matrices", True))
    device = torch.device(task.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    folds, meta = cv_folds.load_folds(task["folds"])
    j = int(task["fold"])
    n_q = int(meta["n_q"])
    pts = _asdf_points(task["asdf"], task["examples_dir"])
    fold, annulus = np.zeros(len(pts["u"]), np.int8), np.zeros(len(pts["u"]), np.int8)
    worst = dict(u_max_rel=0.0, v_max_rel=0.0, weight_max_rel=0.0)
    for d in sorted(int(x) for x in np.unique(pts["ddid"])):
        sel = np.flatnonzero(pts["ddid"] == d)
        fold[sel], annulus[sel], rep = cv_folds.labels_for_ddid({k: v[sel] for k, v in pts.items()}, folds, d, _npz_ddid(task["npz_dir"], d),
                                                                float(task.get("tol_rel", 1e-6)))
        for k in worst:
            worst[k] = max(worst[k], rep.get(k, 0.0))
    held = np.flatnonzero(fold == j)
    if len(held) == 0:
        raise ValueError(f"fold {j} is empty")
    net, mocked = _net(task)
    _, load = _load_checkpoint(net, task["checkpoint"])
    net = net.to(device).eval()
    n_ddid = len(net.log10_sigma_factors)
    nll_mpol = 0.0
    fit = {name: dict(n=np.zeros(size), nll=np.zeros(size), chi2_raw=np.zeros(size), chi2_fit=np.zeros(size)) for name, size in
           (("ddid", n_ddid), ("annulus", n_q), ("ddid_s1", n_ddid), ("annulus_s1", n_q))}
    with torch.no_grad():
        x = _image(net)
        for i in range(0, len(held), batch):
            idx = held[i:i + batch]
            u_b, v_b = (torch.as_tensor(pts[k][idx], dtype=torch.float64, device=device) for k in ("u", "v"))
            d_b = torch.as_tensor(pts["vis"][idx], dtype=torch.complex128, device=device)
            w_b, dd_b = torch.as_tensor(pts["weight"][idx], dtype=torch.float64, device=device), torch.as_tensor(pts["ddid"][idx], device=device)
            vis = net.nufft(x, u_b, v_b, sparse_matrices=sparse)[0]
            w_adj = net.adjust_weights(dd_b, w_b)
            vis_s, w_s = net.adjust_amplitudes(dd_b, vis, w_adj)
            vis_sh = net.adjust_positions(dd_b, u_b, v_b, vis_s)
            nll_mpol -= float(losses.log_likelihood(vis_sh, d_b, w_s))
            res2 = (vis_sh - d_b).abs() ** 2
            r2 = (w_b * res2).cpu().numpy()  # quoted weights, the model with g applied
            r2_fit = (w_s * res2).cpu().numpy()  # /(g s_M)^2
            w_np, ws_np = w_b.cpu().numpy(), w_s.cpu().numpy()
            nll_fit = 0.5 * r2_fit - np.log(ws_np) + cv_folds.LN2PI  # = w r2 / (2 s^2) + 2 ln s + ln 2 pi - ln w, s = g s_M
            nll_s1 = 0.5 * r2 - np.log(w_np) + cv_folds.LN2PI
            dd_np, an_np = pts["ddid"][idx], annulus[idx]
            for name, ids, size, nl, rf in (("ddid", dd_np, n_ddid, nll_fit, r2_fit), ("annulus", an_np, n_q, nll_fit, r2_fit),
                                            ("ddid_s1", dd_np, n_ddid, nll_s1, r2), ("annulus_s1", an_np, n_q, nll_s1, r2)):
                t = cv_folds.tally(ids, size, nl, r2, rf)
                for k in t:
                    fit[name][k] += np.asarray(t[k], float)
    tl = {name: {k: v.tolist() for k, v in t.items()} for name, t in fit.items()}
    n = int(sum(tl["ddid"]["n"]))
    chi2_s1 = float(sum(tl["ddid"]["chi2_raw"]))
    ho = dict(nll=float(sum(tl["ddid"]["nll"])), chi2_s1=chi2_s1, chi2_fit=float(sum(tl["ddid"]["chi2_fit"])),
              by_ddid=tl["ddid"], by_annulus=tl["annulus"], by_ddid_s1=tl["ddid_s1"], by_annulus_s1=tl["annulus_s1"])
    ho["n"] = n
    ho["chi2_raw_per_dof"] = chi2_s1 / (2 * n)
    return dict(command="score-fold", fold=j, n_test=n, folds_sha256=meta["sha256"], folds_params={k: meta[k] for k in ("k", "n_q", "n_phi", "seed")},
                chi2_raw_per_dof=ho["chi2_raw_per_dof"], heldout=ho, nll_mpol=nll_mpol, worst_match=worst, batch=batch, sparse_matrices=sparse,
                device=str(device), dtype="float64", mocked=mocked, **load)


TASKS = {"regularisers": regularisers, "loglike": loglike, "objective": objective, "export_checkpoint": export_checkpoint, "objective_full": objective_full, "asdf_vs_npz": asdf_vs_npz,
         "export_fold_asdf": export_fold_asdf, "score_fold": score_fold}

if __name__ == "__main__":
    task = json.load(open(sys.argv[1]))
    result = TASKS[task["task"]](task)
    import importlib.metadata as m

    result["mpol_version"] = m.version("mpol")
    result["torch_version"] = torch.__version__
    result["torchkbnufft_version"] = m.version("torchkbnufft")
    for pkg in ("asdf", "visread", "numpy"):
        try:
            result[pkg + "_version"] = m.version(pkg)
        except m.PackageNotFoundError:
            pass
    json.dump(result, open(sys.argv[2], "w"))
