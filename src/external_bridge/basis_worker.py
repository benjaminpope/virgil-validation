"""Run inside the BASIS environment (.venv-basis), not ours: Lai et al.'s own BASIS likelihood and models.

    python basis_worker.py task.json result.json

BASIS (Lai et al. 2026, MIT; BASIS v0.1.0) is loaded from the venv; nothing of it is copied here.

Tasks
-----
``parameters``
    BASIS's parameter table of the model list: names, bounds, fixed flags, default values.
``invariants``
    Load a UVFITS with eht-imaging as ``runBilby.py`` does, build ``ModelLikelihood(models, ..., dterms={"ci": 100})``
    and return its closure-invariant data and the sigma of the likelihood (a Monte Carlo estimate, seeded here, which
    BASIS does not seed) for one noise setting, each triangle's label (scan time in MJD seconds, station ids =
    indices of ``obs.tarr``, in the order of ``calc_advariants``), the model invariants at each given parameter
    dictionary, and an alignment check of the triangle table that BASIS evaluates the *model* on
    (``obs.bispec``) against the one its *data* come from (``tlist`` loop).
``render_check``
    The orientation of ``sky_map`` against BASIS's analytic visibilities: for the dihedral transforms T of the
    array, the DFT of T(sky_map) (row 0 North, column 0 East, V = sum I exp(-2 pi i (u x + v y)), x East, y North)
    against the analytic visibility; the best T and the residual. Also the floor Gaussian's centroid in the render
    against the one implied by the analytic visibility's phase.
``posterior_renders``
    Read a bilby result JSON, render ``n`` posterior samples with ``sky_map`` on the given grid, orient them by the
    ``render_check`` transform (row 0 North, column 0 East), and save ``native`` and ``sky`` arrays to an npz.
"""

import json
import sys

import numpy as np

RAD_PER_UAS = np.pi / 180 / 3600 / 1e6
MODELS = ["xsringauss", "gauss", "gauss"]


def _numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _model(models=None, dim=128, fov=225):
    from BASIS.models import base

    return base.BaseModel(model_list=models or MODELS, dim=dim, fov=fov, randomise_params=False)


def _full_params(model, overrides=None):
    p = dict(model.params)
    if overrides:
        unknown = set(overrides) - set(p)
        if unknown:
            raise KeyError(f"not BASIS parameters: {sorted(unknown)[:5]}")
        p.update({k: float(v) for k, v in overrides.items()})
    return p


def parameters(task):
    model = _model(task.get("models"))
    return {
        "names": list(model.params),
        "limits": {k: [float(a) for a in v] for k, v in model.param_limits.items()},
        "fixed": {k: bool(v) for k, v in model.param_fixed.items()},
        "defaults": {k: float(v) for k, v in model.params.items()},
    }


def _triangle_labels(like):
    """(time MJD s, ref, a, b) station indices of each triangle in the order of ``calc_advariants``."""
    import itertools as it

    obs = like.obs
    names = list(obs.tarr["site"])
    index = {n: i for i, n in enumerate(names)}
    labels = []
    for tdata in obs.tlist(conj=True):
        sites = list(set(np.hstack((tdata["t1"], tdata["t2"]))))
        have = {(d["t1"], d["t2"]) for d in tdata}
        for ref, a, b in like.tri_minimal_set(sites, obs.tarr, obs.tkey):
            if (ref, a) in have and (a, b) in have and (b, ref) in have:
                t = (float(obs.mjd) + float(tdata["time"][0]) / 24.0) * 86400.0
                labels.append((t, index[ref], index[a], index[b]))
    return labels, names


def _alignment(like, labels, names):
    """Whether the bispectrum table the model is evaluated on matches the data's triangle list.

    ``obs.bispec`` holds (u1, v1, u2, v2, u3, v3) per triangle; the likelihood's model invariants use these rows in
    order against data built from the ``tlist`` loop. Compare the loop's baselines (legs ref-a, a-b, b-ref with
    tlist's conjugated entries) with the table row by row.
    """
    obs = like.obs
    tab = obs.bispec
    out = {"n_loop": len(labels), "n_bispec": int(len(tab))}
    if len(tab) != len(labels):
        out["aligned"] = False
        return out
    uv = {}
    for tdata in obs.tlist(conj=True):
        for d in tdata:
            uv[(round(float(d["time"]), 9), d["t1"], d["t2"])] = (float(d["u"]), float(d["v"]))
    worst = 0.0
    tdiff = 0.0
    for row, (t, r, a, b) in zip(tab, labels):
        th = (t / 86400.0 - float(obs.mjd)) * 24.0
        tdiff = max(tdiff, abs(float(row["time"]) - th) * 3600.0)
        legs = [(names[r], names[a]), (names[a], names[b]), (names[b], names[r])]
        for k, (s1, s2) in enumerate(legs, 1):
            got = uv.get((round(float(row["time"]), 9), s1, s2))
            if got is None:
                worst = np.inf
                continue
            worst = max(worst, abs(got[0] - float(row[f"u{k}"])) + abs(got[1] - float(row[f"v{k}"])))
    out.update({"aligned": bool(worst < 1.0 and tdiff < 1.0), "max_uv_diff_wavelengths": float(worst),
                "max_time_diff_s": float(tdiff)})
    return out


def _model_ci64(like, params):
    """Model invariants in float64: BASIS's own component functions (``MODEL_LIST[...].sample_vis``, which are
    float64 for float64 uv) summed in complex128, then its numpy ``advariants_to_ci``. ``BaseModel.sample_vis``
    itself accumulates in complex64, so the likelihood BASIS evaluates is float32; this is the reference for an
    exact comparison of the mapping."""
    from BASIS.models.base import MODEL_LIST, _params_for_model, _resolve_model_name

    model = like.model
    model.params = params
    uv = like.uv["ci"]  # (3, 2, N) float64 from obs.bispec
    vis = []
    for leg in range(3):
        total = np.zeros(uv.shape[2], dtype=complex)
        for ind, name in enumerate(model.model_list):
            mp = {k.split("_", 1)[1]: v for k, v in model.params.items() if k.startswith(f"{ind}_")}
            assert set(mp) == set(_params_for_model(name))
            inst = MODEL_LIST[_resolve_model_name(name)](dim=model.dim, fov=model.fov, **mp)
            total += _numpy(inst.sample_vis(np.ascontiguousarray(uv[leg], dtype=np.float64))).reshape(-1).astype(complex)
        vis.append(total)
    adv = vis[0] * np.conj(vis[1]) ** (-1) * vis[2]
    return like.advariants_to_ci(adv, like.obs.bispec["time"])[0]


def invariants(task):
    import ehtim as eh
    import torch
    import BASIS.modules.likelihood as lk

    if task.get("double"):
        torch.set_default_dtype(torch.float64)
    np.random.seed(int(task.get("seed", 0)))
    obs = eh.obsdata.load_uvfits(task["uvfits"])
    like = lk.ModelLikelihood(
        model_names=task.get("models", MODELS), obs=obs, imgdim=int(task.get("imgdim", 128)),
        fov=float(task.get("fov", 225)), dterms={"ci": 100}, static_noise_floor=float(task.get("floor", 0.0)),
        noise_frac=float(task.get("noise_frac", 0.0)), noise_factor=float(task.get("noise_factor", 1.0)),
    )
    like.sample_data(None, dtype="ci")  # fills the cache: data and sigma
    data, sigma = like.cached_data["ci"], like.cached_noise["sigmaci"]
    labels, names = _triangle_labels(like)
    models, models64 = [], []
    for override in task.get("params", []):
        _, _, m = like.sample_data(_full_params(like.model, override), dtype="ci")
        models.append(_numpy(m).tolist())
        models64.append(_model_ci64(like, _full_params(like.model, override)).tolist())
    return {
        "data": np.asarray(data).tolist(), "sigma": np.asarray(sigma).tolist(), "model": models, "model64": models64,
        "labels": [list(map(float, row)) for row in labels], "stations": names, "mjd": float(obs.mjd),
        "n_triangles": len(labels), "alignment": _alignment(like, labels, names),
        "noise": {k: float(task.get(k, d)) for k, d in (("noise_frac", 0.0), ("floor", 0.0), ("noise_factor", 1.0))},
        "loglike": [float(like.log_likelihood(_full_params(like.model, o))) for o in task.get("params", [])],
    }


def _dft(img, uv, fov_uas):
    """V = sum I exp(-2 pi i (u x + v y)), image rows/columns as (row 0 North, column 0 East), uv in wavelengths."""
    n = img.shape[0]
    ps = fov_uas / (n - 1) * RAD_PER_UAS  # BASIS grid is linspace(-fov/2, fov/2, n): pixel centres, spacing fov/(n-1)
    x = ((n - 1) / 2 - np.arange(n)) * ps  # East positive at column 0
    y = ((n - 1) / 2 - np.arange(n)) * ps  # North positive at row 0
    phase = np.exp(-2j * np.pi * (uv[0][:, None, None] * x[None, None, :] + uv[1][:, None, None] * y[None, :, None]))
    return (img[None] * phase).sum(axis=(1, 2))


def _dihedral(img):
    out = {}
    for k in range(4):
        r = np.rot90(img, k)
        out[f"rot{90 * k}"] = r
        out[f"rot{90 * k}_flipud"] = r[::-1]
    return out


def _apply(img, name):
    return _dihedral(img)[name]


def _best_transform(model, params, uv, fov, dim):
    """The dihedral transform T with DFT(T(sky_map)) closest to the analytic visibility, and the residuals."""
    model.params = params
    analytic = _numpy(model.sample_vis(uv))
    sky = _numpy(model.sky_map())
    scale = np.abs(analytic).max()
    res = {name: float(np.abs(_dft(img, uv, fov) - analytic).max() / scale) for name, img in _dihedral(sky).items()}
    best = min(res, key=res.get)
    return best, res


def render_check(task):
    rng = np.random.default_rng(int(task.get("seed", 0)))
    dim, fov = int(task.get("dim", 128)), float(task.get("fov", 225))
    model = _model(task.get("models"), dim, fov)
    base = _full_params(model)
    # a generic ring + thick Gaussian, no floor, no nuisance, no blur: the geometry alone
    base.update({"0_floor_brightness": 0.0, "0_bg_gauss_flux": 0.0, "0_gauss_blur_sigma": 0.0, "1_I0": 1e-9,
                 "2_I0": 1e-9, "0_I0": 1.0, "0_phi": 0.7, "0_ecn": 0.5, "0_gq": 0.4, "0_Rp": 24.0})
    base.update({k: float(v) for k, v in task.get("params", {}).items()})
    uv = rng.uniform(-3e9, 3e9, (2, 30))
    best, res = _best_transform(model, base, uv, fov, dim)
    # floor Gaussian only: centroid in the render vs the one the analytic phase implies
    floor = dict(base)
    floor.update({"0_I0": 1e-9, "0_gq": 0.0, "0_floor_brightness": 1.0, "0_floor_sigma": 8.0})
    model.params = floor
    sky = _apply(_numpy(model.sky_map()), best)
    n = dim
    ps = fov / (n - 1)
    ax = ((n - 1) / 2 - np.arange(n)) * ps
    tot = sky.sum()
    cx, cy = float((sky.sum(0) * ax).sum() / tot), float((sky.sum(1) * ax).sum() / tot)  # East, North (uas)
    uvs = np.array([[1e7, 0.0, -1e7, 0.0], [0.0, 1e7, 0.0, -1e7]])  # (2, N): u row, v row (not (2, 2): ambiguous)
    an = _numpy(model.sample_vis(uvs))
    ax_c = -np.angle(an[0]) / (2 * np.pi * 1e7) / RAD_PER_UAS
    ay_c = -np.angle(an[1]) / (2 * np.pi * 1e7) / RAD_PER_UAS
    d = floor["0_ecn"] * (floor["0_Rp"] - floor["0_Rn_frac"] * floor["0_Rp"])
    return {"best_transform": best, "residual_by_transform": res, "floor_offset_expected_uas": float(d),
            "floor_centroid_render_east_north_uas": [cx, cy], "floor_centroid_analytic_east_north_uas": [float(ax_c), float(ay_c)],
            "phi": floor["0_phi"]}


def posterior_renders(task):
    import bilby

    dim, fov = int(task.get("dim", 128)), float(task.get("fov", 225))
    n = int(task.get("n", 200))
    chk = render_check({"seed": 0, "dim": dim, "fov": fov, "models": task.get("models")})
    transform = task.get("transform", chk["best_transform"])
    res = bilby.result.read_in_result(filename=task["result"])
    post = res.posterior
    rng = np.random.default_rng(int(task.get("seed", 0)))
    pick = rng.choice(len(post), size=min(n, len(post)), replace=False)
    model = _model(task.get("models"), dim, fov)
    names = set(model.params)
    native, sky = [], []
    for i in pick:
        row = {k: float(v) for k, v in post.iloc[int(i)].items() if k in names}
        model.params = _full_params(model, row)
        img = _numpy(model.sky_map())
        native.append(img)
        sky.append(_apply(img, transform))
    np.savez(task["out"], native=np.array(native), sky=np.array(sky), index=pick, transform=transform,
             pixel_uas=fov / (dim - 1), fov_uas=fov)
    cols = [c for c in post.columns if c in names]
    return {"out": task["out"], "n": len(pick), "transform": transform, "render_check": chk,
            "log_likelihood_max": float(post["log_likelihood"].max()) if "log_likelihood" in post else None,
            "log_evidence": float(res.log_evidence), "log_evidence_err": float(res.log_evidence_err),
            "median": {c: float(post[c].median()) for c in cols}, "sd": {c: float(post[c].std()) for c in cols}}


def dump_uvfits(task):
    """eht-imaging's reading of a UVFITS as arrays (time in MJD seconds, station indices of ``obs.tarr``, u, v in
    wavelengths, vis, sigma), saved to ``out`` (npz), to compare with the measurement set CASA makes of it."""
    import ehtim as eh

    obs = eh.obsdata.load_uvfits(task["uvfits"])
    d = obs.data
    index = {n: i for i, n in enumerate(obs.tarr["site"])}
    np.savez(task["out"], time=(float(obs.mjd) + d["time"] / 24.0) * 86400.0,
             t1=np.array([index[s] for s in d["t1"]]), t2=np.array([index[s] for s in d["t2"]]),
             u=d["u"], v=d["v"], vis=d["vis"], sigma=d["sigma"], stations=np.array(obs.tarr["site"]),
             rf_hz=float(obs.rf))
    return {"out": task["out"], "n": int(len(d)), "stations": list(obs.tarr["site"]), "mjd": float(obs.mjd),
            "rf_hz": float(obs.rf), "polrep": str(obs.polrep)}


TASKS = {"parameters": parameters, "invariants": invariants, "render_check": render_check,
         "posterior_renders": posterior_renders, "dump_uvfits": dump_uvfits}

if __name__ == "__main__":
    task = json.load(open(sys.argv[1]))
    result = TASKS[task["task"]](task)
    import importlib.metadata as m

    result["bilby_version"] = m.version("bilby")
    result["ehtim_version"] = m.version("ehtim")
    json.dump(result, open(sys.argv[2], "w"))
