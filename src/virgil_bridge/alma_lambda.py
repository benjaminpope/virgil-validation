"""The pre-registered regulariser-weight rule of the IM Lup deconvolutions (``lambda_selection`` in ``design/alma_imlup_gate_criteria.yml``).

Stage ``sigma``: an unregularised fit (``regularizers=[]``) to C8 fixes the per-ddid error factors s_k^2 = chi^2_k / N_k, once. Stage ``sweep``: with those
factors folded into the data weights, ``virgil.imaging.l_curve`` fits one arm (MEM on a flat prior, or StarletL1) at the log-grid weights, each to C8.
Stage ``final``: ``LCurve.discrepancy`` log-interpolates the weight at which the raw chi^2/N is 1, and the image is refitted there. A grid point is
``w = 10^(k / per_decade)``, so a point is a shard (arm, k); every point is its own directory and a finished one is never refitted.

The grid, the arms and the crop are read from the registered block, never branched on a target. Layout of ``out``::

    sigma.json, sigma_values.npz            stage sigma
    sweep/<arm>/k+040/point.json, values.npz   one grid point
    final/<arm>/lambda_final.json, values.npz, image_raw.npy, image_blur.npy
"""

import hashlib
import json
import os
import pathlib
import re
import time
import warnings

import numpy as np

from . import alma as va
from . import alma_start as vs

ARM_LABEL = {"mem": "MEM", "starlet_l1": "StarletL1"}  # the names the registered grid text uses
MAX_STEPS, MAX_ROUNDS = 10**7, 10**4  # "no step cap": C8 or a stalled optimiser ends a point


def rule():
    """The registered ``lambda_selection`` block with its grid parsed: ``per_decade`` and, per arm, the ``k`` range ``(lo, hi)`` of ``w = 10^(k / per_decade)``."""
    block = dict(vs.CRITERIA["lambda_selection"])
    text = block["grid"]
    per = int(re.search(r"(\d+) points per decade", text).group(1))
    ks = {}
    for arm in block["arms"]:
        lo, hi = re.search(rf"{ARM_LABEL[arm]} 1e(-?\d+)-1e(-?\d+)", text).groups()
        ks[arm] = (per * int(lo), per * int(hi))
    return dict(block, per_decade=per, k_range=ks)


def weight(k, per_decade):
    return float(10.0 ** (k / per_decade))


def regulariser(arm, w):
    """The arm's virgil regulariser on the unblurred pixels at weight ``w`` (``l_curve`` sets the weight itself)."""
    from virgil.imaging import MaxEntropy, StarletL1

    if arm == "mem":
        return MaxEntropy(float(w), prior=None, smoothed=False)
    if arm == "starlet_l1":
        return StarletL1(float(w), smoothed=False)
    raise ValueError(f"arm must be one of {list(ARM_LABEL)}, not {arm!r}")


def fixed_sigma_data(data, s):
    """``data`` with the weights divided by s_k^2 of each ddid (the fixed error factors enter here, so the fit has no sigma sites)."""
    import equinox as eqx
    import jax.numpy as jnp

    s = np.asarray(s, float)
    return eqx.tree_at(lambda d: d.weight, data, jnp.asarray(np.asarray(data.weight) / s[np.asarray(data.ddid)] ** 2))


def build(data, *, npix, pixel_mas, ref_obs=0, chunk_size=va.CHUNK):
    """Track A's forward model on the crop: config F, Toeplitz chi^2, per-obs gains and offsets, flux; no sigma sites and no regularisers (``prob.regs`` is empty)."""
    prob = va.build_problem(data, "F", lam=0.0, npix=npix, pixel_mas=pixel_mas, ref_obs=ref_obs, chunk_size=chunk_size, chi2="toeplitz", reg="mem:1", reg_n=len(data.u), sigmas=False)
    prob.regs = []
    return prob


def start_from_scratch(prob, flux=0.25, fwhm_mas=1000.0):
    """The cold start of the p3 workflow (C9): a 1" Gaussian raised to MPoL's entropy floor so that every pixel can move."""
    import jax.numpy as jnp

    v = va.start_values(prob, flux=flux, fwhm_mas=fwhm_mas)
    eta, flux0, _ = va.floor_start(v, prob.npix, prob.pixel_mas, va.MPOL["prior"], log_pb=va.beam_log(prob.npix, prob.pixel_mas))
    return {**v, "log_brightness": jnp.asarray(eta), "cal.flux": jnp.asarray(flux0)}


def term_chi2(term, values):
    """Raw chi^2 (on the data weights of the term, constants included) at ``values``: the Toeplitz term's ``chi2_and_log_norm`` plus its ``chi2_constant`` (the data-only part it drops)."""
    import jax.numpy as jnp

    return float(term.chi2_and_log_norm({k: jnp.asarray(v) for k, v in values.items()})[0] + term.chi2_constant)


def one_point_fit(arm, w, held):
    """A ``virgil.fit`` stand-in for ``va.run_fit_c8`` that routes every round through ``l_curve`` at the single weight ``w``; the last ``LCurve`` is kept in ``held``."""
    from virgil.imaging import l_curve

    def fit(model, priors, data, *, regularizers=(), **kw):
        held["lc"] = lc = l_curve(model, priors, data, regulariser(arm, w), [w], others=list(regularizers), warm_start=True, **kw)
        return lc.results[0]

    return fit


def run_c8(prob, init, *, fit_fn=None, time_limit=None, gtol=1e-4, log=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return va.run_fit_c8(prob, init, max_steps=MAX_STEPS, time_limit=time_limit, log=log, gtol=gtol, max_rounds=MAX_ROUNDS, fit_fn=fit_fn)


def c8_record(traj, why, info):
    """vv's own C8 recompute from the loss history (virgil's ``converged`` flag is erratic)."""
    v = va.convergence(traj, None)
    return dict(c8_converged=bool(v["converged"]) and (v["window_steps"] or 0) >= 100, c8=v, stopped_because=why, steps=int(info["steps"]), rounds=int(info["rounds"]), loss=float(info["loss"]))


def sigma_stage(data, *, npix, pixel_mas, ref_obs=0, chunk_size=va.CHUNK, time_limit=None, gtol=1e-4, log=print):
    """``(record, values)``: the unregularised fit and s_k^2 = chi^2_k / N_k of each ddid (N_k = 2 points: real and imaginary part). A ddid with no points gets s = 1."""
    prob = build(data, npix=npix, pixel_mas=pixel_mas, ref_obs=ref_obs, chunk_size=chunk_size)
    t = time.time()
    values, info, traj, why = run_c8(prob, start_from_scratch(prob), time_limit=time_limit, gtol=gtol, log=log)
    sc = va.score(data, prob.scene, values, prob.nuis, chunk_size=chunk_size)
    n_k = 2.0 * np.asarray(sc["by_ddid"]["n"], float)
    chi2_k = np.asarray(sc["by_ddid"]["chi2_raw"], float)
    s = np.sqrt(np.where(n_k > 0, chi2_k / np.maximum(n_k, 1), 1.0))
    rec = dict(sigma=s, chi2_k=chi2_k, n_k=n_k, chi2_over_N=float(chi2_k.sum() / n_k.sum()), n_real=float(n_k.sum()), n_pix=int(npix) ** 2, overfit_bias=float(npix) ** 2 / float(n_k.sum()),
               npix=int(npix), pixel_mas=float(pixel_mas), flux_jy=float(values["cal.flux"]), n_vis=len(data.u), wall_s=time.time() - t, **c8_record(traj, why, info))
    return rec, values


def term_scores(data, prob, values, chunk_size):
    sc = va.score(data, prob.scene, values, prob.nuis, chunk_size=chunk_size)
    return sc["chi2_raw"], int(sc["n"])


def point_stage(prob, arm, w, init, *, k=None, time_limit=None, gtol=1e-4, log=print):
    """``(record, values)`` of one fit of ``arm`` at weight ``w`` (grid index ``k``, None for the final refit) to C8 from ``init``. ``chi2_over_N`` is the term's raw chi^2 per real number (fixed s_k)."""
    held, t = {}, time.time()
    values, info, traj, why = run_c8(prob, init, fit_fn=one_point_fit(arm, w, held), time_limit=time_limit, gtol=gtol, log=log)
    lc = held["lc"]
    chi2 = term_chi2(prob.terms[0], values)
    n_real = 2.0 * prob.chi2_info["n_points"]
    rec = dict(arm=arm, k=None if k is None else int(k), weight=w, chi2=chi2, n_real=n_real, chi2_over_N=chi2 / n_real, penalty=float(lc.penalty[0]), virgil_chi2=float(lc.chi2[0]), flux_jy=float(values["cal.flux"]), wall_s=time.time() - t, dots=dots_of(values, prob),
               **c8_record(traj, why, info))
    return rec, values


def lcurve_of(records):
    """An ``LCurve`` of the grid points' records (largest weight first), for virgil's own ``discrepancy`` and ``corner``."""
    from virgil.imaging import LCurve

    r = sorted(records, key=lambda x: -x["weight"])
    return LCurve(np.array([x["weight"] for x in r]), np.array([x["chi2"] for x in r]), np.array([[x["chi2_over_N"]] for x in r]), np.array([x["penalty"] for x in r]), [])


def bracketed(records):
    """``(below, above)``: whether the chi^2/N = 1 crossing is bracketed from the weak end (some point <= 1) and from the strong end (some point > 1)."""
    c = np.array([x["chi2_over_N"] for x in records])
    return bool((c <= 1).any()), bool((c > 1).any())


def interpolate(records, target=1.0):
    """The weight at chi^2/N = ``target``: ``LCurve.discrepancy`` (linear in log w between the bracketing points), or None."""
    return lcurve_of(records).discrepancy(target=target)


def beam_blur(image, pixel_mas, beam, fraction=0.5):
    """``image`` blurred to ``fraction`` of the CLEAN beam ``(bmaj_mas, bmin_mas, pa_deg)``."""
    from crosscheck import alma_maxent as cx

    return cx.convolve(image, pixel_mas, beam[0] * fraction, beam[1] * fraction, beam[2])


# ---------------------------------------------------------------- the stages on disk


def point_dir(out, arm, k):
    return pathlib.Path(out) / "sweep" / arm / f"k{k:+04d}"


def save_values(path, values):
    """Written to a temporary file and renamed, so that a shard reading its neighbour never sees half a file."""
    path = pathlib.Path(path)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **{k: np.asarray(v) for k, v in values.items()})
    os.replace(tmp, path)


def sigma_hash(sigma):
    """Identifies the fixed s_k vector a point was fitted with."""
    return hashlib.sha256(json.dumps([float(x) for x in sigma]).encode()).hexdigest()[:16]


def load_values(path):
    import jax.numpy as jnp

    z = np.load(path)
    return {k: jnp.asarray(z[k]) for k in z.files}


def load_sigma(sigma_dir):
    """``(record, values)`` of stage ``sigma``."""
    d = pathlib.Path(sigma_dir)
    return json.loads((d / "sigma.json").read_text()), load_values(d / "sigma_values.npz")


def flat_start(values):
    """The start of a point with no neighbour on disk: a flat image (as the fast loop did) with the nuisances of the unregularised fit, which serves the s_k and nothing else of the image (its dark pixels underflow, where MaxEntropy's gradient is NaN)."""
    import jax.numpy as jnp

    return {**values, "log_brightness": jnp.zeros_like(values["log_brightness"])}


def dots_of(values, prob):
    """The dots metrics of the point's image (``va.dots_report`` against its MPoL-beam taper): N_eff ratio, in-disk dead fraction (base below 1e-4 of its peak), super-resolution (sub-taper) power."""
    image = va.image_of(values, prob.npix, prob.pixel_mas, va.beam_log(prob.npix, prob.pixel_mas))
    d = va.dots_report(image, va.tapered_image(image, prob.pixel_mas), prob.pixel_mas)
    return {k: d.get(k) for k in ("n_eff_ratio", "base_dead_in_disk_1e4", "sub_taper_power", "hole_fraction", "peak_ratio")}


def ensure_point(prob, out, arm, k, sigma_values, per_decade, write_json, *, sigma, force=False, **kw):
    """The record of grid point ``(arm, k)`` under ``out``, fitted now unless it is there (a cached point must have been fitted with these s_k: ``sigma``). Starts from the stronger neighbour, else the weaker, else a flat image (``flat_start``); the record says which (``start_from``), since it depends on which shards have finished. ``point.json`` is written after ``values.npz`` and marks the point done."""
    d = point_dir(out, arm, k)
    h = sigma_hash(sigma)
    if (d / "point.json").exists() and not force:
        rec = json.loads((d / "point.json").read_text())
        if rec.get("sigma_hash") != h:
            raise SystemExit(f"{d}/point.json was fitted with other s_k (hash {rec.get('sigma_hash')}, sigma.json has {h}); rerun it with --force, or use a new --out")
        return rec
    src = next((j for j in (k + 1, k - 1) if (point_dir(out, arm, j) / "values.npz").exists()), None)
    start = flat_start(sigma_values) if src is None else load_values(point_dir(out, arm, src) / "values.npz")
    d.mkdir(parents=True, exist_ok=True)
    rec, values = point_stage(prob, arm, weight(k, per_decade), start, k=k, **kw)
    rec.update(start_from="flat" if src is None else int(src), sigma_hash=h)
    save_values(d / "values.npz", values)
    write_json(d / "point.json", rec)
    return rec


def setup(data, sigma_dir, *, npix, pixel_mas, ref_obs=0, chunk_size=va.CHUNK):
    """``(problem on the fixed-sigma data, fixed-sigma data, sigma record, unregularised values)``."""
    sig, values = load_sigma(sigma_dir)
    if not sig["c8_converged"]:
        raise SystemExit(f"{sigma_dir}/sigma.json: the sigma stage did not reach C8 (stopped because {sig['stopped_because']}); its s_k are too large and the rule would have no crossing. Rerun stage sigma")
    if (sig["npix"], sig["pixel_mas"]) != (int(npix), float(pixel_mas)):
        raise SystemExit(f"{sigma_dir}: sigma factors were fixed on a {sig['npix']}^2 crop at {sig['pixel_mas']} mas, not {npix}^2 at {pixel_mas}")
    data_s = fixed_sigma_data(data, sig["sigma"])
    return build(data_s, npix=npix, pixel_mas=pixel_mas, ref_obs=ref_obs, chunk_size=chunk_size), data_s, sig, values


def sweep_stage(data, out, sigma_dir, arm, ks, write_json, *, npix, pixel_mas, ref_obs=0, chunk_size=va.CHUNK, force=False, time_limit=None, gtol=1e-4, log=print):
    """Fit the grid points ``ks`` of ``arm`` (default the registered range, strong to weak, each warm-started from the one before)."""
    r = rule()
    prob, _, sig, sv = setup(data, sigma_dir, npix=npix, pixel_mas=pixel_mas, ref_obs=ref_obs, chunk_size=chunk_size)
    ks = sorted(ks if ks else range(r["k_range"][arm][0], r["k_range"][arm][1] + 1), reverse=True)
    return [ensure_point(prob, out, arm, k, sv, r["per_decade"], write_json, sigma=sig["sigma"], force=force, time_limit=time_limit, gtol=gtol, log=log) for k in ks]


def final_stage(data, out, sigma_dir, arm, write_json, *, npix, pixel_mas, beam=None, ref_obs=0, chunk_size=va.CHUNK, max_extend=3, time_limit=None, gtol=1e-4, log=print):
    """Complete the registered grid of ``arm`` (fitting any point missing), extend it by a decade at a time while the chi^2/N = 1 crossing is not bracketed
    (``max_extend`` decades at most; each extension recorded), interpolate the weight with ``LCurve.discrepancy``, refit there and report. Returns the record."""
    r = rule()
    per, (lo, hi) = r["per_decade"], r["k_range"][arm]
    prob, data_s, sig, sv = setup(data, sigma_dir, npix=npix, pixel_mas=pixel_mas, ref_obs=ref_obs, chunk_size=chunk_size)
    kw = dict(time_limit=time_limit, gtol=gtol, log=log)
    point = lambda k: ensure_point(prob, out, arm, k, sv, per, write_json, sigma=sig["sigma"], **kw)  # noqa: E731
    recs = {k: point(k) for k in range(hi, lo - 1, -1)}
    extensions, status, w = [], "ok", None
    while True:
        w = interpolate(list(recs.values()))
        if w is not None:
            break
        chi = np.array([x["chi2_over_N"] for x in recs.values()])
        side = "weak" if not (chi <= 1).any() else "strong" if not (chi > 1).any() else None
        if side is None or len(extensions) >= max_extend:
            status = "no_crossing"
            break
        edge = min(recs) if side == "weak" else max(recs)
        new = [edge - i for i in range(1, per + 1)] if side == "weak" else [edge + i for i in range(1, per + 1)]
        extensions.append(dict(side=side, k=new, reason=f"chi^2/N at the {side} edge is {chi[np.argmin(list(recs)) if side == 'weak' else np.argmax(list(recs))]:.6g}"))
        log(f"extending the grid by a decade at the {side} end: k = {new[0]}..{new[-1]}")
        recs.update({k: point(k) for k in new})
    table = [{k: v for k, v in x.items() if k not in ("c8",)} for _, x in sorted(recs.items(), reverse=True)]
    if w is not None:  # the interpolation rests on the two points that bracket chi^2/N = 1: both must have reached C8
        order = sorted(recs, reverse=True)  # strong to weak
        pair = next((a, b) for a, b in zip(order, order[1:]) if recs[a]["chi2_over_N"] > 1 >= recs[b]["chi2_over_N"])
        if not all(recs[k]["c8_converged"] for k in pair):
            status = "unconverged_bracket"
    try:
        corner = lcurve_of(list(recs.values())).corner()
    except (ValueError, FloatingPointError):
        corner = None
    res = dict(arm=arm, status=status, selected_weight=w, corner_weight=corner, corner_note="reported only: it does not select", grid_k=sorted(recs, reverse=True), per_decade=per, extensions=extensions, max_extend=max_extend, max_extend_note="a cap of this driver, not in the registered block", unconverged_points=sorted((k for k, x in recs.items() if not x["c8_converged"]), reverse=True), points=table,
               registered_grid=dict(k_range=[lo, hi]), sigma=sig["sigma"], npix=int(npix), pixel_mas=float(pixel_mas))
    if w is None:
        return res
    near = min(recs, key=lambda k: abs(np.log(weight(k, per) / w)))
    rec, values = point_stage(prob, arm, w, load_values(point_dir(out, arm, near) / "values.npz"), **kw)
    d = pathlib.Path(out) / "final" / arm
    d.mkdir(parents=True, exist_ok=True)
    save_values(d / "values.npz", values)
    chi2_s, n_s = term_scores(data_s, prob, values, chunk_size)  # direct term at eps 1e-12, the fixed s_k
    chi2_1, n_1 = term_scores(data, prob, values, chunk_size)  # the quoted weights, s_k = 1
    image = va.image_of(values, prob.npix, prob.pixel_mas, va.beam_log(prob.npix, prob.pixel_mas))
    np.save(d / "image_raw.npy", image.astype(np.float32))
    blur = None
    if beam is not None:
        blur = beam_blur(image, prob.pixel_mas, beam)
        np.save(d / "image_blur.npy", blur.astype(np.float32))
    if not rec["c8_converged"]:
        res["status"] = "unconverged_final"
    res["final"] = dict(rec, chi2_over_N_fixed_s=chi2_s / (2 * n_s), chi2_over_N_s1=chi2_1 / (2 * n_1), flux_jy=float(values["cal.flux"]), image_flux_jy=float(image.sum() * (prob.pixel_mas / 1000.0) ** 2),
                        beam_mas_deg=None if beam is None else list(beam), blur_fraction=0.5 if beam is not None else None, peak_raw=float(image.max()), peak_blur=None if blur is None else float(blur.max()))
    return res
