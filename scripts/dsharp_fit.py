"""The production virgil MaxEnt fit of one DSHARP / HL Tau target (GPU), PR-A of ``reports/dsharp-convergence-plan.md`` (virgil-paper, rev 2).

    dsharp_fit.py MODE --target KEY --data DIR --out DIR [--info info.json] [--shat DIR] [--probe probe.json] [options]

MODE is ``probe`` (stage S1: no image fit; the gate G1 numbers), or one of the run modes ``s3`` (certification pilot), ``s4`` (diagnosis) and ``s5``
(production). ``--data`` holds ``ddid_NN.npz`` (the vv format of ``external_bridge/alma_worker.py read``).

The production model (plan section 3, 5.1): ``virgil.LinearImage`` in sqrt-pixels, the pixels own the flux (``fit_flux=False``), classic MEM
(``MaxEntropy`` on the unblurred pixels, flat default), a flat start, fitted per-observation gains and spread offsets, error scales fixed at 1 on weights
rescaled by the model-free pair scatter, the spectral index fixed (decision 8), and the chi^2 engine set explicitly to ``toeplitz`` and read back from the term.
Never ``chi2="auto"``: it falls back to the direct engine with fitted offsets and would change the cost silently.

``probe`` writes ``probe.json``: the weights (s^2 per window, and per window x baseline bin where it varies by more than 20%), the model-free spectral
index, each observation's model-free shift against an offsets-only fit and the pre-alignment rule's verdict, the agreement of the Toeplitz and direct
engines on the production model, and the timings. The run modes read it (``--probe``) and apply its weights and shifts, so one record decides everything.

Run modes fit the lambda ladder (strongest first, each rung warm-started from the last), the held-out folds (fold 0 is the test fold and tunes nothing;
folds 1-3 tune, one left out each) and pick lambda by the one-standard-error rule. Every round checkpoints (``values.npz``) and the result records both commits.
Certification (the tests of plan section 5.3 and checks C1-C6) is PR-B's ``alma_certify``: ``--certify module:function`` plugs it in; without it a result says
``certified: null``. The optimiser is behind ``OPTIMISERS`` (``--optimiser name`` or ``module:function``); the default is virgil's L-BFGS until S2 picks.
"""

import argparse
import importlib
import json
import os
import pathlib
import re
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck import dsharp as ds  # noqa: E402

# ---------------------------------------------------------------------------------------------------- constants (plan sections 3, 5.2, decisions 1-3, 8)
LADDER = (1e6, 3e5, 1e5, 3e4, 1e4, 3e3)  # w/s^2, strongest first; s = 1 after the rescale, so this is w. The last is about the inherited value
K_FOLDS, TEST_FOLD, TUNING_FOLDS = 4, 0, (1, 2, 3)
BOX_MAS = 20.0  # the Toeplitz offsets' box
AGREE_MAS = 5.0  # model-free shift and offsets-only fit agree within this: pre-align
BOX_EDGE = 0.05  # a residual offset within this fraction of the box is reported as near it (alma_certify C6)
BIN_VARIATION = 0.20  # s^2 varying by more than this across baseline bins of a window: rescale per bin
N_BINS = 4
ALPHA_SIGMA = 2.0  # decision 8: fix alpha at the model-free value when it differs from 0 by more than this many sigma
SHIFT_QCUT = 4e5  # wavelengths: below this a 50 mas shift is under 0.5 rad of phase, so the phase slope needs no unwrapping
SHIFT_CELL = 5e3  # wavelengths: the uv cell in which two observations' visibilities are compared
MAS = np.pi / 180.0 / 3600.0 / 1000.0  # radians per mas
CONVERGED_WHY = ("c8", "converged")  # an optimiser stop that is not a cap; any other stop is a failed fit
CHI2_ENGINES = ("direct", "toeplitz")  # never "auto"
GATE_LOSS_REL, GATE_GRAD_REL, GATE_HVP_COST = 1e-10, 1e-8, 4.0  # G1


def say(*a):
    print(*a, flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def heaviest_obs(obs, weight):
    """The observation with the largest total weight: the gauge (gain 1, offset 0) of the fit."""
    ids = np.unique(obs)
    return int(ids[np.argmax([float(weight[obs == o].sum()) for o in ids])])


def provenance(env=None):
    """The commits under test, as AGENTS.md requires of a result: virgil's (the read-only snapshot a job pinned, whose directory is named for the commit,
    else the installed one) and virgil-validation's (the checkout's, else the job's ``PIN_COMMIT``)."""
    from evidence import meta

    env = os.environ if env is None else env
    src = dict(meta.virgil_source())
    pinned = env.get("PIN_VIRGIL_SRC")
    if pinned and re.fullmatch(r"[0-9a-f]{40}", pathlib.Path(pinned).parent.name):
        src = dict(commit=pathlib.Path(pinned).parent.name, url=pinned, dirty=False)  # a snapshot is read-only
    return dict(virgil_commit=src.get("commit"), virgil_dirty=src.get("dirty"), virgil_source=src.get("url"),
                validation_commit=meta.own_commit() or env.get("PIN_COMMIT"), pins=env.get("PIN_DESC"))


# ---------------------------------------------------------------------------------------------------- the data: one dict per ddid file
def load_arrays(directory):
    """The ``ddid_NN.npz`` files of ``directory`` as dicts (the form ``virgil_bridge.alma.load_npz`` takes), so weights and visibilities can be
    adjusted before virgil sees them."""
    files = sorted(pathlib.Path(directory).expanduser().glob("ddid_*.npz"))
    if not files:
        raise FileNotFoundError(f"no ddid_*.npz in {directory}")
    return [dict(np.load(f)) for f in files]


def _vis(z):
    return np.asarray(z["vis"]) if "vis" in z else np.asarray(z["re"], float) + 1j * np.asarray(z["im"], float)


def _set_vis(z, vis):
    if "vis" in z:
        z["vis"] = vis
    else:
        z["re"], z["im"] = vis.real, vis.imag


def obs_of(zs):
    """The observation id of each file."""
    return np.array([int(z["obsid"]) for z in zs])


def pair_scatter(z):
    """The time-adjacent pair differences of one ddid file: ``(q, z2)`` per pair, with ``q`` the baseline length (wavelengths) and
    ``z2 = (dRe^2 + dIm^2) / (2 (1/w_i + 1/w_j))``, whose mean is the model-free s^2 (virgil ``sigma_scale``). Samples of one baseline, channel and scan are
    paired in time order without overlap; the result needs ``time``, ``ant1`` and ``ant2`` (``chan`` and ``scan`` are used when present), else None."""
    if any(k not in z for k in ("time", "ant1", "ant2")):
        return None
    n = len(z["u"])
    zeros = np.zeros(n, int)
    keys = [np.asarray(z.get(k, zeros)) for k in ("ant1", "ant2", "chan", "scan")]
    order = np.lexsort((np.asarray(z["time"]), keys[3], keys[2], keys[1], keys[0]))
    key = np.stack(keys)[:, order]
    first = np.concatenate([[True], np.any(key[:, 1:] != key[:, :-1], axis=0)])
    gid = np.cumsum(first) - 1
    pos = np.arange(n) - np.flatnonzero(first)[gid]
    i = np.flatnonzero((pos % 2 == 0) & (np.arange(n) + 1 < n))
    i = i[gid[np.minimum(i + 1, n - 1)] == gid[i]]
    a, b = order[i], order[i + 1]
    vis, w = _vis(z), np.asarray(z["weight"], float)
    d = vis[a] - vis[b]
    z2 = (d.real**2 + d.imag**2) / (2.0 * (1.0 / w[a] + 1.0 / w[b]))
    q = np.hypot(np.asarray(z["u"], float)[a], np.asarray(z["v"], float)[a])
    return q, z2, np.asarray(z.get("scan", zeros))[a]


def weight_plan(zs, shat=None, n_bins=N_BINS, variation=BIN_VARIATION):
    """The model-free weight scales: per window (ddid) s^2 from the pair scatter, and its spread over ``n_bins`` baseline bins (equal pair counts).
    Where the bins differ by more than ``variation`` (max - min over mean) ``per_bin`` is set and the rescale is per (window, bin). Without the pair columns
    the window's ``shat`` (``--shat``: ``ddid_NN.json``) stands in and nothing is binned. ŝ^2 is an upper bound (sky or phase changing inside a pair
    adds to it), exact for constant signal."""
    plan = {}
    for z in zs:
        d = int(z["ddid"])
        ps = pair_scatter(z)
        if ps is None:
            if shat is None or d >= len(shat):
                raise ValueError(f"ddid {d}: no time/ant1/ant2 in its npz and no --shat to take the model-free scale from")
            plan[d] = dict(s2=float(shat[d]) ** 2, n_pairs=None, source="shat", per_bin=False, edges=[], s2_bins=[], by_scan={})
            continue
        q, z2, scan = ps
        edges = np.quantile(q, np.linspace(0, 1, n_bins + 1)[1:-1])
        b = np.searchsorted(edges, q)
        s2_bins = np.array([z2[b == k].mean() if np.any(b == k) else np.nan for k in range(n_bins)])
        s2 = float(z2.mean())
        spread = float(np.nanmax(s2_bins) - np.nanmin(s2_bins)) / s2
        by_scan = {int(s): float(z2[scan == s].mean()) for s in np.unique(scan)}
        plan[d] = dict(s2=s2, n_pairs=int(len(z2)), source="pairs", per_bin=bool(spread > variation), bin_spread=spread, edges=edges.tolist(),
                       s2_bins=s2_bins.tolist(), by_scan=by_scan)
    return plan


def apply_weights(zs, plan):
    """Rescale each window's weights by its s^2 (per bin where ``plan`` says): ``w' = w / s^2``, so the fit takes s = 1."""
    out = []
    for z in zs:
        z = dict(z)
        p = plan[int(z["ddid"])]
        w = np.asarray(z["weight"], float)
        if p["per_bin"]:
            b = np.searchsorted(np.asarray(p["edges"]), np.hypot(np.asarray(z["u"], float), np.asarray(z["v"], float)))
            z["weight"] = w / np.asarray(p["s2_bins"])[b]
        else:
            z["weight"] = w / p["s2"]
        out.append(z)
    return out


# ---------------------------------------------------------------------------------------------------- spectral index (decision 8)
def short_baseline_flux(z, fraction=0.01, qmax=None):
    """``(flux, error)`` of one window: the weighted mean of Re V on the baselines shorter than ``qmax`` wavelengths (default: the shortest ``fraction`` of
    them) and its standard error. ``spectral_index`` passes one ``qmax`` for every window, so each selects the same angular scale of a resolved source."""
    q = np.hypot(np.asarray(z["u"], float), np.asarray(z["v"], float))
    k = np.argsort(q)[: max(1, int(len(q) * fraction))] if qmax is None else np.flatnonzero(q <= qmax)
    if len(k) == 0:
        k = np.argsort(q)[:1]
    w = np.asarray(z["weight"], float)[k]
    return float(np.sum(w * _vis(z).real[k]) / w.sum()), float(1.0 / np.sqrt(w.sum()))


def spectral_index(zs):
    """The model-free spectral index alpha (flux ~ nu^alpha) and its error from the windows' short-baseline fluxes, with a free level per observation
    (the gains differ between observations): a weighted least squares of ln F on ln(nu / nu0) with one intercept per observation. Needs two windows in
    one observation. Returns ``dict(alpha, sigma, n_windows, windows, used, alpha_fixed)``; ``alpha_fixed`` is alpha when it differs from 0 by more than
    ``ALPHA_SIGMA`` sigma, else 0.0 (decision 8)."""
    rows = []
    # one q range in wavelengths for every window (the largest of the windows' shortest-1% cuts): Re V falls with q on a resolved disc, and a cut that follows
    # each window's own baselines would turn any trend of q with frequency into a spurious slope (review of PR 293)
    qmax = max(float(np.quantile(np.hypot(np.asarray(z["u"], float), np.asarray(z["v"], float)), 0.01)) for z in zs)
    for z in zs:
        f, e = short_baseline_flux(z, qmax=qmax)
        nu = float(np.average(np.asarray(z["freq"], float), weights=np.asarray(z["weight"], float)))
        rows.append((int(z["obsid"]), int(z["ddid"]), nu, f, e))
    nu0 = float(np.mean([r[2] for r in rows]))
    obs = sorted({r[0] for r in rows})
    ok = [r for r in rows if r[3] > 0]
    windows = [dict(obs=r[0], ddid=r[1], freq_hz=r[2], flux_jy=r[3], error_jy=r[4]) for r in rows]
    if not any(sum(x[0] == o for x in ok) > 1 for o in obs):
        return dict(alpha=None, sigma=None, n_windows=len(rows), windows=windows, nu0_hz=nu0, used=False, alpha_fixed=0.0, note="no observation has two windows")
    X = np.zeros((len(ok), 1 + len(obs)))
    y, s = np.zeros(len(ok)), np.zeros(len(ok))
    for k, r in enumerate(ok):
        X[k, 0], X[k, 1 + obs.index(r[0])] = np.log(r[2] / nu0), 1.0
        y[k], s[k] = np.log(r[3]), r[4] / r[3]
    W = 1.0 / s**2
    A = X.T @ (X * W[:, None])
    cov = np.linalg.pinv(A)
    beta = cov @ (X.T @ (W * y))
    alpha, sigma = float(beta[0]), float(np.sqrt(cov[0, 0]))
    use = abs(alpha) > ALPHA_SIGMA * sigma
    return dict(alpha=alpha, sigma=sigma, n_windows=len(rows), windows=windows, nu0_hz=nu0, used=bool(use), alpha_fixed=alpha if use else 0.0)


# ---------------------------------------------------------------------------------------------------- offsets (plan section 3)
def model_free_shift(zs, obs, ref, qcut=SHIFT_QCUT, cell=SHIFT_CELL):
    """Observation ``obs``'s phase-centre shift against ``ref`` in mas ``(dra, ddec)`` and its error, in virgil's convention (the model is
    ``exp(-2 pi i (dra u + ddec v))`` times the sky's visibility), from the phase slope of ``V_obs conj(V_ref)`` over uv cells that both observations fill, below
    ``qcut`` wavelengths. It is the first moment of the cross-power, which is what the centroid of one dirty image against another measures, and needs no image.
    Returns ``dict(shift_mas, error_mas, n_cells)``; ``shift_mas`` is None when fewer than 3 cells overlap."""
    def gridded(sel):
        u = np.concatenate([np.asarray(z["u"], float) for z in zs if int(z["obsid"]) == sel])
        v = np.concatenate([np.asarray(z["v"], float) for z in zs if int(z["obsid"]) == sel])
        vis = np.concatenate([_vis(z) for z in zs if int(z["obsid"]) == sel])
        w = np.concatenate([np.asarray(z["weight"], float) for z in zs if int(z["obsid"]) == sel])
        keep = np.hypot(u, v) < qcut
        u, v, vis, w = u[keep], v[keep], vis[keep], w[keep]
        ci, cj = np.round(u / cell).astype(np.int64), np.round(v / cell).astype(np.int64)
        key = ci * 100003 + cj
        uk, inv = np.unique(key, return_inverse=True)
        sw = np.bincount(inv, w, len(uk))
        sv = np.bincount(inv, w * vis.real, len(uk)) + 1j * np.bincount(inv, w * vis.imag, len(uk))
        su = np.bincount(inv, w * u, len(uk)) / sw
        sv_ = np.bincount(inv, w * v, len(uk)) / sw
        return dict(zip(uk.tolist(), zip(sv / sw, sw, su, sv_)))
    a, b = gridded(obs), gridded(ref)
    common = sorted(set(a) & set(b))
    if len(common) < 3:
        return dict(shift_mas=None, error_mas=None, n_cells=len(common))
    va = np.array([a[k][0] for k in common])
    vb = np.array([b[k][0] for k in common])
    snr2a = np.abs(va) ** 2 * np.array([a[k][1] for k in common])
    snr2b = np.abs(vb) ** 2 * np.array([b[k][1] for k in common])
    u = np.array([0.5 * (a[k][2] + b[k][2]) for k in common])
    v = np.array([0.5 * (a[k][3] + b[k][3]) for k in common])
    phase = np.angle(va * np.conj(vb))
    var = 1.0 / (2 * snr2a) + 1.0 / (2 * snr2b)
    X = -2.0 * np.pi * np.stack([u, v], axis=1)
    W = 1.0 / var
    cov = np.linalg.pinv(X.T @ (X * W[:, None]))
    beta = cov @ (X.T @ (W * phase))
    chi2 = float(np.sum(W * (phase - X @ beta) ** 2) / max(len(common) - 2, 1))
    err = np.sqrt(np.diag(cov) * max(chi2, 1.0))  # the scatter beyond the noise (gain phases) inflates it
    return dict(shift_mas=(beta / MAS).tolist(), error_mas=(err / MAS).tolist(), n_cells=len(common), phase_chi2=chi2)


def prealign(zs, shifts_mas):
    """Undo each observation's phase-centre shift: ``V' = V exp(+2 pi i (dra u + ddec v))`` for ``shifts_mas = {obs: (dra, ddec)}`` (mas, the convention of
    ``model_free_shift``). Observations not in the dict are untouched."""
    out = []
    for z in zs:
        z = dict(z)
        s = shifts_mas.get(int(z["obsid"]))
        if s is not None:
            ph = 2.0 * np.pi * (s[0] * MAS * np.asarray(z["u"], float) + s[1] * MAS * np.asarray(z["v"], float))
            _set_vis(z, _vis(z) * np.exp(1j * ph))
        out.append(z)
    return out


def offset_plan(model_free, fitted, ref, box=BOX_MAS, agree=AGREE_MAS, edge=BOX_EDGE):
    """The rule of plan section 3 for each observation but the reference. ``model_free`` and ``fitted`` map observation -> (dra, ddec) in mas (the
    model-free shift and the offsets-only fit with the image held). If they agree within ``agree`` mas on both axes the observation is **pre-aligned** by the
    model-free shift and its residual (fitted minus the shift) is fitted in ``box``; otherwise the offset was a runaway: **no pre-alignment**, the box
    stays. A residual outside the box (here estimated by the offsets-only fit) puts the disc in ``drop``: it leaves the Toeplitz route and is reported,
    never run direct without Ben's word. Returns ``dict(per_obs, shifts_mas, drop, near_box, box_mas)``."""
    per, shifts, drop, near = {}, {}, [], []
    for o, f in fitted.items():
        if o == ref:
            continue
        f = np.asarray(f, float)
        m = None if model_free.get(o) is None else np.asarray(model_free[o], float)
        ok = m is not None and bool(np.all(np.abs(m - f) <= agree))
        resid = f - m if ok else f
        outside = bool(np.any(np.abs(resid) > box))
        close = bool(np.any(np.abs(resid) > (1.0 - edge) * box)) and not outside
        per[int(o)] = dict(action="prealign" if ok else "none", model_free_mas=None if m is None else m.tolist(), fitted_mas=f.tolist(),
                           residual_mas=resid.tolist(), outside_box=outside, near_box=close)
        if ok:
            shifts[int(o)] = m.tolist()
        if outside:
            drop.append(int(o))
        if close:
            near.append(int(o))
    return dict(per_obs=per, shifts_mas=shifts, drop=drop, near_box=near, box_mas=float(box), agree_mas=float(agree))


# ---------------------------------------------------------------------------------------------------- the lambda rule (decision 1)
def one_se_pick(rungs, scores, n_heldout, fold_scores=None):
    """The one-standard-error rule: of the rungs ``rungs`` (the weights) with summed held-out chi^2 ``scores``, the largest weight whose score is within one
    standard error of the minimum. It prefers the smoother image among those the data cannot tell apart. ``fold_scores`` (rungs x folds, the per-fold held-out
    chi^2) gives the standard error of the summed difference to the best rung from the spread of the per-fold differences, ``sqrt(K) std(d)`` (the usual
    cross-validation rule; rungs are scored on the same points, so this is far tighter than a single chi^2's spread). Without it, or with fewer than two folds,
    the fallback is the spread of one chi^2 under the null, ``2 sqrt(n_heldout)`` (``n_heldout`` points, two components each: 2n degrees of freedom).
    ``flat`` is true when the pick is the strongest rung (the data did not choose)."""
    rungs, scores = np.asarray(rungs, float), np.asarray(scores, float)
    best = int(np.argmin(scores))
    if fold_scores is not None and np.asarray(fold_scores).shape[1] >= 2:
        f = np.asarray(fold_scores, float)
        d = f - f[best]
        tol = np.sqrt(f.shape[1]) * np.std(d, axis=1, ddof=1)
        basis = "folds"
    else:
        tol = np.full(len(rungs), 2.0 * np.sqrt(n_heldout))
        basis = "null"
    ok = scores <= scores.min() + tol
    ok[best] = True
    pick = int(np.argmax(np.where(ok, rungs, -np.inf)))
    return dict(pick=pick, weight=float(rungs[pick]), best=best, tolerance=float(tol[pick]) if basis == "null" else tol.tolist(), basis=basis,
                flat=bool(pick == int(np.argmax(rungs))))


# ---------------------------------------------------------------------------------------------------- optimiser interface (PR-B plugs in here)
def lbfgs_optimiser(prob, init, *, max_steps, time_limit, log, on_round):
    """The default: virgil's L-BFGS through ``alma.run_fit_c8`` (restarts until the loss plateau, safety caps only). Returns ``(values, info, traj, why)``.
    An optimiser is any callable of this signature; ``alma_optim`` of PR-B provides the S2 candidates, and S2's winner replaces this default."""
    import virgil_bridge.alma as va

    return va.run_fit_c8(prob, init, max_steps=max_steps, time_limit=time_limit, log=log, on_round=on_round)


OPTIMISERS = {"lbfgs": lbfgs_optimiser}


def resolve_callable(spec, table=None):
    """``name`` from ``table``, or ``module:function`` imported."""
    if table and spec in table:
        return table[spec]
    if ":" not in spec:
        raise ValueError(f"{spec!r} is neither a known name ({sorted(table or {})}) nor module:function")
    mod, fn = spec.split(":", 1)
    return getattr(importlib.import_module(mod), fn)


# ---------------------------------------------------------------------------------------------------- the production problem (needs virgil)
def engine_of(term):
    """The chi^2 engine a ``fit`` term actually uses, read from the term (the old driver recorded the request, not the engine)."""
    c = type(term)
    return "toeplitz" if "toeplitz" in f"{c.__module__}.{c.__name__}".lower() else "direct"


def flat_pixels(npix, flux):
    """The flat start in sqrt-pixels: every pixel flux/npix^2 (strictly positive; x = 0 is stationary)."""
    return np.full((npix, npix), np.sqrt(flux / npix**2))


def gaussian_floor_pixels(npix, pixel_mas, flux, fwhm_mas=1000.0, floor=1e-4):
    """The old start, for the S4 comparison: a Gaussian of FWHM ``fwhm_mas`` and flux ``flux`` raised to ``floor`` Jy/arcsec^2, as sqrt-pixels of flux
    ``flux`` in total (renormalised)."""
    x = (np.arange(npix) - npix // 2) * pixel_mas
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    g = np.exp(-4.0 * np.log(2.0) * r2 / fwhm_mas**2)
    area = (pixel_mas / 1000.0) ** 2
    img = np.maximum(g / (g.sum() * area) * flux, floor)  # Jy/arcsec^2
    f = img * area
    return np.sqrt(f * flux / f.sum())


def build_production_problem(data, *, npix, pixel_mas, weight, ref_obs, alpha=0.0, box_mas=BOX_MAS, mask=None, blur_mas=None, flux=None, chi2="toeplitz",
                             start=None, hold_pixels=None):
    """The production model on ``data`` (weights already rescaled): ``LinearImage`` sqrt-pixels owning the flux, ``MaxEntropy(weight)`` on the unblurred
    pixels with a flat default, fitted gains and spread offsets in ``box_mas``, ``sigmas=False`` (s = 1), alpha fixed at ``alpha``, the engine ``chi2`` named
    explicitly. ``blur_mas`` puts a Gaussian kernel in the forward model (the ICF arm; the entropy stays on the unblurred pixels). ``start`` is the sqrt-pixel
    start (default flat); ``hold_pixels`` (sqrt-pixels) holds the image: no pixel site and no entropy, only the nuisances move (the offsets-only fit). Returns a namespace with the attributes ``alma.run_fit_c8`` and ``alma.score`` use, and ``engine`` read from the term."""
    if chi2 not in CHI2_ENGINES:
        raise ValueError(f"chi2 must be one of {CHI2_ENGINES}: the production driver never uses 'auto'")
    import types

    import jax.numpy as jnp
    import numpyro.distributions as dist
    import virgil_bridge.alma as va
    from virgil import CalibrationNuisance, LinearImage
    from virgil.imaging import MaxEntropy

    if flux is None:
        flux = ds.auto_flux(np.asarray(data.u), np.asarray(data.v), np.asarray(data.vis), np.asarray(data.weight))
    nuis = CalibrationNuisance(data, ref_obs=ref_obs, alpha=float(alpha), sigmas=False, fit_flux=False, blur_fwhm_mas=blur_mas or None)
    shift = -pixel_mas / 2.0  # the half-pixel map of docs/alma_conventions.md, as grid_scene

    held = None if hold_pixels is None else jnp.asarray(hold_pixels)

    def scene(values):
        return LinearImage(values["pixels"] if held is None else held, pixel_scale_mas=pixel_mas, parametrization="sqrt", reference_flux=float(flux), dra=shift, ddec=shift, ft="nufft")

    if chi2 == "toeplitz":
        term, info = va.toeplitz_term(data, scene, nuis, npix=npix, pixel_mas=pixel_mas, mask=mask, max_offset_mas=float(box_mas))
    else:
        term, info = data.term(scene, nuis, mask=None if mask is None else np.asarray(mask, bool), chi2="direct"), dict(engine="direct")
    engine = engine_of(term)
    say(f"data term: {type(term).__module__}.{type(term).__qualname__} -> engine {engine} (asked {chi2}; the Toeplitz term comes from ComplexVisData.toeplitz, which raises rather than falling back)")
    if engine != chi2:
        raise RuntimeError(f"asked for the {chi2} engine, the term is {engine}")
    priors, init, bound = va.nuisance_sites(nuis, data, offset_bound_mas=float(box_mas), ref_obs=ref_obs)
    if held is None:
        priors["pixels"] = dist.ImproperUniform(dist.constraints.real, (), (npix, npix))
        init = dict(init, pixels=jnp.asarray(flat_pixels(npix, flux) if start is None else start))
    return types.SimpleNamespace(npix=npix, pixel_mas=pixel_mas, scene=scene, nuis=nuis, priors=priors, init=init, terms=[term],
                                 regs=[] if held is not None else [MaxEntropy(float(weight))], model=lambda **values: scene(values), lam=float(weight), engine=engine, chi2_info=info,
                                 offset_bound_mas=float(bound.max()), offset_bound_obs=bound, offset_bound_requested=float(box_mas), flux=float(flux),
                                 blur_mas=blur_mas)


def image_jy_arcsec2(values, pixel_mas):
    """The image in Jy/arcsec^2 (virgil layout) of fitted sqrt-pixels."""
    return np.asarray(values["pixels"], float) ** 2 / (pixel_mas / 1000.0) ** 2


# ---------------------------------------------------------------------------------------------------- S1 probe
def _perturbed(prob, seed=0):
    """A point away from any optimum for the engine comparison: smooth multiplicative noise on the pixels, gains off 1 by 2%, offsets of a few mas."""
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    v = {k: jnp.asarray(x) for k, x in prob.init.items()}
    n = prob.npix
    noise = rng.normal(size=(n, n))
    k = np.fft.fftfreq(n)
    smooth = np.fft.ifft2(np.fft.fft2(noise) * np.exp(-0.5 * (k[:, None] ** 2 + k[None, :] ** 2) / 0.02**2)).real
    v["pixels"] = v["pixels"] * jnp.asarray(np.exp(0.3 * smooth / smooth.std()))
    for key in ("cal.gain", "cal.offset"):
        if key in v:
            x = np.asarray(v[key], float)
            v[key] = jnp.asarray(x * (1.0 + 0.02 * rng.normal(size=x.shape)) if key == "cal.gain" else rng.uniform(-3.0, 3.0, size=x.shape))
    return v


def engine_agreement(data, kw, log=say):
    """G1 item 1 and 5: the production model's loss and gradient on the Toeplitz and the direct engine at a perturbed point (the loss is half the sum of the
    squared residuals of the term), the cost of one value-and-gradient and one Hessian-vector product on the Toeplitz engine, and the precompute time."""
    import jax

    import virgil_bridge.alma as va

    t0 = time.time()
    pt = build_production_problem(data, chi2="toeplitz", **kw)
    t_pre = time.time() - t0
    pd = build_production_problem(data, chi2="direct", **kw)
    v = _perturbed(pt)

    def loss_of(prob):
        term = prob.terms[0]
        return lambda x: 0.5 * jax.numpy.sum(term(x) ** 2)

    out = {}
    with va.ft_eps(va.EPS_SCORE):
        ld, gd = jax.value_and_grad(loss_of(pd))(v)
    lt_fn = loss_of(pt)
    vg = jax.jit(jax.value_and_grad(lt_fn))
    lt, gt = vg(v)
    jax.block_until_ready(gt)
    t0 = time.time()
    for _ in range(3):
        jax.block_until_ready(vg(v)[1])
    t_vg = (time.time() - t0) / 3
    grad_fn = jax.jit(jax.grad(lt_fn))
    hvp = jax.jit(lambda x, d: jax.jvp(grad_fn, (x,), (d,))[1])
    jax.block_until_ready(hvp(v, gt))
    t0 = time.time()
    for _ in range(3):
        jax.block_until_ready(hvp(v, gt))
    t_hvp = (time.time() - t0) / 3
    rel_loss = abs(float(lt) - float(ld)) / abs(float(ld))
    rel_grad = {k: float(np.max(np.abs(np.asarray(gt[k]) - np.asarray(gd[k]))) / max(float(np.max(np.abs(np.asarray(gd[k])))), 1e-300)) for k in gd}
    try:
        peak = max(float(d.memory_stats().get("peak_bytes_in_use", 0)) for d in jax.local_devices()) / 1e9
    except Exception:  # noqa: BLE001 - CPU devices have no stats
        peak = None
    out.update(engine_toeplitz=pt.engine, engine_direct=pd.engine, loss_toeplitz=float(lt), loss_direct=float(ld), loss_rel=rel_loss, grad_rel=rel_grad,
               precompute_s=t_pre, toeplitz_info=pt.chi2_info, value_and_grad_s=t_vg, hvp_s=t_hvp, hvp_over_vg=t_hvp / t_vg, peak_gpu_gb=peak,
               pass_loss=bool(rel_loss < GATE_LOSS_REL), pass_grad=bool(max(rel_grad.values()) < GATE_GRAD_REL), pass_hvp=bool(t_hvp / t_vg <= GATE_HVP_COST))
    log(f"engines: loss rel {rel_loss:.2e}, gradient rel {max(rel_grad.values()):.2e}; vg {t_vg*1e3:.0f} ms, hvp {t_hvp*1e3:.0f} ms ({t_hvp / t_vg:.1f} vg), precompute {t_pre:.1f} s")
    return out


def offsets_only_fit(data, kw, log=say):
    """The model-based offset estimate of S1 item 2: a short joint fit of the image and the offsets (direct engine, a wide box 3 x the plan's, the weakest rung,
    300 L-BFGS steps) from a centred Gaussian start. The image has to move for the offsets to be measurable: held at a flat field its visibility is ~0 beyond
    the first null (4e4 wavelengths), so the offsets would have no gradient and stay at their start (review of PR 293). The reference observation fixes the
    frame, so only differences between observations are measured. Returns ``{obs: (dra, ddec)}`` for every observation but the reference, in mas."""
    import virgil_bridge.alma as va

    wide = dict(kw, box_mas=3.0 * kw["box_mas"])
    prob = build_production_problem(data, chi2="direct", start=gaussian_floor_pixels(kw["npix"], kw["pixel_mas"], kw["flux"]), **wide)
    vals, info, traj, why = va.run_fit_c8(prob, dict(prob.init), max_steps=300, log=log)
    off = np.asarray(vals["cal.offset"], float).reshape(-1, 2)
    obs = [o for o in range(len(data.obs_ids)) if o != kw["ref_obs"]]
    return {o: off[i].tolist() for i, o in enumerate(obs)}, dict(stop=why, steps=info.get("steps"))


def run_probe(args, zs, out):
    """Stage S1: the checks of plan section 3 items 1-5 and gate G1; writes ``probe.json``. No image is fitted."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import virgil_bridge.alma as va

    t = ds.target(args.target)
    npix, px = args.npix or t["npix"], args.pixel_mas or t["pixel_mas"]
    shat = va.read_shat(args.shat, int(max(int(z["ddid"]) for z in zs)) + 1) if args.shat else None
    wplan = weight_plan(zs, shat)
    say("weights (model-free s^2): " + ", ".join(f"ddid {d}: {p['s2']:.4g}{' per-bin' if p['per_bin'] else ''}" for d, p in sorted(wplan.items())))
    alpha = spectral_index(zs)
    say(f"spectral index: {alpha['alpha']} +- {alpha['sigma']} -> fixed at {alpha['alpha_fixed']}")
    zw = apply_weights(zs, wplan)
    obs = sorted({int(z["obsid"]) for z in zs})
    wsum = {o: float(sum(np.asarray(z["weight"], float).sum() for z in zw if int(z["obsid"]) == o)) for o in obs}
    ref_pos = int(np.argmax([wsum[o] for o in obs])) if args.ref_obs == "auto" else int(args.ref_obs)
    free = [o for o in obs if o != obs[ref_pos]]
    mf = {o: model_free_shift(zw, o, obs[ref_pos]) for o in free}
    say("model-free shifts (mas): " + ", ".join(f"obs {o}: {None if mf[o]['shift_mas'] is None else np.round(mf[o]['shift_mas'], 2).tolist()}" for o in free))
    data = va.load_npz(zw)
    flux0 = ds.auto_flux(np.asarray(data.u), np.asarray(data.v), np.asarray(data.vis), np.asarray(data.weight))
    kw = dict(npix=npix, pixel_mas=px, weight=LADDER[-1], ref_obs=ref_pos, alpha=alpha["alpha_fixed"], box_mas=BOX_MAS, flux=flux0)
    fitted, fit_info = ({}, None)
    if len(obs) > 1:
        fitted, fit_info = offsets_only_fit(data, kw)
        say("offsets-only fit (mas): " + ", ".join(f"obs {o}: {np.round(v, 2).tolist()}" for o, v in fitted.items()))
    # obs ids vs positions: virgil indexes observations by position in data.obs_ids
    ids = list(data.obs_ids)
    assert [int(i) for i in ids] == obs, f"virgil orders observations {ids}, not {obs}"
    plan = offset_plan({ids.index(o): mf[o]["shift_mas"] for o in free}, fitted, ref_pos) if fitted else dict(per_obs={}, shifts_mas={}, drop=[], near_box=[], box_mas=BOX_MAS)
    shifts_by_id = {int(ids[o]): s for o, s in plan["shifts_mas"].items()}
    # the engine comparison runs on the data as the production fits will see it: pre-aligned
    zp = prepared_arrays(zs, dict(weights=wplan, shifts_mas=shifts_by_id))
    data = va.load_npz(zp)
    eng = engine_agreement(data, kw)
    gate = dict(G1_loss=eng["pass_loss"], G1_grad=eng["pass_grad"], G1_hvp=eng["pass_hvp"], G1_offsets_in_box=not plan["drop"])
    gate["G1"] = all(gate.values())
    say("G1: " + ", ".join(f"{k} {'PASS' if v else 'FAIL'}" for k, v in gate.items()))
    if not gate["G1"]:
        say("G1 FAILED: stop and return to Ben (plan section 3).")
    res = dict(mode="probe", target=args.target, n=int(len(data.u)), weights=wplan, alpha=alpha, ref_obs=ref_pos, obs_ids=[int(i) for i in ids],
               model_free_shifts=mf, offsets_only=dict(fit=fitted, info=fit_info), offset_plan=plan, shifts_mas=shifts_by_id, engines=eng, gate=gate,
               ladder=list(LADDER), provenance=provenance(), args=vars(args))
    (out / "probe.json").write_text(json.dumps(jsonable(res), indent=1))
    say(f"wrote {out / 'probe.json'}")
    return res


def prepared_arrays(zs, probe):
    """The data a production fit sees: the probe's weight rescale, then its pre-alignment shifts (``probe['shifts_mas']`` keyed by observation id)."""
    wplan = {int(k): v for k, v in probe["weights"].items()}
    shifts = {int(k): v for k, v in probe.get("shifts_mas", {}).items()}
    return prepare(apply_weights(zs, wplan), shifts)


def prepare(zs, shifts):
    return prealign(zs, shifts) if shifts else zs


# ---------------------------------------------------------------------------------------------------- run modes
def fit_rung(prob_kw, data, init, mask, optimiser, args, log, on_round):
    """One fit of the production problem at ``prob_kw['weight']`` on the points of ``mask`` from ``init``. Returns ``(prob, values, info, why, wall)``."""
    prob = build_production_problem(data, mask=mask, start=None if init is None else init["pixels"], **prob_kw)
    start = dict(prob.init)
    if init is not None:
        start.update({k: v for k, v in init.items() if k in start})
    t0 = time.time()
    vals, info, traj, why = optimiser(prob, start, max_steps=args.max_steps, time_limit=args.time_limit, log=log, on_round=on_round)
    return prob, {k: np.asarray(v) for k, v in vals.items()}, info, traj, why, time.time() - t0


def run_ladder(prob_kw, data, mask, optimiser, args, log, on_round, start=None, rungs=LADDER, score_mask=None):
    """The lambda ladder on the points of ``mask`` (strongest first, each rung warm from the last). Returns a list of dicts, one per rung, with ``values``.
    ``score_mask`` (held-out points) gives each rung's held-out chi^2 at the fixed rescaled weights."""
    import virgil_bridge.alma as va

    out, init = [], start
    for w in rungs:
        prob, vals, info, traj, why, wall = fit_rung(dict(prob_kw, weight=w), data, init, mask, optimiser, args, log, lambda *a, w=w: on_round(w, *a))
        sc = va.score(data, prob.scene, vals, prob.nuis, mask=mask)
        row = dict(weight=w, values=vals, steps=info.get("steps"), stop=why, capped=why not in CONVERGED_WHY, wall_s=wall,
                   chi2=sc["chi2_raw"], n=sc["n"], engine=prob.engine)
        if score_mask is not None:
            ho = va.score(data, prob.scene, vals, prob.nuis, mask=score_mask)
            row.update(heldout_chi2=ho["chi2_raw"], heldout_n=ho["n"])
        say(f"  rung w={w:g}: {row['steps']} steps, stop {why}, chi2 {row['chi2']:.6g}, {wall:.0f} s" + (f", held-out chi2 {row['heldout_chi2']:.6g}" if score_mask is not None else ""))
        out.append(row)
        init = vals
    return out


def tuning_scores(prob_kw, data, folds, optimiser, args, log, on_round, rungs=LADDER):
    """The held-out chi^2 of each rung: fold ``j`` in 1-3 is left out of the training set (folds 1-3 minus ``j``; fold 0 never enters) and scored; summed over ``j``."""
    total, n_total, by_fold = np.zeros(len(rungs)), 0, {}
    for j in TUNING_FOLDS:
        train = np.isin(folds, [f for f in TUNING_FOLDS if f != j])
        rows = run_ladder(prob_kw, data, train, optimiser, args, log, lambda *a, j=j: on_round(f"fold{j}", *a), rungs=rungs, score_mask=folds == j)
        by_fold[j] = [dict(weight=r["weight"], heldout_chi2=r["heldout_chi2"], n=r["heldout_n"], stop=r["stop"], capped=r["capped"]) for r in rows]
        total += np.array([r["heldout_chi2"] for r in rows])
        n_total += rows[0]["heldout_n"]
    return total, n_total, by_fold


def run_mode(args, zs, probe, out):
    """The run modes s3, s4 and s5. s3 and s5: the full-data ladder, the tuning-fold ladders, the one-SE pick and the test-fold fit at the pick (s5 also the
    restarts); s4: the full-data ladder from ``--start`` (flat or gaussian) with the ``--arm`` (``mem`` classic MEM, ``icf`` a 30 mas kernel in the model,
    ``r`` MPoL's R objective, the illustration arm)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import virgil_bridge.alma as va

    if probe["offset_plan"]["drop"]:
        raise SystemExit(f"{args.target}: the probe's offsets leave the Toeplitz box for observations {probe['offset_plan']['drop']}: the disc is dropped, not run direct without Ben's word")
    t = ds.target(args.target)
    npix, px = args.npix or t["npix"], args.pixel_mas or t["pixel_mas"]
    data = va.load_npz(prepared_arrays(zs, probe))
    flux0 = ds.auto_flux(np.asarray(data.u), np.asarray(data.v), np.asarray(data.vis), np.asarray(data.weight))
    blur = args.taper_mas if args.arm == "icf" else None
    if args.arm == "r":
        raise SystemExit("--arm r (MPoL's R, the labelled illustration) runs through alma_imlup_maxent.py's R objective, not the production model; see the plan's section 5.1")
    optimiser = resolve_callable(args.optimiser, OPTIMISERS)
    certify = resolve_callable(args.certify, {}) if args.certify else None
    prob_kw = dict(npix=npix, pixel_mas=px, ref_obs=probe["ref_obs"], alpha=probe["alpha"]["alpha_fixed"], box_mas=BOX_MAS, blur_mas=blur, flux=flux0)
    say(f"{t['name']}: {len(data.u):,} points, mode {args.mode}, arm {args.arm}, start {args.start}, optimiser {args.optimiser}, engine toeplitz, alpha {prob_kw['alpha']:g}, "
        f"ref obs {prob_kw['ref_obs']}, offsets ±{BOX_MAS:g} mas; flux start {flux0:.4f} Jy")
    ra = dec = 0.0
    if args.info:
        pc = json.loads(pathlib.Path(args.info).read_text())["phase_centre"]
        ra, dec = pc["ra_deg"], pc["dec_deg"]
    else:
        say("warning: no --info: image.fits is on the sky position (0, 0)")
    start = None
    if args.start == "gaussian":
        start = dict(pixels=gaussian_floor_pixels(npix, px, flux0))

    def on_round(tag, w, vals, info, traj, rnd):  # a checkpoint after every round, so a walltime kill keeps the last point
        np.savez(out / f"values_{tag}_w{w:g}.npz", **{k: np.asarray(v) for k, v in vals.items()})
        say(f"[{tag} w={w:g}] round {rnd}: {info['steps']} steps, stop {info['stop']}, loss {info['loss']:.3f}")

    t0 = time.time()
    rungs = tuple(args.rungs) if args.rungs else LADDER
    full = run_ladder(prob_kw, data, None, optimiser, args, say, lambda w, *a: on_round("full", w, *a), start=start, rungs=rungs)
    res = dict(mode=args.mode, arm=args.arm, start=args.start, target=args.target, name=t["name"], n=int(len(data.u)), npix=npix, pixel_mas=px, ladder=list(rungs),
               engine="toeplitz", alpha=prob_kw["alpha"], ref_obs=prob_kw["ref_obs"], probe=str(args.probe), weights_rescaled=True,
               full=[{k: v for k, v in r.items() if k != "values"} for r in full], any_capped=any(r["capped"] for r in full), provenance=provenance(), args=vars(args))
    sel = None
    if args.mode in ("s3", "s5"):
        folds, _ = va.make_folds(data, k=K_FOLDS)
        scores, n_ho, by_fold = tuning_scores(prob_kw, data, folds, optimiser, args, say, on_round, rungs=rungs)
        fold_matrix = np.array([[by_fold[j][i]["heldout_chi2"] for j in TUNING_FOLDS] for i in range(len(rungs))])
        sel = one_se_pick(rungs, scores, n_ho, fold_matrix)
        say(f"held-out chi2 {np.round(scores, 1).tolist()}; one-SE pick w={sel['weight']:g}{' (flat: the data did not choose)' if sel['flat'] else ''}")
        train = folds != TEST_FOLD
        prob, vals, info, traj, why, wall = fit_rung(dict(prob_kw, weight=sel["weight"]), data, start, train, optimiser, args, say,
                                                     lambda *a: on_round("test", sel["weight"], *a))
        test = va.score(data, prob.scene, vals, prob.nuis, mask=folds == TEST_FOLD)
        res.update(folds=dict(k=K_FOLDS, test=TEST_FOLD, tuning=list(TUNING_FOLDS), sha256=va.folds_sha256({0: folds})), heldout=dict(rungs=list(rungs), scores=scores.tolist(), n=n_ho,
                   by_fold=by_fold), selection=sel, test_fold=dict(weight=sel["weight"], chi2=test["chi2_raw"], n=test["n"], stop=why, capped=why not in CONVERGED_WHY))
        if args.mode == "s5":
            res["restarts"] = run_restarts(prob_kw, data, sel, full[sel["pick"]]["values"], optimiser, args, on_round)
    pick = full[sel["pick"]] if sel else full[-1]
    vals = pick["values"]
    prob = build_production_problem(data, **dict(prob_kw, weight=pick["weight"]))
    base = image_jy_arcsec2(vals, px)
    from crosscheck import alma_maxent as cx

    sky = va.tapered_image(base, px, args.taper_mas) if args.taper_mas else base
    np.savez(out / "values.npz", **vals)
    np.save(out / "image_base.npy", base.astype(np.float32))
    np.save(out / "image_tapered.npy", sky.astype(np.float32))
    cx.write_fits(out / "image.fits", sky, px, ra, dec, OBJECT=t["name"], ORIGIN=f"virgil production {args.mode} {args.arm}, {args.target}", TAPERMAS=float(args.taper_mas))
    res.update(selected_weight=pick["weight"], flux_jy=float(np.sum(np.asarray(vals["pixels"]) ** 2)), gain=vals.get("cal.gain"), offset_mas=vals.get("cal.offset"),
               at_bound=va.at_bound_report(vals, prob.priors),
               offset_bound=va.offset_bound_report(vals.get("cal.offset"), prob.offset_bound_mas, prob.offset_bound_requested, bound_obs=prob.offset_bound_obs, ref_obs=prob.nuis.ref_obs),
               image_fits="the base image convolved with --taper-mas for display only; image_base.npy is the fitted image", wall_s=time.time() - t0)
    res["certified"] = None if certify is None else certify(prob=prob, values=vals, data=data, result=res, out=out)
    (out / "result.json").write_text(json.dumps(jsonable(res), indent=1))
    say(f"selected w {pick['weight']:g}; certified {res['certified'] if certify else 'null (no --certify: PR-B pending)'}; capped {res['any_capped']}; wrote {out}")
    return res


def run_restarts(prob_kw, data, sel, ref_values, optimiser, args, on_round):
    """C3's restarts at the selected rung (recorded; the pass/fail test is ``alma_certify``'s): the 1 arcsec Gaussian with floor image start, and two nuisance
    starts (offsets ±10 mas, gains ±5%, opposite signs). Each runs to the same caps as the main fit."""
    import jax.numpy as jnp
    import virgil_bridge.alma as va

    px, npix = prob_kw["pixel_mas"], prob_kw["npix"]
    starts = {"image_gaussian": dict(ref_values, pixels=gaussian_floor_pixels(npix, px, prob_kw["flux"]))}
    for name, sgn in (("nuisance_plus", 1.0), ("nuisance_minus", -1.0)):
        s = dict(ref_values)
        if "cal.offset" in s:
            s["cal.offset"] = np.full_like(np.asarray(s["cal.offset"], float), sgn * 10.0)
        if "cal.gain" in s:
            s["cal.gain"] = np.asarray(s["cal.gain"], float) * (1.0 + sgn * 0.05)
        starts[name] = s
    out = {}
    for name, init in starts.items():
        init = {k: jnp.asarray(v) for k, v in init.items()}
        prob, vals, info, traj, why, wall = fit_rung(dict(prob_kw, weight=sel["weight"]), data, init, None, optimiser, args, say, lambda *a, n=name: on_round(n, sel["weight"], *a))
        sc = va.score(data, prob.scene, vals, prob.nuis)
        out[name] = dict(stop=why, steps=info.get("steps"), chi2=sc["chi2_raw"], flux_jy=float(np.sum(vals["pixels"] ** 2)), gain=vals.get("cal.gain"), offset_mas=vals.get("cal.offset"),
                         wall_s=wall)
        np.savez(pathlib.Path(args.out) / f"values_restart_{name}.npz", **vals)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("probe", "s3", "s4", "s5"))
    p.add_argument("--target", required=True, choices=ds.CAMPAIGN)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--info", default=None, help="dsharp_ms_info.py's JSON: the phase centre of the image's WCS (omitted: 0, 0 and a warning)")
    p.add_argument("--shat", default=None, help="directory of ddid_NN.json (alma_imlup_read): the window scales, used only where the npz has no time/antenna columns")
    p.add_argument("--probe", default=None, help="probe.json of the S1 probe (required by s3, s4 and s5)")
    p.add_argument("--npix", type=int, default=None)
    p.add_argument("--pixel-mas", type=float, default=None)
    p.add_argument("--taper-mas", type=float, default=30.0, help="ICF kernel (--arm icf) and the display convolution of image.fits")
    p.add_argument("--arm", choices=("mem", "icf", "r"), default="mem")
    p.add_argument("--start", choices=("flat", "gaussian"), default="flat")
    p.add_argument("--rungs", type=float, nargs="*", default=None, help="the weights of the ladder (default: the plan's six)")
    p.add_argument("--ref-obs", default="auto")
    p.add_argument("--optimiser", default="lbfgs", help="name in OPTIMISERS or module:function (the S2 winner)")
    p.add_argument("--certify", default=None, help="module:function of alma_certify (PR-B)")
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--time-limit", type=float, default=18000.0)
    args = p.parse_args(argv)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    zs = load_arrays(args.data)
    if args.mode == "probe":
        return run_probe(args, zs, out)
    if not args.probe:
        raise SystemExit(f"mode {args.mode} needs --probe (the S1 probe.json): it holds the weights, the shifts and alpha")
    return run_mode(args, zs, json.loads(pathlib.Path(args.probe).read_text()), out)


if __name__ == "__main__":
    res = main()
    # a failed G1 stops the afterok chain of the next stages (plan section 3: stop and return to Ben)
    raise SystemExit(3 if isinstance(res, dict) and res.get("mode") == "probe" and not res["gate"]["G1"] else 0)
