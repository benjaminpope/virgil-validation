"""The pre-registered regulariser-weight rule of the IM Lup deconvolutions (``lambda_selection`` in ``design/alma_imlup_gate_criteria.yml``, items 1-9 and D1-D6).

Primary arm only: maximum entropy on sqrt(I) pixels (``LinearImage(parametrisation="sqrt")``, ``MaxEntropy``, L-BFGS-B), the weight w the root of the
degrees-of-freedom-corrected discrepancy m(w) = chi2_q / s^2 - (n_real - G(w)). Stages, each writing one JSON and ending the run with a written reason when it fails::

    sigma  -> sigma.json    s^2 from the data scatter (full data), its error, raw chi^2/N, lag-1 correlation; fails if sigma_s / s^2 > 2e-3
    d4     -> d4.json       bias of that estimator on the pinned CLEAN model plus noise of known s^2; the sweep refuses without a pass
    sweep  -> sweep.json    the fast loop on the full data (D6): grid, G, m, the root, w at s^2 +/- 2 sigma_s, h(w)
    final  -> lambda_final.json   the full data at that w: staged nuisances, KKT and band limit, the item-8 report

There is no fallback weight anywhere: no root, an unmet |m| <= 3 se(G), a nuisance at a box edge or an unconverged KKT test ends the run.
"""

import dataclasses
import hashlib
import json
import pathlib
import warnings

import numpy as np

from . import alma_start as vs

# The registered numbers (a test checks each against the text of the criteria file).
GRID_MAX, GRID_MIN, GRID_STEP = 1e8, 1e2, 10.0**0.5  # D1: w / s^2
NPROBE, CG_TOL, MAX_SECANT = 16, 1e-4, 5  # item 3, D2
SIGMA_REL_MAX = 2e-3  # item 1
D4_REALISATIONS, D4_REL_MAX = 20, 2e-3  # D4
KKT_W_MAX, BAND_MAX, EDGE = 1e-4, 2.0, 0.05  # items 6, 7
ARM = "mem_sqrt_pixels"
PHRASES = (  # what the constants above implement, as the criteria text says it
    "relative uncertainty must be ≤ 2e-3",
    "16 Rademacher probes, CG relative tolerance 1e-4",
    "w/s² from 1e8 down to 1e2 in steps of ×√10",
    "at most 5 steps",
    "the observation with the largest Σw",
    "add 20 realisations of noise with a known s²",
    "|mean − s²| ≤ 2·sd/√20 and ≤ 2e-3·s²",
    "take the crossing at the largest w",
    "kkt_w (rms of g − μ over pixels with b > 1e-3·b_max, in units of w/F) < 1e-4",
    "f_eta/f_floor < 2",
    "within 5% of its prior box edge",
    "D6. **The fast loop runs on the full data** (stride 1)",
)
HP_FWHM_MAS = 60.0  # the high-pass of psi_hp (the diagnosis' smoothing)


class RuleFailure(Exception):
    """A stage of the rule failed: ``record`` is what to write, ``str(self)`` the reason."""

    def __init__(self, reason, record):
        super().__init__(reason)
        self.record = dict(record, status="failed", reason=reason)


def rule(block=None):
    """The registered ``lambda_selection`` block, checked to be the one this driver implements."""
    block = dict(vs.CRITERIA["lambda_selection"] if block is None else block)
    if block.get("rule") != "discrepancy_dof_corrected" or block.get("primary_arm") != "mem_sqrt_pixels":
        raise SystemExit(f"lambda_selection is not the registered dof-corrected rule on MEM sqrt pixels ({block.get('rule')}, {block.get('primary_arm')})")
    text = " ".join(block["text"].split())
    missing = [p for p in PHRASES if " ".join(p.split()) not in text]
    if missing:
        raise SystemExit(f"the criteria text no longer says {missing}: the constants of alma_lambda.py were pinned to it")
    return block


def grid(s2):
    """The D1 weights, largest first: ``s2`` times 1e8 down to 1e2 in steps of x sqrt(10)."""
    n = int(round(np.log(GRID_MAX / GRID_MIN) / np.log(GRID_STEP)))
    return float(s2) * GRID_MAX / GRID_STEP ** np.arange(n + 1)


# ---------------------------------------------------------------- sigma stage (item 1) and D4


def fingerprint(data):
    """Identifies the visibility set a verdict belongs to (geometry, times, weights), not its values."""
    h = hashlib.sha256()
    for name in ("u", "v", "weight", "time", "ddid", "scan", "ant1", "ant2"):
        a = getattr(data, name)
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes() if a is not None else b"-")
    return h.hexdigest()


def reference_obs(data):
    """D3: the observation with the largest sum of weights, from the weights alone, and the sums."""
    sums = np.bincount(np.asarray(data.obs), weights=np.asarray(data.weight, float), minlength=len(data.obs_ids))
    return int(np.argmax(sums)), sums


def flux_scale(data, fraction=0.01):
    """``F0`` (Jy), the normalisation of the entropy: the weighted mean of Re V over the shortest ``fraction`` of baselines. Fixed from the data before any fit; it sets the unit of w."""
    q = np.hypot(np.asarray(data.u), np.asarray(data.v))
    short = q <= np.quantile(q, fraction)
    w = np.asarray(data.weight, float)[short]
    return float(np.sum(w * np.asarray(data.vis).real[short]) / np.sum(w))


def lag1_correlation(data):
    """Lag-1 correlation of successive differences (standardised, Re and Im pooled) along each baseline, channel and scan of each ddid: -0.5 for white noise."""
    need = [n for n in ("time", "ant1", "ant2", "scan") if getattr(data, n) is None]
    if need:
        raise ValueError(f"the data lack {need}")
    num = den1 = den2 = 0.0
    n = 0
    ddid = np.asarray(data.ddid)
    for d in np.unique(ddid):
        idx = np.flatnonzero(ddid == d)
        ant1, ant2 = np.asarray(data.ant1)[idx].astype(np.int64), np.asarray(data.ant2)[idx].astype(np.int64)
        _, chan = np.unique(np.asarray(data.freq)[idx], return_inverse=True)
        key = np.stack([np.asarray(data.scan)[idx].astype(np.int64), ant1 * (ant2.max() + 1) + ant2, chan])
        order = np.lexsort((np.asarray(data.time)[idx], *key[::-1]))
        idx, key = idx[order], key[:, order]
        if idx.size < 3:
            continue
        same = (key[:, 1:] == key[:, :-1]).all(axis=0)  # successive rows on the same run
        inv_w = 1.0 / np.asarray(data.weight, float)[idx]
        dv = np.diff(np.asarray(data.vis)[idx])
        den = inv_w[1:] + inv_w[:-1]
        both = same[1:] & same[:-1]  # three rows on one run: two differences
        for z in (dv.real / np.sqrt(den), dv.imag / np.sqrt(den)):
            a, b = z[:-1][both], z[1:][both]
            num, den1, den2, n = num + np.sum(a * b), den1 + np.sum(a * a), den2 + np.sum(b * b), n + a.size
    return float(num / np.sqrt(den1 * den2)) if n else float("nan"), n


def scatter(data):
    """``(s2, sigma_s, n_pairs)``: s^2 from the differences of time-adjacent pairs per baseline, channel, scan and ddid (virgil's ``sigma_scale``, pooled), and its standard error."""
    ss = data.sigma_scale(by=None)
    s, se = float(ss.s[0]), float(ss.s_se[0])
    return s * s, 2.0 * s * se, int(ss.n_pairs[0])


def sigma_stage(data):
    """Item 1: the record of s^2, sigma_s, raw chi^2/N, the lag-1 correlation, D3's reference observation and F0; ``RuleFailure`` if sigma_s / s^2 > 2e-3.
    ``raw_chi2_over_N_pairs`` is the chi^2/N of the pair differences on the quoted weights (= s^2: noise alone, no image); the image's raw chi^2/N is in the final report."""
    try:
        s2, sigma_s, n_pairs = scatter(data)
        r1, n_lag = lag1_correlation(data)
    except ValueError as e:
        raise SystemExit(f"sigma stage: {e}") from e
    ref, sums = reference_obs(data)
    rec = dict(stage="sigma", s2=s2, sigma_s=sigma_s, sigma_rel=sigma_s / s2, sigma_rel_max=SIGMA_REL_MAX, n_pairs=n_pairs, raw_chi2_over_N_pairs=s2, lag1_correlation=r1, lag1_white_noise=-0.5, n_lag1=n_lag,
               ref_obs=ref, sum_weights=sums, flux_jy=flux_scale(data), n_vis=len(data.u), n_real=2 * len(data.u), fingerprint=fingerprint(data), status="ok")
    if not sigma_s / s2 <= SIGMA_REL_MAX:
        raise RuleFailure(f"sigma_s / s^2 = {sigma_s / s2:.3g} exceeds {SIGMA_REL_MAX:g}: the noise scale is not known well enough ({n_pairs} pairs)", rec)
    return rec


def d4_stage(data, truth, pixel_mas, *, seed=0, chunk_size=10**6, model=None, log=print):
    """D4: ``D4_REALISATIONS`` noise realisations of known s^2 (the data's own estimate) added to the visibilities of ``truth`` (the pinned CLEAN model, Jy/arcsec^2; ``model`` to skip the transform)
    at ``data``'s sampling, each through the scatter estimator. Pass: |mean - s^2| <= 2 sd / sqrt(20) and <= 2e-3 s^2. Returns the verdict record (``RuleFailure`` carries it on a fail)."""
    from . import alma_synth as syn

    s2, sigma_s, _ = scatter(data)
    mv = syn.model_visibilities(data, truth, pixel_mas, chunk_size=chunk_size) if model is None else np.asarray(model)
    est = []
    for r in range(D4_REALISATIONS):
        vis = syn.noisy(mv, np.asarray(data.weight, float) / s2, np.random.default_rng([int(seed), r]))
        est.append(scatter(dataclasses.replace(data, vis=vis))[0])
        log(f"d4 realisation {r}: s2 = {est[-1]:.6g} (injected {s2:.6g})")
    est = np.array(est)
    mean, sd = float(est.mean()), float(est.std(ddof=1))
    se = sd / np.sqrt(D4_REALISATIONS)
    r1, _ = lag1_correlation(data)
    rec = dict(stage="d4", s2_injected=s2, sigma_s_real=sigma_s, estimates=est, mean=mean, sd=sd, se_mean=se, bias=mean - s2, bias_rel=(mean - s2) / s2, bias_in_se=(mean - s2) / se, rel_max=D4_REL_MAX,
               n=D4_REALISATIONS, lag1_correlation_real=r1, fingerprint=fingerprint(data), seed=int(seed))
    ok = abs(mean - s2) <= 2.0 * se and abs(mean - s2) <= D4_REL_MAX * s2
    rec["pass"] = bool(ok)
    if not ok:
        raise RuleFailure(f"D4 failed: mean {mean:.6g} against the injected s^2 {s2:.6g} (bias {rec['bias_rel']:.3g} of s^2, {rec['bias_in_se']:.2f} se); no real-data run is made", rec)
    return dict(rec, status="ok")


def require_d4(verdict, data):
    """The sweep's gate: a D4 verdict that passed on this very visibility set."""
    if not verdict or verdict.get("stage") != "d4" or verdict.get("pass") is not True:
        raise SystemExit("the sweep needs a passing D4 verdict (stage d4); there is none, or it failed")
    if verdict["fingerprint"] != fingerprint(data):
        raise SystemExit("the D4 verdict is of another visibility set (sampling or weights differ); rerun stage d4 on this data")


# ---------------------------------------------------------------- the model and its fits


@dataclasses.dataclass
class Problem:
    """Forward model on the crop: sqrt-pixel image, Toeplitz chi^2, nuisances at nominal."""

    data: object
    scene: object
    nuisance: object
    term: object
    priors: dict
    npix: int
    pixel_mas: float
    flux: float

    def model(self, **values):
        return self.scene(values)

    def start(self):
        import jax.numpy as jnp

        return {"pixels": jnp.full((self.npix, self.npix), np.sqrt(self.flux / self.npix**2))}

    def chi2(self, values):
        """Raw chi^2 on the quoted weights (constants included) of a fit's ``values``."""
        import jax.numpy as jnp

        return float(self.term.full_chi2_and_log_norm({**self.nuisance.defaults(), **{k: jnp.asarray(v) for k, v in values.items()}})[0])

    def curvature(self, model):
        """``K v``: the Hessian-vector product of 1/2 chi^2 in pixel fluxes at ``model`` (the data's Gauss-Newton curvature, exact: the model is linear in flux), for ``dof_discrepancy``."""
        import jax
        import jax.numpy as jnp

        b0 = np.asarray(model.pixels, float) ** 2
        safe = jnp.asarray(np.where(b0 > 0, b0, 1.0))  # free pixels only; the rest are masked by the caller
        defaults = self.nuisance.defaults()
        half = lambda b: 0.5 * self.term.full_chi2_and_log_norm({**defaults, "pixels": jnp.sqrt(b)})[0]  # noqa: E731
        hvp = jax.jit(lambda v: jax.jvp(jax.grad(half), (safe,), (v,))[1])
        return lambda v: np.asarray(hvp(jnp.asarray(v).reshape(b0.shape)), float).ravel()


def build(data, *, npix, pixel_mas, flux, ref_obs, chunk_size=None):
    """The ``Problem`` of ``data`` at the crop, the nuisances nominal (no gains, offsets or sigma sites: the weights are the quoted ones)."""
    import numpyro.distributions as dist
    from virgil import LinearImage
    from virgil.radio import CalibrationNuisance

    shift = -pixel_mas / 2.0

    def scene(values):
        return LinearImage(values["pixels"], pixel_mas, dra=shift, ddec=shift, ft="auto", parametrisation="sqrt", reference_flux=flux)

    nuis = CalibrationNuisance(data, ref_obs=ref_obs, alpha=False, gains=False, offsets=False, sigmas=False, fit_flux=False)
    term = data.term(scene, nuis, chi2="toeplitz")
    priors = {"pixels": dist.ImproperUniform(dist.constraints.real, (), event_shape=(npix, npix))}
    return Problem(data, scene, nuis, term, priors, int(npix), float(pixel_mas), float(flux))


def kkt_of(report):
    """``(kkt_w, band_limit)`` of a ``kkt_report``'s image block (the one image of the model; band_limit is NaN for a flat image)."""
    (img,) = report["images"].values()
    return float(img["kkt_w"]), float(img.get("band_limit", np.nan))


def fit_point(prob, w, init, *, gtol=1e-4, max_rounds=8, max_steps=20000, time_limit=None, n_real=None, log=print):
    """``(FitResult, kkt_report, rounds)``: ``MaxEntropy(w)`` fitted by L-BFGS-B from ``init``, refitted warm with ``gtol`` / 10 each round until ``kkt_w < 1e-4`` or ``max_rounds``
    (virgil's own stop leaves kkt_w at about 1e-2). ``n_real`` replaces the result's data count."""
    from virgil import fit
    from virgil.fitting import kkt_report
    from virgil.imaging import MaxEntropy

    reg, x0, rounds = MaxEntropy(float(w)), init, []
    for r in range(max_rounds):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*cannot see inside a function model.*")
            res = fit(prob.model, prob.priors, (), likelihoods=[prob.term], regularizers=[reg], init=x0, method="lbfgsb", lbfgs_memory=100, max_steps=max_steps, gtol=gtol, time_limit=time_limit)
        x0 = {k: v for k, v in res.values.items()}
        report = kkt_report(dataclasses.replace(res, model=prob.model), prob.priors, (), [reg], likelihoods=[prob.term], weight=float(w))
        kk, band = kkt_of(report)
        rounds.append(dict(gtol=gtol, steps=int(res.info.get("steps", 0)), kkt_w=kk, band_limit=band))
        log(f"  w = {w:.5g} round {r}: gtol {gtol:g}, steps {rounds[-1]['steps']}, kkt_w {kk:.3g}, band_limit {band:.3g}")
        if kk < KKT_W_MAX:
            break
        gtol /= 10.0
    if n_real is not None:
        res = dataclasses.replace(res, info={**res.info, "ndata": [float(n_real)]})
    return res, report, rounds


def penalty_of(result, w):
    """The unweighted entropy of a fit (``MaxEntropy(w).value / w``)."""
    from virgil.imaging import MaxEntropy

    return float(MaxEntropy(float(w)).value(result.model)) / float(w)


def psi_hp(pixels, pixel_mas, floor=1e-30):
    """rms of the high-passed ln b (the diagnosis' psi_hp) on the disc (pixels above 1e-3 of the peak) and in two brightness bins (1e-2 to 1e-1 and above 1e-1 of the peak)."""
    psi, rel = psi_hp_map(pixels, pixel_mas, floor)
    disc = rel >= 1e-3
    return dict(disc=float(psi[disc].std()), lo=float(psi[disc & (rel < 1e-1) & (rel >= 1e-2)].std()), hi=float(psi[rel >= 1e-1].std()))


def psi_hp_map(pixels, pixel_mas, floor=1e-30):
    """``(psi, rel)``: the high-passed ln b of sqrt-pixel ``pixels`` (zero off the disc) and the flux relative to the peak."""
    from scipy.ndimage import gaussian_filter

    flux = np.asarray(pixels, float) ** 2
    b = np.maximum(flux, floor * flux.sum())
    b = b / b.sum()
    eta = np.log(b)
    rel = flux / flux.max()
    disc = rel >= 1e-3
    sig = HP_FWHM_MAS / 2.3548 / pixel_mas
    smooth = gaussian_filter(np.where(disc, eta, 0.0), sig, mode="constant") / np.maximum(gaussian_filter(disc.astype(float), sig, mode="constant"), 1e-12)
    return np.where(disc, eta - smooth, 0.0), rel


# ---------------------------------------------------------------- sweep stage (items 3, 4, D1, D2, D5)


def sweep_stage(data, sigma, *, npix, pixel_mas, seed=0, gtol=1e-4, time_limit=None, cg_maxit=2000, max_steps=20000, log=print):
    """The fast loop. Returns ``(record, start_values)``: the grid fitted from large w down, G by Hutchinson, the D5 crossing with up to ``MAX_SECANT`` secant refits;
    ``RuleFailure`` (with the record) on D1 (m <= 0 at the first point, no sign change) or D2 (|m| > 3 se(G) after the secant steps). ``start_values`` is the root's fit, the start of the final stage."""
    from virgil.imaging import LCurve, MaxEntropy

    s2, sigma_s = float(sigma["s2"]), float(sigma["sigma_s"])
    n_eff = 2.0 * len(data.u)
    prob = build(data, npix=npix, pixel_mas=pixel_mas, flux=sigma["flux_jy"], ref_obs=sigma["ref_obs"])
    weights = grid(s2)
    log(f"sweep: {len(data.u):,} rows (full data, D6), n_real {n_eff:.6g}; {len(weights)} weights, w/s2 {GRID_MAX:g} to {GRID_MIN:g}")
    results, reports, x0, unconverged = [], [], prob.start(), []
    for w in weights:
        res, rep, _ = fit_point(prob, w, x0, gtol=gtol, time_limit=time_limit, n_real=n_eff, max_steps=max_steps, log=log)
        if not kkt_of(rep)[0] < KKT_W_MAX:
            unconverged.append(dict(weight=float(w), kkt_w=kkt_of(rep)[0]))
        results.append(res)
        reports.append(rep)
        x0 = dict(res.values)

    def refit(weight, init):
        res, rep, _ = fit_point(prob, weight, init, gtol=gtol, time_limit=time_limit, n_real=n_eff, max_steps=max_steps, log=log)
        if not kkt_of(rep)[0] < KKT_W_MAX:
            unconverged.append(dict(weight=float(weight), kkt_w=kkt_of(rep)[0]))
        return res

    chi2 = np.array([float(sum(r.info["chi2"])) for r in results])
    curve = LCurve(weights, chi2, (chi2 / n_eff)[:, None], np.array([penalty_of(r, w) for r, w in zip(results, weights)]), results, refit=refit, others=())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = curve.dof_discrepancy((), MaxEntropy(1.0), noise_scale2=s2, noise_scale_sigma=sigma_s, nprobe=NPROBE, tol=CG_TOL, maxit=cg_maxit, seed=seed, secant=True, max_secant=MAX_SECANT, curvature=prob.curvature)
    kkt = [kkt_of(r) for r in reports]
    table = [dict(weight=float(w), w_over_s2=float(w) / s2, chi2=float(c), m=float(m), g=float(g), kkt_w=k[0], band_limit=k[1]) for w, c, m, g, k in zip(weights, chi2, out.m_sweep, out.g_sweep, kkt)]
    rec = dict(stage="sweep", s2=s2, sigma_s=sigma_s, fingerprint=sigma["fingerprint"], seed=int(seed), n_rows=len(data.u), n_real=n_eff, npix=int(npix), pixel_mas=float(pixel_mas), flux_jy=float(sigma["flux_jy"]),
               ref_obs=int(sigma["ref_obs"]), points=table, crossings=[float(x) for x in out.roots], crossings_w_over_s2=[float(x) / s2 for x in out.roots], nprobe=NPROBE, cg_tol=CG_TOL, cg_maxit=int(cg_maxit),
               weight_band=None if out.weight_band is None else [None if x is None else float(x) for x in out.weight_band], band_note="w at s2 + 2 sigma_s and s2 - 2 sigma_s, from the grid fits alone",
               secant_trace=[dict(weight=float(a), m=float(b), g_se=float(c)) for a, b, c in out.trace], max_secant=MAX_SECANT, warnings=sorted({str(x.message) for x in caught}), status="ok")
    rec["unconverged_fits"] = unconverged
    if unconverged:
        raise RuleFailure(f"item 6: {len(unconverged)} fast-loop fit(s) ended with kkt_w >= {KKT_W_MAX:g} (e.g. w = {unconverged[0]['weight']:.5g}, kkt_w {unconverged[0]['kkt_w']:.3g}); refit with a smaller --gtol", rec)
    first_m = float(out.m_sweep[0])
    if not first_m > 0:
        raise RuleFailure(f"D1: m = {first_m:.4g} <= 0 at the first grid point (w/s2 = {GRID_MAX:g}); no weight is chosen", rec)
    if out.weight is None:
        raise RuleFailure(f"D1: m(w) has no sign change on the grid (m from {float(np.min(out.m_sweep)):.4g} to {float(np.max(out.m_sweep)):.4g}); no weight is chosen and the grid is not extended", rec)
    rec.update(weight=float(out.weight), w_over_s2=float(out.weight) / s2, m=float(out.m), g=float(out.g), g_se=float(out.g_se), chi2=float(out.chi2), h=None if out.h is None else float(out.h),
               h_note="classic MaxEnt h(w) = 2 (w/s2) KL - G, a diagnostic only", n_secant=len(out.trace), crossing_chosen="largest w (D5)")
    if not abs(out.m) <= 3.0 * out.g_se:
        raise RuleFailure(f"D2: |m| = {abs(out.m):.4g} exceeds 3 se(G) = {3 * out.g_se:.4g} at w = {out.weight:.5g} after {len(out.trace)} refit(s) (at most {MAX_SECANT} secant steps)", rec)
    stronger = [r for r, w in zip(results, weights) if w >= out.weight]
    start = refit(float(out.weight), dict((stronger[-1] if stronger else results[0]).values))
    return rec, dict(start.values)


# ---------------------------------------------------------------- final stage (items 5-8, D3)


def final_stage(data, sweep, sigma, start, *, npix, pixel_mas, gtol=1e-8, max_steps=20000, time_limit=None, cg_maxit=2000, log=print):
    """The full data at the sweep's w. Returns ``(record, values)``: the image fit to kkt_w < 1e-4, G checked once against the fast loop, the staged nuisances
    (reference gain at the largest sum of weights, D3), the KKT and band-limit report and the item-8 numbers. ``RuleFailure`` (record attached) on a nuisance within 5% of its box edge
    or kkt_w >= 1e-4 or f_eta/f_floor >= 2 of the joint fit."""
    from virgil.imaging import MaxEntropy
    from virgil.radio import fit_staged_nuisances

    w, s2 = float(sweep["weight"]), float(sweep["s2"])
    ref, sums = reference_obs(data)
    if ref != sigma["ref_obs"]:
        raise SystemExit(f"the reference observation of this data ({ref}) is not the sigma stage's ({sigma['ref_obs']}): another data set")
    prob = build(data, npix=npix, pixel_mas=pixel_mas, flux=sigma["flux_jy"], ref_obs=ref)
    res1, rep1, rounds = fit_point(prob, w, start, gtol=1e-4, max_steps=max_steps, time_limit=time_limit, log=log)
    n_real = 2.0 * len(data.u)
    chi2_nominal = float(sum(res1.info["chi2"]))
    log(f"final: image at w = {w:.5g} (w/s2 {w / s2:.5g}); raw chi2/N {chi2_nominal / n_real:.5g}; G {sweep['g']:.6g} +- {sweep['g_se']:.3g} (the sweep's, on the same data)")
    staged = fit_staged_nuisances(data, prob.scene, prob.priors, [MaxEntropy(w)], init={k: v for k, v in res1.values.items() if k in prob.priors}, ref_obs=ref, edge=EDGE, weight=w,
                                  nuisance=dict(alpha=False, gains=True, offsets=True, sigmas=False, fit_flux=False), chi2="toeplitz", method="lbfgsb", lbfgs_memory=100, gtol=gtol, max_steps=max_steps, time_limit=time_limit)
    s3 = staged.stages[2]
    kkt_w, band = kkt_of(staged.kkt)
    pixels = np.asarray(s3.values["pixels"], float)
    nuis = {k: dict(value=np.asarray(v["value"]), lower=np.asarray(v["lower"]), upper=np.asarray(v["upper"]), edge_frac=np.asarray(v["edge_frac"]), at_edge=np.asarray(staged.edge[k])) for k, v in staged.nuisances.items()}
    rec = dict(stage="final", raw_chi2_over_N=float(sum(s3.info["chi2"])) / n_real, raw_chi2_over_N_nominal=chi2_nominal / n_real, s2=s2, sigma_s=float(sweep["sigma_s"]), g=sweep["g"], g_se=sweep["g_se"],
                              sigma_factors="not fitted (weights rescaled by s2, item 1)", weight=w, w_over_s2=w / s2, psi_hp=psi_hp(pixels, pixel_mas), kkt_w=kkt_w, kkt_w_max=KKT_W_MAX, band_limit=band, band_limit_max=BAND_MAX, f_eta_over_f_floor=band,
               image_rounds=rounds, nuisances=nuis, edge=EDGE, ref_obs=int(staged.ref_obs), sum_weights=sums, flux_jy=float(np.sum(pixels**2)), flux_reference_jy=float(sigma["flux_jy"]), npix=int(npix), pixel_mas=float(pixel_mas),
               converged_report=bool(staged.kkt.get("converged")), status="ok")
    edges = sorted(k for k, v in nuis.items() if v["at_edge"].any())
    if staged.failed or edges:
        raise RuleFailure(f"nuisance(s) {edges} end within {EDGE:.0%} of their prior box edge", rec)
    if not kkt_w < KKT_W_MAX:
        raise RuleFailure(f"kkt_w = {kkt_w:.3g} is not below {KKT_W_MAX:g}; refit with a smaller --gtol", rec)
    if not band < BAND_MAX:
        raise RuleFailure(f"f_eta/f_floor = {band:.3g} is not below {BAND_MAX:g}", rec)
    return rec, dict({k: np.asarray(v) for k, v in s3.values.items()})


def read_json(path):
    return json.loads(pathlib.Path(path).read_text())
