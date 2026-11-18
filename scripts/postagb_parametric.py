"""Parametric disc models of the post-AGB PIONIER targets in the SPARCO scene of ``postagb_imaging.py`` (stages ``ladder``, ``jackknife``, ``select``).

The conventions, models, priors, noise arms, evidence and selection rules are pre-registered in ``design/postagb_parametric.md`` and are not
changed after a fit has been looked at. In short: star (UD and index fixed) + grey point secondary (where the paper has one) + a parametric disc
(``MultipoleDisk`` with a ``GaussianRing`` profile and positive-form harmonics) + a grey over-resolved background, fitted under three noise arms
(N0 as read, N1 MacKay-rescaled, N2 N1 plus night terms) from many starts; a Laplace evidence with the prior curvature of every prior by autodiff,
checked by importance sampling; a ladder M0 (blob), M1 (ring), M2 (m = 1), M2o (+ centre offset), M3 (m = 2) and its selection by
Delta ln Z >= 5 and Delta BIC <= -10 under N2.

    python scripts/postagb_parametric.py ladder OUT --target AI_Sco [--arms N0,N1,N2] [--models M0,M1,M2,M2o] [--variant nosec|secfree|bgfree] [--smoke]
    python scripts/postagb_parametric.py select OUT --target AI_Sco
    python scripts/postagb_parametric.py jackknife OUT --target AI_Sco

Each stage writes its files in ``OUT/<target>/`` and ``<stage>.DONE`` last. Only the names of ``postagb_imaging`` that are the same on ``main`` and on
the gp-polar branch are used (``cfg_of, sparco_scene, scene_priors, load, load_files, chi2r, residual_stats, observables, TARGETS, L0``); the night modes
are rebuilt here (``night_modes``) because ``night_noise`` is absent from ``main``. The one private virgil import (``_residual_jacobian``, ``_log_norm``)
is in ``_private``.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import postagb_imaging as P  # noqa: E402  (sets the XLA/OMP thread defaults and float64 before JAX starts)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from numpyro.infer.util import log_density  # noqa: E402
from scipy.special import gammaln, logsumexp  # noqa: E402
from virgil import OIData, read_oifits  # noqa: E402
from virgil import harmonics as vh  # noqa: E402
from virgil.fitting import fit  # noqa: E402
from virgil.likelihood import numpyro_model, whitened_residuals  # noqa: E402
from virgil.models import GaussianRing, Harmonics, MultipoleDisk  # noqa: E402
from virgil.priors import IsotropicInclination  # noqa: E402
from virgil.spectra import PowerLaw  # noqa: E402

NIGHT_SPLIT_MJD = 0.65  # a night runs from MJD .65 (15:36 UT, local noon at Paranal) to the next
NIGHTW = dict(night_gain_baseline=0.03, night_t3_offset_deg=2.0)  # N2: 3 % on log|V| per (night, baseline), 2 deg per (night, triangle)
ARMS = ("N0", "N1", "N2")
SELECTION_ARM = "N2"
R_RANGE, SIGMA_RANGE = (0.6, 25.0), (0.1, 10.0)  # mas; fixed for every target (design section 3)
R_BLOB = 0.01  # M0: a centred Gaussian is a GaussianRing of this radius
D_LNZ, D_BIC = 5.0, -10.0  # acceptance thresholds of a rung over its parent
ESS_MIN, IS_DRAWS, IS_CHUNK = 200.0, 4000, 100
N_R = 64  # quadrature nodes of the disc profile (design: 64; stop rule S3 allows 48 once; the tests use fewer to compile faster)

# MacKay error scales of the stored gp-polar products (design fact table): the fall-back when the stored image.json is not at hand
MACKAY_S = {"AI_Sco": 1.338, "EN_TrA": 1.283, "HD95767": 0.954, "IRAS15469-5311": 0.814, "IW_Car": 1.003, "PS_Gem": 1.059, "HD108015": 0.890, "HR4049": 0.935}
STORED = Path(os.environ.get("POSTAGB_STORED", "~/code/ozstar_scripts-postagb-noise/results/postagb_noise/out")).expanduser()
STORED_RUNS = ("15b154a+6fa8afe", "04994a0+46ce208")  # the first holds six targets with log_evidence_scene, the second HR 4049 and HD 108015

# model id -> structure. ``degree`` is the highest harmonic order; ``offset`` frees the ring centre.
MODELS = {
    "M0": dict(ring=False, degree=0, offset=False),
    "M1": dict(ring=True, degree=0, offset=False),
    "M2": dict(ring=True, degree=1, offset=False),
    "M2o": dict(ring=True, degree=1, offset=True),
    "M3": dict(ring=True, degree=2, offset=False),
    "M3o": dict(ring=True, degree=2, offset=True),
}
PARENT = {"M1": "M0", "M2": "M1", "M2o": "M2", "M3": "M2", "M3o": "M2o"}
CHAIN = ("M0", "M1", "M2", "M2o", "M3", "M3o")
VARIANTS = (None, "nosec", "secfree", "bgfree")


def _private():
    """The two private virgil functions the evidence needs (Gauss-Newton Jacobian and the marginal nuisances' log-normaliser)."""
    try:
        from virgil.imaging import _log_norm, _residual_jacobian  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise ImportError("postagb_parametric needs virgil.imaging._residual_jacobian and ._log_norm (virgil 0.4.x); this virgil lacks them") from e
    return _log_norm, _residual_jacobian


# ----------------------------------------------------------------------------
# noise arms


def night_labels(data):
    """Night, (night, baseline) cell and (night, triangle) cell of every sample (copied from the gp-polar branch's ``night_labels``)."""
    mjd = float(data.t_ref) + np.asarray(data.dt, float)
    night = np.floor(mjd - NIGHT_SPLIT_MJD).astype(int)
    st = np.sort(np.asarray(data.stations, int), axis=1)
    base = np.unique(st, axis=0, return_inverse=True)[1].ravel()
    cell_u = np.unique(np.stack([night, base], 1), axis=0, return_inverse=True)[1].ravel()
    i1, i2, i3 = (np.asarray(i, int) for i in (data.i_cps1, data.i_cps2, data.i_cps3))
    tri = np.unique(np.sort(np.concatenate([st[i1], st[i2], st[i3]], 1), axis=1), axis=0, return_inverse=True)[1].ravel()
    cell_t = np.unique(np.stack([night[i1], tri], 1), axis=0, return_inverse=True)[1].ravel()
    return dict(u_night=night, u_cell=cell_u, t3_cell=cell_t)


def night_modes(data, gain=NIGHTW["night_gain_baseline"], t3_deg=NIGHTW["night_t3_offset_deg"]):
    """The N2 night modes: ``(g, o)``, one row per (night, baseline) on log|V| of width ``gain`` and one per (night, triangle) on the closure
    phases of width ``t3_deg`` (degrees, returned in radians); a zero width gives ``None`` for its group."""
    lab = night_labels(data)
    g = o = None
    if gain > 0:
        g = np.zeros((lab["u_cell"].max() + 1, lab["u_cell"].size))
        g[lab["u_cell"], np.arange(lab["u_cell"].size)] = gain
    if t3_deg > 0:
        o = np.zeros((lab["t3_cell"].max() + 1, lab["t3_cell"].size))
        o[lab["t3_cell"], np.arange(lab["t3_cell"].size)] = np.radians(t3_deg)
    return g, o


def stored_error_scale(cfg):
    """``(s, source, stored)``: the MacKay error scale of the stored gp-polar ``image.json`` (noise ``current``), else the design table's.

    ``stored`` is the parsed json (or None), kept for the gp-polar comparison."""
    key = cfg.get("data_key", cfg["key"])
    for run in STORED_RUNS:
        f = STORED / run / key / f"{key}_gp-polar_unbounded_image.json"
        if f.exists():
            d = json.load(open(f))
            if d.get("error_scale") is not None:
                return float(d["error_scale"]), str(f), d
    return MACKAY_S[key], "design table (stored image.json not found)", None


def noise_arm(data, cfg, arm, s=None):
    """``data`` under noise arm ``arm``: N0 as read; N1 the errors scaled by the MacKay factor ``s`` (``--noise current`` of gp-polar);
    N2 N1 plus night modes (``--noise nightw``). ``s`` defaults to the stored gp-polar factor of the target."""
    if arm == "N0":
        return data
    s = stored_error_scale(cfg)[0] if s is None else float(s)
    if arm == "N1":
        return data.with_error_scale(s) if s != 1.0 else data
    if arm == "N2":
        g, o = night_modes(data)
        d = data.with_error_scale({"vis": s, "phi": s})
        return d.with_gains(telescope=0.0, baseline=0.0, modes=g).with_closure_offsets(triangle=np.radians(0.0), modes=o)
    raise ValueError(f"unknown noise arm {arm!r}")


# ----------------------------------------------------------------------------
# scene, priors, starts


def _cfg_variant(cfg, variant):
    """The target configuration of a variant: the bg index fixed at 0 (free for ``bgfree``), no secondary for ``nosec``."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    c = cfg | dict(bg_index_fixed=None if variant == "bgfree" else 0.0)
    return c | dict(secondary=False) if variant == "nosec" else c


def build_scene(cfg, model_id, start, variant=None):
    """The SPARCO scene of ``model_id`` with its parameters at ``start`` (a dict, see ``starts_for``): ``P.sparco_scene`` with a ``MultipoleDisk``
    environment, a ``Resolved`` background with fixed index, and the secondary where the paper has one (not for ``nosec``)."""
    spec = MODELS[model_id]
    deg = spec["degree"]
    harm = Harmonics(re=[float(x) for x in start["re"][:deg]], im=[float(x) for x in start["im"][:deg]], form="positive", degree=deg) if deg else None
    prof = GaussianRing(float(start["r"]) if spec["ring"] else R_BLOB, float(start["sigma"]))
    c = _cfg_variant(cfg, variant) | dict(f_bg0=float(start["f_bg"]))
    disk = MultipoleDisk(prof, harmonics=harm, inc=float(start["inc"]), pa=float(start["pa"]),
                         dra=float(start.get("dra", 0.0)) if spec["offset"] else 0.0, ddec=float(start.get("ddec", 0.0)) if spec["offset"] else 0.0,
                         n_r=N_R, flux=PowerLaw(float(start["f_env"]), float(start["d_env"]), P.L0))
    scene = P.sparco_scene(c, disk, f_env=float(start["f_env"]), index=float(start["d_env"]))
    for k, path in (("sec_dra", "sec.dra"), ("sec_ddec", "sec.ddec"), ("sec_ratio", "sec.flux.ratio")):
        if start.get(k) is not None and c.get("sec0") is not None and c.get("secondary", True):
            scene = scene.set(path, float(start[k]))
    return scene


def priors_for(cfg, scene, model_id, start, variant=None):
    """The priors of the design (section 3): ``P.scene_priors`` unchanged for the fluxes and the secondary, plus the disc. ``start['pa']`` centres the
    180 deg window of the position angle. ``secfree`` replaces the secondary's position prior by Normal(PMOIRED, 1 mas)."""
    spec = MODELS[model_id]
    c = _cfg_variant(cfg, variant)
    pri = dict(P.scene_priors(c, scene))
    pri["env.inc"] = IsotropicInclination(0.0, 85.0)
    pa0 = float(start["pa"])
    pri["env.pa"] = dist.Uniform(pa0 - 90.0, pa0 + 90.0)
    if spec["ring"]:
        pri["env.profile.radius"] = dist.LogUniform(*R_RANGE)
    pri["env.profile.sigma"] = dist.LogUniform(*SIGMA_RANGE)
    if spec["degree"]:
        for n in ("re", "im"):
            pri[f"env.harmonics.{n}"] = dist.Normal(jnp.zeros(spec["degree"]), 1.0)
    if spec["offset"]:
        pri["env.dra"], pri["env.ddec"] = dist.Normal(0.0, 1.0), dist.Normal(0.0, 1.0)
    if variant == "secfree" and "sec.dra" in pri and "sec_prior" in c:
        sp = c["sec_prior"]
        pri["sec.dra"], pri["sec.ddec"] = dist.Normal(sp["dra"], 1.0), dist.Normal(sp["ddec"], 1.0)
    return pri


def geometry_refs(cfg):
    """Named geometries ``name -> (theta_rim diameter mas, inc deg, PA deg)``: the De Prins image, their PMOIRED fit (where there is one) and the
    geometry of the stored gp-polar run (``<T>_scale.json``, where the file exists)."""
    import postagb_paper as PP  # noqa: PLC0415

    key = cfg["key"]
    refs = {}
    if key in PP.IMAGE:
        im = PP.IMAGE[key]
        refs["deprins_image"] = (im["theta_rim"][0], im["i_rim"][0], im["pa_rim"][0])
    elif cfg.get("paper") and "rim_diam_mas" in cfg["paper"]:  # IW Car is in neither dict
        p = cfg["paper"]
        refs["deprins_image"] = (p["rim_diam_mas"], p["rim_inc_deg"], p["rim_pa_deg"])
    if key in PP.PMOIRED and "theta_rim" in PP.PMOIRED[key]:
        pm = PP.PMOIRED[key]
        refs["pmoired"] = (pm["theta_rim"][0], pm["i_rim"][0], pm["pa_rim"][0])
    dk = cfg.get("data_key", key)
    for run in STORED_RUNS:
        f = STORED / run / dk / f"{dk}_scale.json"
        if f.exists():
            g = json.load(open(f)).get("geometry")
            if g and "radius_mas" in g:
                refs["stored_gp"] = (2.0 * g["radius_mas"], g["inc_deg"], g["pa_deg"])
            break
    return refs


def header_fluxes(cfg):
    """Start fluxes from the published FITS header: ``(f_env, d_env)`` with f_env = (1 - f_prim - f_sec) / f_prim and d_env = SIND0."""
    hdr, _, _ = P.zenodo_header(cfg)
    f1, f2 = float(hdr["SFLU1"]), float(hdr.get("SFLU2", 0.0) or 0.0)
    return max((1.0 - f1 - f2) / f1, 0.05), float(hdr["SIND0"])


def starts_for(cfg, model_id, smoke=False, refs=None, fluxes=None):
    """The starts of one (model, arm): 18 of inc {20, 45, 65} x PA {0, 60, 120} x r0 {0.7, 1.4} x (image theta_rim / 2) plus the named geometries
    (De Prins image, PMOIRED, stored gp-polar) = up to 21; M0 uses the nine (inc, PA) pairs at r0 = theta_rim / 2 and the named ones. ``smoke``: the
    named starts and then grid starts up to four. Each is a dict (inc, pa, r, sigma, re, im, f_env, d_env, f_bg, name)."""
    spec = MODELS[model_id]
    refs = geometry_refs(cfg) if refs is None else refs
    f_env, d_env = header_fluxes(cfg) if fluxes is None else fluxes
    r_ref = refs["deprins_image"][0] / 2.0
    deg = spec["degree"]

    def mk(name, inc, pa, r):
        r = float(np.clip(r, *R_RANGE)) if spec["ring"] else r
        sigma = float(np.clip(0.3 * r if spec["ring"] else 0.6 * r_ref, *SIGMA_RANGE))
        return dict(name=name, inc=float(inc), pa=float(pa), r=r, sigma=sigma, re=[0.1] * deg, im=[0.1] * deg, dra=0.0, ddec=0.0,
                    f_env=f_env, d_env=d_env, f_bg=0.05)

    named = [mk(n, i, p, t / 2.0) for n, (t, i, p) in refs.items()]
    ks = (0.7, 1.4) if spec["ring"] else (1.0,)
    grid = [mk(f"grid_i{i}_p{p}_r{k}", i, p, k * r_ref) for i in (20, 45, 65) for p in (0, 60, 120) for k in ks]
    out = (named + grid[: max(0, 4 - len(named))]) if smoke else grid + named
    return out


def start_from_values(values, model_id, cfg=None, name="parent"):
    """A start dict of ``model_id`` from a fitted parent's ``values`` (path -> number or list): shared parameters are copied, new ones
    (higher harmonic orders 0.1 + 0.1i, the centre offset 0) take their defaults."""
    spec = MODELS[model_id]
    deg = spec["degree"]

    def v(p, default=None):
        x = values.get(p, default)
        return default if x is None else x

    re, im = list(np.atleast_1d(v("env.harmonics.re", []))), list(np.atleast_1d(v("env.harmonics.im", [])))
    re += [0.1] * (deg - len(re))
    im += [0.1] * (deg - len(im))
    st = dict(name=name, inc=float(v("env.inc")), pa=float(v("env.pa")), r=float(v("env.profile.radius", R_BLOB)), sigma=float(v("env.profile.sigma")),
              re=re[:deg], im=im[:deg], dra=float(v("env.dra", 0.0)), ddec=float(v("env.ddec", 0.0)), f_env=float(v("env.flux.ratio")),
              d_env=float(v("env.flux.index")), f_bg=float(v("bg.flux.ratio", 0.05)))
    if not spec["ring"]:
        st["r"] = R_BLOB
    for k, p in (("sec_dra", "sec.dra"), ("sec_ddec", "sec.ddec"), ("sec_ratio", "sec.flux.ratio")):
        if p in values:
            st[k] = float(values[p])
    return st


# ----------------------------------------------------------------------------
# evidence


def _flat(model, paths):
    """``(theta, shapes)``: the free parameters at ``model`` in the order of ``paths``, flattened."""
    shapes = [np.shape(np.asarray(model.get(p), float)) for p in paths]
    theta = np.concatenate([np.ravel(np.asarray(model.get(p), float)) for p in paths])
    return theta, shapes


def _names(paths, shapes):
    return [p if sh == () else f"{p}[{i}]" for p, sh in zip(paths, shapes) for i in range(int(np.prod(sh, dtype=int)))] if shapes else []


def prior_curvature(prior, x):
    """-d^2 ln p / d theta^2 at ``x`` (a scalar or an array of independent parameters) by autodiff, as a flat vector."""
    x = np.asarray(x, float)
    f = lambda xf: jnp.sum(prior.log_prob(xf.reshape(x.shape)))  # noqa: E731
    h = jax.hessian(f)(jnp.asarray(x.ravel()))
    return -np.asarray(jnp.diag(h), float)


def _scene_rule_lambda(prior, x):
    """The Lambda of ``scene_evidence`` in gp-polar: 1 / sigma^2 for a Normal prior, 0 for anything else."""
    x = np.asarray(x, float)
    if type(prior).__name__ != "Normal":
        return np.zeros(x.size)
    return np.broadcast_to(1.0 / np.asarray(prior.scale, float) ** 2, x.shape).ravel()


def param_evidence(model, priors, data):
    """Laplace evidence of a fitted scene with every free parameter integrated, in the coordinates of ``priors``.

    With theta the k free parameters at ``model``, r the whitened residuals, J their Jacobian and L the marginalised nuisances' log-normaliser:
    ln Z = -r.r/2 - L + sum ln p(theta) + (k/2) ln 2 pi - (1/2) ln det(J^T J + Lambda), with Lambda = -d^2 ln p / d theta^2 by autodiff for every prior
    (Normal and TruncatedNormal 1 / sigma^2, Uniform 0, LogUniform -1 / theta^2, the sin i prior its own). ``lnZ_scene_rule`` uses gp-polar's rule
    (Lambda = 1 / sigma^2 for a Normal prior, else 0) and is for the comparison with it only. Returns a dict with the two evidences, ``sign`` and
    ``logdet`` of the Hessian, the posterior covariance ``cov``, ``sd`` per parameter ``names``, and what ``is_evidence`` needs."""
    _log_norm, _residual_jacobian = _private()
    paths = tuple(priors)
    theta, shapes = _flat(model, paths)
    names = _names(paths, shapes)
    r, jac = _residual_jacobian(model, data, paths, on_device=True)
    r, jac = np.asarray(r, float), np.asarray(jac, float)
    if jac.shape[1] != theta.size:
        raise ValueError(f"param_evidence: the Jacobian has {jac.shape[1]} columns for {theta.size} parameters")
    norm = _log_norm(model, data)
    lam, lam_rule, log_p, off = [], [], 0.0, 0
    for p, sh in zip(paths, shapes):
        n = int(np.prod(sh, dtype=int))
        x = theta[off: off + n].reshape(sh)
        lam.append(prior_curvature(priors[p], x))
        lam_rule.append(_scene_rule_lambda(priors[p], x))
        log_p += float(np.sum(np.asarray(priors[p].log_prob(x), float)))
        off += n
    lam, lam_rule = np.concatenate(lam), np.concatenate(lam_rule)
    k = theta.size
    g0 = -0.5 * float(r @ r) - norm + log_p
    jtj = jac.T @ jac

    def lap(lmb):
        sign, logdet = np.linalg.slogdet(jtj + np.diag(lmb))
        return (g0 + 0.5 * k * np.log(2.0 * np.pi) - 0.5 * logdet if sign > 0 else float("-inf")), sign, logdet

    lnz, sign, logdet = lap(lam)
    lnz_rule = lap(lam_rule)[0]
    cov = np.linalg.inv(jtj + np.diag(lam)) if sign > 0 else np.full((k, k), np.nan)
    sd = np.sqrt(np.where(np.diag(cov) > 0, np.diag(cov), np.nan))
    return dict(lnZ_lap=float(lnz), lnZ_scene_rule=float(lnz_rule), sign=float(sign), logdet=float(logdet), cov=cov, sd=sd, names=names, paths=paths,
                shapes=shapes, theta=theta, lam=lam, chi2=float(r @ r), n_res=int(r.size), log_norm=float(norm), log_prior=float(log_p), g0=float(g0), k=k,
                corr=cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov))) if sign > 0 else None)


def _log_target(model, priors, data, paths, shapes):
    """``lj(flat) -> ln p(theta) + ln L(theta) + const`` of numpyro's own log density, for a flat parameter vector in the order of ``paths``."""
    nm = numpyro_model(model, priors, data)
    sizes = [int(np.prod(sh, dtype=int)) for sh in shapes]
    offs = np.concatenate([[0], np.cumsum(sizes)])

    def lj(flat):
        params = {p: flat[o: o + n].reshape(sh) for p, sh, o, n in zip(paths, shapes, offs[:-1], sizes)}
        return log_density(nm, (), {}, params)[0]

    return lj


def _in_support(priors, paths, shapes, x):
    """Boolean per row of ``x`` (n, k): every parameter inside its prior's support."""
    ok, off = np.ones(len(x), bool), 0
    for p, sh in zip(paths, shapes):
        n = int(np.prod(sh, dtype=int))
        sup = getattr(priors[p], "support", None)
        if sup is not None:
            ok &= np.all(np.asarray(sup(jnp.asarray(x[:, off: off + n]))).reshape(len(x), -1), axis=1)
        off += n
    return ok


def is_evidence(model, priors, data, ev, n=IS_DRAWS, chunk=IS_CHUNK, seed=0, df=5.0, scale=1.5):
    """Importance-sampling evidence in the coordinates of ``priors``: a multivariate Student-t (``df`` degrees of freedom) centred on the optimum
    with scale matrix ``scale``^2 (J^T J + Lambda)^-1 proposes ``n`` draws; the target is numpyro's ``log_density`` of ``numpyro_model`` minus its
    constant (the constant being its value at the optimum less ``ev['g0']``), evaluated by jit + vmap in chunks of ``chunk``. Draws outside the
    prior support or with a non-finite density count as zero weight. Returns ``lnZ_is``, the effective sample size ``ess`` and ``frac_outside``."""
    paths, shapes, theta, cov = ev["paths"], ev["shapes"], ev["theta"], ev["cov"]
    k = theta.size
    lc = np.linalg.cholesky(scale**2 * cov)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, k))
    w = rng.chisquare(df, n) / df
    x = theta + (z @ lc.T) / np.sqrt(w)[:, None]
    inside = _in_support(priors, paths, shapes, x)
    x_eval = np.where(inside[:, None], x, theta)  # out-of-support draws are evaluated at the optimum and then discarded
    lj = _log_target(model, priors, data, paths, shapes)
    f = jax.jit(jax.vmap(lj))
    const = float(f(jnp.asarray(theta)[None])[0]) - ev["g0"]
    lt = np.concatenate([np.asarray(f(jnp.asarray(x_eval[i: i + chunk]))) for i in range(0, n, chunk)]) - const
    lt = np.where(inside & np.isfinite(lt), lt, -np.inf)
    dx = np.linalg.solve(lc, (x - theta).T).T
    m2 = np.sum(dx**2, axis=1)
    lq = gammaln((df + k) / 2) - gammaln(df / 2) - 0.5 * k * np.log(df * np.pi) - np.sum(np.log(np.diag(lc))) - 0.5 * (df + k) * np.log1p(m2 / df)
    lw = lt - lq
    if not np.isfinite(lw).any():
        return dict(lnZ_is=float("-inf"), ess=0.0, frac_outside=float(1.0 - inside.mean()), n=int(n))
    mx = lw.max()
    ww = np.exp(lw - mx)
    return dict(lnZ_is=float(mx + np.log(ww.mean())), ess=float(ww.sum() ** 2 / (ww**2).sum()), frac_outside=float(1.0 - inside.mean()), n=int(n))


def bic(own_chi2, log_norm, k, n):
    """BIC = N x (own chi^2 per point) + 2 L + k ln N; ``n`` is the row count of ``P.chi2r`` (the same N as gp-polar's ``n_points``)."""
    return float(n * own_chi2 + 2.0 * log_norm + k * np.log(n))


def chi2s(model, data_raw, data_arm):
    """The three chi^2 of design section 1: ``raw`` (quoted errors, independent, no gains; N its row count), ``raw_v2``, ``raw_cp`` (per point,
    from ``P.residual_stats``) and ``own`` (r.r / N under the arm's own likelihood, ``n`` rows)."""
    raw, n_raw = P.chi2r(model, data_raw)
    v2, t3 = P.observables(model, data_raw)
    st = P.residual_stats(data_raw, v2, t3)
    r = np.asarray(whitened_residuals(model, data_arm))
    return dict(raw=raw, raw_v2=st["chi2_v2"], raw_cp=st["chi2_t3"], own=float(r @ r / r.size), n=int(r.size), n_raw=int(n_raw),
                n_independent=int(data_arm.n_independent))


# ----------------------------------------------------------------------------
# derived quantities and eligibility


def _opt(model, path, default=0.0):
    try:
        return float(np.asarray(model.get(path), float))
    except Exception:  # noqa: BLE001  (the component or leaf is not in this scene)
        return default


def derived(model, cfg=None):
    """Quantities of a fitted scene: inc, pa_mod180, r0, sigma, hole_contrast, linear harmonics ``c``, ``s`` (arrays), ``a1`` = sqrt(c1^2 + s1^2),
    ``bright_pa`` (on-sky PA, 0-360, of the brightest point of the rendered disc about its centre; None without harmonics), flux percentages of the
    total at 1.65 um, ``d_env`` and the ring ``centre`` (dra, ddec)."""
    env = model.get("env")
    inc, pa = _opt(model, "env.inc"), _opt(model, "env.pa")
    r0, sig = _opt(model, "env.profile.radius", R_BLOB), _opt(model, "env.profile.sigma")
    fl = dict(star=1.0, env=_opt(model, "env.flux.ratio"), bg=_opt(model, "bg.flux.ratio"), sec=_opt(model, "sec.flux.ratio"))
    tot = sum(fl.values())
    out = dict(inc=inc, pa_mod180=pa % 180.0, r0=r0, sigma=sig, hole_contrast=float(1.0 - np.exp(-0.5 * (r0 / sig) ** 2)) if r0 > R_BLOB else 0.0,
               d_env=_opt(model, "env.flux.index"), centre=[_opt(model, "env.dra"), _opt(model, "env.ddec")],
               f_prim_pct=100 * fl["star"] / tot, f_sec_pct=100 * fl["sec"] / tot, f_disc_pct=100 * fl["env"] / tot, f_bg_pct=100 * fl["bg"] / tot)
    h = env.harmonics
    if h is not None and h.degree:
        c, s = vh.linear_from_image(vh.to_image_harmonics(h))
        c, s = np.asarray(c, float), np.asarray(s, float)
        out |= dict(c=c.tolist(), s=s.tolist(), a1=float(np.hypot(c[0], s[0])), bright_pa=bright_pa(env))
    else:
        out |= dict(c=[], s=[], a1=0.0, bright_pa=None)
    return out


def bright_pa(env, max_npix=768):
    """On-sky position angle (North through East, 0-360) of the brightest pixel of the rendered disc ``env`` about its centre."""
    e = env.set("dra", 0.0).set("ddec", 0.0)
    r0, sig = float(np.asarray(e.profile.radius)), float(np.asarray(e.profile.sigma))
    fov = 2.4 * (r0 + 4.0 * sig)
    npix = int(np.clip(fov / (0.4 * sig), 128, max_npix))
    img = np.asarray(e.render(npix, fov))
    row, col = np.unravel_index(np.argmax(img), img.shape)
    pix, c = fov / npix, (npix - 1) / 2.0
    east, north = (c - col) * pix, (c - row) * pix  # column 0 is East, row 0 is North
    return float(np.degrees(np.arctan2(east, north)) % 360.0)


def _near_bound(prior, x, frac=0.01):
    """Whether ``x`` is within ``frac`` of the range of a finite prior bound (log range for a LogUniform)."""
    sup = getattr(prior, "support", None)
    lo, hi = getattr(sup, "lower_bound", None), getattr(sup, "upper_bound", None)
    if lo is None or hi is None or not (np.isfinite(float(lo)) and np.isfinite(float(hi))):
        return False
    x = np.asarray(x, float)
    if type(prior).__name__ == "LogUniform":
        x, lo, hi = np.log(x), np.log(float(lo)), np.log(float(hi))
    lo, hi = float(lo), float(hi)
    return bool(np.any((x - lo < frac * (hi - lo)) | (hi - x < frac * (hi - lo))))


def eligible(info, model, priors, ev):
    """``(ok, why_not, flags)`` of a fit (design section 5). E1 converged; E2 ``info['at_bound']`` empty and no parameter within 1 % of a prior
    bound (the PA is excepted; a radius on its floor means "this is M0"); E3 det(J^T J + Lambda) > 0, finite Laplace sds, ``min_intensity() >= 0``.
    Flags (not exclusions): ``full_modulation`` (a1 > 0.95), ``pa_unconstrained`` (Laplace sd of PA > 30 deg)."""
    why = []
    if not info.get("converged"):
        why.append(f"E1: not converged ({info.get('stop')})")
    bound = [b for b in (info.get("at_bound") or ()) if b != "env.pa"]
    if bound:
        why.append(f"E2: at_bound {bound}")
    for p in priors:
        if p != "env.pa" and _near_bound(priors[p], model.get(p)):
            why.append(f"E2: {p} within 1 % of a prior bound" + (" (radius floor: this is M0)" if p == "env.profile.radius" else ""))
    if ev is None or not ev["sign"] > 0 or not np.all(np.isfinite(ev["sd"])):
        why.append("E3: Hessian not positive definite or non-finite sds")
    try:
        if float(np.asarray(model.get("env").min_intensity())) < 0.0:
            why.append("E3: negative intensity")
    except Exception:  # noqa: BLE001
        pass
    flags = []
    d = derived(model)
    if d["a1"] > 0.95:
        flags.append("full_modulation")
    if ev is not None and "env.pa" in ev["names"] and np.isfinite(ev["sd"][ev["names"].index("env.pa")]) and ev["sd"][ev["names"].index("env.pa")] > 30.0:
        flags.append("pa_unconstrained")
    return not why, why, flags


# ----------------------------------------------------------------------------
# the ladder


def _num(x):
    x = np.asarray(x)
    return float(x) if x.ndim == 0 else x.astype(float).tolist()


def values_of(result):
    return {k: _num(v) for k, v in result.values.items()}


def run_fit(scene, priors, data, time_limit=60.0, max_steps=2000):
    """One fit as the design prescribes: L-BFGS/LM by virgil's default; a fit that did not converge (and is not on a bound) is refitted once from
    where it ended with ``method='lbfgsb'``. Returns ``(result or None, record)``. ``own_chi2`` is the fit's chi^2 over the same N rows as ``P.chi2r``
    (virgil's ``info['chi2_red']`` divides by the independent observables only: 1512 of 2184 rows on AI Sco)."""
    t0 = time.time()
    try:
        res = fit(scene, priors, data, max_steps=max_steps, time_limit=time_limit)
        if not res.info.get("converged") and not res.info.get("at_bound"):
            res2 = fit(res.model, priors, data, method="lbfgsb", max_steps=max_steps, time_limit=time_limit)
            if res2.info.get("converged") or float(res2.info["loss"]) < float(res.info["loss"]):
                res = res2
    except Exception as e:  # noqa: BLE001  (a start that cannot be fitted is recorded, not fatal)
        return None, dict(converged=False, stop=f"exception: {type(e).__name__}: {str(e)[:120]}", at_bound=[], seconds=time.time() - t0)
    rec = dict(converged=bool(res.info.get("converged")), stop=res.info.get("stop"), at_bound=list(res.info.get("at_bound") or ()),
               own_chi2=float(np.sum(res.info["chi2"])) / data.n_residuals, loss=float(res.info["loss"]), seconds=time.time() - t0)
    return res, rec


def same_mode(a, b):
    """Two fits are the same mode when |d inc| < 5 deg, |d PA mod 180| < 10 deg and |d ln r0| < 0.1."""
    dpa = abs((a["env.pa"] - b["env.pa"] + 90.0) % 180.0 - 90.0)
    ok = abs(a["env.inc"] - b["env.inc"]) < 5.0 and dpa < 10.0
    if "env.profile.radius" in a and "env.profile.radius" in b:
        ok &= abs(np.log(a["env.profile.radius"] / b["env.profile.radius"])) < 0.1
    return bool(ok)


def fit_one_model(cfg, data_raw, data_arm, arm, model_id, variant=None, smoke=False, do_is=True, extra_starts=(), time_limit=60.0, chunk=IS_CHUNK, log=print):
    """All starts of one (model, arm): fit each (one structure, so the compiled objective is reused), cluster the converged fits into distinct modes,
    compute the Laplace evidence and eligibility of every mode and the importance-sampling check of the best eligible one (N1 and N2 only).
    Returns the dict written as ``<T>_param_<model>_<arm>[_<variant>].json`` (without bookkeeping fields)."""
    t_all = time.time()
    spec = MODELS[model_id]
    starts = list(extra_starts) + starts_for(cfg, model_id, smoke=smoke)
    recs, fits = [], []
    for st in starts:
        scene = build_scene(cfg, model_id, st, variant)
        pri = priors_for(cfg, scene, model_id, st, variant)
        res, rec = run_fit(scene, pri, data_arm, time_limit=time_limit)
        if res is not None and res.info.get("converged") and abs(abs(float(res.values["env.pa"]) - st["pa"]) - 90.0) < 1.8:
            st2 = dict(st, pa=float(res.values["env.pa"]))  # re-centre the PA window on the end point and refit once
            res2, rec2 = run_fit(res.model, priors_for(cfg, res.model, model_id, st2, variant), data_arm, time_limit=time_limit)
            if res2 is not None and res2.info.get("converged"):
                res, pri, rec = res2, priors_for(cfg, res2.model, model_id, st2, variant), dict(rec2, seconds=rec["seconds"] + rec2["seconds"], recentred=True)
        rec = dict(start=st["name"], **rec)
        if res is not None:
            rec |= dict(values=values_of(res), raw_chi2=P.chi2r(res.model, data_raw)[0])
        recs.append(rec)
        fits.append((res, pri))
        log(f"  {model_id} {arm} {st['name']}: converged {rec['converged']} own chi2 {rec.get('own_chi2', float('nan')):.3f} {rec['seconds']:.1f} s")
    # distinct modes among the converged fits, best (lowest loss) first within a cluster
    order = sorted((i for i, (r, _) in enumerate(fits) if r is not None and recs[i]["converged"]), key=lambda i: recs[i]["loss"])
    clusters = []
    for i in order:
        for cl in clusters:
            if same_mode(recs[cl[0]]["values"], recs[i]["values"]):
                cl.append(i)
                break
        else:
            clusters.append([i])
    modes = []
    for cl in clusters:
        res, pri = fits[cl[0]]
        try:
            ev = param_evidence(res.model, pri, data_arm)
        except Exception as e:  # noqa: BLE001
            ev = None
            log(f"  evidence failed: {type(e).__name__}: {e}")
        ok, why, flags = eligible(recs[cl[0]], res.model, pri, ev)
        c2 = chi2s(res.model, data_raw, data_arm)
        k = len(ev["theta"]) if ev else sum(int(np.size(res.values[p])) for p in pri)
        modes.append(dict(values=values_of(res), sd=dict(zip(ev["names"], map(float, ev["sd"]))) if ev else None,
                          lnZ_lap=ev["lnZ_lap"] if ev else float("-inf"), lnZ_scene_rule=ev["lnZ_scene_rule"] if ev else float("-inf"),
                          bic=bic(c2["own"], ev["log_norm"], k, c2["n"]) if ev else float("inf"), eligible=bool(ok), why_not=why, flags=flags,
                          n_starts=len(cl), start=recs[cl[0]]["start"], _res=res, _pri=pri, _ev=ev, _chi2=c2))
    modes.sort(key=lambda m: (not m["eligible"], -m["lnZ_lap"]))
    best = None
    elig = [m for m in modes if m["eligible"]]
    if elig:
        m = elig[0]
        ev, res, pri, c2 = m["_ev"], m["_res"], m["_pri"], m["_chi2"]
        isr = dict(lnZ_is=None, ess=None, frac_outside=None)
        if do_is and arm in ("N1", "N2"):
            try:
                isr = is_evidence(res.model, pri, data_arm, ev, chunk=chunk)
            except Exception as e:  # noqa: BLE001
                log(f"  IS failed: {type(e).__name__}: {e}")
        flags = list(m["flags"])
        use, lap = None, m["lnZ_lap"]
        if isr["ess"] is not None and isr["ess"] >= ESS_MIN and np.isfinite(isr["lnZ_is"]):
            use = isr["lnZ_is"]
            if abs(isr["lnZ_is"] - lap) > 1.0:
                flags.append("laplace_poor")
        else:
            use = lap
            if arm in ("N1", "N2") and do_is:
                flags.append("evidence_unverified")
        near = [x for x in elig[1:] if lap - x["lnZ_lap"] < 5.0 and not same_mode(m["values"], x["values"])]
        if near:
            flags.append("multimodal")
        d = derived(res.model, cfg)
        best = dict(values=m["values"], sd=m["sd"], corr=ev["corr"].tolist() if ev["corr"] is not None else None, raw_chi2=c2["raw"], raw_chi2_v2=c2["raw_v2"],
                    raw_chi2_cp=c2["raw_cp"], own_chi2=c2["own"], log_norm=ev["log_norm"], lnZ_lap=lap, lnZ_scene_rule=m["lnZ_scene_rule"],
                    lnZ_is=isr["lnZ_is"], ess=isr["ess"], frac_outside=isr["frac_outside"], lnZ_use=use,
                    lnZ_modes=float(logsumexp([x["lnZ_lap"] for x in elig])), bic=m["bic"], flags=flags, n_rows=c2["n"], n_raw=c2["n_raw"],
                    n_independent=c2["n_independent"], derived=d, names=ev["names"])
    for m in modes:
        for k in ("_res", "_pri", "_ev", "_chi2"):
            m.pop(k)
    k_free = len(best["names"]) if best else (len(modes[0]["values"]) if modes else None)
    return dict(model=model_id, arm=arm, variant=variant, k=k_free, n_points=best["n_rows"] if best else None,
                priors={p: str(v) for p, v in (priors_for(cfg, build_scene(cfg, model_id, starts[0], variant), model_id, starts[0], variant)).items()},
                starts=recs, modes=modes, best=best, n_unconverged=sum(not r["converged"] for r in recs), seconds=time.time() - t_all)


def fit_ladder(cfg, data, arms=ARMS, models=("M0", "M1", "M2", "M2o"), variant=None, smoke=False, do_is=True, out=None, log=print, s=None, time_limit=60.0,
               prior=None, extra=None):
    """The ladder: every model in ``models`` under every arm in ``arms``, each with the full multi-start. M2o and M3 also start from the two best
    eligible modes of their parent in the same arm (``prior`` holds ladder dicts of an earlier run, keyed ``(model, arm)``, for a parent that is not
    in ``models``). ``out`` (a directory) gets one json per (model, arm). Returns ``{(model, arm): dict}``."""
    results = {}
    s_val, s_src, _ = stored_error_scale(cfg) if s is None else (float(s), "argument", None)
    for arm in arms:
        d_arm = noise_arm(data, cfg, arm, s=s_val)
        for mid in models:
            par = results.get((PARENT.get(mid), arm)) or (prior or {}).get((PARENT.get(mid), arm))
            ex = []
            if mid in ("M2o", "M3", "M3o") and par and par.get("modes"):
                ex = [start_from_values(m["values"], mid, cfg, name=f"parent_{PARENT[mid]}_mode{i}") for i, m in enumerate(par["modes"][:2]) if m["eligible"]]
            log(f"{cfg['key']} {mid} {arm}: {len(ex)} parent starts")
            r = fit_one_model(cfg, data, d_arm, arm, mid, variant, smoke=smoke, do_is=do_is and not smoke, extra_starts=ex, time_limit=time_limit, log=log)
            r |= dict(error_scale=s_val, error_scale_source=s_src, smoke=bool(smoke), **(extra or {}))
            results[(mid, arm)] = r
            if out is not None:
                write_model_json(out, cfg, r)
    return results


def commits():
    """Git heads of this script's repository and of virgil (for the provenance of every json)."""
    def head(cwd):
        try:
            return subprocess.run(["git", "-C", str(cwd), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20).stdout.strip() or "unknown"
        except Exception:  # noqa: BLE001
            return "unknown"

    import virgil  # noqa: PLC0415

    return dict(scripts=head(Path(__file__).resolve().parent), virgil=head(Path(virgil.__file__).resolve().parents[2]), virgil_version=getattr(virgil, "__version__", None))


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, float) and not np.isfinite(x):
        return None if np.isnan(x) else ("Infinity" if x > 0 else "-Infinity")
    return x


def model_path(out, key, model, arm, variant=None):
    return Path(out) / (f"{key}_param_{model}_{arm}" + (f"_{variant}" if variant else "") + ".json")


def write_model_json(out, cfg, r):
    Path(out).mkdir(parents=True, exist_ok=True)
    d = dict(target=cfg["key"], commits=commits(), **r)
    with open(model_path(out, cfg["key"], r["model"], r["arm"], r["variant"]), "w") as f:
        json.dump(_jsonable(d), f, indent=1)


# ----------------------------------------------------------------------------
# selection


def _circ180(a, b):
    return abs((a - b + 90.0) % 180.0 - 90.0)


def _pair(parent, child, arm):
    """The acceptance test of ``child`` over ``parent`` (two ladder dicts) under ``arm``."""
    pb, cb = (parent or {}).get("best"), (child or {}).get("best")
    row = dict(parent=(parent or {}).get("model"), child=(child or {}).get("model"))
    row["pair"] = f"{row['child']}/{row['parent']}"
    if cb is None:
        return row | dict(verdict="child has no eligible fit", accepted=False)
    if pb is None:
        return row | dict(verdict="no eligible parent: child accepted by default", accepted=True, d_bic=None, d_lnZ=None)
    d_bic = cb["bic"] - pb["bic"]
    row |= dict(d_bic=float(d_bic))
    if arm == "N0":
        return row | dict(d_lnZ=None, beyond_deprins_100=bool(d_bic <= -100.0), accepted=bool(d_bic <= D_BIC), verdict="BIC only")
    unv = "evidence_unverified" in (pb["flags"] + cb["flags"])
    key = "lnZ_lap" if unv else "lnZ_use"
    d_z = cb[key] - pb[key]
    acc = bool(d_z >= D_LNZ and d_bic <= D_BIC)
    return row | dict(d_lnZ=float(d_z), basis=key, evidence_unverified=unv, accepted=acc, verdict="accepted" if acc else "rejected")


def select(ladders, arm, models=CHAIN):
    """The selection chain under ``arm``: a rung is accepted over the last accepted rung when Delta lnZ_use >= 5 and Delta BIC <= -10 (both
    eligible; if either has ``evidence_unverified`` the same thresholds apply to ``lnZ_lap`` and the pair is flagged). Order M1, M2, M2o, then M3, M3o.
    ``ladders`` maps ``(model, arm)`` to the ladder dicts (json). Returns ``dict(chain, selected, best, is_ring, flags)`` (None selected if M0 has
    no eligible fit and nothing else does either)."""
    avail = [m for m in models if (m, arm) in ladders]
    if not avail:
        return dict(chain=[], selected=None)
    base = avail[0]
    cur, chain = base, []
    for m in avail[1:]:
        row = _pair(ladders[(cur, arm)], ladders[(m, arm)], arm)
        row["against"] = cur
        row["pair"] = f"{m}/{cur}"
        row["dlnZ"], row["dbic"] = row.get("d_lnZ"), row.get("d_bic")  # the names the report stage reads
        row["flags"] = (["evidence_unverified"] if row.get("evidence_unverified") else [])
        chain.append(row)
        if row["accepted"]:
            cur = m
    best = (ladders[(cur, arm)] or {}).get("best")
    out = dict(chain=chain, selected=cur, best=best, base=base)
    if best is not None:
        d = best["derived"]
        out |= dict(is_ring=bool(cur != "M0" and d["hole_contrast"] >= 0.5), hole_contrast=d["hole_contrast"], flags=best["flags"])
    else:
        out |= dict(is_ring=False, flags=["no eligible fit"])
    return out


def arm_dependent(sel_a, sel_b):
    """Whether two selections differ in model or in geometry (more than 15 deg in inc or 25 deg in PA)."""
    if sel_a.get("selected") != sel_b.get("selected"):
        return True
    a, b = sel_a.get("best"), sel_b.get("best")
    if a is None or b is None:
        return a is not b
    return bool(abs(a["derived"]["inc"] - b["derived"]["inc"]) > 15.0 or _circ180(a["derived"]["pa_mod180"], b["derived"]["pa_mod180"]) > 25.0)


def deprins_comparison(cfg, derived_):
    """Our theta_rim = 2 r0, inc and PA beside the De Prins image and PMOIRED values, as (ours - theirs) in units of their larger quoted sigma."""
    import postagb_paper as PP  # noqa: PLC0415

    key, out = cfg["key"], {}
    ours = dict(theta_rim=2 * derived_["r0"], i_rim=derived_["inc"], pa_rim=derived_["pa_mod180"])
    for name, tab in (("deprins_image", PP.IMAGE), ("pmoired", PP.PMOIRED)):
        t = tab.get(key)
        if not t or "theta_rim" not in t:
            continue
        row = {}
        for q in ("theta_rim", "i_rim", "pa_rim"):
            v, lo, hi = t[q]
            diff = _circ180(ours[q], v) * np.sign(ours[q] - v) if q == "pa_rim" else ours[q] - v
            row[q] = dict(theirs=v, sigma=max(lo, hi), ours=ours[q], diff=float(diff), n_sigma=float(diff / max(lo, hi)))
        out[name] = row
    return out


def gp_polar_block(cfg, sel, ladders, s_info):
    """The comparison with the stored gp-polar evidence (design section 5), per arm N1 and N2: Delta = stored ``log_evidence_scene`` less
    ``lnZ_scene_rule`` of the selected parametric model fitted under the same arm, quoted only if the stored ``error_scale`` equals s to 1e-3 and
    ``n_points`` equals N; otherwise the mismatch is printed. |Delta| < 20 is "not decisive"."""
    out = {}
    key = cfg.get("data_key", cfg["key"])
    files = {"N1": f"{key}_gp-polar_unbounded_image.json", "N2": f"{key}_gp-polar_nightw_unbounded_image.json"}
    for arm, name in files.items():
        stored = None
        for run in STORED_RUNS:
            f = STORED / run / key / name
            if f.exists():
                stored = json.load(open(f))
                break
        row = dict(file=name, stored_found=stored is not None, provisional=key == "PS_Gem")
        lad = ladders.get((sel, arm)) if sel else None
        b = (lad or {}).get("best")
        if stored is None or stored.get("log_evidence_scene") is None or b is None:
            out[arm] = row | dict(note="no stored ln Z or no parametric fit: chi2 only")
            continue
        row |= dict(gp_lnZ=stored["log_evidence_scene"], param_lnZ_scene_rule=b["lnZ_scene_rule"], gp_chi2_own=stored.get("chi2_per_point"),
                    param_chi2_own=b["own_chi2"], param_chi2_raw=b["raw_chi2"], gp_error_scale=stored.get("error_scale"), gp_n_points=stored.get("n_points"),
                    n_rows=b["n_rows"])
        s_ok = stored.get("error_scale") is not None and abs(stored["error_scale"] - s_info["s"]) < 1e-3
        n_ok = stored.get("n_points") == b["n_rows"]
        if s_ok and n_ok:
            d = stored["log_evidence_scene"] - b["lnZ_scene_rule"]
            row |= dict(delta=float(d), mismatch=False, decisive=bool(abs(d) >= 20.0))
        else:
            row |= dict(mismatch=True, note=f"mismatch: stored error_scale {stored.get('error_scale')} vs {s_info['s']}, n_points {stored.get('n_points')} vs {b['n_rows']}; no delta quoted")
        out[arm] = row
    return out


def jackknife_sd(values, centre, circular=False):
    """Leave-one-out jackknife standard deviation of ``values`` (one per left-out night): sqrt((n - 1) / n x sum (x_i - mean)^2); angles mod 180 are
    taken about ``centre``."""
    x = np.asarray(values, float)
    if circular:
        x = centre + (x - centre + 90.0) % 180.0 - 90.0
    n = x.size
    return float(np.sqrt((n - 1) / n * np.sum((x - x.mean()) ** 2))) if n > 1 else float("nan")


def night_subsets(files, load=None):
    """``{night: callable}``: each callable reads the data of ``files`` without that night (a night runs from MJD .65 to the next; a file belongs
    to the night of its mean MJD). The input of ``jackknife``."""
    load = load or (lambda fs: OIData(read_oifits(fs)))
    nights = {}
    for f in files:
        mjd = float(np.mean(np.asarray(OIData(read_oifits([f])).mjd, float)))
        nights.setdefault(int(np.floor(mjd - NIGHT_SPLIT_MJD)), []).append(f)
    return {n: (lambda n=n: load([f for m, fs in nights.items() if m != n for f in fs])) for n in nights}, {n: len(fs) for n, fs in nights.items()}


def jackknife(cfg, subsets, model_id, values, arm="N2", variant=None, s=None, time_limit=60.0, log=print):
    """Leave-one-night-out jackknife at the selected model: refit with each night removed, from the optimum ``values`` (path -> number), under ``arm``
    (the night terms rebuilt on the data that remain). ``subsets`` maps each night to the raw data without it (an ``OIData`` or a function returning
    one; see ``night_subsets``). Returns the per-night results and the jackknife sds of inc, PA, r0, sigma, c1, s1, f_env, f_bg."""
    if len(subsets) < 2:
        raise ValueError("jackknife needs at least two nights")
    st0 = start_from_values(values, model_id, cfg, name="optimum")
    runs = []
    for n in sorted(subsets):
        data_n = subsets[n]() if callable(subsets[n]) else subsets[n]
        d_arm = noise_arm(data_n, cfg, arm, s=s)
        scene = build_scene(cfg, model_id, st0, variant)
        res, rec = run_fit(scene, priors_for(cfg, scene, model_id, st0, variant), d_arm, time_limit=time_limit)
        row = dict(night_removed=int(n), **{k: rec[k] for k in ("converged", "stop", "seconds")})
        if res is not None:
            row |= dict(values=values_of(res), derived=derived(res.model, cfg))
        runs.append(row)
        log(f"  jackknife night {n}: converged {row['converged']}")
    ok = [r for r in runs if r.get("derived") and r["converged"]]
    pa0 = values["env.pa"] % 180.0
    pick = dict(inc=lambda r: r["derived"]["inc"], pa=lambda r: r["derived"]["pa_mod180"], r0=lambda r: r["derived"]["r0"], sigma=lambda r: r["derived"]["sigma"],
                f_env=lambda r: r["values"]["env.flux.ratio"], f_bg=lambda r: r["values"].get("bg.flux.ratio"))
    sds = {k: jackknife_sd([f(r) for r in ok], pa0, circular=k == "pa") if ok else None for k, f in pick.items()}
    withc = [r for r in ok if r["derived"]["c"]]
    for nm, key in (("c1", "c"), ("s1", "s")):
        sds[nm] = jackknife_sd([r["derived"][key][0] for r in withc], 0.0) if withc else None
    return dict(model=model_id, arm=arm, n_nights=len(subsets), n_ok=len(ok), runs=runs, sd=sds)


# ----------------------------------------------------------------------------
# stages


def _prepare(args):
    cfg = P.cfg_of(argparse.Namespace(target=args.target, files=None, smoke=False, noise="current", wide_field=False, bg_index=None, star_index=None))
    return cfg


def _read_ladders(out, key, arms=ARMS, models=CHAIN, variant=None):
    lad = {}
    for a in arms:
        for m in models:
            f = model_path(out, key, m, a, variant)
            if f.exists():
                lad[(m, a)] = json.load(open(f))
    return lad


def _done(out, stage):
    (Path(out) / f"{stage}.DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


def ladder_stage(args, out):
    cfg = _prepare(args)
    data, files = P.load(cfg)
    arms = tuple(args.arms.split(","))
    models = tuple(args.models.split(","))
    sel_file = out / f"{cfg['key']}_param_selected.json"
    if "M3" in models and sel_file.exists() and json.load(open(sel_file)).get("selected", {}).get(SELECTION_ARM) == "M2o":
        models = tuple("M3o" if m == "M3" else m for m in models)  # M3 means "the selected model plus m = 2"
    t0 = time.time()
    fit_ladder(cfg, data, arms, models, args.variant, smoke=args.smoke, do_is=not args.smoke, out=out, time_limit=args.time_limit,
               prior=_read_ladders(out, cfg["key"], arms, CHAIN, args.variant), log=lambda m: print(m, flush=True))
    _done(out, "ladder" if not args.variant else f"ladder_{args.variant}")
    print(f"ladder done in {time.time() - t0:.0f} s", flush=True)


def select_stage(args, out):
    cfg = _prepare(args)
    s, s_src, _ = stored_error_scale(cfg)
    lad = _read_ladders(out, cfg["key"], ARMS, CHAIN, None)
    if not lad:
        raise SystemExit(f"no ladder files in {out}")
    sels = {a: select(lad, a) for a in ARMS if any(k[1] == a for k in lad)}
    sel2 = sels.get(SELECTION_ARM)
    row = dict(target=cfg["key"], commits=commits(), error_scale=s, arms={}, selected={a: v["selected"] for a, v in sels.items()})
    for a, v in sels.items():
        row["arms"][a] = {k: v[k] for k in v if k != "best"} | dict(derived=(v.get("best") or {}).get("derived"), lnZ_use=(v.get("best") or {}).get("lnZ_use"))
    if "N1" in sels and sel2:
        row["arm_dependent"] = arm_dependent(sels["N1"], sel2)
    row["is_ring"] = bool(sel2 and sel2.get("is_ring"))
    row["flags"] = sel2.get("flags") if sel2 else None
    jf = out / f"{cfg['key']}_param_jackknife.json"
    row["jackknife"] = json.load(open(jf))["sd"] if jf.exists() else None
    if sel2 and sel2.get("best"):
        row["comparison_deprins"] = deprins_comparison(cfg, sel2["best"]["derived"])
        row["gp_polar"] = gp_polar_block(cfg, sel2["selected"], lad, dict(s=s))
    with open(out / f"{cfg['key']}_param_selected.json", "w") as f:
        json.dump(_jsonable(row), f, indent=1)
    selr = sel2 or sels.get("N1")  # residuals of the N2 selection (a smoke run with N1 only falls back to N1)
    if selr and selr.get("best") and not args.no_residuals:
        write_residuals(out, cfg, selr["selected"], selr["best"]["values"])
    _done(out, "select")
    for a, v in sels.items():
        print(a, "selected", v["selected"], [(c["child"], c["accepted"], c.get("d_lnZ"), c.get("d_bic")) for c in v["chain"]])


def write_residuals(out, cfg, model_id, values):
    """``<T>_param_residuals.npz``: raw-error residuals of the selected N2 model against baseline length (``v2_baseline_m, v2_resid`` in units of the
    quoted error; ``cp_baseline_m, cp_resid`` in degrees over the quoted error, baseline = the longest of the triangle's three)."""
    data, _ = P.load(cfg)
    m = build_scene(cfg, model_id, start_from_values(values, model_id, cfg))
    m = m.set_many(values) if hasattr(m, "set_many") else _set_values(m, values)
    v2, t3 = P.observables(m, data)
    sel = slice(None) if data.vis_index is None or np.ndim(data.vis_index) == 0 else np.asarray(data.vis_index)
    bl = np.hypot(np.asarray(data.u), np.asarray(data.v))
    b3 = np.max([bl[np.asarray(i, int)] for i in (data.i_cps1, data.i_cps2, data.i_cps3)], axis=0)
    dt = np.degrees(np.asarray(data.phi)) - t3
    np.savez(Path(out) / f"{cfg['key']}_param_residuals.npz", v2_baseline_m=bl[sel], v2_resid=(np.asarray(data.vis) - v2) / np.asarray(data.d_vis),
             cp_baseline_m=b3, cp_resid=(((dt + 180.0) % 360.0) - 180.0) / np.degrees(np.asarray(data.d_phi)))


def _set_values(m, values):
    for k, v in values.items():
        m = m.set(k, np.asarray(v, float))
    return m


def jackknife_stage(args, out):
    cfg = _prepare(args)
    f = out / f"{cfg['key']}_param_selected.json"
    if not f.exists():
        raise SystemExit("run the select stage first")
    sel = json.load(open(f))
    model_id = sel["selected"].get(SELECTION_ARM)
    best = (_read_ladders(out, cfg["key"], (SELECTION_ARM,), (model_id,), None).get((model_id, SELECTION_ARM)) or {}).get("best")
    if best is None:
        raise SystemExit(f"no eligible {model_id} fit under {SELECTION_ARM}")
    subsets, n_files = night_subsets(P.load_files(cfg))
    s, _, _ = stored_error_scale(cfg)
    r = jackknife(cfg, subsets, model_id, best["values"], SELECTION_ARM, s=s, time_limit=args.time_limit)
    with open(out / f"{cfg['key']}_param_jackknife.json", "w") as fh:
        json.dump(_jsonable(dict(target=cfg["key"], commits=commits(), files_per_night=n_files, **r)), fh, indent=1)
    _done(out, "jackknife")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["ladder", "jackknife", "select"])
    ap.add_argument("out", type=Path)
    ap.add_argument("--target", required=True, choices=sorted(MACKAY_S))
    ap.add_argument("--arms", default="N0,N1,N2")
    ap.add_argument("--models", default="M0,M1,M2,M2o")
    ap.add_argument("--variant", choices=["nosec", "secfree", "bgfree"], default=None)
    ap.add_argument("--smoke", action="store_true", help="four starts per fit and no importance sampling")
    ap.add_argument("--no-residuals", action="store_true", help="select: skip the residuals npz (it reads the data)")
    ap.add_argument("--time-limit", type=float, default=60.0, help="seconds per fit")
    args = ap.parse_args()
    out = args.out.expanduser() / args.target
    out.mkdir(parents=True, exist_ok=True)
    dict(ladder=ladder_stage, select=select_stage, jackknife=jackknife_stage)[args.stage](args, out)


if __name__ == "__main__":
    main()
