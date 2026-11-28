"""The PIONIER data layer and noise model of the post-AGB fits, on virgil-vlti: one home for ``postagb_imaging.py`` (gp-polar) and
``postagb_parametric.py`` (disc ladder).

Loading is ``virgil_vlti.to_oidata`` (the INSPIRING products are ESO ``pndrs``-calibrated PIONIER; the PIONIER profile applies no data
cut, and the rows and values equal virgil's own ``OIData(read_oifits(files))``: tests/test_postagb_vlti.py). The night terms, a gain on
log|V| per (night, baseline) and a closure-phase offset per (night, triangle), are ``virgil_vlti.with_night_terms``; this file keeps only
the widths of this project: the pair-test widths per target (``postagb_noise_widths.json``, the frame terms of ``--noise vlti``), the
night widths per target (``postagb_night_widths.json``, ``--noise night``) and the fixed night widths of the review of 18 Nov (``NIGHTW``,
the parametric N2 and gp-polar ``--noise nightw``).
"""

import json
import warnings
from pathlib import Path

import numpy as np
from virgil_vlti import to_oidata, with_night_terms
from virgil_vlti.noise import night_labels, night_modes as _night_modes  # noqa: F401  (night_labels is re-exported)
from virgil_vlti.products import NIGHT_BOUNDARY_UT_HOURS

NIGHT_SPLIT_MJD = NIGHT_BOUNDARY_UT_HOURS / 24.0  # a night runs from 12 UT (Exposure.night); the earlier MJD .65 split gives the same night for every file

# the PIONIER error model of virgil-vlti (design/pionier_systematics.md, measured on IW Car with the pair test: 11 disjoint pairs of
# consecutive exposures, model subtracted). V^2 errors are 0.32 white; the rest is a gain per (frame, telescope) and per (frame,
# baseline) shared by the six channels, widths on log|V| (half the 1.1 % and 0.9 % on V^2); closure-phase errors are 0.65 white, plus an
# offset per (frame, triangle) of 0.21 deg. Each target has its own widths in postagb_noise_widths.json (the same pair test on its own
# exposures; fewer than five pairs: the median of the others).
VLTI_NOISE = dict(v2_white=0.32, t3_white=0.65, gain_telescope=0.0055, gain_baseline=0.0045, t3_offset_deg=0.21)  # IW Car
NOISE_WIDTHS = json.load(open(Path(__file__).with_name("postagb_noise_widths.json")))["targets"]
# night terms: the widths measured per target on the residuals against the published model, before any night fit was looked at
# (scripts/postagb_night_widths.py, design/gp_polar_hyperparameters.md section 9)
_NIGHT_FILE = Path(__file__).with_name("postagb_night_widths.json")
NIGHT_WIDTHS = json.load(open(_NIGHT_FILE))["targets"] if _NIGHT_FILE.exists() else {}
NIGHT_KEYS = ("night_gain_baseline", "night_t3_offset_deg")
# fixed night widths (the 18 Nov review, design section 12): 3 % on log|V| per (night, baseline), 2 deg per (night, triangle)
NIGHTW = dict(night_gain_baseline=0.03, night_t3_offset_deg=2.0)
NO_FRAME = dict(gain_telescope=0.0, gain_baseline=0.0, t3_offset_deg=0.0)


def noise_widths(cfg):
    """The vlti error-model widths of this target (``VLTI_NOISE`` for a target the table does not have)."""
    row = NOISE_WIDTHS.get(cfg.get("data_key", cfg["key"]))
    return VLTI_NOISE if row is None else {k: row[k] for k in VLTI_NOISE}


def night_widths(cfg):
    """This target's night widths (``NIGHT_KEYS``); a target the table does not have gets the median of the others."""
    row = NIGHT_WIDTHS.get(cfg.get("data_key", cfg["key"]))
    if row is None:
        return {k: float(np.median([r[k] for r in NIGHT_WIDTHS.values()])) for k in NIGHT_KEYS}
    return {k: row[k] for k in NIGHT_KEYS}


def load_pionier(files, bandwidth=False):
    """The calibrated PIONIER files as a virgil ``OIData`` through ``virgil_vlti.to_oidata`` (no nuisance modes: the noise model is
    applied by ``pionier_noise``). ``bandwidth`` models bandwidth smearing (``read_oifits(bandwidth=True)``)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="no nuisance modes beside")
        out = to_oidata(files, bandwidth=bandwidth, use_template=False)  # no sidecar modes: the noise model adds them
    assert out.gains is None and out.phase_offsets is None, "sidecar nuisance modes found beside the files"
    return out


def pionier_noise(data, white=None, frame=None, night=None):
    """``data`` under the PIONIER error model: the errors scaled first (``white``: a factor, or ``{"vis": ..., "phi": ...}``), then the
    per-frame gains and closure offsets of ``frame`` (``gain_telescope``, ``gain_baseline``, ``t3_offset_deg``) and the night terms of
    ``night`` (``night_gain_baseline``, ``night_t3_offset_deg``), which the likelihood marginalizes analytically. A zero width leaves its
    term out; ``None`` for ``frame`` or ``night`` leaves out all of its terms."""
    if white is not None:
        data = data.with_error_scale(white)
    f, n = frame or NO_FRAME, night or {}
    return with_night_terms(
        data, night_gain=n.get("night_gain_baseline", 0.0), night_t3_deg=n.get("night_t3_offset_deg", 0.0),
        telescope=f["gain_telescope"], baseline=f["gain_baseline"], triangle_deg=f["t3_offset_deg"])


def vlti_noise(data, widths=VLTI_NOISE):
    """``data`` under the virgil-vlti PIONIER error model (``widths``, see ``noise_widths``): per-observable white scaling (``vis`` is
    the V^2 here, ``phi`` the closure phases), then per-frame gains and closure offsets."""
    return pionier_noise(data, {"vis": widths["v2_white"], "phi": widths["t3_white"]}, widths)


def night_noise(data, widths=VLTI_NOISE, night=NIGHTW):
    """``data`` under ``vlti_noise`` plus the night terms of ``night`` (``NIGHT_KEYS``)."""
    return pionier_noise(data, {"vis": widths["v2_white"], "phi": widths["t3_white"]}, widths, night)


def night_modes(data, gain=NIGHTW["night_gain_baseline"], t3_deg=NIGHTW["night_t3_offset_deg"]):
    """The night modes ``(g, o)`` of ``virgil_vlti.noise.night_modes``, by default at the fixed ``NIGHTW`` widths."""
    return _night_modes(data, gain, t3_deg)


# ----------------------------------------------------------------------------
# simulation (WP-02): one hook for every injection, gp-polar and composites

SIM_NOISES = ("white", "night", "vlti", "nightw")


def white_pair(cfg, white):
    """``white`` as the ``{"vis", "phi"}`` scales of ``pionier_noise``: a ``{"vis", "phi"}`` dict, a ``(V2, T3)`` pair, the contents of a
    ``<target>_white.json`` (``v2_white``, ``t3_white``), or None (the pair-test scales of the target)."""
    if white is None:
        w = noise_widths(cfg)
        return {"vis": w["v2_white"], "phi": w["t3_white"]}
    if isinstance(white, dict):
        return {"vis": white["v2_white"], "phi": white["t3_white"]} if "v2_white" in white else dict(white)
    return {"vis": float(white[0]), "phi": float(white[1])}


class SimResult:
    """``data``: the quoted errors with V^2 and closure phases replaced by the draw; ``fit_data``: the same draw under the error model it
    was drawn from (what the fit reads); ``truth_chi2``: whitened chi^2 of the truth on ``fit_data``, ``{v2, t3, total, n_independent}``;
    ``meta``: noise, seed, scale_drawn, white and the widths."""

    def __init__(self, data, fit_data, truth_chi2, meta):
        self.data, self.fit_data, self.truth_chi2, self.meta = data, fit_data, truth_chi2, meta


def simulate(cfg, truth, *, noise="white", seed=0, scale=None, white=None, files=None, data0=None, key=None, night=None):
    """Simulated PIONIER data from ``truth`` (any virgil model) on a target's (u, v), errors and closure triangles, with the noise of ``noise``:

    * ``white``: ``load_pionier(files).with_error_scale(scale).with_model(truth, key)``; ``scale`` is the REAL target's MacKay factor (1 if None).
    * ``night``: ``pionier_noise(data0, white=<white pair>, frame=noise_widths(cfg), night=night_widths(cfg)).with_model(truth, key)``; closure noise
      is drawn with virgil's triangle covariance (``cp_noise``), never independent per triangle.
    * ``vlti``: ``night`` without the night terms; ``nightw``: the MacKay scale plus the fixed night terms ``NIGHTW`` (arm N2 of the parametric ladder).

    ``seed`` is ``jax.random.PRNGKey(seed)`` unless ``key`` is given. ``data0`` (instead of ``files``) is the unmodified data, for fixtures;
    ``night`` replaces the published night widths (``<T>_white.json``'s ``night_widths_used``).
    The hand-drawn ``common`` noise of the plan (mis-specification tests only) is not built."""
    import jax  # noqa: PLC0415
    from virgil.imaging import whitened_residuals  # noqa: PLC0415

    if noise not in SIM_NOISES:
        raise ValueError(f"noise must be one of {SIM_NOISES}, got {noise!r}")
    data0 = load_pionier(files) if data0 is None else data0
    key = jax.random.PRNGKey(int(seed)) if key is None else key
    s = 1.0 if scale is None else float(scale)
    widths, drawn_white = {}, None
    if noise == "white":
        arm = data0.with_error_scale(s) if s != 1.0 else data0
    elif noise == "nightw":
        arm = pionier_noise(data0, white=s, night=NIGHTW)
        widths = dict(NIGHTW)
    else:
        drawn_white = white_pair(cfg, white)
        fw = noise_widths(cfg)
        nw = (night or night_widths(cfg)) if noise == "night" else None
        arm = pionier_noise(data0, drawn_white, fw, nw)
        widths = fw | (nw or {})
    fit_data = arm.with_model(truth, key=key)
    data = data0.set(["vis", "phi"], [fit_data.vis, fit_data.phi])
    r = np.asarray(whitened_residuals(truth, fit_data))
    n_vis = int(np.asarray(fit_data.vis).size)
    chi = dict(v2=float(np.sum(r[:n_vis] ** 2)), t3=float(np.sum(r[n_vis:] ** 2)), total=float(np.sum(r**2)), n_independent=int(fit_data.n_independent))
    rq = np.asarray(whitened_residuals(truth, data))  # the truth on the QUOTED errors: its sqrt(chi2 / n_independent) is the MacKay factor the fit should find
    meta = dict(noise=noise, seed=int(seed), scale_drawn=s if noise in ("white", "nightw") else None, white=drawn_white, widths=widths,
                scale_fitted=float(np.sqrt(np.sum(rq**2) / int(data.n_independent))))
    return SimResult(data, fit_data, chi, meta)


# ----------------------------------------------------------------------------
# measured white scales and night widths (WP-04): arm (b) of the rerun

WHITE_MIN_CHANNELS = 4  # a (frame, baseline) group with fewer channels than this cannot give a mean, a slope and a scatter: skipped, counted
NIGHT_WIDTH_REPLACE = 0.30  # arm (b) takes the pilot night widths where they differ from the published ones by more than this fraction


def between_cell_width(values, frames, cells):
    """τ of a two-level ANOVA: ``values`` per sample, averaged first per ``frames`` group, then compared across ``cells`` (frames are the
    replicates of a cell). Per-frame terms and white noise sit in the within-cell variance and are not counted again; a negative τ² gives 0.
    Returns ``(τ, number of cells with at least two frames)``."""
    keys = np.unique(np.stack([cells, frames], 1), axis=0)
    frame_mean = np.array([values[(cells == c) & (frames == f)].mean() for c, f in keys])
    means, within = [], []
    for c in np.unique(keys[:, 0]):
        x = frame_mean[keys[:, 0] == c]
        if x.size >= 2:
            means.append(x.mean())
            within.append(x.var(ddof=1) / x.size)
    if len(means) < 2:
        return 0.0, len(means)
    tau2 = np.var(means, ddof=1) - np.mean(within)
    return float(np.sqrt(max(tau2, 0.0))), len(means)


def channel_groups(frame, key, wavel):
    """Index arrays of the samples of each (frame, key) group in wavelength order. ``key``: one integer label per sample (a baseline or a
    triangle)."""
    order = np.lexsort((np.asarray(wavel), np.asarray(key), np.asarray(frame)))
    f, k = np.asarray(frame)[order], np.asarray(key)[order]
    cut = np.flatnonzero((f[1:] != f[:-1]) | (k[1:] != k[:-1])) + 1
    return np.split(order, cut)


def channel_scatter(resid, groups, min_channels=WHITE_MIN_CHANNELS):
    """Pooled rms of ``resid`` (in units of the quoted error) about a straight line in channel number, fitted per group: the mean and the
    slope of a group carry its gain, its offset and the smooth misfit of the pilot, so only the channel-to-channel scatter is left. The
    degrees of freedom of the two fitted terms are taken out (n - 2 per group). Returns ``(rms, n_groups_used, n_groups_skipped)``."""
    ss, dof, used, skipped = 0.0, 0, 0, 0
    for g in groups:
        if len(g) < min_channels:
            skipped += 1
            continue
        x = np.arange(len(g), dtype=float)
        y = np.asarray(resid, float)[g]
        y = y - np.polyval(np.polyfit(x, y, 1), x)
        ss += float(y @ y)
        dof += len(g) - 2
        used += 1
    return (float(np.sqrt(ss / dof)) if dof else float("nan")), used, skipped


def white_scales(data, res_v2, res_t3, min_channels=WHITE_MIN_CHANNELS):
    """Arm (b)'s white scales: ``res_v2`` and ``res_t3`` are the pilot's residuals per V² and per closure phase in units of the quoted errors
    (the closure phase wrapped). V² groups are (frame, baseline), closure groups (frame, triangle). Returns
    ``dict(v2_white, t3_white, n_groups_v2, n_groups_t3, skipped_v2, skipped_t3)``."""
    st = np.sort(np.asarray(data.stations, int), axis=1)
    frame, wavel = np.asarray(data.frame, int), np.asarray(data.wavel, float)
    vi = data.vis_index
    vi = np.arange(frame.size) if vi is None or np.ndim(vi) == 0 else np.asarray(vi)
    base = np.unique(st, axis=0, return_inverse=True)[1].ravel()
    i1, i2, i3 = (np.asarray(i, int) for i in (data.i_cps1, data.i_cps2, data.i_cps3))
    tri = np.unique(np.sort(np.concatenate([st[i1], st[i2], st[i3]], 1), axis=1), axis=0, return_inverse=True)[1].ravel()
    v2, nv, sv = channel_scatter(res_v2, channel_groups(frame[vi], base[vi], wavel[vi]), min_channels)
    t3, nt, stt = channel_scatter(res_t3, channel_groups(frame[i1], tri, wavel[i1]), min_channels)
    return dict(v2_white=v2, t3_white=t3, n_groups_v2=nv, n_groups_t3=nt, skipped_v2=sv, skipped_t3=stt)


def night_widths_about(lab, rel_v2, dphi_deg):
    """Night widths re-measured about the pilot: a gain on log|V| per (night, baseline), half the between-night width of the relative V²
    residuals, and an offset (deg) per (night, triangle). ``lab`` is ``night_labels(data)``; ``rel_v2`` is (V² - V²_pilot) / V²_pilot per V²
    and ``dphi_deg`` the wrapped closure-phase residual in degrees. Same method as ``scripts/postagb_night_widths.py``."""
    tau_v2, n_v = between_cell_width(np.asarray(rel_v2, float), lab["v2_frame"], lab["v2_cell"])
    tau_t3, n_t = between_cell_width(np.asarray(dphi_deg, float), lab["t3_frame"], lab["t3_cell"])
    return dict(night_gain_baseline=0.5 * tau_v2, night_t3_offset_deg=tau_t3, n_cells_v2=n_v, n_cells_t3=n_t)


def night_widths_used(published, pilot, tol=NIGHT_WIDTH_REPLACE):
    """Arm (b)'s night widths (decision D4): the pilot's where they differ from the published ones by more than ``tol`` (a fraction of the
    published width; a zero published width is always replaced), else the published ones."""
    out = {}
    for k in NIGHT_KEYS:
        p, q = float(published[k]), float(pilot[k])
        out[k] = q if (p == 0.0 or abs(q - p) / p > tol) else p
    return out
