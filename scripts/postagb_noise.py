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
