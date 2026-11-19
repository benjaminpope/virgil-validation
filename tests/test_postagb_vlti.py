"""The post-AGB PIONIER data layer and noise model on virgil-vlti (scripts/postagb_noise.py): the loader reproduces virgil's own rows, and
the gp-polar and parametric pipelines build their night terms from the one shared code. The AI Sco tests need the INSPIRING files
(POSTAGB_DATA, default ~/data/postagb_pionier) and skip without them."""

import glob
import os
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
try:
    import virgil_vlti
    import postagb_imaging as P  # noqa: E402
    import postagb_noise as N  # noqa: E402
    import postagb_parametric as Q  # noqa: E402
    from _postagb_parametric_synth import CFG, make_data  # noqa: E402
except ImportError:
    P = None
pytestmark = [
    pytest.mark.skipif(P is None, reason="needs virgil main (MultipoleField) and virgil-vlti with virgil_vlti.with_night_terms (install the postagb extra)"),
    # compares virgil-vlti with virgil's own reader and with a frozen copy of the earlier local construction: self-consistency only
    pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard"),
]
DATA = pathlib.Path(os.environ.get("POSTAGB_DATA", "~/data/postagb_pionier")).expanduser()
AI_SCO = sorted(glob.glob(str(DATA / "AI_Sco" / "INSPIRING" / "*.fits")))
needs_data = pytest.mark.skipif(not AI_SCO, reason="AI Sco INSPIRING files not found")


def gram(m):
    m = np.zeros((0, 1)) if m is None else np.asarray(m)
    return m @ m.T if m.size else m


def old_night_modes(data, gain, t3_deg):
    """Frozen copy of the earlier per-script construction: one mode per (night, baseline) on every u sample, one per (night, triangle) on
    the closure phases, night = floor(MJD - 0.65). Returns ``(g, o)`` (``None`` for a zero width)."""
    mjd = float(data.t_ref) + np.asarray(data.dt, float)
    night = np.floor(mjd - 0.65).astype(int)
    st = np.sort(np.asarray(data.stations, int), axis=1)
    base = np.unique(st, axis=0, return_inverse=True)[1].ravel()
    cell_u = np.unique(np.stack([night, base], 1), axis=0, return_inverse=True)[1].ravel()
    i1, i2, i3 = (np.asarray(i, int) for i in (data.i_cps1, data.i_cps2, data.i_cps3))
    tri = np.unique(np.sort(np.concatenate([st[i1], st[i2], st[i3]], 1), axis=1), axis=0, return_inverse=True)[1].ravel()
    cell_t = np.unique(np.stack([night[i1], tri], 1), axis=0, return_inverse=True)[1].ravel()
    g = o = None
    if gain > 0:
        g = np.zeros((cell_u.max() + 1, cell_u.size))
        g[cell_u, np.arange(cell_u.size)] = gain
    if t3_deg > 0:
        o = np.zeros((cell_t.max() + 1, cell_t.size))
        o[cell_t, np.arange(cell_t.size)] = np.radians(t3_deg)
    return g, o


def old_noise(data, white, frame, night):
    """Frozen copy of the earlier ``vlti_noise`` / ``night_noise`` / N2 calls: scale, then ``with_gains`` and ``with_closure_offsets`` with
    the per-frame widths and the night modes in one call each (``frame`` as ``VLTI_NOISE``, ``night`` as ``NIGHTW``; ``None`` for none)."""
    g, o = old_night_modes(data, *(night["night_gain_baseline"], night["night_t3_offset_deg"]) if night else (0.0, 0.0))
    d = data.with_error_scale(white)
    f = frame or dict(gain_telescope=0.0, gain_baseline=0.0, t3_offset_deg=0.0)
    return d.with_gains(telescope=f["gain_telescope"], baseline=f["gain_baseline"], modes=g).with_closure_offsets(
        triangle=np.radians(f["t3_offset_deg"]), modes=o)


def assert_same_likelihood(old, new):
    from virgil.likelihood import loglike
    from virgil.models import GaussianDisk

    np.testing.assert_allclose(np.asarray(new.d_vis), np.asarray(old.d_vis))
    np.testing.assert_allclose(np.asarray(new.d_phi), np.asarray(old.d_phi))
    for sigma in (0.8, 1.3, 2.0):
        a = float(loglike([sigma], ["sigma"], GaussianDisk(sigma=sigma), old))
        b = float(loglike([sigma], ["sigma"], GaussianDisk(sigma=sigma), new))
        assert b == pytest.approx(a, rel=1e-10)


def spanning_closure_ok(data):
    try:
        data.with_closure_offsets(modes=np.ones((1, np.size(data.phi))))
    except ValueError as e:  # virgil before closure modes could span frames
        pytest.skip(f"this virgil has no frame-spanning closure modes: {e}")


LOAD_ATTRS = ("vis", "d_vis", "phi", "d_phi", "u", "v", "wavel", "mjd", "stations", "frame", "i_cps1", "i_cps2", "i_cps3", "vis_flag", "phi_flag")


def same_attr(a, b, name):
    x, y = getattr(a, name, None), getattr(b, name, None)
    if x is None or y is None:
        assert x is None and y is None, name
        return
    np.testing.assert_array_equal(np.asarray(x), np.asarray(y), err_msg=name)


@needs_data
def test_vlti_loader_reproduces_the_row_counts_and_values_of_virgils_reader_on_ai_sco():
    from virgil import OIData, read_oifits

    ref, new = OIData(read_oifits(AI_SCO)), N.load_pionier(AI_SCO)
    assert np.size(new.vis) == np.size(ref.vis) == 1008  # V^2 before whitening
    assert np.size(new.phi) == np.size(ref.phi)
    assert np.size(new.phi) == 672  # closure phases; 504 independent after whitening
    for a in LOAD_ATTRS:
        same_attr(new, ref, a)
    assert new.gains is None and new.phase_offsets is None  # the profile adds no nuisance modes, cuts or floors of its own


@needs_data
def test_vlti_loader_reads_every_post_agb_target_as_virgil_does_and_nights_agree_with_the_mjd_65_split():
    from virgil import OIData, read_oifits

    dirs = sorted(DATA.glob("*/INSPIRING"))
    assert len(dirs) >= 8
    for d in dirs:
        files = sorted(glob.glob(str(d / "*.fits")))
        ref, new = OIData(read_oifits(files)), N.load_pionier(files)
        assert np.size(new.vis) == np.size(ref.vis) and np.size(new.phi) == np.size(ref.phi), d.parent.name
        for a in LOAD_ATTRS:
            same_attr(new, ref, a)
        # the night that with_night_terms assigns to each sample, against the earlier floor(MJD - 0.65): the same partition
        mine = np.asarray(N.night_labels(new)["u_night"])
        old = np.floor(float(new.t_ref) + np.asarray(new.dt, float) - 0.65).astype(int)
        assert len(set(zip(mine.tolist(), old.tolist()))) == len(set(mine.tolist())) == len(set(old.tolist())), d.parent.name


@needs_data
def test_shared_night_terms_equal_the_earlier_local_modes_on_ai_sco():
    data = N.load_pionier(AI_SCO)
    gn, on = N.night_modes(data)
    g, o = old_night_modes(data, 0.03, 2.0)
    np.testing.assert_allclose(gram(gn.T), gram(g.T))
    np.testing.assert_allclose(gram(on.T), gram(o.T))
    assert_same_likelihood(old_noise(data, {"vis": 1.3, "phi": 1.3}, None, N.NIGHTW), N.pionier_noise(data, white=1.3, night=N.NIGHTW))


@needs_data
def test_frame_terms_and_per_target_night_widths_equal_the_earlier_explicit_calls_on_ai_sco():
    data = N.load_pionier(AI_SCO)
    cfg = dict(key="AI_Sco")
    w, nw = N.noise_widths(cfg), N.night_widths(cfg)
    white = {"vis": w["v2_white"], "phi": w["t3_white"]}
    assert_same_likelihood(old_noise(data, white, w, None), N.vlti_noise(data, w))
    assert_same_likelihood(old_noise(data, white, w, nw), N.night_noise(data, w, nw))


def test_both_pipelines_use_the_same_noise_code():
    for name in ("night_labels", "night_widths", "noise_widths", "pionier_noise", "vlti_noise", "night_noise"):
        assert getattr(P, name) is getattr(N, name)
    for name in ("night_labels", "pionier_noise", "night_modes", "load_pionier", "NIGHTW"):
        assert getattr(Q, name) is getattr(N, name)
    assert P.NIGHTW == N.NIGHTW == Q.NIGHTW


def test_shared_noise_equals_the_earlier_local_constructions_on_synthetic_data():
    """Synthetic data (no files): the parametric N2 and the gp-polar ``vlti`` / ``night`` / ``nightw`` models, each against the frozen
    earlier explicit calls, with non-zero frame, telescope, baseline and triangle widths and non-zero night widths."""
    data = make_data(None, nights=(0, 1, 2), frames=3)
    spanning_closure_ok(data)
    s = 1.25
    nightw = {"vis": s, "phi": s}
    assert_same_likelihood(old_noise(data, nightw, None, P.NIGHTW), Q.noise_arm(data, CFG, "N2", s=s))  # parametric N2
    assert_same_likelihood(old_noise(data, nightw, None, P.NIGHTW), P.pionier_noise(data, white=s, night=P.NIGHTW))  # gp-polar nightw
    w = dict(P.VLTI_NOISE)
    nw = dict(night_gain_baseline=0.02, night_t3_offset_deg=1.0)
    white = {"vis": w["v2_white"], "phi": w["t3_white"]}
    assert_same_likelihood(old_noise(data, white, w, None), P.vlti_noise(data, w))  # gp-polar vlti
    assert_same_likelihood(old_noise(data, white, w, nw), P.night_noise(data, w, nw))  # gp-polar night
    g, o = N.night_modes(data)
    g0, o0 = old_night_modes(data, 0.03, 2.0)
    np.testing.assert_allclose(gram(g.T), gram(g0.T))
    np.testing.assert_allclose(gram(o.T), gram(o0.T))


def test_per_target_widths_are_available_and_n2_defaults_to_the_pre_registered_global_widths():
    w = N.night_widths(dict(key="AI_Sco"))
    assert set(w) == set(N.NIGHT_KEYS) and w != N.NIGHTW  # measured per target, not the global 3 % / 2 deg
    data = make_data(None, nights=(0, 1), frames=2)
    default = Q.noise_arm(data, CFG, "N2", s=1.0)
    own = Q.noise_arm(data, CFG, "N2", s=1.0, night=w)
    assert not np.allclose(np.asarray(default.gains.spanning), np.asarray(own.gains.spanning))
