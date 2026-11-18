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
    pytest.mark.skipif(P is None, reason="needs virgil main and virgil-vlti with with_night_terms (install the postagb extra)"),
    # guards our own bookkeeping: that the data layer returns the rows virgil reads and that both pipelines share one noise model
    pytest.mark.validates("evidence", roots=["statistics"], kind="guard"),
]
DATA = pathlib.Path(os.environ.get("POSTAGB_DATA", "~/data/postagb_pionier")).expanduser()
AI_SCO = sorted(glob.glob(str(DATA / "AI_Sco" / "INSPIRING" / "*.fits")))
needs_data = pytest.mark.skipif(not AI_SCO, reason="AI Sco INSPIRING files not found")


def gram(m):
    m = np.zeros((0, 1)) if m is None else np.asarray(m)
    return m @ m.T if m.size else m


@needs_data
def test_vlti_loader_reproduces_the_row_counts_and_values_of_virgils_reader_on_ai_sco():
    from virgil import OIData, read_oifits

    ref, new = OIData(read_oifits(AI_SCO)), N.load_pionier(AI_SCO)
    assert np.size(new.vis) == np.size(ref.vis) == 1008  # V^2 before whitening
    assert np.size(new.phi) == np.size(ref.phi)
    assert np.size(new.phi) == 672  # closure phases; 504 independent after whitening
    for a in ("vis", "d_vis", "phi", "d_phi", "u", "v", "wavel"):
        np.testing.assert_array_equal(np.asarray(getattr(new, a)), np.asarray(getattr(ref, a)))
    assert new.gains is None and new.phase_offsets is None  # the profile adds no nuisance modes, cuts or floors of its own


@needs_data
def test_vlti_loader_loads_every_post_agb_target_as_virgil_reads_it():
    from virgil import OIData, read_oifits

    for d in sorted(DATA.glob("*/INSPIRING")):
        files = sorted(glob.glob(str(d / "*.fits")))
        ref, new = OIData(read_oifits(files)), N.load_pionier(files)
        assert np.size(new.vis) == np.size(ref.vis) and np.size(new.phi) == np.size(ref.phi), d.parent.name
        np.testing.assert_array_equal(np.asarray(new.vis), np.asarray(ref.vis))


@needs_data
def test_night_of_every_file_is_the_same_under_virgil_vltis_noon_boundary_and_the_earlier_mjd_65_split():
    for f in AI_SCO:
        mjd = float(np.mean(np.asarray(N.load_pionier([f]).mjd, float)))
        assert int(np.floor(mjd - N.NIGHT_SPLIT_MJD)) == int(np.floor(mjd - 0.65))
        assert virgil_vlti.read_exposure(f).night  # the loader's own night label exists


@needs_data
def test_shared_night_terms_equal_the_earlier_local_modes_on_ai_sco():
    """The earlier per-script construction (one mode per (night, baseline) on every u sample, split at MJD .65), kept here as the reference."""
    data = N.load_pionier(AI_SCO)
    mjd = float(data.t_ref) + np.asarray(data.dt, float)
    night = np.floor(mjd - 0.65).astype(int)
    st = np.sort(np.asarray(data.stations, int), axis=1)
    base = np.unique(st, axis=0, return_inverse=True)[1].ravel()
    cell_u = np.unique(np.stack([night, base], 1), axis=0, return_inverse=True)[1].ravel()
    i1, i2, i3 = (np.asarray(i, int) for i in (data.i_cps1, data.i_cps2, data.i_cps3))
    tri = np.unique(np.sort(np.concatenate([st[i1], st[i2], st[i3]], 1), axis=1), axis=0, return_inverse=True)[1].ravel()
    cell_t = np.unique(np.stack([night[i1], tri], 1), axis=0, return_inverse=True)[1].ravel()
    g = np.zeros((cell_u.max() + 1, cell_u.size))
    g[cell_u, np.arange(cell_u.size)] = 0.03
    o = np.zeros((cell_t.max() + 1, cell_t.size))
    o[cell_t, np.arange(cell_t.size)] = np.radians(2.0)
    gn, on = N.night_modes(data)
    np.testing.assert_allclose(gram(gn.T), gram(g.T))
    np.testing.assert_allclose(gram(on.T), gram(o.T))
    # the same likelihood, whatever virgil does with a mode that lies in one frame (a per-frame group) or spans several (spanning)
    from virgil.likelihood import loglike
    from virgil.models import GaussianDisk

    old = data.with_error_scale({"vis": 1.3, "phi": 1.3}).with_gains(telescope=0.0, baseline=0.0, modes=g).with_closure_offsets(
        triangle=0.0, modes=o)
    new = N.pionier_noise(data, white=1.3, night=N.NIGHTW)
    for sigma in (0.8, 1.3, 2.0):
        a = float(loglike([sigma], ["sigma"], GaussianDisk(sigma=sigma), old))
        b = float(loglike([sigma], ["sigma"], GaussianDisk(sigma=sigma), new))
        assert b == pytest.approx(a, rel=1e-10)


def test_both_pipelines_use_the_same_noise_code():
    for name in ("night_labels", "night_widths", "noise_widths", "pionier_noise", "vlti_noise", "night_noise"):
        assert getattr(P, name) is getattr(N, name)
    for name in ("night_labels", "pionier_noise", "night_modes", "load_pionier", "NIGHTW"):
        assert getattr(Q, name) is getattr(N, name)
    assert P.NIGHTW == N.NIGHTW == Q.NIGHTW


def test_both_pipelines_build_identical_noise_modes_for_the_nightw_arm():
    """gp-polar ``--noise nightw`` (``load_scaled``) and the parametric N2 apply the same call to the same data."""
    from virgil.likelihood import whitened_residuals

    data = make_data(None, nights=(0, 1, 2), frames=3)
    try:
        data.with_closure_offsets(modes=np.ones((1, np.size(data.phi))))
    except ValueError as e:  # virgil before closure modes could span frames
        pytest.skip(f"this virgil has no frame-spanning closure modes: {e}")
    s = 1.25
    q = Q.noise_arm(data, CFG, "N2", s=s)
    p = P.pionier_noise(data, white=s, night=P.NIGHTW)
    np.testing.assert_allclose(np.asarray(q.d_vis), np.asarray(p.d_vis))
    np.testing.assert_allclose(gram(np.asarray(q.gains.spanning)), gram(np.asarray(p.gains.spanning)))
    np.testing.assert_allclose(gram(np.asarray(q.phase_offsets.spanning)), gram(np.asarray(p.phase_offsets.spanning)))
    # the gp-polar `night_noise` with the MacKay white factor and no frame terms is the same model again
    white = dict(v2_white=s, t3_white=s, gain_telescope=0.0, gain_baseline=0.0, t3_offset_deg=0.0)
    r = P.night_noise(data, white, P.NIGHTW)
    m = Q.build_scene(CFG, "M2", __import__("_postagb_parametric_synth").start(inc=30.0, f_env=0.3))
    r1, r2 = np.asarray(whitened_residuals(m, q)), np.asarray(whitened_residuals(m, r))
    assert r1 @ r1 == pytest.approx(r2 @ r2, rel=1e-10)


def test_per_target_widths_are_available_and_n2_defaults_to_the_pre_registered_global_widths():
    w = N.night_widths(dict(key="AI_Sco"))
    assert set(w) == set(N.NIGHT_KEYS) and w != N.NIGHTW  # measured per target, not the global 3 % / 2 deg
    data = make_data(None, nights=(0, 1), frames=2)
    default = Q.noise_arm(data, CFG, "N2", s=1.0)
    own = Q.noise_arm(data, CFG, "N2", s=1.0, night=w)
    assert not np.allclose(np.asarray(default.gains.spanning), np.asarray(own.gains.spanning))
