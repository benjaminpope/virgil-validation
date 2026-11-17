"""scripts/postagb_imaging.py: the binary test of design/gp_polar_hyperparameters.md §12, ``MovingComponent`` (a star moving
against the disc) and ``binary_scene`` / ``binary_summary`` (``--binary``)."""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
try:  # needs virgil main (MultipoleField), as tests/test_postagb_imaging.py
    import postagb_imaging as P  # noqa: E402
except ImportError:
    P = None
pytestmark = [
    pytest.mark.skipif(
        P is None, reason="virgil without MultipoleField (install the postagb extra)"
    ),
    # guards our own moving-star component against virgil's static PointSource / UniformDisk at the same per-sample positions;
    # no standard or external code is involved
    pytest.mark.validates(
        "postagb_imaging.binary", roots=["mathematics"], kind="guard"
    ),
]

L0 = 1.65e-6
MJD = np.repeat([100.1, 100.2, 101.1, 108.1], 4)  # four frames on three nights (the MJD .65 split)
N = MJD.size
U = np.linspace(20.0, 80.0, N)
V = np.linspace(-30.0, 40.0, N)
WL = np.tile([1.55e-6, 1.6e-6, 1.7e-6, 1.75e-6], 4)
DT = MJD - 0.5 * (MJD.min() + MJD.max())
GROUP = np.unique(np.floor(MJD - P.NIGHT_SPLIT_MJD if P else MJD), return_inverse=True)[1]


def scene(sec=True):
    parts = dict(
        star=P.UniformDisk(diam=0.4, flux=P.PowerLaw(1.0, -2.8, L0)),
        env=P.MultipoleDisk(
            P.GaussianRing(4.0, 1.0),
            inc=40.0,
            pa=30.0,
            n_r=48,
            flux=P.PowerLaw(0.5, 0.6, L0),
        ),
    )
    if sec:
        parts["sec"] = P.PointSource(flux=P.PowerLaw(0.1, 0.0, L0), dra=-1.0, ddec=-1.4)
    return P.System(**parts)


def test_a_still_moving_component_is_the_static_star():
    s = scene()
    moving = s.set(
        "sec", P.MovingComponent(s.sec, DT, GROUP, flux=s.sec.flux, dra=-1.0, ddec=-1.4)
    ).set("star", P.MovingComponent(s.star, DT, GROUP, flux=s.star.flux))
    np.testing.assert_allclose(
        np.asarray(moving.model(U, V, WL)), np.asarray(s.model(U, V, WL)), atol=1e-7
    )


def test_motion_and_night_offsets_place_the_star_per_sample():
    s = scene()
    m = P.MovingComponent(
        s.sec, DT, GROUP, flux=s.sec.flux, dra=-1.0, ddec=-1.4, vra=0.02, vdec=-0.01,
        off_ra=[0.0, 0.1, -0.2], off_dec=[0.05, 0.0, 0.1],
    )
    ra, dec = (np.asarray(x) for x in m.positions())
    np.testing.assert_allclose(ra, -1.0 + 0.02 * DT + np.array([0.0, 0.1, -0.2])[GROUP])
    np.testing.assert_allclose(dec, -1.4 - 0.01 * DT + np.array([0.05, 0.0, 0.1])[GROUP])
    vis = np.asarray(m.model(U, V, WL))
    for i in (0, 9, 15):  # each sample sees a static point source at its own position
        ref = P.PointSource(dra=float(ra[i]), ddec=float(dec[i])).model(U[i:i + 1], V[i:i + 1], WL[i:i + 1])
        np.testing.assert_allclose(vis[i], np.asarray(ref)[0], atol=1e-7)


def test_a_moving_component_refuses_another_sampling():
    m = P.MovingComponent(scene().sec, DT, GROUP)
    with pytest.raises(ValueError, match="only evaluates on the data"):
        m.model(U[:5], V[:5], WL[:5])


class _Data:  # what binary_scene reads from an OIData: times per sample and night labels
    t_ref, dt = 0.0, MJD


def test_binary_scene_frees_what_each_mode_says(monkeypatch):
    monkeypatch.setattr(P, "night_labels", lambda d: dict(u_night=np.floor(MJD - P.NIGHT_SPLIT_MJD).astype(int)))
    cfg = dict(sec0=(-1.0, -1.4), img_star_index=-2.8)
    s, pri = P.binary_scene(scene(), cfg, _Data, "paper")
    assert pri == {} and not isinstance(s.sec, P.MovingComponent)
    _, pri = P.binary_scene(scene(), cfg, _Data, "free")
    assert set(pri) == {"sec.dra", "sec.ddec", "sec.flux.ratio", "sec.flux.index"}
    s, pri = P.binary_scene(scene(), cfg, _Data, "motion")
    assert {"star.vra", "star.vdec", "sec.vra", "sec.vdec"} <= set(pri)
    assert isinstance(s.star, P.MovingComponent) and isinstance(s.sec, P.MovingComponent)
    for k in pri:  # every prior names a parameter of the scene
        s.set(k, 0.0)
    s, pri = P.binary_scene(scene(sec=False), dict(sec0=None), _Data, "epoch")
    assert set(pri) == {"star.off_ra", "star.off_dec"}  # no secondary: only the primary moves
    assert pri["star.off_ra"].batch_shape == (3,) and np.shape(s.star.off_ra) == (3,)


def test_binary_summary_reads_the_secondary_colour():
    cfg = dict(img_star_index=-2.8)
    s = scene()
    assert P.binary_summary(s.set("sec.flux.index", -2.5), "free", cfg)["bin_sec_colour"] == "photospheric"
    assert P.binary_summary(s.set("sec.flux.index", 0.3), "free", cfg)["bin_sec_colour"] == "dust-like"
    assert P.binary_summary(s.set("sec.flux.index", 1.9), "free", cfg)["bin_sec_colour"] == "neither"
