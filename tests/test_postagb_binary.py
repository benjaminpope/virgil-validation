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
    pytest.mark.validates("evidence", roots=["mathematics"], kind="guard"
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
    cfg = dict(img_star_index=-2.1)  # AI Sco's SIND1: the windows are absolute, not about the primary's index
    s, sd = scene(), {"sec.flux.index": 0.2}
    read = lambda d, sd_=sd: P.binary_summary(s.set("sec.flux.index", d), "free", cfg, sd_)["bin_sec_colour"]  # noqa: E731
    assert read(-3.3) == "photospheric" and read(-2.5) == "photospheric"
    assert read(0.3) == "dust-like" and read(-1.2) == "neither" and read(1.9) == "neither"
    assert read(-3.0, {"sec.flux.index": 0.8}) == "unconstrained" and read(-3.0, None) == "unconstrained"
    out = P.binary_summary(s, "free", cfg, sd)
    assert out["bin_sec_index_sd"] == 0.2


def test_binary_summary_reports_the_relative_velocity():
    s = scene()
    m = s.set("sec", P.MovingComponent(s.sec, DT, GROUP, flux=s.sec.flux, dra=-1.0, ddec=-1.4, vra=0.01, vdec=-0.004)).set(
        "star", P.MovingComponent(s.star, DT, GROUP, flux=s.star.flux, vra=-0.002, vdec=0.001))
    out = P.binary_summary(m, "motion", dict(img_star_index=-2.8), {"sec.vra-star.vra": 3e-4, "sec.vdec-star.vdec": 4e-4, "sec.flux.index": 0.2})
    np.testing.assert_allclose(out["bin_rel_v_mas_per_day"], [0.012, -0.005])
    assert out["bin_rel_v_sd"] == [3e-4, 4e-4] and np.shape(out["bin_sec_offsets_sd"]) == (3, 2)


def test_scene_evidence_integrates_the_scene_parameters_with_their_prior_precision():
    """Flat priors: virgil's log_evidence(priors=) and scene_evidence agree. A Normal prior of width s on one scene parameter
    lowers scene_evidence by ln(1 + sd_flat^2 / s^2) / 2 (the rank-one update of the Hessian), which log_evidence leaves out."""
    import jax
    import numpyro.distributions as dist
    from virgil.coverage import ami_grid_record
    from virgil.fields import GaussianField
    from virgil.fitting import fit
    from virgil.oidata import OIData

    scene = P.System(star=P.PointSource(dra=0.5), env=P.Image(GaussianField(np.zeros((8, 8)), 1.5, 24.0), 12.0, flux=0.3))
    data = OIData(ami_grid_record(pitch_m=0.5)).with_model(scene, key=jax.random.PRNGKey(9))
    flat = P.image_priors(scene) | {"star.dra": dist.Uniform(-20.0, 20.0)}
    result = fit(scene, flat, data, max_steps=5)
    z_flat, sd = P.scene_evidence(result, data, flat)
    assert np.isfinite(z_flat) and sd["star.dra"] > 0
    np.testing.assert_allclose(z_flat, float(P.log_evidence(result, data, priors=flat)), rtol=0, atol=1e-6)
    s = 0.5 * sd["star.dra"]
    normal = flat | {"star.dra": dist.Normal(float(result.model.star.dra), s)}
    z_normal, _ = P.scene_evidence(result, data, normal)
    assert np.isfinite(z_normal)
    log_p_shift = float(normal["star.dra"].log_prob(result.model.star.dra) - flat["star.dra"].log_prob(result.model.star.dra))
    np.testing.assert_allclose(z_normal - z_flat - log_p_shift, -0.5 * np.log1p(sd["star.dra"] ** 2 / s**2), atol=1e-6)


@pytest.mark.parametrize(
    "opts, tag",
    [
        (dict(binary="paper", noise="current"), "AI_Sco_gp-polar_unbounded"),  # paper keeps the earlier unbounded names (job 21054320)
        (dict(binary="motion", noise="current"), "AI_Sco_gp-polar_binmotion_unbounded"),
        (dict(binary="motion", noise="nightw"), "AI_Sco_gp-polar_nightw_binmotion_unbounded"),
        (dict(binary="epoch", noise="current", wide_field=True), "AI_Sco_gp-polar_binepoch_unbounded_wide"),
    ],
)
def test_binary_night_and_wide_runs_are_named_apart(opts, tag):
    import types

    args = types.SimpleNamespace(target="AI_Sco", strategy="gp-polar", star_index=None, bg_index=None, mean=P.DEFAULT_MEAN,
                                 unbounded=True, **opts)
    seen = []
    P.per_strategy(lambda a, out, strat: seen.append(P.TARGETS[a.target]["key"]), args, None)
    assert seen == [tag]


class _UV:  # what support_floor_mas reads: baselines (m) and wavelengths (m)
    u = np.array([9.91, 40.0, 130.0])
    v = np.zeros(3)
    wavel = np.array([1.5e-6, 1.65e-6, 1.8e-6])


def test_the_support_floor_is_where_the_shortest_baseline_halves_a_ring():
    from scipy.special import j0

    floor = P.support_floor_mas(_UV)
    x = 2 * np.pi * 9.91 * floor / 206264806.2 / 1.8e-6
    np.testing.assert_allclose(j0(x), 0.5, atol=2e-3)
    assert P.check_support(dict(key="t", mask_mas=floor + 0.1), _UV) == floor
    with pytest.raises(SystemExit, match="below the data's field"):  # PS Gem's old 5 mas support
        P.check_support(dict(key="PS_Gem", mask_mas=5.0), _UV)


def test_image_degeneracy_flags_flat_and_edge_piled_images():
    n, pix, rs, hole = 64, 0.3, 9.0, 0.4
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx - (n - 1) / 2, yy - (n - 1) / 2) * pix
    ring = np.exp(-0.5 * ((r - 3.0) / 0.6) ** 2)
    flat = (r <= rs).astype(float)
    edge = np.exp(-0.5 * ((r - 8.6) / 0.3) ** 2) + 1e-3 * (r <= rs)  # a faint field to the support edge, flux piled on it
    ok = P.image_degeneracy(ring, pix, rs, hole)
    assert ok["image_degenerate"] == [] and ok["img_flat_frac"] < 0.3 and ok["img_edge_frac"] < 0.01
    re = np.hypot(xx - (n - 1) / 2, 2 * (yy - (n - 1) / 2)) * pix  # elliptical radius: a polar field inclined by 60 deg
    inside = re <= 6.0
    ellipse = (ring + 1e-3) * inside  # the field is an ellipse, well inside the support circle
    ell_edge = (ellipse + np.exp(-0.5 * ((re - 5.8) / 0.2) ** 2)) * inside
    assert P.image_degeneracy(ellipse, pix, rs, hole)["image_degenerate"] == []
    assert P.image_degeneracy(ell_edge, pix, rs, hole)["image_degenerate"][-1].startswith("edge-piled")  # inside the support circle
    assert P.image_degeneracy(flat, pix, rs, hole)["image_degenerate"][0].startswith("flat")
    assert any(w.startswith("edge-piled") for w in P.image_degeneracy(edge, pix, rs, hole)["image_degenerate"])
    assert P.image_degeneracy(0 * ring, pix, rs, hole)["image_degenerate"] == ["empty"]
