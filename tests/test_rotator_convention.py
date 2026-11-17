"""External confirmation of virgil.models.GravityDarkenedStar's pole_pa
convention against PMOIRED's native rotastar model, dLux optical wavefront
propagation, and oimodeler (ledger P13 and P14).

1. Non-circular ``pole_pa`` convention confirmation (``roots=["pmoired"]``):
   * ``PMOIRED`` (Mérand 2022) includes an independent Roche gravity-darkened
     rapid-rotator implementation (``pmoired.rotastar.Vrota``, after Aufdenberg
     et al. 2006) invoked via ``diameq, dist, Tpole, mass, omega, incl, projang``,
     where ``projang`` is the sky position angle of the visible rotational pole
     (degrees North through East).
   * Converting ``PMOIRED``'s critical angular-velocity fraction
     ``w_pm = Omega / sqrt((8/27) G M / R_pole^3)`` to ELR11's equatorial
     Keplerian fraction ``w_elr = w_pm sqrt((8/27) (R_eq / R_pole)^3)`` via the
     exact spherical-Roche ratio ``R_eq / R_pole = (3 / w_pm) cos((pi + arccos w_pm) / 3)``
     matches ``GravityDarkenedStar(..., pa=33)`` across all baselines and
     closure triangles (``max_dv2 < 0.01``, ``max_dcp < 2.5 deg`` on ~168 deg
     closure phases). The ~2 deg residual reflects three physical modelling
     differences between the two codes at ``w_pm = 0.95`` (``w_elr = 0.750``,
     ``T_pole = 9500 K``): ELR11 barotropic gravity darkening (``T_eq = 7490 K``)
     versus von Zeipel ``beta = 0.25`` (``T_eq = 6820 K``), Planck ``B_lambda(T)``
     versus ATLAS9 ``F_lambda(T_eff, log g)`` surface spectra, and isotropic
     emission versus spherical SATLAS limb darkening. Wrong orientations
     (``pa + 90``, ``pa + 180``, and ``-pa``) miss by > 50 deg.
2. Optical wavefront propagation (``roots=["dlux"]``, without claiming
   ``property="pole_pa"`` because ``crosscheck.elr.cloud`` supplies the sky
   rotation):
   * Propagating an inclined ELR11 surface cloud through ``dLux``'s 7-hole
     non-redundant mask reproduces ``GravityDarkenedStar`` across all 21
     baselines and 35 closure phases.
3. Two differences from what we expected in ``oimodeler.oimFastRotator`` (ledger
   ``P13`` and ``P14``, each pinned by its own strict xfail at ``dim=64``):
   * ``P13``: the visible pole's position angle lands about 180 deg from the
     IAU convention we expected; ``fastRotator`` applies ``np.rot90(im)`` before
     ``oimComponentImage`` maps row 0 to South, which may be the source.
   * ``P14``: the equator-on ``V^2`` anisotropy across slow rotation
     (``incl = 90 deg``, invariant under ``P13``'s 180 deg flip and sub-pixel
     shift) has log-log slope ~0.45, not the ``O(omega^2)`` slope 2 of
     centrifugal scaling; ``fastRotator`` sets ``a = (2/3) * rot**0.4`` and
     ``K = sin(1/3) * pi``, which may be the cause. Grid convergence is not
     checked.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crosscheck import array, elr, nrm, simulate, sky
from evidence.plugin import record
from external_bridge import pmoired_models as pm

vm = pytest.importorskip("virgil.models")

pytestmark = pytest.mark.x64

WAVEL_H = 1.65e-6
WAVEL_K = 2.2e-6
WAVEL_K_UM = 2.2
WL_NRM = float(np.float32(4.8e-6))
PIX_NRM = 25.0
ROTATOR_PA = 33.0
ROTATOR_DIAM_EQ = 3.2
ROTATOR_INC = 55.0
ROTATOR_TPOLE = 9500.0
W_PMOIRED = 0.95

# Four-station array scaled to probe both the first lobe and the strong
# post-null closure-phase regime (~168 deg) at 2.2 um cleanly away from the null.
STATIONS_ROT = 0.85 * np.array([
    [0.0, 0.0],
    [110.0, 32.0],
    [-52.0, 130.0],
    [70.0, -82.0],
])


def _pmoired_to_elr_omega(w_pm):
    """Exact Roche conversion from w_pm = Omega / sqrt((8/27) G M / R_pole^3)
    to ELR11's w_elr = Omega / sqrt(G M / R_eq^3)."""
    req_over_rp = (3.0 / w_pm) * np.cos((np.pi + np.arccos(w_pm)) / 3.0)
    return float(w_pm * np.sqrt((8.0 / 27.0) * req_over_rp**3))


@pytest.fixture(scope="module")
def pmoired_rotator_samples(tmp_path_factory):
    """Native PMOIRED rotastar (Vrota) V^2 and closure phases at projang = 33 deg."""
    pytest.importorskip("pmoired")
    path = tmp_path_factory.mktemp("rotator_pmoired") / "rotator_carrier.fits"
    simulate.observe(
        path,
        lambda u, v, w: np.ones(np.broadcast(u, v, w).shape, complex),
        STATIONS_ROT,
        hour_angles_h=np.array([-1.5, 0.5, 2.0]),
        wavelengths=np.array([WAVEL_K]),
        dec_deg=-35.0,
        sigma_v2=1e-3,
        sigma_cp_deg=0.1,
    )
    params = {
        "star,diameq": ROTATOR_DIAM_EQ,
        "star,dist": 5.0,
        "star,Tpole": ROTATOR_TPOLE,
        "star,mass": 2.0,
        "star,omega": W_PMOIRED,
        "star,incl": ROTATOR_INC,
        "star,projang": ROTATOR_PA,
        "star,beta": 0.25,
    }
    return pm.model_samples(path, params)


def _compare_virgil_to_pmoired_rotastar(samples, pa_deg):
    w_elr = _pmoired_to_elr_omega(W_PMOIRED)
    star = vm.GravityDarkenedStar(
        ROTATOR_DIAM_EQ,
        omega=w_elr,
        inc=ROTATOR_INC,
        pa=pa_deg,
        n_lat=96,
        t_pole=ROTATOR_TPOLE,
        wavel0=WAVEL_K,
    )
    u_m = jnp.asarray(samples["u"] * WAVEL_K_UM)
    v_m = jnp.asarray(samples["v"] * WAVEL_K_UM)
    v2 = np.abs(np.asarray(star.model(u_m, v_m, WAVEL_K))) ** 2

    t1 = np.asarray(
        star.model(jnp.asarray(samples["u1"] * WAVEL_K_UM), jnp.asarray(samples["v1"] * WAVEL_K_UM), WAVEL_K)
    )
    t2 = np.asarray(
        star.model(jnp.asarray(samples["u2"] * WAVEL_K_UM), jnp.asarray(samples["v2_"] * WAVEL_K_UM), WAVEL_K)
    )
    t3 = np.asarray(
        star.model(
            jnp.asarray((samples["u1"] + samples["u2"]) * WAVEL_K_UM),
            jnp.asarray((samples["v1"] + samples["v2_"]) * WAVEL_K_UM),
            WAVEL_K,
        )
    )
    cp_deg = np.rad2deg(np.angle(t1 * t2 * np.conj(t3)))
    dv2 = float(np.max(np.abs(v2 - samples["v2"])))
    dcp = float(np.max(np.abs((cp_deg - samples["t3phi"] + 180.0) % 360.0 - 180.0)))
    cp_peak = float(np.max(np.abs(samples["t3phi"])))
    return dv2, dcp, cp_peak


@pytest.mark.external
@pytest.mark.validates(
    "virgil.models.GravityDarkenedStar",
    roots=["pmoired"],
    property="pole_pa",
    headline="max_abs_dV2",
)
def test_gravity_darkened_star_pole_pa_matches_pmoired_rotastar(pmoired_rotator_samples):
    """GravityDarkenedStar(..., pa=33.0) matches PMOIRED's independent rotastar
    model (projang=33.0) across all baselines and strong (~168 deg) closure phases."""
    dv2, dcp, cp_peak = _compare_virgil_to_pmoired_rotastar(pmoired_rotator_samples, ROTATOR_PA)
    record("max_abs_dV2", dv2)
    record("max_abs_dCP_deg", dcp)
    record("max_abs_CP_deg", cp_peak)
    assert cp_peak > 100.0
    assert dv2 < 0.01
    assert dcp < 2.5


@pytest.mark.external
@pytest.mark.parametrize(
    "label,wrong_pa",
    [
        ("pa_plus_90", ROTATOR_PA + 90.0),
        ("pa_plus_180", ROTATOR_PA + 180.0),
        ("minus_pa", -ROTATOR_PA),
    ],
)
@pytest.mark.validates("virgil.models.GravityDarkenedStar", roots=["pmoired"], kind="control")
def test_gravity_darkened_star_wrong_pole_pa_fails_against_pmoired_rotastar(
    pmoired_rotator_samples, label, wrong_pa
):
    """Wrong pole orientations (pa + 90, pa + 180, -pa) miss PMOIRED's rotastar
    closure phases by > 50 deg (> 25x the gravity-darkening, atmosphere, and
    limb-darkening modelling offset)."""
    dv2, dcp, _ = _compare_virgil_to_pmoired_rotastar(pmoired_rotator_samples, wrong_pa)
    record(f"control_{label}_dV2", dv2)
    record(f"control_{label}_dCP_deg", dcp)
    assert dcp > 50.0


@pytest.fixture(scope="module")
def dlux_rotator_observables():
    """Calibrated V^2 and closure phases of an inclined rapid rotator at
    pa = 33 deg propagated through dLux's 7-hole aperture mask."""
    pytest.importorskip("dLux")
    optics = nrm.dlux_optics(npix=192, pixel_mas=PIX_NRM, wf_npix=192)
    cal = nrm.image_dlux(sky.point(), WL_NRM, optics)
    cloud = elr.cloud(
        220.0, 0.95, 60.0, ROTATOR_PA, n_theta=64, n_phi=128, t_pole=8500.0, wavel=WL_NRM
    )
    sci = nrm.image_dlux(cloud, WL_NRM, optics)
    vis_fn = nrm.calibrated_vis_fn(sci, cal, WL_NRM, PIX_NRM)
    u_bl, v_bl = array.pupil_uv(nrm.HOLES)
    v2_dlux = np.abs(vis_fn(u_bl, v_bl, WL_NRM)) ** 2

    tris = list(itertools.combinations(range(len(nrm.HOLES)), 3))
    u1 = np.array([nrm.HOLES[j, 0] - nrm.HOLES[i, 0] for i, j, k in tris])
    v1 = np.array([nrm.HOLES[j, 1] - nrm.HOLES[i, 1] for i, j, k in tris])
    u2 = np.array([nrm.HOLES[k, 0] - nrm.HOLES[j, 0] for i, j, k in tris])
    v2 = np.array([nrm.HOLES[k, 1] - nrm.HOLES[j, 1] for i, j, k in tris])
    cp_dlux = np.rad2deg(
        np.angle(vis_fn(u1, v1, WL_NRM) * vis_fn(u2, v2, WL_NRM) * np.conj(vis_fn(u1 + u2, v1 + v2, WL_NRM)))
    )
    yield {
        "u_bl": u_bl,
        "v_bl": v_bl,
        "v2": v2_dlux,
        "u1": u1,
        "v1": v1,
        "u2": u2,
        "v2_leg": v2,
        "cp_deg": cp_dlux,
    }
    jax.clear_caches()


def _eval_virgil_on_nrm(obs, pa_deg):
    star = vm.GravityDarkenedStar(
        220.0, omega=0.95, inc=60.0, pa=pa_deg, n_lat=96, t_pole=8500.0, wavel0=WL_NRM
    )
    v2_got = np.abs(np.asarray(star.model(jnp.asarray(obs["u_bl"]), jnp.asarray(obs["v_bl"]), WL_NRM))) ** 2
    t1 = np.asarray(star.model(jnp.asarray(obs["u1"]), jnp.asarray(obs["v1"]), WL_NRM))
    t2 = np.asarray(star.model(jnp.asarray(obs["u2"]), jnp.asarray(obs["v2_leg"]), WL_NRM))
    t3 = np.asarray(
        star.model(jnp.asarray(obs["u1"] + obs["u2"]), jnp.asarray(obs["v1"] + obs["v2_leg"]), WL_NRM)
    )
    cp_got = np.rad2deg(np.angle(t1 * t2 * np.conj(t3)))
    dv2 = float(np.max(np.abs(v2_got - obs["v2"])))
    dcp = float(np.max(np.abs((cp_got - obs["cp_deg"] + 180.0) % 360.0 - 180.0)))
    return dv2, dcp


@pytest.mark.external
@pytest.mark.validates(
    "virgil.models.GravityDarkenedStar",
    roots=["dlux"],
    headline="max_abs_dCP_deg",
)
def test_gravity_darkened_star_matches_dlux_wavefront_propagation(dlux_rotator_observables):
    """Propagating the rapid rotator surface cloud through dLux's 7-hole
    diffraction model reproduces GravityDarkenedStar's V^2 anisotropy and
    closure phases across all 21 baselines and 35 closure triangles."""
    dv2, dcp = _eval_virgil_on_nrm(dlux_rotator_observables, ROTATOR_PA)
    cp_peak = float(np.max(np.abs(dlux_rotator_observables["cp_deg"])))
    record("max_abs_dV2", dv2)
    record("max_abs_dCP_deg", dcp)
    record("max_abs_CP_deg", cp_peak)
    assert cp_peak > 2.5
    assert dv2 < 3e-3
    assert dcp < 0.08


@pytest.mark.external
@pytest.mark.parametrize(
    "label,wrong_pa",
    [
        ("pa_plus_90", ROTATOR_PA + 90.0),
        ("pa_plus_180", ROTATOR_PA + 180.0),
        ("minus_pa", -ROTATOR_PA),
    ],
)
@pytest.mark.validates("virgil.models.GravityDarkenedStar", roots=["dlux"], kind="control")
def test_gravity_darkened_star_wrong_pa_fails_against_dlux(dlux_rotator_observables, label, wrong_pa):
    """Rotated models (pa + 90, pa + 180, -pa) miss dLux's closure phases by
    > 5 deg (> 60x the optical discretization discrepancy)."""
    dv2, dcp = _eval_virgil_on_nrm(dlux_rotator_observables, wrong_pa)
    record(f"control_{label}_dV2", dv2)
    record(f"control_{label}_dCP_deg", dcp)
    assert dcp > 5.0


@pytest.mark.external
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="P13: oimodeler oimFastRotator places the pole PA ~180 deg from the IAU convention we expected",
)
@pytest.mark.validates("oimodeler", roots=["mathematics"], kind="upstream")
def test_p13_oimodeler_fast_rotator_pole_pa_inversion():
    """oimodeler's oimFastRotator(dim=64, pa=33) places the bright visible pole
    about 180 deg from the IAU convention we expected (SW instead of NE),
    which negates every translation-invariant first-lobe closure phase relative
    to crosscheck.elr.cloud regardless of even-grid half-pixel centring or P14.
    fastRotator calls np.rot90(im) before oimComponentImage maps row 0 to
    South, which may be the source; this test does not isolate it."""
    oim = pytest.importorskip("oimodeler")
    from oimodeler.oimCustomComponents import oimFastRotator

    w_elr = _pmoired_to_elr_omega(0.95)
    req_over_rp = (3.0 / 0.95) * np.cos((np.pi + np.arccos(0.95)) / 3.0)
    ref_cloud = elr.cloud(
        2.5 * req_over_rp, w_elr, 60.0, 33.0, n_theta=96, n_phi=192, t_pole=8500.0, wavel=WAVEL_H
    )
    fr_33 = oimFastRotator(dim=64, incl=60.0, rot=0.95, Tpole=8500.0, dpole=2.5, pa=33.0, beta=0.25)
    fr_213 = oimFastRotator(dim=64, incl=60.0, rot=0.95, Tpole=8500.0, dpole=2.5, pa=213.0, beta=0.25)
    m_33 = oim.oimModel([fr_33])
    m_213 = oim.oimModel([fr_213])

    triangles = [
        ((38.0, 14.0), (-19.0, 42.0)),
        ((44.0, -16.0), (-12.0, -35.0)),
        ((-41.0, 18.0), (15.0, 36.0)),
    ]
    cp_elr, cp_oim_33, cp_oim_213 = [], [], []
    for (u1, v1), (u2, v2) in triangles:
        u_arr = np.array([u1, u2, u1 + u2])
        v_arr = np.array([v1, v2, v1 + v2])
        v_ref = sky.visibility(ref_cloud, u_arr, v_arr, WAVEL_H)
        cp_elr.append(float(np.rad2deg(np.angle(v_ref[0] * v_ref[1] * np.conj(v_ref[2])))))
        c_33 = m_33.getComplexCoherentFlux(u_arr / WAVEL_H, v_arr / WAVEL_H, np.full(3, WAVEL_H))
        cp_oim_33.append(float(np.rad2deg(np.angle(c_33[0] * c_33[1] * np.conj(c_33[2])))))
        c_213 = m_213.getComplexCoherentFlux(u_arr / WAVEL_H, v_arr / WAVEL_H, np.full(3, WAVEL_H))
        cp_oim_213.append(float(np.rad2deg(np.angle(c_213[0] * c_213[1] * np.conj(c_213[2])))))

    cp_elr_arr = np.array(cp_elr)
    cp_oim_arr = np.array(cp_oim_33)
    # Closure phases are identically shift-invariant on even dim=64 grids: rotating by 180 deg negates them to rounding
    assert float(np.max(np.abs(cp_oim_arr + np.array(cp_oim_213)))) < 1e-10
    assert float(np.min(np.abs(cp_elr_arr))) > 0.15
    assert float(np.min(np.abs(cp_oim_arr))) > 0.30

    worst_sign_product = float(np.min(cp_oim_arr * cp_elr_arr))
    record("min_cp_sign_product_deg2", worst_sign_product)
    record("max_abs_oim_cp_deg", float(np.max(np.abs(cp_oim_arr))))
    assert np.all(cp_oim_arr * cp_elr_arr > 0.0)


@pytest.mark.external
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="P14: oimodeler oimFastRotator equator-on V^2 anisotropy does not follow O(omega^2) at dim=64 (log-log slope ~0.45)",
)
@pytest.mark.validates("oimodeler", roots=["mathematics"], kind="upstream")
def test_p14_oimodeler_fast_rotator_gravity_darkening_scaling():
    """At equator-on inclination (incl = 90 deg, pa = 0 deg), orthogonal squared
    visibilities V^2(B/lambda, 0) - V^2(0, B/lambda) obtained through oimodeler's
    public oimModel.getComplexCoherentFlux API are invariant to P13's 180 deg
    rotation and to sub-pixel translation, vanish identically for any non-rotating
    star, and scale as O(omega^2) (log-log slope 2.0 across rot = 0.01, 0.02, 0.04)
    under physical Roche/von Zeipel/ELR11 centrifugal distortion and gravity
    darkening; oimFastRotator at dim = 64 gives log-log slope ~ 0.45 (its
    a = (2/3) * rot**0.4 and K = sin(1/3) * pi may be the cause). Grid
    convergence is not checked, so a pixelization floor is not excluded."""
    oim = pytest.importorskip("oimodeler")
    from oimodeler.oimCustomComponents import oimFastRotator

    rots = np.array([0.01, 0.02, 0.04])
    b_m = 25.0
    anisotropies = []
    for rot in rots:
        fr = oimFastRotator(dim=64, incl=90.0, rot=float(rot), Tpole=10000.0, dpole=3.0, pa=0.0, beta=0.25)
        model = oim.oimModel([fr])
        ccf = model.getComplexCoherentFlux(
            np.array([0.0, b_m, 0.0]) / WAVEL_K,
            np.array([0.0, 0.0, b_m]) / WAVEL_K,
            np.full(3, WAVEL_K),
        )
        v2 = np.abs(ccf / ccf[0]) ** 2
        anisotropies.append(float(abs(v2[1] - v2[2])))

    log_slope = float(np.polyfit(np.log(rots), np.log(anisotropies), 1)[0])
    record("oim_slow_rotator_log_slope", log_slope)
    record("oim_dv2_at_rot_0p01", anisotropies[0])
    assert abs(log_slope - 2.0) < 0.2
