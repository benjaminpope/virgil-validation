"""External confirmation of virgil.models.GravityDarkenedStar's pole_pa
convention against published rapid-rotator observations, dLux optical
wavefront propagation, and oimodeler (ledger P13).

1. Literature rapid rotators (``roots=["literature"]``):
   * Altair (Monnier et al. 2007, Science 317, 342): visible rotational pole
     at ``PA_pole = 298.2 deg`` (-61.8 deg East of North), placing the hot
     bright pole in the **West-Northwest** sky quadrant (``dra < 0, ddec > 0``)
     and stretching the oblate equator along ``28.2 deg``, so squared
     visibility at fixed baseline length is smaller along ``28.2 deg`` than
     along ``298.2 deg``.
   * Achernar (Domiciano de Souza et al. 2014, A&A 569, A10): visible pole at
     ``PA_pole = 216.9 deg``, placing the bright pole in the **South-Southwest**
     quadrant (``dra < 0, ddec < 0``) and the equatorial major axis along
     ``126.9 deg``.
2. Optical wavefront propagation (``roots=["dlux"]``):
   * An inclined ELR11 gravity-darkened surface cloud at ``pa = 33 deg`` is
     propagated through a 7-hole non-redundant pupil mask via ``dLux``
     Fraunhofer diffraction and calibrated by a point-source PSF. ``virgil``'s
     ``GravityDarkenedStar(..., pa=33)`` matches all 21 baseline ``V^2``
     values and 35 closure phases, whereas ``pa + 90``, ``pa + 180``, and
     ``-pa`` fail by 50-100x the numerical discrepancy.
3. Upstream defect in ``oimodeler.oimFastRotator`` (ledger ``P13``,
   strict upstream xfail at ``dim=64``):
   * ``fastRotator`` applies ``np.rot90(im)`` before ``oimComponentImage``
     maps row index 0 to South, inverting the visible pole's position angle by
     180 deg (placing ``pa = 33 deg`` in the SW quadrant at ``213 deg`` and
     flipping closure-phase sign), and parameterizes gravity darkening via
     ``a = (2/3) * rot**0.4`` and ``K = sin(1/3) * pi`` instead of the O(omega^2)
     Roche gravity law.
"""

import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from crosscheck import array, elr, nrm, sky
from evidence.plugin import record

pytest.importorskip("dLux")
oim = pytest.importorskip("oimodeler")
vm = pytest.importorskip("virgil.models")

pytestmark = [pytest.mark.x64, pytest.mark.external]

WAVEL_H = 1.65e-6
WL_NRM = float(np.float32(4.8e-6))
PIX_NRM = 25.0
DLUX_ROTATOR_PA = 33.0


def _render_photocenter_mas(star, fov_mas, npix=128):
    """Flux-weighted photocentre (dra East, ddec North, sky PA deg) from
    SourceModel.render (row 0 = North, column 0 = East)."""
    img = np.asarray(star.render(npix=npix, fov_mas=fov_mas))
    axis = ((npix - 1) / 2.0 - np.arange(npix)) * (fov_mas / npix)
    total = float(img.sum())
    dra = float((img * axis[None, :]).sum() / total)
    ddec = float((img * axis[:, None]).sum() / total)
    pa_deg = float(np.rad2deg(np.arctan2(dra, ddec)) % 360.0)
    return dra, ddec, pa_deg


@pytest.mark.parametrize(
    "star_name,diam_eq,omega,inc_deg,pa_pole_deg,t_pole,sign_ra,sign_dec",
    [
        # Monnier et al. (2007, Science 317, 342): Altair visible pole at -61.8 deg = 298.2 deg (WNW).
        ("Altair", 3.46, 0.92, 57.2, 298.2, 8450.0, -1, +1),
        # Domiciano de Souza et al. (2014, A&A 569, A10): Achernar visible pole at 216.9 deg (SSW).
        ("Achernar", 2.36, 0.96, 60.6, 216.9, 17124.0, -1, -1),
    ],
)
@pytest.mark.validates(
    "virgil.models.GravityDarkenedStar",
    roots=["literature"],
    property="pole_pa",
    headline="photocenter_pa_error_deg",
)
def test_published_rapid_rotators_pole_quadrant_and_oblate_anisotropy(
    star_name, diam_eq, omega, inc_deg, pa_pole_deg, t_pole, sign_ra, sign_dec
):
    """Altair (PA_pole = 298.2 deg, WNW) and Achernar (PA_pole = 216.9 deg,
    SSW) place the hot visible pole in the published sky quadrant, yield lower
    V^2 along the oblate equatorial major axis (PA_pole + 90 deg) than along
    the polar minor axis, and flip closure-phase sign under PA_pole + 180 deg."""
    star = vm.GravityDarkenedStar(
        diam_eq, omega=omega, inc=inc_deg, pa=pa_pole_deg, n_lat=96, t_pole=t_pole, wavel0=WAVEL_H
    )
    dra, ddec, pa_phot = _render_photocenter_mas(star, fov_mas=3.0 * diam_eq)
    pa_err = abs((pa_phot - pa_pole_deg + 180.0) % 360.0 - 180.0)
    assert np.sign(dra) == sign_ra and abs(dra) > 0.01
    assert np.sign(ddec) == sign_dec and abs(ddec) > 0.01
    assert pa_err < 0.25

    # Rotational oblateness: equatorial major axis at pa_pole + 90 deg is more
    # resolved (smaller V^2) than polar minor axis at matched 150 m H-band baseline.
    b_m = 150.0
    pa_eq_deg = (pa_pole_deg + 90.0) % 360.0
    us = jnp.array([b_m * np.sin(np.deg2rad(pa_pole_deg)), b_m * np.sin(np.deg2rad(pa_eq_deg))])
    vs = jnp.array([b_m * np.cos(np.deg2rad(pa_pole_deg)), b_m * np.cos(np.deg2rad(pa_eq_deg))])
    v2 = np.abs(np.asarray(star.model(us, vs, WAVEL_H))) ** 2
    v2_polar, v2_equatorial = float(v2[0]), float(v2[1])
    assert v2_equatorial < v2_polar

    # Asymmetric closure-phase triangle sensitive to polar skewness; flipping pole
    # by 180 deg must invert the closure phase.
    b1 = 110.0 if star_name == "Altair" else 160.0
    u1, v1 = b1 * np.sin(np.deg2rad(pa_pole_deg)), b1 * np.cos(np.deg2rad(pa_pole_deg))
    u2, v2_leg = 0.7 * b1 * np.sin(np.deg2rad(pa_pole_deg + 25.0)), 0.7 * b1 * np.cos(np.deg2rad(pa_pole_deg + 25.0))
    tri_u = jnp.array([u1, u2, u1 + u2])
    tri_v = jnp.array([v1, v2_leg, v1 + v2_leg])
    vis_tri = np.asarray(star.model(tri_u, tri_v, WAVEL_H))
    cp_deg = float(np.rad2deg(np.angle(vis_tri[0] * vis_tri[1] * np.conj(vis_tri[2]))))

    star_flipped = vm.GravityDarkenedStar(
        diam_eq, omega=omega, inc=inc_deg, pa=pa_pole_deg + 180.0, n_lat=96, t_pole=t_pole, wavel0=WAVEL_H
    )
    vis_flip = np.asarray(star_flipped.model(tri_u, tri_v, WAVEL_H))
    cp_flip = float(np.rad2deg(np.angle(vis_flip[0] * vis_flip[1] * np.conj(vis_flip[2]))))

    record("photocenter_pa_error_deg", pa_err)
    record("v2_polar_minus_equatorial", v2_polar - v2_equatorial)
    record("abs_cp_deg", abs(cp_deg))
    assert abs(cp_deg) > 10.0
    assert cp_flip == pytest.approx(-cp_deg, abs=1e-9)


@pytest.fixture(scope="module")
def dlux_rotator_observables():
    """Calibrated V^2 and closure phases of an inclined rapid rotator at
    pa = 33 deg propagated through dLux's 7-hole aperture mask."""
    optics = nrm.dlux_optics(npix=192, pixel_mas=PIX_NRM, wf_npix=192)
    cal = nrm.image_dlux(sky.point(), WL_NRM, optics)
    cloud = elr.cloud(
        220.0, 0.95, 60.0, DLUX_ROTATOR_PA, n_theta=64, n_phi=128, t_pole=8500.0, wavel=WL_NRM
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
    return {
        "u_bl": u_bl, "v_bl": v_bl, "v2": v2_dlux,
        "u1": u1, "v1": v1, "u2": u2, "v2_leg": v2, "cp_deg": cp_dlux,
    }


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


@pytest.mark.validates(
    "virgil.models.GravityDarkenedStar",
    roots=["dlux"],
    property="pole_pa",
    headline="max_abs_dCP_deg",
)
def test_gravity_darkened_star_pole_pa_matches_dlux_wavefront_propagation(dlux_rotator_observables):
    """Propagating the rapid rotator at pa = 33 deg through dLux's 7-hole
    diffraction model reproduces GravityDarkenedStar's V^2 anisotropy and
    closure phases across all 21 baselines and 35 closure triangles."""
    dv2, dcp = _eval_virgil_on_nrm(dlux_rotator_observables, DLUX_ROTATOR_PA)
    cp_peak = float(np.max(np.abs(dlux_rotator_observables["cp_deg"])))
    record("max_abs_dV2", dv2)
    record("max_abs_dCP_deg", dcp)
    record("max_abs_CP_deg", cp_peak)
    assert cp_peak > 2.5
    assert dv2 < 3e-3
    assert dcp < 0.08


@pytest.mark.parametrize(
    "label,wrong_pa",
    [
        ("pa_plus_90", DLUX_ROTATOR_PA + 90.0),
        ("pa_plus_180", DLUX_ROTATOR_PA + 180.0),
        ("minus_pa", -DLUX_ROTATOR_PA),
    ],
)
@pytest.mark.validates("virgil.models.GravityDarkenedStar", roots=["dlux"], kind="control")
def test_gravity_darkened_star_wrong_pole_pa_fails_against_dlux(dlux_rotator_observables, label, wrong_pa):
    """Wrong pole orientations (pa + 90, pa + 180, -pa) miss dLux's closure
    phases by > 5 deg (> 60x the true discrepancy)."""
    dv2, dcp = _eval_virgil_on_nrm(dlux_rotator_observables, wrong_pa)
    record(f"control_{label}_dV2", dv2)
    record(f"control_{label}_dCP_deg", dcp)
    assert dcp > 5.0


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="P13: oimodeler oimFastRotator flips pole PA by 180 deg via np.rot90 and uses rot**0.4 / K=sin(1/3)*pi",
)
@pytest.mark.validates("oimodeler", roots=["mathematics"], kind="upstream")
def test_p13_oimodeler_fast_rotator_pole_pa_and_gravity_darkening_law():
    """oimodeler's oimFastRotator(dim=64, pa=33) places the bright visible pole
    at PA + 180 deg = 213 deg (SW instead of NE) because fastRotator calls
    np.rot90(im) before oimComponentImage maps row 0 to South, inverting the
    closure-phase sign; and its ad-hoc a = (2/3) rot**0.4, K = sin(1/3) pi
    temperature law over-darkens slow rotators (O(omega^0.4) vs O(omega^2))."""
    from oimodeler.oimCustomComponents import oimFastRotator

    fr = oimFastRotator(dim=64, incl=60.0, rot=0.95, Tpole=8500.0, dpole=2.5, pa=33.0, beta=0.25)
    model = oim.oimModel([fr])

    # Short baselines along East (u > 0) and North (v > 0): under OIFITS sign
    # V(u, v) = integral I(dra, ddec) exp(-2pi i (u dra + v ddec) / wl), a bright
    # pole in the NE quadrant (pa = 33 deg: dra > 0, ddec > 0) requires Im(V) < 0
    # on both +u and +v short baselines, and positive closure phase on the
    # standard rapid-rotator triangle.
    ccf_short = model.getComplexCoherentFlux(
        np.array([15.0, 0.0]) / WAVEL_H,
        np.array([0.0, 15.0]) / WAVEL_H,
        np.full(2, WAVEL_H),
    )
    # Effective photocentre shift (mas) inferred from low-frequency phase slope:
    mas = np.deg2rad(1.0 / 3.6e6)
    dra_eff = float(-np.angle(ccf_short[0]) / (2.0 * np.pi * (15.0 / WAVEL_H) * mas))
    ddec_eff = float(-np.angle(ccf_short[1]) / (2.0 * np.pi * (15.0 / WAVEL_H) * mas))
    pa_eff = float(np.rad2deg(np.arctan2(dra_eff, ddec_eff)) % 360.0)
    pa_flip_err = abs((pa_eff - 33.0 + 180.0) % 360.0 - 180.0)
    record("oim_effective_pole_pa_deg", pa_eff)
    record("oim_pole_pa_error_deg", pa_flip_err)
    assert pa_flip_err < 5.0
