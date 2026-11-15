"""Mathematical check of PositionData's calibration convention and independent
external confirmation of PositionData's astrometric likelihood against PMOIRED
and CANDID.

1. Calibration convention (``property="calibration"``, ``roots=["mathematics"]``):
   * Following Octofitter's convention (Thompson et al. 2023, AJ 166, 164),
     ``PositionData.model(orbit, north_angle=delta, plate_scale=m)`` maps true
     sky relative astrometry ``(rho_sky, theta_sky)`` to measured instrument
     astrometry via ``theta_meas = theta_sky + delta`` and ``rho_meas = m * rho_sky``.
   * Because neither ``PMOIRED`` nor ``CANDID`` defines an astrometric
     ``north_angle`` or ``plate_scale`` nuisance parameter (and no external
     astrometric orbit package in CI defines those nuisances), the calibration
     mapping and multi-epoch recovery of ``(delta, m)`` declare
     ``roots=["mathematics"]`` ONLY, keeping ``PositionData`` honestly at
     ``convention`` until confirmed against an external calibration-aware orbit
     code or published multi-instrument astrometric epochs.
2. Independent external astrometry checks on ``property="likelihood"``
   (``roots=["pmoired"]`` and ``roots=["candid"]``):
   * Standard unrotated, unscaled 5-epoch OIFITS files of a Keplerian binary on
     a 3-station VLTI array are generated from ``crosscheck.orbits.position``
     and ``crosscheck.sky``.
   * ``PMOIRED`` (seeded blindly from its own serial in-process ``VmodelOI`` 2D
     grid scan, with zero coupling to ``CANDID`` or the injected truth) and
     ``CANDID`` (``fitMap``) independently extract ``(sep, pa)`` at every epoch,
     each gated by its own per-test skip guard so the two external roots run
     independently. ``PositionData.from_sep_pa`` reproduces both packages'
     extracted positions to ``< 1e-4 mas``.
"""

import contextlib
import io

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import optimize

from crosscheck import orbits as co
from crosscheck import simulate, sky
from evidence.plugin import record
from external_bridge import candid_bridge
from external_bridge import pmoired_models as pm

pytest.importorskip("jaxoplanet", reason="virgil.orbits needs virgil-astro[orbits]")
vo = pytest.importorskip("virgil.orbits")

pytestmark = pytest.mark.x64

UTS3 = np.array([
    [-9.925, -20.335],
    [14.887, 30.502],
    [103.306, 43.999],
])
ELEMENTS = (400.0, 35.0, 0.25, 52.0, 65.0, 115.0, 9.5)
T_REF = 60000.0
MJDS = np.linspace(60010.0, 60310.0, 5)
PLATE_SCALE_M = 1.045
NORTH_ANGLE_DEG = 7.25
STAR_DIAM_MAS = 0.8
FLUX_RATIO = 0.04


@pytest.fixture(scope="module")
def uncalibrated_binary_oifits(tmp_path_factory):
    """Five noise-free OIFITS epochs of a Keplerian binary in standard unrotated,
    unscaled East/North sky coordinates."""
    base_dir = tmp_path_factory.mktemp("pos_likelihood_oifits")
    dra_sky, ddec_sky = co.position(MJDS - T_REF, *ELEMENTS)
    paths = []
    for idx, (x_s, y_s) in enumerate(zip(dra_sky, ddec_sky)):
        path = base_dir / f"epoch_{idx}.fits"
        simulate.observe(
            path,
            lambda u, v, w, xs=float(x_s), ys=float(y_s): (
                sky.vis_uniform_disk(u, v, w, STAR_DIAM_MAS)
                + FLUX_RATIO * sky.vis_point(u, v, w, xs, ys)
            )
            / (1.0 + FLUX_RATIO),
            UTS3,
            hour_angles_h=np.linspace(-3.0, 3.0, 7),
            wavelengths=np.linspace(1.5e-6, 2.4e-6, 6),
            dec_deg=-50.0,
            sigma_v2=0.005,
            sigma_cp_deg=0.25,
            rng=None,
        )
        paths.append(path)
    return paths


def _blind_pmoired_seed(path, grid_mas=11.0, step_mas=1.0):
    """Serial in-process 2D chi^2 grid scan using PMOIRED's own VmodelOI
    (no multiprocessing, no truth or CANDID input)."""
    import pmoired
    import pmoired.oimodels as om

    with contextlib.redirect_stdout(io.StringIO()):
        oi = pmoired.OI(str(path), verbose=False)
        oi.setupFit({"obs": ["V2", "T3PHI"]}, auto=False)
    data = oi.data[0]
    coords = np.arange(-grid_mas, grid_mas + 0.5 * step_mas, step_mas)
    best_chi2 = np.inf
    best_xy = (1.0, 1.0)
    base_params = {"star,ud": 0.8, "star,f": 1.0, "c,ud": 0.0, "c,f": 0.035}
    for gx in coords:
        for gy in coords:
            if gx == 0.0 and gy == 0.0:
                continue
            m = om.VmodelOI(data, {**base_params, "c,x": float(gx), "c,y": float(gy)})
            c2 = sum(
                float(
                    np.sum(
                        ((m["OI_VIS2"][k]["V2"] - data["OI_VIS2"][k]["V2"]) / data["OI_VIS2"][k]["EV2"]) ** 2
                    )
                )
                for k in data["OI_VIS2"]
            ) + sum(
                float(
                    np.sum(
                        (
                            ((m["OI_T3"][k]["T3PHI"] - data["OI_T3"][k]["T3PHI"] + 180.0) % 360.0 - 180.0)
                            / data["OI_T3"][k]["ET3PHI"]
                        )
                        ** 2
                    )
                )
                for k in data["OI_T3"]
            )
            if c2 < best_chi2:
                best_chi2 = c2
                best_xy = (float(gx), float(gy))
    return best_xy


def _recover_calibration(pos_data, orbit):
    """Recover (north_angle, plate_scale) by maximizing PositionData.loglike."""
    def nll_and_grad(theta):
        val, grad = jax.value_and_grad(
            lambda z: -pos_data.loglike(orbit, north_angle=z[0], plate_scale=z[1])
        )(jnp.asarray(theta, dtype=jnp.float64))
        return float(val), np.asarray(grad, dtype=np.float64)

    res = optimize.minimize(nll_and_grad, x0=np.array([0.0, 1.0]), jac=True, method="BFGS")
    assert res.success
    return float(res.x[0]), float(res.x[1])


@pytest.mark.validates(
    "virgil.orbits.PositionData",
    roots=["mathematics"],
    property="calibration",
    headline="max_residual_mas",
)
def test_position_data_calibration_model_and_recovery():
    """PositionData(..., north_angle=delta, plate_scale=m) maps true sky
    positions via theta_meas = theta_sky + delta and rho_meas = m * rho_sky and
    recovers (delta, m) to < 1e-7 from the multi-epoch astrometric likelihood."""
    dra_sky, ddec_sky = co.position(MJDS - T_REF, *ELEMENTS)
    sep_sky = np.hypot(dra_sky, ddec_sky)
    pa_sky = np.rad2deg(np.arctan2(dra_sky, ddec_sky)) % 360.0

    sep_meas = PLATE_SCALE_M * sep_sky
    pa_meas = (pa_sky + NORTH_ANGLE_DEG) % 360.0
    pos_data = vo.PositionData.from_sep_pa(
        MJDS, sep_meas, pa_meas, np.full(5, 0.01), np.full(5, 0.05), t_ref=T_REF
    )
    orbit = vo.KeplerOrbit(*ELEMENTS, t_ref=T_REF)

    pred_dra, pred_ddec = map(
        np.asarray,
        pos_data.model(orbit, north_angle=NORTH_ANGLE_DEG, plate_scale=PLATE_SCALE_M),
    )
    meas_dra = sep_meas * np.sin(np.deg2rad(pa_meas))
    meas_ddec = sep_meas * np.cos(np.deg2rad(pa_meas))
    max_res_mas = float(np.max(np.hypot(pred_dra - meas_dra, pred_ddec - meas_ddec)))

    delta_hat, m_hat = _recover_calibration(pos_data, orbit)
    delta_err = abs(delta_hat - NORTH_ANGLE_DEG)
    m_err = abs(m_hat - PLATE_SCALE_M)

    record("max_residual_mas", max_res_mas)
    record("abs_north_angle_error_deg", delta_err)
    record("abs_plate_scale_error", m_err)
    assert max_res_mas < 1e-10
    assert delta_err < 1e-7
    assert m_err < 1e-8


@pytest.mark.parametrize(
    "control_name,wrong_delta,wrong_m",
    [
        ("minus_delta", -NORTH_ANGLE_DEG, PLATE_SCALE_M),
        ("inverse_plate_scale", NORTH_ANGLE_DEG, 1.0 / PLATE_SCALE_M),
    ],
)
@pytest.mark.validates("virgil.orbits.PositionData", roots=["mathematics"], kind="control")
def test_position_data_wrong_calibration_sign_and_scale_fail_by_over_50_sigma(
    control_name, wrong_delta, wrong_m
):
    """Reversing the sign of north_angle (-delta) or inverting plate_scale (1 / m)
    degrades PositionData.loglike by > 50 sigma."""
    dra_sky, ddec_sky = co.position(MJDS - T_REF, *ELEMENTS)
    sep_meas = PLATE_SCALE_M * np.hypot(dra_sky, ddec_sky)
    pa_meas = (np.rad2deg(np.arctan2(dra_sky, ddec_sky)) + NORTH_ANGLE_DEG) % 360.0
    pos_data = vo.PositionData.from_sep_pa(
        MJDS, sep_meas, pa_meas, np.full(5, 0.01), np.full(5, 0.05), t_ref=T_REF
    )
    orbit = vo.KeplerOrbit(*ELEMENTS, t_ref=T_REF)
    ll_true = float(pos_data.loglike(orbit, north_angle=NORTH_ANGLE_DEG, plate_scale=PLATE_SCALE_M))
    ll_wrong = float(pos_data.loglike(orbit, north_angle=wrong_delta, plate_scale=wrong_m))
    nsigma_miss = float(np.sqrt(max(0.0, 2.0 * (ll_true - ll_wrong))))
    record(f"control_{control_name}_sigma", nsigma_miss)
    assert nsigma_miss > 50.0


@pytest.mark.external
@pytest.mark.validates(
    "virgil.orbits.PositionData",
    roots=["pmoired"],
    property="likelihood",
    headline="max_residual_mas",
)
def test_position_data_likelihood_matches_pmoired_astrometry(uncalibrated_binary_oifits):
    """PositionData.from_sep_pa reproduces multi-epoch binary astrometry
    extracted blindly by PMOIRED (serial VmodelOI grid scan + doFit) to < 1e-4 mas."""
    pytest.importorskip("pmoired")
    seps, pas = [], []
    for path in uncalibrated_binary_oifits:
        gx, gy = _blind_pmoired_seed(path)
        res = pm.fit(
            path,
            {"star,ud": 0.8, "star,f": 1.0, "c,ud": 0.0, "c,x": gx, "c,y": gy, "c,f": 0.035},
            free=["star,ud", "c,x", "c,y", "c,f"],
        )
        px, py = res["best"]["c,x"], res["best"]["c,y"]
        seps.append(np.hypot(px, py))
        pas.append(np.rad2deg(np.arctan2(px, py)) % 360.0)

    sep_arr, pa_arr = np.array(seps), np.array(pas)
    pos_data = vo.PositionData.from_sep_pa(
        MJDS, sep_arr, pa_arr, np.full(5, 0.01), np.full(5, 0.05), t_ref=T_REF
    )
    orbit = vo.KeplerOrbit(*ELEMENTS, t_ref=T_REF)
    pred_dra, pred_ddec = map(np.asarray, pos_data.model(orbit))
    meas_dra = sep_arr * np.sin(np.deg2rad(pa_arr))
    meas_ddec = sep_arr * np.cos(np.deg2rad(pa_arr))
    max_res_mas = float(np.max(np.hypot(pred_dra - meas_dra, pred_ddec - meas_ddec)))

    wrong_orbit = vo.KeplerOrbit(400.0, 35.0, 0.25, 52.0, 65.0, 122.0, 9.5, t_ref=T_REF)
    dlogl = float(pos_data.loglike(orbit) - pos_data.loglike(wrong_orbit))

    record("max_residual_mas", max_res_mas)
    record("delta_loglike_vs_shifted_Omega", dlogl)
    assert max_res_mas < 1e-4
    assert np.sqrt(max(0.0, 2.0 * dlogl)) > 50.0


@pytest.mark.external
@pytest.mark.validates(
    "virgil.orbits.PositionData",
    roots=["candid"],
    property="likelihood",
    headline="max_residual_mas",
)
def test_position_data_likelihood_matches_candid_astrometry(uncalibrated_binary_oifits):
    """PositionData.from_sep_pa reproduces multi-epoch binary astrometry
    extracted blindly by CANDID (fitMap) to < 1e-4 mas."""
    if not candid_bridge.available():
        pytest.skip("CANDID not installed (.venv-candid)")

    seps, pas = [], []
    for path in uncalibrated_binary_oifits:
        res = candid_bridge.run(
            {
                "task": "fitmap",
                "path": str(path),
                "rmin": 3.0,
                "rmax": 13.0,
                "step": 2.5,
            }
        )
        cx, cy = res["best"]["x"], res["best"]["y"]
        seps.append(np.hypot(cx, cy))
        pas.append(np.rad2deg(np.arctan2(cx, cy)) % 360.0)

    sep_arr, pa_arr = np.array(seps), np.array(pas)
    pos_data = vo.PositionData.from_sep_pa(
        MJDS, sep_arr, pa_arr, np.full(5, 0.01), np.full(5, 0.05), t_ref=T_REF
    )
    orbit = vo.KeplerOrbit(*ELEMENTS, t_ref=T_REF)
    pred_dra, pred_ddec = map(np.asarray, pos_data.model(orbit))
    meas_dra = sep_arr * np.sin(np.deg2rad(pa_arr))
    meas_ddec = sep_arr * np.cos(np.deg2rad(pa_arr))
    max_res_mas = float(np.max(np.hypot(pred_dra - meas_dra, pred_ddec - meas_ddec)))

    wrong_orbit = vo.KeplerOrbit(400.0, 35.0, 0.25, 52.0, 65.0, 122.0, 9.5, t_ref=T_REF)
    dlogl = float(pos_data.loglike(orbit) - pos_data.loglike(wrong_orbit))

    record("max_residual_mas", max_res_mas)
    record("delta_loglike_vs_shifted_Omega", dlogl)
    assert max_res_mas < 1e-4
    assert np.sqrt(max(0.0, 2.0 * dlogl)) > 50.0
