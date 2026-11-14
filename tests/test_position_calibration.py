"""External confirmation of virgil.orbits.PositionData's astrometric
calibration convention against PMOIRED and CANDID.

Following Octofitter's convention (Thompson et al. 2023, AJ 166, 164),
``PositionData.model(orbit, north_angle=delta, plate_scale=m)`` relates the
measured relative astrometry ``(rho_meas, theta_meas)`` of a companion to its
true sky position ``(rho_sky, theta_sky)`` via:

    theta_meas = theta_sky + delta,    rho_meas = m * rho_sky.

To confirm this end-to-end without circularity, we synthesize 5-epoch OIFITS
observations of a Keplerian binary on a 3-station interferometer whose
recorded ``(u_rec, v_rec)`` coordinates carry a baseline scale calibration
factor ``m`` and north-angle rotation ``+delta`` relative to true East/North
(so ``u_true * dra_sky + v_true * ddec_sky = u_rec * dra_meas + v_rec * ddec_meas``).
Both ``PMOIRED`` (``doFit``) and ``CANDID`` (``fitMap``) independently fit the
binary's astrometry from each OIFITS file, and ``PositionData`` with
``(plate_scale=m, north_angle=delta)`` achieves near-zero residual and exact
parameter recovery on both packages' extracted positions while ``-delta`` and
``1 / m`` negative controls fail by > 50 sigma.
"""

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

pytest.importorskip("pmoired")
pytest.importorskip("jaxoplanet", reason="virgil.orbits needs virgil-astro[orbits]")
vo = pytest.importorskip("virgil.orbits")

pytestmark = [
    pytest.mark.x64,
    pytest.mark.external,
    pytest.mark.skipif(not candid_bridge.available(), reason="CANDID not installed (.venv-candid)"),
]

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
def extracted_astrometry(tmp_path_factory):
    """Multi-epoch binary astrometry independently extracted by PMOIRED and
    CANDID from OIFITS files recorded on a miscalibrated interferometer."""
    base_dir = tmp_path_factory.mktemp("pos_cal")
    dra_sky, ddec_sky = co.position(MJDS - T_REF, *ELEMENTS)
    d_rad = np.deg2rad(NORTH_ANGLE_DEG)

    pm_sep, pm_pa = [], []
    ca_sep, ca_pa = [], []

    for idx, (x_s, y_s) in enumerate(zip(dra_sky, ddec_sky)):
        path = base_dir / f"epoch_{idx}.fits"

        def vis_miscal(u_rec, v_rec, w, xs=float(x_s), ys=float(y_s)):
            # Recorded baselines rotated/scaled so u_true . x_sky = u_rec . x_meas
            # with x_meas = m * [[cos d, sin d], [-sin d, cos d]] @ x_sky.
            u_true = PLATE_SCALE_M * (u_rec * np.cos(d_rad) - v_rec * np.sin(d_rad))
            v_true = PLATE_SCALE_M * (u_rec * np.sin(d_rad) + v_rec * np.cos(d_rad))
            return (
                sky.vis_uniform_disk(u_true, v_true, w, STAR_DIAM_MAS)
                + FLUX_RATIO * sky.vis_point(u_true, v_true, w, xs, ys)
            ) / (1.0 + FLUX_RATIO)

        simulate.observe(
            path,
            vis_miscal,
            UTS3,
            hour_angles_h=np.linspace(-3.0, 3.0, 7),
            wavelengths=np.linspace(1.5e-6, 2.4e-6, 6),
            dec_deg=-50.0,
            sigma_v2=0.005,
            sigma_cp_deg=0.25,
            rng=None,
        )

        # Independent extraction 1: CANDID blind 2D grid + LM fitMap
        ca_res = candid_bridge.run(
            {
                "task": "fitmap",
                "path": str(path),
                "rmin": 3.0,
                "rmax": 13.0,
                "step": 2.5,
            }
        )
        cx, cy = ca_res["best"]["x"], ca_res["best"]["y"]
        ca_sep.append(np.hypot(cx, cy))
        ca_pa.append(np.rad2deg(np.arctan2(cx, cy)) % 360.0)

        # Independent extraction 2: PMOIRED least-squares doFit seeded 0.3 mas
        # away from CANDID's blind fit (no true sky coordinates or delta/m used!)
        pm_res = pm.fit(
            path,
            {
                "star,ud": 0.8,
                "star,f": 1.0,
                "c,ud": 0.0,
                "c,x": cx + 0.2,
                "c,y": cy - 0.2,
                "c,f": 0.035,
            },
            free=["star,ud", "c,x", "c,y", "c,f"],
        )
        px, py = pm_res["best"]["c,x"], pm_res["best"]["c,y"]
        pm_sep.append(np.hypot(px, py))
        pm_pa.append(np.rad2deg(np.arctan2(px, py)) % 360.0)

    return {
        "pmoired": (np.array(pm_sep), np.array(pm_pa)),
        "candid": (np.array(ca_sep), np.array(ca_pa)),
    }


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


@pytest.mark.parametrize(
    "pkg",
    [
        pytest.param(
            "pmoired",
            marks=pytest.mark.validates(
                "virgil.orbits.PositionData",
                roots=["pmoired"],
                property="calibration",
                headline="max_residual_mas",
            ),
        ),
        pytest.param(
            "candid",
            marks=pytest.mark.validates(
                "virgil.orbits.PositionData",
                roots=["candid"],
                property="calibration",
                headline="max_residual_mas",
            ),
        ),
    ],
)
def test_position_data_calibration_matches_pmoired_and_candid_astrometry(extracted_astrometry, pkg):
    """PositionData(..., plate_scale=m, north_angle=delta) reproduces the
    binary astrometry extracted by both PMOIRED and CANDID to < 1e-4 mas and
    recovers (delta, m) from the multi-epoch likelihood."""
    sep_meas, pa_meas = extracted_astrometry[pkg]
    sep_err = np.full_like(sep_meas, 0.01)   # 10 uas astrometric uncertainty
    pa_err = np.full_like(pa_meas, 0.05)     # 0.05 deg PA uncertainty

    orbit = vo.KeplerOrbit(*ELEMENTS, t_ref=T_REF)
    pos_data = vo.PositionData.from_sep_pa(MJDS, sep_meas, pa_meas, sep_err, pa_err, t_ref=T_REF)

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
    assert max_res_mas < 1e-4
    assert delta_err < 1e-4
    assert m_err < 1e-5


@pytest.mark.parametrize("pkg", ["pmoired", "candid"])
@pytest.mark.parametrize(
    "control_name,wrong_delta,wrong_m",
    [
        ("minus_delta", -NORTH_ANGLE_DEG, PLATE_SCALE_M),
        ("inverse_plate_scale", NORTH_ANGLE_DEG, 1.0 / PLATE_SCALE_M),
    ],
)
@pytest.mark.validates("virgil.orbits.PositionData", roots=["pmoired", "candid"], kind="control")
def test_position_data_wrong_calibration_sign_and_scale_fail_by_over_50_sigma(
    extracted_astrometry, pkg, control_name, wrong_delta, wrong_m
):
    """Reversing the sign of north_angle (-delta, as in conventions that
    subtract true North) or inverting plate_scale (1 / m) degrades loglike by
    more than 50 sigma against both PMOIRED and CANDID astrometry."""
    sep_meas, pa_meas = extracted_astrometry[pkg]
    pos_data = vo.PositionData.from_sep_pa(
        MJDS, sep_meas, pa_meas, np.full(5, 0.01), np.full(5, 0.05), t_ref=T_REF
    )
    orbit = vo.KeplerOrbit(*ELEMENTS, t_ref=T_REF)
    ll_true = float(pos_data.loglike(orbit, north_angle=NORTH_ANGLE_DEG, plate_scale=PLATE_SCALE_M))
    ll_wrong = float(pos_data.loglike(orbit, north_angle=wrong_delta, plate_scale=wrong_m))
    nsigma_miss = float(np.sqrt(max(0.0, 2.0 * (ll_true - ll_wrong))))
    record(f"control_{pkg}_{control_name}_sigma", nsigma_miss)
    assert nsigma_miss > 50.0
