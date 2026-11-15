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
2. Independent external astrometry and likelihood checks on ``property="likelihood"``
   (``roots=["pmoired"]`` and ``roots=["candid"]``):
   * Realistic noisy 5-epoch OIFITS files (``sigma_v2 = 0.002``, ``sigma_cp_deg = 0.1``)
     of a Keplerian binary on a 3-station VLTI array are generated in standard
     unrotated, unscaled East/North sky coordinates.
   * ``PMOIRED`` (seeded blindly from its own serial in-process ``VmodelOI`` 2D
     grid scan, with zero coupling to ``CANDID`` or the injected truth) and
     ``CANDID`` (``fitMap``) independently extract ``(dra, ddec)`` and
     anisotropic astrometric uncertainties at every epoch, each gated by its own
     per-test skip guard.
   * Both tests check ``PositionData.dra`` and ``PositionData.ddec`` directly
     against the externally extracted coordinates and assert exact equality of
     ``PositionData.loglike`` in value (including ``log_det`` and the
     ``-N ln(2 pi)`` constant, evaluated at ``O(1-3 sigma)`` radial and
     tangential offsets) against SciPy's multivariate normal log-density for
     both Cartesian covariance matrices and ``PositionData.from_sep_pa``.
"""

import contextlib
import io

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import optimize, stats

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
PERTURBED_ELEMENTS = (400.0, 35.015, 0.25, 52.05, 65.0, 115.06, 9.508)
T_REF = 60000.0
MJDS = np.linspace(60010.0, 60310.0, 5)
PLATE_SCALE_M = 1.045
NORTH_ANGLE_DEG = 7.25
STAR_DIAM_MAS = 0.8
FLUX_RATIO = 0.04


@pytest.fixture(scope="module")
def noisy_binary_oifits(tmp_path_factory):
    """Five realistic noisy OIFITS epochs of a Keplerian binary in standard
    unrotated, unscaled East/North sky coordinates."""
    base_dir = tmp_path_factory.mktemp("pos_likelihood_noisy_oifits")
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
            sigma_v2=0.002,
            sigma_cp_deg=0.1,
            rng=np.random.default_rng(17 + idx),
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


def _polar_uncertainties_from_cov(x, y, cov_xy):
    """Propagate Cartesian covariance cov_xy to distinct radial (mas) and
    position-angle (deg) uncertainties via the polar Jacobian."""
    sep = float(np.hypot(x, y))
    pa_deg = float(np.rad2deg(np.arctan2(x, y)) % 360.0)
    u_rad = np.array([x / sep, y / sep])
    u_tan = np.array([y / sep**2, -x / sep**2]) * (180.0 / np.pi)
    sep_err = float(np.sqrt(u_rad @ cov_xy @ u_rad))
    pa_err = float(np.sqrt(u_tan @ cov_xy @ u_tan))
    return sep, pa_deg, sep_err, pa_err


def _sep_pa_logpdf(seps, pas, sep_errs, pa_errs, model_dra, model_ddec):
    """Exact Gaussian log-density in (dRA, dDec) induced by (sep_err, pa_err)
    via the exact analytic polar-to-Cartesian Jacobian."""
    deg = np.pi / 180.0
    total = 0.0
    for k in range(len(seps)):
        th = pas[k] * deg
        s_th, c_th = np.sin(th), np.cos(th)
        mean_xy = [seps[k] * s_th, seps[k] * c_th]
        jac = np.array([
            [s_th, seps[k] * c_th * deg],
            [c_th, -seps[k] * s_th * deg],
        ])
        cov = jac @ np.diag([sep_errs[k] ** 2, pa_errs[k] ** 2]) @ jac.T
        total += float(stats.multivariate_normal(mean_xy, cov).logpdf([model_dra[k], model_ddec[k]]))
    return total


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
    headline="abs_dloglike_cov",
)
def test_position_data_likelihood_matches_pmoired_astrometry(noisy_binary_oifits):
    """PositionData reproduces PMOIRED's blindly extracted (dra, ddec) and full
    anisotropic covariances, matching SciPy's multivariate normal log-density
    (including log_det) to < 1e-10 on Cartesian covariances and < 1e-7 on from_sep_pa."""
    pytest.importorskip("pmoired")
    px_list, py_list, covs_xy = [], [], []
    seps, pas, sep_errs, pa_errs = [], [], [], []
    for path in noisy_binary_oifits:
        gx, gy = _blind_pmoired_seed(path)
        res = pm.fit(
            path,
            {"star,ud": 0.8, "star,f": 1.0, "c,ud": 0.0, "c,x": gx, "c,y": gy, "c,f": 0.035},
            free=["star,ud", "c,x", "c,y", "c,f"],
        )
        px, py = res["best"]["c,x"], res["best"]["c,y"]
        cov_xy = np.asarray(res["cov"][1:3, 1:3], dtype=float)
        px_list.append(px)
        py_list.append(py)
        covs_xy.append(cov_xy)
        sep, pa_deg, s_err, p_err = _polar_uncertainties_from_cov(px, py, cov_xy)
        seps.append(sep)
        pas.append(pa_deg)
        sep_errs.append(s_err)
        pa_errs.append(p_err)

    px_arr, py_arr = np.array(px_list), np.array(py_list)
    cov_arr = np.array(covs_xy)
    sep_arr, pa_arr = np.array(seps), np.array(pas)
    # Scale radial uncertainty by 1.8x relative to tangential so the polar-to-Cartesian
    # rotation in PositionData.from_sep_pa is strongly anisotropic as well.
    sep_err_arr = 1.8 * np.array(sep_errs)
    pa_err_arr = np.array(pa_errs)

    dra_true, ddec_true = co.position(MJDS - T_REF, *ELEMENTS)
    dra_pert, ddec_pert = co.position(MJDS - T_REF, *PERTURBED_ELEMENTS)
    max_astrom_err = float(np.max(np.hypot(px_arr - dra_true, py_arr - ddec_true)))
    assert max_astrom_err < 0.03

    pos_cov = vo.PositionData(MJDS, px_arr, py_arr, cov_arr, t_ref=T_REF)
    pos_sep_pa = vo.PositionData.from_sep_pa(MJDS, sep_arr, pa_arr, sep_err_arr, pa_err_arr, t_ref=T_REF)
    np.testing.assert_allclose(np.asarray(pos_sep_pa.dra), px_arr, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(pos_sep_pa.ddec), py_arr, rtol=0.0, atol=1e-12)

    test_orbit = vo.KeplerOrbit(*PERTURBED_ELEMENTS, t_ref=T_REF)
    want_ll_cov = sum(
        float(stats.multivariate_normal([px_arr[k], py_arr[k]], cov_arr[k]).logpdf([dra_pert[k], ddec_pert[k]]))
        for k in range(len(MJDS))
    )
    got_ll_cov = float(pos_cov.loglike(test_orbit))
    dlogl_cov = abs(got_ll_cov - want_ll_cov)

    want_ll_sep_pa = _sep_pa_logpdf(sep_arr, pa_arr, sep_err_arr, pa_err_arr, dra_pert, ddec_pert)
    got_ll_sep_pa = float(pos_sep_pa.loglike(test_orbit))
    dlogl_sep_pa = abs(got_ll_sep_pa - want_ll_sep_pa)

    record("abs_dloglike_cov", dlogl_cov)
    record("abs_dloglike_sep_pa", dlogl_sep_pa)
    record("max_astrom_offset_mas", max_astrom_err)
    assert dlogl_cov < 1e-10
    assert dlogl_sep_pa < 1e-7


@pytest.mark.external
@pytest.mark.validates(
    "virgil.orbits.PositionData",
    roots=["candid"],
    property="likelihood",
    headline="abs_dloglike_cov",
)
def test_position_data_likelihood_matches_candid_astrometry(noisy_binary_oifits):
    """PositionData reproduces CANDID's blindly extracted (dra, ddec) and
    anisotropic curvature uncertainties, matching SciPy's multivariate normal
    log-density (including log_det) to < 1e-10 on Cartesian covariances and < 1e-7 on from_sep_pa."""
    if not candid_bridge.available():
        pytest.skip("CANDID not installed (.venv-candid)")

    cx_list, cy_list, covs_xy = [], [], []
    seps, pas, sep_errs, pa_errs = [], [], [], []
    for path in noisy_binary_oifits:
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
        scale = np.sqrt(res["chi2r"])
        sx = res["uncer"]["x"] / scale
        sy = res["uncer"]["y"] / scale
        cov_xy = np.diag([sx**2, sy**2])
        cx_list.append(cx)
        cy_list.append(cy)
        covs_xy.append(cov_xy)
        sep, pa_deg, s_err, p_err = _polar_uncertainties_from_cov(cx, cy, cov_xy)
        seps.append(sep)
        pas.append(pa_deg)
        sep_errs.append(s_err)
        pa_errs.append(p_err)

    cx_arr, cy_arr = np.array(cx_list), np.array(cy_list)
    cov_arr = np.array(covs_xy)
    sep_arr, pa_arr = np.array(seps), np.array(pas)
    sep_err_arr = 1.8 * np.array(sep_errs)
    pa_err_arr = np.array(pa_errs)

    dra_true, ddec_true = co.position(MJDS - T_REF, *ELEMENTS)
    dra_pert, ddec_pert = co.position(MJDS - T_REF, *PERTURBED_ELEMENTS)
    max_astrom_err = float(np.max(np.hypot(cx_arr - dra_true, cy_arr - ddec_true)))
    assert max_astrom_err < 0.03

    pos_cov = vo.PositionData(MJDS, cx_arr, cy_arr, cov_arr, t_ref=T_REF)
    pos_sep_pa = vo.PositionData.from_sep_pa(MJDS, sep_arr, pa_arr, sep_err_arr, pa_err_arr, t_ref=T_REF)
    np.testing.assert_allclose(np.asarray(pos_sep_pa.dra), cx_arr, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(pos_sep_pa.ddec), cy_arr, rtol=0.0, atol=1e-12)

    test_orbit = vo.KeplerOrbit(*PERTURBED_ELEMENTS, t_ref=T_REF)
    want_ll_cov = sum(
        float(stats.multivariate_normal([cx_arr[k], cy_arr[k]], cov_arr[k]).logpdf([dra_pert[k], ddec_pert[k]]))
        for k in range(len(MJDS))
    )
    got_ll_cov = float(pos_cov.loglike(test_orbit))
    dlogl_cov = abs(got_ll_cov - want_ll_cov)

    want_ll_sep_pa = _sep_pa_logpdf(sep_arr, pa_arr, sep_err_arr, pa_err_arr, dra_pert, ddec_pert)
    got_ll_sep_pa = float(pos_sep_pa.loglike(test_orbit))
    dlogl_sep_pa = abs(got_ll_sep_pa - want_ll_sep_pa)

    record("abs_dloglike_cov", dlogl_cov)
    record("abs_dloglike_sep_pa", dlogl_sep_pa)
    record("max_astrom_offset_mas", max_astrom_err)
    assert dlogl_cov < 1e-10
    assert dlogl_sep_pa < 1e-7
