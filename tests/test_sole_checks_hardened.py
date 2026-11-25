"""Sole-check hardening for ModulatedGaussianRim, simulate.simulate, and
detection.gaussian_null.

* ``ModulatedGaussianRim``: in the thin-rim regime where the O(m^2 sigma^2 / r0^2)
  definition difference (ledger D2) between virgil's blurred modulated ring
  and PMOIRED's modulated radial profile vanishes (< 1e-6), the two packages
  must agree in closure phase to ``dcp < 1e-4 deg`` on strong (> 15 deg)
  closure phases.
* ``simulate.simulate`` and ``detection.gaussian_null``: standardized
  closure-phase (``phi``) pulls centred against independent ``crosscheck.chi2``
  + ``crosscheck.sky`` noiseless closure phases of asymmetric binaries (and
  exercising ``sim_null(key, scene=...)``) are checked on BOTH a 3-telescope
  template (uncorrelated scalar fallback, ``cp_noise is None``) AND a
  4-telescope template with heterogeneous ``T3PHIERR`` and distinct snapshot vs
  channel dimensions (``5`` snapshots x ``4`` triangles x ``4`` channels,
  ``cp_noise is not None``), verifying marginal zero mean and unit variance at
  effective sample size ``3 * G``, Kolmogorov-Smirnov normality, the signed
  ``4 x 4`` correlation matrix ``(T @ T.T) / 3``, heterogeneous non-vanishing of
  unstandardized closure-of-closures vs vanishing on standardized pulls and
  paired noise differences, and the chi-squared(df=3) Moore-Penrose quadratic form.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.io import fits
from scipy import stats

from crosscheck import chi2, sky
from crosscheck import simulate as csim
from evidence.plugin import record
from external_bridge import pmoired_models as pm

vm = pytest.importorskip("virgil.models")
vsim = pytest.importorskip("virgil.simulate")
vdet = pytest.importorskip("virgil.detection")
OIData = pytest.importorskip("virgil.oidata").OIData

pytestmark = pytest.mark.x64

UTS3 = np.array([[-9.925, -20.335], [14.887, 30.502], [103.306, 43.999]])
UTS4 = np.array([
    [-9.925, -20.335],
    [14.887, 30.502],
    [44.915, 66.183],
    [103.306, 43.999],
])
S2F = 2.0 * np.sqrt(2.0 * np.log(2.0))
UTS4_TRIS = [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]
CLOSURE_NULL_VEC = np.array([1.0, -1.0, 1.0, -1.0])
SCENE_A_ARGS = (4.0, -3.0, 0.05)
SCENE_B_ARGS = (-3.5, 4.5, 0.06)


def _crosscheck_closure_phases_rad(path, dra_mas, ddec_mas, flux):
    """Independent noiseless closure phases (radians, flattened in OIFITS
    row-by-channel order) computed via crosscheck.chi2.load + crosscheck.sky."""
    d = chi2.load(path)
    vis_fn = lambda u, v, w: (
        sky.vis_point(u, v, w) + flux * sky.vis_point(u, v, w, dra_mas, ddec_mas)
    ) / (1.0 + flux)
    t1 = vis_fn(d["u1"], d["v1"], d["wl3"])
    t2 = vis_fn(d["u2"], d["v2_"], d["wl3"])
    t3 = vis_fn(d["u1"] + d["u2"], d["v1"] + d["v2_"], d["wl3"])
    return np.angle(t1 * t2 * np.conj(t3)).ravel()


@pytest.fixture(scope="module")
def three_tel_template(tmp_path_factory):
    path = tmp_path_factory.mktemp("sole_checks_3t") / "uts3.fits"
    csim.observe(
        path,
        lambda u, v, w: (sky.vis_point(u, v, w) + 0.05 * sky.vis_point(u, v, w, 4.0, -3.0)) / 1.05,
        UTS3,
        hour_angles_h=np.linspace(-2.5, 2.5, 5),
        wavelengths=np.linspace(1.6e-6, 2.2e-6, 6),
        dec_deg=-40.0,
        sigma_v2=0.01,
        sigma_cp_deg=0.5,
    )
    return path, OIData(str(path))


@pytest.fixture(scope="module")
def four_tel_template(tmp_path_factory):
    """4-station VLTI template (5 snapshots x 4 closure triangles x 4 channels)
    with heterogeneous T3PHIERR across triangles and channels."""
    path = tmp_path_factory.mktemp("sole_checks_4t") / "uts4.fits"
    csim.observe(
        path,
        lambda u, v, w: (sky.vis_point(u, v, w) + 0.05 * sky.vis_point(u, v, w, 4.0, -3.0)) / 1.05,
        UTS4,
        hour_angles_h=np.linspace(-2.0, 2.0, 5),
        wavelengths=np.linspace(1.6e-6, 2.2e-6, 4),
        dec_deg=-40.0,
        sigma_v2=0.01,
        sigma_cp_deg=0.5,
    )
    rng = np.random.default_rng(123)
    with fits.open(path, mode="update") as hdul:
        err = hdul["OI_T3"].data["T3PHIERR"]
        hdul["OI_T3"].data["T3PHIERR"] = err * rng.uniform(0.5, 2.0, err.shape)
    data = OIData(str(path))
    assert data.cp_noise is not None
    yield path, data
    jax.clear_caches()


def _check_four_tel_correlated_quads(raw_diffs, draws):
    """Verify marginal moments (at exact effective sample size 3 * G), signed
    4 x 4 correlation matrix (T @ T.T) / 3, heterogeneous raw non-vanishing vs
    standardized closure-of-closures vanishing, and chi2(df=3) quadratic form."""
    pulls = draws.ravel()
    mean_err = float(abs(np.mean(pulls)))
    std_err = float(abs(np.std(pulls) - 1.0))

    indep_pulls = draws.reshape(80, 5, 4, 4)[:, :, 0, :].ravel()
    n_eff = 3 * indep_pulls.size
    ks_p = float(stats.kstest(indep_pulls, "norm").pvalue)

    raw_quads = raw_diffs.reshape(80, 5, 4, 4).transpose(0, 1, 3, 2).reshape(-1, 4)
    quads = draws.reshape(80, 5, 4, 4).transpose(0, 1, 3, 2).reshape(-1, 4)
    diff_quads = (quads[0::2] - quads[1::2]) / np.sqrt(2.0)

    t_mat = chi2.triangle_matrix(UTS4_TRIS)
    want_corr = (t_mat @ t_mat.T) / 3.0
    corr = np.corrcoef(quads, rowvar=False)
    corr_err = float(np.max(np.abs(corr - want_corr)))

    raw_null_res = float(np.max(np.abs(raw_quads @ CLOSURE_NULL_VEC)))
    std_null_res = float(np.max(np.abs(quads @ CLOSURE_NULL_VEC)))
    diff_null_res = float(np.max(np.abs(diff_quads @ CLOSURE_NULL_VEC)))
    q_mp = np.einsum("ni,ij,nj->n", quads, np.linalg.pinv(want_corr), quads)
    chi2_3_p = float(stats.kstest(q_mp, stats.chi2(3).cdf).pvalue)
    chi2_4_p = float(stats.kstest(q_mp, stats.chi2(4).cdf).pvalue)

    record("max_signed_corr_error", corr_err)
    record("abs_cp_pull_mean", mean_err)
    record("abs_cp_pull_std_minus_one", std_err)
    record("max_closure_null_residual", diff_null_res)
    record("mean_chi2_per_dof", float(np.mean(q_mp) / 3.0))

    assert mean_err < 4.5 / np.sqrt(n_eff)
    assert std_err < 4.5 / np.sqrt(2.0 * n_eff)
    assert ks_p > 1e-3
    assert corr_err < 0.08
    # Heterogeneous T3PHIERR breaks unstandardized cancellation, while standardized
    # pulls cancel to OIFITS float32 table precision (< 1e-4) and paired noise
    # differences cancel to machine precision (< 1e-12).
    assert raw_null_res > 0.01
    assert std_null_res < 1e-4
    assert diff_null_res < 1e-12
    assert chi2_3_p > 1e-3
    assert chi2_4_p < 1e-3


@pytest.mark.slow
@pytest.mark.external
@pytest.mark.validates(
    "virgil.models.ModulatedGaussianRim",
    roots=["pmoired"],
    property="modulation_sense",
    headline="max_abs_dCP_deg",
    tier="B",
)
def test_thin_modulated_gaussian_rim_matches_pmoired_closure_phase_to_1e_4(three_tel_template):
    """When fwhm << diam so (m sigma / r0)^2 < 1e-5 inside the first visibility
    lobe, ModulatedGaussianRim and PMOIRED's modulated blurred-ring profile
    (evaluated with the overflow-free DLMF 10.40.1 scaled Hankel expansion of
    exp(-z) I_0(z) at z = R r_0 / sigma^2 > 4e5) agree to < 1e-4 deg in closure phase."""
    pytest.importorskip("pmoired")
    path, _ = three_tel_template
    diam, fwhm, inc, pa = 3.2, 0.001, 45.0, 35.0
    amps, pas = (0.55, 0.25), (120.0, 75.0)
    r0, sig = diam / 2.0, fwhm / S2F
    z_expr = f"($R*{r0}/{sig}**2)"
    prof = (
        f"np.exp(-($R-{r0})**2/(2*{sig}**2))/np.sqrt(2*np.pi*{z_expr})"
        f"*(1 + 1/(8*{z_expr}) + 9/(128*{z_expr}**2))"
    )
    params = {
        "diamin": diam - 12.0 * sig,
        "diamout": diam + 12.0 * sig,
        "profile": prof,
        "incl": inc,
        "projang": pa,
    }
    for k, (a, p) in enumerate(zip(amps, pas), start=1):
        params[f"az amp{k}"] = a
        params[f"az projang{k}"] = p - pa

    samples = pm.model_samples(path, params, setup={"Nr": 3000})
    rim = vm.ModulatedGaussianRim(diam, fwhm, inc, pa, np.array(amps), np.array(pas))
    vis = np.asarray(rim.model(jnp.asarray(samples["u"]), jnp.asarray(samples["v"]), 1e-6))
    t1 = np.asarray(rim.model(jnp.asarray(samples["u1"]), jnp.asarray(samples["v1"]), 1e-6))
    t2 = np.asarray(rim.model(jnp.asarray(samples["u2"]), jnp.asarray(samples["v2_"]), 1e-6))
    t3 = np.asarray(
        rim.model(
            jnp.asarray(samples["u1"] + samples["u2"]),
            jnp.asarray(samples["v1"] + samples["v2_"]),
            1e-6,
        )
    )
    cp_vir = np.rad2deg(np.angle(t1 * t2 * np.conj(t3)))
    dv2 = float(np.max(np.abs(np.abs(vis) ** 2 - samples["v2"])))
    dcp = float(np.max(np.abs((cp_vir - samples["t3phi"] + 180.0) % 360.0 - 180.0)))
    cp_max = float(np.max(np.abs(samples["t3phi"])))
    record("max_abs_dV2", dv2)
    record("max_abs_dCP_deg", dcp)
    record("max_abs_CP_deg", cp_max)
    assert cp_max > 15.0
    assert dv2 < 1e-5
    assert dcp < 1e-4


@pytest.mark.slow
@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.validates(
    "virgil.simulate.simulate",
    roots=["mathematics", "statistics"],
    headline="abs_cp_pull_mean",
    tier="B",
)
def test_simulate_closure_phase_pulls_are_standard_normal(three_tel_template, scale):
    """Standardized closure-phase residuals centred against independent
    crosscheck noiseless closure phases on 3-station data have zero mean and
    unit variance."""
    path, data = three_tel_template
    scene = vm.BinaryModelCartesian(*SCENE_A_ARGS)
    ref_phi = _crosscheck_closure_phases_rad(path, *SCENE_A_ARGS)
    assert float(np.rad2deg(np.max(np.abs(ref_phi)))) > 5.0
    d_phi = np.asarray(data.d_phi)
    pulls = np.concatenate([
        (
            np.angle(
                np.exp(
                    1j * (np.asarray(vsim.simulate(scene, data, key=jax.random.key(k), noise_scale=scale).phi) - ref_phi)
                )
            )
            / (scale * d_phi)
        ).ravel()
        for k in range(60)
    ])
    mean_err = float(abs(np.mean(pulls)))
    std_err = float(abs(np.std(pulls) - 1.0))
    ks_p = float(stats.kstest(pulls, "norm").pvalue)
    record("abs_cp_pull_mean", mean_err)
    record("abs_cp_pull_std_minus_one", std_err)
    assert mean_err < 4.0 / np.sqrt(pulls.size)
    assert std_err < 4.0 / np.sqrt(2.0 * pulls.size)
    assert ks_p > 1e-3


@pytest.mark.slow
@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.validates(
    "virgil.simulate.simulate",
    roots=["mathematics", "statistics"],
    headline="max_signed_corr_error",
    tier="B",
)
def test_simulate_correlated_four_tel_closure_phase_pulls_and_covariance(four_tel_template, scale):
    """On a 4-station array with heterogeneous T3PHIERR and 5 snapshots x 4
    channels, simulate.simulate produces standardized closure-phase pulls
    centred on crosscheck's noiseless phases matching the signed rank-3
    correlation matrix (T @ T.T) / 3, closure-of-closures null space, and chi2(df=3)."""
    path, data = four_tel_template
    assert data.cp_noise is not None
    scene = vm.BinaryModelCartesian(*SCENE_A_ARGS)
    ref_phi = _crosscheck_closure_phases_rad(path, *SCENE_A_ARGS)
    d_phi = np.asarray(data.d_phi)
    raw_diffs = np.stack([
        np.angle(
            np.exp(
                1j * (np.asarray(vsim.simulate(scene, data, key=jax.random.key(500 + k), noise_scale=scale).phi) - ref_phi)
            )
        )
        for k in range(80)
    ], axis=0)
    draws = raw_diffs / (scale * d_phi)
    _check_four_tel_correlated_quads(raw_diffs, draws)


@pytest.mark.parametrize("scale", [1.0, 1.5])
@pytest.mark.validates(
    "virgil.detection.gaussian_null",
    roots=["mathematics", "statistics"],
    headline="abs_cp_pull_mean",
)
def test_gaussian_null_closure_phase_pulls_are_standard_normal(three_tel_template, scale):
    """Standardized closure-phase pulls from detection.gaussian_null around an
    asymmetric non-zero-CP null_scene (and under sim_null(key, scene=...))
    match independent crosscheck closure phases with zero mean and unit variance."""
    path, data = three_tel_template
    null_scene = vm.BinaryModelCartesian(*SCENE_A_ARGS)
    alt_scene = vm.BinaryModelCartesian(*SCENE_B_ARGS)
    ref_phi = _crosscheck_closure_phases_rad(path, *SCENE_A_ARGS)
    ref_alt_phi = _crosscheck_closure_phases_rad(path, *SCENE_B_ARGS)
    assert float(np.rad2deg(np.max(np.abs(ref_phi)))) > 5.0

    sim_null = vdet.gaussian_null(null_scene, data, error_scale=scale)
    d_phi = np.asarray(data.d_phi)
    noisy_phis = [np.asarray(sim_null(jax.random.key(200 + k)).phi) for k in range(60)]
    pulls = np.concatenate([
        (np.angle(np.exp(1j * (phi - ref_phi))) / (scale * d_phi)).ravel()
        for phi in noisy_phis
    ])
    # Ignoring null_scene's closure phases (centring on 0) must fail decisively
    uncentred_pulls = np.concatenate([(np.angle(np.exp(1j * phi)) / (scale * d_phi)).ravel() for phi in noisy_phis])
    assert abs(float(np.mean(uncentred_pulls))) > 15.0 / np.sqrt(pulls.size)

    # Also verify the documented scene=... override on sim_null(key, scene=alt_scene)
    alt_pulls = np.concatenate([
        (
            np.angle(np.exp(1j * (np.asarray(sim_null(jax.random.key(300 + k), scene=alt_scene).phi) - ref_alt_phi)))
            / (scale * d_phi)
        ).ravel()
        for k in range(30)
    ])
    assert abs(float(np.mean(alt_pulls))) < 4.0 / np.sqrt(alt_pulls.size)
    assert abs(float(np.std(alt_pulls) - 1.0)) < 4.0 / np.sqrt(2.0 * alt_pulls.size)

    mean_err = float(abs(np.mean(pulls)))
    std_err = float(abs(np.std(pulls) - 1.0))
    ks_p = float(stats.kstest(pulls, "norm").pvalue)
    record("abs_cp_pull_mean", mean_err)
    record("abs_cp_pull_std_minus_one", std_err)
    assert mean_err < 4.0 / np.sqrt(pulls.size)
    assert std_err < 4.0 / np.sqrt(2.0 * pulls.size)
    assert ks_p > 1e-3


@pytest.mark.parametrize("scale", [1.0, 1.5])
@pytest.mark.validates(
    "virgil.detection.gaussian_null",
    roots=["mathematics", "statistics"],
    headline="max_signed_corr_error",
)
def test_gaussian_null_correlated_four_tel_closure_phase_pulls_and_covariance(four_tel_template, scale):
    """On a 4-station array with heterogeneous T3PHIERR and 5 snapshots x 4
    channels, detection.gaussian_null around an asymmetric null_scene (and with
    scene=alt_scene) draws correlated closure phases matching crosscheck's mean,
    signed rank-3 correlation (T @ T.T) / 3, null space, and chi2(df=3)."""
    path, data = four_tel_template
    assert data.cp_noise is not None
    null_scene = vm.BinaryModelCartesian(*SCENE_A_ARGS)
    alt_scene = vm.BinaryModelCartesian(*SCENE_B_ARGS)
    ref_phi = _crosscheck_closure_phases_rad(path, *SCENE_A_ARGS)
    ref_alt_phi = _crosscheck_closure_phases_rad(path, *SCENE_B_ARGS)

    sim_null = vdet.gaussian_null(null_scene, data, error_scale=scale)
    d_phi = np.asarray(data.d_phi)
    raw_diffs = np.stack([
        np.angle(np.exp(1j * (np.asarray(sim_null(jax.random.key(900 + k)).phi) - ref_phi)))
        for k in range(80)
    ], axis=0)
    draws = raw_diffs / (scale * d_phi)
    _check_four_tel_correlated_quads(raw_diffs, draws)

    # Check scene=alt_scene override on the correlated 4-station path as well
    alt_raw = np.angle(np.exp(1j * (np.asarray(sim_null(jax.random.key(999), scene=alt_scene).phi) - ref_alt_phi)))
    alt_quads = (alt_raw / (scale * d_phi)).reshape(5, 4, 4).transpose(0, 2, 1).reshape(-1, 4)
    assert float(np.max(np.abs(alt_quads @ CLOSURE_NULL_VEC))) < 1e-4
