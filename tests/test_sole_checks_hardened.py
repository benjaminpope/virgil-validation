"""Sole-check hardening for ModulatedGaussianRim, simulate.simulate, and
detection.gaussian_null.

* ``ModulatedGaussianRim``: in the thin-rim regime where the O(m^2 sigma^2 / r0^2)
  definition difference (ledger D2) between virgil's blurred modulated ring
  and PMOIRED's modulated radial profile vanishes (< 1e-6), the two packages
  must agree in closure phase to ``dcp < 1e-4 deg`` on strong (> 15 deg)
  closure phases.
* ``simulate.simulate`` and ``detection.gaussian_null``: standardized
  closure-phase (``phi``) pulls are checked on BOTH a 3-telescope template
  (uncorrelated scalar fallback, ``cp_noise is None``) AND a 4-telescope
  template (``cp_noise is not None``, 4 closure triangles per snapshot/channel
  with rank-3 covariance and shared-baseline correlation ``+-1/3``), verifying
  marginal zero mean, unit variance under ``noise_scale`` / ``error_scale``,
  Kolmogorov-Smirnov normality, and the exact ``+-1/3`` shared-baseline
  triangle correlation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from crosscheck import simulate as csim
from crosscheck import sky
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
    """4-station VLTI template (4 closure triangles per snapshot/channel) that
    exercises virgil's correlated ClosureNoise.sample path."""
    path = tmp_path_factory.mktemp("sole_checks_4t") / "uts4.fits"
    csim.observe(
        path,
        lambda u, v, w: (sky.vis_point(u, v, w) + 0.05 * sky.vis_point(u, v, w, 4.0, -3.0)) / 1.05,
        UTS4,
        hour_angles_h=np.linspace(-2.0, 2.0, 5),
        wavelengths=np.linspace(1.6e-6, 2.2e-6, 5),
        dec_deg=-40.0,
        sigma_v2=0.01,
        sigma_cp_deg=0.5,
    )
    data = OIData(str(path))
    assert data.cp_noise is not None
    return path, data


@pytest.mark.external
@pytest.mark.validates(
    "virgil.models.ModulatedGaussianRim",
    roots=["pmoired"],
    property="modulation_sense",
    headline="max_abs_dCP_deg",
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


@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.validates(
    "virgil.simulate.simulate",
    roots=["mathematics", "statistics"],
    headline="abs_cp_pull_mean",
)
def test_simulate_closure_phase_pulls_are_standard_normal(three_tel_template, scale):
    """Standardized closure-phase residuals (phi - phi_clean) / (scale * d_phi)
    from simulate.simulate on 3-station data have zero mean and unit variance."""
    _, data = three_tel_template
    scene = vm.BinaryModelCartesian(4.0, -3.0, 0.05)
    clean = vsim.simulate(scene, data)
    d_phi = np.asarray(data.d_phi)
    pulls = np.concatenate([
        (
            np.angle(
                np.exp(
                    1j
                    * (
                        np.asarray(vsim.simulate(scene, data, key=jax.random.key(k), noise_scale=scale).phi)
                        - np.asarray(clean.phi)
                    )
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


@pytest.mark.parametrize("scale", [1.0, 2.0])
@pytest.mark.validates(
    "virgil.simulate.simulate",
    roots=["mathematics", "statistics"],
    headline="max_offdiag_corr_error",
)
def test_simulate_correlated_four_tel_closure_phase_pulls_and_covariance(four_tel_template, scale):
    """On a 4-station array (cp_noise is not None, 4 triangles per snapshot and
    channel), simulate.simulate produces standardized closure-phase pulls with
    unit marginal variance and shared-baseline off-diagonal correlation |r| = 1/3."""
    _, data = four_tel_template
    assert data.cp_noise is not None
    scene = vm.BinaryModelCartesian(4.0, -3.0, 0.05)
    clean_phi = np.asarray(vsim.simulate(scene, data).phi)
    d_phi = np.asarray(data.d_phi)
    # 80 realizations x 5 HAs x 5 channels = 2000 independent 4-triangle groups
    draws = np.stack([
        np.angle(
            np.exp(
                1j
                * (
                    np.asarray(vsim.simulate(scene, data, key=jax.random.key(500 + k), noise_scale=scale).phi)
                    - clean_phi
                )
            )
        )
        / (scale * d_phi)
        for k in range(80)
    ], axis=0)
    pulls = draws.ravel()
    mean_err = float(abs(np.mean(pulls)))
    std_err = float(abs(np.std(pulls) - 1.0))
    # Marginal normality over one non-redundant triangle per snapshot/channel
    indep_pulls = draws.reshape(80, 5, 4, 5)[:, :, 0, :].ravel()
    ks_p = float(stats.kstest(indep_pulls, "norm").pvalue)

    quads = draws.reshape(80, 5, 4, 5).transpose(0, 1, 3, 2).reshape(-1, 4)
    corr = np.corrcoef(quads, rowvar=False)
    off_diag = np.abs(corr[~np.eye(4, dtype=bool)])
    corr_err = float(np.max(np.abs(off_diag - 1.0 / 3.0)))

    record("max_offdiag_corr_error", corr_err)
    record("abs_cp_pull_mean", mean_err)
    record("abs_cp_pull_std_minus_one", std_err)
    assert mean_err < 4.5 / np.sqrt(indep_pulls.size)
    assert std_err < 4.5 / np.sqrt(2.0 * indep_pulls.size)
    assert ks_p > 1e-3
    assert corr_err < 0.08


@pytest.mark.parametrize("scale", [1.0, 1.5])
@pytest.mark.validates(
    "virgil.detection.gaussian_null",
    roots=["statistics"],
    headline="abs_cp_pull_mean",
)
def test_gaussian_null_closure_phase_pulls_are_standard_normal(three_tel_template, scale):
    """Standardized closure-phase pulls from detection.gaussian_null on 3-station
    data have zero mean and unit variance under error_scale."""
    _, data = three_tel_template
    null_scene = vm.BinaryModelCartesian(0.0, 0.0, 0.0)
    clean_phi = np.asarray(vsim.simulate(null_scene, data).phi)
    sim_null = vdet.gaussian_null(null_scene, data, error_scale=scale)
    d_phi = np.asarray(data.d_phi)
    pulls = np.concatenate([
        (
            np.angle(np.exp(1j * (np.asarray(sim_null(jax.random.key(200 + k)).phi) - clean_phi)))
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


@pytest.mark.parametrize("scale", [1.0, 1.5])
@pytest.mark.validates(
    "virgil.detection.gaussian_null",
    roots=["mathematics", "statistics"],
    headline="max_offdiag_corr_error",
)
def test_gaussian_null_correlated_four_tel_closure_phase_pulls_and_covariance(four_tel_template, scale):
    """On a 4-station array (cp_noise is not None), detection.gaussian_null
    draws correlated closure phases with unit standardized marginal variance and
    shared-baseline off-diagonal correlation |r| = 1/3."""
    _, data = four_tel_template
    assert data.cp_noise is not None
    null_scene = vm.BinaryModelCartesian(0.0, 0.0, 0.0)
    clean_phi = np.asarray(vsim.simulate(null_scene, data).phi)
    sim_null = vdet.gaussian_null(null_scene, data, error_scale=scale)
    d_phi = np.asarray(data.d_phi)
    draws = np.stack([
        np.angle(np.exp(1j * (np.asarray(sim_null(jax.random.key(900 + k)).phi) - clean_phi)))
        / (scale * d_phi)
        for k in range(80)
    ], axis=0)
    pulls = draws.ravel()
    mean_err = float(abs(np.mean(pulls)))
    std_err = float(abs(np.std(pulls) - 1.0))
    indep_pulls = draws.reshape(80, 5, 4, 5)[:, :, 0, :].ravel()
    ks_p = float(stats.kstest(indep_pulls, "norm").pvalue)

    quads = draws.reshape(80, 5, 4, 5).transpose(0, 1, 3, 2).reshape(-1, 4)
    corr = np.corrcoef(quads, rowvar=False)
    off_diag = np.abs(corr[~np.eye(4, dtype=bool)])
    corr_err = float(np.max(np.abs(off_diag - 1.0 / 3.0)))

    record("max_offdiag_corr_error", corr_err)
    record("abs_cp_pull_mean", mean_err)
    record("abs_cp_pull_std_minus_one", std_err)
    assert mean_err < 4.5 / np.sqrt(indep_pulls.size)
    assert std_err < 4.5 / np.sqrt(2.0 * indep_pulls.size)
    assert ks_p > 1e-3
    assert corr_err < 0.08
