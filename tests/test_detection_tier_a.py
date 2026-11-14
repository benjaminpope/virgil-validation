"""Tier-A mathematical and statistical validation of virgil.detection.DetectionMC
and virgil.detection.injection_recovery.

* DetectionMC.threshold: empirical (1 - alpha) quantile of the null draws,
  bootstrap uncertainty, RuntimeWarning when alpha * n < 1, and ValueError
  guards on invalid alpha or empty null arrays.
* DetectionMC.false_alarm_probability: Phipson & Smyth (2010) Monte Carlo
  p-value (k + 1) / (n + 1) and exact Clopper-Pearson Beta confidence bounds
  [lower, upper], including boundary ties k = 0 and k = n.
  Control: rejects the uncalibrated plug-in fraction k / n at k = 0.
* DetectionMC.completeness, contrast_curve, roc, auc: exact empirical step
  curves, isotonic log-flux interpolation, Cartesian and angular sky-position
  match_radius filtering, and convergence to closed-form Gaussian-shift ROC
  theory (C(f) = 1 - Phi(Phi^-1(1 - alpha) - f / sigma_s), AUC = Phi(mu / sqrt(2))).
* DetectionMC.concatenate, save, load: lossless chunk merging, .npz round-trip,
  and ValueError guards on duplicate seeds, incompatible metadata, or unknown
  file-format versions.
* injection_recovery: deterministic 7-hole NRM UV template (n_null=24,
  n_inject=16) matching direct realization-by-realization calls to
  gaussian_null and detection_statistics, batch-size invariance, and monotonic
  completeness in companion flux.
"""

import json
import warnings
import zipfile

import jax
import numpy as np
import pytest
from scipy import integrate, stats

from evidence.plugin import record

vm = pytest.importorskip("virgil.models")
det = pytest.importorskip("virgil.detection")
cov = pytest.importorskip("virgil.coverage")

pytestmark = pytest.mark.x64


def _make_synthetic_mc(
    null_vals,
    inj_vals,
    dra,
    ddec,
    flux,
    best_dra=None,
    best_ddec=None,
    best_flux=None,
    match_radius=None,
    seed=0,
):
    null_vals = np.asarray(null_vals, dtype=float)
    inj_vals = np.asarray(inj_vals, dtype=float)
    dra = np.asarray(dra, dtype=float)
    ddec = np.asarray(ddec, dtype=float)
    flux = np.asarray(flux, dtype=float)
    best_dra = dra.copy() if best_dra is None else np.asarray(best_dra, dtype=float)
    best_ddec = ddec.copy() if best_ddec is None else np.asarray(best_ddec, dtype=float)
    best_flux = flux.copy() if best_flux is None else np.asarray(best_flux, dtype=float)
    n0, n1 = null_vals.size, inj_vals.size
    null = {
        "delta_chi2": null_vals,
        "log_bayes_factor": 0.5 * null_vals - 1.2,
        "max_snr": np.sqrt(np.maximum(null_vals, 0.0)),
        "best_dra": np.zeros(n0),
        "best_ddec": np.zeros(n0),
        "best_flux": np.zeros(n0),
        "flux_peak_steps": np.full(n0, 4.0),
        "converged_fraction": np.ones(n0),
    }
    injected = {
        "delta_chi2": inj_vals,
        "log_bayes_factor": 0.5 * inj_vals - 1.2,
        "max_snr": np.sqrt(np.maximum(inj_vals, 0.0)),
        "dra": dra,
        "ddec": ddec,
        "flux": flux,
        "best_dra": best_dra,
        "best_ddec": best_ddec,
        "best_flux": best_flux,
        "flux_peak_steps": np.full(n1, 4.0),
        "converged_fraction": np.ones(n1),
    }
    meta = {
        "format": 1,
        "virgil_version": "0.4.1",
        "params": ["dra", "ddec", "flux"],
        "flux_param": "flux",
        "names": {"dra": "dra", "ddec": "ddec", "flux": "flux"},
        "grid": {
            "dra": sorted(set(dra.tolist())),
            "ddec": sorted(set(ddec.tolist())),
            "flux": sorted(set(flux.tolist())),
        },
        "model": {"type": "virgil.models.BinaryModelCartesian", "hash": "synthetic_model"},
        "null_scene": {"type": "virgil.models.BinaryModelCartesian", "hash": "synthetic_null"},
        "template": {"n_vis": 21, "n_phi": 35, "hash": "synthetic_tpl"},
        "noise": {"kind": "gaussian", "error_scale": 1.0},
        "match_radius": match_radius,
        "n_null": int(n0),
        "n_injected": int(n1),
        "seeds": [{"seed": int(seed)}],
    }
    return det.DetectionMC(null=null, injected=injected, meta=meta)


def _clopper_pearson(k, n, confidence=0.95):
    """Exact Clopper-Pearson binomial confidence interval via Beta quantiles."""
    alpha = 1.0 - confidence
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


@pytest.mark.validates(
    "virgil.detection.DetectionMC",
    roots=["mathematics", "statistics"],
    headline="max_abs_dthreshold",
)
def test_detection_mc_threshold_order_statistic_and_guards():
    """threshold(stat, fap) equals the (1 - fap) empirical quantile with
    bootstrap uncertainty, warns when fap * n < 1, and raises ValueError on
    out-of-domain fap or empty null samples."""
    rng = np.random.default_rng(2026)
    null_draws = stats.chi2(df=2).rvs(size=500, random_state=rng)
    mc = _make_synthetic_mc(
        null_draws,
        inj_vals=[12.0, 20.0],
        dra=[50.0, 50.0],
        ddec=[0.0, 0.0],
        flux=[1e-3, 5e-3],
    )

    worst_thr = 0.0
    for stat_name in ("delta_chi2", "log_bayes_factor", "max_snr"):
        arr = mc.null[stat_name]
        for fap in (0.20, 0.05, 0.01):
            got_thr, got_err = mc.threshold(stat_name, fap, n_boot=400, seed=11)
            want_thr = float(np.quantile(arr, 1.0 - fap))
            worst_thr = max(worst_thr, abs(got_thr - want_thr))
            boot_rng = np.random.default_rng(77)
            boot_q = np.quantile(
                boot_rng.choice(arr, size=(2000, arr.size), replace=True),
                1.0 - fap,
                axis=1,
            )
            want_err = float(np.std(boot_q))
            assert abs(got_err - want_err) < 0.25 * want_err

    record("max_abs_dthreshold", worst_thr)
    assert worst_thr < 1e-13

    with pytest.warns(RuntimeWarning, match="cannot resolve a FAP"):
        thr_tail, err_tail = mc.threshold("delta_chi2", 1e-3, n_boot=64, seed=3)
    assert thr_tail == pytest.approx(float(np.quantile(null_draws, 1.0 - 1e-3)), abs=1e-13)
    assert np.isfinite(err_tail) and err_tail >= 0.0

    for bad_fap in (-0.05, 0.0, 1.0, 1.05):
        with pytest.raises(ValueError, match="fap"):
            mc.threshold("delta_chi2", bad_fap)

    empty = det.DetectionMC(null={"delta_chi2": np.array([])}, injected={}, meta={"seeds": [{"seed": 0}]})
    with pytest.raises(ValueError, match="no null draws"):
        empty.threshold("delta_chi2", 0.05)
    with pytest.raises(ValueError, match="no null draws"):
        empty.false_alarm_probability("delta_chi2", 4.0)


@pytest.mark.validates(
    "virgil.detection.DetectionMC",
    roots=["mathematics", "statistics"],
    headline="max_abs_dfap",
)
def test_detection_mc_phipson_smyth_fap_and_clopper_pearson_bounds():
    """false_alarm_probability returns (k + 1) / (n + 1) (Phipson & Smyth 2010)
    and exact Clopper-Pearson Beta [lower, upper] bounds across interior,
    boundary-tie, k = 0, and k = n values."""
    null_vals = np.array([0.5, 1.2, 2.0, 2.0, 3.4, 5.1, 7.8, 9.0, 11.5, 14.0, 18.2, 25.0])
    n = null_vals.size
    mc = _make_synthetic_mc(
        null_vals,
        inj_vals=[6.0, 30.0],
        dra=[40.0, 40.0],
        ddec=[0.0, 0.0],
        flux=[2e-3, 1e-2],
    )
    probe = np.array([-1.0, 0.5, 2.0, 6.0, 18.2, 25.0, 40.0])

    worst_fap = 0.0
    worst_ci = 0.0
    for conf in (0.95, 0.90):
        fap, lo, hi = mc.false_alarm_probability("delta_chi2", probe, confidence=conf)
        assert fap.shape == lo.shape == hi.shape == probe.shape
        for idx, val in enumerate(probe):
            k = int(np.sum(null_vals >= val))
            want_fap = (k + 1.0) / (n + 1.0)
            want_lo, want_hi = _clopper_pearson(k, n, confidence=conf)
            worst_fap = max(worst_fap, abs(float(fap[idx]) - want_fap))
            worst_ci = max(worst_ci, abs(float(lo[idx]) - want_lo), abs(float(hi[idx]) - want_hi))
            assert 0.0 <= lo[idx] <= hi[idx] <= 1.0

    fap_scalar, lo_s, hi_s = mc.false_alarm_probability("delta_chi2", 6.0, confidence=0.95)
    k6 = int(np.sum(null_vals >= 6.0))
    want_lo6, want_hi6 = _clopper_pearson(k6, n, confidence=0.95)
    assert float(fap_scalar) == pytest.approx((k6 + 1.0) / (n + 1.0), abs=1e-15)
    assert float(lo_s) == pytest.approx(want_lo6, abs=1e-14)
    assert float(hi_s) == pytest.approx(want_hi6, abs=1e-14)

    record("max_abs_dfap", worst_fap)
    record("max_abs_dci", worst_ci)
    assert worst_fap < 1e-15
    assert worst_ci < 1e-14


@pytest.mark.validates(
    "virgil.detection.DetectionMC",
    roots=["statistics"],
    kind="control",
)
def test_detection_mc_fap_rejects_uncalibrated_zero_tail_fraction():
    """Negative control: when an observed statistic exceeds all n finite null
    draws (k = 0), the uncalibrated plug-in estimator k / n = 0 understates the
    Monte Carlo p-value; DetectionMC must return 1 / (n + 1) > 0."""
    n = 49
    mc = _make_synthetic_mc(
        np.linspace(0.1, 10.0, n),
        inj_vals=[20.0],
        dra=[30.0],
        ddec=[0.0],
        flux=[1e-2],
    )
    fap, lo, hi = mc.false_alarm_probability("delta_chi2", 99.0, confidence=0.95)
    uncalibrated_k_over_n = 0.0
    assert float(fap) == pytest.approx(1.0 / (n + 1.0), abs=1e-15)
    assert float(fap) - uncalibrated_k_over_n >= 0.02
    assert float(lo) == 0.0
    assert float(hi) == pytest.approx(1.0 - (0.025) ** (1.0 / n), abs=1e-14)


@pytest.mark.validates(
    "virgil.detection.DetectionMC",
    roots=["mathematics", "statistics"],
    headline="max_abs_dcontrast_rel",
)
def test_detection_mc_gaussian_shift_roc_completeness_and_contrast_curve():
    """Analytic Gaussian-shift experiment: null Z ~ N(0, 1) and injected
    Z ~ N(f / sigma_s, 1) match closed-form completeness
    1 - Phi(Phi^-1(1 - alpha) - f / sigma_s), contrast curve
    sigma_s * (Phi^-1(1 - alpha) - Phi^-1(1 - c)), and ROC AUC Phi(mu / sqrt(2)),
    plus exact match_radius sky-position filtering."""
    rng = np.random.default_rng(314159)
    n_null = 5000
    null_z = rng.standard_normal(n_null)

    seps = np.array([40.0, 90.0])
    sigmas = {40.0: 3.0e-3, 90.0: 1.0e-3}
    flux_grid = np.geomspace(8e-4, 1.4e-2, 12)
    n_per_cell = 900

    dra_list, ddec_list, flux_list, z_list = [], [], [], []
    for sep in seps:
        sigma_s = sigmas[sep]
        for f in flux_grid:
            angles = rng.uniform(0.0, 2.0 * np.pi, n_per_cell)
            dra_list.append(sep * np.sin(angles))
            ddec_list.append(sep * np.cos(angles))
            flux_list.append(np.full(n_per_cell, f))
            z_list.append(f / sigma_s + rng.standard_normal(n_per_cell))

    dra_all = np.concatenate(dra_list)
    ddec_all = np.concatenate(ddec_list)
    flux_all = np.concatenate(flux_list)
    z_all = np.concatenate(z_list)
    mc = _make_synthetic_mc(null_z, z_all, dra_all, ddec_all, flux_all)

    alpha = 0.02
    t_pop = float(stats.norm.isf(alpha))
    sep_bins = np.array([20.0, 65.0, 120.0])
    comp = mc.completeness("delta_chi2", alpha, sep_bins=sep_bins)
    assert comp["completeness"].shape == (2, flux_grid.size)
    assert np.all(comp["n"] == n_per_cell)

    worst_comp = 0.0
    for i, sep in enumerate(seps):
        want_c = stats.norm.sf(t_pop - flux_grid / sigmas[sep])
        worst_comp = max(worst_comp, float(np.max(np.abs(comp["completeness"][i] - want_c))))

    worst_contrast_rel = 0.0
    for target_c in (0.50, 0.80):
        sep_centres, f_lim = mc.contrast_curve(
            "delta_chi2", alpha, completeness=target_c, sep_bins=sep_bins
        )
        np.testing.assert_allclose(sep_centres, [42.5, 92.5], rtol=0, atol=1e-12)
        for i, sep in enumerate(seps):
            want_f = sigmas[sep] * (t_pop - float(stats.norm.isf(target_c)))
            worst_contrast_rel = max(worst_contrast_rel, abs(float(f_lim[i]) / want_f - 1.0))

    # Closed-form ROC AUC for N(mu, 1) vs N(0, 1): Phi(mu / sqrt(2))
    f_mid = float(flux_grid[4])
    mu_40 = f_mid / sigmas[40.0]
    want_auc = float(stats.norm.cdf(mu_40 / np.sqrt(2.0)))
    got_auc = float(mc.auc("delta_chi2", flux=(0.99 * f_mid, 1.01 * f_mid), sep_bin=(20.0, 65.0)))
    fpr, tpr, thr = mc.roc("delta_chi2", flux=(0.99 * f_mid, 1.01 * f_mid), sep_bin=(20.0, 65.0))
    assert fpr[0] == 0.0 and tpr[0] == 0.0 and np.isposinf(thr[0])
    assert fpr[-1] == 1.0 and tpr[-1] == 1.0
    assert np.all(np.diff(fpr) >= 0.0) and np.all(np.diff(tpr) >= 0.0)
    assert got_auc == pytest.approx(float(integrate.trapezoid(tpr, fpr)), abs=1e-14)
    auc_err = abs(got_auc - want_auc)

    # Geometric flux bin centres, NaN for unbracketed completeness, and match_radius
    flux_bins = np.array([1e-3, 4e-3, 1.6e-2])
    binned = mc.completeness("delta_chi2", alpha, sep_bins=sep_bins, flux_bins=flux_bins)
    np.testing.assert_allclose(binned["flux"], [2e-3, 8e-3], rtol=1e-14)

    mc_partial = _make_synthetic_mc(
        np.linspace(0.0, 9.0, 10),
        inj_vals=[1.0, 15.0, 1.0, 15.0, 15.0, 1.0],  # flux=1e-3: 1/3, flux=5e-3: 2/3
        dra=np.full(6, 40.0),
        ddec=np.zeros(6),
        flux=[1e-3, 1e-3, 1e-3, 5e-3, 5e-3, 5e-3],
    )
    _, unbracketed_hi = mc_partial.contrast_curve("delta_chi2", 0.2, completeness=0.90)
    _, unbracketed_lo = mc_partial.contrast_curve("delta_chi2", 0.2, completeness=0.10)
    assert np.isnan(unbracketed_hi[0]) and np.isnan(unbracketed_lo[0])
    _, mid_interp = mc_partial.contrast_curve("delta_chi2", 0.2, completeness=0.50)
    assert float(mid_interp[0]) == pytest.approx(np.sqrt(1e-3 * 5e-3), rel=1e-12)

    # Angular (sep, pa) sky-distance matching under match_radius
    ang_mc = det.DetectionMC(
        null={"delta_chi2": np.linspace(0.0, 9.0, 10)},
        injected={
            "delta_chi2": np.array([20.0, 20.0]),
            "sep": np.array([50.0, 50.0]),
            "pa": np.array([358.0, 90.0]),
            "flux": np.array([1e-2, 1e-2]),
            "best_sep": np.array([50.0, 50.0]),
            "best_pa": np.array([2.0, 120.0]),  # 4 deg wrap (~3.49 mas) vs 30 deg (~25.88 mas)
            "best_flux": np.array([1e-2, 1e-2]),
        },
        meta={
            "format": 1,
            "params": ["sep", "pa", "flux"],
            "flux_param": "flux",
            "names": {"sep": "sep", "pa": "pa", "flux": "flux"},
            "grid": {"sep": [50.0], "pa": [0.0, 90.0, 180.0, 270.0], "flux": [1e-2]},
            "match_radius": 5.0,
            "seeds": [{"seed": 1}],
        },
    )
    assert np.array_equal(ang_mc.matched(), np.array([True, False]))
    assert np.array_equal(ang_mc.detected("delta_chi2", 0.2), np.array([True, False]))

    record("max_abs_dcontrast_rel", worst_contrast_rel)
    record("max_abs_dcompleteness", worst_comp)
    record("abs_dauc", auc_err)
    assert worst_comp < 0.05
    assert worst_contrast_rel < 0.06
    assert auc_err < 0.04


@pytest.mark.validates(
    "virgil.detection.DetectionMC",
    roots=["mathematics"],
    headline="max_abs_roundtrip_diff",
)
def test_detection_mc_concatenate_and_npz_roundtrip(tmp_path):
    """concatenate merges multi-seed chunks losslessly while rejecting repeated
    seeds or mismatched metadata; save/load round-trips every array and rejects
    unknown format versions."""
    mc1 = _make_synthetic_mc(
        [1.0, 2.5, 4.0],
        [8.0, 12.0],
        dra=[40.0, -40.0],
        ddec=[0.0, 30.0],
        flux=[2e-3, 8e-3],
        match_radius=10.0,
        seed=101,
    )
    mc2 = _make_synthetic_mc(
        [0.5, 3.5],
        [5.0, 18.0, 22.0],
        dra=[40.0, -40.0, 40.0],
        ddec=[0.0, 30.0, 0.0],
        flux=[2e-3, 8e-3, 8e-3],
        match_radius=10.0,
        seed=202,
    )
    merged = det.DetectionMC.concatenate([mc1, mc2])
    assert merged.n_null == 5 and merged.n_injected == 5
    assert merged.meta["seeds"] == [{"seed": 101}, {"seed": 202}]

    out_npz = tmp_path / "detection_mc.npz"
    merged.save(out_npz)
    loaded = det.DetectionMC.load(out_npz)

    worst = 0.0
    for k in merged.null:
        worst = max(worst, float(np.max(np.abs(loaded.null[k] - merged.null[k]))))
    for k in merged.injected:
        worst = max(worst, float(np.max(np.abs(loaded.injected[k] - merged.injected[k]))))
    assert loaded.meta == merged.meta

    record("max_abs_roundtrip_diff", worst)
    assert worst == 0.0

    # Duplicate seed must raise ValueError
    with pytest.raises(ValueError, match="seed"):
        det.DetectionMC.concatenate([mc1, mc1])

    # Incompatible match_radius must raise ValueError
    mc_bad = _make_synthetic_mc(
        [1.0, 2.0], [5.0], dra=[40.0], ddec=[0.0], flux=[2e-3], match_radius=25.0, seed=303
    )
    with pytest.raises(ValueError):
        det.DetectionMC.concatenate([mc1, mc_bad])

    # Unknown format version in saved .npz must raise ValueError on load
    bad_npz = tmp_path / "bad_format.npz"
    with zipfile.ZipFile(out_npz, "r") as zin, zipfile.ZipFile(bad_npz, "w") as zout:
        for item in zin.infolist():
            raw = zin.read(item.filename)
            if item.filename == "meta.npy":
                buf = np.load(out_npz, allow_pickle=False)
                meta_bad = json.loads(str(buf["meta"]))
                meta_bad["format"] = 999
                np.savez(bad_npz, **{k: buf[k] for k in buf.files if k != "meta"}, meta=json.dumps(meta_bad))
                break
    with pytest.raises(ValueError, match="format"):
        det.DetectionMC.load(bad_npz)


@pytest.mark.validates(
    "virgil.detection.injection_recovery",
    roots=["mathematics", "statistics"],
    headline="max_rel_ddelta_chi2",
)
def test_injection_recovery_nrm_matches_realization_by_realization_and_monotone_completeness():
    """On a deterministic 7-hole NRM UV template (n_null=24, n_inject=16),
    injection_recovery matches direct realization-by-realization calls to
    gaussian_null and detection_statistics, is invariant to draw_batch, and
    yields monotonic completeness in companion flux."""
    template = cov.nrm_oidata(wavelength_m=4.3e-6, sigma_v2=0.01, sigma_cp_deg=0.5)
    grid = {
        "dra": np.linspace(-150.0, 150.0, 7),
        "ddec": np.linspace(-150.0, 150.0, 7),
        "flux": np.geomspace(5e-4, 1e-1, 12),
    }
    flux_tiers = np.array([5e-4, 4e-3, 2e-2, 8e-2])
    positions = np.array([[100.0, 0.0], [0.0, 100.0], [-100.0, 50.0], [50.0, -100.0]])
    injections = {
        "dra": np.tile(positions[:, 0], flux_tiers.size),
        "ddec": np.tile(positions[:, 1], flux_tiers.size),
        "flux": np.repeat(flux_tiers, positions.shape[0]),
    }

    seed = 20261115
    null_scene = vm.BinaryModelCartesian(0.0, 0.0, 0.0)
    mc = det.injection_recovery(
        vm.BinaryModelCartesian,
        null_scene,
        template,
        grid,
        seed,
        n_null=24,
        injections=injections,
        match_radius=60.0,
        draw_batch=1,
        progress=False,
    )
    mc_batched = det.injection_recovery(
        vm.BinaryModelCartesian,
        null_scene,
        template,
        grid,
        seed,
        n_null=24,
        injections=injections,
        match_radius=60.0,
        draw_batch=4,
        progress=False,
    )
    assert mc.n_null == 24 and mc.n_injected == 16

    sim_fn = det.gaussian_null(null_scene, template)
    root_key = jax.random.key(seed)
    worst_dc2, worst_logb, worst_snr, worst_batch = 0.0, 0.0, 0.0, 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(24):
            ki = jax.random.fold_in(jax.random.fold_in(root_key, 0), i)
            st = det.detection_statistics(vm.BinaryModelCartesian, sim_fn(ki), grid)
            ref_dc2 = float(st["delta_chi2"])
            ref_lb = float(st["log_bayes_factor"])
            ref_snr = float(st["max_snr"])
            worst_dc2 = max(worst_dc2, abs(float(mc.null["delta_chi2"][i]) - ref_dc2) / max(1.0, abs(ref_dc2)))
            worst_logb = max(worst_logb, abs(float(mc.null["log_bayes_factor"][i]) - ref_lb) / max(1.0, abs(ref_lb)))
            worst_snr = max(worst_snr, abs(float(mc.null["max_snr"][i]) - ref_snr) / max(1.0, abs(ref_snr)))
            worst_batch = max(
                worst_batch,
                abs(float(mc.null["delta_chi2"][i]) - float(mc_batched.null["delta_chi2"][i])) / max(1.0, abs(ref_dc2)),
            )

        for i in range(16):
            ki = jax.random.fold_in(jax.random.fold_in(root_key, 1), i)
            scene_i = vm.BinaryModelCartesian(
                float(injections["dra"][i]),
                float(injections["ddec"][i]),
                float(injections["flux"][i]),
            )
            st = det.detection_statistics(vm.BinaryModelCartesian, sim_fn(ki, scene=scene_i), grid)
            ref_dc2 = float(st["delta_chi2"])
            ref_lb = float(st["log_bayes_factor"])
            ref_snr = float(st["max_snr"])
            worst_dc2 = max(worst_dc2, abs(float(mc.injected["delta_chi2"][i]) - ref_dc2) / max(1.0, abs(ref_dc2)))
            worst_logb = max(worst_logb, abs(float(mc.injected["log_bayes_factor"][i]) - ref_lb) / max(1.0, abs(ref_lb)))
            worst_snr = max(worst_snr, abs(float(mc.injected["max_snr"][i]) - ref_snr) / max(1.0, abs(ref_snr)))
            worst_batch = max(
                worst_batch,
                abs(float(mc.injected["delta_chi2"][i]) - float(mc_batched.injected["delta_chi2"][i])) / max(1.0, abs(ref_dc2)),
            )

    # Monotonic completeness across flux tiers (pooling the 4 positions into one radial bin)
    comp = mc.completeness("delta_chi2", fap=0.10, sep_bins=np.array([50.0, 150.0]))
    det_frac = comp["completeness"][0]
    assert det_frac.shape == (4,)
    assert np.all(np.diff(det_frac) >= -1e-12), det_frac
    assert det_frac[0] <= 0.25 and det_frac[-1] == 1.0

    record("max_rel_ddelta_chi2", worst_dc2)
    record("max_rel_dlogB", worst_logb)
    record("max_rel_dsnr", worst_snr)
    record("max_rel_dbatch", worst_batch)
    assert worst_dc2 < 1e-6
    assert worst_logb < 1e-6
    assert worst_snr < 1e-5
    assert worst_batch < 1e-6


@pytest.mark.validates(
    "virgil.detection.injection_recovery",
    roots=["mathematics"],
    kind="control",
)
def test_injection_recovery_rejects_off_grid_false_position_matches():
    """Negative control: a bright companion injected far outside the searched
    coordinate box produces alias peaks inside the box whose delta_chi2 exceeds
    the null threshold, so ignoring match_radius falsely counts them as
    recoveries while position-matched recovery rejects 100% of them."""
    template = cov.nrm_oidata(wavelength_m=4.3e-6, sigma_v2=0.005, sigma_cp_deg=0.25)
    grid = {
        "dra": np.linspace(-60.0, 60.0, 5),
        "ddec": np.linspace(-60.0, 60.0, 5),
        "flux": np.geomspace(1e-3, 1e-1, 8),
    }
    injections = {
        "dra": np.array([240.0, -240.0, 0.0, 200.0]),
        "ddec": np.array([0.0, 180.0, -240.0, -200.0]),
        "flux": np.full(4, 0.08),
    }
    mc = det.injection_recovery(
        vm.BinaryModelCartesian,
        vm.BinaryModelCartesian(0.0, 0.0, 0.0),
        template,
        grid,
        7,
        n_null=12,
        injections=injections,
        match_radius=30.0,
        progress=False,
    )
    thr, _ = mc.threshold("delta_chi2", 0.15)
    unmatched_exceedance = np.mean(mc.injected["delta_chi2"] > thr)
    matched_completeness = float(mc.completeness("delta_chi2", 0.15, sep_bins=[150.0, 350.0])["completeness"][0, 0])
    assert unmatched_exceedance >= 0.75
    assert matched_completeness == 0.0
