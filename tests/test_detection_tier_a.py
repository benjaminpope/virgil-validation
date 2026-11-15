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
  curves, cumulative-maximum (non-decreasing) log-flux interpolation, Cartesian
  and angular sky-position match_radius filtering, and convergence to
  closed-form Gaussian-shift ROC theory
  (C(f) = 1 - Phi(Phi^-1(1 - alpha) - f / sigma_s), AUC = Phi(mu / sqrt(2))).
* DetectionMC.concatenate, save, load: lossless chunk merging, .npz round-trip,
  and ValueError guards on duplicate seeds, incompatible metadata, or unknown
  file-format versions.
* injection_recovery: noiseless and noisy (bootstrap_null sign-flip)
  3-telescope VLTI datasets generated via crosscheck.simulate and searched
  independently via crosscheck.chi2 + crosscheck.sky across all six outputs
  (delta_chi2, log_bayes_factor, max_snr, best_dra, best_ddec, best_flux),
  noisy 7-hole NRM batch/chunk invariance and draw distinctness, and
  position-matched rejection of off-grid interferometric aliases.
"""

import itertools
import json

from astropy.io import fits
import numpy as np
import pytest
from scipy import integrate, optimize, special, stats

from crosscheck import chi2 as ours, simulate, sky
from evidence.plugin import record

vm = pytest.importorskip("virgil.models")
det = pytest.importorskip("virgil.detection")
cov = pytest.importorskip("virgil.coverage")
from virgil.oidata import OIData  # noqa: E402

pytestmark = pytest.mark.x64

UTS3 = np.array([[-9.925, -20.335], [14.887, 30.502], [103.306, 43.999]])


def _binary_vis(f, x, y):
    def vis(u, v, w):
        return (sky.vis_point(u, v, w) + f * sky.vis_point(u, v, w, x, y)) / (1.0 + f)

    return vis


def _trapezoid_log_weights(n):
    w = np.ones(n)
    w[[0, -1]] = 0.5
    return np.log(w / w.sum())


def _our_grid_search(d, grid):
    """Independent companion search over grid using crosscheck.chi2 and SciPy."""
    null = ours.chi2(d, _binary_vis(0.0, 0.0, 0.0))
    xs, ys, fs = grid["dra"], grid["ddec"], grid["flux"]
    best_dc2, best_snr = 0.0, -np.inf
    best_x, best_y, best_f = float(xs[0]), float(ys[0]), 0.0
    lw = (
        _trapezoid_log_weights(xs.size)[:, None, None]
        + _trapezoid_log_weights(ys.size)[None, :, None]
        + _trapezoid_log_weights(fs.size)[None, None, :]
    )
    terms = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            def c2(f, x=x, y=y):
                return ours.chi2(d, _binary_vis(f, x, y))

            grid_c2 = np.array([c2(f) for f in fs])
            terms.append(-0.5 * (grid_c2 - null) + lw[i, j])
            f_hat = optimize.minimize_scalar(
                c2, bounds=(-0.15, float(fs[-1]) * 1.25), method="bounded", options={"xatol": 1e-12}
            ).x
            f_pos = max(float(f_hat), 0.0)
            dc2 = null - c2(f_pos)
            if dc2 > best_dc2:
                best_dc2 = dc2
                best_x, best_y, best_f = float(x), float(y), f_pos
            h = 1e-4
            curv = (c2(f_hat + h) - 2.0 * c2(f_hat) + c2(f_hat - h)) / h**2
            if curv > 0.0:
                best_snr = max(best_snr, float(f_hat) / float(np.sqrt(2.0 / curv)))
    log_b = float(special.logsumexp(np.concatenate([np.ravel(t) for t in terms])))
    return {
        "delta_chi2": float(best_dc2),
        "log_bayes_factor": log_b,
        "max_snr": float(best_snr),
        "best_dra": best_x,
        "best_ddec": best_y,
        "best_flux": best_f,
    }


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

    # Non-monotone empirical completeness [0.25, 0.75, 0.50, 1.0]: cumulative
    # maximum clamps [0.25, 0.75, 0.75, 1.0], so completeness=0.875 interpolates
    # halfway in log-flux between 4e-3 and 8e-3 (4e-3 * sqrt(2)), distinct from
    # pool-adjacent-violators isotonic regression (4e-3 * 2**(2/3))
    nm_fluxes = np.repeat([1e-3, 2e-3, 4e-3, 8e-3], 4)
    nm_inj = np.array(
        [
            15.0, 1.0, 1.0, 1.0,      # 0.25 at 1e-3
            15.0, 15.0, 15.0, 1.0,    # 0.75 at 2e-3
            15.0, 15.0, 1.0, 1.0,     # 0.50 at 4e-3 (non-monotone dip)
            15.0, 15.0, 15.0, 15.0,   # 1.00 at 8e-3
        ]
    )
    mc_nonmono = _make_synthetic_mc(
        np.linspace(0.0, 9.0, 10),
        inj_vals=nm_inj,
        dra=np.full(16, 40.0),
        ddec=np.zeros(16),
        flux=nm_fluxes,
    )
    _, f_875 = mc_nonmono.contrast_curve("delta_chi2", 0.2, completeness=0.875)
    assert float(f_875[0]) == pytest.approx(4e-3 * np.sqrt(2.0), rel=1e-12)
    assert abs(float(f_875[0]) - 4e-3 * (2.0 ** (2.0 / 3.0))) > 5e-4

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
    with np.load(out_npz, allow_pickle=False) as buf:
        meta_bad = json.loads(str(buf["meta"]))
        meta_bad["format"] = 999
        np.savez(bad_npz, **{k: buf[k] for k in buf.files if k != "meta"}, meta=json.dumps(meta_bad))
    with pytest.raises(ValueError, match="format"):
        det.DetectionMC.load(bad_npz)


@pytest.mark.validates(
    "virgil.detection.injection_recovery",
    roots=["mathematics"],
    headline="max_rel_ddelta_chi2",
)
def test_injection_recovery_matches_independent_crosscheck_grid_search(tmp_path):
    """On both noiseless and noisy (bootstrap_null sign-flip) 3-telescope VLTI
    datasets synthesized via crosscheck.simulate, injection_recovery matches our
    independent crosscheck.chi2 + crosscheck.sky grid search across delta_chi2,
    log_bayes_factor, max_snr, best_dra, best_ddec, and best_flux on null and
    off/on-grid injected draws while realizing distinct noisy draws per key."""
    obs_kw = dict(
        hour_angles_h=np.linspace(-2.5, 2.5, 5),
        wavelengths=np.linspace(1.6e-6, 2.2e-6, 4),
        dec_deg=-45.0,
        sigma_v2=0.01,
        sigma_cp_deg=0.5,
    )
    tpl_path = tmp_path / "tpl.fits"
    simulate.observe(tpl_path, _binary_vis(0.0, 0.0, 0.0), UTS3, rng=None, **obs_kw)
    template = OIData(str(tpl_path))
    null_scene = vm.BinaryModelCartesian(0.0, 0.0, 0.0)

    # Even grid count avoids the (0, 0) primary-star degeneracy; injections mix
    # exact grid nodes ((5.4, -1.8), (-5.4, 5.4)) and off-node positions ((5.0, -2.1), (-5.1, 5.0))
    grid = {
        "dra": np.linspace(-9.0, 9.0, 6),
        "ddec": np.linspace(-9.0, 9.0, 6),
        "flux": np.geomspace(1e-4, 2e-1, 45),
    }
    injections = {
        "dra": np.array([5.4, 5.0, -5.1, -5.4]),
        "ddec": np.array([-1.8, -2.1, 5.0, 5.4]),
        "flux": np.array([3e-3, 9e-3, 2.5e-2, 7e-2]),
    }

    sim0 = det.gaussian_null(null_scene, template, error_scale=0.0)
    mc = det.injection_recovery(
        vm.BinaryModelCartesian,
        null_scene,
        template,
        grid,
        0,
        n_null=4,
        injections=injections,
        noise=sim0,
        match_radius=3.0,
        draw_batch=2,
        progress=False,
    )
    assert mc.n_null == 4 and mc.n_injected == 4

    want_null = _our_grid_search(ours.load(tpl_path), grid)
    worst_dc2, worst_logb, worst_snr, worst_flux = 0.0, 0.0, 0.0, 0.0
    for i in range(mc.n_null):
        worst_dc2 = max(worst_dc2, abs(float(mc.null["delta_chi2"][i]) - want_null["delta_chi2"]))
        worst_logb = max(
            worst_logb,
            abs(float(mc.null["log_bayes_factor"][i]) - want_null["log_bayes_factor"])
            / max(1.0, abs(want_null["log_bayes_factor"])),
        )
        worst_snr = max(worst_snr, abs(float(mc.null["max_snr"][i]) - want_null["max_snr"]))
        worst_flux = max(worst_flux, abs(float(mc.null["best_flux"][i]) - want_null["best_flux"]))

    for k in range(mc.n_injected):
        inj_path = tmp_path / f"inj_{k}.fits"
        simulate.observe(
            inj_path,
            _binary_vis(float(injections["flux"][k]), float(injections["dra"][k]), float(injections["ddec"][k])),
            UTS3,
            rng=None,
            **obs_kw,
        )
        want = _our_grid_search(ours.load(inj_path), grid)
        worst_dc2 = max(
            worst_dc2,
            abs(float(mc.injected["delta_chi2"][k]) - want["delta_chi2"]) / max(1.0, want["delta_chi2"]),
        )
        worst_logb = max(
            worst_logb,
            abs(float(mc.injected["log_bayes_factor"][k]) - want["log_bayes_factor"])
            / max(1.0, abs(want["log_bayes_factor"])),
        )
        worst_snr = max(
            worst_snr,
            abs(float(mc.injected["max_snr"][k]) - want["max_snr"]) / max(1.0, abs(want["max_snr"])),
        )
        worst_flux = max(
            worst_flux,
            abs(float(mc.injected["best_flux"][k]) - want["best_flux"]) / want["best_flux"],
        )
        assert float(mc.injected["best_dra"][k]) == pytest.approx(want["best_dra"], abs=1e-12)
        assert float(mc.injected["best_ddec"][k]) == pytest.approx(want["best_ddec"], abs=1e-12)

    assert np.all(mc.matched())
    assert np.all(mc.detected("delta_chi2", 0.25))

    # Noisy per-draw verification via bootstrap_null on a 4-residual UTS3 OIFITS
    # file (3 V^2 + 1 CP active on channel 0, channel 1 noiseless -> 2^4 = 16
    # exact sign-flip states): verifies every realized noisy null & injected draw
    # against _our_grid_search and proves PRNG key splitting across draws without
    # touching private fold_in internals
    boot_kw = dict(
        hour_angles_h=np.array([0.5]),
        wavelengths=np.array([1.8e-6, 2.2e-6]),
        dec_deg=-45.0,
        sigma_v2=0.015,
        sigma_cp_deg=0.8,
    )
    clean_b = tmp_path / "boot_clean.fits"
    noisy_b = tmp_path / "boot_noisy.fits"
    simulate.observe(clean_b, _binary_vis(0.0, 0.0, 0.0), UTS3, rng=None, **boot_kw)
    simulate.observe(noisy_b, _binary_vis(0.0, 0.0, 0.0), UTS3, rng=np.random.default_rng(2026), **boot_kw)
    with fits.open(noisy_b, mode="update") as hdul:
        hdul["OI_VIS2"].data["VIS2DATA"][:, 1] = 1.0
        hdul["OI_T3"].data["T3PHI"][:, 1] = 0.0

    d_bc = ours.load(clean_b)
    d_bn = ours.load(noisy_b)
    res_v2 = d_bn["v2"] - d_bc["v2"]
    res_cp = d_bn["cp"] - d_bc["cp"]

    grid_b = {
        "dra": np.linspace(-8.0, 8.0, 4),
        "ddec": np.linspace(-8.0, 8.0, 4),
        "flux": np.geomspace(2e-4, 2e-1, 35),
    }
    inj_b = {
        "dra": np.array([8.0, -4.2, 2.3]),
        "ddec": np.array([-8.0, 3.8, -5.1]),
        "flux": np.array([2.5e-2, 4e-2, 6e-2]),
    }
    boot_oi = OIData(str(noisy_b))
    mc_noisy = det.injection_recovery(
        vm.BinaryModelCartesian,
        null_scene,
        boot_oi,
        grid_b,
        12345,
        n_null=24,
        injections=inj_b,
        noise=det.bootstrap_null(null_scene, boot_oi),
        draw_batch=4,
        progress=False,
    )

    null_catalog = []
    for signs in itertools.product((-1.0, 1.0), repeat=4):
        d_s = dict(d_bc)
        sv = np.ones_like(res_v2)
        sv[:, 0] = signs[:3]
        sp = np.ones_like(res_cp)
        sp[:, 0] = signs[3]
        d_s["v2"] = d_bc["v2"] + sv * res_v2
        d_s["cp"] = d_bc["cp"] + sp * res_cp
        null_catalog.append(_our_grid_search(d_s, grid_b))

    matched_states = set()
    for i in range(mc_noisy.n_null):
        best_s, best_dc2_e, best_lb_e, best_snr_e, best_f_e = None, np.inf, np.inf, np.inf, np.inf
        best_tot = np.inf
        for s_idx, ref in enumerate(null_catalog):
            e_dc2 = abs(float(mc_noisy.null["delta_chi2"][i]) - ref["delta_chi2"]) / max(1.0, ref["delta_chi2"])
            e_lb = abs(float(mc_noisy.null["log_bayes_factor"][i]) - ref["log_bayes_factor"]) / max(
                1.0, abs(ref["log_bayes_factor"])
            )
            e_snr = abs(float(mc_noisy.null["max_snr"][i]) - ref["max_snr"]) / max(1.0, abs(ref["max_snr"]))
            e_f = abs(float(mc_noisy.null["best_flux"][i]) - ref["best_flux"])
            e_pos = (
                max(
                    abs(float(mc_noisy.null["best_dra"][i]) - ref["best_dra"]),
                    abs(float(mc_noisy.null["best_ddec"][i]) - ref["best_ddec"]),
                )
                if ref["delta_chi2"] > 1e-8
                else 0.0
            )
            tot = max(e_dc2, e_lb, e_snr, e_f, e_pos)
            if tot < best_tot:
                best_tot, best_s = tot, s_idx
                best_dc2_e, best_lb_e, best_snr_e, best_f_e = e_dc2, e_lb, e_snr, e_f
        matched_states.add(best_s)
        assert best_tot < 1e-5
        worst_dc2 = max(worst_dc2, best_dc2_e)
        worst_logb = max(worst_logb, best_lb_e)
        worst_snr = max(worst_snr, best_snr_e)
        worst_flux = max(worst_flux, best_f_e)
    assert len(matched_states) >= 6  # proves per-draw PRNG key splitting
    assert np.any(mc_noisy.null["delta_chi2"] == 0.0) and np.any(mc_noisy.null["delta_chi2"] > 0.5)

    for k in range(mc_noisy.n_injected):
        pk = tmp_path / f"boot_inj_{k}.fits"
        simulate.observe(
            pk,
            _binary_vis(float(inj_b["flux"][k]), float(inj_b["dra"][k]), float(inj_b["ddec"][k])),
            UTS3,
            rng=None,
            **boot_kw,
        )
        d_ic = ours.load(pk)
        best_tot = np.inf
        best_dc2_e, best_lb_e, best_snr_e, best_f_e = np.inf, np.inf, np.inf, np.inf
        for signs in itertools.product((-1.0, 1.0), repeat=4):
            d_s = dict(d_ic)
            sv = np.ones_like(res_v2)
            sv[:, 0] = signs[:3]
            sp = np.ones_like(res_cp)
            sp[:, 0] = signs[3]
            d_s["v2"] = d_ic["v2"] + sv * res_v2
            d_s["cp"] = d_ic["cp"] + sp * res_cp
            ref = _our_grid_search(d_s, grid_b)
            e_dc2 = abs(float(mc_noisy.injected["delta_chi2"][k]) - ref["delta_chi2"]) / max(1.0, ref["delta_chi2"])
            e_lb = abs(float(mc_noisy.injected["log_bayes_factor"][k]) - ref["log_bayes_factor"]) / max(
                1.0, abs(ref["log_bayes_factor"])
            )
            e_snr = abs(float(mc_noisy.injected["max_snr"][k]) - ref["max_snr"]) / max(1.0, abs(ref["max_snr"]))
            e_f = abs(float(mc_noisy.injected["best_flux"][k]) - ref["best_flux"]) / max(1e-6, ref["best_flux"])
            e_pos = max(
                abs(float(mc_noisy.injected["best_dra"][k]) - ref["best_dra"]),
                abs(float(mc_noisy.injected["best_ddec"][k]) - ref["best_ddec"]),
            )
            tot = max(e_dc2, e_lb, e_snr, e_f, e_pos)
            if tot < best_tot:
                best_tot = tot
                best_dc2_e, best_lb_e, best_snr_e, best_f_e = e_dc2, e_lb, e_snr, e_f
        assert best_tot < 1e-5
        worst_dc2 = max(worst_dc2, best_dc2_e)
        worst_logb = max(worst_logb, best_lb_e)
        worst_snr = max(worst_snr, best_snr_e)
        worst_flux = max(worst_flux, best_f_e)

    record("max_rel_ddelta_chi2", worst_dc2)
    record("max_rel_dlogB", worst_logb)
    record("max_rel_dsnr", worst_snr)
    record("max_rel_dflux", worst_flux)
    assert worst_dc2 < 1e-5
    assert worst_logb < 1e-5
    assert worst_snr < 1e-4
    assert worst_flux < 1e-5


@pytest.mark.validates(
    "virgil.detection.injection_recovery",
    roots=["self-consistency"],
    headline="max_rel_dbatch",
)
def test_injection_recovery_nrm_batch_and_chunk_invariance():
    """On a noisy 7-hole NRM UV template (n_null=24, n_inject=16),
    injection_recovery produces distinct noisy realizations across draws and
    identical per-draw statistics across (draw_batch=1, chunk_size=None) and
    (draw_batch=4, chunk_size=8)."""
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
    mc_serial = det.injection_recovery(
        vm.BinaryModelCartesian,
        null_scene,
        template,
        grid,
        seed,
        n_null=24,
        injections=injections,
        match_radius=60.0,
        draw_batch=1,
        chunk_size=None,
        progress=False,
    )
    mc_chunked = det.injection_recovery(
        vm.BinaryModelCartesian,
        null_scene,
        template,
        grid,
        seed,
        n_null=24,
        injections=injections,
        match_radius=60.0,
        draw_batch=4,
        chunk_size=8,
        progress=False,
    )
    assert mc_serial.n_null == 24 and mc_serial.n_injected == 16
    assert np.unique(mc_serial.null["log_bayes_factor"]).size == mc_serial.n_null
    assert np.unique(mc_serial.null["delta_chi2"]).size >= mc_serial.n_null - 2
    assert float(np.std(mc_serial.null["delta_chi2"])) > 0.0

    worst_batch = 0.0
    for key_name in ("delta_chi2", "log_bayes_factor", "max_snr", "best_dra", "best_ddec", "best_flux"):
        for a, b in ((mc_serial.null[key_name], mc_chunked.null[key_name]), (mc_serial.injected[key_name], mc_chunked.injected[key_name])):
            denom = np.maximum(1.0, np.abs(a))
            worst_batch = max(worst_batch, float(np.max(np.abs(a - b) / denom)))

    record("max_rel_dbatch", worst_batch)
    assert worst_batch < 1e-6


@pytest.mark.validates(
    "virgil.detection.injection_recovery",
    roots=["mathematics"],
    kind="control",
)
def test_injection_recovery_rejects_off_grid_false_position_matches(tmp_path):
    """Negative control: a bright companion injected far outside the searched
    coordinate box produces interferometric sidelobe peaks inside the box whose
    delta_chi2 exceeds the null threshold, so ignoring match_radius falsely
    counts them as recoveries while position-matched recovery rejects 100%."""
    tpl_path = tmp_path / "ctrl_tpl.fits"
    simulate.observe(
        tpl_path,
        _binary_vis(0.0, 0.0, 0.0),
        UTS3,
        hour_angles_h=np.linspace(-2.5, 2.5, 5),
        wavelengths=np.linspace(1.6e-6, 2.2e-6, 4),
        dec_deg=-45.0,
        sigma_v2=0.005,
        sigma_cp_deg=0.25,
        rng=None,
    )
    template = OIData(str(tpl_path))
    grid = {
        "dra": np.linspace(-6.0, 6.0, 5),
        "ddec": np.linspace(-6.0, 6.0, 5),
        "flux": np.geomspace(1e-3, 1e-1, 8),
    }
    injections = {
        "dra": np.array([24.0, -24.0, 0.0, 20.0]),
        "ddec": np.array([0.0, 18.0, -24.0, -20.0]),
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
        match_radius=3.0,
        progress=False,
    )
    thr, _ = mc.threshold("delta_chi2", 0.15)
    unmatched_exceedance = np.mean(mc.injected["delta_chi2"] > thr)
    matched_completeness = float(mc.completeness("delta_chi2", 0.15, sep_bins=[15.0, 40.0])["completeness"][0, 0])
    assert unmatched_exceedance >= 0.75
    assert matched_completeness == 0.0
