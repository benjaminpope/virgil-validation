"""Validation of virgil 0.4.2 L-curve DoF-corrected discrepancy (Hutchinson
Rademacher trace), Laplace GP evidence/samples/error_scale, imaging diagnostics,
MYTHRA randomized ensembles (including GP evidence grids), and radio
time-adjacent noise scales & staged nuisance fits against ``crosscheck.starlet_kkt_ref``.
"""

from __future__ import annotations

import dataclasses
import gc
import tempfile
from pathlib import Path

from astropy.io import fits
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from crosscheck import simulate, starlet_kkt_ref as ref
from evidence.plugin import record

vc = pytest.importorskip("virgil.coverage")
ve = pytest.importorskip("virgil.ensemble")
vf = pytest.importorskip("virgil.fields")
vfit = pytest.importorskip("virgil.fitting")
vi = pytest.importorskip("virgil.imaging")
vl = pytest.importorskip("virgil.likelihood")
vm = pytest.importorskip("virgil.models")
vo = pytest.importorskip("virgil.oidata")
vr = pytest.importorskip("virgil.radio")

pytestmark = pytest.mark.x64


@pytest.fixture(scope="module", autouse=True)
def _clear_jax_caches():
    yield
    jax.clear_caches()
    gc.collect()


_STATIONS4 = np.array([[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]])


def _vlti_like_oidata(
    scene,
    *,
    seed: int = 0,
    n_telescopes: int = 4,
    vis_err: float = 0.015,
    cp_err_deg: float = 0.8,
    error_inflation: float = 1.0,
):
    """Compact synthetic 3- or 4-telescope V^2 + closure-phase ``OIData`` via OIFITS."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sim.fits"
        simulate.observe(
            path,
            lambda u, v, w: np.asarray(scene.model(u, v, w)),
            _STATIONS4[:n_telescopes],
            hour_angles_h=np.array([-1.5, 0.0, 1.5]),
            wavelengths=np.array([2.1e-6, 2.3e-6]),
            dec_deg=-30.0,
            sigma_v2=vis_err,
            sigma_cp_deg=cp_err_deg,
            rng=np.random.default_rng(seed),
        )
        if error_inflation != 1.0:
            with fits.open(path, mode="update") as hdul:
                hdul["OI_VIS2"].data["VIS2ERR"] *= float(error_inflation)
                hdul["OI_T3"].data["T3PHIERR"] *= float(error_inflation)
                hdul.flush()
        return vo.OIData(str(path))


# ---------------------------------------------------------------------------
# LCurve & DofDiscrepancy (WF4: Hutchinson Rademacher probes on LinearImage)
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.imaging.LCurve",
    "virgil.imaging.DofDiscrepancy",
    roots=["mathematics", "statistics"],
    headline="dof_z_score",
)
def test_lcurve_dof_discrepancy_matches_closed_form_tsv_ridge_trace_and_se():
    n, scale, flux = 6, 1.5, 0.35
    shape = ref.gaussian_blob_ref(n, scale, sigma_mas=2.2, dra=0.8, ddec=-0.6) * flux
    truth = vm.System(
        star=vm.PointSource(),
        env=vm.LinearImage(shape, scale, parametrization="linear", reference_flux=flux),
    )
    data = _vlti_like_oidata(truth, seed=19, n_telescopes=3, vis_err=0.015, cp_err_deg=0.8)

    # LinearImage("linear") fitted with bounded L-BFGS-B per LinearImage / image_priors docs
    start = vm.System(
        star=vm.PointSource(),
        env=vm.LinearImage(
            np.full((n, n), flux / (n * n)),
            scale,
            parametrization="linear",
            reference_flux=flux,
        ),
    )
    priors = vi.image_priors(start)
    reg = vi.TSV(1.0, beam_mas=scale, path="env")
    weights = np.geomspace(5.0e4, 0.05, 8)
    curve = vi.l_curve(start, priors, data, reg, weights, method="lbfgsb", max_steps=4000)

    choice = curve.dof_discrepancy(
        data,
        reg,
        nprobe=128,
        seed=12,
        max_secant=5,
        noise_scale2=1.0,
        noise_scale_sigma=0.05,
    )
    assert isinstance(choice, vi.DofDiscrepancy)
    assert choice.weight is not None
    assert choice.accepted is True
    assert abs(float(choice.m)) <= 3.0 * float(choice.g_se)
    assert choice.h is None  # h diagnostic is MaxEntropy-only
    assert choice.weight_band is not None and len(choice.weight_band) == 2
    w_hi, w_lo = choice.weight_band
    want_w_hi, want_w_lo = ref.dof_weight_band_ref(
        np.asarray(choice.weights, dtype=float),
        np.asarray(curve.chi2, dtype=float),
        np.asarray(choice.g_sweep, dtype=float),
        float(choice.n_real),
        noise_scale2=1.0,
        noise_scale_sigma=0.05,
    )
    assert w_hi is not None and w_lo is not None
    assert want_w_hi is not None and want_w_lo is not None
    assert len(choice.roots) >= 1 and float(w_lo) < float(choice.roots[0]) < float(w_hi)
    weight_band_err = max(
        abs(float(w_hi) - float(want_w_hi)),
        abs(float(w_lo) - float(want_w_lo)),
    )
    assert weight_band_err < 1e-12

    # Refit at choice.weight to check exact dense Tr[K (K + H_R)^{-1}] and Rademacher SE over free pixels
    chosen_fit = vfit.fit(
        start,
        priors,
        data,
        [vi.TSV(float(choice.weight), beam_mas=scale, path="env")],
        method="lbfgsb",
        max_steps=4000,
    )
    f_hat = np.asarray(chosen_fit.model.env.fluxes)
    free = f_hat > 0.0
    assert np.any(free)

    jac_full = np.asarray(
        jax.jacrev(
            lambda f_vec: vl.whitened_residuals(
                vm.System(
                    star=vm.PointSource(),
                    env=vm.LinearImage(f_vec.reshape(n, n), scale, parametrization="linear", reference_flux=flux),
                ),
                data,
            )
        )(jnp.asarray(f_hat.ravel()))
    )
    jac_free = jac_full[:, free.ravel()]
    k_mat = jac_free.T @ jac_free
    h_reg = ref.tsv_dirichlet_hessian_ref(
        n,
        weight=float(choice.weight),
        reference_flux=flux,
        pixel_scale_mas=scale,
        beam_mas=scale,
        free_mask=free,
    )
    exact_dof = ref.ridge_dof_and_hutchinson_se_ref(k_mat, h_reg, nprobe=128)

    dof_z = abs(float(choice.g) - exact_dof["g"]) / max(exact_dof["se_rademacher"], 1e-6)
    se_ratio = float(choice.g_se) / exact_dof["se_rademacher"]

    record("dof_z_score", dof_z)
    record("g_exact", exact_dof["g"])
    record("se_ratio", se_ratio)
    assert dof_z < 3.5
    assert 0.45 < se_ratio < 1.85


@pytest.mark.validates(
    "virgil.imaging.LCurve",
    "virgil.imaging.DofDiscrepancy",
    roots=["mathematics"],
    headline="h_diag_err",
)
def test_lcurve_dof_discrepancy_maxent_h_diagnostic_and_custom_curvature():
    n, scale, flux = 6, 1.5, 0.3
    shape = ref.gaussian_blob_ref(n, scale, sigma_mas=2.0, dra=0.8) * flux
    truth = vm.System(
        star=vm.PointSource(),
        env=vm.LinearImage(np.sqrt(shape), scale, parametrization="sqrt", reference_flux=flux),
    )
    data = _vlti_like_oidata(truth, seed=5, n_telescopes=3, vis_err=0.015, cp_err_deg=0.8)

    start = vm.System(
        star=vm.PointSource(),
        env=vm.LinearImage(
            np.sqrt(np.full((n, n), flux / (n * n))),
            scale,
            parametrization="sqrt",
            reference_flux=flux,
        ),
    )
    priors = vi.image_priors(start)
    reg = vi.MaxEntropy(1.0, path="env")
    curve = vi.l_curve(start, priors, data, reg, np.geomspace(500.0, 0.05, 8), max_steps=2000)

    choice = curve.dof_discrepancy(data, reg, nprobe=64, seed=4, max_secant=5)
    assert choice.weight is not None
    assert choice.accepted is True
    assert choice.h is not None

    jac_fn = jax.jit(
        jax.jacrev(
            lambda fv: vl.whitened_residuals(
                vm.System(
                    star=vm.PointSource(),
                    env=vm.LinearImage(fv.reshape(n, n), scale, parametrization="linear", reference_flux=flux),
                ),
                data,
            )
        )
    )

    # Verify BOTH MaxEntropy h(w) = 2 (w / s^2) KL - G (using pure-NumPy skilling_kl_ref,
    # zero virgil.imaging.MaxEntropy.value calls) AND dense Skilling-Hessian ridge trace Tr[K (K + H_ME)^{-1}]
    refit_res = vfit.fit(
        start,
        priors,
        data,
        [vi.MaxEntropy(float(choice.weight), path="env")],
        max_steps=4000,
        gtol=1e-12,
    )
    f_hat = np.asarray(refit_res.model.env.fluxes)
    kl_unweighted = ref.skilling_kl_ref(f_hat, reference_flux=flux)
    want_h = 2.0 * float(choice.weight) * kl_unweighted - float(choice.g)
    h_err = abs(float(choice.h) - want_h)

    free = f_hat > 0.0
    jac_me = np.asarray(jac_fn(jnp.asarray(f_hat).ravel()))[:, free.ravel()]
    k_me = jac_me.T @ jac_me
    h_me = ref.maxent_hessian_ref(f_hat, float(choice.weight), flux, free_mask=free)
    exact_me = ref.ridge_dof_and_hutchinson_se_ref(k_me, h_me, nprobe=64)
    me_dof_z = abs(float(choice.g) - exact_me["g"]) / max(exact_me["se_rademacher"], 1e-6)

    # Custom curvature= callable reproduces G when passed explicitly, and non-OIData sweep
    # without curvature= raises ValueError
    def custom_curv(model):
        f0 = jnp.asarray(model.env.fluxes).ravel()
        jac = np.asarray(jac_fn(f0))
        k_dense = jac.T @ jac
        return lambda v: k_dense @ np.asarray(v, dtype=float)

    choice_curv = curve.dof_discrepancy(
        data, reg, nprobe=8, tol=1e-9, seed=4, secant=False, curvature=custom_curv
    )
    choice_default = curve.dof_discrepancy(data, reg, nprobe=8, tol=1e-9, seed=4, secant=False)
    curv_g_diff = float(np.max(np.abs(np.asarray(choice_curv.g_sweep) - np.asarray(choice_default.g_sweep))))

    with pytest.raises(ValueError):
        curve.dof_discrepancy((), reg, secant=False)

    record("h_diag_err", h_err)
    record("maxent_dof_z_score", me_dof_z)
    record("custom_curvature_g_diff", curv_g_diff)
    assert h_err < 1e-4
    assert me_dof_z < 3.5
    assert curv_g_diff < 1e-7


@pytest.mark.validates(
    "virgil.imaging.LCurve",
    "virgil.imaging.DofDiscrepancy",
    roots=["mathematics"],
    kind="control",
    headline="min_m_sweep",
)
def test_lcurve_dof_discrepancy_no_crossing_returns_none_control():
    n, scale, flux = 6, 1.5, 0.3
    shape = ref.gaussian_blob_ref(n, scale, sigma_mas=1.8, dra=1.5) * flux
    truth = vm.System(
        star=vm.PointSource(),
        env=vm.LinearImage(np.sqrt(shape), scale, parametrization="sqrt", reference_flux=flux),
    )
    data = _vlti_like_oidata(truth, seed=8, n_telescopes=3, vis_err=0.01, cp_err_deg=0.5)

    start = vm.System(
        star=vm.PointSource(),
        env=vm.LinearImage(
            np.sqrt(np.full((n, n), flux / (n * n))),
            scale,
            parametrization="sqrt",
            reference_flux=flux,
        ),
    )
    priors = vi.image_priors(start)
    reg = vi.MaxEntropy(1.0, path="env")
    # Entirely over-regularized weights -> m(w) > 0 everywhere -> never falls back to weak end
    curve = vi.l_curve(start, priors, data, reg, np.array([1e5, 3e4, 1e4]), max_steps=1000)
    with pytest.warns(UserWarning):
        choice = curve.dof_discrepancy(data, reg, nprobe=16, seed=1)

    min_m = float(np.min(choice.m_sweep))
    record("min_m_sweep", min_m)
    assert min_m > 0.0
    assert choice.weight is None
    assert choice.accepted is False
    assert len(choice.roots) == 0


@pytest.mark.validates(
    "virgil.imaging.LCurve",
    roots=["mathematics"],
    headline="classic_maxent_rel_err",
)
def test_lcurve_corner_discrepancy_and_classic_maxent():
    n, scale = 6, 1.5
    blob = ref.gaussian_blob_ref(n, scale, sigma_mas=1.8, dra=1.2)
    truth = vm.System(star=vm.PointSource(), env=vm.Image.from_brightness(blob, scale, flux=0.20))
    data = _vlti_like_oidata(truth, seed=17, n_telescopes=3, vis_err=0.015, cp_err_deg=0.8)

    start = vm.System(star=vm.PointSource(), env=vm.Image(np.zeros((n, n)), scale, flux=0.20))
    priors = vi.image_priors(start)
    reg = vi.MaxEntropy(1.0, path="env")
    weights = np.geomspace(500.0, 0.05, 8)
    curve = vi.l_curve(start, priors, data, reg, weights, max_steps=2000)

    w_corner = float(curve.corner())
    want_corner = ref.lcurve_corner_ref(
        np.asarray(curve.weights), np.asarray(curve.chi2), np.asarray(curve.penalty)
    )
    corner_rel_err = abs(w_corner - float(want_corner)) / float(want_corner)

    w_disc = float(curve.discrepancy(target=1.0))
    w_classic = float(curve.classic_maxent(data, path="env"))

    brightnesses, jacs_b = [], []
    for res in curve.results:
        b = np.asarray(res.model.env.brightness)
        brightnesses.append(b)
        jac_eta = np.asarray(
            jax.jacrev(
                lambda eta_vec: vl.whitened_residuals(
                    res.model.set("env.log_brightness", eta_vec.reshape(n, n)),
                    data,
                )
            )(jnp.asarray(res.model.env.log_brightness).ravel())
        )
        jacs_b.append(jac_eta / b.ravel()[None, :])

    want_classic = ref.classic_maxent_weight_ref(
        np.asarray(curve.weights), np.asarray(curve.penalty), brightnesses, jacs_b
    )
    rel_err = abs(w_classic - float(want_classic)) / float(want_classic)

    record("classic_maxent_rel_err", rel_err)
    record("corner_rel_err", corner_rel_err)
    record("w_discrepancy", w_disc)
    assert corner_rel_err < 1e-12
    assert weights[-1] <= w_disc <= weights[0]
    assert rel_err < 1e-5


# ---------------------------------------------------------------------------
# envelope_fwhm & starting_image
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.imaging.envelope_fwhm",
    "virgil.imaging.starting_image",
    roots=["mathematics"],
    headline="envelope_fwhm_rel_err",
)
def test_envelope_fwhm_undiluted_by_bright_star_and_starting_image_modes():
    true_fwhm = 9.0
    sigma_mas = true_fwhm / ref.FWHM_PER_SIGMA
    scene = vm.System(star=vm.PointSource(), env=vm.GaussianDisk(sigma=sigma_mas, flux=0.35))
    data = _vlti_like_oidata(scene, seed=10, vis_err=0.005, cp_err_deg=0.3)

    env_fwhm = float(vi.envelope_fwhm(data, star=True))
    rel_err = abs(env_fwhm - true_fwhm) / true_fwhm

    s_mom = vi.starting_image(data, star=True, oversample=2.0, start="moments", hole_mas=1.5)
    with pytest.raises(ValueError):
        vi.starting_image(data, star=True, oversample=2.0, start="dirty")
    disco_scene = vm.System(star=vm.PointSource(), env=vm.GaussianDisk(sigma=120.0, flux=0.30))
    disco_data = vo.OIData(vc.ami_grid_record(wavelength_m=4.8e-6)).with_model(
        disco_scene, key=jax.random.PRNGKey(3)
    )
    s_dirty = vi.starting_image(disco_data, star=True, oversample=2.0, start="dirty")
    s_nostar = vi.starting_image(data, star=False, oversample=2.0, pad_to_mas=120.0)

    assert isinstance(s_mom, vm.System) and hasattr(s_mom, "env")
    assert s_mom.env.support is not None
    mid = s_mom.env.support.shape[0] // 2
    assert not bool(np.asarray(s_mom.env.support)[mid, mid])
    assert abs(float(np.sum(s_dirty.env.brightness)) - 1.0) < 1e-12
    assert isinstance(s_nostar, vm.Image)
    assert s_nostar.brightness.shape[0] * float(s_nostar.pixel_scale_mas) >= 120.0 - 1e-9

    record("envelope_fwhm_rel_err", rel_err)
    record("envelope_fwhm_mas", env_fwhm)
    assert rel_err < 0.08


@pytest.mark.validates(
    "virgil.imaging.envelope_fwhm",
    roots=["mathematics"],
    kind="control",
    headline="dilution_ratio",
)
def test_envelope_fwhm_diluted_second_moment_control():
    true_fwhm = 10.0
    scene = vm.System(
        star=vm.PointSource(),
        env=vm.GaussianDisk(sigma=true_fwhm / ref.FWHM_PER_SIGMA, flux=0.22),
    )
    data = _vlti_like_oidata(scene, seed=11, vis_err=0.004, cp_err_deg=0.3)
    diluted = float(vi.second_moment_fwhm(data))
    undiluted = float(vi.envelope_fwhm(data, star=True))
    ratio = diluted / undiluted
    record("dilution_ratio", ratio)
    assert ratio < 0.65


# ---------------------------------------------------------------------------
# log_evidence, laplace_samples, error_scale, chi2_by_observable,
# short_baseline_residual, Diagnosis, diagnose
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.imaging.log_evidence",
    "virgil.imaging.laplace_samples",
    "virgil.imaging.error_scale",
    roots=["mathematics", "statistics"],
    headline="log_evidence_abs_err",
)
def test_gp_log_evidence_laplace_samples_and_error_scale_match_mackay_reference():
    npix, scale = 6, 1.5
    truth = vm.System(
        star=vm.PointSource(),
        env=vm.Image.from_brightness(
            ref.gaussian_blob_ref(npix, scale, sigma_mas=2.2, dra=1.2, ddec=-0.8),
            scale,
            flux=0.25,
        ),
    )
    data = _vlti_like_oidata(truth, seed=31, n_telescopes=4, vis_err=0.012, cp_err_deg=0.6)

    gp = vf.GaussianField(np.zeros((npix, npix)), sigma=0.9, length_mas=2.5)
    start = vm.System(star=vm.PointSource(), env=vm.Image(gp, scale, flux=0.25))
    priors = vi.image_priors(start)
    done = vfit.fit(start, priors, data, max_steps=1500)

    got_logz = float(vi.log_evidence(done, data, path="env"))
    got_s = float(vi.error_scale(done, data, path="env"))

    z_map = np.asarray(done.model.env.log_brightness.latent)
    res_vec = np.asarray(vl.whitened_residuals(done.model, data))

    def res_of_z(z_flat):
        m = done.model.set("env.log_brightness.latent", z_flat.reshape(z_map.shape))
        return vl.whitened_residuals(m, data)

    jac_z = np.asarray(jax.jacrev(res_of_z)(jnp.asarray(z_map.ravel())))
    mk = ref.mackay_laplace_gp_ref(
        res_vec, 0.0, z_map, jac_z, n_independent=int(data.n_independent)
    )
    logz_err = abs(got_logz - float(mk["log_evidence"]))
    s_fp_err = abs(got_s - float(mk["s_fixed_point"]))

    # Laplace posterior draws match N(z_map, (I + J^T J)^{-1})
    draws = vi.laplace_samples(
        done, data, 4000, jax.random.PRNGKey(99), path="env", npix=12, fov_mas=18.0
    )
    z_draws = np.asarray(draws["latents"]).reshape(4000, -1)
    emp_mean_err = float(np.max(np.abs(np.mean(z_draws, axis=0) - z_map.ravel())))
    emp_cov = np.cov(z_draws, rowvar=False)
    cov_fro_err = float(np.linalg.norm(emp_cov - mk["cov"]) / np.linalg.norm(mk["cov"]))
    assert draws["images"].shape == (4000, 12, 12)
    assert np.allclose(np.sum(draws["images"], axis=(1, 2)), 1.0, atol=1e-12)
    for k in range(5):
        want_img = np.asarray(
            done.model.set("env.log_brightness.latent", draws["latents"][k]).render(12, 18.0)
        )
        assert float(np.max(np.abs(np.asarray(draws["images"][k]) - want_img))) < 1e-12
    assert float(np.max(np.std(np.asarray(draws["images"]), axis=0))) > 1e-3
    with pytest.raises(ValueError):
        vi.laplace_samples(done, data, 4, jax.random.PRNGKey(1), path="env", npix=12)

    # Error scale with 2x inflated error bars: verify BOTH scalar and coupled per-observable
    # fixed-point equations directly against mackay_laplace_gp_ref (and distinguish from s_plugin)
    inflated = _vlti_like_oidata(
        truth, seed=31, n_telescopes=4, vis_err=0.012, cp_err_deg=0.6, error_inflation=2.0
    )
    done_inf = vfit.fit(start, priors, inflated, max_steps=1500)
    s_inf = float(vi.error_scale(done_inf, inflated, path="env"))
    s_by_obs = vi.error_scale(done_inf, inflated, path="env", by_observable=True)
    assert set(s_by_obs.keys()) == {"vis", "phi"}

    z_inf = np.asarray(done_inf.model.env.log_brightness.latent)
    res_inf = np.asarray(vl.whitened_residuals(done_inf.model, inflated))
    jac_inf = np.asarray(
        jax.jacrev(
            lambda zf: vl.whitened_residuals(
                done_inf.model.set("env.log_brightness.latent", zf.reshape(z_inf.shape)),
                inflated,
            )
        )(jnp.asarray(z_inf.ravel()))
    )
    # 4 telescopes x 6 frames: 36 V^2 rows (rank 36) followed by 24 T3 rows (rank 18)
    mk_inf = ref.mackay_laplace_gp_ref(
        res_inf,
        0.0,
        z_inf,
        jac_inf,
        n_independent=int(inflated.n_independent),
        observable_blocks={"vis": (slice(0, 36), 36), "phi": (slice(36, len(res_inf)), 18)},
    )
    s_inf_fp_err = abs(s_inf - float(mk_inf["s_fixed_point"]))
    s_by_obs_err = max(
        abs(float(s_by_obs["vis"]) - float(mk_inf["s_by_observable"]["vis"])),
        abs(float(s_by_obs["phi"]) - float(mk_inf["s_by_observable"]["phi"])),
    )

    record("log_evidence_abs_err", logz_err)
    record("s_fixed_point_err", max(s_fp_err, s_inf_fp_err))
    record("s_by_observable_err", s_by_obs_err)
    record("cov_fro_rel_err", cov_fro_err)
    assert logz_err < 1e-10
    assert s_fp_err < 1e-9
    assert s_inf_fp_err < 1e-9
    assert abs(s_inf - float(mk_inf["s_plugin"])) > 1e-2  # unrescaled plug-in misses by ~0.018
    assert s_by_obs_err < 1e-9
    assert emp_mean_err < 0.08
    assert cov_fro_err < 0.15


@pytest.mark.validates(
    "virgil.imaging.error_scale",
    roots=["statistics"],
    kind="control",
    headline="overquoted_scale",
)
def test_gp_error_scale_wrong_direction_control():
    npix, scale = 6, 1.5
    truth = vm.System(star=vm.PointSource(), env=vm.GaussianDisk(sigma=1.5, flux=0.2))
    inflated = _vlti_like_oidata(
        truth, seed=44, n_telescopes=4, vis_err=0.01, cp_err_deg=0.5, error_inflation=3.0
    )
    start = vm.System(
        star=vm.PointSource(),
        env=vm.Image(vf.GaussianField(np.zeros((npix, npix)), sigma=0.8, length_mas=2.5), scale, flux=0.2),
    )
    done = vfit.fit(start, vi.image_priors(start), inflated, max_steps=1000)
    s = float(vi.error_scale(done, inflated, path="env"))
    record("overquoted_scale", s)
    assert s < 0.55


@pytest.mark.validates(
    "virgil.imaging.chi2_by_observable",
    "virgil.imaging.short_baseline_residual",
    "virgil.imaging.Diagnosis",
    "virgil.imaging.diagnose",
    roots=["mathematics"],
    headline="n_gamma_err",
)
def test_chi2_by_observable_short_baseline_residual_and_diagnose():
    npix, scale = 14, 1.5
    b_true_sb = ref.gaussian_blob_ref(npix, scale, sigma_mas=1.6, dra=0.8, ddec=-0.5)
    truth = vm.System(
        star=vm.PointSource(),
        env=vm.Image(
            np.log(np.clip(b_true_sb, 1e-300, None)),
            scale,
            flux=0.30,
        ),
    )
    data4 = _vlti_like_oidata(truth, seed=52, n_telescopes=4, vis_err=0.012, cp_err_deg=0.6)

    gp = vf.GaussianField(
        np.zeros((npix, npix)),
        sigma=0.8,
        length_mas=2.5,
        mean=ref.gaussian_blob_ref(npix, scale, sigma_mas=2.2),
    )
    gp_scene = vm.System(star=vm.PointSource(), env=vm.Image(gp, scale, flux=0.30))
    done = vfit.fit(gp_scene, vi.image_priors(gp_scene), data4, max_steps=1500)

    cbo = vi.chi2_by_observable(done.model, data4, path="env")
    assert cbo["n"]["vis"] == 36
    assert cbo["n"]["phi"] == 18
    assert cbo["n"]["total"] == 54
    assert abs(cbo["chi2"]["vis"] + cbo["chi2"]["phi"] - cbo["chi2"]["total"]) < 1e-12

    z_map = np.asarray(done.model.env.log_brightness.latent)
    jac_z = np.asarray(
        jax.jacrev(
            lambda zf: vl.whitened_residuals(done.model.set("env.log_brightness.latent", zf.reshape(z_map.shape)), data4)
        )(jnp.asarray(z_map.ravel()))
    )
    svals = np.linalg.svdvals(jac_z)
    want_gamma = float(np.sum((svals**2) / (1.0 + svals**2)))
    gamma_err = abs(float(cbo["n_gamma"]) - want_gamma)

    sb_true_mean, sb_se = vi.short_baseline_residual(truth, data4, quantile=0.25)
    sb_star_mean, sb_star_se = vi.short_baseline_residual(vm.PointSource(), data4, quantile=0.25)
    east_sb, north_sb = ref.sky_grid_mas(npix, scale)
    b_true_sb = ref.gaussian_blob_ref(npix, scale, sigma_mas=1.6, dra=0.8, ddec=-0.5)
    phase_sb = (
        2j
        * np.pi
        * (
            (data4.u[:, None] / data4.wavel[:, None]) * east_sb.ravel()[None, :]
            + (data4.v[:, None] / data4.wavel[:, None]) * north_sb.ravel()[None, :]
        )
        * ref.MAS_TO_RAD
    )
    cvis_true = (1.0 + 0.30 * (np.exp(phase_sb) @ b_true_sb.ravel())) / 1.30
    want_true_mean, want_se = ref.short_baseline_v2_residual_ref(
        data4.u, data4.v, data4.wavel, data4.vis, data4.d_vis, cvis_true, quantile=0.25
    )
    want_star_mean, want_star_se = ref.short_baseline_v2_residual_ref(
        data4.u,
        data4.v,
        data4.wavel,
        data4.vis,
        data4.d_vis,
        np.ones_like(data4.u, dtype=complex),
        quantile=0.25,
    )
    se_err = max(abs(float(sb_se) - want_se), abs(float(sb_star_se) - want_star_se))
    sb_mean_err = max(
        abs(float(sb_true_mean) - want_true_mean),
        abs(float(sb_star_mean) - want_star_mean),
    )

    disco = vo.OIData(vc.ami_grid_record(wavelength_m=4.8e-6))
    with pytest.raises(ValueError):
        vi.short_baseline_residual(truth, disco)

    diag_ok = vi.diagnose(done.model, data4)
    assert isinstance(diag_ok, vi.Diagnosis)
    assert diag_ok.checks["anchored"] is True
    assert len(diag_ok.warnings) == 0

    edge_eta = np.full((12, 12), -20.0)
    edge_eta[0, :] = 2.0
    unanchored_edge = vm.Image(edge_eta, 1.5)
    diag_bad = vi.diagnose(unanchored_edge, data4)
    assert diag_bad.checks["anchored"] is False
    assert diag_bad.checks["edge_flux"][0] > 0.5
    assert len(diag_bad.warnings) >= 2

    diag_cent = vi.diagnose(vm.Image(ref.gaussian_blob_ref(12, 1.5, 2.0), 1.5), data4, regularizers=[vi.Centroid(1.0)])
    assert diag_cent.checks["anchored"] is True

    record("n_gamma_err", gamma_err)
    record("short_baseline_se_err", se_err)
    record("short_baseline_mean_err", sb_mean_err)
    record("missing_flux_sb_mean", float(sb_star_mean))
    assert gamma_err < 1e-10
    assert se_err < 1e-12
    assert sb_mean_err < 1e-12
    assert abs(float(sb_true_mean)) < 3.0 * float(sb_se)
    assert float(sb_star_mean) > 4.0


# ---------------------------------------------------------------------------
# virgil.ensemble: Group, Member, Ensemble, reference_starts, run_group,
# GPFit, EvidenceGrid, evidence_grid, gp_members, combine, ensemble
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.ensemble.Group",
    "virgil.ensemble.Member",
    "virgil.ensemble.Ensemble",
    roots=["mathematics"],
    headline="std_pixel_err",
)
def test_ensemble_containers_group_member_and_ensemble_summary():
    # Compute Ensemble via a real multi-member combine() call with unequal envelope fluxes
    # plus a diverged member, and check whole-sky normalized mean.flux, mean.brightness, and std!
    u_m = np.array([15.0, 35.0, -25.0, 45.0, 60.0, -40.0])
    v_m = np.array([20.0, -15.0, 50.0, 30.0, -35.0, 25.0])
    data_v2 = vo.OIData(
        {
            "u": u_m,
            "v": v_m,
            "wavel": np.full(6, 2.2e-6),
            "vis": np.full(6, 0.78),
            "d_vis": np.full(6, 0.04),
            "vis_mode": "v2",
        }
    )
    draw = ve.Draw(0, family="gp", npix=8, pixel_scale_mas=1.5, start="moments", weights=())
    im1 = vm.Image.from_brightness(ref.gaussian_blob_ref(8, 1.5, 2.0, dra=1.0), 1.5, flux=0.25)
    im2 = vm.Image.from_brightness(ref.gaussian_blob_ref(8, 1.5, 2.0, dra=1.0, ddec=0.5), 1.5, flux=0.35)
    im3 = vm.Image.from_brightness(ref.gaussian_blob_ref(8, 1.5, 2.2, dra=0.8, ddec=-0.4), 1.5, flux=0.20)
    im_div = vm.Image.from_brightness(ref.gaussian_blob_ref(8, 1.5, 1.8), 1.5, flux=0.30)

    gfits = []
    for k, im in enumerate((im1, im2, im3)):
        m = vm.System(star=vm.PointSource(), env=im)
        c2 = float(np.sum(np.asarray(vl.whitened_residuals(m, data_v2)) ** 2))
        fr = vfit.FitResult(model=m, values={}, info={"chi2": [c2], "ndata": [6]})
        gfits.append(ve.GPFit("iso", 1.5 * (k + 1), 1.0, -4.0 - 0.1 * k, fr))
    fr_div = vfit.FitResult(
        model=vm.System(star=vm.PointSource(), env=im_div),
        values={},
        info={"chi2": [float("nan")], "ndata": [6]},
    )
    gfits.append(ve.GPFit("iso", 6.0, 1.0, -5.0, fr_div))

    grp = ve.Group(draw=draw, curve=None, gp=tuple(gfits))
    ens = ve.combine(
        data_v2,
        [grp],
        spec=ve.EnsembleSpec(chi2_ratio=10.0, mad_cut=10.0, mean_rtol=10.0, min_kept=3),
        star=True,
    )
    assert isinstance(ens, ve.Ensemble)
    assert len(ens.members) == 4
    assert all(isinstance(m, ve.Member) for m in ens.members)
    assert len(ens.kept) == 3
    assert any((not m.kept) and m.reason == "diverged" for m in ens.members)
    assert "gp" in ens.summary()

    # Independent whole-sky normalized aggregation: extended share e_k = (F_k / (1 + F_k)) * b_k
    e_stack = np.stack(
        [
            (float(m.result.model.env.flux) / (1.0 + float(m.result.model.env.flux)))
            * np.asarray(m.result.model.env.brightness)
            for m in ens.kept
        ],
        axis=0,
    )
    e_mean = np.mean(e_stack, axis=0)
    s_bar = 1.0 - float(np.sum(e_mean))
    want_flux = (1.0 - s_bar) / s_bar
    want_mean_b = e_mean / (1.0 - s_bar)
    want_std_b = np.std(e_stack / (1.0 - s_bar), axis=0, ddof=0)

    mean_err = max(
        abs(float(ens.mean.flux) - want_flux),
        float(np.max(np.abs(np.asarray(ens.mean.brightness) - want_mean_b))),
    )
    std_err = float(np.max(np.abs(np.asarray(ens.std) - want_std_b)))

    record("std_pixel_err", std_err)
    record("mean_sky_err", mean_err)
    assert std_err < 1e-13
    assert mean_err < 1e-13


@pytest.mark.validates(
    "virgil.ensemble.reference_starts",
    "virgil.ensemble.run_group",
    "virgil.ensemble.combine",
    "virgil.ensemble.ensemble",
    roots=["mathematics"],
    headline="mean_ncc",
)
def test_ensemble_reference_starts_run_group_combine_and_end_to_end():
    vlti_truth = vm.System(
        star=vm.PointSource(),
        env=vm.Image.from_brightness(ref.gaussian_blob_ref(8, 1.5, sigma_mas=2.4, dra=1.8, ddec=-1.2), 1.5, flux=0.28),
    )
    vlti_data = _vlti_like_oidata(vlti_truth, seed=61, n_telescopes=4, vis_err=0.012, cp_err_deg=0.6)
    starts = ve.reference_starts(vlti_data, star=True, starts=("moments", "flat"))
    assert set(starts.keys()) == {"moments", "flat"}
    for st_img in starts.values():
        assert isinstance(st_img, vm.Image)
        assert abs(float(np.sum(st_img.brightness)) - 1.0) < 1e-12
    # Closure-phase datasets without complex visibility phases must refuse start="dirty"
    with pytest.raises(ValueError):
        ve.reference_starts(vlti_data, star=True, starts=("dirty",))

    # DISCO complex-visibility modes provide all three starts ("moments", "flat", "dirty")
    npix, scale = 12, 32.0
    truth_blob = ref.gaussian_blob_ref(npix, scale, sigma_mas=42.0, dra=55.0, ddec=-40.0)
    disco_truth = vm.System(star=vm.PointSource(), env=vm.Image.from_brightness(truth_blob, scale, flux=0.35))
    disco_data = vo.OIData(vc.ami_grid_record(wavelength_m=4.8e-6)).with_model(
        disco_truth, key=jax.random.PRNGKey(11)
    )
    starts_disco = ve.reference_starts(disco_data, star=True, starts=("moments", "flat", "dirty"))
    assert set(starts_disco.keys()) == {"moments", "flat", "dirty"}
    assert abs(float(np.sum(starts_disco["dirty"].brightness)) - 1.0) < 1e-12

    # Direct run_group checks across star=True / star=False and family="tsv" / family="gp"
    direct_draw = ve.Draw(
        0, family="tsv", npix=npix, pixel_scale_mas=scale, start="moments", weights=(50.0, 5.0)
    )
    grp_direct = ve.run_group(disco_data, direct_draw, star=True, starts=starts_disco, max_steps=1500)
    assert isinstance(grp_direct, ve.Group)
    assert grp_direct.curve is not None and len(grp_direct.curve.results) == 2
    assert isinstance(grp_direct.curve.results[0].model, vm.System)

    grp_unstarred = ve.run_group(disco_data, direct_draw, star=False, max_steps=600)
    assert grp_unstarred.curve is not None and len(grp_unstarred.curve.results) == 2
    assert isinstance(grp_unstarred.curve.results[0].model, vm.Image)

    gp_direct_draw = ve.Draw(
        1, family="gp", npix=8, pixel_scale_mas=48.0, start="flat", weights=()
    )
    grp_gp_direct = ve.run_group(
        disco_data,
        gp_direct_draw,
        star=False,
        spec=ve.EnsembleSpec(
            gp_sigmas=(0.2, 0.5), gp_lengths=(0.5, 1.5), gp_growths=0, gp_axis_ratio=0.1
        ),
        max_steps=2500,
    )
    assert grp_gp_direct.curve is None and len(grp_gp_direct.gp) >= 1
    assert isinstance(grp_gp_direct.gp[0].result.model, vm.Image)

    spec = ve.EnsembleSpec(
        families=("tsv", "maxent"),
        n_weights=4,
        oversample=(2.0,),
        field_factors=(1.0,),
        starts=("moments", "dirty"),
        max_npix=12,
        field_mas=float(npix * scale),
        window_dex=1.5,
        min_kept=1,
    )
    ens = ve.ensemble(disco_data, 2, jax.random.PRNGKey(7), spec=spec, star=True, max_steps=4000)
    assert isinstance(ens, ve.Ensemble)
    assert len(ens.groups) == 2
    assert len(ens.members) == 8
    assert len(ens.kept) >= 1

    # Non-compounding running-mean gate: every entry of trace stays within (1 + mean_rtol) of trace[0]
    n_ind = float(disco_data.n_independent)
    rtol = float(np.sqrt(2.0 / n_ind))
    first = np.asarray(ens.trace[0])
    for step_chi2 in ens.trace:
        assert np.all(np.asarray(step_chi2) <= first * (1.0 + rtol + 1e-9))

    mean_b = np.asarray(ens.mean.brightness)
    e = mean_b.ravel() - np.mean(mean_b)
    t_resampled = ref.gaussian_blob_ref(
        mean_b.shape[0], float(ens.mean.pixel_scale_mas), sigma_mas=42.0, dra=55.0, ddec=-40.0
    )
    r = t_resampled.ravel() - np.mean(t_resampled)
    ncc_val = float((e @ r) / np.sqrt((e @ e) * (r @ r)))

    # star=False sub-beam alignment check + full deterministic rejection cascade ("window", "diverged", "chi2", "outlier")
    rng_ns = np.random.default_rng(66)
    u_ns = np.array([12.0, 28.0, -18.0, 35.0, -30.0, 22.0])
    v_ns = np.array([18.0, -12.0, 32.0, 24.0, -22.0, -28.0])
    b_base = ref.gaussian_blob_ref(15, 2.0, sigma_mas=2.8, dra=0.0, ddec=0.0)
    b_shift = ref.gaussian_blob_ref(15, 2.0, sigma_mas=2.8, dra=2.0, ddec=-2.0)  # 1px East, 1px South
    im_base = vm.Image.from_brightness(b_base, 2.0)
    im_shift = vm.Image.from_brightness(b_shift, 2.0)
    v2_ns = np.abs(np.asarray(im_base.model(u_ns, v_ns, 2.2e-6))) ** 2 + rng_ns.normal(0.0, 0.012, size=6)
    data_ns = vo.OIData(
        {"u": u_ns, "v": v_ns, "wavel": np.full(6, 2.2e-6), "vis": v2_ns, "d_vis": np.full(6, 0.012), "vis_mode": "v2"}
    )
    c2_base = float(np.sum(np.asarray(vl.whitened_residuals(im_base, data_ns)) ** 2))
    c2_shift = float(np.sum(np.asarray(vl.whitened_residuals(im_shift, data_ns)) ** 2))

    # Build LCurve group (3 weights: 100.0 > corner 10.0 -> "window", 10.0 & 2.0 in window) + GP group with diverged, chi2, outlier
    draw_lc = ve.Draw(0, family="tsv", npix=15, pixel_scale_mas=2.0, start="flat", weights=(100.0, 10.0, 2.0))
    lc_c2 = np.array([c2_base + 0.4, c2_base + 0.05, c2_base])
    lc_pen = np.array([0.02, 0.08, 0.90])  # corner at w = 10.0; w = 100.0 is above corner -> "window"
    lc_res = [
        vfit.FitResult(model=im_base, values={}, info={"chi2": [float(v)], "ndata": [6]})
        for v in lc_c2
    ]
    grp_lc = ve.Group(
        draw=draw_lc,
        curve=vi.LCurve(weights=np.array([100.0, 10.0, 2.0]), chi2=lc_c2, chi2_red=(lc_c2 / 6.0)[:, None], penalty=lc_pen, results=lc_res),
    )
    draw_gp = ve.Draw(1, family="gp", npix=15, pixel_scale_mas=2.0, start="flat", weights=())
    gp_entries = (
        ve.GPFit("iso", 3.0, 1.0, -2.0, vfit.FitResult(model=im_shift, values={}, info={"chi2": [c2_shift], "ndata": [6]})),
        ve.GPFit("iso", 4.0, 1.0, -2.1, vfit.FitResult(model=im_base, values={}, info={"chi2": [float("nan")], "ndata": [6]})),
        ve.GPFit("iso", 5.0, 1.0, -2.2, vfit.FitResult(model=im_base, values={}, info={"chi2": [c2_base * 3.0 + 12.0], "ndata": [6]})),
        ve.GPFit("iso", 6.0, 1.0, -2.3, vfit.FitResult(model=im_base, values={}, info={"chi2": [c2_base * 1.65], "ndata": [6]})),
    )
    grp_gp = ve.Group(draw=draw_gp, curve=None, gp=gp_entries)
    spec_cascade = ve.EnsembleSpec(window_dex=1.0, chi2_ratio=2.0, mad_cut=5.0, mean_rtol=2.0, min_kept=2)
    comb_ns = ve.combine(data_ns, [grp_lc, grp_gp], spec=spec_cascade, star=False)

    chi2_red_mat = np.array([m.chi2_red for m in comb_ns.members], dtype=float)
    in_win_mask = np.array([False, True, True, True, True, True, True])
    want_alive, want_reasons = ref.mythra_filter_indices_ref(
        chi2_red_mat, in_win_mask, np.array([6.0]), chi2_ratio=2.0, mad_cut=5.0
    )
    got_reasons = [m.reason for m in comb_ns.members]
    assert got_reasons == want_reasons
    assert set(got_reasons) >= {None, "window", "diverged", "chi2", "outlier"}
    assert [i for i, m in enumerate(comb_ns.members) if m.kept] == sorted(want_alive)

    # Sub-beam alignment in star=False brings im_shift onto im_base, shrinking spread sharply
    unaligned_std = float(np.max(np.std(np.stack([b_base, b_shift], axis=0), axis=0)))
    aligned_std = float(np.max(np.asarray(comb_ns.std)))
    assert aligned_std < 0.05 * unaligned_std

    record("mean_ncc", ncc_val)
    record("aligned_std_ratio", aligned_std / unaligned_std)
    record("n_kept", float(len(ens.kept)))
    assert ncc_val > 0.80


@pytest.mark.validates(
    "virgil.ensemble.GPFit",
    "virgil.ensemble.EvidenceGrid",
    "virgil.ensemble.evidence_grid",
    "virgil.ensemble.gp_members",
    roots=["mathematics"],
    headline="max_gpfit_logz_err",
)
def test_ensemble_gp_evidence_grid_and_gp_members():
    # Strongly elongated envelope (sigma_maj=3.8 mas, sigma_min=1.1 mas at PA 35 deg) so:
    # 1. best sigma hits the upper edge of gp_sigmas=(0.15, 0.30) and triggers gp_growths=1 -> 3 sigmas & 3 lengths;
    # 2. fitted Gaussian axis_ratio < gp_axis_ratio=0.95 triggers the "aniso" evidence grid!
    east, north = ref.sky_grid_mas(12, 1.5)
    pa = np.deg2rad(35.0)
    along = east * np.sin(pa) + north * np.cos(pa)
    across = east * np.cos(pa) - north * np.sin(pa)
    elong = np.exp(-0.5 * ((along / 3.8) ** 2 + (across / 1.1) ** 2))
    elong /= np.sum(elong)
    truth = vm.System(star=vm.PointSource(), env=vm.Image.from_brightness(elong, 1.5, flux=0.40))
    data = _vlti_like_oidata(truth, seed=73, n_telescopes=4, vis_err=0.01, cp_err_deg=0.5)

    draw = ve.Draw(
        0,
        family="gp",
        npix=8,
        pixel_scale_mas=1.6,
        start="flat",
        weights=(),
    )
    spec = ve.EnsembleSpec(
        gp_sigmas=(0.15, 0.30),
        gp_lengths=(1.0, 2.0),
        gp_growths=1,
        gp_window_nats=2.0,
        gp_axis_ratio=0.95,
        min_kept=1,
    )
    starts = ve.reference_starts(data, star=True, starts=("flat",))
    # Without grid growth (gp_growths=0), best sigma hits the upper boundary 0.30 -> interior is False
    grid_pinned = ve.evidence_grid(
        data,
        draw,
        star=True,
        starts=starts,
        spec=dataclasses.replace(spec, gp_growths=0, gp_axis_ratio=0.1),
        max_steps=300,
    )
    assert grid_pinned.interior is False

    grid = ve.evidence_grid(data, draw, star=True, starts=starts, spec=spec, max_steps=400)
    assert isinstance(grid, ve.EvidenceGrid)
    assert isinstance(grid.best, ve.GPFit)
    assert set(grid.log_evidence.keys()) == {"iso", "aniso"}
    assert len(grid.sigmas["iso"]) == 3 and len(grid.lengths_mas["iso"]) == 3
    assert grid.axis_ratio is not None and float(grid.axis_ratio) < 0.95
    assert grid.pa_deg is not None
    want_interior = (
        grid.best.sigma in grid.sigmas[grid.best.variant][1:-1]
        and grid.best.length_mas in grid.lengths_mas[grid.best.variant][1:-1]
    )
    assert grid.interior is bool(want_interior)

    # Bracketed 3x3 grid on a compact circular blob peaks strictly inside both axes -> interior is True
    blob_truth = vm.System(
        star=vm.PointSource(),
        env=vm.Image.from_brightness(
            ref.gaussian_blob_ref(8, 1.6, sigma_mas=2.2, dra=1.0), 1.6, flux=0.35
        ),
    )
    data_int = _vlti_like_oidata(blob_truth, seed=77, n_telescopes=4, vis_err=0.01, cp_err_deg=0.5)
    draw_int = ve.Draw(0, family="gp", npix=6, pixel_scale_mas=2.0, start="flat", weights=())
    spec_int = ve.EnsembleSpec(
        gp_sigmas=(0.05, 0.6, 4.0),
        gp_lengths=(0.2, 1.3, 6.0),
        gp_growths=0,
        gp_axis_ratio=0.1,
    )
    grid_int = ve.evidence_grid(
        data_int,
        draw_int,
        star=True,
        starts=ve.reference_starts(data_int, star=True, starts=("flat",)),
        spec=spec_int,
        max_steps=300,
    )
    assert grid_int.interior is True

    # Validate EVERY GPFit (both "iso" and "aniso") directly against our independent joint
    # (z, ln flux) Laplace reference in crosscheck.starlet_kkt_ref (zero virgil.imaging.log_evidence calls!)
    lp_lnf = -float(np.log(np.log(1e3 / 1e-4)))
    max_logz_err = 0.0
    variants_seen = set()
    for gf in grid.fits.values():
        variants_seen.add(gf.variant)
        m = gf.result.model
        z_m = np.asarray(m.env.log_brightness.latent)
        f_m = float(m.env.flux)
        r_m = np.asarray(vl.whitened_residuals(m, data))
        j_z = np.asarray(
            jax.jacrev(
                lambda zf: vl.whitened_residuals(
                    m.set("env.log_brightness.latent", zf.reshape(z_m.shape)), data
                )
            )(jnp.asarray(z_m.ravel()))
        )
        j_lnf = np.asarray(
            jax.jacrev(lambda lf: vl.whitened_residuals(m.set("env.flux", jnp.exp(lf)), data))(
                jnp.log(jnp.float64(f_m))
            )
        )[:, None]
        mk = ref.mackay_laplace_gp_ref(r_m, 0.0, z_m, j_z, extra_jac=j_lnf, extra_log_prior=lp_lnf)
        max_logz_err = max(max_logz_err, abs(float(gf.log_evidence) - float(mk["log_evidence"])))
    assert variants_seen == {"iso", "aniso"}

    gp_grp = ve.gp_members(data, draw, star=True, starts=starts, spec=spec, max_steps=400)
    assert gp_grp.curve is None
    assert len(gp_grp.gp) >= 1
    for gfit in gp_grp.gp:
        assert float(grid.best.log_evidence) - float(gfit.log_evidence) <= spec.gp_window_nats + 1e-9

    comb = ve.combine(data, [gp_grp], spec=spec, star=True)
    assert len(comb.kept) >= 1

    record("max_gpfit_logz_err", max_logz_err)
    record("fitted_axis_ratio", float(grid.axis_ratio))
    record("n_gp_members", float(len(gp_grp.gp)))
    assert max_logz_err < 1e-9


@pytest.mark.validates(
    "virgil.ensemble.combine",
    roots=["mathematics"],
    kind="control",
    headline="flipped_member_rejected",
)
def test_ensemble_combine_rejects_corrupted_member_from_running_mean_control():
    n, scale = 9, 1.5
    truth_blob = ref.gaussian_blob_ref(n, scale, sigma_mas=1.4, dra=3.0, ddec=1.5)
    truth = vm.System(star=vm.PointSource(), env=vm.Image.from_brightness(truth_blob, scale, flux=0.35))
    rng = np.random.default_rng(88)
    u_m = np.array([15.0, 35.0, 60.0, -25.0, 45.0, 80.0, 20.0, -55.0, 70.0, 30.0, -40.0, 95.0])
    v_m = np.array([20.0, -15.0, 40.0, 50.0, 30.0, -20.0, -65.0, 25.0, 55.0, 75.0, -45.0, 10.0])
    wavel = np.full_like(u_m, 2.2e-6)
    v2_true = np.abs(np.asarray(truth.model(u_m, v_m, wavel))) ** 2
    d_v2 = np.full_like(v2_true, 0.005)
    data = vo.OIData(
        {
            "u": u_m,
            "v": v_m,
            "wavel": wavel,
            "vis": v2_true + rng.normal(0.0, d_v2),
            "d_vis": d_v2,
            "vis_mode": "v2",
        }
    )

    draw = ve.Draw(0, family="tsv", npix=n, pixel_scale_mas=scale, start="moments", weights=(4.0, 1.5, 0.5))
    pos_img = vm.Image.from_brightness(truth_blob, scale, flux=0.35)
    neg_img = vm.Image.from_brightness(truth_blob[::-1, ::-1], scale, flux=0.35)

    def make_group(env_img, slight_offset: float):
        m = vm.System(star=vm.PointSource(), env=env_img)
        c2 = float(np.sum(np.asarray(vl.whitened_residuals(m, data)) ** 2)) + slight_offset
        n_d = int(data.n_independent)
        c2_arr = np.array([c2 + 4.0, c2 + 0.5, c2])
        pen_arr = np.array([0.05, 0.12, 0.90])
        results = [
            vfit.FitResult(
                model=m,
                values={"env.log_brightness": env_img.log_brightness, "env.flux": jnp.asarray(0.35)},
                info={"chi2": [float(val_c2)], "ndata": [n_d]},
            )
            for val_c2 in c2_arr
        ]
        curve = vi.LCurve(
            weights=np.asarray(draw.weights, dtype=float),
            chi2=c2_arr,
            chi2_red=(c2_arr / n_d)[:, None],
            penalty=pen_arr,
            results=results,
        )
        return ve.Group(draw=draw, curve=curve)

    grp_pos = make_group(pos_img, 0.0)
    grp_neg = make_group(neg_img, 1e-4)
    spec = ve.EnsembleSpec(window_dex=1.0, chi2_ratio=2.0, mad_cut=5.0, mean_rtol=0.15, min_kept=1)
    comb = ve.combine(data, [grp_pos, grp_neg], spec=spec, star=True)

    neg_reasons = [m.reason for m in comb.members[3:]]
    rejected = float("mean" in neg_reasons and all(not m.kept for m in comb.members[3:]))
    record("flipped_member_rejected", rejected)
    assert "mean" in neg_reasons
    assert all(not m.kept for m in comb.members[3:])


# ---------------------------------------------------------------------------
# virgil.radio: SigmaScale, ComplexVisData.sigma_scale, fit_staged_nuisances, StagedFit
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.radio.SigmaScale",
    "virgil.radio.ComplexVisData.sigma_scale",
    roots=["mathematics", "statistics"],
    headline="max_sigma_ref_diff",
)
def test_radio_sigma_scale_unbiased_on_white_noise_and_grouping():
    rng = np.random.default_rng(104)
    true_scales = [1.0, 1.65, 0.75]
    ddid_obs_map = [0, 0, 1]
    u_all, v_all, vis_all, w_all, freq_all = [], [], [], [], []
    ddid_all, obs_all, time_all, a1_all, a2_all, chan_all, scan_all = [], [], [], [], [], [], []

    for d_idx, (s_true, o_id) in enumerate(zip(true_scales, ddid_obs_map)):
        for scan_id in (1, 2):
            for (a1, a2) in ((0, 1), (0, 2), (1, 2)):
                for ch in (0, 1):
                    n_t = 13
                    times = 60000.0 + scan_id * 100.0 + np.arange(n_t) * 6.0
                    perm = rng.permutation(n_t)
                    w_stream = rng.uniform(2.0e5, 8.0e5, size=n_t)
                    base_vis = (0.4 - 0.05 * a2) * np.exp(1j * (0.3 * a1 - 0.2 * ch))
                    sigma_stream = s_true / np.sqrt(w_stream)
                    v_stream = base_vis + sigma_stream * (rng.normal(size=n_t) + 1j * rng.normal(size=n_t))

                    u_all.append(np.full(n_t, 25.0 * (a2 - a1)))
                    v_all.append(np.full(n_t, 15.0 * (a1 + a2)))
                    vis_all.append(v_stream[perm])
                    w_all.append(w_stream[perm])
                    freq_all.append(np.full(n_t, 230.0e9 + ch * 1.0e8))
                    ddid_all.append(np.full(n_t, 10 + d_idx, dtype=int))
                    obs_all.append(np.full(n_t, o_id, dtype=int))
                    time_all.append(times[perm])
                    a1_all.append(np.full(n_t, a1, dtype=int))
                    a2_all.append(np.full(n_t, a2, dtype=int))
                    chan_all.append(np.full(n_t, ch, dtype=int))
                    scan_all.append(np.full(n_t, scan_id, dtype=int))

    data = vr.ComplexVisData.from_arrays(
        np.concatenate(u_all),
        np.concatenate(v_all),
        np.concatenate(vis_all),
        np.concatenate(w_all),
        np.concatenate(freq_all),
        ddid=np.concatenate(ddid_all),
        obs=np.concatenate(obs_all),
        time=np.concatenate(time_all),
        ant1=np.concatenate(a1_all),
        ant2=np.concatenate(a2_all),
        chan=np.concatenate(chan_all),
        scan=np.concatenate(scan_all),
    )

    by_ddid = data.sigma_scale(by="ddid")
    by_obs = data.sigma_scale(by="obs")
    pooled = data.sigma_scale(by=None)
    assert isinstance(by_ddid, vr.SigmaScale)
    assert by_ddid.labels == (10, 11, 12)
    assert by_obs.labels == (0, 1)
    assert pooled.labels == (None,)

    def _compare_group(got_obj: vr.SigmaScale, idx: int, mask: np.ndarray) -> float:
        want = ref.radio_sigma_scale_group_ref(
            np.asarray(data.time)[mask],
            np.asarray(data.ant1)[mask],
            np.asarray(data.ant2)[mask],
            np.asarray(data.chan)[mask],
            np.asarray(data.scan)[mask],
            np.asarray(data.ddid)[mask],
            np.asarray(data.vis)[mask],
            np.asarray(data.weight)[mask],
        )
        assert int(got_obj.n_pairs[idx]) == want["n_pairs"]
        return max(
            abs(float(got_obj.s[idx]) - float(want["s"])),
            abs(float(got_obj.s_se[idx]) - float(want["s_se"])),
            abs(float(got_obj.s_re[idx]) - float(want["s_re"])),
            abs(float(got_obj.s_im[idx]) - float(want["s_im"])),
            abs(float(got_obj.s_re_se[idx]) - float(want["s_re_se"])),
            abs(float(got_obj.s_im_se[idx]) - float(want["s_im_se"])),
            abs(float(got_obj.lag_ratio[idx]) - float(want["lag_ratio"])),
        )

    max_diff = 0.0
    for g_idx, true_s in enumerate(true_scales):
        mask = np.asarray(data.ddid) == g_idx
        assert int(by_ddid.n_pairs[g_idx]) == 72
        max_diff = max(max_diff, _compare_group(by_ddid, g_idx, mask))
        assert abs(float(by_ddid.s[g_idx]) - true_s) < 3.0 * float(by_ddid.s_se[g_idx])
        assert 0.65 < float(by_ddid.lag_ratio[g_idx]) < 1.45

    for o_idx, want_np in ((0, 144), (1, 72)):
        mask_obs = np.asarray(data.obs) == o_idx
        assert int(by_obs.n_pairs[o_idx]) == want_np
        max_diff = max(max_diff, _compare_group(by_obs, o_idx, mask_obs))

    assert int(pooled.n_pairs[0]) == 216
    max_diff = max(max_diff, _compare_group(pooled, 0, np.ones(np.asarray(data.vis).shape, dtype=bool)))

    no_meta = vr.ComplexVisData.from_arrays(
        np.zeros(4), np.zeros(4), np.ones(4, dtype=complex), np.ones(4), np.full(4, 230e9)
    )
    with pytest.raises((ValueError, TypeError)):
        no_meta.sigma_scale()

    record("max_sigma_ref_diff", max_diff)
    record("ddid1_s_z", abs(float(by_ddid.s[1]) - 1.65) / float(by_ddid.s_se[1]))
    assert max_diff < 1e-12


@pytest.mark.validates(
    "virgil.radio.SigmaScale",
    "virgil.radio.ComplexVisData.sigma_scale",
    roots=["statistics"],
    kind="control",
    headline="drift_lag_ratio",
)
def test_radio_sigma_scale_smooth_drift_upper_bound_and_lag_ratio_control():
    rng = np.random.default_rng(109)
    n_t = 40
    times = np.arange(n_t, dtype=float) * 6.0
    weight = np.full(n_t, 1.0e6, dtype=float)
    phase = 0.08 * np.arange(n_t, dtype=float)
    vis = 0.25 * np.exp(1j * phase) + 1.0e-3 * (rng.normal(size=n_t) + 1j * rng.normal(size=n_t))
    data = vr.ComplexVisData.from_arrays(
        np.full(n_t, 50.0),
        np.full(n_t, -30.0),
        vis,
        weight,
        np.full(n_t, 230e9),
        time=times,
        ant1=np.zeros(n_t, dtype=int),
        ant2=np.ones(n_t, dtype=int),
        chan=np.zeros(n_t, dtype=int),
        scan=np.ones(n_t, dtype=int),
    )
    res = data.sigma_scale()
    lag = float(res.lag_ratio[0])
    s_hat = float(res.s[0])

    record("drift_lag_ratio", lag)
    record("drift_s_hat", s_hat)
    assert s_hat > 5.0
    assert lag > 5.0


@pytest.mark.validates(
    "virgil.radio.fit_staged_nuisances",
    "virgil.radio.StagedFit",
    roots=["mathematics"],
    headline="max_gain_err",
)
def test_radio_fit_staged_nuisances_recovers_gains_ref_obs_and_box_edge_guard():
    rng = np.random.default_rng(121)
    n, scale = 6, 35.0
    shape = ref.gaussian_blob_ref(n, scale, sigma_mas=45.0, dra=10.0, ddec=-8.0)
    npts = 180
    max_q = 0.3 / (2.0 * ref.MAS_TO_RAD * scale)
    radius = max_q * np.sqrt(rng.uniform(0.04, 1.0, npts))
    ang = rng.uniform(0.0, 2.0 * np.pi, npts)
    u, v = radius * np.cos(ang), radius * np.sin(ang)
    obs = np.repeat(np.arange(3), 60)
    weight = np.repeat([300.0, 4000.0, 1200.0], 60)
    nominal = vr.ComplexVisData.from_arrays(u, v, np.ones(npts, dtype=complex), weight, np.full(npts, 230e9), obs=obs)
    fixed_nuis = vr.CalibrationNuisance(nominal, gains=False, offsets=False, alpha=False, sigmas=False, fit_flux=False)

    def scene(values):
        return vm.LinearImage(values["pixels"], scale, ft="dft", parametrization="linear", reference_flux=1.0)

    truth_vals = {"pixels": jnp.asarray(shape, dtype=float)}
    model_vis = np.asarray(nominal.model(scene(truth_vals), truth_vals, fixed_nuis))
    true_gains = np.array([1.18, 1.0, 0.84])
    vis = true_gains[obs] * model_vis + (rng.normal(size=npts) + 1j * rng.normal(size=npts)) / np.sqrt(weight)
    radio = vr.ComplexVisData.from_arrays(u, v, vis, weight, np.full(npts, 230e9), obs=obs)

    box_lin = dist.Uniform(jnp.zeros((n, n)), jnp.full((n, n), 2.0))
    staged = vr.fit_staged_nuisances(
        radio,
        scene,
        {"pixels": box_lin},
        [vi.TSV(50.0, beam_mas=scale)],
        init={"pixels": jnp.full((n, n), 1.0 / (n * n))},
        nuisance=dict(offsets=False, sigmas=False, alpha=False, fit_flux=False),
        bounds={"gain": (0.7, 1.3)},
        weight=50.0,
        max_steps=1500,
        gtol=8e-7,
    )
    assert isinstance(staged, vr.StagedFit)
    assert len(staged.stages) == 3
    assert all(s.info["stop"] is None for s in staged.stages)
    assert staged.ref_obs == 1
    assert np.allclose(staged.sum_weights, [1.8e4, 2.4e5, 7.2e4])
    assert staged.failed is False
    assert not np.any(staged.edge["cal.gain"])

    # Image block KKT certificate converges cleanly, and raw nuisance stationarity holds (< 5e-3)
    # as well as relative stationarity against nominal calibration (g = 1, flat pixels): grad_rel < NUISANCE_TOL.
    # (The Stage-3 start= bug that inflates staged.kkt['nuisances']['cal.gain']['grad_rel'] is pinned in F29 below.)
    img_kkt = staged.kkt["images"]["."]
    assert img_kkt["kind"] == "linear"
    assert float(img_kkt["pg_rel"]) < float(vfit.KKT_TOL)
    assert float(img_kkt["zero_violation"]) == 0.0
    assert float(img_kkt["kkt_w"]) < float(vfit.KKT_W_TOL)
    nuis_kkt = staged.kkt["nuisances"]["cal.gain"]
    assert float(nuis_kkt["grad_norm"]) < 5e-3

    nom_nuis = vr.CalibrationNuisance(
        radio, ref_obs=1, offsets=False, sigmas=False, alpha=False, fit_flux=False
    )

    def _gain_loss(lin_g, pix):
        vals = {"pixels": pix, "cal.gain": lin_g}
        m_vis = radio.model(scene(vals), vals, nom_nuis)
        diff = vis - m_vis
        return 0.5 * jnp.sum(weight * (diff.real**2 + diff.imag**2))

    g_cur_lin = np.asarray(
        jax.grad(_gain_loss)(
            jnp.asarray(staged.values["cal.gain"], dtype=float),
            jnp.asarray(staged.values["pixels"], dtype=float),
        )
    )
    g_nom_lin = np.asarray(
        jax.grad(_gain_loss)(jnp.ones(2, dtype=float), jnp.full((n, n), 1.0 / (n * n), dtype=float))
    )
    nom_nuis_ref = ref.kkt_nuisance_grad_rel_ref(
        g_cur_lin,
        g_nom_lin,
        value_cur=np.asarray(staged.values["cal.gain"], dtype=float),
        value_start=np.ones(2, dtype=float),
        lower=0.7,
        upper=1.3,
    )
    assert abs(float(nuis_kkt["grad_norm"]) - nom_nuis_ref["grad_norm"]) < 1e-11
    assert nom_nuis_ref["grad_rel"] < float(vfit.NUISANCE_TOL)

    # Verify linear constrained values["cal.gain"] AND log-space box metadata in nuisances["cal.gain"]
    linear_gains = np.asarray(staged.values["cal.gain"], dtype=float)
    nuis_g = staged.nuisances["cal.gain"]
    log_g = np.asarray(nuis_g["value"], dtype=float)
    lo_g = float(np.asarray(nuis_g["lower"]).ravel()[0])
    hi_g = float(np.asarray(nuis_g["upper"]).ravel()[0])
    want_edge_frac = np.minimum(log_g - np.log(0.7), np.log(1.3) - log_g) / np.log(1.3 / 0.7)

    max_gain_err = float(np.max(np.abs(linear_gains - np.array([1.18, 0.84]))))
    log_box_err = max(
        float(np.max(np.abs(log_g - np.log(linear_gains)))),
        abs(lo_g - float(np.log(0.7))),
        abs(hi_g - float(np.log(1.3))),
        float(np.max(np.abs(np.asarray(nuis_g["edge_frac"], dtype=float) - want_edge_frac))),
    )

    record("max_gain_err", max_gain_err)
    record("log_box_metadata_err", log_box_err)
    record("nominal_nuisance_grad_rel", nom_nuis_ref["grad_rel"])
    record("image_kkt_w", float(img_kkt["kkt_w"]))
    record("min_edge_frac", float(np.min(nuis_g["edge_frac"])))
    assert max_gain_err < 0.02
    assert log_box_err < 1e-12
    assert "pixels" in staged.values


@pytest.mark.validates(
    "virgil.radio.fit_staged_nuisances",
    "virgil.radio.StagedFit",
    roots=["mathematics"],
    kind="finding",
    headline="nuisance_grad_rel",
)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "F29: fit_staged_nuisances passes Stage 2's already-minimized calibration "
        "nuisances instead of nominal calibration (g = 1) as start= to Stage 3's "
        "kkt_report, inflating nuisance grad_rel >> NUISANCE_TOL and forcing "
        "kkt['converged'] = False at a stationary joint solution."
    ),
)
def test_f29_fit_staged_nuisances_stage3_kkt_converged_at_joint_optimum():
    rng = np.random.default_rng(121)
    n, scale = 6, 35.0
    shape = ref.gaussian_blob_ref(n, scale, sigma_mas=45.0, dra=10.0, ddec=-8.0)
    npts = 180
    max_q = 0.3 / (2.0 * ref.MAS_TO_RAD * scale)
    radius = max_q * np.sqrt(rng.uniform(0.04, 1.0, npts))
    ang = rng.uniform(0.0, 2.0 * np.pi, npts)
    u, v = radius * np.cos(ang), radius * np.sin(ang)
    obs = np.repeat(np.arange(3), 60)
    weight = np.repeat([300.0, 4000.0, 1200.0], 60)
    nominal = vr.ComplexVisData.from_arrays(
        u, v, np.ones(npts, dtype=complex), weight, np.full(npts, 230e9), obs=obs
    )
    fixed_nuis = vr.CalibrationNuisance(
        nominal, gains=False, offsets=False, alpha=False, sigmas=False, fit_flux=False
    )

    def scene(values):
        return vm.LinearImage(
            values["pixels"], scale, ft="dft", parametrization="linear", reference_flux=1.0
        )

    truth_vals = {"pixels": jnp.asarray(shape, dtype=float)}
    model_vis = np.asarray(nominal.model(scene(truth_vals), truth_vals, fixed_nuis))
    vis = np.array([1.18, 1.0, 0.84])[obs] * model_vis + (
        rng.normal(size=npts) + 1j * rng.normal(size=npts)
    ) / np.sqrt(weight)
    radio = vr.ComplexVisData.from_arrays(u, v, vis, weight, np.full(npts, 230e9), obs=obs)

    box_lin = dist.Uniform(jnp.zeros((n, n)), jnp.full((n, n), 2.0))
    staged = vr.fit_staged_nuisances(
        radio,
        scene,
        {"pixels": box_lin},
        [vi.TSV(50.0, beam_mas=scale)],
        init={"pixels": jnp.full((n, n), 1.0 / (n * n))},
        nuisance=dict(offsets=False, sigmas=False, alpha=False, fit_flux=False),
        bounds={"gain": (0.7, 1.3)},
        weight=50.0,
        max_steps=1500,
        gtol=8e-7,
    )
    nuis_grad_rel = float(staged.kkt["nuisances"]["cal.gain"]["grad_rel"])
    record("nuisance_grad_rel", nuis_grad_rel)
    assert nuis_grad_rel < float(vfit.NUISANCE_TOL)
    assert staged.kkt["converged"] is True


@pytest.mark.validates(
    "virgil.radio.fit_staged_nuisances",
    "virgil.radio.StagedFit",
    roots=["mathematics"],
    kind="control",
    headline="min_edge_frac",
)
def test_radio_fit_staged_nuisances_box_edge_failure_control():
    rng = np.random.default_rng(127)
    n, scale = 6, 35.0
    shape = ref.gaussian_blob_ref(n, scale, sigma_mas=45.0, dra=10.0, ddec=-8.0)
    npts = 180
    max_q = 0.3 / (2.0 * ref.MAS_TO_RAD * scale)
    radius = max_q * np.sqrt(rng.uniform(0.04, 1.0, npts))
    ang = rng.uniform(0.0, 2.0 * np.pi, npts)
    u, v = radius * np.cos(ang), radius * np.sin(ang)
    obs = np.repeat(np.arange(3), 60)
    weight = np.repeat([300.0, 4000.0, 1200.0], 60)
    nominal = vr.ComplexVisData.from_arrays(u, v, np.ones(npts, dtype=complex), weight, np.full(npts, 230e9), obs=obs)
    fixed_nuis = vr.CalibrationNuisance(nominal, gains=False, offsets=False, alpha=False, sigmas=False, fit_flux=False)

    def scene(values):
        return vm.LinearImage(values["pixels"], scale, ft="dft", parametrization="sqrt", reference_flux=1.0)

    truth_vals = {"pixels": jnp.sqrt(jnp.asarray(shape, dtype=float))}
    model_vis = np.asarray(nominal.model(scene(truth_vals), truth_vals, fixed_nuis))
    vis = np.array([1.48, 1.0, 0.89])[obs] * model_vis + (rng.normal(size=npts) + 1j * rng.normal(size=npts)) / np.sqrt(weight)
    radio = vr.ComplexVisData.from_arrays(u, v, vis, weight, np.full(npts, 230e9), obs=obs)

    box_sq = dist.Uniform(jnp.full((n, n), 1e-2), jnp.full((n, n), 3.0))
    staged = vr.fit_staged_nuisances(
        radio,
        scene,
        {"pixels": box_sq},
        [vi.TSV(50.0, beam_mas=scale)],
        init={"pixels": jnp.sqrt(jnp.full((n, n), 1.0 / (n * n)))},
        nuisance=dict(offsets=False, sigmas=False, alpha=False, fit_flux=False),
        bounds={"gain": (0.7, 1.3)},
        edge=0.05,
        weight=50.0,
        max_steps=800,
    )
    min_edge = float(np.min(staged.nuisances["cal.gain"]["edge_frac"]))
    record("min_edge_frac", min_edge)
    assert staged.failed is True
    assert bool(np.any(staged.edge["cal.gain"])) is True
    assert min_edge < 0.05
