"""Independent end-to-end validation of virgil 0.4.2 scriptable pipelines and
run-folder containers (``BinaryPipeline``, ``StarPipeline``, ``DiskPipeline``,
``ImagingPipeline``, ``OrbitPipeline``, ``Result``, ``load``, ``Check``,
``Stage``, ``SkyImage``).

Every synthetic dataset is generated independently via ``crosscheck.simulate``,
``crosscheck.oifits_writer``, ``crosscheck.sky`` and ``crosscheck.orbit_cells_ref``
without calling ``virgil``'s forward visibility models.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import shutil

import jax
import numpy as np
import pytest
from astropy.io import fits
from scipy import special, stats

from crosscheck import (
    array as carray,
    chi2 as ours_chi2,
    oifits_writer,
    orbit_cells_ref as ref,
    orbits as co,
    simulate,
    sky,
)
from evidence.plugin import record

pytest.importorskip("jaxoplanet")
pytest.importorskip("h5py")
pytest.importorskip("nbformat")
vp = pytest.importorskip("virgil.pipeline")

from virgil.aliases import AliasResult  # noqa: E402
from virgil.models import BinaryModelAngular, QuadraticLimbDarkenedDisk, System, UniformDisk  # noqa: E402
from virgil.oidata import OIData  # noqa: E402

pytestmark = pytest.mark.x64

UTS3 = np.array([[-9.925, -20.335], [14.887, 30.502], [103.306, 43.999]])
UTS4 = np.array([[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]])
WAVELENGTHS = np.linspace(1.9e-6, 2.3e-6, 4)
MAS2RAD = np.deg2rad(1.0 / 3.6e6)
T0 = 60000.0


def _shrink_oifits_errors(src, dst, factor: float):
    shutil.copy(src, dst)
    with fits.open(dst, mode="update") as hdul:
        hdul["OI_VIS2"].data["VIS2ERR"] /= factor
        if "OI_T3" in hdul:
            hdul["OI_T3"].data["T3PHIERR"] /= factor
    return dst


def _inclined_gaussian_ring_vis(r0_mas: float, sigma_mas: float, inc_deg: float, pa_deg: float, disk_flux: float):
    """Independent Fourier visibility of a central point star + inclined narrow Gaussian ring.

    On a thin ring deprojected by inclination ``i`` with major axis at PA ``theta``
    (North through East), spatial frequencies transform to in-plane radius
    ``q = sqrt(u_maj^2 + (u_min cos i)^2)`` where
    ``u_maj = u sin(theta) + v cos(theta)`` and ``u_min = u cos(theta) - v sin(theta)``.
    Radial Gaussian blurring of width ``sigma`` multiplies ``J_0(2 pi q r_0)`` by
    ``exp(-2 pi^2 q^2 sigma^2)`` up to ``O((sigma/r_0)^2)``.
    """
    th = math.radians(pa_deg)
    ci = math.cos(math.radians(inc_deg))
    r0_rad = r0_mas * MAS2RAD
    sig_rad = sigma_mas * MAS2RAD

    def vis(u, v, w):
        bx = u / w
        by = v / w
        u_maj = bx * math.sin(th) + by * math.cos(th)
        u_min = (bx * math.cos(th) - by * math.sin(th)) * ci
        q = np.hypot(u_maj, u_min)
        v_ring = special.j0(2.0 * np.pi * q * r0_rad) * np.exp(-2.0 * (np.pi * q * sig_rad) ** 2)
        return (1.0 + disk_flux * v_ring) / (1.0 + disk_flux)

    return vis


def _write_moving_binary_night(path, mjd_center: float, orbit_tuple, flux: float, seed: int):
    period, dt_peri, ecc, inc, omega, Omega, a_mas = orbit_tuple
    stations = np.column_stack([UTS4 * 0.3, np.zeros(len(UTS4))])
    wl = np.atleast_1d(np.asarray(WAVELENGTHS, np.float32)).astype(float)
    pairs = np.array(carray.baselines(4))
    tris = np.array(carray.triangles(4))
    tmat = ours_chi2.triangle_matrix(tris)
    rng = np.random.default_rng(seed)

    v2_rec = {k: [] for k in ("vis2", "u", "v", "sta", "mjd")}
    t3_rec = {k: [] for k in ("phi", "u1", "v1", "u2", "v2", "sta", "mjd")}
    for idx, ha in enumerate((-1.0, 0.0, 1.0)):
        u, v = carray.snapshot_uv(stations, ha, -45.0, -24.6276)
        mjd = float(mjd_center) + idx * 1e-5
        dra, ddec, _, _ = ref.sky_state(float(mjd_center) - T0, period, dt_peri, ecc, inc, omega, Omega, a_mas)
        x_k, y_k = float(dra[0]), float(ddec[0])

        def vis_at(uu, vv, x=x_k, y=y_k):
            uu2, vv2, ww2 = uu[:, None], vv[:, None], wl[None, :]
            return (sky.vis_point(uu2, vv2, ww2) + flux * sky.vis_point(uu2, vv2, ww2, x, y)) / (1.0 + flux)

        v2_clean = np.abs(vis_at(u, v)) ** 2
        v2_rec["vis2"].append(v2_clean + 0.008 * rng.standard_normal(v2_clean.shape))
        v2_rec["u"].append(u)
        v2_rec["v"].append(v)
        v2_rec["sta"].append(pairs)
        v2_rec["mjd"].append(np.full(len(u), mjd))

        cp, u1, v1, u2, v2_t3 = carray.closure_phase(vis_at, u, v, 4)
        bl_noise = (0.5 / math.sqrt(3.0)) * rng.standard_normal((len(u), len(wl)))
        t3_rec["phi"].append(np.rad2deg(cp) + (tmat @ bl_noise))
        t3_rec["u1"].append(u1)
        t3_rec["v1"].append(v1)
        t3_rec["u2"].append(u2)
        t3_rec["v2"].append(v2_t3)
        t3_rec["sta"].append(tris)
        t3_rec["mjd"].append(np.full(len(u1), mjd))

    v2 = {k: np.concatenate(val) for k, val in v2_rec.items()}
    t3 = {k: np.concatenate(val) for k, val in t3_rec.items()}
    v2["err"] = np.full(v2["vis2"].shape, 0.008)
    t3["err"] = np.full(t3["phi"].shape, 0.5)
    oifits_writer.write(path, wavelengths=wl, stations_xyz=stations, vis2=v2, t3=t3)
    return path


# ---------------------------------------------------------------------------
# 1. Pipeline containers: Check and Stage serialization/execution plumbing
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.pipeline.Check",
    "virgil.pipeline.Stage",
    roots=["self-consistency"],
    headline="dict_roundtrip_err",
)
def test_pipeline_check_and_stage_containers():
    """Check round-trips through JSON dicts across statuses and Stage executes its callable contract."""
    roundtrip_errs = []
    for name, status, val, thr in (
        ("chi2_companion", "pass", 1.04, 2.0),
        ("chi2_companion", "warn", 2.75, 2.0),
        ("ess", "fail", 42.5, 100.0),
    ):
        chk = vp.Check(
            name=name,
            status=status,
            value=val,
            threshold=thr,
            message=f"{name}={val} vs {thr}.",
        )
        rebuilt = vp.Check.from_dict(json.loads(json.dumps(chk.to_dict())))
        assert rebuilt == chk
        roundtrip_errs.append(abs(float(rebuilt.value) - val))
    dict_roundtrip_err = float(max(roundtrip_errs))

    called = {}
    stg = vp.Stage(name="overview", run=lambda p: called.setdefault("ran", p), outputs=("summary.json",))
    assert stg.name == "overview"
    assert stg.outputs == ("summary.json",)
    assert stg.run("sentinel") == "sentinel"

    record("dict_roundtrip_err", dict_roundtrip_err)
    assert dict_roundtrip_err < 1e-15


# ---------------------------------------------------------------------------
# 2. BinaryPipeline + Result + load + Check + Stage + quality-check control
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.pipeline.BinaryPipeline",
    "virgil.pipeline.Result",
    "virgil.pipeline.load",
    "virgil.pipeline.Check",
    "virgil.pipeline.Stage",
    roots=["mathematics", "standards"],
    headline="sep_err_mas",
)
def test_binary_pipeline_recovery_resume_and_load_roundtrip(tmp_path):
    """BinaryPipeline recovers synthetic binary astrometry/flux, caches completed stages on resume, and passes NUTS checks."""
    true_sep = 6.2
    true_pa = 54.0
    true_flux = 0.18
    dra_true = true_sep * math.sin(math.radians(true_pa))
    ddec_true = true_sep * math.cos(math.radians(true_pa))

    oifits_path = tmp_path / "binary.fits"
    simulate.observe(
        oifits_path,
        lambda u, v, w: (sky.vis_point(u, v, w) + true_flux * sky.vis_point(u, v, w, dra_true, ddec_true))
        / (1.0 + true_flux),
        UTS4,
        hour_angles_h=np.linspace(-2.5, 2.5, 5),
        wavelengths=WAVELENGTHS,
        sigma_v2=0.008,
        sigma_cp_deg=0.5,
        rng=np.random.default_rng(701),
        phase_noise="baseline",
    )

    run_dir = tmp_path / "run_binary"
    common_kwargs = dict(
        detection_sigma=2.0,
        n_null=50,
        max_grid=25,
        n_flux=12,
        limit_bins=6,
        num_warmup=250,
        num_samples=300,
        num_chains=2,
        seed=1,
        quicklook_execute=False,
    )
    pipe_part = vp.BinaryPipeline.from_oifits(oifits_path, output=run_dir, **common_kwargs)
    res_part = pipe_part.run(through="search")
    assert res_part.status == "partial"
    assert set(res_part.run["stages"]) == {"load", "overview", "search"}
    started_part = {s: info["started"] for s, info in res_part.run["stages"].items()}

    # Resuming with altered configuration must raise ConfigMismatchError (stage config-hash invalidation)
    pipe_bad = vp.BinaryPipeline.from_oifits(oifits_path, output=run_dir, **(common_kwargs | {"sigma": 4.5}))
    with pytest.raises(vp.ConfigMismatchError):
        pipe_bad.run(resume=True)

    # Resuming with identical configuration via from_config skips completed upstream stages and runs downstream ones
    cfg_path = tmp_path / "binary_cfg.json"
    cfg_path.write_text(
        json.dumps({"data": [str(oifits_path)], "output": str(run_dir), "settings": common_kwargs}),
        encoding="utf-8",
    )
    res_full = vp.BinaryPipeline.from_config(cfg_path).run(resume=True)
    assert res_full.status == "complete"
    for s_done, ts_done in started_part.items():
        assert res_full.run["stages"][s_done]["started"] == ts_done
    assert set(res_full.run["stages"]) >= {"load", "overview", "search", "limits", "fit", "posterior", "quicklook"}

    loaded = vp.load(run_dir)
    assert isinstance(loaded, vp.Result)
    assert loaded.status == "complete"
    assert loaded.run["schema"] == "virgil-pipeline-run-v1"
    for rel_p, sha_hex in loaded.run["files"].items():
        assert hashlib.sha256((run_dir / rel_p).read_bytes()).hexdigest() == sha_hex
    with fits.open(oifits_path) as h_in, fits.open(run_dir / "data" / "processed.oifits") as h_proc:
        assert np.allclose(h_in["OI_VIS2"].data["VIS2DATA"], h_proc["OI_VIS2"].data["VIS2DATA"], atol=1e-12)
        assert np.allclose(h_in["OI_T3"].data["T3PHI"], h_proc["OI_T3"].data["T3PHI"], atol=1e-12)

    assert isinstance(loaded.data(), OIData)
    assert isinstance(loaded.model(), BinaryModelAngular)
    assert len(loaded.checks) >= 1 and all(isinstance(c, vp.Check) for c in loaded.checks)
    assert "sep_mas" in loaded.summary["companion"]
    assert len(loaded.describe()) > 0
    assert "axes" in loaded.grid()
    samples = loaded.samples()
    assert len(samples) > 0
    assert np.asarray(samples["flux"]).size == 600
    assert len(loaded.sample_stats()) > 0
    assert loaded.fit_result() is not None

    comp = loaded.summary["companion"]
    sep_err_mas = abs(float(comp["sep_mas"]) - true_sep)
    pa_diff = abs(((float(comp["pa_deg"]) - true_pa + 180.0) % 360.0) - 180.0)
    dra_err = abs(float(comp["dra_mas"]) - dra_true)
    ddec_err = abs(float(comp["ddec_mas"]) - ddec_true)
    flux_rel_err = abs(float(comp["flux"]) - true_flux) / true_flux
    contrast_rel_err = abs(float(comp["contrast"]) - (1.0 / true_flux)) / (1.0 / true_flux)
    dmag_err = abs(float(comp["delta_mag"]) - (-2.5 * math.log10(true_flux)))

    # Independent Check value verification: correlated reduced chi2 and global Monte Carlo significance floor
    d_raw = ours_chi2.load(oifits_path)
    best_vals = loaded.model_values()
    sep_fit, pa_fit, flux_fit = float(best_vals["sep"]), float(best_vals["pa"]), float(best_vals["flux"])
    dra_fit = sep_fit * math.sin(math.radians(pa_fit))
    ddec_fit = sep_fit * math.cos(math.radians(pa_fit))
    ref_chi2 = ours_chi2.chi2(
        d_raw,
        lambda u, v, w: (sky.vis_point(u, v, w) + flux_fit * sky.vis_point(u, v, w, dra_fit, ddec_fit)) / (1.0 + flux_fit),
        correlated=True,
        sine=True,
    )
    ref_chi2_red = float(ref_chi2) / float(loaded.summary["chi2"]["n_independent"])
    c_chi2 = next(c for c in loaded.checks if c.name == "chi2")
    c_det = next(c for c in loaded.checks if c.name == "detection")
    det_val = float(c_det.value[-1]) if isinstance(c_det.value, (list, tuple)) else float(c_det.value)
    check_chi2_err = abs(float(c_chi2.value) - ref_chi2_red)
    check_det_err = max(
        abs(det_val - float(stats.norm.isf(1.0 / (common_kwargs["n_null"] + 1)))),
        abs(float(loaded.summary["search"]["global_nsigma"]) - float(stats.norm.isf(1.0 / (common_kwargs["n_null"] + 1)))),
    )

    by_name = {c.name: c.status for c in loaded.checks}

    record("sep_err_mas", sep_err_mas)
    record("pa_err_deg", pa_diff)
    record("flux_rel_err", flux_rel_err)
    record("contrast_rel_err", contrast_rel_err)
    record("dmag_err", dmag_err)
    record("check_chi2_err", check_chi2_err)
    assert sep_err_mas < 0.15
    assert pa_diff < 2.0
    assert dra_err < 0.15
    assert ddec_err < 0.15
    assert flux_rel_err < 0.05
    assert contrast_rel_err < 0.05
    assert dmag_err < 0.06
    assert check_chi2_err < 5e-3
    assert check_det_err < 1e-5
    assert by_name["detection"] == "pass"
    assert by_name["chi2"] == "pass"
    for k in ("r_hat", "ess", "divergences", "prior_bound"):
        assert by_name[k] != "fail"
    assert loaded.summary["worst_check"] != "fail"
    jax.clear_caches()
    gc.collect()


@pytest.mark.validates("virgil.pipeline.BinaryPipeline", roots=["mathematics"], kind="control")
def test_binary_pipeline_nondetection_and_bad_chi2_checks_control(tmp_path):
    """Mismatched calibrator transfer function (V^2 > 1, zero closure phase) fails both detection and chi2 checks."""
    raw = tmp_path / "null_star.fits"
    simulate.observe(
        raw,
        lambda u, v, w: sky.vis_point(u, v, w),
        UTS3,
        hour_angles_h=[-2.0, 0.0, 2.0],
        wavelengths=WAVELENGTHS,
        sigma_v2=0.015,
        sigma_cp_deg=0.5,
        rng=np.random.default_rng(702),
        phase_noise="triangle",
    )
    bad = tmp_path / "null_star_overcal.fits"
    shutil.copy(raw, bad)
    with fits.open(bad, mode="update") as hdul:
        # Passive sky binaries satisfy |V|^2 <= 1, so V^2 ~ 1.35 peaks at f = 0 (no companion) while failing chi2
        hdul["OI_VIS2"].data["VIS2DATA"] = np.full_like(hdul["OI_VIS2"].data["VIS2DATA"], 1.35)
        hdul["OI_VIS2"].data["VIS2ERR"] = np.full_like(hdul["OI_VIS2"].data["VIS2ERR"], 0.02)
    res = vp.BinaryPipeline.from_oifits(
        bad,
        output=tmp_path / "run_bin_ctrl",
        detection_sigma=2.0,
        n_null=50,
        max_grid=15,
        n_flux=8,
        limit_bins=5,
        num_warmup=20,
        num_samples=20,
        num_chains=2,
        quicklook_execute=False,
    ).run()
    by_name = {c.name: c.status for c in res.checks}
    chi2_chk = next(c for c in res.checks if c.name == "chi2")
    record("chi2_check_value", float(chi2_chk.value))
    assert by_name["detection"] == "warn"
    assert by_name["chi2"] == "fail"


# ---------------------------------------------------------------------------
# 3. StarPipeline + quality-check control
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.pipeline.StarPipeline",
    roots=["mathematics"],
    headline="diam_err_mas",
)
def test_star_pipeline_recovers_uniform_disk_angular_diameter(tmp_path):
    """StarPipeline fits both uniform and limb-darkened stellar models and passes chi2 checks."""
    true_diam = 3.4
    star_path = tmp_path / "ud_star.fits"
    simulate.observe(
        star_path,
        lambda u, v, w: sky.vis_uniform_disk(u, v, w, true_diam),
        UTS4,
        hour_angles_h=np.linspace(-2.5, 2.5, 5),
        wavelengths=WAVELENGTHS,
        sigma_v2=0.006,
        sigma_cp_deg=0.4,
        rng=np.random.default_rng(710),
        phase_noise="baseline",
    )
    run_dir = tmp_path / "run_star"
    vp.StarPipeline.from_oifits(
        star_path,
        model=None,
        output=run_dir,
        n_scan=250,
        max_lobes=3,
        num_warmup=250,
        num_samples=300,
        num_chains=2,
        seed=3,
        quicklook_execute=False,
    ).run()
    loaded = vp.load(run_dir)
    diam_fit = float(loaded.summary["star"]["diam_mas"])
    diam_err_mas = abs(diam_fit - true_diam)
    by_name = {c.name: c.status for c in loaded.checks}

    assert isinstance(loaded.model("uniform"), UniformDisk)
    assert isinstance(loaded.model("limb_darkened"), QuadraticLimbDarkenedDisk)
    vals_ud = loaded.model_values("uniform")
    vals_ld = loaded.model_values("limb_darkened")
    ld_diam_err_mas = abs(float(vals_ld["diam"]) - true_diam)
    q1 = float(vals_ld["q1"])
    q2 = float(vals_ld["q2"])
    # Kipping (2013) triangular quadratic limb-darkening coefficients u1, u2
    u1 = 2.0 * math.sqrt(max(q1, 0.0)) * q2
    u2 = math.sqrt(max(q1, 0.0)) * (1.0 - 2.0 * q2)

    record("diam_err_mas", diam_err_mas)
    record("ld_diam_err_mas", ld_diam_err_mas)
    assert diam_err_mas < 0.12
    assert abs(float(vals_ud["diam"]) - true_diam) < 0.12
    assert ld_diam_err_mas < 0.18
    assert 0.0 <= q1 <= 1.0 and 0.0 <= q2 <= 1.0
    assert 0.0 <= u1 <= 1.0 and 0.0 <= u1 + u2 <= 1.0
    assert by_name["chi2_uniform"] == "pass"
    assert by_name["chi2_limb_darkened"] == "pass"
    for k in ("r_hat", "ess", "divergences", "prior_bound"):
        assert by_name[k] != "fail"
    assert loaded.summary["worst_check"] != "fail"
    jax.clear_caches()
    gc.collect()


@pytest.mark.validates("virgil.pipeline.StarPipeline", roots=["mathematics"], kind="control")
def test_star_pipeline_mismatched_binary_chi2_check_control(tmp_path):
    """Fitting a high-SNR equal-flux binary with StarPipeline triggers the chi2_uniform failure check."""
    bin_path = tmp_path / "not_a_single_star.fits"
    simulate.observe(
        bin_path,
        lambda u, v, w: (sky.vis_point(u, v, w) + 0.6 * sky.vis_point(u, v, w, 5.0, -4.0)) / 1.6,
        UTS4,
        hour_angles_h=[-2.0, 0.0, 2.0],
        wavelengths=WAVELENGTHS,
        sigma_v2=0.005,
        sigma_cp_deg=0.4,
        rng=np.random.default_rng(711),
        phase_noise="triangle",
    )
    res = vp.StarPipeline.from_oifits(
        bin_path,
        model="uniform",
        output=tmp_path / "run_star_ctrl",
        n_scan=120,
        max_lobes=2,
        num_warmup=20,
        num_samples=20,
        num_chains=2,
        quicklook_execute=False,
    ).run()
    by_name = {c.name: c.status for c in res.checks}
    chi2_u = next(c for c in res.checks if c.name == "chi2_uniform")
    record("chi2_uniform_value", float(chi2_u.value))
    assert by_name["chi2_uniform"] == "fail"


# ---------------------------------------------------------------------------
# 4. DiskPipeline + quality-check control
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.pipeline.DiskPipeline",
    roots=["mathematics"],
    headline="ring_radius_rel_err",
)
def test_disk_pipeline_recovers_inclined_gaussian_ring(tmp_path):
    """DiskPipeline recovers ring geometry/flux/width ('rings') and RadialGP orientation/flux ('gp')."""
    true_r = 5.5
    true_sigma = 0.7
    true_flux = 0.8
    true_inc = 42.0
    true_pa = 68.0
    disk_path = tmp_path / "ring_disk.fits"
    simulate.observe(
        disk_path,
        _inclined_gaussian_ring_vis(true_r, sigma_mas=true_sigma, inc_deg=true_inc, pa_deg=true_pa, disk_flux=true_flux),
        UTS4,
        hour_angles_h=np.linspace(-3.0, 3.0, 6),
        wavelengths=WAVELENGTHS,
        sigma_v2=0.008,
        sigma_cp_deg=0.5,
        rng=np.random.default_rng(720),
        phase_noise="baseline",
    )
    run_dir = tmp_path / "run_disk"
    vp.DiskPipeline.from_oifits(
        disk_path,
        model="rings",
        output=run_dir,
        r_in_mas=1.5,
        r_out_mas=10.5,
        n_inc=6,
        n_pa=8,
        max_rings=2,
        posterior=False,
        quicklook_execute=False,
    ).run()
    loaded = vp.load(run_dir)
    assert isinstance(loaded.model(), System)
    disk_sum = loaded.summary["disk"]
    assert disk_sum["n_rings"] >= 1
    ring0 = disk_sum["rings"][0]
    r_fit = float(ring0["r_mas"])
    sig_fit = float(ring0["sigma_mas"])
    flux_fit = float(ring0["flux"])
    inc_fit = float(disk_sum["inc_deg"])
    pa_fit = float(disk_sum["pa_deg"])
    by_name = {c.name: c.status for c in loaded.checks}

    ring_radius_rel_err = abs(r_fit - true_r) / true_r
    ring_flux_rel_err = abs(flux_fit - true_flux) / true_flux
    ring_sigma_err = abs(sig_fit - true_sigma)
    inc_err_deg = abs(inc_fit - true_inc)
    pa_err_deg = abs(((pa_fit - true_pa + 90.0) % 180.0) - 90.0)

    # Default model="gp" branch: RadialGP + MacKay empirical-Bayes error_scale + laplace_cov
    run_gp = tmp_path / "run_disk_gp"
    vp.DiskPipeline.from_oifits(
        disk_path,
        model="gp",
        output=run_gp,
        r_in_mas=1.5,
        r_out_mas=10.5,
        n_inc=6,
        n_pa=8,
        posterior=False,
        quicklook_execute=False,
    ).run()
    loaded_gp = vp.load(run_gp)
    gp_sum = loaded_gp.summary["disk"]
    by_name_gp = {c.name: c.status for c in loaded_gp.checks}
    gp_flux_rel_err = abs(float(gp_sum["flux"]) - true_flux) / true_flux
    gp_inc_err_deg = abs(float(gp_sum["inc_deg"]) - true_inc)
    gp_pa_err_deg = abs(((float(gp_sum["pa_deg"]) - true_pa + 90.0) % 180.0) - 90.0)

    record("ring_radius_rel_err", ring_radius_rel_err)
    record("ring_flux_rel_err", ring_flux_rel_err)
    record("inc_err_deg", inc_err_deg)
    record("pa_err_deg", pa_err_deg)
    record("gp_flux_rel_err", gp_flux_rel_err)
    assert ring_radius_rel_err < 0.10
    assert ring_flux_rel_err < 0.18
    assert ring_sigma_err < 0.35
    assert inc_err_deg < 5.0
    assert pa_err_deg < 5.0
    assert by_name["chi2_reduced"] == "pass"
    assert gp_sum["model"] == "gp"
    assert gp_flux_rel_err < 0.25
    assert gp_inc_err_deg < 10.0
    assert gp_pa_err_deg < 10.0
    assert by_name_gp["chi2_reduced"] == "pass"
    assert by_name_gp["error_scale"] == "pass"


@pytest.mark.validates("virgil.pipeline.DiskPipeline", roots=["mathematics"], kind="control")
def test_disk_pipeline_corrupted_errors_check_control(tmp_path):
    """Strong asymmetric binary data with under-quoted errors trips DiskPipeline's chi2_reduced failure check."""
    raw = tmp_path / "disk_bad_raw.fits"
    simulate.observe(
        raw,
        lambda u, v, w: (sky.vis_point(u, v, w) + 0.7 * sky.vis_point(u, v, w, 4.5, 3.5)) / 1.7,
        UTS4,
        hour_angles_h=[-2.0, 0.0, 2.0],
        wavelengths=WAVELENGTHS,
        sigma_v2=0.03,
        sigma_cp_deg=4.0,
        rng=np.random.default_rng(721),
        phase_noise="triangle",
    )
    shrunk = _shrink_oifits_errors(raw, tmp_path / "disk_bad_shrunk.fits", factor=10.0)
    res = vp.DiskPipeline.from_oifits(
        shrunk,
        model="rings",
        output=tmp_path / "run_disk_ctrl",
        r_in_mas=1.5,
        r_out_mas=9.0,
        n_inc=4,
        n_pa=5,
        max_rings=1,
        posterior=False,
        quicklook_execute=False,
    ).run()
    by_name = {c.name: c.status for c in res.checks}
    chi2_r = next(c for c in res.checks if c.name == "chi2_reduced")
    record("chi2_reduced_value", float(chi2_r.value))
    assert by_name["chi2_reduced"] == "fail"


# ---------------------------------------------------------------------------
# 5. ImagingPipeline + member staging + quality-check control
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.pipeline.ImagingPipeline",
    "virgil.pipeline.SkyImage",
    roots=["mathematics", "standards"],
    headline="centroid_err_mas",
)
def test_imaging_pipeline_staged_members_and_skyimage_recovery(tmp_path):
    """ImagingPipeline reconstructs an off-axis source at its true world-coordinate centroid with standard FITS WCS."""
    dra_comp, ddec_comp = 4.2, 1.5
    img_oifits = tmp_path / "img_scene.fits"
    simulate.observe(
        img_oifits,
        lambda u, v, w: (
            sky.vis_point(u, v, w) + 0.35 * sky.vis_gaussian(u, v, w, 1.1, dra_comp, ddec_comp)
        )
        / 1.35,
        UTS4,
        hour_angles_h=np.linspace(-3.0, 3.0, 7),
        wavelengths=WAVELENGTHS,
        sigma_v2=0.008,
        sigma_cp_deg=0.5,
        rng=np.random.default_rng(730),
        phase_noise="baseline",
    )
    run_dir = tmp_path / "run_img"
    img_kwargs = dict(
        max_npix=16,
        field_mas=14.0,
        star=True,
        sigmas=[1.0, 2.0],
        lengths_beam=[1.0],
        members=2,
        max_steps=150,
        seed=5,
        quicklook_execute=False,
    )
    pipe = vp.ImagingPipeline.from_oifits(img_oifits, output=run_dir, **img_kwargs)
    pipe.run(through="hyperparameters")
    pipe.run_member(0)
    pipe.run_member(1)
    res = pipe.run(resume=True)

    assert res.status == "complete"
    assert (run_dir / "images" / "mean.fits").exists()
    assert (run_dir / "images" / "sigma.fits").exists()

    loaded = vp.load(run_dir)
    mean_img = loaded.image("mean")
    sig_img = loaded.image("sigma")
    assert isinstance(mean_img, vp.SkyImage)
    assert isinstance(sig_img, vp.SkyImage)
    assert np.all(np.isfinite(mean_img.pixels)) and np.all(mean_img.pixels >= -1e-12)

    # SkyImage world coordinates: row 0 is North (+dDec), col 0 is East (+dRA)
    n_row, n_col = mean_img.pixels.shape
    dra_cols = mean_img.dra + (0.5 * (n_col - 1) - np.arange(n_col)) * mean_img.pixel_scale_mas
    ddec_rows = mean_img.ddec + (0.5 * (n_row - 1) - np.arange(n_row)) * mean_img.pixel_scale_mas
    w_pix = np.maximum(mean_img.pixels, 0.0)
    w_tot = float(np.sum(w_pix))
    assert w_tot > 0.0
    dra_cent = float(np.sum(w_pix * dra_cols[None, :]) / w_tot)
    ddec_cent = float(np.sum(w_pix * ddec_rows[:, None]) / w_tot)

    # Verify FITS WCS standard header and bottom-up row orientation on images/mean.fits
    with fits.open(run_dir / "images" / "mean.fits") as hdul:
        fits_data = np.asarray(hdul[0].data, dtype=float)
        hdr = hdul[0].header
    assert hdr["CTYPE1"] == "RA---SIN" and hdr["CTYPE2"] == "DEC--SIN"
    assert hdr["CUNIT1"] == "deg" and hdr["CUNIT2"] == "deg"
    assert 0.0 < float(mean_img.header["STARFRAC"]) < 1.0
    deg_per_pix = float(mean_img.pixel_scale_mas) / 3.6e6
    fits_reload_err = float(np.max(np.abs(fits_data[::-1] - np.asarray(mean_img.pixels, dtype=float))))
    wcs_scale_err_mas = 3.6e6 * max(
        abs(float(hdr["CDELT1"]) - (-deg_per_pix)),
        abs(float(hdr["CDELT2"]) - (+deg_per_pix)),
    )
    assert abs(float(hdr["CRPIX1"]) - 0.5 * (n_col + 1)) < 1e-12
    assert abs(float(hdr["CRPIX2"]) - 0.5 * (n_row + 1)) < 1e-12
    # World coordinates directly from 1-indexed FITS WCS (CDELT1 < 0 maps col < CRPIX1 to East +dRA; CDELT2 > 0 maps North +dDec)
    fits_dra_cols = float(hdr["CRVAL1"]) * 3.6e6 + (np.arange(1, n_col + 1) - float(hdr["CRPIX1"])) * float(hdr["CDELT1"]) * 3.6e6
    fits_ddec_rows = float(hdr["CRVAL2"]) * 3.6e6 + (np.arange(1, n_row + 1) - float(hdr["CRPIX2"])) * float(hdr["CDELT2"]) * 3.6e6
    w_fits = np.maximum(fits_data, 0.0)
    fits_dra_cent = float(np.sum(w_fits * fits_dra_cols[None, :]) / np.sum(w_fits))
    fits_ddec_cent = float(np.sum(w_fits * fits_ddec_rows[:, None]) / np.sum(w_fits))
    fits_wcs_err_mas = max(
        wcs_scale_err_mas,
        math.hypot(fits_dra_cent - dra_cent, fits_ddec_cent - ddec_cent),
    )

    centroid_err_mas = math.hypot(dra_cent - dra_comp, ddec_cent - ddec_comp)
    transposed_err_mas = math.hypot(dra_cent - ddec_comp, ddec_cent - dra_comp)
    by_name = {c.name: c.status for c in loaded.checks}

    record("centroid_err_mas", centroid_err_mas)
    record("transposed_err_mas", transposed_err_mas)
    record("fits_wcs_err_mas", fits_wcs_err_mas)
    assert fits_reload_err < 1e-14
    assert fits_wcs_err_mas < 1e-9
    assert centroid_err_mas < 0.9
    assert transposed_err_mas > 2.5
    assert by_name["chi2_reduced"] == "pass"
    assert by_name["error_scale"] == "pass"
    assert by_name["star"] == "pass"
    assert by_name["orientation"] == "pass"
    jax.clear_caches()
    gc.collect()


@pytest.mark.validates("virgil.pipeline.ImagingPipeline", roots=["mathematics"], kind="control")
def test_imaging_pipeline_underquoted_errors_check_control(tmp_path):
    """Severe noise with tenfold under-quoted error bars trips ImagingPipeline fit checks."""
    raw = tmp_path / "img_noisy.fits"
    simulate.observe(
        raw,
        lambda u, v, w: sky.vis_uniform_disk(u, v, w, 4.0),
        UTS3,
        hour_angles_h=[-1.0, 1.0],
        wavelengths=WAVELENGTHS[:2],
        sigma_v2=0.08,
        sigma_cp_deg=8.0,
        rng=np.random.default_rng(731),
        phase_noise="triangle",
    )
    shrunk = _shrink_oifits_errors(raw, tmp_path / "img_noisy_shrunk.fits", factor=12.0)
    res = vp.ImagingPipeline.from_oifits(
        shrunk,
        output=tmp_path / "run_img_ctrl",
        max_npix=12,
        field_mas=12.0,
        star=False,
        sigmas=[0.5],
        lengths_beam=[1.5],
        members=2,
        max_steps=30,
        quicklook_execute=False,
    ).run()
    by_name = {c.name: c.status for c in res.checks}
    err_scale = next(c for c in res.checks if c.name == "error_scale")
    record("error_scale_value", float(err_scale.value))
    assert by_name["chi2_reduced"] == "fail"
    assert by_name["error_scale"] == "fail"


# ---------------------------------------------------------------------------
# 6. OrbitPipeline + Result.aliases() + companion-free no_orbit control
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.pipeline.OrbitPipeline",
    roots=["mathematics"],
    headline="rel_period_err",
)
def test_orbit_pipeline_blind_keplerian_recovery(tmp_path):
    """OrbitPipeline blindly recovers a multi-night synthetic Keplerian orbit and saves AliasResult."""
    orbit = (12.0, 1.2, 0.20, 44.0, 38.0, 110.0, 6.0)
    flux = 0.25
    mjds = [T0 + 0.0, T0 + 2.6, T0 + 6.1, T0 + 11.4, T0 + 24.0]
    paths = [
        _write_moving_binary_night(tmp_path / f"orb_n{k}.fits", mjd_k, orbit, flux, seed=740 + k)
        for k, mjd_k in enumerate(mjds)
    ]
    run_dir = tmp_path / "run_orbit"
    res = vp.OrbitPipeline.from_oifits(
        paths,
        output=run_dir,
        max_grid=51,
        n_peaks=2,
        period_range=(9.0, 16.0),
        a_range=(3.0, 10.0),
        flux_range=(0.08, 0.5),
        ecc_max=0.45,
        primary_diam=None,
        vis_amp=False,
        n_candidates=24,
        n_refine=2,
        max_steps=80,
        n_is=200,
        max_modes=1,
        num_warmup=250,
        num_samples=300,
        num_chains=2,
        posterior_max_tree_depth=6,
        seed=2,
        quicklook_execute=False,
    ).run()
    loaded = vp.load(run_dir)
    assert loaded.status == "complete"
    aliases = loaded.aliases()
    assert isinstance(aliases, AliasResult)

    # Posterior summary, MCMC samples, mirror-invariant Thiele-Innes sky positions, and quality checks
    orb_sum = loaded.summary["orbit"]
    assert orb_sum["verdict"] == "orbit"
    assert orb_sum["source"] == "posterior"
    assert orb_sum["n"] == 2
    assert loaded.summary["posterior"]["converged"] is True
    orb_el = orb_sum["elements"]

    samples = loaded.samples()
    assert set(samples) >= {"period", "dt_peri", "ecc", "inc", "omega", "Omega", "a_mas", "flux"}
    assert np.asarray(samples["period"]).size == 600

    p_med = float(orb_el["period"]["median"])
    dtp_med = float(orb_el["dt_peri"]["median"])
    ecc_med = float(orb_el["ecc"]["median"])
    inc_med = float(orb_el["inc"]["median"])
    om_med = float(orb_el["omega"]["median"])
    Om_med = float(orb_el["Omega"]["median"])
    a_med = float(orb_el["a_mas"]["median"])
    flux_med = float(orb_el["flux"]["median"])

    rel_period_err = abs(p_med - orbit[0]) / orbit[0]
    rel_a_err = abs(a_med - orbit[6]) / orbit[6]
    ecc_err = abs(ecc_med - orbit[2])
    rel_flux_err = abs(flux_med - flux) / flux

    # Mirror-invariant Thiele-Innes constants (A, B, F, G) and epoch sky positions
    ti_fit = np.array(co.thiele_innes(a_med, inc_med, om_med, Om_med), dtype=float)
    ti_true = np.array(co.thiele_innes(orbit[6], orbit[3], orbit[4], orbit[5]), dtype=float)
    max_ti_err_mas = float(np.max(np.abs(ti_fit - ti_true)))

    dra_fit, ddec_fit = co.position(
        np.asarray(mjds) - float(aliases.t_ref), p_med, dtp_med, ecc_med, inc_med, om_med, Om_med, a_med
    )
    dra_true, ddec_true = co.position(np.asarray(mjds) - T0, *orbit)
    max_sky_pos_err_mas = float(np.max(np.hypot(dra_fit - dra_true, ddec_fit - ddec_true)))
    by_name = {c.name: c.status for c in loaded.checks}

    record("rel_period_err", rel_period_err)
    record("rel_a_err", rel_a_err)
    record("ecc_err", ecc_err)
    record("rel_flux_err", rel_flux_err)
    record("max_ti_err_mas", max_ti_err_mas)
    record("max_sky_pos_err_mas", max_sky_pos_err_mas)
    assert rel_period_err < 0.02
    assert rel_a_err < 0.04
    assert ecc_err < 0.04
    assert rel_flux_err < 0.05
    assert max_ti_err_mas < 0.20
    assert max_sky_pos_err_mas < 0.15
    assert by_name["no_orbit"] == "pass"
    assert by_name["chi2"] == "pass"
    for k in ("r_hat", "ess", "divergences", "prior_bound"):
        assert by_name[k] != "fail"
    assert loaded.summary["worst_check"] != "fail"
    jax.clear_caches()
    gc.collect()


@pytest.mark.validates(
    "virgil.pipeline.BinaryPipeline",
    "virgil.pipeline.OrbitPipeline",
    "virgil.pipeline.Check",
    roots=["mathematics"],
    kind="control",
)
def test_pipeline_short_nuts_chain_trips_ess_fail_control(tmp_path):
    """Short 30-draw NUTS chains on clean binary/orbit data pass chi2/detection yet fail ess and converged."""
    bin_path = tmp_path / "short_bin.fits"
    simulate.observe(
        bin_path,
        lambda u, v, w: (sky.vis_point(u, v, w) + 0.20 * sky.vis_point(u, v, w, 5.0, 3.6)) / 1.20,
        UTS4,
        hour_angles_h=np.linspace(-2.5, 2.5, 5),
        wavelengths=WAVELENGTHS,
        sigma_v2=0.008,
        sigma_cp_deg=0.5,
        rng=np.random.default_rng(748),
        phase_noise="baseline",
    )
    res_bin = vp.BinaryPipeline.from_oifits(
        bin_path,
        output=tmp_path / "run_short_bin",
        detection_sigma=2.0,
        n_null=50,
        max_grid=25,
        n_flux=12,
        limit_bins=5,
        num_warmup=15,
        num_samples=15,
        num_chains=2,
        seed=4,
        quicklook_execute=False,
    ).run()
    bin_by_name = {c.name: c.status for c in res_bin.checks}
    bin_ess_chk = next(c for c in res_bin.checks if c.name == "ess")

    orbit = (12.0, 1.2, 0.20, 44.0, 38.0, 110.0, 6.0)
    mjds = [T0 + 0.0, T0 + 2.6, T0 + 6.1, T0 + 11.4, T0 + 24.0]
    orb_paths = [
        _write_moving_binary_night(tmp_path / f"short_orb_{k}.fits", mjd_k, orbit, 0.25, seed=760 + k)
        for k, mjd_k in enumerate(mjds)
    ]
    res_orb = vp.OrbitPipeline.from_oifits(
        orb_paths,
        output=tmp_path / "run_short_orb",
        max_grid=41,
        n_peaks=2,
        period_range=(9.0, 16.0),
        a_range=(3.0, 10.0),
        flux_range=(0.08, 0.5),
        ecc_max=0.45,
        primary_diam=None,
        vis_amp=False,
        n_candidates=16,
        n_refine=1,
        max_steps=50,
        n_is=100,
        max_modes=1,
        num_warmup=15,
        num_samples=15,
        num_chains=2,
        posterior_max_tree_depth=4,
        seed=4,
        quicklook_execute=False,
    ).run()
    orb_by_name = {c.name: c.status for c in res_orb.checks}
    orb_ess_chk = next(c for c in res_orb.checks if c.name == "ess")

    short_chain_ess_max = max(float(bin_ess_chk.value), float(orb_ess_chk.value))
    record("short_chain_ess_max", short_chain_ess_max)
    assert bin_by_name["detection"] == "pass"
    assert bin_by_name["chi2"] == "pass"
    assert bin_by_name["ess"] == "fail"
    assert res_bin.summary["worst_check"] == "fail"
    assert orb_by_name["no_orbit"] == "pass"
    assert orb_by_name["chi2"] == "pass"
    assert orb_by_name["ess"] == "fail"
    assert res_orb.summary["posterior"]["converged"] is False
    assert res_orb.summary["worst_check"] == "fail"
    assert short_chain_ess_max < 100.0
    jax.clear_caches()
    gc.collect()


@pytest.mark.validates("virgil.pipeline.OrbitPipeline", roots=["mathematics"], kind="control")
def test_orbit_pipeline_companion_free_noise_control(tmp_path):
    """Companion-free single-star nights trip OrbitPipeline's no_orbit / detection quality checks."""
    paths = []
    for k, dt in enumerate([0.0, 3.0, 8.0, 15.0]):
        p = tmp_path / f"orb_null_{k}.fits"
        simulate.observe(
            p,
            lambda u, v, w: sky.vis_point(u, v, w),
            UTS3,
            hour_angles_h=[-1.5, 0.0, 1.5],
            wavelengths=WAVELENGTHS,
            sigma_v2=0.015,
            sigma_cp_deg=1.2,
            rng=np.random.default_rng(750 + k),
            phase_noise="triangle",
        )
        with fits.open(p, mode="update") as hdul:
            hdul["OI_VIS2"].data["MJD"] += dt
            hdul["OI_T3"].data["MJD"] += dt
        paths.append(p)

    res = vp.OrbitPipeline.from_oifits(
        paths,
        output=tmp_path / "run_orb_ctrl",
        max_grid=21,
        n_peaks=2,
        a_range=(2.0, 12.0),
        flux_range=(0.05, 0.4),
        ecc_max=0.5,
        primary_diam=None,
        vis_amp=False,
        n_bands_max=6,
        n_candidates=12,
        n_refine=1,
        max_steps=25,
        n_is=80,
        max_modes=1,
        num_warmup=20,
        num_samples=20,
        num_chains=2,
        posterior_max_tree_depth=4,
        quicklook_execute=False,
    ).run()
    by_name = {c.name: c.status for c in res.checks}
    n_flagged = sum(c.status in {"warn", "fail"} for c in res.checks)
    record("n_flagged_checks", n_flagged)
    assert by_name.get("no_orbit") == "fail"
    assert n_flagged >= 1


def teardown_module():
    jax.clear_caches()
    gc.collect()
