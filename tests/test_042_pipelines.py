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
import json
import math
import shutil

import jax
import numpy as np
import pytest
from astropy.io import fits
from scipy import special

from crosscheck import array as carray, oifits_writer, orbit_cells_ref as ref, simulate, sky
from evidence.plugin import record

pytest.importorskip("jaxoplanet")
pytest.importorskip("h5py")
pytest.importorskip("nbformat")
vp = pytest.importorskip("virgil.pipeline")

from virgil.aliases import AliasResult  # noqa: E402
from virgil.models import BinaryModelAngular, System  # noqa: E402
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

        v2_rec["vis2"].append(np.abs(vis_at(u, v)) ** 2)
        v2_rec["u"].append(u)
        v2_rec["v"].append(v)
        v2_rec["sta"].append(pairs)
        v2_rec["mjd"].append(np.full(len(u), mjd))

        cp, u1, v1, u2, v2_t3 = carray.closure_phase(vis_at, u, v, 4)
        t3_rec["phi"].append(np.rad2deg(cp))
        t3_rec["u1"].append(u1)
        t3_rec["v1"].append(v1)
        t3_rec["u2"].append(u2)
        t3_rec["v2"].append(v2_t3)
        t3_rec["sta"].append(tris)
        t3_rec["mjd"].append(np.full(len(u1), mjd))

    v2 = {k: np.concatenate(val) for k, val in v2_rec.items()}
    t3 = {k: np.concatenate(val) for k, val in t3_rec.items()}
    v2["vis2"] += 0.008 * rng.standard_normal(v2["vis2"].shape)
    t3["phi"] += 0.5 * rng.standard_normal(t3["phi"].shape)
    v2["err"] = np.full(v2["vis2"].shape, 0.008)
    t3["err"] = np.full(t3["phi"].shape, 0.5)
    oifits_writer.write(path, wavelengths=wl, stations_xyz=stations, vis2=v2, t3=t3)
    return path


# ---------------------------------------------------------------------------
# 1. Pipeline containers: Check, Stage, SkyImage
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.pipeline.Check",
    "virgil.pipeline.Stage",
    "virgil.pipeline.SkyImage",
    roots=["mathematics"],
    headline="dict_roundtrip_err",
)
def test_pipeline_check_stage_and_skyimage_containers():
    """Check round-trips through JSON dicts, Stage holds outputs, and SkyImage preserves pixel flux."""
    chk = vp.Check(
        name="chi2_companion",
        status="warn",
        value=2.75,
        threshold=2.0,
        message="Reduced chi-squared 2.75 exceeds 2.0.",
    )
    rebuilt = vp.Check.from_dict(json.loads(json.dumps(chk.to_dict())))
    assert rebuilt == chk
    dict_roundtrip_err = abs(float(rebuilt.value) - 2.75)

    called = {}
    stg = vp.Stage(name="overview", run=lambda p: called.setdefault("ran", p), outputs=("summary.json",))
    assert stg.name == "overview"
    assert stg.outputs == ("summary.json",)
    assert stg.run("sentinel") == "sentinel"

    pix = np.zeros((8, 8), dtype=float)
    pix[2, 5] = 0.75
    pix[4, 4] = 0.25
    img = vp.SkyImage(pixels=pix, pixel_scale_mas=0.5, dra=0.0, ddec=0.0, header={"STARFRAC": 0.2})
    flux_err = abs(float(np.sum(img.pixels)) - 1.0)

    record("dict_roundtrip_err", max(dict_roundtrip_err, flux_err))
    assert max(dict_roundtrip_err, flux_err) < 1e-15


# ---------------------------------------------------------------------------
# 2. BinaryPipeline + Result + load + ConfigMismatchError + quality-check control
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.pipeline.BinaryPipeline",
    "virgil.pipeline.Result",
    "virgil.pipeline.load",
    roots=["mathematics"],
    headline="sep_err_mas",
)
def test_binary_pipeline_recovery_resume_and_load_roundtrip(tmp_path):
    """BinaryPipeline recovers synthetic binary astrometry/flux, resumes cleanly, and reloads via load()."""
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
        UTS3,
        hour_angles_h=np.linspace(-2.5, 2.5, 5),
        wavelengths=WAVELENGTHS,
        sigma_v2=0.008,
        sigma_cp_deg=0.5,
        rng=np.random.default_rng(701),
        phase_noise="triangle",
    )

    run_dir = tmp_path / "run_binary"
    common_kwargs = dict(
        detection_sigma=2.0,
        n_null=50,
        max_grid=25,
        n_flux=12,
        limit_bins=6,
        num_warmup=60,
        num_samples=60,
        num_chains=2,
        seed=1,
        quicklook_execute=False,
    )
    pipe_part = vp.BinaryPipeline.from_oifits(oifits_path, output=run_dir, **common_kwargs)
    res_part = pipe_part.run(through="search")
    assert res_part.status == "partial"

    # Resuming with altered configuration must raise ConfigMismatchError
    pipe_bad = vp.BinaryPipeline.from_oifits(oifits_path, output=run_dir, **(common_kwargs | {"sigma": 4.5}))
    with pytest.raises(vp.ConfigMismatchError):
        pipe_bad.run(resume=True)

    # Resuming with identical configuration via from_config completes remaining stages
    cfg_path = tmp_path / "binary_cfg.json"
    cfg_path.write_text(
        json.dumps({"data": [str(oifits_path)], "output": str(run_dir), "settings": common_kwargs}),
        encoding="utf-8",
    )
    res_full = vp.BinaryPipeline.from_config(cfg_path).run(resume=True)
    assert res_full.status == "complete"

    loaded = vp.load(run_dir)
    assert isinstance(loaded, vp.Result)
    assert loaded.status == "complete"
    assert isinstance(loaded.data(), OIData)
    assert isinstance(loaded.model(), BinaryModelAngular)
    assert len(loaded.checks) >= 1 and all(isinstance(c, vp.Check) for c in loaded.checks)
    assert "sep_mas" in loaded.summary["companion"]
    assert len(loaded.describe()) > 0
    assert "axes" in loaded.grid()
    samples = loaded.samples()
    assert len(samples) > 0
    assert len(loaded.sample_stats()) > 0
    assert loaded.fit_result() is not None

    comp = loaded.summary["companion"]
    sep_err_mas = abs(float(comp["sep_mas"]) - true_sep)
    pa_diff = abs(((float(comp["pa_deg"]) - true_pa + 180.0) % 360.0) - 180.0)

    record("sep_err_mas", sep_err_mas)
    record("pa_err_deg", pa_diff)
    assert sep_err_mas < 0.15
    assert pa_diff < 2.0


@pytest.mark.validates("virgil.pipeline.BinaryPipeline", roots=["mathematics"], kind="control")
def test_binary_pipeline_nondetection_and_bad_chi2_checks_control(tmp_path):
    """Companion-free noisy OIFITS with under-quoted errors triggers non-pass pipeline quality checks (#663/#696)."""
    raw = tmp_path / "null_star.fits"
    simulate.observe(
        raw,
        lambda u, v, w: sky.vis_point(u, v, w),
        UTS3,
        hour_angles_h=[-2.0, 0.0, 2.0],
        wavelengths=WAVELENGTHS,
        sigma_v2=0.04,
        sigma_cp_deg=3.0,
        rng=np.random.default_rng(702),
        phase_noise="triangle",
    )
    bad = _shrink_oifits_errors(raw, tmp_path / "null_star_shrunk.fits", factor=8.0)
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
    statuses = [c.status for c in res.checks]
    n_flagged = sum(s in {"warn", "fail"} for s in statuses)
    record("n_flagged_checks", n_flagged)
    assert n_flagged >= 1


# ---------------------------------------------------------------------------
# 3. StarPipeline + quality-check control
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.pipeline.StarPipeline",
    roots=["mathematics"],
    headline="diam_err_mas",
)
def test_star_pipeline_recovers_uniform_disk_angular_diameter(tmp_path):
    """StarPipeline recovers the angular diameter of a resolved uniform-disk star."""
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
        phase_noise="triangle",
    )
    run_dir = tmp_path / "run_star"
    res = vp.StarPipeline.from_oifits(
        star_path,
        model="uniform",
        output=run_dir,
        n_scan=250,
        max_lobes=3,
        num_warmup=60,
        num_samples=60,
        num_chains=2,
        quicklook_execute=False,
    ).run()
    loaded = vp.load(run_dir)
    diam_fit = float(loaded.summary["star"]["diam_mas"])
    diam_err_mas = abs(diam_fit - true_diam)

    record("diam_err_mas", diam_err_mas)
    assert diam_err_mas < 0.12
    assert loaded.model("uniform") is not None
    assert "diameter" in loaded.model_values("uniform") or len(loaded.model_values("uniform")) >= 1


@pytest.mark.validates("virgil.pipeline.StarPipeline", roots=["mathematics"], kind="control")
def test_star_pipeline_mismatched_binary_chi2_check_control(tmp_path):
    """Fitting a high-SNR equal-flux binary with StarPipeline triggers chi-squared failure checks."""
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
    n_flagged = sum(c.status in {"warn", "fail"} for c in res.checks)
    record("n_flagged_checks", n_flagged)
    assert n_flagged >= 1


# ---------------------------------------------------------------------------
# 4. DiskPipeline + quality-check control
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.pipeline.DiskPipeline",
    roots=["mathematics"],
    headline="ring_radius_rel_err",
)
def test_disk_pipeline_recovers_inclined_gaussian_ring(tmp_path):
    """DiskPipeline('rings') recovers the radius and geometry of a synthetic inclined ring."""
    true_r = 5.5
    true_inc = 42.0
    true_pa = 68.0
    disk_path = tmp_path / "ring_disk.fits"
    simulate.observe(
        disk_path,
        _inclined_gaussian_ring_vis(true_r, sigma_mas=0.7, inc_deg=true_inc, pa_deg=true_pa, disk_flux=0.8),
        UTS4,
        hour_angles_h=np.linspace(-3.0, 3.0, 6),
        wavelengths=WAVELENGTHS,
        sigma_v2=0.008,
        sigma_cp_deg=0.5,
        rng=np.random.default_rng(720),
        phase_noise="triangle",
    )
    run_dir = tmp_path / "run_disk"
    res = vp.DiskPipeline.from_oifits(
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
    assert loaded.summary["disk"]["n_rings"] >= 1
    rings = loaded.summary["disk"]["rings"]
    # Extract fitted radius of dominant ring (check common key names radius_mas / r_mas / radius)
    r0_dict = rings[0]
    r_key = next(k for k in ("radius_mas", "r_mas", "radius", "r") if k in r0_dict)
    r_fit = float(r0_dict[r_key])
    ring_radius_rel_err = abs(r_fit - true_r) / true_r

    record("ring_radius_rel_err", ring_radius_rel_err)
    assert ring_radius_rel_err < 0.20


@pytest.mark.validates("virgil.pipeline.DiskPipeline", roots=["mathematics"], kind="control")
def test_disk_pipeline_corrupted_errors_check_control(tmp_path):
    """Strong asymmetric binary data with under-quoted errors trips DiskPipeline quality checks."""
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
    n_flagged = sum(c.status in {"warn", "fail"} for c in res.checks)
    record("n_flagged_checks", n_flagged)
    assert n_flagged >= 1


# ---------------------------------------------------------------------------
# 5. ImagingPipeline + member staging + quality-check control
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.pipeline.ImagingPipeline",
    roots=["mathematics"],
    headline="quadrant_flux_excess",
)
def test_imaging_pipeline_staged_members_and_skyimage_recovery(tmp_path):
    """ImagingPipeline reconstructs off-axis companion emission into the NE sky quadrant and writes FITS maps."""
    dra_comp, ddec_comp = 3.2, 2.6
    img_oifits = tmp_path / "img_scene.fits"
    simulate.observe(
        img_oifits,
        lambda u, v, w: (
            sky.vis_point(u, v, w) + 0.35 * sky.vis_gaussian(u, v, w, 1.2, dra_comp, ddec_comp)
        )
        / 1.35,
        UTS4,
        hour_angles_h=np.linspace(-3.0, 3.0, 7),
        wavelengths=WAVELENGTHS,
        sigma_v2=0.008,
        sigma_cp_deg=0.5,
        rng=np.random.default_rng(730),
        phase_noise="triangle",
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

    mean_img = vp.load(run_dir).image("mean")
    sig_img = vp.load(run_dir).image("sigma")
    assert isinstance(mean_img, vp.SkyImage)
    assert isinstance(sig_img, vp.SkyImage)
    assert np.all(np.isfinite(mean_img.pixels)) and np.all(mean_img.pixels >= -1e-12)

    # SkyImage convention: row 0 is North (+dDec), col 0 is East (+dRA) -> NE quadrant is top-left [0:n//2, 0:n//2]
    n = mean_img.pixels.shape[0]
    ne_flux = float(np.sum(mean_img.pixels[: n // 2, : n // 2]))
    sw_flux = float(np.sum(mean_img.pixels[n // 2 :, n // 2 :]))
    total = max(float(np.sum(mean_img.pixels)), 1e-12)
    quadrant_flux_excess = (ne_flux - sw_flux) / total

    record("quadrant_flux_excess", quadrant_flux_excess)
    assert quadrant_flux_excess > 0.02
    assert ne_flux > sw_flux


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
    n_flagged = sum(c.status in {"warn", "fail"} for c in res.checks)
    record("n_flagged_checks", n_flagged)
    assert n_flagged >= 1


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
        num_warmup=40,
        num_samples=40,
        num_chains=2,
        posterior_max_tree_depth=5,
        seed=2,
        quicklook_execute=False,
    ).run()
    loaded = vp.load(run_dir)
    assert loaded.status == "complete"
    aliases = loaded.aliases()
    assert isinstance(aliases, AliasResult)

    p_win = float(loaded.summary["modes"]["winner"]["elements"]["period"])
    rel_period_err = abs(p_win - orbit[0]) / orbit[0]
    record("rel_period_err", rel_period_err)
    assert rel_period_err < 0.03


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
