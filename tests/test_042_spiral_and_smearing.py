"""Validation of virgil 0.4.2 ArchimedeanSpiral and smeared_model (Batch 5)."""

import contextlib
import gc
import io
import json
import os
import pathlib
import subprocess

import jax
import jax.numpy as jnp
from jax.test_util import check_grads
import numpy as np
import pytest

import virgil.models as vm
import virgil.oidata as voidata
import virgil.orbits as vorbits

from crosscheck import nrm, simulate, sky, spiral_smearing
from evidence.plugin import record
from external_bridge import _subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def _clear_jax_caches():
    yield
    jax.clear_caches()
    gc.collect()


def _candid_python() -> str | None:
    """Return a working CANDID Python interpreter (including git worktrees)."""
    candidates = [_subprocess.python("candid")]
    git_file = ROOT / ".git"
    if "CANDID_PYTHON" not in os.environ and git_file.is_file():
        text = git_file.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            gitdir = pathlib.Path(text.split(":", 1)[1].strip())
            if gitdir.parent.name == "worktrees" and gitdir.parent.parent.name == ".git":
                candidates.append(str(gitdir.parent.parent.parent / ".venv-candid" / "bin" / "python"))
    for py in candidates:
        if not py:
            continue
        try:
            res = subprocess.run(
                [py, "-c", "import candid"],
                capture_output=True,
                env={**os.environ, "MPLBACKEND": "Agg"},
                timeout=30,
            )
            if res.returncode == 0:
                return py
        except OSError:
            continue
    return None


def _run_candid_smear_chi2(
    fits_path: pathlib.Path,
    dra: float,
    ddec: float,
    flux_pct: float,
    dwavel_um: float,
    n_smear: int,
) -> dict:
    """Evaluate CANDID's reduced chi^2 with explicit dwavel (um) and Nsmear."""
    py = _candid_python()
    if py is None:
        pytest.skip("CANDID environment (.venv-candid) is not installed")
    code = """
import json, sys, matplotlib
matplotlib.use("Agg")
import candid

payload = json.loads(sys.argv[1])
candid.CONFIG.update({
    "Ncores": 1,
    "long exec warning": None,
    "progress bar": False,
    "Nsmear": int(payload["n_smear"]),
})
o = candid.Open(payload["path"])
o.observables = ["v2", "cp"]
ins = o.instruments[0]
dw = float(payload["dwavel_um"])
o.dwavel = {k: dw for k in o.dwavel}
param = {
    "x": float(payload["dra"]),
    "y": float(payload["ddec"]),
    "f": float(payload["flux_pct"]),
    "diam*": 0.0,
    "alpha*": 0.0,
}
for k in o.dwavel:
    param["dwavel;" + k] = dw
c2 = float(candid._chi2Func(param, o._chi2Data, o.observables, o.instruments))
print(json.dumps({"chi2r": c2, "ndata": int(o.ndata())}))
"""
    payload = json.dumps(
        {
            "path": str(fits_path),
            "dra": dra,
            "ddec": ddec,
            "flux_pct": flux_pct,
            "dwavel_um": dwavel_um,
            "n_smear": n_smear,
        }
    )
    out = subprocess.run(
        [py, "-c", code, payload],
        capture_output=True,
        text=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
        timeout=120,
        check=True,
    )
    last_line = [ln for ln in out.stdout.splitlines() if ln.strip().startswith("{")][-1]
    return json.loads(last_line)


# ---------------------------------------------------------------------------
# 1. Reference checks for crosscheck.spiral_smearing, pmoired, and candid
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "crosscheck.spiral_smearing",
    roots=["mathematics"],
    kind="reference",
    headline="max_err_small_turns_limit",
)
def test_reference_spiral_small_turns_and_arclength_identities():
    """As turns -> 0+, the blurred spiral collapses to an inclined elliptical
    Gaussian at the origin, and the Gauss-Legendre arc-length integral matches
    0.5 * (Theta * sqrt(1 + Theta^2) + arsinh(Theta))."""
    rng = np.random.default_rng(101)
    u = rng.uniform(-80.0, 80.0, 40)
    v = rng.uniform(-80.0, 80.0, 40)
    wavel = np.full(40, 2.2e-6)

    v_spiral = spiral_smearing.spiral_cvis(
        u,
        v,
        wavel,
        pitch=30.0,
        turns=1e-13,
        width=4.5,
        pa=55.0,
        inc=50.0,
        node=22.0,
        fade=0.8,
        dra=6.0,
        ddec=-4.0,
        n_quad=64,
    )
    v_gauss = sky.vis_elliptical_gaussian(
        u,
        v,
        wavel,
        fwhm=4.5,
        ratio=np.cos(np.deg2rad(50.0)),
        pa=22.0,
        dra=6.0,
        ddec=-4.0,
    )
    err_limit = float(np.max(np.abs(v_spiral - v_gauss)))

    # Exact hyperbolic arc-length primitive for fade=0:
    turns = 1.75
    theta_max = 2.0 * np.pi * turns
    x_gl, w_gl = spiral_smearing.roots_legendre(128)
    theta = 0.5 * theta_max * (x_gl + 1.0)
    quad_len = float(0.5 * theta_max * np.sum(w_gl * np.sqrt(1.0 + theta**2)))
    exact_len = float(0.5 * (theta_max * np.sqrt(1.0 + theta_max**2) + np.arcsinh(theta_max)))
    err_arclen = abs(quad_len - exact_len)

    record("max_err_small_turns_limit", err_limit)
    record("arclength_quadrature_err", err_arclen)
    assert err_limit < 1e-11
    assert err_arclen < 5e-14


@pytest.mark.validates(
    "crosscheck.spiral_smearing",
    roots=["mathematics"],
    kind="reference",
    headline="wavenumber_sinc_quadrature_err",
)
def test_reference_smearing_wavenumber_and_incoherent_identities():
    """Gauss-Legendre quadrature over uniform wavenumber sigma = 1/lambda
    matches the closed-form sinc envelope to machine precision, and the
    3-endpoint incoherent rule has variance dlambda^2 / 6 and asymptote
    (1 + f^2) / (1 + f)^2."""
    rng = np.random.default_rng(202)
    u = rng.uniform(-90.0, 90.0, 30)
    v = rng.uniform(-90.0, 90.0, 30)
    wavel = np.linspace(2.0e-6, 2.4e-6, 30)
    bw = np.full(30, 0.04e-6)
    dra_p, ddec_p = -3.0, 2.0
    dra_s, ddec_s, f = 28.0, -19.0, 0.35

    v_quad = spiral_smearing.smeared_binary_wavenumber_quadrature(
        u, v, wavel, bw, dra_s, ddec_s, f, dra_pri=dra_p, ddec_pri=ddec_p, n_quad=64
    )
    v_pri = spiral_smearing.smeared_offset_phasor(u, v, wavel, bw, dra_p, ddec_p)
    v_sec = spiral_smearing.smeared_offset_phasor(u, v, wavel, bw, dra_s, ddec_s)
    v_sinc = (v_pri + f * v_sec) / (1.0 + f)
    err_sinc = float(np.max(np.abs(v_quad - v_sinc)))

    # Dense incoherent average across many washed-out fringes approaches (1 + f^2)/(1 + f)^2:
    v2_incoh, _ = spiral_smearing.incoherent_endpoint_binary_observables(
        np.array([150.0]),
        np.array([90.0]),
        np.array([2.2e-6]),
        np.array([0.8e-6]),
        dra=950.0,
        ddec=680.0,
        flux_ratio=f,
        n_smear=8001,
    )
    asymptote_err = float(abs(v2_incoh[0] - (1.0 + f**2) / (1.0 + f) ** 2))

    record("wavenumber_sinc_quadrature_err", err_sinc)
    record("incoherent_asymptote_err", asymptote_err)
    assert err_sinc < 1e-14
    assert asymptote_err < 5e-4


@pytest.mark.external
@pytest.mark.validates(
    "pmoired",
    roots=["mathematics"],
    kind="reference",
    headline="pmoired_wl_kernel_v2_err",
)
def test_pmoired_explicit_wl_kernel_matches_discrete_gaussian_convolution(tmp_path):
    """Pin PMOIRED's explicit ``'wl kernel'`` (with ``auto=False`` to avoid P1):
    it convolves V^2 incoherently and OI_VIS/|V| + OI_T3 bispectrum coherently
    along the channel axis with ``discrete_gaussian_spectral_kernel``."""
    pmoired = pytest.importorskip("pmoired")
    import pmoired.oimodels as om

    dra, ddec, f = 32.0, -21.0, 0.25
    cloud = sky.mix([sky.point(0.0, 0.0), sky.point(dra, ddec)], [1.0, f])
    uts = np.array([[0.0, 0.0], [38.0, 11.0], [14.0, 52.0], [65.0, -18.0]])
    wls = np.linspace(2.0e-6, 2.4e-6, 41)
    fits_path = tmp_path / "pmoired_wl_kernel.fits"
    simulate.observe(
        fits_path,
        lambda u, v, w: sky.visibility(cloud, u, v, w),
        uts,
        hour_angles_h=[0.0],
        wavelengths=wls,
        dec_deg=-30.0,
        sigma_v2=0.001,
        sigma_cp_deg=0.1,
    )

    wl_kernel_pix = 2.6
    ker = spiral_smearing.discrete_gaussian_spectral_kernel(wl_kernel_pix)
    pad = len(ker) // 2
    sl = slice(pad, -pad)

    params = {"c,ud": 0.0, "c,f": 1.0, "b,ud": 0.0, "b,f": f, "b,x": dra, "b,y": ddec}
    with contextlib.redirect_stdout(io.StringIO()):
        oi_raw = pmoired.OI(str(fits_path), verbose=False)
        oi_raw.setupFit({"obs": ["V2", "T3PHI"]}, auto=False)
        m_raw = om.VmodelOI(oi_raw.data[0], params)

        oi_ker = pmoired.OI(str(fits_path), verbose=False)
        oi_ker.setupFit({"obs": ["V2", "T3PHI"], "wl kernel": wl_kernel_pix}, auto=False)
        m_ker = om.VmodelOI(oi_ker.data[0], params)

    v2_err = 0.0
    for k in m_ker["OI_VIS2"]:
        ref_v2 = np.convolve(m_raw["OI_VIS2"][k]["V2"][0], ker, mode="same")[sl]
        v2_err = max(v2_err, float(np.max(np.abs(m_ker["OI_VIS2"][k]["V2"][0, sl] - ref_v2))))

    cp_err_deg = 0.0
    for k in m_ker["OI_T3"]:
        bisp_raw = m_raw["OI_T3"][k]["T3AMP"][0] * np.exp(1j * np.deg2rad(m_raw["OI_T3"][k]["T3PHI"][0]))
        bisp_conv = np.convolve(bisp_raw, ker, mode="same")[sl]
        ref_cp = np.rad2deg(np.angle(bisp_conv))
        diff = (m_ker["OI_T3"][k]["T3PHI"][0, sl] - ref_cp + 180.0) % 360.0 - 180.0
        cp_err_deg = max(cp_err_deg, float(np.max(np.abs(diff))))

    record("pmoired_wl_kernel_v2_err", v2_err)
    record("pmoired_wl_kernel_cp_err_deg", cp_err_deg)
    assert v2_err < 1e-9  # EFF_WAVE stored as float32 in OIFITS
    assert cp_err_deg < 1e-7


@pytest.mark.external
@pytest.mark.validates(
    "candid",
    roots=["mathematics"],
    kind="reference",
    headline="candid_nsmear3_chi2r",
)
def test_candid_nsmear3_matches_coherent_three_endpoint_rule(tmp_path):
    """Pin CANDID's default ``Nsmear = 3`` bandwidth smearing: it averages
    complex visibilities coherently with equal weights ``(1/3, 1/3, 1/3)`` at
    ``lambda + (-1/2, 0, +1/2) * dwavel`` (spectral variance ``dwavel^2 / 6``),
    and disables smearing when ``Nsmear <= 2``."""
    if _candid_python() is None:
        pytest.skip("CANDID environment (.venv-candid) is not installed")

    dra, ddec, f = 40.0, -25.0, 0.30
    dlambda_m = 0.06e-6  # 0.06 um channel width
    uts = np.array([[0.0, 0.0], [35.0, 12.0], [10.0, 48.0], [62.0, -15.0]])
    wls = np.linspace(2.05e-6, 2.35e-6, 7)

    # Generate geometry via simulate.observe, then overwrite OI_VIS2 / OI_T3
    # with exact 3-endpoint coherent observables.
    fits_path = tmp_path / "candid_nsmear3.fits"
    cloud = sky.mix([sky.point(0.0, 0.0), sky.point(dra, ddec)], [1.0, f])
    simulate.observe(
        fits_path,
        lambda u, v, w: sky.visibility(cloud, u, v, w),
        uts,
        hour_angles_h=[0.0],
        wavelengths=wls,
        dec_deg=-30.0,
        sigma_v2=0.001,
        sigma_cp_deg=0.05,
    )
    from astropy.io import fits

    with fits.open(fits_path, mode="update") as hdul:
        eff_wave = hdul["OI_WAVELENGTH"].data["EFF_WAVE"].astype(float)
        hdul["OI_WAVELENGTH"].data["EFF_BAND"][:] = dlambda_m
        u_v2 = hdul["OI_VIS2"].data["UCOORD"][:, None]
        v_v2 = hdul["OI_VIS2"].data["VCOORD"][:, None]
        v2_ref, _ = spiral_smearing.coherent_endpoint_binary_observables(
            u_v2, v_v2, eff_wave[None, :], dlambda_m, dra, ddec, f, n_smear=3
        )
        hdul["OI_VIS2"].data["VIS2DATA"][:] = v2_ref

        u1 = hdul["OI_T3"].data["U1COORD"][:, None]
        v1 = hdul["OI_T3"].data["V1COORD"][:, None]
        u2 = hdul["OI_T3"].data["U2COORD"][:, None]
        v2_ = hdul["OI_T3"].data["V2COORD"][:, None]
        u_tri = np.stack([u1, u2, u1 + u2], axis=0)
        v_tri = np.stack([v1, v2_, v1 + v2_], axis=0)
        _, cp_rad = spiral_smearing.coherent_endpoint_binary_observables(
            u_tri,
            v_tri,
            eff_wave[None, None, :],
            dlambda_m,
            dra,
            ddec,
            f,
            i_cps=(0, 1, 2),
            n_smear=3,
        )
        hdul["OI_T3"].data["T3PHI"][:] = np.rad2deg(cp_rad)
        hdul.flush()

    res_smeared = _run_candid_smear_chi2(
        fits_path, dra=dra, ddec=ddec, flux_pct=100.0 * f, dwavel_um=dlambda_m * 1e6, n_smear=3
    )
    res_unsmeared = _run_candid_smear_chi2(
        fits_path, dra=dra, ddec=ddec, flux_pct=100.0 * f, dwavel_um=0.0, n_smear=3
    )
    res_nsmear2 = _run_candid_smear_chi2(
        fits_path, dra=dra, ddec=ddec, flux_pct=100.0 * f, dwavel_um=dlambda_m * 1e6, n_smear=2
    )

    record("candid_nsmear3_chi2r", res_smeared["chi2r"])
    record("candid_unsmeared_chi2r", res_unsmeared["chi2r"])
    assert res_smeared["chi2r"] < 1e-6
    assert res_unsmeared["chi2r"] > 50.0
    assert abs(res_nsmear2["chi2r"] - res_unsmeared["chi2r"]) < 1e-10


# ---------------------------------------------------------------------------
# 2. virgil.models.ArchimedeanSpiral (roots: mathematics + dlux, + controls)
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.models.ArchimedeanSpiral",
    roots=["mathematics"],
    headline="max_cvis_err_dense_nodes",
)
def test_archimedean_spiral_cvis_convergence_and_gradients():
    """ArchimedeanSpiral visibilities match independent Gauss-Legendre arc-length
    Fourier quadrature across winding signs, inclinations, off-node sky PA,
    in-plane exponential fade, System mixing, O(N^-2) trapezoidal convergence,
    and AD gradients."""
    rng = np.random.default_rng(303)
    u = rng.uniform(-65.0, 65.0, 36)
    v = rng.uniform(-65.0, 65.0, 36)
    wavel = np.linspace(1.9e-6, 2.4e-6, 36)

    cases = [
        dict(pitch=36.0, turns=1.35, width=4.0, pa=28.0, inc=0.0, node=0.0, fade=0.0, dra=0.0, ddec=0.0),
        dict(pitch=-29.0, turns=1.75, width=3.6, pa=132.0, inc=0.0, node=0.0, fade=1.15, dra=-2.5, ddec=1.8),
        dict(pitch=42.0, turns=1.25, width=4.4, pa=76.0, inc=53.0, node=21.0, fade=0.85, dra=5.2, ddec=-3.4),
        dict(pitch=-31.0, turns=1.15, width=3.9, pa=-38.0, inc=61.0, node=112.0, fade=0.55, dra=-1.7, ddec=4.1),
    ]

    max_dense_err = 0.0
    max_default_err = 0.0
    for kw in cases:
        v_ref = spiral_smearing.spiral_cvis(u, v, wavel, **kw, n_quad=512)
        v_dense = np.asarray(vm.ArchimedeanSpiral(**kw, nodes=4096).model(u, v, wavel))
        v_def = np.asarray(vm.ArchimedeanSpiral(**kw).model(u, v, wavel))
        max_dense_err = max(max_dense_err, float(np.max(np.abs(v_dense - v_ref))))
        max_default_err = max(max_default_err, float(np.max(np.abs(v_def - v_ref))))

    # Flux-weighted System mixing with a central star:
    kw_sys = cases[2]
    sys_mod = vm.System(
        star=vm.PointSource(flux=1.0),
        arm=vm.ArchimedeanSpiral(**kw_sys, flux=0.35, nodes=4096),
    )
    v_sys = np.asarray(sys_mod.model(u, v, wavel))
    v_arm_ref = spiral_smearing.spiral_cvis(u, v, wavel, **kw_sys, n_quad=512)
    v_sys_ref = (1.0 + 0.35 * v_arm_ref) / 1.35
    sys_err = float(np.max(np.abs(v_sys - v_sys_ref)))

    # Trapezoidal endpoint O(N^-2) convergence on a smooth arm:
    kw_conv = dict(pitch=38.0, turns=1.4, width=2.5, pa=50.0, inc=44.0, node=15.0, fade=0.9)
    v_truth = spiral_smearing.spiral_cvis(u, v, wavel, **kw_conv, n_quad=2048)
    e_256 = float(np.max(np.abs(np.asarray(vm.ArchimedeanSpiral(**kw_conv, nodes=256).model(u, v, wavel)) - v_truth)))
    e_512 = float(np.max(np.abs(np.asarray(vm.ArchimedeanSpiral(**kw_conv, nodes=512).model(u, v, wavel)) - v_truth)))
    conv_ratio = e_256 / e_512

    # Automatic differentiation wrt all continuous shape & offset parameters:
    u_j = jnp.asarray(u[:8])
    v_j = jnp.asarray(v[:8])
    w_j = jnp.asarray(wavel[:8])

    def scalar_loss(params):
        pitch, turns, width, pa, inc, node, fade, dra, ddec = params
        m = vm.ArchimedeanSpiral(
            pitch=pitch,
            turns=turns,
            width=width,
            pa=pa,
            inc=inc,
            node=node,
            fade=fade,
            dra=dra,
            ddec=ddec,
            nodes=256,
        )
        cv = m.model(u_j, v_j, w_j)
        return jnp.sum(cv.real**2 + 0.7 * cv.imag**2)

    p0 = jnp.array([34.0, 1.2, 4.2, 62.0, 48.0, 19.0, 0.7, 3.0, -2.0])
    check_grads(scalar_loss, (p0,), order=1, modes=["rev"], atol=1e-4, rtol=1e-4)

    record("max_cvis_err_dense_nodes", max_dense_err)
    record("max_cvis_err_default_nodes", max_default_err)
    record("system_mix_err", sys_err)
    record("trapezoid_doubling_ratio", conv_ratio)
    assert max_dense_err < 5e-6
    assert max_default_err < 4e-4
    assert sys_err < 5e-6
    assert 3.7 < conv_ratio < 4.3


@pytest.mark.validates(
    "virgil.models.ArchimedeanSpiral",
    roots=["mathematics"],
    headline="render_resolved_max_err",
)
def test_archimedean_spiral_render_and_one_pixel_width_floor():
    """ArchimedeanSpiral.render produces a unit-sum East-left, North-up image
    that agrees with continuous Gaussian-ridge rendering and floors sub-pixel
    arm widths at 1 pixel."""
    kw_res = dict(pitch=28.0, turns=1.15, width=6.0, pa=42.0, inc=46.0, node=16.0, fade=0.75, dra=3.0, ddec=-2.0)
    img_res = np.asarray(vm.ArchimedeanSpiral(**kw_res, nodes=2048).render(npix=128, fov_mas=128.0))
    ref_res = spiral_smearing.spiral_render(128, 128.0, **kw_res, n_quad=512, floor_px=True)
    err_res = float(np.max(np.abs(img_res - ref_res)))

    kw_sub = {**kw_res, "width": 0.2}
    img_sub = np.asarray(vm.ArchimedeanSpiral(**kw_sub, nodes=2048).render(npix=128, fov_mas=128.0))
    ref_sub_floored = spiral_smearing.spiral_render(128, 128.0, **kw_sub, n_quad=512, floor_px=True)
    ref_sub_unfloored = spiral_smearing.spiral_render(128, 128.0, **kw_sub, n_quad=512, floor_px=False)
    err_sub_floored = float(np.max(np.abs(img_sub - ref_sub_floored)))
    diff_unfloored = float(np.max(np.abs(img_sub - ref_sub_unfloored)))

    record("render_resolved_max_err", err_res)
    record("render_subpixel_floored_err", err_sub_floored)
    record("render_unfloored_gap", diff_unfloored)
    assert np.isclose(np.sum(img_res), 1.0, atol=1e-7)
    assert err_res < 2e-6
    assert err_sub_floored < 2e-5
    assert diff_unfloored > 2e-3


@pytest.mark.validates(
    "virgil.models.ArchimedeanSpiral",
    roots=["dlux"],
    headline="dlux_spiral_v2_err",
)
def test_archimedean_spiral_dlux_wavefront_propagation():
    """Propagate an independent inclined fading Archimedean spiral + central
    star through dLux's 7-hole Fraunhofer optical model and compare calibrated
    squared visibilities and closure phases against virgil."""
    pytest.importorskip("dLux")

    wavel = 4.3e-6
    npix, pixel_mas = 192, 25.0
    optics = nrm.dlux_optics(npix=npix, pixel_mas=pixel_mas, wf_npix=192)

    kw = dict(pitch=95.0, turns=1.1, width=18.0, pa=62.0, inc=45.0, node=18.0, fade=0.8)
    spiral_2d = spiral_smearing.spiral_cloud_2d(**kw, n_quad=128, n_gauss=8)
    scene_cloud = sky.mix([sky.point(0.0, 0.0), spiral_2d], [1.0, 0.45])

    sci_img = nrm.image_dlux(scene_cloud, wavel, optics, chunk=192)
    cal_img = nrm.image_dlux(sky.point(0.0, 0.0), wavel, optics, chunk=192)

    # Extract all 21 baselines and 35 closure triangles of the 7-hole mask:
    pairs = [(i, j) for i in range(len(nrm.HOLES)) for j in range(i + 1, len(nrm.HOLES))]
    bl = np.array([nrm.HOLES[j] - nrm.HOLES[i] for i, j in pairs])
    u_m, v_m = bl[:, 0], bl[:, 1]
    w_arr = np.full_like(u_m, wavel)

    cvis_dlux = nrm.calibrated_vis_fn(sci_img, cal_img, wavel, pixel_mas)(u_m, v_m, w_arr)
    v_scene = vm.System(
        star=vm.PointSource(flux=1.0),
        arm=vm.ArchimedeanSpiral(**kw, flux=0.45, nodes=1024),
    )
    cvis_virgil = np.asarray(v_scene.model(u_m, v_m, w_arr))

    dv2 = float(np.max(np.abs(np.abs(cvis_dlux) ** 2 - np.abs(cvis_virgil) ** 2)))

    pair_idx = {p: k for k, p in enumerate(pairs)}
    cp_diffs = []
    cp_signal = []
    for i in range(len(nrm.HOLES)):
        for j in range(i + 1, len(nrm.HOLES)):
            for k in range(j + 1, len(nrm.HOLES)):
                i1, i2, i3 = pair_idx[(i, j)], pair_idx[(j, k)], pair_idx[(i, k)]
                b_dl = cvis_dlux[i1] * cvis_dlux[i2] * np.conj(cvis_dlux[i3])
                b_vv = cvis_virgil[i1] * cvis_virgil[i2] * np.conj(cvis_virgil[i3])
                cp_dl = np.rad2deg(np.angle(b_dl))
                cp_vv = np.rad2deg(np.angle(b_vv))
                cp_signal.append(abs(cp_vv))
                cp_diffs.append(abs((cp_dl - cp_vv + 180.0) % 360.0 - 180.0))

    dcp_deg = float(max(cp_diffs))
    max_cp_deg = float(max(cp_signal))

    record("dlux_spiral_v2_err", dv2)
    record("dlux_spiral_cp_err_deg", dcp_deg)
    record("dlux_spiral_max_cp_deg", max_cp_deg)
    assert max_cp_deg > 4.0  # strong non-trivial asymmetric closure-phase signal
    assert dv2 < 8e-4
    assert dcp_deg < 0.08


@pytest.mark.validates(
    "virgil.models.ArchimedeanSpiral",
    roots=["mathematics"],
    kind="control",
    headline="min_wrong_convention_cvis_gap",
)
def test_control_archimedean_spiral_wrong_winding_pa_fade_and_blur_fail():
    """Negative controls: flipping winding (-pitch), rotating departure PA by
    90 or 180 deg, omitting the sqrt(1 + theta^2) arc-length factor in fade,
    or blurring isotropically on the sky instead of in the arm plane all fail."""
    rng = np.random.default_rng(404)
    u = rng.uniform(-60.0, 60.0, 40)
    v = rng.uniform(-60.0, 60.0, 40)
    wavel = np.full(40, 2.2e-6)

    kw = dict(pitch=38.0, turns=1.5, width=5.5, pa=68.0, inc=58.0, node=15.0, fade=1.1)
    v_vv = np.asarray(vm.ArchimedeanSpiral(**kw, nodes=2048).model(u, v, wavel))

    gaps = {
        "neg_pitch": float(np.max(np.abs(v_vv - spiral_smearing.spiral_cvis(u, v, wavel, **{**kw, "pitch": -kw["pitch"]})))),
        "pa_plus_90": float(np.max(np.abs(v_vv - spiral_smearing.spiral_cvis(u, v, wavel, **{**kw, "pa": kw["pa"] + 90.0})))),
        "pa_plus_180": float(np.max(np.abs(v_vv - spiral_smearing.spiral_cvis(u, v, wavel, **{**kw, "pa": kw["pa"] + 180.0})))),
        "angle_only_fade": float(np.max(np.abs(v_vv - spiral_smearing.spiral_cvis(u, v, wavel, **kw, fade_mode="angle_only")))),
        "sky_isotropic_blur": float(np.max(np.abs(v_vv - spiral_smearing.spiral_cvis(u, v, wavel, **kw, blur_mode="sky")))),
    }
    min_gap = min(gaps.values())
    record("min_wrong_convention_cvis_gap", min_gap)
    for label, gap in gaps.items():
        assert gap > 0.015, f"Control {label} did not fail strongly enough: {gap}"


# ---------------------------------------------------------------------------
# 3. virgil.models.smeared_model (roots: mathematics + pmoired, + CANDID D*)
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.models.smeared_model",
    roots=["mathematics"],
    headline="wavenumber_boxcar_max_err",
)
def test_smeared_model_wavenumber_identity_wavelength_scaling_and_trees():
    """smeared_model equals uniform-wavenumber boxcar quadrature to machine
    precision, converges to uniform-wavelength quadrature at O((dlambda/lambda0)^2)
    (16x per 4x bandwidth reduction), accumulates offsets down nested Systems,
    leaves zero-offset and Resolved pieces untouched, and commutes with Rotated."""
    rng = np.random.default_rng(505)
    u = rng.uniform(-90.0, 90.0, 32)
    v = rng.uniform(-90.0, 90.0, 32)
    wavel = np.linspace(2.0e-6, 2.4e-6, 32)
    bw = np.full(32, 0.05e-6)

    # 1. Exact uniform-wavenumber boxcar identity on BinaryModelCartesian & Angular:
    dra, ddec, f = 34.0, -22.0, 0.28
    bin_cart = vm.BinaryModelCartesian(flux=f, dra=dra, ddec=ddec)
    sep = float(np.hypot(dra, ddec))
    pa = float(np.rad2deg(np.arctan2(dra, ddec)) % 360.0)
    bin_ang = vm.BinaryModelAngular(flux=f, sep=sep, pa=pa)

    v_ref_sigma = spiral_smearing.smeared_binary_wavenumber_quadrature(
        u, v, wavel, bw, dra, ddec, f, n_quad=64
    )
    err_cart = float(np.max(np.abs(np.asarray(vm.smeared_model(bin_cart, u, v, wavel, bw)) - v_ref_sigma)))
    err_ang = float(np.max(np.abs(np.asarray(vm.smeared_model(bin_ang, u, v, wavel, bw)) - v_ref_sigma)))

    # 2. Nested System offset accumulation + Resolved + extended UniformDisk at offset:
    sub = vm.System(
        inner_star=vm.PointSource(flux=1.0, dra=2.0, ddec=-1.0),
        disk_comp=vm.UniformDisk(diam=1.8, flux=0.4, dra=15.0, ddec=9.0),
        dra=10.0,
        ddec=-5.0,
        flux=0.7,
    )
    top = vm.System(
        origin_ud=vm.UniformDisk(diam=1.2, flux=1.0, dra=0.0, ddec=0.0),
        halo=vm.Resolved(flux=0.3),
        subsystem=sub,
    )
    v_top = np.asarray(vm.smeared_model(top, u, v, wavel, bw))

    # Manual leaf combination: weights are 1.0, 0.3, and 0.7 split in ratio 1.0 : 0.4
    w_origin = 1.0
    w_halo = 0.3
    w_inner = 0.7 * (1.0 / 1.4)
    w_disk = 0.7 * (0.4 / 1.4)
    w_tot = w_origin + w_halo + w_inner + w_disk
    v_origin = sky.vis_uniform_disk(u, v, wavel, 1.2, 0.0, 0.0)  # zero offset: unsmeared!
    v_inner = spiral_smearing.smeared_offset_phasor(u, v, wavel, bw, 12.0, -6.0)
    v_disk = sky.vis_uniform_disk(u, v, wavel, 1.8, 0.0, 0.0) * spiral_smearing.smeared_offset_phasor(
        u, v, wavel, bw, 25.0, 4.0
    )
    v_top_ref = (w_origin * v_origin + w_inner * v_inner + w_disk * v_disk) / w_tot
    err_nested = float(np.max(np.abs(v_top - v_top_ref)))

    # 3. Rotated scene commutes with rotating (u, v):
    rot_deg = 37.0
    rot_mod = vm.Rotated(top, rot_deg)
    v_rot = np.asarray(vm.smeared_model(rot_mod, u, v, wavel, bw))
    th = np.deg2rad(rot_deg)
    u_r = u * np.cos(th) - v * np.sin(th)
    v_r = u * np.sin(th) + v * np.cos(th)
    v_rot_ref = np.asarray(vm.smeared_model(top, u_r, v_r, wavel, bw))
    err_rot = float(np.max(np.abs(v_rot - v_rot_ref)))

    # 4. O((dlambda / lambda0)^2) convergence between sigma-boxcar and lambda-boxcar:
    u0, v0, lam0 = np.array([40.0]), np.array([-25.0]), np.array([2.2e-6])
    bw_wide = np.array([0.0176e-6])  # dlambda / lambda0 = 0.008
    bw_fine = np.array([0.0044e-6])  # dlambda / lambda0 = 0.002 (4x narrower)
    diff_wide = float(
        np.abs(
            np.asarray(vm.smeared_model(bin_cart, u0, v0, lam0, bw_wide))[0]
            - spiral_smearing.smeared_binary_wavelength_quadrature(u0, v0, lam0, bw_wide, dra, ddec, f, n_quad=64)[0]
        )
    )
    diff_fine = float(
        np.abs(
            np.asarray(vm.smeared_model(bin_cart, u0, v0, lam0, bw_fine))[0]
            - spiral_smearing.smeared_binary_wavelength_quadrature(u0, v0, lam0, bw_fine, dra, ddec, f, n_quad=64)[0]
        )
    )
    quad_scaling_ratio = diff_wide / diff_fine

    record("wavenumber_boxcar_max_err", max(err_cart, err_ang))
    record("nested_system_max_err", err_nested)
    record("rotated_scene_max_err", err_rot)
    record("wavelength_quadrature_4x_ratio", quad_scaling_ratio)
    assert err_cart < 1e-14
    assert err_ang < 1e-14
    assert err_nested < 1e-14
    assert err_rot < 1e-14
    assert 15.6 < quad_scaling_ratio < 16.4


@pytest.mark.validates(
    "virgil.models.smeared_model",
    roots=["mathematics"],
    headline="oidata_bandwidth_roundtrip_err",
)
def test_smeared_model_oidata_bandwidth_orbital_binary_and_type_errors(tmp_path):
    """OIData(..., bandwidth=True) reads EFF_BAND, with_bandwidth overrides or
    clears it, OIData smears OrbitalBinary via at(mjd) snapshots, and calling
    smeared_model directly on Image or OrbitalBinary raises TypeError naming
    the model."""
    uts = np.array([[0.0, 0.0], [42.0, 15.0], [12.0, 55.0]])
    wls = np.array([2.1e-6, 2.2e-6, 2.3e-6])
    dlambda = 0.044e-6
    fits_path = tmp_path / "bw_oidata.fits"
    cloud = sky.mix([sky.point(0.0, 0.0), sky.point(65.0, -40.0)], [1.0, 0.25])
    simulate.observe(
        fits_path,
        lambda u, v, w: sky.visibility(cloud, u, v, w),
        uts,
        hour_angles_h=[-1.0, 1.0],
        wavelengths=wls,
        dec_deg=-30.0,
        sigma_v2=0.002,
        sigma_cp_deg=0.1,
    )
    from astropy.io import fits

    with fits.open(fits_path, mode="update") as hdul:
        hdul["OI_WAVELENGTH"].data["EFF_BAND"][:] = dlambda
        hdul.flush()

    oid_off = voidata.OIData(fits_path, bandwidth=False)
    oid_on = voidata.OIData(fits_path, bandwidth=True)
    assert oid_off.bandwidth is None
    assert oid_on.bandwidth is not None

    scene = vm.BinaryModelCartesian(flux=0.25, dra=65.0, ddec=-40.0)
    pred_on = np.asarray(oid_on.model(scene))
    pred_via_with = np.asarray(oid_off.with_bandwidth(dlambda).model(scene))
    pred_cleared = np.asarray(oid_on.with_bandwidth(None).model(scene))
    pred_off = np.asarray(oid_off.model(scene))

    err_roundtrip = float(np.max(np.abs(pred_on - pred_via_with)))
    err_cleared = float(np.max(np.abs(pred_cleared - pred_off)))
    smearing_shift = float(np.max(np.abs(pred_on - pred_off)))

    # OrbitalBinary smeared per sample through OIData (via at(mjd)):
    orb = vorbits.KeplerOrbit(
        period=40.0, dt_peri=0.0, ecc=0.2, inc=42.0, omega=30.0, Omega=65.0, a_mas=28.0, t_ref=59000.0
    )
    ob_mod = vm.OrbitalBinary(orbit=orb, flux=0.3)
    pred_orb_sm = np.asarray(oid_on.model(ob_mod))
    mjd = np.asarray(oid_on.mjd)
    cvis_orb_ref = np.array(
        [
            np.asarray(
                vm.smeared_model(
                    ob_mod.at(float(mjd[k])),
                    np.array([float(oid_on.u[k])]),
                    np.array([float(oid_on.v[k])]),
                    np.array([float(oid_on.wavel[k])]),
                    np.array([dlambda]),
                )
            )[0]
            for k in range(oid_on.u.size)
        ]
    )
    pred_orb_ref = np.asarray(oid_on.standardize_model(cvis_orb_ref))
    err_orb = float(np.max(np.abs(pred_orb_sm - pred_orb_ref)))

    # Unsupported models raise TypeError naming the model class:
    img_mod = vm.Image(np.zeros((8, 8)), pixel_scale_mas=1.0)
    with pytest.raises(TypeError, match="Image"):
        vm.smeared_model(img_mod, np.array([10.0]), np.array([5.0]), np.array([2.2e-6]), 0.02e-6)
    with pytest.raises(TypeError, match="OrbitalBinary"):
        vm.smeared_model(ob_mod, np.array([10.0]), np.array([5.0]), np.array([2.2e-6]), 0.02e-6)

    record("oidata_bandwidth_roundtrip_err", max(err_roundtrip, err_cleared, err_orb))
    record("oidata_smearing_shift", smearing_shift)
    assert err_roundtrip < 1e-7  # EFF_BAND stored as float32 in OIFITS
    assert err_cleared < 1e-14
    assert err_orb < 1e-6
    assert smearing_shift > 1e-2


@pytest.mark.external
@pytest.mark.validates(
    "virgil.models.smeared_model",
    roots=["pmoired"],
    headline="pmoired_smeared_v2_diff",
)
def test_smeared_model_matches_pmoired_explicit_wl_kernel(tmp_path):
    """Compare smeared_model on a wide high-contrast binary against PMOIRED
    with an explicit variance-matched ``'wl kernel'`` (``auto=False``)."""
    pmoired = pytest.importorskip("pmoired")
    import pmoired.oimodels as om

    dra, ddec, f = 65.0, -42.0, 0.02
    cloud = sky.mix([sky.point(0.0, 0.0), sky.point(dra, ddec)], [1.0, f])
    uts = np.array([[0.0, 0.0], [36.0, 11.0], [12.0, 49.0], [60.0, -14.0]])
    wls = np.linspace(2.05e-6, 2.35e-6, 61)  # 5 nm spacing
    dwl_step = wls[1] - wls[0]
    dlambda = 0.022e-6  # 22 nm channel width
    # Match Gaussian spectral kernel second moment sigma_wl^2 = dlambda^2 / 12:
    wl_kernel_pix = float(np.sqrt(2.0 * np.log(2.0) / 3.0) * (dlambda / dwl_step))

    fits_path = tmp_path / "wide_binary_pmoired.fits"
    simulate.observe(
        fits_path,
        lambda u, v, w: sky.visibility(cloud, u, v, w),
        uts,
        hour_angles_h=[0.0],
        wavelengths=wls,
        dec_deg=-30.0,
        sigma_v2=0.001,
        sigma_cp_deg=0.1,
    )
    pm_params = {"c,ud": 0.0, "c,f": 1.0, "b,ud": 0.0, "b,f": f, "b,x": dra, "b,y": ddec}
    with contextlib.redirect_stdout(io.StringIO()):
        oi_sm = pmoired.OI(str(fits_path), verbose=False)
        oi_sm.setupFit({"obs": ["V2", "T3PHI"], "wl kernel": wl_kernel_pix}, auto=False)
        m_sm = om.VmodelOI(oi_sm.data[0], pm_params)

        oi_un = pmoired.OI(str(fits_path), verbose=False)
        oi_un.setupFit({"obs": ["V2", "T3PHI"]}, auto=False)
        m_un = om.VmodelOI(oi_un.data[0], pm_params)

    sl = slice(12, -12)
    w_in = wls[sl]
    v_mod = vm.System(star=vm.PointSource(flux=1.0), comp=vm.PointSource(flux=f, dra=dra, ddec=ddec))

    max_dv2 = 0.0
    max_dv2_unsmeared = 0.0
    for k in m_sm["OI_VIS2"]:
        u_m = float(m_sm["OI_VIS2"][k]["u"][0])
        v_m = float(m_sm["OI_VIS2"][k]["v"][0])
        vv_v2 = np.abs(np.asarray(vm.smeared_model(v_mod, np.full_like(w_in, u_m), np.full_like(w_in, v_m), w_in, dlambda))) ** 2
        max_dv2 = max(max_dv2, float(np.max(np.abs(vv_v2 - m_sm["OI_VIS2"][k]["V2"][0, sl]))))
        max_dv2_unsmeared = max(max_dv2_unsmeared, float(np.max(np.abs(vv_v2 - m_un["OI_VIS2"][k]["V2"][0, sl]))))

    max_dcp_deg = 0.0
    max_dcp_unsmeared = 0.0
    for k in m_sm["OI_T3"]:
        u1 = float(m_sm["OI_T3"][k]["u1"][0])
        v1 = float(m_sm["OI_T3"][k]["v1"][0])
        u2 = float(m_sm["OI_T3"][k]["u2"][0])
        v2_ = float(m_sm["OI_T3"][k]["v2"][0])
        c1 = np.asarray(vm.smeared_model(v_mod, np.full_like(w_in, u1), np.full_like(w_in, v1), w_in, dlambda))
        c2 = np.asarray(vm.smeared_model(v_mod, np.full_like(w_in, u2), np.full_like(w_in, v2_), w_in, dlambda))
        c3 = np.asarray(vm.smeared_model(v_mod, np.full_like(w_in, u1 + u2), np.full_like(w_in, v1 + v2_), w_in, dlambda))
        vv_cp = np.rad2deg(np.angle(c1 * c2 * np.conj(c3)))
        d_sm = (vv_cp - m_sm["OI_T3"][k]["T3PHI"][0, sl] + 180.0) % 360.0 - 180.0
        d_un = (vv_cp - m_un["OI_T3"][k]["T3PHI"][0, sl] + 180.0) % 360.0 - 180.0
        max_dcp_deg = max(max_dcp_deg, float(np.max(np.abs(d_sm))))
        max_dcp_unsmeared = max(max_dcp_unsmeared, float(np.max(np.abs(d_un))))

    record("pmoired_smeared_v2_diff", max_dv2)
    record("pmoired_smeared_cp_diff_deg", max_dcp_deg)
    record("pmoired_unsmeared_v2_gap", max_dv2_unsmeared)
    assert max_dv2_unsmeared > 1e-3
    assert max_dcp_unsmeared > 0.03
    assert max_dv2 < 8e-5
    assert max_dcp_deg < 5e-3


@pytest.mark.external
@pytest.mark.validates(
    "virgil.models.smeared_model",
    roots=["candid", "mathematics"],
    headline="candid_variance_matched_chi2r",
)
def test_smeared_model_matches_variance_matched_candid_nsmear3(tmp_path):
    """Because CANDID's 3-endpoint stencil ``{-1/2, 0, +1/2} * dwavel`` has
    spectral variance ``dwavel^2 / 6`` (twice a continuous boxcar's
    ``dlambda^2 / 12``), setting ``dwavel = dlambda / sqrt(2)`` matches the
    second moment so CANDID ``Nsmear = 3`` agrees with ``virgil.models.smeared_model``."""
    if _candid_python() is None:
        pytest.skip("CANDID environment (.venv-candid) is not installed")

    dra, ddec, f = 45.0, -28.0, 0.15
    dlambda_m = 0.035e-6
    uts = np.array([[0.0, 0.0], [38.0, 12.0], [11.0, 50.0], [64.0, -16.0]])
    wls = np.linspace(2.05e-6, 2.35e-6, 7)

    v_mod = vm.BinaryModelCartesian(flux=f, dra=dra, ddec=ddec)
    fits_path = tmp_path / "candid_var_matched.fits"
    simulate.observe(
        fits_path,
        lambda u, v, w: np.asarray(vm.smeared_model(v_mod, u, v, w, dlambda_m)),
        uts,
        hour_angles_h=[0.0],
        wavelengths=wls,
        dec_deg=-30.0,
        sigma_v2=5e-4,
        sigma_cp_deg=0.05,
    )

    dw_matched_um = (dlambda_m / np.sqrt(2.0)) * 1e6
    res_matched = _run_candid_smear_chi2(
        fits_path, dra=dra, ddec=ddec, flux_pct=100.0 * f, dwavel_um=dw_matched_um, n_smear=3
    )
    res_unsmeared = _run_candid_smear_chi2(
        fits_path, dra=dra, ddec=ddec, flux_pct=100.0 * f, dwavel_um=0.0, n_smear=3
    )

    record("candid_variance_matched_chi2r", res_matched["chi2r"])
    record("candid_unsmeared_chi2r_gap", res_unsmeared["chi2r"])
    assert res_unsmeared["chi2r"] > 15.0
    assert res_matched["chi2r"] < 0.06


@pytest.mark.validates(
    "virgil.models.smeared_model",
    roots=["mathematics"],
    kind="control",
    headline="coherent_vs_incoherent_v2_gap",
)
def test_control_smearing_incoherent_v2_and_unscaled_candid_nsmear3_fail(tmp_path):
    """Pin both bandwidth-smearing definition differences against
    ``virgil.models.smeared_model``:
    (1) incoherent ``E[|V|^2]`` (PMOIRED ``'wl kernel'`` on ``V2``) retains an
        ``O(f^2)`` floor ``f^2 / (1 + f)^2`` when fringes wash out, and
    (2) unscaled CANDID ``Nsmear = 3`` (``dwavel = dlambda``) has spectral
        variance ``dlambda^2 / 6`` — twice the continuous boxcar's ``dlambda^2 / 12``."""
    dra, ddec, f = 90.0, -65.0, 0.45
    u = np.array([75.0, -52.0, 88.0])
    v = np.array([34.0, 68.0, -41.0])
    wavel = np.full(3, 2.2e-6)
    dlambda = 0.08e-6

    v_mod = vm.BinaryModelCartesian(flux=f, dra=dra, ddec=ddec)
    v2_coh = np.abs(np.asarray(vm.smeared_model(v_mod, u, v, wavel, dlambda))) ** 2
    v2_incoh_dense, _ = spiral_smearing.incoherent_endpoint_binary_observables(
        u, v, wavel, dlambda, dra, ddec, f, n_smear=1001
    )
    v2_cand_unscaled, _ = spiral_smearing.coherent_endpoint_binary_observables(
        np.array([42.0, -35.0, 55.0]),
        np.array([15.0, 48.0, -22.0]),
        wavel,
        0.045e-6,
        45.0,
        -28.0,
        0.35,
        n_smear=3,
    )
    v2_boxcar_mod = (
        np.abs(
            np.asarray(
                vm.smeared_model(
                    vm.BinaryModelCartesian(flux=0.35, dra=45.0, ddec=-28.0),
                    np.array([42.0, -35.0, 55.0]),
                    np.array([15.0, 48.0, -22.0]),
                    wavel,
                    0.045e-6,
                )
            )
        )
        ** 2
    )

    gap_incoh = float(np.max(np.abs(v2_coh - v2_incoh_dense)))
    gap_nsmear3 = float(np.max(np.abs(v2_boxcar_mod - v2_cand_unscaled)))
    expected_floor_gap = (f**2) / (1.0 + f) ** 2  # ~ 0.0963 at f=0.45

    record("coherent_vs_incoherent_v2_gap", gap_incoh)
    record("boxcar_vs_unscaled_nsmear3_v2_gap", gap_nsmear3)
    assert gap_incoh > 0.8 * expected_floor_gap
    assert gap_nsmear3 > 0.005

    # Also verify live CANDID Nsmear=3 with unscaled dwavel=dlambda rejects boxcar-smeared data:
    if _candid_python() is not None:
        uts = np.array([[0.0, 0.0], [38.0, 12.0], [11.0, 50.0], [64.0, -16.0]])
        wls = np.linspace(2.05e-6, 2.35e-6, 7)
        fits_path = tmp_path / "coh_vs_candid_ctrl.fits"
        mod_ctrl = vm.BinaryModelCartesian(flux=0.15, dra=45.0, ddec=-28.0)
        simulate.observe(
            fits_path,
            lambda uu, vv, ww: np.asarray(vm.smeared_model(mod_ctrl, uu, vv, ww, 0.035e-6)),
            uts,
            hour_angles_h=[0.0],
            wavelengths=wls,
            dec_deg=-30.0,
            sigma_v2=5e-4,
            sigma_cp_deg=0.05,
        )
        res_unscaled = _run_candid_smear_chi2(
            fits_path, dra=45.0, ddec=-28.0, flux_pct=15.0, dwavel_um=0.035, n_smear=3
        )
        assert res_unscaled["chi2r"] > 15.0
