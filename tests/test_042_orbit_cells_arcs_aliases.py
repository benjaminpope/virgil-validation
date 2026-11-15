"""Independent validation of virgil 0.4.2 multi-epoch orbit cells, short arcs,
night summaries, shared-flux orbit scoring and period aliases.

All reference expectations are computed from first principles via
``crosscheck.orbit_cells_ref``, ``crosscheck.orbits``, ``crosscheck.sky``,
``crosscheck.chi2`` and SciPy quadrature, with synthetic OIFITS nights written
by ``crosscheck.oifits_writer`` / ``crosscheck.simulate``.
"""

from __future__ import annotations

import dataclasses
import gc
import math

import jax
import numpy as np
import pytest
from scipy import integrate, special

from crosscheck import array as carray, chi2 as ours_chi2, oifits_writer, orbit_cells_ref as ref, orbits as co, simulate, sky
from evidence.plugin import record

pytest.importorskip("jaxoplanet")
va = pytest.importorskip("virgil.aliases")
varcs = pytest.importorskip("virgil.arcs")
ve = pytest.importorskip("virgil.epochs")
vns = pytest.importorskip("virgil.night_summaries")
voc = pytest.importorskip("virgil.orbit_cells")
vos = pytest.importorskip("virgil.orbit_search")

from virgil.models import BinaryModelCartesian, OrbitalBinary, PointSource, System, UniformDisk  # noqa: E402
from virgil.oidata import OIData  # noqa: E402
from virgil.orbits import KeplerOrbit, PositionData  # noqa: E402
from virgil.spectra import PowerLaw  # noqa: E402

pytestmark = pytest.mark.x64

UTS3 = np.array([[-9.925, -20.335], [14.887, 30.502], [103.306, 43.999]])
WAVELENGTHS = np.linspace(2.0e-6, 2.4e-6, 4)
T0 = 60000.0


def _write_moving_night(
    path,
    mjd_center: float,
    hour_angles_h,
    orbit_tuple,
    flux: float,
    *,
    primary_diam_mas: float = 0.0,
    slope: float = 0.0,
    wavel0: float = 2.2e-6,
    sigma_v2: float = 0.008,
    sigma_cp_deg: float = 0.6,
    intra_night_motion: bool = True,
    seed: int | None = None,
):
    """Write a 3-telescope OIFITS night whose companion moves along ``orbit_tuple``."""
    period, dt_peri, ecc, inc, omega, Omega, a_mas = orbit_tuple
    stations = np.column_stack([UTS3, np.zeros(len(UTS3))])
    wl = np.atleast_1d(np.asarray(WAVELENGTHS, np.float32)).astype(float)
    pairs = np.array(carray.baselines(3))
    tris = np.array(carray.triangles(3))
    rng = np.random.default_rng(seed) if seed is not None else None

    v2_rec = {k: [] for k in ("vis2", "u", "v", "sta", "mjd")}
    t3_rec = {k: [] for k in ("phi", "u1", "v1", "u2", "v2", "sta", "mjd")}

    for idx, ha in enumerate(np.atleast_1d(hour_angles_h)):
        u, v = carray.snapshot_uv(stations, float(ha), -45.0, -24.6276)
        mjd = float(mjd_center) + (float(ha) / 24.0 if intra_night_motion else idx * 1e-5)
        t_eval = (mjd if intra_night_motion else float(mjd_center)) - T0
        dra_arr, ddec_arr, _, _ = ref.sky_state(
            t_eval, period, dt_peri, ecc, inc, omega, Omega, a_mas
        )
        x_k, y_k = float(dra_arr[0]), float(ddec_arr[0])

        def vis_at(uu, vv, x=x_k, y=y_k):
            uu2, vv2, ww2 = uu[:, None], vv[:, None], wl[None, :]
            f_wl = flux * (ww2 / wavel0) ** slope
            v_pri = (
                sky.vis_uniform_disk(uu2, vv2, ww2, primary_diam_mas)
                if primary_diam_mas > 0.0
                else sky.vis_point(uu2, vv2, ww2)
            )
            v_sec = sky.vis_point(uu2, vv2, ww2, x, y)
            return (v_pri + f_wl * v_sec) / (1.0 + f_wl)

        v2_rec["vis2"].append(np.abs(vis_at(u, v)) ** 2)
        v2_rec["u"].append(u)
        v2_rec["v"].append(v)
        v2_rec["sta"].append(pairs)
        v2_rec["mjd"].append(np.full(len(u), mjd))

        cp, u1, v1, u2, v2_t3 = carray.closure_phase(vis_at, u, v, 3)
        t3_rec["phi"].append(np.rad2deg(cp))
        t3_rec["u1"].append(u1)
        t3_rec["v1"].append(v1)
        t3_rec["u2"].append(u2)
        t3_rec["v2"].append(v2_t3)
        t3_rec["sta"].append(tris)
        t3_rec["mjd"].append(np.full(len(u1), mjd))

    v2 = {k: np.concatenate(val) for k, val in v2_rec.items()}
    t3 = {k: np.concatenate(val) for k, val in t3_rec.items()}
    if rng is not None:
        v2["vis2"] = v2["vis2"] + sigma_v2 * rng.standard_normal(v2["vis2"].shape)
        t3["phi"] = t3["phi"] + sigma_cp_deg * rng.standard_normal(t3["phi"].shape)
    v2["err"] = np.full(v2["vis2"].shape, max(sigma_v2, 1e-6))
    t3["err"] = np.full(t3["phi"].shape, max(sigma_cp_deg, 1e-6))
    oifits_writer.write(path, wavelengths=wl, stations_xyz=stations, vis2=v2, t3=t3)
    return path


def _make_synthetic_summaries(
    times_mjd,
    orbit_tuple,
    flux: float = 0.14,
    *,
    pos_sigma: float = 0.07,
    vel_sigma: float = 0.30,
    flux_sigma: float = 0.05,
    loglike: float = 110.0,
    ln_scale: float = 0.0,
    outlier_idx: int | None = None,
    extra_scatter_mas: float = 0.0,
    rng_seed: int = 17,
):
    """Build synthetic ``NightSummary`` objects from textbook Keplerian states."""
    period, dt_peri, ecc, inc, omega, Omega, a_mas = orbit_tuple
    dra, ddec, vra, vdec = ref.sky_state(
        np.asarray(times_mjd, float) - T0, period, dt_peri, ecc, inc, omega, Omega, a_mas
    )
    rng = np.random.default_rng(rng_seed)
    s_quote = math.exp(ln_scale)
    base_cov = np.diag([pos_sigma**2, pos_sigma**2, vel_sigma**2, vel_sigma**2, flux_sigma**2])
    quoted_cov = base_cov * (s_quote**2)
    quoted_hess = np.linalg.inv(quoted_cov)[None, :, :]
    n_obs = 24.0
    quoted_chi2 = n_obs / (s_quote**2)
    chi2_null_base = n_obs + 2.0 * float(loglike)
    quoted_loglike = float(loglike) + ref.rescaled_peak_log_scale([n_obs], [n_obs], s_quote)
    quoted_loglike_null = ref.rescaled_peak_log_scale([n_obs], [chi2_null_base], s_quote)

    out = []
    for k, t_k in enumerate(times_mjd):
        x_k, y_k, vx_k, vy_k = float(dra[k]), float(ddec[k]), float(vra[k]), float(vdec[k])
        if extra_scatter_mas > 0.0:
            dx, dy = rng.normal(0.0, extra_scatter_mas, size=2)
            x_k += float(dx)
            y_k += float(dy)
        if outlier_idx is not None and k == outlier_idx:
            # Displaced far off the true Keplerian track to act as a genuine outlier night
            x_k, y_k, vx_k, vy_k = -14.5, 11.2, 1.8, -2.1

        mean = np.array([x_k, y_k, vx_k, vy_k, math.log(flux)], dtype=float)
        pk = vns.NightPeak(
            mean=mean,
            cov=quoted_cov.copy(),
            loglike=quoted_loglike,
            n=np.array([n_obs]),
            chi2=np.array([quoted_chi2]),
            hessian=quoted_hess.copy(),
            rescalable=True,
        )
        out.append(
            vns.NightSummary(
                t_ref=float(t_k),
                t_ss=0.05,
                n_frames=6,
                blocks=("vis",),
                peaks=(pk,),
                loglike_null=quoted_loglike_null,
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# 1. Reference checks for crosscheck.orbit_cells_ref
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "crosscheck.orbit_cells_ref",
    roots=["mathematics"],
    kind="reference",
    headline="max_vel_err",
)
def test_orbit_cells_ref_sky_state_and_thiele_innes_projection():
    """Analytical sky velocity matches 4th-order Richardson finite differences of sky position."""
    t = np.linspace(0.3, 11.7, 9)
    orbit = (12.4, 1.8, 0.58, 128.0, 73.0, 215.0, 9.2)
    dra, ddec, vra, vdec = ref.sky_state(t, *orbit)

    h = 5e-4
    p_p1 = co.position(t + h, *orbit)
    p_m1 = co.position(t - h, *orbit)
    p_p2 = co.position(t + 2 * h, *orbit)
    p_m2 = co.position(t - 2 * h, *orbit)
    fd_vra = (-p_p2[0] + 8.0 * p_p1[0] - 8.0 * p_m1[0] + p_m2[0]) / (12.0 * h)
    fd_vdec = (-p_p2[1] + 8.0 * p_p1[1] - 8.0 * p_m1[1] + p_m2[1]) / (12.0 * h)
    max_vel_err = float(max(np.max(np.abs(vra - fd_vra)), np.max(np.abs(vdec - fd_vdec))))

    cov_pos = np.broadcast_to(np.diag([0.05**2, 0.06**2]), (len(t), 2, 2)).copy()
    cov_vel = np.broadcast_to(np.diag([0.2**2, 0.25**2]), (len(t), 2, 2)).copy()
    theta, chi2 = ref.fit_thiele_innes_state(
        t, dra, ddec, cov_pos, orbit[0], orbit[1], orbit[2], vra=vra, vdec=vdec, cov_vel=cov_vel
    )
    ti_true = np.array(co.thiele_innes(orbit[6], orbit[3], orbit[4], orbit[5]), dtype=float)
    max_ti_err = float(np.max(np.abs(theta - ti_true)))

    record("max_vel_err", max_vel_err)
    record("max_ti_err", max_ti_err)
    assert max_vel_err < 2e-11
    assert max_ti_err < 1e-12
    assert chi2 < 1e-22


@pytest.mark.validates(
    "crosscheck.orbit_cells_ref",
    roots=["mathematics"],
    kind="reference",
    headline="max_box_err",
)
def test_orbit_cells_ref_flux_box_and_jacobi_quadrature():
    """Exact log_flux_box matches adaptive quadrature and Gauss-Jacobi nodes integrate Beta(1, 9)."""
    ell = -9.210340371976184
    test_pts = [(-4.5, 0.25), (-0.1, 0.3), (0.0, 0.2), (0.45, 0.35), (1.4, 0.5)]
    errs = []
    for m_f, s_f in test_pts:
        val, _ = integrate.quad(
            lambda x: np.exp(-0.5 * ((x - m_f) / s_f) ** 2) / (math.sqrt(2.0 * math.pi) * s_f * abs(ell)),
            ell,
            0.0,
            epsabs=0.0,
            epsrel=1e-13,
        )
        errs.append(abs(float(ref.log_flux_box_exact(m_f, s_f, ell)) - math.log(val)))
    max_box_err = float(max(errs))

    # 3-node Gauss-Jacobi rule on [0, 1] for Beta(1, 9) is exact up to degree 2*3 - 1 = 5
    roots, weights = special.roots_jacobi(3, 8.0, 0.0)
    nodes = 0.5 * (roots + 1.0)
    norm_w = weights / np.sum(weights)
    poly_err = abs(float(np.sum(norm_w * nodes**5)) - float(special.beta(6.0, 9.0) / special.beta(1.0, 9.0)))
    eps_lo_err = abs(ref.beta_eps_lo_node(5) - float(np.min(nodes)))

    cells = ref.alias_band_edges([0.0, 3.0, 14.0], (1.5, 40.0))
    mass_err = abs(float(np.sum(np.exp(ref.cell_log_masses(cells, 1.5, 40.0)))) - 1.0)

    record("max_box_err", max_box_err)
    record("poly_err", poly_err)
    assert max_box_err < 1e-13
    assert poly_err < 1e-15
    assert eps_lo_err < 1e-16
    assert mass_err < 1e-15


@pytest.mark.validates(
    "crosscheck.orbit_cells_ref",
    roots=["mathematics"],
    kind="reference",
    headline="gamma_quad_err",
)
def test_orbit_cells_ref_polar_disc_mc_kde_and_rescalable_quad(tmp_path):
    """Verify rescalable scale quadrature against the analytical Gamma closed form, MC KDE normalization, and polar disc convergence."""
    # 1. Exact Gamma closed form for interior flux (log_flux_box == -ln|ell| to < 1e-15) and wide s_max
    ell = -9.210340371976184
    cov5 = np.diag([0.07**2, 0.08**2, 0.25**2, 0.30**2, 0.04**2])
    n_tot, chi2_tot, dim = 30.0, 42.0, 5
    s_wide = 1e4
    got_q = ref.quad_rescalable_outlier_log_density(
        [12.5], [cov5], [n_tot], [chi2_tot], [-6.2], [-4.5], ell, velocity=True, s_max=s_wide
    )
    _, logdet = np.linalg.slogdet(cov5)
    # Substitution t = 0.5 * chi2 * exp(-2 u) gives int exp(-(n - d) u - 0.5 chi2 exp(-2 u)) du = 0.5 * (0.5 * chi2)^(-(n - d)/2) * Gamma((n - d)/2)
    closed_q = (
        12.5
        + 0.5 * dim * math.log(2.0 * math.pi)
        + 0.5 * float(logdet)
        - 6.2
        - math.log(abs(ell))
        - math.log(2.0 * math.log(s_wide))
        + 0.5 * chi2_tot
        - math.log(2.0)
        - 0.5 * (n_tot - dim) * math.log(0.5 * chi2_tot)
        + float(special.gammaln(0.5 * (n_tot - dim)))
    )
    gamma_quad_err = abs(got_q - closed_q)

    # 2. Sky-plane normalization of mc_kepler_state_prior_log_density in 3D and 5D
    r_nodes = np.geomspace(1e-2, 42.0, 56)
    dr_ln = math.log(r_nodes[1] / r_nodes[0])
    pts_r = np.column_stack([r_nodes, np.zeros_like(r_nodes), np.zeros_like(r_nodes), np.zeros_like(r_nodes), np.full_like(r_nodes, -2.0)])
    lp3_r = ref.mc_kepler_state_prior_log_density(pts_r, (5.0, 25.0), (1.0, 20.0), 0.85, ell, velocity=False, n_mc=8192, seed=7)
    mc_norm_3d_err = abs(float(np.sum(np.exp(lp3_r) * abs(ell) * (2.0 * math.pi) * (r_nodes**2) * dr_ln)) - 1.0)

    v_nodes = np.geomspace(1e-4, 28.0, 44)
    dv_ln = math.log(v_nodes[1] / v_nodes[0])
    psi_nodes = np.linspace(0.0, 2.0 * math.pi, 20, endpoint=False)
    dpsi = 2.0 * math.pi / len(psi_nodes)
    vv, pp = np.meshgrid(v_nodes, psi_nodes, indexing="ij")
    # Build one compact query batch of (56 * 44 * 20 = 49,280) points for a single KDE construction
    rrr, vvv, ppp = np.meshgrid(r_nodes, v_nodes, psi_nodes, indexing="ij")
    pts_5 = np.column_stack([
        rrr.ravel(),
        np.zeros(rrr.size),
        (vvv * np.cos(ppp)).ravel(),
        (vvv * np.sin(ppp)).ravel(),
        np.full(rrr.size, -2.0),
    ])
    lp5 = ref.mc_kepler_state_prior_log_density(pts_5, (5.0, 25.0), (1.0, 20.0), 0.85, ell, velocity=True, n_mc=8192, seed=7).reshape(rrr.shape)
    mc_norm_5d_err = abs(float(np.sum(np.exp(lp5) * abs(ell) * (2.0 * math.pi) * (rrr**2) * (vvv**2) * dr_ln * dv_ln * dpsi)) - 1.0)

    # 3. Polar-disc convergence of broad_disc_marginal_loglike + algebraic checks on remaining helpers
    p_fits = tmp_path / "ref_disc.fits"
    simulate.observe(
        p_fits,
        lambda u, v, w: (sky.vis_point(u, v, w) + 0.15 * sky.vis_point(u, v, w, 2.0, -1.5)) / 1.15,
        UTS3 * 0.22,
        hour_angles_h=[-1.0, 1.0],
        wavelengths=WAVELENGTHS,
        sigma_v2=0.015,
        sigma_cp_deg=1.2,
        rng=np.random.default_rng(19),
        phase_noise="triangle",
    )
    d_raw = ours_chi2.load(p_fits)
    b_med = ref.broad_disc_marginal_loglike(d_raw, 4.0, (0.05, 0.35), s_max=10.0, n_r=28, n_theta=56, n_flux=96)
    b_fine = ref.broad_disc_marginal_loglike(d_raw, 4.0, (0.05, 0.35), s_max=10.0, n_r=40, n_theta=80, n_flux=192)
    disc_conv_err = abs(b_med - b_fine)

    # Exact Aitken WLS line recovery and ranking / rescaling / period-bound helpers
    dt = np.array([-0.1, 0.0, 0.15])
    proj = ref.wls_line_projection(dt, weights=np.array([1.0, 2.0, 1.5]))
    pv_rec = proj @ (3.2 + 1.8 * dt)
    assert np.allclose(pv_rec, [3.2, 1.8], atol=1e-14)
    assert list(ref.quantum_rank_scores([1.0002, np.nan, 1.0004, 0.5], quantum=1e-3)) == [0, 2, 3, 1]
    assert abs(ref.period_cut_bound([0.0, 20.0], 1.0, 0.1, resolution=2.0) - 40.0) < 1e-14
    assert ref.period_window_bounds([0.0, 10.0], 0.01, 0.05, resolution=2.0, p_floor=0.3) == (0.3, 40.0)

    record("gamma_quad_err", gamma_quad_err)
    record("mc_norm_5d_err", mc_norm_5d_err)
    record("disc_conv_err", disc_conv_err)
    assert gamma_quad_err < 1e-11
    assert mc_norm_3d_err < 3e-3
    assert mc_norm_5d_err < 6e-3
    assert disc_conv_err < 5e-4


# ---------------------------------------------------------------------------
# 2. virgil.aliases: AliasBand, alias_bands, period_cut, period_window,
#    position_profile, AliasBandResult, AliasResult
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.aliases.AliasBand",
    "virgil.aliases.alias_bands",
    "virgil.aliases.period_cut",
    "virgil.aliases.period_window",
    roots=["mathematics"],
    headline="max_edge_err",
)
def test_alias_bands_partition_and_period_window_bounds():
    """Alias bands partition [p_min, p_max] and period_cut/period_window cover 2*dt_min, duplicates, p_floor, and night_length."""
    times = np.array([60000.0, 60003.5, 60011.0, 60024.0])
    p_range = (3.2, 65.0)
    bands = va.alias_bands(times, p_range)
    assert all(isinstance(b, va.AliasBand) for b in bands)

    ref_bands = {n: (lo, hi) for n, lo, hi in ref.alias_band_edges(times, p_range)}
    assert {b.n for b in bands} == set(ref_bands)

    edge_errs = [
        max(abs(b.p_lo - ref_bands[b.n][0]), abs(b.p_hi - ref_bands[b.n][1]))
        for b in bands
    ]
    # Sorted by period increasing, bands must tile [p_min, p_max] seamlessly without gaps
    by_p = sorted(bands, key=lambda b: b.p_lo)
    gap_errs = [abs(by_p[i + 1].p_lo - by_p[i].p_hi) for i in range(len(by_p) - 1)]
    max_edge_err = float(max(edge_errs + gap_errs + [abs(by_p[0].p_lo - p_range[0]), abs(by_p[-1].p_hi - p_range[1])]))

    # (a) Intra-night smearing bound dominates 2 * dt_min (31.416 d > 7.0 d)
    p_cut_a = va.period_cut(times, field=15.0, night_length=0.2, resolution=1.2, motion_fraction=0.5, p_floor=0.2)
    ref_cut_a = ref.period_cut_bound(times, 15.0, 0.2, resolution=1.2, motion_fraction=0.5, p_floor=0.2)
    p_win_a = va.period_window(times, field=15.0, night_length=0.2, resolution=1.2)
    ref_win_a = ref.period_window_bounds(times, 15.0, 0.2, resolution=1.2)

    # (b) Inter-night Nyquist bound 2 * dt_min = 50.0 d strictly exceeds intra-night smearing 31.416 d
    times_dt = np.array([0.0, 25.0, 62.0]) + T0
    p_cut_b = va.period_cut(times_dt, field=15.0, night_length=0.2, resolution=1.2, motion_fraction=0.5, p_floor=0.2)
    ref_cut_b = ref.period_cut_bound(times_dt, 15.0, 0.2, resolution=1.2, motion_fraction=0.5, p_floor=0.2)
    p_win_b = va.period_window(times_dt, field=15.0, night_length=0.2, resolution=1.2)
    assert p_cut_b > p_win_b[0] + 18.0

    # (c) Duplicate timestamps ignore zero diffs; < 2 unique timestamps fall back to intra-night bound; p_floor binds
    p_cut_dup = va.period_cut(np.array([0.0, 0.0, 25.0, 62.0]) + T0, field=15.0, night_length=0.2, resolution=1.2)
    ref_cut_dup = ref.period_cut_bound(np.array([0.0, 0.0, 25.0, 62.0]) + T0, 15.0, 0.2, resolution=1.2)
    p_cut_one = va.period_cut(np.array([0.0, 0.0]) + T0, field=15.0, night_length=0.2, resolution=1.2)
    ref_cut_one = ref.period_cut_bound(np.array([0.0, 0.0]) + T0, 15.0, 0.2, resolution=1.2)
    p_cut_flr = va.period_cut(np.array([0.0, 0.05, 10.0]) + T0, field=0.02, night_length=0.05, resolution=1.2, p_floor=0.4)
    ref_cut_flr = ref.period_cut_bound(np.array([0.0, 0.05, 10.0]) + T0, 0.02, 0.05, resolution=1.2, p_floor=0.4)

    # (d) Narrow-field regime (2*pi*field < motion_fraction*resolution, night_length=0.6 > p_floor=0.2):
    # virgil's prose docs (orbit_methods.md) clamp at night_length (0.6 d), whereas API docstrings omit night_length (0.31416 d)
    t_narrow = np.array([0.0, 0.1, 10.0]) + T0
    p_win_narrow = va.period_window(t_narrow, field=0.05, night_length=0.6, resolution=1.2, motion_fraction=0.5, p_floor=0.2)
    ref_win_narrow = ref.period_window_bounds(t_narrow, 0.05, 0.6, resolution=1.2, motion_fraction=0.5, p_floor=0.2)
    assert abs(p_win_narrow[0] - 0.6) < 1e-13
    assert abs(ref_win_narrow[0] - 2.0 * math.pi * 0.05 * 0.6 / 0.6) < 1e-13

    cut_err = max(
        abs(p_cut_a - ref_cut_a),
        abs(p_win_a[0] - ref_win_a[0]),
        abs(p_win_a[1] - ref_win_a[1]),
        abs(p_cut_b - ref_cut_b),
        abs(p_cut_dup - ref_cut_dup),
        abs(p_cut_one - ref_cut_one),
        abs(p_cut_flr - ref_cut_flr),
    )

    record("max_edge_err", max(max_edge_err, cut_err))
    record("dt_min_excess_days", float(p_cut_b - p_win_b[0]))
    record("narrow_field_night_clamp_diff", float(p_win_narrow[0] - ref_win_narrow[0]))
    assert max_edge_err < 1e-13
    assert cut_err < 1e-13


@pytest.mark.validates(
    "virgil.aliases.position_profile",
    roots=["mathematics"],
    headline="wls_chi2_diff",
)
def test_position_profile_recovers_keplerian_alias_band():
    """position_profile matches exact Thiele-Innes WLS chi2 per band, picks the true band, and handles sign_free."""
    times = np.array([0.0, 3.0, 6.0, 15.0, 24.0]) + T0
    orbit = (12.0, 1.0, 0.25, 42.0, 55.0, 110.0, 9.0)
    dra, ddec, _, _ = ref.sky_state(times - T0, *orbit)
    cov = np.broadcast_to(np.diag([0.05**2, 0.05**2]), (len(times), 2, 2)).copy()

    pos = PositionData(mjd=times, dra=dra, ddec=ddec, cov=cov, t_ref=T0)
    bands = va.alias_bands(times, (7.0, 18.0))
    prof = va.position_profile(pos, bands, k=12.0, n_best=2, eccs=np.array([0.0, 0.25, 0.5]))

    by_n = {row["n"]: row for row in prof}
    best_chi2 = float(by_n[2]["chi2"])
    wls_diffs = []
    for row in prof:
        orb_r, chi2_r = row["orbits"][0]
        _, chi2_ref = ref.fit_thiele_innes_state(
            times - T0, dra, ddec, cov, float(orb_r.period), float(orb_r.dt_peri), float(orb_r.ecc)
        )
        wls_diffs.append(abs(float(chi2_r) - chi2_ref))
    wls_chi2_diff = float(max(wls_diffs))

    # Flip one night's label by 180 deg: signed profile fails (>1000 chi2), sign_free=True recovers it
    dra_flip = dra.copy()
    ddec_flip = ddec.copy()
    dra_flip[2] *= -1.0
    ddec_flip[2] *= -1.0
    pos_flip = PositionData(mjd=times, dra=dra_flip, ddec=ddec_flip, cov=cov, t_ref=T0)
    prof_signed = va.position_profile(
        pos_flip, [b for b in bands if b.n == 2], k=12.0, n_best=2, eccs=np.array([0.25]), sign_free=False
    )
    prof_sf = va.position_profile(
        pos_flip, [b for b in bands if b.n == 2], k=12.0, n_best=2, eccs=np.array([0.25]), sign_free=True
    )
    signed_min_chi2 = float(min(c for _, c in prof_signed[0]["orbits"]))
    sf_min_chi2 = float(min(c for _, c in prof_sf[0]["orbits"]))

    record("wls_chi2_diff", wls_chi2_diff)
    record("best_chi2", best_chi2)
    record("sf_min_chi2", sf_min_chi2)
    assert wls_chi2_diff < 1e-8
    assert best_chi2 < 1e-2
    assert best_chi2 < 1e-3 * min(float(by_n[1]["chi2"]), float(by_n[3]["chi2"]))
    assert abs(by_n[2]["period"] - 12.0) < 0.15
    assert signed_min_chi2 > 500.0
    assert sf_min_chi2 < 1e-2


@pytest.mark.validates("virgil.aliases.position_profile", roots=["mathematics"], kind="control")
def test_position_profile_wrong_band_control():
    """Incommensurate period band cannot fit multi-epoch Keplerian positions at sub-0.05 mas precision."""
    times = np.array([0.0, 1.8, 4.3, 9.1, 15.7, 24.0]) + T0
    orbit = (12.0, 1.2, 0.3, 42.0, 55.0, 110.0, 9.0)
    dra, ddec, _, _ = ref.sky_state(times - T0, *orbit)
    cov = np.broadcast_to(np.diag([0.04**2, 0.04**2]), (len(times), 2, 2)).copy()
    pos = PositionData(mjd=times, dra=dra, ddec=ddec, cov=cov, t_ref=T0)

    wrong_band = [va.AliasBand(n=5, p_lo=24.0 / 5.5, p_hi=24.0 / 4.5)]
    prof = va.position_profile(pos, wrong_band, k=9.0, n_best=2)
    record("wrong_band_chi2", float(prof[0]["chi2"]))
    assert prof[0]["chi2"] > 50.0


@pytest.mark.validates(
    "virgil.aliases.AliasBandResult",
    "virgil.aliases.AliasResult",
    roots=["mathematics"],
    headline="max_resid_diff",
)
def test_alias_result_and_band_result_json_roundtrip_and_residuals(tmp_path):
    """AliasResult and AliasBandResult round-trip through JSON and match crosscheck.chi2 residuals."""
    # Dyadic MJD offsets representable without rounding in float32 around MJD 60000
    times = np.array([0.0, 2.5, 5.5, 9.75, 16.0]) + T0
    orbit = (11.5, 1.0, 0.2, 45.0, 40.0, 100.0, 6.5)
    flux = 0.22
    datasets = []
    ep_dict = {}
    raw_files = []
    for k, mjd_k in enumerate(times):
        p = _write_moving_night(
            tmp_path / f"alias_n{k}.fits",
            float(mjd_k),
            np.linspace(-2.5, 2.5, 5),
            orbit,
            flux,
            sigma_v2=0.008,
            sigma_cp_deg=0.5,
            intra_night_motion=False,
            seed=100 + k,
        )
        raw_files.append(ours_chi2.load(p))
        d_k = OIData(str(p))
        datasets.append(d_k)
        ep_dict[f"n{k}"] = d_k

    grid_al = {
        "dra": np.linspace(-10.0, 10.0, 41),
        "ddec": np.linspace(-10.0, 10.0, 41),
        "flux": np.geomspace(0.08, 0.45, 7),
    }
    pos_al = ve.epoch_positions(ve.Epochs(ep_dict), grid_al, n_peaks=2)

    res = va.fit_orbit_aliases(
        datasets,
        p_range=(9.5, 14.5),
        positions=pos_al,
        times=times,
        t_ref=T0,
        a_range=(3.0, 14.0),
        flux_range=(0.05, 0.5),
        ecc_max=0.6,
        n_candidates=24,
        n_refine=2,
        max_steps=80,
        n_is=250,
        n_samples=40,
        vis_amp=False,
        primary_diam=None,
        short_arcs=False,
        seed=3,
    )
    assert isinstance(res, va.AliasResult)
    assert isinstance(res.best, va.AliasBandResult)

    json_path = tmp_path / "aliases_full.json"
    res.save(json_path)
    reloaded = va.AliasResult.load(json_path)
    assert len(reloaded.bands) == len(res.bands)
    assert abs(reloaded.best.log_z - res.best.log_z) < 1e-12
    assert len(reloaded.table()) == len(res.bands)
    assert reloaded.tempering is False
    assert reloaded.block_dof() == {}

    scene = reloaded.scene()
    assert isinstance(scene, OrbitalBinary)

    # Compare AliasResult.residuals and chi2_per_night against crosscheck.chi2 at the fitted best parameters
    best = reloaded.best.best
    vg_resids = reloaded.residuals(datasets)
    vg_chi2 = reloaded.chi2_per_night(datasets)

    diffs = []
    for d_raw, r_vg, c_vg in zip(raw_files, vg_resids, vg_chi2):
        dt_k = float(np.mean(d_raw["v2_mjd"]) - reloaded.t_ref)
        xr, yr = co.position(
            dt_k,
            best["period"],
            best["dt_peri"],
            best["ecc"],
            best["inc"],
            best["omega"],
            best["Omega"],
            best["a_mas"],
        )
        x_k, y_k = float(np.squeeze(xr)), float(np.squeeze(yr))
        f_fit = float(best["flux"])
        r_ref = ours_chi2.residuals(
            d_raw,
            lambda u, v, w, f=f_fit, x=x_k, y=y_k: (
                sky.vis_point(u, v, w) + f * sky.vis_point(u, v, w, x, y)
            )
            / (1.0 + f),
        )
        n_v2 = d_raw["v2"].size
        diffs.append(float(np.max(np.abs(np.asarray(r_vg["vis"]) - r_ref[:n_v2]))))
        diffs.append(float(np.max(np.abs(np.asarray(r_vg["phi"]) - r_ref[n_v2:]))))
        diffs.append(abs(float(c_vg["chi2"]) - float(np.sum(r_ref**2))) / max(1.0, float(c_vg["n"])))

    max_resid_diff = float(max(diffs))
    record("max_resid_diff", max_resid_diff)
    assert max_resid_diff < 1e-10

    with pytest.raises(ValueError):
        reloaded.residuals(datasets[:-1])


# ---------------------------------------------------------------------------
# 3. virgil.arcs: NightMotion, night_motion, CycleCounts, unwrap_cycles,
#    arc_profile
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.arcs.NightMotion",
    "virgil.arcs.night_motion",
    roots=["mathematics"],
    headline="max_vel_zscore",
)
def test_night_motion_measures_intranight_velocity_from_moving_oifits(tmp_path):
    """night_motion recovers intra-night companion sky velocity and marks single-exposure nights unmeasured."""
    orbit = (14.0, 0.8, 0.20, 38.0, 50.0, 120.0, 8.0)
    flux = 0.18
    mjds = [T0 + 0.0, T0 + 2.5, T0 + 5.8]
    ep_dict = {}
    for k, mjd_k in enumerate(mjds):
        has = np.linspace(-1.0, 1.0, 6) if k < 2 else np.array([0.0])
        p = _write_moving_night(
            tmp_path / f"arc_m{k}.fits",
            mjd_k,
            has,
            orbit,
            flux,
            sigma_v2=0.005,
            sigma_cp_deg=0.35,
            seed=200 + k,
        )
        ep_dict[f"n{k}"] = OIData(str(p))

    epochs = ve.Epochs(ep_dict)
    grid = {
        "dra": np.linspace(-12.0, 12.0, 49),
        "ddec": np.linspace(-12.0, 12.0, 49),
        "flux": np.geomspace(0.08, 0.35, 9),
    }
    positions = ve.epoch_positions(epochs, grid, n_peaks=3)
    motion = varcs.night_motion(epochs, positions, jitter=0.02)
    assert isinstance(motion, varcs.NightMotion)
    assert list(motion.measured) == [True, True, False]

    sub = motion.take([0, 1])
    assert len(sub.mjd) == 2 and np.all(sub.measured)

    _, _, vra_true, vdec_true = ref.sky_state(sub.mjd - T0, *orbit)
    z_ra = np.abs(sub.vel[:, 0] - vra_true) / np.sqrt(sub.cov[:, 0, 0])
    z_dec = np.abs(sub.vel[:, 1] - vdec_true) / np.sqrt(sub.cov[:, 1, 1])
    max_vel_zscore = float(max(np.max(z_ra), np.max(z_dec)))

    record("max_vel_zscore", max_vel_zscore)
    assert max_vel_zscore < 2.5


@pytest.mark.validates(
    "virgil.arcs.CycleCounts",
    "virgil.arcs.unwrap_cycles",
    "virgil.arcs.arc_profile",
    roots=["mathematics"],
    headline="wls_state_chi2_diff",
)
def test_unwrap_cycles_and_arc_profile_break_alias_comb():
    """Short-arc velocities + close nights unwrap cycle counts across a long seasonal gap."""
    times = np.array([0.0, 2.0, 4.0, 48.0]) + T0
    orbit = (12.0, 0.0, 0.30, 44.0, 62.0, 135.0, 9.5)
    dra, ddec, vra, vdec = ref.sky_state(times - T0, *orbit)

    cov_pos = np.broadcast_to(np.diag([0.20**2, 0.20**2]), (len(times), 2, 2)).copy()
    cov_vel = np.broadcast_to(np.diag([0.15**2, 0.15**2]), (len(times), 2, 2)).copy()
    pos = PositionData(mjd=times, dra=dra, ddec=ddec, cov=cov_pos, t_ref=T0)
    motion = varcs.NightMotion(
        mjd=times,
        vel=np.column_stack([vra, vdec]),
        cov=cov_vel,
        measured=np.ones(len(times), dtype=bool),
    )

    p_range = (8.5, 18.0)
    bands = va.alias_bands(times, p_range)
    assert len(bands) >= 4

    cc = varcs.unwrap_cycles(pos, motion, p_range=p_range, ecc_max=0.6, k=9.0)
    assert isinstance(cc, varcs.CycleCounts)
    admitted = [b for b in bands if cc.admits(b)]
    rejected = [b for b in bands if not cc.admits(b)]
    # Must admit the true N = round(48 / 12) = 4 band and reject the bulk of the alias comb
    assert any(b.n == 4 for b in admitted)
    assert len(admitted) < len(bands)

    prof = varcs.arc_profile(pos, motion, admitted, ecc_max=0.6, k=27.0, n_best=2, eccs=np.array([0.1, 0.3, 0.5]))
    prof_rej = varcs.arc_profile(pos, motion, rejected[:2], ecc_max=0.6, k=18.0, n_best=1, eccs=np.array([0.1, 0.3, 0.5]))
    min_rej_chi2 = min(float(r["chi2"]) for r in prof_rej)

    by_n = {r["n"]: r for r in prof}
    best_orb, best_chi2 = by_n[4]["orbits"][0]
    _, ref_chi2 = ref.fit_thiele_innes_state(
        times - T0,
        dra,
        ddec,
        cov_pos,
        float(best_orb.period),
        float(best_orb.dt_peri),
        float(best_orb.ecc),
        vra=vra,
        vdec=vdec,
        cov_vel=cov_vel,
    )
    wls_state_chi2_diff = abs(float(best_chi2) - ref_chi2)
    period_err_days = abs(float(by_n[4]["period"]) - 12.0)

    record("wls_state_chi2_diff", wls_state_chi2_diff)
    record("period_err_days", period_err_days)
    assert wls_state_chi2_diff < 1e-8
    assert period_err_days < 0.12
    assert float(best_chi2) < 90.0
    assert float(best_chi2) < 0.05 * min_rej_chi2


@pytest.mark.validates("virgil.arcs.arc_profile", roots=["mathematics"], kind="control")
def test_arc_profile_negated_velocity_control():
    """Negating intra-night sky velocities makes arc_profile strongly reject the true Keplerian orbit."""
    times = np.array([0.0, 1.8, 3.9, 60.0]) + T0
    orbit = (12.0, 1.1, 0.30, 44.0, 62.0, 135.0, 9.5)
    dra, ddec, vra, vdec = ref.sky_state(times - T0, *orbit)
    cov_pos = np.broadcast_to(np.diag([0.04**2, 0.04**2]), (len(times), 2, 2)).copy()
    cov_vel = np.broadcast_to(np.diag([0.15**2, 0.15**2]), (len(times), 2, 2)).copy()
    pos = PositionData(mjd=times, dra=dra, ddec=ddec, cov=cov_pos, t_ref=T0)
    bad_motion = varcs.NightMotion(
        mjd=times,
        vel=-np.column_stack([vra, vdec]),
        cov=cov_vel,
        measured=np.ones(len(times), dtype=bool),
    )
    band5 = [b for b in va.alias_bands(times, (10.5, 13.5)) if b.n == 5]
    prof = varcs.arc_profile(pos, bad_motion, band5, ecc_max=0.6, k=9.0, eccs=np.array([0.3]))
    record("negated_vel_chi2", float(prof[0]["chi2"]))
    assert prof[0]["chi2"] > 100.0


# ---------------------------------------------------------------------------
# 4. virgil.night_summaries: NightPeak, NightSummary, keep_peaks,
#    night_summaries
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.night_summaries.NightPeak",
    "virgil.night_summaries.NightSummary",
    roots=["mathematics"],
    headline="max_algebra_err",
)
def test_night_peak_and_summary_rescaling_and_wls_projection():
    """NightPeak closed-form error rescaling and NightSummary WLS projection match textbook algebra."""
    rng = np.random.default_rng(42)
    A1 = rng.standard_normal((5, 5))
    A2 = rng.standard_normal((5, 5))
    H1 = A1.T @ A1 + np.eye(5)
    H2 = A2.T @ A2 + 0.5 * np.eye(5)
    hess = np.stack([H1, H2], axis=0)
    cov1 = np.linalg.inv(H1 + H2)

    pk = vns.NightPeak(
        mean=np.array([4.0, -3.0, 0.5, -0.4, -2.0]),
        cov=cov1,
        loglike=85.0,
        n=np.array([18.0, 12.0]),
        chi2=np.array([22.0, 15.0]),
        hessian=hess,
        rescalable=True,
    )
    scales = np.array([1.4, 0.8])
    prec_err = float(np.max(np.abs(pk.precision(scales) - ref.rescaled_peak_precision(hess, scales))))
    cov_err = float(np.max(np.abs(pk.covariance(scales) - np.linalg.inv(ref.rescaled_peak_precision(hess, scales)))))
    ls_err = abs(float(pk.log_scale(scales)) - ref.rescaled_peak_log_scale(pk.n, pk.chi2, scales))

    offsets = (-0.11, -0.04, 0.03, 0.12)
    weights = (1.0, 2.0, 1.5, 0.8)
    ns = vns.NightSummary(
        t_ref=T0,
        t_ss=float(np.sum(np.square(offsets))),
        n_frames=4,
        blocks=("vis", "phi"),
        peaks=(pk,),
        loglike_null=0.0,
        offsets=offsets,
        weights=weights,
    )
    proj_err = float(np.max(np.abs(ns.projection() - ref.wls_line_projection(offsets, weights))))
    jcov_vg = ns.covariance(pk, scale=scales, jitter=0.25)
    jcov_ref = ref.night_summary_jitter_cov(pk.covariance(scales), 0.25, ns.n_frames, ns.t_ss)
    jcov_err = float(np.max(np.abs(jcov_vg - jcov_ref)))

    ns_single = vns.NightSummary(
        t_ref=T0, t_ss=0.0, n_frames=1, blocks=("vis", "phi"), peaks=(pk,), loglike_null=0.0
    )
    single_cov = ns_single.covariance(pk, scale=1.0, jitter=0.2)
    assert np.isinf(single_cov[2, 2]) and np.isinf(single_cov[3, 3])
    assert np.allclose(ns_single.projection(), np.array([[1.0], [0.0]]))

    max_algebra_err = max(prec_err, cov_err, ls_err, proj_err, jcov_err)
    record("max_algebra_err", max_algebra_err)
    assert max_algebra_err < 1e-12


@pytest.mark.validates(
    "virgil.night_summaries.keep_peaks",
    roots=["mathematics"],
    headline="boundary_margin_err",
)
def test_keep_peaks_matches_gauss_jacobi_beta_node():
    """keep_peaks retains peaks iff (1 - eps_lo) L_j >= delta * eps_lo * exp(b_k) at the Jacobi node."""
    k_nights = 5
    b_k = 12.0
    delta = 0.001
    eps_lo = ref.beta_eps_lo_node(k_nights)
    crit_ll = b_k + math.log(delta * eps_lo / (1.0 - eps_lo))

    def _peak(ll):
        return vns.NightPeak(
            mean=np.zeros(5),
            cov=np.eye(5),
            loglike=float(ll),
            n=np.array([10.0]),
            chi2=np.array([10.0]),
            hessian=np.eye(5)[None, :, :],
        )

    cand_lls = [crit_ll + 5.0, crit_ll + 1e-6, crit_ll - 1e-6, crit_ll - 4.0]
    nights = [
        vns.NightSummary(
            t_ref=T0 + k,
            t_ss=0.04,
            n_frames=5,
            blocks=("vis",),
            peaks=tuple(_peak(ll) for ll in cand_lls) if k == 0 else (_peak(crit_ll + 10.0),),
            loglike_null=0.0,
        )
        for k in range(k_nights)
    ]
    kept = vns.keep_peaks(nights, np.full(k_nights, b_k), delta=delta)
    got_lls = [p.loglike for p in kept[0].peaks]
    exp_mask = ref.keep_peak_mask(cand_lls, b_k, k_nights, delta=delta)
    exp_lls = [ll for ll, m in zip(cand_lls, exp_mask) if m]

    boundary_margin_err = float(max(abs(a - b) for a, b in zip(got_lls, exp_lls))) if got_lls == exp_lls else 1.0
    record("boundary_margin_err", boundary_margin_err)
    assert len(got_lls) == 2
    assert got_lls == exp_lls


@pytest.mark.validates("virgil.night_summaries.keep_peaks", roots=["mathematics"], kind="control")
def test_keep_peaks_prior_mean_eps_control():
    """Replacing the smallest Gauss-Jacobi node eps_lo with E[eps] = 0.1 drops peaks that keep_peaks retains."""
    k_nights = 5
    b_k = 12.0
    delta = 0.001
    eps_lo = ref.beta_eps_lo_node(k_nights)
    eps_mean = 0.1
    crit_jacobi = b_k + math.log(delta * eps_lo / (1.0 - eps_lo))
    crit_mean = b_k + math.log(delta * eps_mean / (1.0 - eps_mean))
    mid_ll = 0.5 * (crit_jacobi + crit_mean)

    pk = vns.NightPeak(
        mean=np.zeros(5),
        cov=np.eye(5),
        loglike=mid_ll,
        n=np.array([10.0]),
        chi2=np.array([10.0]),
        hessian=np.eye(5)[None, :, :],
    )
    nights = [
        vns.NightSummary(t_ref=T0 + k, t_ss=0.04, n_frames=5, blocks=("vis",), peaks=(pk,), loglike_null=0.0)
        for k in range(k_nights)
    ]
    kept = vns.keep_peaks(nights, np.full(k_nights, b_k), delta=delta)
    record("threshold_gap_nats", crit_mean - crit_jacobi)
    assert len(kept[0].peaks) == 1
    assert crit_mean - crit_jacobi > 1.0


@pytest.mark.validates(
    "virgil.night_summaries.night_summaries",
    roots=["mathematics"],
    headline="max_pos_err_mas",
)
def test_night_summaries_visibility_linear_motion_recovery(tmp_path):
    """night_summaries refines whole-night linear companion tracks from visibility OIFITS data."""
    orbit = (10.0, 0.6, 0.2, 40.0, 45.0, 115.0, 7.5)
    flux = 0.18
    mjds = [T0 + 0.0, T0 + 2.5]
    ep_dict = {}
    for k, mjd_k in enumerate(mjds):
        p = _write_moving_night(
            tmp_path / f"ns_n{k}.fits",
            mjd_k,
            np.linspace(-1.2, 1.2, 6),
            orbit,
            flux,
            sigma_v2=0.004,
            sigma_cp_deg=0.25,
            seed=300 + k,
        )
        ep_dict[f"n{k}"] = OIData(str(p))

    epochs = ve.Epochs(ep_dict)
    grid = {
        "dra": np.linspace(-11.0, 11.0, 45),
        "ddec": np.linspace(-11.0, 11.0, 45),
        "flux": np.geomspace(0.08, 0.35, 8),
    }
    positions = ve.epoch_positions(epochs, grid, n_peaks=2)
    summaries = vns.night_summaries(epochs, positions)
    assert len(summaries) == 2

    pos_errs = []
    vel_z = []
    for s in summaries:
        dra_t, ddec_t, vra_t, vdec_t = ref.sky_state(s.t_ref - T0, *orbit)
        best_pk = s.peaks[0]
        pos_errs.append(float(math.hypot(best_pk.mean[0] - dra_t[0], best_pk.mean[1] - ddec_t[0])))
        vel_z.append(float(abs(best_pk.mean[2] - vra_t[0]) / math.sqrt(best_pk.cov[2, 2])))
        vel_z.append(float(abs(best_pk.mean[3] - vdec_t[0]) / math.sqrt(best_pk.cov[3, 3])))

    max_pos_err_mas = float(max(pos_errs))
    record("max_pos_err_mas", max_pos_err_mas)
    assert max_pos_err_mas < 0.12
    assert max(vel_z) < 3.0


# ---------------------------------------------------------------------------
# 5. virgil.orbit_search: SharedFlux, rank_scores, OrbitScores, score_orbits
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.orbit_search.rank_scores",
    roots=["mathematics"],
    headline="max_index_diff",
)
def test_rank_scores_quantum_tie_breaking_and_nonfinite_tail():
    """rank_scores rounds to quantum nats, breaks ties by candidate index, and ranks non-finite scores last."""
    scores = np.array([1.0, 3.0002, 3.0002 + 1e-8, float("-inf"), 2.0, float("nan"), 3.0002 - 1e-8])
    got = vos.rank_scores(scores, quantum=1e-3)
    exp = ref.quantum_rank_scores(scores, quantum=1e-3)
    max_index_diff = int(np.max(np.abs(got - exp)))
    record("max_index_diff", max_index_diff)
    assert max_index_diff == 0


@pytest.mark.validates(
    "virgil.orbit_search.SharedFlux",
    "virgil.orbit_search.OrbitScores",
    "virgil.orbit_search.score_orbits",
    roots=["mathematics"],
    headline="flux_rel_err",
)
def test_score_orbits_shared_flux_and_chromatic_slope_recovery(tmp_path):
    """score_orbits marginalizes shared companion flux and chromatic slope across epochs."""
    orbit = (14.0, 1.5, 0.25, 48.0, 30.0, 95.0, 8.2)
    true_flux = 0.16
    true_slope = 1.5
    wavel0 = 2.2e-6
    ep_dict = {}
    for k, mjd_k in enumerate([T0 + 0.0, T0 + 3.5, T0 + 9.0]):
        p = _write_moving_night(
            tmp_path / f"so_n{k}.fits",
            mjd_k,
            np.linspace(-2.0, 2.0, 4),
            orbit,
            true_flux,
            slope=true_slope,
            wavel0=wavel0,
            sigma_v2=0.004,
            sigma_cp_deg=0.25,
            seed=400 + k,
        )
        ep_dict[f"e{k}"] = OIData(str(p))

    epochs = ve.Epochs(ep_dict)
    cand_true = KeplerOrbit(
        period=orbit[0],
        dt_peri=orbit[1],
        ecc=orbit[2],
        inc=orbit[3],
        omega=orbit[4],
        Omega=orbit[5],
        a_mas=orbit[6],
        t_ref=T0,
    )
    cand_off = KeplerOrbit(
        period=orbit[0] + 1.8,
        dt_peri=orbit[1],
        ecc=orbit[2],
        inc=orbit[3],
        omega=orbit[4] + 25.0,
        Omega=orbit[5],
        a_mas=orbit[6],
        t_ref=T0,
    )
    shared = vos.SharedFlux(grid=(0.02, 0.6, 16), slope=(-2.0, 4.0, 13), wavel0=wavel0)

    def scene_fn(orb, flux_spec):
        return OrbitalBinary(orb, flux_spec)

    scores = vos.score_orbits(scene_fn, epochs, [cand_true, cand_off], shared=shared)
    assert isinstance(scores, vos.OrbitScores)
    order = scores.order()
    assert int(order[0]) == 0

    flux_rel_err = abs(float(scores.flux[0, 0]) - true_flux) / true_flux
    slope_err = abs(float(scores.slope[0, 0]) - true_slope)
    record("flux_rel_err", flux_rel_err)
    record("slope_err", slope_err)
    assert flux_rel_err < 0.06
    assert slope_err < 0.5
    assert float(scores.score[0] - scores.score[1]) > 30.0


@pytest.mark.validates("virgil.orbit_search.score_orbits", roots=["mathematics"], kind="control")
def test_score_orbits_nonlinear_model_control(tmp_path):
    """score_orbits rejects scene callables that violate flux linearity or zero-flux orbit independence."""
    p = _write_moving_night(
        tmp_path / "so_ctrl.fits", T0, [-1.0, 1.0], (14.0, 1.5, 0.2, 45.0, 30.0, 90.0, 8.0), 0.15, seed=410
    )
    epochs = ve.Epochs({"e0": OIData(str(p))})
    orb1 = KeplerOrbit(
        period=14.0, dt_peri=1.5, ecc=0.2, inc=45.0, omega=30.0, Omega=90.0, a_mas=8.0, t_ref=T0
    )
    orb2 = KeplerOrbit(
        period=11.0, dt_peri=0.5, ecc=0.3, inc=60.0, omega=80.0, Omega=140.0, a_mas=6.5, t_ref=T0
    )
    n_rejected = 0
    with pytest.raises(ValueError):
        vos.score_orbits(lambda o, f: OrbitalBinary(o, f * f), epochs, [orb1, orb2])
    n_rejected += 1
    with pytest.raises(ValueError):
        vos.score_orbits(lambda o, f: OrbitalBinary(o, f + 0.2), epochs, [orb1, orb2])
    n_rejected += 1
    record("n_rejected_invalid_scenes", n_rejected)
    assert n_rejected == 2


# ---------------------------------------------------------------------------
# 6. virgil.epochs: EpochPeaks, NightFlag, PrimaryDiameter, broad_loglike,
#    estimate_primary_diameter, sidelobe_flags
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.epochs.EpochPeaks",
    "virgil.epochs.NightFlag",
    "virgil.epochs.sidelobe_flags",
    roots=["mathematics"],
    headline="weight_sum_err",
)
def test_epoch_peaks_night_flag_and_sidelobe_flags(tmp_path):
    """EpochPeaks normalizes weights, NightFlag separates blocking flags, and sidelobe_flags catches secondary peaks."""
    p0 = tmp_path / "ep_clean.fits"
    simulate.observe(
        p0,
        lambda u, v, w: (sky.vis_point(u, v, w) + 0.16 * sky.vis_point(u, v, w, 5.5, -4.0)) / 1.16,
        UTS3,
        hour_angles_h=np.linspace(-2.5, 2.5, 5),
        wavelengths=WAVELENGTHS,
        sigma_v2=0.008,
        sigma_cp_deg=0.6,
        rng=np.random.default_rng(501),
        phase_noise="triangle",
    )
    # Single-snapshot sparse coverage creates competitive fringe sidelobes
    p1 = tmp_path / "ep_close.fits"
    simulate.observe(
        p1,
        lambda u, v, w: (sky.vis_point(u, v, w) + 0.15 * sky.vis_point(u, v, w, 0.25, 0.20)) / 1.15,
        UTS3,
        hour_angles_h=[0.0],
        wavelengths=WAVELENGTHS[:2],
        sigma_v2=0.01,
        sigma_cp_deg=0.8,
        rng=np.random.default_rng(502),
        phase_noise="triangle",
    )
    epochs = ve.Epochs({"n0": OIData(str(p0)), "n1": OIData(str(p1))})
    grid = {
        "dra": np.linspace(-10.0, 10.0, 41),
        "ddec": np.linspace(-10.0, 10.0, 41),
        "flux": np.geomspace(0.05, 0.4, 7),
    }
    pos = ve.epoch_positions(epochs, grid, n_peaks=4, flag_sep_blocks=False)
    pk0 = pos.peaks[0]
    assert isinstance(pk0, ve.EpochPeaks)
    assert np.all(np.diff(pk0.marginal) <= 1e-12)
    weight_sum_err = abs(float(np.sum(pk0.weight)) - 1.0)

    # Check NightFlag blocking vs non-blocking behaviour directly and via sidelobe_flags
    nf_nonblock = ve.NightFlag(code="sep-below-resolution", reason="sep < 0.5 res", blocking=False)
    nf_block = ve.NightFlag(code="low-gap", reason="gap < 5", blocking=True)
    assert nf_nonblock.blocking is False and nf_block.blocking is True

    pred_at_best = (np.array([pos.dra[0], pos.dra[1]]), np.array([pos.ddec[0], pos.ddec[1]]), np.array([pos.flux[0], pos.flux[1]]))
    flags_best = ve.sidelobe_flags(epochs, pos, pred_at_best, k_sigma=3.0, delta_nats=50.0)
    assert flags_best == [None, None]

    # Displace night 0 prediction > 2 sigma onto a secondary peak while keeping wide delta_nats so surface stays within margin
    sec_dra = float(pk0.dra[1]) if len(pk0.dra) > 1 else float(pos.dra[0] + 3.0)
    sec_ddec = float(pk0.ddec[1]) if len(pk0.ddec) > 1 else float(pos.ddec[0] + 3.0)
    sec_flux = float(pk0.flux[1]) if len(pk0.flux) > 1 else float(pos.flux[0])
    pred_sec = (np.array([sec_dra, pos.dra[1]]), np.array([sec_ddec, pos.ddec[1]]), np.array([sec_flux, pos.flux[1]]))
    flags_sec = ve.sidelobe_flags(epochs, pos, pred_sec, k_sigma=2.0, delta_nats=500.0)
    assert isinstance(flags_sec[0], ve.NightFlag)
    assert flags_sec[0].code == "orbit-prefers-secondary"

    record("weight_sum_err", weight_sum_err)
    assert weight_sum_err < 1e-12


@pytest.mark.validates(
    "virgil.epochs.broad_loglike",
    roots=["mathematics"],
    headline="max_broad_diff",
)
def test_broad_loglike_matches_disc_loguniform_quadrature(tmp_path):
    """broad_loglike at sub-fringe resolution with default coarse_nats matches independent polar-disc quadrature."""
    uts_compact = UTS3 * 0.22  # finest fringe spacing ~14.4 mas
    p0 = tmp_path / "broad_bin.fits"
    p1 = tmp_path / "broad_weak.fits"
    simulate.observe(
        p0,
        lambda u, v, w: (sky.vis_point(u, v, w) + 0.18 * sky.vis_point(u, v, w, 2.4, -1.8)) / 1.18,
        uts_compact,
        hour_angles_h=[-1.5, 0.0, 1.5],
        wavelengths=WAVELENGTHS,
        sigma_v2=0.012,
        sigma_cp_deg=1.0,
        rng=np.random.default_rng(510),
        phase_noise="triangle",
    )
    simulate.observe(
        p1,
        lambda u, v, w: sky.vis_point(u, v, w),
        uts_compact,
        hour_angles_h=[-1.5, 0.0, 1.5],
        wavelengths=WAVELENGTHS,
        sigma_v2=0.025,
        sigma_cp_deg=25.0,
        rng=np.random.default_rng(511),
        phase_noise="triangle",
    )
    d0, d1 = OIData(str(p0)), OIData(str(p1))
    r0, r1 = ours_chi2.load(p0), ours_chi2.load(p1)

    a_max = 4.5
    flux_range = (0.05, 0.35)
    dofs = [np.array([1.0, 0.8]), np.array([0.7, 0.9])]

    # Sub-fringe step = 0.25 mas (< 1/50 fringe) with default coarse_nats=30.0 screening active
    got_smax = ve.broad_loglike(
        [d0, d1], dofs, a_max=a_max, flux_range=flux_range, step=0.25, s_max=10.0, n_flux=41
    )
    want_smax = np.array(
        [
            ref.broad_disc_marginal_loglike(r0, a_max, flux_range, dof=dofs[0], s_max=10.0),
            ref.broad_disc_marginal_loglike(r1, a_max, flux_range, dof=dofs[1], s_max=10.0),
        ],
        dtype=float,
    )

    got_unb = ve.broad_loglike(
        [d0, d1], dofs, a_max=a_max, flux_range=flux_range, step=0.25, s_max=None, n_flux=41
    )
    want_unb = np.array(
        [
            ref.broad_disc_marginal_loglike(r0, a_max, flux_range, dof=dofs[0], s_max=None),
            ref.broad_disc_marginal_loglike(r1, a_max, flux_range, dof=dofs[1], s_max=None),
        ],
        dtype=float,
    )

    err_smax = float(np.max(np.abs(got_smax - want_smax)))
    err_unb = float(np.max(np.abs(got_unb - want_unb)))
    max_broad_diff = max(err_smax, err_unb)

    record("max_broad_diff", max_broad_diff)
    record("broad_smax_err", err_smax)
    record("broad_unbounded_err", err_unb)
    assert max_broad_diff < 0.025


@pytest.mark.validates(
    "virgil.epochs.PrimaryDiameter",
    "virgil.epochs.estimate_primary_diameter",
    roots=["mathematics"],
    headline="diam_err_mas",
)
def test_estimate_primary_diameter_resolves_ud_and_rejects_point(tmp_path):
    """estimate_primary_diameter recovers a resolved uniform-disk primary and rejects a point primary."""
    orbit = (15.0, 1.0, 0.15, 35.0, 40.0, 100.0, 9.0)
    ud_true = 1.2
    ep_ud = {}
    ep_pt = {}
    for k, mjd_k in enumerate([T0 + 0.0, T0 + 4.0]):
        p_ud = _write_moving_night(
            tmp_path / f"pri_ud_{k}.fits",
            mjd_k,
            [-2.0, 0.0, 2.0],
            orbit,
            0.16,
            primary_diam_mas=ud_true,
            sigma_v2=0.005,
            sigma_cp_deg=0.35,
            intra_night_motion=False,
            seed=520 + k,
        )
        p_pt = _write_moving_night(
            tmp_path / f"pri_pt_{k}.fits",
            mjd_k,
            [-2.0, 0.0, 2.0],
            orbit,
            0.16,
            primary_diam_mas=0.0,
            sigma_v2=0.005,
            sigma_cp_deg=0.35,
            intra_night_motion=False,
            seed=530 + k,
        )
        ep_ud[f"u{k}"] = OIData(str(p_ud))
        ep_pt[f"p{k}"] = OIData(str(p_pt))

    grid = {
        "dra": np.linspace(-12.0, 12.0, 49),
        "ddec": np.linspace(-12.0, 12.0, 49),
        "flux": np.geomspace(0.08, 0.3, 7),
    }
    d_grid = np.linspace(0.3, 2.4, 15)

    epochs_ud = ve.Epochs(ep_ud)
    pos_ud = ve.epoch_positions(epochs_ud, grid, n_peaks=2)
    est_ud = ve.estimate_primary_diameter(epochs_ud, pos_ud, diam_grid=d_grid)
    assert isinstance(est_ud, ve.PrimaryDiameter)
    assert est_ud.resolved is True
    assert est_ud.delta_nats > 5.0

    epochs_pt = ve.Epochs(ep_pt)
    pos_pt = ve.epoch_positions(epochs_pt, grid, n_peaks=2)
    est_pt = ve.estimate_primary_diameter(epochs_pt, pos_pt, diam_grid=d_grid)
    assert est_pt.resolved is False
    assert est_pt.delta_nats < 5.0

    diam_err_mas = abs(float(est_ud.diam) - ud_true)
    record("diam_err_mas", diam_err_mas)
    assert diam_err_mas < 0.2


# ---------------------------------------------------------------------------
# 7. virgil.orbit_cells: full position-level cell search & outlier stack
# ---------------------------------------------------------------------------


@pytest.mark.validates(
    "virgil.orbit_cells.PeriodDomain",
    "virgil.orbit_cells.period_domain",
    "virgil.orbit_cells.choose_anchors",
    "virgil.orbit_cells.Cell",
    "virgil.orbit_cells.make_cells",
    roots=["mathematics"],
    headline="mass_sum_err",
)
def test_period_domain_anchors_and_cell_prior_mass_conservation():
    """period_domain sets [P_cut, P_max], choose_anchors picks end-quarter minima, and cell masses sum to 1."""
    orbit = (12.0, 1.0, 0.25, 45.0, 40.0, 100.0, 8.5)
    times = np.array([0.0, 0.2, 3.0, 11.0, 21.0, 24.0]) + T0
    sigmas = [0.12, 0.04, 0.09, 0.08, 0.03, 0.11]

    summaries = []
    for t_k, sig in zip(times, sigmas):
        cov = np.diag([sig**2, sig**2, 0.3**2, 0.3**2, 0.05**2])
        pk = vns.NightPeak(
            mean=np.array([5.0, -3.0, 0.2, 0.1, -2.0]),
            cov=cov,
            loglike=90.0,
            n=np.array([20.0]),
            chi2=np.array([20.0]),
            hessian=np.linalg.inv(cov)[None, :, :],
        )
        summaries.append(
            vns.NightSummary(t_ref=float(t_k), t_ss=0.05, n_frames=5, blocks=("vis",), peaks=(pk,), loglike_null=0.0)
        )

    dom = voc.period_domain(summaries, a_range=(1.0, 25.0), gap_days=0.5, p_floor=0.2)
    assert isinstance(dom, voc.PeriodDomain)
    assert dom.overridden is False
    assert dom.binding == "2dt_min"
    assert abs(dom.sigma_min - 0.03) < 1e-12

    i_a, i_b = voc.choose_anchors(summaries)
    # First quarter [0, 6]: index 1 has sigma 0.04 < 0.12, 0.09; last quarter [18, 24]: index 4 has sigma 0.03 < 0.11
    assert (i_a, i_b) == (1, 4)

    cells_default = voc.make_cells(summaries[i_a].t_ref, summaries[i_b].t_ref, dom)
    assert all(isinstance(c, voc.Cell) for c in cells_default)
    mass_default_err = abs(float(np.sum([math.exp(c.log_mass) for c in cells_default])) - 1.0)

    dom_ov = voc.period_domain(summaries, p_range=(6.0, 30.0))
    assert dom_ov.overridden is True and dom_ov.binding == "p_range"
    cells_custom = voc.make_cells(T0, T0 + 20.0, dom_ov, p_edges=[16.0, 9.0])
    ref_lm = ref.cell_log_masses([(c.n, c.p_lo, c.p_hi) for c in cells_custom], 6.0, 30.0)
    custom_lm_err = float(np.max(np.abs(np.array([c.log_mass for c in cells_custom]) - ref_lm)))
    mass_custom_err = abs(float(np.sum([math.exp(c.log_mass) for c in cells_custom])) - 1.0)

    mass_sum_err = max(mass_default_err, mass_custom_err, custom_lm_err)
    record("mass_sum_err", mass_sum_err)
    assert mass_sum_err < 1e-14


@pytest.mark.validates(
    "virgil.orbit_cells.OutlierPrior",
    "virgil.orbit_cells.outlier_prior",
    "virgil.orbit_cells.outlier_log_density",
    roots=["mathematics"],
    headline="max_log_flux_box_err",
)
def test_outlier_prior_log_flux_box_and_laplace_outlier_density():
    """Verify OutlierPrior normalization/KDE and outlier_log_density (fixed and rescalable=True) without self-calls."""
    summaries = _make_synthetic_summaries(np.array([0.0, 3.0, 9.0, 18.0]) + T0, (11.0, 1.0, 0.2, 40.0, 50.0, 110.0, 8.0))
    dom = voc.period_domain(summaries, p_range=(5.0, 25.0))
    prior = voc.outlier_prior(dom, a_range=(1.0, 20.0), e_max=0.85, flux_floor=1e-4, n=8192, seed=0)
    assert isinstance(prior, voc.OutlierPrior)
    abs_ell = abs(prior.ln_f_floor)

    # 1. Exact normal-CDF box convolution on [ln_f_floor, 0], including positive m_f > 0
    m_grid = np.array([-8.5, -3.0, -0.05, 0.0, 0.35, 1.1])
    s_grid = np.array([0.2, 0.15, 0.25, 0.3, 0.4, 0.5])
    got_box = np.asarray(prior.log_flux_box(m_grid, s_grid), dtype=float)
    exp_box = ref.log_flux_box_exact(m_grid, s_grid, prior.ln_f_floor)
    max_log_flux_box_err = float(np.max(np.abs(got_box - exp_box)))
    assert np.all(np.isfinite(got_box))

    # 2. Independent polar sky-plane normalization of OutlierPrior.log_density (3D and chunked 5D) + MC KDE check
    r_nodes = np.geomspace(1e-2, 45.0, 60)
    dr_ln = math.log(r_nodes[1] / r_nodes[0])
    th_nodes = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
    dth = 2.0 * math.pi / len(th_nodes)
    rr, tt = np.meshgrid(r_nodes, th_nodes, indexing="ij")
    pts_3d = np.column_stack([(rr * np.cos(tt)).ravel(), (rr * np.sin(tt)).ravel(), np.zeros(rr.size), np.zeros(rr.size), np.full(rr.size, -2.0)])
    lp3_grid = np.asarray(prior.log_density(pts_3d, velocity=False, sigma_f=None), dtype=float).reshape(rr.shape)
    norm_3d_err = abs(float(np.sum(np.exp(lp3_grid) * abs_ell * (rr**2) * dr_ln * dth)) - 1.0)
    # Azimuthal rotational invariance across theta
    assert float(np.max(np.ptp(lp3_grid, axis=1))) < 1e-12

    v_nodes = np.geomspace(1e-4, 30.0, 44)
    dv_ln = math.log(v_nodes[1] / v_nodes[0])
    psi_nodes = np.linspace(0.0, 2.0 * math.pi, 20, endpoint=False)
    dpsi = 2.0 * math.pi / len(psi_nodes)
    vv, pp = np.meshgrid(v_nodes, psi_nodes, indexing="ij")
    norm_5d = 0.0
    for rv_val in r_nodes:
        pts_slice = np.column_stack([
            np.full(vv.size, rv_val),
            np.zeros(vv.size),
            (vv * np.cos(pp)).ravel(),
            (vv * np.sin(pp)).ravel(),
            np.full(vv.size, -2.0),
        ])
        lp_slice = np.asarray(prior.log_density(pts_slice, velocity=True, sigma_f=None), dtype=float).reshape(vv.shape)
        norm_5d += float(np.sum(np.exp(lp_slice) * abs_ell * (2.0 * math.pi) * (rv_val**2) * (vv**2) * dr_ln * dv_ln * dpsi))
    norm_5d_err = abs(norm_5d - 1.0)

    # Pointwise comparison against independent crosscheck Monte Carlo Keplerian draws + Jacobians
    cov5 = np.diag([0.08**2, 0.09**2, 0.35**2, 0.40**2, 0.30**2])
    sigma_f1 = math.sqrt(cov5[4, 4])
    state_pos_mf = np.array([[4.5, -3.2, 0.4, -0.5, +0.25]])
    state_in_mf = np.array([[4.5, -3.2, 0.4, -0.5, -2.0]])

    mc_lp5_pos = float(ref.mc_kepler_state_prior_log_density(
        state_pos_mf, (5.0, 25.0), (1.0, 20.0), 0.85, prior.ln_f_floor, velocity=True, sigma_f=np.array([sigma_f1]), n_mc=8192, seed=11
    )[0])
    mc_lp3_pos = float(ref.mc_kepler_state_prior_log_density(
        state_pos_mf, (5.0, 25.0), (1.0, 20.0), 0.85, prior.ln_f_floor, velocity=False, sigma_f=np.array([sigma_f1]), n_mc=8192, seed=11
    )[0])
    vg_lp5_pos = float(np.squeeze(prior.log_density(state_pos_mf, velocity=True, sigma_f=np.array([sigma_f1]))))
    vg_lp3_pos = float(np.squeeze(prior.log_density(state_pos_mf, velocity=False, sigma_f=np.array([sigma_f1]))))
    kde_3d_err = abs(vg_lp3_pos - mc_lp3_pos)
    kde_5d_err = abs(vg_lp5_pos - mc_lp5_pos)

    # 3. Fixed-scale (rescalable=False) outlier_log_density against MC reference (zero self-calls)
    pk_pos_mf = vns.NightPeak(
        mean=state_pos_mf[0],
        cov=cov5,
        loglike=75.0,
        n=np.array([20.0]),
        chi2=np.array([20.0]),
        hessian=np.linalg.inv(cov5)[None, :, :],
        rescalable=False,
    )
    ns_5d = vns.NightSummary(t_ref=T0, t_ss=0.05, n_frames=5, blocks=("vis",), peaks=(pk_pos_mf,), loglike_null=-3.5)
    ns_3d = vns.NightSummary(t_ref=T0 + 2.0, t_ss=0.0, n_frames=1, blocks=("vis",), peaks=(pk_pos_mf,), loglike_null=-3.5)
    ns_empty = vns.NightSummary(t_ref=T0 + 4.0, t_ss=0.05, n_frames=5, blocks=("vis",), peaks=(), loglike_null=-4.25)

    bk_fixed = voc.outlier_log_density([ns_5d, ns_3d, ns_empty], prior)
    assert np.all(np.isfinite(bk_fixed))
    assert abs(float(bk_fixed[2]) - (-4.25)) < 1e-12
    ref_b5 = ref.laplace_outlier_log_density_fixed_scale([pk_pos_mf.loglike], [cov5], [mc_lp5_pos], velocity=True)
    ref_b3 = ref.laplace_outlier_log_density_fixed_scale([pk_pos_mf.loglike], [cov5], [mc_lp3_pos], velocity=False)
    bk_fixed_mc_err = max(abs(float(bk_fixed[0]) - ref_b5), abs(float(bk_fixed[1]) - ref_b3))

    # 4. Default rescalable=True scale marginalization vs 1D adaptive quadrature over ln s:
    # interior mode hat = 1.2 exhibits Laplace + Stirling O((3 d^2 + 6 d + 2) / (12 n)) = 107 / (12 n) convergence
    # across n = 24, 96, 384, and boundary stationary point at s_max = 2.0 exhibits the half-Gaussian ln(2) shift.
    mc_lp5_sky = float(ref.mc_kepler_state_prior_log_density(
        state_in_mf, (5.0, 25.0), (1.0, 20.0), 0.85, prior.ln_f_floor, velocity=True, sigma_f=None, n_mc=8192, seed=11
    )[0]) + math.log(abs_ell)
    vg_lp5_sky = float(np.squeeze(prior.log_density(state_in_mf, velocity=True, sigma_f=None))) + math.log(abs_ell)
    sky_kde_diff = vg_lp5_sky - mc_lp5_sky

    laplace_offsets = []
    for n_val in (24.0, 96.0, 384.0):
        chi2_val = 1.44 * n_val  # hat = sqrt(chi2 / n) = 1.2 inside [0.1, 10.0]
        pk_in = vns.NightPeak(
            mean=state_in_mf[0],
            cov=cov5,
            loglike=75.0,
            n=np.array([n_val]),
            chi2=np.array([chi2_val]),
            hessian=np.linalg.inv(cov5)[None, :, :],
            rescalable=True,
        )
        ns_in = vns.NightSummary(t_ref=T0, t_ss=0.05, n_frames=5, blocks=("vis",), peaks=(pk_in,), loglike_null=-50.0)
        bk_vg = float(voc.outlier_log_density([ns_in], prior, s_max=10.0)[0])
        bk_quad = ref.quad_rescalable_outlier_log_density(
            [75.0], [cov5], [n_val], [chi2_val], [mc_lp5_sky], [-2.0], prior.ln_f_floor, velocity=True, s_max=10.0
        )
        # Subtract independent sky-KDE difference (vg_lp5 - mc_lp5) to isolate the 1D ln(s) Laplace error
        laplace_offsets.append((bk_quad - bk_vg) + sky_kde_diff)

    # Check positive-m_f rescalable=True at n = 384 directly against independent MC + quad
    pk_res_pos = vns.NightPeak(
        mean=state_pos_mf[0],
        cov=cov5,
        loglike=75.0,
        n=np.array([384.0]),
        chi2=np.array([1.44 * 384.0]),
        hessian=np.linalg.inv(cov5)[None, :, :],
        rescalable=True,
    )
    ns_res_pos = vns.NightSummary(t_ref=T0, t_ss=0.05, n_frames=5, blocks=("vis",), peaks=(pk_res_pos,), loglike_null=-50.0)
    bk_res_pos_vg = float(voc.outlier_log_density([ns_res_pos], prior, s_max=10.0)[0])
    bk_res_pos_quad = ref.quad_rescalable_outlier_log_density(
        [75.0], [cov5], [384.0], [1.44 * 384.0], [mc_lp5_sky], [+0.25], prior.ln_f_floor, velocity=True, s_max=10.0
    )
    rescalable_n384_err = abs(bk_res_pos_vg - bk_res_pos_quad)

    # Boundary-clipped stationary point: chi2 = (n - 5) * s_max^2 puts exact mode on u = ln(s_max), giving half-Gaussian ln(2)
    s_clip = 2.0
    n_clip = 384.0
    chi2_bdy = (n_clip - 5.0) * (s_clip**2)
    pk_bdy = vns.NightPeak(
        mean=state_in_mf[0],
        cov=cov5,
        loglike=75.0,
        n=np.array([n_clip]),
        chi2=np.array([chi2_bdy]),
        hessian=np.linalg.inv(cov5)[None, :, :],
        rescalable=True,
    )
    ns_bdy = vns.NightSummary(t_ref=T0, t_ss=0.05, n_frames=5, blocks=("vis",), peaks=(pk_bdy,), loglike_null=-500.0)
    bk_bdy_vg = float(voc.outlier_log_density([ns_bdy], prior, s_max=s_clip)[0])
    bk_bdy_quad = ref.quad_rescalable_outlier_log_density(
        [75.0], [cov5], [n_clip], [chi2_bdy], [mc_lp5_sky], [-2.0], prior.ln_f_floor, velocity=True, s_max=s_clip
    )
    bdy_ln2_err = abs(((bk_bdy_vg - bk_bdy_quad) - sky_kde_diff) - math.log(2.0))

    record("max_log_flux_box_err", max_log_flux_box_err)
    record("norm_3d_err", norm_3d_err)
    record("norm_5d_err", norm_5d_err)
    record("rescalable_n24_offset", laplace_offsets[0])
    record("rescalable_n384_err", rescalable_n384_err)
    assert max_log_flux_box_err < 1e-12
    assert norm_3d_err < 5e-3
    assert norm_5d_err < 1e-2
    assert kde_3d_err < 0.05
    assert kde_5d_err < 0.14
    assert bk_fixed_mc_err < 0.14
    # Asymptotic O((3 d^2 + 6 d + 2) / (12 n)) = 107 / (12 n) = 0.3715, 0.0929, 0.0232 nats (exact Gamma diff: 0.4062, 0.0948, 0.0233)
    for idx, n_val in enumerate((24.0, 96.0, 384.0)):
        exact_gamma_diff = (
            float(special.gammaln(0.5 * (n_val - 5.0)))
            - math.log(2.0)
            - 0.5 * (n_val - 5.0) * math.log(0.5 * n_val)
            + 0.5 * n_val
            - 0.5 * math.log(math.pi / n_val)
        )
        assert abs(laplace_offsets[idx] - exact_gamma_diff) < 1e-4
    assert abs(laplace_offsets[0] - 107.0 / (12.0 * 24.0)) < 0.04
    assert abs(laplace_offsets[1] - 107.0 / (12.0 * 96.0)) < 0.003
    assert abs(laplace_offsets[2] - 107.0 / (12.0 * 384.0)) < 0.0005
    assert rescalable_n384_err < 0.15
    assert bdy_ln2_err < 0.01
    assert abs((bk_bdy_vg - bk_bdy_quad) - math.log(2.0)) < 0.10


@pytest.mark.validates("virgil.orbit_cells.OutlierPrior", roots=["mathematics"], kind="control")
def test_outlier_prior_hard_flux_cutoff_control():
    """Naive point evaluation of the flux box at m_f > 0 gives -inf instead of the finite Gaussian-convolved box."""
    summaries = _make_synthetic_summaries(np.array([0.0, 5.0, 12.0]) + T0, (11.0, 1.0, 0.2, 40.0, 50.0, 110.0, 8.0))
    dom = voc.period_domain(summaries, p_range=(5.0, 20.0))
    prior = voc.outlier_prior(dom, a_range=(1.0, 20.0), e_max=0.8, flux_floor=1e-3, n=2048, seed=1)

    pt_eval = float(np.squeeze(prior.log_density(np.array([[4.0, -3.0, 0.2, -0.1, +0.30]]), velocity=True, sigma_f=None)))
    box_eval = float(np.squeeze(prior.log_density(np.array([[4.0, -3.0, 0.2, -0.1, +0.30]]), velocity=True, sigma_f=np.array([0.25]))))
    record("box_eval_at_positive_lnf", box_eval)
    assert pt_eval == -float("inf")
    assert math.isfinite(box_eval)


@pytest.mark.validates(
    "virgil.orbit_cells.search_plan",
    "virgil.orbit_cells.search_cell",
    "virgil.orbit_cells.position_leg",
    "virgil.orbit_cells.CellFit",
    roots=["mathematics"],
    headline="max_scale_period_rel_drift",
)
def test_scale_equivariant_conditional_mle_invariance():
    """Scaling quoted night errors across ln s in {-2.0, 0.0, +0.8} preserves mode recovery and inlier structure (#722)."""
    times = np.array([0.0, 2.3, 5.4, 9.8, 15.1, 24.0]) + T0
    orbit = (12.0, 1.4, 0.32, 46.0, 38.0, 115.0, 8.8)

    dom_ref = voc.PeriodDomain(p_cut=7.0, p_max=17.0, overridden=True)
    op = voc.outlier_prior(dom_ref, a_range=(2.0, 20.0), e_max=0.8, flux_floor=1e-3, n=4096, seed=2)

    recovered_periods = []
    inlier_counts = []
    physical_ln_scales = []
    for ln_s in (-2.0, 0.0, +0.8):
        sums = _make_synthetic_summaries(times, orbit, ln_scale=ln_s, outlier_idx=3)
        bk = voc.outlier_log_density(sums, op, s_max=20.0)
        cells = voc.make_cells(sums[0].t_ref, sums[-1].t_ref, dom_ref)
        target_cell = next(c for c in cells if c.n == 2)

        t_nights = np.array([s.t_ref for s in sums])
        fit = voc.position_leg(
            sums,
            bk,
            target_cell,
            jax.random.PRNGKey(11),
            t_anchor=sums[0].t_ref,
            t_nights=t_nights,
            a_range=(2.0, 20.0),
            e_max=0.8,
            flux_floor=1e-3,
            jitter=0.0,
            n_is=300,
            posterior_draws=False,
        )
        assert isinstance(fit, voc.CellFit)
        assert fit.failed is False
        assert len(fit.modes) >= 1

        # Exercise search_plan and search_cell directly on fit.space
        plan = voc.search_plan(fit.space, key=jax.random.PRNGKey(5))
        info = {}
        modes_direct = voc.search_cell(fit.space, key=jax.random.PRNGKey(5), plan=plan, info=info)
        assert len(modes_direct) >= 1
        assert info["evaluations"] > 0

        dec_mode = voc.decode_particles(dataclasses.replace(fit, draws=np.asarray(modes_direct[0].x)[None, :]))
        recovered_periods.append(float(dec_mode["period"][0]))
        inlier_counts.append(float(modes_direct[0].n_inliers))
        physical_ln_scales.append(float(np.squeeze(dec_mode["ln_s"])) + ln_s)

    max_scale_period_rel_drift = float(
        (max(recovered_periods) - min(recovered_periods)) / orbit[0]
    )
    inlier_drift = float(max(inlier_counts) - min(inlier_counts))
    ln_s_phys_drift = float(max(physical_ln_scales) - min(physical_ln_scales))
    record("max_scale_period_rel_drift", max_scale_period_rel_drift)
    record("inlier_drift", inlier_drift)
    record("ln_s_phys_drift", ln_s_phys_drift)
    assert max_scale_period_rel_drift < 1e-3
    assert inlier_drift < 0.05
    assert all(4.7 <= n_in <= 5.05 for n_in in inlier_counts)
    assert ln_s_phys_drift < 0.08
    assert all(abs(p - orbit[0]) < 0.08 for p in recovered_periods)


@pytest.mark.validates(
    "virgil.orbit_cells.fit_jitter",
    roots=["mathematics"],
    headline="jitter_err_mas",
)
def test_fit_jitter_type_ii_ml_recovers_extra_astrometric_scatter():
    """Type-II ML fit_jitter stays small on zero-scatter circular data and rises on injected scatter."""
    times = np.array([0.0, 2.0, 5.0, 9.0, 14.0, 19.0, 24.0]) + T0
    orbit = (12.0, 0.0, 0.0, 45.0, 0.0, 105.0, 9.0)
    dom = voc.PeriodDomain(p_cut=7.0, p_max=17.0, overridden=True)
    op = voc.outlier_prior(dom, a_range=(2.0, 20.0), e_max=0.8, flux_floor=1e-3, n=2048, seed=4)

    sums_clean = _make_synthetic_summaries(times, orbit, pos_sigma=0.05, extra_scatter_mas=0.0, rng_seed=21)
    bk_clean = voc.outlier_log_density(sums_clean, op)
    jit_clean = voc.fit_jitter(sums_clean, bk_clean, dom, (times[0], times[-1]), n_grid=17, n_best=3)

    injected = 0.35
    sums_noisy = _make_synthetic_summaries(times, orbit, pos_sigma=0.05, extra_scatter_mas=injected, rng_seed=21)
    bk_noisy = voc.outlier_log_density(sums_noisy, op)
    jit_noisy = voc.fit_jitter(sums_noisy, bk_noisy, dom, (times[0], times[-1]), n_grid=17, n_best=3)

    jitter_err_mas = abs(float(jit_noisy) - injected)
    record("jitter_err_mas", jitter_err_mas)
    assert jit_clean < 0.08
    assert jit_noisy > 2.5 * jit_clean
    assert 0.12 <= jit_noisy <= 0.65


@pytest.mark.validates(
    "virgil.orbit_cells.position_cells",
    "virgil.orbit_cells.CellsResult",
    "virgil.orbit_cells.decode_particles",
    roots=["mathematics"],
    headline="rel_period_err",
)
def test_position_cells_blind_eccentric_recovery_bounds_and_boundary_sharing():
    """position_cells recovers injected eccentric orbit blindly, verifies upper bounds (#686) and shares boundary modes (#643)."""
    times = np.array([0.0, 2.4, 5.8, 11.3, 24.0]) + T0
    orbit = (12.0, 1.5, 0.35, 50.0, 42.0, 125.0, 8.6)
    sums = _make_synthetic_summaries(times, orbit, pos_sigma=0.06, vel_sigma=0.25)
    dom = voc.PeriodDomain(p_cut=6.5, p_max=19.0, overridden=True)
    op = voc.outlier_prior(dom, a_range=(2.0, 20.0), e_max=0.85, flux_floor=1e-3, n=4096, seed=7)
    bk = voc.outlier_log_density(sums, op)

    res = voc.position_cells(
        sums,
        dom,
        jax.random.PRNGKey(9),
        log_outlier=bk,
        outlier=op,
        jitter=0.0,
        prune=False,
        delta=0.005,
        n_is=400,
        a_range=(2.0, 20.0),
        e_max=0.85,
        flux_floor=1e-3,
    )
    assert isinstance(res, voc.CellsResult)
    assert not np.any(res.failed)
    assert abs(float(np.sum(res.p_pos)) - 1.0) < 1e-12
    assert res.pruned_mass < 0.005
    assert res.omitted_bound < 0.005

    # Check #686 upper bound property: upper = log_mass + log_l_sup >= log_mass + log_z within IS Monte Carlo noise
    for cell, fit, up in zip(res.cells, res.fits, res.upper):
        assert abs(float(up) - (cell.log_mass + fit.log_l_sup)) < 1e-12
        assert float(up) >= cell.log_mass + fit.log_z - 4.0 * max(fit.log_z_error, 1e-3)

    win = int(np.argmax(res.p_pos))
    assert res.cells[win].n == 2
    assert float(res.p_pos[win]) > 0.95

    dec = voc.decode_particles(res.fits[win])
    med_p = float(np.median(dec["period"]))
    med_a = float(np.median(dec["a"]))
    med_e = float(np.median(dec["ecc"]))
    med_cosi = float(np.median(np.abs(dec["cos_inc"])))
    rel_period_err = abs(med_p - orbit[0]) / orbit[0]

    # Boundary-straddling orbit (#643): span between anchors is 24.0 d -> T / P = 2.5 => P_edge = 9.6 d
    orbit_edge = (9.6, 1.0, 0.20, 42.0, 30.0, 100.0, 8.0)
    sums_edge = _make_synthetic_summaries(times, orbit_edge, pos_sigma=0.08, vel_sigma=0.35)
    bk_edge = voc.outlier_log_density(sums_edge, op)
    res_edge = voc.position_cells(
        sums_edge,
        voc.PeriodDomain(p_cut=7.2, p_max=14.0, overridden=True),
        jax.random.PRNGKey(12),
        log_outlier=bk_edge,
        outlier=op,
        jitter=0.0,
        n_is=300,
        a_range=(2.0, 20.0),
        e_max=0.85,
        flux_floor=1e-3,
    )
    total_shared = sum(len(f.shared) for f in res_edge.fits)

    record("rel_period_err", rel_period_err)
    record("rel_a_err", abs(med_a - orbit[6]) / orbit[6])
    assert rel_period_err < 0.01
    assert abs(med_a - orbit[6]) / orbit[6] < 0.08
    assert abs(med_e - orbit[2]) < 0.08
    assert abs(med_cosi - math.cos(math.radians(orbit[3]))) < 0.08
    assert total_shared >= 1


@pytest.mark.validates("virgil.orbit_cells.position_cells", roots=["mathematics"], kind="control")
def test_position_cells_permuted_epochs_control():
    """Permuting sky states across nights destroys single-cell Keplerian evidence in position_cells."""
    times = np.array([0.0, 2.4, 5.8, 11.3, 24.0]) + T0
    orbit = (12.0, 1.5, 0.35, 50.0, 42.0, 125.0, 8.6)
    sums = list(_make_synthetic_summaries(times, orbit, pos_sigma=0.06, vel_sigma=0.25))
    # Swap peaks across incompatible epochs while keeping original timestamps
    perm = [3, 0, 4, 1, 2]
    scrambled = [
        vns.NightSummary(
            t_ref=sums[k].t_ref,
            t_ss=sums[k].t_ss,
            n_frames=sums[k].n_frames,
            blocks=sums[k].blocks,
            peaks=sums[perm[k]].peaks,
            loglike_null=0.0,
        )
        for k in range(len(sums))
    ]
    dom = voc.PeriodDomain(p_cut=6.5, p_max=19.0, overridden=True)
    op = voc.outlier_prior(dom, a_range=(2.0, 20.0), e_max=0.85, flux_floor=1e-3, n=2048, seed=8)
    bk_clean = voc.outlier_log_density(sums, op)
    res_clean = voc.position_cells(
        sums,
        dom,
        jax.random.PRNGKey(13),
        log_outlier=bk_clean,
        outlier=op,
        jitter=0.0,
        n_is=250,
        a_range=(2.0, 20.0),
        e_max=0.85,
        flux_floor=1e-3,
    )
    bk = voc.outlier_log_density(scrambled, op)
    res_bad = voc.position_cells(
        scrambled,
        dom,
        jax.random.PRNGKey(13),
        log_outlier=bk,
        outlier=op,
        jitter=0.0,
        n_is=250,
        a_range=(2.0, 20.0),
        e_max=0.85,
        flux_floor=1e-3,
    )
    clean_logz = float(max(f.log_z for f in res_clean.fits))
    max_scrambled_logz = float(max(f.log_z for f in res_bad.fits))
    clean_inliers = max(float(m.n_inliers) for f in res_clean.fits for m in f.modes)
    scrambled_inliers = max(float(m.n_inliers) for f in res_bad.fits for m in f.modes)
    logz_drop_nats = clean_logz - max_scrambled_logz
    record("logz_drop_nats", logz_drop_nats)
    assert logz_drop_nats > 35.0
    assert clean_inliers > 4.7
    assert scrambled_inliers < 3.5


def teardown_module():
    jax.clear_caches()
    gc.collect()
