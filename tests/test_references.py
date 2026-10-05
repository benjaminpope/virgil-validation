"""Our own reference code, checked before it judges virgil.

Every check of virgil that goes through one of these modules counts only
if the module is itself verified (scripts/trust.py, ``via``). Each is
checked here against something that is not our code: a closed form derived
by a different route, a statistical law, or a published standard.

* crosscheck.array: uv from Thompson, Moran & Swenson's matrices against the
  baseline projected onto the sky-plane East and North directions, which
  are built from the spherical-trigonometry source direction (a different
  route to the same geometry).
* crosscheck.chi2: closure phases made by hand from independent baseline
  phases are correlated as (σ²/3) T Tᵀ, so the correlated chi-squared must
  follow chi-squared with rank-many degrees of freedom.
* crosscheck.simulate: the noise it draws (V² Gaussian, closure phases
  formed from per-baseline phases) has the stated distributions and
  correlations, and the file holds the uv and errors it was given.
* crosscheck.orbits: Kepler's equation is solved, the radial velocity is
  the rate of change of depth, and positions equal the
  Thiele-Innes form of the textbook visual binary, which uses the
  eccentric anomaly and never the true anomaly.
* crosscheck.disks: the Henyey-Greenstein phase function is normalized with
  mean cosine g; a flat disk's brightness peaks where its documented
  geometry puts the ring.
* crosscheck.oifits_writer: the file has the columns, keywords and units
  that the OIFITS v2 standard (Duvert et al. 2017, Table 1-7) requires.
"""

import numpy as np
import pytest
from astropy.io import fits
from scipy import integrate, stats

from crosscheck import array, chi2, disks, oifits_writer, orbits, simulate
from evidence.plugin import record

UTS4 = np.array([[-9.925, -20.335, 0.0], [14.887, 30.502, 0.0], [44.915, 66.183, 0.0], [103.306, 43.999, 0.0]])


@pytest.mark.parametrize("lat,dec", [(-24.6, -50.0), (-24.6, 10.0), (34.2, 60.0), (0.0, 0.0)])
@pytest.mark.validates("crosscheck.array", roots=["mathematics"], kind="reference")
def test_uv_against_the_sky_plane_projection(lat, dec):
    phi, d = np.deg2rad(lat), np.deg2rad(dec)
    worst = 0.0
    for ha in np.linspace(-4, 4, 9):
        h = np.deg2rad(15 * ha)
        # East, North, Up components of the unit vectors on the sky at the source
        d_dh = np.array([-np.cos(d) * np.cos(h), np.sin(phi) * np.cos(d) * np.sin(h), -np.cos(phi) * np.cos(d) * np.sin(h)])
        east = -d_dh / np.cos(d)  # increasing right ascension is decreasing hour angle
        north = np.array([np.sin(d) * np.sin(h), np.cos(phi) * np.cos(d) + np.sin(phi) * np.sin(d) * np.cos(h),
                          np.sin(phi) * np.cos(d) - np.cos(phi) * np.sin(d) * np.cos(h)])
        u, v = array.snapshot_uv(UTS4, ha, dec, lat)
        b = np.array([UTS4[j] - UTS4[i] for i, j in array.baselines(4)])
        worst = max(worst, np.max(np.abs(u - b @ east)), np.max(np.abs(v - b @ north)))
    record("max_abs_duv_m", worst)
    assert worst < 1e-9


def _baseline_closure_data(rng, n_snap, sigma_v2, sigma_cp, scale=None):
    """A loaded-file dict for a point source seen by four telescopes, its
    closure phases made here from independent per-baseline phases (N(0,
    σ²/3) each), so the four triangles share baselines."""
    pairs = array.baselines(4)
    tris = array.triangles(4)
    rows_v, rows_t = n_snap * len(pairs), n_snap * len(tris)
    d = {"u": np.ones((rows_v, 1)), "v": np.ones((rows_v, 1)), "wl": np.full((rows_v, 1), 2e-6),
         "v2": 1 + sigma_v2 * rng.normal(size=(rows_v, 1)), "dv2": np.full((rows_v, 1), sigma_v2)}
    cp, mjd, sta = [], [], []
    for k in range(n_snap):
        phase = dict(zip(pairs, rng.normal(0.0, sigma_cp / np.sqrt(3.0), len(pairs))))
        for a, b, c in tris:
            cp.append(phase[(a, b)] + phase[(b, c)] - phase[(a, c)])
            mjd.append(60000.0 + k)
            sta.append((a + 1, b + 1, c + 1))
    d.update({"u1": np.ones((rows_t, 1)), "v1": np.ones((rows_t, 1)), "u2": np.ones((rows_t, 1)),
              "v2_": np.ones((rows_t, 1)), "wl3": np.full((rows_t, 1), 2e-6),
              "cp": np.array(cp)[:, None], "dcp": np.full((rows_t, 1), sigma_cp),
              "t3_mjd": np.array(mjd), "t3_sta": np.array(sta)})
    return d


@pytest.mark.validates("crosscheck.chi2", roots=["statistics"], kind="reference")
def test_correlated_chi2_follows_chi2_of_the_rank():
    rng = np.random.default_rng(1)
    point = lambda u, v, w: np.ones(np.broadcast(u, v, w).shape, complex)  # noqa: E731
    n_snap, totals = 5, []
    for _ in range(400):
        d = _baseline_closure_data(rng, n_snap, 0.01, 0.05)
        totals.append(chi2.chi2(d, point, correlated=True, chord=False))
    dof = n_snap * len(array.baselines(4)) + n_snap * 3  # V² plus 3 independent closures of 4 telescopes
    p = stats.kstest(totals, stats.chi2(dof).cdf).pvalue
    record("mean_over_dof", float(np.mean(totals) / dof))
    record("ks_p", float(p))
    assert abs(np.mean(totals) / dof - 1) < 4 * np.sqrt(2 / dof) / np.sqrt(len(totals))
    assert p > 1e-3
    # the uncorrelated form on the same data is not chi-squared of 30: a control
    wrong = [chi2.chi2(_baseline_closure_data(rng, n_snap, 0.01, 0.05), point) for _ in range(200)]
    assert abs(np.mean(wrong) / dof - 1) > 0.05


@pytest.mark.validates("crosscheck.simulate", roots=["statistics", "standards"], kind="reference")
def test_simulated_noise_uv_and_errors(tmp_path):
    vis = lambda u, v, w: np.ones(np.broadcast(u, v, w).shape, complex)  # noqa: E731
    sv2, scp = 0.02, 2.0
    kw = dict(hour_angles_h=[-1.0, 1.0], wavelengths=[1.6e-6, 2.2e-6], dec_deg=-40.0, sigma_v2=sv2, sigma_cp_deg=scp)
    pulls_v2, cps = [], []
    for k in range(150):
        path = tmp_path / f"s{k}.fits"
        simulate.observe(path, vis, UTS4, rng=np.random.default_rng(k), **kw)
        with fits.open(path) as h:
            pulls_v2.append((h["OI_VIS2"].data["VIS2DATA"] - 1.0) / sv2)
            cps.append(h["OI_T3"].data["T3PHI"])
            if k == 0:
                np.testing.assert_array_equal(h["OI_VIS2"].data["VIS2ERR"], sv2)
                np.testing.assert_array_equal(h["OI_T3"].data["T3PHIERR"], scp)
                u, v = array.snapshot_uv(UTS4, -1.0, -40.0, -24.6276)
                np.testing.assert_allclose(h["OI_VIS2"].data["UCOORD"][:6], u, atol=1e-9)
                np.testing.assert_allclose(h["OI_VIS2"].data["VCOORD"][:6], v, atol=1e-9)
    pulls_v2 = np.ravel(pulls_v2)
    assert stats.kstest(pulls_v2, "norm").pvalue > 1e-3
    # closure phases of one snapshot and channel: four triangles from six baseline phases
    x = np.array(cps)[:, :4, 0]  # first snapshot, first channel: (draws, 4 triangles), degrees
    corr = np.corrcoef(x.T)
    T = chi2.triangle_matrix([(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)])
    want = (T @ T.T) / 3.0
    record("max_abs_dcorr", float(np.max(np.abs(corr - want))))
    assert np.allclose(np.diag(np.cov(x.T)), scp**2, rtol=0.35)
    assert np.max(np.abs(corr - want)) < 4 / np.sqrt(len(x))
    # whitened by our own covariance scp^2 T T^T / 3 (rank 3: four triangles of four
    # telescopes close on three independent phases), each snapshot and channel is chi-squared of 3
    x = np.array(cps).reshape(len(cps), -1, 4, 2).transpose(0, 1, 3, 2).reshape(-1, 4)
    q = np.einsum("ni,ij,nj->n", x, np.linalg.pinv(scp**2 * want), x)
    record("mean_chi2_per_dof", float(q.mean() / 3))
    assert stats.kstest(q, stats.chi2(3).cdf).pvalue > 1e-3
    assert stats.kstest(q, stats.chi2(4).cdf).pvalue < 1e-3  # four independent phases would not fit: a control


@pytest.mark.validates("crosscheck.orbits", roots=["mathematics"], kind="reference")
def test_kepler_and_thiele_innes():
    mean = np.linspace(-np.pi, np.pi, 41)
    for e in (0.0, 0.3, 0.7, 0.95):
        E = orbits.eccentric_anomaly(mean, e)
        assert np.max(np.abs(E - e * np.sin(E) - mean)) < 1e-12
    worst = 0.0
    for e, i, w, W in [(0.3, 60.0, 40.0, 110.0), (0.8, 130.0, 250.0, 20.0), (0.0, 0.0, 0.0, 300.0), (0.5, 90.0, 10.0, 75.0)]:
        E = orbits.eccentric_anomaly(mean, e)
        X, Y = np.cos(E) - e, np.sqrt(1 - e**2) * np.sin(E)
        iw, ww, WW = np.deg2rad([i, w, W])
        # Thiele-Innes constants (a = 1): North = A X + F Y, East = B X + G Y
        A = np.cos(ww) * np.cos(WW) - np.sin(ww) * np.sin(WW) * np.cos(iw)
        B = np.cos(ww) * np.sin(WW) + np.sin(ww) * np.cos(WW) * np.cos(iw)
        F = -np.sin(ww) * np.cos(WW) - np.cos(ww) * np.sin(WW) * np.cos(iw)
        G = -np.sin(ww) * np.sin(WW) + np.cos(ww) * np.cos(WW) * np.cos(iw)
        dra, ddec = orbits.sky_position(orbits.true_anomaly(mean, e), e, i, w, W)
        worst = max(worst, np.max(np.abs(ddec - (A * X + F * Y))), np.max(np.abs(dra - (B * X + G * Y))))
    record("max_abs_dpos", worst)
    assert worst < 1e-12
    # face-on circular orbit: prograde means position angle increases (North through East)
    dra, ddec = orbits.sky_position(np.array([0.0, 0.1]), 0.0, 0.0, 0.0, 0.0)
    assert np.arctan2(dra[1], ddec[1]) > np.arctan2(dra[0], ddec[0])


@pytest.mark.validates("crosscheck.orbits", roots=["mathematics"], kind="reference")
def test_radial_velocity_is_the_rate_of_recession():
    """The closed-form RV is the time derivative of z (by central differences
    of our own positions), and the companion recedes through the ascending node."""
    worst = 0.0
    for period, t_peri, e, i, w in [(800.0, 120.0, 0.3, 60.0, 40.0), (40.0, 3.0, 0.75, 130.0, 300.0), (90.0, 0.0, 0.0, 35.0, 0.0)]:
        t, h = np.linspace(0.0, 2 * period, 97), 1e-5 * period
        mean = [np.mod(2 * np.pi * (tt - t_peri) / period + np.pi, 2 * np.pi) - np.pi for tt in (t - h, t + h)]
        z = [orbits.depth(orbits.true_anomaly(m, e), e, i, w, 3.0) for m in mean]
        numeric = (z[1] - z[0]) / (2 * h)
        rv = orbits.radial_velocity(t, period, t_peri, e, i, w, 3.0)
        worst = max(worst, np.max(np.abs(rv - numeric)) / np.max(np.abs(rv)))
    record("max_rel_drv", worst)
    assert worst < 1e-5
    # circular orbit, omega = 0: at periastron (u = 0, the ascending node) z = 0 and rising
    assert orbits.radial_velocity(np.array([0.0]), 90.0, 0.0, 0.0, 35.0, 0.0, 1.0)[0] > 0
    assert orbits.depth(np.array([0.1]), 0.0, 35.0, 0.0)[0] > 0


@pytest.mark.validates("crosscheck.disks", roots=["mathematics"], kind="reference")
def test_disk_phase_function_and_ring_geometry():
    for g in (-0.5, 0.0, 0.3, 0.8):
        norm = integrate.quad(lambda t: disks.phase_hg(t, g) * 2 * np.pi * np.sin(t), 0, np.pi, epsabs=1e-13)[0]
        mean_cos = integrate.quad(lambda t: disks.phase_hg(t, g) * 2 * np.pi * np.sin(t) * np.cos(t), 0, np.pi, epsabs=1e-13)[0]
        assert abs(norm - 1) < 1e-10 and abs(mean_cos - g) < 1e-10
    flat = dict(radius=10.0, fwhm=2.0, phase=lambda t: np.ones_like(t), aspect=0.0)
    for inc, pa in [(0.0, 0.0), (60.0, 30.0), (75.0, 200.0)]:
        p, i = np.deg2rad(pa), np.deg2rad(inc)
        s = np.linspace(0, 20, 4001)
        along = disks.brightness(s * np.sin(p), s * np.cos(p), inc=inc, pa=pa, **flat)  # the major axis, at PA
        across = disks.brightness(s * np.cos(p), -s * np.sin(p), inc=inc, pa=pa, **flat)  # the minor axis
        assert abs(s[np.argmax(along)] - 10.0) < 0.01  # the ring at its radius along the major axis
        assert abs(s[np.argmax(across)] - 10.0 * np.cos(i)) < 0.01  # foreshortened by cos(inc) across it


REQUIRED = {  # OIFITS v2 (Duvert et al. 2017): table -> (columns, keywords)
    "OI_TARGET": ({"TARGET_ID", "TARGET", "RAEP0", "DECEP0", "EQUINOX"}, {"OI_REVN"}),
    "OI_ARRAY": ({"TEL_NAME", "STA_NAME", "STA_INDEX", "DIAMETER", "STAXYZ"}, {"OI_REVN", "ARRNAME", "FRAME"}),
    "OI_WAVELENGTH": ({"EFF_WAVE", "EFF_BAND"}, {"OI_REVN", "INSNAME"}),
    "OI_VIS2": ({"TARGET_ID", "TIME", "MJD", "INT_TIME", "VIS2DATA", "VIS2ERR", "UCOORD", "VCOORD", "STA_INDEX", "FLAG"},
                {"OI_REVN", "DATE-OBS", "ARRNAME", "INSNAME"}),
    "OI_T3": ({"TARGET_ID", "TIME", "MJD", "INT_TIME", "T3AMP", "T3AMPERR", "T3PHI", "T3PHIERR", "U1COORD", "V1COORD",
               "U2COORD", "V2COORD", "STA_INDEX", "FLAG"}, {"OI_REVN", "DATE-OBS", "ARRNAME", "INSNAME"}),
}
UNITS = {("OI_WAVELENGTH", "EFF_WAVE"): "m", ("OI_VIS2", "UCOORD"): "m", ("OI_VIS2", "VCOORD"): "m",
         ("OI_T3", "T3PHI"): "deg", ("OI_T3", "T3PHIERR"): "deg", ("OI_T3", "U1COORD"): "m"}


@pytest.mark.validates("crosscheck.oifits_writer", roots=["standards"], kind="reference")
def test_oifits_writer_meets_the_v2_standard(tmp_path):
    path = tmp_path / "s.fits"
    simulate.observe(path, lambda u, v, w: np.ones(np.broadcast(u, v, w).shape, complex), UTS4,
                     hour_angles_h=[0.0], wavelengths=[2.2e-6], sigma_v2=0.01, sigma_cp_deg=1.0)
    missing = []
    with fits.open(path) as h:
        assert h[0].header.get("CONTENT") == "OIFITS2"
        for table, (cols, keys) in REQUIRED.items():
            names = set(h[table].columns.names)
            missing += [f"{table}.{c}" for c in cols - names]
            missing += [f"{table}:{k}" for k in keys if k not in h[table].header]
            if "OI_REVN" in h[table].header:
                assert int(h[table].header["OI_REVN"]) == 2, table
        for (table, col), unit in UNITS.items():
            assert (h[table].columns[col].unit or "").strip() == unit, (table, col)
    assert not missing, missing
