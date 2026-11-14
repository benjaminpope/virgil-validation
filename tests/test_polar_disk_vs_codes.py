"""MultipoleDisk against other people's codes (stage V, V1 and V2): frank, vis-r, FRAP, PMOIRED, oimodeler.

Reference values come from those packages (installed from PyPI or GitHub; see
pyproject.toml, extra ``external``) through src/external_bridge/radial_codes.py,
never from virgil. Each external radial-profile package is also validated directly
against closed-form analytic Hankel transforms (``kind="reference"``).
"""

import contextlib
import io
import pathlib
import sys

import numpy as np
import pytest
from scipy import special

vm = pytest.importorskip("virgil.models")
if not hasattr(vm, "MultipoleDisk"):
    pytest.skip("could not import 'virgil.models.MultipoleDisk'", allow_module_level=True)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "polar_disk"))

from crosscheck import radial as R  # noqa: E402
from evidence.plugin import record  # noqa: E402

pytestmark = [pytest.mark.external, pytest.mark.x64]

INC, PA, WL = 50.0, 30.0, 1.65e-6


def _scene(name):
    from virgil import GaussianRing, MultipoleDisk, RadialPowerLaw

    if name == "gauss_ring":
        return MultipoleDisk(GaussianRing(10.0, 1.5), inc=INC, pa=PA, n_r=64), R.gaussian_ring(10.0, 1.5)
    return (MultipoleDisk(RadialPowerLaw(5.0, 14.0, -1.0, edge_sigma=0.3), inc=INC, pa=PA, n_r=96),
            R.erf_power(5.0, 14.0, -1.0, 0.3))


def _virgil(disk, u, v):
    return np.asarray(disk.model(u * WL, v * WL, WL)).real


def _closed_form_gaussian(sigma_mas, q):
    """Normalized Hankel transform of I(r) = exp(-r^2 / (2 sigma^2)) / (2 pi sigma^2) on [0, inf):
    V(q) = exp(-2 pi^2 sigma_rad^2 q^2)."""
    sigma_rad = sigma_mas * R.MAS
    return np.exp(-2.0 * (np.pi * sigma_rad * q) ** 2)


def _closed_form_airy(diam_mas, q):
    """Normalized Hankel transform of a uniform disk of diameter D: 2 J_1(pi D q) / (pi D q)."""
    x = np.pi * diam_mas * R.MAS * q
    return np.where(x == 0.0, 1.0, 2.0 * special.j1(np.where(x == 0.0, 1.0, x)) / np.where(x == 0.0, 1.0, x))


@pytest.mark.validates("virgil.models.MultipoleDisk", roots=["frank"], headline="max_abs_dV")
@pytest.mark.parametrize("name", ["gauss_ring", "erf_power"])
def test_matches_frank_discrete_hankel_transform(name):
    pytest.importorskip("frank")
    from external_bridge.radial_codes import Frank

    disk, profile = _scene(name)
    u, v = R.baselines(300, q_max=8e7, seed=3)
    qd, _, _ = R.deproject(u, v, INC, PA)
    f, _, r_hi, _ = profile
    fr = Frank(2.0 * r_hi, 300, qd)
    ref = fr.visibility(f)
    assert fr.q_max > qd.max()  # frank's band limit covers every baseline
    err = record("max_abs_dV", np.abs(_virgil(disk, u, v) - ref).max())
    assert err < 1e-8  # measured 1.3e-9 and 6.7e-12 (V1): virgil's own quadrature floor, not frank's


@pytest.mark.validates("virgil.models.MultipoleDisk", roots=["frap"], headline="max_abs_dV")
@pytest.mark.parametrize("name", ["gauss_ring", "erf_power"])
def test_matches_frap_hankel_matrix(name):
    pytest.importorskip("frap")
    from external_bridge.radial_codes import Frap

    disk, profile = _scene(name)
    u, v = R.baselines(300, q_max=8e7, seed=3)
    qd, _, _ = R.deproject(u, v, INC, PA)
    f, _, r_hi, _ = profile
    ref = Frap(2.0 * r_hi, 300, qd).visibility(f)
    err = record("max_abs_dV", np.abs(_virgil(disk, u, v) - ref).max())
    assert err < 1e-8  # measured 1.3e-9 and 6.7e-12 (V1): virgil's own quadrature floor, not frank's


@pytest.mark.validates("virgil.models.MultipoleDisk", roots=["mathematics"], headline="max_abs_dV_virgil_vs_truth")
def test_vis_r_difference_is_vis_rs_own_discretization_error():
    """vis-r's DHT interpolates linearly in q on a grid set by the field size, with an error ~5e-4 here.
    That is far above virgil's, so vis-r cannot referee virgil at the 1e-6 level and is not counted as a root:
    the claim checked is against the exact transform, and the virgil-vs-vis-r gap is reported as vis-r's error."""
    pytest.importorskip("vis_r")
    from external_bridge.radial_codes import VisR
    from vis_r import functions as vf
    from virgil import GaussianRing, MultipoleDisk

    s = 100.0  # ALMA scale: r0 = 1000 mas
    u, v = R.baselines(300, q_max=8e5, seed=3)
    disk = MultipoleDisk(GaussianRing(10 * s, 1.5 * s), inc=INC, pa=PA, n_r=64)
    truth = R.disk_visibility(R.gaussian_ring(10 * s, 1.5 * s), u, v, INC, PA)
    visr = VisR(u, v, INC, PA, sz=8.84)
    ref = visr.visibility(lambda r: vf.r_prof_gauss(r, [10 * s / 1000.0, 1.5 * s / 1000.0]))
    got = _virgil(disk, u, v)
    err_truth = record("max_abs_dV_virgil_vs_truth", np.abs(got - truth).max())
    visr_err = np.abs(ref - truth)
    record("vis_r_error_vs_truth", visr_err.max())
    assert err_truth < 1e-8
    # vis-r is not a referee at this setting: its own error (~5e-4) is five orders above virgil's, so any
    # agreement test against it is implied by the comparison with the exact transform above (triangle inequality).
    # What is checked of vis-r is that it is the discretization error that separates the two, not a bridge fault.
    record("max_abs_dV_virgil_vs_visr", np.abs(got - ref).max())
    assert 1e-9 < visr_err.max() < 2e-3
    assert np.abs((got - ref) - (truth - ref)).max() < 1e-8


@pytest.mark.validates("virgil.models.MultipoleDisk", "virgil.models.Harmonics", roots=["pmoired"],
                       headline="max_abs_dV2")
def test_matches_pmoired_ring_harmonics(tmp_path):
    pytest.importorskip("pmoired")
    import v2_external as ext

    out = ext.pmoired_section(tmp_path)
    worst = 0.0
    for case, row in out.items():
        record(f"max_abs_dV2_{case}", row["max_abs_dV2"])
        worst = max(worst, float(row["max_abs_dV2"]))
        assert row["passed"], (case, row)
    record("max_abs_dV2", worst)


@pytest.mark.validates("virgil.models.MultipoleDisk", "virgil.models.Harmonics", roots=["oimodeler"],
                       headline="max_abs_dV")
def test_matches_oimodeler_rings():
    pytest.importorskip("oimodeler")
    import v2_external as ext

    out = ext.oimodeler_section()
    assert out["O1_EIRing"]["passed"] and out["O2_Ring"]["passed"]
    assert all(r["passed"] for r in out["O3_ESKRing"].values())
    worst = max(
        out["O1_EIRing"]["max_abs_dV"],
        out["O2_Ring"]["max_abs_dV"],
        *(r["max_abs_dV"] for r in out["O3_ESKRing"].values()),
    )
    record("max_abs_dV", worst)


@pytest.mark.validates("virgil.models.ExpTail", roots=["oimodeler"], kind="upstream")
@pytest.mark.xfail(strict=True, raises=AssertionError, reason="P9: oimodeler's image-based ExpRing is 8e-3 away from the exact transform of "
                   "its own documented profile at every grid size; virgil agrees with the exact value to 1e-14")
def test_oimodeler_exp_ring_matches_the_exact_transform():
    pytest.importorskip("oimodeler")
    import v2_external as ext

    out = ext.oimodeler_section()
    assert out["O4_ExpRing"]["passed"]


# ------------------------- reference checks of the bridges & external codes against closed forms
@pytest.mark.validates("external_bridge.radial_codes", roots=["mathematics"], kind="reference", headline="max_abs_dV")
def test_bridges_reproduce_our_hankel_transform():
    """Frank and Frap bridges match both the closed-form 2D Gaussian Hankel transform and crosscheck.radial."""
    pytest.importorskip("frank")
    pytest.importorskip("frap")
    from external_bridge.radial_codes import Frank, Frap

    u, v = R.baselines(200, q_max=8e7, seed=4)
    qd, _, _ = R.deproject(u, v, INC, PA)
    sigma_mas = 3.0
    gauss = lambda r: np.exp(-0.5 * (r / sigma_mas) ** 2)
    closed = _closed_form_gaussian(sigma_mas, qd)
    f, _, r_hi, _ = R.gaussian_ring(10.0, 1.5)
    truth = R.hankel(R.gaussian_ring(10.0, 1.5), qd)
    worst = 0.0
    for bridge in (Frank(45.0, 300, qd), Frap(45.0, 300, qd)):
        err_g = float(np.abs(bridge.visibility(gauss) - closed).max())
        assert err_g < 1e-12
        worst = max(worst, err_g)
    for bridge in (Frank(2.0 * r_hi, 300, qd), Frap(2.0 * r_hi, 300, qd)):
        err_r = float(np.abs(bridge.visibility(f) - truth).max())
        assert err_r < 1e-10  # V1 measures 1e-15
        worst = max(worst, err_r)
    record("max_abs_dV", worst)


@pytest.mark.validates("external_bridge.radial_codes", roots=["mathematics"], kind="reference", headline="max_abs_dV")
def test_vis_r_bridge_is_within_its_discretization_of_the_truth():
    pytest.importorskip("vis_r")
    from external_bridge.radial_codes import VisR
    from vis_r import functions as vf

    u, v = R.baselines(200, q_max=8e5, seed=4)
    ref = VisR(u, v, INC, PA, sz=8.84).visibility(lambda r: vf.r_prof_gauss(r, [1.0, 0.15]))
    truth = R.disk_visibility(R.gaussian_ring(1000.0, 150.0), u, v, INC, PA)
    err = record("max_abs_dV", np.abs(ref - truth).max())
    assert 1e-9 < err < 2e-3  # it differs (linear interpolation), but only slightly


@pytest.mark.validates("frank", roots=["mathematics"], kind="reference", headline="max_abs_dV")
def test_frank_hankel_matches_closed_form_gaussian():
    """frank's DiscreteHankelTransform reproduces the closed-form 2D Gaussian transform exp(-2 pi^2 sigma^2 q^2)."""
    frank_h = pytest.importorskip("frank.hankel")

    u, v = R.baselines(200, q_max=6e7, seed=11)
    qd, _, _ = R.deproject(u, v, INC, PA)
    sigma_mas, r_max_mas = 3.2, 48.0
    dht = frank_h.DiscreteHankelTransform(r_max_mas * R.MAS, 300)
    assert dht.Qmax > qd.max()
    r_rad = dht.r
    h = dht.coefficients(q=qd)
    h0 = dht.coefficients(q=np.zeros(1))
    vals = np.exp(-0.5 * (r_rad / (sigma_mas * R.MAS)) ** 2)
    got = (h @ vals) / (h0 @ vals)[0]
    err = record("max_abs_dV", np.abs(got - _closed_form_gaussian(sigma_mas, qd)).max())
    assert err < 1e-12


@pytest.mark.validates("frap", roots=["mathematics"], kind="reference", headline="max_abs_dV")
def test_frap_hankel_matches_closed_form_gaussian():
    """FRAP's Fourier-Bessel Hankel matrix reproduces the closed-form 2D Gaussian transform exp(-2 pi^2 sigma^2 q^2)."""
    pytest.importorskip("frap")
    from external_bridge.radial_codes import Frap

    u, v = R.baselines(200, q_max=6e7, seed=12)
    qd, _, _ = R.deproject(u, v, INC, PA)
    sigma_mas = 3.2
    got = Frap(48.0, 300, qd).visibility(lambda r: np.exp(-0.5 * (r / sigma_mas) ** 2))
    err = record("max_abs_dV", np.abs(got - _closed_form_gaussian(sigma_mas, qd)).max())
    assert err < 1e-12


@pytest.mark.validates("oimodeler", roots=["mathematics"], kind="reference", headline="max_abs_dV")
def test_oimodeler_fourier_components_match_closed_forms():
    """oimodeler's analytic Fourier-space components (oimUD, oimGauss, oimEIRing) match closed-form Airy, Gaussian and J_0 visibilities."""
    oim = pytest.importorskip("oimodeler")

    u, v = R.baselines(200, q_max=6e7, seed=13)
    wl = 2.2e-6
    q = np.hypot(u, v)

    def eval_oim(comp):
        with contextlib.redirect_stdout(io.StringIO()):
            return oim.oimModel(comp).getComplexCoherentFlux(u, v, np.full_like(u, wl))

    # 1. Uniform disk: 2 J_1(pi D q) / (pi D q)
    d_mas = 9.0
    err_ud = float(np.abs(eval_oim(oim.oimUD(d=d_mas)) - _closed_form_airy(d_mas, q)).max())
    # 2. Circular Gaussian: fwhm = 2 sqrt(2 ln 2) sigma
    fwhm_mas = 5.5
    sigma_mas = fwhm_mas / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    err_g = float(np.abs(eval_oim(oim.oimGauss(fwhm=fwhm_mas)) - _closed_form_gaussian(sigma_mas, q)).max())
    # 3. Inclined thin elliptical ring: J_0(pi d q_deproj)
    elong = 1.6
    inc_deg = float(np.rad2deg(np.arccos(1.0 / elong)))
    qd, _, _ = R.deproject(u, v, inc_deg, PA)
    ring_truth = special.j0(np.pi * d_mas * R.MAS * qd)
    err_r = float(np.abs(eval_oim(oim.oimEIRing(d=d_mas, pa=PA, elong=elong)).real - ring_truth).max())
    err = record("max_abs_dV", max(err_ud, err_g, err_r))
    assert err < 1e-12


@pytest.mark.validates("visr", roots=["mathematics"], kind="reference", headline="max_abs_dV_nodes")
def test_vis_r_dht_matches_closed_form_gaussian():
    """vis-r's DHT matrix is exact on its native Bessel-zero frequencies Qzero and within linear-interpolation error after uv_trans."""
    pytest.importorskip("vis_r")
    from vis_r import functions as vf

    u, v = R.baselines(200, q_max=8e5, seed=14)
    sigma_arcsec = 0.35
    rnk, qzero, ykm = vf.setup_dht(8.84, u, v)
    f = vf.r_prof_gauss(rnk * (180.0 * 3600.0 / np.pi), [0.0, sigma_arcsec])
    fth = ykm @ f
    fth = fth / fth[0]
    err_nodes = record("max_abs_dV_nodes", np.abs(fth - _closed_form_gaussian(sigma_arcsec * 1e3, qzero)).max())
    assert err_nodes < 1e-12
    _, ruv = vf.uv_trans(u, v, np.deg2rad(PA), np.deg2rad(INC))
    qd, _, _ = R.deproject(u, v, INC, PA)
    interp_err = record("max_abs_dV_interp", np.abs(np.interp(ruv, qzero, fth) - _closed_form_gaussian(sigma_arcsec * 1e3, qd)).max())
    assert interp_err < 1e-3
    # Un-deprojected frequencies must disagree sharply on an inclined disk
    assert np.abs(np.interp(np.hypot(u, v), qzero, fth) - _closed_form_gaussian(sigma_arcsec * 1e3, qd)).max() > 0.1


# ---------------------------------------------------------------- what the other codes get wrong or cannot do
@pytest.mark.validates("visr", roots=["mathematics"], kind="upstream")
@pytest.mark.xfail(strict=True, raises=AssertionError, reason="P10: vis-r's r_prof_erf2_power_ggap indexes its parameters wrongly "
                   "(depth read as the radius and the radius as the depth); r_prof_erf2_power_ggap2 is right")
def test_vis_r_single_gap_profile_matches_its_two_gap_form():
    pytest.importorskip("vis_r")
    from vis_r import functions as vf

    r = np.linspace(0.01, 1.0, 2000)
    base = [0.2, 0.0, 0.05, 0.8, 0.05]
    one = vf.r_prof_erf2_power_ggap(r, base + [0.9, 0.5, 0.05])
    two = vf.r_prof_erf2_power_ggap2(r, base + [0.9, 0.5, 0.05, 0.0, 0.9, 0.05])
    assert np.abs(one - two).max() < 1e-9


@pytest.mark.validates("frank", "frap", roots=["mathematics"], kind="definition")
def test_discrete_hankel_codes_cannot_represent_a_discontinuous_profile():
    """D10: a sharp-edged exponential tail is not band-limited: frank's and FRAP's N = 300 transforms differ from
    the exact one by 5e-3 with Rmax = 18 000 mas (their accuracy statements assume a smooth profile); virgil's rule is not sampled at fixed nodes."""
    pytest.importorskip("frank")
    pytest.importorskip("frap")
    from external_bridge.radial_codes import Frank, Frap

    u, v = R.baselines(200, q_max=8e5, seed=4)
    qd, _, _ = R.deproject(u, v, INC, PA)
    prof = R.exp_tail(200.0, 300.0)
    f, _, r_hi, _ = prof
    truth = R.hankel(prof, qd)
    r_max = 18000.0  # mas, as V1: 60 scale lengths (the tail is gone) and frank's band limit covers every baseline
    for bridge in (Frank(r_max, 300, qd), Frap(r_max, 300, qd)):
        if hasattr(bridge, "q_max"):
            assert bridge.q_max > qd.max(), "frank evaluated beyond its band limit: aliasing, not the discontinuity"
        err = np.abs(bridge.visibility(f) - truth).max()
        assert 1e-3 < err < 1e-1
