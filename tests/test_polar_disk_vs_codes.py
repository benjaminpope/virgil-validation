"""PolarDisk against other people's codes (stage V, V1 and V2): frank, vis-r, FRAP, PMOIRED, oimodeler.

Reference values come from those packages (installed from PyPI or GitHub; see
pyproject.toml, extra ``external``) through src/external_bridge/radial_codes.py,
never from virgil. The benchmark scripts in benchmarks/polar_disk run the
same comparisons at full size and record the numbers and timings.
"""

import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("virgil.models")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "polar_disk"))

from crosscheck import radial as R  # noqa: E402
from evidence.plugin import record  # noqa: E402

vm = pytest.importorskip("virgil.models")
pytestmark = [pytest.mark.external, pytest.mark.x64,
              pytest.mark.skipif(not hasattr(vm, "PolarDisk"), reason="needs a virgil with PolarDisk (P1-P4)")]

INC, PA, WL = 50.0, 30.0, 1.65e-6


def _scene(name):
    from virgil import GaussianRing, PolarDisk, RadialPowerLaw

    if name == "gauss_ring":
        return PolarDisk(GaussianRing(10.0, 1.5), inc=INC, pa=PA, n_r=64), R.gaussian_ring(10.0, 1.5)
    return (PolarDisk(RadialPowerLaw(5.0, 14.0, -1.0, edge=0.3), inc=INC, pa=PA, n_r=96),
            R.erf_power(5.0, 14.0, -1.0, 0.3))


def _virgil(disk, u, v):
    return np.asarray(disk.model(u * WL, v * WL, WL)).real


@pytest.mark.validates("virgil.models.PolarDisk", roots=["frank"])
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
    assert err < 1e-5


@pytest.mark.validates("virgil.models.PolarDisk", roots=["frap"])
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
    assert err < 1e-5


@pytest.mark.validates("virgil.models.PolarDisk", roots=["visr"])
def test_matches_vis_r_up_to_its_own_discretization():
    """vis-r's DHT interpolates linearly in q on a grid set by the field size; its error is its own.
    At an ALMA-like scale (r0 = 1 arcsec) virgil must differ from vis-r by no more than vis-r differs from truth."""
    pytest.importorskip("vis_r")
    from external_bridge.radial_codes import Frank, VisR  # noqa: F401
    from vis_r import functions as vf
    from virgil import GaussianRing, PolarDisk

    s = 100.0  # ALMA scale: r0 = 1000 mas
    u, v = R.baselines(300, q_max=8e5, seed=3)
    disk = PolarDisk(GaussianRing(10 * s, 1.5 * s), inc=INC, pa=PA, n_r=64)
    truth = R.disk_visibility(R.gaussian_ring(10 * s, 1.5 * s), u, v, INC, PA)
    visr = VisR(u, v, INC, PA, sz=8.84)
    ref = visr.visibility(lambda r: vf.r_prof_gauss(r, [10 * s / 1000.0, 1.5 * s / 1000.0]))
    got = _virgil(disk, u, v)
    visr_err = np.abs(ref - truth)
    record("vis_r_error_vs_truth", visr_err.max())
    assert np.abs(got - truth).max() < 1e-6
    assert np.all(np.abs(got - ref) <= 1.1 * visr_err + 1e-6)


@pytest.mark.validates("virgil.models.PolarDisk", "virgil.models.Harmonics", roots=["pmoired"])
def test_matches_pmoired_ring_harmonics(tmp_path):
    pytest.importorskip("pmoired")
    import v2_external as ext

    out = ext.pmoired_section(tmp_path)
    for case, row in out.items():
        record(f"max_abs_dV2_{case}", row["max_abs_dV2"])
        assert row["passed"], (case, row)


@pytest.mark.validates("virgil.models.PolarDisk", "virgil.models.Harmonics", roots=["oimodeler"])
def test_matches_oimodeler_rings():
    pytest.importorskip("oimodeler")
    import v2_external as ext

    out = ext.oimodeler_section()
    assert out["O1_EIRing"]["passed"] and out["O2_Ring"]["passed"]
    assert all(r["passed"] for r in out["O3_ESKRing"].values())


@pytest.mark.validates("virgil.models.ExpTail", roots=["oimodeler"], kind="upstream")
@pytest.mark.xfail(strict=True, reason="P9: oimodeler's image-based ExpRing is 8e-3 away from the exact transform of "
                   "its own documented profile at every grid size; virgil agrees with the exact value to 1e-14")
def test_oimodeler_exp_ring_matches_the_exact_transform():
    pytest.importorskip("oimodeler")
    import v2_external as ext

    out = ext.oimodeler_section()
    assert out["O4_ExpRing"]["passed"]


# ------------------------------------------- the bridges to the codes are themselves checked
@pytest.mark.validates("external_bridge.radial_codes", roots=["mathematics"], kind="reference")
def test_bridges_reproduce_our_hankel_transform():
    pytest.importorskip("frank")
    pytest.importorskip("frap")
    from external_bridge.radial_codes import Frank, Frap

    u, v = R.baselines(200, q_max=8e7, seed=4)
    qd, _, _ = R.deproject(u, v, INC, PA)
    f, _, r_hi, _ = R.gaussian_ring(10.0, 1.5)
    truth = R.hankel(R.gaussian_ring(10.0, 1.5), qd)
    for bridge in (Frank(2.0 * r_hi, 300, qd), Frap(2.0 * r_hi, 300, qd)):
        assert np.abs(bridge.visibility(f) - truth).max() < 1e-6


@pytest.mark.validates("external_bridge.radial_codes", roots=["mathematics"], kind="reference")
def test_vis_r_bridge_is_within_its_discretization_of_the_truth():
    pytest.importorskip("vis_r")
    from external_bridge.radial_codes import VisR
    from vis_r import functions as vf

    u, v = R.baselines(200, q_max=8e5, seed=4)
    ref = VisR(u, v, INC, PA, sz=8.84).visibility(lambda r: vf.r_prof_gauss(r, [1.0, 0.15]))
    truth = R.disk_visibility(R.gaussian_ring(1000.0, 150.0), u, v, INC, PA)
    assert 1e-9 < np.abs(ref - truth).max() < 2e-3  # it differs (linear interpolation), but only slightly


# ---------------------------------------------------------------- what the other codes get wrong or cannot do
@pytest.mark.validates("visr", roots=["mathematics"], kind="upstream")
@pytest.mark.xfail(strict=True, reason="P10: vis-r's r_prof_erf2_power_ggap indexes its parameters wrongly "
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
    the exact one by 5e-3 (their accuracy statements assume a smooth profile); virgil's rule is not sampled at fixed nodes."""
    pytest.importorskip("frank")
    pytest.importorskip("frap")
    from external_bridge.radial_codes import Frank, Frap

    u, v = R.baselines(200, q_max=8e5, seed=4)
    qd, _, _ = R.deproject(u, v, INC, PA)
    prof = R.exp_tail(200.0, 300.0)
    f, _, r_hi, _ = prof
    truth = R.hankel(prof, qd)
    for bridge in (Frank(2.0 * r_hi, 300, qd), Frap(2.0 * r_hi, 300, qd)):
        err = np.abs(bridge.visibility(f) - truth).max()
        assert 1e-3 < err < 1e-1
