"""MultipoleDisk against brute-force Fourier transforms (stage V, V2) and our own Hankel transform (V1).

Everything here is written in this repository: a finely rendered image of the
scene (src/crosscheck/polar_images.py) Fourier-transformed point by point
(crosscheck.sky), and a composite Gauss-Legendre Hankel transform
(crosscheck.radial). The benchmark scripts in benchmarks/polar_disk run the same
cases at full size and record the numbers; these tests use fewer baselines.
"""

import pathlib
import sys

import numpy as np
import pytest
from scipy import special

vm = pytest.importorskip("virgil.models")
if not hasattr(vm, "MultipoleDisk"):
    pytest.skip("could not import 'virgil.models.MultipoleDisk'", allow_module_level=True)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "polar_disk"))

from crosscheck import polar_images as P  # noqa: E402
from crosscheck import radial as R  # noqa: E402
from crosscheck import sky  # noqa: E402
from evidence.plugin import record  # noqa: E402

pytestmark = pytest.mark.x64


def _bf():
    import v2_brute_force as bf

    return bf


@pytest.mark.slow
@pytest.mark.validates("virgil.models.MultipoleDisk", roots=["mathematics"], headline="max_abs_dV", tier="B")
@pytest.mark.parametrize("name", ["gauss_ring", "erf_power", "gap_on_plateau"])
def test_axisymmetric_profiles_match_our_hankel_transform(name):
    from virgil import Gap, GaussianRing, MultipoleDisk, RadialPowerLaw

    inc, pa = 50.0, 30.0
    u, v = R.baselines(300, q_max=8e7, seed=3)
    wl = 1.65e-6
    cases = {
        "gauss_ring": (MultipoleDisk(GaussianRing(10.0, 1.5), inc=inc, pa=pa, n_r=64), R.gaussian_ring(10.0, 1.5)),
        "erf_power": (MultipoleDisk(RadialPowerLaw(5.0, 14.0, -1.0, edge_sigma=0.3), inc=inc, pa=pa, n_r=96),
                      R.erf_power(5.0, 14.0, -1.0, 0.3)),
        "gap_on_plateau": (MultipoleDisk(Gap(RadialPowerLaw(4.0, 20.0, 0.0, edge_sigma=0.3), radius=12.0, sigma=0.8, depth=0.9),
                                         inc=inc, pa=pa, n_r=128),
                           R.with_gap(R.erf_power(4.0, 20.0, 0.0, 0.3), 12.0, 0.8, 0.9)),
    }
    disk, profile = cases[name]
    truth = R.disk_visibility(profile, u, v, inc, pa)
    got = np.asarray(disk.model(u * wl, v * wl, wl))
    err = record("max_abs_dV", np.abs(got - truth).max())
    # V1 records 1.3e-9 (Gaussian ring, a floor independent of n_r), 6.7e-12 and 2.8e-12: 1e-8 leaves headroom
    # but would catch a wrong node weight in one panel (about 1e-7)
    assert err < 1e-8


@pytest.mark.validates("virgil.models.ExpTail", roots=["mathematics"], kind="check", headline="max_abs_dV_n150")
def test_exp_tail_reaches_documented_accuracy_at_the_documented_node_count():
    """F19 (fixed in virgil 1eae3e4 by correcting the docstring, not the numerics): ExpTail used to promise
    1e-5 at about 16 * 2 pi q L nodes; the docstring now states what we measured, about 2e-4 at 128
    nodes and 4e-5 at the 150-node cap in the resolved-tail case. Check those stated numbers."""
    from virgil import ExpTail, MultipoleDisk

    u, v = R.baselines(300, q_max=8e5, seed=3)
    s, wl = 100.0, 1.65e-6
    truth = R.disk_visibility(R.exp_tail(2 * s, 3 * s), u, v, 50.0, 30.0)
    err = {}
    for n_r in (128, 150):
        disk = MultipoleDisk(ExpTail(2 * s, 3 * s), inc=50.0, pa=30.0, n_r=n_r)
        err[n_r] = record(f"max_abs_dV_n{n_r}", np.abs(np.asarray(disk.model(u * wl, v * wl, wl)) - truth).max())
    assert err[128] < 3e-4 and err[150] < 6e-5  # documented 2e-4 and 4e-5, with margin


@pytest.mark.validates("virgil.models.ExpTail", roots=["mathematics"], kind="regression")
def test_exp_tail_error_is_the_pinned_size():
    from virgil import ExpTail, MultipoleDisk

    u, v = R.baselines(300, q_max=8e5, seed=3)
    s, wl = 100.0, 1.65e-6
    disk = MultipoleDisk(ExpTail(2 * s, 3 * s), inc=50.0, pa=30.0, n_r=128)
    truth = R.disk_visibility(R.exp_tail(2 * s, 3 * s), u, v, 50.0, 30.0)
    assert np.abs(np.asarray(disk.model(u * wl, v * wl, wl)) - truth).max() < 1e-3


CASES = {  # case name -> attribute of the benchmark module (resolved lazily, see _bf)
    "linear_M3_thin_ring": "case_linear",
    "positive_M3_gaussian_ring": "case_positive",
    "positive_M1_thin_ring": "case_positive_m1",
    "vonmises_arc_blurred": "case_vonmises",
    "cone_height": "case_cone",
    "flared_with_harmonics": "case_flared_harmonics",
}


@pytest.mark.validates("virgil.models.MultipoleDisk", "virgil.models.Harmonics", "virgil.models.VonMises",
                       "virgil.models.Cone", "virgil.models.PowerLawHeight", roots=["mathematics"],
                       headline="max_abs_dV")
@pytest.mark.parametrize("name", sorted(CASES))
def test_brute_force_render(name):
    bf = _bf()
    u, v = bf.baselines(60)
    row = bf.run_case(name, getattr(bf, CASES[name]), u, v)
    record("max_abs_dV", row["max_abs_dV"])
    assert row["max_abs_dV"] <= bf.TOL
    # controls: the reversed azimuth / height sense must be visibly different
    assert all(c > 100 * bf.TOL for c in row["controls"].values())
    assert row["max_abs_Im_V"] > 0.05  # sensitive to the asymmetry (a non-zero imaginary part)


@pytest.mark.slow
@pytest.mark.validates("virgil.models.MultipoleDisk", "virgil.models.RadialLinear", roots=["mathematics"],
                       headline="max_abs_dV", tier="B")
def test_chromatic_system_weight():
    bf = _bf()
    u, v = bf.baselines(60)
    row = bf.case_chromatic(u, v)
    record("max_abs_dV", row["max_abs_dV"])
    assert row["passed"]


# ----------------------------------------------- our renderer is itself checked
@pytest.mark.validates("crosscheck.polar_images", roots=["mathematics"], kind="reference")
def test_positive_azimuth_is_a_squared_polynomial_with_unit_mean():
    th = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    w = (0.7 - 0.3j, 0.2j, 1.4 + 0.5j)
    f = P.az_positive(th, w)
    assert f.min() >= 0 and abs(f.mean() - 1) < 1e-12
    # |h|^2 has harmonics up to M only: higher Fourier coefficients vanish
    spec = np.abs(np.fft.rfft(f)) / len(th)
    assert spec[len(w) + 1:].max() < 1e-12


@pytest.mark.validates("crosscheck.polar_images", roots=["mathematics"], kind="reference")
def test_rendered_inclined_ring_matches_the_bessel_closed_form():
    """A thin ring seen at inclination i: V = J0(2 pi q' r) at the deprojected frequency."""
    u, v = R.baselines(80, q_max=3e7, seed=5)
    inc, pa, r0, wl = 55.0, 20.0, 6.0, 2.2e-6
    cloud = P.render([r0], [1.0], lambda t: np.ones_like(t), inc, pa, n_theta=2048)
    got = sky.visibility(cloud, u * wl, v * wl, wl)
    qd, _, _ = R.deproject(u, v, inc, pa)
    thin = special.j0(2 * np.pi * qd * r0 * R.MAS)
    assert np.abs(got.real - thin).max() < 1e-4


@pytest.mark.slow
@pytest.mark.validates("virgil.models.Harmonics", roots=["mathematics"], kind="check", tier="B")
def test_harmonic_coefficients_accept_array_paths():
    """F20 (fixed in virgil 1eae3e4): Harmonics.re/im/c/s given as arrays, as the fit and laplace_cov paths
    like 'disk.harmonics.re' require, used to fail in Harmonics._at ('if not entries' on an array)."""
    import jax.numpy as jnp

    from virgil import Harmonics, MultipoleDisk, ThinRing

    disk = MultipoleDisk(ThinRing(5.0), harmonics=Harmonics(w=(0.3 + 0.1j, 0.2j)), inc=30.0, pa=10.0)
    disk = disk.set("harmonics.re", jnp.array([0.3, 0.1])).set("harmonics.im", jnp.array([0.0, 0.2]))
    u = np.array([1e6, 2e6])
    assert np.all(np.isfinite(np.asarray(disk.model(u * 1e-6, 0 * u, 1e-6))))


@pytest.mark.validates("virgil.models.Harmonics", roots=["mathematics"], kind="regression")
def test_harmonic_coefficients_per_element_paths_work():
    from virgil import Harmonics, MultipoleDisk, ThinRing

    disk = MultipoleDisk(ThinRing(5.0), harmonics=Harmonics(w=(0.3 + 0.1j, 0.2j)), inc=30.0, pa=10.0)
    u = np.array([1e6, 2e6])
    changed = disk.set("harmonics.re.0", 0.4)
    assert np.abs(np.asarray(changed.model(u * 1e-6, 0 * u, 1e-6)) - np.asarray(disk.model(u * 1e-6, 0 * u, 1e-6))).max() > 1e-4
