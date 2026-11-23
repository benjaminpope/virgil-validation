"""virgil.models.GaussianArc orientation convention against PMOIRED.

``GaussianArc(radius, width, length, pa)`` places a circle of radius ``r``
mas blurred isotropically on the sky by a Gaussian of FWHM ``width`` mas and
weighted in position angle ``phi`` (degrees North through East) by the von
Mises density ``exp(kappa cos(phi - pa))`` with concentration
``kappa = 8 ln 2 (radius / length)^2`` (so ``length`` is the arc-length FWHM
in mas for a short arc).

Expanding ``exp(kappa cos(phi - pa)) / (2 pi I_0(kappa))`` in a Fourier
series gives unit total flux plus azimuthal harmonics
``A_m = 2 I_m(kappa) / I_0(kappa)`` at position angle ``pa`` for ``m >= 1``.
In PMOIRED (Mérand 2022), a face-on (``incl = 0, projang = 0``) thin native
``ring`` (``diamin = 2 r - 1e-4, diamout = 2 r + 1e-4``) convolved with
``spatial kernel = width`` and modulated by ``az amp{m} = A_m``,
``az projang{m} = pa`` (``m = 1..8``) constructs the same sky brightness
independently in PMOIRED's coordinate frame. Keeping ``epochs * channels <= 30``
per baseline avoids PMOIRED's ring precision cap (ledger P3).
"""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special

from crosscheck import simulate
from evidence.plugin import record
from external_bridge import pmoired_models as pm

pytest.importorskip("pmoired")
vm = pytest.importorskip("virgil.models")

pytestmark = [pytest.mark.x64, pytest.mark.external]

# Compact 4-station array with all baselines <= 25 m (5 HAs x 6 channels = 30
# samples per baseline, staying within PMOIRED's exact ring path; see P3).
STATIONS_25M = np.array([
    [0.0, 0.0],
    [9.5, 3.5],
    [-5.0, 12.0],
    [7.0, -7.5],
])
ARC = {"radius": 5.0, "width": 1.5, "length": 12.0, "pa": 37.0}


@pytest.fixture(scope="module")
def pmoired_arc_samples(tmp_path_factory):
    """PMOIRED native ring samples on a <= 25 m, 30-sample/baseline carrier."""
    path = tmp_path_factory.mktemp("arc_pmoired") / "arc_carrier.fits"
    simulate.observe(
        path,
        lambda u, v, w: np.ones(np.broadcast(u, v, w).shape, complex),
        STATIONS_25M,
        hour_angles_h=np.linspace(-2.0, 2.0, 5),
        wavelengths=np.linspace(1.6e-6, 2.2e-6, 6),
        dec_deg=-25.0,
        sigma_v2=1e-3,
        sigma_cp_deg=0.1,
    )
    r, w, length, pa = ARC["radius"], ARC["width"], ARC["length"], ARC["pa"]
    kappa = 8.0 * np.log(2.0) * (r / length) ** 2
    params = {
        "arc,diamin": 2.0 * r - 1e-4,
        "arc,diamout": 2.0 * r + 1e-4,
        "arc,incl": 0.0,
        "arc,projang": 0.0,
        "arc,spatial kernel": w,
    }
    for m in range(1, 9):
        params[f"arc,az amp{m}"] = float(2.0 * special.iv(m, kappa) / special.i0(kappa))
        params[f"arc,az projang{m}"] = pa
    samples = pm.model_samples(path, params, setup={"Nr": 1000})
    # Verify baselines stay <= 25 m (u, v are in m/um and min wl is 1.6 um).
    b_max_m = float(np.max(np.hypot(samples["u"], samples["v"])) * 1.6)
    assert b_max_m <= 25.0
    return samples


def _compare_arc_to_pmoired(samples, pa_deg):
    """Evaluate virgil GaussianArc at PMOIRED's exact (u, v)/wl frequencies."""
    arc = vm.GaussianArc(ARC["radius"], ARC["width"], ARC["length"], pa=pa_deg)
    # model_samples returns u, v in m/um = 1e6 cycles/rad -> pass wavel = 1e-6 m.
    vis = np.asarray(arc.model(jnp.asarray(samples["u"]), jnp.asarray(samples["v"]), 1e-6))
    v2 = np.abs(vis) ** 2
    t1 = np.asarray(arc.model(jnp.asarray(samples["u1"]), jnp.asarray(samples["v1"]), 1e-6))
    t2 = np.asarray(arc.model(jnp.asarray(samples["u2"]), jnp.asarray(samples["v2_"]), 1e-6))
    t3 = np.asarray(
        arc.model(
            jnp.asarray(samples["u1"] + samples["u2"]),
            jnp.asarray(samples["v1"] + samples["v2_"]),
            1e-6,
        )
    )
    cp_deg = np.rad2deg(np.angle(t1 * t2 * np.conj(t3)))
    dv2 = float(np.max(np.abs(v2 - samples["v2"])))
    dcp = float(np.max(np.abs((cp_deg - samples["t3phi"] + 180.0) % 360.0 - 180.0)))
    cp_peak = float(np.max(np.abs(samples["t3phi"])))
    return dv2, dcp, cp_peak


@pytest.mark.validates(
    "virgil.models.GaussianArc",
    roots=["pmoired"],
    property="orientation",
    headline="max_abs_dV2",
)
def test_gaussian_arc_orientation_matches_pmoired_ring(pmoired_arc_samples):
    """GaussianArc(5.0, 1.5, 12.0, pa=37.0) matches PMOIRED's native modulated
    blurred ring to < 1e-5 in V^2 and < 1e-3 deg on strong closure phases."""
    dv2, dcp, cp_peak = _compare_arc_to_pmoired(pmoired_arc_samples, ARC["pa"])
    record("max_abs_dV2", dv2)
    record("max_abs_dCP_deg", dcp)
    record("max_abs_CP_deg", cp_peak)
    assert cp_peak > 15.0
    assert dv2 < 1e-5
    assert dcp < 1e-3


@pytest.mark.parametrize(
    "label,wrong_pa",
    [
        ("pa_plus_90", ARC["pa"] + 90.0),
        ("pa_plus_180", ARC["pa"] + 180.0),
        ("minus_pa", -ARC["pa"]),
    ],
)
@pytest.mark.validates("virgil.models.GaussianArc", roots=["pmoired"], kind="control")
def test_gaussian_arc_wrong_orientations_fail_against_pmoired(pmoired_arc_samples, label, wrong_pa):
    """Rotating the arc to pa + 90 (East-from-North error), pa + 180 (opposite
    ridge), or -pa (clockwise PA) misses PMOIRED's closure phases by > 10 deg."""
    dv2, dcp, _ = _compare_arc_to_pmoired(pmoired_arc_samples, wrong_pa)
    record(f"control_{label}_dV2", dv2)
    record(f"control_{label}_dCP_deg", dcp)
    assert dcp > 10.0
