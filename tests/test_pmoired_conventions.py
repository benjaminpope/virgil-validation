"""Stage 0: pin PMOIRED's conventions on our OIFITS files.

Each test writes a noise-free file from our references (crosscheck.sky)
and asks PMOIRED for its model at the parameters we think correspond.
Analytic shapes (points, disks, Gaussians) must match to rounding error
(1e-12 in V^2, 1e-10 deg in closure phase). Rings come from PMOIRED's
sampled radial profile, so they are held to tolerances set by its `Nr`
(2e-5 at 100 points, 2e-7 at 1000, 3e-8 at 3000).

The ambiguous mappings (position axes and closure-phase sign, which axis
`projang` is, what a component without a size is, and the azimuthal
modulation's angle) also have negative controls: the plausible wrong
mapping must fail. The others (`ud`, `fwhm`, ring forms, `spatial kernel`)
are positive checks only. See docs/method/pmoired.md.
"""

import numpy as np
import pytest

from crosscheck import simulate, sky

pytest.importorskip("pmoired")
from external_bridge.pmoired_models import model_minus_data  # noqa: E402

pytestmark = pytest.mark.external

UTS = np.array(
    [[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]]
)


@pytest.fixture
def observe(tmp_path):
    def run(vis, params, setup=None):
        path = tmp_path / "pm.fits"
        simulate.observe(
            path, vis, UTS, hour_angles_h=[-2.0, 0.0, 2.0],
            wavelengths=np.linspace(1.5e-6, 2.4e-6, 4), dec_deg=-50.0,
            sigma_v2=0.01, sigma_cp_deg=0.5,
        )
        return model_minus_data(path, params, setup)

    return run


def _cloud(c):
    return lambda u, v, w: sky.visibility(c, u, v, w)


POINT = {"ud": 0.0}  # a component with no size key is fully resolved


@pytest.mark.validates("external_bridge.pmoired_models", roots=["pmoired", "mathematics"], kind="reference")
def test_position_east_and_closure_phase_sign(observe):
    """x is East, y North (mas); fluxes are total, per component; the
    closure-phase sign agrees with OIFITS and ours."""
    params = {"A,ud": 0.0, "A,f": 1.0, "B,ud": 0.0, "B,f": 0.2, "B,x": 3.0, "B,y": 1.0}

    def binary(east):
        return lambda u, v, w: (
            sky.vis_point(u, v, w) + 0.2 * sky.vis_point(u, v, w, east, 1.0)
        ) / 1.2

    dv2, dcp, cp_max = observe(binary(3.0), params)
    assert cp_max > 20 and dv2 < 1e-12 and dcp < 1e-10
    dv2, dcp, _ = observe(binary(-3.0), params)  # mirrored truth
    assert dcp > 10


@pytest.mark.validates("external_bridge.pmoired_models", roots=["pmoired", "mathematics"], kind="reference")
def test_component_without_size_is_resolved(observe):
    """{'f': 1} alone has zero visibility: point sources need ud = 0."""
    dv2, _, _ = observe(
        lambda u, v, w: sky.vis_point(u, v, w), {"A,f": 1.0, "B,f": 1.0}
    )
    assert dv2 > 0.5


@pytest.mark.validates("external_bridge.pmoired_models", roots=["pmoired", "mathematics"], kind="reference")
def test_uniform_disk_and_offsets(observe):
    dv2, dcp, _ = observe(
        lambda u, v, w: sky.vis_uniform_disk(u, v, w, 2.0, 0.5, -0.3),
        {"ud": 2.0, "x": 0.5, "y": -0.3},
    )
    assert dv2 < 1e-12 and dcp < 1e-10


@pytest.mark.validates("external_bridge.pmoired_models", roots=["pmoired", "mathematics"], kind="reference")
def test_gaussian_fwhm(observe):
    dv2, _, _ = observe(
        lambda u, v, w: sky.vis_gaussian(u, v, w, 1.5 / sky.FWHM_PER_SIGMA),
        {"fwhm": 1.5},
    )
    assert dv2 < 1e-12


@pytest.mark.validates("external_bridge.pmoired_models", roots=["pmoired", "mathematics"], kind="reference")
def test_projang_is_the_major_axis(observe):
    """incl compresses by cos(incl) perpendicular to projang (N to E)."""
    params = {"fwhm": 2.0, "incl": 60.0, "projang": 30.0}
    ratio = np.cos(np.deg2rad(60.0))
    dv2, _, _ = observe(
        lambda u, v, w: sky.vis_elliptical_gaussian(u, v, w, 2.0, ratio, 30.0),
        params,
    )
    assert dv2 < 1e-12
    dv2, _, _ = observe(
        lambda u, v, w: sky.vis_elliptical_gaussian(u, v, w, 2.0, ratio, 120.0),
        params,
    )
    assert dv2 > 0.1


@pytest.mark.parametrize("nr,tol", [(100, 2e-5), (1000, 2e-7), (3000, 3e-8)])
@pytest.mark.validates("external_bridge.pmoired_models", roots=["pmoired", "mathematics"], kind="reference")
def test_annulus_converges_with_nr(observe, nr, tol):
    """Rings are computed from a sampled radial profile: the error falls as
    Nr^-2 (1e-5 at Nr = 100, 1e-8 at 3000)."""
    dv2, _, _ = observe(
        lambda u, v, w: sky.vis_annulus(u, v, w, 2.0, 4.0),
        {"diamin": 2.0, "diamout": 4.0},
        {"Nr": nr},
    )
    assert dv2 < tol


@pytest.mark.validates("external_bridge.pmoired_models", roots=["pmoired", "mathematics"], kind="reference")
def test_diam_thick_is_an_annulus(observe):
    """{'diam': D, 'thick': t} is the annulus from D (1 - t) to D."""
    dv2, _, _ = observe(
        lambda u, v, w: sky.vis_annulus(u, v, w, 3.0, 4.0),
        {"diam": 4.0, "thick": 0.25},
        {"Nr": 1000},
    )
    assert dv2 < 1e-6


@pytest.mark.validates("external_bridge.pmoired_models", roots=["pmoired", "mathematics"], kind="reference")
def test_spatial_kernel_is_a_gaussian_blur(observe):
    dv2, _, _ = observe(
        lambda u, v, w: sky.vis_annulus(u, v, w, 2.0, 4.0)
        * sky.gaussian_blur_factor(u, v, w, 0.7),
        {"diamin": 2.0, "diamout": 4.0, "spatial kernel": 0.7},
        {"Nr": 1000},
    )
    assert dv2 < 1e-6


@pytest.mark.parametrize(
    "modulation,relative,match",
    [("disk", True, True), ("disk", False, False), ("sky", True, False), ("sky", False, False)],
)
@pytest.mark.validates("external_bridge.pmoired_models", roots=["pmoired", "mathematics"], kind="reference")
def test_azimuthal_modulation_convention(observe, modulation, relative, match):
    """'az projangN' is an in-plane azimuth measured from the global
    projang: the same convention as virgil's ModulatedGaussianRim, except
    that virgil's az_pas are absolute (projang + az projangN)."""
    pa_m = 30.0 + 70.0 if relative else 70.0
    ring = sky.inclined_annulus(2.0, 4.0, 60.0, 30.0, (0.6,), (pa_m,), modulation)
    params = {
        "diamin": 2.0, "diamout": 4.0, "incl": 60.0, "projang": 30.0,
        "az amp1": 0.6, "az projang1": 70.0,
    }
    dv2, dcp, cp_max = observe(_cloud(ring), params, {"Nr": 1000})
    assert cp_max > 45
    if match:
        assert dv2 < 1e-6 and dcp < 1e-3
    else:
        assert dcp > 5


@pytest.mark.validates("external_bridge.pmoired_models", roots=["pmoired", "mathematics"], kind="reference")
def test_two_harmonics(observe):
    ring = sky.inclined_annulus(
        2.0, 4.0, 50.0, -20.0, (0.4, 0.3), (-20.0 + 40.0, -20.0 + 100.0), "disk"
    )
    params = {
        "diamin": 2.0, "diamout": 4.0, "incl": 50.0, "projang": -20.0,
        "az amp1": 0.4, "az projang1": 40.0, "az amp2": 0.3, "az projang2": 100.0,
    }
    dv2, dcp, _ = observe(_cloud(ring), params, {"Nr": 1000})
    assert dv2 < 1e-6 and dcp < 1e-3
