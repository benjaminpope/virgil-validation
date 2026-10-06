"""Gradient CLEAN recovers a companion next to a known star.

The file is from our simulator: four VLTI UTs, seven snapshots, a star (a
0.3 mas uniform disk) and a point companion at 30% of its flux, placed on
a pixel centre of the grid, with little noise. CLEAN is given the star as
its fixed base. The components' pixel grid follows virgil.models.Image
(row 0 North, column 0 East, centre at ((n-1)/2, (n-1)/2); checked in
tests/test_visibilities.py::test_image_orientation_and_centre), so the
companion belongs at row c - ddec/p and column c - dra/p.
"""

import numpy as np
import pytest

from crosscheck import simulate, sky
from evidence.plugin import record

imaging = pytest.importorskip("virgil.imaging")
vm = pytest.importorskip("virgil.models")
from virgil.oidata import OIData  # noqa: E402

pytestmark = [pytest.mark.x64, pytest.mark.slow]

UTS = np.array([[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]])
NPIX, PIXEL, DRA, DDEC, FLUX = 41, 0.4, 4.0, 1.2, 0.3


def _scene(u, v, w):
    return (sky.vis_uniform_disk(u, v, w, 0.3) + FLUX * sky.vis_point(u, v, w, DRA, DDEC)) / (1 + FLUX)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    path = tmp_path_factory.mktemp("clean") / "c.fits"
    simulate.observe(path, _scene, UTS, hour_angles_h=np.linspace(-3, 3, 7), wavelengths=np.array([1.65e-6, 2.2e-6]),
                     dec_deg=-30.0, sigma_v2=1e-3, sigma_cp_deg=0.1, rng=np.random.default_rng(4))
    return imaging.clean(OIData(str(path)), NPIX, PIXEL, base=vm.UniformDisk(0.3), max_iterations=2000)


@pytest.mark.validates("virgil.imaging.clean", roots=["mathematics"])
def test_clean_recovers_the_companion(result):
    """Within one pixel of the companion: its flux to 5%, the flux-weighted
    centre to half a pixel; elsewhere less than 5% of it; and the final
    chi-squared per point at most 1.5."""
    comp = np.asarray(result.components)
    c = (NPIX - 1) / 2
    row, col = round(c - DDEC / PIXEL), round(c - DRA / PIXEL)
    near = np.zeros_like(comp, bool)
    near[row - 1:row + 2, col - 1:col + 2] = True
    flux_near, flux_far = comp[near].sum(), comp[~near].sum()
    rows, cols = np.indices(comp.shape)
    centre = (np.sum(rows * comp * near) / flux_near, np.sum(cols * comp * near) / flux_near)
    offset = np.hypot(centre[0] - row, centre[1] - col)
    record("rel_flux_error", abs(flux_near / FLUX - 1))
    record("centre_offset_pix", offset)
    record("flux_elsewhere_rel", flux_far / FLUX)
    record("final_chi2_red", float(np.ravel(result.chi2_red)[-1]))
    assert abs(flux_near / FLUX - 1) < 0.05
    assert offset < 0.5
    assert flux_far < 0.05 * FLUX
    assert np.ravel(result.chi2_red)[-1] < 1.5


@pytest.mark.validates("virgil.imaging.clean", roots=["mathematics"], kind="control")
def test_the_mirror_image_position_is_empty(result):
    """A control on orientation: the point reflected through the centre
    (where closure phases alone could put a flipped image) holds almost
    nothing."""
    comp = np.asarray(result.components)
    c = (NPIX - 1) / 2
    row, col = round(c + DDEC / PIXEL), round(c + DRA / PIXEL)
    assert comp[row - 1:row + 2, col - 1:col + 2].sum() < 0.05 * FLUX
