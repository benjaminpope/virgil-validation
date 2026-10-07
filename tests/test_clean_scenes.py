"""Gradient CLEAN on a point and a binary, judged by our own dirty image.

Two scenes from our simulator (four VLTI UTs, seven snapshots, two
channels, little noise): a single point at the phase centre, and that point
with a companion at 30 % of its flux on a pixel centre (4.0 mas East,
1.2 mas North). CLEAN runs without a base scene, so its components alone
make the image, normalized to unit sum and anchored at the centre of the
grid (virgil's docstring). The grid follows virgil.models.Image: row 0
North, column 0 East, centre at ((n-1)/2, (n-1)/2) (checked in
tests/test_visibilities.py).

Each recovered image must put the scene's fluxes within one pixel of the
true positions, and the residual dirty image must be flat. The residual is
computed here, not by virgil: the complex visibilities of the truth minus
those of the components (crosscheck.sky, a direct sum over pixel centres)
at every sampled (u, v, wavelength), back-transformed by a NumPy DFT and
divided by the number of samples, so that the dirty beam peaks at 1 and a
residual point of flux F shows as a peak of height F.
"""

import numpy as np
import pytest

from crosscheck import chi2 as ours, simulate, sky
from evidence.plugin import record

imaging = pytest.importorskip("virgil.imaging")
from virgil.oidata import OIData  # noqa: E402

pytestmark = pytest.mark.x64

UTS = np.array([[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]])
NPIX, PIXEL = 31, 0.4
DRA, DDEC, FLUX = 4.0, 1.2, 0.3
MAS = np.pi / 180 / 3600 / 1000

FLUX_TOL = 0.02  # each recovered flux (fraction of the total) within this of the truth
RESIDUAL_PEAK = 0.01  # residual dirty-image peak, as a fraction of the total flux


def point(u, v, w):
    return sky.vis_point(u, v, w)


def binary(u, v, w):
    return (sky.vis_point(u, v, w) + FLUX * sky.vis_point(u, v, w, DRA, DDEC)) / (1 + FLUX)


SCENES = {
    # name: (visibility, [(dra, ddec, flux)] with fluxes as fractions of the total)
    "point": (point, [(0.0, 0.0, 1.0)]),
    "binary": (binary, [(0.0, 0.0, 1 / (1 + FLUX)), (DRA, DDEC, FLUX / (1 + FLUX))]),
}


def run_clean(name, tmp_path_factory):
    vis, truth = SCENES[name]
    path = tmp_path_factory.mktemp("clean") / f"{name}.fits"
    simulate.observe(path, vis, UTS, hour_angles_h=np.linspace(-3, 3, 7), wavelengths=np.array([1.65e-6, 2.2e-6]),
                     dec_deg=-30.0, sigma_v2=1e-3, sigma_cp_deg=0.1, rng=np.random.default_rng(4))
    result = imaging.clean(OIData(str(path)), NPIX, PIXEL, max_iterations=400)
    return name, vis, truth, np.asarray(result.components), ours.load(path)


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    return run_clean("binary", tmp_path_factory)


@pytest.fixture(scope="module")
def point_case(tmp_path_factory):
    return run_clean("point", tmp_path_factory)


def residual_dirty_peak(d, truth_vis, components):
    """Peak |dirty image| of truth minus components, over the grid's field
    sampled at a quarter pixel, with the dirty beam normalized to 1."""
    u, v, w = (np.ravel(d[k]) for k in ("u", "v", "wl"))
    model = sky.visibility(sky.pixel_image(components, PIXEL), u, v, w) if components.sum() > 0 else 0.0
    resid = truth_vis(u, v, w) - model
    fu, fv = u / w * MAS, v / w * MAS  # cycles per mas, East and North
    half = (NPIX - 1) / 2 * PIXEL
    x = np.arange(-half, half + 1e-9, PIXEL / 4)
    ee, nn = (g.ravel() for g in np.meshgrid(x, x))
    dirty = np.real(np.exp(2j * np.pi * (np.outer(ee, fu) + np.outer(nn, fv))) @ resid) / u.size
    return float(np.max(np.abs(dirty)))


def check_positions_and_fluxes(case):
    """Within one pixel (3 x 3) of each true position, the components' flux
    is the truth's to FLUX_TOL and their flux-weighted centre is within
    half a pixel of it; elsewhere, less than FLUX_TOL in all."""
    name, _, truth, comp, _ = case
    c = (NPIX - 1) / 2
    rows, cols = np.indices(comp.shape)
    claimed = np.zeros(comp.shape, bool)
    for dra, ddec, f in truth:
        row, col = round(c - ddec / PIXEL), round(c - dra / PIXEL)
        near = (np.abs(rows - row) <= 1) & (np.abs(cols - col) <= 1)
        claimed |= near
        got = comp[near].sum()
        centre = np.hypot(np.sum(rows * comp * near) / got - row, np.sum(cols * comp * near) / got - col)
        record(f"{name}_flux_error_{dra:g}_{ddec:g}", abs(got - f))
        record(f"{name}_centre_offset_pix_{dra:g}_{ddec:g}", centre)
        assert abs(got - f) < FLUX_TOL
        assert centre < 0.5
    record(f"{name}_flux_elsewhere", comp[~claimed].sum())
    assert comp[~claimed].sum() < FLUX_TOL


def check_residual_flat(case):
    """Our residual dirty image peaks below RESIDUAL_PEAK of the total flux
    (the companion alone would leave a peak of 0.23)."""
    name, vis, _, comp, d = case
    peak = residual_dirty_peak(d, vis, comp)
    record(f"{name}_residual_dirty_peak", peak)
    assert peak < RESIDUAL_PEAK


@pytest.mark.validates("virgil.imaging.clean", roots=["mathematics"])
def test_clean_recovers_positions_and_fluxes(case):
    check_positions_and_fluxes(case)


@pytest.mark.validates("virgil.imaging.clean", roots=["mathematics"])
def test_clean_residual_dirty_image_is_flat(case):
    check_residual_flat(case)


# The point scene cannot fail on CLEAN's own work: without a base scene CLEAN
# seeds the central pixel (F12), normalizes to unit sum and anchors at the
# centre, so even a CLEAN that stops after the seed puts flux 1 on the centre
# pixel. (An off-centre point is no help: V^2 = 1 and CP = 0 at any position.)
# It only shows that CLEAN does not wander off a correct seed: a regression.
@pytest.mark.validates("virgil.imaging.clean", roots=["mathematics"], kind="regression")
def test_clean_point_stays_on_its_seed(point_case):
    check_positions_and_fluxes(point_case)


@pytest.mark.validates("virgil.imaging.clean", roots=["mathematics"], kind="regression")
def test_clean_point_residual_is_flat(point_case):
    check_residual_flat(point_case)


@pytest.mark.validates("virgil.imaging.clean", roots=["mathematics"], kind="control")
def test_residual_dirty_image_sees_a_wrong_image(case, point_case):
    """The residual test has power: the components flipped East-West and
    North-South (the image closure phases would rule out), or no components
    at all against the binary, leave a peak far above the threshold."""
    _, vis, _, comp, d = case
    assert residual_dirty_peak(d, vis, comp[::-1, ::-1]) > 10 * RESIDUAL_PEAK
    assert residual_dirty_peak(d, vis, np.zeros_like(comp)) > 10 * RESIDUAL_PEAK
    _, pvis, _, pcomp, pd = point_case
    corner = np.zeros_like(pcomp)
    corner[0, 0] = 1.0  # all the flux in a corner pixel
    assert residual_dirty_peak(pd, pvis, corner) > 10 * RESIDUAL_PEAK
