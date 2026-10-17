"""scripts/postagb_imaging.py: the position-angle convention of region_metrics (North up, East left)."""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
pytest.importorskip("virgil")
import postagb_imaging as P  # noqa: E402

# Guards our own script's convention, not virgil against an independent result.
pytestmark = pytest.mark.validates("evidence", roots=["standards"], kind="guard")


@pytest.mark.parametrize("pa", [20.0, 80.0, 130.0, 170.0])
def test_major_axis_pa_is_east_of_north(pa):
    n, scale = 129, 0.5
    yy, xx = np.mgrid[:n, :n]
    east = -(xx - (n - 1) / 2) * scale  # East is left (image columns decrease towards East)
    north = -(yy - (n - 1) / 2) * scale  # row 0 is North
    t = np.radians(pa)
    along = east * np.sin(t) + north * np.cos(t)
    across = east * np.cos(t) - north * np.sin(t)
    img = np.exp(-0.5 * ((along / 6.0) ** 2 + (across / 2.0) ** 2))
    m = P.region_metrics(img, scale, 0.0, 30.0)
    assert m["major_pa_deg"] == pytest.approx(pa, abs=0.5)


def test_centroid_pa_points_to_the_offset_blob():
    n, scale = 129, 0.5
    yy, xx = np.mgrid[:n, :n]
    img = np.exp(-0.5 * (((xx - 64 + 8) ** 2 + (yy - 64 + 8) ** 2) / 4.0))  # up-left on the array: North-East
    m = P.region_metrics(img, scale, 0.0, 30.0)  # up = North, left = East: PA 45 deg
    assert m["centroid_pa_deg"] == pytest.approx(45.0, abs=1.0)


def test_bitmap_to_image_recovers_beam_pa_and_colourbar():
    """A synthetic panel (grey colour bar, blue beam ellipse at PA 30 deg East of North) round-trips."""
    import io

    import postagb_literature as L
    from PIL import Image

    x0, x1, y0, y1, cbar = 171, 650, 61, 540, 672
    im = np.full((600, 800, 3), 255, np.uint8)
    ramp = np.linspace(0, 255, y1 - y0 - 2).astype(np.uint8)  # black (value 1) at the top to white (0) at the bottom
    im[y0 + 1:y1 - 1, cbar, :] = ramp[:, None]
    im[y0:y1, x0:x1, :] = 128  # value ~0.5 everywhere
    pa, a, b = np.radians(30.0), 18.0, 9.0  # semi-axes in pixels; the box for the beam is 45 x 60
    yy, xx = np.mgrid[:45, :60]
    dx, dy = xx - 30.0, yy - 22.0
    u = dx * (-np.sin(pa)) + dy * (-np.cos(pa))  # along the major axis (right = West, down = South)
    v = dx * np.cos(pa) + dy * (-np.sin(pa))
    inside = (u / a) ** 2 + (v / b) ** 2 <= 1
    box = im[y0:y0 + 45, x0:x0 + 60]
    box[inside] = (0, 0, 255)
    buf = io.BytesIO()
    Image.fromarray(im).save(buf, "PNG")
    img, scale, beam, _, _ = L.bitmap_to_image(buf.getvalue(), axes=(x0, x1, y0, y1), cbar_x=cbar)
    assert beam["pa_deg"] == pytest.approx(30.0, abs=2.0)
    assert beam["major_mas"] / beam["minor_mas"] == pytest.approx(a / b, rel=0.1)
    assert beam["major_mas"] == pytest.approx(2 * a * scale, rel=0.1)
    assert np.nanmedian(img) == pytest.approx(0.5, abs=0.03)  # the colour-bar inversion (top = 1)
