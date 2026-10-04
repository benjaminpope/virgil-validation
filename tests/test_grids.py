"""virgil's image <-> uv machinery against analytic results.

Rendering, pixel images on rotated uv lattices (the fast matrix Fourier
transform path), dirty images and beams, Nyquist pixels, fields of view and
beam convolution, each checked against a closed form.
"""

import numpy as np
import pytest
from scipy.special import j1

from crosscheck import sky

vm = pytest.importorskip("virgil.models")
im = pytest.importorskip("virgil.imaging")
from virgil.oidata import OIData, find_uv_grid  # noqa: E402

pytestmark = pytest.mark.x64
WL = 2.0e-6


def _phase_data(u, v, cvis, wl=WL):
    """OIData holding complex visibilities as V^2 and absolute phases."""
    return OIData(
        {
            "u": np.asarray(u), "v": np.asarray(v), "wavel": np.array([wl]),
            "vis": np.abs(cvis) ** 2, "d_vis": np.full(u.shape, 0.01),
            "phi": np.angle(cvis), "d_phi": np.full(u.shape, 0.01),
            "i_cps1": None, "i_cps2": None, "i_cps3": None,
            "v2_flag": True, "cp_flag": False,
        }
    )


def _disk_uv(q_max_m, n_r=48, n_t=96, ratio=1.0, pa_deg=0.0):
    """Samples filling a disk (or ellipse) in the uv plane with equal area
    per sample: radii at sqrt-spaced midpoints, uniform azimuths. Only half
    the plane: virgil counts each sample with its conjugate."""
    r = q_max_m * np.sqrt((np.arange(n_r) + 0.5) / n_r)
    t = np.pi * (np.arange(n_t) + 0.5) / n_t
    rr, tt = np.meshgrid(r, t)
    a, b = rr * np.cos(tt), rr * np.sin(tt) * ratio  # a along PA
    p = np.deg2rad(pa_deg)
    u = a * np.sin(p) + b * np.cos(p)
    v = a * np.cos(p) - b * np.sin(p)
    return u.ravel(), v.ravel()


# ------------------------------------------------------------- dirty image


@pytest.mark.validates("virgil.imaging.dirty_image", roots=["mathematics"])
def test_dirty_image_of_point_is_airy_at_the_source():
    """Uniform coverage of a uv disk of radius q (cycles/rad): the dirty
    beam is 2 J1(2 pi q r) / (2 pi q r), centred on the source."""
    bmax = 100.0
    u, v = _disk_uv(bmax, n_r=64, n_t=128)
    dra, ddec = 6.0, -4.0
    data = _phase_data(u, v, sky.vis_point(u, v, WL, dra, ddec))
    npix, pix = 41, 0.5
    dirty = np.asarray(im.dirty_image(data, npix, pix))
    east = ((npix - 1) / 2 - np.arange(npix)) * pix  # East left
    north = ((npix - 1) / 2 - np.arange(npix)) * pix  # North up
    ee, nn = np.meshgrid(east, north)
    q = bmax / WL
    x = 2 * np.pi * q * np.hypot(ee - dra, nn - ddec) * sky.MAS
    airy = np.where(x > 0, 2 * j1(x) / np.where(x > 0, x, 1), 1.0)
    # sampling the disk with 64 x 128 points: quadrature error ~1e-3
    assert np.max(np.abs(dirty - airy)) < 3e-3
    peak = np.unravel_index(np.argmax(dirty), dirty.shape)
    assert np.isclose(ee[peak], dra) and np.isclose(nn[peak], ddec)


# ------------------------------------------------------------------- beam


@pytest.mark.validates("virgil.imaging.beam", roots=["mathematics"])
def test_beam_of_filled_disk():
    """<u u^T> = q^2 / 4 for a filled disk, so FWHM = 2 sqrt(2 ln 2) / (pi q)."""
    bmax = 100.0
    u, v = _disk_uv(bmax, n_r=200, n_t=200)
    data = _phase_data(u, v, np.ones_like(u, complex))
    beam = im.beam(data)
    q = bmax / WL
    fwhm = sky.FWHM_PER_SIGMA / (np.pi * q) / sky.MAS
    assert np.isclose(beam.major_mas, fwhm, rtol=1e-4)
    assert np.isclose(beam.minor_mas, fwhm, rtol=1e-4)


@pytest.mark.parametrize("uv_pa", [0.0, 30.0, 100.0])
@pytest.mark.validates("virgil.imaging.beam", roots=["mathematics"])
def test_beam_of_filled_ellipse(uv_pa):
    """Coverage elongated along PA p (axes q, q*ratio) gives a beam
    elongated along p + 90, with FWHMs in inverse ratio."""
    bmax, ratio = 100.0, 0.4
    u, v = _disk_uv(bmax, n_r=200, n_t=200, ratio=ratio, pa_deg=uv_pa)
    data = _phase_data(u, v, np.ones_like(u, complex))
    beam = im.beam(data)
    q = bmax / WL
    narrow = sky.FWHM_PER_SIGMA / (np.pi * q) / sky.MAS
    assert np.isclose(beam.minor_mas, narrow, rtol=1e-3)
    assert np.isclose(beam.major_mas, narrow / ratio, rtol=1e-3)
    d = (beam.pa_deg - (uv_pa + 90.0)) % 180.0
    assert min(d, 180.0 - d) < 0.05


@pytest.mark.validates("virgil.imaging.nyquist_pixel_scale", "virgil.imaging.field_of_view", roots=["mathematics"])
def test_nyquist_and_field_of_view():
    u = np.array([3.0, 40.0, -120.0])
    v = np.array([4.0, 0.0, 50.0])
    data = _phase_data(u, v, np.ones(3, complex))
    bmax, bmin = np.hypot(-120, 50), 5.0
    assert np.isclose(
        im.nyquist_pixel_scale(data), WL / (2 * bmax) / sky.MAS, rtol=1e-9
    )
    fov = WL / bmin / sky.MAS
    assert np.isclose(im.field_of_view(data, largest_mas=1e6), fov, rtol=1e-9)
    assert np.isclose(im.field_of_view(data, largest_mas=10.0), min(fov, 10.0))


# -------------------------------------------------------- beam convolution


@pytest.mark.parametrize("pa", [0.0, 35.0, 120.0])
@pytest.mark.validates("virgil.imaging.convolve_beam", roots=["mathematics"])
def test_convolve_beam_of_delta_is_the_beam(pa):
    npix, pix = 101, 0.25
    delta = np.zeros((npix, npix))
    delta[50, 50] = 1.0
    beam = im.Beam(major_mas=4.0, minor_mas=2.0, pa_deg=pa)
    out = np.asarray(im.convolve_beam(delta, pix, beam))
    east = ((npix - 1) / 2 - np.arange(npix)) * pix
    ee, nn = np.meshgrid(east, east)
    assert np.isclose(out.sum(), 1.0, atol=1e-6)
    # second moments of the result against the Gaussian's covariance
    w = out / out.sum()
    cov = np.array(
        [[np.sum(w * ee * ee), np.sum(w * ee * nn)],
         [np.sum(w * ee * nn), np.sum(w * nn * nn)]]
    )
    p = np.deg2rad(pa)
    major = np.array([np.sin(p), np.cos(p)])
    minor = np.array([np.cos(p), -np.sin(p)])
    s1, s2 = 4.0 / sky.FWHM_PER_SIGMA, 2.0 / sky.FWHM_PER_SIGMA
    expected = s1**2 * np.outer(major, major) + s2**2 * np.outer(minor, minor)
    # pixel variance pix^2 / 12 is not part of a sampled kernel
    assert np.allclose(cov, expected, atol=2e-3)


# ---------------------------------------------------------------- render


@pytest.mark.validates("virgil.models.SourceModel.render", roots=["standards"])
def test_render_point_lands_on_its_pixel():
    """East left, North up: a source at (+dra, +ddec) sits left of and above
    the centre."""
    npix, fov = 41, 41.0  # 1 mas pixels, centre pixel 20
    img = np.asarray(vm.PointSource(dra=5.0, ddec=3.0).render(npix, fov))
    row, col = np.unravel_index(np.argmax(img), img.shape)
    assert (row, col) == (20 - 3, 20 - 5)
    assert np.isclose(img.sum(), 1.0)


@pytest.mark.validates("virgil.models.SourceModel.render", roots=["mathematics"])
def test_render_gaussian_matches_its_visibility():
    """The rendered image's DFT (our pixel convention) reproduces the
    analytic Gaussian visibility on baselines well inside the pixel
    Nyquist limit."""
    model = vm.GaussianDisk(3.0, dra=2.0, ddec=-1.5)
    npix, fov = 128, 64.0
    img = np.asarray(model.render(npix, fov))
    cloud = sky.pixel_image(img, fov / npix)
    rng = np.random.default_rng(3)
    u, v = rng.uniform(-40, 40, (2, 200))
    ours = sky.visibility(cloud, u, v, WL)
    assert np.max(np.abs(ours - sky.vis_gaussian(u, v, WL, 3.0, 2.0, -1.5))) < 1e-9


@pytest.mark.validates("virgil.models.Image.from_model", "virgil.models.Image", roots=["mathematics"])
def test_image_from_model_round_trip():
    """Pixelising a Gaussian and transforming it back gives the Gaussian's
    visibility (an Image is exact for its pixels). The default brightness
    floor (1e-6 of the peak, so that log-brightness is finite) adds ~1e-5
    of spurious flux over this field; switch it off to test the transform."""
    g = vm.GaussianDisk(4.0, dra=1.0, ddec=2.0)
    image = vm.Image.from_model(g, 160, 0.5, floor=1e-300)  # +-10 sigma
    rng = np.random.default_rng(4)
    u, v = rng.uniform(-60, 60, (2, 200))
    got = np.asarray(image.model(u, v, WL))
    assert np.max(np.abs(got - sky.vis_gaussian(u, v, WL, 4.0, 1.0, 2.0))) < 1e-9


# ------------------------------------------------- rotated uv lattice (MFT)


@pytest.mark.parametrize("rotation", [0.0, 17.0, -38.0])
@pytest.mark.validates("virgil.oidata.find_uv_grid", "virgil.models.Image.model_on_grid", roots=["mathematics"])
def test_lattice_transform_matches_direct_dft(rotation):
    """An Image whose pixel grid is rotated like its data's uv lattice is
    transformed by a two-sided matrix Fourier transform on the lattice.
    Check it against a direct sum over rotated pixel centres."""
    pitch, nu, nv = 0.35, 15, 9
    ug = (np.arange(nu) - 3) * pitch  # half-plane-ish, off-centre lattice
    vg = (np.arange(nv) - 4) * pitch
    uu, vv = np.meshgrid(ug, vg)
    keep = np.hypot(uu, vv) > 0
    ug_s, vg_s = uu[keep], vv[keep]
    t = np.deg2rad(rotation)
    # grid frame -> sky: rotate by the position angle of the grid's "up"
    u = ug_s * np.cos(t) + vg_s * np.sin(t)
    v = -ug_s * np.sin(t) + vg_s * np.cos(t)
    grid = find_uv_grid(u, v)
    assert grid is not None
    assert np.isclose(((grid.rotation_deg - rotation + 45) % 90) - 45, 0, atol=1e-6)

    rng = np.random.default_rng(6)
    pix = 20.0
    img = rng.random((12, 16))
    image = vm.Image(np.log(img), pix, rotation_deg=float(grid.rotation_deg))
    # OIData attaches a lattice only to AMIGO DISCO records, so pass it in
    fast = np.asarray(image.model_on_grid(u, v, np.array([4.3e-6]), grid))
    # ours: pixel centres in the image frame, rotated onto the sky
    cloud = sky.pixel_image(img, pix).rotated(float(grid.rotation_deg))
    direct = sky.visibility(cloud, u, v, 4.3e-6)
    assert np.max(np.abs(fast - direct)) < 1e-12
    slow = np.asarray(image.model(u, v, 4.3e-6))
    assert np.max(np.abs(slow - direct)) < 1e-12
