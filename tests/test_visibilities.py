"""virgil's model visibilities against independent first-principles ones.

Ours come from crosscheck.sky (quadrature point clouds and SciPy closed
forms); virgil's from its models. Everything runs in float64, where the
two agree to rounding error; a float32 pass checks virgil's default
precision at a looser tolerance.
"""

import numpy as np
import pytest

from crosscheck import sky

vm = pytest.importorskip("virgil.models")

TOL64 = 1e-12
TOL32 = 3e-5


@pytest.fixture(scope="module")
def uvw():
    rng = np.random.default_rng(2026)
    n = 500
    # VLTI-like baselines up to 130 m in the near-infrared
    return (
        rng.uniform(-130, 130, n),
        rng.uniform(-130, 130, n),
        rng.uniform(1.5e-6, 2.4e-6, n),
    )


def virgil_vis(model, u, v, wl):
    return np.asarray(model.model(u, v, wl), complex)


def assert_close(ours, theirs, tol=TOL64):
    err = np.max(np.abs(np.asarray(ours) - np.asarray(theirs)))
    assert err < tol, f"max |dV| = {err:.3e}"


# ------------------------------------------------- quadrature vs closed form
# Our two routes agree with each other first, so a disagreement with virgil
# cannot be blamed on our quadrature.


def test_our_disk_quadrature_matches_airy(uvw):
    u, v, wl = uvw
    assert_close(
        sky.visibility(sky.uniform_disk(3.0, 0.4, -0.7), u, v, wl),
        sky.vis_uniform_disk(u, v, wl, 3.0, 0.4, -0.7),
    )


def test_our_gaussian_quadrature_matches_closed_form(uvw):
    u, v, wl = uvw
    assert_close(
        sky.visibility(sky.elliptical_gaussian(3, 0.4, 35, 1, 2), u, v, wl),
        sky.vis_elliptical_gaussian(u, v, wl, 3, 0.4, 35, 1, 2),
    )


# ------------------------------------------------------------- primitives


@pytest.mark.x64
def test_point_source(uvw):
    u, v, wl = uvw
    assert_close(
        sky.vis_point(u, v, wl, 3.1, -2.2),
        virgil_vis(vm.PointSource(dra=3.1, ddec=-2.2), u, v, wl),
    )


@pytest.mark.x64
@pytest.mark.parametrize("sigma", [0.3, 1.2, 6.0])
def test_gaussian_disk(uvw, sigma):
    u, v, wl = uvw
    assert_close(
        sky.vis_gaussian(u, v, wl, sigma, 0.4, -0.7),
        virgil_vis(vm.GaussianDisk(sigma, dra=0.4, ddec=-0.7), u, v, wl),
    )


@pytest.mark.x64
@pytest.mark.parametrize("pa", [0.0, 35.0, 123.0, 300.0])
def test_elliptical_gaussian(uvw, pa):
    u, v, wl = uvw
    assert_close(
        sky.vis_elliptical_gaussian(u, v, wl, 3.0, 0.4, pa, 1.0, 2.0),
        virgil_vis(
            vm.EllipticalGaussian(3.0, 0.4, pa, dra=1.0, ddec=2.0), u, v, wl
        ),
    )


@pytest.mark.x64
@pytest.mark.parametrize("diam", [0.5, 3.0, 12.0])
def test_uniform_disk_through_nulls(uvw, diam):
    # 12 mas at 130 m / 1.5 um reaches the 4th null of the Airy pattern
    u, v, wl = uvw
    assert_close(
        sky.vis_uniform_disk(u, v, wl, diam, 0.4, -0.7),
        virgil_vis(vm.UniformDisk(diam, dra=0.4, ddec=-0.7), u, v, wl),
    )


@pytest.mark.x64
def test_binaries(uvw):
    u, v, wl = uvw
    sep, pa, f = 5.0, 70.0, 0.1
    dra, ddec = sep * np.sin(np.deg2rad(pa)), sep * np.cos(np.deg2rad(pa))
    ours = (sky.vis_point(u, v, wl) + f * sky.vis_point(u, v, wl, dra, ddec)) / (
        1 + f
    )
    assert_close(ours, virgil_vis(vm.BinaryModelAngular(sep, pa, f), u, v, wl))
    assert_close(
        ours, virgil_vis(vm.BinaryModelCartesian(dra, ddec, f), u, v, wl)
    )


@pytest.mark.x64
@pytest.mark.parametrize("inc", [0.0, 50.0, 80.0])
@pytest.mark.parametrize(
    "amps,pas", [((), ()), ((0.4,), (200.0,)), ((0.4, 0.25), (40.0, 110.0))]
)
def test_modulated_rim_in_plane_azimuth(uvw, inc, amps, pas):
    """virgil's modulation azimuth is the in-plane (deprojected) one."""
    u, v, wl = uvw
    ring = sky.inclined_ring(4.0, inc, 30.0, amps, pas, "disk", 0.3, -0.2)
    ours = sky.visibility(ring, u, v, wl) * sky.gaussian_blur_factor(
        u, v, wl, 0.8
    )
    theirs = vm.ModulatedGaussianRim(
        4.0, 0.8, inc, 30.0, np.array(amps), np.array(pas), dra=0.3, ddec=-0.2
    )
    assert_close(ours, virgil_vis(theirs, u, v, wl))


@pytest.mark.x64
def test_modulated_rim_is_not_modulated_in_sky_angle(uvw):
    """Modulating in on-sky position angle is a different model: virgil's
    az_pas are in-plane azimuths, as its docstring now states."""
    u, v, wl = uvw
    ring = sky.inclined_ring(4.0, 50.0, 30.0, (0.4, 0.25), (40.0, 110.0), "sky")
    ours = sky.visibility(ring, u, v, wl) * sky.gaussian_blur_factor(
        u, v, wl, 0.8
    )
    theirs = vm.ModulatedGaussianRim(
        4.0, 0.8, 50.0, 30.0, np.array([0.4, 0.25]), np.array([40.0, 110.0])
    )
    err = np.max(np.abs(ours - virgil_vis(theirs, u, v, wl)))
    assert err > 1e-3


@pytest.mark.x64
@pytest.mark.parametrize(
    "radius,length", [(5.0, 4.0), (15.0, 20.0), (5.0, 40.0)]
)
def test_gaussian_arc_matches_untruncated_arc(uvw, radius, length):
    """The arc weight is a Gaussian in arc length wrapped once round the
    circle, including when the length FWHM exceeds pi R (virgil >= the
    fix for finding 2; it used to cut the weight at +-3.5 sigma)."""
    u, v, wl = uvw
    ours = sky.visibility(
        sky.arc_curve(radius, length, 250.0), u, v, wl
    ) * sky.gaussian_blur_factor(u, v, wl, 0.6)
    theirs = vm.GaussianArc(radius, 0.6, length, 250.0, nodes=4096)
    assert_close(ours, virgil_vis(theirs, u, v, wl), tol=1e-5)


@pytest.mark.x64
def test_image_orientation_and_centre(uvw):
    """Row 0 North, column 0 East, centre at ((n-1)/2, (m-1)/2): a
    non-square, odd-by-even random image catches any flip or half-pixel."""
    u, v, wl = uvw
    rng = np.random.default_rng(5)
    img = rng.random((7, 10))
    ours = sky.visibility(sky.pixel_image(img, 0.5, 0.2, 0.1), u, v, wl)
    theirs = vm.Image(np.log(img), 0.5, dra=0.2, ddec=0.1)
    assert_close(ours, virgil_vis(theirs, u, v, wl))


@pytest.mark.x64
@pytest.mark.parametrize("angle", [0.0, 70.0, 200.0])
def test_rotated_scene(uvw, angle):
    u, v, wl = uvw
    scene = vm.System(
        a=vm.GaussianDisk(1.0, ddec=3.0),
        b=vm.UniformDisk(2.0, flux=0.5, dra=1.0),
    )
    cloud = sky.mix(
        [sky.gaussian(1.0, 0, 3.0), sky.uniform_disk(2.0, 1.0)], [1, 0.5]
    ).rotated(angle)
    assert_close(
        sky.visibility(cloud, u, v, wl),
        virgil_vis(vm.Rotated(scene, angle), u, v, wl),
    )


@pytest.mark.x64
def test_resolved_flux_dilutes(uvw):
    u, v, wl = uvw
    cloud = sky.mix([sky.uniform_disk(1.0)], [1.0], resolved_flux=0.3)
    scene = vm.System(s=vm.UniformDisk(1.0), r=vm.Resolved(0.3))
    assert_close(sky.visibility(cloud, u, v, wl), virgil_vis(scene, u, v, wl))


# ----------------------------------------------------- random constellations


def _random_constellation(seed):
    """Tens of points, Gaussians, elliptical Gaussians and uniform disks at
    random positions and fluxes, with a nested sub-system. Returns our
    closed-form visibility function and the virgil System."""
    rng = np.random.default_rng(seed)
    parts, ours = {}, []
    for k in range(24):
        kind = rng.choice(["point", "gauss", "egauss", "disk"])
        dra, ddec = rng.uniform(-40, 40, 2)
        f = rng.uniform(0.05, 2.0)
        if kind == "point":
            parts[f"p{k}"] = vm.PointSource(f, dra, ddec)
            ours.append((f, lambda u, v, w, a=dra, b=ddec: sky.vis_point(u, v, w, a, b)))
        elif kind == "gauss":
            s = rng.uniform(0.2, 5)
            parts[f"g{k}"] = vm.GaussianDisk(s, f, dra, ddec)
            ours.append((f, lambda u, v, w, s=s, a=dra, b=ddec: sky.vis_gaussian(u, v, w, s, a, b)))
        elif kind == "egauss":
            fw, r, pa = rng.uniform(0.5, 8), rng.uniform(0.1, 1), rng.uniform(0, 360)
            parts[f"e{k}"] = vm.EllipticalGaussian(fw, r, pa, f, dra, ddec)
            ours.append((f, lambda u, v, w, fw=fw, r=r, pa=pa, a=dra, b=ddec: sky.vis_elliptical_gaussian(u, v, w, fw, r, pa, a, b)))
        else:
            d = rng.uniform(0.3, 10)
            parts[f"d{k}"] = vm.UniformDisk(d, f, dra, ddec)
            ours.append((f, lambda u, v, w, d=d, a=dra, b=ddec: sky.vis_uniform_disk(u, v, w, d, a, b)))
    # a nested group, moved and weighted as a whole
    gdx, gdy, gf = rng.uniform(-20, 20), rng.uniform(-20, 20), 0.7
    parts["group"] = vm.System(
        core=vm.PointSource(),
        halo=vm.GaussianDisk(2.0, 0.5, 1.0, -1.0),
        flux=gf,
        dra=gdx,
        ddec=gdy,
    )

    def group(u, v, w):
        inner = (
            sky.vis_point(u, v, w) + 0.5 * sky.vis_gaussian(u, v, w, 2.0, 1.0, -1.0)
        ) / 1.5
        return inner * sky.shift_phase(u, v, w, gdx, gdy)

    ours.append((gf, group))
    total = sum(f for f, _ in ours)

    def vis(u, v, w):
        return sum(f * fn(u, v, w) for f, fn in ours) / total

    return vis, vm.System(**parts)


@pytest.mark.x64
@pytest.mark.parametrize("seed", range(5))
def test_random_constellation(uvw, seed):
    u, v, wl = uvw
    vis, scene = _random_constellation(seed)
    assert_close(vis(u, v, wl), virgil_vis(scene, u, v, wl))


@pytest.mark.float32
@pytest.mark.parametrize("seed", range(3))
def test_random_constellation_float32(uvw, seed):
    """virgil's default precision: float32 throughout."""
    u, v, wl = uvw
    vis, scene = _random_constellation(seed)
    assert_close(vis(u, v, wl), virgil_vis(scene, u, v, wl), tol=TOL32)
