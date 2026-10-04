"""virgil's limb-darkened disks against first-principles references
(Issue #12).

Ours come from ``crosscheck.limb``: a quadrature point cloud, the Hankel
transform by adaptive quadrature, and Hanbury Brown et al.'s (1974) closed
form for the linear law; the Kipping (2013) maps and physical constraints are
coded from the paper. virgil's conventions are taken from its docstrings:
``diam`` is the limb-darkened (outer) diameter, ``u`` follows starry,
``q1, q2`` follow Kipping's eqs. 15-18 and 23-24, and powers of mu passed to
``cvis_limb_darkened_disk`` must lie in (-2, 22].
"""

import numpy as np
import pytest

from crosscheck import limb, simulate, sky
from evidence.plugin import record

vm = pytest.importorskip("virgil.models")
vb = pytest.importorskip("virgil_bridge")

TOL64 = 1e-10
TOL32 = 3e-5

UTS = np.array(
    [[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]]
)
HA = np.linspace(-3, 3, 7)
WL = np.linspace(1.5e-6, 2.4e-6, 6)
DEC = -50.0


@pytest.fixture(scope="module")
def uvw():
    """VLTI-like random baselines, plus a dense radial cut so that every
    null of a 12 mas disk out to 130 m / 1.5 um has samples beside it."""
    rng = np.random.default_rng(2026)
    n = 300
    radial = np.linspace(0.0, 130.0, 1500)
    pa = np.deg2rad(37.0)
    return (
        np.concatenate([rng.uniform(-130, 130, n), radial * np.sin(pa)]),
        np.concatenate([rng.uniform(-130, 130, n), radial * np.cos(pa)]),
        np.concatenate([rng.uniform(1.5e-6, 2.4e-6, n), np.full(1500, 1.5e-6)]),
    )


def virgil_vis(model, u, v, wl):
    return np.asarray(model.model(u, v, wl), complex)


def assert_close(ours, theirs, tol=TOL64):
    err = record("max_abs_dV", np.max(np.abs(np.asarray(ours) - np.asarray(theirs))))
    assert err < tol, f"max |dV| = {err:.3e}"


LAWS = {
    "uniform": [],
    "linear": [0.6],
    "quadratic": [0.35, 0.25],
    "cubic": [0.3, 0.2, 0.1],
    "order-22": list(np.linspace(0.05, 0.01, 22)),  # the highest documented
}

# ------------------------------------------------ our routes agree first


@pytest.mark.parametrize("diam", [3.0, 12.0])
@pytest.mark.parametrize(
    "profile",
    [limb.polynomial(u) for u in LAWS.values()] + [limb.square_root(0.1, 0.6)],
    ids=list(LAWS) + ["square-root"],
)
@pytest.mark.validates("crosscheck.limb", roots=["mathematics"], kind="reference")
def test_our_cloud_matches_our_hankel_transform(uvw, profile, diam):
    u, v, wl = (a[::10] for a in uvw)  # quad is slow; every 10th sample
    assert_close(
        sky.visibility(limb.disk(diam, profile, 0.4, -0.7), u, v, wl),
        limb.vis_hankel(u, v, wl, diam, profile, 0.4, -0.7),
        tol=1e-12,
    )


@pytest.mark.validates("crosscheck.limb", roots=["literature", "mathematics"], kind="reference")
def test_our_cloud_matches_hanbury_brown_linear_law(uvw):
    u, v, wl = uvw
    assert_close(
        sky.visibility(limb.disk(12.0, limb.polynomial([0.6]), 0.4, -0.7), u, v, wl),
        limb.vis_linear_closed_form(u, v, wl, 12.0, 0.6, 0.4, -0.7),
        tol=1e-12,
    )


@pytest.mark.validates("crosscheck.limb", roots=["mathematics"], kind="reference")
def test_our_uniform_limit_is_the_airy_pattern(uvw):
    u, v, wl = uvw
    assert_close(
        sky.visibility(limb.disk(12.0, limb.polynomial([])), u, v, wl),
        sky.vis_uniform_disk(u, v, wl, 12.0),
        tol=1e-12,
    )


# ------------------------------------------------------- virgil's models


@pytest.mark.x64
@pytest.mark.parametrize("diam", [0.5, 3.0, 12.0])
@pytest.mark.parametrize("law", list(LAWS))
@pytest.mark.validates("virgil.models.LimbDarkenedDisk", roots=["mathematics"])
def test_polynomial_laws(uvw, law, diam):
    u, v, wl = uvw
    ours = sky.visibility(limb.disk(diam, limb.polynomial(LAWS[law]), 0.4, -0.7), u, v, wl)
    theirs = virgil_vis(
        vm.LimbDarkenedDisk(diam, u=LAWS[law], dra=0.4, ddec=-0.7), u, v, wl
    )
    assert_close(ours, theirs)


@pytest.mark.x64
@pytest.mark.parametrize("q1, q2", [(0.36, 0.29), (0.9, 0.1), (0.05, 0.8), (0.5, 0.5)])
@pytest.mark.validates("virgil.models.QuadraticLimbDarkenedDisk", roots=["mathematics", "literature"])
def test_kipping_quadratic(uvw, q1, q2):
    u, v, wl = uvw
    profile = limb.polynomial(limb.kipping_quadratic_u(q1, q2))
    assert_close(
        sky.visibility(limb.disk(12.0, profile, -1.1, 0.3), u, v, wl),
        virgil_vis(vm.QuadraticLimbDarkenedDisk(12.0, q1, q2, dra=-1.1, ddec=0.3), u, v, wl),
    )


@pytest.mark.x64
@pytest.mark.parametrize("q1, q2", [(0.36, 0.29), (0.9, 0.1), (0.05, 0.8), (0.5, 0.5)])
@pytest.mark.validates("virgil.models.SquareRootLimbDarkenedDisk", roots=["mathematics", "literature"])
def test_kipping_square_root(uvw, q1, q2):
    u, v, wl = uvw
    profile = limb.square_root(*limb.kipping_square_root_cd(q1, q2))
    assert_close(
        sky.visibility(limb.disk(12.0, profile, -1.1, 0.3), u, v, wl),
        virgil_vis(vm.SquareRootLimbDarkenedDisk(12.0, q1, q2, dra=-1.1, ddec=0.3), u, v, wl),
    )


@pytest.mark.x64
@pytest.mark.parametrize(
    "coeffs, powers",
    [
        ([1.0], [0.0]),
        ([0.2, 0.5, 0.3], [0.0, 1.0, 0.5]),
        ([0.4, 0.3, 0.2, 0.1], [0.0, 0.5, 1.5, 3.7]),
        ([0.5, 0.5], [0.25, 22.0]),
    ],
    ids=["uniform", "sqrt-law", "fractional", "edge-powers"],
)
@pytest.mark.validates("virgil.models.cvis_limb_darkened_disk", roots=["mathematics"])
def test_cvis_any_powers(uvw, coeffs, powers):
    u, v, wl = uvw

    def profile(mu):
        return sum(a * np.asarray(mu) ** p for a, p in zip(coeffs, powers))

    ours = sky.visibility(limb.disk(12.0, profile, 0.4, -0.7), u, v, wl)
    # the function takes spatial frequencies (baselines over wavelength)
    theirs = np.asarray(
        vm.cvis_limb_darkened_disk(u / wl, v / wl, 12.0, np.asarray(coeffs), tuple(powers), 0.4, -0.7)
    )
    assert_close(ours, theirs)


@pytest.mark.x64
@pytest.mark.validates("virgil.models.cvis_limb_darkened_disk", roots=["mathematics"], kind="guard")
def test_cvis_refuses_undocumented_powers():
    u = v = np.linspace(0, 1e8, 5)
    for powers in [(0.0, 22.5), (-2.0,), (-3.0, 0.0)]:
        with pytest.raises(ValueError):
            vm.cvis_limb_darkened_disk(u, v, 3.0, np.ones(len(powers)), powers)


@pytest.mark.float32
@pytest.mark.parametrize(
    "model, profile",
    [
        pytest.param(
            lambda: vm.LimbDarkenedDisk(6.0, u=[0.35, 0.25]),
            limb.polynomial([0.35, 0.25]),
            id="quadratic",
            marks=pytest.mark.validates("virgil.models.LimbDarkenedDisk", roots=["mathematics"]),
        ),
        pytest.param(
            lambda: vm.SquareRootLimbDarkenedDisk(6.0, 0.64, 0.375),
            limb.square_root(*limb.kipping_square_root_cd(0.64, 0.375)),
            id="square-root",
            marks=pytest.mark.validates("virgil.models.SquareRootLimbDarkenedDisk", roots=["mathematics"]),
        ),
    ],
)
def test_float32(uvw, model, profile):
    u, v, wl = uvw
    assert_close(
        sky.visibility(limb.disk(6.0, profile), u, v, wl),
        virgil_vis(model(), u, v, wl),
        tol=TOL32,
    )


# ---------------------------------------------------- Kipping (2013) maps


Q_GRID = [(a, b) for a in np.linspace(0.01, 0.99, 9) for b in np.linspace(0.01, 0.99, 9)]
# the square's edges: q1 = 0 is a uniform disk (q2 then has no effect),
# q1 = 1 has zero limb intensity, q2 = 0 or 1 a flat centre or a zero slope
# at the limb; each meets one of Kipping's constraints with equality
Q_EDGES = [(a, b) for a in np.linspace(0.0, 1.0, 5) for b in np.linspace(0.0, 1.0, 5)]


@pytest.mark.x64
@pytest.mark.validates("virgil.models.QuadraticLimbDarkenedDisk", roots=["literature"])
def test_quadratic_maps_both_ways():
    for q1, q2 in Q_GRID:
        star = vm.QuadraticLimbDarkenedDisk(3.0, q1, q2)
        u1, u2 = limb.kipping_quadratic_u(q1, q2)
        assert np.isclose(float(star.u1), u1, rtol=0, atol=1e-14)
        assert np.isclose(float(star.u2), u2, rtol=0, atol=1e-14)
        back = vm.QuadraticLimbDarkenedDisk.from_u(3.0, u1, u2)
        assert np.allclose([float(back.q1), float(back.q2)], limb.kipping_quadratic_q(u1, u2), atol=1e-14)
        assert np.allclose([float(back.q1), float(back.q2)], [q1, q2], atol=1e-14)


@pytest.mark.x64
@pytest.mark.validates("virgil.models.SquareRootLimbDarkenedDisk", roots=["literature"])
def test_square_root_maps_both_ways():
    for q1, q2 in Q_GRID:
        star = vm.SquareRootLimbDarkenedDisk(3.0, q1, q2)
        c, d = limb.kipping_square_root_cd(q1, q2)
        assert np.isclose(float(star.c), c, rtol=0, atol=1e-14)
        assert np.isclose(float(star.d), d, rtol=0, atol=1e-14)
        back = vm.SquareRootLimbDarkenedDisk.from_cd(3.0, c, d)
        assert np.allclose([float(back.q1), float(back.q2)], limb.kipping_square_root_q(c, d), atol=1e-14)
        assert np.allclose([float(back.q1), float(back.q2)], [q1, q2], atol=1e-14)


@pytest.mark.x64
@pytest.mark.parametrize(
    "cls, coeffs, physical",
    [
        pytest.param(
            "QuadraticLimbDarkenedDisk", lambda s: (s.u1, s.u2), limb.quadratic_is_physical,
            id="quadratic",
            marks=pytest.mark.validates("virgil.models.QuadraticLimbDarkenedDisk", roots=["literature"]),
        ),
        pytest.param(
            "SquareRootLimbDarkenedDisk", lambda s: (s.c, s.d), limb.square_root_is_physical,
            id="square-root",
            marks=pytest.mark.validates("virgil.models.SquareRootLimbDarkenedDisk", roots=["literature"]),
        ),
    ],
)
def test_unit_square_is_exactly_the_physical_laws(cls, coeffs, physical):
    """Inside the square, virgil's coefficients satisfy Kipping's
    constraints strictly; on its edges (inclusive, as virgil documents),
    with equality allowed; in both cases virgil calls the model physical.
    Just outside, they break a constraint and virgil calls it unphysical."""
    model = getattr(vm, cls)
    for q1, q2 in Q_GRID:
        star = model(3.0, q1, q2)
        assert physical(*map(float, coeffs(star)))
        assert bool(star.is_physical())
    for q1, q2 in Q_EDGES:
        star = model(3.0, q1, q2)
        assert physical(*map(float, coeffs(star)), strict=False), (q1, q2)
        assert bool(star.is_physical()), (q1, q2)
    for q1, q2 in [(1.2, 0.5), (0.5, -0.05), (0.5, 1.05), (1.02, 0.02)]:
        star = model(3.0, q1, q2)
        assert not physical(*map(float, coeffs(star)))
        assert not bool(star.is_physical())


@pytest.mark.x64
@pytest.mark.parametrize(
    "u, positive",
    [([0.6], True), ([-0.3], True), ([0.35, 0.25], True), ([2.0], False), ([3.0], False), ([0.5, 0.6], False)],
)
@pytest.mark.validates("virgil.models.LimbDarkenedDisk", roots=["mathematics"])
def test_polynomial_is_physical_means_non_negative(u, positive):
    """virgil documents ``is_physical`` as false where the profile goes
    negative (and allows limb brightening)."""
    mu = np.linspace(0.0, 1.0, 10001)
    assert bool(np.all(limb.polynomial(u)(mu) >= 0.0)) == positive
    assert bool(vm.LimbDarkenedDisk(3.0, u=u).is_physical()) == positive


# -------------------------------------------------------- image plane


@pytest.mark.x64
@pytest.mark.validates("virgil.models.SourceModel.render", "virgil.models.QuadraticLimbDarkenedDisk", roots=["standards"])
def test_render_matches_our_image():
    """virgil's render is the brightness at pixel centres (East left, North
    up), normalised; our image of the same profile on the same grid."""
    npix, fov = 64, 16.0
    u1, u2 = limb.kipping_quadratic_u(0.36, 0.29)
    ours = limb.image(12.0, limb.polynomial([u1, u2]), npix, fov, dra=1.5, ddec=-0.75)
    theirs = np.asarray(vm.QuadraticLimbDarkenedDisk(12.0, 0.36, 0.29, dra=1.5, ddec=-0.75).render(npix, fov))
    err = record("max_abs_dpixel", np.max(np.abs(ours / ours.sum() - theirs)))
    assert err < 1e-12


@pytest.mark.x64
@pytest.mark.validates("virgil.models.QuadraticLimbDarkenedDisk", "virgil.models.System", roots=["mathematics"])
def test_pixel_image_through_oifits(tmp_path):
    """Item 2 of Issue #12: our image of an off-centre limb-darkened star
    (8x oversampled pixels) with a companion, written to OIFITS; virgil's
    analytic model reproduces its V^2 and closure phases to the
    pixelisation error."""
    diam, q1, q2, dra, ddec = 6.0, 0.36, 0.29, 0.7, -0.4
    comp, flux = (-14.0, 9.0), 0.03
    npix, fov = 128, 8.0
    img = limb.image(diam, limb.polynomial(limb.kipping_quadratic_u(q1, q2)), npix, fov, dra, ddec, oversample=8)
    truth = sky.mix([sky.pixel_image(img, fov / npix), sky.point(*comp)], [1.0, flux])
    path = tmp_path / "pixels.fits"
    simulate.observe(
        path, lambda u, v, w: sky.visibility(truth, u, v, w), UTS,
        hour_angles_h=HA, wavelengths=WL, dec_deg=DEC, sigma_v2=0.01, sigma_cp_deg=0.5,
    )
    data = vb.load(path)
    model = vm.System(
        star=vm.QuadraticLimbDarkenedDisk(diam, q1, q2, dra=dra, ddec=ddec),
        comp=vm.PointSource(flux, *comp),
    )
    observed, _ = data.flatten_data()
    predicted = np.asarray(data.model(model))
    n_v2 = np.asarray(data.vis).size
    dv2 = record("max_abs_dV2", np.max(np.abs(np.asarray(observed)[:n_v2] - predicted[:n_v2])))
    dcp = record("max_abs_dCP_rad", np.max(np.abs(np.asarray(observed)[n_v2:] - predicted[n_v2:])))
    assert dv2 < 1e-4 and dcp < 1e-2


# ---------------------------------------------------- OIFITS end to end


@pytest.mark.x64
@pytest.mark.parametrize("make", vb.LIMB_SCENES, ids=lambda f: f.__name__)
@pytest.mark.validates("virgil.oifits.read_oifits", "virgil.oidata.OIData", "virgil.models.QuadraticLimbDarkenedDisk", "virgil.models.System", roots=["standards", "mathematics"])
def test_file_reproduces_virgil_model(tmp_path, make):
    scene = make()
    path = tmp_path / "s.fits"
    simulate.observe(
        path, scene.vis, UTS, hour_angles_h=HA, wavelengths=WL, dec_deg=DEC,
        sigma_v2=0.01, sigma_cp_deg=0.5,
    )
    data = vb.load(path)
    observed, _ = data.flatten_data()
    model = data.model(scene.template)
    err = record("max_abs_dobs", np.max(np.abs(np.asarray(observed) - np.asarray(model))))
    assert err < 1e-10


@pytest.mark.x64
@pytest.mark.parametrize("make", vb.LIMB_SCENES, ids=lambda f: f.__name__)
@pytest.mark.validates("pipeline:synthetic-vlti-fit", "virgil.fitting.fit", "virgil.models.QuadraticLimbDarkenedDisk", roots=["mathematics"])
def test_noise_free_recovery(tmp_path, make):
    scene = make()
    path = tmp_path / "s.fits"
    simulate.observe(
        path, scene.vis, UTS, hour_angles_h=HA, wavelengths=WL, dec_deg=DEC,
        sigma_v2=0.01, sigma_cp_deg=0.5,
    )
    result, _ = vb.fit_scene(scene, vb.load(path))
    got = vb.flat_values(scene, result.values)
    truth = vb.flat_truth(scene)
    record("max_rel_err", np.max(np.abs(got - truth) / np.abs(truth)))
    np.testing.assert_allclose(got, truth, rtol=1e-6, atol=1e-8)
