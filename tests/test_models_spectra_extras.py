"""Public model and spectrum helpers without a node until now, against
closed forms and our own quadrature (crosscheck.sky, crosscheck.limb).

* cvis_uniform_disk: the Airy pattern 2 J1(x) / x, x = π θ |u|, times the
  shift phase exp(-2πi (u dra + v ddec)), u and v in wavelengths.
* cvis_binary: (1 + f exp(-2πi (u dra + v ddec))) / (1 + f), the primary
  at the origin, normalized at zero baseline.
* EllipticalLimbDarkenedDisk: a circular limb-darkened point cloud
  (crosscheck.limb.disk), squashed by ``ratio`` along the minor axis in the
  image plane, its major axis at ``pa`` North through East as an
  EllipticalGaussian's: no Fourier similarity theorem on our side.
* FlaredDisk: the shared base of the three flared disks has no phase
  function, so evaluating it raises (its brightness is checked through each of them in test_remaining_models).
* Spectrum, flux_at, reference_flux: numbers pass through; spectra give
  their documented value at a wavelength and at wavel0 (None); a user
  subclass that implements only ``_at`` works as a component's flux.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import j1

from crosscheck import limb, sky
from evidence.plugin import record

vm = pytest.importorskip("virgil.models")
from virgil import spectra  # noqa: E402

pytestmark = pytest.mark.x64


@pytest.fixture(scope="module")
def uvw():
    rng = np.random.default_rng(77)
    n = 200
    return rng.uniform(-130, 130, n), rng.uniform(-130, 130, n), rng.uniform(1.5e-6, 2.4e-6, n)


def airy(x):
    safe = np.where(x > 0, x, 1.0)
    return np.where(x > 0, 2 * j1(safe) / safe, 1.0)


@pytest.mark.validates("virgil.models.cvis_uniform_disk", roots=["mathematics"])
def test_cvis_uniform_disk_is_the_airy_pattern(uvw):
    u, v, wl = uvw
    uu, vv = u / wl, v / wl  # wavelengths
    diam, dra, ddec = 2.7, 0.8, -1.3
    want = airy(np.pi * diam * sky.MAS * np.hypot(uu, vv)) * np.exp(-2j * np.pi * (uu * dra + vv * ddec) * sky.MAS)
    got = np.asarray(vm.cvis_uniform_disk(uu, vv, diam, dra, ddec))
    err = record("max_abs_dV", np.max(np.abs(got - want)))
    assert err < 1e-12
    # the first null is at x = 3.8317 (the first zero of J1)
    q0 = 3.8317059702075125 / (np.pi * diam * sky.MAS)
    assert abs(complex(vm.cvis_uniform_disk(q0, 0.0, diam))) < 1e-12


@pytest.mark.validates("virgil.models.cvis_binary", roots=["mathematics"])
def test_cvis_binary_is_the_two_point_closed_form(uvw):
    u, v, wl = uvw
    uu, vv = u / wl, v / wl
    dra, ddec, f = 12.0, -7.5, 0.13
    want = (1 + f * np.exp(-2j * np.pi * (uu * dra + vv * ddec) * sky.MAS)) / (1 + f)
    got = np.asarray(vm.cvis_binary(uu, vv, dra, ddec, f))
    err = record("max_abs_dV", np.max(np.abs(got - want)))
    assert err < 1e-12
    # and the same as our point-cloud binary through crosscheck.sky
    cloud = sky.mix([sky.point(), sky.point(dra, ddec)], [1.0, f])
    assert np.max(np.abs(got - sky.visibility(cloud, u, v, wl))) < 1e-12


def squashed(cloud, ratio, pa, dra, ddec):
    """Compress a centred cloud by ``ratio`` across the axis at ``pa``."""
    p = np.deg2rad(pa)
    along = cloud.east * np.sin(p) + cloud.north * np.cos(p)
    across = (cloud.east * np.cos(p) - cloud.north * np.sin(p)) * ratio
    return sky.Cloud(along * np.sin(p) + across * np.cos(p) + dra,
                     along * np.cos(p) - across * np.sin(p) + ddec, cloud.weight)


@pytest.mark.parametrize("u_ld", [(), (0.5,), (0.6, 0.15)], ids=["uniform", "linear", "quadratic"])
@pytest.mark.validates("virgil.models.EllipticalLimbDarkenedDisk", roots=["mathematics"])
def test_elliptical_limb_darkened_disk_is_a_squashed_disk(uvw, u_ld):
    u, v, wl = uvw
    diam, ratio, pa, dra, ddec = 3.0, 0.7, 35.0, 0.4, -0.2
    profile = limb.polynomial(u_ld) if u_ld else (lambda mu: np.ones_like(mu))
    cloud = squashed(limb.disk(diam, profile), ratio, pa, dra, ddec)
    want = sky.visibility(cloud, u, v, wl)
    got = np.asarray(vm.EllipticalLimbDarkenedDisk(diam, ratio=ratio, pa=pa, u=u_ld, dra=dra, ddec=ddec).model(u, v, wl))
    err = record("max_abs_dV", np.max(np.abs(got - want)))
    assert err < 1e-10
    # control: the major axis at pa + 90 is a different star
    wrong = sky.visibility(squashed(limb.disk(diam, profile), ratio, pa + 90, dra, ddec), u, v, wl)
    assert np.max(np.abs(got - wrong)) > 1e-2


@pytest.mark.validates("virgil.models.EllipticalLimbDarkenedDisk", roots=["mathematics"])
def test_elliptical_limb_darkened_disk_axis_matches_elliptical_gaussian(uvw):
    """Documented: the major axis is oriented exactly as an
    EllipticalGaussian's. Along the minor axis both fall off faster."""
    u, v, wl = uvw
    pa = 120.0
    p = np.deg2rad(pa)
    q = np.array([60.0, 90.0])
    along_u, along_v = q * np.sin(p), q * np.cos(p)
    across_u, across_v = q * np.cos(p), -q * np.sin(p)
    disk = vm.EllipticalLimbDarkenedDisk(4.0, ratio=0.5, pa=pa)
    gauss = vm.EllipticalGaussian(3.0, ratio=0.5, pa=pa)
    w = np.full(2, 2e-6)
    vd_along = np.abs(np.asarray(disk.model(along_u, along_v, w)))
    vd_across = np.abs(np.asarray(disk.model(across_u, across_v, w)))
    # baselines along the major axis resolve the long dimension: lower visibility
    assert np.all(vd_along < vd_across)
    vg_along = np.abs(np.asarray(gauss.model(along_u, along_v, w)))
    vg_across = np.abs(np.asarray(gauss.model(across_u, across_v, w)))
    assert np.all(vg_along < vg_across)


@pytest.mark.validates("virgil.models.FlaredDisk", roots=["mathematics"], kind="guard")
def test_flared_disk_base_is_abstract():
    """It has no phase function: evaluating it raises (it can be built)."""
    base = vm.FlaredDisk(radius=20.0, fwhm=5.0, inc=40.0, pa=30.0, npix=32, pixel_scale_mas=2.0)
    with pytest.raises(NotImplementedError):
        base.model(np.array([10.0]), np.array([5.0]), 2e-6)
    for name in ("FlaredDiskHG", "FlaredDiskGaussian", "FlaredDiskPowerLaw"):
        assert issubclass(getattr(vm, name), vm.FlaredDisk)


# --------------------------------------------------------------- spectra


@pytest.mark.validates("virgil.spectra.flux_at", "virgil.spectra.reference_flux", roots=["mathematics"])
def test_flux_at_and_reference_flux():
    assert float(spectra.flux_at(0.3)) == 0.3 and float(spectra.flux_at(0.3, 2.2e-6)) == 0.3
    assert float(spectra.reference_flux(0.3)) == 0.3
    pl = spectra.PowerLaw(0.2, index=-4.0, wavel0=1.6e-6)
    wl = np.array([1.2e-6, 1.6e-6, 3.2e-6])
    np.testing.assert_allclose(np.asarray(spectra.flux_at(pl, wl)), 0.2 * (wl / 1.6e-6) ** -4.0, rtol=1e-14)
    assert float(spectra.flux_at(pl)) == pytest.approx(0.2, rel=1e-14)
    assert float(spectra.reference_flux(pl)) == pytest.approx(0.2, rel=1e-14)
    # BlackBody: the documented ratio at wavel0, whatever the temperature
    bb = spectra.BlackBody(0.05, 1500.0, wavel0=2.2e-6)
    assert float(spectra.reference_flux(bb)) == pytest.approx(0.05, rel=1e-12)


class Linear(spectra.Spectrum):
    """A user spectrum: slope * λ / wavel0, implementing only ``_at``."""

    slope: float
    wavel0: float

    def _at(self, wavel):
        return self.slope * jnp.asarray(wavel) / self.wavel0


@pytest.mark.validates("virgil.spectra.Spectrum", roots=["mathematics"])
def test_a_user_spectrum_is_a_component_flux(uvw):
    s = Linear(0.2, 2.0e-6)
    assert float(s()) == pytest.approx(0.2) and float(spectra.reference_flux(s)) == pytest.approx(0.2)
    assert float(spectra.flux_at(s, 3.0e-6)) == pytest.approx(0.3)
    u, v, wl = uvw
    model = vm.System(star=vm.PointSource(1.0), comp=vm.PointSource(s, 8.0, 3.0))
    f = 0.2 * wl / 2.0e-6
    want = (1 + f * sky.vis_point(u, v, wl, 8.0, 3.0)) / (1 + f)
    got = np.asarray(model.model(u, v, wl))
    record("max_abs_dV", np.max(np.abs(got - want)))
    assert np.max(np.abs(got - want)) < 1e-12
