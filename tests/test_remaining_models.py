"""The remaining models: spectra, flared disks and the harmonix wrapper.

* Spectra (PowerLaw, BlackBody, Tabulated): their documented closed forms
  (Planck with SciPy constants), and chromatic Systems as flux-weighted
  means channel by channel.
* Flared disks (Blakely et al. 2024): crosscheck.disks evaluates the
  documented brightness at the documented pixel centres; its direct Fourier
  sum is the reference. Separately, refining the grid shows virgil's
  default sampling converges.
* HarmonixModel: stand-in sources with exactly known visibilities check
  the wrapper's units, normalisation and weight inside a System.
"""

import numpy as np
import pytest
from scipy import constants

from crosscheck import disks, sky
from evidence.plugin import record

vm = pytest.importorskip("virgil.models")
spectra = pytest.importorskip("virgil.spectra")

pytestmark = pytest.mark.x64
RNG = np.random.default_rng(12)
U, V = RNG.uniform(-100, 100, (2, 300))
WLS = np.linspace(1.5e-6, 2.5e-6, 7)


def planck(wl, t):
    h, c, k = constants.h, constants.c, constants.k
    return 2 * h * c**2 / wl**5 / np.expm1(h * c / (wl * k * t))


# ------------------------------------------------------------------ spectra


@pytest.mark.validates("virgil.spectra.PowerLaw", roots=["mathematics"])
@pytest.mark.parametrize("index", [0.0, -4.0, 1.7])
def test_power_law(index):
    s = spectra.PowerLaw(0.2, index=index, wavel0=1.8e-6)
    got = np.array([np.ravel(s(w))[0] for w in WLS])
    np.testing.assert_allclose(got, 0.2 * (WLS / 1.8e-6) ** index, rtol=1e-12)


@pytest.mark.validates("virgil.spectra.BlackBody", roots=["mathematics"])
@pytest.mark.parametrize("temperature", [300.0, 1500.0, 9000.0])
def test_black_body_is_a_planck_ratio(temperature):
    s = spectra.BlackBody(0.3, temperature, wavel0=1.65e-6)
    got = np.array([np.ravel(s(w))[0] for w in WLS])
    want = 0.3 * planck(WLS, temperature) / planck(1.65e-6, temperature)
    # virgil carries hc/k to ~10 digits; the ratio's error grows with the
    # wavelength range (max ~1e-8 here), so it is not a finding
    np.testing.assert_allclose(got, want, rtol=1e-7)


@pytest.mark.validates("virgil.spectra.BlackBody", roots=["mathematics"])
def test_black_body_tends_to_rayleigh_jeans():
    s = spectra.BlackBody(1.0, 1e6, wavel0=1e-4)  # hc / lambda k T ~ 1e-4
    w = np.array([2e-4, 5e-4])
    np.testing.assert_allclose(np.array([np.ravel(s(x))[0] for x in w]), (w / 1e-4) ** -4.0, rtol=1e-3)


@pytest.mark.validates("virgil.spectra.Tabulated", roots=["mathematics"])
def test_tabulated_interpolates_linearly_and_holds_its_ends():
    nodes = np.array([2.0e-6, 2.2e-6, 2.5e-6])
    ratio = np.array([0.2, 0.4, 0.1])
    s = spectra.Tabulated(ratio, nodes)
    w = np.array([1.5e-6, 2.0e-6, 2.1e-6, 2.35e-6, 2.5e-6, 3e-6])
    want = np.interp(w, nodes, ratio)  # numpy holds the end values too
    np.testing.assert_allclose(np.array([np.ravel(s(x))[0] for x in w]), want, rtol=1e-12)
    assert np.isclose(float(s(None)), ratio.mean())


@pytest.mark.validates("virgil.spectra.Tabulated", roots=["mathematics"])
def test_tabulated_reference_flux_is_the_documented_mean():
    """Tabulated documents its reference flux as the mean over the nodes.
    Finding F9: reference_flux returned the whole ratio array (fixed in
    virgil#163)."""
    s = spectra.Tabulated(np.array([0.2, 0.4, 0.1]), np.array([2.0e-6, 2.2e-6, 2.5e-6]))
    assert np.ndim(spectra.reference_flux(s)) == 0
    assert np.isclose(float(spectra.reference_flux(s)), float(s(None)))


@pytest.mark.validates("virgil.models.System", "virgil.spectra.BlackBody", "virgil.spectra.PowerLaw", roots=["mathematics"])
def test_a_chromatic_system_is_a_flux_weighted_mean_per_channel():
    """A star (flux 1) with a 1500 K companion and a power-law disk: in
    each channel the visibility is the mean weighted by that channel's
    fluxes."""
    comp = spectra.BlackBody(0.05, 1500.0, wavel0=1.65e-6)
    env = spectra.PowerLaw(0.3, index=-1.5, wavel0=1.65e-6)
    scene = vm.System(
        star=vm.PointSource(),
        comp=vm.PointSource(comp, dra=5.0, ddec=-3.0),
        env=vm.GaussianDisk(2.0, env),
    )
    u, v = np.repeat(U[:40], WLS.size), np.repeat(V[:40], WLS.size)
    w = np.tile(WLS, 40)
    got = np.asarray(scene.model(u, v, w))
    fc = 0.05 * planck(w, 1500.0) / planck(1.65e-6, 1500.0)
    fe = 0.3 * (w / 1.65e-6) ** -1.5
    want = (sky.vis_point(u, v, w) + fc * sky.vis_point(u, v, w, 5.0, -3.0) + fe * sky.vis_gaussian(u, v, w, 2.0)) / (1 + fc + fe)
    record("max_abs_dV", np.max(np.abs(got - want)))
    # 5e-11 from the black body's hc/k (see above)
    np.testing.assert_allclose(got, want, rtol=0, atol=2e-10)


# ------------------------------------------------------------- flared disks

GEOMETRY = dict(radius=40.0, fwhm=20.0, inc=50.0, pa=30.0, skew=1.5, aspect=0.1, flaring=1.25, symmetric=0.3)
PHASES = {
    "FlaredDiskHG": ("g", 0.4, disks.phase_hg),
    "FlaredDiskGaussian": ("sigma_theta", 40.0, disks.phase_gaussian),
    "FlaredDiskPowerLaw": ("n", 3.0, disks.phase_power),
}


def _disk_reference(geometry, phase, npix, scale, u, v, w, dra=0.0, ddec=0.0):
    east, north = disks.pixel_centres(npix, scale)
    b = disks.brightness(east, north, phase=phase, **geometry)
    cloud = sky.Cloud(east + dra, north + ddec, b / b.sum())
    return sky.visibility(cloud, u, v, w)


@pytest.mark.parametrize(
    "name",
    [pytest.param(n, marks=pytest.mark.validates(f"virgil.models.{n}", roots=["mathematics"])) for n in PHASES],
)
@pytest.mark.parametrize("geometry", [GEOMETRY, {**GEOMETRY, "inc": 0.0, "aspect": 0.0, "skew": 0.0, "symmetric": 0.0}, {**GEOMETRY, "inc": 70.0, "pa": 200.0}])
def test_flared_disk_is_the_documented_brightness(name, geometry):
    key, value, fn = PHASES[name]
    npix, scale = 64, 2.5
    model = getattr(vm, name)(**{key: value}, **geometry, npix=npix, pixel_scale_mas=scale, dra=1.5, ddec=-2.0)
    got = np.asarray(model.model(U, V, 2.0e-6))
    want = _disk_reference(geometry, lambda t: fn(t, value), npix, scale, U, V, 2.0e-6, 1.5, -2.0)
    record("max_abs_dV", np.max(np.abs(got - want)))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-13)


@pytest.mark.validates("virgil.models.FlaredDiskHG", roots=["mathematics"], kind="control")
def test_flared_disk_near_side_is_at_pa_plus_90():
    """The forward-scattering peak is on the near side, at pa + 90: the
    pa + 270 reading is a different model."""
    npix, scale = 64, 2.5
    model = vm.FlaredDiskHG(g=0.6, **GEOMETRY, npix=npix, pixel_scale_mas=scale)
    got = np.asarray(model.model(U, V, 2.0e-6))
    flipped = {**GEOMETRY, "pa": GEOMETRY["pa"] + 180.0}
    wrong = _disk_reference(flipped, lambda t: disks.phase_hg(t, 0.6), npix, scale, U, V, 2.0e-6)
    assert np.max(np.abs(got - wrong)) > 1e-2


@pytest.mark.validates("virgil.models.FlaredDiskPowerLaw", roots=["mathematics"])
def test_flared_disk_sampling_converges():
    """The documented sampling is a numerical choice: on baselines that
    resolve the ring, doubling the pixel density changes the visibilities
    by much less than the defaults' own size."""
    u, v = U / 3, V / 3  # baselines up to ~33 m: the ring is resolved
    coarse = vm.FlaredDiskPowerLaw(n=3.0, **GEOMETRY, npix=64, pixel_scale_mas=2.5).model(u, v, 2e-6)
    fine = vm.FlaredDiskPowerLaw(n=3.0, **GEOMETRY, npix=128, pixel_scale_mas=1.25).model(u, v, 2e-6)
    d = np.max(np.abs(np.asarray(coarse) - np.asarray(fine)))
    record("max_abs_dV_refinement", d)
    assert d < 1e-8


# ----------------------------------------------------------------- harmonix


class _UniformDiskSource:
    """A stand-in harmonix source: a uniform disk, offset, whose
    model(u/λ, v/λ) takes spatial frequencies in cycles per radian."""

    def __init__(self, diam, dra, ddec, wavelength_units=True, wavel=None):
        self.diam, self.dra, self.ddec = diam, dra, ddec
        self.wavelength_units, self.wavel = wavelength_units, wavel

    def model(self, u, v, *time):
        u, v = np.asarray(u, float), np.asarray(v, float)
        if not self.wavelength_units:  # metres: divide by our own wavelength
            u, v = u / self.wavel, v / self.wavel
        return sky.vis_uniform_disk(u, v, 1.0, self.diam, self.dra, self.ddec)


@pytest.mark.validates("virgil.models.HarmonixModel", roots=["mathematics"])
def test_harmonix_wrapper_passes_spatial_frequencies():
    src = _UniformDiskSource(2.0, 1.0, -0.5)
    got = np.asarray(vm.HarmonixModel(src).model(U, V, 2.0e-6))
    np.testing.assert_allclose(got, sky.vis_uniform_disk(U, V, 2.0e-6, 2.0, 1.0, -0.5), rtol=0, atol=1e-12)


@pytest.mark.validates("virgil.models.HarmonixModel", roots=["mathematics"])
def test_harmonix_wrapper_can_pass_metres():
    src = _UniformDiskSource(2.0, 1.0, -0.5, wavelength_units=False, wavel=2.0e-6)
    got = np.asarray(vm.HarmonixModel(src, expects_wavelength_units=False).model(U, V, 2.0e-6))
    np.testing.assert_allclose(got, sky.vis_uniform_disk(U, V, 2.0e-6, 2.0, 1.0, -0.5), rtol=0, atol=1e-12)


@pytest.mark.validates("virgil.models.HarmonixModel", "virgil.models.System", roots=["mathematics"])
def test_harmonix_source_has_weight_one_in_a_system():
    src = _UniformDiskSource(2.0, 0.0, 0.0)
    scene = vm.System(star=vm.HarmonixModel(src), comp=vm.PointSource(0.1, 6.0, 2.0))
    got = np.asarray(scene.model(U, V, 2.0e-6))
    want = (sky.vis_uniform_disk(U, V, 2.0e-6, 2.0) + 0.1 * sky.vis_point(U, V, 2.0e-6, 6.0, 2.0)) / 1.1
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)
