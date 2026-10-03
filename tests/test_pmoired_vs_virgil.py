"""Stage 1: virgil and PMOIRED evaluate the same scenes on the same files.

PMOIRED reads one of our OIFITS files and returns its model V^2 and closure
phases with the uv coordinates it used for each sample; virgil is evaluated
at exactly those coordinates. Parameters are mapped with the conventions
pinned in Stage 0 (docs/pmoired_conventions.md).
"""

import numpy as np
import pytest

from crosscheck import nrm, simulate, array

pytest.importorskip("pmoired")
vm = pytest.importorskip("virgil.models")
from external_bridge.pmoired_models import model_samples  # noqa: E402

pytestmark = [pytest.mark.external, pytest.mark.x64]

UTS = np.array(
    [[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]]
)
FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


@pytest.fixture(scope="module")
def vlti_file(tmp_path_factory):
    """A carrier file: 4 UTs, 5 hour angles, 6 channels. Its values do not
    matter (both codes are compared at the file's coordinates). PMOIRED
    computes rings exactly only up to 30 samples per baseline (epochs x
    channels); beyond that they carry ~1e-4 errors (docs/pmoired_notes.md),
    so this file stays at 30."""
    path = tmp_path_factory.mktemp("s1") / "vlti.fits"
    simulate.observe(
        path, lambda u, v, w: np.ones(np.broadcast(u, v, w).shape, complex),
        UTS, hour_angles_h=np.linspace(-3, 3, 5),
        wavelengths=np.linspace(1.5e-6, 2.4e-6, 6), dec_deg=-50.0,
        sigma_v2=0.01, sigma_cp_deg=0.5,
    )
    return path


@pytest.fixture(scope="module")
def mask_file(tmp_path_factory):
    """A 7-hole masking snapshot at 4.8 um: 21 baselines, 35 triangles."""
    path = tmp_path_factory.mktemp("s1") / "mask.fits"
    simulate.observe(
        path, lambda u, v, w: np.ones(np.broadcast(u, v, w).shape, complex),
        nrm.HOLES, hour_angles_h=[0.0], wavelengths=[4.8e-6],
        fixed_uv=array.pupil_uv(nrm.HOLES), sigma_v2=0.01, sigma_cp_deg=0.5,
    )
    return path


def compare(path, scene, params, setup=None):
    """Largest |dV^2| and |d closure phase| (deg) between the codes, and
    the largest |closure phase| (the test's sensitivity)."""
    s = model_samples(path, params, setup)
    # u/wl in m/um: evaluate virgil at wavelength 1 um, baselines u/wl m
    def vis(u, v):
        return np.asarray(scene.model(u, v, 1e-6))

    v2 = np.abs(vis(s["u"], s["v"])) ** 2
    t3 = (
        vis(s["u1"], s["v1"])
        * vis(s["u2"], s["v2_"])
        * np.conj(vis(s["u1"] + s["u2"], s["v1"] + s["v2_"]))
    )
    dcp = (np.rad2deg(np.angle(t3)) - s["t3phi"] + 180.0) % 360.0 - 180.0
    return (
        np.max(np.abs(v2 - s["v2"])),
        np.max(np.abs(dcp)),
        np.max(np.abs(s["t3phi"])),
    )


# ------------------------------------------------- parameter mapping


def pm_point(name, flux=1.0, dra=0.0, ddec=0.0):
    return {f"{name},ud": 0.0, f"{name},f": flux, f"{name},x": dra, f"{name},y": ddec}


def pm_ud(name, diam, flux=1.0, dra=0.0, ddec=0.0):
    return {f"{name},ud": diam, f"{name},f": flux, f"{name},x": dra, f"{name},y": ddec}


def pm_gauss(name, sigma, flux=1.0, dra=0.0, ddec=0.0):
    return {f"{name},fwhm": FWHM * sigma, f"{name},f": flux, f"{name},x": dra, f"{name},y": ddec}


def pm_egauss(name, fwhm, ratio, pa, flux=1.0, dra=0.0, ddec=0.0):
    return {
        f"{name},fwhm": fwhm, f"{name},incl": np.rad2deg(np.arccos(ratio)),
        f"{name},projang": pa, f"{name},f": flux, f"{name},x": dra, f"{name},y": ddec,
    }


# ------------------------------------------------------------ scenes


def test_binary(vlti_file):
    scene = vm.BinaryModelCartesian(4.97, -3.36, 0.05)
    params = {**pm_point("A"), **pm_point("B", 0.05, 4.97, -3.36)}
    dv2, dcp, cp_max = compare(vlti_file, scene, params)
    assert cp_max > 5 and dv2 < 1e-12 and dcp < 1e-9


def test_binary_angular(vlti_file):
    sep, pa = 6.0, 124.0
    scene = vm.BinaryModelAngular(sep, pa, 0.05)
    params = {
        **pm_point("A"),
        **pm_point("B", 0.05, sep * np.sin(np.deg2rad(pa)), sep * np.cos(np.deg2rad(pa))),
    }
    dv2, dcp, _ = compare(vlti_file, scene, params)
    assert dv2 < 1e-12 and dcp < 1e-9


def test_disk_star_and_companion(vlti_file):
    scene = vm.System(
        star=vm.UniformDisk(1.8), comp=vm.PointSource(0.02, -12.0, 8.0)
    )
    params = {**pm_ud("star", 1.8), **pm_point("comp", 0.02, -12.0, 8.0)}
    dv2, dcp, _ = compare(vlti_file, scene, params)
    assert dv2 < 1e-12 and dcp < 1e-9


def test_star_and_elliptical_envelope(vlti_file):
    scene = vm.System(
        star=vm.PointSource(), env=vm.EllipticalGaussian(4.0, 0.5, 60.0, 0.4, 0.7, -0.3)
    )
    params = {**pm_point("star"), **pm_egauss("env", 4.0, 0.5, 60.0, 0.4, 0.7, -0.3)}
    dv2, dcp, cp_max = compare(vlti_file, scene, params)
    assert cp_max > 1 and dv2 < 1e-12 and dcp < 1e-9


def test_resolved_flux(vlti_file):
    scene = vm.System(star=vm.UniformDisk(2.0), halo=vm.Resolved(0.3))
    params = {**pm_ud("star", 2.0), "halo,f": 0.3}
    dv2, _, _ = compare(vlti_file, scene, params)
    assert dv2 < 1e-12


def test_masking_binary(mask_file):
    """The same mapping at masking scales: 150 mas, 7 holes."""
    scene = vm.BinaryModelCartesian(124.4, -83.9, 0.05)
    params = {**pm_point("A"), **pm_point("B", 0.05, 124.4, -83.9)}
    dv2, dcp, cp_max = compare(mask_file, scene, params)
    assert cp_max > 5 and dv2 < 1e-12 and dcp < 1e-9


def _random_scene(seed):
    rng = np.random.default_rng(seed)
    parts, params = {}, {}
    for k in range(24):
        kind = rng.choice(["point", "gauss", "egauss", "disk"])
        dra, ddec = rng.uniform(-20, 20, 2)
        f = rng.uniform(0.05, 2.0)
        name = f"c{k}"
        if kind == "point":
            parts[name] = vm.PointSource(f, dra, ddec)
            params |= pm_point(name, f, dra, ddec)
        elif kind == "gauss":
            s = rng.uniform(0.2, 3)
            parts[name] = vm.GaussianDisk(s, f, dra, ddec)
            params |= pm_gauss(name, s, f, dra, ddec)
        elif kind == "egauss":
            fw, r, pa = rng.uniform(0.5, 6), rng.uniform(0.1, 1), rng.uniform(0, 360)
            parts[name] = vm.EllipticalGaussian(fw, r, pa, f, dra, ddec)
            params |= pm_egauss(name, fw, r, pa, f, dra, ddec)
        else:
            d = rng.uniform(0.3, 5)
            parts[name] = vm.UniformDisk(d, f, dra, ddec)
            params |= pm_ud(name, d, f, dra, ddec)
    return vm.System(**parts), params


@pytest.mark.parametrize("seed", range(3))
def test_random_constellation(vlti_file, seed):
    """24 points, Gaussians, elliptical Gaussians and disks at random
    positions and fluxes: one dictionary of 24 PMOIRED components."""
    scene, params = _random_scene(seed)
    dv2, dcp, cp_max = compare(vlti_file, scene, params)
    assert cp_max > 30 and dv2 < 1e-12 and dcp < 1e-8


# --------------------------------------------------------------- rim

RIM = dict(diam=6.0, fwhm=1.0, inc=45.0, pa=30.0)
RIM_AMPS, RIM_PAS = np.array([0.5, 0.2]), np.array([120.0, 75.0])


def _annulus_params(width):
    """PMOIRED's nearest shape to virgil's rim: an annulus of fractional
    width ``width`` about the rim's diameter, blurred by the spatial kernel
    (which blurs every component, so the rim is alone), modulation angles
    mapped from virgil's absolute to PMOIRED's relative ones."""
    d = RIM["diam"]
    return {
        "diamin": d * (1 - width / 2), "diamout": d * (1 + width / 2),
        "incl": RIM["inc"], "projang": RIM["pa"], "spatial kernel": RIM["fwhm"],
        "az amp1": RIM_AMPS[0], "az projang1": RIM_PAS[0] - RIM["pa"],
        "az amp2": RIM_AMPS[1], "az projang2": RIM_PAS[1] - RIM["pa"],
    }


def test_rim_is_the_thin_annulus_limit(vlti_file):
    """virgil's ModulatedGaussianRim (a thin ring convolved with a
    Gaussian) is the zero-width limit of PMOIRED's blurred annulus: the
    difference falls as width^2, and Richardson extrapolation of PMOIRED
    to zero width lands on virgil."""
    scene = vm.ModulatedGaussianRim(
        RIM["diam"], RIM["fwhm"], RIM["inc"], RIM["pa"], RIM_AMPS, RIM_PAS
    )
    setup = {"Nr": 2000}
    w = 0.05
    s1 = model_samples(vlti_file, _annulus_params(w), setup)
    s2 = model_samples(vlti_file, _annulus_params(w / 2), setup)
    v2 = np.abs(np.asarray(scene.model(s1["u"], s1["v"], 1e-6))) ** 2
    d1, d2 = np.max(np.abs(s1["v2"] - v2)), np.max(np.abs(s2["v2"] - v2))
    assert 3.8 < d1 / d2 < 4.2  # second order in the width
    extrapolated = (4 * s2["v2"] - s1["v2"]) / 3
    assert np.max(np.abs(extrapolated - v2)) < 1e-6  # from 1.2e-4 at w / 2
    t1, t2 = s1["t3phi"], s2["t3phi"]
    extrapolated_cp = (4 * t2 - t1) / 3
    dv, dcp, cp_max = compare(vlti_file, scene, _annulus_params(w / 2), setup)
    assert cp_max > 30
    # virgil's closure phases from the same samples
    def vis(u, v):
        return np.asarray(scene.model(u, v, 1e-6))
    t3 = vis(s1["u1"], s1["v1"]) * vis(s1["u2"], s1["v2_"]) * np.conj(
        vis(s1["u1"] + s1["u2"], s1["v1"] + s1["v2_"])
    )
    cp = np.rad2deg(np.angle(t3))
    err_cp = np.max(np.abs((extrapolated_cp - cp + 180) % 360 - 180))
    assert err_cp < 1e-2 * dcp, (err_cp, dcp)
