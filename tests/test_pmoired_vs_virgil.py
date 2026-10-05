"""Stage 1: virgil and PMOIRED evaluate the same scenes on the same files.

PMOIRED reads one of our OIFITS files and returns its model V^2 and closure
phases with the uv coordinates it used for each sample; virgil is evaluated
at exactly those coordinates. Parameters are mapped with the conventions
pinned in Stage 0 (docs/method/pmoired.md).
"""

import numpy as np
import pytest

from crosscheck import array, nrm, simulate, sky

pytest.importorskip("pmoired")
vm = pytest.importorskip("virgil.models")
from external_bridge.pmoired_models import model_samples  # noqa: E402
from evidence.plugin import record  # noqa: E402

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
    channels); beyond that they carry ~1e-4 errors (docs/method/pmoired.md),
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
        record("max_abs_dV2", np.max(np.abs(v2 - s["v2"]))),
        record("max_abs_dCP_deg", np.max(np.abs(dcp))),
        record("max_abs_CP_deg", np.max(np.abs(s["t3phi"]))),
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


@pytest.mark.validates("virgil.models.BinaryModelCartesian", roots=["pmoired"])
def test_binary(vlti_file):
    scene = vm.BinaryModelCartesian(4.97, -3.36, 0.05)
    params = {**pm_point("A"), **pm_point("B", 0.05, 4.97, -3.36)}
    dv2, dcp, cp_max = compare(vlti_file, scene, params)
    assert cp_max > 5 and dv2 < 1e-12 and dcp < 1e-9


@pytest.mark.validates("virgil.models.BinaryModelAngular", roots=["pmoired"])
def test_binary_angular(vlti_file):
    sep, pa = 6.0, 124.0
    scene = vm.BinaryModelAngular(sep, pa, 0.05)
    params = {
        **pm_point("A"),
        **pm_point("B", 0.05, sep * np.sin(np.deg2rad(pa)), sep * np.cos(np.deg2rad(pa))),
    }
    dv2, dcp, _ = compare(vlti_file, scene, params)
    assert dv2 < 1e-12 and dcp < 1e-9


@pytest.mark.validates("virgil.models.Rotated", roots=["pmoired"], property="rotation_sense")
@pytest.mark.parametrize("angle", [35.0, 250.0])
def test_rotation_is_by_position_angle_in_pmoired(vlti_file, angle):
    """Rotated by an angle, a scene with a companion at position angle 20
    degrees matches PMOIRED's scene with the companion at 20 + angle (x
    East, y North): rotation runs North through East."""
    sep, pa = 7.0, 20.0
    scene = vm.Rotated(vm.System(star=vm.UniformDisk(1.8), comp=vm.PointSource(0.05, *_east_north(sep, pa))), angle)
    params = {**pm_ud("star", 1.8), **pm_point("comp", 0.05, *_east_north(sep, pa + angle))}
    dv2, dcp, cp_max = compare(vlti_file, scene, params)
    assert cp_max > 1 and dv2 < 1e-10 and dcp < 1e-6
    wrong = {**pm_ud("star", 1.8), **pm_point("comp", 0.05, *_east_north(sep, pa - angle))}
    assert compare(vlti_file, scene, wrong)[1] > 100 * max(dcp, 1e-8)  # the other sense: a control


def _east_north(sep, pa):
    return sep * np.sin(np.deg2rad(pa)), sep * np.cos(np.deg2rad(pa))


@pytest.mark.validates("virgil.models.UniformDisk", "virgil.models.PointSource", "virgil.models.System", roots=["pmoired"])
def test_disk_star_and_companion(vlti_file):
    scene = vm.System(
        star=vm.UniformDisk(1.8), comp=vm.PointSource(0.02, -12.0, 8.0)
    )
    params = {**pm_ud("star", 1.8), **pm_point("comp", 0.02, -12.0, 8.0)}
    dv2, dcp, _ = compare(vlti_file, scene, params)
    assert dv2 < 1e-12 and dcp < 1e-9


@pytest.mark.validates("virgil.models.EllipticalGaussian", "virgil.models.System", roots=["pmoired"])
def test_star_and_elliptical_envelope(vlti_file):
    scene = vm.System(
        star=vm.PointSource(), env=vm.EllipticalGaussian(4.0, 0.5, 60.0, 0.4, 0.7, -0.3)
    )
    params = {**pm_point("star"), **pm_egauss("env", 4.0, 0.5, 60.0, 0.4, 0.7, -0.3)}
    dv2, dcp, cp_max = compare(vlti_file, scene, params)
    assert cp_max > 1 and dv2 < 1e-12 and dcp < 1e-9


@pytest.mark.validates("virgil.models.Resolved", roots=["pmoired"])
def test_resolved_flux(vlti_file):
    scene = vm.System(star=vm.UniformDisk(2.0), halo=vm.Resolved(0.3))
    params = {**pm_ud("star", 2.0), "halo,f": 0.3}
    dv2, _, _ = compare(vlti_file, scene, params)
    assert dv2 < 1e-12


@pytest.mark.validates("virgil.models.BinaryModelCartesian", roots=["pmoired"])
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
@pytest.mark.validates("virgil.models.System", "virgil.models.PointSource", "virgil.models.GaussianDisk", "virgil.models.EllipticalGaussian", "virgil.models.UniformDisk", roots=["pmoired"])
def test_random_constellation(vlti_file, seed):
    """24 points, Gaussians, elliptical Gaussians and disks at random
    positions and fluxes: one dictionary of 24 PMOIRED components."""
    scene, params = _random_scene(seed)
    dv2, dcp, cp_max = compare(vlti_file, scene, params)
    assert cp_max > 30 and dv2 < 1e-12 and dcp < 1e-8


# --------------------------------------------------------------- rim

RIM = dict(diam=6.0, fwhm=1.0, inc=45.0, pa=30.0)
SIGMA = RIM["fwhm"] / FWHM


def _blurred_ring_profile(r):
    """Radial profile of a thin ring of radius r0 convolved with a 2D
    Gaussian of width sigma: exp(-(r^2 + r0^2) / 2 sigma^2) I0(r r0 / sigma^2)."""
    from scipy.special import i0

    r0 = RIM["diam"] / 2
    return np.exp(-(r**2 + r0**2) / (2 * SIGMA**2)) * i0(r * r0 / SIGMA**2)


def _pmoired_rim(amps=(), pas=()):
    """PMOIRED ring with the blurred-ring profile, defined in the disk plane
    and inclined (so the blur is isotropic in that plane, as virgil's since
    virgil#139); modulation angles relative to projang."""
    r0, s = RIM["diam"] / 2, SIGMA
    params = {
        "diamin": max(RIM["diam"] - 12 * s, 0.0),
        "diamout": RIM["diam"] + 12 * s,
        "profile": f"np.exp(-($R**2+{r0}**2)/(2*{s}**2))*np.i0($R*{r0}/{s}**2)",
        "incl": RIM["inc"],
        "projang": RIM["pa"],
    }
    for k, (a, p) in enumerate(zip(amps, pas), start=1):
        params[f"az amp{k}"] = a
        params[f"az projang{k}"] = p - RIM["pa"]
    return params


@pytest.mark.validates("virgil.models.ModulatedGaussianRim", roots=["pmoired"], property="profile")
def test_unmodulated_rim_matches_pmoired_blurred_ring_profile(vlti_file):
    """An unmodulated, inclined rim blurred in its own plane is a ring with
    the Bessel-I0 radial profile: PMOIRED and virgil agree exactly. This
    checks virgil#139's in-plane blur against another package."""
    scene = vm.ModulatedGaussianRim(
        RIM["diam"], RIM["fwhm"], RIM["inc"], RIM["pa"]
    )
    dv2, dcp, _ = compare(vlti_file, scene, _pmoired_rim(), {"Nr": 3000})
    assert dv2 < 1e-8 and dcp < 1e-6


class _Cloud:
    def __init__(self, cloud):
        self.cloud = cloud

    def model(self, u, v, wl):
        return sky.visibility(self.cloud, u, v, wl)


@pytest.mark.validates("virgil.models.ModulatedGaussianRim", roots=["pmoired", "mathematics"], kind="control")
def test_modulated_rim_definitions_differ(vlti_file):
    """With modulation the two packages define different things, both
    exactly. PMOIRED multiplies the radial profile by (1 + A cos m phi);
    virgil blurs the modulated thin ring, which gives harmonic m the
    profile I_m instead of I_0. PMOIRED matches our quadrature of its
    definition; virgil matches ours of its own (tests/test_visibilities.py);
    the two differ at O(m^2 sigma^2 / r0^2). Ruled a difference of
    definition (docs/method/pmoired.md)."""
    amps, pas = (0.5, 0.2), (120.0, 75.0)
    r0, s = RIM["diam"] / 2, SIGMA
    separable = sky.inclined_annulus(
        max(RIM["diam"] - 12 * s, 0.0), RIM["diam"] + 12 * s,
        RIM["inc"], RIM["pa"], amps, pas, "disk",
        n_r=96, n_phi=1024, profile=_blurred_ring_profile,
    )
    params = _pmoired_rim(amps, pas)
    dv2, dcp, cp_max = compare(vlti_file, _Cloud(separable), params, {"Nr": 3000})
    assert cp_max > 30 and dv2 < 1e-8 and dcp < 1e-5
    scene = vm.ModulatedGaussianRim(
        RIM["diam"], RIM["fwhm"], RIM["inc"], RIM["pa"],
        np.array(amps), np.array(pas),
    )
    dv2, dcp, _ = compare(vlti_file, scene, params, {"Nr": 3000})
    assert 1e-4 < dv2 < 1e-2  # (m sigma / r0)^2 ~ 0.02 times the modulated part


@pytest.mark.validates("virgil.models.ModulatedGaussianRim", roots=["pmoired"], property="modulation_sense")
def test_modulation_sense_agrees_with_pmoired(vlti_file):
    """The definitions differ only at O(m^2 sigma^2 / r0^2) (D-code above),
    so the sense of the modulation can still be checked against PMOIRED:
    virgil's rim at the same amplitudes and angles is 10 times closer to
    PMOIRED's than with the bright side turned by 180 degrees, or reflected
    about the rim's major axis (the other sense of azimuth)."""
    amps, pas = (0.5, 0.2), (120.0, 75.0)
    params = _pmoired_rim(amps, pas)

    def rim(angles):
        return vm.ModulatedGaussianRim(RIM["diam"], RIM["fwhm"], RIM["inc"], RIM["pa"], np.array(amps), np.array(angles))

    _, dcp, cp_max = compare(vlti_file, rim(pas), params, {"Nr": 3000})
    _, turned, _ = compare(vlti_file, rim([p + 180.0 for p in pas]), params, {"Nr": 3000})
    _, reflected, _ = compare(vlti_file, rim([2 * RIM["pa"] - p for p in pas]), params, {"Nr": 3000})
    record("dcp_ratio_turned", turned / dcp)
    record("dcp_ratio_reflected", reflected / dcp)
    assert cp_max > 30
    assert turned > 10 * dcp and reflected > 10 * dcp


# ------------------------------------------------------ limb darkening


def pm_profile_disk(name, diam, profile, flux=1.0, dra=0.0, ddec=0.0):
    """A disk with brightness ``profile`` (a PMOIRED expression in $MU) is a
    ring from 0 to ``diam`` whose $MU = sqrt(1 - (2r / diamout)^2)."""
    return {
        f"{name},diamin": 0.0, f"{name},diamout": diam, f"{name},profile": profile,
        f"{name},f": flux, f"{name},x": dra, f"{name},y": ddec,
    }


# measured at Nr = 10000: quadratic 3e-7 in V^2 and 6e-4 deg in closure
# phase, square-root 2.5e-6 and 4.5e-3 deg
LIMB_TOL = {"quadratic": (1e-6, 2e-3), "square-root": (1e-5, 2e-2)}


@pytest.mark.parametrize(
    "law",
    [
        pytest.param(
            "quadratic",
            marks=pytest.mark.validates("virgil.models.QuadraticLimbDarkenedDisk", "virgil.models.System", roots=["pmoired"]),
        ),
        pytest.param(
            "square-root",
            marks=pytest.mark.validates("virgil.models.SquareRootLimbDarkenedDisk", "virgil.models.System", roots=["pmoired"]),
        ),
    ],
)
def test_limb_darkened_star_and_companion(vlti_file, law):
    """PMOIRED integrates a sampled radial profile; it converges on virgil's
    analytic visibility as the sampling is refined (|dV^2| ~ Nr^-1.5 for the
    quadratic law, a little slower for the square-root law, whose
    derivative is singular at the limb). The diameter (``diamout`` = the
    limb-darkened diameter) and mu agree. A companion makes the closure
    phases informative."""
    diam, dra, ddec = 6.0, 0.7, -0.4
    if law == "quadratic":
        star = vm.QuadraticLimbDarkenedDisk(diam, 0.36, 0.29, dra=dra, ddec=ddec)
        u1, u2 = float(star.u1), float(star.u2)
        profile = f"1 - {u1!r}*(1-$MU) - {u2!r}*(1-$MU)**2"
    else:
        star = vm.SquareRootLimbDarkenedDisk.from_cd(diam, 0.1, 0.6, dra=dra, ddec=ddec)
        profile = "1 - 0.1*(1-$MU) - 0.6*(1-np.sqrt($MU))"
    scene = vm.System(star=star, comp=vm.PointSource(0.03, -14.0, 9.0))
    params = {
        **pm_profile_disk("star", diam, profile, dra=dra, ddec=ddec),
        **pm_point("comp", 0.03, -14.0, 9.0),
    }
    coarse, _, _ = compare(vlti_file, scene, params, {"Nr": 1000})
    dv2, dcp, cp = compare(vlti_file, scene, params, {"Nr": 10000})
    record("convergence_1000_to_10000", coarse / dv2)
    assert coarse / dv2 > 10  # the difference is PMOIRED's sampling
    tol_v2, tol_cp = LIMB_TOL[law]
    assert dv2 < tol_v2 and dcp < tol_cp and cp > 1.0
