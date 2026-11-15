"""Mathematical check of the near_side convention on virgil's FlaredDisk models.

In ``virgil``'s documented 3D flared-disk geometry (Blakely et al. 2025, AJ 169,
137, Eqs. 2-9), a flared scattering surface ``z(rho) = h_0 (rho / r_0)^beta``
with line-of-nodes major-axis position angle ``pa`` (degrees East of North) and
inclination ``inc`` in ``(0, 90)`` deg tilts the ``pa + 90 deg`` half-plane
towards the observer (smaller dust scattering angle). Forward-peaked dust
scattering (or positive near-side ``skew``) therefore brightens the physical
near-side rim at sky azimuth ``pa + 90 deg`` — for example, at ``(inc = 52 deg,
pa = 160 deg)``, in the West-Southwest sky quadrant (``dra < 0, ddec < 0``,
sky PA ``250 deg``).

Because no external Python package in CI implements a 3D flared scattered-light
ring and no public archival scattered-light OIFITS dataset is bundled in the
repository, these checks declare ``roots=["mathematics"]`` ONLY (using literal
per-class ``@pytest.mark.validates`` decorators so static trust-graph and marker
inventory tools inspect every node). Under ``scripts/trust.py``, ``mathematics``
alone does not confirm a ``convention`` property, keeping ``FlaredDiskGaussian``,
``FlaredDiskHG``, and ``FlaredDiskPowerLaw`` honestly at ``convention`` until
confirmed against public archival data or an external 3D radiative-transfer code.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from crosscheck import disks as cdisks
from crosscheck import sky
from evidence.plugin import record

vm = pytest.importorskip("virgil.models")

pytestmark = pytest.mark.x64

INC_DEG = 52.0
PA_DEG = 160.0
NEAR_SIDE_PA_DEG = 250.0  # pa + 90 deg: WSW quadrant (dra < 0, ddec < 0)

RING_GEOM = {
    "radius": 40.0,
    "fwhm": 15.0,
    "inc": INC_DEG,
    "pa": PA_DEG,
    "skew": 0.6,
    "aspect": 0.12,
    "flaring": 1.15,
    "symmetric": 0.15,
    "npix": 128,
    "pixel_scale_mas": 1.5,
}
WAVEL_M = 2.2e-6

# Four non-collinear baseline triangles (metres) inside the ring's first
# visibility lobe at 2.2 um, where closure phases reach 2-15 deg and their sign
# tracks the WSW vs ENE bright-rim asymmetry.
TRIANGLES_M = [
    ((2.2, 1.1), (-0.9, 2.3)),
    ((2.6, -1.0), (-0.8, -2.0)),
    ((-2.3, 1.4), (0.6, 2.1)),
    ((1.9, 1.2), (0.9, -2.4)),
]


def _photocentre_mas(img, fov_mas):
    """Sky photocentre (dra East, ddec North) of a virgil render() image,
    where row 0 is North (+ddec) and column 0 is East (+dra)."""
    n = img.shape[0]
    axis = ((n - 1) / 2.0 - np.arange(n)) * (fov_mas / n)
    total = float(img.sum())
    dra = float((img * axis[None, :]).sum() / total)
    ddec = float((img * axis[:, None]).sum() / total)
    pa_deg = float(np.rad2deg(np.arctan2(dra, ddec)) % 360.0)
    return dra, ddec, pa_deg


def _triangle_closure_phases_virgil(disk):
    cps = []
    for (u1, v1), (u2, v2) in TRIANGLES_M:
        us = jnp.array([u1, u2, u1 + u2])
        vs = jnp.array([v1, v2, v1 + v2])
        vis = np.asarray(disk.model(us, vs, WAVEL_M))
        cps.append(float(np.rad2deg(np.angle(vis[0] * vis[1] * np.conj(vis[2])))))
    return np.array(cps)


def _triangle_closure_phases_cloud(cloud):
    cps = []
    for (u1, v1), (u2, v2) in TRIANGLES_M:
        us = np.array([u1, u2, u1 + u2])
        vs = np.array([v1, v2, v1 + v2])
        vis = sky.visibility(cloud, us, vs, WAVEL_M)
        cps.append(float(np.rad2deg(np.angle(vis[0] * vis[1] * np.conj(vis[2])))))
    return np.array(cps)


def _check_flared_disk_nearside(disk, phase_fn):
    fov_mas = RING_GEOM["npix"] * RING_GEOM["pixel_scale_mas"]
    img = np.asarray(disk.render(npix=RING_GEOM["npix"], fov_mas=fov_mas))
    dra, ddec, pa_phot = _photocentre_mas(img, fov_mas)
    pa_err = abs((pa_phot - NEAR_SIDE_PA_DEG + 180.0) % 360.0 - 180.0)
    assert dra < -2.0 and ddec < -1.0
    assert pa_err < 1.0

    # Major-axis (160 deg) vs near-side minor-axis (250 deg) V^2 anisotropy.
    b_m = 3.5
    u_maj = b_m * np.sin(np.deg2rad(PA_DEG))
    v_maj = b_m * np.cos(np.deg2rad(PA_DEG))
    u_min = b_m * np.sin(np.deg2rad(NEAR_SIDE_PA_DEG))
    v_min = b_m * np.cos(np.deg2rad(NEAR_SIDE_PA_DEG))
    v_vir = np.asarray(disk.model(jnp.array([u_maj, u_min]), jnp.array([v_maj, v_min]), WAVEL_M))
    v2_maj, v2_min = float(np.abs(v_vir[0]) ** 2), float(np.abs(v_vir[1]) ** 2)
    assert v2_maj < v2_min

    # Direct Fourier sum of the documented 3D flared-disk surface (crosscheck.disks).
    east, north = cdisks.pixel_centres(RING_GEOM["npix"], RING_GEOM["pixel_scale_mas"])
    geom = {k: v for k, v in RING_GEOM.items() if k not in ("npix", "pixel_scale_mas")}
    bright = cdisks.brightness(east, north, phase=phase_fn, **geom)
    ref_cloud = sky.Cloud(east, north, bright / bright.sum())

    cp_vir = _triangle_closure_phases_virgil(disk)
    cp_ref = _triangle_closure_phases_cloud(ref_cloud)
    max_dcp = float(np.max(np.abs((cp_vir - cp_ref + 180.0) % 360.0 - 180.0)))
    assert np.all(np.abs(cp_vir) > 1.0)
    assert max_dcp < 1e-10

    # First-principles 3D scattering-angle ring (crosscheck.disks.scattered_ring)
    # agrees on the WSW photocentre quadrant and every first-lobe closure-phase sign.
    e3d, n3d, w3d = cdisks.scattered_ring(
        RING_GEOM["radius"], RING_GEOM["fwhm"], INC_DEG, PA_DEG, phase_fn
    )
    cp_3d = _triangle_closure_phases_cloud(sky.Cloud(e3d, n3d, w3d))
    assert float(np.sum(e3d * w3d)) < -1.0 and float(np.sum(n3d * w3d)) < -0.5
    assert np.all(np.sign(cp_vir) == np.sign(cp_3d))

    record("max_abs_dCP_deg", max_dcp)
    record("photocentre_pa_error_deg", pa_err)
    record("v2_minor_minus_major", v2_min - v2_maj)


@pytest.mark.validates(
    "virgil.models.FlaredDiskGaussian",
    roots=["mathematics"],
    property="near_side",
    headline="max_abs_dCP_deg",
)
def test_flared_disk_gaussian_nearside_geometry():
    """FlaredDiskGaussian places the forward-scattering near-side rim at
    pa + 90 deg (WSW at pa = 160 deg) and matches the 3D surface Fourier sum."""
    _check_flared_disk_nearside(
        vm.FlaredDiskGaussian(sigma_theta=45.0, **RING_GEOM),
        lambda t: cdisks.phase_gaussian(t, 45.0),
    )


@pytest.mark.validates(
    "virgil.models.FlaredDiskHG",
    roots=["mathematics"],
    property="near_side",
    headline="max_abs_dCP_deg",
)
def test_flared_disk_hg_nearside_geometry():
    """FlaredDiskHG places the forward-scattering near-side rim at pa + 90 deg
    (WSW at pa = 160 deg) and matches the 3D surface Fourier sum."""
    _check_flared_disk_nearside(
        vm.FlaredDiskHG(g=0.55, **RING_GEOM),
        lambda t: cdisks.phase_hg(t, 0.55),
    )


@pytest.mark.validates(
    "virgil.models.FlaredDiskPowerLaw",
    roots=["mathematics"],
    property="near_side",
    headline="max_abs_dCP_deg",
)
def test_flared_disk_powerlaw_nearside_geometry():
    """FlaredDiskPowerLaw places the forward-scattering near-side rim at
    pa + 90 deg (WSW at pa = 160 deg) and matches the 3D surface Fourier sum."""
    _check_flared_disk_nearside(
        vm.FlaredDiskPowerLaw(n=2.5, **RING_GEOM),
        lambda t: cdisks.phase_power(t, 2.5),
    )


@pytest.mark.validates(
    "virgil.models.FlaredDiskGaussian",
    "virgil.models.FlaredDiskHG",
    "virgil.models.FlaredDiskPowerLaw",
    roots=["mathematics"],
    kind="control",
)
def test_flared_disk_farside_controls_flip_quadrant_and_closure_phase_sign():
    """Negative controls: rotating pa by 180 deg (pa = 340 deg -> near side ENE
    at 70 deg) or reversing Henyey-Greenstein scattering to backward (g < 0 on
    an unskewed ring) moves the photocentre to ENE and flips every closure phase."""
    fov_mas = RING_GEOM["npix"] * RING_GEOM["pixel_scale_mas"]
    for cls_name, kw in [
        ("FlaredDiskGaussian", {"sigma_theta": 45.0}),
        ("FlaredDiskHG", {"g": 0.55}),
        ("FlaredDiskPowerLaw", {"n": 2.5}),
    ]:
        disk_wsw = getattr(vm, cls_name)(**kw, **RING_GEOM)
        disk_ene = getattr(vm, cls_name)(**kw, **{**RING_GEOM, "pa": PA_DEG + 180.0})
        dra_ene, ddec_ene, _ = _photocentre_mas(
            np.asarray(disk_ene.render(npix=RING_GEOM["npix"], fov_mas=fov_mas)), fov_mas
        )
        cp_wsw = _triangle_closure_phases_virgil(disk_wsw)
        cp_ene = _triangle_closure_phases_virgil(disk_ene)
        max_dcp_flip = float(np.max(np.abs((cp_wsw - cp_ene + 180.0) % 360.0 - 180.0)))
        record(f"control_{cls_name}_flip_dCP_deg", max_dcp_flip)
        assert dra_ene > 2.0 and ddec_ene > 1.0
        assert np.all(np.sign(cp_ene) == -np.sign(cp_wsw))
        assert max_dcp_flip > 5.0

    unskewed = {**RING_GEOM, "skew": 0.0}
    cp_fwd = _triangle_closure_phases_virgil(vm.FlaredDiskHG(g=0.55, **unskewed))
    cp_bwd = _triangle_closure_phases_virgil(vm.FlaredDiskHG(g=-0.55, **unskewed))
    assert np.all(np.sign(cp_bwd) == -np.sign(cp_fwd))
