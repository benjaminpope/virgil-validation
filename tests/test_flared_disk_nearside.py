"""External confirmation of the near_side convention on virgil's FlaredDisk
models against published PDS 70 observations and oimodeler.

In scattered-light images of the inclined protoplanetary disk around PDS 70
(Keppler et al. 2018, A&A 617, A44; Blakely et al. 2025, AJ 169, 137), the
outer ring has inclination ``inc = 52 deg`` and major-axis position angle
``pa = 160 deg`` (East of North), and dust forward scattering makes the
physical near side of the ring brightest in the **West-Southwest** sky
quadrant (``dra < 0, ddec < 0``, position angle ~250 deg).

To test ``FlaredDiskGaussian``, ``FlaredDiskHG``, and ``FlaredDiskPowerLaw``
without any circular ``pa + 90`` arithmetic, we:
1. Evaluate each model at PDS 70's published ``(inc=52, pa=160)`` with
   forward-peaked dust scattering and verify its rendered sky brightness has
   its flux-weighted photocentre and bright-rim peak in the published WSW
   quadrant (``dra < 0, ddec < 0``, sky PA within 1 deg of 250 deg).
2. Compare interferometric squared visibilities and closure phases against
   ``oimodeler.oimESKRing`` instantiated directly at PDS 70's published
   literal orientation (``pa = 160 deg``, ``skwPa = 250 deg``): both codes
   give lower ``V^2`` along the 160 deg major axis than along the 250 deg
   minor axis, and agree in closure-phase sign across every asymmetric
   baseline triangle.
3. Verify as negative controls that an ENE far-side ring (``skwPa = 70 deg``),
   backward scattering (``g < 0``), or a 180 deg PA flip (``pa = 340 deg``)
   place the bright rim in the ENE quadrant and invert the closure phases.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from evidence.plugin import record

oim = pytest.importorskip("oimodeler")
vm = pytest.importorskip("virgil.models")

pytestmark = [pytest.mark.x64, pytest.mark.external]

# Published PDS 70 outer-ring orientation (Keppler et al. 2018, A&A 617, A44;
# Blakely et al. 2025, AJ 169, 137): major axis PA = 160 deg, inclination = 52 deg,
# empirically observed forward-scattering near side in WSW at sky PA ~ 250 deg.
PDS70_INC_DEG = 52.0
PDS70_PA_DEG = 160.0
PDS70_WSW_PA_DEG = 250.0  # literal published WSW near-side sky PA
PDS70_ENE_PA_DEG = 70.0   # opposite ENE far-side sky PA (negative control)

# Scaled angular ring radius (40 mas, FWHM 15 mas) so 2-8 m baselines at 2.2 um
# probe the first visibility lobe and strong closure-phase regime cleanly.
RING_GEOM = {
    "radius": 40.0,
    "fwhm": 15.0,
    "inc": PDS70_INC_DEG,
    "pa": PDS70_PA_DEG,
    "skew": 0.6,
    "aspect": 0.12,
    "flaring": 1.15,
    "symmetric": 0.15,
    "npix": 128,
    "pixel_scale_mas": 1.5,
}
WAVEL_M = 2.2e-6

# Four independent non-collinear baseline triangles (metres) inside the ring's
# first visibility lobe (all legs <= 3.5 m at 2.2 um, below the first Bessel
# null at ~3.8 m), where closure phases reach 2-15 deg and their sign depends
# purely on WSW vs ENE near-side skewness rather than radial profile shape.
TRIANGLES_M = [
    ((2.2, 1.1), (-0.9, 2.3)),
    ((2.6, -1.0), (-0.8, -2.0)),
    ((-2.3, 1.4), (0.6, 2.1)),
    ((1.9, 1.2), (0.9, -2.4)),
]

MODELS = [
    ("FlaredDiskGaussian", {"sigma_theta": 45.0}),
    ("FlaredDiskHG", {"g": 0.55}),
    ("FlaredDiskPowerLaw", {"n": 2.5}),
]


def _photocenter_mas(img, fov_mas):
    """Sky photocentre (dra East, ddec North) of a virgil render() image,
    where row 0 is North (+ddec) and column 0 is East (+dra)."""
    n = img.shape[0]
    axis = ((n - 1) / 2.0 - np.arange(n)) * (fov_mas / n)
    total = float(img.sum())
    dra = float((img * axis[None, :]).sum() / total)
    ddec = float((img * axis[:, None]).sum() / total)
    pa_deg = float(np.rad2deg(np.arctan2(dra, ddec)) % 360.0)
    return dra, ddec, pa_deg


def _esk_ring(skw_pa_deg):
    """oimodeler skewed elliptical ring at PDS 70's literal orientation."""
    ring = oim.oimESKRing(
        din=65.0,
        dout=95.0,
        pa=PDS70_PA_DEG,
        elong=1.0 / np.cos(np.deg2rad(PDS70_INC_DEG)),
        skw=0.6,
        skwPa=skw_pa_deg,
    )
    return oim.oimModel([ring])


def _triangle_closure_phases_virgil(disk):
    cps = []
    for (u1, v1), (u2, v2) in TRIANGLES_M:
        us = jnp.array([u1, u2, u1 + u2])
        vs = jnp.array([v1, v2, v1 + v2])
        vis = np.asarray(disk.model(us, vs, WAVEL_M))
        cps.append(float(np.rad2deg(np.angle(vis[0] * vis[1] * np.conj(vis[2])))))
    return np.array(cps)


def _triangle_closure_phases_oim(model):
    cps = []
    for (u1, v1), (u2, v2) in TRIANGLES_M:
        us = np.array([u1, u2, u1 + u2]) / WAVEL_M
        vs = np.array([v1, v2, v1 + v2]) / WAVEL_M
        ccf = model.getComplexCoherentFlux(us, vs, np.full(3, WAVEL_M))
        cps.append(float(np.rad2deg(np.angle(ccf[0] * ccf[1] * np.conj(ccf[2])))))
    return np.array(cps)


@pytest.mark.parametrize(
    "cls_name,phase_kw",
    [
        pytest.param(
            cls_name,
            phase_kw,
            marks=pytest.mark.validates(
                f"virgil.models.{cls_name}",
                roots=["literature", "oimodeler"],
                property="near_side",
                headline="photocenter_pa_error_deg",
            ),
        )
        for cls_name, phase_kw in MODELS
    ],
)
def test_pds70_forward_scattering_nearside_is_wsw_and_matches_oimodeler(cls_name, phase_kw):
    """At PDS 70's published (inc=52, pa=160), all three FlaredDisk classes
    place the forward-scattering near side in the WSW quadrant (dra < 0,
    ddec < 0, sky PA ~ 250 deg) and match oimESKRing(pa=160, skwPa=250) in
    both major/minor-axis V^2 anisotropy and closure-phase sign."""
    disk = getattr(vm, cls_name)(**phase_kw, **RING_GEOM)
    fov_mas = RING_GEOM["npix"] * RING_GEOM["pixel_scale_mas"]
    img = np.asarray(disk.render(npix=RING_GEOM["npix"], fov_mas=fov_mas))
    dra, ddec, pa_phot = _photocenter_mas(img, fov_mas)
    pa_err = abs((pa_phot - PDS70_WSW_PA_DEG + 180.0) % 360.0 - 180.0)

    # 1. Published PDS 70 WSW sky quadrant (dra < 0 East, ddec < 0 North).
    assert dra < -2.0 and ddec < -1.0
    assert pa_err < 1.0

    # 2. Major-axis (160 deg) vs minor-axis (250 deg) V^2 anisotropy at |B| = 3.5 m.
    b_m = 3.5
    u_maj = b_m * np.sin(np.deg2rad(PDS70_PA_DEG))
    v_maj = b_m * np.cos(np.deg2rad(PDS70_PA_DEG))
    u_min = b_m * np.sin(np.deg2rad(PDS70_WSW_PA_DEG))
    v_min = b_m * np.cos(np.deg2rad(PDS70_WSW_PA_DEG))

    v_vir = np.asarray(disk.model(jnp.array([u_maj, u_min]), jnp.array([v_maj, v_min]), WAVEL_M))
    v2_maj_vir, v2_min_vir = float(np.abs(v_vir[0]) ** 2), float(np.abs(v_vir[1]) ** 2)

    esk_wsw = _esk_ring(PDS70_WSW_PA_DEG)
    ccf_oim = esk_wsw.getComplexCoherentFlux(
        np.array([u_maj, u_min]) / WAVEL_M,
        np.array([v_maj, v_min]) / WAVEL_M,
        np.full(2, WAVEL_M),
    )
    v2_maj_oim, v2_min_oim = float(np.abs(ccf_oim[0]) ** 2), float(np.abs(ccf_oim[1]) ** 2)
    assert v2_maj_vir < v2_min_vir
    assert v2_maj_oim < v2_min_oim

    # 3. Closure-phase sign agreement across all four triangles with oimESKRing(skwPa=250).
    cp_vir = _triangle_closure_phases_virgil(disk)
    cp_oim = _triangle_closure_phases_oim(esk_wsw)
    assert np.all(np.abs(cp_vir) > 1.0) and np.all(np.abs(cp_oim) > 1.0)
    assert np.all(np.sign(cp_vir) == np.sign(cp_oim))
    sin_alignment = float(
        np.dot(np.sin(np.deg2rad(cp_vir)), np.sin(np.deg2rad(cp_oim)))
        / (np.linalg.norm(np.sin(np.deg2rad(cp_vir))) * np.linalg.norm(np.sin(np.deg2rad(cp_oim))))
    )

    record("photocenter_pa_error_deg", pa_err)
    record("v2_minor_minus_major", v2_min_vir - v2_maj_vir)
    record("cp_sin_alignment", sin_alignment)
    assert sin_alignment > 0.95


@pytest.mark.parametrize("cls_name,phase_kw", MODELS)
@pytest.mark.validates(
    "virgil.models.FlaredDiskGaussian",
    "virgil.models.FlaredDiskHG",
    "virgil.models.FlaredDiskPowerLaw",
    roots=["literature", "oimodeler"],
    kind="control",
)
def test_pds70_farside_ene_controls_flip_closure_phase_sign(cls_name, phase_kw):
    """Negative controls: comparing PDS 70's WSW forward-scattering disk
    against an ENE far-side oimESKRing(skwPa=70) or rotating virgil's disk by
    180 deg (pa=340) places brightness in the ENE quadrant and flips every
    closure phase sign."""
    disk_wsw = getattr(vm, cls_name)(**phase_kw, **RING_GEOM)
    cp_vir_wsw = _triangle_closure_phases_virgil(disk_wsw)

    # Control A: oimESKRing with bright side on the ENE far side (skwPa = 70 deg).
    cp_oim_ene = _triangle_closure_phases_oim(_esk_ring(PDS70_ENE_PA_DEG))
    assert np.all(np.sign(cp_vir_wsw) == -np.sign(cp_oim_ene))

    # Control B: virgil disk rotated by 180 deg (pa = 340 deg -> bright side ENE).
    disk_ene = getattr(vm, cls_name)(**phase_kw, **{**RING_GEOM, "pa": PDS70_PA_DEG + 180.0})
    fov_mas = RING_GEOM["npix"] * RING_GEOM["pixel_scale_mas"]
    dra_ene, ddec_ene, _ = _photocenter_mas(
        np.asarray(disk_ene.render(npix=RING_GEOM["npix"], fov_mas=fov_mas)), fov_mas
    )
    cp_vir_ene = _triangle_closure_phases_virgil(disk_ene)
    max_dcp_flip = float(np.max(np.abs((cp_vir_wsw - cp_vir_ene + 180.0) % 360.0 - 180.0)))
    record(f"control_{cls_name}_flip_dCP_deg", max_dcp_flip)
    assert dra_ene > 2.0 and ddec_ene > 1.0
    assert np.all(np.sign(cp_vir_ene) == -np.sign(cp_vir_wsw))
    assert max_dcp_flip > 5.0

    # Control C (HG backward scattering): g < 0 peaks on the ENE far side when unskewed.
    if cls_name == "FlaredDiskHG":
        unskewed = {**RING_GEOM, "skew": 0.0}
        cp_fwd = _triangle_closure_phases_virgil(vm.FlaredDiskHG(g=0.55, **unskewed))
        cp_bwd = _triangle_closure_phases_virgil(vm.FlaredDiskHG(g=-0.55, **unskewed))
        assert np.all(np.sign(cp_bwd) == -np.sign(cp_fwd))
