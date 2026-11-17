"""crosscheck.disks, checked before it judges virgil: the Henyey-Greenstein phase function
is normalized with mean cosine g; a flat disk's brightness peaks where its documented
geometry puts the ring. The scattered ring is rendered into crosscheck.sky's cloud."""

import numpy as np
import pytest
from scipy import integrate

from crosscheck import disks, sky
from evidence.plugin import record


@pytest.mark.validates("crosscheck.disks", roots=["mathematics"], kind="reference")
def test_scattered_ring_render():
    """Isotropic scattering, or a face-on ring, gives a centrosymmetric image
    (real visibilities); forward scattering brightens the near side, at
    pa + 90; and the flux-weighted inclined ring keeps the major axis's
    extent and shrinks the minor one by cos i."""
    u, v = np.random.default_rng(1).uniform(-100, 100, (2, 200))
    iso = disks.scattered_ring(40.0, 5.0, 50.0, 30.0, lambda t: np.ones_like(t))
    face = disks.scattered_ring(40.0, 5.0, 0.0, 30.0, lambda t: disks.phase_hg(t, 0.6))
    for e, n, w in (iso, face):
        assert np.max(np.abs(sky.visibility(sky.Cloud(e, n, w), u, v, 2e-6).imag)) < 1e-10
    e, n, w = disks.scattered_ring(40.0, 5.0, 50.0, 30.0, lambda t: disks.phase_hg(t, 0.6))
    near = np.array([np.cos(np.deg2rad(30.0)), -np.sin(np.deg2rad(30.0))])
    assert np.sum(w * (e * near[0] + n * near[1])) > 0
    e, n, w = iso
    major = np.array([np.sin(np.deg2rad(30.0)), np.cos(np.deg2rad(30.0))])
    a, b = e * major[0] + n * major[1], e * near[0] + n * near[1]
    assert np.sqrt(np.sum(w * b**2) / np.sum(w * a**2)) == pytest.approx(np.cos(np.deg2rad(50.0)), rel=1e-6)


@pytest.mark.validates("crosscheck.disks", roots=["mathematics"], kind="reference")
def test_disk_phase_function_and_ring_geometry():
    for g in (-0.5, 0.0, 0.3, 0.8):
        norm = integrate.quad(lambda t: disks.phase_hg(t, g) * 2 * np.pi * np.sin(t), 0, np.pi, epsabs=1e-13)[0]
        mean_cos = integrate.quad(lambda t: disks.phase_hg(t, g) * 2 * np.pi * np.sin(t) * np.cos(t), 0, np.pi, epsabs=1e-13)[0]
        assert abs(norm - 1) < 1e-10 and abs(mean_cos - g) < 1e-10
    flat = dict(radius=10.0, fwhm=2.0, phase=lambda t: np.ones_like(t), aspect=0.0)
    for inc, pa in [(0.0, 0.0), (60.0, 30.0), (75.0, 200.0)]:
        p, i = np.deg2rad(pa), np.deg2rad(inc)
        s = np.linspace(0, 20, 4001)
        along = disks.brightness(s * np.sin(p), s * np.cos(p), inc=inc, pa=pa, **flat)  # the major axis, at PA
        across = disks.brightness(s * np.cos(p), -s * np.sin(p), inc=inc, pa=pa, **flat)  # the minor axis
        assert abs(s[np.argmax(along)] - 10.0) < 0.01  # the ring at its radius along the major axis
        assert abs(s[np.argmax(across)] - 10.0 * np.cos(i)) < 0.01  # foreshortened by cos(inc) across it
