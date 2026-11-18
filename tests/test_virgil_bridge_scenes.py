"""The scene definitions in ``virgil_bridge``, our only code that imports virgil.

Each ``Scene`` pairs our own truth (``crosscheck.sky`` and ``crosscheck.limb``) with the virgil model
fitted to it. If that pairing were wrong, every fit that uses a scene would be checking virgil against
a different object, so we check the pairing directly: the virgil template at the scene's own truth
parameters gives the same visibilities as the truth function, with no fit and no noise.

This checks the pairing, not virgil. The truth side is ``crosscheck`` (closed forms and quadrature,
checked in ``tests/test_reference_limb.py`` and elsewhere); the scene fits, fitting helpers and the
``alma_*`` modules of ``virgil_bridge`` have their own checks.
"""

import numpy as np
import pytest

from evidence.plugin import record

vb = pytest.importorskip("virgil_bridge")

pytestmark = pytest.mark.x64

TOL = 1e-12  # observed worst case ~1e-14 (the limb-darkened truth is a quadrature), 1e-12 leaves round-off headroom
WAVELS = (1.65e-6, 2.2e-6)


def _baselines():
    rng = np.random.default_rng(20261119)
    return rng.uniform(-100.0, 100.0, (2, 60))


@pytest.mark.validates("virgil_bridge", roots=["mathematics"], kind="reference", headline="max_abs_template_minus_truth")
def test_bridge_scene_templates_reproduce_their_truth_at_the_truth_parameters():
    """Every ``virgil_bridge`` scene (the four VLTI scenes, the limb-darkened scene and the four
    aperture-masking scenes) has a virgil template whose complex visibility at the truth parameters
    equals the ``crosscheck`` truth visibility to 1e-12, at two wavelengths (4.8 um for the masking scenes) and 60 random baselines."""
    u, v = _baselines()
    cases = [(f(), WAVELS) for f in vb.SCENES + vb.LIMB_SCENES] + [(s, (4.8e-6,)) for s in vb.masking_scenes()]
    worst = 0.0
    n = 0
    for scene, wavels in cases:
        for w in wavels:
            ww = np.full_like(u, w)
            got = np.asarray(scene.template.model(u, v, ww), complex)
            want = np.asarray(scene.vis(u, v, ww), complex)
            worst = max(worst, float(np.max(np.abs(got - want))))
            n += 1
    assert n == 2 * 5 + 4
    record("max_abs_template_minus_truth", worst)
    assert worst < TOL
