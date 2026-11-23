"""crosscheck.limb, checked before it judges virgil (Issue #12): a quadrature point cloud against
the Hankel transform by adaptive quadrature, Hanbury Brown et al.'s (1974) closed form for the
linear law, and the uniform disk's Airy pattern. These checks use crosscheck.limb and
crosscheck.sky only; the tests of virgil's limb-darkened disks that go through crosscheck.simulate
stay in tests/test_limb_darkening.py."""

import numpy as np
import pytest

from _limb_common import LAWS, assert_close, uvw_samples
from crosscheck import limb, sky


@pytest.fixture(scope="module")
def uvw():
    return uvw_samples()


# ------------------------------------------------ our routes agree first


@pytest.mark.slow
@pytest.mark.parametrize("diam", [3.0, 12.0])
@pytest.mark.parametrize(
    "profile",
    [limb.polynomial(u) for u in LAWS.values()] + [limb.square_root(0.1, 0.6)],
    ids=list(LAWS) + ["square-root"],
)
@pytest.mark.validates("crosscheck.limb", roots=["mathematics"], kind="reference", tier="B")
def test_our_cloud_matches_our_hankel_transform(uvw, profile, diam):
    u, v, wl = (a[::10] for a in uvw)  # quad is slow; every 10th sample
    assert_close(
        sky.visibility(limb.disk(diam, profile, 0.4, -0.7), u, v, wl),
        limb.vis_hankel(u, v, wl, diam, profile, 0.4, -0.7),
        tol=1e-12,
    )


@pytest.mark.slow
@pytest.mark.validates("crosscheck.limb", roots=["mathematics"], kind="reference", tier="B")
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
