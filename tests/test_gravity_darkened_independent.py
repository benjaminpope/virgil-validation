"""GravityDarkenedStar against our own ELR11 star (crosscheck.elr), written
from Espinosa Lara & Rieutord (2011, A&A 533, A43) rather than from virgil
or from Dholakia's code (whose golden values virgil's own tests use).

Our reference integrates the brightness over the Roche surface on a fine
midpoint grid in colatitude and azimuth, each point weighted by the ELR11
flux times its projected area; virgil sums the visible triangles of its
mesh (n_lat rings). Checks of the reference itself: omega = 0 is a uniform
disk; the flux at the pole and equator takes ELR11's closed forms; the
surface normal is along the effective gravity (the surface is an
equipotential).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from crosscheck import elr, sky
from evidence.plugin import record

vm = pytest.importorskip("virgil.models")

pytestmark = pytest.mark.x64

WL = 1.65e-6


def baselines(n=40, b_max=220.0, seed=0):
    rng = np.random.default_rng(seed)
    r = b_max * np.sqrt(rng.uniform(0.01, 1.0, n))
    t = rng.uniform(0, 2 * np.pi, n)
    return r * np.sin(t), r * np.cos(t)


@pytest.mark.validates("crosscheck.elr", roots=["mathematics"], kind="reference")
def test_reference_star_limits():
    """The reference against closed forms: a sphere is a uniform disk;
    ELR11's flux is exp(2/3 omega² r_p³) / r_p² at the pole and
    (1 - omega²)^(1/3) at the equator; the normal is along g."""
    u, v = baselines()
    sphere = sky.visibility(elr.cloud(2.0, 0.0, 60.0, 30.0), u, v, WL)
    np.testing.assert_allclose(sphere, sky.vis_uniform_disk(u, v, WL, 2.0), atol=2e-5)
    for omega in (0.5, 0.8, 0.95):
        r_p = elr.radius(np.array([1e-8]), omega)[0]
        pole = elr.flux(np.array([1e-6]), np.array([r_p]), omega)[0]
        assert pole == pytest.approx(np.exp(2 / 3 * omega**2 * r_p**3) / r_p**2, rel=1e-6)
        # 1e-3 from the equator: both sides of the ϑ equation are ~δ³ there, so
        # closer in they cancel in float64 (the grid's nearest point is 4e-3 away);
        # the limit is reached to O(δ² / (1 - omega²))
        t = np.pi / 2 - 1e-3
        eq = elr.flux(np.array([t]), elr.radius(np.array([t]), omega), omega)[0]
        assert eq == pytest.approx((1 - omega**2) ** (1 / 3), rel=1e-3)
        _, dA, _, (T, R, _) = elr.surface(omega, 60, 8)
        g_r, g_t = elr.gravity(T, R, omega)
        # in (r, theta) components on the phi = midpoint meridians: dA is along -g
        st, ct = np.sin(T), np.cos(T)
        n_r = (dA[..., 0] * st * np.cos(np.pi / 8) + dA[..., 1] * st * np.sin(np.pi / 8) + dA[..., 2] * ct)
        n_t = (dA[..., 0] * ct * np.cos(np.pi / 8) + dA[..., 1] * ct * np.sin(np.pi / 8) - dA[..., 2] * st)
        cross = (n_r * g_t - n_t * g_r)[:, 0] / (np.hypot(n_r, n_t) * np.hypot(g_r, g_t))[:, 0]
        assert np.max(np.abs(cross)) < 1e-10


CASES = [(0.6, 60.0, 30.0), (0.9, 85.0, 120.0), (0.95, 30.0, 300.0), (0.8, 0.0, 0.0), (0.7, 90.0, 45.0)]


@pytest.mark.parametrize("omega,inc,pa", CASES, ids=["0.6-i60", "0.9-i85", "0.95-i30", "0.8-pole-on", "0.7-equator-on"])
@pytest.mark.validates("virgil.models.GravityDarkenedStar", "virgil._elr", roots=["mathematics"], property="pole_pa")
def test_gravity_darkened_star_against_elr11(omega, inc, pa):
    u, v = baselines()
    want = sky.visibility(elr.cloud(2.0, omega, inc, pa), u, v, WL)
    errs = {}
    for n in (32, 64, 128):
        got = np.asarray(vm.GravityDarkenedStar(2.0, omega=omega, inc=inc, pa=pa, n_lat=n)
                         .model(jnp.asarray(u), jnp.asarray(v), WL))
        errs[n] = float(np.max(np.abs(got - want)))
    for n, e in errs.items():
        record(f"max_abs_dV_nlat{n}", e)
    assert errs[128] < 2e-3
    # second order: each doubling of n_lat cuts the error by about 4
    for coarse, fine in ((32, 64), (64, 128)):
        record(f"convergence_ratio_{coarse}_{fine}", errs[coarse] / errs[fine])
        assert 3.0 < errs[coarse] / errs[fine] < 5.0, (coarse, errs)


@pytest.mark.parametrize("wavel", [0.7e-6, 1.65e-6, 2.2e-6])
@pytest.mark.validates("virgil.models.GravityDarkenedStar", "virgil._elr", roots=["mathematics"], property="pole_pa")
def test_chromatic_gravity_darkened_star_against_elr11(wavel):
    """t_pole set: every point radiates B_lambda(T) with T from the ELR11
    flux, T_eff^4 proportional to F, so the hot pole outweighs the cool
    equator more at short wavelengths."""
    u, v = baselines()
    omega, inc, pa, t_pole = 0.9, 50.0, 70.0, 9000.0
    want = sky.visibility(elr.cloud(2.0, omega, inc, pa, t_pole=t_pole, wavel=wavel), u, v, wavel)
    got = np.asarray(vm.GravityDarkenedStar(2.0, omega=omega, inc=inc, pa=pa, n_lat=128, t_pole=t_pole)
                     .model(jnp.asarray(u), jnp.asarray(v), wavel))
    err = float(np.max(np.abs(got - want)))
    record("max_abs_dV_nlat128", err)
    assert err < 2e-3
