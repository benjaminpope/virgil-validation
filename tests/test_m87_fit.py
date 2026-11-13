"""Smoke tests of the closure-invariant and model machinery of stages 3-5 of the M87* replication
(src/virgil_bridge/m87_fit.py; design/m87_replication.md). Tiny synthetic array, no data, no fits."""

import contextlib

import numpy as np
import pytest

from virgil_bridge import m87_fit as mf

pytestmark = pytest.mark.validates("virgil_bridge.m87_fit", roots=["self-consistency"], kind="guard")
jax = pytest.importorskip("jax")
needs_crescent = pytest.mark.skipif(not mf.crescent_available(), reason="virgil without Crescent/Blurred (PR #716)")


@contextlib.contextmanager
def x64():
    """float64 for the duration of a test (the process-wide flag is restored after)."""
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", old)


def synthetic(n_stations=5, n_scan=3, seed=0):
    """Baseline rows of an array with every station present in every scan, and a random source's visibilities."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_scan):
        for a in range(n_stations):
            for b in range(a + 1, n_stations):
                rows.append((100.0 * s, a, b))
    time, a1, a2 = (np.array(c) for c in zip(*rows))
    u, v = rng.uniform(-8e9, 8e9, (2, len(time)))
    vis = rng.normal(size=len(time)) + 1j * rng.normal(size=len(time))
    return u, v, vis, time, a1, a2


def test_invariant_count():
    for n in (4, 5, 6):
        _, _, vis, time, a1, a2 = synthetic(n_stations=n)
        tri = mf.build_triangles(time, a1, a2)
        assert tri.n_scan == 3
        assert all(tri.n_per_scan() == (n - 1) * (n - 2) // 2)
        out = np.asarray(mf.ci_vector(vis, tri))
        assert len(out) // 3 == mf.n_invariants(n)  # (N-1)(N-2) normalised numbers per scan
        assert all(tri.n_independent() == mf.n_independent_invariants(n))  # N^2 - 3N + 1 on the sphere
        for s in range(3):
            ii = np.concatenate([np.nonzero(tri.scan == s)[0], tri.n_tri + np.nonzero(tri.scan == s)[0]])
            assert abs(np.linalg.norm(out[ii]) - 1.0) < 1e-5


def test_gain_invariance():
    with x64():
        _, _, vis, time, a1, a2 = synthetic(n_stations=5)
        rng = np.random.default_rng(1)
        gains = np.exp(rng.normal(size=5) + 1j * rng.uniform(-np.pi, np.pi, 5))
        tri = mf.build_triangles(time, a1, a2)
        # the pinned reference station's gain amplitude is a common factor of every triangle, removed by the norm
        ci0 = np.asarray(mf.ci_vector(vis, tri))
        ci1 = np.asarray(mf.ci_vector(mf.apply_station_gains(vis, a1, a2, gains), tri))
        assert np.max(np.abs(ci0 - ci1)) < 1e-10


def test_full_covariance_pd():
    with x64():
        u, v, vis, time, a1, a2 = synthetic(n_stations=5)
        fit = mf.CIFit.from_arrays(u, v, vis, np.full(len(u), 4.0), time, a1, a2)
        C = mf.ci_covariance(vis, fit.weight, fit.tri)
        assert np.max(np.abs(C - C.T)) < 1e-12 * np.abs(C).max()
        assert np.linalg.eigvalsh(C).min() > -1e-10 * np.abs(C).max()  # PSD: singular only along the sphere normal
        assert fit.k == 3 * 11  # N^2 - 3N + 1 independent invariants per scan
        # the whitened residual of tangent-space noise has unit variance per dof
        W = fit.W
        red = W @ C @ W.T
        assert np.allclose(red, np.eye(fit.k), atol=1e-8)


@needs_crescent
def test_composite_flux_and_gradient():
    import jax.numpy as jnp

    p = mf.basis_defaults()
    system = mf.xsringauss_system(p)
    v0 = system.model(jnp.zeros(1), jnp.zeros(1), 1.0)
    assert abs(float(jnp.abs(v0)[0]) - 1.0) < 1e-5  # unit total flux at zero baseline
    u, v, *_ = synthetic()

    def loss(Rp):
        q = dict(p)
        q["0_Rp"] = Rp
        return jnp.sum(jnp.abs(mf.model_vis(mf.xsringauss_system(q), u, v)) ** 2)

    g = jax.grad(loss)(jnp.asarray(p["0_Rp"]))
    assert np.isfinite(float(g)) and float(g) != 0.0
    # native composite (stage 4): finite value and gradient in its own parameters
    spec = mf.native_spec("composite", 1)
    prior = mf.BoxPrior(spec)
    z = prior.init_particles(jax.random.PRNGKey(0), 1)[0]
    scene = mf.native_scene("composite", 1)
    vis = lambda z: mf.model_vis(scene(prior.to_values(z)), u, v)  # noqa: E731
    assert np.all(np.isfinite(np.asarray(vis(z))))
    assert np.all(np.isfinite(np.asarray(jax.jacfwd(lambda z: jnp.abs(vis(z)))(z))))


def test_box_prior_density():
    import jax.numpy as jnp

    prior = mf.BoxPrior([("a", 1.0, 3.0, "uniform"), ("b", 1.0, 100.0, "log")])
    z = prior.init_particles(jax.random.PRNGKey(0), 4000)
    vals = jax.vmap(prior.to_values)(z)
    assert 1.0 <= float(jnp.min(vals["a"])) and float(jnp.max(vals["a"])) <= 3.0
    assert abs(float(jnp.mean(vals["a"])) - 2.0) < 0.05
    assert abs(float(jnp.mean(jnp.log10(vals["b"]))) - 1.0) < 0.05  # log-uniform on [1, 100]


def test_mring_model_flux_and_gradient():
    import jax.numpy as jnp

    prior = mf.BoxPrior(mf.native_spec("mring", 2))
    z = prior.init_particles(jax.random.PRNGKey(1), 1)[0]
    scene = mf.native_scene("mring", 2)
    u, v, *_ = synthetic()
    vis = lambda z: mf.model_vis(scene(prior.to_values(z)), u, v)  # noqa: E731
    v0 = scene(prior.to_values(z)).model(jnp.zeros(1), jnp.zeros(1), 1.0)
    assert abs(float(jnp.abs(v0)[0]) - 1.0) < 1e-5
    assert np.all(np.isfinite(np.asarray(jax.jacfwd(lambda z: jnp.abs(vis(z)))(z))))
