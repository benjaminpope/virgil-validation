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


def swapped_synthetic(n_stations=5, n_scan=3, seed=0, flip_seed=7):
    """``synthetic`` with each baseline randomly stored as (b, a) with conjugated visibility and negated u, v."""
    u, v, vis, time, a1, a2 = synthetic(n_stations, n_scan, seed)
    flip = np.random.default_rng(flip_seed).random(len(time)) < 0.5
    assert flip.any() and not flip.all()
    return (np.where(flip, -u, u), np.where(flip, -v, v), np.where(flip, np.conj(vis), vis), time,
            np.where(flip, a2, a1), np.where(flip, a1, a2))


def test_baseline_direction_invariance():
    with x64():
        _, _, vis, time, a1, a2 = synthetic()
        _, _, vis_s, time_s, b1, b2 = swapped_synthetic()
        assert np.any(b1 > b2)
        t0, t1 = mf.build_triangles(time, a1, a2), mf.build_triangles(time_s, b1, b2)
        assert np.array_equal(t0.labels, t1.labels)
        assert np.max(np.abs(np.asarray(mf.ci_vector(vis, t0)) - np.asarray(mf.ci_vector(vis_s, t1)))) < 1e-12


def test_custom_station_order():
    with x64():
        _, _, vis, time, a1, a2 = swapped_synthetic()
        order = [3, 1, 4, 0, 2]
        tri = mf.build_triangles(time, a1, a2, order=order)
        assert np.all(tri.labels[:, 1] == 3)  # the reference is the first of the order present
        rank = {s: i for i, s in enumerate(order)}
        assert all(rank[int(a)] < rank[int(b)] for _, _, a, b in tri.labels)
        gains = np.exp(np.random.default_rng(2).normal(size=5) + 1j * np.random.default_rng(3).uniform(-3, 3, 5))
        ci0 = np.asarray(mf.ci_vector(vis, tri))
        ci1 = np.asarray(mf.ci_vector(mf.apply_station_gains(vis, a1, a2, gains), tri))
        assert np.max(np.abs(ci0 - ci1)) < 1e-10
        assert not np.allclose(ci0, np.asarray(mf.ci_vector(vis, mf.build_triangles(time, a1, a2))))  # a different pinning


def test_duplicate_baseline_raises_unless_grouped():
    _, _, _, time, a1, a2 = synthetic(n_stations=4, n_scan=2)
    time2, a1_2, a2_2 = (np.concatenate([x, x]) for x in (time, a1, a2))
    with pytest.raises(ValueError, match="occurs twice"):
        mf.build_triangles(time2, a1_2, a2_2)
    group = np.repeat([0, 1], len(time))
    tri = mf.build_triangles(time2, a1_2, a2_2, group=group)
    assert tri.n_scan == 4


def geodesic_scan(fit, alpha, seed=0):
    """Per-scan unit vectors at angle ``alpha`` from the data invariants along a fixed random tangent direction."""
    rng = np.random.default_rng(seed)
    x = fit.ci
    t = rng.normal(size=x.shape)
    cscan = np.concatenate([fit.tri.scan, fit.tri.scan])
    for s in range(fit.tri.n_scan):
        i = cscan == s
        t[i] -= (t[i] @ x[i]) * x[i]
        t[i] /= np.linalg.norm(t[i])
    return np.cos(alpha) * x + np.sin(alpha) * t


def test_full_chi2_monotonic_to_antipode():
    with x64():
        u, v, vis, time, a1, a2 = synthetic(n_stations=5)
        fit = mf.CIFit.from_arrays(u, v, vis, np.full(len(u), 4.0), time, a1, a2)
        alphas = np.linspace(0.0, np.pi, 61)
        chi2 = np.array([float(np.sum(np.asarray(fit.whitened_residual(geodesic_scan(fit, a))) ** 2)) for a in alphas])
        assert chi2[0] < 1e-20
        assert np.all(np.diff(chi2) > 0)  # strictly increasing: the antipode is the worst fit
        assert abs(chi2[-1] / chi2[30] - 4.0) < 1e-6  # chi^2 = theta^2 |w|^2 along a geodesic: pi vs pi / 2
        # small angles: the geodesic residual equals the linear tangent one, W (m - x)
        m = geodesic_scan(fit, 1e-4)
        assert np.allclose(np.asarray(fit.whitened_residual(m)), fit.W @ (m - fit.ci), rtol=1e-6)
        # finite value and gradient at m = x and at the antipode
        loss = lambda m: jax.numpy.sum(fit.whitened_residual(m) ** 2)  # noqa: E731
        for m in (fit.ci, -fit.ci):
            assert np.isfinite(float(loss(m))) and np.all(np.isfinite(np.asarray(jax.grad(loss)(jax.numpy.asarray(m)))))


def test_box_prior_log_jacobian():
    import jax.numpy as jnp

    with x64():
        spec = [("a", -2.0, 3.0, "uniform"), ("b", 1.0, 100.0, "log")]
        prior = mf.BoxPrior(spec)
        z = jnp.array([0.7, -1.3])
        x = jnp.array([float(prior.to_values(z)[n]) for n in prior.names])
        jac = jax.jacfwd(lambda z: jnp.stack([prior.to_values(z)[n] for n in prior.names]))(z)
        # density of z = p_x(x) |dx/dz|, with p_x uniform on [lo, hi] or log-uniform
        log_px = jnp.array([-jnp.log(5.0), -jnp.log(x[1]) - jnp.log(jnp.log(100.0))])
        assert abs(float(prior.logprior(z)) - float(jnp.sum(log_px + jnp.log(jnp.abs(jnp.diag(jac)))))) < 1e-10


def test_weighted_summary_circular():
    rng = np.random.default_rng(0)
    phi = (rng.normal(0.0, 0.05, 2000)) % (2 * np.pi)  # straddles the 0 / 2 pi wrap
    pa = (rng.normal(np.pi, 0.05, 2000)) % np.pi  # and the pi wrap of a position angle
    out = mf.weighted_summary({"0_phi": phi, "1_pa": pa, "R": rng.normal(10, 1, 2000)})
    for k, period in (("0_phi", 2 * np.pi), ("1_pa", np.pi)):
        d = (out[k]["median"] + period / 2) % period - period / 2
        assert abs(d) < 0.01 and 0.04 < out[k]["sd"] < 0.06
    assert abs(out["R"]["median"] - 10) < 0.2


def test_init_particles_finite():
    prior = mf.BoxPrior([("a", 0.0, 1.0, "uniform")])
    z = prior.init_particles(jax.random.PRNGKey(3), 100000)
    assert np.all(np.isfinite(np.asarray(z)))
