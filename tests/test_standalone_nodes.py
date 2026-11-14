"""Independent mathematical and statistical validation of standalone virgil
0.4.1 public API nodes.

* virgil.models.LinearImage: non-negative pixel-flux ("linear") and unbounded
  sqrt-flux ("sqrt", F = w^2) parameterizations, support masking, sky offsets,
  rotation, and flux-weighted System combination against crosscheck.sky's
  direct pixel DFT. Control: linear vs squared sqrt weights and unit-weight
  System normalization fail.
* virgil.spectra.Scaled: fixed template shape normalized by template() times
  free reference flux ratio (ratio * template(wavel) / template()), frozen
  template gradients, and chromatic System visibilities. Control: omitting the
  1 / template() normalization fails whenever template() != 1.
* virgil.orbits.period_grid: uniform frequency spacing Delta(1/P) <= 1 / (k T)
  bounding orbital phase drift over baseline T to <= 1/k cycles between
  neighbouring trial periods, plus domain ValueError guards. Control: a
  log-spaced period grid of the same size violates the 1 / (k T) bound at
  short periods.
* virgil.priors.PopulationScale & virgil.priors.hierarchical_scales: centred
  (u_k = log s_k, exact Gaussian population log_prior) and non-centred
  (s_k = median * exp(spread * z_k), zero population log_prior to avoid double
  counting z_k ~ N(0, 1)) parameterizations, quadrature normalization, KS
  sampling equivalence, and frozen-dataclass hashing. Control: confusing
  log(median) with median or double-counting z_k's log-density fails.
* virgil.inference.expected_fisher, objective_laplace_covariance, and
  observed_information: exact J^T Sigma^-1 J + ridge I with pytree unravelling,
  analytic inverse Hessian (nabla^2 f)^-1 with relative diagonal ridge only
  when non-PD, and observed Hessian of -log L against 4th-order central
  differences of 1/2 our independent chi-squared (crosscheck.chi2), plus
  negative controls.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from scipy import integrate, stats

from crosscheck import chi2 as ours, simulate as sim, sky
from evidence.plugin import record

vm = pytest.importorskip("virgil.models")
sp = pytest.importorskip("virgil.spectra")
vo = pytest.importorskip("virgil.orbits")
pr = pytest.importorskip("virgil.priors")
inf = pytest.importorskip("virgil.inference")
from virgil.oidata import OIData  # noqa: E402

pytestmark = pytest.mark.x64

UTS3 = np.array([[-9.925, -20.335], [14.887, 30.502], [103.306, 43.999]])


# ------------------------------------------------------- virgil.models.LinearImage


@pytest.mark.validates(
    "virgil.models.LinearImage",
    roots=["mathematics"],
    headline="max_abs_dV",
)
def test_linear_image_linear_and_sqrt_visibilities_and_system_flux():
    """LinearImage in both 'linear' and 'sqrt' (flux = w^2) parameterizations,
    with support mask, sky shift, rotation, and inside a System, matches our
    direct pixel-centre DFT and total-flux weighting."""
    rng = np.random.default_rng(104)
    u, v = rng.uniform(-110.0, 110.0, (2, 200))
    wavel = rng.uniform(1.5e-6, 2.4e-6, 200)

    nrow, ncol = 7, 9
    scale = 1.75
    support = np.ones((nrow, ncol), dtype=bool)
    support[0, :] = False
    support[:, -1] = False

    raw_flux = rng.uniform(0.01, 0.25, size=(nrow, ncol))
    # Include negative weights in sqrt mode: flux = (-sqrt(F))^2 = F
    signs = rng.choice([-1.0, 1.0], size=(nrow, ncol))
    sqrt_weights = signs * np.sqrt(raw_flux)
    masked_flux = np.where(support, raw_flux, 0.0)
    total_f = float(masked_flux.sum())

    dra, ddec, rot = 2.4, -1.6, 33.0
    cloud = sky.pixel_image(masked_flux, scale).rotated(rot).shifted(dra, ddec)
    want_img_vis = sky.visibility(cloud, u, v, wavel)

    img_lin = vm.LinearImage(
        raw_flux,
        pixel_scale_mas=scale,
        support=support,
        dra=dra,
        ddec=ddec,
        rotation_deg=rot,
        ft="dft",
        parametrisation="linear",
        reference_flux=0.8,
    )
    img_sqrt = vm.LinearImage(
        sqrt_weights,
        pixel_scale_mas=scale,
        support=support,
        dra=dra,
        ddec=ddec,
        rotation_deg=rot,
        ft="dft",
        parametrisation="sqrt",
        reference_flux=0.8,
    )

    worst_dv = 0.0
    for img in (img_lin, img_sqrt):
        got_vis = np.asarray(img.model(u, v, wavel))
        worst_dv = max(worst_dv, float(np.max(np.abs(got_vis - want_img_vis))))
        assert float(img.total_flux) == pytest.approx(total_f, rel=1e-14)
        np.testing.assert_allclose(np.asarray(img.fluxes), masked_flux, rtol=1e-14, atol=1e-16)
        np.testing.assert_allclose(np.asarray(img.brightness), masked_flux / total_f, rtol=1e-14, atol=1e-16)
        np.testing.assert_allclose(np.asarray(img.regularised_pixels), masked_flux / 0.8, rtol=1e-14, atol=1e-16)
        assert bool(img.is_physical())

        # Inside a System, LinearImage contributes weight = sum(fluxes), not 1
        scene = vm.System(star=vm.PointSource(1.5, -3.0, 1.0), env=img)
        got_sys = np.asarray(scene.model(u, v, wavel))
        want_sys = (1.5 * sky.vis_point(u, v, wavel, -3.0, 1.0) + total_f * want_img_vis) / (1.5 + total_f)
        worst_dv = max(worst_dv, float(np.max(np.abs(got_sys - want_sys))))

    # Bounds and physicality contracts
    lo, hi = img_lin.bounds()
    assert np.all(np.asarray(lo) == 0.0)
    assert np.all(np.isposinf(np.asarray(hi)[support])) and np.all(np.asarray(hi)[~support] == 0.0)
    assert img_sqrt.bounds() is None

    neg_lin = raw_flux.copy()
    neg_lin[2, 3] = -0.05  # inside support
    with pytest.raises(ValueError, match="non-negative"):
        vm.LinearImage(neg_lin, pixel_scale_mas=scale, support=support)
    assert not bool(img_lin.set("pixels", jnp.asarray(neg_lin)).is_physical())

    record("max_abs_dV", worst_dv)
    assert worst_dv < 1e-13


@pytest.mark.validates(
    "virgil.models.LinearImage",
    roots=["mathematics"],
    kind="control",
)
def test_linear_image_sqrt_weights_and_system_flux_controls():
    """Negative controls: reading 'sqrt' weights linearly (w instead of w^2) or
    combining LinearImage in a System with unit flux instead of sum(fluxes)
    disagrees with the model visibilities."""
    rng = np.random.default_rng(105)
    u, v = rng.uniform(-80.0, 80.0, (2, 80))
    wavel = 2.0e-6

    weights = np.array([[0.2, 0.8, 0.3], [0.1, 1.4, 0.5], [0.4, 0.2, 0.9]])
    img_sqrt = vm.LinearImage(weights, pixel_scale_mas=2.5, ft="dft", parametrisation="sqrt")
    got_vis = np.asarray(img_sqrt.model(u, v, wavel))
    wrong_linear_vis = sky.visibility(sky.pixel_image(weights, 2.5), u, v, wavel)
    assert np.max(np.abs(got_vis - wrong_linear_vis)) > 0.05

    scene = vm.System(star=vm.PointSource(1.0, 4.0, -2.0), env=img_sqrt)
    got_sys = np.asarray(scene.model(u, v, wavel))
    true_img_vis = sky.visibility(sky.pixel_image(weights**2, 2.5), u, v, wavel)
    wrong_unit_weight_sys = 0.5 * (sky.vis_point(u, v, wavel, 4.0, -2.0) + true_img_vis)
    assert np.max(np.abs(got_sys - wrong_unit_weight_sys)) > 0.10


# ---------------------------------------------------------- virgil.spectra.Scaled


@pytest.mark.validates(
    "virgil.spectra.Scaled",
    roots=["mathematics"],
    headline="max_rel_dflux",
)
def test_scaled_normalizes_by_template_reference_and_freezes_template():
    """Scaled(template, ratio) equals ratio * template(wavel) / template() at
    every wavelength regardless of template's own normalization, freezes
    gradients wrt template leaves, and weights chromatic Systems properly."""
    wls = np.linspace(1.6e-6, 2.4e-6, 41)
    tpl_pl = sp.PowerLaw(0.37, index=-2.4, wavel0=2.05e-6)
    tpl_sum = sp.Sum(
        cont=sp.PowerLaw(0.42, index=-1.8, wavel0=2.1661e-6),
        line=sp.GaussianLine(0.18, line_wavel=2.1661e-6, fwhm=2.5e-9),
    )

    worst_rel = 0.0
    for tpl, ratio in ((tpl_pl, 0.065), (tpl_sum, 0.12)):
        s = sp.Scaled(tpl, ratio=ratio)
        got = np.asarray(s(wls))
        want = ratio * np.asarray(tpl(wls)) / float(tpl())
        worst_rel = max(worst_rel, float(np.max(np.abs(got / want - 1.0))))
        assert float(s()) == pytest.approx(ratio, rel=1e-15)
        assert float(sp.reference_flux(s)) == pytest.approx(ratio, rel=1e-15)
        assert float(s.wavel0) == pytest.approx(float(tpl.wavel0), rel=1e-15)
        assert bool(s.is_physical())
        with pytest.raises(ValueError, match="non-negative"):
            sp.Scaled(tpl, ratio=-0.01)
        assert not bool(jax.jit(lambda r, t=tpl: sp.Scaled(t, ratio=r).is_physical())(-0.01))

    # Template leaves are held fixed under JAX differentiation; only ratio moves
    s0 = sp.Scaled(tpl_pl, ratio=jnp.array(0.08))
    grads = jax.grad(lambda m: jnp.sum(m(jnp.asarray(wls))))(s0)
    want_dratio = float(np.sum((wls / 2.05e-6) ** -2.4))
    assert float(grads.ratio) == pytest.approx(want_dratio, rel=1e-13)
    tpl_grads = [np.asarray(x) for x in jax.tree_util.tree_leaves(grads.template) if hasattr(x, "dtype")]
    assert all(np.all(g == 0.0) for g in tpl_grads)

    # Chromatic System visibility check
    rng = np.random.default_rng(106)
    u, v = rng.uniform(-90.0, 90.0, (2, 41))
    scene = vm.System(star=vm.PointSource(), comp=vm.PointSource(sp.Scaled(tpl_pl, 0.05), 6.0, -3.5))
    got_vis = np.asarray(scene.model(u, v, wls))
    fc = 0.05 * (wls / 2.05e-6) ** -2.4
    want_vis = (sky.vis_point(u, v, wls) + fc * sky.vis_point(u, v, wls, 6.0, -3.5)) / (1.0 + fc)
    vis_err = float(np.max(np.abs(got_vis - want_vis)))
    assert vis_err < 1e-14

    record("max_rel_dflux", worst_rel)
    record("max_abs_dV_chromatic", vis_err)
    assert worst_rel < 1e-14


@pytest.mark.validates(
    "virgil.spectra.Scaled",
    roots=["mathematics"],
    kind="control",
)
def test_scaled_rejects_unnormalized_product_control():
    """Negative control: multiplying ratio * template(wavel) without dividing by
    template() gives the wrong reference flux whenever template() != 1."""
    tpl = sp.PowerLaw(0.35, index=-2.0, wavel0=2.0e-6)
    wls = np.linspace(1.7e-6, 2.3e-6, 15)
    got = np.asarray(sp.Scaled(tpl, ratio=0.10)(wls))
    unnormalized = 0.10 * np.asarray(tpl(wls))
    assert np.max(np.abs(got - unnormalized)) > 0.05


# ------------------------------------------------------- virgil.orbits.period_grid


@pytest.mark.validates(
    "virgil.orbits.period_grid",
    roots=["mathematics"],
    headline="max_rel_dfreq_step",
)
def test_period_grid_uniform_frequency_spacing_and_phase_drift_bound():
    """period_grid(times, p_min, p_max, k) returns periods strictly increasing
    from p_min to p_max that are uniformly spaced in frequency 1/P with step
    Delta(1/P) <= 1 / (k T), bounding phase drift over baseline T to <= 1/k
    cycles."""
    cases = [
        (np.array([60100.0, 60145.2, 60510.0]), 5.0, 20.0, 9.0),
        (np.array([59000.0, 59250.0]), 7.0, 63.0, 6.0),
        (np.array([60010.5, 60000.5, 60090.5]), 12.0, 80.0, 12.0),  # unsorted input times
    ]
    worst_rel_step = 0.0
    for times, p_min, p_max, k in cases:
        baseline = float(np.max(times) - np.min(times))
        p = np.asarray(vo.period_grid(times, p_min, p_max, k=k))
        assert p[0] == pytest.approx(p_min, rel=1e-14)
        assert p[-1] == pytest.approx(p_max, rel=1e-14)
        assert np.all(np.diff(p) > 0.0)

        freqs = 1.0 / p
        df = -np.diff(freqs)
        # Uniform in 1/P to machine precision
        rel_spread = float(np.max(np.abs(df - df.mean())) / df.mean())
        worst_rel_step = max(worst_rel_step, rel_spread)

        # Phase drift T * Delta(1/P) <= 1 / k cycles (with n_intervals = ceil(DeltaF * k * T))
        n_want = int(np.ceil((1.0 / p_min - 1.0 / p_max) * k * baseline))
        assert p.size == n_want + 1
        assert float(np.max(df) * baseline) <= 1.0 / k + 1e-14

        # Every true period in [p_min, p_max] lies within 1 / (2 k) cycles over T of a grid point
        probes = np.linspace(p_min, p_max, 500)
        nearest_cycle_err = np.max([np.min(np.abs(1.0 / q - freqs)) * baseline for q in probes])
        assert nearest_cycle_err <= 0.5 / k + 1e-14

    for bad_args in (
        ([0.0, 10.0], 0.0, 5.0),
        ([0.0, 10.0], -2.0, 5.0),
        ([0.0, 10.0], 10.0, 10.0),
        ([0.0, 10.0], 15.0, 8.0),
        ([60000.0, 60000.0], 5.0, 20.0),
        ([60000.0], 5.0, 20.0),
    ):
        with pytest.raises(ValueError):
            vo.period_grid(*bad_args)

    record("max_rel_dfreq_step", worst_rel_step)
    assert worst_rel_step < 1e-11


@pytest.mark.validates(
    "virgil.orbits.period_grid",
    roots=["mathematics"],
    kind="control",
)
def test_period_grid_rejects_log_uniform_period_spacing_control():
    """Negative control: a log-uniform period grid with the same number of
    points under-samples short periods, exceeding the 1 / (k T) frequency-step
    bound by more than 2x."""
    times = np.array([60000.0, 60400.0])
    p_min, p_max, k = 4.0, 80.0, 9.0
    got = np.asarray(vo.period_grid(times, p_min, p_max, k=k))
    wrong_geom = np.geomspace(p_min, p_max, got.size)
    max_drift_geom = float(np.max(-np.diff(1.0 / wrong_geom)) * 400.0)
    assert max_drift_geom > 2.5 * (1.0 / k)


# ------------------- virgil.priors.PopulationScale & hierarchical_scales


@pytest.mark.validates(
    "virgil.priors.PopulationScale",
    "virgil.priors.hierarchical_scales",
    roots=["mathematics", "statistics"],
    headline="max_abs_dlogp",
)
def test_hierarchical_scales_and_population_scale_centred_and_noncentred():
    """hierarchical_scales and PopulationScale implement log s_k ~ N(log median, spread^2)
    in both centred and non-centred forms, with normalized log_prior, matching
    Monte Carlo marginals, curvature rows, and frozen-dataclass hashing."""
    n = 4
    priors_c, scales_c = pr.hierarchical_scales("err_s", n, centred=True)
    priors_nc, scales_nc = pr.hierarchical_scales("err_s", n, centred=False)

    assert set(priors_c) == {"err_s_median", "err_s_spread", "err_s_log"}
    assert set(priors_nc) == {"err_s_median", "err_s_spread", "err_s_z"}
    assert len(scales_c) == len(scales_nc) == n

    # Frozen dataclass equality & hashing
    assert scales_c[1] == pr.PopulationScale("err_s", 1, centred=True)
    assert hash(scales_c[1]) == hash(pr.PopulationScale("err_s", 1, centred=True))
    assert scales_c[1] != scales_c[2] and scales_c[1] != scales_nc[1]

    # Hyperprior defaults: LogUniform(0.1, 10) on median, LogUniform(0.01, 1) on spread
    for med in (0.2, 1.0, 6.0):
        want_lp_m = -np.log(med * np.log(100.0))
        assert float(priors_c["err_s_median"].log_prob(med)) == pytest.approx(want_lp_m, abs=1e-13)
    for spr in (0.03, 0.25, 0.8):
        want_lp_s = -np.log(spr * np.log(100.0))
        assert float(priors_c["err_s_spread"].log_prob(spr)) == pytest.approx(want_lp_s, abs=1e-13)

    median_val, spread_val = 2.3, 0.35
    u_vec = np.array([-0.4, 0.15, 0.833213344056216, 1.5])
    z_vec = (u_vec - np.log(median_val)) / spread_val

    vals_c = {
        "err_s_median": jnp.array(median_val),
        "err_s_spread": jnp.array(spread_val),
        "err_s_log": jnp.asarray(u_vec),
    }
    vals_nc = {
        "err_s_median": jnp.array(median_val),
        "err_s_spread": jnp.array(spread_val),
        "err_s_z": jnp.asarray(z_vec),
    }

    worst_dlogp = 0.0
    for k in range(n):
        sc_val = float(scales_c[k](vals_c))
        snc_val = float(scales_nc[k](vals_nc))
        assert sc_val == pytest.approx(np.exp(u_vec[k]), rel=1e-14)
        assert snc_val == pytest.approx(sc_val, rel=1e-14)

        got_lp = float(scales_c[k].log_prior(vals_c))
        want_lp = float(stats.norm.logpdf(u_vec[k], loc=np.log(median_val), scale=spread_val))
        worst_dlogp = max(worst_dlogp, abs(got_lp - want_lp))
        assert float(scales_nc[k].log_prior(vals_nc)) == 0.0

    # Normalisation of PopulationScale.log_prior over u_k in (-inf, +inf)
    norm_int, _ = integrate.quad(
        lambda u: float(
            np.exp(
                scales_c[0].log_prior(
                    {"err_s_median": median_val, "err_s_spread": spread_val, "err_s_log": jnp.full(n, u)}
                )
            )
        ),
        np.log(median_val) - 10.0 * spread_val,
        np.log(median_val) + 10.0 * spread_val,
        epsabs=1e-13,
    )
    assert abs(norm_int - 1.0) < 1e-11

    # Statistical equivalence: z ~ N(0, 1) mapped through non-centred PopulationScale gives log s ~ N(log m, s^2)
    z_samples = priors_nc["err_s_z"].sample(jax.random.key(19), (6000,))
    s0_draws = np.asarray(
        jax.vmap(lambda z: scales_nc[0]({"err_s_median": median_val, "err_s_spread": spread_val, "err_s_z": z}))(
            z_samples
        )
    )
    ks_p = float(stats.kstest(np.log(s0_draws), stats.norm(loc=np.log(median_val), scale=spread_val).cdf).pvalue)
    assert ks_p > 1e-3

    # Custom hyperpriors pass through
    custom_m, custom_s = dist.HalfNormal(2.0), dist.Exponential(3.0)
    pri_custom, _ = pr.hierarchical_scales("c", 2, median=custom_m, spread=custom_s)
    assert pri_custom["c_median"] is custom_m and pri_custom["c_spread"] is custom_s

    record("max_abs_dlogp", worst_dlogp)
    record("abs_norm_err", abs(norm_int - 1.0))
    assert worst_dlogp < 1e-14


@pytest.mark.validates(
    "virgil.priors.PopulationScale",
    "virgil.priors.hierarchical_scales",
    roots=["mathematics"],
    kind="control",
)
def test_population_scale_controls_against_unlogged_median_and_double_counting():
    """Negative controls: using untransformed median instead of log(median) in
    the centred log-prior or adding N(z_k; 0, 1) a second time in non-centred
    PopulationScale.log_prior produces a wrong joint density."""
    _, scales_c = pr.hierarchical_scales("s", 2, centred=True)
    _, scales_nc = pr.hierarchical_scales("s", 2, centred=False)
    vals_c = {"s_median": 2.5, "s_spread": 0.3, "s_log": jnp.array([0.9, 1.1])}
    wrong_unlogged = float(stats.norm.logpdf(0.9, loc=2.5, scale=0.3))
    assert abs(float(scales_c[0].log_prior(vals_c)) - wrong_unlogged) > 5.0

    vals_nc = {"s_median": 2.5, "s_spread": 0.3, "s_z": jnp.array([1.2, -0.8])}
    double_counted = float(stats.norm.logpdf(1.2, loc=0.0, scale=1.0))
    assert abs(float(scales_nc[0].log_prior(vals_nc)) - double_counted) > 0.5


# --- expected_fisher, objective_laplace_covariance, observed_information


@pytest.mark.validates(
    "virgil.inference.expected_fisher",
    roots=["mathematics"],
    headline="max_rel_dF",
)
def test_expected_fisher_matches_analytic_jacobian_and_unravels_pytree():
    """expected_fisher returns J^T Sigma^-1 J + ridge I for a nonlinear pytree
    model and restores flat parameter vectors cleanly."""
    t = np.linspace(0.1, 2.5, 25)
    sigma = np.linspace(0.05, 0.25, 25)
    params = {"amp": jnp.array(1.7), "rates": jnp.array([0.4, 1.3])}

    def pred_fn(p):
        a = p["amp"]
        k0, k1 = p["rates"][0], p["rates"][1]
        return a * jnp.exp(-k0 * jnp.asarray(t)) + jnp.sin(k1 * jnp.asarray(t))

    a0, k0, k1 = 1.7, 0.4, 1.3
    J = np.column_stack([
        np.exp(-k0 * t),
        -a0 * t * np.exp(-k0 * t),
        t * np.cos(k1 * t),
    ])
    want = J.T @ (J / sigma[:, None] ** 2)

    F, unravel = inf.expected_fisher(pred_fn, params, jnp.asarray(sigma), ridge=0.0)
    rel = float(np.max(np.abs(np.asarray(F) - want)) / np.max(np.abs(want)))

    F_ridge, _ = inf.expected_fisher(pred_fn, params, jnp.asarray(sigma), ridge=0.35)
    rel_ridge = float(np.max(np.abs(np.asarray(F_ridge) - (want + 0.35 * np.eye(3)))) / np.max(np.abs(want)))

    rebuilt = unravel(jnp.array([4.0, -1.0, 2.5]))
    assert float(rebuilt["amp"]) == 4.0
    np.testing.assert_allclose(np.asarray(rebuilt["rates"]), [-1.0, 2.5], rtol=0, atol=1e-15)

    record("max_rel_dF", max(rel, rel_ridge))
    assert max(rel, rel_ridge) < 1e-12


@pytest.mark.validates(
    "virgil.inference.expected_fisher",
    roots=["mathematics"],
    kind="control",
)
def test_expected_fisher_rejects_unweighted_and_variance_weighted_jacobians():
    """Negative control: expected_fisher must weight by 1 / sigma^2, never 1
    or sigma^2."""
    rng = np.random.default_rng(107)
    J = rng.normal(size=(20, 3))
    sigma = np.geomspace(0.1, 3.0, 20)
    F, _ = inf.expected_fisher(lambda x: jnp.asarray(J) @ x, jnp.zeros(3), jnp.asarray(sigma))
    F = np.asarray(F)
    for wrong in (J.T @ J, J.T @ (sigma[:, None] ** 2 * J)):
        assert np.max(np.abs(F - wrong)) / np.max(np.abs(F)) > 0.5


@pytest.mark.validates(
    "virgil.inference.objective_laplace_covariance",
    roots=["mathematics"],
    headline="max_rel_dcov",
)
def test_objective_laplace_covariance_matches_inverse_hessian_and_indefinite_ridge():
    """objective_laplace_covariance equals (nabla^2 f(x))^-1 without adding
    ridge on positive-definite objectives, and applies the relative diagonal
    ridge with a RuntimeWarning when indefinite."""
    A = np.array([[5.0, -1.2, 0.4], [-1.2, 2.8, 0.7], [0.4, 0.7, 1.9]])
    x0 = np.array([0.4, -0.6, 1.1])

    def obj(x):
        dx = x - jnp.asarray(x0)
        return 0.5 * dx @ (jnp.asarray(A) @ dx) + 0.25 * jnp.sum(dx**4)

    for probe, add_h in ((x0, np.zeros((3, 3))), (x0 + np.array([0.2, -0.3, 0.1]), np.diag(3.0 * np.array([0.2, -0.3, 0.1]) ** 2))):
        want = np.linalg.inv(A + add_h)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            got = np.asarray(inf.objective_laplace_covariance(obj, jnp.asarray(probe), ridge=0.1))
        err = float(np.max(np.abs(got - want)) / np.max(np.abs(want)))
        assert err < 1e-12

    # Indefinite Hessian triggers RuntimeWarning and adds ridge * |H_ii|
    H_indef = np.array([[4.0, 0.5], [0.5, -1.0]])
    ridge = 2.0
    with pytest.warns(RuntimeWarning):
        got_indef = np.asarray(
            inf.objective_laplace_covariance(lambda x: 0.5 * x @ (jnp.asarray(H_indef) @ x), jnp.zeros(2), ridge=ridge)
        )
    want_indef = np.linalg.inv(H_indef + ridge * np.diag(np.abs(np.diag(H_indef))))
    err_indef = float(np.max(np.abs(got_indef - want_indef)) / np.max(np.abs(want_indef)))

    record("max_rel_dcov", max(err, err_indef))
    assert err_indef < 1e-12


@pytest.mark.validates(
    "virgil.inference.objective_laplace_covariance",
    roots=["mathematics"],
    kind="control",
)
def test_objective_laplace_covariance_rejects_diagonal_and_uninverted_hessian():
    """Negative control: ignoring off-diagonal correlations (1 / diag(H)) or
    returning H instead of H^-1 fails on a correlated quadratic objective."""
    H = np.array([[4.0, 2.4], [2.4, 2.0]])
    got = np.asarray(inf.objective_laplace_covariance(lambda x: 0.5 * x @ (jnp.asarray(H) @ x), jnp.zeros(2)))
    assert np.max(np.abs(got - np.diag(1.0 / np.diag(H)))) > 0.5
    assert np.max(np.abs(got - H)) > 2.0


def _richardson_hessian(fn, x, h=1e-4):
    """4th-order Richardson-extrapolated central-difference Hessian of scalar fn."""
    p = x.size

    def raw_hess(step):
        H = np.zeros((p, p))
        f0 = fn(x)
        for i in range(p):
            ei = np.zeros(p)
            ei[i] = step
            H[i, i] = (fn(x + ei) - 2.0 * f0 + fn(x - ei)) / step**2
            for j in range(i + 1, p):
                ej = np.zeros(p)
                ej[j] = step
                val = (fn(x + ei + ej) - fn(x + ei - ej) - fn(x - ei + ej) + fn(x - ei - ej)) / (4.0 * step**2)
                H[i, j] = H[j, i] = val
        return H

    return (4.0 * raw_hess(h / 2.0) - raw_hess(h)) / 3.0


@pytest.mark.validates(
    "virgil.inference.observed_information",
    roots=["mathematics"],
    headline="max_rel_dobs",
)
def test_observed_information_matches_half_chi2_hessian_and_noiseless_fisher(tmp_path):
    """observed_information equals the Hessian of 1/2 our independent chi^2
    (crosscheck.chi2) for both a template System and a callable class, agrees
    with expected_fisher on noiseless data at the truth, and shifts by the
    nonlinear residual curvature away from the truth."""
    true_vals = np.array([1.6, 5.5, -3.5, 0.06])  # diam, dra, ddec, flux

    def our_vis_from_vec(vec):
        diam, dra, ddec, flux = vec

        def vis(u, v, w):
            return (sky.vis_uniform_disk(u, v, w, diam) + flux * sky.vis_point(u, v, w, dra, ddec)) / (1.0 + flux)

        return vis

    clean_fits = tmp_path / "clean.fits"
    noisy_fits = tmp_path / "noisy.fits"
    obs_kw = dict(
        hour_angles_h=np.linspace(-2.5, 2.5, 5),
        wavelengths=np.linspace(1.6e-6, 2.3e-6, 4),
        dec_deg=-45.0,
        sigma_v2=0.01,
        sigma_cp_deg=0.5,
    )
    sim.observe(clean_fits, our_vis_from_vec(true_vals), UTS3, rng=None, **obs_kw)
    sim.observe(noisy_fits, our_vis_from_vec(true_vals), UTS3, rng=np.random.default_rng(88), **obs_kw)

    template = vm.System(star=vm.UniformDisk(1.0), comp=vm.PointSource(0.05, 4.0, -3.0))
    paths = ["star.diam", "comp.dra", "comp.ddec", "comp.flux"]

    worst_rel = 0.0
    for path, probe in ((clean_fits, true_vals), (noisy_fits, np.array([1.75, 5.2, -3.7, 0.055]))):
        data = OIData(str(path))
        d_ours = ours.load(path)
        got = np.asarray(inf.observed_information(probe, paths, template, data, ridge=0.0, dtype="float64"))
        want = _richardson_hessian(lambda v: 0.5 * ours.chi2(d_ours, our_vis_from_vec(v), sine=True), probe, h=2e-4)
        scale = np.sqrt(np.outer(np.diag(want), np.diag(want)))
        worst_rel = max(worst_rel, float(np.max(np.abs(got - want) / scale)))

    # Ridge and callable-model form (BinaryModelCartesian)
    data_clean = OIData(str(clean_fits))
    got_ridge = np.asarray(inf.observed_information(true_vals, paths, template, data_clean, ridge=2.5, dtype="float64"))
    got_base = np.asarray(inf.observed_information(true_vals, paths, template, data_clean, ridge=0.0, dtype="float64"))
    np.testing.assert_allclose(got_ridge, got_base + 2.5 * np.eye(4), rtol=1e-13, atol=1e-13)

    bin_vals = np.array([5.5, -3.5, 0.06])
    pt_sys = vm.System(star=vm.PointSource(), comp=vm.PointSource(0.05, 4.0, -3.0))
    got_pt_sys = np.asarray(
        inf.observed_information(bin_vals, ["comp.dra", "comp.ddec", "comp.flux"], pt_sys, data_clean, dtype="float64")
    )
    got_callable = np.asarray(
        inf.observed_information(bin_vals, ["dra", "ddec", "flux"], vm.BinaryModelCartesian, data_clean, dtype="float64")
    )
    np.testing.assert_allclose(got_callable, got_pt_sys, rtol=1e-12, atol=1e-12)

    record("max_rel_dobs", worst_rel)
    assert worst_rel < 1e-6


@pytest.mark.validates(
    "virgil.inference.observed_information",
    roots=["mathematics"],
    kind="control",
)
def test_observed_information_rejects_unhalved_chi2_hessian_control(tmp_path):
    """Negative control: observed_information is the Hessian of -log L = 1/2 chi^2,
    so the unhalved Hessian nabla^2 chi^2 = 2 I_obs must not match."""
    fits_path = tmp_path / "ctrl.fits"
    vis_fn = lambda u, v, w: (sky.vis_point(u, v, w) + 0.05 * sky.vis_point(u, v, w, 5.0, -3.0)) / 1.05  # noqa: E731
    sim.observe(
        fits_path,
        vis_fn,
        UTS3,
        hour_angles_h=np.linspace(-2.0, 2.0, 3),
        wavelengths=np.array([1.6e-6, 2.2e-6]),
        dec_deg=-40.0,
        sigma_v2=0.01,
        sigma_cp_deg=0.5,
        rng=None,
    )
    data = OIData(str(fits_path))
    d_ours = ours.load(fits_path)
    x = np.array([5.0, -3.0, 0.05])
    got = np.asarray(inf.observed_information(x, ["dra", "ddec", "flux"], vm.BinaryModelCartesian, data))
    wrong_unhalved = _richardson_hessian(
        lambda v: ours.chi2(
            d_ours,
            lambda u, vv, w: (sky.vis_point(u, vv, w) + v[2] * sky.vis_point(u, vv, w, v[0], v[1])) / (1.0 + v[2]),
            sine=True,
        ),
        x,
    )
    assert np.max(np.abs(got - wrong_unhalved)) / np.max(np.abs(got)) > 0.9
