"""scripts/postagb_parametric.py: evidence, noise arms, conventions and selection rules on synthetic 4-telescope data (no file under ~/data)."""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
try:  # needs virgil main (MultipoleDisk, Harmonics); the CI pin of the orbit pages predates it, so these tests skip there (extra: postagb)
    import jax
    import jax.numpy as jnp
    import numpyro.distributions as dist
    import postagb_imaging as P  # noqa: E402
    import postagb_parametric as Q  # noqa: E402
    from _postagb_parametric_synth import CFG, FLUXES, REFS, make_data, start  # noqa: E402
except ImportError:
    P = Q = None
needs_virgil = pytest.mark.skipif(P is None, reason="needs virgil main (MultipoleDisk) and virgil-vlti with virgil_vlti.with_night_terms (install the postagb extra)")

# Guards our own script's evidence and conventions against closed forms and brute-force quadrature, not virgil against an independent result.
pytestmark = [needs_virgil, pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")]


@pytest.fixture(autouse=True)
def _small_quadrature(monkeypatch):
    monkeypatch.setattr(Q, "N_R", 24)  # compiles faster; the design's 64 is for production


def _truth(model_id="M2", **kw):
    st = start(**kw)
    return st, Q.build_scene(CFG, model_id, st)


def test_prior_curvature_by_autodiff_for_every_prior():
    """Lambda = -d^2 ln p / d theta^2: 1/sigma^2 for Normal and TruncatedNormal, 0 for Uniform, -1/theta^2 for LogUniform, the sin i prior's own."""
    from virgil.priors import IsotropicInclination

    assert Q.prior_curvature(dist.Normal(0.0, 2.0), 0.3)[0] == pytest.approx(0.25)
    assert Q.prior_curvature(dist.Normal(jnp.zeros(2), 0.5), np.array([0.1, 0.2])) == pytest.approx([4.0, 4.0])
    assert Q.prior_curvature(dist.TruncatedNormal(0.03, 0.002, low=0.0), 0.03)[0] == pytest.approx(250000.0)
    assert Q.prior_curvature(dist.Uniform(-6.0, 6.0), 1.0)[0] == pytest.approx(0.0)
    assert Q.prior_curvature(dist.LogUniform(0.01, 20.0), 0.5)[0] == pytest.approx(-4.0)
    inc = 45.0
    assert Q.prior_curvature(IsotropicInclination(0.0, 85.0), inc)[0] == pytest.approx((np.pi / 180.0) ** 2 / np.sin(np.radians(inc)) ** 2, rel=1e-6)
    # gp-polar's rule keeps Normal only
    assert Q._scene_rule_lambda(dist.TruncatedNormal(0.03, 0.002, low=0.0), 0.03)[0] == 0.0
    assert Q._scene_rule_lambda(dist.Normal(0.0, 2.0), 0.3)[0] == pytest.approx(0.25)


def test_laplace_and_importance_sampling_agree_with_quadrature():
    """Design test (a): with the ring fixed and f_env, f_bg free, ln Z by Laplace and by importance sampling equal brute-force quadrature of
    exp(-chi^2/2 + ln p) on a grid, to 0.05 and 0.1."""
    from virgil.fitting import fit
    from virgil.likelihood import whitened_residuals

    st, truth = _truth("M1", f_env=0.5, f_bg=0.1)
    data = make_data(truth, sigma_v2=0.004, sigma_cp_deg=0.5, seed=3)
    pri = {k: v for k, v in Q.priors_for(CFG, truth, "M1", st).items() if k in ("env.flux.ratio", "bg.flux.ratio")}
    res = fit(Q.build_scene(CFG, "M1", start(f_env=0.4, f_bg=0.15)), pri, data, max_steps=500)
    assert res.info["converged"]
    ev = Q.param_evidence(res.model, pri, data)
    assert ev["sign"] > 0 and ev["k"] == 2
    ok = Q.is_evidence(res.model, pri, data, ev, n=4000, chunk=100)
    assert ok["ess"] > 500

    th, sd = ev["theta"], ev["sd"]
    g = np.linspace(-9.0, 9.0, 121)
    xs, ys = th[0] + sd[0] * g, th[1] + sd[1] * g

    def resid2(f_env, f_bg):
        m = res.model.set("env.flux.ratio", f_env).set("bg.flux.ratio", f_bg)
        r = whitened_residuals(m, data)
        return jnp.sum(r**2)

    grid = jax.jit(jax.vmap(lambda a, b: -0.5 * resid2(a, b) + pri["env.flux.ratio"].log_prob(a) + pri["bg.flux.ratio"].log_prob(b)))
    aa, bb = np.meshgrid(xs, ys, indexing="ij")
    lp = np.concatenate([np.asarray(grid(jnp.asarray(aa.ravel()[i: i + 500]), jnp.asarray(bb.ravel()[i: i + 500]))) for i in range(0, aa.size, 500)])
    cell = (xs[1] - xs[0]) * (ys[1] - ys[0])
    z_quad = float(np.log(cell) + np.logaddexp.reduce(lp))
    assert ev["lnZ_lap"] == pytest.approx(z_quad, abs=0.05)
    assert ok["lnZ_is"] == pytest.approx(z_quad, abs=0.1)


def test_tighter_normal_prior_changes_lnZ_as_the_gaussian_formula_says():
    """Design test (b): one Normal prior made 10x tighter, at the same model: Delta ln Z = Delta ln p(x) - (1/2) Delta ln det(H) with the closed
    Normal log density and the rank-one change of the Hessian (H taken from the stored covariance, so the check does not reuse the determinant code)."""
    st, truth = _truth("M2")
    data = make_data(truth, sigma_v2=0.01, seed=5)
    pri = Q.priors_for(CFG, truth, "M2", st)
    ev1 = Q.param_evidence(truth, pri, data)
    sig_old, sig_new = 1.0, 0.1
    pri2 = dict(pri) | {"env.harmonics.re": dist.Normal(jnp.zeros(1), sig_new)}
    ev2 = Q.param_evidence(truth, pri2, data)
    i = ev1["names"].index("env.harmonics.re[0]")
    x = ev1["theta"][i]
    d_logp = -np.log(sig_new / sig_old) - x**2 / 2 * (1 / sig_new**2 - 1 / sig_old**2)
    h_old = np.linalg.inv(ev1["cov"])
    h_new = h_old.copy()
    h_new[i, i] += 1 / sig_new**2 - 1 / sig_old**2
    d_logdet = np.linalg.slogdet(h_new)[1] - np.linalg.slogdet(h_old)[1]
    assert ev2["lnZ_lap"] - ev1["lnZ_lap"] == pytest.approx(d_logp - 0.5 * d_logdet, abs=1e-6)


def test_scene_rule_differs_from_lap_when_a_non_normal_prior_has_curvature():
    st, truth = _truth("M1")
    data = make_data(truth, seed=1)
    pri = Q.priors_for(CFG, truth, "M1", st)
    ev = Q.param_evidence(truth, pri, data)
    assert ev["lnZ_lap"] != pytest.approx(ev["lnZ_scene_rule"], abs=1e-9)  # LogUniform and sin i have curvature that the gp-polar rule drops


def test_n2_reproduces_the_frozen_local_night_noise():
    """Design test (c): our N2 data have the same chi^2 and log-normaliser as the earlier local construction (frozen in test_postagb_vlti.py)."""
    from test_postagb_vlti import old_noise  # noqa: PLC0415
    from virgil.imaging import _log_norm
    from virgil.likelihood import whitened_residuals

    st, truth = _truth("M2")
    data = make_data(truth, seed=2)
    s = 1.3
    ours = Q.noise_arm(data, CFG, "N2", s=s)
    ref = old_noise(data, {"vis": s, "phi": s}, None, Q.NIGHTW)
    m = Q.build_scene(CFG, "M2", start(inc=30.0, f_env=0.3))
    r1, r2 = np.asarray(whitened_residuals(m, ours)), np.asarray(whitened_residuals(m, ref))
    assert r1 @ r1 == pytest.approx(r2 @ r2, rel=1e-10)
    assert _log_norm(m, ours) == pytest.approx(_log_norm(m, ref), rel=1e-10)


def test_n2_has_one_mode_per_night_and_baseline_and_per_night_and_triangle():
    data = make_data(None, nights=(0, 1, 2), frames=3)
    g, o = Q.night_modes(data)
    assert g.shape == (3 * 6, data.u.size) and o.shape == (3 * 4, np.size(data.phi))
    assert set(np.unique(g[g > 0])) == {0.03}
    assert np.allclose(np.unique(o[o > 0]), np.radians(2.0))
    assert np.all((g > 0).sum(axis=0) == 1) and np.all((o > 0).sum(axis=0) == 1)  # every sample is in exactly one cell
    assert Q.night_modes(data, 0.0, 0.0) == (None, None)
    assert Q.noise_arm(data, CFG, "N0") is data
    assert Q.noise_arm(data, CFG, "N2", s=1.2).gains is not None and Q.noise_arm(data, CFG, "N1", s=1.2).gains is None


def test_pa_plus_180_with_odd_harmonics_negated_is_the_same_model():
    """Design test (e), first half."""
    st, a = _truth("M3", pa=40.0, re=[0.5, 0.3], im=[-0.4, 0.2])
    b = Q.build_scene(CFG, "M3", start(pa=220.0, re=[-0.5, 0.3], im=[0.4, 0.2]))  # w1 -> -w1, w2 unchanged
    d = make_data(None)
    va = a.model(d.u, d.v, d.wavel)
    vb = b.model(d.u, d.v, d.wavel)
    assert np.allclose(np.asarray(va), np.asarray(vb), atol=1e-10)
    c = Q.build_scene(CFG, "M3", start(pa=220.0, re=[0.5, 0.3], im=[-0.4, 0.2]))  # without the sign change it is a different model
    assert not np.allclose(np.asarray(va), np.asarray(c.model(d.u, d.v, d.wavel)), atol=1e-3)


@pytest.mark.parametrize("w", [([0.5, 0.3], [-0.4, 0.2]), ([0.0, -0.8], [0.9, 0.1]), ([-1.2, 0.0], [0.1, 0.5])])
def test_derived_linear_harmonics_match_the_numerical_fourier_coefficients(w):
    """Design test (e), second half: (c_m, s_m) of ``derived`` against a DFT of the azimuthal factor on 720 azimuths."""
    _, m = _truth("M3", re=w[0], im=w[1])
    d = Q.derived(m, CFG)
    env = m.get("env")
    theta = np.linspace(0.0, 2 * np.pi, 720, endpoint=False)
    f = np.asarray(env.harmonics.factor(np.array([4.0]), theta)).reshape(-1)
    assert f.mean() == pytest.approx(1.0, abs=1e-12) and f.min() >= 0.0
    for k in (1, 2):
        assert d["c"][k - 1] == pytest.approx(2 / 720 * np.sum(f * np.cos(k * theta)), abs=1e-9)
        assert d["s"][k - 1] == pytest.approx(2 / 720 * np.sum(f * np.sin(k * theta)), abs=1e-9)
    assert d["a1"] == pytest.approx(np.hypot(d["c"][0], d["s"][0]))


@pytest.mark.parametrize("pa", [20.0, 100.0, 170.0])
def test_bright_point_is_on_the_major_axis_for_a_real_w1(pa):
    """A real w1 > 0 puts the maximum at theta = 0, the end of the major axis at position angle ``pa`` (North through East)."""
    _, m = _truth("M2", pa=pa, inc=55.0, re=[1.0], im=[0.0])
    assert Q.derived(m, CFG)["bright_pa"] == pytest.approx(pa, abs=3.0)
    _, m2 = _truth("M2", pa=pa, inc=55.0, re=[-1.0], im=[0.0])
    assert abs((Q.derived(m2, CFG)["bright_pa"] - (pa + 180.0) + 180.0) % 360.0 - 180.0) < 3.0


def test_eligibility_flags_fire_on_constructed_cases():
    """Design test (f): E2 (a prior bound, at_bound), E1, the full_modulation flag and pa_unconstrained."""
    st, m = _truth("M2", re=[0.5], im=[-0.4])
    pri = Q.priors_for(CFG, m, "M2", st)
    names = ["env.pa"]
    ev = dict(sign=1.0, sd=np.array([5.0]), names=names)
    ok, why, flags = Q.eligible(dict(converged=True, at_bound=[]), m, pri, ev)
    assert ok and not why and not flags
    # E2: a flux ratio on its LogUniform floor, and virgil's own at_bound
    low = Q.build_scene(CFG, "M2", start(f_env=0.0101))
    ok, why, _ = Q.eligible(dict(converged=True, at_bound=[]), low, pri, ev)
    assert not ok and any("E2" in w and "env.flux.ratio" in w for w in why)
    ok, why, _ = Q.eligible(dict(converged=True, at_bound=["env.profile.radius"]), m, pri, ev)
    assert not ok and any("at_bound" in w for w in why)
    ok, _, _ = Q.eligible(dict(converged=True, at_bound=["env.pa"]), m, pri, ev)
    assert ok  # the PA is handled by the window, not by E2
    ok, why, _ = Q.eligible(dict(converged=False, at_bound=[], stop="time"), m, pri, ev)
    assert not ok and "E1" in why[0]
    # E3
    ok, why, _ = Q.eligible(dict(converged=True, at_bound=[]), m, pri, dict(sign=-1.0, sd=np.array([1.0]), names=names))
    assert not ok and any("E3" in w for w in why)
    # a full modulation (|w| large) and an unconstrained PA are flags, not exclusions
    full = Q.build_scene(CFG, "M2", start(re=[5.0], im=[0.0]))
    ok, _, flags = Q.eligible(dict(converged=True, at_bound=[]), full, pri, dict(sign=1.0, sd=np.array([40.0]), names=names))
    assert ok and flags == ["full_modulation", "pa_unconstrained"]


def test_scaling_the_errors_by_s_scales_own_chi2_and_leaves_raw_alone():
    """Design test (g): N1 at scale s divides the own chi^2 by s^2 (the independent rows exactly; the periodic penalty rows of the closure phases
    are the only other part) and does not change the raw chi^2."""
    st, truth = _truth("M2")
    data = make_data(truth, seed=4)
    m = Q.build_scene(CFG, "M2", start(inc=47.0, r=4.1))  # near the truth: the penalty rows stay at zero
    s = 1.7
    a, b = Q.chi2s(m, data, data), Q.chi2s(m, data, Q.noise_arm(data, CFG, "N1", s=s))
    assert b["raw"] == pytest.approx(a["raw"], rel=1e-12) and b["raw_v2"] == pytest.approx(a["raw_v2"], rel=1e-12) and b["raw_cp"] == pytest.approx(a["raw_cp"], rel=1e-12)
    assert b["own"] == pytest.approx(a["own"] / s**2, rel=2e-3)
    assert a["own"] == pytest.approx(a["raw"], rel=1e-12)  # under N0 the own likelihood is the raw one
    assert a["n"] == a["n_raw"]


def test_chi2r_rows_are_independent_observables_plus_one_penalty_row_per_closure_phase():
    """The answer to 'what are the 504 extra rows of chi2r on AI Sco': 4-telescope closure phases are correlated (3 independent of 4 triangles), so
    virgil whitens them to 3 independent rows per set and adds one periodic penalty row per closure phase. N = n_independent + n_phi."""
    data = make_data(None, frames=3)
    n_v2, n_cp = np.size(data.vis), np.size(data.phi)
    assert data.n_independent == n_v2 + n_cp * 3 // 4
    assert data.n_residuals == data.n_independent + n_cp


@pytest.mark.parametrize("model_id,n_grid,n_named", [("M0", 9, 1), ("M1", 18, 1), ("M2o", 18, 1)])
def test_starts_are_the_pre_registered_set(model_id, n_grid, n_named):
    st = Q.starts_for(CFG, model_id, refs=REFS, fluxes=FLUXES)
    assert len(st) == n_grid + n_named
    assert st[-1]["name"] == "deprins_image" and st[-1]["inc"] == 45.0 and st[-1]["pa"] == 60.0
    inc_pa = {(s["inc"], s["pa"]) for s in st if s["name"].startswith("grid")}
    assert inc_pa == {(i, p) for i in (20.0, 45.0, 65.0) for p in (0.0, 60.0, 120.0)}
    if model_id != "M0":
        assert {s["r"] for s in st if s["name"].startswith("grid")} == {0.7 * 4.0, 1.4 * 4.0}
    assert len(Q.starts_for(CFG, model_id, refs=REFS, fluxes=FLUXES, smoke=True)) == 4


def test_the_scene_and_priors_follow_the_design():
    st, m = _truth("M2o")
    pri = Q.priors_for(CFG, m, "M2o", st)
    assert set(pri) == {"env.flux.ratio", "env.flux.index", "bg.flux.ratio", "env.inc", "env.pa", "env.profile.radius", "env.profile.sigma",
                        "env.harmonics.re", "env.harmonics.im", "env.dra", "env.ddec"}  # bg index fixed: not a parameter
    assert pri["env.profile.radius"].support.lower_bound == 0.6 and pri["env.profile.radius"].support.upper_bound == 25.0
    assert (pri["env.pa"].low, pri["env.pa"].high) == (-50.0, 130.0)
    assert "bg.flux.index" in Q.priors_for(CFG, m, "M2o", st, "bgfree")
    m0 = Q.build_scene(CFG, "M0", start())
    assert "env.profile.radius" not in Q.priors_for(CFG, m0, "M0", st) and float(m0.get("env.profile.radius")) == Q.R_BLOB
    # warm start from a parent keeps the shared parameters and adds the new ones
    vals = {"env.inc": 30.0, "env.pa": 10.0, "env.profile.radius": 3.0, "env.profile.sigma": 0.7, "env.harmonics.re": [0.2], "env.harmonics.im": [0.1],
            "env.flux.ratio": 0.3, "env.flux.index": 1.0, "bg.flux.ratio": 0.05}
    s3 = Q.start_from_values(vals, "M3")
    assert s3["re"] == [0.2, 0.1] and s3["im"] == [0.1, 0.1] and s3["inc"] == 30.0
    s2o = Q.start_from_values(vals, "M2o")
    assert s2o["dra"] == 0.0 and s2o["r"] == 3.0


def _lad(model, arm, lnz, bic, flags=(), eligible=True, hole=0.9, inc=40.0, pa=50.0):
    best = None if not eligible else dict(lnZ_use=lnz, lnZ_lap=lnz, bic=bic, flags=list(flags), derived=dict(hole_contrast=hole, inc=inc, pa_mod180=pa), lnZ_scene_rule=lnz,
                                          own_chi2=1.0, raw_chi2=2.0, n_rows=100)
    return dict(model=model, arm=arm, best=best, modes=[])


def test_selection_rule_needs_both_thresholds_and_tests_against_the_last_accepted():
    L = {("M0", "N2"): _lad("M0", "N2", 0.0, 0.0), ("M1", "N2"): _lad("M1", "N2", 20.0, -50.0),
         ("M2", "N2"): _lad("M2", "N2", 22.0, -80.0),  # +2 over M1: not accepted over M1
         ("M2o", "N2"): _lad("M2o", "N2", 31.0, -95.0)}  # tested against the last accepted (M1): +11, -45: accepted
    sel = Q.select(L, "N2")
    assert [c["accepted"] for c in sel["chain"]] == [True, False, True] and sel["selected"] == "M2o" and sel["is_ring"]
    assert [c["against"] for c in sel["chain"]] == ["M0", "M1", "M1"]
    # only one of the two thresholds met
    L[("M1", "N2")] = _lad("M1", "N2", 20.0, -5.0)
    assert not Q.select(L, "N2")["chain"][0]["accepted"]
    L[("M1", "N2")] = _lad("M1", "N2", 4.9, -50.0)
    assert not Q.select(L, "N2")["chain"][0]["accepted"]
    # an unverified evidence puts the pair on lnZ_lap and flags it
    L[("M1", "N2")] = _lad("M1", "N2", 20.0, -50.0, flags=["evidence_unverified"])
    row = Q.select(L, "N2")["chain"][0]
    assert row["evidence_unverified"] and row["basis"] == "lnZ_lap"
    # a child without an eligible fit is never accepted; an ineligible parent lets the child in by default
    L[("M1", "N2")] = _lad("M1", "N2", 0.0, 0.0, eligible=False)
    assert Q.select(L, "N2")["chain"][0]["accepted"] is False
    L[("M0", "N2")] = _lad("M0", "N2", 0.0, 0.0, eligible=False)
    L[("M1", "N2")] = _lad("M1", "N2", 20.0, -50.0)
    assert Q.select(L, "N2")["chain"][0]["accepted"] is True
    # N0 is BIC only, with the De Prins -100 line recorded
    N0 = {("M0", "N0"): _lad("M0", "N0", 0.0, 0.0), ("M1", "N0"): _lad("M1", "N0", 0.0, -120.0)}
    row = Q.select(N0, "N0")["chain"][0]
    assert row["accepted"] and row["beyond_deprins_100"] and row["d_lnZ"] is None


def test_is_ring_needs_a_rung_above_M0_and_a_hole():
    L = {("M0", "N2"): _lad("M0", "N2", 0.0, 0.0, hole=0.0), ("M1", "N2"): _lad("M1", "N2", 20.0, -50.0, hole=0.3)}
    assert Q.select(L, "N2")["is_ring"] is False  # selected M1 but a shallow hole
    L[("M1", "N2")] = _lad("M1", "N2", 20.0, -50.0, hole=0.6)
    assert Q.select(L, "N2")["is_ring"] is True
    L[("M1", "N2")] = _lad("M1", "N2", 1.0, -50.0, hole=0.6)
    assert Q.select(L, "N2")["selected"] == "M0" and Q.select(L, "N2")["is_ring"] is False


def test_arm_dependent_flags_model_and_geometry():
    a = dict(selected="M2", best=dict(derived=dict(inc=40.0, pa_mod180=170.0)))
    assert not Q.arm_dependent(a, dict(selected="M2", best=dict(derived=dict(inc=50.0, pa_mod180=5.0))))  # 15 deg apart across the 180 wrap
    assert Q.arm_dependent(a, dict(selected="M1", best=a["best"]))
    assert Q.arm_dependent(a, dict(selected="M2", best=dict(derived=dict(inc=60.0, pa_mod180=170.0))))
    assert Q.arm_dependent(a, dict(selected="M2", best=dict(derived=dict(inc=40.0, pa_mod180=120.0))))


def test_same_mode_and_jackknife_sd():
    a = {"env.inc": 40.0, "env.pa": 10.0, "env.profile.radius": 3.0}
    assert Q.same_mode(a, {"env.inc": 43.0, "env.pa": 187.0, "env.profile.radius": 3.2})  # PA is taken mod 180
    assert not Q.same_mode(a, {"env.inc": 46.0, "env.pa": 10.0, "env.profile.radius": 3.0})
    assert not Q.same_mode(a, {"env.inc": 40.0, "env.pa": 10.0, "env.profile.radius": 3.5})
    x = [1.0, 2.0, 3.0, 4.0]
    assert Q.jackknife_sd(x, 0.0) == pytest.approx(np.sqrt(3 / 4 * 5.0))
    assert Q.jackknife_sd([179.0, 1.0, 178.0], 179.0, circular=True) == pytest.approx(Q.jackknife_sd([179.0, 181.0, 178.0], 0.0))


def test_json_encoding_of_infinities_and_arrays():
    d = Q._jsonable({"a": np.float64(1.5), "b": np.arange(2), "c": float("-inf"), "d": float("nan")})
    assert d == {"a": 1.5, "b": [0, 1], "c": "-Infinity", "d": None}
