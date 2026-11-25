"""The S2 optimiser candidates (``alma_optim``) and the convergence evaluator (``alma_certify``) on 64^2 toys, CPU, no real data.

Two chains. The numpy one: a circulant normal operator (a stand-in for the Toeplitz term) with a dense reference, every method run for a second or two, the evaluator's decrement against a dense
solve, the certification of a polished optimum and the failure of a perturbed one, the step cap, the 'every later snapshot' rule and the joint (gain) objective. The virgil one (skipped without
virgil's Toeplitz term): a toy radio data set at 64^2 through ``build_s2_problem`` and ``VirgilObjective``, a run and its certification, and the ``FitResult`` rebuilt for ``kkt_report`` as the
gate does (vv ``alma_lambda.py`` ``fit_point``): our ``kkt_w`` is virgil's."""

import numpy as np
import pytest

import virgil_bridge.alma_certify as ac
import virgil_bridge.alma_optim as ao

pytestmark = pytest.mark.validates("virgil_bridge.alma_optim", roots=["self-consistency"], kind="guard")

N = 64
FLUX, WEIGHT = 1.0, 50.0


def toy(seed=0, gain=None, gain_true=1.0):
    """A ring on a floor seen through a low-pass normal operator; ``r = g T f_true + T^(1/2) eps``."""
    rng = np.random.default_rng(seed)
    ky, kx = np.meshgrid(np.fft.fftfreq(N), np.fft.rfftfreq(N), indexing="ij")
    kr = np.hypot(kx, ky)
    spectrum = 1e6 * np.exp(-((kr / 0.10) ** 2)) * (kr < 0.2)
    c = (np.arange(N) - N / 2) / N
    rr = np.hypot(c[:, None], c[None, :])
    truth = np.exp(-(((rr - 0.25) / 0.06) ** 2)) + 2e-3
    truth *= FLUX / truth.sum()
    ft = np.fft.rfft2(truth)
    noise = np.fft.irfft2(np.fft.rfft2(rng.standard_normal((N, N))) * np.sqrt(spectrum), s=(N, N))
    r = gain_true * np.fft.irfft2(ft * spectrum, s=(N, N)) + noise
    return ao.CirculantObjective((N, N), spectrum, r, FLUX, WEIGHT, gain=gain), truth.ravel()


@pytest.fixture(scope="module")
def problem():
    obj, truth = toy()
    f_ref, _ = ao.polish(obj, obj.flat(), rounds=100, cg_tol=1e-9)
    ev = ac.Evaluator(obj, reference=f_ref)
    return obj, truth, f_ref, ev


def test_entropy_prox_is_stationary():
    rng = np.random.default_rng(1)
    z, a, q = rng.normal(0, 3, 500), 10.0 ** rng.uniform(-3, 2, 500), 1e-3
    f = ao.entropy_prox(z, a, q)
    assert np.all(f > 0)
    ok = z / a > -600  # below that f underflows and is floored at 1e-300
    assert np.max((np.abs(a * np.log(f / q) + f - z) / (1 + np.abs(z)))[ok]) < 1e-9
    big = ao.entropy_prox(np.array([5000.0, -5000.0]), np.array([1.0, 1.0]), 1e-3)  # no overflow either way
    assert np.all(np.isfinite(big)) and big[0] > 1.0


def test_the_polished_reference_is_stationary_against_a_dense_solve(problem):
    obj, _, f_ref, _ = problem
    T = obj.dense()
    f = f_ref.copy()
    for _ in range(3):  # dense Newton steps from the polished point must not move it
        g = T @ f - obj.r + obj.entropy_grad(f)
        H = T + np.diag(obj.weight / (obj.flux * f))
        df = -np.linalg.solve(H, g)
        f = f + _boundary(f, df) * df
    assert np.max(np.abs(f - f_ref) / f_ref) < 1e-7
    g = T @ f_ref - obj.r + obj.entropy_grad(f_ref)
    assert np.sqrt(np.mean(g[f_ref > 1e-3 * f_ref.max()] ** 2)) / (WEIGHT / FLUX) < 1e-8


def _boundary(f, df):
    return ao._boundary_step(f, df)


def test_the_decrement_is_the_dense_one(problem):
    obj, truth, _, _ = problem
    f = truth * np.exp(0.3 * np.random.default_rng(2).standard_normal(truth.size))
    f *= FLUX / f.sum()
    ev = ac.Evaluator(obj, cg_rtol=1e-10)
    rec = ev.evaluate(f)
    T = obj.dense()
    g = T @ f - obj.r + obj.entropy_grad(f)
    D = 0.5 * g @ np.linalg.solve(T + np.diag(WEIGHT / (FLUX * f)), g)
    assert rec["cg"]["converged"]
    assert rec["decrement"] == pytest.approx(D, rel=1e-6)
    assert rec["decrement_threshold"] == pytest.approx(0.5 * 1e-8 * WEIGHT)
    assert not rec["certified"]


def test_the_optimum_certifies_and_a_perturbed_start_does_not(problem):
    obj, truth, f_ref, ev = problem
    rec = ev.evaluate(f_ref)
    assert rec["certified"], rec["tests"]
    assert rec["kkt_w"] < 1e-6 and rec["J_minus_Jstar"] == 0
    bad = ac.perturb_lognormal(f_ref, 0.5, np.random.default_rng(3))
    assert not ev.evaluate(bad)["certified"]
    near = f_ref * (1 + 1e-3 * np.random.default_rng(4).standard_normal(f_ref.size))  # close, not stationary
    assert not ev.evaluate(near)["certified"]


FAST = ("newton_cg", "lbfgsb_newton", "skilling_bryan")


@pytest.mark.parametrize("name", FAST)
def test_the_second_order_methods_certify_on_the_toy(problem, name, tmp_path):
    obj, _, f_ref, ev = problem
    res = ao.run(name, obj, obj.flat(), budget_s=8.0, out=tmp_path, snap_every=0.4)
    assert res.reason in ("budget", "stalled") and res.geq > 0 and res.snapshots
    records = ac.score_snapshots(ev, str(tmp_path))
    series = ac.certify_series(records)
    assert series["certified"], (name, res.reason, records[-1]["kkt_w"], records[-1]["decrement"], records[-1]["tests"])
    assert records[-1]["image_distance"] < 1e-5 and abs(records[-1]["J_minus_Jstar"]) < 1e-6


@pytest.mark.parametrize("name", ["lbfgsb", "lbfgs", "mirror", "mirror_accel", "fista", "adam", "softmax_lbfgs"])
def test_every_other_method_descends_and_snapshots(problem, name, tmp_path):
    obj, _, f_ref, ev = problem
    start = obj.flat()
    J0 = obj.value_grad(start)[0]
    res = ao.run(name, obj, start, budget_s=2.0, out=tmp_path, snap_every=0.5)
    assert np.all(np.isfinite(res.f)) and np.all(res.f > 0)
    assert obj.value_grad(res.f)[0] < J0
    assert res.snapshots and res.geq == res.grads + 2 * res.hvps
    rec = ev.evaluate(res.f)  # the evaluator accepts any iterate
    assert rec["J_minus_Jstar"] >= -1e-9


def test_a_step_cap_is_a_failed_run(problem):
    obj, _, _, _ = problem
    res = ao.run("newton_cg", obj, obj.flat(), budget_s=30.0, max_steps=2)
    assert res.capped and res.reason == "capped" and res.steps == 2
    assert ac.run_failed(res)


def test_certification_needs_every_later_snapshot():
    rec = lambda c, t: dict(certified=c, t=t, geq=10 * t)  # noqa: E731
    seq = [rec(False, 1), rec(True, 2), rec(False, 3), rec(True, 4), rec(True, 5)]
    out = ac.certify_series(seq)
    assert out["certified"] and out["index"] == 3 and out["t"] == 4
    assert not ac.certify_series(seq[:3])["certified"]
    assert not ac.certify_series([rec(True, 1), rec(False, 2)])["certified"]
    assert not ac.certify_series([])["certified"]


def test_a_cg_that_misses_its_tolerance_is_not_accepted(problem):
    obj, _, f_ref, _ = problem
    ev = ac.Evaluator(obj, cg_maxit=1)
    rec = ev.evaluate(f_ref * np.exp(0.01 * np.random.default_rng(5).standard_normal(f_ref.size)))
    assert not rec["cg"]["converged"] and not rec["tests"]["decrement"] and not rec["certified"]


def test_reference_restarts_and_perturbation_use_the_c3_tolerances(problem):
    obj, _, f_ref, ev = problem
    agree, dJ, dist = ac.reference_agreement(ev, f_ref, f_ref * (1 + 1e-9))
    assert agree and dJ < 1e-6 and dist < 1e-5
    assert ac.compare_fits(ev, f_ref, f_ref)["passes"]
    far = ac.perturb_lognormal(f_ref, 0.5, np.random.default_rng(6))
    assert not ac.compare_fits(ev, f_ref, far)["passes"]
    start = ac.gaussian_floor_start(16, 5.0, 0.25)
    assert start.sum() == pytest.approx(0.25) and start.min() > 0 and start.argmax() in (7 * 16 + 7, 7 * 16 + 8, 8 * 16 + 7, 8 * 16 + 8)


def test_a_two_phase_method_alternates_with_the_nuisance_step_in_a_joint_run(monkeypatch):
    calls = []
    real = ao.newton_nuisance
    monkeypatch.setattr(ao, "newton_nuisance", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    obj, _ = toy(seed=7, gain=1.0, gain_true=1.15)
    ao.run("lbfgsb_newton", obj, obj.flat(), budget_s=4.0, joint_rounds=10)
    assert len(calls) >= 5, len(calls)  # once per slice, not once at the end


def test_the_c3_flux_check_is_two_sided_and_restores_the_nuisances(problem):
    obj, _, f_ref, ev = problem
    below = ac.compare_fits(ev, 1.0005 * f_ref, f_ref)  # the 'restart' sits at a different J, by more than the threshold, whichever side
    assert abs(below["dJ"]) > ac.decrement_threshold(obj.weight) and not below["passes"]
    lower = ac.compare_fits(ev, f_ref, 1.0005 * f_ref)
    assert not lower["passes"]
    jobj, _ = toy(seed=7, gain=1.0, gain_true=1.15)
    jev = ac.Evaluator(jobj)
    jobj.set_theta(np.array([1.3]))
    ac.compare_fits(jev, jobj.flat(), jobj.flat(), theta_ref=np.array([1.0]), theta_other=np.array([1.1]))
    assert jobj.theta[0] == pytest.approx(1.3)


def test_g2_needs_a_reference_and_ignores_uncertified_duplicates():
    import importlib.util, pathlib

    spec = importlib.util.spec_from_file_location("alma_optim_s2", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "alma_optim_s2.py")
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)
    assert not drv.matches_reference(None, {"J_minus_Jstar": 0.0, "decrement_threshold": 1.0})  # no reference: no match
    assert not drv.matches_reference(1e-5, {"J_minus_Jstar": -1.0, "decrement_threshold": 1e-3})  # a much lower J is as wrong as a much higher one
    assert drv.matches_reference(1e-5, {"J_minus_Jstar": -1e-6, "decrement_threshold": 1e-3})
    row = lambda m, w, c, t, ok=True: dict(method=m, weight_mult=w, certified=c, t=t, matches_reference=ok)  # noqa: E731
    rows = [row("a", 1, True, 100.0), row("a", 30, True, 200.0), row("a", 30, False, None), row("a", 30, True, 900.0),  # an uncertified and a slower duplicate
            row("b", 1, True, 50.0), row("b", 30, True, 60.0, ok=False)]  # b does not match at 30
    out = drv.g2_select(rows)
    assert out["winner"] == "a" and out["median_t"] == {"a": 150.0}
    assert drv.g2_select([row("b", 1, True, 5.0, ok=False), row("b", 30, True, 5.0, ok=False)]) is None


def test_the_joint_objective_fits_its_gain_and_certifies_the_nuisance_block():
    obj, _ = toy(seed=7, gain=1.0, gain_true=1.15)
    res = ao.run("newton_cg", obj, obj.flat(), budget_s=10.0, joint_rounds=40)
    ev = ac.Evaluator(obj)
    rec = ev.evaluate(res.f, res.theta)
    # One gain against a free image flux is degenerate (the entropy pulls the flux down and the gain runs away, theta ~ 3 here): this test is about the nuisance block's
    # decrement at the joint optimum and the shape of the C3 starts, not about recovering the true gain, which IM Lup fixes with ``ref_obs``.
    assert res.theta[0] != 1.0
    assert rec["n_nuisance"] == 1 and rec["nuisance_decrement"] < rec["decrement_threshold"], rec
    starts = ac.nuisance_starts(obj, res.theta, gain_frac=0.05)
    assert starts[0][0] == pytest.approx(res.theta[0] * 1.05) and starts[1][0] == pytest.approx(res.theta[0] * 0.95)


# ---------------------------------------------------------------- the chain through virgil

NPIX, PIX, FLUX_JY = 64, 12.0, 0.15


def toy_columns(seed=3, nant=7, ntime=12, nscan=2, w0=3000.0, flux=FLUX_JY, rotate=7.3e-5):
    """Four ddids (two observations) of baselines x times x scans seeing a Gaussian through earth rotation: noise sd 1/sqrt(w) (as ``test_alma_lambda_rule``)."""
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(nant) for j in range(i + 1, nant)]
    q, ph = 10 ** rng.uniform(4.3, 6.0, len(pairs)), rng.uniform(0, np.pi, len(pairs))
    cols = {k: [] for k in "u v vis weight freq ddid obsid time ant1 ant2 scan".split()}
    for d in range(4):
        for sc in range(nscan):
            for t in range(ntime):
                T = sc * 1000 + t * 6.0
                for k, (a, b) in enumerate(pairs):
                    u, v = q[k] * np.cos(ph[k] + rotate * T), q[k] * np.sin(ph[k] + rotate * T)
                    w = w0 * np.exp(rng.normal(0, 0.2))
                    m = flux * np.exp(-((np.hypot(u, v) / 1.2e6) ** 2))
                    z = (rng.normal() + 1j * rng.normal()) / np.sqrt(w)
                    for key, val in zip(cols, (u, v, m + z, w, 2.3e11 + 1e9 * d, d, d // 2, T, a, b, sc)):
                        cols[key].append(val)
    return {k: np.array(v) for k, v in cols.items()}


@pytest.fixture(scope="module")
def virgil_chain(tmp_path_factory):
    pytest.importorskip("virgil.radio_toeplitz")
    pytest.importorskip("virgil.fitting")
    import jax

    jax.config.update("jax_enable_x64", True)
    import virgil_bridge.alma as va
    import virgil_bridge.alma_lambda as vl

    cols, path = toy_columns(), tmp_path_factory.mktemp("toy")
    for d in np.unique(cols["ddid"]):
        s = cols["ddid"] == d
        np.savez(path / f"ddid_{d:02d}.npz", **{k: (v[s] if k not in ("ddid", "obsid") else int(v[s][0])) for k, v in cols.items()})
    data = va.load_dir(path)
    ref, _ = vl.reference_obs(data)
    prob = ac.build_s2_problem(data, npix=NPIX, pixel_mas=PIX, flux=FLUX_JY, ref_obs=ref)
    w = 20.0
    return prob, ac.VirgilObjective(prob, w)


@pytest.mark.validates("virgil.imaging.MaxEntropy", roots=["self-consistency"], kind="guard")
def test_the_virgil_objective_is_chi2_plus_maxentropy_up_to_a_constant(virgil_chain):
    import jax.numpy as jnp
    from virgil.imaging import MaxEntropy

    prob, obj = virgil_chain
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(2):
        f = obj.flat() * np.exp(0.5 * rng.standard_normal(obj.n))
        f *= FLUX_JY / f.sum()
        x = jnp.sqrt(jnp.asarray(f)).reshape(NPIX, NPIX)
        theirs = 0.5 * prob.chi2({"pixels": x}) + float(MaxEntropy(obj.weight).value(prob.model(pixels=x)))
        diffs.append(obj.value_grad(f)[0] - theirs)
    assert diffs[0] == pytest.approx(diffs[1], abs=1e-6 * max(1.0, abs(diffs[0])))


def test_the_virgil_data_term_is_exactly_quadratic(virgil_chain):
    _, obj = virgil_chain
    rng = np.random.default_rng(1)
    f, v = obj.flat() * (1 + 0.1 * rng.random(obj.n)), 0.3 * obj.q * (2 * rng.random(obj.n) - 1)  # f + v stays positive: the model is linear in the flux there
    d0, g0 = obj.data_vg(f)
    d1, g1 = obj.data_vg(f + v)
    assert g1 - g0 == pytest.approx(obj.hvp(v), rel=1e-8, abs=1e-8 * np.abs(g1).max())
    assert d1 - d0 == pytest.approx(g0 @ v + 0.5 * v @ obj.hvp(v), rel=1e-7, abs=1e-8)


@pytest.mark.validates("virgil.fitting.kkt_report", roots=["self-consistency"], kind="guard")
def test_a_run_certifies_and_the_rebuilt_fit_result_gives_the_same_kkt_w(virgil_chain, tmp_path):
    prob, obj = virgil_chain
    res = ao.run("newton_cg", obj, obj.flat(), budget_s=60.0, out=tmp_path, snap_every=2.0, cg_tol=1e-4)
    f, _ = ao.polish(obj, res.f, rounds=60, cg_tol=1e-8)
    ev = ac.Evaluator(obj, reference=f)
    rec = ev.evaluate(f)
    assert rec["certified"], (rec["tests"], rec["kkt_w"], rec["decrement"])
    ours = ev.evaluate(res.f)
    rep = ac.virgil_kkt(obj, res.f)
    assert rep["kkt_w"] == pytest.approx(ours["kkt_w"], rel=1e-3, abs=1e-9)  # the same statistic, through virgil's kkt_report
    assert rep["want_grow"] == pytest.approx(ours["want_grow"], abs=1e-3)
    assert np.isfinite(rep["band_limit"])  # reported by virgil, not gating here
    assert ac.virgil_kkt(obj, f)["kkt_w"] < 1e-4


@pytest.mark.validates("virgil.fitting.kkt_report", roots=["self-consistency"], kind="guard")
def test_the_production_hook_runs_c1_to_c6_with_the_signature_dsharp_fit_calls(virgil_chain, tmp_path, monkeypatch):
    """``certify(prob=, values=, data=, result=, out=)`` as ``dsharp_fit.py --certify alma_certify:certify`` calls it, on a namespace shaped like ``build_production_problem``'s."""
    import inspect
    import types

    prob, obj = virgil_chain
    assert list(inspect.signature(ac.certify).parameters) == ["prob", "values", "data", "result", "out"]
    ns = types.SimpleNamespace(npix=prob.npix, pixel_mas=prob.pixel_mas, scene=prob.scene, nuis=prob.nuisance, priors=prob.priors, init={"pixels": obj.flat().reshape(prob.npix, prob.npix) ** 0.5},
                               terms=[prob.term], regs=[], model=prob.model, lam=obj.weight, flux=prob.flux, offset_bound_mas=20.0, offset_bound_obs=None, offset_bound_requested=20.0)
    res = ao.run("newton_cg", obj, obj.flat(), budget_s=30.0, cg_tol=1e-4)
    f, _ = ao.polish(obj, res.f, rounds=60, cg_tol=1e-8)
    values = {"pixels": np.sqrt(f).reshape(prob.npix, prob.npix)}
    monkeypatch.setenv(ac.ENV_RUNNER_UP, "newton_cg")
    monkeypatch.setenv(ac.ENV_BUDGET, "15")
    rep = ac.certify(prob=ns, values=values, data=prob.data, result={"args": {"optimiser": "lbfgsb"}}, out=tmp_path)
    assert set(("C1", "C2", "C3", "C4", "C5", "C6", "certified", "runner_up")) <= set(rep)
    assert rep["C1"]["passed"] and rep["C2"]["passed"], (rep["C1"], rep["C2"])
    assert set(rep["C3"]["restarts"]) == {"image_gaussian", "nuisance_plus", "nuisance_minus"}
    assert rep["C4"]["optimiser"] == "newton_cg" and "image" in rep["C4"]
    assert "rel_diff" in rep["C5"], rep["C5"]  # the direct NUFFT re-score ran
    assert "at_bound" in rep["C6"], rep["C6"]
    assert rep["certified"] == all(rep[c]["passed"] for c in ("C1", "C2", "C3", "C4", "C5", "C6"))
    import json

    json.dumps(rep, default=float)  # JSON-able as dsharp_fit's result.json needs


@pytest.mark.validates("virgil.fitting.kkt_report", roots=["self-consistency"], kind="guard")
def test_kkt_w_matches_virgils_when_the_iterate_flux_differs_from_the_reference_flux(virgil_chain):
    _, obj = virgil_chain
    f = obj.flat() * (1.0 + 0.5 * np.random.default_rng(2).random(obj.n)) * 1.15  # total flux about 1.4 x the reference flux
    assert f.sum() / obj.flux > 1.3
    ours = ac.Evaluator(obj).evaluate(f)["kkt_w"]
    theirs = ac.virgil_kkt(obj, f)["kkt_w"]
    assert ours == pytest.approx(theirs, rel=1e-3)


# ---------------------------------------------------------------- second review: step rules, joint slices, nuisance safety


class CosNuisance(ao.Objective):
    """One nuisance with ``v(theta) = -cos(theta)``: indefinite (v'' < 0) for theta in (pi/2, 3pi/2), minimum at 0. The image does not enter."""

    def __init__(self, theta, lo=-np.inf, hi=np.inf):
        super().__init__(4, 1.0, 1.0)
        self.theta, self._b = np.array([float(theta)]), (np.array([lo]), np.array([hi]))

    def data_vg(self, f):
        return -float(np.cos(self.theta[0])), np.zeros(self.n)

    def hvp(self, v):
        return np.zeros_like(v)

    def set_theta(self, theta):
        self.theta = np.asarray(theta, float).copy()

    def theta_bounds(self):
        return self._b

    def nuisance_gh(self, f):
        return np.array([np.sin(self.theta[0])]), np.array([[np.cos(self.theta[0])]])


def test_the_nuisance_step_damps_to_positive_definite_and_does_not_climb():
    obj = CosNuisance(2.0)  # H = cos 2 < 0
    v0 = obj.data_vg(None)[0]
    ao.newton_nuisance(obj, np.ones(4))
    assert obj.data_vg(None)[0] < v0 and obj.theta[0] < 2.0


def test_the_nuisance_step_never_evaluates_outside_the_bounds():
    obj = CosNuisance(2.0, lo=1.99, hi=3.0)
    ao.newton_nuisance(obj, np.ones(4))
    assert 1.99 <= obj.theta[0] <= 3.0


def test_an_indefinite_nuisance_hessian_fails_the_nuisance_decrement():
    D, n = ac.nuisance_decrement(CosNuisance(2.0), np.ones(4))
    assert n == 1 and D == np.inf
    D, n = ac.nuisance_decrement(CosNuisance(0.3), np.ones(4))
    assert np.isfinite(D) and D > 0


def test_snapshots_after_the_budget_do_not_count_toward_certification():
    rows = [dict(certified=False, t=1.0, geq=1), dict(certified=True, t=2.0, geq=2), dict(certified=False, t=9.0, geq=3)]
    assert not ac.certify_series(rows)["certified"]
    assert ac.certify_series(rows, budget_s=5.0)["certified"]


def test_the_accelerated_mirror_makes_progress_when_the_entropy_dominates():
    obj, _ = toy(seed=2)
    obj.weight = 30 * WEIGHT  # the entropy gradient outweighs the data gradient: a restart test on the data gradient alone looped at k == 1 with no step
    res = ao.run("mirror_accel", obj, obj.flat() * (1 + 0.3 * np.cos(np.arange(obj.n))), budget_s=1.0)
    assert res.steps > 20


def test_the_mirror_step_comes_from_the_diagonal_of_the_curvature():
    obj, _ = toy(seed=2)
    assert float(obj.diag()[0]) < obj.lipschitz() / 10  # the l1-infinity constant is far smaller than lambda_max
    res_d = ao.run("mirror", obj, obj.flat(), budget_s=1.0)
    res_l = ao.run("mirror", obj, obj.flat(), budget_s=1.0, base="lambda")
    assert obj.value_grad(res_d.f)[0] < obj.value_grad(res_l.f)[0]


def test_the_softmax_control_has_a_free_flux(problem):
    obj, _, f_ref, ev = problem
    res = ao.run("softmax_lbfgs", obj, 0.5 * obj.flat(), budget_s=3.0)  # a start at the wrong flux
    assert res.f.sum() > 0.6 * FLUX and abs(res.f.sum() - f_ref.sum()) < 0.2 * FLUX
    assert ev.evaluate(res.f)["J_minus_Jstar"] < 1.0


@pytest.mark.parametrize("name", ["lbfgsb_newton", "adam", "fista"])
def test_the_joint_slices_return_to_the_nuisance_step_and_end_inside_the_budget(name, monkeypatch, tmp_path):
    calls = []
    real = ao.newton_nuisance
    monkeypatch.setattr(ao, "newton_nuisance", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    obj, _ = toy(seed=7, gain=1.0, gain_true=1.15)
    res = ao.run(name, obj, obj.flat(), budget_s=3.0, joint_rounds=6, out=tmp_path, snap_every=0.3)
    assert len(calls) >= 3, (name, len(calls))
    assert res.theta[0] != 1.0
    assert max(s["t"] for s in res.snapshots) <= 3.0 * (1 + 1e-3)
