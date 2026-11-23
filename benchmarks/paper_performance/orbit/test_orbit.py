"""Light checks of the orbit comparison files (no sampling): data manifest, Kepler constant, gate results, cases.yml."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import common  # noqa: E402
import make_data as md  # noqa: E402


def test_data_matches_manifest():
    assert common.verify_manifest(HERE / "manifest.json", md.DATA) == []


def test_kepler_third_law():
    # 1 au, 1 Msun: P = 365.25 d would give a = 1 au; IAU nominal constants give 365.2568.. d for GM = 1.3271244e20
    a_au = (1.0 * 365.2568983**2 / md.KEPLER_K) ** (1.0 / 3.0)
    assert abs(a_au - 1.0) < 1e-6
    assert abs(md.a_mas(P=200.0, M_tot=2.0, plx=20.0) - 20.0 * (2.0 * 200.0**2 / md.KEPLER_K) ** (1 / 3)) < 1e-12


def test_gates_recorded_and_passing():
    r = json.loads((HERE / "gates_result.json").read_text())
    for k in ("gate1_virgil_pass", "gate1_octofitter_pass", "gate2_virgil_pass", "gate2_octofitter_pass"):
        assert r[k] is True, k
    assert r["gate3_octofitter_rel_diff_vs_neutral"] < 1e-8
    assert r["gate4_reader"]["pass"] is True and r["gate4_reader"]["index_zero"] == 0


def test_cases_yaml_parses():
    import yaml

    c = yaml.safe_load((HERE / "cases.yml").read_text())
    assert set(c["cases"]) == {"O-A-sim", "O-A-sim-pulls", "O-A-real", "O-B-sim"}
    assert c["code_versions"]["octofitter"].startswith("9.0.0")


# ---- chain starts (run_virgil.py "chain starts"; CHANGES.md 2026-11-23) ----------------------------------------------

import numpy as np  # noqa: E402
import pytest  # noqa: E402

CHAINS = 4
A_FILES = [("O-A-sim", "astrometry_O-A-sim.csv"), ("O-A-real", "astrometry_O-A-real.csv")] + [
    ("O-A-sim-pulls", f"astrometry_O-A-sim-pulls_seed{k:02d}.csv") for k in range(md.N_PULL_SEEDS)]


def _x64():
    import jax

    jax.config.update("jax_enable_x64", True)
    import run_virgil as rv

    return rv


def _assert_starts_ok(rv, model, priors, vals, report):
    """Every sampled site of every chain strictly inside its prior's support and inside the [q, 1-q] interval; finite
    unconstrained values, log density and gradient; the report agrees with the values."""
    import jax
    from numpyro.infer.util import potential_energy

    assert len(vals) == CHAINS
    for v in vals:
        for site, prior in rv.sampled_sites(priors).items():
            x = np.asarray(v[site], float)
            if x.ndim:  # angle vector: any non-zero vector of the plane; the starts are unit vectors
                assert np.linalg.norm(x) == pytest.approx(1.0)
                continue
            lo, hi = float(prior.support.lower_bound), float(prior.support.upper_bound)
            qlo, qhi = rv.prior_interval(prior)
            assert lo < qlo <= x <= qhi < hi, (site, x)
    assert report["start_P_d"] == [v["P"] for v in vals]
    wlo, whi = report["period_window_d"]
    assert wlo < report["trial_period_range_d"][0] <= report["trial_period_range_d"][1] < whi
    init = rv.init_params(model, priors, vals, seed=1)
    assert set(init) == set(rv.sampled_sites(priors))
    for k in range(CHAINS):
        z = {s: u[k] for s, u in init.items()}
        assert all(np.all(np.isfinite(np.asarray(u))) for u in z.values())
        pe, grad = jax.value_and_grad(lambda z: potential_energy(model, (), {}, z))(z)
        assert np.isfinite(float(pe)), (k, pe)
        assert all(np.all(np.isfinite(np.asarray(g))) for g in grad.values()), (k, grad)


@pytest.mark.parametrize("phase", ["M0", "theta"])   # M0: matched, theta: native
@pytest.mark.parametrize("case,file", A_FILES, ids=[f[:-4].replace("astrometry_", "") for _, f in A_FILES])
def test_level_a_starts_inside_every_prior_and_finite(case, file, phase):
    rv = _x64()
    info = rv.case_info(case)
    ds = rv.read_astrometry(rv.DATA / file)
    pos = rv.positions_from(ds, info["t_ref"])
    model, priors, _ = rv.level_a_model(pos, info["plx"], info["t_ref"], info["p_prior"], phase)
    vals, report = rv.level_a_values(priors, pos, ds["epoch"], info["plx"], phase, info["t_ref"], CHAINS)
    _assert_starts_ok(rv, model, priors, vals, report)
    again, _ = rv.level_a_values(priors, pos, ds["epoch"], info["plx"], phase, info["t_ref"], CHAINS)
    assert all(np.array_equal(a[k], b[k]) for a, b in zip(vals, again) for k in a)  # deterministic


@pytest.mark.parametrize("phase", ["M0", "theta"])
def test_level_b_starts_inside_every_prior_and_finite(phase):
    rv = _x64()
    info = rv.case_info("O-B-sim")
    epochs = rv.read_epochs(independent_cp=True)
    model, priors, _ = rv.level_b_model(epochs, info["plx"], info["t_ref"], info["p_prior"], phase)
    vals, report = rv.level_b_values(priors, epochs, info["plx"], info["t_ref"], phase, CHAINS)
    assert "flux" in rv.sampled_sites(priors)
    _assert_starts_ok(rv, model, priors, vals, report)


def test_trial_periods_narrow_and_wide_windows():
    """A window narrower than one phase-coherence step (O-A-real) and a wide search (the simulated cases): the trials are
    strictly inside the prior's central interval, never its end points, at least ``n_min`` of them, and no coarser than
    virgil's period_grid."""
    rv = _x64()
    from virgil.orbits import period_grid

    n_edges = {}
    for case, file in A_FILES[:2]:
        times = rv.read_astrometry(rv.DATA / file)["epoch"]
        prior = rv.make_priors(rv.case_info(case)["p_prior"], "M0")["P"]
        lo, hi = rv.prior_interval(prior)
        assert float(prior.support.lower_bound) < lo < hi < float(prior.support.upper_bound)
        edges = period_grid(times, lo, hi)
        n_edges[case] = edges.size
        for n_min in (1, CHAINS, 50):
            trials = rv.trial_periods(times, prior, n_min=n_min)
            assert trials.size == max(edges.size - 1, n_min)
            assert np.all(np.diff(trials) > 0) and lo < trials[0] and trials[-1] < hi
            assert np.max(np.abs(np.diff(1.0 / trials)), initial=0.0) <= np.max(np.abs(np.diff(1.0 / edges))) * (1 + 1e-9)
    # O-A-real is the narrow case (period_grid returns only its two end points); the simulated prior is the wide one
    assert n_edges["O-A-real"] == 2 and n_edges["O-A-sim"] > CHAINS + 1
    assert rv.trial_periods(times, prior)[0] == pytest.approx(rv.REAL["P_mean"], abs=1e-6)


def test_start_values_clip_into_the_prior_and_report_it():
    """Orbit elements on or beyond a prior bound (e = 0, a face-on orbit, a mass above the prior, a period on its bound)
    are moved to the [q, 1-q] interval and named; check_starts refuses a value left on a bound or a missing site."""
    rv = _x64()
    from virgil.orbits import KeplerOrbit

    priors = rv.make_priors(rv.case_info("O-A-sim")["p_prior"], "M0")
    a_big = md.a_mas(P=250.0, M_tot=50.0, plx=rv.SIM_PLX)
    orbit = KeplerOrbit(250.0, 10.0, 0.0, 0.0, 30.0, 40.0, a_big, md.T_REF)
    vals, clipped = rv.start_values(orbit, priors, rv.SIM_PLX, "M0", md.T_REF)
    assert sorted(clipped) == ["M_tot", "P", "e", "inc"]
    for k in clipped:
        assert vals[k] in rv.prior_interval(priors[k])
    rv.check_starts(priors, [vals])
    for site, bad in (("P", 250.0), ("e", 0.0), ("inc", 180.0), ("M_tot", 6.0), ("omega_vec", np.zeros(2))):
        with pytest.raises(ValueError, match=site):
            rv.check_starts(priors, [{**vals, site: bad}])
    with pytest.raises(ValueError, match="no value for M0_vec"):
        rv.check_starts(priors, [{k: v for k, v in vals.items() if k != "M0_vec"}])


def test_octofitter_initialisation_is_seeded():
    """The Julia runners pass an explicit generator to initialize! (its default is the unseeded global one)."""
    for f in ("run_octofitter.jl", "profile_octofitter.jl"):
        src = (HERE / f).read_text()
        assert "initialize!(model;" not in src and "initialize!(Random.Xoshiro(" in src, f
