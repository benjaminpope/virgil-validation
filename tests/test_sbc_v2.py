"""Simulation-based calibration of virgil.likelihood.numpyro_model, version 2
(scripts/sbc_v2.py; criteria and design registered in design/sbc_v2.yml).

Chains start from virgil's grid fit on each replicate's own data, jittered;
the mass matrix is dense on (pa_vec, sep, flux) with a narrower angle ring;
a bright and a low-SNR regime; the joint log-likelihood rank as a sixth test
quantity. This file reads the committed summary (it skips until the OzSTAR
job sbc_v2 has run and been aggregated), and checks the registered file, the
aggregator and the task files on synthetic input.
"""

import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from evidence.meta import virgil_source
from evidence.plugin import record

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "trust" / "campaigns" / "sbc_v2.json"


def script():
    spec = importlib.util.spec_from_file_location("sbc_v2", ROOT / "scripts" / "sbc_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not SUMMARY.exists(), reason="the SBC v2 campaign has not run (ozstar_scripts job sbc_v2)")
@pytest.mark.validates("virgil.likelihood.numpyro_model", roots=["statistics"], tier="C")
def test_sbc_v2_ranks_are_uniform_and_the_sampler_healthy():
    s = script()
    summary = json.loads(SUMMARY.read_text())
    assert summary["criteria_hash"] == s.criteria_hash(), "criteria changed after the campaign ran"
    assert summary["design_hash"] == s.design_hash(), "the experiment changed after the campaign ran"
    installed = virgil_source()["commit"]
    if summary.get("virgil_commit") != installed:
        pytest.skip(f"the campaign ran on virgil {str(summary.get('virgil_commit'))[:7]}, not the installed "
                    f"{str(installed)[:7]}: rerun it")
    for regime, e in summary["regimes"].items():
        record(f"{regime}_replicates", e["n"])
        record(f"{regime}_unhealthy_fraction", e["health"]["unhealthy_fraction"])
        assert e["enough"], f"{regime}: {e['n']} replicates"
        assert e["health"]["pass"], f"{regime}: sampler health {e['health']}"
    assert not summary["rejected"], f"rank tests rejected: {summary['rejected']}"
    assert summary["pass"]


@pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")
def test_sbc_v2_registered_file_is_what_the_script_runs():
    """The hash is the committed file's; the file's noise levels are v1's
    (the script reuses v1's simulator); the family is the stated 60 tests."""
    s = script()
    spec = s.spec()
    registered = (ROOT / "design" / "sbc_v2.yml").read_bytes()
    addendum = (ROOT / "design" / "sbc_v2_addendum.yml").read_bytes()
    assert s.criteria_hash() == hashlib.sha256(registered + b"\n" + addendum).hexdigest()
    assert "low" not in spec["regimes"] and set(spec["regimes"]) == {"bright", "low_astrometric"}
    assert spec["regimes"]["low_astrometric"]["astrometry_sigma_mas"] == 0.5
    assert spec["generative"]["sigma_v2"] == s.v1.SIGMA_V2 and spec["generative"]["sigma_cp_deg"] == s.v1.SIGMA_CP_DEG
    assert (spec["generative"]["diam"]["lo"], spec["generative"]["diam"]["hi"]) == s.v1.BOUNDS["diam"]
    assert (spec["generative"]["sep"]["lo"], spec["generative"]["sep"]["hi"]) == s.v1.BOUNDS["sep"]
    assert spec["regimes"]["bright"]["flux_lo"] == s.v1.BOUNDS["flux"][0] and spec["regimes"]["bright"]["flux_hi"] == s.v1.BOUNDS["flux"][1]
    assert (spec["regimes"]["low_astrometric"]["flux_lo"], spec["regimes"]["low_astrometric"]["flux_hi"]) == (1e-3, 2e-2)
    assert spec["draws_per_replicate"] == 99 and spec["family_alpha"] == 0.01
    n_tests = len(spec["regimes"]) * len(spec["quantities"]) * len(spec["rank_tests"])
    assert n_tests == 60
    for r in spec["regimes"].values():
        assert r["tasks"] * r["replicates_per_task"] >= spec["counts"]["min_replicates_per_regime"] >= 1000
    health = spec["sampler_health"]
    assert (health["max_rhat"], health["min_bulk_ess"]) == (1.01, 400)
    assert spec["sampler"]["ring_width"] < 0.25 and set(spec["sampler"]["dense_mass"]) == {"pa_vec", "sep", "flux"}
    rng = np.random.default_rng(0)
    for regime in ("bright", "low_astrometric"):
        flux = [s.draw_truth(rng, regime)["flux"] for _ in range(2000)]
        lo, hi = spec["regimes"][regime]["flux_lo"], spec["regimes"][regime]["flux_hi"]
        assert lo <= min(flux) and max(flux) <= hi
        assert abs(np.mean(np.log(flux)) - (np.log(lo) + np.log(hi)) / 2) < 0.1  # log-uniform


def _rep(seed, regime, rank, healthy=True):
    s = script()
    q = s.spec()["quantities"]
    return {"seed": seed, "regime": regime, "truth": {}, "ranks": {k: rank(k) for k in q},
            "ess": {k: 1e3 if healthy else 10.0 for k in q}, "rhat": {k: 1.0 for k in q},
            "divergences": 0, "fit_converged": True}


def _write(path, regime, task, reps, commit="a" * 40, **over):
    s = script()
    record_ = {"regime": regime, "task": task, "virgil_commit": commit, "criteria_hash": s.criteria_hash(),
               "design_hash": s.design_hash(), "replicates_planned": len(reps), "override": None, "virgil": "x",
               "complete": True, "replicates": reps, **over}
    (pathlib.Path(path) / f"task_{regime}_{task:04d}.json").write_text(json.dumps(record_))


def _fast(monkeypatch, s):
    full = s.spec()
    monkeypatch.setattr(s, "spec", lambda: {**full, "ecdf_simulations": 300})


def _run(*dirs):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "sbc_v2.py"), "aggregate", *map(str, dirs)],
                          capture_output=True, text=True)


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_sbc_v2_aggregate_passes_uniform_ranks_and_fails_bias_and_ill_health(tmp_path, monkeypatch):
    """Uniform ranks in both regimes pass; a biased flux rank in the low
    regime is rejected under Holm; a regime with 10% unhealthy replicates
    fails the sampler-health criterion whatever its ranks."""
    s = script()
    _fast(monkeypatch, s)
    rng = np.random.default_rng(5)
    uniform = lambda n, regime, start: [_rep(start + i, regime, lambda k: int(rng.integers(0, 100))) for i in range(n)]  # noqa: E731

    def analyse(by_regime):
        return s.analyse(by_regime, s.spec())

    good = analyse({"bright": uniform(1000, "bright", 0), "low_astrometric": uniform(1000, "low_astrometric", 10_000)})
    assert good["tests"] == 60 and good["pass"], (good["rejected"], good["regimes"])

    low = uniform(1000, "low_astrometric", 10_000)
    for r in low:  # flux ranks pushed up: the posterior sits below the truth
        r["ranks"]["flux"] = int(np.clip(rng.normal(70, 20), 0, 99))
    biased = analyse({"bright": uniform(1000, "bright", 0), "low_astrometric": low})
    assert not biased["pass"] and any(k.startswith("low_astrometric:flux:") for k in biased["rejected"])
    assert not any(k.startswith("bright:") for k in biased["rejected"])

    sick = uniform(1000, "low_astrometric", 10_000)
    for r in sick[:100]:
        r["ess"]["diam"] = 50.0
    ill = analyse({"bright": uniform(1000, "bright", 0), "low_astrometric": sick})
    assert not ill["rejected"] and not ill["regimes"]["low_astrometric"]["health"]["pass"] and not ill["pass"]
    assert ill["regimes"]["low_astrometric"]["health"]["unhealthy_fraction"] == 0.1

    divergent = uniform(1000, "low_astrometric", 10_000)
    for r in divergent[:30]:
        r["divergences"] = 3
    assert not analyse({"bright": uniform(1000, "bright", 0), "low_astrometric": divergent})["regimes"]["low_astrometric"]["health"]["pass"]

    few = analyse({"bright": uniform(900, "bright", 0), "low_astrometric": uniform(1000, "low_astrometric", 10_000)})
    assert not few["pass"] and not few["regimes"]["bright"]["enough"]
    assert not analyse({"bright": uniform(1000, "bright", 0)})["pass"]  # a missing regime


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_sbc_v2_aggregate_refuses_mixed_runs(tmp_path):
    """Replicates are counted once each, from one virgil commit, one
    registered file and one design; smoke runs are refused."""
    rank = lambda k: 50  # noqa: E731
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write(a, "bright", 0, [_rep(1, "bright", rank)])
    assert _run(a).returncode == 0
    assert "repeat a seed" in _run(a, a).stderr
    _write(b, "low_astrometric", 0, [_rep(2, "low_astrometric", rank)], commit="b" * 40)
    assert "different or unknown virgil commits" in _run(a, b).stderr
    (b / "task_low_astrometric_0000.json").unlink()
    _write(b, "low_astrometric", 0, [_rep(2, "low_astrometric", rank)], criteria_hash="0" * 64)
    assert "criteria changed" in _run(a, b).stderr
    (b / "task_low_astrometric_0000.json").unlink()
    _write(b, "low_astrometric", 0, [_rep(2, "low_astrometric", rank)], design_hash="0" * 16)
    assert "experiment changed" in _run(a, b).stderr
    (b / "task_low_astrometric_0000.json").unlink()
    _write(b, "low_astrometric", 0, [_rep(2, "low_astrometric", rank)], override=[10, 10, 2])
    assert "smoke run" in _run(a, b).stderr
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "no results found" in _run(empty).stderr


@pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")
def test_sbc_v2_tasks_write_incrementally_and_resume(tmp_path, monkeypatch):
    """A task writes after every replicate, resumes with only the missing
    seeds, and refuses to resume under another commit (a stub for NUTS)."""
    import argparse

    s = script()
    monkeypatch.setenv("PIN_COMMIT", "c" * 40)
    calls = []

    class TimeLimit(Exception):
        pass

    stop = {"after": 2}

    def stub(seed, regime, override):
        if stop["after"] is not None and len(calls) == stop["after"]:
            raise TimeLimit
        calls.append(seed)
        return _rep(seed, regime, lambda k: 1)

    args = argparse.Namespace(regime="low_astrometric", task=3, replicates=4, seed_base=100, warmup=0, samples=0, chains=0,
                              out=str(tmp_path))
    with pytest.raises(TimeLimit):
        s.run(args, replicate_fn=stub)
    saved = json.loads((tmp_path / "task_low_astrometric_0003.json").read_text())
    first = 100 + 10_000_000 + 3 * 4
    assert [r["seed"] for r in saved["replicates"]] == [first, first + 1] and saved["complete"] is False
    calls.clear()
    stop["after"] = None
    s.run(args, replicate_fn=stub)
    saved = json.loads((tmp_path / "task_low_astrometric_0003.json").read_text())
    assert calls == [first + 2, first + 3]
    assert [r["seed"] for r in saved["replicates"]] == [first + k for k in range(4)] and saved["complete"] is True
    monkeypatch.setenv("PIN_COMMIT", "d" * 40)
    with pytest.raises(SystemExit, match="another commit"):
        s.run(args, replicate_fn=stub)


@pytest.mark.slow
@pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")
def test_sbc_v2_smoke_run_makes_a_well_formed_replicate(tmp_path, monkeypatch):
    """One real replicate with a few draws: virgil's grid fit starts the
    chains, the dense-mass kernel runs, and all six ranks are in 0..99. The
    sampler is overridden, so the output is a smoke run and aggregate
    refuses it."""
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sbc_v2.py"), "run", "--regime", "low_astrometric", "--task", "0",
         "--replicates", "1", "--warmup", "10", "--samples", "10", "--chains", "2", "--out", str(tmp_path)],
        capture_output=True, text=True, env={**__import__("os").environ, "PIN_COMMIT": "e" * 40,
                                             "PYTHONPATH": str(ROOT / "src")})
    assert done.returncode == 0, done.stderr[-2000:]
    task = json.loads((tmp_path / "task_low_astrometric_0000.json").read_text())
    rep = task["replicates"][0]
    assert task["complete"] and task["override"] == [10, 10, 2]
    assert set(rep["ranks"]) == {"diam", "flux", "sep", "pa", "vis_scale", "loglike"}
    assert all(0 <= v <= 99 for v in rep["ranks"].values())
    assert rep["fit_converged"] and 1e-3 <= rep["truth"]["flux"] <= 2e-2
    assert "smoke run" in _run(tmp_path).stderr


@pytest.mark.validates("virgil.likelihood.numpyro_model", roots=["statistics"], kind="regression")
def test_sbc_v2_start_is_finite_when_the_grid_peaks_on_a_prior_bound(tmp_path):
    """Replicate 20261008 (bright) of the first run: the grid's best point
    sits at the edge of the offset grid, outside the separation prior, and
    the fit started there returned NaN, so every transition diverged. The
    start is now clipped inside the support, and the chains never start at NaN."""
    pytest.importorskip("virgil")
    from virgil.oidata import OIData

    m = script()
    rng = np.random.default_rng(20261008)
    truth = m.draw_truth(rng, "bright")
    m.observe(tmp_path / "sbc.fits", truth, rng)
    start, _, info = m.start_from_fit(OIData(str(tmp_path / "sbc.fits")), "bright")
    assert np.hypot(info["grid"]["comp.dra"], info["grid"]["comp.ddec"]) > 25.0
    assert all(np.isfinite(v) for v in start.values())
    inits = m.jittered_inits(start, "bright", rng, 4)
    assert all(np.all(np.isfinite(v)) for c in inits for v in c.values())


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_sbc_v2_registered_file_is_unchanged_by_the_addendum():
    """design/sbc_v2.yml is as first registered (commit "Register SBC v2"): the addendum
    changes the low regime in the merged spec only."""
    import yaml

    raw = yaml.safe_load((ROOT / "design" / "sbc_v2.yml").read_text())
    assert set(raw["regimes"]) == {"bright", "low"} and raw["regimes"]["low"]["flux_hi"] == 2.0e-2
    merged = script().spec()
    assert merged["regimes"]["bright"] == raw["regimes"]["bright"]
    for key in ("sampler", "sampler_health", "rank_tests", "quantities", "family_alpha", "draws_per_replicate"):
        assert merged[key] == raw[key]


@pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")
def test_sbc_v2_astrometry_term_and_jitter():
    """The position measurement is a unit-Gaussian whitening of (dra, ddec) from the sampled
    (sep, pa), and the chains' jitter in the addendum's regime is absolute and about 0.1 mas."""
    s = script()
    meas = np.array([3.0, -4.0])
    term = s.astrometry_term(meas, 0.5)
    r = np.asarray(term({"sep": 5.0, "pa": 143.130102354}))  # (3, -4) mas: sin pa = 0.6, cos pa = -0.8
    assert np.allclose(r, 0.0, atol=1e-6)
    r = np.asarray(term({"sep": 5.0, "pa": 143.130102354 + 1e-3}))
    assert 0 < np.linalg.norm(r) < 1e-3
    assert np.allclose(np.asarray(term({"sep": 6.0, "pa": 143.130102354})), [0.6 * 1.0 / 0.5, -0.8 / 0.5])
    rng = np.random.default_rng(0)
    start = {"diam": 1.0, "flux": 5e-3, "sep": 10.0, "pa": 40.0, "vis_scale": 1.0}
    inits = s.jittered_inits(start, "low_astrometric", rng, 4000)
    sep = np.array([c["sep"] for c in inits])
    ang = np.array([np.arctan2(c["pa_vec"][1], c["pa_vec"][0]) for c in inits])
    dra, ddec = sep * np.sin(ang), sep * np.cos(ang)
    p0 = np.deg2rad(40.0)
    assert abs(np.std(dra) - 0.1) < 0.01 and abs(np.std(ddec) - 0.1) < 0.01
    assert abs(np.mean(dra) - 10.0 * np.sin(p0)) < 0.01
