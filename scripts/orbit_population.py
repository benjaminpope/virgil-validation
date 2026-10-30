"""Population injection-recovery gate for virgil's orbit engine (0.4.2).

The gate is pre-registered in design/orbit_gate_042.md (a verbatim copy of
virgil's file; section 6's YAML block is hashed exactly, and every task output
and the summary record the hashes). Populations, templates and seeds are in
``crosscheck.orbit_population`` (numpy only); this script runs virgil on them
and aggregates.

    python scripts/orbit_population.py criteria
    python scripts/orbit_population.py check-gate --virgil-repo ~/code/virgil --commit SHA
    python scripts/orbit_population.py draw --stratum S --index I --out DIR
    python scripts/orbit_population.py fit --task T --out DIR        # one system, T in 0..659
    python scripts/orbit_population.py oracle --task T --out DIR     # one B system, T in 0..239
    python scripts/orbit_population.py aggregate DIR [DIR ...] --summary FILE

Tasks run strata A, B, N, D in turn (200, 240, 100, 120 systems). ``fit``
runs strata A with the explicit prior of gate section 4 (fit_orbit_aliases
settings through OrbitPipeline, so its sampler gives the band posteriors), and
B, N, D with ``OrbitPipeline`` defaults as shipped. ``oracle`` is the same
default run with ``init=`` the truth (M4, M8).

Built on the existing injection-recovery stack: the hashed criteria and
Holm family of scripts/detection_campaign.py, the SBC chain-health rule of
scripts/sbc_v2.py, virgil.detection's FAP convention, and the per-task
resumable layout of scripts/pull_campaign.py. See the module docstring of
``fap`` for what is not implemented on virgil 0.4.x.
"""

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import pathlib
import re
import sys
import time

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import crosscheck.preregistration as prereg  # noqa: E402

GATE_FILE = ROOT / "design" / "orbit_gate_042.md"
CRITERIA_FILE = "design/orbit_gate_042.md"  # a verbatim copy of virgil main's file; runs record its git commit
RANKED = ("period", "ecc", "a_mas", "cos_inc", "flux")  # gate M2
UNLIMITED = 10_000  # max_modes: sample every band above posterior_p_min (stratum A)
LOG2 = math.log(2.0)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------- gate and hashes


def gate_text(text=None):
    return GATE_FILE.read_text() if text is None else text


def gate_block(text=None):
    """The YAML between the fences of section 6, exactly (lines joined by
    newlines, one trailing newline): the text the criteria hash covers."""
    m = re.search(r"## 6\. Criteria.*?```yaml\n(.*?)\n```", gate_text(text), re.S)
    if not m:
        raise SystemExit("section 6's YAML block was not found in the gate file")
    return m.group(1) + "\n"


def criteria_hash(text=None):
    return hashlib.sha256(gate_block(text).encode()).hexdigest()


def gate_hash(text=None):
    return hashlib.sha256(gate_text(text).encode()).hexdigest()


def spec(text=None):
    """The parsed criteria block, strictly: a block that does not parse as YAML
    fails the run (the file is fixed, with a change-log entry and a new pin;
    nothing is repaired at run time)."""
    import yaml

    try:
        return yaml.safe_load(gate_block(text))
    except yaml.YAMLError as err:
        raise SystemExit(f"the criteria block of {CRITERIA_FILE} is not valid YAML: {err}") from err


def design_hash():
    """The experiment apart from the criteria: population and simulator code,
    the geometry and visibility modules they use, and this whole script. Rows are combined only when it matches."""
    h = hashlib.sha256()
    for name in ("orbit_population.py", "array.py", "sky.py", "orbits.py", "oifits_writer.py"):
        h.update((ROOT / "src" / "crosscheck" / name).read_bytes())
    h.update(pathlib.Path(__file__).read_bytes())  # the whole script: fit, oracle and extraction code
    return h.hexdigest()[:16]


def virgil_commit():
    return _load("detection_campaign").virgil_commit()


def strata_sizes():
    s = spec()["strata"]
    return {k: v["n"] for k, v in s.items()}


def locate(task):
    """(stratum, index) of a fit task: A, B, N, D in turn."""
    n = strata_sizes()
    for stratum in ("A", "B", "N", "D"):
        if task < n[stratum]:
            return stratum, int(task)
        task -= n[stratum]
    raise SystemExit("task is beyond the last system")


# ------------------------------------------------------------- statistics


def binomial_q99(n, p):
    from scipy import stats

    return int(stats.binom.ppf(0.99, n, p))


def wilson(k, n, z=1.959964):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, centre - half), min(1.0, centre + half)]


def credible_set(p, m):
    """Indices of the smallest set of bands, by decreasing p, with total >= m,
    and its total mass."""
    p = np.nan_to_num(np.asarray(p, float), nan=0.0)
    order = np.argsort(-p, kind="stable")
    total, chosen = 0.0, []
    for i in order:
        chosen.append(int(i))
        total += p[i]
        if total >= m:
            break
    return chosen, float(total)


def poisson_binomial_p(hits, masses, sided):
    """Normal approximation to a Poisson-binomial sum: ``hits`` of systems
    whose claimed probabilities ``masses`` make the expected count their sum
    and variance sum m(1 - m). ``sided``: two, under (fewer hits than
    claimed). A zero variance gives p = 1 (hits = claim) or 0."""
    from scipy import stats

    masses = np.asarray(masses, float)
    mean, var = masses.sum(), (masses * (1 - masses)).sum()
    if var <= 0:
        return 1.0 if abs(hits - mean) < 1e-9 else 0.0
    z = (hits - mean) / math.sqrt(var)
    return float(2 * stats.norm.sf(abs(z)) if sided == "two" else stats.norm.cdf(z))


def fap(t_obs, t_null):
    """(1 + #{T_null >= T_obs}) / (1 + n_null), the convention of
    ``virgil.detection.DetectionMC.false_alarm_probability`` (orbit_dehack 2G),
    with its exact binomial 95% interval on the exceedance rate.

    Not wired into ``fit``: on virgil 0.4.x the cheap-tier statistic T and the
    DetectionMC user-statistic hook of orbit_dehack Stage 4 do not exist, so
    the baseline records the verdict virgil ships (the nats rule) and the
    aggregator applies M6 and M8 to it, saying so. A later stage that ships
    the FAP verdict supplies it through the same ``verdict`` field."""
    from scipy import stats

    t_null = np.asarray(t_null, float)
    k = int(np.sum(t_null >= t_obs))
    ci = stats.binomtest(k, t_null.size).proportion_ci(0.95, method="exact") if t_null.size else None
    return (1 + k) / (1 + t_null.size), (None if ci is None else [float(ci.low), float(ci.high)])


# ------------------------------------------------------------- fitting


def settings_for(system):
    """OrbitPipeline settings of a system's run. Stratum A states its prior
    explicitly (gate sections 3.2 and 4) and samples every band with
    p >= 0.01 whatever the shipped no-orbit verdict says (the smallest
    positive ``no_orbit_nats`` keeps the verdict from skipping the sampler);
    B, N and D are the defaults as shipped."""
    if system["kind"] != "A":
        return {}
    return dict(period_range=tuple(system["p_range"]), a_range=tuple(system["a_range"]),
                flux_range=tuple(system["flux_range"]), ecc_max=system["ecc_max"], primary_diam=None,
                temper=False, outlier_eps=0.0, drop_outlier_night=False, no_orbit_nats=sys.float_info.min,
                posterior_p_min=0.01, max_modes=UNLIMITED)


def oracle_init(system, t_ref):
    """``init=`` (the dict form) of ``fit_orbit_aliases``: the truth, with the periastron time
    relative to the engine's ``t_ref``."""
    o = system["orbit"]
    return dict(period=o["period"], dt_peri=o["t_peri"] - t_ref, ecc=o["ecc"], inc=o["inc_deg"], omega=o["omega"],
                Omega=o["Omega"], a_mas=o["a_mas"], flux=o["flux"])


def band_table(aliases, n_is):
    """The band rows of an ``AliasResult``: probability, log Z, its standard
    error (virgil's if it reports one, else the importance-sampling estimate
    sqrt(1/ESS - 1/n), from the effective sample size of the normalized
    weights, Kong 1992), flags."""
    rows = []
    for b in aliases.bands:
        se, source = None, None
        for name in ("log_z_se", "log_z_is_se"):
            if getattr(b, name, None) is not None:
                se, source = float(getattr(b, name)), "reported"
        if se is None and b.ess and np.isfinite(b.ess) and b.ess > 0:
            se, source = math.sqrt(max(1.0 / b.ess - 1.0 / n_is, 0.0)), "ess"
        rows.append(dict(n=int(b.n), p_lo=float(b.p_lo), p_hi=float(b.p_hi), p=_num(b.p), log_z=_num(b.log_z),
                         se_lnz=se, se_source=source, ess=_num(b.ess), flags=list(b.flags),
                         error=getattr(b, "error", "") or ""))
    return rows


def _num(x):
    x = float(x)
    return x if np.isfinite(x) else None


def thin(draws, ess_min):
    """Every ceil(n/ESS_min)-th of the n flattened draws (gate section 3.2)."""
    flat = np.asarray(draws, float).reshape(-1)
    step = max(1, math.ceil(flat.size / max(ess_min, 1.0)))
    return flat[::step]


def mixture_u(thinned, weights, truth, jitter):
    """Rank of the truth in the band mixture: sum_n w_n (c_n + U) / (L_n + 1),
    c_n the thinned draws of band n below the truth, U one uniform. For a
    calibrated posterior it is uniform on (0, 1) (randomized SBC rank,
    Talts et al. 2018)."""
    w = np.asarray(weights, float)
    w = w / w.sum()
    return float(sum(wn * (np.sum(d < truth) + jitter) / (len(d) + 1) for wn, d in zip(w, thinned)))


def mixture_interval(thinned, weights, level):
    """Central interval of the band mixture (draws weighted w_n / L_n)."""
    x = np.concatenate(thinned)
    w = np.concatenate([np.full(len(d), wn / len(d)) for wn, d in zip(weights, thinned)])
    order = np.argsort(x)
    x, cw = x[order], np.cumsum(w[order]) / w.sum()
    return [float(np.interp(q, cw, x)) for q in ((1 - level) / 2, (1 + level) / 2)]


def posterior_summary(run, system, bands, jitter):
    """SBC ranks, intervals and chain health from ``run/samples.h5``."""
    import h5py
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin

    truth = dict(period=system["orbit"]["period"], ecc=system["orbit"]["ecc"], a_mas=system["orbit"]["a_mas"],
                 cos_inc=system["orbit"]["cos_inc"], flux=system["orbit"]["flux"])
    p_of = {b["n"]: (b["p"] or 0.0) for b in bands}
    path = pathlib.Path(run) / "samples.h5"
    if not path.exists():
        return dict(sampled=[], reason="no samples.h5 (the sampler did not run)")
    per_band, rhat, ess, div = {}, {}, {}, 0
    with h5py.File(path, "r") as f:
        for name in f:
            if not name.startswith("posterior_n"):
                continue
            n = int(name[len("posterior_n"):])
            g = {k: np.asarray(f[name][k]) for k in f[name]}
            g["cos_inc"] = np.cos(np.deg2rad(g["inc"]))
            stats = f[f"sample_stats_n{n}"] if f"sample_stats_n{n}" in f else {}
            div += int(np.sum(np.asarray(stats["diverging"]))) if "diverging" in stats else 0
            for k in RANKED:
                rhat[f"{n}:{k}"] = float(np.max(split_gelman_rubin(g[k])))
                ess[f"{n}:{k}"] = float(np.min(effective_sample_size(g[k])))
            ess_min = min(ess[f"{n}:{k}"] for k in RANKED)
            per_band[n] = {k: thin(g[k], ess_min) for k in RANKED}
    sampled = sorted(per_band)
    if not sampled:
        return dict(sampled=[], reason="no sampled band in samples.h5")
    weights = [p_of.get(n, 0.0) for n in sampled]
    if sum(weights) <= 0:
        weights = [1.0] * len(sampled)
    w = [x / sum(weights) for x in weights]
    out = dict(sampled=sampled, weights=dict(zip(map(str, sampled), w)), divergences=div,
               rhat={k: v for k, v in rhat.items()}, ess={k: v for k, v in ess.items()},
               thinned={str(n): len(per_band[n]["period"]) for n in sampled}, u={}, interval95={}, interval68={}, covers95={})
    for k in RANKED:
        d = [per_band[n][k] for n in sampled]
        out["u"][k] = mixture_u(d, w, truth[k], jitter)
        out["interval95"][k] = mixture_interval(d, w, 0.95)
        out["interval68"][k] = mixture_interval(d, w, 0.68)
        out["covers95"][k] = bool(out["interval95"][k][0] <= truth[k] <= out["interval95"][k][1])
    return out


def truth_record(system):
    """The part of a system the aggregator needs, copied beside every fit."""
    o = system["orbit"] or {}
    return dict(stratum=system["stratum"], index=system["index"], kind=system["kind"], template=system["template"],
                n_nights=len(system["nights"]), band=system.get("band"), span_days=system["span_days"],
                schedule=system.get("schedule"), misspecified=system["misspecified"],
                misspecification=(system.get("misspecification") or {}).get("kind"),
                envelope_fraction=(system.get("misspecification") or {}).get("fraction"),
                near_twin=o.get("near_twin"), flux=o.get("flux"), c=system["noise"]["c"],
                n_redraws=system["n_redraws"], contaminants=system["contaminants"],
                primary_diam_mas=system["primary"]["diam_mas"], disc_on=system["primary"]["disc_on"],
                period=o.get("period"), ecc=o.get("ecc"), a_mas=o.get("a_mas"), cos_inc=o.get("cos_inc"),
                phase_coverage=_phase_coverage(system))


def _phase_coverage(system):
    """Fraction of the true orbit's period covered by the nights' phases (the
    data-sufficiency table's x axis): the largest phase gap's complement."""
    o = system["orbit"]
    if o is None:
        return None
    t = np.array([n["mjd_mean"] for n in system["nights"]])
    ph = np.sort(np.mod((t - o["t_peri"]) / o["period"], 1.0))
    gaps = np.diff(np.r_[ph, ph[0] + 1.0])
    return float(1.0 - gaps.max())


def run_system(part, stratum, index, out, keep_work=False):
    """One system: draw (truth first), write the files, run virgil, extract.
    The result dict is written by the caller; a virgil failure is recorded
    (status 'failed' with the error), never replaced by a retry."""
    import resource
    import shutil

    from crosscheck import orbit_population as op

    out = pathlib.Path(out)
    system = op.draw_system(stratum, index)
    tag = f"{stratum}{index:04d}"
    (out / f"truth_{tag}.json").write_text(json.dumps(system))  # before any fit
    work = out / "work" / f"{part}_{tag}"
    t0 = time.time()
    result = dict(status="ok", error=None, truth=truth_record(system))
    try:
        paths = op.write_system(system, work / "data")
        result.update(_virgil_run(part, system, paths, work))
    except Exception as err:  # recorded, not silenced: M7 counts it
        import traceback

        result.update(status="failed", error=f"{type(err).__name__}: {err}", traceback=traceback.format_exc()[-1500:])
    result["seconds"] = time.time() - t0
    result["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    return result


def _virgil_run(part, system, paths, work):
    import jax

    jax.config.update("jax_enable_x64", True)
    from virgil.aliases import AliasResult
    from virgil.pipeline import OrbitPipeline

    settings = settings_for(system)
    through = None
    if part == "oracle":
        through = "modes"
    run = work / "run"
    pipeline = OrbitPipeline.from_oifits([str(p) for p in paths], output=run, **settings)
    n_is = pipeline.settings["n_is"]
    if part == "oracle":
        init_calls = _with_init(system, lambda: pipeline.run(through=through, resume=False))
    else:
        pipeline.run(resume=False)
    aliases = AliasResult.load(run / "stages" / "search" / "aliases.json")
    bands = band_table(aliases, n_is)
    modes = json.loads((run / "stages" / "modes" / "report.json").read_text())
    search = json.loads((run / "stages" / "search" / "report.json").read_text())
    null = modes.get("no_orbit") or {}
    raw = null.get("verdict")
    verdict = dict(value=None if raw is None else ("no-orbit" if str(raw).startswith("no-orbit") else "orbit"), raw=raw,
                   source="stages/modes/report.json no_orbit.verdict (the verdict virgil ships at this commit)",
                   winner_n=(modes.get("winner") or {}).get("n"), delta_nats=null.get("delta"))
    out = dict(bands=bands, verdict=verdict, tempering=search.get("tempering"), labels_free=modes.get("labels_free"),
               primary_diam=search.get("primary_diam"), outliers=search.get("outliers"),
               night_flags=search.get("night_flags"), n_bands=search.get("n_bands"))
    if part == "oracle":
        out["oracle_init_calls"] = init_calls
    if part == "fit":
        from crosscheck import orbit_population as op

        jitter = np.random.default_rng(np.random.SeedSequence([op.MASTER_SEED, op.CODES[system["stratum"]],
                                                              system["index"]]).spawn(2)[1]).uniform()
        out["posterior"] = posterior_summary(run, system, bands, jitter) if system["orbit"] is not None else None
    return out


def _with_init(system, call):
    """Run ``call`` with ``fit_orbit_aliases`` given ``init=`` the truth. The
    pipeline's search stage imports it at call time, so the oracle is the
    default run with one extra argument and nothing else changed."""
    import virgil.aliases as aliases

    original = aliases.fit_orbit_aliases
    calls = []

    def with_truth(*args, **kwargs):
        calls.append(1)
        t_ref = float(np.round(np.mean(np.asarray(kwargs["times"], float)), 1))  # the engine's default t_ref
        return original(*args, **{**kwargs, "t_ref": t_ref, "init": oracle_init(system, t_ref)})

    aliases.fit_orbit_aliases = with_truth
    try:
        call()
    finally:
        aliases.fit_orbit_aliases = original
    if not calls:
        raise RuntimeError("init= was never passed: the search stage does not look up virgil.aliases.fit_orbit_aliases at call time")
    return len(calls)


# ------------------------------------------------------------- aggregation


def true_band(r):
    """The fitted band row holding the true period, or None (outside the window)."""
    period = r["truth"]["period"]
    return next((b for b in r.get("bands", []) if b["p_lo"] <= period <= b["p_hi"]), None)


def band_p(r):
    return np.array([b["p"] or 0.0 for b in r["bands"]])


def in_set(r, m):
    chosen, mass = credible_set(band_p(r), m)
    tb = true_band(r)
    return (tb is not None and r["bands"].index(tb) in chosen), mass, len(chosen)


def coverage_test(rs, m, sided):
    hits, masses, sizes = 0, [], []
    for r in rs:
        ok, mass, size = in_set(r, m)
        hits += ok
        masses.append(mass)
        sizes.append(size)
    return dict(n=len(rs), hits=int(hits), expected=float(np.sum(masses)),
                p=poisson_binomial_p(hits, masses, sided) if rs else 1.0,
                set_size_median=float(np.median(sizes)) if sizes else None,
                set_size_q90=float(np.quantile(sizes, 0.9)) if sizes else None,
                realised_mass=float(np.mean(masses)) if masses else None)


def m6_limit(n, table):
    for key, limit in table.items():
        lo, hi = (int(x) for x in str(key).split("-"))
        if lo <= n <= hi:
            return int(limit)
    return binomial_q99(n, 0.01)


def aggregate_results(results, s, rng_seed):
    """The registered analysis of per-system results (dicts as ``fit`` and
    ``oracle`` write them). Reads truths and virgil outputs only."""
    from scipy import stats

    sbc = _load("sbc_v2")
    health = sbc.spec()["sampler_health"]  # the registered chain-health rule, unchanged
    pv = {}
    out = dict(pvalues=pv, must_hold={}, reported={}, unavailable=[], incomplete=[])
    fits = [r for r in results if r["part"] == "fit"]
    by = {k: [r for r in fits if r["truth"]["stratum"] == k] for k in "ABND"}
    oracle = {r["truth"]["index"]: r for r in results if r["part"] == "oracle"}
    ok = lambda rs: [r for r in rs if r["status"] == "ok"]  # noqa: E731
    t, m = s["tests"], s["must_hold"]

    # M1 and M2 on A (calibration)
    a = ok(by["A"])
    for level in t["M1_band_calibration"]["levels"]:
        if a:
            pv[f"M1:{level}"] = coverage_test(a, level, t["M1_band_calibration"]["sided"])["p"]
        else:
            out["incomplete"].append(f"M1:{level}: no stratum A system to test")
    sampled = [r for r in a if (r.get("posterior") or {}).get("sampled")]
    rng = np.random.default_rng(np.random.SeedSequence([s["master_seed"], 99]))
    spec2 = t["M2_parameter_coverage"]
    cover = {}
    for q in spec2["params"]:
        u = np.array([r["posterior"]["u"][q] for r in sampled])
        if not u.size:
            out["incomplete"].append(f"M2:{q}: no sampled stratum A system")
            continue
        pv[f"M2:{q}:ks"] = float(stats.kstest(u, "uniform").pvalue)
        for level in spec2["intervals"]:
            k = int(np.sum(np.abs(u - 0.5) <= level / 2))
            cover[f"{q}:{level}"] = [k, int(u.size)]
            pv[f"M2:{q}:cover{level}"] = float(stats.binomtest(k, u.size, level).pvalue)
    unhealthy = [r for r in sampled if _unhealthy(r["posterior"], health, sbc)]
    out["reported"]["A"] = dict(n=len(by["A"]), n_ok=len(a), n_sampled=len(sampled), coverage=cover,
                                unhealthy=len(unhealthy), unhealthy_fraction=len(unhealthy) / max(len(sampled), 1),
                                temper_f=_summ([(r.get("tempering") or {}).get("f_raw") for r in a]))

    # M3, M5, M9 on B
    b = ok(by["B"])
    mis = [r for r in b if r["truth"]["misspecified"]]
    map_hit, map_p = 0, []
    for r in b:
        p = band_p(r)
        i = int(np.argmax(p)) if p.size else None
        map_p.append(float(p[i]) if p.size else 0.0)
        map_hit += i is not None and r["bands"][i] is true_band(r)
    mp = np.array(map_p)
    if b:
        pv["M3a"] = poisson_binomial_p(map_hit, mp, "under")
    else:
        out["incomplete"].append("M3a: no stratum B system")
    out["reported"]["M3a"] = dict(n=len(b), hits=int(map_hit), expected=float(mp.sum()))
    spec5 = t["M5_coverage_contaminated"]
    floor = float(spec5["condition"].split("_ge_")[1])
    sel = [r for r in b if (true_band(r) or {}).get("p") and true_band(r)["p"] >= floor
           and (r.get("posterior") or {}).get("sampled")]
    for q in spec5["params"]:
        k = int(sum(r["posterior"]["covers95"][q] for r in sel))
        if sel:
            pv[f"M5:{q}"] = float(stats.binom.cdf(k, len(sel), spec5["interval"]))
        else:
            out["incomplete"].append(f"M5:{q}: no B system with the true band at p >= {floor}")
    out["reported"]["M5"] = dict(n=len(sel), covered={q: int(sum(r["posterior"]["covers95"][q] for r in sel))
                                                       for q in spec5["params"]},
                                 misspecified=_m5_split(sel))
    m9 = m["M9_band_credible_set"]
    out["must_hold"]["M9"] = {name: _under(coverage_test(rs, m9["m"], m9["sided"]))
                              for name, rs in (("B_all", b), ("B_misspecified", mis))}
    spec3 = m["M3b_confident_errors"]
    out["must_hold"]["M3b"] = {}
    for name, rs in (("B_all", b), ("B_misspecified", mis)):
        conf = [r for r, p in zip(b, map_p) if p >= spec3["p_map_min"] and r in rs]
        wrong = sum(1 for r in conf if r["bands"][int(np.argmax(band_p(r)))] is not true_band(r))
        limit = binomial_q99(len(conf), 0.01) if conf else 0
        out["must_hold"]["M3b"][name] = dict(n_conf=len(conf), wrong=int(wrong), limit=limit,
                                             **{"pass": bool(wrong <= limit) if conf else None})  # nothing confident: nothing was tested

    # M4 and M8 need the oracle runs
    spec4, spec8 = m["M4_search_misses"], m["M8_false_no_orbit"]
    n_b = strata_sizes()["B"]
    misses, no_oracle, no_true_band = [], [], []
    for r in b:
        o = oracle.get(r["truth"]["index"])
        if o is None or o["status"] != "ok" or not o.get("oracle_init_calls"):
            no_oracle.append(r["truth"]["index"])  # a missing or invalid oracle is a failure, not a skipped system
            continue
        ob, bb = true_band(o), true_band(r)
        if ob is None or ob["log_z"] is None:
            no_true_band.append(r["truth"]["index"])
            continue
        zb = None if bb is None else bb["log_z"]
        se = math.hypot((bb or {}).get("se_lnz") or 0.0, ob.get("se_lnz") or 0.0)
        if zb is None or ob["log_z"] - zb > spec4["nats"] + spec4["plus_se_multiple"] * se:
            misses.append(r["truth"]["index"])
    bad = len(misses) + len(no_oracle) + len(no_true_band)
    out["must_hold"]["M4"] = dict(n=n_b, misses=len(misses), missing_or_invalid_oracle=len(no_oracle),
                                  oracle_without_true_band=len(no_true_band), limit=spec4["max_fraction"] * n_b,
                                  judged_by="virgil", **{"provisional_pass": bool(len(b) == n_b and bad <= spec4["max_fraction"] * n_b)},
                                  se_sources=sorted({str(x.get("se_source")) for r in b for x in r["bands"]}))
    qualifying = [r for r in b if r["truth"]["index"] in oracle and _qualifies(oracle[r["truth"]["index"]])]
    lost = [r for r in qualifying if (r.get("verdict") or {}).get("value") == "no-orbit"]
    out["must_hold"]["M8"] = dict(qualifying=len(qualifying), lost=len(lost), judged_by="virgil",
                                  **{"provisional_pass": bool(len(lost) <= spec8["max_fraction"] * len(qualifying)) if qualifying else None},
                                  basis="oracle verdict 'orbit' (FAP at the Monte Carlo floor is not available at this commit)")
    out["reported"]["M8_by_K"] = _count_by(lost, lambda r: r["truth"]["n_nights"])

    # M6 on N
    n_ = ok(by["N"])
    spec6 = m["M6_false_orbits"]
    out["must_hold"]["M6"] = {}
    for name, rs in (("N_all", n_), ("N_misspecified", [r for r in n_ if r["truth"]["misspecified"]])):
        false = [r for r in rs if _false_orbit(r)]
        limit = spec6["N_all"]["max_count"] if name == "N_all" and len(rs) == spec6["N_all"]["n"] else m6_limit(len(rs), spec6["limits_by_n"])
        out["must_hold"]["M6"][name] = dict(n=len(rs), false_orbits=len(false), limit=int(limit),
                                            static_companion=sum(1 for r in rs if _static(r)), judged_by="virgil",
                                            **{"provisional_pass": bool(len(false) <= limit) if rs else None})
    out["reported"]["M6_by_template"] = _rates(n_, "template")
    out["reported"]["M6_by_kind"] = _rates(n_, "misspecification")
    out["reported"]["M6_by_envelope_fraction"] = _by_envelope(n_)

    # M7: completion and the accuracy rule
    fail_flags = set(m["M7_complete"]["credible_set_accurate"]["fail_flags"])
    crashed = [r for r in results if r["status"] != "ok"]
    inaccurate = [r for r in b if _inaccurate(r, fail_flags, m9["m"])]
    expected = {k: v for k, v in strata_sizes().items()}
    present = {k: len({r["truth"]["index"] for r in by[k]}) for k in "ABND"}
    expected["oracle"], present["oracle"] = expected["B"], len(oracle)
    invalid_oracle = [i for i, o in oracle.items() if o["status"] == "ok" and not o.get("oracle_init_calls")]
    flags_seen = sorted({f for r in b for x in r["bands"] for f in x["flags"]})
    flags_observable = bool(fail_flags & set(flags_seen)) or any(x["se_source"] == "reported" for r in b for x in r["bands"])
    complete = bool(not crashed and not invalid_oracle and all(present[k] == expected[k] for k in expected))
    out["must_hold"]["M7"] = dict(failed=len(crashed), missing={k: expected[k] - present[k] for k in expected if present[k] < expected[k]},
                                  invalid_oracle=len(invalid_oracle), inaccurate_credible_set=len(inaccurate),
                                  **{"pass": False if not complete or inaccurate else (True if flags_observable else None)})
    out["reported"]["band_label_mismatch"] = sum(1 for r in a + b if (true_band(r) or {}).get("n") != r["truth"]["band"])
    out["reported"]["flags_seen_in_B"] = flags_seen
    if not flags_observable:
        out["unavailable"].append("evidence-noisy / evidence-failed flags and per-band standard errors: not reported by this virgil commit; M7's accuracy rule can only count flags it sees, and M4 uses the ESS-based SE (se_source 'ess')")
    out["unavailable"].append("per-data-set empirical FAP verdict: not in virgil 0.4.x (orbit_dehack Stage 4); M6 and M8 use the shipped verdict")
    out["reported"].update(recovery_by_cell(b), tempering_vs_c=_tempering(b), wall_s=_summ([r["seconds"] for r in results]),
                           peak_rss_mb=_summ([r["peak_rss_mb"] for r in results]),
                           redraws_B=int(sum(r["truth"]["n_redraws"] for r in b)),
                           data_sufficiency=_sufficiency(b))

    # Holm over the p-value family (detection_campaign's pattern, sbc_numpyro.holm)
    rejected = sbc.v1.holm(pv, s["family_alpha"]) if pv else {}
    out["rejected"] = sorted(k for k, v in rejected.items() if v)
    out["tests"] = len(pv)
    flat = [v for v in out["must_hold"].values()]
    flat = [w for v in flat for w in (v.values() if not ({"pass", "provisional_pass"} & set(v)) else [v])]
    must = [w["pass"] if "pass" in w else w["provisional_pass"] for w in flat]
    out["counts"] = {k: len(by[k]) for k in "ABND"} | {"oracle": len(oracle)}
    out["judged_by_virgil"] = ["M4", "M6", "M8"]
    failed = bool(not pv or out["rejected"] or any(x is False for x in must))
    # True only if every registered test ran as registered. M4, M6 and M8 are judged by virgil's own verdict and
    # oracle (the independent FAP is not implemented): their results are provisional_pass, a failure still fails the
    # run, but a run with them (or any empty subset) is INCOMPLETE, never a pass.
    out["pass"] = False if failed else (None if out["unavailable"] or out["incomplete"] or any(x is None for x in must) else True)
    out["label"] = ("INCOMPLETE: M4, M6 and M8 are judged by virgil's own verdict, not the registered independent FAP"
                    if out["pass"] is None else None)
    return out


def _under(c):
    return dict(c, **{"pass": (bool(c["p"] > 0.01 or c["hits"] >= c["expected"]) if c["n"] else None)}, note="one-sided under-coverage; p from the Poisson-binomial normal approximation")


def _unhealthy(post, health, sbc):
    return sbc.unhealthy(dict(divergences=post["divergences"], rhat=post["rhat"], ess=post["ess"]), health)


def _summ(x):
    x = np.array([v for v in x if v is not None and np.isfinite(v)], float)
    return None if not x.size else dict(n=int(x.size), median=float(np.median(x)), q10=float(np.quantile(x, 0.1)), q90=float(np.quantile(x, 0.9)))


def _qualifies(o):
    return o["status"] == "ok" and (o.get("fap") is not None and o["fap"] <= 1e-9 or (o.get("verdict") or {}).get("value") == "orbit")


def _static(r):
    return r["truth"]["misspecification"] == "background" and (r.get("verdict") or {}).get("value") == "orbit" \
        and (r["verdict"].get("winner_n") == 0)


def _false_orbit(r):
    v = (r.get("verdict") or {}).get("value")
    return v == "orbit" and not _static(r)


def _inaccurate(r, fail_flags, m):
    chosen, _ = credible_set(band_p(r), m)
    if any(set(r["bands"][i]["flags"]) & fail_flags for i in chosen):
        return True
    return any(set(b["flags"]) & {"evidence-failed"} for b in r["bands"])


def _count_by(rs, key):
    out = {}
    for r in rs:
        out[str(key(r))] = out.get(str(key(r)), 0) + 1
    return out


def _rates(rs, field):
    out = {}
    for r in rs:
        k = str(r["truth"][field])
        e = out.setdefault(k, [0, 0])
        e[0] += _false_orbit(r)
        e[1] += 1
    return {k: dict(false_orbits=a, n=n) for k, (a, n) in out.items()}


def _by_envelope(rs):
    env = [r for r in rs if r["truth"]["misspecification"] == "envelope"]
    if not env:
        return {}
    f = np.log([r["truth"]["envelope_fraction"] for r in env])
    edges = np.linspace(f.min(), f.max(), 4)
    out = {}
    for i in range(3):
        sel = [r for r, x in zip(env, f) if edges[i] <= x <= edges[i + 1] and (i == 2 or x < edges[i + 1])]
        out[f"third{i + 1}"] = dict(false_orbits=sum(_false_orbit(r) for r in sel), n=len(sel))
    return out


def _m5_split(sel):
    out = {}
    for flag in (True, False):
        rs = [r for r in sel if r["truth"]["misspecified"] == flag]
        out["misspecified" if flag else "clean"] = dict(n=len(rs), covered={q: int(sum(r["posterior"]["covers95"][q] for r in rs))
                                                                         for q in ("period", "ecc", "a_mas")})
    return out


def recovery_by_cell(b):
    """Reported, never pass/fail: MAP recovery by schedule x K x contrast bin
    with Wilson 95% intervals (gate M3)."""
    cells = {}
    for r in b:
        p = band_p(r)
        hit = bool(p.size and r["bands"][int(np.argmax(p))] is true_band(r))
        dmag = -2.5 * math.log10(r["truth"]["flux"])
        key = f"{r['truth']['schedule']}|K{r['truth']['n_nights']}|dmag{int(min(dmag // 1.5, 2)) * 1.5:g}+"
        e = cells.setdefault(key, [0, 0])
        e[0] += hit
        e[1] += 1
    return dict(recovery_by_cell={k: dict(hits=h, n=n, wilson95=wilson(h, n)) for k, (h, n) in sorted(cells.items())})


def _tempering(b):
    rows = [(r["truth"]["c"], (r.get("tempering") or {}).get("f")) for r in b if (r.get("tempering") or {}).get("f") is not None]
    return {str(c): _summ([f for cc, f in rows if cc == c]) for c in sorted({c for c, _ in rows})}


def _sufficiency(b):
    out = {}
    for r in b:
        p = band_p(r)
        hit = bool(p.size and r["bands"][int(np.argmax(p))] is true_band(r))
        cov = r["truth"]["phase_coverage"]
        key = f"K{r['truth']['n_nights']}|coverage>={int(cov * 4) / 4:.2f}"
        e = out.setdefault(key, [0, 0])
        e[0] += hit
        e[1] += 1
    return {k: dict(hits=h, n=n) for k, (h, n) in sorted(out.items())}


# ------------------------------------------------------------- commands


def cmd_fit(args, part):
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if part == "oracle":
        stratum, index = "B", args.task
    else:
        stratum, index = locate(args.task)
    if part == "oracle" and not 0 <= index < strata_sizes()["B"]:
        raise SystemExit("oracle tasks are B systems 0..239")
    crit_commit = prereg.criteria_commit(CRITERIA_FILE)  # refuses uncommitted criteria before any draw is made
    commit = virgil_commit()
    if not commit:
        raise SystemExit("cannot tell which virgil commit is running: set PIN_COMMIT")
    import jax

    head = dict(part=part, stratum=stratum, index=index, virgil_commit=commit, criteria_block_hash=criteria_hash(),
                design_hash=design_hash(), jax=jax.__version__,
                device=jax.devices()[0].platform)
    head["criteria_commit"] = crit_commit
    path = out / f"{part}_{stratum}{index:04d}.json"
    if path.exists():  # resume: a finished system is not rerun
        old = json.loads(path.read_text())
        if {k: old.get(k) for k in head if k != "device"} != {k: head[k] for k in head if k != "device"}:
            raise SystemExit(f"{path} was made differently: move it before rerunning")
        print(f"{path} is complete (status {old['status']}; a failure stays a failure)")
        return
    result = run_system(part, stratum, index, out, keep_work=args.keep_work)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({**head, **result}))
    tmp.rename(path)
    print(f"{part} {stratum}{index:04d} status={result['status']} seconds={result['seconds']:.0f}", flush=True)


def cmd_draw(args):
    from crosscheck import orbit_population as op

    system = op.draw_system(args.stratum, args.index)
    out = pathlib.Path(args.out) / f"{args.stratum}{args.index:04d}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "truth.json").write_text(json.dumps(system, indent=1))
    print(json.dumps([str(p) for p in op.write_system(system, out)]))


def cmd_criteria(args):
    parsed = spec()
    print(json.dumps(dict(criteria_commit=prereg.criteria_commit(CRITERIA_FILE), criteria_block_hash=criteria_hash(),
                          design_hash=design_hash(), master_seed=parsed["master_seed"]), indent=1))


def cmd_check_gate(args):
    """The vendored gate file against virgil's at a commit (run on the Mac)."""
    import subprocess

    text = subprocess.run(["git", "-C", args.virgil_repo, "show", f"{args.commit}:design/orbit_gate_042.md"],
                          capture_output=True, text=True, check=True).stdout
    same = gate_hash(text) == gate_hash()
    print(json.dumps(dict(commit=args.commit, file_matches=same, block_matches=criteria_hash(text) == criteria_hash(),
                          virgil_file_hash=gate_hash(text), vendored_file_hash=gate_hash())))
    if not same:
        raise SystemExit("virgil's gate file differs from the vendored copy: copy it and commit it, and record who approved it in the file's change log")


def cmd_aggregate(args):
    crit_commit = prereg.criteria_commit(CRITERIA_FILE)
    files = sorted(f for d in args.dirs for f in pathlib.Path(d).glob("*_[ABND]*.json") if not f.name.startswith("truth_"))
    if not files:
        raise SystemExit(f"no results found in {', '.join(map(str, args.dirs))}")
    results = [json.loads(f.read_text()) for f in files]
    keys = [(r["part"], r["stratum"], r["index"]) for r in results]
    if len(set(keys)) != len(keys):
        raise SystemExit("a system appears twice (a results directory is given twice, or two runs are mixed)")
    if prereg.require_same_commit(results) != crit_commit:  # no override: changed criteria mean a rerun
        raise SystemExit(f"the runs used criteria commit {prereg.require_same_commit(results)}, this checkout has {crit_commit}: rerun")
    for name, now in (("criteria_block_hash", criteria_hash()), ("design_hash", design_hash())):
        seen = {r[name] for r in results}
        if seen != {now}:
            raise SystemExit(f"{name} changed since the runs: {seen} vs {now}")
    commits = {r["virgil_commit"] for r in results}
    if len(commits) != 1:
        raise SystemExit(f"the systems ran on different virgil commits: {commits}")
    parsed = spec()
    summary = dict(version=1, virgil_commit=commits.pop(), criteria_commit=crit_commit, criteria_block_hash=criteria_hash(), design_hash=design_hash(),
                   devices=sorted({r["device"] for r in results}), jax=sorted({r["jax"] for r in results}))
    summary.update(aggregate_results(results, parsed, parsed["master_seed"]))
    text = json.dumps(summary, indent=1)
    if args.summary:
        pathlib.Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.summary).write_text(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("criteria")
    g = sub.add_parser("check-gate")
    g.add_argument("--virgil-repo", required=True)
    g.add_argument("--commit", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--stratum", choices="ABND", required=True)
    d.add_argument("--index", type=int, required=True)
    d.add_argument("--out", required=True)
    for name in ("fit", "oracle"):
        r = sub.add_parser(name)
        r.add_argument("--task", type=int, required=True)
        r.add_argument("--out", required=True)
        r.add_argument("--keep-work", action="store_true")
    a = sub.add_parser("aggregate")
    a.add_argument("dirs", nargs="+")
    a.add_argument("--summary")
    args = ap.parse_args()
    {"criteria": cmd_criteria, "check-gate": cmd_check_gate, "draw": cmd_draw, "aggregate": cmd_aggregate,
     "fit": lambda x: cmd_fit(x, "fit"), "oracle": lambda x: cmd_fit(x, "oracle")}[args.cmd](args)


if __name__ == "__main__":
    main()
