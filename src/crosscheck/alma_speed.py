"""Speed of virgil's L-BFGS fit against MPoL's AdamW run on IM Lup (numpy only; nothing here imports virgil or torch).

Inputs are small saved JSON: virgil's ``result.json`` per fit (``alma_imlup_maxent.py fit``) and MPoL's reference
run (``stage<N>.time.json`` from ``alma_mpol_ref``'s ``train.sbatch``, ``export/objective.json`` from ``objective_full``,
and ``export/objective_chk0.json`` when ``objective_full`` was also run on the end of stage 0). Definitions (also written to
``speed.json``):

* **Objective** ``R = -ln L + lam S_M`` (design C3, C4): MPoL's own, float64, all data. virgil's fit loss differs from it by a
  constant (the terms of -ln L that do not depend on the parameters, e.g. the weights' normalisation); ``constant_of`` takes it
  from the final point, ``R_final - loss_final``, and ``constant_check`` compares it between fits (and with any anchor, a second
  evaluation of R at another step, in ``anchors.json`` in the fit directory).
* **Wall time of a virgil fit** is the sum of its stages' ``wall_s`` (the optimiser loop, from the start of each stage's first
  ``virgil.fit`` call, so it includes the jit compile of that call and the restarts' own overhead); the problem build before the
  first stage (``result["wall_s"]`` minus the stages) is reported as start-up and shown separately; reading the npz is not timed.
  The SB stage of the ``p3`` workflow counts in the time and not on the curve (its loss is on fewer data).
* **Wall time of MPoL** is the whole ``sgd.py`` process per stage (``stage<N>.time.json``): asdf load, CUDA start-up and the
  epochs. It is only known at the end of each stage (the checkpoints), so MPoL's curve has two points.
* **Time to target** is the wall time of the first record at or below the target (records are every 200 steps; a linear
  interpolation between the previous record and that one is given beside it). It is only defined for a cold start.
"""

import re

import numpy as np

CHUNK_DEFAULT = 200  # virgil.fit reports every 200 steps
BENCH_VG_S = (0.44, 0.53)  # A100, f64, eps 1e-6, value+gradient over 4.24e7 points (job 18862592; design/alma_imlup_p3.md "Inputs")
MPOL_BATCH = 40000  # MPoL's minibatch (sgd.py default)


def cold_start(result):
    """``"p3"`` or ``"mpol"`` (a cold start of that workflow), or ``"warm"`` if ``--init`` was given."""
    args = result.get("args") or {}
    if args.get("init"):
        return "warm"
    return result.get("workflow") or args.get("workflow") or "p3"


def constant_of(result):
    """The constant c with R = loss + c, from the final point of the last stage (R configs only)."""
    if result.get("objective_R") is None:
        raise ValueError("no objective_R in result.json: speed compares R fits (config R) only")
    last = result["stages"][-1]
    if last["name"] != "all":
        raise ValueError("the last stage fitted a subset: its loss is not on the all-data objective")
    return float(result["objective_R"]) - float(last["loss_final"])


def constant_check(items, tol=5.0):
    """``items``: {label: (constant, [(steps, R_at_steps, loss_at_steps), ...])}. The constants must agree across fits and
    anchors to ``tol`` (objective units; the fits' eps 1e-6 loss against eps 1e-12 at the final point differ by less)."""
    cs = {k: v[0] for k, v in items.items()}
    for k, (c, anchors) in items.items():
        for a in anchors:
            cs[f"{k}@{int(a['steps'])}"] = float(a["objective"]) - float(a["loss"])
    vals = np.array(list(cs.values()))
    spread = float(vals.max() - vals.min()) if len(vals) else 0.0
    return dict(constants=cs, spread=spread, tol=tol, ok=bool(spread <= tol), n_points=len(vals),
                note=("one constant per fit (final point) only: no second evaluation of R inside a fit" if all(not a for _, a in items.values()) else
                      "includes anchors (second evaluations of R inside a fit)"))


def fit_curve(result):
    """Cumulative wall time and steps of every trajectory record of a fit: list of dict(t_s, steps, loss, stage, round,
    on_curve). ``elapsed`` restarts at every round (a ``run_fit_c8`` restart), so a round's offset is the time of the earlier
    rounds (``restarts[i].wall_s`` when saved, else the last ``elapsed``); stages follow each other by ``wall_s``."""
    pts, t_off, s_off, recon = [], 0.0, 0.0, []
    for st in result["stages"]:
        traj = st.get("trajectory") or []
        restarts = st.get("restarts") or []
        r_off = 0.0
        for i, r in enumerate(sorted({int(p.get("round", 1)) for p in traj})):
            rp = [p for p in traj if int(p.get("round", 1)) == r]
            for p in rp:
                pts.append(dict(t_s=t_off + r_off + float(p["elapsed"]), steps=s_off + float(p["steps"]), loss=float(p["loss"]), stage=st["name"],
                                round=r, on_curve=st["name"] == "all"))
            last_el = max(float(p["elapsed"]) for p in rp)
            r_off += max(float(restarts[i]["wall_s"]) if i < len(restarts) else 0.0, last_el)
        recon.append(dict(stage=st["name"], wall_s=float(st["wall_s"]), rounds_s=r_off, diff_s=float(st["wall_s"]) - r_off))
        t_off += float(st["wall_s"])
        s_off += float(st["steps"])
    return pts, recon


def steady_cost(result, chunk=CHUNK_DEFAULT):
    """Seconds per L-BFGS step from the trajectory of the last stage: Δelapsed/Δsteps over consecutive records exactly one chunk
    apart inside a round (so the jit compile of a round's first chunk and the short closing record are left out). Returns the
    median, the 10th/90th percentiles, the medians of the first and last quarter (the cost drifts as the line search works
    harder), and the compile estimate per round: that round's first chunk minus the median cost of a chunk."""
    st = result["stages"][-1]
    traj = st.get("trajectory") or []
    rates, compile_est = [], []
    for r in sorted({int(p.get("round", 1)) for p in traj}):
        rp = [p for p in traj if int(p.get("round", 1)) == r]
        prev_steps = max([float(p["steps"]) for p in traj if int(p.get("round", 1)) < r] or [0.0])
        for k, p in enumerate(rp):
            ds, de = float(p["steps"]) - (prev_steps if k == 0 else float(rp[k - 1]["steps"])), float(p["elapsed"]) - (0.0 if k == 0 else float(rp[k - 1]["elapsed"]))
            if ds == chunk and k > 0:
                rates.append(de / ds)
            elif ds == chunk and k == 0:
                compile_est.append(dict(round=r, first_chunk_s=de, steps=int(ds)))
    if not rates:
        return dict(n_chunks=0, note="no two records exactly one chunk apart: cost per step not measured")
    rates = np.array(rates)
    q = max(len(rates) // 4, 1)
    med = float(np.median(rates))
    for c in compile_est:
        c["compile_s_est"] = max(c["first_chunk_s"] - med * c["steps"], 0.0)
    return dict(n_chunks=int(len(rates)), s_per_step_median=med, s_per_step_p10=float(np.percentile(rates, 10)), s_per_step_p90=float(np.percentile(rates, 90)),
                s_per_step_first_quarter=float(np.median(rates[:q])), s_per_step_last_quarter=float(np.median(rates[-q:])), chunk=chunk, compile=compile_est)


def time_to(points, target):
    """First record at or below ``target``: dict(reached, t_first_s, steps, t_interp_s). ``points`` are objective points
    (t_s, steps, objective) in time order."""
    prev = None
    for p in points:
        if p["objective"] <= target:
            t_i = p["t_s"] if prev is None else prev["t_s"] + (p["t_s"] - prev["t_s"]) * (prev["objective"] - target) / (prev["objective"] - p["objective"])
            return dict(reached=True, t_first_s=p["t_s"], steps=p["steps"], t_interp_s=float(t_i))
        prev = p
    return dict(reached=False, t_first_s=None, steps=None, t_interp_s=None, best_objective=min((p["objective"] for p in points), default=None))


LOG_RE = dict(job=re.compile(r"\bjob=(\d+)"), host=re.compile(r"\bhost=(\S+)"), gpu=re.compile(r"(NVIDIA [A-Za-z0-9 \-]+?)(?:,|\s{2,}|$)", re.M),
              stage=re.compile(r"\bstage=(\d)"), task=re.compile(r"\btask=(\d+)"), elapsed=re.compile(r"\|(\d\d:\d\d:\d\d)\|"))


def parse_logs(texts):
    """Hardware lines of job logs or digests: for every ``job=… host=…`` line the job, host, stage/task, and the GPU model named in
    the same text (nvidia-smi line or the torch line). ``texts`` is {source: text}; a source without such a line is returned with
    ``host`` None, so the report says "not recorded" rather than guessing."""
    out = []
    for src, text in texts.items():
        gpu = LOG_RE["gpu"].search(text)
        lines = [ln for ln in text.splitlines() if LOG_RE["host"].search(ln) and LOG_RE["job"].search(ln)]
        if not lines:
            out.append(dict(source=src, job=None, host=None, gpu=gpu.group(1).strip() if gpu else None))
        for ln in lines:
            g = lambda k: (LOG_RE[k].search(ln).group(1) if LOG_RE[k].search(ln) else None)  # noqa: E731
            out.append(dict(source=src, job=g("job"), host=g("host"), stage=g("stage"), task=g("task"), gpu=gpu.group(1).strip() if gpu else None))
    return out


def mpol_summary(stages, objective, objective_chk0=None, startup_s=0.0, batch=MPOL_BATCH):
    """MPoL's run: ``stages`` the list of stage<N>.time.json dicts, ``objective`` objective.json (chk1), ``objective_chk0`` or None."""
    stages = sorted(stages, key=lambda s: s["stage"])
    n = int(objective["n"])
    per_epoch = [float(s["wall_s"]) / float(s["epochs"]) for s in stages]
    cum, pts = 0.0, []
    for s in stages:
        cum += float(s["wall_s"])
        o = objective_chk0 if s["stage"] == 0 else objective
        pts.append(dict(t_s=cum - startup_s * (s["stage"] + 1), steps=None, epochs=sum(x["epochs"] for x in stages[:s["stage"] + 1]), stage=s["stage"],
                        objective=None if o is None else float(o["objective_R"]), label=f"end of stage {s['stage']}"))
    return dict(stages=stages, n=n, batches_per_epoch=n / batch, batch=batch, s_per_epoch=per_epoch, s_per_batch=[p / (n / batch) for p in per_epoch],
                stage_ratio=per_epoch[-1] / per_epoch[0] if len(per_epoch) > 1 else None, total_wall_s=cum, startup_subtracted_s=startup_s, points=pts,
                final_objective=float(objective["objective_R"]), final_raw_chi2_over_n=float(objective["chi2_raw_total"]) / (2 * n),
                chk0_objective=None if objective_chk0 is None else float(objective_chk0["objective_R"]),
                chk0_raw_chi2_over_n=None if objective_chk0 is None else float(objective_chk0["chi2_raw_total"]) / (2 * n),
                jobs=[s.get("job") for s in stages], dtype_train="float32 (PyTorch)", dtype_objective=objective.get("dtype"))


def build(fits, mpol, extra_targets=None, vg_seconds=BENCH_VG_S, hardware=None):
    """The whole comparison. ``fits``: {label: result.json dict, optionally with ``anchors`` list}; ``mpol``: ``mpol_summary``.
    Returns the dict of ``speed.json``."""
    extra_targets = dict(extra_targets or {})
    out = dict(definitions=dict(objective="R = -ln L + lam S_M, MPoL's objective, float64, all data (design C3, C4)",
                                virgil_time="sum of the stages' wall_s (optimiser loop incl. jit compile and restart overhead); problem build reported as start-up; npz reading not timed",
                                mpol_time="whole sgd.py process per stage (asdf load, CUDA start-up, epochs), known at the two checkpoints only",
                                time_to_target="wall time of the first record at or below the target (records every 200 steps); interpolated time beside it; cold starts only"),
               fits={}, hardware=hardware or [], mpol=mpol)
    consts = {}
    for label, res in fits.items():
        pts, recon = fit_curve(res)
        c = constant_of(res)
        consts[label] = (c, list(res.get("anchors") or []))
        curve = [dict(t_s=p["t_s"], steps=p["steps"], objective=p["loss"] + c) for p in pts if p["on_curve"]]
        stage_wall = sum(float(s["wall_s"]) for s in res["stages"])
        out["fits"][label] = dict(
            start=cold_start(res), config=res.get("config"), workflow=res.get("workflow"), versions=res.get("versions"), n=res.get("n"),
            steps=sum(int(s["steps"]) for s in res["stages"]), stage_wall_s=stage_wall, startup_s=(float(res["wall_s"]) - stage_wall) if res.get("wall_s") is not None else None,
            stages=[dict(name=s["name"], steps=s["steps"], wall_s=s["wall_s"], stopped_because=s["stopped_because"], converged=s["convergence"]["converged"]) for s in res["stages"]],
            time_reconciliation=recon, constant=c, curve=curve, final_objective=float(res["objective_R"]), final_raw_chi2_over_n=float(res["chi2_raw"]) / (2 * float(res["n"])),
            cost=steady_cost(res), converged=bool(res["stages"][-1]["convergence"]["converged"]), stopped_because=res["stopped_because"])
    out["constant_check"] = constant_check(consts)
    # targets
    cand = {f"{lab} (virgil, final)": f["final_objective"] for lab, f in out["fits"].items()}
    cand["MPoL final (chk1)"] = mpol["final_objective"]
    best_label = min(cand, key=cand.get)
    targets = {"MPoL final (chk1)": mpol["final_objective"]}
    if mpol["chk0_objective"] is not None:
        targets["MPoL end of stage 0 (chk0)"] = mpol["chk0_objective"]
    if best_label != "MPoL final (chk1)":  # else the best known is MPoL's own final objective, already a target
        targets[f"best known: {best_label}"] = cand[best_label]
    targets.update({k: float(v) for k, v in extra_targets.items()})
    rows = []
    for name, tv in targets.items():
        row = dict(target=name, objective=tv, mpol=None, virgil={})
        mp = [p for p in mpol["points"] if p["objective"] is not None]
        hit = next((p for p in mp if p["objective"] <= tv), None)
        row["mpol"] = dict(reached=hit is not None, t_s=hit["t_s"] if hit else None, where=hit["label"] if hit else None, epochs=hit["epochs"] if hit else None)
        for lab, f in out["fits"].items():
            row["virgil"][lab] = time_to(f["curve"], tv) if f["start"] != "warm" else dict(reached=None, note="warm start: not a cold-start time")
        rows.append(row)
    out["targets"] = rows
    # cost per full-data pass
    vg = tuple(vg_seconds)
    cost = dict(mpol_s_per_epoch=mpol["s_per_epoch"], mpol_s_per_batch=mpol["s_per_batch"], bench_vg_s=list(vg), per_fit={})
    for lab, f in out["fits"].items():
        c = f["cost"]
        if c.get("s_per_step_median") is None:
            cost["per_fit"][lab] = c
            continue
        s = c["s_per_step_median"]
        cost["per_fit"][lab] = dict(s_per_step=s, s_per_step_p10_p90=[c["s_per_step_p10"], c["s_per_step_p90"]], s_per_step_first_last_quarter=[c["s_per_step_first_quarter"], c["s_per_step_last_quarter"]],
                                    evals_per_step_est=[s / vg[1], s / vg[0]], evals_recorded=False,
                                    mpol_epoch_over_virgil_vg=[e / vg[1] for e in mpol["s_per_epoch"]] + [e / vg[0] for e in mpol["s_per_epoch"]],
                                    mpol_epoch_over_virgil_step=[e / s for e in mpol["s_per_epoch"]])
        # passes to a target: MPoL epochs against virgil evaluations (steps x evals/step, an estimate)
    cost["note"] = ("virgil's info records no evaluation count, so evaluations per step are estimated as s/step divided by the A100 bench's s per value+gradient "
                    f"({vg[0]}-{vg[1]} s, job 18862592); a pass over all data is one value+gradient for virgil and one epoch (about {mpol['batches_per_epoch']:.0f} minibatches of {mpol['batch']:,}) for MPoL")
    out["cost"] = cost
    return out


def fmt_t(t):
    return "n/a" if t is None else f"{t:,.0f} s ({t / 3600:.2f} h)"


def markdown(res, notes):
    """speed.md: raw chi^2/N first, then the targets, the cost per pass, the fairness notes."""
    m = res["mpol"]
    md = ["# virgil against MPoL on IM Lup: speed", ""]
    md += [f"**Raw chi^2/N on the quoted weights (s = 1) at the final point (chi^2/2N, eps 1e-12, N = {m['n']:,}): MPoL {m['final_raw_chi2_over_n']:.4f}; "
           + "; ".join(f"{lab} {f['final_raw_chi2_over_n']:.4f}" for lab, f in res["fits"].items()) + ".**", ""]
    md += ["## Definitions", ""] + [f"- **{k.replace('_', ' ')}**: {v}" for k, v in res["definitions"].items()] + [""]
    md += ["## The fits", "", "| fit | start | steps | stage wall | start-up (problem build) | final R | raw chi^2/N | stopped because | C8 converged |", "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for lab, f in res["fits"].items():
        md.append(f"| {lab} | {f['start']} | {f['steps']:,} | {fmt_t(f['stage_wall_s'])} | {fmt_t(f['startup_s'])} | {f['final_objective']:,.0f} | {f['final_raw_chi2_over_n']:.4f} | {f['stopped_because']} | {f['converged']} |")
    md.append(f"| MPoL (AdamW, 2 x {m['stages'][0]['epochs']} epochs) | cold (flat) | {m['stages'][0]['epochs'] * len(m['stages'])} epochs | {fmt_t(m['total_wall_s'])} | in the wall | {m['final_objective']:,.0f} | {m['final_raw_chi2_over_n']:.4f} | epochs | n/a |")
    md += ["", "## Time to target (cold starts)", "", "| target | objective R | MPoL | " + " | ".join(res["fits"]) + " |", "|---|---:|---|" + "---|" * len(res["fits"])]
    for r in res["targets"]:
        mp = r["mpol"]
        cells = [f"by {fmt_t(mp['t_s'])}, {mp['where']}" if mp["reached"] else "not reached"]
        for lab in res["fits"]:
            v = r["virgil"][lab]
            cells.append("n/a (warm start)" if v["reached"] is None else (f"{fmt_t(v['t_first_s'])} at step {v['steps']:,.0f}; interpolated {fmt_t(v['t_interp_s'])}" if v["reached"]
                                                                          else f"not reached (best {v['best_objective']:,.0f})"))
        md.append(f"| {r['target']} | {r['objective']:,.1f} | " + " | ".join(cells) + " |")
    md += ["", "MPoL's time is the end of the stage whose checkpoint is at or below the target (its objective is evaluated only at the checkpoints), so it is an upper bound; "
           "a target below MPoL's own final objective is not reached by MPoL."]
    if res["mpol"]["chk0_objective"] is None:
        md.append("MPoL's objective at the end of stage 0 was not computed (no `export/objective_chk0.json`): its curve has one point. See the ozstar_scripts `analyze.sbatch --only-chk0` step.")
    md += ["", "## Cost per full-data pass", "", f"MPoL: {m['s_per_epoch'][0]:,.0f} s per epoch in stage 0" + (f", {m['s_per_epoch'][1]:,.0f} s in stage 1 (a factor {m['stage_ratio']:.2f})" if len(m['s_per_epoch']) > 1 else "")
           + f"; {m['batches_per_epoch']:.0f} minibatches per epoch, so {m['s_per_batch'][0]:.2f} s per minibatch (forward and backward, float32).", ""]
    md += ["| fit | s/step (median; 10-90 %) | first / last quarter | evaluations per step (estimated) | MPoL epoch / virgil step | MPoL epoch / virgil value+gradient (bench) |", "|---|---|---|---|---|---|"]
    for lab, c in res["cost"]["per_fit"].items():
        if c.get("s_per_step") is None:
            md.append(f"| {lab} | {c.get('note')} | | | | |")
            continue
        e0, e1 = m["s_per_epoch"][0], m["s_per_epoch"][-1]
        vg = res["cost"]["bench_vg_s"]
        md.append(f"| {lab} | {c['s_per_step']:.3f} ({c['s_per_step_p10_p90'][0]:.3f}-{c['s_per_step_p10_p90'][1]:.3f}) | {c['s_per_step_first_last_quarter'][0]:.3f} / {c['s_per_step_first_last_quarter'][1]:.3f} | "
                  f"{c['evals_per_step_est'][0]:.1f}-{c['evals_per_step_est'][1]:.1f} | {e0 / c['s_per_step']:,.0f}-{e1 / c['s_per_step']:,.0f} | {e0 / vg[1]:,.0f}-{e1 / vg[0]:,.0f} |")
    md += ["", res["cost"]["note"] + ".", ""]
    cc = res["constant_check"]
    md += ["## Loss to objective", "", "R = loss + c with c from each fit's final point: " + ", ".join(f"{k}: {v:,.2f}" for k, v in cc["constants"].items())
           + f". Spread {cc['spread']:.3g} (tolerance {cc['tol']:g}): **{'consistent' if cc['ok'] else 'NOT consistent: the loss is not R plus a constant here, do not trust the curve'}**. {cc['note']}.", ""]
    md += ["## Start-up, compile and time bookkeeping", ""]
    for lab, f in res["fits"].items():
        comp = f["cost"].get("compile") or []
        md.append(f"- {lab}: problem build before the first stage {fmt_t(f['startup_s'])}; estimated compile per round (first chunk minus a steady chunk): "
                  + (", ".join(f"round {c['round']} {c['compile_s_est']:.0f} s" for c in comp) if comp else "not measured")
                  + "; time bookkeeping (stage wall_s minus the rounds' time): " + ", ".join(f"{r['stage']} {r['diff_s']:+.1f} s" for r in f["time_reconciliation"]) + ".")
    md += ["", "## Fairness notes", ""] + [f"- {n}" for n in notes]
    return "\n".join(md) + "\n"


def fairness_notes(res):
    """The notes written to speed.md and speed.json: hardware, precision, start-up, data loading, stage contention."""
    m = res["mpol"]
    hw = res["hardware"]
    notes = []
    mp_hw = [h for h in hw if h.get("job") in [str(j) for j in m["jobs"] if j]]
    ours = [h for h in hw if h not in mp_hw]
    if mp_hw:
        notes.append("Hardware, MPoL: " + "; ".join(f"job {h['job']} (stage {h.get('stage')}) on {h['host']}, {h.get('gpu') or 'GPU model not in the log'}" for h in mp_hw) + ".")
    else:
        notes.append("Hardware, MPoL: not recorded (no `--log` with its `job=… host=…` lines was given).")
    hosts = {h["host"] for h in ours if h.get("host")}
    if hosts:
        notes.append("Hardware, virgil: " + "; ".join(f"job {h['job']} on {h['host']}, {h.get('gpu') or 'GPU model not in the log'}" for h in ours if h.get("host")) + ".")
    else:
        notes.append("Hardware, virgil: not recorded (no `--log` with a `host=` line was given; `fit.sbatch` prints the host on the first line of the job log and prints no GPU model, "
                     "and result.json records only `cuda:0`), so a node and GPU match with MPoL's is not shown.")
    if m["stage_ratio"] and abs(m["stage_ratio"] - 1) > 0.25:
        same = len({h["host"] for h in mp_hw}) == 1 and len(mp_hw) > 1
        notes.append(f"MPoL's two stages differ by a factor {m['stage_ratio']:.2f} per epoch ({m['s_per_epoch'][0]:,.0f} s and {m['s_per_epoch'][-1]:,.0f} s) for identical work (same code, flags and data; only `--load-checkpoint` differs)"
                     + (f" on the same node ({mp_hw[0]['host']})" if same else "") + ". The cause is not known from the logs; node contention (CPU, memory bandwidth, PCIe from other jobs) is the first suspect, "
                     "so MPoL's wall times are uncertain by that factor and are a measurement of this run, not of the code. Both are reported; the headline uses the total.")
    notes.append("Precision: MPoL trains in float32 (PyTorch); its final objective is evaluated by `objective_full` in float64 with its accurate NuFFT. virgil fits in float64 with nufftax at eps 1e-6, and every reported objective is at eps 1e-12. "
                 "float32 is the faster arithmetic on an A100, so the comparison is not tilted towards virgil by precision.")
    notes.append("Start-up and compile: MPoL's wall times include the asdf load of 42 M points, CUDA start-up and (from the first epoch) torch's own warm-up; they are not separable from `stage<N>.time.json` "
                 f"(subtracted: {m['startup_subtracted_s']:g} s). virgil's problem build is shown separately; the jit compile of each `fit` round is inside the stage wall and estimated in the table above; npz reading is not timed. "
                 "So MPoL's times include data loading and virgil's do not: the bias favours virgil by the (unmeasured, minutes) load, small against hours.")
    notes.append("MPoL runs unmodified (decision C14) from its flat start with its own data export (CORRECTED_DATA); a cold start for virgil is `fit --workflow mpol` (the same start image, g = 1, s = 1, offsets 0) or the default `p3` route (an SB stage, then all data, from a 1\" Gaussian with s at the model-free ŝ). "
                 "Warm starts (`--init`) are drawn dashed and give no time to target.")
    notes.append("MPoL's objective is known only at its checkpoints, virgil's every 200 steps; MPoL's AdamW also decays every parameter by its default weight decay (0.01), so it does not minimise exactly the objective R (Things to tell Ian), "
                 "which is the objective both are scored on.")
    notes.append("The loss-to-objective constant is taken from each fit's own final point, where `report` evaluates R at eps 1e-12; along the curve the loss is at eps 1e-6, whose NUFFT error is far below the targets' separations.")
    return notes


def plot(res, path):
    """Objective against wall time, both codes on one axis, the targets as horizontal lines; the right panel zooms on the targets."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fits = res["fits"]
    tvals = [r["objective"] for r in res["targets"]]
    for ax, zoom in zip(axes, (False, True)):
        for i, (lab, f) in enumerate(fits.items()):
            t = np.array([p["t_s"] for p in f["curve"]]) / 3600
            y = np.array([p["objective"] for p in f["curve"]]) / 1e6
            ax.plot(t, y, color=f"C{i}", ls="--" if f["start"] == "warm" else "-", label=f"{lab} ({f['start']} start)")
        pts = [p for p in res["mpol"]["points"] if p["objective"] is not None]
        if pts:
            ax.plot([p["t_s"] / 3600 for p in pts], [p["objective"] / 1e6 for p in pts], "k^-", ms=8, lw=0.8, alpha=0.8, label="MPoL (checkpoints; no data between)")
        for r in res["targets"]:
            ax.axhline(r["objective"] / 1e6, color="0.5", lw=0.8, ls=":")
            ax.text(0.02, r["objective"] / 1e6, " " + r["target"], transform=ax.get_yaxis_transform(), fontsize=6, va="bottom", color="0.3")
        ax.set_xlabel("wall time (h)")
        ax.set_ylabel("objective R (10$^6$)")
        ax.set_xlim(left=0)
        if zoom and tvals:
            finals = tvals + [f["final_objective"] for f in fits.values()]
            lo, hi = min(finals), max(finals)
            pad = max(hi - lo, 2e5) * 0.15
            ax.set_ylim((lo - pad) / 1e6, (hi + pad) / 1e6)
            ax.set_title("zoom on the targets and the final points")
        else:
            ax.set_title("objective against wall time (MPoL f32 AdamW, virgil f64 L-BFGS)")
    axes[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
