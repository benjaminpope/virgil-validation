"""Toeplitz re-examination (virgil 0.4.2 gate G2), pre-registered in ``design/toeplitz_reexam_042.md`` and ``.yml``.

    python scripts/toeplitz_reexam.py tier1 --out DIR [--scene NAME ...]
    python scripts/toeplitz_reexam.py tier2-agree --data DIR --fit DIR --out DIR [--max-offset-mas 20]
    python scripts/toeplitz_reexam.py tier2-time --method {toeplitz,direct} --data DIR --fit DIR --out DIR
    python scripts/toeplitz_reexam.py summary --out DIR
    python scripts/toeplitz_reexam.py hash

Every command but ``hash`` refuses to run unless the criteria file's sha256 is ``REGISTERED_SHA256``, and every JSON it writes
records that hash and the virgil commit. ``summary`` refuses JSON files whose hash or virgil commit differ (rows from different
runs are never combined) and writes ``reexam.txt``: a row per gated quantity (PASS, FAIL or NOT EVALUATED), the report-only
numbers, and the verdict (ACCEPTED, NOT ACCEPTED or INCOMPLETE).

Tier I: virgil's Toeplitz term against ``crosscheck.radio_exact`` (numpy, no virgil) on the registered held-out scenes
(``crosscheck.toeplitz_scenes``), every error-scale mode and point, within ``radio_exact.budget``. Tier II: IM Lup at R_mpol,
the Toeplitz term against virgil's direct term at eps 1e-12 within the same budget (``eps_ref`` = 1e-12), then the speed-up and
the Toeplitz path's GPU peak on an A100 (``alma_imlup_toeplitz_check``'s timing, one process per method).
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CRITERIA_FILE = ROOT / "design" / "toeplitz_reexam_042.yml"
REGISTERED_SHA256 = "95dfbecd17854d88211bcd106cefcef730ba5c2bec702df3a05159d9dcccd1bf"  # sha256 of CRITERIA_FILE as registered; tests/test_toeplitz_reexam.py checks it
MODES = {
    "fixed": False,
    "fitted": True,
    "profiled": "profile",
}  # error-scale mode -> CalibrationNuisance(sigmas=...)
SITES = {
    "log_brightness": "eta",
    "cal.flux": "flux",
    "cal.gain": "gain",
    "cal.offset": "offset",
    "cal.sigma": "sigma",
}


def criteria_hash():
    return hashlib.sha256(CRITERIA_FILE.read_bytes()).hexdigest()


def criteria():
    """The registered criteria; ``SystemExit`` if the file is not the registered one."""
    import yaml

    if criteria_hash() != REGISTERED_SHA256:
        raise SystemExit(
            f"{CRITERIA_FILE.name} has sha256 {criteria_hash()}, not the registered {REGISTERED_SHA256}: "
            "criteria changed after registration (log the change in the .md, then re-register)"
        )
    return yaml.safe_load(CRITERIA_FILE.read_text())


def virgil_commit():
    """The virgil commit imported: ``$VIRGIL_COMMIT``, else a 40-hex directory in virgil's path (the cluster's snapshots), else
    ``git rev-parse HEAD`` in its source tree (with ``+dirty`` for local changes), else ``unknown``."""
    import virgil

    if re.fullmatch(r"[0-9a-f]{40}", os.environ.get("VIRGIL_COMMIT", "")):
        return os.environ["VIRGIL_COMMIT"]
    path = pathlib.Path(virgil.__file__).resolve()
    hit = re.search(r"/([0-9a-f]{40})/", str(path))
    if hit:
        return hit.group(1)
    try:
        git = ["git", "-C", str(path.parent)]
        head = subprocess.run(
            [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            [*git, "status", "--porcelain", "--", "."],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run_id():
    """One run's identity: ``$SLURM_JOB_ID`` (every step of the cluster job), else ``$REEXAM_RUN_ID``, else host and process (one
    local process). ``summary`` refuses outputs of more than one run."""
    import socket

    return os.environ.get("SLURM_JOB_ID") or os.environ.get("REEXAM_RUN_ID") or f"{socket.gethostname()}-pid{os.getpid()}"


def vv_commit():
    """This repository's commit (the budget's code is part of the criteria): ``$PIN_COMMIT`` (the cluster's snapshot has no
    .git), else ``git rev-parse HEAD`` (with ``+dirty`` for local changes to src/ or scripts/), else ``unknown``."""
    if re.fullmatch(r"[0-9a-f]{40}", os.environ.get("PIN_COMMIT", "")):
        return os.environ["PIN_COMMIT"]
    try:
        git = ["git", "-C", str(ROOT)]
        head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run([*git, "status", "--porcelain", "--", "src", "scripts"], capture_output=True, text=True, check=True).stdout
        return head + ("+dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def provenance():
    import importlib.metadata as m

    import jax
    import virgil

    return dict(
        criteria_sha256=criteria_hash(),
        run_id=run_id(),
        vv_commit=vv_commit(),
        virgil_commit=virgil_commit(),
        virgil_from=virgil.__file__,
        jax=jax.__version__,
        nufftax=m.version("nufftax"),
        device=jax.devices()[0].device_kind,
    )


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def write(path, obj):
    pathlib.Path(path).write_text(json.dumps(_jsonable(obj), indent=1))


# ------------------------------------------------------------------ comparison rows


def compare(where, ref, got, tol, tol_prob):
    """Rows ``(where, quantity, error, tolerance, probabilistic tolerance, relative error)`` for a loss, a ``chi2_d`` (when both have
    it) and each gradient site: ``error`` is an l2 norm per site (per element maximum of error / tolerance for ``sigma``)."""
    rows = []

    def row(q, err, t, tp, scale):
        rows.append(
            dict(
                where=where,
                quantity=q,
                error=float(err),
                tol=float(t),
                tol_prob=float(tp),
                rel=float(err / scale) if scale > 0 else float("inf"),
            )
        )

    row(
        "loss",
        abs(got["loss"] - ref["loss"]),
        tol["loss"],
        tol_prob["loss"],
        abs(ref["loss"]),
    )
    if "chi2_d" in got:
        e = np.abs(np.asarray(got["chi2_d"]) - ref["chi2_d"])
        k = int(np.argmax(e / tol["chi2_d"]))
        row(
            "chi2_d",
            e[k],
            tol["chi2_d"][k],
            tol_prob["chi2_d"][k],
            abs(ref["chi2_d"][k]),
        )
    for site in sorted(set(got["grad"]) - set(ref["grad"])):  # a site the reference has no value for cannot pass
        row(f"grad {site}", float("inf"), 0.0, 0.0, 0.0)
    for site, r in ref["grad"].items():
        if site not in got["grad"]:  # a site the implementation did not return fails, never drops out of the gate
            row(f"grad {site}", float("inf"), 0.0, 0.0, 0.0)
            continue
        r = np.asarray(r, float)
        e = np.asarray(got["grad"][site], float) - r
        if site == "sigma":
            k = int(np.argmax(np.abs(e) / tol["grad"]["sigma"]))
            row(
                "grad sigma",
                abs(e[k]),
                tol["grad"]["sigma"][k],
                tol_prob["grad"]["sigma"][k],
                float(np.max(np.abs(r))),
            )
        else:
            row(
                f"grad {site}",
                np.linalg.norm(e),
                tol["grad"][site],
                tol_prob["grad"][site],
                float(np.linalg.norm(r)),
            )
    for r_ in rows:
        r_["status"] = "PASS" if r_["error"] <= r_["tol"] else "FAIL"  # NaN fails
    return rows


# ------------------------------------------------------------------ Tier I


def tier1_scene(spec, tcfg, chunk=8192, toeplitz_eps=None):
    """Every mode and point of one scene: virgil's Toeplitz term against ``radio_exact``. Returns the rows and scene facts.
    ``toeplitz_eps`` overrides the precompute's kernel tolerance (a control; the budget keeps the documented one)."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    import virgil.radio as radio
    from virgil import Image

    from crosscheck import radio_exact as rx
    from crosscheck import toeplitz_scenes as ts

    case, truth = ts.make_case(spec)
    spread = spec["offsets"] == "spread"
    eps_impl = tcfg["eps_kernel"] + (tcfg["eps_psi_hat"] if spread else 0.0)
    n_fft = (2 * spec["npix"]) ** 2
    opt = dict(
        eta=np.log(np.maximum(truth["b"], 1e-300)),
        flux=truth["flux"],
        gain=truth["gain"],
        offset=truth["offset"],
    )
    part_opt = rx.partials(case, opt, chunk)
    pts = ts.points(case, truth, spec, rx.sigma_hat(part_opt["chi2_d"], case.n_d))
    data = radio.ComplexVisData.from_arrays(
        case.u,
        case.v,
        case.vis,
        case.weight,
        np.full(len(case.u), 230e9),
        ddid=case.ddid,
        obs=case.obs,
    )
    template = Image(
        np.zeros((spec["npix"], spec["npix"])), pixel_scale_mas=spec["pixel_mas"]
    )

    def scene(values):
        return Image(values["log_brightness"], pixel_scale_mas=spec["pixel_mas"])

    rows, facts = [], {}
    for mode in tcfg["modes"]:
        nuis = radio.CalibrationNuisance(
            data, ref_obs=0, sigmas=MODES[mode], alpha=False, offsets=spread
        )
        kw = (
            dict(offsets="spread", max_offset_mas=spec["offset_bound_mas"])
            if spread
            else {}
        )
        if toeplitz_eps is not None:
            kw["eps"] = toeplitz_eps
        tdata = data.toeplitz(template, nuis, **kw)
        facts = dict(
            n_points=len(case.u),
            npix=spec["npix"],
            nyquist_fraction=float(tdata.nyquist_fraction),
            margin=None if tdata.margin is None else float(tdata.margin),
        )  # no margin without offsets: integer lags are exact
        term = tdata.term(scene, nuis)
        vg = eqx.filter_jit(
            eqx.filter_value_and_grad(
                lambda v, t: (lambda c, ln: 0.5 * c + ln)(*t.chi2_and_log_norm(v))
            )
        )
        for pname, vals in pts.items():
            if pname not in tcfg["points"]:
                continue
            part = part_opt if pname == "optimum" else rx.partials(case, vals, chunk)
            sigma = vals["sigma"] if mode == "fitted" else None
            ref = rx.combine(case, part, mode, sigma)
            v = {
                "log_brightness": jnp.asarray(vals["eta"]),
                "cal.flux": jnp.asarray(vals["flux"]),
                "cal.gain": jnp.asarray(vals["gain"][1:]),
            }
            if spread:
                v["cal.offset"] = jnp.asarray(vals["offset"][1:])
            if mode == "fitted":
                v["cal.sigma"] = jnp.asarray(sigma)
            loss, grad = vg(v, term)
            got = dict(loss=float(loss) + float(term.loss_constant), grad={})
            for k, g in grad.items():
                site = SITES[k]
                got["grad"][site] = np.asarray(g)
            ref_g = dict(
                ref["grad"],
                gain=ref["grad"]["gain"][1:],
                offset=ref["grad"]["offset"][1:],
            )
            if mode == "fixed":
                got["chi2_d"] = np.asarray(tdata.chi2_by_ddid(scene, v, nuis)[0])
            st = rx.stats(case, float(vals["flux"]), vals["gain"], vals["eta"])
            eps_ref = rx.phase_error(case, vals["offset"]) + rx.gamma(case.npix**2)
            args = (
                st,
                ref["chi2_d"],
                ref["loss"],
                mode,
                sigma,
                float(vals["flux"]),
                vals["gain"],
                eps_impl,
                n_fft,
                eps_ref,
            )
            tol, tol_prob = (
                rx.tolerances(*args, worst_case=True),
                rx.tolerances(*args, worst_case=False),
            )
            if not spread:  # offsets are not fitted without spread: neither side has the site
                ref_g.pop("offset")
            rows += compare(
                f"{spec['name']}/{mode}/{pname}",
                dict(ref, grad=ref_g),
                got,
                tol,
                tol_prob,
            )
    return rows, facts


def cmd_tier1(args):
    cfg = criteria()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = args.scene or [s["name"] for s in cfg["tier1"]["scenes"]]
    for spec in cfg["tier1"]["scenes"]:
        if spec["name"] not in names:
            continue
        rows, facts = tier1_scene(spec, cfg["tier1"])
        write(
            out / f"tier1_{spec['name']}.json",
            dict(provenance(), scene=spec, facts=facts, rows=rows),
        )
        bad = [r for r in rows if r["status"] != "PASS"]
        print(
            f"{spec['name']}: {len(rows)} rows, {len(bad)} FAIL; worst error/tol "
            f"{max(r['error'] / r['tol'] for r in rows):.2e}; Nyquist {facts['nyquist_fraction']:.3f}, margin {facts['margin']}",
            flush=True,
        )


# ------------------------------------------------------------------ Tier II


def cmd_tier2_agree(args):
    """At the saved solution and the two moved points of ``alma_imlup_toeplitz_check`` (552da46): loss, chi2 per ddid and every
    gradient site, Toeplitz against direct at eps 1e-12, within the budget with ``eps_ref = eps_direct``."""
    cfg = criteria()
    import alma_imlup_toeplitz_check as old

    from crosscheck import radio_exact as rx

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t2 = cfg["tier2"]
    data, prob, values, tdata, pre = old.setup(args)
    term = tdata.term(prob.scene, prob.nuis)
    direct = tdata.direct_term(prob.scene, prob.nuis, chunk_size=args.chunk_size)
    case = rx.Case(
        np.asarray(data.u),
        np.asarray(data.v),
        np.asarray(data.vis),
        np.asarray(data.weight),
        np.asarray(data.ddid),
        np.asarray(data.obs),
        prob.npix,
        prob.pixel_mas,
    )
    pts = {
        "optimum": values,
        "moved": old.moved_point(values, args.max_offset_mas),
        "edge": old.moved_point(values, args.max_offset_mas, seed=1, edge=True),
    }
    eps_impl = cfg["tier1"]["eps_kernel"] + cfg["tier1"]["eps_psi_hat"]
    rows = []
    with old.va.ft_eps(t2["eps_direct"]):
        for pname, v in pts.items():
            lt, gt = old.vg_toeplitz(v, term)
            ld, gd = old.direct_vg()(v, direct)
            if set(gd) != set(gt) or set(gd) - set(SITES):  # every site is compared: none may drop out
                raise SystemExit(f"gradient sites differ or are unknown: toeplitz {sorted(gt)}, direct {sorted(gd)}, known {sorted(SITES)}")
            check = tdata.crosscheck(
                prob.scene, v, prob.nuis, chunk_size=args.chunk_size
            )
            s = np.asarray(v["cal.sigma"], float)
            chi2_t, chi2_d = (
                np.asarray(check["chi2"], float) * s**2,
                np.asarray(check["chi2_direct"], float) * s**2,
            )  # at unit scales
            flux = float(v["cal.flux"])
            gain = np.concatenate([[1.0], np.asarray(v["cal.gain"], float)])
            tol_args = (
                rx.stats(case, flux, gain, np.asarray(v["log_brightness"], float)),
                chi2_d,
                float(ld),
                "fitted",
                s,
                flux,
                gain,
                eps_impl,
                (2 * prob.npix) ** 2,
                t2["eps_direct"],
            )
            ref = dict(
                loss=float(ld),
                chi2_d=chi2_d,
                grad={SITES[k]: np.asarray(g) for k, g in gd.items()},
            )
            got = dict(
                loss=float(lt + term.loss_constant),
                chi2_d=chi2_t,
                grad={SITES[k]: np.asarray(g) for k, g in gt.items()},
            )
            rows += compare(
                f"R_mpol/{pname}",
                ref,
                got,
                rx.tolerances(*tol_args, worst_case=True),
                rx.tolerances(*tol_args, worst_case=False),
            )
    pre = dict(
        pre, nyquist_fraction=float(tdata.nyquist_fraction), margin=float(tdata.margin)
    )
    write(
        out / "tier2_agree.json",
        dict(provenance(), **pre, rows=rows, fit=str(args.fit)),
    )
    print(
        json.dumps(
            [dict(r, where=r["where"]) for r in rows if r["status"] != "PASS"]
            or "all tier2 agreement rows PASS",
            indent=1,
        )
    )


def cmd_tier2_time(args):
    criteria()
    import alma_imlup_toeplitz_check as old

    old.cmd_time(args)
    path = pathlib.Path(args.out) / f"time_{args.method}.json"
    rep = json.loads(path.read_text())
    write(path, dict(rep, **provenance()))


# ------------------------------------------------------------------ summary


def performance_rows(t, perf):
    """Speed-up and the Toeplitz path's peak, judged only when both timing runs were on the registered GPU type."""
    gpus = {t[m].get("device", "unknown") for m in ("toeplitz", "direct")}
    on_gpu = all(re.search(rf"\b{perf['gpu']}\b", g.upper()) for g in gpus)
    speedup = t["direct"]["s_per_vg"] / t["toeplitz"]["s_per_vg"]
    peak = t["toeplitz"]["peak_gb"]

    def status(ok):
        return ("PASS" if ok else "FAIL") if on_gpu else "NOT EVALUATED"

    return [
        dict(
            where="R_mpol",
            quantity=f"speed-up value+grad (direct at eps 1e-6), on {sorted(gpus)}",
            value=speedup,
            limit=f">= {perf['speedup_min']:g}x on {perf['gpu']}",
            status=status(speedup >= perf["speedup_min"]),
        ),
        dict(
            where="R_mpol",
            quantity="GPU peak, Toeplitz path",
            value=peak,
            limit=f"< {perf['peak_gb_max']:g} GB on {perf['gpu']}",
            status=status(peak < perf["peak_gb_max"]),
        ),
        dict(
            where="R_mpol",
            quantity="GPU peak, direct path (information)",
            value=t["direct"]["peak_gb"],
            limit="-",
            status="REPORT",
        ),
    ]


def verdict(statuses):
    if "FAIL" in statuses:
        return "NOT ACCEPTED"
    if "NOT EVALUATED" in statuses:
        return "INCOMPLETE"
    return "ACCEPTED"


def expected_tier1(spec, tcfg):
    """Every ``(where, quantity)`` row a complete Tier I scene writes."""
    spread = spec["offsets"] == "spread"
    out = set()
    for mode in tcfg["modes"]:
        for point in tcfg["points"]:
            if point == "edge" and not spread:
                continue
            sites = ["eta", "flux", "gain"] + (["offset"] if spread else []) + (["sigma"] if mode == "fitted" else [])
            where = f"{spec['name']}/{mode}/{point}"
            out |= {(where, "loss")} | {(where, f"grad {s}") for s in sites} | ({(where, "chi2_d")} if mode == "fixed" else set())
    return out


def expected_tier2(t2):
    """Every ``(where, quantity)`` row a complete Tier II agreement writes (fitted mode, free offsets)."""
    sites = ["eta", "flux", "gain", "offset", "sigma"]
    return {(f"R_mpol/{p}", q) for p in t2["points"] for q in ["loss", "chi2_d"] + [f"grad {s}" for s in sites]}


def missing_rows(rows, expected):
    """A FAIL row for every expected ``(where, quantity)`` an output lacks: a truncated output cannot pass."""
    have = {(r["where"], r["quantity"]) for r in rows}
    return [dict(where=w, quantity=q, value=None, limit="row present", status="FAIL") for w, q in sorted(expected - have)]


def cmd_summary(args):
    cfg = criteria()
    out = pathlib.Path(args.out)
    files = {
        p.name: json.loads(p.read_text())
        for p in sorted(out.glob("*.json"))
        if p.name != "reexam.json"
    }
    hashes = {f["criteria_sha256"] for f in files.values()}
    commits = {f["virgil_commit"] for f in files.values()}
    if hashes - {REGISTERED_SHA256}:
        raise SystemExit(f"outputs from other criteria: {hashes}")
    for key, what in (("run_id", "runs"), ("vv_commit", "virgil-validation commits")):
        found = {f.get(key) for f in files.values()}
        if len(found) > 1 or None in found and files:
            raise SystemExit(f"outputs from several {what} (or unrecorded), never combined: {found}")
    if len(commits) > 1:
        raise SystemExit(
            f"outputs from several virgil commits, never combined: {commits}"
        )
    gated, report = [], []
    for spec in cfg["tier1"]["scenes"]:
        f = files.get(f"tier1_{spec['name']}.json")
        if f is None:
            gated.append(
                dict(
                    where=spec["name"],
                    quantity="Tier I scene",
                    value=None,
                    limit="run",
                    status="NOT EVALUATED",
                )
            )
            continue
        gated += [
            dict(r, value=r["error"], limit=f"<= {r['tol']:.2e}") for r in f["rows"]
        ]
        gated += missing_rows(f["rows"], expected_tier1(spec, cfg["tier1"]))
        report += [
            dict(
                r,
                value=r["error"] / r["tol_prob"],
                limit="error / probabilistic budget",
                status="REPORT",
            )
            for r in f["rows"]
        ]
        if spec["offsets"] == "spread":
            fr = f["facts"]
            gated.append(
                dict(
                    where=spec["name"],
                    quantity="Nyquist fraction below margin",
                    value=fr["nyquist_fraction"],
                    limit=f"< {fr['margin']:.4f}",
                    status="PASS" if fr["nyquist_fraction"] < fr["margin"] else "FAIL",
                )
            )
    a = files.get("tier2_agree.json")
    if a is None:
        gated.append(
            dict(
                where="R_mpol",
                quantity="Tier II agreement",
                value=None,
                limit="run",
                status="NOT EVALUATED",
            )
        )
    else:
        gated += [
            dict(r, value=r["error"], limit=f"<= {r['tol']:.2e}") for r in a["rows"]
        ]
        gated += missing_rows(a["rows"], expected_tier2(cfg["tier2"]))
        report += [
            dict(
                r,
                value=r["error"] / r["tol_prob"],
                limit="error / probabilistic budget",
                status="REPORT",
            )
            for r in a["rows"]
        ]
        gated.append(
            dict(
                where="R_mpol",
                quantity="Nyquist fraction below margin",
                value=a["nyquist_fraction"],
                limit=f"< {a['margin']:.4f}",
                status="PASS" if a["nyquist_fraction"] < a["margin"] else "FAIL",
            )
        )
    t = {m: files.get(f"time_{m}.json") for m in ("toeplitz", "direct")}
    if None in t.values():
        gated.append(
            dict(
                where="R_mpol",
                quantity="performance",
                value=None,
                limit="run",
                status="NOT EVALUATED",
            )
        )
    else:
        perf = performance_rows(t, cfg["tier2"]["performance"])
        gated += [r for r in perf if r["status"] != "REPORT"]
        report += [r for r in perf if r["status"] == "REPORT"]
    rel = [
        r["rel"] for r in gated if "rel" in r and r["where"].split("/")[0] != "R_mpol"
    ]
    if rel:
        report.append(
            dict(
                where="Tier I",
                quantity="worst relative error (virgil's own tests hold 1e-10)",
                value=max(rel),
                limit="1e-10",
                status="REPORT",
            )
        )
    v = verdict([r["status"] for r in gated])

    def fmt(x):
        return "-" if x is None else f"{x:.3e}" if isinstance(x, float) else str(x)

    lines = [
        f"Toeplitz re-examination (G2). criteria sha256 {REGISTERED_SHA256}; virgil {', '.join(sorted(commits)) or 'none'}",
        "",
        f"{'status':13s} {'where':34s} {'quantity':40s} {'value':>11s}   limit",
    ]
    lines += [
        f"{r['status']:13s} {r['where']:34s} {r['quantity']:40s} {fmt(r['value']):>11s}   {r['limit']}"
        for r in gated
    ]
    worst = max(
        (r["value"] / r["tol"] for r in gated if r.get("tol")), default=float("nan")
    )
    lines += [
        "",
        f"worst gated error / budget: {worst:.3e}",
        "",
        "Reported, not gated:",
    ]
    lines += [
        f"{r['status']:13s} {r['where']:34s} {r['quantity']:40s} {fmt(r['value']):>11s}   {r['limit']}"
        for r in report
    ]
    lines += ["", f"VERDICT: {v}"]
    (out / "reexam.txt").write_text("\n".join(lines) + "\n")
    write(
        out / "reexam.json",
        dict(
            verdict=v,
            criteria_sha256=REGISTERED_SHA256,
            virgil_commit=list(commits),
            gated=gated,
            report=report,
        ),
    )
    print(
        "\n".join(
            lines[:3]
            + [x for x in lines if x.startswith(("FAIL", "NOT EVALUATED"))]
            + lines[-1:]
        )
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("tier1")
    q.add_argument("--out", required=True)
    q.add_argument(
        "--scene", action="append", help="a registered scene name (default all)"
    )
    for name in ("tier2-agree", "tier2-time"):
        q = sub.add_parser(name)
        q.add_argument("--data", required=True)
        q.add_argument("--fit", required=True)
        q.add_argument("--out", required=True)
        q.add_argument("--max-offset-mas", type=float, default=20.0)
        q.add_argument("--chunk-size", type=int, default=10**6)
        q.add_argument("--npix", type=int, default=None)
        if name == "tier2-time":
            q.add_argument("--method", choices=("toeplitz", "direct"), required=True)
            q.add_argument("--repeats", type=int, default=10)
    sub.add_parser("summary").add_argument("--out", required=True)
    sub.add_parser("hash")
    args = p.parse_args(argv)
    if args.cmd == "hash":
        print(criteria_hash())
        criteria()  # exits non-zero unless the hash is the registered one (the cluster preflight relies on this)
        return
    {
        "tier1": cmd_tier1,
        "tier2-agree": cmd_tier2_agree,
        "tier2-time": cmd_tier2_time,
        "summary": cmd_summary,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
