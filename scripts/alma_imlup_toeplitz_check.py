"""IM Lup: the Toeplitz radio χ² (virgil#516) against the direct term at a saved solution, and their speed.

    python scripts/alma_imlup_toeplitz_check.py agree --data DIR --fit DIR --out DIR [--max-offset-mas 20]
    python scripts/alma_imlup_toeplitz_check.py time --method {toeplitz,direct} --data DIR --fit DIR --out DIR
    python scripts/alma_imlup_toeplitz_check.py summary --out DIR

For virgil-validation ``scripts/``. ``--data`` holds ``ddid_NN.npz`` (``alma_imlup_read``); ``--fit`` is a config-R
fit directory (``values.npz``, ``result.json``), e.g. R_mpol. Config R as ``va.build_problem``: MPoL's 1028² x 5 mas
grid at its half-pixel map offset, a 30 mas taper, alpha fixed at 0, fitted gains, offsets and error scales. Only the
data term is compared (no entropy). float64 throughout. The agreement is measured against the direct NUFFT at eps 1e-12 (EPS_SCORE, C7); the
speed-up against the direct NUFFT at eps 1e-6 (EPS_FIT), the accuracy a fit runs at (the 1e-12 time is reported beside it).

``agree`` (one process): precompute, χ² per ddid, loss, loss difference to a perturbed image, gradient per site. The gradients are compared
twice: at the saved solution (reported) and at a point moved off it (ŝ x 1.1, gains x 1.05, offsets +-2 mas; gated), because at the optimum
dL/dŝ ~ 0 by cancellation and a difference relative to that gradient's own max measures roundoff. Each site's max |grad| is recorded.
A third point has every offset at +-0.9 x ``--max-offset-mas`` (random signs per observation and axis): the loss and every gradient there are gated too, as
one row, because the ±2 mas point barely exercises the free-offset spread grid (``offsets="spread"``) near the edge of its box.
``time`` (one process per method, so the GPU peak memory is that method's): compile and run value+grad. ``--method direct`` does not
build the Toeplitz precompute, so its peak is the direct term's alone (the larger of its two accuracies). The memory limit applies to the Toeplitz
path; the direct path's peak is reported as information.
``summary``: one table from the JSON files. The speed-up limit (>= 30x) is judged on an A100 only: the ``time`` runs record the GPU name
(``jax.devices()[0].device_kind``), and on any other device the speed-up row is labelled REPORT (value and GPU shown) and cannot make the run NOT ACCEPTED.
"""

import argparse
import json
import pathlib
import re
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import virgil_bridge.alma as va  # noqa: E402

EPS_DIRECT = 1e-12


def say(*a):
    print(*a, flush=True)


def versions():
    import importlib.metadata as m

    import virgil

    # virgil is a snapshot on PYTHONPATH on the cluster, so the installed distribution's version may not be the one imported
    try:
        dist = m.version("virgil-astro")
    except m.PackageNotFoundError:
        dist = None
    return dict(jax=jax.__version__, virgil=dist, virgil_from=virgil.__file__,
                nufftax=m.version("nufftax"), devices=[str(d) for d in jax.devices()])


def gpu_name():
    """The device the run is on, as JAX names it (e.g. ``NVIDIA A100-SXM4-40GB``; ``cpu`` without a GPU)."""
    return jax.devices()[0].device_kind


def peak_gb():
    stats = jax.local_devices()[0].memory_stats() or {}
    return stats.get("peak_bytes_in_use", float("nan")) / 1e9


def setup(args, precompute=True):
    """Data, the R problem's scene and nuisance, the saved values, the Toeplitz precompute (timed; ``tdata`` is None without it)."""
    from virgil import Image

    data = va.load_dir(args.data)
    prob = va.build_problem(data, "R", lam=va.lam_r(len(data.u)), npix=args.npix, chunk_size=args.chunk_size)
    z = np.load(pathlib.Path(args.fit) / "values.npz")
    values = {k: jnp.asarray(z[k]) for k in z.files if k in prob.priors}
    shift = -prob.pixel_mas / 2.0  # va.grid_scene's half-pixel map offset
    template = Image(np.zeros((prob.npix, prob.npix)), pixel_scale_mas=prob.pixel_mas, dra=shift, ddec=shift)
    tdata, pre = None, dict(precompute_s=None, n_points=len(data.u))
    if precompute:
        t = time.time()
        tdata = data.toeplitz(template, prob.nuis, offsets="spread", max_offset_mas=args.max_offset_mas)
        jax.block_until_ready(tdata.arrays)
        pre = dict(precompute_s=time.time() - t, nyquist_fraction=tdata.nyquist_fraction, margin=tdata.margin,
                   max_shift_pix=tdata.max_shift_pix, sizes=list(tdata.sizes),
                   arrays_gb=sum(x.nbytes for x in jax.tree.leaves(tdata.arrays)) / 1e9, n_points=len(data.u))
        say(f"precompute {pre['precompute_s']:.1f} s; Nyquist {pre['nyquist_fraction']:.3f} of margin {pre['margin']:.4f}; "
            f"sizes {pre['sizes']}; arrays {pre['arrays_gb']:.2f} GB")
    biggest = float(np.max(np.abs(np.asarray(values["cal.offset"]))))
    assert biggest < args.max_offset_mas, f"a fitted offset ({biggest:.2f} mas) is past --max-offset-mas"
    return data, prob, values, tdata, pre


# The terms are arguments, never closed over: a closure embeds every visibility in the program.
@eqx.filter_jit
def loss_toeplitz(term, values):
    chi2, log_norm = term.chi2_and_log_norm(values)
    return 0.5 * chi2 + log_norm  # less term.loss_constant


def _loss_direct(term, values):
    r, log_norm = term.residuals_and_log_norm(values)
    return 0.5 * jnp.sum(r**2) + log_norm


loss_direct = eqx.filter_jit(_loss_direct)
vg_toeplitz = eqx.filter_jit(eqx.filter_value_and_grad(lambda v, t: loss_toeplitz(t, v)))


def direct_vg():
    """A fresh jitted value+grad of the direct term. ``ft_eps`` is read when the program is traced and a jit cache ignores it, so each
    NUFFT accuracy needs its own function."""
    return eqx.filter_jit(eqx.filter_value_and_grad(lambda v, t: _loss_direct(t, v)))


def rel_max_diff(a, b):
    """max|a - b| / max|b|; 0 where both vanish, inf where only b does. A NaN in a or b stays NaN."""
    num, den = float(jnp.max(jnp.abs(a - b))), float(jnp.max(jnp.abs(b)))
    return num / den if den > 0 else (0.0 if num == 0 else float("inf"))


EDGE_FRACTION = 0.9  # the edge point's offsets are at +-EDGE_FRACTION x max_offset_mas, where the spread grid's range is stretched most


def moved_point(values, max_offset_mas, seed=0, edge=False):
    """``values`` moved off the optimum: ŝ x 1.1, gains x 1.05, each offset component +-2 mas (signs from ``seed``). At the fit's optimum
    dL/ds ~ 0 by cancellation (-2 chi2/s^3 + N/s), so a gradient compared "relative to its own max" divides roundoff by roundoff there.
    ``edge``: instead each offset component is set to +-0.9 x ``max_offset_mas`` (random signs per observation and axis from ``seed``), which exercises
    the free-offset spread grid (``offsets="spread"``) near the edge of its box; the ±2 mas point barely leaves the fitted offsets."""
    rng = np.random.default_rng(seed)
    off = np.asarray(values["cal.offset"], float)
    if edge:
        off = EDGE_FRACTION * max_offset_mas * rng.choice([-1.0, 1.0], size=off.shape)
        assert np.max(np.abs(off)) <= EDGE_FRACTION * max_offset_mas * (1 + 1e-12)
    else:
        off = off + 2.0 * rng.choice([-1.0, 1.0], size=off.shape)
    assert np.max(np.abs(off)) < max_offset_mas, f"the moved offsets ({np.max(np.abs(off)):.2f} mas) are past --max-offset-mas"
    return dict(values, **{"cal.sigma": 1.1 * values["cal.sigma"], "cal.gain": 1.05 * values["cal.gain"], "cal.offset": jnp.asarray(off)})


def grad_scales(g):
    """Per site, max |g|: the number a relative gradient difference is divided by."""
    return {k: float(jnp.max(jnp.abs(v))) for k, v in g.items()}


def cmd_agree(args):
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data, prob, values, tdata, pre = setup(args)
    term, direct = tdata.term(prob.scene, prob.nuis), tdata.direct_term(prob.scene, prob.nuis, chunk_size=args.chunk_size)
    rng = np.random.default_rng(0)
    v2 = dict(values, log_brightness=values["log_brightness"] + 1e-3 * rng.normal(size=values["log_brightness"].shape))
    with va.ft_eps(EPS_DIRECT):
        check = tdata.crosscheck(prob.scene, values, prob.nuis, chunk_size=args.chunk_size)
        lt, gt = vg_toeplitz(values, term)
        ld, gd = direct_vg()(values, direct)
        moved = moved_point(values, args.max_offset_mas)
        lt_m, gt_m = vg_toeplitz(moved, term)
        ld_m, gd_m = direct_vg()(moved, direct)
        edge = moved_point(values, args.max_offset_mas, seed=1, edge=True)
        lt_e, gt_e = vg_toeplitz(edge, term)
        ld_e, gd_e = direct_vg()(edge, direct)
        dt = float(loss_toeplitz(term, v2) - loss_toeplitz(term, values))
        dd = float(loss_direct(direct, v2) - loss_direct(direct, values))
    grad_rel = {k: rel_max_diff(gt[k], gd[k]) for k in gd}  # at the fit's optimum: reported, not gated (see moved_point)
    grad_rel_moved = {k: rel_max_diff(gt_m[k], gd_m[k]) for k in gd_m}  # the gated gradient comparison
    grad_rel_edge = {k: rel_max_diff(gt_e[k], gd_e[k]) for k in gd_e}  # the same at the edge-offset point (gated, with its loss)
    # The comparison is made here from the two paths' outputs; virgil's own crosscheck verdict is kept beside it, not used for the table.
    chi2, chi2_direct = np.asarray(check["chi2"], float), np.asarray(check["chi2_direct"], float)
    loss, loss_d = float(lt + term.loss_constant), float(ld)
    chi2_rel, loss_rel = float(np.max(np.abs(chi2 - chi2_direct) / np.abs(chi2_direct))), abs(loss - loss_d) / abs(loss_d)
    report = dict(versions=versions(), fit=str(args.fit), **pre,
                  chi2=chi2.tolist(), chi2_direct=chi2_direct.tolist(), loss=loss, loss_direct=loss_d,
                  chi2_rel=chi2_rel, loss_rel=loss_rel, max_rel=float(np.max([chi2_rel, loss_rel])), max_rel_virgil=float(check["max_rel"]),
                  dloss_toeplitz=dt, dloss_direct=dd, dloss_rel=abs(dt - dd) / abs(dd), dloss_abs=abs(dt - dd),
                  grad_rel=grad_rel, grad_scale=grad_scales(gd), grad_rel_moved=grad_rel_moved, grad_scale_moved=grad_scales(gd_m),
                  loss_rel_moved=abs(float(lt_m + term.loss_constant) - float(ld_m)) / abs(float(ld_m)),
                  loss_rel_edge=abs(float(lt_e + term.loss_constant) - float(ld_e)) / abs(float(ld_e)),
                  grad_rel_edge=grad_rel_edge, grad_scale_edge=grad_scales(gd_e),
                  moved="sigma x 1.1, gain x 1.05, offsets +-2 mas (seed 0)",
                  edge=f"sigma x 1.1, gain x 1.05, offsets +-{EDGE_FRACTION * args.max_offset_mas:g} mas (random signs per observation and axis, seed 1)",
                  peak_gb=peak_gb())
    (out / "agree.json").write_text(json.dumps(report, indent=1))
    say(json.dumps({k: report[k] for k in ("max_rel", "max_rel_virgil", "dloss_rel", "dloss_abs", "grad_rel", "grad_rel_moved", "loss_rel_edge", "grad_rel_edge")}, indent=1))


def time_vg(vg, term, values, repeats):
    """``(compile_s, s_per_vg)`` of the jitted value+grad ``vg``."""
    t = time.time()
    jax.block_until_ready(vg(values, term))
    compile_s = time.time() - t
    t = time.time()
    for _ in range(repeats):
        jax.block_until_ready(vg(values, term))
    return compile_s, (time.time() - t) / repeats


def cmd_time(args):
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data, prob, values, tdata, pre = setup(args, precompute=args.method == "toeplitz")
    report = dict(versions=versions(), gpu=gpu_name(), method=args.method, repeats=args.repeats, precompute_s=pre["precompute_s"])
    if args.method == "toeplitz":
        report["compile_s"], report["s_per_vg"] = time_vg(vg_toeplitz, tdata.term(prob.scene, prob.nuis), values, args.repeats)
    else:
        term = data.term(prob.scene, prob.nuis, chunk_size=args.chunk_size)  # tdata.direct_term's term (no mask), without the precompute
        with va.ft_eps(va.EPS_FIT):  # what a fit runs at: the speed-up is measured against this
            report["compile_s"], report["s_per_vg"] = time_vg(direct_vg(), term, values, args.repeats)
        with va.ft_eps(EPS_DIRECT):  # the scoring accuracy, for reference
            report["compile_s_score"], report["s_per_vg_score"] = time_vg(direct_vg(), term, values, args.repeats)
        report.update(eps_fit=va.EPS_FIT, eps_score=EPS_DIRECT)
    report["peak_gb"] = peak_gb()
    (out / f"time_{args.method}.json").write_text(json.dumps(report, indent=1))
    say(f"{args.method} on {report['gpu']}: compile {report['compile_s']:.1f} s, {report['s_per_vg'] * 1e3:.1f} ms per value+grad, peak {report['peak_gb']:.2f} GB"
        + (f"; at eps {EPS_DIRECT:g}: {report['s_per_vg_score'] * 1e3:.1f} ms" if args.method == "direct" else ""))


ACCEPT = dict(max_rel=1e-10, dloss_rel=1e-6, dloss_abs=1e-2, grad_rel=1e-8, speedup=30.0, peak_gb=20.0)  # toeplitz_ozstar_spec.md
SPEEDUP_GPU = "A100"  # the speed-up limit is judged on this GPU type only; on any other device it is reported


def speedup_gated(t):
    """Whether the speed-up limit applies: both timing runs were on an A100. Reports from before the GPU was recorded count as unknown."""
    gpus = {t[m].get("gpu", "unknown") for m in ("toeplitz", "direct")}
    return all(re.search(rf"\b{SPEEDUP_GPU}\b", g.upper()) for g in gpus), t["toeplitz"].get("gpu", "unknown")  # the model as a whole token: not an A1000


def acceptance(a, t):
    """The acceptance table: ``(quantity, value text, limit text, status)``, from agree.json ``a`` and the two time_*.json ``t``.
    ``status`` is True (PASS), False (FAIL) or None (REPORT: shown, not judged; the speed-up on a GPU other than an A100)."""
    worst_grad = float(np.max(np.asarray(list(a["grad_rel_moved"].values()), float)))  # at the moved point; NaN propagates, and fails below
    worst_edge = float(np.max(np.asarray(list(a["grad_rel_edge"].values()), float)))  # NaN propagates and fails, as above
    speedup = t["direct"]["s_per_vg"] / t["toeplitz"]["s_per_vg"]
    gated, gpu = speedup_gated(t)
    peak = t["toeplitz"]["peak_gb"]  # the limit is the Toeplitz path's; the dense direct reference's peak is reported, not gated
    return [
        ("chi2 per ddid + loss, max rel diff", f"{a['max_rel']:.2e}", "<= 1e-10", a["max_rel"] <= ACCEPT["max_rel"]),
        ("loss difference to perturbed image, rel", f"{a['dloss_rel']:.2e}", "<= 1e-6", a["dloss_rel"] <= ACCEPT["dloss_rel"]),
        ("loss difference to perturbed image, abs", f"{a['dloss_abs']:.2e}", "<= 1e-2", a["dloss_abs"] <= ACCEPT["dloss_abs"]),
        ("gradient at a moved point, worst site, rel", f"{worst_grad:.2e}", "<= 1e-8", worst_grad <= ACCEPT["grad_rel"]),
        (f"edge-offset point ({EDGE_FRACTION:g} x max offset): loss, worst grad, rel", f"{a['loss_rel_edge']:.1e} / {worst_edge:.1e}", "<= 1e-10 / <= 1e-8",
         a["loss_rel_edge"] <= ACCEPT["max_rel"] and worst_edge <= ACCEPT["grad_rel"]),
        ("Nyquist fraction below margin", f"{a['nyquist_fraction']:.3f} / {a['margin']:.4f}", "fraction < margin", a["nyquist_fraction"] < a["margin"]),
        ("speed-up per value+grad (direct at eps 1e-6)", f"{speedup:.0f}x",
         f">= 30x on {SPEEDUP_GPU}" if gated else f"report only on {gpu} (>= 30x on {SPEEDUP_GPU})", speedup >= ACCEPT["speedup"] if gated else None),
        ("GPU peak, Toeplitz path", f"{peak:.2f} GB", "< 20 GB", peak < ACCEPT["peak_gb"]),
    ]


def cmd_summary(args):
    out = pathlib.Path(args.out)
    a = json.loads((out / "agree.json").read_text())
    t = {m: json.loads((out / f"time_{m}.json").read_text()) for m in ("toeplitz", "direct")}
    table = acceptance(a, t)
    rows = [
        ("points", f"{a['n_points']:,}"),
        ("precompute (s)", f"{a['precompute_s']:.1f}"),
        ("Toeplitz arrays (GB)", f"{a['arrays_gb']:.2f}"),
        ("chi2 / loss max rel diff (virgil's crosscheck)", f"{a['max_rel_virgil']:.2e}"),
        *[(f"  grad {k}, at the optimum / moved: rel diff", f"{v:.2e} / {a['grad_rel_moved'][k]:.2e}") for k, v in a["grad_rel"].items()],
        *[(f"  grad {k}, max |grad| it is divided by", f"{a['grad_scale'][k]:.3e} / {a['grad_scale_moved'][k]:.3e}") for k in a["grad_rel"]],
        ("GPU (toeplitz / direct)", f"{t['toeplitz'].get('gpu', 'unknown')} / {t['direct'].get('gpu', 'unknown')}"),
        ("value+grad toeplitz / direct eps 1e-6 (ms)", f"{t['toeplitz']['s_per_vg'] * 1e3:.1f} / {t['direct']['s_per_vg'] * 1e3:.1f}"),
        ("direct at eps 1e-12 (ms), speed-up", f"{t['direct']['s_per_vg_score'] * 1e3:.1f}, {t['direct']['s_per_vg_score'] / t['toeplitz']['s_per_vg']:.0f}x"),
        ("compile toeplitz / direct (s)", f"{t['toeplitz']['compile_s']:.1f} / {t['direct']['compile_s']:.1f}"),
        ("peak GPU toeplitz / direct (GB; direct is information)", f"{t['toeplitz']['peak_gb']:.2f} / {t['direct']['peak_gb']:.2f}"),
        ("moved point", a["moved"]),
        ("edge-offset point", a["edge"]),
        *[(f"  grad {k}, edge point: rel diff / max |grad|", f"{v:.2e} / {a['grad_scale_edge'][k]:.3e}") for k, v in a["grad_rel_edge"].items()],
    ]
    text = "\n".join(
        ["Acceptance (virgil#516 Toeplitz chi^2 against the direct NUFFT term, IM Lup at R_mpol):", ""]
        + [f"{'REPORT' if ok is None else 'PASS' if ok else 'FAIL':<6s} {q:46s} {v:>22s}   accept {lim}" for q, v, lim, ok in table]
        + ["", "NOT ACCEPTED: " + ", ".join(q for q, _, _, ok in table if ok is False) if any(ok is False for *_, ok in table) else "ALL ACCEPTED", "", "Detail:"]
        + [f"{k:40s} {v}" for k, v in rows])
    (out / "toeplitz_check.txt").write_text(text + "\n")
    say(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("agree", "time"):
        q = sub.add_parser(name)
        q.add_argument("--data", required=True)
        q.add_argument("--fit", required=True)
        q.add_argument("--out", required=True)
        q.add_argument("--max-offset-mas", type=float, default=20.0,
                       help="the offsets' bound for the precompute; R_mpol's largest |offset| is 4.4 mas")
        q.add_argument("--chunk-size", type=int, default=10**6)
        q.add_argument("--npix", type=int, default=None, help="config R's 1028 unless set (a small smoke test)")
        if name == "time":
            q.add_argument("--method", choices=("toeplitz", "direct"), required=True)
            q.add_argument("--repeats", type=int, default=10)
    q = sub.add_parser("summary")
    q.add_argument("--out", required=True)
    args = p.parse_args(argv)
    {"agree": cmd_agree, "time": cmd_time, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
