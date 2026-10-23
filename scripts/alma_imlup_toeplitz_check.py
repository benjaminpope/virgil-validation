"""IM Lup: the Toeplitz radio χ² (virgil#516) against the direct term at a saved solution, and their speed.

    python scripts/alma_imlup_toeplitz_check.py agree --data DIR --fit DIR --out DIR [--max-offset-mas 20]
    python scripts/alma_imlup_toeplitz_check.py time --method {toeplitz,direct} --data DIR --fit DIR --out DIR
    python scripts/alma_imlup_toeplitz_check.py summary --out DIR

For virgil-validation ``scripts/``. ``--data`` holds ``ddid_NN.npz`` (``alma_imlup_read``); ``--fit`` is a config-R
fit directory (``values.npz``, ``result.json``), e.g. R_mpol. Config R as ``va.build_problem``: MPoL's 1028² x 5 mas
grid at its half-pixel map offset, a 30 mas taper, alpha fixed at 0, fitted gains, offsets and error scales. Only the
data term is compared (no entropy). float64 throughout; the direct term uses the NUFFT at eps 1e-12 (EPS_SCORE, C7).

``agree`` (one process): precompute, χ² per ddid, loss, loss difference to a perturbed image, gradient per site.
``time`` (one process per method, so the GPU peak memory is that method's): compile and run value+grad.
``summary``: one table from the JSON files.
"""

import argparse
import json
import pathlib
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

    return dict(jax=jax.__version__, virgil=m.version("virgil-astro"), nufftax=m.version("nufftax"),
                devices=[str(d) for d in jax.devices()])


def peak_gb():
    stats = jax.local_devices()[0].memory_stats() or {}
    return stats.get("peak_bytes_in_use", float("nan")) / 1e9


def setup(args):
    """Data, the R problem's scene and nuisance, the saved values, the Toeplitz precompute (timed)."""
    from virgil import Image

    data = va.load_dir(args.data)
    prob = va.build_problem(data, "R", lam=va.lam_r(len(data.u)), npix=args.npix, chunk_size=args.chunk_size)
    z = np.load(pathlib.Path(args.fit) / "values.npz")
    values = {k: jnp.asarray(z[k]) for k in z.files if k in prob.priors}
    shift = -prob.pixel_mas / 2.0  # va.grid_scene's half-pixel map offset
    template = Image(np.zeros((prob.npix, prob.npix)), pixel_scale_mas=prob.pixel_mas, dra=shift, ddec=shift)
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


@eqx.filter_jit
def loss_direct(term, values):
    r, log_norm = term.residuals_and_log_norm(values)
    return 0.5 * jnp.sum(r**2) + log_norm


vg_toeplitz = eqx.filter_jit(eqx.filter_value_and_grad(lambda v, t: loss_toeplitz(t, v)))
vg_direct = eqx.filter_jit(eqx.filter_value_and_grad(lambda v, t: loss_direct(t, v)))


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
        ld, gd = vg_direct(values, direct)
        dt = float(loss_toeplitz(term, v2) - loss_toeplitz(term, values))
        dd = float(loss_direct(direct, v2) - loss_direct(direct, values))
    grad_rel = {k: float(jnp.max(jnp.abs(gt[k] - gd[k])) / jnp.max(jnp.abs(gd[k]))) for k in gd}
    report = dict(versions=versions(), fit=str(args.fit), **pre,
                  chi2=np.asarray(check["chi2"]).tolist(), chi2_direct=np.asarray(check["chi2_direct"]).tolist(),
                  loss=float(lt + term.loss_constant), loss_direct=float(ld), max_rel=check["max_rel"],
                  dloss_toeplitz=dt, dloss_direct=dd, dloss_rel=abs(dt - dd) / abs(dd), dloss_abs=abs(dt - dd),
                  grad_rel=grad_rel, peak_gb=peak_gb())
    (out / "agree.json").write_text(json.dumps(report, indent=1))
    say(json.dumps({k: report[k] for k in ("max_rel", "dloss_rel", "dloss_abs", "grad_rel")}, indent=1))


def cmd_time(args):
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data, prob, values, tdata, pre = setup(args)
    if args.method == "toeplitz":
        term, vg = tdata.term(prob.scene, prob.nuis), vg_toeplitz
    else:
        term, vg = tdata.direct_term(prob.scene, prob.nuis, chunk_size=args.chunk_size), vg_direct
        del tdata  # free the precompute before measuring the direct term's peak
    with va.ft_eps(EPS_DIRECT):
        t = time.time()
        jax.block_until_ready(vg(values, term))
        compile_s = time.time() - t
        t = time.time()
        for _ in range(args.repeats):
            jax.block_until_ready(vg(values, term))
        s_per_vg = (time.time() - t) / args.repeats
    report = dict(versions=versions(), method=args.method, compile_s=compile_s, s_per_vg=s_per_vg,
                  repeats=args.repeats, peak_gb=peak_gb(), precompute_s=pre["precompute_s"])
    (out / f"time_{args.method}.json").write_text(json.dumps(report, indent=1))
    say(f"{args.method}: compile {compile_s:.1f} s, {s_per_vg * 1e3:.1f} ms per value+grad, peak {report['peak_gb']:.2f} GB")


ACCEPT = dict(max_rel=1e-10, dloss_rel=1e-6, dloss_abs=1e-2, grad_rel=1e-8, speedup=30.0, peak_gb=20.0)  # toeplitz_ozstar_spec.md


def acceptance(a, t):
    """The acceptance table: ``(quantity, value text, limit text, passed)``, from agree.json ``a`` and the two time_*.json ``t``."""
    worst_grad = max(a["grad_rel"].values())
    speedup = t["direct"]["s_per_vg"] / t["toeplitz"]["s_per_vg"]
    peak = max(t["toeplitz"]["peak_gb"], t["direct"]["peak_gb"])
    return [
        ("chi2 per ddid + loss, max rel diff", f"{a['max_rel']:.2e}", "<= 1e-10", a["max_rel"] <= ACCEPT["max_rel"]),
        ("loss difference to perturbed image, rel", f"{a['dloss_rel']:.2e}", "<= 1e-6", a["dloss_rel"] <= ACCEPT["dloss_rel"]),
        ("loss difference to perturbed image, abs", f"{a['dloss_abs']:.2e}", "<= 1e-2", a["dloss_abs"] <= ACCEPT["dloss_abs"]),
        ("gradient, worst site, rel to its max", f"{worst_grad:.2e}", "<= 1e-8", worst_grad <= ACCEPT["grad_rel"]),
        ("Nyquist fraction below margin", f"{a['nyquist_fraction']:.3f} / {a['margin']:.4f}", "fraction < margin", a["nyquist_fraction"] < a["margin"]),
        ("speed-up per value+grad", f"{speedup:.0f}x", ">= 30x", speedup >= ACCEPT["speedup"]),
        ("GPU peak per process (worst of two)", f"{peak:.2f} GB", "< 20 GB", peak < ACCEPT["peak_gb"]),
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
        *[(f"  grad {k}", f"{v:.2e}") for k, v in a["grad_rel"].items()],
        ("value+grad toeplitz / direct (ms)", f"{t['toeplitz']['s_per_vg'] * 1e3:.1f} / {t['direct']['s_per_vg'] * 1e3:.1f}"),
        ("compile toeplitz / direct (s)", f"{t['toeplitz']['compile_s']:.1f} / {t['direct']['compile_s']:.1f}"),
        ("peak GPU toeplitz / direct (GB)", f"{t['toeplitz']['peak_gb']:.2f} / {t['direct']['peak_gb']:.2f}"),
    ]
    text = "\n".join(
        ["Acceptance (virgil#516 Toeplitz chi^2 against the direct NUFFT term, IM Lup at R_mpol):", ""]
        + [f"{'PASS' if ok else 'FAIL'}  {q:42s} {v:>22s}   accept {lim}" for q, v, lim, ok in table]
        + ["", "ALL ACCEPTED" if all(ok for *_, ok in table) else "NOT ACCEPTED: " + ", ".join(q for q, _, _, ok in table if not ok), "", "Detail:"]
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
