"""MPoL's held-out cross-validation scores beside virgil's, on the same dartboard folds, numpy only.

Inputs: the ``score.json`` of each fold (``mpol_worker.score_fold``: MPoL's checkpoint trained on that fold's training asdf)
and the ``cv.json`` of each (lambda, fold) task of ``alma_imlup_maxent.py cv-task`` (with ``cv_summary.json`` of ``cv-aggregate``,
which names lambda*). Both hold the same ``heldout`` layout (``virgil_bridge.alma.score``: ``nll`` at the fitted s, ``chi2_s1``,
``by_ddid`` / ``by_annulus`` tallies of n, nll, chi2_raw, chi2_fit, and the s = 1 versions), so the numbers are comparable without
conversion. Reported first: raw chi^2/N = ``chi2_s1 / (2 n)`` (the quoted weights, s = 1, N points of two components each), then
the held-out -ln L at each code's fitted error scale and with one global scale profiled out per set (``cv_folds.profiled``),
in total and per dartboard annulus. A lower -ln L is a better prediction of the held-out points.

virgil is read at two strengths: lambda* (its own cross-validation choice) and the grid value nearest, in log, to ``lam0``, the
F-equivalent of MPoL's ``lam_ent`` (decision C11's conversion, ``virgil_bridge.alma.lam_f_equivalent``). MPoL has the one
strength it was trained at.
"""

import json
import pathlib

import numpy as np

try:
    from .cv_folds import profiled
except ImportError:
    from cv_folds import profiled


def load_mpol(mpol_dir):
    """``{fold: score.json}`` under ``mpol_dir``."""
    recs = [json.loads(p.read_text()) for p in sorted(pathlib.Path(mpol_dir).rglob("score.json"))]
    out = {int(r["fold"]): r for r in recs}
    if len(out) != len(recs):
        raise ValueError(f"two score.json for one fold under {mpol_dir}")
    return out


def load_virgil(cv_dir):
    """``({(lam, fold): cv.json}, cv_summary or None)`` under ``cv_dir``."""
    root = pathlib.Path(cv_dir)
    recs = {(float(r["lam"]), int(r["fold"])): r for r in (json.loads(p.read_text()) for p in sorted(root.rglob("cv.json")))}
    summ = root / "cv_summary.json"
    return recs, (json.loads(summ.read_text()) if summ.exists() else None)


def pick_lams(recs, summary, lam0):
    """``(lam_star, lam_match)``: lambda* from ``cv_summary.json`` (else the lambda with the smallest summed held-out -ln L among those
    with every fold), and the grid lambda nearest in log to ``lam0``."""
    lams = sorted({lam for lam, _ in recs})
    folds = sorted({f for _, f in recs})
    if summary is not None:
        star = float(summary["lam_star"])
    else:
        full = [lam for lam in lams if all((lam, f) in recs for f in folds)]
        if not full:
            raise ValueError("no lambda has every fold")
        star = min(full, key=lambda lam: sum(recs[(lam, f)]["heldout"]["nll"] for f in folds))
    match = min(lams, key=lambda lam: abs(np.log(lam / lam0)))
    return star, float(match)


def _sums(tally, key):
    return np.asarray(tally[key], float)


def _measures(ho):
    """The numbers of one held-out set (one fold of one code, or their sum): n, raw chi^2/N, chi^2/N at the fitted s, -ln L at the fitted s,
    -ln L with a global s profiled, and that s, in total and per annulus."""
    n = float(sum(ho["by_ddid"]["n"]))
    nll_s1, chi2_raw = float(sum(ho["by_ddid_s1"]["nll"])), float(sum(ho["by_ddid_s1"]["chi2_raw"]))
    prof, s_hat = profiled(n, nll_s1, chi2_raw)
    an, an1 = ho["by_annulus"], ho["by_annulus_s1"]
    pa, sa = profiled(_sums(an1, "n"), _sums(an1, "nll"), _sums(an1, "chi2_raw"))
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_a = _sums(an1, "chi2_raw") / (2 * _sums(an1, "n"))
        fit_a = _sums(an, "chi2_fit") / (2 * _sums(an, "n"))
    return dict(n=n, chi2_raw_per_n=chi2_raw / (2 * n), chi2_fit_per_n=float(sum(ho["by_ddid"]["chi2_fit"])) / (2 * n), nll=float(sum(ho["by_ddid"]["nll"])),
                nll_profiled=float(prof), s_hat=float(s_hat),
                annulus=dict(n=_sums(an, "n").tolist(), chi2_raw_per_n=raw_a.tolist(), chi2_fit_per_n=fit_a.tolist(), nll=_sums(an, "nll").tolist(),
                             nll_profiled=pa.tolist(), s_hat=sa.tolist()))


def _add(hos):
    """Sum of the tallies of several folds' ``heldout`` dicts."""
    out = {}
    for name in ("by_ddid", "by_annulus", "by_ddid_s1", "by_annulus_s1"):
        out[name] = {k: np.sum([np.asarray(h[name][k], float) for h in hos], 0).tolist() for k in hos[0][name]}
    return out


def compare(mpol, virgil, summary, lam0, lam_ent=1e-9):
    """The comparison as a dict (JSON-able); ``mpol`` is ``load_mpol``'s dict, ``virgil`` and ``summary`` ``load_virgil``'s."""
    folds = sorted(mpol)
    if not folds:
        raise ValueError("no MPoL scores")
    lam_star, lam_match = pick_lams(virgil, summary, lam0)
    problems = []
    shas = {mpol[f]["folds_sha256"] for f in folds} | {virgil[(lam, f)]["folds_sha256"] for lam in (lam_star, lam_match) for f in folds if (lam, f) in virgil}
    if len(shas) != 1:
        problems.append(f"the two codes were scored on different folds (sha256 {sorted(shas)})")
    sources = {"mpol": {f: mpol[f]["heldout"] for f in folds}}
    for name, lam in (("virgil_lam_star", lam_star), ("virgil_lam_match", lam_match)):
        missing = [f for f in folds if (lam, f) not in virgil]
        if missing:
            problems.append(f"virgil has no cv.json for lambda {lam:g} fold(s) {missing}")
        sources[name] = {f: virgil[(lam, f)]["heldout"] for f in folds if (lam, f) in virgil}
    for f in folds:
        n_m = int(sum(mpol[f]["heldout"]["by_ddid"]["n"]))
        for name in ("virgil_lam_star", "virgil_lam_match"):
            if f in sources[name] and int(sum(sources[name][f]["by_ddid"]["n"])) != n_m:
                problems.append(f"fold {f}: {n_m} held-out points for MPoL, {int(sum(sources[name][f]['by_ddid']['n']))} for {name}")
    res = dict(folds=folds, lam_star=lam_star, lam_match=lam_match, lam0=float(lam0), lam_ent=float(lam_ent), folds_sha256=sorted(shas), problems=problems,
               codes={}, per_fold={}, total={})
    for name, by_fold in sources.items():
        full = [f for f in folds if f in by_fold]
        res["codes"][name] = dict(lam=None if name == "mpol" else (lam_star if name.endswith("star") else lam_match), folds=full)
        res["per_fold"][name] = {str(f): _measures(by_fold[f]) for f in full}
        res["total"][name] = _measures(_add([by_fold[f] for f in full])) if full else None
    # nll summed over folds is the CV score, as cv-aggregate's `sum`
    res["mpol_training"] = {str(f): {k: mpol[f].get(k) for k in ("steps", "wall_s", "n_train")} for f in folds}
    return res


def markdown(res):
    """The report: raw chi^2/N first, then -ln L, in total, per fold and per annulus."""
    names = [n for n in ("mpol", "virgil_lam_star", "virgil_lam_match") if res["total"].get(n)]
    label = {"mpol": f"MPoL (lam_ent {res['lam_ent']:g})", "virgil_lam_star": f"virgil at lambda* = {res['lam_star']:.4g}",
             "virgil_lam_match": f"virgil at the grid lambda nearest MPoL's, {res['lam_match']:.4g} (lambda_0 = {res['lam0']:.4g})"}
    md = ["# MPoL and virgil on the same held-out dartboard folds", "",
          f"Folds {res['folds']} (sha256 {', '.join(s[:16] for s in res['folds_sha256'])}). Raw chi^2/N is the held-out chi^2 with the quoted weights (s = 1) over 2 N real "
          "components, so 1 is a perfect error model. -ln L is summed over the folds (lower is better); 'profiled' puts one error scale per set at its "
          "maximum-likelihood value (the s_hat column).", ""]
    if res["problems"]:
        md += ["**Problems**", ""] + [f"- {p}" for p in res["problems"]] + [""]
    md += ["## Total over the folds", "", "| code | N held out | raw chi^2/N (s = 1) | chi^2/N at fitted s | -ln L at fitted s | -ln L profiled | s_hat |", "|---|---|---|---|---|---|---|"]
    for n in names:
        t = res["total"][n]
        md.append(f"| {label[n]} | {t['n']:.0f} | {t['chi2_raw_per_n']:.6f} | {t['chi2_fit_per_n']:.6f} | {t['nll']:.2f} | {t['nll_profiled']:.2f} | {t['s_hat']:.5f} |")
    if "mpol" in names:
        md += ["", "Held-out -ln L of virgil minus MPoL's (negative: virgil predicts the held-out points better):", ""]
        for n in names[1:]:
            t, m = res["total"][n], res["total"]["mpol"]
            md.append(f"- {label[n]}: at fitted s {t['nll'] - m['nll']:+.2f}; profiled {t['nll_profiled'] - m['nll_profiled']:+.2f}; "
                      f"raw chi^2/N {t['chi2_raw_per_n'] - m['chi2_raw_per_n']:+.6f}")
    md += ["", "## Per fold", "", "| fold | code | N | raw chi^2/N | -ln L at fitted s | -ln L profiled |", "|---|---|---|---|---|---|"]
    for f in res["folds"]:
        for n in names:
            t = res["per_fold"][n].get(str(f))
            if t:
                md.append(f"| {f} | {label[n]} | {t['n']:.0f} | {t['chi2_raw_per_n']:.6f} | {t['nll']:.2f} | {t['nll_profiled']:.2f} |")
    md += ["", "## Per dartboard annulus (summed over the folds; annulus 0 is the shortest baselines)", ""]
    n_ann = len(res["total"][names[0]]["annulus"]["n"])
    md += ["| annulus | N | " + " | ".join(f"raw chi^2/N, {n}" for n in names) + " | " + " | ".join(f"-ln L profiled, {n}" for n in names) + " |",
           "|---|---|" + "---|" * (2 * len(names))]
    for a in range(n_ann):
        n_pts = res["total"][names[0]]["annulus"]["n"][a]
        row = [f"{res['total'][n]['annulus']['chi2_raw_per_n'][a]:.5f}" for n in names] + [f"{res['total'][n]['annulus']['nll_profiled'][a]:.2f}" for n in names]
        md.append(f"| {a} | {n_pts:.0f} | " + " | ".join(row) + " |")
    if "mpol" in names:
        md += ["", "Annuli where virgil predicts the held-out points better than MPoL (profiled -ln L lower), "
               + ", ".join(f"{label[n].split(' =')[0].split(' (')[0]}: {[a for a in range(n_ann) if res['total'][n]['annulus']['nll_profiled'][a] < res['total']['mpol']['annulus']['nll_profiled'][a]]}" for n in names[1:]) + ".", ""]
    return "\n".join(md) + "\n"


def plot(res, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in ("mpol", "virgil_lam_star", "virgil_lam_match") if res["total"].get(n)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for n in names:
        a = res["total"][n]["annulus"]
        axes[0].plot(a["chi2_raw_per_n"], marker="o", label=n)
        base = np.asarray(res["total"]["mpol"]["annulus"]["nll_profiled"]) if "mpol" in names else 0.0
        axes[1].plot(np.asarray(a["nll_profiled"]) - base, marker="o", label=n)
    axes[0].set_xlabel("dartboard annulus")
    axes[0].set_ylabel("held-out raw chi^2/N (s = 1)")
    axes[1].set_xlabel("dartboard annulus")
    axes[1].set_ylabel("profiled -ln L minus MPoL's")
    axes[1].axhline(0, color="0.6", lw=0.8)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
