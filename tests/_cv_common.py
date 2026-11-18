"""Shared by tests/test_cv_compare.py and tests/test_alma_mpol_cv.py: a held-out record and the two directories
``cv_compare`` reads. Imports only external_bridge.cv_folds."""

import json

import numpy as np

from external_bridge import cv_folds


def _heldout(rng, n, scale=1.0, shift=0.0):
    """A ``heldout`` dict of ``n`` points at random ddids and annuli: per-point -ln L, chi^2 at the quoted weights and at the fitted s."""
    dd, an = rng.integers(0, 26, n), rng.integers(0, 12, n)
    w = np.exp(rng.normal(5, 0.3, n))
    r2 = rng.exponential(2.0, n) * scale
    s = 1.0 + 0.2 * rng.uniform(size=26)
    t = lambda ids, size, nll, chi_fit: cv_folds.tally(ids, size, nll, r2, chi_fit)  # noqa: E731
    nll_fit = 0.5 * r2 / s[dd] ** 2 + 2 * np.log(s[dd]) + cv_folds.LN2PI - np.log(w) + shift
    nll_s1 = 0.5 * r2 + cv_folds.LN2PI - np.log(w) + shift
    fit = r2 / s[dd] ** 2
    ho = dict(by_ddid=t(dd, 26, nll_fit, fit), by_annulus=t(an, 12, nll_fit, fit), by_ddid_s1=t(dd, 26, nll_s1, r2), by_annulus_s1=t(an, 12, nll_s1, r2))
    ho.update(nll=float(nll_fit.sum()), chi2_s1=float(r2.sum()), chi2_fit=float(fit.sum()))
    return ho


def _cv_dirs(tmp_path, lams=(0.1, 1.0, 10.0), k=3, sha="abc", n=300, mpol_sha=None, record=None, times=True):
    """``record(f)`` returns fold ``f``'s extra score.json fields (default: the provenance of a correct chain, and stage time records)."""
    v, m = tmp_path / "cv", tmp_path / "mpol"
    for lam in lams:
        for f in range(k):
            d = v / f"lam_{lam:g}" / f"fold_{f}"
            d.mkdir(parents=True)
            (d / "cv.json").write_text(json.dumps(dict(lam=lam, fold=f, folds_sha256=sha, n_test=n, heldout=_heldout(np.random.default_rng(10 * f + int(lam * 10)), n, scale=1.0 + 0.1 * abs(np.log10(lam))))))
    (v / "cv_summary.json").write_text(json.dumps(dict(lam_star=1.0)))
    for f in range(k):
        d = m / f"fold_{f}"
        d.mkdir(parents=True)
        extra = record(f) if record else dict(train_fold=f, checkpoint_trained_fold=f)
        (d / "score.json").write_text(json.dumps(dict(fold=f, folds_sha256=mpol_sha or sha, heldout=_heldout(np.random.default_rng(100 + f), n, scale=1.3), **extra)))
        if times:
            for st, wall in ((0, 7000.0), (1, 14000.0)):
                (d / f"stage{st}.time.json").write_text(json.dumps(dict(fold=f, stage=st, exit=0, epochs=10, wall_s=wall)))
    return m, v
