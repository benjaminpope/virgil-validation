"""external_bridge.cv_compare, numpy only: MPoL's held-out scores beside virgil's, per fold and annulus, and the
refusals. Split from tests/test_alma_mpol_cv.py, whose other checks go through external_bridge.mpol_worker and
virgil_bridge."""

import json

import numpy as np
import pytest

from _cv_common import _cv_dirs, _heldout
from external_bridge import cv_compare


@pytest.mark.validates("external_bridge.cv_compare", roots=["mathematics"], kind="control")
def test_cv_compare_flags_a_checkpoint_not_trained_without_its_fold(tmp_path):
    """The chain of custody: a score.json whose training asdf was another fold's, whose checkpoint was trained on another fold (an all-data
    or neighbouring-fold checkpoint scored by a path slip), or that records neither, and a fold without its training times, are all
    reported as problems; the correct chain reports none."""
    def problems(**kw):
        m, v = _cv_dirs(tmp_path / str(len(list(tmp_path.iterdir()))), **kw)
        mpol, virgil, summary = cv_compare.load_mpol(m), *cv_compare.load_virgil(v)
        return cv_compare.compare(mpol, virgil, summary, lam0=1.7)["problems"]

    assert problems() == []
    wrong_asdf = problems(record=lambda f: dict(train_fold=(f + 1) % 3, checkpoint_trained_fold=f))
    assert any("fold 0" in p and "training asdf" in p for p in wrong_asdf)
    wrong_ckpt = problems(record=lambda f: dict(train_fold=f, checkpoint_trained_fold=1 if f == 0 else f))
    assert len(wrong_ckpt) == 1 and "fold 0" in wrong_ckpt[0] and "training fold is 1" in wrong_ckpt[0]
    none = problems(record=lambda f: {}, times=False)
    assert sum("training asdf" in p for p in none) == 3 and sum("no record" in p for p in none) == 3 and sum("time.json" in p for p in none) == 3


@pytest.mark.validates("external_bridge.cv_compare", roots=["mathematics"], kind="reference")
def test_cv_compare_lines_up_the_two_codes_per_fold_and_per_annulus(tmp_path):
    """On synthetic scores in the layout both codes write: lambda* is read from ``cv_summary.json``, the matched lambda is the grid value
    nearest lambda_0 in log, the totals are the sums over folds, per-annulus numbers add up to the totals, raw chi^2/N is reported first
    in the markdown, and s_hat is the root of the raw chi^2/N."""
    m, v = _cv_dirs(tmp_path)
    mpol, virgil, summary = cv_compare.load_mpol(m), *cv_compare.load_virgil(v)
    res = cv_compare.compare(mpol, virgil, summary, lam0=1.7)
    assert res["lam_star"] == 1.0 and res["lam_match"] == 1.0 and res["problems"] == [] and res["folds"] == [0, 1, 2]
    assert cv_compare.compare(mpol, virgil, summary, lam0=4.0)["lam_match"] == 10.0  # nearest in log: 4 is 0.6 decades from 1 and 0.4 from 10
    for name in ("mpol", "virgil_lam_star", "virgil_lam_match"):
        tot = res["total"][name]
        per = [res["per_fold"][name][str(f)] for f in range(3)]
        assert tot["n"] == sum(p["n"] for p in per) == 900
        assert tot["nll"] == pytest.approx(sum(p["nll"] for p in per), rel=1e-12)
        assert tot["chi2_raw_per_n"] == pytest.approx(sum(p["chi2_raw_per_n"] * p["n"] for p in per) / tot["n"], rel=1e-12)
        assert sum(tot["annulus"]["n"]) == tot["n"]
        assert np.sum(np.asarray(tot["annulus"]["chi2_raw_per_n"]) * np.asarray(tot["annulus"]["n"])) / tot["n"] == pytest.approx(tot["chi2_raw_per_n"], rel=1e-12)
    t = res["total"]["mpol"]
    assert t["s_hat"] == pytest.approx(np.sqrt(t["chi2_raw_per_n"]), rel=1e-12)
    md = cv_compare.markdown(res)
    assert md.index("raw chi^2/N") < md.index("-ln L profiled") and "## Per dartboard annulus" in md and "MPoL (lam_ent 1e-09)" in md
    cv_compare.plot(res, tmp_path / "p.png")
    assert (tmp_path / "p.png").stat().st_size > 1000


@pytest.mark.validates("external_bridge.cv_compare", roots=["mathematics"], kind="control")
def test_cv_compare_flags_different_folds_missing_folds_and_unequal_counts(tmp_path):
    """Scores on different folds (another sha256), a lambda missing a fold, or a different number of held-out points are reported as problems, not compared silently."""
    m, v = _cv_dirs(tmp_path / "a", mpol_sha="other")
    mpol, virgil, summary = cv_compare.load_mpol(m), *cv_compare.load_virgil(v)
    assert any("different folds" in p for p in cv_compare.compare(mpol, virgil, summary, 1.0)["problems"])
    m, v = _cv_dirs(tmp_path / "b")
    (v / "lam_1" / "fold_2" / "cv.json").unlink()
    mpol, virgil, summary = cv_compare.load_mpol(m), *cv_compare.load_virgil(v)
    assert any("fold(s) [2]" in p for p in cv_compare.compare(mpol, virgil, summary, 1.0)["problems"])
    m, v = _cv_dirs(tmp_path / "c")
    rec = json.loads((m / "fold_0" / "score.json").read_text())
    rec["heldout"] = _heldout(np.random.default_rng(1), 250)
    (m / "fold_0" / "score.json").write_text(json.dumps(rec))
    mpol, virgil, summary = cv_compare.load_mpol(m), *cv_compare.load_virgil(v)
    assert any("held-out points" in p for p in cv_compare.compare(mpol, virgil, summary, 1.0)["problems"])
    with pytest.raises(ValueError, match="two score.json"):
        shutil_copy = tmp_path / "c" / "mpol" / "fold_0b"
        shutil_copy.mkdir()
        (shutil_copy / "score.json").write_text((m / "fold_0" / "score.json").read_text())
        cv_compare.load_mpol(m)
