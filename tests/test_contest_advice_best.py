"""Gathering and selecting the best fit per contest dataset (scripts/contest_best.py,
used by the dashboard scripts/contest_advice.py), on tiny fake npz files."""

import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cb, ca = _load("contest_best"), _load("contest_advice")


def blob(n=9, x=4, y=4, total=1.0):
    im = np.zeros((n, n))
    im[y, x] = total
    return im


def campaign(path, chi2, scale=1.0, raw=None):
    np.savez(path, ref_image=blob(), ref_fov=9.0, best_chi2_red=chi2, error_scale=scale)
    if raw is not None:
        path.with_suffix(".txt").write_text(f"raw chi2/N={raw} (quoted)\n")


def sweep(path, raw, kept):
    np.savez(path, mean=blob(), pixel_scale_mas=1.0, kept=json.dumps([{}] * kept), raw_chi2_red=raw)
    path.with_suffix(".txt").write_text(f"provenance=informed\nraw chi2/N={raw} (quoted errors)\n")


def lcurve(path, image, raw):
    np.savez(path, images=np.stack([image, image]), index=json.dumps({"discrepancy": 1}), pixel_scale_mas=1.0)
    path.with_suffix(".txt").write_text(f"raw chi2/N={raw} (quoted errors)\nerror_scale=1.5 (kept)\n")


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_gather_and_select(tmp_path):
    r = tmp_path
    (r / "8562a39" / "campaign").mkdir(parents=True)
    (r / "f643e07" / "advice" / "T_a_sweep_r2").mkdir(parents=True)
    (r / "bench_x").mkdir()
    (r / "ensemble_y").mkdir()
    campaign(r / "8562a39" / "campaign" / "T_a_m0.npz", chi2=0.9, scale=1.05)
    campaign(r / "8562a39" / "campaign" / "T_a_m1.npz", chi2=0.9, scale=1.1, raw=1.3)  # the log's raw value wins
    sweep(r / "f643e07" / "advice" / "T_a_sweep_r2" / "T_a_sweep_r2.npz", raw=0.1, kept=2)  # too few members
    sweep(r / "f643e07" / "advice" / "T_a_sweep_r2" / "T_a_sweep_r2_b.npz", raw=0.5, kept=3)
    lcurve(r / "f643e07" / "advice" / "T_a_nan_r2.npz", np.full((9, 9), np.nan), raw=0.2)  # not finite
    lcurve(r / "f643e07" / "advice" / "T_a_neg_r2.npz", blob(total=2.0) - blob(x=1, total=0.5), raw=0.3)  # negative flux
    lcurve(r / "f643e07" / "advice" / "T_a_good_r2.npz", blob(total=7.0), raw=0.7)  # unnormalized is fine
    np.savez(r / "f643e07" / "advice" / "T_a_norecord_r2.npz", image=blob())  # no raw chi2/N anywhere
    campaign(r / "bench_x" / "T_a_m0.npz", chi2=0.001)  # phantoms are not contest data
    campaign(r / "ensemble_y" / "T_a_m0.npz", chi2=0.001)
    campaign(r / "8562a39" / "campaign" / "T_b_m0.npz", chi2=2.0)
    cands, skipped = cb.scan(r, ["T_a", "T_b", "T_a_c"], {"8562a39/campaign": "111", "T_a_good_r2": "222"})
    assert set(cands) == {"T_a", "T_b"}
    assert skipped == {"no recorded raw χ²/N": 1}
    assert {m["stem"] for m in cands["T_a"]} == {"T_a_m0", "T_a_m1", "T_a_sweep_r2", "T_a_sweep_r2_b", "T_a_nan_r2",
                                                  "T_a_neg_r2", "T_a_good_r2"}
    by = {m["stem"]: m for m in cands["T_a"]}
    assert by["T_a_m1"]["raw"] == 1.3 and by["T_a_m0"]["raw"] == 0.9
    assert by["T_a_m0"]["job"] == "111" and by["T_a_good_r2"]["job"] == "222"
    assert by["T_a_good_r2"]["scale"] == 1.5 and by["T_a_sweep_r2"]["informed"] and not by["T_a_m0"]["informed"]
    chosen, rejected = cb.select(cands["T_a"], n=3)
    assert [c["stem"] for c in chosen] == ["T_a_sweep_r2_b", "T_a_good_r2", "T_a_m0"]  # lowest raw chi2/N first
    assert all(abs(c["image"].sum() - 1.0) < 1e-12 for c in chosen)
    why = {m["stem"]: w for m, w in rejected}
    assert "kept members" in why["T_a_sweep_r2"] and why["T_a_nan_r2"] == "not finite" and why["T_a_neg_r2"] == "negative flux"


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_task_matching_and_prompt():
    assert cb.find_task("2008_agn_J_gp_r2", ["2008_agn", "2008_agn_J"]) == "2008_agn_J"
    assert cb.find_task("2008_agnx", ["2008_agn"]) is None
    source = '''x = 1
TASKS = [
    # 2004: released blind.
    dict(label="2004_data1", files=["a.fits"]),
    # 2008: tapered.
    *[dict(label=f"2008_{o}_{b}", files=["b.fits"], field=30.0,
           field_source="readme") for o in "ab" for b in "JH"],
]
'''
    entries = cb.task_entries(source)
    assert "released blind" in cb.task_text(entries, "2004_data1") and 'files=["a.fits"]' in cb.task_text(entries, "2004_data1")
    assert 'field_source="readme"' in cb.task_text(entries, "2008_a_J") and "tapered" in cb.task_text(entries, "2008_a_J")
    assert cb.task_text(entries, "2012_alp") == ""
    assert cb.prompt_text("2026", "K", None, None, "", "", "", "", "") == "no recorded prompt"
    text = cb.prompt_text("2004", "data1", "2004_data1", "blind", "", cb.task_text(entries, "2004_data1"), "fix it", "", "cmd")
    assert "[contests/manifest.yml" in text and "fix it" in text and "round-2 command lines" in text


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_bench_best_picks_lowest_lawson():
    md = """## spotted_star  (noise scatter 0.10 in ln Lawson)

| arm | n | raw chi2/N | NCC | Lawson | n fail | wins |
|---|---|---|---|---|---|---|
| comp_a | 12 | 1.17 | 0.85 | 0.19 | 3 | 1/9 |
| ellipse_star (best existing) | 12 | 1.18 | 0.945 | 0.0953 | 0 | 8/12 |

## thin_disk  (noise scatter 0.05 in ln Lawson)

| arm | n | raw chi2/N | NCC | Lawson | n fail | wins |
|---|---|---|---|---|---|---|
| baseline | 12 | 1.03 | 0.936 | 0.0826 | 0 | - |
| halo | 12 | 1.02 | 0.947 | 0.068 | 0 | 12/12 |
"""
    best = ca.bench_best(md)
    assert [(b["family"], b["arm"]) for b in best] == [("spotted_star", "ellipse_star (best existing)"), ("thin_disk", "halo")]
