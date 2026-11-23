"""summarise.py on synthetic rows: grouping, ratios, counts, table cells, --check coverage."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import summarise  # noqa: E402


def _row(code, t_cold, warm, status="ok", **kw):
    return common.new_row(case="B-sim-6-T2", comparison="binary", code=code, config="matched", status=status,
                          t_cold=t_cold, t_warm=warm, n_data=300, n_param=3, peak_footprint_mb=100.0,
                          settings={"task": "T2", "case_id": "B-sim-6", **({"reason": "x"} if status != "ok" else {})}, **kw)


def test_group_ratio_counts(tmp_path):
    rows = [_row("virgil", 2.0, [0.01, 0.02]), _row("virgil", 2.2, [0.01]), _row("pmoired", 0.1, [0.03, 0.05])]
    g = summarise.group(rows)
    pr = summarise.pairs(g)
    assert len(pr) == 1
    _, v, o = pr[0]
    r = summarise.ratios(v, o, "warm")
    assert r["ratio"] == pytest.approx(0.04 / 0.01) and r["lo"] == pytest.approx(0.03 / 0.02)
    c = summarise.faster_counts(g)
    assert c["warm"] == {"virgil_faster": 1} and c["cold_first_call"] == {"other_faster": 1}
    assert c["fresh"] == {"other_faster": 1}   # no import/load/init recorded: fresh == t_cold here


def test_fresh_adds_import_load_init():
    r = _row("virgil", 2.0, [0.01], t_import=1.5, t_load=0.25, t_init=0.25)
    assert summarise.fresh(r) == pytest.approx(4.0)
    assert summarise.fresh(_row("virgil", None, [], status="timeout")) is None


def test_counted_rules():
    assert summarise.counted(("binary", "B-sim-6", "T2"))
    assert not summarise.counted(("imaging", "I-sim-24", "ladder"))
    assert not summarise.counted(("imaging", "I-sim-24", "stop_default"))
    assert summarise.counted(("imaging", "I-sim-24", "per_eval"))
    assert summarise.counted(("imaging", "I-sim-48", "native"))
    assert not summarise.counted(("imaging", "I-sim-96", "native"))


def test_failed_row_is_printed_not_dropped():
    g = summarise.group([_row("virgil", None, [], status="timeout"), _row("pmoired", 0.1, [0.03])])
    tex = summarise.tex_table(g, {})
    assert "timeout" in tex and "B-sim-6 T2" in tex and "not run" in tex  # other headline rows absent -> "not run"


def test_check_coverage(tmp_path):
    for i, r in enumerate([_row("virgil", 1.0, [0.1]), _row("pmoired", 1.0, [0.1])]):
        common.write_row(r, tmp_path)
    rows, problems = summarise.check(tmp_path, tmp_path, strict_outputs=False)
    assert len(rows) == 2
    assert any("B-sim-60 virgil" in p for p in problems) and not any("B-sim-6 virgil" in p for p in problems)


def test_gpu_rows_are_separate(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    cpu = _row("virgil", 2.0, [0.02])
    gpu = _row("virgil", 3.0, [0.01])
    gpu["settings"] = {**gpu["settings"], "device": "gpu", "gpu": "NVIDIA A100"}
    for i, r in enumerate((cpu, gpu)):
        (raw / f"r{i}.json").write_text(json.dumps(r))
    assert [r["settings"].get("device") for r in summarise.load(raw)] == [cpu["settings"].get("device")]
    assert [r["settings"]["device"] for r in summarise.load(raw, gpu=True)] == ["gpu"]
    out = tmp_path / "gpu.csv"
    summarise.write_gpu_csv(summarise.group(summarise.load(raw, gpu=True)), summarise.group(summarise.load(raw)), out)
    text = out.read_text()
    assert "NVIDIA A100" in text and "t_warm_gpu" in text
