"""Slurm-node platform path of common.py (BENCH_PLATFORM=slurm): wait4 runner and the node's machine check."""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import common  # noqa: E402


def test_wait4_run_captures_and_kills():
    out, err, rc, wall, user, sysc, rss, timed_out = common._wait4_run(
        [sys.executable, "-c", "import sys; print('hi'); print('e', file=sys.stderr)"], None, None, 30)
    assert out.strip() == "hi" and err.strip() == "e" and rc == 0 and not timed_out and rss > 0 and wall > 0
    out, err, rc, wall, *_, timed_out = common._wait4_run([sys.executable, "-c", "import time; time.sleep(60)"], None, None, 0.5)
    assert timed_out and rc is None and wall < 10


def test_timed_process_on_node(monkeypatch):
    monkeypatch.setattr(common, "ON_NODE", True)
    res = common.timed_process([sys.executable, "-c", "print('BENCH_INNER ' + '{\"t_inner_total\": 0.0}')"], wrapper="pyslot")
    assert res["status"] == "ok" and res["wall"] > 0 and res["peak_footprint_mb"] == res["peak_rss_mb"]


def test_node_state_rules(monkeypatch):
    monkeypatch.setattr(common, "ON_NODE", True)
    monkeypatch.setattr(common, "_node_state", common._node_state)
    if not Path("/proc/meminfo").exists():
        pytest.skip("Linux only")
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("BENCH_MAX_LOAD", "1000")
    assert common.machine_state()["ok"]
    monkeypatch.setenv("BENCH_MAX_LOAD", "-1")
    assert "node load" in common.machine_state()["problems"][0]


def test_select_missing_exclude_shard(tmp_path, monkeypatch):
    import argparse
    import json

    import matrix

    specs = matrix.ordered(matrix.finalise(matrix.build()))
    keys = [s.key for s in specs if s.runner not in ("untimed", "not_run")][:12]
    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(json.dumps({"key": k, "status": "not_run"}) for k in keys[:6]) + "\n"
                   + json.dumps({"key": keys[6], "proc": 1}) + "\n")
    ns = dict(missing_from=str(log), exclude_case=[], add_case=[], shard=None)
    got = matrix.select(specs, argparse.Namespace(**ns))
    assert {s.key for s in got} == set(keys[:6])                      # only the not_run keys, not the ones that ran
    ns["exclude_case"] = [got[0].case]
    assert all(s.case != got[0].case for s in matrix.select(specs, argparse.Namespace(**ns)))
    ns["exclude_case"] = []
    parts = []
    for k in (1, 2, 3):
        ns["shard"] = f"{k}/3"
        parts.append({s.key for s in matrix.select(specs, argparse.Namespace(**ns))})
    assert set().union(*parts) == set(keys[:6]) and sum(len(p) for p in parts) == 6      # a partition: every spec in exactly one shard
    monkeypatch.setattr(common, "DEVICE", "gpu")
    ns.update(missing_from=None, shard=None)
    gpu = matrix.select(specs, argparse.Namespace(**ns))
    assert gpu and all(s.code.startswith("virgil") and s.threads == 1 for s in gpu)
