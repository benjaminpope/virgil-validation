"""common.py: time -l parsing, row schema, thread flags, repeat plan, noisy flag, and a dummy row end to end."""

import json

import pytest

import common

TIME_L = """
        1.50 real         1.20 user         0.10 sys
            99876352  maximum resident set size
                   0  average shared memory size
            75595776  peak memory footprint
"""


def test_parse_time_l():
    p = common.parse_time_l(TIME_L)
    assert p["wall"] == 1.5 and p["user"] == 1.2 and p["sys"] == 0.1
    assert p["peak_rss_mb"] == pytest.approx(99.876352)
    assert p["peak_footprint_mb"] == pytest.approx(75.595776)
    assert p["cpu_ratio"] == pytest.approx(1.3 / 1.5)


def test_parse_time_l_garbage_gives_none():
    p = common.parse_time_l("nothing here")
    assert p["wall"] is None and p["cpu_ratio"] is None


def test_parse_inner_takes_last_line():
    out = 'x\nBENCH_INNER {"a": 1}\nBENCH_INNER {"a": 2}\n'
    assert common.parse_inner(out) == {"a": 2}
    assert common.parse_inner("no tag") is None


def test_thread_env_and_limits():
    e1, e2 = common.thread_env(1), common.thread_env(2)
    assert e1["OMP_NUM_THREADS"] == "1" and "multi_thread_eigen=false" in e1["XLA_FLAGS"]
    assert e2["OPENBLAS_NUM_THREADS"] == "2" and "multi_thread_eigen=true" in e2["XLA_FLAGS"]
    assert e1["JAX_PLATFORMS"] == "cpu" and e2["JULIA_NUM_THREADS"] == "2"
    with pytest.raises(ValueError):
        common.thread_env(3)


@pytest.mark.parametrize("ratio,threads,flagged",
                         [(1.0, 1, False), (1.3, 1, True), (2.0, 2, False), (2.5, 2, True), (None, 1, False)])
def test_check_threads(ratio, threads, flagged):
    assert (common.check_threads(ratio, threads) == ["threads_not_held"]) is flagged


def test_repeat_plan():
    assert common.repeat_plan(1.0) == {"processes": 3, "warm": 5, "flags": []}
    assert common.repeat_plan(61.0) == {"processes": 3, "warm": 3, "flags": []}
    assert common.repeat_plan(601.0) == {"processes": 1, "warm": 1, "flags": ["single_run"]}


def test_warm_stats_and_ratio():
    a = common.warm_stats([2.0, 3.0, 4.0])
    b = common.warm_stats([1.0, 1.5, 2.0])
    assert a == {"median": 3.0, "min": 2.0, "max": 4.0}
    r = common.ratio_with_spread(a, b)
    assert r["ratio"] == pytest.approx(2.0) and r["lo"] == pytest.approx(1.0) and r["hi"] == pytest.approx(4.0)
    assert common.ratio_with_spread({"median": None}, b)["ratio"] is None


def _row(**kw):
    return common.new_row(case="c", comparison="binary", code="virgil", config="matched", t_cold=1.0, **kw)


def test_valid_row_and_each_failure():
    assert common.validate_row(_row()) == []
    assert common.validate_row({k: v for k, v in _row().items() if k != "t_warm"}) == ["missing key t_warm"]
    assert any("status" in e for e in common.validate_row(_row(status="great")))
    assert any("flag" in e for e in common.validate_row(_row(flags=["slow"])))
    assert any("threads" in e for e in common.validate_row(_row(threads=4)))
    no_cold = common.new_row(case="c", comparison="binary", code="v", config="matched")
    assert any("t_cold" in e for e in common.validate_row(no_cold))
    assert any("reason" in e for e in common.validate_row(_row(status="not_run")))
    assert common.validate_row(_row(status="not_run", settings={"reason": "budget"})) == []
    bad_cmp = common.new_row(case="c", comparison="x", code="v", config="matched", t_cold=1.0)
    assert any("comparison" in e for e in common.validate_row(bad_cmp))


def test_write_row_rejects_invalid_and_numbers_files(tmp_path):
    with pytest.raises(ValueError):
        common.write_row(_row(status="nope"), tmp_path)
    p0 = common.write_row(_row(), tmp_path)
    p1 = common.write_row(_row(), tmp_path)
    assert p0 != p1 and json.loads(p0.read_text())["case"] == "c"
    assert common.check_rows(tmp_path) == {}
    (tmp_path / "bad.json").write_text(json.dumps({"case": "x"}))
    assert "bad.json" in common.check_rows(tmp_path)


def test_flag_noisy():
    rows = [_row(t_calib=t) for t in (1.0, 1.0, 1.1, 2.0)]
    for r in rows:
        r["language"] = "python"
    common.flag_noisy(rows)
    assert [("noisy" in r["flags"]) for r in rows] == [False, False, False, True]


def test_manifest_verify(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"files": {"a.bin": common.file_sha256(f)}}))
    assert common.verify_manifest(man, tmp_path) == []
    f.write_bytes(b"abd")
    assert common.verify_manifest(man, tmp_path) == ["a.bin"]
    assert common.manifest_hash(tmp_path / "missing.json") is None


def test_machine_state_keys():
    s = common.machine_state()
    assert {"on_ac", "low_power_mode", "pyslot_busy", "load_avg", "ok", "problems"} <= set(s)


def test_provenance_has_core_fields():
    p = common.provenance()
    assert {"vv_commit", "virgil_pin", "versions", "macos", "chip", "thread_env"} <= set(p)
    assert "numpy" in p["versions"]


def test_timed_process_error_and_timeout():
    bad = common.timed_process(["/bin/sh", "-c", "exit 3"], timeout=20)
    assert bad["status"] == "error" and bad["returncode"] == 3
    slow = common.timed_process(["/bin/sh", "-c", "sleep 5"], timeout=1)
    assert slow["status"] == "timeout"


def test_dummy_row_end_to_end(tmp_path):
    """A fresh process under /usr/bin/time -l, inner timings parsed, row valid and complete."""
    path = common.dummy_row(tmp_path)
    row = json.loads(path.read_text())
    assert common.validate_row(row) == []
    assert row["status"] == "ok" and len(row["t_warm"]) == 3 and row["t_cold"] > 0
    assert row["peak_footprint_mb"] and row["peak_footprint_mb"] > 5 and row["peak_rss_mb"] > 5
    assert row["t_process"] is not None and row["t_process"] >= 0
    assert row["cpu_ratio"] is not None and row["threads"] == 1
    assert row["provenance"]["thread_env"]["OMP_NUM_THREADS"] == "1"
    assert row["settings"].get("dummy") is True and row["t_calib"] > 0
