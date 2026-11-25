"""JAX persistent-cache helpers and the thread-hold XLA flags in common.py."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import common  # noqa: E402


def test_xla_flags_hold_only_on_one_thread():
    assert common.XLA_HOLD_FLAGS in common.xla_flags(1)
    assert common.XLA_HOLD_FLAGS not in common.xla_flags(1, hold=False)
    assert common.XLA_HOLD_FLAGS not in common.xla_flags(2)
    assert "intra_op_parallelism_threads=2" in common.xla_flags(2)


def test_cache_cold_clears_populated_keeps(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "BENCH_HOME", tmp_path)
    env, mode = common.jax_cache_env("row/a b", "cold")
    d = Path(env["BENCH_JAX_CACHE_DIR"])
    assert mode == "cold" and d.is_dir() and d.parent == tmp_path / "jax_cache"
    (d / "entry").write_text("x")
    env2, mode2 = common.jax_cache_env("row/a b", "populated")
    assert mode2 == "populated" and (d / "entry").exists() and env2 == env
    common.jax_cache_env("row/a b", "cold")
    assert not (d / "entry").exists()


def test_cache_off_sets_nothing():
    assert common.jax_cache_env(None, "cold") == ({}, "off")
    assert common.jax_cache_env("k", "off") == ({}, "off")
