"""Tests of the binary benchmark scaffolding (run from benchmarks/paper_performance: ``pytest tests binary``).

No test imports virgil or PMOIRED in this process; the smoke tests launch the runners, which spawn their own workers.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _shared as sh  # noqa: E402

common, md = sh.common, sh.md


def test_cases_yml_covers_the_plan():
    c = sh.CASES
    assert c["comparison"] == "binary"
    for case in ("B-sim-6", "B-sim-60", "B-sim-600", "B-sim-ud", "B-pulls-tri", "B-pulls-base", "B-real-pionier"):
        assert case in c["data"]
    assert set(c["tasks"]) == {"T1", "T2", "T3", "pulls"}
    assert c["tasks"]["T2"]["configs"]["tight"]["pmoired"]["ftol"] == 1e-10
    assert c["tasks"]["T2"]["configs"]["tight"]["virgil"]["gtol"] == 1e-8
    assert sh.T2_TOL["tight"] == {"gtol": 1e-8, "ftol": 1e-10}
    assert sh.T2_TOL["matched"] == {"gtol": 1e-4, "ftol": 1e-5}


@pytest.mark.parametrize("n", [8, 20, 40])
def test_pmoired_step_gives_exactly_n_points(n):
    step = sh.pmoired_step(n)
    assert int((2 * sh.BOX) / step + 1) == n
    assert abs(step - 2 * sh.BOX / (n - 1)) < 1e-3 * step


def test_native_pmoired_grid_is_21_points():
    assert int(2 * sh.BOX / sh.T3_NATIVE["pmoired_step"] + 1) == 21


def test_reference_and_start():
    assert sh.start_values("B-sim-6") == pytest.approx({"dra": 6.3, "ddec": -4.3, "flux": 0.13})
    assert "ud" in sh.start_values("B-sim-ud") and sh.start_values("B-sim-ud")["ud"] == pytest.approx(1.3)
    real = sh.reference("B-real-pionier")
    assert real["dra"] == 0.094 and real["ddec"] == 3.588 and real["flux"] == pytest.approx(0.599)


def test_manifest_lists_every_planned_file():
    man = json.loads(md.MANIFEST.read_text())
    assert set(man["files"]) == set(md.planned_files("full"))
    assert set(md.planned_files("tiny")) <= set(man["files"])
    assert len(man["real"]) == 2


def test_make_data_is_deterministic(tmp_path):
    a, b = tmp_path / "a.fits", tmp_path / "b.fits"
    md.write_sim(a, 6, 0.0, md.SEEDS["B-sim-6"])
    md.write_sim(b, 6, 0.0, md.SEEDS["B-sim-6"])
    assert md.sha256(a) == md.sha256(b)
    man = json.loads(md.MANIFEST.read_text())
    assert md.sha256(a) == man["files"]["sim6.fits"]


def test_neutral_model_reproduces_noise_free_file(tmp_path):
    """The neutral scorer's visibilities equal crosscheck's noise-free file at the truth (convention check)."""
    p = tmp_path / "c.fits"
    md.write_sim(p, 6, 0.0, 0, clean=True)
    nd = sh.read_neutral([p])
    c = sh.neutral_chi2(nd, sh.reference("B-sim-6"))
    assert c["chi2"] < 1e-6
    assert c["n_v2"] == 180 and c["n_cp"] == 120


def test_pull_summary_shapes():
    rng = np.random.default_rng(0)
    truth = {"dra": 0.0, "ddec": 0.0, "flux": 0.0}
    b = rng.normal(size=(300, 3))
    s = np.ones((300, 3))
    out = sh.pull_summary(b, s, truth, ["dra", "ddec", "flux"])
    assert out["n"] == 300 and np.allclose(out["pull_sd"], 1, atol=0.2)


@pytest.mark.skipif(not md.DATA.exists() or md.verify({"sim6.fits"}), reason="run make_data.py first")
@pytest.mark.parametrize("code", ["virgil", "pmoired"])
def test_smoke_t2_row_is_valid(tmp_path, code):
    out = subprocess.run(
        [sys.executable, str(HERE / f"run_{code}.py"), "--task", "T2", "--case", "B-sim-6", "--size", "tiny",
         "--out", str(tmp_path)], capture_output=True, text=True, timeout=110)
    assert out.returncode == 0, out.stderr[-1500:]
    rows = common.read_rows(tmp_path)
    assert len(rows) == 1 and rows[0]["status"] == "ok" and common.validate_row(rows[0]) == []
    best = rows[0]["accuracy"]["best"]
    assert abs(best["dra"] - 6.0) < 0.3 and abs(best["flux"] - 0.1) < 0.02
