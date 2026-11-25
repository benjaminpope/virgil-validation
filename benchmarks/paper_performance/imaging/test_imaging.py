"""Tests of the imaging comparison's helpers (run: ``cd benchmarks/paper_performance && pyquick $BENCH_PYTHON -m pytest imaging/test_imaging.py``).

Data-dependent tests skip when ``make_data.py`` has not been run; the cross-code gate itself is ``compare.py gate-report``."""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "src"))

import compare  # noqa: E402
import make_data  # noqa: E402
import neutral  # noqa: E402

needs_data = pytest.mark.skipif(not make_data.fits_path("sim-well").exists(), reason="run imaging/make_data.py first")


def test_cases_yml_matches_code():
    cfg = compare.cases_yml()
    assert set(cfg["cases"]) == set(make_data.CASES)
    for k, c in cfg["cases"].items():
        assert c["dataset"] == make_data.CASES[k][0] and c["npix"] == make_data.CASES[k][1]
        assert c["pixel_mas"] == pytest.approx(make_data.pixel_mas(c["npix"]))
    b = cfg["ladder"]["budgets"]
    assert b == sorted(b) and b[:2] == cfg["ladder"]["budgets_tiny"]


def test_start_and_truth_images():
    for n in (24, 48):
        s = make_data.start_image(n, make_data.pixel_mas(n))
        assert s.sum() == pytest.approx(1.0) and s.min() > 0
        east, north = make_data._pixel_centres(n, make_data.pixel_mas(n))
        assert east[0, 0] > 0 > east[0, -1] and north[0, 0] > 0 > north[-1, 0]  # column 0 East, row 0 North
        assert np.allclose(s, s[::-1, ::-1])
    t = make_data.truth_image("gauss", 48, 0.25)
    assert t.sum() == pytest.approx(1.0)
    i, j = np.unravel_index(np.argmax(t), t.shape)
    assert abs((23.5 - i) * 0.25) < 0.25 and abs((23.5 - j) * 0.25) < 0.25  # brightest pixel at the field centre
    # the companion sits at (3, -2) mas East/North: column 0 is East so a smaller j, row 0 is North so a larger i
    ci, cj = int(round(23.5 - (-2.0) / 0.25)), int(round(23.5 - 3.0 / 0.25))
    assert t[ci, cj] > 0.1 * t.max()


def _ladder_row(js, ts, **extra):
    return {"status": "ok", "t_import": 1.0, "t_load": 0.5, "t_cold": extra.get("cold", ts[0] + 2.0), "t_warm": [ts[0]],
            "accuracy": {"ladder": [{"budget": 10 * (i + 1), "t_warm": [t], "score": {"J": j, "rms_rel_beam": 0.1, "ncc_registered": 0.9}}
                                    for i, (j, t) in enumerate(zip(js, ts))]}}


def test_time_to_precision_rungs_interpolation_and_headline():
    routes = {"virgil": [_ladder_row([100.0, 10.0], [1.0, 3.0])],
              "ehtim": [_ladder_row([500.0, 10.4], [1.0, 2.0])]}      # eht-imaging never gets within 1 % of J_ref = 10
    out = compare.time_to_precision(routes)
    assert out["J_ref"] == 10.0
    v, e = out["routes"]["virgil"], out["routes"]["ehtim"]
    assert v["reached"]["0.001"]["time"] == pytest.approx(3.0, abs=1e-2)
    assert e["reached"]["0.01"] is None and e["reached"]["0.1"]["budget"] == 20     # 10.4 is within 10 % but not 1 %
    assert e["best_gap"] == pytest.approx(0.04)
    assert out["headline_gap"] == 0.1
    # target J = 11: eht-imaging interpolated between (J 500, t 1) and (J 10.4, t 2)
    assert e["reached"]["0.1"]["time"] == pytest.approx(1.0 + (500 - 11) / (500 - 10.4) * 1.0)
    # cold time = import + load + (cold - warm of the first budget) + warm time
    assert v["cold_overhead"] == pytest.approx(1.0 + 0.5 + 2.0)
    assert v["reached"]["0.1"]["cold_time"] == pytest.approx(v["cold_overhead"] + v["reached"]["0.1"]["time"])


def test_time_to_precision_no_common_rung_has_no_headline():
    routes = {"virgil": [_ladder_row([100.0, 10.0], [1.0, 3.0])], "ehtim": [_ladder_row([500.0, 400.0], [1.0, 2.0])]}
    assert compare.time_to_precision(routes)["headline_gap"] is None


def test_task_args_follow_cases_yml():
    cfg = compare.cases_yml()
    a = compare._task_args("virgil", "I-sim-24", "ladder", "tiny", cfg)
    assert a["budgets"] == [25, 50] and a["method"] == "lbfgsb" and a["memory"] == 50 and a["gtol"] == 0.0 and a["reps"] == 1
    a = compare._task_args("ehtim", "I-sim-24", "ladder", "full", cfg)
    assert a["budgets"][-1] == 3200 and a["stop"] == 1e-14 and a["reps"] == 5
    a = compare._task_args("ehtim", "I-sim-24", "stop_default", "full", cfg)
    assert (a["maxit"], a["stop"]) == (200, 1e-6)
    a = compare._task_args("virgil", "I-sim-24", "stop_tight", "full", cfg)
    assert a["gtol"] == 1e-8 and a["max_steps"] == 5000


@needs_data
def test_manifest_and_data_counts():
    assert make_data.check_manifest() == []
    d = make_data.load("sim-well")
    assert d["amp"].size == 13 * 3 * 6 and d["cphase_deg"].size == 13 * 6
    assert np.allclose(d["u3"], -(d["u1"] + d["u2"]))
    rows = make_data.ehtim_rows(d)
    assert len(rows) == 13 * 3 * 6 and all(len(r) == 8 for r in rows)
    # eht-imaging's Fourier sign is opposite: rows carry -u / lambda
    k = 7
    assert rows[k][3] == pytest.approx(-d["sim_u"].ravel()[k] / d["sim_wl"].ravel()[k])


@needs_data
def test_large_case_keeps_the_minimal_triangle_set():
    d = make_data.load("sim-large")
    assert d["amp"].size == 25 * 6 * 30 and d["cphase_deg"].size == 25 * 3 * 30  # 3 of the 4 triangles per snapshot


@needs_data
def test_score_image_matches_neutral_objective():
    d = make_data.load("sim-well")
    pixel = make_data.pixel_mas(24)
    img = make_data.truth_image("gauss", 24, pixel)
    s = compare.score_image("I-sim-24", 5.0 * img)  # any flux: normalised
    ref = neutral.image_objective(img, make_data.neutral_data(d, 24, pixel), compare.BETA, sign=-1)
    assert s["J"] == pytest.approx(ref, rel=1e-12) and s["flux"] == pytest.approx(5.0)
    assert s["ncc_registered"] == pytest.approx(1.0) and s["ncc_shift_rc"] == [0, 0]


@needs_data
def test_virgil_loss_equals_neutral_with_virgil_edges():
    pytest.importorskip("virgil")
    import run_virgil

    pb = run_virgil.Problem("I-sim-24")
    pixel = make_data.pixel_mas(24)
    for img in (pb.start, make_data.truth_image("gauss", 24, pixel)):
        d = make_data.load("sim-well")
        ref = neutral.image_objective(img, make_data.neutral_data(d, 24, pixel), run_virgil.BETA, sign=-1, edges="virgil")
        assert pb.loss_at(img)["loss"] == pytest.approx(ref, rel=1e-9)
