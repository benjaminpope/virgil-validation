"""Round-2 contest features (contests/fit_plan.yml) in scripts/contest_images.py:
regularisers beside MaxEnt with weights spanning the discrepancy principle's
range, and the "informed" provenance of round-2 outputs. Tiny NumPy/virgil
evaluations only; the fits themselves run on OzSTAR."""

import importlib.util
import pathlib
import types

import numpy as np
import pytest

pytest.importorskip("virgil")

ROOT = pathlib.Path(__file__).parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("contest_images", ROOT / "scripts" / "contest_images.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ci = _load()


def _image(n=16, pixel=1.0):
    import jax.numpy as jnp
    import virgil.models as vm

    y, x = np.mgrid[:n, :n] - n / 2
    return vm.Image(jnp.asarray(-(x**2 + y**2) / 8.0), pixel, flux=1.0)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
@pytest.mark.parametrize("name", ["tv", "tsv", "starlet", "logsum"])
def test_regulariser_weights_span_the_documented_range(name):
    """Non-MaxEnt weights are virgil.ensemble's per-data-point range x N, 12 log-spaced, strong to weak."""
    from virgil.ensemble import _default_weight_ranges

    low, high = _default_weight_ranges()["starlet" if name == "logsum" else name]
    data = types.SimpleNamespace(n_independent=500)
    weights = ci.regulariser_weights(name, data)
    assert len(weights) == 12
    np.testing.assert_allclose(weights[0], high * 500, rtol=1e-9)
    np.testing.assert_allclose(weights[-1], low * 500, rtol=1e-9)
    assert np.all(np.diff(np.log(weights)) < 0)
    np.testing.assert_allclose(np.diff(np.log(weights)), np.diff(np.log(weights))[0], rtol=1e-9)
    assert len(ci.regulariser_weights(name, data, smoke=True)) == 3


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_mem_keeps_its_weights():
    """MaxEnt keeps the round-1 weights, and the Adam fallback is off for it, so round-1 runs are unchanged."""
    w = ci.regulariser_weights("mem", types.SimpleNamespace(n_independent=500))
    np.testing.assert_array_equal(w, ci.WEIGHTS)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_informed_provenance():
    assert ci.informed_line(True).startswith("provenance=informed")
    assert ci.informed_line(False) == "provenance=blind"


class _Reg:
    """A stand-in penalty: value = w * P(model), P = model["p"]."""

    def __init__(self, w):
        self.w = float(w)

    def value(self, model):
        return self.w * model["p"]


def _result(chi2, p, stop="converged", converged=True):
    return types.SimpleNamespace(model={"p": p}, values=None, info={
        "converged": converged, "stop": stop, "chi2": [chi2], "ndata": [10]})


def _curve(rows):
    """rows: (weight, chi2, penalty, stop)."""
    import dataclasses

    @dataclasses.dataclass
    class LC:
        weights: object
        results: list
        chi2: object
        chi2_red: object
        penalty: object

    return LC(weights=np.array([r[0] for r in rows]),
              results=[_result(r[1], r[2], r[3], r[3] == "converged") for r in rows],
              chi2=np.array([r[1] for r in rows]), chi2_red=np.array([[r[1] / 10] for r in rows]),
              penalty=np.array([r[2] for r in rows]))


@pytest.fixture
def fake_fit(monkeypatch):
    refits = {}  # weight -> (chi2, p) of the Adam refit

    monkeypatch.setattr(ci, "regulariser", lambda reg, w, path, q: _Reg(w))
    monkeypatch.setattr(ci, "_reference", lambda model: model)
    monkeypatch.setattr(ci, "n_independent", lambda data: 10)

    def fit(start, priors, data, regs, init=None, method=None, max_steps=None):
        c, p = refits[regs[0].w]
        return _result(c, p)

    monkeypatch.setattr(ci, "fit", fit)
    return refits


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_adam_fallback_keeps_the_lower_objective(fake_fit):
    """A refit replaces a stalled L-BFGS point only if chi2 + w*P is lower: lower chi2 alone (less
    regularised) is not enough; converged points are never refitted."""
    # objective of the L-BFGS fits: 100 + 2*5 = 110, 50 + 1*5 = 55, 20 + 0.5*1 = 20.5
    curve = _curve([(2.0, 100.0, 5.0, "stalled"), (1.0, 50.0, 5.0, "limit"), (0.5, 20.0, 1.0, "converged")])
    fake_fit[2.0] = (90.0, 10.0)  # lower chi2 but objective 90 + 20 = 110 is not lower: stays L-BFGS
    fake_fit[1.0] = (45.0, 4.0)  # objective 45 + 4 = 49 < 55: substituted
    patched, methods, pairs = ci.adam_fallback(curve, None, None, object(), "tv", None, None, [])
    assert methods == ["lbfgs", "adam", "lbfgs"]
    np.testing.assert_allclose(patched.chi2, [100.0, 45.0, 20.0])
    np.testing.assert_allclose(patched.penalty, [5.0, 4.0, 1.0])
    assert pairs == [(10.0, 9.0), (5.0, 4.5), None]  # both chi2/N kept as the diagnostic


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_adam_fallback_is_off_for_maxent(fake_fit):
    curve = _curve([(2.0, 100.0, 5.0, "stalled")])
    patched, methods, pairs = ci.adam_fallback(curve, None, None, object(), "mem", None, None, [])
    assert patched is curve and methods == ["lbfgs"] and pairs == [None]


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
@pytest.mark.parametrize("label,scales,expected", [
    ("2006_nostar_clean", (0.0,), "2006_nostar_cleanfinal"),
    ("2006_star_halo_clean", (0.0,), "2006_star_halo_cleanfinal"),
    ("2006_star_clean", (0.0, 1.0), "2006_star_cleanfinal_cleanms"),
])
def test_clean_label_keeps_star_and_halo_suffixes(label, scales, expected):
    assert ci.clean_label({"label": label}, scales) == expected


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_weight_report_with_a_single_dataset():
    """A one-file task's chi2_red may be 1-D (weights,); the report must not raise."""
    curve = types.SimpleNamespace(chi2_red=np.array([3.0, 1.5, 0.9]), discrepancy=lambda: 0.5, corner=lambda: 1.0)
    rep = ci.weight_report(curve, object(), "tv", None, 1)
    assert rep["binding_dataset"] == 0 and rep["chi2_red_by_dataset"] == [1.5] and rep["n_datasets"] == 1
