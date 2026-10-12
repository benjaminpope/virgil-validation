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
def test_regulariser_weights_span_the_discrepancy_scale(name):
    """Non-MaxEnt weights span 10^±2 about N / P0, so the penalty at the middle
    weight equals χ² = N on the start and the discrepancy weight is bracketed."""
    start = _image()
    data = types.SimpleNamespace(n_independent=500)
    weights = ci.regulariser_weights(name, start, data, path=None)
    p0 = float(ci.regulariser(name, 1.0, None).value(start))
    assert p0 > 0
    assert len(weights) == len(ci.REG_SPAN)
    np.testing.assert_allclose(weights[len(weights) // 2] * p0, 500, rtol=1e-6)
    np.testing.assert_allclose(weights[0] / weights[-1], 1e4, rtol=1e-6)
    assert len(ci.regulariser_weights(name, start, data, path=None, smoke=True)) == 3


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_mem_keeps_its_weights():
    """MaxEnt keeps the round-1 weights, so round-1 runs are unchanged."""
    w = ci.regulariser_weights("mem", _image(), types.SimpleNamespace(n_independent=500), path=None)
    np.testing.assert_array_equal(w, ci.WEIGHTS)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_informed_provenance():
    assert ci.informed_line(True).startswith("provenance=informed")
    assert ci.informed_line(False) == "provenance=blind"
