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
    weights = ci.regulariser_weights(name, _image(), data, path=None)
    assert len(weights) == 12
    np.testing.assert_allclose(weights[0], high * 500, rtol=1e-9)
    np.testing.assert_allclose(weights[-1], low * 500, rtol=1e-9)
    assert np.all(np.diff(np.log(weights)) < 0)
    np.testing.assert_allclose(np.diff(np.log(weights)), np.diff(np.log(weights))[0], rtol=1e-9)
    assert len(ci.regulariser_weights(name, _image(), data, path=None, smoke=True)) == 3


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_mem_keeps_its_weights():
    """MaxEnt keeps the round-1 weights, so round-1 runs are unchanged."""
    w = ci.regulariser_weights("mem", _image(), types.SimpleNamespace(n_independent=500), path=None)
    np.testing.assert_array_equal(w, ci.WEIGHTS)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_informed_provenance():
    assert ci.informed_line(True).startswith("provenance=informed")
    assert ci.informed_line(False) == "provenance=blind"


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_pixel_offset_matches_virgil_image():
    """A one-pixel Image and a PointSource at pixel_offset have the same
    visibilities, so CLEAN clusters start point sources in the right place."""
    import jax.numpy as jnp
    import virgil.models as vm

    n, pixel, row, col = 21, 1.0, 8, 4
    lb = np.full((n, n), -40.0)
    lb[row, col] = 0.0
    image = vm.Image(jnp.asarray(lb), pixel, flux=1.0)
    dra, ddec = ci.pixel_offset(row, col, n, pixel)
    u, v, w = np.array([10.0, 0.0, 7.0]), np.array([0.0, 10.0, -5.0]), np.full(3, 1e-6)
    np.testing.assert_allclose(np.asarray(image.model(u, v, w)),
                               np.asarray(vm.PointSource(1.0, dra=float(dra), ddec=float(ddec)).model(u, v, w)), atol=1e-5)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_point_candidates_skip_the_main_source():
    """Candidates come from compact clusters outside the main source,
    brightest first; the main source's own components are never offered."""
    n, pixel = 41, 1.0
    beam = types.SimpleNamespace(major_mas=3.0, minor_mas=3.0)
    c = np.zeros((n, n))
    c[18:23, 18:23] = 1.0  # the main source, 25 units
    c[5, 30] = 0.5  # a bright dot
    c[35, 8] = 0.2  # a faint dot
    y, x = np.mgrid[:n, :n]
    kernel = np.exp(-0.5 * ((x - n // 2) ** 2 + (y - n // 2) ** 2) / 1.3**2)
    restored = np.real(np.fft.ifft2(np.fft.fft2(c) * np.fft.fft2(np.fft.ifftshift(kernel))))
    found = ci.clean_point_candidates(c, restored, pixel, beam, 5)
    assert [round(f["flux"], 3) for f in found] == [0.5, 0.2]
    assert found[0]["dra"] == pytest.approx(ci.pixel_offset(5, 30, n, pixel)[0])
    assert found[0]["ddec"] == pytest.approx(ci.pixel_offset(5, 30, n, pixel)[1])


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_suffixes_and_no_systematics():
    assert ci.suffix(3, [0.01], None, True, reg="tv") == "_tv_ps3_gains_errscale"
    assert ci.suffix() == ""
    data = object()
    assert ci.with_systematics(data, None) is data
