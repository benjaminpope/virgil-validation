"""The 0.4 argument-order change of likelihood_grid (ledger F21).

virgil 0.4 moved the model before the data: ``likelihood_grid(data, model,
samples_dict)`` became ``likelihood_grid(model, data, grid)``. Every
ESO-binaries verdict was marked provisional because the change might have
swapped arguments or axes silently. The audit (``review/ledger.md``) found
that it cannot: virgil recognizes the data by type and rebinds a 0.3-order
call with a FutureWarning, and any other permutation of the three arguments
fails. What is pinned here:

* the 0.3-order call, the keyword call and (where virgil still allows it)
  the 0.4 positional call give minus half our own chi-squared on a grid whose
  axes have different lengths, so an axis swap would show as a wrong shape or
  wrong values (mathematics);
* every one of the six positional orders of (model, data, grid) either
  raises or returns exactly that grid, never a different one (a control);
* the finding, a strict xfail: a misordered positional call fails from
  inside the grid code with an ``AttributeError`` ("'OIData' object has no
  attribute 'keys'"), not an error naming the call. It flips when virgil
  makes ``data`` and ``grid`` keyword-only (virgil branch
  ``likelihood-grid-kwonly``).

The grid axis order itself (axis k follows the k-th key of the dict) is
checked in tests/test_grids_limits.py and is unchanged since 0.3.
"""

import itertools
import warnings

import numpy as np
import pytest

from crosscheck import chi2 as ours, simulate, sky
from evidence.plugin import record

vm = pytest.importorskip("virgil.models")
from virgil.grid_fit import likelihood_grid  # noqa: E402
from virgil.oidata import OIData  # noqa: E402

pytestmark = pytest.mark.x64

UTS3 = np.array([[-9.925, -20.335], [14.887, 30.502], [103.306, 43.999]])
HA = np.linspace(-3, 3, 7)
WL = np.linspace(1.5e-6, 2.4e-6, 6)
DRA, DDEC, FLUX = 6.0, -4.0, 0.03
# Different lengths and different ranges on the two position axes, so any
# exchange of axes (or of the grid with the data) changes the shape.
XS = np.linspace(-10.0, 10.0, 5)
YS = np.linspace(-8.0, 4.0, 4)
GRID = {"dra": XS, "ddec": YS, "flux": [FLUX]}
MODEL = vm.BinaryModelCartesian
ARGS = {"model": MODEL, "data": None, "grid": GRID}


def binary_vis(dra, ddec, flux):
    def vis(u, v, w):
        return (sky.vis_point(u, v, w) + flux * sky.vis_point(u, v, w, dra, ddec)) / (1 + flux)

    return vis


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("order") / "order.fits"
    simulate.observe(
        path, binary_vis(DRA, DDEC, FLUX), UTS3, hour_angles_h=HA,
        wavelengths=WL, dec_deg=-50.0, sigma_v2=0.01, sigma_cp_deg=0.5,
        rng=np.random.default_rng(21),
    )
    return OIData(str(path)), ours.load(path)


def reference(d):
    """Minus half our chi-squared on the (dra, ddec) grid, shape (5, 4)."""
    return np.array([[-0.5 * ours.chi2(d, binary_vis(x, y, FLUX)) for y in YS] for x in XS])


def keyword_call(data):
    return np.asarray(likelihood_grid(MODEL, data=data, grid=GRID))


def agrees(grid, d):
    """The grid equals the reference up to its normalisation constant."""
    got = grid[..., 0]
    want = reference(d)
    return got.shape == want.shape and np.max(np.abs((got - got.max()) - (want - want.max()))) < 1e-9


@pytest.mark.validates("virgil.grid_fit.likelihood_grid", roots=["mathematics"])
def test_f21_keyword_and_old_order_calls_are_minus_half_chi2(dataset):
    data, d = dataset
    new = keyword_call(data)
    assert new.shape == (len(XS), len(YS), 1)
    assert agrees(new, d)
    with pytest.warns(FutureWarning, match="before the data"):
        old = np.asarray(likelihood_grid(data, MODEL, GRID))
    assert np.array_equal(old, new)
    record("old_order_equals_keyword_call", float(np.max(np.abs(old - new))))


@pytest.mark.validates("virgil.grid_fit.likelihood_grid", roots=["mathematics"], kind="control")
def test_f21_no_positional_order_returns_a_different_grid(dataset):
    """All six positional orders of (model, data, grid): each raises, or
    returns the grid of the keyword call. None returns another grid."""
    data, d = dataset
    new = keyword_call(data)
    values = {"model": MODEL, "data": data, "grid": GRID}
    outcomes = {}
    for order in itertools.permutations(values):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            try:
                got = np.asarray(likelihood_grid(*(values[k] for k in order)))
            except Exception as err:  # noqa: BLE001
                outcomes[order] = type(err).__name__
                continue
        assert np.array_equal(got, new), order
        outcomes[order] = "same grid"
    record("positional_orders_returning_the_same_grid", sum(v == "same grid" for v in outcomes.values()))
    record("positional_orders_raising", sum(v != "same grid" for v in outcomes.values()))
    # the 0.3 order is always one of them
    assert outcomes[("data", "model", "grid")] == "same grid"


MISORDERED = [
    ("model", "grid", "data"),
    ("grid", "model", "data"),
    ("grid", "data", "model"),
]


@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="F21: a misordered positional likelihood_grid call fails with an AttributeError from inside the grid code (virgil branch likelihood-grid-kwonly makes data and grid keyword-only)",
)
@pytest.mark.validates("virgil.grid_fit.likelihood_grid", roots=["mathematics"], kind="finding")
@pytest.mark.parametrize("order", MISORDERED, ids="-".join)
def test_f21_misordered_positional_call_is_a_clear_error(dataset, order):
    data, _ = dataset
    values = {"model": MODEL, "data": data, "grid": GRID}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with pytest.raises(TypeError, match="likelihood_grid"):
            likelihood_grid(*(values[k] for k in order))
