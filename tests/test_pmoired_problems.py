"""Reproducers for the PMOIRED problems in docs/pmoired_notes.md.

Each is a strict xfail stating the behaviour we expect: if PMOIRED changes
so that it passes, the test fails and the note must be updated.
"""

import contextlib
import io

import numpy as np
import pytest

from crosscheck import simulate, sky

pmoired = pytest.importorskip("pmoired")
from external_bridge.pmoired_models import model_minus_data  # noqa: E402

pytestmark = pytest.mark.external

UTS = np.array(
    [[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]]
)
ANNULUS = {"diamin": 2.0, "diamout": 4.0}


def _annulus_file(path, n_ha, n_wl):
    simulate.observe(
        path, lambda u, v, w: sky.vis_annulus(u, v, w, 2.0, 4.0), UTS,
        hour_angles_h=np.linspace(-2, 2, n_ha) if n_ha > 1 else [0.0],
        wavelengths=np.linspace(1.5e-6, 2.4e-6, n_wl), dec_deg=-50.0,
        sigma_v2=0.01, sigma_cp_deg=0.5,
    )
    return path


@pytest.mark.xfail(strict=True, reason="P1: auto wl kernel gives NaN models")
def test_p1_default_setup_gives_finite_models(tmp_path):
    import pmoired.oimodels as om

    path = _annulus_file(tmp_path / "p1.fits", 3, 4)
    with contextlib.redirect_stdout(io.StringIO()):
        oi = pmoired.OI(str(path), verbose=False)
        oi.setupFit({"obs": ["V2", "T3PHI"]})  # auto=True, the default
        model = om.VmodelOI(oi.data[0], {"ud": 1.0})
    v2 = np.concatenate([b["V2"].ravel() for b in model["OI_VIS2"].values()])
    assert np.all(np.isfinite(v2))


@pytest.mark.xfail(strict=True, reason="P2: default ring sampling is not Nr = 100")
def test_p2_default_nr_is_100(tmp_path):
    path = _annulus_file(tmp_path / "p2.fits", 3, 4)
    default, _, _ = model_minus_data(path, ANNULUS)
    nr100, _, _ = model_minus_data(path, ANNULUS, {"Nr": 100})
    assert np.isclose(default, nr100, rtol=0.1)


def test_p3_rings_exact_up_to_30_samples(tmp_path):
    """The control: 1 epoch x 30 channels is exact."""
    path = _annulus_file(tmp_path / "p3a.fits", 1, 30)
    dv2, _, _ = model_minus_data(path, ANNULUS, {"Nr": 3000})
    assert dv2 < 3e-8


@pytest.mark.xfail(
    strict=True, reason="P3: rings stop at ~1e-4 beyond 30 samples per baseline"
)
@pytest.mark.parametrize("n_ha,n_wl", [(1, 35), (7, 6)])
def test_p3_rings_exact_beyond_30_samples(tmp_path, n_ha, n_wl):
    path = _annulus_file(tmp_path / "p3b.fits", n_ha, n_wl)
    dv2, _, _ = model_minus_data(path, ANNULUS, {"Nr": 3000})
    assert dv2 < 3e-8
