"""A likelihood must be continuous in the model parameters (finding F11).

Found by the imaging-contest stalls (docs/plan_imaging_contests.md) and
fixed in virgil#174. Before the fix, for closure phases from four or more
telescopes, virgil wrapped each residual
into [-π, π), takes its chord 2 sin(Δ/2), then whitens the chords
together. A chord changes sign under Δ → Δ + 2π. That is harmless in a
single square, but cross terms then make χ² jump wherever a residual
crosses ±π. Image fits on high signal-to-noise four-telescope data stall
on those jumps.

The file is written by our simulator (four VLTI UTs, one snapshot, one
channel). The sweep moves a binary's companion in a straight line, so every
model visibility, and every closure phase, changes smoothly. χ² is then
sampled finely. A continuous χ² changes little between neighbouring
samples. The three-telescope control (one closure phase, nothing
correlated) must pass today.
"""

import numpy as np
import pytest

from crosscheck import simulate, sky
from evidence.plugin import record

vm = pytest.importorskip("virgil.models")
from virgil.likelihood import whitened_residuals  # noqa: E402
from virgil.oidata import OIData  # noqa: E402

pytestmark = pytest.mark.x64

UTS = np.array(
    [[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]]
)


def _binary(dra, ddec, flux):
    def vis(u, v, w):
        return (sky.vis_point(u, v, w) + flux * sky.vis_point(u, v, w, dra, ddec)) / (1 + flux)

    return vis


def _largest_step_ratio(path):
    data = OIData(str(path))
    xs = np.linspace(-6.0, 6.0, 2001)
    chi2 = np.array([
        float(np.sum(np.asarray(whitened_residuals(vm.BinaryModelCartesian(x, -2.0, 0.9), data)) ** 2))
        for x in xs
    ])
    steps = np.abs(np.diff(chi2))
    return steps.max() / np.median(steps)


def _file(tmp_path, stations):
    path = tmp_path / f"{len(stations)}t.fits"
    simulate.observe(
        path, _binary(3.0, 2.0, 0.9), stations,
        hour_angles_h=[0.0], wavelengths=np.array([2.2e-6]), dec_deg=-30.0,
        sigma_v2=0.01, sigma_cp_deg=0.5,
    )
    return path


@pytest.mark.validates("virgil.likelihood.whitened_residuals", roots=["mathematics"])
def test_three_telescope_chi2_is_continuous(tmp_path):
    ratio = _largest_step_ratio(_file(tmp_path, UTS[:3]))
    record("max_step_over_median_3t", ratio)
    assert ratio < 200


@pytest.mark.validates("virgil.likelihood.whitened_residuals", roots=["mathematics"])
def test_four_telescope_chi2_is_continuous(tmp_path):
    ratio = _largest_step_ratio(_file(tmp_path, UTS))
    record("max_step_over_median_4t", ratio)
    assert ratio < 200
