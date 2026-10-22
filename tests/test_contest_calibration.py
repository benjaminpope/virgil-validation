"""The 2010 calibration report and per-channel tasks (scripts/contest_images.py), on synthetic data of known truth.

Data are simulated from a known 16 x 16 image with true scatter 100x (V²) and
10x (closure phases) the quoted errors, as the 2010 data's 1e-4 errors are
too small. At the truth the calibration must recover those scales; the V²
quadrature floor must recover a known floor; and raw χ²/N must stay large
while the rescaled one is ~1 (that is the tautology the script flags).
"""

import importlib.util
import pathlib

import jax
import numpy as np
import pytest

from crosscheck import simulate, sky

cc = importlib.util.module_from_spec(
    spec := importlib.util.spec_from_file_location(
        "contest_images", pathlib.Path(__file__).parents[1] / "scripts" / "contest_images.py"
    )
)
spec.loader.exec_module(cc)

import virgil.models as vm  # noqa: E402
from virgil.fields import GaussianField  # noqa: E402
from virgil.imaging import error_scale  # noqa: E402
from virgil.oidata import OIData  # noqa: E402

UTS3 = np.array([[-9.925, -20.335], [14.887, 30.502], [103.306, 43.999]])
N, PIXEL, TRUE_VIS, TRUE_PHI = 16, 1.5, 100.0, 10.0
WAVELENGTHS = np.linspace(1.5e-6, 2.4e-6, 6)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    path = tmp_path_factory.mktemp("cal") / "d.fits"
    simulate.observe(path, sky.vis_point, UTS3, hour_angles_h=np.linspace(-3, 3, 7),
                     wavelengths=WAVELENGTHS, dec_deg=-50.0, sigma_v2=0.01, sigma_cp_deg=0.5)
    x = (np.arange(N) - (N - 1) / 2) * PIXEL
    blob = np.exp(-0.5 * (x[None, :] ** 2 + x[:, None] ** 2) / 3.0**2) + 0.3 * np.exp(-0.5 * ((x[None, :] - 4) ** 2 + x[:, None] ** 2) / 1.5**2)
    truth = vm.System(env=vm.Image(GaussianField(np.zeros((N, N)), 1.0, 3.0, mean=blob), PIXEL, flux=1.0))
    template = OIData(str(path))
    noisy = template.with_model(truth, key=jax.random.PRNGKey(0), noise_scale=1.0)
    # Quoted errors 100x (V²) and 10x (phases) too small, as the 2010 ones are.
    quoted = noisy.with_error_scale({"vis": 1 / TRUE_VIS, "phi": 1 / TRUE_PHI})
    return str(path), truth, quoted


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_quadrature_floor_recovers_a_known_floor():
    rng = np.random.default_rng(0)
    sigma, floor = np.full(20000, 1e-4), 3e-3
    r = rng.normal(0.0, np.sqrt(sigma**2 + floor**2), sigma.size) / sigma  # whitened on the quoted sigma
    assert cc.quadrature_floor(r, sigma) == pytest.approx(floor, rel=0.05)
    assert cc.quadrature_floor(rng.normal(0.0, 0.5, 100), sigma[:100]) == 0.0  # errors already too large


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_calibration_report_at_the_truth_flags_the_raw_chi2(synthetic):
    _, truth, data = synthetic
    scales = error_scale(truth, data, by_observable=True)
    cal = cc.calibration_report(truth, data, scales)
    raw, rescaled = cal["raw"][0], cal["rescaled"][0]
    # The raw χ²/N shows the miscalibration (true scales 100 and 10); the V² scale is recovered. The phase
    # scale is only bounded below: a 16 x 16 image has more latents than the 42 closure phases, so MacKay's
    # N - γ is small there and s is inflated (it absorbs what the model can fit), which the report must not hide.
    assert raw["chi2_v2"] == pytest.approx(TRUE_VIS**2, rel=0.3)
    assert raw["chi2_cp"] == pytest.approx(TRUE_PHI**2, rel=0.5)
    assert cal["scales"]["vis"] == pytest.approx(TRUE_VIS, rel=0.25)
    assert cal["scales"]["phi"] >= 0.8 * TRUE_PHI
    assert rescaled["chi2_v2"] == pytest.approx(1.0, rel=0.3)  # ~1 by construction: tautological
    assert cal["v2_floor"][0] > 0.5 * TRUE_VIS * float(np.median(np.asarray(data.d_vis)))


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_the_channel_tasks_hold_one_channel_each():
    tasks = [t for t in cc.TASKS if t["label"].startswith("2010_ch")]
    assert [t["label"] for t in tasks] == [f"2010_ch{k:02d}" for k in range(1, 21)]
    centres = 1.521375 + 0.0475 * np.arange(20)  # OI_WAVELENGTH of Mystery-Low_HK.oifits, µm
    for t, w in zip(tasks, centres):
        assert t["wavel"][0] < w < t["wavel"][1]
    assert all(a["wavel"][1] < b["wavel"][0] for a, b in zip(tasks, tasks[1:]))
