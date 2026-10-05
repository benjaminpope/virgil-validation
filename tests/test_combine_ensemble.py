"""The contest campaign's ensemble combiner (scripts/combine_ensemble.py).

Synthetic member files with known images, χ² and evidences: the χ² cutoff
(members within 1.5x of the best, or of 1 if the best over-fits, are kept,
the boundary included), the mean
and σ of the kept unit-sum images, the star minus no-star evidence, and NaN
when one cohort is missing.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "combine_ensemble", pathlib.Path(__file__).parents[1] / "scripts" / "combine_ensemble.py"
)
combine_ensemble = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(combine_ensemble)


def _member(path, image, chi2, log_z, star, fov=10.0):
    np.savez(path, ref_image=image, ref_fov=fov, star=star, best_log_z=log_z,
             best_chi2_red=chi2, error_scale=1.0, flip_dchi2=100.0)


def _images():
    a = np.zeros((5, 5))
    a[2, 2] = 2.0
    b = np.zeros((5, 5))
    b[2, 3] = 4.0
    c = np.zeros((5, 5))
    c[0, 0] = 1.0
    return a, b, c


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_cutoff_mean_sigma_and_star_evidence(tmp_path):
    a, b, c = _images()
    _member(tmp_path / "x_m0.npz", a, 1.0, -10.0, True)
    _member(tmp_path / "x_m1.npz", b, 1.5, -12.0, False)  # exactly 1.5x: kept
    _member(tmp_path / "x_m2.npz", c, 1.6, -5.0, False)  # above the cutoff: dropped
    row = combine_ensemble.combine("x", combine_ensemble.load(tmp_path)["x"], tmp_path)
    out = np.load(tmp_path / "x_ensemble.npz")
    assert sorted(out["kept"]) == [0, 1]
    stack = np.array([a / a.sum(), b / b.sum()])
    np.testing.assert_allclose(out["mean"], stack.mean(0))
    np.testing.assert_allclose(out["sigma"], stack.std(0))
    # The best star member against the best no-star member, dropped ones included.
    assert float(out["star_minus_nostar_log_z"]) == pytest.approx(-10.0 - (-5.0))
    assert row.startswith("| x | 2/3 |")


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_an_overfitting_best_member_does_not_set_the_cutoff(tmp_path):
    a, b, c = _images()
    _member(tmp_path / "z_m0.npz", a, 0.02, -10.0, False)  # over-fits
    _member(tmp_path / "z_m1.npz", b, 1.5, -12.0, False)  # 1.5 x max(0.02, 1): kept
    _member(tmp_path / "z_m2.npz", c, 1.6, -5.0, False)  # dropped
    combine_ensemble.combine("z", combine_ensemble.load(tmp_path)["z"], tmp_path)
    assert sorted(np.load(tmp_path / "z_ensemble.npz")["kept"]) == [0, 1]


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_a_rerun_replaces_the_same_member(tmp_path):
    a, b, _ = _images()
    first, rerun = tmp_path / "first", tmp_path / "rerun"
    first.mkdir()
    rerun.mkdir()
    _member(first / "w_m0.npz", a, 1.0, -10.0, True)
    _member(first / "w_m1.npz", a, 9.0, -50.0, False)
    _member(rerun / "w_m1.npz", b, 1.2, -11.0, False)
    members = combine_ensemble.load(first, rerun)["w"]
    assert sorted(m["member"] for m in members) == [0, 1]
    assert [m["chi2"] for m in members if m["member"] == 1] == [1.2]


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_missing_cohort_and_resampled_grid(tmp_path):
    a, b, _ = _images()
    _member(tmp_path / "y_m0.npz", a, 1.0, -10.0, True)
    _member(tmp_path / "y_m2.npz", b, 1.0, -11.0, True)
    _member(tmp_path / "y_m4.npz", b, 1.0, -11.0, True, fov=20.0)  # another grid: resampled
    combine_ensemble.combine("y", combine_ensemble.load(tmp_path)["y"], tmp_path)
    out = np.load(tmp_path / "y_ensemble.npz")
    assert np.isnan(float(out["star_minus_nostar_log_z"]))
    assert sorted(out["kept"]) == [0, 2, 4]
    assert float(out["fov"]) == 20.0
    assert out["mean"].sum() == pytest.approx(1.0)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"])
def test_resampling_keeps_a_point_at_its_sky_position():
    n = 129
    image = np.zeros((n, n))
    centre = (n - 1) // 2
    image[centre - 16, centre + 16] = 1.0  # 16 pixels up and right on a 10-wide field
    out = combine_ensemble.resample(image, 10.0, 20.0)  # pixels twice as large
    row, col = np.unravel_index(np.argmax(out), out.shape)
    assert (row - centre, col - centre) == (-8, 8)
    assert out.sum() == pytest.approx(1.0)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_members_with_different_pixel_counts_are_stacked(tmp_path):
    # A first-campaign member (9 pixels over 10 mas) and a later one (17 pixels
    # over 20 mas) of the same scene: both are resampled onto 17 pixels over 20 mas.
    small = np.zeros((9, 9))
    small[4, 4] = 1.0
    large = np.zeros((17, 17))
    large[8, 8] = 1.0
    _member(tmp_path / "v_m0.npz", small, 1.0, -10.0, True, fov=10.0)
    _member(tmp_path / "v_m1.npz", large, 1.0, -11.0, False, fov=20.0)
    combine_ensemble.combine("v", combine_ensemble.load(tmp_path)["v"], tmp_path)
    out = np.load(tmp_path / "v_ensemble.npz")
    assert out["mean"].shape == (17, 17) and float(out["fov"]) == 20.0
    assert np.unravel_index(np.argmax(out["mean"]), (17, 17)) == (8, 8)
    assert sorted(out["kept"]) == [0, 1]

