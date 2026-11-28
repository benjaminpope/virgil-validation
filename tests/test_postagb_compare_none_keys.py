"""``compare`` writes None for keys a run does not have (chi2 on the real data under ``--inject``); the plots must not format them."""
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
try:
    import postagb_imaging as P
except Exception:  # noqa: BLE001
    P = None

pytestmark = [
    pytest.mark.skipif(P is None, reason="needs virgil and virgil-vlti (install the postagb extra)"),
    pytest.mark.validates("evidence", roots=["statistics"], kind="guard"),
]


def test_num_formats_numbers_and_none():
    assert P.num(1.234) == "1.23" and P.num(None) == "n/a" and P.num(2.0, ".1f") == "2.0" and P.num(None, none="-") == "-"


@pytest.mark.parametrize("c2_pub", [None, 1.7])
def test_plot_compare_draws_with_or_without_chi2_on_the_real_data(tmp_path, c2_pub):
    cfg = dict(key="T_gp-polar", name="T", half_cmp_mas=5.0, inner_rim_max_mas=None)
    n = 16
    img = np.random.default_rng(0).random((n, n))
    np.savez(P.path(tmp_path, cfg, "compare_images.npz"), ours=img, ours_conv=img, published_conv=img, pixel_mas=0.5)
    json.dump(dict(ncc_zero_shift=0.9, ncc_best_shift=0.91, chi2_per_point_ours=1.1, chi2_per_point_published_on_our_data=c2_pub, smoke=True),
              open(P.path(tmp_path, cfg, "compare.json"), "w"))
    P.plot_compare(tmp_path, cfg)
    assert P.path(tmp_path, cfg, "compare.png").exists()
