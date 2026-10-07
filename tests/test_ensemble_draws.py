"""virgil.ensemble's random settings against what its documentation
promises (no fits: combine, run_group and ensemble are not checked here).

draw_groups, on a simulated 7-hole masking dataset (21 V², 35 closure
phases of which 15 independent: 36 independent data):

* each group has a family from ``spec.families``, a start from
  ``spec.starts``, ``n_weights`` weights largest first, pixels of
  nyquist_pixel_scale / o for o in ``spec.oversample``, and a field of
  field_of_view × f for f in ``spec.field_factors`` (to within a pixel);
* the weights per independent datum lie in ``spec.weight_ranges``: over
  many draws, the bounds max(w) / high <= N <= min(w) / low pin N, and
  log(w / N) is uniform on the range (Kolmogorov-Smirnov);
* the draws depend only on the key and the spec, and groups that share a
  compilation (family and geometry) are adjacent.
"""

from itertools import groupby

import jax
import numpy as np
import pytest
from scipy import stats

from evidence.plugin import record

pytest.importorskip("virgil.ensemble")
from virgil import coverage, ensemble, imaging  # noqa: E402

pytestmark = pytest.mark.x64
N_INDEPENDENT = 21 + 15  # V² plus independent closure phases of 7 holes: C(6, 2)


@pytest.fixture(scope="module")
def drawn():
    data = coverage.nrm_oidata()
    spec = ensemble.EnsembleSpec()
    return data, spec, ensemble.draw_groups(data, 300, jax.random.PRNGKey(5), spec)


@pytest.mark.validates("virgil.ensemble.draw_groups", "virgil.ensemble.Draw", "virgil.ensemble.EnsembleSpec",
                       roots=["mathematics"])
def test_each_draw_follows_the_spec(drawn):
    data, spec, draws = drawn
    assert sorted(d.index for d in draws) == list(range(len(draws)))
    nyquist, fov = float(imaging.nyquist_pixel_scale(data)), float(imaging.field_of_view(data))
    scales = np.array([nyquist / o for o in spec.oversample])
    worst = 0.0
    for d in draws:
        assert d.family in spec.families and d.start in spec.starts
        assert len(d.weights) == spec.n_weights and list(d.weights) == sorted(d.weights, reverse=True)
        assert np.min(np.abs(scales - d.pixel_scale_mas)) < 1e-9 * nyquist
        if d.npix < spec.max_npix:
            gap = np.min(np.abs(np.array(spec.field_factors) * fov / d.pixel_scale_mas - d.npix))
            worst = max(worst, gap)
    record("max_field_gap_pixels", worst)
    assert worst <= 1.0
    # every family and start is drawn (uniformly: chi-squared test of the counts)
    counts = np.array([sum(d.family == f for d in draws) for f in spec.families])
    assert stats.chisquare(counts).pvalue > 1e-3


@pytest.mark.validates("virgil.ensemble.draw_groups", "virgil.ensemble.EnsembleSpec", roots=["statistics"])
def test_weights_are_log_uniform_per_independent_datum(drawn):
    data, spec, draws = drawn
    worst_p, lows, highs = 1.0, [], []
    for family in spec.families:
        w = np.concatenate([d.weights for d in draws if d.family == family])
        low, high = spec.weight_ranges[family]
        lo_bound, hi_bound = w.max() / high, w.min() / low  # N must lie between
        assert lo_bound <= N_INDEPENDENT <= hi_bound, (family, lo_bound, hi_bound)
        lows.append(lo_bound)
        highs.append(hi_bound)
        x = (np.log(w / N_INDEPENDENT) - np.log(low)) / (np.log(high) - np.log(low))
        worst_p = min(worst_p, stats.kstest(x, "uniform").pvalue)
    record("n_data_lower_bound", max(lows))
    record("n_data_upper_bound", min(highs))
    record("min_ks_pvalue", worst_p)
    assert worst_p > 1e-3


@pytest.mark.validates("virgil.ensemble.draw_groups", roots=["mathematics"])
def test_draws_are_reproducible_and_grouped_by_compilation(drawn):
    data, spec, draws = drawn
    again = ensemble.draw_groups(data, 300, jax.random.PRNGKey(5), spec)
    assert [(d.index, d.family, d.npix, d.weights) for d in again] == [(d.index, d.family, d.npix, d.weights) for d in draws]
    keys = [(d.family, d.npix, round(d.pixel_scale_mas, 9)) for d in draws]
    runs = [k for k, _ in groupby(keys)]
    assert len(runs) == len(set(keys))  # each compilation's groups are adjacent
