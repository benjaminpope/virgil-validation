"""scripts/postagb_imaging.py: the night error model (``--noise night``, ``calibrate --arm night``) and the width estimator of
scripts/postagb_night_widths.py (design/gp_polar_hyperparameters.md §9)."""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
try:  # needs virgil main (MultipoleField), as tests/test_postagb_imaging.py
    import postagb_imaging as P  # noqa: E402
    import postagb_night_widths as W  # noqa: E402
except ImportError:
    P = W = None
pytestmark = [
    pytest.mark.skipif(
        P is None, reason="virgil without MultipoleField (install the postagb extra)"
    ),
    # guards our own bookkeeping: which samples share a night term, and a method-of-moments estimate checked on draws with a
    # known answer; no standard or external code is involved
    pytest.mark.validates(
        "postagb_imaging.night_noise", roots=["statistics"], kind="guard"
    ),
]

PAIRS = [(1, 2), (2, 3), (1, 3)]
MJDS = [
    100.1,
    100.2,
    101.1,
    101.2,
]  # two nights of two frames (UT before the MJD .65 split: the night that began the day before)


def toy_data():
    """Three telescopes, two nights x two frames, three baselines, two channels, one closure phase per frame and channel."""
    rows = [
        (f, m, a, c, w)
        for f, m in enumerate(MJDS)
        for a, c in PAIRS
        for w in (1.6e-6, 1.7e-6)
    ]
    r = np.array(rows)
    n = len(rows)
    d = dict(
        u=np.linspace(10, 60, n),
        v=np.linspace(-20, 30, n),
        vis=np.full(n, 0.8),
        d_vis=np.full(n, 0.02),
        vis_flag=np.zeros(n, bool),
        wavel=r[:, 4],
        mjd=r[:, 1],
        stations=r[:, 2:4].astype(int),
        frame=r[:, 0].astype(int),
    )
    i1 = [6 * f + k for f in range(4) for k in range(2)]
    m = len(i1)
    d.update(
        phi=np.zeros(m),
        d_phi=np.full(m, 0.02),
        phi_flag=np.zeros(m, bool),
        i_cps1=np.array(i1),
        i_cps2=np.array(i1) + 2,
        i_cps3=np.array(i1) + 4,
    )
    return P.OIData(d)


def test_night_labels_group_samples_by_night_and_baseline_or_triangle():
    lab = P.night_labels(toy_data())
    assert len(np.unique(lab["u_night"])) == 2
    assert len(np.unique(lab["v2_cell"])) == 6  # 2 nights x 3 baselines
    assert len(np.unique(lab["t3_cell"])) == 2  # 2 nights x 1 triangle
    for c in np.unique(
        lab["v2_cell"]
    ):  # a cell spans both frames of its night, on one baseline
        assert len(np.unique(lab["v2_frame"][lab["v2_cell"] == c])) == 2
        assert len(np.unique(lab["v2_night"][lab["v2_cell"] == c])) == 1


def test_night_noise_adds_one_spanning_mode_per_cell_and_none_for_zero_width():
    d = toy_data()
    nd = P.night_noise(
        d, P.VLTI_NOISE, dict(night_gain_baseline=0.02, night_t3_offset_deg=1.0)
    )
    assert nd.gains.spanning.shape == (d.u.size, 6)
    assert np.shape(nd.phase_offsets.spanning) == (d.phi.size, 2)
    np.testing.assert_allclose(
        np.sort(np.unique(np.abs(np.asarray(nd.gains.spanning)))), [0.0, 0.02]
    )
    n0 = P.night_noise(
        d, P.VLTI_NOISE, dict(night_gain_baseline=0.0, night_t3_offset_deg=0.0)
    )
    assert (
        n0.gains.spanning is None and n0.phase_offsets.spanning is None
    )  # the per-frame vlti terms alone


def test_night_widths_fall_back_to_the_median_of_the_table():
    w = P.night_widths(dict(key="no_such_target"))
    assert w["night_gain_baseline"] == pytest.approx(
        np.median([r["night_gain_baseline"] for r in P.NIGHT_WIDTHS.values()])
    )


def test_between_cell_width_recovers_a_known_night_term_and_not_frame_scatter():
    rng = np.random.default_rng(1)
    n_cell, n_frame, n_ch, tau, sd_frame = 400, 3, 6, 0.02, 0.03
    cells = np.repeat(np.arange(n_cell), n_frame * n_ch)
    frames = np.repeat(np.arange(n_cell * n_frame), n_ch)
    x = np.repeat(rng.normal(0, tau, n_cell), n_frame * n_ch) + np.repeat(
        rng.normal(0, sd_frame, n_cell * n_frame), n_ch
    )
    x += rng.normal(0, 0.05, x.size)
    est, n = W.between_cell_width(x, frames, cells)
    assert n == n_cell and est == pytest.approx(tau, rel=0.25)
    cell_mean = np.array([x[cells == c].mean() for c in range(n_cell)])
    est0, _ = W.between_cell_width(
        x - np.repeat(cell_mean, n_frame * n_ch), frames, cells
    )
    assert est0 == 0.0  # cell means removed: nothing is shared by a night
