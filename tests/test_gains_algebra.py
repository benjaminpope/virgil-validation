"""The structure of virgil.gains' mode objects, against our own triangle and
telescope algebra (crosscheck.chi2.triangle_matrix, our reading of the file).

test_linear_marginals checks the marginal likelihoods these objects give.
Here the objects themselves:

* GainModes: scattering each block's shapes, times its group's width, onto
  the visibility samples gives U, and U Uᵀ (independent of the order of the
  modes) equals ours: per frame, one indicator per telescope on the
  baselines that include it, one per baseline over its channels, one per
  baseline shaped (λ_ref / λ)², λ_ref the median wavelength.
* ClosureOffsets: the same with closure phases, through ``cp_noise.groups``:
  per frame, a baseline offset e reaches the closure phases as T e (T the
  triangle-by-baseline signs), a triangle offset as an indicator; so
  Σ m mᵀ is τ² T Tᵀ per channel, or τ² I.
* Station phases: a phase g_k on telescope k shifts baseline (i, j) by
  g_j - g_i and so never reaches a closure phase, T A = 0 for our T and the
  station incidence A (a reference check of our algebra); and with four
  telescopes the one combination of closure phases outside the span of T
  carries no information, so a closure-offset mode along it leaves virgil's
  likelihood unchanged.
"""

import numpy as np
import pytest

from crosscheck import chi2 as ours, simulate, sky
from evidence.plugin import record

pytest.importorskip("virgil.models")
import virgil.models as vm  # noqa: E402
from virgil import gains  # noqa: E402
from virgil.likelihood import model_loglike  # noqa: E402
from virgil.oidata import OIData  # noqa: E402

pytestmark = pytest.mark.x64

UTS = np.array([[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]])
WL = np.linspace(1.6e-6, 2.4e-6, 4)


def truth(u, v, w):
    return (sky.vis_uniform_disk(u, v, w, 1.5) + 0.05 * sky.vis_point(u, v, w, 6.0, -4.0)) / 1.05


@pytest.fixture(scope="module")
def obs(tmp_path_factory):
    path = tmp_path_factory.mktemp("gains_algebra") / "obs.fits"
    simulate.observe(path, truth, UTS, hour_angles_h=[-2.0, 1.0], wavelengths=WL, dec_deg=-50.0,
                     sigma_v2=0.01, sigma_cp_deg=1.0, rng=np.random.default_rng(3))
    return ours.load(path), OIData(str(path))


def our_gain_modes(d, telescope, baseline, chromatic):
    mjd, sta, wl = d["v2_mjd"], d["v2_sta"], d["wl"]
    nrow, nw = wl.shape
    lref = np.median(np.unique(wl))
    cols = []
    for m in np.unique(mjd):
        rows = np.flatnonzero(mjd == m)
        for s in np.unique(sta[rows]):
            shape = np.zeros((nrow, nw))
            shape[rows[(sta[rows] == s).any(1)], :] = telescope
            cols.append(shape.ravel())
        for row in rows:
            shape = np.zeros((nrow, nw))
            shape[row, :] = baseline
            cols.append(shape.ravel())
            shape = np.zeros((nrow, nw))
            shape[row, :] = chromatic * (lref / wl[row]) ** 2
            cols.append(shape.ravel())
    return np.array(cols).T


@pytest.mark.validates("virgil.gains.GainModes", "virgil.gains.gain_modes", roots=["mathematics"])
def test_gain_modes_span_the_documented_shapes(obs):
    d, data = obs
    widths = dict(telescope=0.02, baseline=0.01, chromatic=0.03)
    g = gains.gain_modes(data, **widths)
    assert g.groups == ("telescope", "baseline", "chromatic") and g.spanning is None
    rows, shapes, group = np.asarray(g.rows), np.asarray(g.shapes), np.asarray(g.group)
    w = np.asarray(g.widths)
    cols = []
    for b in range(rows.shape[0]):
        real = rows[b] < g.n_vis
        for k in range(shapes.shape[2]):
            col = np.zeros(g.n_vis)
            col[rows[b, real]] = shapes[b, real, k] * w[group[b, k]]
            cols.append(col)
    U = np.array(cols).T
    mine = our_gain_modes(d, **widths)
    err = record("max_abs_dUUt", np.max(np.abs(U @ U.T - mine @ mine.T)))
    assert err < 1e-15
    assert U.shape[1] == mine.shape[1]  # one mode per (frame, telescope), and two per (frame, baseline)


def dense_offsets(co, data):
    """Σ τ² m mᵀ over the closure phases, from ClosureOffsets' blocks."""
    slots = np.asarray(data.cp_noise.groups)
    mask = np.asarray(data.cp_noise.mask)
    values, group = np.asarray(co.values), np.asarray(co.group)
    groups, gmask = np.asarray(co.groups), np.asarray(co.group_mask)
    w = np.asarray(co.widths)
    n = np.asarray(data.phi).size
    total = np.zeros((n, n))
    for b in range(values.shape[0]):
        for k in range(values.shape[1]):
            m = np.zeros(n)
            for j, (gi, real) in enumerate(zip(groups[b], gmask[b])):
                if real:
                    keep = mask[gi]
                    m[slots[gi][keep]] += values[b, k, j][keep]
            total += (w[group[b, k]] * m)[:, None] * (w[group[b, k]] * m)[None, :]
    return total


def our_offsets(d, baseline=0.0, triangle=0.0):
    nrow, nw = d["cp"].shape
    total = np.zeros((nrow * nw, nrow * nw))
    for mjd in np.unique(d["t3_mjd"]):
        rows = np.flatnonzero(d["t3_mjd"] == mjd)
        T = ours.triangle_matrix(d["t3_sta"][rows])
        cols = [baseline * T[:, b] for b in range(T.shape[1])] + [triangle * e for e in np.eye(len(rows))]
        for col in cols:
            shape = np.zeros((nrow, nw))
            shape[rows, :] = col[:, None]
            m = shape.ravel()
            total += np.outer(m, m)
    return total


@pytest.mark.parametrize("kw", [dict(baseline=0.02), dict(triangle=0.03), dict(baseline=0.02, triangle=0.03)],
                         ids=["baseline", "triangle", "both"])
@pytest.mark.validates("virgil.gains.ClosureOffsets", "virgil.gains.closure_offsets", roots=["mathematics"])
def test_closure_offsets_enter_as_documented(obs, kw):
    d, data = obs
    co = gains.closure_offsets(data, **kw)
    assert co.groups_present == tuple(k for k in ("baseline", "triangle") if k in kw)
    err = record("max_abs_dcov", np.max(np.abs(dense_offsets(co, data) - our_offsets(d, **kw))))
    assert err < 1e-15


@pytest.mark.validates("crosscheck.chi2", roots=["mathematics"], kind="reference")
def test_station_phases_cancel_in_our_triangle_algebra():
    stations = [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)]
    T = ours.triangle_matrix(stations)
    pairs = sorted({tuple(sorted(p)) for a, b, c in stations for p in ((a, b), (b, c), (a, c))})
    A = np.zeros((len(pairs), 4))
    for k, (i, j) in enumerate(pairs):
        A[k, j - 1], A[k, i - 1] = 1.0, -1.0  # baseline (i, j) gains g_j - g_i
    assert np.max(np.abs(T @ A)) == 0.0
    assert np.linalg.matrix_rank(T) == 3  # 4 closure phases, 3 independent


@pytest.mark.validates("virgil.gains.closure_offsets", "virgil.oidata.OIData.cp_noise", roots=["mathematics"])
def test_offsets_outside_the_span_of_T_do_not_enter(obs):
    """With four telescopes, n with nᵀ T = 0 is the dependent combination of
    a frame's closure phases; an offset along n, of any size, changes
    nothing, while one along a column of T does."""
    d, data = obs
    nrow, nw = d["cp"].shape
    model = vm.System(star=vm.UniformDisk(1.45), comp=vm.PointSource(0.04, 6.2, -3.9))
    base = float(model_loglike(model, data))
    null_modes, live_modes = [], []
    for mjd in np.unique(d["t3_mjd"]):
        rows = np.flatnonzero(d["t3_mjd"] == mjd)
        T = ours.triangle_matrix(d["t3_sta"][rows])
        n = np.linalg.svd(T.T)[2][-1]  # left null vector of T
        assert np.max(np.abs(n @ T)) < 1e-12
        for vec, out in ((n, null_modes), (T[:, 0], live_modes)):
            shape = np.zeros((nrow, nw))
            shape[rows, :] = 0.3 * vec[:, None]
            out.append(shape.ravel())
    null = float(model_loglike(model, data.with_closure_offsets(modes=np.array(null_modes)))) - base
    live = float(model_loglike(model, data.with_closure_offsets(modes=np.array(live_modes)))) - base
    record("abs_dloglike_null", abs(null))
    assert abs(null) < 1e-10 and abs(live) > 1e-3
