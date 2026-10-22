"""Cross-validation folds for MPoL's side of the IM Lup comparison, numpy only.

virgil's dartboard folds (``virgil_bridge.alma.save_folds``, written by ``alma_imlup_maxent.py folds``) are one
``fold_NN`` and ``annulus_NN`` int8 array per ddid, in the order of that ddid's points in its ``ddid_NN.npz``,
and a JSON ``meta`` with the parameters and a sha256 of the assignment. MPoL reads an asdf, not our npz, so its
points must be matched to ours: ``match_index`` pairs them by (channel frequency rank, time, ant1, ant2), the
keys of ``npz_compare`` (neither side's channel order is assumed), and ``labels_for_ddid`` carries the fold and
annulus across. Nothing here imports MPoL or virgil, so the mask logic is tested without either.

``tally`` and ``profiled`` define the scores both codes report, in the convention of
``virgil_bridge.alma.score`` and ``alma_imlup_maxent.py cv-task``: with ``s`` the per-ddid error scale
(``g s_M`` for MPoL), the per-point -ln L is ``w |V - M|^2 / (2 s^2) + 2 ln s + ln 2 pi - ln w``, ``chi2_raw``
sums ``w |V - M|^2`` over the points (so it is about 2 per point for a good fit), and the "s = 1" variants put
``s = 1`` with the model unchanged.
"""

import json

import numpy as np

try:  # as a package module (tests) or next to mpol_worker.py (run as a script in MPoL's env)
    from .npz_compare import _rank
except ImportError:
    from npz_compare import _rank

LN2PI = float(np.log(2 * np.pi))
U_FLOOR = 1.0  # lambda: as in npz_compare


def load_folds(path):
    """``(npz arrays, meta)`` of a fold file: ``meta`` has ``k``, ``n_q``, ``n_phi``, ``seed``, ``sha256``, ``counts``."""
    z = np.load(path)
    return z, json.loads(str(z["meta"]))


def folds_sha256(fold_by_ddid):
    """sha256 of the assignment, as ``virgil_bridge.alma.folds_sha256``: the int8 folds of the ddids in increasing
    order, concatenated."""
    import hashlib

    h = hashlib.sha256()
    for d in sorted(fold_by_ddid):
        h.update(np.ascontiguousarray(fold_by_ddid[d], np.int8).tobytes())
    return h.hexdigest()


def match_index(a, b):
    """For each point of ``a`` (the asdf's), the index of the point of ``b`` (the npz's) with the same keys.

    Both are dicts with ``freq`` (Hz of the point's channel), ``ant1``, ``ant2`` and ``time``. The sort key is
    (rank of the channel's frequency among the frequencies present, time, ant1, ant2), as ``npz_compare.compare_ddid``;
    the rank, not the channel index, because visread numbers channels by decreasing frequency and our npz keeps the MS
    order. Returns ``(index, key_mismatch)``; ``key_mismatch`` counts matched pairs whose keys still differ (0 for a
    true match). A reading without keys raises: without them the order cannot be shown equal.
    """
    for name, p in (("asdf", a), ("npz", b)):
        missing = [k for k in ("freq", "ant1", "ant2", "time") if k not in p]
        if missing:
            raise ValueError(f"the {name} points lack {missing}: fold matching needs freq, ant1, ant2, time on both sides")
    if len(a["freq"]) != len(b["freq"]):
        raise ValueError(f"{len(a['freq'])} asdf points but {len(b['freq'])} npz points: cannot match them")
    if np.isnan(a["time"]).any() or (np.asarray(a["ant1"]) < 0).any():
        raise ValueError("the asdf has no antennas or times for these points")
    ra, rb = _rank(a["freq"]), _rank(b["freq"])
    ia = np.lexsort((a["ant2"], a["ant1"], a["time"], ra))
    ib = np.lexsort((b["ant2"], b["ant1"], b["time"], rb))
    key_mismatch = int(np.count_nonzero(ra[ia] != rb[ib]) + np.count_nonzero(np.asarray(a["ant1"])[ia] != np.asarray(b["ant1"])[ib])
                       + np.count_nonzero(np.asarray(a["ant2"])[ia] != np.asarray(b["ant2"])[ib])
                       + np.count_nonzero(np.abs(np.asarray(a["time"])[ia] - np.asarray(b["time"])[ib]) > 1e-3))
    index = np.empty(len(ia), np.int64)
    index[ia] = ib
    return index, key_mismatch


def labels_for_ddid(a, folds, ddid, npz, tol_rel=1e-6):
    """``(fold, annulus, report)`` of the asdf points ``a`` of one ddid: matched to ``npz`` (that ddid's arrays) and given
    the fold and annulus of the fold file ``folds`` (``fold_NN``, ``annulus_NN``). ``a`` and ``npz`` may carry ``u``, ``v``
    and ``weight`` (in lambda): if both do, the report holds the largest relative differences and a mismatch above
    ``tol_rel`` raises, because a wrong match would put points in the wrong fold. A key mismatch always raises."""
    fold_all, ann_all = np.asarray(folds[f"fold_{ddid:02d}"]), np.asarray(folds[f"annulus_{ddid:02d}"])
    if len(fold_all) != len(npz["freq"]):
        raise ValueError(f"ddid {ddid}: the fold file has {len(fold_all)} points, the npz {len(npz['freq'])}")
    idx, mismatch = match_index(a, npz)
    if mismatch:
        raise ValueError(f"ddid {ddid}: {mismatch} matched points differ in channel, time or antennas")
    rep = dict(n=int(len(idx)), key_mismatch=0)
    for k in ("u", "v"):
        if k in a and k in npz:
            rep[k + "_max_rel"] = float(np.max(np.abs(np.asarray(a[k]) - np.asarray(npz[k])[idx]) / np.maximum(np.abs(np.asarray(npz[k])[idx]), U_FLOOR))) if len(idx) else 0.0
    if "weight" in a and "weight" in npz:
        wb = np.asarray(npz["weight"], float)[idx]
        rep["weight_max_rel"] = float(np.max(np.abs(np.asarray(a["weight"], float) - wb) / np.maximum(np.abs(wb), 1e-300))) if len(idx) else 0.0
    bad = {k: v for k, v in rep.items() if k.endswith("_max_rel") and v > tol_rel}
    if bad:
        raise ValueError(f"ddid {ddid}: matched points differ beyond {tol_rel:g}: {bad}")
    return fold_all[idx].astype(np.int8), ann_all[idx].astype(np.int8), rep


def tally(ids, size, nll, r2, r2_fit):
    """Sums of the per-point -ln L, ``r2 = w |V - M|^2`` (quoted weights) and ``r2_fit = r2 / s^2`` over the points with each
    id in ``0..size-1`` (a ddid or an annulus): the layout of ``virgil_bridge.alma.score``'s ``by_ddid`` / ``by_annulus``."""
    ids = np.asarray(ids).astype(np.int64)
    return dict(n=np.bincount(ids, minlength=size).tolist(), nll=np.bincount(ids, np.asarray(nll, float), size).tolist(),
                chi2_raw=np.bincount(ids, np.asarray(r2, float), size).tolist(), chi2_fit=np.bincount(ids, np.asarray(r2_fit, float), size).tolist())


def add_tallies(x, y):
    """Elementwise sum of two ``tally`` dicts (batches of one set)."""
    return {k: (np.asarray(x[k]) + np.asarray(y[k])).tolist() for k in x}


def profiled(n, nll_s1, chi2_raw):
    """Held-out -ln L with the error scale profiled out: one global ``s`` per set, at its maximum-likelihood value
    ``s^2 = chi2_raw / (2 n)``, from the s = 1 numbers alone. -ln L(s) = (nll_s1 - chi2_raw / 2) + chi2_raw / (2 s^2) + 2 n ln s
    (the bracket is N ln 2 pi - sum ln w, which does not depend on the model), so the profile is
    ``nll_s1 - chi2_raw / 2 + n (1 + ln(chi2_raw / (2 n)))``. Returns ``(profiled_nll, s_hat)``; works on any set, so also on
    one annulus. NaN for an empty set."""
    n = np.asarray(n, float)
    chi2_raw, nll_s1 = np.asarray(chi2_raw, float), np.asarray(nll_s1, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s2 = chi2_raw / (2 * n)
        return nll_s1 - 0.5 * chi2_raw + n * (1 + np.log(s2)), np.sqrt(s2)
