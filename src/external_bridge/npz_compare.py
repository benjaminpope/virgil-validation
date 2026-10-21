"""Point-by-point comparison of two readings of one ddid (an asdf's points against an npz), numpy only.

Used by ``mpol_worker.asdf_vs_npz``. Both inputs are dicts with ``u, v, vis, weight`` (equal length) and,
for key matching, ``ant1, ant2, time`` plus ``chan`` (asdf: channel index, ascending frequency, as visread
numbers them) or ``freq`` (npz: the frequency of each point; our npz keeps the MS channel order, which is
decreasing in frequency for some ddids).
"""
import numpy as np


def _rank(x):
    """Dense rank of each value among the values present (0 = smallest)."""
    return np.unique(x, return_inverse=True)[1].ravel()


def compare_ddid(a, b, tol_vis=1e-5, tol_rel=1e-6):
    """Match the points of ``a`` (asdf) and ``b`` (npz) and report the differences and PASS/FAIL.

    Matching, when both have antennas and time: sort by (channel rank, time, ant1, ant2) on each side, the
    channel rank being the rank of the channel's frequency, ascending (asdf: of its channel index, which is
    already ascending in frequency), so MS channel order does not matter. Otherwise the order is assumed
    equal. ``pass``: every key matches, |dV| < tol_vis (Jy), and the u, v and weight differences, relative to
    the largest |u|, |v| and to each weight, are < tol_rel. Reports the conjugation and unit signatures too.
    """
    mx = lambda x: float(np.max(np.abs(x))) if np.size(x) else 0.0  # noqa: E731
    have_keys = all(k in b for k in ("time", "ant1", "ant2")) and ("freq" in b or "chan" in b) \
        and not np.isnan(a["time"]).any() and (a["ant1"] >= 0).all()
    res = {}
    if have_keys:
        ra = _rank(a["chan"])
        rb = _rank(b["freq"]) if "freq" in b else _rank(b["chan"])
        ia = np.lexsort((a["ant2"], a["ant1"], a["time"], ra))
        ib = np.lexsort((b["ant2"], b["ant1"], b["time"], rb))
        res["matching"] = "keys"
        res["key_mismatch"] = int(np.count_nonzero(ra[ia] != rb[ib]) + np.count_nonzero(a["ant1"][ia] != b["ant1"][ib])
                                  + np.count_nonzero(a["ant2"][ia] != b["ant2"][ib])
                                  + np.count_nonzero(np.abs(a["time"][ia] - b["time"][ib]) > 1e-3))
    else:
        ia = ib = slice(None)
        res["matching"] = "order"
    ua, va, wa, za = a["u"][ia], a["v"][ia], np.asarray(a["weight"][ia], float), np.asarray(a["vis"][ia], complex)
    ub, vb, wb, zb = b["u"][ib], b["v"][ib], np.asarray(b["weight"][ib], float), np.asarray(b["vis"][ib], complex)
    res.update(
        u_max_abs=mx(ua - ub), v_max_abs=mx(va - vb), vis_max_abs=mx(za - zb), weight_max_abs=mx(wa - wb),
        u_max_rel=mx(ua - ub) / max(mx(ub), 1e-300), v_max_rel=mx(va - vb) / max(mx(vb), 1e-300),
        weight_max_rel=mx((wa - wb) / np.maximum(np.abs(wb), 1e-300)),
        vis_conj_max_abs=mx(za - np.conj(zb)), u_flip_max_abs=mx(ua + ub), v_flip_max_abs=mx(va + vb),
        u_ratio_median=float(np.median(np.abs(ua) / np.maximum(np.abs(ub), 1e-300))) if len(ua) else None,
    )
    res["pass"] = bool(res.get("key_mismatch", 0) == 0 and res["vis_max_abs"] < tol_vis and res["u_max_rel"] < tol_rel
                       and res["v_max_rel"] < tol_rel and res["weight_max_rel"] < tol_rel)
    return res
