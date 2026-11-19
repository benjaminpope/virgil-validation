"""Fixed calibration loop (about 1 s): pure-Python arithmetic plus a fixed 300 x 300 float64 matmul.

``t_calib`` is recorded in every row; a row whose ``t_calib`` exceeds 1.3x the session median (per language) is flagged
``noisy`` (it probably ran on an efficiency core or beside other load). Julia's equivalent is ``calib.jl``.
Run: ``python calib.py`` prints ``{"t_calib": seconds}``.
"""

import json
import time


def calibrate(n_loop=2_000_000, n_mat=300, n_rep=5):
    """Wall time of the fixed workload, seconds. The matmul is skipped (and the loop doubled) without NumPy."""
    t0 = time.perf_counter()
    s = 0.0
    for i in range(n_loop):
        s += (i % 7) * 0.5 - (i % 3) * 0.25
    try:
        import numpy as np
    except ImportError:
        for i in range(n_loop):
            s += (i % 5) * 0.5
    else:
        a = np.arange(n_mat * n_mat, dtype=float).reshape(n_mat, n_mat) % 11 / 11.0
        for _ in range(n_rep):
            a = a @ a
            a /= np.abs(a).max()
        s += float(a[0, 0])
    dt = time.perf_counter() - t0
    calibrate.last_checksum = s
    return dt


if __name__ == "__main__":
    print(json.dumps({"t_calib": calibrate()}))
