"""Shared by tests/test_limb_darkening.py and tests/test_reference_limb.py: the baselines, the
tolerance and the comparison. Imports no reference module, so that using it adds nothing to what a
check went through (scripts/trust.py, ``via``)."""

import numpy as np

from evidence.plugin import record

TOL64 = 1e-10

LAWS = {
    "uniform": [],
    "linear": [0.6],
    "quadratic": [0.35, 0.25],
    "cubic": [0.3, 0.2, 0.1],
    "order-22": list(np.linspace(0.05, 0.01, 22)),  # the highest documented
}


def uvw_samples():
    """VLTI-like random baselines, plus a dense radial cut so that every
    null of a 12 mas disk out to 130 m / 1.5 um has samples beside it."""
    rng = np.random.default_rng(2026)
    n = 300
    radial = np.linspace(0.0, 130.0, 1500)
    pa = np.deg2rad(37.0)
    return (
        np.concatenate([rng.uniform(-130, 130, n), radial * np.sin(pa)]),
        np.concatenate([rng.uniform(-130, 130, n), radial * np.cos(pa)]),
        np.concatenate([rng.uniform(1.5e-6, 2.4e-6, n), np.full(1500, 1.5e-6)]),
    )


def assert_close(ours, theirs, tol=TOL64):
    err = record("max_abs_dV", np.max(np.abs(np.asarray(ours) - np.asarray(theirs))))
    assert err < tol, f"max |dV| = {err:.3e}"
