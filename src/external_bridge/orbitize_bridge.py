"""Call orbitize! (Blunt et al. 2020, AJ 159, 89; Blunt et al. 2024, JOSS 9,
6756; github.com/sblunt/orbitize) in its own environment, and map its
orbital elements to virgil's documented ones.

orbitize! is BSD-3-Clause, but it has a C extension and a long dependency
list (astropy, h5py, ptemcee, rebound, dynesty, pymultinest...), so like
CANDID it lives in .venv-orbitize (scripts/setup_external.sh, pinned
version) and runs in a subprocess (orbitize_worker.py), JSON in and out.
It is called, never vendored. Set ORBITIZE_PYTHON to use another
interpreter.

The mapping (docs/orbitize_notes.md, with sources): orbitize!'s standard
basis (sma au, ecc, inc, aop, pan, tau, plx mas, mtot M_sun) gives virgil's
KeplerOrbit as

* ``a_mas = sma * plx``;
* ``period`` = orbitize!'s own period of (sma, mtot), in days;
* ``t_ref = tau_ref_epoch`` and ``dt_peri = tau * period``;
* ``inc``, ``omega = aop``, ``Omega = pan``, in degrees, unchanged: both
  codes' argument of periastron is the companion's, and both ascending
  nodes are the node where the companion recedes;
* ``distance_pc = 1000 / plx``; RV mass ratio ``q = m1 / m0``.

Nothing here imports virgil: the mapping returns plain numbers.
"""

import math

from . import _subprocess

NAME = "orbitize"


def available():
    return _subprocess.available(NAME)


def version():
    return _subprocess.version(NAME)


def run(task, timeout=3600):
    """Run one worker task and return its result (see orbitize_worker.py)."""
    return _subprocess.run(NAME, "orbitize_worker.py", task, timeout=timeout)


def periods(cases):
    """orbitize!'s periods (days) of a list of (sma au, mtot M_sun)."""
    return [r["period_day"] for r in run({"task": "period", "cases": [list(c) for c in cases]})["results"]]


def ephemeris(orbits, epochs, tau_ref_epoch, tolerance=1e-14):
    """orbitize!'s calc_orbit for each element dict at ``epochs`` (MJD).
    ``tolerance`` is the Kepler solver's absolute tolerance on E (radians;
    orbitize!'s default is 1e-9)."""
    task = {"task": "ephemeris", "orbits": list(orbits), "epochs": [float(t) for t in epochs],
            "tau_ref_epoch": float(tau_ref_epoch), "tolerance": tolerance}
    return run(task)["results"]


def to_virgil(el, period_day, tau_ref_epoch):
    """virgil KeplerOrbit keyword arguments for orbitize! elements ``el``
    (angles in radians) with orbitize!'s period ``period_day``."""
    return {
        "period": float(period_day),
        "dt_peri": float(el["tau"]) * float(period_day),
        "ecc": float(el["ecc"]),
        "inc": math.degrees(el["inc"]),
        "omega": math.degrees(el["aop"]),
        "Omega": math.degrees(el["pan"]),
        "a_mas": float(el["sma"]) * float(el["plx"]),
        "t_ref": float(tau_ref_epoch),
    }
