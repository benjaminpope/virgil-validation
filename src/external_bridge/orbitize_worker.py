"""Run inside orbitize!'s own environment (.venv-orbitize), not ours.

    python orbitize_worker.py task.json result.json

orbitize! (Blunt et al. 2020, AJ 159, 89; Blunt et al. 2024, JOSS 9, 6756;
github.com/sblunt/orbitize, BSD-3-Clause) fits Keplerian orbits to relative
astrometry and radial velocities. It is installed at a pinned version by
scripts/setup_external.sh and called, never vendored. This worker imports
only orbitize!, NumPy, astropy and the standard library.

Elements are orbitize!'s standard basis (docs/orbitize_notes.md): ``sma``
(au), ``ecc``, ``inc``, ``aop``, ``pan`` (radians; ``aop`` is the
companion's argument of periastron), ``tau`` (periastron epoch as a fraction
of the period after ``tau_ref_epoch``), ``plx`` (mas) and ``mtot`` (M_sun).
"""

import json
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import orbitize  # noqa: E402
from orbitize import basis, kepler, system  # noqa: E402

KEYS = ("sma", "ecc", "inc", "aop", "pan", "tau", "plx", "mtot")


def _period_basis():
    """orbitize!'s Period basis for one companion (indices are set when its
    priors are constructed)."""
    b = basis.Period(1.0, 0.0, 1.0, 0.0, 1, False)
    b.construct_priors()
    return b


def ephemeris(task):
    """kepler.calc_orbit for each element set at the epochs (MJD): RA and
    Dec offsets of the companion from the primary (mas), their separation
    and PA from system.radec2seppa, and the RV of the body named by
    ``mass_for_Kamp`` (km/s; None: the relative velocity of the companion,
    mass_for_Kamp = mtot)."""
    epochs = np.asarray(task["epochs"], float)
    out = []
    for el in task["orbits"]:
        p = [float(el[k]) for k in KEYS]
        ra, dec, vz = kepler.calc_orbit(
            epochs, *p, mass_for_Kamp=el.get("mass_for_Kamp"),
            tau_ref_epoch=float(task["tau_ref_epoch"]),
            tolerance=float(task.get("tolerance", 1e-9)), use_c=bool(task.get("use_c", True)),
        )
        ra, dec, vz = (np.reshape(np.asarray(x, float), epochs.shape) for x in (ra, dec, vz))
        sep, pa = system.radec2seppa(ra, dec)
        out.append({"raoff": ra.tolist(), "deoff": dec.tolist(), "vz": vz.tolist(),
                    "sep": np.asarray(sep).tolist(), "pa": np.asarray(pa).tolist()})
    return {"results": out}


def period(task):
    """Periods (days) of (sma, mtot) by orbitize!'s Period basis (its
    Kepler's third law with astropy's G and M_sun), and back: the sma that
    the Period basis gives for that period and mtot."""
    b = _period_basis()
    idx = b.standard_basis_idx
    out = []
    for sma, mtot in task["cases"]:
        arr = np.zeros(max(idx.values()) + 1)
        arr[idx["sma1"]], arr[idx["mtot"]], arr[idx["plx"]] = sma, mtot, 1.0
        per_yr = float(b.to_period_basis(arr.copy())[b.param_idx["per1"]])
        back = arr.copy()
        back[b.param_idx["per1"]] = per_yr
        sma_back = float(b.to_standard_basis(back)[idx["sma1"]])
        out.append({"period_yr": per_yr, "period_day": per_yr * YEAR_DAYS, "sma_back": sma_back})
    return {"results": out, "year_days": YEAR_DAYS}


def system_rv(task):
    """RV model rows of an orbitize! System with fitted masses
    (fit_secondary_mass=True): one RV per epoch of each body in
    ``objects`` (0 the primary, 1 the companion), all from one instrument,
    evaluated by System.compute_model at the standard-basis elements plus
    m0, m1 and, when orbitize! fits them, the instrument's gamma (km/s) and
    zero jitter. Also returns which parameters the System fits."""
    from astropy.table import Table
    from orbitize import read_input

    epochs = [float(t) for t in task["epochs"]]
    objects = [int(o) for o in task.get("objects", [0, 1])]
    rows = [(t, o) for o in objects for t in epochs]
    table = Table({
        "epoch": [t for t, _ in rows], "object": [o for _, o in rows],
        "rv": [0.0] * len(rows), "rv_err": [1.0] * len(rows), "instrument": ["spec"] * len(rows),
    })
    el = task["orbit"]
    sys_ = system.System(1, read_input.read_file(table), el["m0"], el["plx"], mass_err=0.1, plx_err=0.1,
                         tau_ref_epoch=float(task["tau_ref_epoch"]), fit_secondary_mass=True)
    lab = sys_.basis.param_idx
    values = {"sma1": el["sma"], "ecc1": el["ecc"], "inc1": el["inc"], "aop1": el["aop"],
              "pan1": el["pan"], "tau1": el["tau"], "plx": el["plx"], "m1": el["m1"], "m0": el["m0"],
              "gamma_spec": float(task.get("gamma", 0.0)), "sigma_spec": 0.0}
    p = np.zeros(len(lab))
    for k, v in values.items():
        if k in lab:
            p[lab[k]] = v
    model = np.asarray(sys_.compute_model(p)[0])[:, 0]
    n = len(epochs)
    out = {name: model[k * n:(k + 1) * n].tolist()
           for k, name in enumerate({0: "primary", 1: "secondary"}[o] for o in objects)}
    out["labels"] = list(sys_.labels)
    return out


def posterior(task):
    """Posterior samples for the companion of an orbitize! data file (e.g.
    the sep/PA rows of orbitize/example_data/betaPic.csv) by ptemcee
    ("MCMC") or OFTI, with the priors of task["priors"]:

    * sma: LogUniformPrior(lo, hi) (au);
    * ecc: UniformPrior(lo, hi);
    * inc, aop, pan, tau: orbitize!'s defaults, which are the invariant ones
      asked for: SinPrior on [0, pi] (uniform in cos i), and uniform on
      [0, 2 pi), [0, 2 pi) and [0, 1) (returned, as the sampler holds them);
    * plx, mtot: Gaussians (plx_err, mass_err).

    As orbitize!'s "Modifying Priors" tutorial says, a sampler copies the
    system's priors when it is made, so the System is built and its priors
    replaced first, and the sampler made from it afterwards (not through
    Driver, which makes both at once).

    Returns the samples of sma, ecc, inc, aop, pan, tau, plx and mtot, and
    the priors the sampler used, by name."""
    from orbitize import priors, read_input, sampler

    data = read_input.read_file(task["path"])
    s = system.System(1, data, task["mtot"], task["plx"], mass_err=task["mtot_err"], plx_err=task["plx_err"],
                      tau_ref_epoch=task["tau_ref_epoch"])
    lab = s.param_idx
    pr = task["priors"]
    s.sys_priors[lab["sma1"]] = priors.LogUniformPrior(*pr["sma"])
    assert pr["ecc"][0] == "uniform"
    s.sys_priors[lab["ecc1"]] = priors.UniformPrior(*pr["ecc"][1:])
    names = ["sma1", "ecc1", "inc1", "aop1", "pan1", "tau1", "plx", "mtot"]
    if task["algorithm"] == "OFTI":
        smp = sampler.OFTI(s)
        samples = smp.run_sampler(int(task["n"]))
        used = s.sys_priors
    else:
        smp = sampler.MCMC(s, **(task.get("mcmc_kwargs") or {}))
        smp.run_sampler(int(task["n"]), burn_steps=int(task.get("burn", 0)), thin=int(task.get("thin", 1)))
        samples = smp.results.post
        used = smp.priors  # the sampler's own copy (fixed parameters dropped; none here)
    samples = np.asarray(samples)
    out = {k.rstrip("1"): samples[:, lab[k]].tolist() for k in names}
    out["priors"] = {k: f"{used[lab[k]]!r} {getattr(used[lab[k]], '__dict__', '')}" for k in names}
    return out


def batch(task):
    """Several tasks in one process (importing orbitize! takes ~10 s):
    ``tasks`` maps names to tasks; returns the results by name."""
    return {"results": {name: TASKS[t["task"]](t) for name, t in task["tasks"].items()}}


YEAR_DAYS = None  # set in main: astropy's year (orbitize!'s period unit), in days

TASKS = {"ephemeris": ephemeris, "period": period, "system_rv": system_rv, "posterior": posterior, "batch": batch}


def main():
    global YEAR_DAYS
    import astropy.units as u

    YEAR_DAYS = float(u.year.to(u.day))
    task = json.load(open(sys.argv[1]))
    result = TASKS[task["task"]](task)
    result["orbitize_version"] = orbitize.__version__
    result["c_solver"] = bool(orbitize.cext)
    json.dump(result, open(sys.argv[2], "w"))


if __name__ == "__main__":
    main()
