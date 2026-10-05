"""virgil's Keplerian orbits against orbitize!.

orbitize! (Blunt et al. 2020, AJ 159, 89; Blunt et al. 2024, JOSS 9, 6756;
github.com/sblunt/orbitize, BSD-3-Clause) is the orbit-fitting package of
the direct-imaging community, written and used by people. It runs in its
own environment (scripts/setup_external.sh) through
src/external_bridge/orbitize_bridge.py; virgil never depends on it.

The mapping between the two (docs/orbitize_notes.md, with sources) is the
identity on the angles: both arguments of periastron are the companion's,
and both ascending nodes are the node where the companion recedes. Lengths
and times map as a_mas = sma * plx, distance_pc = 1000 / plx,
t_ref = tau_ref_epoch, dt_peri = tau * P, with P orbitize!'s own period
of (sma, mtot).

Tolerances (float64). Positions agree to 1e-12 of the semimajor axis: both
codes evaluate the same closed form, orbitize!'s Kepler solve is set to
1e-14 rad in E (its default, 1e-9, would dominate), and mean anomalies of a
few orbits carry ~1e-15 of rounding; the margin of ~100 covers libm
differences between platforms. Radial velocities agree to 1e-12 of the
semi-amplitude for the same reason; through orbitize!'s System, which
solves Kepler's equation at its default 1e-9, to 1e-8. Finite-difference
velocities (five-point stencil, h = P/2000, on orbits with e <= 0.6) agree
to 1e-6 of 2 pi a / P.

Findings (ledger): F14, virgil's total_mass uses a³/P² with P in Julian
years, which is Kepler's third law only to 3.8e-5; P7, orbitize! adds the
instrument's gamma to companion RVs when primary RVs are present, although
its documentation asks for companion RVs relative to the barycentre.
"""

import math

import numpy as np
import pytest
from astropy import constants, units

from evidence.plugin import record
from external_bridge import orbitize_bridge as ob

pytest.importorskip("jaxoplanet", reason="virgil.orbits needs virgil-astro[orbits]")
vo = pytest.importorskip("virgil.orbits")

pytestmark = [
    pytest.mark.x64,
    pytest.mark.external,
    pytest.mark.skipif(not ob.available(), reason="orbitize! not installed (scripts/setup_external.sh orbitize)"),
]

REF = 58849.0  # orbitize!'s default tau_ref_epoch
TOL_POS = 1e-12  # of a_mas
TOL_RV = 1e-12  # of the semi-amplitude
TOL_SYSTEM = 1e-8  # through orbitize.system.System (Kepler solve at 1e-9)
AU_KM = 149597870.7  # IAU 2012 Resolution B2
GM_SUN = 1.3271244e20  # m^3 s^-2, IAU 2015 Resolution B3 (nominal)
JULIAN_YEAR_D = 365.25


def _grid():
    """Element sets (orbitize!'s standard basis, radians) over e, i, omega,
    Omega and phase, with the edge cases: e = 0 and ~0, e up to 0.95,
    i = 0, ~0, ~90, 90 and above 90."""
    rng = np.random.default_rng(14)
    out = []
    for e in (0.0, 1e-7, 0.1, 0.5, 0.8, 0.95):
        for i in (0.0, 1e-4, 35.0, 89.99, 90.0, 120.0, 179.9):
            out.append({
                "sma": float(rng.uniform(1, 40)), "ecc": e, "inc": math.radians(i),
                "aop": float(rng.uniform(0, 2 * np.pi)), "pan": float(rng.uniform(0, 2 * np.pi)),
                "tau": float(rng.uniform()), "plx": float(rng.uniform(5, 100)), "mtot": float(rng.uniform(0.3, 5)),
            })
    for aop, pan in ((0, 0), (90, 0), (0, 90), (180, 270), (270, 180)):  # cardinal angles
        out.append({"sma": 5.0, "ecc": 0.3, "inc": math.radians(60), "aop": math.radians(aop),
                    "pan": math.radians(pan), "tau": 0.0, "plx": 20.0, "mtot": 1.0})
    return out


GRID = _grid()


@pytest.fixture(scope="module")
def grid():
    """orbitize!'s periods and ephemerides over 1.6 periods of each orbit
    (121 epochs), and virgil's KeplerOrbit at the mapped elements."""
    periods = ob.periods([(el["sma"], el["mtot"]) for el in GRID])
    out = []
    for el, p in zip(GRID, periods):
        epochs = REF - 0.3 * p + np.linspace(0, 1.6 * p, 121)
        out.append((el, p, epochs))
    # one worker call per distinct epoch set is wasteful: ask for all at once
    # on a common unit-phase grid instead, by evaluating each orbit separately
    results = [ob.ephemeris([el], epochs, REF)[0] for el, _, epochs in out] if len(out) < 4 else None
    if results is None:
        results = _batched(out)
    return [(el, p, epochs, r, vo.KeplerOrbit(**ob.to_virgil(el, p, REF))) for (el, p, epochs), r in zip(out, results)]


def _batched(items):
    """One orbitize! call for every orbit, each at its own epochs (the
    worker takes one epoch list, so concatenate and slice)."""
    epochs = np.concatenate([e for _, _, e in items])
    res = ob.ephemeris([el for el, _, _ in items], epochs, REF)
    n = len(items[0][2])
    return [{k: np.asarray(r[k])[j * n:(j + 1) * n] for k in ("raoff", "deoff", "vz", "sep", "pa")}
            for j, r in enumerate(res)]


def _sky_error(dra, ddec, r, a):
    return max(np.max(np.abs(np.asarray(dra) - r["raoff"])), np.max(np.abs(np.asarray(ddec) - r["deoff"]))) / a


# ------------------------------------------------------------ orbitize! itself


@pytest.mark.validates("orbitize", "external_bridge.orbitize_bridge", roots=["standards", "mathematics"], kind="reference")
def test_orbitize_period_is_keplers_third_law():
    """orbitize!'s period of (sma, mtot) is P = 2 pi sqrt(a³ / (G M)) with
    the IAU nominal GM_sun (astropy's G * M_sun) and the IAU au, and its
    year is the Julian year (astropy's), so the mapping's P is exact."""
    cases = [(1.0, 1.0), (10.0, 1.7), (0.05, 0.3), (300.0, 20.0)]
    r = ob.run({"task": "period", "cases": [list(c) for c in cases]})
    assert r["year_days"] == JULIAN_YEAR_D
    assert abs(float((constants.G * constants.M_sun).to_value("m3 s-2")) / GM_SUN - 1) < 1e-15
    worst = 0.0
    for (sma, mtot), res in zip(cases, r["results"]):
        expect = 2 * np.pi * np.sqrt((sma * AU_KM * 1e3) ** 3 / (GM_SUN * mtot)) / 86400.0
        worst = max(worst, abs(res["period_day"] / expect - 1), abs(res["sma_back"] / sma - 1))
    record("max_rel_period", worst)
    assert worst < 1e-14


# ----------------------------------------------------------------- ephemerides


@pytest.mark.validates("virgil.orbits.KeplerOrbit", roots=["orbitize"])
def test_relative_astrometry_matches_orbitize(grid):
    """KeplerOrbit.relative (dra, ddec) and separation_pa against
    orbitize!'s calc_orbit and radec2seppa over the element grid (47 orbits,
    121 epochs each over 1.6 periods)."""
    worst_pos = worst_sep = worst_pa = 0.0
    for el, p, epochs, r, orbit in grid:
        a = el["sma"] * el["plx"]
        dra, ddec, _ = orbit.relative(epochs)
        worst_pos = max(worst_pos, _sky_error(dra, ddec, r, a))
        sep, pa = (np.asarray(x) for x in orbit.separation_pa(epochs))
        worst_sep = max(worst_sep, np.max(np.abs(sep - r["sep"])) / a)
        dpa = (pa - np.asarray(r["pa"]) + 180.0) % 360.0 - 180.0
        # an angle error times the separation: a length, well defined at sep -> 0
        worst_pa = max(worst_pa, np.max(np.abs(np.radians(dpa)) * sep) / a)
    record("max_rel_position", worst_pos)
    record("max_rel_separation", worst_sep)
    record("max_rel_pa_times_sep", worst_pa)
    assert worst_pos < TOL_POS and worst_sep < TOL_POS and worst_pa < TOL_POS


@pytest.mark.validates("virgil.orbits.KeplerOrbit", "external_bridge.orbitize_bridge", roots=["orbitize"], kind="control")
def test_wrong_mappings_fail(grid):
    """The mappings a reader could plausibly get wrong are wrong by the size
    of the orbit: the primary's argument of periastron (omega + 180°, as in
    jaxoplanet and spectroscopy), a node counted counterclockwise from East
    (90° - Omega), and the opposite sense of rotation (180° - inc)."""
    el, p, epochs, r, _ = grid[30]  # e = 0.8, i = 35°
    a = el["sma"] * el["plx"]
    good = ob.to_virgil(el, p, REF)
    for change in ({"omega": good["omega"] + 180.0},
                   {"Omega": (90.0 - good["Omega"]) % 360.0},
                   {"inc": 180.0 - good["inc"]}):
        dra, ddec, _ = vo.KeplerOrbit(**{**good, **change}).relative(epochs)
        err = _sky_error(dra, ddec, r, a)
        record("min_rel_error_" + next(iter(change)), err)
        assert err > 0.1, change


@pytest.mark.validates("virgil.orbits.ThieleInnesOrbit", roots=["orbitize", "mathematics"])
def test_thiele_innes_orbit_matches_orbitize(grid):
    """A ThieleInnesOrbit built from the constants of virgil's design note
    §2.4 (A = a(cos w cos W - sin w sin W cos i), ...) reproduces orbitize!'s
    positions; KeplerOrbit.thiele_innes gives those constants; and
    to_kepler returns orbitize!'s elements, with Omega in [0°, 180°) and
    the documented (Omega + 180°, omega + 180°) twin."""
    worst_pos = worst_const = worst_el = 0.0
    for el, p, epochs, r, orbit in grid:
        a, w, W, i = el["sma"] * el["plx"], el["aop"], el["pan"], el["inc"]
        A = a * (math.cos(w) * math.cos(W) - math.sin(w) * math.sin(W) * math.cos(i))
        B = a * (math.cos(w) * math.sin(W) + math.sin(w) * math.cos(W) * math.cos(i))
        F = a * (-math.sin(w) * math.cos(W) - math.cos(w) * math.sin(W) * math.cos(i))
        G = a * (-math.sin(w) * math.sin(W) + math.cos(w) * math.cos(W) * math.cos(i))
        ti = vo.ThieleInnesOrbit(p, el["tau"] * p, el["ecc"], A, B, F, G, t_ref=REF)
        dra, ddec = ti.sky(epochs)
        worst_pos = max(worst_pos, _sky_error(dra, ddec, r, a))
        worst_const = max(worst_const, np.max(np.abs(np.asarray(orbit.thiele_innes()[:4]) - [A, B, F, G])) / a)
        if el["ecc"] > 1e-3 and 1.0 < math.degrees(i) < 179.0:
            k = ti.to_kepler()
            Om, om = math.degrees(W) % 360.0, math.degrees(w)
            if Om >= 180.0:
                Om, om = Om - 180.0, om - 180.0
            d_ang = [((float(x) - y + 180.0) % 360.0) - 180.0
                     for x, y in ((k.Omega, Om), (k.omega, om), (k.inc, math.degrees(i)))]
            worst_el = max(worst_el, np.max(np.abs(d_ang)), abs(float(k.a_mas) / a - 1) * 180)
    record("max_rel_position", worst_pos)
    record("max_rel_constants", worst_const)
    record("max_element_error_deg", worst_el)
    assert worst_pos < TOL_POS and worst_const < 1e-14
    assert worst_el < 1e-6  # degrees: angles from atan2 of the constants


def _fd(fun, t, h):
    """Five-point central difference."""
    return (-fun(t + 2 * h) + 8 * fun(t + h) - 8 * fun(t - h) + fun(t - 2 * h)) / (12 * h)


@pytest.mark.validates("virgil.orbits.StateVectorOrbit", roots=["orbitize"])
def test_state_vector_orbit_matches_orbitize(grid):
    """StateVectorOrbit.from_kepler: its positions follow orbitize!'s; its
    sky velocity (mas/yr) equals the time derivative of orbitize!'s
    positions; its line-of-sight velocity equals orbitize!'s relative RV
    (calc_orbit with mass_for_Kamp = mtot, km/s) at t_ref, converted with
    the distance 1000/plx; mu = 4 pi² a³ / P² (P in Julian years); and
    to_kepler returns orbitize!'s elements with the node fixed absolutely
    (no 180° ambiguity: the state carries dz)."""
    worst_pos = worst_v = worst_vz = worst_el = worst_mu = 0.0
    sub = [g for g in grid if g[0]["ecc"] <= 0.6 and 1e-3 < g[0]["inc"]]
    for el, p, epochs, r, orbit in sub:
        a = el["sma"] * el["plx"]
        state = vo.StateVectorOrbit.from_kepler(orbit)
        dra, ddec, _ = state.relative(epochs)
        worst_pos = max(worst_pos, _sky_error(dra, ddec, r, a))
        h = p / 2000.0
        stencil = REF + h * np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        rr = ob.ephemeris([dict(el, mass_for_Kamp=el["mtot"])], stencil, REF)[0]
        ra, de = np.asarray(rr["raoff"]), np.asarray(rr["deoff"])
        d = lambda f: (-f[4] + 8 * f[3] - 8 * f[1] + f[0]) / (12 * h) * JULIAN_YEAR_D  # noqa: E731
        scale = 2 * np.pi * a / (p / JULIAN_YEAR_D)
        worst_v = max(worst_v, abs(float(state.vra) - d(ra)) / scale, abs(float(state.vdec) - d(de)) / scale)
        vz_kms = float(state.vz) * (1000.0 / el["plx"]) * AU_KM / 1000.0 / (JULIAN_YEAR_D * 86400.0)
        worst_vz = max(worst_vz, abs(vz_kms - rr["vz"][2]) / (scale * (1000.0 / el["plx"]) * AU_KM / 1000.0
                                                             / (JULIAN_YEAR_D * 86400.0)))
        worst_mu = max(worst_mu, abs(float(state.mu) / (4 * np.pi**2 * a**3 / (p / JULIAN_YEAR_D) ** 2) - 1))
        if el["ecc"] > 1e-3 and 1.0 < math.degrees(el["inc"]) < 179.0:
            k = state.to_kepler()
            d_ang = [((float(x) - math.degrees(y) + 180.0) % 360.0) - 180.0
                     for x, y in ((k.Omega, el["pan"]), (k.omega, el["aop"]), (k.inc, el["inc"]))]
            worst_el = max(worst_el, np.max(np.abs(d_ang)), abs(float(k.period) / p - 1),
                           abs(float(k.a_mas) / a - 1), abs(float(k.ecc) - el["ecc"]))
    record("max_rel_position", worst_pos)
    record("max_rel_sky_velocity", worst_v)
    record("max_rel_los_velocity", worst_vz)
    record("max_rel_mu", worst_mu)
    record("max_element_error", worst_el)
    assert worst_pos < TOL_POS and worst_vz < TOL_RV and worst_mu < 1e-13
    assert worst_v < 1e-6
    assert worst_el < 1e-8


# -------------------------------------------------------------- radial velocities

RV_ORBITS = [
    {"sma": 3.0, "ecc": 0.4, "inc": math.radians(60), "aop": math.radians(40), "pan": math.radians(110),
     "tau": 0.3, "plx": 40.0, "m0": 1.4, "m1": 0.6},
    {"sma": 12.0, "ecc": 0.9, "inc": math.radians(140), "aop": math.radians(250), "pan": math.radians(300),
     "tau": 0.8, "plx": 15.0, "m0": 2.5, "m1": 1.9},
    {"sma": 0.8, "ecc": 0.0, "inc": math.radians(89.5), "aop": 0.0, "pan": math.radians(20),
     "tau": 0.5, "plx": 120.0, "m0": 0.9, "m1": 0.05},
]


def _rv_case(el):
    mtot = el["m0"] + el["m1"]
    p = ob.periods([(el["sma"], mtot)])[0]
    epochs = REF + np.linspace(-0.2 * p, 1.3 * p, 80)
    std = {k: el[k] for k in ("sma", "ecc", "inc", "aop", "pan", "tau", "plx")} | {"mtot": mtot}
    orbit = vo.KeplerOrbit(**ob.to_virgil(std, p, REF))
    return std, p, epochs, orbit


def _virgil_rv(orbit, epochs, star, el, gamma):
    z = np.zeros_like(epochs)
    return np.asarray(vo.RVData(epochs, z, z + 1.0, star=star).model(orbit, el["m1"] / el["m0"], gamma, 1000.0 / el["plx"]))


@pytest.mark.validates("virgil.orbits.RVData", "virgil.orbits.KeplerOrbit", roots=["orbitize"])
@pytest.mark.parametrize("case", range(len(RV_ORBITS)))
def test_radial_velocities_match_orbitize(case):
    """RVData.model for the primary and the secondary against orbitize!'s
    calc_orbit: the companion's barycentric RV is calc_orbit with
    mass_for_Kamp = m0, and the primary's is minus calc_orbit with
    mass_for_Kamp = m1 (orbitize!'s manual: omega_* = omega_p + 180°; its
    System uses -m1/m0 times the companion's). q = m1/m0, distance 1000/plx;
    gamma adds to both."""
    el = RV_ORBITS[case]
    std, p, epochs, orbit = _rv_case(el)
    sec, pri = ob.ephemeris([dict(std, mass_for_Kamp=el["m0"]), dict(std, mass_for_Kamp=el["m1"])], epochs, REF)
    gamma = -3.7
    k = np.max(np.abs(sec["vz"])) + np.max(np.abs(pri["vz"]))
    d_sec = np.max(np.abs(_virgil_rv(orbit, epochs, "secondary", el, gamma) - gamma - np.asarray(sec["vz"]))) / k
    d_pri = np.max(np.abs(_virgil_rv(orbit, epochs, "primary", el, gamma) - gamma + np.asarray(pri["vz"]))) / k
    record("max_rel_rv_secondary", d_sec)
    record("max_rel_rv_primary", d_pri)
    assert d_sec < TOL_RV and d_pri < TOL_RV


@pytest.fixture(scope="module")
def system_rvs():
    """orbitize!'s System.compute_model for the first RV orbit, with RVs of
    both bodies from one instrument, and with companion RVs only."""
    el = RV_ORBITS[0]
    std, p, epochs, orbit = _rv_case(el)
    task = {"task": "system_rv", "epochs": epochs.tolist(), "tau_ref_epoch": REF, "orbit": el, "gamma": 7.0}
    both = ob.run({**task, "objects": [0, 1]})
    alone = ob.run({**task, "objects": [1]})
    return el, orbit, epochs, both, alone


@pytest.mark.validates("virgil.orbits.RVData", roots=["orbitize"])
def test_radial_velocities_match_orbitize_system(system_rvs):
    """Through orbitize!'s own System (fit_secondary_mass=True), as a fit
    would see it. With primary RVs present, orbitize! fits the instrument's
    gamma and adds it to every RV of that instrument, the companion's too:
    both equal virgil's model with the same gamma. With companion RVs alone
    it fits no gamma (the companion's RVs are barycentric, as its
    read_input documents), and they equal virgil's with gamma = 0."""
    el, orbit, epochs, both, alone = system_rvs
    k = np.max(np.abs(both["primary"])) + np.max(np.abs(both["secondary"]))
    d = [
        np.max(np.abs(np.asarray(both["primary"]) - _virgil_rv(orbit, epochs, "primary", el, 7.0))) / k,
        np.max(np.abs(np.asarray(both["secondary"]) - _virgil_rv(orbit, epochs, "secondary", el, 7.0))) / k,
        np.max(np.abs(np.asarray(alone["secondary"]) - _virgil_rv(orbit, epochs, "secondary", el, 0.0))) / k,
    ]
    record("max_rel_primary", d[0])
    record("max_rel_secondary_with_primary", d[1])
    record("max_rel_secondary_alone", d[2])
    assert "gamma_spec" in both["labels"] and "gamma_spec" not in alone["labels"]
    assert max(d) < TOL_SYSTEM


@pytest.mark.xfail(strict=True, reason="P7: orbitize! adds gamma to companion RVs when primary RVs are present")
@pytest.mark.validates("orbitize", roots=["mathematics"], kind="upstream")
def test_p7_companion_rvs_are_barycentric_as_documented(system_rvs):
    """orbitize!'s read_input: 'RV measurements of objects that are not the
    primary should be relative to the barycenter RV'. The model of such
    data should then not depend on whether primary RVs are also given.
    It does: with primary RVs present the companion's model gains the
    fitted gamma (here 7 km/s)."""
    _, _, _, both, alone = system_rvs
    d = np.max(np.abs(np.asarray(both["secondary"]) - np.asarray(alone["secondary"])))
    record("companion_model_shift_kms", d)
    assert d < 1e-6


# ------------------------------------------------------------------ symmetries


@pytest.mark.validates("virgil.orbits.KeplerOrbit", "virgil.orbits.RVData", roots=["orbitize", "mathematics"])
def test_symmetries_in_both_codes(grid):
    """The documented symmetries, in each code separately:

    1. (Omega + 180°, omega + 180°): the same sky positions; virgil's dz
       and both codes' radial velocities change sign.
    2. omega + 180° alone: r -> -r (the same as swapping the stars).
    3. i -> 180° - i: the sense of rotation reverses. With i < 90° the
       position angle increases (dPA/dt > 0) in both codes: virgil's from
       relative_velocity, orbitize!'s from finite differences of its
       positions, which also match virgil's rates.
    """
    worst = {"sym1_pos": 0.0, "sym1_dz": 0.0, "sym1_rv": 0.0, "sym2": 0.0, "rate": 0.0}
    sub = [g for g in grid if g[0]["ecc"] <= 0.6 and 0.01 < math.degrees(g[0]["inc"]) < 89.0]
    assert len(sub) >= 6
    for el, p, epochs, r, orbit in sub:
        a = el["sma"] * el["plx"]
        twin = dict(el, aop=el["aop"] + np.pi, pan=el["pan"] + np.pi)
        flip = dict(el, aop=el["aop"] + np.pi)
        mirror = dict(el, inc=np.pi - el["inc"])
        o_twin, o_flip, o_mirror = (vo.KeplerOrbit(**ob.to_virgil(x, p, REF)) for x in (twin, flip, mirror))
        h = p / 2000.0
        stencil = (epochs[::10][:, None] + h * np.array([-2.0, -1.0, 0.0, 1.0, 2.0])).ravel()
        rs = ob.ephemeris([el, twin, flip, mirror], stencil, REF)
        base = {k: np.asarray(rs[0][k]).reshape(-1, 5) for k in ("raoff", "deoff", "vz")}
        t = epochs[::10]
        # 1. in orbitize! and in virgil
        tw = {k: np.asarray(rs[1][k]).reshape(-1, 5) for k in ("raoff", "deoff", "vz")}
        worst["sym1_pos"] = max(worst["sym1_pos"], np.max(np.abs(tw["raoff"] - base["raoff"])) / a,
                                np.max(np.abs(tw["deoff"] - base["deoff"])) / a)
        kv = np.max(np.abs(base["vz"]))
        worst["sym1_rv"] = max(worst["sym1_rv"], np.max(np.abs(tw["vz"] + base["vz"])) / kv)
        x0, y0, z0 = (np.asarray(v) for v in orbit.relative(t))
        x1, y1, z1 = (np.asarray(v) for v in o_twin.relative(t))
        worst["sym1_pos"] = max(worst["sym1_pos"], np.max(np.abs(x1 - x0)) / a, np.max(np.abs(y1 - y0)) / a)
        worst["sym1_dz"] = max(worst["sym1_dz"], np.max(np.abs(z1 + z0)) / a)
        el_rv = {"m0": 1.0, "m1": 0.5, "plx": el["plx"]}
        rv0, rv1 = (_virgil_rv(o, t, "primary", el_rv, 0.0) for o in (orbit, o_twin))
        worst["sym1_rv"] = max(worst["sym1_rv"], np.max(np.abs(rv1 + rv0)) / np.max(np.abs(rv0)))
        # 2.
        fl = {k: np.asarray(rs[2][k]).reshape(-1, 5) for k in ("raoff", "deoff")}
        x2, y2, _ = (np.asarray(v) for v in o_flip.relative(t))
        worst["sym2"] = max(worst["sym2"], np.max(np.abs(fl["raoff"] + base["raoff"])) / a,
                            np.max(np.abs(fl["deoff"] + base["deoff"])) / a,
                            np.max(np.abs(x2 + x0)) / a, np.max(np.abs(y2 + y0)) / a)
        # 3. dPA/dt = (ddec * vra - dra * vdec) / sep²
        mi = {k: np.asarray(rs[3][k]).reshape(-1, 5) for k in ("raoff", "deoff")}
        for code_pos, o in ((base, orbit), (mi, o_mirror)):
            ra, de = code_pos["raoff"], code_pos["deoff"]
            vra = (-ra[:, 4] + 8 * ra[:, 3] - 8 * ra[:, 1] + ra[:, 0]) / (12 * h)
            vde = (-de[:, 4] + 8 * de[:, 3] - 8 * de[:, 1] + de[:, 0]) / (12 * h)
            rate_orbitize = (de[:, 2] * vra - ra[:, 2] * vde) / (ra[:, 2] ** 2 + de[:, 2] ** 2)
            xv, yv, _ = (np.asarray(v) for v in o.relative(t))
            vx, vy, _ = (np.asarray(v) for v in o.relative_velocity(t))
            rate_virgil = (yv * vx - xv * vy) / (xv**2 + yv**2)
            sign = 1.0 if o is orbit else -1.0
            assert np.all(sign * rate_orbitize > 0) and np.all(sign * rate_virgil > 0), (el, sign)
            worst["rate"] = max(worst["rate"], np.max(np.abs(rate_virgil - rate_orbitize)) / (2 * np.pi / p)
                                * np.min(np.hypot(xv, yv)) / a)
    for k, v in worst.items():
        record("max_rel_" + k, v)
    assert worst["sym1_pos"] < TOL_POS and worst["sym1_dz"] < TOL_POS and worst["sym2"] < TOL_POS
    assert worst["sym1_rv"] < TOL_RV
    assert worst["rate"] < 1e-6


# ------------------------------------------------------- derived mass and distance


def _mass_cases(grid):
    return [(el, orbit) for el, _, _, _, orbit in grid[::4]]


@pytest.mark.validates("virgil.orbits.total_mass", "virgil.orbits.distance_pc", roots=["orbitize", "standards"])
def test_total_mass_and_distance_against_orbitize(grid):
    """total_mass(orbit, 1000/plx) recovers orbitize!'s mtot, and
    distance_pc(orbit, mtot) its distance, to the 3.8e-5 (mass) and 1.3e-5
    (distance) of ledger F14: virgil computes the documented a³/P² with P in
    Julian years (pinned here to 1e-13), which equals the exact
    4 pi² a³ / (G M_sun P²) of orbitize! (and of the IAU constants) only
    times (365.25 d / Y)², with Y = 2 pi sqrt(au³ / GM_sun) = 365.2569 d."""
    year_gauss = 2 * np.pi * np.sqrt((AU_KM * 1e3) ** 3 / GM_SUN) / 86400.0
    factor = (JULIAN_YEAR_D / year_gauss) ** 2
    worst_doc = worst_mass = worst_dist = worst_round = 0.0
    for el, orbit in _mass_cases(grid):
        d = 1000.0 / el["plx"]
        m = float(vo.total_mass(orbit, d))
        doc = (float(orbit.a_mas) * d / 1000.0) ** 3 / (float(orbit.period) / JULIAN_YEAR_D) ** 2
        worst_doc = max(worst_doc, abs(m / doc - 1))
        worst_mass = max(worst_mass, abs(m / (el["mtot"] * factor) - 1))
        dist = float(vo.distance_pc(orbit, el["mtot"]))
        worst_dist = max(worst_dist, abs(dist / (d * factor ** (-1 / 3)) - 1))
        worst_round = max(worst_round, abs(float(vo.total_mass(orbit, dist)) / el["mtot"] - 1))
    record("mass_factor_minus_1", factor - 1)
    record("max_rel_vs_julian_year_formula", worst_doc)
    record("max_rel_vs_orbitize_times_factor", worst_mass)
    record("max_rel_distance_vs_orbitize_times_factor", worst_dist)
    record("max_rel_round_trip", worst_round)
    assert abs(factor - 1 + 3.777e-5) < 1e-8
    assert max(worst_doc, worst_mass, worst_dist, worst_round) < 1e-13


@pytest.mark.xfail(strict=True, reason="F14: total_mass uses a³/P² in Julian years, 3.8e-5 from Kepler's third law")
@pytest.mark.validates("virgil.orbits.total_mass", "virgil.orbits.distance_pc", roots=["orbitize", "standards"], kind="finding")
def test_f14_total_mass_is_keplers_third_law(grid):
    """The total mass and dynamical distance of Kepler's third law with the
    IAU nominal GM_sun and au, as orbitize! computes them."""
    worst = 0.0
    for el, orbit in _mass_cases(grid):
        worst = max(worst, abs(float(vo.total_mass(orbit, 1000.0 / el["plx"])) / el["mtot"] - 1),
                    abs(float(vo.distance_pc(orbit, el["mtot"])) * el["plx"] / 1000.0 - 1))
    record("max_rel_error", worst)
    assert worst < 1e-12


# ------------------------------------------------- posterior (campaign, not CI)

PRIORS = {
    # matched on both sides; invariant where the coordinator asked for it:
    "sma": (1.0, 100.0),  # au, log-uniform (log-uniform in P at fixed mtot)
    "ecc": ("uniform", 0.0, 1.0),  # stated explicitly: uniform on [0, 1)
    # inc uniform in cos i on [0°, 180°]; omega, Omega uniform on [0°, 360°);
    # tau (the periastron time, as a fraction of P) uniform on [0, 1)
    "plx": (51.44, 0.12),  # mas, Gaussian (orbitize!'s beta Pic tutorials)
    "mtot": (1.75, 0.05),  # M_sun, Gaussian
}


def _beta_pic_positions(path):
    """The sep/PA rows of orbitize!'s betaPic.csv (Nielsen et al. 2020,
    Table 1), without its single companion RV, so that both codes see
    positions only (and Omega is known modulo 180°)."""
    rows = [line.split(",") for line in open(path).read().splitlines() if line and not line.startswith("#")]
    head, rows = rows[0], rows[1:]
    col = {k: head.index(k) for k in ("epoch", "sep", "sep_err", "pa", "pa_err")}
    keep = [r for r in rows if r[col["sep"]].strip()]
    return {k: np.array([float(r[c]) for r in keep]) for k, c in col.items()}, head, keep


def _fold(omega, Omega):
    """Positions fix Omega modulo 180°: report (Omega, omega) with
    Omega in [0°, 180°), moving omega with it."""
    flip = Omega % 360.0 >= 180.0
    return (omega - 180.0 * flip) % 360.0, Omega % 180.0


@pytest.mark.slow
@pytest.mark.validates("pipeline:orbitize-posterior", "virgil.orbits.PositionData", "virgil.orbits.KeplerOrbit",
                       roots=["orbitize"], tier="C")
def test_beta_pic_posterior_matches_orbitize(tmp_path):
    """Posteriors of beta Pic b's orbit from orbitize!'s example data, with
    matched priors (PRIORS): virgil (PositionData.from_sep_pa likelihood,
    NumPyro NUTS on the same variables: log sma, ecc, cos i, omega, Omega,
    tau, plx, mtot) against orbitize! (parallel-tempered MCMC, ptemcee).
    The 16th, 50th and 84th percentiles of a_mas, e, i, Omega and omega
    (folded to Omega < 180°) and P must agree within 0.15 of the posterior
    standard deviation. Heavy (hours): for OzSTAR, not CI.

    Definition difference: orbitize!'s sep/PA likelihood treats sep and PA
    as independent Gaussians; from_sep_pa's documented "independent errors
    on each" is used on virgil's side, so any linearisation there shows up
    as a small shift at the widest PA errors."""
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    src = pathlib_example("betaPic.csv")
    pos, head, keep = _beta_pic_positions(src)
    csv = tmp_path / "betaPic_positions.csv"
    csv.write_text(",".join(head) + "\n" + "\n".join(",".join(r) for r in keep) + "\n")

    r = ob.run({
        "task": "posterior", "path": str(csv), "algorithm": "MCMC", "tau_ref_epoch": REF,
        "mtot": PRIORS["mtot"][0], "mtot_err": PRIORS["mtot"][1],
        "plx": PRIORS["plx"][0], "plx_err": PRIORS["plx"][1], "priors": PRIORS,
        "mcmc_kwargs": {"num_temps": 20, "num_walkers": 1000, "num_threads": 1},
        "n": 2_000_000, "burn": 5000, "thin": 10,
    }, timeout=48 * 3600)
    ours = {k: np.asarray(v) for k, v in r.items() if isinstance(v, list)}

    data = vo.PositionData.from_sep_pa(pos["epoch"], pos["sep"], pos["pa"], pos["sep_err"], pos["pa_err"], t_ref=REF)
    gm = GM_SUN  # orbitize!'s period: P = 2 pi sqrt(a³ / (G M))

    def model():
        log_sma = numpyro.sample("log_sma", dist.Uniform(*np.log(PRIORS["sma"])))
        ecc = numpyro.sample("ecc", dist.Uniform(*PRIORS["ecc"][1:]))
        cosi = numpyro.sample("cosi", dist.Uniform(-1.0, 1.0))
        omega = numpyro.sample("omega", dist.Uniform(0.0, 360.0))
        Omega = numpyro.sample("Omega", dist.Uniform(0.0, 360.0))
        tau = numpyro.sample("tau", dist.Uniform(0.0, 1.0))
        plx = numpyro.sample("plx", dist.Normal(*PRIORS["plx"]))
        mtot = numpyro.sample("mtot", dist.Normal(*PRIORS["mtot"]))
        sma = jnp.exp(log_sma)
        period = 2 * jnp.pi * jnp.sqrt((sma * AU_KM * 1e3) ** 3 / (gm * mtot)) / 86400.0
        orbit = vo.KeplerOrbit(period, tau * period, ecc, jnp.degrees(jnp.arccos(cosi)), omega, Omega,
                               sma * plx, t_ref=REF)
        numpyro.deterministic("period", period)
        numpyro.factor("positions", data.loglike(orbit))

    mcmc = MCMC(NUTS(model), num_warmup=2000, num_samples=5000, num_chains=4, chain_method="sequential")
    mcmc.run(jax.random.PRNGKey(2020))
    s = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    virgil_post = {"a_mas": np.exp(s["log_sma"]) * s["plx"], "e": s["ecc"], "i": np.degrees(np.arccos(s["cosi"])),
                   "P": s["period"]}
    virgil_post["omega"], virgil_post["Omega"] = _fold(s["omega"], s["Omega"])
    orb_period = 2 * np.pi * np.sqrt((ours["sma"] * AU_KM * 1e3) ** 3 / (gm * ours["mtot"])) / 86400.0
    orbitize_post = {"a_mas": ours["sma"] * ours["plx"], "e": ours["ecc"], "i": np.degrees(ours["inc"]), "P": orb_period}
    orbitize_post["omega"], orbitize_post["Omega"] = _fold(np.degrees(ours["aop"]), np.degrees(ours["pan"]))
    worst = 0.0
    for k in virgil_post:
        sd = np.std(orbitize_post[k])
        dq = np.abs(np.percentile(virgil_post[k], [16, 50, 84]) - np.percentile(orbitize_post[k], [16, 50, 84])) / sd
        record(f"max_dquantile_over_sd_{k}", np.max(dq))
        worst = max(worst, np.max(dq))
    assert worst < 0.15


def pathlib_example(name):
    """A file from orbitize!'s example_data, in its own environment."""
    import json
    import subprocess

    from external_bridge import _subprocess as sp

    code = "import orbitize, os, json; print(json.dumps(os.path.join(os.path.dirname(orbitize.__file__), 'example_data')))"
    out = subprocess.run([sp.python("orbitize"), "-W", "ignore", "-c", code], capture_output=True, text=True, check=True)
    return f"{json.loads(out.stdout.strip().splitlines()[-1])}/{name}"
