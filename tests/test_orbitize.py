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
few orbits carry ~1e-15 of rounding; the margin of ~100 over the 1e-14
observed covers libm differences between platforms. Radial velocities agree
to 1e-12 of the semi-amplitude for the same reason; through orbitize!'s
System, which solves Kepler's equation at its default 1e-9, to 1e-8.
Finite-difference velocities (five-point stencil, h = P/2000, on orbits
with e <= 0.6) agree to 1e-6 of 2 pi a / P (truncation error ~1e-10).

orbitize! takes ~10 s to import, so every orbitize! number the fast tests
use comes from one worker process (the ``runs`` fixture). Its epochs are
chosen from our own Kepler's-law period, which equals orbitize!'s to 1e-14
(test_orbitize_period_is_keplers_third_law); virgil's elements use
orbitize!'s returned period. virgil is evaluated jit-compiled and vmapped
over orbits (eager JAX costs ~0.5 s per orbit and call).

Findings (ledger), fixed in virgil#229 and now checked as ordinary tests:
F14, total_mass used a³/P² with P in Julian years, 3.8e-5 from Kepler's
third law; F15, ThieleInnesOrbit.to_kepler could return Omega = 180.0; F16,
StateVectorOrbit.to_kepler lost the inclination of nearly face-on orbits
(i = 0.01° came back as 0.0106°). Open: P7, orbitize!
adds the instrument's gamma to companion RVs when primary RVs are present,
although its documentation asks for companion RVs relative to the
barycentre.
"""

import math
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy import constants

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
N_EPOCHS = 121
STENCIL = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])


def kepler_period(sma, mtot):
    """P (days) = 2 pi sqrt(a³ / (G M)), IAU constants."""
    return 2 * np.pi * np.sqrt((np.asarray(sma) * AU_KM * 1e3) ** 3 / (GM_SUN * np.asarray(mtot))) / 86400.0


def _grid():
    """Element sets (orbitize!'s standard basis, radians) over e, i, omega,
    Omega and phase, with the edge cases: e = 0 and ~0, e up to 0.95,
    i = 0, ~0, ~90, 90 and above 90; and cardinal angles."""
    rng = np.random.default_rng(14)
    out = []
    for e in (0.0, 1e-7, 0.1, 0.5, 0.8, 0.95):
        for i in (0.0, 1e-4, 35.0, 89.99, 90.0, 120.0, 179.9):
            out.append({
                "sma": float(rng.uniform(1, 40)), "ecc": e, "inc": math.radians(i),
                "aop": float(rng.uniform(0, 2 * np.pi)), "pan": float(rng.uniform(0, 2 * np.pi)),
                "tau": float(rng.uniform()), "plx": float(rng.uniform(5, 100)), "mtot": float(rng.uniform(0.3, 5)),
            })
    for aop, pan in ((0, 0), (90, 0), (0, 90), (180, 270), (270, 180)):
        out.append({"sma": 5.0, "ecc": 0.3, "inc": math.radians(60), "aop": math.radians(aop),
                    "pan": math.radians(pan), "tau": 0.0, "plx": 20.0, "mtot": 1.0})
    return out


GRID = _grid()
EPOCHS = np.stack([REF - 0.3 * p + np.linspace(0, 1.6 * p, N_EPOCHS)
                   for p in kepler_period([e["sma"] for e in GRID], [e["mtot"] for e in GRID])])

# radial velocities: masses m0 (primary) and m1 (companion)
RV_ORBITS = [
    {"sma": 3.0, "ecc": 0.4, "inc": math.radians(60), "aop": math.radians(40), "pan": math.radians(110),
     "tau": 0.3, "plx": 40.0, "m0": 1.4, "m1": 0.6},
    {"sma": 12.0, "ecc": 0.9, "inc": math.radians(140), "aop": math.radians(250), "pan": math.radians(300),
     "tau": 0.8, "plx": 15.0, "m0": 2.5, "m1": 1.9},
    {"sma": 0.8, "ecc": 0.0, "inc": math.radians(89.5), "aop": 0.0, "pan": math.radians(20),
     "tau": 0.5, "plx": 120.0, "m0": 0.9, "m1": 0.05},
]
for _el in RV_ORBITS:
    _el["mtot"] = _el["m0"] + _el["m1"]
RV_EPOCHS = [REF + np.linspace(-0.2, 1.3, 80) * kepler_period(e["sma"], e["mtot"]) for e in RV_ORBITS]
RV_GAMMA, SYSTEM_GAMMA = -3.7, 7.0

# finite-difference stencils for velocities
SV_USE = [j for j, e in enumerate(GRID) if e["ecc"] <= 0.6]
SYM_USE = [j for j, e in enumerate(GRID) if e["ecc"] <= 0.6 and 0.01 < math.degrees(e["inc"]) < 89.0]
VARIANTS = {
    "base": lambda e: e,
    "twin": lambda e: dict(e, aop=e["aop"] + np.pi, pan=e["pan"] + np.pi),  # (Omega, omega) + 180°
    "flip": lambda e: dict(e, aop=e["aop"] + np.pi),  # omega + 180°
    "mirror": lambda e: dict(e, inc=np.pi - e["inc"]),  # i -> 180° - i
}


def _h(j):
    return kepler_period(GRID[j]["sma"], GRID[j]["mtot"]) / 2000.0


def _sym_epochs(j):
    return (EPOCHS[j, ::10][:, None] + _h(j) * STENCIL).ravel()  # 13 epochs x 5


def _std(el):
    return {k: el[k] for k in ("sma", "ecc", "inc", "aop", "pan", "tau", "plx", "mtot")}


def _at(orbits, epochs):
    """Orbit j at its own epochs[j] in one ephemeris task (the worker takes
    one epoch list: concatenate, and slice the result with _slice)."""
    return ob.ephemeris_task(orbits, np.concatenate(epochs), REF)


def _slice(result, n):
    return {k: np.stack([np.asarray(r[k])[j * n:(j + 1) * n] for j, r in enumerate(result["results"])])
            for k in ("raoff", "deoff", "vz", "sep", "pa")}


@pytest.fixture(scope="module")
def runs():
    """Every orbitize! number the fast tests use, from one worker process."""
    tasks = {
        "period_check": {"task": "period", "cases": [[1.0, 1.0], [10.0, 1.7], [0.05, 0.3], [300.0, 20.0]]},
        "periods": {"task": "period", "cases": [[e["sma"], e["mtot"]] for e in GRID + RV_ORBITS]},
        "grid": _at(GRID, list(EPOCHS)),
        "state": _at([dict(GRID[j], mass_for_Kamp=GRID[j]["mtot"]) for j in SV_USE],
                     [REF + _h(j) * STENCIL for j in SV_USE]),
        **{f"sym_{name}": _at([change(GRID[j]) for j in SYM_USE], [_sym_epochs(j) for j in SYM_USE])
           for name, change in VARIANTS.items()},
        **{f"rv_{k}": _at([dict(_std(e), mass_for_Kamp=e["m0"]), dict(_std(e), mass_for_Kamp=e["m1"])], [t, t])
           for k, (e, t) in enumerate(zip(RV_ORBITS, RV_EPOCHS))},
        **{f"system_{name}": {"task": "system_rv", "epochs": RV_EPOCHS[0].tolist(), "tau_ref_epoch": REF,
                              "orbit": RV_ORBITS[0], "gamma": SYSTEM_GAMMA, "objects": objects}
           for name, objects in (("both", [0, 1]), ("alone", [1]))},
    }
    out = ob.batch(tasks)
    out["periods"] = np.array([r["period_day"] for r in out["periods"]["results"]])
    out["grid"] = _slice(out["grid"], N_EPOCHS)
    out["state"] = _slice(out["state"], 5)
    for name in VARIANTS:
        out[f"sym_{name}"] = {k: v.reshape(len(SYM_USE), -1, 5)
                              for k, v in _slice(out[f"sym_{name}"], 13 * 5).items()}
    for k in range(len(RV_ORBITS)):
        out[f"rv_{k}"] = _slice(out[f"rv_{k}"], len(RV_EPOCHS[k]))
    return out


# ---------------------------------------------------------------- virgil side


def _params(el, p):
    """virgil KeplerOrbit arguments (without t_ref), as a list."""
    v = ob.to_virgil(el, p, REF)
    return [v[k] for k in ("period", "dt_peri", "ecc", "inc", "omega", "Omega", "a_mas")]


def _evaluate(prm, t):
    """Everything the tests read from virgil for one orbit."""
    o = vo.KeplerOrbit(*prm, t_ref=REF)
    s = vo.StateVectorOrbit.from_kepler(o)
    k = s.to_kepler()
    return {
        "rel": jnp.stack(o.relative(t)), "seppa": jnp.stack(o.separation_pa(t)),
        "vel": jnp.stack(o.relative_velocity(t)), "ti": jnp.stack(o.thiele_innes()),
        "sv_rel": jnp.stack(s.relative(t)), "sv": jnp.stack([s.vra, s.vdec, s.vz, s.mu]),
        "sv_kepler": jnp.stack([k.period, k.dt_peri, k.ecc, k.inc, k.omega, k.Omega, k.a_mas]),
    }


def _evaluate_ti(prm, abfg, t):
    """A ThieleInnesOrbit from given constants: its sky positions and its
    to_kepler elements."""
    ti = vo.ThieleInnesOrbit(prm[0], prm[1], prm[2], *abfg, t_ref=REF)
    k = ti.to_kepler()
    return {"sky": jnp.stack(ti.sky(t)),
            "kepler": jnp.stack([k.period, k.dt_peri, k.ecc, k.inc, k.omega, k.Omega, k.a_mas])}


EVALUATE = jax.jit(jax.vmap(_evaluate))
EVALUATE_TI = jax.jit(jax.vmap(_evaluate_ti))


@eqx.filter_jit
def _rv_model(data, prm, q, gamma, distance_pc):
    return data.model(vo.KeplerOrbit(*prm, t_ref=REF), q, gamma, distance_pc)


def virgil_rv(prm, epochs, star, q, distance_pc, gamma):
    """RVData.model (km/s) for the KeplerOrbit with arguments ``prm``."""
    z = np.zeros_like(epochs)
    data = vo.RVData(epochs, z, z + 1.0, star=star, t_ref=REF)
    return np.asarray(_rv_model(data, jnp.asarray(prm), q, gamma, distance_pc))


@pytest.fixture(scope="module")
def grid(runs):
    """The element grid: orbitize!'s periods and ephemerides over 1.6
    periods of each orbit (121 epochs), and virgil's at the mapped elements."""
    periods = runs["periods"][:len(GRID)]
    prm = np.array([_params(el, p) for el, p in zip(GRID, periods)])
    return {
        "els": GRID, "periods": periods, "epochs": EPOCHS, "prm": prm,
        "a": np.array([el["sma"] * el["plx"] for el in GRID]),
        "orbitize": runs["grid"],
        "virgil": {k: np.asarray(v) for k, v in EVALUATE(prm, EPOCHS).items()},
    }


def _sky_error(dra, ddec, ra, dec, a):
    """Largest position difference over the semimajor axis."""
    a = np.reshape(a, (-1,) + (1,) * (np.ndim(dra) - 1))
    return np.max(np.maximum(np.abs(dra - ra), np.abs(ddec - dec)) / a)


def _dangle(x, y):
    """x - y wrapped to [-180, 180) degrees."""
    return ((np.asarray(x) - y + 180.0) % 360.0) - 180.0


def _derivative(f, h):
    """Five-point central difference from samples at t + h * STENCIL."""
    return (-f[..., 4] + 8 * f[..., 3] - 8 * f[..., 1] + f[..., 0]) / (12 * h)


# ------------------------------------------------------------ orbitize! itself


@pytest.mark.validates("orbitize", "external_bridge.orbitize_bridge", roots=["standards", "mathematics"], kind="reference")
def test_orbitize_period_is_keplers_third_law(runs):
    """orbitize!'s period of (sma, mtot) is P = 2 pi sqrt(a³ / (G M)) with
    the IAU nominal GM_sun (astropy's G * M_sun) and the IAU au; its year is
    the Julian year (astropy's); its Period basis inverts it. So the
    mapping's P is exact, and our epochs (from kepler_period) are
    orbitize!'s."""
    r = runs["period_check"]
    assert r["year_days"] == JULIAN_YEAR_D
    assert abs(float((constants.G * constants.M_sun).to_value("m3 s-2")) / GM_SUN - 1) < 1e-15
    cases = [(1.0, 1.0), (10.0, 1.7), (0.05, 0.3), (300.0, 20.0)] + [(e["sma"], e["mtot"]) for e in GRID]
    results = r["results"] + [{"period_day": p, "sma_back": e["sma"]} for e, p in zip(GRID, runs["periods"])]
    worst = max(max(abs(res["period_day"] / kepler_period(*c) - 1), abs(res["sma_back"] / c[0] - 1))
                for c, res in zip(cases, results))
    record("max_rel_period", worst)
    assert worst < 1e-14


# ----------------------------------------------------------------- ephemerides


@pytest.mark.validates("virgil.orbits.KeplerOrbit", roots=["orbitize"])
def test_relative_astrometry_matches_orbitize(grid):
    """KeplerOrbit.relative (dra, ddec) and separation_pa against
    orbitize!'s calc_orbit and radec2seppa over the element grid (47 orbits,
    121 epochs each over 1.6 periods)."""
    o, v, a = grid["orbitize"], grid["virgil"], grid["a"][:, None]
    pos = _sky_error(v["rel"][:, 0], v["rel"][:, 1], o["raoff"], o["deoff"], a)
    sep = np.max(np.abs(v["seppa"][:, 0] - o["sep"]) / a)
    # an angle error times the separation: a length, well defined at sep -> 0
    pa = np.max(np.abs(np.radians(_dangle(v["seppa"][:, 1], o["pa"]))) * v["seppa"][:, 0] / a)
    record("max_rel_position", pos)
    record("max_rel_separation", sep)
    record("max_rel_pa_times_sep", pa)
    assert pos < TOL_POS and sep < TOL_POS and pa < TOL_POS


@pytest.mark.validates("virgil.orbits.KeplerOrbit", "external_bridge.orbitize_bridge", roots=["orbitize"], kind="control")
def test_wrong_mappings_fail(grid):
    """The mappings a reader could plausibly get wrong are wrong by the size
    of the orbit: the primary's argument of periastron (omega + 180°, as in
    jaxoplanet and spectroscopy), a node counted counterclockwise from East
    (90° - Omega), and the opposite sense of rotation (180° - inc)."""
    j = 30  # e = 0.8, i = 35°
    base = grid["prm"][j]
    wrong = np.array([base, base, base])
    wrong[0, 4] += 180.0
    wrong[1, 5] = (90.0 - base[5]) % 360.0
    wrong[2, 3] = 180.0 - base[3]
    rel = np.asarray(EVALUATE(wrong, np.stack([EPOCHS[j]] * 3))["rel"])
    o = grid["orbitize"]
    for k, name in enumerate(("omega", "Omega", "inc")):
        err = _sky_error(rel[k, 0], rel[k, 1], o["raoff"][j], o["deoff"][j], grid["a"][j])
        record("min_rel_error_" + name, err)
        assert err > 0.1, name


def _design_note_constants(grid):
    """A, B, F, G from virgil's design note §2.4, written out here."""
    a = grid["a"]
    w, W, i = (np.array([e[k] for e in grid["els"]]) for k in ("aop", "pan", "inc"))
    return np.stack([
        a * (np.cos(w) * np.cos(W) - np.sin(w) * np.sin(W) * np.cos(i)),
        a * (np.cos(w) * np.sin(W) + np.sin(w) * np.cos(W) * np.cos(i)),
        a * (-np.sin(w) * np.cos(W) - np.cos(w) * np.sin(W) * np.cos(i)),
        a * (-np.sin(w) * np.sin(W) + np.cos(w) * np.cos(W) * np.cos(i)),
    ], axis=1)


def _well_posed(el):
    """Orbits whose elements positions determine (not circular, not face-on)."""
    return el["ecc"] > 1e-3 and 1.0 < math.degrees(el["inc"]) < 179.0


@pytest.mark.validates("virgil.orbits.ThieleInnesOrbit", roots=["orbitize", "mathematics"])
def test_thiele_innes_orbit_matches_orbitize(grid):
    """A ThieleInnesOrbit built from the constants of virgil's design note
    §2.4 (A = a(cos w cos W - sin w sin W cos i), ...) reproduces orbitize!'s
    positions; KeplerOrbit.thiele_innes gives those constants; and
    to_kepler returns orbitize!'s elements or their documented twin
    (Omega + 180°, omega + 180°)."""
    abfg = _design_note_constants(grid)
    a = grid["a"]
    out = {k: np.asarray(v) for k, v in EVALUATE_TI(grid["prm"], abfg, EPOCHS).items()}
    o = grid["orbitize"]
    pos = _sky_error(out["sky"][:, 0], out["sky"][:, 1], o["raoff"], o["deoff"], a[:, None])
    const = np.max(np.abs(grid["virgil"]["ti"][:, :4] - abfg) / a[:, None])
    ok = np.array([_well_posed(e) for e in grid["els"]])
    k, want = out["kepler"][ok], grid["prm"][ok]
    # either solution of the documented pair (the range of Omega: F15)
    d_node = np.minimum(*[np.maximum(np.abs(_dangle(k[:, 5], want[:, 5] + s)), np.abs(_dangle(k[:, 4], want[:, 4] + s)))
                          for s in (0.0, 180.0)])
    el_err = max(np.max(d_node), np.max(np.abs(_dangle(k[:, 3], want[:, 3]))), np.max(np.abs(k[:, 6] / want[:, 6] - 1)))
    record("max_rel_position", pos)
    record("max_rel_constants", const)
    record("max_element_error", el_err)
    assert pos < TOL_POS and const < 1e-14
    assert el_err < 1e-9  # degrees (and relative a): atan2 of the constants


@pytest.mark.validates("virgil.orbits.ThieleInnesOrbit", roots=["mathematics"])
def test_f15_thiele_innes_node_in_documented_range(grid):
    """to_kepler documents 0 <= Omega < 180. For a node at 180°, whose
    constants carry sin(180°) ~ 1e-16, it returns 180.0 exactly:
    KeplerOrbit(1000, 0, 0.3, 60, 270, 180, 100).to_thiele_innes().to_kepler()
    has Omega == 180.0 (the same sky orbit as Omega = 0, omega = 90, which
    the test above accepts)."""
    omegas = np.asarray(EVALUATE_TI(grid["prm"], grid["virgil"]["ti"][:, :4], EPOCHS)["kepler"])[:, 5]
    record("max_Omega", np.max(omegas))
    assert np.all((omegas >= 0.0) & (omegas < 180.0))


def _state_errors(grid):
    """Position and inclination errors of StateVectorOrbit.from_kepler's
    round trip, per orbit."""
    v, o = grid["virgil"], grid["orbitize"]
    pos = np.max(np.maximum(np.abs(v["sv_rel"][:, 0] - o["raoff"]), np.abs(v["sv_rel"][:, 1] - o["deoff"])), axis=1)
    inc = np.abs(_dangle(v["sv_kepler"][:, 3], grid["prm"][:, 3]))
    return pos / grid["a"], inc


@pytest.mark.validates("virgil.orbits.StateVectorOrbit", roots=["orbitize"])
def test_state_vector_orbit_matches_orbitize(grid, runs):
    """StateVectorOrbit.from_kepler: its sky velocity (mas/yr) at t_ref
    equals the time derivative of orbitize!'s positions; its line-of-sight
    velocity equals orbitize!'s relative RV (calc_orbit with mass_for_Kamp
    = mtot, km/s) converted with the distance 1000/plx and the Julian year;
    mu = 4 pi² a³ / P² (P in Julian years); its positions follow orbitize!'s
    (to 1e-12 of a, near face-on and circular orbits included since
    virgil#229 fixed F16); and to_kepler returns orbitize!'s elements with the node fixed
    absolutely (the state carries dz), for well-posed orbits."""
    p, a, v = grid["periods"], grid["a"], grid["virgil"]
    pos, inc = _state_errors(grid)
    well = np.array([_well_posed(e) and 5.0 < math.degrees(e["inc"]) < 175.0 for e in GRID])
    rr, h = runs["state"], np.array([_h(j) for j in SV_USE])
    sv, A, P = v["sv"][SV_USE], a[SV_USE], p[SV_USE]
    year = JULIAN_YEAR_D
    scale = 2 * np.pi * A / (P / year)  # mas/yr
    vel = max(np.max(np.abs(sv[:, 0] - _derivative(rr["raoff"], h) * year) / scale),
              np.max(np.abs(sv[:, 1] - _derivative(rr["deoff"], h) * year) / scale))
    kms = (1000.0 / np.array([GRID[j]["plx"] for j in SV_USE])) * AU_KM / 1000.0 / (year * 86400.0)
    vz = np.max(np.abs(sv[:, 2] * kms - rr["vz"][:, 2]) / (scale * kms))
    mu = np.max(np.abs(v["sv"][:, 3] / (4 * np.pi**2 * a**3 / (p / year) ** 2) - 1))
    k, want = v["sv_kepler"][well], grid["prm"][well]
    el_err = max(np.max(np.abs(_dangle(k[:, 3:6], want[:, 3:6]))), np.max(np.abs(k[:, 0] / want[:, 0] - 1)),
                 np.max(np.abs(k[:, 6] / want[:, 6] - 1)), np.max(np.abs(k[:, 2] - want[:, 2])),
                 np.max(np.abs(_dangle(360 * k[:, 1] / want[:, 0], 360 * want[:, 1] / want[:, 0]))))
    record("max_rel_position", np.max(pos))
    record("max_rel_position_well_posed", np.max(pos[well]))
    record("max_rel_sky_velocity", vel)
    record("max_rel_los_velocity", vz)
    record("max_rel_mu", mu)
    record("max_element_error_well_posed", el_err)
    assert vz < TOL_RV and mu < 1e-13 and vel < 1e-6
    assert np.max(pos[well]) < TOL_POS and el_err < 1e-9
    assert np.max(pos) < 1e-12  # face-on and circular too, since F16's fix


@pytest.mark.validates("virgil.orbits.StateVectorOrbit", roots=["orbitize", "mathematics"])
def test_f16_state_vector_round_trip_keeps_float64_precision(grid):
    """The state (dra, ddec, vra, vdec, dz, vz, mu) fixes the orbit as
    precisely face-on as edge-on (the inclination is atan2(|h_xy|, h_z) of
    the angular momentum, good to ~1e-16 rad). virgil's to_kepler returns
    i = 0° for 0.001°, 0.0106° for 0.01° and 179.9894° for 179.99°, and
    the positions then drift by up to 2e-8 of a. Reproducer:
    StateVectorOrbit.from_kepler(KeplerOrbit(1000, 100, 0.3, 0.01, 30, 60, 100)).to_kepler().inc
    is 0.010646, not 0.01."""
    pos, inc = _state_errors(grid)
    o = vo.StateVectorOrbit.from_kepler(vo.KeplerOrbit(1000.0, 100.0, 0.3, 0.01, 30.0, 60.0, 100.0)).to_kepler()
    record("max_rel_position", np.max(pos))
    record("inc_back_for_0.01deg", float(o.inc))
    assert np.max(pos) < TOL_POS and abs(float(o.inc) - 0.01) < 1e-10


# -------------------------------------------------------------- radial velocities


@pytest.mark.validates("virgil.orbits.RVData", "virgil.orbits.KeplerOrbit", roots=["orbitize"])
@pytest.mark.parametrize("case", range(len(RV_ORBITS)))
def test_radial_velocities_match_orbitize(runs, case):
    """RVData.model for the primary and the secondary against orbitize!'s
    calc_orbit: the companion's barycentric RV is calc_orbit with
    mass_for_Kamp = m0, and the primary's is minus calc_orbit with
    mass_for_Kamp = m1 (orbitize!'s manual: omega_* = omega_p + 180°; its
    System uses -m1/m0 times the companion's). q = m1/m0, distance 1000/plx;
    gamma adds to both."""
    el, t = RV_ORBITS[case], RV_EPOCHS[case]
    prm = _params(el, runs["periods"][len(GRID) + case])
    sec, pri = runs[f"rv_{case}"]["vz"]
    q, d = el["m1"] / el["m0"], 1000.0 / el["plx"]
    k = np.max(np.abs(sec)) + np.max(np.abs(pri))
    d_sec = np.max(np.abs(virgil_rv(prm, t, "secondary", q, d, RV_GAMMA) - RV_GAMMA - sec)) / k
    d_pri = np.max(np.abs(virgil_rv(prm, t, "primary", q, d, RV_GAMMA) - RV_GAMMA + pri)) / k
    record("max_rel_rv_secondary", d_sec)
    record("max_rel_rv_primary", d_pri)
    assert d_sec < TOL_RV and d_pri < TOL_RV


@pytest.mark.validates("virgil.orbits.RVData", roots=["orbitize"])
def test_radial_velocities_match_orbitize_system(runs):
    """Through orbitize!'s own System (fit_secondary_mass=True), as a fit
    would see it. With primary RVs present, orbitize! fits the instrument's
    gamma and adds it to every RV of that instrument, the companion's too:
    both equal virgil's model with the same gamma. With companion RVs alone
    it fits no gamma (companion RVs barycentric, as its read_input
    documents), and they equal virgil's with gamma = 0."""
    el, t = RV_ORBITS[0], RV_EPOCHS[0]
    prm = _params(el, runs["periods"][len(GRID)])
    both, alone = runs["system_both"], runs["system_alone"]
    q, d = el["m1"] / el["m0"], 1000.0 / el["plx"]
    k = np.max(np.abs(both["primary"])) + np.max(np.abs(both["secondary"]))
    diffs = [
        np.max(np.abs(np.asarray(both["primary"]) - virgil_rv(prm, t, "primary", q, d, SYSTEM_GAMMA))) / k,
        np.max(np.abs(np.asarray(both["secondary"]) - virgil_rv(prm, t, "secondary", q, d, SYSTEM_GAMMA))) / k,
        np.max(np.abs(np.asarray(alone["secondary"]) - virgil_rv(prm, t, "secondary", q, d, 0.0))) / k,
    ]
    record("max_rel_primary", diffs[0])
    record("max_rel_secondary_with_primary", diffs[1])
    record("max_rel_secondary_alone", diffs[2])
    assert "gamma_spec" in both["labels"] and "gamma_spec" not in alone["labels"]
    assert max(diffs) < TOL_SYSTEM


@pytest.mark.xfail(strict=True, reason="P7: orbitize! adds gamma to companion RVs when primary RVs are present")
@pytest.mark.validates("orbitize", roots=["mathematics"], kind="upstream")
def test_p7_companion_rvs_are_barycentric_as_documented(runs):
    """orbitize!'s read_input asks for the RVs of non-primary bodies
    relative to the barycentre's RV. The model of such data should then not
    depend on whether primary RVs are also given. It does: with primary RVs
    present the companion's model gains the fitted gamma (here 7 km/s)."""
    shift = np.max(np.abs(np.asarray(runs["system_both"]["secondary"]) - np.asarray(runs["system_alone"]["secondary"])))
    record("companion_model_shift_kms", shift)
    assert shift < 1e-6


# ------------------------------------------------------------------ symmetries


@pytest.mark.validates("virgil.orbits.KeplerOrbit", "virgil.orbits.RVData", roots=["orbitize", "mathematics"])
def test_symmetries_in_both_codes(grid, runs):
    """The documented symmetries, in each code separately, on the grid's
    orbits with e <= 0.6 and 0.01° < i < 89°:

    1. (Omega + 180°, omega + 180°): the same sky positions; virgil's dz
       and both codes' radial velocities change sign.
    2. omega + 180° alone: r -> -r (the same as swapping the stars).
    3. i -> 180° - i: the sense of rotation reverses. With i < 90° the
       position angle increases (dPA/dt > 0) in both codes: virgil's rate
       from relative_velocity, orbitize!'s from finite differences of its
       positions; and the two rates agree.
    """
    assert len(SYM_USE) >= 6
    p, a = grid["periods"][SYM_USE], grid["a"][SYM_USE]
    t = EPOCHS[SYM_USE][:, ::10]
    h = np.array([_h(j) for j in SYM_USE])[:, None]
    orb, vir = {}, {}
    for name, change in VARIANTS.items():
        els = [change(GRID[j]) for j in SYM_USE]
        prm = np.array([_params(e, pj) for e, pj in zip(els, p)])
        orb[name] = runs[f"sym_{name}"]
        vir[name] = {k: np.asarray(x) for k, x in EVALUATE(prm, t).items()}
        vir[name]["rv"] = np.array([virgil_rv(q, tt, "primary", 0.5, 1000.0 / GRID[j]["plx"], 0.0)
                                    for q, tt, j in zip(prm, t, SYM_USE)])
    A = a[:, None, None]
    worst = {}
    # 1. orbitize! and virgil, each against itself
    worst["sym1_pos"] = max(
        np.max(np.abs(orb["twin"]["raoff"] - orb["base"]["raoff"]) / A),
        np.max(np.abs(orb["twin"]["deoff"] - orb["base"]["deoff"]) / A),
        np.max(np.abs(vir["twin"]["rel"][:, :2] - vir["base"]["rel"][:, :2]) / A),
    )
    worst["sym1_dz"] = np.max(np.abs(vir["twin"]["rel"][:, 2] + vir["base"]["rel"][:, 2]) / A[:, 0])
    k_o = np.max(np.abs(orb["base"]["vz"]), axis=(1, 2))[:, None, None]
    k_v = np.max(np.abs(vir["base"]["rv"]), axis=1)[:, None]
    worst["sym1_rv"] = max(np.max(np.abs(orb["twin"]["vz"] + orb["base"]["vz"]) / k_o),
                           np.max(np.abs(vir["twin"]["rv"] + vir["base"]["rv"]) / k_v))
    # 2.
    worst["sym2"] = max(
        np.max(np.abs(orb["flip"]["raoff"] + orb["base"]["raoff"]) / A),
        np.max(np.abs(orb["flip"]["deoff"] + orb["base"]["deoff"]) / A),
        np.max(np.abs(vir["flip"]["rel"] + vir["base"]["rel"]) / A),
    )
    # 3. dPA/dt = (ddec * vra - dra * vdec) / sep²
    worst["rate"] = 0.0
    for name, sign in (("base", 1.0), ("mirror", -1.0)):
        ra, de = orb[name]["raoff"], orb[name]["deoff"]
        rate_o = (de[..., 2] * _derivative(ra, h) - ra[..., 2] * _derivative(de, h)) / (ra[..., 2] ** 2 + de[..., 2] ** 2)
        x, y = vir[name]["rel"][:, 0], vir[name]["rel"][:, 1]
        vx, vy = vir[name]["vel"][:, 0], vir[name]["vel"][:, 1]
        rate_v = (y * vx - x * vy) / (x**2 + y**2)
        assert np.all(sign * rate_o > 0) and np.all(sign * rate_v > 0), name
        # the rate error as a velocity error, over 2 pi a / P
        err = np.abs(rate_v - rate_o) * np.hypot(x, y) / (2 * np.pi * A[:, 0] / p[:, None])
        worst["rate"] = max(worst["rate"], np.max(err))
    for k, v in worst.items():
        record("max_rel_" + k, v)
    assert worst["sym1_pos"] < TOL_POS and worst["sym1_dz"] < TOL_POS and worst["sym2"] < TOL_POS
    assert worst["sym1_rv"] < TOL_RV
    assert worst["rate"] < 1e-6


# ------------------------------------------------------- derived mass and distance


def _mass_cases(grid):
    return [(el, vo.KeplerOrbit(*prm, t_ref=REF)) for el, prm in zip(grid["els"][::4], grid["prm"][::4])]


@pytest.mark.validates("virgil.orbits.total_mass", "virgil.orbits.distance_pc", roots=["orbitize", "standards"])
def test_total_mass_and_distance_against_orbitize(grid):
    """total_mass(orbit, 1000/plx) recovers orbitize!'s mtot, and
    distance_pc(orbit, mtot) its distance, both with Kepler's third law and
    the IAU nominal GM_sun and au, since virgil#229 fixed ledger F14 (the
    old a³/P² in Julian years was low by 3.78e-5); and they invert each
    other."""
    worst_mass = worst_dist = worst_round = 0.0
    for el, orbit in _mass_cases(grid):
        d = 1000.0 / el["plx"]
        worst_mass = max(worst_mass, abs(float(vo.total_mass(orbit, d)) / el["mtot"] - 1))
        dist = float(vo.distance_pc(orbit, el["mtot"]))
        worst_dist = max(worst_dist, abs(dist / d - 1))
        worst_round = max(worst_round, abs(float(vo.total_mass(orbit, dist)) / el["mtot"] - 1))
    record("max_rel_mass_vs_orbitize", worst_mass)
    record("max_rel_distance_vs_orbitize", worst_dist)
    record("max_rel_round_trip", worst_round)
    assert max(worst_mass, worst_dist, worst_round) < 1e-12


@pytest.mark.validates("virgil.orbits.total_mass", "virgil.orbits.distance_pc", roots=["orbitize", "standards"])
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

# Matched invariant priors, the same variables on both sides:
#   a (sma, au): log-uniform on [1, 100]. With M_tot known to 3 %, log P =
#     1.5 log a - 0.5 log M is then log-uniform too, up to the ends; both
#     codes sample the same (log a, M_tot), so the priors match exactly.
#   e: uniform on [0, 1) (stated explicitly; orbitize!'s default).
#   i: uniform in cos i on [0°, 180°] (orbitize!'s SinPrior).
#   omega, Omega: uniform on [0°, 360°); the periastron time: uniform over
#     one period (orbitize!'s tau uniform on [0, 1)).
#   parallax and M_tot: Gaussians, 51.44 +- 0.12 mas and 1.75 +- 0.05 M_sun
#     (orbitize!'s beta Pic tutorials).
PRIORS = {"sma": (1.0, 100.0), "ecc": ("uniform", 0.0, 1.0), "plx": (51.44, 0.12), "mtot": (1.75, 0.05)}


def _beta_pic_positions(tmp_path):
    """The sep/PA rows of orbitize!'s betaPic.csv (Nielsen et al. 2020,
    Table 1) without its one companion RV, so that both codes fit positions
    only (Omega then known modulo 180°): a copy for orbitize!, and arrays."""
    src = _example_data("betaPic.csv")
    lines = [line for line in open(src).read().splitlines() if line and not line.startswith("#")]
    head = lines[0].split(",")
    col = {k: head.index(k) for k in ("epoch", "sep", "sep_err", "pa", "pa_err")}
    keep = [line for line in lines[1:] if line.split(",")[col["sep"]].strip()]
    path = tmp_path / "betaPic_positions.csv"
    path.write_text("\n".join([lines[0], *keep]) + "\n")
    pos = {k: np.array([float(line.split(",")[c]) for line in keep]) for k, c in col.items()}
    return path, pos


def _example_data(name):
    """A file of orbitize!'s example_data (in its own environment)."""
    import json
    import subprocess

    from external_bridge import _subprocess as sp

    code = "import orbitize, os, json; print(json.dumps(os.path.join(os.path.dirname(orbitize.__file__), 'example_data')))"
    out = subprocess.run([sp.python("orbitize"), "-W", "ignore", "-c", code], capture_output=True, text=True, check=True)
    return f"{json.loads(out.stdout.strip().splitlines()[-1])}/{name}"


def _fold(omega, Omega):
    """Positions fix Omega modulo 180°: report (omega, Omega) with Omega in
    [0°, 180°), moving omega with it."""
    flip = Omega % 360.0 >= 180.0
    return (omega - 180.0 * flip) % 360.0, Omega % 180.0


def beta_pic_model(pos):
    """The virgil side: PositionData.from_sep_pa's likelihood of
    KeplerOrbit, with PRIORS on (log a, e, cos i, omega, Omega, tau, plx,
    M_tot) and orbitize!'s period P = 2 pi sqrt(a³ / (G M))."""
    import numpyro
    import numpyro.distributions as dist

    data = vo.PositionData.from_sep_pa(pos["epoch"], pos["sep"], pos["pa"], pos["sep_err"], pos["pa_err"], t_ref=REF)

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
        # kepler_period with jax.numpy, so that NUTS can trace it
        period = numpyro.deterministic(
            "period", 2 * jnp.pi * jnp.sqrt((sma * AU_KM * 1e3) ** 3 / (GM_SUN * mtot)) / 86400.0)
        orbit = vo.KeplerOrbit(period, tau * period, ecc, jnp.degrees(jnp.arccos(cosi)), omega, Omega,
                               sma * plx, t_ref=REF)
        numpyro.factor("positions", data.loglike(orbit))

    return model


@pytest.mark.slow
@pytest.mark.campaign
@pytest.mark.skipif(os.environ.get("VALIDATION_CAMPAIGNS") not in ("1", "smoke"),
                    reason="campaign (hours): opt in with VALIDATION_CAMPAIGNS=1 (or =smoke for a minutes-long dry run)")
@pytest.mark.validates("pipeline:orbitize-posterior", "virgil.orbits.PositionData", "virgil.orbits.KeplerOrbit",
                       roots=["orbitize"], tier="C")
def test_beta_pic_posterior_matches_orbitize(tmp_path):
    """Posteriors of beta Pic b's orbit from orbitize!'s example data, with
    matched invariant priors (PRIORS): virgil (PositionData likelihood,
    NumPyro NUTS, 4 x 5000 draws) against orbitize! (ptemcee, 20
    temperatures x 1000 walkers, 2e6 orbits after 5000 burn-in steps,
    thinned by 10). The 16th, 50th and 84th percentiles of a_mas, e, i,
    Omega and omega (folded to Omega < 180°) and P must agree within 0.15
    of orbitize!'s posterior standard deviation. Hours of CPU: an OzSTAR
    campaign (tier C), run only with VALIDATION_CAMPAIGNS=1 (so not by the
    weekly CI's unfiltered pytest); written but not yet run.

    Definition difference to expect: orbitize!'s sep/PA likelihood treats
    sep and PA as independent Gaussians (PA residual wrapped);
    from_sep_pa's documented "independent errors on each" may be
    linearised into (dra, ddec), which matters only where sep * sigma_PA is
    not small against the curvature of the arc."""
    import jax.random
    from numpyro.infer import MCMC, NUTS, init_to_median

    # VALIDATION_CAMPAIGNS=smoke: the same pipeline at toy settings, to check
    # an OzSTAR job end to end in minutes; it records but does not assert
    smoke = os.environ.get("VALIDATION_CAMPAIGNS") == "smoke"
    path, pos = _beta_pic_positions(tmp_path)
    r = ob.run({
        "task": "posterior", "path": str(path), "algorithm": "MCMC", "tau_ref_epoch": REF,
        "mtot": PRIORS["mtot"][0], "mtot_err": PRIORS["mtot"][1],
        "plx": PRIORS["plx"][0], "plx_err": PRIORS["plx"][1], "priors": PRIORS,
        "mcmc_kwargs": {"num_temps": 2 if smoke else 20, "num_walkers": 100 if smoke else 1000, "num_threads": 1},
        "n": 20_000 if smoke else 2_000_000, "burn": 50 if smoke else 5000, "thin": 10,
    }, timeout=48 * 3600)
    theirs = {k: np.asarray(v) for k, v in r.items() if isinstance(v, list)}

    # start at the prior medians: numpyro's default draws each unconstrained
    # site uniformly on (-2, 2), which can give a negative total mass
    mcmc = MCMC(NUTS(beta_pic_model(pos), init_strategy=init_to_median), num_warmup=200 if smoke else 2000, num_samples=200 if smoke else 5000,
                num_chains=1 if smoke else 4, chain_method="sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(2020))
    s = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    ours = {"a_mas": np.exp(s["log_sma"]) * s["plx"], "e": s["ecc"], "i": np.degrees(np.arccos(s["cosi"])), "P": s["period"]}
    ours["omega"], ours["Omega"] = _fold(s["omega"], s["Omega"])
    ref = {"a_mas": theirs["sma"] * theirs["plx"], "e": theirs["ecc"], "i": np.degrees(theirs["inc"]),
           "P": kepler_period(theirs["sma"], theirs["mtot"])}
    ref["omega"], ref["Omega"] = _fold(np.degrees(theirs["aop"]), np.degrees(theirs["pan"]))
    worst = 0.0
    for k in ours:
        q = [16, 50, 84]
        dq = np.abs(np.percentile(ours[k], q) - np.percentile(ref[k], q)) / np.std(ref[k])
        record(f"max_dquantile_over_sd_{k}", np.max(dq))
        worst = max(worst, np.max(dq))
    record("smoke", float(smoke))
    if smoke:
        pytest.skip(f"smoke run: settings too small to compare (worst {worst:.2f} sd)")
    assert worst < 0.15
