# orbitize!: conventions and problems

[orbitize!](https://github.com/sblunt/orbitize) (Blunt et al. 2020, AJ 159,
89; Blunt et al. 2024, JOSS 9, 6756) is the orbit-fitting package of the
direct-imaging community, written and used by people. It is the external
root of trust for `virgil.orbits`: ephemerides, radial velocities, the
symmetries of visual orbits, and masses and distances from Kepler's third
law (see [design](design.md)).

orbitize! is BSD-3-Clause, but it has a C extension (its Kepler solver) and
a long dependency list (astropy, h5py, emcee, ptemcee, rebound, dynesty,
pymultinest, ...). So, like CANDID, it lives in its own environment:
`scripts/setup_external.sh orbitize` installs `orbitize==3.4.0` into
`.venv-orbitize`, and `src/external_bridge/orbitize_bridge.py` runs
`orbitize_worker.py` there in a subprocess, JSON in and out. It is called,
never vendored, and virgil has no dependency on it. Importing orbitize!
takes ~10 s, so the fast tests get every orbitize! number from one worker
process.

## Conventions

Sources. orbitize!: its manual (`docs/manual.rst`, v3.4.0: the ΔRA and Δdecl.
formulae, the RV formulae and ω\* = ω_p + 180°), `orbitize.kepler.calc_orbit`
and `tau_to_manom` (Kepler's equation, the mean anomaly from τ, the
`mass_for_Kamp` RV amplitude), `orbitize.basis.Period` and `tau_to_tp`
(Kepler's third law with astropy's G and M☉; years are astropy's Julian
year), `orbitize.system.System.compute_model` and `radec2seppa` (the stellar
RV as −m₁/m₀ times the companion's; γ; PA = atan2(ΔRA, Δdec) mod 360°) and
`orbitize.read_input.read_file` (companion RVs relative to the barycentre).
virgil: `docs/api/orbits.md`, the `virgil.orbits` docstrings, and
`design/orbit_scene_joint_fitting.md` §2 (definitions, symmetries,
Thiele–Innes constants). Pinned by `tests/test_orbitize.py`.

| Quantity | orbitize! | virgil | Mapping |
| --- | --- | --- | --- |
| relative position | `raoff` (ΔRA, East), `deoff` (Δdec, North), mas, companion − primary | `dra` East, `ddec` North, mas, secondary − primary | identical |
| separation, PA | `radec2seppa`: PA North through East, [0°, 360°) | `separation_pa`: the same | identical |
| semimajor axis | `sma` (au, relative orbit) and `plx` (mas) | `a_mas` (mas, relative orbit) | `a_mas = sma · plx` |
| distance | `plx` (mas) | `distance_pc` | `distance_pc = 1000 / plx` |
| period | from (`sma`, `mtot`): P = 2π √(a³ / (G M_tot)), astropy G and M☉ (G·M☉ = IAU 2015 nominal 1.3271244e20 m³ s⁻²) | `period` (days) | orbitize!'s P, in days (its `per` is in Julian years) |
| periastron time | `tau`: periastron epoch as a fraction of P after `tau_ref_epoch` (MJD; default 58849), M = 2π((t − τ_ref)/P − τ) | `dt_peri` (days) after `t_ref` | `t_ref = tau_ref_epoch`, `dt_peri = tau · P` |
| eccentricity | `ecc` | `ecc` | identical |
| inclination | `inc` (radians, [0, π]); i < 90° moves the PA forward | `inc` (degrees, [0°, 180°)); i < 90° moves the PA forward | degrees |
| argument of periastron | `aop` = ω_p, the **companion's** (manual: ω\* = ω_p + 180° for the star) | `omega`, the **secondary's** | identical, in degrees |
| node | `pan`: PA of the ascending node; the companion recedes there (its RV, K(cos(ω_p + f) + e cos ω_p), is positive at f = −ω_p) | `Omega`: PA of the node where the secondary recedes | identical, in degrees |
| third axis | none output; its RVs are positive receding | `dz` positive away from the observer | consistent: d(dz)/dt has the sign of orbitize!'s companion RV |
| companion RV | `calc_orbit(..., mass_for_Kamp=m0)`: barycentric, positive receding (km/s) | `RVData(star="secondary").model(orbit, q, γ, D)` − γ | identical, with `q = m1 / m0` |
| primary RV | −`calc_orbit(..., mass_for_Kamp=m1)`; in `System`, −(m₁/m₀) × the companion's | `RVData(star="primary").model(...)` − γ | identical |
| relative RV | `calc_orbit(..., mass_for_Kamp=mtot)` (the default) | `StateVectorOrbit.vz` (mas/yr) | `vz · D · au / Julian year` |
| total mass | `mtot` (M☉), with G | `total_mass(orbit, distance_pc)` = a³/P² (au, Julian years) | equal times (365.25 d / 365.2569 d)² = 1 − 3.78e-5 (F14) |
| γ | per instrument, `gamma_<inst>`, fitted only with primary RVs present, then added to every RV row of that instrument | `gamma`, added to every RV | equal when both stars' RVs are given; see P7 |

Positions alone fix the node only modulo 180°. orbitize!'s manual states the
same degeneracy (ω_p + π, Ω − π), and orbitize!'s default priors are already
the invariant ones: uniform in cos i (`SinPrior`), uniform in ω, Ω and τ.

A wrong mapping fails by the size of the orbit (the control test): the
primary's ω (as in jaxoplanet and spectroscopy), a node counted
counterclockwise from East (90° − Ω), or 180° − i each move the companion by
more than 2a.

## Agreement

Float64; tolerances and their reasons are in the test module's docstring.

| Check | Agreement |
| --- | --- |
| orbitize!'s period against 2π√(a³/GM) with the IAU constants; its Period basis round trip | 2e-15 |
| `KeplerOrbit.relative`, `separation_pa`, 47 orbits (e = 0, 1e-7, 0.1, 0.5, 0.8, 0.95; i = 0°, 1e-4°, 35°, 89.99°, 90°, 120°, 179.9°; cardinal ω, Ω) × 121 epochs over 1.6 periods | 1e-14 of a |
| `ThieleInnesOrbit` from the design note's constants; `KeplerOrbit.thiele_innes`; `to_kepler` (up to the documented twin) | 1e-14 of a; constants 9e-16; elements 6e-14° |
| `StateVectorOrbit.from_kepler`: sky velocity vs orbitize!'s finite differences; line-of-sight velocity vs orbitize!'s relative RV; μ; positions and elements on well-posed orbits | 1e-9 (finite differences); 1e-15; 4e-16; 2e-14 of a; 2e-13 |
| `RVData` primary and secondary vs `calc_orbit` (e up to 0.9, i > 90°, circular) | 2e-15 of K |
| `RVData` vs orbitize!'s `System.compute_model` with fitted masses and γ | 3e-10 of K (orbitize!'s default Kepler tolerance, 1e-9) |
| symmetries in both codes: (Ω+180°, ω+180°) positions, dz and RVs; ω + 180°; i → 180° − i reverses dPA/dt (sign at every epoch; rates agree) | 3e-15; 1e-15; 2e-9 (finite differences) |
| `total_mass`, `distance_pc` | the documented a³/P² to 7e-16; orbitize!'s mtot × (1 − 3.78e-5) to 9e-16 (F14) |

Not run here: the posterior comparison on orbitize!'s β Pic b data
(`orbitize/example_data/betaPic.csv`, Nielsen et al. 2020, AJ 159, 71; its
34 sep/PA rows, without the one companion RV), virgil NUTS against orbitize!'s
ptemcee with matched invariant priors (log-uniform in a, hence in P at the
3 % mass; uniform in e on [0, 1); uniform in cos i, ω, Ω and the periastron
time; Gaussian parallax 51.44 ± 0.12 mas and mass 1.75 ± 0.05 M☉). It is
`tests/test_orbitize.py::test_beta_pic_posterior_matches_orbitize`, marked
`slow` and tier C: hours of CPU, for OzSTAR. One difference of definition to
expect there: orbitize!'s sep/PA likelihood is two independent Gaussians in
separation and position angle; virgil's `PositionData.from_sep_pa` documents
"independent errors on each".

## virgil findings

| # | Finding | Evidence |
| --- | --- | --- |
| F14 | `total_mass` and `distance_pc` use a³/P² with P in Julian years. Kepler's third law with the IAU nominal GM☉ and au is M = 4π²a³/(GM☉ P²), equal to a³/P² only with P in Gaussian years (2π√(au³/GM☉) = 365.256898 d). virgil's masses are low by 3.78e-5 and its dynamical distances high by 1.26e-5. Use GM☉ (or the Gaussian year) instead. | `test_total_mass_and_distance_against_orbitize` (pinned); `test_f14_total_mass_is_keplers_third_law` (strict xfail) |
| F15 | `ThieleInnesOrbit.to_kepler` documents 0 ≤ Ω < 180° but returns Ω = 180.0 for a node at 180°: `KeplerOrbit(1000, 0, 0.3, 60, 270, 180, 100).to_thiele_innes().to_kepler().Omega == 180.0`. The orbit is right (the twin of Ω = 0, ω = 90°); only the documented range fails. | `test_f15_thiele_innes_node_in_documented_range` (strict xfail) |
| F16 | `StateVectorOrbit.to_kepler` loses precision near face-on. From `StateVectorOrbit.from_kepler(KeplerOrbit(1000, 100, 0.3, i, 30, 60, 100)).to_kepler()`: i = 1e-4° and 0.001° return 0°; 0.01° returns 0.010646°; 0.1° returns 0.0999970°; 1° returns 1 + 2.7e-10°; 179.9° returns 179.89999975°; 179.99° returns 179.98935°. Near-circular orbits (e = 1e-7) lose ~1e-8° too. Positions then drift by up to 2e-9 of a at 0.01° and 179.99°, and by 1.8e-8 of a on the grid (i = 0°, 1e-4°, 179.9° or e ≤ 1e-7); well-posed orbits agree with orbitize! to 2e-14. The state determines i to float64 precision (for example as atan2(\|h_xy\|, h_z) of the angular momentum), so the loss is in the conversion. | `test_state_vector_orbit_matches_orbitize` (pinned at 1e-7); `test_f16_state_vector_round_trip_keeps_float64_precision` (strict xfail) |

## Problems to raise (with approval)

| # | orbitize! | Problem | Evidence | Severity |
| --- | --- | --- | --- | --- |
| P7 | 3.4.0 | `read_input` asks for the RVs of non-primary bodies relative to the barycentre's RV, but `System.compute_model` adds the instrument's fitted γ to every RV row of that instrument, the companion's included, whenever primary RVs are present; with companion RVs alone no γ is fitted. So the model of barycentric companion RVs moves by γ (7 km/s in the test) when primary RVs are added. Physically, adding γ is right for absolute companion RVs from the same spectrograph, as virgil does; the documentation and the code disagree. | `test_radial_velocities_match_orbitize_system` (pinned); `test_p7_companion_rvs_are_barycentric_as_documented` (strict xfail) | low: documentation, or the joint-RV model |
