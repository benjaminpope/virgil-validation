# Orbits against orbitize!

`virgil.orbits` against orbitize! 3.4.0 (Blunt et al. 2020, 2024), in its
own environment (`tests/test_orbitize.py`; conventions and mapping in
[orbitize_notes.md](../method/orbitize.md)). 47 orbits over e = 0 to 0.95 and
i = 0° to 179.9°, 121 epochs each.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| orbitize!'s period (self-check) | 2π√(a³/GM☉), IAU constants | Analytic | 2e-15 |
| `KeplerOrbit.relative`, `separation_pa` | `calc_orbit`, `radec2seppa` | orbitize! | 1e-14 of a |
| `ThieleInnesOrbit` (design-note constants), `to_kepler` | `calc_orbit` | orbitize! | 1e-14 of a; elements 6e-14° up to the documented twin |
| `StateVectorOrbit` velocities, μ, positions (well-posed orbits) | finite differences of `calc_orbit`, its relative RV | orbitize! | 1e-9 (finite differences), 1e-15, 2e-14 of a; near face-on too since virgil#229 (F16) |
| `RVData`, primary and secondary | `calc_orbit` with `mass_for_Kamp`; `System.compute_model` | orbitize! | 2e-15 of K; 3e-10 through `System` (its Kepler tolerance) |
| (Ω+180°, ω+180°), ω+180°, i → 180°−i | each code against itself | Analytic, orbitize! | 3e-15; sense of rotation at every epoch |
| `total_mass`, `distance_pc` | orbitize!'s `mtot`, `plx` | orbitize!, standards | 1e-12 since virgil#229 (F14; was 3.78e-5 and 1.26e-5) |

