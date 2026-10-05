# Rapid rotators

`tests/test_gravity_darkened_independent.py`. The reference,
`crosscheck.elr`, is written from Espinosa Lara & Rieutord (2011) rather
than from virgil or from Dholakia's code, whose golden values virgil's own
tests use. It covers the Roche surface, the effective gravity and the ELR11
flux, F = |g| tan²ϑ / tan²θ. It integrates F times the projected area over
a 400 × 800 midpoint grid on the surface. virgil sums its mesh of
`n_lat` latitude rings.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| — | the reference: ω = 0 is a uniform disk; ELR11's polar and equatorial fluxes, e^(2ω²r_p³/3)/r_p² and (1 − ω²)^(1/3); the surface normal along g | Mathematics, literature | 2e-5; 1e-6 and 1e-3; 1e-10 |
| `GravityDarkenedStar` (grey), ω = 0.6–0.95, inclinations 0° (pole-on), 30°, 60°, 85° and 90° (equator-on), five position angles | the reference's visibilities | Mathematics | 1.4–3.4e-4 at `n_lat` 128; each doubling of `n_lat` cuts the error by 3.8–4.4 (second order) |
| `GravityDarkenedStar` (chromatic, `t_pole` = 9000 K, ω = 0.9, inclination 50°), 0.7, 1.65 and 2.2 µm | the same, each point radiating B_λ(T) with T⁴ ∝ F | Mathematics | 1.3–1.4e-4 at `n_lat` 128 |

