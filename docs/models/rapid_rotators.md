# Rapid rotators

`GravityDarkenedStar` is compared with a rotating star that we wrote from
Espinosa Lara & Rieutord (2011), not from virgil or from Shashank Dholakia's
code, whose values virgil's own tests use. Our star has the Roche shape, the
effective gravity and the ELR11 flux F = |g| tan²ϑ / tan²θ, integrated over a
400 × 800 grid on its surface; virgil sums a mesh of `n_lat` latitude rings.
Test: [`test_gravity_darkened_independent.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_gravity_darkened_independent.py).

## Grey model

| `GravityDarkenedStar` | Agreement at `n_lat` 128 |
| --- | --- |
| ω = 0.6–0.95; inclinations 0° (pole-on), 30°, 60°, 85° and 90° (equator-on); five position angles | 1.4–3.4e-4 |

Each doubling of `n_lat` cuts the error by 3.8–4.4, so the mesh converges at
second order. The default `n_lat` of 32 is good to 2–6e-3.

## Chromatic model

| `GravityDarkenedStar`, `t_pole` = 9000 K, ω = 0.9, inclination 50° | Agreement at `n_lat` 128 |
| --- | --- |
| 0.7, 1.65 and 2.2 µm, each surface element radiating B_λ(T) with T⁴ ∝ F | 1.3–1.4e-4 |

## Our reference

Checked against closed forms before it is used: with ω = 0 it is a uniform
disk (to 2e-5); its fluxes at the pole and the equator take ELR11's limits,
e^(2ω²r_p³/3)/r_p² and (1 − ω²)^(1/3) (to 1e-6 and 1e-3); and its surface normal
lies along g (to 1e-10).
