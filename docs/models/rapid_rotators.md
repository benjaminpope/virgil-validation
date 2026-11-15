# Rapid rotators

This page checks `GravityDarkenedStar` against a rotating star that we wrote
from Espinosa Lara & Rieutord (2011), independently of virgil and of Shashank
Dholakia's code, whose values virgil's own tests use, and confirms its polar
position-angle convention (`pole_pa`) against `PMOIRED`'s independent
`rotastar` model. Our star has the Roche shape, the effective gravity and the
ELR11 flux F = |g| tan²ϑ / tan²θ, integrated over a 400 × 800 grid on its
surface. virgil sums a mesh of `n_lat` latitude rings.
Tests:
[`test_gravity_darkened_independent.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_gravity_darkened_independent.py),
[`test_rotator_convention.py`](https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_rotator_convention.py).

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

## Pole orientation and external checks

`GravityDarkenedStar` defines `pa` as the sky position angle (degrees North
through East) of the visible rotational pole (`incl` in `(0°, 90°)`), placing
the hot polar cap at `pa` and the oblate equatorial major axis at `pa + 90°`.

| Check | Reference | Result |
| --- | --- | --- |
| `pole_pa` convention (`pa = 33°`, `incl = 55°`, `T_pole = 9500 K`, `ω_PM = 0.95` mapped to `ω_ELR = 0.750`) | `PMOIRED` native `rotastar.Vrota` (`projang = 33°`, [Aufdenberg et al. 2006](https://doi.org/10.1086/504149)) | `|ΔV²| < 9.0e-3`, `|ΔCP| < 2.1°` on `168°` closure phases (ATLAS9 limb-darkening offset only); `pa + 90°`, `pa + 180°` and `−pa` controls miss by `60°–149°` |
| Wavefront synthesis (`pa = 33°`, `incl = 60°`, 7-hole non-redundant mask at 4.8 µm) | `dLux` optical diffraction from the ELR11 surface cloud (checks optical propagation without claiming `pole_pa`) | `|ΔV²| < 3e-3`, `|ΔCP| < 0.08°` across 21 baselines and 35 closure phases |
| `oimodeler.oimFastRotator` (`dim = 64`) | OIFITS low-frequency photocentre slope and equator-on slow-rotator scaling | Upstream bugs [`P13`](../method/findings.md#P13) (`np.rot90` flips pole PA by `180°`) and [`P14`](../method/findings.md#P14) (`rot**0.4` slow-rotation scaling, slope `0.48` vs physical `2.0`) |

![Sky brightness and 2.2 µm closure phases of GravityDarkenedStar against PMOIRED rotastar](../assets/rapid_rotator_convention.png)

## Our reference

We checked it against closed forms before using it: with ω = 0 it is a uniform
disk (to 2e-5); its fluxes at the pole and the equator take ELR11's limits,
e^(2ω²r_p³/3)/r_p² and (1 − ω²)^(1/3) (to 1e-6 and 1e-3); and its surface normal
lies along g (to 1e-10).
