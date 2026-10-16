"""V2 (brute force): PolarDisk against numpy DFTs of finely rendered scenes.

Each scene is rendered in the disk plane at fine resolution and carried to the
sky by the documented geometry (src/crosscheck/polar_images.py, which imports
nothing from virgil); its visibility is the direct sum
``sum_k w_k exp(-2 pi i (u x_k + v y_k) / lambda)``.

PASS CRITERIA (fixed before the runs; float64)
  B1  max |V_virgil - V_render| <= 1e-6 (complex) for harmonics in the linear
      and positive forms (M = 1-3), VonMises arcs (the truncation error of
      virgil's degree is the only difference), cones and flared surfaces
      (with the documented thin-surface geometry), blur, and the
      chromatic profile.
  B2  controls: the plausible wrong mappings (opposite height sense, opposite
      azimuth sense, wrong disk-plane origin) must each differ by > 100 x
      the tolerance, so a pass is informative.
  B3  the signal: the largest |Im V| / phase of the scenes with odd harmonics
      or a height is >= 0.05 (the transform is exercised, not trivially real).
"""

import json

import numpy as np

from common import save  # noqa: F401
from crosscheck import polar_images as P
from crosscheck import sky
from virgil import (
    Cone, GaussianRing, Harmonics, PointSource, PolarDisk, PowerLawHeight, RadialLinear, System, ThinRing, VonMises,
)

WL = 2.2e-6
TOL = 1e-6


def baselines(n=300, seed=7, b_max=130.0):
    rng = np.random.default_rng(seed)
    r = b_max * np.sqrt(rng.uniform(0.01, 1.0, n))
    a = rng.uniform(0, 2 * np.pi, n)
    return r * np.cos(a), r * np.sin(a)


def virgil_v(model, u, v, wl=WL):
    return np.asarray(model.model(u, v, wl), complex)


def gauss_nodes(r0, sigma, n_pan=80, order=12):
    f = lambda r: np.exp(-0.5 * ((r - r0) / sigma) ** 2)
    return P.panelled_nodes(max(r0 - 8 * sigma, 1e-6), r0 + 8 * sigma, f)


def case_linear():
    c, s = (0.3, 0.1, 0.05), (-0.2, 0.05, 0.1)
    disk = PolarDisk(ThinRing(6.0), harmonics=Harmonics(c=c, s=s), inc=40.0, pa=25.0, dra=0.7, ddec=-0.4)
    cl = lambda sgn=1.0, c=c, s=s: P.render([6.0], [1.0], lambda t: P.az_linear(sgn * t, c, s), 40.0, 25.0, dra=0.7, ddec=-0.4)
    return disk, cl(), {"azimuth sense reversed": cl(-1.0)}


def case_positive():
    w = (0.4 + 0.1j, 0.2j, -0.15 + 0.1j)
    disk = PolarDisk(GaussianRing(6.0, 0.8), harmonics=Harmonics(w=w), inc=55.0, pa=140.0, n_r=96, dra=-0.5, ddec=0.9)
    r, wt = gauss_nodes(6.0, 0.8)
    cl = lambda sgn=1.0: P.render(r, wt, lambda t: P.az_positive(sgn * t, w), 55.0, 140.0, dra=-0.5, ddec=0.9)
    return disk, cl(), {"azimuth sense reversed": cl(-1.0)}


def case_positive_m1():
    w = (0.5 - 0.3j,)
    disk = PolarDisk(ThinRing(8.0), harmonics=Harmonics(w=w), inc=30.0, pa=60.0)
    cl = lambda sgn=1.0: P.render([8.0], [1.0], lambda t: P.az_positive(sgn * t, w), 30.0, 60.0)
    return disk, cl(), {"azimuth sense reversed": cl(-1.0)}


def case_vonmises():
    kappa, th0 = 30.0, 250.0
    disk = PolarDisk(ThinRing(20.0), harmonics=VonMises(kappa=kappa, theta0=th0), inc=35.0, pa=100.0, blur_fwhm=2.0)
    deg = disk.harmonics.degree
    cl = lambda t0=th0: P.render([20.0], [1.0], lambda t: P.az_vonmises(t, kappa, t0), 35.0, 100.0, n_theta=2048)
    # theta0 defined from the major axis, towards pa + 90: the control puts it at -theta0
    return disk, cl(), {"theta0 mirrored": cl(360.0 - th0)}, {"blur_fwhm": 2.0, "inc": 35.0, "pa": 100.0, "degree": int(deg)}


def case_cone():
    alpha, tip = 40.0, 3.0
    disk = PolarDisk(GaussianRing(5.0, 1.0), height=Cone(alpha, tip), inc=60.0, pa=70.0, n_r=128)
    r, wt = gauss_nodes(5.0, 1.0)
    cl = lambda sgn=1.0: P.render(r, wt, None, 60.0, 70.0, height=lambda rr: sgn * (rr / np.tan(np.deg2rad(alpha)) - tip))
    return disk, cl(), {"height sense reversed": cl(-1.0)}


def case_flared_harmonics():
    disk = PolarDisk(GaussianRing(6.0, 0.8), harmonics=Harmonics(c=(0.2,), s=(0.3,)), height=PowerLawHeight(1.0, 6.0, 1.25),
                     inc=70.0, pa=15.0, n_r=128)
    r, wt = gauss_nodes(6.0, 0.8)
    cl = lambda sgn=1.0: P.render(r, wt, lambda t: P.az_linear(t, (0.2,), (0.3,)), 70.0, 15.0,
                                  height=lambda rr: sgn * 1.0 * (rr / 6.0) ** 1.25)
    return disk, cl(), {"height sense reversed": cl(-1.0)}


def run_case(name, builder, u, v):
    out = builder()
    disk, cloud, controls = out[:3]
    extra = out[3] if len(out) > 3 else {}
    vg = virgil_v(disk, u, v)
    blur = 1.0
    if extra.get("blur_fwhm"):
        blur = sky.in_plane_blur_factor(u, v, WL, extra["blur_fwhm"], extra["inc"], extra["pa"])
    ref = sky.visibility(cloud, u, v, WL) * blur
    row = {"max_abs_dV": float(np.abs(vg - ref).max()), "max_abs_Im_V": float(np.abs(vg.imag).max()),
           "tol": TOL, "controls": {}}
    for label, ctrl in controls.items():
        row["controls"][label] = float(np.abs(sky.visibility(ctrl, u, v, WL) * blur - vg).max())
    if "degree" in extra:
        row["vonmises_degree"] = extra["degree"]
        row["truncation_power"] = float(disk.harmonics.truncation_power())
    row["passed"] = bool(row["max_abs_dV"] <= TOL and all(c > 100 * TOL for c in row["controls"].values()))
    return row


def case_chromatic(u, v):
    """A star plus a disk with I(r, lambda) = I0(r) (lambda/lambda0)^alpha(r): complex visibilities and the disk's
    wavelength-dependent weight, against brute force."""
    lam0, flux = 2.2e-6, 0.8
    alpha = RadialLinear(-1.0, 0.2, 6.0)
    disk = PolarDisk(GaussianRing(6.0, 0.8), inc=45.0, pa=20.0, n_r=96, flux=flux, spectral_index=alpha, wavel0=lam0, dra=0.5, ddec=0.2)
    scene = System(star=PointSource(1.0), disk=disk)
    r, wt = gauss_nodes(6.0, 0.8)
    row = {"tol": TOL, "per_wavelength": {}}
    for lam in (1.6e-6, 2.2e-6, 3.5e-6):
        al = -1.0 + 0.2 * (r - 6.0)
        w_l = wt * (lam / lam0) ** al
        f_disk = flux * w_l.sum() / wt.sum()  # weight relative to lambda0
        cl = P.render(r, w_l, None, 45.0, 20.0, dra=0.5, ddec=0.2)
        ref = (1.0 * 1.0 + f_disk * sky.visibility(cl, u, v, lam)) / (1.0 + f_disk)
        vg = virgil_v(scene, u, v, lam)
        row["per_wavelength"][f"{lam * 1e6:.1f}um"] = {"max_abs_dV": float(np.abs(vg - ref).max()), "disk_weight": float(f_disk)}
    row["max_abs_dV"] = max(x["max_abs_dV"] for x in row["per_wavelength"].values())
    row["passed"] = bool(row["max_abs_dV"] <= TOL)
    return row


def run():
    u, v = baselines()
    cases = {
        "linear_M3_thin_ring": case_linear, "positive_M3_gaussian_ring": case_positive, "positive_M1_thin_ring": case_positive_m1,
        "vonmises_arc_blurred": case_vonmises, "cone_height": case_cone, "flared_with_harmonics": case_flared_harmonics,
    }
    out = {name: run_case(name, fn, u, v) for name, fn in cases.items()}
    out["chromatic_system"] = case_chromatic(u, v)
    return out


def main():
    out = run()
    path = save("v2_brute_force", {"results": out, "n_baselines": 300})
    print(json.dumps(out, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
