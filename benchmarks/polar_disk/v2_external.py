"""V2 (other codes): PolarDisk against PMOIRED and oimodeler.

PMOIRED: a ring with a radial ``profile`` expression and azimuthal harmonics
(``az ampN``, ``az projangN``; the earlier comparison in docs/method/pmoired.md pinned
that PMOIRED's azimuth is in the disk plane and ``az projangN`` relative to ``projang``)
against PolarDisk with linear Harmonics (c_m = A_m cos(m phi_m), s_m = A_m sin(m phi_m)),
on the same OIFITS file, at the coordinates PMOIRED uses. PMOIRED samples the
radial profile at ``Nr`` points (Nr = 3000 here, trapezoid), and computes rings exactly only
up to 30 samples per baseline (docs/method/pmoired.md, finding P2): the carrier file has 30.

oimodeler (0.8.1, Fourier-space components; its u, v are in cycles/rad):
  EIRing   thin elliptical ring (``d``, ``pa``, ``elong`` = 1/cos(inc)) vs ThinRing;
  Ring     uniform annulus (``din``, ``dout``) vs RadialPowerLaw(index 0, sharp edges);
  ESKRing  skewed thin ring: V = J0(x) - i skw sin(phi) J1(x) with phi from
           (skwPa - pa) and the rotated frequency (read from the source; the docstring
           calls ``skwPa`` an elongation): derivation below, vs ThinRing + linear m = 1;
  ExpRing  exponential tail (image-based radial-profile component, cut at 8 fwhm,
           exp(-0.692 (r - r0)/fwhm) as in its source (0.692, not ln 2), so L = fwhm / 0.692) vs ExpTail(cutoff).

oimodeler mapping of the skew: with q_maj, q_min the frequency components along and across
the major axis (pa), oimodeler uses (xp, yp) = (q_min cos(inc), q_maj) and
phi = (skwPa - pa) + atan2(yp, xp), so sin(phi) = cos(Delta - phi_q) with Delta = skwPa - pa
and phi_q the in-plane azimuth of the frequency (from the major axis towards pa + 90);
virgil's m = 1 term is -i J1 (c1 cos phi_q + s1 sin phi_q), so c1 = skw cos(Delta),
s1 = skw sin(Delta).

PASS CRITERIA (fixed before the runs)
  P  PMOIRED rings with M = 1-3 harmonics and a power-law profile: max |dV2| <= 1e-5
     and |d closure phase| <= 1e-3 degrees (PMOIRED's Nr-sampling floor at Nr = 3000);
     the largest |closure phase| in the file >= 1 degree (sensitivity).
  O1 EIRing vs ThinRing: <= 1e-10. O2 Ring vs sharp annulus: <= 1e-6.
  O3 ESKRing (thin) vs ThinRing + m = 1: <= 1e-10 with the mapping above; the other sense
     (c1, s1 swapped) must differ by > 1e-3.
  O4 ExpRing vs ExpTail with the same cut: <= 5e-3 (image-based; its own Fourier
     sampling), reported against the exact transform too.
"""

import contextlib
import io
import json
import pathlib
import tempfile

import numpy as np

from common import save
from crosscheck import radial as R
from crosscheck import simulate
from external_bridge.pmoired_models import model_samples
from virgil import ExpTail, GaussianRing, Harmonics, PolarDisk, RadialPowerLaw, ThinRing

UTS = np.array([[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]])


def carrier(path):
    simulate.observe(
        path, lambda u, v, w: np.ones(np.broadcast(u, v, w).shape, complex), UTS,
        hour_angles_h=np.linspace(-3, 3, 5), wavelengths=np.linspace(1.5e-6, 2.4e-6, 6), dec_deg=-50.0,
        sigma_v2=0.01, sigma_cp_deg=0.5)


def compare_pmoired(path, disk, params, nr=3000):
    s = model_samples(path, params, {"Nr": nr})

    def vis(u, v):
        return np.asarray(disk.model(u, v, 1e-6))

    v2 = np.abs(vis(s["u"], s["v"])) ** 2
    t3 = vis(s["u1"], s["v1"]) * vis(s["u2"], s["v2_"]) * np.conj(vis(s["u1"] + s["u2"], s["v1"] + s["v2_"]))
    dcp = (np.rad2deg(np.angle(t3)) - s["t3phi"] + 180.0) % 360.0 - 180.0
    return {"max_abs_dV2": float(np.abs(v2 - s["v2"]).max()), "max_abs_dCP_deg": float(np.abs(dcp).max()),
            "max_abs_CP_deg": float(np.abs(s["t3phi"]).max())}


def harmonics_from_pmoired(amps, phis_deg):
    c = tuple(a * np.cos(np.deg2rad(m * p)) for m, (a, p) in enumerate(zip(amps, phis_deg), 1))
    s = tuple(a * np.sin(np.deg2rad(m * p)) for m, (a, p) in enumerate(zip(amps, phis_deg), 1))
    return Harmonics(c=c, s=s)


def pmoired_section(tmp):
    path = tmp / "carrier.fits"
    carrier(path)
    out = {}
    inc, pa, dra, ddec = 50.0, -20.0, 0.5, -0.3
    for label, amps, phis in (("M1", (0.4,), (40.0,)), ("M2", (0.4, 0.3), (40.0, 100.0)),
                              ("M3", (0.4, 0.3, 0.2), (40.0, 100.0, -30.0))):
        disk = PolarDisk(GaussianRing(6.0, 0.8), harmonics=harmonics_from_pmoired(amps, phis), inc=inc, pa=pa,
                         dra=dra, ddec=ddec, n_r=96)
        r0, sg = 6.0, 0.8
        params = {"diamin": 2 * (r0 - 8 * sg), "diamout": 2 * (r0 + 8 * sg), "profile": f"np.exp(-0.5*(($R-{r0})/{sg})**2)",
                  "incl": inc, "projang": pa, "x": dra, "y": ddec}
        for m, (a, p) in enumerate(zip(amps, phis), 1):
            params[f"az amp{m}"], params[f"az projang{m}"] = a, p
        row = compare_pmoired(path, disk, params)
        row["passed"] = bool(row["max_abs_dV2"] <= 1e-5 and row["max_abs_dCP_deg"] <= 1e-3 and row["max_abs_CP_deg"] >= 1.0)
        out[f"gaussian_ring_{label}"] = row
    disk = PolarDisk(RadialPowerLaw(5.0, 14.0, -1.0), harmonics=harmonics_from_pmoired((0.5,), (70.0,)), inc=35.0, pa=110.0,
                     dra=-0.4, ddec=0.6, n_r=96)
    params = {"diamin": 10.0, "diamout": 28.0, "profile": "$R**-1", "incl": 35.0, "projang": 110.0, "x": -0.4, "y": 0.6,
              "az amp1": 0.5, "az projang1": 70.0}
    row = compare_pmoired(path, disk, params)
    row["passed"] = bool(row["max_abs_dV2"] <= 1e-5 and row["max_abs_dCP_deg"] <= 1e-3 and row["max_abs_CP_deg"] >= 1.0)
    out["power_law_M1"] = row
    return out


def oim_vis(component, u, v, wl):
    import oimodeler as oim

    with contextlib.redirect_stdout(io.StringIO()):
        return oim.oimModel(component).getComplexCoherentFlux(u / wl, v / wl, np.full(np.shape(u), wl))


def oimodeler_section():
    import oimodeler as oim

    rng = np.random.default_rng(5)
    r = 130.0 * np.sqrt(rng.uniform(0.01, 1.0, 300))
    a = rng.uniform(0, 2 * np.pi, 300)
    u, v, wl = r * np.cos(a), r * np.sin(a), 2.2e-6
    qu, qv = u / wl, v / wl
    out = {}
    vg = lambda d: np.asarray(d.model(u, v, wl))
    # O1
    elong, pa, d = 1.5, 30.0, 12.0
    inc = np.rad2deg(np.arccos(1.0 / elong))
    ref = oim_vis(oim.oimEIRing(d=d, pa=pa, elong=elong, x=0.4, y=-0.7), u, v, wl)
    out["O1_EIRing"] = {"max_abs_dV": float(np.abs(vg(PolarDisk(ThinRing(d / 2), inc=inc, pa=pa, dra=0.4, ddec=-0.7)) - ref).max())}
    out["O1_EIRing"]["passed"] = out["O1_EIRing"]["max_abs_dV"] <= 1e-10
    # O2
    ref = oim_vis(oim.oimRing(din=6.0, dout=14.0), u, v, wl)
    err = float(np.abs(vg(PolarDisk(RadialPowerLaw(3.0, 7.0, 0.0), n_r=64)) - ref).max())
    out["O2_Ring"] = {"max_abs_dV": err, "passed": err <= 1e-6}
    # O3
    rows = {}
    for skw, skwpa, pa_, elong_ in ((0.6, 40.0, 25.0, 1.4), (0.3, -70.0, 100.0, 1.0), (0.8, 10.0, -50.0, 2.0)):
        ref = oim_vis(oim.oimESKRing(din=12.0, dout=12.0, skw=skw, skwPa=skwpa, pa=pa_, elong=elong_), u, v, wl)
        inc_ = np.rad2deg(np.arccos(1.0 / elong_))
        delta = np.deg2rad(skwpa - pa_)

        def disk(c1, s1):
            return PolarDisk(ThinRing(6.0), harmonics=Harmonics(c=(c1,), s=(s1,)), inc=inc_, pa=pa_)

        good = float(np.abs(vg(disk(skw * np.cos(delta), skw * np.sin(delta))) - ref).max())
        swapped = float(np.abs(vg(disk(skw * np.sin(delta), skw * np.cos(delta))) - ref).max())
        rows[f"skw={skw},skwPa={skwpa},pa={pa_},elong={elong_}"] = {
            "max_abs_dV": good, "swapped_c_s_control": swapped, "passed": bool(good <= 1e-10 and swapped > 1e-3)}
    out["O3_ESKRing"] = rows
    # O4
    d, fwhm = 4.0, 2.1
    L = fwhm / 0.692  # oimodeler's source: exp(-0.692 (r - r0) / fwhm) (0.692, not ln 2)
    n_cut = 8.0 * fwhm / L
    row = {}
    for dim in (512, 2048):
        ref = oim_vis(oim.oimExpRing(d=d, fwhm=fwhm, dim=dim), u, v, wl)
        disk = PolarDisk(ExpTail(d / 2, L, cutoff=n_cut), n_r=128)
        exact = R.hankel(R.exp_tail(d / 2, L, n_scales=n_cut), np.hypot(qu, qv))
        row[f"dim={dim}"] = {"virgil_vs_oimodeler": float(np.abs(vg(disk) - ref).max()),
                             "oimodeler_vs_exact": float(np.abs(ref - exact).max()),
                             "virgil_vs_exact": float(np.abs(vg(disk) - exact).max())}
    row["cut_in_scale_lengths"] = float(n_cut)
    row["passed"] = bool(all(v_["virgil_vs_oimodeler"] <= 5e-3 for k, v_ in row.items() if k.startswith("dim=")))
    out["O4_ExpRing"] = row
    return out


def main():
    with tempfile.TemporaryDirectory() as tmp:
        out = {"pmoired": pmoired_section(pathlib.Path(tmp)), "oimodeler": oimodeler_section()}
    path = save("v2_external", out)
    print(json.dumps(out, indent=1, default=float))
    print("wrote", path)


if __name__ == "__main__":
    main()
