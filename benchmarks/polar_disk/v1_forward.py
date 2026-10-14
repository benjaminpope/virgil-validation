"""V1: forward transforms, accuracy and speed (design/radial_profiles.md, section 12).

PolarDisk (M = 0, no harmonics) against frank's DiscreteHankelTransform,
vis-r's ``r_prof_*`` with ``uv_trans`` / ``setup_dht`` (including its Gaussian
scale-height factor) and FRAP's Hankel matrix (``set_visibility`` without the
primary beam), for identical profiles and geometry, with our own
Gauss-Legendre Hankel transform (src/crosscheck/radial.py) as the truth.

PASS CRITERIA (fixed before the runs; float64, CPU)
  A1  virgil vs truth: max |dV| <= 1e-6 for every scene at the stated n_r
      (ExpTail, whose Gauss-Laguerre rule is documented as 1e-5 at about
      16 * 2 pi q L nodes: <= 1e-5).
  A2  frank (N = 300 nodes) and FRAP (N = 300): each vs truth <= 1e-6 (they
      are exact for a profile inside r_max and band-limited by N); virgil vs
      each <= 1e-5.
  A3  vis-r (setup_dht defaults: sz = 8.84, nhpt = 300; its DHT is built for
      arcsec-scale discs and interpolates linearly in q): its error against
      truth is reported; virgil vs vis-r must satisfy
      |V_virgil - V_visr| <= 1.1 |V_visr - V_truth| + 1e-6 at every baseline
      (any difference is vis-r's own discretization error, not virgil's).
  A4  scale-height factor (vis-r's vertical structure): with the same factor
      applied to virgil's visibilities (PolarDisk has no scale height; the
      harness multiplies by exp(-0.5 (2 pi h r0 sin i u_minor)^2)), the
      inclined-disk geometry must agree as in A3.
  S   speed, reported only (no pass/fail): median wall time per call at
      N_vis = 1e3 and 1e4 (CPU), the matrix built per call ("cold") and
      cached ("hot") for frank and FRAP, jitted for virgil.
GPU and N_vis = 1e5 timings are deferred (OzSTAR).

Two regimes are run: an ALMA-like scale (r0 = 1000 mas, q up to 8e5) in
which vis-r's defaults are meaningful, and an optical/IR scale (r0 = 10 mas,
q up to 8e7) for virgil, frank and FRAP. The transform depends on q r only.
"""

import json
import time

import jax
import numpy as np

from common import save  # noqa: E402  (sets x64 and the path)
from crosscheck import radial as R
from external_bridge import radial_codes as X
from virgil import Gap, GaussianRing, PolarDisk, ExpTail, RadialPowerLaw

INC, PA = 50.0, 30.0
ASPECT = 0.1  # vis-r scale height h = H / r


def scenes(s):
    """name -> (virgil disk, truth profile, vis-r profile(r_arcsec) or None, n_r, tolerance)."""
    a = s / 1000.0  # mas -> arcsec for vis-r
    from vis_r import functions as vf

    out = {}
    out["gauss_ring"] = (
        PolarDisk(GaussianRing(10 * s, 1.5 * s), inc=INC, pa=PA, n_r=64),
        R.gaussian_ring(10 * s, 1.5 * s),
        lambda r: vf.r_prof_gauss(r, [10 * a, 1.5 * a]), 64, 1e-6)
    par = [5 * a, -1.0, 0.3 / 5, 14 * a, 0.3 / 14]  # vis-r: r_i, power, sigma_i/r_i, r_o, sigma_o/r_o
    # one soft-edge width in virgil: 0.3 s at both edges, so vis-r's fractional widths are 0.3/r_i, 0.3/r_o
    out["erf_power"] = (
        PolarDisk(RadialPowerLaw(5 * s, 14 * s, -1.0, edge=0.3 * s), inc=INC, pa=PA, n_r=96),
        R.erf_power(5 * s, 14 * s, -1.0, 0.3 * s),
        lambda r: vf.r_prof_erf2_power(r, par), 96, 1e-6)
    par2 = [4 * a, 0.0, 0.3 / 4, 20 * a, 0.3 / 20, 0.9, 12 * a, 0.8 * a, 0.0, 12 * a, 0.8 * a]
    out["gap_on_plateau"] = (
        PolarDisk(Gap(RadialPowerLaw(4 * s, 20 * s, 0.0, edge=0.3 * s), r0=12 * s, width=0.8 * s, depth=0.9),
                  inc=INC, pa=PA, n_r=128),
        R.with_gap(R.erf_power(4 * s, 20 * s, 0.0, 0.3 * s), 12 * s, 0.8 * s, 0.9),
        lambda r: vf.r_prof_erf2_power_ggap2(r, par2), 128, 1e-6)
    out["exp_tail"] = (
        PolarDisk(ExpTail(2 * s, 3 * s), inc=INC, pa=PA, n_r=128),
        R.exp_tail(2 * s, 3 * s), None, 128, 1e-5)
    return out


def timeit(fn, reps=15):
    fn()  # warm-up
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return float(np.median(t))


def regime(label, s, q_max, n_vis, with_visr, wl):
    u, v = R.baselines(n_vis, q_max=q_max, seed=1)
    qd, umaj, umin = R.deproject(u, v, INC, PA)
    res = {}
    r_ext = {"gauss_ring": 20 * s, "erf_power": 17 * s, "gap_on_plateau": 24 * s, "exp_tail": 2 * s + 60 * 3 * s}
    for name, (disk, prof, visr_prof, n_r, tol) in scenes(s).items():
        truth = R.disk_visibility(prof, u, v, INC, PA)
        vm = jax.jit(lambda uu, vv, d=disk: d.model(uu, vv, wl))
        v_virgil = np.asarray(vm(u * wl, v * wl)).real
        row = {"n_r": n_r, "virgil_vs_truth": float(np.abs(v_virgil - truth).max()), "tol": tol}
        # frank and FRAP: nodes cover the profile with a margin (r_max), N = 300
        r_max = r_ext[name] * 1.5 if name != "exp_tail" else 60 * 3 * s
        f = prof[0]
        fr = X.Frank(r_max, 300, qd)
        row["frank_vs_truth"] = float(np.abs(fr.visibility(f) - truth).max())
        row["virgil_vs_frank"] = float(np.abs(v_virgil - fr.visibility(f)).max())
        row["frank_q_max_ok"] = bool(fr.q_max >= qd.max())
        fp = X.Frap(r_max, 300, qd)
        row["frap_vs_truth"] = float(np.abs(fp.visibility(f) - truth).max())
        row["virgil_vs_frap"] = float(np.abs(v_virgil - fp.visibility(f)).max())
        if with_visr and visr_prof is not None:
            vr = X.VisR(u, v, INC, PA, sz=8.84)
            v_visr = vr.visibility(visr_prof)
            d_vt = np.abs(v_visr - truth)
            row["visr_vs_truth"] = float(d_vt.max())
            row["virgil_vs_visr"] = float(np.abs(v_virgil - v_visr).max())
            row["visr_criterion_margin"] = float((1.1 * d_vt + 1e-6 - np.abs(v_virgil - v_visr)).min())
            row["visr_n_q"] = int(len(vr.Qzero))
            row["visr_r_out_arcsec"] = float(vr.Rnk.max() / X.ARCSEC)
        res[name] = row
    # scale height (vis-r's vertical structure) on the Gaussian ring
    if with_visr:
        name = "gauss_ring"
        disk, prof, visr_prof, _, _ = scenes(s)[name]
        vr = X.VisR(u, v, INC, PA, sz=8.84)
        h_r0_rad = ASPECT * 10 * s * R.MAS
        factor = np.exp(-0.5 * (2 * np.pi * h_r0_rad * np.sin(np.deg2rad(INC)) * umin) ** 2)
        truth_h = R.disk_visibility(prof, u, v, INC, PA) * factor
        v_virgil_h = np.asarray(disk.model(u * wl, v * wl, wl)).real * factor
        v_visr_h = vr.visibility(visr_prof, aspect=ASPECT, r0_arcsec=10 * s / 1000.0)
        d = np.abs(v_visr_h - truth_h)
        res["gauss_ring_scale_height"] = {
            "aspect": ASPECT, "virgil_with_factor_vs_truth": float(np.abs(v_virgil_h - truth_h).max()),
            "visr_vs_truth": float(d.max()), "virgil_vs_visr": float(np.abs(v_virgil_h - v_visr_h).max()),
            "visr_criterion_margin": float((1.1 * d + 1e-6 - np.abs(v_virgil_h - v_visr_h)).min()),
            "factor_min": float(factor.min())}
    return res


def timing(s, q_max, wl, n_vis_list=(1000, 10000)):
    """Wall time per call, one scene (the Gaussian ring), median of 15 calls."""
    from vis_r import functions as vf

    out = {}
    a = s / 1000.0
    for n in n_vis_list:
        u, v = R.baselines(n, q_max=q_max, seed=2)
        qd, _, _ = R.deproject(u, v, INC, PA)
        disk, prof, visr_prof, _, _ = scenes(s)["gauss_ring"]
        f = prof[0]
        t0 = time.perf_counter()
        vm = jax.jit(lambda uu, vv: disk.model(uu, vv, wl))
        vm(u * wl, v * wl).block_until_ready()
        compile_s = time.perf_counter() - t0
        uj, vj = jax.numpy.asarray(u * wl), jax.numpy.asarray(v * wl)
        row = {"virgil_first_call_s": compile_s, "virgil_jit_s": timeit(lambda: vm(uj, vj).block_until_ready())}
        from types import SimpleNamespace

        frozen = disk.freeze_geometry(SimpleNamespace(u=u * wl, v=v * wl, wavel=wl))
        vf_ = jax.jit(lambda uu, vv: frozen.model(uu, vv, wl))
        vf_(uj, vj).block_until_ready()
        row["virgil_frozen_jit_s"] = timeit(lambda: vf_(uj, vj).block_until_ready())
        row["virgil_frozen_vs_unfrozen"] = float(np.abs(np.asarray(vf_(uj, vj)) - np.asarray(vm(uj, vj))).max())
        r_max = 30 * s
        row["frank_cold_s"] = timeit(lambda: X.Frank(r_max, 300, qd).visibility(f), reps=5)
        fr = X.Frank(r_max, 300, qd)
        row["frank_hot_s"] = timeit(lambda: fr.visibility(f))
        row["frank_matvec_only_s"] = timeit(lambda: fr.matrix @ f(fr.r_mas))
        row["frap_cold_s"] = timeit(lambda: X.Frap(r_max, 300, qd), reps=3)
        fp = X.Frap(r_max, 300, qd)
        row["frap_hot_s"] = timeit(lambda: fp.visibility(f))
        t0 = time.perf_counter()
        vr = X.VisR(u, v, INC, PA, sz=8.84 if s >= 100 else 0.5)
        row["visr_setup_s"] = time.perf_counter() - t0
        row["visr_hot_s"] = timeit(lambda: vr.visibility(lambda r: vf.r_prof_gauss(r, [10 * a, 1.5 * a])))
        out[str(n)] = row
    return out


def main():
    out = {"inc_deg": INC, "pa_deg": PA}
    out["accuracy_mm_scale"] = regime("mm", 100.0, 8e5, 3000, True, 1e-3)
    out["accuracy_ir_scale"] = regime("ir", 1.0, 8e7, 3000, False, 1.65e-6)
    out["timing_mm_scale"] = timing(100.0, 8e5, 1e-3)
    out["virgil_n_r_convergence"] = convergence()
    path = save("v1_forward", out)
    print(json.dumps({k: v for k, v in out.items()}, indent=1, default=float))
    print("wrote", path)


def convergence():
    """virgil's error against truth as n_r grows (IR scale), per scene."""
    u, v = R.baselines(2000, q_max=8e7, seed=3)
    out = {}
    for name in ("gauss_ring", "erf_power", "gap_on_plateau", "exp_tail"):
        _, prof, _, _, _ = scenes(1.0)[name]
        truth = R.disk_visibility(prof, u, v, INC, PA)
        row = {}
        for n_r in ((16, 32, 64, 128, 150) if name == "exp_tail" else (16, 32, 64, 128, 256)):  # Gauss-Laguerre: at most 150
            disk0 = scenes(1.0)[name][0]
            disk = PolarDisk(disk0.profile, inc=INC, pa=PA, n_r=n_r)
            row[n_r] = float(np.abs(np.asarray(disk.model(u * 1.65e-6, v * 1.65e-6, 1.65e-6)).real - truth).max())
        out[name] = row
    return out


if __name__ == "__main__":
    main()
