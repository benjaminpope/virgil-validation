"""Gates 1-4 of the orbit comparison (plan section 5, "Gates"), virgil and neutral sides; Octofitter's numbers come from
``gate_octofitter.jl`` (``DATA/gates/octofitter_out.json``).

    gates.py prepare     write DATA/gates/gate_in.json (element sets, epochs, parameter points) for the Julia script
    gates.py virgil      virgil and neutral numbers for the same inputs -> DATA/gates/virgil_out.json
    gates.py compare     compare everything, write orbit/gates_result.json and print a summary

Independence: this file imports virgil (and crosscheck / neutral), never Octofitter; the Julia script never imports
virgil. Octofitter's angles are radians; the mapping of (omega, Omega, phase) found by gate 1 is in cases.yml.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "src"))

import make_data as md  # noqa: E402
import neutral  # noqa: E402

GATE_DIR = md.DATA / "gates"
RESULT = HERE / "gates_result.json"

ELEMENTS = [  # P [d], M_tot [Msun], plx [mas], e, inc, omega, Omega, M0 [deg]; includes i > 90 deg and e = 0.6
    dict(P=200.0, M_tot=2.0, plx=20.0, e=0.34, inc=53.4, omega=76.9, Omega=28.2, M0=120.3),
    dict(P=173.0, M_tot=1.3, plx=20.0, e=0.20, inc=124.0, omega=210.0, Omega=95.0, M0=15.0),
    dict(P=240.0, M_tot=3.1, plx=20.0, e=0.60, inc=75.0, omega=330.0, Omega=181.0, M0=250.0),
    dict(P=156.0, M_tot=0.8, plx=20.0, e=0.05, inc=96.0, omega=300.0, Omega=250.0, M0=300.0),
    dict(P=222.0, M_tot=4.2, plx=20.0, e=0.80, inc=30.0, omega=10.0, Omega=340.0, M0=75.0),
]
POINT_A = [  # two fixed points for the likelihood gates (on the O-A-sim data)
    dict(P=200.0, M_tot=2.0, plx=20.0, e=0.34, inc=53.4, omega=76.9, Omega=28.2, M0=120.3),
    dict(P=201.5, M_tot=2.1, plx=20.0, e=0.30, inc=55.0, omega=80.0, Omega=30.0, M0=118.0),
]
POINT_B = [dict(p, flux=f) for p, f in zip(POINT_A, (0.25, 0.22))]
T_ELEM = (md.T_REF + np.linspace(0.0, 400.0, 20)).tolist()


def sky(t, p):
    """Crosscheck (neutral side) sky position; make_data takes M0 in radians, the gate tables use degrees."""
    q = {k: p[k] for k in ("P", "M_tot", "plx", "e", "inc", "omega", "Omega")}
    return md.sky_position(t, M0=np.deg2rad(p["M0"]), **q)


def prepare():
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    (GATE_DIR / "gate_in.json").write_text(json.dumps(
        {"t_ref": md.T_REF, "elements": ELEMENTS, "epochs": T_ELEM, "point_a": POINT_A, "point_b": POINT_B}, indent=1))
    print("wrote", GATE_DIR / "gate_in.json")


def virgil_orbit(p, t_ref=md.T_REF):
    import jax.numpy as jnp

    from virgil.orbits import KeplerOrbit

    a = md.a_mas(P=p["P"], M_tot=p["M_tot"], plx=p["plx"])
    dt_peri = -np.deg2rad(p["M0"]) * p["P"] / (2 * np.pi)
    return KeplerOrbit(p["P"], dt_peri, p["e"], p["inc"], p["omega"], p["Omega"], a, t_ref)


# ---- neutral Level B reference ------------------------------------------------------------------------------------


def read_epoch_file(path):
    """V2 and closure phase data of one OIFITS epoch (astropy), with the (u, v) of every sample."""
    from astropy.io import fits

    with fits.open(path) as h:
        wl = np.asarray(h["OI_WAVELENGTH"].data["EFF_WAVE"], float)
        v = h["OI_VIS2"].data
        t = h["OI_T3"].data
        return dict(wl=wl, u=np.asarray(v["UCOORD"], float), v=np.asarray(v["VCOORD"], float),
                    v2=np.asarray(v["VIS2DATA"], float), v2err=np.asarray(v["VIS2ERR"], float),
                    sta=np.asarray(v["STA_INDEX"]), t_sta=np.asarray(t["STA_INDEX"]),
                    u1=np.asarray(t["U1COORD"], float), v1=np.asarray(t["V1COORD"], float),
                    u2=np.asarray(t["U2COORD"], float), v2c=np.asarray(t["V2COORD"], float),
                    cp=np.deg2rad(np.asarray(t["T3PHI"], float)), cperr=np.deg2rad(np.asarray(t["T3PHIERR"], float)))


def neutral_chi2_level_b(p, flux):
    """Sum over the 8 epochs of the independent-Gaussian chi2 (V2 + wrapped closure phase), neutral formulas."""
    import csv

    with open(md.DATA / "epochs_O-B-sim.csv") as f:
        meta = list(csv.DictReader(f))
    tot = dict(chi2=0.0, chi2_v2=0.0, chi2_cp=0.0, n_v2=0, n_cp=0)
    for r in meta:
        d = read_epoch_file(md.DATA / "oifits_O-B-sim" / r["file"])
        ra, dec = (float(x[0]) for x in sky(np.array([float(r["epoch"])]), p))
        wl = d["wl"][None, :]

        def vis(u, v):
            return neutral.binary_vis(u[:, None], v[:, None], wl, ra, dec, flux)

        mv2 = np.abs(vis(d["u"], d["v"])) ** 2
        cp = neutral.closure_phase(vis(d["u1"], d["v1"]), vis(d["u2"], d["v2c"]), vis(-d["u1"] - d["u2"], -d["v1"] - d["v2c"]))
        c = neutral.chi2_v2_cp(d["v2"], d["v2err"], mv2, d["cp"], d["cperr"], cp)
        for k in tot:
            tot[k] += c[k]
    return tot


def neutral_loglike_a(p, data_path=md.DATA / "astrometry_O-A-sim.csv"):
    a = np.genfromtxt(data_path, delimiter=",", names=True)
    mra, mdec = sky(a["epoch"], p)
    return neutral.astrometry_loglike(a["ra"], a["dec"], mra, mdec, a["sigma_ra"], a["sigma_dec"], a["cor"])


def run_virgil():
    import jax

    jax.config.update("jax_enable_x64", True)
    import equinox as eqx

    sys.path.insert(0, str(HERE))
    import run_virgil as rv

    gin = json.loads((GATE_DIR / "gate_in.json").read_text())
    t = np.asarray(gin["epochs"])
    out = {"positions": [], "loglike_a": [], "loglike_b_indep": [], "loglike_b_default": []}
    for p in ELEMENTS:
        ra, dec, _ = virgil_orbit(p).relative(t)
        out["positions"].append({"ra": np.asarray(ra).tolist(), "dec": np.asarray(dec).tolist()})
    pos = rv.positions_from(rv.read_astrometry(md.DATA / "astrometry_O-A-sim.csv"), md.T_REF)
    for p in POINT_A:
        out["loglike_a"].append(float(pos.loglike(virgil_orbit(p))))
    from virgil.models import Attached, PointSource, System

    for indep, key in ((True, "loglike_b_indep"), (False, "loglike_b_default")):
        ep = rv.read_epochs(independent_cp=indep)
        for p in POINT_B:
            orbit = virgil_orbit(p)
            scene = System(primary=PointSource(), comp=Attached(PointSource(p["flux"]), orbit))
            out[key].append(float(ep.loglike(scene)))
    out["neutral_a"] = [neutral_loglike_a(p) for p in POINT_A]
    out["neutral_b"] = [neutral_chi2_level_b(p, p_["flux"]) for p, p_ in zip(POINT_A, POINT_B)]
    out["crosscheck_positions"] = []
    for p in ELEMENTS:
        ra, dec = sky(t, p)
        out["crosscheck_positions"].append({"ra": np.asarray(ra).tolist(), "dec": np.asarray(dec).tolist()})
    (GATE_DIR / "virgil_out.json").write_text(json.dumps(out, indent=1))
    print("wrote", GATE_DIR / "virgil_out.json")


def reader_gate(raw):
    """Gate 4: the arrays the Octofitter OIFITS reader stored, against astropy on the same files."""
    import csv

    with open(md.DATA / "epochs_O-B-sim.csv") as f:
        meta = list(csv.DictReader(f))
    worst = dict(vis2=0.0, dvis2=0.0, cp_deg=0.0, dcp_deg=0.0, u_m=0.0, v_m=0.0, tri_u=0.0, tri_v=0.0)
    zero_index, bad_index = 0, 0
    for r, o in zip(meta, raw):
        d = read_epoch_file(md.DATA / "oifits_O-B-sim" / r["file"])
        assert abs(o["epoch"] - float(r["epoch"])) < 1e-9
        pairs = dict(vis2=(o["vis2"], d["v2"]), dvis2=(o["dvis2"], d["v2err"]), cp_deg=(o["cp_deg"], np.rad2deg(d["cp"])),
                     dcp_deg=(o["dcp_deg"], np.rad2deg(d["cperr"])))
        for k, (a, b) in pairs.items():
            worst[k] = max(worst[k], float(np.abs(np.array(a) - b).max()))
        worst["u_m"] = max(worst["u_m"], float(np.abs(np.array(o["u_m"]) - d["u"][:, None]).max()))
        worst["v_m"] = max(worst["v_m"], float(np.abs(np.array(o["v_m"]) - d["v"][:, None]).max()))
        i1, i2, i3 = (np.array(o[k]) for k in ("idx1", "idx2", "idx3"))
        zero_index += int(sum((x == 0).sum() for x in (i1, i2, i3)))
        n = len(d["u"])
        bad_index += int(sum(((x < 1) | (x > n)).sum() for x in (i1, i2, i3)))
        if not bad_index and not zero_index:
            u, v = np.array(o["u_m"])[:, 0], np.array(o["v_m"])[:, 0]
            worst["tri_u"] = max(worst["tri_u"], float(np.abs(u[i1 - 1] - d["u1"]).max()), float(np.abs(u[i2 - 1] - d["u2"]).max()))
            worst["tri_v"] = max(worst["tri_v"], float(np.abs(v[i1 - 1] - d["v1"]).max()), float(np.abs(v[i2 - 1] - d["v2c"]).max()))
    worst["index_zero"], worst["index_out_of_range"] = zero_index, bad_index
    worst["pass"] = bool(all(worst[k] < 1e-6 for k in ("vis2", "dvis2", "cp_deg", "dcp_deg", "u_m", "v_m", "tri_u", "tri_v"))
                         and not zero_index and not bad_index)
    return worst


def compare():
    v = json.loads((GATE_DIR / "virgil_out.json").read_text())
    oc_path = GATE_DIR / "octofitter_out.json"
    oc = json.loads(oc_path.read_text()) if oc_path.exists() else None
    res = {}

    def maxdiff(a, b):
        return max(max(np.abs(np.array(x["ra"]) - np.array(y["ra"])).max(), np.abs(np.array(x["dec"]) - np.array(y["dec"])).max())
                   for x, y in zip(a, b))

    res["gate1_virgil_vs_crosscheck_max_abs_mas"] = maxdiff(v["positions"], v["crosscheck_positions"])
    res["gate1_virgil_pass"] = bool(res["gate1_virgil_vs_crosscheck_max_abs_mas"] < 1e-6)
    d_neu_a = v["neutral_a"][1] - v["neutral_a"][0]
    d_vir_a = v["loglike_a"][1] - v["loglike_a"][0]
    res["gate2_virgil_delta_loglike"] = d_vir_a
    res["gate2_neutral_delta_loglike"] = d_neu_a
    res["gate2_virgil_abs_diff"] = abs(d_vir_a - d_neu_a)
    res["gate2_virgil_rel_diff"] = abs(d_vir_a - d_neu_a) / abs(d_neu_a)
    res["gate2_virgil_pass"] = bool(res["gate2_virgil_rel_diff"] < 1e-8)
    d_neu_b = -0.5 * (v["neutral_b"][1]["chi2"] - v["neutral_b"][0]["chi2"])
    for k in ("indep", "default"):
        d = v[f"loglike_b_{k}"][1] - v[f"loglike_b_{k}"][0]
        res[f"gate3_virgil_{k}_delta_loglike"] = d
        res[f"gate3_virgil_{k}_rel_diff_vs_neutral"] = abs(d - d_neu_b) / abs(d_neu_b)
    res["gate3_neutral_delta_loglike"] = d_neu_b
    res["gate3_neutral_chi2"] = [x["chi2"] for x in v["neutral_b"]]
    if oc:
        res["gate1_octofitter_vs_crosscheck_max_abs_mas"] = maxdiff(oc["positions"], v["crosscheck_positions"])
        res["gate1_octofitter_pass"] = bool(res["gate1_octofitter_vs_crosscheck_max_abs_mas"] < 1e-6)
        res["octofitter_mapping"] = oc.get("mapping")
        if "loglike_a" in oc:
            d = oc["loglike_a"][1] - oc["loglike_a"][0]
            res["gate2_octofitter_delta_loglike"] = d
            res["gate2_octofitter_rel_diff"] = abs(d - d_neu_a) / abs(d_neu_a)
            res["gate2_octofitter_pass"] = bool(res["gate2_octofitter_rel_diff"] < 1e-8)
        if "loglike_b" in oc:
            d = oc["loglike_b"][1] - oc["loglike_b"][0]
            res["gate3_octofitter_delta_loglike"] = d
            res["gate3_octofitter_rel_diff_vs_neutral"] = abs(d - d_neu_b) / abs(d_neu_b)
        if "reader_raw" in oc:
            res["gate4_reader"] = reader_gate(oc["reader_raw"])
        if "gate4_error" in oc:
            res["gate4_error"] = oc["gate4_error"][:500]
    RESULT.write_text(json.dumps(res, indent=1, default=float))
    for k, x in res.items():
        print(f"{k}: {x}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "virgil", "compare"])
    a = ap.parse_args()
    {"prepare": prepare, "virgil": run_virgil, "compare": compare}[a.step]()
