"""Light checks of the orbit comparison files (no sampling): data manifest, Kepler constant, gate results, cases.yml."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import common  # noqa: E402
import make_data as md  # noqa: E402


def test_data_matches_manifest():
    assert common.verify_manifest(HERE / "manifest.json", md.DATA) == []


def test_kepler_third_law():
    # 1 au, 1 Msun: P = 365.25 d would give a = 1 au; IAU nominal constants give 365.2568.. d for GM = 1.3271244e20
    a_au = (1.0 * 365.2568983**2 / md.KEPLER_K) ** (1.0 / 3.0)
    assert abs(a_au - 1.0) < 1e-6
    assert abs(md.a_mas(P=200.0, M_tot=2.0, plx=20.0) - 20.0 * (2.0 * 200.0**2 / md.KEPLER_K) ** (1 / 3)) < 1e-12


def test_gates_recorded_and_passing():
    r = json.loads((HERE / "gates_result.json").read_text())
    for k in ("gate1_virgil_pass", "gate1_octofitter_pass", "gate2_virgil_pass", "gate2_octofitter_pass"):
        assert r[k] is True, k
    assert r["gate3_octofitter_rel_diff_vs_neutral"] < 1e-8
    assert r["gate4_reader"]["pass"] is True and r["gate4_reader"]["index_zero"] == 0


def test_cases_yaml_parses():
    import yaml

    c = yaml.safe_load((HERE / "cases.yml").read_text())
    assert set(c["cases"]) == {"O-A-sim", "O-A-sim-pulls", "O-A-real", "O-B-sim"}
    assert c["code_versions"]["octofitter"].startswith("9.0.0")


def test_oareal_starts_inside_prior_and_init_finite():
    """O-A-real: every start has lo < P < hi (the +-1 sigma start window, not the two prior bounds) and numpyro can initialise from it."""
    import numpy as np

    import run_virgil as rv
    from virgil.likelihood import chain_init_params
    from virgil.orbits import period_grid, starting_orbits

    info = rv.case_info("O-A-real")
    lo, hi = info["p_range"]
    ds = rv.read_astrometry(rv.DATA / "astrometry_O-A-real.csv")
    pos = rv.positions_from(ds, info["t_ref"])
    starts = starting_orbits(pos, period_grid(ds["epoch"], *info["p_start"]), n_best=4)
    for phase in ("M0", "theta"):
        model, _, _ = rv.level_a_model(pos, info["plx"], info["t_ref"], info["p_prior"], phase)
        vals = [rv.start_values(o, info["plx"], phase, info["t_ref"], info["p_range"]) for o, _ in starts]
        assert all(lo < v["P"] < hi for v in vals)
        init = chain_init_params(model, vals)
        assert all(np.all(np.isfinite(np.asarray(x))) for x in init.values())
