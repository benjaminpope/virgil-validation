"""The two campaigns: in step with their sources, valid, and with nothing published in the open."""

import importlib.util
import json
import pathlib
import re
import subprocess
import sys

import pytest

from mandate import scaffold, work
from mandate.model import Campaign, read_notes
from mandate.orders import expand, find

HERE = pathlib.Path(__file__).resolve().parents[1]
REPO = HERE.parent


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_generated_files_match_their_sources():
    done = subprocess.run([sys.executable, str(HERE / "build_registries.py"), "--check"], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("name", ["orbits", "contests"])
def test_the_plan_is_valid_and_reads_back(name):
    c = Campaign(HERE / name)
    assert c.problems() == []
    assert c.plan["campaign"] in scaffold.explain(c)


def test_the_orbit_campaign_is_one_blind_fit_per_system_then_a_sealed_comparison():
    c = Campaign(HERE / "orbits")
    orders = expand(c)
    slugs = sorted(p.stem for p in (REPO / "oidb" / "trackb").glob("*.json"))
    assert [o.key_str for o in orders["orbit"]] == sorted([*slugs, "iota_peg"])
    assert c.stages["compare"].get("sealed") and not c.stages["orbit"].get("sealed")
    # No decision of the fit carries a number: there is nothing to put a published value into.
    for d in c.stages["orbit"]["decisions"].values():
        assert not any(re.search(r"\d", str(text)) for text in d["options"].values())


def published_numbers(record):
    """The published orbit's values as they would be typed: period, eccentricity, semi-major axis, inclination."""
    out = set()
    for key in ("P_day", "e", "a_mas", "i_deg", "omega_deg", "Omega_deg"):
        value = (record.get("orbit") or {}).get(key)
        value = value[0] if isinstance(value, list) else value
        if isinstance(value, (int, float)) and value not in (0, 0.0):
            out.add(f"{value:g}")
    return {v for v in out if len(v.replace(".", "")) >= 3}  # "90" or "0.5" would match anything


def test_nothing_a_blind_orbit_fit_is_given_contains_the_published_orbit():
    c = Campaign(HERE / "orbits")
    for path in sorted((REPO / "oidb" / "trackb").glob("*.json")):
        record = json.loads(path.read_text())
        order = find(c, f"orbit/{record['slug']}")
        given = work.brief(c, order) + json.dumps([c.open_row(u) for u in order.units])
        assert "published_kind" not in json.dumps([c.open_row(u) for u in order.units])
        assert f"sealed_{record['slug']}.md" not in given
        for number in published_numbers(record):
            assert not re.search(rf"(?<![\d.]){re.escape(number)}(?![\d.])", given), (record["slug"], number)
        sealed = [n["file"] for n in read_notes(c, order.units, "compare", sealed=True)]
        assert f"sealed_{record['slug']}.md" in sealed  # and the comparison does get it


def test_the_contest_campaign_runs_eight_members_per_dataset_and_keeps_truths_sealed():
    c = Campaign(HERE / "contests")
    orders = expand(c)
    assert len(orders["member"]) == 8 * len(orders["ensemble"]) and len(orders["score_2004"]) == 1
    member = find(c, "member/2004_data2_m0", orders)
    text = work.brief(c, member)
    assert "given_2004.md" in text and "sealed_2004.md" not in text
    assert "limb-darkened" not in text and "BSMEM" not in text  # the revealed target and the winner
    assert "sealed_2004.md" in work.brief(c, find(c, "score_2004/2004_data2", orders))
    stars = {o.units[0]["member_star"] for o in orders["member"] if o.units[0]["label"] == "2008_agb_J"}
    assert stars == {"point", "none", "disk"}  # every dataset is imaged with and without a central star


def test_a_members_result_comes_from_its_own_file(tmp_path):
    np = pytest.importorskip("numpy")
    step = load(HERE / "contests" / "steps" / "member_result.py", "member_result")
    assert step.result_of(tmp_path / "missing.npz")["checks"] == {"finished": "fail"}
    np.savez(tmp_path / "good.npz", best_chi2_red=0.93, best_log_z=-120.5, error_scale=1.1, image_iso=np.zeros((3, 3)))
    assert step.result_of(tmp_path / "good.npz")["status"] == "pass"
    np.savez(tmp_path / "poor.npz", best_chi2_red=21.0, best_log_z=-9e3, error_scale=4.7)
    poor = step.result_of(tmp_path / "poor.npz")
    assert poor["status"] == "warn" and poor["checks"]["finished"] == "pass" and poor["best_chi2_red"] == 21.0
    np.savez(tmp_path / "nan.npz", best_chi2_red=float("nan"))
    assert step.result_of(tmp_path / "nan.npz")["status"] == "fail"


def test_the_ensemble_step_reads_its_datasets_row():
    step = load(HERE / "contests" / "steps" / "combine.py", "combine_step")
    table = "| dataset | kept | best χ²/N |\n|---|---|---|\n| 2008_agb_J | 6/8 | 0.70 |\n| 2010_lowH | 1/8 | 2512 |\n"
    assert step.number(step.row_of(table, "2008_agb_J")[1]) == 6 and step.number(step.row_of(table, "2010_lowH")[2]) == 2512
    assert step.row_of(table, "2024_obj1_pionier") is None


def test_the_informed_track_is_given_what_the_paper_says_the_object_is_and_none_of_its_values():
    c = Campaign(HERE / "contests")
    orders = expand(c)
    assert sorted(o.key_str for o in orders["informed_fit"]) == ["2004_data2", "2018_disk"]  # the built families
    assert c.level("informed_fit") == "informed" and c.level("member") == "open"
    fit = find(c, "informed_fit/2004_data2", orders)
    assert fit.decisions["model"]["value"] == "spotted_star_binary"
    assert find(c, "informed_fit/2018_disk", orders).decisions["model"] == {"value": "star_rim_planet_disk", "source": "rule:0"}
    text = work.brief(c, fit)
    assert "limb-darkened elliptical disk with one bright spot" in text and "INFORMED stage" in text
    assert "sealed_2004.md" not in text and "5.5 mas" not in text and "PA 90" not in text  # the paper's values
    # The blind members of the same dataset are given none of it.
    member = find(c, "member/2004_data2_m0", orders)
    assert "paper_model" not in json.dumps(work.order_record(c, member, "x")["units"])
    assert "informed_2004.md" not in work.brief(c, member) and "spot" not in work.brief(c, member)
    # No number in the informed table: structure only.
    table = (HERE / "contests" / "informed_models.yml").read_text()
    says = [line for line in table.splitlines() if line.strip().startswith("says:")]
    assert says and not any(re.search(r"\d", line) for line in says)


def test_the_informed_families_take_no_number_from_a_paper():
    source = (REPO / "scripts" / "contest_informed.py").read_text()
    families = source[source.index("FAMILIES = {"):source.index("# A value for the constructor")]
    code = "\n".join(line.split("#")[0] for line in families.splitlines())  # comments cite the papers; code must not
    assert not re.search(r"\d", code)  # prior kinds are words; their scales come from the data's baselines
