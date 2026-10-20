"""The Gl 229 release gate on the aliases page is recomputed from the stored records, not copied or inferred."""

import json
import pathlib
import shutil
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import build_aliases_page as B  # noqa: E402
import pull_back_orbit_results as P  # noqa: E402

pytestmark = pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")


@pytest.fixture
def assets(tmp_path):
    for n in ("gl229", "gl229-cold", "gl229-cold-effort", "gl229-warm-2946041", "gl229-pipeline"):
        shutil.copytree(B.ASSETS / n, tmp_path / n, ignore=shutil.ignore_patterns("*.png"))
    return tmp_path


def _edit(assets, name, fn):
    p = assets / name / "meta.json"
    m = json.load(open(p))
    fn(m)
    json.dump(m, open(p, "w"))


def _gate(assets):
    return B.gate_section(assets, "../assets")


def test_cold_is_read_from_the_run_record():
    assert B.is_cold({"args": {"no_init": True}, "init": None})
    assert not B.is_cold({"args": {"no_init": True}, "init": {"period": 12.1}})
    assert not B.is_cold({"args": {"no_init": False}})
    assert not B.is_cold({})  # no record: not known to be cold


def test_a_warm_run_filed_as_cold_cannot_pass(assets):
    def warm(m):
        m["args"]["no_init"] = False
    _edit(assets, "gl229-cold", warm)
    assert "not recorded as cold" in _gate(assets)
    text = _gate(assets)
    assert "**FAIL.** `reanalyse_orbits.py" in text


def test_the_cold_gate_needs_the_n34_sky_verdict(assets):
    """A run that puts N = 34 first by a wide margin but with a different orbit on the sky does not pass."""
    p = assets / "gl229-cold" / "bands.json"
    b = json.load(open(p))
    for x in b["bands"]:
        x["log_z"] += 1e4 if x["n"] == 34 else 0
    json.dump(b, open(p, "w"))
    text = _gate(assets)
    cold = text[text.index("### Cold orbit_aliases"):]
    assert "N = 34 first by more than" in cold and "**FAIL.**" in cold  # DIFFERENT on the sky


def test_the_blind_pass_is_recomputed_and_needs_the_sky_plane(assets):
    m = json.load(open(assets / "gl229-pipeline" / "meta.json"))
    ref = json.load(open(assets / "gl229" / "meta.json"))
    r = B.pipeline_gate(m, ref)
    assert r["virgil_ok"] and r["margin_ok"] and r["sky_verdict"] == "DIFFERENT" and not r["ok"]
    m["gate"]["delta_log_z"] = 5.0  # margin below the gate although virgil's string still says PASS
    assert not B.pipeline_gate(m, ref)["margin_ok"]
    r = B.pipeline_gate(m, {})  # no reference: the sky plane cannot be checked, so no PASS
    assert r["sky_verdict"] is None and not r["ok"]


def test_the_blind_section_shows_virgils_warnings_version_and_data(assets):
    text = _gate(assets)
    for needle in ("**tempering** (warn)", "**night_flags** (warn)", "f = 41.1", "virgil 0.3.0 commit `dda910f`",
                   "not the 0.4.0 release code", "GRAVI.2023-12-26T05_51_29.891"):
        assert needle in text, needle
    assert B.NOT_RECORDED in text and "a92b97c" not in text


def test_the_pipeline_version_is_not_called_the_release_when_it_is_not(assets):
    _edit(assets, "gl229-pipeline", lambda m: m["virgil"].update(version="0.4.0"))
    assert "release version 0.4.0" in _gate(assets)


def test_the_diagnosis_is_computed_from_the_runs(assets):
    assert "The likelihood is unchanged" in "\n".join(B.diagnosis(assets))
    shutil.rmtree(assets / "gl229-cold-effort")
    shutil.rmtree(assets / "gl229-warm-2946041")
    d = "\n".join(B.diagnosis(assets))
    assert "Diagnosis not established" in d and "unchanged" not in d and "seeds the wrong modes" not in d
    assert "Diagnosis not established" in _gate(assets)


def test_a_changed_likelihood_is_reported_not_asserted_unchanged(assets):
    p = assets / "gl229-warm-2946041" / "bands.json"
    b = json.load(open(p))
    for x in b["bands"]:
        if x["n"] == 34:
            x["log_z"] -= 50
    json.dump(b, open(p, "w"))
    d = "\n".join(B.diagnosis(assets))
    assert "likelihood changed" in d and "seeds the wrong modes" not in d


def test_a_cold_run_without_an_n34_band_does_not_crash(assets):
    p = assets / "gl229-cold" / "bands.json"
    b = json.load(open(p))
    b["bands"] = [x for x in b["bands"] if x["n"] != 34]
    json.dump(b, open(p, "w"))
    assert "absent from this run" in _gate(assets) and "no N = 34 band" in _gate(assets)


def test_the_commit_and_window_come_from_the_records(assets):
    _edit(assets, "gl229-cold", lambda m: (m["virgil"].update(commit_id="abcdef1234"), m.update(p_range=[9.0, 15.0])))
    text = _gate(assets)
    assert "`abcdef1" in text and "window 9 to 15 d" in text


def test_pull_back_files_a_run_by_its_own_record(tmp_path):
    def run(name, no_init, init=None):
        d = tmp_path / name / "gl229"
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({"args": {"no_init": no_init}, "init": init}))
        return d
    assert P.destination(run("tag-cold", False, {"period": 12.1})) == "gl229"  # warm, though the tag says cold
    assert P.destination(run("tag-warm", True)) == "gl229-cold"  # cold, though the tag does not say so
    nometa = tmp_path / "x" / "gl229"
    nometa.mkdir(parents=True)
    assert P.destination(nometa) is None
