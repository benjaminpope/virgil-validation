"""scripts/check_skips.py: a skipped check with a `validates` marker fails CI unless the allow-list says
why it may skip (trust/skip_allowlist.yml). Synthetic evidence files, both outcomes."""

import importlib.util
import json
import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")


@pytest.fixture(scope="module")
def cs():
    spec = importlib.util.spec_from_file_location("check_skips", ROOT / "scripts" / "check_skips.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence(tmp_path, *records):
    path = tmp_path / "evidence.jsonl"
    head = {"record": "run", "date": "2026-11-19T00:00:00+00:00"}
    path.write_text("\n".join(json.dumps(x) for x in [head, *records]) + "\n")
    return path


def rec(test, outcome="skipped", reason="the data file is missing", **extra):
    return {"record": "test", "test": test, "objects": ["virgil.a"], "roots": ["mathematics"], "kind": "check",
            "outcome": outcome, **({"reason": reason} if reason else {}), **extra}


def allowlist(tmp_path, pr=(), weekly=()):
    path = tmp_path / "skip_allowlist.yml"
    path.write_text(yaml.safe_dump({"pr": list(pr), "weekly": list(weekly)}))
    return path


def entry(test, because="data", why="the file is not in CI", message="file is missing", **extra):
    return {"test": test, "because": because, "why": why, "message": message, **extra}


def run(cs, tmp_path, records, pr=(), weekly=(), job="pr", capsys=None):
    code = cs.main([str(evidence(tmp_path, *records)), "--job", job, "--allowlist", str(allowlist(tmp_path, pr, weekly))])
    return code, capsys.readouterr().out if capsys else ""


def test_a_skipped_check_off_the_list_fails_and_is_named(cs, tmp_path, capsys):
    code, out = run(cs, tmp_path, [rec("t.py::test_a"), rec("t.py::test_b", reason="ruff is not installed")], capsys=capsys)
    assert code == 1
    assert "t.py::test_a" in out and "t.py::test_b" in out and "ruff is not installed" in out


def test_passed_failed_and_xfailed_checks_are_not_skips(cs, tmp_path, capsys):
    records = [rec("t.py::test_a", "passed"), rec("t.py::test_b", "failed"), rec("t.py::test_c", "xfailed")]
    assert run(cs, tmp_path, records, capsys=capsys)[0] == 0


def test_a_listed_skip_passes(cs, tmp_path, capsys):
    code, out = run(cs, tmp_path, [rec("t.py::test_a")], pr=[entry("t.py::test_a")], capsys=capsys)
    assert code == 0 and "1 on the pr allow-list" in out


def test_each_job_has_its_own_list(cs, tmp_path, capsys):
    records = [rec("t.py::test_a")]
    assert run(cs, tmp_path, records, pr=[entry("t.py::test_a")], job="weekly", capsys=capsys)[0] == 1
    assert run(cs, tmp_path, records, weekly=[entry("t.py::test_a")], job="weekly", capsys=capsys)[0] == 0
    assert run(cs, tmp_path, records, weekly=[entry("t.py::test_a")], job="pr", capsys=capsys)[0] == 1


def test_a_glob_matches_runs_of_characters_and_brackets_are_literal(cs, tmp_path, capsys):
    listed = [entry("t.py::test_cases[*]"), entry("u.py::test_x[True]")]
    records = [rec("t.py::test_cases[a-b]"), rec("t.py::test_cases[c]"), rec("u.py::test_x[True]")]
    assert run(cs, tmp_path, records, pr=listed, capsys=capsys)[0] == 0
    # [True] is not a character class: it does not match "T", "r", "u" or "e", only the id as written
    code, out = run(cs, tmp_path, [rec("u.py::test_xT")], pr=listed, capsys=capsys)
    assert code == 1 and "u.py::test_xT" in out
    assert run(cs, tmp_path, [rec("t.py::test_cases")], pr=listed, capsys=capsys)[0] == 1  # a glob is not a prefix


def test_an_entry_that_states_a_message_covers_only_a_skip_that_says_it(cs, tmp_path, capsys):
    listed = [entry("t.py::test_a", message="MS missing")]
    assert run(cs, tmp_path, [rec("t.py::test_a", reason="casatools or the IM Lup MS missing")], pr=listed, capsys=capsys)[0] == 0
    code, out = run(cs, tmp_path, [rec("t.py::test_a", reason="No module named 'casatasks'")], pr=listed, capsys=capsys)
    assert code == 1 and "No module named" in out
    assert run(cs, tmp_path, [rec("t.py::test_a", reason="")], pr=listed, capsys=capsys)[0] == 1  # no recorded reason: not covered


def test_a_test_with_several_claims_counts_once(cs, tmp_path, capsys):
    records = [rec("t.py::test_a"), {**rec("t.py::test_a"), "objects": ["virgil.b"]}]
    code, out = run(cs, tmp_path, records, capsys=capsys)
    assert code == 1 and out.count("t.py::test_a\n") == 1 and "1 skipped" in out


def test_a_stale_entry_is_reported_but_does_not_fail(cs, tmp_path, capsys):
    code, out = run(cs, tmp_path, [rec("t.py::test_a", "passed")], pr=[entry("t.py::test_a")], capsys=capsys)
    assert code == 0 and "stale" in out and "t.py::test_a" in out


@pytest.mark.parametrize(
    "bad",
    [
        entry("t.py::test_a", because="slow"),
        entry("t.py::test_a", because="it is flaky"),
        entry("t.py::test_a", why=""),
        {"test": "t.py::test_a", "because": "data", "message": "missing"},
        entry("t.py::test_a", message=""),
        entry("*"),  # covers every skip
        entry("tests/*"),
        entry("t.py::*"),
        entry("*::test_a"),
        entry("t.py"),
    ],
)
def test_an_entry_needs_an_allowed_reason_a_why_a_message_and_a_test_that_is_named(cs, tmp_path, bad):
    with pytest.raises(ValueError):
        cs.load_allowlist(allowlist(tmp_path, pr=[bad]), "pr")


def test_unknown_section_is_an_error(cs, tmp_path):
    path = tmp_path / "skip_allowlist.yml"
    path.write_text("nightly: []\n")
    with pytest.raises(ValueError):
        cs.load_allowlist(path, "pr")


def test_the_committed_allowlist_is_well_formed(cs):
    for job in cs.JOBS:
        for e in cs.load_allowlist(cs.ALLOWLIST, job):
            assert e["test"].startswith("tests/") and "::" in e["test"], e
