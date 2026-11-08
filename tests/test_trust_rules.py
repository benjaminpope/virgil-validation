"""The verdict rules of scripts/trust.py, one synthetic case per rule
(docs/method/index.md): what may verify a part of virgil, and what may not."""

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")


@pytest.fixture(scope="module")
def trust():
    spec = importlib.util.spec_from_file_location("trust", ROOT / "scripts" / "trust.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(test, obj, kind="check", outcome="passed", roots=("mathematics",), **extra):
    return {"record": "test", "test": test, "objects": [obj], "roots": list(roots), "kind": kind,
            "tier": "A", "outcome": outcome, "metrics": {}, **extra}


def evidence(tmp_path, name, records, date="2026-10-06T00:00:00+00:00", commit="c" * 40):
    path = tmp_path / name
    head = {"record": "run", "date": date, "virgil": {"commit": commit}, "runner": "test"}
    path.write_text("\n".join(json.dumps(x) for x in [head, *records]) + "\n")
    return path


def verdicts(trust, tmp_path, records, graph=None, ledger=(), files=None, golden=frozenset(), virgil=None):
    graph = graph or {"nodes": {"virgil.a": {"layer": "models"}}}
    paths = files or [evidence(tmp_path, "latest.jsonl", records)]
    runs, recs = trust.load_evidence(paths, virgil=virgil)
    model = trust.build(graph, list(ledger), recs, runs, None, golden=set(golden))
    return {k: v["verdict"] for k, v in model["nodes"].items()}, model


@pytest.mark.parametrize("kind", ["guard", "control", "reference", "definition"])
def test_only_checks_verify_a_part(trust, tmp_path, kind):
    got, _ = verdicts(trust, tmp_path, [record("t::x", "virgil.a", kind=kind)])
    assert got["virgil.a"] == "unchecked"
    got, _ = verdicts(trust, tmp_path, [record("t::x", "virgil.a")])
    assert got["virgil.a"] == "verified"


def test_regression_is_weak(trust, tmp_path):
    got, model = verdicts(trust, tmp_path, [record("t::x", "virgil.a", kind="regression")])
    assert got["virgil.a"] == "partly" and model["nodes"]["virgil.a"]["strong"] == []


def test_xfailed_check_does_not_count(trust, tmp_path):
    got, _ = verdicts(trust, tmp_path, [record("t::x", "virgil.a", outcome="xfailed")])
    assert got["virgil.a"] == "unchecked"


def test_virgils_own_tests_are_weak(trust, tmp_path):
    got, model = verdicts(trust, tmp_path, [record("t::x", "virgil.a", source="virgil-ci")])
    assert got["virgil.a"] == "partly" and model["nodes"]["virgil.a"]["weak"] == [trust.VIRGIL_CI]


def test_golden_sources_count_once_and_only_if_registered(trust, tmp_path):
    graph = {"nodes": {"virgil.a": {"layer": "models", "roots": 2}}}
    recs = [record("t::x", "virgil.a", roots=["golden:one"]), record("t::y", "virgil.a", roots=["golden:two"])]
    got, model = verdicts(trust, tmp_path, recs, graph, golden={"one", "two"})
    assert model["nodes"]["virgil.a"]["strong"] == [trust.GOLDEN] and got["virgil.a"] == "partly"
    got, _ = verdicts(trust, tmp_path, recs[:1], golden=set())
    assert got["virgil.a"] == "unchecked"


def test_newest_outcome_wins_but_a_campaign_survives_a_skip(trust, tmp_path):
    old = evidence(tmp_path, "old.jsonl", [record("t::x", "virgil.a")], date="2026-10-01")
    new = evidence(tmp_path, "new.jsonl", [record("t::x", "virgil.a", outcome="skipped")], date="2026-10-06")
    got, _ = verdicts(trust, tmp_path, None, files=[old, new])
    assert got["virgil.a"] == "unchecked"  # the old pass does not outlive the newer skip
    camp = evidence(tmp_path, "campaigns.jsonl", [record("t::x", "virgil.a")], date="2026-10-01")
    got, _ = verdicts(trust, tmp_path, None, files=[camp, new])
    assert got["virgil.a"] == "verified"  # a campaign summary is skipped by ordinary runs by design


@pytest.mark.parametrize("status,verdict", [("open", "bug"), ("to-raise", "bug"), ("raised", "bug"), ("fixed", "verified")])
def test_unfixed_virgil_findings_are_bugs(trust, tmp_path, status, verdict):
    ledger = [{"id": "F1", "ruling": "virgil", "status": status, "objects": ["virgil.a"]}]
    got, _ = verdicts(trust, tmp_path, [record("t::x", "virgil.a")], ledger=ledger)
    assert got["virgil.a"] == verdict


def test_open_external_problem_marks_the_package_partly(trust, tmp_path):
    graph = {"nodes": {"pkg": {"layer": "references"}}}
    ledger = [{"id": "P1", "ruling": "external:pkg", "status": "to-raise", "objects": ["pkg"]}]
    got, _ = verdicts(trust, tmp_path, [record("t::x", "pkg", kind="reference")], graph, ledger)
    assert got["pkg"] == "partly"


def test_references_are_verified_by_reference_checks_but_parts_of_virgil_are_not(trust, tmp_path):
    graph = {"nodes": {"crosscheck.x": {"layer": "references"}, "virgil.a": {"layer": "models"}}}
    recs = [record("t::r", "crosscheck.x", kind="reference"), record("t::s", "virgil.a", kind="reference")]
    got, _ = verdicts(trust, tmp_path, recs, graph)
    assert got == {"crosscheck.x": "verified", "virgil.a": "unchecked"}


def test_pipelines_count_only_checks(trust, tmp_path):
    graph = {"nodes": {"virgil.a": {"layer": "models"}}, "pipelines": {"p": {"title": "p", "steps": ["virgil.a"]}}}
    recs = [record("t::a", "virgil.a"), record("t::p", "pipeline:p", kind="reference")]
    _, model = verdicts(trust, tmp_path, recs, graph)
    assert model["pipelines"][0]["status"] == "planned"
    recs[1]["kind"] = "check"
    _, model = verdicts(trust, tmp_path, recs, graph)
    assert model["pipelines"][0]["status"] == "validated"


def test_failure_outweighs_passes(trust, tmp_path):
    got, _ = verdicts(trust, tmp_path, [record("t::x", "virgil.a"), record("t::y", "virgil.a", outcome="failed")])
    assert got["virgil.a"] == "failing"


def test_a_part_relying_on_an_unverified_part_is_not_verified(trust, tmp_path):
    graph = {"nodes": {"virgil.a": {"layer": "models", "depends": ["virgil.b"]}, "virgil.b": {"layer": "models"}}}
    got, _ = verdicts(trust, tmp_path, [record("t::x", "virgil.a")], graph)
    assert got == {"virgil.a": "relies-unverified", "virgil.b": "unchecked"}


def test_a_check_through_unverified_reference_code_does_not_verify(trust, tmp_path):
    graph = {"nodes": {"virgil.a": {"layer": "models"}, "crosscheck.r": {"layer": "references"}}}
    rec = record("t::x", "virgil.a", via=["crosscheck.r"])
    got, model = verdicts(trust, tmp_path, [rec], graph)
    assert got["virgil.a"] == "relies-unverified"
    assert [c["id"] for c in model["nodes"]["virgil.a"]["because"]] == ["crosscheck.r"]
    got, _ = verdicts(trust, tmp_path, [rec, record("t::r", "crosscheck.r", kind="reference")], graph)
    assert got["virgil.a"] == "verified"
    # a reference relying on an unverified reference is not trusted either
    graph["nodes"]["crosscheck.r"]["depends"] = ["crosscheck.s"]
    graph["nodes"]["crosscheck.s"] = {"layer": "references"}
    got, _ = verdicts(trust, tmp_path, [rec, record("t::r", "crosscheck.r", kind="reference")], graph)
    assert got["virgil.a"] == "relies-unverified"


def test_reference_code_missing_from_the_graph_is_not_trusted(trust, tmp_path):
    got, model = verdicts(trust, tmp_path, [record("t::x", "virgil.a", via=["crosscheck.nowhere"])])
    assert got["virgil.a"] == "relies-unverified"


def test_each_property_needs_its_own_independent_roots(trust, tmp_path):
    """Two roots on a part are not enough if they check different things."""
    graph = {"nodes": {"virgil.a": {"layer": "models", "roots": 2, "properties": {"sense": 2}}}}
    recs = [record("t::x", "virgil.a", roots=["mathematics"], properties=["sense"]),
            record("t::y", "virgil.a", roots=["pmoired"], properties=["amplitude"])]
    got, model = verdicts(trust, tmp_path, recs, graph)
    assert got["virgil.a"] == "partly"
    assert model["nodes"]["virgil.a"]["properties"]["sense"] == {"need": 2, "roots": ["mathematics"]}
    recs[1]["properties"] = ["sense"]
    got, _ = verdicts(trust, tmp_path, recs, graph)
    assert got["virgil.a"] == "verified"


@pytest.mark.parametrize("roots,verdict", [
    (["mathematics"], "convention"),  # our own transcription of virgil's docs confirms nothing
    (["pmoired"], "verified"),  # another code agrees on the convention
])
def test_a_convention_from_virgils_docs_needs_outside_confirmation(trust, tmp_path, roots, verdict):
    graph = {"nodes": {"virgil.a": {"layer": "models", "convention": ["sense"]}}}
    rec = record("t::x", "virgil.a", roots=roots, properties=["sense"])
    got, model = verdicts(trust, tmp_path, [rec], graph)
    assert got["virgil.a"] == verdict
    assert model["nodes"]["virgil.a"]["unconfirmed"] == (["sense"] if verdict == "convention" else [])


def test_runs_on_uncommitted_or_foreign_code_are_left_out_of_the_page(trust, tmp_path):
    import subprocess
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    good = {"validation_commit": head, "virgil": {"commit": "c" * 40, "dirty": False}}
    assert trust.provenance_problem(good) is None
    assert "uncommitted" in trust.provenance_problem(good | {"validation_dirty": True})
    assert "virgil" in trust.provenance_problem(good | {"virgil": {"commit": "c" * 40, "dirty": True}})
    assert "history" in trust.provenance_problem(good | {"validation_commit": "0" * 40})
    path = evidence(tmp_path, "latest.jsonl", [record("t::x", "virgil.a")])
    lines = path.read_text().splitlines()
    lines[0] = json.dumps(json.loads(lines[0]) | {"validation_commit": head, "validation_dirty": True})
    path.write_text("\n".join(lines) + "\n")
    assert len(trust.load_evidence([path])[1]) == 1
    assert trust.load_evidence([path], strict=True) == ([], [])


def test_provenance_is_taken_before_the_evidence_file_exists(tmp_path, monkeypatch):
    """In CI the evidence file is written into the checkout; were the header
    taken after opening it, the untracked file would mark every run dirty."""
    import pytest as _pytest

    from evidence import plugin

    path = tmp_path / "evidence.jsonl"
    seen = []
    monkeypatch.setattr(plugin, "run_header", lambda: seen.append(path.exists()) or {"record": "run"})

    class Config:
        stash = _pytest.Stash()

        def getoption(self, name):
            return str(path)

    session = type("Session", (), {"config": Config()})()
    session.config.stash[plugin._RECORDS] = []
    plugin.pytest_sessionstart(session)
    plugin.pytest_sessionfinish(session)
    assert seen == [False]
    assert json.loads(path.read_text().splitlines()[0]) == {"record": "run"}


def test_render_is_not_an_external_or_allowed_root(trust):
    import evidence

    assert "render" not in evidence.ROOTS
    assert "render" not in trust.EXTERNAL


def test_a_resurrected_campaign_pass_is_flagged_with_its_commit_and_shown_as_a_banner(trust, tmp_path):
    camp = evidence(tmp_path, "campaigns.jsonl", [record("t::x", "virgil.a")], date="2026-10-01", commit="a" * 40)
    new = evidence(tmp_path, "new.jsonl", [record("t::y", "virgil.a"), record("t::x", "virgil.a", outcome="skipped")],
                   date="2026-10-06", commit="b" * 40)
    got, model = verdicts(trust, tmp_path, None, files=[camp, new])
    assert got["virgil.a"] == "verified"  # the verdict is unchanged: the fallback is deliberate
    assert model["older_campaigns"] == [("t::x", "a" * 40)]
    assert "aaaaaaa" in trust._older_campaigns_html(model["older_campaigns"], ["bbbbbbb"])
    assert trust._older_campaigns_html(model["older_campaigns"], ["bbbbbbb", "aaaaaaa"]) == ""
    assert trust._older_campaigns_html([], ["bbbbbbb"]) == ""


@pytest.fixture
def two_commits(tmp_path):
    """A virgil checkout with two commits, ``old`` then ``new`` (``old`` is an ancestor of ``new``)."""
    import subprocess
    repo = tmp_path / "virgil"
    repo.mkdir()
    env = ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    shas = []
    for i in range(2):
        (repo / "f").write_text(str(i))
        for cmd in (["init", "-q"] if i == 0 else [], ["add", "f"], [*env, "commit", "-q", "-m", str(i)]):
            if cmd:
                subprocess.run(["git", "-C", str(repo), *cmd], check=True, capture_output=True)
        shas.append(subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip())
    return repo, *shas


def _weekly_fail_then_pin_pass(tmp_path, old, new, **extra):
    """The published case: a weekly run on virgil main (``new``) fails, a later push run on the older pin (``old``) passes."""
    weekly = evidence(tmp_path, "weekly.jsonl", [record("t::x", "virgil.a", outcome="failed")],
                      date="2026-10-01", commit=new)
    head, *rest = weekly.read_text().splitlines()
    weekly.write_text("\n".join([json.dumps(json.loads(head) | {"run_url": "https://ci/weekly"}), *rest]) + "\n")
    push = evidence(tmp_path, "push.jsonl", [record("t::x", "virgil.a")], date="2026-10-06", commit=old)
    return [weekly, push]


def test_a_failure_on_a_newer_virgil_commit_is_shown_not_overwritten(trust, tmp_path, two_commits):
    """A weekly failure on virgil main, then a later push pass on the older pin: the pass is kept and the verdict
    unchanged, but the part and the page carry a badge and a banner naming the failing run."""
    repo, old, new = two_commits
    files = _weekly_fail_then_pin_pass(tmp_path, old, new)
    got, model = verdicts(trust, tmp_path, None, files=files, virgil=repo)
    assert got["virgil.a"] == "verified"  # display only: the verdict is B4's decision
    assert model["nodes"]["virgil.a"]["fails_elsewhere"] == ["t::x"]
    [fail] = model["fails_elsewhere"]
    assert (fail["test"], fail["commit"], fail["run_url"]) == ("t::x", new, "https://ci/weekly")
    html = trust._fails_elsewhere_html(model["fails_elsewhere"])
    assert new[:7] in html and "https://ci/weekly" in html and "virgil.a" in html
    assert "fails on another virgil commit" in trust._fails_badge(model["nodes"]["virgil.a"])
    assert trust._fails_elsewhere_html([]) == "" and trust._fails_badge({"fails_elsewhere": []}) == ""


def test_no_badge_when_the_failure_was_at_an_older_commit_that_a_newer_one_fixed(trust, tmp_path, two_commits):
    repo, old, new = two_commits
    failed = evidence(tmp_path, "old.jsonl", [record("t::x", "virgil.a", outcome="failed")], date="2026-10-01", commit=old)
    fixed = evidence(tmp_path, "new.jsonl", [record("t::x", "virgil.a")], date="2026-10-06", commit=new)
    for virgil in (repo, None):  # by ancestry, and by the run-date fallback
        for files in ([failed, fixed], [fixed, failed]):
            got, model = verdicts(trust, tmp_path, None, files=files, virgil=virgil)
            assert got["virgil.a"] == "verified"
            assert model["fails_elsewhere"] == [] and model["nodes"]["virgil.a"]["fails_elsewhere"] == []


def test_a_pipeline_failing_on_a_newer_commit_gets_a_badge_inside_its_grid_cell(trust, tmp_path, two_commits):
    repo, old, new = two_commits
    graph = {"nodes": {"virgil.a": {"layer": "models"}},
             "pipelines": {"p": {"title": "P", "steps": ["virgil.a"], "data": "simulated"}}}
    weekly = evidence(tmp_path, "weekly.jsonl", [record("t::p", "pipeline:p", outcome="failed")], date="2026-10-01", commit=new)
    push = evidence(tmp_path, "push.jsonl", [record("t::p", "pipeline:p")], date="2026-10-06", commit=old)
    # file order reversed: the result must not depend on it
    _, model = verdicts(trust, tmp_path, None, graph=graph, files=[push, weekly], virgil=repo)
    [pipe] = model["pipelines"]
    assert pipe["status"] == "passes"  # display only: the status is unchanged
    assert pipe["fails_elsewhere"] == ["t::p"]
    for html in (trust._chain(pipe, model["nodes"]), trust._pipeline_card(pipe, model["nodes"])):
        assert "vt-failelse" in html
    chain = trust._chain(pipe, model["nodes"])
    # the badge sits inside the third grid cell, so .vt-chain keeps exactly three children
    assert [chain.count(f'<span class="{c}"') for c in ("vt-chain-t", "vt-steps", "vt-chain-r")] == [1, 1, 1]
    assert chain.index("vt-chain-r") < chain.index("vt-failelse")
    assert chain.endswith("</span></span></div>")  # the badge closes inside vt-chain-r, not beside it


def test_the_newest_record_per_other_commit_is_chosen_whatever_the_input_order(trust, two_commits):
    repo, old, new = two_commits

    def rec(outcome, date):
        return {"outcome": outcome, "_commit": new, "_date": date, "_run_url": date}
    kept = {"outcome": "passed", "_commit": old, "_date": "9"}
    for recs in ([rec("failed", "1"), rec("passed", "2")], [rec("passed", "2"), rec("failed", "1")]):
        assert trust._fails_on_other_commit(recs, kept, repo) is None  # the newest record at that commit passed
    for recs in ([rec("passed", "1"), rec("failed", "2")], [rec("failed", "2"), rec("passed", "1")]):
        assert trust._fails_on_other_commit(recs, kept, repo)["run_url"] == "2"


def test_no_failure_badge_when_the_same_commit_failed_or_the_other_commit_passed(trust, tmp_path, two_commits):
    _, old, new = two_commits
    first = evidence(tmp_path, "first.jsonl", [record("t::x", "virgil.a", outcome="failed")], date="2026-10-01", commit=new)
    second = evidence(tmp_path, "second.jsonl", [record("t::x", "virgil.a")], date="2026-10-06", commit=new)
    _, model = verdicts(trust, tmp_path, None, files=[first, second])  # fixed on the same commit: not a commit gap
    assert model["fails_elsewhere"] == []
    ok = evidence(tmp_path, "ok.jsonl", [record("t::x", "virgil.a")], date="2026-10-01", commit=old)
    _, model = verdicts(trust, tmp_path, None, files=[ok, second])
    assert model["fails_elsewhere"] == [] and model["nodes"]["virgil.a"]["fails_elsewhere"] == []
