"""The verdict rules of scripts/trust.py, one synthetic case per rule
(docs/method/index.md): what may verify a part of virgil, and what may not."""

import importlib.util
import json
import pathlib
import subprocess

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


def test_newest_outcome_wins_and_an_old_pass_does_not_outlive_a_newer_skip(trust, tmp_path):
    old = evidence(tmp_path, "old.jsonl", [record("t::x", "virgil.a")], date="2026-10-01")
    new = evidence(tmp_path, "new.jsonl", [record("t::x", "virgil.a", outcome="skipped")], date="2026-10-06")
    got, _ = verdicts(trust, tmp_path, None, files=[old, new])
    assert got["virgil.a"] == "unchecked"
    camp = evidence(tmp_path, "campaigns-old.jsonl", [record("t::x", "virgil.a")], date="2026-10-01")
    got, _ = verdicts(trust, tmp_path, None, files=[camp, new])
    assert got["virgil.a"] == "unchecked"  # no exemption by file name: a campaign test is stamped and run by CI instead


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


def test_a_pipeline_whose_checks_all_skipped_is_not_run(trust, tmp_path):
    graph = {"nodes": {"virgil.a": {"layer": "models"}},
             "pipelines": {"p": {"title": "P", "steps": ["virgil.a"], "state": "running"},
                           "q": {"title": "Q", "steps": ["virgil.a"], "state": "running"},
                           "r": {"title": "R", "steps": ["virgil.a"], "state": "planned"}}}
    recs = [record("t::p", "pipeline:p", outcome="skipped"),
            record("t::q1", "pipeline:q", outcome="skipped"), record("t::q2", "pipeline:q", outcome="failed")]
    _, model = verdicts(trust, tmp_path, recs, graph)
    status = {p["id"]: p["status"] for p in model["pipelines"]}
    assert status == {"p": "not run", "q": "failing", "r": "planned"}  # no checks at all keeps the written state
    card = trust._pipeline_card(model["pipelines"][0], model["nodes"])
    assert "not run (all 1 check skipped)" in card and "vt-p-not-run" in card


@pytest.mark.parametrize("kind", ["finding", "upstream"])
@pytest.mark.parametrize("outcome, verdict", [("failed", "failing"), ("error", "failing"), ("xpassed", "failing"), ("xfailed", "verified")])
def test_a_strict_xpass_of_an_expected_failure_marks_the_node_failing(trust, tmp_path, kind, outcome, verdict):
    # pytest records a strict XPASS as "failed" with no wasxfail; an xfail that still fails is fine
    graph = {"nodes": {"pkg": {"layer": "references"}}}
    recs = [record("t::ok", "pkg", kind="reference"), record("t::x", "pkg", kind=kind, outcome=outcome)]
    got, _ = verdicts(trust, tmp_path, recs, graph)
    assert got["pkg"] == verdict


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
    monkeypatch.setattr(plugin, "run_header", lambda evidence=None: seen.append(path.exists()) or {"record": "run"})

    class Config:
        stash = _pytest.Stash()

        def getoption(self, name):
            return str(path)

    session = type("Session", (), {"config": Config()})()
    session.config.stash[plugin._RECORDS] = []
    plugin.pytest_sessionstart(session)
    plugin.pytest_sessionfinish(session)
    assert seen == [False]
    header = json.loads(path.read_text().splitlines()[0])
    # provenance from the start of the session, plus how tests were selected, known at its end
    assert header == {"record": "run", "invocation": {"argv": [], "markexpr": "", "keyword": "", "deselected": 0}}


def test_render_is_not_an_external_or_allowed_root(trust):
    import evidence

    assert "render" not in evidence.ROOTS
    assert "render" not in trust.EXTERNAL


def test_a_records_own_virgil_commit_replaces_its_run_headers(trust, tmp_path):
    """A campaign test stamps the commit its summary ran on; the page reads the node at that commit."""
    stamped = record("t::x", "virgil.a", virgil_commit="a" * 40, virgil_dirty=False, criteria_commit="f" * 40)
    plain = record("t::y", "virgil.a")
    f = evidence(tmp_path, "run.jsonl", [stamped, plain], commit="b" * 40)
    runs, recs = trust.load_evidence([f])
    assert {r["test"]: r["_commit"] for r in recs} == {"t::x": "a" * 40, "t::y": "b" * 40}
    _, model = verdicts(trust, tmp_path, None, files=[f])
    assert model["nodes"]["virgil.a"]["commits"] == ["a" * 40, "b" * 40]


def test_a_negative_control_is_never_the_headline_of_a_part(trust):
    base = {"test": "t::c", "roots": ["mathematics"], "tier": "A", "outcome": "passed", "_source": "ours",
            "metrics": {"wrong_mapping_rel_err": 0.4}}
    assert trust._headline({**base, "kind": "control"}) == (None, None)
    assert trust._headline({**base, "kind": "check"}) == ("wrong_mapping_rel_err", 0.4)
    # an explicit headline on a control is still honoured
    assert trust._headline({**base, "kind": "control", "headline": "wrong_mapping_rel_err"})[1] == 0.4


def _ingest_module(monkeypatch):
    import sys

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    sys.modules.pop("ingest_virgil_evidence", None)
    return importlib.import_module("ingest_virgil_evidence")


def _git_repo(path):
    """A git repo at ``path`` with committed ``tracked.txt`` and ``trust/evidence/e.jsonl``."""
    import subprocess

    def git(*args):
        subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)

    path.mkdir()
    git("init", "-q")
    git("config", "user.email", "t@example.org")
    git("config", "user.name", "t")
    git("config", "commit.gpgsign", "false")
    (path / "tracked.txt").write_text("a\n")
    (path / "trust" / "evidence").mkdir(parents=True)
    (path / "trust" / "evidence" / "e.jsonl").write_text("{}\n")
    git("add", ".")
    git("commit", "-q", "-m", "init")


def test_own_provenance_reports_commit_and_a_clean_checkout_as_not_dirty(monkeypatch, tmp_path):
    import subprocess

    ingest = _ingest_module(monkeypatch)
    _git_repo(tmp_path / "r")
    head = subprocess.check_output(["git", "-C", str(tmp_path / "r"), "rev-parse", "HEAD"], text=True).strip()
    assert ingest.own_provenance(root=tmp_path / "r") == (head, False)


def test_own_provenance_a_tracked_modification_is_dirty(monkeypatch, tmp_path):
    ingest = _ingest_module(monkeypatch)
    _git_repo(tmp_path / "r")
    (tmp_path / "r" / "tracked.txt").write_text("changed\n")
    commit, dirty = ingest.own_provenance(root=tmp_path / "r")
    assert len(commit) == 40 and dirty is True


def test_own_provenance_ignores_untracked_files_anywhere(monkeypatch, tmp_path):
    """This implementation passes ``--untracked-files=no``: an untracked file counts as clean, even
    inside a folder of tracked files (``trust/``) or beside tracked code at the root."""
    ingest = _ingest_module(monkeypatch)
    _git_repo(tmp_path / "r")
    (tmp_path / "r" / "stray.txt").write_text("x\n")
    (tmp_path / "r" / "trust" / "stray.py").write_text("x\n")
    assert ingest.own_provenance(root=tmp_path / "r")[1] is False


def test_own_provenance_exempts_any_tracked_edit_under_trust_evidence(monkeypatch, tmp_path):
    """The exemption is a path prefix: editing a committed file under ``trust/evidence/`` stays clean,
    while a tracked edit elsewhere still makes the run dirty."""
    ingest = _ingest_module(monkeypatch)
    _git_repo(tmp_path / "r")
    (tmp_path / "r" / "trust" / "evidence" / "e.jsonl").write_text('{"hand": "edited"}\n')
    assert ingest.own_provenance(root=tmp_path / "r")[1] is False
    (tmp_path / "r" / "tracked.txt").write_text("changed\n")
    assert ingest.own_provenance(root=tmp_path / "r")[1] is True


def test_own_provenance_is_none_none_outside_a_git_checkout(monkeypatch, tmp_path):
    ingest = _ingest_module(monkeypatch)
    (tmp_path / "plain").mkdir()
    assert ingest.own_provenance(root=tmp_path / "plain") == (None, None)


def test_virgils_ingested_ci_evidence_survives_strict_provenance(trust, tmp_path, monkeypatch):
    """The ingest script records this repository's commit, so the file is not
    dropped as 'does not say which commit it ran on'. Dirtiness is fixed to a
    clean checkout of this repository's HEAD, so the test does not depend on the
    state of the working tree; own_provenance itself is tested directly above."""
    import subprocess
    import sys

    ingest = _ingest_module(monkeypatch)
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setattr(ingest, "own_provenance", lambda: (head, False))
    junit = tmp_path / "junit-evidence.xml"
    claim = json.dumps({"objects": ["virgil.a"], "roots": ["mathematics"], "kind": "check", "tier": "A",
                        "properties": []}).replace('"', "&quot;")
    junit.write_text(f'<testsuites><testsuite><testcase classname="tests.t" name="test_x" time="0.1">'
                     f'<properties><property name="validates" value="{claim}"/></properties>'
                     f"</testcase></testsuite></testsuites>")
    out = tmp_path / "virgil-evidence.jsonl"
    monkeypatch.setattr(sys, "argv", ["ingest", str(junit), str(out), "--commit", "d" * 40])
    ingest.main()
    header = json.loads(out.read_text().splitlines()[0])
    assert header["validation_commit"] == head
    assert isinstance(header["validation_dirty"], bool) and header["validation_dirty"] is False
    runs, recs = trust.load_evidence([out], strict=True)
    assert len(runs) == 1 and len(recs) == 1 and recs[0]["_source"] == "virgil-ci"


def test_a_virgil_commit_without_criteria_commit_does_not_override_the_header(trust, tmp_path):
    loose = record("t::x", "virgil.a", virgil_commit="a" * 40)
    _, recs = trust.load_evidence([evidence(tmp_path, "run.jsonl", [loose], commit="b" * 40)])
    assert recs[0]["_commit"] == "b" * 40


def test_strict_provenance_leaves_out_a_stamped_record_that_is_not_recorded_clean(trust, tmp_path):
    head = subprocess.check_output(["git", "-C", str(trust.ROOT), "rev-parse", "HEAD"], text=True).strip()
    recs = [record(f"t::{n}", "virgil.a", virgil_commit="a" * 40, criteria_commit="f" * 40, **extra)
            for n, extra in (("clean", {"virgil_dirty": False}), ("dirty", {"virgil_dirty": True}), ("unrecorded", {}))]
    path = tmp_path / "run.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in [
        {"record": "run", "date": "2026-10-06T00:00:00+00:00", "virgil": {"commit": "b" * 40, "dirty": False},
         "validation_commit": head, "validation_dirty": False, "runner": "test"}, *recs]) + "\n")
    _, kept = trust.load_evidence([path], strict=True)
    assert [r["test"] for r in kept] == ["t::clean"]
    _, kept = trust.load_evidence([path])  # the unpublished (non-strict) build keeps all three
    assert len(kept) == 3


def test_a_stamped_records_commit_decides_whether_a_failure_elsewhere_is_newer(trust, tmp_path, two_commits):
    """The pass is stamped ``old`` although its run installed ``new``: a failure at ``new`` is then on a newer commit and
    shown; stamped ``new``, the same failure at ``old`` is on an older commit and not shown."""
    repo, old, new = two_commits
    fail = evidence(tmp_path, "fail.jsonl", [record("t::x", "virgil.a", outcome="failed")], date="2026-10-01", commit=new)
    ok_old = evidence(tmp_path, "ok_old.jsonl", [record("t::x", "virgil.a", virgil_commit=old, criteria_commit="f" * 40)], date="2026-10-06", commit=new)
    _, recs = trust.load_evidence([fail, ok_old], virgil=repo)
    assert recs[0]["_commit"] == old and recs[0]["_fails_elsewhere"]["commit"] == new
    fail_old = evidence(tmp_path, "fail_old.jsonl", [record("t::x", "virgil.a", outcome="failed")], date="2026-10-01", commit=old)
    ok_new = evidence(tmp_path, "ok_new.jsonl", [record("t::x", "virgil.a", virgil_commit=new, criteria_commit="f" * 40)], date="2026-10-06", commit=old)
    _, recs = trust.load_evidence([fail_old, ok_new], virgil=repo)
    assert recs[0]["_commit"] == new and "_fails_elsewhere" not in recs[0]


def test_a_negative_control_is_never_the_headline_of_a_part(trust):
    base = {"test": "t::c", "roots": ["mathematics"], "tier": "A", "outcome": "passed", "_source": "ours",
            "metrics": {"wrong_mapping_rel_err": 0.4}}
    assert trust._headline({**base, "kind": "control"}) == (None, None)
    assert trust._headline({**base, "kind": "check"}) == ("wrong_mapping_rel_err", 0.4)
    # an explicit headline on a control is still honoured
    assert trust._headline({**base, "kind": "control", "headline": "wrong_mapping_rel_err"})[1] == 0.4


def test_an_explicit_headline_that_was_not_recorded_shows_no_number(trust):
    assert trust._headline({"metrics": {"rel_a": 1e-9}, "headline": "rel_b"}) == (None, None)
    assert trust._headline({"metrics": {"rel_a": 1e-9}, "headline": ""}) == (None, None)  # an empty name is still explicit


def test_the_order_warning_names_the_file_and_the_function(trust):
    html = trust._headline_by_order_html(["tests/test_x.py::test_fn"])
    assert "<code>test_x.py::test_fn</code>" in html and "tests/" not in html


@pytest.mark.parametrize("tests, expected", [
    ([("t.py::a", {"rel_a": 1e-9, "rel_b": 2e-9}, None)], ["t.py::a"]),  # two agreement metrics, no headline
    ([("t.py::a", {"rel_a": 1e-9, "rel_b": 2e-9}, "rel_a")], []),  # an explicit headline is not by order
    ([("t.py::a", {"rel_a": 1e-9, "rel_b": 2e-9}, "")], []),  # an empty headline is still explicit
    ([("t.py::a[x]", {"rel_a": 1e-9, "rel_b": 2e-9}, None), ("t.py::a[y]", {"rel_a": 1e-9, "rel_b": 2e-9}, None)],
     ["t.py::a"]),  # one entry per test function, not per parameter set
    ([("t.py::a", {"rel_a": 1e-9, "n": 3.0}, None)], []),  # one candidate only
])
def test_headline_by_order(trust, tests, expected):
    records = [{"test": t, "kind": "check", "metrics": m, **({"headline": h} if h is not None else {})} for t, m, h in tests]
    assert trust._headline_by_order(records) == expected


def test_a_golden_root_not_listed_in_golden_yml_fails_collection(tmp_path):
    """roots_of drops an unlisted golden root, so the plugin must refuse it at
    collection instead of letting it silently count for nothing."""
    from types import SimpleNamespace

    from evidence.plugin import pytest_collection_modifyitems

    dummy = tmp_path / "test_dummy.py"
    dummy.write_text("pass\n")
    cfg = SimpleNamespace(getoption=lambda name: True)

    def item(root):
        mark = pytest.mark.validates("virgil.models.UniformDisk", roots=[root]).mark
        return SimpleNamespace(nodeid="test_dummy.py::test_x", path=dummy,
                               iter_markers=lambda name: [mark] if name == "validates" else [])

    pytest_collection_modifyitems(cfg, [item("golden:dholakia")])  # listed
    pytest_collection_modifyitems(cfg, [item("golden:visread")])  # listed
    with pytest.raises(pytest.UsageError, match=r"golden roots \['golden:unlisted'\]"):
        pytest_collection_modifyitems(cfg, [item("golden:unlisted")])


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
