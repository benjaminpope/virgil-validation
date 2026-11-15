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


def test_ok_via_does_not_mask_missing_property_roots_or_unconfirmed_conventions(trust, tmp_path):
    """A part checked through an unverified ``via`` reference must still satisfy its own ``properties`` and
    ``convention`` requirements when ``blocked_roots`` are counted; only once those hold may it become ``ok-via``.
    Adding a passing trusted check must never demote ``relies-unverified`` to ``partly`` or ``convention``."""
    graph = {
        "nodes": {
            "crosscheck.r": {"layer": "references"},
            "virgil.prop": {"layer": "models", "roots": 2, "properties": {"sense": 2}},
            "virgil.prop_mix": {"layer": "models", "roots": 1, "properties": {"sense": 2}},
            "virgil.conv": {"layer": "models", "convention": ["sense"]},
        }
    }
    recs = [
        record("t::p1", "virgil.prop", roots=["mathematics"], via=["crosscheck.r"], properties=["sense"]),
        record("t::p2", "virgil.prop", roots=["pmoired"], via=["crosscheck.r"]),
        record("t::c1", "virgil.conv", roots=["mathematics"], via=["crosscheck.r"], properties=["sense"]),
    ]
    got, _ = verdicts(trust, tmp_path, recs, graph)
    assert got["virgil.prop"] == "partly"
    assert got["virgil.conv"] == "convention"

    recs[1]["properties"] = ["sense"]
    recs[2]["roots"] = ["pmoired"]
    got, model = verdicts(trust, tmp_path, recs, graph)
    assert got["virgil.prop"] == "relies-unverified"
    assert got["virgil.conv"] == "relies-unverified"
    assert [c["id"] for c in model["nodes"]["virgil.conv"]["because"]] == ["crosscheck.r"]

    # Monotonicity under mixed evidence: adding a trusted mathematics check must not demote relies-unverified
    recs_mixed = [
        *recs,
        record("t::c2", "virgil.conv", roots=["mathematics"], properties=["sense"]),
        record("t::pm1", "virgil.prop_mix", roots=["mathematics"], properties=["sense"]),
        record("t::pm2", "virgil.prop_mix", roots=["pmoired"], via=["crosscheck.r"], properties=["sense"]),
    ]
    got, model = verdicts(trust, tmp_path, recs_mixed, graph)
    assert got["virgil.conv"] == "relies-unverified"
    assert got["virgil.prop_mix"] == "relies-unverified"
    assert [c["id"] for c in model["nodes"]["virgil.conv"]["because"]] == ["crosscheck.r"]
    assert [c["id"] for c in model["nodes"]["virgil.prop_mix"]["because"]] == ["crosscheck.r"]

    # Converse monotonicity: adding a supplementary via-blocked check (even via a failing reference)
    # must never demote a node already verified by its direct trusted checks alone
    graph_ok = {
        "nodes": {
            "crosscheck.unverified": {"layer": "references"},
            "crosscheck.bad": {"layer": "references"},
            "virgil.already_ok": {"layer": "models", "roots": 1, "properties": {"sense": 1}},
        }
    }
    recs_ok = [
        record("t::ok_direct", "virgil.already_ok", roots=["mathematics"], properties=["sense"]),
        record("t::ok_extra_unv", "virgil.already_ok", roots=["pmoired"], via=["crosscheck.unverified"], properties=["sense"]),
        record("t::ok_extra_bad", "virgil.already_ok", roots=["candid"], via=["crosscheck.bad"]),
        record("t::bad_ref", "crosscheck.bad", kind="reference", outcome="failed"),
    ]
    got_ok, model_ok = verdicts(trust, tmp_path, recs_ok, graph_ok)
    assert got_ok["virgil.already_ok"] == "verified"
    assert model_ok["nodes"]["virgil.already_ok"]["because"] == []


def test_upstream_failing_dependency_marks_dependent_relies_not_relies_unverified(trust, tmp_path):
    graph = {
        "nodes": {
            "virgil.a": {"layer": "models", "depends": ["virgil.b"]},
            "virgil.b": {"layer": "models"},
        }
    }
    recs = [record("t::a", "virgil.a"), record("t::b", "virgil.b", outcome="failed")]
    got, model = verdicts(trust, tmp_path, recs, graph)
    assert got == {"virgil.a": "relies", "virgil.b": "failing"}
    assert model["nodes"]["virgil.a"]["because"] == [{"id": "virgil.b", "findings": [], "status": "failing"}]


def test_changed_below_propagates_through_unchecked_dependencies_and_empty_source_callers(trust, tmp_path):
    repo = tmp_path / "virgil"
    repo.mkdir()
    env = ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    (repo / "src" / "virgil").mkdir(parents=True)
    (repo / "src" / "virgil" / "leaf.py").write_text("x = 1\n")
    for cmd in (["init", "-q"], ["add", "."], [*env, "commit", "-q", "-m", "old"]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True, capture_output=True)
    old = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    (repo / "src" / "virgil" / "leaf.py").write_text("x = 2\n")
    for cmd in (["add", "."], [*env, "commit", "-q", "-m", "new"]):
        subprocess.run(["git", "-C", str(repo), *cmd], check=True, capture_output=True)
    new = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    subprocess.run(["git", "-C", str(repo), "update-ref", "refs/remotes/origin/main", new], check=True, capture_output=True)

    graph = {
        "nodes": {
            "virgil.top": {"layer": "imaging", "source": [], "depends": ["virgil.mid"]},
            "virgil.mid": {"layer": "models", "source": [], "depends": ["virgil.leaf"]},
            "virgil.leaf": {"layer": "data", "source": ["src/virgil/leaf.py"]},
        }
    }
    # 1. Unchecked leaf + empty-source caller at old commit: leaf.py change propagates to top
    path = evidence(tmp_path, "e.jsonl", [record("t::top", "virgil.top")], commit=old)
    runs, recs = trust.load_evidence([path], virgil=repo)
    model = trust.build(graph, [], recs, runs, virgil=repo, golden=set())
    assert model["nodes"]["virgil.top"]["changed"] == ["src/virgil/leaf.py"]
    assert model["nodes"]["virgil.leaf"]["changed"] is None  # unchecked itself: no commit of its own
    assert model["changed"] == 1

    # 2. Leaf re-tested on new commit (so leaf itself is clean), while top remains tested on old commit:
    # top must still report leaf.py changed since top's own evidence commit
    path_leaf_new = evidence(tmp_path, "e_new.jsonl", [record("t::leaf", "virgil.leaf")], date="2026-10-07", commit=new)
    runs, recs = trust.load_evidence([path, path_leaf_new], virgil=repo)
    model = trust.build(graph, [], recs, runs, virgil=repo, golden=set())
    assert model["nodes"]["virgil.leaf"]["changed"] == []
    assert model["nodes"]["virgil.top"]["changed"] == ["src/virgil/leaf.py"]

    # 3. Unreachable commit SHA must report None ("unknown"), never [] ("clean")
    path_bad = evidence(tmp_path, "e_bad.jsonl", [record("t::top_bad", "virgil.top")], commit="0" * 40)
    runs, recs = trust.load_evidence([path_bad], virgil=repo)
    model = trust.build(graph, [], recs, runs, virgil=repo, golden=set())
    assert model["nodes"]["virgil.top"]["changed"] is None

    # 4. Reverse staleness: caller tested on new commit while leaf was only tested on old (or unreachable) commit
    path_top_new = evidence(tmp_path, "e_top_new.jsonl", [record("t::top_new", "virgil.top")], date="2026-10-08", commit=new)
    path_leaf_old = evidence(tmp_path, "e_leaf_old.jsonl", [record("t::leaf_old", "virgil.leaf")], date="2026-10-01", commit=old)
    runs, recs = trust.load_evidence([path_top_new, path_leaf_old], virgil=repo)
    model = trust.build(graph, [], recs, runs, virgil=repo, golden=set())
    assert model["nodes"]["virgil.leaf"]["changed"] == ["src/virgil/leaf.py"]
    assert model["nodes"]["virgil.top"]["changed"] == ["src/virgil/leaf.py"]

    path_leaf_bad = evidence(tmp_path, "e_leaf_bad.jsonl", [record("t::leaf_bad", "virgil.leaf")], commit="0" * 40)
    runs, recs = trust.load_evidence([path_top_new, path_leaf_bad], virgil=repo)
    model = trust.build(graph, [], recs, runs, virgil=repo, golden=set())
    assert model["nodes"]["virgil.top"]["changed"] is None


def test_module_via_is_fail_closed_expands_init_and_tracks_subprocess_workers(tmp_path):
    from evidence import plugin

    test_file = tmp_path / "test_sample_via.py"
    test_file.write_text(
        "import virgil_bridge.alma as vba\n"
        "from virgil_bridge import mpol, binary, unregistered_submod\n"
        "import virgil_bridge.alma_lambda_report as rep\n"
        "import external_bridge._subprocess\n"
        "from external_bridge import _subprocess as sp\n"
        "EHT_WORKER: str = 'ehtim_worker.py'\n"
        "def _w(s, t):\n"
        "    return sp.run('alma', s, t)\n"
        "def _w2(s):\n"
        "    return _w(s, {})\n"
        "_w2('casa_clean_worker.py')\n"
        "external_bridge._subprocess.run('ehtim', EHT_WORKER, {})\n",
        encoding="utf-8",
    )
    plugin._module_via.cache_clear()
    got = plugin._module_via(str(test_file))
    assert got == (
        "crosscheck.limb",
        "crosscheck.sky",
        "external_bridge.casa_clean_worker",
        "external_bridge.ehtim_worker",
        "virgil_bridge",
        "virgil_bridge.alma",
        "virgil_bridge.alma_lambda_report",
        "virgil_bridge.mpol",
        "virgil_bridge.unregistered_submod",
    )

    # Dynamic or unresolvable _subprocess.run calls must emit an unregistered sentinel (fail-closed)
    dyn_file = tmp_path / "test_dyn_via.py"
    dyn_file.write_text(
        "import external_bridge._subprocess\n"
        "WORKERS = {'a': 'alma_worker.py'}\n"
        "external_bridge._subprocess.run('alma', WORKERS['a'], {})\n",
        encoding="utf-8",
    )
    plugin._module_via.cache_clear()
    assert plugin._module_via(str(dyn_file)) == (plugin._UNRESOLVED_WORKER,)

    # Reassigned module constant or locally shadowed worker parameter also fails closed
    rebound = tmp_path / "test_rebound_via.py"
    rebound.write_text(
        "from external_bridge import _subprocess as sp\n"
        "W = 'alma_worker.py'\n"
        "if True:\n"
        "    W = 'mpol_worker.py'\n"
        "sp.run('alma', W, {})\n",
        encoding="utf-8",
    )
    shadowed = tmp_path / "test_shadowed_via.py"
    shadowed.write_text(
        "from external_bridge import _subprocess as sp\n"
        "W = 'alma_worker.py'\n"
        "def run_one():\n"
        "    W = 'mpol_worker.py'\n"
        "    return sp.run('alma', W, {})\n"
        "run_one()\n",
        encoding="utf-8",
    )
    plugin._module_via.cache_clear()
    assert plugin._module_via(str(rebound)) == (plugin._UNRESOLVED_WORKER,)
    assert plugin._module_via(str(shadowed)) == (plugin._UNRESOLVED_WORKER,)


def test_item_via_tracks_local_helper_scripts_and_isolates_function_local_imports(tmp_path):
    from evidence import plugin

    # 1. Synthetic layout with benchmarks/, scripts/, and tests/ helpers + function-local isolation
    (tmp_path / "benchmarks" / "ring").mkdir(parents=True)
    (tmp_path / "scripts").mkdir(parents=True)
    (tmp_path / "tests").mkdir(parents=True)

    (tmp_path / "benchmarks" / "ring" / "bench_ext.py").write_text(
        "from crosscheck import simulate\n"
        "from external_bridge.pmoired_models import model_samples\n",
        encoding="utf-8",
    )
    (tmp_path / "scripts" / "script_clean.py").write_text(
        "from crosscheck import alma_maxent\n"
        "def run_helper():\n"
        "    import virgil_bridge.alma\n"
        "    return 1\n"
        "def unused_helper():\n"
        "    import external_bridge.radial_codes\n"
        "    return 2\n",
        encoding="utf-8",
    )
    mixed_test = tmp_path / "tests" / "test_mixed.py"
    mixed_test.write_text(
        "import pytest\n"
        "from crosscheck import radial\n"
        "\n"
        "@pytest.fixture\n"
        "def clean_fix():\n"
        "    from script_clean import run_helper\n"
        "    return run_helper()\n"
        "\n"
        "def test_with_bench_helper():\n"
        "    import bench_ext\n"
        "    return bench_ext\n"
        "\n"
        "def test_with_script_fixture(clean_fix):\n"
        "    return clean_fix\n"
        "\n"
        "def test_isolated_reference():\n"
        "    import math\n"
        "    return math.pi\n",
        encoding="utf-8",
    )
    plugin._module_via.cache_clear()
    assert plugin._item_via(str(mixed_test), "test_with_bench_helper", None, ()) == (
        "crosscheck.radial",
        "crosscheck.simulate",
        "external_bridge.pmoired_models",
    )
    assert plugin._item_via(str(mixed_test), "test_with_script_fixture", None, ("clean_fix",)) == (
        "crosscheck.alma_maxent",
        "crosscheck.radial",
        "virgil_bridge.alma",
    )
    assert plugin._item_via(str(mixed_test), "test_isolated_reference", None, ()) == (
        "crosscheck.radial",
    )

    # 2. Real repository checks on tests/test_polar_disk_vs_codes.py
    polar_test = ROOT / "tests" / "test_polar_disk_vs_codes.py"
    pmoired_via = plugin._item_via(str(polar_test), "test_matches_pmoired_ring_harmonics", None, ("tmp_path",))
    assert "crosscheck.simulate" in pmoired_via
    assert "external_bridge.pmoired_models" in pmoired_via
    frank_ref_via = plugin._item_via(str(polar_test), "test_frank_hankel_matches_closed_form_gaussian", None, ())
    assert frank_ref_via == ("crosscheck.radial",)
    assert "external_bridge.radial_codes" not in frank_ref_via


def test_reference_depends_cycle_fails_closed(trust, tmp_path):
    # 1. Mutual cycle between references: neither reference nor via-caller may be verified
    graph_cycle = {
        "nodes": {
            "virgil.x": {"layer": "models"},
            "crosscheck.a": {"layer": "references", "depends": ["crosscheck.b"]},
            "crosscheck.b": {"layer": "references", "depends": ["crosscheck.a"]},
        }
    }
    recs_cycle = [
        record("t::ra", "crosscheck.a", kind="reference"),
        record("t::rb", "crosscheck.b", kind="reference"),
        record("t::vx", "virgil.x", via=["crosscheck.a"]),
    ]
    got, _ = verdicts(trust, tmp_path, recs_cycle, graph_cycle)
    assert got == {
        "virgil.x": "relies-unverified",
        "crosscheck.a": "relies-unverified",
        "crosscheck.b": "relies-unverified",
    }

    # 2. Reference depending on a missing node fails closed on both the reference and its caller
    graph_missing = {
        "nodes": {
            "virgil.x": {"layer": "models"},
            "crosscheck.a": {"layer": "references", "depends": ["crosscheck.missing"]},
        }
    }
    got_missing, _ = verdicts(trust, tmp_path, recs_cycle[:1] + recs_cycle[2:], graph_missing)
    assert got_missing == {
        "virgil.x": "relies-unverified",
        "crosscheck.a": "relies-unverified",
    }

    # 3. Reference depending on a subject (virgil.*) node is never trusted, even if unchecked or ok
    graph_subj = {
        "nodes": {
            "virgil.u": {"layer": "models"},
            "virgil.x": {"layer": "models"},
            "crosscheck.a": {"layer": "references", "depends": ["virgil.u"]},
        }
    }
    got_subj, _ = verdicts(trust, tmp_path, recs_cycle[:1] + recs_cycle[2:], graph_subj)
    assert got_subj == {
        "virgil.u": "unchecked",
        "virgil.x": "relies-unverified",
        "crosscheck.a": "relies-unverified",
    }


def test_pipeline_with_failing_step_is_failing_while_bug_step_keeps_passes(trust, tmp_path):
    graph = {
        "nodes": {
            "virgil.fail_step": {"layer": "data"},
            "virgil.relies_fail_step": {"layer": "models", "depends": ["virgil.fail_step"]},
            "virgil.bug_step": {"layer": "likelihood"},
            "virgil.relies_bug_step": {"layer": "models", "depends": ["virgil.bug_step"]},
            "virgil.ok_step": {"layer": "models"},
            "crosscheck.unv": {"layer": "references"},
            "crosscheck.bad_ref": {"layer": "references"},
            "crosscheck.mid_ref": {"layer": "references", "depends": ["crosscheck.bad_ref"]},
            "crosscheck.bug_ref": {"layer": "references"},
        },
        "pipelines": {
            "p_fail": {"title": "Fail step", "steps": ["virgil.fail_step"]},
            "p_relies_fail": {"title": "Step relies on failing dep", "steps": ["virgil.relies_fail_step"]},
            "p_bug": {"title": "Bug step", "steps": ["virgil.bug_step"]},
            "p_relies_bug": {"title": "Step relies on bug dep", "steps": ["virgil.relies_bug_step"]},
            "p_ref_fail": {"title": "Ref check fails", "steps": ["virgil.ok_step"]},
            "p_weak": {"title": "Weak root only", "steps": ["virgil.ok_step"], "state": "running"},
            "p_via_unv": {"title": "Untrusted via", "steps": ["virgil.ok_step"]},
            "p_via_fail": {"title": "Failing via", "steps": ["virgil.ok_step"]},
            "p_via_trans_fail": {"title": "Transitive failing via", "steps": ["virgil.ok_step"]},
            "p_via_cc_bug": {"title": "Crosscheck bug via", "steps": ["virgil.ok_step"]},
            "p_via_monotone": {"title": "Clean check alongside broken via", "steps": ["virgil.ok_step"]},
        },
    }
    ledger = [
        {"id": "F1", "ruling": "virgil", "status": "open", "objects": ["virgil.bug_step"]},
        {"id": "C1", "ruling": "crosscheck", "status": "open", "objects": ["crosscheck.bug_ref"]},
    ]
    recs = [
        record("t::fs", "virgil.fail_step", outcome="failed"),
        record("t::rfs", "virgil.relies_fail_step"),
        record("t::bs", "virgil.bug_step"),
        record("t::rbs", "virgil.relies_bug_step"),
        record("t::oks", "virgil.ok_step"),
        record("t::bad_ref", "crosscheck.bad_ref", kind="reference", outcome="failed"),
        record("t::mid_ref", "crosscheck.mid_ref", kind="reference"),
        record("t::bug_ref", "crosscheck.bug_ref", kind="reference"),
        record("t::pf", "pipeline:p_fail"),
        record("t::prf", "pipeline:p_relies_fail"),
        record("t::pb", "pipeline:p_bug"),
        record("t::prb", "pipeline:p_relies_bug"),
        record("t::prf_ok", "pipeline:p_ref_fail"),
        record("t::prf_bad", "pipeline:p_ref_fail", kind="control", outcome="failed"),
        record("t::pweak", "pipeline:p_weak", roots=["self-consistency"]),
        record("t::pvia_u", "pipeline:p_via_unv", via=["crosscheck.unv"]),
        record("t::pvia_f", "pipeline:p_via_fail", via=["crosscheck.bad_ref"]),
        record("t::pvia_tf", "pipeline:p_via_trans_fail", via=["crosscheck.mid_ref"]),
        record("t::pvia_cb", "pipeline:p_via_cc_bug", via=["crosscheck.bug_ref"]),
        record("t::pm_clean", "pipeline:p_via_monotone", via=["crosscheck.unv"]),
        record("t::pm_broken", "pipeline:p_via_monotone", via=["crosscheck.bad_ref"]),
    ]
    _, model = verdicts(trust, tmp_path, recs, graph=graph, ledger=ledger)
    by_id = {p["id"]: p for p in model["pipelines"]}
    assert {k: p["status"] for k, p in by_id.items()} == {
        "p_fail": "failing",
        "p_relies_fail": "failing",
        "p_bug": "passes",
        "p_relies_bug": "passes",
        "p_ref_fail": "failing",
        "p_weak": "running",
        "p_via_unv": "passes",
        "p_via_fail": "failing",
        "p_via_trans_fail": "failing",
        "p_via_cc_bug": "failing",
        "p_via_monotone": "passes",
    }
    assert by_id["p_ref_fail"]["failing_checks"] == ["t::prf_bad"]
    assert by_id["p_via_unv"]["unverified"] == ["crosscheck.unv"]
    assert by_id["p_via_monotone"]["unverified"] == ["crosscheck.unv"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_evidence_record_and_metric_fixture_reject_non_finite_values(bad):
    from evidence import plugin

    with pytest.raises(ValueError, match="finite"):
        plugin.record("err", bad)
    recorded = []
    inner = plugin.metric.__wrapped__(lambda k, v: recorded.append((k, v)))
    assert inner("ok", 1.25) == 1.25 and recorded == [("ok", 1.25)]
    with pytest.raises(ValueError, match="finite"):
        inner("err", bad)


def test_trust_diff_reports_pipeline_status_changes(tmp_path):
    spec = importlib.util.spec_from_file_location("trust_diff", ROOT / "scripts" / "trust_diff.py")
    td = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(td)
    before = {
        "nodes": {
            "virgil.a": {"verdict": "unchecked", "strong": []},
            "virgil.step": {"verdict": "verified", "strong": ["mathematics"]},
            "virgil.relies_step": {"verdict": "verified", "strong": ["mathematics"]},
            "crosscheck.mid_ref": {"verdict": "verified", "strong": ["mathematics"]},
        },
        "pipelines": [
            {"id": "contest-imaging", "status": "passes", "steps": ["virgil.fitting.fit"], "unverified": ["virgil.fitting.fit"]},
            {"id": "regressed", "status": "validated", "steps": ["virgil.step"], "unverified": []},
            {"id": "trans_regressed", "status": "validated", "steps": ["virgil.relies_step"], "unverified": []},
        ],
    }
    after = {
        "nodes": {
            "virgil.a": {"verdict": "verified", "strong": ["mathematics"]},
            "virgil.step": {"verdict": "failing", "strong": []},
            "virgil.relies_step": {
                "verdict": "relies",
                "strong": ["mathematics"],
                "because": [{"id": "virgil.leaf", "status": "failing", "findings": []}],
            },
            "crosscheck.mid_ref": {
                "verdict": "relies",
                "strong": ["mathematics"],
                "because": [{"id": "crosscheck.leaf", "status": "bug", "findings": ["C1"]}],
            },
        },
        "pipelines": [
            {"id": "contest-imaging", "status": "running", "steps": ["virgil.fitting.fit"], "unverified": ["virgil.fitting.fit"]},
            {"id": "regressed", "status": "failing", "steps": ["virgil.step"], "unverified": ["virgil.step"], "failing_checks": ["t::p"]},
            {
                "id": "trans_regressed",
                "status": "failing",
                "steps": ["virgil.relies_step"],
                "unverified": ["virgil.relies_step", "crosscheck.mid_ref"],
                "failing_checks": [],
            },
        ],
    }
    rows = td.diff_rows(before, after)
    assert any("`virgil.a`" in r and "unchecked" in r and "verified" in r for r in rows)
    assert any("`pipeline:contest-imaging`" in r and "passes" in r and "running" in r for r in rows)
    assert any(
        "`pipeline:regressed`" in r and "failing checks: t::p" in r and "failing steps: virgil.step" in r
        for r in rows
    )
    assert any(
        "`pipeline:trans_regressed`" in r
        and "failing steps: virgil.relies_step" in r
        and "failing via: crosscheck.mid_ref" in r
        for r in rows
    )


def test_via_causes_expands_transitive_failing_and_bug_reference_dependencies(trust, tmp_path):
    graph = {
        "nodes": {
            "virgil.x": {"layer": "models"},
            "crosscheck.mid": {"layer": "references", "depends": ["crosscheck.leaf"]},
            "crosscheck.leaf": {"layer": "references"},
        }
    }
    # 1. Intermediate reference passes its own check ("ok"), leaf reference fails ("failing"):
    # virgil.x must become "relies" (not "relies-unverified") and name crosscheck.leaf in because.
    recs_fail = [
        record("t::rm", "crosscheck.mid", kind="reference"),
        record("t::rl", "crosscheck.leaf", kind="reference", outcome="failed"),
        record("t::vx", "virgil.x", via=["crosscheck.mid"]),
    ]
    got, model = verdicts(trust, tmp_path, recs_fail, graph)
    assert got == {
        "virgil.x": "relies",
        "crosscheck.mid": "relies",
        "crosscheck.leaf": "failing",
    }
    assert model["nodes"]["virgil.x"]["because"] == [
        {"id": "crosscheck.leaf", "findings": [], "status": "failing"},
    ]

    # 2. Intermediate reference is "unchecked" while leaf has an open crosscheck bug:
    # both the unchecked intermediate and the buggy leaf surface, and virgil.x is still "relies".
    ledger_bug = [{"id": "C99", "ruling": "crosscheck", "status": "open", "objects": ["crosscheck.leaf"]}]
    recs_bug = [
        record("t::rl_ok", "crosscheck.leaf", kind="reference"),
        record("t::vx", "virgil.x", via=["crosscheck.mid"]),
    ]
    got_bug, model_bug = verdicts(trust, tmp_path, recs_bug, graph, ledger=ledger_bug)
    assert got_bug["crosscheck.leaf"] == "bug"
    assert got_bug["virgil.x"] == "relies"
    assert model_bug["nodes"]["virgil.x"]["because"] == [
        {"id": "crosscheck.leaf", "findings": ["C99"], "status": "bug"},
        {"id": "crosscheck.mid", "findings": [], "status": "unchecked"},
    ]


def test_unfixed_crosscheck_ruling_marks_reference_bug_and_blocks_virgil_verified(trust, tmp_path):
    graph = {
        "nodes": {
            "crosscheck.r": {"layer": "references"},
            "virgil.caller": {"layer": "models"},
            "virgil.direct": {"layer": "models"},
        }
    }
    recs = [
        record("t::r_pass", "crosscheck.r", kind="reference"),
        record("t::r_xfail", "crosscheck.r", "finding", outcome="xfailed"),
        record("t::caller", "virgil.caller", via=["crosscheck.r"]),
        record("t::direct", "virgil.direct"),
    ]
    ledger_open = [
        {"id": "C10", "ruling": "crosscheck", "status": "open", "objects": ["crosscheck.r"]},
        {"id": "C11", "ruling": "crosscheck", "status": "open", "objects": ["virgil.direct"]},
    ]
    got, model = verdicts(trust, tmp_path, recs, graph, ledger=ledger_open)
    assert got["crosscheck.r"] == "bug"
    assert model["nodes"]["crosscheck.r"]["findings"] == ["C10"]
    assert got["virgil.caller"] == "relies"
    assert model["nodes"]["virgil.caller"]["because"] == [
        {"id": "crosscheck.r", "findings": ["C10"], "status": "bug"},
    ]
    # Non-reference node with an open test-side ruling cannot be "verified", nor mislabelled "bug"
    assert got["virgil.direct"] == "partly"
    assert model["nodes"]["virgil.direct"]["findings"] == ["C11"]

    ledger_fixed = [{**e, "status": "fixed"} for e in ledger_open]
    got_fixed, _ = verdicts(trust, tmp_path, recs, graph, ledger=ledger_fixed)
    assert got_fixed == {
        "crosscheck.r": "verified",
        "virgil.caller": "verified",
        "virgil.direct": "verified",
    }


def test_module_via_follows_local_helpers_rejects_shadowed_workers_and_resolves_subpackages(tmp_path, monkeypatch):
    from evidence import plugin

    # 1. Local sibling tests/ and scripts/ helper imports propagate transitive reference & worker dependencies
    tests_dir = tmp_path / "tests"
    scripts_dir = tmp_path / "scripts"
    tests_dir.mkdir()
    scripts_dir.mkdir()
    (scripts_dir / "helper_script.py").write_text(
        "from crosscheck import alma_maxent\n",
        encoding="utf-8",
    )
    (tests_dir / "helper_test.py").write_text(
        "from external_bridge import _subprocess as sp\n"
        "from virgil_bridge import mpol\n"
        "def run_mpol():\n"
        "    return sp.run('mpol', 'mpol_worker.py', {})\n",
        encoding="utf-8",
    )
    caller = tests_dir / "test_caller.py"
    caller.write_text(
        "import helper_script\n"
        "from helper_test import run_mpol\n"
        "from crosscheck import alma_clean\n",
        encoding="utf-8",
    )
    plugin._module_via.cache_clear()
    assert plugin._module_via(str(caller)) == (
        "crosscheck.alma_clean",
        "crosscheck.alma_maxent",
        "external_bridge.mpol_worker",
        "virgil_bridge.mpol",
    )

    # 2. Reassigned module constants, shadowed locals/parameters, and mutated wrapper args emit _UNRESOLVED_WORKER
    for idx, src in enumerate([
        # Reassigned in conditional branch at module scope
        "from external_bridge import _subprocess as sp\n"
        "WORKER = 'alma_worker.py'\n"
        "if True:\n"
        "    WORKER = 'mpol_worker.py'\n"
        "sp.run('alma', WORKER, {})\n",
        # Shadowed by local variable inside function
        "from external_bridge import _subprocess as sp\n"
        "WORKER = 'alma_worker.py'\n"
        "def f(dyn):\n"
        "    WORKER = dyn\n"
        "    return sp.run('alma', WORKER, {})\n",
        # Wrapper mutates its parameter before calling sp.run
        "from external_bridge import _subprocess as sp\n"
        "def wrap(w):\n"
        "    w = w + '.py'\n"
        "    return sp.run('alma', w, {})\n"
        "wrap('alma_worker')\n",
    ]):
        p = tmp_path / f"bad_worker_{idx}.py"
        p.write_text(src, encoding="utf-8")
        plugin._module_via.cache_clear()
        assert plugin._module_via(str(p)) == (plugin._UNRESOLVED_WORKER,)

    # 3. Nested subpackages and level >= 2 relative imports resolve at full depth
    fake_src = tmp_path / "src"
    (fake_src / "crosscheck" / "sub").mkdir(parents=True)
    (fake_src / "crosscheck" / "__init__.py").write_text("", encoding="utf-8")
    (fake_src / "crosscheck" / "sky.py").write_text("", encoding="utf-8")
    (fake_src / "crosscheck" / "limb.py").write_text("", encoding="utf-8")
    (fake_src / "crosscheck" / "sub" / "__init__.py").write_text("", encoding="utf-8")
    (fake_src / "crosscheck" / "sub" / "baz.py").write_text("", encoding="utf-8")
    bar_py = fake_src / "crosscheck" / "sub" / "bar.py"
    bar_py.write_text(
        "from . import baz\n"
        "from .. import sky\n"
        "from ..limb import disk\n",
        encoding="utf-8",
    )
    bad_rel = fake_src / "crosscheck" / "sub" / "escape.py"
    bad_rel.write_text("from ... import nope\n", encoding="utf-8")
    ext_caller = tmp_path / "test_subpkg.py"
    ext_caller.write_text(
        "import crosscheck.sub.bar\n"
        "from crosscheck.sub import baz\n"
        "from crosscheck.sub.missing import fn\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(plugin, "_SRC_ROOT", fake_src)
    plugin._module_via.cache_clear()
    assert plugin._module_via(str(bar_py)) == (
        "crosscheck.limb",
        "crosscheck.sky",
        "crosscheck.sub.baz",
    )
    assert plugin._module_via(str(bad_rel)) == ("crosscheck.<invalid_relative_import>",)
    assert plugin._module_via(str(ext_caller)) == (
        "crosscheck.sub.bar",
        "crosscheck.sub.baz",
        "crosscheck.sub.missing",
    )


def test_module_via_resolves_call_imports_and_wrapper_id_shadowing(tmp_path, monkeypatch):
    from evidence import plugin

    tests_dir = tmp_path / "tests"
    scripts_dir = tmp_path / "scripts"
    tests_dir.mkdir()
    scripts_dir.mkdir()
    # Cloned external repos under .external/ (such as MPoL-dev/examples on CI) must never collide
    # with _is_provably_external_py_suffix or _repo_py_suffixes.
    ext_sgd = tmp_path / ".external" / "mpol-examples" / "04-IMLup-multi-EB" / "mpol_analysis" / "src" / "sgd.py"
    ext_sgd.parent.mkdir(parents=True)
    ext_sgd.write_text("class Net: pass\n", encoding="utf-8")
    (scripts_dir / "loaded_script.py").write_text(
        "from crosscheck import informed_starts\n",
        encoding="utf-8",
    )
    (scripts_dir / "param_script.py").write_text(
        "from crosscheck import preregistration\n",
        encoding="utf-8",
    )

    # 1. Resolved call imports: importorskip, import_module (incl. unaliased dotted import), __import__,
    #    spec_from_file_location + wrappers + external .external/ suffix exclusion
    ok_file = tests_dir / "test_call_imports.py"
    ok_file.write_text(
        "import importlib.metadata\n"
        "import importlib.util\n"
        "import pathlib\n"
        "import pytest\n"
        "vb = pytest.importorskip('virgil_bridge')\n"
        "sp_mod = __import__('crosscheck.alma_speed', fromlist=['x'])\n"
        "def _try(name):\n"
        "    return importlib.import_module(name)\n"
        "ac = _try('crosscheck.alma_clean')\n"
        "SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / 'scripts'\n"
        "def _load(name):\n"
        "    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')\n"
        "    return spec\n"
        "s1 = _load('loaded_script')\n"
        "def _load_ext(examples_dir):\n"
        "    src = f'{examples_dir}/04-IMLup-multi-EB/mpol_analysis/src'\n"
        "    return importlib.util.spec_from_file_location('imlup_sgd', f'{src}/sgd.py')\n"
        "CASES = [('param_script', 1)]\n"
        "@pytest.mark.parametrize('sname,flag', CASES)\n"
        "def test_p(sname, flag):\n"
        "    return _load(sname)\n",
        encoding="utf-8",
    )
    plugin._module_via.cache_clear()
    plugin._repo_py_suffixes.cache_clear()
    assert plugin._module_via(str(ok_file)) == (
        "crosscheck.alma_clean",
        "crosscheck.alma_speed",
        "crosscheck.informed_starts",
        "crosscheck.limb",
        "crosscheck.preregistration",
        "crosscheck.sky",
        "virgil_bridge",
    )
    with monkeypatch.context() as m:
        m.setattr(plugin, "_REPO_ROOT", tmp_path)
        plugin._repo_py_suffixes.cache_clear()
        assert not any(s.startswith(".external/") for s in plugin._repo_py_suffixes())
        plugin._module_via.cache_clear()
        assert plugin._UNRESOLVED_IMPORT not in plugin._module_via(str(ok_file))
    plugin._repo_py_suffixes.cache_clear()

    # 2. Fail-closed dynamic import/spec calls, *args/**kwargs unpacking, and first-class callables emit _UNRESOLVED_IMPORT
    for idx, src in enumerate([
        "import importlib\ndef f(x):\n    return importlib.import_module(x)\n",
        "import pytest\ndef f(x):\n    return pytest.importorskip(x)\n",
        "import importlib.util\ndef f(d):\n    return importlib.util.spec_from_file_location('m', f'{d}/sky.py')\n",
        "import importlib\nfn = importlib.import_module\n",
        "import importlib.util\ndef f(ext, ldr):\n    return importlib.util.spec_from_file_location('m', f'{ext}/no_such_dir_xyz/no_such_mod.py', loader=ldr)\n",
        "import importlib.util\ndef f(args):\n    return importlib.util.spec_from_file_location(*args, 'decoy.py')\n",
        "import importlib\ndef f(kw):\n    return importlib.import_module('crosscheck.sky', **kw)\n",
    ]):
        bad_p = tests_dir / f"test_bad_call_{idx}.py"
        bad_p.write_text(src, encoding="utf-8")
        plugin._module_via.cache_clear()
        assert plugin._module_via(str(bad_p)) == (plugin._UNRESOLVED_IMPORT,)

    # 3. Method or duplicate function sharing a top-level wrapper's name must NOT suppress _UNRESOLVED_WORKER
    shadow_method = tests_dir / "test_method_wrap.py"
    shadow_method.write_text(
        "from external_bridge import _subprocess as sp\n"
        "def run_w(w):\n"
        "    return sp.run('alma', w, {})\n"
        "run_w('alma_worker.py')\n"
        "class Sneaky:\n"
        "    def run_w(self, dyn):\n"
        "        return sp.run('alma', dyn, {})\n",
        encoding="utf-8",
    )
    plugin._module_via.cache_clear()
    assert plugin._UNRESOLVED_WORKER in plugin._module_via(str(shadow_method))

    dup_fn = tests_dir / "test_dup_wrap.py"
    dup_fn.write_text(
        "from external_bridge import _subprocess as sp\n"
        "def run_w(w):\n"
        "    return sp.run('alma', w, {})\n"
        "def run_w(dyn):\n"
        "    return sp.run('alma', dyn, {})\n"
        "run_w('alma_worker.py')\n",
        encoding="utf-8",
    )
    plugin._module_via.cache_clear()
    assert plugin._UNRESOLVED_WORKER in plugin._module_via(str(dup_fn))


def test_item_via_resolves_spec_from_file_location_in_campaign_suites():
    from evidence import plugin

    plugin._module_via.cache_clear()
    pull_via = set(
        plugin._item_via(
            str(ROOT / "tests" / "test_pull_campaign.py"),
            "test_pull_campaign_pulls_are_standard_normal",
            None,
            (),
        )
    )
    assert {"crosscheck.preregistration", "crosscheck.simulate", "crosscheck.sky", "virgil_bridge"} <= pull_via

    det_via = set(
        plugin._item_via(
            str(ROOT / "tests" / "test_detection_campaign.py"),
            "test_detection_campaign_meets_its_criteria",
            None,
            (),
        )
    )
    assert {"crosscheck.preregistration", "crosscheck.simulate", "crosscheck.sky", "crosscheck.chi2"} <= det_via

    sbc1_via = set(
        plugin._item_via(
            str(ROOT / "tests" / "test_sbc_numpyro.py"),
            "test_sbc_ranks_are_uniform",
            None,
            (),
        )
    )
    assert {"crosscheck.preregistration", "crosscheck.simulate", "crosscheck.sky"} <= sbc1_via

    sbc2_via = set(
        plugin._item_via(
            str(ROOT / "tests" / "test_sbc_v2.py"),
            "test_sbc_v2_ranks_are_uniform_and_the_sampler_healthy",
            None,
            (),
        )
    )
    assert {"crosscheck.preregistration", "crosscheck.simulate", "crosscheck.sky"} <= sbc2_via

    for py in sorted((ROOT / "tests").rglob("test_*.py")):
        mvia = set(plugin._module_via(str(py)))
        assert plugin._UNRESOLVED_IMPORT not in mvia, (py, mvia)
        assert plugin._UNRESOLVED_WORKER not in mvia, (py, mvia)


def test_cross_module_subprocess_wrappers_propagate_and_fail_closed(tmp_path):
    from evidence import plugin

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    # Case A: helper calls its own wrapper internally AND caller calls it via module attribute
    (tests_dir / "helper_called.py").write_text(
        "from external_bridge import _subprocess as sp\n"
        "def _run(worker, task):\n"
        "    return sp.run('alma', worker, task)\n"
        "_run('alma_worker.py', {})\n",
        encoding="utf-8",
    )
    mod_caller_ok = tests_dir / "test_mod_caller_ok.py"
    mod_caller_ok.write_text(
        "import helper_called\n"
        "def test_both():\n"
        "    return helper_called._run('mpol_worker.py', {})\n",
        encoding="utf-8",
    )
    plugin._module_via.cache_clear()
    assert plugin._item_via(str(mod_caller_ok), "test_both", None, ()) == (
        "external_bridge.alma_worker",
        "external_bridge.mpol_worker",
    )

    for idx, bad_src in enumerate([
        "import helper_called\ndef test_dyn(w):\n    return helper_called._run(w, {})\n",
        "import helper_called\ndef test_attr_escape():\n    fn = helper_called._run\n    return fn\n",
        "import helper_called\ndef test_mod_escape():\n    m = helper_called\n    return m\n",
    ]):
        bad_f = tests_dir / f"test_mod_bad_{idx}.py"
        bad_f.write_text(bad_src, encoding="utf-8")
        plugin._module_via.cache_clear()
        fn_name = bad_src.split("def ")[1].split("(")[0]
        assert plugin._UNRESOLVED_WORKER in plugin._item_via(str(bad_f), fn_name, None, ())

    # Case B: helper defines an uncalled wrapper; caller imports symbol directly and passes a static literal
    (tests_dir / "helper_uncalled.py").write_text(
        "from external_bridge import _subprocess as sp\n"
        "def _run(worker, task):\n"
        "    return sp.run('alma', worker, task)\n",
        encoding="utf-8",
    )
    sym_caller_ok = tests_dir / "test_sym_caller_ok.py"
    sym_caller_ok.write_text(
        "from helper_uncalled import _run\n"
        "def test_static():\n"
        "    return _run('mpol_worker.py', {})\n"
        "def test_other():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    mod_uncalled_ok = tests_dir / "test_mod_uncalled_ok.py"
    mod_uncalled_ok.write_text(
        "import helper_uncalled\n"
        "def test_static():\n"
        "    return helper_uncalled._run('ehtim_worker.py', {})\n",
        encoding="utf-8",
    )
    plugin._module_via.cache_clear()
    assert plugin._module_via(str(sym_caller_ok)) == ("external_bridge.mpol_worker",)
    assert plugin._item_via(str(sym_caller_ok), "test_static", None, ()) == ("external_bridge.mpol_worker",)
    assert plugin._item_via(str(sym_caller_ok), "test_other", None, ()) == ()
    assert plugin._item_via(str(mod_uncalled_ok), "test_static", None, ()) == ("external_bridge.ehtim_worker",)

    for idx, bad_sym_src in enumerate([
        "from helper_uncalled import _run\ndef test_dyn(w):\n    return _run(w, {})\n",
        "from helper_uncalled import _run\ndef test_escape():\n    f = _run\n    return f\n",
        "from helper_uncalled import _run\n",
    ]):
        bad_sf = tests_dir / f"test_sym_bad_{idx}.py"
        bad_sf.write_text(bad_sym_src, encoding="utf-8")
        plugin._module_via.cache_clear()
        assert plugin._UNRESOLVED_WORKER in plugin._module_via(str(bad_sf))


def test_item_via_walks_ancestor_conftest_and_fixture_args(tmp_path):
    from evidence import plugin

    sub_dir = tmp_path / "tests" / "sub" / "deep"
    sub_dir.mkdir(parents=True)

    # Root tests/conftest.py has both a top-level reference import and a helper fixture (plus an unused fixture)
    (tmp_path / "tests" / "conftest.py").write_text(
        "import pytest\n"
        "from crosscheck import limb\n"
        "\n"
        "@pytest.fixture\n"
        "def root_helper():\n"
        "    from crosscheck import sky\n"
        "    return sky\n"
        "\n"
        "@pytest.fixture\n"
        "def unused_root():\n"
        "    from external_bridge import pmoired_models\n"
        "    return pmoired_models\n",
        encoding="utf-8",
    )
    # Intermediate conftest.py has an autouse fixture chaining into root_helper via parameter list
    (tmp_path / "tests" / "sub" / "conftest.py").write_text(
        "import pytest\n"
        "\n"
        "@pytest.fixture(autouse=True)\n"
        "def sub_auto(root_helper):\n"
        "    return root_helper\n",
        encoding="utf-8",
    )
    # Deep test file also chains an in-file autouse fixture into an in-file helper fixture via args
    deep_test = sub_dir / "test_deep.py"
    deep_test.write_text(
        "import pytest\n"
        "\n"
        "@pytest.fixture\n"
        "def local_dep():\n"
        "    from crosscheck import radial\n"
        "    return radial\n"
        "\n"
        "@pytest.fixture(autouse=True)\n"
        "def local_auto(local_dep):\n"
        "    return local_dep\n"
        "\n"
        "def test_leaf():\n"
        "    assert True\n",
        encoding="utf-8",
    )
    plugin._module_via.cache_clear()
    assert plugin._item_via(str(deep_test), "test_leaf", None, ()) == (
        "crosscheck.limb",
        "crosscheck.radial",
        "crosscheck.sky",
    )

    # Also verify conftest.py with zero fixtures (c_defs empty) still contributes its top-level imports
    plain_dir = tmp_path / "plain_suite" / "tests" / "nested"
    plain_dir.mkdir(parents=True)
    (tmp_path / "plain_suite" / "tests" / "conftest.py").write_text(
        "from crosscheck import chi2\n",
        encoding="utf-8",
    )
    plain_test = plain_dir / "test_plain.py"
    plain_test.write_text("def test_ok():\n    assert 1 == 1\n", encoding="utf-8")
    plugin._module_via.cache_clear()
    assert plugin._item_via(str(plain_test), "test_ok", None, ()) == ("crosscheck.chi2",)


def test_cross_module_subprocess_reexports_and_subpackage_init_fail_closed(tmp_path, monkeypatch):
    from evidence import plugin

    fake_repo = tmp_path / "repo"
    fake_src = fake_repo / "src"
    fake_tests = fake_repo / "tests"
    (fake_src / "external_bridge").mkdir(parents=True)
    (fake_src / "external_bridge" / "__init__.py").write_text("", encoding="utf-8")
    (fake_src / "external_bridge" / "_subprocess.py").write_text(
        "def run(env, worker, task):\n    return None\n"
        "def available(env, mod=None):\n    return True\n",
        encoding="utf-8",
    )
    subpkg = fake_src / "crosscheck" / "subpkg"
    subpkg.mkdir(parents=True)
    (fake_src / "crosscheck" / "__init__.py").write_text("", encoding="utf-8")
    (subpkg / "__init__.py").write_text(
        "from external_bridge._subprocess import run as sp_run\n"
        "from external_bridge import _subprocess as sp\n"
        "def pkg_wrap(worker):\n"
        "    return sp.run('alma', worker, {})\n",
        encoding="utf-8",
    )
    fake_tests.mkdir(parents=True)
    (fake_tests / "helper_reexport.py").write_text(
        "from external_bridge._subprocess import run as sp_run\n"
        "from external_bridge import _subprocess as sp\n"
        "import external_bridge as eb\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(plugin, "_REPO_ROOT", fake_repo)
    monkeypatch.setattr(plugin, "_SRC_ROOT", fake_src)
    plugin._module_via.cache_clear()

    # Static callers through run_aliases, sp_aliases, eb_aliases, and subpackage __init__.py
    ok_file = fake_tests / "test_reexport_ok.py"
    ok_file.write_text(
        "from helper_reexport import sp_run, sp\n"
        "import helper_reexport as hr\n"
        "from crosscheck import subpkg\n"
        "from crosscheck.subpkg import pkg_wrap\n"
        "def test_all():\n"
        "    sp_run('mpol', 'mpol_worker.py', {})\n"
        "    sp.run('alma', 'alma_worker.py', {})\n"
        "    hr.sp.run('ehtim', 'ehtim_worker.py', {})\n"
        "    hr.eb._subprocess.run('candid', 'candid_worker.py', {})\n"
        "    subpkg.pkg_wrap('fouriever_worker.py')\n"
        "    pkg_wrap('orbitize_worker.py')\n",
        encoding="utf-8",
    )
    plugin._module_via.cache_clear()
    assert plugin._item_via(str(ok_file), "test_all", None, ()) == (
        "crosscheck.subpkg",
        "external_bridge.alma_worker",
        "external_bridge.candid_worker",
        "external_bridge.ehtim_worker",
        "external_bridge.fouriever_worker",
        "external_bridge.mpol_worker",
        "external_bridge.orbitize_worker",
    )

    # Dynamic worker args, escaped callables/modules, and uncalled sp_run must emit _UNRESOLVED_WORKER
    for idx, bad_src in enumerate([
        "from helper_reexport import sp_run\ndef test_b(w):\n    return sp_run('mpol', w, {})\n",
        "import helper_reexport\ndef test_b(w):\n    return helper_reexport.sp.run('mpol', w, {})\n",
        "import helper_reexport\ndef test_b(w):\n    return helper_reexport.eb._subprocess.run('mpol', w, {})\n",
        "from helper_reexport import sp\ndef test_b(w):\n    return sp.run('mpol', w, {})\n",
        "import helper_reexport\ndef test_b():\n    s = helper_reexport.sp\n    return s\n",
        "import helper_reexport\ndef test_b():\n    fn = helper_reexport.sp.run\n    return fn\n",
        "from helper_reexport import sp_run\ndef test_b():\n    return 1\n",
        "from crosscheck import subpkg\ndef test_b(w):\n    return subpkg.sp_run('mpol', w, {})\n",
        "import crosscheck.subpkg as spkg\ndef test_b():\n    m = spkg\n    return m\n",
    ]):
        bf = fake_tests / f"test_reexport_bad_{idx}.py"
        bf.write_text(bad_src, encoding="utf-8")
        plugin._module_via.cache_clear()
        assert plugin._UNRESOLVED_WORKER in plugin._item_via(str(bf), "test_b", None, ()), (idx, bad_src)
        assert plugin._UNRESOLVED_WORKER in plugin._module_via(str(bf)), (idx, bad_src)


def test_spec_from_file_location_requires_leading_slash_for_external_suffix(tmp_path):
    from evidence import plugin

    tests_dir = tmp_path / "tests"
    src_cc = tmp_path / "src" / "crosscheck"
    tests_dir.mkdir(parents=True)
    src_cc.mkdir(parents=True)
    (src_cc / "alma.py").write_text("X = 1\n", encoding="utf-8")

    # Mid-segment dynamic suffix f"{dyn_prefix}check/alma.py" must NOT be treated as external
    mid_seg = tests_dir / "test_mid_seg.py"
    mid_seg.write_text(
        "import importlib.util\n"
        "def test_mid(dyn_prefix):\n"
        "    return importlib.util.spec_from_file_location('m', f'{dyn_prefix}check/alma.py')\n",
        encoding="utf-8",
    )
    # Segment-aligned suffix matching an in-repo file must also fail closed
    repo_seg = tests_dir / "test_repo_seg.py"
    repo_seg.write_text(
        "import importlib.util\n"
        "def test_repo(dyn_prefix):\n"
        "    return importlib.util.spec_from_file_location('m', f'{dyn_prefix}/crosscheck/alma.py')\n",
        encoding="utf-8",
    )
    # Segment-aligned suffix starting with '/' that matches no repo file is provably external
    ext_seg = tests_dir / "test_ext_seg.py"
    ext_seg.write_text(
        "import importlib.util\n"
        "def test_ext(dyn_prefix):\n"
        "    return importlib.util.spec_from_file_location('m', f'{dyn_prefix}/nonrepo_vendor_pkg/ext_only_tool.py')\n",
        encoding="utf-8",
    )
    plugin._module_via.cache_clear()
    assert plugin._UNRESOLVED_IMPORT in plugin._item_via(str(mid_seg), "test_mid", None, ())
    assert plugin._UNRESOLVED_IMPORT in plugin._item_via(str(repo_seg), "test_repo", None, ())
    assert plugin._item_via(str(ext_seg), "test_ext", None, ()) == ()


def test_cross_module_dynamic_import_helpers_propagate_and_fail_closed(tmp_path):
    from evidence import plugin

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True)

    # 1. Helper calls its own wrapper on 'math' AND re-exports import_module and importlib
    (tests_dir / "dyn_helper_called.py").write_text(
        "import importlib\n"
        "import importlib.util\n"
        "import pathlib\n"
        "from importlib import import_module as reexported_imp\n"
        "\n"
        "def load_mod(name):\n"
        "    return importlib.import_module(name)\n"
        "\n"
        "load_mod('math')\n"
        "reexported_imp('math')\n",
        encoding="utf-8",
    )
    # 2. Helper defines uncalled import and spec wrappers (using helper-local __file__)
    (tests_dir / "dyn_helper_uncalled.py").write_text(
        "import importlib\n"
        "import importlib.util\n"
        "import pathlib\n"
        "from pytest import importorskip as skip_or_imp\n"
        "\n"
        "SRC = pathlib.Path(__file__).resolve().parents[1]\n"
        "\n"
        "def wrap_imp(modname):\n"
        "    return importlib.import_module(modname)\n"
        "\n"
        "def wrap_spec(stem):\n"
        "    return importlib.util.spec_from_file_location(stem, f'{SRC}/unused_{stem}.py')\n",
        encoding="utf-8",
    )
    (tmp_path / "unused_helper_target.py").write_text("from crosscheck import sky\n", encoding="utf-8")

    caller_ok = tests_dir / "test_dyn_caller_ok.py"
    caller_ok.write_text(
        "from dyn_helper_called import load_mod, reexported_imp\n"
        "import dyn_helper_called as dhc\n"
        "from dyn_helper_uncalled import wrap_imp, wrap_spec, skip_or_imp\n"
        "\n"
        "def test_static():\n"
        "    load_mod('crosscheck.alma')\n"
        "    reexported_imp('crosscheck.limb')\n"
        "    dhc.load_mod('crosscheck.radial')\n"
        "    dhc.importlib.import_module('virgil_bridge.alma')\n"
        "    wrap_imp('crosscheck.chi2')\n"
        "    skip_or_imp('external_bridge.pmoired_models')\n"
        "    wrap_spec('helper_target')\n",
        encoding="utf-8",
    )
    plugin._module_via.cache_clear()
    assert plugin._item_via(str(caller_ok), "test_static", None, ()) == (
        "crosscheck.alma",
        "crosscheck.chi2",
        "crosscheck.limb",
        "crosscheck.radial",
        "crosscheck.sky",
        "external_bridge.pmoired_models",
        "virgil_bridge.alma",
    )

    for idx, bad_src in enumerate([
        "from dyn_helper_called import load_mod\ndef test_b(m):\n    return load_mod(m)\n",
        "from dyn_helper_called import reexported_imp\ndef test_b(m):\n    return reexported_imp(m)\n",
        "import dyn_helper_called as dhc\ndef test_b(m):\n    return dhc.load_mod(m)\n",
        "import dyn_helper_called as dhc\ndef test_b(m):\n    return dhc.importlib.import_module(m)\n",
        "from dyn_helper_uncalled import wrap_spec\ndef test_b(s):\n    return wrap_spec(s)\n",
        "from dyn_helper_called import load_mod\ndef test_b():\n    fn = load_mod\n    return fn\n",
        "import dyn_helper_called as dhc\ndef test_b():\n    fn = dhc.load_mod\n    return fn\n",
        "import dyn_helper_called as dhc\ndef test_b():\n    m = dhc\n    return m\n",
        "from dyn_helper_uncalled import wrap_imp\ndef test_b():\n    return 1\n",
    ]):
        bf = tests_dir / f"test_dyn_bad_{idx}.py"
        bf.write_text(bad_src, encoding="utf-8")
        plugin._module_via.cache_clear()
        assert plugin._UNRESOLVED_IMPORT in plugin._item_via(str(bf), "test_b", None, ()), (idx, bad_src)
        assert plugin._UNRESOLVED_IMPORT in plugin._module_via(str(bf)), (idx, bad_src)


def test_item_via_conftest_top_helpers_pytest_hooks_and_usefixtures(tmp_path):
    from evidence import plugin

    tests_dir = tmp_path / "tests"
    sub_dir = tests_dir / "sub"
    sub_dir.mkdir(parents=True)

    # Root conftest.py calls a local helper at import time AND defines a pytest_* hook calling another helper,
    # while also defining an unused fixture (whose name matches a hook parameter to check hook args are not fixtures)
    (tests_dir / "conftest.py").write_text(
        "import pytest\n"
        "from external_bridge import _subprocess as sp\n"
        "\n"
        "def _make_shared():\n"
        "    from crosscheck import alma\n"
        "    sp.run('mpol', 'mpol_worker.py', {})\n"
        "    return alma\n"
        "\n"
        "SHARED = _make_shared()\n"
        "\n"
        "def _hook_helper():\n"
        "    from crosscheck import chi2\n"
        "    return chi2\n"
        "\n"
        "def pytest_runtest_setup(item):\n"
        "    return _hook_helper()\n"
        "\n"
        "@pytest.fixture\n"
        "def item():\n"
        "    from external_bridge import pmoired_models\n"
        "    return pmoired_models\n"
        "\n"
        "@pytest.fixture\n"
        "def mod_uf():\n"
        "    from crosscheck import limb\n"
        "    return limb\n"
        "\n"
        "@pytest.fixture\n"
        "def cls_uf():\n"
        "    from crosscheck import radial\n"
        "    return radial\n"
        "\n"
        "@pytest.fixture\n"
        "def chained_uf():\n"
        "    from virgil_bridge import alma as vb_alma\n"
        "    return vb_alma\n"
        "\n"
        "@pytest.fixture\n"
        "@pytest.mark.usefixtures('chained_uf')\n"
        "def fn_uf():\n"
        "    from crosscheck import sky\n"
        "    return sky\n",
        encoding="utf-8",
    )

    tfile = sub_dir / "test_marks.py"
    tfile.write_text(
        "import pytest\n"
        "\n"
        "pytestmark = [pytest.mark.usefixtures('mod_uf')]\n"
        "\n"
        "@pytest.mark.usefixtures('cls_uf')\n"
        "class TestSuite:\n"
        "    @pytest.mark.usefixtures('fn_uf')\n"
        "    def test_case(self):\n"
        "        assert True\n"
        "\n"
        "def test_plain():\n"
        "    assert True\n",
        encoding="utf-8",
    )

    plugin._module_via.cache_clear()
    # Direct _item_via without pytest-resolved fixture_names=() must still pick up conftest top helpers,
    # pytest_* hooks, and module/class/function/fixture usefixtures marks (excluding unused `item` fixture)
    assert plugin._item_via(str(tfile), "test_case", "TestSuite", ()) == (
        "crosscheck.alma",
        "crosscheck.chi2",
        "crosscheck.limb",
        "crosscheck.radial",
        "crosscheck.sky",
        "external_bridge.mpol_worker",
        "virgil_bridge.alma",
    )
    assert plugin._item_via(str(tfile), "test_plain", None, ()) == (
        "crosscheck.alma",
        "crosscheck.chi2",
        "crosscheck.limb",
        "external_bridge.mpol_worker",
    )


def test_local_helper_packages_from_dot_and_dotted_submodules_preserve_init_via(tmp_path):
    from evidence import plugin

    tests_dir = tmp_path / "tests"
    lpkg = tests_dir / "lpkg"
    sub = lpkg / "sub"
    sub.mkdir(parents=True)

    (lpkg / "__init__.py").write_text(
        "import importlib\n"
        "from crosscheck import limb\n"
        "from external_bridge import _subprocess as sp\n"
        "\n"
        "def pkg_sym():\n"
        "    from crosscheck import alma\n"
        "    return alma\n"
        "\n"
        "def pkg_sp_wrap(w):\n"
        "    return sp.run('ehtim', w, {})\n"
        "\n"
        "def pkg_imp_wrap(m):\n"
        "    return importlib.import_module(m)\n",
        encoding="utf-8",
    )
    (sub / "__init__.py").write_text(
        "from crosscheck import radial\n"
        "\n"
        "def sub_sym():\n"
        "    from crosscheck import chi2\n"
        "    return chi2\n",
        encoding="utf-8",
    )
    in_pkg = sub / "test_in_pkg.py"
    in_pkg.write_text(
        "from .. import pkg_sym, pkg_sp_wrap, pkg_imp_wrap\n"
        "from . import sub_sym\n"
        "\n"
        "def test_rel():\n"
        "    pkg_sym()\n"
        "    sub_sym()\n"
        "    pkg_sp_wrap('ehtim_worker.py')\n"
        "    pkg_imp_wrap('virgil_bridge.alma')\n",
        encoding="utf-8",
    )
    (sub / "leaf.py").write_text(
        "from crosscheck import sky\n"
        "\n"
        "def leaf_fn():\n"
        "    return sky\n",
        encoding="utf-8",
    )
    dotted_caller = tests_dir / "test_dotted_pkg.py"
    dotted_caller.write_text(
        "import lpkg.sub.leaf\n"
        "\n"
        "def test_dotted():\n"
        "    lpkg.pkg_sym()\n"
        "    lpkg.pkg_sp_wrap('candid_worker.py')\n"
        "    lpkg.pkg_imp_wrap('external_bridge.pmoired_models')\n"
        "    return lpkg.sub.leaf.leaf_fn()\n",
        encoding="utf-8",
    )
    aliased_dotted = tests_dir / "test_aliased_dotted_pkg.py"
    aliased_dotted.write_text(
        "import lpkg.sub.leaf as leaf_mod\n"
        "\n"
        "def test_aliased():\n"
        "    return leaf_mod.leaf_fn()\n",
        encoding="utf-8",
    )
    from_dotted = tests_dir / "test_from_dotted_pkg.py"
    from_dotted.write_text(
        "from lpkg.sub.leaf import leaf_fn\n"
        "\n"
        "def test_from_dotted():\n"
        "    return leaf_fn()\n",
        encoding="utf-8",
    )
    wild_dot = lpkg / "test_wild_dot.py"
    wild_dot.write_text(
        "from . import *\n"
        "\n"
        "def test_w():\n"
        "    return 1\n",
        encoding="utf-8",
    )
    dyn_dot = lpkg / "test_dyn_dot.py"
    dyn_dot.write_text(
        "from . import pkg_sp_wrap, pkg_imp_wrap\n"
        "\n"
        "def test_d(w, m):\n"
        "    pkg_sp_wrap(w)\n"
        "    pkg_imp_wrap(m)\n",
        encoding="utf-8",
    )

    plugin._module_via.cache_clear()
    assert plugin._item_via(str(in_pkg), "test_rel", None, ()) == (
        "crosscheck.alma",
        "crosscheck.chi2",
        "crosscheck.limb",
        "crosscheck.radial",
        "external_bridge.ehtim_worker",
        "virgil_bridge.alma",
    )
    assert plugin._item_via(str(dotted_caller), "test_dotted", None, ()) == (
        "crosscheck.alma",
        "crosscheck.chi2",
        "crosscheck.limb",
        "crosscheck.radial",
        "crosscheck.sky",
        "external_bridge.candid_worker",
        "external_bridge.pmoired_models",
    )
    assert plugin._item_via(str(aliased_dotted), "test_aliased", None, ()) == (
        "crosscheck.limb",
        "crosscheck.radial",
        "crosscheck.sky",
    )
    assert plugin._item_via(str(from_dotted), "test_from_dotted", None, ()) == (
        "crosscheck.limb",
        "crosscheck.radial",
        "crosscheck.sky",
    )
    wild_via = plugin._item_via(str(wild_dot), "test_w", None, ())
    assert plugin._UNRESOLVED_WORKER in wild_via
    assert plugin._UNRESOLVED_IMPORT in wild_via
    dyn_via = plugin._item_via(str(dyn_dot), "test_d", None, ())
    assert plugin._UNRESOLVED_WORKER in dyn_via
    assert plugin._UNRESOLVED_IMPORT in dyn_via


