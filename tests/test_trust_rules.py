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


def test_transitive_depends_into_ok_via_expands_broken_reference_to_relies_and_pipeline_failing(trust, tmp_path):
    spec = importlib.util.spec_from_file_location("trust_diff", ROOT / "scripts" / "trust_diff.py")
    td = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(td)

    graph = {
        "nodes": {
            "crosscheck.bad": {"layer": "references"},
            "virgil.mid": {"layer": "models"},
            "virgil.top": {"layer": "models", "depends": ["virgil.mid"]},
        },
        "pipelines": {
            "chain": {"title": "Chain", "steps": ["virgil.top"]},
        },
    }
    # 1. Failing reference -> virgil.mid (ok-via) -> virgil.top (depends, own ok) -> pipeline (steps=[virgil.top])
    recs_fail = [
        record("t::ref_fail", "crosscheck.bad", kind="reference", outcome="failed"),
        record("t::mid", "virgil.mid", via=["crosscheck.bad"]),
        record("t::top", "virgil.top"),
        record("t::pipe", "pipeline:chain"),
    ]
    got_fail, model_fail = verdicts(trust, tmp_path, recs_fail, graph)
    assert got_fail == {
        "crosscheck.bad": "failing",
        "virgil.mid": "relies",
        "virgil.top": "relies",
    }
    assert model_fail["nodes"]["virgil.top"]["because"] == [
        {"id": "crosscheck.bad", "findings": [], "status": "failing"},
    ]
    assert model_fail["pipelines"][0]["status"] == "failing"

    recs_ok = [
        record("t::ref_ok", "crosscheck.bad", kind="reference"),
        record("t::mid", "virgil.mid", via=["crosscheck.bad"]),
        record("t::top", "virgil.top"),
        record("t::pipe", "pipeline:chain"),
    ]
    _, model_ok = verdicts(trust, tmp_path, recs_ok, graph)
    assert model_ok["pipelines"][0]["status"] == "validated"
    diff_lines = td.diff_rows(model_ok, model_fail)
    assert any("`pipeline:chain`" in r and "failing steps: virgil.top" in r for r in diff_lines)

    # 2. Open crosscheck bug -> virgil.mid (ok-via) -> virgil.top (depends, own ok) -> pipeline ("passes")
    ledger_bug = [{"id": "C77", "ruling": "crosscheck", "status": "open", "objects": ["crosscheck.bad"]}]
    got_bug, model_bug = verdicts(trust, tmp_path, recs_ok, graph, ledger=ledger_bug)
    assert got_bug == {
        "crosscheck.bad": "bug",
        "virgil.mid": "relies",
        "virgil.top": "relies",
    }
    assert model_bug["nodes"]["virgil.top"]["because"] == [
        {"id": "crosscheck.bad", "findings": ["C77"], "status": "bug"},
    ]
    assert model_bug["pipelines"][0]["status"] == "passes"


def test_reference_nodes_fail_closed_on_unresolved_or_unregistered_via(trust, tmp_path):
    unresolved = "crosscheck.<unresolved_dynamic_import>"  # a name that is in no graph: a via that names it never verifies
    graph = {
        "nodes": {
            "crosscheck.sib": {"layer": "references"},
            "crosscheck.ok_ref": {"layer": "references"},
            "crosscheck.dyn_ref": {"layer": "references"},
            "crosscheck.unreg_ref": {"layer": "references"},
            "virgil.dep_dyn": {"layer": "models"},
            "virgil.dep_unreg": {"layer": "models"},
            "virgil.dep_ok": {"layer": "models"},
        },
        "pipelines": {},
    }
    recs = [
        record("t::sib", "crosscheck.sib", kind="reference", via=[]),
        record("t::ok_ref", "crosscheck.ok_ref", kind="reference", via=["crosscheck.sib"]),
        record("t::dyn_ref", "crosscheck.dyn_ref", kind="reference", via=[unresolved]),
        record("t::unreg_ref", "crosscheck.unreg_ref", kind="reference", via=["crosscheck.not_in_graph"]),
        record("t::dep_ok", "virgil.dep_ok", via=["crosscheck.ok_ref"]),
        record("t::dep_dyn", "virgil.dep_dyn", via=["crosscheck.dyn_ref"]),
        record("t::dep_unreg", "virgil.dep_unreg", via=["crosscheck.unreg_ref"]),
    ]
    got, model = verdicts(trust, tmp_path, recs, graph)
    assert got["crosscheck.ok_ref"] == "verified"
    assert got["virgil.dep_ok"] == "verified"
    assert got["crosscheck.dyn_ref"] == "relies-unverified"
    assert [b["id"] for b in model["nodes"]["crosscheck.dyn_ref"]["because"]] == [unresolved]
    assert got["virgil.dep_dyn"] == "relies-unverified"
    assert got["crosscheck.unreg_ref"] == "relies-unverified"
    assert [b["id"] for b in model["nodes"]["crosscheck.unreg_ref"]["because"]] == ["crosscheck.not_in_graph"]
    assert got["virgil.dep_unreg"] == "relies-unverified"


def test_supplementary_broken_via_check_does_not_demote_relies_unverified_node(trust, tmp_path):
    graph = {
        "nodes": {
            "crosscheck.unv": {"layer": "references"},
            "crosscheck.bad": {"layer": "references"},
            "virgil.one_root": {"layer": "models", "roots": 1},
            "virgil.two_roots": {"layer": "models", "roots": 2},
        },
        "pipelines": {
            "p_one": {"title": "One", "steps": ["virgil.one_root"]},
            "p_two": {"title": "Two", "steps": ["virgil.two_roots"]},
        },
    }
    base_recs = [
        record("t::bad", "crosscheck.bad", outcome="failed", kind="reference"),
        record("t::one", "virgil.one_root", roots=["mathematics"], via=["crosscheck.unv"]),
        record("t::p1", "pipeline:p_one", roots=["mathematics"], via=["crosscheck.unv"]),
        record("t::p2", "pipeline:p_two", roots=["mathematics"], via=["crosscheck.unv"]),
    ]
    got_before, before = verdicts(trust, tmp_path, base_recs, graph)
    assert got_before["virgil.one_root"] == "relies-unverified"
    assert {p["id"]: p["status"] for p in before["pipelines"]}["p_one"] == "passes"

    # Adding a supplementary check via a failing reference must NOT demote virgil.one_root to "relies"
    # or flip pipeline p_one to "failing", whereas virgil.two_roots genuinely needs both roots -> "relies".
    extra_recs = [
        *base_recs,
        record("t::extra", "virgil.one_root", roots=["dLux"], via=["crosscheck.bad"]),
        record("t::r1", "virgil.two_roots", roots=["mathematics"], via=["crosscheck.unv"]),
        record("t::r2", "virgil.two_roots", roots=["dLux"], via=["crosscheck.bad"]),
    ]
    got_after, after = verdicts(trust, tmp_path, extra_recs, graph)
    assert got_after["virgil.one_root"] == "relies-unverified"
    assert [b["id"] for b in after["nodes"]["virgil.one_root"]["because"]] == ["crosscheck.unv"]
    assert got_after["virgil.two_roots"] == "relies"
    assert {b["id"] for b in after["nodes"]["virgil.two_roots"]["because"]} == {
        "crosscheck.unv",
        "crosscheck.bad",
    }
    pipes = {p["id"]: p["status"] for p in after["pipelines"]}
    assert pipes["p_one"] == "passes"
    assert pipes["p_two"] == "failing"


def test_reference_checked_only_through_a_failing_reference_is_not_verified(trust, tmp_path):
    """A reference's check counts only through references that are themselves verified, so
    a reference vouched for only by a failing one is not verified, nor is the part above it."""
    graph = {"nodes": {
        "crosscheck.bad": {"layer": "references"},
        "crosscheck.top": {"layer": "references"},
        "virgil.a": {"layer": "models"},
    }}
    recs = [
        record("t::bad", "crosscheck.bad", kind="reference", outcome="failed"),
        record("t::top", "crosscheck.top", kind="reference", via=["crosscheck.bad"]),
        record("t::a", "virgil.a", via=["crosscheck.top"]),
    ]
    got, model = verdicts(trust, tmp_path, recs, graph)
    assert got == {"crosscheck.bad": "failing", "crosscheck.top": "relies", "virgil.a": "relies"}
    assert [b["id"] for b in model["nodes"]["crosscheck.top"]["because"]] == ["crosscheck.bad"]
    assert [b["id"] for b in model["nodes"]["virgil.a"]["because"]] == ["crosscheck.bad"]


def test_reference_checked_only_through_an_unchecked_reference_is_not_verified(trust, tmp_path):
    graph = {"nodes": {
        "crosscheck.empty": {"layer": "references"},
        "crosscheck.top": {"layer": "references"},
        "virgil.a": {"layer": "models"},
    }}
    recs = [
        record("t::top", "crosscheck.top", kind="reference", via=["crosscheck.empty"]),
        record("t::a", "virgil.a", via=["crosscheck.top"]),
    ]
    got, _ = verdicts(trust, tmp_path, recs, graph)
    assert got == {"crosscheck.empty": "unchecked", "crosscheck.top": "relies-unverified",
                   "virgil.a": "relies-unverified"}


def test_reference_checked_through_a_verified_reference_is_verified(trust, tmp_path):
    graph = {"nodes": {
        "crosscheck.base": {"layer": "references"},
        "crosscheck.top": {"layer": "references"},
        "virgil.a": {"layer": "models"},
    }}
    recs = [
        record("t::base", "crosscheck.base", kind="reference"),
        record("t::top", "crosscheck.top", kind="reference", via=["crosscheck.base"]),
        record("t::a", "virgil.a", via=["crosscheck.top"]),
    ]
    got, _ = verdicts(trust, tmp_path, recs, graph)
    assert got == {"crosscheck.base": "verified", "crosscheck.top": "verified", "virgil.a": "verified"}


def test_references_checked_only_through_each_other_are_not_verified(trust, tmp_path):
    """Two references each checked only through the other: nothing outside the pair backs
    either, so neither is verified, nor a part of virgil checked through them."""
    graph = {"nodes": {
        "crosscheck.a": {"layer": "references"},
        "crosscheck.b": {"layer": "references"},
        "virgil.x": {"layer": "models"},
    }}
    recs = [
        record("t::a", "crosscheck.a", kind="reference", via=["crosscheck.b"]),
        record("t::b", "crosscheck.b", kind="reference", via=["crosscheck.a"]),
        record("t::x", "virgil.x", via=["crosscheck.a"]),
    ]
    got, model = verdicts(trust, tmp_path, recs, graph)
    assert got == {"crosscheck.a": "relies-unverified", "crosscheck.b": "relies-unverified",
                   "virgil.x": "relies-unverified"}
    assert [b["id"] for b in model["nodes"]["virgil.x"]["because"]] == ["crosscheck.a"]
    # an independent check on one of them breaks the cycle: that one is verified, then the other
    recs.append(record("t::a2", "crosscheck.a", kind="reference", roots=["statistics"]))
    got, _ = verdicts(trust, tmp_path, recs, graph)
    assert got["crosscheck.a"] == "verified" and got["crosscheck.b"] == "verified" and got["virgil.x"] == "verified"


def test_depth_is_the_longest_depends_chain_and_nodes_list_their_depends(trust, tmp_path):
    nodes = {
        "virgil.leaf": {"layer": "models"},
        "virgil.mid": {"layer": "models", "depends": ["virgil.leaf"]},
        "virgil.top": {"layer": "models", "depends": ["virgil.leaf", "virgil.mid"]},  # longest chain wins
        "virgil.dangling": {"layer": "models", "depends": ["virgil.nowhere"]},  # not a node: ignored
        "crosscheck.r0": {"layer": "references"},
        "crosscheck.r1": {"layer": "references", "depends": ["crosscheck.r0"]},  # references count as nodes
        "virgil.c1": {"layer": "models", "depends": ["virgil.c2"]},  # a cycle shares one depth,
        "virgil.c2": {"layer": "models", "depends": ["virgil.c1", "virgil.top"]},  # counted from its way out
    }
    pipelines = {"p": {"title": "P", "steps": ["virgil.mid", "virgil.top"]},
                 "q": {"title": "Q", "steps": []}}
    want = {"virgil.leaf": 0, "virgil.mid": 1, "virgil.top": 2, "virgil.dangling": 0,
            "crosscheck.r0": 0, "crosscheck.r1": 1, "virgil.c1": 3, "virgil.c2": 3}
    for order in (list(nodes), list(reversed(list(nodes)))):  # the same whichever order they are listed in
        graph = {"nodes": {k: nodes[k] for k in order}, "pipelines": pipelines}
        _, model = verdicts(trust, tmp_path, [], graph)
        assert {k: v["depth"] for k, v in model["nodes"].items()} == want
    assert model["nodes"]["virgil.top"]["depends"] == ["virgil.leaf", "virgil.mid"]
    assert model["nodes"]["virgil.leaf"]["depends"] == []
    assert {p["id"]: (p["steps"], p["depth"]) for p in model["pipelines"]} == {
        "p": (["virgil.mid", "virgil.top"], 3), "q": ([], 1)}


def test_a_part_depending_on_one_checked_through_a_reference_cycle_is_not_verified(trust, tmp_path):
    """The cycle of references is the cause for whatever depends on it, not just for what is checked through it."""
    graph = {"nodes": {
        "crosscheck.a": {"layer": "references"},
        "crosscheck.b": {"layer": "references"},
        "crosscheck.c": {"layer": "references", "depends": ["crosscheck.a"]},
        "virgil.x": {"layer": "models"},
        "virgil.y": {"layer": "models", "depends": ["virgil.x"]},
        "virgil.z": {"layer": "models"},
    }}
    recs = [
        record("t::a", "crosscheck.a", kind="reference", via=["crosscheck.b"]),
        record("t::b", "crosscheck.b", kind="reference", via=["crosscheck.a"]),
        record("t::c", "crosscheck.c", kind="reference"),
        record("t::x", "virgil.x", via=["crosscheck.a"]),
        record("t::y", "virgil.y"),
        record("t::z", "virgil.z", via=["crosscheck.c"]),
    ]
    got, model = verdicts(trust, tmp_path, recs, graph)
    assert got["virgil.x"] == got["virgil.y"] == got["virgil.z"] == "relies-unverified"
    assert got["crosscheck.c"] == "relies-unverified"
    assert [b["id"] for b in model["nodes"]["virgil.y"]["because"]] == ["virgil.x"]
    # whatever the graph, nothing verified may depend on something that is not
    for node in model["nodes"].values():
        assert node["verdict"] != "verified" or all(
            model["nodes"][d]["verdict"] == "verified" for d in node["depends"])


def test_a_reference_cycle_is_named_beside_another_cause_and_at_the_end_of_a_chain(trust, tmp_path):
    """Review of #266: a reference cycle is a cause even when the part has another cause too, and on a chain
    ``virgil.x -> top -> a <-> b`` the cause is the cycle, not the hop ``top`` that merely passes it on."""
    graph = {"nodes": {
        "crosscheck.a": {"layer": "references"},
        "crosscheck.b": {"layer": "references"},
        "crosscheck.top": {"layer": "references"},
        "crosscheck.empty": {"layer": "references"},  # no checks at all
        "virgil.x": {"layer": "models"},
        "virgil.y": {"layer": "models"},
    }}
    recs = [
        record("t::a", "crosscheck.a", kind="reference", via=["crosscheck.b"]),
        record("t::b", "crosscheck.b", kind="reference", via=["crosscheck.a"]),
        record("t::top", "crosscheck.top", kind="reference", via=["crosscheck.a"]),
        record("t::x", "virgil.x", via=["crosscheck.top"]),
        record("t::y", "virgil.y", via=["crosscheck.empty", "crosscheck.a"]),
    ]
    got, model = verdicts(trust, tmp_path, recs, graph)
    assert got["virgil.x"] == got["virgil.y"] == "relies-unverified"
    cause_x = {b["id"] for b in model["nodes"]["virgil.x"]["because"]}
    assert cause_x and cause_x <= {"crosscheck.a", "crosscheck.b"}
    cause_y = {b["id"] for b in model["nodes"]["virgil.y"]["because"]}
    assert "crosscheck.empty" in cause_y and cause_y & {"crosscheck.a", "crosscheck.b"}


def test_the_causes_walk_visits_each_node_once_not_once_per_route(trust, tmp_path):
    """A layered lattice in which every reference is checked through both references of the layer below has
    2**layers routes to the bottom; the walk must not enumerate them (it did, and trust.py did not finish on the
    CI evidence)."""
    layers = 40
    nodes = {f"crosscheck.l{i}_{j}": {"layer": "references"} for i in range(layers) for j in (0, 1)}
    nodes["virgil.top"] = {"layer": "models"}
    recs = [record(f"t::l{i}_{j}", f"crosscheck.l{i}_{j}", kind="reference",
                   via=[f"crosscheck.l{i - 1}_0", f"crosscheck.l{i - 1}_1"] if i else [])
            for i in range(layers) for j in (0, 1)]
    recs[0]["kind"] = recs[1]["kind"] = "guard"  # the bottom layer is not checked: everything above relies on it
    recs.append(record("t::top", "virgil.top", via=[f"crosscheck.l{layers - 1}_0"]))
    got, model = verdicts(trust, tmp_path, recs, {"nodes": nodes})
    assert got["virgil.top"] == "relies-unverified"
    assert {b["id"] for b in model["nodes"]["virgil.top"]["because"]} == {"crosscheck.l0_0", "crosscheck.l0_1"}


def test_a_cycle_through_via_and_depends_blames_the_other_member_not_itself(trust, tmp_path):
    """Review of #266: ``a`` is checked via ``b``, and ``b`` depends on ``a``. Neither may list itself as its cause."""
    graph = {"nodes": {
        "crosscheck.a": {"layer": "references"},
        "crosscheck.b": {"layer": "references", "depends": ["crosscheck.a"]},
    }}
    recs = [
        record("t::a", "crosscheck.a", kind="reference", via=["crosscheck.b"]),
        record("t::b", "crosscheck.b", kind="reference"),
    ]
    got, model = verdicts(trust, tmp_path, recs, graph)
    assert got["crosscheck.a"] == "relies-unverified" and got["crosscheck.b"] == "relies-unverified"
    for name in ("crosscheck.a", "crosscheck.b"):
        assert name not in {b["id"] for b in model["nodes"][name]["because"]}
    assert [b["id"] for b in model["nodes"]["crosscheck.a"]["because"]] == ["crosscheck.b"]


def test_a_reference_cycle_is_listed_beside_an_unchecked_dependency(trust, tmp_path):
    """A part with an unchecked dependency and a check through a reference cycle lists both causes."""
    graph = {"nodes": {
        "crosscheck.a": {"layer": "references"},
        "crosscheck.b": {"layer": "references"},
        "virgil.u": {"layer": "models"},
        "virgil.x": {"layer": "models", "depends": ["virgil.u"]},
    }}
    recs = [
        record("t::a", "crosscheck.a", kind="reference", via=["crosscheck.b"]),
        record("t::b", "crosscheck.b", kind="reference", via=["crosscheck.a"]),
        record("t::x", "virgil.x", via=["crosscheck.a"]),
    ]
    got, model = verdicts(trust, tmp_path, recs, graph)
    assert got["virgil.x"] == "relies-unverified"
    cause = {b["id"] for b in model["nodes"]["virgil.x"]["because"]}
    assert "virgil.u" in cause and cause & {"crosscheck.a", "crosscheck.b"}


def test_verdicts_do_not_depend_on_the_order_the_nodes_are_listed(trust, tmp_path):
    """The broken targets are a least fixed point: the same graph in any listing order gives the same verdicts."""
    nodes = {
        "crosscheck.a": {"layer": "references"},
        "crosscheck.b": {"layer": "references", "depends": ["crosscheck.a"]},
        "crosscheck.bad": {"layer": "references"},
        "virgil.x": {"layer": "models", "depends": ["crosscheck.b"]},
        "virgil.y": {"layer": "models"},
        "virgil.z": {"layer": "models", "depends": ["virgil.y"]},
    }
    recs = [
        record("t::a", "crosscheck.a", kind="reference", via=["crosscheck.b"]),
        record("t::b", "crosscheck.b", kind="reference"),
        record("t::bad", "crosscheck.bad", kind="reference", outcome="failed"),
        record("t::x", "virgil.x"),
        record("t::y", "virgil.y", via=["crosscheck.bad"]),
        record("t::z", "virgil.z"),
    ]
    results = []
    for order in (list(nodes), list(reversed(list(nodes))), sorted(nodes, key=lambda k: k[::-1])):
        got, model = verdicts(trust, tmp_path, recs, {"nodes": {k: nodes[k] for k in order}})
        results.append((got, {k: [b["id"] for b in v["because"]] for k, v in model["nodes"].items()}))
    assert results[0] == results[1] == results[2]
