"""What the tool promises, one test per promise."""

import json
import pathlib

import pytest
import yaml
from conftest import EXAMPLES, STEP, commit

from mandate import audit, scaffold, work
from mandate.cli import main
from mandate.model import Campaign, MandateError
from mandate.orders import expand, find, superseded

ROWS = [
    {"id": "r1", "role": "ref", "band": "K", "t": "10", "want": "pass", "answer": "1"},
    {"id": "r2", "role": "ref", "band": "K", "t": "20", "want": "pass", "answer": "2"},
    {"id": "r3", "role": "ref", "band": "K", "t": "90", "want": "fail", "answer": "3"},
    {"id": "s1", "role": "sci", "band": "K", "t": "22", "want": "pass", "answer": "4"},
]
PLAN = {
    "sealed_columns": ["answer"],
    "stages": {
        "fit": {"over": {"role": "ref"}, "run": STEP, "requires": ["ok"],
                "decisions": {"order": {"default": 2, "options": [1, 2], "authority": "agent"},
                              "prior": {"default": "flat", "options": ["flat", "log"]}}},
        "basis": {"over": {"role": "ref"}, "by": ["band"], "run": STEP, "needs": {"fit": {"match": ["band"], "min": 2}}},
        "sci": {"over": {"role": "sci"}, "run": STEP,
                "needs": {"basis": {"match": ["band"]}, "fit": {"match": ["band"], "nearest": {"by": "t", "k": 1}}}},
        "score": {"over": {"role": "sci"}, "sealed": True, "run": STEP, "needs": {"sci": {"match": ["id"]}}},
    },
}


def run_all(root, *stages):
    for stage in stages:
        main(["-C", str(root), "run", "--stage", stage])
    return Campaign(root)


def states(c, stage):
    return {o.key_str: o.state for o in expand(c)[stage]}


# ---- fan out, fan in ----------------------------------------------------------------------


def test_downstream_waits_then_an_ensemble_lists_its_members_and_what_it_dropped(make):
    root = make(PLAN, ROWS)
    c = Campaign(root)
    assert states(c, "fit") == {"r1": "ready", "r2": "ready", "r3": "ready"}
    assert states(c, "basis") == {"K": "waiting"}
    c = run_all(root, "fit")
    assert states(c, "fit") == {"r1": "pass", "r2": "pass", "r3": "fail"}
    basis = find(c, "basis/K")
    assert basis.state == "ready"
    assert [m["key"] for m in basis.inputs["fit"]] == ["r1", "r2"]
    assert basis.dropped["fit"] == [{"key": "r3", "why": "fail"}]


def test_too_few_members_blocks_instead_of_running_on_less(make):
    rows = [dict(r, want="fail") if r["id"] == "r2" else r for r in ROWS]
    c = run_all(make(PLAN, rows), "fit")
    basis = find(c, "basis/K")
    assert basis.state == "blocked" and "1 of at least 2" in basis.reason
    with pytest.raises(MandateError):
        work.run(c, basis)


def test_nearest_in_time_picks_the_closest_passing_upstream(make):
    c = run_all(make(PLAN, ROWS), "fit", "basis")
    sci = find(c, "sci/s1")
    assert [m["key"] for m in sci.inputs["fit"]] == ["r2"]  # t=20 is nearest to 22; r3 failed


def test_changing_an_ensembles_membership_makes_a_new_order_and_stales_downstream(make):
    root = make(PLAN, ROWS)
    c = run_all(root, "fit", "basis", "sci")
    old_basis, old_sci = find(c, "basis/K").id, find(c, "sci/s1").id
    c.append("exclude", why="a binary", by="ben", where={"id": "r1"}, stages=["basis"])
    c = Campaign(root)
    basis = find(c, "basis/K")
    assert basis.id != old_basis and basis.state == "blocked"  # only r2 is left
    assert find(c, "sci/s1").state == "blocked"
    assert {f"basis/{old_basis}", f"sci/{old_sci}"} <= set(superseded(c, expand(c)))
    assert states(c, "fit")["r1"] == "pass"  # excluded from the basis only: its own fit stands


def test_a_corrected_registry_row_is_new_work_unless_the_plan_ignores_the_column(make):
    root = make({**PLAN, "ignore_columns": ["want"]}, ROWS)
    c = Campaign(root)
    before = {k: find(c, f"fit/{k}").id for k in ("r1", "r2")}
    c.append("annotate", why="header was wrong", by="ben", where={"id": "r1"}, set={"t": "11"})
    c.append("annotate", why="bookkeeping", by="ben", where={"id": "r2"}, set={"want": "warn"})
    c = Campaign(root)
    assert find(c, "fit/r1").id != before["r1"]  # the step is given a different row
    assert find(c, "fit/r2").id == before["r2"]  # an ignored column is not part of the work
    c.append("annotate", why="the paper's value moved", by="ben", where={"id": "r1"}, set={"answer": "9"})
    assert find(Campaign(root), "fit/r1").id == find(c, "fit/r1").id  # a sealed answer never reaches an open stage


# ---- the mandate ---------------------------------------------------------------------------


def test_a_decision_off_the_menu_is_refused(make):
    c = Campaign(make(PLAN, ROWS))
    with pytest.raises(MandateError, match="not an option"):
        c.append("decide", why="try", stage="fit", key="r1", set={"order": 3})
    with pytest.raises(MandateError, match="no decision"):
        c.append("decide", why="try", stage="fit", key="r1", set={"tolerance": 9})
    assert main(["-C", str(c.root), "decide", "fit", "r1", "order=3", "--why", "x"]) == 2
    assert c.docket == []


def test_an_agent_decides_where_the_plan_lets_it_and_proposes_elsewhere(make):
    root = make(PLAN, ROWS)
    c = Campaign(root)
    before = find(c, "fit/r1").id
    mine = c.append("decide", why="short exposure", stage="fit", key="r1", set={"order": 1})
    theirs = c.append("decide", why="looks log-like", stage="fit", key="r2", set={"prior": "log"})
    assert c.state(mine) == "active" and c.state(theirs) == "proposed"
    c = Campaign(root)
    r1, r2 = find(c, "fit/r1"), find(c, "fit/r2")
    assert r1.id != before and r1.decisions["order"] == {"value": 1, "source": f"docket:{mine['id']}"}
    assert r2.decisions["prior"] == {"value": "flat", "source": "default"}  # not in force yet
    with pytest.raises(MandateError, match="only the plan's humans"):
        c.append("resolve", why="fine", of=theirs["id"], verdict="approve")
    c.append("resolve", why="fine", by="ben", of=theirs["id"], verdict="approve")
    assert find(Campaign(root), "fit/r2").decisions["prior"]["value"] == "log"


def test_a_request_parks_the_order_and_everything_else_carries_on(make):
    root = make(PLAN, ROWS)
    c = Campaign(root)
    entry = c.append("request", why="the frame is saturated; no option covers it", stage="fit", key="r2")
    c = run_all(root, "fit")
    assert states(c, "fit") == {"r1": "pass", "r2": "parked", "r3": "fail"}
    assert find(c, "basis/K").state == "waiting"
    assert [e["id"] for e in c.pending()] == [entry["id"]]
    c.append("resolve", why="use it anyway", by="ben", of=entry["id"], verdict="approve")
    assert states(Campaign(root), "fit")["r2"] == "ready"


def test_nothing_runs_on_an_uncommitted_plan_or_registry(make):
    root = make(PLAN, ROWS)
    plan = yaml.safe_load((root / "mandate.yaml").read_text())
    plan["stages"]["fit"]["decisions"]["order"]["options"] = [1, 2, 3]
    (root / "mandate.yaml").write_text(yaml.safe_dump(plan))
    with pytest.raises(MandateError, match="uncommitted"):
        work.run(Campaign(root), find(Campaign(root), "fit/r1"))
    assert not (root / "ledger").exists()


def test_a_pinned_version_that_does_not_match_refuses(make):
    plan = {**PLAN, "versions": {"code": {"cmd": "echo 1.2.3", "pin": "1.2.4"}}}
    c = Campaign(make(plan, ROWS))
    with pytest.raises(MandateError, match="pins"):
        work.run(c, find(c, "fit/r1"))


# ---- results come from the command, and stay as recorded ----------------------------------


def test_a_step_cannot_pass_by_leaving_a_required_check_out(make):
    plan = {"stages": {"fit": {"over": {"role": "ref"}, "run": STEP, "requires": ["ok", "injection_recovery"]}}}
    c = run_all(make(plan, ROWS[:1]), "fit")
    o = find(c, "fit/r1")
    assert o.state == "fail" and "injection_recovery" in o.attempts[-1]["problem"]


def test_a_crash_or_a_missing_result_is_an_error_not_a_pass(make):
    plan = {"stages": {"a": {"over": {"id": "r1"}, "run": "exit 3"}, "b": {"over": {"id": "r2"}, "run": "true"}}}
    c = run_all(make(plan, ROWS[:2]), "a", "b")
    assert find(c, "a/r1").attempts[-1]["problem"].startswith("exit code 3")
    assert find(c, "b/r2").state == "error" and "no result.json" in find(c, "b/r2").attempts[-1]["problem"]


def test_every_attempt_is_kept_and_a_fresh_rerun_reports_whether_it_reproduced(make):
    root = make(PLAN, ROWS)
    run_all(root, "fit")
    assert main(["-C", str(root), "run", "--fresh", "fit/r1"]) == 0
    o = find(Campaign(root), "fit/r1")
    assert [a["n"] for a in o.attempts] == [1, 2] and o.attempts[-1]["reproduced"] is True
    assert sorted(p.name for p in (root / "ledger" / "fit" / o.id).iterdir()) == [
        "attempt-001.json", "attempt-002.json", "order.json"]


def test_verify_catches_a_changed_run_folder_and_a_missing_attempt(make):
    root = make(PLAN, ROWS)
    c = run_all(root, "fit")
    commit(root)
    assert audit.verify(c)["violations"] == []
    o = find(c, "fit/r1")
    (work.out_dir(c, o) / "result.json").write_text('{"status": "pass", "checks": {"ok": "pass"}, "better": 1}')
    assert any("no longer matches" in v for v in audit.verify(Campaign(root))["violations"])
    main(["-C", str(root), "run", "fit/r3"])  # a second attempt of the failing order
    (root / "ledger" / "fit" / find(c, "fit/r3").id / "attempt-001.json").unlink()
    assert any("not numbered" in v for v in audit.verify(Campaign(root))["violations"])


def test_verify_base_separates_work_from_changes_of_mandate(make):
    root = make(PLAN, ROWS)
    c = run_all(root, "fit")
    c.append("decide", why="short", stage="fit", key="r1", set={"order": 1})
    commit(root, "work")
    base = "HEAD~1"
    report = audit.verify(Campaign(root), base)
    assert report["violations"] == [] and any("in force since" in r for r in report["review"])
    assert not any("MANDATE CHANGED" in r for r in report["review"])
    # Now the three things a worker must not do: edit a record, drop a docket line, change the plan.
    record = next((root / "ledger" / "fit").glob("r3-*/attempt-001.json"))
    record.write_text(record.read_text().replace('"fail"', '"pass"'))
    commit(root, "tamper")
    assert any("ledger record changed" in v for v in audit.verify(Campaign(root), "HEAD~1")["violations"])
    (root / "docket.jsonl").write_text("")
    plan = yaml.safe_load((root / "mandate.yaml").read_text())
    del plan["stages"]["fit"]["requires"]  # quietly dropping a required check
    (root / "mandate.yaml").write_text(yaml.safe_dump(plan))
    commit(root, "more")
    report = audit.verify(Campaign(root), "HEAD~1")
    assert any("append-only" in v for v in report["violations"])
    assert any("MANDATE CHANGED" in r and "mandate.yaml" in r for r in report["review"])
    assert main(["-C", str(root), "verify", "--base", "HEAD~1"]) == 1


# ---- sealed information --------------------------------------------------------------------


def test_sealed_columns_and_notes_reach_sealed_stages_only(make):
    root = make(PLAN, ROWS)
    (root / "notes").mkdir()
    (root / "notes" / "answer.md").write_text("---\napplies_to: {id: s1}\nvisibility: sealed\n---\nseparation 412 mas\n")
    (root / "notes" / "night.md").write_text("---\napplies_to: {id: s1}\n---\nthin cirrus all night\n")
    commit(root)
    c = run_all(root, "fit", "basis", "sci", "score")
    sci, score = find(c, "sci/s1"), find(c, "score/s1")
    given = json.loads((work.out_dir(c, sci) / "order.json").read_text())
    assert "answer" not in given["units"][0]
    assert json.loads((work.out_dir(c, score) / "order.json").read_text())["units"][0]["answer"] == "4"
    assert "412" not in work.brief(c, sci) and "cirrus" in work.brief(c, sci)
    assert "412" in work.brief(c, score)


def test_a_sealed_stage_refuses_to_run_where_the_answers_are_not_kept(make, tmp_path):
    root = make({**PLAN, "sealed_root": str(tmp_path / "vault")}, ROWS)
    c = run_all(root, "fit", "basis", "sci")
    with pytest.raises(MandateError, match="not on this machine"):
        work.run(c, find(c, "score/s1"))
    (tmp_path / "vault" / "notes").mkdir(parents=True)
    (tmp_path / "vault" / "notes" / "s1.md").write_text("the published separation is 412 mas\n")
    assert work.run(c, find(c, "score/s1"))["status"] == "pass"
    assert "412" in work.brief(c, find(c, "score/s1")) and "412" not in work.brief(c, find(c, "sci/s1"))


def test_the_plan_may_not_route_sealed_things_into_open_stages(make):
    plan = json.loads(json.dumps(PLAN))
    plan["stages"]["sci"]["needs"]["score"] = {"match": ["id"]}
    plan["stages"]["fit"]["over"] = {"answer": "1"}
    problems = Campaign(make(plan, ROWS)).problems()
    assert any("reads sealed stage" in p for p in problems) and any("sealed column" in p for p in problems)


# ---- agent stages, briefs, the page --------------------------------------------------------


def test_the_demo_campaign_end_to_end(demo, monkeypatch):
    def cli(*args):
        return main(["-C", str(demo), *args])

    assert cli("check") == 0
    assert cli("run", "--stage", "wavefront") == 1  # the two TOY-REF-E fits fail to converge
    assert cli("decide", "wavefront", "toy_ref_e_f444w", "n_modes=4", "--why", "short exposure") == 0
    assert cli("run", "--stage", "wavefront") == 0
    assert cli("exclude", "target_name=TOY-REF-D", "--stage", "basis", "--why", "known binary", "--by", "ben") == 0
    assert cli("run", "--stage", "basis") == 0
    c = Campaign(demo)
    basis = find(c, "basis/F444W")
    assert [m["key"] for m in basis.inputs["wavefront"]] == [
        "toy_ref_a_f444w", "toy_ref_b_f444w", "toy_ref_c_f444w", "toy_ref_e_f444w"]
    assert {"key": "toy_ref_d_f444w", "why": "excluded"} in basis.dropped["wavefront"]
    assert find(c, "basis/F200W").state == "blocked"

    assert expand(c)["planet_search"] == []  # no science case yet: the agents have not classified
    for target, case in [("TOY-PLANET-1", "planet"), ("TOY-PLANET-2", "planet"), ("TOY-DISK-1", "disk")]:
        text = work.brief(c, find(c, f"classify/{target}"))
        assert "It is your whole mandate" in text and "412" not in text
        work.issue(c, find(c, f"classify/{target}"))
        out = work.out_dir(c, find(c, f"classify/{target}"))
        (out / "result.json").write_text('{"science_case": "galaxy"}')
        assert cli("submit", f"classify/{target}") == 2  # not one of the expected values: nothing is recorded
        (out / "result.json").write_text(json.dumps({"science_case": case, "why": "the abstract"}))
        assert cli("submit", f"classify/{target}") == 0
    assert cli("request", "classify", "TOY-UNKNOWN", "--why", "the abstract does not say") == 0
    c = Campaign(demo)
    assert expand(c)["planet_search"] == []  # proposed annotations are not in force
    for e in [e for e in c.docket if e["kind"] == "annotate"]:
        assert cli("resolve", e["id"], "approve", "--by", "hayden", "--why", "checked") == 0
    assert states(Campaign(demo), "classify") == {  # its own annotation does not make a stage stale
        "TOY-DISK-1": "pass", "TOY-PLANET-1": "pass", "TOY-PLANET-2": "pass", "TOY-UNKNOWN": "parked"}
    assert cli("run", "--stage", "planet_search") == 0
    assert cli("run", "--stage", "disk_deconvolution") == 0
    assert cli("run", "--stage", "score") == 0
    assert cli("run", "--stage", "catalogue") == 0
    c = Campaign(demo)
    planet = find(c, "planet_search/toy_sci_1_f444w")
    assert [m["key"] for m in planet.inputs["wavefront"]] == ["toy_ref_b_f444w", "toy_ref_a_f444w"]
    assert find(c, "planet_search/toy_sci_1_f200w").state == "blocked"
    assert find(c, "score/toy_sci_1_f444w").state == "pass"

    commit(demo)
    assert cli("verify") == 0
    page = audit.status_page(Campaign(demo))
    assert "## Waiting for a person (1)" in page and "TOY-UNKNOWN" in page
    assert "## Departures from the plan's defaults (2)" in page
    assert "Units no stage selects: 1 (toy_planned_f444w)" in page


def test_a_brief_carries_its_history_and_only_the_moves_the_order_allows(make):
    root = make(PLAN, ROWS)
    c = run_all(root, "fit", "basis")
    text = work.brief(c, find(c, "fit/r3"))
    assert "attempt 1 by agent-test: fail" in text and "failing: ok" in text
    assert "`order` = `2` (default). Options: 1, 2; you may change it." in text
    assert "`prior` = `flat` (default). Options: flat, log; a person approves a change." in text
    assert "mandate decide fit r3 NAME=OPTION" in text
    assert "mandate decide basis" not in work.brief(c, find(c, "basis/K"))  # no decisions on that stage


# ---- registry, cluster hand-off, the second example -----------------------------------------


def test_ingest_reads_a_mast_export_and_reports_what_changed(make, tmp_path, capsys):
    root = make(PLAN, ROWS)
    table = tmp_path / "sheet.csv"
    table.write_text("#The following lines are comments\n#id,role,band\n#@string,string,string\nr1,ref,K\nr9,ref,L\n")
    assert main(["-C", str(root), "ingest", str(table)]) == 0
    assert "2 units: 1 added, 3 removed, 1 changed" in capsys.readouterr().out
    assert (root / "registry" / "units.csv").read_text().splitlines()[0] == "id,role,band"
    assert main(["-C", str(root), "ingest", table.as_uri()]) == 0  # a sheet's CSV export URL works the same
    table.write_text("id,role\nr1,ref\nr1,ref\n")
    assert main(["-C", str(root), "ingest", str(table)]) == 2


def test_collect_brings_cluster_records_into_the_ledger_and_slurm_only_writes_a_script(make, tmp_path, monkeypatch):
    root = make(PLAN, ROWS)
    monkeypatch.setenv("MANDATE_RUNS", str(tmp_path / "cluster"))
    c = run_all(root, "fit")
    order = find(c, "fit/r1")
    for path in (root / "ledger").rglob("*.json"):
        path.unlink()
    assert find(Campaign(root), "fit/r1").state == "ready"  # the laptop has no records yet
    assert work.collect(Campaign(root), tmp_path / "cluster") == 6
    c = Campaign(root)
    assert find(c, "fit/r1").state == "pass" and find(c, "fit/r1").id == order.id
    assert "out" not in json.loads((root / "ledger" / "fit" / order.id / "order.json").read_text())
    c.append("resolve", why="x", by="ben", of=c.append("request", why="x", stage="basis", key="K")["id"],
             verdict="approve")
    script = work.slurm_script(c, "basis", [find(Campaign(root), "basis/K")])
    text = script.read_text()
    assert "#SBATCH --array=1-1" in text and "mandate -C" in text
    assert pathlib.Path(str(script).replace(".sh", ".orders")).read_text().startswith("basis/K-")


def test_the_gravity_sketch_groups_epochs_by_night_and_by_target():
    c = Campaign(EXAMPLES / "gravity_sketch")
    assert c.problems() == []
    orders = expand(c)
    assert [o.key_str for o in orders["night_calibration"]] == ["2024-01-10", "2024-03-02", "2025-02-11"]
    assert {o.key_str: len(o.units) for o in orders["orbit"]} == {"TOY-BIN-1": 3, "TOY-BIN-2": 1}
    models = {o.key_str: o.decisions["model"] for o in orders["epoch_fit"]}
    assert models["g04"] == {"value": "disk", "source": "rule:0"} and models["g01"]["source"] == "default"


# ---- what a person meets: starting, reading the plan back, answering ---------------------------


def test_init_gives_a_campaign_that_checks_and_says_what_to_do_next(tmp_path, capsys):
    assert main(["init", str(tmp_path / "new"), "--name", "trial"]) == 0
    assert "mandate guide" in capsys.readouterr().out
    assert Campaign(tmp_path / "new").problems() == []
    assert main(["init", str(tmp_path / "new")]) == 2  # never overwrites a plan
    assert "Never invent a column" in scaffold.GUIDE and main(["guide"]) == 0


def test_columns_lists_what_a_plan_may_select_on_and_hides_sealed_values(make):
    text = scaffold.columns(Campaign(make(PLAN, ROWS)))
    assert "role    ref (3), sci (1)" in text and "answer  (sealed)" in text


def test_explain_reads_the_plan_back_in_plain_words(make):
    text = scaffold.explain(Campaign(make(PLAN, ROWS)))
    assert "Once per band, over rows where role is ref." in text
    assert "Reads every passing result of fit with the same band; needs at least 2." in text
    assert "Reads the 1 passing results nearest in t of fit with the same band." in text
    assert "Choice order: 2 by default (allowed: 1, 2); an agent may change it for one order, with a reason." in text
    assert "Choice prior: flat by default (allowed: flat, log); a change needs a person." in text
    assert "Passes only if these checks pass: ok." in text and "No required checks" in text
    assert "fit -> basis, sci" in text and "score  [code, sealed]" in text


def test_check_suggests_the_name_you_probably_meant(make):
    plan = json.loads(json.dumps(PLAN))
    plan["stages"]["fit"]["over"] = {"rol": "ref"}
    plan["stages"]["fit"]["require"] = ["ok"]
    problems = Campaign(make(plan, ROWS)).problems()
    assert "stage fit: column 'rol' is not in the registry (did you mean 'role'?)" in problems
    assert "stage fit: unknown key 'require' (did you mean 'requires'?)" in problems


def test_the_home_screen_says_what_is_waiting_and_what_to_run_next(make):
    root = make(PLAN, ROWS)
    c = run_all(root, "fit")
    c.append("request", why="saturated", stage="basis", key="K")
    text = scaffold.home(Campaign(root))
    assert "Waiting for a person (1)" in text and "mandate review" in text
    assert "1 failed: give each brief to a worker agent" in text
    (root / "mandate.yaml").write_text((root / "mandate.yaml").read_text() + "\n")
    assert "uncommitted changes" in scaffold.home(Campaign(root))


def test_review_walks_a_person_through_the_docket_and_says_what_each_answer_changes(make):
    root = make(PLAN, ROWS)
    c = run_all(root, "fit")
    c.append("exclude", why="a binary", where={"id": "r1"}, stages=["basis"])
    c.append("decide", why="looks log-like", stage="fit", key="r2", set={"prior": "log"})
    said = []
    scaffold.review(Campaign(root), listing=True, say=said.append)
    assert "      then  to redo: basis/K" in said and "      then  to redo: fit/r2, basis/K" in said
    assert "exclude   id=r1 from basis    (" in said[0] and said[1] == "      why   a binary"
    c.append("request", why="saturated; no option covers it", stage="fit", key="r3")
    with pytest.raises(MandateError, match="not one of the plan's humans"):
        scaffold.review(Campaign(root), ask=lambda _: "a")
    answers = iter(["a", "", "d", "the flat prior is registered", "c", ""])  # the last has no reason: skipped
    n = scaffold.review(Campaign(root), by="ben", ask=lambda _: next(answers), say=said.append)
    c = Campaign(root)
    assert n == 2 and [c.state(e) for e in c.docket[:3]] == ["active", "denied", "open"]
