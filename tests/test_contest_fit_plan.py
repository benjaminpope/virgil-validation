"""The fit plan (contests/fit_plan.yml, scripts/contest_fit_plan.py): every
advice row has an entry with its advice verbatim, and every arm either runs
or names what it needs."""

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("contest_fit_plan", ROOT / "scripts" / "contest_fit_plan.py")
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_plan_covers_every_advice_row():
    rows = fp.load_advice_rows()
    assert rows
    assert fp.check(fp.load_plan(), rows) == []


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_check_reports_missing_and_stale():
    plan = fp.load_plan()
    rows = [dict(r) for r in fp.load_advice_rows()]
    rows.append({"year": "1999", "name": "new", "advice": "x"})
    rows[0]["advice"] += " (edited)"
    problems = fp.check(plan, rows)
    assert any("no plan entry" in p and "1999" in p for p in problems)
    assert any("differs from the CSV" in p for p in problems)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_commands_render_without_jax():
    plan = fp.load_plan()
    tasks = {e["task"] for e in plan["datasets"] if e["task"]}
    indices = {t: i for i, t in enumerate(sorted(tasks))}
    cmds = fp.commands(plan, indices)
    assert cmds and len({(e["task"], a["id"]) for e, a, _ in cmds}) == len(cmds)
    assert all(c.startswith("python scripts/") for *_, c in cmds)
