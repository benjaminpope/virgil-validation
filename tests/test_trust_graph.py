"""The trust graph and ledger stay consistent with the tests."""

import ast
import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
GRAPH = yaml.safe_load(open(ROOT / "trust" / "graph.yml"))
LEDGER = yaml.safe_load(open(ROOT / "trust" / "ledger.yml"))


def _validates_objects():
    """Every string literal passed positionally to a validates marker,
    anywhere in the test files (including pytest.param marks)."""
    objects = set()
    for path in (ROOT / "tests").glob("test_*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "validates"
            ):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        objects.add(arg.value)
                    elif isinstance(arg, ast.Starred):
                        objects.add("<computed>")
    return objects


def _test_ids():
    ids = set()
    for path in (ROOT / "tests").glob("test_*.py"):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                ids.add(f"tests/{path.name}::{node.name}")
    return ids


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_every_validated_object_is_in_the_graph():
    known = set(GRAPH["nodes"]) | {f"pipeline:{p}" for p in GRAPH["pipelines"]} | {"<computed>"}
    missing = _validates_objects() - known
    assert not missing, sorted(missing)


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_graph_dependencies_and_pipeline_steps_are_nodes():
    nodes = set(GRAPH["nodes"])
    for name, node in GRAPH["nodes"].items():
        assert set(node.get("depends", [])) <= nodes, name
    for name, pipe in GRAPH["pipelines"].items():
        assert set(pipe["steps"]) <= nodes, name


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_ledger_entries_point_at_real_tests_and_nodes():
    tests, nodes = _test_ids(), set(GRAPH["nodes"])
    rulings = {"virgil", "definition", "crosscheck"}
    for e in LEDGER:
        assert e["ruling"] in rulings or e["ruling"].startswith("external:"), e["id"]
        assert set(e["objects"]) <= nodes, (e["id"], e["objects"])
        for t in e["evidence"]:
            assert t in tests, (e["id"], t)
    ids = [e["id"] for e in LEDGER]
    assert len(ids) == len(set(ids))


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_pipelines_describe_their_data():
    """The Trust page shows real-data and simulated pipelines apart: each
    says which it is, and real ones say what they reproduce."""
    for name, p in GRAPH.get("pipelines", {}).items():
        assert p.get("data") in ("real", "simulated"), name
        assert p.get("dataset") and p.get("reference"), name
        assert p.get("state", "planned") in ("planned", "running"), name

