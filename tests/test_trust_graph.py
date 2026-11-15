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


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_graph_and_ledger_have_no_duplicate_keys():
    """YAML keeps the last of two equal keys silently, so a node or entry
    added by two branches at once would vanish without this check."""
    import collections

    class Strict(yaml.SafeLoader):
        pass

    def mapping(loader, node, deep=False):
        keys = [loader.construct_object(k, deep=deep) for k, _ in node.value]
        dupes = [k for k, n in collections.Counter(keys).items() if n > 1]
        assert not dupes, dupes
        return yaml.SafeLoader.construct_mapping(loader, node, deep)

    Strict.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    yaml.load(open(ROOT / "trust" / "graph.yml"), Loader=Strict)
    ledger = yaml.load(open(ROOT / "trust" / "ledger.yml"), Loader=Strict)
    ids = [entry["id"] for entry in ledger]
    assert len(ids) == len(set(ids)), [i for i, n in collections.Counter(ids).items() if n > 1]



def _marker_properties():
    """(objects, properties) for every validates marker that names a property."""
    found = []
    for path in (ROOT / "tests").glob("test_*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "validates":
                kw = {k.arg: k.value for k in node.keywords}
                if "property" in kw:
                    prop = ast.literal_eval(kw["property"])
                    objs = {a.value for a in node.args if isinstance(a, ast.Constant)}
                    found.append((path.name, objs, [prop] if isinstance(prop, str) else list(prop)))
    return found


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_properties_named_by_tests_are_declared_in_the_graph():
    """A misspelt property would silently count for nothing."""
    nodes, found = GRAPH["nodes"], _marker_properties()
    assert found
    for where, objs, props in found:
        if not objs:  # computed names (f-strings), as in _validates_objects
            continue
        declared = {p for o in objs if o in nodes
                    for p in [*(nodes[o].get("properties") or {}), *(nodes[o].get("convention") or [])]}
        assert set(props) <= declared, (where, sorted(objs), props)


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_every_graph_source_path_exists_in_virgil():
    """``source:`` paths drive the freshness dot (scripts/trust.py changed_files); a wrong one reads as
    "unchanged" for ever. Graph paths are repo-relative (``src/virgil/x.py``); the installed package
    holds the same files at ``virgil/x.py``."""
    import virgil

    package = pathlib.Path(virgil.__file__).resolve().parent
    prefix = "src/virgil/"
    missing = []
    for name, node in GRAPH["nodes"].items():
        for path in node.get("source", []):
            assert path.startswith(prefix), (name, path)
            if not (package / path[len(prefix):]).exists():
                missing.append((name, path))
    assert not missing, missing


def _find_depends_cycles(nodes: dict) -> list[list[str]]:
    """Cycles among ``depends`` edges in ``nodes`` (empty when ``depends`` is a DAG)."""
    state = {}  # 0: unvisited, 1: active, 2: done
    stack = []
    cycles = []

    def dfs(name: str):
        state[name] = 1
        stack.append(name)
        for dep in (nodes.get(name) or {}).get("depends", []):
            if dep not in nodes:
                continue
            st = state.get(dep, 0)
            if st == 1:
                idx = stack.index(dep)
                cycles.append([*stack[idx:], dep])
            elif st == 0:
                dfs(dep)
        stack.pop()
        state[name] = 2

    for name in sorted(nodes):
        if state.get(name, 0) == 0:
            dfs(name)
    return cycles


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_reference_module_imports_are_declared_in_depends():
    """Every ``src/{crosscheck,external_bridge,virgil_bridge}/*.py`` module (outside
    explicit non-reference ``_PLUMBING``) must be registered under ``layer: references``,
    every reference import or subprocess worker call (including via ``__init__.py``) must
    be declared in ``depends``, and ``depends`` must be acyclic."""
    from evidence import plugin

    plugin._module_via.cache_clear()
    plugin._init_exports.cache_clear()
    plugin._has_executable_defs.cache_clear()
    ref_nodes = {
        name for name, node in GRAPH["nodes"].items()
        if (node or {}).get("layer", "references") == "references"
    }
    unregistered = []
    undeclared = []
    for pkg in plugin._REFERENCE_PACKAGES:
        init_py = ROOT / "src" / pkg / "__init__.py"
        if init_py.is_file():
            if plugin._has_executable_defs(init_py):
                if pkg not in ref_nodes:
                    unregistered.append(pkg)
                else:
                    imported_refs = set(plugin._module_via(str(init_py))) - {pkg}
                    declared = set(GRAPH["nodes"][pkg].get("depends", []))
                    missing = sorted(imported_refs - declared)
                    if missing:
                        undeclared.append((pkg, missing))
            else:
                init_unreg = sorted(set(plugin._module_via(str(init_py))) - ref_nodes)
                if init_unreg:
                    unregistered.append((f"{pkg}.__init__", init_unreg))
        for py in sorted((ROOT / "src" / pkg).rglob("*.py")):
            rel = py.relative_to(ROOT / "src")
            if rel.parts == (pkg, "__init__.py"):
                continue
            mod = ".".join(rel.parts[:-1]) if py.name == "__init__.py" else ".".join((*rel.parts[:-1], py.stem))
            if mod in plugin._PLUMBING:
                continue
            if mod not in ref_nodes:
                unregistered.append(mod)
                continue
            imported_refs = set(plugin._module_via(str(py))) - {mod}
            declared = set(GRAPH["nodes"][mod].get("depends", []))
            missing = sorted(imported_refs - declared)
            if missing:
                undeclared.append((mod, missing))
    assert not unregistered, unregistered
    assert not undeclared, undeclared
    cycles = _find_depends_cycles(GRAPH["nodes"])
    assert not cycles, cycles
