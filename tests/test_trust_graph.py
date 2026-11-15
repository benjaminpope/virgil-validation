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
    ref_nodes = {
        name for name, node in GRAPH["nodes"].items()
        if (node or {}).get("layer", "references") == "references"
    }
    unregistered = []
    undeclared = []
    for pkg in plugin._REFERENCE_PACKAGES:
        init_py = ROOT / "src" / pkg / "__init__.py"
        if init_py.is_file():
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


def _executable_ast_nodes(stmt: ast.AST) -> list[ast.AST]:
    """Executable Call/Attribute/Name AST nodes, skipping type annotations and isinstance/issubclass types."""
    found: list[ast.AST] = []

    class Visitor(ast.NodeVisitor):
        def visit_AnnAssign(self, node: ast.AnnAssign):
            if node.value is not None:
                self.visit(node.value)

        def visit_FunctionDef(self, node: ast.FunctionDef):
            for d in node.decorator_list:
                self.visit(d)
            for default in node.args.defaults + [x for x in node.args.kw_defaults if x is not None]:
                self.visit(default)
            for s in node.body:
                self.visit(s)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ("isinstance", "issubclass"):
                if node.args:
                    self.visit(node.args[0])
                return
            found.append(node)
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute):
            found.append(node)
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name):
            if isinstance(node.ctx, ast.Load):
                found.append(node)

    Visitor().visit(stmt)
    return found


def _build_virgil_ast_reachability(virgil_dir: pathlib.Path):
    """Build static call/import adjacency across installed virgil, including function-local lazy imports and class STAGES."""
    import collections

    def file_to_module(path: pathlib.Path) -> str:
        parts = list(path.relative_to(virgil_dir.parent).parts)
        parts = parts[:-1] if parts[-1] == "__init__.py" else [*parts[:-1], parts[-1][:-3]]
        return ".".join(parts)

    def resolve_relative(current_mod: str, level: int, module: str | None, is_pkg: bool) -> str:
        if level == 0:
            return module or ""
        parts = current_mod.split(".")
        base_parts = (parts if level == 1 else parts[: -(level - 1)]) if is_pkg else parts[:-level]
        return ".".join(base_parts + module.split(".")) if module else ".".join(base_parts)

    parsed: dict[str, tuple[ast.Module, bool]] = {}
    for p in sorted(virgil_dir.rglob("*.py")):
        parsed[file_to_module(p)] = (ast.parse(p.read_text(encoding="utf-8"), filename=str(p)), p.name == "__init__.py")

    mod_imports: dict[str, dict[str, str]] = collections.defaultdict(dict)
    mod_defs: dict[str, set[str]] = collections.defaultdict(set)
    reexports: dict[str, str] = {}

    for mod, (tree, is_pkg) in parsed.items():
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                mod_defs[mod].add(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "virgil" or alias.name.startswith("virgil."):
                        local = alias.asname or alias.name.split(".")[0]
                        mod_imports[mod][local] = alias.name if alias.asname else alias.name.split(".")[0]
            elif isinstance(node, ast.ImportFrom):
                src_mod = resolve_relative(mod, node.level, node.module, is_pkg)
                if src_mod == "virgil" or src_mod.startswith("virgil."):
                    for alias in node.names:
                        if alias.name != "*":
                            local = alias.asname or alias.name
                            target = f"{src_mod}.{alias.name}"
                            mod_imports[mod][local] = target
                            reexports[f"{mod}.{local}"] = target
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        mod_defs[mod].add(t.id)
                        if isinstance(node.value, ast.Name) and node.value.id in mod_imports[mod]:
                            reexports[f"{mod}.{t.id}"] = mod_imports[mod][node.value.id]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                mod_defs[mod].add(node.target.id)

    def canonical(node_id: str) -> str:
        seen, cur = set(), node_id
        while cur in reexports and cur not in seen:
            seen.add(cur)
            cur = reexports[cur]
        return cur

    adj: dict[str, set[str]] = collections.defaultdict(set)

    def collect_local_imports(stmts: list[ast.AST], mod: str, is_pkg: bool) -> dict[str, str]:
        local_imp = dict(mod_imports[mod])
        for stmt in stmts:
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Import):
                    for alias in sub.names:
                        if alias.name == "virgil" or alias.name.startswith("virgil."):
                            local = alias.asname or alias.name.split(".")[0]
                            local_imp[local] = alias.name if alias.asname else alias.name.split(".")[0]
                elif isinstance(sub, ast.ImportFrom):
                    src_mod = resolve_relative(mod, sub.level, sub.module, is_pkg)
                    if src_mod == "virgil" or src_mod.startswith("virgil."):
                        for alias in sub.names:
                            if alias.name != "*":
                                local_imp[alias.asname or alias.name] = f"{src_mod}.{alias.name}"
        return local_imp

    def resolve_attr(node: ast.AST, mod: str, cur_cls: str | None, imports: dict[str, str]) -> str | None:
        parts, cur = [], node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            parts.reverse()
            head = parts[0]
            if head in ("self", "cls") and cur_cls and len(parts) >= 2:
                return f"{cur_cls}.{parts[1]}"
            if head in imports:
                base = imports[head]
                return base if len(parts) == 1 else ".".join([base, *parts[1:]])
            if head in mod_defs[mod]:
                base = f"{mod}.{head}"
                return base if len(parts) == 1 else ".".join([base, *parts[1:]])
        return None

    def extract_refs(stmts: list[ast.AST], owner: str, mod: str, is_pkg: bool, cur_cls: str | None):
        imports = collect_local_imports(stmts, mod, is_pkg)
        for stmt in stmts:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue
            for sub in _executable_ast_nodes(stmt):
                if isinstance(sub, ast.Attribute):
                    res = resolve_attr(sub, mod, cur_cls, imports)
                    if res and (res == "virgil" or res.startswith("virgil.")):
                        adj[owner].add(res)
                    elif sub.attr == "whitened_residuals":
                        adj[owner].add("virgil.likelihood.whitened_residuals")
                elif isinstance(sub, ast.Name):
                    if sub.id in imports and imports[sub.id].startswith("virgil."):
                        adj[owner].add(imports[sub.id])
                    elif sub.id in mod_defs[mod]:
                        adj[owner].add(f"{mod}.{sub.id}")

    for mod, (tree, is_pkg) in parsed.items():
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_id = f"{mod}.{node.name}"
                extract_refs(node.body, fn_id, mod, is_pkg, None)
                extract_refs(node.decorator_list, fn_id, mod, is_pkg, None)
            elif isinstance(node, ast.ClassDef):
                cls_id = f"{mod}.{node.name}"
                for base in node.bases:
                    res = resolve_attr(base, mod, None, mod_imports[mod])
                    if res and res.startswith("virgil."):
                        adj[cls_id].add(res)
                cls_stmts = [it for it in node.body if not isinstance(it, (ast.FunctionDef, ast.AsyncFunctionDef))]
                extract_refs(cls_stmts, cls_id, mod, is_pkg, cls_id)
                for it in node.body:
                    if isinstance(it, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        m_id = f"{cls_id}.{it.name}"
                        adj[cls_id].add(m_id)
                        extract_refs(it.body, m_id, mod, is_pkg, cls_id)
                        extract_refs(it.decorator_list, m_id, mod, is_pkg, cls_id)

    for alias_id, target_id in reexports.items():
        adj[alias_id].add(target_id)

    expanded: dict[str, set[str]] = collections.defaultdict(set)
    for src, dsts in adj.items():
        for dst in dsts:
            expanded[src].add(dst)
            c = canonical(dst)
            if c != dst:
                expanded[src].add(c)
    return expanded, canonical, set(parsed)


def _runtime_virgil_origin(node_name: str, parsed_mods: set[str]) -> str:
    """Resolve PEP 562 ``__getattr__`` re-exports (e.g. ``virgil.pipeline.BinaryPipeline``) to their defining symbol."""
    import importlib

    parts = node_name.split(".")
    for split in range(len(parts), 0, -1):
        mod_name = ".".join(parts[:split])
        if mod_name in parsed_mods:
            rest = parts[split:]
            if not rest:
                return node_name
            try:
                obj = importlib.import_module(mod_name)
                for attr in rest:
                    obj = getattr(obj, attr)
                obj_mod, obj_qn = getattr(obj, "__module__", None), getattr(obj, "__qualname__", None)
                if obj_mod and obj_qn and obj_mod.startswith("virgil"):
                    return f"{obj_mod}.{obj_qn}"
            except Exception:
                return node_name
            break
    return node_name


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_virgil_likelihood_and_fit_callers_depend_on_whitened_residuals():
    """Every ``virgil.*`` node whose symbol transitively calls ``whitened_residuals``, ``model_loglike``,
    ``numpyro_model``, or ``fit`` (including function-local lazy imports, PEP 562 ``__getattr__`` re-exports,
    and class ``STAGES``) must declare a transitive ``depends:`` path in ``trust/graph.yml`` to
    ``virgil.likelihood.whitened_residuals`` so open findings on ``whitened_residuals`` (such as F22)
    propagate honestly as ``relies``."""
    import collections
    import importlib.util

    spec = importlib.util.find_spec("virgil")
    assert spec is not None and spec.origin is not None
    virgil_dir = pathlib.Path(spec.origin).resolve().parent
    expanded, canonical, parsed_mods = _build_virgil_ast_reachability(virgil_dir)

    sinks = {
        "virgil.likelihood.whitened_residuals",
        "virgil.likelihood._whitened_and_errors",
        "virgil.likelihood._curvature_rows",
        "virgil.likelihood.model_loglike",
        "virgil.likelihood.numpyro_model",
        "virgil.fitting.fit",
    }

    def ast_reaches_sink(node_name: str) -> list[str] | None:
        starts = [canonical(node_name)]
        origin = _runtime_virgil_origin(node_name, parsed_mods)
        if origin not in starts:
            starts.append(origin)
        q = collections.deque((s, [s]) for s in starts)
        seen = set(starts)
        while q:
            cur, path = q.popleft()
            if cur in sinks:
                return path
            for nxt in sorted(expanded.get(cur, ())):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, [*path, nxt]))
        return None

    nodes = GRAPH["nodes"]

    def graph_reaches_whitened(start: str, graph_nodes: dict = nodes) -> bool:
        seen, q = {start}, collections.deque([start])
        while q:
            cur = q.popleft()
            if cur == "virgil.likelihood.whitened_residuals":
                return True
            for dep in (graph_nodes.get(cur) or {}).get("depends", []):
                if dep not in seen:
                    seen.add(dep)
                    q.append(dep)
        return False

    # Non-vacuity and mutation controls: true callers reach a sink, pure forward models do not,
    # and dropping a required depends edge fails graph_reaches_whitened.
    assert ast_reaches_sink("virgil.ensemble.draw_groups") is not None
    assert ast_reaches_sink("virgil.inference.laplace_cov") is not None
    assert ast_reaches_sink("virgil.models.UniformDisk") is None
    mutated_nodes = {
        **nodes,
        "virgil.ensemble.draw_groups": {
            **(nodes["virgil.ensemble.draw_groups"] or {}),
            "depends": [
                d for d in (nodes["virgil.ensemble.draw_groups"] or {}).get("depends", []) if d != "virgil.fitting.fit"
            ],
        },
    }
    assert not graph_reaches_whitened("virgil.ensemble.draw_groups", mutated_nodes)

    missing_depends = []
    for name in sorted(nodes):
        if not name.startswith("virgil."):
            continue
        ast_path = ast_reaches_sink(name)
        if ast_path is not None and not graph_reaches_whitened(name):
            missing_depends.append((name, " -> ".join(ast_path)))
    assert not missing_depends, missing_depends
