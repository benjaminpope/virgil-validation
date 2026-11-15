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


def _executable_ast_nodes(stmt: ast.AST) -> list[ast.AST]:
    """Executable Call/Attribute/Name AST nodes in the current scope, skipping type annotations,
    isinstance/issubclass types, and nested function/class/lambda bodies (recursed separately)."""
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

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef):
            for d in node.decorator_list:
                self.visit(d)
            for b in node.bases:
                self.visit(b)
            for kw in node.keywords:
                self.visit(kw.value)

        def visit_Lambda(self, node: ast.Lambda):
            for default in node.args.defaults + [x for x in node.args.kw_defaults if x is not None]:
                self.visit(default)

        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ("isinstance", "issubclass"):
                if node.args:
                    self.visit(node.args[0])
                return
            found.append(node)
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute):
            found.append(node)
            cur = node.value
            while isinstance(cur, ast.Attribute):
                cur = cur.value
            if not isinstance(cur, ast.Name):
                self.visit(cur)

        def visit_Name(self, node: ast.Name):
            if isinstance(node.ctx, ast.Load):
                found.append(node)

    Visitor().visit(stmt)
    return found


def _direct_nested_scopes(
    stmts: list[ast.AST],
) -> list[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda]:
    """Immediate nested function/class/lambda nodes inside ``stmts`` without crossing deeper scopes."""
    nested: list[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda] = []

    class ScopeFinder(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef):
            for d in node.decorator_list:
                self.visit(d)
            for default in node.args.defaults + [x for x in node.args.kw_defaults if x is not None]:
                self.visit(default)
            nested.append(node)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef):
            for d in node.decorator_list:
                self.visit(d)
            for b in node.bases:
                self.visit(b)
            for kw in node.keywords:
                self.visit(kw.value)
            nested.append(node)

        def visit_Lambda(self, node: ast.Lambda):
            for default in node.args.defaults + [x for x in node.args.kw_defaults if x is not None]:
                self.visit(default)
            nested.append(node)

    finder = ScopeFinder()
    for s in stmts:
        finder.visit(s)
    return nested


def _build_virgil_ast_reachability(virgil_dir: pathlib.Path, mod_prefix: str = "virgil"):
    """Build static call/import adjacency across a package tree (default: installed ``virgil``),
    including function-local lazy imports, nested closures, class ``STAGES``, PEP 562 lazy bases,
    ``super()`` / MRO-inherited template methods, and module-level nodes."""
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
    parsed_mods_set = set(parsed)

    mod_imports: dict[str, dict[str, str]] = collections.defaultdict(dict)
    mod_defs: dict[str, set[str]] = collections.defaultdict(set)
    reexports: dict[str, str] = {}

    for mod, (tree, is_pkg) in parsed.items():
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                mod_defs[mod].add(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == mod_prefix or alias.name.startswith(f"{mod_prefix}."):
                        local = alias.asname or alias.name.split(".")[0]
                        mod_imports[mod][local] = alias.name if alias.asname else alias.name.split(".")[0]
            elif isinstance(node, ast.ImportFrom):
                src_mod = resolve_relative(mod, node.level, node.module, is_pkg)
                if src_mod == mod_prefix or src_mod.startswith(f"{mod_prefix}."):
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

    origin_cache: dict[str, str] = {}

    def origin_of(name: str) -> str:
        if name not in origin_cache:
            origin_cache[name] = _runtime_virgil_origin(name, parsed_mods_set, mod_prefix)
        return origin_cache[name]

    adj: dict[str, set[str]] = collections.defaultdict(set)
    cls_bases: dict[str, list[str]] = collections.defaultdict(list)
    cls_methods: dict[str, set[str]] = collections.defaultdict(set)
    cls_attrs: dict[str, set[str]] = collections.defaultdict(set)
    cls_method_self_attrs: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    cls_method_super_calls: dict[tuple[str, str], set[tuple[str, str]]] = collections.defaultdict(set)

    def collect_own_imports(stmts: list[ast.AST], mod: str, is_pkg: bool) -> dict[str, str]:
        own_imp: dict[str, str] = {}

        class ImpFinder(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                return

            visit_AsyncFunctionDef = visit_FunctionDef
            visit_ClassDef = visit_FunctionDef
            visit_Lambda = visit_FunctionDef

            def visit_Import(self, sub: ast.Import):
                for alias in sub.names:
                    if alias.name == mod_prefix or alias.name.startswith(f"{mod_prefix}."):
                        local = alias.asname or alias.name.split(".")[0]
                        own_imp[local] = alias.name if alias.asname else alias.name.split(".")[0]

            def visit_ImportFrom(self, sub: ast.ImportFrom):
                src_mod = resolve_relative(mod, sub.level, sub.module, is_pkg)
                if src_mod == mod_prefix or src_mod.startswith(f"{mod_prefix}."):
                    for alias in sub.names:
                        if alias.name != "*":
                            own_imp[alias.asname or alias.name] = f"{src_mod}.{alias.name}"

        imp_finder = ImpFinder()
        for s in stmts:
            imp_finder.visit(s)
        return own_imp

    def resolve_attr(node: ast.AST, mod: str, cur_cls: str | None, imports: dict[str, str]) -> list[str]:
        parts, cur = [], node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if (
            isinstance(cur, ast.Call)
            and isinstance(cur.func, ast.Name)
            and cur.func.id == "super"
            and cur_cls
            and parts
        ):
            parts.reverse()
            attr_tail = ".".join(parts)
            return [f"{b}.{attr_tail}" for b in cls_bases.get(cur_cls, ())]
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            parts.reverse()
            head = parts[0]
            if head in ("self", "cls") and cur_cls and len(parts) >= 2:
                return [f"{cur_cls}.{parts[1]}"]
            if head in imports:
                base = imports[head]
                return [base if len(parts) == 1 else ".".join([base, *parts[1:]])]
            if head in mod_defs[mod]:
                base = f"{mod}.{head}"
                return [base if len(parts) == 1 else ".".join([base, *parts[1:]])]
        return []

    for mod, (tree, _) in parsed.items():
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                cls_id = f"{mod}.{node.name}"
                for base in node.bases:
                    for b in resolve_attr(base, mod, None, mod_imports[mod]):
                        cb = canonical(b)
                        ob = canonical(origin_of(cb))
                        for cand_b in (cb, ob):
                            if (
                                cand_b == mod_prefix or cand_b.startswith(f"{mod_prefix}.")
                            ) and cand_b not in cls_bases[cls_id]:
                                cls_bases[cls_id].append(cand_b)
                for it in node.body:
                    if isinstance(it, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        cls_methods[cls_id].add(it.name)
                    elif isinstance(it, ast.Assign):
                        for t in it.targets:
                            for n in ast.walk(t):
                                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                                    cls_attrs[cls_id].add(n.id)
                    elif isinstance(it, ast.AnnAssign) and isinstance(it.target, ast.Name):
                        cls_attrs[cls_id].add(it.target.id)

    def _walk_scope_bindings(node: ast.AST, bound: set[str]):
        """Collect lexical bindings within ``node`` without crossing nested scopes."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(node.name)
            for sub in (
                *node.decorator_list,
                *node.args.defaults,
                *(d for d in node.args.kw_defaults if d is not None),
                *(a.annotation for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs) if a.annotation),
                *([node.args.vararg.annotation] if node.args.vararg and node.args.vararg.annotation else []),
                *([node.args.kwarg.annotation] if node.args.kwarg and node.args.kwarg.annotation else []),
                *([node.returns] if node.returns else []),
            ):
                _walk_scope_bindings(sub, bound)
            return
        if isinstance(node, ast.ClassDef):
            bound.add(node.name)
            for sub in (*node.decorator_list, *node.bases, *node.keywords):
                _walk_scope_bindings(sub, bound)
            return
        if isinstance(node, ast.Lambda):
            for sub in (*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)):
                _walk_scope_bindings(sub, bound)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.NamedExpr) and isinstance(sub.target, ast.Name):
                    bound.add(sub.target.id)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name != "*":
                    bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
        for child in ast.iter_child_nodes(node):
            _walk_scope_bindings(child, bound)

    def fn_local_vars(fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
        """Parameter names and own-scope bindings inside ``fn_node``."""
        bound: set[str] = set()
        for a in (*fn_node.args.posonlyargs, *fn_node.args.args, *fn_node.args.kwonlyargs):
            bound.add(a.arg)
        if fn_node.args.vararg:
            bound.add(fn_node.args.vararg.arg)
        if fn_node.args.kwarg:
            bound.add(fn_node.args.kwarg.arg)
        for s in fn_node.body:
            _walk_scope_bindings(s, bound)
        return bound

    def extract_refs(
        stmts: list[ast.AST],
        owner: str,
        mod: str,
        is_pkg: bool,
        cur_cls: str | None,
        parent_imports: dict[str, str],
        parent_masked: frozenset[str] = frozenset(),
        own_bound: set[str] | frozenset[str] = frozenset(),
        self_attrs_out: set[str] | None = None,
        super_calls_out: set[tuple[str, str]] | None = None,
    ):
        own_imp = collect_own_imports(stmts, mod, is_pkg)
        imports = {k: v for k, v in parent_imports.items() if k not in own_bound}
        imports.update(own_imp)
        masked = frozenset((parent_masked | set(own_bound)) - set(own_imp))

        for stmt in stmts:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue
            for sub in _executable_ast_nodes(stmt):
                if isinstance(sub, ast.Attribute):
                    cur_base = sub
                    attr_chain = []
                    while isinstance(cur_base, ast.Attribute):
                        attr_chain.append(cur_base.attr)
                        cur_base = cur_base.value
                    if isinstance(cur_base, ast.Name) and cur_base.id in masked and cur_base.id not in ("self", "cls"):
                        continue
                    if (
                        self_attrs_out is not None
                        and isinstance(cur_base, ast.Name)
                        and cur_base.id in ("self", "cls")
                        and attr_chain
                    ):
                        self_attrs_out.add(attr_chain[-1])
                    if (
                        super_calls_out is not None
                        and cur_cls
                        and attr_chain
                        and isinstance(cur_base, ast.Call)
                        and isinstance(cur_base.func, ast.Name)
                        and cur_base.func.id == "super"
                    ):
                        first_attr = attr_chain[-1]
                        for b in cls_bases.get(cur_cls, ()):
                            super_calls_out.add((b, first_attr))
                    for res in resolve_attr(sub, mod, cur_cls, imports):
                        if res == mod_prefix or res.startswith(f"{mod_prefix}."):
                            adj[owner].add(res)
                elif isinstance(sub, ast.Name):
                    if sub.id in masked:
                        continue
                    if sub.id in imports and (
                        imports[sub.id] == mod_prefix or imports[sub.id].startswith(f"{mod_prefix}.")
                    ):
                        adj[owner].add(imports[sub.id])
                    elif sub.id in mod_defs[mod]:
                        adj[owner].add(f"{mod}.{sub.id}")

        for nscope in _direct_nested_scopes(stmts):
            if isinstance(nscope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nlvars = fn_local_vars(nscope)
                extract_refs(
                    nscope.body, owner, mod, is_pkg, cur_cls,
                    imports, masked, nlvars, self_attrs_out, super_calls_out,
                )
            elif isinstance(nscope, ast.Lambda):
                llvars = {a.arg for a in (*nscope.args.posonlyargs, *nscope.args.args, *nscope.args.kwonlyargs)}
                if nscope.args.vararg:
                    llvars.add(nscope.args.vararg.arg)
                if nscope.args.kwarg:
                    llvars.add(nscope.args.kwarg.arg)
                extract_refs(
                    [nscope.body], owner, mod, is_pkg, cur_cls,
                    imports, masked, llvars, self_attrs_out, super_calls_out,
                )
            elif isinstance(nscope, ast.ClassDef):
                cb_bound: set[str] = set()
                for s in nscope.body:
                    _walk_scope_bindings(s, cb_bound)
                extract_refs(
                    nscope.body, owner, mod, is_pkg, None,
                    imports, masked, cb_bound, None, None,
                )

    for mod, (tree, is_pkg) in parsed.items():
        top_stmts = []
        base_imp = mod_imports[mod]
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_id = f"{mod}.{node.name}"
                adj[mod].add(fn_id)
                lvars = fn_local_vars(node)
                extract_refs(node.body, fn_id, mod, is_pkg, None, base_imp, frozenset(), lvars)
                extract_refs(
                    [*node.decorator_list, *node.args.defaults, *[d for d in node.args.kw_defaults if d is not None]],
                    fn_id, mod, is_pkg, None, base_imp,
                )
            elif isinstance(node, ast.ClassDef):
                cls_id = f"{mod}.{node.name}"
                adj[mod].add(cls_id)
                for cb in cls_bases.get(cls_id, ()):
                    adj[cls_id].add(cb)
                other_cls_stmts = []
                for it in node.body:
                    if isinstance(it, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        m_id = f"{cls_id}.{it.name}"
                        adj[cls_id].add(m_id)
                        lvars = fn_local_vars(it)
                        extract_refs(
                            it.body, m_id, mod, is_pkg, cls_id, base_imp, frozenset(), lvars,
                            self_attrs_out=cls_method_self_attrs[(cls_id, it.name)],
                            super_calls_out=cls_method_super_calls[(cls_id, it.name)],
                        )
                        extract_refs(
                            [*it.decorator_list, *it.args.defaults, *[d for d in it.args.kw_defaults if d is not None]],
                            m_id, mod, is_pkg, cls_id, base_imp,
                        )
                    elif isinstance(it, ast.Assign):
                        attr_names = [
                            n.id for t in it.targets for n in ast.walk(t)
                            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                        ]
                        if attr_names:
                            for aname in attr_names:
                                attr_id = f"{cls_id}.{aname}"
                                adj[cls_id].add(attr_id)
                                extract_refs([it], attr_id, mod, is_pkg, cls_id, base_imp)
                        else:
                            other_cls_stmts.append(it)
                    elif isinstance(it, ast.AnnAssign) and isinstance(it.target, ast.Name):
                        attr_id = f"{cls_id}.{it.target.id}"
                        adj[cls_id].add(attr_id)
                        extract_refs([it], attr_id, mod, is_pkg, cls_id, base_imp)
                    else:
                        other_cls_stmts.append(it)
                if other_cls_stmts:
                    extract_refs(other_cls_stmts, cls_id, mod, is_pkg, cls_id, base_imp)
            else:
                top_stmts.append(node)
        extract_refs(top_stmts, mod, mod, is_pkg, None, base_imp)

    mro_ancestors: dict[str, list[str]] = {}
    all_classes = set(cls_bases) | set(cls_methods) | set(cls_attrs)
    for cls_id in sorted(all_classes):
        ancs: list[str] = []
        seen_b, q_b = {cls_id}, collections.deque(cls_bases.get(cls_id, ()))
        while q_b:
            raw_b = q_b.popleft()
            for b in (canonical(raw_b), canonical(origin_of(raw_b))):
                if b in seen_b or not (b == mod_prefix or b.startswith(f"{mod_prefix}.")):
                    continue
                seen_b.add(b)
                ancs.append(b)
                q_b.extend(cls_bases.get(b, ()))
        mro_ancestors[cls_id] = ancs

    def effective_self_attrs(c: str, m: str, seen_sm: frozenset[tuple[str, str]] = frozenset()) -> set[str]:
        if (c, m) in seen_sm:
            return set()
        nxt_seen = seen_sm | {(c, m)}
        out = set(cls_method_self_attrs.get((c, m), ()))
        for sb, sm in cls_method_super_calls.get((c, m), ()):
            for anc in (sb, *mro_ancestors.get(sb, ())):
                if sm in cls_methods.get(anc, ()):
                    out |= effective_self_attrs(anc, sm, nxt_seen)
                    break
        return out

    for cls_id, ancs in mro_ancestors.items():
        for m_name in cls_methods.get(cls_id, ()):
            m_id = f"{cls_id}.{m_name}"
            for attr in effective_self_attrs(cls_id, m_name):
                adj[m_id].add(f"{cls_id}.{attr}")
        handled_methods = set(cls_methods.get(cls_id, ()))
        handled_attrs = set(cls_attrs.get(cls_id, ()))
        for b in ancs:
            for m_name in cls_methods.get(b, ()):
                sub_m = f"{cls_id}.{m_name}"
                adj[cls_id].add(sub_m)
                if m_name not in handled_methods:
                    handled_methods.add(m_name)
                    adj[sub_m].add(f"{b}.{m_name}")
                    for attr in effective_self_attrs(b, m_name):
                        adj[sub_m].add(f"{cls_id}.{attr}")
            for a_name in cls_attrs.get(b, ()):
                sub_a = f"{cls_id}.{a_name}"
                adj[cls_id].add(sub_a)
                if a_name not in handled_attrs:
                    handled_attrs.add(a_name)
                    adj[sub_a].add(f"{b}.{a_name}")

    for alias_id, target_id in reexports.items():
        adj[alias_id].add(target_id)

    expanded: dict[str, set[str]] = collections.defaultdict(set)
    for src, dsts in adj.items():
        for dst in dsts:
            expanded[src].add(dst)
            c = canonical(dst)
            if c != dst:
                expanded[src].add(c)
    return expanded, canonical, parsed_mods_set


def _runtime_virgil_origin(node_name: str, parsed_mods: set[str], mod_prefix: str = "virgil") -> str:
    """Resolve PEP 562 ``__getattr__`` re-exports (e.g. ``virgil.pipeline.BinaryPipeline``) to their defining symbol."""
    import importlib
    import sys

    parts = node_name.split(".")
    for split in range(len(parts), 0, -1):
        mod_name = ".".join(parts[:split])
        if mod_name in parsed_mods:
            rest = parts[split:]
            if not rest:
                return node_name
            try:
                if mod_name in sys.modules:
                    obj = sys.modules[mod_name]
                elif mod_name == "virgil":
                    obj = importlib.import_module("virgil")
                elif mod_name.startswith("virgil."):
                    sub = mod_name.removeprefix("virgil.")
                    obj = importlib.import_module(f"virgil.{sub}")
                else:
                    return node_name
                for attr in rest:
                    obj = getattr(obj, attr)
                obj_mod, obj_qn = getattr(obj, "__module__", None), getattr(obj, "__qualname__", None)
                if obj_mod and obj_qn and (obj_mod == mod_prefix or obj_mod.startswith(f"{mod_prefix}.")):
                    return f"{obj_mod}.{obj_qn}"
            except Exception:
                return node_name
            break
    return node_name


def _ast_bfs(
    node_name: str,
    expanded: dict[str, set[str]],
    canonical,
    sinks: set[str],
    *,
    parsed_mods: set[str] | None = None,
    stop_at: set[str] = frozenset(),
    mod_prefix: str = "virgil",
) -> list[str] | None:
    origin_cache: dict[str, str] = {}

    def origin_of(name: str) -> str:
        if parsed_mods is None:
            return name
        if name not in origin_cache:
            origin_cache[name] = _runtime_virgil_origin(name, parsed_mods, mod_prefix)
        return origin_cache[name]

    starts = [canonical(node_name)]
    if parsed_mods is not None:
        origin = origin_of(node_name)
        if origin not in starts:
            starts.append(origin)
    import collections

    q = collections.deque((s, [s]) for s in starts)
    seen = set(starts)
    while q:
        cur, path = q.popleft()
        if cur in sinks:
            return path
        if cur not in starts and cur in stop_at:
            continue
        neighbors = set(expanded.get(cur, ()))
        if parsed_mods is not None:
            o_cur = origin_of(cur)
            if o_cur != cur:
                neighbors.add(o_cur)
            c_cur = canonical(cur)
            if c_cur != cur:
                neighbors.add(c_cur)
        for nxt in sorted(neighbors):
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, [*path, nxt]))
    return None


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_virgil_likelihood_and_fit_callers_depend_on_whitened_residuals():
    """Two separate static reachability invariants on ``virgil.*`` nodes in ``trust/graph.yml``:
    1. Optical ``OIData`` whitening: every node reaching ``virgil.likelihood``'s whitening leaves or
       entry points (without routing solely through generic ``virgil.fitting.fit``, which radio callers
       invoke with ``data=(), likelihoods=[term]``) must transitively depend on
       ``virgil.likelihood.whitened_residuals``.
    2. Generic ``fitting.fit`` callers: every node reaching ``virgil.fitting.fit`` must transitively
       depend on ``virgil.fitting.fit``."""
    import collections
    import importlib.util

    spec = importlib.util.find_spec("virgil")
    assert spec is not None and spec.origin is not None
    virgil_dir = pathlib.Path(spec.origin).resolve().parent
    expanded, canonical, parsed_mods = _build_virgil_ast_reachability(virgil_dir)

    optical_sinks = {
        "virgil.likelihood._whiten_with_log_norm",
        "virgil.likelihood._whiten",
        "virgil.likelihood._whitened_errors_and_log_norm",
        "virgil.likelihood._whitened_and_errors",
        "virgil.likelihood._curvature_rows",
        "virgil.likelihood.whitened_residuals",
        "virgil.likelihood.model_loglike",
        "virgil.likelihood.numpyro_model",
    }
    fit_sinks = {"virgil.fitting.fit"}

    nodes = GRAPH["nodes"]

    def graph_reaches(start: str, target: str, graph_nodes: dict = nodes) -> bool:
        seen, q = {start}, collections.deque([start])
        while q:
            cur = q.popleft()
            if cur == target:
                return True
            for dep in (graph_nodes.get(cur) or {}).get("depends", []):
                if dep not in seen:
                    seen.add(dep)
                    q.append(dep)
        return False

    assert _ast_bfs("virgil.inference.laplace_cov", expanded, canonical, optical_sinks, parsed_mods=parsed_mods, stop_at=fit_sinks) is not None
    assert _ast_bfs("virgil.ensemble.draw_groups", expanded, canonical, fit_sinks, parsed_mods=parsed_mods) is not None
    assert _ast_bfs("virgil.ensemble.draw_groups", expanded, canonical, optical_sinks, parsed_mods=parsed_mods, stop_at=fit_sinks) is None
    assert _ast_bfs("virgil.models.UniformDisk", expanded, canonical, optical_sinks | fit_sinks, parsed_mods=parsed_mods) is None
    mutated_nodes = {
        **nodes,
        "virgil.ensemble.draw_groups": {
            **(nodes["virgil.ensemble.draw_groups"] or {}),
            "depends": [
                d for d in (nodes["virgil.ensemble.draw_groups"] or {}).get("depends", []) if d != "virgil.fitting.fit"
            ],
        },
    }
    assert not graph_reaches("virgil.ensemble.draw_groups", "virgil.fitting.fit", mutated_nodes)

    missing_optical, missing_fit = [], []
    for name in sorted(nodes):
        if not name.startswith("virgil."):
            continue
        opt_path = _ast_bfs(name, expanded, canonical, optical_sinks, parsed_mods=parsed_mods, stop_at=fit_sinks)
        if opt_path is not None and not graph_reaches(name, "virgil.likelihood.whitened_residuals"):
            missing_optical.append((name, " -> ".join(opt_path)))
        fit_path = _ast_bfs(name, expanded, canonical, fit_sinks, parsed_mods=parsed_mods)
        if fit_path is not None and not graph_reaches(name, "virgil.fitting.fit"):
            missing_fit.append((name, " -> ".join(fit_path)))
    assert not missing_optical, missing_optical
    assert not missing_fit, missing_fit


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_virgil_ast_reachability_synthetic_controls(tmp_path, monkeypatch):
    """Synthetic package controls for ``_build_virgil_ast_reachability``: private whitening leaves
    (``_whiten_with_log_norm``), ``super().method()`` and MRO-inherited base methods, module-level
    graph nodes, class ``STAGES`` attribute access, nested comprehension/helper non-masking,
    intermediate PEP 562 ``__getattr__`` re-exports, non-collision on ``PositionData.whitened_residuals``,
    and ``fitting.fit`` separation."""
    pkg = tmp_path / "fakepkg"
    (pkg / "lazy").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "likelihood.py").write_text(
        "def _whiten_with_log_norm(d, m): return ()\n"
        "def whitened_residuals(d, m): return _whiten_with_log_norm(d, m)\n",
        encoding="utf-8",
    )
    (pkg / "fitting.py").write_text(
        "from .likelihood import whitened_residuals\n"
        "def fit(m, data=(), likelihoods=()): return whitened_residuals(data, m) if data else ()\n",
        encoding="utf-8",
    )
    (pkg / "orbits.py").write_text(
        "class PositionData:\n"
        "    def whitened_residuals(self): return 0.0\n"
        "def astrometric_score(pos: PositionData):\n"
        "    return pos.whitened_residuals()\n",
        encoding="utf-8",
    )
    (pkg / "modnode.py").write_text(
        "from .likelihood import _whiten_with_log_norm\n"
        "def helper(d, m): return _whiten_with_log_norm(d, m)\n",
        encoding="utf-8",
    )
    (pkg / "lazy" / "_impl.py").write_text(
        "from ..likelihood import _whiten_with_log_norm\n"
        "class RealPipe:\n"
        "    def run(self, d, m): return _whiten_with_log_norm(d, m)\n"
        "class TemplateBase:\n"
        "    STAGES = ()\n"
        "    def _hook(self, d, m): return ()\n"
        "    def execute(self, d, m):\n"
        "        return [s(d, m) for s in self.STAGES] + list(self._hook(d, m))\n",
        encoding="utf-8",
    )
    (pkg / "lazy" / "__init__.py").write_text(
        "def __getattr__(name):\n"
        "    if name == 'LazyPipe':\n"
        "        from ._impl import RealPipe\n"
        "        return RealPipe\n"
        "    if name == 'LazyTemplateBase':\n"
        "        from ._impl import TemplateBase\n"
        "        return TemplateBase\n"
        "    raise AttributeError(name)\n",
        encoding="utf-8",
    )
    (pkg / "pipelines.py").write_text(
        "from .likelihood import _whiten_with_log_norm\n"
        "from .fitting import fit\n"
        "from .lazy import LazyTemplateBase\n"
        "def _stage_fn(d, m): return _whiten_with_log_norm(d, m)\n"
        "class BasePipe:\n"
        "    def run(self, d, m): return _whiten_with_log_norm(d, m)\n"
        "    def step(self, d, m): return _whiten_with_log_norm(d, m)\n"
        "class InheritedPipe(BasePipe):\n"
        "    pass\n"
        "class SuperPipe(BasePipe):\n"
        "    def step(self, d, m): return super().step(d, m)\n"
        "class StagedPipe:\n"
        "    STAGES = (_stage_fn,)\n"
        "class TemplateStageSub(LazyTemplateBase):\n"
        "    STAGES = (_stage_fn,)\n"
        "class TemplateHookSub(LazyTemplateBase):\n"
        "    def _hook(self, d, m): return _whiten_with_log_norm(d, m)\n"
        "class SilentTemplateSub(LazyTemplateBase):\n"
        "    def _hook(self, d, m): return (d, m)\n"
        "def run_via_stages(d, m):\n"
        "    return [s(d, m) for s in StagedPipe.STAGES]\n"
        "def comp_does_not_mask_fit(m, xs):\n"
        "    _ = [fit for fit in xs]\n"
        "    def _inner(fit): return fit\n"
        "    return fit(m, data=(), likelihoods=())\n"
        "def nested_closure_reimports_shadowed_name(fn, d, m):\n"
        "    fn = None\n"
        "    def _closure():\n"
        "        from .likelihood import _whiten_with_log_norm as fn\n"
        "        return fn(d, m)\n"
        "    return _closure()\n"
        "def calls_pep562_hop(d, m):\n"
        "    from .lazy import LazyPipe\n"
        "    return LazyPipe().run(d, m)\n"
        "def radio_staged(m, term):\n"
        "    return fit(m, data=(), likelihoods=(term,))\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    import fakepkg.lazy  # noqa: F401 — populate sys.modules for synthetic PEP 562 test without dynamic non-virgil import_module
    expanded, canonical, parsed_mods = _build_virgil_ast_reachability(pkg, mod_prefix="fakepkg")
    opt_sinks = {"fakepkg.likelihood._whiten_with_log_norm", "fakepkg.likelihood.whitened_residuals"}
    fit_sinks = {"fakepkg.fitting.fit"}

    assert _ast_bfs("fakepkg.modnode", expanded, canonical, opt_sinks, parsed_mods=parsed_mods, stop_at=fit_sinks, mod_prefix="fakepkg") is not None
    assert _ast_bfs("fakepkg.pipelines.InheritedPipe", expanded, canonical, opt_sinks, parsed_mods=parsed_mods, stop_at=fit_sinks, mod_prefix="fakepkg") is not None
    assert _ast_bfs("fakepkg.pipelines.InheritedPipe.run", expanded, canonical, opt_sinks, parsed_mods=parsed_mods, stop_at=fit_sinks, mod_prefix="fakepkg") is not None
    assert _ast_bfs("fakepkg.pipelines.SuperPipe", expanded, canonical, opt_sinks, parsed_mods=parsed_mods, stop_at=fit_sinks, mod_prefix="fakepkg") is not None
    assert _ast_bfs("fakepkg.pipelines.StagedPipe.STAGES", expanded, canonical, opt_sinks, parsed_mods=parsed_mods, stop_at=fit_sinks, mod_prefix="fakepkg") is not None
    assert _ast_bfs("fakepkg.pipelines.TemplateStageSub.execute", expanded, canonical, opt_sinks, parsed_mods=parsed_mods, stop_at=fit_sinks, mod_prefix="fakepkg") is not None
    assert _ast_bfs("fakepkg.pipelines.TemplateHookSub.execute", expanded, canonical, opt_sinks, parsed_mods=parsed_mods, stop_at=fit_sinks, mod_prefix="fakepkg") is not None
    assert _ast_bfs("fakepkg.pipelines.SilentTemplateSub.execute", expanded, canonical, opt_sinks, parsed_mods=parsed_mods, stop_at=fit_sinks, mod_prefix="fakepkg") is None
    assert _ast_bfs("fakepkg.pipelines.run_via_stages", expanded, canonical, opt_sinks, parsed_mods=parsed_mods, stop_at=fit_sinks, mod_prefix="fakepkg") is not None
    assert _ast_bfs("fakepkg.pipelines.comp_does_not_mask_fit", expanded, canonical, fit_sinks, parsed_mods=parsed_mods, mod_prefix="fakepkg") is not None
    assert _ast_bfs("fakepkg.pipelines.nested_closure_reimports_shadowed_name", expanded, canonical, opt_sinks, parsed_mods=parsed_mods, stop_at=fit_sinks, mod_prefix="fakepkg") is not None
    assert _ast_bfs("fakepkg.pipelines.calls_pep562_hop", expanded, canonical, opt_sinks, parsed_mods=parsed_mods, stop_at=fit_sinks, mod_prefix="fakepkg") is not None
    assert _ast_bfs("fakepkg.orbits.astrometric_score", expanded, canonical, opt_sinks, parsed_mods=parsed_mods, stop_at=fit_sinks, mod_prefix="fakepkg") is None
    assert _ast_bfs("fakepkg.pipelines.radio_staged", expanded, canonical, opt_sinks, parsed_mods=parsed_mods, stop_at=fit_sinks, mod_prefix="fakepkg") is None
    assert _ast_bfs("fakepkg.pipelines.radio_staged", expanded, canonical, fit_sinks, parsed_mods=parsed_mods, mod_prefix="fakepkg") is not None
