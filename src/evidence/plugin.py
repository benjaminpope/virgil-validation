"""pytest plugin: the ``validates`` marker, the ``metric`` fixture and the
``--evidence PATH`` option (see evidence/__init__.py)."""

import ast
import collections
import functools
import json
import math
import os
import pathlib
import re

import pytest

from . import KINDS, ROOTS
from .meta import invocation, run_header

_RECORDS = pytest.StashKey[list]()
_STAMP = pytest.StashKey[dict]()
_HEADER = pytest.StashKey[dict]()
_DESELECTED = pytest.StashKey[list]()
_CONFIG = {}


def pytest_addoption(parser):
    parser.addoption(
        "--evidence",
        metavar="PATH",
        default=None,
        help="write evidence records (JSON lines) to PATH",
    )
    parser.addoption(
        "--require-validates",
        action="store_true",
        help="fail collection if a test has no validates marker",
    )


def record(name, value):
    """Record a number with the evidence of the running test, e.g. the
    largest difference an assertion checked. Usable from helpers that have
    no access to fixtures."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"metric {name!r} must be finite, got {value!r}")
    item = _CONFIG.get("item")
    if item is None:
        return value
    item.user_properties.append((name, value))
    return value


def stamp_campaign(summary):
    """For a test that reads a campaign summary (``trust/campaigns/*.json``): put the commits the summary was made at
    into the test's evidence record, as ``virgil_commit`` and ``criteria_commit``. The Trust page reads a test's
    ``virgil_commit`` in preference to its run's, so a campaign measured on one virgil commit is shown at that commit
    whichever virgil the run installed. A summary without a ``criteria_commit`` predates the criteria-commit scheme and is
    not gate evidence (AGENTS.md), one without a ``virgil_commit`` cannot be placed, and one whose ``virgil_dirty`` is
    true or missing ran on an edited or unknown virgil tree: all skip. The record also carries ``virgil_dirty: false``,
    which ``scripts/trust.py --strict-provenance`` requires of a stamped record. ``check_evidence_pin.py --release``
    fails a stamped record at any commit but the release's."""
    if not summary.get("criteria_commit"):
        pytest.skip("the committed summary has no criteria_commit (it predates the criteria-commit scheme): "
                    "not gate evidence; rerun the campaign at the release commit")
    if not summary.get("virgil_commit"):
        pytest.skip("the committed summary does not say which virgil commit it ran on: rerun the campaign")
    if summary.get("virgil_dirty") is not False:
        pytest.skip("the committed summary does not record a clean virgil tree (virgil_dirty is "
                    f"{summary.get('virgil_dirty')!r}): rerun the campaign from a clean virgil checkout")
    item = _CONFIG.get("item")
    if item is not None:
        item.stash[_STAMP] = {"virgil_commit": summary["virgil_commit"], "virgil_dirty": False,
                              "criteria_commit": summary["criteria_commit"]}


def metrics_of(user_properties):
    """{name: value} of the recorded properties. A name recorded more than once keeps every value, as a list in
    recording order (``dict()`` would keep only the last, hiding the other numbers a test checked)."""
    values = {}
    for name, value in user_properties:
        values.setdefault(name, []).append(value)
    return {name: v[0] if len(v) == 1 else v for name, v in values.items()}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    _CONFIG["item"] = item
    try:
        yield
    finally:
        _CONFIG["item"] = None


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "validates(obj, *more, roots, kind='check', tier='A', headline=None): "
        "the virgil objects (or our references) this test validates, against "
        "which roots of trust, and optionally which recorded metric "
        "summarises it on the Trust page",
    )
    config.stash[_RECORDS] = []
    config.stash[_DESELECTED] = []
    _CONFIG["config"] = config


# Our reference code: a check that goes through one of these counts only if
# that module is itself verified (fail-closed across all three packages).
# _subprocess is the JSON-over-interpreter launcher; every _subprocess.run
# call resolves its concrete worker module (or emits an unregistered sentinel).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
_TESTS_ROOT = _REPO_ROOT / "tests"
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
_BENCH_POLAR_ROOT = _REPO_ROOT / "benchmarks" / "polar_disk"
_REFERENCE_PACKAGES = ("crosscheck", "external_bridge", "virgil_bridge")
_PLUMBING = frozenset({"external_bridge._subprocess"})
_UNRESOLVED_WORKER = "external_bridge.<unresolved_worker>"
_SAFE_SP_ATTRS = frozenset({"run", "available", "python", "version", "_env", "ROOT"})


def _keep_reference(mod):
    return mod not in _PLUMBING


def _is_submodule_parts(pkg_parts, name):
    base = _SRC_ROOT.joinpath(*pkg_parts)
    return (base / f"{name}.py").is_file() or (base / name / "__init__.py").is_file()


@functools.lru_cache(maxsize=None)
def _init_exports_cached(src_root, pkg_parts):
    init_py = pathlib.Path(src_root).joinpath(*pkg_parts, "__init__.py")
    if not init_py.is_file():
        return frozenset()
    try:
        tree = ast.parse(init_py.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return frozenset()
    names = set()
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            names.add(stmt.target.id)
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for alias in stmt.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return frozenset(names)


def _init_exports(pkg_parts):
    """Top-level names defined or bound in ``src/<pkg_parts>/__init__.py``."""
    if isinstance(pkg_parts, str):
        pkg_parts = (pkg_parts,)
    return _init_exports_cached(str(_SRC_ROOT), tuple(pkg_parts))


_init_exports.cache_clear = _init_exports_cached.cache_clear


def _reference_pkg_parts(path):
    try:
        rel = pathlib.Path(path).resolve().relative_to(_SRC_ROOT.resolve())
    except ValueError:
        return None
    if len(rel.parts) < 2 or rel.parts[0] not in _REFERENCE_PACKAGES:
        return None
    return rel.parts[:-1]


def _local_helper_dirs(resolved_path):
    try:
        resolved_path.relative_to(_REPO_ROOT.resolve())
        in_repo = True
    except ValueError:
        in_repo = False
    if in_repo:
        return (resolved_path.parent, _TESTS_ROOT, _SCRIPTS_ROOT, _BENCH_POLAR_ROOT, _REPO_ROOT)
    extra = ()
    if resolved_path.parent.name in {"tests", "scripts"}:
        root = resolved_path.parent.parent
        extra = (root / "tests", root / "scripts", root)
    return (resolved_path.parent, *extra)


def _resolve_local_helper(parts, search_dirs):
    if not parts or parts[0] in (*_REFERENCE_PACKAGES, "evidence", "virgil"):
        return None
    src_resolved = _SRC_ROOT.resolve()
    for base in search_dirs:
        for cand in (
            base.joinpath(*parts[:-1], f"{parts[-1]}.py"),
            base.joinpath(*parts, "__init__.py"),
        ):
            if not cand.is_file():
                continue
            try:
                cand.resolve().relative_to(src_resolved)
                continue
            except ValueError:
                return cand
    return None


def _call_arg_expr(call, index, keyword):
    if index is not None and index < len(call.args):
        return call.args[index]
    for kw in call.keywords:
        if kw.arg == keyword:
            return kw.value
    return None


def _resolve_str(expr, const_strings, shadowed=frozenset()):
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    if isinstance(expr, ast.Name) and expr.id not in shadowed and expr.id in const_strings:
        return const_strings[expr.id]
    return None


def _collect_scope_bindings(node, counts, is_scope_root=False):
    """Count lexical name bindings inside ``node``'s scope without crossing into nested scopes."""
    if not is_scope_root and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        counts[node.name] += 1
        for sub in (
            *node.decorator_list,
            *node.args.defaults,
            *(d for d in node.args.kw_defaults if d is not None),
            *(a.annotation for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs) if a.annotation),
            *([node.args.vararg.annotation] if node.args.vararg and node.args.vararg.annotation else []),
            *([node.args.kwarg.annotation] if node.args.kwarg and node.args.kwarg.annotation else []),
            *([node.returns] if node.returns else []),
        ):
            _collect_scope_bindings(sub, counts)
        return
    if not is_scope_root and isinstance(node, ast.ClassDef):
        counts[node.name] += 1
        for sub in (*node.decorator_list, *node.bases, *node.keywords):
            _collect_scope_bindings(sub, counts)
        return
    if not is_scope_root and isinstance(node, ast.Lambda):
        for sub in (*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)):
            _collect_scope_bindings(sub, counts)
        return
    if not is_scope_root and isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        # PEP 572: NamedExpr inside a comprehension binds in the enclosing non-comprehension scope
        for sub in ast.walk(node):
            if isinstance(sub, ast.NamedExpr) and isinstance(sub.target, ast.Name):
                counts[sub.target.id] += 1
        return

    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        counts[node.id] += 1
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            if alias.name == "*":
                counts["*"] += 1
            else:
                counts[alias.asname or alias.name.split(".")[0]] += 1
    elif isinstance(node, ast.ExceptHandler) and node.name:
        counts[node.name] += 1
    elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
        counts[node.name] += 1
    elif isinstance(node, ast.MatchMapping) and node.rest:
        counts[node.rest] += 1

    if is_scope_root and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            counts[a.arg] += 1
        if node.args.vararg:
            counts[node.args.vararg.arg] += 1
        if node.args.kwarg:
            counts[node.args.kwarg.arg] += 1

    if is_scope_root and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for stmt in node.body:
            _collect_scope_bindings(stmt, counts)
        return
    if is_scope_root and isinstance(node, ast.ClassDef):
        for stmt in node.body:
            _collect_scope_bindings(stmt, counts)
        return
    if is_scope_root and isinstance(node, ast.Lambda):
        _collect_scope_bindings(node.body, counts)
        return

    for child in ast.iter_child_nodes(node):
        _collect_scope_bindings(child, counts)


def _subprocess_workers(tree, sp_aliases, run_aliases, eb_aliases, wildcard_sp):
    """Worker scripts invoked through ``external_bridge._subprocess.run`` or local wrappers.

    Any ``_subprocess.run`` call or dynamic alias usage that cannot be statically resolved
    to a concrete worker module emits ``_UNRESOLVED_WORKER`` so ``trust.py`` and
    ``test_trust_graph.py`` fail closed.
    """
    if wildcard_sp:
        return {_UNRESOLVED_WORKER}
    if not sp_aliases and not run_aliases and not eb_aliases:
        return set()

    module_counts = collections.Counter()
    _collect_scope_bindings(tree, module_counts, is_scope_root=True)
    global_names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Global):
            for gname in n.names:
                module_counts[gname] += 1
                global_names.add(gname)

    const_strings = {}
    if module_counts["*"] == 0:
        for stmt in tree.body:
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                for t in stmt.targets:
                    if isinstance(t, ast.Name) and module_counts[t.id] == 1:
                        const_strings[t.id] = stmt.value.value
            elif (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
                and module_counts[stmt.target.id] == 1
            ):
                const_strings[stmt.target.id] = stmt.value.value

    def _is_sp_module_expr(expr):
        return (
            (isinstance(expr, ast.Name) and expr.id in sp_aliases)
            or (
                isinstance(expr, ast.Attribute)
                and expr.attr == "_subprocess"
                and isinstance(expr.value, ast.Name)
                and expr.value.id in eb_aliases
            )
        )

    def _is_sp_run(func):
        return (
            (isinstance(func, ast.Attribute) and func.attr == "run" and _is_sp_module_expr(func.value))
            or (isinstance(func, ast.Name) and func.id in run_aliases)
        )

    fn_nodes = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    fn_counts = {}
    fn_bound = {}
    for fn in fn_nodes:
        c = collections.Counter()
        _collect_scope_bindings(fn, c, is_scope_root=True)
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Nonlocal):
                for nl in sub.names:
                    c[nl] += 1
        fn_counts[id(fn)] = c
        fn_bound[id(fn)] = frozenset(k for k in c if k != "*")

    wrappers = {}
    changed = True
    while changed:
        changed = False
        for fn in fn_nodes:
            if fn.name in wrappers:
                continue
            pos_params = [a.arg for a in (*fn.args.posonlyargs, *fn.args.args)]
            kw_params = {a.arg for a in fn.args.kwonlyargs}
            all_params = set(pos_params) | kw_params
            fc = fn_counts[id(fn)]
            for sub in ast.walk(fn):
                if not isinstance(sub, ast.Call):
                    continue
                wexpr = None
                if _is_sp_run(sub.func):
                    wexpr = _call_arg_expr(sub, 1, "worker")
                elif isinstance(sub.func, ast.Name) and sub.func.id in wrappers:
                    idx, pname = wrappers[sub.func.id]
                    wexpr = _call_arg_expr(sub, idx, pname)
                if (
                    isinstance(wexpr, ast.Name)
                    and wexpr.id in all_params
                    and fc[wexpr.id] == 1
                    and wexpr.id not in global_names
                ):
                    p_idx = pos_params.index(wexpr.id) if wexpr.id in pos_params else None
                    wrappers[fn.name] = (p_idx, wexpr.id)
                    changed = True
                    break

    workers = set()
    called_wrappers = set()
    direct_call_funcs = set()
    attr_parents = {}

    for parent in ast.walk(tree):
        if isinstance(parent, ast.Call):
            direct_call_funcs.add(id(parent.func))
        elif isinstance(parent, ast.Attribute):
            attr_parents[id(parent.value)] = parent

    def _scan_node(node, enclosing_wrapper_param, shadowed):
        if isinstance(node, ast.Call):
            handled = False
            wexpr = None
            if _is_sp_run(node.func):
                handled = True
                wexpr = _call_arg_expr(node, 1, "worker")
            elif isinstance(node.func, ast.Name) and node.func.id in wrappers:
                handled = True
                called_wrappers.add(node.func.id)
                idx, pname = wrappers[node.func.id]
                wexpr = _call_arg_expr(node, idx, pname)
            if handled:
                if (
                    enclosing_wrapper_param is not None
                    and isinstance(wexpr, ast.Name)
                    and wexpr.id == enclosing_wrapper_param
                ):
                    pass
                else:
                    script = _resolve_str(wexpr, const_strings, shadowed)
                    if script:
                        stem = pathlib.Path(script).stem if script.endswith(".py") else script
                        workers.add(f"external_bridge.{stem}")
                    else:
                        workers.add(_UNRESOLVED_WORKER)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in run_aliases and id(node) not in direct_call_funcs:
                workers.add(_UNRESOLVED_WORKER)
            elif node.id in sp_aliases:
                parent_attr = attr_parents.get(id(node))
                if parent_attr is None or parent_attr.attr not in _SAFE_SP_ATTRS:
                    workers.add(_UNRESOLVED_WORKER)
                elif parent_attr.attr == "run" and id(parent_attr) not in direct_call_funcs:
                    workers.add(_UNRESOLVED_WORKER)
            elif node.id in wrappers and id(node) not in direct_call_funcs:
                workers.add(_UNRESOLVED_WORKER)
        elif (
            isinstance(node, ast.Attribute)
            and node.attr == "_subprocess"
            and isinstance(node.value, ast.Name)
            and node.value.id in eb_aliases
        ):
            parent_attr = attr_parents.get(id(node))
            if parent_attr is None or parent_attr.attr not in _SAFE_SP_ATTRS:
                workers.add(_UNRESOLVED_WORKER)
            elif parent_attr.attr == "run" and id(parent_attr) not in direct_call_funcs:
                workers.add(_UNRESOLVED_WORKER)

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in (
                *node.decorator_list,
                *node.args.defaults,
                *(d for d in node.args.kw_defaults if d is not None),
                *(a.annotation for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs) if a.annotation),
                *([node.args.vararg.annotation] if node.args.vararg and node.args.vararg.annotation else []),
                *([node.args.kwarg.annotation] if node.args.kwarg and node.args.kwarg.annotation else []),
                *([node.returns] if node.returns else []),
            ):
                _scan_node(sub, enclosing_wrapper_param, shadowed)
            wparam = wrappers[node.name][1] if node.name in wrappers else None
            fn_shadowed = (shadowed | fn_bound[id(node)]) if "*" not in fn_counts[id(node)] else frozenset(const_strings)
            for stmt in node.body:
                _scan_node(stmt, wparam, fn_shadowed)
            return
        if isinstance(node, ast.ClassDef):
            for sub in (*node.decorator_list, *node.bases, *node.keywords):
                _scan_node(sub, enclosing_wrapper_param, shadowed)
            cls_counts = collections.Counter()
            _collect_scope_bindings(node, cls_counts, is_scope_root=True)
            cls_shadowed = (
                (shadowed | frozenset(k for k in cls_counts if k != "*"))
                if "*" not in cls_counts
                else frozenset(const_strings)
            )
            for stmt in node.body:
                _scan_node(stmt, None, cls_shadowed)
            return
        if isinstance(node, ast.Lambda):
            for sub in (*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)):
                _scan_node(sub, enclosing_wrapper_param, shadowed)
            lam_counts = collections.Counter()
            _collect_scope_bindings(node, lam_counts, is_scope_root=True)
            _scan_node(node.body, None, shadowed | frozenset(lam_counts))
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            comp_counts = collections.Counter()
            _collect_scope_bindings(node, comp_counts, is_scope_root=True)
            comp_shadowed = shadowed | frozenset(comp_counts)
            for child in ast.iter_child_nodes(node):
                _scan_node(child, enclosing_wrapper_param, comp_shadowed)
            return

        for child in ast.iter_child_nodes(node):
            _scan_node(child, enclosing_wrapper_param, shadowed)

    _scan_node(tree, None, frozenset())
    if set(wrappers) - called_wrappers:
        workers.add(_UNRESOLVED_WORKER)
    return workers


def _scan_module_via(path, seen):
    p = pathlib.Path(path)
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p
    if resolved in seen:
        return set()
    seen = seen | {resolved}
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return set()

    pkg_parts = _reference_pkg_parts(resolved)
    local_dirs = () if pkg_parts is not None else _local_helper_dirs(resolved)
    found = set()
    sp_aliases = set()
    run_aliases = set()
    eb_aliases = set()
    wildcard_sp = False

    def _expand_pkg_parts(target_parts):
        if len(target_parts) == 1:
            init_py = _SRC_ROOT / target_parts[0] / "__init__.py"
            if init_py.is_file():
                found.update(_scan_module_via(init_py, seen))
        else:
            found.add(".".join(target_parts))

    def _handle_from_ref(target_parts, aliases):
        nonlocal wildcard_sp
        mod_file = _SRC_ROOT.joinpath(*target_parts[:-1], f"{target_parts[-1]}.py") if len(target_parts) >= 2 else None
        if mod_file is not None and mod_file.is_file():
            mod = ".".join(target_parts)
            found.add(mod)
            if mod == "external_bridge._subprocess":
                for alias in aliases:
                    if alias.name == "*":
                        wildcard_sp = True
                    elif alias.name == "run":
                        run_aliases.add(alias.asname or "run")
            return
        if _SRC_ROOT.joinpath(*target_parts).is_dir():
            exports = _init_exports(target_parts)
            for alias in aliases:
                if _is_submodule_parts(target_parts, alias.name):
                    mod = ".".join((*target_parts, alias.name))
                    found.add(mod)
                    if mod == "external_bridge._subprocess":
                        sp_aliases.add(alias.asname or "_subprocess")
                elif alias.name == "*" or alias.name in exports:
                    _expand_pkg_parts(target_parts)
                else:
                    found.add(".".join((*target_parts, alias.name)))
            return
        found.add(".".join(target_parts))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = tuple(alias.name.split("."))
                top = parts[0]
                if top in _REFERENCE_PACKAGES:
                    if len(parts) >= 2:
                        mod = ".".join(parts)
                        found.add(mod)
                        if mod == "external_bridge._subprocess":
                            if alias.asname:
                                sp_aliases.add(alias.asname)
                            else:
                                eb_aliases.add("external_bridge")
                    else:
                        _expand_pkg_parts(parts)
                        if top == "external_bridge":
                            eb_aliases.add(alias.asname or "external_bridge")
                elif pkg_parts is not None and _is_submodule_parts(pkg_parts, top):
                    mod = ".".join((*pkg_parts, *parts))
                    found.add(mod)
                    if mod == "external_bridge._subprocess" and len(parts) == 1:
                        sp_aliases.add(alias.asname or "_subprocess")
                elif pkg_parts is None:
                    helper = _resolve_local_helper(parts, local_dirs)
                    if helper is not None:
                        found.update(_scan_module_via(helper, seen))
        elif isinstance(node, ast.ImportFrom):
            if node.level >= 1 and pkg_parts is not None:
                if node.level > len(pkg_parts):
                    found.add(f"{pkg_parts[0]}.<invalid_relative_import>")
                else:
                    anchor = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                    target_parts = (*anchor, *node.module.split(".")) if node.module else anchor
                    _handle_from_ref(target_parts, node.names)
            elif node.level >= 1 and pkg_parts is None:
                anchor_dir = resolved.parent
                valid = True
                for _ in range(node.level - 1):
                    if anchor_dir.parent == anchor_dir:
                        valid = False
                        break
                    anchor_dir = anchor_dir.parent
                if valid:
                    if node.module:
                        rel_parts = tuple(node.module.split("."))
                        helper = _resolve_local_helper(rel_parts, (anchor_dir,))
                        if helper is not None:
                            found.update(_scan_module_via(helper, seen))
                        for alias in node.names:
                            if alias.name != "*":
                                sub_helper = _resolve_local_helper((*rel_parts, alias.name), (anchor_dir,))
                                if sub_helper is not None:
                                    found.update(_scan_module_via(sub_helper, seen))
                    else:
                        for alias in node.names:
                            if alias.name != "*":
                                sub_helper = _resolve_local_helper((alias.name,), (anchor_dir,))
                                if sub_helper is not None:
                                    found.update(_scan_module_via(sub_helper, seen))
            elif node.level == 0 and node.module:
                parts = tuple(node.module.split("."))
                top = parts[0]
                if top in _REFERENCE_PACKAGES:
                    _handle_from_ref(parts, node.names)
                elif pkg_parts is None:
                    helper = _resolve_local_helper(parts, local_dirs)
                    if helper is not None:
                        found.update(_scan_module_via(helper, seen))
                    for alias in node.names:
                        if alias.name != "*":
                            sub_helper = _resolve_local_helper((*parts, alias.name), local_dirs)
                            if sub_helper is not None:
                                found.update(_scan_module_via(sub_helper, seen))

    found |= _subprocess_workers(tree, sp_aliases, run_aliases, eb_aliases, wildcard_sp)
    return {m for m in found if _keep_reference(m)}


@functools.lru_cache(maxsize=None)
def _module_via(path):
    """Reference modules and subprocess workers a file imports or invokes, anywhere in it."""
    return tuple(sorted(_scan_module_via(path, frozenset())))


@functools.lru_cache(maxsize=1)
def _golden_names():
    """Names listed in trust/golden.yml (the only golden roots that count)."""
    import yaml

    path = pathlib.Path(__file__).resolve().parents[2] / "trust" / "golden.yml"
    return frozenset(yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.exists() else frozenset()


def _claims(item):
    claims = []
    for mark in item.iter_markers("validates"):
        objects = list(mark.args)
        roots = list(mark.kwargs.get("roots", ()))
        kind = mark.kwargs.get("kind", "check")
        tier = mark.kwargs.get("tier", "A")
        headline = mark.kwargs.get("headline")
        if not objects or not roots:
            raise pytest.UsageError(f"{item.nodeid}: validates needs objects and roots")
        bad = [r for r in roots if r not in ROOTS and not r.startswith("golden:")]
        if bad or kind not in KINDS or tier not in ("A", "B", "C"):
            raise pytest.UsageError(
                f"{item.nodeid}: unknown roots {bad}, kind {kind!r} or tier {tier!r}"
            )
        unlisted = [r for r in roots if r.startswith("golden:") and r[7:] not in _golden_names()]
        if unlisted:
            raise pytest.UsageError(
                f"{item.nodeid}: golden roots {unlisted} are not listed in trust/golden.yml"
            )
        via = set(_module_via(str(item.path))) | set(mark.kwargs.get("via", ()))
        prop = mark.kwargs.get("property", ())
        claim = {"objects": objects, "roots": roots, "kind": kind, "tier": tier,
                 "via": sorted(via - set(objects)),
                 "properties": [prop] if isinstance(prop, str) else list(prop)}
        if headline is not None:
            claim["headline"] = str(headline)
        claims.append(claim)
    return claims


def pytest_collection_modifyitems(config, items):
    # fail early on malformed markers, even without --evidence
    missing = []
    bad_xfail = []
    for item in items:
        claims = _claims(item)
        if not claims:
            missing.append(item.nodeid)
            continue
        xfail_marks = list(item.iter_markers("xfail"))
        has_finding = any(c["kind"] in ("finding", "upstream") for c in claims)
        if xfail_marks:
            if not all(m.kwargs.get("strict") is True for m in xfail_marks):
                bad_xfail.append(f"{item.nodeid}: xfail must set strict=True")
            if not has_finding:
                bad_xfail.append(
                    f"{item.nodeid}: xfail requires kind='finding' or kind='upstream'"
                )
            if not all(m.kwargs.get("raises") for m in xfail_marks):
                bad_xfail.append(
                    f"{item.nodeid}: xfail must set raises=<exception class> "
                    "(a strict xfail counts only the failure it states)"
                )
            else:
                for m in xfail_marks:
                    raises = m.kwargs["raises"]
                    classes = raises if isinstance(raises, tuple) else (raises,)
                    if any(c in (Exception, BaseException) for c in classes):
                        bad_xfail.append(
                            f"{item.nodeid}: xfail raises= must name the specific exception class, "
                            "not Exception or BaseException (they match any failure, including an API change)"
                        )
                        break
        elif has_finding:
            bad_xfail.append(
                f"{item.nodeid}: kind='finding' or 'upstream' requires @pytest.mark.xfail(strict=True, raises=<exception class>)"
            )
    if missing and config.getoption("--require-validates"):
        raise pytest.UsageError(
            "tests without a validates marker (docs/method/index.md):\n  "
            + "\n  ".join(missing)
        )
    if bad_xfail:
        raise pytest.UsageError(
            "invalid xfail / finding markers:\n  " + "\n  ".join(bad_xfail)
        )


@pytest.fixture
def metric(record_property):
    """Record a number with the evidence: ``metric("max_dV2", err)``."""

    def record(name, value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"metric {name!r} must be finite, got {value!r}")
        record_property(name, value)
        return value

    return record


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    report.validates = _claims(item)
    report.validates_where = {**_where(item), **item.stash.get(_STAMP, {})}
    # call phase only: a headline metric recorded in fixture teardown (after a yield) is not yet in user_properties here
    # and would fail with a misleading "not a recorded metric"; record headline metrics in the test body
    if call.when == "call" and report.passed and not hasattr(report, "wasxfail"):
        recorded = {name for name, _ in report.user_properties}
        for claim in report.validates:
            name = claim.get("headline")
            if name is not None and name not in recorded:
                report.outcome = "failed"
                report.longrepr = f"headline {name} is not a recorded metric"
                break


def _where(item):
    """The test's line (1-based) and the first paragraph of its docstring,
    for the Trust page's description of each check."""
    doc = getattr(getattr(item, "function", None), "__doc__", None) or ""
    first = " ".join(doc.strip().split("\n\n")[0].split())
    return {"line": item.location[1] + 1, "doc": first}


def _outcome(report):
    if hasattr(report, "wasxfail"):
        return "xfailed" if report.skipped else "xpassed"
    return report.outcome


def pytest_runtest_logreport(report):
    """One record per test, finished only at teardown, so that a failure
    in any phase decides the outcome as pytest's own summary does: a
    failing setup or teardown is an error even if the body passed."""
    claims = getattr(report, "validates", None)
    if not claims:
        return
    pending = _CONFIG.setdefault("pending", {})
    if report.when == "setup":
        if report.outcome != "passed":
            pending[report.nodeid] = (
                "error" if report.failed else _outcome(report),
                report,
            )
    elif report.when == "call":
        pending[report.nodeid] = (_outcome(report), report)
    elif report.when == "teardown":
        outcome, decisive = pending.pop(report.nodeid, (None, None))
        if outcome is None:
            return
        if report.failed:
            outcome = "error"
        records = _CONFIG["config"].stash[_RECORDS]
        for claim in claims:
            records.append(
                {
                    "record": "test",
                    "test": report.nodeid,
                    **claim,
                    "outcome": outcome,
                    "duration_s": round(decisive.duration, 3),
                    # the teardown report carries every recorded property
                    "metrics": metrics_of(report.user_properties),
                    **getattr(report, "validates_where", {}),
                }
            )


def pytest_sessionstart(session):
    # The provenance of the code under test, taken before any test runs and
    # before the evidence file exists: in CI that file is written into the
    # checkout, and an untracked file there would mark the run dirty.
    if session.config.getoption("--evidence"):
        session.config.stash[_HEADER] = run_header(evidence=session.config.getoption("--evidence"))


def pytest_sessionfinish(session):
    path = session.config.getoption("--evidence")
    if not path:
        return
    header = session.config.stash.get(_HEADER, None) or run_header(evidence=path)
    # known only now: what the run was asked to select and what it left out
    header["invocation"] = {
        **invocation(session.config),
        "deselected": len(set(session.config.stash.get(_DESELECTED, ()))),
    }
    with open(path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for record in session.config.stash[_RECORDS]:
            f.write(json.dumps(record) + "\n")


# Collection skips: a module that skips itself at import has no test items, so
# the hooks above never see it. Its tests are recorded here, from the file.

REQUIRE_VIRGIL_ENV = "VV_REQUIRE_VIRGIL_IMPORTS"
_IMPORT_SKIP = re.compile(r"could not import '(virgil(?:\.[\w.]+)?)'(?:: (.*))?", re.S)


def pytest_deselected(items):
    config = _CONFIG.get("config")
    if config is not None:
        config.stash[_DESELECTED].extend(item.nodeid for item in items)


def _skip_reason(report):
    longrepr = report.longrepr
    reason = longrepr[2] if isinstance(longrepr, tuple) and len(longrepr) == 3 else str(longrepr)
    return reason.removeprefix("Skipped: ")


def _missing_virgil(reason):
    """True when a module skipped because virgil, or a virgil module, is not
    there. A virgil module that fails because an optional extra is missing
    (``No module named 'casatools'``) is not: that skip is legitimate."""
    found = _IMPORT_SKIP.match(reason)
    if not found:
        return False
    missing = re.search(r"No module named '([^']+)'", found.group(2) or "")
    return missing is None or missing.group(1).split(".")[0] == "virgil"


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    """With ``VV_REQUIRE_VIRGIL_IMPORTS=1`` (set in CI) a test file that skips
    because a ``virgil.*`` module is missing is a collection error, so that a
    renamed or removed virgil module cannot silently remove its tests."""
    outcome = yield
    report = outcome.get_result()
    if (
        report.skipped
        and isinstance(collector, pytest.Module)
        and os.environ.get(REQUIRE_VIRGIL_ENV) == "1"
    ):
        reason = _skip_reason(report)
        if _missing_virgil(reason):
            report.outcome = "failed"
            report.longrepr = (
                f"{collector.nodeid}: {reason}\n"
                f"{REQUIRE_VIRGIL_ENV}=1: a missing virgil module is an error, not a skip. "
                "Its tests would otherwise vanish from the evidence."
            )


def _marker_claim(call):
    """A validates(...) call node as a claim, read from literals only."""
    def literal(node):
        try:
            return ast.literal_eval(node)
        except ValueError:
            return None

    objects = [
        a.value for a in call.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
    ] or ["<computed>"]
    kwargs = {k.arg: literal(k.value) for k in call.keywords if k.arg}
    prop = kwargs.get("property") or ()
    roots = kwargs.get("roots") or ()
    headline = kwargs.get("headline")
    claim = {
        "objects": objects,
        "roots": [roots] if isinstance(roots, str) else list(roots),
        "kind": kwargs.get("kind") or "check",
        "tier": kwargs.get("tier") or "A",
        "via": sorted(set(kwargs.get("via") or ()) - set(objects)),
        "properties": [prop] if isinstance(prop, str) else list(prop),
    }
    if headline is not None:
        claim["headline"] = str(headline)
    return claim


def _validates_calls(node):
    """The validates markers in an assignment value or a decorator."""
    return [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "validates"
    ]


def tests_in_file(path):
    """[(name, line, doc, claims)] for every test function of a file, read
    with the AST because the file could not be imported. Module-level
    ``pytestmark`` claims apply to every test."""
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    module = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
        ):
            module += [_marker_claim(c) for c in _validates_calls(node.value)]
    found = []

    def visit(body, prefix):
        for node in body:
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                visit(node.body, prefix + node.name + "::")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                own = [_marker_claim(c) for d in node.decorator_list for c in _validates_calls(d)]
                doc = ast.get_docstring(node) or ""
                first = " ".join(doc.strip().split("\n\n")[0].split())
                found.append((prefix + node.name, node.lineno, first, module + own))

    visit(tree.body, "")
    return found


def pytest_collectreport(report):
    """One ``skipped`` record, with ``phase: "collection"`` and the reason, for
    each test function of a module that skipped itself at import."""
    if not report.skipped or not report.nodeid.endswith(".py"):
        return
    config = _CONFIG.get("config")
    if config is None:
        return
    reason = _skip_reason(report)
    try:
        tests = tests_in_file(config.rootpath / report.nodeid)
    except (OSError, SyntaxError, ValueError):  # ValueError includes UnicodeDecodeError
        return
    records = config.stash[_RECORDS]
    unmarked = [f"{report.nodeid}::{name}" for name, _, _, claims in tests if not claims]
    if unmarked and config.getoption("--require-validates"):
        # the skipped module has no items, so pytest_collection_modifyitems never sees these
        raise pytest.UsageError(
            "tests without a validates marker (docs/method/index.md):\n  "
            + "\n  ".join(unmarked)
        )
    for name, line, doc, claims in tests:
        for claim in claims:
            records.append(
                {
                    "record": "test",
                    "test": f"{report.nodeid}::{name}",
                    **claim,
                    "outcome": "skipped",
                    "phase": "collection",
                    "reason": reason,
                    "duration_s": 0.0,
                    "metrics": {},
                    "line": line,
                    "doc": doc,
                }
            )
