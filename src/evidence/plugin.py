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
from . import via as _via
from .meta import invocation, run_header

_RECORDS = pytest.StashKey[list]()
_STAMP = pytest.StashKey[dict]()
_HEADER = pytest.StashKey[dict]()
_DESELECTED = pytest.StashKey[list]()
_CONFIG = {}
ACCEPTANCE = os.environ.get("VV_VIA_ACCEPTANCE") == "1"  # one-off (B2): also record the old scanner's via as via_old


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


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item):
    _via.set_item(item.nodeid)
    try:
        yield
    finally:
        _via.set_item(None)


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef, request):
    _via.push_fixture(fixturedef)
    try:
        yield
    finally:
        _via.pop_fixture()


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
    _via.install_recorder()
    _via.install_import_recorder()


# Our reference code: a check that goes through one of these counts only if
# that module is itself verified (fail-closed across all three packages).
# _subprocess is the JSON-over-interpreter launcher; every _subprocess.run
# call resolves its concrete worker module (or emits an unregistered sentinel).
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
_REFERENCE_PACKAGES = ("crosscheck", "external_bridge", "virgil_bridge")
_PLUMBING = frozenset({"external_bridge._subprocess"})
_UNRESOLVED_WORKER = "external_bridge.<unresolved_worker>"
_UNRESOLVED_IMPORT = "crosscheck.<unresolved_dynamic_import>"
_SAFE_SP_ATTRS = frozenset({"run", "available", "python", "version", "venv_root", "_env", "ROOT"})
_SKIP_LOCAL_TOPS = frozenset({*_REFERENCE_PACKAGES, "virgil", "drpangloss", "jaxbessel", "evidence"})


def _keep_reference(mod):
    return mod not in _PLUMBING


def _is_all_dunder_assign(stmt):
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
        return stmt.targets[0].id == "__all__"
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id == "__all__"
    return False


@functools.lru_cache(maxsize=None)
def _has_executable_defs_cached(init_py_str):
    init_py = pathlib.Path(init_py_str)
    if not init_py.is_file():
        return False
    try:
        tree = ast.parse(init_py.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return True
    for i, stmt in enumerate(tree.body):
        if i == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue
        if isinstance(stmt, (ast.Pass, ast.Import, ast.ImportFrom)) or _is_all_dunder_assign(stmt):
            continue
        return True
    return False


def _has_executable_defs(init_py):
    """True when ``__init__.py`` defines functions, classes, or non-``__all__`` bindings of its own."""
    return _has_executable_defs_cached(str(pathlib.Path(init_py).resolve()))


_has_executable_defs.cache_clear = _has_executable_defs_cached.cache_clear

_IGNORED_REPO_DIRS = frozenset({
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".external",
    ".claude",
    "build",
    "site",
    "virgil-src",
    "virgil-ci",
})


def _is_ignored_repo_part(part):
    return part.startswith(".venv") or part.endswith(".egg-info") or part in _IGNORED_REPO_DIRS


@functools.lru_cache(maxsize=1)
def _repo_py_suffixes():
    root = _REPO_ROOT.resolve()
    out = []
    for p in root.rglob("*.py"):
        rel = p.relative_to(root)
        if any(_is_ignored_repo_part(part) for part in rel.parts):
            continue
        out.append(rel.as_posix())
    return tuple(out)


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
    except (OSError, SyntaxError, UnicodeDecodeError):
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


def _is_under_src(path):
    try:
        path.resolve().relative_to(_SRC_ROOT.resolve())
        return True
    except ValueError:
        return False


def _local_search_dirs(caller_path):
    """Local directories where non-package repo helpers (``benchmarks/``, ``scripts/``, ``tests/``) live."""
    try:
        resolved = caller_path.resolve()
    except OSError:
        resolved = caller_path
    try:
        resolved.relative_to(_REPO_ROOT.resolve())
        anchor = _REPO_ROOT.resolve()
    except ValueError:
        parent = resolved.parent
        anchor = parent.parent if parent.name in ("tests", "scripts") else parent

    candidates = [
        resolved.parent,
        anchor,
        anchor / "scripts",
        anchor / "tests",
        anchor / "benchmarks",
    ]
    bench = anchor / "benchmarks"
    if bench.is_dir():
        candidates.extend(
            d for d in sorted(bench.iterdir())
            if d.is_dir() and not d.name.startswith(".")
        )
    out = []
    seen_dirs = set()
    for d in candidates:
        if d.is_dir() and not _is_under_src(d) and d not in seen_dirs:
            seen_dirs.add(d)
            out.append(d)
    return tuple(out)


def _ref_target_file(parts):
    if len(parts) >= 2:
        py = _SRC_ROOT.joinpath(*parts[:-1], f"{parts[-1]}.py")
        if py.is_file():
            return py
    init_py = _SRC_ROOT.joinpath(*parts, "__init__.py")
    if init_py.is_file():
        return init_py
    return None


def _enclosing_local_package_inits(dir_path):
    """Outermost-first ``__init__.py`` files of local helper packages enclosing ``dir_path``."""
    repo_root = _REPO_ROOT.resolve()
    inits = []
    cur = dir_path
    while True:
        try:
            cur_res = cur.resolve()
        except OSError:
            cur_res = cur
        if cur_res == repo_root:
            break
        init_py = cur / "__init__.py"
        if not init_py.is_file() or _is_under_src(init_py):
            break
        inits.append(init_py)
        if cur.parent == cur:
            break
        cur = cur.parent
    inits.reverse()
    return tuple(inits)


def _resolve_local_helper_with_inits(parts, search_dirs):
    """Return ``(target_file, enclosing_init_files)`` for a local helper import outside ``src/``."""
    if not parts:
        for base in search_dirs:
            cand = base / "__init__.py"
            if cand.is_file() and not _is_under_src(cand):
                enc = tuple(p for p in _enclosing_local_package_inits(base.parent) if p != cand)
                return cand, enc
        return None, ()
    if parts[0] in _SKIP_LOCAL_TOPS:
        return None, ()
    for base in search_dirs:
        for cand in (
            base.joinpath(*parts[:-1], f"{parts[-1]}.py"),
            base.joinpath(*parts, "__init__.py"),
        ):
            if cand.is_file() and not _is_under_src(cand):
                seen_inits = []
                for p in _enclosing_local_package_inits(base):
                    if p != cand and p not in seen_inits:
                        seen_inits.append(p)
                for k in range(1, len(parts)):
                    init_k = base.joinpath(*parts[:k], "__init__.py")
                    if init_k.is_file() and not _is_under_src(init_k) and init_k != cand and init_k not in seen_inits:
                        seen_inits.append(init_k)
                return cand, tuple(seen_inits)
    return None, ()


def _resolve_local_helper(parts, search_dirs):
    cand, _ = _resolve_local_helper_with_inits(parts, search_dirs)
    return cand


def _call_arg_expr(call, index, keyword):
    if any(isinstance(a, ast.Starred) for a in call.args) or any(kw.arg is None for kw in call.keywords):
        return None
    if index is not None and index < len(call.args):
        return call.args[index]
    for kw in call.keywords:
        if kw.arg == keyword:
            return kw.value
    return None


def _attr_chain(expr):
    parts = []
    cur = expr
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        parts.reverse()
        return tuple(parts)
    return None


def _has_non_import_binding(tree, target_names):
    if not target_names:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in target_names:
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in target_names:
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                if a.arg in target_names:
                    return True
            if node.args.vararg and node.args.vararg.arg in target_names:
                return True
            if node.args.kwarg and node.args.kwarg.arg in target_names:
                return True
        if isinstance(node, ast.ExceptHandler) and node.name in target_names:
            return True
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name in target_names:
            return True
        if isinstance(node, ast.MatchMapping) and node.rest in target_names:
            return True
    return False


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


_EMPTY_SP_EXPORT = {
    "wrappers": {},
    "sp_chains": frozenset(),
    "eb_chains": frozenset(),
    "wildcard": False,
    "uncalled_names": frozenset(),
}
_WILDCARD_SP_EXPORT = {
    "wrappers": {},
    "sp_chains": frozenset(),
    "eb_chains": frozenset(),
    "wildcard": True,
    "uncalled_names": frozenset(),
}


def _uniquely_bound_top_names(root_tree, module_counts, global_names):
    """Module-scope names whose binding is provably unique (including ``import pkg; import pkg.sub``)."""
    if module_counts["*"] > 0:
        return frozenset()
    top_imp_counts = collections.Counter()
    top_imp_sources = collections.defaultdict(set)
    for stmt in root_tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                bname = alias.asname or alias.name.split(".")[0]
                top_imp_counts[bname] += 1
                top_imp_sources[bname].add(("import", alias.name if alias.asname else alias.name.split(".")[0]))
        elif isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                if alias.name != "*":
                    bname = alias.asname or alias.name
                    top_imp_counts[bname] += 1
                    top_imp_sources[bname].add(("from", stmt.level, stmt.module, alias.name))
    return frozenset(
        name
        for name, cnt in module_counts.items()
        if name != "*"
        and name not in global_names
        and (cnt == 1 or (cnt > 1 and top_imp_counts[name] == cnt and len(top_imp_sources[name]) == 1))
    )


def _subprocess_workers(
    scope_tree,
    sp_aliases,
    run_aliases,
    eb_aliases,
    wildcard_sp,
    full_tree=None,
    *,
    imported_sp_wrappers=None,
    mod_sp_wrappers=None,
    mod_sp_chains=frozenset(),
    mod_eb_chains=frozenset(),
    allow_uncalled_wrappers=frozenset(),
    is_full_module=True,
    return_wrappers_only=False,
):
    """Worker scripts invoked through ``external_bridge._subprocess.run`` or local/exported wrappers.

    Any ``_subprocess.run`` call or dynamic alias usage that cannot be statically resolved
    to a concrete worker module emits ``_UNRESOLVED_WORKER`` so ``trust.py`` and
    ``test_trust_graph.py`` fail closed.
    """
    imported_sp_wrappers = imported_sp_wrappers or {}
    mod_sp_wrappers = mod_sp_wrappers or {}
    mod_sp_chains = frozenset(mod_sp_chains)
    mod_eb_chains = frozenset(mod_eb_chains)

    if wildcard_sp:
        return _WILDCARD_SP_EXPORT if return_wrappers_only else {_UNRESOLVED_WORKER}
    if (
        not sp_aliases
        and not run_aliases
        and not eb_aliases
        and not imported_sp_wrappers
        and not mod_sp_wrappers
        and not mod_sp_chains
        and not mod_eb_chains
    ):
        return _EMPTY_SP_EXPORT if return_wrappers_only else set()

    all_sp_chains = (
        {(s,) for s in sp_aliases}
        | {(e, "_subprocess") for e in eb_aliases}
        | set(mod_sp_chains)
        | {(*ec, "_subprocess") for ec in mod_eb_chains}
    )
    all_eb_chains = {(e,) for e in eb_aliases} | set(mod_eb_chains)
    mod_sp_prefixes = {
        chain[:k]
        for chain in (*mod_sp_wrappers, *mod_sp_chains, *mod_eb_chains)
        for k in range(1, len(chain))
    }

    root_tree = full_tree if full_tree is not None else scope_tree
    protected_import_names = (
        set(sp_aliases)
        | set(run_aliases)
        | set(eb_aliases)
        | set(imported_sp_wrappers)
        | {p[0] for p in mod_sp_prefixes}
        | {c[0] for c in mod_sp_wrappers}
        | {c[0] for c in mod_sp_chains}
        | {c[0] for c in mod_eb_chains}
    )
    if protected_import_names and (
        _has_non_import_binding(scope_tree, protected_import_names)
        or _has_non_import_binding(ast.Module(body=list(root_tree.body), type_ignores=[]), protected_import_names)
    ):
        if return_wrappers_only:
            return _WILDCARD_SP_EXPORT
        return {_UNRESOLVED_WORKER}

    module_counts = collections.Counter()
    _collect_scope_bindings(root_tree, module_counts, is_scope_root=True)
    global_names = set()
    for n in ast.walk(root_tree):
        if isinstance(n, ast.Global):
            for gname in n.names:
                module_counts[gname] += 1
                global_names.add(gname)

    const_strings = {}
    if module_counts["*"] == 0:
        for stmt in root_tree.body:
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

    def _is_sp_module_expr(expr, shadowed=frozenset()):
        chain = _attr_chain(expr)
        return chain is not None and chain in all_sp_chains and chain[0] not in shadowed

    def _is_sp_run(func, shadowed=frozenset()):
        return (
            (isinstance(func, ast.Attribute) and func.attr == "run" and _is_sp_module_expr(func.value, shadowed))
            or (isinstance(func, ast.Name) and func.id in run_aliases and func.id not in shadowed)
        )

    fn_nodes = [n for n in ast.walk(scope_tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
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

    top_fns = [
        stmt
        for stmt in scope_tree.body
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not stmt.decorator_list
        and module_counts["*"] == 0
        and module_counts[stmt.name] == 1
        and stmt.name not in global_names
    ]
    wrappers = {}
    wrapper_param_by_id = {}
    changed = True
    while changed:
        changed = False
        for fn in top_fns:
            if fn.name in wrappers:
                continue
            pos_params = [a.arg for a in (*fn.args.posonlyargs, *fn.args.args)]
            kw_params = {a.arg for a in fn.args.kwonlyargs}
            all_params = set(pos_params) | kw_params
            fc = fn_counts[id(fn)]
            fbound = fn_bound[id(fn)]
            for sub in ast.walk(fn):
                if not isinstance(sub, ast.Call):
                    continue
                wexpr = None
                chain = _attr_chain(sub.func)
                if _is_sp_run(sub.func, fbound):
                    wexpr = _call_arg_expr(sub, 1, "worker")
                elif isinstance(sub.func, ast.Name) and sub.func.id in wrappers and sub.func.id not in fbound:
                    idx, pname, _ = wrappers[sub.func.id]
                    wexpr = _call_arg_expr(sub, idx, pname)
                elif isinstance(sub.func, ast.Name) and sub.func.id in imported_sp_wrappers and sub.func.id not in fbound:
                    idx, pname = imported_sp_wrappers[sub.func.id]
                    wexpr = _call_arg_expr(sub, idx, pname)
                elif chain in mod_sp_wrappers and chain[0] not in fbound:
                    idx, pname = mod_sp_wrappers[chain]
                    wexpr = _call_arg_expr(sub, idx, pname)
                if (
                    isinstance(wexpr, ast.Name)
                    and wexpr.id in all_params
                    and fc[wexpr.id] == 1
                    and wexpr.id not in global_names
                ):
                    p_idx = pos_params.index(wexpr.id) if wexpr.id in pos_params else None
                    wrappers[fn.name] = (p_idx, wexpr.id, id(fn))
                    wrapper_param_by_id[id(fn)] = wexpr.id
                    changed = True
                    break

    if return_wrappers_only:
        if module_counts["*"] > 0:
            return _WILDCARD_SP_EXPORT
        top_unique = _uniquely_bound_top_names(root_tree, module_counts, global_names)
        exp_wrappers = {(fn_name,): (idx, pname) for fn_name, (idx, pname, _) in wrappers.items()}
        for rname in run_aliases:
            if rname in top_unique:
                exp_wrappers[(rname,)] = (1, "worker")
        for wname, sig in imported_sp_wrappers.items():
            if wname in top_unique:
                exp_wrappers[(wname,)] = sig
        for wchain, sig in mod_sp_wrappers.items():
            if wchain[0] in top_unique:
                exp_wrappers[wchain] = sig
        exp_sp_chains = (
            {(sname,) for sname in sp_aliases if sname in top_unique}
            | {(ename, "_subprocess") for ename in eb_aliases if ename in top_unique}
            | {schain for schain in mod_sp_chains if schain[0] in top_unique}
            | {(*echain, "_subprocess") for echain in mod_eb_chains if echain[0] in top_unique}
        )
        exp_eb_chains = (
            {(ename,) for ename in eb_aliases if ename in top_unique}
            | {echain for echain in mod_eb_chains if echain[0] in top_unique}
        )
        uncalled = frozenset({c[0] for c in (*exp_wrappers, *exp_sp_chains, *exp_eb_chains) if len(c) == 1})
        return {
            "wrappers": exp_wrappers,
            "sp_chains": frozenset(exp_sp_chains),
            "eb_chains": frozenset(exp_eb_chains),
            "wildcard": False,
            "uncalled_names": uncalled,
        }

    workers = set()
    called_wrappers = set()
    called_imported_wrappers = set()
    called_run_aliases = set()
    referenced_sp_roots = set()
    direct_call_funcs = set()
    attr_parents = {}

    for parent in ast.walk(scope_tree):
        if isinstance(parent, ast.Call):
            direct_call_funcs.add(id(parent.func))
        elif isinstance(parent, ast.Attribute):
            attr_parents[id(parent.value)] = parent

    def _scan_node(node, enclosing_wrapper_param, shadowed):
        if isinstance(node, ast.Call):
            handled = False
            wexpr = None
            chain = _attr_chain(node.func)
            if _is_sp_run(node.func, shadowed):
                handled = True
                if isinstance(node.func, ast.Name) and node.func.id in run_aliases:
                    called_run_aliases.add(node.func.id)
                wexpr = _call_arg_expr(node, 1, "worker")
            elif isinstance(node.func, ast.Name) and node.func.id in wrappers and node.func.id not in shadowed:
                handled = True
                called_wrappers.add(node.func.id)
                idx, pname, _ = wrappers[node.func.id]
                wexpr = _call_arg_expr(node, idx, pname)
            elif isinstance(node.func, ast.Name) and node.func.id in imported_sp_wrappers and node.func.id not in shadowed:
                handled = True
                called_imported_wrappers.add(node.func.id)
                idx, pname = imported_sp_wrappers[node.func.id]
                wexpr = _call_arg_expr(node, idx, pname)
            elif chain in mod_sp_wrappers and chain[0] not in shadowed:
                handled = True
                idx, pname = mod_sp_wrappers[chain]
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
            referenced_sp_roots.add(node.id)
            if node.id in run_aliases and id(node) not in direct_call_funcs:
                workers.add(_UNRESOLVED_WORKER)
            elif (node.id,) in all_sp_chains:
                parent_attr = attr_parents.get(id(node))
                if parent_attr is None or parent_attr.attr not in _SAFE_SP_ATTRS:
                    workers.add(_UNRESOLVED_WORKER)
                elif parent_attr.attr == "run" and id(parent_attr) not in direct_call_funcs:
                    workers.add(_UNRESOLVED_WORKER)
            elif node.id in wrappers and node.id not in shadowed and id(node) not in direct_call_funcs:
                workers.add(_UNRESOLVED_WORKER)
            elif node.id in imported_sp_wrappers and id(node) not in direct_call_funcs:
                workers.add(_UNRESOLVED_WORKER)
            elif ((node.id,) in mod_sp_prefixes or (node.id,) in all_eb_chains) and id(node) not in attr_parents:
                workers.add(_UNRESOLVED_WORKER)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            chain = _attr_chain(node)
            if chain in all_sp_chains:
                parent_attr = attr_parents.get(id(node))
                if parent_attr is None or parent_attr.attr not in _SAFE_SP_ATTRS:
                    workers.add(_UNRESOLVED_WORKER)
                elif parent_attr.attr == "run" and id(parent_attr) not in direct_call_funcs:
                    workers.add(_UNRESOLVED_WORKER)
            elif chain in mod_sp_wrappers and id(node) not in direct_call_funcs:
                workers.add(_UNRESOLVED_WORKER)
            elif (chain in mod_sp_prefixes or chain in all_eb_chains) and id(node) not in attr_parents:
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
            wparam = wrapper_param_by_id.get(id(node))
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

    _scan_node(scope_tree, None, frozenset())
    if set(wrappers) - called_wrappers - set(allow_uncalled_wrappers):
        workers.add(_UNRESOLVED_WORKER)
    if is_full_module:
        if set(imported_sp_wrappers) - called_imported_wrappers - set(allow_uncalled_wrappers):
            workers.add(_UNRESOLVED_WORKER)
        if set(run_aliases) - called_run_aliases - set(allow_uncalled_wrappers):
            workers.add(_UNRESOLVED_WORKER)
        if (set(sp_aliases) | set(eb_aliases)) - referenced_sp_roots - set(allow_uncalled_wrappers):
            workers.add(_UNRESOLVED_WORKER)
    else:
        root_called = {
            n.func.id for n in ast.walk(root_tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        root_loaded = {
            n.id for n in ast.walk(root_tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        }
        if (set(imported_sp_wrappers) | set(run_aliases)) - root_called - set(allow_uncalled_wrappers):
            workers.add(_UNRESOLVED_WORKER)
        if (set(sp_aliases) | set(eb_aliases)) - root_loaded - set(allow_uncalled_wrappers):
            workers.add(_UNRESOLVED_WORKER)
    return workers


def _single_assign_exprs(stmts, counts, global_names):
    if counts.get("*", 0) > 0:
        return {}
    out = {}
    for stmt in stmts:
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name) and counts[t.id] == 1 and t.id not in global_names:
                    out[t.id] = stmt.value
        elif (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.value is not None
            and counts[stmt.target.id] == 1
            and stmt.target.id not in global_names
        ):
            out[stmt.target.id] = stmt.value
    return out


def _free_names(expr):
    return {n.id for n in ast.walk(expr) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _import_call_kwargs_ok(node, kind):
    if any(isinstance(a, ast.Starred) for a in node.args) or any(kw.arg is None for kw in node.keywords):
        return False
    if kind == "__import__":
        lvl_arg = _call_arg_expr(node, 4, "level")
        if lvl_arg is not None and not (isinstance(lvl_arg, ast.Constant) and lvl_arg.value == 0):
            return False
    return True


def _is_provably_external_py_suffix(suf, resolved_file):
    if not suf or not suf.startswith("/") or not suf.endswith(".py"):
        return False
    rel = suf.lstrip("/")
    parts = rel.split("/")
    if len(parts) < 2 or any(p in {"", ".", ".."} for p in parts):
        return False
    if any(rp == rel or rp.endswith("/" + rel) for rp in _repo_py_suffixes()):
        return False
    try:
        resolved_file.relative_to(_REPO_ROOT.resolve())
        return True
    except ValueError:
        base = resolved_file.parent.parent if resolved_file.parent.name in {"tests", "scripts", "src"} else resolved_file.parent
        for cand in base.rglob("*.py"):
            cand_rel_path = cand.relative_to(base)
            if any(_is_ignored_repo_part(part) for part in cand_rel_path.parts):
                continue
            cand_rel = cand_rel_path.as_posix()
            if cand_rel == rel or cand_rel.endswith("/" + rel):
                return False
        return True


_EMPTY_DYN_EXPORT = {
    "call_kinds": {},
    "imp_wrappers": {},
    "spec_wrappers": {},
    "importlib_chains": frozenset(),
    "importlib_util_chains": frozenset(),
    "pytest_chains": frozenset(),
    "builtins_chains": frozenset(),
    "wildcard": False,
    "uncalled_names": frozenset(),
}
_WILDCARD_DYN_EXPORT = {
    **_EMPTY_DYN_EXPORT,
    "wildcard": True,
}


def _scan_call_imports(
    tree,
    resolved,
    pkg_parts,
    local_dirs,
    seen,
    expand_pkg_parts,
    *,
    full_tree=None,
    importlib_aliases,
    importlib_util_aliases,
    import_module_aliases,
    dunder_import_aliases=frozenset(),
    spec_from_file_aliases,
    pytest_aliases,
    importorskip_aliases,
    builtins_aliases,
    pathlib_aliases,
    path_cls_aliases,
    wildcard_import,
    imported_call_kinds=None,
    mod_call_kinds=None,
    imported_imp_wrappers=None,
    mod_imp_wrappers=None,
    imported_spec_wrappers=None,
    mod_spec_wrappers=None,
    mod_importlib_chains=frozenset(),
    mod_importlib_util_chains=frozenset(),
    mod_pytest_chains=frozenset(),
    mod_builtins_chains=frozenset(),
    wildcard_dyn=False,
    allow_uncalled_imp=frozenset(),
    is_full_module=True,
    return_exports_only=False,
):
    """Resolve ``importorskip``, ``import_module``, ``__import__``, and ``spec_from_file_location`` calls.

    Fails closed by emitting ``_UNRESOLVED_IMPORT`` (or ``_UNRESOLVED_WORKER`` when
    ``external_bridge._subprocess`` is dynamically loaded) whenever a target cannot
    be statically resolved or provably ruled out as external to the repository.
    """
    if wildcard_dyn:
        return _WILDCARD_DYN_EXPORT if return_exports_only else {_UNRESOLVED_IMPORT}

    extra = set()
    root_tree = full_tree if full_tree is not None else tree
    imported_call_kinds = imported_call_kinds or {}
    mod_call_kinds = mod_call_kinds or {}
    imported_imp_wrappers = imported_imp_wrappers or {}
    mod_imp_wrappers = mod_imp_wrappers or {}
    imported_spec_wrappers = imported_spec_wrappers or {}
    mod_spec_wrappers = mod_spec_wrappers or {}
    mod_importlib_chains = frozenset(mod_importlib_chains)
    mod_importlib_util_chains = frozenset(mod_importlib_util_chains)
    mod_pytest_chains = frozenset(mod_pytest_chains)
    mod_builtins_chains = frozenset(mod_builtins_chains)
    dunder_import_aliases = set(dunder_import_aliases)

    def _is_path_ctor(func):
        return (
            (isinstance(func, ast.Name) and func.id in (path_cls_aliases | {"Path"}))
            or (
                isinstance(func, ast.Attribute)
                and func.attr == "Path"
                and isinstance(func.value, ast.Name)
                and func.value.id in (pathlib_aliases | {"pathlib"})
            )
        )

    importlib_aliases = set(importlib_aliases)
    pytest_aliases = set(pytest_aliases)
    builtins_aliases = set(builtins_aliases)
    pathlib_aliases = set(pathlib_aliases)
    for _n in ast.walk(tree):
        if isinstance(_n, ast.Import):
            for _alias in _n.names:
                if _alias.asname is None:
                    _top = _alias.name.split(".", 1)[0]
                    if _top == "importlib":
                        importlib_aliases.add("importlib")
                    elif _top == "pytest":
                        pytest_aliases.add("pytest")
                    elif _top == "builtins":
                        builtins_aliases.add("builtins")
                    elif _top == "pathlib":
                        pathlib_aliases.add("pathlib")

    all_importlib_chains = {(a,) for a in importlib_aliases} | set(mod_importlib_chains)
    all_importlib_util_chains = (
        {(a,) for a in importlib_util_aliases}
        | {(*c, "util") for c in all_importlib_chains}
        | set(mod_importlib_util_chains)
    )
    all_pytest_chains = {(a,) for a in pytest_aliases} | set(mod_pytest_chains)
    all_builtins_chains = {(a,) for a in builtins_aliases} | set(mod_builtins_chains)
    mod_dyn_prefixes = {
        chain[:k]
        for chain in (
            *mod_call_kinds,
            *mod_imp_wrappers,
            *mod_spec_wrappers,
            *mod_importlib_chains,
            *mod_importlib_util_chains,
            *mod_builtins_chains,
        )
        for k in range(1, len(chain))
    }

    protected_dyn_names = (
        set(import_module_aliases)
        | set(dunder_import_aliases)
        | set(importorskip_aliases)
        | set(spec_from_file_aliases)
        | set(imported_call_kinds)
        | set(imported_imp_wrappers)
        | set(imported_spec_wrappers)
        | {p[0] for p in mod_dyn_prefixes}
        | {c[0] for c in mod_call_kinds}
        | {c[0] for c in mod_imp_wrappers}
        | {c[0] for c in mod_spec_wrappers}
        | {c[0] for c in mod_importlib_chains}
        | {c[0] for c in mod_importlib_util_chains}
        | {c[0] for c in mod_builtins_chains}
    )
    if protected_dyn_names and (
        _has_non_import_binding(tree, protected_dyn_names)
        or _has_non_import_binding(ast.Module(body=list(root_tree.body), type_ignores=[]), protected_dyn_names)
    ):
        return _WILDCARD_DYN_EXPORT if return_exports_only else {_UNRESOLVED_IMPORT}

    def _call_kind(func, shadowed=frozenset()):
        if isinstance(func, ast.Name):
            if func.id in shadowed:
                return None
            if func.id == "__import__" or func.id in dunder_import_aliases:
                return "__import__"
            if func.id in import_module_aliases:
                return "import_module"
            if func.id in importorskip_aliases:
                return "importorskip"
            if func.id in spec_from_file_aliases:
                return "spec_from_file_location"
            if func.id in imported_call_kinds:
                return imported_call_kinds[func.id]
            return None
        if isinstance(func, ast.Attribute):
            chain = _attr_chain(func)
            if chain is None or chain[0] in shadowed:
                return None
            if chain in mod_call_kinds:
                return mod_call_kinds[chain]
            prefix, attr = chain[:-1], chain[-1]
            if attr == "__import__" and prefix in all_builtins_chains:
                return "__import__"
            if attr == "import_module" and prefix in all_importlib_chains:
                return "import_module"
            if attr == "importorskip" and prefix in all_pytest_chains:
                return "importorskip"
            if attr == "spec_from_file_location" and prefix in all_importlib_util_chains:
                return "spec_from_file_location"
        return None

    has_any_call_or_alias = bool(
        import_module_aliases
        | dunder_import_aliases
        | importorskip_aliases
        | spec_from_file_aliases
        | importlib_aliases
        | importlib_util_aliases
        | pytest_aliases
        | builtins_aliases
        | set(imported_call_kinds)
        | set(mod_call_kinds)
        | set(imported_imp_wrappers)
        | set(mod_imp_wrappers)
        | set(imported_spec_wrappers)
        | set(mod_spec_wrappers)
        | set(mod_importlib_chains)
        | set(mod_importlib_util_chains)
        | set(mod_pytest_chains)
        | set(mod_builtins_chains)
    )
    has_dunder = any(
        isinstance(n, ast.Name) and n.id == "__import__"
        for n in ast.walk(tree)
    )
    if not has_any_call_or_alias and not has_dunder:
        return _EMPTY_DYN_EXPORT if return_exports_only else extra
    if has_dunder and (
        _has_non_import_binding(tree, {"__import__"})
        or _has_non_import_binding(ast.Module(body=list(root_tree.body), type_ignores=[]), {"__import__"})
    ):
        return _WILDCARD_DYN_EXPORT if return_exports_only else {_UNRESOLVED_IMPORT}

    module_counts = collections.Counter()
    _collect_scope_bindings(root_tree, module_counts, is_scope_root=True)
    if wildcard_import:
        module_counts["*"] += 1
    global_names = set()
    for n in ast.walk(root_tree):
        if isinstance(n, ast.Global):
            for gname in n.names:
                module_counts[gname] += 1
                global_names.add(gname)

    mod_expr_map = _single_assign_exprs(root_tree.body, module_counts, global_names)

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

    def _eval_expr(expr, active_map, param_bindings=None, depth=0):
        if depth > 12 or expr is None:
            return None
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return (expr.value,)
        if isinstance(expr, ast.Name):
            if param_bindings and expr.id in param_bindings:
                return param_bindings[expr.id]
            if expr.id == "__file__" and "__file__" not in active_map:
                return (str(resolved),)
            if expr.id in active_map:
                return _eval_expr(active_map[expr.id], active_map, param_bindings, depth + 1)
            return None
        if isinstance(expr, ast.Call):
            if _is_path_ctor(expr.func) and not expr.keywords:
                if len(expr.args) == 0:
                    return (pathlib.Path("."),)
                if len(expr.args) == 1:
                    vals = _eval_expr(expr.args[0], active_map, param_bindings, depth + 1)
                    return tuple(pathlib.Path(v) for v in vals) if vals else None
            if (
                isinstance(expr.func, ast.Name)
                and expr.func.id in {"sorted", "list", "tuple"}
                and len(expr.args) == 1
                and not expr.keywords
            ):
                return _eval_expr(expr.args[0], active_map, param_bindings, depth + 1)
            if isinstance(expr.func, ast.Attribute) and not expr.keywords:
                if expr.func.attr == "resolve" and len(expr.args) == 0:
                    vals = _eval_expr(expr.func.value, active_map, param_bindings, depth + 1)
                    return tuple(pathlib.Path(v).resolve() for v in vals) if vals else None
                if expr.func.attr == "glob" and len(expr.args) == 1:
                    b_vals = _eval_expr(expr.func.value, active_map, param_bindings, depth + 1)
                    p_vals = _eval_expr(expr.args[0], active_map, param_bindings, depth + 1)
                    if b_vals and p_vals and all(isinstance(p, str) for p in p_vals):
                        matched = []
                        for b in b_vals:
                            bp = pathlib.Path(b)
                            if not bp.is_dir():
                                return None
                            for pat in p_vals:
                                matched.extend(sorted(bp.glob(pat)))
                        return tuple(matched) if matched else None
            return None
        if isinstance(expr, ast.Attribute):
            vals = _eval_expr(expr.value, active_map, param_bindings, depth + 1)
            if not vals:
                return None
            if expr.attr == "parent":
                return tuple(pathlib.Path(v).parent for v in vals)
            if expr.attr == "stem":
                return tuple(pathlib.Path(v).stem for v in vals)
            if expr.attr == "name":
                return tuple(pathlib.Path(v).name for v in vals)
            return None
        if (
            isinstance(expr, ast.Subscript)
            and isinstance(expr.value, ast.Attribute)
            and expr.value.attr == "parents"
            and isinstance(expr.slice, ast.Constant)
            and isinstance(expr.slice.value, int)
            and not isinstance(expr.slice.value, bool)
            and expr.slice.value >= 0
        ):
            vals = _eval_expr(expr.value.value, active_map, param_bindings, depth + 1)
            if not vals:
                return None
            idx = expr.slice.value
            out = []
            for v in vals:
                pv = pathlib.Path(v)
                if idx >= len(pv.parents):
                    return None
                out.append(pv.parents[idx])
            return tuple(out)
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Div):
            l_vals = _eval_expr(expr.left, active_map, param_bindings, depth + 1)
            r_vals = _eval_expr(expr.right, active_map, param_bindings, depth + 1)
            if l_vals and r_vals:
                return tuple(pathlib.Path(lv) / str(rv) for lv in l_vals for rv in r_vals)
            return None
        if isinstance(expr, ast.JoinedStr):
            acc = [""]
            for part in expr.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    acc = [a + part.value for a in acc]
                elif isinstance(part, ast.FormattedValue) and part.conversion == -1 and part.format_spec is None:
                    p_vals = _eval_expr(part.value, active_map, param_bindings, depth + 1)
                    if not p_vals:
                        return None
                    acc = [a + str(pv) for a in acc for pv in p_vals]
                else:
                    return None
            return tuple(acc)
        if isinstance(expr, (ast.List, ast.Tuple)):
            out = []
            for elt in expr.elts:
                ev = _eval_expr(elt, active_map, param_bindings, depth + 1)
                if not ev:
                    return None
                out.extend(ev)
            return tuple(out) if out else None
        return None

    def _joinedstr_const_prefix(expr, active_map, depth=0):
        if depth > 8:
            return ""
        if isinstance(expr, ast.Name) and expr.id in active_map:
            return _joinedstr_const_prefix(active_map[expr.id], active_map, depth + 1)
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            l_vals = _eval_expr(expr.left, active_map, None, depth + 1)
            if l_vals and len(l_vals) == 1 and isinstance(l_vals[0], str):
                return l_vals[0] + _joinedstr_const_prefix(expr.right, active_map, depth + 1)
            return _joinedstr_const_prefix(expr.left, active_map, depth + 1)
        if not isinstance(expr, ast.JoinedStr):
            return ""
        pref = ""
        for part in expr.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                pref += part.value
            elif isinstance(part, ast.FormattedValue) and part.conversion == -1 and part.format_spec is None:
                vals = _eval_expr(part.value, active_map, None, depth + 1)
                if vals and len(vals) == 1:
                    pref += str(vals[0])
                else:
                    pref += _joinedstr_const_prefix(part.value, active_map, depth + 1)
                    break
            else:
                break
        return pref

    def _joinedstr_const_suffix(expr, active_map, depth=0):
        if depth > 8:
            return ""
        if isinstance(expr, ast.Name) and expr.id in active_map:
            return _joinedstr_const_suffix(active_map[expr.id], active_map, depth + 1)
        if not isinstance(expr, ast.JoinedStr):
            return ""
        suf = ""
        for part in expr.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                suf += part.value
            elif isinstance(part, ast.FormattedValue) and part.conversion == -1 and part.format_spec is None:
                vals = _eval_expr(part.value, active_map, None, depth + 1)
                if vals and len(vals) == 1:
                    suf += str(vals[0])
                else:
                    suf = _joinedstr_const_suffix(part.value, active_map, depth + 1)
            else:
                suf = ""
        return suf

    def _unwrap_seq_expr(expr, active_map, depth=0):
        if depth > 8 or expr is None:
            return None
        if isinstance(expr, ast.Name) and expr.id in active_map:
            return _unwrap_seq_expr(active_map[expr.id], active_map, depth + 1)
        if (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Name)
            and expr.func.id in {"sorted", "list", "tuple"}
            and len(expr.args) == 1
            and not expr.keywords
        ):
            return _unwrap_seq_expr(expr.args[0], active_map, depth + 1)
        if isinstance(expr, (ast.List, ast.Tuple)):
            return expr.elts
        return None

    loaded_names = {
        sub.id
        for sub in ast.walk(root_tree)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load)
    }
    has_ns_reflection = bool(loaded_names & {"globals", "locals", "vars"})
    param_eligible_fn_ids = {
        id(stmt)
        for stmt in tree.body
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not has_ns_reflection
        and module_counts["*"] == 0
        and module_counts[stmt.name] == 1
        and stmt.name not in global_names
        and stmt.name not in loaded_names
    }

    def _is_pytest_mark_attr(expr):
        return (
            isinstance(expr, ast.Attribute)
            and isinstance(expr.value, ast.Attribute)
            and expr.value.attr == "mark"
            and _attr_chain(expr.value.value) in all_pytest_chains
        )

    def _parametrize_bindings(fn, outer_map):
        if (
            not all_pytest_chains
            or id(fn) not in param_eligible_fn_ids
            or "*" in fn_counts[id(fn)]
        ):
            return {}
        fc = fn_counts[id(fn)]
        bindings = {}
        poisoned = set()
        for dec in fn.decorator_list:
            if _is_pytest_mark_attr(dec):
                if dec.attr == "parametrize":
                    return {}
                continue
            if not (isinstance(dec, ast.Call) and _is_pytest_mark_attr(dec.func)):
                return {}
            if dec.func.attr != "parametrize":
                continue
            if any(isinstance(a, ast.Starred) for a in dec.args) or any(
                kw.arg is None for kw in dec.keywords
            ):
                return {}
            argnames_node = _call_arg_expr(dec, 0, "argnames")
            argvals_node = _call_arg_expr(dec, 1, "argvalues")
            if argnames_node is None or argvals_node is None:
                return {}
            if isinstance(argnames_node, ast.Constant) and isinstance(argnames_node.value, str):
                pnames = [s.strip() for s in argnames_node.value.split(",") if s.strip()]
            elif isinstance(argnames_node, (ast.List, ast.Tuple)) and all(
                isinstance(e, ast.Constant) and isinstance(e.value, str) for e in argnames_node.elts
            ):
                pnames = [e.value.strip() for e in argnames_node.elts if e.value.strip()]
            else:
                return {}
            if not pnames or len(set(pnames)) != len(pnames):
                return {}
            has_indirect = any(
                kw.arg == "indirect"
                and not (isinstance(kw.value, ast.Constant) and kw.value.value in (False, None))
                for kw in dec.keywords
            )
            if has_indirect:
                poisoned.update(pnames)
                for p in pnames:
                    bindings.pop(p, None)
                continue
            if len(pnames) == 1:
                pname = pnames[0]
                vals = _eval_expr(argvals_node, outer_map)
                if (
                    vals
                    and fc[pname] == 1
                    and pname not in global_names
                    and pname not in poisoned
                    and pname not in bindings
                ):
                    bindings[pname] = vals
                else:
                    poisoned.add(pname)
                    bindings.pop(pname, None)
            else:
                rows = _unwrap_seq_expr(argvals_node, outer_map)
                cols = [[] for _ in pnames]
                ok_rows = bool(rows)
                if ok_rows:
                    for row in rows:
                        row_elts = _unwrap_seq_expr(row, outer_map)
                        if not row_elts or len(row_elts) != len(pnames):
                            ok_rows = False
                            break
                        for j, cell in enumerate(row_elts):
                            if cols[j] is None:
                                continue
                            cv = _eval_expr(cell, outer_map)
                            if cv:
                                cols[j].extend(cv)
                            else:
                                cols[j] = None
                if not ok_rows:
                    poisoned.update(pnames)
                    for pname in pnames:
                        bindings.pop(pname, None)
                else:
                    for j, pname in enumerate(pnames):
                        if (
                            cols[j]
                            and fc[pname] == 1
                            and pname not in global_names
                            and pname not in poisoned
                            and pname not in bindings
                        ):
                            bindings[pname] = tuple(cols[j])
                        else:
                            poisoned.add(pname)
                            bindings.pop(pname, None)
        return bindings

    def _fn_active_map(fn, outer_map):
        fc = fn_counts[id(fn)]
        if "*" in fc:
            return {}
        base = {k: v for k, v in outer_map.items() if k not in fn_bound[id(fn)]}
        base.update(_single_assign_exprs(fn.body, fc, global_names))
        return base

    def _is_pytest_entrypoint(fn):
        if resolved.name.startswith("test_") and fn.name.startswith("test_"):
            return True
        if _parametrize_bindings(fn, mod_expr_map):
            return True
        for dec in fn.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(target, ast.Attribute) and target.attr == "fixture" and _attr_chain(target.value) in all_pytest_chains:
                return True
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "validates"
                and isinstance(target.value, ast.Attribute)
                and target.value.attr == "mark"
                and _attr_chain(target.value.value) in all_pytest_chains
            ):
                return True
        return False

    top_fns = [
        stmt
        for stmt in tree.body
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not stmt.decorator_list
        and module_counts["*"] == 0
        and module_counts[stmt.name] == 1
        and stmt.name not in global_names
        and (return_exports_only or not _is_pytest_entrypoint(stmt))
    ]

    import_wrappers = {}
    spec_wrappers = {}
    wrapped_call_ids = set()
    changed = True
    while changed:
        changed = False
        for fn in top_fns:
            if fn.name in import_wrappers or fn.name in spec_wrappers:
                continue
            pos_params = [a.arg for a in (*fn.args.posonlyargs, *fn.args.args)]
            kw_params = {a.arg for a in fn.args.kwonlyargs}
            fbound = fn_bound[id(fn)]
            all_params = {p for p in (set(pos_params) | kw_params) if fn_counts[id(fn)][p] == 1 and p not in global_names}
            if not all_params:
                continue
            fmap = _fn_active_map(fn, mod_expr_map)
            imp_calls = []
            spec_calls = []
            chained_imp = []
            chained_spec = []
            for sub in ast.walk(fn):
                if not isinstance(sub, ast.Call):
                    continue
                kind = _call_kind(sub.func, fbound)
                chain = _attr_chain(sub.func)
                if kind in {"import_module", "importorskip", "__import__"}:
                    imp_calls.append((sub, kind))
                elif kind == "spec_from_file_location":
                    spec_calls.append(sub)
                elif isinstance(sub.func, ast.Name) and sub.func.id not in fbound:
                    if sub.func.id in import_wrappers:
                        idx, pname, pkg_c, orig_p, _ = import_wrappers[sub.func.id]
                        chained_imp.append((sub, idx, pname, pkg_c, orig_p))
                    elif sub.func.id in imported_imp_wrappers:
                        idx, pname, pkg_c, orig_p = imported_imp_wrappers[sub.func.id]
                        chained_imp.append((sub, idx, pname, pkg_c, orig_p))
                    elif sub.func.id in spec_wrappers:
                        idx, kw_p, tgt_p, loc_a, wfmap, orig_r, _ = spec_wrappers[sub.func.id]
                        chained_spec.append((sub, idx, kw_p, tgt_p, loc_a, wfmap, orig_r))
                    elif sub.func.id in imported_spec_wrappers:
                        idx, kw_p, tgt_p, loc_a, wfmap, orig_r = imported_spec_wrappers[sub.func.id]
                        chained_spec.append((sub, idx, kw_p, tgt_p, loc_a, wfmap, orig_r))
                elif chain is not None and chain[0] not in fbound:
                    if chain in mod_imp_wrappers:
                        idx, pname, pkg_c, orig_p = mod_imp_wrappers[chain]
                        chained_imp.append((sub, idx, pname, pkg_c, orig_p))
                    elif chain in mod_spec_wrappers:
                        idx, kw_p, tgt_p, loc_a, wfmap, orig_r = mod_spec_wrappers[chain]
                        chained_spec.append((sub, idx, kw_p, tgt_p, loc_a, wfmap, orig_r))
            total_calls = len(imp_calls) + len(spec_calls) + len(chained_imp) + len(chained_spec)
            if total_calls != 1:
                continue
            if imp_calls:
                sub, kind = imp_calls[0]
                arg0 = _call_arg_expr(sub, 0, "name" if kind != "importorskip" else "modname")
                pkg_kw = _call_arg_expr(sub, 1, "package") if kind == "import_module" else None
                pkg_const = None
                pkg_ok = _import_call_kwargs_ok(sub, kind) and (
                    pkg_kw is None or (isinstance(pkg_kw, ast.Constant) and (pkg_kw.value is None or isinstance(pkg_kw.value, str)))
                )
                if pkg_ok and isinstance(pkg_kw, ast.Constant) and isinstance(pkg_kw.value, str):
                    pkg_const = pkg_kw.value
                if pkg_ok and isinstance(arg0, ast.Name) and arg0.id in all_params:
                    p_idx = pos_params.index(arg0.id) if arg0.id in pos_params else None
                    import_wrappers[fn.name] = (p_idx, arg0.id, pkg_const, pkg_parts, id(fn))
                    wrapped_call_ids.add(id(sub))
                    changed = True
            elif chained_imp:
                sub, c_idx, c_pname, c_pkg, c_orig = chained_imp[0]
                arg0 = _call_arg_expr(sub, c_idx, c_pname)
                if isinstance(arg0, ast.Name) and arg0.id in all_params:
                    p_idx = pos_params.index(arg0.id) if arg0.id in pos_params else None
                    import_wrappers[fn.name] = (p_idx, arg0.id, c_pkg, c_orig, id(fn))
                    wrapped_call_ids.add(id(sub))
                    changed = True
            elif spec_calls:
                sub = spec_calls[0]
                loc_arg = _call_arg_expr(sub, 1, "location")
                loader_arg = _call_arg_expr(sub, 2, "loader")
                loader_ok = (
                    (loader_arg is None or (isinstance(loader_arg, ast.Constant) and loader_arg.value is None))
                    and not any(isinstance(a, ast.Starred) for a in sub.args)
                    and not any(kw.arg is None for kw in sub.keywords)
                )
                if (
                    loader_ok
                    and loc_arg is not None
                    and _eval_expr(loc_arg, fmap) is None
                    and not _is_provably_external_py_suffix(_joinedstr_const_suffix(loc_arg, fmap), resolved)
                ):
                    used_params = _free_names(loc_arg) & all_params
                    if len(used_params) == 1:
                        (pname,) = tuple(used_params)
                        p_idx = pos_params.index(pname) if pname in pos_params else None
                        wfmap = dict(fmap)
                        wfmap.setdefault("__file__", ast.Constant(value=str(resolved)))
                        spec_wrappers[fn.name] = (p_idx, pname, pname, loc_arg, wfmap, resolved, id(fn))
                        wrapped_call_ids.add(id(sub))
                        changed = True
            elif chained_spec:
                sub, c_idx, c_kw, c_tgt, c_loc, c_fmap, c_orig = chained_spec[0]
                arg0 = _call_arg_expr(sub, c_idx, c_kw)
                if isinstance(arg0, ast.Name) and arg0.id in all_params:
                    p_idx = pos_params.index(arg0.id) if arg0.id in pos_params else None
                    spec_wrappers[fn.name] = (p_idx, arg0.id, c_tgt, c_loc, c_fmap, c_orig, id(fn))
                    wrapped_call_ids.add(id(sub))
                    changed = True

    if return_exports_only:
        if module_counts["*"] > 0:
            return _WILDCARD_DYN_EXPORT

        top_unique = _uniquely_bound_top_names(root_tree, module_counts, global_names)

        def _top_bound(name):
            return name in top_unique

        exp_call_kinds = {}
        for n in import_module_aliases:
            if _top_bound(n):
                exp_call_kinds[(n,)] = "import_module"
        for n in dunder_import_aliases:
            if _top_bound(n):
                exp_call_kinds[(n,)] = "__import__"
        for n in importorskip_aliases:
            if _top_bound(n):
                exp_call_kinds[(n,)] = "importorskip"
        for n in spec_from_file_aliases:
            if _top_bound(n):
                exp_call_kinds[(n,)] = "spec_from_file_location"
        for n, k in imported_call_kinds.items():
            if _top_bound(n):
                exp_call_kinds[(n,)] = k
        for c, k in mod_call_kinds.items():
            if _top_bound(c[0]):
                exp_call_kinds[c] = k

        exp_imp_wrappers = {
            (fn_name,): (idx, pname, pkg_c, orig_p)
            for fn_name, (idx, pname, pkg_c, orig_p, _) in import_wrappers.items()
        }
        for wname, sig in imported_imp_wrappers.items():
            if _top_bound(wname):
                exp_imp_wrappers[(wname,)] = sig
        for wchain, sig in mod_imp_wrappers.items():
            if _top_bound(wchain[0]):
                exp_imp_wrappers[wchain] = sig

        exp_spec_wrappers = {
            (fn_name,): (idx, kw_p, tgt_p, loc_a, wfmap, orig_r)
            for fn_name, (idx, kw_p, tgt_p, loc_a, wfmap, orig_r, _) in spec_wrappers.items()
        }
        for wname, sig in imported_spec_wrappers.items():
            if _top_bound(wname):
                exp_spec_wrappers[(wname,)] = sig
        for wchain, sig in mod_spec_wrappers.items():
            if _top_bound(wchain[0]):
                exp_spec_wrappers[wchain] = sig

        exp_il_chains = {c for c in all_importlib_chains if _top_bound(c[0])}
        exp_ilu_chains = {c for c in all_importlib_util_chains if _top_bound(c[0])}
        exp_py_chains = {c for c in all_pytest_chains if _top_bound(c[0])}
        exp_bi_chains = {c for c in all_builtins_chains if _top_bound(c[0])}
        uncalled = frozenset({
            c[0]
            for c in (
                *exp_call_kinds,
                *exp_imp_wrappers,
                *exp_spec_wrappers,
                *exp_il_chains,
                *exp_ilu_chains,
                *exp_bi_chains,
            )
            if len(c) == 1
        })
        return {
            "call_kinds": exp_call_kinds,
            "imp_wrappers": exp_imp_wrappers,
            "spec_wrappers": exp_spec_wrappers,
            "importlib_chains": frozenset(exp_il_chains),
            "importlib_util_chains": frozenset(exp_ilu_chains),
            "pytest_chains": frozenset(exp_py_chains),
            "builtins_chains": frozenset(exp_bi_chains),
            "wildcard": False,
            "uncalled_names": uncalled,
        }

    def _record_module_str(mod_str, pkg_str=None, origin_pkg=pkg_parts):
        if not mod_str:
            extra.add(_UNRESOLVED_IMPORT)
            return
        if mod_str.startswith("."):
            level = len(mod_str) - len(mod_str.lstrip("."))
            rem = mod_str.lstrip(".")
            anchor = tuple(pkg_str.split(".")) if pkg_str else origin_pkg
            if not anchor or level > len(anchor):
                extra.add(_UNRESOLVED_IMPORT)
                return
            base = anchor[: len(anchor) - (level - 1)]
            parts = (*base, *rem.split(".")) if rem else base
        else:
            parts = tuple(mod_str.split("."))
        top = parts[0]
        if top in _REFERENCE_PACKAGES:
            if len(parts) >= 2:
                mod = ".".join(parts)
                extra.add(mod)
                if mod == "external_bridge._subprocess":
                    extra.add(_UNRESOLVED_WORKER)
            else:
                expand_pkg_parts(parts)
                if top == "external_bridge":
                    extra.add(_UNRESOLVED_WORKER)
        elif origin_pkg is not None and _is_submodule_parts(origin_pkg, top):
            mod = ".".join((*origin_pkg, *parts))
            extra.add(mod)
            if mod == "external_bridge._subprocess":
                extra.add(_UNRESOLVED_WORKER)
        elif origin_pkg is None:
            helper, enc_inits = _resolve_local_helper_with_inits(parts, local_dirs)
            for init_py in enc_inits:
                extra.update(_scan_init_top_via(init_py, seen))
            if helper is not None:
                extra.update(_scan_module_via(helper, seen))

    def _record_file_target(cand, origin_res=resolved):
        try:
            cand_res = pathlib.Path(cand).resolve()
        except OSError:
            extra.add(_UNRESOLVED_IMPORT)
            return
        if not cand_res.is_file():
            extra.add(_UNRESOLVED_IMPORT)
            return
        try:
            rel_src = cand_res.relative_to(_SRC_ROOT.resolve())
            if len(rel_src.parts) >= 2 and rel_src.parts[0] in _REFERENCE_PACKAGES:
                if cand_res.name == "__init__.py":
                    expand_pkg_parts(rel_src.parts[:-1])
                    if rel_src.parts == ("external_bridge", "__init__.py"):
                        extra.add(_UNRESOLVED_WORKER)
                else:
                    mod = ".".join((*rel_src.parts[:-1], cand_res.stem))
                    extra.add(mod)
                    if mod == "external_bridge._subprocess":
                        extra.add(_UNRESOLVED_WORKER)
                return
        except ValueError:
            pass
        allowed_roots = [_REPO_ROOT.resolve()]
        for ref_base in (resolved, origin_res):
            try:
                ref_base.relative_to(_REPO_ROOT.resolve())
            except ValueError:
                base = ref_base.parent.parent if ref_base.parent.name in {"tests", "scripts", "src"} else ref_base.parent
                allowed_roots.append(base.resolve())
        if any(cand_res == r or r in cand_res.parents for r in allowed_roots):
            for init_py in _enclosing_local_package_inits(cand_res.parent):
                if init_py.resolve() != cand_res:
                    extra.update(_scan_init_top_via(init_py, seen))
            extra.update(_scan_module_via(cand_res, seen))
        else:
            extra.add(_UNRESOLVED_IMPORT)

    direct_call_funcs = set()
    attr_parents = {}
    for parent in ast.walk(tree):
        if isinstance(parent, ast.Call):
            direct_call_funcs.add(id(parent.func))
        elif isinstance(parent, ast.Attribute):
            attr_parents[id(parent.value)] = parent

    called_imp_wrappers = set()
    called_spec_wrappers = set()
    called_imported_imp = set()
    called_imported_spec = set()
    called_direct_aliases = set()
    referenced_mod_roots = set()
    all_wrapper_names = frozenset(
        set(import_wrappers)
        | set(spec_wrappers)
        | set(imported_imp_wrappers)
        | set(imported_spec_wrappers)
    )
    direct_callable_names = (
        set(import_module_aliases)
        | set(dunder_import_aliases)
        | set(importorskip_aliases)
        | set(spec_from_file_aliases)
        | set(imported_call_kinds)
        | {"__import__"}
    )

    def _eval_imp_wrapper_call(node, idx, pname, pkg_const, orig_pkg, active_map, param_bindings):
        warg = _call_arg_expr(node, idx, pname)
        vals = _eval_expr(warg, active_map, param_bindings)
        if not vals or not all(isinstance(v, str) for v in vals):
            extra.add(_UNRESOLVED_IMPORT)
        else:
            for v in vals:
                _record_module_str(v, pkg_const, orig_pkg)

    def _eval_spec_wrapper_call(node, idx, kw_p, tgt_p, loc_arg, w_fmap, orig_res, active_map, param_bindings):
        warg = _call_arg_expr(node, idx, kw_p)
        vals = _eval_expr(warg, active_map, param_bindings)
        resolved_locs = _eval_expr(loc_arg, w_fmap, {tgt_p: vals}) if vals else None
        if resolved_locs:
            for loc in resolved_locs:
                _record_file_target(loc, orig_res)
        else:
            extra.add(_UNRESOLVED_IMPORT)

    def _walk(node, active_map, shadowed, param_bindings):
        if isinstance(node, ast.Call):
            if id(node) in wrapped_call_ids:
                pass
            else:
                kind = _call_kind(node.func, shadowed)
                chain = _attr_chain(node.func)
                if kind in {"import_module", "importorskip", "__import__"}:
                    if isinstance(node.func, ast.Name):
                        called_direct_aliases.add(node.func.id)
                    arg0 = _call_arg_expr(node, 0, "name" if kind != "importorskip" else "modname")
                    vals = _eval_expr(arg0, active_map, param_bindings)
                    pkg_val = None
                    pkg_ok = _import_call_kwargs_ok(node, kind)
                    if kind == "import_module":
                        pkg_arg = _call_arg_expr(node, 1, "package")
                        if pkg_arg is not None and not (isinstance(pkg_arg, ast.Constant) and pkg_arg.value is None):
                            pvals = _eval_expr(pkg_arg, active_map, param_bindings)
                            if pvals and len(pvals) == 1 and isinstance(pvals[0], str):
                                pkg_val = pvals[0]
                            else:
                                pkg_ok = False
                    if not vals or not pkg_ok or not all(isinstance(v, str) for v in vals):
                        pref = _joinedstr_const_prefix(arg0, active_map) if pkg_ok else ""
                        non_ref_tops = _SKIP_LOCAL_TOPS - set(_REFERENCE_PACKAGES)
                        top_pref = pref.split(".", 1)[0] if "." in pref and not pref.startswith(".") else ""
                        if top_pref in non_ref_tops and (pkg_parts is None or not _is_submodule_parts(pkg_parts, top_pref)):
                            pass
                        else:
                            extra.add(_UNRESOLVED_IMPORT)
                    else:
                        for v in vals:
                            _record_module_str(v, pkg_val)
                elif kind == "spec_from_file_location":
                    if isinstance(node.func, ast.Name):
                        called_direct_aliases.add(node.func.id)
                    loc_arg = _call_arg_expr(node, 1, "location")
                    loader_arg = _call_arg_expr(node, 2, "loader")
                    loader_ok = (
                        (loader_arg is None or (isinstance(loader_arg, ast.Constant) and loader_arg.value is None))
                        and not any(isinstance(a, ast.Starred) for a in node.args)
                        and not any(kw.arg is None for kw in node.keywords)
                    )
                    vals = _eval_expr(loc_arg, active_map, param_bindings) if loader_ok else None
                    if not loader_ok:
                        extra.add(_UNRESOLVED_IMPORT)
                    elif vals:
                        for v in vals:
                            _record_file_target(v)
                    else:
                        suf = _joinedstr_const_suffix(loc_arg, active_map)
                        if not _is_provably_external_py_suffix(suf, resolved):
                            extra.add(_UNRESOLVED_IMPORT)
                elif isinstance(node.func, ast.Name) and node.func.id not in shadowed:
                    if node.func.id in import_wrappers:
                        called_imp_wrappers.add(node.func.id)
                        idx, pname, pkg_const, orig_pkg, _ = import_wrappers[node.func.id]
                        _eval_imp_wrapper_call(node, idx, pname, pkg_const, orig_pkg, active_map, param_bindings)
                    elif node.func.id in imported_imp_wrappers:
                        called_imported_imp.add(node.func.id)
                        idx, pname, pkg_const, orig_pkg = imported_imp_wrappers[node.func.id]
                        _eval_imp_wrapper_call(node, idx, pname, pkg_const, orig_pkg, active_map, param_bindings)
                    elif node.func.id in spec_wrappers:
                        called_spec_wrappers.add(node.func.id)
                        idx, kw_p, tgt_p, loc_arg, w_fmap, orig_res, _ = spec_wrappers[node.func.id]
                        _eval_spec_wrapper_call(node, idx, kw_p, tgt_p, loc_arg, w_fmap, orig_res, active_map, param_bindings)
                    elif node.func.id in imported_spec_wrappers:
                        called_imported_spec.add(node.func.id)
                        idx, kw_p, tgt_p, loc_arg, w_fmap, orig_res = imported_spec_wrappers[node.func.id]
                        _eval_spec_wrapper_call(node, idx, kw_p, tgt_p, loc_arg, w_fmap, orig_res, active_map, param_bindings)
                elif chain is not None and chain[0] not in shadowed:
                    if chain in mod_imp_wrappers:
                        idx, pname, pkg_const, orig_pkg = mod_imp_wrappers[chain]
                        _eval_imp_wrapper_call(node, idx, pname, pkg_const, orig_pkg, active_map, param_bindings)
                    elif chain in mod_spec_wrappers:
                        idx, kw_p, tgt_p, loc_arg, w_fmap, orig_res = mod_spec_wrappers[chain]
                        _eval_spec_wrapper_call(node, idx, kw_p, tgt_p, loc_arg, w_fmap, orig_res, active_map, param_bindings)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            referenced_mod_roots.add(node.id)
            if (
                node.id in direct_callable_names
                and node.id not in shadowed
                and id(node) not in direct_call_funcs
            ):
                extra.add(_UNRESOLVED_IMPORT)
            elif node.id in all_wrapper_names and node.id not in shadowed and id(node) not in direct_call_funcs:
                extra.add(_UNRESOLVED_IMPORT)
            elif (
                node.id not in shadowed
                and (
                    (node.id,) in all_importlib_chains
                    or (node.id,) in all_importlib_util_chains
                    or (node.id,) in all_builtins_chains
                    or (node.id,) in mod_dyn_prefixes
                )
                and id(node) not in attr_parents
            ):
                extra.add(_UNRESOLVED_IMPORT)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            chain = _attr_chain(node)
            if id(node) not in direct_call_funcs and (
                _call_kind(node, shadowed) is not None
                or (chain in mod_imp_wrappers and chain[0] not in shadowed)
                or (chain in mod_spec_wrappers and chain[0] not in shadowed)
            ):
                extra.add(_UNRESOLVED_IMPORT)
            elif (
                chain is not None
                and chain[0] not in shadowed
                and (
                    chain in all_importlib_chains
                    or chain in all_importlib_util_chains
                    or chain in all_builtins_chains
                    or chain in mod_dyn_prefixes
                )
                and id(node) not in attr_parents
            ):
                extra.add(_UNRESOLVED_IMPORT)

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
                _walk(sub, active_map, shadowed, param_bindings)
            fmap = _fn_active_map(node, active_map)
            fshadow = (shadowed | fn_bound[id(node)]) if "*" not in fn_counts[id(node)] else (all_wrapper_names | direct_callable_names)
            fparams = _parametrize_bindings(node, active_map)
            for stmt in node.body:
                _walk(stmt, fmap, fshadow, fparams)
            return
        if isinstance(node, ast.ClassDef):
            for sub in (*node.decorator_list, *node.bases, *node.keywords):
                _walk(sub, active_map, shadowed, param_bindings)
            cls_counts = collections.Counter()
            _collect_scope_bindings(node, cls_counts, is_scope_root=True)
            cbound = frozenset(k for k in cls_counts if k != "*")
            cmap = {} if "*" in cls_counts else {k: v for k, v in active_map.items() if k not in cbound}
            cshadow = (shadowed | cbound) if "*" not in cls_counts else (all_wrapper_names | direct_callable_names)
            for stmt in node.body:
                _walk(stmt, cmap, cshadow, None)
            return
        if isinstance(node, ast.Lambda):
            for sub in (*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)):
                _walk(sub, active_map, shadowed, param_bindings)
            lam_counts = collections.Counter()
            _collect_scope_bindings(node, lam_counts, is_scope_root=True)
            lbound = frozenset(lam_counts)
            _walk(node.body, {k: v for k, v in active_map.items() if k not in lbound}, shadowed | lbound, None)
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            comp_counts = collections.Counter()
            _collect_scope_bindings(node, comp_counts, is_scope_root=True)
            cbound = frozenset(comp_counts)
            cmap = {k: v for k, v in active_map.items() if k not in cbound}
            cshadow = shadowed | cbound
            for child in ast.iter_child_nodes(node):
                _walk(child, cmap, cshadow, param_bindings)
            return

        for child in ast.iter_child_nodes(node):
            _walk(child, active_map, shadowed, param_bindings)

    _walk(tree, mod_expr_map, frozenset(), None)
    if (
        (set(import_wrappers) - called_imp_wrappers - set(allow_uncalled_imp))
        or (set(spec_wrappers) - called_spec_wrappers - set(allow_uncalled_imp))
    ):
        extra.add(_UNRESOLVED_IMPORT)
    single_imported_callables = (
        set(import_module_aliases)
        | set(dunder_import_aliases)
        | set(importorskip_aliases)
        | set(spec_from_file_aliases)
        | set(imported_call_kinds)
        | set(imported_imp_wrappers)
        | set(imported_spec_wrappers)
    )
    if is_full_module:
        if single_imported_callables - (called_direct_aliases | called_imported_imp | called_imported_spec) - set(allow_uncalled_imp):
            extra.add(_UNRESOLVED_IMPORT)
    else:
        root_called = {
            n.func.id for n in ast.walk(root_tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        if single_imported_callables - root_called - set(allow_uncalled_imp):
            extra.add(_UNRESOLVED_IMPORT)
    return extra


@functools.lru_cache(maxsize=None)
def _parsed_module(path):
    p = pathlib.Path(path)
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p
    try:
        return resolved, ast.parse(p.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return resolved, None


def _sp_has_any(exp):
    return bool(exp["wildcard"] or exp["wrappers"] or exp["sp_chains"] or exp["eb_chains"])


def _sp_has_symbol(sym_name, exp):
    return any(c[0] == sym_name for c in (*exp["wrappers"], *exp["sp_chains"], *exp["eb_chains"]))


def _dyn_has_any(exp):
    return bool(
        exp["wildcard"]
        or exp["call_kinds"]
        or exp["imp_wrappers"]
        or exp["spec_wrappers"]
        or exp["importlib_chains"]
        or exp["importlib_util_chains"]
        or exp["pytest_chains"]
        or exp["builtins_chains"]
    )


def _dyn_has_symbol(sym_name, exp):
    return any(
        c[0] == sym_name
        for c in (
            *exp["call_kinds"],
            *exp["imp_wrappers"],
            *exp["spec_wrappers"],
            *exp["importlib_chains"],
            *exp["importlib_util_chains"],
            *exp["pytest_chains"],
            *exp["builtins_chains"],
        )
    )


def _exported_sp_wrappers_inner(path, seen_w):
    resolved, tree = _parsed_module(str(path))
    if tree is None or resolved in seen_w:
        return _EMPTY_SP_EXPORT
    next_seen = seen_w | {resolved}
    pkg_parts = _reference_pkg_parts(resolved)
    local_dirs = () if pkg_parts is not None else _local_search_dirs(resolved)
    sp_aliases = set()
    run_aliases = set()
    eb_aliases = set()
    wildcard_sp = False
    imported_sp_wrappers = {}
    mod_sp_wrappers = {}
    mod_sp_chains = set()
    mod_eb_chains = set()

    def _reg_mod_wrappers(prefix_parts, target_file):
        nonlocal wildcard_sp
        if target_file is not None and target_file.is_file():
            exp = _exported_sp_wrappers(target_file, next_seen)
            if exp["wildcard"]:
                wildcard_sp = True
            for wchain, sig in exp["wrappers"].items():
                mod_sp_wrappers[(*prefix_parts, *wchain)] = sig
            for schain in exp["sp_chains"]:
                mod_sp_chains.add((*prefix_parts, *schain))
            for echain in exp["eb_chains"]:
                mod_eb_chains.add((*prefix_parts, *echain))

    def _reg_sp_symbol(bname, sym_name, exp):
        if (sym_name,) in exp["wrappers"]:
            imported_sp_wrappers[bname] = exp["wrappers"][(sym_name,)]
        for wchain, sig in exp["wrappers"].items():
            if len(wchain) > 1 and wchain[0] == sym_name:
                mod_sp_wrappers[(bname, *wchain[1:])] = sig
        for schain in exp["sp_chains"]:
            if len(schain) == 1 and schain[0] == sym_name:
                sp_aliases.add(bname)
            elif len(schain) > 1 and schain[0] == sym_name:
                mod_sp_chains.add((bname, *schain[1:]))
        for echain in exp["eb_chains"]:
            if len(echain) == 1 and echain[0] == sym_name:
                eb_aliases.add(bname)
            elif len(echain) > 1 and echain[0] == sym_name:
                mod_eb_chains.add((bname, *echain[1:]))

    def _from_ref(target_parts, aliases):
        nonlocal wildcard_sp
        mod_file = _SRC_ROOT.joinpath(*target_parts[:-1], f"{target_parts[-1]}.py") if len(target_parts) >= 2 else None
        if mod_file is not None and mod_file.is_file():
            mod = ".".join(target_parts)
            exp = _exported_sp_wrappers(mod_file, next_seen) if mod != "external_bridge._subprocess" else _EMPTY_SP_EXPORT
            for alias in aliases:
                if mod == "external_bridge._subprocess":
                    if alias.name == "*":
                        wildcard_sp = True
                    elif alias.name == "run":
                        run_aliases.add(alias.asname or "run")
                    elif alias.name not in _SAFE_SP_ATTRS:
                        wildcard_sp = True
                elif alias.name == "*":
                    if _sp_has_any(exp):
                        wildcard_sp = True
                elif _sp_has_symbol(alias.name, exp):
                    _reg_sp_symbol(alias.asname or alias.name, alias.name, exp)
            return
        if _SRC_ROOT.joinpath(*target_parts).is_dir():
            exports = _init_exports(target_parts)
            init_py = _SRC_ROOT.joinpath(*target_parts, "__init__.py")
            init_exp = _exported_sp_wrappers(init_py, next_seen) if init_py.is_file() else _EMPTY_SP_EXPORT
            for alias in aliases:
                if _is_submodule_parts(target_parts, alias.name):
                    mod = ".".join((*target_parts, alias.name))
                    if mod == "external_bridge._subprocess":
                        sp_aliases.add(alias.asname or "_subprocess")
                    else:
                        _reg_mod_wrappers((alias.asname or alias.name,), _ref_target_file((*target_parts, alias.name)))
                elif alias.name == "*" and _sp_has_any(init_exp):
                    wildcard_sp = True
                elif alias.name in exports and _sp_has_symbol(alias.name, init_exp):
                    _reg_sp_symbol(alias.asname or alias.name, alias.name, init_exp)

    def _from_local(rel_parts, aliases, search_dirs):
        nonlocal wildcard_sp
        local_mod, _ = _resolve_local_helper_with_inits(rel_parts, search_dirs)
        local_exp = _exported_sp_wrappers(local_mod, next_seen) if local_mod is not None else _EMPTY_SP_EXPORT
        for alias in aliases:
            if alias.name == "*":
                if _sp_has_any(local_exp):
                    wildcard_sp = True
                continue
            sub_local, _ = _resolve_local_helper_with_inits((*rel_parts, alias.name), search_dirs)
            if sub_local is not None:
                _reg_mod_wrappers((alias.asname or alias.name,), sub_local)
            elif _sp_has_symbol(alias.name, local_exp):
                _reg_sp_symbol(alias.asname or alias.name, alias.name, local_exp)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = tuple(alias.name.split("."))
                top = parts[0]
                bprefix = (alias.asname,) if alias.asname else parts
                if top in _REFERENCE_PACKAGES:
                    if len(parts) >= 2:
                        mod = ".".join(parts)
                        if mod == "external_bridge._subprocess":
                            if alias.asname:
                                sp_aliases.add(alias.asname)
                            else:
                                eb_aliases.add("external_bridge")
                        else:
                            _reg_mod_wrappers(bprefix, _ref_target_file(parts))
                    else:
                        if top == "external_bridge":
                            eb_aliases.add(alias.asname or "external_bridge")
                        _reg_mod_wrappers(bprefix, _SRC_ROOT.joinpath(top, "__init__.py"))
                elif pkg_parts is not None and _is_submodule_parts(pkg_parts, top):
                    full_parts = (*pkg_parts, *parts)
                    mod = ".".join(full_parts)
                    if mod == "external_bridge._subprocess" and len(parts) == 1:
                        sp_aliases.add(alias.asname or "_subprocess")
                    else:
                        _reg_mod_wrappers(bprefix, _ref_target_file(full_parts))
                elif pkg_parts is None:
                    helper, _ = _resolve_local_helper_with_inits(parts, local_dirs)
                    if alias.asname is None:
                        for k in range(1, len(parts)):
                            init_k, _ = _resolve_local_helper_with_inits(parts[:k], local_dirs)
                            if init_k is not None:
                                _reg_mod_wrappers(parts[:k], init_k)
                    if helper is not None:
                        _reg_mod_wrappers(bprefix, helper)
        elif isinstance(node, ast.ImportFrom):
            if node.level >= 1 and pkg_parts is not None:
                if node.level <= len(pkg_parts):
                    anchor = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                    target_parts = (*anchor, *node.module.split(".")) if node.module else anchor
                    _from_ref(target_parts, node.names)
            elif node.level >= 1 and pkg_parts is None:
                anchor_dir = resolved.parent
                valid = True
                for _ in range(node.level - 1):
                    if anchor_dir.parent == anchor_dir:
                        valid = False
                        break
                    anchor_dir = anchor_dir.parent
                if valid:
                    rel_parts = tuple(node.module.split(".")) if node.module else ()
                    _from_local(rel_parts, node.names, (anchor_dir,))
            elif node.level == 0 and node.module:
                parts = tuple(node.module.split("."))
                if parts[0] in _REFERENCE_PACKAGES:
                    _from_ref(parts, node.names)
                elif pkg_parts is None:
                    _from_local(parts, node.names, local_dirs)

    return _subprocess_workers(
        tree,
        sp_aliases,
        run_aliases,
        eb_aliases,
        wildcard_sp,
        full_tree=tree,
        imported_sp_wrappers=imported_sp_wrappers,
        mod_sp_wrappers=mod_sp_wrappers,
        mod_sp_chains=mod_sp_chains,
        mod_eb_chains=mod_eb_chains,
        return_wrappers_only=True,
    )


@functools.lru_cache(maxsize=None)
def _exported_sp_wrappers_cached(resolved_str):
    return _exported_sp_wrappers_inner(resolved_str, frozenset())


def _exported_sp_wrappers(path, seen_w=frozenset()):
    try:
        resolved_str = str(pathlib.Path(path).resolve())
    except OSError:
        resolved_str = str(path)
    if not seen_w:
        return _exported_sp_wrappers_cached(resolved_str)
    return _exported_sp_wrappers_inner(resolved_str, seen_w)


def _exported_dyn_imports_inner(path, seen_d):
    resolved, tree = _parsed_module(str(path))
    if tree is None or resolved in seen_d:
        return _EMPTY_DYN_EXPORT
    next_seen = seen_d | {resolved}
    pkg_parts = _reference_pkg_parts(resolved)
    local_dirs = () if pkg_parts is not None else _local_search_dirs(resolved)
    importlib_aliases = set()
    importlib_util_aliases = set()
    import_module_aliases = set()
    dunder_import_aliases = set()
    spec_from_file_aliases = set()
    pytest_aliases = set()
    importorskip_aliases = set()
    builtins_aliases = set()
    pathlib_aliases = set()
    path_cls_aliases = set()
    wildcard_import = False
    wildcard_dyn = False
    imported_call_kinds = {}
    mod_call_kinds = {}
    imported_imp_wrappers = {}
    mod_imp_wrappers = {}
    imported_spec_wrappers = {}
    mod_spec_wrappers = {}
    mod_importlib_chains = set()
    mod_importlib_util_chains = set()
    mod_pytest_chains = set()
    mod_builtins_chains = set()

    def _reg_dyn_mod(prefix_parts, target_file):
        nonlocal wildcard_dyn
        if target_file is not None and target_file.is_file():
            exp = _exported_dyn_imports(target_file, next_seen)
            if exp["wildcard"]:
                wildcard_dyn = True
            for cchain, k in exp["call_kinds"].items():
                mod_call_kinds[(*prefix_parts, *cchain)] = k
            for wchain, sig in exp["imp_wrappers"].items():
                mod_imp_wrappers[(*prefix_parts, *wchain)] = sig
            for wchain, sig in exp["spec_wrappers"].items():
                mod_spec_wrappers[(*prefix_parts, *wchain)] = sig
            for c in exp["importlib_chains"]:
                mod_importlib_chains.add((*prefix_parts, *c))
            for c in exp["importlib_util_chains"]:
                mod_importlib_util_chains.add((*prefix_parts, *c))
            for c in exp["pytest_chains"]:
                mod_pytest_chains.add((*prefix_parts, *c))
            for c in exp["builtins_chains"]:
                mod_builtins_chains.add((*prefix_parts, *c))

    def _reg_dyn_symbol(bname, sym_name, exp):
        if (sym_name,) in exp["call_kinds"]:
            imported_call_kinds[bname] = exp["call_kinds"][(sym_name,)]
        for cchain, k in exp["call_kinds"].items():
            if len(cchain) > 1 and cchain[0] == sym_name:
                mod_call_kinds[(bname, *cchain[1:])] = k
        if (sym_name,) in exp["imp_wrappers"]:
            imported_imp_wrappers[bname] = exp["imp_wrappers"][(sym_name,)]
        for wchain, sig in exp["imp_wrappers"].items():
            if len(wchain) > 1 and wchain[0] == sym_name:
                mod_imp_wrappers[(bname, *wchain[1:])] = sig
        if (sym_name,) in exp["spec_wrappers"]:
            imported_spec_wrappers[bname] = exp["spec_wrappers"][(sym_name,)]
        for wchain, sig in exp["spec_wrappers"].items():
            if len(wchain) > 1 and wchain[0] == sym_name:
                mod_spec_wrappers[(bname, *wchain[1:])] = sig
        for c in exp["importlib_chains"]:
            if len(c) == 1 and c[0] == sym_name:
                importlib_aliases.add(bname)
            elif len(c) > 1 and c[0] == sym_name:
                mod_importlib_chains.add((bname, *c[1:]))
        for c in exp["importlib_util_chains"]:
            if len(c) == 1 and c[0] == sym_name:
                importlib_util_aliases.add(bname)
            elif len(c) > 1 and c[0] == sym_name:
                mod_importlib_util_chains.add((bname, *c[1:]))
        for c in exp["pytest_chains"]:
            if len(c) == 1 and c[0] == sym_name:
                pytest_aliases.add(bname)
            elif len(c) > 1 and c[0] == sym_name:
                mod_pytest_chains.add((bname, *c[1:]))
        for c in exp["builtins_chains"]:
            if len(c) == 1 and c[0] == sym_name:
                builtins_aliases.add(bname)
            elif len(c) > 1 and c[0] == sym_name:
                mod_builtins_chains.add((bname, *c[1:]))

    def _from_ref(target_parts, aliases):
        nonlocal wildcard_dyn
        mod_file = _SRC_ROOT.joinpath(*target_parts[:-1], f"{target_parts[-1]}.py") if len(target_parts) >= 2 else None
        if mod_file is not None and mod_file.is_file():
            exp = _exported_dyn_imports(mod_file, next_seen)
            for alias in aliases:
                if alias.name == "*":
                    if _dyn_has_any(exp):
                        wildcard_dyn = True
                elif _dyn_has_symbol(alias.name, exp):
                    _reg_dyn_symbol(alias.asname or alias.name, alias.name, exp)
            return
        if _SRC_ROOT.joinpath(*target_parts).is_dir():
            exports = _init_exports(target_parts)
            init_py = _SRC_ROOT.joinpath(*target_parts, "__init__.py")
            init_exp = _exported_dyn_imports(init_py, next_seen) if init_py.is_file() else _EMPTY_DYN_EXPORT
            for alias in aliases:
                if _is_submodule_parts(target_parts, alias.name):
                    _reg_dyn_mod((alias.asname or alias.name,), _ref_target_file((*target_parts, alias.name)))
                elif alias.name == "*" and _dyn_has_any(init_exp):
                    wildcard_dyn = True
                elif alias.name in exports and _dyn_has_symbol(alias.name, init_exp):
                    _reg_dyn_symbol(alias.asname or alias.name, alias.name, init_exp)

    def _from_local(rel_parts, aliases, search_dirs):
        nonlocal wildcard_dyn
        local_mod, _ = _resolve_local_helper_with_inits(rel_parts, search_dirs)
        local_exp = _exported_dyn_imports(local_mod, next_seen) if local_mod is not None else _EMPTY_DYN_EXPORT
        for alias in aliases:
            if alias.name == "*":
                if _dyn_has_any(local_exp):
                    wildcard_dyn = True
                continue
            sub_local, _ = _resolve_local_helper_with_inits((*rel_parts, alias.name), search_dirs)
            if sub_local is not None:
                _reg_dyn_mod((alias.asname or alias.name,), sub_local)
            elif _dyn_has_symbol(alias.name, local_exp):
                _reg_dyn_symbol(alias.asname or alias.name, alias.name, local_exp)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = tuple(alias.name.split("."))
                top = parts[0]
                bound_name = alias.asname or top
                bprefix = (alias.asname,) if alias.asname else parts
                if alias.name == "importlib":
                    importlib_aliases.add(bound_name)
                elif alias.name == "importlib.util":
                    if alias.asname:
                        importlib_util_aliases.add(alias.asname)
                    else:
                        importlib_aliases.add("importlib")
                elif alias.name == "pytest":
                    pytest_aliases.add(bound_name)
                elif alias.name == "builtins":
                    builtins_aliases.add(bound_name)
                elif alias.name == "pathlib":
                    pathlib_aliases.add(bound_name)
                if top in _REFERENCE_PACKAGES:
                    if len(parts) >= 2:
                        _reg_dyn_mod(bprefix, _ref_target_file(parts))
                    else:
                        _reg_dyn_mod(bprefix, _SRC_ROOT.joinpath(top, "__init__.py"))
                elif pkg_parts is not None and _is_submodule_parts(pkg_parts, top):
                    _reg_dyn_mod(bprefix, _ref_target_file((*pkg_parts, *parts)))
                elif pkg_parts is None:
                    helper, _ = _resolve_local_helper_with_inits(parts, local_dirs)
                    if alias.asname is None:
                        for k in range(1, len(parts)):
                            init_k, _ = _resolve_local_helper_with_inits(parts[:k], local_dirs)
                            if init_k is not None:
                                _reg_dyn_mod(parts[:k], init_k)
                    if helper is not None:
                        _reg_dyn_mod(bprefix, helper)
        elif isinstance(node, ast.ImportFrom):
            if any(a.name == "*" for a in node.names):
                wildcard_import = True
            if node.level == 0 and node.module in {"importlib", "importlib.util", "pytest", "builtins", "pathlib"}:
                for alias in node.names:
                    bname = alias.asname or alias.name
                    if node.module == "importlib" and alias.name == "import_module":
                        import_module_aliases.add(bname)
                    elif node.module == "importlib" and alias.name == "util":
                        importlib_util_aliases.add(bname)
                    elif node.module == "importlib.util" and alias.name == "spec_from_file_location":
                        spec_from_file_aliases.add(bname)
                    elif node.module == "pytest" and alias.name == "importorskip":
                        importorskip_aliases.add(bname)
                    elif node.module == "builtins" and alias.name == "__import__":
                        dunder_import_aliases.add(bname)
                    elif node.module == "pathlib" and alias.name == "Path":
                        path_cls_aliases.add(bname)
                    elif alias.name == "*" and node.module in {"importlib", "importlib.util", "pytest", "builtins"}:
                        wildcard_dyn = True
            if node.level >= 1 and pkg_parts is not None:
                if node.level <= len(pkg_parts):
                    anchor = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                    target_parts = (*anchor, *node.module.split(".")) if node.module else anchor
                    _from_ref(target_parts, node.names)
            elif node.level >= 1 and pkg_parts is None:
                anchor_dir = resolved.parent
                valid = True
                for _ in range(node.level - 1):
                    if anchor_dir.parent == anchor_dir:
                        valid = False
                        break
                    anchor_dir = anchor_dir.parent
                if valid:
                    rel_parts = tuple(node.module.split(".")) if node.module else ()
                    _from_local(rel_parts, node.names, (anchor_dir,))
            elif node.level == 0 and node.module:
                parts = tuple(node.module.split("."))
                if parts[0] in _REFERENCE_PACKAGES:
                    _from_ref(parts, node.names)
                elif pkg_parts is None:
                    _from_local(parts, node.names, local_dirs)

    return _scan_call_imports(
        tree,
        resolved,
        pkg_parts,
        local_dirs,
        frozenset(),
        lambda _: None,
        full_tree=tree,
        importlib_aliases=importlib_aliases,
        importlib_util_aliases=importlib_util_aliases,
        import_module_aliases=import_module_aliases,
        dunder_import_aliases=dunder_import_aliases,
        spec_from_file_aliases=spec_from_file_aliases,
        pytest_aliases=pytest_aliases,
        importorskip_aliases=importorskip_aliases,
        builtins_aliases=builtins_aliases,
        pathlib_aliases=pathlib_aliases,
        path_cls_aliases=path_cls_aliases,
        wildcard_import=wildcard_import,
        imported_call_kinds=imported_call_kinds,
        mod_call_kinds=mod_call_kinds,
        imported_imp_wrappers=imported_imp_wrappers,
        mod_imp_wrappers=mod_imp_wrappers,
        imported_spec_wrappers=imported_spec_wrappers,
        mod_spec_wrappers=mod_spec_wrappers,
        mod_importlib_chains=mod_importlib_chains,
        mod_importlib_util_chains=mod_importlib_util_chains,
        mod_pytest_chains=mod_pytest_chains,
        mod_builtins_chains=mod_builtins_chains,
        wildcard_dyn=wildcard_dyn,
        return_exports_only=True,
    )


@functools.lru_cache(maxsize=None)
def _exported_dyn_imports_cached(resolved_str):
    return _exported_dyn_imports_inner(resolved_str, frozenset())


def _exported_dyn_imports(path, seen_d=frozenset()):
    try:
        resolved_str = str(pathlib.Path(path).resolve())
    except OSError:
        resolved_str = str(path)
    if not seen_d:
        return _exported_dyn_imports_cached(resolved_str)
    return _exported_dyn_imports_inner(resolved_str, seen_d)


def _scan_stmts_via(
    stmts,
    full_tree,
    p,
    seen,
    *,
    allow_uncalled_wrappers=frozenset(),
    allow_uncalled_imp=frozenset(),
):
    try:
        resolved = p.resolve()
    except OSError:
        resolved = p
    pkg_parts = _reference_pkg_parts(resolved)
    local_dirs = () if pkg_parts is not None else _local_search_dirs(resolved)
    found = set()
    sp_aliases = set()
    run_aliases = set()
    eb_aliases = set()
    wildcard_sp = False
    wildcard_import = False
    wildcard_dyn = False
    imported_sp_wrappers = {}
    mod_sp_wrappers = {}
    mod_sp_chains = set()
    mod_eb_chains = set()
    import_sources = collections.defaultdict(set)
    importlib_aliases = set()
    importlib_util_aliases = set()
    import_module_aliases = set()
    dunder_import_aliases = set()
    spec_from_file_aliases = set()
    pytest_aliases = set()
    importorskip_aliases = set()
    builtins_aliases = set()
    pathlib_aliases = set()
    path_cls_aliases = set()
    imported_call_kinds = {}
    mod_call_kinds = {}
    imported_imp_wrappers = {}
    mod_imp_wrappers = {}
    imported_spec_wrappers = {}
    mod_spec_wrappers = {}
    mod_importlib_chains = set()
    mod_importlib_util_chains = set()
    mod_pytest_chains = set()
    mod_builtins_chains = set()

    def _register_mod_wrappers(prefix_parts, target_file):
        nonlocal wildcard_sp, wildcard_dyn
        if target_file is not None and target_file.is_file():
            exp_sp = _exported_sp_wrappers(target_file)
            if exp_sp["wildcard"]:
                wildcard_sp = True
            for wchain, sig in exp_sp["wrappers"].items():
                mod_sp_wrappers[(*prefix_parts, *wchain)] = sig
            for schain in exp_sp["sp_chains"]:
                mod_sp_chains.add((*prefix_parts, *schain))
            for echain in exp_sp["eb_chains"]:
                mod_eb_chains.add((*prefix_parts, *echain))

            exp_dyn = _exported_dyn_imports(target_file)
            if exp_dyn["wildcard"]:
                wildcard_dyn = True
            for cchain, k in exp_dyn["call_kinds"].items():
                mod_call_kinds[(*prefix_parts, *cchain)] = k
            for wchain, sig in exp_dyn["imp_wrappers"].items():
                mod_imp_wrappers[(*prefix_parts, *wchain)] = sig
            for wchain, sig in exp_dyn["spec_wrappers"].items():
                mod_spec_wrappers[(*prefix_parts, *wchain)] = sig
            for c in exp_dyn["importlib_chains"]:
                mod_importlib_chains.add((*prefix_parts, *c))
            for c in exp_dyn["importlib_util_chains"]:
                mod_importlib_util_chains.add((*prefix_parts, *c))
            for c in exp_dyn["pytest_chains"]:
                mod_pytest_chains.add((*prefix_parts, *c))
            for c in exp_dyn["builtins_chains"]:
                mod_builtins_chains.add((*prefix_parts, *c))
            return exp_sp["uncalled_names"], exp_dyn["uncalled_names"]
        return frozenset(), frozenset()

    def _register_symbol_exports(bname, sym_name, exp_sp, exp_dyn):
        sp_hit = False
        if (sym_name,) in exp_sp["wrappers"]:
            imported_sp_wrappers[bname] = exp_sp["wrappers"][(sym_name,)]
            sp_hit = True
        for wchain, sig in exp_sp["wrappers"].items():
            if len(wchain) > 1 and wchain[0] == sym_name:
                mod_sp_wrappers[(bname, *wchain[1:])] = sig
                sp_hit = True
        for schain in exp_sp["sp_chains"]:
            if len(schain) == 1 and schain[0] == sym_name:
                sp_aliases.add(bname)
                sp_hit = True
            elif len(schain) > 1 and schain[0] == sym_name:
                mod_sp_chains.add((bname, *schain[1:]))
                sp_hit = True
        for echain in exp_sp["eb_chains"]:
            if len(echain) == 1 and echain[0] == sym_name:
                eb_aliases.add(bname)
                sp_hit = True
            elif len(echain) > 1 and echain[0] == sym_name:
                mod_eb_chains.add((bname, *echain[1:]))
                sp_hit = True

        dyn_hit = False
        if (sym_name,) in exp_dyn["call_kinds"]:
            imported_call_kinds[bname] = exp_dyn["call_kinds"][(sym_name,)]
            dyn_hit = True
        for cchain, k in exp_dyn["call_kinds"].items():
            if len(cchain) > 1 and cchain[0] == sym_name:
                mod_call_kinds[(bname, *cchain[1:])] = k
                dyn_hit = True
        if (sym_name,) in exp_dyn["imp_wrappers"]:
            imported_imp_wrappers[bname] = exp_dyn["imp_wrappers"][(sym_name,)]
            dyn_hit = True
        for wchain, sig in exp_dyn["imp_wrappers"].items():
            if len(wchain) > 1 and wchain[0] == sym_name:
                mod_imp_wrappers[(bname, *wchain[1:])] = sig
                dyn_hit = True
        if (sym_name,) in exp_dyn["spec_wrappers"]:
            imported_spec_wrappers[bname] = exp_dyn["spec_wrappers"][(sym_name,)]
            dyn_hit = True
        for wchain, sig in exp_dyn["spec_wrappers"].items():
            if len(wchain) > 1 and wchain[0] == sym_name:
                mod_spec_wrappers[(bname, *wchain[1:])] = sig
                dyn_hit = True
        for c in exp_dyn["importlib_chains"]:
            if len(c) == 1 and c[0] == sym_name:
                importlib_aliases.add(bname)
                dyn_hit = True
            elif len(c) > 1 and c[0] == sym_name:
                mod_importlib_chains.add((bname, *c[1:]))
                dyn_hit = True
        for c in exp_dyn["importlib_util_chains"]:
            if len(c) == 1 and c[0] == sym_name:
                importlib_util_aliases.add(bname)
                dyn_hit = True
            elif len(c) > 1 and c[0] == sym_name:
                mod_importlib_util_chains.add((bname, *c[1:]))
                dyn_hit = True
        for c in exp_dyn["pytest_chains"]:
            if len(c) == 1 and c[0] == sym_name:
                pytest_aliases.add(bname)
                dyn_hit = True
            elif len(c) > 1 and c[0] == sym_name:
                mod_pytest_chains.add((bname, *c[1:]))
                dyn_hit = True
        for c in exp_dyn["builtins_chains"]:
            if len(c) == 1 and c[0] == sym_name:
                builtins_aliases.add(bname)
                dyn_hit = True
            elif len(c) > 1 and c[0] == sym_name:
                mod_builtins_chains.add((bname, *c[1:]))
                dyn_hit = True
        return sp_hit, dyn_hit

    def _expand_pkg_parts(target_parts, allow_sp=frozenset(), allow_imp=frozenset()):
        init_py = _SRC_ROOT.joinpath(*target_parts, "__init__.py")
        if len(target_parts) == 1:
            if init_py.is_file():
                if _has_executable_defs(init_py):
                    found.add(target_parts[0])
                found.update(
                    _scan_module_via(
                        init_py,
                        seen,
                        allow_uncalled_wrappers=allow_sp,
                        allow_uncalled_imp=allow_imp,
                    )
                )
        else:
            found.add(".".join(target_parts))
            if init_py.is_file():
                found.update(
                    _scan_module_via(
                        init_py,
                        seen,
                        allow_uncalled_wrappers=allow_sp,
                        allow_uncalled_imp=allow_imp,
                    )
                )

    def _handle_from_ref(target_parts, aliases):
        nonlocal wildcard_sp, wildcard_dyn
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
                    elif alias.name not in _SAFE_SP_ATTRS:
                        wildcard_sp = True
            else:
                exp_sp = _exported_sp_wrappers(mod_file)
                exp_dyn = _exported_dyn_imports(mod_file)
                for alias in aliases:
                    if alias.name == "*":
                        if _sp_has_any(exp_sp):
                            wildcard_sp = True
                        if _dyn_has_any(exp_dyn):
                            wildcard_dyn = True
                    else:
                        _register_symbol_exports(alias.asname or alias.name, alias.name, exp_sp, exp_dyn)
            return
        if _SRC_ROOT.joinpath(*target_parts).is_dir():
            exports = _init_exports(target_parts)
            init_py = _SRC_ROOT.joinpath(*target_parts, "__init__.py")
            init_sp = _exported_sp_wrappers(init_py) if init_py.is_file() else _EMPTY_SP_EXPORT
            init_dyn = _exported_dyn_imports(init_py) if init_py.is_file() else _EMPTY_DYN_EXPORT
            for alias in aliases:
                if _is_submodule_parts(target_parts, alias.name):
                    mod = ".".join((*target_parts, alias.name))
                    found.add(mod)
                    if mod == "external_bridge._subprocess":
                        sp_aliases.add(alias.asname or "_subprocess")
                    else:
                        sub_target = _ref_target_file((*target_parts, alias.name))
                        sub_sp, sub_imp = _register_mod_wrappers((alias.asname or alias.name,), sub_target)
                        if sub_target is not None and sub_target.name == "__init__.py":
                            _expand_pkg_parts((*target_parts, alias.name), sub_sp, sub_imp)
                elif alias.name == "*" or alias.name in exports:
                    _expand_pkg_parts(target_parts, init_sp["uncalled_names"], init_dyn["uncalled_names"])
                    if alias.name == "*":
                        if _sp_has_any(init_sp):
                            wildcard_sp = True
                        if _dyn_has_any(init_dyn):
                            wildcard_dyn = True
                    else:
                        _register_symbol_exports(alias.asname or alias.name, alias.name, init_sp, init_dyn)
                else:
                    found.add(".".join((*target_parts, alias.name)))
            return
        found.add(".".join(target_parts))

    def _scan_enclosing_inits(enc_inits):
        for init_py in enc_inits:
            try:
                init_res = init_py.resolve()
            except OSError:
                init_res = init_py
            if init_res == resolved:
                continue
            found.update(_scan_init_top_via(init_py, seen))

    if pkg_parts is None:
        _scan_enclosing_inits(_enclosing_local_package_inits(resolved.parent))

    def _handle_from_local(rel_parts, aliases, search_dirs):
        nonlocal wildcard_sp, wildcard_dyn
        local_mod, local_enc = _resolve_local_helper_with_inits(rel_parts, search_dirs)
        _scan_enclosing_inits(local_enc)
        local_sp = _exported_sp_wrappers(local_mod) if local_mod is not None else _EMPTY_SP_EXPORT
        local_dyn = _exported_dyn_imports(local_mod) if local_mod is not None else _EMPTY_DYN_EXPORT
        for alias in aliases:
            if alias.name == "*":
                if local_mod is not None:
                    if _sp_has_any(local_sp):
                        wildcard_sp = True
                    if _dyn_has_any(local_dyn):
                        wildcard_dyn = True
                    found.update(
                        _scan_module_via(
                            local_mod,
                            seen,
                            allow_uncalled_wrappers=local_sp["uncalled_names"],
                            allow_uncalled_imp=local_dyn["uncalled_names"],
                        )
                    )
                continue
            sub_local, sub_enc = _resolve_local_helper_with_inits((*rel_parts, alias.name), search_dirs)
            if sub_local is not None:
                _scan_enclosing_inits(sub_enc)
                sub_sp, sub_imp = _register_mod_wrappers((alias.asname or alias.name,), sub_local)
                found.update(
                    _scan_module_via(
                        sub_local,
                        seen,
                        allow_uncalled_wrappers=sub_sp,
                        allow_uncalled_imp=sub_imp,
                    )
                )
            elif local_mod is not None:
                _register_symbol_exports(alias.asname or alias.name, alias.name, local_sp, local_dyn)
                found.update(
                    _scan_symbol_via(
                        local_mod,
                        alias.name,
                        seen,
                        allow_uncalled_wrappers=local_sp["uncalled_names"],
                        allow_uncalled_imp=local_dyn["uncalled_names"],
                    )
                )

    scope_mod = ast.Module(body=list(stmts), type_ignores=[])
    for node in ast.walk(scope_mod):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = tuple(alias.name.split("."))
                top = parts[0]
                bound_name = alias.asname or top
                bprefix = (alias.asname,) if alias.asname else parts
                import_sources[bound_name].add(("import", alias.name if alias.asname else top))
                if alias.name == "importlib":
                    importlib_aliases.add(bound_name)
                elif alias.name == "importlib.util":
                    if alias.asname:
                        importlib_util_aliases.add(alias.asname)
                    else:
                        importlib_aliases.add("importlib")
                elif alias.name == "pytest":
                    pytest_aliases.add(bound_name)
                elif alias.name == "builtins":
                    builtins_aliases.add(bound_name)
                elif alias.name == "pathlib":
                    pathlib_aliases.add(bound_name)
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
                            sub_target = _ref_target_file(parts)
                            sub_sp, sub_imp = _register_mod_wrappers(bprefix, sub_target)
                            if sub_target is not None and sub_target.name == "__init__.py":
                                _expand_pkg_parts(parts, sub_sp, sub_imp)
                    else:
                        if top == "external_bridge":
                            eb_aliases.add(alias.asname or "external_bridge")
                        top_sp, top_imp = _register_mod_wrappers(bprefix, _SRC_ROOT.joinpath(top, "__init__.py"))
                        _expand_pkg_parts(parts, top_sp, top_imp)
                elif pkg_parts is not None and _is_submodule_parts(pkg_parts, top):
                    full_parts = (*pkg_parts, *parts)
                    mod = ".".join(full_parts)
                    found.add(mod)
                    if mod == "external_bridge._subprocess" and len(parts) == 1:
                        sp_aliases.add(alias.asname or "_subprocess")
                    else:
                        sub_target = _ref_target_file(full_parts)
                        sub_sp, sub_imp = _register_mod_wrappers(bprefix, sub_target)
                        if sub_target is not None and sub_target.name == "__init__.py":
                            _expand_pkg_parts(full_parts, sub_sp, sub_imp)
                elif pkg_parts is None:
                    helper, enc_inits = _resolve_local_helper_with_inits(parts, local_dirs)
                    _scan_enclosing_inits(enc_inits)
                    if alias.asname is None:
                        for k in range(1, len(parts)):
                            init_k, _ = _resolve_local_helper_with_inits(parts[:k], local_dirs)
                            if init_k is not None:
                                init_sp_u, init_imp_u = _register_mod_wrappers(parts[:k], init_k)
                                found.update(
                                    _scan_module_via(
                                        init_k,
                                        seen,
                                        allow_uncalled_wrappers=init_sp_u,
                                        allow_uncalled_imp=init_imp_u,
                                    )
                                )
                    if helper is not None:
                        exp_sp_u, exp_imp_u = _register_mod_wrappers(bprefix, helper)
                        found.update(
                            _scan_module_via(
                                helper,
                                seen,
                                allow_uncalled_wrappers=exp_sp_u,
                                allow_uncalled_imp=exp_imp_u,
                            )
                        )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    import_sources[alias.asname or alias.name].add(("from", node.level, node.module, alias.name))
            if any(a.name == "*" for a in node.names):
                wildcard_import = True
            if node.level == 0 and node.module in {"importlib", "importlib.util", "pytest", "builtins", "pathlib"}:
                for alias in node.names:
                    bname = alias.asname or alias.name
                    if node.module == "importlib" and alias.name == "import_module":
                        import_module_aliases.add(bname)
                    elif node.module == "importlib" and alias.name == "util":
                        importlib_util_aliases.add(bname)
                    elif node.module == "importlib.util" and alias.name == "spec_from_file_location":
                        spec_from_file_aliases.add(bname)
                    elif node.module == "pytest" and alias.name == "importorskip":
                        importorskip_aliases.add(bname)
                    elif node.module == "builtins" and alias.name == "__import__":
                        dunder_import_aliases.add(bname)
                    elif node.module == "pathlib" and alias.name == "Path":
                        path_cls_aliases.add(bname)
                    elif alias.name == "*" and node.module in {"importlib", "importlib.util", "pytest", "builtins"}:
                        found.add(_UNRESOLVED_IMPORT)
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
                    rel_parts = tuple(node.module.split(".")) if node.module else ()
                    _handle_from_local(rel_parts, node.names, (anchor_dir,))
            elif node.level == 0 and node.module:
                parts = tuple(node.module.split("."))
                top = parts[0]
                if top in _REFERENCE_PACKAGES:
                    _handle_from_ref(parts, node.names)
                elif pkg_parts is None:
                    _handle_from_local(parts, node.names, local_dirs)

    for w_alias in (
        set(imported_sp_wrappers)
        | sp_aliases
        | run_aliases
        | eb_aliases
        | {chain[0] for chain in (*mod_sp_wrappers, *mod_sp_chains, *mod_eb_chains)}
    ):
        if len(import_sources.get(w_alias, ())) > 1:
            wildcard_sp = True

    for d_alias in (
        set(imported_call_kinds)
        | set(imported_imp_wrappers)
        | set(imported_spec_wrappers)
        | import_module_aliases
        | dunder_import_aliases
        | importorskip_aliases
        | spec_from_file_aliases
        | {
            chain[0]
            for chain in (
                *mod_call_kinds,
                *mod_imp_wrappers,
                *mod_spec_wrappers,
                *mod_importlib_chains,
                *mod_importlib_util_chains,
                *mod_builtins_chains,
            )
        }
    ):
        if len(import_sources.get(d_alias, ())) > 1:
            wildcard_dyn = True

    found |= _scan_call_imports(
        scope_mod,
        resolved,
        pkg_parts,
        local_dirs,
        seen,
        _expand_pkg_parts,
        full_tree=full_tree,
        importlib_aliases=importlib_aliases,
        importlib_util_aliases=importlib_util_aliases,
        import_module_aliases=import_module_aliases,
        dunder_import_aliases=dunder_import_aliases,
        spec_from_file_aliases=spec_from_file_aliases,
        pytest_aliases=pytest_aliases,
        importorskip_aliases=importorskip_aliases,
        builtins_aliases=builtins_aliases,
        pathlib_aliases=pathlib_aliases,
        path_cls_aliases=path_cls_aliases,
        wildcard_import=wildcard_import,
        imported_call_kinds=imported_call_kinds,
        mod_call_kinds=mod_call_kinds,
        imported_imp_wrappers=imported_imp_wrappers,
        mod_imp_wrappers=mod_imp_wrappers,
        imported_spec_wrappers=imported_spec_wrappers,
        mod_spec_wrappers=mod_spec_wrappers,
        mod_importlib_chains=mod_importlib_chains,
        mod_importlib_util_chains=mod_importlib_util_chains,
        mod_pytest_chains=mod_pytest_chains,
        mod_builtins_chains=mod_builtins_chains,
        wildcard_dyn=wildcard_dyn,
        allow_uncalled_imp=allow_uncalled_imp,
        is_full_module=(stmts is full_tree.body),
    )
    found |= _subprocess_workers(
        scope_mod,
        sp_aliases,
        run_aliases,
        eb_aliases,
        wildcard_sp,
        full_tree=full_tree,
        imported_sp_wrappers=imported_sp_wrappers,
        mod_sp_wrappers=mod_sp_wrappers,
        mod_sp_chains=mod_sp_chains,
        mod_eb_chains=mod_eb_chains,
        allow_uncalled_wrappers=allow_uncalled_wrappers,
        is_full_module=(stmts is full_tree.body),
    )
    return {m for m in found if _keep_reference(m)}


def _is_fixture_def(fn_node):
    if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for dec in fn_node.decorator_list:
        func = dec.func if isinstance(dec, ast.Call) else dec
        if (isinstance(func, ast.Name) and func.id == "fixture") or (
            isinstance(func, ast.Attribute) and func.attr == "fixture"
        ):
            return True
    return False


def _is_autouse_fixture(fn_node):
    for dec in fn_node.decorator_list:
        if isinstance(dec, ast.Call):
            func = dec.func
            is_fix = (
                (isinstance(func, ast.Name) and func.id == "fixture")
                or (isinstance(func, ast.Attribute) and func.attr == "fixture")
            )
            if is_fix:
                for kw in dec.keywords:
                    if kw.arg == "autouse" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        return True
    return False


def _usefixtures_in_expr(expr):
    out = set()
    if expr is None:
        return out
    for node in ast.walk(expr):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "usefixtures"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    out.add(arg.value)
    return out


def _usefixtures_in_stmts(stmts):
    out = set()
    for stmt in stmts:
        if isinstance(stmt, ast.Assign):
            targets = []
            for t in stmt.targets:
                if isinstance(t, ast.Name):
                    targets.append(t.id)
                elif isinstance(t, (ast.Tuple, ast.List)):
                    targets.extend(elt.id for elt in t.elts if isinstance(elt, ast.Name))
            if "pytestmark" in targets:
                out |= _usefixtures_in_expr(stmt.value)
        elif (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == "pytestmark"
            and stmt.value is not None
        ):
            out |= _usefixtures_in_expr(stmt.value)
        elif (
            isinstance(stmt, ast.AugAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == "pytestmark"
        ):
            out |= _usefixtures_in_expr(stmt.value)
    return out


def _reachable_local_stmts_and_fixtures(
    tree,
    func_name,
    cls_name=None,
    fixture_names=(),
    include_autouse=True,
    include_hooks=False,
    is_pytest_target=False,
):
    """Top-level statements plus the target function/method, reachable local helpers/fixtures, and requested fixture names."""
    top_stmts = []
    top_defs = {}
    autouse = set()
    hooks = set()
    classes = {}
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            top_defs[stmt.name] = stmt
            if include_autouse and _is_autouse_fixture(stmt):
                autouse.add(stmt.name)
            if include_hooks and stmt.name.startswith("pytest_"):
                hooks.add(stmt.name)
        elif isinstance(stmt, ast.ClassDef):
            classes[stmt.name] = stmt
            top_defs[stmt.name] = stmt
        else:
            top_stmts.append(stmt)

    cls_non_methods = []
    cls_methods = {}
    cls_autouse = set()
    cls_usefixtures = set()
    if cls_name:
        cls_node = classes.get(cls_name)
        if cls_node is None:
            return None, set()
        # Walk enclosing class and any in-file base classes
        cls_queue = [cls_node]
        seen_cls = set()
        while cls_queue:
            c = cls_queue.pop()
            if c.name in seen_cls:
                continue
            seen_cls.add(c.name)
            for dec in c.decorator_list:
                cls_usefixtures |= _usefixtures_in_expr(dec)
            for b in c.bases:
                if isinstance(b, ast.Name) and b.id in classes:
                    cls_queue.append(classes[b.id])
            for body_stmt in c.body:
                if isinstance(body_stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    cls_methods.setdefault(body_stmt.name, body_stmt)
                    if include_autouse and _is_autouse_fixture(body_stmt):
                        cls_autouse.add(body_stmt.name)
                else:
                    cls_non_methods.append(body_stmt)
        cls_usefixtures |= _usefixtures_in_stmts(cls_non_methods)
        target_node = cls_methods.get(func_name) if func_name is not None else None
        if func_name is not None and target_node is None:
            return None, set()
    elif func_name is not None:
        target_node = top_defs.get(func_name)
        if target_node is None:
            return None, set()
    else:
        target_node = None

    requested_fixtures = set(fixture_names)
    if is_pytest_target:
        requested_fixtures |= _usefixtures_in_stmts(top_stmts) | cls_usefixtures
        if target_node is not None:
            for dec in target_node.decorator_list:
                requested_fixtures |= _usefixtures_in_expr(dec)

    seed_nodes = [target_node] if target_node is not None else []
    for fname in autouse | hooks:
        seed_nodes.append(top_defs[fname])
    for fname in requested_fixtures & set(top_defs):
        if _is_fixture_def(top_defs[fname]):
            seed_nodes.append(top_defs[fname])
    for fname in cls_autouse:
        seed_nodes.append(cls_methods[fname])
    for fname in requested_fixtures & set(cls_methods):
        if _is_fixture_def(cls_methods[fname]):
            seed_nodes.append(cls_methods[fname])

    visited_ids = set()
    ordered_defs = []
    queue = list(seed_nodes)

    def _enqueue_refs(ast_root):
        if isinstance(ast_root, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            _is_fixture_def(ast_root) or (is_pytest_target and target_node is not None and ast_root is target_node)
        ):
            inj_names = {
                a.arg
                for a in (*ast_root.args.posonlyargs, *ast_root.args.args, *ast_root.args.kwonlyargs)
                if a.arg not in {"self", "cls"}
            }
            for dec in ast_root.decorator_list:
                inj_names |= _usefixtures_in_expr(dec)
            for fx in inj_names:
                requested_fixtures.add(fx)
                if fx in cls_methods and _is_fixture_def(cls_methods[fx]):
                    m = cls_methods[fx]
                    if id(m) not in visited_ids:
                        queue.append(m)
                if fx in top_defs and _is_fixture_def(top_defs[fx]):
                    cand = top_defs[fx]
                    if id(cand) not in visited_ids:
                        queue.append(cand)
        for sub in ast.walk(ast_root):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load) and sub.id in top_defs:
                cand = top_defs[sub.id]
                if id(cand) not in visited_ids:
                    queue.append(cand)
            elif isinstance(sub, ast.Attribute) and sub.attr in cls_methods:
                m = cls_methods[sub.attr]
                if id(m) not in visited_ids:
                    queue.append(m)

    for ts in (*top_stmts, *cls_non_methods):
        _enqueue_refs(ts)

    while queue:
        curr = queue.pop()
        if id(curr) in visited_ids:
            continue
        visited_ids.add(id(curr))
        ordered_defs.append(curr)
        _enqueue_refs(curr)

    return [*top_stmts, *cls_non_methods, *ordered_defs], requested_fixtures


def _reachable_local_stmts(tree, func_name, cls_name=None, fixture_names=(), include_autouse=True):
    """Top-level statements plus the target function/method and any local helpers/classes/fixtures it reaches."""
    stmts, _ = _reachable_local_stmts_and_fixtures(
        tree,
        func_name,
        cls_name=cls_name,
        fixture_names=fixture_names,
        include_autouse=include_autouse,
        is_pytest_target=include_autouse,
    )
    return stmts


def _scan_symbol_via(
    path,
    symbol_name,
    seen,
    *,
    allow_uncalled_wrappers=frozenset(),
    allow_uncalled_imp=frozenset(),
):
    resolved, tree = _parsed_module(str(path))
    if tree is None:
        return {_UNRESOLVED_IMPORT} if _keep_reference(_UNRESOLVED_IMPORT) else set()
    key = (resolved, symbol_name)
    if key in seen:
        return set()
    stmts = _reachable_local_stmts(tree, symbol_name, include_autouse=False)
    if stmts is None:
        return _scan_module_via(
            path,
            seen,
            allow_uncalled_wrappers=allow_uncalled_wrappers,
            allow_uncalled_imp=allow_uncalled_imp,
        )
    return _scan_stmts_via(
        stmts,
        tree,
        pathlib.Path(path),
        seen | {key},
        allow_uncalled_wrappers=allow_uncalled_wrappers,
        allow_uncalled_imp=allow_uncalled_imp,
    )


def _scan_init_top_via(path, seen):
    resolved, tree = _parsed_module(str(path))
    if tree is None:
        return {_UNRESOLVED_IMPORT} if _keep_reference(_UNRESOLVED_IMPORT) else set()
    key = (resolved, "<init-top>")
    if key in seen or resolved in seen:
        return set()
    isp = _exported_sp_wrappers(path)
    idyn = _exported_dyn_imports(path)
    stmts, _ = _reachable_local_stmts_and_fixtures(tree, None, include_autouse=False)
    return _scan_stmts_via(
        stmts if stmts is not None else tree.body,
        tree,
        pathlib.Path(path),
        seen | {key},
        allow_uncalled_wrappers=isp["uncalled_names"],
        allow_uncalled_imp=idyn["uncalled_names"],
    )


def _scan_module_via(
    path,
    seen,
    *,
    allow_uncalled_wrappers=frozenset(),
    allow_uncalled_imp=frozenset(),
):
    resolved, tree = _parsed_module(str(path))
    if resolved in seen:
        return set()
    if tree is None:
        return {_UNRESOLVED_IMPORT} if _keep_reference(_UNRESOLVED_IMPORT) else set()
    return _scan_stmts_via(
        tree.body,
        tree,
        pathlib.Path(path),
        seen | {resolved},
        allow_uncalled_wrappers=allow_uncalled_wrappers,
        allow_uncalled_imp=allow_uncalled_imp,
    )


@functools.lru_cache(maxsize=None)
def _cached_module_via(path):
    return tuple(sorted(_scan_module_via(path, frozenset())))


def _ancestor_conftest_files(resolved):
    """All existing ``conftest.py`` files from ``resolved.parent`` up to the enclosing ``tests`` root."""
    repo_root = _REPO_ROOT.resolve()
    repo_tests = (repo_root / "tests").resolve()
    start_dir = resolved.parent
    if start_dir == repo_tests or repo_tests in resolved.parents:
        stop_dir = repo_tests
    elif start_dir == repo_root or repo_root in resolved.parents:
        stop_dir = repo_root
    else:
        tests_dirs = [d for d in (start_dir, *resolved.parents) if d.name == "tests"]
        stop_dir = tests_dirs[0] if tests_dirs else start_dir

    out = []
    cur = start_dir
    while True:
        cand = cur / "conftest.py"
        if cand.is_file():
            out.append(cand)
        if cur == stop_dir or cur.parent == cur:
            break
        cur = cur.parent
    return out


@functools.lru_cache(maxsize=None)
def _cached_item_via(path, func_name=None, cls_name=None, fixture_names=()):
    """Reference modules and subprocess workers used by a single pytest item (top-level + reachable helpers/fixtures)."""
    if func_name is None:
        return _cached_module_via(path)
    resolved, tree = _parsed_module(str(path))
    if tree is None:
        return (_UNRESOLVED_IMPORT,) if _keep_reference(_UNRESOLVED_IMPORT) else ()
    stmts, needed_fixtures = _reachable_local_stmts_and_fixtures(
        tree,
        func_name,
        cls_name=cls_name,
        fixture_names=fixture_names,
        include_autouse=True,
        include_hooks=True,
        is_pytest_target=True,
    )
    if stmts is None:
        found = set(_cached_module_via(path))
        needed_fixtures = set(fixture_names)
    else:
        found = set(_scan_stmts_via(stmts, tree, pathlib.Path(path), frozenset({(resolved, cls_name, func_name)})))

    in_file_fixtures = {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_fixture_def(n)
    }
    scanned_test_fixtures = set(needed_fixtures) & in_file_fixtures

    conftest_trees = []
    scanned_cdefs = set()
    for conftest_py in _ancestor_conftest_files(resolved):
        c_res, ctree = _parsed_module(str(conftest_py))
        if ctree is None:
            if _keep_reference(_UNRESOLVED_IMPORT):
                found.add(_UNRESOLVED_IMPORT)
            continue
        conftest_trees.append((conftest_py, c_res, ctree))
        c_init_stmts, c_init_req = _reachable_local_stmts_and_fixtures(
            ctree,
            None,
            include_autouse=True,
            include_hooks=True,
            is_pytest_target=True,
        )
        needed_fixtures.update(c_init_req)
        for s in ctree.body:
            if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                _is_autouse_fixture(s) or s.name.startswith("pytest_")
            ):
                scanned_cdefs.add((c_res, s.name))
        if c_init_stmts is not None:
            found.update(_scan_stmts_via(c_init_stmts, ctree, conftest_py, frozenset({(c_res, "<top>")})))

    changed = True
    while changed:
        changed = False
        if stmts is not None:
            newly_needed_in_test = (needed_fixtures & in_file_fixtures) - scanned_test_fixtures
            if newly_needed_in_test:
                scanned_test_fixtures.update(newly_needed_in_test)
                changed = True
                t_stmts, t_req = _reachable_local_stmts_and_fixtures(
                    tree,
                    func_name,
                    cls_name=cls_name,
                    fixture_names=tuple(sorted(needed_fixtures)),
                    include_autouse=True,
                    include_hooks=True,
                    is_pytest_target=True,
                )
                needed_fixtures.update(t_req)
                if t_stmts is not None:
                    found.update(
                        _scan_stmts_via(t_stmts, tree, pathlib.Path(path), frozenset({(resolved, cls_name, func_name)}))
                    )
        for conftest_py, c_res, ctree in conftest_trees:
            for s in ctree.body:
                if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if (c_res, s.name) in scanned_cdefs:
                    continue
                if s.name in needed_fixtures and _is_fixture_def(s):
                    scanned_cdefs.add((c_res, s.name))
                    changed = True
                    c_stmts, c_req = _reachable_local_stmts_and_fixtures(
                        ctree,
                        s.name,
                        include_autouse=False,
                        is_pytest_target=True,
                    )
                    needed_fixtures.update(c_req)
                    if c_stmts is None:
                        found.update(_scan_module_via(conftest_py, frozenset()))
                    else:
                        found.update(_scan_stmts_via(c_stmts, ctree, conftest_py, frozenset({(c_res, s.name)})))
    return tuple(sorted(found))


def _item_via(path, func_name=None, cls_name=None, fixture_names=()):
    """Reference modules and subprocess workers used by a single pytest item (top-level + reachable helpers/fixtures)."""
    return _cached_item_via(path, func_name, cls_name, tuple(fixture_names))


def _module_via(path):
    """Reference modules and subprocess workers a file imports or invokes, anywhere in it."""
    return _cached_module_via(path)


def _clear_via_caches():
    _parsed_module.cache_clear()
    _init_exports.cache_clear()
    _has_executable_defs.cache_clear()
    _repo_py_suffixes.cache_clear()
    _exported_sp_wrappers_cached.cache_clear()
    _exported_dyn_imports_cached.cache_clear()
    _cached_module_via.cache_clear()
    _cached_item_via.cache_clear()


_module_via.cache_clear = _clear_via_caches
_item_via.cache_clear = _clear_via_caches


@functools.lru_cache(maxsize=1)
def _golden_names():
    """Names listed in trust/golden.yml (the only golden roots that count)."""
    import yaml

    path = pathlib.Path(__file__).resolve().parents[2] / "trust" / "golden.yml"
    return frozenset(yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.exists() else frozenset()


def _claims(item, runtime=frozenset()):
    func_name = getattr(item, "originalname", None) or getattr(getattr(item, "obj", None), "__name__", None)
    cls_obj = getattr(item, "cls", None)
    cls_name = cls_obj.__name__ if cls_obj is not None else None
    fixture_names = tuple(sorted(getattr(item, "fixturenames", ()) or ()))
    base_via = set(_via.static_via(str(item.path))) | set(runtime)
    old_via = set(_item_via(str(item.path), func_name, cls_name, fixture_names)) if ACCEPTANCE else None
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
        via = base_via | set(mark.kwargs.get("via", ()))
        prop = mark.kwargs.get("property", ())
        claim = {"objects": objects, "roots": roots, "kind": kind, "tier": tier,
                 "via": sorted(via - set(objects)),
                 "properties": [prop] if isinstance(prop, str) else list(prop)}
        if old_via is not None:
            claim["via_old"] = sorted((old_via | set(mark.kwargs.get("via", ()))) - set(objects))
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
    report.validates = _claims(item, _via.recorded_via(item))
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
    is_module = isinstance(collector, pytest.Module)
    if is_module:
        _via.set_collecting(collector.path)
    outcome = yield
    if is_module:
        _via.set_collecting(None)
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
