"""Which of our reference nodes does a test go through? (a record's ``via``)

``scripts/trust.py`` counts a passing check of a node only if every node in
its ``via`` is itself verified. ``via`` gates only checks that passed, and a
check that passed was executed, so what the check ran is known exactly at run
time. Two sources, then, and no deep static analysis:

* **Run time (complete):** ``external_bridge._subprocess.run`` is the single
  place a worker script is started. A wrapper installed at ``pytest_configure``
  records the worker's file name, as the node ``external_bridge.<stem>``, for
  the running test, for every test that uses a fixture that started it, and
  for every test in a file that started it while being collected.
* **Imports (complete for what ran):** ``builtins.__import__``,
  ``importlib.import_module`` and ``importlib.util.spec_from_file_location``
  are wrapped likewise, so a reference imported by a script that a test loads
  by file path, by a helper, or by name, is noted when the statement runs,
  even if an earlier test already imported the module. A script loaded by
  path also counts as what its file scan finds, even if its functions are
  never called.
* **Static backstop (coarse):** every reference module that the test file,
  its ancestor ``conftest.py`` files and the local helper modules they import
  (``tests/``, ``scripts/``, ``benchmarks/``) import at module level or in
  helpers, and every string literal naming a worker script there. The bodies
  of ``test_*`` functions are left out: a test that passed ran its body,
  which the recorders saw, and counting every test's body for every test in
  the file would make references that are only checked in one file depend on
  each other. It still over-approximates (module level and helpers count for
  the whole file); where that wrongly blocks a node, split the test file.

Explicit ``via=[...]`` on the marker covers what none of these see: code
started some other way (a subprocess that is not ``_subprocess.run``).
``tests/test_via_declared.py`` fails when a test file loads a reference
package by a computed name without declaring ``via``.
"""

import ast
import collections
import contextlib
import functools
import importlib
import importlib.util
import os
import pathlib
import sys

REFERENCE_PACKAGES = ("crosscheck", "external_bridge", "virgil_bridge")
PLUMBING = frozenset({"external_bridge._subprocess"})  # starts other code, is not itself a reference
IGNORED_REPO_DIRS = frozenset({
    ".git", "__pycache__", ".pytest_cache", ".external", ".claude", "build", "site", "virgil-src", "virgil-ci",
})

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
_REPO_PREFIX = str(REPO_ROOT)
_NOT_LOCAL = frozenset({*REFERENCE_PACKAGES, "virgil", "drpangloss", "jaxbessel", "evidence"})


def is_ignored_repo_part(part):
    return part.startswith(".venv") or part.endswith(".egg-info") or part in IGNORED_REPO_DIRS


# ---------------------------------------------------------------- static backstop


@functools.lru_cache(maxsize=None)
def has_executable_defs(init_py):
    """True when a package ``__init__.py`` defines or binds anything of its own (so the package is a node)."""
    init_py = pathlib.Path(init_py)
    if not init_py.is_file():
        return False
    try:
        tree = ast.parse(init_py.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return True
    for i, stmt in enumerate(tree.body):
        if i == 0 and isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue  # docstring
        if isinstance(stmt, (ast.Pass, ast.Import, ast.ImportFrom)):
            continue
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and ast.unparse(stmt).startswith("__all__"):
            continue
        return True
    return False


def _is_submodule(pkg, name):
    base = SRC_ROOT / pkg
    return (base / f"{name}.py").is_file() or (base / name / "__init__.py").is_file()


def _ref_nodes_of_import(parts, names, own_pkg_parts=None, level=0):
    """Reference nodes named by one import statement. ``parts``: the dotted module as a tuple (empty for
    ``from . import x``); ``names``: imported names of a from-import, else ``()``."""
    if level:
        if own_pkg_parts is None:
            return set()
        parts = own_pkg_parts[: len(own_pkg_parts) - (level - 1)] + tuple(parts)
    if not parts or parts[0] not in REFERENCE_PACKAGES:
        return set()
    found = set()
    top = parts[0]
    if len(parts) == 1:
        if names:  # from crosscheck import sky, chi2
            for name in names:
                if _is_submodule(top, name):
                    found.add(f"{top}.{name}")
                elif has_executable_defs(SRC_ROOT / top / "__init__.py"):
                    found.add(top)  # a name the package defines itself
        elif has_executable_defs(SRC_ROOT / top / "__init__.py"):
            found.add(top)  # import virgil_bridge
    else:
        found.add(".".join(parts))
    return found - PLUMBING


STRING_LOADERS = frozenset({"importorskip", "import_module", "__import__"})


def _loader_name(call):
    return call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", None)


def _own_pkg_parts(path):
    """Package of a file under ``src/<reference package>/``, as a tuple, else None."""
    try:
        rel = path.resolve().relative_to(SRC_ROOT.resolve())
    except ValueError:
        return None
    return rel.parts[:-1] if rel.parts and rel.parts[0] in REFERENCE_PACKAGES else None


def _search_dirs(path):
    bench = REPO_ROOT / "benchmarks"
    dirs = [path.parent, REPO_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "tests", bench]
    if bench.is_dir():
        dirs += [d for d in sorted(bench.iterdir()) if d.is_dir() and not d.name.startswith(".")]
    return [d for d in dirs if d.is_dir()]


def _local_helper(parts, path):
    """A local helper module (outside ``src/``) that an import names, else None."""
    if not parts or parts[0] in _NOT_LOCAL:
        return None
    for base in _search_dirs(path):
        for cand in (base.joinpath(*parts[:-1], f"{parts[-1]}.py"), base.joinpath(*parts, "__init__.py")):
            if cand.is_file():
                return cand.resolve()
    return None


def _worker_nodes(value):
    """``external_bridge.<stem>`` for a string literal that names a worker script in ``src/external_bridge``."""
    if value.endswith(".py") and "/" not in value and value != "__init__.py" and (SRC_ROOT / "external_bridge" / value).is_file():
        return {f"external_bridge.{value[:-3]}"}
    return set()


def _walk(tree, skip_tests):
    """``ast.walk`` that leaves out the bodies (and decorators) of ``test*`` functions when asked."""
    todo = [tree]
    while todo:
        node = todo.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if skip_tests and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith("test"):
                continue
            todo.append(child)


def _sibling_nodes(own, parts):
    """A bare ``import cv_folds`` inside ``src/external_bridge/`` (worker scripts run with that directory on
    ``sys.path``) names the sibling module ``external_bridge.cv_folds``."""
    if own and parts and parts[0] not in REFERENCE_PACKAGES and _is_submodule(own[0], parts[0]):
        return {f"{own[0]}.{parts[0]}"}
    return set()


@functools.lru_cache(maxsize=None)
def _scan_file(path, include_tests=False):
    """(reference nodes this file imports or names, local helper files it imports); a file that cannot be
    read or parsed contributes nothing (collecting it fails the run anyway). The bodies of ``test*``
    functions outside ``src/`` count only with ``include_tests``."""
    path = pathlib.Path(path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return frozenset(), ()
    own = _own_pkg_parts(path)
    found, helpers = set(), []

    def helper(parts, level=0):
        if level:
            cand = _local_helper_relative(path, level, parts)  # relative to the file's own directory
        else:
            cand = _local_helper(parts, path)
        if cand is not None and cand != path.resolve():
            helpers.append(cand)

    for node in _walk(tree, skip_tests=own is None and not include_tests):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = tuple(alias.name.split("."))
                found |= _ref_nodes_of_import(parts, ())
                found |= _sibling_nodes(own, parts)
                if own is None:
                    helper(parts)
        elif isinstance(node, ast.ImportFrom):
            parts = tuple((node.module or "").split(".")) if node.module else ()
            names = tuple(a.name for a in node.names if a.name != "*")
            found |= _ref_nodes_of_import(parts, names, own, node.level)
            if not node.level:
                found |= _sibling_nodes(own, parts)
            if own is None:
                if node.level:
                    helper(parts, node.level)
                    for name in names:
                        helper((*parts, name), node.level)
                else:
                    helper(parts)
                    for name in names:
                        helper((*parts, name))
        elif isinstance(node, ast.Call) and _loader_name(node) in STRING_LOADERS and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                found |= _ref_nodes_of_import(tuple(first.value.split(".")), ())
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            found |= _worker_nodes(node.value)
    return frozenset(found - PLUMBING), tuple(dict.fromkeys(helpers))


def _local_helper_relative(path, level, parts):
    base = path.resolve().parent
    for _ in range(level - 1):
        base = base.parent
    if not parts:
        cand = base / "__init__.py"
        return cand if cand.is_file() else None
    for cand in (base.joinpath(*parts[:-1], f"{parts[-1]}.py"), base.joinpath(*parts, "__init__.py")):
        if cand.is_file():
            return cand.resolve()
    return None


def ancestor_conftests(path):
    """Existing ``conftest.py`` files from the file's directory up to the repository root."""
    out, cur = [], pathlib.Path(path).resolve().parent
    root = REPO_ROOT.resolve()
    while True:
        if (cur / "conftest.py").is_file():
            out.append(cur / "conftest.py")
        if cur == root or cur.parent == cur:
            break
        cur = cur.parent
    return out


@functools.lru_cache(maxsize=None)
def file_via(path, include_tests=False):
    """The reference nodes a file imports or names, with those of the local helper modules it imports
    (transitively). For a source file under ``src/<reference package>/`` this is what it depends on.
    ``include_tests``: the bodies of ``test*`` functions count too (the graph test checks every name)."""
    path = pathlib.Path(path).resolve()
    found, todo, seen = set(), [path], set()
    while todo:
        cur = todo.pop()
        if cur in seen:
            continue
        seen.add(cur)
        refs, helpers = _scan_file(str(cur), include_tests)
        found |= refs
        todo.extend(helpers)
    return frozenset(found)


@functools.lru_cache(maxsize=None)
def static_via(path):
    """Static backstop of a test file: its own imports and those of its ancestor ``conftest.py`` files."""
    out = set(file_via(str(path)))
    for conftest in ancestor_conftests(path):
        out |= file_via(str(conftest))
    return frozenset(out)


def clear_caches():
    for fn in (has_executable_defs, _scan_file, file_via, static_via):
        fn.cache_clear()


# ---------------------------------------------------------------- run-time recording

_STATE = {"item": None, "collecting": None, "fixtures": []}
_BY_ITEM = collections.defaultdict(set)  # nodeid -> worker nodes started while it was set up, run or torn down
_BY_FILE = collections.defaultdict(set)  # file -> worker nodes started while it was collected
_BY_FIXTURE = collections.defaultdict(set)  # FixtureDef -> worker nodes its setup started


def worker_node(worker):
    return f"external_bridge.{pathlib.Path(str(worker)).stem}"


def note(nodes):
    """Credit reference nodes to whatever is running: the test, else the file being collected; and to every
    fixture being set up."""
    if not nodes:
        return
    if _STATE["item"] is not None:
        _BY_ITEM[_STATE["item"]].update(nodes)
    elif _STATE["collecting"] is not None:
        _BY_FILE[_STATE["collecting"]].update(nodes)
    for fixturedef in _STATE["fixtures"]:
        _BY_FIXTURE[fixturedef].update(nodes)


def install_recorder():
    """Wrap ``external_bridge._subprocess.run`` (idempotent). Callers go through the module attribute, so the
    wrapper sees every worker start; one that bound ``run`` earlier would be missed, and the acceptance
    comparison of the pull request that introduced this would have shown it."""
    try:
        from external_bridge import _subprocess
    except ImportError:
        return False
    original = _subprocess.run
    if getattr(original, "_vv_recording", False):
        return True

    @functools.wraps(original)
    def run(*args, **kwargs):
        worker = args[1] if len(args) > 1 else kwargs.get("worker")
        if worker is not None:
            note({worker_node(worker)})
        return original(*args, **kwargs)

    run._vv_recording = True
    _subprocess.run = run
    return True


@functools.lru_cache(maxsize=None)
def _import_nodes(name, fromlist):
    return frozenset(_ref_nodes_of_import(tuple(name.split(".")), fromlist))


def _local_module_via(name, fromlist=()):
    """The reference nodes a local helper module (``tests/``, ``scripts/``, ``benchmarks/``) imports, by the
    file scan, when an import statement names it (or, for ``from pkg import helper``, names a helper of the
    package). Its own import statements ran when it was first imported, possibly for an earlier test; each
    test that imports it again relies on what it imports."""
    found = set()
    for dotted in (name, *(f"{name}.{sub}" for sub in fromlist or () if isinstance(sub, str))):
        module = sys.modules.get(dotted)
        file = getattr(module, "__file__", None)
        if file and file.endswith(".py"):
            found |= _local_file_via(file)
    return frozenset(found)


@functools.lru_cache(maxsize=None)
def _local_file_via(file):
    path = pathlib.Path(file).resolve()
    rel = path.relative_to(REPO_ROOT.resolve()) if path.is_relative_to(REPO_ROOT.resolve()) else None
    if rel is None or any(is_ignored_repo_part(part) for part in rel.parts) or rel.parts[0] == "src":
        return frozenset()
    return file_via(str(path))


def _is_reference_name(name):
    return name.split(".", 1)[0] in REFERENCE_PACKAGES


def _absolute_name(name, level, globals):
    """The absolute dotted name of a relative import made from a module with these globals, else None."""
    globals = globals or {}
    package = globals.get("__package__") or globals.get("__name__", "").rpartition(".")[0]
    if not package:
        return None
    bits = package.rsplit(".", level - 1)
    if len(bits) < level:
        return None
    return f"{bits[0]}.{name}" if name else bits[0]


def _loaded_file_nodes(location):
    """What a file loaded by ``spec_from_file_location`` counts as: a reference source file is its node, any
    other repository file is what its file scan finds (the whole file, not only what runs)."""
    path = pathlib.Path(os.fspath(location)).resolve()
    own = _own_pkg_parts(path) if path.is_relative_to(SRC_ROOT.resolve()) else None
    if own is not None:
        if path.name == "__init__.py":
            return frozenset({".".join(own)}) if has_executable_defs(path) else frozenset()
        return frozenset({".".join((*own, path.stem))}) - PLUMBING
    return _local_file_via(str(path))


def install_import_recorder():
    """Wrap ``builtins.__import__``, ``importlib.import_module`` and ``importlib.util.spec_from_file_location``
    (idempotent) to note the reference nodes every import executed while a test, fixture or collection runs
    names, wherever the statement is: in a script that a test loads by file path, a helper, or the test
    itself. Statements run again each time a loaded script is executed, so a reference that an earlier test
    already imported is still noted. Not seen: code that a test starts in a subprocess other than through
    ``_subprocess.run`` (declare ``via``)."""
    import builtins

    original = builtins.__import__
    if getattr(original, "_vv_recording", False):
        return
    original_import_module = importlib.import_module
    original_spec = importlib.util.spec_from_file_location

    def recording_import(name, globals=None, locals=None, fromlist=(), level=0):
        resolved = name if not level else _absolute_name(name, level, globals)
        if resolved is None:
            return original(name, globals, locals, fromlist, level)
        if _is_reference_name(resolved):
            # inside a reference source file an import is an edge of the graph (``depends``), not the test's
            if not level or not _is_reference_name((globals or {}).get("__name__", "")):
                note(_import_nodes(resolved, tuple(fromlist or ())))
            return original(name, globals, locals, fromlist, level)
        result = original(name, globals, locals, fromlist, level)
        note(_local_module_via(resolved, tuple(fromlist or ())))
        return result

    @functools.wraps(original_import_module)
    def recording_import_module(name, package=None):
        try:
            resolved = importlib.util.resolve_name(name, package) if name.startswith(".") else name
        except (ValueError, TypeError):
            resolved = None
        if resolved is not None and _is_reference_name(resolved):
            note(_import_nodes(resolved, ()))
            return original_import_module(name, package)
        result = original_import_module(name, package)
        if resolved is not None:
            note(_local_module_via(resolved))
        return result

    @functools.wraps(original_spec)
    def recording_spec(name, location=None, *args, **kwargs):
        if location is not None:
            try:
                note(_loaded_file_nodes(location))
            except (TypeError, ValueError, OSError):
                pass
        return original_spec(name, location, *args, **kwargs)

    recording_import._vv_recording = True
    importlib.import_module = recording_import_module
    importlib.util.spec_from_file_location = recording_spec
    builtins.__import__ = recording_import


def set_item(nodeid):
    _STATE["item"] = nodeid


@contextlib.contextmanager
def running(nodeid):
    """Run a block as if ``nodeid`` were the running test (for tests of this module): what is recorded for it
    is yielded as a set and credited to the real running test too, and nothing is left behind."""
    previous = _STATE["item"]
    _STATE["item"] = nodeid
    got = set()
    try:
        yield got
    finally:
        _STATE["item"] = previous
        got |= _BY_ITEM.pop(nodeid, set())
        if previous is not None:
            _BY_ITEM[previous] |= got


def set_collecting(path):
    """Set the file being collected (None for none); returns what was set before, to restore."""
    previous = _STATE["collecting"]
    _STATE["collecting"] = None if path is None else str(path)
    return previous


def push_fixture(fixturedef):
    _STATE["fixtures"].append(fixturedef)


def pop_fixture():
    if _STATE["fixtures"]:
        _STATE["fixtures"].pop()


def recorded_via(item):
    """Worker nodes a test went through: started in its own setup, call or teardown, in the setup of any
    fixture it used (a module- or session-scoped fixture is built once, by the first test that needs it, but
    every test that uses it relies on it), or while its file was collected."""
    found = set(_BY_ITEM.get(item.nodeid, ())) | _BY_FILE.get(str(item.path), set())
    fixture_defs = getattr(getattr(item, "_request", None), "_fixture_defs", None) or {}
    for fixturedef in fixture_defs.values():
        found |= _BY_FIXTURE.get(fixturedef, set())
    return found
