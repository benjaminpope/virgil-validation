"""Which of our reference nodes does a test go through? (a record's ``via``)

``scripts/trust.py`` counts a passing check of a node only if every node in
its ``via`` is itself verified. ``via`` gates only checks that passed, and a
check that passed was executed, so what the check ran is known exactly at run
time. Two sources, then, and no deep static analysis:

* **Run time (complete):** ``external_bridge._subprocess.run`` is the single
  place a worker script is started. A wrapper installed at ``pytest_configure``
  records the worker's file name, as the node ``external_bridge.<stem>``, for
  the running test, for every test that uses a fixture that started it, and for
  every test in a file that started it while being collected.
* **Imports (complete for what ran):** ``builtins.__import__`` is wrapped likewise, so a reference imported by a
  script that a test loads by file path (``spec_from_file_location``), or inside any helper, is noted when the
  statement runs, even if an earlier test already imported the module.
* **Static backstop (coarse):** every reference module that the test file, its
  ancestor ``conftest.py`` files and the local helper modules they import
  (``tests/``, ``scripts/``, ``benchmarks/``) import anywhere, and every string
  literal naming a worker script. It over-approximates (the whole file counts,
  used or not); where that wrongly blocks a node, split the test file.

Explicit ``via=[...]`` on the marker covers what neither sees: a reference
loaded by name at run time (``importlib``, ``__import__``,
``spec_from_file_location``). ``tests/test_via_declared.py`` fails when a test
file does that to a reference package without declaring ``via``.
"""

import ast
import collections
import functools
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


@functools.lru_cache(maxsize=None)
def _scan_file(path):
    """(reference nodes this file imports or names, local helper files it imports); a file that cannot be
    read or parsed contributes nothing (collecting it fails the run anyway)."""
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

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = tuple(alias.name.split("."))
                found |= _ref_nodes_of_import(parts, ())
                if own is None:
                    helper(parts)
        elif isinstance(node, ast.ImportFrom):
            parts = tuple((node.module or "").split(".")) if node.module else ()
            names = tuple(a.name for a in node.names if a.name != "*")
            found |= _ref_nodes_of_import(parts, names, own, node.level)
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
def file_via(path):
    """The reference nodes a file imports or names, with those of the local helper modules it imports
    (transitively). For a source file under ``src/<reference package>/`` this is what it depends on."""
    path = pathlib.Path(path).resolve()
    found, todo, seen = set(), [path], set()
    while todo:
        cur = todo.pop()
        if cur in seen:
            continue
        seen.add(cur)
        refs, helpers = _scan_file(str(cur))
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


def _local_module_via(name):
    """The reference nodes a local helper module (``tests/``, ``scripts/``, ``benchmarks/``) imports, by the
    file scan, when an import statement names it. Its own import statements ran when it was first imported,
    possibly for an earlier test; each test that imports it again relies on what it imports."""
    module = sys.modules.get(name)
    file = getattr(module, "__file__", None)
    if not file or not file.endswith(".py") or not file.startswith(_REPO_PREFIX):
        return frozenset()
    return _local_file_via(file)


@functools.lru_cache(maxsize=None)
def _local_file_via(file):
    path = pathlib.Path(file).resolve()
    rel = path.relative_to(REPO_ROOT.resolve()) if path.is_relative_to(REPO_ROOT.resolve()) else None
    if rel is None or any(is_ignored_repo_part(part) for part in rel.parts) or rel.parts[0] == "src":
        return frozenset()
    return file_via(str(path))


def install_import_recorder():
    """Wrap ``builtins.__import__`` (idempotent) to note the reference nodes every import statement executed
    while a test, fixture or collection runs names, wherever that statement is: in a script that a test loads
    by file path, a helper, or the test itself. Statements run again each time a loaded script is executed,
    so a reference that an earlier test already imported is still noted. (``importlib.import_module`` does not
    pass through here; a literal one is read by the file scan, any other must declare ``via``.)"""
    import builtins

    original = builtins.__import__
    if getattr(original, "_vv_recording", False):
        return

    def recording_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0:
            if name.startswith(REFERENCE_PACKAGES):
                note(_import_nodes(name, tuple(fromlist or ())))
            else:
                result = original(name, globals, locals, fromlist, level)
                note(_local_module_via(name))
                return result
        return original(name, globals, locals, fromlist, level)

    recording_import._vv_recording = True
    builtins.__import__ = recording_import


def set_item(nodeid):
    _STATE["item"] = nodeid


def set_collecting(path):
    _STATE["collecting"] = None if path is None else str(path)


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
