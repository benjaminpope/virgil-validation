"""pytest plugin: the ``validates`` marker, the ``metric`` fixture and the
``--evidence PATH`` option (see evidence/__init__.py)."""

import ast
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
    item = _CONFIG.get("item")
    if item is None:
        return value
    value = float(value)
    item.user_properties.append((name, value if math.isfinite(value) else str(value)))
    return value


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
# that module is itself verified. _subprocess only runs other code.
_REFERENCE_PACKAGES = ("crosscheck", "external_bridge")
_PLUMBING = {"external_bridge._subprocess"}


@functools.lru_cache(maxsize=None)
def _module_via(path):
    """Reference modules a test file imports, anywhere in it."""
    try:
        tree = ast.parse(open(path).read())
    except (OSError, SyntaxError):
        return ()
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _REFERENCE_PACKAGES and "." in alias.name:
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            top = node.module.split(".")[0]
            if top not in _REFERENCE_PACKAGES:
                continue
            if node.module == top:  # from crosscheck import sky, chi2
                found |= {f"{top}.{alias.name}" for alias in node.names}
            else:  # from crosscheck.sky import vis_point
                found.add(node.module)
    return tuple(sorted(found - _PLUMBING))


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
        elif has_finding:
            bad_xfail.append(
                f"{item.nodeid}: kind='finding' or 'upstream' requires @pytest.mark.xfail(strict=True)"
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
        record_property(name, value if math.isfinite(value) else str(value))
        return value

    return record


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    report.validates = _claims(item)
    report.validates_where = _where(item)


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
        session.config.stash[_HEADER] = run_header()


def pytest_sessionfinish(session):
    path = session.config.getoption("--evidence")
    if not path:
        return
    header = session.config.stash.get(_HEADER, None) or run_header()
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
