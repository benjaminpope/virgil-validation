"""pytest plugin: the ``validates`` marker, the ``metric`` fixture and the
``--evidence PATH`` option (see evidence/__init__.py)."""

import ast
import functools
import json
import math

import pytest

from . import KINDS, ROOTS
from .meta import run_header

_RECORDS = pytest.StashKey[list]()
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
        claim = {"objects": objects, "roots": roots, "kind": kind, "tier": tier,
                 "via": sorted(via - set(objects))}
        if headline is not None:
            claim["headline"] = str(headline)
        claims.append(claim)
    return claims


def pytest_collection_modifyitems(config, items):
    # fail early on malformed markers, even without --evidence
    missing = [item.nodeid for item in items if not _claims(item)]
    if missing and config.getoption("--require-validates"):
        raise pytest.UsageError(
            "tests without a validates marker (docs/method/index.md):\n  "
            + "\n  ".join(missing)
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
                    "metrics": dict(report.user_properties),
                    **getattr(report, "validates_where", {}),
                }
            )


def pytest_sessionfinish(session):
    path = session.config.getoption("--evidence")
    if not path:
        return
    with open(path, "w") as f:
        f.write(json.dumps(run_header()) + "\n")
        for record in session.config.stash[_RECORDS]:
            f.write(json.dumps(record) + "\n")
