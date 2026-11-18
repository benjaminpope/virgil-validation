"""The ``via`` of a record: a file-level import scan as a coarse backstop, and the worker scripts a test really
started, recorded at ``external_bridge._subprocess.run`` (src/evidence/via.py)."""

import json
import os
import pathlib
import subprocess
import sys

import pytest

from evidence import via

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_file_scan_finds_reference_imports_anywhere_in_a_file(tmp_path):
    """Plain, from- and function-local imports count; ``_subprocess`` is plumbing; a name the package
    defines itself is the package; other packages are ignored."""
    via.clear_caches()
    f = _write(
        tmp_path / "test_a.py",
        "import crosscheck.sky\n"
        "from crosscheck import chi2, nothing_here\n"
        "from crosscheck.limb import x\n"
        "from external_bridge import _subprocess as sp\n"
        "from virgil_bridge import Scene\n"
        "import numpy, virgil.models\n"
        "def test_x():\n"
        "    from external_bridge.cv_folds import f\n"
        "    vb = pytest.importorskip('virgil_bridge.alma')\n"
        "    importlib.import_module('crosscheck.radial')\n"
        "    importlib.import_module(name)\n",
    )
    assert via.static_via(str(f)) == {
        "crosscheck.sky", "crosscheck.chi2", "crosscheck.limb", "virgil_bridge", "external_bridge.cv_folds",
        "virgil_bridge.alma", "crosscheck.radial",
    }


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_file_scan_names_workers_from_string_literals_and_follows_local_helpers_and_conftests(tmp_path):
    """A string that names a worker script is the worker's node; local helper modules (also relative ones)
    and ancestor conftests count as part of the file."""
    via.clear_caches()
    _write(tmp_path / "pkg" / "conftest.py", "from crosscheck import radial\n")
    _write(tmp_path / "pkg" / "_helper.py", "from crosscheck import sky\nimport _deeper\n")
    _write(tmp_path / "pkg" / "_deeper.py", "W = 'mpol_worker.py'\n")
    _write(tmp_path / "pkg" / "_rel.py", "from crosscheck import array\n")
    f = _write(
        tmp_path / "pkg" / "test_b.py",
        "import _helper\nfrom . import _rel\nOTHER = 'not_a_worker.py'\n",
    )
    assert via.file_via(str(f)) == {"crosscheck.sky", "crosscheck.array", "external_bridge.mpol_worker"}
    assert via.static_via(str(f)) == via.file_via(str(f)) | {"crosscheck.radial"}


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_file_scan_of_reference_sources_resolves_relative_imports():
    """Relative imports inside ``src/<reference package>/`` name nodes (the graph test checks ``depends``)."""
    via.clear_caches()
    got = via.file_via(str(ROOT / "src" / "external_bridge" / "cv_compare.py"))
    assert "external_bridge.cv_folds" in got
    assert not got & via.PLUMBING


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_import_recorder_notes_reference_imports_each_time_the_statement_runs():
    """A script loaded by path that imports a reference names it for the running test, even when the module
    was imported before; other packages are not noted."""
    via.install_import_recorder()
    via.set_item("tests/test_fake.py::test_two")
    try:
        from crosscheck import sky  # noqa: F401
        from crosscheck import sky  # noqa: F401,F811  (imported already: still noted)
        import crosscheck.chi2  # noqa: F401
        import json  # noqa: F401
        from external_bridge import _subprocess  # noqa: F401
    finally:
        via.set_item(None)
    assert via._BY_ITEM["tests/test_fake.py::test_two"] == {"crosscheck.sky", "crosscheck.chi2"}


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_worker_recorder_wraps_the_single_choke_point_and_notes_the_worker_node(monkeypatch):
    """Every worker start is noted for the running test, positional or keyword, and still runs."""
    from external_bridge import _subprocess as sp

    calls = []
    monkeypatch.setattr(sp, "run", lambda name, worker, task, timeout=0: calls.append(worker) or {"ok": 1})
    assert via.install_recorder()
    via.set_item("tests/test_fake.py::test_one")
    try:
        assert sp.run("alma", "alma_worker.py", {}) == {"ok": 1}
        sp.run("mpol", worker="mpol_worker.py", task={})
    finally:
        via.set_item(None)
    assert calls == ["alma_worker.py", "mpol_worker.py"]
    assert via._BY_ITEM["tests/test_fake.py::test_one"] == {"external_bridge.alma_worker", "external_bridge.mpol_worker"}
    # installing twice wraps once
    wrapped = sp.run
    via.install_recorder()
    assert sp.run is wrapped


INNER = '''
import pytest
from external_bridge import _subprocess as sp

def start(worker):
    try:
        sp.run("no_such_env", worker, {})
    except Exception:
        pass

@pytest.fixture(scope="module")
def shared():
    start("alma_" + "worker.py")  # not a literal: only the recorder can know

@pytest.mark.validates("evidence", roots=["standards"])
def test_first(shared):
    pass

@pytest.mark.validates("evidence", roots=["standards"])
def test_second(shared):
    pass

@pytest.mark.validates("evidence", roots=["standards"])
def test_own_call():
    start("cv_" + "folds.py")

@pytest.mark.validates("evidence", roots=["standards"], via=["crosscheck.declared"])
def test_declared():
    pass

@pytest.mark.validates("evidence", roots=["standards"])
def test_nothing():
    pass
'''


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_recorded_workers_reach_the_evidence_of_every_test_that_relied_on_them(tmp_path):
    """In a real run: a worker started in a module-scoped fixture is in the ``via`` of every test using the
    fixture (not only the first), one started in a test body only in that test, and a declared ``via`` is kept."""
    _write(tmp_path / "test_inner.py", INNER)
    evidence = tmp_path / "evidence.jsonl"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--evidence", str(evidence), "test_inner.py"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    records = {r["test"].split("::")[-1]: r["via"] for r in map(json.loads, evidence.read_text().splitlines()[1:])}
    assert records == {
        "test_first": ["external_bridge.alma_worker"],
        "test_second": ["external_bridge.alma_worker"],
        "test_own_call": ["external_bridge.cv_folds"],
        "test_declared": ["crosscheck.declared"],
        "test_nothing": [],
    }
