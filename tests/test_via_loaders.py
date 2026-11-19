"""Loaders of reference code by name or path (``importlib.import_module``, ``spec_from_file_location``) are
recorded at run time (src/evidence/via.py). The markers here declare ``via`` for what the probes load, as
``tests/test_via_declared.py`` demands of a file that loads reference packages by a computed name."""

import sys

import pytest

from evidence import via
from test_via import ROOT, _write


@pytest.mark.validates(
    "evidence", roots=["standards"], kind="guard",
    via=["external_bridge.alma_worker", "crosscheck.sky", "crosscheck.limb", "crosscheck.radial", "crosscheck.chi2"],
)
def test_loaders_by_name_and_path_are_recorded(tmp_path, monkeypatch):
    """``importlib.import_module`` (also relative), a reference source loaded by path, a script loaded by
    path (credited with what its file scan finds, though no function of it runs), ``from helper_pkg import
    helper`` served from ``sys.modules``, and a relative import inside a local package."""
    import importlib
    import importlib.util

    via.clear_caches()
    via.install_import_recorder()
    script = _write(ROOT / "tests" / "_via_probe_script.py", "def run():\n    from crosscheck import chi2\n")
    pkg = ROOT / "tests" / "_via_probe_pkg"
    _write(pkg / "__init__.py", "")
    _write(pkg / "inner.py", "from crosscheck import radial\n")
    _write(pkg / "outer.py", "from . import inner\n")
    try:
        monkeypatch.syspath_prepend(str(ROOT / "tests"))
        with via.running("tests/test_fake.py::names") as got:
            importlib.import_module("crosscheck.sky")
            importlib.import_module(".limb", "crosscheck")
            importlib.import_module("json")
        assert got == {"crosscheck.sky", "crosscheck.limb"}
        with via.running("tests/test_fake.py::path") as got:
            importlib.util.spec_from_file_location("w", ROOT / "src" / "external_bridge" / "alma_worker.py")
            importlib.util.spec_from_file_location("s", str(script))
            importlib.util.spec_from_file_location("p", ROOT / "src" / "crosscheck" / "sky.py")
        assert got == {"external_bridge.alma_worker", "crosscheck.chi2", "crosscheck.sky"}
        for nodeid in ("tests/test_fake.py::first", "tests/test_fake.py::second"):
            with via.running(nodeid) as got:
                from _via_probe_pkg import inner  # noqa: F401  (cached after the first)
            assert got == {"crosscheck.radial"}, nodeid
        sys.modules.pop("_via_probe_pkg.outer", None)
        with via.running("tests/test_fake.py::relative") as got:
            import _via_probe_pkg.outer  # noqa: F401  (``from . import inner`` inside it)
        assert got == {"crosscheck.radial"}
    finally:
        for name in [n for n in sys.modules if n == "_via_probe_pkg" or n.startswith("_via_probe_pkg.")]:
            sys.modules.pop(name)
        script.unlink()
        for f in pkg.iterdir():
            f.unlink()
        pkg.rmdir()
        via.clear_caches()
