"""evidence.plugin.metrics_of: a metric name recorded twice keeps both values."""
import pytest

from evidence.plugin import metrics_of


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
def test_repeated_metric_names_keep_every_value():
    props = [("err", 1.0), ("other", 5.0), ("err", 2.0), ("err", "inf")]
    assert metrics_of(props) == {"err": [1.0, 2.0, "inf"], "other": 5.0}
    assert dict(props)["err"] == "inf"  # what dict() did: the earlier values were lost


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
def test_unrepeated_metrics_stay_scalars():
    assert metrics_of([("a", 1.0), ("b", 2.0)]) == {"a": 1.0, "b": 2.0}
    assert metrics_of([]) == {}


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
def test_the_trust_headline_reduces_a_repeated_metric_to_its_worst_case():
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    import trust

    rec = {"metrics": {"rel_err": [1e-9, -3e-7, 2e-8], "n": 3.0}}
    assert trust._headline(rec) == ("rel_err", -3e-7)
    assert trust._headline(dict(rec, headline="rel_err")) == ("rel_err", -3e-7)
    assert trust._headline({"metrics": {"rel_err": [1e-9, "inf"]}, "headline": "rel_err"}) == ("rel_err", "inf")
    assert trust._headline({"metrics": {"rel_err": 1e-9}}) == ("rel_err", 1e-9)  # a scalar is unchanged


def _run_inner(tmp_path, body):
    """Run a one-test file in a subprocess with the evidence plugin and return (returncode, output)."""
    import os
    import pathlib
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parents[1]
    (tmp_path / "test_inner.py").write_text("import pytest\n\n" + body)
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "PYTEST_ADDOPTS": ""}
    run = subprocess.run([sys.executable, "-m", "pytest", "-c", "/dev/null", "--rootdir", str(tmp_path),
                          "-p", "no:cacheprovider", str(tmp_path / "test_inner.py")],
                         capture_output=True, text=True, env=env, cwd=tmp_path)
    return run.returncode, run.stdout + run.stderr


_INNER = '''@pytest.mark.validates("evidence", roots=["mathematics"], headline="{headline}")
def test_inner(metric):
    metric("rel_a", 1e-9)
'''


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
def test_a_passing_test_whose_explicit_headline_was_not_recorded_fails(tmp_path):
    code, out = _run_inner(tmp_path, _INNER.format(headline="rel_b"))
    assert code != 0 and "not a recorded metric" in out


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
def test_a_test_that_recorded_its_explicit_headline_passes(tmp_path):
    code, out = _run_inner(tmp_path, _INNER.format(headline="rel_a"))
    assert code == 0 and "not a recorded metric" not in out
