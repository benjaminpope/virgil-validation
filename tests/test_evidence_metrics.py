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
