"""A campaign test stamps its summary's commits into its evidence record (evidence.plugin.stamp_campaign), and a summary
that is not gate evidence skips."""
import json
import subprocess
import sys
import textwrap

import pytest

V, C = "a" * 40, "b" * 40

INNER = """
import json, pytest
from evidence.plugin import stamp_campaign

@pytest.mark.validates("virgil.x", roots=["statistics"], tier="C")
def test_campaign():
    stamp_campaign(json.loads(open("summary.json").read()))
"""


def run(tmp_path, summary):
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "test_inner.py").write_text(textwrap.dedent(INNER))
    out = tmp_path / "evidence.jsonl"
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--evidence", str(out),
                    str(tmp_path / "test_inner.py")], cwd=tmp_path, check=False, capture_output=True)
    return [json.loads(line) for line in out.read_text().splitlines()][1:]


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
def test_a_summary_with_commits_is_stamped_into_the_record(tmp_path):
    (rec,) = run(tmp_path, {"virgil_commit": V, "criteria_commit": C})
    assert rec["outcome"] == "passed" and rec["virgil_commit"] == V and rec["criteria_commit"] == C


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
@pytest.mark.parametrize("summary", [{"virgil_commit": V}, {"criteria_commit": C}, {"virgil_commit": V, "criteria_commit": None}])
def test_a_summary_without_both_commits_skips_and_is_not_stamped(tmp_path, summary):
    (rec,) = run(tmp_path, summary)
    assert rec["outcome"] == "skipped" and "virgil_commit" not in rec and "criteria_commit" not in rec
