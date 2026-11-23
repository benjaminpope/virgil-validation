"""scripts/fetch_ci_evidence.sh against a stub ``gh``: a cancelled or timed-out run uploads no
evidence artefact, so the script must fall back to the newest older run that has one, say so on
stderr for each run it skips, and fail only when none has one."""

import os
import pathlib
import stat
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fetch_ci_evidence.sh"

# The stub answers the three gh calls the script makes. FAKE_RUNS_<event> holds "createdAt id" lines
# (what the real call prints after its --jq), FAKE_WITH_ARTEFACT the ids of the runs that have an
# evidence artefact (FAKE_WITH_EVIDENCE: those whose artefact holds evidence.jsonl; a cancelled run's holds only mem.log); a download writes the run's id into DIR/evidence.jsonl.
STUB = r"""#!/usr/bin/env bash
case "$1 $2" in
  "run list")
    while [ $# -gt 0 ]; do [ "$1" = -e ] && event=$2; [ "$1" = -L ] && limit=$2; shift; done
    var=FAKE_RUNS_$event
    printf '%s\n' "${!var}" | sed '/^$/d' | sort -r | sed -n "1,${limit}p" ;;
  "api repos/"*)
    run=${2#*/runs/}; run=${run%/artifacts}
    for ok in $FAKE_WITH_ARTEFACT; do [ "$ok" = "$run" ] && echo "evidence-push-$run"; done ;;
  "run download")
    run=$3; while [ $# -gt 0 ]; do [ "$1" = -D ] && dir=$2; shift; done
    mkdir -p "$dir"; echo mem > "$dir/mem.log"
    for ok in $FAKE_WITH_EVIDENCE; do [ "$ok" = "$run" ] && echo "run $run" > "$dir/evidence.jsonl"; done ;;
esac
exit 0
"""


@pytest.fixture
def run_script(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    gh = bindir / "gh"
    gh.write_text(STUB)
    gh.chmod(gh.stat().st_mode | stat.S_IXUSR)

    def run(*args, push=(), full=(), with_artefact=(), cancelled=()):
        env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "GITHUB_REPOSITORY": "o/r",
               "FAKE_RUNS_push": "\n".join(push),
               "FAKE_RUNS_schedule": "\n".join(full),
               "FAKE_RUNS_workflow_dispatch": "",
               "FAKE_WITH_ARTEFACT": " ".join(map(str, [*with_artefact, *cancelled])),
               "FAKE_WITH_EVIDENCE": " ".join(map(str, with_artefact))}
        out = tmp_path / "ci"
        proc = subprocess.run(["bash", str(SCRIPT), *args, str(out)], env=env, capture_output=True, text=True)
        return proc, out

    return run


PUSH = ["2026-11-23T00:00:00Z 300", "2026-11-22T00:00:00Z 200", "2026-11-21T00:00:00Z 100"]
FULL = ["2026-11-16T00:00:00Z 40", "2026-11-09T00:00:00Z 30"]


@pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")
def test_newest_run_without_an_artefact_falls_back_to_the_next_older_run(run_script):
    proc, out = run_script(push=PUSH, full=FULL, with_artefact=[100, 30], cancelled=[200, 40])
    assert proc.returncode == 0, proc.stderr
    assert (out / "push" / "evidence.jsonl").read_text() == "run 100\n"
    assert (out / "full" / "evidence.jsonl").read_text() == "run 30\n"
    skipped = [line for line in proc.stderr.splitlines() if "has no evidence artefact; using an older run" in line]
    assert skipped == [
        "run 300 has no evidence artefact; using an older run",
        "run 200 has no evidence artefact; using an older run",
        "run 40 has no evidence artefact; using an older run",
    ]


@pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")
def test_the_newest_run_is_used_when_it_has_an_artefact_and_the_layout_is_unchanged(run_script):
    proc, out = run_script(push=PUSH, full=FULL, with_artefact=[300, 200, 100, 40, 30])
    assert proc.returncode == 0, proc.stderr
    assert "no evidence artefact" not in proc.stderr
    assert (out / "push" / "evidence.jsonl").read_text() == "run 300\n"
    assert (out / "full" / "evidence.jsonl").read_text() == "run 40\n"
    assert "evidence-push-300 (https://github.com/o/r/actions/runs/300)" in proc.stdout


@pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")
def test_it_fails_only_when_no_recent_run_has_an_artefact(run_script):
    proc, out = run_script(push=PUSH, full=FULL, with_artefact=[], cancelled=[300])
    assert proc.returncode != 0
    assert proc.stderr.count("has no evidence artefact; using an older run") == 3
    assert "none of the last 10 finished runs (push) has an evidence artefact" in proc.stderr
    assert not (out / "push" / "evidence.jsonl").exists()


@pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")
def test_the_full_lookup_also_fails_when_every_weekly_run_lacks_one_but_not_when_there_is_none_yet(run_script):
    proc, _ = run_script(push=PUSH, full=FULL, with_artefact=[300])
    assert proc.returncode != 0
    assert "none of the last 10 finished runs (schedule workflow_dispatch) has an evidence artefact" in proc.stderr
    proc, out = run_script(push=PUSH, full=[], with_artefact=[300])
    assert proc.returncode == 0, proc.stderr
    assert "no weekly or manual run yet" in proc.stderr
    assert not (out / "full").exists()


@pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")
def test_push_only_never_looks_up_the_full_run_and_still_falls_back(run_script):
    proc, out = run_script("--push-only", push=PUSH, full=FULL, with_artefact=[200])
    assert proc.returncode == 0, proc.stderr
    assert (out / "push" / "evidence.jsonl").read_text() == "run 200\n"
    assert not (out / "full").exists()
    assert proc.stderr.count("using an older run") == 1
