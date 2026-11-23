"""scripts/fetch_ci_evidence.sh against a stub ``gh``: a cancelled or timed-out run uploads no
evidence artefact, so the script must fall back to the newest older run that has one, say so on
stderr for each run it skips, and fail only when none has one."""

import os
import pathlib
import shutil
import stat
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fetch_ci_evidence.sh"

# The stub answers the three gh calls the script makes. FAKE_RUNS_<event> holds "createdAt id" lines
# (what the real call prints after its --jq: createdAt, id, conclusion), FAKE_WITH_ARTEFACT the ids of the runs that have an
# evidence artefact (FAKE_WITH_EVIDENCE: those whose artefact holds evidence.jsonl; a cancelled run's holds only mem.log); a download writes the run's id into DIR/evidence.jsonl.
STUB = r"""#!/usr/bin/env bash
case "$1 $2" in
  "run list")
    while [ $# -gt 0 ]; do [ "$1" = -e ] && event=$2; [ "$1" = -L ] && limit=$2; shift; done
    [ "$event" = "$FAKE_LIST_FAILS" ] && exit 1
    var=FAKE_RUNS_$event
    printf '%s\n' "${!var}" | sed '/^$/d' | sort -r | sed -n "1,${limit}p" ;;
  "api repos/"*)
    run=${2#*/runs/}; run=${run%/artifacts}
    [ "$run" = "$FAKE_API_FAILS" ] && exit 1
    for ok in $FAKE_WITH_ARTEFACT; do [ "$ok" = "$run" ] && echo "evidence-push-$run"; done ;;
  "run download")
    run=$3; while [ $# -gt 0 ]; do [ "$1" = -D ] && dir=$2; shift; done
    [ "$run" = "$FAKE_DOWNLOAD_FAILS" ] && exit 1
    [ -e "$dir/mem.log" ] && exit 1  # the real gh refuses to overwrite
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

    def run(*args, push=(), full=(), with_artefact=(), cancelled=(), download_fails=None, list_fails=None, api_fails=None, stale=False):
        env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "GITHUB_REPOSITORY": "o/r",
               "FAKE_RUNS_push": "\n".join(push),
               "FAKE_RUNS_schedule": "\n".join(full),
               "FAKE_RUNS_workflow_dispatch": "",
               "FAKE_WITH_ARTEFACT": " ".join(map(str, [*with_artefact, *cancelled])),
               "FAKE_DOWNLOAD_FAILS": str(download_fails),
               "FAKE_LIST_FAILS": str(list_fails), "FAKE_API_FAILS": str(api_fails),
               "FAKE_WITH_EVIDENCE": " ".join(map(str, with_artefact))}
        out = tmp_path / "ci"
        if stale:  # an OUTDIR left by an earlier fetch
            (out / "push").mkdir(parents=True)
            (out / "push" / "mem.log").write_text("old")
            (out / "push" / "evidence.jsonl").write_text("old")
        proc = subprocess.run(["bash", str(SCRIPT), *args, str(out)], env=env, capture_output=True, text=True)
        return proc, out

    return run


PUSH = ["2026-11-23T00:00:00Z 300 cancelled", "2026-11-22T00:00:00Z 200 cancelled", "2026-11-21T00:00:00Z 100 cancelled"]
FULL = ["2026-11-16T00:00:00Z 40 cancelled", "2026-11-09T00:00:00Z 30 cancelled"]


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


@pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")
def test_a_failed_download_stops_the_script_instead_of_reading_as_success(run_script):
    proc, out = run_script(push=PUSH, full=FULL, with_artefact=[300, 40], download_fails=300)
    assert proc.returncode != 0
    assert "downloading evidence-push-300 of run 300 failed" in proc.stderr
    assert not (out / "push" / "evidence.jsonl").exists()


@pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")
def test_a_failed_run_list_or_artefact_lookup_is_an_error_not_no_run_or_no_artefact(run_script):
    proc, out = run_script(push=PUSH, full=FULL, with_artefact=[300, 40], list_fails="schedule")
    assert proc.returncode != 0 and "listing the runs of (schedule workflow_dispatch) failed" in proc.stderr
    assert "no weekly or manual run yet" not in proc.stderr
    shutil.rmtree(out)  # the first call fetched push evidence before it failed on the weekly lookup
    proc, out = run_script(push=PUSH, full=FULL, with_artefact=[200], api_fails=300)
    assert proc.returncode != 0 and "looking up the artefacts of run 300 failed" in proc.stderr
    assert not (out / "push" / "evidence.jsonl").exists()  # it did not fall back to run 200


@pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")
def test_an_outdir_from_an_earlier_fetch_is_replaced_not_mixed(run_script):
    proc, out = run_script(push=PUSH, with_artefact=[200], cancelled=[300], stale=True)
    assert proc.returncode == 0, proc.stderr
    assert (out / "push" / "evidence.jsonl").read_text() == "run 200\n"
    assert not (out / "push" / "mem.log").read_text().startswith("old")


@pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")
def test_a_failed_run_without_evidence_is_an_error_not_something_to_skip(run_script):
    push = ["2026-11-23T00:00:00Z 300 failure", "2026-11-22T00:00:00Z 200 success"]
    proc, out = run_script(push=push, with_artefact=[200], cancelled=[300])
    assert proc.returncode != 0
    assert "run 300 (failure) has no evidence artefact and was not cancelled" in proc.stderr
    assert not (out / "push" / "evidence.jsonl").exists()  # it did not fall back to run 200
    # a run that failed its tests still has evidence, and is used
    proc, out = run_script(push=push, with_artefact=[300, 200])
    assert proc.returncode == 0, proc.stderr
    assert (out / "push" / "evidence.jsonl").read_text() == "run 300\n"
