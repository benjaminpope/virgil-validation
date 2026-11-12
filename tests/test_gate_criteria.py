"""Gate criteria on the git scheme (crosscheck.preregistration; Ben's rule, 2026-10-30): every gate has one committed
criteria file with a change-log table inside it; a run records the file's last commit and refuses a file with
uncommitted changes; an aggregate refuses runs from mixed commits. No hashing, signing or override flag.

``criteria_commit`` itself (dirty, untracked, archive snapshot, mixed commits) is tested in test_orbit_population.py.
This file checks that every gate script is wired to it and that nothing of the earlier SHA-256 registry is left."""

import argparse
import importlib.util
import json
import pathlib
import re
import shutil
import subprocess

import pytest
import yaml

from crosscheck import preregistration as prereg

ROOT = pathlib.Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")

# gate script -> its criteria file
GATES = {
    "detection_campaign": "design/detection_v2.yml",
    "pull_campaign": "design/pull_campaign.yml",
    "sbc_v2": "design/sbc_v2.yml",
    "sbc_numpyro": "design/sbc_addendum.yml",
    "alma_imlup_toeplitz_check": "design/toeplitz_acceptance.yml",
    "alma_imlup_maxent": "design/alma_imlup_gate_criteria.yml",
    "check_radio_trust_criteria": "design/radio_trust_criteria.yml",
    "trackb_fit": "design/trackb_criteria.md",
    "orbit_population": "design/orbit_gate_042.md",
    "toeplitz_reexam": "design/toeplitz_reexam_042.yml",
    "contest_informed": "design/contest_informed_criteria.md",
}
# criteria files that no script enforces yet (the script is not written): the file is still tracked and has its change log
UNENFORCED = ["design/oidb_binaries_criteria.md"]
OVERRIDE_FLAG = re.compile(r"""add_argument\(\s*["']--(allow[-_]\w*|force\w*|override\w*|unregistered|skip[-_]criteria\w*)""")


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tracked(path):
    return subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", "--", path], capture_output=True).returncode == 0


@pytest.mark.parametrize("name", sorted(GATES))
def test_every_gate_script_records_the_criteria_commit_and_has_no_override_flag(name):
    src = (ROOT / "scripts" / f"{name}.py").read_text()
    if name == "alma_imlup_maxent":  # the gate module is virgil_bridge.alma_start: the script asks it
        assert "vs.criteria_commit(" in src
        sources = [src, (ROOT / "src" / "virgil_bridge" / "alma_start.py").read_text()]  # the CLI flags live in the script, the check in the module
        src = sources[1]
    else:
        sources = [src]
    assert "criteria_commit(" in src and "preregistration" in src
    for text in sources:
        assert not OVERRIDE_FLAG.search(text), f"{name} has an override flag"
        assert "--allow" not in text


@pytest.mark.parametrize("aggregating", ["detection_campaign", "pull_campaign", "sbc_v2", "sbc_numpyro", "orbit_population", "alma_imlup_toeplitz_check", "trackb_summary", "toeplitz_reexam"])
def test_every_aggregate_refuses_mixed_commits(aggregating):
    src = (ROOT / "scripts" / f"{aggregating}.py").read_text()
    assert "require_same_commit(" in src


@pytest.mark.parametrize("name,path", sorted(GATES.items()) + [(None, p) for p in UNENFORCED])
def test_every_criteria_file_exists_is_tracked_and_has_a_change_log(name, path):
    assert (ROOT / path).is_file(), path
    if shutil.which("git") and subprocess.run(["git", "-C", str(ROOT), "rev-parse"], capture_output=True).returncode == 0:
        assert tracked(path), f"{path} is not tracked by git"
    text = (ROOT / path).read_text()
    assert re.search(r"change log", text, re.I), f"{path} has no change log"
    assert re.search(r"changed after seeing a result", text, re.I), f"{path}: the change log has no 'changed after seeing a result' column"


@pytest.fixture
def scratch_repo(tmp_path, monkeypatch):
    """A scratch git repository that ``prereg`` takes as the root, with ``put(rel, text)`` committing a file."""
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    import os

    git = lambda *a: subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True, env={**os.environ, **env})  # noqa: E731
    git("init", "-q")
    monkeypatch.setattr(prereg, "ROOT", tmp_path)

    def put(rel, text):
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
        git("add", rel)
        git("commit", "-q", "-m", "criteria")
        return f

    return put


CAMPAIGNS = [("detection_campaign", "CRITERIA_FILE", "run"), ("detection_campaign", "CRITERIA_FILE", "aggregate"),
             ("pull_campaign", "CRITERIA_FILE", "run"), ("pull_campaign", "CRITERIA_FILE", "aggregate"),
             ("sbc_v2", "CRITERIA_FILE", "run"), ("sbc_v2", "CRITERIA_FILE", "aggregate"),
             ("sbc_numpyro", "ADDENDUM", "run"), ("sbc_numpyro", "ADDENDUM", "aggregate")]


@pytest.mark.parametrize("name,attr,cmd", CAMPAIGNS)
def test_campaign_scripts_refuse_an_edited_criteria_file_before_any_draw(tmp_path, monkeypatch, scratch_repo, name, attr, cmd):
    """Ports the old 'a changed criteria file is refused' test: ``run`` and ``aggregate`` exit on uncommitted changes."""
    mod = script(name)
    rel = f"design/{getattr(mod, attr).name}"
    f = scratch_repo(rel, getattr(mod, attr).read_text())
    monkeypatch.setattr(mod, attr, f)
    assert prereg.criteria_commit(f)  # committed and clean: the gate would go on
    f.write_text(f.read_text() + "\n# edited after the commit\n")
    args = argparse.Namespace(out=str(tmp_path / "out"), dirs=[str(tmp_path)], summary=None, task=0, replicates=1, part="null",
                              regime="bright", cell=next(iter(script("pull_campaign").spec()["cells"])), draws=1, seed_base=1,
                              warmup=1, samples=1, chains=1)
    with pytest.raises(SystemExit, match="uncommitted changes"):
        getattr(mod, cmd)(args)


def test_the_other_gates_refuse_an_edited_criteria_file_too(tmp_path, monkeypatch, scratch_repo):
    cases = []
    chk = script("alma_imlup_toeplitz_check")
    f = scratch_repo("design/toeplitz_acceptance.yml", chk.CRITERIA_FILE.read_text())
    monkeypatch.setattr(chk, "CRITERIA_FILE", f)
    cases.append((lambda: chk.main(["summary", "--out", str(tmp_path)]), f))
    radio = script("check_radio_trust_criteria")
    g = scratch_repo("design/radio_trust_criteria.yml", radio.CRITERIA_FILE.read_text())
    monkeypatch.setattr(radio, "CRITERIA_FILE", g)
    cases.append((lambda: radio.main([]), g))
    import virgil_bridge.alma_start as vs

    h = scratch_repo("design/alma_imlup_gate_criteria.yml", vs.CRITERIA_FILE.read_text())
    monkeypatch.setattr(vs, "CRITERIA_FILE", h)
    cases.append((vs.criteria_commit, h))
    tb = script("trackb_fit")
    k = scratch_repo("design/trackb_criteria.md", (ROOT / "design" / "trackb_criteria.md").read_text())
    monkeypatch.setattr(tb, "REPO", tmp_path)
    cases.append((tb.criteria_block, k))
    ci = script("contest_informed")
    m = scratch_repo("design/contest_informed_criteria.md", (ROOT / "design" / "contest_informed_criteria.md").read_text())
    monkeypatch.setattr(ci, "REPO", tmp_path)
    # the run refuses before it reads any data; with a clean file it goes on and fails for want of the data (not an exit)
    cases.append((lambda: ci.run(0, "star_with_faint_companion", tmp_path / "no_data", tmp_path / "out", smoke=True), m))

    def refuses(call):
        try:
            call()
        except SystemExit as e:
            return "uncommitted changes" in str(e)
        except Exception:  # a later step may fail for want of data: the criteria check came first and passed
            return False
        return False

    for call, file in cases:
        assert not refuses(call)  # clean: the criteria check passes
        file.write_text(file.read_text() + "\n# edited after the commit\n")
        assert refuses(call)


def test_the_toeplitz_limits_are_the_ones_the_script_had():
    """Extracted verbatim from the former ACCEPT literal (origin/main 473d8cc)."""
    c = yaml.safe_load((ROOT / "design" / "toeplitz_acceptance.yml").read_text())
    assert c["accept"] == dict(max_rel=1e-10, dloss_rel=1e-6, dloss_abs=1e-2, grad_rel=1e-8, speedup=30.0, peak_gb=20.0)
    assert c["speedup_gpu"] == "A100"


def test_the_imlup_thresholds_are_the_ones_the_module_had():
    """Extracted verbatim from alma_start's constants (origin/main 473d8cc)."""
    c = yaml.safe_load((ROOT / "design" / "alma_imlup_gate_criteria.yml").read_text())
    assert c["cold_start"] == dict(flux_tol=0.10, dead_tol=0.01, reference_flag=0.05)
    b = c["better_than_mpol"]  # the extracted values are unchanged (virgil-validation#205 only added keys)
    assert {k: b[k] for k in ("mpol_raw_chi2n", "neff_min", "dead_max")} == dict(mpol_raw_chi2n=2.7062, neff_min=0.7, dead_max=0.01)
    assert b["flux_tol"] == c["cold_start"]["flux_tol"] == 0.10


def test_the_radio_trust_gate(tmp_path):
    mod = script("check_radio_trust_criteria")
    graph = yaml.safe_load((ROOT / "trust" / "graph.yml").read_text())["nodes"]
    crit = yaml.safe_load(mod.CRITERIA_FILE.read_text())
    assert mod.problems(crit, graph) == []
    graph["virgil.radio.ComplexVisData.read_ms"]["properties"] = {"conjugation": 0}  # a node loosened
    del graph["virgil.radio.primary_beam"]
    bad = mod.problems(crit, graph)
    assert any("conjugation" in b for b in bad) and any("primary_beam: not in the graph" in b for b in bad)
    out = tmp_path / "r.json"
    assert mod.main(["--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["criteria_commit"] == prereg.criteria_commit(mod.CRITERIA_FILE) and "registered" not in result


def test_gate_scripts_have_no_undefined_names():
    """A blanket edit once left ``regime`` undefined in sbc_v2.run, which the refusal tests never reach."""
    ruff = shutil.which("ruff")
    if ruff is None:
        pytest.skip("ruff is not installed")
    files = [f"scripts/{n}.py" for n in [*GATES, "trackb_summary"]]  # the aggregator is a script of its own
    done = subprocess.run([ruff, "check", "--select", "F821", "--no-cache", *files, "src/crosscheck/preregistration.py"], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout


def test_nothing_of_the_old_registry_layer_is_left():
    """The SHA-256 registry, its signed sign-offs and the override flag are gone from the code, tests and docs."""
    gone = ["allow_unregistered", "allow-unregistered", "criteria_registry", "allowed_signers", "ssh-keygen -Y"]
    me = pathlib.Path(__file__).resolve()
    for sub in ("scripts", "src", "tests", "docs"):
        for f in sorted((ROOT / sub).rglob("*")):
            if not f.is_file() or f.resolve() == me or f.suffix not in {".py", ".md", ".yml", ".yaml", ".json", ".toml", ".txt", ".html"}:
                continue
            text = f.read_text(errors="ignore")
            for s in gone:
                assert s not in text, f"{f.relative_to(ROOT)} still mentions {s!r}"
    assert not (ROOT / "design" / "criteria_registry.json").exists()
    assert not (ROOT / "trust" / "allowed_signers").exists() and not (ROOT / "trust" / "signoffs.yml").exists()
    assert not hasattr(prereg, "check") and not hasattr(prereg, "stamp") and not hasattr(prereg, "REGISTRY")


def test_the_alma_gate_files_carry_the_commit_the_run_started_under(tmp_path, monkeypatch):
    """pin_criteria() reads the commit once, at the start of a command: a later commit of the criteria file (a change-log
    row added while a long fit runs) must not relabel the files the run writes."""
    sys_path = str(ROOT / "scripts")
    import sys
    sys.path.insert(0, sys_path)
    try:
        import alma_imlup_maxent as m
    finally:
        sys.path.remove(sys_path)
    monkeypatch.setattr(m.vs, "criteria_commit", lambda: "a" * 40)
    monkeypatch.setattr(m, "_CRITERIA_COMMIT", None)
    assert m.pin_criteria() == "a" * 40
    monkeypatch.setattr(m.vs, "criteria_commit", lambda: "b" * 40)  # the file is committed again mid-run
    m.dump(tmp_path / "result.json", {"x": 1})
    assert json.loads((tmp_path / "result.json").read_text())["criteria_commit"] == "a" * 40


def test_the_sbc_numpyro_thresholds_are_in_the_criteria_file_under_its_change_log():
    """The pass/fail thresholds are the ``campaign:`` block of design/sbc_addendum.yml, not a dict in the script, so its
    commit and change log cover them. Values as they were in the script when the campaign was registered."""
    s = script("sbc_numpyro")
    assert s.CRITERIA == yaml.safe_load((ROOT / "design" / "sbc_addendum.yml").read_text())["campaign"]
    assert s.CRITERIA == {"draws_per_replicate": 99, "rank_bins": 20, "family_alpha": 0.01, "coverage_levels": [0.68, 0.95],
                          "coverage_band": 0.99, "min_bulk_ess": 400, "max_rhat": 1.01, "min_replicates": 400}


def test_the_trackb_summary_refuses_results_from_another_criteria_commit(tmp_path, monkeypatch):
    mod = script("trackb_summary")
    (tmp_path / "sys.json").write_text(json.dumps({"system": "s", "epochs": [], "criteria": {"file": "design/trackb_criteria.md", "commit": "0" * 40}}))
    monkeypatch.setattr(mod.sys, "argv", ["trackb_summary", str(tmp_path)])
    with pytest.raises(SystemExit, match="rerun"):
        mod.main()
    # control: results that name no commit are refused too
    (tmp_path / "sys.json").write_text(json.dumps({"system": "s", "epochs": [], "criteria": {"file": "design/trackb_criteria.md"}}))
    with pytest.raises(SystemExit, match="rerun"):
        mod.main()
