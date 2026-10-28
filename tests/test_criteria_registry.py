"""Pre-registered gate criteria (crosscheck.preregistration): a mismatched hash fails, a matching one passes, the
unregistered flag stamps the output, and every gate script is wired to the registry."""

import hashlib
import importlib.util
import json
import pathlib
import re

import pytest
import yaml

from crosscheck import preregistration as prereg

ROOT = pathlib.Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")


def pinned(tmp_path, text="threshold: 1\n", pin=None):
    """A repository root with one criteria file and a registry pinning it (to ``pin``, else its true hash)."""
    f = tmp_path / "design" / "c.yml"
    f.parent.mkdir()
    f.write_text(text)
    reg = tmp_path / "design" / "criteria_registry.json"
    reg.write_text(json.dumps({"criteria": {"design/c.yml": {"sha256": pin or hashlib.sha256(text.encode()).hexdigest()}}}))
    return f, reg


def test_a_matching_hash_passes(tmp_path):
    f, reg = pinned(tmp_path)
    r = prereg.check(f, registry=reg, root=tmp_path)
    assert r.registered and r.sha256 == r.pinned and r.file == "design/c.yml"
    assert prereg.stamp({"pass": True}, r) == {"pass": True, "criteria_file": "design/c.yml", "criteria_hash": r.sha256, "registered": True}


def test_a_mismatched_hash_fails(tmp_path):
    f, reg = pinned(tmp_path)
    f.write_text("threshold: 2\n")  # a criterion edited after it was pinned
    with pytest.raises(SystemExit) as e:
        prereg.check(f, registry=reg, root=tmp_path)
    assert e.value.code != 0 and "changed after registration" in str(e.value.code)


def test_a_file_with_no_pin_fails(tmp_path):
    f, reg = pinned(tmp_path)
    g = tmp_path / "design" / "other.yml"
    g.write_text("x: 1\n")
    with pytest.raises(SystemExit, match="no pinned hash"):
        prereg.check(g, registry=reg, root=tmp_path)


def test_the_unregistered_flag_stamps_the_output(tmp_path, capsys):
    f, reg = pinned(tmp_path, pin="0" * 64)
    r = prereg.check(f, allow_unregistered=True, registry=reg, root=tmp_path)
    assert not r.registered
    out = prereg.stamp({"pass": True, "verdict": "PASS"}, r)
    assert out["stamp"] == prereg.UNREGISTERED_STAMP == "UNREGISTERED: not gate evidence"
    assert out["registered"] is False and out["pass"] is False
    assert out["verdict"] == "UNREGISTERED"  # the IM Lup gates report a verdict, not a pass flag
    assert prereg.stamp({"verdict": "FAIL"}, r)["verdict"] == "FAIL"  # only a PASS is downgraded
    lines = prereg.stamp_lines(["table"], r)
    assert lines[0].startswith(f"**{prereg.UNREGISTERED_STAMP}**") and lines[-1] == "table"
    assert prereg.UNREGISTERED_STAMP in capsys.readouterr().err


def test_runs_made_unregistered_do_not_make_registered_evidence(tmp_path):
    f, reg = pinned(tmp_path)
    r = prereg.check(f, registry=reg, root=tmp_path)
    prereg.require_registered_runs([{"registered": True}, {}], r)  # older records carry no flag
    with pytest.raises(SystemExit, match="not gate evidence"):
        prereg.require_registered_runs([{"registered": True}, {"registered": False}], r)


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_pinned_file_matches_its_pin():
    for key, entry in prereg.load_registry().items():
        assert prereg.sha256_file(ROOT / key) == entry["sha256"], f"{key} differs from design/criteria_registry.json: criteria changed after registration"
        assert "changed_after_seeing_a_result" in entry and entry["used_by"]


@pytest.mark.parametrize("name,cmd", [("detection_campaign", "run"), ("detection_campaign", "aggregate"), ("pull_campaign", "run"), ("pull_campaign", "aggregate"),
                                      ("sbc_v2", "run"), ("sbc_v2", "aggregate")])
def test_campaign_scripts_exit_on_a_changed_criteria_file(tmp_path, monkeypatch, name, cmd):
    """``run`` and ``aggregate`` refuse before doing anything when their criteria file is not the pinned one."""
    import argparse

    mod = script(name)
    key = f"design/{mod.CRITERIA_FILE.name}"
    (tmp_path / "design").mkdir()
    copy = tmp_path / key
    copy.write_text(mod.CRITERIA_FILE.read_text())
    (tmp_path / "design" / "criteria_registry.json").write_text(json.dumps({"criteria": {key: {"sha256": prereg.sha256_file(copy)}}}))
    monkeypatch.setattr(prereg, "ROOT", tmp_path)
    monkeypatch.setattr(prereg, "REGISTRY", tmp_path / "design" / "criteria_registry.json")
    monkeypatch.setattr(mod, "CRITERIA_FILE", copy)
    assert prereg.for_args(argparse.Namespace(), copy).registered  # the pin matches: the gate would go on
    copy.write_text(copy.read_text() + "\n# edited after registration\n")
    with pytest.raises(SystemExit, match="changed after registration"):
        getattr(mod, cmd)(argparse.Namespace(out=str(tmp_path / "out"), dirs=[str(tmp_path)], summary=None))


@pytest.mark.parametrize("name", ["detection_campaign", "pull_campaign", "sbc_v2", "alma_imlup_toeplitz_check", "alma_imlup_maxent", "check_radio_trust_criteria"])
def test_every_gate_script_uses_the_helper(name):
    src = (ROOT / "scripts" / f"{name}.py").read_text()
    assert "crosscheck.preregistration" in src and ("prereg.check(" in src or "prereg.for_args(" in src)
    assert "prereg.add_argument(" in src


def test_the_toeplitz_limits_are_the_ones_the_script_had():
    """Extracted verbatim from the former ACCEPT literal (origin/main 473d8cc)."""
    c = yaml.safe_load((ROOT / "design" / "toeplitz_acceptance.yml").read_text())
    assert c["accept"] == dict(max_rel=1e-10, dloss_rel=1e-6, dloss_abs=1e-2, grad_rel=1e-8, speedup=30.0, peak_gb=20.0)
    assert c["speedup_gpu"] == "A100"


def test_the_imlup_thresholds_are_the_ones_the_module_had():
    """Extracted verbatim from alma_start's constants (origin/main 473d8cc)."""
    c = yaml.safe_load((ROOT / "design" / "alma_imlup_gate_criteria.yml").read_text())
    assert c["cold_start"] == dict(flux_tol=0.10, dead_tol=0.01, reference_flag=0.05)
    assert c["better_than_mpol"] == dict(mpol_raw_chi2n=2.7062, neff_min=0.7, dead_max=0.01)


def test_trackb_pin_in_the_script_agrees_with_the_registry():
    pin = prereg.load_registry()["design/trackb_criteria.md"]["sha256"]
    assert re.search(rf'CRITERIA_SHA256 = "{pin}"', (ROOT / "scripts" / "trackb_fit.py").read_text())


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
    assert mod.main(["--out", str(out)]) == 0 and json.loads(out.read_text())["registered"] is True


def test_gate_scripts_have_no_undefined_names():
    """A blanket edit once left ``regime`` undefined in sbc_v2.run, which the refusal tests never reach."""
    import shutil
    import subprocess

    ruff = shutil.which("ruff")
    if ruff is None:
        pytest.skip("ruff is not installed")
    files = [f"scripts/{n}.py" for n in ("detection_campaign", "pull_campaign", "sbc_v2", "alma_imlup_toeplitz_check", "alma_imlup_maxent", "check_radio_trust_criteria")]
    done = subprocess.run([ruff, "check", "--select", "F821", "--no-cache", *files, "src/crosscheck/preregistration.py"], cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout
