"""The Trust page generator (scripts/trust.py) on a small, made-up graph."""

import importlib.util
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("trust", ROOT / "scripts" / "trust.py")
trust = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trust)

GRAPH = {
    "nodes": {
        "virgil.oidata.OIData": {"layer": "data"},
        "virgil.models.UniformDisk": {"layer": "models"},
        "virgil.likelihood.model_loglike": {"layer": "likelihood", "depends": ["virgil.oidata.OIData"]},
        "virgil.fitting.fit": {"layer": "likelihood", "depends": ["virgil.likelihood.model_loglike"]},
        "virgil.limits.nsigma": {"layer": "inference", "roots": 2},
        "virgil.imaging.l_curve": {"layer": "imaging"},
        "virgil.imaging.TSV": {"layer": "imaging"},
    },
    "pipelines": {
        "fits": {"title": "Simulated fits", "data": "simulated", "dataset": "VLTI", "reference": "truth",
                 "steps": ["virgil.oidata.OIData", "virgil.fitting.fit"]},
        "binaries": {"title": "Published binaries", "data": "real", "dataset": "ESO", "reference": "paper",
                     "state": "running", "steps": ["virgil.fitting.fit"]},
    },
}
LEDGER = [
    {"id": "F1", "title": "a bug", "objects": ["virgil.oidata.OIData"], "ruling": "virgil", "status": "open",
     "referee": "maths", "links": ["https://github.com/benjaminpope/virgil/pull/7"]},
    {"id": "D1", "title": "a definition", "objects": ["virgil.models.UniformDisk"], "ruling": "definition",
     "status": "documented", "referee": "maths"},
]


def rec(obj, roots, outcome="passed", kind="check", metrics=None, test="tests/test_x.py::test_a"):
    return {"test": test, "objects": [obj], "roots": roots, "kind": kind, "tier": "A", "outcome": outcome,
            "metrics": metrics or {}, "_commit": None, "_source": "ours", "line": 3, "doc": "what it checks"}


RECORDS = [
    rec("virgil.oidata.OIData", ["standards"]),
    rec("virgil.models.UniformDisk", ["mathematics"], metrics={"max_rel_err": 1e-14}),
    rec("virgil.likelihood.model_loglike", ["candid"]),
    rec("virgil.fitting.fit", ["pmoired"]),
    rec("virgil.limits.nsigma", ["mathematics"]),
    rec("virgil.imaging.TSV", ["ehtim"], outcome="failed"),
    rec("pipeline:fits", ["statistics"], metrics={"pull_sd_err": 0.02}),
]


@pytest.fixture(scope="module")
def model():
    return trust.build(GRAPH, LEDGER, RECORDS, [{"virgil": {"commit": "abcdef1234"}, "runner": "local", "date": "d"}])


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_verdicts_trace_reasons_to_the_finding(model):
    n = model["nodes"]
    assert n["virgil.oidata.OIData"]["verdict"] == "bug"
    assert n["virgil.models.UniformDisk"]["verdict"] == "verified"
    # its own check passes, but it relies on OIData, which has open F1;
    # the reason names the bottom of the chain, not just the next part
    assert n["virgil.fitting.fit"]["verdict"] == "relies"
    assert [b["id"] for b in n["virgil.fitting.fit"]["because"]] == ["virgil.oidata.OIData"]
    assert n["virgil.fitting.fit"]["because"][0]["findings"] == ["F1"]
    assert n["virgil.limits.nsigma"]["verdict"] == "partly"  # needs two roots, has one
    assert n["virgil.imaging.TSV"]["verdict"] == "failing"
    assert n["virgil.imaging.l_curve"]["verdict"] == "unchecked"
    assert n["virgil.oidata.OIData"]["dependents"] == ["virgil.likelihood.model_loglike"]


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_freshness_does_not_change_the_verdict(model):
    """Without a virgil checkout freshness is unknown; the verdicts stand."""
    assert model["changed"] is None
    assert all(n["changed"] is None for n in model["nodes"].values())
    assert model["counts"]["verified"] == 1


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_pipelines_real_and_simulated(model):
    p = {x["id"]: x for x in model["pipelines"]}
    assert p["fits"]["status"] == "passes" and p["fits"]["unverified"]  # steps rely on F1
    assert p["binaries"]["status"] == "running" and p["binaries"]["data"] == "real"


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_page_sections_links_and_embedded_data(model):
    page = trust.render(model)
    assert page.index("# Can virgil be trusted?") < page.index("> *Ma però che già mai") < page.index("virgil's calculations")
    for anchor in ("real-data", "simulated-data", "parts", "roots", "ledger", "precision"):
        assert f'id="{anchor}"' in page
    assert re.search(r'class="vt-big">1</span> of 7 parts of virgil verified', page)  # parts of virgil only
    assert ">virgil#7<" in page  # ledger links read repo#number
    assert 'href="https://github.com/benjaminpope/virgil-validation/blob/main/tests/test_x.py#L3"' not in page  # links live in the data
    data = json.loads(re.search(r'<script type="application/json" id="vt-data">(.*?)</script>', page, re.S).group(1))
    assert data["nodes"]["virgil.fitting.fit"]["because"][0]["id"] == "virgil.oidata.OIData"
    assert data["nodes"]["virgil.models.UniformDisk"]["checks"][0]["url"].endswith("tests/test_x.py#L3")
    for verdict in trust.VERDICTS:
        assert verdict in data["verdicts"]
