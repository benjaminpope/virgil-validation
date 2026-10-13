"""The binary orbit gallery (scripts/orbit_gallery.py): the trimmed summaries, the sky polylines and
the generated page. Fixtures in tests/fixtures/orbits/ are a hand-made fake fit of Gl 229 B (not a real
fit): ``fit_<id>.json`` in the shape of scripts/oidb_fit.py's output, and its extracted summary."""

import importlib.util
import json
import pathlib
import re

import numpy as np
import pytest

from crosscheck import orbits
from evidence.plugin import record

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "orbits"
PI1GRU = "19f7e2cf-2a03-4bb2-b7e2-cf2a03bbb245"
HD45166 = "f4afc4cd-fd31-40d3-afc4-cdfd3150d340"  # dropped from the gallery: design/hd45166_diagnosis.md
GL229 = "782185b2-0727-42b0-a185-b2072732b047"
ARXIV = re.compile(r"^\d{4}\.\d{4,5}$")


@pytest.fixture(scope="module")
def og():
    spec = importlib.util.spec_from_file_location("orbit_gallery", ROOT / "scripts" / "orbit_gallery.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def data(og):
    return og.build(FIX / "summaries")


def _target(data, slug):
    return next(t for t in data["targets"] if t["slug"] == slug)


@pytest.mark.validates("pipeline:published-binary", roots=["standards"], kind="guard")
def test_extract_trims_a_fit_to_the_committed_summary(og, tmp_path):
    (path,) = og.extract(FIX, tmp_path)
    summary = json.loads(path.read_text())
    assert summary == json.loads((FIX / "summaries" / f"{GL229}.json").read_text())
    full = json.loads((FIX / f"fit_{GL229}.json").read_text())
    assert path.stat().st_size < (FIX / f"fit_{GL229}.json").stat().st_size
    assert "datasets" not in summary["orbit"] and "map_sampled" not in summary["orbit"]
    assert summary["orbit"]["map"] == pytest.approx(full["orbit"]["map"], rel=1e-9)
    assert len(summary["orbit"]["modes"]) == len(full["orbit"]["modes"]) == 2
    assert summary["orbit"]["nuts"]["elements"] and "n_eff" not in summary["orbit"]["nuts"]
    assert summary["provenance"] == {"virgil_commit": "0123abc", "validation_commit": "4567def"}
    assert summary["instrument"] == "VLTI/GRAVITY"


@pytest.mark.validates("pipeline:published-binary", roots=["standards"], kind="guard")
def test_compare_reads_summaries_through_the_adapter(og):
    summary = json.loads((FIX / "summaries" / f"{GL229}.json").read_text())
    import oidb_compare as oc
    rows, missing = oc.compare(None, fits={GL229: summary})
    assert GL229 not in missing and len(missing) == len(oc.COMPARE) - 1 - 1  # the L2 workshop is skipped
    mine = [r for r in rows if r["collection"] == GL229]
    assert any(r["quantity"] == "period_day" and r["status"] == "REPORT" for r in mine)
    assert not [r for r in mine if r["status"] in ("FAIL", "MISSING")]


@pytest.mark.validates("pipeline:published-binary", roots=["standards"], kind="guard")
def test_every_reference_target_gets_a_section(og, data):
    refs = {p.stem for p in (ROOT / "oidb" / "references").glob("*.json")} - {PI1GRU, HD45166}  # pi1 Gru is a single star: no tile
    assert {t["collection_id"] for t in data["targets"]} == refs == set(og.TARGETS)
    assert len({t["slug"] for t in data["targets"]}) == len(data["targets"]) == len(og.TILES)  # the workshop collection gives two tiles
    text = og.page(data)
    names = re.findall(r"^\*\*(.+)\*\*$", text, flags=re.M)  # the plain-text section of each target
    assert names == [t["target"] for t in data["targets"]]
    record("targets", len(names))


@pytest.mark.validates("pipeline:published-binary", roots=["standards"], kind="guard")
def test_page_explains_the_gallery(og, data):
    text = og.page(data)
    for heading in ("How to read a plot", "The badges", "Where the data come from", "Method"):
        assert re.search(rf"^## {heading}$", text, flags=re.M), heading
    assert "OiDB" in text and "CC BY-NC-SA" in text and "Jean-Marie Mariotti" in text
    assert "??? note" in text and all(f"({page}.md)" in text for page in ("orbitize", "epochs", "aliases"))
    for badge in ("agrees", "tension", "reported", "awaiting fit"):
        assert f"**{badge}.**" in text, badge
    assert {g["key"] for g in data["guide"]} >= {"sky", "ours", "pub", "alt", "eps", "alias", "phase"}


@pytest.mark.validates("pipeline:published-binary", roots=["standards"], kind="guard")
def test_workshop_collection_splits_into_two_tiles(og, data):
    iota, sigma = _target(data, "iota-peg-workshop"), _target(data, "sigma-ori")
    assert iota["collection_id"] == sigma["collection_id"] == og.WORKSHOP
    assert iota["status"] != "awaiting" and sigma["status"] != "awaiting"
    assert iota["target"] != sigma["target"]
    assert [e["label"] for e in iota["epochs"]] != [e["label"] for e in sigma["epochs"]]


def _summaries_copy(tmp_path):
    for f in (FIX / "summaries").glob("*.json"):
        (tmp_path / f.name).write_text(f.read_text())
    return tmp_path


@pytest.mark.validates("pipeline:published-binary", roots=["standards"], kind="guard")
def test_alias_nights_are_flagged_by_label_and_by_the_summary_field(og, tmp_path):
    """The known Gl 229 alias nights are flagged by label; a night the summary itself marks ``alias`` is too."""
    path = _summaries_copy(tmp_path) / f"{GL229}.json"
    summary = json.loads(path.read_text())
    pos = summary["orbit"]["positions"]
    pos[0]["name"], pos[1]["name"], pos[2]["alias"] = "2024-02-28", "2024-03-01", True
    path.write_text(json.dumps(summary))
    flagged = {e["label"] for e in _target(og.build(tmp_path), "gl229b")["epochs"] if e.get("alias")}
    assert flagged == {"2024-02-28", pos[2]["name"]}  # the ordinary night (2024-03-01) is not flagged


@pytest.mark.validates("pipeline:published-binary", roots=["standards"], kind="guard")
def test_workshop_badge_is_scored_by_the_comparison_rule(og, data, tmp_path):
    """The fixture sits on the published sigma Ori position: it agrees. Moving it by many sigma is a tension."""
    assert _target(data, "sigma-ori")["status"] == "agrees"
    path = _summaries_copy(tmp_path) / f"{og.WORKSHOP}.json"
    summary = json.loads(path.read_text())
    for e in summary["targets"]["sig Ori"]["epochs"]:
        e["values"]["rho_mas"][0] += 1.0  # tens of the published sigmas, far beyond the epoch rule's limit
    path.write_text(json.dumps(summary))
    t = _target(og.build(tmp_path), "sigma-ori")
    assert t["status"] == "tension" and any(r["status"] == "FAIL" for r in t["table"])
    assert _target(og.build(tmp_path), "iota-peg-workshop")["status"] == "agrees"  # the other tile is untouched


@pytest.mark.validates("pipeline:published-binary", roots=["standards"], kind="guard")
def test_arxiv_links_are_well_formed(og, data):
    text = og.page(data)
    for t in data["targets"]:
        a = t["paper"]["arxiv"]
        assert ARXIV.match(a), a
        assert f"[arXiv:{a}](https://arxiv.org/abs/{a})" in text
    assert not re.findall(r"arxiv\.org/abs/(?!\d{4}\.\d{4,5}\))", text)


@pytest.mark.validates("pipeline:published-binary", roots=["standards"], kind="guard")
def test_embedded_json_parses_and_matches_the_json_file(og, tmp_path):
    out, js = tmp_path / "orbits" / "binaries.md", tmp_path / "assets" / "orbits.json"
    og.main(["render", "--out", str(out), "--json", str(js), "--summaries", str(FIX / "summaries")])
    text = out.read_text()
    assert text.count('id="og-data"') == 1 and '<div class="og"></div>' in text
    (blob,) = re.findall(r'<script type="application/json" id="og-data">(.*?)</script>', text, flags=re.S)
    assert "\n\n" not in blob  # a blank line would end the Markdown HTML block
    assert json.loads(blob) == json.loads(js.read_text())


@pytest.mark.validates("pipeline:published-binary", roots=["self-consistency"], kind="guard")
def test_polylines_close_and_pass_through_the_map_positions(data):
    t = _target(data, "gl229b")
    assert t["status"] == "agrees" and t["kind"] == "orbit"
    el = t["ours"]["elements"]
    paths = [t["ours"]["path"]] + [a["path"] for a in t["alternatives"]] + [p["path"] for p in t["published"]]
    for path in paths:
        assert path[0] == path[-1] and len(path) >= 200
    mjds = [e["mjd"] for e in t["epochs"]]
    expect = np.column_stack([np.ravel(x) for x in orbits.position(
        mjds, el["period_day"], el["t_peri_mjd"], el["ecc"], el["inc_deg"], el["omega_deg"], el["Omega_deg"],
        el["a_mas"])])
    pts = np.array(t["ours"]["path"])
    for want in expect:
        assert np.min(np.hypot(*(pts - want).T)) < 1e-5
    # the other paths pass through their own orbit at the same epochs
    for p in t["published"] + t["alternatives"]:
        e = p["elements"]
        want = np.column_stack([np.ravel(x) for x in orbits.position(
            mjds, e["period_day"], e["t_peri_mjd"], e["ecc"], e["inc_deg"], e["omega_deg"], e["Omega_deg"], e["a_mas"])])
        pp = np.array(p["path"])
        assert all(np.min(np.hypot(*(pp - w).T)) < 1e-5 for w in want)


@pytest.mark.validates("pipeline:published-binary", roots=["standards"], kind="guard")
def test_published_gl229_orbit_has_the_corrected_conventions(data):
    """Xuan+24's omega is the primary's (+180 for the companion) and a is in au (x 173.574 mas)."""
    ref = json.loads((ROOT / "oidb" / "references" / f"{GL229}.json").read_text())["orbit"]["pmoired_frequentist"]
    el = _target(data, "gl229b")["published"][0]["elements"]
    assert el["omega_deg"] == pytest.approx((ref["omega_primary_deg"][0] + 180) % 360)
    assert el["a_mas"] == pytest.approx(ref["a_au"][0] * 173.574)
    peak = max(np.hypot(*np.array(_target(data, "gl229b")["published"][0]["path"]).T))
    assert peak <= el["a_mas"] * (1 + el["ecc"]) * 1.001  # the projected apoastron cannot exceed a(1+e)


@pytest.mark.validates("pipeline:published-binary", roots=["standards"], kind="guard")
def test_target_without_a_summary_is_awaiting(og, tmp_path):
    data = og.build(tmp_path)  # an empty summaries directory
    assert {t["status"] for t in data["targets"]} == {"awaiting"}
    for t in data["targets"]:
        assert t["ours"] is None and t["epochs"] == [] and t["table"] == []
    gl = _target(data, "gl229b")
    assert len(gl["published"]) == 2  # the published orbits are still drawn
    assert "Awaiting fit" in og.page(data)
    # one summary present: only that tile leaves the awaiting state
    mixed = og.build(FIX / "summaries")
    assert [t["slug"] for t in mixed["targets"] if t["status"] != "awaiting"] == ["gl229b", "iota-peg-workshop", "sigma-ori"]
