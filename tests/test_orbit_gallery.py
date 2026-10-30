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


@pytest.mark.validates("pipeline:binary-orbits", roots=["self-consistency"], kind="guard")
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


@pytest.mark.validates("pipeline:binary-orbits", roots=["self-consistency"], kind="guard")
def test_compare_reads_summaries_through_the_adapter(og):
    summary = json.loads((FIX / "summaries" / f"{GL229}.json").read_text())
    import oidb_compare as oc
    rows, missing = oc.compare(None, fits={GL229: summary})
    assert GL229 not in missing and len(missing) == len(oc.COMPARE) - 1
    mine = [r for r in rows if r["collection"] == GL229]
    assert any(r["quantity"] == "period_day" and r["status"] == "REPORT" for r in mine)
    assert not [r for r in mine if r["status"] in ("FAIL", "MISSING")]


@pytest.mark.validates("pipeline:binary-orbits", roots=["self-consistency"], kind="guard")
def test_every_reference_target_gets_a_section(og, data):
    refs = {p.stem for p in (ROOT / "oidb" / "references").glob("*.json")} - {PI1GRU, HD45166, og.ESO_9SGR, og.ESO_HD168137}  # pi1 Gru is a single star: no tile; 9 Sgr and HD 168137 are off the page (no JMMC pipeline setup)
    trackb = {f"trackb-{p.stem}" for p in og.TRACKB.glob("*.json")}  # the Track B targets are tiles of their own
    assert trackb and not trackb & refs
    assert {t["collection_id"] for t in data["targets"]} == refs | trackb
    assert refs | trackb == set(og.TARGETS)
    assert len({t["slug"] for t in data["targets"]}) == len(data["targets"]) == len(og.TILES)  # the workshop collection gives two tiles
    text = og.page(data)
    names = re.findall(r"^\*\*(.+)\*\*$", text, flags=re.M)  # the plain-text section of each target
    assert names == [t["target"] for t in data["targets"]]
    record("targets", len(names))


@pytest.mark.validates("pipeline:binary-orbits", roots=["standards"], kind="guard")
def test_page_explains_the_gallery(og, data):
    text = og.page(data)
    for heading in ("How to read a plot", "The badges", "Where the data come from", "Method"):
        assert re.search(rf"^## {heading}$", text, flags=re.M), heading
    assert "OiDB" in text and "CC BY-NC-SA" in text and "Jean-Marie Mariotti" in text
    assert "??? note" in text and all(f"({page}.md)" in text for page in ("orbitize", "epochs", "aliases"))
    for badge in ("agrees", "tension", "reported", "awaiting fit"):
        assert f"**{badge}.**" in text, badge
    assert {g["key"] for g in data["guide"]} >= {"sky", "ours", "pub", "alt", "eps", "alias", "phase"}


@pytest.mark.validates("pipeline:binary-orbits", roots=["standards"], kind="guard")
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


@pytest.mark.validates("pipeline:binary-orbits", roots=["standards"], kind="guard")
def test_alias_nights_come_from_the_summary_else_from_the_label_list(og, tmp_path):
    """A summary without ``epoch_checks`` falls back to the known Gl 229 labels; one with them uses its "flagged" nights."""
    path = _summaries_copy(tmp_path) / f"{GL229}.json"
    summary = json.loads(path.read_text())
    pos = summary["orbit"]["positions"]
    pos[0]["name"], pos[1]["name"] = "2024-02-28", "2024-03-01"
    path.write_text(json.dumps(summary))
    flagged = lambda: {e["label"] for e in _target(og.build(tmp_path), "gl229b")["epochs"] if e.get("alias")}
    assert flagged() == {"2024-02-28"}  # the label list; the ordinary night (2024-03-01) is not flagged
    summary["orbit"]["epoch_checks"] = [dict(name=pos[1]["name"], reasons=["flagged: single-night alias peak (Xuan et al. flag: weak signal)"], barred=False),
                                        dict(name=pos[0]["name"], reasons=[], barred=False)]
    path.write_text(json.dumps(summary))
    assert flagged() == {"2024-03-01"}  # the summary's own checks win


@pytest.mark.validates("pipeline:binary-orbits", roots=["standards"], kind="guard")
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


@pytest.mark.validates("pipeline:binary-orbits", roots=["standards"], kind="guard")
def test_arxiv_links_are_well_formed(og, data):
    text = og.page(data)
    for t in data["targets"]:
        a = t["paper"]["arxiv"]
        if a is None:  # the workshop tile cites two papers (iota Peg, sigma Ori) and links neither
            assert t["collection_id"].startswith("647a22a9"), t["slug"]
            continue
        assert ARXIV.match(a), a
        assert f"[arXiv:{a}](https://arxiv.org/abs/{a})" in text
    assert not re.findall(r"arxiv\.org/abs/(?!\d{4}\.\d{4,5}\))", text)


@pytest.mark.validates("pipeline:binary-orbits", roots=["standards"], kind="guard")
def test_embedded_json_parses_and_matches_the_json_file(og, tmp_path):
    out, js = tmp_path / "orbits" / "binaries.md", tmp_path / "assets" / "orbits.json"
    og.main(["render", "--out", str(out), "--json", str(js), "--summaries", str(FIX / "summaries")])
    text = out.read_text()
    assert text.count('id="og-data"') == 1 and '<div class="og"></div>' in text
    (blob,) = re.findall(r'<script type="application/json" id="og-data">(.*?)</script>', text, flags=re.S)
    assert "\n\n" not in blob  # a blank line would end the Markdown HTML block
    assert json.loads(blob) == json.loads(js.read_text())


@pytest.mark.validates("pipeline:binary-orbits", roots=["self-consistency"], kind="guard")
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


@pytest.mark.validates("pipeline:binary-orbits", roots=["standards"], kind="guard")
def test_published_gl229_orbit_has_the_corrected_conventions(data):
    """Xuan+24's omega is the primary's (+180 for the companion) and a is in au (x 173.574 mas)."""
    ref = json.loads((ROOT / "oidb" / "references" / f"{GL229}.json").read_text())["orbit"]["pmoired_frequentist"]
    el = _target(data, "gl229b")["published"][0]["elements"]
    assert el["omega_deg"] == pytest.approx((ref["omega_primary_deg"][0] + 180) % 360)
    assert el["a_mas"] == pytest.approx(ref["a_au"][0] * 173.574)
    peak = max(np.hypot(*np.array(_target(data, "gl229b")["published"][0]["path"]).T))
    assert peak <= el["a_mas"] * (1 + el["ecc"]) * 1.001  # the projected apoastron cannot exceed a(1+e)


@pytest.mark.validates("pipeline:binary-orbits", roots=["standards"], kind="guard")
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


@pytest.mark.validates("pipeline:binary-orbits", roots=["standards"], kind="guard")
def test_seeded_and_single_mode_fits_are_labelled_provisional_and_a_failed_tile_says_so(og, data):
    gl = _target(data, "gl229b")
    assert gl["flags"] == []  # the fixture fit is blind and has two modes
    summary = json.loads((FIX / "summaries" / f"{GL229}.json").read_text())
    tile = next(t for t in og.TILES if t["slug"] == "gl229b")
    seeded = json.loads(json.dumps(summary))
    seeded["orbit"]["start"] = "published"
    (flag,) = og.flags_of(tile, seeded)
    assert flag["key"] == "provisional" and "seeded from the published orbit" in flag["text"]
    single = json.loads(json.dumps(summary))
    single["orbit"]["modes"] = single["orbit"]["modes"][:1]
    (flag,) = og.flags_of(tile, single)
    assert flag["key"] == "provisional" and "single-mode" in flag["text"]
    iota = next(t for t in og.TILES if t["slug"] == "iota-peg")
    assert [f["key"] for f in og.flags_of(iota, None)] == ["failed"]
    assert "Provisional: single-mode fit" in og.page(dict(data, targets=[dict(_target(data, "gl229b"), flags=[flag])]))


@pytest.mark.validates("pipeline:binary-orbits", roots=["standards"], kind="guard")
def test_a_pass_on_unconverged_nuts_is_only_reported_and_the_tile_says_so(og, tmp_path):
    summary = json.loads((FIX / "summaries" / f"{GL229}.json").read_text())
    summary["orbit"]["nuts"]["r_hat"] = {k: 1.001 for k in summary["orbit"]["nuts"]["r_hat"]}
    converged = tmp_path / "ok"
    converged.mkdir()
    (converged / f"{GL229}.json").write_text(json.dumps(summary))
    ok = _target(og.build(converged), "gl229b")
    assert ok["status"] == "agrees" and not any(f["key"] == "nonconverged" for f in ok["flags"])
    summary["orbit"]["nuts"]["r_hat"]["ecc"] = 10.0
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / f"{GL229}.json").write_text(json.dumps(summary))
    bad = _target(og.build(bad_dir), "gl229b")
    assert bad["status"] == "reported" and "not converged" in bad["status_why"]
    assert [f["key"] for f in bad["flags"] if f["key"] == "nonconverged"] == ["nonconverged"]


@pytest.mark.validates("pipeline:binary-orbits", roots=["literature"], kind="guard")
@pytest.mark.parametrize("cid,pub,limit", [("eso-9sgr", 0, 0.3), ("eso-9sgr", 1, 0.3), ("eso-hd168137", 0, 0.45)])
def test_eso_references_reproduce_their_published_epochs(og, cid, pub, limit):
    """Transcription check of the 9 Sgr and HD 168137 references against the literature, without fitting (it cannot check
    the sign of omega and Omega: positions are invariant under their joint 180 degree flip): the
    published orbit, evaluated by crosscheck.orbits at the published epochs, lands within the rms limit
    (0.19, 0.16 and 0.30 mas when written; the Fabry+21 orbit is scored on the 23 epochs and LB17's on its 18)."""
    ref = json.loads((ROOT / "oidb" / "references" / f"{cid}.json").read_text())
    assert ref["conventions"]["omega_shift_deg"] == 0 and ref["paper"]["arxiv"] == "1608.03525"
    el = og.published_orbits("9sgr" if cid == "eso-9sgr" else "hd168137", ref)[pub][1]
    rows = [e for e in ref["epochs"] if pub == 1 or not e.get("new_in_fabry21")]
    off = []
    for e in rows:
        dra, ddec = (float(np.ravel(x)[0]) for x in orbits.position(
            e["mjd"], el["period_day"], el["t_peri_mjd"], el["ecc"], el["inc_deg"], el["omega_deg"], el["Omega_deg"], el["a_mas"]))
        rho, pa = e["rho_mas"][0], np.radians(e["pa_deg"][0])
        off.append(np.hypot(dra - rho * np.sin(pa), ddec - rho * np.cos(pa)))
    rms = float(np.sqrt(np.mean(np.square(off))))
    assert rms < limit, rms
    assert len(rows) == (23 if pub == 1 else 18 if cid == "eso-9sgr" else 11)
    record("rms_offset_mas", round(rms, 3))


@pytest.mark.validates("pipeline:binary-orbits", roots=["self-consistency"], kind="guard")
def test_legacy_label_follows_the_fit_and_downgrades_a_pass(og):
    fit = {"orbit": {"modes": [1, 2]}}
    tile = og.TARGETS["trackb-ll_aqr"]
    (flag,) = [f for f in og.flags_of(tile, fit) if f["key"] == "legacy"]
    assert "uncalibrated GRAVITY V²" in flag["text"] and "rerun pending" in flag["text"]
    nn = [f for f in og.flags_of(og.TARGETS["trackb-nn_del"], fit) if f["key"] == "legacy"]
    assert len(nn) == 1 and "gain model" in nn[0]["text"]
    assert not [f for f in og.flags_of(tile, None) if f["key"] == "legacy"]  # no fit, nothing to label
    # a rerun records its own data path, which clears the tile's default
    rerun = dict(fit, provenance={"data_path": "calibrated"})
    assert not [f for f in og.flags_of(tile, rerun) if f["key"] == "legacy"]
    # OiDB tiles are not labelled
    assert not [t for t in og.TILES if not t.get("trackb") and t.get("data_path")]
    table = [dict(status="PASS", quantity="x", pull=0.1)]
    assert og.status_of(table, fit)[0] == "agrees"
    assert og.status_of(table, fit, og.legacy_text(tile, fit))[0] == "reported"


@pytest.mark.validates("pipeline:binary-orbits", roots=["standards"], kind="guard")
def test_only_alias_peaks_flags_mark_an_alias_and_other_flags_carry_their_reason(og, tmp_path):
    """sep-below-resolution, chi2-high etc. are flagged nights, not aliases; each says why and whether it was excluded."""
    path = _summaries_copy(tmp_path) / f"{GL229}.json"
    summary = json.loads(path.read_text())
    pos = summary["orbit"]["positions"]
    for k, p in enumerate(pos[:5]):
        p["name"] = f"n{k}"
    summary["orbit"]["epoch_checks"] = [
        dict(name="n0", reasons=["flagged: alias-peaks: two peaks"], barred=True),
        dict(name="n1", reasons=["flagged: sep-below-resolution: 0.7 mas below 1.9 mas"], barred=True),
        dict(name="n2", reasons=["flagged: chi2-high: 138 > 122"], barred=False),
        dict(name="n3", reasons=[], barred=False),
        dict(name="n4", reasons=["flagged: chi2-high: 9 > 3", "flagged: alias-peaks: two peaks"], barred=False)]
    path.write_text(json.dumps(summary))
    eps = {e["label"]: e for e in _target(og.build(tmp_path), "gl229b")["epochs"]}
    assert eps["n0"].get("alias") and "flag" not in eps["n0"]
    assert not eps["n1"].get("alias") and eps["n1"]["flag"] == dict(
        code="sep-below-resolution", reason="sep-below-resolution: 0.7 mas below 1.9 mas", excluded=True)
    assert not eps["n2"].get("alias") and eps["n2"]["flag"]["code"] == "chi2-high" and not eps["n2"]["flag"]["excluded"]
    assert not eps["n3"].get("alias") and "flag" not in eps["n3"]
    assert eps["n4"].get("alias") and eps["n4"]["flag"]["code"] == "chi2-high"  # order of the reasons does not matter
    assert og.ALIAS_CODE in og.FLAG_CODES


@pytest.mark.validates("pipeline:binary-orbits", roots=["standards"], kind="guard")
def test_implausible_alternative_modes_are_hidden_and_counted(og, tmp_path):
    path = _summaries_copy(tmp_path) / f"{GL229}.json"
    summary = json.loads(path.read_text())
    modes = summary["orbit"]["modes"]
    far = json.loads(json.dumps(modes[1]))
    far["loss"] = modes[0]["loss"] + og.MODE_DLOSS_MAX + 1.0
    modes.append(far)
    path.write_text(json.dumps(summary))
    t = _target(og.build(tmp_path), "gl229b")
    assert t["modes_hidden"] == 1 and t["mode_dloss_max"] == og.MODE_DLOSS_MAX
    assert len(t["alternatives"]) == 1 and t["alternatives"][0]["dloss"] <= og.MODE_DLOSS_MAX
    assert str(int(og.MODE_DLOSS_MAX)) in next(g for g in og.GUIDE if g["key"] == "alt")["text"]


@pytest.mark.validates("pipeline:binary-orbits", roots=["self-consistency"], kind="guard")
def test_sky_axes_ignore_alternative_modes_in_the_renderer():
    js = (ROOT / "docs" / "assets" / "orbits.js").read_text()
    body = js.split("function allSkyPoints", 1)[1].split("function bbox", 1)[0]
    assert "alternatives" not in body and "clip-path" in js


@pytest.mark.validates("pipeline:binary-orbits", roots=["self-consistency"], kind="guard")
def test_stored_gl229_summary_keeps_both_alias_nights(og):
    """The committed Gl 229 summary uses oidb_fit's uncoded wording; both nights must still be aliases."""
    data = og.build(og.SUMMARIES)
    eps = _target(data, "gl229b")["epochs"]
    assert {e["label"] for e in eps if e.get("alias")} == og.ALIAS_NIGHTS[GL229]
    assert not any(e.get("flag") for e in eps if e.get("alias"))


@pytest.mark.validates("pipeline:binary-orbits", roots=["self-consistency"], kind="guard")
def test_the_gl229_tile_labels_its_exclusions_as_post_hoc(og):
    tile = next(t for t in og.TILES if t["cid"] == GL229)
    note = next(n for n in tile["notes"] if n["title"] == "Post-hoc exclusions")
    assert "2024-12-19" in note["text"] and "GRAVI.2023-12-26T05_51_29.891" in note["text"]
    assert "has not been run" in note["text"]  # no all-nights fit exists; the page must not imply one
