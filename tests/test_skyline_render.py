"""The Trust page's skyline and foundation (scripts/trust_skyline.py) on a small, made-up model."""

import importlib.util
import pathlib
import re
import xml.etree.ElementTree as ET

import pytest

pytestmark = pytest.mark.validates("evidence", roots=["standards"], kind="guard")

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("trust", ROOT / "scripts" / "trust.py")
trust = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trust)
import trust_skyline as sky  # noqa: E402  (scripts/ is on sys.path once trust.py is loaded)


def node(name, layer, verdict, depth, strong=(), depends=()):
    return {"id": f"virgil.{name}", "label": name, "layer": layer, "verdict": verdict, "depth": depth,
            "strong": list(strong), "weak": [], "depends": [f"virgil.{d}" for d in depends]}


NODES = [
    node("oidata.read", "data", "verified", 0, ["standards", "pmoired"]),
    node("oidata.OIData", "data", "verified", 1, ["standards"], ["oidata.read"]),
    node("models.Disk", "models", "verified", 0, ["mathematics", "frank", "frap"]),
    node("models.Ring", "models", "unchecked", 0),
    node("likelihood.loglike", "likelihood", "relies-unverified", 2, ["mathematics"], ["models.Disk", "oidata.OIData"]),
    node("imaging.tsv", "imaging", "bug", 3, ["golden", "mathematics"], ["likelihood.loglike"]),
    {**node("pmoired.fit", "references", "verified", 0, ["literature"]), "id": "pmoired.fit"},
]
MODEL = {
    "nodes": {n["id"]: n for n in NODES},
    "docs": {"virgil.oidata.read": "tests/oidata.md#read", "virgil.models.Disk": "tests/models/index.md",
             "virgil.likelihood.loglike": "tests/likelihood.md#loglike", "virgil.imaging.tsv": "tests/imaging.md",
             "virgil.oidata.OIData": "tests/oidata.md", "virgil.models.Ring": "tests/models/index.md#ring"},
}
ROOTS = {
    "classes": {"mathematics": {"name": "Mathematics"}, "statistics": {"name": "Statistics"},
                "standards": {"name": "Standards"}, "other-peoples-code": {"name": "Other people's code"},
                "published-results": {"name": "Published results"}, "golden-values": {"name": "Golden values"},
                "self-consistency": {"name": "Self-consistency"}},
    "roots": [{"id": "mathematics", "class": "mathematics"}, {"id": "standards", "class": "standards"},
              {"id": "pmoired", "class": "other-peoples-code"}, {"id": "frank", "class": "other-peoples-code"},
              {"id": "frap", "class": "other-peoples-code", "counts_as": "frank"},
              {"id": "literature", "class": "published-results"}, {"id": "golden", "class": "golden-values"},
              {"id": "self-consistency", "class": "self-consistency"}],
}


def render(model=MODEL, base="../"):
    out = sky.skyline_html(model, ROOTS, base, names=trust)
    return out, ET.fromstring(out)  # raises if the output is not well-formed XML


def squares(tree):
    return [r for r in tree.iter("rect") if "sq" in r.get("class", "").split()]


def title_of(rect):
    return rect.find("title").text


def test_parses_as_xml_and_has_no_blank_line():
    out, tree = render()
    assert tree.tag == "div"
    assert "\n\n" not in out and not out.startswith("\n")


def test_one_square_per_node_with_name_verdict_and_depth():
    _, tree = render()
    rects = squares(tree)
    assert len(rects) == len(NODES)
    titles = {title_of(r) for r in rects}
    for n in NODES:
        word = trust.VERDICTS[n["verdict"]][0]
        assert f'{n["label"]} · {word} · depth {n["depth"]}' in titles


def test_columns_follow_layers_and_height_follows_depth():
    _, tree = render()
    by_label = {title_of(r).split(" · ")[0]: r for r in squares(tree)}
    def x(label):
        return float(by_label[label].get("x"))

    def y(label):
        return float(by_label[label].get("y"))

    # references stand leftmost, then data, models, likelihood and imaging as in LAYER_ORDER read bottom-up
    assert x("pmoired.fit") < x("oidata.read") < x("models.Disk") < x("likelihood.loglike") < x("imaging.tsv")
    # deeper squares sit higher (smaller y); a part is in the column of its layer whatever its depth
    assert y("oidata.OIData") < y("oidata.read")
    assert y("imaging.tsv") < y("likelihood.loglike") < y("oidata.read")
    assert x("oidata.read") <= x("oidata.OIData") < x("models.Disk")
    # depth 0 squares of different layers share a baseline band, above the ground line
    ground = max(float(r.get("y")) for r in tree.iter("rect") if r.get("height") == "3")
    assert all(y(label) + 11 <= ground for label in by_label)
    assert abs(y("pmoired.fit") - y("models.Ring")) <= 3 * sky.PITCH


def test_squares_do_not_overlap():
    _, tree = render()
    boxes = [(float(r.get("x")), float(r.get("y"))) for r in squares(tree)]
    for i, (x1, y1) in enumerate(boxes):
        for x2, y2 in boxes[i + 1:]:
            assert abs(x1 - x2) >= sky.CELL or abs(y1 - y2) >= sky.CELL, ((x1, y1), (x2, y2))


def test_every_square_is_inside_the_viewbox():
    _, tree = render()
    w, h = (float(v) for v in next(s for s in tree.iter("svg") if s.get("role") == "group").get("viewBox").split()[2:])
    for r in squares(tree):
        x, y = float(r.get("x")), float(r.get("y"))
        assert 0 <= x and x + sky.CELL <= w and 0 <= y and y + sky.CELL <= h


def test_each_square_links_to_its_function_test_page():
    _, tree = render()
    parent = {c: p for p in tree.iter() for c in p}
    links = {}
    for r in squares(tree):
        a = parent[r]
        links[title_of(r).split(" · ")[0]] = a.get("href") if a.tag == "a" else None
    assert links["oidata.read"] == "../" + trust.doc_url("tests/oidata.md#read")
    assert links["models.Disk"] == "../" + trust.doc_url("tests/models/index.md")
    assert links["pmoired.fit"] is None  # no page of its own: a square, not a dead link
    linked = [r for r in squares(tree) if parent[r].tag == "a"]
    assert all(parent[r].get("aria-label") == title_of(r) for r in linked)


def test_foundation_counts_parts_per_class_and_layer():
    counts = sky.foundation_counts(MODEL, ROOTS, trust.ROOT_ALIAS)
    assert counts["standards"] == {"data": 2}
    # two roots of one class (frank and frap, which alias to frank) still count the part once
    assert counts["other-peoples-code"] == {"data": 1, "models": 1}
    assert counts["mathematics"] == {"models": 1, "likelihood": 1, "imaging": 1}
    assert counts["golden-values"] == {"imaging": 1}
    assert counts["published-results"] == {"references": 1}
    assert "statistics" not in counts and "unlisted" not in counts


def test_foundation_counts_are_drawn_and_weak_root_has_no_row():
    out, tree = render()
    tips = [t.text for t in tree.iter("title") if t.text and "counted check against" in t.text]
    assert "2 data parts with a counted check against standards" in tips
    assert "1 imaging part with a counted check against golden values" in tips
    assert "Self-consistency" not in out and "Statistics" in out  # a class with no counts keeps its row
    assert len(tips) == sum(len(v) for v in sky.foundation_counts(MODEL, ROOTS, trust.ROOT_ALIAS).values())


def test_unlisted_root_is_not_lost():
    model = {"nodes": {"a": node("a", "data", "verified", 0, ["mystery"]) | {"id": "a"}}}
    assert sky.foundation_counts(model, ROOTS) == {"unlisted": {"data": 1}}
    out, _ = render(model)
    assert "Root not in the registry" in out


def test_legend_has_the_verdict_words_icons_and_counts():
    out, tree = render()
    for v, n in {"verified": 4, "unchecked": 1, "relies-unverified": 1, "bug": 1}.items():
        assert f"{trust.VERDICTS[v][0]} ({n})" in out
        assert trust.ICON[v] in out
    assert trust.VERDICTS["partly"][0] not in out  # only verdicts that occur


def test_palette_is_the_pages_and_text_follows_the_scheme():
    out, _ = render()
    for v in ("verified", "bug", "unchecked"):
        assert f".vt-sky .k-{v}{{fill:{trust.COLOURS[v]}}}" in out
    assert 'fill="currentColor"' in out and "overflow-x:auto" in out


def test_depth_labels_run_from_ground_up():
    _, tree = render()
    labels = {t.text: float(t.get("y")) for t in tree.iter("text")
              if t.text in {"0", "1", "2", "3"} and t.get("text-anchor") == "end"}
    assert labels["3"] < labels["2"] < labels["1"] < labels["0"]


def test_empty_model():
    out = sky.skyline_html({"nodes": {}}, ROOTS, names=trust)
    ET.fromstring(out)


def test_trust_page_draws_the_skyline_and_keeps_the_chips_in_a_fold():
    graph = {"nodes": {"virgil.a.A": {"layer": "data"}, "virgil.b.B": {"layer": "models", "depends": ["virgil.a.A"]}},
             "pipelines": {}}
    records = [{"test": "tests/t.py::t", "objects": ["virgil.a.A"], "roots": ["standards"], "kind": "check", "tier": "A",
                "outcome": "passed", "metrics": {}, "_commit": None, "_source": "ours", "line": 1, "doc": ""}]
    model = trust.build(graph, [], records, [])
    model["docs"] = {"virgil.a.A": "tests/a.md#a"}
    page = trust.render(model, "../")
    start = page.index('<h2 id="parts">')
    assert page.index('class="vt-sky"', start) < page.index('<details class="vt-fold" id="parts-list">', start)
    assert page.index('<details class="vt-fold" id="parts-list">') < page.index('class="vt-map"') < page.index("</details>", start)
    assert 'href="../tests/a/#a"' in page
    assert page.count('class="sq ') == 2


def test_references_column_keeps_its_layer_name_and_depth_zero_claims_no_roots():
    out, tree = render()
    headings = [t.text for t in tree.iter("text")]
    assert "references" in headings and "our references" not in headings
    assert "depth 0 relies on no other part" in out and "stands on the roots" not in out


def test_legend_counts_every_square_drawn_references_included():
    out, tree = render()
    assert f"{trust.VERDICTS['verified'][0]} (4)" in out  # pmoired.fit is a reference square, drawn and counted
    assert "6 parts of virgil and 1 reference nodes" in out
    words = {v: trust.VERDICTS[v][0] for v in ("verified", "unchecked", "relies-unverified", "bug")}
    assert sum(int(re.search(rf"{re.escape(w)} \((\d+)\)", out).group(1)) for w in words.values()) == len(squares(tree))


def test_render_alias_folds_into_the_foundation():
    # trust_render receives ROOT_ALIAS from trust.py: aliased roots count under their target's class
    assert trust.trust_render.ROOT_ALIAS == trust.ROOT_ALIAS
    model = {"nodes": {"a": node("a", "data", "verified", 0, ["frap"]) | {"id": "a"}}}
    assert sky.foundation_counts(model, ROOTS, {"frap": "frank"}) == {"other-peoples-code": {"data": 1}}


def test_corner_marks_show_changed_and_fails_elsewhere():
    n = node("a", "data", "verified", 0, ["standards"]) | {"changed": ["x.py"], "fails_elsewhere": True}
    m = node("b", "data", "verified", 0)
    out, tree = render({"nodes": {n["id"]: n, m["id"]: m}})
    tips = {title_of(r) for r in squares(tree)}
    assert "a · Verified · depth 0 · virgil has changed this since its evidence was measured · fails on another virgil commit" in tips
    assert "b · Verified · depth 0" in tips
    assert len(list(tree.iter("circle"))) >= 2 and out.count('class="mk') >= 4  # two on the square, one each in the legend keys
    assert "corner mark: virgil has changed it since its evidence (1)" in out
    assert "corner mark" not in render()[0]
