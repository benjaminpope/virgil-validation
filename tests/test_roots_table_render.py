"""The roots table counts from a model, marks draft text, leaves links uncounted and labels roots from ROOT_NAMES."""

import importlib.util
import pathlib
import re

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")


def _exec(spec):
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# literal paths, so the via scanner can resolve each import (a computed path is an unresolved dynamic import)
table = _exec(importlib.util.spec_from_file_location("trust_roots_table", ROOT / "scripts" / "trust_roots_table.py"))
trust = _exec(importlib.util.spec_from_file_location("trust", ROOT / "scripts" / "trust.py"))

REGISTRY = {
    "classes": {
        "mathematics": {"name": "Mathematics", "what": "Results on paper.", "why": "Textbook.",
                        "cannot_catch": "A `misread` formula.", "reviewed": True},
        "other-peoples-code": {"name": "Other people's code", "what": "Code by others.", "why": "Used in science.",
                               "cannot_catch": "Shared bugs."},
        "self-consistency": {"name": "Self-consistency", "what": "virgil against itself.", "why": "Catches regressions.",
                             "cannot_catch": "Consistent mistakes.", "reviewed": True},
    },
    "links": {
        "reference-code": {"name": "Our reference code", "what": "crosscheck.", "why": "A link.",
                           "cannot_catch": "Agent errors.", "reviewed": True},
    },
    "roots": [
        {"id": "mathematics", "class": "mathematics", "what": "Closed forms.", "why": "Mature libraries.",
         "cannot_catch": "Transcription.", "reviewed": True},
        {"id": "pmoired", "class": "other-peoples-code", "what": "PMOIRED.", "why": "Published.",
         "cannot_catch": "Own bugs.", "reviewed": False},
        {"id": "frank", "class": "other-peoples-code", "what": "frank.", "why": "Published.",
         "cannot_catch": "Own bugs.", "reviewed": True},
        {"id": "frap", "class": "other-peoples-code", "counts_as": "frank", "what": "FRAP.", "why": "Published.",
         "cannot_catch": "Own bugs."},
        {"id": "self-consistency", "class": "self-consistency", "what": "Itself.", "why": "Regressions.",
         "cannot_catch": "Consistent errors.", "reviewed": True},
    ],
}
NAMES = {"mathematics": "MATHS LABEL", "pmoired": "PMOIRED LABEL", "frank": "FRANK LABEL", "frap": "FRAP LABEL",
         "self-consistency": "ITSELF LABEL"}


def _node(name, strong, weak=()):
    return {"id": name, "strong": list(strong), "weak": list(weak)}


MODEL = {"nodes": {n["id"]: n for n in [
    _node("virgil.a", ["mathematics"]),                      # alone on mathematics
    _node("virgil.b", ["mathematics", "pmoired"]),           # both classes
    _node("virgil.c", ["pmoired", "frank"]),                 # two roots, one class: alone on other-peoples-code
    _node("virgil.d", [], ["self-consistency"]),             # weak only: counted nowhere
    _node("crosscheck.ref", ["mathematics"]),                # reference code: not a part
]}}


def _rows(html_text):
    """{row header: [cell texts]} for the class and link rows."""
    out = {}
    for tr in re.findall(r"<tr(?: class=\"vt-link\")?><th scope=\"row\">.*?</tr>", html_text):
        head = re.sub(r"<[^>]+>", "", re.search(r"<th scope=\"row\">(.*?)</th>", tr).group(1))
        out[head.replace("draft", "").strip()] = [re.sub(r"<[^>]+>", "", c) for c in re.findall(r"<td[^>]*>(.*?)</td>", tr)]
    return out


def test_counts_parts_and_alone():
    counts = table.class_counts(MODEL, REGISTRY)
    assert counts["mathematics"] == (2, 1)           # a and b; a alone; the reference node is not a part
    assert counts["other-peoples-code"] == (2, 1)    # b and c; c alone (two roots, one class)
    assert counts["self-consistency"] is None        # weak: never counted


def test_rendered_counts_and_uncounted_rows():
    rows = _rows(table.render_roots_table(MODEL, REGISTRY, NAMES))
    assert rows["Mathematics"][-2:] == ["2", "1"]
    assert rows["Other people's code"][-2:] == ["2", "1"]
    assert rows["Self-consistency"][-2:] == ["–", "–"]


def test_links_are_not_counted():
    rows = _rows(table.render_roots_table(MODEL, REGISTRY, NAMES))
    assert rows["Our reference code"][-2:] == ["–", "–"]
    assert 'class="vt-link"' in table.render_roots_table(MODEL, REGISTRY, NAMES)
    # the link's checks are not folded into any class
    assert table.class_counts(MODEL, REGISTRY)["mathematics"][0] == 2


def test_draft_marker_is_text_and_follows_reviewed_true_only():
    out = table.render_roots_table(MODEL, REGISTRY, NAMES)
    assert ">draft<" in out
    # the class without `reviewed` is a draft; the reviewed ones are not
    assert re.search(r"Other people's code <span class=\"vt-draft\"", out)
    assert not re.search(r"Mathematics <span class=\"vt-draft\"", out)
    # roots: pmoired (reviewed: false) and frap (no key) are drafts; frank (true) is not
    assert re.search(r"PMOIRED LABEL</strong>\s*<span class=\"vt-draft\"", out)
    assert re.search(r"FRAP LABEL</strong>.*?vt-draft", out)
    assert not re.search(r"FRANK LABEL</strong> <span class=\"vt-draft\"", out)
    # a draft text claims to be a draft, not only by colour
    assert "text no one has yet reviewed" in out


def test_root_labels_come_from_names():
    out = table.render_roots_table(MODEL, REGISTRY, NAMES)
    for label in NAMES.values():
        assert label in out
    # a root absent from `names` falls back to its id, never to a name in the registry
    assert "<strong>golden</strong>" in table.render_roots_reference(
        {**REGISTRY, "roots": REGISTRY["roots"] + [{"id": "golden", "class": "mathematics", "what": "w", "why": "y",
                                                    "cannot_catch": "c"}]}, NAMES)
    assert "counted as FRANK LABEL" in out


def test_labels_are_the_real_root_names():
    registry = table.load_registry(ROOT / "trust" / "roots.yml")
    out = table.render_roots_reference(registry, trust.ROOT_NAMES)
    for root in registry["roots"]:
        label = trust.ROOT_NAMES.get(root["id"], root["id"])
        assert f"<strong>{label}</strong>" in out.replace("&amp;", "&"), root["id"]


def test_reference_has_no_counts_and_no_caption_of_counts():
    out = table.render_roots_reference(REGISTRY, NAMES)
    assert "Parts" not in out and "Alone" not in out
    assert 'class="n"' not in out
    assert "Our reference code" in out


def test_text_is_escaped_and_code_spans_rendered():
    reg = yaml.safe_load(yaml.safe_dump(REGISTRY))
    reg["classes"]["mathematics"]["what"] = "x < y & `z`"
    out = table.render_roots_table(MODEL, reg, NAMES)
    assert "x &lt; y &amp; <code>z</code>" in out


def test_unknown_counted_root_is_an_error():
    model = {"nodes": {"virgil.x": _node("virgil.x", ["not-a-root"])}}
    with pytest.raises(KeyError, match="virgil.x"):
        table.class_counts(model, REGISTRY)


def test_trust_page_and_method_page_carry_the_table(tmp_path):
    graph = {"nodes": {"virgil.a": {"layer": "models"}}}
    path = tmp_path / "latest.jsonl"
    rec = {"record": "test", "test": "t::x", "objects": ["virgil.a"], "roots": ["mathematics"], "kind": "check",
           "tier": "A", "outcome": "passed", "metrics": {}}
    head = {"record": "run", "date": "2026-10-06T00:00:00+00:00", "virgil": {"commit": "c" * 40}, "runner": "test"}
    path.write_text("\n".join(__import__("json").dumps(x) for x in [head, rec]) + "\n")
    runs, records = trust.load_evidence([path])
    model = trust.build(graph, [], records, runs, None, golden=set())
    model["docs"] = {}
    page = trust.render(model)
    assert page.count('<table class="vt-roots">') == 1 and "Who checks what" in page
    assert ">draft<" in page  # the shipped registry is still a draft
    method = trust.render_roots()
    assert method.startswith("<!-- Generated") and "# The roots of trust" in method
    assert 'class="n"' not in method  # no counts on the method page
    assert "(index.md)" in method or "[How trust is built](index.md)" in method


def test_parts_equal_the_skyline_foundation_counts_over_parts_layers(monkeypatch):
    """The table's Parts and the skyline's per-layer counts are one number: the skyline's, summed over the
    layers of virgil (not ``references``), is the table's Parts for the same model."""
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    # skips only while scripts/trust_skyline.py is missing; a renamed foundation_counts then fails (AttributeError)
    sky = pytest.importorskip("trust_skyline")
    layers = {"virgil.a": "models", "virgil.b": "inference", "virgil.c": "inference", "virgil.d": "models",
              "crosscheck.ref": "references"}
    model = {"nodes": {k: {**v, "layer": layers[k]} for k, v in MODEL["nodes"].items()}}
    registry = table.load_registry(ROOT / "trust" / "roots.yml")
    counts = table.class_counts(model, registry)
    found = sky.foundation_counts(model, registry, trust.ROOT_ALIAS)
    for class_id, value in counts.items():
        if value is None:  # the weak class is counted nowhere
            assert class_id not in found
            continue
        per_layer = found.get(class_id, {})
        assert value[0] == sum(n for layer, n in per_layer.items() if layer != "references"), class_id
