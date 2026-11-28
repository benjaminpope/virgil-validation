"""What the Trust page carries for the cone panel: skyline hooks, the panel frame and the embedded data."""

import importlib.util
import json
import pathlib
import xml.etree.ElementTree as ET

import pytest
import yaml

pytestmark = pytest.mark.validates("evidence", roots=["standards"], kind="guard")

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("trust", ROOT / "scripts" / "trust.py")
trust = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trust)
import trust_cone  # noqa: E402  (scripts/ is on sys.path once trust.py is loaded)
import trust_render  # noqa: E402
import trust_skyline  # noqa: E402


def node(name, depth, depends=(), verdict="verified"):
    return {"id": name, "label": name, "layer": "models", "verdict": verdict, "depth": depth,
            "strong": ["mathematics"], "weak": [], "depends": list(depends)}


MODEL = {"nodes": {"a": node("a", 0), "b": node("b", 1, ["a"])}, "docs": {"a": "tests/models/index.md#a"},
         "pipelines": [{"id": "p", "title": "P", "status": "planned", "steps": ["b"], "data": "simulated", "page": "orbits/"}]}
ROOTS = yaml.safe_load(open(ROOT / "trust" / "roots.yml"))


def test_every_skyline_square_names_its_part_for_the_cone():
    tree = ET.fromstring(trust_skyline.skyline_html(MODEL, ROOTS, "../", names=trust))
    marked = {el.get("data-node"): el.tag for el in tree.iter() if el.get("data-node")}
    # a part with a page is a link (it works without JavaScript); one without is a bare group
    assert marked == {"a": "a", "b": "g"}
    for el in tree.iter("a"):
        assert el.get("href")


def test_panel_frame_says_what_to_do_without_javascript():
    html = trust_render._cone_panel()
    tree = ET.fromstring(html)
    assert tree.get("id") == "vt-cone"
    assert "JavaScript" in "".join(tree.itertext())


def test_embedded_cone_data_is_json_and_closed_under_depends():
    data = trust_cone.embed(MODEL, ROOTS, trust.ROOT_NAMES, trust.ROOT_ALIAS)
    data = json.loads(json.dumps(data))  # what the page carries
    names = {k: [data["order"][i] for i in v] for k, v in data["cones"].items()}
    assert names == {"a": ["a"], "b": ["b", "a"], "p": ["b", "a"]}
    assert data["pipelines"]["p"]["steps"] == ["b"]
    assert data["root_class"]["mathematics"] == "mathematics" and data["root_class"]["standards"] == "standards"
    assert data["lens"]["b"]["chain"] == [data["order"].index(n) for n in ("b", "a")]
    assert data["lens"]["a"]["ver"] == [0, []] and data["blocks"] == [0, 0]
    assert data["short"]["verified"] == "verified"

