"""The cone and its sentence (scripts/trust_cone.py) on a small, made-up model."""

import pathlib
import sys

import pytest

pytestmark = pytest.mark.validates("evidence", roots=["standards"], kind="guard")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import trust_cone as tc  # noqa: E402


def node(name, verdict, depends=(), depth=0, strong=(), findings=()):
    return {"id": name, "label": name, "verdict": verdict, "depends": list(depends), "depth": depth,
            "strong": list(strong), "weak": [], "findings": list(findings)}


NODES = [
    node("ground", "verified", depth=0, strong=["mathematics", "standards"]),
    node("ground2", "unchecked", depth=0),
    node("mid", "verified", ["ground"], 1, ["pmoired"]),
    node("buggy", "bug", ["ground2"], 1, ["mathematics"], ["F27"]),
    node("top", "relies-unverified", ["mid", "buggy"], 2, ["mathematics"]),
    node("loneliest", "unchecked"),
    # a cycle that the engine would not build, to check that the walk ends
    node("a", "partly", ["b"], 1), node("b", "partly", ["a"], 1),
]
MODEL = {
    "nodes": {n["id"]: n for n in NODES},
    "pipelines": [
        {"id": "pipe", "title": "Example", "data": "simulated", "status": "passes", "steps": ["mid", "buggy", "gone"]},
        {"id": "top", "title": "Shadowed by the part", "status": "planned", "steps": ["ground"]},
    ],
}
NAMES = {"mathematics": "mathematics", "standards": "standards", "pmoired": "PMOIRED"}


def test_cone_of_a_part_is_everything_beneath_it():
    c = tc.cone(MODEL, "top")
    assert c["kind"] == "node" and c["top"] == ["top"]
    assert c["nodes"] == {"buggy": 1, "ground": 0, "ground2": 0, "mid": 1, "top": 2}
    assert c["edges"] == [("buggy", "ground2"), ("mid", "ground"), ("top", "buggy"), ("top", "mid")]


def test_cone_of_a_part_on_the_ground_is_itself():
    c = tc.cone(MODEL, "loneliest")
    assert c["nodes"] == {"loneliest": 0} and c["edges"] == []


def test_a_cycle_ends_the_walk():
    c = tc.cone(MODEL, "a")
    assert set(c["nodes"]) == {"a", "b"} and c["edges"] == [("a", "b"), ("b", "a")]
    assert "relies on 1 part" in tc.sentence(MODEL, "a")


def test_cone_of_a_pipeline_is_the_union_of_its_steps():
    c = tc.cone(MODEL, "pipe")
    assert c["kind"] == "pipeline"
    assert c["top"] == ["mid", "buggy"]  # a step that is not a part of the model is left out
    assert set(c["nodes"]) == {"mid", "buggy", "ground", "ground2"}
    assert c["nodes"] == {n: MODEL["nodes"][n]["depth"] for n in c["nodes"]}


def test_a_part_wins_over_a_pipeline_of_the_same_id():
    assert tc.cone(MODEL, "top")["kind"] == "node"


def test_unknown_id_raises():
    for f in (tc.cone, tc.sentence):
        with pytest.raises(KeyError):
            f(MODEL, "nothing-here")


def test_sentence_for_a_part_with_unverified_parts_beneath():
    s = tc.sentence(MODEL, "top", NAMES)
    assert s.startswith("top works, but relies on a part that is not yet verified. ")
    assert "Its own checks agree with mathematics." in s
    assert "It relies on 4 parts, of which 2 are not yet verified (ground2: not yet checked; buggy: known bug, F27)." in s
    assert s.endswith("Its cone reaches the ground in 2 parts.")


def test_sentence_for_a_verified_part_on_the_ground():
    s = tc.sentence(MODEL, "ground", NAMES)
    assert s == ("ground is verified. Its own checks agree with mathematics and standards. "
                 "It relies on no other part, so it stands directly on the roots.")


def test_sentence_when_everything_beneath_is_verified():
    s = tc.sentence(MODEL, "mid", NAMES)
    assert s == ("mid is verified. Its own checks agree with PMOIRED. It relies on 1 part, all verified. "
                 "Its cone reaches the ground in 1 part.")


def test_sentence_for_a_bug_and_for_an_unchecked_part():
    assert tc.sentence(MODEL, "buggy").startswith("buggy has a known bug, fix pending (F27). ")
    assert tc.sentence(MODEL, "loneliest").startswith("loneliest has no independent check yet. It relies on no other part")
    assert "None of its checks counts yet" in tc.sentence(MODEL, "a")


def test_sentence_for_a_pipeline():
    s = tc.sentence(MODEL, "pipe", NAMES)
    assert s == ("The pipeline “Example” runs 2 steps end to end on simulated data. "
                 "Its end-to-end check passes, but not every step is verified yet. "
                 "Its steps rest on 4 parts in all, of which 2 are not yet verified (ground2: not yet checked; buggy: known bug, F27). "
                 "Its cone reaches the ground in 2 parts.")


@pytest.mark.parametrize("id", ["top", "ground", "pipe", "buggy", "a"])
def test_no_sentence_claims_more_than_the_model_holds(id):
    s = tc.sentence(MODEL, id, NAMES)
    for word in ("independently verified", "corroborated"):
        assert word not in s


def test_embed_carries_every_cone_and_sentence_once():
    roots = {"roots": [{"id": "mathematics", "class": "mathematics"}, {"id": "frank", "class": "other-peoples-code"},
                       {"id": "self-consistency", "class": "self-consistency"}]}
    e = tc.embed(MODEL, roots, NAMES, alias={"frap": "frank"})
    assert e["order"] == sorted(MODEL["nodes"])
    assert set(e["cones"]) == set(e["sentences"]) == set(MODEL["nodes"]) | {"pipe"}
    idx = lambda ids: [e["order"][i] for i in ids]  # noqa: E731
    assert idx(e["cones"]["top"])[0] == "top" and set(idx(e["cones"]["top"])) == set(tc.cone(MODEL, "top")["nodes"])
    assert idx(e["cones"]["pipe"])[:2] == ["mid", "buggy"]  # a pipeline's steps come first, in order
    assert e["letters"] == {"mathematics": "M", "frank": "C", "frap": "C"}
    assert e["pipelines"]["pipe"]["steps"] == ["mid", "buggy", "gone"]
    assert e["sentences"]["top"] == tc.sentence(MODEL, "top", NAMES)
