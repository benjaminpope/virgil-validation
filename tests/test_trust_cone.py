"""The cone and its sentence (scripts/trust_cone.py) on a small, made-up model."""

import pathlib
import sys

import pytest

pytestmark = pytest.mark.validates("evidence", roots=["standards"], kind="guard")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import trust_cone as tc  # noqa: E402


def node(name, verdict, depends=(), depth=0, strong=(), findings=(), because=(), layer="models"):
    return {"id": name, "label": name, "layer": layer, "verdict": verdict, "depends": list(depends), "depth": depth,
            "strong": list(strong), "weak": [], "findings": list(findings),
            "because": [{"id": b, "findings": [], "status": "unchecked"} for b in because]}


NODES = [
    node("ground", "verified", depth=0, strong=["mathematics", "standards"], layer="data"),
    node("ground2", "unchecked", depth=0),
    node("mid", "verified", ["ground"], 1, ["pmoired"]),
    node("buggy", "bug", ["ground2"], 1, ["mathematics"], ["F27"]),
    node("top", "relies-unverified", ["mid", "buggy"], 2, ["mathematics"]),
    node("loneliest", "unchecked"),
    # reaches no other part through `depends`, but its checks go through reference code that is not verified (`via`)
    node("viaonly", "relies-unverified", strong=["mathematics"], because=["ground2", "buggy"]),
    node("viamix", "relies-unverified", ["mid"], 2, ["mathematics"], because=["mid", "ground2"]),
    # a cycle that the engine would not build, to check that the walk ends
    node("a", "partly", ["b"], 1), node("b", "partly", ["a"], 1),
]
MODEL = {
    "nodes": {n["id"]: n for n in NODES},
    "pipelines": [
        {"id": "viapipe", "title": "Via", "data": "simulated", "status": "passes", "steps": ["mid"],
         "unverified": ["ground2", "mid"]},
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
    assert "It relies on 4 parts, of which 2 are not yet verified (buggy: known bug, F27; ground2: not yet checked)." in s
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
    assert tc.sentence(MODEL, "loneliest") == (
        "loneliest has no independent check yet. It relies on no other part.")  # not "stands on the roots": nothing verifies it
    assert "None of its checks counts yet" in tc.sentence(MODEL, "a")


def test_a_cone_that_is_a_loop_has_no_ground_and_says_so():
    s = tc.sentence(MODEL, "a")
    assert "Its cone has no ground part" in s and "in 0 parts" not in s


def test_ground_counts_only_edges_inside_the_cone():
    # a part whose only dependency is not in the model stands on the ground; a self-loop is not a reliance
    m = {"nodes": {"x": node("x", "verified", ["missing", "x"]), "y": node("y", "verified", ["x"], 1)}, "pipelines": []}
    assert tc.sentence(m, "y").endswith("Its cone reaches the ground in 1 part.")


def test_sentence_for_a_pipeline():
    s = tc.sentence(MODEL, "pipe", NAMES)
    assert s == ("The pipeline “Example” runs 2 steps end to end on simulated data. "
                 "Its end-to-end check passes, but not every step is verified yet. "
                 "Its steps rest on 4 parts in all, of which 2 are not yet verified (buggy: known bug, F27; ground2: not yet checked). "
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
    assert set(e["cones"]) == set(e["sentences"]) == set(MODEL["nodes"]) | {"pipe", "viapipe"}
    idx = lambda ids: [e["order"][i] for i in ids]  # noqa: E731
    assert idx(e["cones"]["top"])[0] == "top" and set(idx(e["cones"]["top"])) == set(tc.cone(MODEL, "top")["nodes"])
    assert idx(e["cones"]["pipe"])[:2] == ["mid", "buggy"]  # a pipeline's steps come first, in order
    assert e["root_class"] == {"mathematics": "mathematics", "frank": "other-peoples-code", "self-consistency": "self-consistency",
                               "frap": "other-peoples-code"}
    assert e["pipelines"]["pipe"]["steps"] == ["mid", "buggy", "gone"]
    assert e["sentences"]["top"] == tc.sentence(MODEL, "top", NAMES)


def test_a_part_that_relies_through_via_does_not_claim_to_stand_on_the_roots():
    s = tc.sentence(MODEL, "viaonly", NAMES)
    assert s.startswith("viaonly works, but relies on a part that is not yet verified. ")
    assert "stands directly on the roots" not in s and "all verified" not in s
    assert s.endswith("It relies on no other part. Its checks also go through 2 reference parts that are not yet verified "
                      "(buggy: known bug, F27; ground2: not yet checked).")


def test_via_parts_are_added_to_a_cone_that_is_otherwise_all_verified():
    s = tc.sentence(MODEL, "viamix", NAMES)
    assert "It relies on 2 parts, all verified. Its checks also go through 1 reference part that is not yet verified (ground2: not yet checked)." in s
    assert "mid" not in s.split("go through")[1]  # a part already in the cone is not named twice


def test_a_pipeline_names_the_untrusted_references_outside_its_steps():
    s = tc.sentence(MODEL, "viapipe", NAMES)
    assert "Its steps rest on 2 parts in all, all verified. Its checks also go through 1 reference part that is not yet verified (ground2: not yet checked)." in s


def test_the_most_serious_unverified_parts_are_named_first():
    deep = [node("deepbug", "bug", ["g1"], 3, findings=["F1"]), node("g1", "unchecked", depth=0),
            node("g2", "unchecked", depth=0), node("g3", "unchecked", depth=0), node("g4", "unchecked", depth=0),
            node("failer", "failing", ["g2"], 2),
            node("root", "relies", ["deepbug", "failer", "g3", "g4"], 4)]
    m = {"nodes": {n["id"]: n for n in deep}, "pipelines": []}
    s = tc.sentence(m, "root")
    assert "(failer: check failing; deepbug: known bug, F1; g1: not yet checked; and 3 more)" in s


def test_a_golden_root_has_words_even_when_the_caller_gives_none():
    m = {"nodes": {"g": node("g", "verified", strong=["golden", "golden:pmoired.foo"])}, "pipelines": []}
    assert tc.sentence(m, "g").startswith("g is verified. Its own checks agree with golden values and reference code: pmoired.foo.")


def test_a_pipeline_shadowed_by_a_part_is_not_embedded():
    e = tc.embed(MODEL, {"roots": []}, NAMES)
    assert "top" not in e["pipelines"] and "pipe" in e["pipelines"]


# ---- the lens: blocks, critical path, longest chain, fix list


def test_blocks_counts_the_not_verified_parts_that_rely_on_a_part():
    b = tc.blocks(MODEL)
    assert b["ground2"] == 2   # buggy and top rely on it and are not verified
    assert b["ground"] == 2    # mid is verified and does not count; top and viamix are not verified
    assert b["top"] == 0 and b["loneliest"] == 0
    assert b["a"] == b["b"] == 1  # a cycle ends the walk, and a part never blocks itself


def test_critical_path_runs_to_the_worst_part_beneath_the_selection():
    # buggy (a known bug) is worse than ground2 (not checked), though ground2 blocks more
    assert tc.critical_path(MODEL, "top") == ["top", "buggy"]
    # the selection is never its own target: a pipeline's steps are the top, so the path starts at a step
    assert tc.critical_path(MODEL, "pipe") == ["buggy", "ground2"]


def test_a_verified_cone_has_no_critical_path_and_nothing_to_fix():
    assert tc.critical_path(MODEL, "mid") == [] and tc.fix_list(MODEL, "mid") == []
    assert tc.critical_path(MODEL, "ground") == [] and tc.fix_list(MODEL, "ground") == []


def test_critical_path_ties_go_to_the_part_that_blocks_more_then_to_the_id():
    ns = [node("r", "relies-unverified", ["x", "y", "z"], 1), node("x", "unchecked"), node("y", "unchecked"),
          node("z", "unchecked"), node("p", "relies-unverified", ["y"], 1)]
    m = {"nodes": {n["id"]: n for n in ns}, "pipelines": []}
    assert tc.critical_path(m, "r") == ["r", "y"]   # y blocks r and p; x and z only r
    ns[-1]["depends"] = []
    assert tc.critical_path(m, "r") == ["r", "x"]   # equal blocks: the smaller id


def test_critical_path_is_the_shortest_chain():
    ns = [node("t", "relies-unverified", ["m", "w"], 3), node("m", "relies-unverified", ["w"], 2), node("w", "bug", ["g"], 1),
          node("g", "verified", depth=0)]
    ns[0]["depends"] = ["m", "w"]
    m = {"nodes": {n["id"]: n for n in ns}, "pipelines": []}
    assert tc.critical_path(m, "t") == ["t", "w"]   # directly, not by way of m


def test_longest_chain_follows_the_deepest_part_to_the_ground():
    assert tc.longest_chain(MODEL, "top") == ["top", "buggy", "ground2"]  # mid and buggy tie on depth: the smaller id
    assert tc.longest_chain(MODEL, "pipe") == ["buggy", "ground2"]        # from the deepest step
    assert tc.longest_chain(MODEL, "ground") == ["ground"]
    assert tc.longest_chain(MODEL, "a") == ["a", "b"]                     # a cycle ends it


def test_fix_list_orders_by_what_each_part_blocks_and_drops_a_relies_only_selection():
    assert tc.fix_list(MODEL, "top") == [("ground2", ""), ("buggy", "")]
    assert tc.fix_list(MODEL, "loneliest") == [("loneliest", "this part")]  # its own problem stays in its list
    assert tc.fix_list(MODEL, "buggy")[0] == ("ground2", "")                 # a bug of its own is listed too
    assert ("buggy", "this part") in tc.fix_list(MODEL, "buggy")


def test_fix_list_names_the_reference_parts_a_part_or_pipeline_goes_through():
    assert tc.fix_list(MODEL, "viaonly") == [("ground2", "via"), ("buggy", "via")]
    assert tc.fix_list(MODEL, "viapipe") == [("ground2", "via")]  # mid is a step, so already in its cone


def test_verified_beneath_is_a_count_by_layer_without_the_selection():
    assert tc.verified_beneath(MODEL, "top") == (2, [("data", 1), ("models", 1)])
    assert tc.verified_beneath(MODEL, "ground") == (0, [])
    assert tc.verified_beneath(MODEL, "pipe")[0] == 2  # mid and ground


def test_embed_carries_the_lens_as_indices():
    e = tc.embed(MODEL, {"roots": []}, NAMES)
    name = lambda i: e["order"][i]  # noqa: E731
    lens = e["lens"]["top"]
    assert [name(i) for i in lens["crit"]] == ["top", "buggy"] and [name(i) for i in lens["chain"]] == ["top", "buggy", "ground2"]
    assert [(name(i), n) for i, n in lens["fix"]] == [("ground2", 0), ("buggy", 0)]
    assert lens["ver"] == [2, [["data", 1], ["models", 1]]]
    assert [e["blocks"][e["order"].index(n)] for n in ("ground2", "top")] == [2, 0]
    assert [(name(i), n) for i, n in e["lens"]["viaonly"]["fix"]] == [("ground2", 2), ("buggy", 2)]
    assert set(e["lens"]) == set(e["cones"])
