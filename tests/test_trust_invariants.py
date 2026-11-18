"""Invariants of the verdict engine (scripts/trust.py ``build``) on random small graphs: making one input
worse never verifies anything new, the verdicts and their causes do not depend on the order of the inputs,
and ``build`` ends on every graph, cycles included. Plain ``random`` with fixed seeds, so a failure repeats;
a failure prints the graph cut down by ``shrink`` to a minimal one."""

import copy
import hashlib
import importlib.util
import pathlib
import random
import signal

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")

N_GRAPHS = 600
ROOT_NAMES = ["mathematics", "literature", "standards", "frap", "frank", "self-consistency", "golden:g1", "golden:unlisted"]
GOLDEN = {"g1"}


@pytest.fixture(scope="module")
def trust():
    spec = importlib.util.spec_from_file_location("trust", ROOT / "scripts" / "trust.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_graph(seed):
    """(nodes, records, ledger): 5-15 nodes of both layers with random ``depends`` (self-loops and cycles
    allowed), ``roots``, ``properties`` and ``convention``; records of every outcome, some from virgil's own
    tests, some of two objects, with random ``via``; ledger entries of every ruling and status."""
    rng = random.Random(seed)
    names = [f"{rng.choice(['ref', 'virgil'])}.{i}" for i in range(rng.randint(5, 15))]
    refs = [n for n in names if n.startswith("ref")]
    nodes = {}
    for n in names:
        nodes[n] = {"layer": "references" if n in refs else "models", "roots": rng.choice([1, 1, 1, 1, 2]),
                    "depends": rng.sample(names, rng.choice([0, 0, 1, 1, 2, 3]) % len(names))}
        if rng.random() < 0.15:
            nodes[n]["properties"] = {"p": 2}
        if rng.random() < 0.1:
            nodes[n]["convention"] = ["p"]
    records = []
    for n in names:
        for _ in range(rng.choice([0, 1, 1, 1, 2])):
            records.append({
                "test": f"t::{len(records)}", "objects": [n] + ([rng.choice(names)] if rng.random() < 0.1 else []),
                "roots": rng.sample(ROOT_NAMES, rng.choice([1, 1, 2, 3])),
                "kind": rng.choice(["check"] * 10 + ["reference", "regression", "guard"]),
                "outcome": rng.choice(["passed"] * 25 + ["failed", "skipped", "xfailed", "xpassed", "error"]),
                "via": rng.sample(refs, rng.randint(1, min(2, len(refs)))) if refs and rng.random() < 0.3 else [],
                "properties": ["p"] if rng.random() < 0.2 else [],
                "_commit": None, "_source": rng.choice(["ours"] * 9 + ["virgil-ci"]), "metrics": {}, "tier": "A"})
    ledger = [{"id": f"F{i}", "ruling": rng.choice(["virgil", "virgil", "crosscheck", "external:pkg"]),
               "status": rng.choice(["open", "to-raise", "raised", "fixed", "documented"]), "objects": [rng.choice(names)]}
              for i in range(rng.choice([0, 0, 0, 1, 2]))]
    return nodes, records, ledger


def run(trust, nodes, records, ledger):
    return trust.build({"nodes": nodes}, [dict(e) for e in ledger], [dict(r) for r in records], [],
                       virgil=None, golden=GOLDEN)["nodes"]


def verified(model):
    return {n for n, v in model.items() if v["verdict"] == "verified"}


def describe(nodes, records, ledger):
    lines = [f"{n} ({v['layer']}, roots={v['roots']}) depends={v['depends']} {v.get('properties', '')}{v.get('convention', '')}"
             for n, v in nodes.items()]
    lines += [f"{r['test']} {r['objects']} {r['kind']} {r['outcome']} roots={r['roots']} via={r['via']} {r['_source']}"
              for r in records]
    return "\n".join(lines + [str(e) for e in ledger])


def shrink(fails, nodes, records, ledger):
    """Drop nodes, records, ledger entries and ``depends`` edges while ``fails`` still holds."""
    nodes, records, ledger = copy.deepcopy((nodes, records, ledger))
    changed = True
    while changed:
        changed = False
        for n in list(nodes):
            trial = {k: dict(v, depends=[d for d in v["depends"] if d != n]) for k, v in nodes.items() if k != n}
            recs = [dict(r, objects=[o for o in r["objects"] if o != n], via=[v for v in r["via"] if v != n]) for r in records]
            recs = [r for r in recs if r["objects"]]
            led = [e for e in ledger if e["objects"] != [n]]
            if fails(trial, recs, led):
                nodes, records, ledger, changed = trial, recs, led, True
        for coll in (records, ledger):
            for x in list(coll):
                trial = [y for y in coll if y is not x]
                if fails(*((nodes, trial, ledger) if coll is records else (nodes, records, trial))):
                    coll[:] = trial
                    changed = True
        for n in nodes:
            for d in list(nodes[n]["depends"]):
                trial = copy.deepcopy(nodes)
                trial[n]["depends"].remove(d)
                if fails(trial, records, ledger):
                    nodes, changed = trial, True
    return nodes, records, ledger


# ------------------------------------------------------------------ the worsening moves
# A move is a concrete edit, ("kind", target...), picked once on the graph that failed and re-applied unchanged to
# every smaller graph the shrinker tries, so the shrunk graph shows the same edit.


def pick(kind, rng, trust, nodes, records, ledger):
    if kind == "failing":
        passing = [r["test"] for r in records if r["outcome"] == "passed"]
        return ("failing", rng.choice(passing)) if passing else None
    if kind == "remove-reference-check":
        ok = sorted(n for n in verified(run(trust, nodes, records, ledger)) if nodes[n]["layer"] == "references"
                    and any(n in r["objects"] for r in records))
        return ("remove-reference-check", rng.choice(ok)) if ok else None
    if kind == "depends":
        n, d = rng.sample(sorted(nodes), 2)
        return ("depends", n, d) if d not in nodes[n]["depends"] else None
    return ("finding", rng.choice(sorted(nodes)))


def apply(trust, move, nodes, records, ledger):
    """The graph made worse by ``move``, or None if the move does not apply to it."""
    kind, *target = move
    if kind == "failing":
        if not any(r["test"] == target[0] and r["outcome"] == "passed" for r in records):
            return None
        return nodes, [dict(r, outcome="failed") if r["test"] == target[0] else r for r in records], ledger
    if kind == "remove-reference-check":  # a verified reference loses its checks, so it is unchecked
        n = target[0]
        if n not in nodes or n not in verified(run(trust, nodes, records, ledger)) or not any(n in r["objects"] for r in records):
            return None
        return nodes, [r for r in records if n not in r["objects"]], ledger
    if kind == "depends":
        n, d = target
        if n not in nodes or d not in nodes or d in nodes[n]["depends"]:
            return None
        return {k: dict(v, depends=v["depends"] + [d] if k == n else v["depends"]) for k, v in nodes.items()}, records, ledger
    if target[0] not in nodes:
        return None
    return nodes, records, ledger + [{"id": "NEW", "ruling": "virgil", "status": "open", "objects": [target[0]]}]


@pytest.mark.parametrize("kind", ["failing", "remove-reference-check", "depends", "finding"])
def test_making_an_input_worse_never_verifies_anything(trust, kind):
    tried = 0
    for seed in range(N_GRAPHS):
        graph = make_graph(seed)
        move = pick(kind, random.Random(f"{kind}-{seed}"), trust, *graph)
        worse = apply(trust, move, *graph) if move else None
        if worse is None:
            continue
        tried += 1
        newly = verified(run(trust, *worse)) - verified(run(trust, *graph))
        if newly:
            def fails(nd, rc, ld):
                w = apply(trust, move, nd, rc, ld)
                return w is not None and bool(verified(run(trust, *w)) - verified(run(trust, nd, rc, ld)))
            pytest.fail(f"{move}, seed {seed}: newly verified {sorted(newly)}\n{describe(*shrink(fails, *graph))}")
    assert tried > N_GRAPHS // 8  # the move applied to enough graphs to mean something


# ------------------------------------------------------------------ order


def signature(model):
    """Everything a reader sees of a verdict, with the cause lists as sets."""
    return {n: (v["verdict"], v["verdict_label"], v["label"], v["depth"], tuple(v["strong"]), tuple(v["weak"]), v["skipped"],
                tuple(v["unconfirmed"]), tuple(sorted(v["findings"])), tuple(sorted(v["dependents"])),
                frozenset((c["id"], c["status"], tuple(sorted(c["findings"]))) for c in v["because"]))
            for n, v in model.items()}


def reorder(nodes, records, ledger, salt):
    """The same graph in another order, fixed by ``salt`` (so the same relative order on any subgraph): the nodes,
    each ``depends`` list, the records, each record's ``roots`` and ``via``, the ledger."""
    def shuffle(xs, key=str):
        return sorted(xs, key=lambda x: hashlib.md5(f"{salt}|{key(x)}".encode()).hexdigest())
    return ({n: dict(nodes[n], depends=shuffle(nodes[n]["depends"])) for n in shuffle(list(nodes))},
            [dict(r, roots=shuffle(r["roots"]), via=shuffle(r["via"]), objects=shuffle(r["objects"]))
             for r in shuffle(records, lambda r: r["test"])], shuffle(ledger, lambda e: e["id"]))


def test_order_of_nodes_depends_records_via_and_ledger_does_not_matter(trust):
    for seed in range(N_GRAPHS):
        graph = make_graph(seed)
        base = signature(run(trust, *graph))
        for salt in range(3):
            if signature(run(trust, *reorder(*graph, f"{seed}-{salt}"))) != base:
                def fails(nd, rc, ld):
                    return signature(run(trust, nd, rc, ld)) != signature(run(trust, *reorder(nd, rc, ld, f"{seed}-{salt}")))
                got = signature(run(trust, *reorder(*graph, f"{seed}-{salt}")))
                bad = sorted(n for n in base if base[n] != got[n])
                pytest.fail(f"seed {seed}: differs at {bad}\n{describe(*shrink(fails, *graph))}")


def record(n, via=()):
    return {"test": f"t::{n}", "objects": [n], "roots": ["mathematics"], "kind": "check", "outcome": "passed", "via": list(via),
            "_commit": None, "_source": "ours", "metrics": {}, "tier": "A"}


def test_a_cycle_below_a_part_is_named_whole_and_never_as_a_part_own_cause(trust):
    """Review of #271: ``because`` named whichever cycle member a depth-first walk met first, so it followed the
    order of ``depends``. It now names every member of a cycle the part reaches, and nothing for a cycle through the part."""
    def causes(step):
        deps = {"virgil.0": ["virgil.3", "ref.1"], "ref.1": ["virgil.3"], "virgil.3": ["ref.1"], "virgil.a": ["virgil.0"]}
        nodes = {n: {"layer": "references" if n.startswith("ref") else "models", "depends": d[::step]} for n, d in deps.items()}
        model = run(trust, nodes, [record("ref.1"), record("virgil.a")], [])
        return {n: sorted(c["id"] for c in v["because"]) for n, v in model.items()}
    assert causes(1) == causes(-1)
    assert causes(1)["virgil.0"] == ["ref.1", "virgil.3"] and causes(1)["ref.1"] == ["virgil.3"]
    assert causes(1)["virgil.3"] == [] and "virgil.0" not in causes(1)["virgil.0"]


# ------------------------------------------------------------------ termination


def test_build_ends_on_every_graph_cycles_included(trust):
    def timeout(*_):
        raise TimeoutError("build did not end within 10 s")
    alarm = hasattr(signal, "SIGALRM")
    old = signal.signal(signal.SIGALRM, timeout) if alarm else None
    try:
        for seed in range(N_GRAPHS):
            nodes, records, ledger = make_graph(seed)
            if alarm:
                signal.alarm(10)
            try:
                model = run(trust, nodes, records, ledger)
            finally:
                if alarm:
                    signal.alarm(0)
            assert set(model) == set(nodes)
    finally:
        if alarm:
            signal.signal(signal.SIGALRM, old)
    # a deliberate cycle: two parts that rely on each other, and a reference that vouches for itself
    nodes = {"virgil.a": {"layer": "models", "depends": ["virgil.b"]}, "virgil.b": {"layer": "models", "depends": ["virgil.a"]},
             "ref.c": {"layer": "references", "depends": ["ref.c"]}}
    for via in ([], ["ref.c"]):
        model = run(trust, nodes, [record(n, via if n == "virgil.a" else []) for n in nodes], [])
        assert set(model) == set(nodes)
        for n, v in model.items():
            assert n not in [c["id"] for c in v["because"]], f"{n} is its own cause"  # a cycle never vouches for itself
    # the reference that depends on itself, and the part checked through it, are not verified
    assert model["ref.c"]["verdict"] != "verified" and model["virgil.a"]["verdict"] != "verified"
