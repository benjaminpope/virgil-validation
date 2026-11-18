"""Invariants of the verdict engine (scripts/trust.py ``build``) on random small graphs: making one input
worse never verifies anything new, the verdicts and their causes do not depend on the order of the inputs,
and ``build`` ends on every graph, cycles included. Plain ``random`` with fixed seeds, so a failure repeats;
``describe`` prints the failing graph and ``shrink`` cuts it down to a minimal one."""

import copy
import importlib.util
import pathlib
import random
import signal

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")

N_GRAPHS = 600
ROOT_NAMES = ["mathematics", "literature", "standards"]


@pytest.fixture(scope="module")
def trust():
    spec = importlib.util.spec_from_file_location("trust", ROOT / "scripts" / "trust.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_graph(seed):
    """(nodes, records, ledger): 5-15 nodes, a mix of reference and virgil layers, random ``depends``
    (cycles allowed), random ``via`` on checks, random outcomes and open findings."""
    rng = random.Random(seed)
    names = [f"{rng.choice(['ref', 'virgil'])}.{i}" for i in range(rng.randint(5, 15))]
    refs = [n for n in names if n.startswith("ref")]
    nodes = {}
    for n in names:
        others = [m for m in names if m != n]
        nodes[n] = {"layer": "references" if n in refs else "models", "roots": rng.choice([1, 1, 1, 1, 2]),
                    "depends": rng.sample(others, rng.choice([0, 0, 1, 1, 2, 3]) % (len(others) + 1))}
    records = []
    for n in names:
        for _ in range(rng.choice([0, 1, 1, 1, 2])):
            records.append({
                "test": f"t::{len(records)}", "objects": [n], "roots": rng.sample(ROOT_NAMES, rng.randint(1, 3)),
                "kind": rng.choice(["check"] * 10 + ["reference", "regression", "guard"]),
                "outcome": rng.choice(["passed"] * 25 + ["failed", "skipped"]),
                "via": rng.sample(refs, rng.randint(1, min(2, len(refs)))) if refs and rng.random() < 0.3 else [],
                "_commit": None, "_source": "ours", "metrics": {}, "tier": "A"})
    ledger = [{"id": f"F{i}", "ruling": rng.choice(["virgil", "virgil", "crosscheck", "external:pkg"]),
               "status": rng.choice(["open", "to-raise", "fixed"]), "objects": [rng.choice(names)]}
              for i in range(rng.choice([0, 0, 0, 1, 2]))]
    return nodes, records, ledger


def run(trust, nodes, records, ledger):
    return trust.build({"nodes": nodes}, [dict(e) for e in ledger], [dict(r) for r in records], [],
                       virgil=None, golden=set())["nodes"]


def verified(model):
    return {n for n, v in model.items() if v["verdict"] == "verified"}


def describe(nodes, records, ledger):
    lines = [f"{n} ({v['layer']}, roots={v['roots']}) depends={v['depends']}" for n, v in nodes.items()]
    lines += [f"{r['test']} {r['objects']} {r['kind']} {r['outcome']} roots={r['roots']} via={r['via']}" for r in records]
    return "\n".join(lines + [str(e) for e in ledger])


def shrink(fails, nodes, records, ledger):
    """Drop nodes, records, ledger entries and ``depends`` edges while ``fails`` still holds."""
    nodes, records, ledger = copy.deepcopy((nodes, records, ledger))
    changed = True
    while changed:
        changed = False
        for n in list(nodes):
            trial = {k: dict(v, depends=[d for d in v["depends"] if d != n]) for k, v in nodes.items() if k != n}
            recs = [dict(r, via=[v for v in r["via"] if v != n]) for r in records if r["objects"] != [n]]
            led = [e for e in ledger if e["objects"] != [n]]
            if fails(trial, recs, led):
                nodes, records, ledger, changed = trial, recs, led, True
        for coll in (records, ledger):
            for x in list(coll):
                trial = [y for y in coll if y is not x]
                args = (nodes, trial, ledger) if coll is records else (nodes, records, trial)
                if fails(*args):
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


def worsen_failing(rng, nodes, records, ledger):
    passing = [i for i, r in enumerate(records) if r["outcome"] == "passed"]
    if not passing:
        return None
    records = copy.deepcopy(records)
    records[rng.choice(passing)]["outcome"] = "failed"
    return nodes, records, ledger


def worsen_remove_reference_check(rng, nodes, records, ledger, trust=None):
    model = run(trust, nodes, records, ledger)
    ok = [n for n in verified(model) if nodes[n]["layer"] == "references" and any(r["objects"] == [n] for r in records)]
    if not ok:
        return None
    gone = rng.choice(sorted(ok))
    return nodes, [r for r in records if r["objects"] != [gone]], ledger


def worsen_depends(rng, nodes, records, ledger):
    n, d = rng.sample(sorted(nodes), 2)
    if d in nodes[n]["depends"]:
        return None
    nodes = copy.deepcopy(nodes)
    nodes[n]["depends"].append(d)
    return nodes, records, ledger


def worsen_finding(rng, nodes, records, ledger):
    return nodes, records, ledger + [{"id": "NEW", "ruling": "virgil", "status": "open", "objects": [rng.choice(sorted(nodes))]}]


WORSENINGS = {"failing": worsen_failing, "remove-reference-check": worsen_remove_reference_check,
              "depends": worsen_depends, "finding": worsen_finding}


@pytest.mark.parametrize("kind", sorted(WORSENINGS))
def test_making_an_input_worse_never_verifies_anything(trust, kind):
    tried = 0
    for seed in range(N_GRAPHS):
        rng = random.Random(f"{kind}-{seed}")
        nodes, records, ledger = make_graph(seed)
        move = WORSENINGS[kind]
        worse = (move(rng, nodes, records, ledger, trust=trust) if kind == "remove-reference-check"
                 else move(rng, nodes, records, ledger))
        if worse is None:
            continue
        tried += 1
        before, after = verified(run(trust, nodes, records, ledger)), verified(run(trust, *worse))
        if not after <= before:
            def fails(nd, rc, ld):  # the same worsening, drawn with the same seed, on the smaller graph
                w = (move(random.Random(f"{kind}-{seed}"), nd, rc, ld, trust=trust) if kind == "remove-reference-check"
                     else move(random.Random(f"{kind}-{seed}"), nd, rc, ld))
                return w is not None and not verified(run(trust, *w)) <= verified(run(trust, nd, rc, ld))
            small = shrink(fails, nodes, records, ledger) if fails(nodes, records, ledger) else (nodes, records, ledger)
            pytest.fail(f"{kind}, seed {seed}: newly verified {sorted(after - before)}\n{describe(*small)}")
    assert tried > N_GRAPHS // 8  # the move applied to enough graphs to mean something


# ------------------------------------------------------------------ order


def signature(model):
    """Everything a reader sees of a verdict, with the cause lists as sets."""
    return {n: (v["verdict"], v["verdict_label"], v["label"], tuple(v["strong"]), tuple(v["weak"]), tuple(sorted(v["findings"])),
                frozenset((c["id"], c["status"], tuple(sorted(c["findings"]))) for c in v["because"]))
            for n, v in model.items()}


def test_order_of_nodes_records_and_ledger_does_not_matter(trust):
    for seed in range(N_GRAPHS):
        rng = random.Random(f"order-{seed}")
        nodes, records, ledger = make_graph(seed)
        base = signature(run(trust, nodes, records, ledger))
        for _ in range(3):
            order = rng.sample(sorted(nodes), len(nodes))
            shuffled = {n: nodes[n] for n in order}
            recs, led = rng.sample(records, len(records)), rng.sample(ledger, len(ledger))
            got = signature(run(trust, shuffled, recs, led))
            if got != base:
                def fails(nd, rc, ld):
                    return signature(run(trust, nd, rc, ld)) != signature(run(trust, dict(reversed(list(nd.items()))), rc[::-1], ld[::-1]))
                small = shrink(fails, nodes, records, ledger) if fails(nodes, records, ledger) else (nodes, records, ledger)
                bad = sorted(n for n in base if base[n] != got[n])
                pytest.fail(f"seed {seed}: differs at {bad}\n{describe(*small)}")


def reversed_depends(nodes):
    return {n: dict(v, depends=v["depends"][::-1]) for n, v in nodes.items()}


def test_order_of_depends_does_not_change_a_verdict(trust):
    for seed in range(N_GRAPHS):
        nodes, records, ledger = make_graph(seed)
        got = {n: v["verdict"] for n, v in run(trust, reversed_depends(nodes), records, ledger).items()}
        assert got == {n: v["verdict"] for n, v in run(trust, nodes, records, ledger).items()}, f"seed {seed}"


@pytest.mark.xfail(strict=True, reason=(
    "the causes (`because`) of a part in a cycle depend on the order of its `depends` list: _raw_problems names the "
    "target of whichever back edge its depth-first walk meets first. 79 of 600 random graphs differ; verdicts never do. "
    "Smallest case: virgil.0 depends [virgil.3, ref.1]; ref.1 depends [virgil.3]; virgil.3 depends [ref.1]; one passing "
    "check of ref.1. virgil.0.because is [virgil.3] in that order and [ref.1, virgil.3] with virgil.0's depends reversed."))
def test_order_of_depends_does_not_change_the_causes(trust):
    for seed in range(N_GRAPHS):
        nodes, records, ledger = make_graph(seed)
        assert (signature(run(trust, reversed_depends(nodes), records, ledger))
                == signature(run(trust, nodes, records, ledger))), f"seed {seed}"


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
    # a deliberate cycle: two parts that each rely on the other, and a reference that vouches for itself
    nodes = {"virgil.a": {"layer": "models", "depends": ["virgil.b"]}, "virgil.b": {"layer": "models", "depends": ["virgil.a"]},
             "ref.c": {"layer": "references", "depends": ["ref.c"]}}
    records = [{"test": f"t::{n}", "objects": [n], "roots": ["mathematics"], "kind": "check", "outcome": "passed",
                "via": ["ref.c"] if n == "virgil.a" else [], "_commit": None, "_source": "ours", "metrics": {}, "tier": "A"}
               for n in nodes]
    model = run(trust, nodes, records, [])
    assert set(model) == set(nodes)
    for n, v in model.items():
        assert n not in [c["id"] for c in v["because"]], f"{n} is its own cause"  # a cycle never vouches for itself
        assert v["verdict"] != "verified", f"{n} verified through a cycle"
