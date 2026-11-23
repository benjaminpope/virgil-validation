"""The cone of a part or pipeline on the Trust page: everything beneath it, down to the ground.

``cone(model, id)`` and ``sentence(model, id)`` are the one definition of both. The Trust page embeds
what they return (``embed``) and draws it in ``docs/assets/trust.js``; the paper figure will call the
same functions, so the page and the paper cannot disagree.

``model`` is the dict that ``scripts/trust.py`` writes to ``docs/assets/trust.json``. A part is a key
of ``model["nodes"]``; a pipeline is an entry of ``model["pipelines"]`` (its ``id`` and its ``steps``).
When a part and a pipeline share an id, the part wins. An id that is neither raises ``KeyError``.

The cone of a part is the part itself and everything it relies on, directly or through other parts
(its ``depends``, followed transitively; a cycle ends the walk, it does not loop). The cone of a
pipeline is the union of the cones of its ``steps``. A name in ``depends`` or ``steps`` that is not a
part of the model is left out.

The sentence says only what the model holds: which verdict the part has, which roots its own checks
agree with, how many parts it relies on and how many of those are not yet verified, and how many
parts its cone reaches the ground in. It never grades the roots (one kind or several): that is the
job of the grade column, which is added to the model separately.

The lens draws a cone on the skyline (the figure of ``trust_skyline``) and describes it in a panel. The functions
that decide what it shows are here: ``blocks`` (how many not-verified parts each part holds back),
``critical_path`` (the shortest chain to the worst not-verified part), ``longest_chain`` (the chain that
reaches the ground), ``fix_list`` (what would make the selection verified) and ``verified_beneath`` (what is fine
as it is). ``embed`` carries all of it to the page, which only draws.
"""

import collections

# One word or short phrase per verdict, for the parts a cone lists as not yet verified.
SHORT = {
    "verified": "verified", "relies": "relies on a known bug", "relies-unverified": "relies on an unverified part",
    "bug": "known bug", "partly": "partly checked", "convention": "convention from virgil's docs",
    "failing": "check failing", "unchecked": "not yet checked",
}
OPENING = {
    "verified": "{n} is verified.",
    "relies": "{n} works, but relies on a part with a known bug.",
    "relies-unverified": "{n} works, but relies on a part that is not yet verified.",
    "bug": "{n} has a known bug, fix pending{f}.",
    "partly": "{n} is only partly checked: it needs another independent check.",
    "convention": "{n}'s checks agree, but a convention they rest on comes from virgil's own documentation only.",
    "failing": "A check of {n} fails.",
    "unchecked": "{n} has no independent check yet.",
}
PIPELINE_STATE = {
    "validated": "Its end-to-end check passes and every step is verified.",
    "passes": "Its end-to-end check passes, but not every step is verified yet.",
    "failing": "Its end-to-end check fails.",
    "planned": "It is planned: there is no result yet.",
    "running": "It is running: there is no result yet.",
    "not run": "Its checks exist but have not run.",
}
LISTED = 3  # parts named in "of which K are not yet verified (...)"

# Root letters on the cone's boxes, by class of root (the ``class`` of each root in trust/roots.yml).
CLASS_LETTER = {"mathematics": "M", "statistics": "Σ", "standards": "§", "published-results": "L",
                "other-peoples-code": "C", "golden-values": "G"}
LETTER_ORDER = "MΣ§LCG"
LETTER_NAMES = {"M": "mathematics", "Σ": "statistics", "§": "standards", "L": "published result",
                "C": "other people's code", "G": "golden values"}


def _pipeline_of(model, id):
    if id in model["nodes"]:
        return None
    for p in model.get("pipelines", []):
        if p["id"] == id:
            return p
    raise KeyError(id)


def cone(model, id):
    """The cone of a part or pipeline.

    Returns ``{"id", "kind", "top", "nodes", "edges"}``: ``kind`` is ``"node"`` or ``"pipeline"``;
    ``top`` the parts the cone hangs from (the part itself, or a pipeline's steps in order);
    ``nodes`` maps each part in the cone to its depth (the model's ``depth``, the longest chain
    beneath it); ``edges`` is a sorted list of ``(part, part it relies on)`` pairs inside the cone.
    """
    nodes = model["nodes"]
    pipe = _pipeline_of(model, id)
    top = [s for s in pipe["steps"] if s in nodes] if pipe else [id]
    seen, stack = set(), list(reversed(top))
    while stack:
        n = stack.pop()
        if n in seen or n not in nodes:
            continue
        seen.add(n)
        stack.extend(reversed(nodes[n].get("depends", [])))
    edges = sorted((a, b) for a in seen for b in nodes[a].get("depends", []) if b in seen)
    return {"id": id, "kind": "pipeline" if pipe else "node", "top": top,
            "nodes": {n: int(nodes[n].get("depth", 0)) for n in sorted(seen)}, "edges": edges}


def _plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def _list(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _name(node):
    return node.get("label") or node["id"]


def _severity(verdict):
    """0 failing, 1 known bug, 2 a cause of its own (not checked, partly, convention), 3 merely relies on another."""
    return {"failing": 0, "bug": 1}.get(verdict, 3 if verdict.startswith("relies") else 2)


def _unverified(model, c, exclude=()):
    """The parts of the cone that are not verified, the most serious first (a failing check, a known bug, then a part
    with a cause of its own, then those that merely rely on another), each closest to the ground first."""
    nodes = model["nodes"]
    bad = [n for n in c["nodes"] if nodes[n]["verdict"] != "verified" and n not in exclude]
    return sorted(bad, key=lambda n: (_severity(nodes[n]["verdict"]), c["nodes"][n], _name(nodes[n])))


def _extras(model, c, ids):
    """Parts named in ``ids`` (a part's ``because``, a pipeline's ``unverified``) that are not in the cone and are not
    verified: reference code its checks go through (``via``), which the cone's ``depends`` walk does not reach."""
    nodes = model["nodes"]
    out = [i for i in dict.fromkeys(ids) if i not in c["nodes"] and i in nodes and nodes[i]["verdict"] != "verified"]
    return sorted(out, key=lambda n: (_severity(nodes[n]["verdict"]), _name(nodes[n])))


def _named(model, parts):
    """"label: reason" for the first LISTED parts, then "; and M more"."""
    nodes = model["nodes"]
    named = []
    for n in parts[:LISTED]:
        why = SHORT.get(nodes[n]["verdict"], nodes[n]["verdict"])
        if nodes[n].get("findings"):
            why += ", " + " ".join(nodes[n]["findings"])
        named.append(f"{_name(nodes[n])}: {why}")
    more = len(parts) - len(named)
    return "; ".join(named) + (f"; and {more} more" if more else "")


def _reliance(model, c, exclude=()):
    """The tail of "It relies on N parts, ...": "all verified", or "of which K are not yet verified (...)"."""
    bad = _unverified(model, c, exclude)
    if not bad:
        return "all verified"
    return f"of which {len(bad)} {'is' if len(bad) == 1 else 'are'} not yet verified ({_named(model, bad)})"


def _via(model, extras):
    n = len(extras)
    return (f"Its checks also go through {n} reference {'part' if n == 1 else 'parts'} that "
            f"{'is' if n == 1 else 'are'} not yet verified ({_named(model, extras)}).")


def _ground(model, c):
    """How many parts of the cone stand on the ground (rely on no other part in it; a self-loop is not a reliance)."""
    relied = {a for a, b in c["edges"] if a != b}
    n = sum(1 for p in c["nodes"] if p not in relied)
    if not n:
        return "Its cone has no ground part: the parts in it rely on one another in a loop."
    return f"Its cone reaches the ground in {_plural(n, 'part')}."


# Words for roots that the caller's ``root_names`` does not cover (``golden`` has no entry in trust.ROOT_NAMES).
DEFAULT_ROOT_NAMES = {"golden": "golden values"}


def _root_word(r, names):
    if r in names:
        return names[r]
    if r in DEFAULT_ROOT_NAMES:
        return DEFAULT_ROOT_NAMES[r]
    return "reference code: " + r[len("golden:"):] if r.startswith("golden:") else r


def sentence(model, id, root_names=None):
    """The generated sentence for a part or pipeline; ``root_names`` maps a root id to its words."""
    c = cone(model, id)
    names = root_names or {}
    nodes = model["nodes"]
    if c["kind"] == "pipeline":
        p = _pipeline_of(model, id)
        out = [f"The pipeline “{p['title']}” runs {_plural(len(c['top']), 'step')} end to end"
               + (f" on {p['data']} data." if p.get("data") else "."),
               PIPELINE_STATE.get(p["status"], f"Its state is {p['status']}.")]
        extras = _extras(model, c, p.get("unverified", []))
        if c["nodes"]:
            out.append(f"Its steps rest on {_plural(len(c['nodes']), 'part')} in all, " + _reliance(model, c) + ".")
        if extras:
            out.append(_via(model, extras))
        if c["nodes"]:
            out.append(_ground(model, c))
        return " ".join(out)
    n = nodes[id]
    word = _name(n)
    opening = OPENING.get(n["verdict"], "{n}: " + str(n["verdict"]) + ".")
    out = [opening.format(n=word, f=(" (" + ", ".join(n["findings"]) + ")") if n.get("findings") else "")]
    strong = [_root_word(r, names) for r in n.get("strong", [])]
    if strong:
        out.append(f"Its own checks agree with {_list(strong)}.")
    elif n["verdict"] != "unchecked":
        out.append("None of its checks counts yet as agreement with a root.")
    extras = _extras(model, c, [b["id"] for b in n.get("because", [])])
    if len(c["nodes"]) == 1:
        out.append("It relies on no other part, so it stands directly on the roots."
                   if n["verdict"] == "verified" and not extras else "It relies on no other part.")
    else:
        out.append(f"It relies on {_plural(len(c['nodes']) - 1, 'part')}, "
                   + _reliance(model, c, exclude=(id,)) + ".")
    if extras:
        out.append(_via(model, extras))
    if len(c["nodes"]) > 1:
        out.append(_ground(model, c))
    return " ".join(out)


# ---- the lens: what the figure draws on the skyline and what the panel says about it

def _dependents(model):
    """{part: parts that list it in ``depends``}, for parts of the model only."""
    out = {n: [] for n in model["nodes"]}
    for a, node in model["nodes"].items():
        for b in node.get("depends", []):
            if b in out and b != a:
                out[b].append(a)
    return out


def blocks(model):
    """{part: how many parts in the whole graph, not themselves verified, have it in their cone}.

    The number answers "if this were fixed, how many parts would stop being held back": it counts every part that
    relies on it, directly or through others, whose own verdict is not ``verified``. A cycle ends the walk and a
    part never blocks itself.
    """
    nodes, dependents = model["nodes"], _dependents(model)
    out = {}
    for n in nodes:
        seen, stack = set(), list(dependents[n])
        while stack:
            m = stack.pop()
            if m in seen or m == n:
                continue
            seen.add(m)
            stack.extend(dependents[m])
        out[n] = sum(1 for m in seen if nodes[m]["verdict"] != "verified")
    return out


def _worst(model, c, blk):
    """The worst not-verified part in the cone, never the selection itself: the most serious verdict, then the
    one that blocks most parts, then the smaller id. ``None`` when every part beneath is verified."""
    nodes = model["nodes"]
    bad = [n for n in c["nodes"] if n not in c["top"] and nodes[n]["verdict"] != "verified"]
    return min(bad, key=lambda n: (_severity(nodes[n]["verdict"]), -blk[n], n)) if bad else None


def critical_path(model, id, blk=None):
    """The shortest chain of ``depends`` from the selection to its worst not-verified part: a list of ids, the
    selection (the closest of a pipeline's steps) first, the worst part last. Empty when the cone holds no
    not-verified part beneath the selection (a verified cone has no critical path)."""
    c = cone(model, id)
    blk = blocks(model) if blk is None else blk
    target = _worst(model, c, blk)
    if target is None:
        return []
    nodes, best = model["nodes"], None
    for start in c["top"]:
        prev, queue = {start: None}, [start]
        for x in queue:  # breadth first: the first time target is reached is by a shortest chain
            if x == target:
                break
            for y in sorted(nodes[x].get("depends", [])):
                if y in c["nodes"] and y not in prev:
                    prev[y] = x
                    queue.append(y)
        if target in prev:
            path, x = [], target
            while x is not None:
                path.append(x)
                x = prev[x]
            path.reverse()
            if best is None or len(path) < len(best):
                best = path
    return best or []


def longest_chain(model, id):
    """The longest chain of ``depends`` from the selection down to the ground (a depth-0 part): a list of ids.
    From the deepest of a pipeline's steps, each part is followed to the dependency of greatest depth (the smaller
    id on a tie), so the chain is as long as the depth says. A cycle ends the walk."""
    c = cone(model, id)
    nodes = model["nodes"]
    if not c["top"]:
        return []
    x = min(c["top"], key=lambda n: (-c["nodes"][n], n))
    chain, seen = [x], {x}
    while True:
        nxt = [y for y in nodes[x].get("depends", []) if y in c["nodes"] and y not in seen]
        if not nxt:
            return chain
        x = min(nxt, key=lambda n: (-c["nodes"][n], n))
        chain.append(x)
        seen.add(x)


def fix_list(model, id, blk=None):
    """What would make the selection verified: the not-verified parts in its cone, and the reference parts its
    checks go through (``via``), as ``[(id, note)]`` ordered by how many parts they block, then verdict, depth, id.
    ``note`` is ``""``, ``"this part"`` (the selection itself) or ``"via"`` (reference code outside the cone).
    A selection whose only problem is relying on others is left out of its own list, and a verified cone
    has an empty list."""
    c = cone(model, id)
    nodes = model["nodes"]
    blk = blocks(model) if blk is None else blk
    pipe = c["kind"] == "pipeline"
    because = (_pipeline_of(model, id).get("unverified", []) if pipe
               else [b["id"] for b in nodes[id].get("because", [])])
    via = _extras(model, c, because)
    items = []
    for n in c["nodes"]:
        if nodes[n]["verdict"] == "verified":
            continue
        if not pipe and n == id and nodes[n]["verdict"].startswith("relies"):
            continue
        items.append((n, "this part" if (not pipe and n == id) else ""))
    items += [(n, "via") for n in via]
    items.sort(key=lambda t: (-blk[t[0]], _severity(nodes[t[0]]["verdict"]), c["nodes"].get(t[0], 0), t[0]))
    return items


def verified_beneath(model, id):
    """How many verified parts the cone holds besides the selection, split by layer: ``(total, [(layer, n)])``
    with the biggest layers first."""
    c = cone(model, id)
    nodes = model["nodes"]
    count = collections.Counter(nodes[n]["layer"] for n in c["nodes"] if nodes[n]["verdict"] == "verified"
                                and not (c["kind"] == "node" and n == id))
    return sum(count.values()), sorted(count.items(), key=lambda t: (-t[1], t[0]))


def root_letters(roots, alias=None):
    """{root id: letter} from the parsed ``trust/roots.yml``; a root with no counted class gets no letter."""
    out = {r["id"]: CLASS_LETTER[r["class"]] for r in roots.get("roots", []) if r.get("class") in CLASS_LETTER}
    for a, target in (alias or {}).items():
        if target in out:
            out.setdefault(a, out[target])
    return out


def root_classes(roots, alias=None):
    """{root id: class id} for every root of ``trust/roots.yml``, with each alias resolved to its target's class."""
    out = {r["id"]: r["class"] for r in roots.get("roots", [])}
    for a, target in (alias or {}).items():
        if target in out:
            out.setdefault(a, out[target])
    return out


def embed(model, roots, root_names=None, alias=None):
    """What the page carries for the lens, as plain JSON-able data.

    ``order`` lists every part id once; ``cones`` maps each part and pipeline id to the indices into ``order`` of its
    cone (a part's own index first); ``sentences`` the sentence of each; ``lens`` the figure's and the panel's paths
    and lists for each id, as indices: ``crit`` (critical path), ``chain`` (longest chain), ``fix`` (``[index, note]``
    with note 0 none, 1 this part, 2 via), ``ver`` (verified parts beneath: ``[total, [[layer, n], ...]]``);
    ``blocks`` the blocks count of each part, parallel to ``order``; ``root_class`` the class of each root;
    ``pipelines`` the little the panel needs of each pipeline.
    """
    order = sorted(model["nodes"])
    index = {n: i for i, n in enumerate(order)}
    ids = order + [p["id"] for p in model.get("pipelines", []) if p["id"] not in model["nodes"]]
    blk = blocks(model)
    note_code = {"": 0, "this part": 1, "via": 2}
    cones, lens = {}, {}
    for i in ids:
        c = cone(model, i)
        cones[i] = [index[n] for n in c["top"]] + [index[n] for n in c["nodes"] if n not in c["top"]]
        total, layers = verified_beneath(model, i)
        lens[i] = {"crit": [index[n] for n in critical_path(model, i, blk)],
                   "chain": [index[n] for n in longest_chain(model, i)],
                   "fix": [[index[n], note_code[note]] for n, note in fix_list(model, i, blk)],
                   "ver": [total, [list(t) for t in layers]]}
    return {
        "order": order, "cones": cones, "lens": lens, "blocks": [blk[n] for n in order],
        "sentences": {i: sentence(model, i, root_names) for i in ids},
        "root_class": root_classes(roots, alias), "short": SHORT,
        "pipelines": {p["id"]: {k: p.get(k) for k in ("title", "status", "steps", "data", "page")}
                      for p in model.get("pipelines", []) if p["id"] not in model["nodes"]},
    }
