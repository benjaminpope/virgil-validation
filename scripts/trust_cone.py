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
"""

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
    n = sum(1 for p in c["nodes"] if not model["nodes"][p].get("depends"))
    return f"Its cone reaches the ground in {_plural(n, 'part')}."


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
    strong = [names.get(r, r) for r in n.get("strong", [])]
    if strong:
        out.append(f"Its own checks agree with {_list(strong)}.")
    elif n["verdict"] != "unchecked":
        out.append("None of its checks counts yet as agreement with a root.")
    extras = _extras(model, c, [b["id"] for b in n.get("because", [])])
    if len(c["nodes"]) == 1:
        out.append("It relies on no other part."
                   if extras else "It relies on no other part, so it stands directly on the roots.")
    else:
        out.append(f"It relies on {_plural(len(c['nodes']) - 1, 'part')}, "
                   + _reliance(model, c, exclude=(id,)) + ".")
    if extras:
        out.append(_via(model, extras))
    if len(c["nodes"]) > 1:
        out.append(_ground(model, c))
    return " ".join(out)


def root_letters(roots, alias=None):
    """{root id: letter} from the parsed ``trust/roots.yml``; a root with no counted class gets no letter."""
    out = {r["id"]: CLASS_LETTER[r["class"]] for r in roots.get("roots", []) if r.get("class") in CLASS_LETTER}
    for a, target in (alias or {}).items():
        if target in out:
            out.setdefault(a, out[target])
    return out


def embed(model, roots, root_names=None, alias=None):
    """What the page carries for the cone panel, as plain JSON-able data.

    ``order`` lists every part id once; ``cones`` maps each part and pipeline id to the indices into
    ``order`` of its cone (a part's own index first); ``sentences`` the sentence of each; ``letters``
    the root letter of each root; ``pipelines`` the little the panel needs of each pipeline.
    """
    order = sorted(model["nodes"])
    index = {n: i for i, n in enumerate(order)}
    ids = order + [p["id"] for p in model.get("pipelines", []) if p["id"] not in model["nodes"]]
    cones = {}
    for i in ids:
        c = cone(model, i)
        cones[i] = [index[n] for n in c["top"]] + [index[n] for n in c["nodes"] if n not in c["top"]]
    return {
        "order": order, "cones": cones,
        "sentences": {i: sentence(model, i, root_names) for i in ids},
        "letters": root_letters(roots, alias), "letter_names": LETTER_NAMES, "letter_order": LETTER_ORDER, "short": SHORT,
        "pipelines": {p["id"]: {k: p.get(k) for k in ("title", "status", "steps", "data", "page")}
                      for p in model.get("pipelines", [])},
    }
