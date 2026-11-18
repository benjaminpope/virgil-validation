"""The roots table: what each root is, why we believe it, what it cannot catch.

Pure functions from two inputs, the trust model (the dict ``scripts/trust.py``
writes to ``docs/assets/trust.json``) and the registry in ``trust/roots.yml``.
They return HTML; nothing here reads a file except ``load_registry``.

* ``render_roots_table(model, registry, names)``: the Trust page's table, one
  row per root class with how many parts rest on it, then the ``links:`` rows,
  with each class's roots in a fold under its row.
* ``render_roots_reference(registry, names)``: the same text with no counts,
  for the Method section (``docs/method/roots.md``, generated).

``names`` is ``ROOT_NAMES`` from ``scripts/trust.py``, the only source of a
root's display label. A root with no entry there (``golden``, the engine's one
root for every golden source) shows under its id. Text is shown as reviewed
only when its registry entry says ``reviewed: true``; anything else carries a
"draft" marker, so a missing key is never read as a review.

Counts come from the model's per-part ``strong`` list, the roots the part's
counted checks pass against (roots of ``via``-blocked checks are not in it).
``Parts`` is the number of parts of virgil with at least one root of the
class in ``strong``. ``Alone`` is the number whose ``strong`` roots all belong
to that class. ``self-consistency`` is the weak root and never counted, so its
row shows "–".
"""

import html
import pathlib
import re

import yaml

PARTS_PREFIX = "virgil."  # parts of virgil; the other nodes are our reference code
NOT_COUNTED = "–"
FIELDS = (("what", "What it is"), ("why", "Why we believe it"), ("cannot_catch", "What it cannot catch"))
DRAFT = ('<span class="vt-draft" title="Draft text, not yet reviewed by the maintainer">draft</span>')


def load_registry(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _inline(text):
    """Escaped text with `code` spans."""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", html.escape(str(text or "").strip(), quote=False))


def _is_draft(entry):
    return entry.get("reviewed") is not True


def _label(root, names):
    return names.get(root, root)


def _class_of(registry):
    """root id -> class id, from ``roots:``."""
    return {r["id"]: r["class"] for r in registry["roots"]}


def class_counts(model, registry):
    """{class id: (parts, alone)} from the model; the weak ``self-consistency``
    class maps to None. Raises ``KeyError`` for a counted root the registry
    does not know, since its row would otherwise silently be wrong."""
    class_of = _class_of(registry)
    counted = {c: [0, 0] for c in registry["classes"]}
    for node in model["nodes"].values():
        if not node["id"].startswith(PARTS_PREFIX):
            continue
        try:
            kinds = {class_of[r] for r in node["strong"]}
        except KeyError as err:
            raise KeyError(f"{node['id']} counts a root that trust/roots.yml does not list: {err}") from None
        for c in kinds:
            counted[c][0] += 1
        if len(kinds) == 1:
            counted[next(iter(kinds))][1] += 1
    return {c: (None if c == "self-consistency" else tuple(v)) for c, v in counted.items()}


def _marker(entry):
    return " " + DRAFT if _is_draft(entry) else ""


def _roots_fold(class_id, registry, names, cols):
    roots = [r for r in registry["roots"] if r["class"] == class_id]
    if not roots:
        return ""
    items = []
    for r in roots:
        note = ""
        if r.get("counts_as"):
            note = f' <span class="vt-muted">(counted as {html.escape(_label(r["counts_as"], names), quote=False)})</span>'
        parts = "".join(f"<dt>{label}</dt><dd>{_inline(r.get(key))}</dd>" for key, label in FIELDS)
        items.append(f'<li><strong>{html.escape(_label(r["id"], names), quote=False)}</strong>{note}{_marker(r)}<dl>{parts}</dl></li>')
    noun = "root" if len(roots) == 1 else "roots"
    return (f'<tr class="vt-roots-detail"><td colspan="{cols}"><details class="vt-fold">'
            f"<summary>{len(roots)} {noun} in this class</summary><ul>{''.join(items)}</ul></details></td></tr>")


def _class_row(class_id, entry, registry, names, counts):
    cells = "".join(f"<td>{_inline(entry.get(key))}</td>" for key, _ in FIELDS)
    if counts is None:
        numbers = ""
    elif counts.get(class_id) is None:
        numbers = f'<td class="n" title="weak root: never counted">{NOT_COUNTED}</td>' * 2
    else:
        parts, alone = counts[class_id]
        numbers = f'<td class="n">{parts}</td><td class="n">{alone}</td>'
    return f'<tr><th scope="row">{html.escape(entry["name"], quote=False)}{_marker(entry)}</th>{cells}{numbers}</tr>'


def _link_row(entry, counts):
    cells = "".join(f"<td>{_inline(entry.get(key))}</td>" for key, _ in FIELDS)
    numbers = "" if counts is None else f'<td class="n">{NOT_COUNTED}</td><td class="n">{NOT_COUNTED}</td>'
    return f'<tr class="vt-link"><th scope="row">{html.escape(entry["name"], quote=False)}{_marker(entry)}</th>{cells}{numbers}</tr>'


def _table(registry, names, counts):
    head = "".join(f'<th scope="col">{label}</th>' for _, label in FIELDS)
    if counts is not None:
        head += '<th scope="col">Parts</th><th scope="col">Alone</th>'
        cols = 6
    else:
        cols = 4
    head = f'<th scope="col">Class</th>{head}'
    body = []
    for class_id, entry in registry["classes"].items():
        body.append(_class_row(class_id, entry, registry, names, counts))
        body.append(_roots_fold(class_id, registry, names, cols))
    for entry in registry.get("links", {}).values():
        body.append(_link_row(entry, counts))
    return (f'<div class="vt-roots-wrap"><table class="vt-roots"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def render_roots_table(model, registry, names):
    """The Trust page's table: a row per class with its counts, then the links."""
    counts = class_counts(model, registry)
    total = sum(1 for n in model["nodes"].values() if n["id"].startswith(PARTS_PREFIX))
    caption = (
        f'<p class="vt-muted"><strong>Parts</strong> counts the parts of virgil (of {total}) with at least one '
        "counted check against a root of that class; <strong>Alone</strong> counts those for which it is the only kind "
        "of root. Our reference code is not a root: it is a link, never counted, and it verifies nothing by itself. "
        "Rows marked <em>draft</em> have text no one has yet reviewed.</p>")
    return _table(registry, names, counts) + caption


def render_roots_reference(registry, names):
    """The same table with no counts, for the Method section."""
    return _table(registry, names, None)
