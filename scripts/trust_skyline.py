"""The Trust page's skyline: every part as one square, standing on the roots it was checked against.

``skyline_html(model, roots)`` turns the model that ``scripts/trust.py`` writes to
``docs/assets/trust.json`` and the parsed ``trust/roots.yml`` into one HTML string: a legend, then
an inline SVG inside a container that scrolls sideways on a small screen. Nothing needs JavaScript.

Reading the figure:

* Columns are the layers of virgil. ``references`` is the reference side: our own independent
  implementations in crosscheck and external_bridge, the adapters, and the packages we compare
  with (pmoired, candid, ...). It is a link and not a root (``links: reference-code`` in
  ``trust/roots.yml``): trusted only as far as its own checks. Its column keeps the layer name
  and stands leftmost behind a dashed line, because the other layers are compared with it; it is
  not a layer above or below. The other layers follow in the order ``LAYER_ORDER`` gives, bottom
  layer first. The layers are not strictly stacked
  (imaging and inference depend on each other), which is why height is depth and not layer.
* Height is ``depth``: the length of the longest chain of ``depends`` beneath a part. Depth 0
  stands directly on the ground line and relies on no other part.
* One square per part, coloured by verdict (the palette of the rest of the page, solid only). It
  links to the function-test page that explains the part, and its ``<title>`` names the part, its
  verdict and its depth.
* Under the ground line, the foundation matrix: for each class of root (``classes:`` in
  ``trust/roots.yml``) and each layer, the number of parts with at least one counted check against
  a root of that class. "Counted" is the model's ``strong`` list: the engine already folds aliases
  (frap into frank) and every golden source into ``golden``. Virgil checked against itself is a weak
  root that never counts, so the matrix leaves it out.

Two corner marks carry what the verdict leaves out: a white dot (virgil has changed this part since its evidence was measured) and a black triangle (a check of it fails on another virgil commit). Colour is the only thing a square's fill says; the verdict word is in its tooltip and the legend
gives each word its icon. Text and lines use ``currentColor`` so the figure follows the site's
light and dark schemes.
"""

import collections
import html

PITCH, CELL = 13, 11  # a square is CELL wide on a PITCH grid
GRID_ROWS = 4  # a layer's cell is about this many squares tall before it widens
MIN_COLS = 3
HEADING_PX = 6.4  # width of a column heading per character at 11.5 px, so a heading never spills over the divider
LEFT, GAP, TOP, RIGHT = 150, 20, 46, 12
BAND_PAD = 4  # blank space under each depth band
MROW = 24  # foundation row height
MAX_R = MROW / 2 - 2  # radius of the largest foundation circle
WEAK_CLASSES = {"self-consistency"}  # a weak root never counts, so it has no row in the matrix


def _e(text):
    return html.escape(str(text), quote=True)


def column_order(nodes, layer_order):
    """Layers left to right: references first, then the rest bottom layer first. Unknown layers go last."""
    present = {n["layer"] for n in nodes.values()}
    known = [layer for layer in reversed(layer_order) if layer in present]
    return known + sorted(present - set(layer_order))


def root_class_of(roots):
    """{root id: class id} for every root in roots.yml."""
    return {r["id"]: r["class"] for r in roots.get("roots", [])}


def foundation_counts(model, roots, alias=None):
    """{class id: {layer: parts in that layer with a counted check against a root of that class}}.

    Counts the model's ``strong`` roots, resolving ``alias`` (trust.ROOT_ALIAS) first. A root the
    registry does not list is counted under the class ``"unlisted"`` so it is never silently lost.
    """
    alias = {} if alias is None else alias
    cls_of = root_class_of(roots)
    out = collections.defaultdict(lambda: collections.defaultdict(int))
    for n in model["nodes"].values():
        classes = {cls_of.get(alias.get(r, r), "unlisted") for r in n.get("strong", [])}
        for c in classes:
            out[c][n["layer"]] += 1
    return {c: dict(v) for c, v in out.items()}


def _depth(n):
    return int(n.get("depth", 0))


def _sort_key(n, order):
    return (order.index(n["verdict"]) if n["verdict"] in order else len(order), n["label"])


def layout(nodes, columns, verdict_order):
    """Where each square goes: {id: (layer, depth, slot)}, plus the grid shape.

    Squares of one layer and depth share a cell of ``cols[layer]`` squares across; a depth band is as
    many rows tall as its fullest cell, so a layer never overlaps its neighbour. Returns
    ``(slots, cols, band_rows, max_depth)``.
    """
    cell = collections.defaultdict(list)
    for n in nodes.values():
        cell[(n["layer"], _depth(n))].append(n)
    max_depth = max((d for _, d in cell), default=0)
    cols = {}
    for layer in columns:
        biggest = max((len(v) for (lay, _), v in cell.items() if lay == layer), default=1)
        cols[layer] = max(MIN_COLS, -(-biggest // GRID_ROWS), -int(-len(layer) * HEADING_PX // PITCH))
    band_rows = [max([1] + [-(-len(cell.get((layer, d), [])) // cols[layer]) for layer in columns]) for d in range(max_depth + 1)]
    slots = {}
    for (layer, d), group in cell.items():
        for i, n in enumerate(sorted(group, key=lambda n: _sort_key(n, verdict_order))):
            slots[n["id"]] = (layer, d, i)
    return slots, cols, band_rows, max_depth


def _style(colours):
    swatches = "".join(f".vt-sky .k-{v}{{fill:{c}}}" for v, c in colours.items())
    return (
        "<style>"
        ".vt-sky-scroll{overflow-x:auto;border:1px solid color-mix(in srgb,currentColor 18%,transparent);border-radius:6px;padding:.4rem}"
        ".vt-sky svg{display:block;margin:0 auto;font-family:inherit}"
        ".vt-sky rect.sq,.vt-sky rect.sw{stroke:currentColor;stroke-opacity:.3;stroke-width:.5}"
        ".vt-sky a:hover rect.sq,.vt-sky a:focus-visible rect.sq{stroke-opacity:1;stroke-width:1.6}"
        ".vt-sky a:focus-visible{outline:none}"
        ".vt-sky .mut{opacity:.68}"
        ".vt-sky .mk{fill:#fff;stroke:#000;stroke-width:.7;pointer-events:none}.vt-sky .mf{fill:#000;stroke:#fff}"
        ".vt-sky-legend{display:flex;flex-wrap:wrap;gap:.3rem 1rem;font-size:.75rem;margin:.3rem 0 .5rem}"
        ".vt-sky-key{display:inline-flex;align-items:center;gap:.35em}"
        ".vt-sky-key svg{display:inline-block;margin:0;width:.9rem;height:.9rem}"
        + swatches + "</style>"
    )


def _legend(counts, order, icons, words, n_marks=(0, 0)):
    """Verdict keys (counts are parts of virgil only), then a key for each corner mark that occurs."""
    items = []
    for v in order:
        if counts.get(v):
            items.append(
                f'<span class="vt-sky-key"><svg viewBox="0 0 {CELL} {CELL}" aria-hidden="true">'
                f'<rect class="sw k-{_e(v)}" width="{CELL}" height="{CELL}" rx="2"/></svg>'
                f'<span aria-hidden="true">{_e(icons.get(v, ""))}</span> {_e(words[v])} ({counts[v]})</span>'
            )
    for n, shape, text in zip(n_marks, ('<circle class="mk" cx="8.5" cy="2.5" r="1.8"/>', '<path class="mk mf" d="M.5,10.5h5l-5,-5z"/>'),
                              ("virgil has changed it since its evidence", "fails on another virgil commit")):
        if n:
            items.append(f'<span class="vt-sky-key"><svg viewBox="0 0 {CELL} {CELL}" aria-hidden="true"><rect class="sw k-unchecked" '
                         f'width="{CELL}" height="{CELL}" rx="2"/>{shape}</svg> corner mark: {_e(text)} ({n})</span>')
    return '<div class="vt-sky-legend">' + "".join(items) + "</div>"


def skyline_html(model, roots, base="", *, names):
    """The skyline and foundation as one HTML string (no blank lines, so Markdown keeps it as one block).

    ``model`` is the dict written to ``docs/assets/trust.json``; ``roots`` the parsed
    ``trust/roots.yml``; ``base`` the relative prefix the page's function-test links need. ``names``
    is the object that holds ``COLOURS``, ``ICON``, ``ORDER``, ``VERDICTS``, ``LAYER_ORDER``,
    ``ROOT_ALIAS`` and ``doc_url`` (``trust_render`` passes itself; tests pass ``trust``).
    """
    t = names
    nodes = model["nodes"]
    if not nodes:
        return '<div class="vt-sky"><p class="vt-muted">No parts to draw.</p></div>'
    docs = model.get("docs", {})
    columns = column_order(nodes, t.LAYER_ORDER)
    slots, cols, band_rows, max_depth = layout(nodes, columns, t.ORDER)

    # x of each column, y of each depth band (depth 0 lowest)
    col_x, x = {}, LEFT
    for layer in columns:
        col_x[layer] = x
        x += cols[layer] * PITCH + GAP
    width = x - GAP + RIGHT
    sky_h = sum(r * PITCH + BAND_PAD for r in band_rows)
    band_bottom, y = [], TOP + sky_h
    for d in range(max_depth + 1):
        band_bottom.append(y)
        y -= band_rows[d] * PITCH + BAND_PAD
    ground = TOP + sky_h + 4

    classes = {c: v for c, v in roots.get("classes", {}).items() if c not in WEAK_CLASSES}
    found = foundation_counts(model, roots, t.ROOT_ALIAS)
    rows = list(classes) + (["unlisted"] if "unlisted" in found else [])
    names = {c: v.get("name", c) for c, v in classes.items()} | {"unlisted": "Root not in the registry"}
    height = ground + 26 + len(rows) * MROW + 8

    out = []
    # depth axis
    for d in range(max_depth + 1):
        mid = band_bottom[d] - BAND_PAD - band_rows[d] * PITCH / 2 + 3
        out.append(f'<text x="{LEFT - 14}" y="{mid:.1f}" text-anchor="end" font-size="10" class="mut" fill="currentColor">{d}</text>')
        if d:
            ly = band_bottom[d] - BAND_PAD / 2
            out.append(f'<line x1="{LEFT - 8}" x2="{width - RIGHT}" y1="{ly:.1f}" y2="{ly:.1f}" stroke="currentColor" stroke-opacity=".12" stroke-width=".6"/>')
    out.append(
        f'<text x="14" y="{TOP + sky_h / 2:.1f}" font-size="10.5" class="mut" fill="currentColor" text-anchor="middle" '
        f'transform="rotate(-90 14 {TOP + sky_h / 2:.1f})">depth</text>'
    )
    # column headings and squares
    verdict_words = {v: t.VERDICTS[v][0] for v in t.VERDICTS}
    for layer in columns:
        members = [n for n in nodes.values() if n["layer"] == layer]
        ok = sum(1 for n in members if n["verdict"] == "verified")
        cx = col_x[layer] + cols[layer] * PITCH / 2
        out.append(f'<text x="{cx:.1f}" y="17" text-anchor="middle" font-size="11.5" font-weight="500" fill="currentColor">{_e(layer)}</text>')
        out.append(f'<text x="{cx:.1f}" y="31" text-anchor="middle" font-size="10" class="mut" fill="currentColor">{ok}/{len(members)}</text>')
    out.append(f'<text x="{LEFT - 14}" y="31" text-anchor="end" font-size="10" class="mut" fill="currentColor">verified / parts</text>')
    for n in sorted(nodes.values(), key=lambda n: slots[n["id"]]):
        layer, d, i = slots[n["id"]]
        sx = col_x[layer] + (i % cols[layer]) * PITCH
        sy = band_bottom[d] - BAND_PAD - (i // cols[layer] + 1) * PITCH
        word = verdict_words.get(n["verdict"], n["verdict"])
        title = f'{n["label"]} · {word} · depth {_depth(n)}'
        marks = ""
        if n.get("changed"):
            title += " · virgil has changed this since its evidence was measured"
            marks += f'<circle class="mk" cx="{sx + CELL - 2.5}" cy="{sy + 2.5:.1f}" r="1.8"/>'
        if n.get("fails_elsewhere"):
            title += " · fails on another virgil commit"
            marks += f'<path class="mk mf" d="M{sx + .5},{sy + CELL - .5:.1f}h5l-5,-5z"/>'
        sq = (f'<rect class="sq k-{_e(n["verdict"])}" x="{sx}" y="{sy:.0f}" width="{CELL}" height="{CELL}" rx="2">'
              f'<title>{_e(title)}</title></rect>{marks}')
        doc = docs.get(n["id"])
        # data-node lets docs/assets/trust.js select the part's cone; a square with no page is a bare <g> that the
        # script makes focusable, so without JavaScript it is only a coloured square with a tooltip
        if doc:
            sq = f'<a href="{_e(base + t.doc_url(doc))}" aria-label="{_e(title)}" data-node="{_e(n["id"])}">{sq}</a>'
        else:
            sq = f'<g data-node="{_e(n["id"])}" aria-label="{_e(title)}">{sq}</g>'
        out.append(sq)
    # ground line and the dashed divider after the references column
    out.append(f'<rect x="{LEFT - 8}" y="{ground}" width="{width - LEFT - RIGHT + 8}" height="3" fill="currentColor" fill-opacity=".75"/>')
    out.append(f'<text x="{LEFT - 14}" y="{ground + 5}" text-anchor="end" font-size="10.5" font-weight="500" fill="currentColor">roots of trust</text>')
    if columns and columns[0] == "references" and len(columns) > 1:
        dx = col_x["references"] + cols["references"] * PITCH + GAP / 2
        out.append(f'<line x1="{dx:.1f}" x2="{dx:.1f}" y1="6" y2="{height - 6}" stroke="currentColor" stroke-opacity=".3" stroke-dasharray="3 3"/>')
    # foundation matrix
    biggest = max([1] + [n for per in found.values() for n in per.values()])
    for k, c in enumerate(rows):
        yy = ground + 26 + k * MROW
        out.append(f'<text x="{LEFT - 14}" y="{yy + 4}" text-anchor="end" font-size="11" fill="currentColor">{_e(names[c])}</text>')
        for layer in columns:
            count = found.get(c, {}).get(layer, 0)
            if not count:
                continue
            cx = col_x[layer] + cols[layer] * PITCH / 2
            r = max(3.0, MAX_R * (count / biggest) ** 0.5)  # area follows the count, the largest filling MAX_R
            tip = f'{count} {layer} part{"s" if count != 1 else ""} with a counted check against {names[c].lower()}'
            out.append(f'<g><title>{_e(tip)}</title><circle cx="{cx - 12:.1f}" cy="{yy}" r="{r:.1f}" fill="currentColor" fill-opacity=".55"/>'
                       f'<text x="{cx - 12 + r + 4:.1f}" y="{yy + 4}" font-size="10.5" class="mut" fill="currentColor">{count}</text></g>')

    # the legend counts parts of virgil only, as the headline and the bar at the top of the page do
    ours = [n for n in nodes.values() if n["id"].startswith("virgil.")]
    verdict_counts = collections.Counter(n["verdict"] for n in ours)
    n_marks = (sum(1 for n in ours if n.get("changed")), sum(1 for n in ours if n.get("fails_elsewhere")))
    desc = (f"{len(ours)} parts of virgil and {len(nodes) - len(ours)} reference nodes, one square each, in columns by layer "
            "and stacked by dependency depth, standing on a matrix of the classes of root each layer was checked against.")
    svg = (
        f'<svg viewBox="0 0 {width} {height:.0f}" width="{width}" height="{height:.0f}" role="group" '
        f'aria-labelledby="vt-sky-t vt-sky-d" style="min-width:{int(width * 0.75)}px;max-width:{width}px;width:100%;height:auto">'
        f'<title id="vt-sky-t">Skyline of every part</title><desc id="vt-sky-d">{_e(desc)}</desc>'
        + "".join(out) + "</svg>"
    )
    return (
        '<div class="vt-sky">' + _style(t.COLOURS)
        + _legend(verdict_counts, t.ORDER, t.ICON, verdict_words, n_marks)
        + '<div class="vt-sky-scroll" role="region" aria-label="Skyline of every part; scrolls sideways on a small screen" tabindex="0">'
        + svg + "</div>"
        '<p class="vt-muted">One square per part, linked to the page that checks it; the counts above are parts of virgil, and the references column is counted in the headline. Columns are layers; height is dependency depth, '
        "the longest chain of parts beneath it, so depth 0 relies on no other part. Under the ground line, the number of parts in each "
        "layer with at least one counted check against that class of root.</p></div>"
    )
