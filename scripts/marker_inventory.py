"""The validates markers of a test tree, and what changed between two trees.

    python scripts/marker_inventory.py BASE_ROOT HEAD_ROOT

Prints a Markdown table of tests added or removed (with their kind) and of
tests whose roots, kind, tier, objects, headline, property or via changed.
Markers are read from the AST, as tests/test_trust_graph.py does, so nothing
is imported. It always exits 0: a pull request's job summary shows the diff,
it never fails.
"""

import ast
import pathlib
import sys


def _literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return "<computed>"


def _listed(value):
    return sorted(map(str, value)) if isinstance(value, (list, tuple)) else [str(value)]


def _claim(call):
    """One validates call as a dict, with every kwarg that changes the claim's
    meaning (the ones src/evidence/plugin.py reads)."""
    kw = {k.arg: _literal(k.value) for k in call.keywords if k.arg}
    return {
        "objects": sorted(str(_literal(a)) for a in call.args),
        "roots": _listed(kw.get("roots", [])),
        "kind": kw.get("kind", "check"),
        "tier": kw.get("tier", "A"),
        "headline": kw.get("headline"),
        "property": _listed(kw.get("property", [])),
        "via": _listed(kw.get("via", [])),
    }


def _is_validates(node):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "validates"


def _decorator_claims(node):
    """The validates decorators of a function or class. Only the decorator
    list is read, never the body (``schema.validates(data)`` there is not a
    marker)."""
    return [_claim(d) for d in node.decorator_list if _is_validates(d)]


def _pytestmark_claims(body):
    """The validates marks of a ``pytestmark = ...`` in a module or class body."""
    out = []
    for stmt in body:
        if isinstance(stmt, ast.Assign):
            targets, value = stmt.targets, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets, value = [stmt.target], stmt.value
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in targets):
            continue
        items = value.elts if isinstance(value, (ast.List, ast.Tuple)) else [value]
        out += [_claim(i) for i in items if _is_validates(i)]
    return out


_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)


def inventory(root):
    """{test id: [claim, ...]} for every test under ROOT/tests carrying a validates marker.

    As in pytest, and so in src/evidence/plugin.py (one claim per marker from
    item.iter_markers), the marks of a test are those of the function, its
    class and the module: all apply, one claim per marker.
    """
    inv = {}
    for path in sorted((pathlib.Path(root) / "tests").glob("test_*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        module_claims = _pytestmark_claims(tree.body)
        for node in tree.body:
            if isinstance(node, _FUNCS):
                tests = [(node.name, node, [])]
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                class_claims = _decorator_claims(node) + _pytestmark_claims(node.body)
                tests = [(f"{node.name}::{n.name}", n, class_claims) for n in node.body if isinstance(n, _FUNCS)]
            else:
                continue
            for name, fn, outer in tests:
                if not name.split("::")[-1].startswith("test_"):
                    continue
                claims = _decorator_claims(fn) + outer + module_claims
                if claims:
                    inv[f"tests/{path.name}::{name}"] = claims
    return inv


def _key(claims):
    return sorted(
        (c["kind"], c["tier"], str(c["roots"]), str(c["objects"]), str(c["headline"]), str(c["property"]), str(c["via"]))
        for c in claims
    )


def _show(claims):
    return "; ".join(
        f"{c['kind']}/{c['tier']} roots={','.join(c['roots'])} objects={','.join(c['objects'])}"
        + (f" headline={c['headline']}" if c["headline"] else "")
        + (f" property={','.join(c['property'])}" if c["property"] else "")
        + (f" via={','.join(c['via'])}" if c["via"] else "")
        for c in claims
    )


def diff(base, head):
    rows = []
    for t in sorted(set(base) | set(head)):
        a, b = base.get(t), head.get(t)
        if a is None:
            rows.append(f"| `{t}` | added | {_show(b)} |")
        elif b is None:
            rows.append(f"| `{t}` | removed | {_show(a)} |")
        elif _key(a) != _key(b):
            rows.append(f"| `{t}` | changed | {_show(a)} -> {_show(b)} |")
    table = "| Test | Change | Marker |\n| --- | --- | --- |"
    return table + "\n" + ("\n".join(rows) if rows else "| (no marker changes) | | |")


if __name__ == "__main__":
    print(diff(inventory(sys.argv[1]), inventory(sys.argv[2])))
