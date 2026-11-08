"""The validates markers of a test tree, and what changed between two trees.

    python scripts/marker_inventory.py BASE_ROOT HEAD_ROOT

Prints a Markdown table of tests added or removed (with their kind) and of
tests whose roots, kind, tier, objects or headline changed. Markers are read
from the AST, as tests/test_trust_graph.py does, so nothing is imported. It
always exits 0: a pull request's job summary shows the diff, it never fails.
"""

import ast
import pathlib
import sys


def _literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return "<computed>"


def _claims(node):
    """The validates calls under node, each as a dict."""
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr == "validates":
            kw = {k.arg: _literal(k.value) for k in sub.keywords if k.arg}
            roots = kw.get("roots", [])
            out.append({
                "objects": sorted(str(_literal(a)) for a in sub.args),
                "roots": sorted(map(str, roots)) if isinstance(roots, (list, tuple)) else [str(roots)],
                "kind": kw.get("kind", "check"),
                "tier": kw.get("tier", "A"),
                "headline": kw.get("headline"),
            })
    return out


def inventory(root):
    """{test id: [claim, ...]} for every test under ROOT/tests carrying a validates marker."""
    inv = {}
    for path in sorted((pathlib.Path(root) / "tests").glob("test_*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        module_level = []
        for node in tree.body:
            funcs = [(node.name, node)] if isinstance(node, ast.FunctionDef) else []
            if isinstance(node, ast.ClassDef):
                funcs = [(f"{node.name}::{n.name}", n) for n in node.body if isinstance(n, ast.FunctionDef)]
            if not funcs:
                if isinstance(node, ast.Assign):
                    module_level += _claims(node)
                continue
            for name, fn in funcs:
                claims = _claims(fn)
                if name.split("::")[-1].startswith("test_") and claims:
                    inv[f"tests/{path.name}::{name}"] = claims
        if module_level:
            inv[f"tests/{path.name}::<module pytestmark>"] = module_level
    return inv


def _key(claims):
    return sorted((c["kind"], c["tier"], str(c["roots"]), str(c["objects"]), str(c["headline"])) for c in claims)


def _show(claims):
    return "; ".join(
        f"{c['kind']}/{c['tier']} roots={','.join(c['roots'])} objects={','.join(c['objects'])}"
        + (f" headline={c['headline']}" if c["headline"] else "")
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
