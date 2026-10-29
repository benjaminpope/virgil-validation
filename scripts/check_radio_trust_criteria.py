"""Gate: the radio trust nodes of trust/graph.yml are all present and ask for at least the roots design/radio_trust_criteria.yml registers.

It pins the node set and each node's ``roots`` and ``properties`` as written in the graph. Every radio node registers the default of one root,
so in practice it guards the node list and the ``conjugation`` property; the status rules themselves are scripts/trust.py's.

The criteria file is pinned in design/criteria_registry.json (``crosscheck.preregistration``); the script exits
non-zero if it is not the pinned file, or if a radio node in the graph needs fewer roots or properties than registered
(or is missing). ``--allow-unregistered`` is for development runs only and stamps the output.

    python scripts/check_radio_trust_criteria.py [--graph trust/graph.yml] [--out FILE] [--allow-unregistered]
"""

import argparse
import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import crosscheck.preregistration as prereg  # noqa: E402

CRITERIA_FILE = ROOT / "design" / "radio_trust_criteria.yml"


def problems(criteria, graph_nodes):
    """Every way a node of the graph asks for less than the registered criteria."""
    out = []
    for name, want in criteria["nodes"].items():
        node = graph_nodes.get(name)
        if node is None:
            out.append(f"{name}: not in the graph")
            continue
        if node.get("roots", 1) < want["roots"]:
            out.append(f"{name}: needs {node.get('roots', 1)} roots, registered {want['roots']}")
        have = node.get("properties", {})
        for prop, n in want.get("properties", {}).items():
            if have.get(prop, 0) < n:
                out.append(f"{name}: property {prop} needs {have.get(prop, 0)} roots, registered {n}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--graph", default=str(ROOT / "trust" / "graph.yml"))
    ap.add_argument("--out", default=None)
    prereg.add_argument(ap)
    args = ap.parse_args(argv)
    reg = prereg.for_args(args, CRITERIA_FILE)
    nodes = yaml.safe_load(pathlib.Path(args.graph).read_text())["nodes"]
    bad = problems(yaml.safe_load(CRITERIA_FILE.read_text()), nodes)
    result = prereg.stamp({"pass": not bad, "problems": bad}, reg)
    text = json.dumps(result, indent=1)
    if args.out:
        pathlib.Path(args.out).write_text(text + "\n")
    print(text)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
