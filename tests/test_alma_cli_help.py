"""`--help` must not crash on any alma script or subcommand (argparse expands a bare `%` in help strings)."""
import argparse
import importlib.util
import pathlib
import sys

import pytest

SCRIPTS = sorted((pathlib.Path(__file__).resolve().parents[1] / "scripts").glob("alma_*.py"))


def _parsers(script):
    """Build the script's top-level parser (and its subparsers) by running main() with ArgumentParser.parse_args intercepted."""
    spec = importlib.util.spec_from_file_location(f"_help_{script.stem}", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    found = []
    orig = argparse.ArgumentParser.parse_args

    def grab(self, *a, **k):
        found.append(self)
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = grab
    try:
        try:
            mod.main(["--help"]) if mod.main.__code__.co_argcount else mod.main()
        except SystemExit:
            pass
    finally:
        argparse.ArgumentParser.parse_args = orig
    if not found:
        return []
    out, stack = [], [found[0]]
    while stack:
        p = stack.pop()
        out.append(p)
        for act in p._actions:
            if isinstance(act, argparse._SubParsersAction):
                stack.extend(act.choices.values())
    return out


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda s: s.name)
def test_help_renders(script):
    parsers = _parsers(script)
    assert parsers, f"{script.name}: no ArgumentParser reached parse_args"
    for p in parsers:
        p.format_help()
