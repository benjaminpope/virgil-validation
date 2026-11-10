"""Build the registries and the generated notes of the mandate campaigns from this repository's own records.

    python campaigns/build_registries.py            # rewrite them
    python campaigns/build_registries.py --check    # fail if they are out of date (CI)

Nothing here is typed from memory or from a paper. The orbit campaign's rows come from ``oidb/trackb/*.json`` and the
``MIRCX`` table of ``scripts/gallery_orbits.py``; the contest campaign's rows from ``TASKS`` and the member tables of
``scripts/contest_images.py``, and its notes from ``contests/manifest.yml``. What a contestant or a blind fit may know
goes into open columns and open notes; what a paper revealed afterwards goes into sealed columns and sealed notes,
which only a campaign's sealed stages are given. Needs PyYAML only (no virgil, no JAX).
"""

import argparse
import ast
import csv
import io
import json
import pathlib
import sys

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
HERE = REPO / "campaigns"
MEMBERS = 8  # ensemble members per contest dataset (design/plan_imaging_contests.md, C2d)


def assigned(path, name):
    """The value a script assigns to a module-level name, evaluated on its own (the script is not imported)."""
    tree = ast.parse((REPO / path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return eval(compile(ast.Expression(node.value), str(path), "eval"), {"dict": dict})  # our own file
    raise KeyError(f"{path} assigns no {name}")


def table(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def note(front, body):
    return "---\n" + yaml.safe_dump(front, sort_keys=False, allow_unicode=True, width=1000) + "---\n" + body.strip() + "\n"


def orbits():
    """{relative path: text} of the orbit campaign's generated files."""
    out, rows = {}, []
    for path in sorted((REPO / "oidb" / "trackb").glob("*.json")):
        r = json.loads(path.read_text())
        split = r.get("split") or {}
        files = r.get("files") or []
        rows.append({
            "slug": r["slug"], "target": r["target"], "instrument": r["instrument"], "band": r.get("band", ""),
            "source": "trackb", "bibcode": (r.get("paper") or {}).get("bibcode", ""),
            "template": r.get("template") or "", "phase3_files": len(files),
            "phase3_nights": len({f.get("night_mjd") for f in files}),
            "heldout_nights": split.get("heldout_nights", ""),
            # Sealed: both are read off the published orbit.
            "published_kind": r.get("kind", ""), "published_epochs_below_resolution": split.get("paper_epochs_below_lambda_2B", ""),
        })
        paper = r.get("paper") or {}
        body = [f"Published solution: `oidb/trackb/{path.name}` ({paper.get('authors', '')} {paper.get('year', '')}, {paper.get('journal', '')}).",
                "", f"Kind (from the published orbit): {r.get('kind')}.", "", f"Third light: {r.get('third_light')}", "",
                f"Caveats recorded with the reference: {r.get('caveats')}", "",
                f"Conventions: {(r.get('conventions') or {}).get('text', '')}"]
        out[f"orbits/notes/sealed_{r['slug']}.md"] = note(
            {"applies_to": {"slug": r["slug"]}, "visibility": "sealed", "source": paper.get("bibcode", ""),
             "generated_by": "campaigns/build_registries.py"}, "\n".join(body))
    for slug, cfg in sorted(assigned("scripts/gallery_orbits.py", "MIRCX").items()):
        rows.append({"slug": slug, "target": slug.replace("_", " "), "instrument": cfg["instrument"], "band": "H", "source": "oidb",
                     "bibcode": "", "template": "", "phase3_files": "", "phase3_nights": "", "heldout_nights": "",
                     "published_kind": "", "published_epochs_below_resolution": ""})
    out["orbits/registry/units.csv"] = table(sorted(rows, key=lambda x: x["slug"]))
    return out


def contests():
    """{relative path: text} of the contest campaign's generated files."""
    tasks = assigned("scripts/contest_images.py", "TASKS")
    stars = assigned("scripts/contest_images.py", "MEMBER_STARS")
    out, rows = {}, []
    for index, t in enumerate(tasks):
        for k in range(MEMBERS):
            rows.append({
                "id": f"{t['label']}_m{k}", "label": t["label"], "year": t["label"].split("_")[0], "task": index, "member": k,
                "files": ";".join(t["files"]),
                "field_mas": t.get("field", ""), "field_source": t.get("field_source", "chosen from the data"),
                "start": f"gauss {t['prior'][1]:g} mas" if t.get("prior") else "CLEAN",
                "star_given": "yes" if t.get("star") else "no", "star_source": t.get("star_source", ""),
                "channels_um": "{:g}-{:g}".format(*t["wavel"]) if t.get("wavel") else "",
                "chromatic_star": "yes" if t.get("sparco") else "no",
                # What this member tries: contest_images.member_settings.
                "member_star": stars[k % len(stars)],
                "member_halo": "yes" if t.get("halo_members") and k % 8 >= 4 else "no",
            })
    out["contests/registry/units.csv"] = table(rows)
    manifest = yaml.safe_load((REPO / "contests" / "manifest.yml").read_text())["contests"]
    used = {r["year"] for r in rows}
    for year, c in sorted(manifest.items()):
        if year not in used:
            continue
        files = [f["name"] for f in c.get("files", []) if f.get("role") == "contest"]
        tests = [f["name"] for f in c.get("files", []) if f.get("role") == "test"]
        # The manifest's `instrument` text also says what the organisers simulated, so it stays sealed.
        body = [f"Contest {year} ({c.get('bibcode')}).", "",
                f"What contestants were given before the deadline: {c.get('presubmission', 'nothing recorded')}", "",
                "Contest data files: " + ", ".join(files) + ".",
                ("Test binaries handed out to check conventions: " + ", ".join(tests) + ".") if tests else ""]
        out[f"contests/notes/given_{year}.md"] = note(
            {"applies_to": {"year": year}, "source": f"contests/manifest.yml ({c.get('bibcode')})",
             "generated_by": "campaigns/build_registries.py"}, "\n".join(body))
        targets = c.get("targets") or {}
        body = ["Revealed after the deadline. For scoring only, never for imaging.", "",
                f"Instrument and simulation: {c.get('instrument')}", "",
                "Targets: " + "; ".join(f"{k}: {v}" for k, v in targets.items()), "",
                f"Winner: {c.get('winner')}", "", f"Notes: {c.get('notes', '')}", "", f"Truth: {c.get('truth')}"]
        out[f"contests/notes/sealed_{year}.md"] = note(
            {"applies_to": {"year": year}, "visibility": "sealed", "source": f"contests/manifest.yml ({c.get('bibcode')})",
             "generated_by": "campaigns/build_registries.py"}, "\n".join(body))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="do not write; fail if a generated file is out of date")
    a = ap.parse_args(argv)
    stale = []
    for rel, text in {**orbits(), **contests()}.items():
        path = HERE / rel
        if a.check:
            if not path.exists() or path.read_text() != text:
                stale.append(rel)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print("wrote", path.relative_to(REPO))
    if stale:
        sys.exit("out of date, rerun campaigns/build_registries.py: " + ", ".join(stale))
    return 0


if __name__ == "__main__":
    sys.exit(main())
