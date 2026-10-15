"""The image field and beam of every contest dataset, from the data headers alone.

    python scripts/contest_field_beam.py [--data ~/data/imaging_contests] > field_beam.json

No fit and no JAX computation: ``field_of_view``, ``beam`` and, where the
installed virgil has it, ``image_field`` (virgil#361) on each task's files.
Run it under two virgil versions and compare (``--compare old.json new.json``)
to see what the unflagged-baseline beam and the data-sized field changed.
``image_field`` is called with the second-moment FWHM only; ``starting_image``
passes the envelope fit's FWHM instead, so the field it picks can be larger.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def measure(data_dir):
    import virgil.imaging as imaging
    from virgil import OIData

    from contest_images import TASKS

    out = {}
    for spec in TASKS:
        paths = [data_dir / f for f in spec["files"]]
        if not all(p.exists() for p in paths):
            continue
        try:
            data = OIData(str(paths[0])) if len(paths) == 1 else [OIData(str(p)) for p in paths]
            b = imaging.beam(data)
            row = dict(beam=[b.major_mas, b.minor_mas], fov=float(imaging.field_of_view(data)))
            if hasattr(imaging, "image_field"):
                row["image_field"] = float(imaging.image_field(data))
        except Exception as exc:  # a dataset virgil cannot read is reported, not fatal
            row = dict(error=f"{type(exc).__name__}: {exc}")
        out[spec["label"]] = row
    return out


def compare(old, new):
    print(f"{'task':28s} {'beam before':>14s} {'beam after':>14s} {'field before':>13s} {'field after':>12s}")
    for k in new:
        o, n = old.get(k, {}), new[k]
        if "error" in o or "error" in n:
            print(f"{k:28s} error: {o.get('error') or n.get('error')}")
            continue
        fmt = lambda b: f"{b[0]:.1f}x{b[1]:.1f}"
        print(f"{k:28s} {fmt(o['beam']):>14s} {fmt(n['beam']):>14s} {o['fov']:13.1f} {n.get('image_field', n['fov']):12.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="~/data/imaging_contests")
    ap.add_argument("--compare", nargs=2, metavar=("OLD", "NEW"))
    a = ap.parse_args()
    if a.compare:
        compare(*(json.load(open(f)) for f in a.compare))
    else:
        json.dump(measure(pathlib.Path(a.data).expanduser()), sys.stdout, indent=1)
