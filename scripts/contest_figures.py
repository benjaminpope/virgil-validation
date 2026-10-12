"""Extract figures from the contest papers for the gallery (local only).

    python scripts/contest_figures.py inventory YEAR [--data DIR]
    python scripts/contest_figures.py split YEAR FIGURE [--data DIR]
    python scripts/contest_figures.py crop [YEAR ...] [--data DIR]

PDFs are read from DIR/papers/<year>.pdf (default ~/data/imaging_contests),
and everything is written under DIR/figures/<sha256 of the PDF>/, outside the
repository: the figures are copyrighted (contests/figures.yml records each
paper's licence). Needs PyMuPDF (``pip install pymupdf``).

- ``inventory`` lists each figure: its page, caption ("Fig./Figure N"), bbox
  in PDF points (origin top left) and the embedded raster images inside it,
  with their native pixel sizes. Writes inventory.json, a 72 dpi thumbnail per
  figure and a page overview with the bboxes drawn, for a reviewer (a person
  or a subagent) to choose panels from and write contests/figures.yml.
- ``split`` cuts a figure into panels along whitespace gutters (projection
  profiles of the rendered figure) and prints the panel bboxes in PDF points,
  with a thumbnail per panel to check; overrides go in the manifest by hand.
- ``crop`` makes each panel listed in contests/figures.yml: from the embedded
  bitmap at its native resolution where the panel lies within one image
  (lossless PNG), else rendered at ``DPI``. The PDF's sha256 must match the
  manifest's, so the bboxes refer to the same file.
"""

import argparse
import hashlib
import json
import pathlib
import re

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
DPI = 500  # vector figures
THUMB_DPI = 72
CAPTION = re.compile(r"^\s*(Fig\.|Figure)\s*(\d+)[.:]?", re.I)
WHITE = 245  # a gutter pixel is at least this bright in every channel
GUTTER = 0.995  # fraction of a row/column that must be white to be a gutter
MIN_SIZE = 10  # points: smaller "figures" are rules or underlines
MIN_PANEL = 0.08  # panels narrower than this fraction of the figure are merged


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def outdir(data, digest):
    out = data / "figures" / digest
    out.mkdir(parents=True, exist_ok=True)
    return out


def _union(rects):
    import pymupdf

    out = pymupdf.Rect()
    for r in rects:
        out |= r
    return out


def page_figures(page):
    """Figures on one page: caption-anchored. A figure's bbox is the union of
    the images and vector drawings lying between its caption and the text
    block above it (the previous caption or body text)."""
    import pymupdf

    blocks = [b for b in page.get_text("blocks") if b[6] == 0]  # text blocks
    images = []
    for info in page.get_images(full=True):
        xref = info[0]
        for rect in page.get_image_rects(xref):
            images.append({"xref": xref, "rect": rect, "width": info[2], "height": info[3]})
    drawings = [d["rect"] for d in page.get_drawings() if d["rect"].width > 2 or d["rect"].height > 2]
    figures = []
    captions = sorted((b for b in blocks if CAPTION.match(b[4])), key=lambda b: b[1])
    for cap in captions:
        top_of_caption = cap[1]
        # The figure starts below the lowest wide text block above its caption
        # (body text or the previous caption); narrow blocks are axis labels.
        wide = [b[3] for b in blocks if b[3] <= top_of_caption
                and ((b[2] - b[0]) > 0.5 * page.rect.width or CAPTION.match(b[4]))]
        ceiling = max([0.0] + wide)
        inside = lambda r: r.y0 >= ceiling - 1 and r.y1 <= top_of_caption + 1  # noqa: E731
        ims = [im for im in images if inside(im["rect"])]
        draws = [r for r in drawings if inside(r)]
        labels = [pymupdf.Rect(b[:4]) for b in blocks if inside(pymupdf.Rect(b[:4])) and not CAPTION.match(b[4])]
        rects = [im["rect"] for im in ims] + draws
        if not rects:
            continue
        near = _union(rects) + (-20, -20, 20, 20)
        bbox = _union(rects + [r for r in labels if r.intersects(near)])
        bbox &= page.rect
        if min(bbox.width, bbox.height) < MIN_SIZE:
            continue  # a rule or underline, not a figure
        number = int(CAPTION.match(cap[4]).group(2))
        figures.append({
            "figure": f"Fig. {number}",
            "page": page.number + 1,
            "bbox": [round(v, 1) for v in bbox],
            "caption": " ".join(cap[4].split())[:400],
            "images": [{"xref": im["xref"], "rect": [round(v, 1) for v in im["rect"]],
                        "native": [im["width"], im["height"]]} for im in ims],
            "vector": bool(draws) and not ims,
        })
    return figures


def render(page, bbox, dpi):
    """The page region ``bbox`` (points) as an RGB array."""
    import pymupdf

    pix = page.get_pixmap(clip=pymupdf.Rect(bbox), dpi=dpi, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]


def save(array, path):
    from PIL import Image

    Image.fromarray(array).save(path, optimize=True)


def inventory(pdf, data):
    import pymupdf

    digest = sha256(pdf)
    out = outdir(data, digest) / "inventory"
    out.mkdir(exist_ok=True)
    doc = pymupdf.open(pdf)
    figures = []
    for page in doc:
        found = page_figures(page)
        for f in found:
            save(render(page, f["bbox"], THUMB_DPI), out / f"p{f['page']:02d}_fig{f['figure'].split()[-1]}.png")
        if found:
            shape = page.new_shape()
            for f in found:
                shape.draw_rect(pymupdf.Rect(f["bbox"]))
            shape.finish(color=(1, 0, 0), width=1.5)
            shape.commit()
            save(render(page, page.rect, THUMB_DPI), out / f"page{page.number + 1:02d}.png")
        figures += found
    record = {"pdf": str(pdf), "sha256": digest, "pages": doc.page_count, "figures": figures}
    (out / "inventory.json").write_text(json.dumps(record, indent=1))
    return record


def gutters(profile, threshold=GUTTER):
    """Runs of non-gutter indices along one axis, as (start, stop) pairs; a
    gutter is where the fraction of white pixels is at least ``threshold``.
    Segments shorter than MIN_PANEL of the axis are merged into a neighbour."""
    content = profile < threshold
    runs, start = [], None
    for i, c in enumerate(content):
        if c and start is None:
            start = i
        elif not c and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(content)])
    n = len(profile)
    merged = []
    for r in runs:
        if merged and (r[1] - r[0] < MIN_PANEL * n or merged[-1][1] - merged[-1][0] < MIN_PANEL * n):
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [tuple(r) for r in merged]


def split_array(rgb):
    """Panels of a figure image as (row0, row1, col0, col1) pixel boxes:
    split by rows on full-width gutters, then each band by columns."""
    white = np.all(rgb >= WHITE, axis=-1)
    boxes = []
    for r0, r1 in gutters(white.mean(axis=1)):
        band = white[r0:r1]
        for c0, c1 in gutters(band.mean(axis=0)):
            # Trim the band's white margin above and below this panel.
            rows = np.nonzero(~band[:, c0:c1].all(axis=1))[0]
            boxes.append((r0 + int(rows[0]), r0 + int(rows[-1]) + 1, c0, c1))
    return boxes


def split(pdf, data, figure):
    import pymupdf

    digest = sha256(pdf)
    inv = json.loads((outdir(data, digest) / "inventory" / "inventory.json").read_text())
    match = [f for f in inv["figures"] if f["figure"] == figure]
    if not match:
        raise SystemExit(f"{figure} is not in the inventory of {pdf}")
    f = match[0]
    page = pymupdf.open(pdf)[f["page"] - 1]
    dpi = 150
    rgb = render(page, f["bbox"], dpi)
    x0, y0 = f["bbox"][:2]
    out = outdir(data, digest) / "inventory"
    panels = []
    for k, (r0, r1, c0, c1) in enumerate(split_array(rgb)):
        s = 72 / dpi
        bbox = [round(x0 + c0 * s, 1), round(y0 + r0 * s, 1), round(x0 + c1 * s, 1), round(y0 + r1 * s, 1)]
        save(rgb[r0:r1, c0:c1], out / f"p{f['page']:02d}_{figure.replace('. ', '')}_panel{k}.png")
        panels.append({"panel": k, "bbox": bbox})
    print(yaml.safe_dump(panels, default_flow_style=None, sort_keys=False))
    return panels


def crop_panel(doc, panel):
    """The panel's pixels: from the one upright embedded image that contains
    it, at native resolution, else rendered at DPI. Returns (array, source)."""
    import pymupdf

    page = doc[panel["page"] - 1]
    bbox = pymupdf.Rect(panel["bbox"])
    for info in page.get_images(full=True):
        xref = info[0]
        for rect, matrix in page.get_image_rects(xref, transform=True):
            # Native pixels only from an image drawn upright and unflipped;
            # a rotated or mirrored placement is rendered as printed.
            upright = abs(matrix.b) < 1e-6 and abs(matrix.c) < 1e-6 and matrix.a > 0 and matrix.d > 0
            if upright and bbox in rect + (-1, -1, 1, 1):
                pix = pymupdf.Pixmap(doc, xref)
                if pix.alpha or pix.n > 3:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
                if img.shape[-1] == 1:
                    img = np.repeat(img, 3, axis=-1)
                # Points to native pixels (the image is drawn unrotated).
                sx, sy = pix.width / rect.width, pix.height / rect.height
                c0, c1 = int(round((bbox.x0 - rect.x0) * sx)), int(round((bbox.x1 - rect.x0) * sx))
                r0, r1 = int(round((bbox.y0 - rect.y0) * sy)), int(round((bbox.y1 - rect.y0) * sy))
                return img[max(r0, 0):r1, max(c0, 0):c1, :3], f"embedded xref {xref}, native"
    return render(page, bbox, DPI), f"rendered at {DPI} dpi"


def crop(years, data):
    import pymupdf

    manifest = yaml.safe_load((ROOT / "contests" / "figures.yml").read_text())
    for year, paper in manifest["papers"].items():
        if years and year not in years:
            continue
        pdf = data / "papers" / f"{year}.pdf"
        if not paper.get("panels") or not pdf.exists():
            continue
        digest = sha256(pdf)
        if paper.get("sha256") and paper["sha256"] != digest:
            raise SystemExit(f"{pdf}: sha256 {digest} is not the manifest's {paper['sha256']}")
        doc = pymupdf.open(pdf)
        out = outdir(data, digest)
        for panel in paper["panels"]:
            array, source = crop_panel(doc, panel)
            save(array, out / f"{panel['id']}.png")
            print(f"{year} {panel['id']}: {array.shape[1]}x{array.shape[0]} px, {source}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["inventory", "split", "crop"])
    parser.add_argument("args", nargs="*")
    parser.add_argument("--data", default="~/data/imaging_contests")
    a = parser.parse_args()
    data = pathlib.Path(a.data).expanduser()
    if a.command == "inventory":
        for year in a.args:
            record = inventory(data / "papers" / f"{year}.pdf", data)
            for f in record["figures"]:
                print(f"{year} p{f['page']} {f['figure']}: {len(f['images'])} images, bbox {f['bbox']} | {f['caption'][:80]}")
    elif a.command == "split":
        year, figure = a.args[0], " ".join(a.args[1:])
        split(data / "papers" / f"{year}.pdf", data, figure)
    else:
        crop(a.args, data)


if __name__ == "__main__":
    main()
