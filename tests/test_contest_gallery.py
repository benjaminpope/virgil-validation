"""The contest gallery (scripts/contest_gallery.py) and the paper-figure
extraction (scripts/contest_figures.py): orientation and flux on the display
grid, the member choice, panel splitting and native-resolution crops on a
synthetic PDF, and the public page's rules (our renders only, no design
links)."""

import importlib.util
import pathlib
import re

import numpy as np
import pytest
import yaml

ROOT = pathlib.Path(__file__).parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gallery = _load("contest_gallery")
figures = _load("contest_figures")


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_display_grid_keeps_east_left_and_flux():
    n, fov = 101, 20.0
    c = (50 - np.arange(n)) * fov / n  # offset East of column j, North of row i
    east, north = np.meshgrid(c, c)
    image = np.exp(-((east - 5.0) ** 2 + north**2) / 2)  # 5 mas East, σ 1 mas
    image /= image.sum()
    out = gallery.on_grid(image, fov, half=10.0, npix=161)
    assert out.sum() == pytest.approx(1.0, rel=1e-3)
    i, j = np.unravel_index(np.argmax(out), out.shape)
    east = (80 - j) * 20.0 / 161  # column 0 is East, as in the campaign's images
    assert east == pytest.approx(5.0, abs=0.15) and i == 80


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_enclosing_radius_of_a_ring():
    n, fov = 201, 40.0
    c = (np.arange(n) - 100) * fov / n
    r = np.hypot(*np.meshgrid(c, c))
    ring = (np.abs(r - 8.0) < 0.3).astype(float)
    assert gallery.enclosing_radius(ring, fov, 0.97) == pytest.approx(8.3, abs=0.2)
    assert gallery.nice(8.3 * 1.1) == 10.0


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_best_member_is_kept_and_highest_evidence():
    group = [
        {"member": 0, "chi2": 0.9, "log_z": -10.0},
        {"member": 1, "chi2": 1.4, "log_z": -5.0},  # kept (≤ 1.5) and best evidence
        {"member": 2, "chi2": 3.0, "log_z": 0.0},  # dropped: poor fit despite the evidence
    ]
    best, kept = gallery.best_member(group)
    assert best["member"] == 1 and kept == 2


def _synthetic_pdf(path):
    """A page with body text, a 2 x 2 grid of panels as one embedded bitmap
    (white gutters between panels) and a caption below it."""
    import io

    import pymupdf
    from PIL import Image

    grid = np.full((400, 600, 3), 255, np.uint8)
    for r0, c0, v in ((20, 20, 40), (20, 320, 90), (220, 20, 140), (220, 320, 190)):
        grid[r0 : r0 + 160, c0 : c0 + 260] = v
    buf = io.BytesIO()
    Image.fromarray(grid).save(buf, format="PNG")
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(pymupdf.Rect(72, 60, 540, 120), "Body text of the paper, wide enough to bound the figure from above. " * 3, fontsize=9)
    page.insert_image(pymupdf.Rect(156, 140, 456, 340), stream=buf.getvalue())  # 300 x 200 pt
    page.insert_textbox(pymupdf.Rect(72, 350, 540, 380), "Figure 3. Four panels: truth, winner and two entries.", fontsize=9)
    doc.save(path)
    return grid


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_figure_inventory_split_and_native_crop(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    pdf = tmp_path / "papers" / "2099.pdf"
    pdf.parent.mkdir()
    grid = _synthetic_pdf(pdf)
    record = figures.inventory(pdf, tmp_path)
    (fig,) = record["figures"]
    assert fig["figure"] == "Fig. 3" and fig["page"] == 1
    assert fig["images"][0]["native"] == [600, 400]
    assert fig["bbox"] == pytest.approx([156, 140, 456, 340], abs=1)
    # Whitespace projection finds the four panels.
    doc = pymupdf.open(pdf)
    rgb = figures.render(doc[0], fig["bbox"], 144)  # 2 px per point
    boxes = figures.split_array(rgb)
    assert len(boxes) == 4
    # Panel b (top right) in points: image columns 320-580 of 600 over 300 pt.
    bbox = [156 + 320 / 2, 140 + 20 / 2, 156 + 580 / 2, 140 + 180 / 2]
    array, source = figures.crop_panel(doc, {"page": 1, "bbox": bbox})
    assert "native" in source and array.shape == (160, 260, 3)
    assert np.all(array == grid[20:180, 320:580])  # lossless


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_public_page_shows_only_our_renders():
    config = yaml.safe_load((ROOT / "contests" / "gallery.yml").read_text())
    labels = {r["label"] for y in config["years"].values() for r in y["rows"] if r.get("label")}
    assets = ROOT / "docs" / "assets" / "contests"
    for png in assets.glob("*"):
        assert png.suffix == ".png" and png.stem in labels, png  # nothing but our renders
        assert png.stat().st_size < 100_000
    page = (ROOT / "docs" / "contest_gallery.md").read_text()
    assert not re.search(r"design[^)\s]*\.md", page)
    papers = yaml.safe_load((ROOT / "contests" / "figures.yml").read_text())["papers"]
    for year, paper in papers.items():
        # A paper whose licence is not open contributes links only.
        if not paper["licence"]["reusable"]:
            assert f"assets/contests/paper/{year}" not in page
