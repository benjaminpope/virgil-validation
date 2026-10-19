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


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_informed_run_is_shown_only_when_closer_to_one():
    runs = [{"name": "tv", "chi2": 1.3}, {"name": "gp", "chi2": 0.8}]
    assert gallery.closeness(0.5) == gallery.closeness(2.0) == 2.0
    assert gallery.choose_informed(runs, {"chi2": 1.1}) is None  # 0.8 is as far below 1 as 1.25 above
    assert gallery.choose_informed(runs, {"chi2": 2.0})["name"] == "gp"
    assert gallery.choose_informed(runs, None)["name"] == "gp"  # no blind fit: shown
    assert gallery.choose_informed([{"name": "gp", "chi2": 0.2}], {"chi2": 0.1}) is None  # both overfit
    assert gallery.choose_informed([], None) is None


def _advice_run(root, name, **arrays):
    run = root / name
    run.mkdir(parents=True)
    np.savez(run / f"{name}.npz", pixel_scale_mas=0.5, npix=4, **arrays)
    (run / f"{name}.txt").write_text("residuals=[{'chi2_v2': 1.5, 'chi2_cp': 0.5}]\n")


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_informed_runs_read_each_output_format(tmp_path):
    img = np.ones((4, 4))
    _advice_run(tmp_path, "x_gp_r2", image=img, raw_chi2_red=1.2)
    _advice_run(tmp_path, "x_kern_r2", image_iso=img, image_aniso=2 * img, log_z_iso=np.array([-5.0]),
                log_z_aniso=np.array([-3.0]), raw_chi2_red=2.0)
    _advice_run(tmp_path, "x_tv_r2", images=np.stack([img, 3 * img]), index=np.array('{"discrepancy": 1}'), chi2_red=np.ones((2, 1)))
    (tmp_path / "x_tv_r2" / "x_tv_r2.txt").write_text("raw chi2/N=0.9 (quoted errors)\n")
    _advice_run(tmp_path, "x_sweep_r2", mean=img, raw_chi2_red=1.0)  # an ensemble mean: skipped
    by_name = {r["name"]: r for r in gallery.informed_runs(tmp_path, "x")}
    assert set(by_name) == {"gp", "kern", "tv"}
    assert by_name["gp"]["fov"] == 2.0 and by_name["gp"]["chi2_v2"] == 1.5 and by_name["gp"]["chi2_cp"] == 0.5
    assert by_name["kern"]["image"].max() == 2.0  # the higher-evidence kernel
    assert by_name["tv"]["image"].max() == 3.0 and by_name["tv"]["chi2"] == 0.9
    assert gallery.informed_runs(tmp_path, "xx") == []  # a prefix match needs the underscore


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_page_badges_informed_fits_and_keeps_failures():
    page = (ROOT / "docs" / "contest_gallery.md").read_text()
    assert "Where we stand" in page and "cg-informed" in page
    # Every informed card says so in its caption, and a failed fit is never shown without the word.
    assert page.count("so it is not a blind result") >= page.count("<figcaption>virgil, informed (")


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
    # A horizontal rule above a caption-like line: not a figure.
    page.draw_line(pymupdf.Point(72, 420), pymupdf.Point(540, 420))
    page.insert_textbox(pymupdf.Rect(72, 430, 540, 460), "Figure 4. A table caption under a rule.", fontsize=9)
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
    # A rotated placement is rendered as printed, never cut from the bitmap.
    page = doc[0]
    page.insert_image(pymupdf.Rect(156, 480, 356, 780), stream=page.parent.extract_image(fig["images"][0]["xref"])["image"], rotate=90)
    array, source = figures.crop_panel(doc, {"page": 1, "bbox": [160, 490, 350, 770]})
    assert "rendered" in source


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="guard")
def test_public_page_shows_only_our_renders():
    config = yaml.safe_load((ROOT / "contests" / "gallery.yml").read_text())
    labels = {r.get("label") or r.get("advice") for y in config["years"].values() for r in y["rows"]} - {None}
    assets = ROOT / "docs" / "assets" / "contests"
    for png in assets.glob("*"):
        assert png.suffix == ".png" and png.stem.rsplit("_", 1)[0] in labels, png  # nothing but our renders
        assert png.stat().st_size < 100_000
    page = (ROOT / "docs" / "contest_gallery.md").read_text()
    assert not re.search(r"design[^)\s]*\.md", page)
    papers = yaml.safe_load((ROOT / "contests" / "figures.yml").read_text())["papers"]
    for year, paper in papers.items():
        # A paper whose licence is not open contributes links only.
        if not paper["licence"]["reusable"]:
            assert f"assets/contests/paper/{year}" not in page
