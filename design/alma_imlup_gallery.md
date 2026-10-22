# IM Lup gallery on the landing page: what it expects

The landing page has a third showcase slot for the ALMA IM Lup results
([`alma_imlup.md`](alma_imlup.md)). It is wired in `SHOWCASE` in
`scripts/trust.py` and stays hidden until its files exist. No images are made
here.

## Files the tile expects

All paths are under `docs/`.

| File | What it shows |
|---|---|
| `imaging/imlup.md` | The IM Lup results page. The tile links here. |
| `imaging/imlup/maxent.png` | virgil's MaxEnt image from the measurement sets. |
| `imaging/imlup/multipole_gp.png` | virgil's multipole GP image (`MultipoleDisk`). |
| `imaging/imlup/dsharp.png` | The DSHARP image, for comparison (with its attribution on the page). |

The tile is drawn once the page and at least one image exist. With two or
more images it becomes a strip of thumbnails. Missing images are skipped.
Keep the images square, on a white or transparent background, about 600 px
wide: the strip shows each at 8rem high, `object-fit: contain`.

## What the later PR also does

- Add `imaging/imlup.md` to the `nav` in `mkdocs.yml` (next to "Disc images"),
  or `zensical build --strict` warns.
- Set `page: imaging/imlup/` on the `alma-imlup` pipeline in
  `trust/graph.yml`, so its "Does virgil reproduce published results?" card
  links to the page. Update its `state` and `headline`.
- Change the file names in `SHOWCASE` if the results use other names.
