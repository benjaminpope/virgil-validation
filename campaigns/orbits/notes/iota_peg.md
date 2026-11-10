---
applies_to:
  slug: iota_peg
source: scripts/gallery_orbits.py (MIRCX, MIRCX_GAINS)
drafted_by: agent, from the repository's records; review before relying on it
---
CHARA/MIRC-X, read from an OiDB collection (the authors' calibrated viscal files, one night per UT date).
There is no measured calibration template for MIRC-X, so the gain widths on log|V| and the closure offset are
placeholders; the per-night chi2/N says whether they suffice. The primary is resolved: the pipeline fits its
uniform-disk diameter itself.
