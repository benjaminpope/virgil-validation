---
applies_to:
  bibcode: 2023A&A...672A.119G
source: oidb/trackb records (caveats); scripts/gallery_orbits.py
drafted_by: agent, from the repository's records; review before relying on it
---
The ESO Phase 3 GRAVITY products of these systems are uncalibrated (SINGLE_SCI_VIS). The fit does not read them
directly: it reads the files calibrated by this group under `calibrated/`, V² with gains and closure phases with
offsets, or closure phases only on a night that has no calibrator. The loader refuses the uncalibrated V².

The published astrometry of these systems is from the science channel only, and its quoted errors include
systematic floors, so a published error ellipse is not a statistical error.
