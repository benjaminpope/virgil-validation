# PMOIRED: problems to raise upstream

A running list of behaviour in [PMOIRED](https://github.com/amerand/PMOIRED)
that our checks suggest is a problem on PMOIRED's side, not ours. Each entry
has a reproducer in this repository. They will be raised as Issues on
PMOIRED once there are enough of them, and only with Ben's approval.

| # | Version | Behaviour | Evidence | Severity |
| --- | --- | --- | --- | --- |
| P1 | 26.10.1 | `OI.setupFit` with the default `auto=True` sets `'wl kernel': 0.0` on our files (a few channels with a nominal 1 nm `EFF_BAND`), and every model observable is then NaN, with no warning. `auto=False` avoids it. | `docs/pmoired_conventions.md`; any file from `crosscheck.simulate.observe` | medium: silent NaNs |
| P2 | 26.10.1 | Rings with the default radial sampling differ from the analytic annulus by 4e-5 in V², but by 1e-5 with an explicit `'Nr': 100`, the documented default. | `tests/test_pmoired_conventions.py::test_annulus_converges_with_nr` | low: docs |
| P3 | 26.10.1 | Ring visibilities are exact (converging as Nr⁻² to 1e-9) when a baseline has at most 30 samples (epochs × channels), but stop at a floor of ~1e-4 in V² (0.1° in closure phase for a modulated ring) once it has more, whatever `Nr` is. It looks like a speed approximation (e.g. interpolation) that switches on silently. Point, disk and Gaussian components are unaffected. | 1 epoch × 30 channels: 1.1e-8; 1 × 35: 7.4e-5; 7 × 6: 1.4e-4 (2–4 mas annulus, 4 UTs, `Nr` 1000–8000) | medium: precision |
