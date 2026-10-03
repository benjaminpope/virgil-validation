# PMOIRED: problems to raise upstream

A running list of behaviour in [PMOIRED](https://github.com/amerand/PMOIRED)
that our checks suggest is a problem on PMOIRED's side, not ours. Each entry
has a reproducer in `tests/test_pmoired_problems.py`, written as a strict
`xfail` of the behaviour we expect, so the suite fails (and this page must
be updated) if PMOIRED changes. They will be raised as Issues on
PMOIRED once there are enough of them, and only with Ben's approval.

| # | Version | Behaviour | Evidence | Severity |
| --- | --- | --- | --- | --- |
| P1 | 26.10.1 | `OI.setupFit` with the default `auto=True` sets `'wl kernel': 0.0` on our files (a few channels with a nominal 1 nm `EFF_BAND`), and every model observable is then NaN, with no warning. `auto=False` avoids it. | `test_p1_default_setup_gives_finite_models` | medium: silent NaNs |
| P2 | 26.10.1 | Rings with the default radial sampling differ from the analytic annulus by 4e-5 in V², but by 1e-5 with an explicit `'Nr': 100`, the documented default. | `test_p2_default_nr_is_100` | low: docs |
| P3 | 26.10.1 | Ring visibilities are exact (converging as Nr⁻² to 1e-9) when a baseline has at most 30 samples (epochs × channels), but stop at a floor of ~1e-4 in V² (0.1° in closure phase for a modulated ring) once it has more, whatever `Nr` is. It looks like a speed approximation (e.g. interpolation) that switches on silently. Point, disk and Gaussian components are unaffected. | `test_p3_rings_exact_up_to_30_samples` (control) and `test_p3_rings_exact_beyond_30_samples`: 1 epoch × 30 channels: 1.1e-8; 1 × 35: 7.4e-5; 7 × 6: 1.4e-4 (2–4 mas annulus, 4 UTs, `Nr` 1000–8000) | medium: precision |
