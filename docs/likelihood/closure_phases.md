# Correlated closure phases against fouriever

`tests/test_fouriever.py`; conventions in [fouriever](../method/fouriever.md).
These checks use a four-UT VLTI file (five snapshots, four channels, a 3 %
companion), plus a copy whose closure-phase errors are scaled by random
factors from 0.5 to 2.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `OIData.cp_noise` correlation matrix | fouriever's `CPCOV` / σσ; our T Tᵀ/3 from the file's station indices | fouriever + Mathematics | exact (1e-16) |
| χ² with correlated closure phases, equal errors, 12 binaries | fouriever `chi2_bin(cov=True)`, which equals our plain-residual rᵀC⁺r to 1e-15; virgil equals its sine-plus-penalty form (since virgil#174) to 1e-15 | fouriever + Mathematics | 5e-6 at the truth (definition D5 elsewhere) |
| the same, unequal errors | each code's own generalised inverse, written independently | fouriever + Mathematics | 1e-15 each; the two differ by 2–7 % (definition D6) |
| control: fouriever without its covariance | — | fouriever | differs by 6–12 %, as it should |
| our plain reference across the ±180° cut (companion brighter than the primary) | fouriever's unwrapped residual | fouriever + Mathematics | 1e-15 (a wrapped reference would be off by up to 100 %) |
| — | fouriever on data 3° across the cut from a 0.99 flux-ratio binary | Mathematics | fouriever's χ² explodes, virgil's does not (problem P5) |
| which inverse is calibrated: virgil's `cp_noise.whiten` on 100 000 closures of unequal baseline-phase noise | mean χ² of the independent combinations (3 for four telescopes) | Statistics | virgil closer to 3 than the pseudo-inverse of C; the exact model T S Tᵀ gives 3 |

