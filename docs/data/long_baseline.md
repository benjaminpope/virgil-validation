# Long-baseline interferometry

Four VLTI UTs, 7 hour angles, 6 channels in H–K, declination −50°
(`tests/test_vlti.py`).

| Check | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| uv tracks vs `virgil.coverage.vlti_oidata` | Earth-rotation synthesis from station coordinates | First principles | 1e-9 m |
| our OIFITS file read by `OIData`; `data.model(...)` vs the V² and closure phases we wrote, for 4 scenes | our writer, our closure phases (T3 = V12 V23 V13*) | First principles | 5e-16 |
| noise-free fit recovery: binary; uniform-disk star + companion; star + elliptical envelope; star + modulated rim | injected truth | First principles + Analytic | 1e-10 to 1e-13 relative (rim 1e-7, optimiser tolerance) |
| pulls (fit − truth)/σ over 200 noisy realisations per case, closure-phase noise per baseline (correlated) or per triangle | N(0, 1) | Statistical | four scenes including the rim; means within ±0.17, sds 0.87–1.10 (sampling sd 0.05) |
| `laplace_cov` with array-valued parameters (rim `az_amps`, `az_pas`) | a finite, symmetric, positive-definite covariance over the flattened parameters | Mathematics | passes since virgil#135 (finding 5); the rim now has Laplace σ and joins the pull tests |

