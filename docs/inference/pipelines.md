# Scriptable pipelines

`tests/test_042_pipelines.py` checks the run-folder containers and end-to-end
synthetic truth recovery plus named quality-check controls across all five
`virgil.pipeline` classes against independent four-telescope (`UTS4`,
station-correlated closure phases) and three-telescope `crosscheck` forward
simulations (`crosscheck.simulate`, `crosscheck.oifits_writer`,
`crosscheck.sky`, `crosscheck.orbits`, and `crosscheck.orbit_cells_ref`).

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `Check`, `Stage`, `SkyImage`, `Result`, `load` | JSON dictionary round-trip, stage execution contract, pixel flux conservation, and HDF5/OIFITS/JSON run-folder reload without pickles (`ConfigMismatchError` on altered settings) | Mathematics | 1e-15 |
| `BinaryPipeline` | four-telescope synthetic binary recovery across `sep_mas`, `pa_deg`, `(dra_mas, ddec_mas)`, `flux`, `contrast = 1 / flux`, and `delta_mag = -2.5 log10(flux)` with `detection == "pass"` and `chi2 == "pass"`; over-calibrated null star (`V² > 1`) trips `detection == "warn"` and `chi2 == "fail"` | Mathematics | 0.15 mas; 2° PA; 5% flux/contrast |
| `StarPipeline` | four-telescope resolved uniform-disk recovery fitting both `UniformDisk` (`"uniform"`) and `QuadraticLimbDarkenedDisk` (`"limb_darkened"`) under `model=None` with `chi2_uniform == "pass"` and `chi2_limb_darkened == "pass"`; unresolved binary trips `chi2_uniform == "fail"` | Mathematics | 0.12 mas |
| `DiskPipeline` | inclined Gaussian ring recovery under `model="rings"` (`r_mas`, `sigma_mas`, `flux`, `inc_deg`, `pa_deg`, `chi2_reduced == "pass"`) and non-parametric `RadialGP` + MacKay empirical-Bayes rescaling under default `model="gp"` (`flux`, `inc_deg`, `pa_deg`, `chi2_reduced == "pass"`, `error_scale == "pass"`); under-quoted binary errors trip `chi2_reduced == "fail"` | Mathematics | 10% ring radius, 18% flux, 5° `(inc, pa)` |
| `ImagingPipeline` | staged regularized ensemble reconstruction (`run_member`) of a non-diagonal off-axis companion `(+4.2, +1.5)` mas (`star=True`), verifying world-coordinate flux-weighted centroid `(dra, ddec)` against transposed `(ddec, dra)` and `chi2_reduced`, `error_scale`, `star`, `orientation == "pass"`; tenfold under-quoted errors trip `chi2_reduced == "fail"` and `error_scale == "fail"` | Mathematics | < 0.9 mas centroid (> 2.5 mas vs transposed) |
| `OrbitPipeline` | five-night four-telescope blind Keplerian recovery (`AliasResult`, `summary["orbit"]` posterior medians and `samples()` across `period`, `a_mas`, `ecc`, `flux`, mirror-invariant Thiele–Innes `(A, B, F, G)` and epoch sky positions, `no_orbit == "pass"`, `chi2 == "pass"`); single-star noise trips `no_orbit == "fail"`. Note: upstream `period_window` / `period_cut` carry open API-docstring finding [F28](../method/findings.md) | Mathematics | < 2% period, < 4% `a`, < 0.15 mas sky |
