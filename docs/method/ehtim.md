# eht-imaging: conventions and notes

[eht-imaging](https://github.com/achael/eht-imaging) (A. Chael et al.) is the
regularised maximum-likelihood imager of the Event Horizon Telescope
Collaboration. It is the external root for virgil's image model
(`Image`), its image regularisers and its RML imaging pipeline.
`scripts/setup_external.sh` installs it at a pinned version (1.3.2) into
its own environment, and `src/external_bridge/ehtim_worker.py` runs it in
a subprocess. It has no OIFITS reader, so it gets arrays or complex
visibilities rather than our files.

## Conventions

Read from `ehtim/imaging/imager_utils.py` and `ehtim/imager.py` (1.3.2),
and pinned by `tests/test_ehtim_imaging.py`,
`tests/test_imaging_regularisers.py` and
`tests/test_ehtim_reconstruction.py::test_objective_matches_ehtim`
(values to 1e-15, gradients to 1e-12).

| Quantity | eht-imaging | virgil |
| --- | --- | --- |
| Fourier sign | exp(+2πi(ux + vy)) (`ftmatrix`) | exp(−2πi(ux + vy)): virgil's visibility is eht-imaging's complex conjugate; the bridge passes (−u, −v) |
| pixel orientation | row 0 North, column 0 East | the same |
| amplitude χ² | `chisq_amp` = (1/N_a) Σ ((a − \|Ab\|)/σ)² | residual (\|V\| − a)/σ |
| closure-phase χ² | `chisq_cphase` = (2/N_c) Σ (1 − cos Δ)/σ², minimal set of triangles | chord 2 sin(Δ/2)/σ, whose square is 2(1 − cos Δ)/σ² (independent closure phases) |
| closure-phase error | from the bispectrum, √Σ(σ_i/\|V_i\|)² | as written in `OI_T3` (the same formula in our simulator) |
| total cost | Σ_d α_d (χ²_d − 1) + Σ_r β_r R_r (`Imager.objfunc`) | ½ Σ r² + regularisers |
| data weights | α_d | α_d = N_d/2 makes each term ½ Σ r² − α_d |
| squared TV | `stv2` (cost term −stv2 = Σ (Δb)²), zero padding, but only the steps out of the last row and column | `TSV(w)` = w Σ (Δb)² over every step, including those into the first row and column: TSV(w) = w (−stv2 + Σ b[0, :]² + Σ b[:, 0]²) (ledger **D9**) |
| regulariser normalisation | `norm_reg=True` divides by flux and pixel factors | none; we run eht-imaging with `norm_reg=False` |
| total flux | a soft `flux` term β (Σb − F)² | exactly 1 (softmax) |
| parameterisation | log pixels (`transform=['log']`), L-BFGS-B | softmax logits |

## Differences that are not errors

* **D9, TSV edges.** eht-imaging's `stv2` pads the image with zeros but
  counts only the steps out of its last row and column; virgil's `TSV`
  counts the steps across all four edges. The difference is
  w (Σ b[0, :]² + Σ b[:, 0]²), exact to 1e-16.
* **Flux.** eht-imaging holds the total flux with a penalty. To compare
  minima, the reconstruction raises its weight from 1e4 to 1e10 so the
  flux ends within 1e-6 of 1.

## Smoothness of the objective

With amplitudes and closure phases, the objective is undefined wherever a
model visibility is zero, because the closure phase is. On data near the
noise floor, eht-imaging's minimum can sit exactly on such nulls (|V|
~1e-10 on the faint scene of `test_ehtim_minimum_on_the_faint_scene`),
where neither code has a gradient. Stationarity is therefore checked on a
scene whose visibilities are all at least 30σ.

## Problems raised upstream

| # | eht-imaging | Problem | Evidence | Severity |
| --- | --- | --- | --- | --- |
| P6 | 1.3.2 | Under NumPy 2, `Obsdata.tlist` and `bllist` return tuples when every time group has the same size, so closure phases fail. Our worker works around it inside eht-imaging's `obsdata` module only. | `test_p6_closure_phases_with_equal_time_groups` (strict xfail) | to raise |
