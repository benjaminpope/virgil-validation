# Against eht-imaging

## Against eht-imaging's transform and data term

`tests/test_ehtim_imaging.py`. eht-imaging has no OIFITS reader, so it is
given arrays read from our files with astropy.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `Image.model` (16², 15², 20² random images; three pixel sizes) | eht-imaging's `ftmatrix`, delta pixel response: virgil's visibility is its complex conjugate (eht-imaging's transform has the opposite sign; same pixel orientation, North row 0, East column 0) | eht-imaging | 1e-13 |
| control | flipped images, or no conjugate | eht-imaging | differ by > 1e-2 |
| closure-phase χ² of random images, three-telescope file, 2° noise | `chisq_cphase` × N (its 2(1 − cos Δ) is virgil's chord²) | eht-imaging | 1e-10 |
| eht-imaging itself: `ftmatrix` of a sampled, offset Gaussian (σ = 2 mas, 0.25 mas pixels) | exp(−2π²σ²q²) exp(+2πi q·x₀), its documented sign | Mathematics (reference) | 1e-12 |

## Reconstructions against eht-imaging's imager

`tests/test_ehtim_reconstruction.py`. The two codes get the same simulated
data: three VLTI UTs, 13 snapshots and 6 channels, with complex Gaussian
noise on the visibilities of a two-Gaussian scene.
`crosscheck.simulate.observe_visibilities` writes these for virgil as
amplitudes (`OI_VIS` `VISAMP`) plus closure phases from the same
visibilities, and gives the visibilities themselves to eht-imaging.

The weights make the two objectives the same function of the image: ½χ² of
amplitudes and closure phases plus w Σ(Δb)², using eht-imaging's α_d = N_d/2
and unnormalised `tv2`. Two differences of definition remain: the edge
steps in TSV (ledger D9, an exact correction), and the total flux, which
virgil fixes exactly and eht-imaging holds with a penalty
([conventions](../method/ehtim.md)).

| virgil | eht-imaging | Tag | Agreement |
| --- | --- | --- | --- |
| data term and `TSV` on random images; gradients with respect to the logits | `Imager.objfunc` and its terms; `Imager.objgrad`, projected for the softmax | eht-imaging | values 1e-15; gradients 1e-12 |
| gradient of the objective at eht-imaging's minimum (bright scene, weights w and 2w), relative to the data term's | `Imager.make_image_I` run to convergence: flux weight 1e4 → 1e10, L-BFGS-B restarts | eht-imaging | 6e-4 and 3e-4; eht-imaging's own gradient there is 2e-3 and 1e-3 |
| control: the 2w minimum scored with w, and the w minimum with 2w | | eht-imaging | 0.50 and 1.00 |
| `fit` with `TSV`, started from eht-imaging's reconstruction (faint scene; regression) | `Imager.make_image_I` | eht-imaging | virgil's objective no lower than eht-imaging's minimum by more than 5e-8 relative; images equal to 1e-7 (L2) |

Stationarity needs a smooth objective, and with closure phases the
objective is undefined wherever a model visibility vanishes. On the faint
scene (data at ~1σ on the longest baselines) eht-imaging's minimum sits
exactly on such nulls (|V| ~ 1e-10), where neither code has a gradient.
The stationarity check uses a bright scene instead: two 0.8 mas Gaussians
with σ = 0.002, every visibility above 30σ.

The objective is not convex. From the same smooth start, the two codes'
parameterisations (softmax against log pixels) can end in different local
minima, and which one is lower depends on the data. So comparisons are
made in one basin.

Problem **P6** (eht-imaging, to raise with approval): under NumPy 2,
`Obsdata.tlist` and `bllist` turn records into tuples whenever every time
has the same baselines, which is true of all simulated data, and closure
phases then fail. Our worker works around it inside eht-imaging's
`obsdata` module only.

