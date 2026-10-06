# Against eht-imaging

## Against eht-imaging's transform and data term

`tests/test_ehtim_imaging.py`. eht-imaging has no OIFITS reader, so it is
given arrays read from our files with astropy.

| virgil | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| `Image.model` (16², 15², 20² random images; three pixel sizes) | eht-imaging's `ftmatrix`, delta pixel response: virgil's visibility is its complex conjugate (eht-imaging's transform has the opposite sign; same pixel orientation, North row 0, East column 0) | eht-imaging | 1e-13 |
| control | flipped images, or no conjugate | eht-imaging | differ by > 1e-2 |
| closure-phase χ² of random images, three-telescope file, 2° noise | `chisq_cphase` × N (its 2(1 − cos Δ) is virgil's chord²) | eht-imaging | 1e-10 |

## Reconstructions against eht-imaging's imager

`tests/test_ehtim_reconstruction.py`. The two codes get the same simulated
data: three VLTI UTs, 13 snapshots and 6 channels, with complex Gaussian
noise on the visibilities of a two-Gaussian scene.
`crosscheck.simulate.observe_visibilities` writes these for virgil as
amplitudes (`OI_VIS` `VISAMP`) plus closure phases from the same
visibilities, and gives the visibilities themselves to eht-imaging.

The weights make the two objectives the same function of the image: ½χ² of
amplitudes and closure phases plus w Σ(Δb)², using eht-imaging's α_d = N_d/2
and unnormalised `tv2`. Two small differences of definition remain: the
edge steps in TSV, and the total flux, which virgil fixes exactly and
eht-imaging only softly.

| virgil | eht-imaging | Tag | Agreement |
| --- | --- | --- | --- |
| `fit` with `TSV`, started from eht-imaging's reconstruction | `Imager.make_image_I` (amplitudes, closure phases, `tv2`, flux) | eht-imaging | virgil's objective no lower than eht-imaging's minimum by more than 5e-8 relative; images equal to 1e-7 (L2), correlation 1 − 1e-14 |

The objective is not convex. From the same smooth start, the two codes'
parameterisations (softmax against log pixels) can end in different local
minima, and which one is lower depends on the data. So the comparison is
made in one basin. The problem is also stiff: the gradient at the shared
minimum is large, yet any step along it raises the loss.

Problem **P6** (eht-imaging, to raise with approval): under NumPy 2,
`Obsdata.tlist` and `bllist` turn records into tuples whenever every time
has the same baselines, which is true of all simulated data, and closure
phases then fail. Our worker works around it inside eht-imaging's
`obsdata` module only.

