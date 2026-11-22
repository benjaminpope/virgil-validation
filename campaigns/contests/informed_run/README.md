# Informed fits: second pass

One run of `scripts/contest_informed.py` per dataset, outside mandate, on a 4-core CPU with 4 random starts per rung
(`--starts 4`), virgil 0.4.1, this repository at `31f4f98`. These are informed fits: the model family comes from the
contest paper, every value from the data. They are not blind results, and nothing here has been compared with a
published value. The per-dataset reports are the JSON files beside this page.

This replaces a first pass (grey models, free spots, symmetric disks) whose families were found to be mis-specified
when its quicklook was reviewed. The changes: stellar surfaces are spherical-harmonic maps, the 2006 disk may have
azimuthal structure, models are chromatic, a non-parametric image is fitted on top of each chosen model, and disk
families also get a multipolar Gaussian-process disk.

| dataset | chosen rung | chi2/N | vis. rms | phase rms (rad) | starts at best | image on top: chi2/N | its flux / model's | GP disk: chi2/N |
|---|---|---|---|---|---|---|---|---|
| 2004_data1 | star+shell+halo | 259 | 0.0106 | 1.22 | 1/5 | 2.81 | 2 | 11.8 |
| 2004_data2 | star+companion+spot | 28.4 | 0.0274 | 0.466 | 3/10 | 28.4 | 0.000101 |  |
| 2006_disk | disk | 272 | 0.00254 | 1.94 | 1/4 | 36 | 0.744 | 1.34e+04 |
| 2008_agb_J | surface degree 3+envelope | 0.785 | 0.0173 | 1.58 | 2/5 | 0.73 | 0.00987 |  |
| 2008_agb_H | surface degree 3+envelope | 0.898 | 0.0219 | 1.36 | 1/5 | 0.896 | 0.0057 |  |
| 2008_agb_K | surface degree 3+envelope | 1.44 | 0.0325 | 0.665 | 1/5 | 0.924 | 0.0795 |  |
| 2008_agn_J | core+dust+extended | 15.1 | 0.056 | 1.53 | 2/10 | 1.1 | 1.99 |  |
| 2008_agn_H | core+dust+extended | 19.6 | 0.0746 | 1.45 | 1/10 | 1.18 | 2 |  |
| 2008_agn_K | core+dust+extended | 13.9 | 0.0584 | 1.18 | 1/10 | 1.42 | 2 |  |
| 2010_lowH | surface degree 3+companion | 4.74e+03 | 0.0048 | 1.24 | 1/5 | 4.15e+03 | 0.0376 |  |
| 2022_gravity | star+halo+companion | 4.39e+03 | 0.0449 | 0.785 | 1/10 | 2.88e+03 | 0.206 |  |
| 2022_ami | star+halo+companion | 1.39e+06 | 0.0391 | 0.0361 | 4/10 | 1.06e+04 | 2 |  |
| 2024_obj1_pionier | star+envelope+three arcs | 1.5 | 0.0374 | 1.01 | 1/5 | 1.53 | 1.99 |  |
| 2024_obj1_gravity | star+envelope+three arcs | 1.17 | 0.0207 | 0.636 | 1/5 | 1.17 | 0.000451 |  |
| 2024_obj2_pionier | star+three rings+companion | 1.44 | 0.0438 | 0.0678 | 2/10 | 1.44 | 0.000312 | 1.24 |
| 2024_obj2_gravity | star+three rings+companion | 48.6 | 0.0185 | 0.0589 | 1/10 | 45.3 | 0.00363 | 265 |
| 2010_lowK | surface degree 3+companion | 6.05e+03 | 0.00644 | 1.22 | 1/5 | 4.44e+03 | 0.298 |  |

Missing: 2024 Obj1 MATISSE L and N and Obj2 MATISSE L (the run was stopped at its time limit before reaching them),
2024 Obj2 MATISSE N (timed out twice, at 40 and 60 minutes), and 2018 (its data could not be downloaded).

How to read it. chi2/N is on the quoted errors, which several contests made far smaller than any model residual, so the
residual columns say more. An image-on-top flux near 2 is the upper edge of its prior: the image is replacing the
parametric model, which means the family is the wrong shape (2004 data1, the 2008 AGN, 2022 AMI, 2024 Obj1 PIONIER).

What it shows. The harmonic surface fits the 2008 AGB star to chi2/N near 1 with little left over in J and H. 2010 no
longer has features far from the star, but its closure phases are still unmatched and the image on top finds compact
structure on the stellar disk: degree 3 is too coarse. 2006 is not solved: the rungs with a star and azimuthal harmonics
fitted worse than the symmetric disk, so those fits failed, and the GP disk failed too. The GP disk beats the ladder for
2024 Obj2 PIONIER (1.24 against 1.44) and 2004 data1 (12 against 259). Most optima were reached by one or two of the
starts, so all of this is provisional until it is run with more starts on a GPU.
