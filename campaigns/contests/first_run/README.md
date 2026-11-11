# Informed fits: first run

One run of `scripts/contest_informed.py` per dataset, outside mandate, on a 4-core CPU with 8 random starts per rung
(`--starts 8`), virgil 0.4.1, this repository at `0a85d74`. These are informed fits: the model family comes from
the contest paper, every value from the data. They are not blind results. The per-dataset reports are the JSON files
beside this page. `agree` is how many starts reached the best chi2 of the chosen rung.

| dataset | family | chosen rung | chi2/N | converged | agree | time (s) |
|---|---|---|---|---|---|---|
| 2004_data1 | star_in_shell | star+shell+halo | 46.3 | yes | 1/9 | 40 |
| 2004_data2 | spotted_star_binary | star+companion+spot | 28.4 | yes | 3/14 | 65 |
| 2006_disk | thin_disk | disk | 210 | yes | 4/8 | 76 |
| 2008_agb_J | evolved_star_with_surface_structure | star+envelope+two spots | 0.81 | no | 1/14 | 466 |
| 2008_agb_H | evolved_star_with_surface_structure | star+envelope+two spots | 1.13 | no | 1/14 | 470 |
| 2008_agb_K | evolved_star_with_surface_structure | star+envelope+two spots | 3.64 | yes | 1/14 | 301 |
| 2008_agn_J | agn | core+dust+extended | 36 | no | 4/9 | 20 |
| 2008_agn_H | agn | core+dust+extended | 36.5 | yes | 5/9 | 18 |
| 2008_agn_K | agn | core+dust+extended | 52.4 | yes | 1/9 | 19 |
| 2010_lowH | structured_supergiant_with_companion | star+companion+two spots | 4.5e+03 | no | 1/14 | 803 |
| 2022_gravity | star_with_faint_companion | star+halo+companion | 4.47e+03 | no | 1/14 | 73 |
| 2022_ami | star_with_faint_companion | star+halo+companion | 1.39e+06 | yes | 5/14 | 23 |
| 2024_obj1_pionier | pinwheel_arcs | star+envelope+three arcs | 1.35 | yes | 1/9 | 254 |
| 2024_obj1_gravity | pinwheel_arcs | star+envelope+three arcs | 1.6 | yes | 2/9 | 228 |
| 2024_obj2_pionier | ringed_disk_with_companions | star+three rings+companion | 2.61 | yes | 2/14 | 653 |
| 2024_obj2_gravity | ringed_disk_with_companions | star+three rings+companion | 422 | yes | 4/14 | 1369 |
| 2010_lowK | structured_supergiant_with_companion | star+companion+two spots | 5.61e+03 | no | 2/14 | 788 |
| 2024_obj1_matisseL | pinwheel_arcs | star+envelope+three arcs | 2.41 | no | 1/9 | 1989 |
| 2024_obj1_matisseN | pinwheel_arcs | star+envelope+three arcs | 0.164 | yes | 2/9 | 1471 |
| 2024_obj2_matisseL | ringed_disk_with_companions | star+three rings+companion | 0.609 | yes | 2/14 | 1505 |

Not in the table: `2024_obj2_matisseN` did not finish in 40 minutes, and the 2018 data could not be downloaded.

What this shows. Every family builds and fits on its real data. Seven datasets reach chi2/N below 3: the 2008 AGB star
in J and H, 2024 Obj1 in PIONIER, GRAVITY and MATISSE L, and 2024 Obj2 in PIONIER and MATISSE L. 2024 Obj1 MATISSE N at
0.16 is an over-fit or has over-stated errors. The rest do not fit. The 2010 and 2022 data have very small quoted errors,
and 2010 has chromatic structure a grey model cannot follow. The 2004, 2006 and AGN families are too simple or stuck in
a local optimum. In most datasets only one or two starts reach the best chi2, so eight starts are too few to call any
optimum found, and seven chosen rungs did not converge. None of these numbers has been compared with a published value.
