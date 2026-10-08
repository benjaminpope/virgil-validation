# Orbit-alias re-analysis: data inventory (2026-10-09)

Planning note, off the site (not linked from the docs). Which systems have undersampled
multi-epoch interferometric data, where it lives, and what reference orbit exists. The
loaders and registry are in `scripts/reanalyse_orbits.py`; the results page is
`docs/orbits/aliases.md`. Local paths are on the laptop; Track B and Gl 229 are mirrored on
OzSTAR under `/fred/oz440/bpope/eso_binaries/`.

| system | instrument | data path | epochs (MJD) | reference orbit | used here |
|---|---|---|---|---|---|
| Gl 229 Ba-Bb | GRAVITY K, SC closure phases | `~/data/eso_binaries/gravity/oifits/<night>/` (40 files) | 7 nights: 60304.2, 60308.2, 60368.1, 60398.0, 60429.97, 60663.3, 60718.1 | Xuan+2024, Nature 634, 1070, Table 1: P 12.13 d, e 0.23, i 31 deg, a 7.3 mas (`xuan2024_table1.json`; omega is the primary's; Omega node unverified) | yes, p_range 10-14 d |
| Apep | GRAVITY dual-field | `~/data/apep_gravity/{reduced,calibrated}/` | 6 nights, 60042.2 to 60747.3 | plume-model elements (White+2025), Omega convention unresolved | no: calibration script outside the repo, data not on /fred |
| Apep | NACO SAM | `~/data/apep_naco/oifits/` | 4 nights in one run (58562-58566) | none | no: not an orbit |
| 9 Sgr, delta Vel, HD 136164 | NACO SAM | `~/data/eso_binaries/naco/oifits/<sys>/` | 1 epoch each | Fabry+2021; none; Balmer+2024 | no: one epoch |
| HR 4049 | PIONIER H | `~/code/nuHor/data/HR4049_pionier_data_all/` | 7 nights, 58884-58899 | none (no period) | no: no period, notebook loaders only |
| hd41255 | GRAVITY K | `trackb/hd41255/oifits/` | 14, 59231-59968 | Gallenne+2023 Table 6: P 148.3 d | yes |
| hd188088 | GRAVITY K | `trackb/hd188088/oifits/` | 16, 59440-59879 | Gallenne+2023: P 46.8 d | yes |
| omi_leo | GRAVITY K | `trackb/omi_leo/oifits/` | 9, 59638-59971 | Gallenne+2023: P 14.5 d | yes |
| hd70937 | GRAVITY K | `trackb/hd70937/oifits/` | 14, 59197-59971 | Gallenne+2023: P 27.9 d | yes |
| hd210763 | GRAVITY K | `trackb/hd210763/oifits/` | 9, 59442-59804 | Gallenne+2023: P 42.4 d | yes |
| kap_vel | GRAVITY K | `trackb/kap_vel/oifits/` | 8 of 9, 60670-60731 | orbit_fit_table: P 116.8 d | yes (two months of data) |
| zet_boo, eta_oph | GRAVITY K | `trackb/<sys>/oifits/` | 4; 6 | ORB6 + GRAVITY | no: no tabulated P |
| del_cir | GRAVITY + PIONIER | `trackb/del_cir/oifits/` | 8 of 12 | spinOS, P 1603 d | no: mixed instruments |
| tz_for | GRAVITY + PIONIER | `trackb/tz_for/oifits/` | manifest 0, files 16 | Gallenne+2018: P 75.7 d | no: manifest conflict |
| psi_cen | PIONIER H | `trackb/psi_cen/oifits/` | 6, 57481-57816 | Gallenne+2019 Table 2: P 38.8 d | yes |
| nn_del | PIONIER H | `trackb/nn_del/oifits/` | 4, 57954-58016 | Gallenne+2019: P 99.3 d | yes |
| al_dor | PIONIER H | `trackb/al_dor/oifits/` | 18 of 19, 57387-58118 | Gallenne+2019: P 14.9 d (twins: compare mod 180 deg) | yes |
| alf_equ | PIONIER H | `trackb/alf_equ/oifits/` | 7 of 12, 56937-57625 | VB+SB orbit table: P 98.8 d | yes |
| 9sgr, hd152314, hd168137, kq_vel, cpd-71_172, tyc1703-394-1 | PIONIER / GRAVITY | `trackb/<sys>/oifits/` | 1-2 epochs | various | no: too few epochs |

Track B layout: `trackb/<sys>/system.json` (epochs, reference positions and orbit),
`manifest.json`, `catalogue.json`. Loaders reused: `scripts/trackb_fit.py` (`epochs_of`, `load`,
`grid_data`, `geometry`) for Track B; the per-night reader of
`~/data/eso_binaries/gravity/scripts/fit_nights.py` (copied, with its excluded exposure) for Gl 229.
For Track B the period prior window is P_ref/1.25 to 1.25 P_ref, a window around the published
period and not a blind search; Gl 229 uses 10-14 d.

Independence: reference orbits and every drawn orbit track use `crosscheck.orbits`; the Track B
reference omega is the RV (primary's) value plus 180 deg, and Omega is as published.
