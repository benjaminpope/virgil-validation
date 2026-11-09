# IM Lup stride-1000 subsample

Every 1000th row of each of the 26 spectral windows (ddids) of the DSHARP IM Lup Band 6 continuum
measurement set: 42,424 visibilities in `ddid_00.npz` to `ddid_25.npz` (u, v in metres, real and imaginary
parts, weights, frequency, antenna pair, channel, time, scan, EB index) plus `info.json` (rows, channels,
channel width and integration time per ddid). Visibilities, uv coordinates, weights and bookkeeping only;
no pickled objects. 2.5 MB.

Source: ALMA public archive data, IM Lup, reduced for DSHARP (Andrews et al. 2018). Programmes
2013.1.00226.S, 2013.1.00694.S, 2013.1.00798.S and 2016.1.00484.L (seven execution blocks; the IM Lup
validation is described in `docs/imaging/imlup.md`; Andrews et al. 2018, ApJ 869, L41, is the DSHARP overview).

Made with `python scripts/alma_imlup_subsample.py --stride 1000` from the 1.9 GB measurement set
(not in git). It is what `tests/test_alma_mpol.py` reads; set `IMLUP_SUBSAMPLE` to point elsewhere.
