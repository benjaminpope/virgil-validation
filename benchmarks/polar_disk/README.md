# PolarDisk benchmarks (stage V)

Run from the repository root with the repo venv (`.venv`, Python 3.12, `pip install -e ".[polar-disk]"`).
Every script writes its JSON (provenance: virgil commit, versions, machine) to `$PD_OUT`
(default `benchmarks/polar_disk/results`). Pinned virgil: `791f9fef65eb49ebc387c5c365a13c7af2342341`
(`origin/polar-p4`, PR #343); each result records `virgil_pin_ok`. Times and memory below are
estimates from the single laptop runs (M-series, 1 CPU thread-ish) unless marked; none was timed on
OzSTAR. Run the scripts one at a time on a laptop; they are meant for OzSTAR.

`export PD_OUT=/path/to/out V3_DATA=/path/to/data` then:

| Script | Command | Env needs | Runtime | Memory |
|---|---|---|---|---|
| V1 forward vs frank, vis-r, FRAP | `.venv/bin/python benchmarks/polar_disk/v1_forward.py` | none | 1-2 min | < 2 GB |
| V2 brute force | `.venv/bin/python benchmarks/polar_disk/v2_brute_force.py` | none | 1-2 min | < 2 GB |
| V2 PMOIRED, oimodeler | `.venv/bin/python benchmarks/polar_disk/v2_external.py` | none | 1 min | < 2 GB |
| V2 cone old vs new | `.venv/bin/python benchmarks/polar_disk/v2_cone_old_new.py` | `.venv-old/` with virgil `a1dae85584b78d19771e42a19a6146ff74414f2f` installed | 2-5 min | < 2 GB |
| V3 HR 4796 (vis-r) | `.venv/bin/python benchmarks/polar_disk/v3_hr4796.py` | `V3_DATA/visr/hr4796.selfcal.npy` (12.9 MB) | 2-5 min | ~2 GB |
| V3 AS 209 (frank) | `.venv/bin/python benchmarks/polar_disk/v3_as209.py` | `V3_DATA/frank/AS209_continuum.npz` (37 MB), `AS209_clean_profile.txt` | frank 2 min + virgil 1 min | ~3 GB |
| V3 HD 169142 (FRAP) | `.venv/bin/python benchmarks/polar_disk/v3_frap.py` | `V3_DATA/frap/HD169142_band{3,6}.pkl` | 1-3 min | < 2 GB |
| V4 laptop | `.venv/bin/python benchmarks/polar_disk/v4_injection_recovery.py --config laptop` | none | ~2 min | ~2 GB |
| V4 full (one array task) | `.venv/bin/python benchmarks/polar_disk/v4_injection_recovery.py --config full --instrument nrm --degree 2 --seed-offset 0` | none | est. 200 injections x 8 starts at n_r = 64: 10-30 core-hours per task (untimed) | ~4 GB |

V4 full: 2 instruments x 3 degrees = 6 tasks (`--instrument` in nrm, vlti; `--degree` in 1, 2, 3); further
`--seed-offset` values add independent replicates. Merge the `v4_full_*.json` rows afterwards; K1 to K4 are
in the script docstring. Data downloads are listed in `docs/models/polar_disk.md`; they are untrusted, keep
each in its own directory (`V3_DATA/{visr,frank,frap}`).
