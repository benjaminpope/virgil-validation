# orbit_population: OzSTAR job files (staged here)

Job for the population injection-recovery gate of design/orbit_gate_042.md
(scripts/orbit_population.py). These files are staged in virgil-validation; they belong in
`ozstar_scripts/scripts/orbit_population/` (copy the directory, then `bin/oz push orbit_population`).
They follow `scripts/detection_v3/`: `submit.sh` pins virgil and virgil-validation by full commit,
`run_all.sh <virgil sha> <virgil-validation sha>` submits the fit array (660 tasks), the oracle
array (240 B-system tasks) and the aggregate. The SLURM time and memory limits are placeholders
until a first run records `seconds` and `peak_rss_mb` in each task file.
Results: `out/<pin>/{fit,oracle}_<S><index>.json`, `out/summary_<pin>.json`.
