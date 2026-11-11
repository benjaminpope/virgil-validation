Read the "Binary orbits: solved problems" table in this repository's `AGENTS.md` (and virgil's
`design/orbit_solved_problems.md`) before reporting anything. It is the authority; this is a pointer to it.

What you may do here is small on purpose. The pipeline chooses the model; you do not. Never add a period, a
window, an eccentricity cap, a diameter, a starting orbit or a night list from a paper, and never propose one.

| What you see | What it is | What to do |
|---|---|---|
| `data` fails: "no OIFITS under ..." or "N nights; at least 3 are needed" | Files not downloaded or not calibrated on this machine | `mandate request`: say which folder is missing. Do not change the night rule. |
| A GRAVITY system refuses `nights=all` | GRAVITY reads calibrated files only; the extra nights are not calibrated | Leave `nights` at `paper`. |
| Several alias bands with comparable evidence | Integer orbit count between nights is not determined by the data | A result. Report the bands; do not pick one. |
| A single night's position disagrees with every band | Single-night alias peak; the pipeline flags the night (it seeds no position, its data still count) | Nothing. If the flag did not fire, `mandate request` with the night. |
| Omega and omega differ from a reference by 180 degrees, or the companion is on the other side | Omega/omega twin or label swap: invisible to the sky positions | Nothing. Comparisons are in Thiele-Innes sky coordinates. |
| The fitted companion is the brighter star | Flux flip | Nothing: the pipeline tries it. |
| `no-orbit` wins, or the posterior is multimodal on few nights | The data do not determine an orbit | A result. Report it. Never add a rescue pass. |
| chi2/N on the quoted errors far above 1 on one night | Calibration of that night (gains, closure offsets) | Read the run's per-night table. `mandate request` naming the night; do not exclude it yourself. |
| R-hat, ESS or divergence check fails | The sampler did not converge | It stays a failure. Do not downgrade it. A rerun is a second attempt and is counted. |
| Out of memory in the grid search with gains | Known: grid batches are sized without the gains | The driver's `--grid-batch-vis`; `mandate request` if the default is not enough. |
