# AGENTS.md

Guidance for AI coding agents (Claude Code, Copilot and similar) working in
this repository.

## Purpose

`virgil-validation` checks [virgil](https://github.com/benjaminpope/virgil)
(`pip install virgil-astro`) against code that shares nothing with it:
geometric primitives, uv geometry, OIFITS files and aperture-masking images
are all built here from first principles or with
[dLux](https://github.com/LouisDesdoigts/dLux), and virgil only gets to
read, model and fit them.

## The independence rule

- `src/crosscheck/` is the independent side. It **must not** import virgil,
  drpangloss or jaxbessel (`tests/test_independence.py` enforces this), and
  it must not copy or paraphrase virgil's implementation. Write each piece
  from the physics or from a cited textbook formula, and say which in its
  docstring. Bessel and other special functions come from SciPy.
- `src/virgil_bridge/` is the only code that imports virgil: scene
  definitions pairing our truth with the virgil model to fit.
- `src/external_bridge/` calls other people's packages (PMOIRED, CANDID,
  fouriever, eht-imaging, MPoL, orbitize!), the heavy ones in their own
  environments through a worker script. It may not import virgil either. Pin their conventions from
  their documentation and our references first (`docs/*_conventions.md`),
  and only then compare them with virgil. Never copy their code (CANDID
  states no licence).
- Read virgil's *documentation* (docstrings, docs site) to learn its
  parameter conventions; do not read its source to learn how to compute
  something. When the documentation is ambiguous, implement the readings
  that are physically sensible and let the tests show which one virgil
  uses; report the ambiguity.
- A disagreement is a finding, not something to tune away. Keep it as a
  test: a passing test pinned to the behaviour virgil documents, and a
  `pytest.mark.xfail(strict=True)` test stating what is wrong, so that it
  flips when virgil is fixed. Record it in the README's findings table.

## Evidence records

Every test declares what it validates, against which roots of trust:

```python
@pytest.mark.validates("virgil.models.UniformDisk", roots=["mathematics"])
```

`kind` is `check` (the default), `control` (a wrong mapping that must fail),
`finding` (strict xfail: a known virgil problem), `upstream` (strict xfail: a
known problem elsewhere), `reference` (checks our own reference code) or
`guard`; `tier` is `A` (every PR, the default), `B` (weekly) or `C`
(campaigns). A test without the marker fails collection
(`--require-validates` in pyproject). Helpers record the numbers they check
with `evidence.plugin.record(name, value)`. `pytest --evidence PATH` writes
the records, after a header naming the exact virgil commit and package
versions; `scripts/evidence_table.py` renders them (`docs/evidence.md`, and
the CI job summary). Roots are listed in `src/evidence/__init__.py`;
`self-consistency` is the weakest and never enough on its own.

## Trust graph and ledger

- `trust/graph.yml`: virgil's objects (with their virgil source files,
  dependencies and how many distinct roots they need) and the end-to-end
  pipelines. Every object a test's `validates` marker names must be a node or
  a `pipeline:<name>` (`tests/test_trust_graph.py`).
- `trust/ledger.yml`: every mismatch, ruled `virgil`, `external:<package>`,
  `definition` or `crosscheck`, with the tests that pin it. Add an entry with
  every finding.
- `trust/evidence/`: evidence records (`latest.jsonl` from our suite,
  `virgil.jsonl` from virgil's own CI). Regenerate with
  `.venv/bin/python -m pytest --evidence trust/evidence/latest.jsonl`.
- `scripts/trust.py --evidence ... --virgil <virgil checkout> --out docs/trust.md --index docs/index.md` (the site's home page; the docs workflow regenerates both at deploy)
  writes the Trust page (statuses, chart, pipelines, ledger); weekly CI does
  the same against virgil's main.
- Roots are a pool (mathematics, standards, statistics, literature, dLux,
  PMOIRED, CANDID, fouriever, eht-imaging, MPoL, orbitize!): validate any link against
  whichever suits it. Do not claim a root a test does not really exercise.

## Binary orbits: frozen code and solved problems

**Freeze.** The T1-T5 orbit code (`scripts/reanalyse_orbits.py`, the orbit parts of `scripts/oidb_fit.py`,
`scripts/oidb_compare.py`, `scripts/orbit_gallery.py`, `scripts/orbit_diagnose.py`, and the peak picker of
`scripts/trackb_fit.py`) may only be deleted, never extended; new orbit work goes into virgil's `OrbitPipeline`
(plan Phases 1-2: one blind pipeline in virgil, vv keeps only a thin driver, the independent comparison and
the rendering). Fixes to these files are limited to what blocks a merge.

**Gallery runs.** The blind orbit fits of the ESO (Track B) gallery targets go through `scripts/gallery_orbits.py`, a thin
caller of virgil's `OrbitPipeline` (OzSTAR job `gallery_orbits` in ozstar_scripts). Do not add a fit path to the frozen files.
Exception, ruled by Ben on 2026-10-20: `scripts/orbit_gallery.py` has two rendering hooks (`flags_of` and `build_target` read the
`blind` block of a summary written by `scripts/gallery_pull_back.py`). They only display; nothing else of the frozen files may be extended.
Blind tiles always read "reported", because every row is `REPORT`.

### Binary orbits: solved problems (read first)

Each row was solved before; never re-derive one or report its symptom as an open problem. Rows without a
pinning test say which plan phase adds it. Memory-note pointer: `orbit-solved-toolkit`.

| Problem | Fix | Where | Test |
|---|---|---|---|
| Integer orbit count (alias band N = round(T/P)) | band scan over a period window, mode-keeping, tempered evidences | virgil `aliases.fit_orbit_aliases`; `reanalyse_orbits.py`; `oidb_fit.fit_orbit` | `test_oidb_fits.py::test_alias_bands_tile_the_window_and_split_the_o1_orbits`, `::test_orbit_search_is_blind_...` |
| Single-night alias peaks (Gl 229 2024-02-27, 2024-03-28) | flag the night: it seeds no position, its data still count | `design/orbit_comparison.md` s5; `oidb_fit.check_epochs` | `test_oidb_fits.py::test_check_epochs_flags_nights_that_may_not_seed_the_orbits` |
| Gl 229 2024-12-18 disagreement (OPEN) | left out of the files, reported as a finding, not tuned away | `oidb_fit.GL229_EXCLUDED_NIGHTS` | test pending: plan phase 3 (per-night flag row) |
| Published orbit leaking into fits (warm starts, seeds) | blind starts only; `fit_orbit` has no `published` argument; reference used after the fit | `oidb_fit.fit_orbit`; `oidb_compare` | `test_oidb_fits.py::test_orbit_search_is_blind_...`, `::test_compare_marks_a_gl229_orbit_seeded_by_the_published_orbit` |
| Period windows and e caps taken from papers | windows computed from the data; the Gl 229 10-14 d window is reference-informed | `oidb_fit.GL229_PERIODS` | test pending: plan phase 1c (`period_window`), phase 2d (grep test) |
| Omega/omega twin | compare in Thiele-Innes sky coordinates, never raw Omega/omega | `crosscheck.skyplane`, `crosscheck.orbits` | `test_skyplane.py`, `test_orbit_comparison.py::test_mirror_is_invisible_to_a_and_b_but_not_to_raw_elements` |
| Primary vs secondary omega in references | `PRIMARY_OMEGA` convention table | `reanalyse_orbits.py`, `oidb_compare` | `test_reanalyse_orbits.py::test_reference_epoch_units_and_the_primary_omega`; `omega_of` schema test pending: plan phase 2a |
| Twin fold / sign-free (label swap) | the swap is a point reflection and counts only with a label swap | `crosscheck.skyplane.score(sign_free=...)` | `test_skyplane.py::test_the_label_swap_is_a_point_reflection`, `::test_a_reflection_without_a_label_swap_fails` |
| Flux flip (companion brighter than the primary) | `flip_flux` option | virgil `aliases`; `reanalyse_orbits.py` | test pending: plan phase 1c |
| Resolved (UD) primary (iota Peg, 1.05 mas) | UD primary, UD-aware seeds | `reanalyse_orbits.py`; virgil `epoch_positions(primary_diam=)` | `test_reanalyse_orbits.py::test_every_system_has_a_primary_diameter_entry`; recovery test pending: plan phase 1b |
| Separation below the 0.82 mas floor | flag, do not score | `orbit_diagnose.py` | test pending: plan phase 1b (night flags) |
| Closure triangles whose leg has no V^2 row (T3 filter) | virgil places the leg from the T3 U/V | virgil `read_oifits` | `test_oidb_fits.py::test_triangles_with_a_baseline_in_no_vis2_row_are_kept` |
| V^2 amplitude (contrast loss) | per-night amplitude, for data without gains only (the legacy Gl 229 loader); calibrated gains supersede it and virgil refuses both (`calibrated_load.py`) | `oidb_fit.vis_amp_of`, `fit_orbit_vis_amp` | `test_oidb_fits.py::test_vis_amp_of_recovers_a_contrast_loss`, `::test_vis_amp_variant_...` |
| Untempered evidence (band p = 1.000 at chi2/N 12) | tempered likelihood, effective channels | virgil, plan phase 1d | test pending: plan phase 1d |
| Serial `lax.map` in the search | batched evaluation | virgil `aliases` | test pending: plan phase 1e |
| Unconverged NUTS reported as a result | `converged` flag, `r_hat`, `ess`, `divergences` checks | `oidb_fit`; virgil pipeline checks | test pending: plan phase 1e |
| HD 45166 binary calibrator | paper's adopted value is a mean over four calibrations; scored as "adopted" | `oidb_compare` | `test_oidb_fits.py::test_published_numbers_fix_the_conventions` |
| HR 6819 | reported, never scored (ruling 2026-10-07) | `oidb_fit.recipe_hr6819`, `oidb_compare` | `test_oidb_fits.py::test_hr6819_omega_table3_equals_table2_so_no_shift` |
| Reference transcription errors | references checked against the paper's own positions; papers can be wrong | `oidb_fit.reference_positions`, `oidb_compare` | `test_oidb_fits.py::test_reference_positions_agree_with_the_published_epochs`; `test_reanalyse_orbits.py::test_a_reference_that_misses_the_published_positions_is_an_error` |
| Stale verdicts on the aliases page | recompute verdicts from stored statistics under the current rule | `scripts/build_aliases_page.py` | test pending: plan phase 2c (`VERDICT_RULE_VERSION`) |
| False orbits from noise | `no-orbit` comparison against free positions and a single star | virgil `OrbitPipeline` | test pending: plan phase 1e / 4 (null injections) |

## Requesting other validations

If you want something validated that is not covered here, open an Issue on
this repository (https://github.com/benjaminpope/virgil-validation/issues)
describing the model or function, the analytic or independent result it
should match, and the precision you expect. Agents asked for a new
validation should check the open Issues first.

## Setup and commands

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ../drpangloss -e .   # local virgil checkout
# or: uv pip install --python .venv/bin/python -e .              # virgil-astro from PyPI
```

| Task | Command |
| --- | --- |
| Fast tests | `.venv/bin/python -m pytest -m "not slow"` |
| External packages (PMOIRED) | `uv pip install --python .venv/bin/python -e ".[external]"`, then `.venv/bin/python -m pytest -m external` |
| Everything (dLux masking, noisy pulls; ~10 min) | `.venv/bin/python -m pytest` |
| Campaigns (hours; OzSTAR), e.g. the orbitize! posterior | `VALIDATION_CAMPAIGNS=1 .venv/bin/python -m pytest -m campaign` |
| Regenerate the report and figures | `.venv/bin/python scripts/report.py` |
| Build the docs site | `uv pip install --python .venv/bin/python -e ".[docs]"`, `.venv/bin/python scripts/build_docs_index.py`, `.venv/bin/python scripts/trust.py --evidence trust/evidence/latest.jsonl --evidence trust/evidence/virgil.jsonl --out docs/trust.md --index docs/index.md`, `.venv/bin/zensical build --clean --strict` (Zensical, as virgil) |

The site (https://benjaminpope.github.io/virgil-validation/) is built from
`docs/` by `.github/workflows/docs.yml` and deployed on every push to
`main`; pull requests build it in strict mode. Add new pages to `nav` in
`mkdocs.yml`. The home page, `docs/index.md`, is the Trust page,
generated by `scripts/trust.py` at deploy (git-ignored); the Findings page,
`docs/method/findings.md`, is generated from `trust/ledger.yml` by the same
script. Planning documents live in `design/`, off the site.

Tests run virgil in float64 (`tests/conftest.py`) unless marked
`float32`. Heavy dLux runs use one process: do not run them in parallel
on a laptop.
