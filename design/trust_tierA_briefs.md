# Tier A work packages: trust graph and provenance

Baseline: origin/main **98b7dda** (#213 merged, 2026-11-08 21:58Z). Every file:line below is at 98b7dda.
"Published evidence" means what `docs.yml:49-51` builds the Trust page from today:
- push run 42467174675 (vv b70b370, virgil d79696d);
- weekly run 41381202862 (2026-11-02, virgil main 8ad1d11);
- `trust/evidence/campaigns.jsonl`.

The push run for 98b7dda was still in progress when I measured, so the page will shift slightly once it lands. Each worker should re-measure its "expected change" on the newest artefacts (`scripts/fetch_ci_evidence.sh`) before opening the PR.

**Tier A** means all of these hold:
- the package changes no science number, threshold or criteria file;
- it only makes a hidden state visible, adds provenance, or adds a lint that fails on a real inconsistency;
- no verdict changes, except where the brief names a verdict that is already wrong;
- it is size S or M, and easy to revert.

## Preamble: paste above every brief

> Repository: benjaminpope/virgil-validation, branch from origin/main. Read AGENTS.md first.
>
> Python rules:
> - Wrap every python or pytest call as `~/.claude/bin/pyslot <cmd>`, with `JAX_NUM_THREADS`/`OMP_NUM_THREADS` ≤ 2.
> - Use `.venv/bin/python`.
> - Run only the targeted tests named in this brief, never the full suite.
>
> Scope rules:
> - Change no science number, tolerance, threshold or criteria file (`design/*criteria*`).
> - If a test you touch fails for a reason outside this brief, report it in the PR; do not tune it.
> - The PR description must include `scripts/trust_diff.py before.json after.json`, built from the published CI evidence with `scripts/trust.py --json` before and after your change. It must also name every verdict or pipeline status that changed. A change not listed in this brief is a stop-and-ask.

## Shared core files and order

Packages that touch the same core file must land one at a time:

| Core file | Packages |
|---|---|
| `scripts/trust.py` | A1, A2, A6, A8b, A11 |
| `src/evidence/plugin.py` | A1, A3, A4, A5 |
| `src/evidence/meta.py` | A4, A7 |

Suggested order:
1. Wave 1, in parallel (only A1 touches core files): A1, A8a, A9, A10, A12, A13, A14, A15, A16.
2. Wave 2, one per core file: A2 (`trust.py`), A3 (`plugin.py`), A7 (`meta.py`).
3. Wave 3: A11 (`trust.py`), A4 (`plugin.py` and `meta.py`), then A5 (`plugin.py`) and A6 (`trust.py`).
4. Wave 4: A1b, A1c and A1d (markers only, parallel), then A8b (`trust.py`).

A1 must land before A1b-d.

---

### A1. Headlines: an explicit headline must exist, and the live rml-imaging card is fixed (folds old T1 into T8)

- **Goal.** The page never shows a number the test did not choose.
  - Today, `_headline` (`scripts/trust.py:195-210`) silently falls back to the first metric in recording order that matches `AGREEMENT` when a marker's `headline=` names no recorded metric.
  - Only 6 records in the whole suite set `headline=`.
  - **Live defect:** the published `rml-imaging` pipeline card shows `ehtim_rel_log_gradient` (6.0e-5 and 1.6e-4). That is eht-imaging's own convergence, not the check. The test asserts on `rel_gradient_at_ehtim_minimum` (`tests/test_ehtim_reconstruction.py:289-309`, records at 306-307, assert at 309).
  - **Latent defect:** `tests/test_alma_mpol.py:509` has `headline="MPoL's objective on IM Lup"`, which is prose. The test skips in CI, so the card is empty today.
- **Files and changes.**
  - `src/evidence/plugin.py`: in `pytest_runtest_makereport` (177) or `pytest_runtest_logreport` (198), when the test passed and its marker has a `headline` that is not among the recorded metric names, turn the outcome into a failure with the message "headline X is not a recorded metric".
  - `scripts/trust.py` `_headline`: when `record["headline"]` is set but missing from `metrics`, return `(None, None)`. Do not fall back.
  - Add `headline="rel_gradient_at_ehtim_minimum"` to the marker at `tests/test_ehtim_reconstruction.py:289`.
  - Change `tests/test_alma_mpol.py:509` to the virgil metric that the test asserts on. Read the test to pick it.
  - Add a page warning listing check records with more than one candidate and no explicit headline. A1b-d fix those.
- **Done when:**
  - a synthetic test with a bad `headline=` fails;
  - a synthetic record with a missing explicit headline shows no number;
  - the `rml-imaging` card in `trust.py --json` shows `rel_gradient_at_ehtim_minimum`.
- **Tests:**
  - `tests/test_evidence_metrics.py` (add the synthetic cases here);
  - `tests/test_trust_rules.py`;
  - `tests/test_ehtim_reconstruction.py::test_ehtim_minimum_is_stationary_on_a_well_posed_problem` (needs ehtim; it skips otherwise; slow, so run it once, in the background).
- **Core files:** `trust.py`, `plugin.py`.
- **Page change:** the rml-imaging headline number changes, and a "headline chosen by order" warning lists 66 test functions. No verdict changes.
- **Size:** S.

### A1b / A1c / A1d. Explicit headlines for the 66 order-dependent check functions (old T8)

- **Goal.** Every check record with more than one candidate metric names its headline. Today 105 records in 66 functions are order-dependent; that list is the A1 warning.
- **Rule for choosing.** Pick the metric that the test's assert bounds and that compares virgil with the reference. Never a reference-versus-truth metric, and never a metric that measures the reference's own convergence. When it is unclear, leave the test out and list it in the PR.
- **Split by file group,** to avoid conflicts with other workers:
  - **A1b, radio/ALMA/imaging:** `test_alma_clean_start`, `test_alma_gp`, `test_alma_maxent`, `test_alma_mpol`, `test_alma_radio`, `test_alma_start_floor`, `test_clean_recovery`, `test_clean_scenes`, `test_ehtim_reconstruction`, `test_imaging_regularisers` (12), `test_l_curve`, `test_radio_innovations`, `test_radio_spread_offsets`, `test_radio_trust_numpy` (9).
  - **A1c, VLTI models and companions:** `test_candid`, `test_detection`, `test_flux_visphi`, `test_fouriever`, `test_fouriever_limits`, `test_gravity_darkened_independent` (8), `test_limb_darkening`, `test_pmoired_fits`, `test_pmoired_vs_virgil` (10).
  - **A1d, orbits and inference:** `test_epochs`, `test_inference_joint`, `test_linear_marginals`, `test_orbitize` (9), `test_orbits_more`.
- **Change:** add `headline=` to the markers only. No assertion, metric or tolerance changes.
- **Done when:** the A1 warning lists no function from the group.
- **Tests:** `--collect-only -q` on the touched files (cheap). Run at most three of the touched tests, the fastest, to confirm that A1's teardown check passes.
- **Core files:** none (test markers only). Depends on A1.
- **Page change:** headline numbers on some node cards change. No verdict changes.
- **Size:** M each, mechanical with judgement.

### A2. A strict XPASS marks the node failing (old T2)

- **Goal.** Make the code do what its docstring says ("failing: a check of it fails (or a strict xfail passes)", `scripts/trust.py:20`).
  - pytest records a strict XPASS as `failed` with no `wasxfail`.
  - `trust.py:381` excludes kinds `finding` and `upstream` from `failing` whatever their outcome, so an XPASS is dropped.
- **Change:** at `trust.py:381`, count a `finding`/`upstream` record whose outcome is `failed` or `error` as failing. Keep excluding `xfailed`.
- **Measured effect** on the published evidence with the patch applied:
  - no change today (no XPASS exists);
  - a simulated XPASS of a pmoired or ehtim upstream test moves only that node (`pmoired`/`ehtim`: partly → failing). There is no cascade, because these external nodes are already "partly" and so not trusted references.
  - Main has 0 `kind="finding"` tests and 10 `upstream` tests.
- **Done when:** a new case in `tests/test_trust_rules.py` (a passing check plus a failed upstream record → `failing`) passes.
- **Tests:** `tests/test_trust_rules.py`.
- **Core files:** `trust.py`.
- **Page change:** none today. A verdict changes only when an XPASS happens, and then the old verdict was wrong.
- **Size:** S.

### A3. `raises=` on every strict xfail (old T4)

- **Goal.** A strict xfail counts only the failure it states, not any exception, such as an API change.
  - 9 of the 10 strict xfails have no `raises=`:
    - `tests/test_candid.py:312`;
    - `tests/test_ehtim_reconstruction.py:369`;
    - `tests/test_fouriever.py:189`;
    - `tests/test_orbitize.py:484`;
    - `tests/test_pmoired_problems.py:36,50,67`;
    - `tests/test_polar_disk_vs_codes.py:123,162`.
  - `tests/test_fouriever_limits.py:139` already has `raises=RuntimeError`.
  - Not `tests/test_evidence_pin.py:93`: that is the plugin's self-test, not a finding.
- **Files and changes:**
  - `src/evidence/plugin.py:140-152`: for `finding`/`upstream`, also require a `raises=` kwarg.
  - The same lint rejects `raises=Exception` and `raises=BaseException`, also inside a tuple, because they match any failure. For `AssertionError`, the `reason=` must name the asserted quantity. Implemented in PR #234.
  - The 9 sites: take the exception type from the CI log of the newest push run. Use `gh run view <id> --log` and look at the xfail reason, or run the test locally once.
  - Update `tests/test_evidence_pin.py::test_collection_enforces_strict_xfail_and_finding_bi_implication`.
- **Done when:**
  - collection passes;
  - each of the 9 tests still xfails, with the stated exception, on CI's environment. Check the PR's own CI run.
  - If one now fails, it was xfailing for the wrong reason. Report it and stop. Do not change its assertion.
- **Tests:** the 7 test files above, then `tests/test_evidence_pin.py`.
- **Core files:** `plugin.py`.
- **Page change:** none expected. If a test turns out to xfail for the wrong reason, that external node's verdict was wrong. Name it in the PR.
- **Size:** S.

### A4. Make missing evidence visible: collection skips, test selection, and virgil import skips (old T5, plus the critique's addition)

- **Goal.** No test vanishes without trace.
  - Module-level skips leave no record, because the plugin writes records only in `pytest_runtest_logreport` (`plugin.py:198-233`). There are 98 such sites in 67 files (counted by AST over `tests/*.py`): `importorskip` plus `skip(allow_module_level=True)`. The single `pytestmark = skipif` site (`test_radio_trust_casa.py:45`) does record.
  - **84 of the 98 are `importorskip("virgil.…")` imports, across 61 files.** If virgil removes or renames a module, those 61 files' tests disappear and CI stays green.
- **Files and changes:**
  - (a) `src/evidence/plugin.py`: add `pytest_collectreport`. When a module is skipped, read its `validates` markers with the AST (the way `tests/test_trust_graph.py:_validates_objects` does). Write one record per test function with `outcome: "skipped"`, `phase: "collection"` and the reason.
  - (b) Same file: when the environment variable `VV_REQUIRE_VIRGIL_IMPORTS=1` is set, a collection skip whose reason is a failed import of `virgil` or `virgil.*` is a collection **error**. Do not edit the 61 test files.
  - (c) `src/evidence/meta.py` `run_header` (89): add `argv` (`config.invocation_params.args`), the `-m`/`-k` expressions, and the deselected count. `pytest_sessionstart` (236) takes the header, so pass `config` in or fill these at `pytest_sessionfinish` (244).
  - (d) `.github/workflows/tests.yml`: set `VV_REQUIRE_VIRGIL_IMPORTS=1` on the pytest steps (48 and the full-run step).
- **Done when:**
  - running `tests/test_calibrated_load.py` without `virgil_vlti` writes 9 records with `phase: "collection"`;
  - a pytester test shows a module importing a nonexistent `virgil.nope` errors with the variable set and skips without it;
  - the header names `-m "not slow"`;
  - the PR's own CI run is green (all 84 virgil import skips resolve at `VIRGIL_REF`). If one does not resolve, that is a real finding: report it.
- **Tests:** a new pytester test in `tests/test_evidence_metrics.py`; `tests/test_trust_rules.py::test_provenance_is_taken_before_the_evidence_file_exists`.
- **Core files:** `plugin.py`, `meta.py`.
- **Page change:** nodes gain skipped records. The only file fully skipped in CI today is `test_calibrated_load.py`, 9 guards on node `evidence`. No verdict changes.
- **Size:** M.

### A5. Unlisted `golden:` roots are an error, not silently nothing

- **Goal.** `roots_of` (`trust.py:148-163`) drops a `golden:<name>` root that is not in `trust/golden.yml` without saying so. `tests/test_alma_conventions.py:137` uses `golden:visread`, which is not listed (only `dholakia` is).
- **Files and changes:**
  - `src/evidence/plugin.py` `_claims` (105): a `golden:` root not in `trust/golden.yml` is a collection error.
  - `trust/golden.yml`: add `visread`, filling `what`, `produced_by` and `used_in` from the test's docstring. Ask in the PR if provenance is unclear; do not guess a commit.
- **Measured effect:** `crosscheck.alma` is already verified on `casa`, `mathematics` and `statistics` (need 1), so adding the golden root changes no verdict.
- **Done when:** a synthetic marker with `golden:unlisted` fails collection, and collection of the tree passes.
- **Tests:** `tests/test_trust_rules.py`, `tests/test_alma_conventions.py --collect-only`.
- **Core files:** `plugin.py`.
- **Page change:** `crosscheck.alma` lists one more root. No verdict changes.
- **Size:** S.

### A6. Pipelines whose checks all skipped show "not run" (old T19)

- **Goal.** At `trust.py:528`, a pipeline with no passing and no failing check takes its hand-written `state`, so a skipped check reads as "planned" or "running".
- **Change:** when a pipeline has checks and all of them skipped, its status is `not run`, with the skip reason. Add a legend entry for it.
- **Measured effect** on the published evidence:
  - `alma-imlup`: planned → not run (its 1 check skips: the IM Lup subsample is not in CI);
  - `orbitize-posterior`: running → not run (1 skipped);
  - `binary-orbits` has no checks, so it keeps `running`.
- **Done when:** a synthetic pipeline whose checks all skipped shows `not run` in `tests/test_trust_rules.py`.
- **Tests:** `tests/test_trust_rules.py`, `tests/test_docs_links.py`.
- **Core files:** `trust.py`.
- **Page change:** the two pipeline statuses named above. Both were wrong.
- **Size:** S.

### A7. Richer run header, and `validation_dirty` that sees edited evidence (old T7)

- **Goal.** A record says where and how it ran.
  - `run_header` (`src/evidence/meta.py:89-110`) has no platform, JAX backend or precision, job id or thread caps.
  - `runner` comes from `GITHUB_ACTIONS` (103), which anyone can set.
  - `validation_dirty` ignores all of `trust/evidence/` (98), so a run with a hand-edited `campaigns.jsonl` reads as clean.
- **Changes in `meta.py`:**
  - Add `platform` (OS and arch);
  - `jax` (version, backend, devices, `jax_enable_x64` as read at session start);
  - `env` (`XLA_FLAGS`, `OMP_NUM_THREADS`, `JAX_NUM_THREADS`);
  - `host` (hostname);
  - `slurm` (`SLURM_JOB_ID`, `SLURM_ARRAY_JOB_ID`, `SLURM_ARRAY_TASK_ID` when set);
  - `github_run_id` (raw env value). Keep `runner` as is, and document in its docstring that it is the runner's claim.
  - Make `validation_dirty` ignore only the evidence output path (`--evidence PATH`), not the whole `trust/evidence/` folder.
  - No hashing.
- **Done when:** a local run's header has the new fields, and a run with an edited `trust/evidence/campaigns.jsonl` reports `validation_dirty: true`.
- **Tests:**
  - `tests/test_trust_rules.py::test_provenance_is_taken_before_the_evidence_file_exists`;
  - `tests/test_trust_rules.py::test_runs_on_uncommitted_or_foreign_code_are_left_out_of_the_page`;
  - `tests/test_evidence_pin.py`.
- **Core files:** `meta.py`. Also update `scripts/evidence_table.py` if it prints the header.
- **Page change:** none.
- **Size:** S.

### A8a. Ledger lint: status vocabulary, and documentation entries made explicit

- **Goal.** A ledger status outside `UNFIXED` (`trust.py:343`) reads as fixed, so a typo such as `opne` silently clears a "Known bug" verdict. No test checks `status` (`tests/test_trust_graph.py:58-68` checks only ruling, objects and evidence ids).
- **Change:** a new file, `tests/test_ledger_consistency.py`, does not touch `test_trust_graph.py`. It checks:
  1. `status` ∈ {open, to-raise, raised, documented, fixed}, the vocabulary at `trust/ledger.yml:10`;
  2. every `kind="upstream"` (and `finding`) test is cited by an entry with an unfixed status and a matching ruling (`external:*` for upstream, `virgil` for finding);
  3. every `fixed` entry cites at least one test;
  4. an entry with `evidence: []` carries a `pending:` field saying why. Today F22 (433) and D15-D17 (523, 533, 542) need it. For F22, name the branch with its tests; do not invent tests.
  - Entries whose `kind` starts with `documentation` (F23 446, P11 465) are exempt from (2). They cannot have a strict xfail, because the code is right.
- **Done when:** the new test passes, and `docs/method/findings.md` is regenerated (`scripts/trust.py --findings`).
- **Tests:** the new file, `tests/test_trust_graph.py`, `tests/test_docs_links.py::test_findings_page_is_the_ledger`.
- **Core files:** none. Edits only `trust/ledger.yml` (adds `pending:` fields).
- **Page change:** none.
- **Size:** S-M.

### A8b. Structured `post_result` on analysis decisions, and a badge (old T11 field, plus the badge half of T12)

- **Goal.** The decision whether something was decided after seeing a result is free text inside `kind` today. Make it a field and show it.
  - D15 and D22 are post-result;
  - D16, D20 and D21 are "unconfirmed";
  - D17 is "false in intent".
- **Changes:**
  - `trust/ledger.yml`: add `post_result: true|false|unconfirmed` and `affects: [node or pipeline ids]` to D15-D17 and D20-D22. Copy the values from the existing text. Do not reinterpret them.
  - Extend `tests/test_ledger_consistency.py` to require these fields on `ruling: definition` entries whose `kind` mentions post-result.
  - `scripts/trust.py`: render a "rests on post-result decision Dn" badge on the affected node and pipeline cards. **No cap and no verdict change**; the cap is tier B.
- **Done when:** the D15 and D22 badges appear in `trust.py --json` and the page, and the tests pass.
- **Tests:** `tests/test_ledger_consistency.py`, `tests/test_trust_rules.py`, `tests/test_docs_links.py`.
- **Core files:** `trust.py`. Depends on A8a.
- **Page change:** badges only.
- **Size:** M.

### A9. virgil's CI evidence reaches the page honestly (old T6, ingest route only)

- **Goal.** `scripts/ingest_virgil_evidence.py:58` writes `validation_commit: None`, so `--strict-provenance` drops the file. Reproduced: "does not say which commit of virgil-validation it ran on".
- **Change:** record this repository's commit (`git rev-parse HEAD` at ingest) and dirty flag in the header. Do not add an exemption in `provenance_problem`.
- **Measured effect** of including the file: one verdict changes, `virgil.detection.injection_recovery` unchecked → partly (a weak root only). All 134 virgil-ci records passed; no node can reach verified from them.
- **Done when:** a test feeds the ingest output to `load_evidence(strict=True)` and finds it kept.
- **Tests:** a new case in `tests/test_trust_rules.py`, or the ingest script's own test if one exists.
- **Core files:** none.
- **Page change:** weak `virgil-ci` roots appear, and injection_recovery reads partly. That verdict was wrong: virgil does test it.
- **Size:** S.

### A10. `test_calibrated_load.py` runs in CI (old T18)

- **Goal.** Its 9 tests (`kind="guard"` on node `evidence`) never ran in CI, because `virgil_vlti` is not installed (`tests/test_calibrated_load.py:13`).
- **Change:** in `.github/workflows/tests.yml`, install `virgil-vlti` (public) at a pinned commit next to `VIRGIL_REF`, as virgil is (13, 26-29). Use the newest virgil-vlti main commit and say so in the PR.
- **Done when:** the PR's CI evidence holds 9 records from that file. Check the artefact.
- **Tests:** `tests/test_calibrated_load.py` locally if `virgil_vlti` is installed; otherwise rely on the PR's CI.
- **Core files:** none.
- **Page change:** 9 guard records; no verdict.
- **Size:** S.
- **Coordination note:** A4 makes this gap visible first; A10 closes it. They can land in either order.

### A11. Weekly failures on virgil main are shown, not overwritten (new)

- **Goal.** The newest weekly run (41381202862, virgil main 8ad1d11) has **18 failed and 2 errored** tests:
  - `TSV.__init__() missing ... 'beam_mas'` across `test_imaging_regularisers`, `test_l_curve` and `test_ehtim_reconstruction`;
  - `test_inference_joint.py::test_regularized_inverse_and_its_warning`, which asserts 1.47 < 1e-12.
  - `load_evidence` (`trust.py:100-136`) keeps the newest record per test by run date. The later push run at the *older* pin d79696d passed all 20, so every failure is overwritten. The page shows `virgil.imaging.TSV`, `TV`, `models.Image`, `inference.regularized_inverse` and `virgil_bridge.alma` as verified, with no sign that they fail on virgil main.
- **Change (display only):**
  - in `build`, collect tests whose newest record from a run on a *different virgil commit* failed while the kept record passed;
  - render a "fails on virgil main (8ad1d11, run link)" badge on those nodes, plus a banner listing them, next to `_older_campaigns_html`;
  - do not change the verdict; that is tier B (B4).
- **Done when:**
  - a synthetic case in `tests/test_trust_rules.py` (an old failure at commit X, then a newer pass at commit Y) shows the badge;
  - the badge applies only when the failing run's virgil commit is newer than the kept record's. A synthetic reverse-order case (a pass at the newer commit Y, and a failure at the older commit X, so the kept record is the newer one) must show no badge. Being fixed in PR #236;
  - on the published evidence the banner lists the 20 tests and these nodes: TSV, Image, whitened_residuals, TV, l_curve, fitting.fit, regularized_inverse, `virgil_bridge.alma`, `pipeline:rml-imaging`.
- **Tests:** `tests/test_trust_rules.py`, `tests/test_docs_links.py`.
- **Core files:** `trust.py`.
- **Page change:** a banner and badges. No verdict changes.
- **Size:** S-M.

### A12. Criteria commits are ancestors of HEAD (the safe half of old T13)

- **Goal.** A committed campaign or gate summary's `criteria_commit` must be in this repository's history. Otherwise the recorded commit cannot be checked: a squash merge or force-push would orphan it.
- **Change:** in a new test, `tests/test_criteria_history.py`, for every committed JSON under `trust/campaigns/` and the gate outputs listed in `tests/test_gate_criteria.py`, check that a `criteria_commit` present there is an ancestor of HEAD (`git merge-base --is-ancestor`). Skip with a reason on a shallow clone. No hashing; do not touch any criteria file or its change log.
- **Done when:** the test passes on a full clone, and a scratch-repo case with a non-ancestor commit fails.
- **Tests:** the new file, `tests/test_gate_criteria.py`.
- **Core files:** none.
- **Page change:** none.
- **Size:** S.

### A13. Every graph `source` path exists in virgil at the pin (cheap first step of R9)

- **Goal.** `source:` in `trust/graph.yml` is hand-written and drives the freshness dot (`changed_files`, `trust.py:166-180`). A wrong path silently reads as "unchanged".
- **Change:** a new test resolves virgil's source root from the installed `virgil.__file__` and asserts that every `source` path of every node exists there.
- **Done when:** the test passes, or the PR fixes the paths that do not exist (graph bookkeeping only; list each one).
- **Tests:** the new test, `tests/test_trust_graph.py`.
- **Core files:** none.
- **Page change:** freshness dots may change where a path was wrong. No verdict changes.
- **Size:** S.

### A14. A verdict diff in every PR's job summary (display half of old T16)

- **Goal.** Nobody sees verdict diffs today. `scripts/trust_diff.py` exists but no workflow runs it.
- **Change:** in `.github/workflows/tests.yml` (pull_request only):
  - build `trust.py --json` on the PR's evidence and on the base branch's newest push artefact (`fetch_ci_evidence.sh`);
  - append `trust_diff.py` output and a marker diff to `$GITHUB_STEP_SUMMARY`. The marker diff covers tests added or removed and kind, roots or headline changed. Read markers with the AST, as `tests/test_trust_graph.py:_validates_objects` does; no imports needed.
  - **Never fail the job.** The ratchet that fails on reductions is tier B (B3).
- **Done when:** the PR's own job summary shows the (empty or small) diff.
- **Tests:** a unit test of the marker-diff script on two small synthetic trees.
- **Core files:** none (`scripts/marker_inventory.py` is new).
- **Page change:** none.
- **Size:** M.

### A15. AGENTS.md points findings at the ledger

- **Goal.** `AGENTS.md:38` still says "Record it in the README's findings table". The ledger (`trust/ledger.yml`) is the table, and `tests/test_docs_links.py::test_findings_page_is_the_ledger` enforces it.
- **Change:** replace the line with "Add an entry to `trust/ledger.yml`", plus the field list. Do not touch `docs/method/index.md` principle 7; that rewording is for Ben (C-list).
- **Tests:** `tests/test_docs_links.py`.
- **Core files:** none.
- **Page change:** none.
- **Size:** S.

### A16. Triage the 20 weekly failures into ledger entries

- **Goal.** AGENTS.md says a disagreement is kept as a test and recorded in `trust/ledger.yml`. The 20 failures of weekly run 41381202862 (virgil main 8ad1d11) include `test_inference_joint.py::test_regularized_inverse_and_its_warning` (1.47 against `< 1e-12`). That is a finding, not only something for A11 to display.
- **Change:** one entry per failure, or per shared cause (for example the `TSV.__init__() missing ... 'beam_mas'` group), in `trust/ledger.yml`:
  - `ruling: virgil` if virgil regressed;
  - `ruling: crosscheck` if the test code has to follow a virgil API change;
  - `status: open`.
- **Needs Ben.** Choosing the ruling needs a look at each failure (reproduce it first, per AGENTS.md). Rule only the clear cases. List borderline ones in the PR for Ben; do not guess.
- **Done when:** every one of the 20 tests is cited by an open ledger entry, and the ledger lint (A8a, if landed) passes.
- **Tests:** `tests/test_docs_links.py`.
- **Core files:** none.
- **Page change:** new open ledger rows. No verdict changes.
- **Size:** S.
