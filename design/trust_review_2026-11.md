# The Trust page: what "verified" means, and how to tighten it

*Review of virgil-validation's trust graph and provenance, revised after an independent critique.*
*Baseline: origin/main 98b7dda (#213 merged). Measured on the evidence the published page is built from.*

---

## One-page summary

### What "verified" means today

A part of virgil is "verified" when, among the newest record of each test that names it:
- enough tests labelled `check` passed;
- the roots of trust they claim add up to the number the graph asks for (usually one);
- nothing it depends on is unverified;
- no open virgil bug sits on it in the ledger.

The verdict never looks at the numbers: how close the agreement was, or what was asserted. Each test is taken at its newest run, whichever virgil commit that run used.

Today 124 virgil parts are verified. For **97 of them, every root is one written in this repository** (mathematics, standards, statistics). No other code confirms them. **63 rest on a single test function.**

### The five most important problems

1. **The page hides failures on virgil's main branch.** The weekly run (2026-11-02, virgil main 8ad1d11) had 20 failing tests: a `TSV` API change, and `regularized_inverse` off by 1.47 against a 1e-12 bound. The next push run, on the *older* pinned virgil, passed them. The page keeps the newest run, so all 20 failures vanished. `TSV`, `TV`, `Image`, `regularized_inverse` and `virgil_bridge.alma` show as verified.
2. **Tests can disappear without trace.** 84 module-level `virgil.*` import skips across 61 files skip themselves if a `virgil.*` module is missing, and a skip at that stage writes no record. If virgil renames a module, those tests vanish and CI stays green. `test_calibrated_load.py` has never run in CI.
3. **The ledger can clear a bug unseen.** A ledger status outside the five allowed words reads as "fixed", and nothing checks the status. Decisions made after seeing a result (D15, D22) are not shown on the cards they affect.
4. **Headline numbers are picked by the order of `record()` calls.** This is true for 105 records, and only 6 tests name their headline. The live rml-imaging card shows eht-imaging's own convergence (6e-5), not virgil's gradient, which is what the test checks.
5. **Provenance thins out beyond the Trust page.**
   - Only 1 of 170 showcase result files records a commit.
   - The run header has no platform, JAX precision or job id.
   - `campaigns.jsonl` is committed by hand.
   - In #213's scheme, the criteria change log sits inside the file it audits, so writing a row creates a new criteria commit.

### What is being fixed overnight (tier A)

These packages change no science number, threshold or criteria file. They make hidden states visible, add provenance, or add lints. Full briefs are in `design/trust_tierA_briefs.md`.

- **A1, A1b-d:** headlines are named explicitly, and a missing headline is an error. This fixes the rml-imaging card.
- **A2:** a strict expected-failure that starts passing marks its part failing, as `trust.py` already claims.
- **A3:** each of the 9 strict expected-failures states the exception it expects.
- **A4:** collection skips leave records, and CI fails if a `virgil.*` import is missing.
- **A5:** golden roots that are not listed become an error.
- **A6:** a pipeline whose checks all skipped shows "not run".
- **A7:** the run header records platform, JAX and job details, and edited evidence counts as dirty.
- **A8a, A8b:** a ledger lint, and badges for post-result decisions.
- **A9:** virgil's CI evidence carries an honest commit.
- **A10:** `test_calibrated_load` runs in CI.
- **A11:** a banner and badges show tests that fail on virgil main.
- **A12:** criteria commits must be ancestors of HEAD.
- **A13:** graph source paths must exist.
- **A14:** every PR gets a verdict diff in its job summary.
- **A15:** AGENTS.md points findings at the ledger.
- **A16:** the 20 weekly failures are triaged into open ledger entries (may need Ben for borderline cases).

Expected page changes:
- `alma-imlup` and `orbitize-posterior` read "not run";
- `virgil.detection.injection_recovery` goes from unchecked to partly (virgil's own CI tests it);
- the rml-imaging headline changes;
- new banners and badges appear.

No other verdict changes.

### What needs your decision

- **Which commit is the verdict for?** Should the page judge each part at the pinned virgil, at virgil main, or show both? The 20 masked failures make this live (B4).
- **Should roots written here count as independent?** 97 of 124 verified parts rest only on them. A shadow column would show the drop first (B2).
- **The change log inside the gated criteria file** cannot audit itself. Should it move beside the file (C7)? And is a byte-for-byte content comparison, with no hashing, acceptable for snapshot integrity (B9)?
- **Should open post-result decisions cap a pipeline at "passes"** (B5)?
- **Should a documentation-only virgil finding stop reading "Known bug"?** F23 is one: virgil is right to 1e-15 (B6).
- **Principle 7** ("People sign off the conventions") has no implementation since #213. Amend the principle or restore a route?
- **The orbit freeze reference commit** (C9), and whether `campaigns.jsonl` retires (B7).

---

## 1. How this was measured, and what changed since the draft

**Baseline.** File:line citations are at origin/main **98b7dda**, the #213 merge of 2026-11-08 21:58Z.

**Published evidence.** This is what `.github/workflows/docs.yml:49-51` feeds to `scripts/trust.py --strict-provenance`:
- push run 42467174675 (vv b70b370, virgil d79696d);
- weekly run 41381202862 (2026-11-02, virgil 8ad1d11, concluded `failure`);
- `trust/evidence/campaigns.jsonl`.

I ran 98b7dda's `trust.py` `build()` over exactly those files. The push run for 98b7dda itself was in progress at the time, so small shifts follow. The draft's use of the stale committed `latest.jsonl` is retired.

**Verification.** Every number below I re-measured myself (scripts in the scratchpad `v2/` folder). Claims I could not re-verify are dropped, not hedged.

**Corrections to the draft that I accept from the critique:**
- **Single-root nodes.** "About 70 rest on one mathematics test" mixed two conditions. The measured split:
  - 78 verified nodes have `mathematics` as their only strong root;
  - 63 rest on a single function;
  - 54 meet both;
  - 97 rest only on roots written here.
- **Showcase JSONs:** 1 of 170 records a commit (`orbit_aliases/gl229-cold/meta.json`), not 10.
- **Module-level skips:** 98, not 107. 84 of them import `virgil.*`. Counted by AST over `tests/*.py`: 98 sites in 67 files; the 84 `importorskip("virgil.…")` sites are in 61 files. `pytestmark = skipif` does leave records.
- **alma-mpol headline:** `test_alma_mpol.py:509`'s prose headline is latent (the test skips in CI). The live case is rml-imaging.
- **Strict xfails without `raises=`:** 9, not 10. `test_evidence_pin.py:93` is the plugin's self-test.
- **Finding tests:** there are no `kind="finding"` tests at all. The 10 strict xfails are `upstream`.
- **Resurrected campaigns:** only one campaign record is resurrected (`test_sbc_ranks_are_uniform`), and removing it changes no verdict. Old WP-T21 is dropped.
- **Ledger:** F23 and P11 are documentation findings with correct code, so neither *can* have a strict xfail.
- **`numpyro_model`** is "relies", not verified.
- **R10b/c and WP-T14/T15** reintroduce hashing, which Ben's rule forbids (`AGENTS.md:73`, `tests/test_gate_criteria.py:3`). They are dropped. B9 replaces them with a content comparison.
- **WP-T13 is circular as written.** The change log is inside the gated file (`design/radio_trust_criteria.yml:1-6`, whose row reads "this WP-2 commit"), so a row cannot name its own commit.

**Where I disagree with the critique** (one line each, measured):
- *"T2 cascades through `external_bridge.*`."* No. With the fix applied, a simulated XPASS moves only that node (pmoired or ehtim: partly → failing). The external nodes are already "partly", so they are never trusted references.
- *"The weekly run's evidence has no failed records."* It has 18 failed and 2 errors. All 20 are overwritten by the newer push run's passes at the older pin. That is problem 1 above.
- *"R4 (per-commit verdicts) has no effect today."* It does. Five verified nodes fail on virgil main (A11 and B4).
- *"The ledger's ruling is not validated."* The ruling vocabulary is validated (`tests/test_trust_graph.py:61-63`). Status is not. A `virgil` → `definition` edit is still in vocabulary and unchecked.
- *"12 test functions have no record."* True of the tree at b70b370. At 98b7dda it is 46, because #213 added tests that no published run has executed yet. The persistent class is `test_calibrated_load` (9), plus slow tests until the next weekly run (3 in `test_alma_lambda_rule`).

---

## 2. Findings

### Q1. What "verified" means

**1.1 The page shows a verdict at a mix of virgil commits, and hides failures on main (severe, new).**
- `load_evidence` (`scripts/trust.py:100-136`) keeps each test's newest record *by run date*. The fast suite runs on every push at the pinned `VIRGIL_REF` (`.github/workflows/tests.yml:13`, d79696d). The full suite runs weekly on virgil **main**.
- A weekly failure on main is overwritten by the next push's pass on the older pin.
- Measured: all 20 failing or erroring records of run 41381202862 are superseded this way. Examples: `TSV.__init__() missing ... 'beam_mas'` in `test_imaging_regularisers`, `test_l_curve` and `test_ehtim_reconstruction`, and `test_inference_joint.py::test_regularized_inverse_and_its_warning` (1.47 against < 1e-12). Under AGENTS.md this one is a finding, not a display problem: every weekly failure needs a `trust/ledger.yml` entry (`ruling: virgil` or `crosscheck`, `status: open`). That is A16, and it may need Ben for borderline cases.
- Verified nodes affected: `virgil.imaging.TSV`, `virgil.imaging.TV`, `virgil.models.Image`, `virgil.inference.regularized_inverse`, `virgil_bridge.alma`.
- The freshness dot hints that virgil changed, but nothing says a test fails there.
- Fixes: A11 (display), B4 (verdict).

**1.2 Verified is outcome-only (severe).**
- `own_status` (`trust.py:371-416`) counts passing records of kind `check` (`AGREEING`, line 49; `regression` records are counted but give only a weak root). It then compares the number of distinct strong roots with `node.roots` (default 1, line 399).
- Metrics and tolerances are never read. 189 of 647 passing checks record no number.
- I found no assertion-free check, so the risk is latent.
- An asserted-bound rule would be easy to game (`check("x", 0, "<", 1)`). C1 therefore keeps bounds for headlines only.

**1.3 Roots written here carry most verdicts (severe for the claim).**
- Of the 124 verified virgil nodes:
  - 97 rest only on mathematics, standards or statistics;
  - 78 rest on mathematics alone;
  - 63 rest on a single test function.
- The method doc lists these as roots (`docs/method/index.md:18-25`). It also asks for "two independent roots for anything a model could get wrong in a correlated way" (`:249-250`).
- The only alias is `ROOT_ALIAS = {"frap": "frank", "render": "mathematics"}` (`trust.py:145`), and `render` is a legacy rename.
- Conventions already need an external root (`unconfirmed_conventions`, `trust.py:422-425`).
- Whether the authored roots are independent of each other is a policy question (B2).

**1.4 Credit spreads across objects (moderate).**
- One marker credits every object it names with all its roots (`trust.py:353-356`). 46 check records name 3 or more non-pipeline objects.
- `tests/test_oidb_fits.py:178-179` credits five objects from one orbit recovery.
- The critique is right that the draft's objection to its `standards` root was wrong: the test writes and reads OIFITS. The over-credit of objects stands. See B1.

**1.5 A strict XPASS does not mark failing (moderate).**
- `trust.py:381` excludes `finding` and `upstream` from `failing` whatever the outcome. The docstring (`:20`) says otherwise.
- Today there is no XPASS. A simulated one leaves `pmoired` "partly" instead of "failing". Fix: A2.

**1.6 Strict xfails accept any exception (moderate).**
- 9 of the 10 strict xfails have no `raises=`:
  - `test_candid.py:312`;
  - `test_ehtim_reconstruction.py:369`;
  - `test_fouriever.py:189`;
  - `test_orbitize.py:484`;
  - `test_pmoired_problems.py:36,50,67`;
  - `test_polar_disk_vs_codes.py:123,162`.
- The plugin requires only `strict=True` (`src/evidence/plugin.py:140-152`). Fix: A3.

**1.7 Headlines are chosen by order (moderate; live).**
- `_headline` (`trust.py:195-210`) falls back to the first `AGREEMENT`-matching metric. It does so even when an explicit headline is set but missing.
- 105 records in 66 functions have more than one candidate; only 6 records set `headline=`.
- Live: `pipeline:rml-imaging` shows `ehtim_rel_log_gradient`. The test asserts on `rel_gradient_at_ehtim_minimum` (`tests/test_ehtim_reconstruction.py:289-309`).
- `_worst` (`:186-192`) takes the largest magnitude, which is wrong for larger-is-better metrics.
- Fixes: A1, A1b-d.

**1.8 Hand-written edges (minor).**
- `source` and `depends` in `trust/graph.yml` are not checked against virgil. A13 checks that the paths exist; C6 derives them.
- `problems()` (`trust.py:462-472`) reads dependencies' *own* status, so cycles cannot support themselves.

**1.9 Guards claim strong roots (minor, no verdict effect).**
- 112 `kind="guard"` markers claim statistics (45), standards (41), mathematics (20) or literature (6).
- Guards never count toward a verdict. Retagging them (old T9) touches about 60 files for no verdict change, so it is dropped. A14's marker diff shows any guard → check change.

**1.10 Golden roots (minor).** `golden:visread` (`tests/test_alma_conventions.py:137`) is not in `trust/golden.yml`, so `roots_of` (`trust.py:148-163`) silently ignores it. Fix: A5.

### Q2. Provenance

**2.1 Missing evidence is invisible (severe).**
- Module-level `importorskip` and `skip(allow_module_level=True)` occur at 98 sites in 67 files. They leave no record, because the plugin writes only in `pytest_runtest_logreport` (`plugin.py:198-233`).
- 84 of them (across 61 files) import `virgil.*`, so a removed or renamed virgil module turns those 61 files into silent skips.
- `test_calibrated_load.py:13` (9 guards) has never run in CI.
- Fixes: A4, A10.

**2.2 `campaigns.jsonl` is the one hand-committed input to the page (moderate).**
- `docs.yml:51` reads it. Its header says `runner: local`, `run_url: null`.
- PR review sees edits, but nothing checks it mechanically against `trust/campaigns/*.json`.
- Its one resurrected record (`test_sbc_ranks_are_uniform`, virgil 8c51dc5) comes from a summary with no `criteria_commit`. By `AGENTS.md:74-75` such a summary "is not gate evidence". Removing the record changes no verdict.
- Fix: B7.

**2.3 virgil's CI evidence is dropped (minor).**
- `scripts/ingest_virgil_evidence.py:58` writes `validation_commit: None`, and `--strict-provenance` leaves the file out (`trust.py:89-91`).
- Including it changes one verdict: `virgil.detection.injection_recovery` goes from unchecked to partly.
- Fix: A9 (record the real commit; no exemption).

**2.4 Thin run header (moderate).**
- `src/evidence/meta.py:89-110` has no platform, JAX backend, x64 state, thread caps, SLURM ids or pytest selection.
- `runner` is read from `GITHUB_ACTIONS` (`:103`).
- `validation_dirty` ignores all of `trust/evidence/` (`:98`), so an edited `campaigns.jsonl` reads as clean.
- Fixes: A7 and A4(c).

**2.5 Showcase numbers are untraceable (moderate).** 1 of 170 JSON files under `docs/` carries a commit. 149 are post-AGB imaging, so one helper covers most of them (C8).

**2.6 Committed `latest.jsonl` (minor).** The page does not read it; the release pin (`scripts/check_evidence_pin.py`) does. Have the release check read CI artefacts, as `docs.yml` does, and stop committing the file (C3).

### Q2b. #213 (merged): pre-registration by git commit

#213 is a real simplification:
- one rule;
- dirty and untracked files refused;
- aggregates refuse mixed commits (`src/crosscheck/preregistration.py:30-56`);
- no signer layer.

What remains:

1. **The change log sits inside the gated file**, as `AGENTS.md:70-71` requires.
   - Adding a row, or backfilling a hash, makes a new `criteria_commit`. Under "changed criteria mean a rerun" (`preregistration.py:5`), that invalidates committed runs.
   - A row cannot name its own commit; today's rows say "this WP-2 commit".
   - So post-hoc auditing against `git log` is self-defeating as designed. Moving the log beside the file needs Ben (C7).
2. **The `CRITERIA_COMMIT` environment variable is trusted on `git archive` snapshots** (`preregistration.py:36-40`). No hashing is allowed. A content check works instead: the task copies the criteria file's text into its output, and the Mac-side aggregate compares it with `git show <commit>:<path>` (B9).
3. **The scoring code is not pinned.** Pinning it needs either hashing or `git diff` (unavailable on snapshots). Ben's rerun rule already says what to do when it changes, so there is no WP.
4. **A criteria commit should be an ancestor of HEAD.** A12 adds the check.
5. **Principle 7** ("People sign off the conventions", `docs/method/index.md:61`) has no implementation since `signoffs.yml` was removed. This is Ben's question.

### Q3. Gaming and decay

| Move | Detected today? | After tier A |
|---|---|---|
| A virgil module renamed, so 61 files (84 import skips) silently skip | No | A4: CI error |
| A weekly failure on main overwritten by a pass at the pin | No | A11: banner (B4 for the verdict) |
| Ledger status typo, or a valid but false status change | No | A8a catches typos; A14 shows the change in the PR |
| `virgil` → `definition` ruling edit | No | A14 shows it; B3 can fail it |
| Retag a check as `upstream` with a strict xfail | No | A8a: an upstream test must be cited by an unfixed external entry |
| Strict xfail failing for a new reason | No | A3 |
| Headline typo, or metric reorder | No | A1, A1b-d |
| Loosen a tolerance | Review only | A14 shows the verdict diff; tolerances stay a code diff |
| Delete a test | No | A14 lists it; B3 can fail it |
| Lower `roots:` in graph.yml | Radio nodes only (`scripts/check_radio_trust_criteria.py`) | A14 shows it |
| Edit `campaigns.jsonl` | PR review only | A7 makes a local run dirty; B7 retires the file |
| Criteria edit, commit, rerun | Change-log row only (self-declared) | Unchanged; C7 |
| Edit a frozen orbit script | No | C9 |
| Dirty tree, non-strict xfail, control as headline | Yes | Yes |

---

## 3. Packages by tier

### Tier A: overnight (briefs in `design/trust_tierA_briefs.md`)

| Id | What | Old id | Size |
|---|---|---|---|
| A1 | Explicit headline enforced; rml-imaging and alma-mpol markers fixed; order warning | T1 + T8 core | S |
| A1b-d | `headline=` on 66 functions, in three file groups | T8 | M each |
| A2 | Strict XPASS marks the node failing | T2 | S |
| A3 | `raises=` on 9 strict xfails, required by the plugin | T4 | S |
| A4 | Collection skips recorded; selection in the header; missing `virgil.*` import fails CI | T5 + critique | M |
| A5 | Unlisted golden roots are an error; `visread` listed | W-d | S |
| A6 | All-skipped pipelines show "not run" | T19 | S |
| A7 | Richer header; `validation_dirty` sees edited evidence | T7 | S |
| A8a | Ledger lint: status vocabulary, upstream ↔ ledger, `pending:` for empty evidence | T3 + T11 core + critique | S-M |
| A8b | `post_result`/`affects` fields; badges (no cap) | T11 + T12 badge | M |
| A9 | Ingest records the real vv commit | T6 (ingest route) | S |
| A10 | `virgil-vlti` pinned in CI | T18 | S |
| A11 | Badge and banner for tests failing on virgil main | new | S-M |
| A12 | `criteria_commit` is an ancestor of HEAD | T13 (half) | S |
| A13 | Graph `source` paths exist at the pin | R9 step 1 | S |
| A14 | Verdict and marker diff in each PR's job summary (never fails) | T16 (display) | M |
| A15 | AGENTS.md:38 points at the ledger | W-c (half) | S |
| A16 | Triage the 20 weekly failures into ledger entries | AGENTS.md ledger rule | S (Ben for borderline) |

### Tier B: changes what a verdict means (draft PR, marked needs-Ben)

- **B1. Trim over-credited multi-object markers (old T20).**
  - Scope: 46 records. Each remaining object needs a docstring line saying why it is directly checked.
  - Verdicts may drop. The PR carries a `trust_diff` table and needs the second reviewer that `AGENTS.md:77-84` requires.
  - Size M.
- **B2. A shadow column, "verified (independent)" (R3).**
  - Roots get classes (authored-here, external code, external result); `roots: 2` then needs two classes.
  - Expected: up to 97 of 124 drop in the shadow column.
  - Cheaper variant: apply the rule only to `roots: 2` nodes and properties.
  - Size M.
- **B3. A ratchet on A14.**
  - The PR job fails when a test disappears, a kind or roots weaken, or `roots:` drops, unless `trust/decisions.yml` names the change.
  - Size S on top of A14.
- **B4. Verdict at virgil main.**
  - A node with a failing record on main in the newest weekly run reads "failing on main" (or "verified at pin, failing on main").
  - Today that applies to the five nodes in 1.1.
  - Size S-M on top of A11.
- **B5. Cap on post-result decisions.** An open decision with `post_result: true` caps the affected pipeline at "passes". The categories are marked "for Ben to check" (`trust/ledger.yml:570`). Size S on top of A8b.
- **B6. Documentation findings.**
  - A `virgil` entry whose `kind` is documentation, with a passing pinned test, reads "documented", not "Known bug".
  - This changes `virgil.fitting.gauss_newton_diag` (F23).
  - Size S.
- **B7. Campaign tests stamp the summary's `virgil_commit` instead of skipping (R5b-lite), and `campaigns.jsonl` is deleted.**
  - The SBC summary has no `criteria_commit`, so it is not gate evidence (`AGENTS.md:74-75`). It must be rerun or dropped from the page.
  - Measured verdict change: none.
  - Size M.
- **B8. Mutation rerun, then strengthen regression pins.**
  - The untracked report (virgil 0c1be69, 2026-10-18) lists regression mutants M108, M109 (`virgil.oifits`) and M111 (`virgil.imaging.clean`) as surviving.
  - Rerun at `VIRGIL_REF` first, on OzSTAR (Ben submits); about 21 minutes per baseline.
  - Then fix only what still survives.
- **B9. Snapshot integrity without hashing.**
  - Tasks copy the criteria file text into their output, and the aggregate compares it with `git show <criteria_commit>:<path>`.
  - It touches #213's scheme, hence needs-Ben.
  - Size S.

### Tier C: architecture and policy (recommendations only)

- **C1. Asserted bounds (R1).** An optional `evidence.check(name, value, op, bound)` helper, used for headlines and shown next to the value. Not a verdict rule, because it is easy to game.
- **C2. Per-commit verdicts (R4).** Superseded in practice by A11 and B4. A full per-commit model conflicts with slow tests running only weekly on main.
- **C3. Stop committing `latest.jsonl`.** `check_evidence_pin.py --release` reads CI artefacts. Attestation (R5a/d) then has little left to protect.
- **C4. Per-record `virgil_commit` and `precision` (R6, reduced).** The rest of schema v2 (source hashes, data hashes) costs L and runs into the no-hashing rule.
- **C5. A full decisions ledger (R7)** beyond A8: `decided_by`, `reproduced_by`, `second_review` (`AGENTS.md:83-84` asks for these), and "fixed" citing a passing test of the fix.
- **C6. Derive `source` and `depends` from coverage (R9).**
- **C7. Move each criteria change log beside its file** (`design/<gate>.changelog.md`, not gated), then test it against `git log`. This changes #213's rule.
- **C8. A showcase provenance block** (`evidence.provenance()` in `postagb_imaging.py` and the other writers). The baseline is 1 of 170.
- **C9. A freeze guard (R12).** A test pins the frozen orbit files at a commit Ben names. Content comparison is enough; no hashing.

**Dropped:**
- T9: guard retag. No verdict effect, about 60 files to edit; A14 covers kind changes.
- T10: transitive `via`. No demonstrated verdict effect.
- T13 as written: circular.
- T14, T15, R10b/c: hashing.
- T21: no verdict effect, measured.
- T6 via a `provenance_problem` exemption: the exemption would be spoofable; A9 replaces it.

---

## 4. Open questions, and which the repository already answers

| # | Question | Answer |
|---|---|---|
| 1 | Must "verified" include a root from outside this repository? | **Partly decided.** Conventions already need one (`trust.py:422-425`), and the method doc asks for independent roots where errors could correlate (`docs/method/index.md:249-250`). Whether mathematics, standards and statistics are independent of each other is **Ben's** call (B2) |
| 2 | Should asserted bounds be required? | **Ben.** Recommendation: no, as a verdict rule (C1) |
| 3 | Should resurrected campaign passes count? | **Decided.** `trust.py:132` keeps them deliberately, with a banner. But `AGENTS.md:74-75` says a summary without `criteria_commit` "is not gate evidence", which covers the one resurrected record (SBC). No verdict changes (B7) |
| 4 | Do post-result decisions cap pipelines? | **Ben.** `trust/ledger.yml:570`: the categories are "for Ben to check" (B5) |
| 5 | Conventions without sign-offs? | **Ben.** Principle 7 (`docs/method/index.md:61`) is unimplemented since #213: amend it or restore a route |
| 6 | Rerun or excuse when gate code changes? | **Decided: rerun** (`AGENTS.md:73`, `preregistration.py:5`), which is also why T15 is dropped |
| 7 | Do the orbit freeze edits stand? | **Partly decided.** The two `orbit_gallery.py` hooks are allowed (`AGENTS.md:116-117`); anything else, and the reference commit for C9, is **Ben's** |
| 8 | Retire `campaigns.jsonl`? | **Ben** (B7 makes it cheap) |
| 9 | Sigstore attestations? | **Ben.** Low value once `latest.jsonl` is not committed (C3) |
| 10 | Mutation results on node cards? | **Not yet.** The data is stale; `AGENTS.md:77-80` says reproduce first (B8) |
| 11 | *New:* which commit is a verdict for, the pin or main? | **Ben** (B4) |
| 12 | *New:* move the change log out of the gated file? | **Ben** (C7) |
| 13 | *New:* is a content comparison (no hash) acceptable for snapshot integrity? | **Ben** (B9) |

---

## 5. Note on this review's own provenance

Under `AGENTS.md:77-84`, this review is a set of claims. The draft's figures came from a stale file, and several were wrong. They are corrected in §1.

Every number in this version was re-measured on the published evidence at 98b7dda. The scripts are in the scratchpad folder `v2/` (`pub.py`, `skips.py`, `res.py`, `t2.py`, `misc.py`, `amb.py`, `g.py`).

Any package that flips a verdict (A2 in future, A6, A9, and all of tier B) names the change in its PR. Tier B needs the second reviewer that AGENTS requires.
