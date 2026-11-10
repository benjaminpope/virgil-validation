# Trust graph and provenance review: strategy (2026-11-08)

This note explains what was done overnight, and why, so that you can review it in the morning.

## Goal

Make the trust graph and the provenance easier to rely on and easier to read. When a node says "verified", that should mean something a reader can check. Every published number should trace back to its code, data, criteria and run. Hidden states (skips, expected failures, dropped evidence, headlines picked by accident) should be visible.

## How the review is produced

1. **Draft (Opus, high effort): done.** An Opus agent reviewed `scripts/trust.py`, `src/evidence/plugin.py`, `trust/ledger.yml`, the trust tests, the pull campaign, PR #213 and issues #204 and #207. It produced findings with file:line references, 12 architectural recommendations, 21 work packages (WP-T1 to WP-T21) and 10 open questions for you.
2. **Independent check (a second Opus agent, with no stake in the draft).** It re-verifies every count and factual claim against the code and marks each confirmed, wrong or unverified. It also asks whether each package is the right size, whether its "done" can be tested, whether it could be gamed, and what the draft missed.
3. **Revision.** The drafting agent revises against the critique and moves its line references to current main, which now includes #213. Claims the critique could not confirm are dropped, not hedged.
4. **The result lands in the repo** as `design/trust_review_2026-11.md` through a PR: a one-page summary, then the detail. That PR is the document for you to review.

## What I fix overnight and what I leave for you

Each work package goes into one of three tiers.

- **Tier A, uncontroversial (I start Sonnet PR workers now).** A package is tier A only if all of these hold:
  - it changes no science number, threshold or criteria file;
  - it can only make hidden things visible, add missing provenance, or add a lint that fails on a real inconsistency;
  - it is small and easy to revert.

  Typical examples: require `raises=` on expected failures, lint that a headline names a recorded metric, record collection-level skips, a richer run header, explicit headlines where there are several candidates, and showing skipped pipelines as "not run".
- **Tier B, changes what a verdict means.** Examples: an unexpectedly passing "finding" marks its node failing; post-result ledger decisions cap verdicts; campaign passes that #213 now rejects stop showing "verified". These get draft PRs marked "(needs Ben)" with a plain statement of which nodes change state. I will not ask anyone to merge them before you have read them.
- **Tier C, architecture and policy.** Examples: typed measured-versus-asserted evidence records, independence classes for roots, attested CI-only evidence replacing `campaigns.jsonl`, the freeze ruling, and whether "verified" needs a root from outside this repo. These stay as recommendations in the document, with costs and a migration path, for you to decide.

The tier of each package is decided after the critique, not from the draft alone.

## Rules the workers follow

- One package per PR, each from current main in its own worktree. Packages that touch the same core files (`scripts/trust.py`, `src/evidence/plugin.py`) run one after another, never in parallel.
- At most about four workers at once. Every python or pytest call goes through `pyslot`, and only targeted tests run locally; CI does the full run.
- Each PR gets an Argon review. Nobody merges: you merge.
- **Not as planned:** the app refuses to start new sessions from a turn that a background event triggered, so the workers are Sonnet subagents in their own worktrees, not sessions. Subagents can't have auto-fix. I pass each Argon review back to the subagent that wrote the PR myself.
- Each PR body says in plain words what changes on the Trust page, and lists the nodes whose state changes (for tier A this should be "none").

## Avoiding drift

- There is one source of truth: the revised review. Every worker brief names its WP id and quotes that package's "done" condition.
- The coordinating session stays thin. It delegates, reads each worker's single report, and checks the diff before telling you something is done.
