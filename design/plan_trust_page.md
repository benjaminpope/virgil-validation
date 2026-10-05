# Plan: the Trust page

The site's home page shows what has been checked, against what, and what
the checks found, for a reader who has not followed the work.

## Decisions

- **Verdict and freshness are separate.** The verdict comes from the
  evidence at the pinned virgil commit. Whether virgil has changed since is
  a quiet note (a dot on the part, a count in the headline), never a colour.
  Otherwise every merge in virgil would turn most of the page "stale".
- **Plain words for the verdicts.** They are: verified; works, but relies
  on a known bug; known bug, fix pending; partly checked; check failing;
  not yet checked. A part that relies on a known bug names it, so the
  reader sees one specific issue rather than a general doubt.
- **Layers, not a node-link graph.** There are 73 parts and only 30
  dependencies, so a hairball of edges says little. Parts are chips in
  bands from data up to imaging. Selecting one highlights what it relies
  on and what relies on it.
- **Real data first.** Reproducing published results is what a scientist
  will look for, so these cards come straight after the headline. The
  synthetic end-to-end chains follow, as rows of steps.
- **Readable without JavaScript.** The page is static HTML with inline SVG,
  and every chip has a full text tooltip. A small script adds the detail
  panel and the highlighting. Colour is never the only signal: every
  status also has an icon and words.

## Pieces

- `scripts/trust.py` builds the model (verdicts, reasons traced to the
  findings at the bottom of each chain, per-check headline numbers, test
  links) and renders `docs/trust.md` and the home page, `docs/index.md`.
  It writes `docs/assets/trust.json` for other tools.
- The evidence plugin records each test's line, the first paragraph of its
  docstring and an optional `headline=` metric.
- `trust/graph.yml` pipelines carry `data` (real or simulated), `dataset`,
  `reference`, `url`, `headline` and, while there is no evidence yet,
  `state` (planned or running).
- `docs/assets/trust.css` and `trust.js` style the page and add its
  interaction.
- The docs workflow regenerates the page at deploy, from the committed
  evidence and with a virgil checkout for freshness, then builds the site
  with Zensical, as virgil does.

## Later

- Cards for each real-data result, filled in as their evidence arrives
  (orbitize! β Pic, published binaries, imaging contests).
- A history strip: the verified count per week, from the weekly runs.
