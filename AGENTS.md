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
`mkdocs.yml`. The home page, `docs/index.md`, is the Trust page and
`docs/overview.md` is generated from `README.md` (edit the README); both
are regenerated at deploy, so they are git-ignored.

Tests run virgil in float64 (`tests/conftest.py`) unless marked
`float32`. Heavy dLux runs use one process: do not run them in parallel
on a laptop.
