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
- `src/external_bridge/` calls other people's packages (PMOIRED, later
  CANDID). It may not import virgil either. Pin their conventions from
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
| Regenerate the report and figures | `.venv/bin/python scripts/report.py` |

Tests run virgil in float64 (`tests/conftest.py`) unless marked
`float32`. Heavy dLux runs use one process: do not run them in parallel
on a laptop.
