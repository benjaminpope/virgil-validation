"""pytest plugin for scripts/mutation.py (loaded with ``-p mutation_order``).

* Runs first the tests that validate the mutant's trust-graph nodes, then those
  validating objects that depend on them, then the rest (``MUTATION_RANK`` is a
  JSON object ``{node: rank}``), so ``-x`` stops at the first kill early.
* Deselects tests of kind ``finding`` and ``upstream``: those are strict xfails
  that fail the moment virgil's behaviour changes at all, so a "kill" by one
  says nothing about whether the suite checks the property that was broken
  (``MUTATION_SKIP_KINDS``, comma separated).
"""

import json
import os


def _claims(item):
    for mark in item.iter_markers("validates"):
        yield list(mark.args), mark.kwargs.get("kind", "check")


def pytest_collection_modifyitems(config, items):
    skip = {k for k in os.environ.get("MUTATION_SKIP_KINDS", "finding,upstream").split(",") if k}
    rank = json.loads(os.environ.get("MUTATION_RANK", "{}"))
    keep, dropped = [], []
    for item in items:
        claims = list(_claims(item))
        if claims and all(kind in skip for _, kind in claims):
            dropped.append(item)
        else:
            keep.append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)

    def key(item):
        ranks = [rank[o] for objs, kind in _claims(item) if kind not in skip for o in objs if o in rank]
        return min(ranks) if ranks else 99

    # sorted() is stable: the original order is kept within a rank
    ordered = sorted(keep, key=key)
    # MUTATION_MAX_RANK: run only the tests of the mutant's nodes (0) or of those and their
    # dependents (1): a quick lower bound on survival when the whole suite is too slow.
    max_rank = os.environ.get("MUTATION_MAX_RANK")
    if max_rank is not None:
        cut = [i for i in ordered if key(i) > int(max_rank)]
        ordered = [i for i in ordered if key(i) <= int(max_rank)]
        config.hook.pytest_deselected(items=cut)
    items[:] = ordered
