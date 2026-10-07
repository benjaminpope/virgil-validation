"""Private (L2) OiDB collections: their ids are kept out of the public repository.

The tracked tree names such a collection by a placeholder, ``private-<name>`` (for
example ``private-workshop``). The real OiDB id is needed only to talk to OiDB or to
find files on disk, and is read from, in order:

1. the environment variable ``OIDB_PRIVATE_COLLECTIONS``: comma-separated
   ``<name>=<collection id>`` pairs, e.g. ``workshop=<id>``;
2. the git-ignored local file ``~/.config/virgil-validation/private_collections.txt``
   (same ``name=id`` pairs, one per line; ``#`` starts a comment).

When neither is set the private tasks are skipped with a message, not an error.
"""

import os
import sys
from pathlib import Path

ENV = "OIDB_PRIVATE_COLLECTIONS"
FILE = Path.home() / ".config" / "virgil-validation" / "private_collections.txt"
PREFIX = "private-"


def private_ids():
    """``{"private-<name>": real id}`` from the environment or the local file ({} if unset)."""
    text = os.environ.get(ENV)
    if not text and FILE.exists():
        text = FILE.read_text()
    out = {}
    for item in (text or "").replace(",", "\n").splitlines():
        item = item.split("#", 1)[0].strip()
        if item:
            name, _, cid = item.partition("=")
            if not cid.strip():
                raise SystemExit(f"{ENV}/{FILE}: expected name=id, got {item!r}")
            out[PREFIX + name.strip()] = cid.strip()
    return out


def key_of(collection):
    """The tracked-tree key (``private-<name>``) for a collection id; other ids pass through."""
    for key, cid in private_ids().items():
        if collection == cid:
            return key
    return collection


def real_id(key):
    """The OiDB id of a collection key, or ``None`` for a private one that is not configured."""
    if not key.startswith(PREFIX):
        return key
    return private_ids().get(key)


def skip_message(key):
    return (f"{key}: skipped, the private collection id is not set (set {ENV}=<name>=<id> or write it to {FILE})")


def warn_skip(key):
    print(skip_message(key), file=sys.stderr)
