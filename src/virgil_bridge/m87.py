"""virgil's side of the M87* replication (design/m87_replication.md): ring features with
``virgil.metrics.ring_features`` (virgil#715), in the units the crosscheck side uses (μas, degrees),
and the provenance of the installed virgil."""

import importlib.metadata
import json


def has_ring_features():
    try:
        from virgil.metrics import ring_features  # noqa: F401
    except ImportError:
        return False
    return True


def ring_features(image, pixel_scale_uas):
    """virgil's ring features of an array (row 0 North, column 0 East) with pixel ``pixel_scale_uas``,
    as a dict like ``crosscheck.ring_extraction.ring_features``: lengths in μas, angles in degrees."""
    from virgil.metrics import ring_features as rf

    f = rf(image, pixel_scale_uas / 1000.0)
    uas = lambda v: 1000.0 * float(v)
    return {
        "centre": tuple(uas(c) for c in f.centre),
        "diameter": uas(f.diameter), "diameter_std": uas(f.diameter_std),
        "width": uas(f.width), "width_std": uas(f.width_std),
        "pa": float(f.pa) % 360, "pa_std": float(f.pa_std),
        "asymmetry": float(f.asymmetry), "asymmetry_std": float(f.asymmetry_std),
        "central_fraction": float(f.central_fraction),
    }


def provenance():
    """``virgil_commit`` and ``virgil_dirty`` of the installed virgil-astro: the commit of a git
    install (pip's direct_url.json), dirty when installed editable or from a release (unknown tree)."""
    dist = importlib.metadata.distribution("virgil-astro")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    vcs = direct.get("vcs_info", {})
    editable = direct.get("dir_info", {}).get("editable", False)
    commit = vcs.get("commit_id")
    return {"virgil_version": dist.version, "virgil_commit": commit, "virgil_dirty": bool(editable or not commit)}
