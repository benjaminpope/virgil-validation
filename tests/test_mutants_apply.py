"""The mutation list (mutants/mutants.yml) applies cleanly to the pinned virgil commit.

The harness (scripts/mutation.py) measures how sensitive the suite is by patching
virgil in a scratch copy. A snippet that no longer matches (virgil moved, or the
pin changed) would silently turn a mutant into a no-op, so each is applied here in
memory against ``git show <pin>:<file>``: fast, no virgil import.
"""

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("mutation", ROOT / "scripts" / "mutation.py")
mutation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mutation)

PIN, MUTANTS = mutation.load_mutants()
GRAPH = mutation.load_graph()

# Slips the brief asks the list to cover: every class must have a mutant.
REQUIRED_CLASSES = {
    "pa_sign", "pa_origin", "uv_swap", "mas_rad", "deg_rad", "wavelength_axis", "wavelength_off_by_one",
    "flag_ignored", "flag_inverted", "closure_order", "closure_sign", "whitening_transpose",
    "covariance_inverse", "mjd_time_grouping", "v_vs_v2", "sqrt_missing", "bessel_factor",
    "flux_contrast", "limb_darkening", "ft_conj_sign",
}


def _need_virgil_history():
    repo = mutation.virgil_repo()
    if not mutation.have_commit(repo, PIN):
        pytest.skip(f"virgil commit {PIN[:10]} not in {repo} (set VIRGIL_REPO to a virgil git checkout)")
    return repo


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_mutant_list_is_well_formed():
    problems = mutation.validate_list(MUTANTS, GRAPH)
    assert not problems, problems
    assert len(MUTANTS) >= 50
    missing = REQUIRED_CLASSES - {m["cls"] for m in MUTANTS}
    assert not missing, sorted(missing)


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_every_mutant_applies_cleanly_to_the_pinned_commit():
    repo = _need_virgil_history()
    cache = {}

    def read(file):
        if file not in cache:
            cache[file] = mutation.git_show(repo, PIN, file)
        return cache[file]

    bad = []
    for m in MUTANTS:
        try:
            patched = mutation.apply_mutant(m, read)
        except mutation.MutationError as err:
            bad.append(str(err))
            continue
        for file, text in patched.items():
            assert text != read(file), f"{m['id']} changes nothing in {file}"
    assert not bad, "\n".join(bad)


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_a_stale_snippet_fails_loudly():
    """The patcher refuses a snippet that is missing, ambiguous or unchanged."""
    text = "a = 1\nb = a + a\n"
    mutant = {"id": "T", "edits": [{"file": "x.py", "original": "a = 1", "replacement": "a = 2", "count": 1}]}
    assert mutation.apply_mutant(mutant, lambda f: text)["x.py"] == "a = 2\nb = a + a\n"
    for original, replacement, count in [("zzz", "q", 1), ("a", "q", 1), ("a = 1", "a = 1", 1), ("a = 1", "a = ((", 1)]:
        bad = {"id": "T", "edits": [{"file": "x.py", "original": original, "replacement": replacement, "count": count}]}
        with pytest.raises(mutation.MutationError):
            mutation.apply_mutant(bad, lambda f: text)


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_a_mutant_is_found_before_the_whole_suite_runs():
    """Tests of the mutant's own nodes rank first, then those that depend on them."""
    ranks = mutation.node_ranks({"nodes": ["virgil.models.Image"]}, GRAPH)
    assert ranks["virgil.models.Image"] == 0
    assert ranks["virgil.models.Image.model_on_grid"] == 1  # depends on Image
