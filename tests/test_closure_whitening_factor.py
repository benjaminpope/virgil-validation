"""Why mutant M64 (whitening with L^T instead of L) is equivalent, kept as a guard.

virgil whitens the closure phases of a frame with the Cholesky factor L of
Q R Q^T, where the rows of Q are the left singular vectors of the triangle-by-
baseline matrix T. Q R Q^T = diag(s^2 / 3) is diagonal, so L is diagonal and
``solve_triangular(L, a)`` equals ``solve_triangular(L, a, trans=1)``: swapping them
changes nothing. If a later virgil changes the basis so that L is not diagonal, the
mutant stops being equivalent and this test fails to say so (mutants/mutants.yml, M64).
"""

import itertools

import numpy as np
import pytest

closure = pytest.importorskip("virgil._closure")


@pytest.mark.parametrize("n_tel", [4, 5, 6])
@pytest.mark.validates("virgil.oidata.OIData.cp_noise", roots=["mathematics"], kind="guard")
def test_the_closure_cholesky_factor_is_diagonal(n_tel):
    baselines = {pair: i for i, pair in enumerate(itertools.combinations(range(n_tel), 2))}
    triangles = list(itertools.combinations(range(n_tel), 3))
    i1 = [baselines[(a, b)] for a, b, c in triangles]
    i2 = [baselines[(b, c)] for a, b, c in triangles]
    i3 = [baselines[(a, c)] for a, b, c in triangles]
    noise = closure.ClosureNoise.from_indices(i1, i2, i3)
    for chol in np.asarray(noise.chol):
        assert np.max(np.abs(chol - np.diag(np.diag(chol)))) < 1e-12
