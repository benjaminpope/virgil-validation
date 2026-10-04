"""Evidence records: what each test validates, against which roots of
trust, on exactly which code (docs/design.md, "Making it traceable").

Tests declare their claim with a marker::

    @pytest.mark.validates("virgil.models.UniformDisk", roots=["mathematics"])

and may record numbers with the ``metric`` fixture. Run pytest with
``--evidence PATH`` to write one JSON line per test, after a header line
describing the run (virgil commit, package versions, runner).
"""

# The pool of roots: any link in the trust graph may be validated against
# whichever of these suits it; no stage needs every package.
ROOTS = (
    "mathematics",  # closed forms, checked numerically with SciPy/NumPy
    "standards",  # OIFITS, Thompson-Moran-Swenson uv geometry
    "dlux",
    "pmoired",
    "candid",
    "fouriever",
    "ehtim",  # eht-imaging (Chael et al.): imaging and its regularisers
    "mpol",  # MPoL (Czekala et al.): RML imaging, gridding, Fourier transforms
    "literature",  # a published result, reproduced from the authors' data
    "statistics",  # ensembles with a known distribution
    "self-consistency",  # virgil against virgil: the weakest
)
# also allowed: "golden:<source>", reference values from someone's code

KINDS = (
    "check",  # virgil must agree with the roots
    "control",  # a deliberately wrong mapping must fail
    "finding",  # strict xfail: a known virgil problem
    "upstream",  # strict xfail: a known problem in an external package
    "reference",  # checks our own reference code (crosscheck, bridges)
    "guard",  # repository rules, e.g. independence
)
