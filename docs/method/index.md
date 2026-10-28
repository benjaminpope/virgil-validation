# How trust is built

virgil is largely written by language models (Claude and Copilot), and so are
most of its tests. Those tests mostly check virgil against other parts of
virgil. That shows the code is self-consistent, not that it is right. A sign
convention, a factor of two or a misread paper can be consistent everywhere
and still wrong.

This repository **bootstraps trust**. We start from a few things we need not
take on faith. We then extend trust upwards one checked step at a time, until
every capability of virgil hangs from that foundation by a chain of
independent checks. The sections below set out the rules (principles), what
the checks can and cannot show, how the evidence is recorded, and how the
[Trust page](../index.md) turns it into verdicts.

## Principles

1. **Roots of trust are never LLM-written.** A root is one of:
    * **mathematics**: closed-form results (Fourier transforms of
      Gaussians and disks, the shift and convolution theorems, the Airy
      pattern), checked numerically with mature, human-written libraries
      (NumPy, SciPy's Cephes Bessel functions);
    * **standards**: published definitions (OIFITS v1 and v2, position
      angle North through East, Earth-rotation synthesis in Thompson,
      Moran & Swenson);
    * **mature external packages** written and used by people:
      [dLux](https://github.com/LouisDesdoigts/dLux),
      [PMOIRED](https://github.com/amerand/PMOIRED),
      [CANDID](https://github.com/amerand/CANDID),
      [fouriever](https://github.com/kammerje/fouriever) (Jens Kammerer),
      and for image reconstruction
      [eht-imaging](https://github.com/achael/eht-imaging) and
      [MPoL](https://github.com/MPoL-dev/MPoL), and for Keplerian orbits and radial velocities
      [orbitize!](https://github.com/sblunt/orbitize) (Blunt et al. 2020,
      2024);
    * **literature**: published results, reproduced from the authors' own
      data;
    * **statistics**: ensembles of simulations whose distribution is known
      (pulls of fit − truth over σ must be N(0, 1)).
2. **Our own reference code is a link, not a root.** `crosscheck` is also
   LLM-written. Before it may judge virgil, each reference must agree with a
   root by a second, independent route (for example, quadrature against a
   closed form). It must never import or copy virgil.
3. **Trust flows upwards only.** A virgil capability is trusted when it is
   checked against something already trusted *and* everything it is built
   on is trusted. A fit is only as good as the likelihood, which is only as
   good as the model and the data reader.
4. **When roots disagree, mathematics wins.** If PMOIRED and virgil
   disagree, our analytic references decide which is wrong (as with
   PMOIRED's ring precision, [P3](pmoired.md)).
5. **Disagreements are findings, never tolerances.** We never loosen a
   tolerance to hide one. Each is kept as a test (pinned behaviour plus a
   strict `xfail` for the fix), fixed in virgil by a PR, or reported upstream.
6. **Reviews are checked like findings.** An adversarial review (by a
   person or a model) is a list of claims, not verdicts. Reproduce each
   claim, with a test or a number, before ruling on it. Anything high-stakes
   (a ruling that flips a verdict, lifts a "provisional" label or closes a
   finding) also needs a second reviewer who did not write the first
   judgement: another model family or a person. A review's claim that is
   not reproduced is recorded as unconfirmed, not acted on.
7. **People sign off the conventions.** Every mapping between conventions
   (axes, signs, which angle is which) is written on a page a person can
   check, because a model can agree with itself about a convention.

## What this does and does not establish

The independent code was written by an AI agent (Claude), the same kind of
author as most of virgil. It reads virgil's *documentation* to learn
parameter conventions, never its implementation, and `src/crosscheck` cannot
import virgil (a test enforces it). Shared misconceptions are still possible.
Four guards reduce the risk:

* Each of our numerical routes is checked against an analytic form before it
  is used to judge virgil.
* Conventions come from external standards (OIFITS sign and orientation,
  position angle North through East), not from virgil.
* Masking data are simulated two ways, by dLux (human-written, by other
  authors) and by a closed-form interferogram, and the two must agree.
* Every disagreement is a finding to be explained, never a tolerance to
  loosen.

External packages written by other people (PMOIRED, CANDID, fouriever,
orbitize!, eht-imaging, MPoL) are a stronger check still. Each has its own
page under Method.

## Two kinds of grounding

Trust in virgil has two halves, grounded in different ways.

**Models and data input/output are grounded externally.** Every forward
model (a component's visibility, a composite scene, a chromatic flux),
every reader and writer (OIFITS, AMIGO DISCO products) and every observable
operator (V², closure phases and their covariance) is compared with
mathematics or with a published pipeline: PMOIRED, CANDID, dLux, harmonix,
AMICAL, AMIGO. Every mismatch is ruled to be **ours**, **theirs**, or a
**difference of definition**, and recorded. This is the main job of this
repository, because it needs code that virgil must not contain.

**Retrievals are grounded internally.** Once a forward model is trusted,
virgil can simulate data with it. Injection-recovery and Monte Carlo
statistics then test everything downstream: fits, uncertainties, grid
searches, detection limits, sampling and imaging. No external code is
needed, so this half can run inside virgil. Its correctness is statistical:
pulls must be N(0, 1), credible intervals must cover at their nominal rate,
and false-alarm rates must match their thresholds.

**virgil's own CI counts as evidence.** Internal properties are already
tested in virgil and are not reimplemented here: `model_on_grid` equals
`model`, float32 agrees with float64, gradients are correct, the whitening
algebra holds, and small recoveries succeed. Such a test becomes evidence for
a node of the flow once it is tagged (see [Evidence records](#evidence-records)).
It counts only when the forward model it relies on is itself externally
grounded.

Where each kind of check lives:

| Check | Where | When |
| --- | --- | --- |
| Internal consistency, small recoveries | virgil `tests/` | every virgil PR |
| External parity of models and I/O (cheap) | here, `tests/` | every PR here; weekly against virgil `main` |
| External parity (expensive: dLux images, CANDID maps) | here, `campaigns/` | weekly, or on demand |
| Monte Carlo statistics of retrievals | here, `campaigns/`, using only virgil | weekly (small), before releases and on OzSTAR (large) |

## Kinds of reference

Each check is tagged with the kind of reference virgil is compared with:

| Tag | Meaning |
| --- | --- |
| **Analytic** | A closed-form result from a textbook or the physics (Airy pattern `2 J1(x)/x`, Gaussian transforms, the shift theorem, second moments), evaluated with NumPy/SciPy. |
| **Quadrature** | Our own numerical integral of the brightness distribution: Gauss–Legendre, Gauss–Hermite or trapezoid rules over a point cloud, summed directly. It is checked against the analytic form wherever both exist. |
| **First principles** | Our own implementation of a standard: uv tracks from Earth rotation (Thompson, Moran & Swenson ch. 4), closure phases and the OIFITS v2 format (Pauls et al. 2005; Duvert et al. 2017). |
| **dLux** | An independent optical simulator written by other people (Desdoigts et al. 2023), propagating wavefronts through a sampled pupil. |
| **Statistical** | Ensembles of noisy simulations testing that virgil's uncertainties are calibrated. |
| **PMOIRED** | Another package's model of the same file (Mérand 2022), evaluated at mapped parameters. |
| **virgil (internal)** | Agreement between two virgil code paths. Listed only where the other path is itself validated here. |

## The flow and its status

The flow of trust, from the roots up through virgil's objects, and its
current state are generated from the evidence, not written by hand. See
**[Trust](../index.md)**, the site's home page. It keeps two questions apart.

* The **verdict** comes from the evidence at the virgil commit it ran on. It
  is one of: verified; works, but relies on a known bug (or on a part not yet
  verified); convention from virgil's docs; known bug, fix pending; partly
  checked; check failing; or not yet checked.
* **Freshness** is separate: whether virgil's source under a part has changed
  since.

The page also shows the end-to-end results on real and simulated data, which
roots check which parts, every finding, and how precisely each check agrees.

### Roots are a pool

All the roots listed in the [principles](#principles) form one pool:
mathematics, standards, statistics, **literature** (published results
reproduced from the authors' own data), and the trusted packages. Any link in
the graph may be validated against whichever root suits it, and no stage needs
every package. A node asks only for a number of *distinct* roots. Conventions
need two, because a single reference could share a mistake.

### Units and pipelines

Most evidence is **unit** evidence: one object against one root. It says
*where* something is right or wrong. **Pipeline** evidence checks a whole
chain at once: reading OIFITS, building `OIData`, a model, the fit, its
uncertainties, a grid search and limits.

Reproducing a **published binary** from the authors' archival data is the
strongest pipeline check (root: literature). The people, pipeline and data
are all independent, and real data bring quirks no simulation has.

A pipeline counts as validated when its own evidence passes and every step is
trusted on its own. If the end-to-end check fails while every unit passes,
the fault is in the joins between them, which is itself a finding. The graph
lists the pipelines checked so far (simulated VLTI fits, dLux masking fits,
identical fits with PMOIRED) and those planned (published binaries,
regularised imaging against eht-imaging or MPoL).

## Making it traceable and auditable

The goal is that any claim about virgil's correctness, such as "`UniformDisk`
is right to 1e-15", can be followed to the evidence behind it. That evidence
is the test or campaign, the exact virgil commit and package versions it ran
against, the seed, the numbers, the criterion it had to meet, and the
command to rerun it. The Trust page is generated from that evidence, never
written by hand.

### Evidence records

Every check, whether a test here, a test in virgil or a Monte Carlo
campaign, produces a small JSON record:

```json
{
  "id": "parity.uniform_disk.pmoired",
  "claim": "UniformDisk visibilities equal PMOIRED's ud",
  "validates": ["virgil.models.UniformDisk"],
  "roots": ["mathematics", "pmoired"],
  "source": "tests/test_pmoired_vs_virgil.py::test_disk_star_and_companion",
  "command": "pytest tests/test_pmoired_vs_virgil.py -k disk_star",
  "virgil_commit": "5239a70",
  "versions": {"jax": "0.11.2", "scipy": "1.16.2", "pmoired": "26.10.1"},
  "seed": null,
  "metrics": {"max_dV2": 4.4e-16, "max_dCP_deg": 2.1e-13},
  "criterion": "max_dV2 < 1e-12 and max_dCP_deg < 1e-9",
  "passed": true,
  "runner": "github-actions",
  "date": "2026-10-04"
}
```

Tests declare what they validate with a marker. A pytest plugin writes the
records (`conftest.py` here; a five-line copy in virgil's `conftest.py`):

```python
@pytest.mark.validates("virgil.models.UniformDisk", roots=["mathematics", "pmoired"])
def test_disk_star_and_companion(...): ...
```

The weekly job commits records to `evidence/` on an `evidence` branch, which
keeps main's history clean, and they are published with the docs. Large
outputs (campaign samples, figures) are uploaded as release assets, or kept
on OzSTAR `/fred` with a checksum in the record.

**One virgil commit per release.** Evidence from different virgil commits
cannot be read together: a pass at one commit says nothing about another,
and the Trust page would show a verdict for no virgil that exists. Before a
virgil release, every evidence file (`trust/evidence/latest.jsonl`,
`virgil.jsonl`, `campaigns.jsonl`) is regenerated against the release commit
and carries a run header naming that one commit, from a clean tree.
`scripts/check_evidence_pin.py --release --expect <sha>` (also the tier B
test `tests/test_evidence_pin.py`, run with `VALIDATION_RELEASE=1`) fails
when the headers name more than one commit, when a header does not record
whether the tree was clean (virgil's own CI evidence is exempt: it is a fresh
checkout at the sha), or when the run was dirty. A Monte Carlo campaign too
long to repeat is rerun against the release commit; the gate has no "stale"
status, so a campaign that was not rerun fails it.

### The trust graph

`trust/graph.yml`, written by hand, lists virgil's objects as nodes, with:

* their **dependencies** (a `System` depends on its components; `fit` on
  the likelihood, which depends on the models and `OIData`);
* the **virgil source paths** each depends on (`src/virgil/models.py`,
  `oifits.py`, ...);
* the **roots** they need: two independent roots for anything a model
  could get wrong in a correlated way, conventions above all.

`scripts/trust.py` joins the graph with the latest evidence and gives
each node a **verdict**, from the evidence at the virgil commit it ran on.
The verdicts are:

* **verified**: passing evidence from its required roots, and every node
  it relies on verified;
* **works, but relies on a known bug** (or on a part not yet verified):
  its own evidence passes, but something below it does not; the page
  names the part at the bottom of the chain and its open finding;
* **known bug, fix pending**: a ledger entry against virgil is open;
* **convention from virgil's docs**: the checks agree, but a convention
  they test (a rotation sense, which side of a disk is near) is defined only
  by virgil's documentation, which our code was written from too. It needs
  another code or a published result to agree, or an entry in
  `trust/signoffs.yml` (a paper equation, or Ben's sign-off at a commit;
  `by:` entries are checked via `git blame` for human author metadata and
  no `Co-Authored-By:` trailer, which catches agents that disclose themselves
  rather than cryptographic signatures);
* **partly checked**: evidence from too few independent roots, overall or
  for one of the properties the graph lists for the part (a model's
  profile and its modulation sense are separate properties, each with its
  own count);
* **check failing**: a check of it fails;
* **not yet checked**: no evidence.

**Freshness** is reported separately, never folded into the verdict. It
says whether virgil's source under a node has changed since that commit
(from `git diff <commit> origin/main -- <paths>` on virgil). The home page
shows the verdicts as a layered map with a detail panel per node, linking
each check's evidence and test. "Verified at virgil `5239a70` on
2026-10-04" is then a statement anyone can audit.

### The mismatch ledger: ours, theirs, or definition

`ledger.yml` replaces the separate findings tables (virgil's findings in
the README, PMOIRED's in [pmoired_notes.md](pmoired.md)). Every
disagreement gets an entry, ruled by a fixed procedure.

1. **Reproduce minimally**: the smallest scene and file that shows it, as a
   test.
2. **Check the conventions page**: is it a mapping error on our side?
3. **Referee with mathematics**: if a closed form or our two-route
   quadrature covers the case, whoever disagrees with it is wrong.
4. **Otherwise referee with a third code**, and say so (weaker evidence).
5. **Rule** it as `virgil` (our fault), `external:<package>` (theirs),
   `definition` (both right, defined differently: document the mapping) or
   `crosscheck` (our reference was wrong: fix it, and add the missing
   second route that would have caught it).
6. **Act**:
    * on virgil, a strict `xfail` test pinned to the fix, and a PR into
      virgil (delegated to an agent with the reproducer);
    * on an external package, a batch of issues raised upstream once there
      are enough, with Ben's approval;
    * on a definition, an entry in the conventions page.

Each entry records the evidence ids, the ruling, the referee, and the PR or
issue link, so a ruling can itself be audited.

### Monte Carlo campaigns for retrievals

`campaigns/` holds scripts that use only virgil (plus our trusted
simulators for data). Each takes `--draws`, `--seed` and `--out` and
writes evidence records, so the same campaign runs small in CI and large
on OzSTAR (via `ozstar_scripts`). Every campaign states its acceptance
criteria and sample size **before** it runs.

| Campaign | Statistic | Criterion | Draws |
| --- | --- | --- | --- |
| Parameter pulls, every scene and fitter | mean, sd of (fit − truth)/σ | mean within ±3/√N; sd within 1 ± 3/√(2N) | 450 detects a 10 % error-bar miscalibration at 3σ |
| Interval coverage | fraction of 68 % and 95 % intervals containing truth | binomial test, p > 0.001 | 1000 |
| Sampling (`numpyro_model`) | simulation-based calibration ranks | χ² test of uniform ranks | 1000 posteriors |
| Detection thresholds | false-alarm rate on companion-free data | binomial at the nominal rate (e.g. 0.27 % at 3σ) | 10⁴ |
| Contrast limits | detection fraction of injected companions at the quoted limit | matches the quoted confidence; agrees with CANDID where methods coincide | 10⁴ |
| Imaging | recovery metrics on simulated scenes | stated per scene against a regularisation-free reference | 100 per scene |

The pulls already in `tests/test_vlti.py` become the first campaign, with
a 60-draw smoke version left in CI.

### Cost tiers

| Tier | Runs | Budget | Contents |
| --- | --- | --- | --- |
| A | every PR, here and in virgil | ~2 min | conventions, analytic and PMOIRED parity, smoke recoveries |
| B | weekly, against virgil `main` | ~1 h on CI | dLux images, PMOIRED fits, campaigns at modest N |
| C | before a virgil release, or on demand | hours on OzSTAR | campaigns at full N, CANDID maps and limits |

A tier B or C result counts until virgil's code under its node changes.
The page then marks the node as changed since its evidence (see Freshness
above), so the expensive runs repeat only where virgil has changed.

### Release gate

virgil's release checklist gains these items (also in virgil's
`RELEASING.md`):

1. Tier A of this repository passes against the release commit.
2. All evidence is regenerated and pinned to that single commit
   (`scripts/check_evidence_pin.py --release`, above).
3. No ledger finding ruled `virgil` with severity `bug` (its `kind` begins
   `bug`) is open without a recorded decision: fixed, or accepted with a
   reason in the entry's `decision` field and in the release notes.
4. Gl 229 is no longer a release gate (after 0.4.1). Its alias result is on the
   [aliases pages](../orbits/index.md#period-aliases), marked as a warm start, with the stored cold result.
5. For anything the release changes, the trust page shows no node that is
   changed since its evidence, has a known bug, is failing or is not yet
   checked, or the release notes say which ones remain and why.

*Waiver, 0.4.1:* Ben waived the rerun of the `numpyro_model` SBC campaign for 0.4.1 (2026-10-23), so `campaigns.jsonl` stays at 0.4.0 (8c51dc5) and the pin check fails on it by design for this release.

## Order of work

1. ✅ **Evidence records**: the `validates` marker and plugin here, tagging
   the existing tests; the same marker in virgil's `conftest.py`, with
   virgil's CI uploading its records as an artifact that our weekly job
   downloads. (2 days)
2. ✅ **Trust graph**: `trust/graph.yml` with dependencies and source paths;
   `scripts/trust.py` for propagation and freshness; the Trust page
   generated from it (rebuilt as the site's home page, `plan_trust_page.md`).
   (2 days)
3. ✅ **Ledger**: `trust/ledger.yml` with the existing findings (virgil 1–5,
   PMOIRED P1–P3) and its rendered page. (half a day)
4. **Campaign runner**: move the pulls into `campaigns/`, add interval
   coverage, and an OzSTAR job template. (2 days)
5. **Fill the gaps**, roots first: the table below, and the plan's CANDID
   and PMOIRED stages 2–4. Each is a PR that turns a node green and adds
   its evidence.
6. **Release gate** in virgil's release checklist.

### External gaps, roots first

In dependency order, each with the root it would rest on:

| Gap | Root |
| --- | --- |
| OIFITS reader on real files | PMOIRED's and CANDID's readers on ESO archive data (GRAVITY, PIONIER, MATISSE) |
| Spectra, bandwidth smearing | mathematics (channel integrals of closed forms); PMOIRED's spectra |
| `GravityDarkenedStar` | the non-rotating limit (a limb-darkened disk, closed form); Shashank Dholakia's original code (golden values already in virgil) |
| `HarmonixModel` | harmonix/starry spherical-harmonic maps against direct surface quadrature |
| Flared disks | direct quadrature of the documented brightness distribution (done) |
| Fits with other optimisers | statistics (pulls), PMOIRED's fits (plan Stage 2) |
| Grid search, detection limits | CANDID (done: maps, significance, Absil limits, fits); injection-recovery detection rates (statistics) |
| Sampling | simulation-based calibration (rank statistics uniform) |
| AMIGO DISCO records | the AMIGO package's own forward model on a common scene |
| Regularised imaging | regularisers against eht-imaging and MPoL (done); reconstructions against eht-imaging's imager on the same files; recovery statistics on simulated scenes |


Related: virgil's own note on [matching PMOIRED's
features](https://github.com/benjaminpope/virgil/blob/main/design/pmoired_parity.md).
