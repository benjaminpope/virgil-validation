# Campaigns

Two [mandate](../tools/mandate/README.md) campaigns over this repository's published-data benchmarks. Each wraps the
drivers that already exist in `scripts/`; neither adds a fit path or a model.

| | `orbits/` | `contests/` |
|---|---|---|
| Units | 15 systems (14 Track B records, plus iota Peg on MIRC-X) | 22 contest datasets x 8 ensemble members |
| Model | virgil's `OrbitPipeline`, every setting its default | CLEAN-started Gaussian-process image, evidence-chosen hyperparameters, star on and off |
| Stages | `data` (dry run) -> `orbit` -> `compare` (sealed) | `member` -> `ensemble` -> `score_2004` (sealed) |
| Informed track | none | `informed_fit`: the model family the contest paper describes, fitted from the data (2004 data2, 2018) |
| Open knowledge | `notes/*.md`: data facts from the papers and our records | `notes/given_<year>.md`: what contestants had before the deadline |
| Sealed | the published orbit (`notes/sealed_*.md`, `oidb/trackb/`) | targets, truths, winners, scores (`notes/sealed_<year>.md`) |

```bash
pip install -e tools/mandate
python campaigns/build_registries.py     # registries and generated notes, from oidb/trackb, TASKS and the manifest
cd campaigns/orbits
mandate explain                          # the plan in plain words
mandate                                  # where it stands and what to run next
mandate run --stage data                 # on a machine with $ESO_BIN_ROOT; then --stage orbit --slurm on OzSTAR
```

Set `MANDATE_RUNS` to put run folders on `/fred`, `ESO_BIN_ROOT` / `OIDB_ROOT` for the orbit data and `CONTEST_DATA`
for the contest data.

What is and is not checked: `campaigns/tests/` and CI check that the registries match their sources, that both plans
are valid, that nothing given to a blind orbit fit contains the published period, eccentricity, axis or angles, and
that a contest member's brief has the pre-deadline note and not the sealed one. The stages themselves have not been
run through mandate yet: they need the data and a GPU.

Three levels of knowledge. *Open* is what a blind analysis may use. *Informed* is what a publication says the
object is: `contests/informed_models.yml` holds one line per dataset (a spotted star with a companion; a star, a
rim, a planet and a disk) and no values, and only `informed_fit` is given it. *Sealed* is the published values, for
scoring. An informed result is never a blind one and is labelled so in every record. To use a paper for another
dataset: add its line to `informed_models.yml`, add the family to `scripts/contest_informed.py`, and add the option
to the plan.

`informed_fit` has been run once through mandate, on 2004 data2 on a CPU: it converges on a star, a companion and a
spot from the data alone, with chi2/N about 28, so the family is recovered and the fit is not yet good. The 2018
family has only been smoke-tested, on the 2004 file.

Known limits. `scripts/gallery_orbits.py` still opens `oidb/trackb/<slug>.json`, which holds the published orbit
next to the file list, for the instrument and template; `tests/test_gallery_orbits.py` is what keeps it from reading
more. The hand-written notes in `orbits/notes/` were drafted by an agent from the records' `caveats` and
`third_light` fields, leaving out anything that states a result; each says so in its front matter and wants a
person's read. Per-channel imaging of the chromatic contests (2010, 2024 Obj2) is not a stage here because it is
not built.
