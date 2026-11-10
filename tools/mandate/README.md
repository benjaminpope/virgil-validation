# mandate

Fan agents and cluster jobs out over many datasets under one plan a person approved, and check
afterwards that they did that and nothing else.

It is one small Python package (PyYAML is the only dependency) and it lives here for now because
virgil-validation is its first user. Nothing in it imports virgil or this repository, so it can
move to its own repository unchanged.

## The idea

Agents do three things wrong at scale: they lose context, they bend the task until it passes, and
they do good work nobody asked for. `mandate` answers each with a file, not with a rule to remember.

| Problem | Answer |
|---|---|
| Losing context | State lives in the campaign directory, never in a conversation. `mandate brief ORDER` prints everything one worker needs for one order, so a fresh agent can pick up any piece of work. |
| Bending the task | Numbers come only from the stage's command. A decision is a choice from a menu in the plan. Checks named in the plan must be present and passing. Every attempt is kept and counted. The run folder is hashed. |
| Acting without a mandate | The plan is the mandate. Anything it does not allow is a `request`, which parks that one order for a person and lets the rest carry on. `mandate verify --base main` sorts every change into work, or a change of mandate. |

The LLM is not in the path that produces numbers. Code stages fan out over datasets; agents fan
out over the judgement calls (classifying targets, reading papers) and over the exceptions.

## What a campaign is

```
mandate.yaml        the plan: stages, decision menus, who may change what       (people edit)
registry/units.csv  one row per unit of data: a snapshot of your spreadsheet    (people edit)
notes/*.md          human notes and paper extracts, each open or sealed         (people, librarian agents)
docket.jsonl        every departure from the plan's defaults, append-only       (written by `mandate`)
ledger/             one small record per work order and per attempt             (written by `mandate`)
STATUS.md           the page a person reads                                     (generated)
runs/               run folders; large, outside git, usually on the cluster
```

A **stage** is one step applied to groups of units. `over` selects units, `by` groups them (one
order per group), `needs` names the upstream results it reads. That one shape covers fan-out
(`by: [obs_id]`), fan-in (`by: [filter]` with `needs` matching on `filter`: a PCA basis over every
reference star in a filter), multi-epoch grouping (`by: [target]`), "the two calibrators nearest in
time" (`nearest: {by: t_min, k: 2}`), and branching on science case (`over: {science_case: disk}`).

An **order** is one stage on one group. Its id is a hash of the stage, the unit rows it is given,
the resolved decisions, the pinned versions and the exact upstream results it reads. An ensemble
order lists its members and what it dropped and why. Change the membership of a basis and the
basis is a new order; everything downstream of the old one shows as waiting again, and the old
results stay in the ledger as superseded.

The **docket** is the one list of everything that is not the plan's default: a decision changed
for one order, units excluded, a registry column filled in by an agent, a request. Each entry has
a reason and an author. Entries the plan reserves for people are `proposed` until one of the
plan's `humans` runs `mandate resolve`.

## Try it

```bash
pip install -e "tools/mandate[test]"
bash tools/mandate/examples/nircam_demo/demo.sh
```

The demo is a toy of the NIRCam workflow with synthetic numbers: a wavefront fit per PSF reference
star, a PCA basis per filter, a point-source search or a deconvolution per science target using the
basis and the two references nearest in time, a sealed scoring stage for the published target. In
it an agent is refused an option that is not on the menu, picks one that is, proposes excluding a
binary reference star, and files a request for a target it cannot classify. It ends with
`mandate verify` and the status page. `examples/gravity_sketch/` is the same shape for VLTI data
(per-night calibration, epochs grouped into orbits); it is checked, not run.

## Starting from a chat

```bash
mandate init my-campaign
mandate -C my-campaign ingest sheet.csv
```

Then tell a chat agent: "run `mandate guide` and help me write the plan". The guide tells the
agent what to ask you (units, kinds of data, steps, combined products, checks, choices, published
answers, people) and what never to invent (a column, a command, a check, an option). It uses
`mandate columns` to see what the registry offers, writes `mandate.yaml`, runs `mandate check`, and
shows you `mandate explain`: the plan in plain words. You approve by committing.

## Day to day

`mandate` with no arguments is the home screen: what is waiting for a person, a count of orders
per stage and state, and the next commands to run. `mandate review` walks you through everything
waiting for you, says what each approval would change (new orders, orders to redo), and takes
approve, deny, approve-all-of-this-kind or skip.

## Commands

| | |
|---|---|
| `mandate` | Home screen: waiting for a person, counts, what to do next. |
| `mandate init DIR` | Start a campaign directory. |
| `mandate guide` | Instructions for an agent helping someone write a plan. |
| `mandate columns` | The registry's columns and their values. |
| `mandate explain` | The plan in plain words. |
| `mandate review [--list]` | Answer everything waiting for a person, with the effect of each answer. |
| `mandate ingest FILE_OR_URL` | Snapshot a CSV (or a sheet's CSV export URL) as the registry; reports added, removed, changed. |
| `mandate check` | Validate the plan against the registry. |
| `mandate status [--write]` | The status page: waiting for a person, counts per stage, failures, departures from defaults, what ensembles dropped, reruns. |
| `mandate orders [--stage S] [--state ready] [--worker agent]` | List orders. |
| `mandate brief ORDER` | The self-contained brief for one order. |
| `mandate run ORDER...` / `--stage S` | Run code-stage orders. `--slurm` writes a job array and prints the `sbatch` line; it never submits. `--fresh` reruns from empty and records whether the result reproduced bit for bit. |
| `mandate submit ORDER` | Hand back an agent stage's `result.json`; it is checked against the plan's `expects`. |
| `mandate decide STAGE KEY NAME=OPTION --why` | Choose a listed option for one order. |
| `mandate exclude COL=VALUE [--stage S] --why` | Drop units from some or all stages. |
| `mandate annotate COL=VALUE --set COL=VALUE --why` | Fill in registry columns. |
| `mandate request STAGE [KEY] --why` | Ask a person; parks the order (or the stage). |
| `mandate resolve ID approve\|deny --why` | A person answers. |
| `mandate verify [--base REF] [--strict]` | Audit. Exit 1 on a violation. |
| `mandate collect [DIR]` | Copy order and attempt records from run folders into the ledger. |
| `mandate protocol` | How a lead agent, worker agents and people work a campaign. |

`ORDER` is `stage/key` (for example `basis/F444W`) or `stage/id`.

## The plan

```yaml
campaign: nircam-wavefront
unit_id: obs_id                    # the registry column that names a unit
humans: [ben, hayden]              # who may approve and resolve
sealed_columns: [published_sep]    # answers: only sealed stages are given them
ignore_columns: [publication_status]   # bookkeeping columns: a change is not new work
sealed_root: ../sealed             # optional: sealed notes kept outside the campaign
exclusions: human                  # or `agent`: who may exclude units without approval
versions:
  nircam: {git: ../nircam, pin: 4f2a9c1}   # refuse to run on another commit or a dirty checkout
stages:
  wavefront:
    doc: Fit a wavefront model to one PSF reference observation.
    over: {role: reference}        # {column: value} or {column: [values]}
    by: [obs_id]                   # default: one order per unit; [] is one order for everything
    needs:
      upstream_stage: {match: [filter], min: 3, nearest: {by: t_min, k: 2, within: 30}}
    run: python fit.py --order {order} --modes {decision.n_modes}
    result: result.json            # JSON with a status (pass, warn, fail) and named checks
    status_key: status             # `worst_check` reads a virgil pipeline's summary.json
    requires: [fit_converged]      # checks that must be present and passing
    digest: ["summary.json", "*.h5"]   # what to hash; default is everything but logs
    decisions:
      n_modes:
        default: 6
        options: [4, 6, 8]
        authority: agent           # default is human: an agent's choice is a proposal
        rules: [{when: {filter: F200W}, use: 8}]
    uses: [nircam]                 # which pinned versions this stage depends on (default: all)
    slurm: ["--gres=gpu:1", "--time=02:00:00"]
    sealed: false                  # true: may read sealed columns, notes and sealed stages
    playbook: playbooks/wavefront.md   # shown in the brief: symptom, diagnosis, allowed move
  classify:
    worker: agent                  # the work is a brief; the hand-back is result.json
    prompt: prompts/classify.md
    expects: {science_case: [planet, disk]}
    promote: {science_case: science_case}   # result key -> registry column, as a docket entry
    review: human                  # or `none`
```

Commands may use `{out}`, `{order}` (the path of the run folder's `order.json`), `{root}`,
`{key.COL}`, `{unit.COL}` (when the order has one value for it), `{decision.NAME}` and
`{input.STAGE}` (when exactly one upstream order feeds it). A step that needs more reads
`order.json`: units, decisions, and the run folder of every input.

A note is a Markdown file with optional front matter: `applies_to` (a selector), `stages`,
`visibility: sealed`, `source`.

## How it stays honest

- **Nothing runs on an uncommitted plan or registry.** Each order records the commit that last
  changed them. This is the criteria-commit rule this repository already uses for gates.
- **A decision off the menu is refused**, with "file a request". A new option is a plan change.
- **A step cannot pass by dropping a check**: `requires` names them.
- **Attempts are never overwritten.** A rerun is attempt 2 of the same order and shows on the
  status page. A changed decision is a different order.
- **`verify`** recomputes each run folder's hash against its record, checks attempt numbering and
  docket references, and with `--base` reports a changed or deleted ledger record or docket line
  as a violation and any edit to the plan, registry, notes or prompts as `MANDATE CHANGED`.
- **Sealed material** (the published answer, for a blind reanalysis) reaches only stages marked
  `sealed`, and `check` refuses a plan that routes a sealed stage into an open one. With
  `sealed_root` the answers are not in the campaign at all, and a sealed stage refuses to run on a
  machine that lacks them.

## What it does not do

- It detects, it does not prevent. An agent with a shell can edit any file; `verify --base` on the
  pull request is what catches it, so run it in CI and let people merge. For hard separation give
  each worker its own checkout or container without the sealed directory.
- `by` is a declared name. `verify` lists every entry in force on a person's name so the person
  can say "I did not approve that".
- It does not make a wrong check right. The checks, the injection tests and the playbooks are
  yours; this only makes sure they ran and were not edited.
- It does not stop a model recalling a published result. It keeps the result out of the inputs and
  the brief, and the only thing a worker can turn recall into is a logged choice from a menu.
- It is not a scheduler. It writes job arrays and reads results back. If a campaign needs retries,
  resource-aware scheduling or more than a handful of stages, generate a Snakemake workflow from
  the orders and keep the docket and briefs.

## On OzSTAR

Compute nodes have no internet and logins need a second factor, so agents stay off the cluster
and the boundary is two file transfers. Push the campaign repository; on a login node run
`mandate run --stage wavefront --slurm` and `sbatch` the script it prints. Each array task writes
its attempt into the run folder and the ledger of that checkout. Either commit the ledger there,
or pull the small records back (`rsync` with `--include='*/' --include='order.json'
--include='attempt-*.json' --exclude='*'`) and run `mandate collect DIR`. Set `MANDATE_RUNS` to put
run folders on `/fred`.

## Tests

```bash
cd tools/mandate && python -m pytest
```
