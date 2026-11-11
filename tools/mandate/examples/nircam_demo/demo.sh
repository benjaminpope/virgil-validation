#!/bin/bash
# The whole toy campaign, start to finish, in a scratch git repository.
# Usage: bash demo.sh [scratch-dir]     (needs `mandate` and numpy on the PATH's python)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
work="${1:-$(mktemp -d)}"
cp -r "$here"/. "$work"/ && cd "$work" && rm -f demo.sh
git init -q . && git add -A && git -c user.name=demo -c user.email=demo@example.org commit -qm "plan and registry"
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "1. The plan, applied to the registry"
mandate check
mandate orders --state ready

say "2. Fan out: a wavefront fit per reference star (one fails its convergence check)"
MANDATE_ACTOR=runner mandate run --stage wavefront || true

say "3. A triage agent gets the failed order's brief, and may only choose from the menu"
mandate brief wavefront/toy_ref_e_f444w | sed -n '/## Decisions/,/## Notes/p'
MANDATE_ACTOR=agent-triage-1 mandate decide wavefront toy_ref_e_f444w n_modes=7 \
  --why "try an option that is not on the menu" || true
MANDATE_ACTOR=agent-triage-1 mandate decide wavefront toy_ref_e_f444w n_modes=4 \
  --why "fit_converged fails at 6 modes; the plan allows 4 for short exposures"
MANDATE_ACTOR=runner mandate run --stage wavefront

say "4. The same agent reads the note on TOY-REF-D and proposes an exclusion; a person approves"
MANDATE_ACTOR=agent-triage-1 mandate exclude target_name=TOY-REF-D --stage basis \
  --why "notes/ref_d_binary.md: a known binary must not enter a PSF basis"
mandate orders --stage basis
id=$(tail -1 docket.jsonl | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
mandate resolve "$id" approve --by ben --why "agreed"
mandate orders --stage basis

say "5. Fan in: one PCA basis per filter, with its members listed"
MANDATE_ACTOR=runner mandate run --stage basis
python3 -c "
import json, glob
for f in sorted(glob.glob('ledger/basis/*/order.json')):
    o = json.load(open(f)); print(o['id'], [m['key'] for m in o['inputs']['wavefront']], 'dropped', o['dropped']['wavefront'])"

say "6. Agents classify the science targets; nothing downstream exists until a person approves"
for target in TOY-PLANET-1:planet TOY-PLANET-2:planet TOY-DISK-1:disk; do
  name=${target%%:*}; case=${target##*:}
  mandate brief classify/$name > /dev/null          # what the worker agent would be handed
  echo "{\"science_case\": \"$case\", \"why\": \"proposal abstract for $name\"}" > "$(mandate where classify/$name)/result.json"
  MANDATE_ACTOR=agent-classify mandate submit classify/$name
done
MANDATE_ACTOR=agent-classify mandate request classify TOY-UNKNOWN \
  --why "the abstract does not say whether TOY-UNKNOWN has a disk; planet or disk?"
mandate orders --stage planet_search
for id in $(python3 -c "
import json
for line in open('docket.jsonl'):
    e = json.loads(line)
    if e['kind'] == 'annotate': print(e['id'])"); do mandate resolve "$id" approve --by hayden --why "checked against the proposal" > /dev/null; done
mandate orders --stage planet_search

say "7. Fan out again over the science targets, by science case"
MANDATE_ACTOR=runner mandate run --stage planet_search
MANDATE_ACTOR=runner mandate run --stage disk_deconvolution

say "8. Score the published target (the only stage that sees the answer), then the catalogue"
MANDATE_ACTOR=runner mandate run --stage score
MANDATE_ACTOR=runner mandate run --stage catalogue
echo "orders that were given the published separation:"
grep -l published_sep_mas runs/*/*/order.json || true

say "9. Rerun one order from scratch: is it reproducible?"
MANDATE_ACTOR=runner mandate run --fresh basis/F444W

say "10. Audit, and the page a person reads"
git add -A && git -c user.name=demo -c user.email=demo@example.org commit -qm "campaign run"
mandate verify
mandate status --write > /dev/null && cat STATUS.md
echo; echo "campaign left in $work"
