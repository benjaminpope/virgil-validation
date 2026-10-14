#!/bin/bash
# One prototype run of the orbit solver v2 design (design/orbit_solver_v2.md §9): one line of tasks.txt, in its
# own work directory. Used by the OzSTAR job orbit_v2_proto (one array task per line) and for local smoke tests.
#   bash run_task.sh TASK_ID OUTDIR      (TASK_ID: 0-based index of the non-comment lines of tasks.txt)
# Real-data lines run S2 (exp2_screen.py) and S3-S4 (exp3_lm.py) on the committed S1 catalogue. Injection lines
# first build the data (inject.py, residual transplant), then S1 with closure phases only (exp1_peaks.py, as in
# §9.6), the f table (exp4_neff.py), S2 and S3-S4. WINDOW blind is the §9.4 window (P 3-1660 d); narrow is 10-14 d.
# SMOKE=1 shrinks the grid and the LM to seconds, to check paths and imports. PY (default python) runs the scripts.
set -eo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python}
TASK=${1:?task id}; OUT=${2:?output directory}
LINE=$(grep -v -e '^#' -e '^[[:space:]]*$' "$HERE/tasks.txt" | sed -n "$((TASK + 1))p")   # bash 3 on macOS: no mapfile
if [[ -z $LINE ]]; then echo "task $TASK: no such line in tasks.txt; nothing to do"; exit 0; fi
read -r NAME SEL WINDOW INJARGS <<< "$LINE"
WORK=$OUT/$NAME
if [[ -f $WORK/DONE ]]; then echo "task $TASK ($NAME): already done in $WORK"; exit 0; fi
mkdir -p "$WORK"
cp "$HERE"/*.json "$WORK"/
export PROTO_WORK=$WORK
case $WINDOW in
  blind)  export PLO=3 PHI=1660 ESTEP=0.1 NTAU=24 PERCELL=8 DSCREEN=1e9 PERBAND=2; KP=3 ;;
  narrow) export PLO=10 PHI=14 CAP=3000; KP=5 ;;
  *) echo "unknown window $WINDOW" >&2; exit 1 ;;
esac
NST=256; NIT=40
if [[ -n $SMOKE ]]; then export PLO=11.9 PHI=12.4 ESTEP=0.3 NTAU=6; NST=4; NIT=2; fi
echo "task $TASK: $NAME selection=$SEL window=$WINDOW P=$PLO-$PHI Kp=$KP starts=$NST x $NIT inj=[${INJARGS:-none}] work=$WORK"
if [[ -n $INJARGS ]]; then
  export TAG=inj$NAME
  (unset INJ; $PY -W ignore "$HERE/inject.py" "$NAME" $INJARGS)
  export INJ=$WORK/inj_$NAME.json
  CPONLY=1 $PY -W ignore "$HERE/exp1_peaks.py" 25 18
  $PY -W ignore "$HERE/exp4_neff.py" > "$WORK/exp4.log"
else
  export TAG=gl229
fi
$PY -W ignore "$HERE/exp2_screen.py" "$SEL" measured "$KP" 0.5
$PY -W ignore "$HERE/exp3_lm.py" "$SEL" "exp2_${TAG}_${SEL}_measured_K${KP}_sg0.5.json" "$NST" "$NIT"
rm -f "$WORK"/inj_*.pkl   # 3.6 MB each; inject.py remakes them exactly
$PY - "$WORK/exp3_${TAG}_${SEL}.json" "$NAME" <<'PY'
import sys, json
r = json.load(open(sys.argv[1]))["rows"][0]
print(f"top mode: P {r['P']:.4f} e {r['e']:.3f} i {r['inc']:.1f} a {r['a']:.2f} flux {r['flux']:.3f} "
      f"track rms from reference {r['ref_rms']:.2f} mas -> {'RECOVERED' if r['ref_rms'] < 0.5 else 'not recovered'}"
      + (" (null: the reference is the real orbit, so RECOVERED means the subtraction left it in)" if sys.argv[2] == "N" else ""))
PY
touch "$WORK/DONE"
