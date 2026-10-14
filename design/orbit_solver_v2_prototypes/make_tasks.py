"""Write tasks.txt, the run list of the OzSTAR job orbit_v2_proto (one line per array task; see run_task.sh).

Lines: NAME SELECTION WINDOW and, for injections, the inject.py arguments P e inc omega Omega a flux dtp.
  * blind_all, blind_sec5: the real Gl 229 nights in the blind window (design §9.4), both selections (decision 3);
  * A, B, N: the §9.6 injections and the null, now in the blind window;
  * R00..: seeded random injections matched to the Gl 229 regime: P log-uniform 3-400 d, e uniform 0-0.6,
    isotropic inclination, a 3-11 mas (so a(1+e) stays inside the 18 mas S1 grid), flux 0.1-0.6.
usage: python make_tasks.py [n_random] [seed]
"""
import sys, pathlib
import numpy as onp

n, seed = (int(sys.argv[1]) if len(sys.argv) > 1 else 20), (int(sys.argv[2]) if len(sys.argv) > 2 else 2026)
rows = ["blind_all all blind", "blind_sec5 sec5 blind",
        "A all blind 11.37 0.45 62 40 120 6.5 0.45 3.0",
        "B all blind 13.2 0.10 140 300 30 4.0 0.30 -5",
        "N all blind 12.0 0.0 0 0 0 0.0 0.0 0"]
rng = onp.random.default_rng(seed)
for k in range(n):
    P = float(onp.exp(rng.uniform(onp.log(3), onp.log(400))))
    e = rng.uniform(0, 0.6); inc = onp.degrees(onp.arccos(rng.uniform(-1, 1)))
    om, Om = rng.uniform(0, 360, 2); a = rng.uniform(3, 11); fl = rng.uniform(0.1, 0.6); dtp = rng.uniform(0, P)
    rows.append(f"R{k:02d} all blind {P:.4f} {e:.3f} {inc:.1f} {om:.1f} {Om:.1f} {a:.3f} {fl:.3f} {dtp:.3f}")
pathlib.Path(__file__).with_name("tasks.txt").write_text(
    f"# written by make_tasks.py {n} {seed}; one array task per line (task id = line number among non-comment lines)\n"
    + "\n".join(rows) + "\n")
print(len(rows), "tasks")
