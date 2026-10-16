"""V2 (old against new): ``TruncatedCone`` before and after it became a ``PolarDisk``.

The pre-subsumption cone (virgil main a1dae85, midpoint rule in slant distance,
second-order in ``n_rings``) runs in its own environment (.venv-old: ``uv venv
--python 3.12 .venv-old``, then ``uv pip install "virgil-astro @
git+https://github.com/benjaminpope/virgil@<hash>"``) through cone_worker.py; the
new one is the pinned stage-V commit. Both drop the last 0.7 % of the tail (5
lengths), so the old form converges to the new one as ``n_rings`` grows.

PASS CRITERIA (fixed before the runs)
  C1  new (default n_rings = 48) vs old at n_rings = 8000 (converged): max |dV| <= 1e-5
      for the plain and the modulated cone.
  C2  the old form's error against its own converged value falls as n_rings^-2 (midpoint
      rule): ratio of errors between n and 2n of 3-5 (reported).
  C3  new at n_rings = 48 is at least 100x closer to the converged value than old at 48.
"""

import json
import pathlib
import subprocess
import tempfile

import numpy as np

from common import ROOT, save
import cone_worker as cw

OLD = ROOT / ".venv-old" / "bin" / "python"
OLD_HASH = "a1dae85584b78d19771e42a19a6146ff74414f2f"
WL = 2.2e-6


def main():
    rng = np.random.default_rng(11)
    r = 130.0 * np.sqrt(rng.uniform(0.01, 1.0, 400))
    a = rng.uniform(0, 2 * np.pi, 400)
    u, v = r * np.cos(a), r * np.sin(a)
    old_n = np.array([24, 48, 96, 192, 384, 768, 8000])
    new_n = np.array([12, 24, 48, 96, 192])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        np.savez(tmp / "old_in.npz", u=u, v=v, wl=WL, n_rings=old_n)
        np.savez(tmp / "new_in.npz", u=u, v=v, wl=WL, n_rings=new_n)
        script = pathlib.Path(cw.__file__)
        subprocess.run([str(OLD), str(script), str(tmp / "old.npz"), str(tmp / "old_in.npz")], check=True)
        subprocess.run([__import__("sys").executable, str(script), str(tmp / "new.npz"), str(tmp / "new_in.npz")], check=True)
        old, new = np.load(tmp / "old.npz"), np.load(tmp / "new.npz")
    out = {"old_virgil_hash": OLD_HASH, "old_virgil_version": str(old["virgil_version"]), "n_baselines": 400}
    for name in ("plain", "modulated"):
        ref = old[f"{name}_8000"]
        row = {"old_vs_converged": {}, "new_vs_converged": {}, "time_s": {"old": {}, "new": {}}}
        for n in old_n[:-1]:
            row["old_vs_converged"][int(n)] = float(np.abs(old[f"{name}_{n}"] - ref).max())
            row["time_s"]["old"][int(n)] = float(old[f"{name}_{n}_time"])
        for n in new_n:
            row["new_vs_converged"][int(n)] = float(np.abs(new[f"{name}_{n}"] - ref).max())
            row["time_s"]["new"][int(n)] = float(new[f"{name}_{n}_time"])
        e = row["old_vs_converged"]
        row["old_error_ratios_n_to_2n"] = {f"{n}->{2 * n}": e[n] / e[2 * n] for n in (24, 48, 96, 192, 384) if 2 * n in e}
        row["C1_pass"] = bool(row["new_vs_converged"][48] <= 1e-5)
        row["C3_pass"] = bool(row["old_vs_converged"][48] / max(row["new_vs_converged"][48], 1e-16) >= 100)
        out[name] = row
    out["converged_note"] = "old at n_rings = 8000 is the converged value; both forms drop the tail beyond 5 lengths"
    path = save("v2_cone_old_new", out)
    print(json.dumps(out, indent=1, default=float))
    print("wrote", path)


if __name__ == "__main__":
    main()
