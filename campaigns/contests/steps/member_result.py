"""Turn one ensemble member's <label>.npz (scripts/contest_images.py --member) into a result.json.

    python steps/member_result.py OUT_DIR LABEL

`finished` passes when the member file exists and its best chi2/N is finite. The status is `warn` when chi2/N on
the quoted errors is above 1.5: the level at which scripts/combine_ensemble.py stops keeping a member when the
best one fits to the noise. A warning is not a failure; whether the member is used is the ensemble's decision.
NumPy only.
"""

import json
import pathlib
import sys

import numpy as np

KEEP_FACTOR = 1.5  # combine_ensemble.py's chi2 filter


def result_of(path):
    if not path.exists():
        return {"status": "fail", "checks": {"finished": "fail"}, "problem": f"no {path.name}"}
    z = np.load(path, allow_pickle=False)
    numbers = {k: float(z[k]) for k in ("best_chi2_red", "best_log_z", "error_scale", "rescaled_chi2_red", "gamma")
               if k in z.files and np.ndim(z[k]) == 0}
    chi2 = numbers.get("best_chi2_red", float("nan"))
    if not np.isfinite(chi2):
        return {"status": "fail", "checks": {"finished": "fail"}, "problem": "best_chi2_red is missing or not finite", **numbers}
    status = "warn" if chi2 > KEEP_FACTOR else "pass"
    return {"status": status, "checks": {"finished": "pass", "chi2_within_keep_factor": status},
            **{k: (v if np.isfinite(v) else None) for k, v in numbers.items()}}


if __name__ == "__main__":
    out, label = pathlib.Path(sys.argv[1]), sys.argv[2]
    (out / "result.json").write_text(json.dumps(result_of(out / f"{label}.npz"), indent=1) + "\n")
