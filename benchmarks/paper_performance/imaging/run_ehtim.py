"""eht-imaging side of the imaging comparison: the driver. Imports neither virgil nor eht-imaging (the workers run in
``EHTIM_PYTHON``: ``ehtim_bench_worker.py``). Same commands as ``run_virgil.py``:

    python run_ehtim.py gate --case I-sim-24
    python run_ehtim.py run  --case I-sim-24 --task ladder --config matched [--size tiny] [--threads 1]
    tasks: ladder | per_eval | stop_default | stop_tight | native
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    import compare

    compare.main("ehtim", sys.argv[1:])
