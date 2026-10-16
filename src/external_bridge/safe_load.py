"""Load downloaded pickled numpy data without executing arbitrary code.

vis-r's ``hr4796.selfcal.npy`` is an object array and FRAP's example data are
plain pickles, so ``np.load(..., allow_pickle=True)`` would run whatever the
file says. These loaders unpickle with a whitelist (numpy array
reconstruction, builtins that cannot execute anything) and refuse the rest.
"""

import pickle

import numpy as np

_MODULES = {
    "numpy", "numpy.core.multiarray", "numpy._core.multiarray", "numpy.core.numeric", "numpy._core.numeric",
    "numpy.dtypes", "numpy.core._multiarray_umath", "numpy._core._multiarray_umath", "builtins", "collections",
}
_BLOCKED = {"eval", "exec", "open", "__import__", "compile", "getattr", "setattr", "input", "breakpoint"}


class _Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if module in _MODULES and name not in _BLOCKED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def load_pickle(path):
    with open(path, "rb") as f:
        return _Restricted(f).load()


def load_object_npy(path):
    """An ``.npy`` file holding an object array (a pickle after the npy header)."""
    with open(path, "rb") as f:
        major, _ = np.lib.format.read_magic(f)
        (np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0)(f)
        return _Restricted(f).load()
