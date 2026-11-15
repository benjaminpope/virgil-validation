"""Load downloaded pickled numpy data without executing arbitrary code.

vis-r's ``hr4796.selfcal.npy`` is an object array and FRAP's example data are
plain pickles, so ``np.load(..., allow_pickle=True)`` would run whatever the
file says. These loaders unpickle with an explicit ``(module, name)``
constructor allowlist (pure NumPy array/scalar/dtype reconstruction primitives
and safe builtin scalar/container types) and refuse every other global.
"""

import ast
import pickle
import struct

import numpy as np

_ALLOWED = frozenset({
    *(
        ("builtins", name)
        for name in (
            "bool", "int", "float", "complex", "str", "bytes", "bytearray",
            "tuple", "list", "dict", "set", "frozenset", "slice", "range",
        )
    ),
    ("collections", "OrderedDict"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.numeric", "_frombuffer"),
    *(
        ("numpy.dtypes", name)
        for name in dir(np.dtypes)
        if name.endswith("DType")
    ),
})


class _Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")


def _read_npy_header(f):
    ver = np.lib.format.read_magic(f)
    if ver == (1, 0):
        return np.lib.format.read_array_header_1_0(f)
    if ver == (2, 0):
        return np.lib.format.read_array_header_2_0(f)
    if ver == (3, 0):
        raw_len = f.read(4)
        if len(raw_len) != 4:
            raise ValueError("truncated .npy v3.0 header length")
        (hlen,) = struct.unpack("<I", raw_len)
        if hlen > 10000:
            raise ValueError(f".npy header length {hlen} exceeds safe limit")
        raw_hdr = f.read(hlen)
        if len(raw_hdr) != hlen:
            raise ValueError("truncated .npy v3.0 header")
        d = ast.literal_eval(raw_hdr.decode("utf-8"))
        if not isinstance(d, dict) or set(d) != {"descr", "fortran_order", "shape"}:
            raise ValueError(f"malformed .npy v3.0 header keys: {d!r}")
        if not isinstance(d["shape"], tuple) or not all(isinstance(x, int) for x in d["shape"]):
            raise ValueError(f"invalid .npy v3.0 shape: {d['shape']!r}")
        if not isinstance(d["fortran_order"], bool):
            raise ValueError(f"invalid .npy v3.0 fortran_order: {d['fortran_order']!r}")
        return d["shape"], d["fortran_order"], np.lib.format.descr_to_dtype(d["descr"])
    raise ValueError(f"unsupported .npy format version {ver!r}")


def load_pickle(path):
    with open(path, "rb") as f:
        return _Restricted(f).load()


def load_object_npy(path):
    """An ``.npy`` file holding an object array (a pickle after the npy header)."""
    with open(path, "rb") as f:
        _, _, dtype = _read_npy_header(f)
        if not dtype.hasobject:
            raise ValueError(f"expected object-dtype .npy array, got {dtype!r}")
        return _Restricted(f).load()
