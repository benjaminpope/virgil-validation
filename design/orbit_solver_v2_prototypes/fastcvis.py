"""Scratch-only monkeypatch: vectorised complex visibilities for orbital binaries (no per-sample lax.map)."""
import jax.numpy as np
from virgil import oidata
from virgil.models import OrbitalBinary
_orig = oidata.OIData._cvis_in_time
def _fast(self, model):
    if not isinstance(model, OrbitalBinary):
        return _orig(self, model)
    wavel = np.broadcast_to(self.wavel, np.shape(self.u))
    return model.at(self.dt, self.t_ref).model(self.u, self.v, wavel)
def enable():
    oidata.OIData._cvis_in_time = _fast
def disable():
    oidata.OIData._cvis_in_time = _orig
