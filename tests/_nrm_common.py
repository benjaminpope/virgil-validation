"""Shared by tests/test_nrm.py, tests/test_nrm_calibrated.py and tests/test_nrm_injection.py: the
mask, the wavelength and the pixel scale. Imports nothing but crosscheck.array and crosscheck.nrm, so
that what a check went through (scripts/trust.py, ``via``) is not another file's imports."""

import numpy as np

from crosscheck import array, nrm

WL = float(np.float32(4.8e-6))  # as OIFITS will store it
PIX = 30.0
UV = array.pupil_uv(nrm.HOLES)
