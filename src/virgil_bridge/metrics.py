"""virgil's image-recovery metrics (``virgil.metrics``) behind plain NumPy
signatures, so that the tests compare them with ``crosscheck.image_metrics``
without touching JAX types. Nothing here computes a metric itself."""

import numpy as np

import virgil.metrics as vm
from virgil.imaging import Beam


def circular_beam(fwhm_mas):
    return Beam(major_mas=fwhm_mas, minor_mas=fwhm_mas, pa_deg=0.0)


def beam(major_mas, minor_mas, pa_deg):
    return Beam(major_mas=major_mas, minor_mas=minor_mas, pa_deg=pa_deg)


def ncc(image, truth):
    return float(vm.ncc(image, truth))


def l1_score(image, truth):
    return float(vm.l1_score(image, truth))


def lawson(image, truth):
    return float(vm.lawson_sigma_over_peak(image, truth))


def rms_convolved(image, truth, pixel, beam=None, relative=False):
    return float(vm.rms_convolved(image, truth, pixel, beam, relative=relative))


def resample(image, pixel, npix, new_pixel):
    return np.asarray(vm.resample(image, pixel, npix, new_pixel))


def align(image, truth, max_shift_mas, pixel):
    shifted, shift = vm.align(image, truth, max_shift_mas, pixel)
    return np.asarray(shifted), tuple(float(s) for s in shift)


def score(image, truth, pixel, **kwargs):
    return vm.score(image, truth, pixel_scale_mas=pixel, **kwargs)
