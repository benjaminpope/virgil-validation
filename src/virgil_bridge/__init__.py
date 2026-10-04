"""The only place that imports virgil: scene definitions pairing our truth
(``crosscheck.sky``) with the virgil model we fit, and the fitting helpers.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import numpyro.distributions as dist

import virgil.models as vm
from virgil.fitting import fit
from virgil.inference import laplace_cov
from virgil.oidata import OIData

from crosscheck import limb, sky


@dataclass
class Scene:
    name: str
    vis: Callable  # our truth: vis(u, v, wavel) -> complex
    template: object  # virgil model at the truth
    truth: dict  # path -> value
    priors: dict  # path -> numpyro distribution
    start: dict  # path -> starting value for the fit
    cloud: object = field(default=None)  # our point cloud, for dLux


def _u(lo, hi):
    return dist.Uniform(lo, hi)


def binary(sep=6.0, pa=124.0, flux=0.05):
    dra, ddec = sep * np.sin(np.deg2rad(pa)), sep * np.cos(np.deg2rad(pa))

    def vis(u, v, w):
        return (sky.vis_point(u, v, w) + flux * sky.vis_point(u, v, w, dra, ddec)) / (1 + flux)

    return Scene(
        "binary",
        vis,
        vm.BinaryModelCartesian(dra, ddec, flux),
        {"dra": dra, "ddec": ddec, "flux": flux},
        {"dra": _u(-10 * sep, 10 * sep), "ddec": _u(-10 * sep, 10 * sep), "flux": _u(0, 1)},
        {"dra": dra + 0.05 * sep, "ddec": ddec - 0.05 * sep, "flux": flux * 0.7},
        sky.mix([sky.point(), sky.point(dra, ddec)], [1.0, flux]),
    )


def resolved_star_companion(diam=1.8, dra=-12.0, ddec=8.0, flux=0.02):
    def vis(u, v, w):
        return (sky.vis_uniform_disk(u, v, w, diam) + flux * sky.vis_point(u, v, w, dra, ddec)) / (1 + flux)

    r = np.hypot(dra, ddec)
    template = vm.System(
        star=vm.UniformDisk(diam), comp=vm.PointSource(flux, dra, ddec)
    )
    truth = {"star.diam": diam, "comp.dra": dra, "comp.ddec": ddec, "comp.flux": flux}
    return Scene(
        "disk star + companion",
        vis,
        template,
        truth,
        {"star.diam": _u(0, 5 * diam), "comp.dra": _u(-10 * r, 10 * r), "comp.ddec": _u(-10 * r, 10 * r), "comp.flux": _u(0, 1)},
        {"star.diam": 0.85 * diam, "comp.dra": dra + 0.03 * r, "comp.ddec": ddec + 0.02 * r, "comp.flux": 0.75 * flux},
        sky.mix([sky.uniform_disk(diam, n_r=24, n_theta=64), sky.point(dra, ddec)], [1.0, flux]),
    )


def star_envelope(fwhm=4.0, ratio=0.5, pa=60.0, flux=0.4):
    def vis(u, v, w):
        return (sky.vis_point(u, v, w) + flux * sky.vis_elliptical_gaussian(u, v, w, fwhm, ratio, pa)) / (1 + flux)

    template = vm.System(
        star=vm.PointSource(), env=vm.EllipticalGaussian(fwhm, ratio, pa, flux)
    )
    truth = {"env.fwhm": fwhm, "env.ratio": ratio, "env.pa": pa, "env.flux": flux}
    return Scene(
        "star + elliptical envelope",
        vis,
        template,
        truth,
        {"env.fwhm": _u(0.02 * fwhm, 5 * fwhm), "env.ratio": _u(0.05, 1), "env.pa": _u(0, 180), "env.flux": _u(0, 5)},
        {"env.fwhm": 0.9 * fwhm, "env.ratio": 0.6, "env.pa": pa - 10.0, "env.flux": 0.75 * flux},
        sky.mix([sky.point(), sky.elliptical_gaussian(fwhm, ratio, pa, n=16)], [1.0, flux]),
    )


def star_rim(diam=6.0, fwhm=1.0, inc=45.0, pa=30.0, amp=0.5, az_pa=120.0, flux=0.8):
    ring = sky.inclined_ring(diam, inc, pa, (amp,), (az_pa,), "disk")

    def vis(u, v, w):
        # virgil >= #139 blurs the rim isotropically in its own plane
        rim = sky.visibility(ring, u, v, w) * sky.in_plane_blur_factor(u, v, w, fwhm, inc, pa)
        return (sky.vis_point(u, v, w) + flux * rim) / (1 + flux)

    template = vm.System(
        star=vm.PointSource(),
        rim=vm.ModulatedGaussianRim(
            diam, fwhm, inc, pa, np.array([amp]), np.array([az_pa]), flux
        ),
    )
    truth = {
        "rim.diam": diam, "rim.fwhm": fwhm, "rim.inc": inc, "rim.pa": pa,
        "rim.az_amps": np.array([amp]), "rim.az_pas": np.array([az_pa]),
        "rim.flux": flux,
    }
    priors = {
        "rim.diam": _u(0.2 * diam, 3 * diam), "rim.fwhm": _u(0.01 * diam, diam), "rim.inc": _u(0, 85),
        # array-shaped bounds; .expand([1]) priors now get LM too (finding 7,
        # fixed in virgil#142, checked in tests/test_vlti.py)
        "rim.pa": _u(-90, 180), "rim.az_amps": _u(np.zeros(1), np.ones(1)),
        "rim.az_pas": _u(np.zeros(1), np.full(1, 360.0)), "rim.flux": _u(0, 5),
    }
    start = {
        "rim.diam": 0.92 * diam, "rim.fwhm": 1.2 * fwhm, "rim.inc": inc - 5.0,
        "rim.pa": pa + 5.0, "rim.az_amps": np.array([0.8 * amp]),
        "rim.az_pas": np.array([az_pa - 10.0]), "rim.flux": 0.85 * flux,
    }
    cloud_ring = sky.blurred(
        sky.inclined_ring(diam, inc, pa, (amp,), (az_pa,), "disk", n_phi=128),
        fwhm,
        n=5,
        inc=inc,
        pa=pa,
    )
    return Scene(
        "star + modulated rim", vis, template, truth, priors, start,
        sky.mix([sky.point(), cloud_ring], [1.0, flux]),
    )


def limb_darkened_star_companion(
    diam=6.0, q1=0.36, q2=0.29, dra=0.7, ddec=-0.4, comp=(-14.0, 9.0), flux=0.03
):
    """An off-centre star with Kipping-parametrized quadratic limb darkening,
    large enough that VLTI baselines reach its second lobe, and a companion.
    The star's offset is held fixed; its diameter and q1, q2 are fitted."""
    u1, u2 = limb.kipping_quadratic_u(q1, q2)
    profile = limb.polynomial([u1, u2])
    star = limb.disk(diam, profile, dra, ddec)

    def vis(u, v, w):
        return (sky.visibility(star, u, v, w) + flux * sky.vis_point(u, v, w, *comp)) / (1 + flux)

    r = np.hypot(*comp)
    template = vm.System(
        star=vm.QuadraticLimbDarkenedDisk(diam, q1, q2, dra=dra, ddec=ddec),
        comp=vm.PointSource(flux, *comp),
    )
    truth = {
        "star.diam": diam, "star.q1": q1, "star.q2": q2,
        "comp.dra": comp[0], "comp.ddec": comp[1], "comp.flux": flux,
    }
    return Scene(
        "limb-darkened star + companion",
        vis,
        template,
        truth,
        {
            "star.diam": _u(0, 3 * diam), "star.q1": _u(0, 1), "star.q2": _u(0, 1),
            "comp.dra": _u(-10 * r, 10 * r), "comp.ddec": _u(-10 * r, 10 * r), "comp.flux": _u(0, 1),
        },
        {
            "star.diam": 0.95 * diam, "star.q1": 0.5, "star.q2": 0.5,
            "comp.dra": comp[0] + 0.03 * r, "comp.ddec": comp[1] - 0.02 * r, "comp.flux": 0.75 * flux,
        },
        sky.mix([limb.disk(diam, profile, dra, ddec, n_t=24, n_theta=64), sky.point(*comp)], [1.0, flux]),
    )


SCENES = [binary, resolved_star_companion, star_envelope, star_rim]
# Not in SCENES: the noisy-pulls test (sigma_v2 = 0.02) assumes a Gaussian
# Laplace posterior, which weakly constrained q1, q2 near the unit square's
# edges do not give. tests/test_limb_darkening.py runs its own end-to-end
# checks on it.
LIMB_SCENES = [limb_darkened_star_companion]


def masking_scenes():
    """The same scenes at aperture-masking scales (lambda / B ~ 190 mas at
    4.8 um on a 5.3 m mask)."""
    return [
        binary(sep=150.0, pa=124.0, flux=0.05),
        resolved_star_companion(diam=90.0, dra=-180.0, ddec=120.0, flux=0.03),
        star_envelope(fwhm=160.0, ratio=0.5, pa=60.0, flux=0.4),
        star_rim(diam=240.0, fwhm=40.0, inc=45.0, pa=30.0, amp=0.5, az_pa=120.0, flux=0.8),
    ]


def fit_scene(scene, data, start=None):
    """MAP fit from ``start`` (default ``scene.start``). Returns the
    FitResult and the Laplace covariance at the optimum, over the flattened
    parameters in the order of ``flat_truth``."""
    start = scene.start if start is None else start
    template = scene.template
    for path, value in start.items():
        template = template.set(path, np.asarray(value, float))
    result = fit(template, scene.priors, data)
    # laplace_cov takes every path's elements flattened in order, as
    # flat_values gives them (array-valued paths since virgil#135)
    cov = np.asarray(
        laplace_cov(
            flat_values(scene, result.values), list(scene.truth), data,
            result.model,
        )
    )
    return result, cov


def flat_truth(scene):
    return np.concatenate([np.atleast_1d(np.asarray(scene.truth[p], float)) for p in scene.truth])


def flat_values(scene, values):
    return np.concatenate([np.atleast_1d(np.asarray(values[p], float)) for p in scene.truth])


def load(path):
    return OIData(str(path))
