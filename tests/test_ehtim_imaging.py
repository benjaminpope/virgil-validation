"""virgil's image model and closure-phase data term against eht-imaging.

eht-imaging (Chael et al. 2016, 2018) 1.3.2 runs in its own environment
(scripts/setup_external.sh) through src/external_bridge/ehtim_worker.py.
It has no OIFITS reader, so the worker is given arrays read from our files
with astropy.

Conventions (pinned below):

* eht-imaging's direct transform (obs_helpers.ftmatrix) is
  sum b exp(+2 pi i (u x + v y)), with row 0 of the image North and column 0
  East, as in virgil. With the delta-function pixel response, virgil's
  visibility of an Image is the complex conjugate of eht-imaging's, which
  for a real image is eht-imaging's at (-u, -v); flipped images do not
  match.
* eht-imaging's closure-phase chi-squared, (2/N) sum (1 - cos Δ) / σ², is
  the chord χ² (2 sin(Δ/2))² / σ² that virgil uses, divided by N. On a
  three-telescope file (no correlated closure phases) N times it must equal
  virgil's closure-phase part exactly.
"""

import numpy as np
import pytest

from crosscheck import chi2 as ours, simulate, sky
from evidence.plugin import record
from external_bridge import _subprocess as sp

vm = pytest.importorskip("virgil.models")
from virgil.likelihood import whitened_residuals  # noqa: E402
from virgil.oidata import OIData  # noqa: E402

pytestmark = [
    pytest.mark.x64,
    pytest.mark.external,
    pytest.mark.skipif(not sp.available("ehtim"), reason="eht-imaging not installed (scripts/setup_external.sh)"),
]

MAS = np.pi / 180 / 3600 / 1000
N, PIXEL = 16, 0.7
UTS3 = np.array([[-9.925, -20.335], [14.887, 30.502], [103.306, 43.999]])


def random_image(seed, n=N, scale=PIXEL):
    return vm.Image(np.random.default_rng(seed).normal(size=(n, n)), scale)


def ehtim_ft(b, scale, u, v, wl):
    res = sp.run("ehtim", "ehtim_worker.py", {"task": "ft", "image": np.asarray(b).tolist(), "pdim": scale * MAS,
                                              "uv": np.c_[u / wl, v / wl].tolist()})
    return np.array(res["re"]) + 1j * np.array(res["im"])


@pytest.mark.validates("virgil.models.Image", roots=["ehtim"])
def test_image_visibilities_match_ehtims_transform():
    rng = np.random.default_rng(1)
    u, v = rng.uniform(-80, 80, (2, 40))
    wl = 2.0e-6
    worst = 0.0
    for seed, n, scale in [(1, 16, 0.7), (2, 15, 1.3), (3, 20, 0.4)]:  # odd and even sizes
        image = random_image(seed, n, scale)
        b = np.asarray(image.brightness)
        got = np.asarray(image.model(u, v, wl))
        worst = max(worst, np.max(np.abs(got - np.conj(ehtim_ft(b, scale, u, v, wl)))))
    record("max_abs_dV", worst)
    assert worst < 1e-13


@pytest.mark.validates("virgil.models.Image", roots=["ehtim"], kind="control")
def test_flipped_images_do_not_match():
    """The orientation is pinned: a flipped image, or the visibility
    without the conjugate, differs."""
    rng = np.random.default_rng(1)
    u, v = rng.uniform(-80, 80, (2, 40))
    wl = 2.0e-6
    image = random_image(1)
    b = np.asarray(image.brightness)
    got = np.asarray(image.model(u, v, wl))
    assert np.max(np.abs(got - ehtim_ft(b, PIXEL, u, v, wl))) > 1e-2
    for flipped in (b[::-1], b[:, ::-1]):
        assert np.max(np.abs(got - np.conj(ehtim_ft(flipped, PIXEL, u, v, wl)))) > 1e-2


@pytest.mark.validates("virgil.likelihood.whitened_residuals", "virgil.models.Image", roots=["ehtim"])
def test_closure_phase_chi2_matches_ehtim(tmp_path):
    """virgil's closure-phase χ² of random images against eht-imaging's
    chisq_cphase on the same three-telescope file (2° closure-phase noise,
    so residuals are large and the chord matters)."""
    path = tmp_path / "im3.fits"

    def scene(u, v, w):
        return (sky.vis_gaussian(u, v, w, 3.0) + 0.3 * sky.vis_point(u, v, w, 4.0, -2.0)) / 1.3

    simulate.observe(path, scene, UTS3, hour_angles_h=np.linspace(-3, 3, 7), wavelengths=np.linspace(1.5e-6, 2.4e-6, 4),
                     dec_deg=-50.0, sigma_v2=0.01, sigma_cp_deg=2.0, rng=np.random.default_rng(3))
    d = ours.load(path)
    data = OIData(str(path))
    w = d["wl3"].ravel()
    worst = 0.0
    for seed in (5, 6, 7):
        image = random_image(seed, 24, 0.6)
        b = np.asarray(image.brightness)
        res = sp.run("ehtim", "ehtim_worker.py", {
            "task": "chisq_cphase", "image": b.tolist(), "pdim": 0.6 * MAS,
            "uv1": np.c_[d["u1"].ravel() / w, d["v1"].ravel() / w].tolist(),
            "uv2": np.c_[d["u2"].ravel() / w, d["v2_"].ravel() / w].tolist(),
            "cp_deg": np.rad2deg(d["cp"]).ravel().tolist(), "sigma_deg": np.rad2deg(d["dcp"]).ravel().tolist(),
        })
        want = res["chisq"] * res["n"]
        r = np.asarray(whitened_residuals(image, data))
        got = float(np.sum(r[d["v2"].size:] ** 2))
        assert res["n"] == r.size - d["v2"].size
        worst = max(worst, abs(got / want - 1))
    record("max_rel_chi2", worst)
    assert worst < 1e-10


@pytest.mark.validates("ehtim", roots=["mathematics"], kind="reference")
def test_ehtims_transform_of_a_gaussian_is_analytic():
    """eht-imaging's direct transform of a sampled circular Gaussian
    (sigma 2 mas on 0.25 mas pixels, centred 1.5 mas East and 0.75 mas
    North of the image centre), divided by the sum of the pixels, against
    the continuous transform exp(-2 pi^2 sigma^2 |q|^2) exp(+2 pi i q.x0)
    in its documented sign. Sampling errors (aliases at 1 / pixel,
    exp(-2 pi^2 (sigma / pixel)^2)) are below 1e-200 and truncation errors
    (the field reaches over 9 sigma from the Gaussian's centre) below
    1e-19, so they agree to rounding."""
    n, scale, sigma, east, north = 160, 0.25, 2.0, 1.5, 0.75
    offsets = ((n - 1) / 2 - np.arange(n)) * scale  # row 0 North, column 0 East
    ee, nn = np.meshgrid(offsets, offsets)
    b = np.exp(-((ee - east) ** 2 + (nn - north) ** 2) / (2 * sigma**2))
    rng = np.random.default_rng(2)
    wl = 2.0e-6
    u, v = rng.uniform(-60, 60, (2, 40))  # metres; |q| up to 0.4 / sigma
    got = ehtim_ft(b, scale, u, v, wl) / b.sum()
    qu, qv = u / wl * MAS, v / wl * MAS  # cycles per mas
    want = np.exp(-2 * np.pi**2 * sigma**2 * (qu**2 + qv**2)) * np.exp(2j * np.pi * (qu * east + qv * north))
    record("max_abs_dV", np.max(np.abs(got - want)))
    assert np.max(np.abs(got - want)) < 1e-12
    assert np.max(np.abs(got - np.conj(want))) > 1e-2  # the sign is pinned, not free
