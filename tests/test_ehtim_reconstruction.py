"""Regularised image reconstruction: virgil against eht-imaging's imager.

The same simulated data go to both codes: noisy complex visibilities
(crosscheck.simulate.observe_visibilities), written for virgil as
amplitudes (OI_VIS VISAMP) and closure phases from the same visibilities,
and given to eht-imaging (1.3.2, its own environment) as its observation
table. With the weights below the two objectives are the same function of
the image, up to two differences of definition at the edges:

* data: ½ χ²(amplitudes) + ½ χ²(closure phases). eht-imaging's cost is
  Σ α_d (χ²_d / N_d − 1), so α_d = N_d / 2; its closure-phase term is the
  chord one, as virgil's (tests/test_ehtim_imaging.py).
* regulariser: w Σ (Δb)², virgil's TSV(w) and eht-imaging's 'tv2' with
  β = w, unnormalised. virgil also counts the step into the image at the
  first row and column; the scene is compact, so the edge pixels are
  nearly empty.
* flux: virgil's image sums to 1 exactly (a softmax); eht-imaging holds it
  near 1 with a 'flux' term.

The objective is not convex. From the same smooth start the two codes,
whose parameterisations differ (softmax against log pixels), can stop in
different local minima, and which one finds the lower depends on the data
(both ways round were seen while building this test). The comparison is
therefore made in one basin: virgil is started from eht-imaging's image,
and must reach an objective at least as low, with nearly the same image.
"""

import numpy as np
import pytest

from crosscheck import simulate, sky
from evidence.plugin import record
from external_bridge import _subprocess as sp

vm = pytest.importorskip("virgil.models")
vi = pytest.importorskip("virgil.imaging")
from virgil.fitting import fit  # noqa: E402
from virgil.likelihood import whitened_residuals  # noqa: E402
from virgil.oidata import OIData  # noqa: E402

pytestmark = [
    pytest.mark.x64,
    pytest.mark.external,
    pytest.mark.skipif(not sp.available("ehtim"), reason="eht-imaging not installed (scripts/setup_external.sh)"),
]

MAS = np.pi / 180 / 3600 / 1000
UTS3 = np.array([[-9.925, -20.335], [14.887, 30.502], [103.306, 43.999]])
N, PIXEL, SIGMA, WEIGHT = 24, 0.5, 0.01, 3e4


def scene(u, v, w):
    return (sky.vis_gaussian(u, v, w, 2.0) + 0.4 * sky.vis_gaussian(u, v, w, 1.5, dra=3.0, ddec=-2.0)) / 1.4


@pytest.fixture(scope="module")
def problem(tmp_path_factory):
    return build_problem(tmp_path_factory.mktemp("recon") / "amp.fits")


def build_problem(path):
    sim = simulate.observe_visibilities(
        path, scene, UTS3, hour_angles_h=np.linspace(-4, 4, 13), wavelengths=np.linspace(1.5e-6, 2.4e-6, 6),
        dec_deg=-50.0, sigma=SIGMA, rng=np.random.default_rng(1),
    )
    data = OIData(str(path))
    # eht-imaging: one time per snapshot and channel (triangles close within
    # a channel); its transform has the opposite sign, so (u, v) -> -(u, v)
    epochs = list(np.unique(sim["mjd"]))
    nw = sim["vis"].shape[1]
    names = {1: "A", 2: "B", 3: "C"}
    rows = []
    for k in range(sim["vis"].shape[0]):
        for c in range(nw):
            vis = sim["vis"][k, c]
            rows.append([
                float(epochs.index(sim["mjd"][k, c]) * nw + c), names[sim["sta"][k, 0]], names[sim["sta"][k, 1]],
                float(-sim["u"][k, c] / sim["wl"][k, c]), float(-sim["v"][k, c] / sim["wl"][k, c]),
                float(vis.real), float(vis.imag), SIGMA,
            ])
    x = (np.arange(N) - (N - 1) / 2) * PIXEL
    east, north = np.meshgrid(-x, x[::-1])  # row 0 North, column 0 East
    start = np.exp(-(east**2 + north**2) / (2 * 3.0**2))
    start /= start.sum()
    n_cp = np.asarray(data.phi).size
    return data, rows, start, len(rows), n_cp


def virgil_loss(b, data):
    image = vm.Image(np.log(np.maximum(b, 1e-300)), PIXEL)
    return 0.5 * float(np.sum(np.asarray(whitened_residuals(image, data)) ** 2)) + float(vi.TSV(WEIGHT).value(image))


def ehtim(rows, start, n_amp, n_cp, score=()):
    res = sp.run("ehtim", "ehtim_worker.py", {
        "task": "reconstruct", "rows": rows, "pdim": PIXEL * MAS, "init": np.asarray(start).tolist(),
        "alpha_amp": n_amp / 2, "alpha_cphase": n_cp / 2, "beta_tv2": WEIGHT, "maxit": 3000,
        "score": [np.asarray(s).tolist() for s in score],
    })
    image = np.array(res["image"])
    return image / image.sum(), res


def virgil_fit(start, data):
    template = vm.Image(np.log(np.maximum(start, 1e-300)), PIXEL)
    result = fit(template, vi.image_priors(template), data, regularisers=(vi.TSV(WEIGHT),))
    return np.asarray(result.model.brightness)


@pytest.mark.validates(
    "virgil.fitting.fit", "virgil.imaging.TSV", "virgil.models.Image", "pipeline:rml-imaging",
    roots=["ehtim"],
)
def test_reconstruction_matches_ehtim_in_one_basin(problem):
    data, rows, start, n_amp, n_cp = problem
    image_e, res = ehtim(rows, start, n_amp, n_cp)
    assert res["n_amp"] == n_amp and res["n_cphase"] == n_cp
    image_v = virgil_fit(image_e, data)
    loss_e, loss_v = virgil_loss(image_e, data), virgil_loss(image_v, data)
    rel = np.linalg.norm(image_v - image_e) / np.linalg.norm(image_e)
    corr = np.corrcoef(image_v.ravel(), image_e.ravel())[0, 1]
    record("virgil_loss_at_ehtim_image", loss_e)
    record("virgil_loss_at_virgil_image", loss_v)
    record("rel_l2_image_difference", rel)
    record("image_correlation", corr)
    # the problem is stiff (a large gradient, yet any step along it raises
    # the loss), so agreement is judged by the loss and the image: virgil,
    # started at eht-imaging's minimum, goes at least as low, by very little,
    # and stays put
    assert loss_v <= loss_e * (1 + 1e-9)
    assert (loss_e - loss_v) / loss_e < 1e-6
    assert rel < 1e-4
    assert corr > 1 - 1e-6
