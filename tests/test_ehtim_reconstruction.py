"""Regularised image reconstruction: virgil against eht-imaging's imager.

The same simulated data go to both codes: noisy complex visibilities
(crosscheck.simulate.observe_visibilities), written for virgil as
amplitudes (OI_VIS VISAMP) and closure phases from the same visibilities,
and given to eht-imaging (1.3.2, its own environment) as its observation
table. With the weights below the two objectives are the same function of
the image, up to a constant and two differences of definition:

* data: ½ χ²(amplitudes) + ½ χ²(closure phases). eht-imaging's cost is
  Σ α_d (χ²_d / N_d − 1), so α_d = N_d / 2; its closure-phase term is the
  chord one, as virgil's (tests/test_ehtim_imaging.py).
* regulariser: w Σ (Δb)², virgil's TSV(w) and eht-imaging's 'tv2' with
  β = w, unnormalised. Both pad the image with zeros, but eht-imaging
  counts only the steps out of the last row and column, virgil the steps
  across all four edges: TSV(w) = w (tv2 + Σ b[0, :]² + Σ b[:, 0]²)
  (edge_sum below; row 0 North, column 0 East in both).
* flux: virgil's image sums to 1 exactly (a softmax); eht-imaging holds it
  at 1 with a 'flux' term, whose weight is raised until it does.

test_objective_matches_ehtim checks this mapping on random images: values
to 1e-12 and gradients to 1e-9. The checks then need a problem whose
minima are smooth. On the faint scene used by the regression tests,
eht-imaging's minimum puts model visibilities exactly on nulls of the
longest baselines (where the data are at ~1 sigma); the closure phase is
undefined there, and so is the gradient of either code's objective. On a
brighter scene (every |V| >= 0.06 at sigma = 0.002), eht-imaging's
minimum, converged with restarts, is a stationary point of virgil's
objective to eht-imaging's own convergence, and with the regulariser's
weight doubled or halved it is not (controls).
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


def build_problem(path, vis_fn=scene, sigma=SIGMA):
    sim = simulate.observe_visibilities(
        path, vis_fn, UTS3, hour_angles_h=np.linspace(-4, 4, 13), wavelengths=np.linspace(1.5e-6, 2.4e-6, 6),
        dec_deg=-50.0, sigma=sigma, rng=np.random.default_rng(1),
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
                float(vis.real), float(vis.imag), sigma,
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


@pytest.fixture(scope="module")
def ehtim_image(problem):
    data, rows, start, n_amp, n_cp = problem
    image_e, res = ehtim(rows, start, n_amp, n_cp)
    assert res["n_amp"] == n_amp and res["n_cphase"] == n_cp
    return image_e


def _gradients(b, data, weight):
    """Gradients of virgil's data term and regulariser with respect to the
    image's logits (zero-sum: the softmax keeps the flux at 1)."""
    import jax
    import jax.numpy as jnp

    def data_term(logits):
        return 0.5 * jnp.sum(whitened_residuals(vm.Image(logits, PIXEL), data) ** 2)

    def reg_term(logits):
        return vi.TSV(weight).value(vm.Image(logits, PIXEL))

    logits = jnp.log(jnp.maximum(jnp.asarray(b), 1e-300))
    return np.asarray(jax.grad(data_term)(logits)), np.asarray(jax.grad(reg_term)(logits))


@pytest.mark.validates(
    "virgil.imaging.TSV", "virgil.models.Image", "virgil.likelihood.whitened_residuals", "pipeline:rml-imaging",
    roots=["ehtim"], kind="regression",
)
def test_ehtim_minimum_on_the_faint_scene(problem, ehtim_image):
    """Recorded, not a check: on this scene the data are at about 1 sigma
    on the longest baselines, and eht-imaging's minimum puts model
    visibilities exactly on nulls there (|V| ~ 1e-10), where the closure
    phase is undefined and the objective has no gradient (eht-imaging's own
    gradient fails a finite-difference test at that image). Neither code
    can be stationary there; the check is
    test_ehtim_minimum_is_stationary_on_a_well_posed_problem."""
    data = problem[0]
    g_data, g_reg = _gradients(ehtim_image, data, WEIGHT)
    scale = min(np.linalg.norm(g_data), np.linalg.norm(g_reg))
    stationary = np.linalg.norm(g_data + g_reg) / scale
    record("rel_gradient_at_ehtim_minimum", stationary)
    _, g_reg2 = _gradients(ehtim_image, data, 2 * WEIGHT)
    doubled = np.linalg.norm(g_data + g_reg2) / scale
    record("rel_gradient_with_weight_doubled", doubled)
    assert scale > 0 and np.isfinite(stationary) and np.isfinite(doubled)


# A well-posed problem for the checks: every visibility is far from a null
# (true |V| >= 0.06 at sigma = 0.002), so the objective is smooth near its
# minima. The weight is scaled with the data's precision (1/sigma^2).
SIGMA_HI, WEIGHT_HI = 0.002, 7.5e5


def bright_scene(u, v, w):
    return (sky.vis_gaussian(u, v, w, 0.8) + 0.3 * sky.vis_gaussian(u, v, w, 0.8, dra=3.0, ddec=-2.0)) / 1.3


@pytest.fixture(scope="module")
def well_posed(tmp_path_factory):
    return build_problem(tmp_path_factory.mktemp("recon_hi") / "amp.fits", bright_scene, SIGMA_HI)


def edge_sum(b):
    """The steps virgil's TSV counts and eht-imaging's tv2 does not: into
    the image across its first row and first column (row 0 North, column
    0 East; the corner pixel is counted once for each). eht-imaging's
    stv2 pads with zeros but counts only the steps out of the last row
    and column."""
    return b[0, :] @ b[0, :] + b[:, 0] @ b[:, 0]


def _logit_gradients(b, data, weight):
    """Gradients with respect to virgil's logits of its data term and of
    TSV(weight) minus weight * edge_sum, i.e. of the regulariser
    eht-imaging minimises. Softmax logits: already projected onto images
    of unit flux."""
    import jax
    import jax.numpy as jnp

    def data_term(z):
        return 0.5 * jnp.sum(whitened_residuals(vm.Image(z, PIXEL), data) ** 2)

    def reg_term(z):
        image = vm.Image(z, PIXEL)
        return vi.TSV(weight).value(image) - weight * edge_sum(image.brightness)

    z = jnp.log(jnp.asarray(b))
    return np.asarray(jax.grad(data_term)(z)), np.asarray(jax.grad(reg_term)(z))


def ehtim_objective(rows, images, n_amp, n_cp, weight):
    return sp.run("ehtim", "ehtim_worker.py", {
        "task": "objective", "rows": rows, "pdim": PIXEL * MAS, "images": [np.asarray(b).tolist() for b in images],
        "alpha_amp": n_amp / 2, "alpha_cphase": n_cp / 2, "beta_tv2": weight, "flux_weight": 0.0,
    })


@pytest.mark.validates(
    "virgil.imaging.TSV", "virgil.models.Image", "virgil.likelihood.whitened_residuals", roots=["ehtim"],
)
def test_objective_matches_ehtim(well_posed):
    """The mapping, term by term, on random images of unit flux.

    eht-imaging (imager_utils): chi2_amp = (1/N_a) sum ((a - |A b|)/sigma)^2;
    chi2_cphase = (2/N_c) sum (1 - cos(phi - phi_model))/sigma^2; cost =
    sum_d alpha_d (chi2_d - 1) + beta (-stv2) + flux term. virgil:
    amplitude residuals (|V| - a)/sigma and closure-phase chords
    2 sin(delta/2)/sigma, whose square is 2(1 - cos delta)/sigma^2. So with
    alpha_d = N_d/2, each data term is half virgil's sum of squares, minus
    the constant alpha_d; and TSV(w) = w (-stv2 + edge_sum). Values to
    1e-12, and the gradients with respect to virgil's logits (eht-imaging's
    hand-written log-pixel gradient, projected for the softmax) to 1e-9."""
    data, rows, start, n_amp, n_cp = well_posed
    rng = np.random.default_rng(3)
    images = [start] + [rng.uniform(0.1, 1, (N, N)) for _ in range(2)] + [start * np.exp(0.5 * rng.standard_normal((N, N)))]
    images = [b / b.sum() for b in images]
    res = ehtim_objective(rows, images, n_amp, n_cp, WEIGHT_HI)
    assert (res["n_amp"], res["n_cphase"]) == (n_amp, n_cp)
    worst_value = worst_grad = 0.0
    for b, r in zip(images, res["results"]):
        w = np.asarray(whitened_residuals(vm.Image(np.log(b), PIXEL), data))
        amp, cp = 0.5 * np.sum(w[:n_amp] ** 2), 0.5 * np.sum(w[n_amp:] ** 2)
        tsv = float(vi.TSV(WEIGHT_HI).value(vm.Image(np.log(b), PIXEL)))
        pairs = [(amp, n_amp / 2 * r["chi2_amp"]), (cp, n_cp / 2 * r["chi2_cphase"]),
                 (tsv, WEIGHT_HI * (r["tv2"] + edge_sum(b))),
                 (amp + cp + tsv - WEIGHT_HI * edge_sum(b), r["cost"] + n_amp / 2 + n_cp / 2)]
        worst_value = max(worst_value, *(abs(x / y - 1) for x, y in pairs))
        g_data, g_reg = _logit_gradients(b, data, WEIGHT_HI)
        g = np.array(r["grad_log"])  # b * dcost/db
        g_eht = g - b * g.sum()  # softmax projection: b * (dcost/db - b . dcost/db)
        worst_grad = max(worst_grad, np.linalg.norm(g_data + g_reg - g_eht) / np.linalg.norm(g_eht))
    record("worst_rel_value_difference", worst_value)
    record("worst_rel_gradient_difference", worst_grad)
    assert worst_value < 1e-12
    assert worst_grad < 1e-9


@pytest.fixture(scope="module")
def ehtim_converged(well_posed):
    """eht-imaging's minimum for the TSV weights w and 2w, each run to
    convergence: the flux term's weight raised 1e4 -> 1e10 so that the
    total flux is 1 (to ~1e-6), restarting L-BFGS-B at each stage until the
    cost stops falling."""
    data, rows, start, n_amp, n_cp = well_posed
    out = {}
    for weight in (WEIGHT_HI, 2 * WEIGHT_HI):
        res = sp.run("ehtim", "ehtim_worker.py", {
            "task": "reconstruct", "rows": rows, "pdim": PIXEL * MAS, "init": start.tolist(),
            "alpha_amp": n_amp / 2, "alpha_cphase": n_cp / 2, "beta_tv2": weight, "maxit": 5000, "stop": 1e-16,
            "flux_weights": [1e4, 1e6, 1e8, 1e10], "max_restarts": 5,
        })
        out[weight] = res
    return out


@pytest.mark.validates(
    "virgil.imaging.TSV", "virgil.models.Image", "virgil.likelihood.whitened_residuals", "pipeline:rml-imaging",
    roots=["ehtim"],
)
@pytest.mark.parametrize("factor", [1, 2])
def test_ehtim_minimum_is_stationary_on_a_well_posed_problem(well_posed, ehtim_converged, factor):
    """eht-imaging's minimum for weight w is a stationary point of virgil's
    objective (with the edge_sum correction) for weight w, and not for 2w or
    w/2 (controls; at a stationary point the gradient left over is then
    exactly 1 or 1/2 of the data gradient). The measure is the gradient
    with respect to virgil's logits, relative to the data term's; its floor
    is eht-imaging's own convergence (L-BFGS-B stops with a log-pixel
    gradient ~1e-3 of the data term's), which is recorded alongside."""
    data, rows, _, n_amp, n_cp = well_posed
    weight = factor * WEIGHT_HI
    res = ehtim_converged[weight]
    b = np.array(res["image"])
    b = b / b.sum()
    g_data, g_reg = _logit_gradients(b, data, weight)
    scale = np.linalg.norm(g_data)
    rel = {c: np.linalg.norm(g_data + c * g_reg) / scale for c in (1.0, 2.0, 0.5)}
    record("flux_minus_one", res["flux"] - 1)
    record("ehtim_rel_log_gradient", res["grad_log_norm"] / scale)
    record("rel_gradient_at_ehtim_minimum", rel[1.0])
    record("rel_gradient_with_weight_doubled", rel[2.0])
    record("rel_gradient_with_weight_halved", rel[0.5])
    assert abs(res["flux"] - 1) < 1e-5
    assert rel[1.0] < 5e-3
    assert abs(rel[2.0] - 1) < 0.05 and abs(rel[0.5] - 0.5) < 0.05


@pytest.mark.validates("virgil.fitting.fit", "virgil.imaging.TSV", "virgil.models.Image", roots=["ehtim"], kind="regression")
def test_both_codes_from_the_common_start(problem, ehtim_image):
    """Both from the smooth start: the losses and the images, recorded."""
    data, _, start, _, _ = problem
    image_v = virgil_fit(start, data)
    record("virgil_loss_at_ehtim_image", virgil_loss(ehtim_image, data))
    record("virgil_loss_at_virgil_image", virgil_loss(image_v, data))
    record("image_correlation", np.corrcoef(image_v.ravel(), ehtim_image.ravel())[0, 1])
    assert np.all(np.isfinite(image_v))


@pytest.mark.validates("virgil.fitting.fit", "virgil.imaging.TSV", "virgil.models.Image", roots=["ehtim"], kind="regression")
def test_reconstruction_matches_ehtim_in_one_basin(problem, ehtim_image):
    data = problem[0]
    image_e = ehtim_image
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


@pytest.mark.xfail(strict=True, reason="P6: eht-imaging's tlist returns tuples under NumPy 2 for equal-sized time groups")
@pytest.mark.validates("ehtim", roots=["mathematics"], kind="upstream")
def test_p6_closure_phases_with_equal_time_groups():
    """Unpatched eht-imaging: three sites at two times (equal-sized time
    groups) should give two closure phases. Under NumPy 2 it raises
    TypeError instead; the reconstruction above works around it."""
    res = sp.run("ehtim", "ehtim_worker.py", {"task": "p6_probe"})
    assert res["ok"], res["error"]
    assert res["n_cphase"] == 2

