"""virgil's image regularisers against eht-imaging, MPoL and mathematics.

eht-imaging (Chael et al. 2016, 2018; ehtim.imaging.imager_utils) and MPoL
(Czekala et al.; mpol.losses) run in their own environments
(scripts/setup_external.sh) through src/external_bridge. Their definitions
differ from virgil's at the image edges and in normalisation; the mapping:

* virgil pads the image with zeros and counts the steps into and out of it
  at every edge; eht-imaging counts forward steps including the last one
  out of the image; MPoL counts interior steps only. On images whose outer
  ring of pixels is zero (virgil's ``support``), all three sums coincide,
  except for TV's softening, which every cell carries (virgil has
  (N + 1)², eht-imaging N², MPoL (N - 1)² cells).
* virgil's TV is the mean over the image's four flips of a forward-
  difference TV; eht-imaging's and MPoL's are single forward-difference
  sums, so they are averaged over the same flips here.
* virgil's TV softening is ε² with ε = epsilon / Npix; eht-imaging's and
  MPoL's softening is added unsquared.
* With their normalisation off (eht-imaging ``norm_reg=False``, MPoL
  ``tot_flux=1``), eht-imaging's regularisers are the negated penalties
  (they are maximised) and MPoL's are the penalties.

Gradients: virgil differentiates with respect to the log-brightness (the
pixels are a softmax over the support). eht-imaging's hand-written
gradients and MPoL's autograd are with respect to the pixels; the softmax
chain rule, written here, maps one to the other.

Laplacian, StarletL1 and LogSum have no counterpart in either package; they
are checked against their documented formulas, written with SciPy.
"""

import jax
import numpy as np
import pytest
from scipy import ndimage

from evidence.plugin import record
from external_bridge import _subprocess as sp

vm = pytest.importorskip("virgil.models")
vi = pytest.importorskip("virgil.imaging")

pytestmark = pytest.mark.x64

N = 24
EPS_FRACTION = 1e-2
WEIGHT = 1.7

needs_ehtim = pytest.mark.skipif(not sp.available("ehtim"), reason="eht-imaging not installed (scripts/setup_external.sh)")
needs_mpol = pytest.mark.skipif(not sp.available("mpol"), reason="MPoL not installed (scripts/setup_external.sh)")


@pytest.fixture(scope="module", params=[3, 11])
def scene(request):
    """A random image on a circular support with a zero outer ring."""
    rng = np.random.default_rng(request.param)
    support = np.asarray(vm.circular_support(N, 1.0, 9.0))
    eta = rng.normal(size=(N, N))
    image = vm.Image(eta, 1.0, support=support)
    b = np.asarray(image.brightness)
    assert b[0].sum() == b[-1].sum() == b[:, 0].sum() == b[:, -1].sum() == 0.0
    return eta, support, b


def flips(a):
    return [a, a[::-1], a[:, ::-1], a[::-1, ::-1]]


def unflip(grads):
    g0, g1, g2, g3 = (np.asarray(g) for g in grads)
    return [g0, g1[::-1], g2[:, ::-1], g3[::-1, ::-1]]


def chain(b, support, grad_b):
    """d(penalty)/d(log-brightness) from d(penalty)/d(pixels), through the
    softmax over the support: b * (g - Σ b g)."""
    g = np.asarray(grad_b)
    return np.where(support, b * (g - np.sum(b * g)), 0.0)


def virgil_value_and_grad(reg, eta, support):
    def f(x):
        return reg.value(vm.Image(x, 1.0, support=support))

    return float(f(eta)), np.asarray(jax.grad(f)(eta))


def rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(b)))


# -------------------------------------------------------------- eht-imaging


@needs_ehtim
@pytest.mark.external
@pytest.mark.validates("virgil.imaging.TSV", roots=["ehtim"])
def test_tsv_matches_ehtim(scene):
    eta, support, b = scene
    e = sp.run("ehtim", "ehtim_worker.py", {"task": "regularisers", "images": [b.tolist()],
                                            "priors": [np.ones_like(b).tolist()], "tv_epsilon": 0.0})["results"][0]
    value, grad = virgil_value_and_grad(vi.TSV(WEIGHT), eta, support)
    want = -WEIGHT * e["tv2"]
    want_grad = chain(b, support, -WEIGHT * np.asarray(e["tv2_grad"]))
    record("rel_value", abs(value / want - 1))
    record("rel_grad", rel(grad, want_grad))
    assert abs(value / want - 1) < 1e-12
    assert rel(grad, want_grad) < 1e-12


@needs_ehtim
@pytest.mark.external
@pytest.mark.validates("virgil.imaging.TV", roots=["ehtim"])
def test_tv_matches_ehtim(scene):
    eta, support, b = scene
    eps = EPS_FRACTION / b.size
    res = sp.run("ehtim", "ehtim_worker.py", {"task": "regularisers", "images": [f.tolist() for f in flips(b)],
                                              "priors": [np.ones_like(b).tolist()] * 4, "tv_epsilon": eps**2})["results"]
    value, grad = virgil_value_and_grad(vi.TV(WEIGHT, epsilon=EPS_FRACTION), eta, support)
    extra = (N + 1) ** 2 - N**2  # virgil's cells beyond eht-imaging's, each sqrt(eps²)
    want = WEIGHT * (np.mean([-r["tv"] for r in res]) + extra * eps)
    want_grad = chain(b, support, -WEIGHT * np.mean(unflip([r["tv_grad"] for r in res]), axis=0))
    record("rel_value", abs(value / want - 1))
    record("rel_grad", rel(grad, want_grad))
    assert abs(value / want - 1) < 1e-12
    assert rel(grad, want_grad) < 1e-12


@needs_ehtim
@pytest.mark.external
@pytest.mark.validates("virgil.imaging.MaxEntropy", roots=["ehtim"])
def test_max_entropy_matches_ehtim(scene):
    """Against a flat default image over the support (virgil's default)
    and a random positive one."""
    eta, support, b = scene
    rng = np.random.default_rng(5)
    custom = rng.uniform(0.5, 2.0, b.shape)
    for prior in (None, custom):
        q = (support if prior is None else prior) / np.sum(support if prior is None else prior)
        e = sp.run("ehtim", "ehtim_worker.py", {"task": "regularisers", "images": [b.tolist()],
                                                "priors": [np.where(q > 0, q, 1.0).tolist()], "tv_epsilon": 0.0})["results"][0]
        value, grad = virgil_value_and_grad(vi.MaxEntropy(WEIGHT, prior=prior), eta, support)
        want = -WEIGHT * e["simple"]
        want_grad = chain(b, support, -WEIGHT * np.asarray(e["simple_grad"]))
        record("rel_value", abs(value / want - 1))
        assert abs(value / want - 1) < 1e-12
        assert rel(grad, want_grad) < 1e-12


# --------------------------------------------------------------------- MPoL


@needs_mpol
@pytest.mark.external
@pytest.mark.validates("virgil.imaging.TSV", "virgil.imaging.TV", roots=["mpol"])
def test_tsv_and_tv_match_mpol(scene):
    eta, support, b = scene
    eps = EPS_FRACTION / b.size
    res = sp.run("mpol", "mpol_worker.py", {"task": "regularisers", "images": [f.tolist() for f in flips(b)],
                                            "priors": [None] * 4, "tv_epsilon": eps**2})["results"]
    value, grad = virgil_value_and_grad(vi.TSV(WEIGHT), eta, support)
    assert abs(value / (WEIGHT * res[0]["tsv"]) - 1) < 1e-12
    assert rel(grad, chain(b, support, WEIGHT * np.asarray(res[0]["tsv_grad"]))) < 1e-12
    value, grad = virgil_value_and_grad(vi.TV(WEIGHT, epsilon=EPS_FRACTION), eta, support)
    extra = (N + 1) ** 2 - (N - 1) ** 2
    want = WEIGHT * (np.mean([r["tv"] for r in res]) + extra * eps)
    want_grad = chain(b, support, WEIGHT * np.mean(unflip([r["tv_grad"] for r in res]), axis=0))
    record("rel_tv_value", abs(value / want - 1))
    record("rel_tv_grad", rel(grad, want_grad))
    assert abs(value / want - 1) < 1e-12
    assert rel(grad, want_grad) < 1e-12


@needs_mpol
@pytest.mark.external
@pytest.mark.validates("virgil.imaging.MaxEntropy", roots=["mpol"])
def test_max_entropy_matches_mpol():
    """MPoL's entropy needs positive pixels and a scalar prior: an image
    without a support, against a flat default."""
    eta = np.random.default_rng(7).normal(size=(N, N))
    b = np.asarray(vm.Image(eta, 1.0).brightness)
    m = sp.run("mpol", "mpol_worker.py", {"task": "regularisers", "images": [b.tolist()],
                                          "priors": [1.0 / b.size], "tv_epsilon": 1e-10})["results"][0]
    value, grad = virgil_value_and_grad(vi.MaxEntropy(WEIGHT), eta, None)
    support = np.ones_like(b, bool)
    assert abs(value / (WEIGHT * m["entropy"]) - 1) < 1e-12
    assert rel(grad, chain(b, support, WEIGHT * np.asarray(m["entropy_grad"]))) < 1e-12


# -------------------------------------------------------------- mathematics


@pytest.mark.validates("virgil.imaging.Laplacian", roots=["mathematics"])
def test_laplacian_is_the_five_point_laplacian(scene):
    """SciPy's five-point Laplacian of the image padded with a ring of
    zeros (the documented ring of pixels around the image)."""
    eta, support, b = scene
    lap = ndimage.laplace(np.pad(b, 1), mode="constant", cval=0.0)
    want = WEIGHT * np.sum(lap**2)
    value, grad = virgil_value_and_grad(vi.Laplacian(WEIGHT), eta, support)
    # its gradient in the pixels: 2 w L(L b) (the Laplacian is symmetric)
    g_b = 2 * WEIGHT * ndimage.laplace(lap, mode="constant", cval=0.0)[1:-1, 1:-1]
    record("rel_value", abs(value / want - 1))
    assert abs(value / want - 1) < 1e-12
    assert rel(grad, chain(b, support, g_b)) < 1e-12


def starlet_reference(image, scales):
    """The à-trous starlet transform (Starck, Murtagh & Fadili 2010): B3
    spline [1, 4, 6, 4, 1] / 16 with holes, zero beyond the edges."""
    kernel = np.array([1, 4, 6, 4, 1]) / 16
    smooth, details = image, []
    for j in range(scales):
        holes = np.zeros(4 * 2**j + 1)
        holes[:: 2**j] = kernel
        smoother = ndimage.convolve1d(smooth, holes, axis=0, mode="constant")
        smoother = ndimage.convolve1d(smoother, holes, axis=1, mode="constant")
        details.append(smooth - smoother)
        smooth = smoother
    return np.array(details), smooth


@pytest.mark.parametrize("scales", [1, 3, 4])
@pytest.mark.validates("virgil.imaging.StarletL1", roots=["mathematics"])
def test_starlet_l1_matches_the_a_trous_transform(scene, scales):
    eta, support, b = scene
    details, coarse = starlet_reference(b, scales)
    got_details, got_coarse = (np.asarray(a) for a in vi.starlet(b, scales))
    assert np.max(np.abs(got_details - details)) < 1e-15
    assert np.max(np.abs(got_details.sum(0) + got_coarse - b)) < 1e-15
    eps = EPS_FRACTION / b.size
    want = WEIGHT * np.sum(np.sqrt(details**2 + eps**2))
    value, _ = virgil_value_and_grad(vi.StarletL1(WEIGHT, scales=scales, epsilon=EPS_FRACTION), eta, support)
    record("rel_value", abs(value / want - 1))
    assert abs(value / want - 1) < 1e-12


@pytest.mark.validates("virgil.imaging.LogSum", roots=["mathematics"])
def test_log_sum_is_its_formula(scene):
    """w Σ log(1 + b / (ε b̄)), b̄ the mean pixel flux over the support; its
    gradient in the pixels w / (b + ε b̄)."""
    eta, support, b = scene
    mean = 1.0 / support.sum()
    want = WEIGHT * np.sum(np.log1p(b / (EPS_FRACTION * mean)))
    value, grad = virgil_value_and_grad(vi.LogSum(WEIGHT, epsilon=EPS_FRACTION), eta, support)
    g_b = WEIGHT / (b + EPS_FRACTION * mean)
    assert abs(value / want - 1) < 1e-12
    assert rel(grad, chain(b, support, g_b)) < 1e-12
