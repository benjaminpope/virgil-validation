"""virgil.simulate.bias_test against biases we predict ourselves.

bias_test fits a model to n noisy simulations of a scene and returns the
fitted values, to show biases (its docstring). Two cases on one three-UT
file of a 3 mas uniform disk (V² errors 0.02; the closure phases of a
centred disk are zero and carry no information):

* Correct model, nearly linear problem: the fitted diameters must scatter
  about the truth plus the second-order bias of nonlinear least squares
  (Box 1971, J. R. Stat. Soc. B 33, 171: b = -½ F⁻² Σ w J H for one
  parameter, with J and H the first and second derivatives of the model
  V² and F = Σ w J² the Fisher information), and with the Fisher spread
  F^-1/2. Both computed here with crosscheck.sky and finite differences.
  The bias is 0.3 % of the spread, so this is a test of zero bias.
* Wrong model, a known nonlinear bias: a Gaussian fitted to the disk. With
  noise small, the fits scatter about the pseudo-true sigma that minimizes
  the noiseless χ² (White 1982, Econometrica 50, 1), which we find with
  SciPy: 0.787 mas, 5 % above the 0.75 mas that matches the disk's
  curvature at zero baseline (σ = D / 4). The control shows that the
  small-baseline answer is excluded.
"""

import jax
import numpy as np
import pytest
from scipy import optimize, stats

from crosscheck import chi2 as ours, simulate, sky
from evidence.plugin import record

vm = pytest.importorskip("virgil.models")
from virgil import simulate as vs  # noqa: E402
from virgil.oidata import OIData  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

pytestmark = pytest.mark.x64

UTS3 = np.array([[-9.925, -20.335], [14.887, 30.502], [103.306, 43.999]])
DIAM, N_DRAWS = 3.0, 40


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    path = tmp_path_factory.mktemp("bias") / "ud.fits"
    simulate.observe(path, lambda u, v, w: sky.vis_uniform_disk(u, v, w, DIAM), UTS3,
                     hour_angles_h=np.linspace(-3, 3, 5), wavelengths=np.array([1.65e-6, 2.2e-6]), dec_deg=-30.0,
                     sigma_v2=0.02, sigma_cp_deg=1.0, rng=np.random.default_rng(1))
    return OIData(str(path)), ours.load(path)


def model_v2(d, vis, p):
    return np.abs(vis(d["u"], d["v"], d["wl"], p)) ** 2


@pytest.mark.validates("virgil.simulate.bias_test", roots=["statistics", "mathematics"])
def test_correct_model_scatters_about_the_truth(template):
    """Mean within 4 standard errors of truth + Box bias; sample variance
    consistent with the Fisher variance (chi-squared test, p > 1e-3)."""
    data, d = template
    out = vs.bias_test(vm.UniformDisk(DIAM), data, vm.UniformDisk(DIAM), {"diam": dist.LogUniform(0.1, 10.0)},
                       N_DRAWS, jax.random.key(3))
    x = np.asarray(out["diam"])
    assert x.shape == (N_DRAWS,) and np.asarray(out["chi2_red"]).shape[0] == N_DRAWS
    h, w = 1e-4, 1.0 / d["dv2"] ** 2
    up, mid, down = (model_v2(d, sky.vis_uniform_disk, DIAM + s) for s in (h, 0.0, -h))
    J, H = (up - down) / (2 * h), (up - 2 * mid + down) / h**2
    fisher = np.sum(w * J * J)
    sigma, box = fisher**-0.5, -0.5 * np.sum(w * J * H) / fisher**2
    pull = (x.mean() - DIAM - box) / (sigma / np.sqrt(N_DRAWS))
    var_stat = (N_DRAWS - 1) * x.var(ddof=1) / sigma**2
    p_var = 2 * min(stats.chi2.cdf(var_stat, N_DRAWS - 1), stats.chi2.sf(var_stat, N_DRAWS - 1))
    record("box_bias_over_sigma", box / sigma)
    record("mean_pull", pull)
    record("std_over_fisher", x.std(ddof=1) / sigma)
    record("p_variance", p_var)
    assert abs(box) < 0.01 * sigma  # the problem is linear enough: zero bias expected
    assert abs(pull) < 4
    assert p_var > 1e-3


@pytest.fixture(scope="module")
def misfit(template):
    data, d = template
    out = vs.bias_test(vm.UniformDisk(DIAM), data, vm.GaussianDisk(1.0), {"sigma": dist.LogUniform(0.05, 5.0)},
                       N_DRAWS, jax.random.key(4))
    target, w = model_v2(d, sky.vis_uniform_disk, DIAM), 1.0 / d["dv2"] ** 2
    star = optimize.minimize_scalar(lambda s: np.sum(w * (model_v2(d, sky.vis_gaussian, s) - target) ** 2),
                                    bounds=(0.1, 3.0), method="bounded", options={"xatol": 1e-10}).x
    return np.asarray(out["sigma"]), float(star)


@pytest.mark.validates("virgil.simulate.bias_test", roots=["mathematics"])
def test_wrong_model_scatters_about_the_pseudo_true_value(misfit):
    """The fitted Gaussian sigmas average to our pseudo-true sigma, within
    4 standard errors."""
    y, star = misfit
    pull = (y.mean() - star) / (y.std(ddof=1) / np.sqrt(y.size))
    record("pseudo_true_sigma", star)
    record("mean_pull", pull)
    assert abs(pull) < 4


@pytest.mark.validates("virgil.simulate.bias_test", roots=["mathematics"], kind="control")
def test_wrong_model_is_not_the_small_baseline_match(misfit):
    """sigma = D / 4 (no nonlinear bias) is excluded by over 20 standard errors."""
    y, _ = misfit
    assert abs(y.mean() - DIAM / 4) / (y.std(ddof=1) / np.sqrt(y.size)) > 20
