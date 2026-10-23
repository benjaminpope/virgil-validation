"""Independent checks of the derived and experimental items of virgil's radio innovation register.

One node per item (``design/radio_imaging_innovations.md`` in virgil, sections B2, B5, B6; B4 is listed in
``trust/not_covered.md``). References are numpy only (``crosscheck.radio_innovations``).

* B2 (derived), ``precondition="diag"``: the Gauss-Newton diagonal against the dense one, the probe
  estimate against its stated error, the entropy metric against the Skilling-Bryan metric pulled back
  through the softmax. Needs ``virgil.fitting.gauss_newton_diag`` (after virgil 0.4.0).
* B5 (derived), free per-observation offsets of the Toeplitz chi-squared: against a direct DFT chi-squared,
  to the documented Nyquist margin and offset bound. Needs ``virgil.radio_toeplitz`` (virgil#516).
* B6 (derived, experimental), the float32 linear-delta-V loss: no virgil code exists yet, so this checks
  the method itself, written here from its derivation.

CI's pinned virgil (0.4.0) has none of the three virgil features, so those tests skip there.
"""

import importlib

import numpy as np
import pytest

from crosscheck import radio_innovations as ri
from evidence.plugin import record


def _try(name):
    try:
        return importlib.import_module(name)
    except Exception:  # noqa: BLE001  (a missing optional dependency such as nufftax raises its own error)
        return None


radio = _try("virgil.radio")
toeplitz = _try("virgil.radio_toeplitz")
fitting = _try("virgil.fitting")
has_diag = fitting is not None and radio is not None and hasattr(fitting, "gauss_newton_diag")
has_toeplitz = radio is not None and toeplitz is not None and hasattr(radio.ComplexVisData, "toeplitz")
needs_diag = pytest.mark.skipif(not has_diag, reason="virgil.fitting.gauss_newton_diag (after virgil 0.4.0) is not installed")
needs_toeplitz = pytest.mark.skipif(not has_toeplitz, reason="virgil.radio_toeplitz (virgil#516) is not installed")

C_LIGHT = 299792458.0
MAS = ri.MAS


# ------------------------------------------------------------------ B2


def _imaging_problem(npix, seed=1, n=300, pixel=8.0, flux=0.4):
    """A noise-like visibility set on ``n`` points of one channel and one observation, a random log-brightness."""
    import numpyro.distributions as dist
    from virgil import Image

    rng = np.random.default_rng(seed)
    q = 10 ** rng.uniform(5.0, 6.7, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    u, v = q * np.cos(phi), q * np.sin(phi)
    weight = 10 ** rng.uniform(2.5, 4.0, n)
    vis = 0.01 * (rng.normal(size=n) + 1j * rng.normal(size=n))
    data = radio.ComplexVisData.from_arrays(u, v, vis, weight, np.full(n, 230e9))
    nuis = radio.CalibrationNuisance(data, gains=False, offsets=False, sigmas=False, alpha=False)
    scene = lambda values: Image(values["log_brightness"], pixel_scale_mas=pixel)  # noqa: E731
    priors = nuis.priors()
    priors["log_brightness"] = dist.ImproperUniform(dist.constraints.real, (), (npix, npix))
    eta = rng.normal(0.0, 1.0, (npix, npix))
    return dict(u=u, v=v, weight=weight, eta=eta, pixel=pixel, flux=flux, scene=scene, priors=priors,
                values={"cal.flux": flux, "log_brightness": eta}, term=data.term(scene, nuis))


def _virgil_diag(p, **kw):
    out = fitting.gauss_newton_diag(lambda **values: p["scene"](values), p["priors"], (), p["values"],
                                    likelihoods=[p["term"]], **kw)
    return np.asarray(out["log_brightness"])


@needs_diag
@pytest.mark.validates("virgil.fitting.gauss_newton_diag", roots=["mathematics"])
def test_gn_diagonal_equals_the_dense_diagonal_of_jtj():
    """6 x 6 image, 300 points: the exact diagonal (36 coordinates, no probes) equals the sum of squares of
    the dense Jacobian columns to 1e-9. Source: Schraudolph 2002, doi:10.1162/08997660260028683."""
    p = _imaging_problem(6)
    ref = ri.gn_diagonal(p["u"], p["v"], p["weight"], p["eta"], p["pixel"], p["flux"])
    got = _virgil_diag(p)
    err = np.abs(got / ref - 1).max()
    record("max_rel_error_exact_diagonal", err)
    assert err < 1e-9


@needs_diag
@pytest.mark.validates("virgil.fitting.gauss_newton_diag", roots=["statistics"])
def test_probe_estimate_is_within_its_stated_stochastic_error_of_the_exact_diagonal():
    """12 x 12 image, 1024 Rademacher probes: every coordinate is within 4 sqrt(2/m) of the dense diagonal and
    the rms error is within 1.5 sqrt(2/m) (the stated per-coordinate error of m probes). Source: Bekas et al.
    2007, 2007ApNM...57.1214B."""
    m = 1024
    p = _imaging_problem(12)
    ref = ri.gn_diagonal(p["u"], p["v"], p["weight"], p["eta"], p["pixel"], p["flux"])
    rel = _virgil_diag(p, n_probes=m) / ref - 1
    stated = np.sqrt(2.0 / m)
    record("max_probe_error_over_stated", np.abs(rel).max() / stated)
    record("rms_probe_error_over_stated", np.sqrt(np.mean(rel**2)) / stated)
    assert np.abs(rel).max() < 4 * stated
    assert np.sqrt(np.mean(rel**2)) < 1.5 * stated


@needs_diag
@pytest.mark.validates("virgil.fitting.gauss_newton_diag", "virgil.imaging.MaxEntropy", roots=["mathematics", "literature"])
@pytest.mark.parametrize("weight", [1e-3, 1.0, 1e3])
def test_entropy_metric_replaces_the_data_curvature_with_the_pulled_back_skilling_bryan_metric(weight):
    """With MaxEntropy(weight) the diagonal on the image pixels is the Skilling-Bryan metric
    weight * sum db^2/b pulled back through the softmax, which is weight * b * (1 - b), whatever the data
    (it is the data's curvature that is dropped, at any weight). The register writes weight * b, the b << 1
    limit: a million-pixel image differs by 1e-6, a 6 x 6 one here by up to 20%. Source: Skilling & Bryan
    1984, 1984MNRAS.211..111S."""
    from virgil.imaging import MaxEntropy

    p = _imaging_problem(6)
    ref = ri.entropy_metric_diagonal(weight, p["eta"])
    got = _virgil_diag(p, regularizers=[MaxEntropy(weight)])
    data_gn = ri.gn_diagonal(p["u"], p["v"], p["weight"], p["eta"], p["pixel"], p["flux"])
    b = ri.softmax_image(p["eta"])
    err = np.abs(got / ref - 1).max()
    record(f"max_rel_error_entropy_metric_w{weight:g}", err)
    record(f"max_rel_difference_from_w_b_w{weight:g}", np.abs(got / (weight * b) - 1).max())
    assert err < 1e-9
    assert np.abs(got / (weight * b) - 1).max() > 0.05  # the 1 - b factor is real at this size
    assert np.abs(got / data_gn - 1).max() > 0.1  # and the data's curvature is not in it


# ------------------------------------------------------------------ B5

PIXEL = 5.0  # mas
NPIX = 16
OFFSET_BOUND = 20.0  # mas, the precompute's max_offset_mas
DDIDS = np.array([0, 0, 1, 2])  # observation of each ddid
FREQS = np.array([230e9, 245e9, 230e9, 245e9])
GAINS = np.array([1.0, 1.08, 0.93])
FLUX = 0.7


def _toeplitz_case(fraction, offsets_mas, seed=0, n=600):
    """600 random points in four ddids and three observations, the longest baseline at ``fraction`` of the
    pixel Nyquist limit (``t / pi``), random complex data, a random 16 x 16 image."""
    from virgil import Image

    rng = np.random.default_rng(seed)
    q = fraction * 0.5 / (PIXEL * MAS) * np.sqrt(rng.uniform(0, 1, n))
    q[0] = fraction * 0.5 / (PIXEL * MAS)
    phi = rng.uniform(0, 2 * np.pi, n)
    u, v = q * np.cos(phi), q * np.sin(phi)
    ddid = rng.integers(0, 4, n)
    obs = DDIDS[ddid]
    weight = 10 ** rng.uniform(2.5, 4.0, n)
    vis = 0.05 * (rng.normal(size=n) + 1j * rng.normal(size=n))
    data = radio.ComplexVisData.from_arrays(u, v, vis, weight, FREQS[ddid], ddid=ddid, obs=obs)
    nuis = radio.CalibrationNuisance(data, ref_obs=0, sigmas=False, alpha=False)
    eta = rng.normal(0.0, 1.0, (NPIX, NPIX))
    scene = lambda values: Image(values["log_brightness"], pixel_scale_mas=PIXEL)  # noqa: E731
    offsets = np.asarray(offsets_mas, float)
    values = {"cal.flux": FLUX, "cal.gain": GAINS[1:], "cal.offset": offsets, "log_brightness": eta}
    template = Image(eta, pixel_scale_mas=PIXEL)
    return dict(u=u, v=v, vis=vis, weight=weight, ddid=ddid, obs=obs, eta=eta, data=data, nuis=nuis,
                scene=scene, values=values, template=template, offsets=np.vstack([[0.0, 0.0], offsets]))


def _chi2_both_ways(case, tdata):
    got, _ = tdata.chi2_by_ddid(case["scene"], case["values"], case["nuis"])
    ref = ri.direct_chi2_by_ddid(case["u"], case["v"], case["vis"], case["weight"], case["ddid"], case["obs"],
                                 ri.softmax_image(case["eta"]), PIXEL, FLUX, GAINS, case["offsets"])
    return np.asarray(got), ref


@pytest.fixture(scope="module")
def margin():
    """The Nyquist margin the precompute documents (``tdata.margin``), read once from a small case."""
    case = _toeplitz_case(0.3, [[3.1, -2.2], [-1.7, 4.4]])
    tdata = case["data"].toeplitz(case["template"], case["nuis"], offsets="spread", max_offset_mas=OFFSET_BOUND)
    assert 0.0 < tdata.margin < 1.0
    return tdata.margin


@needs_toeplitz
@pytest.mark.validates("virgil.radio_toeplitz.ToeplitzVisData", roots=["mathematics"])
@pytest.mark.parametrize("where", ["half", "at_margin"])
@pytest.mark.parametrize("offsets", [[[3.1, -2.2], [-1.7, 4.4]], [[20.0, 0.0], [0.0, -20.0]], [[14.1, 14.1], [-14.1, 14.1]]],
                         ids=["few_mas", "axis_bound", "diagonal_bound"])
def test_free_offsets_chi2_equals_the_direct_dft_chi2_up_to_the_documented_bounds(where, offsets, margin):
    """Per-ddid chi-squared of the Toeplitz term with free offsets equals the brute-force DFT chi-squared to
    1e-12 (relative), for offsets up to the precompute's bound (20 mas on an axis, 20 mas radially) and a longest
    baseline at half the Nyquist limit or 0.5% inside the documented margin |t| <= pi (1 - w/n_f).
    Source: Barnett et al. 2019, 2019SJSC...41C.479B (Sect. 3.3, the transform at non-integer positions)."""
    fraction = 0.5 if where == "half" else margin - 0.005
    case = _toeplitz_case(fraction, offsets)
    tdata = case["data"].toeplitz(case["template"], case["nuis"], offsets="spread", max_offset_mas=OFFSET_BOUND)
    got, ref = _chi2_both_ways(case, tdata)
    err = np.abs(got / ref - 1).max()
    record(f"max_rel_error_{where}_{'-'.join(f'{abs(x):g}' for x in np.ravel(offsets))}", err)
    assert tdata.nyquist_fraction == pytest.approx(fraction, abs=1e-3)
    assert err < 1e-12


@needs_toeplitz
@pytest.mark.validates("virgil.radio_toeplitz.ToeplitzVisData", roots=["mathematics"], kind="control")
def test_control_the_reference_with_the_offsets_dropped_does_not_match():
    """The same case against the direct chi-squared with every offset set to zero misses by order one in the
    ddids of the offset observations, so the agreement above is a test of the offsets."""
    case = _toeplitz_case(0.5, [[14.1, 14.1], [-14.1, 14.1]])
    tdata = case["data"].toeplitz(case["template"], case["nuis"], offsets="spread", max_offset_mas=OFFSET_BOUND)
    got, _ = _chi2_both_ways(case, tdata)
    ref0 = ri.direct_chi2_by_ddid(case["u"], case["v"], case["vis"], case["weight"], case["ddid"], case["obs"],
                                  ri.softmax_image(case["eta"]), PIXEL, FLUX, GAINS, np.zeros((3, 2)))
    miss = np.abs(got[2:] / ref0[2:] - 1).max()
    record("max_rel_miss_without_offsets", miss)
    assert miss > 1e-2


@needs_toeplitz
@pytest.mark.validates("virgil.radio_toeplitz.ToeplitzVisData", roots=["mathematics"], kind="guard")
def test_baselines_past_the_margin_and_offsets_past_the_bound_are_refused(margin):
    """The two conditions of the exactness claim are enforced: a baseline 3% past the margin raises at the
    precompute, an offset 1 mas past ``max_offset_mas`` raises at evaluation."""
    over = _toeplitz_case(margin + 0.03, [[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        over["data"].toeplitz(over["template"], over["nuis"], offsets="spread", max_offset_mas=OFFSET_BOUND)
    case = _toeplitz_case(0.3, [[OFFSET_BOUND + 1.0, 0.0], [0.0, 0.0]])
    tdata = case["data"].toeplitz(case["template"], case["nuis"], offsets="spread", max_offset_mas=OFFSET_BOUND)
    with pytest.raises(Exception):  # noqa: B017  (a checkify error from inside jit, whatever its class)
        tdata.chi2_by_ddid(case["scene"], case["values"], case["nuis"])


# ------------------------------------------------------------------ B6


def _delta_problem(seed=4, npix=16, n=3000, pixel=5.0, noise=1e-3):
    rng = np.random.default_rng(seed)
    q = 10 ** rng.uniform(np.log10(2e5), np.log10(1.5e7), n)
    phi = rng.uniform(0, 2 * np.pi, n)
    east, north = ri.pixel_positions(npix, pixel)
    dft = ri.dft_matrix(q * np.cos(phi), q * np.sin(phi), east, north)
    z_ref = rng.normal(0.0, 1.0, npix * npix) - 3.0
    truth = dft @ np.exp(z_ref + 0.05 * rng.normal(size=npix * npix))
    sigma = noise * np.abs(truth).max()
    vis = truth + sigma * (rng.normal(size=n) + 1j * rng.normal(size=n))
    return ri.DeltaProblem(dft, z_ref, vis, np.full(n, 1.0 / sigma**2)), rng


@pytest.mark.validates("crosscheck.radio_innovations.DeltaProblem", roots=["mathematics"], kind="reference")
@pytest.mark.parametrize("step", [1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
def test_float32_delta_loss_matches_the_float64_chi2_difference(step):
    """3000 points, 16 x 16 image, steps of 1e-2 to 1e-6 in the log-brightness about a float64 reference: the
    float32 delta-V difference of chi-squared equals the float64 one to 1e-6 of the sum of the absolute values of
    its terms (float32 rounding is 6e-8; measured about 3e-8), while the difference of two float32 chi-squared
    values is at least 100 times worse by 1e-5. Source: Higham 2002, doi:10.1137/1.9780898718027 (residual-based
    mixed precision)."""
    problem, rng = _delta_problem()
    z = problem.z_ref + step * rng.normal(size=problem.z_ref.size)
    exact = problem.exact_delta(z)
    scale = problem.absolute_scale(z)
    delta_err = abs(problem.delta_chi2_float32(z) - exact) / scale
    naive_err = abs(problem.naive_delta_float32(z) - exact) / scale
    record(f"delta_loss_error_over_scale_step{step:g}", delta_err)
    record(f"naive_float32_error_over_scale_step{step:g}", naive_err)
    assert delta_err < 1e-6
    if step <= 1e-5:
        assert naive_err > 100 * delta_err


@pytest.mark.validates("crosscheck.radio_innovations.DeltaProblem", roots=["mathematics"], kind="reference")
def test_the_delta_formula_is_exact_in_float64():
    """In float64 the identity V(z) = V_ref + E (exp(z_ref) expm1(z - z_ref)) gives chi2(z) - chi2(z_ref)
    to 1e-12 of the scale at a step of 0.3, so the float32 error above is rounding, not the derivation."""
    problem, rng = _delta_problem()
    z = problem.z_ref + 0.3 * rng.normal(size=problem.z_ref.size)
    dv = problem.dft @ (np.exp(problem.z_ref) * np.expm1(z - problem.z_ref))
    delta = np.sum(problem.weight * (np.abs(dv) ** 2 - 2.0 * (problem.resid_ref.conj() * dv).real))
    err = abs(delta - problem.exact_delta(z)) / np.sum(problem.weight * (np.abs(dv) ** 2 + 2.0 * np.abs(problem.resid_ref) * np.abs(dv)))
    record("float64_identity_error_over_scale", err)
    assert err < 1e-12
