"""Independent checks of virgil.radio's remaining trust nodes, NumPy references only (virgil issue #533).

* ``ComplexVisData.from_arrays``, ``dirty_beam`` and ``dirty_image`` against a direct sum over the
  points (``crosscheck.radio_trust.dirty``), for natural, uniform and Briggs weights and a uv taper;
* the joint default fit of ``CalibrationNuisance`` (flux, gains, offsets, per-ddid error scales and spectral
  index fitted together, audit finding F2) against the Fisher matrix of the model equation, on a small
  synthetic set;
* ``gauss_newton_diag`` against finite-difference Jacobians (D13: ``w b (1 - b)`` in softmax-logit coordinates);
* the Stage-1 Laplace marginal of the antenna gains against Gauss-Hermite quadrature on a three-antenna array.

Everything is tier A and runs in seconds on a laptop (no CASA).
"""

import numpy as np
import pytest

radio = pytest.importorskip("virgil.radio")

from crosscheck import radio_trust as rt  # noqa: E402
from evidence.plugin import record  # noqa: E402

MAS = rt.MAS


def _points(n=300, seed=0, qmin=2e4, qmax=3e5, nu=2.3e11):
    rng = np.random.default_rng(seed)
    q = 10 ** rng.uniform(np.log10(qmin), np.log10(qmax), n)
    phi = rng.uniform(0, 2 * np.pi, n)
    u, v = q * np.cos(phi), q * np.sin(phi)
    # a few exact Hermitian-partner and on-axis points exercise the cell folding
    u[:3], v[:3] = [5e4, -5e4, 7e4], [0.0, 0.0, 0.0]
    w = 10 ** rng.uniform(1.0, 3.0, n)
    vis = rng.normal(size=n) + 1j * rng.normal(size=n)
    return u, v, vis, w, np.full(n, nu)


# ------------------------------------------------ 1. construction and dirty images


@pytest.mark.validates("virgil.radio.ComplexVisData", roots=["mathematics"])
def test_from_arrays_keeps_the_data_broadcasts_scalars_relabels_and_refuses_bad_input():
    """Arrays come back unchanged, scalars broadcast over the points, labels are renumbered 0..n-1 in
    sorted order of the originals (a ddid shared by two observations gets one label per observation), and
    non-finite values, non-positive weights and mismatched shapes raise."""
    u, v, vis, w, freq = _points(12)
    ddid = np.array([7, 7, 7, 9, 9, 9, 7, 7, 9, 9, 7, 9])
    obs = np.array([4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2])
    d = radio.ComplexVisData.from_arrays(u, v, vis, w, freq, ddid=ddid, obs=obs)
    assert len(d) == 12
    for name, ref in (("u", u), ("v", v), ("vis", vis), ("weight", w), ("freq", freq)):
        assert np.array_equal(np.asarray(getattr(d, name)), ref), name
    assert d.obs_ids == (2, 4)
    assert np.array_equal(np.asarray(d.obs), (obs == 4).astype(int))  # sorted originals: 2 -> 0, 4 -> 1
    pairs = sorted(set(zip(ddid.tolist(), obs.tolist())), key=lambda p: (p[0] * 3 + (p[1] == 4)))
    label = {p: i for i, p in enumerate(sorted(set(zip(ddid.tolist(), (obs == 4).astype(int).tolist()))))}
    assert len(label) == len(pairs) == len(d.ddid_ids) == 4  # (7, obs 2), (7, obs 4), (9, obs 2), (9, obs 4)
    want = np.array([label[(a, int(b == 4))] for a, b in zip(ddid, obs)])
    assert np.array_equal(np.asarray(d.ddid), want)
    assert [d.ddid_ids[i] for i in want] == ddid.tolist()
    s = radio.ComplexVisData.from_arrays(u, v, vis, 3.0, 2.3e11)
    assert np.all(np.asarray(s.weight) == 3.0) and np.all(np.asarray(s.freq) == 2.3e11)
    bad = dict(u=u, v=v, vis=vis, weight=w, freq=freq)
    for key, value in [("weight", -w), ("weight", np.where(np.arange(12) == 0, 0.0, w)), ("freq", -freq),
                       ("vis", np.where(np.arange(12) == 1, np.nan, vis)), ("u", np.where(np.arange(12) == 2, np.inf, u)),
                       ("v", v[:5])]:
        with pytest.raises(ValueError):
            radio.ComplexVisData.from_arrays(**{**bad, key: value})


CASES = [
    dict(npix=33, pix=20.0, weighting="natural"),
    dict(npix=32, pix=20.0, weighting="natural"),  # even grid: the phase centre is between four pixels
    dict(npix=33, pix=20.0, weighting="uniform"),
    dict(npix=32, pix=20.0, weighting="briggs", robust=0.0),
    dict(npix=33, pix=20.0, weighting="briggs", robust=-1.0, uvtaper=(150.0, 90.0, 35.0)),
    dict(npix=33, pix=20.0, weighting="natural", uvtaper=120.0),
]


def _weights(case, u, v, w):
    """The reference imaging weights of a case: density factor (cell 1/FoV) then the taper."""
    cell = 1.0 / (case["npix"] * case["pix"] * MAS)
    kind = case["weighting"]
    out = w if kind == "natural" else rt.density_weights(u, v, w, cell, kind, case.get("robust", 0.5))
    t = case.get("uvtaper")
    if t is not None:
        bmaj, bmin, pa = (t, t, 0.0) if np.ndim(t) == 0 else t
        out = out * rt.uv_taper(u, v, bmaj, bmin, pa)
    return out


def _ids(case):
    return "-".join(str(case[k]) for k in ("npix", "weighting") if k in case) + ("-taper" if "uvtaper" in case else "")


@pytest.mark.validates("virgil.radio.ComplexVisData.dirty_image", "virgil.radio.ComplexVisData.dirty_beam", roots=["mathematics"])
@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_dirty_image_and_beam_are_the_direct_sums_with_briggs_weights_and_the_gaussian_taper(case):
    """Dirty image sum w Re[V exp(+2 pi i (u x + v y))] / sum w and dirty beam sum w cos(2 pi (u x + v y)) /
    sum w, on a grid with row 0 North, column 0 East and the phase centre at index (npix - 1) / 2, with the
    weights w the natural 1/sigma^2 times Briggs's density factor (square cells of 1/FoV, Hermitian partners
    in one cell) times the Fourier transform of the Gaussian taper; both agree with an explicit loop over
    the points to 1e-9 of the peak."""
    u, v, vis, w, freq = _points()
    data = radio.ComplexVisData.from_arrays(u, v, vis, w, freq)
    kw = {k: case[k] for k in ("weighting", "robust", "uvtaper") if k in case}
    got_i = np.asarray(data.dirty_image(case["npix"], case["pix"], **kw))
    got_b = np.asarray(data.dirty_beam(case["npix"], case["pix"], **kw))
    wt = _weights(case, u, v, w)
    ref_i = rt.dirty(u, v, vis, wt, case["npix"], case["pix"])
    ref_b = rt.dirty(u, v, np.ones_like(vis), wt, case["npix"], case["pix"])
    err_i = np.abs(got_i - ref_i).max() / np.abs(ref_i).max()
    err_b = np.abs(got_b - ref_b).max()
    record("max_rel_error_dirty_image", err_i)
    record("max_abs_error_dirty_beam", err_b)
    assert err_i < 1e-9 and err_b < 1e-9
    if case["npix"] % 2:  # a pixel on the phase centre: the beam is exactly 1 there
        c = case["npix"] // 2
        assert got_b[c, c] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.validates("virgil.radio.ComplexVisData.dirty_image", "virgil.radio.ComplexVisData.dirty_beam", roots=["mathematics"], kind="control")
def test_control_the_wrong_sign_or_a_flipped_axis_of_the_reference_misses_by_order_one():
    """The agreement above is orientation-sensitive: the reference with exp(-2 pi i ...) (sign), with columns
    reversed (East on the right) or with the image transposed does not match at 1e-2 of the peak, so the
    1e-9 test pins the sign, the East-left orientation and the row/column order."""
    u, v, vis, w, freq = _points()
    data = radio.ComplexVisData.from_arrays(u, v, vis, w, freq)
    got = np.asarray(data.dirty_image(33, 20.0))
    ref = rt.dirty(u, v, vis, w, 33, 20.0)
    conj = rt.dirty(u, v, np.conj(vis), w, 33, 20.0)
    for name, wrong in (("conj", conj), ("flip_east", ref[:, ::-1]), ("transpose", ref.T), ("flip_north", ref[::-1])):
        miss = np.abs(got - wrong).max() / np.abs(ref).max()
        record(f"control_miss_{name}", miss)
        assert miss > 1e-2, name


# ------------------------------------------------ 3. calibration nuisances, joint default fit

PIX = 12.0
NPIX = 24
N_OBS = 3
DDIDS = np.array([0, 0, 1, 1, 2, 2])  # observation of each ddid; two spectral windows per observation
FREQS = np.array([225e9, 240e9, 225e9, 240e9, 225e9, 240e9])
TRUTH = {
    "cal.flux": 0.35,
    "cal.gain": np.array([1.07, 0.94]),
    "cal.offset": np.array([[3.0, -2.0], [-1.5, 2.5]]),  # mas
    "cal.sigma": np.array([1.0, 1.3, 1.6, 1.1, 1.8, 1.2]),
    "cal.alpha": 2.2,
}


def _blob(npix=NPIX, pix=PIX):
    """Unit-sum asymmetric scene: an elliptical Gaussian plus a smaller blob (numpy)."""
    k = ((npix - 1) / 2.0 - np.arange(npix)) * pix
    east, north = np.meshgrid(k, k)
    big = np.exp(-0.5 * ((east - 10) ** 2 / 45.0**2 + (north + 5) ** 2 / 30.0**2))
    small = 0.3 * np.exp(-0.5 * ((east + 55) ** 2 + (north - 35) ** 2) / 20.0**2)
    img = big + small
    return img / img.sum()


def _reference_model(z, truth, scene_vis, nu0):
    """M = F g_e (nu / nu0)^alpha exp(-2 pi i (da u + dd v)) V_scene, offsets in mas (the model equation of
    the radio guide), reference observation 0 with g = 1 and no offset."""
    gain = np.concatenate([[1.0], truth["cal.gain"]])[z["obs"]]
    off = np.vstack([[0.0, 0.0], truth["cal.offset"]])[z["obs"]]
    return (truth["cal.flux"] * gain * (z["freq"] / nu0) ** truth["cal.alpha"]
            * np.exp(-2j * np.pi * MAS * (off[:, 0] * z["u"] + off[:, 1] * z["v"])) * scene_vis)


def _fisher_sigmas(z, truth, scene_vis, nu0):
    """1-sigma errors, in the order flux, gain, offset (row-major), error scale, alpha, from analytic
    derivatives of M (Re and Im Gaussian of variance s_d^2 / w; the scales decouple at the truth, 4 N_d / s_d^2)."""
    obs = z["obs"]
    gain = np.concatenate([[1.0], truth["cal.gain"]])[obs]
    m = _reference_model(z, truth, scene_vis, nu0)
    cols = [m / truth["cal.flux"]]
    cols += [np.where(obs == e, m / gain, 0.0) for e in range(1, N_OBS)]
    for e in range(1, N_OBS):
        cols += [np.where(obs == e, -2j * np.pi * MAS * z["u"] * m, 0.0), np.where(obs == e, -2j * np.pi * MAS * z["v"] * m, 0.0)]
    cols.append(np.log(z["freq"] / nu0) * m)
    jac = np.array(cols)
    wt = z["weight"] / truth["cal.sigma"][z["ddid"]] ** 2
    info = (jac * wt).real @ jac.real.T + (jac * wt).imag @ jac.imag.T
    p = np.sqrt(np.diag(np.linalg.inv(info)))
    n_d = np.bincount(z["ddid"])
    n_g = N_OBS - 1
    return np.concatenate([[p[0]], p[1:1 + n_g], p[1 + n_g:1 + 3 * n_g], truth["cal.sigma"] / (2 * np.sqrt(n_d)), [p[-1]]])


def _flat(values):
    return np.concatenate([np.ravel(values[k]) for k in ("cal.flux", "cal.gain", "cal.offset", "cal.sigma", "cal.alpha")])


@pytest.mark.validates("virgil.radio.CalibrationNuisance", "virgil.radio.ComplexVisData.term", roots=["statistics"])
def test_joint_default_fit_of_every_nuisance_is_unbiased_to_the_fisher_errors():
    """F2: with the defaults (flux, 2 gains, 2 offsets, 6 per-ddid error scales and the spectral index all
    fitted at once, default priors, cold start) on 900 synthetic points of a known scene, injected with
    gains 1.07 and 0.94, offsets of a few mas, error scales 1.0-1.8 and alpha 2.2 (noise seed 0), every
    parameter is recovered within 3 sigma of the Fisher matrix of the model equation, written here with
    analytic derivatives (the same joint model virgil fits; nothing is held fixed)."""
    from virgil import fit

    import virgil_bridge.alma as vba

    rng = np.random.default_rng(0)
    n = 900
    u, v, _, w, _ = _points(n, seed=1, qmin=3e4, qmax=1.5e5)
    ddid = rng.integers(0, 6, n)
    ddid[:6] = np.arange(6)
    z = dict(u=u, v=v, weight=10 ** rng.uniform(2.0, 3.0, n), ddid=ddid, obs=DDIDS[ddid], freq=FREQS[ddid])
    nu0 = float(np.sum(z["weight"] * z["freq"]) / np.sum(z["weight"]))
    img = _blob()
    scene_vis = rt.image_model(np.log(img), PIX, u, v)
    mean = _reference_model(z, TRUTH, scene_vis, nu0)
    sig = TRUTH["cal.sigma"][ddid] / np.sqrt(z["weight"])
    vis = mean + sig * (rng.normal(size=n) + 1j * rng.normal(size=n))
    data = radio.ComplexVisData.from_arrays(u, v, vis, z["weight"], z["freq"], ddid=ddid, obs=z["obs"])
    nuis = radio.CalibrationNuisance(data, nu0_hz=nu0)  # every default: the joint fit
    assert nuis.gains and nuis.offsets and nuis.sigmas is True and nuis.alpha
    scene = vba.fixed_image_scene(img, PIX)
    term = data.term(scene, nuis)
    sigma = _fisher_sigmas(z, TRUTH, scene_vis, nu0)
    result = fit(lambda **values: scene(values), nuis.priors(), data=(), likelihoods=[term], init=nuis.defaults(),
                 method="lbfgs", max_steps=2000)
    pull = (_flat(result.values) - _flat(TRUTH)) / sigma
    record("n_points", n)
    record("max_abs_pull", np.abs(pull).max())
    record("median_abs_pull", np.median(np.abs(pull)))
    assert len(pull) == 1 + 2 + 4 + 6 + 1
    assert np.abs(pull).max() < 3.0, np.round(pull, 2)


# ------------------------------------------------ 5. gauss_newton_diag, finite differences


def _gn_problem(npix=6, n=250, seed=1, pixel=8.0, flux=0.4):
    import numpyro.distributions as dist
    from virgil import Image

    rng = np.random.default_rng(seed)
    q = 10 ** rng.uniform(5.0, 6.7, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    u, v = q * np.cos(phi), q * np.sin(phi)
    w = 10 ** rng.uniform(2.5, 4.0, n)
    vis = 0.01 * (rng.normal(size=n) + 1j * rng.normal(size=n))
    data = radio.ComplexVisData.from_arrays(u, v, vis, w, np.full(n, 230e9))
    nuis = radio.CalibrationNuisance(data, gains=False, offsets=False, sigmas=False, alpha=False)
    scene = lambda values: Image(values["log_brightness"], pixel_scale_mas=pixel)  # noqa: E731
    priors = nuis.priors()
    priors["log_brightness"] = dist.ImproperUniform(dist.constraints.real, (), (npix, npix))
    eta = rng.normal(0.0, 1.0, (npix, npix))
    return dict(u=u, v=v, vis=vis, w=w, eta=eta, pixel=pixel, flux=flux, scene=scene, priors=priors,
                values={"cal.flux": flux, "log_brightness": eta}, term=data.term(scene, nuis))


def _central_jacobian(fun, x, h=1e-6):
    """Columns (fun(x + h e_j) - fun(x - h e_j)) / 2h for a real vector x; fun returns a real vector."""
    cols = []
    for j in range(x.size):
        d = np.zeros(x.size)
        d[j] = h
        cols.append((fun(x + d) - fun(x - d)) / (2 * h))
    return np.array(cols).T


def _gn_diag(p, **kw):
    from virgil import fitting

    out = fitting.gauss_newton_diag(lambda **values: p["scene"](values), p["priors"], (), p["values"],
                                    likelihoods=[p["term"]], **kw)
    return np.asarray(out["log_brightness"])


def _has_gn_diag():
    try:
        from virgil import fitting
    except Exception:  # noqa: BLE001
        return False
    return hasattr(fitting, "gauss_newton_diag")


@pytest.mark.skipif(not _has_gn_diag(), reason="virgil.fitting.gauss_newton_diag not installed")
@pytest.mark.validates("virgil.fitting.gauss_newton_diag", roots=["mathematics"])
def test_data_diagonal_is_the_finite_difference_diagonal_of_jtj_and_the_entropy_metric_is_w_b_one_minus_b():
    """6 x 6 image of softmax logits: (a) the data diagonal of J^T J, J the central finite-difference
    Jacobian (step 1e-6) of the weighted residual sqrt(w) [Re M, Im M] with respect to the logits, M my own
    DFT of the softmax image, matches `gauss_newton_diag` to 1e-6; (b) with `MaxEntropy(weight)` the diagonal
    equals that of the finite-difference Jacobian of 2 sqrt(weight b) (the Skilling-Bryan metric rows),
    which is weight * b * (1 - b) (D13), and differs from weight * b by more than 5% at this size."""
    from virgil.imaging import MaxEntropy

    p = _gn_problem()
    shape = p["eta"].shape

    def resid(x):
        m = rt.image_model(x.reshape(shape), p["pixel"], p["u"], p["v"], p["flux"])
        return np.concatenate([np.sqrt(p["w"]) * m.real, np.sqrt(p["w"]) * m.imag])

    jac = _central_jacobian(resid, p["eta"].ravel())
    ref = (jac**2).sum(axis=0).reshape(shape)
    got = _gn_diag(p)
    err = np.abs(got / ref - 1).max()
    record("max_rel_error_data_diagonal", err)
    assert err < 1e-6
    weight = 2.5
    jac_e = _central_jacobian(lambda x: 2.0 * np.sqrt(weight * rt.softmax(x.reshape(shape)).ravel()), p["eta"].ravel())
    ref_e = (jac_e**2).sum(axis=0).reshape(shape)
    b = rt.softmax(p["eta"])
    assert np.allclose(ref_e, weight * b * (1 - b), rtol=1e-8)  # the reference is itself the closed form
    got_e = _gn_diag(p, regularizers=[MaxEntropy(weight)])
    err_e = np.abs(got_e / ref_e - 1).max()
    record("max_rel_error_entropy_metric", err_e)
    record("max_rel_difference_from_w_b", np.abs(got_e / (weight * b) - 1).max())
    assert err_e < 1e-6
    assert np.abs(got_e / (weight * b) - 1).max() > 0.05


# ------------------------------------------------ 6. antenna gains, Stage 1


def _gains_case(scale, seed=3):
    """Three antennas, one interval, two integrations per baseline; data are the model times known gains plus
    noise, with weights set so that w |M|^2 = scale / 0.1^2 (a posterior width of about 0.1 / sqrt(scale))."""
    rng = np.random.default_rng(seed)
    a1 = np.array([0, 0, 1, 0, 0, 1])
    a2 = np.array([1, 2, 2, 1, 2, 2])
    model = (0.8 + 0.5j) * np.array([1.0, 0.7, 1.2, 0.9, 1.1, 0.8]) * np.exp(1j * np.array([0.2, -0.4, 0.9, 0.3, -0.5, 0.6]))
    w = np.full(6, scale / 0.1**2) / np.abs(model) ** 2
    lam, phi = np.array([0.05, -0.08, 0.04]), np.array([0.3, -0.2, 0.1])
    g = np.exp(lam[a1] + lam[a2] + 1j * (phi[a1] - phi[a2])) * model
    vis = g + (rng.normal(size=6) + 1j * rng.normal(size=6)) / np.sqrt(w)
    return model, vis, w, a1, a2


def _virgil_marginal(model, vis, w, a1, a2, tau_a, tau_p, steps=8):
    """virgil's Stage-1 negative log marginal: half chi^2 at the gain solution + half the prior chi^2 + its
    normalisation term (which carries 1/2 log det H and log tau_a tau_p), and the solution."""
    from virgil import _antenna_gains as ag

    n = len(model)
    lam, phi, norm, _ = ag.marginal(model, vis, w, np.zeros(n, np.int32), a1.astype(np.int32), a2.astype(np.int32), 1, 3, tau_a, tau_p, steps)
    lam, phi = np.asarray(lam)[0], np.asarray(phi)[0]
    g = np.exp(lam[a1] + lam[a2] + 1j * (phi[a1] - phi[a2])) * model
    value = 0.5 * np.sum(w * np.abs(vis - g) ** 2) + 0.5 * np.sum(lam**2 / tau_a**2 + phi**2 / tau_p**2) + float(norm)
    return value, np.concatenate([lam, phi])


@pytest.mark.validates("virgil._antenna_gains.marginal", roots=["mathematics"])
def test_laplace_antenna_gain_marginal_matches_gauss_hermite_quadrature_and_improves_with_information():
    """Three antennas, six baseline samples, the documented default widths (0.1 amplitude, 0.5 phase). The
    Stage-1 Laplace value -log integral exp(-chi^2 / 2) N(lam) N(phi) d lam d phi (six dimensions) differs
    from an 11^6-point Gauss-Hermite rule centred on the numerically found mode by less than 0.05 at a
    posterior width of 0.1 and by less than 0.01 at four times the weights, the error shrinks by more than
    2x when the information is quadrupled (Laplace is exact as w grows), and the solution is the
    numerical mode to 1e-3."""
    pytest.importorskip("virgil._antenna_gains")
    errs = []
    for scale in (1.0, 4.0):
        model, vis, w, a1, a2 = _gains_case(scale)
        ref, mode = rt.gain_marginal_brute_force(model, vis, w, a1, a2, 3, 0.1, 0.5)
        got, sol = _virgil_marginal(model, vis, w, a1, a2, 0.1, 0.5)
        errs.append(abs(got - ref))
        record(f"abs_error_scale_{scale:g}", errs[-1])
        record(f"mode_error_scale_{scale:g}", np.abs(sol - mode).max())
        assert np.abs(sol - mode).max() < 1e-3
    assert errs[0] < 0.05 and errs[1] < 0.01
    assert errs[1] < errs[0] / 2


@pytest.mark.validates("virgil._antenna_gains.marginal", roots=["mathematics"], kind="control")
def test_control_a_reference_with_the_wrong_prior_widths_misses_the_laplace_value():
    """Doubling both prior widths in the quadrature reference moves it by far more than the tolerance of the
    test above (so that test does pin the prior widths and their normalisation)."""
    pytest.importorskip("virgil._antenna_gains")
    model, vis, w, a1, a2 = _gains_case(1.0)
    wrong, _ = rt.gain_marginal_brute_force(model, vis, w, a1, a2, 3, 0.1, 0.5, prior_scale=2.0)
    got, _ = _virgil_marginal(model, vis, w, a1, a2, 0.1, 0.5)
    record("control_miss", abs(got - wrong))
    assert abs(got - wrong) > 0.5
