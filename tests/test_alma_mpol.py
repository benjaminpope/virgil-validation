"""virgil.radio's likelihood against MPoL's, as the IM Lup example evaluates it.

MPoL (Czekala et al. 2025) is run in its own environment (`.venv-mpol`,
mpol 0.3.1; MPoL main 8850b48 differs from it in none of the modules used
here) through `external_bridge.mpol_worker`, which imports the multi-EB
example's own ``Net`` (MPoL-dev/examples, 04-IMLup-multi-EB, `sgd.py`) from a
clone pinned at ``EXAMPLES_COMMIT`` (`scripts/setup_external.sh mpol`).
virgil is `virgil.radio` from the pinned commit of `virgil_bridge.mpol`.
The parameter map between the two is in docs/alma_conventions.md ("MPoL and
virgil.radio"); in float64 throughout.

* test 1, loss normalisation (tier A): MPoL's ``log_likelihood`` and
  ``neg_log_likelihood_avg`` equal virgil's whitened chi-squared plus its log
  normalisation plus the explicit constant, with s = g s_M; the map s = s_M is
  the control;
* test 2, forward model (tier A): MPoL's visibilities, virgil's ``data.model``
  and an exact numpy DFT of the pixels (`crosscheck.alma`) on 256 x 256 pixels
  at 5 mas, 10^4 uv points of IM Lup, non-trivial g, offsets and the taper;
  orientation controls;
* test 3, objective and gradients (tier A): -ln L and its gradients with
  respect to the pixels and every nuisance, torch autograd against jax, and
  against exact references;
* test 4, IM Lup scale (tier B): the objective on all 26 ddids and 7
  observations of the stride-1000 subsample.

CI's pinned virgil has no ``virgil.radio`` and CI has no `.venv-mpol`, so the
module skips there.
"""

import os
import pathlib
import subprocess

import numpy as np
import pytest

pytest.importorskip("virgil.radio")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import virgil_bridge.alma as vba  # noqa: E402
import virgil_bridge.mpol as vbm  # noqa: E402
from crosscheck import alma  # noqa: E402
from evidence.plugin import record  # noqa: E402
from external_bridge import _subprocess as sp  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = pathlib.Path(os.environ.get("MPOL_EXAMPLES", ROOT / ".external" / "mpol-examples"))
EXAMPLES_COMMIT = "5a7c3f0b373f62341ca4baf5869f95e916bbabb3"  # MPoL-dev/examples, fetched 2026-10-19
SUBSAMPLE = pathlib.Path("~/data/alma_imlup/subsample_1000").expanduser()
MAS = alma.MAS
OBSID_DICT = {0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9, 10], 2: [11, 12], 3: [13, 14], 4: [15, 16, 17], 5: [18, 19, 20, 21], 6: [22, 23, 24, 25]}
OBS_MAP = vbm.obs_of_ddid(OBSID_DICT, 26)
NPIX, PIXEL, FWHM = 256, 5.0, 0.03  # pixels, mas, arcsec (the example: 1028 pixels, 5 mas, 30 mas)
RTOL_NORM = 1e-12


def _example_commit():
    try:
        out = subprocess.run(["git", "-C", str(EXAMPLES), "rev-parse", "HEAD"], capture_output=True, text=True)
    except OSError:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


pytestmark = [
    pytest.mark.skipif(not sp.available("mpol", "mpol"), reason="MPoL not installed (scripts/setup_external.sh mpol)"),
    pytest.mark.skipif(
        _example_commit() != EXAMPLES_COMMIT,
        reason=f"MPoL-dev/examples not cloned at {EXAMPLES_COMMIT[:8]} in {EXAMPLES} (scripts/setup_external.sh mpol)",
    ),
]


def _task(name, **kw):
    return dict(task=name, examples_dir=str(EXAMPLES), cell_size=kw.pop("cell_size", PIXEL / 1000), npix=kw.pop("npix", NPIX), fwhm=FWHM, obsid_dict=OBSID_DICT, **kw)


@pytest.fixture(scope="module", autouse=True)
def float64_nufft():
    """virgil's NUFFT at eps 1e-12 in float64, restored afterwards."""
    from virgil import set_ft_defaults

    old = set_ft_defaults(ft="nufft", eps_f64=1e-12)
    yield
    set_ft_defaults(**old)


# ------------------------------------------------------------------ data


def subsample_points(n=None, seed=7):
    """Points of the stride-1000 IM Lup subsample (every point if ``n`` is None, else
    a sorted random draw), or, without the subsample, synthetic points with the
    same ddid layout and a spread of weights. Returns the arrays and the source."""
    files = sorted(SUBSAMPLE.glob("ddid_*.npz"))
    rng = np.random.default_rng(seed)
    if len(files) == 26:
        zs = [dict(np.load(f)) for f in files]
        cat = lambda k: np.concatenate([np.asarray(z[k]) for z in zs])  # noqa: E731
        pts = dict(
            u=cat("u"), v=cat("v"), vis=cat("re").astype(float) + 1j * cat("im").astype(float), w=cat("weight").astype(float), freq=cat("freq"),
            ddid=np.concatenate([np.full(len(z["u"]), int(z["ddid"])) for z in zs]),
        )
        source = "IM Lup stride-1000 subsample"
    else:
        m = 42000
        q = 10 ** rng.uniform(np.log10(2e4), 7.0, m)
        phi = rng.uniform(0, np.pi, m)
        pts = dict(u=q * np.cos(phi), v=q * np.sin(phi), vis=0.05 * (rng.normal(size=m) + 1j * rng.normal(size=m)), w=np.exp(rng.normal(7.5, 1.0, m)),
                   freq=rng.uniform(2.3e11, 2.47e11, m), ddid=rng.integers(0, 26, m))
        source = "synthetic uv, IM Lup's ddid layout"
    if n is not None and n < len(pts["u"]):
        keep = np.sort(rng.choice(len(pts["u"]), n, replace=False))
        pts = {k: a[keep] for k, a in pts.items()}
    pts["obs"] = OBS_MAP[pts["ddid"]]
    return pts, source


def asymmetric_image(rng, total_flux=0.253):
    """An IM Lup-like image in virgil's layout, in Jy/arcsec^2: an inclined ring and an off-centre
    blob (not symmetric under any flip) with a 30% pixel-to-pixel texture, on 256 x 256 pixels at 5 mas."""
    img = vba.ring_and_blob(NPIX, PIXEL, radius_mas=400.0, width_mas=60.0, inc_deg=47.0, pa_deg=144.0, blob=(0.25, 150.0, -250.0, 70.0))
    img = img * (1 + 0.3 * rng.uniform(size=img.shape))
    img = np.maximum(img / img.sum(), 1e-8 * img.max() / img.sum())
    return total_flux * img / (PIXEL / 1000) ** 2


def nontrivial_theta(rng, image_sky, log10_sigma=None):
    """MPoL's parameters: the base cube of the image (packed), sigma factors, amplitude
    factors of observations 1-6, offsets in arcsec (up to 10 mas, a phase of a few radians)."""
    return dict(
        base_cube=vbm.softplus_inverse(vbm.pack(image_sky)),
        log10_sigma=rng.uniform(0.0, 0.3, 26) if log10_sigma is None else np.asarray(log10_sigma, float),
        log10_amp=rng.uniform(-0.04, 0.04, 6),
        delta_ra=rng.uniform(-0.01, 0.01, 6),
        delta_dec=rng.uniform(-0.01, 0.01, 6),
    )


# ------------------------------------------------ the two codes and the exact reference


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def exact_model(theta, pts):
    """The exact model visibilities: a numpy DFT of the pixel image, then the nuisances
    (gains, offset ramps and the Gaussian taper, written from their definitions)."""
    n = NPIX
    image = np.log1p(np.exp(theta["base_cube"]))  # Jy / arcsec^2, packed
    flux = np.fft.fftshift(image) * (PIXEL / 1000) ** 2  # Jy per pixel, column c East at (c - n/2) dx
    x = (np.arange(n) - n // 2) * PIXEL * MAS
    v0 = alma.image_visibilities(flux, x, x, pts["u"], pts["v"])
    return v0, _factors(theta, pts) * v0


def _factors(theta, pts):
    gain = 10.0 ** np.concatenate([[0.0], theta["log10_amp"]])[pts["obs"]]
    dra = np.concatenate([[0.0], theta["delta_ra"]])[pts["obs"]] * vbm.ARCSEC
    ddec = np.concatenate([[0.0], theta["delta_dec"]])[pts["obs"]] * vbm.ARCSEC
    q = np.hypot(pts["u"], pts["v"])
    taper = np.exp(-((np.pi * FWHM * vbm.ARCSEC * q) ** 2) / (4 * np.log(2)))
    return gain * np.exp(-2j * np.pi * (dra * pts["u"] + ddec * pts["v"])) * taper


def formula_nll(model, data, weight):
    """-ln L of complex Gaussian data, normalising constant included, in numpy:
    N ln 2 pi - sum ln w + (1/2) sum w |d - M|^2, with the weights actually used."""
    return len(data) * np.log(2 * np.pi) - np.sum(np.log(weight)) + 0.5 * np.sum(weight * np.abs(data - model) ** 2)


def used_weight(theta, pts):
    """MPoL's weights after the example's rescaling: w / (s_M g)^2."""
    s_m = 10.0 ** theta["log10_sigma"][pts["ddid"]]
    gain = 10.0 ** np.concatenate([[0.0], theta["log10_amp"]])[pts["obs"]]
    return pts["w"] / (s_m * gain) ** 2


def exact_nll(theta, pts, v0):
    return formula_nll(_factors(theta, pts) * v0, pts["vis"], used_weight(theta, pts))


def exact_gradients(theta, pts, v0):
    """Exact gradients of the exact -ln L: central differences of the closed form for the
    nuisances (the unmodified visibilities ``v0`` do not depend on them) and the adjoint DFT
    for the pixels (through the softplus)."""
    out = {}
    for key in ("log10_sigma", "log10_amp", "delta_ra", "delta_dec"):
        g = np.zeros_like(theta[key])
        for i in range(len(g)):
            h = 1e-6
            up, dn = {**theta, key: theta[key].copy()}, {**theta, key: theta[key].copy()}
            up[key][i] += h
            dn[key][i] -= h
            g[i] = (exact_nll(up, pts, v0) - exact_nll(dn, pts, v0)) / (2 * h)
        out[key] = g
    w = used_weight(theta, pts)
    c = _factors(theta, pts)
    resid = pts["vis"] - c * v0
    # d(nll)/d(pixel flux A at unpacked (r, c)) = -Re sum_k w conj(d - M) c_k exp(-2 pi i (u x_c + v y_r))
    n = NPIX
    x = (np.arange(n) - n // 2) * PIXEL * MAS
    d_flux = -alma.adjoint_image(w * np.conj(resid) * c, x, x, pts["u"], pts["v"])
    sigmoid = 1.0 / (1.0 + np.exp(-theta["base_cube"]))
    out["base_cube"] = np.fft.ifftshift(d_flux) * (PIXEL / 1000) ** 2 * sigmoid  # back to the packed layout
    return out


def run_mpol(theta, pts, tmp, *, grad=True):
    np.savez(tmp / "in.npz", u=pts["u"], v=pts["v"], ddid=pts["ddid"], data_re=pts["vis"].real, data_im=pts["vis"].imag, weight=pts["w"],
             **{k: np.asarray(a) for k, a in theta.items()})
    res = sp.run("mpol", "mpol_worker.py", _task("objective", npz=str(tmp / "in.npz"), out_npz=str(tmp / "out.npz"), grad=grad))
    z = dict(np.load(tmp / "out.npz"))
    z["model"] = z["vis_model_re"] + 1j * z["vis_model_im"]
    z["model_sparse"] = z["vis_model_sparse_re"] + 1j * z["vis_model_sparse_im"]
    return z, res


def run_virgil(theta, pts, *, grad=True, wrong_scene=None, wrong_image=None):
    """virgil's model, -ln L (written out from the term's whitened residuals and log
    normalisation) and, with ``grad``, its jax gradients with respect to MPoL's parameters."""
    from virgil.radio import CalibrationNuisance, ComplexVisData

    data = ComplexVisData.from_arrays(pts["u"], pts["v"], pts["vis"], pts["w"], pts["freq"], ddid=pts["ddid"], obs=pts["obs"])
    nuis = CalibrationNuisance(data, ref_obs=0, alpha=False, taper_fwhm_mas=1000 * FWHM)
    scene = wrong_scene or vbm.imlup_scene(NPIX, PIXEL)
    term = data.term(scene, nuis)
    th = {k: jnp.asarray(a) for k, a in theta.items()}
    obs_of_ddid = jnp.asarray(OBS_MAP)

    def values_of(t):
        vals = vbm.virgil_values(t, obs_of_ddid, PIXEL)
        if wrong_image is not None:
            vals["log_brightness"] = wrong_image(vals["log_brightness"])
        return vals

    vals = values_of(th)
    out = dict(
        model=np.asarray(data.model(scene(vals), vals, nuis)),
        nll=float(vbm.neg_log_likelihood(term, vals, data)),
        loglike=float(term.loglike(vals)),
        data=data, term=term, values=vals,
    )
    if grad:
        out["grad"] = {k: np.asarray(a) for k, a in jax.grad(lambda t: vbm.neg_log_likelihood(term, values_of(t), data))(th).items()}
    return out


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    """One 10^4-point, 256^2, non-trivial-nuisance problem through MPoL, virgil and the exact references."""
    rng = np.random.default_rng(11)
    pts, source = subsample_points(10_000)
    theta = nontrivial_theta(rng, asymmetric_image(rng))
    tmp = tmp_path_factory.mktemp("mpol_case")
    mp, info = run_mpol(theta, pts, tmp)
    vg = run_virgil(theta, pts)
    v0, ex = exact_model(theta, pts)
    return dict(pts=pts, source=source, theta=theta, mpol=mp, info=info, virgil=vg, v0=v0, exact=ex)


# ------------------------------------------- 1. loss normalisation (A)


def _normalisation_problem(rng, m=3000):
    from virgil.radio import CalibrationNuisance, ComplexVisData

    ddid = rng.integers(0, 26, m)
    ddid[:26] = np.arange(26)  # every ddid occurs
    obs = OBS_MAP[ddid]
    u, v = rng.uniform(-5e6, 5e6, (2, m))
    w = np.exp(rng.normal(5.0, 1.0, m))
    freq = rng.uniform(2.3e11, 2.47e11, m)
    scene = vba.point_scene([(0.6, 100.0, 50.0), (0.4, -200.0, 300.0)])
    shape = ComplexVisData.from_arrays(u, v, np.zeros(m, complex), w, freq, ddid=ddid, obs=obs)
    nuis = CalibrationNuisance(shape, ref_obs=0, offsets=False, alpha=False)
    unit = nuis.defaults()
    unit["cal.flux"] = 0.3
    model1 = np.asarray(shape.model(scene(unit), unit, nuis))  # g = 1, s = 1: the model before MPoL's own gain
    vis = 1.05 * model1 + (rng.normal(size=m) + 1j * rng.normal(size=m)) / np.sqrt(w)
    data = ComplexVisData.from_arrays(u, v, vis, w, freq, ddid=ddid, obs=obs)
    gain = 10.0 ** np.concatenate([[0.0], rng.uniform(-0.1, 0.1, 6)])  # g != 1
    s_m = 10.0 ** rng.uniform(-0.1, 0.3, 26)  # s_M != 1
    return dict(data=data, nuis=nuis, scene=scene, unit=unit, model1=model1, vis=vis, w=w, ddid=ddid, obs=obs, gain=gain, s_m=s_m)


def _virgil_nll(p, sigma):
    """-ln L from virgil with the given per-ddid error scales, and its pieces."""
    vals = dict(p["unit"])
    vals["cal.gain"] = jnp.asarray(p["gain"][1:])
    vals["cal.sigma"] = jnp.asarray(sigma)
    term = p["data"].term(p["scene"], p["nuis"])
    r = np.asarray(term(vals))
    half_chi2, log_norm = 0.5 * float(np.sum(r**2)), float(term.log_norm(vals))
    # the explicit constant, per ddid: sum over its points of (ln 2 pi - ln w_k)
    const = sum(np.sum(np.log(2 * np.pi) - np.log(p["w"][p["ddid"] == d])) for d in range(26))
    return dict(nll=half_chi2 + log_norm + const, half_chi2=half_chi2, log_norm=log_norm, const=const, loglike=float(term.loglike(vals)))


def _mpol_loglike(p, tmp):
    np.savez(tmp / "norm.npz", model_re=p["model1"].real, model_im=p["model1"].imag, data_re=p["vis"].real, data_im=p["vis"].imag, weight=p["w"], ddid=p["ddid"],
             log10_sigma=np.log10(p["s_m"]), log10_amp=np.log10(p["gain"][1:]))
    return sp.run("mpol", "mpol_worker.py", _task("loglike", npz=str(tmp / "norm.npz"), npix=8))


@pytest.mark.validates("virgil.radio.ComplexVisData.term", "virgil.radio.CalibrationNuisance", roots=["mpol"])
def test_mpol_log_likelihood_equals_virgil_with_s_equal_g_times_s_mpol(tmp_path):
    """On random complex data, model and weights, with g != 1 and s_M != 1: MPoL's
    `log_likelihood` = -(1/2 sum r^2 + log_norm) - sum_d (N_d ln 2 pi - sum_k ln w_k), with
    virgil's s = g s_M, to 1e-12 relative, and `neg_log_likelihood_avg` * 2N is -ln L as well."""
    p = _normalisation_problem(np.random.default_rng(4))
    sigma = p["s_m"] * p["gain"][OBS_MAP]  # the map: virgil s = g * s_MPoL, per ddid
    v = _virgil_nll(p, sigma)
    m = _mpol_loglike(p, tmp_path)
    n = len(p["w"])
    assert m["n"] == n
    expected_ll = -(v["half_chi2"] + v["log_norm"]) - v["const"]
    err_ll = abs(m["log_likelihood"] - expected_ll) / abs(expected_ll)
    err_avg = abs(m["neg_log_likelihood_avg"] * 2 * n + m["log_likelihood"]) / abs(m["log_likelihood"])
    err_loglike = abs(m["log_likelihood"] - v["loglike"]) / abs(v["loglike"])  # virgil's own normalised log density
    record("rel_error_log_likelihood", err_ll)
    record("rel_error_neg_log_likelihood_avg_times_2N", err_avg)
    record("rel_error_vs_virgil_loglike", err_loglike)
    record("mpol_log_likelihood", m["log_likelihood"])
    assert err_ll < RTOL_NORM
    assert err_avg < RTOL_NORM
    assert err_loglike < RTOL_NORM


@pytest.mark.validates("virgil.radio.ComplexVisData.term", "virgil.radio.CalibrationNuisance", roots=["mpol"], kind="control")
def test_control_the_map_s_equal_s_mpol_does_not_match(tmp_path):
    """The wrong map (virgil s = s_M, forgetting that MPoL also divides the weights by g^2) misses MPoL's
    log likelihood by 2 sum ln g plus a change of chi-squared: far more than rounding, so the 1e-12 test above pins s = g s_M."""
    p = _normalisation_problem(np.random.default_rng(4))
    v = _virgil_nll(p, p["s_m"])
    m = _mpol_loglike(p, tmp_path)
    miss = abs(m["log_likelihood"] + v["nll"]) / abs(m["log_likelihood"])
    record("rel_miss_with_s_equal_s_mpol", miss)
    assert miss > 1e-3


# ------------------------------------------------ 2. forward model (A)


@pytest.mark.validates("virgil.radio.ComplexVisData.model", "virgil.radio.CalibrationNuisance", "virgil.models.Image", roots=["mpol", "mathematics"])
def test_mpol_virgil_and_the_exact_dft_agree_on_the_forward_model(case):
    """MPoL's forward visibilities (torchkbnufft, the example's Net with its nuisance layers),
    virgil's `data.model` and a float64 numpy DFT, on the IM Lup uv of 10^4 points: virgil is
    exact to 1e-10 (NUFFT eps 1e-12); MPoL's table interpolation errs by 6e-4; and MPoL against
    virgil is within the sum of the two (the triangle inequality: the content is that each
    agrees with the DFT, and MPoL's error is its transform's, not a convention)."""
    ex, mp, vg = case["exact"], case["mpol"]["model"], case["virgil"]["model"]
    e_mpol, e_virgil, e_both = rel(mp, ex), rel(vg, ex), rel(mp, vg)
    record("rel_l2_mpol_vs_dft", e_mpol)
    record("rel_l2_virgil_vs_dft", e_virgil)
    record("rel_l2_mpol_vs_virgil", e_both)
    record("max_abs_jy_mpol_vs_dft", np.abs(mp - ex).max())
    record("max_abs_jy_virgil_vs_dft", np.abs(vg - ex).max())
    record("max_abs_jy_mpol_vs_virgil", np.abs(mp - vg).max())
    assert e_virgil < 1e-10
    assert e_mpol < 2e-3  # torchkbnufft's default table interpolation
    assert np.linalg.norm(mp - vg) <= np.linalg.norm(mp - ex) + np.linalg.norm(vg - ex) + 1e-12
    assert case["info"]["taper_dtype"] == "torch.float32"  # GaussConvFourier's buffer: part of the measured error


@pytest.mark.validates("virgil.radio.ComplexVisData.model", "virgil.models.Image", roots=["mpol", "mathematics"])
def test_mpol_with_its_accurate_nufft_agrees_to_1e_5(case):
    """With `sparse_matrices=True`, MPoL's own more accurate option, the same model matches the
    exact DFT and virgil to 1e-5: the 6e-4 above is table interpolation, so every convention in the map
    (orientation, centre, offsets, taper, gains) is right to that level, not merely to 6e-4."""
    ex, sp_model, vg = case["exact"], case["mpol"]["model_sparse"], case["virgil"]["model"]
    e_dft, e_virgil = rel(sp_model, ex), rel(sp_model, vg)
    record("rel_l2_mpol_sparse_vs_dft", e_dft)
    record("rel_l2_mpol_sparse_vs_virgil", e_virgil)
    assert e_dft < 1e-5
    assert e_virgil < 1e-5


WRONG_MAPS = {
    "no_half_pixel_offset": dict(scene=lambda: vbm.imlup_scene(NPIX, PIXEL, half_pixel=False), image=None),
    "no_flip": dict(scene=None, image=lambda eta: eta[::-1, ::-1]),  # virgil given MPoL's unpacked layout
    "ra_not_flipped": dict(scene=None, image=lambda eta: eta[:, ::-1]),
    "dec_not_flipped": dict(scene=None, image=lambda eta: eta[::-1, :]),
}


@pytest.mark.validates("virgil.radio.ComplexVisData.model", "virgil.models.Image", roots=["mpol"], kind="control")
@pytest.mark.parametrize("wrong", sorted(WRONG_MAPS))
def test_control_a_wrong_pixel_map_misses_mpol_by_far_more_than_the_matched_map_does(case, wrong):
    """Dropping the half-pixel offset, or not reversing the columns (RA increases to the left), the rows,
    or both, makes virgil miss MPoL's visibilities (the accurate-NUFFT path, which the matched map
    reproduces to 1e-5) by more than 1e-3, up to order one: the agreement above pins the orientation and
    the phase centre. A missing half pixel is the smallest of them: 2.5 mas is a 1% change of the
    visibilities of this tapered image."""
    spec = WRONG_MAPS[wrong]
    vg = run_virgil(case["theta"], case["pts"], grad=False, wrong_scene=spec["scene"]() if spec["scene"] else None, wrong_image=spec["image"])
    miss = rel(case["mpol"]["model_sparse"], vg["model"])
    record("rel_l2_miss", miss)
    assert miss > 1e-3


# ------------------------------------------------ 3. objective and gradients (A)


@pytest.mark.validates("virgil.radio.ComplexVisData.term", "virgil.radio.CalibrationNuisance", roots=["mpol", "mathematics"])
def test_objective_agrees_and_each_equals_its_own_loss_formula(case):
    """-ln L (normalising constant included) at the same parameters: MPoL's equals the closed
    formula N ln 2 pi - sum ln w + 1/2 sum w |d - M|^2 evaluated on MPoL's own model and weights, virgil's the same
    on its model, to 1e-12; and the two differ by what their models differ by (1e-4 of |ln L| at most)."""
    pts, th = case["pts"], case["theta"]
    nll_m, nll_v = float(case["mpol"]["neg_log_likelihood"]), case["virgil"]["nll"]
    nll_exact = exact_nll(th, pts, case["v0"])
    own_m = formula_nll(case["mpol"]["model"], pts["vis"], case["mpol"]["weight_used"])
    own_v = formula_nll(case["virgil"]["model"], pts["vis"], used_weight(th, pts))
    record("neg_log_likelihood_mpol", nll_m)
    record("neg_log_likelihood_virgil", nll_v)
    record("neg_log_likelihood_exact_dft", nll_exact)
    record("rel_mpol_vs_exact", abs(nll_m - nll_exact) / abs(nll_exact))
    record("rel_virgil_vs_exact", abs(nll_v - nll_exact) / abs(nll_exact))
    record("rel_mpol_vs_virgil", abs(nll_m - nll_v) / abs(nll_v))
    assert abs(nll_m - own_m) < RTOL_NORM * abs(nll_m)
    assert abs(nll_v - own_v) < RTOL_NORM * abs(nll_v)
    assert abs(case["virgil"]["loglike"] + nll_v) < RTOL_NORM * abs(nll_v)  # virgil's own log density
    assert abs(nll_v - nll_exact) < 1e-9 * abs(nll_exact)
    assert abs(nll_m - nll_exact) < 1e-4 * abs(nll_exact)
    assert np.allclose(case["mpol"]["weight_used"], used_weight(th, pts), rtol=1e-12)  # MPoL's weights are w / (s_M g)^2


GRADIENT_KEYS = ("base_cube", "log10_sigma", "log10_amp", "delta_ra", "delta_dec")


@pytest.mark.validates("virgil.radio.ComplexVisData.term", "virgil.radio.CalibrationNuisance", "virgil.models.Image", roots=["mpol", "mathematics"])
def test_gradients_of_the_objective_agree_with_respect_to_the_pixels_and_every_nuisance(case):
    """d(-ln L)/d(parameter) for the 65,536 base-cube pixels (packed layout, through the softplus and the
    orientation map), 26 sigma factors, 6 amplitude factors and 12 offsets: torch autograd through the
    example's Net against jax through virgil's term with the map s = g s_M applied by the chain rule (the
    jax gradient is of the composite). Both against exact references (adjoint DFT for the pixels,
    central differences of the closed form for the nuisances)."""
    pts, th = case["pts"], case["theta"]
    exact = exact_gradients(th, pts, case["v0"])
    for key in GRADIENT_KEYS:
        gm, gv, ge = case["mpol"]["grad_" + key], case["virgil"]["grad"][key], exact[key]
        record(f"rel_l2_grad_{key}_mpol_vs_virgil", rel(gm, gv))
        record(f"rel_l2_grad_{key}_virgil_vs_exact", rel(gv, ge))
        record(f"rel_l2_grad_{key}_mpol_vs_exact", rel(gm, ge))
        record(f"norm_grad_{key}", np.linalg.norm(ge))
        assert rel(gv, ge) < 1e-6, key
        assert rel(gm, ge) < 1e-2, key
        assert rel(gm, gv) < 1e-2, key


@pytest.mark.validates("virgil.radio.CalibrationNuisance", roots=["mpol", "mathematics"])
def test_the_sigma_and_gain_gradients_obey_the_chain_rule_of_the_map(case):
    """MPoL's gradient with respect to log10 g of an observation is the sum of virgil's gradients with
    respect to its gain and to the s of every ddid in it (s = g s_M), times g ln 10 for the gain; written out
    from virgil's gradients with respect to its own sites and compared with MPoL's autograd."""
    th, pts = case["theta"], case["pts"]
    vg_data, term = case["virgil"]["data"], case["virgil"]["term"]
    vals = {k: jnp.asarray(a) for k, a in case["virgil"]["values"].items()}
    own = jax.grad(lambda v: vbm.neg_log_likelihood(term, v, vg_data))(vals)  # d/d(cal.gain), d/d(cal.sigma), ...
    gain = 10.0 ** th["log10_amp"]
    got = np.zeros(6)
    for o in range(1, 7):
        ddids = OBSID_DICT[o]
        got[o - 1] = np.log(10) * (gain[o - 1] * float(own["cal.gain"][o - 1]) + gain[o - 1] * sum(float(own["cal.sigma"][d]) * 10.0 ** th["log10_sigma"][d] for d in ddids))
    mp = case["mpol"]["grad_log10_amp"]
    record("rel_l2_chain_rule_vs_mpol", rel(got, mp))
    assert rel(got, mp) < 1e-2


# ------------------------------------------------ 4. IM Lup scale (B)

# the model-free weight-scale estimates of docs/alma_conventions.md (rounded): the sigma factors MPoL's fit
# is expected to find; the amplitude factors and offsets are those of a few per cent and a few mas
IMLUP_SIGMA = [1.12, 1.15, 1.13, 1.13, 1.13, 1.14, 1.14, 1.14, 1.14, 1.13, 1.14, 1.11, 1.10, 1.08, 1.06, 1.13, 1.13, 1.07, 1.67, 1.67, 1.66, 1.75, 1.99, 1.99, 1.99, 2.08]


@pytest.mark.validates("pipeline:alma-imlup", roots=["mpol", "mathematics"], tier="B", headline="MPoL's objective on IM Lup")
@pytest.mark.skipif(len(list(SUBSAMPLE.glob("ddid_*.npz"))) != 26, reason="the stride-1000 subsample is not in ~/data/alma_imlup/subsample_1000")
def test_the_objective_matches_mpol_on_all_26_ddids_and_7_observations_of_im_lup(tmp_path):
    """On the whole stride-1000 subsample (42,424 visibilities, 26 ddids, 7 observations) at MPoL-like
    parameters (the model-free sigma scales as sigma factors, amplitude factors within 4%, offsets of
    a few mas, 30 mas taper, an IM Lup-like image of 0.253 Jy on 256^2 pixels at 5 mas), the
    objective from MPoL's example, from virgil and from an exact DFT agree."""
    rng = np.random.default_rng(23)
    pts, _ = subsample_points(None)
    assert len(np.unique(pts["ddid"])) == 26 and len(np.unique(pts["obs"])) == 7 and len(pts["u"]) > 40_000
    theta = nontrivial_theta(rng, asymmetric_image(rng), log10_sigma=np.log10(IMLUP_SIGMA))
    theta["delta_ra"] = rng.uniform(-0.004, 0.004, 6)  # arcsec
    theta["delta_dec"] = rng.uniform(-0.004, 0.004, 6)
    mp, _ = run_mpol(theta, pts, tmp_path, grad=False)
    vg = run_virgil(theta, pts, grad=False)
    v0, ex = exact_model(theta, pts)
    nll_exact = exact_nll(theta, pts, v0)
    nll_m, nll_v = float(mp["neg_log_likelihood"]), vg["nll"]
    record("n_visibilities", len(pts["u"]))
    record("neg_log_likelihood_mpol", nll_m)
    record("neg_log_likelihood_virgil", nll_v)
    record("neg_log_likelihood_exact_dft", nll_exact)
    record("rel_mpol_vs_exact", abs(nll_m - nll_exact) / abs(nll_exact))
    record("rel_virgil_vs_exact", abs(nll_v - nll_exact) / abs(nll_exact))
    record("rel_mpol_vs_virgil", abs(nll_m - nll_v) / abs(nll_v))
    record("rel_l2_model_mpol_vs_dft", rel(mp["model"], ex))
    record("rel_l2_model_virgil_vs_dft", rel(vg["model"], ex))
    assert abs(nll_v - nll_exact) < 1e-9 * abs(nll_exact)
    assert abs(nll_m - nll_exact) < 1e-4 * abs(nll_exact)
    assert abs(nll_m - nll_v) < 1e-4 * abs(nll_v)
