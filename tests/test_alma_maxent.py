"""The IM Lup MaxEnt analysis (``scripts/alma_imlup_maxent.py``, ``virgil_bridge.alma``), on the laptop.

The design is ``design/alma_imlup_p3.md``. Nothing here runs the full data: the subsample
(<= 4.3e4 points) or a synthetic set, at most 128^2 pixels, one process.

* R1 (tier A, root mpol): the R objective, the data term plus MPoL's entropy of the tapered image
  at ``lam = 2 N lam_M``, equals ``2 N`` times MPoL's per-sample loss from the example's own ``Net``
  and ``mpol.losses.entropy``. Controls: virgil's `MaxEntropy`, or S_M of the untapered image, in its place.
* R2: ``fit --config R`` falls from its start and writes its files; the ``score`` helper equals the term.
* R3: ``folds``, ``cv-task`` and ``cv-aggregate`` on 2 lambda x 2 folds: disjoint, covering, Hermitian-closed.
* R4: ``compare`` on MPoL's export format (mocked from our own fit until W8's worker is merged) and on
  a self-comparison, which gives zero differences.

Everything needs ``virgil.radio`` (virgil from c446485): CI's pinned virgil has none, so the module
skips there, as ``test_alma_radio.py`` does. R1 also needs ``.venv-mpol`` and the examples clone
(``MPOL_PYTHON``, ``MPOL_EXAMPLES``).
"""

import json
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("virgil.radio")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import virgil_bridge.alma as va  # noqa: E402
import virgil_bridge.mpol as vbm  # noqa: E402
from crosscheck import alma_maxent as cx  # noqa: E402
from evidence.plugin import record  # noqa: E402
from external_bridge import _subprocess as sp  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import alma_imlup_maxent as script  # noqa: E402
import test_alma_mpol as tm  # noqa: E402

PIX = 5.0
LAM_M = 1e-9
# R1's data term: virgil's NUFFT (eps 1e-12) against MPoL's own accurate (sparse-matrix) NuFFT, which is a float32 interpolation
# (test_alma_mpol: 1e-5 on visibilities); the brief's 1e-6 is met by the entropy (1e-7), not by this.
RTOL_R1 = 1e-5


def _call(*argv):
    script.main([str(a) for a in argv])


# ------------------------------------------------------------------ data


def synthetic_dir(path, n=400, n_obs=7, per_obs=2, seed=3):
    """A small set in the vv format (``ddid_NN.npz``: u, v, complex vis, weight, freq, ddid, obsid): ``n_obs``
    observations of ``per_obs`` ddids, smooth visibilities plus noise at the quoted weights."""
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True, exist_ok=True)
    d = 0
    for o in range(n_obs):
        for _ in range(per_obs):
            q = 10 ** rng.uniform(4.3, 6.6, n)
            phi = rng.uniform(0, np.pi, n)
            w = np.exp(rng.normal(8.0, 0.3, n))
            vis = 0.15 * np.exp(-((q / 1.5e6) ** 2)) * np.exp(-2j * np.pi * q * np.cos(phi) * 40 * cx.MAS)
            vis = vis + (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(w)
            np.savez(path / f"ddid_{d:02d}.npz", u=q * np.cos(phi), v=q * np.sin(phi), vis=vis, weight=w, freq=np.full(n, 2.3e11 + 1e9 * d), ddid=d, obsid=o)
            d += 1
    return path


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    return synthetic_dir(tmp_path_factory.mktemp("syn"))


@pytest.fixture(scope="module")
def r_fit(syn, tmp_path_factory):
    """``fit --config R`` at 128^2: both stages, 20 steps each."""
    out = tmp_path_factory.mktemp("rfit")
    _call("fit", "--config", "R", "--data", syn, "--out", out, "--npix", 128, "--max-steps", 20)
    return out


# ------------------------------------------------------------------ the pieces, in numpy


@pytest.mark.validates("virgil_bridge.alma", roots=["mathematics"])
def test_entropy_of_the_tapered_image_equals_the_numpy_formula():
    """`mpol_entropy_jax` (softmax image, flux, FFT taper) against ``crosscheck.alma_maxent`` (the EHT-IV formula on a numpy
    Gaussian convolution of the same image), to 1e-12; the taper changes S_M by far more than that."""
    rng = np.random.default_rng(2)
    n, flux = 64, 0.253
    eta = rng.normal(size=(n, n)) * 2 - 0.01 * np.add.outer(np.arange(n), np.arange(n)) ** 1.0
    img = flux * np.exp(eta) / np.exp(eta).sum() / (PIX / 1000) ** 2
    ref = cx.mpol_entropy(cx.convolve(img, PIX, 30.0), 1e-4, 0.253)
    got = float(va.mpol_entropy_jax(jnp.asarray(eta), flux, pixel_mas=PIX, taper_mas=30.0, prior=1e-4, f_tot=0.253))
    bare = float(va.mpol_entropy_jax(jnp.asarray(eta), flux, pixel_mas=PIX, taper_mas=0.0, prior=1e-4, f_tot=0.253))
    record("rel_error_vs_numpy", abs(got - ref) / abs(ref))
    record("rel_change_from_the_taper", abs(bare - ref) / abs(ref))
    assert abs(got - ref) < 1e-12 * abs(ref)
    assert abs(bare - ref) > 1e-4 * abs(ref)


@pytest.mark.validates("virgil_bridge.alma", roots=["mathematics"])
def test_lambda_conversion_gives_virgils_maxentropy_the_same_pull_on_the_shape():
    """C11: for images of one flux, S_M(b1) - S_M(b2) = F / (F_tot dx^2) [K(b1) - K(b2)] exactly (untapered), with K the
    relative entropy of virgil's `MaxEntropy` against a flat image: lam_F = lam_R F / (F_tot dx^2) has the same difference."""
    from virgil.imaging import MaxEntropy

    rng = np.random.default_rng(5)
    n, flux = 16, 0.2
    etas = [rng.normal(size=(n, n)) for _ in range(2)]
    s = [float(va.mpol_entropy_jax(jnp.asarray(e), flux, pixel_mas=PIX, taper_mas=0.0, prior=1e-4, f_tot=0.253)) for e in etas]
    from virgil import Image

    k = [MaxEntropy(1.0).value(Image(e, pixel_scale_mas=PIX)) for e in etas]
    lam_r, lam_f = 0.3, va.lam_f_equivalent(0.3, flux=flux)
    assert abs(lam_r * (s[0] - s[1]) - lam_f * (float(k[0]) - float(k[1]))) < 1e-9 * abs(lam_r * (s[0] - s[1]))
    assert abs(va.lam_f_equivalent(va.lam_r(42_421_131)) - va.lam_r(42_421_131) / (PIX / 1000) ** 2) < 1e-9


@pytest.mark.validates("virgil_bridge.alma", roots=["mathematics"])
def test_config_f_image_is_the_sky_whose_beam_attenuated_flux_is_cal_flux():
    """In F the fitted pixels are sky x beam: image_of(..., log_pb) times the beam sums to cal.flux (and exceeds it
    without the beam); in R (no beam) the image sums to cal.flux."""
    rng = np.random.default_rng(1)
    n = 48
    eta = rng.normal(size=(n, n))
    lpb = va.beam_log(n, 300.0)  # a coarse grid so that the beam matters: the field is 14 arcsec
    v = {"log_brightness": eta, "cal.flux": 0.2}
    dx2 = (300.0 / 1000) ** 2
    sky = va.image_of(v, n, 300.0, lpb)
    assert abs(np.sum(sky * np.exp(lpb)) * dx2 - 0.2) < 1e-12
    assert np.sum(sky) * dx2 > 0.2 * (1 + 1e-4)
    assert abs(np.sum(va.image_of(v, n, 300.0)) * dx2 - 0.2) < 1e-12


@pytest.mark.validates("crosscheck.alma_maxent", roots=["mathematics"])
def test_sky_wcs_and_the_elliptical_beam_have_the_documented_orientation():
    """A pixel West of the phase centre has a smaller RA, one North a larger Dec (the FITS array is flipped), and a
    beam of position angle 0 is elongated North-South, one of 90 degrees East-West."""
    n = 64
    w = cx.sky_wcs(n, PIX, 239.0, -37.9)
    r0, c0 = cx.phase_centre_cell(n)
    # FITS array index (row from the South, column): a cell 3 columns towards West and 2 rows towards North
    ra, dec = w.all_pix2world([c0 + 3], [(n - 1 - r0) + 2], 0)
    assert ra[0] < 239.0 and dec[0] > -37.9
    assert abs((239.0 - ra[0]) * np.cos(np.deg2rad(-37.9)) * 3.6e6 - 3 * PIX) < 1e-3  # mas, tangent plane at the centre
    img = np.zeros((n, n))
    img[r0, c0] = 1.0
    for pa, (axis_long) in ((0.0, 0), (90.0, 1)):
        b = cx.convolve(img, PIX, 60.0, 20.0, pa)
        assert abs(b.sum() - 1) < 1e-12
        rr, cc = np.indices(b.shape)
        var = [np.sum(b * (g - g0) ** 2) for g, g0 in ((rr, r0), (cc, c0))]
        assert var[axis_long] > 4 * var[1 - axis_long]
    # PA = 45 degrees (East of North) points the major axis North-East: North is decreasing row and East decreasing column
    # in this layout, so the row-column covariance is positive; -45 degrees gives a negative one
    for pa, sign in ((45.0, 1), (-45.0, -1)):
        b = cx.convolve(img, PIX, 60.0, 20.0, pa)
        rr, cc = np.indices(b.shape)
        assert sign * np.sum(b * (rr - r0) * (cc - c0)) > 0


# ------------------------------------------------------------------ R1


def _r1_case(lam_m, rng_seed=17):
    """The R objective and MPoL's loss at one random state, on the subsample (or synthetic uv) at 128^2."""
    rng = np.random.default_rng(rng_seed)
    npix = 128
    pts, source = tm.subsample_points(None)
    img = vbm_image = tm.vba.ring_and_blob(npix, PIX, radius_mas=180.0, width_mas=40.0, inc_deg=47.0, pa_deg=144.0, blob=(0.25, 100.0, -150.0, 40.0))
    img = img * (1 + 0.3 * rng.uniform(size=img.shape))
    sky = 0.253 * img / img.sum() / (PIX / 1000) ** 2
    theta = tm.nontrivial_theta(rng, sky)
    return npix, pts, source, theta, sky, vbm_image


def _mpol_side(npix, pts, theta, lam_m, tmp):
    """MPoL's data term with its accurate NuFFT (formula_nll on the example's own model and weights; the formula
    equals ``losses.log_likelihood`` to 1e-12, test_alma_mpol) and its entropy from ``mpol.losses.entropy`` on its tapered image."""
    np.savez(tmp / "in.npz", u=pts["u"], v=pts["v"], ddid=pts["ddid"], data_re=pts["vis"].real, data_im=pts["vis"].imag, weight=pts["w"],
             **{k: np.asarray(a) for k, a in theta.items()})
    task = tm._task("objective", npz=str(tmp / "in.npz"), out_npz=str(tmp / "out.npz"), grad=False, npix=npix, cell_size=PIX / 1000)
    info = sp.run("mpol", "mpol_worker.py", task)
    tm.check_mocked(info)
    z = dict(np.load(tmp / "out.npz"))
    nll = tm.formula_nll(z["vis_model_sparse_re"] + 1j * z["vis_model_sparse_im"], pts["vis"], z["weight_used"])
    reg = sp.run("mpol", "mpol_worker.py", dict(task="regularisers", images=[np.maximum(z["image_packed"][0], 1e-30).tolist()], priors=[1e-4], tv_epsilon=1e-3))
    s_mpol = reg["results"][0]["entropy"] / 0.253  # the worker's entropy has tot_flux = 1
    n = len(pts["u"])
    const = n * va.LN2PI - np.sum(np.log(z["weight_used"]))  # the part of -ln L that does not depend on the model
    return dict(nll=nll, entropy=s_mpol, loss_per_sample=nll / (2 * n) + lam_m * s_mpol, n=n, const=const, tapered=z["image_packed"][0])


def _virgil_side(npix, pts, theta, lam_m):
    from virgil.radio import ComplexVisData

    data = ComplexVisData.from_arrays(pts["u"], pts["v"], pts["vis"], pts["w"], pts["freq"], ddid=pts["ddid"], obs=pts["obs"])
    n = len(pts["u"])
    lam = va.lam_r(n, lam_m)
    prob = va.build_problem(data, "R", lam=lam, npix=npix, pixel_mas=PIX)
    values = vbm.virgil_values({k: jnp.asarray(a) for k, a in theta.items()}, jnp.asarray(tm.OBS_MAP), PIX)
    sc = va.score(data, prob.scene, values, prob.nuis)
    return data, prob, values, sc, lam


@pytest.mark.validates("virgil_bridge.alma", roots=["mpol", "mathematics"])
@pytest.mark.skipif(not sp.available("mpol", "mpol"), reason="MPoL not installed (scripts/setup_external.sh mpol)")
@pytest.mark.skipif(tm._example_commit() != tm.EXAMPLES_COMMIT, reason="MPoL-dev/examples not cloned at the pinned commit")
def test_r1_the_r_objective_equals_2n_times_mpols_loss(tmp_path):
    """R1. At random image and nuisance values on the IM Lup subsample (128^2 pixels at 5 mas, 30 mas taper), the R objective
    -ln L + lam_R S_M at lam_R = 2 N lam_M (C4) equals 2 N (neg_log_likelihood_avg + lam_M entropy) of the example's own Net
    and mpol.losses (accurate NuFFT), to 1e-5 of the model-dependent part (the constants cancel). lam_M = 1e-5 makes the penalty several times |nll|, so that
    1e-5 of the total pins S_M to 1e-5; the S_M alone agrees to the float32 of MPoL's taper buffer, and lam_M = 1e-9 (the paper's)
    is checked as well."""
    npix, pts, source, theta, sky, _ = _r1_case(LAM_M)
    record("uv_is_im_lup_subsample", float(source.startswith("IM Lup")))
    m = _mpol_side(npix, pts, theta, LAM_M, tmp_path)
    data, prob, values, sc, _ = _virgil_side(npix, pts, theta, LAM_M)
    jv = {k: jnp.asarray(v) for k, v in values.items()}
    s_virgil = float(prob.terms[1].entropy(jv))
    n = m["n"]
    assert m["n"] == len(pts["u"]) <= 43_000
    rel_s = abs(s_virgil - m["entropy"]) / abs(m["entropy"])
    record("rel_error_entropy_S_M", rel_s)
    chi2_part = m["nll"] - m["const"]  # 1/2 sum w |r|^2: the constants N ln 2 pi - sum ln w cancel exactly and would dilute the bound
    record("rel_error_data_term_total", abs(sc["nll"] - m["nll"]) / abs(m["nll"]))
    record("rel_error_data_term_chi2_part", abs(sc["nll"] - m["nll"]) / abs(chi2_part))
    assert rel_s < 1e-7  # float32 taper buffer in MPoL (P12): 3.5e-8
    assert abs(sc["nll"] - m["nll"]) < RTOL_R1 * abs(chi2_part)
    for lam_m in (1e-5, LAM_M):
        obj = sc["nll"] + va.lam_r(n, lam_m) * s_virgil
        mpol_obj = 2 * n * (m["nll"] / (2 * n) + lam_m * m["entropy"])
        record(f"penalty_fraction_lam_m_{lam_m:g}", va.lam_r(n, lam_m) * s_virgil / obj)
        record(f"rel_error_objective_lam_m_{lam_m:g}", abs(obj - mpol_obj) / abs(mpol_obj))
        assert abs(obj - mpol_obj) < RTOL_R1 * abs(chi2_part + va.lam_r(n, lam_m) * m["entropy"])  # model-dependent part of the objective
    # the penalty exceeds |nll| at lam_m = 1e-5, so the objective test constrains S_M to about 1e-5
    assert 0.1 < abs(va.lam_r(n, 1e-5) * s_virgil / sc["nll"]) < 100  # -ln L is negative (constants), so compare magnitudes


@pytest.mark.validates("virgil_bridge.alma", roots=["mpol"], kind="control")
@pytest.mark.skipif(not sp.available("mpol", "mpol"), reason="MPoL not installed (scripts/setup_external.sh mpol)")
@pytest.mark.skipif(tm._example_commit() != tm.EXAMPLES_COMMIT, reason="MPoL-dev/examples not cloned at the pinned commit")
def test_r1_control_virgils_maxentropy_or_the_untapered_image_does_not_match(tmp_path):
    """The R1 sum with virgil's `MaxEntropy` (unit-sum image, flat prior) in place of S_M, with the same strength, or with S_M
    of the untapered image, misses MPoL's objective by far more than R1's bound."""
    from virgil import Image
    from virgil.imaging import MaxEntropy

    npix, pts, source, theta, sky, _ = _r1_case(LAM_M)
    m = _mpol_side(npix, pts, theta, 1e-5, tmp_path)
    data, prob, values, sc, _ = _virgil_side(npix, pts, theta, 1e-5)
    n, lam = m["n"], va.lam_r(m["n"], 1e-5)
    mpol_obj = m["nll"] + lam * m["entropy"]
    wrong_maxent = sc["nll"] + lam * float(MaxEntropy(1.0).value(Image(values["log_brightness"], pixel_scale_mas=PIX)))
    wrong_bare = sc["nll"] + lam * float(va.mpol_entropy_jax(values["log_brightness"], values["cal.flux"], pixel_mas=PIX, taper_mas=0.0, prior=1e-4, f_tot=0.253))
    kw = dict(pixel_mas=PIX, taper_mas=30.0, prior=1e-4, f_tot=0.253)
    flux = values["cal.flux"]
    lb = values["log_brightness"]
    wrong_p = sc["nll"] + lam * float(va.mpol_entropy_jax(lb, flux, **{**kw, "prior": 1e-3}))  # near misses: units right
    wrong_ftot = sc["nll"] + lam * float(va.mpol_entropy_jax(lb, flux, **{**kw, "f_tot": 0.5 * 0.253}))
    scale = abs(m["nll"] - m["const"] + lam * m["entropy"])
    for name, wrong in (("maxentropy", wrong_maxent), ("untapered", wrong_bare), ("prior_1e-3", wrong_p), ("f_tot_halved", wrong_ftot)):
        miss = abs(wrong - mpol_obj) / scale
        record(f"rel_miss_{name}", miss)
        assert miss > 10 * RTOL_R1, name  # maxentropy misses by units alone; the others are the wrong prior, taper or normalisation


# ------------------------------------------------------------------ R2


@pytest.mark.validates("virgil_bridge.alma", roots=["mathematics"])
def test_r2_fit_r_falls_and_writes_its_files(syn, r_fit):
    """R2. ``fit --config R`` for 20 steps per stage at 128^2: the objective falls from the Gaussian start, the trajectory,
    stop reason, convergence verdict, per-ddid chi^2/N (s = 1 first), image, FITS and one residual dirty image per observation exist."""
    res = json.loads((r_fit / "result.json").read_text())
    data = va.load_dir(syn)
    n = len(data.u)
    prob = va.build_problem(data, "R", lam=va.lam_r(n), npix=128)
    start = va.start_values(prob)
    start_obj = va.r_objective(data, {k: np.asarray(v) for k, v in start.items()}, lam=va.lam_r(n), npix=128)["objective"]
    record("objective_start", start_obj)
    record("objective_final", res["objective_R"])
    assert res["objective_R"] < start_obj
    assert [s["name"] for s in res["stages"]] == ["sb", "all"]
    traj = res["stages"][-1]["trajectory"]
    assert traj and np.isfinite(traj[-1]["loss"]) and traj[-1]["steps"] <= 20
    assert isinstance(res["converged"], bool)
    assert len(res["chi2_s1"]["chi2_raw_over_n"]) == 14 and len(res["chi2_fitted_s"]["chi2_fit_over_n"]) == 14
    for f in ["values.npz", "image.npy", "image_tapered.npy", "image.fits", "result.json"] + [f"resid_dirty_obs{e}.npy" for e in range(7)]:
        assert (r_fit / f).exists(), f
    img = np.load(r_fit / "image.npy")
    assert img.shape == (128, 128) and abs(img.sum() * (PIX / 1000) ** 2 - res["flux_jy"]) < 1e-5 * res["flux_jy"]
    d, w, _ = cx.read_fits(r_fit / "image.fits")
    assert np.allclose(d[::-1], img, rtol=1e-6)


@pytest.mark.validates("virgil_bridge.alma", "virgil.fitting.fit", roots=["mathematics"])
def test_the_loss_virgil_fit_minimises_contains_lambda_s_m(syn):
    """C3 in the fit itself. `fit`'s final loss minus r_objective (-ln L + lam S_M, outside `fit`) at the same values is a constant
    (the priors' normalisation), the same at lam = 0 and lam = 1; so `fit` adds lam S_M, with the plus sign. Control: a problem
    without the MPoLEntropy term has a different constant, by lam S_M, so the comparison does fail when the term is missing."""
    data = va.load_dir(syn)
    consts = {}
    for name, lam, drop in (("zero", 0.0, False), ("one", 1.0, False), ("without", 1.0, True)):
        prob = va.build_problem(data, "R", lam=lam, npix=64)
        if drop:
            prob.terms = prob.terms[:1]
        values, info, _ = va.run_fit(prob, va.start_values(prob), max_steps=15)
        obj = va.r_objective(data, values, lam=lam, npix=64)
        consts[name] = (info["loss"] - obj["objective"], obj["penalty"])
    record("const_lam0", consts["zero"][0])
    record("const_lam1", consts["one"][0])
    record("const_without_term", consts["without"][0])
    pen = abs(consts["one"][1])
    assert pen > 10  # the term matters here
    assert abs(consts["one"][0] - consts["zero"][0]) < 1e-3 * pen
    assert abs(consts["without"][0] - consts["zero"][0]) > 0.5 * pen


@pytest.mark.validates("virgil_bridge.alma", "virgil.radio.ComplexVisData.term", roots=["mathematics"])
def test_score_equals_the_terms_negative_log_likelihood(syn, r_fit):
    """`score` (per-point decomposition, eps 1e-12) equals 1/2 sum r^2 + log_norm + sum (ln 2 pi - ln w) of the fit's own term,
    to 1e-12 relative, with and without a mask, and its per-ddid and per-annulus tallies add up."""
    data = va.load_dir(syn)
    values = {k: np.asarray(v) for k, v in np.load(r_fit / "values.npz").items()}
    prob = va.build_problem(data, "R", lam=1.0, npix=128)
    rng = np.random.default_rng(0)
    mask = rng.uniform(size=len(data.u)) < 0.7
    annulus = rng.integers(0, 12, len(data.u))
    for m in (None, mask):
        sc = va.score(data, prob.scene, values, prob.nuis, mask=m, annulus=annulus)
        term = data.term(prob.scene, prob.nuis, mask=m)
        jv = {k: jnp.asarray(v) for k, v in values.items()}
        with va.ft_eps(1e-12):
            r = term(jv)
            keep = np.ones(len(data.u), bool) if m is None else m
            ref = 0.5 * float(jnp.sum(r**2)) + float(term.log_norm(jv)) + float(np.sum(va.LN2PI - np.log(np.asarray(data.weight))[keep]))
        record("rel_error_score_vs_term", abs(sc["nll"] - ref) / abs(ref))
        assert abs(sc["nll"] - ref) < 1e-10 * abs(ref)
        assert abs(sum(sc["by_ddid"]["nll"]) - sc["nll"]) < 1e-9 * abs(sc["nll"])
        assert abs(sum(sc["by_annulus"]["nll"]) - sc["nll"]) < 1e-9 * abs(sc["nll"])
        assert sum(sc["by_ddid"]["n"]) == sc["n"] == (len(data.u) if m is None else int(m.sum()))


@pytest.mark.validates("virgil_bridge.alma", roots=["mathematics"])
def test_convergence_rule_and_acceptance():
    """C8 on a synthetic trajectory: a loss that changed by 0.05 over 200 steps has converged, one that changed by 5 has not;
    C13's acceptance rule on s / ŝ."""
    flat = [dict(steps=200, loss=100.0), dict(steps=400, loss=99.0), dict(steps=600, loss=98.96)]
    steep = [dict(steps=200, loss=100.0), dict(steps=400, loss=90.0), dict(steps=600, loss=85.0)]
    assert va.convergence(flat)["converged"] and not va.convergence(steep)["converged"]
    shat = np.full(26, 1.1)
    ok = script.acceptance(shat * np.r_[np.full(18, 1.02), np.full(8, 1.0)], shat)
    bad = script.acceptance(shat * np.r_[np.full(18, 1.05), np.full(8, 1.02)], shat)
    assert ok["sb_ok"] and ok["lb_ok"] and not bad["sb_ok"] and not bad["lb_ok"]


# ------------------------------------------------------------------ R3


@pytest.mark.validates("virgil_bridge.alma", "virgil.radio.dartboard_folds", roots=["mathematics", "statistics"])
def test_r3_folds_are_disjoint_covering_and_hermitian_closed(syn, tmp_path):
    """R3 (the folds). The saved folds assign every point once, train (fold != j) and held-out (fold == j) are disjoint and
    cover all points, the file's checksum reloads, and a point and its Hermitian partner (-u, -v, conj V) share a fold and an annulus."""
    f = tmp_path / "folds.npz"
    _call("folds", "--data", syn, "--out", f, "--k", 4)
    data = va.load_dir(syn)
    fold, annulus, meta = va.load_folds(f, data)
    assert meta["k"] == 4 and meta["n"] == len(data.u) and len(meta["sha256"]) == 64
    counts = np.bincount(fold, minlength=4)
    assert np.all(counts > 0.5 * len(fold) / 4) and np.all(counts < 1.5 * len(fold) / 4)  # no empty or lopsided fold
    for j in range(4):
        train, test = fold != j, fold == j
        assert not np.any(train & test) and np.all(train | test)
    # the file is tied to the points' order: a ddid with the same points permuted, or a subset of ddids, is checked too
    sub, _, _ = va.load_folds(f, va.load_dir(syn, ddids=[0, 1]))
    assert len(sub) == 2 * 400
    shuffled = tmp_path / "shuf"
    shuffled.mkdir()
    z = dict(np.load(syn / "ddid_00.npz"))
    order = np.random.default_rng(0).permutation(len(z["u"]))
    np.savez(shuffled / "ddid_00.npz", **{k: (a[order] if np.ndim(a) else a) for k, a in z.items()})
    with pytest.raises(ValueError, match="differ"):
        va.load_folds(f, va.load_dir(shuffled))
    from virgil.radio import ComplexVisData

    u, v = np.asarray(data.u), np.asarray(data.v)
    both = ComplexVisData.from_arrays(np.r_[u, -u], np.r_[v, -v], np.r_[np.asarray(data.vis), np.conj(np.asarray(data.vis))], np.r_[np.asarray(data.weight)] .repeat(1).tolist() * 2,
                                      np.r_[np.asarray(data.freq), np.asarray(data.freq)], ddid=np.r_[np.asarray(data.ddid_ids)[np.asarray(data.ddid)]] .tolist() * 2,
                                      obs=np.asarray(data.obs_ids)[np.asarray(data.obs)].tolist() * 2)
    f2, a2 = va.make_folds(both, 4)
    n = len(u)
    assert np.array_equal(f2[:n], f2[n:]) and np.array_equal(a2[:n], a2[n:])


@pytest.mark.validates("virgil_bridge.alma", roots=["mathematics"])
def test_r3_cv_task_and_aggregate_on_two_lambdas_and_two_folds(syn, tmp_path):
    """R3 (the sweep). F at 64^2: a joint fit, then ``cv-task`` for 2 lambda x 2 folds from it, then ``cv-aggregate``: the
    held-out and training counts add to N, the held-out scores of the folds are each finite and the summary, plots and
    markdown exist with the right grid."""
    folds = tmp_path / "folds.npz"
    _call("folds", "--data", syn, "--out", folds, "--k", 2)
    init = tmp_path / "init"
    _call("fit", "--config", "F", "--data", syn, "--out", init, "--npix", 64, "--lam", 1e3, "--max-steps", 10, "--no-dirty")
    cv = tmp_path / "cv"
    n = 14 * 400
    data = va.load_dir(syn)
    fold, _, _ = va.load_folds(folds, data)
    ids = np.asarray(data.ddid_ids)[np.asarray(data.ddid)]
    for lam in (1e2, 1e4):
        for j in (0, 1):
            _call("cv-task", "--data", syn, "--folds", folds, "--fold", j, "--init", init, "--lam", lam, "--npix", 64, "--max-steps", 10,
                  "--out", cv / f"lam_{lam:g}" / f"fold_{j}")
            r = json.loads((cv / f"lam_{lam:g}" / f"fold_{j}" / "cv.json").read_text())
            held = fold == j  # the folds file read independently of cv-task
            assert r["n_test"] == int(held.sum()) and r["n_train"] == int((~held).sum())
            assert r["heldout"]["by_ddid"]["n"] == [int(held[ids == d].sum()) for d in np.unique(ids)]
            assert abs(r["lam_train"] - lam * r["n_train"] / n) < 1e-12 * lam  # lam is the full-data strength, rescaled to the training size
            assert np.isfinite(r["heldout"]["nll"]) and sum(r["heldout"]["by_annulus"]["n"]) == r["n_test"]
    _call("cv-aggregate", cv)
    s = json.loads((cv / "cv_summary.json").read_text())
    assert s["lams"] == [1e2, 1e4] and s["folds"] == [0, 1] and s["lam_star"] in (1e2, 1e4) and s["at_edge"]
    assert len(s["folds_sha256"]) == 1
    for f in ("cv_curve.png", "cv_annulus.png", "cv.md"):
        assert (cv / f).exists()
    # the two folds partition the data, so each lambda's summed held-out points are all N
    tot = sum(json.loads((cv / "lam_100" / f"fold_{j}" / "cv.json").read_text())["n_test"] for j in (0, 1))
    assert tot == n


# ------------------------------------------------------------------ R4


def mock_mpol_export(fit_dir, out, *, scale_amp=1.0, tap=True):
    """MPoL's ``export_checkpoint`` format (W8: ``export.npz`` with ``image`` (tapered), ``base_image``, ``cell_size_arcsec``,
    ``amp``, ``sigma`` (s_M, fixed observation included) and ``delta_{ra,dec}_arcsec``) made from one of our R fits: its
    solution, as MPoL would store it (s_M = s / g)."""
    z = {k: np.asarray(v) for k, v in np.load(fit_dir / "values.npz").items()}
    npix = z["log_brightness"].shape[0]
    base = va.image_of(z, npix, PIX)
    amp = np.concatenate([[1.0], z["cal.gain"] * scale_amp])  # the fixed observation stays at 1
    obs_of = np.repeat(np.arange(7), 2)
    off = np.concatenate([np.zeros((1, 2)), z["cal.offset"]]) / 1000.0
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "export.npz", image=va.tapered_image(base, PIX) if tap else base, base_image=base, cell_size_arcsec=PIX / 1000, amp=amp,
             sigma=z["cal.sigma"] / amp[obs_of], delta_ra_arcsec=off[:, 0], delta_dec_arcsec=off[:, 1])
    (out / "export.json").write_text(json.dumps(dict(npix=npix)))
    return out


@pytest.mark.validates("virgil_bridge.alma", roots=["mathematics"])
def test_r4_compare_self_comparison_gives_zero_differences(syn, r_fit, tmp_path):
    """R4. ``compare`` of a fit with an MPoL export made from the same solution (mocked in W8's format: the 1028^2 grid is
    replaced by the fit's own): s = g s_M, gains, offsets, flux and the three image differences are zero and the two R objectives are equal
    (virgil at eps 1e-12 on both). With the export's gains 1% off, the gain difference is nonzero and the objective at MPoL's
    solution changes. This is plumbing: the mock shares the mapping's conventions, so it is not evidence about MPoL's export (W8's real one is)."""
    mp = mock_mpol_export(r_fit, tmp_path / "mpol")
    out = tmp_path / "cmp"
    _call("compare", "--virgil", r_fit, "--mpol", mp, "--data", syn, "--out", out)
    c = json.loads((out / "compare.json").read_text())
    assert c["sigma"]["max_abs_dev"] < 1e-12 and c["gain"]["max_abs_diff"] < 1e-12 and c["offset_mas"]["max_abs_diff"] < 1e-12
    assert abs(c["flux_jy"]["diff"]) < 1e-12
    for k in ("untapered", "tapered_30mas"):
        assert c["image"][k]["rms_over_peak"] < 1e-6 and abs(c["image"][k]["flux_ratio"] - 1) < 1e-6
    o = c["objective"]
    assert abs(o["ours"]["objective"] - o["mpol_solution_in_virgil"]["objective"]) < 1e-9 * abs(o["ours"]["objective"])
    assert (out / "compare.md").exists()
    mp2 = mock_mpol_export(r_fit, tmp_path / "mpol2", scale_amp=1.01)
    _call("compare", "--virgil", r_fit, "--mpol", mp2, "--data", syn, "--out", tmp_path / "cmp2")
    c2 = json.loads((tmp_path / "cmp2" / "compare.json").read_text())
    assert c2["gain"]["max_abs_diff"] > 1e-3  # the amplitude is a gauge for s = g s_M, so only the gain moves
    assert abs(c2["objective"]["mpol_solution_in_virgil"]["objective"] - c["objective"]["mpol_solution_in_virgil"]["objective"]) > 1e-6 * abs(c["objective"]["ours"]["objective"])


@pytest.mark.validates("virgil_bridge.alma", "crosscheck.alma_maxent", roots=["mathematics"])
def test_r4_f_against_dsharp_runs_through_compare_and_a_shifted_grid_fails(syn, r_fit, tmp_path):
    """The DSHARP step of ``compare``, run as the command: the fit's image (here the R fit's, standing in for F's) convolved to a CLEAN
    beam and written as a 4-axis restored FITS with a hand-written header (East-left CDELT1, row 0 South, the phase centre in cell
    (N/2 - 1, N/2 - 1), so no repository WCS helper builds the header) gives ``F_vs_dsharp`` zero differences; the same FITS with its
    reference pixel shifted by 3 pixels, which is what an off-by-pixel WCS would do, does not."""
    from astropy.io import fits

    img = np.load(r_fit / "image.npy").astype(float)
    n = img.shape[0]
    beam = (120.0, 80.0, 30.0)
    conv = cx.convolve(img, PIX, *beam) * cx.beam_area_arcsec2(*beam[:2])
    mp = mock_mpol_export(r_fit, tmp_path / "mpol")

    def run(name, shift):
        hdr = fits.Header()
        hdr.update(CTYPE1="RA---SIN", CTYPE2="DEC--SIN", CRVAL1=script.RA_DEG, CRVAL2=script.DEC_DEG, CDELT1=-PIX / 3.6e6, CDELT2=PIX / 3.6e6,
                   CRPIX1=n // 2 - 1 + 1 + shift, CRPIX2=n // 2 - 1 + 1, CUNIT1="deg", CUNIT2="deg", BMAJ=beam[0] / 3.6e6, BMIN=beam[1] / 3.6e6, BPA=beam[2], BUNIT="Jy/beam")
        # FITS row 0 is South, the phase centre (repository row N/2 - 1 from the North) is FITS row n - 1 - (N/2 - 1) = N/2: CRPIX2 = N/2 + 1
        hdr["CRPIX2"] = n - 1 - (n // 2 - 1) + 1
        fits.PrimaryHDU(conv[::-1][None, None].astype(np.float32), hdr).writeto(tmp_path / f"{name}.fits")
        _call("compare", "--virgil", r_fit, "--mpol", mp, "--data", syn, "--f-dir", r_fit, "--dsharp", tmp_path / f"{name}.fits", "--out", tmp_path / name)
        return json.loads((tmp_path / name / "compare.json").read_text())["F_vs_dsharp"]

    good, bad = run("good", 0), run("shifted", 3)
    record("rms_over_peak_matched", good["rms_over_peak"])
    record("rms_over_peak_shifted_3_pixels", bad["rms_over_peak"])
    assert good["rms_over_peak"] < 1e-5 and abs(good["flux_ratio"] - 1) < 1e-5
    assert bad["rms_over_peak"] > 100 * good["rms_over_peak"]
