"""``virgil.radio`` against the independent ALMA side of this repository.

Everything here needs ``virgil.radio`` (virgil from 9ee600f, the ``radio``
extra); CI's pinned virgil has none, so the module skips there until
``VIRGIL_REF`` moves. The reference code is ``crosscheck.sky`` and
``crosscheck.alma`` (numpy and scipy only) and the CASA worker of
``external_bridge.alma_worker`` in ``.venv-alma``.

* reader agreement (tier B, root casa): ``ComplexVisData.read_ms``, with the
  casatools of virgil's own venv, against our reader on CASA-simulated
  measurement sets and on a stride-1000 read of IM Lup;
* the model, the primary beam, the likelihood normalisation and the folds
  (tier A, mathematics and statistics);
* the error-scale fit on CASA's noise, and an injection-recovery on IM Lup's
  uv coverage (tier B).
"""

import pathlib
import re

import numpy as np
import pytest

radio = pytest.importorskip("virgil.radio")

import virgil_bridge.alma as vba  # noqa: E402  (sets CASASITECONFIG before casatools loads)
from crosscheck import alma, sky  # noqa: E402
from evidence.plugin import record  # noqa: E402
from external_bridge import _subprocess as sp  # noqa: E402

C = 299792458.0
MAS = np.pi / 180 / 3600 / 1000
SOURCES = [(0.010, 300.0, 100.0), (0.005, -200.0, -400.0)]  # Jy, East mas, North mas (the CASA simulation)
IMLUP_MS = pathlib.Path("~/data/alma_imlup/ms/IMLup_continuum.ms").expanduser()
SUBSAMPLE = pathlib.Path("~/data/alma_imlup/subsample_1000").expanduser()


def _have_casa(name, module="casatools"):
    return sp.available(name, module)


def _worker(task):
    return sp.run("alma", "alma_worker.py", task)


def _load(files):
    return [dict(np.load(f["file"])) for f in files]


def _virgil_casatools():
    try:
        import casatools
    except Exception:  # noqa: BLE001  (casatools raises its own config errors)
        return None
    return casatools.version_string()


def _version_code(v):
    """6.7.6.14 (or 6.7.6-14) as 6070614.0: major*1e6 + minor*1e4 + patch*1e2 + build, a float for `record`."""
    major, minor, patch, build = (int(x) for x in re.split(r"[.-]", str(v))[:4])
    return float(major * 10**6 + minor * 10**4 + patch * 10**2 + build)


# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def ms_clean(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("radio_clean")
    ms = str(tmp / "clean.ms")
    _worker({"task": "simulate_ms", "path": ms})
    return ms, tmp


@pytest.fixture(scope="module")
def ms_noisy(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("radio_noisy")
    ms = str(tmp / "noisy.ms")
    _worker({"task": "simulate_ms", "path": ms, "noise_jy": 1e-3})
    return ms, tmp


# ------------------------------------------------- 1. reader agreement (B)


def _by_key(z):
    """Order of a sample set by (ddid-local) antenna pair, time, channel."""
    return np.lexsort((z["chan"], z["time"], z["ant2"], z["ant1"]))


def _compare_readers(ms, stride, tmp):
    """Compare virgil's read_ms with ours, ddid by ddid; return the worst numbers."""
    data = radio.ComplexVisData.read_ms(ms, stride=stride)
    res = _worker({"task": "read", "path": ms, "out": str(tmp / f"ours{stride}"), "stride": stride})
    ours = _load(res["files"])
    ddids = [int(z["ddid"]) for z in ours]
    obsids = [int(z["obsid"]) for z in ours]
    # labels: one label per (DATA_DESC_ID, observation) pair, in sorted order; our files are one per DATA_DESC_ID
    assert list(data.ddid_ids) == ddids and list(data.obs_ids) == sorted(set(obsids))
    assert [data.obs_ids[o] for o in data.ddid_obs] == obsids
    worst = dict(uv=0.0, vis=0.0, weight=0.0, freq=0.0)
    n_total = 0
    for lab, z in enumerate(ours):
        sel = np.asarray(data.ddid) == lab
        n = int(sel.sum())
        assert n == len(z["u"]), (lab, n, len(z["u"]))  # point counts per ddid
        n_total += n
        got = {k: np.asarray(getattr(data, k))[sel] for k in ("u", "v", "weight", "freq", "time", "ant1", "ant2", "chan", "vis")}
        o_virgil, o_ours = _by_key(got), _by_key(z)
        for k in ("time", "ant1", "ant2", "chan"):
            assert np.array_equal(got[k][o_virgil], z[k][o_ours]), k
        for k in ("u", "v"):
            ref = z[k][o_ours]
            worst["uv"] = np.maximum(worst["uv"], np.abs(got[k][o_virgil] - ref).max() / np.abs(ref).max())
        gv = got["vis"][o_virgil]
        ov = (z["re"] + 1j * z["im"])[o_ours].astype(complex)
        worst["vis"] = np.maximum(worst["vis"], np.abs(gv - ov).max() / np.abs(ov).max())
        worst["weight"] = np.maximum(worst["weight"], np.abs(got["weight"][o_virgil] / z["weight"][o_ours] - 1).max())
        worst["freq"] = np.maximum(worst["freq"], np.abs(got["freq"][o_virgil] / z["freq"][o_ours] - 1).max())
    assert n_total == len(data)
    return worst, n_total, res["casatools"]


def _check_reader(worst):
    assert worst["uv"] < 1e-9
    assert worst["vis"] < 1.2e-7  # float32 rounding of the stored V (6e-8 relative to |V|max)
    assert worst["weight"] < 1.2e-7 and worst["freq"] < 1e-12


def _record_reader(worst, n, ours_ct):
    ver = _virgil_casatools()
    record("casatools_virgil_version_code", _version_code(ver))
    record("casatools_alma_worker_version_code", _version_code(ours_ct))
    for k, v in worst.items():
        record(f"max_rel_{k}", v)
    record("n_points", n)


@pytest.mark.validates("virgil.radio.ComplexVisData.read_ms", roots=["casa"], tier="B")
@pytest.mark.skipif(not _have_casa("alma") or _virgil_casatools() is None, reason="casatools missing here or in .venv-alma")
@pytest.mark.parametrize("which", ["clean", "noisy"])
def test_read_ms_agrees_with_our_reader_on_the_synthetic_ms(which, ms_clean, ms_noisy):
    """virgil's `read_ms` (casatools 6.7.6.14, its own venv) and our reader (casatools
    6.7.0.31, `.venv-alma`) give the same points on the CASA-simulated MS:
    counts per ddid, u and v to 1e-9, V and weight to float32 precision,
    frequency, times, antennas, channels, and the (ddid, observation) labels."""
    ms, tmp = ms_clean if which == "clean" else ms_noisy
    worst, n, ours_ct = _compare_readers(ms, 1, tmp)
    _record_reader(worst, n, ours_ct)
    _check_reader(worst)


@pytest.mark.validates("virgil.radio.ComplexVisData.read_ms", roots=["casa"], tier="B")
@pytest.mark.skipif(
    not _have_casa("alma") or _virgil_casatools() is None or not IMLUP_MS.exists(), reason="casatools or the IM Lup MS missing"
)
def test_read_ms_agrees_with_our_reader_on_im_lup_at_stride_1000(tmp_path):
    """The same on the 26 ddids and 7 observations of DSHARP IM Lup, stride 1000
    (42,424 points); `DATA_DESC_ID` k is label k, and each is in one observation."""
    worst, n, ours_ct = _compare_readers(str(IMLUP_MS), 1000, tmp_path)
    _record_reader(worst, n, ours_ct)
    assert n == 42424
    _check_reader(worst)


def casa_point_sources(data):
    """CASA's predicted V: the two simulated point sources, in the documented convention
    V = sum F exp(-2 pi i (u x + v y)), x East, y North (u, v in wavelengths)."""
    u, v = np.asarray(data.u), np.asarray(data.v)
    return sum(f * np.exp(-2j * np.pi * MAS * (u * e + v * n)) for f, e, n in SOURCES)


@pytest.mark.validates("virgil.radio.ComplexVisData.read_ms", roots=["casa"], tier="B", property="conjugation")
@pytest.mark.skipif(not _have_casa("alma") or _virgil_casatools() is None, reason="casatools missing here or in .venv-alma")
def test_read_ms_returns_the_conjugate_of_data_that_casa_predicted_for_known_point_sources(ms_clean):
    """virgil's `read_ms` of CASA's own prediction (10 mJy at (+300, +100) mas, 5 mJy at
    (-200, -400) mas) equals sum F exp(-2 pi i (u x + v y)) to float32 precision: DATA is the
    conjugate of V at u = +UVW[0] nu / c, as docs/alma_conventions.md found independently.
    For a real sky conj V(u, v) = V(-u, -v), so the visibilities alone fix only the product of the
    conjugation and the sign of u: the test also checks u and v against our reader (UVW read
    from the MS by the CASA worker, u = +UVW[0] nu / c), which makes it stand on its own."""
    ms, tmp = ms_clean
    data = radio.ComplexVisData.read_ms(ms)
    ours = _load(_worker({"task": "read", "path": ms, "out": str(tmp / "ours_conj")})["files"])
    for lab, z in enumerate(ours):
        sel = np.asarray(data.ddid) == lab
        a, b = _by_key({k: np.asarray(getattr(data, k))[sel] for k in ("time", "ant1", "ant2", "chan")}), _by_key(z)
        for k in ("u", "v"):
            ref = z[k][b]
            assert np.abs(np.asarray(getattr(data, k))[sel][a] - ref).max() < 1e-9 * np.abs(ref).max()
    miss = np.abs(np.asarray(data.vis) - casa_point_sources(data)).max()
    record("max_abs_vis_error_jy", miss)
    # measured 6.7e-8 Jy (4e-6 of the 15 mJy): CASA's single-precision predict, not float32 storage
    # (1e-9); the wrong mappings miss by 0.03, so 1e-6 leaves a margin of 15 on the measurement
    assert miss < 1e-6


@pytest.mark.validates("virgil.radio.ComplexVisData.read_ms", roots=["casa"], tier="B", kind="control", property="conjugation")
@pytest.mark.skipif(not _have_casa("alma") or _virgil_casatools() is None, reason="casatools missing here or in .venv-alma")
def test_control_reading_data_without_the_conjugation_misses_the_prediction(ms_clean):
    """The wrong mapping (`conjugate=False`, DATA taken as V) misses CASA's prediction by about
    twice the visibility amplitude, so the test above does fix the conjugation."""
    ms, _ = ms_clean
    data = radio.ComplexVisData.read_ms(ms, conjugate=False)
    miss = np.abs(np.asarray(data.vis) - casa_point_sources(data)).max()
    record("max_abs_vis_miss_jy", miss)
    assert miss > 0.02


# ------------------------------------------------ numpy-made uv and model


PIXEL = 5.0  # mas, for the 64 x 64 test image (Nyquist q = 20 Mlambda)
DDIDS = [(0, 230e9), (0, 245e9), (1, 230e9), (2, 245e9)]  # (observation, frequency): ddid = index
POINT = (0.3, 17.0, -23.0)  # flux fraction, East, North (mas): the image has the remaining 0.7


def make_uv(n=5000, seed=3):
    """About 5000 points, |q| log-uniform in 0.05-10 Mlambda, random angles, four
    ddids in three observations at two frequencies, made in numpy."""
    rng = np.random.default_rng(seed)
    q = 10 ** rng.uniform(np.log10(0.05e6), np.log10(10e6), n)
    phi = rng.uniform(0, 2 * np.pi, n)
    ddid = rng.integers(0, 4, n)
    return dict(
        u=q * np.cos(phi), v=q * np.sin(phi), ddid=ddid,
        obs=np.array([d[0] for d in DDIDS])[ddid], freq=np.array([d[1] for d in DDIDS])[ddid],
        weight=10 ** rng.uniform(2.5, 4.0, n),
    )


def random_brightness(seed=5, npix=64):
    """A positive random image, normalised to unit sum (numpy), and its log."""
    eta = np.random.default_rng(seed).normal(0.0, 1.0, (npix, npix))
    b = np.exp(eta - eta.max())
    return eta, b / b.sum()


def ref_visibilities(z, values, brightness, pixel, point=POINT, nu0=None, flip=1.0, taper_fwhm=None):
    """Independent M_k = F g_e (nu/nu0)^alpha shift(offset_e) T V_scene, numpy only.

    V_scene is 0.7 of the image (``crosscheck.sky`` pixel DFT, float64) plus
    the point source; the shift, taper and sums are written here from the
    documented formula."""
    u, v, nu = z["u"], z["v"], z["freq"]
    cloud = sky.pixel_image(brightness, pixel)
    scene = (1 - point[0]) * sky.visibility(cloud, u, v, 1.0) + point[0] * sky.vis_point(u, v, 1.0, point[1], point[2])
    obs = z["obs"]
    gain = np.concatenate([[1.0], np.atleast_1d(values["cal.gain"])])[obs]
    off = np.vstack([[0.0, 0.0], np.atleast_2d(values["cal.offset"])])[obs]
    nu0 = np.sum(z["weight"] * nu) / np.sum(z["weight"]) if nu0 is None else nu0
    out = values["cal.flux"] * gain * (nu / nu0) ** values["cal.alpha"]
    out = out * sky.shift_phase(u, v, 1.0, flip * off[:, 0], flip * off[:, 1]) * scene
    if taper_fwhm is not None:
        out = out * sky.gaussian_blur_factor(u, v, 1.0, taper_fwhm).real
    return out


def virgil_data(z, vis=None):
    return radio.ComplexVisData.from_arrays(
        z["u"], z["v"], np.zeros(len(z["u"]), complex) if vis is None else vis, z["weight"], z["freq"],
        ddid=z["ddid"], obs=z["obs"],
    )


def virgil_scene(eta, point=POINT):
    from virgil import Image, PointSource, System

    return System(img=Image(eta, pixel_scale_mas=PIXEL, flux=1 - point[0]), pt=PointSource(flux=point[0], dra=point[1], ddec=point[2]))


NUISANCE = {
    "cal.flux": 0.083, "cal.gain": np.array([1.07, 0.93]),
    "cal.offset": np.array([[3.1, -2.2], [-1.7, 4.4]]),  # mas
    "cal.sigma": np.array([1.2, 0.9, 1.7, 1.05]), "cal.alpha": 2.3,
}
TAPER = 20.0  # mas


class ft_defaults:
    """Set virgil's Fourier-transform defaults for a block and restore them."""

    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        from virgil import set_ft_defaults

        self.old = set_ft_defaults(**self.kw)

    def __exit__(self, *exc):
        from virgil import set_ft_defaults

        set_ft_defaults(**self.old)


@pytest.fixture(scope="module")
def problem():
    z = make_uv()
    eta, brightness = random_brightness()
    data = virgil_data(z)
    nu0 = float(np.sum(z["weight"] * z["freq"]) / np.sum(z["weight"]))
    return dict(z=z, eta=eta, b=brightness, data=data, nu0=nu0,
                nuis=radio.CalibrationNuisance(data, nu0_hz=nu0, taper_fwhm_mas=TAPER))


def _model(problem, nuis=None):
    return np.asarray(problem["data"].model(virgil_scene(problem["eta"]), NUISANCE, nuis or problem["nuis"]))


def _ref(problem, **kw):
    return ref_visibilities(problem["z"], NUISANCE, problem["b"], PIXEL, nu0=problem["nu0"], taper_fwhm=TAPER, **kw)


# ----------------------------------------------- 2. model against the DFT (A)


@pytest.mark.validates("virgil.radio.ComplexVisData.model", "virgil.radio.CalibrationNuisance", roots=["mathematics"])
def test_model_matches_the_float64_dft_of_the_pixel_image_with_every_nuisance_set(problem):
    """F, per-observation gains and offsets, a spectral index, the Gaussian taper and a
    point source on top of a random 64 x 64 image, 5000 uv points in four ddids and
    three observations: virgil's exact transform equals the numpy product to 1e-10."""
    with ft_defaults(ft="dft"):
        got = _model(problem)
    ref = _ref(problem)
    err = np.abs(got - ref).max() / np.abs(ref).max()
    pointwise = np.abs(got / ref - 1).max()
    record("max_rel_error_dft", err)
    record("max_pointwise_rel_error_dft", pointwise)
    assert np.abs(ref).min() > 0
    assert pointwise < 1e-10


@pytest.mark.validates("virgil.radio.ComplexVisData.model", "virgil.radio.CalibrationNuisance", roots=["mathematics"])
@pytest.mark.parametrize("eps", [1e-12, 1e-6])
def test_model_with_the_nufft_is_within_three_eps_of_the_dft(problem, eps):
    """The same product with ft='nufft' at eps_f64 = 1e-12 and 1e-6: the relative l2
    difference is at most 3 eps."""
    with ft_defaults(ft="nufft", eps_f64=eps):
        got = _model(problem)
    ref = _ref(problem)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    record("rel_l2_error", rel)
    record("eps", eps)
    assert rel <= 3 * eps


@pytest.mark.validates("virgil.radio.ComplexVisData.model", "virgil.radio.CalibrationNuisance", kind="control", roots=["mathematics"])
def test_control_the_offset_with_the_opposite_sign_does_not_match(problem):
    """The reference with the offsets' sign flipped misses by order one, so the
    1e-10 test above does pin the sign of the phase ramp (dra, ddec as a
    component's offsets: exp(-2 pi i (u dra + v ddec)))."""
    with ft_defaults(ft="dft"):
        got = _model(problem)
    wrong = _ref(problem, flip=-1.0)
    miss = np.abs(got / wrong - 1).max()
    record("max_pointwise_miss", miss)
    assert miss > 1e-2


@pytest.mark.validates("virgil.radio.CalibrationNuisance", roots=["mathematics"])
def test_default_reference_frequency_is_the_weight_averaged_frequency(problem):
    """Without `nu0_hz` the spectral index pivots at sum(w nu)/sum(w), as documented."""
    nuis = radio.CalibrationNuisance(problem["data"], taper_fwhm_mas=TAPER)
    with ft_defaults(ft="dft"):
        got = _model(problem, nuis)
    ref = _ref(problem)
    record("max_pointwise_rel_error", np.abs(got / ref - 1).max())
    assert np.abs(got / ref - 1).max() < 1e-10


# --------------------------------------------------- 3. primary beam (A)


def gaussian_beam_log(npix, pixel_mas, fwhm_mas):
    """ln of a Gaussian that is 1/2 at r = fwhm/2, on the pixel grid of
    `crosscheck.sky.pixel_image` (row 0 North, centre at (n-1)/2)."""
    k = ((npix - 1) / 2.0 - np.arange(npix)) * pixel_mas
    east, north = np.meshgrid(k, k)
    return -np.log(2.0) * (east**2 + north**2) / (fwhm_mas / 2.0) ** 2


# (name, frequency, dish diameter, beam factor): ALMA Band 6, 12 m, 1.13 (ALMA Technical Handbook); ATCA 16 cm, 22 m
BEAMS = [("alma", 230e9, 12.0, 1.13), ("atca", 2.1e9, 22.0, 1.13), ("atca-factor-1", 2.1e9, 22.0, 1.0)]


@pytest.mark.validates("virgil.radio.primary_beam", roots=["mathematics"])
@pytest.mark.parametrize("name,freq,dish,factor", BEAMS, ids=[b[0] for b in BEAMS])
@pytest.mark.parametrize("npix", [64, 65])
def test_primary_beam_is_the_gaussian_with_half_power_at_half_the_fwhm(name, freq, dish, factor, npix):
    """FWHM = factor c / (nu D); the log attenuation is -4 ln 2 r^2 / FWHM^2 on the pixel
    grid, and with 10 pixels to r = FWHM/2 the value there is ln(1/2)."""
    fwhm_mas = factor * C / (freq * dish) / MAS
    pixel = fwhm_mas / 20.0  # half power at ten pixels
    got = np.asarray(radio.primary_beam(npix, pixel, freq, dish, factor))
    ref = gaussian_beam_log(npix, pixel, fwhm_mas)
    record("max_abs_error", np.abs(got - ref).max())
    assert got.shape == (npix, npix)
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-12)
    if npix == 65:  # a pixel at the centre: ten pixels East of it is r = FWHM/2
        c = npix // 2
        assert got[c, c] == pytest.approx(0.0, abs=1e-12)
        assert got[c, c + 10] == pytest.approx(-np.log(2.0), rel=1e-12)
        assert got[c - 10, c] == pytest.approx(-np.log(2.0), rel=1e-12)


@pytest.mark.validates("virgil.radio.primary_beam", "virgil.radio.ComplexVisData.model", roots=["mathematics"])
def test_adding_the_beam_to_the_log_brightness_gives_the_visibilities_of_image_times_beam():
    """Image(eta + beam) is softmax(eta + beam) = image x beam / sum: its visibilities
    (flux 1, through `data.model`) equal the DFT of that product from `crosscheck.sky`."""
    npix, freq, dish, factor = 64, 230e9, 12.0, 1.13
    fwhm_mas = factor * C / (freq * dish) / MAS
    pixel = fwhm_mas / 24.0  # the beam FWHM is 24 pixels, so the product is visibly tapered
    eta, b = random_brightness(seed=9, npix=npix)
    rng = np.random.default_rng(4)
    n = 600
    q = 10 ** rng.uniform(3.0, np.log10(2e4 * 24 / 20), n)  # up to the Nyquist limit of this grid / 2
    phi = rng.uniform(0, 2 * np.pi, n)
    z = dict(u=q * np.cos(phi), v=q * np.sin(phi), ddid=np.zeros(n, int), obs=np.zeros(n, int), freq=np.full(n, freq), weight=np.full(n, 1e3))
    data = virgil_data(z)
    nuis = radio.CalibrationNuisance(data, gains=False, offsets=False, sigmas=False, alpha=False)
    from virgil import Image

    beam = radio.primary_beam(npix, pixel, freq, dish, factor)
    with ft_defaults(ft="dft"):
        got = np.asarray(data.model(Image(eta + beam, pixel_scale_mas=pixel), {"cal.flux": 1.0}, nuis))
    product = b * np.exp(gaussian_beam_log(npix, pixel, fwhm_mas))
    ref = sky.visibility(sky.pixel_image(product, pixel), z["u"], z["v"], 1.0)
    untapered = sky.visibility(sky.pixel_image(b, pixel), z["u"], z["v"], 1.0)
    record("max_abs_error", np.abs(got - ref).max())
    assert np.abs(got - ref).max() < 1e-12
    assert np.abs(got - untapered).max() > 1e-3  # the beam matters at this size


# --------------------------------------------- 4. likelihood normalisation (A)


def scipy_logpdf_sum(data_vis, model, weight, s, mask=None):
    """sum of norm.logpdf over Re and Im with sigma = s / sqrt(w)."""
    from scipy.stats import norm

    sig = s / np.sqrt(weight)
    m = np.ones(len(sig), bool) if mask is None else mask
    return sum(norm.logpdf(getattr(data_vis, part)[m], loc=getattr(model, part)[m], scale=sig[m]).sum() for part in ("real", "imag"))


def noisy_vis(problem, seed=11):
    z = problem["z"]
    ref = _ref(problem)
    rng = np.random.default_rng(seed)
    sig = NUISANCE["cal.sigma"][z["ddid"]] / np.sqrt(z["weight"])
    return ref + sig * (rng.normal(size=len(ref)) + 1j * rng.normal(size=len(ref)))


def _two_parameter_sets():
    a = dict(NUISANCE)
    b = dict(NUISANCE, **{"cal.sigma": np.array([1.6, 1.1, 0.8, 1.35]), "cal.flux": 0.079, "cal.alpha": 2.1,
                          "cal.gain": np.array([1.05, 0.95])})
    return [a, b]


@pytest.mark.validates("virgil.radio.ComplexVisData.term", roots=["statistics"])
@pytest.mark.parametrize("which", [0, 1])
def test_half_chi2_plus_log_norm_is_the_negative_scipy_log_density_up_to_a_constant(problem, which):
    """1/2 sum(r^2) + log_norm(values) = -sum norm.logpdf(Re, Im; sigma = s_d/sqrt(w)) minus
    the parameter-independent sum(1/2 ln 2 pi - 1/2 ln w) over Re and Im; and
    `term.loglike` is the scipy sum itself. Two parameter sets (different s_d, F, alpha, g)."""
    values = _two_parameter_sets()[which]
    z, vis = problem["z"], noisy_vis(problem)
    data = virgil_data(z, vis)
    nuis = radio.CalibrationNuisance(data, nu0_hz=problem["nu0"], taper_fwhm_mas=TAPER)
    term = data.term(lambda _: virgil_scene(problem["eta"]), nuis)
    with ft_defaults(ft="dft"):
        res = np.asarray(term(values))
        ln = float(term.log_norm(values))
        ll = float(term.loglike(values))
    ref_model = ref_visibilities(z, values, problem["b"], PIXEL, nu0=problem["nu0"], taper_fwhm=TAPER)
    s = values["cal.sigma"][z["ddid"]]
    logpdf = scipy_logpdf_sum(vis, ref_model, z["weight"], s)
    const = np.sum(2 * (0.5 * np.log(2 * np.pi) - 0.5 * np.log(z["weight"])))
    lhs = 0.5 * np.sum(res**2) + ln
    record("diff_half_chi2_plus_log_norm", lhs - (-logpdf - const))
    record("diff_loglike", ll - logpdf)
    assert lhs == pytest.approx(-logpdf - const, rel=1e-10)
    assert ll == pytest.approx(logpdf, rel=1e-10)
    assert ln == pytest.approx(2 * np.sum(np.log(s)), rel=1e-12)
    assert len(res) == 2 * len(vis)


@pytest.mark.validates("virgil.radio.ComplexVisData.term", roots=["statistics"])
def test_masked_points_drop_out_of_chi2_and_the_normalisation(problem):
    """With a 0/1 mask, the masked points change neither 1/2 chi^2 nor log_norm: they equal the
    scipy sums over the unmasked points, and corrupting the masked data changes nothing."""
    values = _two_parameter_sets()[1]
    z, vis = problem["z"], noisy_vis(problem)
    mask = np.random.default_rng(2).random(len(vis)) < 0.6
    ref_model = ref_visibilities(z, values, problem["b"], PIXEL, nu0=problem["nu0"], taper_fwhm=TAPER)
    s = values["cal.sigma"][z["ddid"]]
    logpdf = scipy_logpdf_sum(vis, ref_model, z["weight"], s, mask)
    const = np.sum(2 * (0.5 * np.log(2 * np.pi) - 0.5 * np.log(z["weight"][mask])))
    out = []
    for v in (vis, np.where(mask, vis, 1e3 + 1e3j)):  # the second corrupts the masked points
        data = virgil_data(z, v)
        nuis = radio.CalibrationNuisance(data, nu0_hz=problem["nu0"], taper_fwhm_mas=TAPER)
        term = data.term(lambda _: virgil_scene(problem["eta"]), nuis, mask=mask)
        with ft_defaults(ft="dft"):
            res = np.asarray(term(values))
            out.append((0.5 * float(np.sum(res**2)), float(term.log_norm(values)), float(term.loglike(values))))
    assert out[0] == pytest.approx(out[1], rel=1e-12)
    half_chi2, ln, ll = out[0]
    record("diff_half_chi2_plus_log_norm", half_chi2 + ln - (-logpdf - const))
    assert half_chi2 + ln == pytest.approx(-logpdf - const, rel=1e-10)
    assert ll == pytest.approx(logpdf, rel=1e-10)
    assert ln == pytest.approx(2 * np.sum(np.log(s[mask])), rel=1e-12)


# ---------------------------------------------- 5. cross-validation folds (A)


def hermitian_doubled(z):
    return {k: np.concatenate([v, -v if k in ("u", "v") else v]) for k, v in z.items()}


def our_annulus(q, n_q):
    edges = np.logspace(np.log10(q.min()), np.log10(q.max()), n_q + 1)
    return np.clip(np.digitize(q, edges) - 1, 0, n_q - 1)


@pytest.mark.validates("virgil.radio.dartboard_folds", roots=["mathematics"])
@pytest.mark.parametrize("k", [3, 5])
def test_dartboard_folds_keep_hermitian_partners_together_and_follow_log_q_bins(k):
    """A point and its partner at (-u, -v) get the same fold; the annulus is our log-spaced
    q bin (12 between the smallest and largest q); every fold is occupied; and every uv
    cell (annulus x sector of arctan2(v, u) mod pi) sits in one fold."""
    z = hermitian_doubled(make_uv(n=3000))
    n = len(z["u"]) // 2
    data = virgil_data(z)
    fold, ann = (np.asarray(a) for a in radio.dartboard_folds(data, k=k, n_q=12, n_phi=16, seed=0))
    assert np.array_equal(fold[:n], fold[n:]) and np.array_equal(ann[:n], ann[n:])
    q = np.hypot(z["u"], z["v"])
    mine = our_annulus(q, 12)
    record("annulus_mismatches", np.sum(mine != ann))
    assert np.array_equal(ann, mine)
    assert set(np.unique(fold)) == set(range(k))
    sizes = np.bincount(fold, minlength=k)
    record("smallest_fold", sizes.min())
    assert sizes.min() > 0
    sector = np.floor((np.arctan2(z["v"], z["u"]) % np.pi) / (np.pi / 16)).astype(int) % 16
    cell = mine * 16 + sector
    for c in np.unique(cell):
        assert len(np.unique(fold[cell == c])) == 1
    assert len(np.unique(cell)) > 3 * k  # many cells, so the split is a real dartboard
    assert np.array_equal(fold, np.asarray(radio.dartboard_folds(data, k=k, n_q=12, n_phi=16, seed=0)[0]))  # seeded


@pytest.mark.validates("virgil.radio.ComplexVisData.chi2_by_ddid", roots=["mathematics"])
def test_chi2_by_ddid_of_a_held_out_fold_matches_numpy(problem):
    """chi2_by_ddid(mask = fold == i): per ddid, sum w |V - M|^2 / s_d^2 over the held-out
    points and their count, against numpy with the independent model; with no mask, all points."""
    z, vis = problem["z"], noisy_vis(problem)
    data = virgil_data(z, vis)
    nuis = radio.CalibrationNuisance(data, nu0_hz=problem["nu0"], taper_fwhm_mas=TAPER)
    fold = np.asarray(radio.dartboard_folds(data, k=5)[0])
    ref = _ref(problem)
    s = NUISANCE["cal.sigma"][z["ddid"]]
    r2 = z["weight"] * np.abs(vis - ref) ** 2 / s**2  # Re and Im each 1/sigma^2 = w / s^2
    for mask in (fold == 2, None):
        m = np.ones(len(vis), bool) if mask is None else mask
        with ft_defaults(ft="dft"):
            chi2, cnt = data.chi2_by_ddid(virgil_scene(problem["eta"]), NUISANCE, nuis, mask=mask)
        want = np.array([r2[m & (z["ddid"] == d)].sum() for d in range(4)])
        want_n = np.array([(m & (z["ddid"] == d)).sum() for d in range(4)])
        record("max_rel_chi2_error", np.abs(np.asarray(chi2) / want - 1).max())
        assert np.asarray(chi2) == pytest.approx(want, rel=1e-10)
        assert np.array_equal(np.asarray(cnt).astype(int), want_n)


# ------------------------------------- 6. sigma factor on CASA's noise (B)


@pytest.mark.validates("virgil.radio.ComplexVisData.term", "virgil.radio.CalibrationNuisance", roots=["statistics", "casa"], tier="B")
@pytest.mark.skipif(not _have_casa("alma"), reason="CASA not installed (scripts/setup_external.sh alma)")
def test_sigma_fit_on_casa_noise_is_one_and_matches_the_successive_difference_estimate(ms_noisy):
    """On the noisy synthetic MS (1 mJy simplenoise on each of XX and YY, so w = 2e6 Jy^-2 is
    the right weight), with the sky held at CASA's two point sources, `virgil.fit` of
    `cal.sigma` (and the total flux) gives s_d = 1 within 3 standard errors, s/(2 sqrt N),
    and agrees with `crosscheck.alma.successive_difference_sigma` within the errors of both."""
    from virgil import fit

    ms, tmp = ms_noisy
    zs = _load(_worker({"task": "read", "path": ms, "out": str(tmp / "ours")})["files"])
    data = vba.load_npz(zs)
    nuis = radio.CalibrationNuisance(data, gains=False, offsets=False, sigmas=True, alpha=False)
    scene = vba.point_scene(SOURCES)
    term = data.term(scene, nuis)
    init = nuis.defaults()
    init["cal.flux"] = 0.0149
    result = fit(lambda **values: scene(values), nuis.priors(), data=(), likelihoods=[term], init=init, method="lbfgs")
    s = np.asarray(result.values["cal.sigma"])
    n = np.array([len(z["u"]) for z in zs])
    se = s / (2 * np.sqrt(n))
    assert len(s) == 2
    record("flux_jy", float(result.values["cal.flux"]))
    assert float(result.values["cal.flux"]) == pytest.approx(0.015, rel=0.01)
    for d, z in enumerate(zs):
        sd = alma.successive_difference_sigma(z)
        se_sd = sd["s2_err"] / (2 * sd["s"])
        record(f"s_fit_{d}", s[d])
        record(f"se_fit_{d}", se[d])
        record(f"s_successive_difference_{d}", sd["s"])
        assert abs(s[d] - 1.0) < 3 * se[d]
        assert abs(s[d] - sd["s"]) < 3 * np.hypot(se[d], se_sd)


# ------------------------------- 7. injection-recovery on IM Lup's uv (B)

IM_PIXEL, IM_NPIX = 25.0, 48
OFFSET_BOUND = (-20.0, 20.0)  # mas: the prior of the offset fits, `priors(offset=...)`


def im_lup_points(n_max=10_000, seed=0):
    """At most 10^4 points of the stride-1000 subsample (uv, frequency, weight,
    ddid, observation), a seeded random choice over the 26 ddids."""
    zs = [dict(np.load(f)) for f in sorted(SUBSAMPLE.glob("ddid_*.npz"))]
    cat = lambda k: np.concatenate([z[k] for z in zs])  # noqa: E731
    reps = [len(z["u"]) for z in zs]
    z = dict(u=cat("u"), v=cat("v"), freq=cat("freq"), weight=cat("weight").astype(float),
             ddid=np.repeat([int(z["ddid"]) for z in zs], reps), obs=np.repeat([int(z["obsid"]) for z in zs], reps))
    keep = np.sort(np.random.default_rng(seed).choice(len(z["u"]), min(n_max, len(z["u"])), replace=False))
    return {k: a[keep] for k, a in z.items()}


def injected_truth(z):
    """Truth nuisances, drawn once (seed 1): g in 0.9-1.1, offsets of a few mas,
    alpha = 2.5, s_d in 1.1-2.0, F = 2.5 Jy (ten times IM Lup's flux, so that a few-mas offset is measurable with 10^4 points)."""
    rng = np.random.default_rng(1)
    n_obs, n_ddid = len(np.unique(z["obs"])), len(np.unique(z["ddid"]))
    return {"cal.flux": 2.5, "cal.gain": rng.uniform(0.9, 1.1, n_obs - 1),
            "cal.offset": rng.uniform(-4.0, 4.0, (n_obs - 1, 2)), "cal.sigma": rng.uniform(1.1, 2.0, n_ddid), "cal.alpha": 2.5}


def injected_model(z, truth, scene_vis, nu0):
    """M = F g_e (nu/nu0)^alpha exp(-2 pi i (da u + dd v)) S, numpy only (reference observation: g = 1, no offset)."""
    gain = np.concatenate([[1.0], truth["cal.gain"]])[z["obs"]]
    off = np.vstack([[0.0, 0.0], truth["cal.offset"]])[z["obs"]]
    return (truth["cal.flux"] * gain * (z["freq"] / nu0) ** truth["cal.alpha"]
            * np.exp(-2j * np.pi * MAS * (off[:, 0] * z["u"] + off[:, 1] * z["v"])) * scene_vis)


def nuisance_fisher(z, truth, scene_vis, nu0):
    """Fisher matrix of (F, g, offsets, alpha) and the error scales, analytic derivatives, numpy only.

    The model is M = F g_e (nu/nu0)^alpha exp(-2 pi i (da u + dd v)) S, with Re and Im each
    Gaussian of variance s_d^2 / w. The mean parameters have
    I_ij = sum (w / s_d^2) Re(dM_i conj(dM_j)); each s_d has 4 N_d / s_d^2 (2 N_d components)
    and no cross terms with the means at the truth. Returns the covariance diagonal in the
    order flux, gain, offset (row-major), sigma, alpha."""
    obs = z["obs"]
    n_obs = obs.max() + 1
    gain = np.concatenate([[1.0], truth["cal.gain"]])[obs]
    lam = np.log(z["freq"] / nu0)
    m = injected_model(z, truth, scene_vis, nu0)
    cols = [m / truth["cal.flux"]]  # flux
    cols += [np.where(obs == e, m / gain, 0.0) for e in range(1, n_obs)]  # gains
    for e in range(1, n_obs):  # offsets: d/d(da) = -2 pi i MAS u M
        cols += [np.where(obs == e, -2j * np.pi * MAS * z["u"] * m, 0.0), np.where(obs == e, -2j * np.pi * MAS * z["v"] * m, 0.0)]
    cols.append(lam * m)  # alpha
    jac = np.array(cols)  # (p, N)
    w = z["weight"] / truth["cal.sigma"][z["ddid"]] ** 2
    info = (jac * w).real @ jac.real.T + (jac * w).imag @ jac.imag.T
    cov = np.linalg.inv(info)
    n_d = np.bincount(z["ddid"])
    sig_var = truth["cal.sigma"] ** 2 / (4 * n_d)
    p = np.sqrt(np.diag(cov))
    n_g = n_obs - 1
    order = np.concatenate([[p[0]], p[1:1 + n_g], p[1 + n_g:1 + 3 * n_g], np.sqrt(sig_var), [p[-1]]])
    return order


def flatten(values):
    return np.concatenate([np.ravel(values[k]) for k in ("cal.flux", "cal.gain", "cal.offset", "cal.sigma", "cal.alpha")])


def _names(truth):
    n_obs = len(truth["cal.gain"]) + 1
    out = ["flux"] + [f"gain[{e}]" for e in range(1, n_obs)]
    out += [f"offset[{e}].{ax}" for e in range(1, n_obs) for ax in ("ra", "dec")]
    return out + [f"sigma[{d}]" for d in range(len(truth["cal.sigma"]))] + ["alpha"]


def _table(truth, values, sigma):
    want, got = flatten(truth), flatten(values)
    rows = ["| parameter | truth | fit | sigma (Fisher) | pull |", "|---|---|---|---|---|"]
    rows += [f"| {n} | {t:.4g} | {g:.4g} | {sg:.2g} | {(g - t) / sg:+.2f} |" for n, t, g, sg in zip(_names(truth), want, got, sigma)]
    return "\n".join(rows)


def _free_image_fit(data, nuis, nu0, brightness, truth, sigma, init):
    """One fit with the 48 x 48 image free (flat start, MaxEntropy 1e-3 x N, L-BFGS, 45 s limit):
    reported, not a pass condition."""
    import time

    import numpyro.distributions as dist
    from virgil import fit
    from virgil.imaging import MaxEntropy

    scene = vba.image_scene(IM_PIXEL)
    priors = nuis.priors(offset=OFFSET_BOUND)
    priors["log_brightness"] = dist.Normal(0.0, 3.0)
    start = dict(init, log_brightness=np.zeros((IM_NPIX, IM_NPIX)))
    t0 = time.time()
    with ft_defaults(ft="dft"):
        result = fit(lambda **values: scene(values), priors, data=(), regularizers=[MaxEntropy(1e-3 * len(data))],
                     likelihoods=[data.term(scene, nuis)], init=start, method="lbfgs", max_steps=3000, time_limit=45)
    pull = (flatten(result.values) - flatten(truth)) / sigma
    record("free_image_max_abs_pull", np.abs(pull).max())
    record("free_image_median_abs_pull", np.median(np.abs(pull)))
    record("free_image_seconds", time.time() - t0)
    img = np.asarray(result.model.brightness) if hasattr(result.model, "brightness") else None
    err = None if img is None else np.abs(img - brightness).sum() / 2
    return f"free image: stop={result.info.get('stop')}, {time.time() - t0:.0f} s, max |pull| {np.abs(pull).max():.1f}, median {np.median(np.abs(pull)):.1f}, total-variation distance to truth {err}\n" + _table(truth, result.values, sigma)


@pytest.mark.validates("virgil.radio.ComplexVisData.term", "virgil.radio.CalibrationNuisance", roots=["statistics"], tier="B")
@pytest.mark.skipif(not SUBSAMPLE.exists(), reason="IM Lup subsample missing (scripts/alma_imlup_subsample.py)")
def test_nuisances_are_recovered_within_three_sigma_on_im_lups_uv_coverage():
    """Inject a numpy ring + blob (48^2, 25 mas) with g in 0.9-1.1 over 6 observations, offsets of a few mas,
    alpha 2.5, s_d in 1.1-2.0 into <= 10^4 points of IM Lup's uv, frequencies, weights and labels
    (noise seed 0); `virgil.fit` with the image held at the truth recovers every nuisance
    within 3 sigma of the numpy Fisher matrix (analytic derivatives). Then one fit with the image free
    (MaxEntropy, small weight) is run and its errors are recorded only."""
    import time

    from virgil import fit

    z = im_lup_points()
    n = len(z["u"])
    assert n <= 10_000 and len(np.unique(z["ddid"])) == 26 and len(np.unique(z["obs"])) == 7
    truth = injected_truth(z)
    nu0 = float(np.sum(z["weight"] * z["freq"]) / np.sum(z["weight"]))
    brightness = vba.ring_and_blob(IM_NPIX, IM_PIXEL)
    scene_vis = sky.visibility(sky.pixel_image(brightness, IM_PIXEL), z["u"], z["v"], 1.0)
    mean = injected_model(z, truth, scene_vis, nu0)
    rng = np.random.default_rng(0)
    sig = truth["cal.sigma"][z["ddid"]] / np.sqrt(z["weight"])
    vis = mean + sig * (rng.normal(size=n) + 1j * rng.normal(size=n))
    data = virgil_data(z, vis)
    nuis = radio.CalibrationNuisance(data, nu0_hz=nu0)
    scene = vba.fixed_image_scene(brightness, IM_PIXEL)
    init = nuis.defaults()
    term = data.term(scene, nuis)
    sigma = nuisance_fisher(z, truth, scene_vis, nu0)
    t0 = time.time()
    with ft_defaults(ft="dft"):
        cold = fit(lambda **values: scene(values), nuis.priors(), data=(), likelihoods=[term], init=init, method="lbfgs", max_steps=2000)
        result = fit(lambda **values: scene(values), nuis.priors(offset=OFFSET_BOUND), data=(), likelihoods=[term], init=init, method="lbfgs", max_steps=2000)
    t_fixed = time.time() - t0
    # observation, not a pass condition: the default offset prior is +-1/(2 q_min), about 8000 mas; at IM Lup's
    # own flux (0.25 Jy) a cold start with it ended in a phase-wrapped local mode (gains at their bounds)
    record("cold_start_default_offset_prior_max_abs_pull", np.abs((flatten(cold.values) - flatten(truth)) / sigma).max())
    record("cold_start_loglike_minus_truth", float(term.loglike(cold.values)) - float(term.loglike(truth)))
    fitted, want = flatten(result.values), flatten(truth)
    pull = (fitted - want) / sigma
    record("n_points", n)
    record("max_abs_pull", np.abs(pull).max())
    record("fit_seconds_fixed_image", t_fixed)
    table = _table(truth, result.values, sigma)
    print("\n" + table)
    free = _free_image_fit(data, nuis, nu0, brightness, truth, sigma, init)
    print(free)
    assert len(pull) == 1 + 6 + 12 + 26 + 1
    assert np.abs(pull).max() < 3.0, np.round(pull, 2)


@pytest.mark.validates("external_bridge.alma_worker", roots=["self-consistency"], kind="guard")
def test_read_records_the_datacolumn_and_refuses_a_column_the_ms_lacks(ms_clean):
    """The column is an explicit, recorded choice: the result names it (upper case), and CORRECTED_DATA on an
    MS without it is an error, not a silent fall back to DATA."""
    ms, tmp = ms_clean
    res = _worker({"task": "read", "path": ms, "out": str(tmp / "dc"), "datacolumn": "data"})
    assert res["datacolumn"] == "DATA"
    with pytest.raises(Exception, match="CORRECTED_DATA"):
        _worker({"task": "read", "path": ms, "out": str(tmp / "dc2"), "datacolumn": "CORRECTED_DATA"})
