"""Stage O1 of design/plan_oidb.md: the fit recipes of scripts/oidb_fit.py, on tiny
simulated data with a known answer, before they run on the authors' OiDB files on
OzSTAR.

The truth comes from our own closed forms (crosscheck.sky, crosscheck.orbits) and
OIFITS writer (crosscheck.simulate), never from virgil; the scripts fit with virgil.
Each model the O1 recipes use gets one recovery check: a point binary per epoch, the
CANDID-style binary with bandwidth smearing (and a control: the same fit without
smearing must miss the flux), the uniform disk of the pi1 Gru recipe, and a Keplerian
orbit through start_from_positions(..., scales="marginal").
"""

import importlib.util
import pathlib

import numpy as np
import pytest
from astropy.io import fits

from crosscheck import orbits, simulate, sky
from evidence.plugin import record

ROOT = pathlib.Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.x64

UTS = np.array([[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]])  # VLTI UTs, E-N (m)
ATS = UTS * 0.25  # a compact quadruplet (7-33 m), like the small AT configuration PIONIER used for pi1 Gru


@pytest.fixture(scope="module")
def of():
    spec = importlib.util.spec_from_file_location("oidb_fit", ROOT / "scripts" / "oidb_fit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _observe(path, vis_fn, *, stations=UTS, wavelengths=(2.1e-6, 2.2e-6, 2.3e-6), hours=(-2, -1, 0, 1, 2),
             sigma_v2=0.01, sigma_cp=0.5, seed=1, mjd0=None, resolving=None, cp=True):
    """crosscheck.simulate.observe, then (astropy) shift the times to start at mjd0 and
    set EFF_BAND = lambda / R."""
    simulate.observe(path, vis_fn, stations, hour_angles_h=np.asarray(hours, float),
                     wavelengths=np.asarray(wavelengths), sigma_v2=sigma_v2, sigma_cp_deg=sigma_cp,
                     rng=np.random.default_rng(seed), closure_phases=cp)
    if mjd0 is None and resolving is None:
        return path
    with fits.open(path, mode="update") as h:
        for hdu in h:
            name = hdu.header.get("EXTNAME")
            if mjd0 is not None and name in ("OI_VIS2", "OI_T3"):
                hdu.data["MJD"] = hdu.data["MJD"] - 60000.0 + mjd0
            if resolving is not None and name == "OI_WAVELENGTH":
                hdu.data["EFF_BAND"] = hdu.data["EFF_WAVE"] / resolving
    return path


def _binary(dra, ddec, flux):
    def vis(u, v, w):
        return (sky.vis_point(u, v, w) + flux * sky.vis_point(u, v, w, dra, ddec)) / (1 + flux)
    return vis


def _within(value, sigma, truth, n=4.0):
    return abs(value - truth) <= n * sigma


@pytest.mark.validates("virgil.models.BinaryModelCartesian", "virgil.fitting.fit", "virgil.inference.laplace_cov",
                       "virgil.oifits.read_oifits", roots=["mathematics", "standards"])
def test_epoch_fit_recovers_a_point_binary(of, tmp_path):
    truth = dict(dra=3.0, ddec=-4.0, flux=0.3)
    path = _observe(tmp_path / "b.fits", _binary(**truth))
    data = of.load([path])
    grid = of._grid(8.0, 0.25, np.linspace(0.1, 1.0, 10))
    priors = {"dra": of.dist.Uniform(-10, 10), "ddec": of.dist.Uniform(-10, 10), "flux": of.dist.LogUniform(0.05, 1)}
    e = of.fit_epoch("sim", data, of.point_binary, priors, grid)
    for k, t in truth.items():
        assert _within(*e["values"][k], t), (k, e["values"][k], t)
    rho, pa = np.hypot(3.0, -4.0), np.degrees(np.arctan2(3.0, -4.0)) % 360
    assert _within(*e["values"]["rho_mas"], rho) and _within(*e["values"]["pa_deg"], pa)
    # the raw chi2/N on the quoted errors comes first, and is ~1 for honest errors
    assert 0.5 < e["chi2_raw"]["all"]["chi2_red"] < 2.0
    assert 0.6 < e["scales"]["vis_scale"] < 1.6 and 0.6 < e["scales"]["phi_scale"] < 1.6


def _smeared_candid(ud1, dra, ddec, flux, resolved, resolving, n=101):
    """The band-averaged CANDID binary: a direct average over n sub-channels of the
    closed-form visibilities (top-hat channel of width lambda/R, flat spectrum)."""
    def vis(u, v, w):
        out = 0.0
        for o in (np.arange(n) + 0.5) / n - 0.5:
            wk = w * (1 + o / resolving)
            out = out + sky.vis_uniform_disk(u, v, wk, ud1) + flux * sky.vis_point(u, v, wk, dra, ddec)
        return out / n / (1 + flux + resolved)
    return vis


@pytest.mark.validates("virgil.models.UniformDisk", "virgil.models.System", "virgil.fitting.fit",
                       roots=["mathematics", "standards"])
def test_candid_binary_with_smearing_recovers_flux(of, tmp_path):
    """CANDID's model with bandwidth smearing (the A-star recipe) recovers the truth;
    without smearing the same fit misses the flux ratio (the control below)."""
    truth = dict(ud1=1.0, dra=20.0, ddec=12.0, flux=0.05, resolved=-0.03)
    resolving = 30.0
    path = _observe(tmp_path / "c.fits", _smeared_candid(**truth, resolving=resolving),
                    wavelengths=(1.55e-6, 1.6e-6, 1.65e-6, 1.7e-6, 1.75e-6), sigma_v2=0.002, sigma_cp=0.2,
                    resolving=resolving)
    assert abs(of.resolving_power([path]) - resolving) < 0.01
    priors = {"dra": of.dist.Uniform(-35, 35), "ddec": of.dist.Uniform(-35, 35), "flux": of.dist.LogUniform(1e-3, 1),
              "ud1": of.dist.LogUniform(0.05, 3.0), "resolved": of.dist.Uniform(-0.3, 0.3)}
    grid = of._grid(30.0, 0.5, np.geomspace(0.01, 1.0, 6))
    e = of.fit_epoch("sim", of.load([path]), of.candid_scene(of.resolving_power([path]), ud2=0.01), priors, grid,
                     extra={"ud1": 0.5, "resolved": 0.0})
    for k, t in truth.items():
        assert _within(*e["values"][k], t), (k, e["values"][k], t)
    assert e["chi2_raw"]["all"]["chi2_red"] < 2.0


@pytest.mark.validates("virgil.models.UniformDisk", "virgil.models.System", "virgil.fitting.fit",
                       roots=["mathematics"], kind="control")
def test_candid_binary_without_smearing_misses_the_flux(of, tmp_path):
    truth = dict(ud1=1.0, dra=20.0, ddec=12.0, flux=0.05, resolved=-0.03)
    path = _observe(tmp_path / "c.fits", _smeared_candid(**truth, resolving=30.0),
                    wavelengths=(1.55e-6, 1.6e-6, 1.65e-6, 1.7e-6, 1.75e-6), sigma_v2=0.002, sigma_cp=0.2,
                    resolving=30.0)
    priors = {"dra": of.dist.Uniform(-35, 35), "ddec": of.dist.Uniform(-35, 35), "flux": of.dist.LogUniform(1e-3, 1),
              "ud1": of.dist.LogUniform(0.05, 3.0), "resolved": of.dist.Uniform(-0.3, 0.3)}
    grid = of._grid(30.0, 0.5, np.geomspace(0.01, 1.0, 6))
    e = of.fit_epoch("sim", of.load([path]), of.candid_scene(None, ud2=0.01), priors, grid,
                     extra={"ud1": 0.5, "resolved": 0.0})
    assert not _within(*e["values"]["flux"], truth["flux"]) or e["chi2_raw"]["all"]["chi2_red"] > 2.0


@pytest.mark.validates("virgil.models.UniformDisk", "virgil.fitting.fit", "virgil.oifits.read_oifits",
                       roots=["mathematics", "standards"])
def test_pi1gru_recipe_recovers_a_uniform_disk(of, tmp_path):
    """The pi1 Gru recipe end to end (file discovery, the chi2 scan, both variants) on a
    simulated 18 mas disk observed in three H channels."""
    def vis(u, v, w):
        return sky.vis_uniform_disk(u, v, w, 18.0)
    _observe(tmp_path / "PI_GRU_forImage.fits", vis, stations=ATS, wavelengths=(1.625e-6, 1.678e-6, 1.73e-6),
             hours=np.linspace(-3, 3, 7), sigma_v2=0.002, cp=False)
    out = of.recipe_pi1gru(str(tmp_path), {})
    assert _within(*out["parametric"]["values"]["ud_mas"], 18.0)
    assert 0.5 < out["parametric"]["chi2_raw"]["all"]["chi2_red"] < 2.0
    lobe = out["variants"][0]
    assert lobe["n_vis"] < out["parametric"]["n_vis"]
    assert _within(*lobe["values"]["ud_mas"], 18.0)


ORBIT = dict(period=12.1, t_peri=60005.0, ecc=0.25, inc=35.0, omega=60.0, Omega=210.0, a=6.0, flux=0.5)


def _wrap(x):
    return (x + 180.0) % 360.0 - 180.0


@pytest.mark.validates("virgil.orbits.KeplerOrbit", "virgil.orbits.starting_orbits", "virgil.models.Attached",
                       "virgil.fitting.fit", "virgil.likelihood.numpyro_model", roots=["mathematics", "standards"])
def test_orbit_from_marginal_start_recovers_elements(of, tmp_path):
    """Five simulated nights of a point binary on a Kepler orbit, each night five
    one-snapshot files an hour apart, the companion moving between them (~0.1 mas/h;
    positions at each file's own time from crosscheck.orbits, our NumPy evaluator).
    Fitted as recipe_gl229 does: merged nights for start_from_positions(scales=
    "marginal"), then the refit on the nested Epochs({night: {file: data}}), every file
    a snapshot at its own time. The elements come back, (Omega, omega) jointly up to
    the (Omega + 180, omega + 180) twin (positions alone cannot tell them apart), and
    the time of periastron, which checks that KeplerOrbit's dt_peri is t_peri - t_ref as
    kepler() and elements() assume."""
    per_night, per_file, nights = {}, {}, [60000.2, 60003.2, 60009.2, 60015.2, 60030.2]
    for k, m in enumerate(nights):
        paths = []
        for j, h in enumerate((-2.0, -1.0, 0.0, 1.0, 2.0)):
            t = m + j / 24.0
            dra, ddec = (float(x) for x in np.ravel(orbits.position(
                t, ORBIT["period"], ORBIT["t_peri"], ORBIT["ecc"], ORBIT["inc"], ORBIT["omega"], ORBIT["Omega"],
                ORBIT["a"])))
            paths.append(_observe(tmp_path / f"n{k}_{j}.fits", _binary(dra, ddec, ORBIT["flux"]), hours=(h,),
                                  sigma_cp=0.3, sigma_v2=0.005, seed=10 + 5 * k + j, mjd0=t))
        per_night[f"n{k}"] = of.load(paths)
        per_file[f"n{k}"] = {p.name: of.load([p]) for p in paths}
    priors = of.orbit_priors((11.5, 12.8), (2.0, 20.0), (0.05, 1.0))
    grid = of._grid(9.0, 0.2, np.linspace(0.2, 1.0, 5))
    fit_data = of.Epochs(per_file)
    assert len(fit_data.data) == 25 and len(set(np.round(fit_data.times, 4))) == 25
    out = of.fit_orbit(of.Epochs(per_night), fit_data, of.orbital_point_binary, priors, grid=grid,
                       periods=np.arange(11.8, 12.4, 0.002), eccs=np.arange(0.0, 0.5, 0.1), n_candidates=60,
                       n_refine=2, min_gap=3.0, nuts=dict(warmup=20, samples=20, chains=2))
    m = out["map"]
    assert abs(m["period_day"] - ORBIT["period"]) < 0.05
    assert abs(m["a_mas"] - ORBIT["a"]) < 0.2
    assert abs(m["ecc"] - ORBIT["ecc"]) < 0.05
    assert abs(m["inc_deg"] - ORBIT["inc"]) < 3.0
    assert abs(m["flux_ratio"] - ORBIT["flux"]) < 0.05
    # every file fitted at its own time: the motion within a night is in the model
    assert 0.5 < out["chi2_raw"]["chi2_red"] < 2.0
    assert len(out["datasets"]) == 25
    same = max(abs(_wrap(m["Omega_deg"] - ORBIT["Omega"])), abs(_wrap(m["omega_deg"] - ORBIT["omega"])))
    twin = max(abs(_wrap(m["Omega_deg"] - ORBIT["Omega"] - 180)), abs(_wrap(m["omega_deg"] - ORBIT["omega"] - 180)))
    assert min(same, twin) < 5.0, (m["Omega_deg"], m["omega_deg"])
    P = ORBIT["period"]
    dt = (m["t_peri_mjd"] - ORBIT["t_peri"] + P / 2) % P - P / 2
    assert abs(dt) < 0.02 * P, dt
    # one fit per period band N = round(T/P), each free to leave its band; the winner is at the true period
    won = min((b for b in out["bands"] if "loss" in b), key=lambda b: b["loss"])
    assert abs(won["elements"]["period_day"] - ORBIT["period"]) < 0.05, out["bands"]
    assert won["n_fit"] == round(30.0 / ORBIT["period"])
    assert [r["name"] for r in out["epoch_checks"]] == [f"n{k}" for k in range(5)]
    # the NUTS path runs end to end (a smoke test: 20 draws are not a posterior)
    assert set(out["nuts"]["elements"]) >= {"period_day", "a_mas", "ecc"}
    assert np.isfinite(out["nuts"]["elements"]["a_mas"]["median"])


AMPS = [0.30, 0.60, 0.90, 0.50, 0.20]  # the injected per-night contrast loss: V^2 -> A V^2


def _orbit_nights(of, tmp_path, amps, *, per_night_only=False, nights=(60000.2, 60003.2, 60009.2, 60015.2, 60030.2)):
    """The simulated Kepler orbit of ORBIT, five files an hour apart on each night, night k's V^2 scaled by amps[k]
    (a contrast loss, the same on every baseline; closure phases do not see it)."""
    per_night, per_file = {}, {}
    for k, m in enumerate(nights):
        paths = []
        for j, h in enumerate((-2.0, -1.0, 0.0, 1.0, 2.0)):
            t = m + j / 24.0
            dra, ddec = (float(x) for x in np.ravel(orbits.position(
                t, ORBIT["period"], ORBIT["t_peri"], ORBIT["ecc"], ORBIT["inc"], ORBIT["omega"], ORBIT["Omega"],
                ORBIT["a"])))
            binary = _binary(dra, ddec, ORBIT["flux"])
            vis = lambda u, v, w, binary=binary, a=amps[k]: np.sqrt(a) * binary(u, v, w)  # noqa: E731
            paths.append(_observe(tmp_path / f"n{k}_{j}.fits", vis, hours=(h,), sigma_cp=0.3, sigma_v2=0.005,
                                  seed=10 + 5 * k + j, mjd0=t))
        per_night[f"n{k}"] = of.load(paths)
        per_file[f"n{k}"] = {q.name: of.load([q]) for q in paths}
    return per_night, per_file


@pytest.mark.validates("virgil.likelihood.whitened_residuals", roots=["mathematics"], kind="check")
def test_vis_amp_of_recovers_a_contrast_loss(of, tmp_path):
    """One night of a point binary with V^2 scaled by 0.4: the amplitude of the true model is 0.4 (to the
    noise), and 1 without the loss; a model with no companion gives a different amplitude (the estimator
    depends on the model); closure-phase-only data give None. The claim about virgil is that its V^2
    residuals are linear in ``vis``, with errors that do not depend on the data: ``vis_amp_of`` solves for
    the amplitude from two evaluations (the data and twice the data), and it recovers the injected loss."""
    truth = _binary(3.0, -4.0, 0.3)
    scene = of.point_binary
    for loss in (1.0, 0.4):
        path = _observe(tmp_path / f"a{loss}.fits", lambda u, v, w, l=loss: np.sqrt(l) * truth(u, v, w),
                        sigma_v2=0.005, seed=3)
        data = of.load([path])
        a = of.vis_amp_of(scene(dra=3.0, ddec=-4.0, flux=0.3), data)
        assert abs(a - loss) < 0.02, (loss, a)
    assert abs(of.vis_amp_of(scene(dra=3.0, ddec=-4.0, flux=0.0), data) - 0.4) > 0.05  # the model matters
    assert of.vis_amp_of(scene(dra=3.0, ddec=-4.0, flux=0.3), data.select(observables="phi")) is None
    record("vis_amp_recovered", a)


@pytest.mark.validates("virgil.orbits.KeplerOrbit", "virgil.fitting.fit", "virgil.likelihood.numpyro_model",
                       roots=["mathematics", "standards"])
def test_published_start_joins_the_modes_and_refits_to_the_orbit(of, tmp_path):
    """``published`` (a paper's orbit, here ORBIT displaced by 0.03 d in P, 10 % in a and 3 deg in i) is one more
    start: tagged ``published`` in the modes and reported in ``published_start``, and refitted to the
    simulated truth. The blind search runs as always."""
    per_night, per_file = _orbit_nights(of, tmp_path, [1.0] * 5)
    priors = of.orbit_priors((11.5, 12.8), (2.0, 20.0), (0.05, 1.0))
    grid = of._grid(9.0, 0.2, np.linspace(0.2, 1.0, 5))
    pub = dict(period=ORBIT["period"] + 0.03, t_peri=ORBIT["t_peri"] + 0.3, ecc=ORBIT["ecc"], inc=ORBIT["inc"] + 3.0,
               omega=ORBIT["omega"], Omega=ORBIT["Omega"], a=ORBIT["a"] * 1.1, flux=ORBIT["flux"])
    out = of.fit_orbit(of.Epochs(per_night), of.Epochs(per_file), of.orbital_point_binary, priors, grid=grid,
                       periods=np.arange(11.8, 12.4, 0.002), eccs=np.arange(0.0, 0.5, 0.1), n_candidates=30,
                       n_refine=1, min_gap=3.0, published=pub)
    ps = out["published_start"]
    assert abs(ps["elements"]["period_day"] - ORBIT["period"]) < 0.01 and abs(ps["elements"]["a_mas"] - ORBIT["a"]) < 0.2
    assert "published" in {m["band"] for m in out["modes"]}
    assert abs(out["map"]["period_day"] - ORBIT["period"]) < 0.01
    assert out["start"] in ("published", "blind") and out["blind_best"]["loss"] >= out["loss"] - 1e-6
    # control: a published orbit in the wrong band (a different period and other elements) is not imposed
    wrong = dict(pub, period=ORBIT["period"] * 11 / 10, ecc=0.6, inc=80.0, a=ORBIT["a"] * 2, flux=0.2)
    out2 = of.fit_orbit(of.Epochs(per_night), of.Epochs(per_file), of.orbital_point_binary, priors, grid=grid,
                        periods=np.arange(11.8, 12.4, 0.002), eccs=np.arange(0.0, 0.5, 0.1), n_candidates=30,
                        n_refine=1, min_gap=3.0, published=wrong)
    assert out2["modes"][0]["band"] != "published" and out2["start"] == "blind"
    assert abs(out2["map"]["period_day"] - ORBIT["period"]) < 0.02
    assert out2["published_start"]["loss"] > out2["loss"] + 100.0  # the wrong start is far worse and loses
    assert of.fit_orbit(of.Epochs(per_night), of.Epochs(per_file), of.orbital_point_binary, priors, grid=grid,
                        periods=np.arange(11.8, 12.4, 0.002), eccs=np.arange(0.0, 0.5, 0.1), n_candidates=30,
                        n_refine=1, min_gap=3.0)["published_start"] is None


@pytest.mark.validates("pipeline:published-binary", roots=["mathematics"], kind="check")
def test_vis_amp_variant_recovers_the_orbit_and_the_per_night_losses(of, tmp_path):
    """The V^2 + closure-phase orbit with a free amplitude on each night (``virgil.aliases``): V^2 scaled by
    AMPS night by night, a warm start displaced from the truth. The orbit and every night's loss
    come back."""
    aliases = pytest.importorskip("virgil.aliases")
    import inspect
    if "vis_amp" not in inspect.signature(aliases.fit_orbit_aliases).parameters:
        pytest.skip("this virgil's fit_orbit_aliases has no vis_amp (needs virgil a92b97c)")
    per_night, _ = _orbit_nights(of, tmp_path, AMPS)
    pub = dict(period=ORBIT["period"] + 0.03, t_peri=ORBIT["t_peri"] + 0.3, ecc=ORBIT["ecc"], inc=ORBIT["inc"] + 3.0,
               omega=ORBIT["omega"], Omega=ORBIT["Omega"], a=ORBIT["a"] * 1.1, flux=ORBIT["flux"])
    out = of.fit_orbit_vis_amp(per_night, (11.5, 12.8), a_range=(2.0, 20.0), flux_range=(0.05, 1.0), init=pub,
                               n_candidates=20, n_refine=1)
    el = out["elements"]
    assert abs(el["period_day"] - ORBIT["period"]) < 0.03 and abs(el["a_mas"] - ORBIT["a"]) < 0.3
    assert abs(el["ecc"] - ORBIT["ecc"]) < 0.06 and abs(el["inc_deg"] - ORBIT["inc"]) < 4.0
    got = [e["vis_amp"] for e in out["epochs"]]
    assert np.allclose(got, AMPS, atol=0.05), (got, AMPS)
    record("vis_amp_max_error", float(np.max(np.abs(np.array(got) - AMPS))))


@pytest.mark.validates("pipeline:published-binary", roots=["standards"], kind="guard")
def test_alias_bands_tile_the_window_and_split_the_o1_orbits(of):
    """Band N = round(T/P) of the first-to-last-night span T. Gl 229 (T = 125.7 d): Xuan et al.'s
    12.134 d is band 10 and the 11.81 d of the O1 MAP is band 11; iota Peg (T = 288.2 d): the
    published 10.213 d and the O1 MAP's 10.332 d share band 28, so that is no cycle-count alias."""
    def band_of(bands, p):
        (n,) = [n for n, lo, hi in bands if lo <= p <= hi]
        return n
    gl = of.alias_bands([60304.23, 60429.97], (11.8, 12.5))
    assert [n for n, _, _ in gl] == [10, 11]  # by N, so the longest period first
    assert band_of(gl, 12.134) == 10 and band_of(gl, 11.809) == 11
    assert gl[1][1] == 11.8 and gl[0][2] == 12.5 and gl[0][1] == gl[1][2]  # tile the window, no gaps
    ip = of.alias_bands([58413.16, 58701.33], (10.0, 10.4))
    assert band_of(ip, 10.213) == band_of(ip, 10.332) == 28
    with pytest.raises(ValueError):
        of.alias_bands([60000.0], (10.0, 11.0))


@pytest.mark.validates("pipeline:published-binary", roots=["standards"], kind="guard")
def test_reference_positions_agree_with_the_published_epochs(of):
    """The reference rows of the epoch checks: the Xuan et al. orbit through crosscheck.orbits (our
    own solver) gives the positions that virgil's Kepler solver gives at the nights the fits agree
    on (26 Dec 2023, 29 Apr 2024) and the two they do not (28 Feb, 29 Mar 2024); iota Peg's come
    from Table 2 of Anugu et al."""
    from types import SimpleNamespace
    nights = {n: SimpleNamespace(mjd=np.array([m])) for n, m in
              {"2023-12-26": 60304.2327, "2024-02-28": 60368.137, "2024-03-29": 60398.024, "2024-04-29": 60429.974}.items()}
    ref = of.reference_positions("782185b2-0727-42b0-a185-b2072732b047", nights)
    for night, (rho, pa) in {"2023-12-26": (5.57, 177.9), "2024-02-28": (6.16, 313.0), "2024-03-29": (8.04, 67.8),
                             "2024-04-29": (7.47, 347.1)}.items():
        dra, ddec = ref[night]
        assert abs(np.hypot(dra, ddec) - rho) < 0.05, night
        assert abs(_wrap(np.degrees(np.arctan2(dra, ddec)) - pa)) < 0.5, night
    ip = of.reference_positions("fac164e1-d9d0-4500-8164-e1d9d0450099", {"2018-11-21": None, "2019-08-06": None})
    assert abs(np.hypot(*ip["2018-11-21"]) - 10.1793) < 1e-6 and abs(np.hypot(*ip["2019-08-06"]) - 1.1297) < 1e-6


@pytest.mark.validates("pipeline:published-binary", roots=["standards"], kind="guard")
def test_check_epochs_flags_nights_that_may_not_seed_the_orbits(of, tmp_path):
    """Two simulated nights of a point binary, 5 mas apart in one and 6 in the other. A night in
    ``flagged`` and a night whose best peak is closer than ``min_sep`` are barred from seeding, with
    the reason; the reference row gives the offset of the best peak from the reference position."""
    per_night = {}
    for k, (dra, ddec) in enumerate([(3.0, 4.0), (-3.6, 4.8)]):
        per_night[f"n{k}"] = of.load([_observe(tmp_path / f"n{k}.fits", _binary(dra, ddec, 0.5), sigma_cp=0.3,
                                               sigma_v2=0.005, seed=3 + k, mjd0=60000.2 + 20 * k)])
    grid = of._grid(8.0, 0.25, np.linspace(0.2, 1.0, 5))
    rows, _ = of.check_epochs(of.Epochs(per_night), grid, flagged={"n1": "alias peak"}, min_sep=5.5,
                              reference={"n0": (3.0, 4.0)})
    n0, n1 = rows
    assert n0["barred"] and any("closer than" in r for r in n0["reasons"])  # 5.0 mas < 5.5
    assert n1["barred"] and n1["reasons"][0] == "flagged: alias peak"
    assert n0["reference"]["best_offset"] < 0.3 and "reference" not in n1
    rows, _ = of.check_epochs(of.Epochs(per_night), grid)
    assert [r["seeds"] for r in rows] == [True, True] and not any(r["barred"] for r in rows)


def _remove_vis2_baseline(path, pair):
    """Drop every OI_VIS2 row of one baseline, as MIRC-X files lack some (astropy)."""
    with fits.open(path) as h:
        hdus = []
        for hdu in h:
            if hdu.header.get("EXTNAME") == "OI_VIS2":
                keep = np.array([set(map(int, s)) != set(pair) for s in hdu.data["STA_INDEX"]])
                hdu = fits.BinTableHDU(data=hdu.data[keep], header=hdu.header, name="OI_VIS2")
            hdus.append(hdu)
        fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


@pytest.mark.validates("virgil.oifits.read_oifits", roots=["standards"], kind="guard")
def test_triangles_with_a_baseline_in_no_vis2_row_are_kept(of, tmp_path):
    """A T3 row whose leg has no V^2 row (MIRC-X; AL Dor's PIONIER file) used to be dropped
    before virgil read the file. virgil places such a leg from the T3 row's own U/V
    coordinates (virgil 48f8727), so ``load`` keeps every closure phase, unchanged."""
    path = _observe(tmp_path / "t3.fits", _binary(3.0, -4.0, 0.3))
    full = of.load([path])
    with fits.open(path) as h:
        pair = tuple(int(x) for x in h["OI_VIS2"].data["STA_INDEX"][0])
        t3 = np.asarray(h["OI_T3"].data["STA_INDEX"], int)
    uses = np.array([len({*pair} & {*row}) == 2 for row in t3])
    assert uses.any() and not uses.all()
    _remove_vis2_baseline(path, pair)
    of.LOAD_NOTES.clear()
    data = of.load([path])
    assert of.LOAD_NOTES == []  # nothing is dropped, so nothing is noted
    assert np.asarray(data.phi).size == len(t3) * 3 == np.asarray(full.phi).size
    np.testing.assert_allclose(np.asarray(data.phi), np.asarray(full.phi))


@pytest.mark.validates("virgil.oifits.read_oifits", roots=["standards"], kind="guard")
def test_pick_target_reads_the_named_target_and_refuses_another_star(of, tmp_path):
    """Rows under a TARGET_ID that OI_TARGET does not list (the MIRC-X A-star files):
    the named target is read. With a pattern, a file of another star is refused."""
    path = _observe(tmp_path / "ids.fits", _binary(3.0, -4.0, 0.3))
    with fits.open(path, mode="update") as h:
        h["OI_TARGET"].data["TARGET"][0] = "37 And"
        name = "37 And"
        first = float(np.min(h["OI_VIS2"].data["MJD"]))
        n_wave = 3
        left = {e: int(np.sum(np.abs(h[e].data["MJD"] - first) >= 1e-6)) for e in ("OI_VIS2", "OI_T3")}
        for ext in ("OI_VIS2", "OI_T3"):  # the first snapshot under an id OI_TARGET does not list
            h[ext].data["TARGET_ID"][np.abs(h[ext].data["MJD"] - first) < 1e-6] = 7
    assert of.pick_target(str(path)) == name
    of.LOAD_NOTES.clear()
    assert np.asarray(of.load([path]).vis).size == left["OI_VIS2"] * n_wave  # the id-7 snapshot is left out
    note = of.LOAD_NOTES[-1]
    assert note["target_read"] == name and note["rows_read"] == sum(left.values())
    assert note["rows_total"] > note["rows_read"]
    with pytest.raises(ValueError, match="no target matches"):
        of.pick_target(str(path), r"HD.?45166")
    single = _observe(tmp_path / "one.fits", _binary(3.0, -4.0, 0.3))
    with fits.open(single, mode="update") as h:
        h["OI_TARGET"].data["TARGET"][0] = "TYC_732-806-1"
    assert of.pick_target(str(single), r"TYC.?732") is None
    with pytest.raises(ValueError, match="no target matches"):
        of.pick_target(str(single), r"HD.?45166")


@pytest.mark.validates("evidence", roots=["standards"], kind="guard")
def test_every_o1_collection_has_a_recipe(of):
    import json
    lines = (ROOT / "oidb" / "manifest.yml").read_text().splitlines()
    manifest = json.loads("\n".join(x for x in lines if not x.startswith("#")))
    o1 = {c["id"] for c in manifest["collections"] if c["stage"] == "O1"}
    assert o1 == set(of.RECIPES)



# ----------------------------------------------------------------------------- the comparison table

@pytest.fixture(scope="module")
def oc():
    spec = importlib.util.spec_from_file_location("oidb_compare", ROOT / "scripts" / "oidb_compare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ref(cid):
    import json
    return json.loads((ROOT / "oidb" / "references" / f"{cid}.json").read_text())


def _epoch(label, mjd, rho, pa, **extra):
    p = np.radians(pa)
    values = dict(rho_mas=[rho, 0.01], pa_deg=[pa, 0.01], dra=[rho * np.sin(p), 0.01], ddec=[rho * np.cos(p), 0.01])
    values.update({k: [v, 0.01] for k, v in extra.items()})
    return dict(label=label, mjd=mjd, values=values, chi2_raw={"all": {"chi2_red": 1.1}},
                scales={"vis_scale": 1.0, "phi_scale": 1.0})


def _orbit(el):
    return dict(map=el, chi2_raw={"chi2": 110.0, "n": 100, "chi2_red": 1.1}, datasets=[])


def _fake_fits(oc, tmp_path, shift=0.0):
    """fit JSONs that reproduce every published number (plus ``shift`` mas in
    separation), in the format scripts/oidb_fit.py writes."""
    import json
    fits = {}
    g = _ref("782185b2-0727-42b0-a185-b2072732b047")["orbit"]["pmoired_frequentist"]
    gl_el = dict(period_day=g["P_day"][0], t_peri_mjd=g["T_peri_mjd"][0] + 3 * g["P_day"][0], ecc=g["e"][0],
                 inc_deg=g["i_deg"][0], omega_deg=(g["omega_primary_deg"][0] + 180) % 360,
                 Omega_deg=g["Omega_deg"][0], a_mas=g["a_au"][0] * 173.574, flux_ratio=g["flux_ratio_2um"][0])
    fits["782185b2-0727-42b0-a185-b2072732b047"] = dict(epochs=[_epoch("2023-12-26", 60305.1, 7.0, 40.0, flux=0.47)],
                                                         orbit=_orbit(gl_el))
    h = _ref("696baf06-6c3c-424d-abaf-066c3c324d99")
    ho = h["orbit"]
    fits["696baf06-6c3c-424d-abaf-066c3c324d99"] = dict(
        epochs=[_epoch(r["date"], r["mjd"] + 0.1, r["rho_mas"][0], r["pa_deg"][0],
                       flux=r["flux_ratio"][0] / (1 - r["flux_ratio"][0])) for r in h["epochs"]],
        orbit=_orbit(dict(period_day=ho["P_day"][0], t_peri_mjd=ho["T_peri_mjd"][0], ecc=ho["e"][0],
                          inc_deg=ho["i_deg"][0], omega_deg=(ho["omega_deg"][0] + 180) % 360,  # the (Omega+180, omega+180) twin; omega is the companion's, no shift
                          Omega_deg=(ho["Omega_deg"][0] + 180) % 360, a_mas=ho["a_mas"][0], flux_ratio=0.8)))
    i = _ref("fac164e1-d9d0-4500-8164-e1d9d0450099")
    io = i["orbit"]
    fits["fac164e1-d9d0-4500-8164-e1d9d0450099"] = dict(
        epochs=[_epoch(r["date"], None, r["rho_mas"][0] + shift, r["pa_deg"][0], flux=1 / r["flux_ratio"][0])
                for r in i["epochs"]],
        orbit=_orbit(dict(period_day=io["P_day"][0], t_peri_mjd=io["T_peri_unstated_system"][0], ecc=io["e"][0],
                          inc_deg=io["i_deg"][0], omega_deg=io["omega_deg"][0], Omega_deg=io["Omega_deg"][0],
                          a_mas=io["a_mas"][0], flux_ratio=0.2)))
    w = _ref("647a22a9-5047-4220-ba22-a95047022072")["targets"]
    wi = w["iot Peg"]["epoch_2018-10-22"]
    so = next(e for e in w["sig Ori"]["epochs"] if e["date"] == "2011-09-29")
    sig = _epoch("2011-09-29", so["hjd_minus_2400000"] - 0.5, so["rho_mas"][0], so["pa_deg"][0])
    sig["fractions"] = {k: v[0] for k, v in so["fractions"].items()}
    fits["647a22a9-5047-4220-ba22-a95047022072"] = dict(targets={
        "iot Peg": dict(epochs=[_epoch("2018-10-22", None, wi["rho_mas"][0], wi["pa_deg"][0], flux=1 / wi["flux_ratio"][0])]),
        "sig Ori": dict(epochs=[sig], variants=[])})
    a = _ref("bda75673-61c6-49f0-a756-7361c699f0c4")
    fits["bda75673-61c6-49f0-a756-7361c699f0c4"] = dict(targets={
        r["star"]: dict(epochs=[_epoch(r["star"], r["mjd"], r["rho_mas"], r["pa_deg_E_of_N"],
                                       flux=r["contrast"]["flux_ratio_percent_of_primary"] / 100,
                                       resolved=r["resolved_flux_percent_primary"]["value"] / 100, ud1=r["UD1_mas"],
                                       **({"ud2": r["UD2_mas"]} if r["UD2_mas"] else {}))])
        for r in a["epochs"]})
    d = _ref("f4afc4cd-fd31-40d3-afc4-cdfd3150d340")["epochs"][0]
    fits["f4afc4cd-fd31-40d3-afc4-cdfd3150d340"] = dict(
        epochs=[_epoch(d["date"], d["mjd"], d["rho_mas"][0], d["pa_deg"][0], flux=d["flux_ratio"][0])], variants=[])
    fits["19f7e2cf-2a03-4bb2-b7e2-cf2a03bbb245"] = dict(
        parametric=dict(values={"ud_mas": [18.37, 0.05]}, chi2_raw={"all": {"chi2_red": 1.3}}, scales={}), variants=[])
    assert set(fits) == set(oc.COMPARE)
    for cid, f in fits.items():
        (tmp_path / f"fit_{cid}.json").write_text(json.dumps(dict(f, collection=cid)))
    return tmp_path


@pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")
def test_compare_table_passes_the_published_numbers(oc, tmp_path):
    """Fits that reproduce the published values: every scored row passes (including the
    conventions: omega of the primary + 180 for Gl 229 (none for HR 6819), a in au times the parallax, the (Omega,
    omega) twin, 1/flux for iota Peg, r/(1+r) for HR 6819, T_peri folded by whole
    periods), HR 6819 is never scored, and the workshop (cleared, no longer L2) is in the table."""
    rows, missing = oc.compare(str(_fake_fits(oc, tmp_path)))
    assert not missing
    assert any(r["collection"] == "647a22a9-5047-4220-ba22-a95047022072" for r in rows)
    scored = [r for r in rows if r["status"] in ("PASS", "FAIL")]
    assert len(scored) > 30 and all(r["status"] == "PASS" for r in scored), [r for r in scored if r["status"] == "FAIL"]
    assert all(r["status"] == "NOT-CLEAN" for r in rows if r["target"] == "HR 6819")
    orbit = [r for r in rows if r["rule"] == "orbit" and r["dev_sigma"] is not None and r["published_err"]]
    exact = [r for r in orbit if "vs Octofitter" not in r["note"]]  # the fakes reproduce PMOIRED's Gl 229 orbit
    assert len(exact) == 7 * 3 and all(abs(r["dev_sigma"]) < 1e-6 for r in exact), exact
    assert not oc.L2  # no collection is withheld
    work = [r for r in rows if r["collection"] == "647a22a9-5047-4220-ba22-a95047022072"]
    assert any(r["status"] == "PASS" for r in work) and not any(r["status"] in ("FAIL", "WITHHELD") for r in work)
    text = oc.markdown(rows)
    assert oc.CRITERIA_HASH in text and "WITHHELD" not in text
    assert not [r for r in rows if r["status"] == "MISSING"]


@pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")
def test_compare_marks_a_gl229_orbit_seeded_by_the_published_orbit(oc, tmp_path):
    """A headline refitted from the published orbit says so on every orbit row, and the blind search's own
    best orbit is a REPORT row set (a miss stays visible); a blind headline gets neither."""
    import json
    cid = "782185b2-0727-42b0-a185-b2072732b047"
    fit = json.loads((_fake_fits(oc, tmp_path) / f"fit_{cid}.json").read_text())
    blind = dict(loss=-6141.3, band=10, elements=dict(fit["orbit"]["map"], inc_deg=72.8))
    seeded = dict(fit, orbit=dict(fit["orbit"], start="published", blind_best=blind, loss=-6321.2))
    rows, _ = oc.compare(None, fits={cid: seeded})
    mine = [r for r in rows if r["collection"] == cid]
    orbit = [r for r in mine if r["epoch"] == "orbit"]
    assert orbit and all("SEEDED by the published orbit" in r["note"] for r in orbit)
    miss = [r for r in mine if r["epoch"] == "orbit (blind start)" and r["quantity"] == "inc_deg"]
    assert len(miss) == 1 and miss[0]["status"] == "REPORT" and abs(miss[0]["virgil"] - 72.8) < 1e-9  # against PMOIRED
    plain, _ = oc.compare(None, fits={cid: dict(fit, orbit=dict(fit["orbit"], start="blind", blind_best=blind))})
    assert not any("SEEDED" in r["note"] or r["epoch"] == "orbit (blind start)" for r in plain if r["collection"] == cid)


@pytest.mark.validates("evidence", roots=["mathematics"], kind="control")
def test_compare_table_counts_missing_fits_as_failures(oc, tmp_path):
    """A star missing from a fit gives MISSING rows (one per scored quantity of that
    star, not silence), and a missing fit file gives exactly one MISSING row."""
    import json
    d = _fake_fits(oc, tmp_path)
    astar = d / "fit_bda75673-61c6-49f0-a756-7361c699f0c4.json"
    fit = json.loads(astar.read_text())
    del fit["targets"]["HD 29388"]
    astar.write_text(json.dumps(fit))
    rows, _ = oc.compare(str(d))
    miss = [r for r in rows if r["status"] == "MISSING"]
    assert miss and {r["target"] for r in miss} == {"HD 29388"}
    assert len(miss) == 5  # rho, PA, flux, resolved flux, UD1
    (d / "fit_19f7e2cf-2a03-4bb2-b7e2-cf2a03bbb245.json").unlink()
    rows, missing = oc.compare(str(d))
    miss = [r for r in rows if r["status"] == "MISSING" and r["collection"].startswith("19f7e2cf")]
    assert len(miss) == 1 and missing == ["19f7e2cf-2a03-4bb2-b7e2-cf2a03bbb245"]
    assert "Failures (FAIL + MISSING): 6" in oc.markdown(rows, missing)


@pytest.mark.validates("evidence", roots=["literature"], kind="guard")
def test_published_numbers_fix_the_conventions():
    """Where a paper quotes numbers that fix a convention independently of our
    comparator, pin it. HR 6819 (Klement et al. 2025, Table A.1) prints dRA, dDec and
    rho, PA for each epoch: they agree with PA = atan2(dRA, dDec) east of north, dRA
    east, to the printed rounding. Gl 229 B (Xuan et al. 2024, Table 1): a, P and
    M_tot obey Kepler's third law with a in au and P in days, so a_mas = a_au x
    parallax is the right conversion."""
    import json
    import math
    h = json.loads((ROOT / "oidb" / "references" / "696baf06-6c3c-424d-abaf-066c3c324d99.json").read_text())
    for e in h["epochs"]:
        o = e["other"]
        assert abs(math.hypot(o["dRA_mas"], o["dDec_mas"]) - e["rho_mas"][0]) < 2e-3
        pa = math.degrees(math.atan2(o["dRA_mas"], o["dDec_mas"])) % 360
        assert abs(_wrap(pa - e["pa_deg"][0])) < 0.05
    g = json.loads((ROOT / "oidb" / "references" / "782185b2-0727-42b0-a185-b2072732b047.json").read_text())["orbit"]
    for o in g.values():
        mtot_msun = o["M_tot_mjup"][0] * 9.5458e-4  # IAU 2015: M_J / M_sun
        assert abs(o["a_au"][0] ** 3 / (o["P_day"][0] / 365.25) ** 2 / mtot_msun - 1) < 0.01


@pytest.mark.validates("evidence", roots=["mathematics"], kind="control")
def test_compare_table_fails_a_shifted_separation(oc, tmp_path):
    """A separation 0.5 sigma_pub off (iota Peg, fac164e1) fails the 0.25 sigma epoch rule."""
    shift = 0.5 * 0.0176
    rows, _ = oc.compare(str(_fake_fits(oc, tmp_path, shift=shift)))
    hit = [r for r in rows if r["collection"].startswith("fac164e1") and r["quantity"] == "rho_mas"
           and r["epoch"] == "2018-10-22" and r["status"] in ("PASS", "FAIL")]
    assert len(hit) == 1 and hit[0]["status"] == "FAIL" and abs(hit[0]["dev_sigma"] - 0.5) < 1e-6


@pytest.mark.validates("evidence", roots=["mathematics"], kind="check")
def test_error_ellipse_projection(oc):
    """An ellipse aligned with the separation gives sigma_rho = its major axis and
    sigma_PA = minor / rho; rotated by 90 degrees, the other way round."""
    rho, pa = 10.0, 30.0
    p = np.radians(pa)
    dra, ddec = rho * np.sin(p), rho * np.cos(p)
    s_r, s_p = oc.project(dra, ddec, oc.ellipse_cov(0.2, 0.05, pa))
    assert np.isclose(s_r, 0.2) and np.isclose(s_p, np.degrees(0.05 / rho))
    s_r, s_p = oc.project(dra, ddec, oc.ellipse_cov(0.2, 0.05, pa + 90))
    assert np.isclose(s_r, 0.05) and np.isclose(s_p, np.degrees(0.2 / rho))
    # an asymmetric error uses the side the deviation falls on
    assert oc.err_toward([-0.1, 0.3], +1.0) == 0.3 and oc.err_toward([-0.1, 0.3], -1.0) == 0.1


@pytest.mark.validates("evidence", roots=["statistics"], kind="guard")
def test_criteria_are_frozen(oc):
    """The pass/fail rules were fixed before the real fits ran: a change to them must
    change this hash on purpose."""
    assert oc.CRITERIA["epoch"]["max_dev_sigma"] == 0.25 and oc.CRITERIA["parametric"]["max_dev_sigma"] == 2.0
    assert oc.CRITERIA_HASH == "1b35907f4268e909"


@pytest.mark.validates("evidence", roots=["literature"], kind="guard")
def test_hr6819_omega_table3_equals_table2_so_no_shift(oc):
    """Klement+2025's Table 3 omega (labelled omega_Be) equals Table 2's omega_pre-sd within errors
    with the same Omega, so no +180 applies (assuming one node convention throughout the paper).
    Pins the reference file's convention; says nothing about virgil's fit."""
    ref = _ref("696baf06-6c3c-424d-abaf-066c3c324d99")
    assert ref["conventions"]["omega_shift_deg"] == 0.0
    o = ref["orbit"]
    t2 = o["all_solutions"]["astrometry_only (Table 2, p. 5; omega is omega_pre-sd)"]["eccentric"]
    assert abs(oc.wrap180(o["omega_deg"][0] - t2["omega_deg"][0])) < o["omega_deg"][1]
    assert abs(oc.wrap180(o["Omega_deg"][0] - t2["Omega_deg"][0])) < 1.5 * o["Omega_deg"][1]  # same 12 positions: errors correlated
