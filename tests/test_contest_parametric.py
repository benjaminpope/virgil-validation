"""The parametric ladder for the 2004 data (scripts/contest_parametric.py).

The geometry is checked against NumPy (the model's visibilities against closed
forms, the star-frame spot position against score_2004's truth, the rendered
image against a direct sum), and the ladder must recover a simulated spotted
star plus companion at the 2004 coverage: bench phantom 0's star and its
off-centre spot (crosscheck.phantoms, as contest_bench draws it for the
2004 data2 coverage), with phantom 1's companion added (phantom 0 has none),
observed at data2's baselines with its quoted errors. The truth is never used
as a start or a prior.
"""

import importlib.util
import os
import pathlib
import sys

import numpy as np
import pytest
from scipy.special import j1

ROOT = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


needs_ellipse = pytest.mark.skipif(
    importlib.util.find_spec("virgil") is None, reason="needs virgil")
DATA2 = pathlib.Path(os.environ.get("CONTEST_DATA", "~/data/imaging_contests")).expanduser() / "2004" / "2004-data2.fits"
needs_data2 = pytest.mark.skipif(not DATA2.exists(), reason="contest data not fetched")
MAS = np.pi / 180 / 3600e3
LAM = 5.5e-7


@pytest.fixture(scope="module")
def cp():
    pytest.importorskip("virgil")
    import virgil.models as vm

    if not hasattr(vm, "EllipticalLimbDarkenedDisk"):
        pytest.skip("needs virgil with EllipticalLimbDarkenedDisk")
    return _load("contest_parametric")


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_gain_system_matches_numpy(cp):
    """V = g (V_star + r V_comp) / (1 + r): two uniform disks, by hand."""
    rng = np.random.default_rng(0)
    u, v = rng.uniform(-30, 30, (2, 12))
    gain, ratio, diam, dra, ddec, cdiam = 1.1, 0.2, 3.0, 4.0, -2.5, 0.8
    rung = cp.spec("ud", comp=True)
    p = {"gain": gain, "diam": diam, "comp_flux": ratio, "comp_dra": dra, "comp_ddec": ddec, "comp_diam": cdiam}
    got = np.asarray(cp.build(p, rung).model(u, v, np.full(u.shape, LAM)))
    q = np.pi * diam * MAS * np.hypot(u, v) / LAM
    star = 2 * j1(q) / q
    qc = np.pi * cdiam * MAS * np.hypot(u, v) / LAM
    comp = 2 * j1(qc) / qc * np.exp(-2j * np.pi * (u * dra + v * ddec) * MAS / LAM)
    want = gain * (star + ratio * comp) / (1 + ratio)
    assert got == pytest.approx(want, abs=1e-9)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_star_frame_position_matches_the_truth_geometry(cp):
    """score_2004's spot (2.5 mas at PA 150°, on a 5.5 mas star with its
    major axis at PA 120° and axis ratio 0.7) sits at the star-frame position
    that ``sky_offset`` maps back to the same sky offset; and the truth's
    normalisation gives the gain the model would need."""
    score = _load("score_2004")
    truth = cp.truth_parameters()
    p = {"diam": 5.5, "ratio": 0.7, "pa": 120.0}
    x, y = (float(t) for t in cp.star_frame(p, truth["spot_dra"], truth["spot_ddec"]))
    assert 1.0 < np.hypot(x, y) < 1.1  # the truth spot is centred on the limb (not clipped there)
    dra, ddec = (float(t) for t in cp.sky_offset(p, x, y))
    assert (dra, ddec) == pytest.approx((truth["spot_dra"], truth["spot_ddec"]), abs=1e-9)
    assert (truth["spot_dra"], truth["spot_ddec"]) == pytest.approx(score.offset(150.0, 2.5), abs=1e-12)
    assert truth["gain"] == pytest.approx(
        score.truth_vis_image(score.FITTED["ratio"], score.FITTED["u_ld"], score.FITTED["spot_contrast"]).sum(), rel=1e-9)


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_render_agrees_with_the_visibilities(cp):
    """The rendered image (East left, North up, on score_2004's grid) has the
    model's visibilities, up to the gain: the spot's and the companion's
    positions and the ellipse's orientation are consistent with the image."""
    score = _load("score_2004")
    rung = cp.spec("eld", 1, True)
    p = {"gain": 1.07, "diam": 5.5, "u": 0.4, "ratio": 0.7, "pa": 120.0, "spot_x": 0.4, "spot_y": -0.3, "spot_diam": 1.0,
         "spot_flux": 0.08, "comp_dra": 10.0, "comp_ddec": 0.0, "comp_flux": 0.11, "comp_diam": 0.5}
    model = cp.build(p, rung)
    image = np.asarray(model.render(cp.SCORE_NPIX, cp.SCORE_FOV))
    assert image.sum() == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    u, v = rng.uniform(-25, 25, (2, 10))  # metres: λ/B well above the 0.08 mas pixels
    direct = score.visibilities(image, u / LAM, v / LAM)
    got = np.asarray(model.model(u, v, np.full(u.shape, LAM))) / p["gain"]
    # disks on a 0.08 mas grid: percent-level
    assert np.abs(direct - got).max() < 0.02


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_starts_are_inside_the_priors(cp):
    import numpyro.distributions as dist

    rung = cp.spec("eld", 1, True)
    h = type("H", (), {"bmin": 1.5, "bmaj": 5.5, "half": 13.0})()
    priors = cp.make_priors(h, rung)
    assert all(isinstance(p, (dist.LogUniform, dist.Uniform)) for p in priors.values())  # Jeffreys
    centre = {k: float(p.high) for k, p in priors.items()}  # every value on a bound
    starts = cp.make_starts([centre], priors, 6, np.random.default_rng(0))
    for s in starts:
        for k, p in priors.items():
            span = float(p.high) - float(p.low) if isinstance(p, dist.Uniform) else np.log(float(p.high) / float(p.low))
            val = s[k] if isinstance(p, dist.Uniform) else np.log(s[k] / float(p.low))
            low = float(p.low) if isinstance(p, dist.Uniform) else 0.0
            assert low + 0.04 * span < val < low + 0.96 * span, k


def _phantom_data(tmp_path):
    from astropy.io import fits  # noqa: F401

    import contest_bench
    from crosscheck import phantoms, sky

    beam = contest_bench.beam_mas([DATA2])
    star = phantoms.sample("spotted_star", np.random.default_rng([0, 0, 0]), beam)  # bench phantom 0
    other = phantoms.sample("spotted_star", np.random.default_rng([0, 0, 1]), beam)  # phantom 1's companion
    params = dict(star, spots=[star["spots"][1]], companion=other["companion"])
    npix, pixel = contest_bench.truth_grid(params, beam)
    cloud = sky.pixel_image(phantoms.render(params, npix, pixel), pixel)
    path = tmp_path / "phantom.fits"
    contest_bench.simulate_file(DATA2, path, cloud, np.random.default_rng(5))
    return path, params


@pytest.mark.slow
@needs_data2
@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_ladder_recovers_a_spotted_star_and_companion(cp, tmp_path):
    from virgil.oidata import OIData

    path, params = _phantom_data(tmp_path)
    data = OIData(str(path))
    h = cp.Hyper(data)
    cp.MAX_AXIS = 81
    records = cp.run_ladder(data, h, n_starts=4, max_steps=None, rng=np.random.default_rng(0), second_spot=False)
    best, _ = cp.choose(records)
    win = records[best]
    assert win["rung"]["comp"] and win["rung"]["n_spots"] == 1, best
    assert win["blocks"]["total"] < 2.0
    sigma = cp.spot_positions_sigma(win, win["sigma"])
    c = params["companion"]
    east, north = c["sep"] * np.sin(np.radians(c["pa"])), c["sep"] * np.cos(np.radians(c["pa"]))
    v = win["values"]
    for fit, true, key in ((v["comp_dra"], east, "comp_dra"), (v["comp_ddec"], north, "comp_ddec")):
        assert abs(fit - true) < max(5 * sigma[key], 0.1), (key, fit, true, sigma[key])
    # the ratio is to the star alone, the phantom's to the total, and the free diameter trades with it
    assert v["comp_flux"] == pytest.approx(c["ratio"], rel=0.35)
    star = params["spots"][0]
    a = 0.5 * params["diam"]
    spot = (star["r"] * a * np.sin(np.radians(star["pa"])), star["r"] * a * np.cos(np.radians(star["pa"])))
    dra, ddec = (float(t) for t in cp.sky_offset(v, v["spot_x"], v["spot_y"]))
    assert np.hypot(dra - spot[0], ddec - spot[1]) < 0.5 * h.bmin
