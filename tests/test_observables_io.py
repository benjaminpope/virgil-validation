"""virgil's OIFITS writer and the extra observables, against our own reading.

* write_oifits: tables written by virgil, read back with astropy (not
  virgil's reader), hold what was given: wavelengths, V², closure phases
  and triple amplitudes, OI_VIS amplitudes and phases with their declared
  AMPTYP/PHITYP, OI_FLUX, station indices, and the OIFITS2 revision. Our
  own reader (crosscheck.chi2.load) reads the file too.
* VisibilityAmplitude and TripleAmplitude (read_oifits extras "visamp",
  "t3amp"): the chi-squared they add to virgil's likelihood equals
  sum ((model - data) / sigma)^2 with |V| and |V_ab V_bc V_ac| computed by
  our own closed-form visibilities.
* continuum_operator: the least-squares fit of a mean, or of a mean and a
  slope in wavenumber, over the continuum channels, evaluated everywhere,
  against our own normal equations.
"""

import numpy as np
import pytest
from astropy.io import fits

from crosscheck import array, chi2 as ours, sky
from evidence.plugin import record

oifits = pytest.importorskip("virgil.oifits")
obs = pytest.importorskip("virgil.observables")
vm = pytest.importorskip("virgil.models")
from virgil.likelihood import whitened_residuals  # noqa: E402
from virgil.oidata import OIData  # noqa: E402

pytestmark = pytest.mark.x64

UTS3 = np.array([[-9.925, -20.335, 0.0], [14.887, 30.502, 0.0], [103.306, 43.999, 0.0]])
WL = np.array([1.6e-6, 1.8e-6, 2.0e-6, 2.2e-6])


def truth(u, v, w):
    return (sky.vis_uniform_disk(u, v, w, 1.2) + 0.05 * sky.vis_point(u, v, w, 6.0, -4.0)) / 1.05


def tables(rng):
    """Two snapshots of three telescopes, every observable, noisy values."""
    rows_v, rows_t = [], []
    for k, ha in enumerate((-1.0, 1.5)):
        u, v = array.snapshot_uv(UTS3, ha, -50.0, -24.6276)
        mjd = 60000.0 + k / 24
        for (i, j), uu, vv in zip(array.baselines(3), u, v):
            rows_v.append((mjd, i, j, uu, vv))
        cp, u1, v1, u2, v2 = array.closure_phase(lambda a, b: truth(a[:, None], b[:, None], WL[None, :]), u, v, 3)
        for t, (a, b, c) in enumerate(array.triangles(3)):
            rows_t.append((mjd, (a, b, c), u1[t], v1[t], u2[t], v2[t], cp[t]))
    n, nw = len(rows_v), WL.size
    U = np.array([r[3] for r in rows_v])
    V = np.array([r[4] for r in rows_v])
    vis = truth(U[:, None], V[:, None], WL[None, :])
    t = len(rows_t)
    u1 = np.array([r[2] for r in rows_t]); v1 = np.array([r[3] for r in rows_t])
    u2 = np.array([r[4] for r in rows_t]); v2 = np.array([r[5] for r in rows_t])
    t3 = (truth(u1[:, None], v1[:, None], WL) * truth(u2[:, None], v2[:, None], WL)
          * np.conj(truth((u1 + u2)[:, None], (v1 + v2)[:, None], WL)))
    sta_v = np.array([[r[1] + 1, r[2] + 1] for r in rows_v])
    sta_t = np.array([[x + 1 for x in r[1]] for r in rows_t])
    mjd_v = np.array([r[0] for r in rows_v]); mjd_t = np.array([r[0] for r in rows_t])
    e = 0.01
    return {
        "OI_WAVELENGTH": {"EFF_WAVE": WL, "EFF_BAND": np.full(nw, 5e-8)},
        "OI_VIS2": {"MJD": mjd_v, "UCOORD": U, "VCOORD": V, "STA_INDEX": sta_v,
                    "VIS2DATA": np.abs(vis) ** 2 + e * rng.normal(size=(n, nw)), "VIS2ERR": np.full((n, nw), e)},
        "OI_VIS": {"MJD": mjd_v, "UCOORD": U, "VCOORD": V, "STA_INDEX": sta_v, "AMPTYP": "absolute",
                   "PHITYP": "differential",
                   "VISAMP": np.abs(vis) + e * rng.normal(size=(n, nw)), "VISAMPERR": np.full((n, nw), e),
                   "VISPHI": np.rad2deg(np.angle(vis)), "VISPHIERR": np.full((n, nw), 1.0)},
        "OI_T3": {"MJD": mjd_t, "U1COORD": u1, "V1COORD": v1, "U2COORD": u2, "V2COORD": v2, "STA_INDEX": sta_t,
                  "T3PHI": np.rad2deg(np.angle(t3)) + rng.normal(size=(t, nw)), "T3PHIERR": np.ones((t, nw)),
                  "T3AMP": np.abs(t3) + e * rng.normal(size=(t, nw)), "T3AMPERR": np.full((t, nw), e)},
        "OI_FLUX": {"MJD": mjd_v[:3], "STA_INDEX": np.array([[1], [2], [3]]),
                    "FLUXDATA": 1.0 + 0.01 * rng.normal(size=(3, nw)), "FLUXERR": np.full((3, nw), 0.01)},
        "info": {"OBJECT": "test", "INSNAME": "SIM", "ARRNAME": "VLTI", "STAXY": UTS3[:, :2]},
    }


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    t = tables(np.random.default_rng(1))
    path = oifits.write_oifits(t, tmp_path_factory.mktemp("io") / "all.fits")
    return t, path


@pytest.mark.validates("virgil.oifits.write_oifits", roots=["standards"])
def test_write_oifits_round_trip(written):
    t, path = written
    with fits.open(path) as h:
        names = [x.name for x in h]
        for ext in ("OI_WAVELENGTH", "OI_VIS2", "OI_VIS", "OI_T3", "OI_FLUX", "OI_TARGET", "OI_ARRAY"):
            assert ext in names, ext
            if ext not in ("OI_WAVELENGTH",):
                assert int(h[ext].header["OI_REVN"]) == 2, ext
        np.testing.assert_allclose(h["OI_WAVELENGTH"].data["EFF_WAVE"], WL, rtol=1e-7)  # float32 column
        for ext, cols in (("OI_VIS2", ("VIS2DATA", "VIS2ERR", "UCOORD", "VCOORD")),
                          ("OI_VIS", ("VISAMP", "VISAMPERR", "VISPHI")),
                          ("OI_T3", ("T3PHI", "T3PHIERR", "T3AMP", "U1COORD", "V2COORD")),
                          ("OI_FLUX", ("FLUXDATA", "FLUXERR"))):
            for c in cols:
                np.testing.assert_allclose(np.asarray(h[ext].data[c], float), np.asarray(t[ext][c], float).reshape(h[ext].data[c].shape),
                                           rtol=1e-12, err_msg=f"{ext}.{c}")
        np.testing.assert_array_equal(h["OI_VIS2"].data["STA_INDEX"], t["OI_VIS2"]["STA_INDEX"])
        np.testing.assert_array_equal(h["OI_T3"].data["STA_INDEX"], t["OI_T3"]["STA_INDEX"])
        assert h["OI_VIS"].header["AMPTYP"].strip() == "absolute"
        assert h["OI_VIS"].header["PHITYP"].strip() == "differential"
        assert not np.any(h["OI_VIS2"].data["FLAG"])
    d = ours.load(path)
    np.testing.assert_allclose(d["v2"], t["OI_VIS2"]["VIS2DATA"], rtol=1e-12)
    np.testing.assert_allclose(np.rad2deg(d["cp"]), t["OI_T3"]["T3PHI"], rtol=1e-12)


def chi2(model, data):
    return float(np.sum(np.asarray(whitened_residuals(model, data)) ** 2))


MODEL = vm.System(star=vm.UniformDisk(1.15), comp=vm.PointSource(0.045, 6.1, -3.9))


def model_vis(u, v, w):
    return (sky.vis_uniform_disk(u, v, w, 1.15) + 0.045 * sky.vis_point(u, v, w, 6.1, -3.9)) / 1.045


@pytest.mark.validates("virgil.observables.VisibilityAmplitude", "virgil.oifits.read_oifits", roots=["mathematics", "standards"])
def test_visibility_amplitudes_add_their_chi2(written):
    t, path = written
    base = chi2(MODEL, OIData(str(path)))
    extra = chi2(MODEL, OIData(str(path), extras=("visamp",))) - base
    o = t["OI_VIS"]
    want = np.sum(((np.abs(model_vis(o["UCOORD"][:, None], o["VCOORD"][:, None], WL)) - o["VISAMP"]) / o["VISAMPERR"]) ** 2)
    record("rel_chi2_visamp", abs(extra / want - 1))
    assert abs(extra / want - 1) < 1e-6  # the file stores EFF_WAVE in float32


@pytest.mark.validates("virgil.observables.TripleAmplitude", "virgil.oifits.read_oifits", roots=["mathematics", "standards"])
def test_triple_amplitudes_add_their_chi2(written):
    t, path = written
    base = chi2(MODEL, OIData(str(path)))
    extra = chi2(MODEL, OIData(str(path), extras=("t3amp",))) - base
    o = t["OI_T3"]
    u1, v1, u2, v2 = (o[k][:, None] for k in ("U1COORD", "V1COORD", "U2COORD", "V2COORD"))
    amp = np.abs(model_vis(u1, v1, WL) * model_vis(u2, v2, WL) * np.conj(model_vis(u1 + u2, v1 + v2, WL)))
    want = np.sum(((amp - o["T3AMP"]) / o["T3AMPERR"]) ** 2)
    record("rel_chi2_t3amp", abs(extra / want - 1))
    assert abs(extra / want - 1) < 1e-6


@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.validates("virgil.observables.continuum_operator", roots=["mathematics"])
def test_continuum_operator_is_the_least_squares_fit(order):
    wl = np.linspace(2.10e-6, 2.20e-6, 40)
    cont = (wl < 2.155e-6) | (wl > 2.175e-6)
    L = np.asarray(obs.continuum_operator(wl, cont, order=order))
    X = np.vander(1 / wl, order + 1, increasing=True)  # 1, 1/lambda
    Xc = X[cont]
    want = X @ np.linalg.solve(Xc.T @ Xc, Xc.T)  # fit on continuum channels, evaluate everywhere
    full = np.zeros((wl.size, wl.size))
    full[:, cont] = want
    np.testing.assert_allclose(L, full, atol=1e-10)
    # a polynomial of that order in 1/lambda is fitted exactly
    x = X @ np.arange(1, order + 2)
    np.testing.assert_allclose(L @ x, x, rtol=1e-10)
