"""Synthetic 4-telescope PIONIER-like data for the tests of ``scripts/postagb_parametric.py``: no file under ~/data is read."""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

STATIONS = np.array([[0.0, 0.0], [32.0, 10.0], [-14.0, 46.0], [58.0, -22.0]])  # metres
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
TRIANGLES = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
WAVELS = np.array([1.55e-6, 1.65e-6, 1.75e-6])

CFG = dict(key="SYN", ud_fix=0.001, img_star_index=-3.0, bg=True, sec0=None, f_sec0=None)  # the keys of P.sparco_scene and P.scene_priors that matter here
REFS = dict(deprins_image=(8.0, 45.0, 60.0))
FLUXES = (0.5, 0.0)


def layout(nights=(0, 1, 2), frames=4):
    """The sample layout: per frame f, baseline b, channel c the index ((f * 6) + b) * n_ch + c. Returns the arrays shared by every dataset."""
    u, v, wl, mjd, frame, st = [], [], [], [], [], []
    f = 0
    for night in nights:
        for k in range(frames):
            ha = np.radians(-50.0 + 100.0 * k / max(frames - 1, 1) + 7.0 * night)  # the baselines rotate with hour angle
            c, s = np.cos(ha), np.sin(ha)
            for (i, j) in PAIRS:
                d = STATIONS[j] - STATIONS[i]
                for w in WAVELS:
                    u.append(c * d[0] - s * d[1])
                    v.append(s * d[0] + c * d[1])
                    wl.append(w)
                    mjd.append(60000.9 + night + 0.01 * k)  # after 12 UT (MJD .5): night label = floor(mjd - 0.5) = 60000 + night
                    frame.append(f)
                    st.append((i, j))
            f += 1
    n_ch, n_b = len(WAVELS), len(PAIRS)
    i1, i2, i3 = [], [], []
    for fr in range(f):
        for (a, b, c) in TRIANGLES:
            for ch in range(n_ch):
                base = lambda p, q: ((fr * n_b) + PAIRS.index((p, q))) * n_ch + ch  # noqa: E731
                i1.append(base(a, b))
                i2.append(base(b, c))
                i3.append(base(a, c))
    return dict(u=np.array(u), v=np.array(v), wavel=np.array(wl), mjd=np.array(mjd), frame=np.array(frame), stations=np.array(st), i_cps1=np.array(i1),
                i_cps2=np.array(i2), i_cps3=np.array(i3))


def make_data(model=None, nights=(0, 1, 2), frames=4, sigma_v2=0.01, sigma_cp_deg=1.0, seed=0, noise=True):
    """An ``OIData`` of the layout above: V^2 and closure phases of ``model`` (a virgil scene; unity visibilities without one) with Gaussian noise."""
    from virgil import OIData  # noqa: PLC0415

    import postagb_imaging as P  # noqa: PLC0415

    lay = layout(nights, frames)
    n, nt = lay["u"].size, lay["i_cps1"].size
    raw = dict(lay, vis=np.ones(n), d_vis=np.full(n, sigma_v2), phi=np.zeros(nt), d_phi=np.full(nt, np.radians(sigma_cp_deg)))
    d0 = OIData(raw)
    if model is None:
        return d0
    v2, t3 = P.observables(model, d0)
    rng = np.random.default_rng(seed)
    e_v, e_t = (rng.standard_normal(n), rng.standard_normal(nt)) if noise else (np.zeros(n), np.zeros(nt))
    raw["vis"] = np.asarray(v2) + sigma_v2 * e_v
    raw["phi"] = np.radians(np.asarray(t3)) + np.radians(sigma_cp_deg) * e_t
    return OIData(raw)


def start(**kw):
    """A start dict (see ``postagb_parametric.starts_for``) with the truth of the ring tests unless overridden."""
    st = dict(name="truth", inc=50.0, pa=40.0, r=4.0, sigma=1.0, re=[0.5], im=[-0.4], dra=0.0, ddec=0.0, f_env=0.5, d_env=0.0, f_bg=0.1)
    return st | kw
