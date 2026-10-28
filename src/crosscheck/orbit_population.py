"""Population draws and simulator for the 0.4.2 orbit gate.

Everything here is numpy/scipy only and independent of virgil (the gate's
section 2): the populations of design/orbit_gate_042.md section 3 are drawn
from their registered seeds, written to OIFITS with our own array geometry,
Kepler solver (``crosscheck.orbits``) and visibility formulae
(``crosscheck.sky``), and the truth is a plain dict, saved before any fit.

A system is a dict (JSON-serializable): ``draw_system(stratum, index)`` makes
it, ``write_system(system, directory)`` writes one OIFITS file per night.
Section numbers in comments refer to the gate file.
"""

import math

import numpy as np
from scipy.special import j1

from . import array, oifits_writer, orbits, sky

MASTER_SEED = 20261028  # gate section 3, and the criteria block
CODES = {"A": 1, "B": 2, "N": 3, "D": 4}  # gate section 3
MAS = 180.0 / np.pi * 3600.0 * 1000.0  # radians to milliarcseconds
SIDEREAL = 1.00273790935  # sidereal days per solar day: hour angle h advances 24 h per 0.99727 d

# Station tables, metres East, North (up omitted, as detection_campaign.UTS4).
# T1: VLTI UT1-UT4 (the ESO station table, as in scripts/detection_campaign.py).
# T2: A0, G1, J2, K0 from the OI_ARRAY of an ESO GRAVITY file, whose STAXYZ is
#     minus (East, North), the sign that reproduces the UT table above.
# T3: CHARA S1, S2, E1, E2, W1, W2 from the OI_ARRAY of an Aspro2 file,
#     (x, y, z) -> (East = y, North = -sin(lat) x + cos(lat) z) at the array
#     latitude 34.2244 deg; East and North agree with ten Brummelaar et al.
#     2005, ApJ 628, 453, Table 1 (E1 = 125.333, 305.932) to 0.002 m.
UT = [[-9.925, -20.335], [14.887, 30.502], [44.915, 66.183], [103.306, 43.999]]
AT = [[-14.619, -55.781], [66.716, -95.485], [114.460, -62.151], [106.410, -14.172]]
CHARA = [[0.0, 0.0], [-5.747, 33.582], [125.332, 305.934], [70.395, 269.715],
         [-175.072, 216.321], [-69.091, 199.335]]

TEMPLATES = {
    "T1": dict(stations=UT, latitude=-24.6276, wavelengths=(2.00e-6, 2.40e-6), n_channels=12, dec=(-70.0, 20.0)),
    "T2": dict(stations=AT, latitude=-24.6276, wavelengths=(1.53e-6, 1.77e-6), n_channels=6, dec=(-70.0, 20.0)),
    "T3": dict(stations=CHARA, latitude=34.2244, wavelengths=(1.50e-6, 1.75e-6), n_channels=8, dec=(-20.0, 80.0)),
}
TEMPLATE_NAMES = tuple(TEMPLATES)

# Sampling limit of virgil's documented `period_window` (its docstring): the
# companion may move a fraction 0.5 of the resolution in a night (the gate's
# "more than half a resolution element", section 3.3), the period is at least
# one night and at least the documented floor of 0.2 d.
MOTION_FRACTION = 0.5
P_FLOOR = 0.2

# Stratum A prior, gate section 3.2 (the engine's stated prior, passed explicitly).
A_NIGHTS = (4, 5, 6, 8, 10)
A_SEASON_DAYS = 120.0
A_NIGHT_HOURS = (1.0, 4.0)
A_ECC_MAX = 0.9
A_A_RANGE_RES = (0.3, 10.0)
A_FLUX_RANGE = (0.01, 1.0)
SIGMA_V2 = (0.005, 0.05)
SIGMA_CP_DEG = (0.2, 3.0)

# Stratum B and N, gate sections 3.3 and 3.4.
B_NIGHTS = (4, 6, 8, 12)
B_ECC_MAX = 0.8
B_A_RES = (0.5, 10.0)
B_A_MAS_DEFAULT = (1.0, 30.0)  # the engine's default a_range today (gate section 3.3)
B_DMAG = 4.5
TWIN_PROBABILITY, TWIN_DMAG = 0.1, 0.1
POINT_PROBABILITY = 0.7
DISC_RES = (0.2, 0.8)
C_CHOICES = (0.0, 0.3, 0.6)
BAD_NIGHT = dict(probability=0.2, probability_null=0.4, v2_factor=(0.6, 0.9), cp_noise_factor=4.0)
SIGN_ERROR_PROBABILITY = 0.1
MISSPEC_PROBABILITY = 0.3
MISSPEC_SEP_RES = (2.0, 6.0)
MISSPEC_FLUX = (0.02, 0.08)
N_DISC_FIRST = 30  # the first indices of N carry a disc primary
MISSPEC_NULL_PROBABILITY = 0.3
ENVELOPE_FLUX = (0.01, 0.5)
ENVELOPE_SIZE_RES = (0.3, 10.0)
RING_WIDTH = 0.2
LINE_ENHANCEMENT = (1.5, 10.0)
LINE_SIZE_FACTOR = (0.5, 3.0)
ELONGATED_MAJOR_RES = (0.3, 0.8)
ELONGATED_RATIO = (0.6, 0.9)
BACKGROUND_FLUX = (0.005, 0.05)
BACKGROUND_SEP_RES = (2.0, 6.0)
S1_DAYS, S2_DAYS, S2_OFFSET, S3_DENSE_DAYS, S3_SPARSE_DAYS = 90.0, 60.0, 365.0, 10.0, 730.0
MJD0 = 60000.0
HOUR_ANGLES = (-3.0, 3.0)
SNAPSHOTS = (4, 8)
DEV_B = 60  # D: indices below are drawn as B, the rest as N (gate section 3.5)


def system_streams(stratum, index):
    """(rng, noise_rng) of system ``index``: gate section 3's
    ``default_rng(SeedSequence([20261028, code, index]))`` for the draws, and
    its first spawned child for the noise realization (a separate stream, so
    the noise never shifts the draws)."""
    ss = np.random.SeedSequence([MASTER_SEED, CODES[stratum], int(index)])
    return np.random.default_rng(ss), np.random.default_rng(ss.spawn(1)[0])


def template_geometry(name):
    """Stations, wavelengths, longest baseline (m) and resolution (mas)."""
    t = TEMPLATES[name]
    st = np.asarray(t["stations"], float)
    wl = np.linspace(*t["wavelengths"], t["n_channels"])
    b_max = max(np.hypot(*(st[i] - st[j])) for i, j in array.baselines(len(st)))
    return st, wl, float(b_max), float(wl.min() / b_max * MAS)


def sampling_limit(a_mas, night_days, res_mas):
    """Shortest period (d) that moves a companion at ``a_mas`` less than
    ``MOTION_FRACTION`` of ``res_mas`` in ``night_days``; virgil's documented
    `period_window` lower end, 2 pi a L / (f res), at least a night and ``P_FLOOR``."""
    return max(2.0 * math.pi * a_mas * night_days / (MOTION_FRACTION * res_mas), night_days, P_FLOOR)


def band_of(period, span):
    """The alias band N = round(T/P) of a period, T the span (the definition
    of virgil's documentation: band N holds T/(N + 1/2) <= P <= T/(N - 1/2);
    N = 0 above 2T)."""
    return int(math.floor(span / period + 0.5))


def _loguniform(rng, lo, hi, size=None):
    return np.exp(rng.uniform(math.log(lo), math.log(hi), size))


def _nights_a(rng, k):
    times = np.sort(rng.uniform(0.0, A_SEASON_DAYS, k))
    return times


def _nights_b(rng, schedule, k):
    if schedule == "S1":
        return np.sort(rng.uniform(0.0, S1_DAYS, k))
    if schedule == "S2":
        k1 = (k + 1) // 2
        return np.sort(np.r_[rng.uniform(0.0, S2_DAYS, k1), S2_OFFSET + rng.uniform(0.0, S2_DAYS, k - k1)])
    k1 = k // 2
    return np.sort(np.r_[rng.uniform(0.0, S3_DENSE_DAYS, k1), rng.uniform(0.0, S3_SPARSE_DAYS, k - k1)])


def _snapshots(rng, times, window):
    """Per night: hour angles (sorted) and snapshot MJDs. ``window`` True is
    stratum A (a night of length uniform on 1-4 h inside +-3 h); False draws
    hour angles uniformly on +-3 h (gate section 3.1)."""
    nights = []
    n_snap = rng.integers(SNAPSHOTS[0], SNAPSHOTS[1] + 1, len(times))
    lengths = rng.uniform(*A_NIGHT_HOURS, len(times)) if window else None
    starts = rng.uniform(0.0, 1.0, len(times))
    for k, (t, n) in enumerate(zip(times, n_snap)):
        if window:
            lo = HOUR_ANGLES[0] + starts[k] * (HOUR_ANGLES[1] - HOUR_ANGLES[0] - lengths[k])
            ha = np.sort(rng.uniform(lo, lo + lengths[k], n))
        else:
            ha = np.sort(rng.uniform(*HOUR_ANGLES, n))
        mjd = MJD0 + t + (ha - ha.mean()) / (24.0 * SIDEREAL)
        nights.append(dict(mjd_mean=float(MJD0 + t), ha_h=ha.tolist(), mjd=mjd.tolist(),
                           bad=False, sign_error=False, v2_factor=1.0))
    return nights


def _night_length(nights):
    return max(max(n["mjd"]) - min(n["mjd"]) for n in nights)


def _orbit(rng, elements):
    """Stratum B's draws of cos i, omega, Omega and tau (one call each)."""
    return dict(cos_inc=float(rng.uniform(-1.0, 1.0)), omega=float(rng.uniform(0.0, 360.0)),
                Omega=float(rng.uniform(0.0, 180.0)), tau=float(rng.uniform(0.0, 1.0)), **elements)


def _noise(rng, correlated):
    return dict(sigma_v2=float(_loguniform(rng, *SIGMA_V2)), sigma_cp_deg=float(_loguniform(rng, *SIGMA_CP_DEG)),
                c=float(rng.choice(C_CHOICES)) if correlated else 0.0)


def _contaminants(rng, nights, null):
    """Bad night, sign error (B and N) with every quantity always drawn."""
    k = len(nights)
    bad_p = BAD_NIGHT["probability_null"] if null else BAD_NIGHT["probability"]
    bad, bad_night = rng.uniform() < bad_p, int(rng.integers(1, k))
    factor = float(rng.uniform(*BAD_NIGHT["v2_factor"]))
    sign, sign_night = rng.uniform() < SIGN_ERROR_PROBABILITY, int(rng.integers(1, k))
    if bad:
        nights[bad_night].update(bad=True, v2_factor=factor)
    if sign:
        nights[sign_night]["sign_error"] = True
    return dict(bad_night=bad_night if bad else None, v2_factor=factor if bad else None,
                sign_error_night=sign_night if sign else None)


def _background(rng, res):
    flux = _loguniform(rng, *BACKGROUND_FLUX)
    sep, pa = rng.uniform(*BACKGROUND_SEP_RES) * res, rng.uniform(0.0, 360.0)
    return dict(kind="background", flux=float(flux), dra=float(sep * math.sin(math.radians(pa))),
                ddec=float(sep * math.cos(math.radians(pa))))


def _misspecified_null(rng, res, n_channels):
    """Gate section 3.4's single-but-not-simple star; every quantity is drawn
    whatever the kind, so the stream does not depend on it."""
    flag = bool(rng.uniform() < MISSPEC_NULL_PROBABILITY)
    kind = ("envelope", "elongated", "background")[int(rng.integers(0, 3))]
    shape = "gaussian" if rng.uniform() < 0.5 else "ring"
    fraction = float(_loguniform(rng, *ENVELOPE_FLUX))
    size = float(_loguniform(rng, *ENVELOPE_SIZE_RES) * res)
    width = int(rng.integers(1, math.ceil(n_channels / 2) + 1))
    start = int(rng.integers(0, n_channels - width + 1))
    enhancement = float(_loguniform(rng, *LINE_ENHANCEMENT))
    resize = bool(rng.uniform() < 0.5)
    resize_factor = float(_loguniform(rng, *LINE_SIZE_FACTOR))
    major = float(rng.uniform(*ELONGATED_MAJOR_RES) * res)
    ratio, pa = float(rng.uniform(*ELONGATED_RATIO)), float(rng.uniform(0.0, 360.0))
    bg = _background(rng, res)
    if not flag:
        return None
    if kind == "envelope":
        return dict(kind=kind, shape=shape, fraction=fraction, size=size, line=[start, start + width],
                    enhancement=enhancement, resize=resize_factor if resize else 1.0)
    if kind == "elongated":
        return dict(kind=kind, major=major, ratio=ratio, pa=pa)
    return bg


def draw_system(stratum, index):
    """The truth and simulation recipe of system ``index`` of ``stratum``
    ('A', 'B', 'N' or 'D'), drawn in the gate's order."""
    rng, _ = system_streams(stratum, index)
    kind = stratum if stratum != "D" else ("B" if index < DEV_B else "N")
    template = TEMPLATE_NAMES[int(rng.integers(0, len(TEMPLATE_NAMES)))]
    t = TEMPLATES[template]
    sin_dec = rng.uniform(math.sin(math.radians(t["dec"][0])), math.sin(math.radians(t["dec"][1])))
    _, _, b_max, res = template_geometry(template)
    system = dict(stratum=stratum, index=int(index), kind=kind, template=template,
                  dec_deg=math.degrees(math.asin(sin_dec)), latitude_deg=t["latitude"], res_mas=res, b_max_m=b_max)
    if kind == "A":
        _draw_a(rng, system)
    elif kind == "B":
        _draw_b(rng, system)
    else:
        _draw_n(rng, system)
    return system


def _draw_a(rng, s):
    res = s["res_mas"]
    k = int(rng.choice(A_NIGHTS))
    times = _nights_a(rng, k)
    nights = _snapshots(rng, times, window=True)
    mean_times = np.array([n["mjd_mean"] for n in nights])
    span = float(np.ptp(mean_times))
    a_range = (A_A_RANGE_RES[0] * res, A_A_RANGE_RES[1] * res)
    p_range = (sampling_limit(a_range[1], _night_length(nights), res), 2.0 * span)
    period = float(_loguniform(rng, *p_range))
    tau = float(rng.uniform(0.0, 1.0))  # gate order: log P, tau, e, cos i, omega, Omega, log a, log flux
    ecc = float(rng.uniform(0.0, A_ECC_MAX))
    orbit = dict(period=period, ecc=ecc, tau=tau, cos_inc=float(rng.uniform(-1.0, 1.0)),
                 omega=float(rng.uniform(0.0, 360.0)), Omega=float(rng.uniform(0.0, 180.0)))
    orbit["a_mas"] = float(_loguniform(rng, *a_range))
    orbit["flux"] = float(_loguniform(rng, *A_FLUX_RANGE))
    s.update(nights=nights, noise=_noise(rng, correlated=False), orbit=_finish_orbit(orbit, mean_times),
             primary=dict(diam_mas=0.0, disc_on="none"), contaminants={}, misspecified=False,
             n_redraws=0, span_days=span, p_range=list(p_range), a_range=list(a_range),
             flux_range=list(A_FLUX_RANGE), ecc_max=A_ECC_MAX, band=band_of(period, span))


def _finish_orbit(orbit, mean_times):
    orbit["inc_deg"] = math.degrees(math.acos(orbit["cos_inc"]))
    orbit["t_peri"] = float(mean_times.mean() + orbit["tau"] * orbit["period"])
    return orbit


def _draw_b(rng, s):
    res = s["res_mas"]
    schedule = ("S1", "S2", "S3")[int(rng.integers(0, 3))]
    k = int(rng.choice(B_NIGHTS))
    times = _nights_b(rng, schedule, k)
    nights = _snapshots(rng, times, window=False)
    mean_times = np.array([n["mjd_mean"] for n in nights])
    span = float(np.ptp(mean_times))
    longest = _night_length(nights)
    redraws = 0
    while True:  # gate section 3.3: redraw, counting, an orbit outside the stated domain
        period = float(_loguniform(rng, 1.0, 3.0 * span))
        ecc = float(rng.uniform(0.0, B_ECC_MAX))
        orbit = _orbit(rng, dict(period=period, ecc=ecc))
        a_mas = float(_loguniform(rng, *B_A_RES) * res)
        if period >= sampling_limit(a_mas, longest, res) and B_A_MAS_DEFAULT[0] <= a_mas <= B_A_MAS_DEFAULT[1]:
            break
        redraws += 1
    orbit["a_mas"] = a_mas
    dmag, twin, twin_dmag = float(rng.uniform(0.0, B_DMAG)), rng.uniform() < TWIN_PROBABILITY, float(rng.uniform(0.0, TWIN_DMAG))
    orbit["flux"] = float(10.0 ** (-0.4 * (twin_dmag if twin else dmag)))
    orbit["near_twin"] = bool(twin)
    point, diam, on_fainter = rng.uniform() < POINT_PROBABILITY, float(rng.uniform(*DISC_RES) * res), rng.uniform() < 0.5
    primary = dict(diam_mas=0.0 if point else diam, disc_on="none" if point else ("fainter" if on_fainter else "brighter"))
    noise = _noise(rng, correlated=True)
    contaminants = _contaminants(rng, nights, null=False)
    miss, sep, pa, flux = (rng.uniform() < MISSPEC_PROBABILITY, rng.uniform(*MISSPEC_SEP_RES) * res,
                           rng.uniform(0.0, 360.0), rng.uniform(*MISSPEC_FLUX))
    third = dict(kind="third_source", flux=float(flux), dra=float(sep * math.sin(math.radians(pa))),
                 ddec=float(sep * math.cos(math.radians(pa))))
    s.update(nights=nights, noise=noise, orbit=_finish_orbit(orbit, mean_times), primary=primary,
             contaminants=contaminants, misspecified=bool(miss), misspecification=third if miss else None,
             schedule=schedule, n_redraws=redraws, span_days=span, band=band_of(period, span))


def _draw_n(rng, s):
    res = s["res_mas"]
    schedule = ("S1", "S2", "S3")[int(rng.integers(0, 3))]
    k = int(rng.choice(B_NIGHTS))
    nights = _snapshots(rng, _nights_b(rng, schedule, k), window=False)
    span = float(np.ptp([n["mjd_mean"] for n in nights]))
    diam = float(rng.uniform(*DISC_RES) * res)
    disc = s["index"] < N_DISC_FIRST
    noise = _noise(rng, correlated=True)
    contaminants = _contaminants(rng, nights, null=True)
    misspec = _misspecified_null(rng, res, TEMPLATES[s["template"]]["n_channels"])
    s.update(nights=nights, noise=noise, orbit=None, primary=dict(diam_mas=diam if disc else 0.0, disc_on="single" if disc else "none"),
             contaminants=contaminants, misspecified=misspec is not None, misspecification=misspec,
             schedule=schedule, n_redraws=0, span_days=span, band=None)


# --------------------------------------------------------------- scenes


def vis_elliptical_disk(u, v, wavel, major, ratio, pa):
    """Uniform ellipse (major-axis diameter ``major`` mas, minor/major
    ``ratio``, major axis at position angle ``pa`` deg, North through East).
    An ellipse is the linear image of a disc, so its visibility is the
    circular one at the transformed frequency (Bracewell, The Fourier
    Transform and Its Applications, similarity theorem)."""
    fu, fv = u / wavel * MAS, v / wavel * MAS
    p = math.radians(pa)
    q_major, q_minor = fu * math.sin(p) + fv * math.cos(p), fu * math.cos(p) - fv * math.sin(p)
    x = np.pi * major * np.hypot(q_major, ratio * q_minor)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(x > 0, 2.0 * j1(x) / np.where(x > 0, x, 1.0), 1.0) + 0j


def _star(primary_diam):
    if primary_diam > 0.0:
        return lambda u, v, w, dra=0.0, ddec=0.0: sky.vis_uniform_disk(u, v, w, primary_diam, dra, ddec)
    return lambda u, v, w, dra=0.0, ddec=0.0: sky.vis_point(u, v, w, dra, ddec)


def companion_position(system, t):
    """(dRA, dDec) in mas of the companion at MJD ``t`` (array)."""
    o = system["orbit"]
    return orbits.position(t, o["period"], o["t_peri"], o["ecc"], o["inc_deg"], o["omega"], o["Omega"], o["a_mas"])


def _envelope_parts(system, wl):
    """Per-channel (envelope flux ratio, size factor) of a chromatic envelope."""
    m = system["misspecification"]
    n = len(wl)
    ratio = np.full(n, m["fraction"] / (1.0 - m["fraction"]))
    size = np.ones(n)
    lo, hi = m["line"]
    ratio[lo:hi] *= m["enhancement"]
    size[lo:hi] = m["resize"]
    return ratio, size


def scene(system, snapshot=None, companion=None):
    """``vis(u, v, wavel)`` of the system's scene (normalized by its total
    flux). The star is flux 1, the companion ``orbit.flux`` at ``companion``
    ((dRA, dDec) mas); contaminants as the system records them."""
    primary = system["primary"]
    wl_grid = template_geometry(system["template"])[1]
    star = _star(primary["diam_mas"])
    parts = []  # (flux, callable)
    m = system.get("misspecification")
    kind = m["kind"] if m else None
    if system["orbit"] is not None:
        f = system["orbit"]["flux"]
        on_companion = primary["disc_on"] == "fainter"
        disc = _star(primary["diam_mas"])
        parts.append((1.0, (lambda u, v, w: sky.vis_point(u, v, w)) if on_companion else (lambda u, v, w: star(u, v, w))))
        dra, ddec = companion
        parts.append((f, (lambda u, v, w: disc(u, v, w, dra, ddec)) if on_companion
                      else (lambda u, v, w: sky.vis_point(u, v, w, dra, ddec))))
    elif kind == "elongated":
        parts.append((1.0, lambda u, v, w: vis_elliptical_disk(u, v, w, m["major"], m["ratio"], m["pa"])))
    else:
        parts.append((1.0, lambda u, v, w: star(u, v, w)))
    if m and kind in ("third_source", "background"):
        parts.append((m["flux"], lambda u, v, w: sky.vis_point(u, v, w, m["dra"], m["ddec"])))

    if kind == "envelope":
        ratio, size_factor = _envelope_parts(system, wl_grid)

        def envelope(u, v, w):
            ch = np.argmin(np.abs(np.asarray(w)[..., None] - wl_grid), axis=-1)
            r, sz = ratio[ch], m["size"] * size_factor[ch]
            if m["shape"] == "gaussian":
                env = sky.vis_gaussian(u, v, w, sz / sky.FWHM_PER_SIGMA)
            else:
                env = sky.vis_annulus(u, v, w, (1.0 - RING_WIDTH) * sz, sz)
            return r, env

        def total(u, v, w):
            base = sum(fl * fn(u, v, w) for fl, fn in parts)
            flux = sum(fl for fl, _ in parts)
            r, env = envelope(u, v, w)
            return (base + flux * r * env) / (flux * (1.0 + r))

        return total
    flux = sum(fl for fl, _ in parts)
    return lambda u, v, w: sum(fl * fn(u, v, w) for fl, fn in parts) / flux


# --------------------------------------------------------------- simulator


def _write_night(path, system, k, night, rng):
    """One night's OIFITS: V^2 with a channel-common multiplicative term of
    variance fraction c, closure phases from per-baseline phase noise
    (``phase_noise='baseline'`` of crosscheck.simulate.observe), and the night's
    contaminants (a bad night's V^2 factor and 4x closure-phase noise, a
    closure-phase sign error)."""
    stations, wl, _, _ = template_geometry(system["template"])
    wl = wl.astype(np.float32).astype(float)  # the stored precision, as simulate.observe
    n_tel = len(stations)
    pairs = np.array(array.baselines(n_tel))
    tris = np.array(array.triangles(n_tel))
    index = {tuple(p): i for i, p in enumerate(pairs)}
    ab = np.array([index[(a, b)] for a, b, c in tris])
    bc = np.array([index[(b, c)] for a, b, c in tris])
    ac = np.array([index[(a, c)] for a, b, c in tris])
    sv, scp, c = system["noise"]["sigma_v2"], system["noise"]["sigma_cp_deg"], system["noise"]["c"]
    cp_noise = scp * (BAD_NIGHT["cp_noise_factor"] if night["bad"] else 1.0)
    rec = {k_: [] for k_ in ("vis2", "u", "v", "sta", "mjd")}
    t3 = {k_: [] for k_ in ("phi", "u1", "v1", "u2", "v2", "sta", "mjd")}
    for ha, mjd in zip(night["ha_h"], night["mjd"]):
        u, v = array.snapshot_uv(stations, ha, system["dec_deg"], system["latitude_deg"])
        comp = None
        if system["orbit"] is not None:
            dra, ddec = companion_position(system, np.array([mjd]))
            comp = (float(dra[0]), float(ddec[0]))
        vis = scene(system, companion=comp)(u[:, None], v[:, None], wl[None, :])
        v2 = night["v2_factor"] * np.abs(vis) ** 2
        rms = np.sqrt(np.mean(v2**2, axis=1, keepdims=True))
        gain = np.where(rms > 0, math.sqrt(c) * sv / np.where(rms > 0, rms, 1.0), 0.0) * rng.standard_normal((len(u), 1))
        v2_obs = v2 * (1.0 + gain) + math.sqrt(1.0 - c) * sv * rng.standard_normal(v2.shape)
        phase = np.exp(1j * np.deg2rad(cp_noise / math.sqrt(3.0)) * rng.standard_normal(vis.shape))
        noisy = vis * phase
        cp = np.angle(noisy[ab] * noisy[bc] * np.conj(noisy[ac]))
        cp = np.rad2deg(cp) * (-1.0 if night["sign_error"] else 1.0)
        rec["vis2"].append(v2_obs)
        rec["u"].append(u)
        rec["v"].append(v)
        rec["sta"].append(pairs)
        rec["mjd"].append(np.full(len(u), mjd))
        t3["phi"].append(cp)
        t3["u1"].append(u[ab])
        t3["v1"].append(v[ab])
        t3["u2"].append(u[bc])
        t3["v2"].append(v[bc])
        t3["sta"].append(tris)
        t3["mjd"].append(np.full(len(ab), mjd))
    v2d = {k_: np.concatenate(x) for k_, x in rec.items()}
    v2d["err"] = np.full(v2d["vis2"].shape, max(sv, 1e-6))
    t3d = {k_: np.concatenate(x) for k_, x in t3.items()}
    t3d["err"] = np.full(t3d["phi"].shape, max(scp, 1e-6))
    oifits_writer.write(path, wavelengths=wl, stations_xyz=np.column_stack([stations, np.zeros(n_tel)]),
                        vis2=v2d, t3=t3d, mjd=float(night["mjd_mean"]))


def write_system(system, directory):
    """Write ``night_<k>.fits`` for every night; returns the paths. The noise
    stream is the system's own (``system_streams``), so the files are a
    function of (stratum, index) alone."""
    import pathlib

    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _, rng = system_streams(system["stratum"], system["index"])
    paths = []
    for k, night in enumerate(system["nights"]):
        path = directory / f"night_{k:02d}.fits"
        _write_night(path, system, k, night, rng)
        paths.append(path)
    return paths
