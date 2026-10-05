"""Grey-scaled spectra (FluxSpectrum) and differential phases
(DifferentialPhase), against dense Gaussians built from their documented
definitions, on a four-telescope file with a line, written by
virgil.oifits.write_oifits (whose round trip test_observables_io checks).

FluxSpectrum (virgil#217: the prior on the grey scale is stated, never
taken from the data):

* "flux": the samples of each scale group are Gaussian about mu t with
  covariance D + sum_j c_j c_j^T, t the model's total spectrum over its
  group mean, c_0 = s t (the stated prior sd of k) and c_j = tau mu t x^j
  (x the wavelength scaled to [-1, 1] across the group). The block's
  log-likelihood is SciPy's multivariate normal, per dataset and per
  station, with and without a polynomial.
* "nflux": the same with t the total spectrum over its mean on the
  continuum channels of its row, and the default prior (1, 0.1).

DifferentialPhase:

* projection (the pipeline's): per frame, y = (Q^T x N_line) phi, with N =
  I - L the continuum operator, Q an orthonormal basis of the
  telescope-differenced phases (our own incidence matrix), Gaussian with
  covariance (Q^T x N_line) D (Q x N_line^T). Its log-likelihood is SciPy's
  multivariate normal.
* restricted maximum likelihood: the documented claim that the projection
  is the flat-prior limit of marginalising its null space. Differences of
  the log-likelihood between two models equal those of -1/2 r^T P r, with
  P = D^-1 - D^-1 X (X^T D^-1 X)^+ X^T D^-1 the generalised least-squares
  projector for nuisances X that we construct ourselves: the closure
  directions in every channel, and, per baseline, offsets and delays (no
  windows) or every phase pattern the line rows of N do not see (windows).
* prior_width: per frame and channel, the closure-free combinations of the
  phases, Gaussian with covariance (Q^T x I)(D + offsets and slopes of
  every baseline with the stated widths)(Q x I), slope in the wavenumber
  centred and scaled to span 1 over the channels used. At very wide priors
  the differences between models tend to the projection's (no windows).
"""

import numpy as np
import pytest
from astropy.io import fits
from scipy import linalg, stats

from crosscheck import array, sky
from evidence.plugin import record

oifits = pytest.importorskip("virgil.oifits")
vm = pytest.importorskip("virgil.models")
sp = pytest.importorskip("virgil.spectra")
from virgil.likelihood import model_loglike  # noqa: E402
from virgil.oidata import OIData  # noqa: E402

pytestmark = pytest.mark.x64

UTS = np.array([[-9.925, -20.335, 0.0], [14.887, 30.502, 0.0], [44.915, 66.183, 0.0], [103.306, 43.999, 0.0]])
WL = np.linspace(2.150e-6, 2.182e-6, 14).astype(np.float32).astype(float)  # as EFF_WAVE stores it
LINE, FWHM = 2.1661e-6, 2.5e-9
CONT = [(2.149e-6, 2.1615e-6), (2.1705e-6, 2.183e-6)]
LINES = [(2.1616e-6, 2.1704e-6)]
NODE_WL, NODE_F = np.array([2.14e-6, 2.19e-6]), np.array([0.05, 0.07])


def comp_flux(w, amp):
    return np.interp(w, NODE_WL, NODE_F) + amp * np.exp(-4 * np.log(2) * ((w - LINE) / FWHM) ** 2)


def total(w, amp):
    return 1.0 + comp_flux(w, amp)


def vis(u, v, w, x, y, amp):
    f = comp_flux(w, amp)
    return (sky.vis_uniform_disk(u, v, w, 1.0) + f * sky.vis_point(u, v, w, x, y)) / (1 + f)


def scene(x, y, amp):
    flux = sp.Sum(cont=sp.Nodes(NODE_F, NODE_WL), line=sp.GaussianLine(amp, line_wavel=LINE, fwhm=FWHM))
    return vm.System(star=vm.UniformDisk(1.0), comp=vm.PointSource(flux, x, y))


TRUTH = (6.0, -4.0, 0.3)
OTHER = (5.0, -2.5, 0.15)


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    rng = np.random.default_rng(5)
    rows, tri = [], []
    for k, ha in enumerate((-1.0, 1.5)):
        u, v = array.snapshot_uv(UTS, ha, -50.0, -24.6276)
        mjd = 60000.0 + k / 24
        for (i, j), uu, vv in zip(array.baselines(4), u, v):
            rows.append((mjd, i, j, uu, vv))
        cp, u1, v1, u2, v2 = array.closure_phase(lambda a, b: vis(a[:, None], b[:, None], WL[None, :], *TRUTH), u, v, 4)
        for t, abc in enumerate(array.triangles(4)):
            tri.append((mjd, abc, u1[t], v1[t], u2[t], v2[t], cp[t]))
    n, nw = len(rows), WL.size
    mjd_v = np.array([r[0] for r in rows])
    U, V = np.array([r[3] for r in rows]), np.array([r[4] for r in rows])
    sta_v = np.array([[r[1] + 1, r[2] + 1] for r in rows])
    cvis = vis(U[:, None], V[:, None], WL[None, :], *TRUTH)
    phi_err = np.deg2rad(0.5) * (1 + rng.uniform(0, 1, (n, nw)))  # unequal errors
    t = len(tri)
    u1, v1, u2, v2 = (np.array([r[q] for r in tri]) for q in (2, 3, 4, 5))
    mjd_f = np.repeat([60000.0, 60000.0 + 1 / 24], 4)
    sta_f = np.tile(np.arange(1, 5), 2)[:, None]
    flux_err = 0.01 * (1 + rng.uniform(0, 1, (8, nw)))
    tables = {
        "OI_WAVELENGTH": {"EFF_WAVE": WL, "EFF_BAND": np.full(nw, 2e-9)},
        "OI_VIS2": {"MJD": mjd_v, "UCOORD": U, "VCOORD": V, "STA_INDEX": sta_v,
                    "VIS2DATA": np.abs(cvis) ** 2 + 0.01 * rng.normal(size=(n, nw)), "VIS2ERR": np.full((n, nw), 0.01)},
        "OI_VIS": {"MJD": mjd_v, "UCOORD": U, "VCOORD": V, "STA_INDEX": sta_v, "AMPTYP": "absolute",
                   "PHITYP": "differential", "VISAMP": np.abs(cvis), "VISAMPERR": np.full((n, nw), 0.01),
                   "VISPHI": np.rad2deg(np.angle(cvis) + phi_err * rng.normal(size=(n, nw))),
                   "VISPHIERR": np.rad2deg(phi_err)},
        "OI_T3": {"MJD": np.array([r[0] for r in tri]), "U1COORD": u1, "V1COORD": v1, "U2COORD": u2, "V2COORD": v2,
                  "STA_INDEX": np.array([[x + 1 for x in r[1]] for r in tri]),
                  "T3PHI": np.rad2deg(np.array([r[6] for r in tri])) + rng.normal(size=(t, nw)),
                  "T3PHIERR": np.ones((t, nw))},
        "OI_FLUX": {"MJD": mjd_f, "STA_INDEX": sta_f,
                    "FLUXDATA": 3.0 * total(WL, TRUTH[2]) / total(WL, TRUTH[2]).mean() * (1 + 0.02 * np.linspace(-1, 1, nw))
                    + flux_err * rng.normal(size=(8, nw)),
                    "FLUXERR": flux_err},
        "info": {"OBJECT": "test", "INSNAME": "SIM", "ARRNAME": "VLTI", "STAXY": UTS[:, :2]},
    }
    path = oifits.write_oifits(tables, tmp_path_factory.mktemp("fv") / "line.fits")
    with fits.open(path) as h:  # the references use what the file holds
        assert np.array_equal(h["OI_WAVELENGTH"].data["EFF_WAVE"].astype(float), WL)
        for ext, cols in tables.items():
            if ext.startswith("OI_") and ext != "OI_WAVELENGTH":
                for c in cols:
                    if c in h[ext].columns.names:
                        cols[c] = np.asarray(h[ext].data[c], float).reshape(np.shape(cols[c]))
    return tables, path


def block(model, data_with, data_without):
    return float(model_loglike(model, data_with)) - float(model_loglike(model, data_without))


# ------------------------------------------------------------- FluxSpectrum


def flux_reference(tables, params, scale, per, poly_order, poly_width, kind="flux"):
    t = tables["OI_FLUX"]
    d, e = t["FLUXDATA"], t["FLUXERR"]
    F = np.broadcast_to(total(WL, params[2]), d.shape)
    if kind == "flux":
        groups = np.zeros(d.shape, int) if per == "dataset" else np.broadcast_to(t["STA_INDEX"], d.shape)
    else:
        groups = np.zeros(d.shape, int)
    out = 0.0
    for g in np.unique(groups):
        m = groups == g
        if kind == "flux":
            tmpl = F[m] / F[m].mean()
        else:
            cont = np.zeros(WL.size, bool)
            for lo, hi in CONT:
                cont |= (WL >= lo) & (WL <= hi)
            level = np.broadcast_to(F[:, cont].mean(axis=1, keepdims=True), d.shape)
            tmpl = (F / level)[m]
        mean, sd = scale
        w = np.broadcast_to(WL, d.shape)[m]
        x = (w - 0.5 * (w.min() + w.max())) / (0.5 * (w.max() - w.min()))
        cols = [sd * tmpl] + [poly_width * mean * tmpl * x**j for j in range(1, poly_order + 1)]
        C = np.diag(e[m] ** 2) + sum(np.outer(c, c) for c in cols)
        out += stats.multivariate_normal(mean * tmpl, C).logpdf(d[m])
    return out


@pytest.mark.parametrize("per,poly_order", [("dataset", 0), ("station", 0), ("dataset", 1), ("station", 2)])
@pytest.mark.validates("virgil.observables.FluxSpectrum", "virgil.oidata.OIData.with_flux_scale", roots=["mathematics"])
def test_flux_spectrum_is_a_dense_gaussian(written, per, poly_order):
    tables, path = written
    base = OIData(str(path))
    scale = (3.2, 0.4)
    data = OIData(str(path), extras=("flux",)).with_flux_scale(scale=scale, per=per, poly_order=poly_order,
                                                               poly_width=0.05)
    worst = 0.0
    for params in (TRUTH, OTHER):
        got = block(scene(*params), data, base)
        want = flux_reference(tables, params, scale, per, poly_order, 0.05)
        gap = abs(got - want)
        assert np.isfinite(gap), (got, want)
        worst = max(worst, gap)
    record("max_abs_dloglike", worst)
    assert worst < 1e-8


@pytest.mark.validates("virgil.observables.FluxSpectrum", roots=["mathematics"])
def test_normalised_flux_is_a_dense_gaussian(written):
    tables, path = written
    base = OIData(str(path))
    data = OIData(str(path), extras=("nflux",)).with_continuum(CONT, order=0)
    worst = 0.0
    for params in (TRUTH, OTHER):
        got = block(scene(*params), data, base)
        want = flux_reference(tables, params, (1.0, 0.1), "dataset", 0, 0.1, kind="nflux")
        gap = abs(got - want)
        assert np.isfinite(gap), (got, want)
        worst = max(worst, gap)
    record("max_abs_dloglike", worst)
    assert worst < 1e-8


@pytest.mark.validates("virgil.observables.FluxSpectrum", roots=["mathematics"])
def test_flux_scale_prior_is_required_and_not_from_the_data(written):
    _, path = written
    data = OIData(str(path), extras=("flux",))
    with pytest.raises(ValueError, match="prior"):
        model_loglike(scene(*TRUTH), data)


# -------------------------------------------------------- DifferentialPhase


def frames(tables, params):
    """Per frame: residuals (baseline-major over channels), variances, the
    station pairs, from the file's tables and our own model."""
    o = tables["OI_VIS"]
    model = np.angle(vis(o["UCOORD"][:, None], o["VCOORD"][:, None], WL[None, :], *params))
    data = np.deg2rad(o["VISPHI"])
    err = np.deg2rad(o["VISPHIERR"])
    out = []
    for m in np.unique(o["MJD"]):
        rows = np.flatnonzero(o["MJD"] == m)
        out.append(((model - data)[rows], err[rows] ** 2, o["STA_INDEX"][rows]))
    return out


def incidence(pairs):
    tel = np.unique(pairs)
    A = np.zeros((len(pairs), tel.size))
    for b, (i, j) in enumerate(pairs):
        A[b, np.searchsorted(tel, i)] += 1
        A[b, np.searchsorted(tel, j)] -= 1
    return A


def wavenumber():
    """1/lambda, centred and scaled: the same span as 1/lambda, but well
    conditioned beside a constant over a narrow band."""
    sigma = 1 / WL
    return (sigma - sigma.mean()) / sigma.std()


def our_continuum_fit(cont):
    """L: least squares in (1, 1/lambda) over the continuum, evaluated everywhere."""
    X = np.vander(wavenumber(), 2, increasing=True)
    L = np.zeros((WL.size, WL.size))
    L[:, cont] = X @ np.linalg.solve(X[cont].T @ X[cont], X[cont].T)
    return L


def windows():
    cont = np.zeros(WL.size, bool)
    for lo, hi in CONT:
        cont |= (WL >= lo) & (WL <= hi)
    return cont, ~cont


def reml(r, var, X):
    """-1/2 r^T P r with P the D-weighted projector off X."""
    r, Dinv = r.ravel(), 1 / var.ravel()
    Xw = X * Dinv[:, None]
    coef = np.linalg.lstsq(X.T @ Xw, Xw.T @ r, rcond=None)[0]
    return -0.5 * (r @ (Dinv * r) - (Xw.T @ r) @ coef)


def nuisances(pairs, window):
    nb, nc = len(pairs), WL.size
    closures = linalg.null_space(incidence(pairs).T)  # baseline patterns no telescope phase makes
    cols = [np.kron(c, np.eye(nc)[k]) for c in closures.T for k in range(nc)]
    if window:
        cont, line = windows()
        L = our_continuum_fit(cont)
        for j in np.flatnonzero(cont):
            v = np.zeros(nc)
            v[j] = 1.0
            v[line] = L[line] @ v  # unseen by the line rows of N = I - L
            cols += [np.kron(np.eye(nb)[b], v) for b in range(nb)]
    else:
        for basis in (np.ones(nc), wavenumber()):
            cols += [np.kron(np.eye(nb)[b], basis) for b in range(nb)]
    return np.array(cols).T


@pytest.mark.parametrize("window", [False, True], ids=["all-channels", "windows"])
@pytest.mark.validates("virgil.observables.DifferentialPhase", "virgil.oidata.OIData.with_continuum",
                       roots=["mathematics", "statistics"])
def test_differential_phase_is_restricted_maximum_likelihood(written, window):
    tables, path = written
    base = OIData(str(path))
    data = OIData(str(path), extras=("visphi",))
    if window:
        data = data.with_continuum(CONT, lines=LINES)
    got = block(scene(*TRUTH), data, base) - block(scene(*OTHER), data, base)
    want = 0.0
    for (ra, va, pa), (rb, _, _) in zip(frames(tables, TRUTH), frames(tables, OTHER)):
        X = nuisances(pa, window)
        want += reml(ra, va, X) - reml(rb, va, X)
    record("abs_ddloglike", abs(got - want))
    assert abs(got - want) < 1e-7 * max(1.0, abs(want))


@pytest.mark.validates("virgil.observables.DifferentialPhase", roots=["mathematics"])
def test_differential_phase_projection_is_a_dense_gaussian(written):
    tables, path = written
    base = OIData(str(path))
    data = OIData(str(path), extras=("visphi",)).with_continuum(CONT, lines=LINES)
    cont, line = windows()
    N_line = (np.eye(WL.size) - our_continuum_fit(cont))[line]
    worst = 0.0
    for params in (TRUTH, OTHER):
        got = block(scene(*params), data, base)
        want = 0.0
        for r, var, pairs in frames(tables, params):
            M = np.kron(linalg.orth(incidence(pairs)).T, N_line)
            want += stats.multivariate_normal(np.zeros(M.shape[0]), M @ np.diag(var.ravel()) @ M.T).logpdf(M @ r.ravel())
        gap = abs(got - want)
        assert np.isfinite(gap), (got, want)
        worst = max(worst, gap)
    record("max_abs_dloglike", worst)
    assert worst < 1e-7


def prior_reference(tables, params, widths):
    sigma = 1 / WL
    x = (sigma - sigma.mean()) / (sigma.max() - sigma.min())
    out = 0.0
    for r, var, pairs in frames(tables, params):
        nb = len(pairs)
        C = np.diag(var.ravel())
        for b in range(nb):
            for w, shape in zip(widths, (np.ones(WL.size), x)):
                c = np.kron(np.eye(nb)[b], w * shape)
                C = C + np.outer(c, c)
        M = np.kron(linalg.orth(incidence(pairs)).T, np.eye(WL.size))
        out += stats.multivariate_normal(np.zeros(M.shape[0]), M @ C @ M.T).logpdf(M @ r.ravel())
    return out


@pytest.mark.validates("virgil.observables.DifferentialPhase", roots=["mathematics"])
def test_differential_phase_with_finite_priors_is_a_dense_gaussian(written):
    tables, path = written
    base = OIData(str(path))
    widths = (0.3, 0.2)
    data = OIData(str(path), extras=("visphi",)).with_continuum(CONT, lines=LINES, prior_width=widths)
    worst = 0.0
    for params in (TRUTH, OTHER):
        got = block(scene(*params), data, base)
        want = prior_reference(tables, params, widths)
        gap = abs(got - want)
        assert np.isfinite(gap), (got, want)
        worst = max(worst, gap)
    record("max_abs_dloglike", worst)
    assert worst < 1e-7


@pytest.mark.validates("virgil.observables.DifferentialPhase", roots=["mathematics"])
def test_wide_priors_tend_to_the_projection(written):
    """Without windows (every channel both continuum and line), the flat
    limit of the finite priors is the pipeline's projection."""
    _, path = written
    base = OIData(str(path))
    flat = OIData(str(path), extras=("visphi",))
    want = block(scene(*TRUTH), flat, base) - block(scene(*OTHER), flat, base)
    gaps = []
    for w in (1e1, 1e3, 1e5):
        data = OIData(str(path), extras=("visphi",)).with_continuum(prior_width=w)
        gaps.append(abs(block(scene(*TRUTH), data, base) - block(scene(*OTHER), data, base) - want))
    record("gap_at_1e5", gaps[-1])
    assert gaps[-1] < 1e-6 * max(1.0, abs(want)) and gaps[0] > gaps[-1]
