"""scripts/postagb_imaging.py: the azimuthal cap of the gp-polar field (LowOrderField), the calibration's image-error metric and the
cold, logged grid cells (design/gp_polar_hyperparameters.md)."""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
try:  # needs virgil main (MultipoleField), as tests/test_postagb_imaging.py
    import postagb_imaging as P  # noqa: E402
except ImportError:
    P = None
needs_multipole = pytest.mark.skipif(
    P is None, reason="needs virgil main (MultipoleField) and virgil-vlti with virgil_vlti.with_night_terms (install the postagb extra)"
)

# Guards our own script: a spectral low-pass and its variance, checked against the Fourier series and MultipoleField's documented
# sigma, and the bookkeeping of the grid; no standard or external code is involved.
pytestmark = pytest.mark.validates("evidence", roots=["mathematics"], kind="guard"
)


def _low_order(m_max, latent, **kw):
    return P.LowOrderField(
        latent,
        image_shape=(24, 24),
        r_max_mas=6.0,
        r_min_mas=1.0,
        m_max=m_max,
        length_r=1.0,
        **kw,
    )


@needs_multipole
@pytest.mark.parametrize("m_max", [0, 2, 3])
def test_low_order_field_keeps_only_orders_up_to_its_cap(m_max):
    latent = np.random.default_rng(1).standard_normal((10, 32))
    spec = np.abs(
        np.fft.fft(
            np.asarray(
                _low_order(m_max, latent, length_theta=0.0625, sigma=2.0).polar_values()
            ),
            axis=1,
        )
    )
    m = np.minimum(np.arange(32), 32 - np.arange(32))
    assert np.all(spec[:, m > m_max] < 1e-9 * spec.max())
    assert np.all(spec[:, m <= m_max].max(axis=0) > 1e-3 * spec.max())


@needs_multipole
@pytest.mark.parametrize(
    "length_theta", [0.0625, 0.25, 0.5]
)  # where the kept orders hold 26-83 % of the variance, so the rescaling matters
def test_low_order_field_keeps_sigma_as_the_rms_contrast(length_theta):
    # MultipoleField documents sigma as the field's standard deviation; the kept orders are rescaled so the capped field keeps it
    # (a white azimuthal prior would otherwise lose 74 % of its variance at m_max = 3)
    rng = np.random.default_rng(2)
    sigma, n = 1.5, 400
    capped = np.stack(
        [
            np.asarray(
                _low_order(
                    3,
                    rng.standard_normal((10, 32)),
                    length_theta=length_theta,
                    sigma=sigma,
                ).polar_values()
            )
            for _ in range(n)
        ]
    )
    free = np.stack(
        [
            np.asarray(
                P.MultipoleField(
                    rng.standard_normal((10, 32)),
                    (24, 24),
                    6.0,
                    r_min_mas=1.0,
                    length_r=1.0,
                    length_theta=length_theta,
                    sigma=sigma,
                ).polar_values()
            )
            for _ in range(n)
        ]
    )
    assert capped.var() == pytest.approx(sigma**2, rel=0.08)
    assert free.var() == pytest.approx(sigma**2, rel=0.08)


@needs_multipole
def test_low_order_field_matches_the_uncapped_field_when_the_cap_removes_nothing_of_weight():
    # at l_theta = 4 rad the orders above 3 carry < 1e-6 of the variance, so the cap changes nothing visible
    latent = np.random.default_rng(3).standard_normal((10, 32))
    kw = dict(r_min_mas=1.0, length_r=1.0, length_theta=4.0, sigma=1.0, nu_theta=np.inf)
    free = np.asarray(P.MultipoleField(latent, (24, 24), 6.0, **kw).polar_values())
    capped = np.asarray(
        P.LowOrderField(latent, (24, 24), 6.0, m_max=3, **kw).polar_values()
    )
    assert np.allclose(capped, free, atol=1e-4 * np.abs(free).max())


@needs_multipole
def test_low_order_field_renders_through_an_image_and_keeps_its_cap_when_untied():
    field = _low_order(2, np.zeros((10, 32)), length_theta=0.5, inc=40.0, pa=30.0)
    img = P.Image(field, 0.5, flux=0.3)
    assert np.all(np.isfinite(np.asarray(img.brightness)))
    assert field.with_geometry(10.0, 20.0, 0.0, 0.0).m_max == 2
    with pytest.raises(ValueError, match="m_max"):
        _low_order(16, np.zeros((10, 32)))


@needs_multipole
def test_the_cap_reaches_the_rendered_image():
    # the cap lives in polar_values(); pin that the Image's brightness goes through it: on a ring at mid-radius of a face-on field
    # the rendered log-brightness has (almost) no azimuthal power above m = 3 with the cap, and plenty without it
    from scipy import ndimage  # noqa: PLC0415

    latent = np.random.default_rng(4).standard_normal((20, 32))
    kw = dict(
        r_min_mas=1.0, length_r=1.0, length_theta=0.0625, sigma=1.0, inc=0.0, pa=0.0
    )
    n, pix = 64, 0.2
    phi = np.linspace(0.0, 2 * np.pi, 128, endpoint=False)
    rows, cols = (
        (n - 1) / 2 + 3.5 / pix * np.cos(phi),
        (n - 1) / 2 + 3.5 / pix * np.sin(phi),
    )

    def high_order_share(field):
        log_b = np.log(np.asarray(P.Image(field, pix, flux=0.3).brightness))
        ring = ndimage.map_coordinates(log_b, [rows, cols], order=3)
        power = np.abs(np.fft.rfft(ring - ring.mean())) ** 2
        return power[4:].sum() / power[1:].sum()

    capped = high_order_share(P.LowOrderField(latent, (n, n), 6.0, m_max=3, **kw))
    free = high_order_share(P.MultipoleField(latent, (n, n), 6.0, **kw))
    assert capped < 0.02 and free > 0.3


@needs_multipole
def test_image_errors_are_zero_for_the_truth_and_grow_with_dots():
    yy, xx = np.mgrid[:48, :48] - 23.5
    r, th = np.hypot(xx, yy) * 0.25, np.arctan2(yy, xx)
    ring = np.exp(-0.5 * ((r - 3.0) / 0.6) ** 2) * (1 + 0.3 * np.cos(th))
    dots = ring * (np.cos(16 * th) > 0.8)  # the same ring broken into 16 dots
    mask = r < 5.5
    from virgil.imaging import Beam  # noqa: PLC0415

    bm = Beam(1.0, 1.0, 0.0)
    same = P.image_errors(ring, ring, 0.25, bm, mask)
    assert same["nrmse_raw"] == pytest.approx(0.0, abs=1e-12) and same[
        "ncc_beam"
    ] == pytest.approx(1.0)
    worse = P.image_errors(dots, ring, 0.25, bm, mask)
    assert worse["nrmse_raw"] > 0.5 and worse["nrmse_raw"] > worse["nrmse_beam"]


def _parts(fn):
    """A stand-in for ``evidence_parts`` whose ln Z is ``fn(fit, data)`` (the cells' fits here are fakes without a model)."""
    return lambda f, d, path="env": dict(log_z=fn(f, d), chi2_sum=0.0, log_norm=0.0, z2=0.0, logdet=0.0, n_eff=0.0)


@needs_multipole
def test_polar_cells_start_every_cell_cold_and_record_its_loss_history(monkeypatch):
    starts = []

    def fake_fit(scene, priors, data, init=None, progress=None, **kw):
        starts.append(init)
        progress(dict(steps=10, loss=5.0, grad_norm=1.0, elapsed=0.1))
        return type(
            "F",
            (),
            dict(
                info=dict(
                    steps=10,
                    stop="gtol",
                    converged=True,
                    loss=5.0,
                    grad_norm=1e-5,
                    chi2_red=0.9,
                ),
                model=None,
            ),
        )()

    monkeypatch.setattr(P, "fit", fake_fit)
    monkeypatch.setattr(P, "evidence_parts", _parts(lambda f, d: -1.0))
    monkeypatch.setattr(P, "polar_scene_fn", lambda *a: (lambda sg, lr, lt: None))
    monkeypatch.setattr(P, "image_priors", lambda sc: {})
    monkeypatch.setattr(
        P, "compare_beam", lambda d: type("B", (), dict(minor_mas=1.0))()
    )
    args = type("A", (), dict(smoke=False))()
    start_env = type("E", (), dict(pixel_scale_mas=0.25))()
    cells = P.polar_cells(
        {},
        None,
        start_env,
        None,
        {},
        None,
        args,
        {},
        1.0,
        axes=[[0.5, 1.0], [0.25, 1.0], [1.0, 2.0]],
        extend=False,
    )[0]
    assert (
        len(cells) == 8 and starts == [None] * 8
    )  # no cell is warm-started from another
    rec = P.cell_records(cells, [[0.5, 1.0], [0.25, 1.0], [1.0, 2.0]])
    assert (
        rec[0]["history"] == [[10, 5.0, 1.0]]
        and rec[0]["stop"] == "gtol"
        and rec[0]["converged"]
    )


@needs_multipole
def test_polar_cells_never_pick_or_extend_past_the_prior_bounds(monkeypatch):
    """An evidence that always prefers more contrast and shorter radial lengths (the real-data failure) stops at the bounds:
    length_r = 1 beam and sigma = 2, with no extension past either."""

    def fake_fit(scene, priors, data, init=None, progress=None, **kw):
        info = dict(
            steps=1, stop="gtol", converged=True, loss=0.0, grad_norm=0.0, chi2_red=1.0
        )
        return type("F", (), dict(info=info, model=None, vals=scene))()

    monkeypatch.setattr(P, "fit", fake_fit)
    monkeypatch.setattr(
        P, "evidence_parts", _parts(lambda f, d: f.vals[0] - f.vals[1])
    )  # sigma - length_r
    monkeypatch.setattr(
        P, "polar_scene_fn", lambda *a: (lambda sg, lr, lt: (sg, lr, lt))
    )
    monkeypatch.setattr(P, "image_priors", lambda sc: {})
    monkeypatch.setattr(
        P, "compare_beam", lambda d: type("B", (), dict(minor_mas=1.2))()
    )
    args = type("A", (), dict(smoke=False))()
    start_env = type("E", (), dict(pixel_scale_mas=0.3))()
    cfg = dict(
        polar_lr_beams=(0.5, 1.0, 2.0)
    )  # a per-target grid below the bound is trimmed to it
    _, axes, best, _, extended = P.polar_cells(
        cfg, None, start_env, None, {}, None, args, {}, 1.0
    )
    assert min(axes[0]) == pytest.approx(1.2) and max(axes[2]) == P.POLAR_SIGMA_MAX
    assert best[0] == pytest.approx(1.2) and best[2] == P.POLAR_SIGMA_MAX
    assert not [e for e in extended if (e[0], e[1]) in ((0, "lower"), (2, "upper"))]


@needs_multipole
def test_unbounded_polar_cells_reach_below_one_beam_and_above_sigma_two(monkeypatch):
    def fake_fit(scene, priors, data, init=None, progress=None, **kw):
        info = dict(
            steps=1, stop="gtol", converged=True, loss=0.0, grad_norm=0.0, chi2_red=1.0
        )
        return type("F", (), dict(info=info, model=None, vals=scene))()

    monkeypatch.setattr(P, "fit", fake_fit)
    monkeypatch.setattr(P, "evidence_parts", _parts(lambda f, d: f.vals[0] - f.vals[1]))
    monkeypatch.setattr(
        P, "polar_scene_fn", lambda *a: (lambda sg, lr, lt: (sg, lr, lt))
    )
    monkeypatch.setattr(P, "image_priors", lambda sc: {})
    monkeypatch.setattr(
        P, "compare_beam", lambda d: type("B", (), dict(minor_mas=1.2))()
    )
    args = type("A", (), dict(smoke=False, unbounded=True))()
    start_env = type("E", (), dict(pixel_scale_mas=0.25))()
    _, axes, best, _, _ = P.polar_cells(
        {}, None, start_env, None, {}, None, args, {}, 1.0
    )
    assert (
        best[0] < 1.2 and best[2] > P.POLAR_SIGMA_MAX
    )  # the diagnostic grid is the one before the bounds


@needs_multipole
def test_reg_label_names_a_capped_polar_field(tmp_path):
    import json

    cfg = dict(key="X_gp-polar")
    json.dump(
        dict(
            regulariser="MultipoleField, azimuthal orders <= 3",
            gp_sigma=2.0,
            gp_length_r_mas=1.2,
            gp_length_theta=0.5,
        ),
        open(tmp_path / "X_gp-polar_image.json", "w"),
    )
    assert P.reg_label(tmp_path, cfg).startswith("polar GP")
