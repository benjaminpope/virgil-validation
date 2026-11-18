"""scripts/postagb_parametric_report.py: rendering orientation, tables and the section 9 branches, on synthetic JSON/NPZ."""

import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
try:  # numpy, scipy and matplotlib only; the virgil cross-check skips on its own
    import postagb_parametric_report as R  # noqa: E402
except ImportError:  # pragma: no cover
    R = None
needs_report = pytest.mark.skipif(R is None, reason="postagb_parametric_report needs scipy and matplotlib")

# Guards our own script's conventions (orientation, selection plumbing), not virgil against an independent result.
pytestmark = [needs_report, pytest.mark.validates("evidence", roots=["mathematics"], kind="guard")]


# --- fixtures ----------------------------------------------------------------------------


def _ring_values(inc, pa, r0, sigma, w=(), dra=0.0, ddec=0.0):
    v = {"env.inc": inc, "env.pa": pa, "env.profile.radius": r0, "env.profile.sigma": sigma,
         "env.harmonics.re": [float(np.real(x)) for x in w], "env.harmonics.im": [float(np.imag(x)) for x in w],
         "env.dra": dra, "env.ddec": ddec}
    return v


def _cs(w, n=720):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    f = R.positive_factor(th, np.asarray(w, complex))
    spec = np.fft.rfft(f) / n
    m = np.arange(1, len(w) + 1)
    return list(2 * spec[m].real), list(-2 * spec[m].imag)


def model_json(model, arm, *, inc=45.0, pa=135.0, r0=2.4, sigma=0.7, w=(), lnz=0.0, bic=0.0, flags=(),
               modes=None, dra=0.0, ddec=0.0, eligible=True, hole=None, variant=None):
    c, s = _cs(w) if len(w) else ([], [])
    p = dict(inc=inc, pa=pa, r0=r0, sigma=sigma, dra=dra, ddec=ddec, w=np.asarray(w, complex), c=np.array(c), s=np.array(s))
    hole = R.hole_contrast(p) if hole is None else hole
    best_mode = dict(values=_ring_values(inc, pa, r0, sigma, w, dra, ddec), sd={}, lnZ_lap=lnz, bic=bic,
                     eligible=eligible, why_not=None)
    values = _ring_values(inc, pa, r0, sigma, w, dra, ddec)
    values.update({"sec.dra": -1.0, "sec.ddec": -1.4})
    return dict(
        target="X", model=model, arm=arm, variant=variant, k=12, n_points=2184,
        modes=[best_mode] + (modes or []),
        best=dict(values=values, sd={}, raw_chi2=3.1, raw_chi2_v2=3.0, raw_chi2_cp=3.3, own_chi2=1.02, lnZ_lap=lnz,
                  lnZ_use=lnz, bic=bic, flags=list(flags)),
        derived=dict(inc=inc, pa_mod180=pa % 180, r0=r0, sigma=sigma, hole_contrast=hole, c=c, s=s,
                     a1=float(np.hypot(c[0], s[0])) if c else 0.0, f_prim_pct=64.0, f_sec_pct=2.8,
                     f_disc_pct=25.0, f_bg_pct=8.0, d_env=-0.6, centre=[dra, ddec]),
    )


def write_target(out, T, fits, selected=None, residuals=False):
    d = pathlib.Path(out) / T
    d.mkdir(parents=True, exist_ok=True)
    for (model, arm, variant), j in fits.items():
        j = dict(j, target=T)
        name = f"{T}_param_{model}_{arm}" + (f"_{variant}" if variant else "") + ".json"
        (d / name).write_text(json.dumps(j))
    if selected is not None:
        (d / f"{T}_param_selected.json").write_text(json.dumps(dict(selected, target=T)))
    if residuals:
        b = np.linspace(5, 120, 50)
        np.savez(d / f"{T}_param_residuals.npz", v2_baseline_m=b, v2_resid=np.sin(b), cp_baseline_m=b[:20], cp_resid=np.cos(b[:20]))
    return d


def selected_json(model="M2", jk=None, chain_flags=(), is_ring=None):
    chain = [dict(pair="M1/M0", dlnZ=30.0, dbic=-60.0, verdict="accept", flags=list(chain_flags)),
             dict(pair="M2/M1", dlnZ=12.0, dbic=-25.0, verdict="accept" if model in ("M2", "M2o") else "reject", flags=[])]
    a2 = dict(chain=chain, selected=model, flags=[])
    if is_ring is not None:
        a2["is_ring"] = is_ring
    return dict(arms={"N2": a2, "N1": dict(a2)}, jackknife=jk if jk is not None else dict(inc=2.0, pa=4.0, r0=0.1),
                gp_polar={"N1": dict(delta=-3.0), "N2": dict(delta=-41.0, mismatch=None)})


def ladder(T, *, inc=45.0, pa=135.0, r0=2.4, sigma=0.7, w=(0.5 + 0.2j,), flags=(), modes=None, m2_arm_flags=None):
    f = {}
    for arm in ARMS:
        f[("M0", arm, None)] = model_json("M0", arm, inc=inc, pa=pa, r0=0.01, sigma=2.0, lnz=-700, bic=1000)
        f[("M1", arm, None)] = model_json("M1", arm, inc=inc, pa=pa, r0=r0, sigma=sigma, lnz=-670, bic=940)
        f[("M2", arm, None)] = model_json("M2", arm, inc=inc, pa=pa, r0=r0, sigma=sigma, w=w, lnz=-655, bic=910,
                                          flags=flags, modes=modes)
    return f


ARMS = ("N0", "N1", "N2")


def stored_npz(path, n=128, pix=0.3, blob=None):
    """A stored compare_images.npz: ours/published blobs (optionally one off-centre point) plus conv versions."""
    idx = np.arange(n) - (n - 1) / 2
    east = -idx[None, :] * pix * np.ones((n, 1))
    north = -idx[:, None] * pix * np.ones((1, n))
    e0, n0 = blob if blob else (0.0, 0.0)
    ours = np.exp(-0.5 * ((east - e0) ** 2 + (north - n0) ** 2) / 0.5**2)
    ours /= ours.sum()
    conv = R.convolve(ours, (3.0, 3.0, 0.0))
    np.savez(path, pub_v2=np.zeros(3), pub_t3=np.zeros(3), ours=ours, published=ours, ours_conv=conv,
             published_conv=conv, pixel_mas=pix)


# --- rendering ---------------------------------------------------------------------------


@pytest.mark.parametrize("pa", [20.0, 80.0, 130.0, 170.0])
def test_ring_major_axis_at_input_pa_and_unit_sum(pa):
    p = dict(inc=60.0, pa=pa, r0=3.0, sigma=0.5, dra=0.0, ddec=0.0, w=np.zeros(0, complex), c=np.zeros(0), s=np.zeros(0))
    img = R.render_ring(p, 129, 0.12)
    assert img.sum() == pytest.approx(1.0)
    n = img.shape[0]
    yy, xx = np.mgrid[:n, :n]
    dx = -(xx - (n - 1) / 2) * 0.12  # East
    dy = -(yy - (n - 1) / 2) * 0.12  # North
    cov = np.array([[(img * dx * dx).sum(), (img * dx * dy).sum()], [(img * dx * dy).sum(), (img * dy * dy).sum()]])
    ev, vec = np.linalg.eigh(cov)
    major = vec[:, 1]
    got = np.degrees(np.arctan2(major[0], major[1])) % 180.0
    assert abs(R._angdiff180(got, pa)) < 1.0
    assert np.sqrt(ev[0] / ev[1]) == pytest.approx(np.cos(np.radians(60.0)), abs=0.02)


@pytest.mark.parametrize("w, expect_offset", [(0.6 + 0j, 0.0), (-0.6 + 0j, 180.0), (0.0 + 0.6j, -90.0), (0.0 - 0.6j, 90.0)])
@pytest.mark.parametrize("pa", [30.0, 100.0])
def test_bright_pa_for_real_and_imaginary_w1(w, expect_offset, pa):
    p = dict(inc=50.0, pa=pa, r0=3.0, sigma=0.5, dra=0.0, ddec=0.0, w=np.array([w]), c=np.zeros(0), s=np.zeros(0))
    got = R.bright_pa(p)
    assert abs(R._angdiff180(got, (pa + expect_offset) % 360)) < 0.2 or abs(((got - (pa + expect_offset)) + 180) % 360 - 180) < 0.2
    # the same answer from the rendered image (peak pixel about the ring centre)
    img = R.render_ring(p, 257, 0.05)
    iy, ix = np.unravel_index(np.argmax(img), img.shape)
    east, north = -(ix - 128) * 0.05, -(iy - 128) * 0.05
    from_img = np.degrees(np.arctan2(east, north)) % 360
    assert abs(((from_img - got) + 180) % 360 - 180) < 6.0


def test_harmonics_latents_match_derived_coefficients():
    w = np.array([0.8 - 0.3j, 0.2 + 0.4j])
    c, s = _cs(w)
    th = np.linspace(0, 2 * np.pi, 50)
    assert np.allclose(R.positive_factor(th, w), R.linear_factor(th, c, s), atol=1e-9)
    assert R.positive_factor(th, w).min() >= 0


def test_render_matches_virgil_multipoledisk():
    try:
        from virgil.models import GaussianRing, Harmonics, MultipoleDisk
    except ImportError:
        pytest.skip("virgil MultipoleDisk not installed")
    inc, pa, r0, sig, w = 55.0, 40.0, 3.0, 0.6, 0.5 + 0.4j
    disk = MultipoleDisk(GaussianRing(radius=r0, sigma=sig), harmonics=Harmonics(re=[w.real], im=[w.imag], form="positive"),
                         inc=inc, pa=pa, dra=0.4, ddec=-0.3, n_r=64)
    fov, npix = 16.0, 128
    ref = np.asarray(disk.render(npix, fov))
    p = dict(inc=inc, pa=pa, r0=r0, sigma=sig, dra=0.4, ddec=-0.3, w=np.array([w]), c=np.zeros(0), s=np.zeros(0))
    mine = R.render_ring(p, npix, fov / npix)
    assert R.ncc(ref, mine) > 0.97


def test_render_stage_and_off_centre_quadrant_in_all_panels(tmp_path):
    out, stored = tmp_path / "out", tmp_path / "stored"
    stored.mkdir()
    T = "IRAS15469-5311"
    stored_npz(stored / f"{T}_gp-polar_nightw_unbounded_compare_images.npz", blob=(2.0, 1.0))
    # a thin, nearly round ring offset by (2, 1) mas: its centroid is the offset
    fits = {(m, "N2", None): model_json(m, "N2", inc=10.0, pa=60.0, r0=1.5, sigma=0.3, w=(), dra=2.0, ddec=1.0)
            for m in ("M0", "M1", "M2")}
    write_target(out, T, fits, selected_json("M2"))
    path = R.render_stage(out, T, str(stored))
    z = np.load(path, allow_pickle=True)
    zs = np.load(stored / f"{T}_gp-polar_nightw_unbounded_compare_images.npz")
    n = z["convolved"].shape[0]
    centre = (n - 1) / 2
    for img in (z["convolved"], zs["ours_conv"], zs["published_conv"]):
        yy, xx = np.mgrid[:n, :n]
        cy, cx = (img * yy).sum() / img.sum(), (img * xx).sum() / img.sum()
        assert cy < centre and cx < centre  # North up (smaller row) and East left (smaller column)
        assert np.hypot(cx - centre, cy - centre) * 0.3 == pytest.approx(np.hypot(2.0, 1.0), abs=0.3)
    assert z["native"].sum() == pytest.approx(1.0) and z["convolved"].sum() == pytest.approx(1.0)
    assert float(z["beam_fit_rel_rms"]) < 0.02
    assert float(z["beam_pix"][0]) == pytest.approx(3.0, abs=0.2)
    png = R.compare_stage(out, T, str(stored))
    assert png.exists() and png.stat().st_size > 5000


def test_compare_without_stored_or_residuals(tmp_path):
    out = tmp_path / "out"
    write_target(out, "AI_Sco", ladder("AI_Sco"), selected_json("M2"))
    assert R.render_stage(out, "AI_Sco", None) is not None
    assert R.compare_stage(out, "AI_Sco", None).exists()


def test_render_missing_fit_returns_none(tmp_path):
    (tmp_path / "AI_Sco").mkdir()
    assert R.render_stage(tmp_path, "AI_Sco", None) is None
    assert R.compare_stage(tmp_path, "AI_Sco", None) is None


# --- tables ------------------------------------------------------------------------------


def test_table_builds_and_marks_missing_cells(tmp_path):
    out = tmp_path
    write_target(out, "AI_Sco", ladder("AI_Sco"), selected_json("M2"))
    (out / "EN_TrA").mkdir()  # nothing run
    md = R.table_stage(out)
    text = md.read_text()
    assert "AI_Sco" in text and "EN_TrA | not run" in text
    rows = json.loads((out / "postagb_parametric_table.json").read_text())
    ai = next(r for r in rows if r["target"] == "AI_Sco")
    assert ai["selected"] == "M2" and ai["is_ring"] is True
    assert ai["lnZ_use"]["N2"]["M2"] == -655 and ai["dbic_n0"]["M2"] == pytest.approx(-90)
    assert ai["gp_delta"]["N2"] == -41.0 and "-3.0 (not decisive)" in text and "-41.0 |" in text
    lit = ai["literature"]["i_rim"]
    assert lit["image"]["value"] == 46 and lit["pmoired"]["value"] == pytest.approx(45.6)
    assert lit["ours"] == pytest.approx(45.0)
    assert "M3" not in ai["lnZ_use"]["N2"]  # P2 not run: absent, not an error
    assert "## Literature comparison" in text


def test_table_dbic_and_gp_cells(tmp_path):
    f = ladder("AI_Sco")
    sel = selected_json("M2")
    sel["gp_polar"]["N1"] = dict(delta=None, mismatch="error_scale 1.338 != 1.280")
    write_target(tmp_path, "AI_Sco", f, sel)
    text = R.table_stage(tmp_path).read_text()
    assert "error_scale 1.338 != 1.280" in text
    assert "-41.0" not in text.split("dlnZ gp")[0]


# --- section 9 branches ------------------------------------------------------------------


def geo(tmp_path, T, fits, sel):
    write_target(tmp_path, T, fits, sel)
    path = R.geometry_stage(tmp_path)
    return json.loads(path.read_text())[T]


def test_ai_sco_ours_near_de_prins(tmp_path):
    g = geo(tmp_path, "AI_Sco", ladder("AI_Sco", inc=50.0, pa=140.0), selected_json("M2"))
    assert g["usable_for_gp"] and g["source"] == "ours" and g["inc"] == 50.0 and g["pa"] == pytest.approx(140.0)
    assert len(g["c"]) == 1 and g["jackknife_sd"]["inc"] == 2.0


def test_ai_sco_usable_but_elsewhere_falls_back(tmp_path):
    g = geo(tmp_path, "AI_Sco", ladder("AI_Sco", inc=70.0, pa=100.0), selected_json("M2"))
    assert g["usable_for_gp"] is True
    assert g["source"] == "De Prins image" and g["inc"] == 46 and g["pa"] == 135
    assert g["r0"] == pytest.approx(2.4) and g["branch"].startswith("ours usable but elsewhere")
    assert g["ours"]["inc"] == 70.0 and any("more than 15 deg" in r for r in g["reasons"])


def test_ai_sco_no_ring_falls_back(tmp_path):
    f = ladder("AI_Sco")
    g = geo(tmp_path, "AI_Sco", f, selected_json("M0"))
    assert not g["usable_for_gp"] and g["source"] == "De Prins image"
    assert g["branch"].startswith("no ring or unstable") and "selected model is not a ring" in g["reasons"]


def test_jackknife_too_large_and_pa_waived_at_low_inc(tmp_path):
    g = geo(tmp_path, "EN_TrA", ladder("EN_TrA", inc=40.0, pa=30.0), selected_json("M2", jk=dict(inc=12.0, pa=4.0)))
    assert not g["usable_for_gp"] and any("sd(inc)" in r for r in g["reasons"]) and g["source"] == "De Prins image"
    g = geo(tmp_path, "HD95767", ladder("HD95767", inc=40.0, pa=10.0), selected_json("M2", jk=dict(inc=3.0, pa=20.0)))
    assert not g["usable_for_gp"] and any("sd(PA)" in r for r in g["reasons"])
    g = geo(tmp_path, "HD108015", ladder("HD108015", inc=15.0, pa=80.0), selected_json("M2", jk=dict(inc=3.0, pa=40.0)))
    assert g["usable_for_gp"] and g["source"] == "ours"


def test_evidence_unverified_and_multimodal_block_usable(tmp_path):
    g = geo(tmp_path, "IRAS15469-5311", ladder("IRAS15469-5311", inc=35.0, pa=51.0),
            selected_json("M2", chain_flags=["evidence_unverified"]))
    assert not g["usable_for_gp"] and any("evidence_unverified" in r for r in g["reasons"])
    other = dict(values=_ring_values(65.0, 120.0, 2.4, 0.7), lnZ_lap=-657.0, eligible=True)
    g = geo(tmp_path, "HR4049", ladder("HR4049", inc=48.0, pa=95.0, modes=[other]), selected_json("M2"))
    assert not g["usable_for_gp"] and any("multimodal" in r for r in g["reasons"])
    near = dict(values=_ring_values(50.0, 100.0, 2.4, 0.7), lnZ_lap=-657.0, eligible=True)
    g = geo(tmp_path, "PS_Gem", ladder("PS_Gem", inc=48.0, pa=95.0, modes=[near]), selected_json("M2"))
    assert g["usable_for_gp"]


def test_missing_target_gets_de_prins_with_reason(tmp_path):
    (tmp_path / "IW_Car").mkdir()
    path = R.geometry_stage(tmp_path)
    g = json.loads(path.read_text())["IW_Car"]
    assert g["source"] == "De Prins image" and g["inc"] == 48.0 and g["pa"] == 144.0 and g["r0"] == pytest.approx(3.45)
    assert g["usable_for_gp"] is False and g["reasons"]


def test_cli_runs_all_stages(tmp_path):
    write_target(tmp_path, "AI_Sco", ladder("AI_Sco"), selected_json("M2"), residuals=True)
    for stage in ("render", "compare", "table", "geometry"):
        assert R.main([stage, str(tmp_path)]) == 0
    assert (tmp_path / "AI_Sco" / "AI_Sco_param_compare.png").exists()
    assert (tmp_path / "postagb_parametric_geometry.json").exists()
