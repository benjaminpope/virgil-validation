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

# Guards our own script's conventions (orientation, selection plumbing) and, where virgil is installed, its agreement
# with virgil's MultipoleDisk; none of it is an independent result, so the root is self-consistency.
pytestmark = [needs_report, pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")]


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
    assert abs(((got - (pa + expect_offset)) + 180) % 360 - 180) < 0.2  # modulo 360: a 180 deg error must fail
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
        pytest.skip("virgil MultipoleDisk not installed: the theta origin and PA sense are NOT checked against virgil here")
    inc, pa, r0, sig = 55.0, 40.0, 3.0, 0.6
    w = np.array([1.6 + 0.9j, -0.8 + 1.5j])  # |w| > 1.5 at both orders: the tanh map and the step-up recursion matter
    disk = MultipoleDisk(GaussianRing(radius=r0, sigma=sig),
                         harmonics=Harmonics(re=list(w.real), im=list(w.imag), form="positive"),
                         inc=inc, pa=pa, dra=0.4, ddec=-0.3, n_r=64)
    fov, npix = 16.0, 128
    ref = np.asarray(disk.render(npix, fov))
    p = dict(inc=inc, pa=pa, r0=r0, sigma=sig, dra=0.4, ddec=-0.3, w=w, c=np.zeros(0), s=np.zeros(0))
    mine = R.render_ring(p, npix, fov / npix)
    np.testing.assert_allclose(mine, ref, rtol=1e-12, atol=1e-14)


def test_axisymmetric_has_no_bright_pa():
    p = dict(inc=40.0, pa=30.0, r0=3.0, sigma=0.5, dra=0.0, ddec=0.0, w=np.zeros(0, complex), c=np.zeros(0), s=np.zeros(0))
    assert R.bright_pa(p) is None


def _beam_pair(tmp_path, major, minor, pa, name="IW_Car", run=None):
    """A stored npz (round source blob) with an elliptical beam applied, plus the sibling compare.json."""
    d = tmp_path / "stored" / (run or "")
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{name}_gp-polar_nightw_unbounded_compare_images.npz"
    stored_npz(f, blob=(0.0, 0.0))
    z = dict(np.load(f))
    beam = R.beam_from_mas(major, minor, pa, 0.3)
    z["ours_conv"] = R.convolve(z["ours"], beam)
    np.savez(f, **z)
    return f, beam


@pytest.mark.parametrize("pa", [45.0, 135.0])
def test_fit_beam_recovers_elliptical_beam_on_axisymmetric_source(tmp_path, pa):
    f, beam = _beam_pair(tmp_path, 1.8, 0.9, pa)
    z = np.load(f)
    got, rms = R.fit_beam(z["ours"], z["ours_conv"])
    assert rms < 0.02
    assert sorted(got[:2]) == pytest.approx(sorted(beam[:2]), rel=0.1)
    assert abs(got[0] - got[1]) > 0.3 * beam[0]  # not trapped on a circular beam


def test_render_stage_reads_beam_from_compare_json(tmp_path):
    T = "IW_Car"
    f, beam = _beam_pair(tmp_path, 1.8, 0.9, 45.0, name=T)
    (f.parent / f"{T}_gp-polar_nightw_unbounded_compare.json").write_text(
        json.dumps(dict(beam=dict(major_mas=1.8, minor_mas=0.9, pa_deg=45.0))))
    out = tmp_path / "out"
    write_target(out, T, ladder(T), selected_json("M2"))
    z = np.load(R.render_stage(out, T, str(f.parent)), allow_pickle=True)
    assert tuple(z["beam_pix"]) == pytest.approx(beam)
    assert float(z["beam_fit_rel_rms"]) < 1e-6


def test_stored_products_found_recursively_preferring_the_named_run(tmp_path):
    T = "HD95767"
    a, _ = _beam_pair(tmp_path, 1.8, 1.8, 0.0, name=T, run="04994a0+46ce208/" + T)
    b, _ = _beam_pair(tmp_path, 1.8, 1.8, 0.0, name=T, run="15b154a+6fa8afe/" + T)
    assert R.find_stored_compare(str(tmp_path / "stored"), T) == str(b)
    assert R.find_stored_compare(str(tmp_path / "stored"), "PS_Gem") is None
    a.unlink()
    b.unlink()
    assert R.find_stored_compare(str(tmp_path / "stored"), T) is None


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
    # the core's real schema: mismatch is a boolean, the text is in note, provisional marks PS Gem
    sel["gp_polar"]["N1"] = dict(delta=None, mismatch=True, note="mismatch: stored error_scale 1.338 != 1.280", provisional=False)
    sel["gp_polar"]["N2"] = dict(delta=-41.0, mismatch=False, note=None, provisional=True)
    write_target(tmp_path, "AI_Sco", f, sel)
    text = R.table_stage(tmp_path).read_text()
    assert "n/a (mismatch: stored error_scale 1.338 != 1.280)" in text
    assert "n/a (True)" not in text
    assert "-41.0 (provisional)" in text


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


def test_m3o_is_a_model(tmp_path):
    f = ladder("AI_Sco")
    f[("M2o", "N2", None)] = model_json("M2o", "N2", w=(0.5,), lnz=-650, bic=900)
    f[("M3o", "N2", None)] = model_json("M3o", "N2", w=(0.5, 0.2), lnz=-648, bic=895)
    write_target(tmp_path, "AI_Sco", f, selected_json("M3o"))
    R.table_stage(tmp_path)
    ai = json.loads((tmp_path / "postagb_parametric_table.json").read_text())[0]
    assert ai["selected"] == "M3o" and ai["lnZ_use"]["N2"]["M3o"] == -648


def test_ineligible_selected_best_does_not_crash(tmp_path):
    f = ladder("AI_Sco")
    for arm in ARMS:
        f[("M0", arm, None)]["best"] = None
        f[("M2", arm, None)]["best"] = None
    write_target(tmp_path, "AI_Sco", f, selected_json("M2"))
    assert R.render_stage(tmp_path, "AI_Sco", None) is None
    R.table_stage(tmp_path)
    g = json.loads(R.geometry_stage(tmp_path).read_text())["AI_Sco"]
    assert not g["usable_for_gp"] and g["source"] == "De Prins image"


def test_axisymmetric_selection_has_nan_bright_pa_and_wrapped_pa_in_npz(tmp_path):
    f = ladder("AI_Sco", pa=-8.4)
    write_target(tmp_path, "AI_Sco", f, selected_json("M1"))
    z = np.load(R.render_stage(tmp_path, "AI_Sco", None), allow_pickle=True)
    assert np.isnan(float(z["bright_pa"])) and float(z["pa"]) == pytest.approx(171.6)
    assert "dra" in z.files and "ddec" in z.files
    assert R.compare_stage(tmp_path, "AI_Sco", None).exists()  # title omits the brightest PA
    R.table_stage(tmp_path)
    assert json.loads((tmp_path / "postagb_parametric_table.json").read_text())[0]["bright_pa"] is None


def test_selected_model_chain_fallback_respects_accepted(tmp_path):
    sel = dict(arms={"N2": dict(chain=[dict(pair="M2/M0", accepted=True, verdict="BIC only", flags=[]),
                                       dict(pair="M3/M2", accepted=False, verdict="accept?", flags=[])])})
    assert R.selected_model(dict(selected=sel, models={}), "N2") == "M2"


def test_unverified_ring_over_blob_step_blocks_when_m1_is_ineligible(tmp_path):
    f = ladder("IRAS15469-5311", inc=35.0, pa=51.0)
    for arm in ARMS:
        f[("M1", arm, None)]["best"] = None
    sel = selected_json("M2")
    chain = [dict(pair="M1/M0", dlnZ=None, verdict="child has no eligible fit", flags=[]),
             dict(pair="M2/M0", dlnZ=40.0, verdict="accept", accepted=True, flags=["evidence_unverified"])]
    sel["arms"]["N2"]["chain"] = chain
    g = geo(tmp_path, "IRAS15469-5311", f, sel)
    assert not g["usable_for_gp"] and any("evidence_unverified" in r and "M2/M0" in r for r in g["reasons"])
    row = R.table_row(R.read_target(tmp_path, "IRAS15469-5311"), *R.literature())
    assert "evidence_unverified" in row["flags"]
    # and a flag on the selected fit alone blocks too
    f2 = ladder("HR4049", inc=48.0, pa=95.0, flags=["evidence_unverified"])
    g = geo(tmp_path, "HR4049", f2, selected_json("M2"))
    assert not g["usable_for_gp"] and any("selected fit" in r for r in g["reasons"])


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


# --- regression: the core's real output layout (AI Sco, trimmed copy of the fe70d14+d0f7086 files) -------------------

REAL = pathlib.Path(__file__).parent / "fixtures" / "postagb_parametric"


def test_real_layout_table_has_no_missing_geometry_cells(tmp_path):
    """derived lives in ``best["derived"]``: c1, s1, a1, fluxes and d_env must be read, PA wrapped, statuses shown."""
    R.table_stage(REAL, outdir=tmp_path)
    ai = json.loads((tmp_path / "postagb_parametric_table.json").read_text())[0]
    assert ai["selected"] == "M2o"
    assert ai["pa"] == pytest.approx(171.6005, abs=1e-3)  # wrapped, not the raw -8.4
    assert ai["c1"] == pytest.approx(-0.98988, abs=1e-4) and ai["s1"] == pytest.approx(-0.10624, abs=1e-4)  # odd order negated by the wrap
    assert ai["a1"] == pytest.approx(0.99557, abs=1e-4)
    assert (ai["f_prim_pct"], ai["f_sec_pct"]) == (pytest.approx(69.312, abs=1e-2), pytest.approx(4.296, abs=1e-2))
    assert ai["f_disc_pct"] == pytest.approx(17.099, abs=1e-2) and ai["f_bg_pct"] == pytest.approx(9.292, abs=1e-2)
    assert ai["d_env"] == pytest.approx(-1.37856, abs=1e-4)
    assert ai["bright_pa"] == pytest.approx(353.33, abs=0.01)
    text = (tmp_path / "postagb_parametric_table.md").read_text()
    row = next(ln for ln in text.splitlines() if ln.startswith("| AI_Sco"))
    # the M1 fit ran but has no eligible best: say so, never "not run"
    n2 = row.split(" | ")[-4]  # lnZ_use N2 column (the fixture holds only the N2 ladder plus M2o under N1)
    assert n2.startswith("M0 -1056.5 / M1 no eligible fit (bound, Hessian) / M2 -972.9 / M2o -892.7"), n2
    assert "n/a" not in " | ".join(row.split(" | ")[10:21])  # c1 .. d_env all filled


def test_wrap_pa_negates_odd_orders_only_for_odd_shifts():
    assert R.wrap_pa(-8.4, [0.5, 0.2], [0.1, -0.3]) == (pytest.approx(171.6), [-0.5, 0.2], [-0.1, -0.3])
    assert R.wrap_pa(200.0, [0.5], [0.1]) == (pytest.approx(20.0), [-0.5], [-0.1])
    assert R.wrap_pa(-188.4, [0.5], [0.1])[1:] == ([0.5], [0.1])  # two half-turns: unchanged (-188.4 -> 171.6)
    assert R.wrap_pa(60.0, [0.5], [0.1]) == (60.0, [0.5], [0.1])


def test_geometry_wrap_keeps_the_bright_side(tmp_path):
    """pa = 220 with w1 != 0: the wrapped (pa, c, s) in the geometry JSON must describe the same bright side."""
    w = (0.5 + 0.4j,)
    fits = ladder("EN_TrA", inc=55.0, pa=220.0, w=w)
    g = geo(tmp_path, "EN_TrA", fits, selected_json("M2"))
    assert g["source"] == "ours" and g["pa"] == pytest.approx(40.0)
    raw = dict(inc=55.0, pa=220.0, r0=2.4, sigma=0.7, dra=0.0, ddec=0.0, w=np.array(w), c=np.zeros(0), s=np.zeros(0))
    wrapped = dict(raw, pa=g["pa"], w=np.zeros(0, complex), c=np.array(g["c"]), s=np.array(g["s"]))
    want = R.bright_pa(raw)
    got = R.bright_pa(wrapped)
    assert abs(((got - want) + 180) % 360 - 180) < 0.2
    unflipped = dict(wrapped, c=-np.array(g["c"]), s=-np.array(g["s"]))  # the bug: wrap without flipping odd orders
    assert abs(((R.bright_pa(unflipped) - want) + 180) % 360 - 180) > 90
    # the table agrees
    row = R.table_row(R.read_target(tmp_path, "EN_TrA"), *R.literature())
    assert row["pa"] == pytest.approx(40.0) and row["c1"] == pytest.approx(g["c"][0])
    assert abs(((row["bright_pa"] - want) + 180) % 360 - 180) < 0.2


def test_real_layout_geometry_has_c_s_dra_ddec_d_env(tmp_path):
    path = R.geometry_stage(REAL, outfile=tmp_path / "geometry.json")
    g = json.loads(path.read_text())["AI_Sco"]
    ours = g["ours"]  # usable but elsewhere: De Prins geometry is the entry, our fit is kept beside it
    assert g["source"] == "De Prins image"
    for k in ("c", "s", "dra", "ddec", "d_env"):
        assert ours[k] is not None, k
    assert ours["pa"] == pytest.approx(171.6005, abs=1e-3)
    assert ours["c"][0] == pytest.approx(-0.98988, abs=1e-4) and ours["d_env"] == pytest.approx(-1.37856, abs=1e-4)
    assert (ours["dra"], ours["ddec"]) == (pytest.approx(1.6591, abs=1e-3), pytest.approx(-2.3406, abs=1e-3))
