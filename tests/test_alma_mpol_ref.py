"""MPoL's reference-run tasks (``external_bridge.mpol_worker``): checkpoint export, the full-data
objective, and the asdf-against-npz reader comparison. Formats: docs/alma_conventions.md Part 4.

The reference run on OzSTAR trains the IM Lup example's ``sgd.py`` on its own export of the
measurement set, then reads the checkpoint and the asdf back through these tasks. These tests
exercise them on small inputs in `.venv-mpol` (`scripts/setup_external.sh mpol`; the module
skips without it or without the examples clone) and need neither virgil nor CASA:

* checkpoint round trip: a randomly initialised ``Net`` saved with ``torch.save`` comes back with
  every parameter exactly, and an image spike at a known (l, m) lands where the exported axes say
  (East left, North up);
* ``objective_full`` on the stride-1000 subsample, in batches, equals vv#146's ``objective`` task
  (the whole sample in one batch, no batching) to float64 round-off, and with ``sparse_matrices``
  equals a numpy -ln L from that task's sparse-matrix model visibilities;
* ``asdf_vs_npz`` on an asdf built from the subsample (the same layout ``export_asdf.py``
  writes, made here from our npz since there is no CASA on the laptop) agrees point by point, and a
  conjugated asdf is *reported* (not corrected). The real comparison, visread's reading of the
  measurement set against ours on all 26 ddids, is the OzSTAR job's.
"""

import json
import os
import pathlib
import subprocess
import textwrap

import numpy as np
import pytest

from evidence.plugin import record
from external_bridge import _subprocess as sp

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXAMPLES = pathlib.Path(os.environ.get("MPOL_EXAMPLES", ROOT / ".external" / "mpol-examples"))
EXAMPLES_COMMIT = "5a7c3f0b373f62341ca4baf5869f95e916bbabb3"
WORKER_DIR = ROOT / "src" / "external_bridge"
SUBSAMPLE = pathlib.Path("~/data/alma_imlup/subsample_1000").expanduser()
OBSID_DICT = {0: [0, 1, 2, 3, 4], 1: [5, 6, 7, 8, 9, 10], 2: [11, 12], 3: [13, 14], 4: [15, 16, 17], 5: [18, 19, 20, 21], 6: [22, 23, 24, 25]}
C = 299792458.0
ARCSEC = np.pi / 180.0 / 3600.0
NPIX = 128  # the example's 1028 is used once, for the export (no NuFFT)


def _example_commit():
    try:
        out = subprocess.run(["git", "-C", str(EXAMPLES), "rev-parse", "HEAD"], capture_output=True, text=True)
    except OSError:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _asdf_ok():
    try:
        return sp.available("mpol", "asdf") and sp.available("mpol", "visread.process")
    except OSError:
        return False


pytestmark = [
    pytest.mark.skipif(not sp.available("mpol", "mpol"), reason="MPoL not installed (scripts/setup_external.sh mpol)"),
    pytest.mark.skipif(_example_commit() != EXAMPLES_COMMIT, reason=f"MPoL-dev/examples not cloned at {EXAMPLES_COMMIT[:8]} in {EXAMPLES}"),
]

needs_subsample = pytest.mark.skipif(len(list(SUBSAMPLE.glob("ddid_*.npz"))) != 26, reason="the stride-1000 subsample is not in ~/data/alma_imlup/subsample_1000")
needs_asdf = pytest.mark.skipif(not _asdf_ok(), reason="asdf/visread not in .venv-mpol (rerun scripts/setup_external.sh mpol)")


def _py(code, *args):
    """Run ``code`` in MPoL's interpreter with ``mpol_worker``'s directory on the path."""
    full = f"import sys; sys.path.insert(0, {str(WORKER_DIR)!r})\n" + textwrap.dedent(code)
    out = subprocess.run([sp.python("mpol"), "-c", full, *map(str, args)], capture_output=True, text=True, env=sp._env("mpol"), cwd=WORKER_DIR)  # not the repo root: its `src/` would shadow the example's own `src` package
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout


def _run(task, tmp_path):
    return sp.run("mpol", "mpol_worker.py", dict(examples_dir=str(EXAMPLES), **task))


def make_checkpoint(path, truth, npix, seed=0, spike=None, forward=False):
    """A randomly initialised example ``Net`` (nonzero everywhere) saved as ``sgd.py`` saves it, and its
    parameters in ``truth`` (npz). ``spike`` = (row, col) in the packed layout sets one bright base pixel;
    ``forward`` runs the forward chain first so the state dict holds ``icube.packed_cube`` as a trained
    one does."""
    _py(
        """
        import sys, numpy as np, torch, mpol_worker as w
        path, truth, npix, seed, spike, forward, examples = sys.argv[1:8]
        npix, seed = int(npix), int(seed)
        net, _ = w._net(dict(examples_dir=examples, npix=npix))
        rng = np.random.default_rng(seed)
        with torch.no_grad():
            if spike == "none":
                net.bcube.base_cube.copy_(torch.tensor(rng.normal(-3.0, 1.0, net.bcube.base_cube.shape)))
            else:
                r, c = map(int, spike.split(","))
                net.bcube.base_cube.fill_(-30.0)
                net.bcube.base_cube[0, r, c] = 5.0
            net.log10_sigma_factors.copy_(torch.tensor(rng.uniform(0.0, 0.3, 26)))
            net.log10_amp_factors.copy_(torch.tensor(rng.uniform(-0.05, 0.05, 6)))
            net.delta_ra.copy_(torch.tensor(rng.uniform(-0.01, 0.01, 6)))
            net.delta_dec.copy_(torch.tensor(rng.uniform(-0.01, 0.01, 6)))
        if forward == "1":
            with torch.no_grad():
                w._image(net)
        torch.save({"model_state_dict": net.state_dict()}, path)
        np.savez(truth, base_cube=net.bcube.base_cube.detach().numpy()[0], log10_sigma=net.log10_sigma_factors.detach().numpy(),
                 log10_amp=net.log10_amp_factors.detach().numpy(), delta_ra=net.delta_ra.detach().numpy(), delta_dec=net.delta_dec.detach().numpy())
        """,
        path, truth, npix, seed, "none" if spike is None else f"{spike[0]},{spike[1]}", int(forward), EXAMPLES,
    )
    return dict(np.load(truth))


# ------------------------------------------------------------------ export_checkpoint


@pytest.mark.validates("external_bridge.mpol_worker", roots=["mpol"], kind="reference")
def test_export_checkpoint_round_trips_a_saved_net(tmp_path):
    """Every parameter of a ``torch.save``d random ``Net`` comes back exactly, the fixed observation is
    inserted as 0, and the per-observation sigma means are ``analyze_scale_factors.py``'s averages. Run at
    the example's own 1028 x 1028, 5 mas."""
    truth = make_checkpoint(tmp_path / "chk.pt", tmp_path / "truth.npz", 1028, forward=True)
    res = _run(dict(task="export_checkpoint", checkpoint=str(tmp_path / "chk.pt"), out_npz=str(tmp_path / "chk.npz")), tmp_path)
    z = np.load(tmp_path / "chk.npz")
    assert res["missing_keys"] == [] and res["unexpected_keys"] == ["icube.packed_cube"]  # a buffer that is None until the first forward pass: sgd.py loads with strict=False for it
    assert z["image"].shape == z["base_image"].shape == z["base_cube"].shape == (1028, 1028)
    assert z["cell_size_arcsec"] == 0.005 and len(z["l_east_arcsec"]) == 1028
    np.testing.assert_array_equal(z["base_cube"], truth["base_cube"])
    np.testing.assert_array_equal(z["log10_sigma"], truth["log10_sigma"])
    np.testing.assert_array_equal(z["log10_amp"][1:], truth["log10_amp"])  # observation 0 is fixed at 0
    assert z["log10_amp"][0] == 0.0 and z["delta_ra_arcsec"][0] == 0.0 and z["delta_dec_arcsec"][0] == 0.0
    np.testing.assert_array_equal(z["delta_ra_arcsec"][1:], truth["delta_ra"])
    np.testing.assert_array_equal(z["delta_dec_arcsec"][1:], truth["delta_dec"])
    np.testing.assert_allclose(z["amp"], 10.0 ** z["log10_amp"], rtol=0, atol=0)
    np.testing.assert_allclose(res["delta_ra_mas"], 1e3 * z["delta_ra_arcsec"])
    for k, ddids in OBSID_DICT.items():  # analyze_scale_factors.py: np.average(10**log10_sigma over the observation's ddids)
        assert res["sigma_eb_mean"][str(k)] == pytest.approx(np.average(10.0 ** truth["log10_sigma"][ddids]), rel=1e-15)
    assert res["flux_jy"] == pytest.approx(z["image"].sum() * 0.005**2, rel=1e-14)
    assert res["stored_packed_cube_max_rel_diff"] < 1e-12  # the checkpoint's own icube.packed_cube (float64 here)
    record("export_checkpoint flux (Jy) of the random image", res["flux_jy"])


@pytest.mark.validates("external_bridge.mpol_worker", roots=["mpol", "mathematics"], kind="reference")
def test_exported_image_axes_are_the_sky_the_forward_model_sees(tmp_path):
    """A base-cube spike at packed (row 3, column 5) is, in the export, the brightest pixel at
    (l, m) = (``l_east_arcsec[col]``, ``m_north_arcsec[row]``), East left and North up. The axes are tied
    to MPoL's own forward model, not to the indexing that wrote them: the example's NuFFT visibilities of the
    spike have the phase of a point source there, V = F exp(-2 pi i (u l + m v)) (Part 1's convention, l East,
    m North) with a real positive taper, to the NuFFT's accuracy; the mirrored position, the control, is
    several radians off."""
    truth = make_checkpoint(tmp_path / "chk.pt", tmp_path / "truth.npz", NPIX, spike=(3, 5))
    _run(dict(task="export_checkpoint", checkpoint=str(tmp_path / "chk.pt"), out_npz=str(tmp_path / "chk.npz"), npix=NPIX), tmp_path)
    z = np.load(tmp_path / "chk.npz")
    row, col = np.unravel_index(np.argmax(z["image"]), z["image"].shape)
    assert row < NPIX // 2 and col < NPIX // 2  # North of centre is the top half; East of centre the left half
    assert z["m_north_arcsec"][0] > z["m_north_arcsec"][-1] and z["l_east_arcsec"][0] > z["l_east_arcsec"][-1]
    l, m = z["l_east_arcsec"][col] * ARCSEC, z["m_north_arcsec"][row] * ARCSEC
    rng = np.random.default_rng(1)
    u, v = rng.uniform(-3e6, 3e6, 40), rng.uniform(-3e6, 3e6, 40)
    np.savez(
        tmp_path / "spike.npz", u=u, v=v, data_re=np.zeros(40), data_im=np.zeros(40), weight=np.ones(40), ddid=np.zeros(40, int),
        log10_sigma=truth["log10_sigma"], log10_amp=truth["log10_amp"], delta_ra=truth["delta_ra"], delta_dec=truth["delta_dec"], base_cube=truth["base_cube"],
    )
    _run(dict(task="objective", npz=str(tmp_path / "spike.npz"), out_npz=str(tmp_path / "spike_out.npz"), grad=False, npix=NPIX), tmp_path)
    vis = (lambda o: o["vis_raw_re"] + 1j * o["vis_raw_im"])(np.load(tmp_path / "spike_out.npz"))
    resid = np.angle(vis * np.exp(2j * np.pi * (u * l + v * m)))  # phase left after removing the point source at (l, m)
    assert np.max(np.abs(resid)) < 1e-2
    flipped = np.angle(vis * np.exp(2j * np.pi * (-u * l + v * m)))  # control: l towards West
    assert np.max(np.abs(flipped)) > 0.5
    # the flux the taper leaves is the image's: the zero-spacing amplitude is of order the image flux
    assert z["image"].sum() * (0.005) ** 2 == pytest.approx(5.0067 * 0.005**2, rel=1e-3)


# ------------------------------------------------------------------ data from the subsample


def subsample_rows():
    """Per ddid: our npz's arrays, grouped into (row, channel) the way a measurement set holds them."""
    out = {}
    for d in range(26):
        z = np.load(SUBSAMPLE / f"ddid_{d:02d}.npz")
        out[d] = {k: np.asarray(z[k]) for k in ("u", "v", "re", "im", "weight", "freq", "ant1", "ant2", "chan", "time")}
    return out


def write_asdf(path, rows, conjugate=False):
    """An asdf in the layout of ``casa_analysis/src/export_asdf.py`` (``obsids/<k>/ddids/<d>`` with
    ``frequencies``, ``uu``, ``vv`` in metres, ``antenna1``, ``antenna2``, ``time``, ``data``, ``flag``,
    ``weight``), from the subsample's points: one row per (time, antenna pair), the channels it has unflagged.
    Our points are already V = conj(DATA), as visread's export is, so ``data`` is V (``conjugate`` flips it)."""
    npz = path.with_suffix(".npz")
    flat = {}
    for d, r in rows.items():
        key = np.stack([r["time"], r["ant1"], r["ant2"]], 1)
        uniq, inv = np.unique(key, axis=0, return_inverse=True)
        nchan, nrow = int(r["chan"].max()) + 1, len(uniq)
        freqs = np.zeros(nchan)
        freqs[r["chan"]] = r["freq"]
        data = np.zeros((nchan, nrow), complex)
        flag = np.ones((nchan, nrow), bool)
        uu, vv, weight = np.zeros(nrow), np.zeros(nrow), np.zeros(nrow)
        vis = r["re"].astype(float) + 1j * r["im"].astype(float)
        data[r["chan"], inv.ravel()] = np.conj(vis) if conjugate else vis
        flag[r["chan"], inv.ravel()] = False
        lam = C / r["freq"]
        uu[inv.ravel()], vv[inv.ravel()], weight[inv.ravel()] = r["u"] * lam, r["v"] * lam, r["weight"]
        for name, a in dict(frequencies=freqs, uu=uu, vv=vv, antenna1=uniq[:, 1].astype(int), antenna2=uniq[:, 2].astype(int), time=uniq[:, 0], data=data, flag=flag, weight=weight).items():
            flat[f"d{d}_{name}"] = a
    np.savez(npz, **flat)
    _py(
        """
        import sys, numpy as np, asdf
        npz, out, obsid = np.load(sys.argv[1]), sys.argv[2], eval(sys.argv[3])
        tree = {"obsids": {k: {"ddids": {d: {f: npz[f"d{d}_{f}"] for f in ("frequencies", "uu", "vv", "antenna1", "antenna2", "time", "data", "flag", "weight")} for d in ds}} for k, ds in obsid.items()}}
        asdf.AsdfFile(tree).write_to(out)
        """,
        npz, path, repr(OBSID_DICT),
    )


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """A random checkpoint at NPIX, and the subsample as an asdf."""
    tmp = tmp_path_factory.mktemp("ref")
    truth = make_checkpoint(tmp / "chk.pt", tmp / "truth.npz", NPIX, seed=3)
    rows = subsample_rows()
    write_asdf(tmp / "sub.asdf", rows)
    write_asdf(tmp / "sub_conj.asdf", rows, conjugate=True)
    return dict(tmp=tmp, chk=str(tmp / "chk.pt"), asdf=str(tmp / "sub.asdf"), asdf_conj=str(tmp / "sub_conj.asdf"), truth=truth, rows=rows)


# ------------------------------------------------------------------ objective_full


@needs_subsample
@needs_asdf
@pytest.mark.validates("external_bridge.mpol_worker", roots=["mpol", "mathematics"], kind="reference")
def test_objective_full_equals_the_one_batch_objective(reference):
    """vv#146's ``objective`` (the example's ``forward``, ``adjust_*`` and ``neg_log_likelihood_avg`` on the
    whole sample at once) and ``objective_full`` (batches of 7,000, MPoL's per-batch ``log_likelihood``
    summed) agree to float64 round-off with the interpolation table the example uses; with
    ``sparse_matrices`` -ln L equals a numpy sum over that task's sparse-matrix model visibilities."""
    rows, tmp = reference["rows"], reference["tmp"]
    cat = lambda k: np.concatenate([rows[d][k] for d in range(26)])  # noqa: E731
    # the same points in loaddata's order (ddid, then channel-major within it), so both tasks see one sample
    order = [np.lexsort((r["ant2"], r["ant1"], r["time"], r["chan"])) for r in (rows[d] for d in range(26))]
    cat = lambda k: np.concatenate([rows[d][k][o] for d, o in zip(range(26), order)])  # noqa: E731
    z = reference["truth"]
    np.savez(
        tmp / "obj.npz", u=cat("u"), v=cat("v"), data_re=cat("re").astype(float), data_im=cat("im").astype(float), weight=cat("weight").astype(float),
        ddid=np.concatenate([np.full(len(rows[d]["u"]), d) for d in range(26)]), log10_sigma=z["log10_sigma"], log10_amp=z["log10_amp"], delta_ra=z["delta_ra"],
        delta_dec=z["delta_dec"], base_cube=z["base_cube"],
    )
    base = dict(npix=NPIX)
    one = _run(dict(task="objective", npz=str(tmp / "obj.npz"), out_npz=str(tmp / "obj_out.npz"), grad=False, **base), tmp)
    full = _run(dict(task="objective_full", checkpoint=reference["chk"], asdf=reference["asdf"], batch=7000, sparse_matrices=False, **base), tmp)
    n = len(cat("u"))
    assert full["n"] == one["n"] == n
    assert full["neg_log_likelihood"] == pytest.approx(one["neg_log_likelihood"], rel=1e-11)
    # the sparse-matrix path, against numpy from the model visibilities `objective` writes for it
    o = np.load(tmp / "obj_out.npz")
    model = o["vis_model_sparse_re"] + 1j * o["vis_model_sparse_im"]
    data, w = cat("re").astype(float) + 1j * cat("im").astype(float), o["weight_used"]
    nll_np = n * np.log(2 * np.pi) - np.sum(np.log(w)) + 0.5 * np.sum(w * np.abs(model - data) ** 2)
    sparse = _run(dict(task="objective_full", checkpoint=reference["chk"], asdf=reference["asdf"], batch=7000, sparse_matrices=True, **base), tmp)
    assert sparse["neg_log_likelihood"] == pytest.approx(nll_np, rel=1e-10)
    assert abs(sparse["neg_log_likelihood"] / full["neg_log_likelihood"] - 1) > 1e-12  # the two interpolations differ, so the flag does something
    # -ln L from the per-ddid pieces: N ln 2 pi - sum ln w + chi2 / 2
    assert sparse["neg_log_likelihood"] == pytest.approx(n * np.log(2 * np.pi) - sum(sparse["sum_ln_weight_fit"]) + 0.5 * sparse["chi2_fit_total"], rel=1e-12)
    assert sum(sparse["n_ddid"]) == n and sparse["objective_R"] == pytest.approx(sparse["neg_log_likelihood"] + 2 * n * 1e-9 * sparse["entropy"])
    assert sparse["chi2_raw_total"] != sparse["chi2_fit_total"]
    # S_M against MPoL's documented formula, sum I ln(I / p) / F_tot, from the exported tapered image
    _run(dict(task="export_checkpoint", checkpoint=reference["chk"], out_npz=str(tmp / "exp.npz"), npix=NPIX), tmp)
    img = np.load(tmp / "exp.npz")["image"]
    s_m = np.sum(img * np.log(img / 1e-4)) / 0.253
    assert sparse["entropy"] == pytest.approx(s_m, rel=1e-12)
    assert sparse["loss_mpol"] == pytest.approx(sparse["neg_log_likelihood_avg"] + 1e-9 * sparse["entropy"], rel=1e-14)
    assert sparse["neg_log_likelihood_avg"] == pytest.approx(sparse["neg_log_likelihood"] / (2 * n), rel=1e-14)
    record("objective_full -ln L (sparse) over the stride-1000 subsample", sparse["neg_log_likelihood"])


# ------------------------------------------------------------------ asdf_vs_npz


@needs_subsample
@needs_asdf
@pytest.mark.validates("external_bridge.mpol_worker", roots=["mpol"], kind="reference")
def test_asdf_vs_npz_agrees_on_an_export_of_the_subsample(reference):
    """On an asdf with the layout ``export_asdf.py`` writes, built from the subsample, MPoL's reader
    (``loaddata``) returns our points: u and v to float64 round-off (metres to lambda and back), V and the
    weight exactly, all keys matched."""
    res = _run(dict(task="asdf_vs_npz", asdf=reference["asdf"], npz_dir=str(SUBSAMPLE)), reference["tmp"])
    assert res["n_asdf_total"] == res["n_npz_total"] > 40000
    for d, r in res["ddids"].items():
        assert r["matching"] == "keys" and r["key_mismatch"] == 0, (d, r)
        assert r["u_max_rel"] < 1e-12 and r["vis_max_abs"] < 1e-12 and r["weight_max_rel"] < 1e-12, (d, r)
        assert r["v_max_abs"] < 1e-12 * max(1.0, np.abs(reference["rows"][int(d)]["v"]).max()), (d, r)
        # the convention signatures must be non-zero when the readers agree (two-sided partner of the control)
        assert r["vis_conj_max_abs"] > 1e-3 and r["u_flip_max_abs"] > 0 and r["v_flip_max_abs"] > 0, (d, r)
        assert r["u_ratio_median"] == pytest.approx(1.0, abs=1e-12), (d, r)
    assert res["worst"]["n_mismatch_ddids"] == [] and res["worst"]["key_mismatch_max"] == 0
    record("asdf_vs_npz worst |du| (lambda), synthetic asdf", res["worst"]["u_max_abs"])


@needs_subsample
@needs_asdf
@pytest.mark.validates("external_bridge.mpol_worker", roots=["mpol"], kind="control")
def test_asdf_vs_npz_reports_a_conjugation_and_does_not_fix_it(reference):
    """The same export with V conjugated: the visibilities differ by up to twice the amplitude and
    ``vis_conj_max_abs`` is zero (the signature of a conjugation); u and v are untouched."""
    res = _run(dict(task="asdf_vs_npz", asdf=reference["asdf_conj"], npz_dir=str(SUBSAMPLE)), reference["tmp"])
    assert res["worst"]["vis_max_abs"] > 1e-2
    for r in res["ddids"].values():
        assert r["vis_conj_max_abs"] < 1e-12 and r["u_max_rel"] < 1e-12 and r["v_max_abs"] < 1e-6


@needs_subsample
@needs_asdf
@pytest.mark.validates("external_bridge.mpol_worker", roots=["mpol"], kind="control")
def test_asdf_vs_npz_summary_flags_a_missing_ddid(reference, tmp_path):
    """With one ddid's npz absent, ``worst`` lists it in ``n_mismatch_ddids`` instead of looking clean."""
    import shutil

    for f in SUBSAMPLE.glob("ddid_*.npz"):
        if f.name != "ddid_07.npz":
            shutil.copy(f, tmp_path / f.name)
    res = _run(dict(task="asdf_vs_npz", asdf=reference["asdf"], npz_dir=str(tmp_path)), reference["tmp"])
    assert res["worst"]["n_mismatch_ddids"] == [7]
