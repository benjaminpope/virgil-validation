"""MPoL's side of the IM Lup cross-validation: the training asdf of a fold and the held-out score of a checkpoint on virgil's
dartboard folds (``external_bridge.mpol_worker`` tasks ``export_fold_asdf`` and ``score_fold``, ``external_bridge.cv_folds``).
Design: ``design/alma_imlup_p3.md`` ("Built: MPoL on the dartboard folds"), formats ``docs/alma_conventions.md`` Part 6.

* the numpy-only checks of ``cv_folds`` and ``cv_compare`` are in tests/test_cv_folds.py and tests/test_cv_compare.py (split from this file).
* with MPoL's environment (``.venv-mpol``, asdf, the examples clone) and virgil (``virgil.radio``), on a synthetic keyed set of 26
  ddids (visibilities with antennas, times and channels that run up in frequency for some ddids and down for others, plus
  explicit Hermitian partners) written as npz and as an asdf of the layout ``export_asdf.py`` writes:
  - the fold masks agree: the points ``loaddata`` returns from fold J's training asdf are exactly those of virgil's fold file with
    fold != J, in every fold, Hermitian partners together; an npz that is not this asdf's data is refused;
  - the held-out scores agree: MPoL's ``score_fold`` and virgil's own ``score`` (``virgil_bridge.alma.score``, the scorer of ``cv-task``)
    on the same held-out points, per ddid and per annulus, at the same parameters (MPoL's accurate NuFFT against virgil at 1e-12), and
    the four folds add up to ``objective_full``'s all-data chi^2.
"""

import json
import pathlib
import sys

import numpy as np
import pytest

from evidence.plugin import record
from _cv_common import _cv_dirs

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import test_alma_mpol as tm  # noqa: E402
import test_alma_mpol_ref as tr  # noqa: E402

C = 299792458.0


# ------------------------------------------------------------------ the synthetic set, MPoL's environment and virgil

try:  # CI's pinned virgil has no `virgil.radio`: the numpy tests above still run there
    import jax.numpy as jnp

    import virgil_bridge.alma as va
    import virgil_bridge.mpol as vbm

    HAVE_VIRGIL = True
except ImportError:
    HAVE_VIRGIL = False


def needs_env(f):
    """MPoL installed and the examples clone at the pinned commit (``test_alma_mpol_ref``'s marks), asdf and visread importable, and virgil.radio."""
    for m in [*tr.pytestmark, tr.needs_asdf, pytest.mark.skipif(not HAVE_VIRGIL, reason="virgil.radio (virgil from c446485) is not installed")]:
        f = m(f)
    return f


NPIX, K = 128, 4
OBS_OF_DDID = vbm.obs_of_ddid(tr.OBSID_DICT, 26) if HAVE_VIRGIL else None


def synthetic_rows(seed=5, n=40):
    """26 ddids in ``tests/test_alma_mpol_ref.subsample_rows``'s layout. Row r has a unique time and a baseline; channels run
    up in frequency for even ddids and down for odd ones; ``n // 2`` rows are Hermitian partners of others (baseline swapped,
    uvw and V conjugated). Visibilities are the sum of a point source and noise at the quoted weights."""
    rng = np.random.default_rng(seed)
    out = {}
    for d in range(26):
        nchan = 3 + d % 2
        freqs = (2.3e11 + 1e8 * d + 5e7 * np.arange(nchan))[:: 1 if d % 2 == 0 else -1]
        a1, a2 = rng.integers(0, 20, n), rng.integers(20, 40, n)
        uvw = rng.uniform(-2.5e3, 2.5e3, (n, 2))
        times = 5.0e9 + 6.0 * np.arange(n)
        part = rng.choice(n, n // 2, replace=False)  # partners: swapped antennas, -uvw, time + 3 s (an unused slot)
        a1, a2 = np.concatenate([a1, a2[part]]), np.concatenate([a2, a1[part]])
        uvw, times = np.concatenate([uvw, -uvw[part]]), np.concatenate([times, times[part] + 3.0])
        conj = np.concatenate([np.zeros(n, bool), np.ones(len(part), bool)])
        row, ch = np.nonzero(rng.uniform(size=(len(times), nchan)) > 0.12)
        u, v = uvw[row, 0] * freqs[ch] / C, uvw[row, 1] * freqs[ch] / C
        w = np.exp(rng.normal(8.0, 0.3, len(times)))[row]  # one weight per row, as the asdf holds it
        vis = 0.05 * np.exp(-2j * np.pi * (u * 3e-8 + v * 1e-8)) + (rng.normal(size=len(row)) + 1j * rng.normal(size=len(row))) / np.sqrt(w)
        vis = np.where(conj[row], np.conj(vis), vis)
        out[d] = dict(u=u, v=v, re=vis.real.astype(np.float32), im=vis.imag.astype(np.float32), weight=w.astype(np.float32), freq=freqs[ch],
                      ant1=a1[row].astype(np.int32), ant2=a2[row].astype(np.int32), chan=ch.astype(np.int16), time=times[row])
    return out


def smooth_checkpoint(path, truth, seed):
    """A checkpoint at NPIX whose image is IM Lup-like (a ring and a blob, 0.253 Jy, with the 30% texture of R1) and whose nuisances are random:
    a smooth image keeps MPoL's NuFFT interpolation error small, as in R1 (a white-noise image has power at every image frequency, and the
    interpolation of ``torchkbnufft`` then differs from an exact transform by 1e-3 of the flux). Saved as ``sgd.py`` saves it."""
    rng = np.random.default_rng(seed)
    img = tm.vba.ring_and_blob(NPIX, 5.0, radius_mas=180.0, width_mas=40.0, inc_deg=47.0, pa_deg=144.0, blob=(0.25, 100.0, -150.0, 40.0))
    img = img * (1 + 0.3 * rng.uniform(size=img.shape))
    sky = 0.253 * img / img.sum() / (5.0 / 1000) ** 2
    theta = tm.nontrivial_theta(rng, sky)
    np.savez(truth, **theta)
    tr._py(
        """
        import sys, numpy as np, torch, mpol_worker as w
        path, truth, npix, examples = sys.argv[1:5]
        net, _ = w._net(dict(examples_dir=examples, npix=int(npix)))
        w._set(net, np.load(truth))
        torch.save({"model_state_dict": net.state_dict()}, path)
        """,
        path, truth, NPIX, tr.EXAMPLES,
    )
    return {k: np.asarray(a) for k, a in theta.items()}


@pytest.fixture(scope="module")
def cv(tmp_path_factory):
    """The synthetic set as npz and as an asdf, virgil's fold file on it, a random MPoL checkpoint at 128^2, and virgil's data."""
    tmp = tmp_path_factory.mktemp("cv")
    rows = synthetic_rows()
    npz = tmp / "vis"
    npz.mkdir()
    for d, r in rows.items():
        np.savez(npz / f"ddid_{d:02d}.npz", **r, obsid=int(OBS_OF_DDID[d]), ddid=d)
    tr.write_asdf(tmp / "full.asdf", rows)
    data = va.load_dir(npz)
    fold, annulus = va.make_folds(data, k=K, n_q=12, n_phi=16, seed=0)
    meta = va.save_folds(tmp / "folds.npz", data, fold, annulus, dict(k=K, n_q=12, n_phi=16, seed=0))
    truth = smooth_checkpoint(tmp / "chk.pt", tmp / "truth.npz", seed=11)
    return dict(tmp=tmp, rows=rows, npz=str(npz), asdf=str(tmp / "full.asdf"), folds=str(tmp / "folds.npz"), meta=meta, data=data, fold=fold, annulus=annulus,
                chk=str(tmp / "chk.pt"), truth=truth)


def _training_points(cv, j, tmp_path):
    """What the example's ``loaddata`` returns from fold ``j``'s training asdf (run in MPoL's environment): the keys of each point."""
    res = tr._run(dict(task="export_fold_asdf", asdf=cv["asdf"], npz_dir=cv["npz"], folds=cv["folds"], fold=j, out_asdf=str(tmp_path / f"train{j}.asdf")), tmp_path)
    tr._py(
        """
        import sys, numpy as np, mpol_worker as w
        pts = w._asdf_points(sys.argv[1], sys.argv[3])
        np.savez(sys.argv[2], **pts)
        """,
        tmp_path / f"train{j}.asdf", tmp_path / f"train{j}_pts.npz", tr.EXAMPLES,
    )
    return res, dict(np.load(tmp_path / f"train{j}_pts.npz"))


def _keys(ddid, freq, time, a1, a2):
    return set(zip(np.asarray(ddid).tolist(), np.round(np.asarray(freq) / 1e3).astype(np.int64).tolist(), np.round(np.asarray(time) * 1e3).astype(np.int64).tolist(),
                   np.asarray(a1).tolist(), np.asarray(a2).tolist()))


@pytest.mark.slow
@pytest.mark.validates("external_bridge.mpol_worker", roots=["mpol", "mathematics"], kind="reference", tier="B")
@needs_env
def test_training_asdf_keeps_exactly_the_points_of_virgils_other_folds(cv, tmp_path):
    """For each fold J, the points ``loaddata`` returns from the training asdf are the points of virgil's fold file with fold != J
    (as the keys ddid, channel frequency, time, ant1, ant2), none of fold J are left, the counts add up, and the visibilities and
    weights of what is kept are untouched. The four held-out sets partition the data."""
    data_ids = np.concatenate([np.full(len(cv["rows"][d]["u"]), d) for d in range(26)])
    fold = cv["fold"]  # va.load_dir reads the ddids in order, so these are the points of the rows below in the same order
    rows = cv["rows"]
    cat = lambda k: np.concatenate([rows[d][k] for d in range(26)])  # noqa: E731
    all_keys = _keys(data_ids, cat("freq"), cat("time"), cat("ant1"), cat("ant2"))
    assert len(all_keys) == len(data_ids)  # the keys identify the points
    held_total = 0
    for j in range(K):
        res, pts = _training_points(cv, j, tmp_path)
        want = fold != j
        assert res["n_total"] == len(fold) and res["n_held"] == int((fold == j).sum()) and res["n_train"] == int(want.sum()) == len(pts["u"])
        assert res["folds_sha256"] == cv["meta"]["sha256"]
        assert _keys(pts["ddid"], pts["freq"], pts["time"], pts["ant1"], pts["ant2"]) == _keys(data_ids[want], cat("freq")[want], cat("time")[want], cat("ant1")[want], cat("ant2")[want])
        # what is kept is the data, unchanged: match by the same keys and compare V and w
        ours = {k: v for k, v in zip(zip(data_ids.tolist(), np.round(cat("freq") / 1e3).astype(np.int64).tolist(), np.round(cat("time") * 1e3).astype(np.int64).tolist(),
                                         cat("ant1").tolist(), cat("ant2").tolist()), cat("re").astype(float) + 1j * cat("im").astype(float))}
        got = dict(zip(zip(pts["ddid"].tolist(), np.round(pts["freq"] / 1e3).astype(np.int64).tolist(), np.round(pts["time"] * 1e3).astype(np.int64).tolist(),
                           pts["ant1"].tolist(), pts["ant2"].tolist()), pts["vis"]))
        assert max(abs(got[k] - ours[k]) for k in got) < 1e-12
        held_total += res["n_held"]
    assert held_total == len(fold)
    record("export_fold_asdf worst relative u difference to the npz", res["worst_match"]["u_max_rel"])


@pytest.mark.slow
@pytest.mark.validates("external_bridge.mpol_worker", roots=["mpol", "mathematics"], kind="reference", tier="B")
@needs_env
def test_hermitian_partners_are_in_the_same_training_asdf_or_both_out(cv, tmp_path):
    """Every point and its explicit Hermitian partner (baseline swapped, (u, v) -> (-u, -v), V conjugated; the synthetic set
    has them) is kept or dropped together, in fold 1's training asdf."""
    _, pts = _training_points(cv, 1, tmp_path)
    kept = _keys(pts["ddid"], pts["freq"], pts["time"], pts["ant1"], pts["ant2"])
    n_pairs = 0
    # the partner rows carry time + 3 s with swapped antennas: find pairs through the data, not through the kept set
    allk = {}
    for d in range(26):
        r = cv["rows"][d]
        for i in range(len(r["u"])):
            allk[(d, int(round(r["freq"][i] / 1e3)), int(round(r["time"][i] * 1e3)), int(r["ant1"][i]), int(r["ant2"][i]))] = i
    for (d, f, t, a1, a2), i in allk.items():
        other = (d, f, t - 3000, a2, a1)  # the original row's key, when this row is a partner (time + 3 s)
        if (t - 5_000_000_000_000) % 6000 == 3000 and other in allk:  # partner rows are at 3 s past a 6 s tick (times start at 5e9 s)
            n_pairs += 1
            assert ((d, f, t, a1, a2) in kept) == (other in kept), (d, f, t, a1, a2)
    assert n_pairs > 200
    record("hermitian pairs checked in a training asdf", n_pairs)


@pytest.mark.slow
@pytest.mark.validates("external_bridge.mpol_worker", roots=["mpol"], kind="control", tier="B")
@needs_env
def test_score_fold_checks_the_training_asdf_is_the_folds_own(cv, tmp_path):
    """``train_asdf`` of the right fold is recorded (fold, flagged count, hash of its flags, echoed checkpoint fold); another fold's
    training asdf, or the full export itself (nothing held out), raises instead of producing a score that looks normal."""
    j = 1
    asdfs = {f: str(tmp_path / f"train{f}.asdf") for f in (1, 2)}
    for f, path in asdfs.items():
        tr._run(dict(task="export_fold_asdf", asdf=cv["asdf"], npz_dir=cv["npz"], folds=cv["folds"], fold=f, out_asdf=path), tmp_path)
    base = dict(task="score_fold", checkpoint=cv["chk"], asdf=cv["asdf"], npz_dir=cv["npz"], folds=cv["folds"], fold=j, npix=NPIX, batch=700)
    ok = tr._run(dict(base, train_asdf=asdfs[1], checkpoint_fold=1), tmp_path)
    assert ok["train_fold"] == 1 and ok["checkpoint_trained_fold"] == 1 and ok["train_flagged"] == ok["n_test"] and len(ok["train_flags_sha256"]) == 64
    with pytest.raises(Exception, match="not fold 1's training set"):
        tr._run(dict(base, train_asdf=asdfs[2]), tmp_path)
    with pytest.raises(Exception, match="flags 0 points beyond the export"):  # the full export itself: no point held out
        tr._run(dict(base, train_asdf=cv["asdf"]), tmp_path)


@pytest.mark.slow
@pytest.mark.validates("external_bridge.mpol_worker", roots=["mpol"], kind="control", tier="B")
@needs_env
def test_export_fold_asdf_refuses_an_npz_that_is_not_the_asdfs_data(cv, tmp_path):
    """The fold file of one dataset against the asdf of another (other times and baselines) raises instead of writing a mask."""
    other = tmp_path / "other"
    other.mkdir()
    for d, r in synthetic_rows(seed=99).items():
        np.savez(other / f"ddid_{d:02d}.npz", **r, obsid=int(OBS_OF_DDID[d]), ddid=d)
    with pytest.raises(Exception, match="the fold file has \\d+ points, the npz \\d+|channel, time or antennas|differ beyond|cannot match"):
        tr._run(dict(task="export_fold_asdf", asdf=cv["asdf"], npz_dir=str(other), folds=cv["folds"], fold=0, out_asdf=str(tmp_path / "bad.asdf")), tmp_path)
    assert not (tmp_path / "bad.asdf").exists()


def _virgil_scores(cv, held_mask, values):
    prob = va.build_problem(cv["data"], "R", lam=1.0, npix=NPIX, pixel_mas=5.0)
    sc = va.score(cv["data"], prob.scene, values, prob.nuis, mask=held_mask, annulus=cv["annulus"])
    s1 = va.score(cv["data"], prob.scene, {**values, "cal.sigma": np.ones_like(np.asarray(values["cal.sigma"]))}, prob.nuis, mask=held_mask, annulus=cv["annulus"])
    return sc, s1


@pytest.mark.slow
@pytest.mark.validates("external_bridge.mpol_worker", roots=["mpol", "mathematics"], kind="check", tier="B")
@needs_env
def test_score_fold_agrees_with_virgils_score_on_the_same_held_out_points(cv, tmp_path):
    """The held-out scores of ``score_fold`` (MPoL's own model: ``Net``, accurate NuFFT, ``adjust_*``) and of
    ``virgil_bridge.alma.score`` (what ``cv-task`` reports) at the same parameters on fold 2's points: the same counts per ddid and
    annulus, and chi^2 and -ln L, at the fitted s and at s = 1, per ddid and per annulus, to the accuracy of MPoL's interpolated
    NuFFT against virgil's at 1e-12 (the model-dependent part: the constants N ln 2 pi - sum ln w cancel). The same also holds when
    virgil's values come from MPoL's export (``values_from_mpol_export``), which is how ``compare`` reads a checkpoint."""
    j = 2
    mp = tr._run(dict(task="score_fold", checkpoint=cv["chk"], asdf=cv["asdf"], npz_dir=cv["npz"], folds=cv["folds"], fold=j, npix=NPIX, batch=700), tmp_path)
    held = cv["fold"] == j
    z = {k: jnp.asarray(v) for k, v in cv["truth"].items()}
    values = vbm.virgil_values(z, jnp.asarray(OBS_OF_DDID), 5.0)
    sc, s1 = _virgil_scores(cv, held, values)
    ho = mp["heldout"]
    assert mp["n_test"] == ho["n"] == sc["n"] == int(held.sum()) and mp["folds_sha256"] == cv["meta"]["sha256"]
    assert ho["by_ddid"]["n"] == sc["by_ddid"]["n"] + [0] * (len(ho["by_ddid"]["n"]) - len(sc["by_ddid"]["n"]))
    assert ho["by_annulus"]["n"] == sc["by_annulus"]["n"]
    for name, mine, theirs in (("by_ddid", ho["by_ddid"], sc["by_ddid"]), ("by_annulus", ho["by_annulus"], sc["by_annulus"]),
                               ("by_ddid_s1", ho["by_ddid_s1"], s1["by_ddid"]), ("by_annulus_s1", ho["by_annulus_s1"], s1["by_annulus"])):
        m = len(theirs["n"])
        for key in ("chi2_raw", "chi2_fit"):
            a, b = np.asarray(mine[key][:m]), np.asarray(theirs[key])
            rel = np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))
            record(f"score_fold vs virgil, max relative difference of {key} {name}", float(rel))
            assert rel < 1e-4, (name, key, rel)
        a, b = np.asarray(mine["nll"][:m]), np.asarray(theirs["nll"])
        tol = 1e-4 * np.maximum(np.asarray(theirs["chi2_fit"]) * 0.5, 1.0)
        assert np.all(np.abs(a - b) <= tol), (name, a - b)
    assert ho["chi2_s1"] == pytest.approx(s1["chi2_raw"], rel=1e-4) and ho["chi2_fit"] == pytest.approx(sc["chi2_fit"], rel=1e-4)
    assert ho["nll"] == pytest.approx(sc["nll"], abs=1e-4 * sc["chi2_fit"] / 2)
    assert mp["chi2_raw_per_dof"] == pytest.approx(s1["chi2_raw"] / (2 * sc["n"]), rel=1e-4)
    # MPoL's own -ln L (mpol.losses.log_likelihood on the adjusted weights) is the sum the worker tallies
    assert mp["nll_mpol"] == pytest.approx(ho["nll"], rel=1e-12)
    # the same through the export, as `compare` reads a checkpoint
    exp = tmp_path / "exp.npz"
    tr._run(dict(task="export_checkpoint", checkpoint=cv["chk"], out_npz=str(exp), npix=NPIX), tmp_path)
    ex = dict(np.load(exp))
    v2 = va.values_from_mpol_export(ex, OBS_OF_DDID)
    sc2, _ = _virgil_scores(cv, held, v2)
    assert sc2["chi2_fit"] == pytest.approx(sc["chi2_fit"], rel=1e-9) and sc2["nll"] == pytest.approx(sc["nll"], rel=1e-9)
    record("held-out chi2_fit/(2n), fold 2, MPoL model", ho["chi2_fit"] / (2 * ho["n"]))


@pytest.mark.validates("external_bridge.cv_compare", roots=["mathematics"], kind="reference")
@pytest.mark.skipif(not HAVE_VIRGIL, reason="virgil.radio (virgil from c446485) is not installed")
def test_cv_compare_subcommand_writes_its_files_and_converts_lam_ent_by_c11(tmp_path):
    """``alma_imlup_maxent.py cv-compare`` writes the json, the markdown and the plot; N is the sum of the held-out counts, and
    lambda_0 = lam_F-equivalent of 2 N lam_ent (C4, C11)."""
    import alma_imlup_maxent as script

    n = 300
    m, v = _cv_dirs(tmp_path, lams=(1e-9, 1.0), n=n, k=3)
    lam_r = 2 * 3 * n * 1e-9
    lam0 = va.lam_f_equivalent(lam_r)
    assert lam0 == pytest.approx(lam_r / (5e-3) ** 2 * (0.253 / 0.253), rel=1e-12)
    script.main(["cv-compare", "--mpol", str(m), "--virgil", str(v), "--out", str(tmp_path / "out")])
    res = json.loads((tmp_path / "out" / "mpol_cv_compare.json").read_text())
    assert res["n_all"] == 3 * n and res["lam0"] == pytest.approx(lam0, rel=1e-12) and res["lam_match"] == 1.0  # 1.0 is nearer lam0 = 72 than 1e-9
    assert (tmp_path / "out" / "mpol_cv_compare.md").read_text().startswith("# MPoL and virgil") and (tmp_path / "out" / "mpol_cv_compare.png").exists()


@pytest.mark.slow
@pytest.mark.validates("external_bridge.mpol_worker", roots=["mpol", "mathematics"], kind="check", tier="B")
@needs_env
def test_held_out_scores_of_all_folds_add_up_to_the_all_data_score(cv, tmp_path):
    """The four folds' held-out counts and ``chi2_raw`` per ddid sum to virgil's score of every point at the same parameters (counts
    exactly, chi^2 to the NuFFT accuracy of the two codes), so no point is scored twice or not at all."""
    n_ddid, chi2 = np.zeros(26), np.zeros(26)
    for j in range(K):
        mp = tr._run(dict(task="score_fold", checkpoint=cv["chk"], asdf=cv["asdf"], npz_dir=cv["npz"], folds=cv["folds"], fold=j, npix=NPIX), tmp_path)
        n_ddid += mp["heldout"]["by_ddid"]["n"]
        chi2 += mp["heldout"]["by_ddid_s1"]["chi2_raw"]
    values = vbm.virgil_values({k: jnp.asarray(v) for k, v in cv["truth"].items()}, jnp.asarray(OBS_OF_DDID), 5.0)
    _, s1 = _virgil_scores(cv, None, values)
    np.testing.assert_array_equal(n_ddid, s1["by_ddid"]["n"])
    rel = np.max(np.abs(chi2 - s1["by_ddid"]["chi2_raw"]) / np.asarray(s1["by_ddid"]["chi2_raw"]))
    record("sum over folds of score_fold chi2_raw vs virgil all-data, max relative difference per ddid", float(rel))
    assert rel < 1e-4
