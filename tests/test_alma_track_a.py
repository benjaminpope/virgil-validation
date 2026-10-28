"""Track A / Track B of the IM Lup better-than-MPoL plan (``design/alma_imlup_p3.md``) on toy data (CPU, seconds): ``--reg``, the GP grid, the synthetic truth, the pass block and the MPoL replica."""

import json
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("virgil.radio")

import jax.numpy as jnp  # noqa: E402

import crosscheck.alma_maxent as cx  # noqa: E402
import virgil_bridge.alma as va  # noqa: E402
import virgil_bridge.alma_gp as vg  # noqa: E402
import virgil_bridge.alma_replica as vr  # noqa: E402
import virgil_bridge.alma_start as vs  # noqa: E402
import virgil_bridge.alma_synth as vy  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from test_alma_maxent import _call, script, synthetic_dir  # noqa: E402

PIX = 5.0


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    return synthetic_dir(tmp_path_factory.mktemp("ta"))


# ------------------------------------------------------------------ --reg


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_parse_reg_and_its_refusals():
    assert va.parse_reg("mem:1e-9,tsv:1e-4") == [("mem", 1e-9), ("tsv", 1e-4)]
    for bad in ("mem", "foo:1", "mem:0", "mem:1,mem:2", "tsv:-1"):
        with pytest.raises(ValueError):
            va.parse_reg(bad)


@pytest.mark.validates("virgil_bridge.alma", roots=["mathematics"])
def test_reg_mem_is_mpols_lambda_m_pull_and_tsv_is_the_zero_padded_squared_difference():
    """Evaluated, not read back: with a flat prior, the change of virgil's ``MaxEntropy(smoothed=False)`` between two unit-sum images equals ``2 n L`` times the change of MPoL's entropy
    ``(1/F) sum I ln(I/p)`` (I = F b / dx^2, F = 0.253 Jy; the prior only adds a flux-only term, which cancels); ``TSV`` equals weight x the numpy squared steps of the zero-padded image. The training
    fraction scales both weights."""
    n, npix, lam = 10_000, 24, 1e-9
    scene = va.grid_scene(npix, PIX)
    rng = np.random.default_rng(1)
    eta = [rng.normal(size=(npix, npix)) for _ in range(2)]
    b = [np.exp(e) / np.exp(e).sum() for e in eta]
    for scale in (1.0, 0.8):
        mem, tsv = va.reg_regularisers(va.parse_reg(f"mem:{lam},tsv:1e-4"), n=n, pixel_mas=PIX, scale=scale)
        assert not mem.smoothed and not tsv.smoothed
        val = lambda r, e: float(r.value(scene({"log_brightness": jnp.asarray(e)})))  # noqa: E731
        ent = lambda x: cx.mpol_entropy(0.253 * x / (PIX / 1000) ** 2, 1e-4, 0.253)  # noqa: E731
        assert val(mem, eta[1]) - val(mem, eta[0]) == pytest.approx(scale * 2 * n * lam * (ent(b[1]) - ent(b[0])), rel=1e-5)
        pad = lambda x: np.pad(x, 1)  # noqa: E731
        sq = lambda x: np.sum(np.diff(pad(x), axis=0) ** 2) + np.sum(np.diff(pad(x), axis=1) ** 2)  # noqa: E731
        assert val(tsv, eta[0]) == pytest.approx(scale * 1e-4 * sq(b[0]), rel=1e-10)
    # the pull, not only the value: the gradient of the virgil term in the log-pixels equals that of 2 n L x MPoL's loss term, written out as in mpol 0.3.1 losses.entropy
    # (1 / tot_flux * sum(cube * log(cube / prior_intensity))) on the cube I = F softmax(eta) / dx^2
    import jax

    mem = va.reg_regularisers(va.parse_reg(f"mem:{lam}"), n=n, pixel_mas=PIX)[0]
    mpol_entropy = lambda e: (lambda I: jnp.sum(I * jnp.log(I / 1e-4)) / 0.253)(0.253 * jax.nn.softmax(e.ravel()).reshape(e.shape) / (PIX / 1000) ** 2)  # noqa: E731
    g_v = jax.grad(lambda e: mem.value(scene({"log_brightness": e})))(jnp.asarray(eta[0]))
    g_m = 2 * n * lam * jax.grad(mpol_entropy)(jnp.asarray(eta[0]))
    assert float(jnp.max(jnp.abs(g_v - g_m)) / jnp.max(jnp.abs(g_m))) < 1e-5


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_reg_is_config_f_only_and_toeplitz_equals_direct(syn):
    data = va.load_dir(syn)
    with pytest.raises(ValueError):
        va.build_problem(data, "R", lam=0.0, npix=48, reg="mem:1e-9")
    kw = dict(lam=0.0, npix=48, pixel_mas=PIX, reg="mem:1e-9,tsv:1e-4", reg_n=len(data.u))
    pd, pt = va.build_problem(data, "F", chi2="direct", **kw), va.build_problem(data, "F", chi2="toeplitz", **kw)
    v = {k: jnp.asarray(x) for k, x in pd.init.items()}
    v["log_brightness"] = jnp.asarray(np.random.default_rng(0).normal(size=(48, 48)))
    with va.ft_eps(1e-12):
        ld, lt = float(vg.term_loss(pd.terms[0], v)), float(vg.term_loss(pt.terms[0], v))
    assert abs(ld - lt) < 1e-8 * abs(ld)
    assert len(pd.terms) == 1 and len(pd.regs) == 2 and pd.nuis.blur_fwhm_mas is None


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_fit_reg_runs_refuses_lam_and_cv_task_takes_reg_and_gp_arms(syn, tmp_path):
    out = tmp_path / "f"
    _call("fit", "--config", "F", "--reg", "mem:1e-9,tsv:1e-4", "--chi2", "toeplitz", "--data", syn, "--out", out, "--npix", 48, "--max-steps", 5, "--no-dirty")
    res = json.loads((out / "result.json").read_text())
    assert res["config"] == "F" and np.isfinite(res["nll"])
    with pytest.raises(SystemExit):
        _call("fit", "--config", "F", "--reg", "mem:1e-9", "--lam", 1.0, "--data", syn, "--out", tmp_path / "x", "--npix", 48)
    with pytest.raises(SystemExit):
        _call("fit", "--config", "R", "--reg", "mem:1e-9", "--data", syn, "--out", tmp_path / "y", "--npix", 48)
    folds = tmp_path / "folds.npz"
    _call("folds", "--data", syn, "--out", folds, "--k", 2)
    cv = tmp_path / "cv"
    for j in (0, 1):
        _call("cv-task", "--config", "F", "--reg", "mem:1e-9,tsv:1e-4", "--data", syn, "--folds", folds, "--fold", j, "--init", out, "--npix", 48, "--max-steps", 5, "--out", cv / f"a_{j}")
    r = json.loads((cv / "a_0" / "cv.json").read_text())
    assert r["arm"] and r["key"] == "mem:1e-9,tsv:1e-4" and np.isfinite(r["heldout"]["nll"])
    _call("cv-aggregate", cv)
    assert json.loads((cv / "cv_summary.json").read_text())["folds"] == [0, 1]
    # a second arm under the same directory is refused (one arm per cv directory)
    _call("cv-task", "--config", "F", "--reg", "mem:1e-9,tsv:2e-4", "--data", syn, "--folds", folds, "--fold", 0, "--init", out, "--npix", 48, "--max-steps", 5, "--out", cv / "b_0")
    with pytest.raises(SystemExit):
        _call("cv-aggregate", cv)


@pytest.mark.validates("virgil_bridge.alma_gp", roots=["self-consistency"], kind="guard")
def test_gp_grid_follows_the_hyperparameters():
    n_r0, n_t0 = vg.auto_grid(2400.0, 100.0, 1.0)
    assert (n_r0, n_t0) == (vg.GP["n_r"], vg.GP["n_theta"])  # the default field is unchanged
    n_r, n_t = vg.auto_grid(3200.0, 10.0, 0.1)
    assert n_r >= 2 * 3200.0 / 10.0 and n_t >= 2 * np.pi * 2 / 0.1 and n_r > n_r0


# ------------------------------------------------------------------ the pass block


@pytest.mark.validates("virgil_bridge.alma_start", roots=["self-consistency"], kind="guard")
def test_better_than_mpol_passes_only_when_all_five_do():
    good = dict(nrmse={"0.5": 0.10, "1": 0.05}, n_eff_ratio=0.9, base_dead_in_disk_1e4=0.0)
    r = dict(nrmse={"0.5": 0.20, "1": 0.06}, n_eff_ratio=0.12, base_dead_in_disk_1e4=0.5)
    kw = dict(heldout_ours=[2.69, 2.69, 2.69, 2.69, 2.69], heldout_mpol=[2.70, 2.70, 2.70, 2.70, 2.70], chi2n_ours=2.7044, synth=good, synth_r=r, flux_ours=0.26, flux_ref=0.2548, flux_ref_name="CLEAN", stages_converged=[True, True])
    assert vs.better_than_mpol(**kw)["verdict"] == "PASS"
    for key, bad in (("heldout_ours", [2.71] * 5), ("heldout_ours", [2.60, 2.80, 2.60, 2.80, 2.69]), ("chi2n_ours", 2.71), ("flux_ours", 0.30), ("stages_converged", [True, False]),
                     ("synth", {**good, "n_eff_ratio": 0.5}), ("synth", {**good, "nrmse": {"0.5": 0.3}}), ("synth", {**good, "base_dead_in_disk_1e4": 0.02})):
        assert vs.better_than_mpol(**{**kw, key: bad})["verdict"] == "FAIL", key
    assert vs.better_than_mpol(**{**kw, "synth": None})["verdict"] == "INCOMPLETE"
    assert "no uncertainty" in vs.better_than_mpol(**{**kw, "heldout_ours": 2.69, "heldout_mpol": 2.70})["items"][0]["value"]
    assert "PASS" in vs.better_markdown(vs.better_than_mpol(**kw))[0]


# ------------------------------------------------------------------ the synthetic truth


def _truth(npix=64):
    """A smooth ring (about 20 mas wide) at the toy scale, 0.25 Jy."""
    r0 = (npix - 1) / 2
    rr = np.hypot(*(np.indices((npix, npix)) - r0)) * PIX
    return vy.truth_image(np.exp(-0.5 * ((rr - 120) / 20) ** 2) + 0.02, PIX, blur_mas=15.0)


@pytest.mark.validates("virgil_bridge.alma_synth", roots=["mathematics"])
def test_truth_flux_and_model_visibilities_match_an_independent_dft(syn):
    """The synthetic data's forward model against ``crosscheck.sky`` (a numpy pixel DFT built from the documented conventions: row 0 North, half-pixel shift -dx/2), to 1e-10: a wrong half pixel,
    row order or Fourier sign in the model that makes the data would otherwise cancel in every arm."""
    from crosscheck import sky

    truth = _truth()
    assert truth.sum() * (PIX / 1000) ** 2 == pytest.approx(0.25, rel=1e-12)
    data = va.load_dir(syn)
    m = vy.model_visibilities(data, truth, PIX)
    ref = 0.25 * sky.visibility(sky.pixel_image(truth, PIX, -PIX / 2, -PIX / 2), np.asarray(data.u), np.asarray(data.v), 1.0)
    assert np.abs(m - ref).max() / np.abs(ref).max() < 1e-10
    mirrored = 0.25 * sky.visibility(sky.pixel_image(np.roll(truth, 6, axis=1), PIX, -PIX / 2, -PIX / 2), np.asarray(data.u), np.asarray(data.v), 1.0)
    assert np.abs(m - mirrored).max() / np.abs(ref).max() > 1e-3  # the comparison can fail: a shifted image is not the model
    d2 = vy.noisy(m, np.asarray(data.weight), np.random.default_rng(0))
    z = (d2 - m) * np.sqrt(np.asarray(data.weight))
    assert abs(z.real.std() - 1) < 0.1 and abs(z.imag.std() - 1) < 0.1  # complex noise of variance 1 / w per component


@pytest.mark.validates("virgil_bridge.alma_synth", roots=["self-consistency"], kind="control")
def test_score_image_separates_truth_from_dotted_and_zero_error_for_itself():
    truth = _truth()
    beam = (60.0, 40.0, 30.0)
    s0 = vy.score_image(truth, truth, PIX, beam)
    assert s0["nrmse"]["0.5"] == pytest.approx(0.0, abs=1e-12) and s0["flux_error"] == pytest.approx(0.0, abs=1e-12) and s0["n_eff_ratio"] > 0.7
    rng = np.random.default_rng(1)
    dots = truth * (rng.uniform(size=truth.shape) < 0.05) / 0.05  # same mean, dotty
    s1 = vy.score_image(dots, truth, PIX, beam)
    assert s1["n_eff_ratio"] < 0.5 < s0["n_eff_ratio"] and s1["nrmse"]["0.5"] > 0.05


@pytest.mark.validates("virgil_bridge.alma_synth", roots=["self-consistency"])
def test_synthetic_truth_has_no_dots_in_the_mem_arm_but_r_with_a_wide_blur_has_fewer_effective_pixels(syn, tmp_path):
    """The CI regression of Track A: smooth truth at the toy uv; the MEM arm (config F, ``--reg mem``) keeps the image's N_eff ratio >= 0.7, and R with a 30 mas kernel in the model (the dot
    mechanism) ends clearly lower."""
    truth = _truth(48)
    files = vy.make_data(va.ddid_files(syn), tmp_path / "synth", truth, PIX, seed=1)
    assert len(files) == 14
    arms = {}
    for name, extra in (("mem", ["--config", "F", "--reg", "mem:1e-9"]), ("r30", ["--config", "R", "--taper-mas", 30, "--lam-m", 1e-9])):
        out = tmp_path / name
        _call("fit", *extra, "--data", tmp_path / "synth", "--out", out, "--npix", 48, "--pixel-mas", PIX, "--max-steps", 200, "--no-dirty")
        arms[name] = vy.score_image(np.load(out / "image.npy").astype(float), truth, PIX, (60.0, 40.0, 30.0), taper_mas=30.0)
    assert arms["mem"]["n_eff_ratio"] >= 0.7, arms["mem"]["n_eff_ratio"]
    assert arms["r30"]["n_eff_ratio"] < 0.7, arms["r30"]["n_eff_ratio"]  # R really fails the dots criterion (N_eff ratio >= 0.7), not only "lower than MEM"
    assert arms["r30"]["n_eff_ratio"] < arms["mem"]["n_eff_ratio"] - 0.1, (arms["r30"]["n_eff_ratio"], arms["mem"]["n_eff_ratio"])


# ------------------------------------------------------------------ the replica


@pytest.mark.validates("virgil_bridge.alma_replica", roots=["mathematics"])
def test_replica_entropy_equals_mpols_formula_on_a_toy_cube():
    rng = np.random.default_rng(4)
    cube = np.exp(rng.normal(size=(32, 32)) - 2.0)
    got = float(vr.mpol_entropy(jnp.asarray(cube)))
    assert got == pytest.approx(cx.mpol_entropy(cube, 1e-4, 0.253), rel=1e-12)
    written = float(1 / 0.253 * np.sum(cube * np.log(cube / 1e-4)))  # mpol 0.3.1 losses.entropy: 1 / tot_flux * torch.sum(cube * torch.log(cube / prior_intensity))
    assert got == pytest.approx(written, rel=1e-12)
    try:  # and the package itself, where it can be imported
        import torch
        from mpol import losses

        assert got == pytest.approx(float(losses.entropy(torch.tensor(cube), 1e-4, 0.253)), rel=1e-10)
    except ImportError:
        pass
    blur = np.asarray(vr.gauss_blur(jnp.asarray(cube), 30.0, PIX))
    assert np.allclose(blur, cx.convolve(cube, PIX, 30.0), atol=1e-12 * cube.max())


@pytest.mark.validates("virgil_bridge.alma_replica", roots=["self-consistency"], kind="control")
def test_replica_adamw_decays_the_parameters_and_its_absence_does_not():
    import optax

    params = {"base": jnp.full((4, 4), 2.0)}
    zero = {"base": jnp.zeros((4, 4))}
    for wd, expect in ((1e-2, 2.0 * (1 - 1e-2 * 1e-2)), (0.0, 2.0)):
        opt = vr.make_optimizer(1e-2, wd)
        upd, _ = opt.update(zero, opt.init(params), params)  # zero gradient: only the decoupled decay moves the parameters
        assert float(optax.apply_updates(params, upd)["base"][0, 0]) == pytest.approx(expect, rel=1e-6)


@pytest.mark.validates("virgil_bridge.alma_replica", roots=["self-consistency"], kind="guard")
def test_replica_compare_passes_mpols_own_image_on_the_numbers_it_can_and_fails_noise():
    n = 64
    rr = np.hypot(*(np.indices((n, n)) - (n - 1) / 2)) * PIX
    base = 10.0 * np.exp(-0.5 * ((rr - 100) / 25) ** 2)
    blurred = vr.blurred_of(base, PIX)
    res = vr.replica_compare(base, blurred, base, blurred, 0.294, mpol_flux=0.294, pixel_mas=PIX)
    items = {i["key"]: i for i in res["items"]}
    assert items["a"]["status"] == "PASS" and items["c"]["status"] == "PASS" and items["d"]["status"] == "PASS"
    noise = np.random.default_rng(0).uniform(size=(n, n))
    bad = vr.replica_compare(noise, vr.blurred_of(noise, PIX), base, blurred, 0.294, pixel_mas=PIX)
    assert bad["verdict"] == "FAIL" and {i["key"]: i["status"] for i in bad["items"]}["a"] == "FAIL"
    assert vr.steps_for(10_883_819) == 10 * 273 and vr.steps_for(42_421_131) == 10 * 1061


@pytest.mark.validates("virgil_bridge.alma_replica", roots=["self-consistency"])
def test_replica_workflow_runs_end_to_end_and_ablation_flags_change_the_run(syn, tmp_path):
    outs = {}
    for name, extra in (("base", []), ("nowd", ["--replica-no-wd"]), ("softmax", ["--replica-softmax"])):
        out = tmp_path / name
        _call("fit", "--config", "R", "--taper-mas", 30, "--workflow", "mpol-replica", "--replica-stages", "all", "--replica-steps", 200, *extra, "--data", syn, "--out", out, "--npix", 48,
              "--pixel-mas", PIX, "--no-dirty")
        outs[name] = np.load(out / "image.npy")
        assert np.all(np.isfinite(outs[name])) and outs[name].min() >= 0
    assert not np.allclose(outs["base"], outs["nowd"]) and not np.allclose(outs["base"], outs["softmax"])


@pytest.mark.validates("virgil_bridge.alma_replica", roots=["self-consistency"], kind="guard")
def test_replica_offsets_may_leave_the_default_toeplitz_box_but_not_the_replicas(syn):
    """MPoL's offsets are free (AdamW at lr 1e-2 arcsec: 10 mas a step), so the replica's Toeplitz box must contain wherever they go (job 19396964 died on the 100 mas box). From a start with
    offsets of 0.5 arcsec, past both the 20 mas default and the old 100 mas box, steps run in the replica's box with the peak reported, and a 20 mas box raises."""
    import equinox as eqx

    data = va.load_dir(syn)
    n_obs, n_ddid = len(np.unique(np.asarray(data.obs))), int(np.asarray(data.ddid).max()) + 1

    def run(bound):
        prob = va.build_problem(data, "R", lam=va.lam_r(len(data.u)), npix=32, pixel_mas=PIX, chi2="toeplitz", offset_bound_mas=bound)
        params = vr.start_params(prob, n_obs, n_ddid)
        params["delta"] = jnp.full_like(params["delta"], 0.5)
        kw = dict(n_points=len(data.u), obs_of_ddid=jnp.asarray(vr.obs_of_ddid_of(data)), pixel_mas=PIX, lam_m=1e-9, blur_mas=30.0, entropy_blur_mas=30.0, softmax=False)
        stats = {}
        _, losses, _, stopped = vr.run_stage(params, prob.terms[0], steps=3, opt=vr.make_optimizer(), loss_kw=kw, chunk=3, stats=stats)
        return losses, stopped, stats

    losses, stopped, stats = run(script.REPLICA_OFFSET_BOUND_MAS)
    assert stopped is None and np.all(np.isfinite(losses)) and stats["peak_offset_mas"] > 400.0
    with pytest.raises(eqx.EquinoxRuntimeError, match="beyond the bound"):
        run(va.TOEPLITZ_OFFSET_BOUND_MAS)


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_cv_task_gp_arm_runs_on_the_same_folds_and_keys_the_grid_by_length(syn, tmp_path):
    """The GP in cross-validation: ``--config G`` with a hyperparameter point, from a start of the problem's own defaults; cv.json carries length_r as ``lam`` and both lengths in ``key``."""
    gp = dict(length_r=20.0, length_theta=1.0, npix=48, pixel_mas=PIX, n_r=10, n_theta=16, r_max_mas=120.0)
    prob = vg.build_gp_problem(va.load_dir(syn), chi2="toeplitz", **gp)
    init = tmp_path / "init"
    init.mkdir()
    start = {k: np.asarray(v) for k, v in prob.init.items()}
    start["field.latent"] = np.zeros((10, 16))
    start["field.sigma"] = np.asarray(1.0)
    np.savez(init / "values.npz", **start)
    folds = tmp_path / "folds.npz"
    _call("folds", "--data", syn, "--out", folds, "--k", 2)
    out = tmp_path / "g0"
    _call("cv-task", "--config", "G", "--length-r-mas", 20, "--length-theta-rad", 1.0, "--r-max-mas", 120, "--n-r", 10, "--n-theta", 16, "--data", syn, "--folds", folds, "--fold", 0, "--init", init,
          "--npix", 48, "--max-steps", 3, "--out", out)
    r = json.loads((out / "cv.json").read_text())
    assert r["lam"] == 20.0 and r["key"] == "lr20_lt1" and r["arm"]["n_r"] == 10 and np.isfinite(r["heldout"]["nll"])


@pytest.mark.validates("virgil_bridge.alma_synth", roots=["self-consistency"], kind="guard")
def test_synth_script_make_run_score_and_the_better_and_replica_compare_commands(syn, tmp_path):
    import alma_imlup_synth as synth

    truth_img = tmp_path / "base.npy"
    np.save(truth_img, _truth(48) + 0.0)
    out = tmp_path / "s"
    synth.main(["make", "--data", str(syn), "--out", str(out), "--truth-image", str(truth_img), "--truth-blur-mas", "0", "--flux", "0.25", "--pixel-mas", str(PIX)])
    assert (out / "truth.npy").exists() and len(list(out.glob("ddid_*.npz"))) == 14
    fit = tmp_path / "fit"
    synth.main(["run", "--arm", "maxent", "--data", str(out), "--truth", str(out / "truth.npy"), "--out", str(fit), "--beam", "60", "40", "30", "--", "--config", "F", "--reg", "mem:1e-9", "--npix", "48",
                "--pixel-mas", str(PIX), "--max-steps", "10", "--no-dirty"])
    sc = json.loads((fit / "synth.json").read_text())
    assert set(sc["nrmse"]) == {"0.5", "1"} and np.isfinite(sc["flux_error"]) and sc["n_eff_ratio"] > 0
    _call("better", "--fit", fit, "--synth", fit / "synth.json", "--synth-r", fit / "synth.json", "--out", tmp_path / "b")
    b = json.loads((tmp_path / "b" / "better_than_mpol.json").read_text())
    assert b["verdict"] in ("FAIL", "INCOMPLETE") and {i["key"]: i["status"] for i in b["items"]}["1"] == "NOT EVALUATED"
    mp = tmp_path / "mp"
    mp.mkdir()
    img = np.load(fit / "image.npy").astype(float)
    np.savez(mp / "export.npz", base_image=img, image=vr.blurred_of(img, PIX), cell_size_arcsec=PIX / 1000.0)
    _call("replica-compare", "--fit", fit, "--mpol", mp, "--out", tmp_path / "rc")
    rc = json.loads((tmp_path / "rc" / "replica_compare.json").read_text())
    assert {i["key"]: i["status"] for i in rc["items"]}["a"] == "PASS"


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_cv_aggregate_refuses_a_mix_of_arms_even_with_distinct_grid_points(syn, tmp_path):
    folds = tmp_path / "folds.npz"
    _call("folds", "--data", syn, "--out", folds, "--k", 2)
    init = tmp_path / "init"
    _call("fit", "--config", "F", "--reg", "mem:1e-9", "--chi2", "toeplitz", "--data", syn, "--out", init, "--npix", 48, "--max-steps", 3, "--no-dirty")
    cv = tmp_path / "cv"
    for j in (0, 1):
        for reg in ("mem:1e-9", "tsv:1e-4"):  # two regulariser grids, no lam in common
            _call("cv-task", "--config", "F", "--reg", reg, "--data", syn, "--folds", folds, "--fold", j, "--init", init, "--npix", 48, "--max-steps", 3, "--out", cv / f"{reg}_{j}")
    with pytest.raises(SystemExit, match="more than one arm"):
        _call("cv-aggregate", cv)


@pytest.mark.validates("virgil_bridge.alma_replica", roots=["self-consistency"], kind="guard")
def test_replica_compare_warns_when_the_export_is_not_the_p3a_run():
    n = 64
    rr = np.hypot(*(np.indices((n, n)) - (n - 1) / 2)) * PIX
    base = 10.0 * np.exp(-0.5 * ((rr - 100) / 25) ** 2)
    other = 3.0 * base  # another MPoL run: peak 3x outside the band
    res = vr.replica_compare(base, vr.blurred_of(base, PIX), other, vr.blurred_of(other, PIX), 0.294, mpol_flux=1.0, pixel_mas=PIX)
    assert res["warning"] and res["mpol"]["export_in_band"]["peak_base"] is False and "Warning" in "\n".join(vr.replica_markdown(res))


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_blur_mas_alias_zero_and_none_and_the_flag_conflict(syn, tmp_path):
    data = va.load_dir(syn)
    assert va.build_problem(data, "R", lam=1e-5, npix=48, blur_mas=0).nuis.blur_fwhm_mas is None  # 0: classic MEM, no kernel
    assert va.build_problem(data, "R", lam=1e-5, npix=48, blur_mas=None).nuis.blur_fwhm_mas == va.MPOL["taper_mas"]  # None: MPoL's 30 mas
    assert va.build_problem(data, "R", lam=1e-5, npix=48, taper_mas=12.0).nuis.blur_fwhm_mas == 12.0  # the old keyword
    assert va.build_problem(data, "R", lam=1e-5, npix=48, blur_mas=7.0, taper_mas=12.0).nuis.blur_fwhm_mas == 7.0
    outs = {}
    for flag in ("--blur-mas", "--taper-mas"):
        out = tmp_path / flag.strip("-")
        _call("fit", "--config", "R", flag, 20, "--data", syn, "--out", out, "--npix", 32, "--max-steps", 3, "--no-dirty")
        outs[flag] = json.loads((out / "result.json").read_text())["args"]["taper_mas"]
    assert outs["--blur-mas"] == outs["--taper-mas"] == 20.0
    with pytest.raises(SystemExit):
        _call("fit", "--config", "R", "--blur-mas", 20, "--taper-mas", 20, "--data", syn, "--out", tmp_path / "both", "--npix", 32)


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_cv_task_refuses_lam_with_reg_and_replica_refuses_the_other_engines(syn, tmp_path):
    folds = tmp_path / "folds.npz"
    _call("folds", "--data", syn, "--out", folds, "--k", 2)
    with pytest.raises(SystemExit, match="no --lam"):
        _call("cv-task", "--config", "F", "--reg", "mem:1e-9", "--lam-m", 1e-9, "--data", syn, "--folds", folds, "--fold", 0, "--init", tmp_path, "--npix", 32, "--out", tmp_path / "x")
    with pytest.raises(SystemExit, match="do not apply"):
        _call("fit", "--config", "R", "--workflow", "mpol-replica", "--method", "adam", "--data", syn, "--out", tmp_path / "y", "--npix", 32)


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_old_virgil_is_refused_with_the_required_version(monkeypatch):
    import virgil

    class Old:
        def __init__(self, data, taper_fwhm_mas=None):
            pass

    monkeypatch.setattr(virgil, "CalibrationNuisance", Old)
    with pytest.raises(RuntimeError, match="884aa7e"):
        va.require_blur_fwhm()
