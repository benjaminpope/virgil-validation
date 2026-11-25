"""The numpy side of the production DSHARP driver (``scripts/dsharp_fit.py``): the model-free weights, the spectral index, the shift estimate and the
pre-alignment rule, the lambda pick, the starts and the optimiser interface. Nothing here imports virgil or needs a GPU: the production model's
agreement with the direct engine (gate G1) is checked by the S1 probe job, not here."""

import importlib.util
import pathlib
import subprocess
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAS = np.pi / 180.0 / 3600.0 / 1000.0


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fit = _script("dsharp_fit")


def _window(rng, ddid, obsid, s=2.0, n_ant=6, n_t=40, freq=2.3e11, noise_of_q=None, vis_of_uv=None, shift_mas=(0.0, 0.0), w0=4.0):
    """One ddid file with the time, antenna and scan columns: baselines of ``n_ant`` antennas over ``n_t`` integrations, noise ``s / sqrt(w)`` per component."""
    ant1, ant2 = np.triu_indices(n_ant, 1)
    nb = len(ant1)
    u0, v0 = rng.uniform(-3e5, 3e5, nb), rng.uniform(-3e5, 3e5, nb)
    t = np.repeat(np.arange(n_t), nb)
    a1, a2 = np.tile(ant1, n_t), np.tile(ant2, n_t)
    u, v = np.tile(u0, n_t), np.tile(v0, n_t)
    w = np.full(len(u), w0)
    sig = s / np.sqrt(w)
    if noise_of_q is not None:
        sig = sig * noise_of_q(np.hypot(u, v))
    vis = (1.0 if vis_of_uv is None else vis_of_uv(u, v)) * np.ones(len(u), complex)
    vis = vis * np.exp(-2j * np.pi * (shift_mas[0] * MAS * u + shift_mas[1] * MAS * v))
    vis = vis + sig * (rng.normal(size=len(u)) + 1j * rng.normal(size=len(u)))
    return dict(u=u, v=v, vis=vis, weight=w, freq=np.full(len(u), freq), ddid=ddid, obsid=obsid, time=t.astype(float), ant1=a1, ant2=a2,
                chan=np.zeros(len(u), int), scan=(t // 10).astype(int))


@pytest.mark.validates("crosscheck.dsharp", roots=["self-consistency"], kind="guard")
def test_pair_scatter_recovers_the_error_scale():
    z = _window(np.random.default_rng(1), 0, 0, s=2.0)
    q, z2, scan = fit.pair_scatter(z)
    assert abs(z2.mean() - 4.0) < 0.3  # s^2 = 4; no sky variation inside a pair
    assert len(q) == len(z2) == len(scan) and len(z2) == 15 * 20


@pytest.mark.validates("crosscheck.dsharp", roots=["self-consistency"], kind="guard")
def test_pair_scatter_needs_the_time_and_antenna_columns():
    z = _window(np.random.default_rng(1), 0, 0)
    for k in ("time", "ant1", "ant2"):
        assert fit.pair_scatter({a: b for a, b in z.items() if a != k}) is None


@pytest.mark.validates("crosscheck.dsharp", roots=["self-consistency"], kind="guard")
def test_weights_are_rescaled_so_that_s_is_one_and_binned_when_the_scatter_depends_on_baseline():
    rng = np.random.default_rng(2)
    flat = _window(rng, 0, 0, s=3.0)
    tilt = _window(rng, 1, 0, s=3.0, noise_of_q=lambda q: 1.0 + 1.5 * q / 4e5)  # noise grows with baseline length
    plan = fit.weight_plan([flat, tilt])
    assert not plan[0]["per_bin"] and plan[1]["per_bin"] and plan[1]["bin_spread"] > fit.BIN_VARIATION
    assert plan[0]["s2"] == pytest.approx(9.0, rel=0.1)
    after = fit.apply_weights([flat, tilt], plan)
    for z in after:  # the rescaled weights have unit scatter, window by window and bin by bin
        q, z2, _ = fit.pair_scatter(z)
        assert z2.mean() == pytest.approx(1.0, rel=1e-9)
    q, z2, _ = fit.pair_scatter(after[1])
    b = np.searchsorted(np.array(plan[1]["edges"]), q)
    assert all(z2[b == k].mean() == pytest.approx(1.0, rel=1e-9) for k in range(fit.N_BINS))
    assert flat["weight"][0] == 4.0  # the input is not modified


@pytest.mark.validates("crosscheck.dsharp", roots=["self-consistency"], kind="guard")
def test_without_pair_columns_the_shat_stands_in_and_nothing_is_binned():
    z = {k: v for k, v in _window(np.random.default_rng(3), 0, 0).items() if k not in ("time", "ant1", "ant2")}
    plan = fit.weight_plan([z], shat=np.array([1.5]))
    assert plan[0]["s2"] == 2.25 and plan[0]["source"] == "shat" and not plan[0]["per_bin"]
    with pytest.raises(ValueError, match="no --shat"):
        fit.weight_plan([z])


@pytest.mark.validates("crosscheck.dsharp", roots=["self-consistency"], kind="guard")
def test_the_spectral_index_is_measured_with_a_free_level_per_observation_and_fixed_only_when_significant():
    rng = np.random.default_rng(4)
    zs = []
    for obsid, level in ((0, 1.0), (1, 1.07)):  # a gain difference between observations is not an index
        for d, nu in enumerate((2.25e11, 2.30e11, 2.35e11, 2.40e11)):
            z = _window(rng, 10 * obsid + d, obsid, s=1.0, freq=nu, w0=1e6, n_ant=10)
            z["vis"] = z["vis"] * level * (nu / 2.3e11) ** 2.0  # alpha = 2
            zs.append(z)
    a = fit.spectral_index(zs)
    assert a["alpha"] == pytest.approx(2.0, abs=4 * a["sigma"] + 0.05) and a["used"] and a["alpha_fixed"] == a["alpha"]
    flat = [dict(z, vis=z["vis"] / (np.asarray(z["freq"]) / 2.3e11) ** 2.0) for z in zs]
    b = fit.spectral_index(flat)
    assert abs(b["alpha"]) < ds_sigma(b) * 3 + 0.1
    one = fit.spectral_index([zs[0], dict(zs[4])])  # one window per observation: nothing to fit
    assert one["alpha"] is None and one["alpha_fixed"] == 0.0 and not one["used"]


def ds_sigma(a):
    return a["sigma"]


def _disc(u, v):
    r = np.hypot(u, v)
    return np.exp(-0.5 * (r / 3e5) ** 2)  # a real, smooth visibility: a centred Gaussian source


@pytest.mark.validates("crosscheck.dsharp", roots=["self-consistency"], kind="guard")
def test_the_model_free_shift_recovers_a_known_offset_and_prealign_undoes_it():
    rng = np.random.default_rng(5)
    shift = (7.0, -4.0)
    ref = [_window(rng, 0, 0, s=0.02, n_ant=10, n_t=60, vis_of_uv=_disc)]
    mov = [_window(rng, 1, 1, s=0.02, n_ant=10, n_t=60, vis_of_uv=_disc, shift_mas=shift)]
    # the two observations share uv cells because the same baselines recur
    mov[0]["u"], mov[0]["v"] = ref[0]["u"].copy(), ref[0]["v"].copy()
    mov[0]["vis"] = _disc(mov[0]["u"], mov[0]["v"]) * np.exp(-2j * np.pi * (shift[0] * MAS * mov[0]["u"] + shift[1] * MAS * mov[0]["v"]))
    mov[0]["vis"] = mov[0]["vis"] + 0.01 * (rng.normal(size=len(mov[0]["u"])) + 1j * rng.normal(size=len(mov[0]["u"])))
    est = fit.model_free_shift(ref + mov, 1, 0)
    assert est["shift_mas"] == pytest.approx(list(shift), abs=max(1.0, 4 * max(est["error_mas"])))
    aligned = fit.prealign(ref + mov, {1: est["shift_mas"]})
    after = fit.model_free_shift(aligned, 1, 0)
    assert np.all(np.abs(after["shift_mas"]) < 1.0)
    assert np.all(aligned[0]["vis"] == ref[0]["vis"])  # the reference is untouched
    assert mov[0]["vis"][0] != aligned[1]["vis"][0]  # and the input is not modified in place


@pytest.mark.validates("crosscheck.dsharp", roots=["self-consistency"], kind="guard")
def test_the_model_free_shift_is_none_without_overlapping_cells():
    rng = np.random.default_rng(6)
    a, b = _window(rng, 0, 0, vis_of_uv=_disc), _window(rng, 1, 1, vis_of_uv=_disc)
    b["u"] = b["u"] + 1e9
    assert fit.model_free_shift([a, b], 1, 0)["shift_mas"] is None


@pytest.mark.validates("crosscheck.dsharp", roots=["self-consistency"], kind="guard")
def test_the_offset_rule_prealigns_when_the_estimates_agree_and_treats_a_disagreement_as_a_runaway():
    p = fit.offset_plan({1: (8.0, 1.0), 2: (2.0, 2.0)}, {0: (0.0, 0.0), 1: (9.5, 1.2), 2: (1.7, 50.49)}, ref=0)
    assert p["per_obs"][1]["action"] == "prealign" and p["shifts_mas"] == {1: [8.0, 1.0]}
    assert p["per_obs"][1]["residual_mas"] == pytest.approx([1.5, 0.2])
    r = p["per_obs"][2]  # RU Lup's observation 2: the fit sits at 50.49 mas, the model-free shift says 2 mas
    assert r["action"] == "none" and r["outside_box"] and p["drop"] == [2]
    assert 0 not in p["per_obs"]


@pytest.mark.validates("crosscheck.dsharp", roots=["self-consistency"], kind="guard")
def test_a_confirmed_large_shift_is_prealigned_so_its_residual_fits_the_box_but_an_unresolved_one_drops_the_disc():
    ok = fit.offset_plan({1: (49.0, 0.0)}, {1: (50.0, 0.5)}, ref=0)
    assert ok["per_obs"][1]["action"] == "prealign" and ok["drop"] == [] and ok["shifts_mas"][1] == [49.0, 0.0]
    gone = fit.offset_plan({1: None}, {1: (30.0, 0.0)}, ref=0)  # no model-free shift to confirm it
    assert gone["per_obs"][1]["action"] == "none" and gone["drop"] == [1]
    near = fit.offset_plan({1: (0.0, 0.0)}, {1: (19.5, 0.0)}, ref=0)
    assert near["drop"] == [] and near["near_box"] == [1]


@pytest.mark.validates("crosscheck.dsharp", roots=["self-consistency"], kind="guard")
def test_the_one_standard_error_rule_takes_the_strongest_weight_within_the_tolerance():
    rungs = [1e6, 3e5, 1e5, 3e4, 1e4, 3e3]
    n = 2500  # null-spread fallback 2 sqrt(n) = 100 (chi^2 sums two components a point)
    s = fit.one_se_pick(rungs, [900.0, 520.0, 410.0, 400.0, 405.0, 520.0], n)
    assert s["weight"] == 1e5 and s["best"] == 3 and not s["flat"] and s["tolerance"] == pytest.approx(100.0)
    f = fit.one_se_pick(rungs, [1000.0, 1010.0, 1005.0, 1000.0, 1003.0, 1002.0], n)
    assert f["weight"] == 1e6 and f["flat"]  # the data did not choose: the smoothest rung, and the flag says so
    f3 = np.array([[300, 300, 300], [170, 175, 175], [137, 137, 136], [133, 134, 133], [135, 135, 135], [173, 174, 173]], float)  # rung x fold
    c = fit.one_se_pick(rungs, f3.sum(axis=1), n, f3)  # standard error of the summed difference from the spread over folds
    assert c["basis"] == "folds" and c["best"] == 3 and c["weight"] == 3e4
    e = fit.one_se_pick(rungs, [600.0, 520.0, 410.0, 400.0, 405.0, 300.0], n)
    assert e["weight"] == 3e4 and e["best"] == 5  # the minimum is the weakest rung; the rule still prefers the smoother of the rungs within tolerance


@pytest.mark.validates("crosscheck.dsharp", roots=["self-consistency"], kind="guard")
def test_the_ladder_is_strongest_first_and_ends_near_the_inherited_weight():
    assert list(fit.LADDER) == sorted(fit.LADDER, reverse=True) and fit.LADDER[0] == 1e6 and fit.LADDER[-1] == 3e3
    assert fit.TEST_FOLD not in fit.TUNING_FOLDS and sorted(fit.TUNING_FOLDS + (fit.TEST_FOLD,)) == list(range(fit.K_FOLDS))


@pytest.mark.validates("crosscheck.dsharp", roots=["self-consistency"], kind="guard")
def test_the_flat_start_is_strictly_positive_and_carries_the_flux_and_the_gaussian_start_is_floored():
    x = fit.flat_pixels(32, 0.25)
    assert np.all(x > 0) and np.sum(x**2) == pytest.approx(0.25)
    g = fit.gaussian_floor_pixels(64, 50.0, 0.001)  # a 3.2 arcsec field: the far Gaussian is below the 1e-4 Jy/arcsec^2 floor
    img = g**2 / (50.0 / 1000.0) ** 2
    assert np.sum(g**2) == pytest.approx(0.001) and 0.5e-4 < img.min() <= 1e-4  # at the entropy floor in the far field (renormalised to the flux)
    assert img.max() == pytest.approx(img[31, 31])


@pytest.mark.validates("crosscheck.dsharp", roots=["self-consistency"], kind="guard")
def test_the_engine_is_read_from_the_term_not_from_the_request():
    class _ToeplitzTerm:
        pass

    class _RadioTerm:
        pass

    assert fit.engine_of(_ToeplitzTerm()) == "toeplitz" and fit.engine_of(_RadioTerm()) == "direct"


@pytest.mark.validates("crosscheck.dsharp", roots=["self-consistency"], kind="guard")
def test_the_optimiser_is_behind_an_interface_and_resolves_a_name_or_module_function():
    assert fit.resolve_callable("lbfgs", fit.OPTIMISERS) is fit.lbfgs_optimiser
    assert fit.resolve_callable("os.path:join") is __import__("os").path.join
    with pytest.raises(ValueError, match="neither a known name"):
        fit.resolve_callable("nonsense", fit.OPTIMISERS)


@pytest.mark.validates("crosscheck.dsharp", roots=["self-consistency"], kind="guard")
def test_the_problem_builder_refuses_chi2_auto():
    with pytest.raises(ValueError, match="never uses 'auto'"):
        fit.build_production_problem(None, npix=8, pixel_mas=5.0, weight=1.0, ref_obs=0, chi2="auto")


@pytest.mark.validates("crosscheck.dsharp", roots=["self-consistency"], kind="guard")
@pytest.mark.parametrize("mode", ["probe", "s3", "s4", "s5"])
def test_the_run_modes_exist_and_a_run_mode_needs_the_probe(mode, tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "dsharp_fit.py"), mode, "--help"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "--probe" in r.stdout
    d = tmp_path / "d"
    d.mkdir()
    np.savez(d / "ddid_00.npz", u=np.zeros(2), v=np.zeros(2))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "dsharp_fit.py"), mode, "--target", "imlup", "--data", str(d), "--out", str(tmp_path / "o")],
                       capture_output=True, text=True, timeout=60)
    if mode != "probe":
        assert r.returncode != 0 and "needs --probe" in (r.stderr + r.stdout)
    else:
        assert r.returncode != 0  # the probe needs real ddid arrays; this 2-point file is refused, not fitted


@pytest.mark.validates("crosscheck.dsharp", roots=["self-consistency"], kind="guard")
def test_a_resolved_achromatic_source_gives_no_spectral_slope():
    # baselines are fixed in metres, so q in wavelengths grows with frequency; V(q) falls with q. A per-window shortest-1% cut would read that as a slope.
    rng = np.random.default_rng(5)
    zs = []
    for d, nu in enumerate((2.0e11, 2.2e11, 2.4e11, 2.6e11)):
        z = _window(rng, d, 0, s=1.0, freq=nu, w0=1e8, n_ant=40, n_t=2)
        z["u"], z["v"] = z["u"] * nu / 2.3e11, z["v"] * nu / 2.3e11
        z["vis"] = np.exp(-(np.hypot(z["u"], z["v"]) / 1.2e5) ** 2) * np.ones(len(z["u"]), complex)
        zs.append(z)
    a = fit.spectral_index(zs)
    assert abs(a["alpha"]) < 0.05, a


@pytest.mark.validates("crosscheck.dsharp", roots=["self-consistency"], kind="guard")
def test_the_engines_agree_on_a_tiny_problem_at_the_pinned_virgil():
    pytest.importorskip("virgil.radio_toeplitz")  # the pin of the convergence plan; older virgils have no explicit Toeplitz term
    import jax

    jax.config.update("jax_enable_x64", True)
    import virgil_bridge.alma as va

    rng = np.random.default_rng(0)
    z = _window(rng, 0, 0, s=1.0, n_ant=8, n_t=4)
    z["u"], z["v"] = z["u"] / 20, z["v"] / 20
    z["vis"] = np.exp(-(np.hypot(z["u"], z["v"]) / 2e4) ** 2) * np.ones(len(z["u"]), complex)
    e = fit.engine_agreement(va.load_npz([z]), dict(npix=16, pixel_mas=40.0, weight=1.0, ref_obs=0, box_mas=20.0, flux=1.0), log=lambda *a: None)
    assert e["engine_toeplitz"] == "toeplitz" and e["engine_direct"] == "direct" and e["pass_loss"] and e["pass_grad"]
