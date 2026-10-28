"""``fit --chi2 toeplitz`` of ``scripts/alma_imlup_maxent.py`` on a small synthetic IM Lup-like set (CPU, seconds): the Toeplitz term of the fit's
problem equals the direct one in loss and gradient, the engine and the precompute are recorded, ``at_bound`` is reported, and a problem Toeplitz
cannot take exactly is refused. Skipped on a virgil without ``radio_toeplitz``."""

import json
import pathlib
import sys

import numpy as np
import pytest

pytest.importorskip("virgil.radio_toeplitz")

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import virgil_bridge.alma as va  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import alma_imlup_maxent as script  # noqa: E402
from test_alma_maxent import synthetic_dir  # noqa: E402

NPIX = 32


def _call(*argv):
    script.main([str(a) for a in argv])


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    return synthetic_dir(tmp_path_factory.mktemp("syn"), n=150, n_obs=3, per_obs=2)


def _loss(term, values):
    """-ln L of a data term, up to its constant: 0.5 chi^2 + the log normalisation, from either engine's own API."""
    if hasattr(term, "chi2_and_log_norm"):
        chi2, log_norm = term.chi2_and_log_norm(values)
        return 0.5 * chi2 + log_norm
    r, log_norm = term.residuals_and_log_norm(values)
    return 0.5 * jnp.sum(r**2) + log_norm


def _rel(a, b):
    return float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
@pytest.mark.parametrize("use_sb", [False, True])
def test_toeplitz_and_direct_terms_of_the_fit_problem_give_the_same_loss_and_gradient(syn, use_sb):
    """The two engines of ``build_problem`` at one set of off-start parameters (all data, and the SB-stage subset with a mask): loss to 1e-10 and every
    gradient leaf to 1e-10 of its own largest component; ``chi2_info`` records the engine and the precompute."""
    data = va.load_dir(syn)
    use = data.select(obs=[0, 1]) if use_sb else None
    mask = np.random.default_rng(5).uniform(size=len(use.u)) < 0.8 if use_sb else None
    lam = va.lam_r(len(data.u))
    direct = va.build_problem(data, "R", lam=lam, npix=NPIX, use=use, mask=mask)
    toep = va.build_problem(data, "R", lam=lam, npix=NPIX, use=use, mask=mask, chi2="toeplitz")
    assert direct.chi2_info == dict(engine="direct")
    ci = toep.chi2_info
    assert ci["engine"] == "toeplitz" and ci["precompute_s"] > 0 and ci["arrays_gb"] > 0 and ci["offsets"] == "spread" and ci["max_offset_mas"] == va.TOEPLITZ_OFFSET_BOUND_MAS
    assert toep.offset_bound_mas == va.TOEPLITZ_OFFSET_BOUND_MAS  # the prior box is the precompute's
    rng = np.random.default_rng(1)
    v = {k: jnp.asarray(x) for k, x in direct.init.items()}
    v["log_brightness"] = jnp.asarray(rng.normal(size=(NPIX, NPIX)))
    v["cal.flux"] = v["cal.flux"] * 1.3
    v["cal.gain"] = v["cal.gain"] * (1.0 + 0.03 * rng.normal(size=v["cal.gain"].shape))
    v["cal.offset"] = jnp.asarray(rng.uniform(-4, 4, size=v["cal.offset"].shape))
    v["cal.sigma"] = v["cal.sigma"] * (1.0 + 0.1 * rng.uniform(size=v["cal.sigma"].shape))
    vg = lambda term: eqx.filter_value_and_grad(lambda x: _loss(term, x))(v)  # noqa: E731
    with va.ft_eps(1e-12):
        ld, gd = vg(direct.terms[0])
        lt, gt = vg(toep.terms[0])
    assert float(abs(ld - lt) / abs(ld)) < 1e-10, (float(ld), float(lt))
    for k in gd:
        assert _rel(gt[k], gd[k]) < 1e-10, (k, _rel(gt[k], gd[k]))


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_toeplitz_without_fitted_offsets_needs_no_box(syn):
    """The image-first stage has no offsets: the precompute is built with ``offsets="none"`` and the prior box is untouched."""
    data = va.load_dir(syn)
    prob = va.build_problem(data, "R", lam=1.0, npix=NPIX, image_only=True, chi2="toeplitz")
    assert prob.chi2_info["offsets"] == "none" and prob.chi2_info["max_offset_mas"] is None and prob.offset_bound_requested is None


@pytest.fixture(scope="module")
def syn7(tmp_path_factory):
    return synthetic_dir(tmp_path_factory.mktemp("syn7"), n=150)  # 7 observations, so the SB stage (obs 0-4) exists


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="guard")
def test_fit_chi2_toeplitz_records_engine_precompute_and_at_bound_and_the_compare_notes_show_them(syn7, tmp_path):
    """A short two-stage fit with ``--chi2 toeplitz``: result.json holds the engine, a precompute per stage, its total time and array size, and
    ``at_bound``; the direct fit says ``direct`` with none; ``compare``'s notes carry both."""
    out = tmp_path / "t"
    _call("fit", "--config", "R", "--chi2", "toeplitz", "--data", syn7, "--out", out, "--npix", NPIX, "--max-steps", 5, "--no-dirty")
    r = json.loads((out / "result.json").read_text())
    assert r["complete"] and r["chi2_engine"] == "toeplitz" and r["args"]["chi2"] == "toeplitz"
    assert [s["stage"] for s in r["chi2_stages"]] == ["sb", "all"] and all(s["engine"] == "toeplitz" for s in r["chi2_stages"])
    assert r["toeplitz_precompute_s"] == pytest.approx(sum(s["precompute_s"] for s in r["chi2_stages"])) and r["toeplitz_arrays_gb"] > 0
    assert set(r["at_bound"]) == {"edge", "at_bound", "n_checked", "warning"} and r["at_bound"]["n_checked"] > 0
    assert all(h["site"] in ("cal.flux", "cal.offset") for h in r["at_bound"]["at_bound"])
    assert r["stages"][0]["chi2"]["engine"] == "toeplitz"
    md = "\n".join(script.fit_notes(r))
    assert "toeplitz" in md and "At a prior bound" in md
    d = tmp_path / "d"
    _call("fit", "--config", "R", "--data", syn7, "--out", d, "--npix", NPIX, "--max-steps", 5, "--no-dirty")
    rd = json.loads((d / "result.json").read_text())
    assert rd["chi2_engine"] == "direct" and rd["toeplitz_precompute_s"] is None and rd["args"]["chi2"] == "direct"
    assert r["args"]["offset_bound_mas"] == va.TOEPLITZ_OFFSET_BOUND_MAS and r["offset_bound"]["bound_mas"] == va.TOEPLITZ_OFFSET_BOUND_MAS
    _call("report", "--data", syn7, "--out", out, "--no-dirty")  # recomputes the report with the direct scorer, keeps the engine and the fitted box
    rr = json.loads((out / "result.json").read_text())
    assert rr["chi2_engine"] == "toeplitz" and rr["offset_bound"]["bound_mas"] == va.TOEPLITZ_OFFSET_BOUND_MAS


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="control")
def test_at_bound_flags_a_flux_and_an_offset_at_their_prior_bounds_and_only_those(syn):
    """Control: a flux at the log-uniform ceiling and one offset at its box edge are listed; interior values are not."""
    data = va.load_dir(syn)
    prob = va.build_problem(data, "R", lam=1.0, npix=NPIX, offset_bound_mas=50.0)
    v = {k: np.asarray(x) for k, x in prob.init.items()}
    assert va.at_bound_report(v, prob.priors)["at_bound"] == []
    v["cal.flux"], off = np.asarray(99.99), np.zeros((2, 2))
    off[1, 0] = -49.9
    v["cal.offset"] = off
    hits = va.at_bound_report(v, prob.priors)["at_bound"]
    assert {(h["site"], tuple(h["index"]), h["side"]) for h in hits} == {("cal.flux", (), "high"), ("cal.offset", (1, 0), "low")}


@pytest.mark.validates("virgil_bridge.alma", roots=["self-consistency"], kind="control")
def test_chi2_toeplitz_is_refused_with_offsets_it_cannot_take_and_for_config_f(syn, syn7, tmp_path):
    """Control: pixels so large that the baselines pass the Nyquist margin free offsets allow are refused with virgil's reason (the box is kept inside the
    precompute, so it is the baselines that refuse); config F is refused; ``--chi2 direct`` takes the same problem."""
    with pytest.raises(SystemExit, match=r"cannot take this problem exactly: .*Nyquist"):
        _call("fit", "--config", "R", "--chi2", "toeplitz", "--data", syn7, "--out", tmp_path / "a", "--npix", 16, "--pixel-mas", 60.0, "--max-steps", 2)
    with pytest.raises(SystemExit, match="config R, or F with --reg"):
        _call("fit", "--config", "F", "--chi2", "toeplitz", "--data", syn, "--out", tmp_path / "b", "--npix", 16)
    with pytest.raises(va.ToeplitzRefused, match=r"Nyquist"):
        va.build_problem(va.load_dir(syn), "R", lam=1.0, npix=16, pixel_mas=60.0, chi2="toeplitz")
    va.build_problem(va.load_dir(syn), "R", lam=1.0, npix=16, pixel_mas=60.0)  # the direct engine takes it
