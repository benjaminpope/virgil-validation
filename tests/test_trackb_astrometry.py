"""crosscheck.astrometry, the Track B comparison maths (design/trackb_criteria.md), against SciPy."""

import json
import math
import pathlib

import numpy as np
import pytest
from scipy import stats
from scipy.spatial import distance

from crosscheck import astrometry as A

TRACKB = pathlib.Path.home() / "data" / "eso_binaries" / "trackb"


@pytest.mark.validates("crosscheck.astrometry", roots=["mathematics"], kind="reference")
@pytest.mark.parametrize("smaj,smin,theta", [(0.3, 0.1, 0.0), (0.3, 0.1, 90.0), (0.05, 0.02, 37.0), (1.0, 1.0, -120.0)])
def test_ellipse_covariance_axes(smaj, smin, theta):
    cov = A.ellipse_covariance(smaj, smin, theta)
    w, v = np.linalg.eigh(cov)
    assert w == pytest.approx([smin**2, smaj**2], rel=1e-12)
    if smaj > smin:  # major axis along PA theta (East of North), up to 180 deg
        pa = math.degrees(math.atan2(v[0, 1], v[1, 1]))
        assert (pa - theta + 90) % 180 - 90 == pytest.approx(0.0, abs=1e-9)
    # the 1-sigma contour along the major axis has d^2 = 1
    t = math.radians(theta)
    assert A.mahalanobis2([smaj * math.sin(t), smaj * math.cos(t)], cov) == pytest.approx(1.0)


@pytest.mark.validates("crosscheck.astrometry", roots=["mathematics"], kind="reference")
def test_d2_matches_scipy_mahalanobis():
    rng = np.random.default_rng(1)
    for _ in range(20):
        a, b = rng.normal(size=2), rng.normal(size=2)
        m1, m2 = rng.normal(size=(2, 2)), rng.normal(size=(2, 2))
        c1, c2 = m1 @ m1.T + 0.1 * np.eye(2), m2 @ m2.T + 0.1 * np.eye(2)
        d2, s, delta = A.d2_compare(a, c1, b, c2)
        assert s == 1 and delta == pytest.approx(a - b)
        assert d2 == pytest.approx(distance.mahalanobis(a, b, np.linalg.inv(c1 + c2)) ** 2, rel=1e-10)


@pytest.mark.validates("crosscheck.astrometry", roots=["mathematics"], kind="reference")
def test_fold180_picks_the_nearer_image():
    cov = np.eye(2) * 0.01**2
    d2, s, delta = A.d2_compare([-3.0, 4.0], cov, [3.01, -4.0], cov, fold180=True)
    assert s == -1 and delta == pytest.approx([-0.01, 0.0])
    assert d2 == pytest.approx(0.01**2 / (2 * 0.01**2))
    d2u, su, _ = A.d2_compare([-3.0, 4.0], cov, [3.01, -4.0], cov)
    assert su == 1 and d2u > 1e5


@pytest.mark.validates("crosscheck.astrometry", roots=["statistics"], kind="reference")
def test_d2_is_chi2_2_for_gaussian_errors():
    rng = np.random.default_rng(2)
    c1 = A.ellipse_covariance(0.03, 0.01, 30.0)
    c2 = A.ellipse_covariance(0.02, 0.02, 0.0)
    truth = np.array([2.0, -1.0])
    a = rng.multivariate_normal(truth, c1, 4000)
    b = rng.multivariate_normal(truth, c2, 4000)
    d2 = np.array([A.d2_compare(x, c1, y, c2)[0] for x, y in zip(a, b)])
    assert stats.kstest(d2, stats.chi2(2).cdf).pvalue > 1e-3


@pytest.mark.validates("crosscheck.astrometry", roots=["mathematics"], kind="reference")
def test_constants_and_conversions():
    assert A.D2_FLAG == pytest.approx(2 * math.log(1000.0))  # chi^2_2 survival is exp(-x/2)
    half = A.FWHM_PER_SIGMA / 2
    assert stats.norm.pdf(half) == pytest.approx(stats.norm.pdf(0) / 2)
    assert A.sep_pa_to_offsets(5.0, 90.0) == pytest.approx((5.0, 0.0))  # PA 90 = East
    assert A.sep_pa_to_offsets(5.0, 0.0) == pytest.approx((0.0, 5.0))   # PA 0 = North
    assert A.offsets_to_sep_pa(-1.0, 0.0) == pytest.approx((1.0, 270.0))
    assert A.rowan_sigmas("sigma_dRA=0.002 mas, sigma_dDec=0.003 mas, correlation rho(dRA,dDec)=-0.33; A0") == (0.002, 0.003, -0.33)


@pytest.mark.validates("crosscheck.astrometry", roots=["mathematics"], kind="reference")
def test_published_conventions():
    fwhm = dict(sep_mas=10.0, pa_deg=90.0, sigma_major_mas=2.3548200450309493, sigma_minor_mas=1.17741002251547,
                sigma_pa_deg=0.0, note="")
    p = A.published_position(fwhm, "2017A&A...601A..34L")
    assert p["pos"] == pytest.approx([10.0, 0.0])
    assert np.linalg.eigvalsh(p["cov"]) == pytest.approx([0.25, 1.0])
    rowan = dict(dra_mas=1.0, ddec_mas=2.0, note="sigma_dRA=0.002 mas, sigma_dDec=0.003 mas, correlation rho(dRA,dDec)=0.5;")
    p = A.published_position(rowan, "2026PASP..138b4203R")
    s = math.exp(0.8)
    assert p["cov"][0, 0] == pytest.approx((s * 0.002) ** 2) and p["cov"][0, 1] == pytest.approx(0.5 * s**2 * 6e-6)
    assert p["cov_alt"][1, 1] == pytest.approx(0.003**2)
    halb = dict(sep_mas=11.0, pa_deg=-50.0, sigma_major_mas=0.1, sigma_minor_mas=0.05, sigma_pa_deg=10.0)
    p = A.published_position(halb, "2020MNRAS.496.1355H")
    assert np.linalg.eigvalsh(p["cov_alt"]) == pytest.approx(np.linalg.eigvalsh(p["cov"]) / (0.1626 * 1.0856) ** 2)
    w = A.published_for_system("zet_boo", dict(dra_mas=-67.4, ddec_mas=-31.1), "2025OJAp....8E..63W")
    assert w["cov"] == pytest.approx(np.eye(2) * 0.04**2)


@pytest.mark.validates("crosscheck.astrometry", roots=["statistics"], kind="reference")
def test_system_and_overall_statistics():
    rng = np.random.default_rng(3)
    null = stats.chi2(2).rvs(200, random_state=rng)
    s = A.system_statistic(null[:10])
    assert s["dof"] == 20 and s["p_upper"] == pytest.approx(stats.chi2.sf(null[:10].sum(), 20))
    ok = A.overall_ks(null)
    assert not ok["fails"]
    big = A.overall_ks(3 * null)      # d^2 too large: fails the one-sided test
    assert big["fails"] and big["p_larger"] < 1e-6
    small = A.overall_ks(0.3 * null)  # too small (shared data, conservative errors): not a failure
    assert not small["fails"] and small["p_two_sided"] < 1e-6


@pytest.mark.validates("crosscheck.astrometry", roots=["mathematics"], kind="reference")
@pytest.mark.skipif(not TRACKB.is_dir(), reason="Track B data are local (~/data/eso_binaries/trackb)")
def test_every_comparable_epoch_has_a_valid_covariance():
    n = 0
    for f in sorted(TRACKB.glob("*/system.json")):
        s = json.loads(f.read_text())
        for rec in s["reference_positions"]:
            if not rec.get("phase3_dp_ids"):
                continue
            p = A.published_for_system(f.parent.name, rec, s["reference"]["bibcode"])
            assert np.all(np.isfinite(p["pos"])), (f.parent.name, rec["date"])
            assert np.all(np.linalg.eigvalsh(p["cov"]) > 0), (f.parent.name, rec["date"], p["convention"])
            n += 1
    assert n == 130  # 121 counted + 9 reported separately (design/trackb_criteria.md)


@pytest.mark.validates("crosscheck.astrometry", roots=["standards"], kind="guard")
def test_fit_script_names_the_registered_criteria():
    """scripts/trackb_fit.py carries the commit and SHA-256 of design/trackb_criteria.md."""
    import hashlib
    import subprocess
    root = pathlib.Path(__file__).resolve().parents[1]
    src = (root / "scripts" / "trackb_fit.py").read_text()
    sha = hashlib.sha256((root / "design" / "trackb_criteria.md").read_bytes()).hexdigest()
    assert f'CRITERIA_SHA256 = "{sha}"' in src
    commit = src.split('CRITERIA_COMMIT = "')[1][:40]
    try:
        blob = subprocess.run(["git", "-C", str(root), "show", f"{commit}:design/trackb_criteria.md"],
                              capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("no git history (e.g. a source snapshot)")
    assert hashlib.sha256(blob).hexdigest() == sha
