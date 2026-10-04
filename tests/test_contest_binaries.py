"""The imaging contests' test binaries (contests/manifest.yml).

Before each contest the organisers handed out a binary with published
parameters, so that contestants could check they read the data the right
way round. The files were simulated by other people's codes (OYSTER for
2004, Cotton's CHARA simulator for 2008, ASPRO for 2010), and the parameters
come from the contest pages, so these test virgil's reading of real-world
OIFITS conventions (East, PA, which star is which, closure-phase sign)
against the literature.

After the fit, the companion is moved to its mirror image (180° away, where
a sign error in East, PA or closure phase would put it) and held there while
the fluxes and diameters are refitted; that hypothesis must fit the data far
worse.

The data are not in the repository: run scripts/fetch_contests.py first
(or set CONTEST_DATA). The tests skip when the files are absent.
"""

import os
import pathlib

import numpy as np
import numpyro.distributions as dist
import pytest

from evidence.plugin import record

vm = pytest.importorskip("virgil.models")
from virgil.fitting import fit  # noqa: E402
from virgil.likelihood import whitened_residuals  # noqa: E402
from virgil.oidata import OIData  # noqa: E402

pytestmark = pytest.mark.x64

DATA = pathlib.Path(os.environ.get("CONTEST_DATA", "~/data/imaging_contests")).expanduser()

# Published parameters: separation (mas), PA (deg E of N, bright to faint),
# faint/bright flux ratio, uniform-disk diameters (mas) of bright and faint.
BINARIES = {
    # 2004 contest page: "rho=21.2 mas and pa=341.6 deg, flux ratio 5.75,
    # component diameters of 0.6 mas"
    "2004": ("2004/2004-BSC1948I.fits", 21.2, 341.6, 1 / 5.75, (0.6, 0.6)),
    # 2008 readme: 5.0 mas, PA 30 (bright to faint), ratio 8.9, UD 1.2 and 0.75 mas
    "2008": ("2008/2008-Contest_Binary.oifits", 5.0, 30.0, 1 / 8.9, (1.2, 0.75)),
    # 2010 Challenge.txt: flux ratio 0.1, 18 mas at PA 128 (point sources)
    "2010": ("2010/Binary-test-Med_H.oifits", 18.0, 128.0, 0.1, None),
}


def _scene(dra, ddec, flux, diams, sep, fixed_position=False):
    span = dist.Uniform(-5 * sep, 5 * sep)
    if diams is None:
        model = vm.BinaryModelCartesian(dra, ddec, flux)
        priors = {"flux": dist.Uniform(0, 1)}
        if not fixed_position:
            priors |= {"dra": span, "ddec": span}
        return model, priors, ("dra", "ddec", "flux")
    model = vm.System(
        a=vm.UniformDisk(diams[0]),
        b=vm.UniformDisk(diams[1], flux=flux, dra=dra, ddec=ddec),
    )
    priors = {
        "a.diam": dist.Uniform(0, 5 * max(diams)),
        "b.diam": dist.Uniform(0, 5 * max(diams)),
        "b.flux": dist.Uniform(0, 1),
    }
    if not fixed_position:
        priors |= {"b.dra": span, "b.ddec": span}
    return model, priors, ("b.dra", "b.ddec", "b.flux")


def _chi2(model, data):
    return float(np.sum(np.asarray(whitened_residuals(model, data)) ** 2))


def _fit(data, sep, pa, flux, diams):
    """Fit from the published parameters; then hold the companion at the
    mirror image of the fitted position and refit the rest."""
    dra, ddec = sep * np.sin(np.deg2rad(pa)), sep * np.cos(np.deg2rad(pa))
    model, priors, keys = _scene(dra, ddec, flux, diams, sep)
    result = fit(model, priors, data)
    dra, ddec, f = (float(result.values[k]) for k in keys)
    d = None if diams is None else tuple(float(result.values[k]) for k in ("a.diam", "b.diam"))
    mirror, mirror_priors, _ = _scene(-dra, -ddec, f, d, sep, fixed_position=True)
    mirrored = fit(mirror, mirror_priors, data)
    sep_fit, pa_fit = np.hypot(dra, ddec), np.rad2deg(np.arctan2(dra, ddec)) % 360
    return sep_fit, pa_fit, f, _chi2(result.model, data), _chi2(mirrored.model, data)


@pytest.mark.validates(
    "virgil.oifits.read_oifits", "virgil.oidata.OIData", "virgil.fitting.fit",
    roots=["literature"],
)
@pytest.mark.parametrize("year", sorted(BINARIES))
def test_contest_binary_matches_published_parameters(year):
    name, sep, pa, flux, diams = BINARIES[year]
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{path} absent: run scripts/fetch_contests.py")
    data = OIData(str(path))
    n = np.asarray(whitened_residuals(vm.PointSource(), data)).size

    s, p, f, chi2, chi2_mirror = _fit(data, sep, pa, flux, diams)
    record(f"{year}_sep_mas", s)
    record(f"{year}_pa_deg", p)
    record(f"{year}_flux_ratio", f)
    record(f"{year}_chi2_reduced", chi2 / n)
    record(f"{year}_chi2_mirror_minus_published", chi2_mirror - chi2)

    assert s == pytest.approx(sep, rel=0.02)
    assert (p - pa + 180) % 360 - 180 == pytest.approx(0, abs=2.0)
    assert f == pytest.approx(flux, rel=0.2)
    assert chi2_mirror - chi2 > 25  # the mirror image is rejected at > 5 sigma


@pytest.mark.xfail(strict=True, reason="F10: T3 and VIS2 under different INSNAMEs, reversed legs (virgil#167)")
@pytest.mark.validates("virgil.oifits.read_oifits", "virgil.oidata.OIData", roots=["standards"], kind="finding")
def test_2006_files_with_separate_t3_insname_are_read():
    """The 2006 contest files (simulated AMBER, OIFITS v1) keep their closure
    phases under INSNAMEs like AMBER-LR_TR01_OB01 and their V² under
    AMBER-LR_OB01, with identical OI_WAVELENGTH tables, and store baseline
    (2, 0) where the triangle's leg is (0, 2). The standard allows both;
    virgil raised ValueError until virgil#167."""
    path = DATA / "2006/2006-03-03.fits"
    if not path.exists():
        pytest.skip(f"{path} absent: run scripts/fetch_contests.py")
    data = OIData(str(path))
    v2 = np.asarray(data.model(vm.PointSource()))
    assert np.isfinite(v2).all()
    assert np.asarray(whitened_residuals(vm.PointSource(), data)).size >= 648 + 216
