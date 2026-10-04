"""The imaging contests' test binaries (contests/manifest.yml).

Before each contest the organisers handed out a binary with published
parameters, so that contestants could check they read the data the right
way round. The files were simulated by other people's codes (OYSTER for
2004, Cotton's CHARA simulator for 2008, ASPRO for 2010), and the parameters
come from the contest pages, so these test virgil's reading of real-world
OIFITS conventions (East, PA, which star is which, closure-phase sign)
against the literature.

Each fit starts at the published position and at its mirror image (180°
away, the position a closure-phase sign error would prefer); the published
side must fit the data far better.

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


def _scene(sep, pa, flux, diams):
    dra, ddec = sep * np.sin(np.deg2rad(pa)), sep * np.cos(np.deg2rad(pa))
    span = dist.Uniform(-5 * sep, 5 * sep)
    if diams is None:
        model = vm.BinaryModelCartesian(dra, ddec, flux)
        priors = {"dra": span, "ddec": span, "flux": dist.Uniform(0, 1)}
        return model, priors, ("dra", "ddec", "flux")
    model = vm.System(
        a=vm.UniformDisk(diams[0]),
        b=vm.UniformDisk(diams[1], flux=flux, dra=dra, ddec=ddec),
    )
    priors = {
        "a.diam": dist.Uniform(0, 5 * max(diams)),
        "b.diam": dist.Uniform(0, 5 * max(diams)),
        "b.dra": span,
        "b.ddec": span,
        "b.flux": dist.Uniform(0, 1),
    }
    return model, priors, ("b.dra", "b.ddec", "b.flux")


def _fit(data, sep, pa, flux, diams):
    model, priors, keys = _scene(sep, pa, flux, diams)
    result = fit(model, priors, data)
    chi2 = float(np.sum(np.asarray(whitened_residuals(result.model, data)) ** 2))
    dra, ddec, f = (float(result.values[k]) for k in keys)
    return np.hypot(dra, ddec), np.rad2deg(np.arctan2(dra, ddec)) % 360, f, chi2


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

    s, p, f, chi2 = _fit(data, sep, pa, flux, diams)
    _, _, _, chi2_mirror = _fit(data, sep, (pa + 180) % 360, flux, diams)
    record(f"{year}_sep_mas", s)
    record(f"{year}_pa_deg", p)
    record(f"{year}_flux_ratio", f)
    record(f"{year}_chi2_reduced", chi2 / n)
    record(f"{year}_chi2_mirror_minus_published", chi2_mirror - chi2)

    assert s == pytest.approx(sep, rel=0.02)
    assert (p - pa + 180) % 360 - 180 == pytest.approx(0, abs=2.0)
    assert f == pytest.approx(flux, rel=0.2)
    assert chi2_mirror - chi2 > 25  # the published side is preferred by > 5 sigma
