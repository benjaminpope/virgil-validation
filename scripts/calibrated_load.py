"""The one way the orbit scripts read ESO data: virgil-vlti's ``to_oidata`` on calibrated files (no noise code here).

- GRAVITY: ``<root>/trackb/<slug>/calibrated/<MODE>/<night>/*.fits`` written by ``virgil-vlti-calibrate``
  (design/orbit_noise_models.md in virgil, 3.1). A night with at least one V²-calibrated file is read from those files,
  with V² and closure phases, gains and closure offsets from the template modes beside them (data path
  ``calibrated``). A night with none is read from its closure-phase-only files, closure phases and offsets only (data
  path ``cp-only``). The wavelengths are the Track B ranges (``trackb_fit.GRAVITY_RANGES``), Brγ excluded.
- PIONIER: the files are already calibrated; flat gains and closure offsets from a pair test (data path
  ``pair-test gains``).
- The uncalibrated ``*_SCI_VIS`` V² of GRAVITY is refused (``refuse_uncalibrated_gravity``).

The nights of a system are labelled ``MJD <n>`` (UT-noon-to-noon, as ``trackb_fit.epochs_of``), with the mode appended
(``MJD <n> HIGH``) if the system has more than one.
"""

import datetime
import pathlib

import numpy as np

CALIBRATED = "calibrated"
CP_ONLY = "cp-only"
PAIR_GAINS = "pair-test gains"
MJD_ZERO = datetime.date(1858, 11, 17)

# Fractional rms of the PIONIER V² gains per telescope and per baseline and the closure offset in degrees, from the
# pair test on IW Car (virgil-vlti docs/guides/pionier.md, design/pionier_systematics.md): not measured on NN Del.
PIONIER_WIDTHS = dict(telescope=0.011, baseline=0.009, offset=0.21)
PIONIER_TEMPLATE = "pionier_pair_test_iwcar"


def refuse_uncalibrated_gravity(what="the uncalibrated SCI_VIS V²"):
    raise ValueError(
        f"{what} of GRAVITY is not used: its V² have no transfer-function calibration, and per-night amplitudes "
        "(vis_amp) cannot stand in for it. Read the calibrated files under <system>/calibrated/ with "
        "calibrated_load.gravity_nights (V² with gains where a calibrator exists, closure phases only elsewhere)."
    )


def night_of(mjd):
    """The UT-noon-to-noon night that an MJD belongs to."""
    return int(np.floor(mjd - 0.5))


def night_label(mjd, mode=None):
    return f"MJD {mjd}" + (f" {mode}" if mode else "")


def _folder_mjd(name):
    return (datetime.date.fromisoformat(name) - MJD_ZERO).days


def _is_cp_only(path):
    from astropy.io import fits
    from virgil_vlti.write import CP_ONLY_KEY

    return bool(fits.getheader(path, 0).get(CP_ONLY_KEY, False))


def _check_night_folder(name, mjd, files):
    """The folder is named by the UT-noon-to-noon night (virgil-vlti): a file's MJD-OBS must fall in it."""
    from astropy.io import fits

    for f in files:
        obs = fits.getheader(f, 0).get("MJD-OBS")
        if obs is not None and night_of(float(obs)) != mjd:
            raise ValueError(f"{f}: MJD-OBS {obs} is in night {night_of(float(obs))}, not the folder's {name} (MJD {mjd})")


def gravity_nights(sysdir):
    """Nights of a system's calibrated GRAVITY files: ``[dict(mjd, mode, path, files, notes)]`` by mode and night.

    ``path`` is ``calibrated`` if any file of the night has calibrated V² and then only those files are used (a
    closure-phase-only file of the same night is noted and left out); otherwise ``cp-only``.
    """
    root = pathlib.Path(sysdir) / "calibrated"
    if not root.is_dir():
        raise FileNotFoundError(f"no calibrated GRAVITY files under {root}: calibrate the system first (virgil-vlti)")
    nights = []
    for moded in sorted(p for p in root.iterdir() if p.is_dir()):
        for nd in sorted(p for p in moded.iterdir() if p.is_dir()):
            files = sorted(nd.glob("*.fits"))
            if not files:
                continue
            cp = {f: _is_cp_only(f) for f in files}
            full = [f for f in files if not cp[f]]
            notes = []
            if full:
                notes = [f"{f.name}: closure phases only, left out of a night with calibrated V²" for f in files if cp[f]]
                used, path = full, CALIBRATED
            else:
                used, path = files, CP_ONLY
            mjd = _folder_mjd(nd.name)
            _check_night_folder(nd.name, mjd, files)
            nights.append(dict(mjd=mjd, mode=moded.name, path=path, files=used, notes=notes))
    return nights


def load_gravity_night(night, ranges=None):
    """OIData of one ``gravity_nights`` entry: V² with gains and closure phases with offsets, or closure phases only.

    If the night's files fail together (``ValueError``) each is tried alone: a file virgil-vlti cannot load (e.g. a
    three-telescope closure-phase-only file, whose offsets need four telescopes) is left out of ``night["files"]`` with
    a reason in ``night["notes"]``, not worked around; a night with no loadable file raises."""
    import trackb_fit as T
    import virgil_vlti

    obs = ("vis", "phi") if night["path"] == CALIBRATED else "phi"

    def read(files):
        return virgil_vlti.to_oidata(files, "GRAVITY_SC", ranges=ranges or T.GRAVITY_RANGES, observables=obs)

    try:
        return read(night["files"])
    except ValueError as err:
        if len(night["files"]) == 1:
            raise
        first = err
    good = []
    for f in night["files"]:
        try:
            read([f])
            good.append(f)
        except ValueError as err:
            night["notes"].append(f"{f.name}: left out: {str(err)[:200]}")
    if not good:
        raise first
    night["files"] = good
    return read(good)


def pionier_template():
    from virgil_vlti.templates import gain_template

    return gain_template(PIONIER_TEMPLATE, **PIONIER_WIDTHS, provenance=dict(source="IW Car pair test, not NN Del"))


def _insname(path):
    from astropy.io import fits

    with fits.open(path) as h:
        return next(str(x.header["INSNAME"]).strip() for x in h if x.name == "OI_VIS2")


def load_pionier_night(files):
    """OIData of one night of calibrated PIONIER files with the pair-test gains and offsets."""
    import virgil_vlti

    names = {_insname(f) for f in files}
    if len(names) != 1:  # PIONIER carries its wavelength solution in INSNAME: one solution per read
        raise ValueError(f"PIONIER files of one night with several INSNAMEs: {sorted(names)}")
    return virgil_vlti.to_oidata([str(f) for f in files], names.pop(), template=pionier_template())


def load_pionier_night_robust(files):
    """``(OIData, dropped)``: ``load_pionier_night``, a file virgil cannot read left out as ``trackb_fit.load`` does."""
    try:
        return load_pionier_night(files), []
    except ValueError:
        if len(files) == 1:
            raise
    good, dropped = [], []
    for f in files:
        try:
            load_pionier_night([f])
            good.append(f)
        except ValueError as err:
            dropped.append(dict(file=f.name, error=str(err)[:300]))
    if not good:
        raise ValueError(f"no readable file: {dropped}")
    return load_pionier_night(good), dropped


def system_data_path(paths):
    """One value for a system: the nights' data path if all agree, else ``mixed``."""
    paths = set(paths)
    return paths.pop() if len(paths) == 1 else "mixed"
