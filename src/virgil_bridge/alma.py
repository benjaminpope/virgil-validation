"""virgil.radio on the ALMA files of ``external_bridge.alma_worker`` (the only
module here that imports virgil for radio data).

``load_npz`` turns W2's one-file-per-DATA_DESC_ID ``.npz`` files into a
``ComplexVisData``; the scene builders pair numpy-made truth images with the
virgil ``Image`` that fits them. Nothing here is an instrument default: the
caller passes the pixel scale, the dish and so on.
"""

import os
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
C = 299792458.0

# casatools finds its measures data through this file, written by
# scripts/setup_external.sh alma. Set it before the first casatools import
# (virgil.radio imports casatools lazily) so that a second casatools, the one
# of the virgil venv, shares .venv-alma's measures data.
_site = ROOT / ".venv-alma" / "casasiteconfig.py"
if "CASASITECONFIG" not in os.environ and _site.exists():
    os.environ["CASASITECONFIG"] = str(_site)


def load_npz(zs):
    """A ``ComplexVisData`` from W2's per-ddid dicts (or ``.npz`` paths).

    The reader of W2 has already conjugated ``DATA``; ``re``/``im`` are V.
    ``ddid`` and ``obs`` are the MS's own ids, as virgil's ``from_arrays``
    wants them.
    """
    from virgil.radio import ComplexVisData

    zs = [dict(np.load(z)) if isinstance(z, (str, pathlib.Path)) else z for z in zs]
    cat = lambda k: np.concatenate([np.asarray(z[k]) for z in zs])  # noqa: E731
    n = [len(z["u"]) for z in zs]
    return ComplexVisData.from_arrays(
        cat("u"), cat("v"), cat("re").astype(float) + 1j * cat("im").astype(float),
        cat("weight").astype(float), cat("freq"),
        ddid=np.repeat([int(z["ddid"]) for z in zs], n), obs=np.repeat([int(z["obsid"]) for z in zs], n),
        time=cat("time"), ant1=cat("ant1"), ant2=cat("ant2"), chan=cat("chan"),
    )


def image_scene(pixel_scale_mas, site="log_brightness"):
    """``scene(values)`` of a free pixel image, for ``data.term``."""
    from virgil import Image

    return lambda values: Image(values[site], pixel_scale_mas=pixel_scale_mas)


def fixed_image_scene(image, pixel_scale_mas, floor=1e-12):
    """``scene(values)`` of an image held at a numpy brightness (rows North
    first, columns East first), with no free pixels: the nuisance fits."""
    import jax.numpy as jnp

    from virgil import Image

    eta = jnp.log(np.maximum(np.asarray(image, float), floor * np.max(image)))
    return lambda values: Image(eta, pixel_scale_mas=pixel_scale_mas)


def point_scene(sources, pixel_scale_mas=None):
    """``scene(values)`` of fixed point sources ``[(flux, east, north)]``
    (flux relative; ``cal.flux`` carries the total), in mas."""
    from virgil import PointSource, System

    tot = sum(f for f, _, _ in sources)
    return lambda values: System(**{f"p{i}": PointSource(flux=f / tot, dra=e, ddec=n) for i, (f, e, n) in enumerate(sources)})


def ring_and_blob(npix=48, pixel_mas=25.0, radius_mas=300.0, width_mas=60.0, inc_deg=45.0, pa_deg=30.0,
                  blob=(0.2, 150.0, -250.0, 70.0)):
    """The IM Lup stand-in scene in numpy: an inclined Gaussian ring plus an
    asymmetric Gaussian blob (fraction of the flux, East, North, sigma in mas).
    Rows North first, columns East first, centred on ((n-1)/2, (n-1)/2), unit sum."""
    k = ((npix - 1) / 2.0 - np.arange(npix)) * pixel_mas
    east, north = np.meshgrid(k, k)
    pa, inc = np.deg2rad(pa_deg), np.deg2rad(inc_deg)
    major = east * np.sin(pa) + north * np.cos(pa)
    minor = (east * np.cos(pa) - north * np.sin(pa)) / np.cos(inc)
    ring = np.exp(-0.5 * ((np.hypot(major, minor) - radius_mas) / width_mas) ** 2)
    frac, be, bn, bs = blob
    spot = np.exp(-0.5 * ((east - be) ** 2 + (north - bn) ** 2) / bs**2)
    img = (1 - frac) * ring / ring.sum() + frac * spot / spot.sum()
    return img / img.sum()
