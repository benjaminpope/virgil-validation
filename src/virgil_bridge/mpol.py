"""virgil.radio in MPoL's parametrisation (the IM Lup multi-EB example).

The map between MPoL's ``Net`` parameters and virgil's scene and nuisance
sites is written down in ``docs/alma_conventions.md`` ("MPoL and virgil.radio").
In short, for pixel size ``dx`` (mas) and ``N`` pixels:

* MPoL's image is in Jy/arcsec^2 in the *packed* layout, ``softplus`` of the
  base cube; unpacked (``fftshift``) the column index ``c`` is ``l = (c - N/2) dx``
  towards East and the row index ``r`` is ``m = (r - N/2) dx`` towards North.
* virgil's ``Image`` has row 0 North and column 0 East, the centre at
  ``((N-1)/2, (N-1)/2)``, brightness ``softmax(log_brightness)`` times flux.
  So ``virgil = unpacked[::-1, ::-1]`` with ``dra = ddec = -dx/2`` and
  ``flux = sum(I) dx^2``.
* ``cal.gain = g``, ``cal.sigma = g * s_M`` per ddid, ``cal.offset = 1000 *
  (delta_ra, delta_dec)`` mas, ``blur_fwhm_mas = 1000 * FWHM``, ``alpha`` off.

Only this module imports virgil (the independence rule); everything MPoL
runs in ``external_bridge.mpol_worker``.
"""

import numpy as np

ARCSEC = np.pi / 180.0 / 3600.0


def unpack(packed):
    """MPoL packed layout -> virgil layout (row 0 North, column 0 East)."""
    return np.fft.fftshift(np.asarray(packed))[::-1, ::-1]


def pack(sky):
    """virgil layout -> MPoL packed layout (inverse of ``unpack``)."""
    return np.fft.ifftshift(np.asarray(sky)[::-1, ::-1])


def softplus_inverse(image):
    """The base cube whose ``torch.nn.Softplus`` is ``image`` (positive)."""
    image = np.asarray(image, float)
    return image + np.log(-np.expm1(-image))


def imlup_scene(npix, pixel_mas, *, half_pixel=True, flip=True):
    """``scene(values)`` for ``values['log_brightness']``, laid out as
    ``unpack`` returns it. The keywords switch the map's two parts off, for
    the controls: ``half_pixel`` is the ``-dx/2`` offset, ``flip`` is the
    column and row reversal (the latter is applied to the image by the
    caller, so only the offset is a scene property)."""
    from virgil import Image

    shift = -pixel_mas / 2.0 if half_pixel else 0.0
    return lambda values: Image(values["log_brightness"], pixel_scale_mas=pixel_mas, dra=shift, ddec=shift, ft="nufft")


def obs_of_ddid(obsid_dict, n_ddid):
    out = np.zeros(n_ddid, int)
    for obs, ddids in obsid_dict.items():
        out[list(ddids)] = int(obs)
    return out


def virgil_values(theta, obs_map, pixel_mas, *, sigma_map="g_times_s"):
    """virgil's values (a dict of ``cal.*`` sites and ``log_brightness``) from
    MPoL's parameters ``theta``: ``base_cube`` (packed, softplus domain),
    ``log10_sigma`` (per ddid), ``log10_amp`` (n_obs - 1, observation 0 fixed),
    ``delta_ra`` and ``delta_dec`` (arcsec, n_obs - 1). Written in jax, so a
    gradient with respect to ``theta`` is the chain rule through the map.
    ``sigma_map="s_only"`` is the control, ``s = s_M``."""
    import jax
    import jax.numpy as jnp

    cell = pixel_mas / 1000.0  # arcsec
    image = jax.nn.softplus(theta["base_cube"])  # Jy / arcsec^2
    sky = jnp.fft.fftshift(image)[::-1, ::-1]
    gain_obs = 10.0 ** jnp.concatenate([jnp.zeros(1), theta["log10_amp"]])
    s_m = 10.0 ** theta["log10_sigma"]
    sigma = s_m * gain_obs[obs_map] if sigma_map == "g_times_s" else s_m
    return {
        "cal.flux": jnp.sum(image) * cell**2,
        "cal.gain": gain_obs[1:],
        "cal.offset": 1000.0 * jnp.stack([theta["delta_ra"], theta["delta_dec"]], axis=-1),
        "cal.sigma": sigma,
        "log_brightness": jnp.log(sky),
    }


def neg_log_likelihood(term, values, data):
    """-ln L of virgil's term, normalising constant included, written out:
    ``1/2 sum r^2 + log_norm + sum_k (ln 2 pi - ln w_k)`` (jax)."""
    import jax.numpy as jnp

    r = term(values)
    return 0.5 * jnp.sum(r**2) + term.log_norm(values) + jnp.sum(jnp.log(2 * jnp.pi) - jnp.log(jnp.asarray(data.weight)))
