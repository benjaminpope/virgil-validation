"""NumPy (and astropy) side of the IM Lup MaxEnt analysis. Nothing here imports virgil.

Images follow ``crosscheck.alma``: ``image[row, col]`` with row 0 North and
column 0 East. The Fourier transform is

    V(u, v) = sum I(x, y) exp(-2 pi i (u x + v y)),   x East, y North,

so a Gaussian of FWHM theta (radians) multiplies the visibilities by
``exp(-(pi theta q)^2 / (4 ln 2))`` (Bracewell, *The Fourier Transform and
Its Applications*, the Gaussian pair), which is the Gaussian convolution of
the image; the elliptical beam of a CLEAN image is the same with q resolved
along and across its position angle (East of North). The entropy is the
formula of the EHT-IV paper (Event Horizon Telescope Collaboration 2019,
ApJ 875, L4) that MPoL's ``entropy`` implements, ``(1/F) sum I ln(I / p)``.

The grid of the MPoL example ("MPoL layout" below) has its phase centre in
the unpacked cell ``(N/2, N/2)``, i.e. ``l = (c - N/2) dx`` towards East and
``m = (r - N/2) dx`` towards North; in the layout of this repository (row 0
North, column 0 East) that cell is ``(N/2 - 1, N/2 - 1)`` (``docs/alma_conventions.md``,
the half-pixel map). ``phase_centre_cell`` returns it.
"""

import numpy as np

ARCSEC = np.pi / 180.0 / 3600.0
MAS = ARCSEC / 1000.0


def phase_centre_cell(npix):
    """(row, column) of the cell holding the phase centre, in the repository's layout."""
    return npix // 2 - 1, npix // 2 - 1


def _freqs(npix, pixel_mas):
    """(f_east, f_north) in cycles per radian on the FFT grid of an image with row 0 North and
    column 0 East: the column index runs towards West and the row index towards South."""
    f = np.fft.fftfreq(npix, d=pixel_mas * MAS)
    return -f[None, :] * np.ones((npix, 1)), -f[:, None] * np.ones((1, npix))


def beam_transfer(npix, pixel_mas, bmaj_mas, bmin_mas, bpa_deg=0.0):
    """Transfer function of an elliptical Gaussian of FWHM ``bmaj`` and ``bmin``, major axis at
    position angle ``bpa`` (degrees East of North), on the FFT grid."""
    fe, fn = _freqs(npix, pixel_mas)
    pa = np.deg2rad(bpa_deg)
    along = fe * np.sin(pa) + fn * np.cos(pa)
    across = fe * np.cos(pa) - fn * np.sin(pa)
    return np.exp(-(np.pi**2) * ((bmaj_mas * MAS * along) ** 2 + (bmin_mas * MAS * across) ** 2) / (4 * np.log(2)))


def convolve(image, pixel_mas, bmaj_mas, bmin_mas=None, bpa_deg=0.0):
    """Circular convolution of ``image`` with a Gaussian beam (FFT), as MPoL's ``GaussConvFourier`` does."""
    image = np.asarray(image, float)
    t = beam_transfer(image.shape[0], pixel_mas, bmaj_mas, bmaj_mas if bmin_mas is None else bmin_mas, bpa_deg)
    return np.fft.ifft2(np.fft.fft2(image) * t).real


def mpol_entropy(image, prior=1e-4, tot_flux=0.253):
    """``(1/F) sum I ln(I / p)`` for an image in Jy/arcsec^2 (MPoL's ``losses.entropy``)."""
    image = np.asarray(image, float)
    return float(np.sum(image * np.log(image / prior)) / tot_flux)


def beam_area_arcsec2(bmaj_mas, bmin_mas):
    """Solid angle of a Gaussian beam, pi bmaj bmin / (4 ln 2), in arcsec^2."""
    return np.pi * (bmaj_mas / 1000.0) * (bmin_mas / 1000.0) / (4 * np.log(2))


def gaussian_log_image(npix, pixel_mas, fwhm_mas, centre_cell=None):
    """ln of a circular Gaussian (peak 1) centred on a cell, in the repository's layout."""
    r0, c0 = phase_centre_cell(npix) if centre_cell is None else centre_cell
    rr, cc = np.meshgrid(np.arange(npix) - r0, np.arange(npix) - c0, indexing="ij")
    return -4 * np.log(2) * (rr**2 + cc**2) * pixel_mas**2 / fwhm_mas**2


# ------------------------------------------------------------------ FITS


def sky_wcs(npix, pixel_mas, ra_deg, dec_deg):
    """A SIN WCS (``astropy.wcs.WCS``) for an array in FITS orientation (row 0 South, so the
    repository's layout flipped upside down), the phase centre in the cell of ``phase_centre_cell``."""
    from astropy.wcs import WCS

    r0, c0 = phase_centre_cell(npix)
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.crval = [ra_deg, dec_deg]
    w.wcs.crpix = [c0 + 1, (npix - 1 - r0) + 1]  # FITS counts from 1, rows from the South
    w.wcs.cdelt = [-pixel_mas / 3.6e6, pixel_mas / 3.6e6]
    w.wcs.cunit = ["deg", "deg"]
    return w


def write_fits(path, image, pixel_mas, ra_deg, dec_deg, bunit="Jy/arcsec2", **header):
    """Write an image (the repository's layout) as a FITS file with a SIN WCS on the phase centre."""
    from astropy.io import fits

    image = np.asarray(image, float)
    hdr = sky_wcs(image.shape[0], pixel_mas, ra_deg, dec_deg).to_header()
    hdr["BUNIT"] = bunit
    for k, v in header.items():
        hdr[k] = v
    fits.PrimaryHDU(image[::-1].astype(np.float32), hdr).writeto(path, overwrite=True)


def read_fits(path):
    """(image in FITS orientation squeezed to 2-d, celestial WCS, header) of a radio FITS image
    (Stokes and frequency axes dropped)."""
    from astropy.io import fits
    from astropy.wcs import WCS

    with fits.open(path) as hdul:
        hdr = hdul[0].header
        data = np.asarray(hdul[0].data, float)
    data = data.reshape(data.shape[-2:])
    return data, WCS(hdr).celestial, hdr


def beam_of(hdr):
    """(BMAJ, BMIN) in mas and BPA in degrees from a CLEAN image's header."""
    return float(hdr["BMAJ"]) * 3.6e6, float(hdr["BMIN"]) * 3.6e6, float(hdr["BPA"])


def resample(image, wcs_src, wcs_dst, shape_dst):
    """Bilinear resampling of an image (FITS orientation, ``wcs_src``) onto the pixels of ``wcs_dst``;
    pixels outside the source are zero."""
    from scipy.ndimage import map_coordinates

    yy, xx = np.meshgrid(np.arange(shape_dst[0]), np.arange(shape_dst[1]), indexing="ij")
    ra, dec = wcs_dst.all_pix2world(xx.ravel(), yy.ravel(), 0)
    sx, sy = wcs_src.all_world2pix(ra, dec, 0)
    out = map_coordinates(np.nan_to_num(image), [sy, sx], order=1, mode="constant", cval=0.0)
    return out.reshape(shape_dst)


def difference_stats(a, b, region=None):
    """Differences of two images on one grid: rms(a - b) over the peak of b, the correlation
    coefficient, the flux ratio sum(a)/sum(b) and the peak ratio; within a boolean ``region``."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if region is not None:
        a, b = a[region], b[region]
    d = a - b
    return dict(
        rms_over_peak=float(np.sqrt(np.mean(d**2)) / np.max(np.abs(b))),
        max_abs_over_peak=float(np.max(np.abs(d)) / np.max(np.abs(b))),
        correlation=float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
        flux_ratio=float(a.sum() / b.sum()),
        peak_ratio=float(a.max() / b.max()),
    )
