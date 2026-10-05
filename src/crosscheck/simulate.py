"""Turn a visibility function into an OIFITS file of V^2 and closure phases."""

import numpy as np

from . import array, oifits_writer


def observe(
    path,
    vis_fn,
    stations_enu,
    *,
    hour_angles_h,
    wavelengths,
    dec_deg=-30.0,
    latitude_deg=-24.6276,
    sigma_v2=0.0,
    sigma_cp_deg=0.0,
    rng=None,
    fixed_uv=None,
    phase_noise="baseline",
    closure_phases=True,
):
    """Simulate and write one OIFITS file.

    ``vis_fn(u, v, wavel)`` returns complex visibilities (any shape).
    Telescopes move with Earth rotation unless ``fixed_uv`` gives the
    baselines directly (a pupil mask, (u, v) for each pair i < j), in which
    case ``hour_angles_h`` only sets the number of snapshots.

    With ``closure_phases=False`` the file holds V² alone (no OI_T3).

    Errors written to the file are ``sigma_v2`` and ``sigma_cp_deg`` (made
    at least 1e-6 so the file stays valid); noise is added only if ``rng``.
    With ``phase_noise="baseline"`` (the default) the phase noise is drawn
    per baseline, sigma_cp / sqrt(3) each, and closure phases are formed
    from the noisy phases, so that triangles sharing a baseline are
    correlated as in real data. ``"triangle"`` draws independent noise per
    closure phase instead.
    Returns the noise-free (v2, cp_deg) arrays for reference.
    """
    stations_enu = np.asarray(stations_enu, float)
    if stations_enu.shape[1] == 2:
        stations_enu = np.column_stack(
            [stations_enu, np.zeros(len(stations_enu))]
        )
    n_tel = len(stations_enu)
    # OIFITS stores EFF_WAVE as float32: simulate at the stored values so
    # that the file describes exactly what was simulated.
    wl = np.atleast_1d(np.asarray(wavelengths, np.float32)).astype(float)
    pairs = np.array(array.baselines(n_tel))
    tris = np.array(array.triangles(n_tel))

    v2_rec = {k: [] for k in ("vis2", "u", "v", "sta", "mjd")}
    t3_rec = {k: [] for k in ("phi", "u1", "v1", "u2", "v2", "sta", "mjd")}
    for k, ha in enumerate(np.atleast_1d(hour_angles_h)):
        if fixed_uv is None:
            u, v = array.snapshot_uv(stations_enu, ha, dec_deg, latitude_deg)
        else:
            u, v = map(np.asarray, fixed_uv)
        mjd = 60000.0 + k / 24.0

        if rng is not None and phase_noise == "baseline" and closure_phases:
            noise = np.exp(
                1j * np.deg2rad(sigma_cp_deg / np.sqrt(3.0))
                * rng.standard_normal((len(u), len(wl)))
            )
        else:
            noise = np.ones((len(u), len(wl)))

        def at(uu, vv):
            return vis_fn(uu[:, None], vv[:, None], wl[None, :])

        def noisy(uu, vv):
            """Visibility with the baseline phase noise; the closing
            baseline u1 + u2 = ac is looked up the same way."""
            dist = np.hypot(uu[:, None] - u[None, :], vv[:, None] - v[None, :])
            k = np.argmin(dist, axis=1)
            assert np.all(dist[np.arange(len(k)), k] < 1e-6)
            return at(uu, vv) * noise[k]

        v2_rec["vis2"].append(np.abs(at(u, v)) ** 2)
        v2_rec["u"].append(u)
        v2_rec["v"].append(v)
        v2_rec["sta"].append(pairs)
        v2_rec["mjd"].append(np.full(len(u), mjd))
        if not closure_phases:
            # V² alone: no triangles are formed (a two-telescope array has
            # none) and no phase noise is drawn.
            continue
        cp, u1, v1, u2, v2 = array.closure_phase(noisy, u, v, n_tel)
        clean_cp, *_ = array.closure_phase(at, u, v, n_tel)
        t3_rec.setdefault("clean", []).append(np.rad2deg(clean_cp))
        t3_rec["phi"].append(np.rad2deg(cp))
        t3_rec["u1"].append(u1)
        t3_rec["v1"].append(v1)
        t3_rec["u2"].append(u2)
        t3_rec["v2"].append(v2)
        t3_rec["sta"].append(tris)
        t3_rec["mjd"].append(np.full(len(u1), mjd))

    v2 = {k: np.concatenate(x) for k, x in v2_rec.items()}
    if closure_phases:
        t3 = {k: np.concatenate(x) for k, x in t3_rec.items()}
        clean_cp = t3.pop("clean")
    else:
        t3, clean_cp = None, np.zeros((0, len(wl)))
    clean = (v2["vis2"].copy(), clean_cp)
    if rng is not None:
        v2["vis2"] = v2["vis2"] + sigma_v2 * rng.standard_normal(v2["vis2"].shape)
        if t3 is not None and phase_noise == "triangle":
            t3["phi"] = clean[1] + sigma_cp_deg * rng.standard_normal(t3["phi"].shape)
    v2["err"] = np.full(v2["vis2"].shape, max(sigma_v2, 1e-6))
    if t3 is not None:
        t3["err"] = np.full(t3["phi"].shape, max(sigma_cp_deg, 1e-6))
    oifits_writer.write(
        path,
        wavelengths=wl,
        stations_xyz=stations_enu,
        vis2=v2,
        t3=t3,
    )
    return clean


def observe_visibilities(path, vis_fn, stations_enu, *, hour_angles_h, wavelengths, dec_deg=-30.0,
                         latitude_deg=-24.6276, sigma=0.0, rng=None):
    """Simulate complex visibilities with complex Gaussian noise (``sigma``
    on each of the real and imaginary parts) and write them as visibility
    amplitudes (``OI_VIS`` ``VISAMP``, error ``sigma``, ``AMPTYP``
    absolute) plus closure phases (``OI_T3``) formed from the same noisy
    visibilities, with errors propagated to first order,
    sqrt(sum (sigma / |V|)^2) over the triangle's baselines. No ``OI_VIS2``.

    For codes that fit complex visibilities rather than files (e.g.
    eht-imaging): returns, per baseline row and channel, the MJD, station
    pair (1-based), u, v (m), wavelength (m) and the noisy visibility, as a
    dict of arrays of shape (n_row, n_channel)."""
    from astropy.io import fits

    observe(path, vis_fn, stations_enu, hour_angles_h=hour_angles_h, wavelengths=wavelengths,
            dec_deg=dec_deg, latitude_deg=latitude_deg)
    with fits.open(path) as h:
        v2h, t3d = h["OI_VIS2"], h["OI_T3"].data.copy()
        d = v2h.data
        wl = np.asarray(h["OI_WAVELENGTH"].data["EFF_WAVE"], float)
        u, v, sta, mjd = d["UCOORD"], d["VCOORD"], d["STA_INDEX"], d["MJD"]
        clean = vis_fn(u[:, None], v[:, None], wl[None, :])
        noise = np.zeros(clean.shape, complex)
        if rng is not None:
            noise = sigma * (rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape))
        noisy = clean + noise
        amp = np.abs(noisy)

        def baseline(m, i, j):
            k = np.flatnonzero((mjd == m) & (((sta[:, 0] == i) & (sta[:, 1] == j)) | ((sta[:, 0] == j) & (sta[:, 1] == i))))[0]
            return (noisy[k] if sta[k, 0] == i else np.conj(noisy[k])), amp[k]

        phi, err = [], []
        for row in range(len(t3d)):
            a, b, c = t3d["STA_INDEX"][row]
            m = t3d["MJD"][row]
            (vab, aab), (vbc, abc), (vac, aac) = baseline(m, a, b), baseline(m, b, c), baseline(m, a, c)
            phi.append(np.rad2deg(np.angle(vab * vbc * np.conj(vac))))
            err.append(np.rad2deg(np.sqrt((sigma / aab) ** 2 + (sigma / abc) ** 2 + (sigma / aac) ** 2)))
        t3d["T3PHI"] = np.array(phi)
        t3d["T3PHIERR"] = np.maximum(np.array(err), 1e-6)
        n, nw = amp.shape
        keep = ("TARGET_ID", "TIME", "MJD", "INT_TIME", "UCOORD", "VCOORD", "STA_INDEX", "FLAG")
        cols = [fits.Column(name=c.name, format=c.format, unit=c.unit, array=d[c.name]) for c in v2h.columns if c.name in keep]
        cols += [
            fits.Column(name="VISAMP", format=f"{nw}D", array=amp),
            fits.Column(name="VISAMPERR", format=f"{nw}D", array=np.full(amp.shape, max(sigma, 1e-6))),
            fits.Column(name="VISPHI", format=f"{nw}D", unit="deg", array=np.zeros(amp.shape)),
            fits.Column(name="VISPHIERR", format=f"{nw}D", unit="deg", array=np.ones(amp.shape)),
        ]
        vis_hdu = fits.BinTableHDU.from_columns(cols)
        for key in ("OI_REVN", "DATE-OBS", "ARRNAME", "INSNAME"):
            if key in v2h.header:
                vis_hdu.header[key] = v2h.header[key]
        vis_hdu.header["EXTNAME"] = "OI_VIS"
        vis_hdu.header["AMPTYP"] = "absolute"
        vis_hdu.header["PHITYP"] = "absolute"
        t3_hdu = fits.BinTableHDU(data=t3d, header=h["OI_T3"].header)
        hdus = [x.copy() for x in h if x.name not in ("OI_VIS2", "OI_T3")] + [vis_hdu, t3_hdu]
    fits.HDUList(hdus).writeto(path, overwrite=True)
    shape = noisy.shape
    return {
        "mjd": np.broadcast_to(mjd[:, None], shape), "sta": sta,
        "u": np.broadcast_to(u[:, None], shape), "v": np.broadcast_to(v[:, None], shape),
        "wl": np.broadcast_to(wl[None, :], shape), "vis": noisy,
    }
