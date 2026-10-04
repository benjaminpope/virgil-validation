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

        if rng is not None and phase_noise == "baseline":
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
    t3 = {k: np.concatenate(x) for k, x in t3_rec.items()}
    clean = (v2["vis2"].copy(), t3.pop("clean"))
    if rng is not None:
        v2["vis2"] = v2["vis2"] + sigma_v2 * rng.standard_normal(v2["vis2"].shape)
        if phase_noise == "triangle":
            t3["phi"] = clean[1] + sigma_cp_deg * rng.standard_normal(t3["phi"].shape)
    v2["err"] = np.full(v2["vis2"].shape, max(sigma_v2, 1e-6))
    t3["err"] = np.full(t3["phi"].shape, max(sigma_cp_deg, 1e-6))
    oifits_writer.write(
        path,
        wavelengths=wl,
        stations_xyz=stations_enu,
        vis2=v2,
        t3=t3 if closure_phases else None,
    )
    return clean
