"""Interferometer geometry and observables, from first principles.

uv coordinates follow Thompson, Moran & Swenson (2017), "Interferometry and
Synthesis in Radio Astronomy", eqs. 4.1-4.4: a baseline given in local
(East, North, Up) metres is turned into equatorial (X, Y, Z) at the site
latitude, then projected onto the sky at hour angle H and declination dec.
u points East and v North on the sky.

For a baseline from telescope 1 to telescope 2 we take B = x_2 - x_1. Its
visibility is V(u, v) of that B; the reversed baseline sees the complex
conjugate.
"""

from itertools import combinations

import numpy as np


def enu_to_xyz(enu, latitude_deg):
    east, north, up = np.moveaxis(np.asarray(enu, float), -1, 0)
    lat = np.deg2rad(latitude_deg)
    x = -np.sin(lat) * north + np.cos(lat) * up
    y = east
    z = np.cos(lat) * north + np.sin(lat) * up
    return np.stack([x, y, z], -1)


def uv_from_xyz(xyz, hour_angle_h, dec_deg):
    x, y, z = np.moveaxis(np.asarray(xyz, float), -1, 0)
    h = np.deg2rad(15.0 * np.asarray(hour_angle_h, float))
    d = np.deg2rad(dec_deg)
    u = np.sin(h) * x + np.cos(h) * y
    v = -np.sin(d) * np.cos(h) * x + np.sin(d) * np.sin(h) * y + np.cos(d) * z
    return u, v


def baselines(n_tel):
    return list(combinations(range(n_tel), 2))


def triangles(n_tel):
    return list(combinations(range(n_tel), 3))


def snapshot_uv(stations_enu, hour_angle_h, dec_deg, latitude_deg):
    """u, v (metres) for every pair (i < j) with B = x_j - x_i."""
    stations_enu = np.asarray(stations_enu, float)
    if stations_enu.shape[1] == 2:
        stations_enu = np.column_stack(
            [stations_enu, np.zeros(len(stations_enu))]
        )
    pairs = baselines(len(stations_enu))
    b_enu = np.array([stations_enu[j] - stations_enu[i] for i, j in pairs])
    return uv_from_xyz(
        enu_to_xyz(b_enu, latitude_deg), hour_angle_h, dec_deg
    )


def pupil_uv(holes_xy):
    """For a mask on the sky-oriented pupil: B = x_j - x_i directly."""
    holes_xy = np.asarray(holes_xy, float)
    pairs = baselines(len(holes_xy))
    b = np.array([holes_xy[j] - holes_xy[i] for i, j in pairs])
    return b[:, 0], b[:, 1]


def closure_phase(vis_fn, u, v, n_tel):
    """Closure phases for every triangle (a < b < c) of one snapshot.

    T3 = V(ab) V(bc) conj(V(ac)), with (u1, v1) = ab and (u2, v2) = bc, as
    in the OIFITS definition V(u1, v1) V(u2, v2) V(-u1-u2, -v1-v2).
    Returns (cp radians, u1, v1, u2, v2).
    """
    index = {pair: k for k, pair in enumerate(baselines(n_tel))}
    tri = triangles(n_tel)
    ab = np.array([index[(a, b)] for a, b, c in tri])
    bc = np.array([index[(b, c)] for a, b, c in tri])
    u1, v1, u2, v2 = u[ab], v[ab], u[bc], v[bc]
    t3 = vis_fn(u1, v1) * vis_fn(u2, v2) * np.conj(vis_fn(u1 + u2, v1 + v2))
    return np.angle(t3), u1, v1, u2, v2
