"""Keplerian relative orbits from the textbook formulae, with SciPy.

The visual-binary convention: the node Omega and position angles are
measured North through East, omega from the ascending node in the direction
of motion, u = omega + f the argument of latitude, and

    dDec = r (cos Omega cos u - sin Omega sin u cos i)
    dRA  = r (sin Omega cos u + cos Omega sin u cos i)

so for i < 90 deg the position angle increases with time. Along the line
of sight, z = r sin u sin i is positive away from the observer: the
ascending node is the one where the companion recedes. Its radial velocity
relative to the primary is then the textbook

    v_z = 2 pi a sin i / (P sqrt(1 - e^2)) (cos u + e cos omega),

positive receding. Kepler's equation M = E - e sin E is solved by Newton's
method in SciPy.
"""

import numpy as np
from scipy import optimize


def eccentric_anomaly(mean, ecc):
    mean = np.atleast_1d(np.asarray(mean, float))
    return np.array([optimize.newton(lambda E, m=m: E - ecc * np.sin(E) - m, m + ecc * np.sin(m),
                                     fprime=lambda E: 1 - ecc * np.cos(E), tol=1e-15, maxiter=100)
                     for m in mean])


def true_anomaly(mean, ecc):
    E = eccentric_anomaly(mean, ecc)
    return 2 * np.arctan2(np.sqrt(1 + ecc) * np.sin(E / 2), np.sqrt(1 - ecc) * np.cos(E / 2))


def sky_position(f, ecc, inc, omega, Omega, a=1.0):
    """(dRA, dDec) at true anomaly f (radians); angles in degrees."""
    i, w, W = np.deg2rad([inc, omega, Omega])
    r = a * (1 - ecc**2) / (1 + ecc * np.cos(f))
    u = w + f
    ddec = r * (np.cos(W) * np.cos(u) - np.sin(W) * np.sin(u) * np.cos(i))
    dra = r * (np.sin(W) * np.cos(u) + np.cos(W) * np.sin(u) * np.cos(i))
    return dra, ddec


def position(t, period, t_peri, ecc, inc, omega, Omega, a):
    """(dRA, dDec) in a's units at times t (days)."""
    mean = 2 * np.pi * (np.asarray(t, float) - t_peri) / period
    return sky_position(true_anomaly(np.mod(mean + np.pi, 2 * np.pi) - np.pi, ecc), ecc, inc, omega, Omega, a)


def position_angle(f, ecc, inc, omega, Omega):
    """Position angle (radians, North through East) at true anomaly f."""
    dra, ddec = sky_position(f, ecc, inc, omega, Omega)
    return np.arctan2(dra, ddec)


def depth(f, ecc, inc, omega, a=1.0):
    """z, positive away from the observer, at true anomaly f (radians)."""
    r = a * (1 - ecc**2) / (1 + ecc * np.cos(f))
    return r * np.sin(np.deg2rad(omega) + f) * np.sin(np.deg2rad(inc))


def radial_velocity(t, period, t_peri, ecc, inc, omega, a):
    """dz/dt of the companion relative to the primary (a's units per day,
    positive receding) at times t (days)."""
    mean = 2 * np.pi * (np.asarray(t, float) - t_peri) / period
    f = true_anomaly(np.mod(mean + np.pi, 2 * np.pi) - np.pi, ecc)
    k = 2 * np.pi * a * np.sin(np.deg2rad(inc)) / (period * np.sqrt(1 - ecc**2))
    return k * (np.cos(np.deg2rad(omega) + f) + ecc * np.cos(np.deg2rad(omega)))
