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


# ---------------------------------------------------------------------------
# Projected-orbit helpers for the sky-plane comparison (crosscheck.skyplane,
# design/orbit_comparison.md). Samples are dicts of equal-length arrays with
# keys P (days), e, t_p (days), a (mas), inc, omega, Omega (degrees).
# ---------------------------------------------------------------------------

def thiele_innes(a, inc, omega, Omega):
    """Thiele-Innes constants (A, B, F, G), degrees in, a's units out.

    Expanding `sky_position` in cos f and sin f (textbook, e.g. Heintz 1978):
    dDec = A x + F y and dRA = B x + G y with x = cos E - e and
    y = sqrt(1 - e^2) sin E. So A, F are the Dec coefficients and B, G the RA
    ones. This is the only place the constants are computed (the copy in
    `crosscheck.skyplane` was deleted; it imports this one); never take them
    from a quoted table.

    Solved problem, the (Omega + 180, omega + 180) twin: every constant is a
    product of one cosine or sine of omega with one of Omega, so shifting both
    angles by 180 deg leaves A, B, F, G unchanged and the two element sets are
    one sky orbit. Shifting only one of them negates all four: the point
    reflection, i.e. the primary/secondary label swap. Compare orbits through
    these constants, never through raw (Omega, omega).

    Parameters
    ----------
    a : float or array_like
        Semi-major axis, in any unit (returned in the same unit).
    inc, omega, Omega : float or array_like
        Inclination, argument of periastron of the companion and node, degrees.

    Returns
    -------
    A, B, F, G : numpy.ndarray
        The Thiele-Innes constants, in the unit of ``a``.
    """
    i, w, W = (np.deg2rad(np.asarray(v, float)) for v in (inc, omega, Omega))
    a = np.asarray(a, float)
    A = a * (np.cos(w) * np.cos(W) - np.sin(w) * np.sin(W) * np.cos(i))
    B = a * (np.cos(w) * np.sin(W) + np.sin(w) * np.cos(W) * np.cos(i))
    F = -a * (np.sin(w) * np.cos(W) + np.cos(w) * np.sin(W) * np.cos(i))
    G = -a * (np.sin(w) * np.sin(W) - np.cos(w) * np.cos(W) * np.cos(i))
    return A, B, F, G


def _kepler_E(mean, ecc):
    """Vectorised Newton solution of M = E - e sin E (starting at pi for high e).

    Parameters
    ----------
    mean : array_like
        Mean anomaly, radians.
    ecc : float or array_like
        Eccentricity, broadcast against ``mean``.

    Returns
    -------
    numpy.ndarray
        Eccentric anomaly E, radians, in [-pi, pi).

    Raises
    ------
    RuntimeError
        If Newton's method has not converged in 100 steps.
    """
    mean = np.mod(mean + np.pi, 2 * np.pi) - np.pi
    E = mean + ecc * np.sin(mean)
    E = np.where(ecc > 0.8, np.where(mean < 0, -np.pi, np.pi), E)
    for _ in range(100):
        step = (E - ecc * np.sin(E) - mean) / (1 - ecc * np.cos(E))
        E = E - step
        if np.max(np.abs(step)) < 1e-14:
            break
    else:
        raise RuntimeError("Kepler's equation did not converge")
    return E


def predict_track(samples, t):
    """Relative positions (dRA, dDec) of every sample at times t (days).

    Built from the Thiele-Innes form, so it equals `position` to rounding.

    Parameters
    ----------
    samples : dict of array_like
        Equal-length arrays P (days), e, t_p (days), a, inc, omega, Omega (degrees).
    t : array_like
        Times, days.

    Returns
    -------
    numpy.ndarray
        Positions (dRA, dDec) with shape (n_samples, len(t), 2).
    """
    t = np.asarray(t, float)
    P, e, tp = (np.asarray(samples[k], float)[:, None] for k in ("P", "e", "t_p"))
    A, B, F, G = (v[:, None] for v in thiele_innes(*(np.asarray(samples[k], float)
                                                     for k in ("a", "inc", "omega", "Omega"))))
    E = _kepler_E(2 * np.pi * (t[None, :] - tp) / P, e)
    x, y = np.cos(E) - e, np.sqrt(1 - e**2) * np.sin(E)
    return np.stack([B * x + G * y, A * x + F * y], axis=-1)


def fold_samples(omega, Omega, Omega_ref):
    """Fold to Omega in [Omega_ref - 90, Omega_ref + 90), shifting omega with it.

    Solved problem, the (Omega + 180, omega + 180) twin: the two shifts together
    leave the sky orbit unchanged (see `thiele_innes`), so samples are folded
    by moving both. A sample that needed an odd number of shifts is in the
    other node mode, which is still the same orbit on the sky.

    Parameters
    ----------
    omega, Omega : array_like
        Sample angles, degrees.
    Omega_ref : float
        Node to fold around, degrees.

    Returns
    -------
    omega, Omega, mirror : numpy.ndarray
        Folded angles (omega in [0, 360)) and a bool array, True for samples
        that needed an odd number of 180 degree shifts.
    """
    omega, Omega = np.asarray(omega, float), np.asarray(Omega, float)
    k = np.floor((Omega - Omega_ref + 90.0) / 180.0)
    return np.mod(omega + 180.0 * k, 360.0), Omega - 180.0 * k, (k % 2 == 1)


def mirror_fraction(Omega, Omega_ref):
    """Posterior weight of the node mirror mode, before folding.

    Parameters
    ----------
    Omega : array_like
        Sample nodes, degrees.
    Omega_ref : float
        Reference node, degrees.

    Returns
    -------
    float
        Fraction of samples more than 90 deg (modulo 360) from ``Omega_ref``.
    """
    return float(np.mean(fold_samples(np.zeros_like(np.asarray(Omega, float)), Omega, Omega_ref)[2]))
