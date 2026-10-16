"""Other people's radial-profile visibility codes, called as documented.

frank (Jennings et al. 2020), vis-r (Kennedy 2025) and FRAP (Yoshida et al.),
each installed from PyPI / GitHub into this repository's environment
(``pip install frank astro-frap "vis-r @ git+https://github.com/drgmk/vis-r"``).
This module may not import virgil. Units: radii in arcsec for the codes,
spatial frequencies in cycles per radian (= "wavelengths" for u, v).

Conventions were read from the packages' own sources and then pinned by the
benchmark (benchmarks/polar_disk/v1_forward.py):

* frank: ``DiscreteHankelTransform(Rmax, N)`` (Baddour & Chouinard 2015);
  ``coefficients(q)`` is the matrix Y_km of ``H[f](q) = int f J0(2 pi q r)
  2 pi r dr``, with ``Rmax`` in radians and ``q`` in cycles/rad, acting on
  ``f`` at the collocation points ``Rnk``.
* vis-r: ``setup_dht(sz, u, v)`` builds frank's matrix at a grid of ``q``
  spaced by ``uvmin`` (set by the field size ``sz`` in arcsec) and
  ``lnprob`` interpolates it linearly onto ``ruv`` from ``uv_trans``;
  the vertical structure of an inclined ring is the factor
  ``exp(-0.5 (2 pi h r0 sin(i) u_minor)^2)`` (``h`` the aspect ratio, ``r0``
  the ring radius), the visibility of a Gaussian slab of scale height
  ``h r0`` seen at inclination ``i``.
* FRAP: ``model(incl, r_out, N_GP)`` with Fourier-Bessel nodes
  ``r_k = r_out j_{0,k} / j_{0,N+1}`` and Hankel matrix
  ``H = (4 pi r_out^2 / (j_{0,N+1}^2 J1(j_{0,k})^2)) J0(2 pi q r_out
  j_{0,k} / j_{0,N+1})`` set by ``set_visibility`` (here without the primary
  beam, ``pbcor=None``), applied to the intensity at the nodes.
"""

import contextlib
import io

import numpy as np

ARCSEC = np.pi / 180.0 / 3600.0  # radians per arcsec


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ---------------------------------------------------------------- frank
class Frank:
    """frank's discrete Hankel transform of a profile ``f(r_mas)`` at deprojected frequencies."""

    def __init__(self, r_max_mas, n, q):
        from frank.hankel import DiscreteHankelTransform

        self.dht = DiscreteHankelTransform(r_max_mas * 1e-3 * ARCSEC, n)
        self.r_mas = self.dht.r / (1e-3 * ARCSEC)  # nodes in mas
        self.q = np.asarray(q, float)
        self.matrix = self.dht.coefficients(q=self.q)

    def visibility(self, f):
        """Normalized: V(q)/V(0), with V(0) from the same transform."""
        fk = f(self.r_mas)
        v = self.matrix @ fk
        v0 = self.dht.coefficients(q=np.array([0.0])) @ fk
        return v / v0[0]

    @property
    def q_max(self):
        return self.dht.Qmax


# ---------------------------------------------------------------- vis-r
class VisR:
    """vis-r's profile -> visibility step (``vis_r_emcee.lnprob`` for one radial component).

    ``profile(r_arcsec)`` is a vis-r ``r_prof_*`` function bound to its
    parameters. ``u``, ``v`` in wavelengths; ``sz`` is vis-r's field-size
    argument in arcsec (it sets the q grid and so the interpolation error).
    """

    def __init__(self, u, v, inc_deg, pa_deg, sz, nhpt=300):
        from vis_r import functions

        self.f = functions
        with quiet():
            self.Rnk, self.Qzero, self.Ykm = functions.setup_dht(sz, u, v, nhpt=nhpt)
        self.urot, self.ruv = functions.uv_trans(u, v, np.deg2rad(pa_deg), np.deg2rad(inc_deg))
        self.sin_i = np.sin(np.deg2rad(inc_deg))

    def visibility(self, profile, aspect=0.0, r0_arcsec=0.0):
        arcsec2pi = ARCSEC * 2 * np.pi
        f = profile(self.Rnk / ARCSEC)
        fth = self.Ykm @ f
        fth = fth / fth[0]  # vis-r normalizes on the shortest (zero) baseline
        vis = np.interp(self.ruv, self.Qzero, fth)
        if aspect > 0:
            rz = aspect * r0_arcsec * self.sin_i * self.urot * arcsec2pi
            vis = vis * np.exp(-0.5 * np.square(rz))
        return vis


# ----------------------------------------------------------------- FRAP
class Frap:
    """FRAP's Hankel matrix at deprojected frequencies, without the primary beam."""

    def __init__(self, r_out_mas, n_gp, q):
        import frap

        self.model = frap.model(incl=0.0, r_out=r_out_mas * 1e-3, N_GP=n_gp)
        q = np.asarray(q, float)
        with quiet():
            self.model.set_visibility(
                "b", {0: q}, {0: np.zeros_like(q)}, {0: np.ones_like(q)},
                f_s=0.1, f_mean=1.0, nu=[2.3e11], Nch=1, pbcor=None,
            )
        self.obs = self.model._observations["b"][0]
        self.r_mas = np.asarray(self.model._r_GP) * 1e3  # nodes in mas
        self.h = np.asarray(self.obs.H)

    def visibility(self, f):
        """Normalized: ``H @ f`` over its q -> 0 limit (sum of the weights times f)."""
        fk = f(self.r_mas)
        v = self.h @ fk
        v0 = self.model._HT_prefactor @ fk
        return v / v0
