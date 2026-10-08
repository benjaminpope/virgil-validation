# Comparing fitted orbits with a reference when there are no radial velocities

A general note. It defines what an orbit fit without RVs is compared on, and how
the comparison is reported. Per-system plans point here. Gl 229 Ba–Bb
(GRAVITY-only fits against Xuan et al. 2024) is the worked example. The comparison
functions are text-only so far: no science code is added by this note.

## 1. The degeneracy

Relative sky positions $(\Delta\alpha^*,\Delta\delta)$ depend on $(P, e, t_p)$ and
the Thiele–Innes constants

$$A = a(\cos\omega\cos\Omega - \sin\omega\sin\Omega\cos i),\quad
B = a(\cos\omega\sin\Omega + \sin\omega\cos\Omega\cos i),$$
$$F = -a(\sin\omega\cos\Omega + \cos\omega\sin\Omega\cos i),\quad
G = -a(\sin\omega\sin\Omega - \cos\omega\cos\Omega\cos i)$$

Axis assignment: $A,F$ are the $\Delta\delta$ coefficients of $r\cos f$ and
$r\sin f$, and $B,G$ the $\Delta\alpha^*$ ones (expanded from `sky_position` in
`src/crosscheck/orbits.py`). The invariance argument below holds for any
convention, but statistic B does not: a reference that quotes $A..G$ with
$x$ = RA instead of $x$ = Dec swaps $A\leftrightarrow B$ and $F\leftrightarrow G$
and the test fails without the cause being obvious. So both sides are computed
from Campbell elements with one function, never from quoted Thiele–Innes
constants.

Both $\omega$ and $\Omega$ enter only through the angle sums and differences
above, so $(\omega+180^\circ,\ \Omega+180^\circ)$ leaves $A,B,F,G$ unchanged.
Without RVs, $\omega$ and $\Omega$ are identifiable only jointly, up to that
shift. $P$, $e$, $t_p$, $a$ and $i$, including the sense of motion, are
identifiable.

A shift of $\omega$ by $180^\circ$ alone flips the sign of all four constants.
That is the primary/secondary labelling convention ($\omega$ of the secondary or
of the primary). It is a real difference of convention, not a mode, and it must
be checked and stated for every reference (Xuan et al. quote $\omega_{\rm
secondary}=\omega_{\rm primary}+180^\circ$).

Comparing full Campbell elements with a published orbit therefore gives spurious
pulls when a fit lands in the mirror mode. The Gl 229 Ba–Bb GRAVITY-only fits
gave $\omega$ pulls of $-90\sigma$ and $\Omega$ pulls of $-28\sigma$ for a mode
that fits the sky positions equally well.

With RVs, or a Gaia acceleration sense, the degeneracy is broken: compare the
full 3D elements as before.

## 2. Statistic A: the sky track (primary)

Always applicable, including $e\to0$ and short arcs.

Draw posterior-predictive relative positions $\mu=(\Delta\alpha^*,\Delta\delta)$
from our samples and from the reference orbit, at every observed epoch (flagged
epochs excluded from fits but reported) and at held-out epochs. Per epoch, with
$\Delta\mu=\bar\mu_{\rm ours}-\bar\mu_{\rm ref}$ and sample covariances,

$$d^2 = \Delta\mu^\top (C_{\rm ours}+C_{\rm ref})^{-1}\Delta\mu \ \sim\ \chi^2_2 .$$

Jointly, the stacked $2N$-vector lies near a manifold of about 7 dimensions
(the orbit has $P,e,t_p,A,B,F,G$), so $C$ is nearly rank-deficient. The rank-7
statement holds only to first order, for one posterior: $C_{\rm ours}$ and
$C_{\rm ref}$ are tangent at different points, so their sum can reach rank 14,
and sampling noise and nonlinearity fill in the small eigenvalues. Truncating
to the broadest directions would drop the components of $\Delta\mu$ where the
posteriors are tightest, which is where a disagreement matters most. So:

- Truncate by an eigenvalue threshold (eigenvalues of $C_{\rm ours}+C_{\rm
  ref}$ below $10^{-6}$ of the largest, fixed in the criteria file), not by a
  fixed $r$. Let $r$ be the retained rank.
- $d^2_{\rm joint}=\Delta\mu^\top (C_{\rm ours}+C_{\rm ref})^{+}_{r}\Delta\mu\ \sim\ \chi^2_r$.
- Always report the residual $\lVert\Delta\mu_\perp\rVert$ outside the retained
  subspace beside $d^2_{\rm joint}$, in units of the per-epoch position errors.
  The comparison fails if it exceeds 1 (criteria file may tighten this).
- Statistic B is the joint test; statistic A's joint row is a cross-check on it.

The reference distribution is $\chi^2_r$ under the null that the two posteriors
agree; it is approximate where the posteriors are non-Gaussian, and the report
says so.

Also report the maximum separation between the two median tracks over one full
period, as a fraction of $a$.

**Reference samples.** Use the published posterior samples where available.
Otherwise do not draw from the marginal errors independently: visual-orbit
elements are strongly correlated ($P$–$t_p$, $a$–$i$–$e$, $\omega$–$\Omega$),
and independent draws inflate $C_{\rm ref}$ in the predicted positions, so
$d^2$ shrinks and a wrong fit passes. With no published samples or covariance,
no statistic against that reference is scored, per-epoch rows included. Report
instead the $d^2$ using $C_{\rm ours}$ alone (the stricter bound) and the one
with the independent-draw $C_{\rm ref}$, both labelled unscored.

Open question: whether a Xuan et al. 2024 posterior (samples or covariance) is
public for Gl 229. If not, every Gl 229 statistic A and B row falls under this
case and is reported, not scored, until it is found or reconstructed.

## 3. Statistic B: the projected elements

Use when $e\gtrsim0.1$ and the arc covers $\gtrsim\tfrac12$ period.

$$\theta=(P,\ e,\ t_p,\ A,\ B,\ F,\ G),$$

all invariant under the ambiguity. Reference $t_p$ to the periastron nearest the
data's mean epoch, to avoid wrapping by $P$. Then

$$d^2=\Delta\theta^\top (C_{\rm ours}+C_{\rm ref})^{-1}\Delta\theta\ \sim\ \chi^2_7,$$

with covariances from samples. Both orbits are in mas, so distance does not
enter.

For $e\to0$, $t_p$ (and $\omega$ at fixed $A,B,F,G$) is undefined and the
covariance is singular along it: use $A$ only there (statistic A, whose
predicted positions do not depend on $t_p$ separately from the phase).

## 4. Reporting (not scored)

Corner plots use Campbell elements folded to a canonical representative: map
each sample to $\Omega\in[\Omega_{\rm ref}-90^\circ,\Omega_{\rm ref}+90^\circ)$,
shifting $\omega$ by $180^\circ$ whenever $\Omega$ is shifted. The cut then sits
away from the posterior mass. Plot $P$, $e$, $i$, $a$, $t_p$, $\Omega$ and
$\omega$, with the reference as truth lines.

Report the posterior weight of each mode before folding: the mirror-mode
fraction. Pulls on folded $\omega$ and $\Omega$ are descriptive only.

## 5. Bad epochs

Single-epoch alias peaks are flagged, not used as constraints. Worked example,
Gl 229 Ba–Bb: 2024-02-27 and 2024-03-28 are single-night alias peaks that Xuan
et al. also flag (weak signal; a failing fringe tracker on 03-28).

2024-12-18 is listed separately. Xuan et al. do not flag it, and
`plan_eso_binaries.md` classes it as a disagreement to be ruled on (shared-flux
scorer or virgil-validation#69). Until that ruling it is excluded from the orbit
fit but kept as an open per-epoch disagreement, with a scored data-vs-reference
row: the offset of virgil's best peak from the reference position, and the
$\Delta$loss between them. The orbit-vs-orbit track comparison never involves the
measured position on that night, so it cannot detect a peak-finding error
there; this row can.

Statistic A compares the orbit fit's predicted positions with the reference
orbit's, so flagged epochs still get a track comparison, which is orbit against
orbit only.

Lead every comparison with the raw $\chi^2/N$ on the quoted errors, before any
error rescaling.

## 6. Registration

New campaigns register the choice of A, B or both, and the thresholds (for
example per-epoch $d^2$ $p>0.01$, Holm-corrected across epochs; joint $p>0.01$),
in their criteria file before fitting. Holm is a correction for false
rejections, so here it loosens the test: with $N$ epochs the smallest $p$ needs
only $p>0.01/N$, and with about 10 GRAVITY epochs one epoch at roughly $3\sigma$
still passes. The criteria file states the effective per-epoch threshold, and
requires the joint statistic as well, so a run of moderate per-epoch offsets
cannot pass. Systems already fitted (Gl 229) use the comparison post hoc,
labelled as not preregistered.

## 7. Implementation pointers (text only)

- `src/crosscheck/orbits.py` has the Campbell sky projection (`sky_position`);
  Thiele–Innes is to be added, with the axis assignment of §1. The comparison
  functions belong there: numpy only, never importing virgil.
- Our samples come from the orbit-fitting outputs,
  `~/data/orbit_fits/<system>/<run>/{fit.json,samples.npz}` (schema in
  `~/data/orbit_fits/SCHEMA.md`).
- Deliverable tests. Positive control: a synthetic orbit and its mirror
  $(\omega+180^\circ,\Omega+180^\circ)$ must give $d^2\approx0$ in A and B, but
  large raw $\omega$ and $\Omega$ differences. That alone cannot fail (a statistic
  that always returns 0 passes it), so it is paired with `kind="control"` cases
  that must give a large $d^2$ in both A and B:
  - $\omega+180^\circ$ alone (all four constants flip sign: the
    primary/secondary labelling error of §1);
  - $i\to180^\circ-i$ (reversed sense of motion);
  - a small $\Delta t_p$ or $\Delta P$ at about $3\sigma$ of the synthetic
    posterior, which checks that the statistic responds at the scale it is used.
