# Binary orbits: GRAVITY and PIONIER

Many stars are pairs too close to split with a single telescope: a few milliarcseconds apart. The VLTI
combines the light of four telescopes, and in GRAVITY or PIONIER data a companion shows up as a ripple in
the closure phases and visibilities. The spacing and size of the ripple give the companion's separation,
direction and brightness. Each night gives one position, and a Keplerian orbit is the ellipse, period and
timing that threads those positions together.

These pages ask whether virgil, given the archival data behind a published orbit, gets the authors' answer.
The data are the authors' own, so a difference comes from the analysis: the model, the likelihood, the
optimizer, a convention such as which star an angle belongs to, or calibration choices the archive does
not record. Nothing is tuned towards the published numbers.

## How the orbits were fitted

1. **Data.** Calibrated OIFITS files from the JMMC OiDB database, or public ESO Phase 3 products, as
   listed under each binary.
2. **Positions night by night.** A multi-start search finds the peaks of the likelihood on each night.
3. **One orbit across all nights.** Each candidate is refitted as a single Keplerian orbit over every
   night, and the lowest loss is the best (maximum a posteriori) orbit. Where sampling has been run,
   NUTS gives the posterior medians and errors.
4. **Comparison.** The published orbit is converted to virgil's conventions and drawn beside ours. The
   curves come from our own independent Kepler solver, never from virgil, and the pass limits of the
   comparison tables were fixed before the fits.

When the nights are sparse, several periods can thread the same positions almost equally well: these are
period aliases. The search scans a window of candidate periods and never starts at the published orbit.
The windows are meant to come from the data; the Gl 229 window (10 to 14 days) was taken from the paper,
so for that system the winning period is a choice among a few aliases, not a blind discovery.

## The pages

- **[Orbit gallery](binaries.md)**: one tile per binary, with the orbit on the sky, separation and
  position angle through the orbit, and every fitted quantity against the paper. Tiles marked *awaiting*
  are onboarded for the blind orbit solver but not yet fitted.
- **[Period aliases](aliases.md)**: for each undersampled system, the alias bands, the evidence for each,
  and which one wins.

The machinery underneath is tested part by part under Function tests:
[Kepler orbits](kepler.md), [Against orbitize!](orbitize.md) and [Multi-epoch orbits](epochs.md).
