---
applies_to:
  slug: hr6819_blind
visibility: sealed
source: 2025A&A...694A.208K
generated_by: campaigns/build_registries.py
---
Published solution: `oidb/trackb/hr6819_blind.json` (Klement, R.; et al. 2025, A&A 694, A208).

Kind (from the published orbit): alias_track.

Third light: Be disk (resolved)

Caveats recorded with the reference: A resolved Be disk and a third body make this a reported comparison, never scored. The data are the provider's own calibrated OiDB files (the calibrators are not in them), binned by 8 channels (scripts/gravity_bin_calibrated.py), with the virgil-vlti HIGH template laid out on them for gains and closure offsets. The two 2021 split-polarisation nights are refused by the combined-polarisation template and left out. The older tile of this target (T1-T5, oidb_fit) stays until this fit is checked.

Conventions: Same star as virgil: the companion (pre-sd) relative to the Be star, omega from the ascending node (the node where the companion recedes), Omega as PA north through east. The paper labels Table 3's omega 'omega_Be', but that number is the companion's: with the same 12 positions, Table 2 (astrometry only) gives omega_pre-sd = 117.6+/-8.9, Omega = 23.1+/-2.1 and Table 3 gives 111.9+/-9.0, Omega = 25.4+/-2.1; a true omega_Be would be 297.6 with the same Omega, which Table 3 is not; the (Omega+180, omega+180) twin of (23.1, 297.6) is (203.1, 117.6), so this argument assumes the paper uses ONE Omeg
