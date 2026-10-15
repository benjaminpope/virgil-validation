# Triage of the ESO-archive binaries sheet

Planning note, off the site. Ben's spreadsheet of VLTI and VLT-masking binaries (`eso_vlti_sam_binaries.xlsx`, ESO TAP queried 2026-10-04) holds 138 systems: 93 "clean" and 45 "other". This note puts each in a tier for the blind orbit solver (`design/orbit_solver_v2.md`) and the gallery. Nothing here is onboarded. Ben chooses.

## Is it the Track B catalogue?

Not the same file, but its parent. The sheet's counts (138 systems, 93 + 45, 1409 observation sets, 98 papers) are the ones `design/plan_eso_binaries.md` quotes for the catalogue the Track B session started from. Track B's `~/data/eso_binaries/trackb/catalogue.json` (built 2026-10-07) is a second, paper-driven catalogue of 91 records (20 clean, 70 not clean, 1 owned elsewhere). Each record is a paper's table, transcribed with per-epoch positions, and checked for Phase 3 coverage. The sheet is system-driven: each row is a target with its archive nights.

What agrees:

- The sheet's Phase 3 counts match a fresh `ivoa.ObsCore` query by cone for the five Araucaria-I SB2s of this PR: LL Aqr 32 GRAVITY products, HD 9312 13, AK For 22, HD 224974 13, V963 Cen 16 GRAVITY + 10 PIONIER.
- The orbit periods and `Orbit_status` agree with the Track B element records where both exist.

What differs:

- **Overlap.** By name, 81 of the sheet's 138 systems have a Track B record. The 57 without one are mostly the sheet's "other" group, the Trapezium and the massive stars, plus GJ 65 AB, α Psc, 97 Aqr, HD 5143, HD 16753, γ2 Vel, ρ Oph A, κ Tuc A, HD 148937 and HR 6819. Track B has only four of the six Gardner 2022 triples (HD 29573, 31297, 48581, HD 199766) and lumps the "other" systems into one record.
- **Only in Track B.** KW Hya, HIP 104987 (α Equ), HD 199766, SU Cyg, Y Car, BG Cru, HD 221950 and TYC 1703−394−1 have no row in the sheet. KW Hya and α Equ (the latter already in the alias track) matter most.
- **Cleanliness means different things.** The sheet's "clean" is about circumstellar emission; Track B's is also about resolution (λ/2B), the fibre and Phase 3. That is why the five Araucaria-I SB2s are "clean" in the sheet and "not clean" in Track B.
- **Public.** The sheet's `Public` is mostly `unknown`; Track B checked release dates. Where the two disagree, trust Track B.
- **κ Vel.** The sheet gives e = 0.1764 ± 0.0004, Track B's `system.json` 0.1761 ± 0.0004. A transcription difference to settle against Rowan+2026 Table 3 before either is used (the PR's `oidb/trackb/kap_vel.json` is the one to check).
- **Sheet caveat 2 holds.** Early PIONIER visitor-mode programmes are absent from the ESO archive, and Track B hit the same wall (HIP 14157, HIP 20601, χ Lup, HD 54662).

## Method

Every system is scored on four things, read from the sheet's observation-sets tab (science rows only; calibrator rows left out):

1. a published visual or joint orbit with full elements (not just P and a from ORB6);
2. several epochs;
3. public **Phase 3 OIFITS** (GRAVITY or PIONIER) for them, as against raw frames only (AMBER, NACO and MATISSE have none): a raw-only system needs the ESO pipeline first and is not a one-line onboarding;
4. a stress test for the blind solver: period aliases, annual gaps, sub-resolution separations, short arcs, a triple, or mixed instruments.

Tiers:

- **V (validation):** all four.
- **D (demonstration):** famous or striking, or a triple, or a resolved or massive primary, with usable data, even where the published elements are partial or the geometry is not a clean point-source pair.
- **X (skip)**, with the reason.

Coverage already in hand is not re-scored: the gallery (Gl 229, HR 6819, σ Ori and the CHARA tiles), the alias track (HD 41255, HD 70937, HD 188088, HD 210763, ο Leo, ψ Cen, AL Dor), this PR's onboarding set (LL Aqr, HD 224974, AK For, V963 Cen, HD 9312, κ Vel, NN Del) and the other workers' systems (9 Sgr, ζ Boo, η Oph). They are listed at the end.

"Nights by instrument" is the sheet's count of distinct science nights. "Phase 3" is the number of Phase 3 products by instrument (G = GRAVITY, P = PIONIER), which may exceed the nights because a night can have several products. The sheet's own caveat applies: element values come from abstracts, ORB6 and whereistheplanet, not from the papers' tables, so "orbit" here means "reported", and every V needs its table read first.

Tier counts (of the sheet's 138): 9 V, 21 D, 88 X, 20 already covered. KW Hya, the tenth V below, is not in the sheet.

## Tier V: validation

9 systems (9 clean, 0 other).

| System | Type | Orbit (sheet) | Nights by instrument | Phase 3 | Verdict |
|---|---|---|---|---|---|
| HD 150136 | Massive triple-O | orbit; 8.61±0.02 yr | AMBER 3, GRAVITY 5, NACO 3, PIONIER 25 | G 6; P 14 | O3.5+O6 pair on an 8.6 yr, e 0.68 orbit around an unresolved 2.7 d binary (a triple); 25 PIONIER + 5 GRAVITY nights, 20 Phase 3 products; stress: 9-yr gaps, e 0.68, third light |
| TZ For | Eclipsing giant binary | orbit; 75.7 d | GRAVITY 2, PIONIER 16 | G 5; P 11 | 75.7 d giant binary (Gallenne 2016/18), 16 nights, 16 Phase 3 products from both GRAVITY (5) and PIONIER (11); stress: mixed instruments, 1.6–2.9 mas; staged in Track B but its manifest lists 0 epochs, so fix that first |
| V4090 Sgr | Eclipsing binary (Araucaria) | orbit | PIONIER 18 | P 37 | Araucaria PIONIER eclipsing SB1, 6 of 6 epochs in Phase 3 (37 products); 1.2–1.6 mas, at the resolution limit, with masses tied to a Gaia prior |
| AI Phe | Eclipsing binary (Araucaria) | unverified | PIONIER 25 | P 65 | 12 published PIONIER epochs, 11 in Phase 3 (65 products over 25 nights); sub-resolution 0.8–1.4 mas with a known RV tertiary (≈109 yr drift); sheet doubts it is a fitted target, so confirm in Gallenne 2019 first |
| δ Cir | Massive triple with eclipsing pair | orbit; 1603.24±0.19 d (outer) | GRAVITY 3, NACO 2, PIONIER 12 | G 3; P 5 | outer 1603 d orbit of an O+eclipsing-pair triple, 8 of 12 epochs in Phase 3 (GRAVITY 3 + PIONIER 5), 2011–2026; stress: triple, mixed instruments, yearly gaps; data from 2026 partly proprietary |
| 97 Aqr | Intermediate-mass binary + inner companion | orbit; ORB6 64.3 yr | GRAVITY 16, PIONIER 1 | G 16 | 64 yr outer orbit with a new inner companion (Gardner 2022); 16 GRAVITY nights, 16 Phase 3, 2014–2021; stress: a short arc (7 of 64 yr) |
| HD 29573 | Intermediate-mass binary + inner companion | orbit; ORB6 50.1 yr | GRAVITY 11 | G 14 | ARMADA triple with a 59.2 d inner pair; 11 GRAVITY nights, 14 Phase 3 products, public; stress: aliases at ~1 yr sampling, but a triple scene (A+Ba+Bb) |
| HD 31297 | Intermediate-mass binary + inner companion | orbit; ORB6 84.7 yr | GRAVITY 10 | G 11 | ARMADA triple with a 29.5 d inner pair; 10 GRAVITY nights, 11 Phase 3, public; stress: aliases, triple scene |
| HD 48581 | Intermediate-mass binary + inner companion | orbit | GRAVITY 14 | G 18 | ARMADA triple with a 4.84 d inner pair (candidate); 14 GRAVITY nights to 2024, 18 Phase 3, public; stress: aliases, later epochs, triple scene |

## Tier D: demonstration

21 systems (6 clean, 15 other).

| System | Type | Orbit (sheet) | Nights by instrument | Phase 3 | Verdict |
|---|---|---|---|---|---|
| δ Vel Aa–Ab | Eclipsing A-star binary (triple) | orbit; ORB6 0.124 yr | AMBER 8, NACO 1, VINCI 4 | raw only | eclipsing A-star triple, famous; AMBER/VINCI raw only, but one NACO SAM night is already on disk (`naco/oifits/delvel`); ORB6 45 d orbit gives a reference |
| θ1 Ori C | Young massive O binary | orbit; ~11.3 yr | AMBER 20, GRAVITY 3, MIDI 1, PIONIER 1, VINCI 3 | G 1 | the famous 11.4 yr O+O orbit; AMBER raw plus one GRAVITY Phase 3 product, so a gallery tile from published positions, not a fit |
| HD 167971 | Massive hierarchical triple | orbit; ORB6 21.6 yr | AMBER 4, GRAVITY 2, NACO 1, PIONIER 9 | P 5 | massive hierarchical triple with eclipsing inner pair (P 21.6 yr outer, ORB6); PIONIER Phase 3 (5) but none for the published epochs, some data proprietary |
| HD 93206 A (QZ Car) | Massive quadruple | partial | GRAVITY 4, NACO 1, PIONIER 16 | G 3; P 12 | massive quadruple, famous; 15 GRAVITY/PIONIER Phase 3 products on a partial orbit, some proprietary |
| GJ 65 AB | Nearby M-dwarf binary (dual-field) | orbit; ORB6 26.3 yr | GRAVITY 83, MATISSE 1, PIONIER 4 | G 629 | UV Cet pair: 83 GRAVITY nights (629 Phase 3 products), 26 yr orbit, 2″ so dual-field (positions as data, not single-fibre visibilities); a showpiece orbit |
| α Psc | Hierarchical triple | orbit; ~25 d (Ba–Bb) | GRAVITY 11 | G 3 | Alrescha: hierarchical triple with a ≈25 d inner pair; GRAVITY 11 nights but 3 Phase 3 products, public; 1947 yr outer orbit |
| γ2 Vel (WR 11) | Massive WR+O binary | orbit; ~78.5 d (spectroscopic, lit.) | AMBER 11, GRAVITY 4 | G 4 | famous WR+O (78.5 d, e 0.32) with a GRAVITY 2026 full visual orbit, but only 4 GRAVITY Phase 3 products (AMBER raw); 4 nights make an alias showpiece rather than a test |
| X Sgr | Cepheid + companion | 1–2 epochs | AMBER 31, GRAVITY 8, MATISSE 1, MIDI 1, PIONIER 22 | G 9; P 33 | Cepheid + companion with a resolved, pulsating primary: 31 AMBER, 22 PIONIER (33 Phase 3), 8 GRAVITY nights; no orbit published |
| Herschel 36 (Her 36 A) | Young massive triple | orbit; 492.81±0.69 d (outer, spectroscopi | AMBER 5, GRAVITY 4, PIONIER 2 | G 5 | young massive triple in the Lagoon, GRAVITY 4 nights (5 Phase 3), some proprietary |
| AX Cir | Cepheid + companion | 1–2 epochs | GRAVITY 1, MIDI 3, PIONIER 19 | G 1; P 64 | Cepheid + companion: 19 PIONIER nights, 64 Phase 3 products, but 1–2 epoch positions only |
| HD 93129 Aa,Ab | Very massive O+O binary | partial | GRAVITY 1, NACO 5, PIONIER 18 | G 1; P 22 | the very massive O2If* pair, e > 0.92; 18 PIONIER nights (22 Phase 3), a partial orbit, some proprietary |
| U Aql | Cepheid + companion | partial; ORB6 5.02 yr | GRAVITY 6, MATISSE 1, PIONIER 13 | G 5; P 25 | Cepheid + B companion on a 5.0 yr, ~10 mas orbit; GRAVITY 6 + PIONIER 13 nights, 30 Phase 3, some proprietary; resolved pulsating primary |
| S Mus | Cepheid + companion | partial | AMBER 2, GRAVITY 1, PIONIER 15 | P 54 | Cepheid with a B companion, a partial orbit; 15 PIONIER nights (54 Phase 3), some proprietary |
| V350 Sgr | Cepheid + companion | orbit | GRAVITY 1, PIONIER 4 | P 16 | Cepheid + companion with a published 3D orbit (Evans+2025); PIONIER 4 nights (16 Phase 3), some proprietary |
| HD 98800 (Aa–Ab, Ba–Bb) | PMS quadruple | orbit | AMBER 1, MIDI 4, NACO 1, PIONIER 12, VISIR 1 | P 16 | pre-main-sequence quadruple (206 yr + 0.86 yr pairs); PIONIER 12 nights (16 Phase 3); disk-bearing, so a demonstration |
| GG Tau Ab | PMS multiple | orbit; ~8 yr (Ab) | AMBER 1, GRAVITY 6, MIDI 13, NACO 2, PIONIER 3 | G 10 | PMS multiple in the famous GG Tau ring; GRAVITY 6 nights (10 Phase 3); circumstellar light rules out validation |
| HD 148937 | Massive binary (merger product) | orbit | GRAVITY 3, NACO 2, PIONIER 6 | P 5 | O6f?p magnetic merger product, e 0.75 (Frost+2024 Science); PIONIER 6 nights, 5 Phase 3 |
| WR 48 | WR + wide companion | 1–2 epochs | GRAVITY 14, NACO 1, PIONIER 2 | G 24 | WC5+B0 with a wide companion; 14 GRAVITY nights, 24 Phase 3, dust |
| ρ Oph A | B binary with magnetosphere | orbit | GRAVITY 9 | G 8 | magnetospheric B binary, 9 GRAVITY nights (8 Phase 3), a visual orbit; flagged because of the magnetosphere |
| κ Tuc A | F star + low-mass companion | orbit | GRAVITY 3, MATISSE 16, PIONIER 7 | G 4; P 7 | F star with a very eccentric (e 0.94) low-mass companion; GRAVITY, MATISSE and PIONIER (11 Phase 3 products), a visual orbit |
| HD 206893 B | Star + BD/planet (dual-field) | orbit; post. 29.5 yr | GRAVITY 21, SPHERE 1 | G 234 | the best-sampled dual-field system (21 GRAVITY nights, 234 Phase 3 products); a gallery tile from published positions |

## Tier X: skip

88 systems (60 clean, 28 other).

Reasons in short: no Phase 3 product (raw only), one or two epochs, orbit periods far longer than the data span, dual-field geometry outside the single-fibre scope, or circumstellar light.

| System | Type | Orbit (sheet) | Nights by instrument | Phase 3 | Verdict |
|---|---|---|---|---|---|
| NU Ori (HD 37061) | Massive hierarchical triple | partial; 476±1 d (C around Aab) | AMBER 3, GRAVITY 2, PIONIER 7 | G 2; P 2 | partial orbit of a triple; 2 Phase 3 products of 12 nights, rest raw AMBER/PIONIER |
| HD 93250 | Massive O+O binary | orbit | AMBER 3, NACO 2, PIONIER 14 | P 6 | sub-resolution (0.9–1.5 mas) and only 3 of 15 published epochs have Phase 3; Track B rejected it |
| χ Lup | HgMn + Am SB2 | orbit; 15.25 d | GRAVITY 2 | G 2 | 15 d HgMn SB2 from 2012 PIONIER (raw); the only Phase 3 are 2 unrelated GRAVITY nights |
| HD 152246 | Massive hierarchical triple | orbit; 470 d (outer) | NACO 2, PIONIER 9 | P 3 | triple, outer orbit 470 d, e 0.87; 9 PIONIER nights but 3 Phase 3 products |
| σ Sco | Young massive multiple | orbit; ORB6 0.0904 yr | AMBER 1 | raw only | one AMBER night, no Phase 3; a famous multiple but nothing to fit |
| HR 6027 | Young B multiple | detection | AMBER 1 | raw only | detection only, one AMBER night |
| HIP 14157 | F–K SB2 (Gaia mass check) | orbit | PIONIER 11 | raw only | Halbwachs+2016 SB2s from 2013–14 PIONIER visitor data, absent from the ESO archive (sheet caveat 2); only reachable via OiDB if at all |
| HIP 20601 | F–K SB2 (Gaia mass check) | orbit | PIONIER 6 | raw only | Halbwachs+2016 SB2s from 2013–14 PIONIER visitor data, absent from the ESO archive (sheet caveat 2); only reachable via OiDB if at all |
| HIP 117186 | F–K SB2 (Gaia mass check) | orbit | PIONIER 7 | raw only | Halbwachs+2016 SB2s from 2013–14 PIONIER visitor data, absent from the ESO archive (sheet caveat 2); only reachable via OiDB if at all |
| ξ Tau | Compact hierarchical quadruple | orbit | AMBER 1 | raw only | compact quadruple but one AMBER night, no Phase 3 |
| HD 54662 | Massive O binary | orbit; 2103.4 d | AMBER 6, NACO 2, PIONIER 8 | raw only | 2103 d orbit but no Phase 3 for any published epoch (pre-2015 AMBER/PIONIER) |
| HD 152233 | Massive O binary | orbit; ORB6 2.39 yr | AMBER 4, GRAVITY 1, NACO 2, PIONIER 10 | G 1; P 1 | no published epoch in Phase 3; separations down to 0.8 mas |
| HD 152247 | Massive O binary | orbit | AMBER 7, NACO 2, PIONIER 6 | raw only | no published epoch in Phase 3; separations down to 0.8 mas |
| HD 152314 | Massive O binary | orbit | AMBER 4, NACO 3, PIONIER 6 | P 3 | staged in Track B, but only 1 of 10 published epochs has Phase 3 |
| HD 168137 | Massive O binary | orbit; ORB6 5.03 yr | AMBER 6, PIONIER 11 | P 6 | staged in Track B (5.0 yr, 2.6–6.8 mas), but only 2 of 11 published epochs have Phase 3 |
| CPD−47 2963 (CD−47 4551) | Massive O binary | orbit; ORB6 1.8 yr | NACO 1, PIONIER 12 | P 3 | only 2 Phase 3 epochs, both at 1.03 mas (below λ/2B); possible 59 d SB1 |
| 41 Eri | HgMn SB2 | orbit; ~5 d | GRAVITY 2, PIONIER 2 | G 2 | 1.6–1.9 mas near-twin SB2, no Phase 3 for the 2012–14 PIONIER epochs |
| V923 Sco | Detached eclipsing binary | orbit | AMBER 14, PIONIER 1 | raw only | eclipsing SB2, no Phase 3 at all (AMBER and unreleased GRAVITY) |
| 21 Hya | Binary (Araucaria) | unverified | PIONIER 9 | P 18 | status of the orbit unverified in the sheet; Track B catalogue has no record of it |
| ε Lup A | Doubly magnetic B binary | partial; ORB6 737 yr | PIONIER 2 | raw only | 736 yr partial orbit, 2 PIONIER nights, no Phase 3 |
| KQ Vel (HD 94660) | Ap star + NIR companion | 1–2 epochs | GRAVITY 8, PIONIER 1 | G 9; P 1 | staged in Track B, one epoch at 18.7 mas (wide, long period) |
| HIP 87813 (HD 163336) | A-star multiple (VAST) | partial | GRAVITY 2 | G 2 | partial orbit, 2 GRAVITY nights in 2021–22 |
| FK Aqr | M-dwarf SB2 | partial | AMBER 18, PIONIER 7 | raw only | M-dwarf SB2 from AMBER/PIONIER 2007–14, no Phase 3 |
| BD−07 5977 | sdO/B + MS wide binary | 1–2 epochs | GRAVITY 2 | G 1 | Deshmukh+2026 sdO/B + MS pairs: 1–2 GRAVITY epochs each, no orbit, some proprietary |
| CPD−71 172 | sdO/B + MS wide binary | 1–2 epochs | GRAVITY 2 | G 2 | Deshmukh+2026 sdO/B + MS pairs: 1–2 GRAVITY epochs each, no orbit, some proprietary |
| HD 128220 | sdO/B + MS wide binary | 1–2 epochs | GRAVITY 1 | G 1 | Deshmukh+2026 sdO/B + MS pairs: 1–2 GRAVITY epochs each, no orbit, some proprietary |
| HD 283048 | sdO/B + MS wide binary | 1–2 epochs | GRAVITY 1 | G 1 | Deshmukh+2026 sdO/B + MS pairs: 1–2 GRAVITY epochs each, no orbit, some proprietary |
| BD+10 2357 | sdO/B + MS wide binary | 1–2 epochs | GRAVITY 1 | G 1 | Deshmukh+2026 sdO/B + MS pairs: 1–2 GRAVITY epochs each, no orbit, some proprietary |
| GALEX J2227+2006 | sdO/B + MS wide binary | 1–2 epochs | GRAVITY 2 | G 1 | Deshmukh+2026 sdO/B + MS pairs: 1–2 GRAVITY epochs each, no orbit, some proprietary |
| HD 5143 | Intermediate-mass binary + inner companion | orbit; ORB6 57.8 yr | GRAVITY 17 | G 18 | outer orbit only (58 yr; 310 yr) with a 2-yr arc; 17 and 15 GRAVITY nights, public; keep for a triple-model test later |
| HD 16753 | Intermediate-mass binary + inner companion | orbit; ORB6 310 yr | GRAVITY 15 | G 14 | outer orbit only (58 yr; 310 yr) with a 2-yr arc; 17 and 15 GRAVITY nights, public; keep for a triple-model test later |
| HD 4150 | A star + new companion | 1–2 epochs | PIONIER 1 | raw only | Marion+2014 companions: 1–2 epochs, no orbit |
| HD 16555 | A star + new companion | orbit; ORB6 3.17 yr | PIONIER 1 | raw only | 3.17 yr orbit but one PIONIER night, no Phase 3 |
| HD 29388 | A star + new companion | 1–2 epochs | PIONIER 1 | raw only | Marion+2014 companions: 1–2 epochs, no orbit |
| HD 202730 | A star + new companion | 1–2 epochs | GRAVITY 1, PIONIER 2 | raw only | Marion+2014 companions: 1–2 epochs, no orbit |
| HD 224392 | A star + new companion | 1–2 epochs | AMBER 4, GRAVITY 1, NACO 1, PIONIER 4 | G 2 | Marion+2014 companions: 1–2 epochs, no orbit |
| HIP 14807 | Young star + M companion (SAM) | 1–2 epochs | none | raw only | Evans+2012 NACO SAM companions: no archive data for the published epochs, or one AMBER night |
| HD 16760 | Star + low-mass companion (SAM) | orbit; ORB6 1.28 yr | none | raw only | Evans+2012 NACO SAM companions: no archive data for the published epochs, or one AMBER night |
| HD 113449 | Young binary (SAM) | orbit; ORB6 0.594 yr | AMBER 1 | raw only | Evans+2012 NACO SAM companions: no archive data for the published epochs, or one AMBER night |
| HD 160934 | Young binary (SAM) | orbit; ORB6 10.3 yr | none | raw only | Evans+2012 NACO SAM companions: no archive data for the published epochs, or one AMBER night |
| HD 32642 | Dual-field binary calibrator | orbit; ORB6 760 yr | GRAVITY 3 | G 12 | Nowak+2024 dual-field calibrator: periods of 64–34,000 yr, so an arc of 1–2 yr; outside the single-fibre scope and the dual-field track's business |
| HD 123227 | Dual-field binary calibrator | orbit; ORB6 415 yr | GRAVITY 6 | G 45 | Nowak+2024 dual-field calibrator: periods of 64–34,000 yr, so an arc of 1–2 yr; outside the single-fibre scope and the dual-field track's business |
| HD 1663 | Dual-field binary calibrator | orbit; ORB6 471 yr | GRAVITY 1 | G 9 | Nowak+2024 dual-field calibrator: periods of 64–34,000 yr, so an arc of 1–2 yr; outside the single-fibre scope and the dual-field track's business |
| HD 211299 | Dual-field binary calibrator | unverified; ORB6 983 yr | GRAVITY 1 | G 4 | Nowak+2024 dual-field calibrator: periods of 64–34,000 yr, so an arc of 1–2 yr; outside the single-fibre scope and the dual-field track's business |
| HD 37904 | Dual-field binary calibrator | unverified; ORB6 106 yr | GRAVITY 2 | G 10 | Nowak+2024 dual-field calibrator: periods of 64–34,000 yr, so an arc of 1–2 yr; outside the single-fibre scope and the dual-field track's business |
| HD 46780 | Dual-field binary calibrator | orbit; ORB6 119 yr | GRAVITY 1 | G 4 | Nowak+2024 dual-field calibrator: periods of 64–34,000 yr, so an arc of 1–2 yr; outside the single-fibre scope and the dual-field track's business |
| HD 90444 | Dual-field binary calibrator | orbit; ORB6 140 yr | GRAVITY 2 | G 8 | Nowak+2024 dual-field calibrator: periods of 64–34,000 yr, so an arc of 1–2 yr; outside the single-fibre scope and the dual-field track's business |
| 34 Sex | Dual-field binary calibrator | unverified; ORB6 81.7 yr | GRAVITY 2 | G 8 | Nowak+2024 dual-field calibrator: periods of 64–34,000 yr, so an arc of 1–2 yr; outside the single-fibre scope and the dual-field track's business |
| HD 16638 | Dual-field binary calibrator | unverified; ORB6 289 yr | GRAVITY 1 | G 4 | Nowak+2024 dual-field calibrator: periods of 64–34,000 yr, so an arc of 1–2 yr; outside the single-fibre scope and the dual-field track's business |
| lam01 Scl | Dual-field binary calibrator | orbit; post. 342 yr | GRAVITY 7 | G 30 | Nowak+2024 dual-field calibrator: periods of 64–34,000 yr, so an arc of 1–2 yr; outside the single-fibre scope and the dual-field track's business |
| HD 25535 | Dual-field binary calibrator | orbit; post. 359 yr | GRAVITY 10 | G 55 | Nowak+2024 dual-field calibrator: periods of 64–34,000 yr, so an arc of 1–2 yr; outside the single-fibre scope and the dual-field track's business |
| HD 36584 | Dual-field binary calibrator | orbit; post. 459 yr | GRAVITY 3 | G 25 | Nowak+2024 dual-field calibrator: periods of 64–34,000 yr, so an arc of 1–2 yr; outside the single-fibre scope and the dual-field track's business |
| HD 46716 | Dual-field binary calibrator | orbit; post. 405 yr | GRAVITY 2 | G 17 | Nowak+2024 dual-field calibrator: periods of 64–34,000 yr, so an arc of 1–2 yr; outside the single-fibre scope and the dual-field track's business |
| HD 73900 | Dual-field binary calibrator | orbit; post. 64.3 yr | GRAVITY 5 | G 36 | Nowak+2024 dual-field calibrator: periods of 64–34,000 yr, so an arc of 1–2 yr; outside the single-fibre scope and the dual-field track's business |
| HD 91881 | Dual-field binary calibrator | orbit; post. 156 yr | GRAVITY 6 | G 47 | Nowak+2024 dual-field calibrator: periods of 64–34,000 yr, so an arc of 1–2 yr; outside the single-fibre scope and the dual-field track's business |
| HD 174536 | Dual-field binary calibrator | orbit; post. 3.41e+04 yr | GRAVITY 7 | G 21 | Nowak+2024 dual-field calibrator: periods of 64–34,000 yr, so an arc of 1–2 yr; outside the single-fibre scope and the dual-field track's business |
| HD 196885 | Dual-field binary calibrator | orbit; post. 80.9 yr | GRAVITY 11 | G 40 | Nowak+2024 dual-field calibrator: periods of 64–34,000 yr, so an arc of 1–2 yr; outside the single-fibre scope and the dual-field track's business |
| HD 30003 | Dual-field binary calibrator | orbit; post. 1.1e+03 yr | GRAVITY 17 | G 6 | Nowak+2024 dual-field calibrator: periods of 64–34,000 yr, so an arc of 1–2 yr; outside the single-fibre scope and the dual-field track's business |
| HR 5362 | Dual-field binary calibrator | orbit; post. 7.15e+03 yr | GRAVITY 9 | G 1 | Nowak+2024 dual-field calibrator: periods of 64–34,000 yr, so an arc of 1–2 yr; outside the single-fibre scope and the dual-field track's business |
| kap01 Scl | Dual-field binary calibrator | orbit; post. 598 yr | GRAVITY 1 | G 9 | Nowak+2024 dual-field calibrator: periods of 64–34,000 yr, so an arc of 1–2 yr; outside the single-fibre scope and the dual-field track's business |
| θ1 Ori A | Young massive multiple | partial | AMBER 3, GRAVITY 6, PIONIER 1 | G 1 | Trapezium members: 1–2 epochs or sparse Phase 3, nebular field, no orbit |
| θ1 Ori D | Young massive binary | orbit; 53 d | AMBER 5, GRAVITY 1, PIONIER 1 | G 1 | Trapezium members: 1–2 epochs or sparse Phase 3, nebular field, no orbit |
| θ1 Ori B | Young massive multiple | 1–2 epochs | GRAVITY 2, PIONIER 2 | raw only | Trapezium members: 1–2 epochs or sparse Phase 3, nebular field, no orbit |
| θ2 Ori A | Young massive binary | 1–2 epochs | AMBER 1, GRAVITY 1, PIONIER 5 | G 1 | Trapezium members: 1–2 epochs or sparse Phase 3, nebular field, no orbit |
| θ2 Ori B | Young massive binary | 1–2 epochs | PIONIER 4 | raw only | Trapezium members: 1–2 epochs or sparse Phase 3, nebular field, no orbit |
| θ2 Ori C | Young massive binary | 1–2 epochs | PIONIER 1 | raw only | Trapezium members: 1–2 epochs or sparse Phase 3, nebular field, no orbit |
| HD 59717 (σ Pup) | Giant + faint companion | 1–2 epochs; ORB6 0.706 yr | AMBER 2 | raw only | giant + faint companion, two AMBER nights, no Phase 3 |
| AW Per | Cepheid + companion | orbit | none | raw only | no archive data found |
| HR 10 (HD 256) | A-type binary | orbit; 747.6 d | PIONIER 4 | P 8 | 747.6 d, 4 PIONIER nights (8 Phase 3), no full elements; too sparse for a test |
| BP Cir | Cepheid + companion | 1–2 epochs | GRAVITY 1, PIONIER 10 | G 1; P 25 | Cepheid + possible companion, 1–2 epochs |
| V636 Sco | Cepheid + companion (possible) | 1–2 epochs | GRAVITY 1, PIONIER 9 | P 20 | Cepheid + possible companion, 1–2 epochs |
| UX Tau B | PMS binary | orbit; ~32 yr (B) | AMBER 1, GRAVITY 2, MIDI 2 | G 3 | PMS binary, 2 GRAVITY nights, disk |
| WR 89 | WR + wide companion | 1–2 epochs | GRAVITY 2, PIONIER 2 | G 2 | WR wide-companion detections: 1–2 epochs |
| WR 93 | WR + wide companion | 1–2 epochs | GRAVITY 1, PIONIER 2 | G 1 | WR wide-companion detections: 1–2 epochs |
| WR 115 | WR + wide companion | 1–2 epochs | GRAVITY 1 | G 1 | WR wide-companion detections: 1–2 epochs |
| WR 25 | WR + O hierarchical triple | 1–2 epochs; 208 d (inner, spectroscopic) | PIONIER 1 | P 1 | WR wide-companion detections: 1–2 epochs |
| HD 45166 | qWR + B7V binary | 1–2 epochs; 22.5 yr (spectroscopic) | GRAVITY 1 | G 1 | dropped from the gallery until its calibration is solved (`design/hd45166_diagnosis.md`) |
| HD 44637 | Be + pre-subdwarf binary | partial | GRAVITY 2 | G 2 | Be + pre-sdB, partial orbit, 2 GRAVITY nights |
| AB Dor C | Brown-dwarf binary candidate | 1–2 epochs | AMBER 2, GRAVITY 1, MIDI 1, NACO 3 | G 1 | 1–2 epochs of a candidate |
| HD 72946 B | Star + brown dwarf (dual-field) | orbit; post. 16.1 yr | GRAVITY 8, MATISSE 2 | G 17 | dual-field star + substellar companion: positions-as-data in the dual-field track, not single-fibre visibilities; HD 984 B unpublished |
| HD 136164 Ab | Star + brown dwarf (dual-field) | orbit; post. 75 yr | GRAVITY 4, NACO 2 | G 28 | dual-field star + substellar companion: positions-as-data in the dual-field track, not single-fibre visibilities; HD 984 B unpublished |
| HD 167665 B | Star + T dwarf (dual-field) | orbit | GRAVITY 4 | G 35 | dual-field star + substellar companion: positions-as-data in the dual-field track, not single-fibre visibilities; HD 984 B unpublished |
| HD 984 B | Star + brown dwarf (dual-field) | orbit; post. 36.8 yr | GRAVITY 3, MATISSE 1 | G 20 | dual-field star + substellar companion: positions-as-data in the dual-field track, not single-fibre visibilities; HD 984 B unpublished |
| HD 17155 | Gaia NSS candidate companion | 1–2 epochs | GRAVITY 2 | G 10 | dual-field star + substellar companion: positions-as-data in the dual-field track, not single-fibre visibilities; HD 984 B unpublished |
| HD 128819 | Sco-Cen B/A/F star + low-mass companion (SAM) | 1–2 epochs | NACO 1 | raw only | Sco-Cen low-mass companions: one night each (NACO SAM), no orbit |
| HD 135778 | Sco-Cen star + low-mass companion (SAM) | 1–2 epochs | GRAVITY 1, NACO 4, SPHERE 2 | G 3 | Sco-Cen low-mass companions: one night each (NACO SAM), no orbit |
| HD 133803 | Sco-Cen star + low-mass companion (SAM) | 1–2 epochs | NACO 2, SPHERE 1 | raw only | Sco-Cen low-mass companions: one night each (NACO SAM), no orbit |
| HD 142851 | Sco-Cen star + low-mass companion (SAM) | 1–2 epochs | NACO 1 | raw only | Sco-Cen low-mass companions: one night each (NACO SAM), no orbit |

## Already covered elsewhere

20 systems have a home already (gallery, alias track, this PR's onboarding set, or another worker).

| System | Covered by |
|---|---|
| σ Ori AB | gallery tile (CHARA workshop, one night); the GRAVITY 4 nights are a second route to the 143 d orbit if wanted |
| 9 Sgr | handled by the 9 Sgr onboarding worker |
| AL Dor | alias track (Gallenne 2019, 14.9 d) |
| NN Del | this PR (short arc, PIONIER, 99.3 d) |
| ψ Cen | alias track (Gallenne 2019, 38.8 d) |
| HD 9312 | this PR (Araucaria-I SB2, Gallenne+2023) |
| HD 41255 | alias track, with later epochs held out in this PR |
| HD 70937 | alias track, with later epochs held out in this PR |
| HD 210763 | alias track, with later epochs held out in this PR |
| HD 224974 | this PR (Araucaria-I SB2, Gallenne+2023) |
| ο Leo | alias track, with later epochs held out in this PR |
| V963 Cen | this PR (Araucaria-I SB2, Gallenne+2023) |
| AK For | this PR (Araucaria-I SB2, Gallenne+2023) |
| HD 188088 | alias track, with later epochs held out in this PR |
| LL Aqr | this PR (Araucaria-I SB2, Gallenne+2023) |
| ζ Boo | handled by the ζ Boo / η Oph onboarding worker |
| η Oph | handled by the ζ Boo / η Oph onboarding worker |
| κ Vel | this PR (short arc, GRAVITY, 116.8 d) |
| HR 6819 | gallery tile, Be star (never scored) |
| Gl 229 Ba–Bb | gallery tile (Xuan+2024) |

## Recommended next ten (V)

In this order: each adds something the current set does not test. Elements to be read from the papers' tables before any fit, as for the other tiers.

1. **TZ For:** mixed GRAVITY + PIONIER, 75.7 d, 16 Phase 3 products; Track B staged it but its manifest lists no epochs (a quick fix).
2. **δ Cir:** a triple's outer 1603 d orbit, GRAVITY + PIONIER over 2011–2026, yearly gaps; staged in Track B.
3. **AI Phe:** sub-resolution with a tertiary drifting underneath, 11 of 12 epochs in Phase 3.
4. **KW Hya** (only in Track B): sub-resolution Araucaria PIONIER, 6 of 6 epochs in Phase 3.
5. **V4090 Sgr:** at the resolution limit, SB1 with a Gaia mass prior, 6 of 6 epochs in Phase 3.
6. **HD 150136:** 8.6-year, e 0.68 orbit around a hidden binary: long gaps and third light.
7. **97 Aqr:** a seven-year arc of a 64-year orbit, 16 GRAVITY Phase 3 products: the short-arc test at the other extreme from NN Del.
8. **HD 31297:** the 29.5 d inner pair of an ARMADA triple (needs a three-body scene).
9. **HD 29573:** the 59.2 d inner pair, the same triple scene.
10. **HD 48581:** the 4.84 d inner pair, 14 GRAVITY nights to 2024 (later epochs for hold-out).

Runners-up: HD 168137 and HD 152314 (Track B, only 1–2 published epochs in Phase 3, so they wait on the PIONIER pipeline for raw frames).

## Recommended next five (D)

1. **γ2 Vel (WR 11):** the most famous WR+O pair, with a full GRAVITY visual orbit; 4 Phase 3 nights show aliasing at its worst.
2. **θ1 Ori C:** the 11.4-year O+O orbit; gallery tile from published positions and the one GRAVITY product.
3. **HD 93206 A (QZ Car):** a massive quadruple with GRAVITY and PIONIER Phase 3 products.
4. **U Aql:** a resolved, pulsating Cepheid with a companion on a 5-year orbit; 30 Phase 3 products.
5. **δ Vel Aa–Ab:** eclipsing A-star triple; the NACO SAM night is already on disk, so no download.

Runners-up: HD 98800 (quadruple), HD 206893 B (the best-sampled dual-field system) and GJ 65 AB (83 GRAVITY nights).
