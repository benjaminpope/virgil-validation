# Contest scoring: metrics and published score tables

What each imaging contest paper says about how entries were scored, and the
score tables it published, so that virgil's images can be scored the same
way and placed in each year's table. Compiled 2026-10-06 from the papers and
contest pages listed per section; numbers are transcribed, definitions are
paraphrased.

**Confidence key.** *High*: read in the full paper (or the organisers' own
code). *Medium*: read in the pre-contest rules or the award talk, not the
paper. *Low*: inferred by us; flagged as such.

## Summary

| Year | Metric (one line) | Better | Score table found | Source read |
|---|---|---|---|---|
| 2004 | Reference-weighted RMS of (entry − truth), divided by truth peak, summed over 2 data sets | lower | yes (4 entries) | full paper (OLBIN copy) |
| 2006 | Common-beam convolve, regrid, normalise, 10⁶ × RMS difference | lower | **no** | rules page + award talk (truncated) |
| 2008 | As 2006, normalisation box with ~90% of flux; sum over 6 images | lower | **no** | rules page only |
| 2010 | Align on Gaussian centroid, truth ⊗ 0.4 mas Gaussian, unit sum in box, 10⁶ × RMS | lower | yes | full paper (arXiv) |
| 2012 | Align by cross-correlation (flips allowed), unit sum in box, 10⁶ × RMS, sum of 2 targets | lower | yes (9 entries) | full paper (arXiv) |
| 2014 | Real data, no truth: +1/0/−1 per consensus feature | higher | yes (10 entries) | full paper (Exeter/figshare) + talk |
| 2016 | Γ(x)=sign·\|x\|^0.7, p=2 distance after λ/2B<sub>max</sub> blur; optimal scale and shift per channel; summed over channels | higher | **no** | ImageMetrics docs (organisers' code); abstract only |
| 2018 | Qualitative: two yes/no criteria | — | yes (qualitative) | full paper (ESO library) |
| 2022 | Four metrics (χ², Pearson r, UIQI, component fluxes) → rank points, summed | higher | yes | full paper (KU Leuven Lirias) |
| 2024 | 1 − L1/ΣL1(ref) after λ/(2·130 m) blur, optimal flux scale and shift; mean per band, summed | higher | yes | full paper (arXiv) + organisers' code |

Not found: the 2006 and 2008 score tables (papers are closed-access SPIE; no
preprints on arXiv, NTRS, HAL, OpenAlex or the OLBIN Wayback copies), and the
2016 score table (closed-access; the metric is known from ImageMetrics).

---

## 2004 — NPOI, simulated (Lawson et al., SPIE 5491, 886)

**Source.** Full paper, the OLBIN copy
`http://olbin.jpl.nasa.gov/iau/2004/beautyAll.pdf`, Wayback capture
`20041101014847` (14 pp.). Section 7 ("Rapport sur le concours"), Eq. 2,
Table 2, Figs. 1, 2, 8, 9. Confidence *high*.

### Metric

- Entries were resampled onto the grid of the reference image; orientation
  was added by hand (no submitted FITS carried it) and each entry was
  aligned on a fiducial feature. Pixel scales were those stated by the
  entrants. Done by W. Cotton in AIPS.
- Reference images: data1, Tuthill's LkHα 101 model (242 × 242 pixels of
  0.05 mas); data2, an image made from the analytic model below. **No
  convolution of the truth or the entries is mentioned.**
- The comparison box was drawn on the reference image to contain all of the
  emission.
- Measure (Eq. 2): a reference-weighted RMS,
  σ = [ Σ p_ref (p_i − p_ref)² / Σ p_ref ]^½,
  where p_i is the entry pixel and p_ref the reference pixel, summed over
  the box.
- Combination: each σ is divided by the reference image's peak, and the two
  σ/peak values are summed. Lower is better.
- Flux normalisation of entries before differencing is **not stated**. The
  reference peaks are given as 2.239 × 10⁻⁴ (data1, consistent with a
  unit-sum image on the 242² grid) and "3.0677 × 10⁻¹" (data2; see the
  inconsistency below).

### Score table (Table 2)

| Entry | data1 σ | data1 σ/peak | data2 σ | data2 σ/peak | Σ σ/peak |
|---|---|---|---|---|---|
| BSMEM (Thorsteinsson & Young) | 0.000079 | 0.38 | 0.00035 | 0.116 | **0.50** |
| WISARD (Meimon, Mugnier, Le Besnerais) | 0.00034 | 1.52 | 0.00049 | 0.163 | 1.68 |
| VLBMEM (Monnier & Zhao) | 0.00024 | 1.07 | 0.0024 | 0.798 | 1.87 |
| MIRA (Thiébaut) | 0.0012 | 5.36 | 0.0016 | 0.532 | 5.98 |

Difmap (Monnier & Zhao) was excluded from the comparison.

Inconsistencies in the published table (our arithmetic, *low* confidence as
to which number is the typo):
- The data2 σ/peak values imply a data2 peak of about 3.0 × 10⁻³ (e.g.
  0.0016/0.532), not the printed 3.0677 × 10⁻¹; the exponent is probably a
  typo for 10⁻³.
- BSMEM data1: 0.000079/2.239×10⁻⁴ = 0.35, not 0.38.
- MIRA total: 5.36 + 0.532 = 5.89, not 5.98.

### data2 truth model (Fig. 2, an OYSTER model file)

| Item | Value |
|---|---|
| Component A | limb-darkened disk (OYSTER mode 3), major-axis diameter 5.5 mas, minor/major axis ratio 0.7, major axis at PA 120° (E of N) |
| A's limb darkening | from a model atmosphere with T<sub>eff</sub> = 7000 K, log g = 4.0, evaluated by OYSTER (law not written out; the observation is a single 550 nm channel per the manifest) |
| Spot on A | bright spot at 1.5 × T<sub>eff</sub> of the star, 2.5 mas from A's centre at PA 150°, 1.0 mas diameter |
| Component B | uniform disk 0.5 mas, circular, wavelength-independent diameter (T<sub>eff</sub> 25000 K and log g 5 are marked "ignore") |
| Magnitudes | B is given magnitudes = [0.0]; **A's magnitude is not shown in the published file**, so the A:B flux ratio is not stated |
| Binary | B is 10.0 mas from A at PA 90° (due east) |

Gaps: the flux ratio, the exact limb-darkening law (OYSTER's
atmosphere-based tabulation), how the spot is blended with the disk (it is
specified by temperature, so its surface brightness follows from the Planck
ratio at the observing wavelength), and the reference image's pixel grid are
not given. The OIFITS data constrain the flux ratio and can be used to fix
it.

---

## 2006 — VLTI/AMBER, simulated (Lawson et al., SPIE 6268, 62681U)

**Sources.** Paper closed-access (Semantic Scholar/OpenAlex: no open copy;
NTRS 20070011737 has metadata only). Rules: the archived contest page
`olbin.jpl.nasa.gov/iau/2006/beauty.html` (local copy
`~/data/imaging_contests/2006/pre_submission/beauty2006.html`). Award talk
`olbin.jpl.nasa.gov/iau/2006/beauty2006.pdf` (Wayback; the PDF is truncated
after slide 11, the .ppt copy holds images only). Confidence *medium*.

### Metric (rules page and talk slide "Judging of Entries")

1. Convolve the truth and every entry to a common resolution with a
   Gaussian PSF (the rules say this explicitly; the width is **not
   stated**).
2. Regrid onto a common grid, aligning on an image feature (no astrometry
   in the data).
3. Normalise all images, the model included.
4. Subtract, and take the RMS scaled by 10⁶. The rules page also mentions
   σ/peak as the quality measure, with "closest by eye" as a fallback if no
   entry got the correct image.

Lower is better. Truth: Chesneau's thin-disk model, 301 × 301 pixels of
0.35 mas, grey (no wavelength dependence), 4 AMBER UT triplets.

### Score table

**Not found.** Entrants (from the talk): BSMEM (Baron & Young, winner),
Building Block Method (Kraus, Hofmann, Weigelt), MACIM (Ireland & Monnier),
MIRA (Thiébaut), Recursive Phase Reconstruction (Rengaswamy). The paper's
abstract states five algorithms were evaluated but gives no scores.

---

## 2008 — CHARA, simulated J/H/K (Cotton et al., SPIE 7013, 70131N)

**Sources.** Paper closed-access (no open copy found). Rules page
`olbin.jpl.nasa.gov/iau/2008/Contest08.html` (local copy
`~/data/imaging_contests/2008/pre_submission/Contest08.html`); data readme
`2008-readme`. Confidence *medium* (announced procedure; the paper may
differ in detail).

### Metric (announced)

1. Each **model** image is convolved to a resolution representative of its
   data set and regridded onto a suitable grid.
2. Each **entry** is convolved to the model's effective resolution, using
   the resolution the entrant declared.
3. The entry is aligned on image features and regridded onto the model grid
   (using the entry's stated geometry).
4. Entry pixels are normalised to the model by the pixel sum in a box
   holding about 90% of the model flux.
5. Difference (entry − model) × 10⁶; the quality measure is the RMS about
   zero inside the normalisation box.
6. An entry's score is the sum over its images (2 targets × J, H, K = 6).
   Lowest wins.

Truth: an AGB star and an AGN, each made of components with different
spectra, so the model differs between J, H and K (grey within each band);
the field is tapered by a 15 mas FWHM Gaussian (readme). Component
parameters are not public.

### Score table

**Not found.** The abstract reports only that most entries reproduced the
models well; the winner was MiRA (Thiébaut), per the 2010 and 2024 papers.

---

## 2010 — VLTI/AMBER, simulated, polychromatic (Malbet et al., SPIE 7734, 77342N; arXiv:1007.4473)

**Source.** Full paper, arXiv:1007.4473v1, §2.3 (rules), §4 (metric),
Table 1. Confidence *high*.

### Metric (§4)

1. Align every image on the centroid of a Gaussian fitted to the main star.
2. Interpolate the entries onto the grid of the model images.
3. Convolve the **model** images with a 0.4 mas circular Gaussian (to bring
   them nearer the entries' resolution; the paper does not say whether
   0.4 mas is the FWHM, but that is the usual meaning). Entries are not
   convolved.
4. Normalise every plane of every image, models included, to unit sum
   inside the box with corners [25,25] and [214,214] (1-based pixel
   indices on the model grid).
5. Score per image: 10⁶ × RMS pixel difference (entry − model) over the
   union of that box and a companion box with corners [246,1013] and
   [266,1034].
6. Grey category: the average of the three grey images (Med-H all
   channels; Low-HK channels 1–10; Low-HK channels 11–20), "grey" meaning
   uniform weighting of the channels. Cube category: the average over all
   planes.

Lower is better. The model grid's pixel scale and size are **not stated**.
The two boxes imply a grid at least 1034 pixels on one axis. If the
companion box sits at the companion (84.28 mas from the star), the pixel
scale would be roughly 0.09 mas, but this is our inference (*low*).

Truth: a CO5BOLD/OPTIM3D red-supergiant surface (Chiavassa et al. 2009)
scaled to a star of about 20 mas, plus an unresolved companion 5 mag fainter
at 84.28 mas, PA 261.31°. Not public.

### Score table (Table 1)

| Entry | LR-H | LR-K | MR-H | Combined grey | H cube |
|---|---|---|---|---|---|
| BSMEM (Young, Baron, Buscher) | 6.6 | 5.9 | 10.4 | **7.6** | 13.9 |
| SQUEEZE (Baron, Kloppenborg, Monnier) | 7.9 | 5.9 | 22.7 | 12.2 | not submitted |
| WISARD (Vannier & Mugnier) | 11.5 | 11.4 | 37.9 | 20.3 | 47.0 |
| RPR (Rengaswamy; printed "PPR") | 40.0 | 31.8 | 42.1 | 38.0 | 45.6 |

The "Combined grey" column equals the mean of the first three columns. The
table's "LR-K" is the Low-HK channels 11–20 image and "LR-H" channels 1–10.

---

## 2012 — CHARA/MIRC-6T, simulated (Baron et al., SPIE 8445, 84451E; arXiv:1207.7141)

**Source.** Full paper, arXiv:1207.7141v2, §2.3, §4, Table 1. Confidence
*high*.

### Metric (§4, judged by W. Cotton)

1. Align each entry on the truth by cross-correlation; flips are applied if
   the entrant used another orientation convention (contest convention:
   north up, east left).
2. Interpolate onto the truth grid if needed. Entrants were asked to use
   0.15 mas pixels with a 64 × 64 field; entries already at 0.15 mas were
   not resampled.
3. Normalise all images, truth included, to unit sum in a box. Alp Fak:
   the rectangle with corners [7,7] and [119,119] (over 90% of the
   emission). Bet Fak: the box within 14 pixels of the centre (essentially
   all of the emission).
4. Score per object: 10⁶ × RMS pixel difference inside that box.
5. Total = Alp Fak + Bet Fak. Lowest wins. Where a team sent several
   images for one object, its best one counted.

**No convolution of truth or entries is mentioned.** The OLBIN truth file
names (`Alp_Fak_MIRC6T_LowH_truth_p005_w128.fits`,
`Bet_Fak_MIRC6T_LowH_truth_p015_w64.fits`) suggest an Alp Fak truth of
128 × 128 at 0.05 mas and a Bet Fak truth of 64 × 64 at 0.15 mas. That fits
the [7,7]–[119,119] box, but it conflicts with step 2, which implies a
0.15 mas Alp Fak master grid (*low*). Truth: Alp Fak is a TORUS T Tauri
star + disk (inner radius 40 au, outer 200 au, star 3 R<sub>⊙</sub> offset
about 1.3 au, image rotated by 63.5°). Bet Fak is a MIRC image of AZ Cyg,
observed with the real uv coverage and noise. Neither is archived.

### Score table (Table 1)

| Team | Software | Alp Fak | Bet Fak | Total |
|---|---|---|---|---|
| Monnier | MACIM | 37.8 | 242.5 | **280** |
| Hofmann, Schertl & Weigelt | IRS | 28.2 | 285.1 | 313 |
| Millour & Vannier | MiRA | 27.7 | 299.8 | 327 |
| Thiébaut & Soulez | MiRA | 24.8 | 337.6 | 362 |
| Mary & Vannier | MIROIRS | 29.4 | 381.9 | 411 |
| Young | BSMEM | 40.2 | 659.9 | 700 |
| Millour & Vannier | BSMEM | 32.3 | 871.7 | 903 |
| Elias | CASA | 41.2 | 1285.9 | 1327 |
| Rengaswamy | SR | 235.6 | 1636.6 | 1872 |

---

## 2014 — VLTI/PIONIER, real data (Monnier et al., SPIE 9146, 91461Q)

**Sources.** Full paper, author copy in the Exeter repository (figshare
article 29720141, file `The 2014 Interferometric Imaging Beauty
Contest.pdf`), §4.2–4.5 and Fig. 9. SPIE talk slides
`gears.iaa.es/sites/default/files/2014SPIE_MONNIER_BEAUTY.pdf`, slide 27.
Confidence *high*.

### Metric

There is no truth (real VY CMa and R Car data). The organisers built
consensus **median images** per star and per channel:
- regrid all 10 entries to 0.1 mas pixels;
- align them by cross-correlation (after a 1.5 mas FWHM Gaussian blur, used
  only for the alignment);
- rescale to a common stellar surface brightness;
- take the median over entries.

Each entry then got +1 (reproduced unambiguously), 0 (partly) or −1
(missing) on four consensus features: for VY CMa, the two spots and their
placement; for R Car, the water/CO shells and the two spots. Score = sum.
Higher is better. The authors call it subjective.

### Score table (paper Fig. 9 = talk slide 27)

| Submitter (code) | VY CMa: 2 spots | VY CMa: placement | R Car: shells | R Car: 2 spots | Score |
|---|---|---|---|---|---|
| Hummel (PEARL/CLEAN) | 0 | −1 | −1 | −1 | −2 |
| Hofmann (IRBis) | +1 | 0 | 0 | +1 | +2 |
| Young (BSMEM) | +1 | +1 | +1 | 0 | +3 |
| Sanchez (BSMEM) | +1 | +1 | +1 | +1 | **+4** |
| Kohler (MIRA) | 0 | +1 | +1 | −1 | +1 |
| Soulez (MIRA3D) | +1 | +1 | +1 | 0 | +3 |
| Kluska (MIRA-SPARCO) | −1 | +1 | −1 | 0 | −1 |
| Duvert (WISARD) | +1 | 0 | 0 | −1 | 0 |
| Kraus (SQUEEZE/MACIM) | +1 | 0 | −1 | +1 | +1 |
| Kloppenborg (SQUEEZE-poly) | +1 | 0 | 0 | 0 | +1 |

The faces were read from the rendered slide. Hummel's faces sum to −3, but
the published score is −2. For virgil, 2014 is a qualitative check against
the median images only.

---

## 2016 — VLTI GRAVITY + MATISSE, simulated, chromatic (Sanchez-Bermudez et al., SPIE 9907, 99071D)

**Sources.** Paper closed-access (OpenAlex/Semantic Scholar: closed; an
edoc.mpg.de record has no file). Abstract via OpenAlex. Metric from the
organisers' own documentation in
[JMMC-OpenDev/ImageMetrics](https://github.com/JMMC-OpenDev/ImageMetrics)
`docs/src/bc2016.md` and `docs/src/general.md`, written by É. Thiébaut, a
2016 co-organiser. Confidence: metric *medium–high* (organisers' code, not
the paper); table *not found*.

### Metric (ImageMetrics `bc2016.md`)

Per spectral channel λ:
- **Reference** y_λ = truth z_λ convolved with a PSF of FWHM
  ω_λ = λ/(2 B<sub>max</sub>), on the truth's own pixel grid. The page says
  that pixel size is 3 mas per pixel; that probably refers to one of the
  2016 truths, and is *low* confidence.
- **Entry** x_λ is resampled to the truth pixel size (magnification = ratio
  of pixel sizes), blurred to the same resolution, and shifted by t_λ.
  Pixels outside either field are taken as 0. The E–W axis may be
  inverted.
- Brightness correction Γ(v) = sign(v)·|v|^γ with **γ = 0.7**, distance
  exponent **p = 2**, no bias (β = 0), out-of-field value η = 0.
- With p = 2 the optimal flux scale is closed-form, and the channel score is
  S_λ = max over t_λ of
  (Σ x̃ỹ)² / (Σ x̃² · Σ ỹ²),
  where x̃ = Γ(resampled entry), ỹ = Γ(reference), summed over the union of
  the two fields. This equals 1 − D/Σỹ², a squared normalised
  cross-correlation, with values in [0, 1].
- Total score = **sum over channels**. Higher is better.
- The shift is optimised separately in **each** channel (unlike 2024).

### Score table

**Not found.** Winner: K.-H. Hofmann and the MPIfR group (abstract). Other
entrants per the reference list: Thiébaut, Millour, Schutz, Ferrari,
Vannier, Mary, Young. The 2024 paper lists BSMEM, IRBis (winner), PAINTER,
self-cal, CHIPS and LITpro for 2016. ImageMetrics says it can score the 2016
datasets, but no 2016 truth or entries are in the repository.

---

## 2018 — CHARA/MIRC + VLTI/PIONIER, simulated (Mérand et al., SPIE 10701, 107011U)

**Source.** Full paper, ESO library copy
`https://www.hq.eso.org/sci/libraries/SPIE2018/10701-103.pdf` (also
HAL hal-01978553), §2, §4, Fig. 1. Confidence *high* for what is stated.

### Metric

**No numerical image metric.** All five entries recovered the disk
orientation and planet position. Two yes/no criteria then decided the
contest:
1. Was the gap between the inner and outer disks recovered at the right
   place? Yes: Kluska, Thiébaut, Sanchez-Bermudez et al.
2. Was the whole inner disk reconstructed? Yes: Sanchez-Bermudez et al.,
   Young, Tallon et al.

Only Sanchez-Bermudez, Alberdi & Schödel (SQUEEZE then BSMEM) met both, and
they won.

### Score table (qualitative)

| Entry | Code | Gap | Full inner disk | Reported planet / disk |
|---|---|---|---|---|
| Kluska | MiRA + SPARCO | yes | no | planet Δx 1.8, Δy 3.3 mas (±0.3); disk PA 45.5 ± 2° |
| Thiébaut | MiRA | yes | no | ρ 3.71 mas, PA 28.1°, 2.0% flux; star 65.6%; disk PA ≈ 45° |
| Sanchez-Bermudez, Alberdi, Schödel | SQUEEZE → BSMEM | yes | yes | ρ ≈ 3.8 mas, PA ≈ 28°, ≈ 2.2%; star 65.8%, rim 29.4%; disk PA ≈ 47° |
| Young | BSMEM | no | yes | ρ 3.76 mas, PA 28.6°; disk PA 43° |
| Tallon, Tallon-Bosc, Thiébaut, Soulez | LITpro (model fit) | no | yes | ρ 3.73 mas, PA 27.3°, 1.9%; star 65.3%; disk PA 42.4°; Gaussian halo 4.2% |

### Truth model (§2 text and Fig. 1 panels)

The visibilities were computed from analytic formulas (no image), grey,
R ≈ 50. Noise: Gaussian per V² and CP, plus a random offset per baseline or
triangle per pointing.

| Component | Published parameters |
|---|---|
| Star | uniform disk 0.4 mas diameter, 65% of total flux |
| Inner disk with bright rim | 29% in total. Fig. 1 splits it into an "inner ring" of 13% and a "spot" (the bright rim arc) of 16%. "Projection angle 44.2°" (probably the major-axis PA; could be read as the inclination, *low*) |
| Planet | uniform disk 0.3 mas, 3.75 mas from the star at PA 27°, 2% of flux |
| Outer disk | extends to 20 mas; Fig. 1 labels it "outer ring", 3% |

Fig. 1 confirms the geometry: disk major axis running NE–SW (north up,
east left), bright rim arc on the NW side of the inner ring, and a gap
between the inner and outer rings. Fig. 1's labels add up to 99%. **Not
stated:** ring radii and widths, inclination (beyond "44.2°"), the rim's
azimuthal brightness law, and the outer disk's inner radius and profile.
Fig. 1 has a 5 mas scale bar from which radii could be measured roughly. The
model was matched to Lazareff et al. 2017 (Herbig AeBe PIONIER survey), so
it was probably built from that paper's ring-model family (*low*). The data
can be fitted to recover the parameters.

---

## 2022 — VLTI/GRAVITY + JWST/NIRISS AMI, simulated (Sanchez-Bermudez et al., SPIE 12183, 121831G)

**Source.** Full paper, KU Leuven Lirias open copy
(`lirias.kuleuven.be/retrieve/29a86feb-dde3-4a4d-aee7-73e160d4badc`), §2,
§4, Tables 1–5. Confidence *high*.

### Metric (§4): four metrics, each turned into rank points (max 6), summed

- **Metric 1, data χ².** χ² = (1/N) Σ (M − D)²/σ², computed from the
  submitted images with no transformation, separately for V² and CP; the
  object score uses their mean. Multi-wavelength entries are evaluated per
  channel. Best gets 6 points, then decreasing "linearly".
- **Common preprocessing for metrics 2–4.**
  1. Rescale entries to the truth pixel scale: 0.1 mas (object 1) or
     10 mas (object 2), with `scipy.ndimage.zoom`, order 0
     (nearest-neighbour).
  2. Average image cubes over wavelength.
  3. Renormalise so the pixel sum is preserved.
  4. Crop to a common field: 30 mas (object 1) or 720 mas (object 2).

  No shift or flux-scale optimisation and no beam convolution are
  mentioned.
- **Metric 2.** Pearson correlation coefficient (`numpy.corrcoef`) between
  truth and entry.
- **Metric 3.** Universal Image Quality Index Q (Wang & Bovik; the
  `sewar` implementation), the product of correlation, mean-luminance and
  contrast terms.
- **Metric 4.** Flux fractions of the components (ring, central source,
  companion; or arc and star), measured in hand-drawn regions and scored
  per component, then averaged. Entries without the morphology get 1 point.

Higher total is better.

### Score tables

Table 1 (metric 1, χ²):

| Entry | O1 χ² V² | O1 χ² CP | O1 pts | O2 χ² V² | O2 χ² CP | O2 pts |
|---|---|---|---|---|---|---|
| Young, BSMEM | 26 | 19 | 6 | 1.6 | 0.01 | 6 |
| Kluska, SPARCO/MiRA | 291 | 302 | 5 | 8e3 | 3.6e5 | 4 |
| Kluska, SPARCO/ORGANIC | 812 | 317 | 4 | 8.6e4 | 1e6 | 3 |
| Millour/Drevon, MiRA+S | 341 | 128 | 5 | 2.1e4 | 5.8e3 | 4 |
| Millour/Drevon, RHAPSODY | 7.2e3 | 1.3e4 | 1 | – | – | 1 |
| Vermot, NNs | 933 | 2.3e3 | 2 | 890 | 4e4 | 5 |

Table 2 (metric 2, Pearson r) and Table 3 (metric 3, UIQI; captioned
"second metric" in the paper):

| Entry | O1 r | pts | O2 r | pts | O1 Q | pts | O2 Q | pts |
|---|---|---|---|---|---|---|---|---|
| Young, BSMEM | 0.14 | 6 | 0.44 | 5 | 0.13 | 5 | 0.13 | 4 |
| Kluska, SPARCO/MiRA | 0.09 | 3 | 0.5 | 6 | 0.53 | 6 | 0.19 | 5 |
| Kluska, SPARCO/ORGANIC | 0.12 | 5 | 0.25 | 2 | 0.12 | 4 | 0.11 | 3 |
| Millour/Drevon, MiRA+S | 0.1 | 4 | 0.31 | 3 | 0.09 | 3 | 0.06 | 2 |
| Millour/Drevon, RHAPSODY | 0.05 | 1 | – | 1 | 8.1e-5 | 1 | – | 1 |
| Vermot, NNs | 0.06 | 2 | 0.33 | 4 | 0.08 | 2 | 0.19 | 6 |

Table 4 (metric 4, recovered flux fractions):

| Entry | O1 central | O1 companion | O1 ring | pts | O2 arc | O2 central | pts |
|---|---|---|---|---|---|---|---|
| Young, BSMEM | 0.18 | 0.15 | 0.61 | 5 | 0.14 | 0.88 | 6 |
| Kluska, SPARCO/MiRA | 0.18 | 0.13 | 0.67 | 6 | 0.19 | 0.8 | 2 |
| Kluska, SPARCO/ORGANIC | 0.17 | 0.13 | 0.61 | 5 | 0.1 | 0.86 | 5 |
| Millour/Drevon, MiRA+S | 0.16 | 0.13 | 0.44 | 4 | 0.11 | 1.0 | 4 |
| Vermot, NNs | – | – | – | 1 | 0.145 | 0.845 | 6 |

Table 5 (totals): Young **43**, Kluska SPARCO/MiRA 37, Kluska ORGANIC 31,
Millour/Drevon MiRA+S 29, Vermot 28, RHAPSODY 8. The row labels in Table 5
are garbled ("Metric 1 (Obj.3)" etc.). Matching the points against Tables
1–4 shows the eight rows are, in order: M1-O1, M1-O2, M2-O1, M2-O2, M3-O1,
M3-O2, M4-O1, M4-O2.

### Truth (not public)

- **Object 1 (GRAVITY K).** Azimuthally modulated ring after the model of
  the paper's ref. 4: major axis PA 50°, inclination 50°, upper side
  slightly brighter. Central source ~18%, ring ~69%, companion 13% at
  ΔRA = +2, ΔDec = +10 mas. 501 × 501 pixels of 0.1 mas.
- **Object 2 (JWST AMI, 3.8 µm, monochromatic).** Stellar bow shock: star
  85%, arc 15%. 10 mas pixels.

---

## 2024 — VLTI PIONIER/GRAVITY/MATISSE L+N, simulated, chromatic (Millour et al., SPIE 13095; arXiv:2609.35064)

**Sources.** Full paper, arXiv:2609.35064v1, §3, §5 (Eqs. 5–6), Table 3.
Pre-contest guidelines (local `2024/pre_submission/guidelines_2024.txt`).
The organisers' scoring code: ImageMetrics `python/image_utils.py` and
`python/figure5_images_obj1_compare.py` (F. Millour's script behind the
paper's histograms), plus `docs/src/ic2024.md` (Thiébaut's Julia version).
Confidence *high* for the paper; the code details are *high* as to what the
script does.

### Metric

Paper (§5):
- L(x, y) = Σ|x_k − y_k| / Σ|y_k|. Here y is the truth blurred to a
  prescribed resolution.
- Channel score = 1 − min over (α, δ) of L(α·T_δ x_λ, y_λ). Overall
  S = (1/N<sub>λ</sub>) Σ<sub>λ</sub> of the channel scores (Eq. 6).
  Higher is better.
- The entry is resampled bilinearly onto the model's (finer) grid, and
  linearly in wavelength. Out-of-field pixels are 0. α and δ are found
  numerically.
- §6 says the model and entry images were shown convolved to λ/(4 B<sub>max</sub>).

The organisers' Python script (what produced Table 3, as far as we can
tell) does the following:
1. Normalise each model channel to unit sum.
2. Blur both model and entry with a Gaussian of **FWHM = λ/(2 B<sub>max</sub>)**,
   with **B<sub>max</sub> = 130 m** (`calc_beamsize(..., fac=2)`; the
   Gaussian σ is FWHM/2.355). This does not match the guidelines' and
   §6's λ/4B<sub>max</sub>.
3. Bilinearly resample both onto the larger of the two grids, then
   interpolate the model linearly to the entry's wavelengths.
4. Per channel: centre by the cross-correlation peak, then three passes of
   coordinate descent over (a) the flux scale α, (b) an integer-pixel x
   roll, (c) an integer-pixel y roll, minimising L1.
5. Channel score = 1 − L1/Σ(model).
6. Band score = mean over channels (histograms in Figs. 10–11).

Translation is therefore optimised **per channel** (as in Eq. 6). The
guidelines and the Julia `ic2024.md` instead use one global shift for all
channels, and the guidelines also mention a free beam width.

The script also rotates the truth cubes before comparison: the spiral by
−73°, with its pixel scale divided by 10, and the disk by +23°. So the raw
truth FITS headers do not match the data's orientation and scale.

Totals (Table 3) = sum of the band scores over both objects, **excluding
spiral K** because neither entry resembled the input.

### Score table (Table 3; captioned "as in eq. 5" but the values are scores, 1 − L)

| Participant | Spiral H | Spiral K | Spiral L | Spiral N | Disk H | Disk K | Disk L | Disk N |
|---|---|---|---|---|---|---|---|---|
| J. Drevon (MiRA) | 0.14 | 0.13 | 0.27 | 0.78 | 0.65 | 0.52 | 0.31 | 0.38 |
| R. Norris (SQUEEZE + OITOOLS.jl) | 0.15 | 0.09 | 0.65 | 0.84 | 0.27 | 0.52 | 0.22 | 0.24 |

| Participant | Spiral (H+L+N) | Disk (H+K+L+N) | Total |
|---|---|---|---|
| J. Drevon | 1.19 | 1.86 | **3.05** |
| R. Norris | 1.64 | 1.25 | 2.89 |

The arithmetic checks out. Drevon won with the higher total, which confirms
that higher is better.

### Truth (not public; ImageMetrics demos read it from the organisers' disks)

- **Obj1.** AMHRA pinwheel: first turn about 10 mas, 3 turns, opening angle
  40°, inclination 24°, rotation 75°, hollow 4 mas, binary separation 1 mas,
  dust 1500 K, WR 70 000 K + OB 40 000 K (paper Table 1).
- **Obj2.** Ray-traced disk at 184 pc, i = 70°: R<sub>in</sub> 1 au,
  R<sub>out</sub> 150 au, gaps at 2–3, 7–11 and 30–40 au. Point sources
  with ExoREM spectra at (−6.5, −7.5), (20.5, −0.5) and (101.5, 171.5) mas
  (paper Table 2).

---

## Implications for scoring virgil

- **Quantitative, re-implementable with published tables:** 2004 (data2
  only, if the A:B flux ratio is fixed from the data), 2010 and 2012
  (truths not public), 2022 (truth not public), 2024 (truth not public). Of
  these, only 2004 data2 has a truth we can rebuild. 2018 has an analytic
  truth but only a qualitative published ranking.
- **Without truths,** virgil can be placed in the 2010/2012/2022/2024
  tables only once the organisers supply truth images (see the manifest's
  `missing` notes).
- For 2024, the ImageMetrics Python path (λ/2B<sub>max</sub>,
  B<sub>max</sub> = 130 m, per-channel integer shifts, unit-sum model,
  rotated truths) is the one that reproduces the published numbers, not
  the λ/4B<sub>max</sub> stated in the guidelines.

## Status of the comparison with the winners

- **2004 data2:** the only year scored numerically against the published
  table. The truth model is rebuilt from the paper, with the A:B flux
  ratio fixed by a fit to the data. The paper is read only for scoring,
  never for imaging.
- **2018:** scored against the published qualitative criteria: whether
  the image recovers the gap between the inner and outer disks, and the
  whole inner disk.
- **2024: pending** (Ben, 2026-10-06). The truth images have not been
  requested yet. Until they arrive, 2024 is assessed on the data alone
  (χ²/N, residuals, agreement within the ensemble) and kept out of the
  comparison with the winners.
- **2026:** the truth cubes are in hand (John Young, 2026-10-07); the data are not yet.
- **2006, 2008, 2010, 2012, 2016, 2022:** need either the organisers'
  truth images or the closed-access score tables. Each year's `missing`
  note in the manifest records which.
