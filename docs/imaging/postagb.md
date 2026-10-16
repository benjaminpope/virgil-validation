# Post-AGB discs (VLTI/PIONIER)

The imaging-contest data cannot all be shown in public, so this page tests
virgil's image reconstruction on data that anyone can fetch and against
images that experts have already published: the VLTI/PIONIER observations of
post-AGB circumbinary discs in ESO programme 1103.D-0797, reconstructed
independently by De Prins et al. (2026) with SPARCO + ORGANIC. We
reconstruct each target with a SPARCO-like model (below), put both images on
one grid and one beam, and say plainly whether they agree. One tab per
target; the others are added as they are done.

| Target | Published method | virgil verdict | Image cross-correlation |
| --- | --- | --- | --- |
| [IW Car](#iw-car) | SPARCO + ORGANIC (De Prins+2026; SQUEEZE check) | agree on morphology, differ in outer-arc detail | 0.95 |
| HR 4049, HD 108015 | SPARCO + ORGANIC | pending | pending |
| AI Sco, EN TrA, HD 95767, IRAS 15469-5311, PS Gem | SPARCO + ORGANIC | pending | pending |
| [IRAS 08544-4431](#iras-08544-4431) | SPARCO image (Hillen+2016) | agree on the ring, the asymmetry and the central point source | 0.96 |

=== "IW Car"

    ### IW Car { #iw-car }

    <figure markdown>
    ![IW Car: ORGANIC image (left) and virgil image (right) at the PIONIER beam](postagb/IW_Car_images.png)
    <figcaption>IW Car in H band, 23 PIONIER exposures (MJD 58812 to 58893, up to 2020-02-14). Left: the
    ORGANIC image of De Prins et al. (2026), re-rendered here from their Zenodo FITS
    (10.5281/zenodo.20155600). Right: virgil, MaxEntropy with the weight at the
    L-curve corner. Both are the star-and-secondary-subtracted disc emission, convolved to the PIONIER
    beam (grey ellipse), peak-normalised, North up, East left; the star is the gold marker.</figcaption>
    </figure>

    !!! success "Verdict: the two reconstructions agree"
        At the PIONIER beam the images correlate at 0.95 (0.95 after the best sub-beam shift), against 0.14 for the
        East-West mirror of the published image, 0.04 for its 180° rotation and 0.60 for its azimuthal average
        (the same flux at the same radii with no arcs): the same inner arcs
        near the star and the same large western arc. They differ in detail: the outer arc
        sits further out and is oriented differently in virgil (flux-weighted radius 11.0 against 9.9 mas,
        major-axis PA 138° against 163°). De Prins et al.
        themselves note that the inner arcs' azimuthal profile is not well constrained (Appendix D, SQUEEZE).

    **Parameters.** De Prins et al. could not fit a PMOIRED geometric model to IW Car (a single disc does not
    reproduce it), so there is no PMOIRED row to compare with. The comparison is with what the paper does
    report for IW Car: the SPARCO parameters of the ORGANIC reconstruction (FITS headers), and the inner-rim
    ellipses fitted to the image (Table 2; inner arcs and outer arc separately). The "virgil" columns are two
    fits to the same 23 files: the SPARCO parameters fitted together with the image, and a
    PMOIRED-style parametric model (UD star + point secondary + power-law rim with an m = 1 modulation +
    resolved background). virgil's fits are point estimates; the σ column uses the paper's uncertainty, where it gives one.
    The image fit starts at the paper's SPARCO values (UD 0.5 mas, environment index 0.9, secondary at the
    paper's position under a 1 mas prior) and is a local optimisation, so agreement in this table
    shows that the data do not pull the values away from the published ones, not that virgil recovers them blind.
    The secondary's spectral index is 0 in both models (the paper's SIND2 = 0).

    | Parameter | De Prins+2026 | virgil (image fit) | virgil (parametric) | Difference |
    | --- | --- | --- | --- | --- |
    | Primary flux fraction, H (%) | 63.6 | 63.4 | 63.5 | −0.2 / −0.1 points |
    | Secondary flux fraction, H (%) | 1.99 | 2.44 | 1.86 | +0.45 / −0.13 points |
    | Secondary offset (Δα, Δδ) (mas) | (1.12, −1.90) | (1.16, −1.72) | (1.14, −1.83) | ≤ 0.2 mas |
    | Primary UD diameter (mas) | 0.50 | 0.53 | 0.45 | +0.03 / −0.05 mas |
    | Disc spectral index (image) | 0.89 | 0.885 | not comparable (rim 3.2) | −0.005 |
    | Inner-arc rim diameter (mas) | 6.9 ± 0.5 | see metrics | 16.1 (outer-arc scale) | not the same structure |
    | Rim inclination (°) | 48 ± 5 (inner), 57 ± 2 (outer) | n/a | 55.5 | −0.7σ against the outer arc |
    | Rim position angle (°) | 144 (+8, −7) (inner), 176 ± 3 (outer) | n/a | 165.8 | −3.4σ against the outer arc, +2.7σ against the inner |
    | Rim azimuthal modulation (m = 1) | not reported | n/a | amplitude 0.85 at 313° | none to compare |

    The parametric rim lands between the two orientations the paper finds for the inner and outer arcs
    (and near the SPHERE outer disc, i = 41.7°, PA = 161°, Andrych et al. 2023), which says the single-ring
    model is describing a blend of structures, as the paper concluded. Its χ² per point is 3.3, against
    0.46 for the image fit, for the same reason.

    **Image metrics** (both images on one 0.35 mas grid, convolved to the beam; inner = within 5 mas of the star,
    outer = beyond; the paper's own rim fit needs per-pixel significances, so both images get the same moments.
    As an external check, the moments of the ORGANIC image give an outer major axis at PA 163° against the paper's
    176 ± 3° from its rim fit, within 13°, so the two definitions are consistent):

    | Metric | ORGANIC | virgil |
    | --- | --- | --- |
    | Normalised cross-correlation (controls: E-W mirror 0.14, 180° rotation 0.04, azimuthal average 0.60) | 1 | 0.95 |
    | Inner flux fraction / outer flux fraction | 16.7 % / 77.5 % | 14.9 % / 75.9 % |
    | Inner flux-weighted radius (mas) | 3.38 | 3.35 |
    | Outer flux-weighted radius (mas) | 9.93 | 11.05 |
    | Inner major-axis PA (°; nearly round, axis ratio 0.95 / 0.88) | 119 | 122 |
    | Outer major-axis PA (°; paper's outer-arc rim fit 176 ± 3) | 163 | 138 |
    | Asymmetry angle, inner flux centroid (° East of North), offset (mas) | 321, 1.08 | 340, 1.11 |
    | Asymmetry angle, outer flux centroid (°), offset (mas) | 268, 4.04 | 260, 2.94 |

    ??? note "Data and method"

        **Data.** Phase 3 calibrated OIFITS from the ESO archive, programme 1103.D-0797 (PI J. Kluska,
        INSPIRING), the 23 exposures listed in the paper's observation log: 828 V² and 552
        closure-phase rows across the H band (1.51–1.76 µm). `postagb_data.py check` confirms
        that the union of ESO's two Phase 3 releases (PIONIER and INSPIRING; 31 exposures) matches the earlier reduced copy, and the 23 used here are the INSPIRING release. The PIONIER beam is 3.26 × 2.29 mas at PA 142°.

        **Model.** SPARCO: a uniform-disc primary (power-law index −3.165, as in the paper) plus a point
        secondary plus a `virgil.models.Image` environment (128 × 128 pixels of 0.35 mas, the paper's grid),
        each with a power-law spectrum from `virgil.spectra`. The secondary starts at the paper's position
        (so it is not independent of ORGANIC) and the fit then moves it. Regulariser `MaxEntropy`; weight from
        `l_curve` over nine weights (10⁴ to 10⁻²), taking the corner (weight 56); start from
        `starting_image`. ORGANIC differs: a GAN generator as its regulariser, with the primary's parameters tuned with pyBADS. Neither is a ground truth.

        **Fit quality.** On the same data the virgil image has χ² per point 0.46 and the ORGANIC image, with its
        published SPARCO parameters unchanged, 2.06 (V² residual rms 0.88 against 1.53σ; T3 0.68 against 2.09σ).
        This does not say virgil's image is better: it has about 16 000 free pixels for 1 794 data points and
        is over-fitted at this weight, while ORGANIC is a more strongly regularised image. The test that matters here is the image agreement above.
        The 1 794 is the length of virgil's whitened-residual vector, not the number of data: the 828 V² and 552
        closure phases (1 380 samples) become 1 794 residuals because each closure phase of four or more
        telescopes also adds a periodic penalty term. Dividing by 1 380 instead would give about 0.59 and 2.67.
        Flipping the published image East-West or North-South worsens its χ² to 60 and 8.9, so the
        orientation is right.

        ![uv coverage, V² and closure phases](postagb/IW_Car_inspect.png)

        ![V² and T3 fits and residuals for virgil and the ORGANIC image](postagb/IW_Car_fit.png)

        ![L-curve](postagb/IW_Car_lcurve.png)

        ![All four comparison panels](postagb/IW_Car_compare.png)

        <figure markdown>
        ![SQUEEZE against ORGANIC for IRAS 15469-5311 and IW Car](postagb/literature/IW_Car_squeeze.png)
        <figcaption>ORGANIC (left) against SQUEEZE (right) for IRAS 15469-5311 and IW Car (inner and outer
        arcs). Reproduced from arXiv:2605.13445v2, Fig. 15 (Appendix D), the arXiv version, with the authors'
        permission. The authors find both methods recover the inner arcs and the large outer arc.</figcaption>
        </figure>

        <figure markdown>
        ![The published ORGANIC image of IW Car as it appears in the paper](postagb/literature/IW_Car_disc_organic.png)
        <figcaption>As published: the ORGANIC image of IW Car with the fitted rims of the inner and outer arcs,
        arXiv:2605.13445v2, Fig. 4. Reproduced from the arXiv version with the authors' permission.</figcaption>
        </figure>

        <figure markdown>
        ![ORGANIC image of IW Car without a SPARCO secondary](postagb/literature/IW_Car_disc_organic_no_sparco_secondary.png)
        <figcaption>The same reconstruction with no secondary in the SPARCO model: the unresolved flux peak
        near the green cross is the secondary. arXiv:2605.13445v2, Fig. 2. Reproduced from the arXiv version
        with the authors' permission.</figcaption>
        </figure>

        <figure markdown>
        ![SPHERE/IRDIS against PIONIER for IW Car](postagb/literature/IW_Car_sphere_vs_pionier.png)
        <figcaption>SPHERE/IRDIS polarised light (Andrych et al. 2023) against the PIONIER image,
        arXiv:2605.13445v2, Fig. 6. Reproduced from the arXiv version with the authors' permission.</figcaption>
        </figure>

        Provenance, permissions and credits: [`DATA_ATTRIBUTION.md`](https://github.com/benjaminpope/virgil-validation/blob/main/DATA_ATTRIBUTION.md).
        Reproduce with `scripts/postagb_data.py fetch`, `scripts/postagb_literature.py` and
        `scripts/postagb_imaging.py {inspect,parametric,image,compare,replot} docs/imaging/postagb`.

=== "IRAS 08544-4431"

    ### IRAS 08544-4431 { #iras-08544-4431 }

    <figure markdown>
    ![IRAS 08544-4431: Hillen et al. image (left) and virgil image (right)](postagb/IRAS08544_images.png)
    <figcaption>IRAS 08544-4431 in H band, 27 PIONIER observing blocks (2015-01-21, 2015-01-24 and 2015-02-23).
    Left: the single-star-subtracted SPARCO image of Hillen et al. (2016) as published
    (their Fig. 4, upper panel; blue contours are their 5, 3 and 1σ levels). Right: virgil, MaxEntropy with the
    weight at the L-curve corner, primary subtracted, drawn in the published figure's own colour map, on the same
    field of view, orientation (North up, East left), linear 0 to 1 peak-normalised stretch, and convolved with the beam
    ellipse drawn on the published figure (2.5 × 1.8 mas, PA 168°). The paper's image exists only as a bitmap, so its
    stretch and its beam cannot be matched exactly. The star is the marker at the origin.</figcaption>
    </figure>

    !!! success "Verdict: the two reconstructions agree"
        At the published beam the images correlate at 0.96 (mirroring either axis drops it to 0.71 and 0.66): the same
        ~15 mas ring, brighter in the north-east, with the same dark gaps to the north-west and south-east, and the same
        point-like source just south-west of the star. The differences are small: virgil's ring is 0.3 mas larger
        (flux-weighted radius 8.5 against 8.2 mas) and its central source is further from the star (1.6 against 1.1 mas, at
        the same position angle, 205° against 200°). The secondary's separation, angle and flux fraction in the
        parametric fit differ from the paper's, as set out below.

    **Parameters.** Hillen et al. fit a 16-parameter parametric model (Table 1, best model) by MCMC: uniform-disc
    primary (fixed at 0.5 mas), point-source secondary, inclined Gaussian ring with m = 1 and m = 2 azimuthal
    modulations, and a resolved background, each with a black-body or Kurucz spectrum. The "virgil" column is the same
    geometry refitted to the same 27 files with virgil components, the primary fixed at 0.5 mas as in the paper and
    power-law spectra (so the flux fractions are at 1.65 µm and are not exactly the paper's band-averaged ones).
    The fit is blind: four generic starting points (none uses the paper's values) and the best χ² is kept.
    They end in two minima, χ² per point 1.617 (two starts) and 1.535 (two starts, the one shown); the table's
    ring rows are the better one. virgil's fit is a point estimate; the σ column uses the paper's uncertainty.

    | Parameter | Hillen+2016 | virgil | Difference |
    | --- | --- | --- | --- |
    | Primary flux fraction (%) | 59.7 ± 0.6 | 56.6 | −5.2σ |
    | Secondary flux fraction (%) | 3.9 ± 0.7 | 7.0 | +4.4σ |
    | Ring flux fraction (%) | 20.9 ± 0.9 | 20.8 | −0.1σ |
    | Background flux fraction (%) | 15.5 ± 0.5 | 15.6 | +0.2σ |
    | Ring diameter (mas) | 14.15 ± 0.10 | 14.26 | +1.1σ |
    | Ring Gaussian width, FWHM (mas) | 3.2 ± 0.1 | 3.17 | −0.3σ |
    | Ring inclination (°) | 19 ± 2 | 18.5 | −0.2σ |
    | Ring position angle (°) | 6 ± 6 | 0.4 | −0.9σ |
    | Ring m = 1 amplitude | 0.43 ± 0.06 | 0.42 | −0.1σ |
    | Ring m = 2 amplitude | 0.27 ± 0.05 | 0.29 | +0.3σ |
    | Ring m = 1 azimuth (virgil convention, °)\* | 41.5 ± 10 | 40.7 | −0.1σ |
    | Ring m = 2 azimuth (virgil convention, °)\* | 56.3 ± 7 | 57.6 | +0.2σ |
    | Secondary separation (mas) | 0.81 ± 0.05 | 0.63 | −3.6σ |
    | Secondary position angle from the primary (°)\*\* | 236 ± 3 (56 + 180) | 209 | −9σ |
    | Primary UD diameter (mas) | 0.5 (fixed) | 0.5 (fixed) | none |

    The ring (size, width, tilt, orientation, brightness and both modulation amplitudes) is recovered well, from blind starts.
    The secondary is not: it is a sub-beam point source next to the star, and virgil puts more flux in it, nearer the
    star, and at a different angle than the paper.
    A possible cause, the free primary diameter trading off against the secondary, is ruled out: with the primary
    fixed the secondary rows barely moved. The flux rows depend on the spectral assumptions (power laws here, Kurucz and
    black bodies there), and the primary's fraction trades off against the secondary's.

    \* Hillen et al. define the modulation through coefficients c₁, s₁, c₂, s₂ with the convention in Kluska et al.
    (2012), which the paper does not restate. The paper's coefficients give modulation phases atan2(s₁, c₁) = −35.5° and
    ½ atan2(s₂, c₂) = −50.3°. They agree with virgil's fitted azimuths (which are in-plane azimuths from North, in the
    sense of position angle) if the paper counts its azimuth from the ring's major axis (PA 6°) in the opposite sense:
    6° − (−35.5°) = 41.5° and 6° + 50.3° = 56.3°. That mapping was chosen after seeing the fit, so these two rows show
    the two readings are consistent, not that the convention is pinned. The independent check is in image space: the
    ring's flux-centroid direction agrees (below).

    \*\* Fig. 4 puts the secondary south-west of the primary, so 56° is taken to be the primary's position from the
    secondary (236° from the primary). The published image's own central source lies at 200° (below), 36° from the
    paper's 236°, and virgil's parametric 209° is close to it, so this row is as much a statement about the paper's
    image against its model as about virgil.

    The image-fit scene has no secondary (it stays in the image, as in the paper): there the primary's fraction is 62.1 %
    against the paper's 61 % grid value, and the environment spectral index is 0.36 against 0.42.
    The reduced χ² of the parametric fit is 1.5 (the paper quotes 2.3 after inflating its uncertainties by √6).

    **Image metrics** (both images on one 0.08 mas grid, virgil convolved to the published beam, the published bitmap
    used as it stands; the central source is the brightest pixel within 2.5 mas of the star, refined by the centroid within
    0.8 mas of it; both images get the same moments):

    | Metric | Hillen+2016 | virgil |
    | --- | --- | --- |
    | Normalised cross-correlation | 1 | 0.96 (0.96 after the best sub-beam shift) |
    | Inside 3.5 mas / ring (3.5 to 16 mas) flux fraction | 5.8 % / 78.2 % | 6.3 % / 76.5 % |
    | Central point source: separation (mas), position angle (° East of North) | 1.09, 200 | 1.63, 205 |
    | Ring flux-weighted radius (mas) | 8.21 | 8.49 |
    | Ring major-axis PA (°) | 70 | 99 (the ring is nearly circular, axis ratio 0.92 and 0.91, so this is ill-defined) |
    | Asymmetry angle, ring flux centroid (° East of North), offset (mas) | 166, 1.01 | 170, 0.85 |

    ??? note "Data and method"

        **Data.** The 27 PIONIER observing blocks of ESO programme 094.D-0865 (PI Hillen), reduced by the authors
        with `pndrs`; ESO has no Phase 3 products for this programme, so this is the authors' reduced copy.
        They are two folders, which we identified from the MJDs: `A` has the nights of 2015-01-21 (9 blocks) and
        2015-01-24 (10), dated a day later in UT in the file names, and `B` has 2015-02-23 (8). The paper uses all three
        nights, as do we. The data are 828 V² points and 504 closure-phase rows (we use all four triangles per
        block, the paper's count of independent closure phases is 390), across the H band (1.53–1.77 µm).
        virgil's own dirty beam is 3.2 × 2.5 mas at PA 164°; the images above use the published ellipse instead,
        so that both are shown at the resolution of the figure in the paper.

        **Model.** The paper's single-star-subtracted scene: a uniform-disc primary (0.5 mas, fixed, flux ∝ λ⁻⁴ as in
        the paper) plus a `virgil.models.Image` environment with a power-law spectrum from `virgil.spectra` on the
        paper's grid (256 × 256 pixels of 0.15 mas), starting from `starting_image`. The secondary is left in the
        image, as in the paper. Regulariser `MaxEntropy`, weight from `l_curve` over nine weights (10⁴ to 10⁻²), taking
        the corner (weight 56). The paper instead fixes the primary's flux fraction and the environment index from a
        20 × 20 grid and uses a regularisation weight of 10¹² in its own units, so the two are not the same
        optimisation. Neither image is a ground truth.

        **Fit quality.** The virgil image has χ² per point 0.32 on 1 722 residuals. That says it is over-fitted (about
        65 000 free pixels for those residuals), not that it is better; the paper does not tabulate a χ² for its image.

        **The published image.** The published image is recovered from the bitmap by mapping each pixel's colour to
        its value through the figure's colour bar; the blue contours, star markers, beam ellipse and compass are
        filled from their neighbours. That gives morphology only, with the published stretch and beam, never
        photometry. It is a restored image already, so it is not convolved again.

        ![uv coverage, V² and closure phases](postagb/IRAS08544_inspect.png)

        ![V² and T3 fits and residuals for virgil](postagb/IRAS08544_fit.png)

        ![L-curve](postagb/IRAS08544_lcurve.png)

        ![virgil, the published image recovered from the bitmap, and their difference](postagb/IRAS08544_compare.png)

        <figure markdown>
        ![The published single-star-subtracted image of IRAS 08544-4431](postagb/literature/IRAS08544_hillen_image.png)
        <figcaption>As published: the single-star-subtracted reconstruction of IRAS 08544-4431 with 5, 3 and 1σ contours,
        Hillen et al. (2016), arXiv:1603.03023, Fig. 4 (upper panel). Reproduced from the arXiv version; Devika Kamath
        (Macquarie University), a co-author, permits its use here.</figcaption>
        </figure>

        Data: authors' reduced copy of ESO 094.D-0865, in `~/code/nuHor/data/ep1/{A,B}` (not committed).
        Provenance, permissions and credits: [`DATA_ATTRIBUTION.md`](https://github.com/benjaminpope/virgil-validation/blob/main/DATA_ATTRIBUTION.md).
        Reproduce with `scripts/postagb_literature.py --targets IRAS08544` and
        `scripts/postagb_imaging.py {inspect,parametric,image,compare,replot} docs/imaging/postagb --target IRAS08544`
        (about two minutes on a laptop).

## Data and permissions

Most targets use public ESO Phase 3 products from programme 1103.D-0797 (PI Kluska); IRAS 08544-4431 uses the authors' reduced copy of programme 094.D-0865.
Published figures are shown only where the authors have given permission, from the arXiv version.

| Target | Data | Published comparison | Status |
| --- | --- | --- | --- |
| IW Car | ESO 1103.D-0797, 23 files used (31 in the archive) | De Prins+2026 (Figs 2, 4, 6, 15; Zenodo FITS) | done |
| HR 4049 | ESO 1103.D-0797 | De Prins+2026 | pending |
| HD 108015 | ESO 1103.D-0797 | De Prins+2026 | pending |
| AI Sco | ESO 1103.D-0797 | De Prins+2026 | pending |
| EN TrA | ESO 1103.D-0797 | De Prins+2026 | pending |
| HD 95767 | ESO 1103.D-0797 | De Prins+2026 | pending |
| IRAS 15469-5311 | ESO 1103.D-0797 | De Prins+2026 | pending |
| PS Gem | ESO 1103.D-0797 | De Prins+2026 | pending |
| IRAS 08544-4431 | ESO 094.D-0865 (PI Hillen), reduced by the authors; no Phase 3 products | Hillen+2016 (Fig. 4; bitmap from the arXiv source) | done |

## Method parity

virgil's model follows the published one so that the comparison is like for like: a resolved primary with a
fixed stellar power-law index, a point secondary, and a pixel image with its own power-law spectrum
(SPARCO, Kluska et al. 2014), on the paper's grid and with the same data. What differs on purpose is the
regulariser and optimiser (MaxEntropy and L-BFGS against ORGANIC's GAN and pyBADS), the choice of weight (L-curve
corner against ORGANIC's tuned regularisation), and the uncertainty treatment (here the images are single reconstructions, without the paper's per-pixel significance maps). Images are therefore
compared at the PIONIER beam, never at native resolution, where regularised reconstructions differ by construction.
