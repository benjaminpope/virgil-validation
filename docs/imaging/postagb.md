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
| IRAS 08544-4431 | SPARCO image (Hillen+2016) | pending | pending |

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
        At the PIONIER beam the images correlate at 0.95 (0.95 after the best sub-beam shift): the same inner arcs
        near the star and the same large western arc. They differ in detail: the outer arc
        sits further out and is oriented differently in virgil (flux-weighted radius 11.0 against 9.9 mas,
        major-axis PA 42° against 17°), and the south-east inner arc is weaker in virgil. De Prins et al.
        themselves note that the inner arcs' azimuthal profile is not well constrained (Appendix D, SQUEEZE).

    **Parameters.** De Prins et al. could not fit a PMOIRED geometric model to IW Car (a single disc does not
    reproduce it), so there is no PMOIRED row to compare with. The comparison is with what the paper does
    report for IW Car: the SPARCO parameters of the ORGANIC reconstruction (FITS headers), and the inner-rim
    ellipses fitted to the image (Table 2; inner arcs and outer arc separately). The "virgil" columns are two
    fits to the same 23 files: the SPARCO parameters fitted together with the image, and a
    PMOIRED-style parametric model (UD star + point secondary + power-law rim with an m = 1 modulation +
    resolved background). virgil's fits are point estimates; the σ column uses the paper's uncertainty, where it gives one.

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
    outer = beyond; the paper's own rim fit needs per-pixel significances, so both images get the same moments):

    | Metric | ORGANIC | virgil |
    | --- | --- | --- |
    | Normalised cross-correlation | 1 | 0.95 |
    | Inner flux fraction / outer flux fraction | 16.7 % / 77.5 % | 14.9 % / 75.9 % |
    | Inner flux-weighted radius (mas) | 3.38 | 3.35 |
    | Outer flux-weighted radius (mas) | 9.93 | 11.05 |
    | Inner major-axis PA (°) | 61 | 58 |
    | Outer major-axis PA (°) | 17 | 42 |
    | Asymmetry angle, inner flux centroid (° East of North), offset (mas) | 219, 1.08 | 200, 1.11 |
    | Asymmetry angle, outer flux centroid (°), offset (mas) | 272, 4.04 | 280, 2.94 |

    ??? note "Data and method"

        **Data.** Phase 3 calibrated OIFITS from the ESO archive, programme 1103.D-0797 (PI J. Kluska,
        INSPIRING), the 23 exposures listed in the paper's observation log: 828 V² and 552
        closure-phase rows across the H band (1.51–1.76 µm). `postagb_data.py check` confirms
        these are the same exposures as the earlier reduced copy. The PIONIER beam is 3.26 × 2.29 mas at PA 142°.

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

## Data and permissions

The targets use public ESO Phase 3 products from programme 1103.D-0797 (PI Kluska).
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
| IRAS 08544-4431 | ESO 1103.D-0797 | Hillen+2016 | pending |

## Method parity

virgil's model follows the published one so that the comparison is like for like: a resolved primary with a
fixed stellar power-law index, a point secondary, and a pixel image with its own power-law spectrum
(SPARCO, Kluska et al. 2014), on the paper's grid and with the same data. What differs on purpose is the
regulariser and optimiser (MaxEntropy and L-BFGS against ORGANIC's GAN and pyBADS), the choice of weight (L-curve
corner against ORGANIC's tuned regularisation), and the uncertainty treatment (here the images are single reconstructions, without the paper's per-pixel significance maps). Images are therefore
compared at the PIONIER beam, never at native resolution, where regularised reconstructions differ by construction.
