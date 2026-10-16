"""Numbers transcribed from De Prins et al. (2026, arXiv:2605.13445v2) for the post-AGB page.

``PMOIRED`` is Table 3 (best-fit parameters of the final geometric models), each entry
``(value, minus, plus)`` with the paper's asymmetric 1-sigma errors (or ``None`` when the
model has no such component). ``IMAGE`` is Table 2 (ORGANIC/SPARCO parameters and
image-derived inner-rim ellipse). Keys: f_prim, f_sec, f_bg in %, ud in mas, sec = offset
(East, North) mas, theta_rim = rim diameter mas, i_rim and pa_rim degrees, a_mod and psi_mod
the m = 1 modulation amplitude and phase, r_hl half-light radius.
"""

PMOIRED = {
    "AI_Sco": dict(chi2=4.98, f_prim=(68.1, .4, .4), f_sec=(5.9, .2, .2), sec_dra=(-1.08, .03, .04), sec_ddec=(-1.38, .06, .06),
                   theta_rim=(5.83, .07, .09), i_rim=(45.6, 1.6, 2.0), pa_rim=(60, 5, 7), a_mod=(.87, .05, .05), psi_mod=(-9, 5, 4), r_hl=(3.41, .08, .09)),
    "EN_TrA": dict(chi2=2.40, f_prim=(72.1, .4, .4), f_sec=(5.0, .2, .2), sec_dra=(-.70, .02, .02), sec_ddec=(-1.65, .04, .04),
                   theta_rim=(5.63, .14, .16), i_rim=(58.9, 1.2, 1.1), pa_rim=(32.5, 1.7, 1.9), a_mod=(.72, .04, .04), psi_mod=(-13, 4, 4), r_hl=(4.02, .08, .09)),
    "HD95767": dict(chi2=1.97, f_prim=(48.5, .7, .7), f_sec=(21.0, .7, .7), sec_dra=(-.719, .013, .013), sec_ddec=(-.510, .009, .009),
                    theta_rim=(4.67, .04, .05), i_rim=(25.6, 1.7, 1.7), pa_rim=(14, 4, 4), a_mod=(.451, .020, .019), psi_mod=(43, 4, 4), r_hl=(3.41, .03, .03)),
    "HD108015": dict(chi2=2.08, f_prim=(61.8, .3, .3), f_bg=(8.6, .4, .5),
                     theta_rim=(5.63, .03, .03), i_rim=(31.2, .8, .7), pa_rim=(82.5, 1.1, 1.2), a_mod=(.502, .007, .008), psi_mod=(-66, 2, 2), r_hl=(3.419, .014, .014)),
    "HR4049": dict(chi2=1.71, f_prim=(65.92, .14, .14), ud=(.74, .02, .02), f_sec=(.42, .05, .05), sec_dra=(-1.70, .08, .07), sec_ddec=(-.69, .09, .10),
                   theta_rim=(14.33, .18, .16), i_rim=(56.8, 1.0, 1.0), pa_rim=(82.32, .8, 1.0), a_mod=(.42, .03, .03), psi_mod=(-38, 3, 3), r_hl=(15.44, .45, .46)),
    "IRAS15469-5311": dict(chi2=2.08, f_prim=(59.5, .6, .7), f_bg=(11.1, 1.1, 1.2),
                           theta_rim=(7.80, .06, .05), i_rim=(40.2, .7, .8), pa_rim=(53.5, 1.1, 1.0), a_mod=(.86, .02, .02), psi_mod=(74.6, 1.5, 1.6), r_hl=(5.00, .05, .05)),
    "PS_Gem": dict(chi2=0.94, f_prim=(89.01, .14, .14), ud=(.44, .02, .02),
                   theta_rim=(8.4, .6, .6), i_rim=(39, 7, 6), pa_rim=(156, 6, 5), a_mod=(.65, .05, .06), psi_mod=(35, 11, 9), r_hl=(9.1, .8, 1.3)),
}

# Table 2: ORGANIC SPARCO values and the image-derived inner-rim ellipse (theta_rim = ellipse major diameter).
IMAGE = {
    "AI_Sco": dict(f_prim=64.5, f_sec=2.76, image_index=-.59, theta_rim=(4.8, .5, .8), i_rim=(46, 14, 10), pa_rim=(135, 7, 7), r_hl=(2.89, .23, .12)),
    "EN_TrA": dict(f_prim=72.1, f_sec=5.37, image_index=.30, theta_rim=(6.1, .2, .3), i_rim=(50, 4, 3), pa_rim=(37, 7, 5), r_hl=(3.99, .27, .13)),
    "HD95767": dict(f_prim=44.1, f_sec=24.75, image_index=2.36, theta_rim=(6.01, .05, .05), i_rim=(23.7, .5, 1.0), pa_rim=(7, 5, 8), r_hl=(3.43, .13, .13)),
    "HD108015": dict(f_prim=56.0, image_index=1.45, theta_rim=(6.28, .06, .06), i_rim=(30, 3, 3), pa_rim=(81, 5, 5), r_hl=(3.78, .16, .16)),
    "HR4049": dict(f_prim=65.4, ud=.72, f_sec=.24, image_index=2.39, theta_rim=(17.2, 1.2, 1.5), i_rim=(48, 4, 4), pa_rim=(95, 8, 8), r_hl=(16.5, 1.9, 3.6)),
    "IRAS15469-5311": dict(f_prim=52.9, image_index=1.84, theta_rim=(8.99, .09, .09), i_rim=(35, 2, 2), pa_rim=(51, 3, 3), r_hl=(5.3, .2, .3)),
    "PS_Gem": dict(f_prim=87.0, ud=.37, image_index=.61, theta_rim=(9.0, .3, .4), i_rim=(28, 3, 5), pa_rim=(161, 12, 7), r_hl=(7.5, .8, .8)),
}
