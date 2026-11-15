# Upstream bug reports (MWEs)

Cross-checking against external interferometry, imaging, and orbit-fitting packages surfaced thirteen upstream defects or documentation-versus-code mismatches (`P1`–`P11` and `P13`–`P14`) alongside one documented numerical-precision note (`P12`). All fourteen entries are recorded in `trust/ledger.yml` and summarized under [Findings](findings.md#problems-found-in-other-packages).

!!! important "Upstream reporting policy"
    Automated agents in this repository **never** open issues or pull requests on external repositories. Every upstream defect remains pinned locally by a `@pytest.mark.xfail(strict=True)` test (`kind="upstream"`) so that continuous integration alerts us immediately if a version upgrade changes or resolves the behaviour. Filing issues upstream is reserved for human review, using the standalone minimal working examples below.

Every Python snippet on this page is completely self-contained: it imports only Python's standard library, `numpy`, `scipy`, `astropy`, `torch`, and the target package's public interface, with zero imports from this validation suite or the code under test and zero copied upstream source.

## Summary of upstream entries (`P1`–`P14`)

Entries `P1`–`P11` and `P13`–`P14` are upstream defects or documentation mismatches (`P5` is already filed at [`kammerje/fouriever#26`](https://github.com/kammerje/fouriever/issues/26); the remaining twelve are `to-raise`). Entry `P12` (`documented`) is a numerical-precision note included at the bottom of the page for completeness.

| ID | Package & version | One-line summary | Status | Pinning test(s) |
| --- | --- | --- | --- | --- |
| [P1](#p1) | `pmoired` `26.10.1` | `OI.setupFit(auto=True)` sets `'wl kernel': 0.0` on few-channel OIFITS files, producing silent `NaN` model observables | `to-raise` | `tests/test_pmoired_problems.py::test_p1_default_setup_gives_finite_models` |
| [P2](#p2) | `pmoired` `26.10.1` | Default radial ring sampling is coarser ($\sim 4\times$ larger $V^2$ error) than the documented `Nr = 100` | `to-raise` | `tests/test_pmoired_problems.py::test_p2_default_nr_is_100` |
| [P3](#p3) | `pmoired` `26.10.1` | Ring visibilities hit a $\sim 10^{-4}$ precision floor once a baseline exceeds 30 `(epoch × channel)` samples | `to-raise` | `tests/test_pmoired_problems.py::test_p3_rings_exact_up_to_30_samples`, `::test_p3_rings_exact_beyond_30_samples` |
| [P4](#p4) | `candid` `1.1.0` (`c255e90`) | `detectionLimit` (Absil method) underestimates $3\sigma$ flux limits by up to $1.3\%$ via $1.4\times$ bracketing and linear interpolation | `to-raise` | `tests/test_candid.py::test_candid_public_limits_are_within_their_interpolation`, `::test_p4_candid_public_limits_solve_its_criterion` |
| [P5](#p5) | `fouriever` `0.4.3` | `util.chi2_bin` does not wrap closure-phase residuals modulo $360^\circ$, inflating $\chi^2$ across the $\pm 180^\circ$ branch cut | `raised` ([`fouriever#26`](https://github.com/kammerje/fouriever/issues/26)) | `tests/test_fouriever.py::test_p5_residual_wraps_across_the_phase_cut` |
| [P6](#p6) | `ehtim` `1.3.2` | `Obsdata.tlist()` and `bllist()` coerce equal-length structured slices into 2-D tuple arrays under NumPy 2, breaking `c_phases()` | `to-raise` | `tests/test_ehtim_reconstruction.py::test_p6_closure_phases_with_equal_time_groups` |
| [P7](#p7) | `orbitize` `3.4.0` | `System.compute_model` adds the primary's systemic velocity $\gamma$ to companion RVs despite `read_input` documenting companion RVs as barycentric | `to-raise` | `tests/test_orbitize.py::test_radial_velocities_match_orbitize_system`, `::test_p7_companion_rvs_are_barycentric_as_documented` |
| [P8](#p8) | `fouriever` `0.4.3` | `uvfit.data.detlim` raises `TypeError` under NumPy 2 and SciPy 1.18 due to 1-element array-to-scalar conversions | `to-raise` | `tests/test_fouriever_limits.py::test_fouriever_public_detlim_against_its_exact_criteria` |
| [P9](#p9) | `oimodeler` `0.8.1` | Image-based `oimExpRing` differs by $8 \times 10^{-3}$ from the exact Hankel transform of its documented profile across grid sizes | `to-raise` | `tests/test_polar_disk_vs_codes.py::test_oimodeler_exp_ring_matches_the_exact_transform` |
| [P10](#p10) | `vis-r` `0.0.1` (`f534670`) | `r_prof_erf2_power_ggap` swaps its gap depth and gap radius parameter indices relative to its docstring and `r_prof_erf2_power_ggap2` | `to-raise` | `tests/test_polar_disk_vs_codes.py::test_vis_r_single_gap_profile_matches_its_two_gap_form` |
| [P11](#p11) | `mpol` `0.3.1` | `losses.log_likelihood` docstring writes $+\sum_i w_i$ instead of $+\sum_i \ln w_i$ for the Gaussian log-normalization | `to-raise` | `tests/test_alma_mpol.py::test_mpol_log_likelihood_equals_virgil_with_s_equal_g_times_s_mpol` |
| [P12](#p12) | `mpol` `0.3.1` | Precision note: default `NuFFT` table interpolation ($6.3 \times 10^{-4}$ relative $L_2$) and `float32` `GaussConvFourier.taper_2D` buffer under `float64` | `documented` | `tests/test_alma_mpol.py::test_mpol_virgil_and_the_exact_dft_agree_on_the_forward_model`, `::test_mpol_with_its_accurate_nufft_agrees_to_1e_5` |
| [P13](#p13) | `oimodeler` `0.8.1` | `oimFastRotator` flips the visible pole's sky position angle by $180^\circ$ via `np.rot90` | `to-raise` | `tests/test_rotator_convention.py::test_p13_oimodeler_fast_rotator_pole_pa_inversion` |
| [P14](#p14) | `oimodeler` `0.8.1` | `oimFastRotator` replaces quadratic centrifugal gravity darkening with an unphysical `rot**0.4` power law | `to-raise` | `tests/test_rotator_convention.py::test_p14_oimodeler_fast_rotator_gravity_darkening_scaling` |

---

## PMOIRED

* **Upstream repository**: [`https://github.com/amerand/PMOIRED`](https://github.com/amerand/PMOIRED) (see [PMOIRED conventions](pmoired.md))
* **Tested environment**: `pmoired==26.10.1`, Python `3.12.12`, NumPy `2.5.4`, SciPy `1.18.1`, Astropy `8.0.2`

### P1: Silent `NaN` visibilities from `OI.setupFit(auto=True)` on few-channel OIFITS files { #p1 }

* **Status**: `to-raise`
* **Mechanism & diagnosis**: By default, `OI.setupFit` runs with `auto=True` and estimates a spectral-smearing kernel width (`'wl kernel'`) from the `OI_WAVELENGTH` table. On OIFITS files with only a few widely spaced channels and narrow `EFF_BAND` (here four channels across 1.5–2.4 µm with 1 nm bandwidth), `setupFit` sets `'wl kernel': 0.0`. Subsequent calls to `oimodels.VmodelOI` divide by zero inside the zero-width wavelength-convolution kernel and return `NaN` for every model observable (`V2` and `T3PHI`) without warning. Passing `auto=False` bypasses automatic kernel estimation and restores finite values.
* **Pinning test**: `tests/test_pmoired_problems.py::test_p1_default_setup_gives_finite_models`

**Standalone MWE**:

```python
import tempfile
import numpy as np
from astropy.io import fits
import pmoired, pmoired.oimodels as om

wl = np.linspace(1.5e-6, 2.4e-6, 4)
hdul = fits.HDUList([
    fits.PrimaryHDU(),
    fits.BinTableHDU.from_columns([
        fits.Column("EFF_WAVE", "D", unit="m", array=wl),
        fits.Column("EFF_BAND", "D", unit="m", array=np.full_like(wl, 1e-9)),
    ], name="OI_WAVELENGTH", header=fits.Header({"OI_REVN": 2, "INSNAME": "SYNTH"})),
    fits.BinTableHDU.from_columns([
        fits.Column("TARGET_ID", "I", array=[1]), fits.Column("TARGET", "16A", array=["STAR"]),
    ], name="OI_TARGET", header=fits.Header({"OI_REVN": 2})),
    fits.BinTableHDU.from_columns([
        fits.Column("TARGET_ID", "I", array=[1, 1]), fits.Column("MJD", "D", array=[60000.0, 60000.1]),
        fits.Column("UCOORD", "D", unit="m", array=[40.0, 60.0]), fits.Column("VCOORD", "D", unit="m", array=[20.0, -30.0]),
        fits.Column("VIS2DATA", "4D", array=np.ones((2, 4)) * 0.8), fits.Column("VIS2ERR", "4D", array=np.ones((2, 4)) * 0.01),
        fits.Column("STA_INDEX", "2I", array=[[1, 2], [1, 2]]), fits.Column("FLAG", "4L", array=np.zeros((2, 4), bool)),
    ], name="OI_VIS2", header=fits.Header({"OI_REVN": 2, "INSNAME": "SYNTH", "ARRNAME": "VLTI", "DATE-OBS": "2024-01-01"})),
])
with tempfile.NamedTemporaryFile(suffix=".fits") as f:
    hdul.writeto(f.name, overwrite=True)
    oi = pmoired.OI(f.name, verbose=False)
    oi.setupFit({"obs": ["V2"]})  # default auto=True sets 'wl kernel' = 0.0
    wl_k = oi.data[0]["fit"].get("wl kernel")
    v2_auto = next(iter(om.VmodelOI(oi.data[0], {"ud": 1.0})["OI_VIS2"].values()))["V2"]
    oi.setupFit({"obs": ["V2"]}, auto=False)
    v2_noauto = next(iter(om.VmodelOI(oi.data[0], {"ud": 1.0})["OI_VIS2"].values()))["V2"]
print("wl kernel (auto=True):", wl_k, "all NaN?:", np.all(np.isnan(v2_auto)))
print("finite (auto=False)?:", np.all(np.isfinite(v2_noauto)))
```

**Observed output**:

```text
-- Please aknowledge the use of PMOIRED -----------
https://github.com/amerand/PMOIRED#using-and-quoting-pmoired
--------------------------------------------------- 
  > Warning: no OI_ARRAY extension, telescopes will have default names
setupFit> 0/SYNTH: setting 'wl kernel':0.0 according to WL definition
wl kernel (auto=True): 0.0 all NaN?: True
finite (auto=False)?: True
```

* **Expected behaviour**: `v2_auto` should be finite and match `v2_noauto`, or `setupFit(auto=True)` should disable spectral smearing whenever `'wl kernel'` evaluates to zero.

### P2: Default radial ring sampling is coarser than the documented `Nr = 100` { #p2 }

* **Status**: `to-raise`
* **Mechanism & diagnosis**: PMOIRED's documentation states that radial ring models (`diamin`, `diamout`) default to `Nr = 100` radial samples. Evaluating a uniform 2–4 mas annulus against its closed-form two-Airy-disk visibility gives a maximum squared-visibility error of $3.0 \times 10^{-5}$ under a default `setupFit`, compared with $7.4 \times 10^{-6}$ when `'Nr': 100` is passed explicitly. Because radial quadrature on a sharp-edged annulus converges at second order in the radial step, the $4.1\times$ larger error indicates an effective default of roughly 50 radial samples rather than 100.
* **Pinning test**: `tests/test_pmoired_problems.py::test_p2_default_nr_is_100`

**Standalone MWE**:

```python
import numpy as np
from scipy.special import j1
import pmoired, pmoired.oifake as oifake, pmoired.oimodels as om

MAS = np.pi / 180 / 3600 / 1000
fake = oifake.makeFakeVLTI(["U1", "U2"], target=(0, -50), lst=[0.0], wl=np.linspace(1.5, 2.4, 20), model={"ud": 0.0})
b_wl = fake["OI_VIS2"]["U1U2"]["B/wl"] * 1e6  # m/um -> rad^-1
airy = lambda d: 2 * j1(np.pi * d * MAS * b_wl) / (np.pi * d * MAS * b_wl)
v2_exact = ((16.0 * airy(4.0) - 4.0 * airy(2.0)) / 12.0) ** 2

oi = pmoired.OI(); oi.data = [fake]
oi.setupFit({"obs": ["V2"]}, auto=False)
err_default = np.max(np.abs(om.VmodelOI(oi.data[0], {"diamin": 2.0, "diamout": 4.0})["OI_VIS2"]["U1U2"]["V2"] - v2_exact))
oi.data[0]["fit"]["Nr"] = 100
err_nr100 = np.max(np.abs(om.VmodelOI(oi.data[0], {"diamin": 2.0, "diamout": 4.0})["OI_VIS2"]["U1U2"]["V2"] - v2_exact))
print(f"max |dV2| default: {err_default:.2e}, explicit Nr=100: {err_nr100:.2e}")
```

**Observed output**:

```text
-- Please aknowledge the use of PMOIRED -----------
https://github.com/amerand/PMOIRED#using-and-quoting-pmoired
--------------------------------------------------- 
max |dV2| default: 3.02e-05, explicit Nr=100: 7.39e-06
```

* **Expected behaviour**: Leaving `'Nr'` unset in `setupFit` should produce the same accuracy ($7.4 \times 10^{-6}$) as setting `'Nr': 100` explicitly, or the documentation should record the actual default.

### P3: Ring visibilities hit a $\sim 10^{-4}$ precision floor beyond 30 samples per baseline { #p3 }

* **Status**: `to-raise`
* **Mechanism & diagnosis**: When a baseline contains at most 30 `(epoch × channel)` samples, PMOIRED evaluates the radial ring transform directly at each spatial frequency and converges at second order in `Nr` down to $8.1 \times 10^{-9}$ at `Nr = 3000`. As soon as a baseline exceeds 30 samples (such as 1 epoch × 35 channels below, or 7 epochs × 6 channels in multi-epoch fits), an undocumented radial-transform interpolation switches on silently and caps squared-visibility accuracy around $2.5 \times 10^{-5}$ to $1.4 \times 10^{-4}$ regardless of `Nr`. Point, uniform-disk, and Gaussian components are unaffected.
* **Pinning tests**: `tests/test_pmoired_problems.py::test_p3_rings_exact_up_to_30_samples` (control), `tests/test_pmoired_problems.py::test_p3_rings_exact_beyond_30_samples`

**Standalone MWE**:

```python
import numpy as np
from scipy.special import j1
import pmoired.oifake as oifake, pmoired.oimodels as om

MAS = np.pi / 180 / 3600 / 1000
for n_wl in (30, 35):
    fake = oifake.makeFakeVLTI(["U1", "U2"], target=(0, -50), lst=[0.0], wl=np.linspace(1.5, 2.4, n_wl), model={"ud": 0.0})
    fake["fit"] = {"obs": ["V2"], "Nr": 3000}
    b_wl = fake["OI_VIS2"]["U1U2"]["B/wl"] * 1e6  # m/um -> rad^-1
    airy = lambda d: 2 * j1(np.pi * d * MAS * b_wl) / (np.pi * d * MAS * b_wl)
    v2_exact = ((16.0 * airy(4.0) - 4.0 * airy(2.0)) / 12.0) ** 2
    v2_pm = om.VmodelOI(fake, {"diamin": 2.0, "diamout": 4.0})["OI_VIS2"]["U1U2"]["V2"]
    print(f"n_samples={n_wl}: max |dV2| at Nr=3000 = {np.max(np.abs(v2_pm - v2_exact)):.2e}")
```

**Observed output**:

```text
n_samples=30: max |dV2| at Nr=3000 = 8.05e-09
n_samples=35: max |dV2| at Nr=3000 = 2.47e-05
```

* **Expected behaviour**: Increasing `n_samples` from 30 to 35 at `Nr = 3000` should maintain $\sim 10^{-8}$ accuracy rather than jumping by three orders of magnitude, or an explicit flag should control whether interpolation is used.

---

## CANDID

* **Upstream repository**: [`https://github.com/amerand/CANDID`](https://github.com/amerand/CANDID) (see [CANDID conventions](candid.md))
* **Tested environment**: `candid 1.1.0 | 2023/07/20` (commit `c255e908f721a99c8ecdd5576f949a6fe0efee83`), Python `3.12.12`, NumPy `2.5.3`, SciPy `1.18.1`, Astropy `8.0.1`

### P4: `detectionLimit` (Absil method) underestimates $3\sigma$ flux limits via $1.4\times$ bracketing and linear interpolation { #p4 }

* **Status**: `to-raise`
* **Mechanism & diagnosis**: Under the [Absil et al. (2011)](https://doi.org/10.1051/0004-6361/201117719) detection-limit criterion – which converts the chi-squared tail probability between the single-star null and a fixed-position binary model into a two-sided Gaussian significance – `candid.Open.detectionLimit(methods=["Absil"])` brackets the $3\sigma$ flux ratio by stepping candidate companion fluxes in multiplicative factors of $1.4$ and linearly interpolating `candid._nSigmas` across the crossing. Because `_nSigmas(f)` is convex in companion flux $f$ (curving upward as $\Delta\chi^2$ grows quadratically with $f$), the linear secant chord lies above the curve and crosses $3\sigma$ at a lower flux than the exact root found by [Brent's method](https://en.wikipedia.org/wiki/Brent%27s_method) (which combines root bracketing, bisection, and inverse quadratic interpolation to machine precision). On the minimal 3-baseline ($N = 12$) synthetic file below, sublinear chi-squared-to-Gaussian tail scaling partially offsets the quadratic $\Delta\chi^2$ growth, leaving a mean bias of $-0.13\%$ and a worst-case bias of $-0.34\%$. On realistic 4-UT observations ($N \gtrsim 100$ data points in `tests/test_candid.py`), where a $3\sigma$ excursion is a small fractional perturbation ($\Delta\chi^2 \ll N$) and `_nSigmas` preserves steeper quadratic curvature in $f$, the secant underestimate averages $-1.0\%$ and reaches $-1.3\%$ ($\sim 0.01\text{ mag}$ optimistic).
* **Pinning tests**: `tests/test_candid.py::test_candid_public_limits_are_within_their_interpolation`, `tests/test_candid.py::test_p4_candid_public_limits_solve_its_criterion`

**Standalone MWE**:

```python
import contextlib, io, tempfile, warnings, numpy as np
from astropy.io import fits
from scipy.optimize import brentq
import candid

warnings.filterwarnings("ignore")
candid.CONFIG.update({"Ncores": 1, "progress bar": False, "long exec warning": None})
wl = np.linspace(1.6e-6, 2.2e-6, 4)
u, v = np.array([30.0, 55.0, -25.0]), np.array([20.0, -35.0, -55.0])
hdul = fits.HDUList([
    fits.PrimaryHDU(),
    fits.BinTableHDU.from_columns([fits.Column("EFF_WAVE", "D", unit="m", array=wl), fits.Column("EFF_BAND", "D", unit="m", array=np.zeros(4))],
                                  name="OI_WAVELENGTH", header=fits.Header({"OI_REVN": 2, "INSNAME": "SYN"})),
    fits.BinTableHDU.from_columns([fits.Column("TARGET_ID", "I", array=[1]), fits.Column("TARGET", "8A", array=["S"])],
                                  name="OI_TARGET", header=fits.Header({"OI_REVN": 2})),
    fits.BinTableHDU.from_columns([
        fits.Column("TEL_NAME", "8A", array=["T1", "T2", "T3"]), fits.Column("STA_NAME", "8A", array=["S1", "S2", "S3"]),
        fits.Column("STA_INDEX", "I", array=[1, 2, 3]), fits.Column("DIAMETER", "E", unit="m", array=[8.0, 8.0, 8.0]),
        fits.Column("STAXYZ", "3D", unit="m", array=np.zeros((3, 3))),
    ], name="OI_ARRAY", header=fits.Header({"OI_REVN": 2, "ARRNAME": "A", "FRAME": "GEOCENTRIC", "ARRAYX": 0.0, "ARRAYY": 0.0, "ARRAYZ": 0.0})),
    fits.BinTableHDU.from_columns([
        fits.Column("TARGET_ID", "I", array=[1, 1, 1]), fits.Column("MJD", "D", array=[60000.0]*3),
        fits.Column("UCOORD", "D", unit="m", array=u), fits.Column("VCOORD", "D", unit="m", array=v),
        fits.Column("VIS2DATA", "4D", array=np.full((3, 4), 0.985)), fits.Column("VIS2ERR", "4D", array=np.full((3, 4), 0.01)),
        fits.Column("STA_INDEX", "2I", array=[[1, 2], [2, 3], [1, 3]]), fits.Column("FLAG", "4L", array=np.zeros((3, 4), bool)),
    ], name="OI_VIS2", header=fits.Header({"OI_REVN": 2, "INSNAME": "SYN", "ARRNAME": "A", "DATE-OBS": "2024-01-01"})),
])
with tempfile.NamedTemporaryFile(suffix=".fits") as f, contextlib.redirect_stdout(io.StringIO()):
    hdul.writeto(f.name, overwrite=True)
    o = candid.Open(f.name, rmin=2.0, rmax=10.0); o.observables = ["v2"]
    o.detectionLimit(step=2.0, diam=0.0, fig=None, drawMaps=False, rmin=2.0, rmax=10.0, methods=["Absil"])
    ax = np.linspace(-10.0, 10.0, int(np.ceil(20.0 / 2.0)))
    c0 = candid._chi2Func({"x": 0.0, "y": 0.0, "f": 0.0, "diam*": 0.0, "alpha*": 0.0}, o._chi2Data, ["v2"], o.instruments)
    diffs = [o.allf3s["Absil"][j, i] / brentq(lambda fp: candid._nSigmas(candid._chi2Func(
        {"x": x0, "y": y0, "f": fp, "diam*": 0.0, "alpha*": 0.0}, o._chi2Data, ["v2"], o.instruments), c0, o.ndata()) - 3.0, 1e-3, 50.0) - 1
        for j, y0 in enumerate(ax) for i, x0 in enumerate(ax) if 2.0 <= np.hypot(x0, y0) <= 10.0]
print(f"n_cells={len(diffs)}: mean rel diff={np.mean(diffs):+.3%}, worst rel diff={min(diffs):+.3%}")
```

**Observed output**:

```text
n_cells=56: mean rel diff=-0.130%, worst rel diff=-0.343%
```

* **Expected behaviour**: Interpolating in $f^2$ or refining the $1.4\times$ bracket with a few root-finding steps makes `o.allf3s["Absil"]` solve CANDID's own `_nSigmas == 3.0` criterion to $< 10^{-3}$ relative across both small and large `ndata`.

---

## fouriever

* **Upstream repository**: [`https://github.com/kammerje/fouriever`](https://github.com/kammerje/fouriever) (see [fouriever conventions](fouriever.md))
* **Tested environment**: `fouriever==0.4.3`, Python `3.12.12`, NumPy `2.5.3`, SciPy `1.18.1`, Astropy `8.0.1`

### P5: Closure-phase residual in `util.chi2_bin` is not wrapped modulo $360^\circ$ { #p5 }

* **Status**: `raised` – filed upstream as [`kammerje/fouriever#26`](https://github.com/kammerje/fouriever/issues/26)
* **Mechanism & diagnosis**: `fouriever.util.chi2_bin` computes the model closure phase (in radians) as the linear combination `cpmat @ phase` of baseline phases and subtracts the measured `cp` directly without wrapping the angular difference into $(-\pi, \pi]$. Near-equal binaries ($f \approx 1$) routinely produce closure phases near $\pm 180^\circ$. For the $f = 0.99$ binary below, the exact model closure phase is $+178.01^\circ$ (where `util.chi2_bin` evaluates to $1.0 \times 10^{-26}$); stepping the measurement $+3.0^\circ$ across the branch cut to $-178.99^\circ$ with $0.5^\circ$ uncertainty represents a $6\sigma$ ($3.0^\circ$) excursion with true $\chi^2 = 36.0$, whereas `util.chi2_bin` forms an unwrapped $357.0^\circ$ difference and returns $\chi^2 = (357.0 / 0.5)^2 = 509\,796.0$.
* **Pinning test**: `tests/test_fouriever.py::test_p5_residual_wraps_across_the_phase_cut`

**Standalone MWE**:

```python
import numpy as np
from fouriever import util

mas = np.deg2rad(1.0 / 3.6e6)
f, dra, ddec, wl = 0.99, 5.91, -3.0, 2.0e-6
u1, v1, u2, v2 = np.array([[45.0]]), np.array([[10.0]]), np.array([[35.0]]), np.array([[-25.0]])
u3, v3 = -(u1 + u2), -(v1 + v2)
vis = lambda u, v: (1.0 + f * np.exp(-2j * np.pi * (u * dra + v * ddec) * mas / wl)) / (1.0 + f)
cp_true_deg = np.rad2deg(np.angle(vis(u1, v1)) + np.angle(vis(u2, v2)) + np.angle(vis(u3, v3))).item()
cp_meas_deg = (cp_true_deg + 3.0 + 180.0) % 360.0 - 180.0  # 3 deg across +-180 deg branch cut

def eval_chi2(cp_deg):
    data = [{"uu": np.array([u1[0, 0], u2[0, 0], -u3[0, 0]])[:, None] / wl,
             "vv": np.array([v1[0, 0], v2[0, 0], -v3[0, 0]])[:, None] / wl,
             "cpmat": np.array([[1.0, 1.0, -1.0]]), "cp": np.array([[np.deg2rad(cp_deg)]]),
             "dcp": np.array([[np.deg2rad(0.5)]]), "pa": 0.0, "covflag": False}]
    return float(util.chi2_bin(np.array([f, dra, ddec]), data, ["cp"], cov=False))

print(f"cp_true={cp_true_deg:.2f} deg, cp_meas={cp_meas_deg:.2f} deg")
print(f"chi2 at cp_true (expected 0.0): {eval_chi2(cp_true_deg):.2e}")
print(f"chi2 at cp_meas (expected 36.0): {eval_chi2(cp_meas_deg):.1f}")
```

**Observed output**:

```text
cp_true=178.01 deg, cp_meas=-178.99 deg
chi2 at cp_true (expected 0.0): 1.04e-26
chi2 at cp_meas (expected 36.0): 509796.0
```

* **Expected behaviour**: Wrapping the closure-phase residual into $(-\pi, \pi]$ before squaring (or before multiplying by the inverse covariance) yields $\chi^2 = 36.0$ for a $3.0^\circ$ residual at $0.5^\circ$ uncertainty.

### P8: `uvfit.data.detlim` crashes with `TypeError` under NumPy 2 and SciPy 1.18 { #p8 }

* **Status**: `to-raise`
* **Mechanism & diagnosis**: NumPy 2.0 removed implicit conversion of 1-element arrays to Python scalars (PEP 0050), and SciPy $\ge 1.14$ enforces scalar objective values in `scipy.optimize.minimize`. In `fouriever==0.4.3`, `uvfit.data.detlim` fails on two separate paths: when `v2` observables are included, formatting the fitted uniform-disk diameter via `'%.3f' % fit['p'][0]` passes a 1-D array of length 1 to `%` formatting; and on closure-phase-only runs (`observables=["cp"]`), `lim_absil` and `lim_injection` pass `util.nsigma`'s `(1,)`-shaped array output directly into `scipy.optimize.minimize(..., method="L-BFGS-B")`. Both paths raise `TypeError: only 0-dimensional arrays can be converted to Python scalars`.
* **Pinning test**: `tests/test_fouriever_limits.py::test_fouriever_public_detlim_against_its_exact_criteria`

**Standalone MWE**:

```python
import os, tempfile, numpy as np
from astropy.io import fits
from fouriever import uvfit

wl = np.linspace(1.6e-6, 2.2e-6, 4)
u, v = np.array([30.0, 55.0, -25.0]), np.array([20.0, -35.0, -55.0])
hdul = fits.HDUList([
    fits.PrimaryHDU(),
    fits.BinTableHDU.from_columns([fits.Column("EFF_WAVE", "D", unit="m", array=wl), fits.Column("EFF_BAND", "D", unit="m", array=np.full(4, 1e-8))],
                                  name="OI_WAVELENGTH", header=fits.Header({"OI_REVN": 2, "INSNAME": "SYN"})),
    fits.BinTableHDU.from_columns([fits.Column("TARGET_ID", "I", array=[1]), fits.Column("TARGET", "8A", array=["S"])],
                                  name="OI_TARGET", header=fits.Header({"OI_REVN": 2})),
    fits.BinTableHDU.from_columns([
        fits.Column("TEL_NAME", "8A", array=["T1", "T2", "T3"]), fits.Column("STA_NAME", "8A", array=["S1", "S2", "S3"]),
        fits.Column("STA_INDEX", "I", array=[1, 2, 3]), fits.Column("DIAMETER", "E", unit="m", array=[8.0, 8.0, 8.0]),
        fits.Column("STAXYZ", "3D", unit="m", array=np.zeros((3, 3))),
    ], name="OI_ARRAY", header=fits.Header({"OI_REVN": 2, "ARRNAME": "A", "FRAME": "GEOCENTRIC", "ARRAYX": 0.0, "ARRAYY": 0.0, "ARRAYZ": 0.0})),
    fits.BinTableHDU.from_columns([
        fits.Column("TARGET_ID", "I", array=[1, 1, 1]), fits.Column("MJD", "D", array=[60000.0]*3),
        fits.Column("UCOORD", "D", unit="m", array=u), fits.Column("VCOORD", "D", unit="m", array=v),
        fits.Column("VIS2DATA", "4D", array=np.ones((3, 4))), fits.Column("VIS2ERR", "4D", array=np.full((3, 4), 0.01)),
        fits.Column("STA_INDEX", "2I", array=[[1, 2], [2, 3], [1, 3]]), fits.Column("FLAG", "4L", array=np.zeros((3, 4), bool)),
    ], name="OI_VIS2", header=fits.Header({"OI_REVN": 2, "INSNAME": "SYN", "ARRNAME": "A", "DATE-OBS": "2024-01-01"})),
    fits.BinTableHDU.from_columns([
        fits.Column("TARGET_ID", "I", array=[1]), fits.Column("MJD", "D", array=[60000.0]),
        fits.Column("T3AMP", "4D", array=np.ones((1, 4))), fits.Column("T3AMPERR", "4D", array=np.full((1, 4), 0.01)),
        fits.Column("T3PHI", "4D", array=np.zeros((1, 4))), fits.Column("T3PHIERR", "4D", array=np.full((1, 4), 0.5)),
        fits.Column("U1COORD", "D", unit="m", array=[30.0]), fits.Column("V1COORD", "D", unit="m", array=[20.0]),
        fits.Column("U2COORD", "D", unit="m", array=[55.0]), fits.Column("V2COORD", "D", unit="m", array=[-35.0]),
        fits.Column("STA_INDEX", "3I", array=[[1, 2, 3]]), fits.Column("FLAG", "4L", array=np.zeros((1, 4), bool)),
    ], name="OI_T3", header=fits.Header({"OI_REVN": 2, "INSNAME": "SYN", "ARRNAME": "A", "DATE-OBS": "2024-01-01"})),
])
with tempfile.TemporaryDirectory() as tmp:
    hdul.writeto(os.path.join(tmp, "syn.fits"), overwrite=True)
    d = uvfit.data(tmp, ["syn.fits"])
    try:
        d.detlim(sigma=3.0, cov=False, sep_range=(2.0, 10.0), step_size=4.0)
    except Exception as exc:
        print("detlim raised:", repr(exc))
```

**Observed output**:

```text
Computing best fit uniform disk diameter (DO NOT TRUST UNCERTAINTIES)
detlim raised: TypeError('only 0-dimensional arrays can be converted to Python scalars')
```

* **Expected behaviour**: Extracting Python scalars via `.item()` (or `float(val[0])`) in diameter logging and inside `util.nsigma` / `lim_absil` / `lim_injection` allows `detlim` to complete under NumPy 2 and SciPy 1.18.

---

## eht-imaging

* **Upstream repository**: [`https://github.com/achael/eht-imaging`](https://github.com/achael/eht-imaging) (see [eht-imaging conventions](ehtim.md))
* **Tested environment**: `ehtim==1.3.2`, Python `3.12.12`, NumPy `2.5.3`, SciPy `1.18.1`, Astropy `8.0.1`

### P6: `Obsdata.tlist()` and `bllist()` coerce equal-length structured slices into 2-D tuple arrays under NumPy 2 { #p6 }

* **Status**: `to-raise`
* **Mechanism & diagnosis**: `Obsdata.tlist()` (and `bllist()`) partitions visibility records into a Python list `outlist` of structured NumPy sub-arrays (one per timestamp or baseline) and returns `np.array(outlist, dtype=object)`. Whenever every timestamp contains the same number of baselines – as in any unflagged synthetic observation – NumPy 2 unpacks the equal-length structured arrays into a **2-D** object array whose `(i, j)` entries are plain Python `tuple`s rather than a 1-D object array of structured `ndarray` slices. Downstream methods such as `obs.c_phases()` then index each row by field name (`tdata['time']`, `tdata['t1']`) and crash immediately with `TypeError: tuple indices must be integers or slices, not str`.
* **Pinning test**: `tests/test_ehtim_reconstruction.py::test_p6_closure_phases_with_equal_time_groups`

**Standalone MWE**:

```python
import numpy as np
import ehtim as eh
from ehtim import const_def as ehc

sites = ["S1", "S2", "S3"]
tarr = np.zeros(3, dtype=ehc.DTARR)
tarr["site"], tarr["x"], tarr["sefdr"], tarr["sefdl"] = sites, [1.0, 2.0, 3.0], 1.0, 1.0
rows = [(t, a, b) for t in (0.0, 1.0) for a, b in (("S1", "S2"), ("S2", "S3"), ("S1", "S3"))]
table = np.zeros(len(rows), dtype=ehc.DTPOL_STOKES)
for k, (t, a, b) in enumerate(rows):
    table[k]["time"], table[k]["tint"], table[k]["t1"], table[k]["t2"] = t, 1.0, a, b
    table[k]["u"], table[k]["v"], table[k]["vis"], table[k]["sigma"] = 1e6 * (k + 1), 0.0, 1.0 + 0j, 0.01
obs = eh.obsdata.Obsdata(0.0, -50.0, 230e9, 1e9, table, tarr, polrep="stokes")
tl = obs.tlist()
print("tlist shape:", tl.shape, "entry (0, 0) type:", type(tl[0, 0]))
try:
    obs.c_phases(count="min")
except Exception as exc:
    print("c_phases raised:", repr(exc))
```

**Observed output**:

```text
Warning: No NFFT installed!
Welcome to eht-imaging!

tlist shape: (2, 3) entry (0, 0) type: <class 'tuple'>
c_phases raised: TypeError('tuple indices must be integers or slices, not str')
```

* **Expected behaviour**: Allocating an explicit 1-D object container (`out = np.empty(len(outlist), dtype=object); out[:] = outlist`) preserves structured `ndarray` elements for equal-sized groups so `obs.c_phases(count="min")` returns the two closure-phase records.

---

## orbitize!

* **Upstream repository**: [`https://github.com/sblunt/orbitize`](https://github.com/sblunt/orbitize) (see [orbitize! conventions](orbitize.md))
* **Tested environment**: `orbitize==3.4.0`, Python `3.12.12`, NumPy `2.5.3`, SciPy `1.18.1`, Astropy `8.0.1`

### P7: `System.compute_model` adds the primary's systemic velocity $\gamma$ to companion RVs despite `read_input` documenting companion RVs as barycentric { #p7 }

* **Status**: `to-raise`
* **Mechanism & diagnosis**: `orbitize.read_input.read_file` documents radial-velocity measurements with `object > 0` (companions) as velocities **relative to the system barycentre**; accordingly, when a table contains companion RVs alone, `system.System` introduces no `gamma_<instrument>` parameter. However, as soon as primary RV rows (`object == 0`) sharing the same `instrument` label are included in the input table, `System.compute_model` adds `gamma_<instrument>` to **every** RV row carrying that instrument name – shifting the companion's predicted RV by the systemic velocity (`+7.0 km/s` below). Physically, adding $\gamma$ is right when primary and companion RVs are both measured on an absolute wavelength scale by the same spectrograph, whereas omitting $\gamma$ is right when companion RVs are pre-referenced to the barycentre; the documentation in `read_input` and the indexing in `System.compute_model` state opposite conventions.
* **Pinning tests**: `tests/test_orbitize.py::test_radial_velocities_match_orbitize_system`, `tests/test_orbitize.py::test_p7_companion_rvs_are_barycentric_as_documented`

**Standalone MWE**:

```python
import numpy as np
from astropy.table import Table
from orbitize import read_input, system

def comp_rv(objects, gamma=7.0):
    rows = [(59000.0, o) for o in objects]
    tbl = Table({"epoch": [r[0] for r in rows], "object": [r[1] for r in rows],
                 "rv": [0.0]*len(rows), "rv_err": [1.0]*len(rows), "instrument": ["spec"]*len(rows)})
    sys_ = system.System(1, read_input.read_file(tbl), 1.2, 50.0, mass_err=0.1, plx_err=0.1,
                         tau_ref_epoch=58849.0, fit_secondary_mass=True)
    idx = sys_.basis.param_idx
    p = np.zeros(len(idx))
    for k, v in {"sma1": 5.0, "ecc1": 0.2, "inc1": 1.0, "aop1": 0.5, "pan1": 1.2, "tau1": 0.3,
                 "plx": 50.0, "m1": 0.3, "m0": 1.2, "gamma_spec": gamma, "sigma_spec": 0.0}.items():
        if k in idx:
            p[idx[k]] = v
    return float(np.asarray(sys_.compute_model(p)[0])[-1, 0])

print("companion RV (alone, no primary RVs):", comp_rv([1]))
print("companion RV (with primary RVs, gamma=7 km/s):", comp_rv([0, 1]))
print("difference (km/s):", comp_rv([0, 1]) - comp_rv([1]))
```

**Observed output**:

```text
companion RV (alone, no primary RVs): 2.9006121153505244
companion RV (with primary RVs, gamma=7 km/s): 9.900612115350524
difference (km/s): 7.0
```

* **Expected behaviour**: Either `System.compute_model` should restrict `gamma_<instrument>` to `object == 0` so companion RVs remain barycentric regardless of whether primary RVs are present (`difference == 0.0 km/s`), or `read_input` should document shared-instrument companion RVs as absolute (including $\gamma$) and allocate `gamma_<instrument>` when fitting absolute companion RVs alone.

---

## oimodeler

* **Upstream repository**: [`https://github.com/oimodeler/oimodeler`](https://github.com/oimodeler/oimodeler) (see [PolarDisk](../models/polar_disk.md) and [Rapid rotators](../models/rapid_rotators.md))
* **Tested environment**: `oimodeler==0.8.1`, Python `3.12.12`, NumPy `2.5.4`, SciPy `1.18.1`, Astropy `8.0.2`

### P9: Image-based `oimExpRing` differs by $8 \times 10^{-3}$ from the exact Hankel transform of its documented profile across grid resolutions { #p9 }

* **Status**: `to-raise`
* **Mechanism & diagnosis**: `oimExpRing` documents an axisymmetric sharp-inner-edge exponential ring profile $I(r) \propto \exp(-0.692\,(r - r_0) / \text{fwhm})$ for $r \ge r_0 = d / 2$, truncated at $r_0 + 8\,\text{fwhm}$. By circular symmetry, its normalized visibility is the order-zero [Hankel transform](https://en.wikipedia.org/wiki/Hankel_transform) ([NIST DLMF §10.22](https://dlmf.nist.gov/10.22), which reduces the 2-D Fourier transform of any axisymmetric sky brightness to a 1-D radial $J_0$ integral). Comparing `oimModel([oimExpRing(d=4.0, fwhm=2.1, dim=N)]).getComplexCoherentFlux` against adaptive Gauss–Kronrod quadrature (`scipy.integrate.quad`) of that exact radial profile leaves a resolution-invariant discrepancy of $8.0 \times 10^{-3}$ to $8.5 \times 10^{-3}$ across `dim = 256`, `512`, and `1024` (where Fourier-space components in `oimodeler` agree to $10^{-15}$).
* **Pinning test**: `tests/test_polar_disk_vs_codes.py::test_oimodeler_exp_ring_matches_the_exact_transform`

**Standalone MWE**:

```python
import numpy as np
from scipy.integrate import quad
from scipy.special import j0
import oimodeler as oim

mas = np.deg2rad(1.0 / 3.6e6)
d, fwhm, wl = 4.0, 2.1, 2.2e-6
r0, B = d / 2.0, np.linspace(0.0, 160.0, 32)
qs = (B / wl) * mas
den, _ = quad(lambda r: np.exp(-0.692 * (r - r0) / fwhm) * r, r0, r0 + 8 * fwhm, limit=400)
v_exact = np.array([quad(lambda r: np.exp(-0.692 * (r - r0) / fwhm) * j0(2 * np.pi * q * r) * r,
                         r0, r0 + 8 * fwhm, limit=400)[0] / den for q in qs])
for dim in (256, 512, 1024):
    m = oim.oimModel([oim.oimExpRing(d=d, fwhm=fwhm, dim=dim)])
    ccf = m.getComplexCoherentFlux(B / wl, np.zeros_like(B), np.full_like(B, wl)).real
    v_oim = ccf / m.getComplexCoherentFlux(np.array([0.0]), np.array([0.0]), np.array([wl]))[0].real
    print(f"dim={dim}: max |v_oim - v_exact| = {np.max(np.abs(v_oim - v_exact)):.4e}")
```

**Observed output**:

```text
dim=256: max |v_oim - v_exact| = 8.0136e-03
dim=512: max |v_oim - v_exact| = 8.5240e-03
dim=1024: max |v_oim - v_exact| = 8.1478e-03
```

* **Expected behaviour**: Refining the image grid `dim` from 256 to 1024 should drive `max |v_oim - v_exact|` monotonically toward zero rather than stalling near $8 \times 10^{-3}$.

### P13: `oimFastRotator` flips the visible pole's sky position angle by $180^\circ$ via `np.rot90` { #p13 }

* **Status**: `to-raise`
* **Mechanism & diagnosis**: Under the OIFITS and IAU convention ($u$ East, $v$ North, position angle measured North through East), placing the bright visible polar cap at `pa = 33.0` degrees puts the short-baseline photocentre in the North-East quadrant ($\Delta\text{RA} > 0$, $\Delta\text{Dec} > 0$). Inside `oimodeler.oimCustomComponents.fastRotator`, the polar cap for positive inclination is constructed in the upper half of the array and then rotated by `np.rot90(im)` before `oimComponentImage` maps row 0 to South, landing the bright pole in the South-West quadrant near $\text{PA} + 180^\circ$ (`220.9°` on a `dim = 64` grid at $B = 15\text{ m}$). Every short-baseline visibility phase and closure phase consequently carries the opposite sign from the requested IAU position angle.
* **Pinning test**: `tests/test_rotator_convention.py::test_p13_oimodeler_fast_rotator_pole_pa_inversion`

**Standalone MWE**:

```python
import numpy as np
import oimodeler as oim
from oimodeler.oimCustomComponents import oimFastRotator

wl, B = 1.65e-6, 15.0
mas = np.deg2rad(1.0 / 3.6e6)
fr = oimFastRotator(dim=64, incl=60.0, rot=0.95, Tpole=8500.0, dpole=2.5, pa=33.0, beta=0.25)
ccf = oim.oimModel([fr]).getComplexCoherentFlux(
    np.array([B / wl, 0.0]), np.array([0.0, B / wl]), np.array([wl, wl])
)
dra_mas = -np.angle(ccf[0]) / (2 * np.pi * (B / wl) * mas)
ddec_mas = -np.angle(ccf[1]) / (2 * np.pi * (B / wl) * mas)
pa_photocentre = np.rad2deg(np.arctan2(dra_mas, ddec_mas)) % 360.0
print(f"input pa = 33.0 deg -> photocentre PA = {pa_photocentre:.1f} deg (flipped by 180 deg)")
```

**Observed output**:

```text
input pa = 33.0 deg -> photocentre PA = 220.9 deg (flipped by 180 deg)
```

* **Expected behaviour**: For `pa = 33.0` degrees, the photocentre position angle of the hot visible polar cap should lie near `33.0` degrees (North-East) rather than `~213–221` degrees (South-West).

### P14: `oimFastRotator` replaces quadratic Roche gravity darkening with an unphysical `rot**0.4` power law { #p14 }

* **Status**: `to-raise`
* **Mechanism & diagnosis**: Under Roche equipotential geometry with either [von Zeipel's theorem (1924)](https://doi.org/10.1093/mnras/84.9.665) (emergent radiative flux proportional to local effective surface gravity $g_{\text{eff}}^{\,4\beta}$) or [Espinosa Lara & Rieutord (2011)](https://doi.org/10.1051/0004-6361/201117252) radiative-flux conservation, the centrifugal perturbation to both equatorial radius and polar-to-equatorial brightness scales quadratically as $\mathcal{O}(\omega^2)$ in the slow-rotation limit ($\omega \ll 1$). Consequently, the equator-on (`incl = 90°`, `pa = 0°`) squared-visibility anisotropy $|V^2_{\text{E}} - V^2_{\text{N}}|$ between East–West and North–South baselines must scale with log–log slope $2.0$ as rotation approaches zero. Instead, `fastRotator` hardcodes `a = 2./3 * (rot)**0.4` and `K = np.sin(1./3) * np.pi`, yielding an empirical log–log slope of $0.45$ across `rot = [0.01, 0.02, 0.04]` and over-predicting slow-rotator gravity darkening by more than two orders of magnitude.
* **Pinning test**: `tests/test_rotator_convention.py::test_p14_oimodeler_fast_rotator_gravity_darkening_scaling`

**Standalone MWE**:

```python
import numpy as np
import oimodeler as oim
from oimodeler.oimCustomComponents import oimFastRotator

wl, B = 2.2e-6, 25.0
rots = np.array([0.01, 0.02, 0.04])
aniso = []
for rot in rots:
    fr = oimFastRotator(dim=64, incl=90.0, rot=float(rot), Tpole=10000.0, dpole=3.0, pa=0.0, beta=0.25)
    ccf = oim.oimModel([fr]).getComplexCoherentFlux(
        np.array([0.0, B / wl, 0.0]), np.array([0.0, 0.0, B / wl]), np.full(3, wl)
    )
    v2 = np.abs(ccf / ccf[0]) ** 2
    aniso.append(abs(float(v2[1] - v2[2])))
slope = np.polyfit(np.log(rots), np.log(aniso), 1)[0]
print("equator-on |V2_E - V2_N| at rot=[0.01, 0.02, 0.04]:", np.round(aniso, 6))
print(f"log-log slope = {slope:.2f} (expected 2.00 from O(omega^2) centrifugal gravity)")
```

**Observed output**:

```text
equator-on |V2_E - V2_N| at rot=[0.01, 0.02, 0.04]: [0.000562 0.000765 0.001055]
log-log slope = 0.45 (expected 2.00 from O(omega^2) centrifugal gravity)
```

* **Expected behaviour**: In the slow-rotation regime (`rot` $\ll 1$), equator-on visibility anisotropy and polar-to-equatorial flux contrast should vanish quadratically with log–log slope $2.00$.

---

## vis-r

* **Upstream repository**: [`https://github.com/drgmk/vis-r`](https://github.com/drgmk/vis-r) (see [PolarDisk](../models/polar_disk.md))
* **Tested environment**: `vis-r==0.0.1` (commit `f53467077b04878986c967b501d88c54ff3efb86`), Python `3.12.12`, NumPy `2.5.4`, SciPy `1.18.1`

### P10: `r_prof_erf2_power_ggap` swaps its gap depth and gap radius parameter indices { #p10 }

* **Status**: `to-raise`
* **Mechanism & diagnosis**: Both `vis_r.functions.r_prof_erf2_power_ggap` (one Gaussian gap) and `r_prof_erf2_power_ggap2` (two Gaussian gaps) document gap parameters ordered as `[..., depth, r_gap, w_gap]` after the five base power-law ring parameters `[r_in, alpha, w_in, r_out, w_out]`. Inside `r_prof_erf2_power_ggap`, the indices for gap depth and gap centre radius are transposed: passing `depth = 0.9` and `r_gap = 0.5` carves a $50\%$-deep gap at $r = 0.9$ instead of a $90\%$-deep gap at $r = 0.5$, leaving the intensity at $r = 0.5$ at `0.9969` rather than `0.1000` (which `r_prof_erf2_power_ggap2` with a zero-depth second gap returns as documented).
* **Pinning test**: `tests/test_polar_disk_vs_codes.py::test_vis_r_single_gap_profile_matches_its_two_gap_form`

**Standalone MWE**:

```python
import numpy as np
from vis_r import functions as vf

r = np.array([0.5])  # evaluate directly at the gap centre r = 0.5
base = [0.2, 0.0, 0.05, 0.8, 0.05]  # [r_in, alpha, w_in, r_out, w_out]
one_gap = vf.r_prof_erf2_power_ggap(r, base + [0.9, 0.5, 0.05])[0]              # depth=0.9, r_gap=0.5, w_gap=0.05
two_gap = vf.r_prof_erf2_power_ggap2(r, base + [0.9, 0.5, 0.05, 0.0, 0.9, 0.05])[0]  # second gap depth=0.0
print(f"at r=0.5: r_prof_erf2_power_ggap = {one_gap:.4f}, r_prof_erf2_power_ggap2 (d2=0) = {two_gap:.4f}")
```

**Observed output**:

```text
at r=0.5: r_prof_erf2_power_ggap = 0.9969, r_prof_erf2_power_ggap2 (d2=0) = 0.1000
```

* **Expected behaviour**: `r_prof_erf2_power_ggap(r, base + [0.9, 0.5, 0.05])` should match `r_prof_erf2_power_ggap2(r, base + [0.9, 0.5, 0.05, 0.0, 0.9, 0.05])` and evaluate to `0.1000` at $r = 0.5$.

---

## MPoL

* **Upstream repository**: [`https://github.com/MPoL-dev/MPoL`](https://github.com/MPoL-dev/MPoL) (see [ALMA measurement-set conventions](../alma_conventions.md))
* **Tested environment**: `mpol==0.3.1`, PyTorch `2.14.1`, `torchkbnufft==1.5.2`, Python `3.12.12`, NumPy `2.5.3`, SciPy `1.18.1`

### P11: `mpol.losses.log_likelihood` docstring writes $\sum_i w_i$ instead of $\sum_i \ln w_i$ for the Gaussian normalization { #p11 }

* **Status**: `to-raise`
* **Mechanism & diagnosis**: For $N$ complex visibilities with inverse-variance weights $w_i = 1 / \sigma_i^2$ on both real and imaginary parts, the normalized circular complex Gaussian log-likelihood is $-N \ln(2\pi) + \sum_{i=1}^N \ln w_i - \frac{1}{2}\sum_{i=1}^N w_i |V_{\text{data},i} - V_{\text{model},i}|^2$. The implementation in `mpol.losses.log_likelihood` evaluates this exact expression (agreeing to `0.0` in `float64`), but its Sphinx docstring omits `\ln` in the weight normalization term and writes `- N \ln 2 \pi + \sum_i^N w_i - ...`.
* **Pinning test**: `tests/test_alma_mpol.py::test_mpol_log_likelihood_equals_virgil_with_s_equal_g_times_s_mpol`

**Standalone MWE**:

```python
import inspect, torch
from mpol import losses

torch.set_default_dtype(torch.float64)
model = torch.tensor([1.0 + 0.5j, -0.2 + 0.3j])
data = torch.tensor([1.1 + 0.4j, -0.1 + 0.2j])
weight = torch.tensor([100.0, 400.0])

ll_code = float(losses.log_likelihood(model, data, weight))
ll_exact_ln_w = float(-len(data) * torch.log(torch.tensor(2 * torch.pi)) + torch.sum(torch.log(weight))
                      - 0.5 * torch.sum(weight * torch.abs(data - model) ** 2))
print("code vs (-N ln 2pi + sum ln w - chi2/2):", abs(ll_code - ll_exact_ln_w))
for line in inspect.getdoc(losses.log_likelihood).splitlines():
    if "sum" in line or "w_i" in line:
        print("docstring:", line)
```

**Observed output**:

```text
code vs (-N ln 2pi + sum ln w - chi2/2): 0.0
docstring:     - N \ln 2 \pi +  \sum_i^N w_i -
docstring: This function is agnostic as to whether the sum should include the Hermitian
docstring:     array of weight values representing :math:`w_i`
docstring:     the :math:`\ln\mathcal{L}` log likelihood, summed over all dimensions
```

* **Expected behaviour**: The docstring equation in `mpol.losses.log_likelihood` should read `\sum_i^N \ln w_i` to match the implementation.

---

## Documented precision notes (not bugs)

### P12: Default `NuFFT` table interpolation ($6.3 \times 10^{-4}$ relative $L_2$) and `float32` `GaussConvFourier` taper buffer { #p12 }

* **Status**: `documented` (recorded for cross-code tolerance accounting; not a defect unless `float64` buffer preservation is desired upstream)
* **Mechanism & diagnosis**: By default, `mpol.fourier.NuFFT` uses `torchkbnufft` Kaiser–Bessel table interpolation (`sparse_matrices=False`), which differs from an exact `float64` discrete Fourier transform by $6.3 \times 10^{-4}$ in relative $L_2$ norm on a $256 \times 256$ image ($1.6 \times 10^{-6}$ when `sparse_matrices=True` is passed). In addition, even after calling `torch.set_default_dtype(torch.float64)`, `mpol.images.GaussConvFourier` constructs its meshgrid via explicit `dtype=torch.float32` tensors and registers `taper_2D` in single precision, introducing a $\sim 3.5 \times 10^{-8}$ floor when comparing `float64` forward models.
* **Pinning tests**: `tests/test_alma_mpol.py::test_mpol_virgil_and_the_exact_dft_agree_on_the_forward_model`, `tests/test_alma_mpol.py::test_mpol_with_its_accurate_nufft_agrees_to_1e_5`

**Standalone MWE**:

```python
import torch
from mpol import coordinates
from mpol.images import GaussConvFourier

torch.set_default_dtype(torch.float64)
coords = coordinates.GridCoords(cell_size=0.005, npix=64)
gcf = GaussConvFourier(coords=coords, FWHM_maj=0.03, FWHM_min=0.03)
print("default dtype:", torch.get_default_dtype(), "GaussConvFourier taper_2D dtype:", gcf.taper_2D.dtype)
```

**Observed output**:

```text
default dtype: torch.float64 GaussConvFourier taper_2D dtype: torch.float32
```

* **Expected behaviour**: Documented design trade-off for GPU throughput (`sparse_matrices=True` recovers $1.6 \times 10^{-6}$ NuFFT precision; respecting `torch.get_default_dtype()` in `GaussConvFourier` would remove the remaining $3.5 \times 10^{-8}$ single-precision floor).
