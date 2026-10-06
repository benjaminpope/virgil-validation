# Detection campaign, version 2

Version 1 checked virgil's detection statistics on companions with
noncentralities λ = 146–2582. Near the detection threshold (λ of a few to a
few tens) its noncentral test said nothing, and it excluded λ < 9, where the
flux constraint matters. Its two-sample KS tests compare the bulk of the
null distributions, where false-alarm thresholds do not live. Version 1's
output is kept but not credited.

Version 2:

* **Injections** at fluxes 3e-4 to 3e-3 (λ ≈ 1–80 over 5–15 mas), so that
  most draws lie near the threshold.
* **Reference at the true position:** with the flux constrained to be
  non-negative, Δχ² = max(0, √λ + Z)² for standard normal Z (linear regime).
  Its CDF is Φ(√t − √λ) for t > 0, with an atom Φ(−√λ) at 0. It is tested
  for every λ by the randomized PIT and by the standardized mean.
* **Null at a fixed position:** Chernoff's mixture, tested on the zero
  fraction, the positive part and the mean (1/2, variance 5/4).
* **Tails:** exceedance counts above virgil's empirical 1e-2 threshold,
  ours against virgil's (Fisher's exact test), and the k-sample
  Anderson–Darling test, which weights the tails more than KS.
* **Counts:** 2×10⁴ of our null and injected draws (CPU), and 10⁵ of
  virgil's null simulations (GPU).
* **Precision:** the summary states the 95% interval on the ratio of
  exceedance rates at 1e-2 and 1e-3. That interval is the false-alarm
  precision the campaign validates, and nothing finer.
* **Multiplicity:** one Holm family at 0.01 over every p-value.
