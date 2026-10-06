# Visibility-only data

virgil#158 lets virgil read and fit data with no phases
(`tests/test_v2_only.py`). Our simulator writes V² alone (no OI_T3).

| Check | Reference | Tag | Agreement |
| --- | --- | --- | --- |
| a V²-only file read by `OIData`: no phase block, model equals what we wrote | our writer | Standards | 1e-12 |
| uniform-disk diameter fitted from noise-free V² alone | injected truth | Mathematics | 1e-6 relative |
| the same file fitted by PMOIRED (`ud`, V² only) | PMOIRED | PMOIRED | best fits 2e-4 σ apart; σ equal to 4e-4 (no closure phases, so no difference of definition) |
| diameter pulls over 200 noisy V²-only files | N(0, 1) | Statistics | mean 0.03, sd 1.01 |
| every closure phase flagged | the same V² without OI_T3 | Standards | warns, then exactly the V²-only likelihood (finding 8, superseded by virgil#158) |

