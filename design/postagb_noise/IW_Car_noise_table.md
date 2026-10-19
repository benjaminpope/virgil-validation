### IW_Car

| strategy | noise / mean | hyperparameters (edge?) | χ²/pt V² | χ²/pt T3 | common-mode V² (white = 1) | adjacent corr | NCC (best control) | ΔlnZ | f_env % | f_bg % | bg index |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mem | current / blob | – | 1.50 | 1.35 | 4.09 | 0.78 | 0.89 (0.51) | – | 34.1 | 0.0 | 5.45 |
| mem | current / flat | – | 1.50 | 1.35 | 4.09 | 0.78 | 0.89 (0.51) | – | 34.1 | 0.0 | 5.45 |
| mem | vlti / blob | – | 1.52 | 0.51 | 5.11 | 0.85 | 0.65 (0.34) | – | 34.2 | 0.0 | -6.00 (bound) |
| mem | vlti / flat | – | 1.52 | 0.51 | 5.11 | 0.85 | 0.65 (0.34) | – | 34.2 | 0.0 | -6.00 (bound) |
| gp-iso | current / blob | sigma 2, length_mas 1.21 | 0.85 | 0.66 | 4.00 | 0.74 | 0.89 (0.55) | 0 | 29.8 | 4.4 | -6.00 (bound) |
| gp-iso | current / flat | sigma 2, length_mas 2.42 | 0.98 | 0.78 | 3.53 | 0.69 | 0.90 (0.56) | 0 | 34.3 | 0.0 | -4.04 |
| gp-iso | vlti / blob | sigma 2, length_mas 1.21 | 1.80 | 0.68 | 5.17 | 0.86 | 0.86 (0.47) | -386 | 34.1 | 0.0 | -6.00 (bound) |
| gp-iso | vlti / flat | sigma 4, length_mas 0.303 **edge** | 1.63 | 0.55 | 5.43 | 0.90 | 0.69 (0.42) | 0 | 30.6 | 5.1 | -6.00 (bound) |
| gp-polar | current / blob | sigma 2, length_r_mas 0.605, length_theta 0.25 **edge** | 1.00 | 0.77 | 4.12 | 0.79 | 0.91 (0.58) | -19 | 27.4 | 6.4 | -6.00 (bound) |
| gp-polar | current / flat | sigma 2, length_r_mas 0.605, length_theta 0.25 **edge** | 1.27 | 1.01 | 3.94 | 0.77 | 0.89 (0.56) | -155 | 34.0 | 0.0 | -5.22 |
| gp-polar | vlti / blob | sigma 2, length_r_mas 0.605, length_theta 0.25 **edge** | 2.02 | 0.70 | 5.43 | 0.91 | 0.88 (0.54) | 0 | 27.1 | 6.3 | -6.00 (bound) |
| gp-polar | vlti / flat | sigma 4, length_r_mas 0.605, length_theta 0.0625 **edge** | 3.15 | 1.00 | 5.45 | 0.92 | 0.42 (0.27) | -648 | 30.7 | 7.8 | -6.00 (bound) |
