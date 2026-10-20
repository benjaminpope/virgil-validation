### IW_Car

| strategy | noise / mean | hyperparameters (edge?) | χ²/pt V² | χ²/pt T3 | common-mode V² (white = 1) | adjacent corr | after MAP gains: χ²/pt (own) · common-mode · adjacent | NCC (best control) | ΔlnZ | f_env % | f_bg % | bg index |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mem | current / blob | – | 1.50 | 1.35 | 4.09 | 0.78 | 1.50 (–) · 4.09 · 0.78 | 0.89 (0.51) | – | 34.1 | 0.0 | -5.99 (bound) |
| mem | current / flat | – | 1.50 | 1.35 | 4.09 | 0.78 | 1.50 (–) · 4.09 · 0.78 | 0.89 (0.51) | – | 34.1 | 0.0 | -5.99 (bound) |
| mem | vlti / blob | – | 1.52 | 0.51 | 5.11 | 0.85 | 0.21 (2.32) · 0.08 · -0.06 | 0.65 (0.35) | – | 34.2 | 0.0 | -6.00 (bound) |
| mem | vlti / flat | – | 1.52 | 0.51 | 5.11 | 0.85 | 0.21 (2.32) · 0.08 · -0.06 | 0.65 (0.35) | – | 34.2 | 0.0 | -6.00 (bound) |
| gp-iso | current / blob | sigma 2, length_mas 1.21 | 0.85 | 0.66 | 4.00 | 0.74 | 0.85 (–) · 4.00 · 0.74 | 0.89 (0.55) | 0 | 29.8 | 4.4 | -6.00 (bound) |
| gp-iso | current / flat | sigma 2, length_mas 2.42 | 0.98 | 0.78 | 3.53 | 0.69 | 0.98 (–) · 3.53 · 0.69 | 0.90 (0.56) | 0 | 34.3 | 0.0 | -4.03 |
| gp-iso | vlti / blob | sigma 2, length_mas 1.21 | 1.80 | 0.68 | 5.17 | 0.86 | 0.23 (2.57) · 0.08 · -0.05 | 0.86 (0.47) | -436 | 34.1 | 0.0 | -6.00 (bound) |
| gp-iso | vlti / flat | sigma 4, length_mas 0.303 **edge** | 1.63 | 0.55 | 5.43 | 0.90 | 0.15 (1.83) · 0.15 · -0.11 | 0.69 (0.42) | -61 | 30.6 | 5.1 | -6.00 (bound) |
| gp-aniso | current / blob | sigma 2, length_mas 0.428 | 0.86 | 0.67 | 4.00 | 0.74 | 0.86 (–) · 4.00 · 0.74 | 0.88 (0.56) | -12 | 29.7 | 4.5 | -6.00 (bound) |
| gp-aniso | current / flat | sigma 2, length_mas 0.605 | 0.95 | 0.72 | 3.53 | 0.68 | 0.95 (–) · 3.53 · 0.68 | 0.87 (0.53) | -23 | 34.5 | 0.0 | -4.83 |
| gp-aniso | vlti / blob | sigma 2, length_mas 0.428 | 1.87 | 0.61 | 5.47 | 0.91 | 0.15 (1.85) · 0.14 · -0.12 | 0.88 (0.51) | 0 | 27.4 | 6.2 | -6.00 (bound) |
| gp-aniso | vlti / flat | sigma 4, length_mas 0.428 | 1.66 | 0.55 | 5.43 | 0.90 | 0.15 (1.83) · 0.14 · -0.12 | 0.75 (0.47) | -86 | 30.1 | 4.9 | -6.00 (bound) |
| gp-polar | current / blob | sigma 2, length_r_mas 0.605, length_theta 0.25 **edge** | 1.00 | 0.77 | 4.12 | 0.79 | 1.00 (–) · 4.12 · 0.79 | 0.91 (0.58) | -19 | 27.4 | 6.4 | -6.00 (bound) |
| gp-polar | current / flat | sigma 2, length_r_mas 0.605, length_theta 0.25 **edge** | 1.27 | 1.01 | 3.94 | 0.77 | 1.27 (–) · 3.94 · 0.77 | 0.89 (0.56) | -155 | 34.0 | 0.0 | -5.29 |
| gp-polar | vlti / blob | sigma 2, length_r_mas 0.605, length_theta 0.25 **edge** | 2.01 | 0.72 | 5.43 | 0.91 | 0.17 (2.06) · 0.12 · -0.02 | 0.88 (0.55) | -56 | 27.1 | 6.3 | -6.00 (bound) |
| gp-polar | vlti / flat | sigma 2, length_r_mas 0.605, length_theta 0.5 **edge** | 2.13 | 0.79 | 5.44 | 0.91 | 0.17 (2.12) · 0.12 · 0.00 | 0.89 (0.55) | 0 | 27.0 | 6.3 | -6.00 (bound) |

### IRAS08544

| strategy | noise / mean | hyperparameters (edge?) | χ²/pt V² | χ²/pt T3 | common-mode V² (white = 1) | adjacent corr | after MAP gains: χ²/pt (own) · common-mode · adjacent | NCC (best control) | ΔlnZ | f_env % | f_bg % | bg index |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mem | current / blob | – | 0.72 | 0.46 | 3.70 | 0.66 | 0.72 (–) · 3.70 · 0.66 | 0.96 (0.85) | – | 37.0 | – | – |
| mem | current / flat | – | 0.72 | 0.46 | 3.70 | 0.66 | 0.72 (–) · 3.70 · 0.66 | 0.96 (0.85) | – | 37.0 | – | – |
| mem | vlti / blob | – | 1.20 | 0.57 | 4.85 | 0.80 | 0.22 (1.11) · 0.19 · -0.02 | 0.96 (0.86) | – | 37.5 | – | – |
| mem | vlti / flat | – | 1.20 | 0.57 | 4.85 | 0.80 | 0.22 (1.11) · 0.19 · -0.02 | 0.96 (0.86) | – | 37.5 | – | – |
| gp-iso | current / blob | sigma 2, length_mas 4.9 **edge** | 0.73 | 0.52 | 3.83 | 0.67 | 0.73 (–) · 3.83 · 0.67 | 0.95 (0.89) | 0 | 36.2 | – | – |
| gp-iso | current / flat | sigma 2, length_mas 2.45 | 0.65 | 0.45 | 3.64 | 0.64 | 0.65 (–) · 3.64 · 0.64 | 0.96 (0.88) | 0 | 36.3 | – | – |
| gp-iso | vlti / blob | sigma 1, length_mas 1.22 | 1.09 | 0.52 | 4.81 | 0.80 | 0.21 (1.06) · 0.24 · -0.04 | 0.92 (0.85) | -26 | 38.0 | – | – |
| gp-iso | vlti / flat | sigma 2, length_mas 2.45 | 0.99 | 0.48 | 4.70 | 0.77 | 0.21 (1.04) · 0.21 · -0.05 | 0.96 (0.90) | 0 | 36.7 | – | – |
| gp-aniso | current / blob | sigma 2, length_mas 0.612 | 0.62 | 0.43 | 3.61 | 0.63 | 0.62 (–) · 3.61 · 0.63 | 0.95 (0.87) | -5 | 36.5 | – | – |
| gp-aniso | current / flat | sigma 2, length_mas 0.612 | 0.61 | 0.43 | 3.56 | 0.62 | 0.61 (–) · 3.56 · 0.62 | 0.95 (0.87) | -17 | 36.5 | – | – |
| gp-aniso | vlti / blob | sigma 2, length_mas 2.45 | 1.11 | 0.64 | 4.83 | 0.80 | 0.21 (1.04) · 0.22 · -0.04 | 0.95 (0.88) | 0 | 36.7 | – | – |
| gp-aniso | vlti / flat | sigma 0.25, length_mas 0.15 **edge** | 6.13 | 2.09 | 5.22 | 0.94 | 0.75 (4.56) · 0.36 · 0.50 | 0.86 (0.69) | -3386 | 37.8 | – | – |
| gp-polar | current / blob | sigma 2, length_r_mas 0.612, length_theta 0.5 **edge** | 1.15 | 0.59 | 3.90 | 0.77 | 1.15 (–) · 3.90 · 0.77 | 0.93 (0.89) | -266 | 36.3 | – | – |
| gp-polar | current / flat | sigma 2, length_r_mas 0.612, length_theta 0.5 **edge** | 1.15 | 0.59 | 3.91 | 0.77 | 1.15 (–) · 3.91 · 0.77 | 0.93 (0.89) | -276 | 36.3 | – | – |
| gp-polar | vlti / blob | sigma 2, length_r_mas 0.612, length_theta 2 **edge** | 2.08 | 0.83 | 5.17 | 0.88 | 0.27 (1.52) · 0.30 · 0.14 | 0.94 (0.91) | -118 | 36.8 | – | – |
| gp-polar | vlti / flat | sigma 4, length_r_mas 0.612, length_theta 2 **edge** | 2.07 | 0.70 | 5.17 | 0.88 | 0.28 (1.53) · 0.31 · 0.15 | 0.84 (0.79) | -125 | 38.4 | – | – |

### HR4049

| strategy | noise / mean | hyperparameters (edge?) | χ²/pt V² | χ²/pt T3 | common-mode V² (white = 1) | adjacent corr | after MAP gains: χ²/pt (own) · common-mode · adjacent | NCC (best control) | ΔlnZ | f_env % | f_bg % | bg index |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mem | current / blob | – | 1.65 | 0.50 | 5.13 | 0.93 | 1.65 (–) · 5.13 · 0.93 | 0.86 (0.64) | – | 34.3 | 0.0 | 4.39 |
| mem | current / flat | – | 1.65 | 0.50 | 5.13 | 0.93 | 1.65 (–) · 5.13 · 0.93 | 0.86 (0.64) | – | 34.3 | 0.0 | 4.39 |
| mem | vlti / blob | – | 5.52 | 0.58 | 5.86 | 0.95 | 0.08 (1.78) · 0.12 · 0.00 | 0.88 (0.71) | – | 36.2 | 0.0 | -6.00 (bound) |
| mem | vlti / flat | – | 5.52 | 0.58 | 5.86 | 0.95 | 0.08 (1.78) · 0.12 · 0.00 | 0.88 (0.71) | – | 36.2 | 0.0 | -6.00 (bound) |
| gp-iso | current / blob | sigma 2, length_mas 5.55 | 0.93 | 0.44 | 5.20 | 0.91 | 0.93 (–) · 5.20 · 0.91 | 0.89 (0.81) | -2 | 33.7 | 0.0 | 5.73 |
| gp-iso | current / flat | sigma 2, length_mas 5.55 | 0.89 | 0.45 | 5.19 | 0.91 | 0.89 (–) · 5.19 · 0.91 | 0.88 (0.78) | 0 | 34.4 | 0.0 | 6.00 (bound) |
| gp-iso | vlti / blob | sigma 2, length_mas 2.77 | 1.60 | 0.42 | 5.75 | 0.96 | 0.06 (1.17) · 0.10 · -0.16 | 0.87 (0.81) | 0 | 33.7 | 0.0 | 6.00 (bound) |
| gp-iso | vlti / flat | sigma 4, length_mas 0.347 **edge** | 2.64 | 0.36 | 5.85 | 0.97 | 0.05 (1.09) · 0.09 · -0.21 | 0.11 (0.10) | 0 | 40.2 | 5.4 | -3.39 |
| gp-aniso | current / blob | sigma 1, length_mas 0.694 | 0.95 | 0.44 | 5.20 | 0.91 | 0.95 (–) · 5.20 · 0.91 | 0.86 (0.80) | 0 | 33.8 | 0.0 | 5.55 |
| gp-aniso | current / flat | sigma 2, length_mas 1.39 | 0.84 | 0.42 | 5.19 | 0.90 | 0.84 (–) · 5.19 · 0.90 | 0.81 (0.70) | -8 | 34.5 | 0.0 | 6.00 (bound) |
| gp-aniso | vlti / blob | sigma 2, length_mas 1.39 | 1.52 | 0.44 | 5.74 | 0.95 | 0.06 (1.21) · 0.10 · -0.13 | 0.85 (0.79) | -20 | 33.7 | 0.0 | 5.77 |
| gp-aniso | vlti / flat | sigma 4, length_mas 0.347 **edge** | 2.65 | 0.38 | 5.85 | 0.97 | 0.05 (1.07) · 0.10 · -0.22 | 0.11 (0.12) | -30 | 40.2 | 5.3 | -3.64 |
| gp-polar | current / blob | sigma 1, length_r_mas 2.77, length_theta 0.125 | 1.01 | 0.46 | 5.10 | 0.92 | 1.01 (–) · 5.10 · 0.92 | 0.85 (0.79) | -18 | 33.5 | 0.5 | 5.92 (bound) |
| gp-polar | current / flat | sigma 0.125, length_r_mas 0.694, length_theta 0.125 **edge** | 2.85 | 1.24 | 5.19 | 0.95 | 2.85 (–) · 5.19 · 0.95 | 0.78 (0.63) | -2194 | 34.6 | 0.0 | -6.00 (bound) |
| gp-polar | vlti / blob | sigma 2, length_r_mas 0.694, length_theta 0.125 **edge** | 2.75 | 0.46 | 5.76 | 0.97 | 0.07 (1.50) · 0.08 · -0.02 | 0.81 (0.76) | -121 | 33.6 | 0.0 | 4.91 |
| gp-polar | vlti / flat | sigma 2, length_r_mas 0.694, length_theta 0.25 **edge** | 2.66 | 0.47 | 5.77 | 0.97 | 0.07 (1.45) · 0.10 · -0.04 | 0.78 (0.72) | -67 | 33.7 | 0.0 | 5.98 (bound) |

### HD108015

| strategy | noise / mean | hyperparameters (edge?) | χ²/pt V² | χ²/pt T3 | common-mode V² (white = 1) | adjacent corr | after MAP gains: χ²/pt (own) · common-mode · adjacent | NCC (best control) | ΔlnZ | f_env % | f_bg % | bg index |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mem | current / blob | – | 1.27 | 0.83 | 3.10 | 0.57 | 1.27 (–) · 3.10 · 0.57 | 0.98 (0.83) | – | 43.2 | 0.0 | -5.98 (bound) |
| mem | current / flat | – | 1.27 | 0.83 | 3.10 | 0.57 | 1.27 (–) · 3.10 · 0.57 | 0.98 (0.83) | – | 43.2 | 0.0 | -5.98 (bound) |
| mem | vlti / blob | – | 2.15 | 0.76 | 4.36 | 0.71 | 0.55 (1.73) · 0.28 · 0.09 | 0.98 (0.82) | – | 43.5 | 0.0 | -5.91 (bound) |
| mem | vlti / flat | – | 2.15 | 0.76 | 4.36 | 0.71 | 0.55 (1.73) · 0.28 · 0.09 | 0.98 (0.82) | – | 43.5 | 0.0 | -5.91 (bound) |
| gp-iso | current / blob | sigma 2, length_mas 1.18 | 0.74 | 0.68 | 2.76 | 0.38 | 0.74 (–) · 2.76 · 0.38 | 0.98 (0.82) | 0 | 40.8 | 2.4 | -6.00 (bound) |
| gp-iso | current / flat | sigma 2, length_mas 1.18 | 0.80 | 0.68 | 2.42 | 0.39 | 0.80 (–) · 2.42 · 0.39 | 0.97 (0.82) | -46 | 43.2 | 0.0 | -5.37 |
| gp-iso | vlti / blob | sigma 2, length_mas 1.18 | 1.11 | 0.73 | 3.73 | 0.56 | 0.42 (1.25) · 0.30 · -0.10 | 0.98 (0.82) | -110 | 43.1 | 0.0 | -5.99 (bound) |
| gp-iso | vlti / flat | sigma 0.25, length_mas 0.295 **edge** | 5.50 | 2.16 | 4.80 | 0.87 | 1.05 (3.84) · 0.89 · 0.43 | 0.85 (0.66) | -4775 | 42.4 | 0.0 | -6.00 (bound) |
| gp-aniso | current / blob | sigma 2, length_mas 0.418 | 0.74 | 0.66 | 2.77 | 0.38 | 0.74 (–) · 2.77 · 0.38 | 0.98 (0.82) | -15 | 40.9 | 2.4 | -6.00 (bound) |
| gp-aniso | current / flat | sigma 8, length_mas 2.36 | 0.73 | 0.66 | 2.70 | 0.37 | 0.73 (–) · 2.70 · 0.37 | 0.98 (0.83) | 0 | 40.8 | 2.4 | -6.00 (bound) |
| gp-aniso | vlti / blob | sigma 2, length_mas 0.295 **edge** | 0.93 | 0.69 | 3.59 | 0.52 | 0.37 (1.11) · 0.30 · -0.15 | 0.97 (0.81) | 0 | 40.0 | 3.2 | -6.00 (bound) |
| gp-aniso | vlti / flat | sigma 0.25, length_mas 0.295 **edge** | 15.82 | 3.14 | 5.34 | 0.91 | 1.26 (5.85) · 1.16 · 0.36 | 0.80 (0.68) | -4945 | 29.9 | 13.2 | -6.00 (bound) |
| gp-polar | current / blob | sigma 2, length_r_mas 0.591, length_theta 0.5 **edge** | 0.81 | 0.72 | 2.93 | 0.43 | 0.81 (–) · 2.93 · 0.43 | 0.97 (0.81) | -20 | 39.9 | 3.5 | -5.69 |
| gp-polar | current / flat | sigma 2, length_r_mas 0.591, length_theta 0.5 **edge** | 0.97 | 0.80 | 2.84 | 0.49 | 0.97 (–) · 2.84 · 0.49 | 0.97 (0.81) | -152 | 43.3 | 0.0 | -4.58 |
| gp-polar | vlti / blob | sigma 2, length_r_mas 0.591, length_theta 0.5 **edge** | 1.46 | 0.83 | 4.18 | 0.63 | 0.43 (1.32) · 0.31 · -0.08 | 0.97 (0.82) | -139 | 43.1 | 0.0 | -6.00 (bound) |
| gp-polar | vlti / flat | sigma 4, length_r_mas 0.591, length_theta 0.25 **edge** | 6.24 | 1.53 | 5.13 | 0.87 | 0.84 (3.15) · 0.89 · 0.35 | 0.58 (0.45) | 0 | 41.5 | 9.1 | -6.00 (bound) |

### AI_Sco

| strategy | noise / mean | hyperparameters (edge?) | χ²/pt V² | χ²/pt T3 | common-mode V² (white = 1) | adjacent corr | after MAP gains: χ²/pt (own) · common-mode · adjacent | NCC (best control) | ΔlnZ | f_env % | f_bg % | bg index |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mem | current / blob | – | 3.01 | 1.77 | 4.22 | 0.77 | 3.01 (–) · 4.22 · 0.77 | 0.75 (0.51) | – | 22.7 | 0.0 | -5.65 |
| mem | current / flat | – | 3.01 | 1.77 | 4.22 | 0.77 | 3.01 (–) · 4.22 · 0.77 | 0.75 (0.51) | – | 22.7 | 0.0 | -5.65 |
| mem | vlti / blob | – | 6.08 | 0.89 | 5.39 | 0.90 | 0.47 (2.26) · 0.29 · 0.07 | 0.73 (0.47) | – | 30.2 | 0.0 | 5.95 (bound) |
| mem | vlti / flat | – | 6.08 | 0.89 | 5.39 | 0.90 | 0.47 (2.26) · 0.29 · 0.07 | 0.73 (0.47) | – | 30.2 | 0.0 | 5.95 (bound) |
| gp-iso | current / blob | sigma 8, length_mas 0.375 | 1.76 | 0.95 | 3.50 | 0.69 | 1.76 (–) · 3.50 · 0.69 | 0.78 (0.49) | 0 | 27.6 | 0.0 | 6.00 (bound) |
| gp-iso | current / flat | sigma 16, length_mas 0.612 | 3.71 | 4.03 | 3.70 | 0.84 | 3.71 (–) · 3.70 · 0.84 | 0.51 (0.37) | -693 | 14.7 | 5.8 | -6.00 (bound) |
| gp-iso | vlti / blob | sigma 8, length_mas 0.375 | 5.63 | 0.96 | 5.34 | 0.89 | 0.48 (2.29) · 0.30 · 0.09 | 0.73 (0.47) | 0 | 29.3 | 0.7 | 6.00 (bound) |
| gp-iso | vlti / flat | sigma 8, length_mas 0.23 **edge** | 5.62 | 0.99 | 5.34 | 0.89 | 0.48 (2.31) · 0.34 · 0.09 | 0.71 (0.46) | 0 | 29.7 | 0.5 | 6.00 (bound) |
| gp-aniso | current / blob | sigma 8, length_mas 0.23 **edge** | 1.75 | 0.94 | 3.48 | 0.69 | 1.75 (–) · 3.48 · 0.69 | 0.78 (0.48) | -8 | 26.9 | 0.0 | 6.00 (bound) |
| gp-aniso | current / flat | sigma 16, length_mas 0.375 | 3.76 | 3.79 | 3.83 | 0.85 | 3.76 (–) · 3.83 · 0.85 | 0.53 (0.41) | -681 | 15.0 | 5.7 | -6.00 (bound) |
| gp-aniso | vlti / blob | sigma 8, length_mas 0.23 **edge** | 5.70 | 0.98 | 5.35 | 0.89 | 0.48 (2.29) · 0.31 · 0.09 | 0.74 (0.49) | -11 | 28.8 | 0.7 | 6.00 (bound) |
| gp-aniso | vlti / flat | sigma 8, length_mas 0.23 **edge** | 5.82 | 0.95 | 5.36 | 0.89 | 0.48 (2.30) · 0.31 · 0.09 | 0.73 (0.48) | -23 | 30.0 | 0.4 | 6.00 (bound) |
| gp-polar | current / blob | sigma 4, length_r_mas 0.612, length_theta 0.0625 **edge** | 2.18 | 1.49 | 4.02 | 0.77 | 2.18 (–) · 4.02 · 0.77 | 0.82 (0.53) | -146 | 23.7 | 2.4 | 6.00 (bound) |
| gp-polar | current / flat | sigma 4, length_r_mas 0.612, length_theta 0.0625 **edge** | 2.41 | 1.43 | 3.94 | 0.77 | 2.41 (–) · 3.94 · 0.77 | 0.77 (0.47) | 0 | 26.0 | 0.0 | 4.37 |
| gp-polar | vlti / blob | sigma 4, length_r_mas 0.612, length_theta 0.0625 **edge** | 4.90 | 1.92 | 5.20 | 0.89 | 0.61 (2.85) · 0.41 · 0.23 | 0.77 (0.56) | -330 | 29.0 | 2.4 | 6.00 (bound) |
| gp-polar | vlti / flat | sigma 0.125, length_r_mas 0.612, length_theta 0.125 **edge** | 9.76 | 4.82 | 5.38 | 0.92 | 0.91 (4.90) · 0.89 · 0.42 | 0.71 (0.50) | -2955 | 21.7 | 0.0 | -6.00 (bound) |

### EN_TrA

| strategy | noise / mean | hyperparameters (edge?) | χ²/pt V² | χ²/pt T3 | common-mode V² (white = 1) | adjacent corr | after MAP gains: χ²/pt (own) · common-mode · adjacent | NCC (best control) | ΔlnZ | f_env % | f_bg % | bg index |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mem | current / blob | – | 2.96 | 1.26 | 4.36 | 0.78 | 2.96 (–) · 4.36 · 0.78 | 0.95 (0.76) | – | 20.9 | 0.0 | -5.61 |
| mem | current / flat | – | 2.96 | 1.26 | 4.36 | 0.78 | 2.96 (–) · 4.36 · 0.78 | 0.95 (0.76) | – | 20.9 | 0.0 | -5.61 |
| mem | vlti / blob | – | 2.55 | 0.71 | 4.69 | 0.77 | 0.49 (1.67) · 0.27 · -0.07 | 0.95 (0.75) | – | 21.3 | 0.0 | -5.96 (bound) |
| mem | vlti / flat | – | 2.55 | 0.71 | 4.69 | 0.77 | 0.49 (1.67) · 0.27 · -0.07 | 0.95 (0.75) | – | 21.3 | 0.0 | -5.96 (bound) |
| gp-iso | current / blob | sigma 8, length_mas 0.265 **edge** | 1.59 | 0.87 | 3.41 | 0.62 | 1.59 (–) · 3.41 · 0.62 | 0.87 (0.73) | 0 | 23.6 | 0.0 | -5.99 (bound) |
| gp-iso | current / flat | not run | | | | | | | | | | |
| gp-iso | vlti / blob | not run | | | | | | | | | | |
| gp-iso | vlti / flat | sigma 8, length_mas 0.265 **edge** | 2.30 | 0.71 | 4.61 | 0.75 | 0.48 (1.60) · 0.27 · -0.08 | 0.82 (0.62) | 0 | 20.7 | 0.0 | 6.00 (bound) |
| gp-aniso | current / blob | sigma 4, length_mas 0.265 **edge** | 1.64 | 0.91 | 3.48 | 0.63 | 1.64 (–) · 3.48 · 0.63 | 0.91 (0.75) | -2 | 23.5 | 0.0 | -5.99 (bound) |
| gp-aniso | current / flat | sigma 4, length_mas 0.265 **edge** | 1.62 | 0.91 | 3.44 | 0.63 | 1.62 (–) · 3.44 · 0.63 | 0.88 (0.71) | 0 | 23.1 | 0.0 | -5.97 (bound) |
| gp-aniso | vlti / blob | sigma 4, length_mas 0.265 **edge** | 2.38 | 0.76 | 4.59 | 0.76 | 0.50 (1.66) · 0.27 · -0.06 | 0.91 (0.73) | 0 | 21.8 | 0.0 | -4.28 |
| gp-aniso | vlti / flat | sigma 4, length_mas 0.265 **edge** | 2.39 | 0.74 | 4.61 | 0.76 | 0.49 (1.65) · 0.27 · -0.07 | 0.89 (0.68) | -21 | 21.1 | 0.0 | 3.82 |
| gp-polar | current / blob | sigma 4, length_r_mas 0.606, length_theta 0.0625 **edge** | 1.75 | 0.94 | 3.59 | 0.65 | 1.75 (–) · 3.59 · 0.65 | 0.92 (0.77) | -20 | 23.0 | 0.0 | -5.99 (bound) |
| gp-polar | current / flat | sigma 4, length_r_mas 0.606, length_theta 0.125 **edge** | 1.76 | 0.92 | 3.59 | 0.65 | 1.76 (–) · 3.59 · 0.65 | 0.89 (0.73) | -19 | 22.4 | 0.0 | -5.97 (bound) |
| gp-polar | vlti / blob | sigma 4, length_r_mas 0.606, length_theta 0.0625 **edge** | 2.47 | 0.80 | 4.61 | 0.76 | 0.50 (1.71) · 0.29 · -0.04 | 0.92 (0.75) | -17 | 21.6 | 0.0 | -5.55 |
| gp-polar | vlti / flat | sigma 4, length_r_mas 0.606, length_theta 0.25 **edge** | 2.48 | 0.81 | 4.60 | 0.76 | 0.51 (1.72) · 0.28 · -0.03 | 0.88 (0.69) | -44 | 20.8 | 0.0 | -6.00 (bound) |

### HD95767

| strategy | noise / mean | hyperparameters (edge?) | χ²/pt V² | χ²/pt T3 | common-mode V² (white = 1) | adjacent corr | after MAP gains: χ²/pt (own) · common-mode · adjacent | NCC (best control) | ΔlnZ | f_env % | f_bg % | bg index |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mem | current / blob | – | 1.50 | 0.80 | 3.48 | 0.62 | 1.50 (–) · 3.48 · 0.62 | 0.97 (0.91) | – | 30.8 | 0.0 | -5.90 (bound) |
| mem | current / flat | – | 1.50 | 0.80 | 3.48 | 0.62 | 1.50 (–) · 3.48 · 0.62 | 0.97 (0.91) | – | 30.8 | 0.0 | -5.90 (bound) |
| mem | vlti / blob | – | 1.78 | 0.78 | 4.35 | 0.71 | 0.44 (1.71) · 0.29 · 0.01 | 0.97 (0.91) | – | 30.0 | 1.0 | -6.00 (bound) |
| mem | vlti / flat | – | 1.78 | 0.78 | 4.35 | 0.71 | 0.44 (1.71) · 0.29 · 0.01 | 0.97 (0.91) | – | 30.0 | 1.0 | -6.00 (bound) |
| gp-iso | current / blob | sigma 4, length_mas 0.263 **edge** | 0.77 | 0.66 | 2.53 | 0.42 | 0.77 (–) · 2.53 · 0.42 | 0.91 (0.83) | 0 | 29.3 | 1.1 | -6.00 (bound) |
| gp-iso | current / flat | sigma 8, length_mas 0.263 **edge** | 0.96 | 0.99 | 2.52 | 0.47 | 0.96 (–) · 2.52 · 0.47 | 0.44 (0.45) | -169 | 42.5 | 2.4 | -6.00 (bound) |
| gp-iso | vlti / blob | sigma 4, length_mas 0.263 **edge** | 0.96 | 0.67 | 3.59 | 0.56 | 0.36 (1.27) · 0.19 · -0.11 | 0.89 (0.81) | 0 | 28.6 | 2.0 | -6.00 (bound) |
| gp-iso | vlti / flat | sigma 0.25, length_mas 0.263 **edge** | 7.68 | 2.75 | 5.31 | 0.91 | 0.73 (4.13) · 0.83 · 0.32 | 0.78 (0.66) | -5565 | 25.3 | 3.2 | -6.00 (bound) |
| gp-aniso | current / blob | sigma 4, length_mas 0.263 **edge** | 0.76 | 0.65 | 2.43 | 0.41 | 0.76 (–) · 2.43 · 0.41 | 0.92 (0.84) | -10 | 29.2 | 1.0 | -6.00 (bound) |
| gp-aniso | current / flat | sigma 4, length_mas 0.263 **edge** | 0.79 | 0.63 | 2.24 | 0.41 | 0.79 (–) · 2.24 · 0.41 | 0.91 (0.83) | 0 | 29.7 | 0.0 | -6.00 (bound) |
| gp-aniso | vlti / blob | sigma 4, length_mas 0.263 **edge** | 0.97 | 0.68 | 3.60 | 0.57 | 0.36 (1.28) · 0.19 · -0.11 | 0.94 (0.85) | -13 | 28.4 | 1.9 | -6.00 (bound) |
| gp-aniso | vlti / flat | sigma 4, length_mas 0.263 **edge** | 0.98 | 0.66 | 3.66 | 0.57 | 0.36 (1.27) · 0.19 · -0.11 | 0.92 (0.84) | 0 | 28.2 | 1.9 | -6.00 (bound) |
| gp-polar | current / blob | sigma 4, length_r_mas 0.602, length_theta 0.0625 **edge** | 0.80 | 0.70 | 2.64 | 0.45 | 0.80 (–) · 2.64 · 0.45 | 0.92 (0.83) | -27 | 28.9 | 1.7 | -6.00 (bound) |
| gp-polar | current / flat | sigma 2, length_r_mas 0.602, length_theta 0.25 **edge** | 0.95 | 0.72 | 2.65 | 0.50 | 0.95 (–) · 2.65 · 0.50 | 0.94 (0.85) | -57 | 30.4 | 0.0 | -5.68 |
| gp-polar | vlti / blob | sigma 2, length_r_mas 0.602, length_theta 0.5 **edge** | 1.27 | 0.73 | 3.85 | 0.63 | 0.42 (1.49) · 0.18 · -0.05 | 0.94 (0.86) | -120 | 30.8 | 0.0 | -6.00 (bound) |
| gp-polar | vlti / flat | sigma 2, length_r_mas 0.602, length_theta 0.5 **edge** | 1.26 | 0.73 | 3.82 | 0.63 | 0.42 (1.50) · 0.18 · -0.05 | 0.94 (0.85) | -85 | 30.6 | 0.0 | -6.00 (bound) |

### IRAS15469-5311

| strategy | noise / mean | hyperparameters (edge?) | χ²/pt V² | χ²/pt T3 | common-mode V² (white = 1) | adjacent corr | after MAP gains: χ²/pt (own) · common-mode · adjacent | NCC (best control) | ΔlnZ | f_env % | f_bg % | bg index |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mem | current / blob | – | 1.09 | 0.68 | 2.93 | 0.64 | 1.09 (–) · 2.93 · 0.64 | 0.96 (0.83) | – | 46.9 | 0.0 | -2.87 |
| mem | current / flat | – | 1.09 | 0.68 | 2.93 | 0.64 | 1.09 (–) · 2.93 · 0.64 | 0.96 (0.83) | – | 46.9 | 0.0 | -2.87 |
| mem | vlti / blob | – | 1.80 | 0.64 | 4.63 | 0.77 | 0.38 (1.79) · 0.36 · 0.15 | 0.98 (0.83) | – | 47.3 | 0.0 | -5.99 (bound) |
| mem | vlti / flat | – | 1.80 | 0.64 | 4.63 | 0.77 | 0.38 (1.79) · 0.36 · 0.15 | 0.98 (0.83) | – | 47.3 | 0.0 | -5.99 (bound) |
| gp-iso | current / blob | sigma 2, length_mas 2.32 | 0.57 | 0.62 | 2.94 | 0.49 | 0.57 (–) · 2.94 · 0.49 | 0.98 (0.82) | -8 | 42.9 | 3.9 | -6.00 (bound) |
| gp-iso | current / flat | sigma 2, length_mas 2.32 | 0.63 | 0.61 | 2.58 | 0.49 | 0.63 (–) · 2.58 · 0.49 | 0.99 (0.82) | 0 | 47.1 | 0.0 | -4.72 |
| gp-iso | vlti / blob | sigma 2, length_mas 2.32 | 1.04 | 0.68 | 4.32 | 0.71 | 0.28 (1.34) · 0.34 · -0.04 | 0.98 (0.81) | -98 | 47.0 | 0.0 | -5.99 (bound) |
| gp-iso | vlti / flat | sigma 4, length_mas 0.291 **edge** | 1.27 | 0.65 | 4.63 | 0.74 | 0.30 (1.40) · 0.37 · -0.02 | 0.33 (0.27) | -198 | 51.9 | 6.8 | -6.00 (bound) |
| gp-aniso | current / blob | sigma 2, length_mas 0.581 | 0.57 | 0.63 | 2.94 | 0.49 | 0.57 (–) · 2.94 · 0.49 | 0.96 (0.78) | -6 | 42.6 | 4.2 | -5.99 (bound) |
| gp-aniso | current / flat | sigma 2, length_mas 0.581 | 0.63 | 0.61 | 2.56 | 0.49 | 0.63 (–) · 2.56 · 0.49 | 0.96 (0.80) | -8 | 47.0 | 0.0 | -4.98 |
| gp-aniso | vlti / blob | sigma 2, length_mas 0.581 | 0.86 | 0.65 | 4.25 | 0.67 | 0.25 (1.16) · 0.34 · -0.10 | 0.94 (0.77) | 0 | 42.3 | 4.4 | -6.00 (bound) |
| gp-aniso | vlti / flat | sigma 2, length_mas 1.16 | 0.90 | 0.66 | 4.29 | 0.68 | 0.26 (1.19) · 0.38 · -0.08 | 0.94 (0.78) | 0 | 42.5 | 4.3 | -6.00 (bound) |
| gp-polar | current / blob | sigma 2, length_r_mas 2.32, length_theta 2 | 0.70 | 0.72 | 3.30 | 0.58 | 0.70 (–) · 3.30 · 0.58 | 0.98 (0.82) | 0 | 40.1 | 6.9 | -3.53 |
| gp-polar | current / flat | sigma 2, length_r_mas 1.16, length_theta 2 | 0.84 | 0.71 | 3.14 | 0.61 | 0.84 (–) · 3.14 · 0.61 | 0.98 (0.81) | -45 | 46.9 | 0.0 | -5.12 |
| gp-polar | vlti / blob | sigma 2, length_r_mas 0.581, length_theta 2 **edge** | 1.32 | 0.77 | 4.52 | 0.75 | 0.31 (1.50) · 0.34 · -0.00 | 0.98 (0.81) | -77 | 46.6 | 0.0 | -6.00 (bound) |
| gp-polar | vlti / flat | sigma 0.125, length_r_mas 0.581, length_theta 0.125 **edge** | 11.40 | 2.32 | 5.07 | 0.90 | 1.53 (9.24) · 1.35 · 0.50 | 0.91 (0.72) | -8035 | 30.4 | 15.4 | -6.00 (bound) |

### PS_Gem

| strategy | noise / mean | hyperparameters (edge?) | χ²/pt V² | χ²/pt T3 | common-mode V² (white = 1) | adjacent corr | after MAP gains: χ²/pt (own) · common-mode · adjacent | NCC (best control) | ΔlnZ | f_env % | f_bg % | bg index |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mem | current / blob | – | 1.00 | 1.31 | 3.59 | 0.60 | 1.00 (–) · 3.59 · 0.60 | 0.20 (0.02) | – | 5.7 | 6.0 | 3.02 |
| mem | current / flat | – | 1.00 | 1.31 | 3.59 | 0.60 | 1.00 (–) · 3.59 · 0.60 | 0.20 (0.02) | – | 5.7 | 6.0 | 3.02 |
| mem | vlti / blob | – | 1.33 | 1.04 | 4.49 | 0.72 | 0.33 (3.77) · 0.16 · -0.06 | 0.74 (0.75) | – | 12.0 | 3.6 | 6.00 (bound) |
| mem | vlti / flat | – | 1.33 | 1.04 | 4.49 | 0.72 | 0.33 (3.77) · 0.16 · -0.06 | 0.74 (0.75) | – | 12.0 | 3.6 | 6.00 (bound) |
| gp-iso | current / blob | sigma 4, length_mas 0.295 **edge** | 0.89 | 1.06 | 3.39 | 0.56 | 0.89 (–) · 3.39 · 0.56 | 0.06 (-0.09) | 0 | 6.0 | 5.6 | 3.29 |
| gp-iso | current / flat | sigma 4, length_mas 0.295 **edge** | 0.89 | 1.06 | 3.39 | 0.56 | 0.89 (–) · 3.39 · 0.56 | 0.06 (-0.10) | 0 | 6.0 | 5.6 | 3.30 |
| gp-iso | vlti / blob | sigma 8, length_mas 0.295 **edge** | 1.33 | 1.05 | 4.49 | 0.72 | 0.33 (3.78) · 0.16 · -0.06 | 0.74 (0.75) | 0 | 11.7 | 3.7 | 5.89 (bound) |
| gp-iso | vlti / flat | sigma 8, length_mas 0.295 **edge** | 1.34 | 1.09 | 4.49 | 0.72 | 0.33 (3.78) · 0.16 · -0.06 | 0.25 (0.33) | 0 | 8.8 | 3.7 | 5.84 |
| gp-aniso | current / blob | sigma 4, length_mas 0.295 **edge** | 0.89 | 1.12 | 3.40 | 0.56 | 0.89 (–) · 3.40 · 0.56 | -0.00 (-0.13) | -5 | 5.6 | 5.8 | 3.23 |
| gp-aniso | current / flat | sigma 4, length_mas 0.295 **edge** | 0.89 | 1.11 | 3.40 | 0.56 | 0.89 (–) · 3.40 · 0.56 | -0.00 (-0.12) | -5 | 5.6 | 5.8 | 3.28 |
| gp-aniso | vlti / blob | sigma 4, length_mas 0.295 **edge** | 1.37 | 1.12 | 4.52 | 0.73 | 0.33 (3.82) · 0.17 · -0.06 | 0.13 (0.21) | -4 | 8.2 | 3.9 | 5.60 |
| gp-aniso | vlti / flat | sigma 4, length_mas 0.295 **edge** | 1.37 | 1.12 | 4.52 | 0.73 | 0.33 (3.82) · 0.17 · -0.06 | 0.12 (0.20) | -5 | 8.2 | 3.9 | 5.60 |
| gp-polar | current / blob | sigma 4, length_r_mas 0.591, length_theta 0.0625 **edge** | 1.02 | 1.42 | 3.76 | 0.61 | 1.02 (–) · 3.76 · 0.61 | 0.06 (-0.05) | -48 | 4.5 | 7.1 | 2.36 |
| gp-polar | current / flat | sigma 4, length_r_mas 0.591, length_theta 0.0625 **edge** | 1.02 | 1.42 | 3.76 | 0.61 | 1.02 (–) · 3.76 · 0.61 | 0.05 (-0.06) | -48 | 4.5 | 7.1 | 2.38 |
| gp-polar | vlti / blob | sigma 0.125, length_r_mas 0.591, length_theta 0.125 **edge** | 1.60 | 1.99 | 4.46 | 0.74 | 0.39 (4.89) · 0.32 · 0.01 | 0.29 (0.29) | -274 | 0.9 | 10.4 | 0.15 |
| gp-polar | vlti / flat | sigma 4, length_r_mas 0.591, length_theta 0.125 **edge** | 1.57 | 1.93 | 4.54 | 0.74 | 0.36 (4.49) · 0.30 · -0.04 | 0.01 (0.00) | -202 | 0.9 | 10.3 | 0.08 |

### Summary over targets

NCC against ORGANIC (zero shift), per target; Δ is the cell minus current / blob.

| strategy | cell | IW_Car | IRAS08544 | HR4049 | HD108015 | AI_Sco | EN_TrA | HD95767 | IRAS15469-5311 | PS_Gem | median NCC | median Δ | worse by > 0.05 | grid-edge fits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mem | current / blob | 0.89 | 0.96 | 0.86 | 0.98 | 0.75 | 0.95 | 0.97 | 0.96 | 0.20 | 0.95 | +0.00 | 0 | – |
| mem | vlti / blob | 0.65 (-0.24) | 0.96 (-0.00) | 0.88 (+0.01) | 0.98 (+0.00) | 0.73 (-0.02) | 0.95 (+0.00) | 0.97 (+0.00) | 0.98 (+0.02) | 0.74 (+0.54) | 0.95 | +0.00 | 1 | – |
| gp-iso | current / blob | 0.89 | 0.95 | 0.89 | 0.98 | 0.78 | 0.87 | 0.91 | 0.98 | 0.06 | 0.89 | +0.00 | 0 | 4 of 9 |
| gp-iso | current / flat | 0.90 (+0.01) | 0.96 (+0.00) | 0.88 (-0.00) | 0.97 (-0.00) | 0.51 (-0.27) | – | 0.44 (-0.47) | 0.99 (+0.00) | 0.06 (-0.01) | 0.89 | -0.00 | 2 | 2 of 8 |
| gp-iso | vlti / blob | 0.86 (-0.04) | 0.92 (-0.04) | 0.87 (-0.02) | 0.98 (+0.00) | 0.73 (-0.05) | – | 0.89 (-0.02) | 0.98 (-0.00) | 0.74 (+0.68) | 0.88 | -0.02 | 1 | 2 of 8 |
| gp-iso | vlti / flat | 0.69 (-0.21) | 0.96 (+0.00) | 0.11 (-0.77) | 0.85 (-0.13) | 0.71 (-0.07) | 0.82 (-0.05) | 0.78 (-0.13) | 0.33 (-0.65) | 0.25 (+0.19) | 0.71 | -0.13 | 6 | 8 of 9 |
| gp-aniso | current / blob | 0.88 | 0.95 | 0.86 | 0.98 | 0.78 | 0.91 | 0.92 | 0.96 | -0.00 | 0.91 | +0.00 | 0 | 4 of 9 |
| gp-aniso | current / flat | 0.87 (-0.00) | 0.95 (+0.00) | 0.81 (-0.05) | 0.98 (+0.00) | 0.53 (-0.26) | 0.88 (-0.04) | 0.91 (-0.01) | 0.96 (+0.01) | -0.00 (-0.00) | 0.88 | -0.00 | 1 | 3 of 9 |
| gp-aniso | vlti / blob | 0.88 (+0.00) | 0.95 (-0.00) | 0.85 (-0.01) | 0.97 (-0.00) | 0.74 (-0.04) | 0.91 (+0.00) | 0.94 (+0.01) | 0.94 (-0.02) | 0.13 (+0.13) | 0.91 | -0.00 | 0 | 5 of 9 |
| gp-aniso | vlti / flat | 0.75 (-0.13) | 0.86 (-0.09) | 0.11 (-0.75) | 0.80 (-0.17) | 0.73 (-0.06) | 0.89 (-0.02) | 0.92 (-0.00) | 0.94 (-0.02) | 0.12 (+0.12) | 0.80 | -0.06 | 5 | 7 of 9 |
| gp-polar | current / blob | 0.91 | 0.93 | 0.85 | 0.97 | 0.82 | 0.92 | 0.92 | 0.98 | 0.06 | 0.92 | +0.00 | 0 | 7 of 9 |
| gp-polar | current / flat | 0.89 (-0.02) | 0.93 (+0.00) | 0.78 (-0.07) | 0.97 (-0.00) | 0.77 (-0.04) | 0.89 (-0.03) | 0.94 (+0.02) | 0.98 (-0.00) | 0.05 (-0.01) | 0.89 | -0.01 | 1 | 8 of 9 |
| gp-polar | vlti / blob | 0.88 (-0.02) | 0.94 (+0.01) | 0.81 (-0.04) | 0.97 (+0.00) | 0.77 (-0.04) | 0.92 (-0.00) | 0.94 (+0.02) | 0.98 (-0.01) | 0.29 (+0.23) | 0.92 | -0.00 | 0 | 9 of 9 |
| gp-polar | vlti / flat | 0.89 (-0.02) | 0.84 (-0.09) | 0.78 (-0.07) | 0.58 (-0.38) | 0.71 (-0.10) | 0.88 (-0.04) | 0.94 (+0.01) | 0.91 (-0.08) | 0.01 (-0.04) | 0.84 | -0.07 | 5 | 9 of 9 |

Median residuals over targets and strategies (V²; plain = on the quoted errors, MAP = after subtracting the MAP gains; own = under the fit's own covariance).

| cell | χ²/pt V² plain | common-mode plain | adjacent plain | χ²/pt V² MAP | common-mode MAP | adjacent MAP | χ²/pt V² own | χ²/pt T3 |
|---|---|---|---|---|---|---|---|---|
| current / blob | 0.97 | 3.55 | 0.64 | 0.97 | 3.55 | 0.64 | – | 0.71 |
| current / flat | 0.98 | 3.56 | 0.64 | 0.98 | 3.56 | 0.64 | – | 0.80 |
| vlti / blob | 1.60 | 4.63 | 0.77 | 0.33 | 0.27 | -0.04 | 1.67 | 0.72 |
| vlti / flat | 2.23 | 4.96 | 0.86 | 0.37 | 0.27 | -0.03 | 1.81 | 0.75 |