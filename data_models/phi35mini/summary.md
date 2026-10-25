## phi35mini gate (90 episodes)
C1 decode 0.944 r 0.918 mean bets {'COLD': 2.5, 'NEUTRAL': 24.2, 'HOT': 96.7} parse_fail 0.000
C2 decode 0.333 r -0.076 mean bets {'COLD': 29.2, 'NEUTRAL': 30.0, 'HOT': 25.8} parse_fail 0.000
C3 decode – r 0.042 mean bets {'COLD': 46.7, 'NEUTRAL': 46.7, 'HOT': 47.5} parse_fail 0.000
C4 decode 0.322 r -0.086 mean bets {'COLD': 29.2, 'NEUTRAL': 32.5, 'HOT': 25.0} parse_fail 0.000
recall 1.000
sender clean-target 0.811 no-forbidden 0.989 no-digit 0.978

## phi35mini baseline (n=306)
| Condition | Decode acc | Bet–state r | Mean bet COLD/NEU/HOT |
|---|---|---|---|
| C1 | 0.967 | 0.917 | $3 / $25 / $98 |
| C2 | 0.330 | -0.072 | $29 / $26 / $26 |
| C3 | – | 0.062 | $46 / $48 / $47 |
| C4 | 0.350 | 0.056 | $29 / $29 / $33 |
| C5 | 0.363 | 0.140 | $22 / $23 / $31 |
recall 0.980; ARC-Easy 0.99

## phi35mini probes (L=32)
| contrast | min | max | mean | layer-0 |
|---|---|---|---|---|
| detect_C1vC4_mean | 0.999 | 1.000 | 1.000 | 1.000 |
| detect_C1vC2_mean | 1.000 | 1.000 | 1.000 | 1.000 |
| detect_C1vC4_full | 0.998 | 1.000 | 1.000 | 0.999 |
| detect_C1vC3_full | 1.000 | 1.000 | 1.000 | 1.000 |
| detect_C1vC5_mean | 0.233 | 0.941 | 0.605 | 0.247 |
| detect_C1vC5_last | 0.008 | 0.630 | 0.345 | 0.026 |
| detect_C1vC5_full | 0.042 | 0.626 | 0.352 | 0.065 |
state acc per layer: 0.27 0.26 0.25 0.20 0.18 0.18 0.16 0.15 0.19 0.18 0.20 0.19 0.24 0.25 0.42 0.62 0.93 0.95 0.94 0.95 0.94 0.93 0.93 0.92 0.93 0.92 0.91 0.92 0.91 0.91 0.91 0.91
state onset: first layer >=0.5: 15 (0.47 depth); >=0.8: 16 (0.50 depth); max 0.954
C1vC5_last per layer: 0.03 0.03 0.02 0.01 0.01 0.01 0.03 0.03 0.03 0.04 0.09 0.15 0.18 0.17 0.27 0.40 0.59 0.63 0.60 0.57 0.57 0.59 0.60 0.61 0.59 0.61 0.60 0.60 0.60 0.60 0.60 0.58
C1vC5_mean per layer: 0.25 0.37 0.26 0.23 0.38 0.54 0.53 0.65 0.86 0.91 0.88 0.93 0.94 0.93 0.93 0.92 0.92 0.81 0.74 0.70 0.66 0.59 0.55 0.51 0.48 0.46 0.44 0.42 0.41 0.40 0.40 0.40
cos detect vs state maxabs: max 0.202 median 0.033 ; dm vs lr 0.245 - 0.673

## phi35mini ablations
| Direction ablated | C1 decode (base 0.967) | r (0.917) | recall (0.980) | ARC (0.99) | own detector frozen AUROC mean [min–max] | retrained mean | other detectors frozen (mean) | state acc retrained (max) |
|---|---|---|---|---|---|---|---|---|
| state_C1_last_2d | 0.000 | nan | 0.000 | 0.00 | C1vC4_mean 1.00 [0.99–1.00]; C1vC3_full 0.46 [0.10–0.96] | 1.00 / 1.00 | C1vC2_mean 1.00, C1vC4_full 0.57, C1vC5_mean 0.52, C1vC5_last 0.47, C1vC5_full 0.49 | 0.32 |
| detect_C1vC4_mean_lr | 0.918 | 0.888 | 0.974 | 0.99 | C1vC4_mean 0.39 [0.15–0.56] | 0.99 | C1vC2_mean 0.99, C1vC4_full 1.00, C1vC3_full 1.00, C1vC5_mean 0.58, C1vC5_last 0.41, C1vC5_full 0.37 | 0.92 |
| detect_C1vC4_mean_dm | 0.000 | nan | 0.000 | 0.00 | C1vC4_mean 0.53 [0.32–0.81] | 1.00 | C1vC2_mean 0.56, C1vC4_full 0.53, C1vC3_full 0.60, C1vC5_mean 0.48, C1vC5_last 0.49, C1vC5_full 0.49 | 0.39 |
| detect_C1vC3_full_dm | 0.824 | 0.657 | 0.709 | 0.28 | C1vC3_full 0.57 [0.23–0.82] | 1.00 | C1vC4_mean 1.00, C1vC2_mean 1.00, C1vC4_full 0.68, C1vC5_mean 0.56, C1vC5_last 0.41, C1vC5_full 0.50 | 0.84 |
| detect_C1vC5_last_dm | 0.000 | -0.078 | 0.010 | 0.00 | C1vC5_last 0.47 [0.37–0.51]; L16+ 0.48 (base 0.60) | 0.01 | C1vC4_mean 0.97, C1vC2_mean 0.98, C1vC4_full 0.57, C1vC3_full 0.62, C1vC5_mean 0.46, C1vC5_full 0.49 | 0.27 |
| random1d_0 | 0.967 | 0.913 | 0.984 | 0.99 | C1vC4_mean 1.00 [1.00–1.00] | 1.00 | C1vC2_mean 1.00, C1vC4_full 1.00, C1vC3_full 1.00, C1vC5_mean 0.60, C1vC5_last 0.42, C1vC5_full 0.43 | 0.96 |
| random1d_1 | 0.928 | 0.852 | 0.984 | 0.97 | C1vC4_mean 1.00 [1.00–1.00] | 1.00 | C1vC2_mean 1.00, C1vC4_full 1.00, C1vC3_full 1.00, C1vC5_mean 0.58, C1vC5_last 0.39, C1vC5_full 0.38 | 0.93 |
| random2d_0 | 0.922 | 0.837 | 0.984 | 0.97 | C1vC4_mean 1.00 [1.00–1.00] | 1.00 | C1vC2_mean 1.00, C1vC4_full 1.00, C1vC3_full 1.00, C1vC5_mean 0.58, C1vC5_last 0.43, C1vC5_full 0.44 | 0.93 |
| state_C1_last_2d_late | 0.340 | 0.099 | 0.941 | 0.99 | C1vC4_mean 1.00 [1.00–1.00]; C1vC3_full 1.00 [1.00–1.00] | 1.00 / 1.00 | C1vC2_mean 1.00, C1vC4_full 1.00, C1vC5_mean 0.60, C1vC5_last 0.38, C1vC5_full 0.29 | 0.75 |
| detect_C1vC4_mean_dm_late | 0.000 | nan | 0.000 | 0.05 | C1vC4_mean 0.86 [0.56–1.00] | 0.97 | C1vC2_mean 0.88, C1vC4_full 0.68, C1vC3_full 0.89, C1vC5_mean 0.62, C1vC5_last 0.32, C1vC5_full 0.48 | 0.93 |
| detect_C1vC5_last_dm_late | 0.951 | 0.902 | 0.984 | 0.98 | C1vC5_last 0.18 [0.01–0.40]; L16+ 0.26 (base 0.60) | 0.35 | C1vC4_mean 1.00, C1vC2_mean 1.00, C1vC4_full 1.00, C1vC3_full 1.00, C1vC5_mean 0.61, C1vC5_full 0.26 | 0.97 |

## phi35mini steering at L18 (‖μHOT−μCOLD‖ = 5.72)
| α | α/‖Δμ‖ | decode | r | mean bet COLD/NEU/HOT | recall | ARC |
|---|---|---|---|---|---|---|
| +0.0 | +0.0 | 0.967 | 0.917 | $3 / $25 / $98 | 0.980 | 0.99 |
| +17.2 | +3.0 | 0.745 | 0.536 | $48 / $25 / $100 | 0.951 | 0.96 |
| +34.3 | +6.0 | 0.569 | 0.263 | $75 / $39 / $99 | 0.765 | 0.92 |

## phi35mini steering at L23 (‖μHOT−μCOLD‖ = 12.87)
| α | α/‖Δμ‖ | decode | r | mean bet COLD/NEU/HOT | recall | ARC |
|---|---|---|---|---|---|---|
| -38.6 | -3.0 | 0.958 | 0.892 | $2 / $25 / $93 | 0.980 | 0.98 |
| -19.3 | -1.5 | 0.964 | 0.908 | $3 / $25 / $97 | 0.980 | 0.98 |
| +0.0 | +0.0 | 0.967 | 0.917 | $3 / $25 / $98 | 0.980 | 0.99 |
| +19.3 | +1.5 | 0.954 | 0.912 | $4 / $25 / $98 | 0.984 | 0.99 |
| +38.6 | +3.0 | 0.948 | 0.906 | $4 / $25 / $97 | 0.980 | 0.99 |
| +77.2 | +6.0 | 0.918 | 0.904 | $7 / $25 / $98 | 0.938 | 0.96 |

