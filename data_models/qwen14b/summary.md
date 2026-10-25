## qwen14b gate (90 episodes)
C1 decode 1.000 r 0.961 mean bets {'COLD': 0.0, 'NEUTRAL': 25.0, 'HOT': 100.0} parse_fail 0.000
C2 decode 0.344 r 0.000 mean bets {'COLD': 24.2, 'NEUTRAL': 25.0, 'HOT': 24.2} parse_fail 0.000
C3 decode – r 0.000 mean bets {'COLD': 41.7, 'NEUTRAL': 40.8, 'HOT': 41.7} parse_fail 0.000
C4 decode 0.344 r 0.060 mean bets {'COLD': 22.5, 'NEUTRAL': 22.5, 'HOT': 24.2} parse_fail 0.000
recall 1.000
sender clean-target 0.911 no-forbidden 0.989 no-digit 1.000

## qwen14b baseline (n=306)
| Condition | Decode acc | Bet–state r | Mean bet COLD/NEU/HOT |
|---|---|---|---|
| C1 | 0.997 | 0.953 | $0 / $25 / $99 |
| C2 | 0.330 | -0.047 | $25 / $24 / $24 |
| C3 | – | 0.011 | $43 / $44 / $43 |
| C4 | 0.340 | -0.025 | $25 / $25 / $24 |
| C5 | 0.291 | -0.021 | $30 / $21 / $28 |
recall 1.000; ARC-Easy 0.99

## qwen14b probes (L=48)
| contrast | min | max | mean | layer-0 |
|---|---|---|---|---|
| detect_C1vC4_mean | 0.986 | 1.000 | 0.996 | 0.986 |
| detect_C1vC2_mean | 1.000 | 1.000 | 1.000 | 1.000 |
| detect_C1vC4_full | 0.961 | 1.000 | 0.990 | 0.961 |
| detect_C1vC3_full | 1.000 | 1.000 | 1.000 | 1.000 |
| detect_C1vC5_mean | 0.221 | 0.965 | 0.751 | 0.221 |
| detect_C1vC5_last | 0.015 | 0.918 | 0.516 | 0.024 |
| detect_C1vC5_full | 0.173 | 0.921 | 0.570 | 0.239 |
state acc per layer: 0.12 0.13 0.16 0.19 0.17 0.15 0.14 0.15 0.17 0.17 0.19 0.17 0.14 0.14 0.15 0.16 0.15 0.18 0.14 0.15 0.13 0.17 0.22 0.35 0.60 0.58 0.87 0.96 0.98 0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.99 0.98 0.98 0.98 0.98
state onset: first layer >=0.5: 24 (0.50 depth); >=0.8: 26 (0.54 depth); max 0.993
C1vC5_last per layer: 0.02 0.03 0.03 0.01 0.02 0.03 0.03 0.02 0.05 0.10 0.11 0.12 0.14 0.15 0.16 0.15 0.15 0.17 0.21 0.19 0.24 0.30 0.35 0.65 0.78 0.73 0.85 0.86 0.91 0.91 0.91 0.91 0.92 0.91 0.91 0.91 0.91 0.91 0.91 0.91 0.90 0.90 0.90 0.90 0.90 0.88 0.88 0.88
C1vC5_mean per layer: 0.22 0.25 0.24 0.35 0.36 0.31 0.33 0.64 0.73 0.76 0.79 0.86 0.85 0.81 0.82 0.82 0.87 0.92 0.91 0.94 0.97 0.96 0.96 0.96 0.96 0.95 0.96 0.96 0.96 0.95 0.93 0.92 0.91 0.89 0.86 0.82 0.78 0.77 0.76 0.73 0.70 0.70 0.67 0.66 0.66 0.65 0.66 0.63
cos detect vs state maxabs: max 0.059 median 0.024 ; dm vs lr 0.379 - 0.633

## qwen14b ablations
| Direction ablated | C1 decode (base 0.997) | r (0.953) | recall (1.000) | ARC (0.99) | own detector frozen AUROC mean [min–max] | retrained mean | other detectors frozen (mean) | state acc retrained (max) |
|---|---|---|---|---|---|---|---|---|
| state_C1_last_2d | 0.389 | 0.503 | 0.752 | 0.82 | C1vC4_mean 0.99 [0.98–1.00]; C1vC3_full 0.98 [0.64–1.00] | 0.99 / 1.00 | C1vC2_mean 1.00, C1vC4_full 0.68, C1vC5_mean 0.65, C1vC5_last 0.45, C1vC5_full 0.46 | 0.86 |
| detect_C1vC4_mean_lr | 0.984 | 0.936 | 1.000 | 0.99 | C1vC4_mean 0.31 [0.15–0.46] | 0.96 | C1vC2_mean 0.99, C1vC4_full 0.98, C1vC3_full 1.00, C1vC5_mean 0.61, C1vC5_last 0.53, C1vC5_full 0.59 | 0.99 |
| detect_C1vC4_mean_dm | 0.971 | 0.928 | 0.993 | 0.98 | C1vC4_mean 0.63 [0.29–0.96] | 0.97 | C1vC2_mean 0.95, C1vC4_full 0.96, C1vC3_full 1.00, C1vC5_mean 0.71, C1vC5_last 0.57, C1vC5_full 0.62 | 0.95 |
| detect_C1vC3_full_dm | 0.980 | 0.926 | 0.997 | 0.98 | C1vC3_full 0.82 [0.00–1.00] | 1.00 | C1vC4_mean 1.00, C1vC2_mean 1.00, C1vC4_full 0.88, C1vC5_mean 0.71, C1vC5_last 0.57, C1vC5_full 0.53 | 0.97 |
| detect_C1vC5_last_dm | 0.876 | 0.862 | 0.951 | 0.93 | C1vC5_last 0.42 [0.22–0.58]; L26+ 0.45 (base 0.90) | 0.32 | C1vC4_mean 1.00, C1vC2_mean 1.00, C1vC4_full 0.76, C1vC3_full 0.95, C1vC5_mean 0.70, C1vC5_full 0.45 | 0.92 |
| random1d_0 | 0.993 | 0.950 | 1.000 | 0.99 | C1vC4_mean 1.00 [0.99–1.00] | 1.00 | C1vC2_mean 1.00, C1vC4_full 0.99, C1vC3_full 1.00, C1vC5_mean 0.74, C1vC5_last 0.53, C1vC5_full 0.58 | 0.99 |
| random1d_1 | 0.997 | 0.953 | 1.000 | 0.99 | C1vC4_mean 1.00 [0.99–1.00] | 1.00 | C1vC2_mean 1.00, C1vC4_full 0.99, C1vC3_full 1.00, C1vC5_mean 0.74, C1vC5_last 0.52, C1vC5_full 0.58 | 0.99 |
| random2d_0 | 0.984 | 0.942 | 0.997 | 0.98 | C1vC4_mean 1.00 [0.99–1.00] | 1.00 | C1vC2_mean 1.00, C1vC4_full 0.99, C1vC3_full 1.00, C1vC5_mean 0.75, C1vC5_last 0.54, C1vC5_full 0.56 | 0.99 |
| state_C1_last_2d_late | 0.837 | 0.868 | 0.997 | 0.99 | C1vC4_mean 1.00 [0.99–1.00]; C1vC3_full 1.00 [1.00–1.00] | 1.00 / 1.00 | C1vC2_mean 1.00, C1vC4_full 0.97, C1vC5_mean 0.75, C1vC5_last 0.51, C1vC5_full 0.49 | 0.97 |

## qwen14b steering at L28 (‖μHOT−μCOLD‖ = 11.78)
| α | α/‖Δμ‖ | decode | r | mean bet COLD/NEU/HOT | recall | ARC |
|---|---|---|---|---|---|---|
| +0.0 | +0.0 | 0.997 | 0.953 | $0 / $25 / $99 | 1.000 | 0.99 |
| +35.3 | +3.0 | 0.876 | 0.627 | $33 / $28 / $100 | 0.958 | 0.95 |
| +70.7 | +6.0 | 0.699 | 0.075 | $93 / $28 / $99 | 0.869 | 0.88 |

## qwen14b steering at L34 (‖μHOT−μCOLD‖ = 19.00)
| α | α/‖Δμ‖ | decode | r | mean bet COLD/NEU/HOT | recall | ARC |
|---|---|---|---|---|---|---|
| -57.0 | -3.0 | 0.997 | 0.953 | $0 / $25 / $99 | 1.000 | 0.98 |
| -28.5 | -1.5 | 0.997 | 0.953 | $0 / $25 / $99 | 1.000 | 0.99 |
| +0.0 | +0.0 | 0.997 | 0.953 | $0 / $25 / $99 | 1.000 | 0.99 |
| +28.5 | +1.5 | 0.993 | 0.950 | $0 / $26 / $99 | 1.000 | 0.98 |
| +57.0 | +3.0 | 0.984 | 0.945 | $0 / $27 / $99 | 1.000 | 0.98 |
| +114.0 | +6.0 | 0.784 | 0.736 | $22 / $28 / $100 | 0.899 | 0.96 |

