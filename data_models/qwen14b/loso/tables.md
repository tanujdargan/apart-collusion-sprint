# LOSO robustness check — generated tables (exp/loso_analyze.py)

Pooled over 6 held-out folds, n = 306 episodes (51 per fold). Wilson 95% CIs. Parse failures = bet unparseable.

`*_global` = directions fitted on all 306 episodes INCLUDING the evaluated scheme (in-sample); `*_fold` = fitted without the evaluated scheme (held-out). Ablation: all decoder layers + embedding output (the embedding uses the layer-0 direction, an inherited convention). ~150 uncorrected tests across models: descriptive only.

| config | C1 decode [95% CI] | C1 bet-exact | C1 parse fail | C1 decoded None | C1 decoded C/N/H | C6 bet-exact [95% CI] | C6 parse fail | recall | ARC (mean of folds) |
|---|---|---|---|---|---|---|---|---|---|
| none | 0.997 [0.982–0.999] | 0.997 | 0 | 0 | 103/102/101 | 1.000 [0.988–1.000] | 0 | 1.000 | 0.990 |
| dm_global | 0.971 [0.945–0.984] | 0.977 | 0 | 2 | 103/106/95 | 1.000 [0.988–1.000] | 0 | 0.993 | 0.980 |
| dm_fold | 0.974 [0.949–0.987] | 0.977 | 0 | 1 | 103/104/98 | 1.000 [0.988–1.000] | 0 | 0.993 | 0.978 |
| pc1d_global | 0.961 [0.933–0.977] | 0.961 | 0 | 0 | 101/92/113 | 1.000 [0.988–1.000] | 0 | 0.958 | 0.980 |
| pc1d_fold | 0.990 [0.972–0.997] | 0.990 | 0 | 0 | 103/100/103 | 1.000 [0.988–1.000] | 0 | 0.984 | 0.990 |
| state_global | 0.382 [0.330–0.438] | 0.624 | 12 | 145 | 126/9/26 | 0.709 [0.656–0.757] | 0 | 0.745 | 0.820 |
| state_fold | 0.464 [0.409–0.520] | 0.614 | 17 | 112 | 135/50/9 | 0.771 [0.721–0.815] | 0 | 0.745 | 0.753 |
| pc2d_global | 0.114 [0.083–0.155] | 0.637 | 6 | 257 | 7/14/28 | 0.866 [0.823–0.900] | 29 | 0.931 | 0.350 |
| pc2d_fold | 0.288 [0.240–0.341] | 0.546 | 78 | 187 | 39/9/71 | 0.663 [0.609–0.714] | 68 | 0.794 | 0.365 |
| lr_global | 0.984 [0.962–0.993] | 0.984 | 0 | 0 | 103/106/97 | 1.000 [0.988–1.000] | 0 | 1.000 | 0.990 |
| lr_fold | 0.990 [0.972–0.997] | 0.990 | 0 | 0 | 103/104/99 | 1.000 [0.988–1.000] | 0 | 1.000 | 0.990 |

Paired changes vs matched baseline (`none`, same episodes and batching); stratified bootstrap 95% CI; exact McNemar p.

| config | Δ C1 decode [CI] | McNemar p | Δ C1 bet-exact [CI] | Δ C6 bet-exact [CI] | McNemar p (C6) |
|---|---|---|---|---|---|
| dm_global | -0.026 [-0.046, -0.010] | 0.0078 | -0.020 [-0.036, -0.007] | +0.000 [+0.000, +0.000] | 1 |
| dm_fold | -0.023 [-0.042, -0.007] | 0.016 | -0.020 [-0.036, -0.007] | +0.000 [+0.000, +0.000] | 1 |
| pc1d_global | -0.036 [-0.056, -0.016] | 0.0034 | -0.036 [-0.056, -0.016] | +0.000 [+0.000, +0.000] | 1 |
| pc1d_fold | -0.007 [-0.016, +0.000] | 0.5 | -0.007 [-0.016, +0.000] | +0.000 [+0.000, +0.000] | 1 |
| state_global | -0.614 [-0.667, -0.562] | 5.1e-57 | -0.373 [-0.425, -0.320] | -0.291 [-0.340, -0.239] | 3.2e-27 |
| state_fold | -0.533 [-0.585, -0.480] | 1.7e-49 | -0.382 [-0.435, -0.330] | -0.229 [-0.275, -0.183] | 1.7e-21 |
| pc2d_global | -0.882 [-0.915, -0.846] | 1.1e-81 | -0.359 [-0.412, -0.307] | -0.134 [-0.173, -0.098] | 9.1e-13 |
| pc2d_fold | -0.709 [-0.755, -0.660] | 5.2e-64 | -0.451 [-0.500, -0.402] | -0.337 [-0.376, -0.297] | 2e-31 |
| lr_global | -0.013 [-0.026, -0.003] | 0.12 | -0.013 [-0.026, -0.003] | +0.000 [+0.000, +0.000] | 1 |
| lr_fold | -0.007 [-0.016, +0.000] | 0.5 | -0.007 [-0.016, +0.000] | +0.000 [+0.000, +0.000] | 1 |
| dm_fold_minus_global | +0.003 [-0.007, +0.016] | 1 | – | +0.000 [+0.000, +0.000] | 1 |
| pc1d_fold_minus_global | +0.029 [+0.010, +0.049] | 0.012 | – | +0.000 [+0.000, +0.000] | 1 |
| state_fold_minus_global | +0.082 [+0.029, +0.137] | 0.011 | – | +0.062 [+0.023, +0.101] | 0.0034 |
| pc2d_fold_minus_global | +0.173 [+0.121, +0.225] | 8.5e-08 | – | -0.203 [-0.252, -0.154] | 6.7e-10 |
| lr_fold_minus_global | +0.007 [+0.000, +0.016] | 0.5 | – | +0.000 [+0.000, +0.000] | 1 |
| dm_fold_minus_pc1d_fold | -0.016 [-0.036, +0.003] | 0.18 | – | +0.000 [+0.000, +0.000] | 1 |
| state_fold_minus_pc2d_fold | +0.176 [+0.108, +0.245] | 2.1e-05 | – | +0.108 [+0.042, +0.170] | 0.005 |

Pre-specified decisions (C1 decode; drop = none − ablated):

| direction | drop global | drop fold [CI] | fold/global | mechanical verdict | gated verdict (global ≥ 10 pp) | C6 drop (upper CI) | C6 spared? | per-fold consistency |
|---|---|---|---|---|---|---|---|---|
| dm | 0.026 | 0.023 [0.007, 0.042] | 0.87 | SURVIVES | **n/a (global effect < 10 pp or CI includes 0)** | -0.000 (-0.000) | True | 4/6 (folds with drop_fold ≥ 0.5·drop_global > 0) |
| pc1d | 0.036 | 0.007 [-0.000, 0.016] | 0.18 | DOES NOT SURVIVE | **n/a (global effect < 10 pp or CI includes 0)** | -0.000 (-0.000) | True | 0/6 (folds with drop_fold ≥ 0.5·drop_global > 0) |
| state | 0.614 | 0.533 [0.480, 0.585] | 0.87 | SURVIVES | **SURVIVES** | 0.229 (0.275) | False | 5/6 (folds with drop_fold ≥ 0.5·drop_global > 0) |
| pc2d | 0.882 | 0.709 [0.660, 0.755] | 0.80 | SURVIVES | **SURVIVES** | 0.337 (0.376) | False | 5/6 (folds with drop_fold ≥ 0.5·drop_global > 0) |
| lr | 0.013 | 0.007 [-0.000, 0.016] | 0.50 | DOES NOT SURVIVE | **n/a (global effect < 10 pp or CI includes 0)** | -0.000 (-0.000) | True | 1/6 (folds with drop_fold ≥ 0.5·drop_global > 0) |

Per-fold C1 decode (n = 51 each):

| direction | fold | none | global | fold-fit | C6 bet-exact (fold-fit) |
|---|---|---|---|---|---|
| dm | 0 | 1.000 | 0.941 | 0.961 | 1.000 |
| dm | 1 | 1.000 | 0.980 | 0.961 | 1.000 |
| dm | 2 | 0.980 | 0.980 | 0.980 | 1.000 |
| dm | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| dm | 4 | 1.000 | 0.961 | 0.980 | 1.000 |
| dm | 5 | 1.000 | 0.961 | 0.961 | 1.000 |
| pc1d | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| pc1d | 1 | 1.000 | 0.784 | 0.961 | 1.000 |
| pc1d | 2 | 0.980 | 1.000 | 0.980 | 1.000 |
| pc1d | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| pc1d | 4 | 1.000 | 1.000 | 1.000 | 1.000 |
| pc1d | 5 | 1.000 | 0.980 | 1.000 | 1.000 |
| state | 0 | 1.000 | 0.471 | 0.804 | 0.667 |
| state | 1 | 1.000 | 0.412 | 0.608 | 0.902 |
| state | 2 | 0.980 | 0.353 | 0.255 | 0.706 |
| state | 3 | 1.000 | 0.275 | 0.569 | 0.824 |
| state | 4 | 1.000 | 0.275 | 0.333 | 0.667 |
| state | 5 | 1.000 | 0.510 | 0.216 | 0.863 |
| pc2d | 0 | 1.000 | 0.275 | 0.118 | 0.196 |
| pc2d | 1 | 1.000 | 0.020 | 0.549 | 0.373 |
| pc2d | 2 | 0.980 | 0.039 | 0.353 | 0.980 |
| pc2d | 3 | 1.000 | 0.216 | 0.157 | 0.961 |
| pc2d | 4 | 1.000 | 0.039 | 0.059 | 0.549 |
| pc2d | 5 | 1.000 | 0.098 | 0.490 | 0.922 |
| lr | 0 | 1.000 | 0.961 | 0.961 | 1.000 |
| lr | 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| lr | 2 | 0.980 | 0.980 | 0.980 | 1.000 |
| lr | 3 | 1.000 | 0.980 | 1.000 | 1.000 |
| lr | 4 | 1.000 | 1.000 | 1.000 | 1.000 |
| lr | 5 | 1.000 | 0.980 | 1.000 | 1.000 |

Determinism vs original 306-episode runs (different batch composition):

| config | C1 identical output | C1 same decoded | C1 decode, orig rows same ids | C6 identical output | C6 same bet |
|---|---|---|---|---|---|
