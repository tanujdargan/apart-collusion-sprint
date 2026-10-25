# LOSO robustness check — generated tables (exp/loso_analyze.py)

Pooled over 6 held-out folds, n = 306 episodes (51 per fold). Wilson 95% CIs. Parse failures = bet unparseable.

`*_global` = directions fitted on all 306 episodes INCLUDING the evaluated scheme (in-sample); `*_fold` = fitted without the evaluated scheme (held-out). Ablation: all decoder layers + embedding output (the embedding uses the layer-0 direction, an inherited convention). ~150 uncorrected tests across models: descriptive only.

| config | C1 decode [95% CI] | C1 bet-exact | C1 parse fail | C1 decoded None | C1 decoded C/N/H | C6 bet-exact [95% CI] | C6 parse fail | recall | ARC (mean of folds) |
|---|---|---|---|---|---|---|---|---|---|
| none | 0.971 [0.945–0.984] | 0.971 | 0 | 0 | 101/104/101 | 1.000 [0.988–1.000] | 0 | 0.974 | 0.980 |
| dm_global | 0.915 [0.878–0.941] | 0.915 | 0 | 0 | 103/119/84 | 0.997 [0.982–0.999] | 0 | 0.958 | 0.970 |
| dm_fold | 0.925 [0.890–0.949] | 0.925 | 0 | 0 | 101/119/86 | 0.997 [0.982–0.999] | 0 | 0.958 | 0.973 |
| pc1d_global | 0.971 [0.945–0.984] | 0.971 | 0 | 0 | 101/104/101 | 1.000 [0.988–1.000] | 0 | 0.974 | 0.990 |
| pc1d_fold | 0.971 [0.945–0.984] | 0.971 | 0 | 0 | 101/104/101 | 1.000 [0.988–1.000] | 0 | 0.974 | 0.992 |
| state_global | 0.928 [0.894–0.952] | 0.928 | 0 | 0 | 108/104/94 | 1.000 [0.988–1.000] | 0 | 0.971 | 0.900 |
| state_fold | 0.886 [0.845–0.917] | 0.886 | 0 | 0 | 110/120/76 | 0.993 [0.976–0.998] | 0 | 0.980 | 0.883 |
| pc2d_global | 0.974 [0.949–0.987] | 0.974 | 0 | 0 | 105/100/101 | 1.000 [0.988–1.000] | 0 | 0.974 | 0.980 |
| pc2d_fold | 0.974 [0.949–0.987] | 0.961 | 4 | 0 | 103/102/101 | 0.993 [0.976–0.998] | 0 | 0.977 | 0.952 |
| lr_global | 0.977 [0.954–0.989] | 0.977 | 0 | 0 | 99/105/102 | 1.000 [0.988–1.000] | 0 | 0.967 | 0.990 |
| lr_fold | 0.977 [0.954–0.989] | 0.977 | 0 | 0 | 101/103/102 | 1.000 [0.988–1.000] | 0 | 0.971 | 0.987 |

Paired changes vs matched baseline (`none`, same episodes and batching); stratified bootstrap 95% CI; exact McNemar p.

| config | Δ C1 decode [CI] | McNemar p | Δ C1 bet-exact [CI] | Δ C6 bet-exact [CI] | McNemar p (C6) |
|---|---|---|---|---|---|
| dm_global | -0.056 [-0.085, -0.026] | 0.00049 | -0.056 [-0.085, -0.026] | -0.003 [-0.010, +0.000] | 1 |
| dm_fold | -0.046 [-0.072, -0.020] | 0.0026 | -0.046 [-0.075, -0.020] | -0.003 [-0.010, +0.000] | 1 |
| pc1d_global | +0.000 [+0.000, +0.000] | 1 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 1 |
| pc1d_fold | +0.000 [+0.000, +0.000] | 1 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 1 |
| state_global | -0.042 [-0.069, -0.020] | 0.00098 | -0.042 [-0.065, -0.020] | +0.000 [+0.000, +0.000] | 1 |
| state_fold | -0.085 [-0.118, -0.056] | 2.6e-06 | -0.085 [-0.118, -0.052] | -0.007 [-0.016, +0.000] | 0.5 |
| pc2d_global | +0.003 [-0.010, +0.016] | 1 | +0.003 [-0.010, +0.016] | +0.000 [+0.000, +0.000] | 1 |
| pc2d_fold | +0.003 [-0.007, +0.013] | 1 | -0.010 [-0.026, +0.007] | -0.007 [-0.016, +0.000] | 0.5 |
| lr_global | +0.007 [+0.000, +0.016] | 0.5 | +0.007 [+0.000, +0.016] | +0.000 [+0.000, +0.000] | 1 |
| lr_fold | +0.007 [+0.000, +0.016] | 0.5 | +0.007 [+0.000, +0.016] | +0.000 [+0.000, +0.000] | 1 |
| dm_fold_minus_global | +0.010 [-0.003, +0.023] | 0.38 | – | +0.000 [+0.000, +0.000] | 1 |
| pc1d_fold_minus_global | +0.000 [+0.000, +0.000] | 1 | – | +0.000 [+0.000, +0.000] | 1 |
| state_fold_minus_global | -0.042 [-0.072, -0.010] | 0.019 | – | -0.007 [-0.016, +0.000] | 0.5 |
| pc2d_fold_minus_global | +0.000 [-0.010, +0.010] | 1 | – | -0.007 [-0.016, +0.000] | 0.5 |
| lr_fold_minus_global | +0.000 [-0.010, +0.010] | 1 | – | +0.000 [+0.000, +0.000] | 1 |
| dm_fold_minus_pc1d_fold | -0.046 [-0.075, -0.020] | 0.0026 | – | -0.003 [-0.010, +0.000] | 1 |
| state_fold_minus_pc2d_fold | -0.088 [-0.121, -0.056] | 1.4e-06 | – | +0.000 [-0.013, +0.013] | 1 |

Pre-specified decisions (C1 decode; drop = none − ablated):

| direction | drop global | drop fold [CI] | fold/global | mechanical verdict | gated verdict (global ≥ 10 pp) | C6 drop (upper CI) | C6 spared? | per-fold consistency |
|---|---|---|---|---|---|---|---|---|
| dm | 0.056 | 0.046 [0.020, 0.072] | 0.82 | SURVIVES | **n/a (global effect < 10 pp or CI includes 0)** | 0.003 (0.010) | True | 4/6 (folds with drop_fold ≥ 0.5·drop_global > 0) |
| pc1d | -0.000 | -0.000 [-0.000, -0.000] | – | DOES NOT SURVIVE | **n/a (global effect < 10 pp or CI includes 0)** | -0.000 (-0.000) | True | 0/6 (folds with drop_fold ≥ 0.5·drop_global > 0) |
| state | 0.042 | 0.085 [0.056, 0.118] | 2.00 | SURVIVES | **n/a (global effect < 10 pp or CI includes 0)** | 0.007 (0.016) | True | 3/6 (folds with drop_fold ≥ 0.5·drop_global > 0) |
| pc2d | -0.003 | -0.003 [-0.013, 0.007] | – | DOES NOT SURVIVE | **n/a (global effect < 10 pp or CI includes 0)** | 0.007 (0.016) | True | 1/6 (folds with drop_fold ≥ 0.5·drop_global > 0) |
| lr | -0.007 | -0.007 [-0.016, -0.000] | – | DOES NOT SURVIVE | **n/a (global effect < 10 pp or CI includes 0)** | -0.000 (-0.000) | True | 0/6 (folds with drop_fold ≥ 0.5·drop_global > 0) |

Per-fold C1 decode (n = 51 each):

| direction | fold | none | global | fold-fit | C6 bet-exact (fold-fit) |
|---|---|---|---|---|---|
| dm | 0 | 0.961 | 0.824 | 0.863 | 1.000 |
| dm | 1 | 0.980 | 0.941 | 0.941 | 0.980 |
| dm | 2 | 0.941 | 0.863 | 0.843 | 1.000 |
| dm | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| dm | 4 | 0.941 | 0.863 | 0.902 | 1.000 |
| dm | 5 | 1.000 | 1.000 | 1.000 | 1.000 |
| pc1d | 0 | 0.961 | 0.961 | 0.961 | 1.000 |
| pc1d | 1 | 0.980 | 0.980 | 0.980 | 1.000 |
| pc1d | 2 | 0.941 | 0.941 | 0.941 | 1.000 |
| pc1d | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| pc1d | 4 | 0.941 | 0.941 | 0.941 | 1.000 |
| pc1d | 5 | 1.000 | 1.000 | 1.000 | 1.000 |
| state | 0 | 0.961 | 0.961 | 0.980 | 1.000 |
| state | 1 | 0.980 | 0.882 | 0.902 | 1.000 |
| state | 2 | 0.941 | 0.941 | 0.961 | 0.961 |
| state | 3 | 1.000 | 0.980 | 0.882 | 1.000 |
| state | 4 | 0.941 | 0.804 | 0.588 | 1.000 |
| state | 5 | 1.000 | 1.000 | 1.000 | 1.000 |
| pc2d | 0 | 0.961 | 1.000 | 1.000 | 1.000 |
| pc2d | 1 | 0.980 | 0.961 | 0.961 | 1.000 |
| pc2d | 2 | 0.941 | 0.922 | 0.941 | 1.000 |
| pc2d | 3 | 1.000 | 1.000 | 1.000 | 0.961 |
| pc2d | 4 | 0.941 | 0.961 | 0.941 | 1.000 |
| pc2d | 5 | 1.000 | 1.000 | 1.000 | 1.000 |
| lr | 0 | 0.961 | 0.961 | 0.980 | 1.000 |
| lr | 1 | 0.980 | 0.980 | 0.980 | 1.000 |
| lr | 2 | 0.941 | 0.980 | 0.961 | 1.000 |
| lr | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| lr | 4 | 0.941 | 0.941 | 0.941 | 1.000 |
| lr | 5 | 1.000 | 1.000 | 1.000 | 1.000 |

Determinism vs original 306-episode runs (different batch composition):

| config | C1 identical output | C1 same decoded | C1 decode, orig rows same ids | C6 identical output | C6 same bet |
|---|---|---|---|---|---|
