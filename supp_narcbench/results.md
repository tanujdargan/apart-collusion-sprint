# NARCBench (Qwen3-32B) confound analysis: supplementary results

Everything below was computed by `analysis.py` (CPU, about 5.5 min on an M5). All numbers are copied from its output (`tables.md`, `results.json`, `run.log`).

## Summary (5 lines)
1. **Reproduction holds.** With the paper's code (`reproduce_LR.py`, imported read-only, 10 seeds, L26-30), Core held-out = 1.000 for all 5 methods and Stego (blackjack) = 1.000 Peak Suspicion. Transfer Peak Suspicion ranges from 0.644 (cicd) to 0.978 (code_review/research). The README and reproduce_LR.py state no numeric targets, so there was nothing to compare against beyond the code itself.
2. **Stego is fully confounded by length.** In the final phase every colluder has more tokens than every honest agent (1819-2748 vs 774-1231). n_tokens alone, with no model, gives colluder-vs-honest AUROC 1.000 and run-level Peak AUROC 1.000, the same as the probe. Regressing log n_tokens out of the probe score drops Stego colluder AUROC from 0.961 to 0.516 and Peak from 1.000 to 0.891. On this data, the Stego result cannot tell "detects colluders" apart from "detects longer contexts".
3. **Core is partly length-confounded but survives control.** n_tokens alone gives colluder AUROC 0.64 (0.77 with round/phase), Core run-Peak 0.877, and rho(score, n_tokens) is about 0.3. Still, the probe stays at 1.000 within round x n_tokens-quintile cells and at 0.975 after residualizing on n_tokens + round/phase.
4. **Transfer is the cleanest evidence.** n_tokens and the other surface features are near chance there (role 0.51-0.53, run-Peak 0.576). The zero-shot probe gets role AUROC 0.805 and Peak 0.831 (simple probe, mean over layers), and residualizing on length does not hurt it (0.822 / 0.843). The probe mostly flags the colluding agent, not the run: honest agents in collusion runs vs control runs score only 0.54-0.59 (Transfer) and 0.56-0.58 (Core).
5. **Probe directions are mostly tier-specific.** Probes trained separately per tier have cosine about 0.20-0.23 for core-transfer (shuffled baseline 0.00 ± 0.02, z about 12-15) and only 0.02-0.10 for core-stego and 0.01-0.07 for transfer-stego. Within-tier split-half ceilings are 0.70-0.74 (core), 0.55-0.56 (transfer) and 0.96-0.98 (stego). Stego has its own very stable direction that is nearly orthogonal to Core's, even though the Core probe separates Stego colluders zero-shot. That pattern fits Stego being trivially separable along many directions, including length.

## Label choice (reasoning)
- `mode` (collusion/control) is a **run-level** label. Collusion and control runs come in pairs per `scenario_id` (Core 31 scenarios / 62 runs, Transfer 36/72, Stego 20/40).
- `role` (colluder/honest) is an **agent-level** label. Colluders exist only in collusion runs. Each collusion run has 2 colluders and 2 honest agents, and every control-run agent is honest.
- The paper (reproduce_LR.py `train_deception_probe`) trains the probe on **role** within collusion runs (public channel, discussion phase) and evaluates on **mode** at run level (Peak Suspicion = max agent score, etc.). I use both: `role` for the probe target and agent-level tests, and `mode` for row/run-level detection.
- `channel=private` / `phase=private` rows exist **only** for colluders in collusion runs (62 in Core, 40 in Stego, 0 in Transfer), so channel is a perfect leak for `mode` on any all-rows evaluation. The paper's pipeline uses public rows only. `flagged` is False everywhere.
- Stored layers are only 26-30 (5120-d float32). Rows per tier: Core 1054, Transfer 288, Stego 1320.

## Method notes
- Probe hyperparameters are copied from reproduce_LR.py: StandardScaler + LogisticRegression(C=25, l2, liblinear, max_iter=8), with majority-class subsampling. max_iter=8 causes liblinear ConvergenceWarnings; that is the paper's own choice. The cosine analysis also reports a converged C=1.0 lbfgs probe ("S").
- Within-tier evaluation always uses **GroupKFold(5) by scenario_id**, so a scenario's collusion and control runs are never split across train and test. Core probe scores in 2b/3 are out-of-fold. Transfer/Stego scores come from a probe trained on all Core rows (zero-shot).
- Eval set follows the paper: Core = public discussion rows (run-level metrics average each agent over rounds), Transfer/Stego = public final rows.
- Surface-feature models: log n_tokens, channel one-hot, phase one-hot, round one-hot, role, each fed to LR(C=1), out-of-fold AUROC.
- Residualization: within-tier OLS of probe score on covariates (no label), then AUROC of the residuals.
- Cosine: per-tier probe directions mapped to raw activation space (w/sigma). Baseline: 20 independent label permutations per tier. Ceiling: split-half by scenario (10 halvings).

## Caveats
- `n_tokens` is presumably the input context length (prompt + history). Colluders' contexts likely include their private coordination/instructions, which would explain the length gap. I did not verify this against raw transcripts, and the scenario text was not downloaded. The same logic implies the Core/Stego probe may be reading the *presence of collusion instructions in the prompt* rather than collusive behaviour. Length control cannot rule that out.
- In Stego, length perfectly separates role in the final phase. Its "within n_tokens quintile" AUROCs (0.92-1.0) rest on only 64 of 1600 colluder-honest pairs, all in bins straddling the gap, so they are not meaningful. Treat the residualized number (0.516) as the informative one, with the caveat that length and role are not identifiable in this data.
- Transfer per-family results use only 12 runs each (6 vs 6), so AUROC steps are coarse.
- Only Qwen3-32B and only layers 26-30 were analysed. No paper PDF numbers were checked, because the README and reproduce_LR.py state none.
- Residualization fits are in-sample within tier, which is standard for confound removal but slightly optimistic for the covariates.

## Exact commands
```bash
cd supp_narcbench
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python numpy scikit-learn matplotlib huggingface_hub scipy
.venv/bin/python analysis.py > run.log 2>&1   # downloads activations/qwen3_32b/{core,transfer,stego} into data/ if missing
```
Outputs: `tables.md` (all tables), `results.json`, `run.log`, `figs/auroc_by_layer.png`, `figs/confound_auroc.png`, `figs/runlevel_probe_vs_ntok.png`, `figs/cosine_heatmap_P.png`, `figs/cosine_heatmap_S.png`.

# Full tables (auto-generated by analysis.py)

## 1. Data inspection

**core**: 1054 rows; npz keys ['layer_26', 'layer_27', 'layer_28', 'layer_29', 'layer_30']; each (1054, 5120) float32; 62 runs, 31 scenario_ids
- mode: {'collusion': 558, 'control': 496}
- role: {'colluder': 310, 'honest': 744}
- channel: {'private': 62, 'public': 992}
- phase: {'private': 62, 'discussion': 744, 'final': 248}
- round: {'0': 62, '1': 248, '2': 248, '3': 248, '4': 248}
- (mode, role, phase) counts and mean n_tokens:
  - ('collusion', 'colluder', 'discussion'): n=186, mean n_tokens=1688
  - ('collusion', 'colluder', 'final'): n=62, mean n_tokens=2340
  - ('collusion', 'colluder', 'private'): n=62, mean n_tokens=569
  - ('collusion', 'honest', 'discussion'): n=186, mean n_tokens=1350
  - ('collusion', 'honest', 'final'): n=62, mean n_tokens=2340
  - ('control', 'honest', 'discussion'): n=372, mean n_tokens=1290
  - ('control', 'honest', 'final'): n=124, mean n_tokens=2245

**transfer**: 288 rows; npz keys ['layer_26', 'layer_27', 'layer_28', 'layer_29', 'layer_30']; each (288, 5120) float32; 72 runs, 36 scenario_ids
- mode: {'collusion': 144, 'control': 144}
- role: {'colluder': 72, 'honest': 216}
- channel: {'public': 288}
- phase: {'final': 288}
- round: {'5': 96, '4': 192}
- (mode, role, phase) counts and mean n_tokens:
  - ('collusion', 'colluder', 'final'): n=72, mean n_tokens=5537
  - ('collusion', 'honest', 'final'): n=72, mean n_tokens=5514
  - ('control', 'honest', 'final'): n=144, mean n_tokens=5510

**stego**: 1320 rows; npz keys ['layer_26', 'layer_27', 'layer_28', 'layer_29', 'layer_30']; each (1320, 5120) float32; 40 runs, 20 scenario_ids
- mode: {'collusion': 680, 'control': 640}
- role: {'colluder': 360, 'honest': 960}
- channel: {'private': 40, 'public': 1280}
- phase: {'private': 40, 'discussion': 1120, 'final': 160}
- round: {'0': 40, '1': 160, '2': 160, '3': 160, '4': 160, '5': 160, '6': 160, '7': 160, '8': 160}
- (mode, role, phase) counts and mean n_tokens:
  - ('collusion', 'colluder', 'discussion'): n=280, mean n_tokens=1910
  - ('collusion', 'colluder', 'final'): n=40, mean n_tokens=2162
  - ('collusion', 'colluder', 'private'): n=40, mean n_tokens=1086
  - ('collusion', 'honest', 'discussion'): n=280, mean n_tokens=875
  - ('collusion', 'honest', 'final'): n=40, mean n_tokens=969
  - ('control', 'honest', 'discussion'): n=560, mean n_tokens=999
  - ('control', 'honest', 'final'): n=80, mean n_tokens=1087

## 2a. Paper-faithful reproduction (reproduce_LR.py imported read-only, 10 seeds x layers 26-30)

Mean AUROC ± SEM over 10 seeds x 5 layers (same aggregation as reproduce_LR.py). Run-level labels (collusion vs control run).

| group | sorted_concat | activation_alignment | peak_suspicion | deception_split | asymmetry_probe |
|---|---|---|---|---|---|
| Held-out | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| cicd | 0.528 ± 0.004 | 0.806 ± 0.000 | 0.644 ± 0.004 | 0.894 ± 0.013 | 0.767 ± 0.005 |
| code_review | 0.994 ± 0.002 | 0.217 ± 0.002 | 0.978 ± 0.006 | 0.756 ± 0.017 | 0.933 ± 0.012 |
| news | 0.733 ± 0.009 | 0.711 ± 0.005 | 0.794 ± 0.010 | 0.883 ± 0.008 | 0.867 ± 0.010 |
| pump_dump | 1.000 ± 0.000 | 0.322 ± 0.002 | 0.939 ± 0.009 | 0.722 ± 0.014 | 0.956 ± 0.008 |
| rag | 0.656 ± 0.010 | 0.472 ± 0.000 | 0.967 ± 0.005 | 0.906 ± 0.006 | 0.967 ± 0.006 |
| research | 1.000 ± 0.000 | 0.683 ± 0.002 | 0.978 ± 0.003 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| blackjack | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.999 ± 0.000 | 1.000 ± 0.000 |

Per-layer Peak Suspicion (mean over 10 seeds):

| group | L26 | L27 | L28 | L29 | L30 |
|---|---|---|---|---|---|
| Held-out | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| cicd | 0.611 | 0.611 | 0.667 | 0.667 | 0.667 |
| code_review | 0.889 | 1.000 | 1.000 | 1.000 | 1.000 |
| news | 0.694 | 0.806 | 0.750 | 0.806 | 0.917 |
| pump_dump | 0.833 | 0.972 | 0.889 | 1.000 | 1.000 |
| rag | 0.944 | 0.972 | 0.917 | 1.000 | 1.000 |
| research | 1.000 | 1.000 | 0.944 | 0.972 | 0.972 |
| blackjack | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## 2b. Simple probe (paper hyperparameters), per stored layer

Probe: StandardScaler + LogisticRegression(C=25, l2, liblinear, max_iter=8), trained on Core collusion-run public discussion rows, label role (colluder=1 vs honest=0), majority class subsampled (seed 0). Core numbers are out-of-fold from GroupKFold(5) grouped by scenario_id (collusion+control runs of a scenario never split). Transfer/Stego: zero-shot with the probe trained on all Core rows.

Metrics: `role` = colluder vs honest rows inside collusion runs (eval phase, public); `mode-row` = collusion-run vs control-run rows (eval phase, public); `peak` / `split` = run-level Peak Suspicion / Deception Split; eval phase = Core public discussion (agent score averaged over rounds), Transfer/Stego public final.

| tier | metric | L26 | L27 | L28 | L29 | L30 | mean |
|---|---|---|---|---|---|---|---|
| core | role | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| core | mode-row | 0.778 | 0.780 | 0.779 | 0.787 | 0.790 | 0.783 |
| core | peak | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| core | split | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| transfer | role | 0.801 | 0.829 | 0.792 | 0.797 | 0.807 | 0.805 |
| transfer | mode-row | 0.670 | 0.696 | 0.689 | 0.699 | 0.705 | 0.692 |
| transfer | peak | 0.789 | 0.845 | 0.797 | 0.846 | 0.876 | 0.831 |
| transfer | split | 0.764 | 0.885 | 0.840 | 0.907 | 0.941 | 0.867 |
| stego | role | 1.000 | 0.994 | 0.871 | 0.960 | 0.979 | 0.961 |
| stego | mode-row | 0.832 | 0.866 | 0.753 | 0.770 | 0.791 | 0.802 |
| stego | peak | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| stego | split | 1.000 | 1.000 | 0.995 | 1.000 | 1.000 | 0.999 |

Transfer per-family run-level Peak Suspicion (simple probe, mean over layers 26-30; 12 runs each):

| family | peak AUROC |
|---|---|
| cicd | 0.644 |
| code_review | 0.978 |
| news | 0.794 |
| pump_dump | 0.939 |
| rag | 0.967 |
| research | 0.978 |

## 3a. Surface features alone (one-hot + LR, GroupKFold(5) by scenario_id, out-of-fold AUROC)

Targets: `mode/all` = collusion-run vs control-run over ALL rows (incl. private); `mode/eval` = same on eval-phase public rows only (the paper's evaluation set); `role/coll` = colluder vs honest inside collusion runs, eval-phase public rows (the probe's own training target). Constant features in a tier give 0.5 by construction (e.g. channel in Transfer).

| tier | features | mode/all | mode/eval | role/coll |
|---|---|---|---|---|
| core | n_tokens | 0.511 | 0.599 | 0.642 |
| core | channel | 0.556 | 0.500 | 0.500 |
| core | phase | 0.556 | 0.500 | 0.500 |
| core | round | 0.556 | 0.500 | 0.500 |
| core | role | 0.778 | 0.750 | nan |
| core | all except role | 0.718 | 0.703 | 0.768 |
| core | all | 0.815 | 0.788 | nan |
| transfer | n_tokens | 0.504 | 0.504 | 0.509 |
| transfer | channel | 0.500 | 0.500 | 0.500 |
| transfer | phase | 0.500 | 0.500 | 0.500 |
| transfer | round | 0.500 | 0.500 | 0.500 |
| transfer | role | 0.750 | 0.750 | nan |
| transfer | all except role | 0.566 | 0.566 | 0.533 |
| transfer | all | 0.770 | 0.770 | nan |
| stego | n_tokens | 0.675 | 0.675 | 1.000 |
| stego | channel | 0.529 | 0.500 | 0.500 |
| stego | phase | 0.529 | 0.500 | 0.500 |
| stego | round | 0.529 | 0.500 | 0.500 |
| stego | role | 0.765 | 0.750 | nan |
| stego | all except role | 0.697 | 0.675 | 1.000 |
| stego | all | 0.831 | 0.826 | nan |

Note: `role` predicts `mode/all` trivially because only collusion runs contain colluders; `channel=private` likewise exists only in collusion runs (colluders' private coordination turns).

Run-level n_tokens surrogate: replace the probe score with log(n_tokens) and apply the paper's Peak Suspicion / Deception Split aggregation on the eval-phase public rows (no fitting at all). Compare to the probe (2b, mean over layers):

| tier | peak(ntok) | peak(probe) | split(ntok) | split(probe) | role AUROC(ntok raw) | role AUROC(probe) |
|---|---|---|---|---|---|---|
| core | 0.877 | 1.000 | 0.683 | 1.000 | 0.643 | 1.000 |
| transfer | 0.576 | 0.831 | 0.609 | 0.867 | 0.532 | 0.805 |
| stego | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 | 0.961 |

Transfer per-family Peak(n_tokens):

| family | peak(ntok) AUROC |
|---|---|
| cicd | 0.806 |
| code_review | 0.583 |
| news | 0.569 |
| pump_dump | 0.681 |
| rag | 1.000 |
| research | 0.778 |

## 3b. Correlation of the Core probe score with surface features

Rows = all rows of the tier (Core: out-of-fold scores). Spearman rho with n_tokens; Pearson r (point-biserial) with private-channel indicator, colluder indicator, and collusion-run indicator. Also Spearman(score, n_tokens) within honest-only and colluder-only rows (eval phase, public) to separate length from role.

| tier | layer | rho(ntok) | r(private) | r(colluder) | r(collusion run) | rho(ntok|honest,eval) | rho(ntok|colluder,eval) |
|---|---|---|---|---|---|---|---|
| core | L26 | 0.322 | 0.274 | 0.894 | 0.556 | 0.392 | 0.009 |
| core | L27 | 0.315 | 0.306 | 0.902 | 0.560 | 0.369 | 0.020 |
| core | L28 | 0.316 | 0.310 | 0.892 | 0.553 | 0.350 | 0.030 |
| core | L29 | 0.317 | 0.283 | 0.892 | 0.557 | 0.322 | 0.010 |
| core | L30 | 0.275 | 0.274 | 0.899 | 0.563 | 0.275 | 0.022 |
| transfer | L26 | 0.170 | nan | 0.451 | 0.296 | 0.244 | -0.052 |
| transfer | L27 | 0.075 | nan | 0.514 | 0.341 | 0.123 | -0.075 |
| transfer | L28 | 0.210 | nan | 0.470 | 0.327 | 0.270 | 0.085 |
| transfer | L29 | 0.092 | nan | 0.480 | 0.344 | 0.122 | 0.009 |
| transfer | L30 | 0.037 | nan | 0.493 | 0.354 | 0.046 | 0.052 |
| stego | L26 | 0.452 | -0.110 | 0.729 | 0.500 | -0.111 | 0.099 |
| stego | L27 | 0.341 | -0.171 | 0.567 | 0.433 | -0.591 | 0.070 |
| stego | L28 | 0.314 | -0.174 | 0.555 | 0.425 | -0.127 | 0.247 |
| stego | L29 | 0.305 | -0.175 | 0.488 | 0.376 | 0.225 | 0.368 |
| stego | L30 | 0.170 | -0.289 | 0.324 | 0.282 | -0.049 | 0.273 |

## 3c. Does the probe still work within single strata?

All on public rows. `mode|public` = collusion vs control rows, all public rows. `mode|eval` = eval phase only. `mode|honest` = honest agents in collusion runs vs honest agents in control runs (eval phase) -- tests whether the probe sees 'being in a colluding run' rather than 'being a colluder'. `role|coll` = colluder vs honest in collusion runs (eval phase). `coll vs ctrl-honest` = colluders vs control-run honest agents (eval phase). `role|round` = role AUROC within each round (phase x round), pairwise-pooled. `role|ntok-quintile` = role AUROC within n_tokens quintiles (quintiles of the eval-phase collusion-run rows), pairwise-pooled. `role|round&ntok` = within round x ntok-quintile cells.

| tier | layer | mode|public | mode|eval | mode|honest | role|coll | coll vs ctrl-honest | role|round | role|ntok-quintile | role|round&ntok |
|---|---|---|---|---|---|---|---|---|---|
| core | L26 | 0.765 | 0.778 | 0.556 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| core | L27 | 0.766 | 0.780 | 0.561 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| core | L28 | 0.764 | 0.779 | 0.557 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| core | L29 | 0.769 | 0.787 | 0.573 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| core | L30 | 0.772 | 0.790 | 0.580 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| transfer | L26 | 0.670 | 0.670 | 0.535 | 0.801 | 0.806 | 0.800 | 0.834 | 0.843 |
| transfer | L27 | 0.696 | 0.696 | 0.559 | 0.829 | 0.832 | 0.828 | 0.862 | 0.872 |
| transfer | L28 | 0.689 | 0.689 | 0.565 | 0.792 | 0.813 | 0.808 | 0.828 | 0.844 |
| transfer | L29 | 0.699 | 0.699 | 0.584 | 0.797 | 0.813 | 0.799 | 0.836 | 0.849 |
| transfer | L30 | 0.705 | 0.705 | 0.592 | 0.807 | 0.817 | 0.815 | 0.846 | 0.858 |
| stego | L26 | 0.793 | 0.832 | 0.664 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| stego | L27 | 0.777 | 0.866 | 0.733 | 0.994 | 0.999 | 0.994 | 1.000 | 1.000 |
| stego | L28 | 0.771 | 0.753 | 0.586 | 0.871 | 0.921 | 0.871 | 0.922 | 0.922 |
| stego | L29 | 0.747 | 0.770 | 0.557 | 0.960 | 0.982 | 0.960 | 0.969 | 0.969 |
| stego | L30 | 0.711 | 0.791 | 0.592 | 0.979 | 0.990 | 0.979 | 0.984 | 0.984 |

Same strata using raw n_tokens as the score (reference for how much is explainable by length):

| tier | mode|public | mode|eval | mode|honest | role|coll | coll vs ctrl-honest | role|round |
|---|---|---|---|---|---|---|
| core | 0.585 | 0.599 | 0.530 | 0.643 | 0.669 | 0.776 |
| transfer | 0.516 | 0.516 | 0.502 | 0.532 | 0.530 | 0.558 |
| stego | 0.679 | 0.675 | 0.350 | 1.000 | 1.000 | 1.000 |

## 3d. Residualized probe scores

OLS of probe score on covariates (fit within tier on the evaluated rows; covariates do not include the label), AUROC of residuals. Covariates: R1 = log n_tokens; R2 = log n_tokens + phase/round one-hot; R3 = R2 + private-channel (for all-row mode target). For run-level Peak, per-row residuals (R1/R2 fit on eval rows) are aggregated exactly like the raw score. Mean over layers 26-30; per-layer in results.json.

| tier | role raw | role R1 | role R2 | mode-all raw | mode-all R3 | mode-eval raw | mode-eval R1 | peak raw | peak R1 | peak R2 |
|---|---|---|---|---|---|---|---|---|---|---|
| core | 1.000 | 1.000 | 0.975 | 0.793 | 0.717 | 0.783 | 0.763 | 1.000 | 1.000 | 0.984 |
| transfer | 0.805 | 0.822 | 0.830 | 0.692 | 0.698 | 0.692 | 0.694 | 0.831 | 0.843 | 0.844 |
| stego | 0.961 | 0.516 | 0.516 | 0.738 | 0.700 | 0.802 | 0.686 | 1.000 | 0.891 | 0.891 |

## 3e. Is the Core probe direction aligned with a length direction?

Ridge-free OLS is ill-posed (5120 dims), so we fit LR-style linear regression via sklearn Ridge(alpha=1e3) on standardized Core activations to predict log n_tokens (GroupKFold R^2), and report cosine between the Core probe direction (standardized space) and the length-regression direction.

| layer | GroupKFold R^2 (log n_tokens) | cos(probe, length dir) |
|---|---|---|
| L26 | 0.984 | 0.192 |
| L27 | 0.984 | 0.193 |
| L28 | 0.984 | 0.196 |
| L29 | 0.984 | 0.194 |
| L30 | 0.983 | 0.189 |

## 4. Cross-tier probe direction similarity

Per tier, a probe is trained on colluder-vs-honest rows from collusion runs (all public rows of that tier: Core discussion+final, Transfer final, Stego discussion+final). Two probe configs: P = paper hyperparameters (C=25, liblinear, max_iter=8, balanced subsample), S = C=1.0 lbfgs max_iter=5000 (converged). Cosine is between unit directions mapped to raw activation space (w / sigma). Baseline: same procedure with labels permuted independently in each tier (20 shuffles), mean ± sd. Ceiling: split-half (by scenario_id, 10 random halvings) cosine within a tier.

| cfg | layer | pair | cosine | shuffled mean ± sd | z |
|---|---|---|---|---|---|
| P | L26 | core-transfer | 0.202 | 0.007 ± 0.015 | 13.441 |
| P | L26 | core-stego | 0.043 | -0.004 ± 0.013 | 3.632 |
| P | L26 | transfer-stego | 0.067 | -0.006 ± 0.021 | 3.601 |
| P | L27 | core-transfer | 0.223 | 0.006 ± 0.017 | 12.556 |
| P | L27 | core-stego | 0.022 | -0.003 ± 0.015 | 1.682 |
| P | L27 | transfer-stego | 0.058 | -0.003 ± 0.021 | 2.879 |
| P | L28 | core-transfer | 0.218 | 0.004 ± 0.018 | 11.621 |
| P | L28 | core-stego | 0.017 | -0.002 ± 0.016 | 1.218 |
| P | L28 | transfer-stego | 0.038 | -0.002 ± 0.020 | 1.937 |
| P | L29 | core-transfer | 0.211 | 0.006 ± 0.016 | 13.066 |
| P | L29 | core-stego | 0.022 | -0.004 ± 0.015 | 1.746 |
| P | L29 | transfer-stego | 0.026 | -0.003 ± 0.017 | 1.670 |
| P | L30 | core-transfer | 0.200 | 0.003 ± 0.016 | 12.631 |
| P | L30 | core-stego | 0.038 | -0.005 ± 0.018 | 2.310 |
| P | L30 | transfer-stego | 0.028 | -0.002 ± 0.017 | 1.769 |
| S | L26 | core-transfer | 0.209 | 0.007 ± 0.013 | 15.358 |
| S | L26 | core-stego | 0.099 | -0.005 ± 0.014 | 7.644 |
| S | L26 | transfer-stego | 0.065 | -0.007 ± 0.022 | 3.327 |
| S | L27 | core-transfer | 0.232 | 0.006 ± 0.016 | 13.961 |
| S | L27 | core-stego | 0.071 | -0.004 ± 0.017 | 4.445 |
| S | L27 | transfer-stego | 0.048 | -0.003 ± 0.022 | 2.287 |
| S | L28 | core-transfer | 0.225 | 0.004 ± 0.018 | 11.995 |
| S | L28 | core-stego | 0.067 | -0.004 ± 0.017 | 4.170 |
| S | L28 | transfer-stego | 0.027 | -0.002 ± 0.021 | 1.355 |
| S | L29 | core-transfer | 0.221 | 0.007 ± 0.015 | 14.013 |
| S | L29 | core-stego | 0.053 | -0.005 ± 0.015 | 3.898 |
| S | L29 | transfer-stego | 0.013 | -0.003 ± 0.018 | 0.889 |
| S | L30 | core-transfer | 0.210 | 0.004 ± 0.015 | 13.796 |
| S | L30 | core-stego | 0.060 | -0.006 ± 0.019 | 3.501 |
| S | L30 | transfer-stego | 0.017 | -0.002 ± 0.018 | 1.067 |

Split-half (within-tier) cosine ceiling, mean ± sd over 10 halvings:

| cfg | layer | core | transfer | stego |
|---|---|---|---|---|
| P | L26 | 0.703 ± 0.013 | 0.551 ± 0.018 | 0.958 ± 0.002 |
| P | L27 | 0.718 ± 0.012 | 0.563 ± 0.018 | 0.960 ± 0.002 |
| P | L28 | 0.701 ± 0.013 | 0.556 ± 0.018 | 0.960 ± 0.003 |
| P | L29 | 0.716 ± 0.012 | 0.563 ± 0.019 | 0.959 ± 0.003 |
| P | L30 | 0.716 ± 0.015 | 0.559 ± 0.021 | 0.958 ± 0.003 |
| S | L26 | 0.732 ± 0.018 | 0.551 ± 0.019 | 0.977 ± 0.002 |
| S | L27 | 0.740 ± 0.016 | 0.563 ± 0.017 | 0.978 ± 0.002 |
| S | L28 | 0.728 ± 0.017 | 0.553 ± 0.016 | 0.980 ± 0.002 |
| S | L29 | 0.735 ± 0.016 | 0.561 ± 0.018 | 0.981 ± 0.002 |
| S | L30 | 0.738 ± 0.017 | 0.557 ± 0.020 | 0.980 ± 0.002 |
