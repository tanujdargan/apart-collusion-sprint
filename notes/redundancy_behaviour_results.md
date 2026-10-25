# Core behavioural matrix: results (exploratory; plan notes/redundancy_plan.md §5; log notes/redundancy_log.md)
Full tables: data/redund/tables.md · raw: data/redund/{7b,32b}/runs, caps · analysis: exp/redund_behaviour_analyze.py ·
figure: figures/fig12_placement. Cost this stage ≈ $1.91 (billing at analysis time; may lag slightly), hard stop $6, no resources left running.

Design recap: held-out fold directions (M2 = fold state span; PC2 = top-2 PCs ⊥ M2, LOWER variance, not matched), the
frozen 7B message set, 6 folds × 51 = 306 held-out episodes per model. `all` = all decoder layers + embedding hook
(original recipe); `late` = decoder layers ≥ ceil(2(L−1)/3), no embedding hook. Placement contrasts compare BUNDLED
recipes (layer set and embedding hook change together); they are not an isolated embedding-hook test.
Verification: matched baseline / M2_all / PC2_all reproduce the earlier LOSO outputs exactly (1.000). Projection
ratio onto the fold span ≈ 0.0001 at hooked layers and baseline-like at unhooked layers (7B early 0.162 vs baseline
0.162; 32B 0.139 vs 0.139). Same-pipeline unablated capture vs saved acts: max rel diff 0.031 (7B) / 0.037 (32B).

## Behaviour (C1 covert decode Δ vs matched none, paired, scheme-stratified bootstrap 95% CI)
| | 7B | 32B |
|---|---|---|
| M2_all (original recipe) | −0.516 [−0.575, −0.458] | −0.085 [−0.118, −0.052] |
| M2_late | −0.141 [−0.186, −0.098] | +0.007 [0.000, +0.016] |
| PC2_all | −0.379 [−0.438, −0.320] | +0.003 [−0.007, +0.013] |
| PC2_late | −0.059 [−0.088, −0.033] | 0.000 [−0.010, +0.010] |
| M2_late − M2_all (bundled recipe contrast) | +0.376 [+0.304, +0.448] | +0.092 [+0.062, +0.124] |
| M2_late − PC2_late | −0.082 [−0.134, −0.029] | +0.007 [0.000, +0.016] |
| M2_all − PC2_all | −0.137 [−0.206, −0.069] | −0.088 [−0.121, −0.056] |
Task/control performance:
- 7B: C6 explicit bet-exact none 1.000 / M2_all 0.951 / M2_late 1.000 / PC2_all 0.866 / PC2_late 1.000;
  recall 0.967 / 0.935 / 0.961 / 0.895 / 0.961; ARC 0.970 / 0.915 / 0.968 / 0.937 / 0.965.
- 32B: C6 1.000 / 0.993 / 1.000 / 0.993 / 1.000; recall 0.974 / 0.980 / 0.971 / 0.977 / 0.971; ARC 0.980 / 0.883
  (fold 0: 0.65) / 0.998 / 0.952 / 1.000.
Parse failures:
- 7B: only M2_all (C1 bet None 7, decoded None 42).
- 32B: only PC2_all (C1 bet None 4).
- C6: none in either model.
Decoded distribution (C/N/H): 7B M2_all 222/24/18 (COLD default), M2_late 80/76/150 (shift toward HOT), PC2_all 102/201/3.

## Ablated-pass state readout (decision token; LR retrained on the ablated activations of the 255 training episodes,
## tested on the 51 held-out; late-layer mean, n = 306)
| | 7B | 32B |
|---|---|---|
| unablated (same pipeline) | 0.918 [0.890, 0.944] | 0.967 [0.947, 0.984] |
| retrained, M2_all | 0.697 (Δ −0.221 [−0.263, −0.181]) | 0.922 (Δ −0.045 [−0.063, −0.028]) |
| retrained, M2_late | 0.832 (Δ −0.086 [−0.108, −0.065]) | 0.939 (Δ −0.028 [−0.035, −0.022]) |
| frozen unablated probe, M2_all / M2_late | 0.333 / 0.340 | 0.320 / 0.346 |
Per layer (M2_late, retrained):
- 7B L17 0.75 → **L18 (first hooked) 0.35** → L19 0.88 → L20 0.91 … L27 0.85.
- 32B L41 0.72 → **L42 0.42** → L43 0.96 … L63 0.97.
- The low value at the first hooked layer is largely by construction: the captured output there is a static
  projection of the episodes the span was fitted on, cf. the degeneracy in notes/redundancy_log.md.
- From the next layer on, activations computed under ablation again carry readable state in directions the
  unablated probe does not use (frozen ≈ chance).

## What this does and does not show (exploratory)
1. **Recipe dependence (bundled).** In both models the post-onset-only removal of the state span has a much smaller
   behavioural effect than the original all-layer + embedding recipe:
   - 7B: −0.14 vs −0.52.
   - 32B: +0.01 vs −0.085.
   Most of the original 7B effect, and all of the 32B effect, therefore depends on also hooking pre-onset layers
   and/or the embedding, where the per-layer "state" directions were fitted on near-chance data. Which of the two
   (pre-onset layers vs embedding) matters is NOT separable here.
2. **State vs control at the late placement.**
   - 7B: the state span is somewhat more disruptive than the lower-variance PC control (Δ −0.082 [−0.134, −0.029]),
     with C6, recall and ARC intact.
   - 32B: no detectable difference, CI [0.000, +0.016]. This is not evidence of equivalence; small effects within
     the CI are not excluded.
3. **The 7B-vs-32B contrast in the original recipe** (−0.52 vs −0.085) is mostly a contrast in how the two models
   respond to the bundled pre-onset/embedding component, not a contrast in the effect of removing the state span
   where the state is decodable (late: −0.14 vs +0.01; both smaller).
4. **Readout under ablation.** Within one block after the removal starts, both models again carry linearly
   decodable state in new directions (retrained ≈ 0.88–0.96; frozen ≈ chance).
   - This is consistent with downstream re-derivation of the state, e.g. from token positions or directions the
     hook does not remove. It does NOT prove reconstruction: the information could also be carried by
     untouched components and merely become linearly readable.
   - The retained readout is not a measure of information retained.
   - In 7B, behaviour still drops 14 points while readout recovers to ≈ 0.9; in 32B both stay high. Readout
     recovery and behavioural robustness therefore do not coincide in 7B.
5. **Limits.** One message set; 7B and 32B only; one fixed late cutoff (no layer search); controls are lower-variance;
   PC2_all in 7B is itself strongly disruptive, so the original "state − control" comparison mixed placement effects.
   McNemar/bootstrap CIs are episode-level within these frozen folds.
