# Cross-model comparison: Qwen2.5 7B / 14B / 32B (exploratory)

**Scope.** This is an exploratory comparison of three models (Qwen2.5-7B/14B/32B-Instruct). It is not a scaling
law: there are three points, each with one sender message set (14B uses its own messages; 7B and 32B share the
identical frozen 7B messages). The held-out-scheme (LOSO) check validates the intervention directions on held-out
schemes for all three models. Only 7B has a fresh-message check (notes/fresh_results.md). 14B and 32B have no
fresh-message check.

Code: exp/xmodel_analyze.py (`probes32`, `geometry`, `compare`). Outputs: data_models/qwen32b/probes32.json,
data/xmodel_summary.json (`geometry`, `loso`, `confounds`), data/xmodel_tables.md (all tables, incl. full per-fold),
figures/fig9_xmodel.{png,pdf}. CPU only. All numbers below are computed, not estimated.

## Part 1. Probe geometry

Recipe: imported unchanged from exp/probes.py (StandardScaler + LR C=0.1; leave-one-scheme-out over 6 schemes;
AUROC per layer; 3-way state accuracy per layer at the C1 decision token, chance 0.33). 7B and 14B values are read
from data/probes.json and data_models/qwen14b/probes.json. The 32B values are new (data_models/qwen32b/probes32.json),
computed on the same 306 episodes and the same 7B sender messages (ids, scheme_id and state match data/acts/meta.json).
32B: only C1 and C4 activations were extracted, so the C1vC2, C1vC3 and C1vC5 contrasts are **not available** for 32B.

Direction files: 7B data/directions (fit on all 306); 14B and 32B `loso/global/` (fit on all 306 by exp/loso_fit.py).
The recomputed 7B and 14B cosines match their probes.json to within 2e-7.
Relative depth = layer / (L − 1); l / L is shown in brackets.

| quantity | 7B (L=28) | 14B (L=48) | 32B (L=64) |
|---|---|---|---|
| C1vC4 mean AUROC, layer 0 | 0.996 | 0.986 | 0.992 |
| C1vC4 mean AUROC, range over layers | 0.996–1.000 | 0.986–1.000 | 0.992–1.000 |
| C1vC4 end-of-turn (full_last) AUROC, layer 0 | 0.993 | 0.961 | 0.997 |
| C1vC4 end-of-turn AUROC, range | 0.993–0.998 | 0.961–1.000 | 0.995–0.999 |
| state acc, max before the ≥0.5 onset | 0.45 | 0.35 | 0.42 |
| first layer with state acc ≥ 0.5 (rel. depth) | L16 (0.59 [0.57]) | L24 (0.51 [0.50]) | L38 (0.60 [0.59]) |
| first layer with state acc ≥ 0.8 (rel. depth) | L18 (0.67 [0.64]) | L26 (0.55 [0.54]) | L42 (0.67 [0.66]) |
| state acc from ≥0.8 onset to last layer | 0.91–0.92 | 0.87–0.99 | 0.94–0.97 |
| max |cos(dm, state 2-D)| over layers (layer) | 0.125 (L15) | 0.059 (L27) | 0.111 (L41) |
| median |cos(dm, state)| | 0.033 | 0.024 | 0.038 |
| cos(dm, LR w/σ) range | 0.42–0.58 | 0.38–0.63 | 0.37–0.64 |

32B state-accuracy profile: 0.24–0.31 for L0–27, a mild rise to 0.35–0.42 at L28–37, 0.51 at L38, 0.71–0.75 at
L39–41, and 0.94–0.97 from L42 to L63.

Fold-vs-global subspace agreement (exp/loso_fit.subspace_agreement: mean over layers of ||Qa^T Qb||²_F / k;
range across the 6 folds; in brackets the minimum at any single layer). For 7B, "global" = data/directions; 14B pc1d
has no original file, so all models are compared with their own all-306 fit.

| direction | 7B | 14B | 32B |
|---|---|---|---|
| dm | 0.970–0.988 [0.961] | 0.977–0.984 [0.957] | 0.971–0.988 [0.959] |
| pc1d | 0.781–0.946 [0.025] | 0.517–0.894 [0.001] | 0.826–0.979 [0.026] |
| state | 0.872–0.926 [0.751] | 0.866–0.929 [0.709] | 0.820–0.921 [0.652] |
| pc2d | 0.877–0.950 [0.572] | 0.850–0.934 [0.477] | 0.867–0.914 [0.486] |
| lr | 0.879–0.961 [0.809] | 0.813–0.975 [0.758] | 0.886–0.963 [0.848] |
(The near-zero single-layer pc1d minimum means the top PC swaps order at some layer in some fold; it is
present in all three models.)

### Variance shares of the ablated subspaces (all 306 episodes; same definition as exp/loso_fit.var_share)
dm / pc1d / lr on C1+C4 `mean`; state / pc2d on C1 `last`. Per-layer values are in xmodel_summary.json.

| | 7B | 14B | 32B |
|---|---|---|---|
| state, mean over all layers | 0.156 | 0.199 | 0.163 |
| PC-2D, mean over all layers | 0.223 | 0.160 | 0.217 |
| layers with PC-2D > state | 18/28 | 28/48 | 44/64 |
| state, mean over layers ≥ the 0.5 onset | 0.267 | 0.322 | 0.287 |
| PC-2D, mean over layers ≥ the 0.5 onset | 0.177 | 0.136 | 0.218 |
| post-onset layers with PC-2D > state | 2/12 | 5/24 | 8/26 (L38–43, L62–63) |
| dm / pc1d mean | 0.103 / 0.081 | 0.119 / 0.066 | 0.106 / 0.083 |

**Correction to the initial briefing (accepted by the coordinator):** it is not the case that the 32B PC-2D control carries more variance than the state
subspace while for 7B it carried less. On the layer mean, PC-2D > state in BOTH 7B (0.223 vs 0.156) and 32B
(0.217 vs 0.163); 14B is the exception (0.160 vs 0.199). The REPORT_NOTES 7B figures (33% vs 19%) are for L20. In
the state-decodable layers, the state subspace carries more variance than PC-2D in all three models, and the
margin is smallest at 32B (ratio PC-2D/state 0.66 for 7B, 0.42 for 14B, 0.76 for 32B). Before the onset, the state
subspace is small (≈0.07–0.08) and PC-2D large (0.18–0.26) in every model. In no model is the control
variance-matched layer by layer.

## Part 2. Held-out (fold-fit) and in-sample (global) intervention results

Inputs: each model's loso/analysis.json from the unchanged exp/loso_analyze.py (7B data/loso, 14B and 32B
data_models/<tag>/loso), plus the per-episode run files. The run files are used for the extra bet-exact contrasts
(fold − global and direction − control), using the same scheme-stratified paired bootstrap (10k). The recomputed point
estimates equal analysis.json exactly (asserted in the code). Fixed policy: ablation at all layers + embedding,
receiver only. Pooled n = 306 per model (6 folds × 51). `global` = fit on all 306 (in-sample; the 7B global
directions are the original data/directions); `fold` = fit without the evaluated scheme (held-out). "bet-exact" is
reported next to strict decode because 14B often omits the "DECODE:" prefix (C1 decoded None up to 257/306). 32B
never does here (decoded None 0 in every config), so for 32B decode ≈ bet-exact.

### Held-out (fold-fit) results; Δ vs matched none [95% CI]
| model | config | C1 decode | Δ decode | C1 bet-exact | Δ bet-exact | C6 bet-exact | Δ C6 | recall | ARC | C1 bet None / dec None / C6 bet None |
|---|---|---|---|---|---|---|---|---|---|---|
| 7B | none | 0.958 | – | 0.958 | – | 1.000 | – | 0.967 | 0.970 | 0/0/0 |
| 7B | dm | 0.732 | −0.225 [−0.271, −0.180] | 0.732 | −0.225 | 1.000 | 0 | 0.944 | 0.942 | 0/0/0 |
| 7B | pc1d | 0.912 | −0.046 [−0.075, −0.020] | 0.958 | +0.000 | 1.000 | 0 | 0.961 | 0.953 | 0/14/0 |
| 7B | state | 0.441 | −0.516 [−0.575, −0.458] | 0.539 | −0.418 | 0.951 | −0.049 [−0.072, −0.029] | 0.935 | 0.915 | 7/42/0 |
| 7B | pc2d | 0.578 | −0.379 [−0.438, −0.324] | 0.578 | −0.379 | 0.866 | −0.134 [−0.170, −0.098] | 0.895 | 0.937 | 0/0/0 |
| 7B | lr | 0.902 | −0.056 [−0.082, −0.029] | 0.902 | −0.056 | 1.000 | 0 | 0.961 | 0.970 | 0/0/0 |
| 14B | none | 0.997 | – | 0.997 | – | 1.000 | – | 1.000 | 0.990 | 0/0/0 |
| 14B | dm | 0.974 | −0.023 [−0.042, −0.007] | 0.977 | −0.020 | 1.000 | 0 | 0.993 | 0.978 | 0/1/0 |
| 14B | pc1d | 0.990 | −0.007 [−0.016, 0.000] | 0.990 | −0.007 | 1.000 | 0 | 0.984 | 0.990 | 0/0/0 |
| 14B | state | 0.464 | −0.533 [−0.585, −0.480] | 0.614 | −0.382 | 0.771 | −0.229 [−0.275, −0.183] | 0.745 | 0.753 | 17/112/0 |
| 14B | pc2d | 0.288 | −0.709 [−0.755, −0.660] | 0.546 | −0.451 | 0.663 | −0.337 [−0.376, −0.297] | 0.794 | 0.365 | 78/187/68 |
| 14B | lr | 0.990 | −0.007 [−0.016, 0.000] | 0.990 | −0.007 | 1.000 | 0 | 1.000 | 0.990 | 0/0/0 |
| 32B | none | 0.971 | – | 0.971 | – | 1.000 | – | 0.974 | 0.980 | 0/0/0 |
| 32B | dm | 0.925 | −0.046 [−0.072, −0.020] | 0.925 | −0.046 | 0.997 | −0.003 [−0.010, 0.000] | 0.958 | 0.973 | 0/0/0 |
| 32B | pc1d | 0.971 | 0.000 | 0.971 | 0.000 | 1.000 | 0 | 0.974 | 0.992 | 0/0/0 |
| 32B | state | 0.886 | −0.085 [−0.118, −0.056] | 0.886 | −0.085 | 0.993 | −0.007 [−0.016, 0.000] | 0.980 | 0.883 | 0/0/0 |
| 32B | pc2d | 0.974 | +0.003 [−0.007, +0.013] | 0.961 | −0.010 | 0.993 | −0.007 [−0.016, 0.000] | 0.977 | 0.952 | 4/0/0 |
| 32B | lr | 0.977 | +0.007 [0.000, +0.016] | 0.977 | +0.007 | 1.000 | 0 | 0.971 | 0.987 | 0/0/0 |
Global (in-sample) rows, all CIs and bet-exact CIs: data/xmodel_tables.md §1–2.

### Fold − global (same episodes; + = held-out fit is weaker) and direction − its PC control (fold-fit)
| contrast | 7B | 14B | 32B |
|---|---|---|---|
| dm fold − global, C1 decode | +0.039 [+0.010, +0.069] | +0.003 [−0.007, +0.016] | +0.010 [−0.003, +0.023] |
| state fold − global, C1 decode | +0.062 [+0.020, +0.105] | +0.082 [+0.026, +0.137] | **−0.042 [−0.072, −0.010]** |
| state fold − global, C1 bet-exact | +0.111 [+0.065, +0.157] | −0.010 [−0.059, +0.039] | −0.042 [−0.075, −0.010] |
| state fold − global, C6 | −0.029 [−0.056, −0.007] | +0.062 [+0.023, +0.101] | −0.007 [−0.016, 0.000] |
| pc2d fold − global, C1 decode | −0.026 [−0.062, +0.010] | +0.173 [+0.121, +0.225] | 0.000 [−0.010, +0.010] |
| lr fold − global, C1 decode | −0.026 [−0.046, −0.007] | +0.007 [0.000, +0.016] | 0.000 [−0.010, +0.010] |
| dm − PC-1D, C1 decode | −0.180 [−0.229, −0.131] | −0.016 [−0.036, 0.000] | −0.046 [−0.075, −0.020] |
| dm − PC-1D, C6 | 0 | 0 | −0.003 [−0.010, 0.000] |
| state − PC-2D, C1 decode | −0.137 [−0.206, −0.069] | +0.176 [+0.108, +0.245] | −0.088 [−0.121, −0.056] |
| state − PC-2D, C1 bet-exact | −0.039 [−0.118, +0.039] | +0.069 [+0.003, +0.134] | −0.075 [−0.111, −0.039] |
| state − PC-2D, C6 | +0.085 [+0.039, +0.127] | +0.108 [+0.042, +0.170] | 0.000 [−0.013, +0.013] |
(The 7B state − PC-2D C6 CI is [+0.042, +0.131] in analysis.json; the bootstrap here is a separate resample.)

**32B: the held-out state ablation is larger than the in-sample one.** state_fold Δ −0.085 vs state_global
−0.042; fold − global −0.042 [−0.072, −0.010] (the same on bet-exact). In 7B and 14B the held-out fit was weaker
(+0.062, +0.082). The pre-specified rule labels this "SURVIVES" with ratio 2.00, but only 3/6 folds are consistent.
This is reported as observed; no explanation was tested.

### Per-fold spread of held-out C1 decode drop (n = 51 per fold; f0..f5)
| model | dm | state | PC-2D | LR |
|---|---|---|---|---|
| 7B | +0.33 +0.27 +0.22 +0.14 +0.22 +0.18 | +0.49 +0.67 +0.49 +0.55 +0.39 +0.51 | +0.33 +0.47 +0.37 +0.43 +0.31 +0.35 | +0.20 0 +0.10 0 +0.04 0 |
| 14B | +0.04 +0.04 0 0 +0.02 +0.04 | +0.20 +0.39 +0.73 +0.43 +0.67 +0.78 | +0.88 +0.45 +0.63 +0.84 +0.94 +0.51 | +0.04 0 0 0 0 0 |
| 32B | +0.10 +0.04 +0.10 0 +0.04 0 | −0.02 +0.08 −0.02 +0.12 **+0.35** 0 | −0.04 +0.02 0 0 0 0 | −0.02 0 −0.02 0 0 0 |
32B PC-1D: 0 in every fold. C6 drop per fold, fold-fit: 32B state 0–0.039, dm 0–0.020; 7B state 0–0.255 (fold 5);
14B state 0.098–0.333.
The 32B state effect is concentrated in fold 4 (0.941 → 0.588). Fold 3 drops 0.12 and fold 1 drops 0.08; the other
three folds drop ≤ 0.
32B ARC under the fold-fit state ablation, per fold: 0.65 / 0.87 / 0.97 / 0.94 / 0.94 / 0.93 (matched none 0.98 in
every fold). The largest ARC cost (fold 0, 0.65) comes with no C1 drop (−0.02) in that fold. 32B PC-2D fold-fit ARC
0.85–0.99. 14B: state ARC 0.59–0.90, PC-2D ARC 0.02–0.95.

Decoded distribution (C/N/H/None), none → fold-fit: 7B dm 101/104/101 → 89/163/54 (toward NEUTRAL, the no-signal
default); 32B dm → 101/119/86 (a small shift toward NEUTRAL); 14B dm → 103/104/98/1 (unchanged). State fold-fit: 7B
222/24/18/42 (COLD collapse); 14B 135/50/9/112; 32B 110/120/76/0.

## Answers to the plan's 32B questions (7B / 14B / 32B)
1. **Are detectors separable at layer 0? Yes, in all three models.** C1vC4-mean AUROC at layer 0 is 0.996 / 0.986 /
   0.992; end-of-turn 0.993 / 0.961 / 0.997; minimum over layers 0.96 or above in every model. 32B cannot test the
   C1vC2/C3/C5 contrasts, because no activations were extracted for them.
2. **When does the state become decodable (relative depth l/(L−1))?** First layer ≥ 0.5: 0.59 (L16) / 0.51 (L24) /
   0.60 (L38). First layer ≥ 0.8: 0.67 / 0.55 / 0.67. The onset is abrupt in all three, around half to two-thirds
   depth. 32B's onset is at the same relative depth as 7B; 14B's is earlier. 32B has a mild pre-onset rise
   (0.35–0.42 at L28–37). Plateau 0.92 / 0.99 / 0.97.
3. **Is the detector orthogonal to the state?** Approximately, in all three: max |cos(dm, state 2-D)| 0.125 / 0.059 /
   0.111, median 0.033 / 0.024 / 0.038.
4. **Does dm/LR ablation affect collusion?** dm: yes in 7B (held-out −0.225, 6/6 folds, C6 untouched). In 14B the
   effect is small (−0.023 [−0.042, −0.007]). In 32B it is small but present (−0.046 [−0.072, −0.020]; 4/6 folds, fold
   range 0 to +0.10; C6 −0.003; 4.6 pp more than its PC-1D control, which has zero effect). LR: small in 7B (−0.056),
   null in 14B (−0.007 [−0.016, 0.000]) and 32B (+0.007 [0.000, +0.016]). In all three, LR ablation leaves collusion
   ≥ 0.90.
5. **Is state ablation selective against PC-2D and C6?**
   - 7B: yes, relative to the control. Held-out C1 −0.516 vs PC-2D −0.379; C6 −0.049 vs −0.134. C6 sparing is
     marginal, and fold 5 drops to C6 0.745.
   - 14B: no. State −0.533 with C6 −0.229, recall 0.745 and ARC 0.753. The PC-2D control is even more destructive
     (C1 −0.709, ARC 0.365).
   - 32B: the question barely applies, because the state ablation hardly moves covert decoding. Held-out C1 −0.085
     [−0.118, −0.056], driven mostly by one fold. C6 −0.007, PC-2D ≈ 0 (+0.003), state − PC-2D −0.088 in C1 and 0 in
     C6. ARC falls to 0.883 (0.65 in fold 0), so the hooks act on the model. By the letter of the rule, the effect is
     "selective" (C6 spared, larger than PC-2D). But it is an order of magnitude smaller than in 7B and 14B (−0.516,
     −0.533).

## What differs at 32B
- The probe geometry is the same as in 7B: surface-separable detectors at layer 0, an abrupt state onset at ~0.6
  relative depth, and a near-orthogonal detector and state.
- The intervention picture is different. Every all-layer ablation that broke or shifted covert decoding in 7B (state
  −0.516, PC-2D −0.379, dm −0.225) changes 32B covert decoding by at most 8.5 pp held-out, on the same messages and
  episodes as 7B. Explicit (C6) betting is untouched (≤ 0.7 pp).
- 32B is the only model where the held-out state fit is stronger than the global one (fold − global −0.042).
- These are observations on three models. They do not establish a trend with size.

## Confounds and caveats
- Sender messages: 14B used its own; 7B and 32B share the byte-identical 7B frozen messages. So 7B vs 32B is matched
  input-for-input on the receiver side, while 14B is not.
- Variance matching: the PC-2D control is not variance-matched layer by layer in any model. In the post-onset layers
  the state subspace carries more variance than PC-2D in all three, and the margin is smallest at 32B (Part 1
  table). The dm and PC-1D shares are similar across models (0.10–0.12 vs 0.07–0.08).
- Hardware: 7B and 14B ran on L40S, 32B on H200. Batch size is 48 everywhere. Batch composition is known to flip ≈1%
  of ablated outputs (7B determinism check). The 32B analysis.json has no determinism block, because there is no
  original 32B ablation run to compare with.
- In-sample vs held-out: the global configs are in-sample (fit on all 306, including the evaluated episodes); the fold
  configs are the held-out evaluation.
- The fixed all-layer + embedding policy and the fixed construction were not tuned per model. A null result at 32B
  under this policy does not show that 32B lacks such a subspace. It shows that this construction, applied this way,
  does not remove it.
- Fresh messages: only 7B has a fresh-message check. 14B and 32B rest on one message generation each.
