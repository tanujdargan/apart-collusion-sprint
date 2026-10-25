# 7B placement unbundling: results (exploratory; plan notes/unbundle_plan.md; log notes/unbundle_log.md)
Tables: data/unbundle/tables.md · analysis: exp/unbundle_analyze.py (frozen before results) · raw: data/unbundle/runs,
data/unbundle/validation. Cost: see the log; new spend for this extension stayed within the approved $1.00.

## Gates
- Reproduction: re-run none / M2_late / M2_all reproduce the core-matrix C1 and C6 outputs exactly (1.000).
- Hook locations (fold-0 direction, 9 episodes, same container as its matched none):
  - hooked locations ≤ 2.3e-4 residual projection in prefill AND all 11 decode steps; unhooked ≥ 0.008;
  - unhooked upstream locations bitwise identical to the matched none (M2_late emb+L0–17; PRE_LATE emb);
  - EMB_LATE L0–17 differ (downstream of the hooked embedding).
- Earlier "6-fold bitwise" statement, corrected: the only same-pipeline unablated capture is cap_redund_none
  (fold-0 container). Fold-0 is a same-container match; folds 1–5 are CROSS-container comparisons to that capture
  (still bitwise identical at L0–17, which also shows cross-container determinism). No per-fold same-container
  unablated capture exists for folds 1–5.
- Saved acts (full-transcript extraction) vs same-pipeline captures (prompt-only): a deterministic, shape-dependent
  difference.
  - Batch 4 differs (max rel 0.031, padded full length 377); batch 9 is identical (375).
  - Each pipeline reproduces its own earlier output bitwise.
  - The kernel path is not identified.

## Factorial (C1 covert decode; paired, context bootstrap stratified by scheme; n = 306)
| contrast | Δ C1 decode [95% CI] | Δ C6 | Δ recall | Δ ARC (item bootstrap) |
|---|---|---|---|---|
| embedding effect, no pre-onset (EMB_LATE − M2_late) | 0.000 [−0.016, +0.016] | 0.000 | 0.000 | −0.002 [−0.005, 0.000] |
| embedding effect, with pre-onset (M2_all − PRE_LATE) | −0.003 [−0.020, +0.010] | +0.007 [0.000, +0.016] | +0.003 | +0.002 [−0.005, +0.010] |
| pre-onset effect, no embedding (PRE_LATE − M2_late) | **−0.373 [−0.444, −0.304]** | −0.056 [−0.078, −0.033] | −0.029 [−0.052, −0.010] | −0.055 [−0.095, −0.018] |
| pre-onset effect, with embedding (M2_all − EMB_LATE) | **−0.376 [−0.448, −0.307]** | −0.049 [−0.072, −0.029] | −0.026 [−0.046, −0.007] | −0.052 [−0.092, −0.015] |
| interaction | −0.003 [−0.026, +0.016] | +0.007 [0.000, +0.016] | +0.003 | +0.003 [−0.005, +0.012] |
Absolute values:
- C1: none 0.958; M2_late 0.817; EMB_LATE 0.817; PRE_LATE 0.444; M2_all 0.441.
- C6: 1.000 / 1.000 / 1.000 / 0.944 / 0.951.
- ARC: 0.970 / 0.968 / 0.967 / 0.913 / 0.915.
- Parse failures (C1 bet None / decoded None): PRE_LATE 8/43, M2_all 7/42, the others 0/0; C6: none.
- Per fold C1: PRE_LATE 0.47 0.31 0.43 0.45 0.51 0.49 vs M2_all 0.47 0.31 0.43 0.45 0.49 0.49.

## Research verdict (exploratory)
1. Within the original 7B recipe, the extra effect beyond late-layer removal comes from hooking the pre-onset decoder
   layers (L0–17): about −0.37 on covert decoding, plus costs on explicit decisions (≈ −0.05), recall (≈ −0.03) and
   ARC (≈ −0.05).
2. The embedding hook (original layer-0 direction, mismatched location) adds no detectable effect, with or without
   pre-onset removal, and no detectable interaction. The CIs are about ±2 pp, so effects of that size are not
   excluded; this is not a demonstration of equivalence.
3. So the original −0.52 decomposes roughly as −0.14 (late layers, where the state is decodable) and −0.37 (pre-onset
   layers, where held-out state decodability is near chance).
4. These cells use the state directions only, so they do NOT show that the pre-onset effect is specific to the state
   direction. The core matrix's PC2_all (−0.38) was also strongly disruptive, with lower variance, not matched. The
   pre-onset per-layer "state" directions are class-mean contrasts fitted where the state is not linearly decodable
   held-out. What they encode (e.g. topic/scheme features feeding the later computation, or generic high-variance
   structure) is untested.
5. Whether the embedding non-effect relates to the layer-location mismatch is untested.
6. The dm detector's all-layer effect remains placement-dependent; its late-placement effect is untested.

## Message-span findings (free CPU analysis; narrow reading)
Pooled sender-message-span means give low held-out probe accuracy:
- 7B late 0.23, below chance; 14B 0.07; 32B 0.39 with a transient 0.79 near its onset.
Interpretation:
- This does not establish that state information is absent at message positions, and it does not rule out a recovery
  substrate. These are pooled linear probes, not per-token tests.
- Topic remapping across schemes is consistent with the below-chance values; it is not shown to be what the probe uses.
- The 32B span signal is evidence of availability in the unablated model only. It says nothing about survival under
  the hooks or a role in recovery.
