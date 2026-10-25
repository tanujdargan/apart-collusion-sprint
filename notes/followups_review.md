# Critical methods/statistics review of the follow-ups (fable-advisor, 2026-10-25 ~08:15 PDT)
Saved verbatim-in-substance by the coordinator; the reviewer was read-only. Status of each item is tracked at the end.

Summary verdict: fold fits use training rows only for every statistic; nothing selected on evaluated data; fresh runs
use frozen direction files; hook install/remove guarded; batching matched; parse failures counted as wrong. No bug
found that would manufacture the 32B result. Problems: labelling, mechanical rules applied to near-zero effects, one
too-narrow CI.

Issues (ranked):
1. 32B "state SURVIVES" is a rule artefact (global 4.2 pp, fold 8.5 pp driven by fold 4: 0.941→0.588). Gate the
   SURVIVES/ATTENUATED/REPLICATES vocabulary on a global effect ≥ 10 pp with CI excluding 0; else "n/a".
2. 32B robustness needs a decisive check: capture activations under the state ablation; verify ||hQ||/||h|| ≈ 0 at
   every layer and measure re-trained state decodability.
3. Fresh pooled CIs understate uncertainty (the same 306 contexts in all 5 sets): use a cluster bootstrap on orig_id.
4. Message overlap (≈⅓ near-duplicate some original message): report paired Δ on the novel subset (max ratio < 0.9,
   no same-episode exact copy), with that subset's baseline.
5. 14B/32B "global" configs are in-sample; label them in tables and figure legends.
6. 14B PC-2D is a format-breaking perturbation (decoded None 257/306 global; 25% parse failures fold), not a usable
   control; do not use state−PC-2D at 14B as selectivity evidence.
7. Near-zero verdicts (7B pc1d, fresh pc1d, 32B lr) are covered by fix 1.
8. Embedding hook applies the layer-0 direction (fit on decoder-layer-0 output) to the embedding output: inherited
   convention, to be documented.
9. ≈150 uncorrected tests: descriptive only; don't interpret |Δ| < 5 pp or p > 1e-3; the n=5 seed SD is indicative
   (95% CI ≈ [0.6, 2.9]×SD).

Claims to reword:
- 32B state "SURVIVES" → "only 4–8 pp; one fold accounts for most; the 7B/14B collapse does not transfer with this
  fixed recipe; no layer/strength search".
- dm "load-bearing" → "in 7B only (−26 pp); −2.6 pp at 14B, −5 pp at 32B".
- "replicate on fresh messages" → "unchanged under resampling of the sender at T=0.7 for the same contexts; about a
  third of resampled messages duplicate training text".
- 14B "PC-2D control" → "PC-2D perturbation (format breakdown)".
- 14B/32B global numbers → "(in-sample)".
- "pooled n=1530" → "pooled over 5 sets of the same 306 contexts (clustered CI)".
- fresh pc1d/lr → "small (3–5 pp) effect again; pc1d is a DECODE-field formatting effect (bets unchanged)".

What the follow-ups jointly establish:
- 7B: dm and state ablation effects stable to sender resampling (5 seeds, seed SD ≤ 2.5 pp) and held-out schemes.
- 7B: C6 spared for dm; marginal for state; PC-2D never selective.
- 14B: state ablation disrupts covert decoding out of sample (−53 pp) but also C6 (−23 pp), recall, ARC; dm ≈ −2 pp.
- 32B: probe geometry replicates (layer-0 separability, ~0.6-depth state onset, near-orthogonality).
- 32B: no fixed-recipe ablation materially disrupts collusion (≤ 8.5 pp) while ARC drops.
- Cross-model magnitudes are confounded (message set for 14B, depth, single message sets); only per-model directions
  of effects are supported, not a magnitude ordering.

## Status (coordinator)
- 2: DONE before this review landed (V1, exp/verify_32b_ablation.py): residual projection ≤ 0.021% of activation norm
  at all 64 layers; retrained state accuracy L42+ 0.966 → 0.913 ⇒ real redundancy, not a bug.
- 1, 5, 8: applied in exp/loso_analyze.py (logged as a post-hoc reporting amendment; mechanical verdicts kept alongside).
- 3, 4, 1 (fresh): delegated to the fresh-analysis owner (exp/fresh_analyze.py).
- 6, 9 and the rewordings: applied in REPORT_NOTES.md.
