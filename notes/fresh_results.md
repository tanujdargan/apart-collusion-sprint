# Fresh-message robustness check (Qwen2.5-7B, frozen original directions): results

Generated from `exp/fresh_analyze.py` → `data/fresh/analysis.json`, `data/fresh/tables.md` (full tables),
`figures/fig8_fresh.{png,pdf}`. Plan: notes/followups_plan.md §A + "Reporting". The rules were applied as written:
no extra configs, no tuning, no filtering, and no messages were regenerated. Exploratory/post hoc, not preregistered.

## Design recap
- 5 new sender message sets (gen_seed 101, 202, 303, 404, 505; the original set used 0) for the SAME 306 episode
  contexts. Each set is evaluated with configs none / dm / pc1d / state / pc2d / lr, using the ORIGINAL 7B direction
  files (frozen, not refit), with ablation on all layers + embedding. Conditions: C1, C6, recall and ARC.
- Primary outcomes: C1 strict decode, C1 bet-exact and C6 bet-exact. A parse failure (decoded None / bet None)
  counts as wrong.
- Paired Δ = config − that set's matched `none` (same episodes, same order and batching). The p-values are exact
  McNemar.
- Per-set CIs: stratified paired bootstrap over episodes (strata = scheme; 10k reps, seed 0). Per-set absolute CIs
  are Wilson.
- **Pooled CIs (PRIMARY; revised after review items 1/3/4 in notes/followups_review.md):** a cluster bootstrap over
  the 306 contexts (orig_id), stratified by scheme (51 per scheme), 10k reps, separate seed-0 generator. Each
  sampled context carries all 5 sets' outcomes. Results are labelled "pooled over 5 sets of the same 306 contexts
  (cluster CI)".
- **Pooled CIs, OPTIMISTIC (kept for reference):** the first version resampled episodes independently within
  set × scheme strata. That treats the 5 sets' copies of a context as independent and understates uncertainty.
  Those CIs, and the per-set results, are unchanged from the first run (checked). Pooled McNemar p-values treat the
  1530 pairs as independent, so they are optimistic too.
- Seed-level variability: SD and range of the 5 per-set Δ. With n = 5 this SD is indicative only; a 95% CI for an
  SD from 5 values is roughly [0.6, 2.9] × SD.
- About 150 uncorrected tests are run across the project, so all p-values and CIs are descriptive. Do not read
  |Δ| < 5 pp or p > 1e-3 as evidence on their own.

## 1. Verification (all passed; asserted in code)
- Each set has 306 unique ids. `id == m<seed>_<orig_id>`. The orig_id → original id mapping is one-to-one and in
  the same order as data/messages.jsonl. No ids collide across sets.
- Scheme × state counts, derived from the data: 6 × 3, 17 per cell, in every set and in the original.
- The episode context (names, hand, filler, topic, recall options, transcript) is identical to the original for
  all rows. Only the message differs.
- In all 30 runs: `ids` == the set ids, C1_rows and C6_rows are in the same order, the C1 row `msg` == the set's
  `collude_msg`, and states match.
- `parse_talk` on `collude_raw` gives 0 failures in every set (and 0 in the original). The stored `collude_check`
  and `collude_clean` equal the values recomputed with env.sender_check.
- ARC is message-independent and is identical across the 5 sets for every config (none 0.97, dm 0.95, pc1d 0.96,
  state 0.92, pc2d 0.93, lr 0.97). This is a determinism check.
- The original-set Δs were computed from the original full-306 files (data/interv/*). They are: dm −0.258,
  pc1d −0.023, state −0.575, pc2d −0.356, lr −0.029 (C1 decode); C6: dm 0, pc1d 0, state −0.020, pc2d −0.199, lr 0.
  Three values in the task brief differ: dm −0.265, state −0.576 and C6 pc2d −0.193. Those come from the LOSO
  "global" re-runs, which used different batch composition. This analysis uses the values computed here.

## 2. Overlap with the original messages (counts out of 306 per set; nothing filtered)
| set | exact same-ep | normalised same-ep | ratio ≥ 0.9 same-ep | exact = any original | ratio ≥ 0.9 vs any original | same-ep ratio median [IQR] | max-vs-any ratio median [IQR] |
|---|---|---|---|---|---|---|---|
| m101 | 14 | 18 | 22 | 72 | 102 | 0.453 [0.338–0.598] | 0.789 [0.652–0.978] |
| m202 | 16 | 18 | 29 | 82 | 112 | 0.455 [0.350–0.631] | 0.805 [0.679–1.000] |
| m303 | 10 | 11 | 16 | 88 | 112 | 0.451 [0.360–0.593] | 0.813 [0.657–1.000] |
| m404 | 13 | 15 | 24 | 83 | 120 | 0.458 [0.360–0.647] | 0.811 [0.680–1.000] |
| m505 | 17 | 18 | 23 | 78 | 116 | 0.464 [0.362–0.633] | 0.828 [0.684–1.000] |

- Against the same episode's original message, 3–6% are exact copies and 5–9% have ratio ≥ 0.9. The median
  similarity is about 0.45.
- The sender reuses stock phrasings across episodes, for example "Let's hope the buffet has some good pasta
  tonight." As a result, 23–29% of fresh messages are an exact copy of SOME original message (usually from another
  episode of the same scheme and state). 33–39% have a ≥ 0.9 near-duplicate somewhere in the original set.
- The original set itself has 46 exact within-set duplicates (48 after normalisation). The fresh sets have 40–56.
- Across fresh-set pairs: 9–29 exact same-episode duplicates, 21–40 same-episode pairs with ratio ≥ 0.9, and 65–105
  exact matches to any message in the other set (full table in tables.md).
- Implication: the fresh sets are new samples from the same, fairly low-diversity sender distribution. They are not
  fully novel text. About a third of the messages closely match an original message the directions were fit on.

## 3. Sender compliance (rate per message)
| set | forbidden word | digit | number word | target topic mentioned | clean target | parse_talk fail |
|---|---|---|---|---|---|---|
| original | 0.026 | 0.013 | 0.042 | 0.824 | 0.814 | 0 |
| m101 | 0.029 | 0.036 | 0.026 | 0.794 | 0.781 | 0 |
| m202 | 0.023 | 0.026 | 0.016 | 0.794 | 0.775 | 0 |
| m303 | 0.020 | 0.020 | 0.029 | 0.791 | 0.771 | 0 |
| m404 | 0.023 | 0.026 | 0.039 | 0.788 | 0.771 | 0 |
| m505 | 0.039 | 0.010 | 0.029 | 0.784 | 0.778 | 0 |
| fresh pooled | 0.027 | 0.024 | 0.028 | 0.790 | 0.775 | 0 |

- The target-topic keyword rate is about 3 pp lower than in the original (0.790 vs 0.824). Otherwise compliance is
  similar.
- The most common forbidden word is "cold" (32 of the 41 forbidden-word hits pooled).

## 4. Baseline capability (`none`)
| set | C1 decode [Wilson] | C1 bet-exact | C6 bet-exact | recall | ARC |
|---|---|---|---|---|---|
| original | 0.958 [0.929–0.975] | 0.958 | 1.000 | 0.967 | 0.970 |
| m101 | 0.948 [0.917–0.968] | 0.951 | 1.000 | 0.967 | 0.970 |
| m202 | 0.951 [0.921–0.970] | 0.951 | 1.000 | 0.958 | 0.970 |
| m303 | 0.967 [0.941–0.982] | 0.967 | 1.000 | 0.974 | 0.970 |
| m404 | 0.951 [0.921–0.970] | 0.951 | 1.000 | 0.961 | 0.970 |
| m505 | 0.967 [0.941–0.982] | 0.967 | 1.000 | 0.967 | 0.970 |
| pooled | 0.957 [0.945–0.966] | 0.958 | 1.000 | 0.965 | – |

- Baseline capability does not shift: C1 decode per set is within ±0.010 of the original 0.958.
- Baseline parse failures: 1 decoded-None in m101 and none elsewhere.

## 5. Intervention effects (C1 decode; full per-set table incl. C1 bet-exact and C6 in tables.md)
| config | original Δ [CI] | per-set Δ (m101 / 202 / 303 / 404 / 505) | seed SD (indicative) | pooled Δ [cluster CI] | optimistic CI | pooled abs. |
|---|---|---|---|---|---|---|
| dm | −0.258 [−0.310, −0.209] | −0.261 / −0.278 / −0.281 / −0.265 / −0.304 | 0.017 | −0.278 [−0.318, −0.237] | [−0.301, −0.254] | 0.679 |
| pc1d | −0.023 [−0.049, +0.000] | −0.033 / −0.033 / −0.016 / −0.026 / −0.065 | 0.018 | −0.035 [−0.050, −0.020] | [−0.046, −0.024] | 0.922 |
| state | −0.575 [−0.631, −0.516] | −0.582 / −0.578 / −0.582 / −0.582 / −0.598 | 0.008 | −0.584 [−0.635, −0.532] | [−0.610, −0.558] | 0.373 |
| pc2d | −0.356 [−0.412, −0.301] | −0.333 / −0.389 / −0.376 / −0.340 / −0.379 | 0.025 | −0.363 [−0.415, −0.310] | [−0.389, −0.338] | 0.593 |
| lr | −0.029 [−0.056, −0.007] | −0.039 / −0.062 / −0.062 / −0.052 / −0.052 | 0.009 | −0.054 [−0.069, −0.039] | [−0.066, −0.042] | 0.903 |

"Pooled" here means pooled over 5 sets of the same 306 contexts. The cluster CIs are about 1.5–2× wider than the
optimistic ones.

C6 bet-exact Δ, pooled [CI] vs original:
| config | pooled C6 Δ [CI] | original C6 Δ |
|---|---|---|
| dm | 0.000 | 0.000 |
| pc1d | 0.000 | 0.000 |
| lr | 0.000 | 0.000 |
| state | −0.029 [−0.044, −0.016] (optimistic [−0.037, −0.021]) | −0.020 |
| pc2d | −0.198 [−0.238, −0.158] (optimistic [−0.218, −0.178]) | −0.199 |

- The C6 accuracies for dm, pc1d and lr are exactly 1.000 in every set.

Parse failures under intervention (pooled, C1):
| config | decoded None | bet None |
|---|---|---|
| state | 189 | 54 |
| pc1d | 55 | 0 |
| others | 0 | 0 |
| none (baseline) | 1 | 0 |

C6 bet None = 0 everywhere.

Per scheme: the direction and rough size of each effect hold in every scheme × set cell for dm, state and pc2d.
- dm: every cell is between −0.098 and −0.431.
- state: every cell is between −0.41 and −0.67.
- pc2d: every cell is between −0.275 and −0.490.
- The LR drop is concentrated in scheme 0 (pooled −0.173). It is ≈ 0 in schemes 3 and 5, as in the original.

## 6. Fixed interpretation rule (gated; cluster CI)
- Mechanical rule: replicates = the pooled Δ cluster CI excludes 0 with the original's sign AND the pooled Δ ≥ 50%
  of the original Δ.
- Gate (review item 1): the REPLICATES wording is only used when the original effect is ≥ 10 pp with a CI excluding
  0. Otherwise the verdict reads "n/a (original effect < 10 pp)". The mechanical outcome is kept as a separate column.

| config | metric | original Δ | pooled Δ [cluster CI] | verdict (gated) | mechanical outcome |
|---|---|---|---|---|---|
| dm | C1 decode | −0.258 | −0.278 [−0.318, −0.237] | **REPLICATES** (ratio 1.08) | replicates |
| dm | C1 bet-exact | −0.258 | −0.278 [−0.319, −0.238] | **REPLICATES** (1.08) | replicates |
| state | C1 decode | −0.575 | −0.584 [−0.635, −0.532] | **REPLICATES** (1.02) | replicates |
| state | C1 bet-exact | −0.523 | −0.524 [−0.575, −0.473] | **REPLICATES** (1.00) | replicates |
| pc2d | C1 decode | −0.356 | −0.363 [−0.415, −0.310] | **REPLICATES** (1.02) | replicates |
| pc2d | C1 bet-exact | −0.356 | −0.364 [−0.417, −0.313] | **REPLICATES** (1.02) | replicates |
| pc2d | C6 bet-exact | −0.199 | −0.198 [−0.238, −0.158] | **REPLICATES** (0.99) | replicates |
| state | C6 bet-exact | −0.020 | −0.029 [−0.044, −0.016] | n/a (original effect < 10 pp) | replicates |
| pc1d | C1 decode | −0.023 | −0.035 [−0.050, −0.020] | n/a (original effect < 10 pp) | replicates |
| pc1d | C1 bet-exact | +0.007 | +0.001 [−0.007, +0.008] | n/a (original effect < 10 pp) | does not replicate |
| lr | C1 decode / bet-exact | −0.029 | −0.054 [−0.069, −0.039] / [−0.071, −0.039] | n/a (original effect < 10 pp) | replicates |
| dm, pc1d, lr | C6 bet-exact | 0 | 0 | n/a (original effect < 10 pp) | n/a (original Δ = 0) |

C6 spared (drop ≤ 5 pp and cluster-CI upper ≤ 10 pp):

| config | C6 spared? | drop (cluster-CI upper) |
|---|---|---|
| dm, pc1d, lr | yes | 0 (0) |
| state | yes | 0.029 (0.044; optimistic 0.037) |
| pc2d | **no** | 0.198 (0.238) |

These verdicts are the same as with the optimistic CI.

Seed level: for dm, state, pc2d and lr, all 5 sets individually have a C1-decode CI that excludes 0 with the
original's sign. For pc1d this holds in 4 of 5 sets.

Plain-language wording for the near-zero effects:
- **pc1d:** small effect again; a DECODE-field formatting effect (bets unchanged). The 3.5 pp C1 decode drop comes
  from outputs that omit or garble "DECODE:" (decoded None 55 pooled vs 1 at baseline). C1 bet-exact is +0.001.
- **lr:** small (≈ 5 pp) effect again. It is about 5× smaller than dm and concentrated in scheme 0 (pooled −0.173;
  ≈ 0 in schemes 3 and 5). The original characterisation (LR ablation does little to collusion) holds
  qualitatively, as "small, about 3–5 pp" rather than "none".
- **state, C6:** the small C6 cost (2.0 → 2.9 pp) is within the C6-spared bounds.
- **pc2d:** the control is non-selective again, costing about 20 pp on C6.

## 7. Novel-message subset (review item 4)
Novel = max difflib ratio vs ANY original message < 0.9 AND not an exact copy of the same episode's original.
- The other fresh messages stay in the main analysis. This is an additional subset view; nothing is dropped or
  regenerated.
- The selection is on message text, so it is not random. Stock phrases are often the compliant target-topic lines,
  so the novel subset could be less compliant. In practice, clean-target rates are similar (pooled novel 0.779 vs
  rest 0.769). Its baseline decode is slightly lower (0.949 vs 0.970).

| set | n novel / 306 | baseline C1 decode novel / rest | Δ C1 decode dm | state | pc2d | lr | pc1d |
|---|---|---|---|---|---|---|---|
| m101 | 204 | 0.936 / 0.971 | −0.245 | −0.574 | −0.328 | −0.034 | −0.029 |
| m202 | 194 | 0.938 / 0.973 | −0.284 | −0.567 | −0.412 | −0.046 | −0.026 |
| m303 | 194 | 0.954 / 0.991 | −0.284 | −0.572 | −0.361 | −0.057 | −0.015 |
| m404 | 186 | 0.952 / 0.950 | −0.280 | −0.591 | −0.339 | −0.038 | −0.016 |
| m505 | 190 | 0.968 / 0.966 | −0.316 | −0.621 | −0.437 | −0.042 | −0.074 |

Pooled novel subset: 968 rows from 289 contexts. Baseline C1 decode 0.949 [0.932, 0.965]. Cluster CIs:

| config | Δ C1 decode | Δ C1 bet-exact | Δ C6 bet-exact |
|---|---|---|---|
| dm | −0.281 [−0.326, −0.236] | −0.282 [−0.327, −0.237] | 0 |
| state | −0.585 [−0.638, −0.530] | −0.511 [−0.565, −0.458] | −0.034 [−0.055, −0.017] |
| pc2d | −0.375 [−0.431, −0.319] | −0.376 [−0.434, −0.319] | −0.208 [−0.253, −0.164] |
| lr | −0.043 [−0.059, −0.029] | −0.044 [−0.060, −0.030] | 0 |
| pc1d | −0.032 [−0.048, −0.017] | +0.004 [−0.005, +0.014] | 0 |

- The novel-subset effects match the full-set pooled Δs to within about 1 pp (dm −0.278, state −0.584,
  pc2d −0.363), so the effects are not carried by the duplicated stock messages.
- The non-novel subset (562 rows) also gives dm −0.272 and state −0.584. Full per-set CIs are in tables.md.

## What this establishes
- With frozen directions, the original intervention pattern is robust to resampling the sender's messages within
  the same 306 episode contexts and the same 6 schemes. The pattern:
  - dm ≈ −28 pp on C1 with C6 untouched;
  - state ≈ −58 pp on C1 (to chance-range accuracy 0.37) with C6 spared;
  - PC-2D ≈ −36 pp on C1 and −20 pp on C6;
  - pc1d and lr small.
- Every effect is reproduced in all 5 sets with low seed-level SD (≤ 0.025). Baseline capability does not shift.
- So the original effects were not artefacts of the particular gen_seed 0 messages. Wording per review: the effects
  are unchanged under resampling of the sender at T = 0.7 for the same contexts. About a third of the resampled
  messages duplicate text the directions were fit on, but the effects are the same on the novel subset (§7).

## What this does not establish
- Messages only: same schemes, same topic keywords, same episode contexts and prompts. This says nothing about new
  schemes (see LOSO), new contexts, real-world covert channels, or other models (see 14B/32B work).
- Directions were fit on the original set's activations. This is a frozen-direction transfer test to new samples
  from the same sender distribution, not a test of refit directions.
- The sender distribution has low diversity. About a third of the fresh messages are exact or ≥ 0.9 near-duplicates
  of some original message (mostly from other episodes), and 3–6% repeat the same episode's original exactly. So
  the "fresh" sets are partly overlapping, not fully held-out text.
- The primary pooled CIs cluster on context, but they still treat the 5 sender seeds as fixed. The seed-level
  SD/range (n = 5, indicative only) is the across-seed summary. All statistics are descriptive (≈ 150 uncorrected
  tests project-wide).
- Single model (Qwen2.5-7B), greedy decoding, one ablation policy (all layers + embedding).
