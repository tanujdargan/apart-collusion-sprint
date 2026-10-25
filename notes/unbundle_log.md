# Unbundling execution log (plan: notes/unbundle_plan.md). Approved by user: up to $1.00 new spend, no extra configs.
- 14:45 start. Spend baseline: $13.090 today. Correction recorded: the earlier "6-fold bitwise
  identical" check compared each fold's M2_late capture to ONE unablated capture (cap_redund_none.npy, produced in the
  fold-0 container); folds 1-5 are cross-container comparisons, not per-fold matched baselines. Saved acts
  (data/acts, full-transcript extraction pipeline) are distinct from same-pipeline captures (data/redund/7b/caps).
- 14:47 smoke OK (57 s wall): layer sets as intended; arc_items saved. Validation start.
- 14:48 validation (fold-0 direction, 9 fold-0 held-out episodes, same container as its matched 'none'):
  1 prefill + 11 decode calls per location per config; hooked max ratio ≤ 2.33e-4 (prefill and decode); unhooked min
  ≥ 0.008; prefill fingerprints at unhooked upstream locations == matched none (M2_late emb+L0–17: 19/19; PRE_LATE emb);
  EMB_LATE L0–17 differ (downstream of hooked emb) → all pre-stated criteria met.
- Pipeline test: re-extracted batch 4 (rows 64–79) and batch 9 (144–159) on the same GPU type. Full-transcript ==
  saved acts bitwise and prompt-only == core capture bitwise for both batches. Prompt-only vs full differ for batch 4
  (max rel 0.031; padded full length 377) but are identical for batch 9 (375). → The saved-vs-capture difference is
  a deterministic, shape-dependent numerical path between the two pipelines; the exact kernel is not identified.
- today $13.121  followups delta $5.724. Main run start (6 × L40S, 5 configs each).
- 14:49 main run done: 30/30 configs, 78 s wall. today $13.259  followups delta $5.862. No apps running.
- 14:50 analysis + results note written. today $13.377  followups delta $5.980.
