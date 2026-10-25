# 7B placement unbundling — PLAN (written 2026-10-25, before any paid run; awaiting user approval)
Exploratory follow-up to notes/redundancy_behaviour_results.md. State directions only (fold-fit M2 = data/loso/fold<k>/state.npy,
unchanged); 6-fold LOSO; frozen 7B messages; greedy decoding; same scoring. No 32B, no higher-rank, no controls in this
extension (so no direction-specific selectivity claim per component).

## Cells (2×2, conditional on late-layer removal L18–27 always present)
| | no pre-onset (L0–17) removal | pre-onset removal |
|---|---|---|
| no embedding hook | M2_late (core; re-run) | PRE_LATE = hooks L0–27, no embedding (new) |
| embedding hook (original layer-0 direction on embedding OUTPUT) | EMB_LATE = embedding + L18–27 (new) | M2_all (core; re-run) |
Plus the matched baseline `none` (re-run). The embedding hook reuses the original convention: the LAYER-0 direction, fitted on
layer-0 output, applied to the embedding output. It is not refitted, so the cells decompose the original recipe. They do
NOT test whether the location mismatch causes the embedding effect.

## Implementation (exp/modal_app.py; written, not run)
- `_install`: a layer list may contain "emb", which adds the embedding hook with the layer-0 direction. "all" is unchanged.
- `_arc`: stores per-item predictions; `_intervene_many` saves `arc_items`. The ARC return value is unchanged.
- `unbundle` entrypoint: per fold one L40S container runs none, M2_late, M2_all, EMB_LATE, PRE_LATE on the fold's 51
  held-out episodes in the core order and batching: C1, C6, recall, ARC (per item). Hook-leak assertion active.
- `unbundle_validate`: one container, fold-0 direction, 9 fold-0 held-out episodes. Probe hooks registered AFTER the
  intervention hooks on the embedding output and all 28 decoder-layer outputs record, per forward call, the phase
  (prefill T>1 / decode T=1), ‖hQ‖/‖h‖ onto that location's span, and float64 fingerprints. Also re-extracts
  extraction batches 4 (rows 64–79, the only batch whose saved acts differ from the capture) and 9 (identical) via the
  prompt-only and full-transcript pipelines on the same hardware.

## Validation criteria (stated in advance)
1. At hooked locations, ratio ≈ 0 (≤ 1e-3, bf16 recast) in BOTH prefill and every decode step; at unhooked locations,
   > 0.
   - M2_late: emb + L0–17 unhooked.
   - EMB_LATE: emb + L18–27 hooked; L0–17 unhooked but downstream of the hooked embedding.
   - PRE_LATE: emb unhooked; L0–27 hooked.
   - M2_all: all hooked.
2. Prefill fingerprints at unhooked locations that are upstream of every hook must equal the matched unablated (none)
   run exactly: M2_late emb + L0–17; PRE_LATE emb. Unhooked locations downstream of a hook (EMB_LATE L0–17) must differ.
3. Decode-step calls must appear for every location in every config, so the hooks are exercised during generation.
4. Reproducibility: the re-run none / M2_late / M2_all must reproduce the core-matrix C1/C6 outputs exactly before core
   results are reused. If not, report the discrepancy and use only the new same-container runs.
5. Pipeline test: if prompt-only ≠ full-transcript for batch 4 but = for batch 9 on the same hardware, the saved-vs-
   capture difference is a shape-dependent numerical path of the extraction pipeline. Otherwise it stays unexplained.

## Analysis (exp/unbundle_analyze.py, to be written before results are read)
- Per config, absolute values with Wilson CIs (n = 306): C1 strict decode, C1 bet-exact, C6 bet-exact, recall, ARC
  (600 = 6 folds × 100 items); parse failures (bet None, decoded None).
- Paired Δ vs none per metric.
- Factorial contrasts per metric, computed per unit (episode for C1/C6/recall; fold×item for ARC):
  - E|noPre = EMB_LATE − M2_late; E|Pre = M2_all − PRE_LATE;
  - P|noEmb = PRE_LATE − M2_late; P|Emb = M2_all − EMB_LATE;
  - Interaction = M2_all − EMB_LATE − PRE_LATE + M2_late.
- Uncertainty (same treatment as the core):
  - Bootstrap over CONTEXTS (episodes), stratified by scheme, each resampled context carrying all its configs' outcomes
    (10k, seed 0).
  - For ARC, bootstrap over the 100 ITEMS, each carrying all 6 folds × configs, since the same items recur in every
    fold.
  - Exact McNemar only for binary paired contrasts (not for the interaction).
- Continuous effects; no thresholds; a CI including 0 is not called equivalence.

## Cost estimate (Modal list price checked today: L40S $0.000542/s ≈ $1.95/h)
Basis: core 7B container ≈ 40–60 s startup/load; ≈ 9 s per config for 51 episodes (C1 + C6 + recall) + ARC-100.
| item | estimate |
|---|---|
| smoke (fold 0, 2 episodes, 5 configs incl. ARC) | ~1.5 min → $0.05 |
| validation container (5 configs × 9 episodes with probes + 2 pipeline batches × 2 pipelines) | ~1.5–2 min → $0.06 |
| main: 6 fold containers × (~50 s load + 5 configs × ~9 s) | ~6 × 1.7 min ≈ 10 min → $0.33 |
| total | ≈ $0.44 (≈ $0.66 with ×1.5 margin) |
Proposed hard stop: $1.00 of new spend for this extension. No captures for readouts (not requested); no other paid work.
