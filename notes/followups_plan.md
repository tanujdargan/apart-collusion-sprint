# Follow-ups plan: fresh messages (7B), 14B LOSO, focused 32B extension

Written 2026-10-25 ~07:55 PDT, BEFORE any run in this batch. Exploratory/post-hoc extensions, not preregistered
confirmation. The rules below are fixed before seeing new outcomes; deviations (bug fixes only) are logged with
timestamps in notes/followups_log.md.

## Budget
- Cap = min(user-reported credits $43.70, verifiable balance). The Modal CLI exposes spend, not the credit balance,
  so the balance itself cannot be verified; the cap is therefore $43.70.
- Spend is tracked as the delta of `modal billing report --for today`; baseline at start of batch = $7.397.
- **Internal hard stop: $25 of new spend** (≈43% safety margin). If the running total passes $18, stop launching
  new GPU work and finish only what is already running.
- Prices (modal.com/pricing, 07:50): H200 $0.001261/s ($4.54/h), H100 $0.001097/s, A100-80 $0.000694/s,
  L40S $0.000542/s ($1.95/h); CPU $0.0000131/core/s; memory $0.00000222/GiB/s.

| Work | GPU | Containers | Est. GPU time | Est. cost |
|---|---|---|---|---|
| A. fresh messages 7B: 5 sets × (306 sender gens + 6 configs × [C1, C6, recall] on 306 + ARC) | L40S | 5 parallel | 5 × ~6 min | ~$1.0 |
| B. 14B LOSO: 6 folds × 11 configs on 51 held-out episodes (C1, C6, recall, ARC) | L40S | 6 parallel | 6 × ~6 min | ~$1.2 |
| C0. 32B weight prefetch to volume (~65 GB) | CPU only | 1 | ~10 min CPU | <$0.05 |
| C1. 32B baseline (C1/C2/C4/C6/recall/ARC on 306) + activation extraction C1, C4 | H200 | 1 | ~15 min | ~$1.2 |
| C2. 32B LOSO: 6 folds × 11 configs | H200 | 6 parallel | 6 × ~8 min | ~$3.6 |
| smoke tests, retries, load overhead (×1.5 margin) | – | – | – | ~$3.5 |
| **Total estimate** | | | | **≈ $10.5** |
Storage: 32B acts C1+C4 × 3 poolings × [306, 64, 5120] fp16 ≈ 1.2 GB (volume + local); local disk has 146 GB free.

## Coordination
- The coordinator (main session) owns all GPU launches and all edits to exp/modal_app.py and exp/loso_fit.py.
- Agents do CPU-only analysis in their own files:
  - fresh-msg analysis agent → exp/fresh_analyze.py, data/fresh/analysis*, figures/fig8_fresh*, notes/fresh_results.md
  - cross-model agent → exp/xmodel_analyze.py, figures/fig9_xmodel*, notes/xmodel_results.md
  - critical-review agent → notes/followups_review.md (read-only otherwise)
- Existing artifacts are never overwritten. New outputs:
  - data/fresh/<set>/
  - data_models/qwen14b/loso/
  - data_models/qwen32b/ (acts, directions, loso)
  - volume namespaces /fresh, /qwen14b/loso, /qwen32b

## A. Fresh-message robustness (Qwen2.5-7B)
- New sender messages for the SAME 306 episode contexts: `env.episodes(range(17))`, same prompts, schemes, state
  distribution (6 × 3 × 17) and sampling settings (T=0.7, top_p=1, max_new 120). Only the sampling seed changes.
  Original collude messages used gen_seed=0 (torch seed 0·100003+batch). New sets use gen_seed ∈ {101, 202, 303,
  404, 505}.
- IDs: `m{gen_seed}_{orig_id}` (unique), with `orig_id` kept for pairing. Generated once per set; no example is
  regenerated or filtered.
- FROZEN: the original 7B direction files (data/directions/*: detect_C1vC4_mean_dm, pc_ctrl_1d, state_C1_last_2d,
  pc_ctrl_2d, detect_C1vC4_mean_lr), hooks (ablate, all layers + embedding), scoring and parsers. No refit.
- Per set, one container evaluates configs {none, dm, pc1d, state, pc2d, lr} on all 306 new episodes in the
  original order (bs 48, so batching is matched across configs): C1, C6, recall, ARC (ARC does not depend on
  messages; it is computed per set anyway as a determinism check). The hook-leak assertion stays active.
- Checks:
  - exact duplicates and near-duplicates vs the original set (same episode and across episodes; normalised exact
    match, and difflib ratio ≥ 0.9);
  - sender compliance (forbidden words, digits, target-topic keyword);
  - baseline C1 decode per set before interpreting any intervention effect.
- Analysis:
  - per set and pooled across sets, report absolute accuracy, Wilson CIs, paired Δ vs the matched none
    (stratified by scheme within set), McNemar p per set, and per-scheme breakdown;
  - across-set variability is reported as the range and SD of per-set Δ over the 5 sets (seed-level unit);
  - compared with the original frozen set (Δ dm −0.265, state −0.578, etc.).
  - Interpretation rule: an effect "replicates on fresh messages" if, pooled across sets, the paired Δ CI excludes 0
    with the same sign and the pooled Δ is ≥ 50% of the original Δ. C6-spared uses the same rule as LOSO (drop
    ≤ 5 pp and CI upper ≤ 10 pp).
  - Scope: new messages within the same schemes and contexts; not new schemes, not real-world generalisation.

## B. 14B held-out-scheme check
- Existing: data_models/qwen14b/acts/{C1,C4}.npz (mean, last), 306 episodes (14B's own frozen messages,
  /vol/qwen14b/data/messages.jsonl). No extraction needed.
- exp/loso_fit.py gets a DATA_DIR parameter. Check mode: refit on all 306 must reproduce the existing 14B
  direction files (dm, state, pc2d, lr; pc1d did not exist for 14B, so it is fitted here with the same construction).
- Global = fit on all 306 (data_models/qwen14b/loso/global/), folds = without scheme k. Same 11-config matrix,
  same fixed policy as the 14B canonical ablation (all 48 layers + embedding), conditions C1, C6, recall, ARC on 51
  held-out episodes per fold. Matched baseline per fold.
- Analysis and decision rules identical to notes/loso_robustness_plan.md. 7B findings are not assumed to hold:
  the 14B state ablation was already non-selective (C6 0.71) in-sample.

## C. Focused 32B extension (Qwen/Qwen2.5-32B-Instruct, bf16 — same precision as 7B/14B; no quantisation)
- Question: does the detector/intervention relationship change at 32B?
  1. Are detectors still separable at layer 0?
  2. When does the covert state become decodable?
  3. Are detector directions orthogonal to the state?
  4. Does dm/LR ablation affect collusion?
  5. Is state ablation selective vs PC-2D and C6?
- Hardware: H200 (141 GB). Weights ≈ 65.5 GB; headroom for KV (bs 48 × ~700 tokens ≈ 9 GB) and activation
  capture (8 seqs × 64 layers ≈ 4–8 GB). If H200 is unavailable: H100 80 GB, smaller acts batch, gen bs 48 if it
  fits. Report the boundary if neither works. Weights are prefetched to the volume on CPU (no GPU billed during
  download).
- Inputs: the 7B frozen message set (data/messages.jsonl, same 306 episode ids, prompts, scoring), so the
  receiver-side comparison with 7B is matched input-for-input. (14B used its own messages; noted as a difference.)
- Step C1: baseline capability (C1, C2, C4 chance controls, C6, recall, ARC). If C1 decode < 0.70 (the original
  gate rule), stop and report.
- Extract C1 and C4 activations (mean, last, full_last) only.
- Probes (same recipe: LOSO AUROC per layer; state acc per layer); cosines detector vs state.
- Step C2: directions fitted with the original constructions and fixed hyperparameters, global (all 306) AND
  held-out-scheme folds; the same 11-config LOSO matrix as 7B, with the fixed all-layer + embedding policy. The
  pooled global configs are an in-sample evaluation (labelled as such); fold configs are the held-out evaluation.
- No layer/strength tuning. No steering in this batch.

## Reporting
For every experiment: absolute accuracy, paired Δ vs matched baseline, denominators, parse failures, Wilson /
paired-bootstrap CIs (resampling unit = episode, stratified by scheme; seeds/sets reported separately), exact
McNemar for binary outcomes only, and per-scheme/per-set tables. No arbitrary success cutoffs beyond the descriptive
LOSO rules already used; no significance chasing.
