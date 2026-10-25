# Post-hoc robustness check: held-out-scheme (LOSO) intervention directions — PLAN

Written 2026-10-25 ~07:35 PDT, BEFORE any fold-specific run. Exploratory and post hoc: designed after seeing the
global results, so it is not preregistered confirmation. The rules below are fixed before seeing new outcomes. Any
deviation (bug fixes only) gets logged in notes/loso_robustness_log.md with a timestamp.

## Problem (verified against code)
- exp/probes.py `baseline()` fits every intervention direction on ALL 306 episodes, i.e. all 6 schemes:
  - `detect_C1vC4_mean_dm` = unit(mean C1 − mean C4), `mean` pooling, per layer.
  - `state_C1_last_2d` = QR([μHOT−μCOLD, μNEU−(μHOT+μCOLD)/2]) on C1 `last`, per layer.
  - `detect_C1vC4_mean_lr` = StandardScaler + LR(C=0.1) on C1-vs-C4 `mean` acts; raw-space unit(w/σ), per layer.
- PC controls were built inline (REPORT_NOTES §4.5a) on all 306 episodes:
  - `pc_ctrl_2d` = top-2 right-singular vectors of centred C1 `last` after projecting out `state_C1_last_2d`.
  - `pc_ctrl_1d` = top-1 of centred concat(C1, C4) `mean` after projecting out `detect_C1vC4_mean_dm`.
- The intervention results (§4.5, §4.5a) were then evaluated on the same 306 episodes, so each episode's own scheme
  contributed to the direction ablated on it.
- Data-driven selection in the original method: none beyond the fitting above.
  - Layer policy fixed a priori: all 28 decoder layers + embedding output, with the layer-0 direction on the embedding.
  - LR C=0.1 fixed a priori (not tuned).
  - k = 1 (dm, LR, PC-1D) / k = 2 (state, PC-2D) fixed by construction.
  - So no nested validation is needed; no layer selection is introduced.

## Verified data facts
- 306 episodes, ids `s{scheme}_{STATE}_{seed}`. Integer scheme labels 0–5 with 51 each, 17 per state per scheme.
- `meta.json` ids/scheme/state order == `messages.jsonl` order. The frozen messages on the Modal volume are
  byte-identical to the local copy (sha1 0eaad810…).
- Scoring (env.decode_metrics):
  - C1 primary = strict `decode_acc` (decoded == state), with parse failures counted as wrong; denominator = all
    evaluated episodes.
  - C6 primary = `bet_exact_acc`, chosen because C6 outputs often omit "DECODE:" (20/306 at baseline).
  - Parse failures and decoded=None are reported separately. Nothing is dropped.
- Generation is greedy but batched with left padding (bs=48). Evaluating a 51-episode subset changes batch
  composition, so the original 306-episode rows are NOT assumed to be a matched baseline: a matched baseline is
  re-run on every held-out subset. Agreement with the original rows is reported as a determinism check only.

## Folds and fitting (exp/loso_fit.py → data/loso/)
- For each held-out scheme k ∈ {0..5}: train = the 255 episodes of the other 5 schemes; eval = the 51 episodes of k.
- Fit on train only, with the SAME construction as the original (same pooling, conditions, centring on the train
  mean, per-layer QR/SVD, StandardScaler fit on train): dm_k, lr_k, state_k, pc2d_k (orthogonal to state_k),
  pc1d_k (orthogonal to dm_k).
- Saved: data/loso/fold{k}/*.npy, data/loso/splits.json (train/eval ids), data/loso/provenance.json (sha256 of input
  activation files, fitting code version, train-set variance shares).

## Run matrix (Qwen2.5-7B-Instruct, bf16, same hooks: ablate, layers=all + embedding)
For each fold k, ONE container evaluates all configs on the SAME 51 held-out episodes, in the same order and batch
composition:
| config | direction | role |
|---|---|---|
| none | – | matched baseline |
| dm_global / dm_fold | original detect_C1vC4_mean_dm / dm_k | priority 1 |
| pc1d_global / pc1d_fold | original pc_ctrl_1d / pc1d_k | matched control for dm |
| state_global / state_fold | original state_C1_last_2d / state_k | priority 2 |
| pc2d_global / pc2d_fold | original pc_ctrl_2d / pc2d_k | control for state |
| lr_global / lr_fold | original detect_C1vC4_mean_lr / lr_k | lower priority (negative result check) |
Conditions per config: C1 (covert decode) and C6 (explicit state), plus topic recall (C1 messages) and ARC-Easy
(first 100 items). The *_global configs on the same subset isolate the effect of fold fitting from batching effects.
Total: 6 folds × 11 configs.

## Outcomes and analysis rules (fixed now)
- Per fold and pooled over all 306 held-out evaluations, report:
  - C1 decode_acc; C1 bet_exact; C1 parse failures (bet None) and decoded=None counts; decoded distribution.
  - C6 bet_exact; C6 bet None count.
  - Recall; ARC.
- Pooled proportions get Wilson 95% CIs. Paired change vs the matched `none` on identical episodes: Δ with a 95% CI
  from a paired bootstrap over episodes stratified by scheme (10,000 reps, seed 0), plus an exact McNemar p.
  Fold-vs-global on identical episodes is reported the same way.
- Decision rules (C1 decode, pooled):
  - SURVIVES: the fold-direction drop vs `none` has a 95% CI excluding 0 AND is ≥ 50% of the global-direction drop
    on the same episodes.
  - ATTENUATED: the CI excludes 0 but the drop is < 50% of the global drop.
  - DOES NOT SURVIVE: the CI includes 0.
- C6 specificity (fold direction): the "spares explicit decisions" claim holds if the pooled C6 bet_exact drop vs
  `none` is ≤ 5 pp AND its 95% CI upper bound is ≤ 10 pp.
- Control comparison: for each priority direction, report fold-direction minus fold-PC-control on C1 and C6.
  - No uniqueness claim for the state subspace regardless of the outcome; at most "selective".
  - dm vs pc1d tests whether a matched-variance direction does the same.
- Per-fold consistency (descriptive only): the number of folds (of 6) where the fold drop is ≥ 50% of that fold's
  global drop.
- LR: the same statistics; the original claim (no effect) is re-checked, not assumed.
- No tuning, no extra configs, and no reruns after seeing outcomes except to fix code errors (logged). Messages
  are not regenerated.

## Budget
Cap: $43 of Modal credits (user). Price: L40S $0.000542/s (≈ $1.95/h), checked 07:28 PDT. Estimate per fold:
~1 min container start + weight load, plus 11 configs × ~15 s ≈ 4 min, so 6 folds ≈ 25 GPU-min ≈ $0.8, plus a smoke
test ≈ $0.1. Self-imposed task cap $5. The run stops if billing for today exceeds $12 (it was $7.00 before this task).
