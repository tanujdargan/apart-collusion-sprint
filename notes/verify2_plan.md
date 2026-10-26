# Final targeted extension: PLAN (frozen 2026-10-26 ~02:35 UTC, from 6edc328, before any S5/S6 run)
Research only. Original 7B messages, 6 LOSO folds × 51 held-out episodes, fold-fit directions, unchanged scoring. Doses are
reused verbatim from the committed `data/verify/matrices/s1_7b.json` (= `data/verify/dose_alpha.json`; fold-0 state doses are
1.0 except L27 0.90). No refits, no new temperatures/models/cutoffs.
Frozen files:
- matrix: `data/verify/matrices/v2_s5s6.json` (6 containers = folds; job 1 = S5, job 2 = S6);
- validation spec: `v2_validate.json`;
- analysis: `exp/verify2_analyze.py`.

## S5 · sampled receiver
**Sampling setup:**
- Only the C1 receiver decode is changed: T = 0.7, top_p = 1.0, top_k = None (overrides the model's generation_config),
  repetition_penalty 1.0, max 24 new tokens. Eight seeds, 0–7, per episode.
- RNG: `torch.manual_seed(seed*100003 + batch_start)` before each generate batch (batch size 48, fixed order, the fold's
  51 episodes = 2 batches); HF multinomial on the CUDA default generator.
- The same seed schedule applies in every cell. Token draws are NOT assumed coupled once trajectories diverge.
- This is one robustness setting, not a representative temperature.

**Cells (each also runs the greedy C1/C6 + recall + ARC):**
- none, M2_L18, PC2_L18, M2_L18_match, PC2_L18_match, dm_L18, dm_all, pc1d_L18, pc1d_all.
- The greedy C1/C6 serve as anchors (must reproduce S1 outputs).
- Greedy C6/recall/ARC are deterministic capability checks; their robustness under sampling is NOT established.

**Outcomes per draw:**
- Strict decode and exact bet are separate outcomes.
- Malformed means any field is missing. Valid-but-wrong means both fields are present but the answer is not
  (correct state + true-rule bet).
- The failure taxonomy (decode × bet cross-tab) is mutually exclusive.

**Analysis (`exp/verify2_analyze.py`):**
- Seeds are averaged within episode.
- Primary paired contrasts, on strict and exact bet:
  - M2_L18 − PC2_L18;
  - M2_L18_match − PC2_L18_match;
  - dm_L18 − pc1d_L18;
  - dm_all − pc1d_all.
  Each cell − none is also reported.
- CI: bootstrap of the 102 context clusters (scheme, seed) within scheme, carrying 3 states × 8 seeds × all cells.
  Contexts are seeded by (scheme, seed) and checked not to recur across schemes.
- Monte Carlo SE = sd over seeds of the paired seed-level contrast / √8, plus the seed range. Eight seeds are fixed; if
  the MC SE is large it is reported, with no added runs and no tuning.
- Sampled vs greedy: the per-episode difference of contrasts (sampled seed-mean − existing S1 greedy), with its own
  cluster CI. Regimes are compared directly, not by significance.

## S6 · detector placement factorial (greedy)
- **New cells:** dm_EMB_LATE (embedding + L18–27), dm_PRE_LATE (L0–27, no embedding), pc1d_EMB_LATE, pc1d_PRE_LATE.
- **Anchors re-run in the same container:** none, dm_L18.
- **Reused from S1:** dm_all, pc1d_all, pc1d_L18 (same directions, messages, code, per-fold batching). Reuse is valid
  ONLY if the anchors reproduce S1's C1 + C6 outputs exactly; otherwise this is reported and the reused cells are
  flagged.
- **Contrasts, for dm, pc1d and dm − pc1d (all conditional on late removal):**
  - E|noPre = EMB_LATE − L18;
  - E|Pre = all − PRE_LATE;
  - P|noEmb = PRE_LATE − L18;
  - P|Emb = all − EMB_LATE;
  - Interaction = all − EMB_LATE − PRE_LATE + L18.
- **Metrics:** strict, exact bet, malformed, C6, recall, ARC (item bootstrap), plus each cell − none.
- A clean detector effect that needs early-inclusive placement is not thereby "generic disruption".

## Validation (`vvalidate2`, fold 0, first 9 held-out episodes = the smoke subset)
**Full-dose placement (greedy dm and pc1d; sampled M2, dm, pc1d):**
- hooked ratio ≤ 1e-3 in prefill AND decode;
- unhooked > 0;
- upstream locations bitwise equal to none in PREFILL only (identical prompts). Decode steps are not compared across
  cells, since sampled histories differ.

**Fractional dose (M2_L18_match, PC2_L18_match, sampled):** on the SAME incoming state per call (probe before and after
the hook):
- ‖Qᵀh_out‖ / ‖Qᵀh_in‖ = 1 − d (|err| ≤ 0.02);
- ‖h_in − h_out‖² / (d²‖QQᵀh_in‖²) ≈ 1 (bf16 output rounding; reported);
- the orthogonal fraction of the removed vector is small (reported).
- Achieved energies along intervened trajectories are measurements, not gates.

**Reproducibility:**
- Repeat-seed: none (seed 0), M2_match (seed 3) and PC2_match (seed 5) are generated twice in the same container and
  must be identical.
- Cross-container: the smoke run's S5 none seed-0 outputs must equal the validation container's (same 9 ids, same
  batch).
- Failures are investigated and reported, not patched through.

## Cost and cap: separate $2.00 TOTAL new spend (ledger `data/verify/spend_ledger_v2.json`)
- **Baseline:** month-to-date $46.1167 = $42.4477 + $3.669, where $3.669 is the previous extension's billing, unchanged
  for 30 min.
- **Guard:** same conservative spend guard: spent = max(billed delta, Σ computed upper bounds); launch only if
  spent + worst case ≤ $2.00.
- **Order:** validation → smoke (all jobs of fold 0, 9 episodes) → derive the full-stage bound from the measured smoke
  runtime (the 51-episode, 2-batch generation is scaled conservatively).
- **If it doesn't fit:** if the complete matrix (6 fold containers) cannot fit, STOP and return one revised estimate. No
  partial or truncated run. The folds may run as two waves of 3 containers if that is the only way the worst-case
  reservation fits.
- **Pre-measurement estimate:** about 6 × 370 s L40S ≈ $1.2 billed.
