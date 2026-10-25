# 32B ablation resistance: placement, residual linear decodability, and a behavioural test — PLAN

Written 2026-10-25 ~12:50 PDT, BEFORE computing any result below. Exploratory/post hoc: the question arose from the
32B follow-up (§4.11). It does not establish redundancy or a scaling law. CPU-only on saved activations; no paid
compute in this stage. Any later paid run needs the user's explicit approval.

## 1. Intervention placement (verified in exp/modal_app.py, not assumed)
- **Modified locations.** `_install` registers forward hooks on every decoder-layer module `model.model.layers[l]`
  (l = 0..L−1) and replaces its output hidden state, i.e. the residual stream after block l. Ablation is
  h ← h − (h·Q)Qᵀ in fp32, then cast back to bf16. It applies at EVERY token position: during prefill (system prompt,
  scheme transcript, hand, chat, question) and for every newly generated token. Keys/values cached by later layers
  are therefore computed from ablated states.
- **Embedding hook.** With `layers="all"`, an extra hook on `embed_tokens` output (= the input to layer 0) applies
  the LAYER-0 direction. That direction was fitted on layer-0 OUTPUT activations, so the embedding hook applies a
  direction fitted at a different location (inherited convention).
- **Readout location.** Probes read the same decoder-layer outputs, captured by later-registered forward hooks
  (they see the modified output). `last` = last prompt token (the decision position after
  `<|im_start|>assistant\n`); `mean` = mean over the sender-message span. The last layer is pre-final-norm.
- **Same convention in all models.** Every 7B, 14B and 32B LOSO and fresh run used layers="all" + embedding (only
  the earlier Phi exploratory runs differed).
- **Onset (pooled LOSO state-probe accuracy, C1 `last`; first layer ≥ 0.8):** 7B L18 (rel 0.667), 14B L26 (0.553),
  32B L42 (0.667). Before onset the per-layer class-mean "state" directions are fitted where the state is not
  decodable (≈ chance), so the all-layer recipe also removes near-noise directions in 18/28 (7B) and 42/64 (32B)
  layers, plus the embedding.

### Frozen post-onset placement (P_late)
Ablate only decoder-layer outputs with l ≥ ceil(2(L−1)/3), no embedding hook:
- 7B L18–27 (10 layers)
- 14B L32–47 (16 layers)
- 32B L42–63 (22 layers)
Rationale (existing evidence only, no behavioural data):
- One model-independent constant (relative depth 2/3).
- At or after state-probe onset in all three models: 7B and 32B reach 0.8 exactly at 2/3 depth; 14B earlier.
- Avoids the near-noise pre-onset directions and the cross-location embedding hook.

Dependence note: the onset values come from pooled LOSO probe curves that use all schemes for evaluation.
Training-only sanity check (computed now): for each outer fold, inner leave-one-scheme-out state accuracy on the 5
training schemes; report the first layer ≥ 0.8. If any fold's training-only onset falls after the cutoff, report it.
The rule is NOT changed in response.
The original all-layer + embedding placement (P_all) is kept as the comparator. No other layer sets will be tried.

## 2. Residual linear decodability: iterative removal on saved activations (7B and 32B; 14B as supplementary)
- **Data:** C1 `last` activations [306, L, D] (decision token), labels = state (3 classes), groups = scheme (6).
- **Outer folds:** leave-one-scheme-out (6). Everything below is fitted on the 255 training episodes of the fold
  only: centring mean, scalers, removal directions, readouts. Held-out episodes are only transformed and scored.
- **Layers analysed:** all layers with rel. depth ≥ 0.5 (reported per layer). The summary metric = mean over the
  P_late layers (rel ≥ 2/3), fixed now.
- **Two removal sequences**, both evaluated at the fixed ranks r ∈ {0, 2, 4, 6, 8, 10, 12, 14, 16}. Max rank 16 is
  fixed in advance (≤ 8 removal steps of ≤ 2 dims each).
  - **M (mean-first):**
    - Step 1 = the 2-D class-mean contrast of training residuals [μHOT−μCOLD, μNEU−(μHOT+μCOLD)/2]. This is exactly
      the original state-subspace construction, fold-specific.
    - After step 1 the training class means coincide in the residual space, so a three-class mean contrast cannot
      supply more directions. Steps ≥ 2 are therefore probe-weight directions (below).
    - No "8-D mean contrast" and no arbitrary PCs are used.
  - **P (probe-weight only; INLP-style, Ravfogel et al. 2020):** every step uses probe-weight directions.
  - **Probe-weight step:**
    - Fit StandardScaler on the current training residuals; fit multinomial logistic regression (C=0.1,
      max_iter 3000, the project's probe recipe).
    - Map the weights to raw space: W_raw = W/σ (3×D). Centre the rows (subtract the class-mean row), whose span is
      ≤ 2-D, the subspace a softmax readout uses.
    - Gram–Schmidt against the accumulated basis U (drop components with residual norm < 1e-6) and append.
  - **Why both:** class-mean directions capture where the class centroids differ (what the original intervention
    removed and what dominates activation variance). Probe-weight directions capture what a discriminative linear
    readout uses, including low-variance, noise-cancelling directions. If information survives M-step-1 but is
    removed within a few P-steps, it is low-rank but not in the mean contrast. If it survives many steps, it is
    spread over many directions.
- **Projection:** residual(x) = (x − μ_train) − ((x − μ_train) Uᵀ) U, applied identically to training and held-out rows.
- **Readout at each rank:** a FRESH multinomial LR (training-only StandardScaler, C=0.1) trained on the training
  residuals, scored on the held-out residuals. The rank-0 readout = the usual LOSO state accuracy.
- **Reported:**
  - held-out accuracy vs rank, per layer, per fold and pooled (n = 306), with a Wilson 95% CI;
  - paired Δ vs rank 0 with a scheme-stratified bootstrap over episodes (10k, seed 0; unit = episode, episodes
    within a scheme treated as exchangeable);
  - the training-variance share of the accumulated span U at each rank (and on held-out), for C1 `last` and C1 `mean`.
- **Not a causal result:** removing linearly readable information from saved activations says nothing by itself
  about behaviour. The model can recompute information downstream of an intervention; saved activations come from
  unablated forward passes.

### Training-only rank selection for any behavioural higher-rank removal (fixed now)
- For each outer fold, run the SAME sequence M inside an inner leave-one-scheme-out over that fold's 5 training
  schemes. Fit removal and readout on 4 schemes, score the 5th.
- Average inner accuracy over the P_late layers.
- r*(fold) = the smallest r ∈ {2, 4, …, 16} with inner accuracy ≤ 0.43 (= chance 1/3 + 0.10); if none, r* = 16.
- Outer held-out accuracy is never used to choose r*, the layers, or the sequence.

## 3. Controls and variance (planned)
- For a rank-r removal at layer l, report its share of activation variance on training C1 `last` and `mean`
  activations, plus the same quantity on held-out activations (post hoc check only).
- Candidate controls, all fitted on training data only and orthogonal to the removal span U_r:
  - (a) top-r training PCs: generally MORE variance than U_r. This is the original construction and is not matched.
  - (b) a greedy variance-matched control: walk down the training PC list and pick r PCs (orthogonal to U_r) whose
    cumulative training variance share is closest to that of U_r at that layer. This matches TRAINING variance only;
    held-out match is reported, not tuned.
  - (c) random rank-r subspaces: near-zero variance (lower bound).
- Variance at the decision token and message span does not capture variance at all other positions where hooks act.

## 4. Deliverables of this stage (no paid compute)
- exp/redundancy_analysis.py; data/redundancy/*.json; figures/fig11_residual_decodability.{png,pdf};
  notes/redundancy_results.md; REPORT_NOTES §4.13 + correction of the cross-model readout comparison.
- A behavioural run matrix with runtime/cost estimate for user approval (no runs launched).

## 5. PROPOSED behavioural stage — written 2026-10-25 ~13:20 PDT, AFTER the static analysis (§2 as amended in
notes/redundancy_log.md; results data/redundancy/tables.md), BEFORE any behavioural run. NOT APPROVED, NOT RUN.

### Rationale from the static analysis
- Removing the 2-D class-mean span leaves similar static state information in 7B and 32B.
  - Cross-fitted, late layers: 7B 0.918 → 0.536; 32B 0.966 → 0.602 (chance 0.333).
  - Under actual ablated forward passes, the readout differs sharply: 7B ≈ 0.62 vs 32B ≈ 0.92 (in-sample, V1).
  - So the 7B–32B difference appears only when the model runs with the ablation. The next informative test is
    placement (P_all vs P_late) plus activation capture under held-out-fitted ablation.
- A higher-rank state removal is NOT warranted by the static evidence:
  - MS (rank 8–10) does not lower 32B readability beyond M2 (0.611 vs 0.602).
  - Probe-weight sequences barely move it (P16 0.955).
  - MS is therefore listed only as an optional tier. Its 10-D span carries ≈ 0.55 of late decision-token variance,
    so any effect would be confounded by size.
- Controls cannot be variance-matched orthogonally to the state span in late layers. The best two orthogonal
  directions carry ≈ 0.18–0.22 of decision-token variance vs ≈ 0.31–0.35 for the span (training data). PC controls
  are therefore LOWER-variance controls and will be called that, never "matched".

### Run matrix (held-out fitting: fold directions from exp/loso_fit.py; frozen 7B message set; 7B on L40S, 32B on H200)
For each model × outer fold (6), ONE container evaluates on that fold's 51 held-out episodes, same order and
batching: C1, C6, recall, ARC-100, with parse failures and hook-leak assertion.
| config | span (fold-fit, training schemes only) | placement |
|---|---|---|
| none | – | – (matched baseline) |
| M2_all | state 2-D (= existing state_fold) | all layers + embedding (original) |
| M2_late | state 2-D | l ≥ ceil(2(L−1)/3), no embedding (frozen P_late) |
| PC2_all | top-2 PCs ⊥ state (= existing pc2d_fold; lower variance) | all + embedding |
| PC2_late | same | P_late |
| *(optional)* MS_late | per-scheme class-mean span (rank ≤ 10) | P_late |
| *(optional)* PC10_late | top-10 PCs ⊥ MS (lower variance) | P_late |
Activation capture (same container): with the fold-k M2 direction under P_all and under P_late, a forward pass
(prompt only, C1) on ALL 306 episodes; save `last` fp16 [306, L, D].
- Verification: ‖hQ‖/‖h‖ per layer must be ≈ 0 at ablated layers and baseline-like at non-ablated layers
  (placement check).
- Readout: LR trained on the 255 training episodes' ablated activations, tested on the 51 held-out ablated
  activations, per layer, pooled over folds. Compare with the unablated LOSO readout and the static cross-fit
  residual (0.54/0.60).
- Code change needed (≈ 30 lines, backward compatible): per-config `layers` in `_intervene_many`, plus a
  `capture_ablated` method reusing `_acts` with hooks installed. Smoke test 1 fold × 2 episodes per model.

### Analysis rules (fixed now)
- Per model, pooled n = 306 and per fold (51):
  - absolute C1 strict decode and bet-exact, C6 bet-exact, recall, ARC, parse failures (bet None / decoded None);
  - paired Δ vs matched none: scheme-stratified episode bootstrap, 10k, seed 0; exact McNemar for binary outcomes only.
- Key paired contrasts:
  - (i) M2_late − none
  - (ii) M2_late − M2_all
  - (iii) M2_late − PC2_late
  - (iv) M2_all − PC2_all
  - (v) the same for 7B vs 32B, descriptively; no cross-model test, since the confounds are known.
- Continuous effects with CIs. No threshold verdicts; similar point estimates are not called equivalent.
- Interpretation guide (exploratory):
  - 32B M2_late ≫ M2_all in effect → the original recipe (pre-onset/embedding hooks) masked a late-layer effect.
  - M2_late ≈ M2_all ≈ small in 32B and readout under ablation stays high → consistent with downstream
    re-encoding, not proof of redundancy.
  - 7B M2_late ≪ M2_all → the 7B effect partly relied on pre-onset/embedding perturbation.

### Cost estimate (Modal list prices checked 13:20 PDT: L40S $0.000542/s, H200 $0.001261/s)
Measured from the previous stage: 7B ≈ 8.6 s/config/fold and 32B ≈ 12 s/config/fold, plus load and startup
(7B ≈ 40–60 s, 32B ≈ 75–100 s per container).
| item | 7B (L40S) | 32B (H200) |
|---|---|---|
| 6 fold containers × (startup + 5 configs + 2 capture passes over 306) | ≈ 6 × 2.0 min = 12 min → $0.39 | ≈ 6 × 3.8 min = 23 min → $1.74 |
| optional MS_late + PC10_late | +6 × 0.3 min → $0.06 | +6 × 0.4 min → $0.18 |
| smoke tests | ≈ 2 min → $0.07 | ≈ 2.5 min → $0.19 |
| **subtotal** | **≈ $0.52** | **≈ $2.11** |
- Total core ≈ $2.4 (with optional ≈ $2.6); with a ×1.5 margin, ≤ $4. Proposed hard stop: $6 of new spend.
- Storage: captured `last` acts ≈ 0.7 GB (7B) + 2.4 GB (32B) on the volume and locally.
- Wall time: ≈ 10 min if folds run in parallel (6 + 6 containers).
