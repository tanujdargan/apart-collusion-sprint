# Verification extension: PLAN (Phase 0, written 2026-10-25 17:15 PDT, BEFORE any new paid run; awaiting approval)
Frozen inputs: fold-fit directions `data/{loso,data_models/qwen14b/loso,data_models/qwen32b/loso}/fold<k>/*.npy` (unchanged),
frozen messages, greedy decoding, unchanged scoring (`env.parse_decode` / `parse_bet`). Matrices are frozen as JSON by
`exp/verify_matrix.py` (`data/verify/matrices/*.json`) and committed before each run. The analysis script
`exp/verify_analyze.py` implements §6 exactly and is committed before any S-stage output is read.

## 0. Phase 0 findings (free; `exp/phase0_free.py` → `data/verify/phase0_*.json`)
### 0.1 Existing-run protocol facts that constrain reuse
- **Fresh sets m101–m505.** These used GLOBAL frozen directions (`data/directions/*`), all-layer + embedding placement,
  and one 306-episode batch per config. The new fold-fit, per-fold-batched runs (§2) are a different protocol, so they are
  NOT pooled with these runs. The old fresh `none` is direction-independent, but its batching differs; it is compared only
  as a batching-sensitivity check.
- **Per-item ARC** (`arc_items`) exists only in `data/unbundle/runs`. Core-matrix, LOSO, fresh, 14B and 32B runs lack it.
  An ARC item bootstrap across folds therefore needs the anchor cells re-run in the same container. This also gives a
  matched baseline and a reproduction check. Reuse without a re-run is limited to the unbundle cells, which have exact
  reproduction and per-item ARC.
- **Message-free inputs.** ARC prompts contain no sender messages, so ARC for a (direction, placement) pair is
  message-independent. Fresh-set runs skip ARC and inherit it from the original-message run of the same cell.

### 0.2 Parse accounting (mutually exclusive; n = 306 per row)
Malformed counts are split as (decode missing / bet missing / both); the number after the semicolon is how many
decode-missing rows still have the true-rule bet. Valid-but-wrong counts are split as (decode correct, bet wrong /
decode wrong, bet coherent with that decode / decode wrong, true-rule bet / decode wrong, other bet).

| run | strict decode | exact bet | both correct | malformed | valid-but-wrong | C6 exact / bet-missing |
|---|---|---|---|---|---|---|
| 7B none (unbundle) | 0.958 | 0.958 | 0.958 | 0 | 13 (0/13/0/0) | 1.000 / 0 |
| 7B M2_late | 0.817 | 0.817 | 0.817 | 0 | 56 (0/56/0/0) | 1.000 / 0 |
| 7B PRE_LATE | 0.444 | 0.529 | 0.428 | 43 (35/0/8; 31) | 132 (5/127/0/0) | 0.944 / 0 |
| 7B M2_all | 0.441 | 0.539 | 0.428 | 42 (35/0/7; 34) | 133 (4/129/0/0) | 0.951 / 0 |
| 7B PC2_late / PC2_all | 0.899 / 0.578 | same | same | 0 / 0 | 31 / 129 (all coherent-wrong) | 1.000 / 0.866 |
| 7B dm_fold (all) | 0.732 | 0.732 | 0.732 | 0 | 82 (0/82/0/0) | 1.000 |
| 7B pc1d_fold (all) | 0.912 | 0.958 | 0.912 | 14 (14/0/0; 14) | 13 | 1.000 |
| 7B fresh m101 state (global) | 0.366 | 0.415 | 0.366 | 33 (23/0/10; 15) | 161 (0/161/0/0) | 0.980 |
| 14B none | 0.997 | 0.997 | 0.997 | 0 | 1 | 1.000 |
| 14B state_fold (all) | 0.464 | **0.614** | 0.464 | **112** (95/0/17; 46) | 52 | 0.771 / 0 |
| 14B pc2d_fold (all) | 0.288 | **0.546** | 0.288 | **187** (109/0/78; 79) | 31 | 0.663 / **68** |
| 32B M2_all / PC2_all | 0.886 / 0.974 | 0.886 / 0.961 | 0.886 / 0.961 | 0 / 4 (0/4/0) | 35 / 8 | 0.993 / 0.993 |

Reading:
- **7B.** The wrong answers are almost entirely COHERENT: the receiver decodes the wrong state and bets that state's
  rule bet. Malformed output is confined to the cells that hook the pre-onset layers (L0–17): about 14%, mostly a
  missing `DECODE:` tag with a correct bet.
- **14B.** At all-layer placement the strict-decode drop is mostly format disruption (37–61% malformed). Exact-bet
  accuracy is 15–26 pp higher than strict decode, and the PC2 control also breaks C6 formatting (68 missing bets).
  14B strict-decode effects at all-layer placement must not be read as covert-decoding effects without the exact-bet
  and malformed columns.
- Full table (148 run groups): `data/verify/phase0_parse.json`.

### 0.3 Direction stability (fold-internal context bootstrap; no evaluation data)
Setup:
- Resampling unit: context cluster (scheme, seed), all 3 states together, resampled within scheme. Each fold has 85
  training contexts (255 episodes).
- B = 200 for M2 and dm (B = 100 for 7B PC2).
- Metric: projector distance d = ‖P_b − P_0‖_F/√(2k) (RMS sine of principal angles; random subspaces ≈ 1.0), and
  the largest principal angle.

| model | M2 late, within-fold d (median; per-fold 95% range) | M2 late, max angle | M2 pre-onset d | dm late d | PC2 late d (7B) | between-fold d, late (M2 / dm / PC2) |
|---|---|---|---|---|---|---|
| 7B | 0.10–0.12 (0.07–0.17) | ≈ 7–8° | 0.40–0.47 (≈ 30°) | 0.13 | 0.27–0.35 (0.19–0.45) | 0.25 / 0.22 / 0.38 |
| 14B | 0.08–0.09 (0.06–0.12) | ≈ 6° | 0.36–0.47 | 0.12 | – | 0.24 / 0.21 / – |
| 32B | 0.08–0.10 (0.06–0.18) | ≈ 6° | 0.36–0.49 | 0.13 | – | 0.17 / 0.21 / – |

Reading:
- **Stable where it matters.** Late-layer state spans and dm are stable under resampling of training contexts.
- **Pre-onset spans are not.** They are poorly determined by the data (≈ 30° wobble), consistent with near-chance
  held-out decodability there. This bears on interpreting PRE_LATE: those hooks remove poorly determined, sample-specific
  directions.
- **Scheme composition matters.** Between-fold distances (dropping one of 6 schemes) exceed within-fold sampling
  variation, so the fitted spans depend on which schemes are in training. This is not fold-to-fold sampling noise.
- **The control is less determined than the state span.** The PC2 control is roughly 3× less stable than the state
  span (small eigengaps).
- Check: refit on the full training set reproduces the stored fold M2 (d ≤ 1e-4).

### 0.4 Ablated-pass readout protocol (exp/redund_behaviour_analyze.py `captures()`): verified
- **Held out by scheme, as documented.** The readout trains on the 255 episodes of schemes ≠ k and tests on scheme k,
  using captures made under fold k's direction. The test scheme never enters direction fitting.
- **Real dependence problem.** The readout training episodes ARE the episodes used to fit the removed span. At the
  FIRST hooked layer the trajectory is still unablated, so removing that span leaves the TRAINING class means almost
  equal:
  - between-class spread 0.0002–0.0006 vs 0.05 unablated (7B L18; 32B L42: 0.0007 vs 0.013);
  - held-out-scheme class means stay separated (0.04–0.06).
  The retrained readout at that layer (7B 0.35, 32B 0.42) is therefore near-degenerate by construction, not
  evidence about removal.
- **Sensitivity (free, no refit).** Excluding the first hooked layer changes the late-window means:
  - late_retrained, 7B: 0.832 → 0.886 (unablated 0.918);
  - late_retrained, 32B: 0.939 → 0.964 (unablated 0.967);
  - frozen probes are unchanged at chance.
- **Consequence for wording.** "Recovers within one block after the first hooked layer" should become "readout is
  near-degenerate at the first hooked layer by construction; from the next layer on, retrained readout is ≥ 0.88 (7B) /
  0.96 (32B)". No re-run is needed for this; a cross-fitted readout would need new captures and is not proposed.
  All-layer captures (first hooked = embedding/L0) are outside the late window and unaffected.

### 0.5 Dose-energy preview: the premise flips under the uncentred perturbation norm
The decision-token preview (`last`, training episodes; `phase0_energy_preview.json`) separates two measures:
- **Centred variance share (the earlier "lower-variance control" statement).** State 0.23–0.36 vs PC2 0.13–0.19 at
  L18–27. This holds.
- **Uncentred expected squared perturbation E‖hQ‖² (what removal actually subtracts).** PC2 is LARGER than M2 at almost
  every late layer and fold (ratio PC2/M2 0.2–21; median ≈ 3; < 1 only at L20 in 5/6 folds and L23/L27 in a few).
  The control spans pick up the large mean component.
- **Consequence.** The dose rule below is symmetric: per layer the larger-energy subspace is attenuated to the common
  attainable target. The orthogonal-PC2 comparison at late placement was NOT a "control removes less" comparison under
  this measure. The GPU dose fit over all prompt positions (§1A) decides the actual doses; the preview may not hold at
  all positions.

## 1. 7B original messages: surviving late effect (stage S1; `s1_7b`; 6 fold containers × 1 job; C1, C6, recall, ARC-100 per item)
Per fold k: 51 held-out episodes in the core order; fold-k directions; meter on every ablation config.
| cfg | direction | layers | dose | purpose |
|---|---|---|---|---|
| none | – | – | – | matched baseline |
| M2_L18, PC2_L18 | state / pc2d | 18–27 | 1 | anchors; must reproduce core-matrix outputs |
| M2_L18_match, PC2_L18_match | state / pc2d | 18–27 | c_M2,l / c_PC2,l (§1A) | matched-dose pair |
| M2_L18_half, PC2_L18_half | state / pc2d | 18–27 | 0.5 | dose-curve points |
| dm_L18, dm_all | dm (1-D) | 18–27 / all+emb | 1 | §1B; dm_all must reproduce LOSO dm_fold |
| M2_L16, PC2_L16, M2_L20, PC2_L20 | state / pc2d | 16–27 / 20–27 | 1 | §1C predeclared cutoff sensitivity (L18 = frozen) |
| *(optional)* pc1d_L18, pc1d_all | pc1d | 18–27 / all+emb | 1 | 1-D orthogonal control for the dm comparison |

### 1A. Dose definition, fixed before the fit (stage A0 `vdosefit`)
- **Norm.** Per-token squared L2 norm of the removed component, ‖δ‖² = c²‖hQ‖², with Q an orthonormal basis of the
  fold's layer-l span.
- **Fitting distribution.** Unablated, prompt-only forward of the C1 receiver prompts of the fold's 255 TRAINING
  episodes; all non-pad prompt positions (where the prefill hook acts).
- **Energy.** E_X,l = Σ_i Σ_t ‖h_{i,t} Q_X,l‖² / Σ_i n_i.
- **Doses.** T_l = min(E_M2,l, E_PC2,l); c_X,l = √(T_l / E_X,l) ∈ (0, 1]. Layers 18–27 only.
- **When the target is below a span's full removal.** If E_M2,l < E_PC2,l at a layer, the state span stays at full
  removal there and the control is attenuated. This is reported per layer, never forced above full removal (no
  amplification, no reflection).
- **Diagnostic only.** E_mid excludes position 0 (attention-sink token) and the decision token; it is reported, not
  used for doses.
- **Doses are frozen.** They are computed by `exp/verify_matrix.py s1` → `data/verify/dose_alpha.json` and committed
  BEFORE S1.
- **Matching is approximate.** Doses are fitted on unablated trajectories; downstream layers see intervened
  trajectories, generated tokens are not in the fit, and the fit uses right padding where generation uses left. The
  meter records the ACHIEVED Σ‖δ‖², Σ‖h‖² and token counts per hooked layer (C1 prefill and decode separately), and
  these are reported for every config.

## 2. Fresh-message placement robustness (stage S2; `s2_fresh`; 5 containers = sets m101–m505, 6 fold jobs each; C1, C6, recall; no ARC, see 0.1)
- **Protocol.** FOLD-FIT directions, the same as the original-message unbundle/core cells. Each fresh episode m<s>_<id>
  gets the direction of the fold that holds out its scheme.
- **Configs per (set, fold):** none, M2_all, M2_late, EMB_LATE, PRE_LATE, PC2_late. Optional: PC2_all, PC2_PRE_LATE.
  The PC2 cells are needed for any claim of selective robustness; without them, PRE_LATE robustness is non-selective.
- **Original-message reference (no new cost).** The existing unbundle cells (none, M2_late, M2_all, EMB_LATE, PRE_LATE;
  exact reproductions, per-item ARC) plus the S1 anchors (PC2_L18) use the same fold-fit protocol, messages, code and
  per-fold batching.
- **Not comparable.** The old global-direction fresh results stay in their own table.
- **Wording.** The sets share contexts, scheme/state assignments, templates and model with the original. This is
  MESSAGE-REALIZATION robustness, not independent replication.
- **Novel-message subset.** Defined as in `exp/fresh_analyze.py`: max difflib ratio vs any original < 0.9 and not an
  exact same-episode copy; 186–204 per set. It is reported alongside the full sets.

## 3. Cross-model coverage
- **3A. 32B factorial (stage S3; `s3_32b`; 6 H200 containers).**
  - Cells, per fold: none, M2_late (42–63), M2_all, EMB_LATE (emb + 42–63), PRE_LATE (0–63, no emb). All are re-run
    in the same container, because the old 32B cells lack per-item ARC; reproduction is checked against
    `data/redund/32b/runs`.
  - Hook validation (V32, `v32b`) on fold 0, 9 episodes: same criteria as notes/unbundle_plan.md (hooked ≤ 1e-3 in
    prefill and every decode step; unhooked upstream locations bitwise equal to the matched none).
  - Contrasts are conditional on late removal. No embedding or early-layer effect is inferred without late removal.
- **3B. 14B (stage S4; `s4_14b`; 6 L40S containers; 14B's own frozen messages).**
  - Cells, per fold: none, M2_all, PC2_all (re-run anchors vs LOSO state_fold / pc2d_fold), M2_L32, PC2_L32,
    M2_L26, PC2_L26.
  - L32 = the frozen relative-depth cutoff. L26 = a FOLLOW-UP motivated by the previously observed onset, not an
    untouched preregistered choice. Both are reported; neither is selected.
  - Late-list hooks are the same code path validated on 7B and 32B, so validation here is a 2-episode smoke plus
    the meter (achieved δ > 0 only at hooked layers).

## 4. Generalization tiers (SEPARATELY PRICED, optional)
### 4a. Fresh-message refit (7B; G4a)
- **Procedure:**
  - Prompt-only C1 `last` capture of sets m101–m404 (unablated, 4 × 306).
  - Per fold k, refit M2/PC2 (same constructions as `exp/loso_fit.py`) on the 4 × 255 episodes of schemes ≠ k.
  - Evaluate on m505 scheme-k episodes: none, M2f_all, M2f_L18, PC2f_all, PC2f_L18, with C1, C6, recall and ARC
    (ARC is needed because the directions are new).
  - Paired with S2's m505 cells (original fold-fit directions) on the same episodes.
- **What is new:** the message realizations used to fit and to evaluate (fit ⟂ eval sets); directions never touched
  original messages; eval contexts are outside the fold's fitting contexts.
- **What is shared:** the 306 contexts (cards, names, hands), all 6 scheme mappings, topic vocabulary, templates, model
  and sender sampling procedure. ~25–30% of fresh messages copy an original message. Activations come from the
  prompt-only pipeline (a known small numerical difference from the saved full-transcript activations).
- **Unit of generalization:** message realizations within fixed contexts and mappings. NOT independent.

### 4b. New contexts + new mappings (7B; G4b)
- **Construction, FROZEN NOW** (`exp/verify_matrix.py schemes4b` reproduces and asserts it; recorded before any generation):
  - 6 new (COLD, NEUTRAL, HOT) schemes over the same 9 topics. Each topic sits in exactly 2 schemes under 2 different
    states, one of which is its originally unused state. No topic pair co-occurs in any original scheme or in two new
    schemes.
  - Enumeration order + `random.seed(0)` choice among 40 solutions:
    `[(food, family, pets), (sports, music, movies), (travel, cars, weather), (movies, travel, food),
    (cars, sports, family), (pets, weather, music)]`.
  - Each topic keeps exactly one of its original (topic, state) pairs (unavoidable with 9 topics); 9/18 pairs are
    shared.
  - Contexts: seeds 100–116 (new names, cards, hands, distractors via the same generator); ids prefixed `n`.
- **Pipeline:**
  - Sender generation (same prompts/sampling, seed 0), then C1/C6/recall baseline.
  - Feasibility gate reused from the original pipeline: baseline C1 strict decode ≥ 0.70, otherwise stop and report,
    with no prompt tuning.
  - C1 `last` capture, then LOSO refit on the new schemes, then eval of 6 folds × (none, M2n_all, M2n_L18, PC2n_all,
    PC2n_L18) with ARC.
- **Code change needed:** `env.make_episode` gets a `schemes` argument (default = original; backward compatible),
  written after approval.
- **What is new:** contexts, topic→state mappings, messages, directions.
- **What is shared:** topic vocabulary and descriptions, prompt templates, task family/rules, model, construction
  procedure and analysis code; 9/18 (topic, state) pairs.
- **Unit of generalization:** new contexts and mappings within this task family and model. NOT a new task or model.

## 5. Gates (stated in advance; reused checks are not re-run)
1. **A0 dose-path validation (`v7b_dose`, fold 0, 9 episodes).**
   - `late`: hooked ratio ≤ 1e-3 in prefill and every decode step.
   - `late_half`: at the FIRST hooked layer, prefill ratio_after = 0.5 × ratio_none (± 2%), since upstream is
     identical.
   - L0–17 bitwise equal to none.
2. **Reproduction.** Same-container anchors must reproduce earlier outputs:
   - 7B M2_L18 / PC2_L18 vs core M2_late / PC2_late; 7B dm_all vs LOSO dm_fold; 14B M2_all / PC2_all vs LOSO
     state_fold / pc2d_fold; 32B none / M2_late / M2_all vs core.
   - Reported as the fraction of identical outputs. If < 1, only same-container cells are used and the discrepancy is
     reported.
   - full-dose (dose=None vs dose list of 1.0) is the same floating-point op (×1.0 is exact).
3. **32B hook validation** as in §3A, before S3.
4. **Smoke** (2 episodes, first job) before each main stage.

## 6. Analysis (frozen; `exp/verify_analyze.py`, written before S-outputs are read)
- **Metrics, per config:**
  - C1 strict decode; C1 exact bet; malformed counts using 0.2's mutually exclusive categories; valid-but-wrong;
  - C6 exact bet + missing bets; recall; ARC (per item);
  - achieved perturbation (meter).
- **Contrasts.** Paired per episode on the same evaluation units:
  - S1: each config − none; M2_L18_match − PC2_L18_match (primary dose-matched contrast); M2_Lc − PC2_Lc for
    c ∈ {16, 18, 20}; dm_L18 − dm_all.
  - Dose curves: C1 (and exact bet) vs the ACHIEVED summed Σ_l mean-token ‖δ_l‖² (and relative Σ‖δ‖²/Σ‖h‖²), with
    points {0, 0.5, match, 1} per span.
  - S2, per set and pooled: late − none, EMB_LATE − late, PRE_LATE − late, all − PRE_LATE, M2_late − PC2_late;
    fresh − original difference-in-differences paired by orig_id.
  - S3: the 7B factorial contrasts (E|noPre, E|Pre, P|noEmb, P|Emb, interaction).
  - S4: M2_Lc − none, PC2_Lc − none, M2_Lc − PC2_Lc for c ∈ {32, 26}.
- **Uncertainty.**
  - Bootstrap over CONTEXT CLUSTERS (scheme, seed), 102 clusters with 3 states each, stratified by scheme; 10k
    replicates, seed 0. Each resampled cluster carries all its episodes × configs × folds × fresh sets, preserving
    shared-context dependence across sets.
  - The earlier episode-level bootstrap is also reported, for continuity.
  - ARC: bootstrap over the 100 items, each carrying all folds × configs.
  - Exact McNemar for binary pairs only.
- **Reading rules.**
  - Report continuous effects with CIs; no thresholds; a CI including 0 is not equivalence.
  - Schemes are fixed (6) and stratified, not sampled. Inference is to new contexts and messages under THESE schemes,
    model and prompts; per-fold values are shown.
- **Interpretation limits:**
  - A surviving matched-dose gap is evidence beyond this energy measure, not proof that every selectivity confound is
    gone. A vanishing gap does not show dose is the sole explanation.
  - The controls are not variance-matched, and their stability is lower (0.3).
  - Readout recovery is not reconstruction; near-chance probes are not absence of information.

## 7. Cost (Modal list prices: L40S $0.000542/s, H200 $0.001261/s)
### Timing basis, from logged runs
Per-config times (51 episodes, C1 + C6 + recall + ARC):
- 7B: 7–11 s (LOSO/unbundle logs) → 10 s with meter; 7 s without ARC.
- 14B (L40S): 14–19 s → 17 s.
- 32B (H200): 11–14 s → 15 s.

Fixed cost per container (load + startup, plus the 30 s idle `scaledown_window`):
- 7B: 65 s (unbundle: 6 containers × 78 s wall ⇒ billed $0.256);
- 14B: 80 s;
- 32B: 120 s (32B LOSO: 66 configs, 208 s wall ⇒ ≈ $1.38 billed).

Earlier stages billed 30–40% below their estimates.

### Itemized estimate
| stage | contents | GPU-s | est. |
|---|---|---|---|
| A0 | 7B dose fit (306 prompts) + dose-path validation + S1 smoke | ≈ 270 | $0.15 |
| S1 | 7B 6 × (65 + 13 × 10 s) | 1170 | $0.63 |
| S1-opt | + pc1d_L18, pc1d_all | 120 | $0.07 |
| S2 | 5 sets × (65 + 36 × 7 s) + smoke | 1660 | $0.90 |
| S2-opt | + PC2_all, PC2_PRE_LATE (60 cfgs) | 420 | $0.23 |
| V32 | 32B hook validation + smoke | ≈ 305 H200 | $0.38 |
| S3 | 32B 6 × (120 + 5 × 15 s) | 1170 H200 | $1.48 |
| S4 | 14B smoke + 6 × (80 + 7 × 17 s) | 1300 | $0.71 |
| **Necessary subtotal** | A0, S1, S2, V32, S3, S4 | | **$4.25** |
| G4a | capture 4 × 306 + eval 30 cfgs + smoke | ≈ 570 | $0.31 |
| G4b | generation + baseline + capture + eval 30 cfgs + smoke | ≈ 665 | $0.36 |
| **All optional** | S1-opt, S2-opt, G4a, G4b | | **$0.97** |
| **Total estimate** | | | **$5.22** |
| Retry reserve | one full S3 retry ($1.48) + 20% of everything ($1.04) | | $2.52 |

- **Proposed hard cap: $8.00 of NEW spend**, everything included (≈ 21% of the user-reported $37.76).
- **If optional tiers are struck: $6.50.**
- **Spend tracking.** Month-to-date Modal billing (`modal billing report --for "this month"`) minus the baseline
  $42.448 (00:13 UTC Oct 26). The "today" bucket rolls over at 00:00 UTC, so it is not used. Billing lags a few
  minutes, so before each launch: billed delta + stage estimate ≤ cap, or STOP.
- **Stop rules.** A stage whose billed cost exceeds 1.5× its estimate means STOP and report. No apps are left running
  between stages.
