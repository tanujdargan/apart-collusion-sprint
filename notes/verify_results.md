# Verification extension: results (exploratory; plan notes/verify_plan.md @ 7652382; log notes/verify_log.md)
Tables: `data/verify/tables.md` · analysis `exp/verify_analyze.py` (frozen before any run; unchanged after) · figure
`data/verify/fig_dose.png`.

Conventions:
- Units: episodes. CIs: context-cluster bootstrap (102 clusters = (scheme, seed), stratified by scheme); the episode-level CI
  is in the tables. ARC uses an item bootstrap (100 items × folds). n = 306 per stage (S2 pooled n = 1530 over 5 sets).
- Strict decode / exact bet / malformed / C6 / recall / ARC are all in the tables; the text quotes strict decode unless noted.
- Doses use uncentred removed energy, which ranks the spans OPPOSITE to centred variance.

## Validation and reproduction (all pre-stated gates met)
- **Dose path (7B):** hooked ≤ 1.7e-4; upstream bitwise = none 19/19; half dose 0.501× at the first hooked layer.
- **32B hooks:** hooked ≤ 3.3e-4 in prefill and decode; upstream bitwise 43/43 (M2_late), 1/1 (PRE_LATE); EMB_LATE L0–41
  differ 42/42.
- **Same-container anchors reproduce earlier outputs exactly (1.000):**
  - 7B: none, M2_L18, PC2_L18, dm_all (vs LOSO dm_fold), pc1d_all;
  - 32B: none, M2_late, M2_all;
  - 14B: none, M2_all, PC2_all.
  - G4a none = S2 m505 none.
  - Fresh-set none with per-fold batches = the old 306-episode batch outputs (1.000).
- **G4b feasibility gate passed:** C1 0.931 ≥ 0.70.

## 1. 7B surviving late effect (S1, original messages, fold-fit, L18–27)
- **Full dose:** state −0.141 [−0.176, −0.105]; PC2 −0.059 [−0.085, −0.033]; gap −0.082 [−0.121, −0.042]. The control
  removed ~2.4× MORE achieved prefill energy than the state span (Σ_l mean-token ‖δ‖² 32.0k vs 13.5k).
- **Matched dose (primary):** state −0.134 [−0.170, −0.098]; PC2 −0.003 [−0.023, +0.013]; gap −0.131 [−0.167, −0.092].
  - Matching attenuated the CONTROL: over all prompt positions E_PC2 > E_M2 at nearly every layer, so c_M2 ≈ 1 and c_PC2
    had a median of 0.37–0.96.
  - Achieved summed prefill energy: 12.7k (state) vs 11.0k (control); decode 3.9k vs 1.9k.
  - Per-layer matching held at the first hooked layer only (achieved/fitted 1.00). Downstream, achieved per-layer ratios
    range 0.4–10× because the trajectories are intervened.
  - So matching is approximate, holding in aggregate rather than per layer.
- **Half dose:** state −0.039 [−0.069, −0.013]; PC2 −0.007 [−0.026, +0.010]; gap −0.033 [−0.065, 0.000].
- **Reading:** under this energy measure the gap does not shrink when doses are matched. This is evidence beyond the dose
  measure, not proof that all selectivity confounds are gone; the PC2 control is also less stable and not variance-matched.
- **Detector (dm):**
  - At late placement −0.010 [−0.026, +0.007], vs −0.225 [−0.265, −0.186] at all-layer placement
    (late − all +0.216 [+0.176, +0.255]).
  - At late placement it is indistinguishable from its 1-D control (−0.010 [−0.026, +0.007]); pc1d_L18 changed no outputs.
  - **The detector-direction effect does not survive placement correction in 7B;** it depends on the all-layer recipe.
  - At all-layer placement, dm − pc1d is −0.180 (strict) / −0.225 (exact bet); pc1d_all's strict drop is all malformed
    output with correct bets.
- **Cutoff sensitivity (predeclared; L18 frozen):**

  | first hooked layer | state | control | state − control |
  |---|---|---|---|
  | L16 | −0.072 | −0.108 | **+0.036 [0.000, +0.072]** (control larger) |
  | L18 | −0.141 | −0.059 | −0.082 [−0.121, −0.042] |
  | L20 | −0.101 | −0.007 | −0.095 [−0.127, −0.062] |

  The state-over-control gap is present at L18 and L20 and REVERSES at L16. Starting the state hooks at L16 also makes
  the state effect smaller, not larger (the L16–17 spans are poorly determined: bootstrap d 0.34 / 0.23). Unexplained.
- **Capability:** late configs show no detectable C6/ARC cost and recall ≤ 1 pp; no malformed output.

## 2. Fresh-message placement robustness (S2; fold-fit; 5 sets; message-realization robustness, NOT independent replication)
- **Pooled:**
  - late −0.189 [−0.206, −0.172]; embedding+late −0.179; all-decoder/no-embedding −0.512; all −0.512;
  - PC2 late −0.075; state − PC2 at late −0.114 [−0.135, −0.093].
  - The novel-message subset (n = 968) is similar: late −0.204, gap −0.115.
- **Fresh vs original (difference-in-differences):**
  - late effect −0.048 [−0.083, −0.014] (larger on fresh);
  - late gap −0.032 [−0.071, +0.008];
  - all-layer and pre-onset effects ≈ unchanged (+0.004, +0.001).
- **Factorial:**
  - pre-onset effect −0.323 [−0.354, −0.291];
  - embedding effect +0.010 [+0.003, +0.017] without pre-onset hooks, −0.001 with them;
  - interaction −0.010 [−0.020, −0.001].
  The embedding cell is small, not zero, on fresh messages; it was 0.000 [−0.016, +0.016] on the original messages.
- **Pre-onset selectivity is mostly format:**
  - PRE_LATE − PC2_PRE_LATE: strict −0.138 [−0.167, −0.107], but exact bet only −0.042 [−0.073, −0.011]; malformed
    +0.131.
  - The PC2 pre-onset cells instead cost C6 (−0.135) and recall (−0.072).

## 3. Cross-model
- **32B factorial (S3):**
  - late +0.007 [0.000, +0.016]; embedding+late +0.007; all-decoder/no-embedding −0.088 [−0.111, −0.065]; all −0.085.
  - Pre-onset effect −0.095 [−0.118, −0.072]; embedding 0.000 / +0.003; interaction +0.003.
  - The pre-onset hooks also cost ARC −0.11 [−0.15, −0.08] (0.98 → 0.89); late hooks cost none.
  - So the small 32B all-vs-late difference sits in the pre-onset hooks and comes with a general capability cost. There is
    no control in this factorial, so no selectivity claim.
  - Per fold the effect is heterogeneous (fold 4 drops to 0.59).
- **14B (S4, its own messages):**
  - All-layer placement is destructive for BOTH spans:
    - state: strict −0.533, exact bet −0.382, malformed +0.366, ARC −0.237, C6 −0.229;
    - control: −0.709 / −0.451, malformed +0.611, ARC −0.625.
  - L32 (frozen cutoff): state −0.020 [−0.036, −0.007], control −0.013; gap −0.007 [−0.020, +0.003].
  - L26 (follow-up, motivated by the prior onset, not preregistered): state −0.098 [−0.118, −0.078], control −0.013;
    gap −0.085 [−0.108, −0.062]. No ARC/C6 cost; the L26 control costs recall −0.026.

## 4. Generalization tiers
- **G4a (directions refit on fresh sets m101–m404; eval m505 held-out schemes; shared contexts, mappings, templates, model):**
  - late: state −0.163 [−0.193, −0.134], control −0.088; gap −0.075 [−0.108, −0.042] (novel subset −0.063 [−0.110, −0.016]).
  - Refit vs original directions on the same episodes: late +0.033 weaker; all-layer +0.163 weaker (−0.363 vs −0.526).
  - All-layer: state − control +0.065 strict / +0.154 exact bet (the control is the more disruptive one).
- **G4b (new contexts, seeds 100–116, + new frozen mappings; shared vocabulary, templates, model, 9/18 topic–state pairs):**
  - Gate 0.931 (passed).
  - late: state −0.137 [−0.173, −0.105], control −0.056; gap −0.082 [−0.124, −0.039] (exact bet −0.098).
  - All-layer: state −0.212, control −0.458; gap +0.245 (the control is far more disruptive).

## Research verdict (exploratory)
1. **The late (post-onset) state-span removal effect beats the control.** In 7B it is modest (−0.14 at L18, leaving
   covert decoding at 0.82) and larger than an orthogonal PC2 control's, including when the control is attenuated to a
   matched uncentred-energy dose (gap −0.13). The same gap direction recurs on fresh messages (−0.11), refit directions
   (−0.08), new contexts and mappings (−0.08), L20 (−0.10) and the post hoc 14B L26 cutoff (−0.09).
   - It is NOT robust to every predeclared choice: it reverses at L16 (+0.04).
   - It is absent at the frozen 14B L32 cutoff (−0.01) and absent for 32B late removal (+0.01).
   - Removal reduces covert decoding; it never eliminates it.
2. **The detector-direction effect is a placement artefact.** It vanishes at late placement in 7B (−0.01 vs −0.23 at all
   layers).
3. **The extra all-layer effect is pre-onset and not cleanly selective.** It comes from the pre-onset hooks (7B ≈ −0.32 to
   −0.37; 32B −0.10). In 7B it is mostly format failure relative to the PC2 control (exact-bet gap −0.04); in 32B it comes
   with ARC damage; in 14B both spans wreck the output. At all-layer placement the control is often as disruptive or more
   (14B, G4a, G4b).
4. **The embedding hook adds ~nothing** (≤ ±0.01).

## Remaining uncertainties
- **Dose matching** holds at the first hooked layer and in aggregate, not per downstream layer. It is one measure: centred
  variance ranks the spans the other way, and the decision-token preview is not the full trajectory.
- **The L16 reversal** is unexplained; possible factors are unstable L16–17 spans and different downstream trajectories.
- **Inference is limited** to 6 fixed schemes per setting, one model family, and greedy decoding.
  - Fresh sets and G4a share contexts and mappings.
  - G4b shares vocabulary, templates, task, model and half the topic–state pairs. Neither is an independent replication.
- **Cluster vs episode CIs.** Some cluster CIs are narrower than episode CIs (negative within-context correlation when
  removal collapses decoding toward one state); both are reported.
- **Readout caveat stands.** The first-hooked-layer readout dip is partly built into the method (span and readout fitted on
  the same training episodes). Higher readout one layer later is not shown to be recovery or reconstruction.

## Post-submission: L16 cutoff explained (reviewer follow-up; `python3 exp/l16_error_direction.py`, CPU, existing S1 runs)
- State and control ablations break decoding in opposite directions. State ablation pushes true COLD/NEUTRAL answers to
  HOT (0.015 -> 0.250 at L18). The PC2 control breaks true HOT answers to other states (0.039 -> 0.196 at L18).
- Scored on its own signature (->HOT), the state effect beats the control at every cutoff, L16 included:
  L16 +0.108 [+0.069, +0.157], L18 +0.240, L20 +0.167 (episode bootstrap, not cluster).
- The accuracy gap reverses at L16 for two reasons. First, the control's HOT->other damage grows (0.196 -> 0.324).
  Second, the state's ->HOT collapse halves (0.250 -> 0.118) and the state hook adds HOT->other errors of its own (0.029 -> 0.069).
- The L16 state hook removes 23046 energy/token (3.7% of ||h||^2), 17x the L18 hook. The L16 direction is unstable
  (bootstrap angle 27 deg vs 7 deg at L18+) and mostly uncentred (centred share 0.05 vs ~0.35 at L18+; phase0 files).
  So the L16 hook acts mostly as a large generic perturbation with the control's error signature.
- Size test (GPU, exp/verify_matrix.py s1s, 13 configs; f0_M2_L16_repro reproduced the S1 run exactly, 51/51
  outputs identical). Each arm's L16/L17 dose was scaled to that arm's own L18 energy (state L16 23046 -> 1362/token).
  Prediction: the ->HOT collapse returns to ~0.25 and the gap comes back. **Mostly refuted.**
  - The state ->HOT rate moves only 0.118 -> 0.157 (+0.039 [+0.010, +0.074]) and stays below L18 (-0.093 [-0.132, -0.054]).
  - The control HOT->other rate moves 0.324 -> 0.304 and stays above L18 (+0.108 [+0.049, +0.176]).
  - Accuracy: state 0.876, control 0.856, so the gap stays reversed (+0.02).
  - The ->HOT signature still separates the arms: +0.147 [+0.098, +0.196].
  - Centring (subtracting the mean before projecting) was also checked in an 8-episode smoke test. It barely changes the
    L16 energy (14118 -> 13970), so the large L16 removal is token-level variance, not a shared offset.
- Conclusion: the reversal is not caused by the size of the L16 hook. Even at the L18 energy, hooking L16/L17 (i) halves the state
  arm's ->HOT collapse and (ii) makes the control more damaging to HOT answers. Both point to the direction/placement
  at L16/L17 mattering, not the dose. Where the direction is unstable (L16-17, ~27 deg bootstrap angle), editing it
  appears to interfere with how L18+ forms the state rather than just adding noise. Next test (not run): put the stable
  L18 direction at L16/L17 instead of the unstable fitted ones.
- Direction test (GPU, exp/verify_matrix.py s1x): hook L16/L17 with the stable L18 subspace in place of the fitted
  L16/L17 subspaces, which are ~70-83 deg (principal angles) from L18 in fold 0. **Confirmed, strongly.**
  - The state ->HOT rate rises to 0.382 (L18: 0.250). Accuracy is 0.735; L16x minus L18 is -0.082 [-0.114, -0.049].
  - State minus control (accuracy) is -0.163 [-0.219, -0.101], the largest late gap in the study. On the ->HOT signature it is +0.373 [+0.309, +0.436].
  - C6 1.0, ARC 0.96-0.97, recall 0.86-1.0.
  - The removed energy at L16 is 1357/token, matching the L18 hook (1362). L17 and L18 are then ~0: once the component is
    removed at L16, layers 16-17 do not rewrite it.
- Answer to the L16 question: the cutoff sensitivity is a direction-estimation failure, not an absent state.
  - The state carrier already exists at L16, in the L18 subspace. Removing it there is more effective than removing it at L18.
  - The subspace fitted at L16 (state decodability 0.54 there) misses that carrier. It mostly removes large non-state token variance.
  - Post hoc caveat: this config was designed after the S1 cutoff results, so treat it as a follow-up, not a pre-registered test.
