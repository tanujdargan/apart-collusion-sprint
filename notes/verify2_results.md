# Final targeted extension: results (plan notes/verify2_plan.md; log notes/verify2_log.md; tables data/verify/v2_tables.md)
7B, original messages, 6 LOSO folds × 51 held-out episodes, fold-fit directions, committed doses.
CIs: context-cluster bootstrap (102 clusters; none recur across schemes). S5 is seed-averaged within episode, with a Monte
Carlo SE over the 8 seeds.

## Validation status
- **Full-dose removal** at the intended locations: ≤ 4.1e-4 in prefill AND generation, for dm/pc1d at
  L18/all/EMB_LATE/PRE_LATE (greedy) and M2/dm/pc1d (sampled). Upstream prefill is bitwise equal to none.
- **Fractional dose, same incoming state:** span ratio within 0.007 of 1 − d; energy ratio 0.99–1.03; orthogonal part
  of the removed vector ≤ 19% (bf16 rounding).
- **Greedy anchors** reproduce S1 exactly: all 9 S5 cells, and the S6 none and dm_L18 cells. S1 reuse is valid.
  - The S6 gate first reported a failure. The cause was an analysis bug (C1 compared against C6), found after the outcome;
    the fix and its evidence are logged.
- **Sampling config:** only TemperatureLogitsWarper(0.7) is active. Model defaults are overridden (CPU check).
- **Repeat-seed and cross-container checks** were identical, but only on a 9-episode subset where no draw left greedy:
  WEAK evidence. Determinism of divergent sampled draws is not established.
- **How much sampling actually changed** (descriptive, outside the frozen analysis): T = 0.7 changed only 0.4% (none) to
  6.7% (M2_L18_match) of draws relative to greedy. This setting is a MILD perturbation of this confident receiver.

## S5: sampled receiver (T = 0.7, top_p = 1.0, seeds 0–7)
| contrast | strict Δ [CI] (MC SE) | exact bet Δ [CI] | greedy Δ (S1) | sampled − greedy [CI] |
|---|---|---|---|---|
| M2_L18 − PC2_L18 | −0.092 [−0.124, −0.058] (0.005) | −0.092 [−0.125, −0.059] | −0.082 | −0.010 [−0.030, +0.010] |
| M2_L18_match − PC2_L18_match | −0.147 [−0.178, −0.117] (0.006) | −0.151 [−0.181, −0.120] | −0.131 | −0.017 [−0.038, +0.004] |
| dm_L18 − pc1d_L18 | −0.011 [−0.021, −0.002] (0.002) | −0.012 [−0.022, −0.002] | −0.010 | −0.002 [−0.012, +0.008] |
| dm_all − pc1d_all | −0.167 [−0.199, −0.134] (0.006) | −0.229 [−0.261, −0.197] | −0.180 | +0.013 [−0.009, +0.034] |

- **Absolute (strict / exact bet / malformed):**
  - none 0.960 / 0.960 / 0.000; M2_L18 0.800 / 0.800 / 0.001; PC2_L18 0.891 / 0.892 / 0;
  - M2_match 0.801 / 0.803 / 0.002; PC2_match 0.949 / 0.953 / 0.004;
  - dm_L18 0.949 / 0.949 / 0; dm_all 0.729 / 0.729 / 0;
  - pc1d_L18 0.960 / 0.960 / 0; pc1d_all 0.896 / 0.958 / 0.062.
  - Valid-but-wrong is the complement of correct in every cell except pc1d_all, whose strict drop is entirely malformed
    output with correct bets.
- **Vs baseline:** sampled effects are slightly larger than greedy for the state cells (M2_L18 − none: sampled − greedy
  −0.019 [−0.037, −0.002]; M2_match −0.024 [−0.043, −0.005]) and for pc1d_all strict (−0.018, all malformed).
- **MC SE ≤ 0.007** for every contrast; seed-to-seed range about ±0.02.
- **Greedy capability checks** (C6/recall/ARC) are unchanged from S1. They were not sampled.

## S6: detector placement factorial (greedy; late removal always present)
| contrast | dm strict | pc1d strict | dm − pc1d strict | dm − pc1d exact bet |
|---|---|---|---|---|
| embedding effect, no early layers (EMB_LATE − L18) | −0.003 [−0.010, 0.000] | 0.000 | −0.003 [−0.010, 0.000] | −0.003 |
| embedding effect, with early layers (all − PRE_LATE) | −0.010 [−0.026, +0.003] | +0.003 | −0.013 [−0.033, +0.007] | −0.016 [−0.036, 0.000] |
| early-layer effect, no embedding (PRE_LATE − L18) | **−0.206 [−0.242, −0.170]** | −0.049 [−0.075, −0.026] | **−0.157 [−0.196, −0.118]** | −0.199 [−0.235, −0.163] |
| early-layer effect, with embedding (all − EMB_LATE) | **−0.212 [−0.252, −0.173]** | −0.046 | **−0.167 [−0.209, −0.124]** | −0.212 [−0.255, −0.173] |
| interaction | −0.007 [−0.023, +0.010] | +0.003 | −0.010 [−0.033, +0.010] | −0.013 [−0.033, +0.003] |

- **dm's early-layer effect is a clean decode change:** no malformed outputs, C6 unchanged, recall −0.016, ARC −0.027
  [−0.060, −0.002].
- **pc1d's early-layer effect is almost entirely format:** strict −0.049 but exact bet −0.007; malformed +0.042.

## Verdict
1. **Sampling (this setting only).** The 7B late state-vs-control effect is present under T = 0.7 sampled decoding:
   −0.09 at full dose, −0.15 at matched dose (MC SE ≤ 0.006). Sampled and greedy contrasts differ by only −0.010 and
   −0.017, with CIs including 0. That is not equivalence; the regimes were compared directly. The caveat is strong: this
   sampling changed ≤ 7% of draws, so it tests robustness to mild sampling noise, not to substantially different decoding.
2. **Detector placement.** The detector's clean all-layer effect comes from the EARLY decoder-layer hooks (L0–17):
   −0.21 with or without the embedding hook. It is selective relative to the matched 1-D control (dm − pc1d about −0.16
   strict / −0.20 exact bet), and it changes decodes without malformed output. The embedding hook contributes about
   nothing (≤ 0.01 in magnitude), and there is no detectable interaction (CIs include 0, with half-widths about 0.02).
   - Late-only, the detector does almost nothing (−0.011 sampled, −0.010 greedy).
   - So its effect needs early-inclusive placement. That makes it an early-layer effect that is direction-selective
     relative to pc1d, not generic disruption, though pc1d is one control, not variance-matched, with small capability
     costs (ARC −0.03).

## Uncertainty
- **Fixed design:** one temperature, 8 seeds, 6 fixed schemes, one model, original messages only.
- **The sampling perturbation was mild:** ≤ 7% of draws differed from greedy.
- **Determinism checks were weak:** repeat-seed and cross-container checks ran on non-divergent draws.
- **dm and pc1d differ in more than direction:** the comparison is against one orthogonal 1-D control; the two differ in
  variance and in the early-layer spans' stability.
- **Interaction is bounded loosely:** CIs of about ±0.02 rule out only large interactions.

## Spend
New spend under the separate $2.00 cap: computed upper $1.612, billed $1.205. All 33 apps are stopped. No retries,
refusals or extra configurations.
