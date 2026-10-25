# Redundancy-analysis execution log (plan: notes/redundancy_plan.md)

- 12:55 Unit check BEFORE any outer-fold result (7B, folds 0/3, layers 20/25):
  - Sequence-M rank 2 reproduces data/loso/fold<k>/state.npy exactly (subspace agreement 1.000000), and the
    training class means coincide after step 1 (max diff ~9e-7).
  - DEGENERACY FOUND: the next probe-weight step returns W = 0 (n_iter 0, intercepts 0, train acc 0.333). With
    balanced classes and identical class means, the gradient of the L2-regularised multinomial loss at W=0 is
    exactly zero and the loss is convex ⇒ W=0 is the global optimum. All first-order linear (logistic/linear-
    discriminant) information in a balanced training set lies in the ≤2-D class-mean span of that same set.
  - Consequence: the planned "fit removal on the training episodes, then train a fresh readout on the SAME training
    residuals" is chance BY CONSTRUCTION after mean-span removal. It cannot measure residual information, and
    sequence M cannot continue past rank 2.
  - DEVIATION (methodological, decided before any outer result; not outcome-driven): cross-fitting. Within each
    outer fold, for each of the 5 training schemes j, fit the removal span on the OTHER 4 training schemes and
    project scheme j's rows (out-of-fit residuals). Train the fresh readout on the pooled out-of-fit residuals of all
    5 training schemes. Project the held-out scheme with each of the 5 four-scheme spans and average the readout's
    class probabilities. Removal-span types (fixed now; no adaptive selection, no outer-accuracy-based choices):
      - M2: pooled 3-class mean contrast (rank 2; the original construction);
      - MS: span of per-scheme class-mean contrasts, 2 per fitted scheme (rank ≤ 8 with 4 fitted schemes; ≤ 10 when
        fitted on all 5 training schemes for behaviour). Principled scheme-specific state encodings, not PCs;
      - P_r: probe-weight (INLP) sequence fitted on the 4 schemes, ranks 2..16 (descriptive only).
  - The in-sample (non-cross-fitted) version is reported only as the degenerate reference (chance by construction).
  - The planned inner r* selection is DROPPED: no adaptive rank choice. Behavioural candidates are fixed a priori
    by construction (M2 rank 2 = original; MS rank ≤ 10). The P sequence is descriptive.
  - Static saved-activation analysis cannot detect downstream recomputation ("self-repair") under an actual
    ablation. The V1 0.91 result came from ablated forward passes, so that question needs captured activations
    under ablation (paid; proposed, not run).
- 12:45 unit check of cross-fitted design OK (7B fold 2 L22: degenerate ref 0.333 with coef norm 0.0). Full run 7b 32b 14b started (CPU).
