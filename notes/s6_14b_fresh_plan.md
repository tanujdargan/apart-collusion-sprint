# S6 · 14B L26 retest on fresh messages (pre-registration, written before any S6 run)

Reviewer request: the 14B state effect was absent at the frozen L32 cutoff (gap -0.007) and present only at L26,
which was chosen after seeing results (-0.09 on the original messages). Retest with L26 fixed in advance on new data.

- **Data:** new Qwen2.5-14B colluding sender messages for the same 306 contexts, sampling seeds 1401, 1402, 1403
  (`modal run exp/modal_app.py::fresh14`; sampling settings as the original). 918 episodes.
- **Directions (frozen, no refit):** data_models/qwen14b/loso/fold{k}/{state,pc2d}.npy fitted on the ORIGINAL 14B
  messages without scheme k (exp/loso_fit.py; inputs match provenance hashes). Fold k is applied only to scheme-k
  episodes, so directions never saw the scheme or the messages.
- **Configs:** none; M2_L26 and PC2_L26 (layers 26-47, full dose) = PRIMARY; M2_L32 and PC2_L32 = secondary.
- **Primary outcome:** C1 strict decode, paired contrast M2_L26 - PC2_L26 pooled over 918 episodes, context-cluster
  bootstrap CI (cluster = original context id, 102 clusters). The original L26 gap was about -0.09.
- **Decision rule:** the 14B effect replicates if the upper CI bound is < 0. Report it as not replicated otherwise, whatever the point estimate.
- **Secondary:** ->HOT / HOT->other error split (exp/l16_error_direction.py logic); C6, ARC, recall; per-seed gaps; L32 gap.
- Matrix: data/verify/matrices/s6_14b_fresh.json (`python exp/verify_matrix.py s6`), run with `::verify --matrix s6_14b_fresh --maxc 6`.

## Results (run after the plan was committed in 142d117; `python exp/s6_analyze.py` -> data/verify/s6_analysis.json)
- Fresh 14B messages: parse_ok 306/306 per seed; clean 278 / 276 / 281 (original set: 273).
- **PRIMARY: M2_L26 - PC2_L26 C1 strict = -0.097 [-0.112, -0.082]** (918 episodes, 102 context clusters). **Replicates**
  under the decision rule. Per seed: -0.085 / -0.105 / -0.101. The original post hoc L26 gap was about -0.09.
- Absolute: none 0.995, M2_L26 0.885, PC2_L26 0.981 (control costs only -0.013).
- Error signature: same as 7B. State ablation pushes non-HOT answers to HOT (0.160 vs control 0.008; contrast +0.152 [+0.132, +0.172]).
- Capability: C6 1.0 in all configs; ARC 0.985 vs 0.99 (none); recall 1.0.
- Secondary L32: gap +0.007 [+0.000, +0.014], absent again, matching the original L32 result.
- Interpretation: in 14B the state is ablatable from L26 (rel. depth 0.55), as in 7B from L18 (0.67), but not from L32. The frozen relative-depth cutoff (L32) was too late for 14B.
