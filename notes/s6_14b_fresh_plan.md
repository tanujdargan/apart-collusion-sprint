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
