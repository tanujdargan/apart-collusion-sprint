# Cut the Wire: Collusion-Probe Ablation Effects Shrink or Relocate Under Format, Dose and Placement Controls

Apart Research AI Collusion Research Sprint (Oct 23–25 2026). A causal test (directional ablation + steering) of
activation-based collusion detectors in a controlled steganographic signalling game ("Signal Table"), on
Qwen2.5-7B-Instruct, with replication on Qwen2.5-14B/32B-Instruct and Phi-3.5-mini-instruct.

**Paper (submitted version): [Cut-the-Wire.pdf](Cut-the-Wire.pdf)**

- **REPORT_NOTES.md**: working notes (claims, results tables, review rounds); the PDF is the authoritative version
- **exp/**: code (`env.py` prompts, `modal_app.py` GPU pipeline on Modal, `probes.py`, analysis and figure scripts)
- **figures/**, **supp_narcbench/figs/**: all paper figures (PNG + PDF; mapping below)
- **notes/**: gate, run log, replication, related work + BibTeX, notes on Rose et al. 2026
- **supp_narcbench/**: confound audit of NARCBench's released activations (`analysis.py`)
- **data/**, **data_models/**: messages, receiver outputs, metrics (large activation arrays are not committed;
  regenerate them with the Modal pipeline)

## Reproducing

There is no single `make` target; run the stages in order as listed in [`exp/README.md`](exp/README.md)
(Modal L40S, about $16 total for every run in the paper). The NARCBench audit runs on CPU:
`python3 supp_narcbench/analysis.py`.

## Paper figure → file

| Paper | File |
|---|---|
| Fig 1 (setup) | `figures/fig1_setup` |
| Fig 2 (NARCBench length confound) | `supp_narcbench/figs/confound_auroc.png` |
| Fig 3 (placement and dose) | `figures/fig12_placement` |
| S1 (specificity map) | `figures/fig4_2x2` |
| S2 (NARCBench run-level score vs length) | `supp_narcbench/figs/runlevel_probe_vs_ntok.png` |
| S3 (detector AUROC by layer) | `figures/fig2_probe_layers` |
| S4 (token spike) | `figures/fig3_token_trace` |
| S5 (steering) | `figures/fig5_steer` |
| S6 (cross-model depth) | `figures/fig6_models` |
| S7 (held-out-scheme refits) | `figures/fig7_loso` |
| S8 (fresh messages) | `figures/fig8_fresh` |
| S9 (cross-model summary) | `figures/fig9_xmodel` |
| S10 (decodability after ablation) | `figures/fig10_redundancy` |
