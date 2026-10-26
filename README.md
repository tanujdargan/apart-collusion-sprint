# Cut the Wire: Collusion-Probe Ablation Effects Shrink or Relocate Under Format, Dose and Placement Controls

Apart Research AI Collusion Research Sprint (Oct 23–25 2026). A causal test (directional ablation + steering) of
activation-based collusion detectors in a controlled steganographic signalling game ("Signal Table"), on
Qwen2.5-7B-Instruct, with replication on Qwen2.5-14B/32B-Instruct and Phi-3.5-mini-instruct.

**Paper (submitted version): [Cut-the-Wire.pdf](Cut-the-Wire.pdf)**

- **REPORT_NOTES.md**: claims, abstract, method, all results tables, figure captions, limitations, outline
- **figures/**: Fig 1–6 (PNG + PDF)
- **exp/**: code (`env.py` prompts, `modal_app.py` GPU pipeline on Modal, `probes.py`, figure scripts); see `exp/README.md`
- **notes/**: gate, run log, replication, related work + BibTeX, notes on Rose et al. 2026
- **supp_narcbench/**: confound audit of NARCBench's released activations
- **data/**, **data_models/**: messages, receiver outputs, metrics (large activation arrays are not committed;
  regenerate them with the Modal pipeline)
