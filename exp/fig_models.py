"""Fig 6: cross-model probe summary. (a) covert-state decodability vs relative depth; (b) C1vC4 detector AUROC vs
relative depth. Models: Qwen2.5-7B (data/), Qwen2.5-14B, Phi-3.5-mini (data_models/<tag>/).
  .venv-analysis/bin/python exp/fig_models.py -> figures/fig6_models.{png,pdf}"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
M = [("Qwen2.5-7B", ROOT / "data/probes.json", "#0072B2"), ("Qwen2.5-14B", ROOT / "data_models/qwen14b/probes.json", "#D55E00"),
     ("Phi-3.5-mini", ROOT / "data_models/phi35mini/probes.json", "#009E73")]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.3))
for name, f, c in M:
    p = json.load(open(f))
    s = np.array(p["state_acc"]["C1_last"]); x = np.arange(len(s)) / (len(s) - 1)
    a.plot(x, s, color=c, label=name, lw=1.4)
    b.plot(x, p["auroc"]["detect_C1vC4_mean"], color=c, lw=1.4, label=name)
    b.plot(x, p["auroc"]["detect_C1vC5_last"], color=c, lw=1.0, ls="--")
a.axhline(1 / 3, color="k", lw=0.5, ls=":")
a.set(xlabel="relative depth (layer / (L−1))", ylabel="state acc. (held-out scheme)", ylim=(0, 1.02),
      title="(a) Covert state appears at ~half depth")
a.legend(fontsize=6, frameon=False)
b.axhline(0.5, color="k", lw=0.5, ls=":")
b.set(xlabel="relative depth", ylabel="AUROC (LOSO)", ylim=(0, 1.02), title="(b) C1vC4 (solid) vs C1vC5 (dashed)")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(ROOT / f"figures/fig6_models.{ext}", dpi=300)
