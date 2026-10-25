"""Fig 10: covert-state decodability (LOSO 3-way, C1 decision token) before vs after ablating each model's global
(in-sample) 2-D state subspace at all layers, by relative depth. 7B/14B: data/.../interv/abl_state_C1_last_2d/
post_probes.json (state_acc_retrained); 32B: verification run V1 (exp/verify_32b_ablation.py).
  .venv-analysis/bin/python exp/fig_redundancy.py -> figures/fig10_redundancy.{png,pdf}"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
M = [("7B", json.load(open(ROOT / "data/probes.json"))["state_acc"]["C1_last"],
      json.load(open(ROOT / "data/interv/abl_state_C1_last_2d/post_probes.json"))["state_acc_retrained"], "#0072B2"),
     ("14B", json.load(open(ROOT / "data_models/qwen14b/probes.json"))["state_acc"]["C1_last"],
      json.load(open(ROOT / "data_models/qwen14b/interv/abl_state_C1_last_2d/post_probes.json"))["state_acc_retrained"], "#E69F00"),
     ("32B", *(lambda v: (v["state_acc_baseline"], v["state_acc_ablated_retrained"]))(
         json.load(open(ROOT / "data_models/qwen32b/interv/verify_state_global_acts/verify.json"))), "#CC79A7")]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(4.2, 2.6))
for name, base, abl, c in M:
    x = np.arange(len(base)) / (len(base) - 1)
    ax.plot(x, base, color=c, lw=1.2, label=f"{name} baseline")
    ax.plot(x, abl, color=c, lw=1.2, ls="--", label=f"{name} after state ablation")
ax.axhline(1 / 3, color="k", lw=0.5, ls=":")
ax.set(xlabel="relative depth l/(L−1)", ylabel="state acc. (LOSO, retrained)", ylim=(0, 1.02),
       title="State info left after 2-D state ablation", )
ax.legend(fontsize=5.5, frameon=False, ncol=2, loc="upper left")
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(ROOT / f"figures/fig10_redundancy.{ext}", dpi=300)
