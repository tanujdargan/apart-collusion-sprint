"""Raw-space LR detector direction (no StandardScaler): per layer, unit(coef) of LogisticRegression(C=0.1) on C1 vs C4
`mean` activations, all episodes. Made inline during the sprint; this script reproduces that file (cosine 1.0 per layer).

  .venv-analysis/bin/python exp/make_lrraw.py           # writes <DATA_DIR>/directions/detect_C1vC4_mean_lrraw.npy
  .venv-analysis/bin/python exp/make_lrraw.py --check   # asserts agreement with the existing file, no writes
"""
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

from probes import ACTS, DIRS, load, unit

Xp, Xn = load(ACTS, "C1", "mean"), load(ACTS, "C4", "mean")
y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
w = unit(np.array([LogisticRegression(C=0.1, max_iter=3000).fit(np.r_[Xp[:, l], Xn[:, l]], y).coef_[0]
                   for l in range(Xp.shape[1])]))
f = DIRS / "detect_C1vC4_mean_lrraw.npy"
if "--check" in sys.argv:
    cos = (w * np.load(f)).sum(-1)
    assert cos.min() > 0.999, f"min cosine {cos.min():.4f}"
    print("lrraw check ok, min cosine", round(float(cos.min()), 5))
else:
    np.save(f, w)
