"""Write <DATA_DIR>/directions/state_hotcold_L<l>.npy (unit muHOT-muCOLD, C1 decision token) + _norm.json.
Usage: DATA_DIR=data_models/<tag> .venv-analysis/bin/python exp/make_steer_dir.py <layer>"""
import json
import os
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent.parent / os.environ.get("DATA_DIR", "data")
l = int(sys.argv[1])
X = np.load(D / "acts/C1.npz")["last"][:, l].astype(np.float32)
s = np.array(json.load(open(D / "acts/meta.json"))["state"])
v = X[s == "HOT"].mean(0) - X[s == "COLD"].mean(0)
n = float(np.linalg.norm(v))
np.save(D / f"directions/state_hotcold_L{l}.npy", v / n)
json.dump(dict(layer=l, norm=n, last_token_norm=float(np.linalg.norm(X, axis=-1).mean())),
          open(D / f"directions/state_hotcold_L{l}_norm.json", "w"))
print(l, n, float(np.linalg.norm(X, axis=-1).mean()))
