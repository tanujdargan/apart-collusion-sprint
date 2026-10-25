"""Variance-matched 2-D PC control: per layer, the top-2 principal components of C1 `last` (decision-token)
activations after projecting out that layer's state_C1_last_2d subspace. Also reports the share of C1-last variance
captured by the state subspace vs the PC control.

  DATA_DIR=data_models/qwen14b .venv-analysis/bin/python exp/make_pc_ctrl.py          # writes directions/pc_ctrl_2d.npy
                                                                                      #   + pc_ctrl_var_shares.json
  .venv-analysis/bin/python exp/make_pc_ctrl.py --check                               # compare to an existing file, no writes
Refuses to overwrite an existing pc_ctrl_2d.npy (so the 7B data/ files are never touched).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent.parent / os.environ.get("DATA_DIR", "data")
X = np.load(D / "acts/C1.npz")["last"].astype(np.float32)        # [N,L,D]
Qs = np.load(D / "directions/state_C1_last_2d.npy")               # [L,2,D] orthonormal
L = X.shape[1]
P, shares = [], {}
for l in range(L):
    Xc = X[:, l] - X[:, l].mean(0)
    Q = Qs[l]
    R = Xc - (Xc @ Q.T) @ Q                                       # state subspace projected out
    _, _, Vt = np.linalg.svd(R, full_matrices=False)
    pc = Vt[:2]                                                   # [2,D], orthonormal, orthogonal to Q
    tot = float((Xc ** 2).sum())
    shares[l] = dict(state=float(((Xc @ Q.T) ** 2).sum() / tot), pc2=float(((Xc @ pc.T) ** 2).sum() / tot))
    P.append(pc)
P = np.stack(P).astype(np.float32)
out = D / "directions/pc_ctrl_2d.npy"
if "--check" in sys.argv:
    ref = np.load(out)
    # subspace agreement per layer: mean squared cosine of principal angles (1 = same subspace; sign-invariant)
    agree = [float(np.linalg.norm(ref[l] @ P[l].T) ** 2 / 2) for l in range(L)]
    print("subspace agreement with existing file: min %.4f mean %.4f" % (min(agree), np.mean(agree)))
    vs = D / "variance_shares.json"
    if vs.exists():
        v = json.load(open(vs)).get("state_vs_pc2_var_share_C1last", {})
        dev = max(abs(v[str(l)][k] - shares[l][k]) for l in range(L) for k in ("state", "pc2") if str(l) in v)
        print("max abs deviation from variance_shares.json:", dev)
    sys.exit(0)
assert not out.exists(), f"{out} exists; refusing to overwrite"
np.save(out, P)
json.dump(shares, open(D / "directions/pc_ctrl_var_shares.json", "w"), indent=1)
print({l: {k: round(v, 3) for k, v in s.items()} for l, s in shares.items()})
