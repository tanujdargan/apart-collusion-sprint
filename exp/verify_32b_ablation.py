"""V1 verification (notes/followups_log.md): is the 32B state ablation applied correctly, and is the state still
linearly decodable afterwards? Compares baseline C1 `last` activations with those captured under ablation of the
global 32B state subspace (all 64 layers + embedding).
  .venv-analysis/bin/python exp/verify_32b_ablation.py -> data_models/qwen32b/interv/verify_state_global_acts/verify.json"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
from probes import loso_state_acc  # noqa: E402  (same recipe as all state-accuracy numbers)

D = ROOT / "data_models/qwen32b"
V = D / "interv/verify_state_global_acts"
meta = json.load(open(D / "acts/meta.json"))
assert json.load(open(V / "meta.json"))["ids"] == meta["ids"], "row order mismatch"
groups, state = np.array(meta["scheme_id"]), np.array(meta["state"])
B = np.load(D / "acts/C1.npz")["last"].astype(np.float32)
A = np.load(V / "C1.npz")["last"].astype(np.float32)
Q = np.load(D / "loso/global/state.npy").astype(np.float32)          # [64,2,5120]
Qo = np.stack([np.linalg.qr(q.T)[0].T for q in Q])
L = B.shape[1]
proj_b = [float(np.linalg.norm(B[:, l] @ Qo[l].T, axis=1).mean()) for l in range(L)]
proj_a = [float(np.linalg.norm(A[:, l] @ Qo[l].T, axis=1).mean()) for l in range(L)]
norm_a = [float(np.linalg.norm(A[:, l], axis=1).mean()) for l in range(L)]
acc_b = [loso_state_acc(B[:, l], state, groups) for l in range(L)]
acc_a = [loso_state_acc(A[:, l], state, groups) for l in range(L)]
out = dict(proj_norm_baseline=proj_b, proj_norm_ablated=proj_a, act_norm_ablated=norm_a,
           state_acc_baseline=acc_b, state_acc_ablated_retrained=acc_a,
           max_rel_residual_projection=float(max(a / n for a, n in zip(proj_a, norm_a))),
           late_mean_state_acc=dict(baseline=float(np.mean(acc_b[42:])), ablated=float(np.mean(acc_a[42:]))))
json.dump(out, open(V / "verify.json", "w"), indent=1)
print("max residual projection / act norm:", round(out["max_rel_residual_projection"], 5))
print("proj norm baseline L20/L42/L60:", [round(proj_b[i], 2) for i in (20, 42, 60)], "ablated:", [round(proj_a[i], 4) for i in (20, 42, 60)])
print("state acc L42+ baseline", round(out["late_mean_state_acc"]["baseline"], 3), "→ ablated (retrained)", round(out["late_mean_state_acc"]["ablated"], 3))
print("per layer ablated (every 4th):", [round(x, 2) for x in acc_a[::4]])
