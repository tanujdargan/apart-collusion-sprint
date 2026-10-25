"""Held-out-scheme (LOSO) refit of the intervention directions — post-hoc robustness check (notes/loso_robustness_plan.md).

  .venv-analysis/bin/python exp/loso_fit.py check   # refit on ALL episodes; must reproduce data/directions/* exactly
  .venv-analysis/bin/python exp/loso_fit.py fit     # per held-out scheme k: fit on other schemes -> data/loso/fold{k}/
  DATA_DIR=data_models/qwen14b .venv-analysis/bin/python exp/loso_fit.py {check|global|fit}   # other models
  (global: fit on ALL episodes -> <DATA_DIR>/loso/global/, the in-sample "global" configs for models without originals)

Construction is identical to the originals (exp/probes.py baseline + the inline PC controls of REPORT_NOTES §4.5a),
except that only the training episodes are used for means, centring, scaling and SVD.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
# DATA_DIR selects the model namespace (default data/ = Qwen2.5-7B; e.g. data_models/qwen14b, data_models/qwen32b).
DATA = ROOT / os.environ.get("DATA_DIR", "data")
ACTS, DIRS, OUT = DATA / "acts", DATA / "directions", DATA / "loso"
STATES = ["COLD", "NEUTRAL", "HOT"]


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def load():
    meta = json.load(open(ACTS / "meta.json"))
    A = {c: np.load(ACTS / f"{c}.npz") for c in ("C1", "C4")}
    return (meta, A["C1"]["mean"].astype(np.float32), A["C4"]["mean"].astype(np.float32),
            A["C1"]["last"].astype(np.float32))


def fit(idx, P, N, X, state, with_lr=True):
    """idx: training row indices. P/N: C1/C4 `mean` acts [N,L,D]; X: C1 `last` acts. Returns dict of directions."""
    P, N, X, st = P[idx], N[idx], X[idx], state[idx]
    L = P.shape[1]
    d = {}
    d["dm"] = unit(P.mean(0) - N.mean(0))                                         # [L,D]
    mu = {s: X[st == s].mean(0) for s in STATES}
    v1, v2 = mu["HOT"] - mu["COLD"], mu["NEUTRAL"] - (mu["HOT"] + mu["COLD"]) / 2
    d["state"] = np.stack([np.linalg.qr(np.stack([v1[l], v2[l]], 1))[0].T for l in range(L)])  # [L,2,D]
    pc2, pc1 = [], []
    for l in range(L):
        Xc = X[:, l] - X[:, l].mean(0)
        Xo = Xc - (Xc @ d["state"][l].T) @ d["state"][l]
        pc2.append(np.linalg.svd(Xo, full_matrices=False)[2][:2])
        Z = np.r_[P[:, l], N[:, l]]
        Z = Z - Z.mean(0)
        Zo = Z - np.outer(Z @ d["dm"][l], d["dm"][l])
        pc1.append(np.linalg.svd(Zo, full_matrices=False)[2][0])
    d["pc2d"], d["pc1d"] = np.stack(pc2), np.array(pc1)
    if with_lr:
        w = []
        y = np.r_[np.ones(len(P)), np.zeros(len(N))]
        for l in range(L):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(np.r_[P[:, l], N[:, l]], y)
            w.append(m[-1].coef_[0] / m[0].scale_)
        d["lr"] = unit(np.array(w))
    return d


def var_share(idx, P, N, X, d):
    """Share of train-set variance captured, mean over layers (dm/pc1d/lr on C1+C4 mean; state/pc2d on C1 last)."""
    out = {}
    for k, v in d.items():
        r = []
        for l in range(P.shape[1]):
            Z = np.r_[P[idx, l], N[idx, l]] if k in ("dm", "pc1d", "lr") else X[idx, l]
            Z = Z - Z.mean(0)
            Q = v[l][None] if v.ndim == 2 else v[l]
            Q = np.linalg.qr(Q.T)[0].T
            r.append(float(((Z @ Q.T) ** 2).sum() / (Z ** 2).sum()))
        out[k] = dict(mean=float(np.mean(r)), L20=r[20])
    return out


def subspace_agreement(a, b):
    """Mean over layers of ||Qa^T Qb||_F^2 / k (1 = identical span)."""
    a = a[:, None] if a.ndim == 2 else a
    b = b[:, None] if b.ndim == 2 else b
    s = []
    for x, y in zip(a, b):
        qa, qb = np.linalg.qr(x.T)[0], np.linalg.qr(y.T)[0]
        s.append(float((np.linalg.norm(qa.T @ qb) ** 2) / qa.shape[1]))
    return float(np.mean(s)), float(np.min(s))


ORIG = dict(dm="detect_C1vC4_mean_dm", state="state_C1_last_2d", pc2d="pc_ctrl_2d", pc1d="pc_ctrl_1d",
            lr="detect_C1vC4_mean_lr")

if __name__ == "__main__":
    meta, P, N, X = load()
    state, scheme, ids = np.array(meta["state"]), np.array(meta["scheme_id"]), meta["ids"]
    if sys.argv[1] in ("check", "global"):
        d = fit(np.arange(len(ids)), P, N, X, state)
        for k, f in ORIG.items():
            if (DIRS / f"{f}.npy").exists():
                print(k, "agreement with original (mean, min over layers):", subspace_agreement(d[k], np.load(DIRS / f"{f}.npy")))
            else:
                print(k, "no original file", DIRS / f"{f}.npy")
        if sys.argv[1] == "global":
            gd = OUT / "global"
            gd.mkdir(parents=True, exist_ok=True)
            for name, v in d.items():
                np.save(gd / f"{name}.npy", v.astype(np.float32))
            json.dump(dict(n=len(ids), var_share_all=var_share(np.arange(len(ids)), P, N, X, d),
                           fit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                      open(gd / "provenance.json", "w"), indent=1)
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        sha = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (ACTS / "C1.npz", ACTS / "C4.npz", ACTS / "meta.json")}
        splits, prov = {}, dict(input_sha256=sha, fit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                construction={k: f"same as data/directions/{v}.npy, train episodes only" for k, v in ORIG.items()},
                                folds={})
        for k in sorted(set(scheme.tolist())):
            tr, ev = np.where(scheme != k)[0], np.where(scheme == k)[0]
            d = fit(tr, P, N, X, state)
            fd = OUT / f"fold{k}"
            fd.mkdir(exist_ok=True)
            for name, v in d.items():
                np.save(fd / f"{name}.npy", v.astype(np.float32))
            splits[str(k)] = dict(train_ids=[ids[i] for i in tr], eval_ids=[ids[i] for i in ev])
            prov["folds"][str(k)] = dict(n_train=len(tr), n_eval=len(ev), train_schemes=sorted(set(scheme[tr].tolist())),
                                         var_share_train=var_share(tr, P, N, X, d),
                                         agreement_with_global={n: subspace_agreement(d[n], np.load(DIRS / f"{ORIG[n]}.npy"))[0]
                                                                for n in d if (DIRS / f"{ORIG[n]}.npy").exists()})
            print(k, len(tr), len(ev), {n: round(v, 3) for n, v in prov["folds"][str(k)]["agreement_with_global"].items()})
        json.dump(splits, open(OUT / "splits.json", "w"), indent=1)
        json.dump(prov, open(OUT / "provenance.json", "w"), indent=1)
