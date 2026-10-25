"""Local (CPU) probe training + intervention directions + post-intervention detector evaluation.

  .venv-analysis/bin/python exp/probes.py baseline              # probes on data/acts -> data/directions, data/probes.json
  .venv-analysis/bin/python exp/probes.py post data/interv/<tag> # detector AUROC on intervened activations
  DATA_DIR=data_models/qwen14b .venv-analysis/bin/python exp/probes.py baseline   # another model's namespace

Splits are leave-one-scheme-out (6 folds); paired conditions share episodes, so both sides of a pair are always in
the same fold. Probe = StandardScaler + L2 logistic regression. Intervention directions are raw residual-space
difference-of-means (Arditi et al. 2024 style), unit-normalised per layer.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
# DATA_DIR (env var, relative to the project root or absolute) selects the model namespace, e.g. data_models/qwen14b.
DATA = ROOT / os.environ.get("DATA_DIR", "data")
ACTS, DIRS = DATA / "acts", DATA / "directions"
STATES = ["COLD", "NEUTRAL", "HOT"]
# name -> (positive condition, negative condition, pooling)
CONTRASTS = {
    "detect_C1vC4_mean": ("C1", "C4", "mean"),       # primary: identical rx prompt, both msgs mention a topic
    "detect_C1vC2_mean": ("C1", "C2", "mean"),       # identical rx prompt, honest msg
    "detect_C1vC4_full": ("C1", "C4", "full_last"),  # Rose et al. pooling (end-of-turn token)
    "detect_C1vC3_full": ("C1", "C3", "full_last"),  # NARCBench-style: colluding vs honest rx prompt (confounded)
    "detect_C1vC5_mean": ("C1", "C5", "mean"),       # scheme-mismatch: identical message, msg is not a signal for rx
    "detect_C1vC5_last": ("C1", "C5", "last"),
    "detect_C1vC5_full": ("C1", "C5", "full_last"),
}
PRIMARY = "detect_C1vC4_mean"


def load(d, cond, pool):
    return np.load(d / f"{cond}.npz")[pool].astype(np.float32)


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))


def loso_auroc(Xp, Xn, groups):
    """Leave-one-scheme-out AUROC of a fresh probe at one layer. Xp/Xn: [N,D] paired by episode."""
    X, y, g = np.concatenate([Xp, Xn]), np.r_[np.ones(len(Xp)), np.zeros(len(Xn))], np.r_[groups, groups]
    s = np.zeros(len(y))
    for k in np.unique(groups):
        te = g == k
        s[te] = lr().fit(X[~te], y[~te]).decision_function(X[te])
    return roc_auc_score(y, s)


def loso_state_acc(X, state, groups):
    s = np.empty(len(X), object)
    for k in np.unique(groups):
        te = groups == k
        s[te] = lr().fit(X[~te], state[~te]).predict(X[te])
    return float((s == state).mean())


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def baseline():
    meta = json.load(open(ACTS / "meta.json"))
    groups, state = np.array(meta["scheme_id"]), np.array(meta["state"])
    L = meta["n_layers"]
    DIRS.mkdir(parents=True, exist_ok=True)
    out = dict(n=len(groups), n_layers=L, auroc={}, state_acc={}, cos={})
    rng = np.random.default_rng(0)

    for name, (p, n, pool) in CONTRASTS.items():
        Xp, Xn = load(ACTS, p, pool), load(ACTS, n, pool)
        out["auroc"][name] = [loso_auroc(Xp[:, l], Xn[:, l], groups) for l in range(L)]
        dm = unit(Xp.mean(0) - Xn.mean(0))                       # [L,D] diff-of-means, raw space
        np.save(DIRS / f"{name}_dm.npy", dm)
        if name == PRIMARY:                                      # LR direction in raw space (w/sigma) for comparison
            w = []
            for l in range(L):
                m = lr().fit(np.r_[Xp[:, l], Xn[:, l]], np.r_[np.ones(len(Xp)), np.zeros(len(Xn))])
                w.append(m[-1].coef_[0] / m[0].scale_)
            w = unit(np.array(w))
            np.save(DIRS / f"{name}_lr.npy", w)
            out["cos"]["primary_dm_vs_lr"] = (dm * w).sum(-1).tolist()
        print(name, "best layer", int(np.argmax(out["auroc"][name])), round(max(out["auroc"][name]), 3), flush=True)

    # state (content of the covert message) at the receiver's decision token, C1 only
    X = load(ACTS, "C1", "last")
    out["state_acc"]["C1_last"] = [loso_state_acc(X[:, l], state, groups) for l in range(L)]
    mu = {s: X[state == s].mean(0) for s in STATES}
    v1, v2 = mu["HOT"] - mu["COLD"], mu["NEUTRAL"] - (mu["HOT"] + mu["COLD"]) / 2
    Q = np.stack([np.linalg.qr(np.stack([v1[l], v2[l]], 1))[0].T for l in range(L)])  # [L,2,D] orthonormal
    np.save(DIRS / "state_C1_last_2d.npy", Q)
    dm = np.load(DIRS / f"{PRIMARY}_dm.npy")
    out["cos"]["detect_vs_state_maxabs"] = np.abs(np.einsum("ld,lkd->lk", dm, Q)).max(-1).tolist()
    print("state best layer", int(np.argmax(out["state_acc"]["C1_last"])), round(max(out["state_acc"]["C1_last"]), 3))

    # random controls: matched dimensionality (1-D and 2-D), fresh per layer
    D = X.shape[-1]
    for i in range(5):
        np.save(DIRS / f"random1d_{i}.npy", unit(rng.standard_normal((L, D))))
        np.save(DIRS / f"random2d_{i}.npy", np.stack([np.linalg.qr(rng.standard_normal((D, 2)))[0].T for _ in range(L)]))
    json.dump(out, open(DATA / "probes.json", "w"), indent=1)
    return out


def post(d):
    """Detector evaluation on intervened activations: (a) frozen probe trained on baseline acts, (b) re-trained
    probe on intervened acts (amnesic-probing logic: is the information still linearly recoverable?)."""
    d = Path(d)
    meta = json.load(open(ACTS / "meta.json"))
    groups, state = np.array(meta["scheme_id"]), np.array(meta["state"])
    res = dict(frozen={}, retrained={})
    for name, (p, n, pool) in CONTRASTS.items():
        if not ((d / f"{p}.npz").exists() and (d / f"{n}.npz").exists()):
            continue
        Bp, Bn, Ip, In = load(ACTS, p, pool), load(ACTS, n, pool), load(d, p, pool), load(d, n, pool)
        L = Bp.shape[1]
        fz, rt = [], []
        y, g = np.r_[np.ones(len(Bp)), np.zeros(len(Bn))], np.r_[groups, groups]
        for l in range(L):
            B, I, s = np.r_[Bp[:, l], Bn[:, l]], np.r_[Ip[:, l], In[:, l]], np.zeros(len(y))
            for k in np.unique(groups):  # frozen probe trained on baseline, other schemes; scored on intervened
                te = g == k
                s[te] = lr().fit(B[~te], y[~te]).decision_function(I[te])
            fz.append(roc_auc_score(y, s))
            rt.append(loso_auroc(Ip[:, l], In[:, l], groups))
        res["frozen"][name], res["retrained"][name] = fz, rt
    if (d / "C1.npz").exists():
        X = load(d, "C1", "last")
        res["state_acc_retrained"] = [loso_state_acc(X[:, l], state, groups) for l in range(X.shape[1])]
    json.dump(res, open(d / "post_probes.json", "w"), indent=1)
    return res


def post_excl(d, dirfile):
    """Fairer frozen detector: train on baseline acts with the ablated subspace already projected out, so the
    detector cannot be 'blinded' merely by construction; then score intervened acts. Adds res['frozen_excl']."""
    d = Path(d)
    Q = np.load(dirfile).astype(np.float32)
    Q = Q[:, None] if Q.ndim == 2 else Q                       # [L,k,D]
    Q = np.stack([np.linalg.qr(q.T)[0].T for q in Q])          # orthonormal rows per layer
    groups = np.array(json.load(open(ACTS / "meta.json"))["scheme_id"])
    res = json.load(open(d / "post_probes.json"))
    res["frozen_excl"] = {}
    for name, (p, n, pool) in CONTRASTS.items():
        if not ((d / f"{p}.npz").exists() and (d / f"{n}.npz").exists()):
            continue
        Bp, Bn, Ip, In = load(ACTS, p, pool), load(ACTS, n, pool), load(d, p, pool), load(d, n, pool)
        y, g, out = np.r_[np.ones(len(Bp)), np.zeros(len(Bn))], np.r_[groups, groups], []
        for l in range(Bp.shape[1]):
            B = np.r_[Bp[:, l], Bn[:, l]]
            B = B - (B @ Q[l].T) @ Q[l]
            I, s = np.r_[Ip[:, l], In[:, l]], np.zeros(len(y))
            for k in np.unique(groups):
                te = g == k
                s[te] = lr().fit(B[~te], y[~te]).decision_function(I[te])
            out.append(roc_auc_score(y, s))
        res["frozen_excl"][name] = out
    json.dump(res, open(d / "post_probes.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    {"baseline": lambda: baseline(), "post": lambda: post(sys.argv[2]),
     "post_excl": lambda: post_excl(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
