"""Held-out state decodability: sender-message-span MEAN vs DECISION-TOKEN activations (CPU only, saved unablated acts).

  .venv-analysis/bin/python exp/span_vs_decision.py  -> data/span_vs_decision.json, data/span_vs_decision.md

Representations (C1, unablated forward pass; decoder-layer outputs; exp/modal_app.py `_acts`):
  `mean` = mean over the sender-message tokens in the receiver prompt (a POOLED feature, not per-token evidence);
  `last` = last prompt token (decision position, `<|im_start|>assistant\\n`).
Protocol: outer leave-one-scheme-out (6 folds, 255 train / 51 held-out). StandardScaler + multinomial LR (C=0.1) fit on
the training schemes only; no hyperparameter, layer or representation selection. Each episode is evaluated once per
representation and layer. Summaries are fixed in advance: the mean over all layers and the mean over late layers
(l >= ceil(2(L-1)/3)), with a paired span−decision Δ per episode. CI: bootstrap over episodes (each context appears
once per representation) stratified by scheme, carrying both representations together (10k, seed 0).
Positive decodability shows availability in the UNABLATED model only.
"""
import json
import math
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"7b": ROOT / "data", "14b": ROOT / "data_models/qwen14b", "32b": ROOT / "data_models/qwen32b"}


def held_out_correct(X, y, g):
    out = np.zeros(len(y), bool)
    for k in np.unique(g):
        tr, te = g != k, g == k
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X[tr], y[tr])
        out[te] = m.predict(X[te]) == y[te]
    return out


def boot(v, g, seed=0, nb=10000):
    rng = np.random.default_rng(seed)
    idx = [np.where(g == s)[0] for s in np.unique(g)]
    b = np.array([np.concatenate([v[rng.choice(ix, len(ix))] for ix in idx]).mean() for _ in range(nb)])
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def run(m):
    d = MODELS[m]
    meta = json.load(open(d / "acts/meta.json"))
    y, g = np.array(meta["state"]), np.array(meta["scheme_id"])
    z = np.load(d / "acts/C1.npz")
    R = {"span_mean": z["mean"].astype(np.float32), "decision": z["last"].astype(np.float32)}
    L = R["decision"].shape[1]
    C = {}
    for rep, X in R.items():
        C[rep] = np.array(Parallel(n_jobs=-1)(delayed(held_out_correct)(X[:, l], y, g) for l in range(L)), float)  # [L,N]
    late = list(range(math.ceil(2 * (L - 1) / 3), L))
    out = dict(model=m, L=L, n=len(y), late=[late[0], L - 1], per_layer={}, summary={}, first_ge_0_8={}, per_fold_late={})
    for rep in R:
        out["per_layer"][rep] = C[rep].mean(1).tolist()
        out["first_ge_0_8"][rep] = next((l for l in range(L) if C[rep][l].mean() >= 0.8), None)
        for name, ls in (("all_layers", list(range(L))), ("late_layers", late)):
            v = C[rep][ls].mean(0)
            out["summary"][f"{rep}_{name}"] = dict(mean=float(v.mean()), ci=boot(v, g))
        out["per_fold_late"][rep] = {str(k): float(C[rep][late].mean(0)[g == k].mean()) for k in np.unique(g)}
    for name, ls in (("all_layers", list(range(L))), ("late_layers", late)):
        dv = C["span_mean"][ls].mean(0) - C["decision"][ls].mean(0)
        out["summary"][f"delta_span_minus_decision_{name}"] = dict(mean=float(dv.mean()), ci=boot(dv, g))
    return out


if __name__ == "__main__":
    res = {m: run(m) for m in MODELS}
    json.dump(res, open(ROOT / "data/span_vs_decision.json", "w"), indent=1)
    T = ["# Held-out state decodability: message-span mean vs decision token (unablated, CPU; exp/span_vs_decision.py)\n",
         "Outer LOSO (6 folds), training-only fitting, no selection; n = 306 per model; bootstrap over episodes stratified by "
         "scheme. Chance 0.333. `span_mean` is a pooled feature (mean over message tokens), not per-token evidence.\n",
         "| model | span_mean all-layer | decision all-layer | span_mean late | decision late | Δ span−decision late [CI] | first layer ≥0.8 (span / decision) |",
         "|---|---|---|---|---|---|---|"]
    for m, r in res.items():
        s = r["summary"]
        f = lambda k: f"{s[k]['mean']:.3f} [{s[k]['ci'][0]:.3f}, {s[k]['ci'][1]:.3f}]"
        T.append(f"| {m} | {f('span_mean_all_layers')} | {f('decision_all_layers')} | {f('span_mean_late_layers')} | "
                 f"{f('decision_late_layers')} | {f('delta_span_minus_decision_late_layers')} | "
                 f"{r['first_ge_0_8']['span_mean']} / {r['first_ge_0_8']['decision']} (of {r['L']}) |")
    T.append("\nPer-layer curves: data/span_vs_decision.json (per_layer).")
    (ROOT / "data/span_vs_decision.md").write_text("\n".join(T) + "\n")
    for m, r in res.items():
        print(m, {k: round(v["mean"], 3) for k, v in r["summary"].items()}, r["first_ge_0_8"])
