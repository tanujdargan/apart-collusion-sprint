"""Residual linear decodability of the covert state after removing state subspaces — CPU only, saved activations.
Plan: notes/redundancy_plan.md; DEVIATION (cross-fitting; logged before any outer result): notes/redundancy_log.md.

  .venv-analysis/bin/python exp/redundancy_analysis.py run 7b 32b 14b   -> data/redundancy/<model>.json
  .venv-analysis/bin/python exp/redundancy_analysis.py report           -> data/redundancy/summary.json, tables.md, figures/fig11_*

Outer folds = leave-one-scheme-out. Within outer fold k (training schemes T, |T| = 5), for each j in T: fit the
removal span on T\\{j} (4 schemes), centre with that 4-scheme mean, and project scheme j's rows ("out-of-fit
residuals"). A fresh readout (StandardScaler + multinomial LR, C=0.1) is trained on the pooled out-of-fit residuals
of all 5 training schemes. The held-out scheme is projected with each of the 5 four-scheme spans, and the readout's
class probabilities are averaged. Removal spans (fixed a priori):
  M2 = pooled 3-class mean contrast (rank 2; the original construction)
  MS = span of per-scheme class-mean contrasts (2 per fitted scheme; rank <= 8)
  P_r = probe-weight (INLP-style) sequence, ranks 2..16 (descriptive)
"degenerate" = M2 fitted on all 5 training schemes AND readout trained on the same residuals (chance by construction).
"""
import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
MODELS = {"7b": ROOT / "data", "14b": ROOT / "data_models/qwen14b", "32b": ROOT / "data_models/qwen32b"}
STATES = np.array(["COLD", "NEUTRAL", "HOT"])
P_RANKS = [2, 4, 6, 8, 10, 12, 14, 16]
TOL = 1e-6


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))


def late_layers(L):
    return list(range(int(np.ceil(2 * (L - 1) / 3)), L))


def gs(U, dirs):
    """Append Gram-Schmidt-orthonormalised rows of dirs to basis U (rows); drop near-dependent ones."""
    for v in dirs:
        v = np.asarray(v, np.float64)
        n0 = np.linalg.norm(v)
        if n0 == 0:
            continue
        for _ in range(2):
            if len(U):
                v = v - U.T @ (U @ v)
        if np.linalg.norm(v) / n0 < TOL:
            continue
        U = np.vstack([U, v / np.linalg.norm(v)]) if len(U) else (v / np.linalg.norm(v))[None]
    return U


def res(X, U):
    return X if len(U) == 0 else X - (X @ U.T) @ U


def mean_contrast(X, y):
    mu = {s: X[y == s].mean(0) for s in STATES}
    return np.stack([mu["HOT"] - mu["COLD"], mu["NEUTRAL"] - (mu["HOT"] + mu["COLD"]) / 2])


def span_M2(X, y):
    return gs(np.zeros((0, X.shape[1])), mean_contrast(X, y))


def span_MS(X, y, g):
    U = np.zeros((0, X.shape[1]))
    for s in np.unique(g):
        U = gs(U, mean_contrast(X[g == s], y[g == s]))
    return U


def spans_P(Xc, y):
    """INLP-style probe-weight sequence; returns {rank: basis}. Stops early if the probe weights vanish."""
    U, out = np.zeros((0, Xc.shape[1])), {}
    for r in P_RANKS:
        if U.shape[0] < r - 1:
            m = lr().fit(res(Xc, U), y)
            W = m[-1].coef_ / m[0].scale_
            U = gs(U, W - W.mean(0))
        out[r] = U.copy()
    return out


def var_share(Xc, U):
    return float(((Xc @ U.T) ** 2).sum() / (Xc ** 2).sum()) if len(U) else 0.0


def crossfit(Xl, y, g, k):
    """Held-out correctness per span type at one layer, outer fold k (cross-fitted readout)."""
    T = [s for s in np.unique(g) if s != k]
    te = g == k
    fits = {}
    for j in T:
        f = np.isin(g, [s for s in T if s != j])
        mu = Xl[f].mean(0)
        Xf = Xl[f] - mu
        sp = {"r0": np.zeros((0, Xl.shape[1])), "M2": span_M2(Xf, y[f]), "MS": span_MS(Xf, y[f], g[f])}
        sp.update({f"P{r}": U for r, U in spans_P(Xf, y[f]).items()})
        fits[j] = (mu, sp)
    out = {}
    for name in fits[T[0]][1]:
        Rtr = np.vstack([res(Xl[g == j] - fits[j][0], fits[j][1][name]) for j in T])
        ytr = np.concatenate([y[g == j] for j in T])
        clf = lr().fit(Rtr, ytr)
        proba = np.mean([clf.predict_proba(res(Xl[te] - fits[j][0], fits[j][1][name])) for j in T], axis=0)
        pred = clf.classes_[proba.argmax(1)]
        out[name] = dict(correct=(pred == y[te]).tolist(),
                         rank=float(np.mean([fits[j][1][name].shape[0] for j in T])))
    # degenerate reference: span and readout fitted on the SAME 5 training schemes
    tr = g != k
    mu = Xl[tr].mean(0)
    U = span_M2(Xl[tr] - mu, y[tr])
    clf = lr().fit(res(Xl[tr] - mu, U), y[tr])
    out["M2_degenerate"] = dict(correct=(clf.predict(res(Xl[te] - mu, U)) == y[te]).tolist(), rank=2.0,
                                coef_norm=float(np.linalg.norm(clf[-1].coef_)))
    return out


def behaviour_spans_and_controls(Xl, Ml, y, g, k):
    """Candidate behavioural spans fitted on all 5 training schemes of fold k (M2, MS) + rank-matched controls:
    top-r PCs of the training residual orthogonal to the span, and a training-only variance-matched contiguous PC window.
    Variance shares on train/test, C1 `last` (decision token) and C1 `mean` (message span)."""
    tr, te = g != k, g == k
    mu, mm = Xl[tr].mean(0), Ml[tr].mean(0)
    Xtr, Xte, Mtr, Mte = Xl[tr] - mu, Xl[te] - mu, Ml[tr] - mm, Ml[te] - mm
    out = {}
    for name, U in (("M2", span_M2(Xtr, y[tr])), ("MS", span_MS(Xtr, y[tr], g[tr]))):
        r = U.shape[0]
        _, S, Vt = np.linalg.svd(res(Xtr, U), full_matrices=False)
        pcs = S ** 2 / (Xtr ** 2).sum()
        target = var_share(Xtr, U)
        sums = np.convolve(pcs, np.ones(r), "valid")
        s = int(np.argmin(np.abs(sums - target)))
        V_top, V_m = Vt[:r], Vt[s:s + r]
        out[name] = dict(rank=r, span=dict(last_train=target, last_test=var_share(Xte, U), mean_train=var_share(Mtr, U),
                                           mean_test=var_share(Mte, U)),
                         top=dict(last_train=var_share(Xtr, V_top), last_test=var_share(Xte, V_top),
                                  mean_train=var_share(Mtr, V_top), mean_test=var_share(Mte, V_top)),
                         matched=dict(start=s, last_train=var_share(Xtr, V_m), last_test=var_share(Xte, V_m),
                                      mean_train=var_share(Mtr, V_m), mean_test=var_share(Mte, V_m)))
    return out


def onset_acc(Xl, y, g, k):
    """Training-only inner leave-one-scheme-out state accuracy (rank 0) at one layer, outer fold k."""
    c = []
    for j in [s for s in np.unique(g) if s != k]:
        itr, ite = (g != k) & (g != j), g == j
        c += (lr().fit(Xl[itr], y[itr]).predict(Xl[ite]) == y[ite]).tolist()
    return float(np.mean(c))


def run(model):
    d = MODELS[model]
    meta = json.load(open(d / "acts/meta.json"))
    y, g = np.array(meta["state"]), np.array(meta["scheme_id"])
    z = np.load(d / "acts/C1.npz")
    X, M = z["last"].astype(np.float32), z["mean"].astype(np.float32)
    L = X.shape[1]
    lay = [l for l in range(L) if l / (L - 1) >= 0.5]
    folds = [int(f) for f in np.unique(g)]
    print(model, "L", L, "analysed", lay[0], "-", lay[-1], "late", late_layers(L)[0], flush=True)
    out = dict(model=model, L=L, layers=lay, late_layers=late_layers(L), n=len(y), folds=folds,
               state=y.tolist(), scheme=g.tolist(), crossfit={}, behaviour={}, onset_inner={})
    jobs = [(l, k) for l in lay for k in folds]
    for (l, k), o in zip(jobs, Parallel(n_jobs=-1)(delayed(crossfit)(X[:, l], y, g, k) for l, k in jobs)):
        out["crossfit"].setdefault(str(l), {})[str(k)] = o
    print(model, "crossfit done", flush=True)
    jobs = [(l, k) for l in range(L) for k in folds]
    for (l, k), o in zip(jobs, Parallel(n_jobs=-1)(delayed(behaviour_spans_and_controls)(X[:, l], M[:, l], y, g, k)
                                                   for l, k in jobs)):
        out["behaviour"].setdefault(str(l), {})[str(k)] = o
    for (l, k), a in zip(jobs, Parallel(n_jobs=-1)(delayed(onset_acc)(X[:, l], y, g, k) for l, k in jobs)):
        out["onset_inner"].setdefault(str(k), {})[str(l)] = a
    out["onset_first_ge_0.8"] = {str(k): next((l for l in range(L) if out["onset_inner"][str(k)][str(l)] >= 0.8), None)
                                 for k in folds}
    print(model, "training-only onset per fold", out["onset_first_ge_0.8"], "| frozen cutoff", late_layers(L)[0], flush=True)
    (ROOT / "data/redundancy").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / f"data/redundancy/{model}.json", "w"))




# ---------------------------------------------------------------------------------------------------- reporting
def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def strat_boot(v, groups, nboot=10000, seed=0):
    """Mean and 95% CI of per-episode values, resampling episodes within scheme (unit = episode)."""
    rng = np.random.default_rng(seed)
    idx = [np.where(groups == s)[0] for s in np.unique(groups)]
    b = np.array([np.concatenate([v[rng.choice(ix, len(ix))] for ix in idx]).mean() for _ in range(nboot)])
    return float(v.mean()), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def summarise(model):
    r = json.load(open(ROOT / f"data/redundancy/{model}.json"))
    folds, late, L = r["folds"], r["late_layers"], r["L"]
    g = np.array(r["scheme"])
    names = list(r["crossfit"][str(r["layers"][0])][str(folds[0])].keys())
    order = np.concatenate([np.where(g == k)[0] for k in folds])      # pooled episode order = fold order
    gs_ = g[order]
    S = dict(model=model, L=L, late=[late[0], late[-1]], per_layer={}, late_mean={}, paired_vs_r0={}, per_fold_late={},
             rank_mean={}, degenerate_coef_norm_max=0.0)
    corr = {}
    for nm in names:
        mat = np.array([[c for k in folds for c in r["crossfit"][str(l)][str(k)][nm]["correct"]] for l in r["layers"]],
                       float)                                           # [layers, 306]
        corr[nm] = mat
        S["per_layer"][nm] = {str(l): float(mat[i].mean()) for i, l in enumerate(r["layers"])}
        li = [r["layers"].index(l) for l in late]
        v = mat[li].mean(0)                                             # per-episode accuracy averaged over late layers
        S["late_mean"][nm] = dict(zip(("mean", "ci"), strat_boot(v, gs_)))
        S["per_fold_late"][nm] = {str(k): float(v[gs_ == k].mean()) for k in folds}
        S["rank_mean"][nm] = float(np.mean([r["crossfit"][str(l)][str(k)][nm]["rank"] for l in late for k in folds]))
    li = [r["layers"].index(l) for l in late]
    base = corr["r0"][li].mean(0)
    for nm in names:
        if nm != "r0":
            S["paired_vs_r0"][nm] = dict(zip(("delta", "ci"), strat_boot(corr[nm][li].mean(0) - base, gs_)))
    S["degenerate_coef_norm_max"] = float(max(r["crossfit"][str(l)][str(k)]["M2_degenerate"]["coef_norm"]
                                              for l in r["layers"] for k in folds))
    # variance shares of behavioural candidate spans (fitted on 5 training schemes) and controls, mean over folds
    vs = {}
    for nm in ("M2", "MS"):
        for kind in ("span", "top", "matched"):
            for key in ("last_train", "last_test", "mean_train", "mean_test"):
                vs.setdefault(nm, {}).setdefault(kind, {})[key] = [
                    float(np.mean([r["behaviour"][str(l)][str(k)][nm][kind][key] for k in folds])) for l in range(L)]
        vs[nm]["rank"] = [float(np.mean([r["behaviour"][str(l)][str(k)][nm]["rank"] for k in folds])) for l in range(L)]
    S["variance"] = vs
    S["onset_first_ge_0.8"] = r["onset_first_ge_0.8"]
    S["pooled_onset_cutoff"] = late[0]
    return S


def report(models=("7b", "14b", "32b")):
    import matplotlib.pyplot as plt
    out = {m: summarise(m) for m in models if (ROOT / f"data/redundancy/{m}.json").exists()}
    json.dump(out, open(ROOT / "data/redundancy/summary.json", "w"), indent=1)
    T = ["# Residual state decodability after removal (cross-fitted; exp/redundancy_analysis.py)\n",
         "Held-out (outer LOSO) state accuracy, mean over the frozen late layers (rel. depth ≥ 2/3), pooled n = 306; "
         "95% CI = scheme-stratified bootstrap over episodes. Chance = 0.333. Removal spans fitted on 4 of 5 training "
         "schemes; readout trained on out-of-fit residuals (see notes/redundancy_log.md).\n",
         "| model | span | mean rank | late-mean acc [95% CI] | Δ vs no removal [95% CI] | per-fold (schemes 0–5) |",
         "|---|---|---|---|---|---|"]
    for m, S in out.items():
        for nm in S["late_mean"]:
            lm, pr = S["late_mean"][nm], S["paired_vs_r0"].get(nm)
            T.append(f"| {m} | {nm} | {S['rank_mean'][nm]:.1f} | {lm['mean']:.3f} [{lm['ci'][0]:.3f}, {lm['ci'][1]:.3f}] | "
                     + (f"{pr['delta']:+.3f} [{pr['ci'][0]:+.3f}, {pr['ci'][1]:+.3f}]" if pr else "–") + " | "
                     + " ".join(f"{S['per_fold_late'][nm][k]:.2f}" for k in sorted(S["per_fold_late"][nm])) + " |")
    T += ["\nVariance share of the candidate behavioural spans (fitted on all 5 training schemes; mean over folds) and "
          "rank-matched controls, mean over late layers / over all layers:\n",
          "| model | span | pooling | span train | span test | top-r PC ctrl train | matched-window ctrl train | matched test |",
          "|---|---|---|---|---|---|---|---|"]
    for m, S in out.items():
        late = list(range(S["late"][0], S["L"]))
        for nm in ("M2", "MS"):
            v = S["variance"][nm]
            for pool in ("last", "mean"):
                f = lambda kind, key: (np.mean([v[kind][f"{pool}_{key}"][l] for l in late]),
                                       np.mean(v[kind][f"{pool}_{key}"]))
                cells = [f("span", "train"), f("span", "test"), f("top", "train"), f("matched", "train"), f("matched", "test")]
                T.append(f"| {m} | {nm} (rank {np.mean(v['rank']):.1f}) | {pool} | " +
                         " | ".join(f"{a:.3f} / {b:.3f}" for a, b in cells) + " |")
    T += ["\nTraining-only onset (first layer with inner-LOSO state acc ≥ 0.8) per outer fold vs the frozen cutoff:\n"]
    for m, S in out.items():
        T.append(f"- {m}: {S['onset_first_ge_0.8']} (cutoff L{S['pooled_onset_cutoff']}); "
                 f"degenerate reference max |coef| = {S['degenerate_coef_norm_max']:.2e}")
    (ROOT / "data/redundancy/tables.md").write_text("\n".join(T) + "\n")
    # figure
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    col = {"7b": "#0072B2", "14b": "#E69F00", "32b": "#CC79A7"}
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    xs = ["r0", "M2", "MS"] + [f"P{k}" for k in P_RANKS]
    for i, (m, S) in enumerate(out.items()):
        mu = [S["late_mean"][n]["mean"] for n in xs]
        lo = [S["late_mean"][n]["mean"] - S["late_mean"][n]["ci"][0] for n in xs]
        hi = [S["late_mean"][n]["ci"][1] - S["late_mean"][n]["mean"] for n in xs]
        ax[0].errorbar(np.arange(len(xs)) + (i - 1) * 0.15, mu, yerr=[lo, hi], fmt="o", ms=3, color=col[m],
                       label=m.upper(), capsize=1.5, elinewidth=0.6)
        for nm, ls in (("r0", "-"), ("M2", "--"), ("MS", ":")):
            pl = S["per_layer"][nm]
            ls_ = sorted(int(k) for k in pl)
            ax[1].plot([l / (S["L"] - 1) for l in ls_], [pl[str(l)] for l in ls_], ls, color=col[m], lw=1,
                       label={"r0": "no removal", "M2": "M2 removed", "MS": "MS removed"}[nm] if i == 0 else None)
        vM = S["variance"]["M2"]
        x = np.arange(S["L"]) / (S["L"] - 1)
        ax[2].plot(x, vM["span"]["last_train"], "-", color=col[m], lw=1, label=f"{m.upper()} M2 span")
        ax[2].plot(x, vM["top"]["last_train"], ":", color=col[m], lw=1)
    ax[0].set_xticks(range(len(xs)))
    ax[0].set_xticklabels(["none", "M2", "MS"] + [f"P{k}" for k in P_RANKS], rotation=90, fontsize=6)
    ax[0].axhline(1 / 3, color="k", lw=0.5, ls=":")
    ax[0].set(ylim=(0, 1.02), ylabel="held-out state acc. (late layers)", title="(a) after removal (cross-fit)")
    ax[0].legend(fontsize=6, frameon=False)
    ax[1].axhline(1 / 3, color="k", lw=0.5, ls=":")
    ax[1].set(ylim=(0, 1.02), xlabel="relative depth", title="(b) per layer: none / M2 / MS")
    ax[1].legend(fontsize=5, frameon=False, loc="lower right")
    ax[2].set(xlabel="relative depth", ylabel="train variance share (decision tok.)", title="(c) var.: M2 span — / top-2 PCs ···")
    ax[2].legend(fontsize=5.5, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"figures/fig11_residual_decodability.{ext}", dpi=300)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "run":
        for m in sys.argv[2:]:
            run(m)
    else:
        for m, S in report().items():
            print(m, {k: round(v["mean"], 3) for k, v in S["late_mean"].items()})
