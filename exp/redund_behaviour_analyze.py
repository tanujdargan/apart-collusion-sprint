"""Analysis of the core behavioural matrix (notes/redundancy_plan.md §5): none, M2_all, M2_late, PC2_all, PC2_late on
Qwen2.5-7B and -32B (held-out fold directions, frozen 7B messages), plus activation captures.

  .venv-analysis/bin/python exp/redund_behaviour_analyze.py  -> data/redund/analysis.json, data/redund/tables.md,
                                                               figures/fig12_placement.{png,pdf}
Placement contrasts (M2_late vs M2_all, PC2_late vs PC2_all) compare BUNDLED recipes: the late recipe differs from the
all-layer recipe both in the decoder layers hooked AND in the absence of the embedding hook.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
BASE = {"7b": ROOT / "data/loso", "32b": ROOT / "data_models/qwen32b/loso"}
OLD = {"7b": (ROOT / "data/loso/runs", "loso_f{k}_{c}"), "32b": (ROOT / "data_models/qwen32b/loso/runs", "loso_f{k}_{c}")}
OLDMAP = {"none": "none", "M2_all": "state_fold", "PC2_all": "pc2d_fold"}
CFGS = ["none", "M2_all", "M2_late", "PC2_all", "PC2_late"]
BETS = {"COLD": 0, "NEUTRAL": 25, "HOT": 100}
FOLDS = [str(k) for k in range(6)]
NBOOT = 10000


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def boot(v, g, seed=0):
    rng = np.random.default_rng(seed)
    idx = [np.where(g == s)[0] for s in np.unique(g)]
    b = np.array([np.concatenate([v[rng.choice(ix, len(ix))] for ix in idx]).mean() for _ in range(NBOOT)])
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def paired(a, b, g):
    a, b = a.astype(float), b.astype(float)
    n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    return dict(delta=float((a - b).mean()), ci=boot(a - b, g),
                mcnemar_p=float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0, a_only=n10, b_only=n01)


def behaviour(m):
    splits = json.load(open(BASE[m] / "splits.json"))
    R = {}
    for k in FOLDS:
        for c in CFGS:
            r = json.load(open(ROOT / f"data/redund/{m}/runs/redund_f{k}_{c}/metrics.json"))
            assert r["ids"] == splits[k]["eval_ids"], (m, k, c)
            for cond in ("C1", "C6"):
                assert [x["id"] for x in r[cond + "_rows"]] == r["ids"]
            assert [x["id"] for x in r["recall"]] == r["ids"]
            R[(k, c)] = r
    V = {}
    for c in CFGS:
        v = {key: [] for key in ("c1", "c1bet", "c1betnone", "c1decnone", "c6", "c6betnone", "recall", "g")}
        for k in FOLDS:
            r = R[(k, c)]
            for x in r["C1_rows"]:
                v["c1"].append(x["decoded"] == x["state"]); v["c1bet"].append(x["bet"] == BETS[x["state"]])
                v["c1betnone"].append(x["bet"] is None); v["c1decnone"].append(x["decoded"] is None); v["g"].append(int(k))
            for x in r["C6_rows"]:
                v["c6"].append(x["bet"] == BETS[x["state"]]); v["c6betnone"].append(x["bet"] is None)
            v["recall"] += [x["answer"] == x["topic"] for x in r["recall"]]
        V[c] = {kk: np.array(vv) for kk, vv in v.items()}
    g = V["none"]["g"]
    out = dict(per_config={}, paired={}, per_fold={}, determinism={})
    for c in CFGS:
        v, n = V[c], len(V[c]["c1"])
        out["per_config"][c] = dict(
            n=n, c1_decode=float(v["c1"].mean()), c1_ci=wilson(v["c1"].sum(), n), c1_bet_exact=float(v["c1bet"].mean()),
            c1_bet_none=int(v["c1betnone"].sum()), c1_decoded_none=int(v["c1decnone"].sum()),
            c6_bet_exact=float(v["c6"].mean()), c6_ci=wilson(v["c6"].sum(), n), c6_bet_none=int(v["c6betnone"].sum()),
            recall=float(v["recall"].mean()), arc_mean=float(np.mean([R[(k, c)]["arc_acc"] for k in FOLDS])),
            arc_by_fold=[R[(k, c)]["arc_acc"] for k in FOLDS],
            decoded_dist={s: int(sum(R[(k, c)]["C1"]["decoded_dist"][s] for k in FOLDS)) for s in ("COLD", "NEUTRAL", "HOT", "None")})
        out["per_fold"][c] = {k: dict(c1=float(np.mean([x["decoded"] == x["state"] for x in R[(k, c)]["C1_rows"]])),
                                      c6=float(np.mean([x["bet"] == BETS[x["state"]] for x in R[(k, c)]["C6_rows"]])),
                                      arc=R[(k, c)]["arc_acc"]) for k in FOLDS}
    contrasts = [(c, "none") for c in CFGS[1:]] + [("M2_late", "M2_all"), ("M2_late", "PC2_late"), ("M2_all", "PC2_all"),
                                                   ("PC2_late", "PC2_all")]
    for a, b in contrasts:
        out["paired"][f"{a} - {b}"] = {key: paired(V[a][key], V[b][key], g) for key in ("c1", "c1bet", "c6", "recall")}
    old_dir, pat = OLD[m]
    for c, oc in OLDMAP.items():
        same, n = 0, 0
        for k in FOLDS:
            f = old_dir / pat.format(k=k, c=oc) / "metrics.json"
            o = {x["id"]: x["output"] for x in json.load(open(f))["C1_rows"]}
            for x in R[(k, c)]["C1_rows"]:
                same += x["output"] == o[x["id"]]; n += 1
        out["determinism"][c] = dict(vs=f"{oc} (earlier LOSO run)", c1_identical_output=same / n)
    return out


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))


def captures(m):
    meta = json.load(open((ROOT / "data/acts/meta.json") if m == "7b" else (ROOT / "data_models/qwen32b/acts/meta.json")))
    y, g = np.array(meta["state"]), np.array(meta["scheme_id"])
    cd = ROOT / f"data/redund/{m}/caps"
    N = np.load(cd / "cap_redund_none.npy").astype(np.float32)          # same-pipeline unablated capture (all 306)
    L = N.shape[1]
    late0 = math.ceil(2 * (L - 1) / 3)
    out = dict(L=L, late_start=late0, proj_ratio={}, readout={}, baseline_vs_saved_maxrel=None)
    saved = np.load((ROOT / "data/acts/C1.npz") if m == "7b" else (ROOT / "data_models/qwen32b/acts/C1.npz"))["last"].astype(np.float32)
    out["baseline_vs_saved_maxrel"] = float(np.max(np.linalg.norm(N - saved, axis=-1) / np.linalg.norm(saved, axis=-1)))
    del saved
    corr = {p: np.zeros((L, len(y)), bool) for p in ("base", "all_retrained", "late_retrained", "all_frozen", "late_frozen")}
    pr = {p: np.zeros((6, L)) for p in ("base", "all", "late")}
    for k in range(6):
        Q = np.load(BASE[m] / f"fold{k}/state.npy").astype(np.float64)
        Qo = np.stack([np.linalg.qr(q.T)[0].T for q in Q])
        A = {"all": np.load(cd / f"cap_redund_f{k}_M2_all.npy").astype(np.float32),
             "late": np.load(cd / f"cap_redund_f{k}_M2_late.npy").astype(np.float32)}
        tr, te = g != k, g == k
        for l in range(L):
            for p, X in (("base", N), ("all", A["all"]), ("late", A["late"])):
                pr[p][k, l] = float((np.linalg.norm(X[:, l] @ Qo[l].T, axis=1) / np.linalg.norm(X[:, l], axis=1)).mean())
            mb = lr().fit(N[tr, l], y[tr])
            corr["base"][l, te] = mb.predict(N[te, l]) == y[te]
            for p in ("all", "late"):
                corr[f"{p}_retrained"][l, te] = lr().fit(A[p][tr, l], y[tr]).predict(A[p][te, l]) == y[te]
                corr[f"{p}_frozen"][l, te] = mb.predict(A[p][te, l]) == y[te]
        print(m, "fold", k, "done", flush=True)
    out["proj_ratio"] = {p: v.mean(0).tolist() for p, v in pr.items()}
    late = list(range(late0, L))
    for p, c in corr.items():
        v = c[late].mean(0).astype(float)
        out["readout"][p] = dict(per_layer=c.mean(1).tolist(), late_mean=float(v.mean()), late_ci=boot(v, g),
                                 per_fold_late={str(k): float(v[g == k].mean()) for k in range(6)})
    for p in ("all", "late"):
        for kind in ("retrained", "frozen"):
            d = (corr[f"{p}_{kind}"][late].mean(0) - corr["base"][late].mean(0)).astype(float)
            out["readout"][f"{p}_{kind}"]["delta_vs_base"] = dict(delta=float(d.mean()), ci=boot(d, g))
    return out


def main():
    res = {}
    for m in ("7b", "32b"):
        res[m] = dict(behaviour=behaviour(m), captures=captures(m))
    json.dump(res, open(ROOT / "data/redund/analysis.json", "w"), indent=1)
    T = ["# Core behavioural matrix: placement × state/control (held-out fold directions; frozen 7B messages)\n",
         "Pooled n = 306 per model (6 folds × 51). Wilson 95% CI for absolute accuracy; paired Δ with scheme-stratified "
         "bootstrap (10k, unit = episode) and exact McNemar (binary outcomes). `late` = decoder layers ≥ ceil(2(L−1)/3) "
         "and NO embedding hook; `all` = all decoder layers + embedding hook: placement contrasts are bundled recipes.\n"]
    for m, r in res.items():
        b = r["behaviour"]
        T += [f"\n## {m}\n", "| config | C1 decode [CI] | C1 bet-exact | C1 bet None / decoded None | C1 decoded C/N/H | "
              "C6 bet-exact [CI] | C6 bet None | recall | ARC mean (folds) |", "|---|---|---|---|---|---|---|---|---|"]
        for c, p in b["per_config"].items():
            dd = p["decoded_dist"]
            T.append(f"| {c} | {p['c1_decode']:.3f} [{p['c1_ci'][0]:.3f}, {p['c1_ci'][1]:.3f}] | {p['c1_bet_exact']:.3f} | "
                     f"{p['c1_bet_none']} / {p['c1_decoded_none']} | {dd['COLD']}/{dd['NEUTRAL']}/{dd['HOT']} | "
                     f"{p['c6_bet_exact']:.3f} [{p['c6_ci'][0]:.3f}, {p['c6_ci'][1]:.3f}] | {p['c6_bet_none']} | "
                     f"{p['recall']:.3f} | {p['arc_mean']:.3f} ({min(p['arc_by_fold']):.2f}–{max(p['arc_by_fold']):.2f}) |")
        T += ["\n| contrast | Δ C1 decode [CI] (McNemar p) | Δ C1 bet-exact [CI] | Δ C6 bet-exact [CI] | Δ recall [CI] |",
              "|---|---|---|---|---|"]
        for name, p in b["paired"].items():
            f = lambda x: f"{x['delta']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"
            T.append(f"| {name} | {f(p['c1'])} ({p['c1']['mcnemar_p']:.2g}) | {f(p['c1bet'])} | {f(p['c6'])} | {f(p['recall'])} |")
        T += ["\nPer-fold C1 decode (schemes 0–5):\n"]
        for c in CFGS:
            T.append(f"- {c}: " + " ".join(f"{b['per_fold'][c][k]['c1']:.2f}" for k in FOLDS) +
                     " | C6: " + " ".join(f"{b['per_fold'][c][k]['c6']:.2f}" for k in FOLDS) +
                     " | ARC: " + " ".join(f"{b['per_fold'][c][k]['arc']:.2f}" for k in FOLDS))
        T += ["\nDeterminism vs earlier LOSO runs (same fold directions, episodes, order): " +
              ", ".join(f"{c} {v['c1_identical_output']:.3f}" for c, v in b["determinism"].items())]
        cp = r["captures"]
        T += [f"\nCaptures ({m}, L={cp['L']}, late from L{cp['late_start']}): same-pipeline unablated capture vs saved acts "
              f"max rel diff {cp['baseline_vs_saved_maxrel']:.3f}.",
              "State-probe accuracy on ablated-pass decision-token activations (late-layer mean, held-out folds, n = 306):\n",
              "| readout | late-mean acc [CI] | Δ vs unablated (same pipeline) [CI] | per fold |", "|---|---|---|---|"]
        for p, v in cp["readout"].items():
            dv = v.get("delta_vs_base")
            T.append(f"| {p} | {v['late_mean']:.3f} [{v['late_ci'][0]:.3f}, {v['late_ci'][1]:.3f}] | " +
                     (f"{dv['delta']:+.3f} [{dv['ci'][0]:+.3f}, {dv['ci'][1]:+.3f}]" if dv else "–") + " | " +
                     " ".join(f"{v['per_fold_late'][k]:.2f}" for k in sorted(v["per_fold_late"])) + " |")
        prr = cp["proj_ratio"]
        T.append(f"\nProjection ratio ‖hQ‖/‖h‖ onto the fold M2 span (mean over folds/episodes): baseline, early layers "
                 f"{np.mean(prr['base'][:cp['late_start']]):.3f}, late {np.mean(prr['base'][cp['late_start']:]):.3f}; "
                 f"M2_all early {np.mean(prr['all'][:cp['late_start']]):.4f}, late {np.mean(prr['all'][cp['late_start']:]):.4f}; "
                 f"M2_late early {np.mean(prr['late'][:cp['late_start']]):.3f}, late {np.mean(prr['late'][cp['late_start']:]):.4f}.")
    (ROOT / "data/redund/tables.md").write_text("\n".join(T) + "\n")
    figure(res)
    return res


def figure(res):
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    col = {"7b": "#0072B2", "32b": "#CC79A7"}
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
    x = np.arange(len(CFGS))
    for i, (m, r) in enumerate(res.items()):
        pc = r["behaviour"]["per_config"]
        for j, (key, ci, mk) in enumerate((("c1_decode", "c1_ci", "o"), ("c6_bet_exact", "c6_ci", "s"))):
            v = np.array([pc[c][key] for c in CFGS])
            e = np.array([pc[c][ci] for c in CFGS])
            ax[0].errorbar(x + (i - 0.5) * 0.3 + j * 0.08, v, yerr=[v - e[:, 0], e[:, 1] - v], fmt=mk, ms=3.5, color=col[m],
                           mfc=col[m] if j == 0 else "white", capsize=1.5, elinewidth=0.6,
                           label=f"{m.upper()} {'C1 covert' if j == 0 else 'C6 explicit'}")
        cp = r["captures"]
        rel = np.arange(cp["L"]) / (cp["L"] - 1)
        ax[1].plot(rel, cp["proj_ratio"]["base"], color=col[m], lw=0.8, ls=":")
        ax[1].plot(rel, cp["proj_ratio"]["late"], color=col[m], lw=1.2, label=f"{m.upper()} M2_late")
        ax[1].plot(rel, cp["proj_ratio"]["all"], color=col[m], lw=1.2, ls="--", label=f"{m.upper()} M2_all")
        for p, ls in (("base", ":"), ("all_retrained", "--"), ("late_retrained", "-")):
            ax[2].plot(rel, cp["readout"][p]["per_layer"], ls, color=col[m], lw=1,
                       label=f"{m.upper()} {p.replace('_retrained', ' (retrained)')}")
    ax[0].axhspan(0.30, 0.37, color="#eeeeee", zorder=0)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(CFGS, rotation=30, fontsize=6.5)
    ax[0].set(ylim=(0, 1.05), ylabel="accuracy (held-out, n=306)", title="(a) behaviour")
    ax[0].legend(fontsize=5, frameon=False, loc="lower left")
    ax[1].set(xlabel="relative depth", ylabel="‖hQ‖/‖h‖ (fold M2 span)", title="(b) placement check")
    ax[1].legend(fontsize=5, frameon=False)
    ax[2].axhline(1 / 3, color="k", lw=0.5, ls=":")
    ax[2].set(xlabel="relative depth", ylabel="state-probe acc.", ylim=(0, 1.02), title="(c) ablated-pass readout")
    ax[2].legend(fontsize=4.5, frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"figures/fig12_placement.{ext}", dpi=300)


if __name__ == "__main__":
    main()
