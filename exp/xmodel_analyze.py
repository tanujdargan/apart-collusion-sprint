"""Cross-model (Qwen2.5 7B / 14B / 32B) comparison — exploratory, CPU only (notes/followups_plan.md section C).

  .venv-analysis/bin/python exp/xmodel_analyze.py probes32   # PART 1: 32B probe geometry -> data_models/qwen32b/probes32.json
  .venv-analysis/bin/python exp/xmodel_analyze.py geometry   # PART 1: 3-model geometry  -> data/xmodel_summary.json["geometry"]
  .venv-analysis/bin/python exp/xmodel_analyze.py compare    # PART 2: matched LOSO comparison (+ tables, figure)

Probe recipe is imported unchanged from exp/probes.py (StandardScaler + LR C=0.1; leave-one-scheme-out AUROC per
layer; 3-way state accuracy per layer). Only C1/C4 activations exist for 32B, so C1vC2/C3/C5 contrasts are not
computable there.
"""
import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
from probes import loso_auroc, loso_state_acc  # noqa: E402  (recipe functions only; paths are our own)
from loso_fit import subspace_agreement  # noqa: E402

M32 = ROOT / "data_models/qwen32b"
OUT32 = M32 / "probes32.json"
SUMMARY = ROOT / "data/xmodel_summary.json"

# per model: acts dir, probes.json, global direction files (7B originals are the global fits), fold dir
MODELS = {
    "7B": dict(acts=ROOT / "data/acts", probes=ROOT / "data/probes.json", L=28,
               glob={k: ROOT / f"data/directions/{f}.npy" for k, f in
                     dict(dm="detect_C1vC4_mean_dm", state="state_C1_last_2d", pc2d="pc_ctrl_2d", pc1d="pc_ctrl_1d",
                          lr="detect_C1vC4_mean_lr").items()},
               folds=ROOT / "data/loso", analysis=ROOT / "data/loso/analysis.json"),
    "14B": dict(acts=ROOT / "data_models/qwen14b/acts", probes=ROOT / "data_models/qwen14b/probes.json", L=48,
                glob={k: ROOT / f"data_models/qwen14b/loso/global/{k}.npy" for k in ("dm", "state", "pc2d", "pc1d", "lr")},
                folds=ROOT / "data_models/qwen14b/loso", analysis=ROOT / "data_models/qwen14b/loso/analysis.json"),
    "32B": dict(acts=M32 / "acts", probes=OUT32, L=64,
                glob={k: M32 / f"loso/global/{k}.npy" for k in ("dm", "state", "pc2d", "pc1d", "lr")},
                folds=M32 / "loso", analysis=M32 / "loso/analysis.json"),
}
DIRS = ("dm", "pc1d", "state", "pc2d", "lr")
NJ = 8


def load_acts(d, cond, pool):
    return np.load(d / f"{cond}.npz")[pool].astype(np.float32)


def maxabs_cos(dm, Q):
    """max_k |<dm_l, q_lk>| per layer; dm [L,D] unit, Q [L,2,D] orthonormal rows."""
    return np.abs(np.einsum("ld,lkd->lk", dm, Q)).max(-1)


def probes32():
    meta = json.load(open(M32 / "acts/meta.json"))
    groups, state = np.array(meta["scheme_id"]), np.array(meta["state"])
    L = meta["n_layers"]
    out = dict(model=meta["model"], n=len(groups), n_layers=L, auroc={}, state_acc={}, cos={},
               recipe="exp/probes.py: StandardScaler + LogisticRegression(C=0.1), leave-one-scheme-out (6 folds)",
               unavailable=["detect_C1vC2_mean", "detect_C1vC3_full", "detect_C1vC5_mean", "detect_C1vC5_last",
                            "detect_C1vC5_full"],
               unavailable_reason="only C1 and C4 activations were extracted for 32B")
    for name, pool in (("detect_C1vC4_mean", "mean"), ("detect_C1vC4_full", "full_last")):
        Xp, Xn = load_acts(M32 / "acts", "C1", pool), load_acts(M32 / "acts", "C4", pool)
        out["auroc"][name] = [float(a) for a in Parallel(n_jobs=NJ)(
            delayed(loso_auroc)(Xp[:, l], Xn[:, l], groups) for l in range(L))]
        print(name, "min", round(min(out["auroc"][name]), 4), "L0", round(out["auroc"][name][0], 4), flush=True)
        del Xp, Xn
    X = load_acts(M32 / "acts", "C1", "last")
    out["state_acc"]["C1_last"] = [float(a) for a in Parallel(n_jobs=NJ)(
        delayed(loso_state_acc)(X[:, l], state, groups) for l in range(L))]
    print("state acc", [round(a, 2) for a in out["state_acc"]["C1_last"]], flush=True)
    dm, Q = np.load(M32 / "loso/global/dm.npy"), np.load(M32 / "loso/global/state.npy")
    out["cos"]["detect_vs_state_maxabs"] = maxabs_cos(dm, Q).tolist()
    out["cos"]["source"] = "data_models/qwen32b/loso/global/{dm,state}.npy (fit on all 306)"
    out["cos"]["primary_dm_vs_lr"] = (dm * np.load(M32 / "loso/global/lr.npy")).sum(-1).tolist()
    json.dump(out, open(OUT32, "w"), indent=1)
    return out


def per_layer_var_share(acts, d, k):
    """Share of variance captured per layer, same definition as exp/loso_fit.var_share (all 306 episodes)."""
    if k in ("dm", "pc1d", "lr"):
        P, N = load_acts(acts, "C1", "mean"), load_acts(acts, "C4", "mean")
    else:
        P, N = load_acts(acts, "C1", "last"), None
    r = []
    for l in range(P.shape[1]):
        Z = P[:, l] if N is None else np.r_[P[:, l], N[:, l]]
        Z = Z - Z.mean(0)
        v = d[l][None] if d.ndim == 2 else d[l]
        q = np.linalg.qr(v.T)[0].T
        r.append(float(((Z @ q.T) ** 2).sum() / (Z ** 2).sum()))
    return r


def onset(acc, thr):
    for l, a in enumerate(acc):
        if a >= thr:
            return l
    return None


def geometry():
    g = {}
    for m, c in MODELS.items():
        p = json.load(open(c["probes"]))
        L = p["n_layers"]
        G = {k: np.load(f).astype(np.float32) for k, f in c["glob"].items()}
        e = dict(n_layers=L)
        e["auroc_C1vC4_mean"] = p["auroc"]["detect_C1vC4_mean"]
        e["auroc_C1vC4_full"] = p["auroc"]["detect_C1vC4_full"]
        e["state_acc"] = p["state_acc"]["C1_last"]
        cos = maxabs_cos(G["dm"], G["state"])
        e["cos_dm_state_maxabs"] = cos.tolist()
        e["cos_dm_state_from_probes_json"] = p["cos"]["detect_vs_state_maxabs"]
        e["cos_dm_lr"] = (G["dm"] * G["lr"]).sum(-1).tolist()
        ac = np.array(e["state_acc"])
        e["summary"] = dict(
            auroc_mean_L0=e["auroc_C1vC4_mean"][0], auroc_mean_min=min(e["auroc_C1vC4_mean"]),
            auroc_full_L0=e["auroc_C1vC4_full"][0], auroc_full_min=min(e["auroc_C1vC4_full"]),
            state_onset05=onset(ac, 0.5), state_onset08=onset(ac, 0.8),
            state_onset05_rel=onset(ac, 0.5) / (L - 1), state_onset08_rel=onset(ac, 0.8) / (L - 1),
            state_max=float(ac.max()), state_argmax=int(ac.argmax()),
            state_max_before_onset05=float(ac[:onset(ac, 0.5)].max()),
            cos_maxabs_max=float(cos.max()), cos_maxabs_median=float(np.median(cos)), cos_argmax=int(cos.argmax()),
            cos_dm_lr_range=[float(np.min(e["cos_dm_lr"])), float(np.max(e["cos_dm_lr"]))])
        vs = {k: per_layer_var_share(c["acts"], G[k], k) for k in DIRS}
        e["var_share_per_layer"] = vs
        st, pc = np.array(vs["state"]), np.array(vs["pc2d"])
        e["summary"]["var_share_mean"] = {k: float(np.mean(v)) for k, v in vs.items()}
        e["summary"]["pc2d_gt_state_layers"] = int((pc > st).sum())
        e["summary"]["pc2d_gt_state_layers_after_onset05"] = int((pc[onset(ac, 0.5):] > st[onset(ac, 0.5):]).sum())
        e["summary"]["n_layers_after_onset05"] = int(L - onset(ac, 0.5))
        agr = {}
        for f in range(6):
            fd = c["folds"] / f"fold{f}"
            agr[str(f)] = {k: subspace_agreement(np.load(fd / f"{k}.npy").astype(np.float32), G[k]) for k in DIRS}
        e["fold_vs_global_agreement"] = agr  # (mean over layers, min over layers)
        e["summary"]["fold_vs_global_agreement_range_of_means"] = {
            k: [min(agr[f][k][0] for f in agr), max(agr[f][k][0] for f in agr)] for k in DIRS}
        e["summary"]["fold_vs_global_agreement_min_any_layer"] = {k: min(agr[f][k][1] for f in agr) for k in DIRS}
        g[m] = e
        print(m, json.dumps(e["summary"], indent=None), flush=True)
    s = json.load(open(SUMMARY)) if SUMMARY.exists() else {}
    s["geometry"] = g
    json.dump(s, open(SUMMARY, "w"), indent=1)
    # mirror the 32B-only geometry into probes32.json
    p32 = json.load(open(OUT32))
    p32["fold_vs_global_agreement"] = g["32B"]["fold_vs_global_agreement"]
    p32["var_share_per_layer"] = g["32B"]["var_share_per_layer"]
    p32["summary"] = g["32B"]["summary"]
    json.dump(p32, open(OUT32, "w"), indent=1)
    return g


# ----------------------------------------------------------------------------------------------- PART 2: LOSO compare
BETS = {"COLD": 0, "NEUTRAL": 25, "HOT": 100}
CTRL = dict(dm="pc1d", state="pc2d")
COL = {"7B": "#0072B2", "14B": "#E69F00", "32B": "#CC79A7"}   # Okabe-Ito
MK = dict(dm="o", pc1d="s", state="^", pc2d="v", lr="D")
FIG = ROOT / "figures/fig9_xmodel"
TABLES = ROOT / "data/xmodel_tables.md"


def run_vectors(c, cfg):
    """Pooled per-episode outcomes over folds 0..5 (same order as exp/loso_analyze.vectors)."""
    v = {k: [] for k in ("c1_dec", "c1_bet", "c6_bet", "scheme")}
    arc, rec = [], []
    for k in range(6):
        m = json.load(open(c["folds"] / f"runs/loso_f{k}_{cfg}/metrics.json"))
        for r in m["C1_rows"]:
            v["c1_dec"].append(r["decoded"] == r["state"])
            v["c1_bet"].append(r["bet"] == BETS[r["state"]])
            v["scheme"].append(k)
        for r in m["C6_rows"]:
            v["c6_bet"].append(r["bet"] == BETS[r["state"]])
        arc.append(m["arc_acc"])
        rec.append(m["recall_acc"])
    out = {k: np.array(x) for k, x in v.items()}
    out["arc_by_fold"], out["recall_by_fold"] = arc, rec
    return out


def boot(a, b, scheme, rng, nb=10000):
    """Δ = mean(a − b), scheme-stratified paired bootstrap 95% CI (same scheme as exp/loso_analyze.paired)."""
    d = a.astype(float) - b.astype(float)
    idx = [np.where(scheme == s)[0] for s in np.unique(scheme)]
    bs = np.array([np.concatenate([d[rng.choice(ix, len(ix))] for ix in idx]).mean() for _ in range(nb)])
    return dict(delta=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])


def fmt_d(x):
    return f"{x['delta']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"


def compare():
    rng = np.random.default_rng(0)
    S = json.load(open(SUMMARY))
    res = {}
    for m, c in MODELS.items():
        A = json.load(open(c["analysis"]))
        V = {cfg: run_vectors(c, cfg) for cfg in A["per_config"]}
        sch = V["none"]["scheme"]
        e = dict(per_config=A["per_config"], decisions=A["decisions"], paired_from_analysis=A["paired"],
                 per_fold=A["per_fold"], extra={}, per_fold_extra={})
        # sanity: recomputed point estimates must equal exp/loso_analyze's
        for cfg in A["paired"]:
            if cfg in V:
                for key in ("c1_dec", "c6_bet"):
                    assert abs((V[cfg][key].mean() - V["none"][key].mean()) - A["paired"][cfg][key]["delta"]) < 1e-12
        # extra contrasts not in analysis.json: bet-exact for fold−global and direction − control
        for d in DIRS:
            e["extra"][f"{d}_fold_minus_global"] = {k: boot(V[f"{d}_fold"][k], V[f"{d}_global"][k], sch, rng)
                                                   for k in ("c1_dec", "c1_bet", "c6_bet")}
        for d, ct in CTRL.items():
            for w in ("fold", "global"):
                e["extra"][f"{d}_{w}_minus_{ct}_{w}"] = {k: boot(V[f"{d}_{w}"][k], V[f"{ct}_{w}"][k], sch, rng)
                                                        for k in ("c1_dec", "c1_bet", "c6_bet")}
        # per-fold spread (fold-fit and global), drops vs matched none within fold
        for d in DIRS:
            rows = []
            for k in range(6):
                s = sch == k
                r = dict(fold=k)
                for w in ("global", "fold"):
                    for key in ("c1_dec", "c1_bet", "c6_bet"):
                        r[f"{key}_drop_{w}"] = float(V["none"][key][s].mean() - V[f"{d}_{w}"][key][s].mean())
                    r[f"arc_{w}"] = V[f"{d}_{w}"]["arc_by_fold"][k]
                rows.append(r)
            e["per_fold_extra"][d] = rows
            e["per_fold_extra"][d + "_range"] = {key: [min(r[key] for r in rows), max(r[key] for r in rows)]
                                                 for key in rows[0] if key != "fold"}
        e["arc_none_by_fold"] = V["none"]["arc_by_fold"]
        res[m] = e
    S["loso"] = res
    S["confounds"] = [
        "14B used its own sender messages; 7B and 32B share the identical 7B frozen messages.",
        "PC-2D control is not variance-matched layer by layer in any model; in post-onset layers the state subspace "
        "carries more variance than PC-2D in all three, with the smallest margin at 32B (see geometry.var_share).",
        "GPUs differ (7B/14B L40S, 32B H200); batch size 48 unchanged.",
        "global configs are in-sample (fit on all 306 incl. evaluated episodes); fold configs are held-out.",
        "Exploratory 3-model comparison, one sender message set per model; not a scaling law.",
    ]
    json.dump(S, open(SUMMARY, "w"), indent=1)
    write_xtables(res, S["geometry"])
    xfigure(res, S["geometry"])
    return res


def write_xtables(res, G):
    L = ["# Cross-model tables (exp/xmodel_analyze.py compare) — Qwen2.5 7B / 14B / 32B, exploratory\n",
         "Sources: data/loso/analysis.json, data_models/qwen14b/loso/analysis.json, data_models/qwen32b/loso/analysis.json"
         " (exp/loso_analyze.py, unchanged) + per-episode runs for the extra bet-exact contrasts. Pooled n = 306 per "
         "model (6 held-out folds × 51). `global` = fit on all 306 (in-sample); `fold` = fit without the evaluated "
         "scheme (held-out). 7B global = original data/directions. 14B uses its own sender messages; 7B and 32B share "
         "the 7B messages. Paired Δ vs matched `none`, scheme-stratified paired bootstrap 95% CI (10k).\n",
         "## 1. Absolute results\n",
         "| model | config | C1 decode [Wilson CI] | C1 bet-exact | C6 bet-exact | recall | ARC (mean folds) | C1 bet None | C1 decoded None | C6 bet None |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m, e in res.items():
        for cfg, p in e["per_config"].items():
            L.append(f"| {m} | {cfg} | {p['c1_decode']:.3f} [{p['c1_decode_ci'][0]:.3f}–{p['c1_decode_ci'][1]:.3f}] | "
                     f"{p['c1_bet_exact']:.3f} | {p['c6_bet_exact']:.3f} | {p['recall']:.3f} | {p['arc_mean']:.3f} | "
                     f"{p['c1_bet_none']} | {p['c1_decoded_none']} | {p['c6_bet_none']} |")
    L += ["\n## 2. Paired Δ vs matched none\n",
          "| model | config | Δ C1 decode [CI] | Δ C1 bet-exact [CI] | Δ C6 bet-exact [CI] |", "|---|---|---|---|---|"]
    for m, e in res.items():
        for d in DIRS:
            for w in ("fold", "global"):
                p = e["paired_from_analysis"][f"{d}_{w}"]
                L.append(f"| {m} | {d}_{w} | {fmt_d(p['c1_dec'])} | {fmt_d(p['c1_bet'])} | {fmt_d(p['c6_bet'])} |")
    L += ["\n## 3. Fold − global (same episodes) and direction − its PC control\n",
          "| model | contrast | C1 decode [CI] | C1 bet-exact [CI] | C6 bet-exact [CI] |", "|---|---|---|---|---|"]
    for m, e in res.items():
        for k, p in e["extra"].items():
            L.append(f"| {m} | {k} | {fmt_d(p['c1_dec'])} | {fmt_d(p['c1_bet'])} | {fmt_d(p['c6_bet'])} |")
    L += ["\n## 4. Pre-specified LOSO decisions (from each model's analysis.json)\n",
          "| model | direction | drop global | drop fold [CI] | fold/global | verdict | C6 drop (CI upper) | C6 spared | folds consistent |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m, e in res.items():
        for d in DIRS:
            x = e["decisions"][d]
            ratio = f"{x['ratio']:.2f}" if x.get("ratio") is not None else "–"
            L.append(f"| {m} | {d} | {x['drop_global']:.3f} | {x['drop_fold']:.3f} [{x['ci_fold_drop'][0]:.3f}, "
                     f"{x['ci_fold_drop'][1]:.3f}] | {ratio} | {x['verdict']} | {x['c6_drop']:.3f} ({x['c6_drop_ci_upper']:.3f}) | "
                     f"{x['c6_spared']} | {x['folds_consistent'].split(' ')[0]} |")
    L += ["\n## 5. Per-fold spread of held-out (fold-fit) drops vs matched none (n = 51 per fold; min – max over 6 folds)\n",
          "| model | direction | C1 decode drop | per fold (f0..f5) | C1 bet-exact drop | C6 bet-exact drop | ARC fold-fit per fold |",
          "|---|---|---|---|---|---|---|"]
    for m, e in res.items():
        for d in DIRS:
            rg, rows = e["per_fold_extra"][d + "_range"], e["per_fold_extra"][d]
            pf = " ".join(f"{r['c1_dec_drop_fold']:+.2f}" for r in rows)
            arc = " ".join(f"{r['arc_fold']:.2f}" for r in rows)
            L.append(f"| {m} | {d} | {rg['c1_dec_drop_fold'][0]:+.3f} – {rg['c1_dec_drop_fold'][1]:+.3f} | {pf} | "
                     f"{rg['c1_bet_drop_fold'][0]:+.3f} – {rg['c1_bet_drop_fold'][1]:+.3f} | "
                     f"{rg['c6_bet_drop_fold'][0]:+.3f} – {rg['c6_bet_drop_fold'][1]:+.3f} | {arc} |")
    L += ["\nARC for matched none per fold: " + "; ".join(f"{m} " + " ".join(f"{a:.2f}" for a in e["arc_none_by_fold"])
                                                  for m, e in res.items()),
          "\n## 6. Probe geometry (see notes/xmodel_results.md Part 1)\n",
          "| model | L | C1vC4-mean AUROC L0 | C1vC4-full AUROC L0 | state ≥0.5 onset (l/(L−1)) | ≥0.8 onset | max abs cos(dm, state) | var state / PC-2D post-onset |",
          "|---|---|---|---|---|---|---|---|"]
    for m, g in G.items():
        s, L_ = g["summary"], g["n_layers"]
        on = s["state_onset05"]
        vs = g["var_share_per_layer"]
        L.append(f"| {m} | {L_} | {s['auroc_mean_L0']:.3f} | {s['auroc_full_L0']:.3f} | L{on} ({s['state_onset05_rel']:.2f}) | "
                 f"L{s['state_onset08']} ({s['state_onset08_rel']:.2f}) | {s['cos_maxabs_max']:.3f} | "
                 f"{np.mean(vs['state'][on:]):.3f} / {np.mean(vs['pc2d'][on:]):.3f} |")
    TABLES.write_text("\n".join(L) + "\n")


def xfigure(res, G):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.5})
    fig = plt.figure(figsize=(7, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.45])
    a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    for m, g in G.items():
        x = np.arange(g["n_layers"]) / (g["n_layers"] - 1)
        a1.plot(x, g["state_acc"], color=COL[m], lw=1.3, label=f"{m} (L={g['n_layers']})")
        a2.plot(x, g["auroc_C1vC4_mean"], color=COL[m], lw=1.3)
        a2.plot(x, g["auroc_C1vC4_full"], color=COL[m], lw=1.0, ls="--")
    a1.axhline(1 / 3, color="0.5", lw=0.7, ls=":")
    a1.text(0.80, 0.36, "chance", color="0.4", fontsize=6.5)
    a1.set(xlabel="relative depth  l/(L\u22121)", ylabel="3-way state acc. (LOSO)", ylim=(0, 1.04), xlim=(0, 1),
           title="(a) covert state decodable at decision token")
    a1.legend(loc="upper left", frameon=False)
    a2.set(xlabel="relative depth  l/(L\u22121)", ylabel="C1 vs C4 AUROC (LOSO)", ylim=(0.955, 1.002), xlim=(0, 1),
           title="(b) detector separability (incl. layer 0)")
    a2.legend(handles=[Line2D([], [], color="0.3", lw=1.3, label="msg-mean pooling"),
                       Line2D([], [], color="0.3", lw=1.0, ls="--", label="end-of-turn token")],
              loc="lower right", frameon=False)

    def pts(ax, ms, dodge=0.0):
        for mi, (m, e) in enumerate(res.items()):
            for d in DIRS:
                p = e["paired_from_analysis"][f"{d}_fold"]
                x, xl = -p["c1_dec"]["delta"], sorted(-np.array(p["c1_dec"]["ci"]))
                y, yl = -p["c6_bet"]["delta"], sorted(-np.array(p["c6_bet"]["ci"]))
                off = (1 - mi) * dodge                    # display-only vertical dodge (inset), 7B up, 32B down
                y, yl = y + off, [yl[0] + off, yl[1] + off]
                ctrl = d in ("pc1d", "pc2d")
                ax.errorbar(x, y, xerr=[[x - xl[0]], [xl[1] - x]], yerr=[[y - yl[0]], [yl[1] - y]], fmt=MK[d],
                            ms=ms, color=COL[m], mfc="white" if ctrl else COL[m], mew=1.1, elinewidth=0.8,
                            capsize=1.5, zorder=3, alpha=0.9)
        ax.axhline(0, color="0.6", lw=0.6, zorder=1)
        ax.axvline(0, color="0.6", lw=0.6, zorder=1)

    pts(a3, 5.5)
    a3.set(xlabel="C1 covert decode drop vs matched none (held-out fit)",
           ylabel="C6 explicit bet-exact drop vs matched none",
           title="(c) held-out (fold-fit) ablations, pooled n=306, 95% CI", xlim=(-0.05, 0.8), ylim=(-0.04, 0.42))
    h = [Line2D([], [], color=COL[m], marker="o", ls="", ms=5, label=m) for m in res]
    names = dict(dm="dm detector", pc1d="PC-1D ctrl", state="state 2-D", pc2d="PC-2D ctrl", lr="LR detector")
    h += [Line2D([], [], color="0.3", marker=MK[d], ls="", ms=5, mfc="white" if d in ("pc1d", "pc2d") else "0.3",
                 label=names[d]) for d in DIRS]
    a3.legend(handles=h, loc="lower right", frameon=True, framealpha=0.9, edgecolor="0.8", ncol=2,
              handletextpad=0.3, columnspacing=0.8)
    ins = a3.inset_axes([0.16, 0.55, 0.41, 0.38])
    pts(ins, 4.5, dodge=0.003)
    ins.set_xlim(-0.03, 0.13)
    ins.set_ylim(-0.008, 0.022)
    ins.tick_params(labelsize=6)
    ins.set_title("zoom near 0; models dodged by \u00b10.003 in y", fontsize=6, pad=2)
    a3.indicate_inset_zoom(ins, edgecolor="0.5")
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG}.{ext}", dpi=200)


if __name__ == "__main__":
    {"probes32": probes32, "geometry": geometry, "compare": compare}[sys.argv[1]]()
