"""Fig 2 (probe AUROC / state decoding by layer) and Fig 3 (per-token projection test of Rose et al.'s spike claim).
  .venv-analysis/bin/python exp/figs_probes.py
Writes figures/fig2_probe_layers.{png,pdf}, figures/fig3_token_trace.{png,pdf}, data/token_trace_stats.json."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
import env  # noqa: E402

FIG, ACTS = ROOT / "figures", ROOT / "data/acts"
C = dict(blue="#0072B2", orange="#D55E00", green="#009E73", grey="#7f7f7f", purple="#CC79A7", sky="#56B4E9")
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "font.family": "sans-serif"})


def fig2():
    p = json.load(open(ROOT / "data/probes.json"))
    L = np.arange(p["n_layers"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 3.0))
    lab = {"detect_C1vC4_mean": ("C1 vs C4 (msg span)", C["blue"], "-"),
           "detect_C1vC2_mean": ("C1 vs C2 (msg span)", C["sky"], "--"),
           "detect_C1vC4_full": ("C1 vs C4 (end-of-turn)", C["purple"], "-."),
           "detect_C1vC3_full": ("C1 vs C3, prompt-confounded (end-of-turn)", C["grey"], ":")}
    if "detect_C1vC5_mean" in p["auroc"]:
        lab["detect_C1vC5_mean"] = ("C1 vs C5 scheme-mismatch (msg span)", C["orange"], "-")
        lab["detect_C1vC5_last"] = ("C1 vs C5 scheme-mismatch (decision token)", C["green"], "-")
    for k, (t, c, ls) in lab.items():
        a.plot(L, p["auroc"][k], color=c, ls=ls, label=t, lw=1.4)
    a.set(xlabel="layer", ylabel="AUROC (leave-one-scheme-out)", title="(a) Collusion detectors", ylim=(0, 1.02))
    a.axhline(0.5, color="k", lw=0.5, ls=":")
    a.legend(fontsize=5.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    b.plot(L, p["state_acc"]["C1_last"], color=C["green"], lw=1.4, label="hidden state S (3-way), decision token")
    b.axhline(1 / 3, color="k", lw=0.5, ls=":")
    b.text(0.5, 1 / 3 + 0.02, "chance", fontsize=6)
    b.set(xlabel="layer", ylabel="accuracy (held-out scheme)", title="(b) Decoded covert content", ylim=(0, 1.02))
    b.legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig2_probe_layers.{ext}", dpi=300)


def fig3():
    meta = json.load(open(ACTS / "meta.json"))
    msgs = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/messages.jsonl")}
    groups = np.array(meta["scheme_id"])
    TL = meta["trace_layers"]
    X1, X4 = (np.load(ACTS / f"{c}.npz")["mean"].astype(np.float32) for c in ("C1", "C4"))
    out, per_layer = {}, {}
    for cond in ("C1", "C2", "C5"):
        z = np.load(ACTS / f"{cond}_traces.npz")
        acts, lens, idx = z["acts"].astype(np.float32), z["lengths"], z["idx"]
        cm = meta["conditions"][cond]
        for li, l in enumerate(TL):
            rows = []
            for i, k in enumerate(idx):
                # leave-one-scheme-out direction: trained without this example's scheme (no circularity)
                tr = groups != groups[k]
                d = X1[tr, l].mean(0) - X4[tr, l].mean(0)
                d /= np.linalg.norm(d)
                mid = (X1[tr, l] @ d).mean() / 2 + (X4[tr, l] @ d).mean() / 2  # midpoint between class means
                proj = acts[i, li, :lens[i]] @ d - mid
                toks = cm["traces"]["tokens"][str(k)]
                n_msg = cm["traces"]["msg_len"][str(k)]
                topic = msgs[meta["ids"][k]]["topic"]
                kw = np.array([bool(env.TOPIC_RE[topic].search(t)) for t in toks[:n_msg]])
                rows.append(dict(k=int(k), proj=proj.tolist(), n_msg=n_msg, kw=kw.tolist(), toks=toks))
            per_layer[(cond, l)] = rows
    # stats: within-message keyword vs non-keyword tokens (C1), and C1 vs C2 message mean
    for l in TL:
        r1, r2, r5 = per_layer[("C1", l)], per_layer[("C2", l)], per_layer[("C5", l)]
        # C5 = identical message tokens as C1 (same trace idx), receiver's scheme lacks the topic
        kw5 = [np.mean(np.array(b["proj"][:b["n_msg"]])[np.array(a["kw"])]) for a, b in zip(r1, r5) if any(a["kw"])]
        kw_m = [np.mean(np.array(r["proj"][:r["n_msg"]])[np.array(r["kw"])]) for r in r1 if any(r["kw"])]
        nk_m = [np.mean(np.array(r["proj"][:r["n_msg"]])[~np.array(r["kw"])]) for r in r1 if any(r["kw"])]
        post1 = [np.mean(r["proj"][r["n_msg"]:]) for r in r1 if len(r["proj"]) > r["n_msg"]]
        m1 = [np.mean(r["proj"][:r["n_msg"]]) for r in r1]
        m2 = [np.mean(r["proj"][:r["n_msg"]]) for r in r2]
        out[l] = dict(n_with_kw=len(kw_m), kw_mean=float(np.mean(kw_m)), nonkw_mean=float(np.mean(nk_m)),
                      kw_vs_nonkw_wilcoxon_p=float(stats.wilcoxon(kw_m, nk_m).pvalue),
                      C1_msg_mean=float(np.mean(m1)), C2_msg_mean=float(np.mean(m2)),
                      C1_post_msg_mean=float(np.mean(post1)),
                      C1_vs_C2_mannwhitney_p=float(stats.mannwhitneyu(m1, m2).pvalue),
                      C5_kw_mean=float(np.mean(kw5)), C5_msg_mean=float(np.mean([np.mean(r["proj"][:r["n_msg"]]) for r in r5])),
                      C1_vs_C5_kw_wilcoxon_p=float(stats.wilcoxon(kw_m, kw5).pvalue))
    json.dump(out, open(ROOT / "data/token_trace_stats.json", "w"), indent=1)

    l = 20 if 20 in TL else TL[len(TL) // 2]
    fig = plt.figure(figsize=(7, 2.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    for row, cond in enumerate(("C1", "C5")):
        ax = fig.add_subplot(gs[row, 0])
        r = per_layer[(cond, l)][next(i for i, r in enumerate(per_layer[("C1", l)]) if any(r["kw"]) and r["n_msg"] <= 22)]
        n = min(len(r["proj"]), 32)
        v = np.array(r["proj"][:n])
        ax.bar(range(n), v, color=[C["orange"] if (j < r["n_msg"] and r["kw"][j]) else
                                   (C["blue"] if j < r["n_msg"] else C["grey"]) for j in range(n)], width=0.85)
        ax.set_xticks(range(n))
        ax.set_xticklabels([t.replace("\n", "⏎") for t in r["toks"][:n]], rotation=90, fontsize=5)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel({"C1": "C1: signal", "C5": "C5: same msg,\nnot a signal"}[cond], fontsize=7)
        if row == 0:
            ax.set_title(f"(a) Projection on d_detect per token, layer {l} (orange: topic word; grey: after message)",
                         fontsize=6.5, loc="left")
    ax = fig.add_subplot(gs[:, 1])
    xs = np.arange(len(TL))
    for j, (key, lab, c) in enumerate([("kw_mean", "C1 topic tokens", C["orange"]),
                                       ("C5_kw_mean", "same topic tokens, C5 (not a signal)", C["purple"]),
                                       ("nonkw_mean", "C1 other msg tokens", C["blue"]),
                                       ("C2_msg_mean", "C2 (honest msg) tokens", C["grey"])]):
        ax.bar(xs + (j - 1.5) * 0.2, [out[t][key] for t in TL], 0.2, color=c, label=lab)
    ax.set_xticks(xs)
    ax.set_xticklabels(TL)
    ax.set(xlabel="layer", ylabel="mean projection (centred)", title="(b) Mean over 40 messages")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig3_token_trace.{ext}", dpi=300)
    return out


if __name__ == "__main__":
    fig2()
    print(json.dumps(fig3(), indent=1))
