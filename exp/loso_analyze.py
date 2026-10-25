"""Analysis of the post-hoc LOSO robustness check, implementing the rules fixed in notes/loso_robustness_plan.md.

  .venv-analysis/bin/python exp/loso_analyze.py  -> data/loso/analysis.json, data/loso/tables.md, figures/fig7_loso.{png,pdf}
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
RUNS, OUT = ROOT / "data/loso/runs", ROOT / "data/loso"
FOLDS = [str(k) for k in range(6)]
DIRS = ["dm", "pc1d", "state", "pc2d", "lr"]
CFGS = ["none"] + [f"{d}_{w}" for d in DIRS for w in ("global", "fold")]
BETS = {"COLD": 0, "NEUTRAL": 25, "HOT": 100}
RNG = np.random.default_rng(0)
NBOOT = 10000


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def load():
    splits = json.load(open(OUT / "splits.json"))
    R = {}
    for k in FOLDS:
        for c in CFGS:
            m = json.load(open(RUNS / f"loso_f{k}_{c}" / "metrics.json"))
            assert m["ids"] == splits[k]["eval_ids"], (k, c, "episode ids differ from split")
            for cond in ("C1", "C6"):
                assert [r["id"] for r in m[cond + "_rows"]] == m["ids"], (k, c, cond, "row order")
            R[(k, c)] = m
    ids = sum((splits[k]["eval_ids"] for k in FOLDS), [])
    assert len(ids) == len(set(ids)) == 306, "held-out sets must partition the 306 episodes"
    return splits, R, ids


def vectors(R, c):
    """Pooled per-episode outcome vectors over the 6 folds (fold order), plus scheme labels."""
    v = {key: [] for key in ("c1_dec", "c1_bet", "c1_betnone", "c1_decnone", "c6_bet", "c6_betnone", "scheme")}
    for k in FOLDS:
        m = R[(k, c)]
        for r in m["C1_rows"]:
            v["c1_dec"].append(r["decoded"] == r["state"])
            v["c1_bet"].append(r["bet"] == BETS[r["state"]])
            v["c1_betnone"].append(r["bet"] is None)
            v["c1_decnone"].append(r["decoded"] is None)
            v["scheme"].append(int(k))
        for r in m["C6_rows"]:
            v["c6_bet"].append(r["bet"] == BETS[r["state"]])
            v["c6_betnone"].append(r["bet"] is None)
    return {k: np.array(x) for k, x in v.items()}


def paired(a, b, scheme):
    """Δ = mean(a) − mean(b) on identical episodes; stratified-by-scheme paired bootstrap 95% CI; exact McNemar p."""
    a, b = a.astype(float), b.astype(float)
    diff = a - b
    idx = [np.where(scheme == s)[0] for s in np.unique(scheme)]
    boots = np.empty(NBOOT)
    for i in range(NBOOT):
        boots[i] = np.concatenate([diff[RNG.choice(ix, len(ix))] for ix in idx]).mean()
    n01, n10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return dict(delta=float(diff.mean()), ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                mcnemar_p=float(p), a_only=n10, b_only=n01)


def main():
    splits, R, ids = load()
    V = {c: vectors(R, c) for c in CFGS}
    sch = V["none"]["scheme"]
    res = dict(n_total=len(ids), per_config={}, paired={}, decisions={}, per_fold={}, determinism={})
    for c in CFGS:
        v = V[c]
        n = len(v["c1_dec"])
        res["per_config"][c] = dict(
            n=n, c1_decode=float(v["c1_dec"].mean()), c1_decode_ci=wilson(v["c1_dec"].sum(), n),
            c1_bet_exact=float(v["c1_bet"].mean()), c1_bet_none=int(v["c1_betnone"].sum()),
            c1_decoded_none=int(v["c1_decnone"].sum()),
            c6_bet_exact=float(v["c6_bet"].mean()), c6_bet_exact_ci=wilson(v["c6_bet"].sum(), n),
            c6_bet_none=int(v["c6_betnone"].sum()),
            recall=float(np.mean([R[(k, c)]["recall_acc"] for k in FOLDS])),
            arc_mean=float(np.mean([R[(k, c)]["arc_acc"] for k in FOLDS])),
            arc_by_fold=[R[(k, c)]["arc_acc"] for k in FOLDS],
            decoded_dist={s: int(sum(R[(k, c)]["C1"]["decoded_dist"][s] for k in FOLDS)) for s in ("COLD", "NEUTRAL", "HOT", "None")})
        if c != "none":
            res["paired"][c] = {m: paired(V[c][m], V["none"][m], sch) for m in ("c1_dec", "c1_bet", "c6_bet")}
    for d in DIRS:
        g, f = f"{d}_global", f"{d}_fold"
        res["paired"][f"{d}_fold_minus_global"] = {m: paired(V[f][m], V[g][m], sch) for m in ("c1_dec", "c6_bet")}
        drop_f = -res["paired"][f]["c1_dec"]["delta"]
        drop_g = -res["paired"][g]["c1_dec"]["delta"]
        ci = res["paired"][f]["c1_dec"]["ci"]
        excl0 = ci[1] < 0 or ci[0] > 0
        ratio = drop_f / drop_g if drop_g > 0 else None
        verdict = ("DOES NOT SURVIVE" if not excl0 else
                   "SURVIVES" if (ratio is not None and ratio >= 0.5 and drop_f > 0) else "ATTENUATED")
        c6 = res["paired"][f]["c6_bet"]
        c6_drop, c6_drop_hi = -c6["delta"], -c6["ci"][0]
        per_fold = []
        for k in FOLDS:
            mk = lambda c: np.mean([r["decoded"] == r["state"] for r in R[(k, c)]["C1_rows"]])
            fd, gd = mk("none") - mk(f), mk("none") - mk(g)
            per_fold.append(dict(fold=int(k), none=float(mk("none")), global_=float(mk(g)), fold_=float(mk(f)),
                                 drop_fold=float(fd), drop_global=float(gd),
                                 c6_fold=float(np.mean([r["bet"] == BETS[r["state"]] for r in R[(k, f)]["C6_rows"]]))))
        consistent = sum(p["drop_global"] > 0 and p["drop_fold"] >= 0.5 * p["drop_global"] for p in per_fold)
        res["per_fold"][d] = per_fold
        res["decisions"][d] = dict(drop_fold=drop_f, drop_global=drop_g, ratio=ratio, ci_fold_drop=[-ci[1], -ci[0]],
                                   verdict=verdict, c6_drop=c6_drop, c6_drop_ci_upper=c6_drop_hi,
                                   c6_spared=bool(c6_drop <= 0.05 and c6_drop_hi <= 0.10),
                                   folds_consistent=f"{consistent}/6 (folds with drop_fold ≥ 0.5·drop_global > 0)")
    for d, ctrl in (("dm", "pc1d"), ("state", "pc2d")):
        res["paired"][f"{d}_fold_minus_{ctrl}_fold"] = {m: paired(V[f"{d}_fold"][m], V[f"{ctrl}_fold"][m], sch)
                                                       for m in ("c1_dec", "c6_bet")}
    # determinism: matched baseline vs original 306-episode rows (different batching), and global subset vs original
    orig = dict(none=("baseline", "base_C6"), dm_global=("abl_detect_C1vC4_mean_dm", "abl_detect_C1vC4_mean_dm_C6"),
                state_global=("abl_state_C1_last_2d", "abl_state_C1_last_2d_C6"), pc2d_global=("abl_pc_ctrl_2d", "abl_pc_ctrl_2d"),
                pc1d_global=("abl_pc_ctrl_1d", "abl_pc_ctrl_1d"), lr_global=("abl_detect_C1vC4_mean_lr", "abl_detect_C1vC4_mean_lr_C6"))
    for c, (t1, t6) in orig.items():
        o1 = {r["id"]: r for r in json.load(open(ROOT / f"data/interv/{t1}/metrics.json"))["C1_rows"]}
        o6 = {r["id"]: r for r in json.load(open(ROOT / f"data/interv/{t6}/metrics.json"))["C6_rows"]}
        new1 = [r for k in FOLDS for r in R[(k, c)]["C1_rows"]]
        new6 = [r for k in FOLDS for r in R[(k, c)]["C6_rows"]]
        res["determinism"][c] = dict(
            c1_identical_output=float(np.mean([r["output"] == o1[r["id"]]["output"] for r in new1])),
            c1_same_decoded=float(np.mean([r["decoded"] == o1[r["id"]]["decoded"] for r in new1])),
            c1_decode_orig_same_ids=float(np.mean([o1[r["id"]]["decoded"] == r["state"] for r in new1])),
            c6_identical_output=float(np.mean([r["output"] == o6[r["id"]]["output"] for r in new6])),
            c6_same_bet=float(np.mean([r["bet"] == o6[r["id"]]["bet"] for r in new6])))
    json.dump(res, open(OUT / "analysis.json", "w"), indent=1)
    write_tables(res)
    figure(res)
    return res


def write_tables(res):
    L = ["# LOSO robustness check — generated tables (exp/loso_analyze.py)\n",
         f"Pooled over 6 held-out folds, n = {res['n_total']} episodes (51 per fold). Wilson 95% CIs. Parse failures = bet unparseable.\n",
         "| config | C1 decode [95% CI] | C1 bet-exact | C1 parse fail | C1 decoded None | C1 decoded C/N/H | C6 bet-exact [95% CI] | C6 parse fail | recall | ARC (mean of folds) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for c, p in res["per_config"].items():
        dd = p["decoded_dist"]
        L.append(f"| {c} | {p['c1_decode']:.3f} [{p['c1_decode_ci'][0]:.3f}–{p['c1_decode_ci'][1]:.3f}] | {p['c1_bet_exact']:.3f} | "
                 f"{p['c1_bet_none']} | {p['c1_decoded_none']} | {dd['COLD']}/{dd['NEUTRAL']}/{dd['HOT']} | "
                 f"{p['c6_bet_exact']:.3f} [{p['c6_bet_exact_ci'][0]:.3f}–{p['c6_bet_exact_ci'][1]:.3f}] | {p['c6_bet_none']} | "
                 f"{p['recall']:.3f} | {p['arc_mean']:.3f} |")
    L += ["\nPaired changes vs matched baseline (`none`, same episodes and batching); stratified bootstrap 95% CI; exact McNemar p.\n",
          "| config | Δ C1 decode [CI] | McNemar p | Δ C1 bet-exact [CI] | Δ C6 bet-exact [CI] | McNemar p (C6) |", "|---|---|---|---|---|---|"]
    for c, p in res["paired"].items():
        a, b = p["c1_dec"], p["c6_bet"]
        bb = p.get("c1_bet")
        L.append(f"| {c} | {a['delta']:+.3f} [{a['ci'][0]:+.3f}, {a['ci'][1]:+.3f}] | {a['mcnemar_p']:.2g} | "
                 + (f"{bb['delta']:+.3f} [{bb['ci'][0]:+.3f}, {bb['ci'][1]:+.3f}]" if bb else "–")
                 + f" | {b['delta']:+.3f} [{b['ci'][0]:+.3f}, {b['ci'][1]:+.3f}] | {b['mcnemar_p']:.2g} |")
    L += ["\nPre-specified decisions (C1 decode; drop = none − ablated):\n",
          "| direction | drop global | drop fold [CI] | fold/global | verdict | C6 drop (upper CI) | C6 spared? | per-fold consistency |",
          "|---|---|---|---|---|---|---|---|"]
    for d, x in res["decisions"].items():
        ratio = "–" if x["ratio"] is None else f"{x['ratio']:.2f}"
        L.append(f"| {d} | {x['drop_global']:.3f} | {x['drop_fold']:.3f} [{x['ci_fold_drop'][0]:.3f}, {x['ci_fold_drop'][1]:.3f}] | "
                 f"{ratio} | **{x['verdict']}** | "
                 f"{x['c6_drop']:.3f} ({x['c6_drop_ci_upper']:.3f}) | {x['c6_spared']} | {x['folds_consistent']} |")
    L += ["\nPer-fold C1 decode (n = 51 each):\n", "| direction | fold | none | global | fold-fit | C6 bet-exact (fold-fit) |", "|---|---|---|---|---|---|"]
    for d, pf in res["per_fold"].items():
        for p in pf:
            L.append(f"| {d} | {p['fold']} | {p['none']:.3f} | {p['global_']:.3f} | {p['fold_']:.3f} | {p['c6_fold']:.3f} |")
    L += ["\nDeterminism vs original 306-episode runs (different batch composition):\n",
          "| config | C1 identical output | C1 same decoded | C1 decode, orig rows same ids | C6 identical output | C6 same bet |", "|---|---|---|---|---|---|"]
    for c, x in res["determinism"].items():
        L.append(f"| {c} | {x['c1_identical_output']:.3f} | {x['c1_same_decoded']:.3f} | {x['c1_decode_orig_same_ids']:.3f} | "
                 f"{x['c6_identical_output']:.3f} | {x['c6_same_bet']:.3f} |")
    open(OUT / "tables.md", "w").write("\n".join(L) + "\n")


def figure(res):
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 2.7))
    x = np.arange(len(DIRS))
    for ax, key, ci_key, title in ((a, "c1_decode", "c1_decode_ci", "(a) Covert decoding (C1)"),
                                   (b, "c6_bet_exact", "c6_bet_exact_ci", "(b) Explicit-state decision (C6)")):
        for j, (w, col) in enumerate((("global", "#999999"), ("fold", "#0072B2"))):
            vals = [res["per_config"][f"{d}_{w}"][key] for d in DIRS]
            cis = np.array([res["per_config"][f"{d}_{w}"][ci_key] for d in DIRS])
            ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, color=col,
                   yerr=[np.array(vals) - cis[:, 0], cis[:, 1] - np.array(vals)], capsize=2, error_kw=dict(lw=0.6),
                   label="fitted on all schemes (original)" if w == "global" else "fitted without held-out scheme")
        if key == "c1_decode":
            for i, d in enumerate(DIRS):
                ax.plot([i + 0.19] * 6, [p["fold_"] for p in res["per_fold"][d]], ".", color="k", ms=2.5, zorder=4)
        ax.axhline(res["per_config"]["none"][key], color="k", lw=0.7, ls="--", label="matched baseline")
        ax.set_xticks(x)
        ax.set_xticklabels(["dm\ndetector", "PC-1D\ncontrol", "state\n2-D", "PC-2D\ncontrol", "LR\ndetector"], fontsize=7)
        ax.set(ylim=(0, 1.05), title=title, ylabel="accuracy (pooled held-out, n=306)")
    a.axhspan(0.30, 0.37, color="#eeeeee", zorder=0)
    b.legend(fontsize=6, frameon=False, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"figures/fig7_loso.{ext}", dpi=300)


if __name__ == "__main__":
    r = main()
    for d, x in r["decisions"].items():
        print(d, x["verdict"], f"drop_global {x['drop_global']:.3f} drop_fold {x['drop_fold']:.3f}", "C6 spared", x["c6_spared"], x["folds_consistent"])
