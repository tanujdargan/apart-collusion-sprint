"""Analysis of the 7B placement unbundling (notes/unbundle_plan.md). Written before any unbundling result was read.

  .venv-analysis/bin/python exp/unbundle_analyze.py  -> data/unbundle/analysis.json, data/unbundle/tables.md

Cells: none, M2_late, M2_all, EMB_LATE, PRE_LATE (fold-fit M2 state span; 6 held-out folds × 51 episodes).
Factorial contrasts per unit (episode for C1/C6/recall; fold×item for ARC):
  E|noPre = EMB_LATE − M2_late;  E|Pre = M2_all − PRE_LATE;  P|noEmb = PRE_LATE − M2_late;  P|Emb = M2_all − EMB_LATE;
  Interaction = M2_all − EMB_LATE − PRE_LATE + M2_late.
Uncertainty: bootstrap over contexts (episodes) stratified by scheme, each context carrying all its configs' outcomes;
ARC: bootstrap over the 100 items, each carrying all folds × configs. Exact McNemar for binary paired contrasts only.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data/unbundle/runs"
CFGS = ["none", "M2_late", "M2_all", "EMB_LATE", "PRE_LATE"]
FOLDS = [str(k) for k in range(6)]
BETS = {"COLD": 0, "NEUTRAL": 25, "HOT": 100}
CONTRASTS = {"vs_none:M2_late": {"M2_late": 1, "none": -1}, "vs_none:M2_all": {"M2_all": 1, "none": -1},
             "vs_none:EMB_LATE": {"EMB_LATE": 1, "none": -1}, "vs_none:PRE_LATE": {"PRE_LATE": 1, "none": -1},
             "E|noPre (EMB_LATE−M2_late)": {"EMB_LATE": 1, "M2_late": -1},
             "E|Pre (M2_all−PRE_LATE)": {"M2_all": 1, "PRE_LATE": -1},
             "P|noEmb (PRE_LATE−M2_late)": {"PRE_LATE": 1, "M2_late": -1},
             "P|Emb (M2_all−EMB_LATE)": {"M2_all": 1, "EMB_LATE": -1},
             "Interaction (M2_all−EMB_LATE−PRE_LATE+M2_late)": {"M2_all": 1, "EMB_LATE": -1, "PRE_LATE": -1, "M2_late": 1}}
NBOOT = 10000


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def load():
    splits = json.load(open(ROOT / "data/loso/splits.json"))
    R = {}
    for k in FOLDS:
        for c in CFGS:
            r = json.load(open(RUNS / f"unb_f{k}_{c}" / "metrics.json"))
            assert r["ids"] == splits[k]["eval_ids"], (k, c)
            for cond in ("C1", "C6"):
                assert [x["id"] for x in r[cond + "_rows"]] == r["ids"], (k, c, cond)
            assert [x["id"] for x in r["recall"]] == r["ids"]
            assert len(r["arc_items"]) == 100
            R[(k, c)] = r
    return R


def units(R):
    """Per-config arrays: episode-level outcomes [306] (fold order) and ARC item-level [6 folds, 100 items]."""
    U = {}
    for c in CFGS:
        e = {m: [] for m in ("c1", "c1bet", "c6", "recall", "c1betnone", "c1decnone", "c6betnone", "scheme")}
        arc = np.zeros((6, 100))
        for fi, k in enumerate(FOLDS):
            r = R[(k, c)]
            for x in r["C1_rows"]:
                e["c1"].append(x["decoded"] == x["state"]); e["c1bet"].append(x["bet"] == BETS[x["state"]])
                e["c1betnone"].append(x["bet"] is None); e["c1decnone"].append(x["decoded"] is None); e["scheme"].append(int(k))
            for x in r["C6_rows"]:
                e["c6"].append(x["bet"] == BETS[x["state"]]); e["c6betnone"].append(x["bet"] is None)
            e["recall"] += [x["answer"] == x["topic"] for x in r["recall"]]
            arc[fi] = [it["pred"] == it["gold"] for it in r["arc_items"]]
        U[c] = {m: np.array(v, float) for m, v in e.items()}
        U[c]["arc"] = arc
    return U


def boot_episode(vals, scheme, rng):
    idx = [np.where(scheme == s)[0] for s in np.unique(scheme)]
    return np.array([np.concatenate([vals[rng.choice(ix, len(ix))] for ix in idx]).mean() for _ in range(NBOOT)])


def boot_items(mat, rng):  # mat [6, 100]: resample items (columns), keep all folds
    return np.array([mat[:, rng.integers(0, mat.shape[1], mat.shape[1])].mean() for _ in range(NBOOT)])


def main():
    R = load()
    U = units(R)
    sch = U["none"]["scheme"]
    out = dict(per_config={}, contrasts={}, per_fold={}, reproduction={})
    for c in CFGS:
        u, n = U[c], len(U[c]["c1"])
        out["per_config"][c] = dict(
            n=n, c1=float(u["c1"].mean()), c1_ci=wilson(u["c1"].sum(), n), c1_bet=float(u["c1bet"].mean()),
            c6=float(u["c6"].mean()), c6_ci=wilson(u["c6"].sum(), n), recall=float(u["recall"].mean()),
            recall_ci=wilson(u["recall"].sum(), n), arc=float(u["arc"].mean()), arc_ci=wilson(u["arc"].sum(), u["arc"].size),
            c1_bet_none=int(u["c1betnone"].sum()), c1_decoded_none=int(u["c1decnone"].sum()), c6_bet_none=int(u["c6betnone"].sum()))
        out["per_fold"][c] = {k: dict(c1=float(u["c1"][sch == int(k)].mean()), c6=float(u["c6"][sch == int(k)].mean()),
                                      arc=float(u["arc"][int(k)].mean())) for k in FOLDS}
    for name, w in CONTRASTS.items():
        res = {}
        for m in ("c1", "c1bet", "c6", "recall"):
            v = sum(wt * U[c][m] for c, wt in w.items())
            b = boot_episode(v, sch, np.random.default_rng(0))
            d = dict(delta=float(v.mean()), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])
            if len(w) == 2:
                (a, _), (bb, _) = sorted(w.items(), key=lambda x: -x[1])
                n10 = int(((U[a][m] == 1) & (U[bb][m] == 0)).sum()); n01 = int(((U[a][m] == 0) & (U[bb][m] == 1)).sum())
                d["mcnemar_p"] = float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
            res[m] = d
        va = sum(wt * U[c]["arc"] for c, wt in w.items())
        b = boot_items(va, np.random.default_rng(0))
        res["arc"] = dict(delta=float(va.mean()), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])
        out["contrasts"][name] = res
    # reproduction of the core matrix (same fold directions, episodes, order, batching)
    for c, core in (("none", "none"), ("M2_late", "M2_late"), ("M2_all", "M2_all")):
        same1 = same6 = n = 0
        for k in FOLDS:
            a = R[(k, c)]
            b = json.load(open(ROOT / f"data/redund/7b/runs/redund_f{k}_{core}/metrics.json"))
            same1 += sum(x["output"] == y["output"] for x, y in zip(a["C1_rows"], b["C1_rows"]))
            same6 += sum(x["output"] == y["output"] for x, y in zip(a["C6_rows"], b["C6_rows"]))
            n += len(a["C1_rows"])
        out["reproduction"][c] = dict(c1_identical=same1 / n, c6_identical=same6 / n)
    json.dump(out, open(ROOT / "data/unbundle/analysis.json", "w"), indent=1)
    T = ["# 7B placement unbundling (exp/unbundle_analyze.py)\n",
         "n = 306 held-out episodes (6 folds × 51); ARC n = 600 (6 folds × 100 items). Wilson 95% CIs (absolute). Contrast "
         "CIs: context bootstrap stratified by scheme (episodes), item bootstrap for ARC. McNemar for binary pairs only.\n",
         "| config | C1 decode [CI] | C1 bet-exact | C6 bet-exact [CI] | recall | ARC [CI] | C1 bet None / decoded None | C6 bet None |",
         "|---|---|---|---|---|---|---|---|"]
    for c, p in out["per_config"].items():
        T.append(f"| {c} | {p['c1']:.3f} [{p['c1_ci'][0]:.3f}, {p['c1_ci'][1]:.3f}] | {p['c1_bet']:.3f} | "
                 f"{p['c6']:.3f} [{p['c6_ci'][0]:.3f}, {p['c6_ci'][1]:.3f}] | {p['recall']:.3f} | "
                 f"{p['arc']:.3f} [{p['arc_ci'][0]:.3f}, {p['arc_ci'][1]:.3f}] | {p['c1_bet_none']} / {p['c1_decoded_none']} | {p['c6_bet_none']} |")
    T += ["\n| contrast | Δ C1 decode [CI] (p) | Δ C1 bet-exact [CI] | Δ C6 [CI] | Δ recall [CI] | Δ ARC [CI] |",
          "|---|---|---|---|---|---|"]
    f = lambda d: f"{d['delta']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]"
    for name, r in out["contrasts"].items():
        p = f" ({r['c1']['mcnemar_p']:.2g})" if "mcnemar_p" in r["c1"] else ""
        T.append(f"| {name} | {f(r['c1'])}{p} | {f(r['c1bet'])} | {f(r['c6'])} | {f(r['recall'])} | {f(r['arc'])} |")
    T += ["\nPer-fold C1 decode (schemes 0–5):"] + [
        f"- {c}: " + " ".join(f"{out['per_fold'][c][k]['c1']:.2f}" for k in FOLDS) for c in CFGS]
    T += ["\nReproduction of the core matrix (identical outputs): " +
          ", ".join(f"{c} C1 {v['c1_identical']:.3f} / C6 {v['c6_identical']:.3f}" for c, v in out["reproduction"].items())]
    (ROOT / "data/unbundle/tables.md").write_text("\n".join(T) + "\n")
    return out


if __name__ == "__main__":
    main()
