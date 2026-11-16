"""S6 (notes/s6_14b_fresh_plan.md): pre-registered 14B L26 retest on fresh messages. Primary: C1 strict decode
M2_L26 - PC2_L26 pooled over 918 episodes, context-cluster bootstrap (verify_analyze.Boot: clusters pool seeds,
stratified by scheme). Replicates iff the upper CI bound < 0. Output: data/verify/s6_analysis.json."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from verify_analyze import Boot, ROOT  # noqa: E402

RUNS = ROOT / "data/verify/runs/s6_14b_fresh"
SEEDS, CFGS = (1401, 1402, 1403), ("none", "M2_L26", "PC2_L26", "M2_L32", "PC2_L32")


def load(cfg):
    out, ext = {}, {"C6": [], "arc": [], "recall": []}
    for s in SEEDS:
        for k in range(6):
            r = json.load(open(RUNS / f"m{s}_f{k}_{cfg}/metrics.json"))
            out.update({x["id"]: x for x in r["C1_rows"]})
            ext["C6"].append(r["C6"]["bet_exact_acc"]); ext["arc"].append(r.get("arc_acc")); ext["recall"].append(r["recall_acc"])
    return out, {k: float(np.mean(v)) for k, v in ext.items()}


def main():
    R, X = zip(*(load(c) for c in CFGS))
    R, X = dict(zip(CFGS, R)), dict(zip(CFGS, X))
    ids = sorted(R["none"])
    assert len(ids) == 918 and all(sorted(r) == ids for r in R.values())
    B = Boot(ids)
    acc = {c: np.array([R[c][i]["decoded"] == R[c][i]["state"] for i in ids], float) for c in CFGS}
    tohot = {c: np.array([R[c][i]["state"] != "HOT" and R[c][i]["decoded"] == "HOT" for i in ids], float) for c in CFGS}
    awayhot = {c: np.array([R[c][i]["state"] == "HOT" and R[c][i]["decoded"] != "HOT" for i in ids], float) for c in CFGS}
    nh = np.array([R["none"][i]["state"] != "HOT" for i in ids]); h = ~nh
    ia = np.array(ids)

    def est(v, mask=None):
        m = np.ones(len(ids), bool) if mask is None else mask
        return dict(mean=float(v[m].mean()), ci=B.ci(list(ia[m]), v[m]))

    res = dict(n=len(ids), n_clusters=len(B.clusters), absolute={}, contrasts={}, per_seed={}, capability=X)
    for c in CFGS:
        res["absolute"][c] = dict(acc=est(acc[c]), to_hot=est(tohot[c], nh), hot_to_other=est(awayhot[c], h))
    for name, a, b in (("PRIMARY L26: M2 - PC2", "M2_L26", "PC2_L26"), ("L32: M2 - PC2", "M2_L32", "PC2_L32"),
                       ("M2_L26 - none", "M2_L26", "none"), ("PC2_L26 - none", "PC2_L26", "none")):
        res["contrasts"][name] = dict(acc=est(acc[a] - acc[b]), to_hot=est(tohot[a] - tohot[b], nh),
                                      hot_to_other=est(awayhot[a] - awayhot[b], h))
    for s in SEEDS:
        m = np.array([i.startswith(f"m{s}_") for i in ids])
        res["per_seed"][s] = float((acc["M2_L26"] - acc["PC2_L26"])[m].mean())
    p = res["contrasts"]["PRIMARY L26: M2 - PC2"]["acc"]
    res["replicates"] = p["ci"][1] < 0
    json.dump(res, open(ROOT / "data/verify/s6_analysis.json", "w"), indent=1)

    f = lambda e: f"{e['mean']:+.3f} [{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}]"
    print(f"n={res['n']} clusters={res['n_clusters']}")
    for c, v in res["absolute"].items():
        print(f"{c:8s} acc {f(v['acc'])}  ->HOT {f(v['to_hot'])}  HOT->other {f(v['hot_to_other'])}  {X[c]}")
    for n, v in res["contrasts"].items():
        print(f"{n:24s} acc {f(v['acc'])}  ->HOT {f(v['to_hot'])}  HOT->other {f(v['hot_to_other'])}")
    print("per-seed L26 gap", res["per_seed"], "| REPLICATES:", res["replicates"])


if __name__ == "__main__":
    main()
