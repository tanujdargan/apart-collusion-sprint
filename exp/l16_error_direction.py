"""Post-submission follow-up (reviewer 2): why does the 7B state-over-control accuracy gap reverse at the L16 cutoff?
Splits C1 decode errors by direction (non-HOT -> HOT vs HOT -> other) for the S1 cutoff runs, and reports per-layer
removed energy. CPU only; reads committed runs s1_7b, s1s_7b, s1x_7b. CIs: episode bootstrap (not cluster)."""
import json, random

R = "data/verify/runs/s1_7b"
CFGS = ["none", "M2_L16", "M2_L18", "M2_L20", "PC2_L16", "PC2_L18", "PC2_L20", "M2_L16s", "PC2_L16s", "M2_L16x", "PC2_L16x"]
# *_L16x: L16/L17 hooked with the L18 subspace (verify_matrix.py s1x)
# *_L16s: L16/L17 hooks scaled to each arm's own L18 energy (exp/verify_matrix.py s1s, follow-up GPU run)


def load(c):
    d = {"L16s": "data/verify/runs/s1s_7b", "L16x": "data/verify/runs/s1x_7b"}.get(c[-4:], R)
    return [json.load(open(f"{d}/f{f}_{c}/metrics.json")) for f in range(6)]


def rows(c):
    return {x["id"]: x for m in load(c) for x in m["C1_rows"]}


def to_hot(x): return x["state"] != "HOT" and x["decoded"] == "HOT"
def away_hot(x): return x["state"] == "HOT" and x["decoded"] != "HOT"


def diff_ci(a, b, fn, hot_true, B=2000):
    ids = [i for i in a if (a[i]["state"] == "HOT") == hot_true]
    d = [fn(a[i]) - fn(b[i]) for i in ids]
    bs = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(B))
    return sum(d) / len(d), bs[int(.025 * B)], bs[int(.975 * B)]


if __name__ == "__main__":
    random.seed(0)
    cfg = {c: rows(c) for c in CFGS}
    assert all(set(r) == set(cfg["none"]) for r in cfg.values()), "configs must cover the same 306 episodes"
    print("config    acc        ->HOT (true non-HOT)  HOT->other (true HOT)")
    for c, r in cfg.items():
        nh = [x for x in r.values() if x["state"] != "HOT"]; h = [x for x in r.values() if x["state"] == "HOT"]
        acc = sum(x["decoded"] == x["state"] for x in r.values()) / len(r)
        print(f"{c:9s} acc {acc:.3f}  {sum(map(to_hot, nh)) / len(nh):.3f}                 {sum(map(away_hot, h)) / len(h):.3f}")
    fmt = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
    for cut in ("16", "16s", "16x", "18", "20"):
        a, b = cfg["M2_L" + cut], cfg["PC2_L" + cut]
        print(f"L{cut} state-control: ->HOT {fmt(diff_ci(a, b, to_hot, False))}  HOT->other {fmt(diff_ci(a, b, away_hot, True))}")
    print("\nper-layer prefill removed energy / token (C1, mean of folds), % of ||h||^2")
    for c in ("M2_L16", "M2_L16s", "M2_L16x", "M2_L18", "PC2_L16", "PC2_L16s", "PC2_L16x", "PC2_L18"):
        acc = {}
        for m in load(c):
            for k, (d2, h2, n) in m["meter"]["C1"].items():
                if k.startswith("prefill"):
                    acc.setdefault(int(k.split(":")[1]), []).append((d2 / n, 100 * d2 / h2))
        print(c, " ".join(f"L{l}:{sum(e for e, _ in v) / len(v):.0f}({sum(p for _, p in v) / len(v):.1f}%)" for l, v in sorted(acc.items())))
