"""Final targeted extension analysis (notes/verify2_plan.md). FROZEN before any S5/S6 run; later fixes logged in
notes/verify2_log.md.

  .venv-analysis/bin/python exp/verify2_analyze.py   -> data/verify/v2_analysis.json, data/verify/v2_tables.md

S5 (sampled C1 receiver, T=0.7, top_p=1.0, seeds 0-7):
- Per draw: strict decode, exact bet (separate outcomes), malformed (any field missing), valid-but-wrong (both fields present,
  not strict-correct-with-true-rule-bet); mutually exclusive failure taxonomy = exp/phase0_free.c1_cell cross-tab.
- Main analysis: average the 8 seeds WITHIN episode; paired contrasts per episode; CI = bootstrap over original context
  clusters (scheme, seed) within scheme, each carrying its 3 states, all seeds and all conditions (exp/verify_analyze.Boot).
  Contexts are generated from (scheme, seed) and do not recur across schemes (checked below), so scheme-stratified
  cluster resampling preserves all context dependence.
- Monte Carlo uncertainty: per seed s, the paired contrast over episodes Delta_s; MC SE of the seed-mean =
  sd_s(Delta_s)/sqrt(8); seed range reported. 8 seeds fixed; no extension.
- Greedy comparison: per episode, sampled (seed-mean) contrast minus the existing greedy S1 contrast; cluster CI of the
  difference (the regimes are compared directly, not by significance status).
- C6, recall, ARC are greedy deterministic capability checks (not sampled).
S6 (greedy): 2x2 placement factorial conditional on late removal (L18-27), for dm and pc1d and dm - pc1d. S1 cells (dm_all,
pc1d_all, pc1d_L18) are reused only if the S6 same-container anchors (none, dm_L18) reproduce S1 outputs exactly.
No thresholds; CIs including 0 are not equivalence.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_free import c1_cell  # noqa: E402
from verify_analyze import Boot, arc_ci, contrast, episodes, ctx, fmt, table  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
V = Path(__import__("os").environ.get("VERIFY_DIR", ROOT / "data/verify"))  # override only for code tests
BETS = {"COLD": 0, "NEUTRAL": 25, "HOT": 100}
S5C = ["none", "M2_L18", "PC2_L18", "M2_L18_match", "PC2_L18_match", "dm_L18", "dm_all", "pc1d_L18", "pc1d_all"]
PRIMARY = {"M2_L18 − PC2_L18": ("M2_L18", "PC2_L18"), "M2_L18_match − PC2_L18_match": ("M2_L18_match", "PC2_L18_match"),
           "dm_L18 − pc1d_L18": ("dm_L18", "pc1d_L18"), "dm_all − pc1d_all": ("dm_all", "pc1d_all")}
OUTS = ("c1", "bet", "malformed", "valid_wrong")
SEEDS = [str(s) for s in range(8)]


def draw(x):
    dc, bc = c1_cell(x)
    mal = dc == "missing" or bc == "missing"
    return dict(c1=float(dc == "correct"), bet=float(bc == "true_rule"), malformed=float(mal),
                valid_wrong=float(not mal and not (dc == "correct" and bc == "true_rule")), cat=f"{dc}|{bc}")


def load(prefix):
    R = {}
    for f in sorted((V / "runs/v2_s5s6").glob(f"{prefix}_f*/metrics.json")):
        r = json.load(open(f))
        k, c = re.fullmatch(rf"{prefix}_f(\d)_(.+)", r["tag"]).groups()
        R.setdefault(c, {})[int(k)] = r
    return R


def s1_runs():
    R = {}
    for f in sorted((V / "runs/s1_7b").glob("f*/metrics.json")):
        r = json.load(open(f))
        k, c = re.fullmatch(r"f(\d)_(.+)", r["tag"]).groups()
        R.setdefault(c, {})[int(k)] = r
    return R


def context_check():
    rows = [json.loads(l) for l in open(ROOT / "data/messages.jsonl")]
    key = {}
    for r in rows:
        key.setdefault((r["sender"], r["receiver"], r["hand"]), set()).add(r["scheme_id"])
    return dict(n_contexts=len(key), recurring_across_schemes=sum(len(v) > 1 for v in key.values()))


def s5(out, T):
    R, S1 = load("s5"), s1_runs()
    if not R:
        out["s5"] = "missing"; T.append("\n## S5: not run\n"); return
    D, G = {}, {}            # D[c][id] = {seed: draw}; G[c][id] = greedy S1 outcomes
    anchor = {}
    for c in S5C:
        D[c] = {}
        same = n = 0
        for k, r in R[c].items():
            for sd in SEEDS:
                for x in r["C1_sampled_rows"][sd]:
                    D[c].setdefault(x["id"], {})[sd] = draw(x)
            s1 = {x["id"]: x["output"] for x in S1[c][k]["C1_rows"]}
            s16 = {x["id"]: x["output"] for x in S1[c][k]["C6_rows"]}
            same += sum(x["output"] == s1[x["id"]] for x in r["C1_rows"]) + sum(x["output"] == s16[x["id"]] for x in r["C6_rows"])
            n += len(r["C1_rows"]) + len(r["C6_rows"])
        anchor[c] = dict(greedy_C1_C6_identical_to_S1=same / n)
        G[c] = {}
        for k in S1[c]:
            G[c].update(episodes(S1[c][k]))
    ids = sorted(D["none"])
    assert all(len(D[c][i]) == 8 for c in S5C for i in ids)
    B, Be = Boot(ids), Boot(ids, episode_level=True)
    M = {c: {m: np.array([np.mean([D[c][i][s][m] for s in SEEDS]) for i in ids]) for m in OUTS} for c in S5C}
    res = dict(n_episodes=len(ids), n_draws_per_cell=len(ids) * 8, anchors=anchor, absolute={}, contrasts={},
               capability={}, meter={}, runtime={})
    for c in S5C:
        cats = {}
        for i in ids:
            for s in SEEDS:
                cats[D[c][i][s]["cat"]] = cats.get(D[c][i][s]["cat"], 0) + 1
        res["absolute"][c] = {m: dict(mean=float(M[c][m].mean()), ci=B.ci(ids, M[c][m])) for m in OUTS}
        res["absolute"][c]["taxonomy_draws"] = cats
        res["absolute"][c]["per_seed_c1"] = [float(np.mean([D[c][i][s]["c1"] for i in ids])) for s in SEEDS]
        gk = {}
        for r in R[c].values():
            gk.setdefault("c6", []).extend(x["bet"] == BETS[x["state"]] for x in r["C6_rows"])
            gk.setdefault("recall", []).extend(x["answer"] == x["topic"] for x in r["recall"])
        arc = np.array([[it["pred"] == it["gold"] for it in R[c][k]["arc_items"]] for k in sorted(R[c])], float)
        res["capability"][c] = dict(c6=float(np.mean(gk["c6"])), recall=float(np.mean(gk["recall"])),
                                    arc=float(arc.mean()), arc_ci=arc_ci(arc))
        mt = [sum(v[0] / v[2] for kk, v in R[c][k].get("meter", {}).get(f"C1_s{s}", {}).items() if kk.startswith(ph))
              for k in R[c] for s in SEEDS for ph in ("prefill",)]
        md = [sum(v[0] / v[2] for kk, v in R[c][k].get("meter", {}).get(f"C1_s{s}", {}).items() if kk.startswith("decode"))
              for k in R[c] for s in SEEDS]
        res["meter"][c] = dict(prefill_sum_per_token_mean_over_folds_seeds=float(np.mean(mt)) if mt else None,
                               decode_sum_per_token_mean=float(np.mean(md)) if md else None)
        res["runtime"][c] = float(np.mean([r.get("seconds_sampled", 0) for r in R[c].values()]))
    con = dict(PRIMARY)
    con.update({f"{c} − none": (c, "none") for c in S5C if c != "none"})
    for name, (a, b) in con.items():
        r = {}
        for m in OUTS:
            v = M[a][m] - M[b][m]
            per_seed = np.array([np.mean([D[a][i][s][m] - D[b][i][s][m] for i in ids]) for s in SEEDS])
            d = dict(delta=float(v.mean()), ci=B.ci(ids, v), ci_episode=Be.ci(ids, v),
                     mc_se=float(per_seed.std(ddof=1) / np.sqrt(8)), seed_min=float(per_seed.min()),
                     seed_max=float(per_seed.max()))
            if m in ("c1", "bet") and a in G and b in G:
                g = np.array([G[a][i][m] - G[b][i][m] for i in ids])
                d["greedy"] = dict(delta=float(g.mean()), ci=B.ci(ids, g))
                d["sampled_minus_greedy"] = dict(delta=float((v - g).mean()), ci=B.ci(ids, v - g))
            r[m] = d
        res["contrasts"][name] = r
    out["s5"] = res
    T.append("\n## S5 · sampled C1 receiver (T = 0.7, top_p = 1.0, seeds 0–7; one robustness setting)\n")
    T.append(f"n = {len(ids)} episodes × 8 draws per cell; seed-averaged within episode; context-cluster CIs. C6/recall/ARC are "
             "greedy capability checks. Greedy anchors (C1+C6 outputs identical to S1): " +
             ", ".join(f"{c} {v['greedy_C1_C6_identical_to_S1']:.3f}" for c, v in anchor.items()) + "\n")
    rows = []
    for c in S5C:
        a, cap = res["absolute"][c], res["capability"][c]
        rows.append([c, fmt(a["c1"], "mean"), fmt(a["bet"], "mean"), fmt(a["malformed"], "mean"), fmt(a["valid_wrong"], "mean"),
                     f"{min(a['per_seed_c1']):.3f}–{max(a['per_seed_c1']):.3f}", f"{cap['c6']:.3f} / {cap['recall']:.3f} / {cap['arc']:.3f}",
                     f"{res['runtime'][c]:.0f}s"])
    T += table(rows, ["cell", "strict decode", "exact bet", "malformed", "valid-but-wrong", "per-seed strict range",
                      "greedy C6 / recall / ARC", "8-draw time"])
    rows = []
    for name, r in res["contrasts"].items():
        for m in ("c1", "bet", "malformed"):
            d = r[m]
            g = d.get("greedy")
            rows.append([name if m == "c1" else "", {"c1": "strict", "bet": "exact bet", "malformed": "malformed"}[m], fmt(d),
                         f"{d['mc_se']:.3f} ({d['seed_min']:+.3f}…{d['seed_max']:+.3f})",
                         fmt(g) if g else "–", fmt(d["sampled_minus_greedy"]) if g else "–"])
    T += ["\nPaired contrasts (sampled, seed-averaged) with Monte Carlo SE over seeds and the greedy S1 contrast:\n"]
    T += table(rows, ["contrast", "outcome", "sampled Δ [cluster CI]", "MC SE (seed range)", "greedy Δ [CI]",
                      "sampled − greedy [CI]"])
    T.append("\nFailure taxonomy over all draws (decode|bet, mutually exclusive): " + "; ".join(
        f"{c}: " + ", ".join(f"{k} {v}" for k, v in sorted(res["absolute"][c]["taxonomy_draws"].items())) for c in S5C))


FACT = lambda x: {"E|noPre (EMB_LATE − L18)": {f"{x}_EMB_LATE": 1, f"{x}_L18": -1},
                  "E|Pre (all − PRE_LATE)": {f"{x}_all": 1, f"{x}_PRE_LATE": -1},
                  "P|noEmb (PRE_LATE − L18)": {f"{x}_PRE_LATE": 1, f"{x}_L18": -1},
                  "P|Emb (all − EMB_LATE)": {f"{x}_all": 1, f"{x}_EMB_LATE": -1},
                  "Interaction (all − EMB_LATE − PRE_LATE + L18)": {f"{x}_all": 1, f"{x}_EMB_LATE": -1,
                                                                   f"{x}_PRE_LATE": -1, f"{x}_L18": 1}}


def s6(out, T):
    R6, S1 = load("s6"), s1_runs()
    if not R6:
        out["s6"] = "missing"; T.append("\n## S6: not run\n"); return
    E, A = {}, {}

    def add(name, runs):
        for k, r in runs.items():
            E.setdefault(name, {}).update(episodes(r))
            A.setdefault(name, {})[k] = np.array([it["pred"] == it["gold"] for it in r["arc_items"]], float)
    for c, runs in R6.items():
        add(c, runs)
    anchors = {}
    for c in ("none", "dm_L18"):
        n = same = 0
        for k, r in R6[c].items():
            for cond in ("C1_rows", "C6_rows"):  # fix 2: compare C1 with C1 and C6 with C6 (ids are shared across conditions)
                s1 = {x["id"]: x["output"] for x in S1[c][k][cond]}
                same += sum(x["output"] == s1[x["id"]] for x in r[cond])
            n += len(r["C1_rows"]) + len(r["C6_rows"])
        anchors[c] = same / n
    reuse_ok = all(v == 1.0 for v in anchors.values())
    for c in ("dm_all", "pc1d_all", "pc1d_L18"):
        add(c, S1[c])
    ids = sorted(E["none"])
    B, Be = Boot(ids), Boot(ids, episode_level=True)
    res = dict(anchors_identical_to_S1=anchors, s1_reuse_valid=reuse_ok, contrasts={})
    con = {f"{c} − none": {c: 1, "none": -1} for c in E if c != "none"}
    for x in ("dm", "pc1d"):
        con.update({f"{x}: {k}": w for k, w in FACT(x).items()})
    for k in FACT("dm"):
        w = dict(FACT("dm")[k])
        for c, wt in FACT("pc1d")[k].items():
            w[c] = w.get(c, 0) - wt
        con[f"dm − pc1d: {k}"] = w
    for name, w in con.items():
        res["contrasts"][name] = contrast(E, w, B, Be, A)
    out["s6"] = res
    T.append("\n## S6 · detector placement factorial (greedy; conditional on late removal L18–27)\n")
    T.append(f"S6 anchors identical to S1 (C1+C6 outputs): {json.dumps(anchors)}; reuse of S1 dm_all/pc1d_all/pc1d_L18 "
             f"{'VALID' if reuse_ok else 'NOT VALID (reported, cells flagged)'}.\n")
    rows = []
    for name, r in res["contrasts"].items():
        rows.append([name, fmt(r["c1"]), fmt(r["bet"]), fmt(r["malformed"]), fmt(r["c6"]), fmt(r["recall"]),
                     fmt(r["arc"]) if "arc" in r else "–", f"[{r['c1']['ci_episode'][0]:+.3f}, {r['c1']['ci_episode'][1]:+.3f}]"])
    T += table(rows, ["contrast", "Δ strict", "Δ exact bet", "Δ malformed", "Δ C6", "Δ recall", "Δ ARC (item CI)",
                      "strict episode-level CI"])


def validation(out, T):
    f = V / "validation/v2_validate.json"
    if not f.exists():
        out["validation"] = None; return
    Vd = json.load(open(f))["results"]
    res = {}
    for name in ("dm_greedy", "pc1d_greedy", "M2_sampled", "dm_sampled", "pc1d_sampled"):
        C = Vd[name]
        hk = {"none": set(), "L18": {f"L{l}" for l in range(18, 28)}, "all": {"emb"} | {f"L{l}" for l in range(28)},
              "EMB_LATE": {"emb"} | {f"L{l}" for l in range(18, 28)}, "PRE_LATE": {f"L{l}" for l in range(28)}}
        r0 = {r["loc"]: r for r in C["none"]["records"] if r["phase"] == "prefill"}
        rr = {}
        for tag, v in C.items():
            h = hk[tag.split("_", 1)[1] if tag != "none" else "none"]
            recs = v["records"]
            pre = {r["loc"]: r for r in recs if r["phase"] == "prefill"}
            first = min((int(l[1:]) for l in h if l != "emb"), default=99)
            up = [] if "emb" in h else (["emb"] + [f"L{l}" for l in range(min(first, 28)) if f"L{l}" not in h])
            rr[tag] = dict(hooked_max={ph: max([r["ratio"] for r in recs if r["loc"] in h and r["phase"] == ph], default=None)
                                       for ph in ("prefill", "decode")},
                           unhooked_min=min([r["ratio"] for r in recs if r["loc"] not in h], default=None),
                           upstream_prefill_bitwise=f"{sum((pre[l]['s1'], pre[l]['s2']) == (r0[l]['s1'], r0[l]['s2']) for l in up)}/{len(up)}",
                           decode_calls_min=min(sum(r["loc"] == l and r["phase"] == "decode" for r in recs) for l in pre))
        res[name] = rr
    for name in ("M2_match_sampled", "PC2_match_sampled"):
        recs = Vd[name]["records"]
        res[name] = dict(
            n_calls=len(recs),
            max_abs_span_ratio_minus_1_minus_d={ph: max(abs(r["span_ratio"] - (1 - r["dose"])) for r in recs if r["phase"] == ph)
                                                for ph in ("prefill", "decode")},
            energy_ratio_range={ph: [min(r["energy_ratio"] for r in recs if r["phase"] == ph),
                                     max(r["energy_ratio"] for r in recs if r["phase"] == ph)] for ph in ("prefill", "decode")},
            orth_frac_max=max(r["orth_frac"] for r in recs),
            repeats={k: v["identical"] for k, v in Vd[name]["repeats"].items()},
            gen_config={k: Vd[name]["generation_config"].get(k) for k in ("do_sample", "temperature", "top_p", "top_k",
                                                                          "repetition_penalty")})
    # cross-container repeat: validation none seed 0 vs smoke s5 none seed 0 on the same 9 ids, same batch
    sm = V / "runs/v2_s5s6/smoke_s5_f0_none/metrics.json"
    if sm.exists():
        so = [x["output"] for x in json.load(open(sm))["C1_sampled_rows"]["0"]]
        res["cross_container_seed0_none_identical"] = so == Vd["M2_match_sampled"]["repeats"]["none_s0"]["outputs"]
    out["validation"] = res
    T.append("\n## Validation\n")
    T += [f"- {k}: {json.dumps(v)}" for k, v in res.items()]


def main():
    out, T = {"context_check": context_check()}, ["# Final targeted extension (exp/verify2_analyze.py; plan notes/verify2_plan.md)\n"]
    T.append(f"Context identity check: {json.dumps(out['context_check'])}\n")
    validation(out, T)
    s5(out, T)
    s6(out, T)
    json.dump(out, open(V / "v2_analysis.json", "w"), indent=1, default=float)
    (V / "v2_tables.md").write_text("\n".join(T) + "\n")
    print("wrote", V / "v2_tables.md")


if __name__ == "__main__":
    main()
