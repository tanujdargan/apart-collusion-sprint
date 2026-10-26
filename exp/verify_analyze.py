"""Verification-extension analysis (notes/verify_plan.md §6). FROZEN before any S/G-stage outcome existed; later fixes
are logged in notes/verify_log.md as such (and marked outcome-independent or not).

  .venv-analysis/bin/python exp/verify_analyze.py   -> data/verify/analysis.json, data/verify/tables.md, data/verify/fig_dose.png

Units & dependence
- Evaluation unit: episode. Resampling unit: CONTEXT CLUSTER = (prefix, scheme, seed) - the 3 states of a context share
  names/cards/hand; fresh set m<s>_<id> maps to its original context, so pooled fresh-set contrasts resample each context
  with ALL its sets. Stratified by scheme; 10k replicates, seed 0; one replicate matrix shared by every contrast of a
  stage. Episode-level bootstrap (earlier analyses' unit) is reported as a continuity sensitivity.
- ARC: per-item outcomes [fold, item]; bootstrap over the 100 items, each item carrying all folds x configs.
- Exact McNemar only for two-config binary contrasts.
- Schemes are fixed (6), stratified not sampled: CIs cover new contexts/messages under these schemes, model, prompts.
Metrics: C1 strict decode, C1 exact bet, C1 malformed (any field missing), C6 exact bet, C6 bet missing, recall, ARC;
mutually exclusive parse categories (exp/phase0_free.c1_cell); achieved perturbation (meter) per phase.
No thresholds, no verdicts; a CI including 0 is not equivalence.
"""
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase0_free import c1_cell  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
V = Path(__import__("os").environ.get("VERIFY_DIR", ROOT / "data/verify"))  # override only for code tests
RUNS = V / "runs"
BETS = {"COLD": 0, "NEUTRAL": 25, "HOT": 100}
NBOOT = 10000
METRICS = ("c1", "bet", "malformed", "c6", "c6_missing", "recall")
FRESH = [101, 202, 303, 404, 505]


# ----------------------------------------------------------------------------- loading
def ctx(eid):
    m = re.fullmatch(r"(?:m\d+_)?([sn])(\d)_(COLD|NEUTRAL|HOT)_(\d+)", eid)
    return (m.group(1), int(m.group(2)), int(m.group(4))), int(m.group(2))


def episodes(r):
    """Per-episode outcomes of one run, keyed by id."""
    out = {}
    c6 = {x["id"]: x for x in r["C6_rows"]}
    rc = {x["id"]: x for x in r["recall"]}
    for x in r["C1_rows"]:
        dc, bc = c1_cell(x)
        y = c6[x["id"]]
        out[x["id"]] = dict(c1=float(dc == "correct"), bet=float(bc == "true_rule"),
                            malformed=float(dc == "missing" or bc == "missing"), cat=f"{dc}|{bc}",
                            c6=float(y["bet"] == BETS[y["state"]]), c6_missing=float(y["bet"] is None),
                            recall=float(rc[x["id"]]["answer"] == rc[x["id"]]["topic"]), output=x["output"],
                            c6_output=y["output"])
    return out


def load_matrix(name, parse):
    """-> {cfg: {id: outcomes}}, {cfg: {fold: arc_items}}, {cfg: {fold: run}} ; parse(tag) -> (unitset, fold, cfg)."""
    d = RUNS / name
    if not d.exists():
        return None
    E, A, R = {}, {}, {}
    for f in sorted(d.glob("*/metrics.json")):
        if f.parent.name.startswith("smoke_"):
            continue
        r = json.load(open(f))
        us, k, c = parse(r["tag"])
        key = c if us is None else (us, c)
        E.setdefault(key, {}).update(episodes(r))
        R.setdefault(key, {})[k] = r
        if "arc_items" in r:
            A.setdefault(key, {})[k] = np.array([it["pred"] == it["gold"] for it in r["arc_items"]], float)
    return E, A, R


def p_fold(tag):
    m = re.fullmatch(r"f(\d)_(.+)", tag)
    return None, int(m.group(1)), m.group(2)


def p_fresh(tag):
    m = re.fullmatch(r"m(\d+)_f(\d)_(.+)", tag)
    return int(m.group(1)), int(m.group(2)), m.group(3)


def load_ref(path_glob, strip):
    """Reference runs (earlier stages) -> {cfg: {id: outcomes}} for reproduction / original-message references."""
    E = {}
    for f in sorted(ROOT.glob(path_glob)):
        r = json.load(open(f))
        c = re.sub(strip, "", f.parent.name)
        E.setdefault(c, {}).update(episodes(r))
    return E


# ----------------------------------------------------------------------------- statistics
class Boot:
    def __init__(self, ids, episode_level=False, seed=0):
        keys = [(i if episode_level else ctx(i)[0]) for i in ids]
        strat = {k: ctx(i)[1] for k, i in zip(keys, ids)}
        self.clusters = sorted(set(keys))
        self.ix = {c: n for n, c in enumerate(self.clusters)}
        rng = np.random.default_rng(seed)
        W = np.zeros((NBOOT, len(self.clusters)))
        for s in sorted(set(strat.values())):
            cs = np.array([self.ix[c] for c in self.clusters if strat[c] == s])
            draw = cs[rng.integers(0, len(cs), (NBOOT, len(cs)))]
            for b in range(NBOOT):
                np.add.at(W[b], draw[b], 1)
        self.W, self.ep = W, episode_level

    def ci(self, ids, v):
        cl = np.array([self.ix[i if self.ep else ctx(i)[0]] for i in ids])
        s = np.bincount(cl, v, len(self.clusters))
        n = np.bincount(cl, None, len(self.clusters))
        m = (self.W @ s) / (self.W @ n)
        return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def arc_ci(mat, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, mat.shape[1], (NBOOT, mat.shape[1]))
    b = mat[:, idx].mean(axis=(0, 2))
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def contrast(E, w, B, Be, A=None, ids=None, subset=None):
    """w: {cfg: weight}. Paired per-episode linear combination over the common ids (optionally restricted)."""
    ids = ids or sorted(set.intersection(*(set(E[c]) for c in w)))
    if subset is not None:
        ids = [i for i in ids if i in subset]
    out = dict(n=len(ids))
    if not ids:
        return out
    for m in METRICS:
        v = np.array([sum(wt * E[c][i][m] for c, wt in w.items()) for i in ids])
        d = dict(delta=float(v.mean()), ci=B.ci(ids, v), ci_episode=Be.ci(ids, v))
        if len(w) == 2 and m in ("c1", "bet", "c6", "recall"):
            a, b = sorted(w, key=lambda c: -w[c])
            n10 = sum(E[a][i][m] == 1 and E[b][i][m] == 0 for i in ids)
            n01 = sum(E[a][i][m] == 0 and E[b][i][m] == 1 for i in ids)
            d["mcnemar_p"] = float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
        out[m] = d
    if A is not None and all(c in A for c in w):
        folds = sorted(set.intersection(*(set(A[c]) for c in w)))
        mat = np.array([sum(wt * A[c][k] for c, wt in w.items()) for k in folds])
        out["arc"] = dict(delta=float(mat.mean()), ci=arc_ci(mat), n_folds=len(folds))
    return out


def absolute(E, c, B, A=None, subset=None):
    ids = sorted(E[c]) if subset is None else sorted(i for i in E[c] if i in subset)
    out = dict(n=len(ids))
    for m in METRICS:
        v = np.array([E[c][i][m] for i in ids])
        out[m] = dict(mean=float(v.mean()), ci=B.ci(ids, v))
    cats = {}
    for i in ids:
        cats[E[c][i]["cat"]] = cats.get(E[c][i]["cat"], 0) + 1
    g = lambda dc=None, bc=None: sum(v for k, v in cats.items() if (dc is None or k.split("|")[0] == dc)
                                     and (bc is None or k.split("|")[1] == bc))
    out["parse"] = dict(crosstab=cats, both_correct=g("correct", "true_rule"),
                        malformed=dict(total=sum(v for k, v in cats.items() if "missing" in k),
                                       decode_missing_bet_present=g("missing") - g("missing", "missing"),
                                       decode_present_bet_missing=g("correct", "missing") + g("wrong", "missing"),
                                       both_missing=g("missing", "missing"),
                                       decode_missing_bet_true_rule=g("missing", "true_rule")),
                        valid_wrong=dict(decode_correct_bet_wrong=g("correct", "decoded_rule") + g("correct", "other"),
                                         decode_wrong_bet_coherent=g("wrong", "decoded_rule"),
                                         decode_wrong_bet_true_rule=g("wrong", "true_rule"),
                                         decode_wrong_bet_other=g("wrong", "other")))
    if A is not None and c in A:
        mat = np.array([A[c][k] for k in sorted(A[c])])
        out["arc"] = dict(mean=float(mat.mean()), ci=arc_ci(mat), n_folds=mat.shape[0])
    return out


def repro(E, ref, pairs):
    out = {}
    for a, b in pairs:
        if a not in E or b not in ref:
            out[f"{a}~{b}"] = None
            continue
        ids = sorted(set(E[a]) & set(ref[b]))
        out[f"{a}~{b}"] = dict(n=len(ids), c1_identical=float(np.mean([E[a][i]["output"] == ref[b][i]["output"] for i in ids])),
                               c6_identical=float(np.mean([E[a][i]["c6_output"] == ref[b][i]["c6_output"] for i in ids])))
    return out


def meter(R, c):
    """Pooled achieved perturbation for config c: per location and phase, per-token mean ||delta||^2 and relative
    sum||delta||^2 / sum||h||^2 (C1 only); plus sums over hooked layers. Prefill and decode kept separate."""
    acc = {}
    for k, r in R[c].items():
        for loc, (d2, h2, n) in r.get("meter", {}).get("C1", {}).items():
            a = acc.setdefault(loc, [0.0, 0.0, 0.0])
            a[0] += d2; a[1] += h2; a[2] += n
    per = {loc: dict(per_token=v[0] / v[2], relative=v[0] / v[1], tokens=v[2]) for loc, v in acc.items() if v[2]}
    tot = {ph: dict(sum_per_token=sum(v["per_token"] for l, v in per.items() if l.startswith(ph + ":")),
                    relative=(sum(acc[l][0] for l in acc if l.startswith(ph + ":")) /
                              max(1e-30, sum(acc[l][1] for l in acc if l.startswith(ph + ":")))))
           for ph in ("prefill", "decode")}
    return dict(per_location=per, total=tot)


def meter_fold(R, c, k, loc):
    v = R[c][k].get("meter", {}).get("C1", {}).get(loc)
    return v[0] / v[2] if v and v[2] else None


# ----------------------------------------------------------------------------- stages
def table(rows, cols):
    return ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows]


def fmt(d, key="delta"):
    return f"{d[key]:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]" if key == "delta" else \
        f"{d[key]:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"


def stage_block(title, E, A, cfgs, contrasts, subset=None, R=None):
    ids = sorted(set().union(*(set(E[c]) for c in cfgs if c in E)))
    B, Be = Boot(ids), Boot(ids, episode_level=True)
    res = dict(absolute={}, contrasts={}, n_ids=len(ids), n_clusters=len(B.clusters))
    for c in cfgs:
        if c in E and (subset is None or set(E[c]) & subset):
            res["absolute"][c] = absolute(E, c, B, A, subset)
            if R is not None and c in R:
                res["absolute"][c]["meter"] = meter(R, c)
    for name, w in contrasts.items():
        if all(c in E for c in w):
            r = contrast(E, w, B, Be, A, subset=subset)
            if r["n"]:
                res["contrasts"][name] = r
    T = [f"\n### {title}\n", f"n = {res['n_ids']} episodes, {res['n_clusters']} context clusters"
         + (f"; subset n = {len(subset)}" if subset is not None else "") + ". Absolute: mean [cluster-bootstrap CI].\n"]
    rows = []
    for c, a in res["absolute"].items():
        p = a["parse"]["malformed"]
        rows.append([c, fmt(a["c1"], "mean"), fmt(a["bet"], "mean"), f"{p['total']} ({p['decode_missing_bet_present']}/"
                     f"{p['decode_present_bet_missing']}/{p['both_missing']}; {p['decode_missing_bet_true_rule']})",
                     f"{a['parse']['valid_wrong']['decode_wrong_bet_coherent']}", fmt(a["c6"], "mean") + f" / {int(a['c6_missing']['mean'] * a['n'])}",
                     f"{a['recall']['mean']:.3f}", fmt(a["arc"], "mean") if "arc" in a else "–",
                     (f"{a['meter']['total']['prefill']['sum_per_token']:.1f} / {a['meter']['total']['decode']['sum_per_token']:.1f}"
                      if a.get("meter") and a["meter"]["per_location"] else "–")])
    T += table(rows, ["config", "C1 strict decode", "C1 exact bet", "malformed (dec-miss/bet-miss/both; dec-miss & bet ok)",
                      "wrong-but-coherent", "C6 exact / missing", "recall", "ARC", "achieved Σ_l E‖δ‖² prefill / decode"])
    rows = []
    for name, r in res["contrasts"].items():
        p = f" (p={r['c1']['mcnemar_p']:.2g})" if "mcnemar_p" in r["c1"] else ""
        rows.append([name, fmt(r["c1"]) + p, fmt(r["bet"]), fmt(r["malformed"]), fmt(r["c6"]), fmt(r["recall"]),
                     fmt(r["arc"]) if "arc" in r else "–", f"[{r['c1']['ci_episode'][0]:+.3f}, {r['c1']['ci_episode'][1]:+.3f}]"])
    T += ["\nPaired contrasts (Δ [context-cluster CI]; McNemar p for two-config C1):\n"]
    T += table(rows, ["contrast", "Δ C1 strict", "Δ exact bet", "Δ malformed", "Δ C6", "Δ recall", "Δ ARC (item CI)",
                      "C1 episode-level CI"])
    return res, T


def vs_none(cfgs):
    return {f"{c} − none": {c: 1, "none": -1} for c in cfgs if c != "none"}


def s1(out, T):
    L = load_matrix("s1_7b", p_fold)
    if L is None:
        out["s1"] = "missing"; T.append("\n## S1: not run\n"); return
    E, A, R = L
    cfgs = ["none", "M2_L18", "PC2_L18", "M2_L18_match", "PC2_L18_match", "M2_L18_half", "PC2_L18_half", "dm_L18", "dm_all",
            "M2_L16", "PC2_L16", "M2_L20", "PC2_L20", "pc1d_L18", "pc1d_all"]
    con = vs_none(cfgs)
    con.update({"PRIMARY matched dose: M2_L18_match − PC2_L18_match": {"M2_L18_match": 1, "PC2_L18_match": -1},
                "full dose: M2_L18 − PC2_L18": {"M2_L18": 1, "PC2_L18": -1},
                "half dose: M2_L18_half − PC2_L18_half": {"M2_L18_half": 1, "PC2_L18_half": -1},
                "M2_L18_match − M2_L18": {"M2_L18_match": 1, "M2_L18": -1},
                "PC2_L18_match − PC2_L18": {"PC2_L18_match": 1, "PC2_L18": -1},
                "cutoff L16: M2 − PC2": {"M2_L16": 1, "PC2_L16": -1}, "cutoff L18: M2 − PC2": {"M2_L18": 1, "PC2_L18": -1},
                "cutoff L20: M2 − PC2": {"M2_L20": 1, "PC2_L20": -1},
                "dm_L18 − dm_all": {"dm_L18": 1, "dm_all": -1}, "dm_L18 − pc1d_L18": {"dm_L18": 1, "pc1d_L18": -1},
                "dm_all − pc1d_all": {"dm_all": 1, "pc1d_all": -1}})
    res, t = stage_block("S1 · 7B original messages (fold-fit; 6 folds × 51)", E, A, cfgs, con, R=R)
    ref = load_ref("data/redund/7b/runs/redund_f*/metrics.json", r"^redund_f\d_")
    ref.update({"LOSO_" + k: v for k, v in load_ref("data/loso/runs/loso_f*/metrics.json", r"^loso_f\d_").items()})
    res["reproduction"] = repro(E, ref, [("none", "none"), ("M2_L18", "M2_late"), ("PC2_L18", "PC2_late"),
                                         ("dm_all", "LOSO_dm_fold"), ("pc1d_all", "LOSO_pc1d_fold")])
    # dose: fitted vs achieved (C1 prefill, per fold and layer), matched-pair achieved ratio per layer
    al = json.load(open(V / "dose_alpha.json")) if (V / "dose_alpha.json").exists() else None
    dose = dict(per_layer={})
    for l in range(18, 28):
        row = {}
        for ph in ("prefill", "decode"):
            ms = [meter_fold(R, "M2_L18_match", k, f"{ph}:{l}") for k in R.get("M2_L18_match", {})]
            ps = [meter_fold(R, "PC2_L18_match", k, f"{ph}:{l}") for k in R.get("PC2_L18_match", {})]
            rr = [a / b for a, b in zip(ms, ps) if a and b]
            row[f"{ph}_achieved_ratio_M2_over_PC2"] = dict(median=float(np.median(rr)), min=float(min(rr)), max=float(max(rr))) if rr else None
        if al:
            row["fitted"] = {k: dict(target=al[k][str(l)]["target"], c_M2=al[k][str(l)]["c_M2"], c_PC2=al[k][str(l)]["c_PC2"],
                                     E_M2=al[k][str(l)]["E_M2"], E_PC2=al[k][str(l)]["E_PC2"],
                                     E_mid_ratio_PC2_over_M2=al[k][str(l)]["E_mid_PC2"] / al[k][str(l)]["E_mid_M2"])
                             for k in al}
            fa = []
            for k in R.get("M2_L18_match", {}):
                for c, x in (("M2_L18_match", "M2"), ("PC2_L18_match", "PC2")):
                    a = meter_fold(R, c, k, f"prefill:{l}")
                    if a:
                        fa.append(a / (al[str(k)][str(l)]["target"]))
            row["prefill_achieved_over_fitted_target"] = dict(median=float(np.median(fa)), min=float(min(fa)), max=float(max(fa))) if fa else None
        dose["per_layer"][l] = row
    res["dose"] = dose
    out["s1"] = res
    T.append("\n## S1 · 7B surviving late effect: dose, detector, cutoffs\n")
    T += t
    T.append("\nReproduction (identical outputs): " + ", ".join(f"{k} C1 {v['c1_identical']:.3f}/C6 {v['c6_identical']:.3f}"
                                                              if v else f"{k} n/a" for k, v in res["reproduction"].items()))
    rows = []
    for l, row in dose["per_layer"].items():
        f = row.get("fitted")
        rows.append([str(l), f"{np.median([v['E_PC2'] / v['E_M2'] for v in f.values()]):.2f}" if f else "–",
                     f"{np.median([v['c_M2'] for v in f.values()]):.2f} / {np.median([v['c_PC2'] for v in f.values()]):.2f}" if f else "–",
                     f"{np.median([v['E_mid_ratio_PC2_over_M2'] for v in f.values()]):.2f}" if f else "–",
                     (lambda x: f"{x['median']:.2f} ({x['min']:.2f}–{x['max']:.2f})" if x else "–")(row["prefill_achieved_ratio_M2_over_PC2"]),
                     (lambda x: f"{x['median']:.2f} ({x['min']:.2f}–{x['max']:.2f})" if x else "–")(row["decode_achieved_ratio_M2_over_PC2"]),
                     (lambda x: f"{x['median']:.2f} ({x['min']:.2f}–{x['max']:.2f})" if x else "–")(row.get("prefill_achieved_over_fitted_target"))])
    T += ["\nDose fit (training folds, unablated prompt-only, all prompt tokens) vs achieved (C1, intervened trajectory). "
          "Ratios are medians over folds (range):\n"]
    T += table(rows, ["layer", "fitted E_PC2/E_M2", "dose c_M2 / c_PC2", "E_mid PC2/M2 (diag.)",
                      "achieved M2/PC2 prefill", "achieved M2/PC2 decode", "achieved/fitted target prefill"])
    fig_dose(res)


def fig_dose(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = res["absolute"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for span, col in (("M2", "C0"), ("PC2", "C3")):
        pts = [("none", 0.0)] + [(f"{span}_L18{s}", None) for s in ("_half", "_match", "")]
        xs, ys, yb, lo, hi = [], [], [], [], []
        for c, x0 in pts:
            if c not in a:
                continue
            x = x0 if x0 is not None else a[c]["meter"]["total"]["prefill"]["sum_per_token"]
            xs.append(x); ys.append(a[c]["c1"]["mean"]); yb.append(a[c]["bet"]["mean"])
            lo.append(a[c]["c1"]["mean"] - a[c]["c1"]["ci"][0]); hi.append(a[c]["c1"]["ci"][1] - a[c]["c1"]["mean"])
        o = np.argsort(xs)
        xs, ys, yb, lo, hi = (np.array(v)[o] for v in (xs, ys, yb, lo, hi))
        ax[0].errorbar(xs, ys, yerr=[lo, hi], marker="o", color=col, label=f"{span} (strict decode)", capsize=2)
        ax[0].plot(xs, yb, ls="--", marker="x", color=col, label=f"{span} (exact bet)")
    ax[0].set(xlabel="achieved Σ_l mean-token ‖δ‖², C1 prefill, L18–27", ylabel="C1 accuracy (n=306)",
              title="7B late-placement dose curves", xscale="symlog")
    ax[0].legend(fontsize=7)
    cuts = [16, 18, 20]
    for span, col in (("M2", "C0"), ("PC2", "C3")):
        v = [a.get(f"{span}_L{c}") for c in cuts]
        ax[1].errorbar(cuts, [x["c1"]["mean"] for x in v], yerr=[[x["c1"]["mean"] - x["c1"]["ci"][0] for x in v],
                       [x["c1"]["ci"][1] - x["c1"]["mean"] for x in v]], marker="o", color=col, label=span, capsize=2)
    ax[1].axhline(a["none"]["c1"]["mean"], color="k", lw=0.8, ls=":", label="none")
    ax[1].set(xlabel="first hooked layer (to L27)", ylabel="C1 strict decode", title="cutoff sensitivity (L18 frozen)",
              xticks=cuts)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(V / "fig_dose.png", dpi=160)
    plt.close(fig)


FACT = {"E|noPre (EMB_LATE − M2_late)": {"EMB_LATE": 1, "M2_late": -1},
        "E|Pre (M2_all − PRE_LATE)": {"M2_all": 1, "PRE_LATE": -1},
        "P|noEmb (PRE_LATE − M2_late)": {"PRE_LATE": 1, "M2_late": -1},
        "P|Emb (M2_all − EMB_LATE)": {"M2_all": 1, "EMB_LATE": -1},
        "Interaction": {"M2_all": 1, "EMB_LATE": -1, "PRE_LATE": -1, "M2_late": 1}}


def s2(out, T):
    L = load_matrix("s2_fresh", p_fresh)
    if L is None:
        out["s2"] = "missing"; T.append("\n## S2: not run\n"); return
    E2, _, R2 = L
    cfgs = ["none", "M2_all", "M2_late", "EMB_LATE", "PRE_LATE", "PC2_late", "PC2_all", "PC2_PRE_LATE"]
    con = vs_none(cfgs)
    con.update(FACT)
    con.update({"M2_late − PC2_late": {"M2_late": 1, "PC2_late": -1}, "M2_all − PC2_all": {"M2_all": 1, "PC2_all": -1},
                "PRE_LATE − PC2_PRE_LATE": {"PRE_LATE": 1, "PC2_PRE_LATE": -1}})
    nov = json.load(open(ROOT / "data/fresh/analysis.json"))["overlap"]
    novel = set().union(*(set(nov[f"m{s}"]["novel_ids"]) if f"m{s}" in nov else set(nov[str(s)]["novel_ids"]) for s in FRESH))
    res = dict(per_set={}, pooled=None, pooled_novel=None)
    T.append("\n## S2 · fresh-message placement robustness (fold-fit directions; message-realization robustness, NOT "
             "independent replication; shared contexts/mappings)\n")
    pooled = {c: {} for c in cfgs}
    for s in FRESH:
        Es = {c: E2[(s, c)] for c in cfgs if (s, c) in E2}
        for c, v in Es.items():
            pooled[c].update(v)
        r, t = stage_block(f"set m{s}", Es, None, cfgs, con)
        res["per_set"][s] = r
        T += t
    pooled = {c: v for c, v in pooled.items() if v}
    res["pooled"], t = stage_block("pooled over 5 sets (each context resampled with all its sets)", pooled, None, cfgs, con)
    T += t
    res["pooled_novel"], t = stage_block("pooled, NOVEL-message subset (data/fresh/analysis.json definition)", pooled, None,
                                         cfgs, con, subset=novel)
    T += t
    # original-message reference (same fold-fit protocol): unbundle cells + S1 PC2_L18 + core PC2_all
    orig = load_ref("data/unbundle/runs/unb_f*/metrics.json", r"^unb_f\d_")
    s1E = load_matrix("s1_7b", p_fold)
    if s1E:
        orig["PC2_late"] = s1E[0].get("PC2_L18", {})
    orig["PC2_all"] = load_ref("data/redund/7b/runs/redund_f*/metrics.json", r"^redund_f\d_").get("PC2_all", {})
    did = {}
    oid = lambda i: re.sub(r"^m\d+_", "", i)
    for name, w in con.items():
        if not all(c in pooled and orig.get(c) for c in w):
            continue
        rows = {}
        for i in set.intersection(*(set(pooled[c]) for c in w)):
            rows.setdefault(oid(i), []).append(sum(wt * pooled[c][i]["c1"] for c, wt in w.items()))
        ids = sorted(i for i in rows if all(i in orig[c] for c in w))
        v = np.array([np.mean(rows[i]) - sum(wt * orig[c][i]["c1"] for c, wt in w.items()) for i in ids])
        B = Boot(ids)
        did[name] = dict(n=len(ids), delta=float(v.mean()), ci=B.ci(ids, v),
                         orig=float(np.mean([sum(wt * orig[c][i]["c1"] for c, wt in w.items()) for i in ids])),
                         fresh_mean=float(np.mean([np.mean(rows[i]) for i in ids])))
    res["fresh_minus_original_c1"] = did
    T += ["\nFresh − original difference-in-differences (C1 strict; per original episode: mean over sets of the fresh contrast "
          "minus the original contrast; context-cluster CI):\n"]
    T += table([[k, f"{v['orig']:+.3f}", f"{v['fresh_mean']:+.3f}", fmt(v)] for k, v in did.items()],
               ["contrast", "original", "fresh (mean of sets)", "fresh − original"])
    # batching sensitivity: per-fold-batched none vs earlier 306-batch none (global-protocol run, direction-free)
    bs = {}
    for s in FRESH:
        old = episodes(json.load(open(ROOT / f"data/fresh/m{s}/runs/none/metrics.json")))
        new = E2.get((s, "none"), {})
        ids = sorted(set(old) & set(new))
        if ids:
            bs[s] = dict(n=len(ids), c1_identical=float(np.mean([old[i]["output"] == new[i]["output"] for i in ids])),
                         c1_new=float(np.mean([new[i]["c1"] for i in ids])), c1_old=float(np.mean([old[i]["c1"] for i in ids])))
    res["batching_none_vs_old_fresh"] = bs
    T.append("\nBatching sensitivity (none, per-fold batches vs earlier 306-episode batch): " +
             ", ".join(f"m{s} identical {v['c1_identical']:.3f}, C1 {v['c1_new']:.3f} vs {v['c1_old']:.3f}" for s, v in bs.items()))
    out["s2"] = res


def s3(out, T):
    L = load_matrix("s3_32b", p_fold)
    if L is None:
        out["s3"] = "missing"; T.append("\n## S3: not run\n"); return
    E, A, R = L
    cfgs = ["none", "M2_late", "M2_all", "EMB_LATE", "PRE_LATE"]
    con = vs_none(cfgs)
    con.update(FACT)
    res, t = stage_block("S3 · 32B embedding × pre-onset factorial, conditional on late removal (L42–63)", E, A, cfgs, con, R=R)
    ref = load_ref("data/redund/32b/runs/redund_f*/metrics.json", r"^redund_f\d_")
    res["reproduction"] = repro(E, ref, [("none", "none"), ("M2_late", "M2_late"), ("M2_all", "M2_all")])
    out["s3"] = res
    T.append("\n## S3 · Qwen2.5-32B factorial\n")
    T += t
    T.append("\nReproduction vs core matrix: " + ", ".join(f"{k} C1 {v['c1_identical']:.3f}/C6 {v['c6_identical']:.3f}"
                                                          if v else f"{k} n/a" for k, v in res["reproduction"].items()))


def s4(out, T):
    L = load_matrix("s4_14b", p_fold)
    if L is None:
        out["s4"] = "missing"; T.append("\n## S4: not run\n"); return
    E, A, R = L
    cfgs = ["none", "M2_all", "PC2_all", "M2_L32", "PC2_L32", "M2_L26", "PC2_L26"]
    con = vs_none(cfgs)
    con.update({"M2_all − PC2_all": {"M2_all": 1, "PC2_all": -1}, "L32 (frozen): M2 − PC2": {"M2_L32": 1, "PC2_L32": -1},
                "L26 (follow-up): M2 − PC2": {"M2_L26": 1, "PC2_L26": -1}, "M2_L26 − M2_L32": {"M2_L26": 1, "M2_L32": -1},
                "PC2_L26 − PC2_L32": {"PC2_L26": 1, "PC2_L32": -1}})
    res, t = stage_block("S4 · 14B (own frozen messages); L32 frozen cutoff, L26 follow-up motivated by prior onset", E, A,
                         cfgs, con, R=R)
    ref = load_ref("data_models/qwen14b/loso/runs/loso_f*/metrics.json", r"^loso_f\d_")
    res["reproduction"] = repro(E, ref, [("none", "none"), ("M2_all", "state_fold"), ("PC2_all", "pc2d_fold")])
    out["s4"] = res
    T.append("\n## S4 · Qwen2.5-14B placement and controls\n")
    T += t
    T.append("\nReproduction vs 14B LOSO: " + ", ".join(f"{k} C1 {v['c1_identical']:.3f}/C6 {v['c6_identical']:.3f}"
                                                        if v else f"{k} n/a" for k, v in res["reproduction"].items()))


def g4(out, T):
    cfgs = ["none", "M2f_all", "M2f_L18", "PC2f_all", "PC2f_L18"]
    con = vs_none(cfgs)
    con.update({"M2f_L18 − PC2f_L18": {"M2f_L18": 1, "PC2f_L18": -1}, "M2f_all − PC2f_all": {"M2f_all": 1, "PC2f_all": -1}})
    # 4a: fresh-refit directions on m505, paired with S2's m505 cells under the ORIGINAL fold-fit directions
    L = load_matrix("g4a_eval", p_fold)
    T.append("\n## G4a · fresh-message refit (fit m101–m404, eval m505 held-out schemes). Shared: contexts, mappings, "
             "templates, model; NOT independent\n")
    if L is None:
        out["g4a"] = "missing"; T.append("not run\n")
    else:
        E, A, R = L
        s2m = load_matrix("s2_fresh", p_fresh)
        if s2m:
            for c in ("none", "M2_all", "M2_late", "PC2_all", "PC2_late"):
                if (505, c) in s2m[0]:
                    E["orig_" + c] = s2m[0][(505, c)]
        con2 = dict(con)
        con2.update({"M2f_all − orig M2_all": {"M2f_all": 1, "orig_M2_all": -1},
                     "M2f_L18 − orig M2_late": {"M2f_L18": 1, "orig_M2_late": -1},
                     "PC2f_all − orig PC2_all": {"PC2f_all": 1, "orig_PC2_all": -1},
                     "PC2f_L18 − orig PC2_late": {"PC2f_L18": 1, "orig_PC2_late": -1}})
        allc = cfgs + [c for c in E if c.startswith("orig_")]
        res, t = stage_block("G4a", E, A, allc, con2, R=R)
        nov = json.load(open(ROOT / "data/fresh/analysis.json"))["overlap"]
        nv = set(nov["m505"]["novel_ids"] if "m505" in nov else nov["505"]["novel_ids"])
        res["novel"], t2 = stage_block("G4a novel-message subset", E, A, allc, con2, subset=nv)
        res["none_reproduction_vs_s2"] = repro(E, {"orig_none": E.get("orig_none", {})}, [("none", "orig_none")])
        out["g4a"] = res
        T += t + t2
        T.append("\nnone (G4a container) vs S2 m505 none: " + json.dumps(res["none_reproduction_vs_s2"]))
    # 4b: new contexts + new mappings; gate result always reported
    gate = V / "g4b/gate.json"
    T.append("\n## G4b · new contexts (seeds 100–116) + new frozen mappings. Shared: topic vocabulary, templates, task "
             "family, model, 9/18 (topic, state) pairs; NOT a new task or model\n")
    out["g4b"] = dict(gate=json.load(open(gate)) if gate.exists() else None)
    T.append("Feasibility gate (baseline C1 strict ≥ 0.70, reused from the original pipeline): " +
             (json.dumps(out["g4b"]["gate"]) if gate.exists() else "not run"))
    L = load_matrix("g4b_eval", p_fold)
    if L is not None:
        E, A, R = L
        res, t = stage_block("G4b", E, A, cfgs, con, R=R)
        out["g4b"].update(res)
        T += t


def validation(out, T):
    T.append("\n## Validation\n")
    res = {}
    for spec, hooked in (("v7b_dose", {"none": set(), "late": {f"L{l}" for l in range(18, 28)},
                                       "late_half": {f"L{l}" for l in range(18, 28)}}),
                         ("v32b", {"none": set(), "M2_late": {f"L{l}" for l in range(42, 64)},
                                   "M2_all": {"emb"} | {f"L{l}" for l in range(64)},
                                   "EMB_LATE": {"emb"} | {f"L{l}" for l in range(42, 64)},
                                   "PRE_LATE": {f"L{l}" for l in range(64)}})):
        f = V / f"validation/{spec}.json"
        if not f.exists():
            res[spec] = None
            continue
        C = json.load(open(f))["configs"]
        locs = sorted({r["loc"] for r in C["none"]["records"]})
        pre = lambda c: {r["loc"]: r for r in C[c]["records"] if r["phase"] == "prefill"}
        r0 = pre("none")
        rr = {}
        for c, hk in hooked.items():
            recs = C[c]["records"]
            full = c not in ("late_half",)
            hk_max = {ph: max([r["ratio"] for r in recs if r["loc"] in hk and r["phase"] == ph], default=None) for ph in ("prefill", "decode")}
            un_min = min([r["ratio"] for r in recs if r["loc"] not in hk], default=None)
            first = min((int(l[1:]) for l in hk if l != "emb"), default=None)
            up = [l for l in locs if l != "emb" and first is not None and int(l[1:]) < first] + (["emb"] if "emb" not in hk else [])
            if "emb" in hk:
                up = []  # everything downstream of the hooked embedding
            same = {l: (pre(c)[l]["s1"], pre(c)[l]["s2"]) == (r0[l]["s1"], r0[l]["s2"]) for l in up}
            d = dict(hooked_max_ratio=hk_max if full else None, unhooked_min_ratio=un_min,
                     upstream_bitwise_equal_to_none=f"{sum(same.values())}/{len(same)}",
                     decode_calls_per_location=min(sum(r["loc"] == l and r["phase"] == "decode" for r in recs) for l in locs))
            if c == "late_half":
                d["first_hooked_prefill_ratio_over_none"] = pre(c)["L18"]["ratio"] / r0["L18"]["ratio"]
            if c == "EMB_LATE":
                d["L0_41_differ_from_none"] = sum((pre(c)[f"L{l}"]["s1"], pre(c)[f"L{l}"]["s2"]) != (r0[f"L{l}"]["s1"], r0[f"L{l}"]["s2"])
                                                  for l in range(42))
            rr[c] = d
        res[spec] = rr
        T.append(f"- {spec}: " + json.dumps(rr))
    out["validation"] = res


def main():
    out, T = {}, ["# Verification extension results (exp/verify_analyze.py; plan notes/verify_plan.md)\n",
                  "Strict decode, exact bet, malformed and capability controls are reported together. Contrast CIs: "
                  "context-cluster bootstrap stratified by scheme (episode-level CI shown for continuity); ARC: item bootstrap. "
                  "Continuous effects only; no thresholds; CIs including 0 are not equivalence.\n"]
    validation(out, T)
    for f in (s1, s2, s3, s4, g4):
        f(out, T)
    json.dump(out, open(V / "analysis.json", "w"), indent=1, default=float)
    (V / "tables.md").write_text("\n".join(T) + "\n")
    print("wrote", V / "tables.md")


if __name__ == "__main__":
    main()
