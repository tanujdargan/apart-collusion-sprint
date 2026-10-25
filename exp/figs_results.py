"""Fig 4 (headline: detector AUROC vs collusion success under each ablation) and Fig 5 (steering dose-response).
  .venv-analysis/bin/python exp/figs_results.py   -> figures/fig4_2x2.{png,pdf}, figures/fig5_steer.{png,pdf},
                                                     data/results_table.csv"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IV, FIG = ROOT / "data/interv", ROOT / "figures"
C = dict(blue="#0072B2", orange="#D55E00", green="#009E73", grey="#999999", purple="#CC79A7", sky="#56B4E9", black="#000000")
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
LATE = slice(18, None)  # C1vC5 detectors: use layers >= 18 (early layers carry a transcript-identity artefact, §4.4)

# tag -> (label, own detector key, layer slice, colour, marker)
RUNS = {
    "abl_random1d_0": ("random 1-D", "detect_C1vC4_mean", slice(None), C["grey"], "o"),
    "abl_random1d_1": ("random 1-D", "detect_C1vC4_mean", slice(None), C["grey"], "o"),
    "abl_random1d_2": ("random 1-D", "detect_C1vC4_mean", slice(None), C["grey"], "o"),
    "abl_random2d_0": ("random 2-D", "detect_C1vC4_mean", slice(None), C["grey"], "s"),
    "abl_random2d_1": ("random 2-D", "detect_C1vC4_mean", slice(None), C["grey"], "s"),
    "abl_random2d_2": ("random 2-D", "detect_C1vC4_mean", slice(None), C["grey"], "s"),
    "abl_state_C1_last_2d": ("covert-state subspace", "detect_C1vC4_mean", slice(None), C["green"], "D"),
    "abl_detect_C1vC4_mean_lr": ("detector C1vC4 (LR)", "detect_C1vC4_mean", slice(None), C["blue"], "^"),
    "abl_detect_C1vC4_mean_dm": ("detector C1vC4 (diff-means)", "detect_C1vC4_mean", slice(None), C["sky"], "^"),
    "abl_detect_C1vC2_mean_dm": ("detector C1vC2 (diff-means)", "detect_C1vC2_mean", slice(None), C["sky"], "v"),
    "abl_detect_C1vC3_full_dm": ("NARCBench-style C1vC3", "detect_C1vC3_full", slice(None), C["purple"], "P"),
    "abl_detect_C1vC5_last_dm": ("C1vC5 decision token", "detect_C1vC5_last", LATE, C["orange"], "X"),
    "abl_detect_C1vC5_mean_dm": ("C1vC5 msg span", "detect_C1vC5_mean", slice(5, 20), C["orange"], "*"),
}


def wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def table():
    base = json.load(open(ROOT / "data/probes.json"))["auroc"]
    rows = []
    for tag, (lab, key, sl, col, mk) in RUNS.items():
        m, p = json.load(open(IV / tag / "metrics.json")), json.load(open(IV / tag / "post_probes.json"))
        rows.append(dict(tag=tag, label=lab, detector=key, colour=col, marker=mk, n=m["C1"]["n"],
                         decode=m["C1"]["decode_acc"], r=m["C1"]["bet_state_corr"], recall=m["recall_acc"],
                         arc=m["arc_acc"], det_base=float(np.mean(base[key][sl])),
                         det_frozen=float(np.mean(p["frozen"][key][sl])),
                         det_retrained=float(np.mean(p["retrained"][key][sl])) if key in p["retrained"] else None))
    b = json.load(open(IV / "baseline/metrics.json"))
    rows.append(dict(tag="baseline", label="no intervention", detector="detect_C1vC4_mean", colour=C["black"], marker="o",
                     n=b["C1"]["n"], decode=b["C1"]["decode_acc"], r=b["C1"]["bet_state_corr"], recall=b["recall_acc"],
                     arc=b["arc_acc"], det_base=float(np.mean(base["detect_C1vC4_mean"])),
                     det_frozen=float(np.mean(base["detect_C1vC4_mean"])), det_retrained=None))
    with open(ROOT / "data/results_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


# Fig 4 spec: tag -> (label, colour, marker, run holding C1, run holding C6)
SPEC = [
    ("none", C["black"], "o", "baseline", "base_C6"),
    ("random 1-D", C["grey"], "o", "abl_random1d_0", "abl_random1d_0_C6"),
    ("random 2-D", C["grey"], "s", "abl_random2d_0", "abl_random2d_0_C6"),
    ("PC control 1-D (8% var)", C["grey"], "v", "abl_pc_ctrl_1d", "abl_pc_ctrl_1d"),
    ("PC control 2-D (19% var)", C["grey"], "D", "abl_pc_ctrl_2d", "abl_pc_ctrl_2d"),
    ("decision-state 2-D", C["green"], "D", "abl_state_C1_last_2d", "abl_state_C1_last_2d_C6"),
    ("detector C1vC4 diff-means", C["blue"], "^", "abl_detect_C1vC4_mean_dm", "abl_detect_C1vC4_mean_dm_C6"),
    ("detector C1vC2 diff-means", C["sky"], "^", "abl_detect_C1vC2_mean_dm", "abl_detect_C1vC2_mean_dm_C6"),
    ("detector C1vC4 LR", C["sky"], "v", "abl_detect_C1vC4_mean_lr", "abl_detect_C1vC4_mean_lr_C6"),
    ("detector C1vC4 raw LR", C["sky"], "<", "abl_detect_C1vC4_mean_lrraw", "abl_detect_C1vC4_mean_lrraw_C6"),
    ("NARCBench-style C1vC3", C["purple"], "P", "abl_detect_C1vC3_full_dm", "abl_detect_C1vC3_full_dm_C6"),
    ("signal-for-me C1vC5 (decision)", C["orange"], "X", "abl_detect_C1vC5_last_dm", "abl_detect_C1vC5_last_dm_C6"),
    ("signal-for-me C1vC5 (msg span)", C["orange"], "*", "abl_detect_C1vC5_mean_dm", "abl_detect_C1vC5_mean_dm_C6"),
]
DET = [("C1vC4 LR", "abl_detect_C1vC4_mean_lr", "detect_C1vC4_mean", slice(None)),
       ("C1vC4 raw LR", "abl_detect_C1vC4_mean_lrraw", "detect_C1vC4_mean", slice(None)),
       ("C1vC4 diff-means", "abl_detect_C1vC4_mean_dm", "detect_C1vC4_mean", slice(None)),
       ("C1vC2 diff-means", "abl_detect_C1vC2_mean_dm", "detect_C1vC2_mean", slice(None)),
       ("NARCBench C1vC3", "abl_detect_C1vC3_full_dm", "detect_C1vC3_full", slice(None)),
       ("C1vC5 decision", "abl_detect_C1vC5_last_dm", "detect_C1vC5_last", slice(18, None))]


def fig4(rows=None):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw=dict(width_ratios=[1.25, 1]))
    a.axhspan(0.30, 0.37, color="#eeeeee", zorder=0)
    a.text(0.34, 0.335, "chance", fontsize=6, va="center", color="#666666")
    spec_rows = []
    for lab, col, mk, t1, t6 in SPEC:
        m1, m6 = json.load(open(IV / t1 / "metrics.json")), json.load(open(IV / t6 / "metrics.json"))
        y, x, n = m1["C1"]["decode_acc"], m6["C6"]["bet_exact_acc"], m1["C1"]["n"]
        lo, hi = wilson(y, n)
        a.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt=mk, color=col, ms=6 if mk != "*" else 9, mec="k", mew=0.4,
                   elinewidth=0.6, capsize=1.5, label=lab, zorder=3)
        spec_rows.append(dict(label=lab, C1_decode=y, C6_explicit_bet_acc=x, arc=m1.get("arc_acc"), recall=m1.get("recall_acc")))
    kw = dict(fontsize=6.5, color="#444444", style="italic")
    a.text(1.003, 0.235, "breaks covert decoding,\nspares explicit decision", ha="right", **kw)
    a.text(0.34, 1.0, "no effect: top right", **kw)
    a.set(xlim=(0.33, 1.02), ylim=(0.2, 1.03), xlabel="explicit-state decision intact (C6 bet acc.)",
          ylabel="covert collusion (C1 decode acc.)", title="(a) Which directions carry the collusion?")
    a.legend(fontsize=5, frameon=False, loc="upper left", bbox_to_anchor=(-0.02, -0.2), ncol=2)
    labs, dec, ret = [], [], []
    for lab, tag, key, sl in DET:
        m, p = json.load(open(IV / tag / "metrics.json")), json.load(open(IV / tag / "post_probes.json"))
        labs.append(lab); dec.append(m["C1"]["decode_acc"]); ret.append(float(np.mean(p["retrained"][key][sl])))
    yy = np.arange(len(labs))
    b.barh(yy + 0.2, dec, 0.38, color=C["orange"], label="collusion (C1 decode)")
    b.barh(yy - 0.2, ret, 0.38, color=C["blue"], label="detector AUROC, retrained after ablation")
    b.set_yticks(yy); b.set_yticklabels(labs, fontsize=6.5); b.invert_yaxis()
    b.axvline(0.958, color=C["orange"], lw=0.6, ls=":")
    b.set(xlim=(0, 1.05), xlabel="value after ablating that detector's own direction", title="(b) Detection survives; collusion mostly does too")
    b.legend(fontsize=5.5, frameon=False, loc="upper left", bbox_to_anchor=(-0.05, -0.2))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig4_2x2.{ext}", dpi=300, bbox_inches="tight")
    with open(ROOT / "data/fig4_specificity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(spec_rows[0])); w.writeheader(); w.writerows(spec_rows)


def fig5():
    al = [-40, -20, -10, 0, 10, 20, 40]
    M = [json.load(open(IV / ("baseline" if a == 0 else f"steer_state_L20_a{a}") / "metrics.json")) for a in al]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.3))
    for s, c in (("COLD", C["blue"]), ("NEUTRAL", C["grey"]), ("HOT", C["orange"])):
        a.plot(al, [m["C1"]["mean_bet"][s] for m in M], "-o", color=c, ms=3, label=f"true state {s}")
    rc = [json.load(open(IV / f"steer_random_L20_a{x}" / "metrics.json"))["C1"]["mean_bet"]["COLD"] for x in (-40, 40)]
    a.plot([-40, 40], rc, "x", color=C["blue"], ms=5, mew=1.2, label="COLD, random direction")
    a.set(xlabel="steering α along HOT−COLD (layer 20)", ylabel="receiver mean bet ($)", title="(a) Covert decision")
    a.legend(fontsize=6, frameon=False)
    b.plot(al, [m["C1"]["decode_acc"] for m in M], "-o", color=C["black"], ms=3, label="decode acc.")
    b.plot(al, [m["recall_acc"] for m in M], "--s", color=C["green"], ms=3, label="topic recall")
    b.plot(al, [m["arc_acc"] for m in M], ":^", color=C["purple"], ms=3, label="ARC-Easy")
    b.set(xlabel="steering α", ylabel="accuracy", ylim=(0.5, 1.02), title="(b) Collusion vs utility")
    b.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig5_steer.{ext}", dpi=300)


if __name__ == "__main__":
    rows = table()
    for r in rows:
        print(f"{r['tag']:28s} decode {r['decode']:.3f} det {r['det_base']:.2f}->{r['det_frozen']:.2f} arc {r['arc']}")
    fig4(rows)
    fig5()
