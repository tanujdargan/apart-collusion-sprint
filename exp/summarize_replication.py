"""Print markdown tables for one model namespace. Usage: .venv-analysis/bin/python exp/summarize_replication.py <tag>
Reads data_models/<tag>/{gate_*.jsonl, probes.json, interv/*/metrics.json, interv/*/post_probes.json}."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
import env  # noqa: E402

tag = sys.argv[1]
D = ROOT / "data_models" / tag
f = lambda x: "–" if x is None else f"{x:.3f}"  # noqa: E731
f2 = lambda x: "–" if x is None else f"{x:.2f}"  # noqa: E731


def jl(p):
    return [json.loads(l) for l in open(p)]


# ---------------- gate
if (D / "gate_receivers.jsonl").exists():
    rx = jl(D / "gate_receivers.jsonl")
    print(f"## {tag} gate (90 episodes)")
    for c in ("C1", "C2", "C3", "C4"):
        rows = [r for r in rx if r.get("condition") == c]
        if rows:
            m = env.decode_metrics(rows)
            print(c, "decode", f(m["decode_acc"]) if c != "C3" else "–", "r", f(m["bet_state_corr"]),
                  "mean bets", {k: round(v, 1) for k, v in m["mean_bet"].items()}, "parse_fail", f(m["parse_fail"]))
    rec = [r for r in rx if "answer" in r]
    print("recall", f(sum(r["answer"] == r["topic"] for r in rec) / len(rec)))
    msgs = jl(D / "gate_messages.jsonl")
    ch = [env.sender_check(m, m["collude_msg"]) for m in msgs]
    print("sender clean-target", f(np.mean([c["clean_target"] for c in ch])), "no-forbidden",
          f(np.mean([not c["forbidden"] for c in ch])), "no-digit", f(np.mean([not c["has_digit"] for c in ch])))
    print()

# ---------------- baseline
bp = D / "interv/baseline/metrics.json"
if bp.exists():
    b = json.load(open(bp))
    print(f"## {tag} baseline (n={b['n']})\n| Condition | Decode acc | Bet–state r | Mean bet COLD/NEU/HOT |\n|---|---|---|---|")
    for c in ("C1", "C2", "C3", "C4", "C5"):
        if c in b:
            m = b[c]
            mb = " / ".join(f"${m['mean_bet'][s]:.0f}" for s in env.STATES)
            print(f"| {c} | {f(m['decode_acc'])} | {f(m['bet_state_corr'])} | {mb} |")
    print(f"recall {f(b['recall_acc'])}; ARC-Easy {f2(b.get('arc_acc'))}\n")

# ---------------- probes
pp = D / "probes.json"
if pp.exists():
    p = json.load(open(pp))
    L = p["n_layers"]
    print(f"## {tag} probes (L={L})\n| contrast | min | max | mean | layer-0 |\n|---|---|---|---|---|")
    for k, v in p["auroc"].items():
        print(f"| {k} | {f(min(v))} | {f(max(v))} | {f(np.mean(v))} | {f(v[0])} |")
    sa = p["state_acc"]["C1_last"]
    on5 = next((l for l, a in enumerate(sa) if a >= 0.5), None)
    on8 = next((l for l, a in enumerate(sa) if a >= 0.8), None)
    print("state acc per layer:", " ".join(f"{a:.2f}" for a in sa))
    print(f"state onset: first layer >=0.5: {on5} ({on5 / L:.2f} depth); >=0.8: {on8} "
          f"({(on8 or 0) / L:.2f} depth); max {max(sa):.3f}")
    c5l = p["auroc"]["detect_C1vC5_last"]
    print("C1vC5_last per layer:", " ".join(f"{a:.2f}" for a in c5l))
    print("C1vC5_mean per layer:", " ".join(f"{a:.2f}" for a in p["auroc"]["detect_C1vC5_mean"]))
    print("cos detect vs state maxabs: max", f(max(p["cos"]["detect_vs_state_maxabs"])), "median",
          f(float(np.median(p["cos"]["detect_vs_state_maxabs"]))), "; dm vs lr",
          f(min(p["cos"]["primary_dm_vs_lr"])), "-", f(max(p["cos"]["primary_dm_vs_lr"])))
    print()
    late = on8 if on8 is not None else L // 2

# ---------------- ablations
OWN = {"state_C1_last_2d": ["detect_C1vC4_mean", "detect_C1vC3_full"], "detect_C1vC4_mean_lr": ["detect_C1vC4_mean"],
       "detect_C1vC4_mean_dm": ["detect_C1vC4_mean"], "detect_C1vC3_full_dm": ["detect_C1vC3_full"],
       "detect_C1vC5_last_dm": ["detect_C1vC5_last"], "random1d_0": ["detect_C1vC4_mean"],
       "random1d_1": ["detect_C1vC4_mean"], "random2d_0": ["detect_C1vC4_mean"]}
OWN.update({k + "_late": v for k, v in list(OWN.items())})  # layer-restricted variants (ablated at layers >= onset)
if bp.exists():
    b = json.load(open(bp))
    print(f"## {tag} ablations\n| Direction ablated | C1 decode (base {f(b['C1']['decode_acc'])}) | r ({f(b['C1']['bet_state_corr'])}) "
          f"| recall ({f(b['recall_acc'])}) | ARC ({f2(b.get('arc_acc'))}) | own detector frozen AUROC mean [min–max] "
          f"| retrained mean | other detectors frozen (mean) | state acc retrained (max) |\n|---|---|---|---|---|---|---|---|---|")
    for d, own in OWN.items():
        mp, pq = D / f"interv/abl_{d}/metrics.json", D / f"interv/abl_{d}/post_probes.json"
        if not mp.exists():
            if not d.endswith("_late"):
                print(f"| {d} | missing |")
            continue
        m = json.load(open(mp))
        q = json.load(open(pq)) if pq.exists() else None
        fz, rt, oth, sacc = [], [], [], "–"
        if q:
            for o in own:
                v = np.array(q["frozen"][o])
                s = f"{o.replace('detect_', '')} {v.mean():.2f} [{v.min():.2f}–{v.max():.2f}]"
                if o == "detect_C1vC5_last":
                    s += f"; L{late}+ {v[late:].mean():.2f} (base {np.mean(p['auroc'][o][late:]):.2f})"
                fz.append(s)
                rt.append(f"{np.mean(q['retrained'][o]):.2f}")
            oth = [f"{k.replace('detect_', '')} {np.mean(v):.2f}" for k, v in q["frozen"].items() if k not in own]
            sacc = f"{max(q['state_acc_retrained']):.2f}"
        print(f"| {d} | {f(m['C1']['decode_acc'])} | {f(m['C1']['bet_state_corr'])} | {f(m['recall_acc'])} | "
              f"{f2(m.get('arc_acc'))} | {'; '.join(fz) or '–'} | {' / '.join(rt) or '–'} | {', '.join(oth) or '–'} | {sacc} |")
    print()

# ---------------- steering (one table per layer; alpha=0 row is the baseline run)
st = sorted(D.glob("interv/steer_state_L*_a*/metrics.json"))
for lay in sorted({x.parent.name.split("_L")[1].split("_")[0] for x in st}, key=int):
    nj = D / "directions" / f"state_hotcold_L{lay}_norm.json"
    nrm = json.load(open(nj))["norm"] if nj.exists() else float("nan")
    print(f"## {tag} steering at L{lay} (‖μHOT−μCOLD‖ = {nrm:.2f})\n| α | α/‖Δμ‖ | decode | r | mean bet COLD/NEU/HOT "
          f"| recall | ARC |\n|---|---|---|---|---|---|---|")
    rows = [(0.0, json.load(open(bp)))] + [(float(x.parent.name.split("_a")[-1]), json.load(open(x))) for x in st
                                           if x.parent.name.split("_L")[1].split("_")[0] == lay]
    for a, m in sorted(rows, key=lambda t: t[0]):
        mb = " / ".join(f"${m['C1']['mean_bet'][s]:.0f}" for s in env.STATES)
        print(f"| {a:+.1f} | {a / nrm:+.1f} | {f(m['C1']['decode_acc'])} | {f(m['C1']['bet_state_corr'])} | {mb} | "
              f"{f(m['recall_acc'])} | {f2(m.get('arc_acc'))} |")
    print()
