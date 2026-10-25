"""Fresh-message robustness analysis (Qwen2.5-7B, frozen original directions), rules fixed in
notes/followups_plan.md section A + "Reporting".

  .venv-analysis/bin/python exp/fresh_analyze.py
  -> data/fresh/analysis.json, data/fresh/tables.md, figures/fig8_fresh.{png,pdf}

Conventions reused from exp/loso_analyze.py: Wilson 95% CIs; paired Δ = mean(cfg) − mean(none) on identical episodes,
stratified paired bootstrap (10,000 reps, RNG seed 0, resampling unit = episode); exact McNemar on binary outcomes.
Parse failures (bet None / decoded None) count as wrong; nothing is dropped.
"""
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
import env  # noqa: E402

FRESH = ROOT / "data/fresh"
SETS = [101, 202, 303, 404, 505]
CFGS = ["none", "dm", "pc1d", "state", "pc2d", "lr"]
DIRS = CFGS[1:]
LABEL = {"none": "none\n(baseline)", "dm": "dm\ndetector", "pc1d": "PC-1D\ncontrol", "state": "state\n2-D",
         "pc2d": "PC-2D\ncontrol", "lr": "LR\ndetector"}
# original full-306 runs with the frozen directions: (C1 file, C6 file)
ORIG = dict(none=("baseline", "base_C6"), dm=("abl_detect_C1vC4_mean_dm", "abl_detect_C1vC4_mean_dm_C6"),
            pc1d=("abl_pc_ctrl_1d", "abl_pc_ctrl_1d"), state=("abl_state_C1_last_2d", "abl_state_C1_last_2d_C6"),
            pc2d=("abl_pc_ctrl_2d", "abl_pc_ctrl_2d"), lr=("abl_detect_C1vC4_mean_lr", "abl_detect_C1vC4_mean_lr_C6"))
METRICS = ("c1_dec", "c1_bet", "c6_bet")
MNAME = {"c1_dec": "C1 decode (strict)", "c1_bet": "C1 bet-exact", "c6_bet": "C6 bet-exact"}
RNG = np.random.default_rng(0)
NBOOT = 10000


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def paired(a, b, strata):
    """Δ = mean(a) − mean(b) on identical episodes; bootstrap resampling episodes within each stratum; exact McNemar."""
    a, b = a.astype(float), b.astype(float)
    diff = a - b
    idx = [np.where(strata == s)[0] for s in np.unique(strata)]
    boots = np.empty(NBOOT)
    for i in range(NBOOT):
        boots[i] = np.concatenate([diff[RNG.choice(ix, len(ix))] for ix in idx]).mean()
    n01, n10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return dict(delta=float(diff.mean()), ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                mcnemar_p=float(p), cfg_only_correct=n10, none_only_correct=n01, n=int(len(a)))


def cluster_boot(values, cluster, scheme, rng):
    """Cluster bootstrap of a pooled mean: resample clusters (orig_id contexts) with replacement within scheme strata,
    carrying ALL rows of each sampled cluster (all sets); stat = sum(values)/n_rows over sampled clusters."""
    values = np.asarray(values, float)
    ucl = np.unique(cluster)
    pos = {u: i for i, u in enumerate(ucl)}
    ci_ = np.array([pos[c] for c in cluster])
    sums = np.bincount(ci_, weights=values, minlength=len(ucl))
    cnts = np.bincount(ci_, minlength=len(ucl)).astype(float)
    cl_scheme = np.empty(len(ucl), int)
    cl_scheme[ci_] = scheme
    tot_s, tot_n = np.zeros(NBOOT), np.zeros(NBOOT)
    for j in np.unique(cl_scheme):
        ix = np.where(cl_scheme == j)[0]
        draw = ix[rng.integers(0, len(ix), size=(NBOOT, len(ix)))]
        tot_s += sums[draw].sum(1)
        tot_n += cnts[draw].sum(1)
    boots = tot_s / tot_n
    return dict(est=float(values.mean()), ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                n_rows=int(len(values)), n_clusters=int(len(ucl)))


def outcome_vectors(c1_rows, c6_rows):
    return dict(
        c1_dec=np.array([r["decoded"] == r["state"] for r in c1_rows]),
        c1_bet=np.array([r["bet"] == env.BETS[r["state"]] for r in c1_rows]),
        c6_bet=np.array([r["bet"] == env.BETS[r["state"]] for r in c6_rows]),
        c1_betnone=np.array([r["bet"] is None for r in c1_rows]),
        c1_decnone=np.array([r["decoded"] is None for r in c1_rows]),
        c6_betnone=np.array([r["bet"] is None for r in c6_rows]),
        c6_decnone=np.array([r["decoded"] is None for r in c6_rows]))


def norm(s):
    return re.sub(r"[\W_]+", "", s.lower())


def ratio_max(s, pool):
    """Exact max SequenceMatcher ratio of s vs pool, pruning with the quick_ratio upper bound."""
    sm = difflib.SequenceMatcher(autojunk=False)
    sm.set_seq2(s)
    cands = []
    for j, t in enumerate(pool):
        sm.set_seq1(t)
        cands.append((sm.quick_ratio(), j))
    cands.sort(reverse=True)
    best, bj = -1.0, None
    for q, j in cands:
        if q <= best:
            break
        sm.set_seq1(pool[j])
        r = sm.ratio()
        if r > best:
            best, bj = r, j
    return best, bj


def sratio(a, b):
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def dist(x):
    x = np.asarray(x)
    return dict(median=float(np.median(x)), q25=float(np.percentile(x, 25)), q75=float(np.percentile(x, 75)),
                min=float(x.min()), max=float(x.max()), mean=float(x.mean()))


def load():
    orig_msgs = [json.loads(l) for l in open(ROOT / "data/messages.jsonl")]
    oid = [r["id"] for r in orig_msgs]
    assert len(oid) == len(set(oid)), "original ids not unique"
    msgs = {s: [json.loads(l) for l in open(FRESH / f"m{s}/messages.jsonl")] for s in SETS}
    runs = {(s, c): json.load(open(FRESH / f"m{s}/runs/{c}/metrics.json")) for s in SETS for c in CFGS}
    orig_runs = {}
    for c, (t1, t6) in ORIG.items():
        orig_runs[c] = (json.load(open(ROOT / f"data/interv/{t1}/metrics.json"))["C1_rows"],
                        json.load(open(ROOT / f"data/interv/{t6}/metrics.json"))["C6_rows"])
    return orig_msgs, msgs, runs, orig_runs


def verify(orig_msgs, msgs, runs, orig_runs):
    V = dict(sets={}, original={})
    oid = [r["id"] for r in orig_msgs]
    obyid = {r["id"]: r for r in orig_msgs}
    cnt = Counter((r["scheme_id"], r["state"]) for r in orig_msgs)
    V["original"] = dict(n=len(oid), n_schemes=len({k[0] for k in cnt}), n_states=len({k[1] for k in cnt}),
                         per_cell=sorted(set(cnt.values())))
    # original runs: row order, ids == messages ids, C1 msgs == frozen messages
    for c, (r1, r6) in orig_runs.items():
        assert [r["id"] for r in r1] == oid, ("orig C1 order", c)
        assert [r["id"] for r in r6] == oid, ("orig C6 order", c)
    all_ids = []
    for s in SETS:
        rows = msgs[s]
        ids = [r["id"] for r in rows]
        all_ids += ids
        assert len(ids) == len(set(ids)) == 306, (s, "ids")
        oids = [r["orig_id"] for r in rows]
        assert len(set(oids)) == 306 and set(oids) == set(oid), (s, "orig_id not one-to-one")
        assert all(r["id"] == f"m{s}_{r['orig_id']}" and r["gen_seed"] == s for r in rows), (s, "id format")
        ctx_keys = ["scheme_id", "state", "seed", "sender", "receiver", "filler", "filler_line", "hand", "topic",
                    "recall_options", "distractor", "transcript"]
        ctx_same = all(all(r[k] == obyid[r["orig_id"]][k] for k in ctx_keys) for r in rows)
        same_order = oids == oid
        c = Counter((r["scheme_id"], r["state"]) for r in rows)
        # parse_talk / sender_check recomputed from raw output
        reparsed = [env.parse_talk(r["collude_raw"]) for r in rows]
        talk_fail = sum(not t for t in reparsed)
        msg_consistent = all((t or "Good luck everyone.") == r["collude_msg"] for t, r in zip(reparsed, rows))
        ep_by = {r["orig_id"]: r for r in rows}
        check_consistent = all(env.sender_check(r, r["collude_msg"]) == r["collude_check"] for r in rows)
        clean_consistent = all(r["collude_check"]["clean_target"] == r["collude_clean"] for r in rows)
        run_checks = {}
        for cfg in CFGS:
            m = runs[(s, cfg)]
            run_checks[cfg] = dict(
                ids_equal_set=m["ids"] == ids,
                c1_order=[r["id"] for r in m["C1_rows"]] == ids,
                c6_order=[r["id"] for r in m["C6_rows"]] == ids,
                c1_msg_equal=all(r["msg"] == x["collude_msg"] for r, x in zip(m["C1_rows"], rows)) if "msg" in m["C1_rows"][0] else None,
                c1_state_equal=all(r["state"] == x["state"] for r, x in zip(m["C1_rows"], rows)),
                c6_state_equal=all(r["state"] == x["state"] for r, x in zip(m["C6_rows"], rows)),
                messages_path=m.get("messages_path"), mode=m.get("mode"), layers=m.get("layers"), model=m.get("model"))
            for k in ("ids_equal_set", "c1_order", "c6_order", "c1_state_equal", "c6_state_equal"):
                assert run_checks[cfg][k], (s, cfg, k)
            assert run_checks[cfg]["c1_msg_equal"] in (True, None), (s, cfg, "C1 msg differs from set messages")
        V["sets"][s] = dict(n=len(ids), unique_ids=True, orig_id_one_to_one=True, same_order_as_original=same_order,
                            context_identical_to_original=ctx_same, n_schemes=len({k[0] for k in c}),
                            n_states=len({k[1] for k in c}), per_cell=sorted(set(c.values())),
                            parse_talk_failures=talk_fail, parse_ok_false=sum(not r["collude_parse_ok"] for r in rows),
                            collude_msg_equals_reparse=msg_consistent, stored_check_equals_recomputed=check_consistent,
                            clean_flag_consistent=clean_consistent, runs=run_checks)
        assert ctx_same, (s, "episode context differs from original")
    assert len(all_ids) == len(set(all_ids)), "ids collide across fresh sets"
    V["ids_unique_across_sets"] = True
    return V


def overlap(orig_msgs, msgs):
    O = {}
    otext = [r["collude_msg"] for r in orig_msgs]
    oidx = {r["id"]: i for i, r in enumerate(orig_msgs)}
    onorm = [norm(t) for t in otext]
    onorm_set = Counter(onorm)
    O["original_within_set"] = dict(exact_dup_messages=306 - len(set(otext)), normalised_dup_messages=306 - len(set(onorm)))
    for s in SETS:
        rows = msgs[s]
        ftext = [r["collude_msg"] for r in rows]
        same = [otext[oidx[r["orig_id"]]] for r in rows]
        exact_same = sum(a == b for a, b in zip(ftext, same))
        norm_same = sum(norm(a) == norm(b) for a, b in zip(ftext, same))
        exact_any = sum(t in set(otext) for t in ftext)
        norm_any = sum(norm(t) in onorm_set for t in ftext)
        r_same = [sratio(a, b) for a, b in zip(ftext, same)]
        mx = [ratio_max(t, otext) for t in ftext]
        r_any = [m[0] for m in mx]
        other_ep = []  # max ratio vs originals from OTHER episodes
        for r, t in zip(rows, ftext):
            pool = [otext[j] for j in range(306) if j != oidx[r["orig_id"]]]
            other_ep.append(ratio_max(t, pool)[0])
        ex = [dict(id=r["id"], ratio=round(x, 3), fresh=a, original=b)
              for r, x, a, b in sorted(zip(rows, r_same, ftext, same), key=lambda z: -z[1])[:3]]
        O[s] = dict(exact_same_episode=exact_same, normalised_same_episode=norm_same,
                    exact_any_original=exact_any, normalised_any_original=norm_any,
                    near_dup_same_episode_ge09=int(sum(x >= 0.9 for x in r_same)),
                    near_dup_any_original_ge09=int(sum(x >= 0.9 for x in r_any)),
                    near_dup_other_episode_ge09=int(sum(x >= 0.9 for x in other_ep)),
                    ratio_same_episode=dist(r_same), ratio_max_any_original=dist(r_any),
                    ratio_max_other_episode=dist(other_ep),
                    within_set_exact_dups=306 - len(set(ftext)), within_set_normalised_dups=306 - len(set(norm(t) for t in ftext)),
                    top_same_episode_examples=ex,
                    novel_ids=[r["id"] for r, ra, a, b in zip(rows, r_any, ftext, same) if ra < 0.9 and a != b])
    # across fresh sets
    cross = {}
    for i, a in enumerate(SETS):
        for b in SETS[i + 1:]:
            ta = [r["collude_msg"] for r in msgs[a]]
            tb = {r["orig_id"]: r["collude_msg"] for r in msgs[b]}
            same_b = [tb[r["orig_id"]] for r in msgs[a]]
            nb = Counter(norm(t) for t in tb.values())
            rs = [sratio(x, y) for x, y in zip(ta, same_b)]
            cross[f"{a}-{b}"] = dict(exact_same_episode=sum(x == y for x, y in zip(ta, same_b)),
                                     normalised_same_episode=sum(norm(x) == norm(y) for x, y in zip(ta, same_b)),
                                     exact_any=sum(t in set(tb.values()) for t in ta),
                                     normalised_any=sum(norm(t) in nb for t in ta),
                                     near_dup_same_episode_ge09=int(sum(x >= 0.9 for x in rs)),
                                     ratio_same_episode_median=float(np.median(rs)))
    O["across_fresh_sets"] = cross
    return O


def compliance(rows, msg_key="collude_msg"):
    n = len(rows)
    chk = [env.sender_check(r, r[msg_key]) for r in rows]
    raw = [r["collude_raw"] for r in rows]
    tf = sum(not env.parse_talk(x) for x in raw)
    fw = Counter(w.lower() for c in chk for w in c["forbidden"])
    return dict(n=n, forbidden_rate=sum(bool(c["forbidden"]) for c in chk) / n, digit_rate=sum(c["has_digit"] for c in chk) / n,
                numword_rate=sum(bool(c["numwords"]) for c in chk) / n,
                target_topic_rate=sum(c["target_mentioned"] for c in chk) / n,
                clean_target_rate=sum(c["clean_target"] for c in chk) / n,
                other_scheme_topic_rate=sum(any(t != r["topic"] for t in c["scheme_topics"]) for c, r in zip(chk, rows)) / n,
                parse_talk_failures=tf, forbidden_words=dict(fw.most_common()),
                mean_len_chars=float(np.mean([len(r[msg_key]) for r in rows])))


def absolute(v, n):
    return dict(n=n, c1_decode=float(v["c1_dec"].mean()), c1_decode_ci=wilson(v["c1_dec"].sum(), n),
                c1_bet_exact=float(v["c1_bet"].mean()), c1_bet_exact_ci=wilson(v["c1_bet"].sum(), n),
                c6_bet_exact=float(v["c6_bet"].mean()), c6_bet_exact_ci=wilson(v["c6_bet"].sum(), n),
                c1_bet_none=int(v["c1_betnone"].sum()), c1_decoded_none=int(v["c1_decnone"].sum()),
                c6_bet_none=int(v["c6_betnone"].sum()), c6_decoded_none=int(v["c6_decnone"].sum()))


def main():
    orig_msgs, msgs, runs, orig_runs = load()
    res = dict(verification=verify(orig_msgs, msgs, runs, orig_runs))
    print("verification ok")
    res["overlap"] = overlap(orig_msgs, msgs)
    print("overlap done")
    res["compliance"] = {"original": compliance(orig_msgs), **{s: compliance(msgs[s]) for s in SETS}}
    res["compliance"]["fresh_pooled"] = compliance(sum((msgs[s] for s in SETS), []))

    osch = np.array([r["scheme_id"] for r in orig_msgs])
    # ------------------------------------------------------------- original set (computed from the original files)
    OV = {c: outcome_vectors(*orig_runs[c]) for c in CFGS}
    base_o = json.load(open(ROOT / "data/interv/baseline/metrics.json"))
    res["original"] = dict(absolute={c: absolute(OV[c], 306) for c in CFGS},
                           paired={c: {m: paired(OV[c][m], OV["none"][m], osch) for m in METRICS} for c in DIRS},
                           recall=base_o["recall_acc"], arc=base_o["arc_acc"])
    # ------------------------------------------------------------- fresh sets
    FV = {(s, c): outcome_vectors(runs[(s, c)]["C1_rows"], runs[(s, c)]["C6_rows"]) for s in SETS for c in CFGS}
    sch = {s: np.array([r["scheme_id"] for r in msgs[s]]) for s in SETS}
    per_set = {}
    for s in SETS:
        d = dict(absolute={}, paired={}, per_scheme={})
        for c in CFGS:
            m = runs[(s, c)]
            d["absolute"][c] = dict(absolute(FV[(s, c)], 306), recall=m["recall_acc"], arc=m["arc_acc"],
                                    decoded_dist=m["C1"]["decoded_dist"], c6_decoded_dist=m["C6"].get("decoded_dist"))
        for c in DIRS:
            d["paired"][c] = {k: paired(FV[(s, c)][k], FV[(s, "none")][k], sch[s]) for k in METRICS}
            d["per_scheme"][c] = {k: {int(j): float((FV[(s, c)][k][sch[s] == j].astype(float)
                                                     - FV[(s, "none")][k][sch[s] == j]).mean()) for j in range(6)}
                                  for k in METRICS}
        per_set[s] = d
    res["per_set"] = per_set
    # baseline capability per set (descriptive comparison with the original baseline)
    ob = res["original"]["absolute"]["none"]
    res["baseline_capability"] = {s: dict(c1_decode=per_set[s]["absolute"]["none"]["c1_decode"],
                                          c1_decode_ci=per_set[s]["absolute"]["none"]["c1_decode_ci"],
                                          c1_bet_exact=per_set[s]["absolute"]["none"]["c1_bet_exact"],
                                          c6_bet_exact=per_set[s]["absolute"]["none"]["c6_bet_exact"],
                                          recall=per_set[s]["absolute"]["none"]["recall"],
                                          arc=per_set[s]["absolute"]["none"]["arc"],
                                          c1_decode_minus_original=per_set[s]["absolute"]["none"]["c1_decode"] - ob["c1_decode"],
                                          c1_bet_none=per_set[s]["absolute"]["none"]["c1_bet_none"],
                                          c1_decoded_none=per_set[s]["absolute"]["none"]["c1_decoded_none"])
                                  for s in SETS}
    res["baseline_capability"]["original"] = dict(c1_decode=ob["c1_decode"], c1_decode_ci=ob["c1_decode_ci"],
                                                  c1_bet_exact=ob["c1_bet_exact"], c6_bet_exact=ob["c6_bet_exact"],
                                                  recall=base_o["recall_acc"], arc=base_o["arc_acc"])
    # ARC determinism: ARC does not depend on messages, so per config it should be identical across sets
    res["arc_by_config"] = {c: [runs[(s, c)]["arc_acc"] for s in SETS] for c in CFGS}
    # ------------------------------------------------------------- pooled across sets
    strata = np.concatenate([s * 10 + sch[s] for s in SETS])  # set x scheme strata
    PV = {c: {k: np.concatenate([FV[(s, c)][k] for s in SETS]) for k in FV[(SETS[0], c)]} for c in CFGS}
    pooled = dict(absolute={c: dict(absolute(PV[c], 306 * len(SETS)),
                                    recall_mean=float(np.mean([runs[(s, c)]["recall_acc"] for s in SETS])))
                            for c in CFGS},
                  paired={}, seed_level={}, per_scheme={})
    psch = np.concatenate([sch[s] for s in SETS])
    for c in DIRS:
        pooled["paired"][c] = {k: paired(PV[c][k], PV["none"][k], strata) for k in METRICS}
        pooled["seed_level"][c] = {}
        for k in METRICS:
            ds = np.array([per_set[s]["paired"][c][k]["delta"] for s in SETS])
            pooled["seed_level"][c][k] = dict(per_set=ds.tolist(), mean=float(ds.mean()), sd=float(ds.std(ddof=1)),
                                              min=float(ds.min()), max=float(ds.max()),
                                              n_sets_ci_excl0_same_sign_as_orig=int(sum(
                                                  (per_set[s]["paired"][c][k]["ci"][1] < 0 if res["original"]["paired"][c][k]["delta"] < 0 else
                                                   per_set[s]["paired"][c][k]["ci"][0] > 0 if res["original"]["paired"][c][k]["delta"] > 0 else False)
                                                  for s in SETS)))
        pooled["per_scheme"][c] = {k: {int(j): float((PV[c][k][psch == j].astype(float) - PV["none"][k][psch == j]).mean())
                                       for j in range(6)} for k in METRICS}
    res["pooled"] = pooled
    res["original_per_scheme"] = {c: {k: {int(j): float((OV[c][k][osch == j].astype(float) - OV["none"][k][osch == j]).mean())
                                          for j in range(6)} for k in METRICS} for c in DIRS}
    # ------------------------------------------------------------- cluster bootstrap over contexts (PRIMARY pooled CI)
    CRNG = np.random.default_rng(0)
    ctx = np.concatenate([np.arange(306) for s in SETS])  # row position == orig_id position (order verified)
    for c in CFGS:
        for k, key in (("c1_dec", "c1_decode"), ("c1_bet", "c1_bet_exact"), ("c6_bet", "c6_bet_exact")):
            pooled["absolute"][c][key + "_cluster_ci"] = cluster_boot(PV[c][k], ctx, psch, CRNG)["ci"]
    for c in DIRS:
        for k in METRICS:
            cb = cluster_boot(PV[c][k].astype(float) - PV["none"][k], ctx, psch, CRNG)
            pp = pooled["paired"][c][k]
            pp["ci_optimistic"] = pp["ci"]  # set x scheme strata, episodes resampled independently: optimistic
            pp["ci_cluster"] = cb["ci"]
            pp["ci"] = cb["ci"]  # primary
            pp["n_contexts"] = cb["n_clusters"]
    # ------------------------------------------------------------- fixed interpretation rules
    rules = {}
    for c in DIRS:
        rules[c] = {}
        for k in METRICS:
            od = res["original"]["paired"][c][k]["delta"]
            p = pooled["paired"][c][k]
            lo, hi = p["ci"]
            if od == 0:
                rep = None
                verdict = "n/a (original Δ = 0; sign undefined)"
            else:
                same_sign_excl0 = (hi < 0) if od < 0 else (lo > 0)
                ratio = p["delta"] / od
                rep = bool(same_sign_excl0 and ratio >= 0.5)
                verdict = "REPLICATES" if rep else ("CI excludes 0, same sign, <50% of original" if same_sign_excl0
                                                    else "does not replicate (CI includes 0 or opposite sign)")
            oci = res["original"]["paired"][c][k]["ci"]
            gate = abs(od) >= 0.10 and (oci[1] < 0 or oci[0] > 0)
            lo2, hi2 = p["ci_optimistic"]
            mech_opt = (None if od == 0 else bool(((hi2 < 0) if od < 0 else (lo2 > 0)) and p["delta"] / od >= 0.5))
            rules[c][k] = dict(original_delta=od, original_ci=oci, pooled_delta=p["delta"], pooled_ci=p["ci"],
                               pooled_ci_optimistic=p["ci_optimistic"],
                               ratio_to_original=(p["delta"] / od if od != 0 else None),
                               gate_original_ge_10pp_ci_excl0=bool(gate),
                               verdict=(verdict if gate else "n/a (original effect < 10 pp)"),
                               mechanical_replicates=rep, mechanical_verdict=verdict,
                               mechanical_replicates_optimistic_ci=mech_opt)
        c6 = pooled["paired"][c]["c6_bet"]
        drop, drop_hi = -c6["delta"], -c6["ci"][0]
        drop_hi_opt = -c6["ci_optimistic"][0]
        rules[c]["c6_spared"] = dict(drop=drop, drop_ci_upper=drop_hi, spared=bool(drop <= 0.05 and drop_hi <= 0.10),
                                     drop_ci_upper_optimistic=drop_hi_opt,
                                     spared_optimistic=bool(drop <= 0.05 and drop_hi_opt <= 0.10))
        oc6 = res["original"]["paired"][c]["c6_bet"]
        rules[c]["c6_spared_original"] = dict(drop=-oc6["delta"], drop_ci_upper=-oc6["ci"][0],
                                              spared=bool(-oc6["delta"] <= 0.05 and -oc6["ci"][0] <= 0.10))
    res["rules"] = rules
    res["novel_subset"] = novel_subset(res, msgs, FV, sch, ctx, psch, CRNG)
    res["notes"] = [
        "PRIMARY pooled CI = cluster bootstrap over the 306 contexts (orig_id), stratified by scheme (51 per scheme), "
        "10k reps, RNG seed 0 (separate generator); each sampled context carries all 5 sets' outcomes.",
        "OPTIMISTIC pooled CI (kept for reference) = episodes resampled independently within set x scheme strata; it "
        "treats the 5 sets' copies of a context as independent and understates uncertainty.",
        "REPLICATES vocabulary is gated: only used when the original effect is >= 10 pp with a CI excluding 0; otherwise "
        "'n/a (original effect < 10 pp)'. The mechanical rule outcome is reported in a separate column.",
        "The n=5 seed-level SD is indicative only: a 95% CI for an SD from 5 values is roughly [0.6, 2.9] x SD.",
        "About 150 uncorrected tests are run across the project; all p-values and CIs here are descriptive (no "
        "multiplicity correction); do not interpret |delta| < 5 pp or p > 1e-3 as evidence on their own."]
    res["assumptions"] = ("Pooled bootstrap resamples episodes within set x scheme strata (30 strata of 51): sets are "
                          "treated as fixed and episodes within a set as exchangeable. It does not treat the 5 sets as "
                          "a random sample of sender seeds; seed-level variability is reported separately (SD/range of "
                          "the 5 per-set Δ). The 5 sets share the same 306 episode contexts, so pooled episodes are "
                          "not independent across sets.")
    json.dump(res, open(FRESH / "analysis.json", "w"), indent=1, default=str)
    write_tables(res)
    figure(res)
    return res


def novel_subset(res, msgs, FV, sch, ctx, psch, rng):
    """Paired Δ on the 'novel' fresh messages: max difflib ratio vs ANY original < 0.9 AND not an exact copy of the
    same episode's original. Not a random subset (selection on message text); nothing is dropped from the main analysis."""
    out = dict(definition="max SequenceMatcher ratio vs any original message < 0.9 AND not an exact copy of the same "
                          "episode's original message", per_set={}, pooled={})
    masks = {}
    for s in SETS:
        nov = set(res["overlap"][s]["novel_ids"])
        mk = np.array([r["id"] in nov for r in msgs[s]])
        masks[s] = mk
        rows = msgs[s]
        cn = compliance([r for r, m in zip(rows, mk) if m])
        co = compliance([r for r, m in zip(rows, mk) if not m])
        d = dict(n=int(mk.sum()), n_other=int((~mk).sum()),
                 per_scheme_n={int(j): int(mk[sch[s] == j].sum()) for j in range(6)},
                 baseline_c1_decode=float(FV[(s, "none")]["c1_dec"][mk].mean()),
                 baseline_c1_decode_other=float(FV[(s, "none")]["c1_dec"][~mk].mean()),
                 clean_target_rate=cn["clean_target_rate"], clean_target_rate_other=co["clean_target_rate"],
                 target_topic_rate=cn["target_topic_rate"], target_topic_rate_other=co["target_topic_rate"],
                 paired={})
        for c in DIRS:
            d["paired"][c] = {k: paired(FV[(s, c)][k][mk], FV[(s, "none")][k][mk], sch[s][mk]) for k in METRICS}
        out["per_set"][s] = d
    M = np.concatenate([masks[s] for s in SETS])
    for name, sel in (("novel", M), ("non_novel", ~M)):
        PVn = {c: {k: np.concatenate([FV[(s, c)][k] for s in SETS])[sel] for k in METRICS} for c in CFGS}
        cx, sc = ctx[sel], psch[sel]
        allrows = sum((msgs[s] for s in SETS), [])
        comp = compliance([r for r, m in zip(allrows, sel) if m])
        g = dict(n=int(sel.sum()), n_contexts=int(len(np.unique(cx))),
                 baseline={k: cluster_boot(PVn["none"][k], cx, sc, rng) for k in METRICS},
                 clean_target_rate=comp["clean_target_rate"], target_topic_rate=comp["target_topic_rate"],
                 forbidden_rate=comp["forbidden_rate"], paired={})
        for c in DIRS:
            g["paired"][c] = {}
            for k in METRICS:
                diff = PVn[c][k].astype(float) - PVn["none"][k]
                cb = cluster_boot(diff, cx, sc, rng)
                a, b = PVn[c][k], PVn["none"][k]
                n01, n10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
                pv = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
                g["paired"][c][k] = dict(delta=cb["est"], ci=cb["ci"], mcnemar_p=float(pv), abs=float(a.mean()))
        out["pooled"][name] = g
    return out


def f3(x):
    return f"{x:.3f}"


def fd(p):
    return f"{p['delta']:+.3f} [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}]"


def write_tables(res):
    L = ["# Fresh-message robustness (Qwen2.5-7B, frozen original directions) — generated by exp/fresh_analyze.py\n"]
    L += ["## Verification\n", "| set | n | unique ids / 1:1 orig_id | schemes×states, per cell | same order as original | context identical | parse_talk failures | all 6 runs: ids, C1/C6 order, msgs match |",
          "|---|---|---|---|---|---|---|---|"]
    for s, v in res["verification"]["sets"].items():
        ok = all(r["ids_equal_set"] and r["c1_order"] and r["c6_order"] and r["c1_msg_equal"] in (True, None) for r in v["runs"].values())
        L.append(f"| m{s} | {v['n']} | {v['unique_ids']} / {v['orig_id_one_to_one']} | {v['n_schemes']}×{v['n_states']}, {v['per_cell']} | "
                 f"{v['same_order_as_original']} | {v['context_identical_to_original']} | {v['parse_talk_failures']} | {ok} |")
    o = res["verification"]["original"]
    L.append(f"| original | {o['n']} | – | {o['n_schemes']}×{o['n_states']}, {o['per_cell']} | – | – | {res['compliance']['original']['parse_talk_failures']} | – |")
    L += ["\n## Overlap with the original frozen messages (no filtering applied)\n",
          "| set | exact same-ep | norm. same-ep | exact any-orig | norm. any-orig | ratio≥0.9 same-ep | ratio≥0.9 any-orig | ratio≥0.9 other-ep | same-ep ratio median [IQR] | max-any ratio median [IQR] | within-set exact/norm dups |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in SETS:
        x = res["overlap"][s]
        a, b = x["ratio_same_episode"], x["ratio_max_any_original"]
        L.append(f"| m{s} | {x['exact_same_episode']} | {x['normalised_same_episode']} | {x['exact_any_original']} | {x['normalised_any_original']} | "
                 f"{x['near_dup_same_episode_ge09']} | {x['near_dup_any_original_ge09']} | {x['near_dup_other_episode_ge09']} | "
                 f"{a['median']:.3f} [{a['q25']:.3f}–{a['q75']:.3f}] | {b['median']:.3f} [{b['q25']:.3f}–{b['q75']:.3f}] | "
                 f"{x['within_set_exact_dups']}/{x['within_set_normalised_dups']} |")
    ow = res["overlap"]["original_within_set"]
    L.append(f"\nOriginal set within-set duplicates (exact/normalised): {ow['exact_dup_messages']}/{ow['normalised_dup_messages']}.\n")
    L += ["Across fresh sets (pairs):\n", "| pair | exact same-ep | norm. same-ep | exact any | norm. any | ratio≥0.9 same-ep | same-ep ratio median |", "|---|---|---|---|---|---|---|"]
    for k, x in res["overlap"]["across_fresh_sets"].items():
        L.append(f"| {k} | {x['exact_same_episode']} | {x['normalised_same_episode']} | {x['exact_any']} | {x['normalised_any']} | "
                 f"{x['near_dup_same_episode_ge09']} | {x['ratio_same_episode_median']:.3f} |")
    L += ["\n## Sender compliance (n = 306 per set)\n",
          "| set | forbidden-word rate | digit rate | number-word rate | target topic mentioned | clean target (collude_clean) | other scheme topic present | parse_talk failures | mean length (chars) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in ["original"] + SETS + ["fresh_pooled"]:
        x = res["compliance"][s]
        L.append(f"| {s if s in ('original', 'fresh_pooled') else 'm%d' % s} | {x['forbidden_rate']:.3f} | {x['digit_rate']:.3f} | {x['numword_rate']:.3f} | "
                 f"{x['target_topic_rate']:.3f} | {x['clean_target_rate']:.3f} | {x['other_scheme_topic_rate']:.3f} | {x['parse_talk_failures']} | {x['mean_len_chars']:.0f} |")
    L += ["\n## Baseline capability (config `none`)\n",
          "| set | C1 decode [Wilson] | C1 bet-exact | C6 bet-exact | recall | ARC | C1 decode − original | C1 bet None / decoded None |", "|---|---|---|---|---|---|---|---|"]
    for s in ["original"] + SETS:
        x = res["baseline_capability"][s]
        extra = (f"{x['c1_decode_minus_original']:+.3f} | {x['c1_bet_none']} / {x['c1_decoded_none']}" if s != "original"
                 else f"– | {res['original']['absolute']['none']['c1_bet_none']} / {res['original']['absolute']['none']['c1_decoded_none']}")
        L.append(f"| {s if s == 'original' else 'm%d' % s} | {x['c1_decode']:.3f} [{x['c1_decode_ci'][0]:.3f}–{x['c1_decode_ci'][1]:.3f}] | "
                 f"{x['c1_bet_exact']:.3f} | {x['c6_bet_exact']:.3f} | {x['recall']:.3f} | {x['arc']:.3f} | {extra} |")
    L.append("\nARC per config across sets (message-independent; determinism check): "
             + "; ".join(f"{c}: {sorted(set(v))}" for c, v in res["arc_by_config"].items()) + "\n")
    L += ["## Absolute accuracy per set and config\n",
          "| config | set | C1 decode [Wilson] | C1 bet-exact | C6 bet-exact [Wilson] | C1 bet None | C1 decoded None | C6 bet None | recall | ARC |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c in CFGS:
        x = res["original"]["absolute"][c]
        L.append(f"| {c} | original | {x['c1_decode']:.3f} [{x['c1_decode_ci'][0]:.3f}–{x['c1_decode_ci'][1]:.3f}] | {x['c1_bet_exact']:.3f} | "
                 f"{x['c6_bet_exact']:.3f} [{x['c6_bet_exact_ci'][0]:.3f}–{x['c6_bet_exact_ci'][1]:.3f}] | {x['c1_bet_none']} | {x['c1_decoded_none']} | {x['c6_bet_none']} | – | – |")
        for s in SETS:
            x = res["per_set"][s]["absolute"][c]
            L.append(f"| {c} | m{s} | {x['c1_decode']:.3f} [{x['c1_decode_ci'][0]:.3f}–{x['c1_decode_ci'][1]:.3f}] | {x['c1_bet_exact']:.3f} | "
                     f"{x['c6_bet_exact']:.3f} [{x['c6_bet_exact_ci'][0]:.3f}–{x['c6_bet_exact_ci'][1]:.3f}] | {x['c1_bet_none']} | {x['c1_decoded_none']} | {x['c6_bet_none']} | "
                     f"{x['recall']:.3f} | {x['arc']:.3f} |")
        x = res["pooled"]["absolute"][c]
        L.append(f"| {c} | **pooled over 5 sets of the same 306 contexts (cluster CI)** | {x['c1_decode']:.3f} [{x['c1_decode_cluster_ci'][0]:.3f}–{x['c1_decode_cluster_ci'][1]:.3f}] | {x['c1_bet_exact']:.3f} | "
                 f"{x['c6_bet_exact']:.3f} [{x['c6_bet_exact_cluster_ci'][0]:.3f}–{x['c6_bet_exact_cluster_ci'][1]:.3f}] | {x['c1_bet_none']} | {x['c1_decoded_none']} | {x['c6_bet_none']} | {x['recall_mean']:.3f} | – |")
    L += ["\n## Paired Δ vs matched `none` (stratified paired bootstrap 95% CI; exact McNemar p)\n",
          "Per set: strata = scheme (6 × 51). **Pooled (primary)**: cluster bootstrap over the 306 contexts (orig_id), stratified by scheme, each sampled context carrying all 5 sets. *Pooled, optimistic*: episodes resampled independently within set × scheme strata (understates uncertainty; reference only). McNemar p for pooled rows treats the 1530 pairs as independent (optimistic).\n",
          "| config | set | Δ C1 decode [CI] | p | Δ C1 bet-exact [CI] | p | Δ C6 bet-exact [CI] | p |", "|---|---|---|---|---|---|---|---|"]
    for c in DIRS:
        pp = res["pooled"]["paired"][c]
        opt = {k: dict(pp[k], ci=pp[k]["ci_optimistic"]) for k in METRICS}
        rows = ([("original", res["original"]["paired"][c])] + [(f"m{s}", res["per_set"][s]["paired"][c]) for s in SETS]
                + [("**pooled (cluster CI)**", pp), ("pooled (optimistic CI)", opt)])
        for name, p in rows:
            L.append(f"| {c} | {name} | " + " | ".join(f"{fd(p[k])} | {p[k]['mcnemar_p']:.2g}" for k in METRICS) + " |")
    L += ["\n## Seed-level summary (5 sets; per-set Δ)\n",
          "| config | metric | original Δ | mean Δ | SD | range | sets with CI excl. 0, same sign as original |", "|---|---|---|---|---|---|---|"]
    for c in DIRS:
        for k in METRICS:
            x = res["pooled"]["seed_level"][c][k]
            L.append(f"| {c} | {MNAME[k]} | {res['original']['paired'][c][k]['delta']:+.3f} | {x['mean']:+.3f} | {x['sd']:.3f} | "
                     f"[{x['min']:+.3f}, {x['max']:+.3f}] | {x['n_sets_ci_excl0_same_sign_as_orig']}/5 |")
    L += ["\n## Fixed interpretation rule (plan §A), gated\n",
          "Mechanical rule: replicates = pooled Δ CI (cluster) excludes 0 with the original's sign AND pooled Δ ≥ 50% of the original Δ. "
          "Gate (review item 1): the REPLICATES wording is only used when the original effect is ≥ 10 pp with a CI excluding 0; otherwise 'n/a (original effect < 10 pp)'. "
          "C6 spared = pooled C6 drop ≤ 5 pp AND CI upper bound of the drop ≤ 10 pp.\n",
          "| config | metric | original Δ [CI] | pooled Δ [cluster CI] | optimistic CI | ratio | verdict (gated) | mechanical outcome (cluster CI) |", "|---|---|---|---|---|---|---|---|"]
    for c in DIRS:
        for k in METRICS:
            x = res["rules"][c][k]
            r = "–" if x["ratio_to_original"] is None else f"{x['ratio_to_original']:.2f}"
            L.append(f"| {c} | {MNAME[k]} | {x['original_delta']:+.3f} [{x['original_ci'][0]:+.3f}, {x['original_ci'][1]:+.3f}] | "
                     f"{x['pooled_delta']:+.3f} [{x['pooled_ci'][0]:+.3f}, {x['pooled_ci'][1]:+.3f}] | "
                     f"[{x['pooled_ci_optimistic'][0]:+.3f}, {x['pooled_ci_optimistic'][1]:+.3f}] | {r} | **{x['verdict']}** | {x['mechanical_verdict']} |")
    L += ["\n| config | C6 drop fresh pooled (cluster CI upper) | C6 spared (fresh, cluster) | optimistic CI upper | C6 drop original (CI upper) | C6 spared (original) |", "|---|---|---|---|---|---|"]
    for c in DIRS:
        x, y = res["rules"][c]["c6_spared"], res["rules"][c]["c6_spared_original"]
        L.append(f"| {c} | {x['drop']:+.3f} ({x['drop_ci_upper']:+.3f}) | {x['spared']} | {x['drop_ci_upper_optimistic']:+.3f} | {y['drop']:+.3f} ({y['drop_ci_upper']:+.3f}) | {y['spared']} |")
    L += ["\n## Per-scheme Δ C1 decode (n = 51 per cell; original, each set, pooled)\n",
          "| config | scheme | original | " + " | ".join(f"m{s}" for s in SETS) + " | pooled |", "|---|---|---|" + "---|" * (len(SETS) + 1)]
    for c in DIRS:
        for j in range(6):
            L.append(f"| {c} | {j} | {res['original_per_scheme'][c]['c1_dec'][j]:+.3f} | "
                     + " | ".join(f"{res['per_set'][s]['per_scheme'][c]['c1_dec'][j]:+.3f}" for s in SETS)
                     + f" | {res['pooled']['per_scheme'][c]['c1_dec'][j]:+.3f} |")
    L += ["\n## Per-scheme Δ C6 bet-exact (original vs pooled fresh; n = 51 / 255 per cell)\n",
          "| config | " + " | ".join(f"s{j} orig / fresh" for j in range(6)) + " |", "|---|" + "---|" * 6]
    for c in DIRS:
        L.append(f"| {c} | " + " | ".join(f"{res['original_per_scheme'][c]['c6_bet'][j]:+.3f} / {res['pooled']['per_scheme'][c]['c6_bet'][j]:+.3f}"
                                          for j in range(6)) + " |")
    ns = res["novel_subset"]
    L += ["\n## Novel-message subset (review item 4)\n",
          f"Definition: {ns['definition']}. Selected on message text, so NOT a random subset: stock phrases are often the compliant "
          "target-topic lines, so the novel subset may be less compliant (see clean-target rates). Nothing is dropped from the main analysis.\n",
          "| set | n novel / 306 | baseline C1 decode novel / rest | clean target novel / rest | Δ C1 decode dm | state | pc2d | lr | pc1d |", "|---|---|---|---|---|---|---|---|---|"]
    for s in SETS:
        d = ns["per_set"][s]
        L.append(f"| m{s} | {d['n']} | {d['baseline_c1_decode']:.3f} / {d['baseline_c1_decode_other']:.3f} | {d['clean_target_rate']:.3f} / {d['clean_target_rate_other']:.3f} | "
                 + " | ".join(fd(d["paired"][c]["c1_dec"]) for c in ("dm", "state", "pc2d", "lr", "pc1d")) + " |")
    L += ["\nPooled over sets (cluster bootstrap over contexts, stratified by scheme):\n",
          "| subset | n rows (contexts) | baseline C1 decode [CI] | clean target | config | Δ C1 decode [CI] | Δ C1 bet-exact [CI] | Δ C6 bet-exact [CI] | full-set pooled Δ C1 decode |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name in ("novel", "non_novel"):
        g = ns["pooled"][name]
        b = g["baseline"]["c1_dec"]
        for c in DIRS:
            L.append(f"| {name} | {g['n']} ({g['n_contexts']}) | {b['est']:.3f} [{b['ci'][0]:.3f}, {b['ci'][1]:.3f}] | {g['clean_target_rate']:.3f} | {c} | "
                     + " | ".join(fd(g["paired"][c][k]) for k in METRICS) + f" | {res['pooled']['paired'][c]['c1_dec']['delta']:+.3f} |")
    L.append("\n## Notes\n")
    L += [f"- {n}" for n in res["notes"]]
    L.append("\n" + res["assumptions"] + "\n")
    open(FRESH / "tables.md", "w").write("\n".join(L) + "\n")


def figure(res):
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), sharey=True)
    x = np.arange(len(CFGS))
    C_ORIG, C_SET, C_POOL = "#000000", "#56B4E9", "#0072B2"
    for ax, key, ci_key, title in ((axes[0], "c1_decode", "c1_decode_ci", "(a) Covert decoding (C1 decode)"),
                                   (axes[1], "c6_bet_exact", "c6_bet_exact_ci", "(b) Explicit-state decision (C6 bet-exact)")):
        ob = res["original"]["absolute"]["none"][key]
        pb = res["pooled"]["absolute"]["none"][key]
        ax.axhline(ob, color=C_ORIG, lw=0.7, ls="--", zorder=1, label="baseline, original set")
        ax.axhline(pb, color=C_POOL, lw=0.7, ls=":", zorder=1, label="baseline, fresh pooled")
        for i, c in enumerate(CFGS):
            ov = res["original"]["absolute"][c]
            oci = ov[ci_key]
            ax.errorbar(i - 0.22, ov[key], yerr=[[max(0, ov[key] - oci[0])], [max(0, oci[1] - ov[key])]], fmt="D", color=C_ORIG,
                        ms=3.5, lw=0.8, capsize=2, zorder=3, label="original set (n=306)" if i == 0 else None)
            sv = [res["per_set"][s]["absolute"][c][key] for s in SETS]
            jit = np.linspace(-0.06, 0.06, len(SETS))
            ax.plot(i + jit, sv, "o", color=C_SET, mec="#1f4e79", mew=0.4, ms=3.2, zorder=3,
                    label="each fresh set (n=306)" if i == 0 else None)
            pv = res["pooled"]["absolute"][c]
            pci = pv[ci_key.replace("_ci", "_cluster_ci")]
            ax.errorbar(i + 0.22, pv[key], yerr=[[max(0, pv[key] - pci[0])], [max(0, pci[1] - pv[key])]], fmt="s", color=C_POOL,
                        ms=3.8, lw=0.8, capsize=2, zorder=3, label="fresh pooled: 5 sets × same 306 contexts (cluster CI)" if i == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[c] for c in CFGS], fontsize=7)
        ax.set_xlim(-0.6, len(CFGS) - 0.4)
        ax.set(ylim=(0, 1.05), title=title)
        ax.title.set_fontsize(8.5)
    axes[0].axhspan(0.30, 0.37, color="#eeeeee", zorder=0)
    axes[0].text(5.55, 0.335, "chance\nrange", fontsize=6, color="#666666", va="center", ha="right")
    axes[0].set_ylabel("accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(ROOT / f"figures/fig8_fresh.{ext}", dpi=300)


if __name__ == "__main__":
    r = main()
    for c in DIRS:
        print(c, {k: (round(r["rules"][c][k]["original_delta"], 3), round(r["rules"][c][k]["pooled_delta"], 3),
                      [round(v, 3) for v in r["rules"][c][k]["pooled_ci"]], r["rules"][c][k]["verdict"]) for k in METRICS},
              "C6 spared", r["rules"][c]["c6_spared"]["spared"])
