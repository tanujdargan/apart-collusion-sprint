"""Build the FROZEN verification matrices (notes/verify_plan.md). CPU only; writes data/verify/matrices/<name>.json.

  .venv-analysis/bin/python exp/verify_matrix.py base          # v7b_dose, v32b, s2_fresh, s3_32b, s4_14b, smoke specs
  .venv-analysis/bin/python exp/verify_matrix.py s1            # needs data/verify/dose_fit.json -> dose_alpha.json, s1_7b
  add --necessary-only to drop the optional configs (meta.optional = True).

Dose rule (§1A, fixed before the fit): per fold k and late layer l (18-27), from the fold's TRAINING episodes only,
E_X,l = sum_i sum_t ||h_i,t Q_X,l||^2 / sum_i n_i over all non-pad C1 prompt tokens (unablated forward). Common
attainable target T_l = min(E_M2,l, E_PC2,l); doses c_M2,l = sqrt(T_l / E_M2,l), c_PC2,l = sqrt(T_l / E_PC2,l) (both <= 1;
the larger-energy subspace is attenuated, the other stays at full removal).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/verify/matrices"
OPT = "--necessary-only" not in sys.argv
FRESH = [101, 202, 303, 404, 505]


def cfg(tag, d, layers="all", dose=None, optional=False, **meta):
    return dict(tag=tag, dir=d, layers=layers, dose=dose, meta=dict(optional=optional, **meta))


def keep(cs):
    return [c for c in cs if OPT or not c["meta"]["optional"]]


def job(cfgs, ids, group, mp="", arc=True):
    return dict(cfgs=keep(cfgs), ids=ids, conditions=["C1", "C6"], do_arc=arc, group=group, messages_path=mp)


def save(name, M):
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(M, open(OUT / f"{name}.json", "w"), indent=1)
    n = sum(len(j["cfgs"]) for c in M.get("containers", []) for j in c)
    print(name, "containers", len(M.get("containers", [])), "configs", n)


def late(a, L):
    return list(range(a, L))


def base():
    sp7 = json.load(open(ROOT / "data/loso/splits.json"))
    # ---- 7B validation of the dose/meter path (fold-0 M2, 9 fold-0 held-out episodes, as notes/unbundle_plan.md)
    save("v7b_dose", dict(model="Qwen/Qwen2.5-7B-Instruct", gpu="L40S", dir="data/loso/fold0/state.npy",
                          ids=sp7["0"]["eval_ids"][::6], messages_path="",
                          configs=[dict(tag="none", layers=None), dict(tag="late", layers=late(18, 28)),
                                   dict(tag="late_half", layers=late(18, 28), dose=0.5)]))
    # ---- §2 fresh-message placement (fold-fit directions; one container per fresh set, folds sequential; no ARC:
    # ARC prompts do not contain messages, so ARC per direction x placement is the original-message run's)
    conts = []
    for s in FRESH:
        jobs = []
        for k in range(6):
            d, c = f"data/loso/fold{k}/state.npy", f"data/loso/fold{k}/pc2d.npy"
            ids = [f"m{s}_{i}" for i in sp7[str(k)]["eval_ids"]]
            p = f"m{s}_f{k}_"
            jobs.append(job([cfg(p + "none", None), cfg(p + "M2_all", d), cfg(p + "M2_late", d, late(18, 28)),
                             cfg(p + "EMB_LATE", d, ["emb"] + late(18, 28)), cfg(p + "PRE_LATE", d, list(range(28))),
                             cfg(p + "PC2_late", c, late(18, 28)),
                             cfg(p + "PC2_all", c, optional=True), cfg(p + "PC2_PRE_LATE", c, list(range(28)), optional=True)],
                            ids, "verify/s2_fresh", f"/vol/fresh/m{s}/messages.jsonl", arc=False))
        conts.append(jobs)
    save("s2_fresh", dict(model="Qwen/Qwen2.5-7B-Instruct", gpu="L40S", containers=conts))
    # ---- §3A 32B factorial (fold-fit M2; original 7B frozen messages)
    sp32 = json.load(open(ROOT / "data_models/qwen32b/loso/splits.json"))
    conts = []
    for k in range(6):
        d = f"data_models/qwen32b/loso/fold{k}/state.npy"
        p = f"f{k}_"
        conts.append([job([cfg(p + "none", None), cfg(p + "M2_late", d, late(42, 64)), cfg(p + "M2_all", d),
                           cfg(p + "EMB_LATE", d, ["emb"] + late(42, 64)), cfg(p + "PRE_LATE", d, list(range(64)))],
                          sp32[str(k)]["eval_ids"], "verify/s3_32b", "/vol/data/messages.jsonl")])
    save("s3_32b", dict(model="Qwen/Qwen2.5-32B-Instruct", gpu="H200", containers=conts))
    save("v32b", dict(model="Qwen/Qwen2.5-32B-Instruct", gpu="H200", dir="data_models/qwen32b/loso/fold0/state.npy",
                      ids=sp32["0"]["eval_ids"][::6], messages_path="/vol/data/messages.jsonl",
                      configs=[dict(tag="none", layers=None), dict(tag="M2_late", layers=late(42, 64)),
                               dict(tag="M2_all", layers="all"), dict(tag="EMB_LATE", layers=["emb"] + late(42, 64)),
                               dict(tag="PRE_LATE", layers=list(range(64)))]))
    # ---- §3B 14B (its own frozen messages; L32 = frozen relative-depth cutoff; L26 = follow-up motivated by the
    # previously observed onset, NOT an untouched preregistered choice)
    sp14 = json.load(open(ROOT / "data_models/qwen14b/loso/splits.json"))
    conts = []
    for k in range(6):
        d, c = f"data_models/qwen14b/loso/fold{k}/state.npy", f"data_models/qwen14b/loso/fold{k}/pc2d.npy"
        p = f"f{k}_"
        conts.append([job([cfg(p + "none", None), cfg(p + "M2_all", d), cfg(p + "PC2_all", c),
                           cfg(p + "M2_L32", d, late(32, 48)), cfg(p + "PC2_L32", c, late(32, 48)),
                           cfg(p + "M2_L26", d, late(26, 48), followup=True), cfg(p + "PC2_L26", c, late(26, 48), followup=True)],
                          sp14[str(k)]["eval_ids"], "verify/s4_14b", "/vol/qwen14b/data/messages.jsonl")])
    save("s4_14b", dict(model="Qwen/Qwen2.5-14B-Instruct", gpu="L40S", containers=conts))


def alphas():
    F = json.load(open(ROOT / "data/verify/dose_fit.json"))
    meta = json.load(open(ROOT / "data/acts/meta.json"))
    assert F["ids"] == meta["ids"]
    g = np.array(meta["scheme_id"])
    n = np.array(F["n"])
    lay = F["layers"]
    out = {}
    for k in range(6):
        tr = g != k
        row = {}
        for j, l in enumerate(lay):
            E = {x: float(np.array(F["E"][f"f{k}_{x}"])[tr, j, 0].sum() / n[tr].sum()) for x in ("M2", "PC2")}
            Em = {x: float(np.array(F["E"][f"f{k}_{x}"])[tr, j, 1].sum() / (n[tr] - 2).sum()) for x in ("M2", "PC2")}
            T = min(E.values())
            row[l] = dict(E_M2=E["M2"], E_PC2=E["PC2"], E_h=float(np.array(F["H"])[tr, j].sum() / n[tr].sum()),
                          E_mid_M2=Em["M2"], E_mid_PC2=Em["PC2"], target=T,
                          c_M2=float(np.sqrt(T / E["M2"])), c_PC2=float(np.sqrt(T / E["PC2"])))
        out[str(k)] = row
    json.dump(out, open(ROOT / "data/verify/dose_alpha.json", "w"), indent=1)
    return out


def s1():
    A = json.loads(json.dumps(alphas()))  # string keys, as stored in dose_alpha.json
    sp7 = json.load(open(ROOT / "data/loso/splits.json"))
    conts = []
    for k in range(6):
        d, c, dm, p1 = (f"data/loso/fold{k}/{x}.npy" for x in ("state", "pc2d", "dm", "pc1d"))
        dose = {x: [1.0] * 28 for x in ("M2", "PC2")}
        for l in range(18, 28):
            dose["M2"][l], dose["PC2"][l] = A[str(k)][str(l)]["c_M2"], A[str(k)][str(l)]["c_PC2"]
        p = f"f{k}_"
        L18 = late(18, 28)
        conts.append([job([
            cfg(p + "none", None),
            cfg(p + "M2_L18", d, L18), cfg(p + "PC2_L18", c, L18),                       # anchors (core-matrix repro)
            cfg(p + "M2_L18_match", d, L18, dose["M2"]), cfg(p + "PC2_L18_match", c, L18, dose["PC2"]),
            cfg(p + "M2_L18_half", d, L18, 0.5), cfg(p + "PC2_L18_half", c, L18, 0.5),
            cfg(p + "dm_L18", dm, L18), cfg(p + "dm_all", dm),                           # dm_all = LOSO dm_fold repro
            cfg(p + "M2_L16", d, late(16, 28)), cfg(p + "PC2_L16", c, late(16, 28)),
            cfg(p + "M2_L20", d, late(20, 28)), cfg(p + "PC2_L20", c, late(20, 28)),
            cfg(p + "pc1d_L18", p1, L18, optional=True), cfg(p + "pc1d_all", p1, optional=True)],
            sp7[str(k)]["eval_ids"], "verify/s1_7b")])
    save("s1_7b", dict(model="Qwen/Qwen2.5-7B-Instruct", gpu="L40S", containers=conts))


STATES = ["COLD", "NEUTRAL", "HOT"]


def fit_state_pc2(X, st):
    """Same constructions as exp/loso_fit.py fit() for state (M2) and pc2d, on the given training rows only."""
    L = X.shape[1]
    mu = {s: X[st == s].mean(0) for s in STATES}
    v1, v2 = mu["HOT"] - mu["COLD"], mu["NEUTRAL"] - (mu["HOT"] + mu["COLD"]) / 2
    M2 = np.stack([np.linalg.qr(np.stack([v1[l], v2[l]], 1))[0].T for l in range(L)])
    pc2 = []
    for l in range(L):
        Xc = X[:, l] - X[:, l].mean(0)
        Xo = Xc - (Xc @ M2[l].T) @ M2[l]
        pc2.append(np.linalg.svd(Xo, full_matrices=False)[2][:2])
    return M2.astype(np.float32), np.stack(pc2).astype(np.float32)


def _sha(f):
    import hashlib
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def _eval_matrix(name, base_dir, splits, mp, group):
    jobs = []
    for k in range(6):
        d, c = f"{base_dir}/fold{k}/state.npy", f"{base_dir}/fold{k}/pc2d.npy"
        p = f"f{k}_"
        jobs.append(job([cfg(p + "none", None), cfg(p + "M2f_all", d), cfg(p + "M2f_L18", d, late(18, 28)),
                         cfg(p + "PC2f_all", c), cfg(p + "PC2f_L18", c, late(18, 28))], splits[str(k)], group, mp))
    save(name, dict(model="Qwen/Qwen2.5-7B-Instruct", gpu="L40S", containers=[jobs]))


def g4a():
    """Refit on fresh sets m101-m404 (training schemes only), evaluate on m505 held-out schemes."""
    D = ROOT / "data/verify/g4a"
    X, st, g = [], [], []
    for sd in (101, 202, 303, 404):
        meta = json.load(open(D / f"cap_m{sd}.json"))
        rows = {r["id"]: r for r in (json.loads(l) for l in open(ROOT / f"data/fresh/m{sd}/messages.jsonl"))}
        X.append(np.load(D / f"cap_m{sd}.npy").astype(np.float32))
        st += [rows[i]["state"] for i in meta["ids"]]
        g += [rows[i]["scheme_id"] for i in meta["ids"]]
    X, st, g = np.concatenate(X), np.array(st), np.array(g)
    prov = dict(inputs={f"cap_m{sd}.npy": _sha(D / f"cap_m{sd}.npy") for sd in (101, 202, 303, 404)}, folds={})
    for k in range(6):
        tr = g != k
        M2, PC2 = fit_state_pc2(X[tr], st[tr])
        (D / f"fold{k}").mkdir(exist_ok=True)
        np.save(D / f"fold{k}/state.npy", M2)
        np.save(D / f"fold{k}/pc2d.npy", PC2)
        prov["folds"][str(k)] = dict(n_train=int(tr.sum()), state_sha=_sha(D / f"fold{k}/state.npy"),
                                     pc2d_sha=_sha(D / f"fold{k}/pc2d.npy"))
    json.dump(prov, open(D / "provenance.json", "w"), indent=1)
    sp7 = json.load(open(ROOT / "data/loso/splits.json"))
    _eval_matrix("g4a_eval", "data/verify/g4a", {k: [f"m505_{i}" for i in sp7[k]["eval_ids"]] for k in sp7},
                 "/vol/fresh/m505/messages.jsonl", "verify/g4a")


def g4b():
    """Feasibility gate (baseline C1 strict decode >= 0.70 on all 306 new episodes; reused from the original pipeline),
    then LOSO refit on the new schemes and the eval matrix. A gate failure is recorded and stops 4b (no tuning)."""
    D = ROOT / "data/verify/g4b"
    b = json.load(open(D / "baseline306_metrics.json"))
    acc = b["C1"]["decode_acc"]
    gate = dict(c1_strict_decode=acc, c1_exact_bet=b["C1"]["bet_exact_acc"], threshold=0.70, passed=acc >= 0.70)
    json.dump(gate, open(D / "gate.json", "w"), indent=1)
    print("G4b gate", gate)
    if not gate["passed"]:
        return
    rows = [json.loads(l) for l in open(D / "messages.jsonl")]
    X = np.load(D / "cap_none.npy").astype(np.float32)
    st, g = np.array([r["state"] for r in rows]), np.array([r["scheme_id"] for r in rows])
    splits, prov = {}, dict(inputs={"cap_none.npy": _sha(D / "cap_none.npy")}, folds={})
    for k in range(6):
        tr = g != k
        M2, PC2 = fit_state_pc2(X[tr], st[tr])
        (D / f"fold{k}").mkdir(exist_ok=True)
        np.save(D / f"fold{k}/state.npy", M2)
        np.save(D / f"fold{k}/pc2d.npy", PC2)
        splits[str(k)] = [r["id"] for r, t in zip(rows, ~tr) if t]
        prov["folds"][str(k)] = dict(n_train=int(tr.sum()), state_sha=_sha(D / f"fold{k}/state.npy"),
                                     pc2d_sha=_sha(D / f"fold{k}/pc2d.npy"))
    json.dump(prov, open(D / "provenance.json", "w"), indent=1)
    json.dump(splits, open(D / "splits.json", "w"), indent=1)
    _eval_matrix("g4b_eval", "data/verify/g4b", splits, "/vol/verify/g4b/messages.jsonl", "verify/g4b")


def v2():
    """Final targeted extension (notes/verify2_plan.md): S5 sampled-receiver cells + S6 detector placement factorial, one
    container per fold (S5 job, then S6 job); validation spec. Doses copied verbatim from the committed s1_7b matrix."""
    sp7 = json.load(open(ROOT / "data/loso/splits.json"))
    S1 = json.load(open(OUT / "s1_7b.json"))
    L18, PRE = late(18, 28), list(range(28))
    samp = dict(c1_seeds=list(range(8)), c1_temperature=0.7, c1_top_p=1.0)
    conts = []
    for k in range(6):
        s1d = {c["tag"].split("_", 1)[1]: c for c in S1["containers"][k][0]["cfgs"]}
        d, c, dm, p1 = (f"data/loso/fold{k}/{x}.npy" for x in ("state", "pc2d", "dm", "pc1d"))
        assert s1d["M2_L18_match"]["dir"] == d and s1d["PC2_L18_match"]["dir"] == c
        p5, p6 = f"s5_f{k}_", f"s6_f{k}_"
        s5 = [dict(cfg(p5 + "none", None), **samp), dict(cfg(p5 + "M2_L18", d, L18), **samp),
              dict(cfg(p5 + "PC2_L18", c, L18), **samp),
              dict(cfg(p5 + "M2_L18_match", d, L18, s1d["M2_L18_match"]["dose"]), **samp),
              dict(cfg(p5 + "PC2_L18_match", c, L18, s1d["PC2_L18_match"]["dose"]), **samp),
              dict(cfg(p5 + "dm_L18", dm, L18), **samp), dict(cfg(p5 + "dm_all", dm), **samp),
              dict(cfg(p5 + "pc1d_L18", p1, L18), **samp), dict(cfg(p5 + "pc1d_all", p1), **samp)]
        s6 = [cfg(p6 + "none", None), cfg(p6 + "dm_L18", dm, L18),                     # reproduction anchors vs S1
              cfg(p6 + "dm_EMB_LATE", dm, ["emb"] + L18), cfg(p6 + "dm_PRE_LATE", dm, PRE),
              cfg(p6 + "pc1d_EMB_LATE", p1, ["emb"] + L18), cfg(p6 + "pc1d_PRE_LATE", p1, PRE)]
        ids = sp7[str(k)]["eval_ids"]
        conts.append([job(s5, ids, "verify/v2"), job(s6, ids, "verify/v2")])
    save("v2_s5s6", dict(model="Qwen/Qwen2.5-7B-Instruct", gpu="L40S", containers=conts))
    s1d = {c["tag"].split("_", 1)[1]: c for c in S1["containers"][0][0]["cfgs"]}
    g = dict(temperature=0.7, top_p=1.0, seed=0)
    det = lambda x: [dict(tag="none", layers=None), dict(tag=f"{x}_L18", layers=L18), dict(tag=f"{x}_all", layers="all"),
                     dict(tag=f"{x}_EMB_LATE", layers=["emb"] + L18), dict(tag=f"{x}_PRE_LATE", layers=PRE)]
    json.dump(dict(ids=sp7["0"]["eval_ids"][:9],
                   hooks={"dm_greedy": dict(dir="data/loso/fold0/dm.npy", configs=det("dm")),
                          "pc1d_greedy": dict(dir="data/loso/fold0/pc1d.npy", configs=det("pc1d")),
                          "M2_sampled": dict(dir="data/loso/fold0/state.npy", gen=g,
                                             configs=[dict(tag="none", layers=None), dict(tag="M2_L18", layers=L18)]),
                          "dm_sampled": dict(dir="data/loso/fold0/dm.npy", gen=g, configs=det("dm")[:3]),
                          "pc1d_sampled": dict(dir="data/loso/fold0/pc1d.npy", gen=g, configs=det("pc1d")[:3])},
                   algebra={"M2_match_sampled": dict(dir="data/loso/fold0/state.npy", layers=L18,
                                                     dose=s1d["M2_L18_match"]["dose"], gen=g,
                                                     repeats=[["none_s0", None, None, 0],
                                                              ["M2_match_s3", L18, s1d["M2_L18_match"]["dose"], 3]]),
                            "PC2_match_sampled": dict(dir="data/loso/fold0/pc2d.npy", layers=L18,
                                                      dose=s1d["PC2_L18_match"]["dose"], gen=g,
                                                      repeats=[["PC2_match_s5", L18, s1d["PC2_L18_match"]["dose"], 5]])}),
              open(OUT / "v2_validate.json", "w"), indent=1)


FROZEN_4B = [("food", "family", "pets"), ("sports", "music", "movies"), ("travel", "cars", "weather"),
             ("movies", "travel", "food"), ("cars", "sports", "family"), ("pets", "weather", "music")]


def schemes4b():
    """Tier 4b scheme construction (notes/verify_plan.md §4b): enumerate 6 (C,N,H) triples over env.TOPICS with each topic
    in exactly 2 triples under 2 different states incl. its originally unused state, and no topic pair co-occurring in
    an original scheme or in two new ones; pick with random.seed(0). Asserts the frozen choice."""
    import itertools
    import random
    sys.path.insert(0, str(ROOT / "exp"))
    import env
    T, S = env.TOPICS, env.SCHEMES
    unused = {t: ({0, 1, 2} - {i for sc in S for i, x in enumerate(sc) if x == t}).pop() for t in T}
    co = {frozenset(q) for sc in S for q in itertools.combinations(sc, 2)}
    cand = [x for x in itertools.permutations(T, 3) if all(frozenset(q) not in co for q in itertools.combinations(x, 2))]
    sols = []

    def rec(ch, cnt, st):
        if len(ch) == 6:
            if all(cnt[t] == 2 and unused[t] in st[t] for t in T):
                sols.append(list(ch))
            return
        used = {frozenset(q) for c in ch for q in itertools.combinations(c, 2)}
        for x in cand[cand.index(ch[-1]) + 1 if ch else 0:]:
            if any(cnt[t] >= 2 for t in x) or any(j in st[t] for j, t in enumerate(x)) or \
                    any(frozenset(q) in used for q in itertools.combinations(x, 2)):
                continue
            for j, t in enumerate(x):
                cnt[t] += 1; st[t].add(j)
            rec(ch + [x], cnt, st)
            for j, t in enumerate(x):
                cnt[t] -= 1; st[t].discard(j)
    rec([], {t: 0 for t in T}, {t: set() for t in T})
    random.seed(0)
    pick = random.choice(sols)
    assert pick == FROZEN_4B, pick
    print(len(sols), "solutions; frozen choice reproduced:", pick)


if __name__ == "__main__":
    {"base": base, "s1": s1, "schemes4b": schemes4b, "g4a": g4a, "g4b": g4b, "v2": v2}[sys.argv[1]]()
