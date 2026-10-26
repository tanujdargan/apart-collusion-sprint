"""Verification extension, Phase 0 free analyses (CPU, local data only; notes/verify_plan.md).

  .venv-analysis/bin/python exp/phase0_free.py [parse|stability|readout|energy|all]  -> data/verify/phase0_*.json, phase0.md

parse     : mutually exclusive parse accounting of every existing receiver run (C1: decode × bet cross-tab; C6: bet).
stability : bootstrap of each fold's TRAINING contexts (cluster = (scheme, seed), all 3 states kept together, resampled
            within scheme) -> refit state span M2 [k=2], detector dm [k=1] and (7B) PC2 control [k=2]; compare subspaces
            to the fold estimate by projector distance d = ||P_b - P_0||_F / sqrt(2k) (= RMS sine of principal angles) and
            largest principal angle. No evaluation data or outcomes are used.
readout   : audit of the ablated-pass readout protocol (exp/redund_behaviour_analyze.py captures()): splits, and whether
            the readout training episodes are the ones the removed subspace was fitted on (train-side dependence).
energy    : PREVIEW ONLY (decision-token `last`, saved acts) of uncentred E||hQ||^2 for fold M2 vs PC2 at layers >= 16.
            The dose fit itself uses all prompt positions on GPU (notes/verify_plan.md §1A); this preview only checks
            which subspace carries more energy where.
"""
import glob
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/verify"
STATES = ["COLD", "NEUTRAL", "HOT"]
BETS = {"COLD": 0, "NEUTRAL": 25, "HOT": 100}
MODELS = {"7b": ROOT / "data", "14b": ROOT / "data_models/qwen14b", "32b": ROOT / "data_models/qwen32b"}


# ----------------------------------------------------------------------------- parse accounting
def c1_cell(x):
    """Mutually exclusive (decode, bet) category for a colluding-receiver row."""
    d, b, s = x["decoded"], x["bet"], x["state"]
    dc = "missing" if d is None else ("correct" if d == s else "wrong")
    if b is None:
        bc = "missing"
    elif b == BETS[s]:
        bc = "true_rule"                    # bet equals the rule bet for the TRUE state
    elif d is not None and b == BETS[d]:
        bc = "decoded_rule"                 # coherent with a wrong decode
    else:
        bc = "other"
    return dc, bc


def parse_accounting():
    files = sorted(glob.glob(str(ROOT / "data/**/metrics.json"), recursive=True) +
                   glob.glob(str(ROOT / "data_models/**/metrics.json"), recursive=True))
    groups = {}
    for f in files:
        r = json.load(open(f))
        if "C1_rows" not in r or "/smoke" in f or "smoke_" in Path(f).parent.name:
            continue
        rel = Path(f).relative_to(ROOT).parent
        tag = rel.name
        grp = str(rel.parent)
        cfg = tag
        for pre in ("loso_f", "redund_f", "unb_f"):  # pool folds: <pre><k>_<cfg>
            if tag.startswith(pre):
                cfg = tag[len(pre) + 2:]
        g = groups.setdefault((grp, cfg), dict(c1=Counter(), c6=Counter(), n_files=0))
        g["n_files"] += 1
        for x in r["C1_rows"]:
            g["c1"]["|".join(c1_cell(x))] += 1
        for x in r.get("C6_rows", []):
            b = x["bet"]
            g["c6"]["missing" if b is None else ("true_rule" if b == BETS[x["state"]] else "other")] += 1
    out = {}
    for (grp, cfg), g in sorted(groups.items()):
        c, n = g["c1"], sum(g["c1"].values())
        get = lambda dc=None, bc=None: sum(v for k, v in c.items() if (dc is None or k.split("|")[0] == dc)
                                           and (bc is None or k.split("|")[1] == bc))
        malformed = n - get("correct", "true_rule") - get("correct", "decoded_rule") - get("correct", "other") \
            - get("wrong", "true_rule") - get("wrong", "decoded_rule") - get("wrong", "other")
        n6 = sum(g["c6"].values())
        out[f"{grp}::{cfg}"] = dict(
            group=grp, config=cfg, n_files=g["n_files"], n=n,
            strict_decode=get("correct") / n, exact_bet=get(bc="true_rule") / n,
            both_correct=get("correct", "true_rule") / n,
            malformed=dict(total=malformed, decode_missing_bet_present=get("missing") - get("missing", "missing"),
                           decode_present_bet_missing=get("correct", "missing") + get("wrong", "missing"),
                           both_missing=get("missing", "missing"),
                           decode_missing_but_bet_true_rule=get("missing", "true_rule")),
            valid_wrong=dict(total=n - malformed - get("correct", "true_rule"),
                             decode_correct_bet_wrong=get("correct", "decoded_rule") + get("correct", "other"),
                             decode_wrong_bet_coherent=get("wrong", "decoded_rule"),
                             decode_wrong_bet_true_rule=get("wrong", "true_rule"),
                             decode_wrong_bet_other=get("wrong", "other")),
            crosstab=dict(c), c6=dict(n=n6, exact_bet=g["c6"]["true_rule"] / n6 if n6 else None,
                                      bet_missing=g["c6"]["missing"], bet_wrong=g["c6"]["other"]))
        assert out[f"{grp}::{cfg}"]["malformed"]["total"] == sum(
            v for k, v in out[f"{grp}::{cfg}"]["malformed"].items() if k not in ("total", "decode_missing_but_bet_true_rule"))
    return out


# ----------------------------------------------------------------------------- subspace stability
def orth(V):  # V [k, D] -> orthonormal rows
    return np.linalg.qr(V.T)[0].T


def proj_dist(A, B):
    """A, B orthonormal rows [k, D]. Returns (RMS sine of principal angles, largest principal angle in degrees)."""
    s = np.clip(np.linalg.svd(A @ B.T, compute_uv=False), 0, 1)
    k = A.shape[0]
    return float(np.sqrt(max(0.0, (k - (s ** 2).sum()) / k))), float(np.degrees(np.arccos(s.min())))


def fit_dirs(X, P, N, st, layers, with_pc2):
    """Same constructions as exp/loso_fit.py fit(): M2 from C1 `last` class means, dm from C1/C4 `mean`, PC2 = top-2 PCs
    of centred C1 `last` orthogonal to M2. Returns {name: [len(layers), k, D] orthonormal}."""
    d = {"M2": [], "dm": [], "PC2": []}
    for l in layers:
        mu = {s: X[st == s, l].mean(0) for s in STATES}
        m2 = orth(np.stack([mu["HOT"] - mu["COLD"], mu["NEUTRAL"] - (mu["HOT"] + mu["COLD"]) / 2]))
        d["M2"].append(m2)
        v = P[:, l].mean(0) - N[:, l].mean(0)
        d["dm"].append((v / np.linalg.norm(v))[None])
        if with_pc2:
            Xc = X[:, l] - X[:, l].mean(0)
            Xo = Xc - (Xc @ m2.T) @ m2
            d["PC2"].append(np.linalg.svd(Xo, full_matrices=False)[2][:2])
    return {k: np.array(v) for k, v in d.items() if v}


def one_boot(b, tr, clusters, X, P, N, st, layers, with_pc2, ref):
    rng = np.random.default_rng(1000 + b)
    sch_of = {c: c[0] for c in clusters}
    pick = []
    for s in sorted({c[0] for c in clusters}):  # resample contexts WITHIN scheme (keeps scheme and state balance)
        cs = [c for c in clusters if sch_of[c] == s]
        pick += [cs[i] for i in rng.integers(0, len(cs), len(cs))]
    idx = np.concatenate([clusters[c] for c in pick])
    d = fit_dirs(X[idx], P[idx], N[idx], st[idx], layers, with_pc2)
    return {k: np.array([proj_dist(d[k][i], ref[k][i]) for i in range(len(layers))]) for k in d}


def stability(m, B=200, B_pc2=100):
    D = MODELS[m]
    meta = json.load(open(D / "acts/meta.json"))
    ids, st, g = meta["ids"], np.array(meta["state"]), np.array(meta["scheme_id"])
    seed = np.array([int(i.rsplit("_", 1)[1]) for i in ids])
    X = np.load(D / "acts/C1.npz")["last"].astype(np.float32)
    P = np.load(D / "acts/C1.npz")["mean"].astype(np.float32)
    N = np.load(D / "acts/C4.npz")["mean"].astype(np.float32)
    L = X.shape[1]
    layers = list(range(L))
    late0 = math.ceil(2 * (L - 1) / 3)
    with_pc2 = m == "7b"
    res = dict(model=m, L=L, late_start=late0, B=B, B_pc2=B_pc2 if with_pc2 else 0, folds={}, between_folds={})
    refs = {}
    for k in range(6):
        tr = np.where(g != k)[0]
        clusters = {}
        for i in tr:
            clusters.setdefault((int(g[i]), int(seed[i])), []).append(i)
        clusters = {c: np.array(v) for c, v in clusters.items()}
        # reference = the fold's own refit on its full training set (must equal the stored fold directions)
        ref = fit_dirs(X[tr], P[tr], N[tr], st[tr], layers, with_pc2)
        stored = np.load(D / f"loso/fold{k}/state.npy").astype(np.float64)
        chk = max(proj_dist(orth(stored[l]), ref["M2"][l])[0] for l in layers)
        refs[k] = ref
        bs = Parallel(n_jobs=-1)(delayed(one_boot)(b, tr, clusters, X, P, N, st, layers, False, ref) for b in range(B))
        fold = dict(n_train_episodes=int(len(tr)), n_train_contexts=len(clusters), max_dist_to_stored_M2=chk)
        for name in ("M2", "dm"):
            dist = np.array([x[name][:, 0] for x in bs])   # [B, L]
            ang = np.array([x[name][:, 1] for x in bs])
            fold[name] = dict(dist_median=np.median(dist, 0).tolist(), dist_q975=np.quantile(dist, 0.975, 0).tolist(),
                              angle_median=np.median(ang, 0).tolist(), angle_q975=np.quantile(ang, 0.975, 0).tolist(),
                              late_dist_median=float(np.median(dist[:, late0:].mean(1))),
                              late_dist_ci=np.quantile(dist[:, late0:].mean(1), [0.025, 0.975]).tolist())
        if with_pc2:
            bp = Parallel(n_jobs=-1)(delayed(one_boot)(b, tr, clusters, X, P, N, st, list(range(late0 - 2, L)), True,
                                                       {kk: v[late0 - 2:] for kk, v in ref.items()}) for b in range(B_pc2))
            dist = np.array([x["PC2"][:, 0] for x in bp])
            ang = np.array([x["PC2"][:, 1] for x in bp])
            fold["PC2_layers"] = list(range(late0 - 2, L))
            fold["PC2"] = dict(dist_median=np.median(dist, 0).tolist(), dist_q975=np.quantile(dist, 0.975, 0).tolist(),
                               angle_median=np.median(ang, 0).tolist(),
                               late_dist_median=float(np.median(dist[:, 2:].mean(1))),
                               late_dist_ci=np.quantile(dist[:, 2:].mean(1), [0.025, 0.975]).tolist())
        res["folds"][str(k)] = fold
        print(m, "fold", k, "M2 late median dist", round(fold["M2"]["late_dist_median"], 3), flush=True)
    for name in ("M2", "dm") + (("PC2",) if with_pc2 else ()):
        pair = []
        for a in range(6):
            for b in range(a + 1, 6):
                pair.append([proj_dist(refs[a][name][l], refs[b][name][l])[0] for l in range(L)])
        pair = np.array(pair)
        res["between_folds"][name] = dict(per_layer_mean=pair.mean(0).tolist(), late_mean=float(pair[:, late0:].mean()),
                                          late_min=float(pair[:, late0:].mean(1).min()),
                                          late_max=float(pair[:, late0:].mean(1).max()))
    # chance reference: random k-dim subspaces in D dims have RMS sine ~ sqrt(1 - k/D) ~ 1
    res["random_reference_dist"] = float(np.sqrt(1 - 2 / X.shape[2]))
    return res


# ----------------------------------------------------------------------------- readout protocol audit
def readout_audit(m):
    """Protocol facts + train-side dependence: at the FIRST hooked layer of a capture, the intervened trajectory is still
    the unablated one, so removing the fold span (fitted on the same training episodes' class means) leaves training
    class means ~equal -> the retrained readout there is near-degenerate by construction, not evidence."""
    D = MODELS[m]
    meta = json.load(open(D / "acts/meta.json"))
    st, g = np.array(meta["state"]), np.array(meta["scheme_id"])
    cd = ROOT / f"data/redund/{m}/caps"
    Nn = np.load(cd / "cap_redund_none.npy").astype(np.float32)
    L = Nn.shape[1]
    late0 = math.ceil(2 * (L - 1) / 3)
    splits = json.load(open(D / "loso/splits.json"))
    ids = meta["ids"]
    out = dict(model=m, late_start=late0, folds={})
    for k in range(6):
        tr = g != k
        fit_ids = set(splits[str(k)]["train_ids"])
        Q = np.load(D / f"loso/fold{k}/state.npy").astype(np.float64)
        A = {"late": np.load(cd / f"cap_redund_f{k}_M2_late.npy").astype(np.float32),
             "all": np.load(cd / f"cap_redund_f{k}_M2_all.npy").astype(np.float32)}
        f = dict(readout_train_equals_direction_fit_set=fit_ids == {ids[i] for i in np.where(tr)[0]},
                 test_scheme_in_fit=bool(k in {int(i.split("_")[0][1:]) for i in fit_ids}))

        def spread(X, sel):  # between-class spread of class means, relative to mean activation norm
            mu = np.stack([X[sel & (st == s)].mean(0) for s in STATES])
            return float(np.linalg.norm(mu - mu.mean(0), axis=1).mean() / np.linalg.norm(X[sel], axis=1).mean())
        for p, l0 in (("late", late0), ("all", 0)):
            f[p] = {f"L{l}": dict(train_spread_ablated=spread(A[p][:, l], tr), train_spread_unablated=spread(Nn[:, l], tr),
                                  test_spread_ablated=spread(A[p][:, l], ~tr))
                    for l in (l0, l0 + 1, l0 + 2)}
        out["folds"][str(k)] = f
    return out


def readout_sensitivity():
    """Late-mean retrained readout with and without the first hooked layer (frozen analysis output; no refits)."""
    a = json.load(open(ROOT / "data/redund/analysis.json"))
    out = {}
    for m in ("7b", "32b"):
        c = a[m]["captures"] if "captures" in a[m] else a[m]
        r = c["readout"]
        late0 = c["late_start"]
        out[m] = {p: dict(late_mean=r[p]["late_mean"],
                          late_mean_excl_first_hooked=float(np.mean(r[p]["per_layer"][late0 + 1:])) if p.startswith("late")
                          else r[p]["late_mean"], first_hooked_layer_acc=r[p]["per_layer"][late0] if p.startswith("late") else None,
                          next_layer_acc=r[p]["per_layer"][late0 + 1])
                  for p in ("base", "late_retrained", "all_retrained", "late_frozen", "all_frozen")}
    return out


# ----------------------------------------------------------------------------- energy preview
def energy_preview(m="7b"):
    D = MODELS[m]
    meta = json.load(open(D / "acts/meta.json"))
    g = np.array(meta["scheme_id"])
    X = np.load(D / "acts/C1.npz")["last"].astype(np.float64)
    L = X.shape[1]
    out = {}
    for k in range(6):
        tr = g != k
        S = np.load(D / f"loso/fold{k}/state.npy").astype(np.float64)
        C = np.load(D / f"loso/fold{k}/pc2d.npy").astype(np.float64)
        row = {}
        for l in range(16, L):
            Qs, Qc = orth(S[l]), orth(C[l])
            h = X[tr, l]
            es, ec, hh = float(((h @ Qs.T) ** 2).sum(1).mean()), float(((h @ Qc.T) ** 2).sum(1).mean()), float((h ** 2).sum(1).mean())
            hc = h - h.mean(0)
            row[l] = dict(E_state=es, E_pc2=ec, E_h=hh, ratio_pc2_over_state=ec / es,
                          centred_share_state=float(((hc @ Qs.T) ** 2).sum() / (hc ** 2).sum()),
                          centred_share_pc2=float(((hc @ Qc.T) ** 2).sum() / (hc ** 2).sum()))
        out[str(k)] = row
    return out


def main(which):
    OUT.mkdir(parents=True, exist_ok=True)
    if which in ("parse", "all"):
        json.dump(parse_accounting(), open(OUT / "phase0_parse.json", "w"), indent=1)
    if which in ("stability", "all"):
        json.dump({m: stability(m) for m in MODELS}, open(OUT / "phase0_stability.json", "w"), indent=1)
    if which in ("readout", "all"):
        json.dump(dict(audit={m: readout_audit(m) for m in ("7b", "32b")}, sensitivity=readout_sensitivity()),
                  open(OUT / "phase0_readout.json", "w"), indent=1)
    if which in ("energy", "all"):
        json.dump({"7b": energy_preview("7b")}, open(OUT / "phase0_energy_preview.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
