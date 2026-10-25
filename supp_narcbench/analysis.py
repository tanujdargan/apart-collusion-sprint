"""NARCBench (qwen3_32b) confound analysis -- single reproducible script.

Usage (from supp_narcbench/):
    uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python numpy scikit-learn matplotlib huggingface_hub scipy
    .venv/bin/python analysis.py            # downloads data if missing, writes tables.md, results.json, figs/*.png

Sections
  1. download + inspect
  2a. paper-faithful reproduction (imports external/narcbench/reproduce/reproduce_LR.py read-only, 10 seeds)
  2b. simple row-level probe (paper hyperparameters), Core within-tier eval via GroupKFold by scenario_id
  3. confound tests (surface features, correlations, strata, residualization)
  4. cross-tier probe direction cosine similarity vs shuffled-label baseline
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "activations" / "qwen3_32b"
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)
EXT = HERE.parent / "external" / "narcbench"
TIERS = ["core", "transfer", "stego"]
LAYERS = [26, 27, 28, 29, 30]

# Paper hyperparameters (reproduce_LR.py): C=25, l2, liblinear, max_iter=8
P_C, P_PEN, P_SOLVER, P_ITER = 25.0, "l2", "liblinear", 8

OUT = {}
MD = []


def md(s=""):
    print(s)
    MD.append(s)


def table(header, rows, fmt="{:.3f}"):
    md("| " + " | ".join(header) + " |")
    md("|" + "---|" * len(header))
    for r in rows:
        md("| " + " | ".join(fmt.format(c) if isinstance(c, (float, np.floating)) else str(c) for c in r) + " |")
    md()


def auc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


# ───────────────────────────── 1. download + inspect ─────────────────────────────

def download():
    if all((DATA / t / "activations_gen.npz").exists() for t in TIERS):
        return
    from huggingface_hub import snapshot_download
    snapshot_download("aaronrose227/narcbench", repo_type="dataset",
                      allow_patterns=[f"activations/qwen3_32b/{t}/*" for t in TIERS],
                      local_dir=str(HERE / "data"))


class Tier:
    def __init__(self, name):
        self.name = name
        d = DATA / name
        self.meta = json.load(open(d / "metadata_gen.json"))
        z = np.load(d / "activations_gen.npz")
        self.keys = list(z.files)
        self.X = {L: z[f"layer_{L}"][: len(self.meta)].astype(np.float32) for L in LAYERS}
        m = self.meta
        self.mode = np.array([int(x["mode"] == "collusion") for x in m])
        self.role = np.array([int(x["role"] == "colluder") for x in m])
        self.private = np.array([int(x["channel"] == "private") for x in m])
        self.phase = np.array([x["phase"] for x in m])
        self.round = np.array([x["round"] for x in m])
        self.ntok = np.array([x["n_tokens"] for x in m], dtype=float)
        self.run = np.array([x["run_id"] for x in m])
        self.scen = np.array([x["scenario_id"] for x in m])
        self.agent = np.array([x["agent_name"] for x in m])
        self.family = np.array([x["family"] for x in m])
        self.public = self.private == 0
        # evaluation phase used by the paper: Core = public discussion (averaged over rounds), Transfer/Stego = public final
        self.eval_phase = "discussion" if name == "core" else "final"


def inspect(T):
    md("## 1. Data inspection")
    md()
    for t in TIERS:
        x = T[t]
        md(f"**{t}**: {len(x.meta)} rows; npz keys {x.keys}; each {x.X[LAYERS[0]].shape} float32; "
           f"{len(set(x.run))} runs, {len(set(x.scen))} scenario_ids")
        for f in ["mode", "role", "channel", "phase", "round"]:
            md(f"- {f}: {dict(Counter(str(m[f]) for m in x.meta))}")
        md(f"- (mode, role, phase) counts and mean n_tokens:")
        d = defaultdict(list)
        for m in x.meta:
            d[(m["mode"], m["role"], m["phase"])].append(m["n_tokens"])
        for k in sorted(d):
            md(f"  - {k}: n={len(d[k])}, mean n_tokens={np.mean(d[k]):.0f}")
        md()


# ───────────────────────────── probe helpers ─────────────────────────────

def fit_probe(X, y, seed=0, C=P_C, solver=P_SOLVER, max_iter=P_ITER):
    """Paper-style probe: balance classes by subsampling majority (seeded), StandardScaler, LR.
    Returns (scaler, unit weight vector in standardized space)."""
    rng = np.random.RandomState(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n = min(len(pos), len(neg))
    pos = pos if len(pos) == n else rng.choice(pos, n, replace=False)
    neg = neg if len(neg) == n else rng.choice(neg, n, replace=False)
    idx = np.concatenate([neg, pos])
    sc = StandardScaler().fit(X[idx])
    clf = LogisticRegression(C=C, penalty="l2", solver=solver, max_iter=max_iter, random_state=seed)
    clf.fit(sc.transform(X[idx]), y[idx])
    w = clf.coef_[0]
    return sc, w / np.linalg.norm(w)


def score(sc, w, X):
    return sc.transform(X) @ w


def core_train_mask(x):
    """Paper's probe training set: collusion runs, public channel, discussion phase; label = role."""
    return (x.mode == 1) & x.public & (x.phase == "discussion")


def run_level(x, s, mask, agg="peak"):
    """Aggregate per-agent scores into a run score (paper's Peak Suspicion / Deception Split).
    Per agent: mean over rows in mask (Core discussion averages rounds). Returns (runs, y, scores)."""
    per = defaultdict(list)
    for i in np.where(mask)[0]:
        per[(x.run[i], x.agent[i])].append(s[i])
    runs = defaultdict(list)
    for (r, a), v in per.items():
        runs[r].append(np.mean(v))
    rids = sorted(r for r in runs if len(runs[r]) == 4)
    ys = np.array([int(r.endswith("__collusion")) for r in rids])
    out = []
    for r in rids:
        v = np.sort(runs[r])[::-1]
        out.append(v[0] if agg == "peak" else v[:2].mean() - v[2:].mean())
    return rids, ys, np.array(out)


def eval_mask(x):
    return x.public & (x.phase == x.eval_phase)


# ───────────────────────────── 2a. paper-faithful reproduction ─────────────────────────────

def paper_repro():
    md("## 2a. Paper-faithful reproduction (reproduce_LR.py imported read-only, 10 seeds x layers 26-30)")
    md()
    sys.path.insert(0, str(EXT))
    sys.path.insert(0, str(EXT / "reproduce"))
    import reproduce_LR as R
    R.CORE_DIR, R.TRANSFER_DIR, R.STEGO_DIR = DATA / "core", DATA / "transfer", DATA / "stego"
    cm, cn = R.load_data(R.CORE_DIR)
    tm, tn = R.load_data(R.TRANSFER_DIR)
    sm, sn = R.load_data(R.STEGO_DIR)
    all_raw, gnames = [], None
    for seed in range(R.N_SEEDS):
        gnames, raw = R.compute_all_results(cm, cn, tm, tn, sm, sn, seed=seed)
        all_raw.append(raw)
    res = {}
    rows = []
    for g in gnames:
        res[g] = {}
        row = [g]
        for m in R.METHOD_KEYS:
            vals = np.array([v for raw in all_raw for v in raw[g][m]], dtype=float)
            res[g][m] = {"mean": float(np.nanmean(vals)), "sd": float(np.nanstd(vals)),
                         "sem": float(np.nanstd(vals) / np.sqrt(np.sum(~np.isnan(vals)))),
                         "per_layer": [float(np.nanmean([raw[g][m][li] for raw in all_raw])) for li in range(len(LAYERS))]}
            row.append(f"{res[g][m]['mean']:.3f} ± {res[g][m]['sem']:.3f}")
        rows.append(row)
    md("Mean AUROC ± SEM over 10 seeds x 5 layers (same aggregation as reproduce_LR.py). Run-level labels (collusion vs control run).")
    md()
    table(["group"] + R.METHOD_KEYS, rows)
    md("Per-layer Peak Suspicion (mean over 10 seeds):")
    md()
    table(["group"] + [f"L{L}" for L in LAYERS],
          [[g] + res[g]["peak_suspicion"]["per_layer"] for g in gnames])
    OUT["paper_repro"] = res
    return res


# ───────────────────────────── 2b. simple probe + Core GroupKFold ─────────────────────────────

def probe_scores(T, L, seed=0):
    """Returns dict tier -> per-row probe scores at layer L.
    Core: out-of-fold scores from GroupKFold(5) by scenario_id (probe trained on train-fold scenarios only).
    Transfer/Stego: zero-shot scores from probe trained on all Core."""
    c = T["core"]
    X, y, tr = c.X[L], c.role, core_train_mask(c)
    s_core = np.full(len(y), np.nan)
    for tr_s, te_s in GroupKFold(5).split(X, y, c.scen):
        trm = np.zeros(len(y), bool); trm[tr_s] = True
        sc, w = fit_probe(X[trm & tr], y[trm & tr], seed)
        s_core[te_s] = score(sc, w, X[te_s])
    sc, w = fit_probe(X[tr], y[tr], seed)
    S = {"core": s_core}
    for t in ["transfer", "stego"]:
        S[t] = score(sc, w, T[t].X[L])
    return S, (sc, w)


def baseline(T):
    md("## 2b. Simple probe (paper hyperparameters), per stored layer")
    md()
    md("Probe: StandardScaler + LogisticRegression(C=25, l2, liblinear, max_iter=8), trained on Core collusion-run public "
       "discussion rows, label role (colluder=1 vs honest=0), majority class subsampled (seed 0). Core numbers are "
       "out-of-fold from GroupKFold(5) grouped by scenario_id (collusion+control runs of a scenario never split). "
       "Transfer/Stego: zero-shot with the probe trained on all Core rows.")
    md()
    md("Metrics: `role` = colluder vs honest rows inside collusion runs (eval phase, public); `mode-row` = collusion-run "
       "vs control-run rows (eval phase, public); `peak` / `split` = run-level Peak Suspicion / Deception Split; "
       "eval phase = Core public discussion (agent score averaged over rounds), Transfer/Stego public final.")
    md()
    res = {t: defaultdict(list) for t in TIERS}
    S_all = {}
    for L in LAYERS:
        S, _ = probe_scores(T, L)
        S_all[L] = S
        for t in TIERS:
            x, s, em = T[t], S[t], eval_mask(T[t])
            cm = em & (x.mode == 1)
            res[t]["role"].append(auc(x.role[cm], s[cm]))
            res[t]["mode-row"].append(auc(x.mode[em], s[em]))
            _, ys, ps = run_level(x, s, em, "peak")
            res[t]["peak"].append(auc(ys, ps))
            _, ys, ds = run_level(x, s, em, "split")
            res[t]["split"].append(auc(ys, ds))
    rows = []
    for t in TIERS:
        for k in ["role", "mode-row", "peak", "split"]:
            rows.append([t, k] + res[t][k] + [float(np.mean(res[t][k]))])
    table(["tier", "metric"] + [f"L{L}" for L in LAYERS] + ["mean"], rows)
    # transfer per family, peak
    md("Transfer per-family run-level Peak Suspicion (simple probe, mean over layers 26-30; 12 runs each):")
    md()
    x = T["transfer"]
    fam_rows = []
    for f in sorted(set(x.family)):
        v = []
        for L in LAYERS:
            em = eval_mask(x) & (x.family == f)
            _, ys, ps = run_level(x, S_all[L]["transfer"], em, "peak")
            v.append(auc(ys, ps))
        fam_rows.append([f, float(np.mean(v))])
    table(["family", "peak AUROC"], fam_rows)
    OUT["baseline"] = {t: dict(v) for t, v in res.items()}
    OUT["transfer_family_peak"] = {r[0]: r[1] for r in fam_rows}
    return S_all, res


# ───────────────────────────── 3. confounds ─────────────────────────────

def surface_matrix(x, sets):
    cols = []
    if "ntok" in sets:
        cols.append(np.log(x.ntok)[:, None])
    if "channel" in sets:
        cols.append(x.private[:, None].astype(float))
    if "phase" in sets:
        for p in sorted(set(x.phase)):
            cols.append((x.phase == p)[:, None].astype(float))
    if "round" in sets:
        for r in sorted(set(x.round)):
            cols.append((x.round == r)[:, None].astype(float))
    if "role" in sets:
        cols.append(x.role[:, None].astype(float))
    return np.hstack(cols)


def cv_auc(F, y, groups, k=5):
    if len(np.unique(y)) < 2:
        return float("nan")
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(min(k, len(set(groups)))).split(F, y, groups):
        if len(np.unique(y[tr])) < 2:
            oof[te] = 0.5
            continue
        sc = StandardScaler().fit(F[tr])
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(F[tr]), y[tr])
        oof[te] = clf.decision_function(sc.transform(F[te]))
    return auc(y, oof)


FEATSETS = [("n_tokens", ["ntok"]), ("channel", ["channel"]), ("phase", ["phase"]), ("round", ["round"]),
            ("role", ["role"]), ("all except role", ["ntok", "channel", "phase", "round"]),
            ("all", ["ntok", "channel", "phase", "round", "role"])]


def confound_a(T, base):
    md("## 3a. Surface features alone (one-hot + LR, GroupKFold(5) by scenario_id, out-of-fold AUROC)")
    md()
    md("Targets: `mode/all` = collusion-run vs control-run over ALL rows (incl. private); `mode/eval` = same on eval-phase public rows only "
       "(the paper's evaluation set); `role/coll` = colluder vs honest inside collusion runs, eval-phase public rows "
       "(the probe's own training target). Constant features in a tier give 0.5 by construction (e.g. channel in Transfer).")
    md()
    res = {}
    rows = []
    for t in TIERS:
        x = T[t]
        em = eval_mask(x)
        cm = em & (x.mode == 1)
        res[t] = {}
        for name, fs in FEATSETS:
            F = surface_matrix(x, fs)
            a1 = cv_auc(F, x.mode, x.scen)
            a2 = cv_auc(F[em], x.mode[em], x.scen[em])
            a3 = cv_auc(F[cm], x.role[cm], x.scen[cm]) if name not in ("role", "all") else float("nan")
            res[t][name] = [a1, a2, a3]
            rows.append([t, name, a1, a2, a3])
    table(["tier", "features", "mode/all", "mode/eval", "role/coll"], rows)
    md("Note: `role` predicts `mode/all` trivially because only collusion runs contain colluders; `channel=private` likewise "
       "exists only in collusion runs (colluders' private coordination turns).")
    md()
    # run-level n_tokens surrogate (same aggregation as Peak Suspicion / Deception Split)
    md("Run-level n_tokens surrogate: replace the probe score with log(n_tokens) and apply the paper's Peak Suspicion / "
       "Deception Split aggregation on the eval-phase public rows (no fitting at all). Compare to the probe (2b, mean over layers):")
    md()
    rr = []
    res["run_ntok"] = {}
    for t in TIERS:
        x = T[t]
        em = eval_mask(x)
        _, ys, ps = run_level(x, np.log(x.ntok), em, "peak")
        _, ys2, ds = run_level(x, np.log(x.ntok), em, "split")
        # role-level: n_tokens alone for colluder vs honest
        cm = em & (x.mode == 1)
        ra = auc(x.role[cm], x.ntok[cm])
        res["run_ntok"][t] = [auc(ys, ps), auc(ys2, ds), ra]
        rr.append([t, auc(ys, ps), float(np.mean(base[t]["peak"])), auc(ys2, ds), float(np.mean(base[t]["split"])),
                   ra, float(np.mean(base[t]["role"]))])
    table(["tier", "peak(ntok)", "peak(probe)", "split(ntok)", "split(probe)", "role AUROC(ntok raw)", "role AUROC(probe)"], rr)
    # Transfer per family for ntok surrogate
    x = T["transfer"]
    fr = []
    for f in sorted(set(x.family)):
        em = eval_mask(x) & (x.family == f)
        _, ys, ps = run_level(x, np.log(x.ntok), em, "peak")
        fr.append([f, auc(ys, ps)])
    md("Transfer per-family Peak(n_tokens):")
    md()
    table(["family", "peak(ntok) AUROC"], fr)
    OUT["confound_a"] = res
    return res


def confound_b(T, S_all):
    md("## 3b. Correlation of the Core probe score with surface features")
    md()
    md("Rows = all rows of the tier (Core: out-of-fold scores). Spearman rho with n_tokens; Pearson r (point-biserial) "
       "with private-channel indicator, colluder indicator, and collusion-run indicator. Also Spearman(score, n_tokens) "
       "within honest-only and colluder-only rows (eval phase, public) to separate length from role.")
    md()
    rows = []
    res = {}
    for t in TIERS:
        x = T[t]
        em = eval_mask(x)
        for L in LAYERS:
            s = S_all[L][t]
            r_nt = spearmanr(s, x.ntok)[0]
            r_pr = pearsonr(s, x.private)[0] if x.private.std() > 0 else float("nan")
            r_ro = pearsonr(s, x.role)[0]
            r_mo = pearsonr(s, x.mode)[0]
            h = em & (x.role == 0)
            c = em & (x.role == 1)
            r_h = spearmanr(s[h], x.ntok[h])[0]
            r_c = spearmanr(s[c], x.ntok[c])[0]
            res[(t, L)] = [r_nt, r_pr, r_ro, r_mo, r_h, r_c]
            rows.append([t, f"L{L}", r_nt, r_pr, r_ro, r_mo, r_h, r_c])
    table(["tier", "layer", "rho(ntok)", "r(private)", "r(colluder)", "r(collusion run)", "rho(ntok|honest,eval)", "rho(ntok|colluder,eval)"], rows)
    OUT["confound_b"] = {f"{k[0]}_L{k[1]}": v for k, v in res.items()}


def strat_auc(y, s, strata):
    """Within-stratum AUROC, averaged weighted by n_pos*n_neg (= pooled pairwise comparisons within strata)."""
    num, den = 0.0, 0.0
    for v in np.unique(strata):
        m = strata == v
        npos, nneg = y[m].sum(), (1 - y[m]).sum()
        if npos == 0 or nneg == 0:
            continue
        num += roc_auc_score(y[m], s[m]) * npos * nneg
        den += npos * nneg
    return num / den if den else float("nan")


def ntok_bins(x, m, q=5):
    edges = np.quantile(np.log(x.ntok[m]), np.linspace(0, 1, q + 1))
    return np.clip(np.searchsorted(edges, np.log(x.ntok), side="right") - 1, 0, q - 1)


def confound_c(T, S_all):
    md("## 3c. Does the probe still work within single strata?")
    md()
    md("All on public rows. `mode|public` = collusion vs control rows, all public rows. `mode|eval` = eval phase only. "
       "`mode|honest` = honest agents in collusion runs vs honest agents in control runs (eval phase) -- tests whether the "
       "probe sees 'being in a colluding run' rather than 'being a colluder'. `role|coll` = colluder vs honest in collusion "
       "runs (eval phase). `coll vs ctrl-honest` = colluders vs control-run honest agents (eval phase). "
       "`role|round` = role AUROC within each round (phase x round), pairwise-pooled. `role|ntok-quintile` = role AUROC within "
       "n_tokens quintiles (quintiles of the eval-phase collusion-run rows), pairwise-pooled. `role|round&ntok` = within "
       "round x ntok-quintile cells.")
    md()
    rows = []
    res = {}
    for t in TIERS:
        x = T[t]
        em = eval_mask(x)
        pub = x.public
        cm = em & (x.mode == 1)
        hm = em & (x.role == 0)
        cvh = em & ((x.role == 1) | (x.mode == 0))
        bins = ntok_bins(x, cm)
        rnd = np.array([f"{p}_{r}" for p, r in zip(x.phase, x.round)])
        for L in LAYERS:
            s = S_all[L][t]
            v = [auc(x.mode[pub], s[pub]), auc(x.mode[em], s[em]), auc(x.mode[hm], s[hm]),
                 auc(x.role[cm], s[cm]), auc(x.role[cvh], s[cvh]),
                 strat_auc(x.role[cm], s[cm], rnd[cm]), strat_auc(x.role[cm], s[cm], bins[cm]),
                 strat_auc(x.role[cm], s[cm], np.char.add(rnd[cm], bins[cm].astype(str)))]
            res[(t, L)] = v
            rows.append([t, f"L{L}"] + v)
    table(["tier", "layer", "mode|public", "mode|eval", "mode|honest", "role|coll", "coll vs ctrl-honest",
           "role|round", "role|ntok-quintile", "role|round&ntok"], rows)
    # same strata for the raw n_tokens surrogate, for reference
    md("Same strata using raw n_tokens as the score (reference for how much is explainable by length):")
    md()
    rr = []
    for t in TIERS:
        x = T[t]
        em = eval_mask(x)
        cm = em & (x.mode == 1)
        hm = em & (x.role == 0)
        cvh = em & ((x.role == 1) | (x.mode == 0))
        rnd = np.array([f"{p}_{r}" for p, r in zip(x.phase, x.round)])
        s = x.ntok
        rr.append([t, auc(x.mode[x.public], s[x.public]), auc(x.mode[em], s[em]), auc(x.mode[hm], s[hm]),
                   auc(x.role[cm], s[cm]), auc(x.role[cvh], s[cvh]), strat_auc(x.role[cm], s[cm], rnd[cm])])
    table(["tier", "mode|public", "mode|eval", "mode|honest", "role|coll", "coll vs ctrl-honest", "role|round"], rr)
    OUT["confound_c"] = {f"{k[0]}_L{k[1]}": v for k, v in res.items()}
    OUT["confound_c_ntok"] = {r[0]: r[1:] for r in rr}


def residualize(s, C, m):
    lr = LinearRegression().fit(C[m], s[m])
    return s - lr.predict(C)


def confound_d(T, S_all):
    md("## 3d. Residualized probe scores")
    md()
    md("OLS of probe score on covariates (fit within tier on the evaluated rows; covariates do not include the label), "
       "AUROC of residuals. Covariates: R1 = log n_tokens; R2 = log n_tokens + phase/round one-hot; "
       "R3 = R2 + private-channel (for all-row mode target). For run-level Peak, per-row residuals (R1/R2 fit on eval rows) "
       "are aggregated exactly like the raw score. Mean over layers 26-30; per-layer in results.json.")
    md()
    res = {}
    rows = []
    for t in TIERS:
        x = T[t]
        em = eval_mask(x)
        cm = em & (x.mode == 1)
        C1 = np.log(x.ntok)[:, None]
        C2 = surface_matrix(x, ["ntok", "phase", "round"])
        C3 = surface_matrix(x, ["ntok", "channel", "phase", "round"])
        acc = defaultdict(list)
        for L in LAYERS:
            s = S_all[L][t]
            allm = np.ones(len(s), bool)
            acc["role raw"].append(auc(x.role[cm], s[cm]))
            acc["role R1"].append(auc(x.role[cm], residualize(s, C1, cm)[cm]))
            acc["role R2"].append(auc(x.role[cm], residualize(s, C2, cm)[cm]))
            acc["mode-all raw"].append(auc(x.mode, s))
            acc["mode-all R3"].append(auc(x.mode, residualize(s, C3, allm)))
            acc["mode-eval raw"].append(auc(x.mode[em], s[em]))
            acc["mode-eval R1"].append(auc(x.mode[em], residualize(s, C1, em)[em]))
            _, ys, ps = run_level(x, s, em, "peak")
            acc["peak raw"].append(auc(ys, ps))
            for nm, C in [("R1", C1), ("R2", C2)]:
                _, ys, ps = run_level(x, residualize(s, C, em), em, "peak")
                acc[f"peak {nm}"].append(auc(ys, ps))
        res[t] = dict(acc)
        rows.append([t] + [float(np.mean(acc[k])) for k in acc])
        keys = list(acc)
    table(["tier"] + keys, rows)
    OUT["confound_d"] = res


# ───────────────────────────── 4. cross-tier cosine ─────────────────────────────

def tier_train_mask(x):
    return (x.mode == 1) & x.public


def raw_dir(sc, w):
    v = w / sc.scale_  # direction in raw activation space: w.(x-mu)/sigma = (w/sigma).x + const
    return v / np.linalg.norm(v)


def cosine_analysis(T, n_shuf=20):
    md("## 4. Cross-tier probe direction similarity")
    md()
    md("Per tier, a probe is trained on colluder-vs-honest rows from collusion runs (all public rows of that tier: Core "
       "discussion+final, Transfer final, Stego discussion+final). Two probe configs: P = paper hyperparameters "
       "(C=25, liblinear, max_iter=8, balanced subsample), S = C=1.0 lbfgs max_iter=5000 (converged). Cosine is between "
       "unit directions mapped to raw activation space (w / sigma). Baseline: same procedure with labels permuted "
       f"independently in each tier ({n_shuf} shuffles), mean ± sd. Ceiling: split-half (by scenario_id, 10 random halvings) "
       "cosine within a tier.")
    md()
    cfgs = {"P": dict(C=P_C, solver=P_SOLVER, max_iter=P_ITER), "S": dict(C=1.0, solver="lbfgs", max_iter=5000)}
    pairs = [("core", "transfer"), ("core", "stego"), ("transfer", "stego")]
    res = {}
    rows = []
    for cname, cfg in cfgs.items():
        for L in LAYERS:
            dirs = {}
            for t in TIERS:
                x = T[t]
                m = tier_train_mask(x)
                dirs[t] = raw_dir(*fit_probe(x.X[L][m], x.role[m], 0, **cfg))
            shuf = {p: [] for p in pairs}
            rng = np.random.RandomState(123)
            for k in range(n_shuf):
                sd = {}
                for t in TIERS:
                    x = T[t]
                    m = tier_train_mask(x)
                    sd[t] = raw_dir(*fit_probe(x.X[L][m], rng.permutation(x.role[m]), k, **cfg))
                for a, b in pairs:
                    shuf[(a, b)].append(float(sd[a] @ sd[b]))
            half = {}
            for t in TIERS:
                x = T[t]
                m = tier_train_mask(x)
                sc_ids = np.array(sorted(set(x.scen[m])))
                cs = []
                for k in range(10):
                    r2 = np.random.RandomState(1000 + k)
                    h = set(r2.permutation(sc_ids)[: len(sc_ids) // 2])
                    ina = np.array([s in h for s in x.scen])
                    da = raw_dir(*fit_probe(x.X[L][m & ina], x.role[m & ina], 0, **cfg))
                    db = raw_dir(*fit_probe(x.X[L][m & ~ina], x.role[m & ~ina], 0, **cfg))
                    cs.append(float(da @ db))
                half[t] = (float(np.mean(cs)), float(np.std(cs)))
            for a, b in pairs:
                real = float(dirs[a] @ dirs[b])
                mu, sd_ = float(np.mean(shuf[(a, b)])), float(np.std(shuf[(a, b)]))
                res[(cname, L, a, b)] = {"cos": real, "shuf_mean": mu, "shuf_sd": sd_, "z": (real - mu) / sd_}
                rows.append([cname, f"L{L}", f"{a}-{b}", real, f"{mu:.3f} ± {sd_:.3f}", (real - mu) / sd_])
            for t in TIERS:
                res[(cname, L, t, "split-half")] = {"mean": half[t][0], "sd": half[t][1]}
    table(["cfg", "layer", "pair", "cosine", "shuffled mean ± sd", "z"], rows)
    md("Split-half (within-tier) cosine ceiling, mean ± sd over 10 halvings:")
    md()
    hr = []
    for cname in cfgs:
        for L in LAYERS:
            hr.append([cname, f"L{L}"] + [f"{res[(cname, L, t, 'split-half')]['mean']:.3f} ± {res[(cname, L, t, 'split-half')]['sd']:.3f}" for t in TIERS])
    table(["cfg", "layer"] + TIERS, hr)
    OUT["cosine"] = {"|".join(map(str, k)): v for k, v in res.items()}
    return res


def ntok_direction(T):
    """How linearly decodable is n_tokens from activations, and how aligned is the Core probe with it?"""
    md("## 3e. Is the Core probe direction aligned with a length direction?")
    md()
    md("Ridge-free OLS is ill-posed (5120 dims), so we fit LR-style linear regression via sklearn Ridge(alpha=1e3) on "
       "standardized Core activations to predict log n_tokens (GroupKFold R^2), and report cosine between the Core probe "
       "direction (standardized space) and the length-regression direction.")
    md()
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    rows = []
    c = T["core"]
    for L in LAYERS:
        X = c.X[L]
        y = np.log(c.ntok)
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, c.scen):
            sc = StandardScaler().fit(X[tr])
            oof[te] = Ridge(alpha=1e3).fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[te]))
        m = core_train_mask(c)
        sc_p, w_p = fit_probe(X[m], c.role[m], 0)
        rg = Ridge(alpha=1e3).fit(sc_p.transform(X), y).coef_
        rows.append([f"L{L}", r2_score(y, oof), float(w_p @ rg / np.linalg.norm(rg))])
    table(["layer", "GroupKFold R^2 (log n_tokens)", "cos(probe, length dir)"], rows)
    OUT["ntok_direction"] = rows


# ───────────────────────────── figures ─────────────────────────────

def figures(T, base, paper, conf_a):
    # Fig 1: AUROC by layer/tier
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for t, c in zip(TIERS, ["#1b485e", "#d94749", "#579149"]):
        axes[0].plot(LAYERS, base[t]["role"], "-o", color=c, label=f"{t}: colluder vs honest")
        axes[0].plot(LAYERS, base[t]["peak"], "--s", color=c, label=f"{t}: run-level peak")
    axes[0].set_title("Simple probe (Core GroupKFold OOF; Transfer/Stego zero-shot)")
    for g, c in zip(paper, plt.cm.tab10.colors):
        axes[1].plot(LAYERS, paper[g]["peak_suspicion"]["per_layer"], "-o", color=c, label=g)
    axes[1].set_title("Paper-faithful Peak Suspicion (10 seeds)")
    for a in axes:
        a.axhline(0.5, color="grey", ls=":")
        a.set_xlabel("layer"); a.set_xticks(LAYERS); a.legend(fontsize=6)
    axes[0].set_ylabel("AUROC")
    fig.tight_layout(); fig.savefig(FIGS / "auroc_by_layer.png", dpi=200); plt.close(fig)

    # Fig 2: confound bar chart (eval-set targets)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    names = ["n_tokens", "round", "all except role", "probe (mean L26-30)"]
    w = 0.2
    for j, (tgt, ti, bk) in enumerate([("mode on eval rows (row-level)", 1, "mode-row"), ("colluder vs honest (collusion runs)", 2, "role")]):
        ax = axes[j]
        for i, nm in enumerate(names):
            vals = [float(np.mean(base[t][bk])) if nm.startswith("probe") else conf_a[t][nm][ti] for t in TIERS]
            ax.bar(np.arange(3) + (i - 1.5) * w, vals, w, label=nm)
        ax.axhline(0.5, color="grey", ls=":")
        ax.set_xticks(range(3)); ax.set_xticklabels(TIERS); ax.set_ylim(0, 1.05); ax.set_title(tgt)
        ax.set_ylabel("AUROC (GroupKFold by scenario)")
    axes[0].legend(fontsize=7)
    # add run-level ntok surrogate panel as text-free extra bars
    fig.tight_layout(); fig.savefig(FIGS / "confound_auroc.png", dpi=200); plt.close(fig)

    # Fig 2b: run-level peak: probe vs n_tokens surrogate
    fig, ax = plt.subplots(figsize=(5, 3.4))
    pv = [float(np.mean(base[t]["peak"])) for t in TIERS]
    nv = [conf_a["run_ntok"][t][0] for t in TIERS]
    ax.bar(np.arange(3) - 0.18, pv, 0.36, label="probe Peak Suspicion")
    ax.bar(np.arange(3) + 0.18, nv, 0.36, label="log n_tokens Peak (no model)")
    ax.axhline(0.5, color="grey", ls=":"); ax.set_xticks(range(3)); ax.set_xticklabels(TIERS)
    ax.set_ylim(0, 1.05); ax.set_ylabel("run-level AUROC"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGS / "runlevel_probe_vs_ntok.png", dpi=200); plt.close(fig)


def cosine_fig(res):
    for cname in ["P", "S"]:
        fig, axes = plt.subplots(1, len(LAYERS), figsize=(3 * len(LAYERS), 3))
        for ax, L in zip(axes, LAYERS):
            M = np.eye(3)
            for i, a in enumerate(TIERS):
                for j, b in enumerate(TIERS):
                    k = (cname, L, a, b) if (cname, L, a, b) in res else (cname, L, b, a)
                    if i != j:
                        M[i, j] = res[k]["cos"]
            im = ax.imshow(M, vmin=-0.2, vmax=1, cmap="viridis")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w" if M[i, j] < 0.6 else "k", fontsize=8)
            sh = np.mean([res[(cname, L, a, b)]["shuf_mean"] for a, b in [("core", "transfer"), ("core", "stego"), ("transfer", "stego")]])
            ax.set_title(f"L{L} (shuffled ~{sh:.3f})", fontsize=9)
            ax.set_xticks(range(3)); ax.set_xticklabels(TIERS, fontsize=7)
            ax.set_yticks(range(3)); ax.set_yticklabels(TIERS, fontsize=7)
        fig.colorbar(im, ax=axes, shrink=0.8)
        fig.suptitle(f"Cosine between per-tier colluder probes (config {cname})", fontsize=10)
        fig.savefig(FIGS / f"cosine_heatmap_{cname}.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def main():
    download()
    T = {t: Tier(t) for t in TIERS}
    inspect(T)
    paper = paper_repro()
    S_all, base = baseline(T)
    conf_a = confound_a(T, base)
    confound_b(T, S_all)
    confound_c(T, S_all)
    confound_d(T, S_all)
    ntok_direction(T)
    cres = cosine_analysis(T)
    figures(T, base, paper, conf_a)
    cosine_fig(cres)
    (HERE / "tables.md").write_text("\n".join(MD))
    json.dump(OUT, open(HERE / "results.json", "w"), indent=1, default=lambda o: o if not isinstance(o, np.floating) else float(o))


if __name__ == "__main__":
    main()
