# exp/ — Cut the Wire GPU pipeline

Files: `env.py` (Signal Table environment: schemes, all prompt texts, parsers, metrics; pure python),
`modal_app.py` (Modal app `cutwire`: generation, receiver decode, activation extraction, interventions),
`analyze_gate.py` (sender compliance + decode metrics).

Setup (once): `uv venv -p 3.12 && uv pip install modal numpy scikit-learn`. **Use `.venv/bin/modal`** — the global
`modal` runs in a Python without numpy, and the `intervene` entrypoint needs numpy to load `.npy` files.
GPU: Modal L40S, volume `cutwire-vol` (HF cache at `/hf`, data at `/data`, acts at `/acts`, interventions at `/interv`).
Model: `Qwen/Qwen2.5-7B-Instruct`, bf16, transformers 4.46.3, sdpa. Every function has `timeout=3600` and
`scaledown_window=30`, so containers stop on their own.

## Stages (run from the project root)
| stage | command | wall | output |
|---|---|---|---|
| A gate | `.venv/bin/modal run exp/modal_app.py::gate --tag gate2` then `python3 exp/analyze_gate.py gate2` | ~1 min (+1 min first weight download) | `data/gate2_*.jsonl`, `notes/gate.md` |
| B data + acts | `.venv/bin/modal run exp/modal_app.py::stage_b` (17 seeds → 306 episodes) | 3.2 min | `data/messages.jsonl`, `data/receiver_outputs.jsonl`, `data/recall_outputs.jsonl`, volume `/acts/*` |
| pull acts | `for f in C1 C2 C3 C4 C1_traces C2_traces; do .venv/bin/modal volume get --force cutwire-vol /acts/$f.npz data/acts/; done; .venv/bin/modal volume get --force cutwire-vol /acts/meta.json data/acts/` | ~1 min | `data/acts/` |
| C baseline | `.venv/bin/modal run exp/modal_app.py::intervene --mode none --tag baseline` | 1.6 min | `data/interv/baseline/` |

Stage B messages are uploaded to the volume (`/data/messages.jsonl`). `intervene` reads them from there, so the inputs stay fixed.

## Conditions (receiver side)
C1 = colluding receiver + colluding message · C2 = colluding receiver + honest message (same prompt, only the partner text
differs) · C3 = honest receiver (no scheme) + colluding message · C4 = colluding receiver + honest message that casually
mentions an off-scheme topic (the episode's recall distractor) · C5 = scheme mismatch: C1's message, but the receiver's
private transcript uses `env.alt_scheme_id(ep)`, a scheme without the message's topic (`stage_c5` entrypoint; acts merged into
`data/acts/C5*.npz` plus `meta.json["conditions"]["C5"]["alt_scheme_id"]`).

## Activation files
`data/acts/{C1,C2,C3,C4}.npz`: arrays `mean`, `last`, `full_last`, each float16 `[306, 28, 3584]`. Layer l is the output
of decoder layer l, captured with a forward hook (equal to HF `hidden_states[l+1]`; the last layer is taken before the final norm).
- `mean`: mean over the sender-message tokens in the receiver prompt.
- `last`: last prompt token (`\n` after `<|im_start|>assistant`), i.e. the decision position.
- `full_last`: last token of prompt + greedy response + `<|im_end|>` (Rose et al. pooling). One causal forward pass
  covers the full transcript, so the prompt positions are identical to a prompt-only pass.
Row order = `meta.json["ids"]` (with `scheme_id` and `state`). `meta.json["conditions"][c]` holds `sender_msg`,
`receiver_output`, `spans` (token `[start, end)` of the sender message, plus the prompt length), and the per-layer mean norms
(`mean_span_norm_per_layer`, `last_token_norm_per_layer`), which help when choosing a steering alpha.
`data/acts/{C1,C2,C5}_traces.npz`: `acts` float16 `[40, 5, T, 3584]` (layers 8,12,16,20,24; sender span plus up to 10 following
tokens, zero padded), `lengths`, `idx` (row indices). Token strings are in `meta.json["conditions"][c]["traces"]["tokens"]`.
Message-level fields in `data/messages.jsonl`: `collude_msg`, `honest_msg`, `offtopic_msg` (plus raw outputs), and
`collude_clean` (keyword check: the message names only the target scheme topic).

## Intervention CLI
```
.venv/bin/modal run exp/modal_app.py::intervene \
   --direction-file dirs/d_detect.npy   # local .npy; or 'random' (seed-0 Gaussian); omit with --mode none
   --mode ablate|steer|none \
   --layers all|8,12,16                 # 'all' (ablate also hooks the embedding output) or a comma list
   --alpha 4.0                          # steer only: h += alpha * u (u unit-normalised)
   --conditions C1,C2,C3,C4,C5          # default C1-C4; C5 is supported
   --n 0                                # 0 = all 306; otherwise n evenly spaced episodes
   --tag my_run [--subspace] [--no-arc] [--no-acts] [--no-pull]
```
Directions are **raw residual-space** vectors. No standardisation is applied; vectors are unit-normalised inside. Accepted shapes:
`[D]` (one direction for all layers), `[L,D]` (per layer, L=28), `[k,D]` (k-dim subspace for all layers; pass
`--subspace` if k == 28 so it is not read as per-layer), `[L,k,D]` (per-layer subspace). Ablation:
`h <- h - (h @ Q) Q^T`, where Q is the QR orthonormal basis of each layer's vectors, applied at every position of each selected
decoder-layer output. Steering is 1-D: it uses the first vector of each layer.
For each condition it re-runs the greedy receiver decode under the intervention, plus topic recall (C1 messages, 4-way), plus
ARC-Easy test[:100] (letter answer). It also re-extracts `mean`, `last`, `full_last` with the hooks active (no traces).
Outputs land locally in `data/interv/<tag>/`: `metrics.json` (per-condition `decode_acc`, `bet_exact_acc`,
`bet_state_corr`, `parse_fail`, `mean_bet`; `recall_acc`; `arc_acc`; `<C>_rows` with raw outputs) and
`{C}.npz` + `meta.json` (same format as `data/acts`). The volume copy is at `/interv/<tag>/`.
Runtime for the full n=306 run with 4 conditions + ARC + acts: ~1.6 min of GPU plus ~1 min to pull ~740 MB, about $0.06 per run.

## Cost (Modal billing report, 2026-10-25)
All runs so far (2 gates, Stage B, 3 smoke tests, baseline, C5): **~$0.78** at 05:30 PDT. The L40S rate is ~$1.95/h.

## Other models (replication; notes/replication.md)
Every entrypoint takes `--model <hf id> --out-tag <short_tag>`. Volume paths become `/<short_tag>/{data,acts,interv}/...`
and local outputs go to `data_models/<short_tag>/` (same layout as `data/`). With no `--out-tag`, paths and behaviour are
the original 7B ones. `stage_b --conditions C1,C2,C3,C4,C5` runs C5 in the same pass. The end-of-turn string for
`full_last` is derived from the chat template (Qwen `<|im_end|>`, Phi-3.5 `<|end|>`, OLMo-2 `<|endoftext|>`).
OLMo-2 needs `CUTWIRE_TRANSFORMERS=4.48.3` in the shell (baked into the image). Local analysis reads `DATA_DIR`:
```
.venv/bin/modal run exp/modal_app.py::gate    --model Qwen/Qwen2.5-14B-Instruct --out-tag qwen14b
.venv/bin/modal run exp/modal_app.py::stage_b --model Qwen/Qwen2.5-14B-Instruct --out-tag qwen14b --conditions C1,C2,C3,C4,C5
for f in C1.npz C2.npz C3.npz C4.npz C5.npz meta.json; do .venv/bin/modal volume get --force cutwire-vol /qwen14b/acts/$f data_models/qwen14b/acts/; done
DATA_DIR=data_models/qwen14b .venv-analysis/bin/python exp/probes.py baseline
DATA_DIR=data_models/qwen14b .venv-analysis/bin/python exp/make_steer_dir.py 34
exp/run_replication.sh Qwen/Qwen2.5-14B-Instruct qwen14b ablate        # baseline + 8 ablations + probes post
exp/run_replication.sh Qwen/Qwen2.5-14B-Instruct qwen14b steer 34 -56.99,-28.50,28.50,56.99
.venv-analysis/bin/python exp/summarize_replication.py qwen14b          # markdown tables
```

## Post-hoc LOSO robustness check (Qwen2.5-7B; notes/loso_robustness_plan.md, REPORT_NOTES §4.8)
```
.venv-analysis/bin/python exp/loso_fit.py check   # refit on all episodes == original direction files
.venv-analysis/bin/python exp/loso_fit.py fit     # data/loso/fold<k>/{dm,state,pc1d,pc2d,lr}.npy + splits/provenance
.venv/bin/modal run exp/modal_app.py::loso        # 6 folds x 11 configs (none, <dir>_global, <dir>_fold); C1+C6+recall+ARC
.venv-analysis/bin/python exp/loso_analyze.py     # data/loso/analysis.json, tables.md, figures/fig7_loso
```
`Runner.intervene_many` evaluates several configs on one held-out episode subset in a fixed order, so batching is
identical across configs; it asserts that no hook leaks between configs. Outputs: data/loso/runs/<tag>/metrics.json.
