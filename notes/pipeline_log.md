# Pipeline log (GPU experiment pipeline)

Model: Qwen/Qwen2.5-7B-Instruct, bf16, HF transformers 4.46.3, torch 2.5.1, sdpa attention, Modal L40S.
Prompts: all prompt texts live in exp/env.py (SENDER_*, RECEIVER_*, TRANSCRIPT, Q_*). Code: exp/modal_app.py.
Decoding: sender temperature 0.7, top_p 1.0, top_k off, repetition_penalty 1.0 (Qwen default gen-config overridden),
max_new_tokens 120. Receiver/recall/ARC: greedy, repetition_penalty 1.0, max_new 24/6/4.

## Log
- 05:07 PDT start. Read PLAN.md + narcbench stego.py prompts. Wrote exp/env.py (6 schemes over 9 topics, each topic
  in 2 schemes under different states), exp/modal_app.py, exp/analyze_gate.py.
- 05:10-05:13 Gate v1 (7B, 90 episodes, sender not told topic explicitly): C1 decode 0.733 (PASS >=0.70), 0.983 when
  sender encodes cleanly; C2 0.322 (chance); C3 honest-rx bet-state r = -0.03; recall 0.722. Cost ~3.5 min L40S.
- 05:15 Coordinator constraints applied: (1) no state leak — episode context now seeded by (scheme, seed) only;
  (2) also save last-token activation of full transcript prompt+response+<|im_end|> ("full_last"); (3) intervention
  harness takes RAW residual-space directions (unit-normalised inside, no standardisation).
- 05:18 Prompt iteration 1: sender user prompt now names the topic for the state explicitly. Added control C4 =
  colluding receiver + honest msg casually mentioning an off-scheme topic (SENDER_OFFTOPIC_USER).
- 05:14 Gate v2 (90 episodes): C1 decode 0.978, bet-state r 0.952; C2 0.322, C4 0.322 (chance); C3 r=-0.066;
  recall 0.978. Wall 61 s. See notes/gate.md.
- 05:16-05:19 Stage B (gen_seed 1000, 17 seeds x 6 schemes x 3 states = 306 episodes; wall 192 s):
  colluding sender: no-forbidden 0.974, no-digit 0.987, keyword only-target-topic 0.814.
  C1 decode 0.958 (bet exact 0.958, bet-state r 0.913, mean bets 2.0/26.0/96.6); 0.984 on keyword-clean msgs (n=249),
  0.842 otherwise (n=57). C2 0.346 / r 0.068 (decodes mostly NEUTRAL). C3 honest-rx r 0.050 (mean bets ~49 flat).
  C4 0.337 / r -0.084. Topic recall 0.967. C1 acc by scheme: 0.96/0.98/0.92/1.00/0.88/1.00.
  Acts: data/acts/{C1..C4}.npz (mean,last,full_last; [306,28,3584] fp16), C1/C2 traces [40,5,32,3584].
- 05:20 Quick sanity probe (logreg on standardised acts, train schemes 0-3, test 4-5): C1 vs C2 test AUROC
  ~0.99-1.00 at layers 8/16/24 for all poolings; C1 vs C4 also ~0.99-1.00 (layers 4, 16). WARNING: the separation is
  likely driven largely by message style (colluding messages are short single-topic one-liners; honest ones talk about
  cards). Even C4 does not remove this. Probe-based "detection" here is easy, and part of it may be surface features.
- 05:21-05:23 Harness smoke tests (n=10): random 1-D ablate (all layers + embedding): C1 1.0, recall 1.0, ARC 0.97,
  |proj| onto d after ablation 0.06 (bf16 rounding) vs 0.9-1.5 baseline -> hooks work. Random [3,D] subspace
  ablate: C1 1.0. Steer alpha=200 at layer 12 with a random direction: outputs degenerate (C1 0.0) -> steering hooks bite.
  Added k-dim subspace ablation ([k,D], [L,k,D]; QR orthonormalisation) per coordinator request.
- 05:23-05:25 Baseline (mode none, n=306, C1-C4, ARC, acts): identical to Stage B (C1 0.958, C2 0.346, C3 r 0.050,
  C4 0.337), recall 0.967, ARC-Easy[:100] 0.97. Baseline acts are bit-identical to data/acts (max abs diff 0).
- Cost to date: ~$0.33 (modal billing report). No apps left running.
- 05:26-05:28 Added C5 "scheme-mismatch" (coordinator request): colluding receiver + the SAME colluding message as C1,
  but the private transcript uses env.alt_scheme_id(ep) = first scheme cyclically after the episode's own whose topics
  exclude ep["topic"] (env.mismatch). Names, hand, filler line and question are identical to C1. Entry point:
  `modal run exp/modal_app.py::stage_c5`. (First attempt wrote to a non-volume path and was lost; reran, ~$0.04.)
  C5 decode (n=306): acc vs true state 0.369 (chance 0.333), bet exact 0.369, bet-state r 0.146, parse_fail 0.003;
  decoded COLD 123 / NEUTRAL 144 / HOT 38 / unparsed 1; bets $25 x145, $0 x123, $100 x38; mean bets
  16.9/27.9/27.9. Only 3/306 messages keyword-hit a topic in the alt scheme. The receiver always answers with a
  confident in-format guess, never "no signal".
  Files: data/receiver_outputs_C5.jsonl, data/acts/C5.npz (mean,last,full_last [306,28,3584] fp16, same row order),
  data/acts/C5_traces.npz (same 40 trace idx as C1), meta.json["conditions"]["C5"] incl. alt_scheme_id (other
  fields verified unchanged).
  Quick probe C1 vs C5 (logreg, train schemes 0-3, test 4-5): mean-span L8 0.931, L16 0.959, L24 0.531;
  last L8 0.422, L16 0.526, L24 0.868; full_last L8 0.523, L16 0.372, L24 0.772. Much weaker than C1 vs C2/C4 (~1.0),
  consistent with the style confound inflating those. Caveat: in C5 the scheme text before the message differs too,
  so span activations can also reflect which topics the transcript lists.
- 05:30 Cost to date per modal billing report: $0.78 (the earlier $0.33 figure was lagging). No apps running.
