# REPORT NOTES — "Cut the Wire" (Apart AI Collusion Sprint, Oct 23–25 2026)

Everything needed to write the final 8-page report. Status markers: ✅ done · ⏳ running · ❌ failed/dropped.
Deadline: Mon Oct 26 04:59 PDT (= Sun 23:59 AoE). Template: official Apart template (sprint page).

## File map (where everything lives)
- This file: claims, abstract, method, every results table, figure captions, limitations/ethics, outline, judge Q&A.
- Figures: figures/fig1_setup … fig6_models (.png + .pdf, 300 dpi); supplementary: supp_narcbench/figs/*.png.
- Citations: notes/sources.md (verified table + references.bib); exact Rose et al. quotes: notes/rose_paper.md.
  Extra citation to add: Sun, Chen, Kolter & Liu 2024, "Massive Activations in Large Language Models", arXiv
  2402.17762 (from memory; verify) — for the Phi massive-activation note.
- Replication detail: notes/replication.md · gate: notes/gate.md · run log: notes/pipeline_log.md · NARCBench audit:
  supp_narcbench/results.md.
- Raw numbers: data/results_table.csv, data/fig4_specificity.csv, data/probes.json, data/token_trace_stats.json,
  data/token_trace_C1vC5dir.json, data/variance_shares.json, data/interv/<run>/metrics.json (+ post_probes.json);
  other models: data_models/<tag>/.
- Code: exp/ (env.py prompts, modal_app.py GPU pipeline, probes.py, figs_*.py, fig_models.py, run_*.sh); exp/README.md.

## 0. Submission checklist
- [ ] Title + abstract (150–250 words)
- [ ] Author name + affiliation
- [ ] ≤ 8 pages main text (refs + appendices excluded)
- [ ] Appendix: Limitations and Dual-Use / Ethical Considerations (REQUIRED)
- [ ] Optional: public GitHub repo link
- Track / question: Track 1 (1.3/1.4 detection) + 2.3 evidence. Builds on Rose et al. 2026.

## 1. Working title & one-sentence claim  (FINAL — updated after follow-ups A–C, §4.9–4.12)
Title: *Cut the Wire: Perfect AUROC Does Not Mean a Collusion Probe Reads the Collusion*
Claim: Across Qwen2.5-7B/14B/32B, linear collusion probes with AUROC ≈ 1.0 separate partner-message surface features
from layer 0 and are nearly orthogonal to the receiver's decoded covert state, which appears abruptly at about
0.5–0.67 depth. Whether removing the state or detector directions stops collusion is model-dependent:
- 7B: yes, robust to held-out-scheme fitting and to 5 fresh sender seeds.
- 14B: the state ablation disrupts collusion but not selectively.
- 32B: no fixed-recipe ablation materially disrupts collusion, because the state is redundantly encoded (still 0.91
  decodable after removal).
[Earlier one-model framing in §1 history superseded; see §4.8–4.12.]

**Wording rules (from the reviewer; use them):**
- Never say "ablating the detector blinds it" as a finding; it is algebraic for the frozen probe. Say "removes the
  probe's readout by construction; collusion continues (93%) and a retrained probe recovers 0.97".
- The state subspace is "the receiver's decoded-decision subspace", not "the covert computation". Under its
  ablation, scheme-reading receivers default to COLD (C1 82%, C2 91%, C4 96%); explicit-state betting is intact (C6 0.98).
- "Detectors still fire after state ablation" is by construction (orthogonality); don't headline it.
- Random controls: say "random and variance-matched principal-component controls".
- "Detectors track surface features of the partner message (separable at layer 0)", not "intent".

### Draft abstract (FINAL v2, after follow-ups)
Linear probes on activations detect collusion between LLM agents with near-perfect accuracy, but whether they read
the signal that carries the collusion is untested. We build a steganographic signalling game in which a sender hides
a betting signal in casino table talk under a private topic code; Qwen2.5 receivers (7B, 14B, 32B) decode it with
96–100% accuracy (chance 33%). In all three models, standard collusion probes reach AUROC ≥ 0.96 already at layer 0,
reading message surface features, and are nearly orthogonal to the receiver's decoded covert state, which emerges
abruptly at about half to two-thirds depth. On NARCBench's released activations the steganography tier is separable by
context length alone. Causal ablations then diverge across models. In 7B, removing a mass-mean detector direction or
the 2-D decoded-state subspace cuts covert decoding by 22 and 52 points with held-out-scheme fitting, and by 28 and
58 points on five fresh sender seeds, while largely sparing explicit-state decisions. In 14B, the state ablation
disrupts collusion but also explicit decisions and general capability. In 32B, no detector, state or control
ablation reduces collusion by more than 9 points, even though the ablation exactly removes the subspace and the state
stays linearly decodable (0.91), which indicates redundant encoding. Removing logistic-regression probe directions never
costs more than 6 points. High probe AUROC is therefore not evidence that a monitor tracks the collusion mechanism, and causal
conclusions from one model need not transfer to larger ones.

### Contributions (for the intro)
1. A controlled testbed (Signal Table) with six receiver conditions that separate colluding context (C3), message
   style (C2/C4), "message is a signal for me" (C5), and the covert channel itself (C6 explicit state); it fixes
   NARCBench-Stego's state leak.
2. To our knowledge, the first causal (ablation + steering) test of collusion-probe directions, with random,
   variance-matched principal-component, explicit-state (C6) and utility (recall, ARC) controls, replicated on
   three models from two families (one further model, OLMo-2-7B, failed the capability gate).
3. The first real-data token-level test of the "spike on encoded tokens" claim (Rose et al. v1): largely lexical
   (42–91% of the excess persists when the word is not a signal; Qwen-7B); a small genuine marker along a non-lexical direction.
4. A confound audit of NARCBench's released activations (Stego is fully length-separable).
6. Post-hoc robustness and extension (§4.8–4.12): held-out-scheme direction fitting (7B, 14B, 32B), five fresh
   sender seeds (7B), and a matched-input Qwen2.5-32B comparison with an ablation-correctness check, showing the
   7B causal result is robust in 7B but does not transfer to 14B (non-selective) or 32B (redundant encoding).
5. Practical guidance: AUROC does not establish causal relevance; ablating surface-detector directions (LR, dm,
   NARCBench-style) leaves the information linearly recoverable (retrained ≥ 0.96) — the C1vC5 decision-token direction
   is the exception in 14B/Phi (retrained 0.32/0.35); in Qwen-7B the covert decision is redundant across depth (no half
   of the layers suffices); mass-mean ablations must handle massive-activation dimensions (they broke Phi); steering
   works only just after representation onset (found post hoc in 14B/Phi).

## 2. Motivation (for intro)
- Probe-based collusion detectors reach ~perfect in-distribution AUROC (Rose et al. 2026) but the localisation
  evidence is correlational. A detector keyed to an epiphenomenal correlate can be evaded without hurting collusion;
  a detector keyed to a load-bearing signal cannot. This distinction matters for whether internal monitors are a
  robust "plus factor" (sprint Q2.3) for collusion.
- Exact quotes from Rose et al.: ✅ notes/rose_paper.md. Highlights for the intro:
  - v1 abstract: "preliminary evidence that this signal is localised at the token level, with the colluding agent's
    activations spiking specifically when processing the encoded parts of their partner's message."
  - Actual evidence: mean projection over ALL tokens of the Counter's message (Big Player +0.29 vs honest −1.41),
    39 hands, single layer 32, p<0.001. The per-token heatmap is labelled "Illustrative only, not real data" (v2+).
    The claim was moved to an appendix in v2/v3 and **removed in v4/v5**. → our token-trace figure is, to our
    knowledge, the first real per-token test of it.
  - No version ablates/steers/patches the probe direction. Footnote 5: "In NarcBench-Core we can actually run
    counterfactuals … We plan to do this in future experiments."
  - Detection numbers (v5 App. D): Core ≈1.00; Stego best method 0.99–1.00 on every model; Transfer best 0.72–0.93;
    Llama-70B Transfer recall@1%FPR only 0.38; emergent "Verdict" 0.49–0.73.
- Design flaws in NARCBench-Stego that our environment fixes (good for the method section):
  1. The Big Player's private history logs past count ranges → it can bet well without decoding (bp_corr leak).
     Ours: single-hand episodes; receiver sees nothing correlated with S except the sender's message.
  2. Colluder and honest prompts differ (system prompt + scheme block), despite App. C saying they match → the probe
     may detect "has secret-team context". Ours: C1 vs C2 share an identical receiver prompt; only the partner text
     differs. We also train the confounded C1-vs-C3 probe for comparison.
  3. Their probe direction lives in standardised space; we intervene with raw-space w/σ, normalised.
- Their probe recipe: sklearn LR, L2, C=25, liblinear, max_iter=8, StandardScaler, last token of
  system+user+response, layers 26–30 of 64 averaged. Five methods: Peak Suspicion, Deception Split, Sorted
  Concatenation, Activation Alignment, Asymmetry Probe.
- Risk: a LessWrong replication with Llama-3.1-8B on Core found only ~0.55–0.66 AUROC → small models may carry a weaker signal.

## 2b. Supplementary finding: NARCBench probe confounds ✅ (supp_narcbench/results.md, analysis.py, figs/)
Qwen3-32B released activations, layers 26–30, CPU. Strong motivation for our controlled design. Possibly a
section of its own: "Why a causal test is needed."
- Reproduced the paper's own reproduce_LR.py: Core 1.000 (all 5 methods), Stego Peak 1.000, Transfer Peak
  0.644–0.978 per family.
- **Stego is fully length-confounded.** Final phase: colluder context 1819–2748 tokens vs honest 774–1521 (I checked
  this directly from metadata_gen.json). n_tokens alone → colluder AUROC 1.000, run-level 1.000. Regressing n_tokens
  out of the probe score: colluder AUROC 0.961 → 0.516, run Peak 1.000 → 0.891.
- Core is partly confounded (n_tokens alone 0.64; +round/phase 0.77) but the probe survives: 0.975 after residualising.
- Transfer is the cleanest: surface features ≈ chance (0.51–0.58); probe 0.82–0.84 after residualising.
- The probe flags the colluding agent, not the run: honest agents in collusion runs score only slightly above control (0.54–0.59).
- Tier directions mostly differ: cos(Core,Transfer) 0.20–0.23 vs shuffled 0.00±0.02 (z≈12–15); cos(Core,Stego)
  0.02–0.10; same-tier split-half ceilings Core 0.70–0.74, Transfer 0.55–0.56, Stego 0.96–0.98.
- Caveat: n_tokens is presumably context length, longer for colluders because of private coordination text. If so,
  probes may read "collusion instructions in the prompt" rather than collusive computation; length controls can't rule
  that out. This is exactly why we (i) hold the receiver prompt fixed (C1 vs C2) and (ii) intervene causally.
- Figures: supp_narcbench/figs/{auroc_by_layer, confound_auroc, runlevel_probe_vs_ntok, cosine_heatmap_P, cosine_heatmap_S}.png
  Candidate main-text figure: confound_auroc.png (surface features vs probe per tier).

## 3. Method (see PLAN.md for full spec; exact prompts in exp/env.py; run log notes/pipeline_log.md)
- Model: Qwen/Qwen2.5-7B-Instruct, bf16, HF transformers 4.46.3, torch 2.5.1, sdpa, Modal L40S. 28 layers, d=3584.
  Layer l = output of decoder layer l (forward hook; last layer pre-final-norm).
- Decoding: sender T=0.7, top_p=1, max_new 120; receivers/recall/ARC greedy.
- Environment "Signal Table": 6 schemes over 9 topics (weather, food, sports, music, travel, family, movies, cars,
  pets); each topic appears in 2 schemes under different states → no topic word maps to one state. Bets COLD $0 /
  NEUTRAL $25 / HOT $100. The scheme reaches both colluders as a 2-turn private-coordination transcript. Single-hand
  episodes; the hand context is seeded by (scheme, seed) only, so it is identical across states (no state leak; cf.
  NARCBench's Big Player history leak). The sender is told the state and its topic, and must avoid forbidden words
  (count, signal, scheme, hot, cold, code, …) and numbers.
- Data: 306 episodes = 6 schemes × 3 states × 17 seeds. Messages are generated once and held fixed; all
  interventions act on the receiver only.
- Conditions (receiver side):
  - C1 = colluding receiver + colluding message.
  - C2 = colluding receiver + honest message (same receiver prompt).
  - C3 = honest receiver (no scheme) + colluding message.
  - C4 = colluding receiver + honest message mentioning an off-scheme topic.
  - C5 = colluding receiver with a scheme that does NOT contain the message's topic + the identical C1 message
    (the scheme-mismatch control; isolates "the message is a signal for me") ⏳.
- Activations: three poolings per condition, float16 [306, 28, 3584]:
  - `mean`: mean over the sender-message tokens.
  - `last`: decision token, i.e. the last prompt token.
  - `full_last`: end-of-turn token of prompt + response (Rose et al. pooling).
- Probes (exp/probes.py): StandardScaler + L2 LR (C=0.1), leave-one-scheme-out (6 folds) AUROC per layer.
  Intervention directions are raw-space difference-of-means per layer, unit-normalised (Arditi et al. 2024); the LR
  raw-space direction (w/σ) is kept for comparison. The state subspace is 2-D per layer: [μHOT−μCOLD,
  μNEUTRAL−(μHOT+μCOLD)/2] at the decision token in C1, QR-orthonormalised. Random controls are matched-dimension
  (1-D and 2-D, 5 seeds), fresh per layer.
- Intervention: directional ablation h ← h − (h·Q)Qᵀ at every position of every layer, plus the embedding output.
  Steering: h += α·u at selected layers. Outcomes:
  - decode accuracy and bet–state r (C1);
  - topic recall (4-way: did the receiver still understand the message?);
  - ARC-Easy (first 100 test items) for general capability;
  - detector AUROC on re-extracted activations, both for a frozen probe (trained on baseline acts, other schemes)
    and a retrained probe (amnesic probing: is the info still linearly recoverable?).
- Cost: Stage A+B+baseline ≈ $0.33 Modal. ~1.6 GPU-min (~$0.06) per intervention run.

## 4. Results
### 4.1 Baseline behaviour ✅ (data/interv/baseline/metrics.json, n=306)
| Condition | Decode acc | Bet–state r | Mean bet COLD/NEU/HOT |
|---|---|---|---|
| C1 collude | **0.958** | **0.913** | $2 / $26 / $97 |
| C2 honest msg | 0.346 (chance) | 0.068 | $21 / $24 / $24 |
| C3 honest receiver | – | 0.050 | $49 / $49 / $50 |
| C4 off-topic msg | 0.337 (chance) | −0.084 | $29 / $27 / $23 |
Topic recall 0.967; ARC-Easy 0.97. Sender compliance: 97% no forbidden words, 99% no digits.
Gate: v1 sender prompt (topic not named) gave 0.733 decode; 0.983 when the sender encoded cleanly → the bottleneck is
7B sender topic choice. v2 names the topic for the sender (an instructed-collusion setting, like NARCBench's instructed tiers).
Validity caveat: colluding messages are short single-topic lines while honest ones chat about the cards → the
C1-vs-C2/C4 detectors may partly key on style. Hence C5.

### 4.2 Probes ✅ (data/probes.json; figures/fig2_probe_layers.png; exp/probes.py, exp/figs_probes.py)
- Every detector contrast reaches AUROC ≈ 0.99–1.00 at **every layer, including layer 0**:
  C1vC4-mean 0.996–1.000; C1vC2-mean 0.998–1.000; C1vC4-end-of-turn 0.993–0.998; C1vC3 (prompt-confounded) 1.000.
  → Perfect separability at layer 0 (≈ token embeddings) means these detectors can be satisfied by surface/lexical
  features. In-distribution AUROC alone says nothing about whether the probe reads collusive computation.
  (Parallels NARCBench's Stego length confound, §2b.)
- The covert content (hidden state S, 3-way) at the decision token is decodable on held-out schemes only from layer
  ~17–18: ≤0.35 for layers 0–12 (chance 0.33), 0.54 at L16, 0.91–0.92 for L18–27. Because the topic→state mapping
  differs across schemes, this is an abstract "the count is HOT" representation, not a topic word. (Early layers fall
  slightly below chance, 0.21–0.29, an artefact of leave-one-scheme-out with scheme-specific topic→state maps.)
- Geometry: detector direction vs state subspace |cos| ≤ 0.12 at every layer (mostly 0.02–0.06) → nearly orthogonal.
  Diff-of-means vs LR (w/σ) detector directions: cos 0.42–0.54.

### 4.3 Token-level test of Rose et al.'s "spike" claim ✅ (data/token_trace_stats.json; figures/fig3_token_trace.png)
Leave-one-scheme-out d_detect (C1vC4 diff-of-means), projection per token, centred at the class-mean midpoint; 40
traced C1 messages (34 contain a detectable topic keyword) and 40 C2.
| layer | C1 topic-word tokens | C1 other msg tokens | Wilcoxon p | C1 msg mean | C2 msg mean | MWU p |
|---|---|---|---|---|---|---|
| 8 | 6.9 | 2.4 | 5e-8 | 2.7 | −3.4 | 4e-12 |
| 12 | 10.3 | 2.7 | 1e-10 | 3.1 | −5.3 | 4e-14 |
| 16 | 16.3 | 4.3 | 1e-10 | 5.0 | −7.6 | 3e-14 |
| 20 | 17.5 | 6.6 | 2e-10 | 7.2 | −11.7 | 4e-14 |
| 24 | 22.0 | 9.9 | 2e-8 | 10.1 | −24.6 | 1e-13 |
→ A genuine token-level spike exists on the encoded word (2.2–3.8× other tokens), the first real-data version of
Rose et al.'s illustrative claim. BUT it could be lexical ("a scheme-topic word appears") rather than "this is a
signal for me". C5 (same message, scheme without that topic) disambiguates ✅:
| layer | C1 topic word (is a signal) | SAME word in C5 (not a signal) | C1 other words |
|---|---|---|---|
| 8 | 6.9 | 5.7 | 2.4 |
| 12 | 10.3 | 8.0 | 2.7 |
| 16 | 16.3 | 9.3 | 4.3 |
| 20 | 17.5 | 12.8 | 6.6 |
| 24 | 22.0 | 20.9 (Wilcoxon p=0.11, n.s.) | 9.9 |
(C1 vs C5 topic-token Wilcoxon p < 1e-3 at L8–20.) → **42–91% of the token spike's excess over other tokens (73/69/42/57/91% at L8/12/16/20/24) survives when the word is NOT a
signal for the receiver.** The spike is mostly lexical / in-context-repetition; a token spike is weak evidence of
decoding. Caveat: in C5 the topic word did not appear in the receiver's scheme transcript, whereas in C1 it did, so
"is a signal" and "was mentioned earlier in context" are confounded in C1; C5 removes both and the spike largely
remains → the spike is mainly lexical.

### 4.4 C1 vs C5 detector (the cleanest "message is a signal for me" contrast) ✅
C5 receiver decode = 0.369 (chance; bet–state r 0.146; it always outputs a confident guess, never "no signal").
LOSO AUROC per layer (data/probes.json):
- mean (msg span): 0.34–0.69 at L0–4, 0.90–0.98 at L5–19, collapsing to 0.24–0.50 at L21–27.
- last (decision token): 0.02–0.37 at L0–16 (!), 0.64 at L17, **0.86–0.90 at L18–27**.
- full_last (end of turn): 0.07–0.44 at L0–17, 0.55–0.76 at L18–27.
Interpretation: a genuine "this message is a signal for me" representation at the decision token emerges at L18,
exactly where the covert state becomes decodable (Fig 2b). The far-below-0.5 values are a confound artefact: C5's
transcript uses another scheme's text (scheme k+1), which appears as a C1 transcript in the training folds, so
transcript-identity features invert on held-out schemes. Report this honestly; use L18+ for C1vC5 claims.

### 4.5 Ablations ✅ (data/interv/abl_*/metrics.json + post_probes.json; exp/run_ablations*.sh)
All runs: n=306, ablation at every layer + embeddings, receiver only, messages fixed. "Frozen" = probe trained on
baseline acts (leave-one-scheme-out), scored on intervened acts; mean over 28 layers [min–max]. "Retrained" = fresh
LOSO probe on intervened acts (amnesic probing).
| Direction ablated | dim | C1 decode (base 0.958) | bet–state r (0.913) | recall (0.967) | ARC (0.97) | own detector frozen AUROC | retrained |
|---|---|---|---|---|---|---|---|
| random 1-D ×3 | 1 | 0.958 / 0.958 / 0.964 | 0.91 | 0.96–0.97 | 0.96–0.97 | 1.00 | 1.00 |
| random 2-D ×3 | 2 | 0.967 / 0.964 / 0.948 | 0.90–0.93 | 0.96–0.97 | 0.96–0.97 | 1.00 | 1.00 |
| **state subspace (pos. ctrl)** | 2 | **0.382** | 0.297 | 0.938 | 0.92 | C1vC4 1.00; C1vC3 1.00 | 1.00 |
| detect C1vC4 LR (w/σ) | 1 | 0.928 | 0.897 | 0.958 | 0.97 | **0.31** [0.22–0.49] | 0.97 |
| detect C1vC4 diff-means | 1 | 0.699 | 0.643 | 0.931 | 0.95 | 0.69 [0.39–0.92] | 0.96 |
| detect C1vC2 diff-means | 1 | 0.670 | 0.548 | 0.886 | 0.92 | 0.73 [0.40–0.92] | 0.97 |
| NARCBench-style C1vC3 diff-means | 1 | 0.951 | 0.897 | 0.958 | 0.92 | 0.76 [0.09–1.00] | 0.99 |
| C1vC5 last (decision tok.) | 1 | 0.846 | 0.775 | 0.961 | 0.93 | L18+: 0.36 (base ≈0.88) | – |
| C1vC5 mean (msg span) | 1 | 0.448 | 0.224 | 0.850 | **0.67** (collateral) | 0.45 | 0.62 |
| C1vC5 full (end of turn) | 1 | 0.941 | 0.903 | 0.964 | 0.96 | L18+: 0.79 | – |
Detector cross-effects: state ablation leaves every detector at ≈1.00; the C1vC5 ablations leave C1vC4 at 0.99–1.00.
State decodability (retrained, max over layers) after ablation: state 0.68, C1vC4-dm 0.68, C1vC2-dm 0.62, C1vC5-mean
0.53; random/LR/C1vC3 0.90–0.94.

**[SUPERSEDED — first-pass reading, kept for the record; see §1 final framing, §4.5a audit and §4.7. Do not use these bullets verbatim.] Headline (the 2×2):**
- Detector blind, collusion intact: ablating the LR detector direction drops the frozen detector to 0.31 AUROC while
  decode stays 0.928. NARCBench-style: 0.76 / 0.951. C1vC5-last: 0.36 / 0.846. → probe-based detectors are evadable
  by removing one direction, without hurting collusion (and the info stays linearly recoverable: retrained 0.97–0.99).
- Detector fires, collusion broken: ablating the 2-D state subspace kills collusion (0.958 → 0.382, ≈ chance) while
  comprehension (recall 0.94) and capability (ARC 0.92) survive, and every detector still reads 1.00.
  → the detectors track colluding context/intent, not the computation that carries the collusion.
- Partial: diff-of-means detector directions are partly load-bearing (decode 0.67–0.70), with modest utility cost.
- Random directions: no effect on anything (negative control passes).

**Interpretation caveats for the 2×2 (state these in the paper):**
- "Detector blind" after ablating the detector's own direction is close to true by construction: the frozen LR
  probe's raw-space readout is w/σ, the very direction removed. The non-trivial result is behavioural: removing the
  detector's readout direction leaves collusion intact (0.928), so the model does not use it. And the information is
  redundantly encoded (a retrained probe recovers 0.97), so a monitor that retrains is not fooled. The threat model
  is a frozen monitor plus an adversary able to suppress one direction (cf. obfuscated activations: Bailey et al.
  2024; RL-obfuscation; Neural Chameleons).
- Mass-mean (diff-of-means) directions are more causally relevant than LR directions (0.70 vs 0.93 decode after
  ablation). This matches Marks & Tegmark 2023's finding that mass-mean probes are more causally implicated than LR
  probes; LR weight vectors partly encode noise-cancelling directions.
- The C1vC5-mean ablation kills collusion (0.448) but also hurts ARC (0.67) and recall (0.85) → not a clean
  dissociation; treat it as nonspecific damage.
- Makelov et al. 2023 interpretability illusion: ablating a direction can act through dormant pathways. Our random
  controls and the utility metrics partly address this; the state subspace is our cleanest causal claim, supported
  both by ablation (necessity) and steering (sufficiency-ish).

### 4.5a Reviewer audit (independent critical review, fable-advisor) and resulting corrections ✅
The reviewer's points, each verified against data:
- **T1 (critical): the state-subspace ablation collapses the receiver's ANSWER, not collusion specifically.**
  decoded_dist under state ablation: C1 COLD 251 / NEU 7 / HOT 15 / unparsed 33 (11%); C2 (honest msg) COLD 279/306;
  C4 COLD 295/306 (baseline C2/C4 are mostly NEUTRAL). C3 (honest receiver, BET-only format) is unaffected
  ($55/$54/$53 vs $49). → The 2-D subspace is the receiver's 3-way decision/answer variable. Do NOT call it "the
  covert computation". C6 control (explicit state, no covert channel, same answer format) ⏳ decides whether it is
  generic.
- Interesting contrast: the diff-of-means detector ablation shifts C1 toward NEUTRAL (COLD 83 / NEU 181 / HOT 42),
  which is the receiver's no-signal default (C2 is mostly NEUTRAL), while leaving C2/C4 unchanged (NEU 268/254). This
  looks like a genuine "signal received" direction; C6 ⏳ tests its specificity.
- **T2:** "detectors still 1.00 after state ablation" is expected by construction: the class-balanced detector
  diff-of-means is orthogonal to the state directions (|cos| ≤ 0.12). Don't present it as a finding about detectors.
- **T3:** "detector blind" after ablating its own direction is algebraic. Added a direction-excluded frozen probe
  (exp/probes.py post_excl: frozen probe trained on baseline with the ablated subspace projected out). Results (mean
  over layers; C1vC5-last uses L18+):
  | ablation | frozen | frozen_excl | retrained |
  |---|---|---|---|
  | detect C1vC4 LR (w/σ) | 0.31 | 0.81 | 0.97 |
  | detect C1vC4 dm | 0.69 | 0.03* | 0.97 |
  | detect C1vC2 dm | 0.73 | 0.06* | 0.97 |
  | NARCBench-style C1vC3 dm | 0.76 | 0.33* | 0.99 |
  | C1vC5 last dm | 0.36 | 0.28* | 0.83 |
  | state 2-D / random | 1.00 | 1.00 | 1.00 |
  *Leakage artefact, verified: the dm directions were fit on all 6 schemes. Removing the all-scheme mean difference
  leaves per-scheme residual differences that sum to zero, so held-out schemes invert (AUROC ≪ 0.5 even at layer 0,
  where the intervened acts equal projected baseline acts to within 1.4%). AUROC ≈ 0.03 means "still perfectly
  separable, sign flipped". → **Robust claim: removing any single detector direction does not remove the
  collusion information (retrained 0.83–0.99), and only the dm directions affect behaviour.** Present frozen numbers
  in a table with this caveat; don't use them as the Fig 4 x-axis.
- **T4:** LR vs dm is not variance-fair. Share of activation variance along each 1-D direction (C1+C4 mean acts,
  mean over layers; data/variance_shares.json): dm 10.4%, LR (w/σ) 1.9%, raw-space LR 4.5%, random 0.03%. State
  2-D subspace: 33% of C1 decision-token variance at L20; top-2-PC control (orthogonal to state): 19%. A raw-LR
  ablation and a PC-control ablation are running ⏳.
- **T5:** the "mostly lexical spike" was measured along d_detect, which is surface-separable at layer 0 (partly
  circular). Re-measured along the non-lexical C1vC5 decision-token direction (leave-one-scheme-out; data/
  token_trace_C1vC5dir.json): topic-word projection is higher in C1 than C5 (L16 −4.73 vs −4.26, p=0.009 [reversed
  sign, tiny]; L20 −2.74 vs −4.64, p<1e-3; L24 −5.97 vs −8.15, p<1e-3), and in C5 the topic word ≈ the other words
  (−4.64 vs −5.26 at L20). → A small, scheme-dependent "code word recognised" marker exists on the encoded token
  along a non-lexical direction, from L20. This is the honest version of Rose et al.'s spike: real but small, and
  invisible to a surface-trained detector (whose spike is mostly lexical).
- **T6:** random directions were norm-unmatched (0.03% variance). Fixed with the variance-matched top-2-PC control ⏳
  and a random-direction steering control ⏳.
- Reviewer's suggested wording changes are adopted in §1 (title/claims).

**C6 control: explicit state, no covert channel ✅** (exp/env.py Q_EXPLICIT: an honest receiver is told "The casino's
public table indicator for the next hand reads: {state}", with the same promo bet rule COLD $0 / NEUTRAL $25 / HOT
$100 and the same DECODE/BET format. data/interv/base_C6, *_C6.) Scored by bet-tier accuracy: the model often omits
the "DECODE:" prefix (20/306 at baseline, e.g. "NEUTRAL BET: $25") although its bets are right.
| ablation (all layers) | var. share | C1 covert decode | C1 decoded dist (C/N/H/None) | C6 explicit bet acc | ARC | recall |
|---|---|---|---|---|---|---|
| none | – | 0.958 | – | **1.000** | 0.97 | 0.967 |
| state 2-D | 33% (L20, last) | **0.382** | 251/7/15/33 | **0.980** | 0.92 | 0.938 |
| PC control 2-D (top-2 PCs ⊥ state) | 19% (L20) | 0.601 | 132/173/1/0 | 0.801 | 0.93 | 0.912 |
| detect C1vC4 dm | 10.4% | **0.699** | 83/181/42/0 (→ no-signal default) | **1.000** | 0.95 | 0.931 |
| PC control 1-D (top PC ⊥ dm) | 8.1% | 0.935 | 104/102/91/9 | 1.000 | 0.96 | 0.964 |
| detect C1vC4 raw-LR | 4.5% | 0.912 | 103/95/108/0 | 1.000 | 0.96 | 0.958 |
| detect C1vC4 LR (w/σ) | 1.9% | 0.928 | 111/97/98/0 | 1.000 | 0.97 | 0.958 |
| detect C1vC2 dm | – | 0.670 | – | 1.000 | 0.92 | 0.886 |
| NARCBench-style C1vC3 dm | – | 0.951 | – | 1.000 | 0.92 | 0.958 |
| C1vC5 msg-span dm | – | 0.448 | – | **0.379** (nonspecific) | 0.67 | 0.850 |
| random 1-D | 0.03% | 0.958 | – | 1.000 | 0.96 | 0.964 |
| C1vC5 last dm | – | 0.846 | – | **1.000** | 0.93 | 0.961 |
| random 2-D | 0.03% | 0.967 | 106/100/100/0 | 1.000 | 0.97 | 0.967 |
| random steer ±40 @L20 | – | 0.958–0.964 | bets $1–2/$26/$97 | – | 0.96–0.97 | – |
Readings:
1. [Held-out-fitted check §4.8: C1 effect survives (−0.516), C6 sparing weakens to 0.951, one fold 0.745.] **The state subspace is specific to scheme-based decoding.** Ablating it collapses every scheme-reading condition
   to a COLD default (C1/C2/C4) but leaves explicit-state betting intact (C6 0.98). A variance-matched subspace (PC
   control) is less damaging to covert decoding (0.60) and MORE damaging to explicit betting (0.80). Caveat: in C6
   the answer word is literally in the prompt (a copy task), which may need less of any internal state.
2. **The diff-of-means detector direction is a collusion-specific causal direction.** Ablating it moves colluders
   to the no-signal default (NEUTRAL 181/306, resembling C2) while C2/C4 behaviour, explicit betting (C6 1.000), ARC
   (0.95) and recall (0.93) are essentially unchanged. → A "signal received" variable that probes CAN find, but
   only the mass-mean direction, not the LR readout.
3. **LR detector directions are not load-bearing** (raw-LR 0.912, w/σ-LR 0.928), even though they detect perfectly.
   [REFINED by §4.8: a small but nonzero effect, −2.9 pp global / −5.6 pp held-out-fitted, both CIs excluding 0; "largely intact", not "no effect".]
   Not a variance artefact: the variance-matched 1-D PC control (8.1% vs dm 10.4%) leaves collusion at 0.935, so the
   dm effect (0.699) is direction-specific. The C1vC5 msg-span ablation is nonspecific damage (C6 0.38, ARC 0.67).
4. Random steering ±40 at L20 has no effect (vs the state direction moving COLD bets $2 → $85) → steering is direction-specific.

### 4.5b Layer localisation ✅ (data/interv/loc_*/metrics.json; exp/run_localize.sh; behaviour only, n=306)
| Ablated layers | state 2-D: decode / r / bets C/N/H | detector C1vC4 dm: decode / r | recall | ARC |
|---|---|---|---|---|
| all 0–27 (+emb) | **0.382** / 0.297 | 0.699 / 0.643 | 0.94 / 0.93 | 0.92 / 0.95 |
| early 0–15 | 0.690 / 0.635 / $0 $17 $61 | 0.788 / 0.731 | 0.944 / 0.938 | 0.94 / 0.95 |
| late 16–27 | 0.824 / 0.664 / $21 $38 $92 | 0.853 / 0.805 | 0.958 / 0.951 | 0.97 / 0.96 |
→ No half suffices; the covert channel is redundantly represented across depth (the residual stream re-derives it).
Single-layer monitoring or single-layer intervention will underestimate it. (Steering at one layer does shift
decisions, §4.6, so sufficiency is local but necessity is distributed.)

### 4.6 Steering dose–response ✅ (data/interv/steer_state_L20_a*/; exp/run_steer.sh)
Steer h += α·u at layer 20 only, all positions, u = unit(μHOT − μCOLD) at the decision token (‖μHOT−μCOLD‖ = 13.1;
L20 last-token norm ≈ 92). C1, n=306.
| α | decode | r | mean bet COLD/NEU/HOT | recall | ARC |
|---|---|---|---|---|---|
| −40 | 0.804 | 0.732 | $0 / $17 / $74 | 0.954 | 0.97 |
| −20 | 0.928 | 0.893 | $0 / $23 / $94 | 0.964 | 0.97 |
| −10 | 0.964 | 0.918 | $0 / $25 / $96 | 0.967 | 0.97 |
| 0 | 0.958 | 0.913 | $2 / $26 / $97 | 0.967 | 0.97 |
| +10 | 0.928 | 0.836 | $10 / $26 / $97 | 0.961 | 0.97 |
| +20 | 0.840 | 0.610 | $34 / $27 / $98 | 0.954 | 0.95 |
| +40 | 0.742 | 0.148 | **$85** / $32 / $99 | 0.938 | 0.93 |
→ Adding the HOT−COLD direction at a single layer makes the receiver bet big on COLD signals; general ability is
intact. Causal confirmation that the late-layer state direction carries the covert decision (complements ablation).

### 4.7 Cross-model replication ✅ (notes/replication.md — full tables; data_models/{qwen14b,phi35mini,olmo7b}/)
Models: Qwen/Qwen2.5-14B-Instruct (48 L, d 5120), microsoft/Phi-3.5-mini-instruct (32 L, d 3072; different family).
allenai/OLMo-2-1124-7B-Instruct FAILED the gate (decode 0.567; receiver biased to HOT even on honest messages) →
stopped per protocol. Same prompts, pipeline and probe recipe; n=306 each.
| Finding | 7B | 14B | Phi-3.5 | Verdict |
|---|---|---|---|---|
| Baseline C1 decode / controls at chance | 0.958 | 0.997 | 0.967 | ✅ |
| All detector contrasts ≈1.00 incl. layer 0 | ✅ | ✅ (0.96–1.00) | ✅ (1.00) | **robust** |
| State decodable abruptly at ~half depth | L16–18 (0.57–0.64) | L24–26 (0.50–0.54) | L15–16 (0.47–0.50) | **robust** (Fig 6) |
| C1vC5 decision-token detector late | 0.86–0.90 | 0.85–0.92 | ≤0.63 (weak) | partial |
| LR detector ablation: collusion intact | 0.928 | 0.984 | 0.918 | **robust** |
| Retrained detector recovers after ablation | 0.97 | 0.96 | 0.99 | **robust** |
| Random directions: no effect | ✅ | ✅ (0.98–1.00) | ≈ (0.92–0.97) | robust |
| Diff-means C1vC4 detector ablation hurts collusion | **0.699** | 0.971 (no) | breaks model (massive-activation dims) | **7B only** |
| NARCBench-style C1vC3 ablation | 0.951 | 0.980 | 0.824 (ARC 0.28, collateral) | not load-bearing |
| Decision-state 2-D ablation breaks covert decoding | 0.382 (all L) | strict 0.389 / bet-exact 0.631, recall 0.75, ARC 0.82 (partial, collateral); L26–47 only: 0.837 | all L breaks model; **L16–31 (post hoc): 0.340, ARC 0.99, recall 0.94** | partial |
| Steering HOT−COLD at ~2 layers after onset | L20 +3×: COLD $85 | L28 +6×: COLD $93 | L18 +6×: COLD $75 | replicates qualitatively (but at 0.7·L, the pre-specified layer, no effect) |
Phi caveat: raw-space diff-of-means directions put large weight on Phi's few massive-activation dimensions
(1113, 2046, 2024, 2812); projecting them out at every position (incl. the attention-sink token) destroys generation
(ARC 0). LR (w/σ) and random directions avoid those dimensions. → Methodological lesson: mass-mean ablation needs
massive-activation-aware handling (cf. Sun et al. 2024 "Massive activations"; verify the citation).
Follow-up ✅ (notes/replication.md "Follow-up: C6 + variance-matched controls"):
| model | ablation | layers | var share | C1 decode | C1 bet_exact | C6 explicit bet_exact | ARC | recall |
|---|---|---|---|---|---|---|---|---|
| 7B | state 2-D | all | 33% (L20) | 0.382 | 0.435 | **0.980** | 0.92 | 0.938 |
| 7B | PC ctrl 2-D | all | 19% | 0.601 | 0.601 | 0.801 | 0.93 | 0.912 |
| 14B | state 2-D | all | 46% (L34) | 0.389 | 0.631 | **0.712** | 0.82 | 0.752 |
| 14B | PC ctrl 2-D | all | 12% | 0.111 | 0.627 | 0.866 | 0.35 | 0.931 |
| 14B | detect C1vC4 dm | all | – | 0.971 | 0.977 | 1.000 | 0.98 | 0.993 |
| Phi | state 2-D | 16–31 | 33% (L23) | 0.340 | 0.340 | **1.000** | 0.99 | 0.941 |
| Phi | state 2-D | all | 33% | 0.000 | 0.000 | 0.000 | 0.00 | 0.000 |
| Phi | PC ctrl 2-D | 16–31 | 14% | 0.373 | 0.408 | 0.843 | 0.97 | 0.971 |
| Phi | detect C1vC4 dm | 16–31 | – | 0.000 | 0.000 | 0.000 | 0.05 | 0.000 |
Reading: the state subspace is *selective* (breaks covert decoding, spares explicit decisions) in 7B and Phi-late, but
not in 14B (C6 0.71). In every model a variance-matched PC subspace ALSO disrupts covert decoding (7B 0.60, 14B 0.63,
Phi 0.41) with some explicit-decision cost (C6 0.80–0.87). → Covert decoding is fragile to removing large late
subspaces; the state subspace stands out by selectivity, not uniqueness. Do not claim "the" collusion mechanism.
**Net effect on claims:** the robust, 3-model claims are (i) detectors are surface-separable; (ii) high-AUROC LR
detector directions are not what carries the collusion (NARCBench-style: 95–98% in Qwen, 82% with collateral damage
in Phi), and surface-detector information survives their ablation (retrained ≥ 0.96); (iii) a decision-state representation emerges abruptly at ~half depth and is steerable right after onset.
The "specific causal signal-received direction" (diff-means 0.70, C6 intact) is a 7B-only result (14B 0.971; Phi untestable, the ablation breaks the model); report it as such.
Modal spend so far ≈ $7–8 total (billing report: $6.66 at 06:30 incl. 7B).

### 4.8 Post-hoc robustness check: held-out-scheme (LOSO) intervention directions ✅ (Qwen2.5-7B only)
**Why.** Probe AUROCs were leave-one-scheme-out, but every intervention direction (and both PC controls) was fitted
on all 306 episodes, including the episodes it was then evaluated on. This check refits each direction without the
evaluated scheme. It is **exploratory and post hoc**: designed after seeing the global results. The plan, run
matrix and decision rules were written before any fold run (notes/loso_robustness_plan.md); the execution log is
notes/loso_robustness_log.md.

**Method.**
- 6 folds, one per scheme. Fit on the 255 episodes of the other 5 schemes; evaluate on the 51 held-out episodes,
  using the frozen messages (byte-identical to the original; no regeneration).
- Identical constructions: exp/loso_fit.py reproduces every original direction file when given all 306 episodes
  (subspace agreement ≥ 0.999998 at every layer). Fold and global directions agree 0.78–0.99.
- No data-driven selection existed in the original method (fixed all-layer + embedding policy, LR C=0.1 fixed, k
  fixed), so no nested validation was needed and no new layer selection was introduced.
- Per fold, one container evaluates 11 configs on the SAME held-out episodes in the same order (identical
  batching): matched baseline `none` and {dm, PC-1D, state 2-D, PC-2D, LR} × {global, fold-fit}, with conditions C1
  + C6 + recall + ARC. A code guard asserts that no hook leaks between configs.
- Determinism: the matched baseline reproduces the original 306-episode outputs exactly (C1 and C6, 100%).
  Global-direction ablations on the subsets reproduce 98.7–100% of the original outputs (batch composition flips
  ≈1% of ablated outputs), so all comparisons use the matched baseline.
- Statistics: pooled n = 306 held-out evaluations (each episode evaluated once, with the direction that excluded
  its scheme); Wilson CIs; paired Δ vs matched baseline with a scheme-stratified paired bootstrap (10k) and exact
  McNemar p. Decision rules as pre-specified (SURVIVES = fold drop CI excludes 0 AND ≥ 50% of the global drop on the
  same episodes; C6 spared = drop ≤ 5 pp AND CI upper ≤ 10 pp). Parse failures are counted as wrong and reported;
  nothing was dropped.
- Code/data: exp/loso_fit.py, exp/modal_app.py (`intervene_many`, `loso` entrypoint), exp/loso_analyze.py;
  data/loso/{splits.json, provenance.json, fold*/, runs/loso_f*_*/metrics.json, analysis.json, tables.md, full_run.log};
  figures/fig7_loso.{png,pdf}.

**Pooled held-out results (n = 306; Wilson 95% CI).**
| config | C1 decode | Δ vs matched none [95% CI] | C1 parse fail / decoded None | C6 bet-exact | Δ C6 [CI] | recall | ARC |
|---|---|---|---|---|---|---|---|
| none (matched) | 0.958 [0.929–0.975] | – | 0 / 0 | 1.000 | – | 0.967 | 0.970 |
| dm global | 0.693 | −0.265 [−0.317, −0.216] | 0 / 0 | 1.000 | 0 | 0.931 | 0.950 |
| **dm fold** | **0.732** [0.680–0.779] | **−0.225** [−0.271, −0.180] | 0 / 0 | **1.000** | 0 | 0.944 | 0.942 |
| PC-1D global | 0.931 | −0.026 [−0.052, 0.000] | 0 / 10 | 1.000 | 0 | 0.964 | 0.960 |
| PC-1D fold | 0.912 | −0.046 [−0.075, −0.020] | 0 / 14 | 1.000 | 0 | 0.961 | 0.953 |
| state global | 0.379 | −0.578 [−0.637, −0.520] | 9 / 33 | 0.980 | −0.020 [−0.036, −0.007] | 0.938 | 0.920 |
| **state fold** | **0.441** [0.387–0.497] | **−0.516** [−0.575, −0.458] | 7 / 42 | **0.951** [0.921–0.970] | **−0.049** [−0.072, −0.029] | 0.935 | 0.915 |
| PC-2D global | 0.605 | −0.353 [−0.412, −0.294] | 0 / 0 | 0.807 | −0.193 [−0.239, −0.150] | 0.912 | 0.930 |
| PC-2D fold | 0.578 | −0.379 [−0.438, −0.324] | 0 / 0 | 0.866 | −0.134 [−0.170, −0.098] | 0.895 | 0.937 |
| LR global | 0.928 | −0.029 [−0.056, −0.007] | 0 / 0 | 1.000 | 0 | 0.958 | 0.970 |
| LR fold | 0.902 | −0.056 [−0.082, −0.029] | 0 / 0 | 1.000 | 0 | 0.961 | 0.970 |
Fold vs global on identical episodes: dm +0.039 [+0.010, +0.069] (fold ablation slightly weaker); state +0.062
[+0.020, +0.105] in C1 and −0.029 [−0.052, −0.007] in C6; PC-2D −0.026 n.s.; LR −0.026 [−0.046, −0.007].
Direction vs its own control (fold-fit, same episodes): dm − PC-1D = −0.180 [−0.229, −0.131] in C1, 0 in C6;
state − PC-2D = −0.137 [−0.206, −0.069] in C1 and +0.085 [+0.042, +0.131] in C6.

**Per fold (C1 decode, n = 51 each; none → global → fold-fit; C6 fold-fit).**
- dm: 0.961→0.686→0.627 · 0.980→0.667→0.706 · 0.922→0.686→0.706 · 1.000→0.824→0.863 · 0.882→0.627→0.667 ·
  1.000→0.667→0.824; C6 1.000 in all folds. Drop in 6/6 folds.
- state: 0.961→0.392→0.471 · 0.980→0.392→0.314 · 0.922→0.373→0.431 · 1.000→0.392→0.451 · 0.882→0.392→0.490 ·
  1.000→0.333→0.490; C6 1.000/1.000/1.000/0.961/1.000/**0.745** (fold 5). Drop in 6/6 folds.
- PC-2D: fold-fit 0.627/0.510/0.549/0.569/0.569/0.647; C6 0.765/0.980/0.686/1.000/0.980/0.784.
- LR: fold-fit 0.765/0.980/0.824/1.000/0.843/1.000 (drops concentrated in folds 0, 2, 4).

**Pre-specified verdicts.**
- **dm detector direction: SURVIVES.** Fold drop 0.225 = 85% of the global drop 0.265, CI [0.180, 0.271], 6/6
  folds; C6 spared (0 drop). It also exceeds its matched 1-D PC control by 18 pp. The leakage inflated the effect
  only slightly (+3.9 pp).
- **state 2-D subspace: SURVIVES** for covert decoding (fold drop 0.516 = 89% of 0.578, 6/6 folds). C6 is spared
  **only marginally** by the pre-set rule: drop 4.9 pp (CI upper 7.2 pp, threshold 5 / 10 pp), and one fold (scheme
  5) drops to 0.745. So the selectivity claim is weaker out of sample than in-sample (C6 0.980 → 0.951).
- **PC-2D control** still disrupts covert decoding (−0.379) at a larger explicit-decision cost (−0.134). The state
  subspace remains more disruptive to C1 (by 13.7 pp) and less to C6 (by 8.5 pp) than this control →
  selectivity relative to the control survives; uniqueness was never claimed and is not supported.
- **LR detector direction: small but nonzero effect.** The original reading "not load-bearing / collusion intact
  (0.928)" is refined. Global −2.9 pp [−5.6, −0.7] and fold-fit −5.6 pp [−8.2, −2.9], both CIs excluding 0; collusion
  stays at 0.90. The mechanical verdict "SURVIVES" here means "a small effect is present", not "a large effect".
  PC-1D likewise shows a small nonzero fold-fit effect (−4.6 pp). So single high-variance or discriminative
  directions perturb decoding by a few points; only dm (and state) do so substantially.

**What this establishes / does not.**
- Establishes: the two positive 7B effects (dm direction → no-signal default with explicit decisions intact; state
  subspace → covert decoding collapses) do not depend on fitting the direction on the evaluated scheme; most (85–89%)
  of each effect remains, consistently across all 6 folds.
- [UPDATED: fresh messages → §4.9; 14B LOSO → §4.10; 32B → §4.11.] Does not establish: generalisation to new messages (single frozen generation), to new schemes outside the 6
  topic sets, or to 14B/Phi (their directions are still fitted on all episodes; not rechecked here); detector AUROC
  after fold-fit ablation (behaviour only, no activations re-extracted); a unique mechanism (PC controls also disrupt).
- Cost: Modal billing today rose $7.00 → $7.37 for the smoke test + 66 configs (115 s wall, 6 parallel L40S
  containers); billing may lag slightly. No apps left running.

### 4.9 Follow-up A: fresh-message robustness, Qwen2.5-7B ✅ (data/fresh/; notes/fresh_results.md; post hoc)
- Design: 5 new COLLUDING sender message sets for the SAME 306 episode contexts (env.episodes(range(17)): same
  prompts, schemes, 6×3×17 state distribution, T=0.7, top_p=1, max_new 120). Only the sender sampling seed changes:
  gen_seed ∈ {101, 202, 303, 404, 505}; original = 0. IDs are `m<seed>_<orig_id>`. Nothing was regenerated or filtered.
- Directions and analysis are FROZEN (original data/directions files, no refit); same hooks and scoring. Per set,
  one L40S container evaluates none/dm/PC-1D/state/PC-2D/LR on all 306 episodes in the original order (matched
  batching): C1, C6, recall, ARC. 30/30 configs, 497 s wall.
- Checks: ids one-to-one with orig_id; counts derived from data; 0 message parse failures; ARC identical across sets
  for each config (determinism).
- Overlap with the original set: 3–6% exact same-episode copies, 5–9% same-episode near-duplicates (difflib ≥ 0.9;
  median same-episode similarity ≈ 0.45); because of stock phrases, 23–29% exactly match SOME original message and
  33–39% have a near-duplicate somewhere. Sender compliance is similar (target-topic keyword 0.790 vs 0.824).
- Baseline capability unchanged: C1 decode 0.948–0.967 per set (original 0.958); C6 1.000 in every set.
- Results (C1 decode Δ vs each set's matched none; pooled over 5 sets of the SAME 306 contexts; **primary CI =
  cluster bootstrap over contexts**, stratified by scheme):
| config | original Δ | fresh pooled Δ [cluster 95% CI] | per-set range / seed SD | C6 Δ fresh [CI] | gated verdict |
|---|---|---|---|---|---|
| dm | −0.258 | **−0.278 [−0.318, −0.237]** | SD 0.017 | 0 | REPLICATES |
| state 2-D | −0.575 | **−0.584 [−0.635, −0.532]** | SD 0.008 | −0.029 [−0.044, −0.016] (spared) | REPLICATES |
| PC-2D | −0.356 | −0.363 [−0.415, −0.310] | SD 0.025 | −0.198 [−0.238, −0.158] (not spared) | REPLICATES |
| LR | −0.029 | −0.054 [−0.069, −0.039] | SD 0.009 | 0 | n/a: small (≈5 pp) effect again |
| PC-1D | −0.023 | −0.035 [−0.050, −0.020] | SD 0.018 | 0 | n/a: DECODE-field formatting effect (bets unchanged, +0.001) |
  (Original Δ recomputed from the full-306 original runs; the LOSO "global" re-runs differ slightly due to batching.)
- Novel-message subset (max similarity to ANY original < 0.9 and no same-episode exact copy; 186–204 per set, 968
  rows, 289 contexts): baseline 0.949 vs 0.970 for the rest; effects within ≈1 pp of the full set (dm −0.281, state
  −0.585, PC-2D −0.375, LR −0.043).
- The n=5 seed SD is indicative only (95% CI ≈ [0.6, 2.9]×SD). The old episode-level pooled CI is kept but labelled
  optimistic.
- Establishes: the 7B dm and state effects (and the non-selective PC-2D effect) are unchanged when the sender is
  resampled at T=0.7 for the same contexts, including on genuinely novel lines. Does not establish: new schemes or
  contexts, real-world channels, other models (14B/32B have no fresh-message check); this is a frozen-direction
  transfer test.


### 4.10 Follow-up B: 14B held-out-scheme (LOSO) check ✅ (data_models/qwen14b/loso/; post hoc)
Plan: notes/followups_plan.md §B; log: notes/followups_log.md; review: notes/followups_review.md.
- Inputs: the existing 14B activations (C1/C4 mean/last, [306,48,5120]) and 14B's OWN frozen sender messages
  (/vol/qwen14b/data/messages.jsonl), so there was no extraction cost.
- Fitting: exp/loso_fit.py (DATA_DIR=data_models/qwen14b) reproduces the existing 14B dm/state/PC-2D/LR files
  exactly when given all 306 episodes (agreement ≥ 0.999998). PC-1D did not exist for 14B, so it was built here with
  the same construction. Fold-vs-global agreement: 0.81–0.98.
- Run: 6 folds × 11 configs, L40S, 214 s wall, same fixed all-layer + embedding policy. Analysis by
  exp/loso_analyze.py, unchanged.
- Pooled held-out results (n = 306; strict C1 decode and bet-exact both reported, because 14B often drops "DECODE:"):
| config | C1 decode | C1 bet-exact | Δ C1 decode [CI] | C6 bet-exact (Δ) | recall | ARC | C1/C6 parse fail |
|---|---|---|---|---|---|---|---|
| none | 0.997 | 0.997 | – | 1.000 | 1.000 | 0.990 | 0/0 |
| dm fold | 0.974 | 0.977 | −0.023 [−0.042, −0.007] | 1.000 (0) | 0.993 | 0.978 | 0/0 |
| PC-1D fold | 0.990 | 0.990 | −0.007 [−0.016, 0.000] | 1.000 | 0.984 | 0.990 | 0/0 |
| state global (in-sample) | 0.382 | 0.624 | −0.614 | 0.709 (−0.291) | 0.745 | 0.820 | 12/0 |
| **state fold** | **0.464** | **0.614** | **−0.533 [−0.585, −0.480]** | **0.771 (−0.229 [−0.275, −0.183])** | 0.745 | 0.753 | 17/0 |
| PC-2D fold | 0.288 | 0.546 | −0.709 [−0.755, −0.660] | 0.663 (−0.337) | 0.794 | 0.365 | 78/68 |
| LR fold | 0.990 | 0.990 | −0.007 [−0.016, 0.000] | 1.000 | 1.000 | 0.990 | 0/0 |
- Gated verdicts (global effect ≥ 10 pp required): state SURVIVES (fold 0.533 = 87% of global, 5/6 folds) but is NOT
  selective (C6 −0.229, recall 0.745, ARC 0.753). dm, PC-1D and LR: n/a (in-sample effects < 10 pp).
- 14B "PC-2D" is a format-breaking perturbation, not a usable control: 257/306 C1 outputs unparsed (global), 78 C1
  and 68 C6 parse failures (fold), ARC 0.35–0.37. So state − PC-2D comparisons at 14B are not evidence of selectivity.
- Reading: on 14B the decoded-state subspace is needed for covert decoding but its removal also damages
  explicit-state decisions, recall and ARC. The 7B "selective" pattern does not hold on 14B out of sample, and the
  7B dm detector effect is absent at 14B.

### 4.11 Follow-up C: focused Qwen2.5-32B-Instruct extension ✅ (data_models/qwen32b/; post hoc, exploratory)
- Setup: bf16 (same precision as 7B/14B; no quantisation), revision 5ede1c97, 64 layers, d 5120. 65.5 GB weights
  were prefetched to the volume on CPU (489 s). Modal **H200** (141 GB); generation batch 48 and activation batch 8,
  as for 14B. Inputs: the **7B frozen messages and episode ids** (same prompts and scoring) → matched input-for-input
  with 7B (14B used its own messages).
- Capability gate (n = 306): C1 0.971 (r 0.934), C2 0.324, C4 0.350 (chance controls), C6 1.000, recall 0.974,
  ARC 0.98, 0 parse failures → passes the 0.70 gate.
- Activations: C1 and C4 only (mean/last/full_last), [306,64,5120] fp16, 1.2 GB. Directions use the original
  constructions and hyperparameters: global (all 306 = in-sample) and 6 held-out folds. Variance share (mean over
  layers): dm 0.106, state 0.163, PC-2D 0.217, PC-1D 0.083, LR 0.019.
- Probe geometry (notes/xmodel_results.md; data_models/qwen32b/probes32.json):
  - C1vC4 AUROC at layer 0 is 0.992 (mean) / 0.997 (end-of-turn), and 0.96–1.00 at all layers.
  - State is at chance up to L27; ≥ 0.5 at L38 (0.60 depth); ≥ 0.8 at L42 (0.67); plateau 0.94–0.97.
  - Detector vs state max |cos| 0.111.
  → 32B matches 7B's geometry (onset depth 0.60/0.67 vs 7B 0.59/0.67; 14B earlier at 0.51/0.55).
  - C1vC2/C3/C5 were not extracted for 32B.
- LOSO interventions (6 folds × 11 configs, 6 × H200, 208 s wall):
| config | C1 decode | Δ C1 decode [CI] | C6 bet-exact | recall | ARC |
|---|---|---|---|---|---|
| none | 0.971 | – | 1.000 | 0.974 | 0.980 |
| dm global / fold | 0.915 / 0.925 | −0.056 / **−0.046 [−0.072, −0.020]** | 0.997 / 0.997 | 0.958 | 0.970 / 0.973 |
| PC-1D global / fold | 0.971 / 0.971 | 0 / 0 | 1.000 | 0.974 | 0.990 |
| state global / fold | 0.928 / 0.886 | −0.042 / **−0.085 [−0.118, −0.056]** | 1.000 / 0.993 | 0.971 / 0.980 | 0.900 / 0.883 |
| PC-2D global / fold | 0.974 / 0.974 | +0.003 / +0.003 | 1.000 / 0.993 | 0.974 | 0.980 / 0.952 |
| LR global / fold | 0.977 / 0.977 | +0.007 / +0.007 | 1.000 | 0.967 | 0.990 |
  - Gated verdicts: all n/a (no in-sample effect ≥ 10 pp). Mechanical "state SURVIVES" is a rule artefact: the fold
    effect exceeds the in-sample one (fold − global −0.042 [−0.072, −0.010]) and comes mostly from fold 4 (−0.35);
    folds 1 and 3 drop 0.08 and 0.12, the other three ≈ 0. ARC under the state ablation falls to 0.65 in fold 0
    with no C1 drop.
- **Verification V1 (decisive check, exp/verify_32b_ablation.py):** the 32B global state ablation was re-run with
  activation capture (n=306; C1 decode 0.922).
  - Residual projection onto the subspace is ≤ 0.021% of activation norm at all 64 layers (e.g. L60 128 → 0.07),
    so the ablation is applied correctly.
  - A re-trained LOSO state probe still reads the state at 0.913 (L42+; baseline 0.966).
  → the 32B covert decision is redundantly encoded outside the 2-D subspace. Comparison (post-onset mean,
    retrained, after the same in-sample all-layer state ablation): **7B 0.918 → 0.624; 14B 0.981 → 0.759; 32B 0.966
    → 0.913** (figures/fig10_redundancy). Behaviour: 7B −0.58, 14B −0.61 (non-selective), 32B −0.04 (in-sample).
- **Answer to the 32B question** (exploratory; 3 models, not a scaling law): the detector/probe geometry is unchanged
  at 32B (layer-0 separability, ~0.6-depth state onset, detector ⟂ state), but the intervention relationship
  changes. With the fixed recipe, no single direction or 2-D subspace (detector, state, or control) materially
  disrupts 32B's covert decoding (≤ 8.5 pp held-out), although the state stays linearly decodable and the hooks act.
  Whether this reflects scale or this recipe (all-layer linear 2-D ablation, one message set) is open; no
  layer/strength search was done.

### 4.12 Cross-model synthesis (notes/xmodel_results.md; data/xmodel_tables.md; figures/fig9_xmodel)
Held-out (fold-fit) Δ in C1 covert decode / Δ in C6 explicit decisions:
| direction | 7B | 14B (own messages) | 32B (7B messages) |
|---|---|---|---|
| dm detector | −0.225 / 0 | −0.023 / 0 | −0.046 / −0.003 |
| LR detector | −0.056 / 0 | −0.007 / 0 | +0.007 / 0 |
| state 2-D | −0.516 / −0.049 | −0.533 / −0.229 | −0.085 / −0.007 |
| PC-2D | −0.379 / −0.134 | −0.709 / −0.337 (format breakdown) | +0.003 / −0.007 |
| PC-1D | −0.046 / 0 | −0.007 / 0 | 0 / 0 |
Robust across all three: detectors are surface-separable (layer-0 AUROC ≥ 0.986), the decoded state appears abruptly
at about half to two-thirds depth, and detector directions are near-orthogonal to it (max |cos| ≤ 0.125).
Model-specific: the dm detector's causal role (7B only); state-ablation selectivity (7B marginal, 14B none, 32B no
effect). Only the per-model direction of each effect is supported; magnitude orderings across models are confounded
(14B's own messages, depth under "all layers", one message set each for 14B/32B, L40S vs H200).

## 5. Figures (planned)
| # | File | What it shows | Source data / script |
|---|---|---|---|
| 1 ✅ | figures/fig1_setup.png (+.pdf) | Environment + intervention schematic. Caption: "Experimental setup. (a) Signal Table: a colluding sender encodes a hidden state (COLD/NEUTRAL/HOT) as the topic of casual table talk via a private scheme; the colluding receiver, which knows the scheme, decodes and bets; a pit-boss monitor sees only public chat. Controls vary whether the message is a signal (C2/C4 honest messages, C5 scheme mismatch) or remove the covert channel (C6 explicit state). (b) Linear probes on the receiver residual stream give a collusion score. (c) We project a direction or subspace out of the receiver at every layer (h ← h − (h·Q)Qᵀ) and ask whether collusion, explicit-state betting (C6) and a retrained probe survive, for detector directions, the decoded-state subspace, and random and principal-component controls." (receiver label fixed: "knows scheme, reads public chat") | figures/make_fig1.py |
| 2 ✅ | figures/fig2_probe_layers.{png,pdf} | (a) Detector AUROC by layer, leave-one-scheme-out: C1vC2/C1vC4/C1vC3 are ≈1.00 at every layer incl. 0 (surface-separable); C1vC5 (message-is-a-signal) only works mid-span (msg) / L18+ (decision token). (b) Covert-state decoding accuracy on held-out schemes, at chance until L12, 0.92 from L18. | data/probes.json; exp/probes.py → exp/figs_probes.py |
| 3 ✅ | figures/fig3_token_trace.{png,pdf} | (a) Per-token projection on d_detect (layer 20) for one message read as a signal (C1) vs the same message when it is not a signal (C5): nearly identical, with a spike on "weather" in both. (b) Mean over 40 msgs × 5 layers: topic-word spike in C1 is 2.2–3.8× other tokens, but 42–91% of the excess over other tokens persists in C5 → the spike is largely lexical (Qwen-7B, n=34). | data/acts/*_traces.npz, data/token_trace_stats.json; exp/figs_probes.py |
| 4 ✅ HEADLINE (revised) | figures/fig4_2x2.{png,pdf} | Qwen2.5-7B only (14B/Phi numbers in Table 2 / appendix). (a) Specificity map: x = explicit-state decision intact (C6 bet acc.), y = covert collusion (C1 decode, Wilson 95% CI); one point per ablation. Bottom right = breaks covert decoding but spares explicit decisions (decision-state 2-D; detector diff-means at 0.70); top right = no effect (random, PC-1D, LR, raw-LR, NARCBench-style); PC-2D and C1vC5 msg-span sit to the left (nonspecific). (b) For each detector-direction ablation: retrained detector AUROC (blue, ≥0.83: the information survives) vs C1 decode (orange). Dotted line = no-ablation decode. | data/fig4_specificity.csv, data/interv/*; exp/figs_results.py |
| 5 ✅ | figures/fig5_steer.{png,pdf} | Steering along HOT−COLD at L20: (a) mean bet by true state vs α (COLD bets $2→$85 at α=+40; × = random direction at α=±40, COLD bets $1–2); (b) decode acc vs recall vs ARC. | data/interv/steer_*; exp/figs_results.py |
| 6 ✅ | figures/fig6_models.{png,pdf} | Cross-model (Qwen2.5-7B, Qwen2.5-14B, Phi-3.5-mini) vs relative depth: (a) covert-state decodability on held-out schemes rises from chance at ~0.5–0.6 depth to ≥0.9 within 2–4 layers in all three; (b) C1vC4 detector AUROC ≈1.0 at every depth (solid) vs C1vC5 'signal-for-me' detector (dashed), which only works after state onset (weak in Phi). | data/probes.json, data_models/*/probes.json; exp/fig_models.py |
| 7 ✅ (post hoc) | figures/fig7_loso.{png,pdf} | Qwen2.5-7B, LOSO robustness: pooled held-out (n=306) accuracy for each ablation fitted on all schemes (grey, original) vs fitted without the held-out scheme (blue), Wilson 95% CIs; dots = per-fold fold-fit values; dashed = matched baseline. (a) C1 covert decoding, (b) C6 explicit-state decision. dm and state effects persist; state's C6 sparing weakens slightly (0.98 → 0.95). | data/loso/analysis.json, tables.md; exp/loso_analyze.py |
| 8 ✅ (post hoc) | figures/fig8_fresh.{png,pdf} | 7B fresh-message robustness: per config, original-set accuracy (black) vs the 5 fresh sender seeds (dots) and the fresh pooled value (cluster CI), for (a) C1 covert decoding and (b) C6 explicit-state betting. | data/fresh/analysis.json; exp/fresh_analyze.py |
| 9 ✅ (post hoc) | figures/fig9_xmodel.{png,pdf} | 7B/14B/32B: (a) state decodability vs relative depth; (b) C1vC4 detector AUROC vs depth (incl. layer 0); (c) held-out ablation effects, C1 covert drop vs C6 explicit drop, per direction and model, with 95% CIs. | data/xmodel_summary.json, data_models/*/loso/analysis.json; exp/xmodel_analyze.py |
| 10 ✅ (post hoc) | figures/fig10_redundancy.{png,pdf} | State decodability (retrained LOSO probe) before vs after the in-sample all-layer 2-D state ablation: 7B drops to ≈0.62, 14B to ≈0.76, 32B stays ≈0.91 → redundancy at 32B. | */interv/abl_state_C1_last_2d/post_probes.json, data_models/qwen32b/interv/verify_state_global_acts/verify.json; exp/fig_redundancy.py |
| 7b/7c | figures/fig7_loso_qwen14b.*, fig7_loso_qwen32b.* | Per-model versions of Fig 7 (global = in-sample) | exp/loso_analyze.py with DATA_DIR |
| S1 | supp_narcbench/figs/confound_auroc.png | NARCBench Qwen3-32B: surface features (n_tokens etc.) vs probe AUROC per tier; Stego fully length-confounded | supp_narcbench/analysis.py |
| S2 | supp_narcbench/figs/cosine_heatmap_P.png | Cross-tier probe-direction cosine vs shuffled baseline | supp_narcbench/analysis.py |

## 6. Limitations / dual-use (appendix draft — REQUIRED section)
**Limitations**
- Instructed collusion, not emergent: the scheme is given to both agents and the sender is told the topic. A real
  emergent covert channel could be represented differently (cf. Rose et al.'s weak emergent-tier results).
- Scale/models: three open models of 3.8–14B from two families (Qwen2.5-7B/14B, Phi-3.5-mini); OLMo-2-7B failed the
  capability gate. Frontier models may represent collusion differently. Detector-validity results and state onset at
  ~half depth are robust across all three; state-subspace selectivity, the dm detector's causal role, and the steering
  layer are model-dependent or post hoc.
- C6 (explicit state) is close to a copy task (the answer word is in the prompt), so sparing C6 is a weak test of
  "decision-generic" representations.
- Steering multiples and layers differ per model (7B 3×‖Δμ‖ at L20; 14B/Phi 6× two layers after onset); at 3× the
  14B/Phi effects are small (COLD $33/$48).
- The C1vC5 decision-token ablation is the one case where linearly recoverable information drops (retrained 0.32 14B,
  0.35 Phi), but it also barely affects collusion (0.88 / 0.95).
- Follow-ups (§4.9–4.12) are post hoc and exploratory; plans were written before runs (notes/followups_plan.md),
  with one reporting amendment after review (gated verdict vocabulary; mechanical verdicts kept).
  - Fresh messages share the original episode contexts, and ≈⅓ near-duplicate some original line; the
    novel-subset result is similar.
  - 14B used its own sender messages while 7B/32B share messages; 14B/32B have one message set each and no
    fresh-message check.
  - 32B "global" configs are in-sample; only fold configs are held-out.
  - The PC controls are not variance-matched layer by layer in any model (PC-2D > state averaged over layers in
    7B/32B; state > PC-2D post-onset in all three); 14B PC-2D breaks output format (unusable as a control).
  - The embedding hook reuses the layer-0 direction (inherited convention).
  - About 150 uncorrected tests across the project: descriptive; small effects (< 5 pp) are not interpreted.
  - GPUs differ (L40S vs H200).
  - The 32B redundancy explanation rests on one verification run (in-sample global direction); no layer/strength
    search was done, so the 32B result may be recipe-specific rather than scale-specific.
- Direction-fitting leakage (addressed post hoc, 7B only, §4.8): the original intervention directions and PC controls
  were fitted on all episodes, including the evaluated ones. Refitting without the evaluated scheme keeps 85–89% of the
  dm and state effects (6/6 folds), but the state subspace's sparing of explicit decisions is only marginal (C6 0.951;
  scheme-5 fold 0.745), and the LR direction shows a small nonzero effect (−5.6 pp). The 14B/Phi interventions were not
  rechecked and still use all-episode directions. The check is exploratory (designed after seeing the global
  results), evaluates behaviour only (no post-ablation activations), and uses 6 folds of 51 episodes; the bootstrap
  treats episodes within a scheme as exchangeable.
- Post-hoc choices: for Phi, the layer-restricted (16–31) ablations and all "2 layers after onset" steering runs were
  added after the pre-specified versions failed (all-layer ablation broke the model; 0.7·L steering had no effect).
  They are labelled exploratory.
- Mass-mean ablations interact with massive-activation dimensions (Phi), so "ablation breaks the model" there is
  uninformative about collusion.
- Single game and one simple channel (topic choice); the channel lives entirely in the receiver's reading of one message.
- Linear interventions only: directional ablation and additive steering. Nonlinear or distributed encodings could
  carry the signal; post-ablation state decodability is still 0.53–0.68 for some ablations (partial removal).
- "Detector blind" after ablating the detector's own direction is near-tautological for a frozen probe; the
  behavioural result (collusion survives) is the claim. Retrained probes recover, so this models a frozen-monitor setting.
- C5 differs from C1 in the scheme transcript text as well, causing a transcript-identity artefact (AUROC < 0.5 at
  early layers); C1vC5 claims are restricted to L18+ (decision token).
- In C1 the topic word also appears earlier in the receiver's context (scheme transcript); "is a signal" and
  "repeated in context" cannot be fully separated in this design.
- Greedy decoding for receivers; uncertainty from 306 episodes (Wilson 95% CI about ±2–5 pp); no multiple-seed
  repeat of sender generation.
- Interpretability illusions (Makelov et al. 2023) can make subspace interventions misleading; utility controls
  partly mitigate this.
- Time-limited (weekend sprint): no hyperparameter sweep for probe regularisation (C=0.1 fixed).

**Dual-use / ethics**
- Our main negative result (probe directions can be removed without hurting collusion) could inform evasion of
  activation monitors. However, the intervention requires white-box write access to the model's activations or
  weights, which an external colluding agent normally does not have; and we show that retraining the monitor
  recovers detection. The defensive takeaway (validate monitors causally; use retrained or ensembled probes; check
  for surface confounds) outweighs the marginal uplift. Obfuscation of activation monitors is already documented
  (Bailey et al. 2024; Gupta & Jenner 2025; McGuinness et al. 2025).
- Steering experiments show that one direction can make an agent act on a covert signal; the same tool could be
  used defensively (suppress the covert-state subspace). In Qwen-7B the utility cost is small (ARC 0.97→0.92), but
  in 14B it costs explicit decisions (C6 0.71), recall (0.75) and ARC (0.82), and lower-variance PC subspaces are also
  disruptive, so this is not yet a specific mitigation.
- No human subjects; all data is synthetic; the casino card-counting framing is a toy (card counting is legal but
  banned by casinos); no real-world deployment.
- Compute: Modal ≈ $11.2 total for the project today (7B study ≈ $1.5; first replication ≈ $6; 7B LOSO ≈ $0.4; follow-ups A–C incl. 32B on H200 ≈ $3.8, tracked in notes/followups_log.md); analysis on a laptop CPU.

## 7. Sources
✅ notes/sources.md: 36 verified refs + references.bib + draft Related Work paragraphs. notes/rose_paper.md ⏳.
Key citation facts (from sources agent):
- The "receiver spikes on encoded parts of partner's message" claim is in Rose et al. **v1** abstract (Apr 2026) —
  cite `arXiv:2604.01151v1` when quoting. Not found in v5 (Oct 2026, HELIOS reframe). v5 defines a "causality"
  criterion for collusion that it says usually can't be tested without counterfactuals → our ablations supply that.
- **Closest prior work: Das et al. 2026 (arXiv 2609.03035)** — NARCBench blackjack; probes stay accurate when agents
  are told they're monitored. Behavioural robustness vs our mechanistic necessity test. Cite prominently.
- Novelty wording: "to our knowledge, the first causal test of collusion-probe directions" (only a SPAR proposal found).
- Interpretability-illusion caveat to cite in limitations: Makelov et al. 2023 (subspace patching illusion).
- Plus factors: cite Kovacic et al. 2011; Berger et al. 2025 real title is "Coordination by Code" (Antitrust 40(1)).
- Venues marked [K] in sources.md (Arditi NeurIPS'24, CAA ACL'24, LEACE NeurIPS'23, Azaria&Mitchell EMNLP-F'23) are
  from memory — double-check before submission.

## 8. Compute & reproducibility
Modal L40S; model Qwen/Qwen2.5-7B-Instruct bf16; costs: ⏳. Code in exp/, run instructions exp/README.md.


## 9. Report outline (8 pages main text) — from the final review
1. **Intro + contributions (1.0 p).** Motivation: probe localisation evidence is correlational (Rose et al. v1 spike
   claim, removed in v4/v5); monitors as a "plus factor" (Q2.3). Four contributions (move NARCBench audit into §2).
2. **Why a causal test (0.5 p).** NARCBench Stego is separable by length (n_tokens AUROC 1.0; probe 0.961 → 0.516
   after residualising); Transfer is clean. S1/S2 go in the appendix.
3. **Signal Table + interventions (1.25 p, Fig 1).** Conditions C1–C6, probes (LOSO), ablation/steering, controls
   (random, PC, C6), utility (recall, ARC).
4.1 **Detectors are surface-separable (0.75 p, Fig 2).**
4.2 **The token spike is largely lexical; small non-lexical marker (0.5 p, Fig 3 at half width).**
4.3 **Ablation specificity (1.5 p, Fig 4 + Table 1 = §4.5a C6 table; compact frozen/excluded/retrained table).**
4.4 **Steering (0.5 p, Fig 5 at half width, beside Fig 3).**
4.5 **Replication (1.0 p, Fig 6 + Table 2 = §4.7 verdict table).**
5. **Discussion + recommendations for monitors (0.75 p).** 6. **Conclusion (0.25 p).**
Appendix: Limitations/dual-use (required), S1, S2, layer localisation (§4.5b), 14B/Phi ablation + steering tables
(notes/replication.md), C6 prompt, massive-activation note, all prompts (exp/env.py).

## 10. Likely judge questions (with answers grounded in data)
- *"Isn't 'detector blind after ablating its own direction' trivial?"* Yes, by construction. The claim is behavioural:
  collusion stays at 90–98% after LR ablation (7B held-out-fitted: 0.902, a small −5.6 pp effect, §4.8), a retrained probe recovers 0.96–0.99, and the
  direction-excluded frozen LR probe gives 0.81 (7B).
- *"Is the state subspace just the answer/format variable? Is the PC control fair?"* C6 explicit-state betting stays
  0.98 (7B) / 1.00 (Phi) but drops to 0.71 (14B). The lower-variance PC-2D control also disrupts decoding (0.60 / 0.63 /
  0.41), so we claim selectivity, not uniqueness, and note that C6 is close to a copy task.
- *"Why did the diff-means result and the steering layer not replicate?"* The dm result is 7B-only (14B 0.971; in Phi
  the ablation breaks the model through massive-activation dimensions). The pre-specified 0.7·L steering failed; the
  "2 layers after onset" layer was found post hoc (2 layers × 2 α) and is labelled exploratory.
