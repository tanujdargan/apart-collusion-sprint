# Cut the Wire: reproduce the paper.
#   make setup      both venvs (uv)
#   make analysis   CPU only, free: every table/figure from the committed outputs (+ saved activations if present)
#   make narcbench  CPU: the NARCBench length-confound audit (downloads ~GBs of NARCBench activations)
#   make pipeline   full GPU re-run on Modal (~$16 at the paper's prices), then analysis
# Phases can run alone: make main | replication | robustness | x32b | verify. JOBS = parallel Modal runs.
# Intervention runs are replayed from their committed metrics.json (exp/replay_interv.py), so tags and args are exact.

MODAL := .venv/bin/modal
RUN   := $(MODAL) run exp/modal_app.py
PY    := .venv-analysis/bin/python
JOBS  ?= 6
Q14   := Qwen/Qwen2.5-14B-Instruct
Q32   := Qwen/Qwen2.5-32B-Instruct
PHI   := microsoft/Phi-3.5-mini-instruct
OLMO  := allenai/OLMo-2-1124-7B-Instruct

# $(call interv,<namespace>): replay every intervention run of one model, then post-hoc probes on runs with activations
define interv
	python3 exp/replay_interv.py cmds $(1) | xargs -P $(JOBS) -I{} sh -c '{}'
	python3 exp/replay_interv.py post $(1) | sh
endef

# $(call pull,<volume dir>,<local dir>,<files>)
define pull
	mkdir -p $(2) && for f in $(3); do $(MODAL) volume get --force cutwire-vol $(1)/$$f $(2)/; done
endef

.PHONY: setup pipeline main replication robustness x32b verify analysis narcbench

setup:
	uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python modal numpy scikit-learn
	uv venv -p 3.12 .venv-analysis && uv pip install -p .venv-analysis/bin/python numpy scipy scikit-learn matplotlib

pipeline: main replication robustness x32b verify analysis narcbench

# Qwen2.5-7B: capability gate, data + activations, directions, all-layer/localisation/steering/control runs
main:
	$(RUN)::gate --tag gate2 && python3 exp/analyze_gate.py gate2
	$(RUN)::stage_b
	$(call pull,/acts,data/acts,C1.npz C2.npz C3.npz C4.npz C1_traces.npz C2_traces.npz meta.json)
	$(RUN)::stage_c5
	$(PY) exp/probes.py baseline
	$(PY) exp/make_lrraw.py
	$(PY) exp/make_pc_ctrl.py
	$(PY) exp/loso_fit.py global && cp data/loso/global/pc1d.npy data/directions/pc_ctrl_1d.npy
	$(PY) exp/make_steer_dir.py 20
	$(call interv,data)

# Qwen2.5-14B and Phi-3.5-mini (OLMo-2-7B: gate only; it failed the capability gate)
replication:
	CUTWIRE_TRANSFORMERS=4.48.3 $(RUN)::gate --model $(OLMO) --out-tag olmo7b
	$(RUN)::gate --model $(Q14) --out-tag qwen14b
	$(RUN)::stage_b --model $(Q14) --out-tag qwen14b --conditions C1,C2,C3,C4,C5
	$(call pull,/qwen14b/acts,data_models/qwen14b/acts,C1.npz C2.npz C3.npz C4.npz C5.npz meta.json)
	DATA_DIR=data_models/qwen14b $(PY) exp/probes.py baseline
	DATA_DIR=data_models/qwen14b $(PY) exp/make_pc_ctrl.py
	for l in 28 34; do DATA_DIR=data_models/qwen14b $(PY) exp/make_steer_dir.py $$l; done
	$(call interv,data_models/qwen14b)
	$(RUN)::gate --model $(PHI) --out-tag phi35mini
	$(RUN)::stage_b --model $(PHI) --out-tag phi35mini --conditions C1,C2,C3,C4,C5
	$(call pull,/phi35mini/acts,data_models/phi35mini/acts,C1.npz C2.npz C3.npz C4.npz C5.npz meta.json)
	DATA_DIR=data_models/phi35mini $(PY) exp/probes.py baseline
	DATA_DIR=data_models/phi35mini $(PY) exp/make_pc_ctrl.py
	for l in 18 23; do DATA_DIR=data_models/phi35mini $(PY) exp/make_steer_dir.py $$l; done
	$(call interv,data_models/phi35mini)

# Held-out-scheme refits (7B, 14B), fresh messages, placement unbundling
robustness:
	$(PY) exp/loso_fit.py check && $(PY) exp/loso_fit.py fit
	$(RUN)::loso
	DATA_DIR=data_models/qwen14b $(PY) exp/loso_fit.py global && DATA_DIR=data_models/qwen14b $(PY) exp/loso_fit.py fit
	$(RUN)::loso --out-tag qwen14b --model $(Q14)
	$(RUN)::fresh
	$(RUN)::unbundle_validate && $(RUN)::unbundle

# Qwen2.5-32B on the 7B messages (H200): baseline + activations, refits, removal check, placement matrix
x32b:
	$(RUN)::prefetch
	$(RUN)::stage32
	$(PY) exp/xmodel_analyze.py probes32
	DATA_DIR=data_models/qwen32b $(PY) exp/loso_fit.py global && DATA_DIR=data_models/qwen32b $(PY) exp/loso_fit.py fit
	$(RUN)::loso --out-tag qwen32b --model $(Q32) --gpu H200 --messages /vol/data/messages.jsonl
	$(call interv,data_models/qwen32b)
	$(RUN)::redund --which 7b && $(RUN)::redund --which 32b
	$(PY) exp/redundancy_analysis.py run 7b 32b 14b
	$(PY) exp/verify_32b_ablation.py

# Verification extensions: frozen matrices in data/verify/matrices (rebuilt from the same code), hook validation first
verify:
	$(PY) exp/verify_matrix.py base
	$(RUN)::vdosefit && $(PY) exp/verify_matrix.py s1
	$(RUN)::vvalidate --spec v7b_dose && $(RUN)::vvalidate --spec v32b
	for m in s1_7b s2_fresh s3_32b s4_14b; do $(RUN)::verify --matrix $$m --maxc 6; done
	$(RUN)::vg4a_capture && $(PY) exp/verify_matrix.py g4a && $(RUN)::verify --matrix g4a_eval
	$(RUN)::vg4b_gen && $(PY) exp/verify_matrix.py g4b && $(RUN)::verify --matrix g4b_eval
	$(PY) exp/verify_matrix.py v2 && $(RUN)::vvalidate2 && $(RUN)::verify --matrix v2_s5s6 --maxc 6

# Tables and figures from committed outputs; the guarded steps need the gitignored arrays written by the GPU phases
analysis:
	$(PY) exp/figs_results.py
	$(PY) exp/loso_analyze.py && DATA_DIR=data_models/qwen14b $(PY) exp/loso_analyze.py
	$(PY) exp/fresh_analyze.py && $(PY) exp/unbundle_analyze.py
	$(PY) exp/redundancy_analysis.py report && $(PY) exp/fig_redundancy.py
	$(PY) exp/phase0_free.py parse && $(PY) exp/verify_analyze.py && $(PY) exp/verify2_analyze.py
	for m in qwen14b phi35mini; do $(PY) exp/summarize_replication.py $$m; done
	@if [ -f data/acts/C1.npz ]; then $(PY) exp/figs_probes.py && $(PY) exp/fig_models.py && $(PY) exp/span_vs_decision.py && \
	  $(PY) exp/xmodel_analyze.py geometry && $(PY) exp/phase0_free.py all; \
	else echo "skipped Fig S3/S4/S6 and probe geometry: needs activations (make main replication x32b)"; fi
	$(PY) exp/xmodel_analyze.py compare
	@if [ -d data/redund/7b/caps ]; then $(PY) exp/redund_behaviour_analyze.py; \
	else echo "skipped Fig 3 (placement): needs captures (make x32b)"; fi

narcbench:
	[ -d external/narcbench ] || git clone https://github.com/aaronrose227/narcbench external/narcbench
	cd supp_narcbench && uv venv -p 3.12 .venv && uv pip install -p .venv/bin/python numpy scikit-learn matplotlib huggingface_hub scipy && .venv/bin/python analysis.py
