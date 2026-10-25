#!/bin/bash
# Launch all ablation runs in parallel on Modal. Outputs: data/interv/<tag>/
cd "$(dirname "$0")/.." && mkdir -p data/interv/logs
run() { .venv/bin/modal run exp/modal_app.py::intervene --direction-file "data/directions/$1.npy" --mode ablate \
          --layers all --conditions C1,C2,C3,C4 --tag "abl_$1" > "data/interv/logs/abl_$1.log" 2>&1; echo "done $1 $?"; }
for d in detect_C1vC4_mean_dm detect_C1vC4_mean_lr detect_C1vC3_full_dm detect_C1vC2_mean_dm state_C1_last_2d \
         random1d_0 random1d_1 random1d_2 random2d_0 random2d_1 random2d_2; do run $d & sleep 2; done; wait
