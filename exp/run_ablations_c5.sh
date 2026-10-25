#!/bin/bash
# Second batch: C1-vs-C5 (scheme-mismatch) directions; conditions include C5. Outputs: data/interv/<tag>/
cd "$(dirname "$0")/.." && mkdir -p data/interv/logs
run() { .venv/bin/modal run exp/modal_app.py::intervene --direction-file "data/directions/$1.npy" --mode ablate \
          --layers all --conditions C1,C2,C4,C5 --tag "abl_$1" > "data/interv/logs/abl_$1.log" 2>&1; echo "done $1 $?"; }
for d in detect_C1vC5_mean_dm detect_C1vC5_last_dm detect_C1vC5_full_dm; do run $d & sleep 2; done; wait
