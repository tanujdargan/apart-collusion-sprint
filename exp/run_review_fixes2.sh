#!/bin/bash
# 1-D variance-matched control + C6 (explicit-state) coverage for all remaining ablation directions.
cd "$(dirname "$0")/.." && mkdir -p data/interv/logs
M=".venv/bin/modal run exp/modal_app.py::intervene"
go() { tag=$1; shift; $M "$@" --tag $tag > data/interv/logs/$tag.log 2>&1; echo "done $tag $?"; }
go abl_pc_ctrl_1d --direction-file data/directions/pc_ctrl_1d.npy --mode ablate --layers all --conditions C1,C2,C4,C6 --no-acts &
for d in detect_C1vC4_mean_lr detect_C1vC4_mean_lrraw detect_C1vC2_mean_dm detect_C1vC3_full_dm detect_C1vC5_mean_dm random1d_0; do
  sleep 2; go abl_${d}_C6 --direction-file data/directions/$d.npy --mode ablate --layers all --conditions C6 --no-acts --no-arc &
done; wait
