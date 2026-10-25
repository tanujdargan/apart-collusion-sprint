#!/bin/bash
# Reviewer fixes: C6 (explicit, no covert channel) controls, variance-matched PC control, raw-LR detector,
# random steering control. Outputs data/interv/<tag>/
cd "$(dirname "$0")/.." && mkdir -p data/interv/logs
M=".venv/bin/modal run exp/modal_app.py::intervene"
go() { tag=$1; shift; $M "$@" --tag $tag > data/interv/logs/$tag.log 2>&1; echo "done $tag $?"; }
go base_C6 --mode none --conditions C6 --no-acts --no-arc &
sleep 2; go abl_state_C1_last_2d_C6 --direction-file data/directions/state_C1_last_2d.npy --mode ablate --layers all --conditions C6 --no-acts --no-arc &
sleep 2; go abl_detect_C1vC4_mean_dm_C6 --direction-file data/directions/detect_C1vC4_mean_dm.npy --mode ablate --layers all --conditions C6 --no-acts --no-arc &
sleep 2; go abl_detect_C1vC5_last_dm_C6 --direction-file data/directions/detect_C1vC5_last_dm.npy --mode ablate --layers all --conditions C6 --no-acts --no-arc &
sleep 2; go abl_random2d_0_C6 --direction-file data/directions/random2d_0.npy --mode ablate --layers all --conditions C6 --no-acts --no-arc &
sleep 2; go abl_pc_ctrl_2d --direction-file data/directions/pc_ctrl_2d.npy --mode ablate --layers all --conditions C1,C2,C4,C6 --no-acts &
sleep 2; go abl_detect_C1vC4_mean_lrraw --direction-file data/directions/detect_C1vC4_mean_lrraw.npy --mode ablate --layers all --conditions C1,C2,C3,C4 &
sleep 2; go steer_random_L20_a40 --direction-file data/directions/random1d_3.npy --mode steer --layers 20 --alpha 40 --conditions C1 --no-acts &
sleep 2; go steer_random_L20_a-40 --direction-file data/directions/random1d_3.npy --mode steer --layers 20 --alpha -40 --conditions C1 --no-acts &
wait
