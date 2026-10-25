#!/bin/bash
# Replication driver for a non-default model. Usage:
#   exp/run_replication.sh <hf_model_id> <out_tag> ablate            # baseline + 8 ablations (C1-C5), then probes post
#   exp/run_replication.sh <hf_model_id> <out_tag> steer <layer> <a1,a2,...>   # steering along HOT-COLD at <layer>
# Requires DATA_DIR=data_models/<out_tag> probes baseline to have been run (directions in data_models/<tag>/directions).
# For OLMo-2 export CUTWIRE_TRANSFORMERS=4.48.3 before calling.
cd "$(dirname "$0")/.." || exit 1
MODEL=$1; TAG=$2; WHAT=$3
D=data_models/$TAG; mkdir -p "$D/logs"
if [ "$WHAT" = ablate ]; then
  .venv/bin/modal run exp/modal_app.py::intervene --model "$MODEL" --out-tag "$TAG" --mode none \
      --conditions C1,C2,C3,C4,C5 --no-acts --tag baseline > "$D/logs/baseline.log" 2>&1 &
  run() { .venv/bin/modal run exp/modal_app.py::intervene --model "$MODEL" --out-tag "$TAG" \
            --direction-file "$D/directions/$1.npy" --mode ablate --layers all --conditions C1,C2,C3,C4,C5 \
            --tag "abl_$1" > "$D/logs/abl_$1.log" 2>&1
          echo "done $1 $?"; }
  DIRS="state_C1_last_2d detect_C1vC4_mean_lr detect_C1vC4_mean_dm detect_C1vC3_full_dm detect_C1vC5_last_dm
        random1d_0 random1d_1 random2d_0"
  for d in $DIRS; do run $d & sleep 3; done; wait
  for d in $DIRS; do  # sequential: each post loads several GB of activations
    DATA_DIR=$D .venv-analysis/bin/python exp/probes.py post "$D/interv/abl_$d" > "$D/logs/post_$d.log" 2>&1
    echo "post $d $?"
  done
elif [ "$WHAT" = steer ]; then
  LAYER=$4
  for a in ${5//,/ }; do
    .venv/bin/modal run exp/modal_app.py::intervene --model "$MODEL" --out-tag "$TAG" \
      --direction-file "$D/directions/state_hotcold_L$LAYER.npy" --mode steer --layers "$LAYER" --alpha "$a" \
      --conditions C1 --no-acts --tag "steer_state_L${LAYER}_a$a" > "$D/logs/steer_$a.log" 2>&1 &
    sleep 3
  done; wait
fi
echo all done
