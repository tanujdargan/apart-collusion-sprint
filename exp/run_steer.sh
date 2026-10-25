#!/bin/bash
# Steering dose-response along the HOT-COLD state direction at layer 20 (receiver, all positions). Outputs data/interv/steer_*/
cd "$(dirname "$0")/.." && mkdir -p data/interv/logs
for a in -40 -20 -10 10 20 40; do
  .venv/bin/modal run exp/modal_app.py::intervene --direction-file data/directions/state_hotcold_L20.npy --mode steer \
    --layers 20 --alpha $a --conditions C1 --no-acts --tag "steer_state_L20_a$a" > "data/interv/logs/steer_$a.log" 2>&1 &
  sleep 2
done; wait; echo done
