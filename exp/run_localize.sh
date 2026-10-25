#!/bin/bash
# Layer localisation: ablate per-layer directions in early (0-15) vs late (16-27) layers only. Behaviour only (--no-acts).
cd "$(dirname "$0")/.." && mkdir -p data/interv/logs
E=$(seq -s, 0 15 | sed "s/,$//"); L=$(seq -s, 16 27 | sed "s/,$//")
for d in state_C1_last_2d detect_C1vC4_mean_dm; do
  for part in early late; do
    ly=$E; [ $part = late ] && ly=$L
    .venv/bin/modal run exp/modal_app.py::intervene --direction-file data/directions/$d.npy --mode ablate --layers $ly \
      --conditions C1 --no-acts --tag "loc_${d}_$part" > data/interv/logs/loc_${d}_$part.log 2>&1 &
    sleep 2
  done
done; wait; echo done
