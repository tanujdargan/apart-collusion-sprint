# Held-out state decodability: message-span mean vs decision token (unablated, CPU; exp/span_vs_decision.py)

Outer LOSO (6 folds), training-only fitting, no selection; n = 306 per model; bootstrap over episodes stratified by scheme. Chance 0.333. `span_mean` is a pooled feature (mean over message tokens), not per-token evidence.

| model | span_mean all-layer | decision all-layer | span_mean late | decision late | Δ span−decision late [CI] | first layer ≥0.8 (span / decision) |
|---|---|---|---|---|---|---|
| 7b | 0.245 [0.213, 0.277] | 0.547 [0.522, 0.571] | 0.226 [0.192, 0.261] | 0.918 [0.890, 0.944] | -0.692 [-0.736, -0.647] | None / 18 (of 28) |
| 14b | 0.148 [0.128, 0.168] | 0.557 [0.542, 0.573] | 0.067 [0.048, 0.087] | 0.988 [0.976, 0.997] | -0.921 [-0.941, -0.898] | None / 26 (of 48) |
| 32b | 0.291 [0.260, 0.323] | 0.553 [0.530, 0.576] | 0.385 [0.350, 0.420] | 0.966 [0.946, 0.984] | -0.581 [-0.618, -0.544] | 43 / 42 (of 64) |

Per-layer curves: data/span_vs_decision.json (per_layer).
