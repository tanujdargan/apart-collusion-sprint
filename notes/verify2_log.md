# Final targeted extension: log (plan notes/verify2_plan.md; cap $2.00 total new spend)
- Code added before any run:
  - `_intervene_many` takes optional `c1_seeds`/`c1_temperature`/`c1_top_p` for sampled C1 draws (greedy conditions
    unchanged);
  - `_receivers(gen=...)`;
  - `validate_hooks` accepts 1-D [L,D] directions and sampling kwargs;
  - new `validate_dose_algebra`;
  - `verify --smoke` now smoke-tests EVERY job of the first container (previously only the first job);
  - guard ledger/cap configurable by env.
- `exp/verify2_analyze.py` was code-tested only on S1 runs relabelled under /tmp (no new outcomes existed). Contexts
  checked: 102, none recurring across schemes.
- Spend baseline: month-to-date $46.1167 at 02:35 UTC. The previous extension's billed delta was stable at $3.669 from
  02:02 to 02:31, so any late billing from it would be charged here (conservative).
- Analysis fix 1 (outcome-independent crash): validation parser upstream range for the unhooked 'none' cell ran past L27 (KeyError L28); capped at 28. No outcome logic changed.
- V2 validation (1 × L40S, 46 s, computed upper $0.063):
  - Full-dose placements (dm, pc1d × L18/all/EMB_LATE/PRE_LATE greedy; M2, dm, pc1d sampled): hooked ≤ 4.1e-4 in prefill
    and decode; upstream prefill bitwise = none (19/19 for L18; emb 1/1 for PRE_LATE).
  - Fractional dose, same incoming state: |span ratio − (1−d)| ≤ 0.0038 prefill / 0.0067 decode (M2_match), ≤ 0.0044
    (PC2_match); energy ratio 0.996–1.017; orthogonal part of the removed vector ≤ 15% of its norm (bf16 output rounding).
  - Repeat-seed: none s0, M2_match s3 and PC2_match s5 identical.
  - LIMITATION FOUND: on this 9-episode subset (all scheme-0 COLD) every sampled output equalled the greedy output in every
    cell, so the repeat-seed and cross-container checks were made on draws that never diverged from greedy. They are WEAK
    evidence; determinism of divergent sampled draws is not established and is not claimed.
  - Sampling config checked on CPU (exp/check_sampling_processors.py, transformers 4.46.3): S5 kwargs build only
    TemperatureLogitsWarper(0.7). The model defaults (top_k 20, top_p 0.8, repetition penalty) are overridden; greedy
    builds none.
