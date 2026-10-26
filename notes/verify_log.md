# Verification extension: execution log (plan notes/verify_plan.md @ 7652382; approved cap $8.00 TOTAL new spend)

## Pre-launch gates
- Analysis frozen: `exp/verify_analyze.py` committed BEFORE any A0/S/G run exists. It was code-tested only on old, already-analysed
  runs relabelled into the new layout under /tmp (no new outcomes existed). Any later change will be logged below, marked
  outcome-independent (bug/crash fixes found before reading results) or outcome-driven.
- Spend meter: `modal billing report --for "this month"` = $42.4477 at 01:08 UTC Oct 26, equal to the Phase 0 reference $42.448:
  no spend between Phase 0 and this extension. Earlier spending is excluded from the cap by this baseline and none is omitted.
  The "today" bucket rolled over at 00:00 UTC, so the month total is used. No apps running. The workspace's only other project
  (`unlearn-ortho`) has no running apps; any spend from it would land in the delta (conservative).
- Hard-cap enforcement (`exp/verify_guard.py`): spent = max(billed delta, Σ computed upper bounds of finished stages). The
  computed bound is n_containers × (measured local wall + 60 s) × GPU rate × 1.10. A stage launches only if spent + its WORST
  CASE n_containers × (wall_limit + 60 s) × rate × 1.10 ≤ $8.00. The wall limit is enforced locally (the client is interrupted
  and the app stopped) and in Modal (per-input timeout = limit, max_containers = n, retries = 0, scaledown 15 s). Stages run
  sequentially, so there are no outstanding commitments at launch.
- G4 code paths added before any run:
  - `env.make_episode`/`episodes` take `schemes`/`prefix`; original episodes were asserted byte-identical (306/306).
  - `Runner.g4b_gen`, `capture_set`, plus entrypoints `vg4a_capture`/`vg4b_gen`.
  - `exp/verify_matrix.py g4a/g4b`: the fit reproduces the stored fold-0 state span exactly and the PC2 control to |cos| ≥ 0.99999994.

## Stages
- A0 dose fit (01:14 UTC): wall 33 s in-app. Guard loop bug (EOF spin) held the client to the 360 s limit, so the ledger keeps a
  conservative $0.251 bound. Fixed in `exp/verify_guard.py` (outcome-independent). Builder bug (int vs string keys in
  `verify_matrix.py s1`) fixed before S1 was built (outcome-independent).
- Doses (`data/verify/dose_alpha.json`, training folds only, symmetric common-target rule, no dose > 1):
  - Over ALL prompt positions E_PC2 > E_M2 at nearly every late layer (fold-median ratio 1.1–7.8; < 1 only in a few fold-layers
    at L20, L22–25, L27).
  - So c_M2 ≈ 1 (fold-layer min 0.62) and c_PC2 is median 0.37–0.96 (min 0.16).
  - The diagnostic excluding position 0 and the decision token gives the same ratios.
  - These are uncentred removed energy, not centred variance; matching is on unablated prompt-only trajectories.
- A0 validation (`v7b_dose`): all pre-stated criteria met.
  - Hooked L18–27 max ratio 9.3e-5 (prefill) / 1.7e-4 (decode).
  - emb + L0–17 bitwise equal to none (19/19).
  - 11 decode calls per location.
  - Half dose: first-hooked prefill ratio 0.5013× none.
  - Full-dose values equal the earlier unbundle validation to all digits (unchanged numerics).
- Ledger after A0: computed $0.306, billed (lagging) $0.017.
- S1 smoke OK (15 cfgs, meter present only on ablation cfgs, per-layer doses recorded). S1 main: 90/90 cfgs, 160 s wall, 6 × L40S; computed upper $0.805.
- V32 validation passed (hooked ≤ 3.3e-4 prefill/decode; M2_late emb+L0–41 bitwise = none 43/43; PRE_LATE emb 1/1; EMB_LATE L0–41 differ 42/42). S3 smoke OK; S3 main 30/30 cfgs, 121 s, computed upper $1.506.
- S4 smoke OK; S4 main 42/42 cfgs, 196 s, computed upper $0.918.
- S2 smoke OK; S2 main 240/240 cfgs, 383 s, computed upper $1.320.
- G4a capture (4 × 306, prompt-only) 88 s, computed upper $0.088; refit on 4 × 255 training-scheme episodes per fold; directions' sha256 in data/verify/g4a/provenance.json; g4a_eval matrix committed before eval.
- G4a smoke OK; G4a eval 30/30 cfgs, 297 s, computed upper $0.213.
- G4b generation + 306-episode baseline + capture: 96 s, computed upper $0.093. Feasibility gate PASSED (C1 strict 0.931 ≥ 0.70; exact bet 0.931; C6 1.000; recall 0.938; ARC 0.97). LOSO refit on new schemes + g4b_eval matrix committed before eval.
- S/G stages complete; all apps stopped (17 cutwire apps, state stopped). Analysis run unchanged from the frozen version
  (no post-freeze edits to exp/verify_analyze.py). 02:03 UTC ledger: spent(max) $6.210  billed $3.669  computed $6.210  cap $8.0  remaining $1.790.
  No stage was refused, retried or wall-limited (A0_dosefit's 361 s is the guard-loop artefact noted above). The reserve was
  not used.
