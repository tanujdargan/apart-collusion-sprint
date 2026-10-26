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
