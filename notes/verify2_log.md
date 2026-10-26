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
