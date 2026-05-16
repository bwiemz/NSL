# Tier B Planner — D-2 Floor Derivation Findings

**Date:** 2026-05-16
**Branch:** `worktree-feat-pca-tier-b-dispatch`
**Spec inheritance:** Dispatch spec §6 (protocol unchanged); planner spec §7 (consumption layer moved to runtime).
**Plan:** Task P-1 (re-dispatched after P-1.0 bench-fixture expansion in commit `267c012d`).
**Hardware:** Windows 11 / CUDA 13.2 / RTX 5070 Ti (sm_120) — driver `591.86`.
**Bench unit:** median wall-time reported in **microseconds (µs)** per IR-012 contract; CSV columns are `_us` not `_ns`.

## Reproducer (P-1.1)

Gate-fixture (`gate_4096`, sparsity=50%): win = **74.75%** (ON median 9264.35 µs, OFF median 36685.31 µs).
Matches `2026-05-13-tier-b-m2-m6-findings.md`'s 73.33% within ±10%? **YES** (drift = +1.42 pp, well inside the ±10 pp envelope).

The full-sweep gate_4096 datapoint (74.74%) and this independent reproducer (74.75%) agree to within 0.01 pp, confirming bench stability across separate cargo runs.

## Sweep results

See `2026-05-15-tier-b-floor-derivation.csv`.

| seq_len | wall_time_on (µs) | wall_time_off (µs) | win %  | classification           |
|---------|-------------------|--------------------|--------|--------------------------|
| 128     | 176.58            | 291.36             | 39.40% | above 10% floor          |
| 256     | 221.66            | 590.24             | 62.45% | above 10% floor          |
| 512     | 309.50            | 1164.03            | 73.41% | above 10% floor          |
| 1024    | 604.38            | 2319.87            | 73.95% | above 10% floor          |
| 2048    | 2334.66           | 9187.10            | 74.59% | above 10% floor          |
| 4096    | 9231.74           | 36552.35           | 74.74% | above 10% floor (gate)   |
| 8192    | 36105.86          | 145162.52          | 75.13% | above 10% floor          |

All 7 measured `skip_ratio` values match the fixture's `target_sparsity` family
(0.5 at seq=128, 0.75 at seq=256, 0.875 at seq≥512) and `skip_ratio=0` for every
OFF run, as expected.

## Curve shape

**Monotonic** (strictly increasing in win %). Win climbs steeply from 39.40% at seq=128 to ~73% by seq=512, then asymptotes near ~74–75% for seq≥1024. Interpretation: at very small seq_len the Tier-A baseline already has so little work that the SMEM-bitmap skip path's fixed overhead consumes a larger fraction of OFF runtime; once seq_len grows past ~512 the skip ratio dominates and the win plateaus near the theoretical (1 − target_sparsity) ≈ 75% bound for these fixtures.

## TIER_B_SEQ_LEN_FLOOR resolution

**`TIER_B_SEQ_LEN_FLOOR = 128`** for P-2's shared-crate declaration.

Outcome per dispatch spec §6.4 / planner spec §7.6: **clear pass** — the sweep showed positive wall-time win ≥ 10% at every tested seq_len, so the §6.4 "positive at all seq_lens → floor = 128" branch applies directly. No §7.6 case-(a) sub-threshold concern (smallest win is 39.40%, ≈4× the 10% bar) and no §7.6 case-(b) blocker (no zero or negative wins).

## Joint-outcome compatibility check (risk #9)

V-Bii-SMEM resolved `TIER_B_MAX_BAKED_SEQ_LEN = 16384`. D-2 resolved `TIER_B_SEQ_LEN_FLOOR = 128`. Compatibility:

- `FLOOR (128) <= MAX_BAKED (16384)`: **PASS**.

P-2's const assertion catches FAIL at compile time, but the values are independently compatible: the admission window `[128, 16384]` spans 7 octaves of seq_len, which is the full sweep range plus headroom on the high side (8192 is the largest measured point, MAX_BAKED extends one octave beyond).

## Cross-references

- Dispatch spec §6.4 — floor rule (clear-pass / threshold / non-monotonic).
- Planner spec §7.2 — runtime consumption layer.
- Planner spec §7.6 — case (a)/(b) split.
- Planner spec §11 risk #9 — joint-outcome compatibility.
- V-Bii-SMEM findings (`2026-05-15-tier-b-vbii-smem-findings.md`) — joint-outcome counterpart resolving MAX_BAKED.
- M2/M6 findings (`2026-05-13-tier-b-m2-m6-findings.md`) — 73.33% gate-fixture baseline used for the P-1.1 reproducer check.
- P-1.0 bench fixture expansion (commit `267c012d`) — fixtures `floor_sweep_<seq>` enabling this sweep.

## Re-run triggers (append-only log per IR-012)

- (none yet)
