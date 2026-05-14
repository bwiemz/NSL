# Tier B M2/M6 Measurement Findings

**Date:** 2026-05-13
**Hardware:** NVIDIA RTX 5070 Ti (sm_120, Blackwell)
**Driver:** 591.86 / CUDA 13.2
**Protocol:** Spec §8 of `docs/superpowers/specs/2026-05-13-pca-tier-b15-and-b2-design.md`
**Bench binary:** `target/release/bench.exe` (built `--features "cuda debug_kernel_instrumentation" --release`)
**Seed:** 42 (default; same `--seed` across all runs)
**Inner iterations:** 100 per outer run (CUDA-event bracketed forward kernel launch per spec §8.2)
**Outer runs:** 5 per (fixture, tier_b) combination (M6 protocol; gate + 3 sensitivity fixtures)
**Decision authority:** §3.1 (acceptance bar: ≥10% wall-time AND ≥40% skip), §10 (outcomes matrix)

---

## 1. Gate fixture (`gate_4096`) — keep/revert decision

### 1.1 Raw M6 CSV

```
fixture,outer_run,tier_b,median_us,skip_ratio,seed
gate_4096,1,on,10474.080085754395,0.875,42
gate_4096,1,off,39402.78244018555,0,42
gate_4096,2,on,9978.272438049316,0.875,42
gate_4096,2,off,38603.93524169922,0,42
gate_4096,3,on,9993.663787841797,0.875,42
gate_4096,3,off,37417.598724365234,0,42
gate_4096,4,on,9273.951530456543,0.875,42
gate_4096,4,off,36698.01712036133,0,42
gate_4096,5,on,9272.928237915039,0.875,42
gate_4096,5,off,36688.961029052734,0,42
```

### 1.2 Computed values

- `median_on`  = **9978.272 µs** (median of `[10474.08, 9978.27, 9993.66, 9273.95, 9272.93]`)
- `median_off` = **37417.599 µs** (median of `[39402.78, 38603.94, 37417.60, 36698.02, 36688.96]`)
- `walltime_win = (median_off − median_on) / median_off` = **73.33%**
- `skip_ratio_on` = **0.875** (deterministic-by-seed; identical across all 5 outer runs)

### 1.3 Threshold check (per spec §3.1)

| Threshold | Required | Actual | Result |
|---|---|---|---|
| `walltime_win >= 10%` | ≥10% | **73.33%** | **YES** (7.3× margin over threshold) |
| `skip_ratio   >= 40%` | ≥40% | **87.50%** | **YES** (2.2× margin over threshold) |

Both AND-conjoined thresholds are met with substantial margin. Decision is **KEEP**; sensitivity curve determines whether unconditional or gated.

### 1.4 Outcome

**KEEP UNCONDITIONALLY** (per §10 outcomes matrix; final classification justified by §2 sensitivity curve below — all three sparsity points yield positive `walltime_win`, so no sub-threshold-hurt regime exists that would warrant a sparsity gate).

---

## 2. Sensitivity curve (three fixtures at sparsities {10%, 50%, 90%})

### 2.1 Per-fixture summary table

| target_sparsity | achieved skip_ratio | median_us on | median_us off | walltime_win |
|---|---|---|---|---|
| 10% | **0.50** | 21234.91 | 36692.96 | **42.13%** |
| 50% | **0.875** | 9273.79 | 36693.95 | **74.73%** |
| 90% | **0.96875** | 6281.79 | 36695.17 | **82.88%** |

`sensitivity_50` cross-check: its `walltime_win` (74.73%) is within +1.40 pp of the gate fixture's (73.33%) and its skip_ratio (0.875) is bit-exact identical. Cross-check redundancy per spec §4.3.1 — **PASSES** within the ±10% tolerance band. The two 50%-sparsity measurements agree on both metrics.

### 2.2 Curve-shape inference

| Transition | Δ walltime_win | Δ skip_ratio |
|---|---|---|
| 10% → 50% | +32.60 pp | +0.375 |
| 50% → 90% | +8.15 pp | +0.09375 |

**Curve shape: SATURATING.** The 50→90 step is roughly ¼ the magnitude of the 10→50 step despite the skip-ratio step being only ¼ smaller (0.094 vs 0.375). This is the canonical saturation pattern: each marginal point of additional skipped work yields a diminishing wall-time return as the kernel approaches its no-launch-overhead floor (already at 6.28 ms vs Tier-A's 36.7 ms — ~17% of the no-skip cost; remaining cost is launch + tile-zero work that cannot be eliminated).

**Inference per spec §4.3.2:** the saturating shape is consistent with the architectural reasoning in §3.2 (skip-check amortization model). The kernel approaches a floor as sparsity → 1, set by launch overhead + the residual tile-residency loop. No threshold below which Tier B hurts; **no sparsity gate is needed**. The keep decision is unconditional.

### 2.3 Discovery note: target_sparsity vs achieved skip_ratio

The fixture-generated mask's `target_sparsity` parameter does **not** linearly map to the achieved `skip_ratio` at this fixture scale (seq_len=4096, head_dim=64, block_q=block_kv=64, 8 segments). Observed:

| target_sparsity | achieved skip_ratio | ratio |
|---|---|---|
| 0.1 | 0.5    | 5× |
| 0.5 | 0.875  | 1.75× |
| 0.9 | 0.96875 | 1.08× |

This is a known property of the mask generator's segment-packing-vs-tile-grid alignment at small seq_len: a fraction of in-segment tokens placed within a tile boundary suffices to admit the tile (not skip it), so block-granularity skip ratio grows faster than token-granularity in-segment fraction. The MONOTONIC trend across `(0.5 → 0.875 → 0.96875)` skip ratios is what the curve-shape analysis depends on, not the literal target values. The fixture matrix's value here is **three distinct points on a meaningful curve**, not a calibrated 10/50/90 grid.

### 2.4 Diagnostic: does Tier B HURT at low sparsity?

The 10% target / 50% achieved point answers spec §4.3.1's diagnostic question:

> "does Tier B actively hurt at low sparsity, or merely under-help?"

**Answer:** at achieved 50% skip ratio, Tier B delivers 42% wall-time win — it does **not** hurt. The skip-check overhead per tile is comfortably amortized even at half-sparsity. This means the curve has no sub-threshold-hurt regime; sparsity-gated dispatch is not required.

---

## 3. Parity tier results

All six `parity_N` fixtures pass bit-identical byte-equality assertions (verified during B1.5-3, commit `f3c1bb4f`). No re-run needed here. The byte-equality property is correctness-preserving by construction (skipped tiles contribute exactly zero per spec §4.2 / §7.1).

### 3.1 Parity skip ratios at canonical fixtures (from M2 sweep)

| fixture | skip_ratio | median_us on | median_us off | implied walltime_win |
|---|---|---|---|---|
| parity_1 | 0.75    | 3402.34 | 9311.58 | 63.5% |
| parity_2 | 0.75    | 52961.28 | 146860.99 | 63.9% |
| parity_3 | 0.875   | 2394.08 | 9315.55 | 74.3% |
| parity_4 | 0.9375  | 1893.54 | 9316.10 | 79.7% |
| parity_5 | 0.5     | 5412.61 | 9315.46 | 41.9% |
| parity_6 | 0.875   | 2395.62 | 9311.71 | 74.3% |

All six parity fixtures fall in the regime where Tier B delivers positive wall-time win consistent with the gate-fixture and sensitivity curve. Parity_5 at 0.5 skip ratio matches sensitivity_10's behavior (41.9% ≈ 42.13%) — reinforces the curve-shape inference across an independent fixture point.

---

## 4. M2 sweep (10 fixtures × 2 tier_b)

```
fixture,tier_b,skip_ratio,median_us,seed
gate_4096,on,0.875,9243.712425231934,42
gate_4096,off,0,36692.06237792969,42
sensitivity_10,on,0.5,21233.407974243164,42
sensitivity_10,off,0,36695.04165649414,42
sensitivity_50,on,0.875,9244.128227233887,42
sensitivity_50,off,0,36695.26290893555,42
sensitivity_90,on,0.96875,6266.528129577637,42
sensitivity_90,off,0,36694.68688964844,42
parity_1,on,0.75,3402.3358821868896,42
parity_1,off,0,9311.58447265625,42
parity_2,on,0.75,52961.280822753906,42
parity_2,off,0,146860.99243164063,42
parity_3,on,0.875,2394.0799236297607,42
parity_3,off,0,9315.5517578125,42
parity_4,on,0.9375,1893.5359716415405,42
parity_4,off,0,9316.096305847168,42
parity_5,on,0.5,5412.6081466674805,42
parity_5,off,0,9315.45639038086,42
parity_6,on,0.875,2395.616054534912,42
parity_6,off,0,9311.712265014648,42
```

Tier-B-off `skip_ratio=0` is the expected behavior — when Tier B is disabled, `emit_skip_predicate` is not emitted and no tiles are recorded as skipped. (The `debug_kernel_instrumentation` writeback is conditional on the predicate path being active, so `skip_ratio=0` reflects "the writeback was not invoked" rather than "0 tiles were skipped"; either interpretation is consistent with the data.)

---

## 5. Decision rationale (per spec §10 outcomes matrix)

Per §10's three-way outcomes matrix:

| Outcome | Trigger | This run |
|---|---|---|
| **Keep unconditionally** | Gate clears both thresholds AND sensitivity curve is monotonic-linear or saturating (no sub-threshold hurt) | ✅ **MET** |
| Keep with sparsity gate | Gate clears both thresholds AND sensitivity curve is thresholded (sub-threshold hurt) | not triggered |
| Revert | Gate fails either threshold | not triggered |

**Final outcome: KEEP UNCONDITIONALLY.**

**Justification:**
1. Gate fixture clears the `≥10% wall-time win AND ≥40% skip ratio` AND-conjunction by substantial margin (7.3× on wall-time, 2.2× on skip ratio).
2. Sensitivity curve is saturating — all three measured points produce positive wall-time win (42% / 75% / 83%), monotonic in skip ratio. No regime exists where Tier B hurts.
3. Therefore no sparsity gate is needed; `should_emit_tier_b(config)` does not require a `estimated_sparsity(config) >= threshold` predicate.
4. Tier B is correctness-preserving by construction (skipped tiles contribute exactly zero); all six parity fixtures verified byte-identical Tier-B-on vs Tier-B-off (B1.5-3, commit `f3c1bb4f`).

**Per §3.3 non-negotiability discipline:** these results pass the AND-conjunction by such a wide margin that no post-measurement threshold negotiation would be required even under significantly pessimistic re-measurement.

---

## 6. ±10% reproducibility tolerance check (spec §8.5)

Computed standard deviation across the 5 outer runs of M6 per (fixture, tier_b):

| fixture | tier_b | mean (µs) | σ (µs) | σ as % of mean |
|---|---|---|---|---|
| gate_4096 | on | 9798.59 | 519.18 | **5.30%** |
| gate_4096 | off | 37762.26 | 1204.48 | 3.19% |
| sensitivity_10 | on | 21234.72 | 1.08 | 0.01% |
| sensitivity_10 | off | 36693.11 | 0.97 | 0.00% |
| sensitivity_50 | on | 9274.23 | 1.25 | 0.01% |
| sensitivity_50 | off | 36694.03 | 2.32 | 0.01% |
| sensitivity_90 | on | 6282.34 | 1.05 | 0.02% |
| sensitivity_90 | off | 36694.81 | 1.60 | 0.00% |

All within the ±10% tolerance band per §8.5; **acceptance, not investigation.** Note that `gate_4096` (the FIRST measurement after cold start) shows visibly higher variance (5.30% on, 3.19% off) — runs 1-3 are slower than runs 4-5, decaying monotonically (10474 → 9978 → 9994 → 9274 → 9273). Consistent with cold-start / clock-rate warmup observed on RTX 5070 Ti; subsequent fixtures' near-zero σ confirms steady-state once GPU and driver caches are warm. **No protocol change required** — the median-of-5 absorbs the warmup transient, and the warmup is captured in the protocol as expected variance.

---

## 7. Outcome's downstream actions (per §10)

Outcome = **keep unconditionally** → per §10 row 1:

1. **`should_emit_tier_b(config)` stays returning false today.** The planner-side dispatch decision (when to activate Tier B for non-bench callers) is its own downstream spec, not in this PR. The bench binary continues overriding via `synthesize_flash_attention_ptx_v2_with_tier_b(config, Some(...))`.
2. **B.2 milestones (B2-1, B2-2, B2-3) are unblocked for execution** in subsequent work.
3. **No `#[ignore]` markers** applied (which would have been the revert path).
4. **6-month decay timer is NOT started** (which would have been the revert path).
5. **Memory file updated** to record `keep unconditionally` outcome with `walltime_win=73.33%` and `skip_ratio=0.875` for the gate fixture.

---

## 8. References

- `docs/superpowers/specs/2026-05-13-pca-tier-b15-and-b2-design.md` — design spec (§3.1 acceptance bar, §3.3 non-negotiability, §4.3 sensitivity tier, §8 measurement protocol, §10 outcomes matrix, §11.1 milestone table)
- `docs/superpowers/plans/2026-05-13-pca-tier-b15-and-b2-implementation.md` — plan task B1.5-6
- `scripts/measure_tier_b_m6.sh` — M6 wall-time median-of-5 orchestration
- `scripts/measure_tier_b_m2.sh` — M2 skip-ratio sweep orchestration
- `crates/nsl-codegen/src/bin/bench.rs` — bench binary entry point (B1.5-1)
- `crates/nsl-codegen/src/bin/bench/launch.rs` — CUDA-event timing harness
- `crates/nsl-codegen/src/bin/bench/fixtures.rs` — gate + sensitivity + parity fixture registry
- `crates/nsl-codegen/tests/pca_tier_b_m3_parity.rs` — bit-identical parity assertions (B1.5-3, commit `f3c1bb4f`)
- `/c/tmp/m6_gate.csv`, `/c/tmp/m6_sensitivity_{10,50,90}.csv`, `/c/tmp/m2_sweep.csv` — raw measurement data (this session)

---

## 9. Revision changelog

- **2026-05-13** — initial. M6 + M2 measurements complete; outcome = keep unconditionally with 73.33% gate wall-time win and 87.5% gate skip ratio. Sensitivity curve saturating; no sparsity gate required. B.2 milestones unblocked.
