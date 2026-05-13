# Tier B Measurement Procedure (M2 + M6)

**Spec:** §7 of `docs/superpowers/specs/2026-05-02-pca-tier-b-tile-skip-design.md` (M2: FLOP reduction, M6: single_doc regression).

## Prerequisites

- CUDA toolkit 13.x with Nsight Compute (`ncu`) installed at `$CUDA_PATH/bin/ncu`.
- A bench binary `nsl-codegen-bench` with `--features cuda` that:
  - Loads a packing fixture by name (`standard_3doc`, `single_doc`, etc.).
  - Launches the Tier B forward kernel via `synthesize_flash_attention_ptx_v2_with_tier_b`.
  - Supports `--tier-b={on,off}` switch.
  - Supports `--emit-time-only` for wall-time-only output (useful for median computation).
- Hardware: sm_80 / sm_90 / sm_120 — Nsight metrics are architecture-stable.

## Bench harness deferral

The bench binary doesn't currently exist in this repo. The Tier B forward kernel + instrumentation are landed (Tasks 1-10); building the launch harness is deferred to a Tier B.1.5 follow-up alongside the M3 parity test harness.

Reference launch shape: `crates/nsl-codegen/tests/pca_tier_a_forward_correctness::launch_forward`. Adapt with a `skip_decisions_ptr` arg (gated on `debug_kernel_instrumentation` for the M3 path; M2/M6 don't need instrumentation).

## M2 — FLOP reduction (≥30% on standard_3doc)

Run `scripts/measure_tier_b_m2.sh`. The script:

1. Builds the bench binary in release mode.
2. Runs `ncu` with the three FLOP counters (`fadd`, `ffma`, `fmul`).
3. Computes `FLOPs = fadd + 2*ffma + fmul` per spec §7 M2 formula.
4. Reports the ratio `1 - (B/A)`.

Pass: ≥ 0.30.

## M6 — single_doc wall-time regression (≤1%)

Run `scripts/measure_tier_b_m6.sh`. The script:

1. Runs the bench binary 5 times in each mode (`--tier-b={off,on}`).
2. Computes the median wall time per mode.
3. Reports the ratio `B/A`.

Pass: ≤ 1.01.

## When to run

- Before merging any Tier B emission change touching the preamble or predicate.
- After any FA2 v2 kernel restructuring that could shift the kv-tile loop body or scratch register allocation.
- As part of a quarterly regression-check sweep.

Results recorded in this document under a dated heading.

## Results log

### TBD — initial measurement

(Fill in after the bench harness lands and the first measurement run completes. Expected: M2 ≥ 0.30 (paper claims ~0.50 for 3-doc packing), M6 ≤ 1.01 (single_doc is worst case where Tier B is pure overhead).)
