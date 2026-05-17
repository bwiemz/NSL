# Tier B Planner — V-Bii-SMEM Probe Findings

**Date:** 2026-05-15
**Branch:** `worktree-feat-pca-tier-b-dispatch`
**Spec:** [`2026-05-15-pca-tier-b-planner-design.md`](2026-05-15-pca-tier-b-planner-design.md) §3
**Plan:** [`2026-05-15-pca-tier-b-planner-implementation.md`](../plans/2026-05-15-pca-tier-b-planner-implementation.md) Task P-0
**Hardware:** Windows 11 / CUDA 13.2 / RTX 5070 Ti (sm_120, Blackwell) / NVIDIA driver 591.86

## Probe protocol

Forced-access SMEM kernel emitted at `crates/nsl-codegen/tests/tier_b_bii_smem_probe.rs`. Per planner spec §3.2 the kernel allocates a static `seg_smem[8192]` (Tier-B-off baseline mirroring `tier_b_smem_probe.rs`) plus an extern `shmem[]` sized at launch via `cuLaunchKernel`'s `sharedMemBytes` to `compute_range_table_bytes(MAX, block, block)`. Every thread writes sentinel `0xA5` to both regions; `bar.sync 0`; lane-0 reads back from `seg_smem[0]` and writes a u32 to a global output buffer. Pass = `cuModuleLoadDataEx` + `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)` + `cuLaunchKernel` + `cuCtxSynchronize` all succeed AND the readback byte equals `0xA5`.

sm_80 PTX targets are loaded via `cuModuleLoadDataEx` on the Blackwell host; the driver JIT-compiles sm_80 PTX down to native sm_120 SASS (approach (a) per the implementation plan), so the launch outcome reflects the SMEM cap binding on the resident GPU. The `cap_bytes` column records the architecture's nominal cap (sm_80 = 100 KB, sm_120 = 99 KB) per planner spec §3.1 — that's the value the decision matrix binds, not the GPU's physical cap.

## Sweep results (12 configs)

Run command:

```bash
cargo test -p nsl-codegen --features cuda --release \
  --test tier_b_bii_smem_probe probe_full_sweep -- --nocapture --test-threads=1
```

| MAX | block | target_sm | seg_smem (B) | tier_b_extern (B) | total (B) | cap (B) | util % | outcome |
|-----|-------|-----------|--------------|-------------------|-----------|---------|--------|---------|
| 4096  | 32 | sm_80  | 8192 | 1024 | 9216  | 102400 |  9.00 | Pass |
| 4096  | 64 | sm_80  | 8192 |  512 | 8704  | 102400 |  8.50 | Pass |
| 8192  | 32 | sm_80  | 8192 | 2048 | 10240 | 102400 | 10.00 | Pass |
| 8192  | 64 | sm_80  | 8192 | 1024 | 9216  | 102400 |  9.00 | Pass |
| 16384 | 32 | sm_80  | 8192 | 4096 | 12288 | 102400 | 12.00 | Pass |
| 16384 | 64 | sm_80  | 8192 | 2048 | 10240 | 102400 | 10.00 | Pass |
| 4096  | 32 | sm_120 | 8192 | 1024 | 9216  | 101376 |  9.09 | Pass |
| 4096  | 64 | sm_120 | 8192 |  512 | 8704  | 101376 |  8.59 | Pass |
| 8192  | 32 | sm_120 | 8192 | 2048 | 10240 | 101376 | 10.10 | Pass |
| 8192  | 64 | sm_120 | 8192 | 1024 | 9216  | 101376 |  9.09 | Pass |
| 16384 | 32 | sm_120 | 8192 | 4096 | 12288 | 101376 | 12.12 | Pass |
| 16384 | 64 | sm_120 | 8192 | 2048 | 10240 | 101376 | 10.10 | Pass |

> **Note (per planner spec §3.2.1 co-measurement):** the `seg_smem (B)` column is the Tier-B-off baseline; `tier_b_extern (B)` is Tier B's incremental SMEM contribution; `total (B)` is the Tier-B-on combined allocation. The probe doubles as a SMEM-allocation regression checkpoint for the entire FA-2 v2 kernel.

All 12 configurations pass; outcomes are strictly monotonic with respect to MAX (larger MAX → larger range-table bytes → larger total SMEM, with no fail boundary crossed at any MAX). Results are reproducible across runs (single-threaded test execution; CUDA primary context retained per process).

## Outcome (per §3.4 decision matrix)

**Sub-variant resolved: B-ii unrestricted.**

The MAX=16384/block=32 configuration — the worst-case combination in the sweep (smallest block → largest `num_q_tiles + num_kv_tiles` → largest range-table footprint) — passes on **both** sm_80 (12.00% util) and sm_120 (12.12% util). Per the §3.4 row "MAX=16384/block=32 fits across all configs → **B-ii unrestricted**", the highest tier of single-emission is feasible.

This is dominated by an architectural property of `compute_range_table_bytes`: the table is `2 × (num_q + num_kv) × sizeof(u16) = 4 × (seq_len/block_q + seq_len/block_kv)` bytes. For the worst-case sweep config (seq_len=16384, block_q=block_kv=32) this is 4 × (512 + 512) = 4096 bytes — only ~4% of the 99 KB cap. Even adding the 8 KB static `seg_smem[]` baseline, total SMEM stays near 12 KB, well below the cap on either architecture.

## SMEM headroom (for §3.5 v2-trigger ladders)

**At MAX=16384/block=32 (the binding worst case):**

- **sm_120: 12.12% utilization → <60% bucket → significant headroom.** Per §3.5 ladder, v2 trigger is *"extend baked max to MAX=32768"* with a follow-up probe. Workload-driven; no immediate action.
- **sm_80: 12.00% utilization → <60% bucket.** Same v2 trigger ladder applies to the secondary architecture.

**At MAX=8192/block=32:** Not applicable — B-ii-restricted ladder fires only when MAX=16384 doesn't fit, which is not the case here. Headroom data still recorded above for completeness (10.10% on sm_120, 10.00% on sm_80).

The MAX=16384/block=32 result has substantial room (~87% of cap unused on sm_120). The B-ii unrestricted resolution leaves enough headroom that future range-table additions (e.g., M3 debug instrumentation buffers per the existing `emit_skip_decision_writeback`, or seg-id residency variants from revision spec §1) can co-exist without re-running V-Bii-SMEM.

## TIER_B_MAX_BAKED_SEQ_LEN resolution

**`TIER_B_MAX_BAKED_SEQ_LEN = 16384`** for P-2's shared-crate declaration.

This satisfies:

- `TIER_B_MAX_BAKED_SEQ_LEN ∈ {4096, 8192, 16384}` (planner spec §5.3 build-time assertion).
- `compute_range_table_bytes(16384, 32, 32) = 4096 B ≤ TIER_B_RANGE_TABLE_BUDGET_BYTES = 8192 B` (existing emission budget in `pca_tilerange.rs:118` — the chosen MAX stays within the v1 codegen budget that gates `should_emit_tier_b`).
- The downstream P-2 const assertion `TIER_B_SEQ_LEN_FLOOR ≤ TIER_B_MAX_BAKED_SEQ_LEN` (planner spec §8 risk #9) will be checkable once V-D-2 produces a floor value; with the existing dispatch-spec sketch floor of `seg_min_seq_len ≈ 1024` from the unactivated dispatch sketch, `1024 ≤ 16384` holds with substantial margin.

## Re-run triggers (append-only log per IR-012)

- (none yet — initial measurement)

## Cross-references

- Planner spec §3.4 decision matrix → row 1 fired ("MAX=16384/block=32 fits across all configs → B-ii unrestricted").
- Planner spec §3.5 SMEM-headroom ladder → MAX=16384 / <60% utilization bucket → v2 trigger "extend baked max to MAX=32768 (follow-up probe)".
- Planner spec §3.4.1 investigation procedure → not triggered (no non-monotonic results; no probe-implementation suspicion).
- V-planner-options findings: this probe was the gating verification per planner spec §3.8.
- Revision-spec §1 / `tier_b_smem_probe.rs`: structural precedent for the forced-access pattern, kernel emission discipline, and outcome classification.
- Range-table sizing: `crates/nsl-codegen/src/pca_tilerange.rs::compute_range_table_bytes`.

## Notes / caveats

- All sm_80 launches succeeded on Blackwell hardware via driver JIT (approach (a) per the plan; approach (b) ptxas-only validation was unnecessary). The sm_80 cap of 100 KB is the recorded value in the decision matrix even though the resident SMEM hardware is the sm_120 99 KB region — this is consistent with §3.4 which binds the matrix to the architecture's nominal cap.
- ptxas accepted the sm_120 PTX with target version 8.7 (`.version 8.7` covers sm_120 per the original probe's rationale; backward-compatible with sm_80).
- One latent bug surfaced and was fixed during P-0.3 author cycle: the initial PTX raw-string body contained a Unicode em-dash and right-arrow inside comments, which ptxas rejects with "Unexpected non-ASCII character" (per `feedback_ptx_comment_ascii_only` invariant). Comments are now ASCII-only inside the kernel string literal.
