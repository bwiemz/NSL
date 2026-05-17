# PCA Tier B Planner — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Activate PCA Tier B in production via Option B (runtime dispatch in the kernel-launch wrapper), replacing the dispatch spec's stalled codegen-time activation with a measurement-gated runtime gate.

**Architecture:** Five sequential milestones with P-0 / P-1 parallel-independent. P-0 (V-Bii-SMEM probe) and P-1 (D-2 floor derivation) produce two empirical constants. P-2 (shared crate) declares them with const assertions. P-3 (~170 LOC implementation) extends `nsl_flash_attention_csha` and 5 sibling FFI entry points with a 2-param Tier-B-on PTX pointer pair; codegen-side emits a Tier-B-on PTX blob alongside the existing base PTX for `segment_masked` configs; runtime dispatcher picks variant based on a four-condition gate. P-4 re-baselines affected PTX snapshots and opens the activation PR.

**Tech Stack:** Rust 1.95.0, Cranelift + PTX synthesis, cudarc 0.19 dynamic-linking, `cargo test --tests`, `cargo run -p nsl-codegen --bin nsl-codegen-bench`, insta snapshot framework.

**Design spec:** `docs/superpowers/specs/2026-05-15-pca-tier-b-planner-design.md` — read §3 (V-Bii-SMEM probe), §4 (FFI extension), §5 (codegen emission), §6 (runtime dispatcher), §7 (floor rehabilitation), §8 (test surface), §10 (milestones), §11 (risks — especially #9 joint-outcome) before starting.

---

## Branch and Worktree

The plan is executed on `worktree-feat-pca-tier-b-dispatch` (current branch, 8 commits ahead of `origin/main`). The 8 prior commits are: dispatch spec + 3 refinements + dispatch implementation plan + D-1 findings + §14 amendment + V-planner-options findings + planner spec + outcome matrix reconciliation. No branch switch needed.

---

## File Structure

**Read-only references (no edits, just consumed):**

- `docs/superpowers/specs/2026-05-15-pca-tier-b-planner-design.md` — planner spec.
- `docs/superpowers/specs/2026-05-14-pca-tier-b-dispatch-design.md` — dispatch spec with §14 amendment.
- `docs/superpowers/specs/2026-05-14-tier-b-dispatch-integration-findings.md` — V-dispatch-integration findings.
- `docs/superpowers/specs/2026-05-15-tier-b-planner-options-findings.md` — V-planner-options findings.
- `docs/superpowers/specs/2026-05-13-tier-b-m2-m6-findings.md` — KEEP UNCONDITIONALLY measurement (cited for §7.3 wrong-ON-benign).
- `crates/nsl-codegen/src/pca_tilerange.rs:39` — `compute_range_table_bytes` (consumed by V-Bii-SMEM probe; unchanged).
- `crates/nsl-codegen/src/flash_attention.rs:142` — `FlashAttentionConfig` (unchanged; V-planner-options confirmed no-seq_len invariant).
- `crates/nsl-codegen/tests/tier_b_smem_probe.rs` — original SMEM probe (V-Bii-SMEM mirrors this discipline).

**Created in P-0:**

- `crates/nsl-codegen/tests/tier_b_bii_smem_probe.rs` — V-Bii-SMEM probe kernel + 12-config sweep + sentinel-readback verification.
- `docs/superpowers/specs/2026-05-15-tier-b-bii-smem-probe-findings.md` — five-outcome-matrix outcome + SMEM-headroom percentages + Tier-B-off baselines.

**Created in P-1:**

- `docs/superpowers/specs/2026-05-15-tier-b-floor-derivation-findings.md` — D-2 floor + per-seq_len win % CSV + curve shape.
- `docs/superpowers/specs/2026-05-15-tier-b-floor-derivation.csv` — raw measurement data (7 rows).

**Created in P-2:**

- `nsl-tier-b-constants/Cargo.toml` — new crate manifest (or fallback per §6.2 (α) — runtime-as-source).
- `nsl-tier-b-constants/src/lib.rs` — `TIER_B_SEQ_LEN_FLOOR` + `TIER_B_MAX_BAKED_SEQ_LEN` constants with const assertions (including joint-outcome `FLOOR <= MAX_BAKED` per risk #9).
- `Cargo.toml` (workspace root) — register the new crate as a workspace member.

**Created in P-3:**

- `crates/nsl-codegen/src/pca_tier_b.rs` — codegen-side: `should_emit_tier_b_at_codegen`, `emit_tier_b_variants_for_config`, `flash_attention_kernel_name_v2_tier_b_on`, helper-function sentinel constructors (`tier_b_disabled_sentinel`, `tier_b_enabled`).
- `crates/nsl-runtime/src/pca_tier_b_runtime.rs` — runtime-side: `should_dispatch_tier_b_at_runtime`, `assert_tier_b_sentinels`.

**Modified in P-3:**

- `crates/nsl-codegen/src/lib.rs` — add `pub mod pca_tier_b;`
- `crates/nsl-runtime/src/lib.rs` — add `pub mod pca_tier_b_runtime;`
- `crates/nsl-runtime/src/flash_attention.rs` — extend 6 `pub extern "C" fn nsl_flash_attention*` signatures with `tier_b_ptx_ptr: i64, tier_b_name_ptr: i64`; insert `assert_tier_b_sentinels(...)` + 2-line dispatch branch at each entry point.
- `crates/nsl-codegen/src/compiler/kernel.rs` — migrate 3 production callers at lines 750, 1050, 1173 to use `emit_tier_b_variants_for_config` + helpers.

**Created in P-4:**

- `crates/nsl-codegen/tests/pca_tier_b_emission.rs` — 3 emission-helper tests + 1 compile-time const assertion test.
- `crates/nsl-runtime/tests/pca_tier_b_dispatch.rs` — 6 runtime-gate truth-table tests + 1 dispatcher-integration test.
- `crates/nsl-runtime/tests/pca_tier_b_ffi_sentinel.rs` — 2 FFI sentinel-discipline tests under `bench-internal` feature.

**Re-baselined in P-4** (per §8.3 cascade; exact set surfaces at P-4 start):

- `crates/nsl-codegen/tests/snapshots/pca_forward_kernel_snapshot__forward_kernel_segment_masked_tier_b_causal_32_32_32.snap`
- `crates/nsl-codegen/tests/snapshots/pca_forward_kernel_snapshot__forward_kernel_segment_masked_tier_b_causal_64_64_64.snap`
- `crates/nsl-codegen/tests/snapshots/pca_backward_kernel_snapshot__backward_kernel_segment_masked_tier_b_on_causal_32_32_32.snap`

**Expected-stable in P-4** (verify byte-identical, do NOT re-baseline):

- `crates/nsl-codegen/tests/snapshots/pca_tier_b_preamble_isolation__*.snap`
- `crates/nsl-codegen/tests/snapshots/pca_tier_b_predicate_isolation__*.snap`

**Modified in P-4 (registry maintenance per §12.1):**

- `docs/wiki/institutional-rules.md` — extend IR-008 and IR-011 "Cited from" lists.

---

## Task P-0: V-Bii-SMEM Probe

**Scope:** Probe SMEM feasibility envelope across `MAX ∈ {4096, 8192, 16384}` × `block ∈ {32, 64}` × `arch ∈ {sm_80, sm_120}` (12 configs). Five-outcome decision matrix resolves the sub-variant (B-i / B-ii / B-ii-restricted / investigation). Co-measure Tier-B-off baseline for regression checkpoint.

**Budget:** ~4 hours total (~3 hours probe authoring + sweep + ~1 hour findings doc).

**Files:**

- Create: `crates/nsl-codegen/tests/tier_b_bii_smem_probe.rs`
- Create: `docs/superpowers/specs/2026-05-15-tier-b-bii-smem-probe-findings.md`
- Read-only: `crates/nsl-codegen/tests/tier_b_smem_probe.rs` (original probe; V-Bii-SMEM mirrors it).
- Read-only: `crates/nsl-codegen/src/pca_tilerange.rs:39` (`compute_range_table_bytes`).

### Task P-0.1: Read original SMEM probe for pattern

- [ ] **Step 1:** Read `crates/nsl-codegen/tests/tier_b_smem_probe.rs` end-to-end to understand probe-kernel emission, sentinel-write pattern, readback verification, host-side launch sequence, and outcome-recording structure.
- [ ] **Step 2:** Record specific patterns to mirror in your notes: kernel-name format, `cuModuleLoadData` invocation pattern, sentinel-write/readback assertion, per-config success/fail tally structure.

### Task P-0.2: Author probe kernel scaffold

- [ ] **Step 1:** Create `crates/nsl-codegen/tests/tier_b_bii_smem_probe.rs` with module-level doc-comment citing the planner spec §3 + the original probe.

- [ ] **Step 2:** Add test fixtures and probe-kernel emission. The probe must emit:

```rust
use nsl_codegen::pca_tilerange::compute_range_table_bytes;
use nsl_codegen::flash_attention::FlashAttentionConfig;

const PROBE_MAX_VALUES: &[u32] = &[4096, 8192, 16384];
const PROBE_BLOCK_VALUES: &[u32] = &[32, 64];

fn emit_probe_ptx(max_seq_len: u32, block_q: u32, block_kv: u32, target_sm: &str) -> String {
    let range_table_bytes = compute_range_table_bytes(
        max_seq_len as u64,
        block_q as u64,
        block_kv as u64,
    );
    let seg_smem_bytes: usize = 8192; // Match original probe's seg_smem size.
    let total_extern_bytes = range_table_bytes as usize;

    format!(
        r#".version 7.5
.target {target_sm}
.address_size 64
.visible .entry tier_b_bii_smem_probe(.param .u64 out_ptr)
{{
    .shared .align 4 .b8 seg_smem[{seg_smem_bytes}];
    .extern .shared .align 16 .b8 shmem[];

    .reg .u64 %rd<5>;
    .reg .u32 %r<3>;
    .reg .u8  %rb<2>;

    // Force-write to both regions to confirm allocation succeeds.
    mov.u32 %r1, 0xA5;
    cvt.u16.u32 %rb1, %r1;
    st.shared.u8 [seg_smem], %rb1;
    st.shared.u8 [shmem], %rb1;
    bar.sync 0;

    // Readback sentinel to caller for verification.
    ld.shared.u8 %rb1, [seg_smem];
    cvt.u32.u8 %r2, %rb1;
    ld.param.u64 %rd1, [out_ptr];
    st.global.u32 [%rd1], %r2;
    ret;
}}
"#
    )
}
```

- [ ] **Step 3:** Verify with a placeholder test that compiles cleanly:

```rust
#[test]
fn probe_kernel_emits_valid_ptx_shape() {
    let ptx = emit_probe_ptx(4096, 64, 64, "sm_120");
    assert!(ptx.contains(".version 7.5"));
    assert!(ptx.contains(".target sm_120"));
    assert!(ptx.contains(".shared .align 4 .b8 seg_smem"));
    assert!(ptx.contains(".extern .shared .align 16 .b8 shmem"));
}
```

Run: `cargo test -p nsl-codegen --test tier_b_bii_smem_probe probe_kernel_emits_valid_ptx_shape`
Expected: PASS.

- [ ] **Step 4:** Commit:

```bash
git add crates/nsl-codegen/tests/tier_b_bii_smem_probe.rs
git commit -m "test(pca-tier-b-planner): P-0.2 — V-Bii-SMEM probe kernel scaffold"
```

### Task P-0.3: Sweep loop + per-config result recording

- [ ] **Step 1:** Add the sweep harness:

```rust
#[derive(Debug, Clone)]
struct ProbeResult {
    max_seq_len: u32,
    block: u32,
    target_sm: String,
    seg_smem_bytes: u32,           // Tier-B-off baseline.
    tier_b_extern_bytes: u32,      // Tier B's incremental contribution.
    total_smem_bytes: u32,
    cap_bytes: u32,
    utilization_pct: f32,
    outcome: ProbeOutcome,
}

#[derive(Debug, Clone, PartialEq)]
enum ProbeOutcome {
    Pass,
    LaunchFailed(String),
    PtxasRejected(String),
}

fn run_probe_config(
    max_seq_len: u32,
    block: u32,
    target_sm: &str,
) -> ProbeResult {
    let seg_smem_bytes: u32 = 8192;
    let tier_b_extern_bytes = compute_range_table_bytes(
        max_seq_len as u64, block as u64, block as u64,
    ) as u32;
    let total_smem_bytes = seg_smem_bytes + tier_b_extern_bytes;
    let cap_bytes: u32 = if target_sm == "sm_120" { 99 * 1024 } else { 100 * 1024 };
    let utilization_pct = (total_smem_bytes as f32 / cap_bytes as f32) * 100.0;

    let ptx = emit_probe_ptx(max_seq_len, block, block, target_sm);

    // For sm matching the host's architecture, launch and verify readback.
    // For non-matching sm, perform ptxas-only validation (compile but don't launch).
    let outcome = launch_or_validate_ptx(&ptx, tier_b_extern_bytes, target_sm);

    ProbeResult {
        max_seq_len, block, target_sm: target_sm.into(),
        seg_smem_bytes, tier_b_extern_bytes, total_smem_bytes,
        cap_bytes, utilization_pct, outcome,
    }
}

#[cfg(feature = "cuda")]
fn launch_or_validate_ptx(ptx: &str, dynamic_smem: u32, target_sm: &str) -> ProbeOutcome {
    use nsl_runtime::cuda::ensure_context;
    ensure_context();
    // ... cuModuleLoadData -> cuModuleGetFunction -> cuLaunchKernel with dynamic_smem
    //     readback sentinel byte; assert == 0xA5
    //     If any step fails, map to ProbeOutcome::LaunchFailed or ::PtxasRejected.
    todo!("CUDA launch + readback; see tier_b_smem_probe.rs for the existing pattern")
}

#[cfg(not(feature = "cuda"))]
fn launch_or_validate_ptx(_ptx: &str, _dynamic_smem: u32, _target_sm: &str) -> ProbeOutcome {
    ProbeOutcome::PtxasRejected("cuda feature disabled".into())
}
```

- [ ] **Step 2:** Fill in `launch_or_validate_ptx`'s body by adapting the pattern from `crates/nsl-codegen/tests/tier_b_smem_probe.rs`'s existing launch path. Use `cuLaunchKernel` with `sharedMemBytes = dynamic_smem` for the extern shmem allocation.

- [ ] **Step 3:** Verify the sweep harness compiles and runs at one fixture (smoke test):

```rust
#[test]
#[cfg(feature = "cuda")]
fn probe_smoke_max4096_block64_sm120() {
    let result = run_probe_config(4096, 64, "sm_120");
    eprintln!("smoke: {result:#?}");
    // For MAX=4096/block=64/sm_120: range_table_bytes = 2 * (64+64) * 2 = 512.
    // total = seg_smem(8192) + 512 = 8704. Cap = 101376. Util ≈ 8.6%. PASS expected.
    assert_eq!(result.outcome, ProbeOutcome::Pass);
    assert!(result.utilization_pct < 10.0);
}
```

Run: `cargo test -p nsl-codegen --features cuda --test tier_b_bii_smem_probe probe_smoke_max4096_block64_sm120`
Expected: PASS (the smoke config is far below cap).

- [ ] **Step 4:** Commit:

```bash
git add crates/nsl-codegen/tests/tier_b_bii_smem_probe.rs
git commit -m "test(pca-tier-b-planner): P-0.3 — V-Bii-SMEM sweep harness + smoke test"
```

### Task P-0.4: Run full 12-config sweep

- [ ] **Step 1:** Add the sweep test that runs all 12 configurations:

```rust
#[test]
#[cfg(feature = "cuda")]
fn probe_full_sweep() {
    let targets = &["sm_80", "sm_120"];
    let mut results = Vec::new();

    for &target_sm in targets {
        for &max_seq_len in PROBE_MAX_VALUES {
            for &block in PROBE_BLOCK_VALUES {
                let result = run_probe_config(max_seq_len, block, target_sm);
                eprintln!(
                    "MAX={max_seq_len} block={block} sm={target_sm} \
                     util={:.1}% outcome={:?}",
                    result.utilization_pct, result.outcome
                );
                results.push(result);
            }
        }
    }

    // Emit summary CSV-ish lines for the findings doc.
    eprintln!("\n=== SWEEP SUMMARY (paste into findings doc) ===");
    eprintln!("max_seq_len,block,target_sm,seg_smem,tier_b_extern,total,cap,util_pct,outcome");
    for r in &results {
        eprintln!(
            "{},{},{},{},{},{},{},{:.1},{:?}",
            r.max_seq_len, r.block, r.target_sm,
            r.seg_smem_bytes, r.tier_b_extern_bytes, r.total_smem_bytes,
            r.cap_bytes, r.utilization_pct, r.outcome,
        );
    }
}
```

- [ ] **Step 2:** Run the sweep:

```bash
cargo test -p nsl-codegen --features cuda --release --test tier_b_bii_smem_probe probe_full_sweep -- --nocapture
```

Expected: 12 result lines + summary CSV-ish output. Some configs may fail (`LaunchFailed` or `PtxasRejected`) at high MAX × small block; those failures are the data the probe is measuring.

- [ ] **Step 3:** Save the full output to a transient file for the findings doc:

```bash
cargo test -p nsl-codegen --features cuda --release --test tier_b_bii_smem_probe probe_full_sweep -- --nocapture 2>&1 | tee /tmp/tier_b_bii_smem_sweep.txt
```

### Task P-0.5: Classify outcome per §3.4 decision matrix

- [ ] **Step 1:** Inspect the sweep output. Identify the highest `MAX_BAKED` where ALL probe configs for that MAX pass at BOTH sm_80 AND sm_120:

  - If MAX=16384 passes everywhere → outcome row: **B-ii unrestricted**.
  - If MAX=8192 passes everywhere but MAX=16384 has at least one failure → **B-ii-restricted (MAX=8192)**.
  - If MAX=4096 passes everywhere but MAX=8192 has at least one failure → **B-i (single-emission at MAX=4096)**.
  - If MAX=4096 has at least one failure → **Investigation; B-iii becomes default** (per §3.4 reconciled matrix; P-2 blocks).
  - If results are non-monotonic with respect to MAX → **Investigation** (per §3.4.1 procedure).

- [ ] **Step 2:** Record SMEM-headroom % at the chosen MAX (and at MAX=8192 if B-ii-restricted fires). These drive v2 trigger ladders per §3.5.

### Task P-0.6: Write V-Bii-SMEM findings doc

- [ ] **Step 1:** Create `docs/superpowers/specs/2026-05-15-tier-b-bii-smem-probe-findings.md` with this template populated:

```markdown
# Tier B Planner — V-Bii-SMEM Probe Findings

**Date:** 2026-05-15
**Branch:** `worktree-feat-pca-tier-b-dispatch`
**Spec:** [`2026-05-15-pca-tier-b-planner-design.md`](2026-05-15-pca-tier-b-planner-design.md) §3
**Plan:** [`2026-05-15-pca-tier-b-planner-implementation.md`](../plans/2026-05-15-pca-tier-b-planner-implementation.md) Task P-0
**Hardware:** [RTX 5070 Ti / CUDA 13.2 / driver 591.86] — fill in actual.

## Probe protocol

Mirrors the original SMEM probe (revision spec §1). Forced-access kernel allocates `seg_smem[8192]` (Tier-B-off baseline) + `.extern .shared shmem[]` sized to `compute_range_table_bytes(MAX, block, block)`. Per-thread writes sentinel `0xA5`, bar.sync, readback to global verifies allocation.

## Sweep results (12 configs)

| MAX | block | target_sm | seg_smem (B) | tier_b_extern (B) | total (B) | cap (B) | util % | outcome |
|-----|-------|-----------|--------------|-------------------|-----------|---------|--------|---------|
| [paste full 12 rows from /tmp/tier_b_bii_smem_sweep.txt summary]

## Outcome (per §3.4 decision matrix)

**Sub-variant resolved: <B-ii unrestricted | B-ii-restricted (MAX=8192) | B-i (MAX=4096) | investigation>.**

Justification: [which MAX value's row passed all configs; cite the specific entry].

## SMEM headroom (for §3.5 v2-trigger ladders)

- **At MAX=16384/block=32:** [util %; classify as <60% / 60–85% / >85%] — drives v2 trigger #1/#2 ladder.
- **At MAX=8192/block=32** (if applicable for B-ii-restricted): [util %; classify] — drives v2 trigger #3/#4 ladder.

## TIER_B_MAX_BAKED_SEQ_LEN resolution

**`TIER_B_MAX_BAKED_SEQ_LEN = <4096 | 8192 | 16384>`** for P-2's shared-crate declaration.

## Re-run triggers (append-only log per IR-012)

- (none yet)

## Cross-references

- Spec §3.4 decision matrix; §3.5 SMEM-headroom ladder; §3.4.1 investigation procedure if non-monotonic.
- V-planner-options findings: this probe was the gating verification per §3.8.
```

- [ ] **Step 2:** Paste the 12-row sweep table from `/tmp/tier_b_bii_smem_sweep.txt`. Verify each row's outcome column matches the probe's empirical result.

- [ ] **Step 3:** Commit:

```bash
git add -f docs/superpowers/specs/2026-05-15-tier-b-bii-smem-probe-findings.md
git commit -m "docs(pca-tier-b-planner): P-0.6 — V-Bii-SMEM findings (sub-variant: <N>)"
```

**Gate to P-2:** findings doc committed with non-investigation outcome row; `TIER_B_MAX_BAKED_SEQ_LEN` value pinned.

---

## Task P-1: D-2 Floor Derivation

**Scope:** Inherit dispatch spec's D-2 protocol (7-point seq_len sweep at sparsity=50%). The protocol is unchanged; only the consumption layer moves to runtime per planner spec §7.2.

**Budget:** ~1 hour (7 measurements + findings doc).

**Files:**

- Create: `docs/superpowers/specs/2026-05-15-tier-b-floor-derivation-findings.md`
- Create: `docs/superpowers/specs/2026-05-15-tier-b-floor-derivation.csv`
- Read-only: `crates/nsl-codegen/src/bin/bench/launch.rs` — existing bench harness from B.1.5-5.

**Sequencing:** P-1 is parallel-independent with P-0; can run concurrently. Synchronization at P-2 entry.

### Task P-1.1: Reproduce M2/M6 gate-fixture measurement (sanity)

- [ ] **Step 1:** Run the gate-fixture measurement at seq_len=4096, sparsity=50% with Tier-B-on:

```bash
cargo run -p nsl-codegen --release --bin nsl-codegen-bench -- \
  --fixture m2 --seq-len 4096 --sparsity 50 --iterations 100 --tier-b on \
  --seed 0xDEADBEEF 2>&1 | tee /tmp/tier_b_floor_smoke_on.txt
```

- [ ] **Step 2:** Same with Tier-B-off:

```bash
cargo run -p nsl-codegen --release --bin nsl-codegen-bench -- \
  --fixture m2 --seq-len 4096 --sparsity 50 --iterations 100 --tier-b off \
  --seed 0xDEADBEEF 2>&1 | tee /tmp/tier_b_floor_smoke_off.txt
```

- [ ] **Step 3:** Compute `wall_time_win_pct = (off - on) / off * 100`. Verify within ±10% of the 73.33% recorded in `docs/superpowers/specs/2026-05-13-tier-b-m2-m6-findings.md`. If drift >10%, STOP — re-investigate measurement protocol before proceeding (per dispatch spec §6).

### Task P-1.2: Run 7-point sweep

- [ ] **Step 1:** For each `seq_len ∈ {128, 256, 512, 1024, 2048, 4096, 8192}`, run both `--tier-b on` and `--tier-b off`:

```bash
for s in 128 256 512 1024 2048 4096 8192; do
  cargo run -p nsl-codegen --release --bin nsl-codegen-bench -- \
    --fixture m2 --seq-len $s --sparsity 50 --iterations 100 --tier-b on \
    --seed 0xDEADBEEF 2>&1 | tee -a /tmp/tier_b_floor_on.txt
  cargo run -p nsl-codegen --release --bin nsl-codegen-bench -- \
    --fixture m2 --seq-len $s --sparsity 50 --iterations 100 --tier-b off \
    --seed 0xDEADBEEF 2>&1 | tee -a /tmp/tier_b_floor_off.txt
done
```

- [ ] **Step 2:** Extract `wall_time_ns_median` per (seq_len, mode) pair from `tier_b_bench_result:` lines (per the bench binary's contract, IR-012). Compute win % per seq_len.

### Task P-1.3: Write CSV

- [ ] **Step 1:** Create `docs/superpowers/specs/2026-05-15-tier-b-floor-derivation.csv`:

```csv
seq_len,wall_time_on_ns,wall_time_off_ns,win_pct
128,<v>,<v>,<v>
256,<v>,<v>,<v>
512,<v>,<v>,<v>
1024,<v>,<v>,<v>
2048,<v>,<v>,<v>
4096,<v>,<v>,<v>
8192,<v>,<v>,<v>
```

Fill in actual values from the sweep.

### Task P-1.4: Derive floor + classify outcome

- [ ] **Step 1:** Apply dispatch spec §6.4 rule: floor = smallest seq_len with `win_pct ≥ 10%`.

  - Monotonic with positive win at all 7 points → **case clear pass**; floor = 128 (or smallest passing seq_len).
  - Threshold in the middle → **case clear pass**; floor = smallest seq_len at/above the threshold.
  - All seq_lens have positive win but < 10% (5-8%) → **case (a) sub-threshold** per planner spec §7.6. Stop and investigate; STOP before P-2.
  - Zero or negative wins → **case (b)** per planner spec §7.6. Stop and re-evaluate Option B viability; STOP before P-2.
  - Non-monotonic → **case (a) investigation**; STOP.

### Task P-1.5: Write D-2 findings doc

- [ ] **Step 1:** Create `docs/superpowers/specs/2026-05-15-tier-b-floor-derivation-findings.md`:

```markdown
# Tier B Planner — D-2 Floor Derivation Findings

**Date:** 2026-05-15
**Branch:** `worktree-feat-pca-tier-b-dispatch`
**Spec inheritance:** Dispatch spec §6 (protocol unchanged); planner spec §7 (consumption layer moved to runtime).
**Plan:** Task P-1.

## Reproducer (P-1.1)

Gate-fixture (seq_len=4096, sparsity=50%): win = <X%>. Matches `2026-05-13-tier-b-m2-m6-findings.md`'s 73.33% within ±10%? <YES/NO>.

## Sweep results

See `2026-05-15-tier-b-floor-derivation.csv`.

| seq_len | win % | classification |
|---------|-------|----------------|
| 128  | <v>% | <below/above floor> |
| 256  | <v>% | |
| 512  | <v>% | |
| 1024 | <v>% | |
| 2048 | <v>% | |
| 4096 | <v>% | |
| 8192 | <v>% | |

## Curve shape

<monotonic | threshold | non-monotonic>

## TIER_B_SEQ_LEN_FLOOR resolution

**`TIER_B_SEQ_LEN_FLOOR = <value>`** for P-2's shared-crate declaration.

Outcome per planner spec §7.6: <clear pass | case (a) sub-threshold | case (b) Option B re-evaluation>.

## Cross-references

- Planner spec §7.2 — runtime consumption layer.
- Planner spec §7.6 — case-(a)/case-(b) split.
- V-Bii-SMEM findings — joint-outcome compatibility per risk #9 (FLOOR ≤ MAX_BAKED).
```

- [ ] **Step 2:** Commit both findings doc + CSV:

```bash
git add -f docs/superpowers/specs/2026-05-15-tier-b-floor-derivation-findings.md \
            docs/superpowers/specs/2026-05-15-tier-b-floor-derivation.csv
git commit -m "docs(pca-tier-b-planner): P-1 — D-2 floor derivation (FLOOR=<N>)"
```

**Gate to P-2:** findings doc committed with non-case-(b) outcome; `TIER_B_SEQ_LEN_FLOOR` value pinned.

---

## Task P-2: Shared-Crate Creation

**Scope:** Create `nsl-tier-b-constants` crate per planner spec §6.2 (β) commitment, OR fall back to (α) runtime-as-source if dependency-graph inspection reveals existing nsl-codegen ↔ nsl-runtime coupling. Declare `TIER_B_SEQ_LEN_FLOOR` + `TIER_B_MAX_BAKED_SEQ_LEN` with const assertions including joint-outcome validity (risk #9 mitigation).

**Budget:** ~1 hour (5-min dep-graph inspection + ~30 min crate creation + ~25 min const-assertion authoring + verification).

**Files:**

- Create: `nsl-tier-b-constants/Cargo.toml`
- Create: `nsl-tier-b-constants/src/lib.rs`
- Modify: `Cargo.toml` (workspace root)
- Modify: `crates/nsl-codegen/Cargo.toml`
- Modify: `crates/nsl-runtime/Cargo.toml`

### Task P-2.1: Dependency-graph inspection

- [ ] **Step 1:** Verify the current dependency direction:

```bash
cargo tree -p nsl-codegen 2>&1 | grep -E "^[[:space:]]+nsl-runtime\b"
cargo tree -p nsl-runtime 2>&1 | grep -E "^[[:space:]]+nsl-codegen\b"
```

Expected: either both empty (no existing coupling — proceed with (β) shared crate), OR one direction has the other as dependency (proceed with (α) runtime-as-source).

- [ ] **Step 2:** Record the inspection result. If `nsl-codegen` already depends on `nsl-runtime` (or vice versa), document in the activation PR description and proceed with (α). Otherwise (β) is the choice.

### Task P-2.2 (β path): Create shared crate

(Skip to P-2.3 if (α) fallback fired.)

- [ ] **Step 1:** Create `nsl-tier-b-constants/Cargo.toml`:

```toml
[package]
name = "nsl-tier-b-constants"
version = "0.1.0"
edition = "2021"
publish = false
description = "Shared compile-time constants for PCA Tier B dispatch (codegen + runtime)."

[lib]
path = "src/lib.rs"
```

- [ ] **Step 2:** Create `nsl-tier-b-constants/src/lib.rs`:

```rust
//! PCA Tier B dispatch constants — shared between nsl-codegen and nsl-runtime.
//!
//! These two values together define the runtime dispatch gate per planner spec §6.1:
//!   - `TIER_B_SEQ_LEN_FLOOR`: minimum seq_len where Tier B is empirically profitable.
//!   - `TIER_B_MAX_BAKED_SEQ_LEN`: maximum seq_len the Tier-B-on PTX's SMEM allocation handles.
//!
//! The runtime gate fires Tier-B-on iff `seq_len ∈ [FLOOR, MAX_BAKED]` (plus the other gate
//! conditions per §6.1). Both consumed at compile time on both sides; single source of truth.

/// Empirical seq_len floor (wall-time win ≥ 10% per dispatch spec §6).
/// Derived from `docs/superpowers/specs/2026-05-15-tier-b-floor-derivation-findings.md`.
pub const TIER_B_SEQ_LEN_FLOOR: u32 = /* value from P-1.5 findings doc */ 1024;

/// Conservative-max seq_len baked into Tier-B-on PTX SMEM allocation.
/// Derived from `docs/superpowers/specs/2026-05-15-tier-b-bii-smem-probe-findings.md`.
pub const TIER_B_MAX_BAKED_SEQ_LEN: u32 = /* value from P-0.6 findings doc */ 16384;

// Compile-time assertion: MAX_BAKED is a probe-validated value (planner spec §5.3).
const _: () = assert!(
    TIER_B_MAX_BAKED_SEQ_LEN == 4096
        || TIER_B_MAX_BAKED_SEQ_LEN == 8192
        || TIER_B_MAX_BAKED_SEQ_LEN == 16384,
    "TIER_B_MAX_BAKED_SEQ_LEN must be one of {4096, 8192, 16384} per V-Bii-SMEM probe's \
     five-outcome matrix; investigation-row outcomes require resolving the probe anomaly \
     before this constant is set."
);

// Compile-time assertion: joint-outcome validity (planner spec §11 risk #9).
// FLOOR > MAX_BAKED would produce a runtime dispatch gate that's never satisfied
// (seq_len >= FLOOR && seq_len <= MAX_BAKED is empty when FLOOR > MAX_BAKED).
const _: () = assert!(
    TIER_B_SEQ_LEN_FLOOR <= TIER_B_MAX_BAKED_SEQ_LEN,
    "TIER_B_SEQ_LEN_FLOOR must be <= TIER_B_MAX_BAKED_SEQ_LEN; otherwise the runtime \
     dispatch gate is never satisfied (Tier B ships but never activates). \
     V-Bii-SMEM and D-2 produced incompatible values; investigate both findings docs."
);
```

Replace the placeholder values (`1024`, `16384`) with the actual values from P-0.6 and P-1.5 findings docs.

- [ ] **Step 3:** Register the crate in the workspace. Modify root `Cargo.toml`:

```toml
[workspace]
members = [
    "crates/nsl-codegen",
    "crates/nsl-runtime",
    # ... existing members ...
    "nsl-tier-b-constants",  # NEW
]
```

- [ ] **Step 4:** Add the new crate as a dependency in both consumers. Modify `crates/nsl-codegen/Cargo.toml`:

```toml
[dependencies]
nsl-tier-b-constants = { path = "../../nsl-tier-b-constants" }
# ... existing dependencies ...
```

Modify `crates/nsl-runtime/Cargo.toml` with the same dependency line.

- [ ] **Step 5:** Verify the workspace builds:

```bash
cargo build --workspace
```

Expected: clean build. The const assertions fire at compile time iff `FLOOR` or `MAX_BAKED` are out of range — those values come from the findings docs so should be valid.

- [ ] **Step 6:** Commit:

```bash
git add -f nsl-tier-b-constants/ Cargo.toml crates/nsl-codegen/Cargo.toml crates/nsl-runtime/Cargo.toml
git commit -m "feat(pca-tier-b-planner): P-2 — shared-crate nsl-tier-b-constants (β commitment)"
```

### Task P-2.3 (α path): Runtime-as-source fallback

(Only if dep-graph inspection in P-2.1 revealed nsl-codegen → nsl-runtime exists.)

- [ ] **Step 1:** Declare constants in `crates/nsl-runtime/src/pca_tier_b_runtime.rs` (created in P-3.3; for now, just hold this plan until P-3.3 lands).

- [ ] **Step 2:** Add to `crates/nsl-codegen/src/pca_tier_b.rs` (created in P-3.1):

```rust
pub use nsl_runtime::pca_tier_b_runtime::{TIER_B_SEQ_LEN_FLOOR, TIER_B_MAX_BAKED_SEQ_LEN};
```

- [ ] **Step 3:** No separate crate; no workspace change; both consumers get the constants via `nsl_runtime`'s re-export. Skip P-2.2 commit.

**Gate to P-3:** workspace builds clean; both constants resolvable from both crates.

---

## Task P-3: Implementation

**Scope:** ~170 LOC implementation per planner spec §4-§6 LOC budget. Two new files (`pca_tier_b.rs` codegen-side + `pca_tier_b_runtime.rs` runtime-side); 6 FFI entry-point extensions; 3 production call-site migrations.

**Budget:** ~1-2 days.

### Task P-3.1: Codegen-side `pca_tier_b.rs`

- [ ] **Step 1:** Create `crates/nsl-codegen/src/pca_tier_b.rs`:

```rust
//! PCA Tier B — codegen-side dispatch helpers.
//!
//! Per planner spec §5. Decides whether to emit a Tier-B-on PTX variant
//! alongside the base PTX for each FA-2 kernel emission. The case-(β-ii)
//! collapse from the dispatch spec's §14 amendment: at codegen, the heuristic
//! reduces to `config.segment_masked`; the seq_len floor gate fires at runtime.

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::{
    flash_attention_kernel_name_v2,
    synthesize_flash_attention_ptx_v2_with_tier_b,
};
use crate::pca_segment::SegmentResidency;

pub use nsl_tier_b_constants::{TIER_B_MAX_BAKED_SEQ_LEN, TIER_B_SEQ_LEN_FLOOR};

/// Codegen-time gate: should this config's emission include a Tier-B-on PTX variant?
///
/// Returns `true` iff `config.segment_masked`. The seq_len floor gate is applied at
/// runtime by the launch wrapper (per planner spec §6), not here.
pub fn should_emit_tier_b_at_codegen(config: &FlashAttentionConfig) -> bool {
    config.segment_masked
}

/// Result of codegen-side Tier B variant emission for a single FA-2 config.
pub struct TierBEmissionResult {
    pub base_ptx: Vec<u8>,
    pub base_kernel_name: String,
    /// `Some` iff `should_emit_tier_b_at_codegen(config)`.
    pub tier_b_on_ptx: Option<Vec<u8>>,
    pub tier_b_on_kernel_name: Option<String>,
}

/// Emit base + (optional) Tier-B-on PTX variants for a config.
///
/// Single edit point for the codegen emission policy (planner spec §5.4).
pub fn emit_tier_b_variants_for_config(config: &FlashAttentionConfig) -> TierBEmissionResult {
    let base_ptx = synthesize_flash_attention_ptx_v2_with_tier_b(config, None);
    let base_kernel_name = flash_attention_kernel_name_v2(config);

    let (tier_b_on_ptx, tier_b_on_kernel_name) = if should_emit_tier_b_at_codegen(config) {
        let tier_b_args = Some((TIER_B_MAX_BAKED_SEQ_LEN, SegmentResidency::Tiled));
        let on_ptx = synthesize_flash_attention_ptx_v2_with_tier_b(config, tier_b_args);
        let on_name = flash_attention_kernel_name_v2_tier_b_on(config);
        (Some(on_ptx), Some(on_name))
    } else {
        (None, None)
    };

    TierBEmissionResult { base_ptx, base_kernel_name, tier_b_on_ptx, tier_b_on_kernel_name }
}

/// Kernel-name for the Tier-B-on variant. Encodes `MAX_BAKED` per planner spec §5.5
/// to eliminate cross-PR / cross-architecture collision class.
pub fn flash_attention_kernel_name_v2_tier_b_on(config: &FlashAttentionConfig) -> String {
    format!(
        "{}_tier_b_max{}",
        flash_attention_kernel_name_v2(config),
        TIER_B_MAX_BAKED_SEQ_LEN,
    )
}
```

- [ ] **Step 2:** Register the module in `crates/nsl-codegen/src/lib.rs`:

Add `pub mod pca_tier_b;` near the other `pub mod` declarations (alongside `pub mod pca_tilerange;`).

- [ ] **Step 3:** Verify the codegen crate builds:

```bash
cargo build -p nsl-codegen
```

Expected: clean build.

- [ ] **Step 4:** Commit:

```bash
git add crates/nsl-codegen/src/pca_tier_b.rs crates/nsl-codegen/src/lib.rs
git commit -m "feat(pca-tier-b-planner): P-3.1 — codegen-side pca_tier_b module"
```

### Task P-3.2: Cranelift-side sentinel helpers

The sentinel-construction helpers live at the Cranelift-emitter boundary — the same point where `kernel.rs` constructs FFI call argument lists. They emit either two zero constants (disabled) or two data-section address references (enabled).

- [ ] **Step 1:** Locate the Cranelift-emitter boundary. The 3 production callers in `crates/nsl-codegen/src/compiler/kernel.rs:750, 1050, 1173` each build a Vec of Cranelift `Value`s for the FFI call. Search for the existing pattern:

```bash
grep -n "builder.ins().iconst" crates/nsl-codegen/src/compiler/kernel.rs | head -5
```

Note the conventions for emitting i64 constants and data-section addresses.

- [ ] **Step 2:** Add the helpers in `crates/nsl-codegen/src/pca_tier_b.rs` (extending P-3.1's file). Append:

```rust
use cranelift::prelude::*;
use cranelift_module::{DataId, Module};

/// Emit sentinel pair `(0, 0)` indicating "no Tier B variant available."
/// Cranelift Value[2]: two i64 zero constants.
pub fn tier_b_disabled_sentinel(builder: &mut FunctionBuilder) -> [Value; 2] {
    [builder.ins().iconst(types::I64, 0), builder.ins().iconst(types::I64, 0)]
}

/// Emit sentinel pair `(ptx_addr, name_addr)` indicating Tier-B-on variant available.
/// Cranelift Value[2]: two i64 data-section addresses.
pub fn tier_b_enabled<M: Module>(
    builder: &mut FunctionBuilder,
    module: &mut M,
    tier_b_ptx_data: DataId,
    tier_b_name_data: DataId,
) -> [Value; 2] {
    let func = builder.func.dfg.signatures.values().next().unwrap();
    let _ = func; // suppress unused warning in scaffold; real implementation matches existing pattern.
    let ptx_addr = {
        let gv = module.declare_data_in_func(tier_b_ptx_data, builder.func);
        builder.ins().symbol_value(types::I64, gv)
    };
    let name_addr = {
        let gv = module.declare_data_in_func(tier_b_name_data, builder.func);
        builder.ins().symbol_value(types::I64, gv)
    };
    [ptx_addr, name_addr]
}
```

(Adjust the exact `Module` trait bound and address-emission pattern to match the existing `kernel.rs` Cranelift-emitter calls. The pattern above is canonical; the specifics may need minor adjustment to the existing emitter's idioms.)

- [ ] **Step 3:** Verify the codegen crate still builds:

```bash
cargo build -p nsl-codegen
```

Expected: clean build.

- [ ] **Step 4:** Commit:

```bash
git add crates/nsl-codegen/src/pca_tier_b.rs
git commit -m "feat(pca-tier-b-planner): P-3.2 — sentinel-construction helpers (IR-001 discipline)"
```

### Task P-3.3: Runtime-side `pca_tier_b_runtime.rs`

- [ ] **Step 1:** Create `crates/nsl-runtime/src/pca_tier_b_runtime.rs`:

```rust
//! PCA Tier B — runtime-side dispatch gate.
//!
//! Per planner spec §6. The runtime gate's four-condition logic determines whether
//! the kernel launch dispatches to the Tier-B-on PTX variant (when codegen emitted
//! one for this config) or to the base Tier-B-off PTX (always present).

pub use nsl_tier_b_constants::{TIER_B_MAX_BAKED_SEQ_LEN, TIER_B_SEQ_LEN_FLOOR};

/// Runtime gate: should this kernel launch dispatch to the Tier-B-on variant?
///
/// Returns `true` iff ALL FOUR conditions hold:
/// 1. Codegen emitted a Tier-B-on variant for this config (`tier_b_ptx_ptr != 0`).
/// 2. The caller passed a non-null segment_ids pointer (`segment_ids_ptr != 0`).
/// 3. seq_len is at or above the empirical profitability floor (`seq_len >= TIER_B_SEQ_LEN_FLOOR`).
/// 4. seq_len fits the conservative-max baked into the Tier-B-on PTX
///    (`seq_len <= TIER_B_MAX_BAKED_SEQ_LEN`).
pub fn should_dispatch_tier_b_at_runtime(
    tier_b_ptx_ptr: i64,
    segment_ids_ptr: i64,
    seq_len: u32,
) -> bool {
    tier_b_ptx_ptr != 0
        && segment_ids_ptr != 0
        && seq_len >= TIER_B_SEQ_LEN_FLOOR
        && seq_len <= TIER_B_MAX_BAKED_SEQ_LEN
}

/// Asserts the Tier B sentinel pair has both values either zero or both non-zero.
/// Panics (via process abort) if mismatched — catches helper-bypass at call sites
/// per planner spec §4.3.
#[inline(always)]
pub fn assert_tier_b_sentinels(
    entry_point: &'static str,
    tier_b_ptx_ptr: i64,
    tier_b_name_ptr: i64,
) {
    if (tier_b_ptx_ptr == 0) != (tier_b_name_ptr == 0) {
        eprintln!(
            "FATAL [{entry_point}]: tier_b_ptx_ptr={tier_b_ptx_ptr:#x} but \
             tier_b_name_ptr={tier_b_name_ptr:#x}; sentinel pair must agree \
             (both zero = disabled, both non-zero = enabled). Call site emitted \
             via inline literals instead of tier_b_disabled_sentinel() / \
             tier_b_enabled()?"
        );
        std::process::abort();
    }
}
```

- [ ] **Step 2:** Register the module in `crates/nsl-runtime/src/lib.rs`:

Add `pub mod pca_tier_b_runtime;` near the other `pub mod` declarations.

- [ ] **Step 3:** Verify the runtime crate builds:

```bash
cargo build -p nsl-runtime
```

Expected: clean build.

- [ ] **Step 4:** Commit:

```bash
git add crates/nsl-runtime/src/pca_tier_b_runtime.rs crates/nsl-runtime/src/lib.rs
git commit -m "feat(pca-tier-b-planner): P-3.3 — runtime-side pca_tier_b_runtime module"
```

### Task P-3.4: FFI extension at 6 entry points

The extension appends `tier_b_ptx_ptr: i64, tier_b_name_ptr: i64` to each of 6 `pub extern "C" fn nsl_flash_attention*` signatures, inserts `assert_tier_b_sentinels(...)` at the entry, and inserts the 2-line dispatch branch picking `(effective_ptx_ptr, effective_name_ptr)`.

- [ ] **Step 1:** Read each entry point's current shape to understand existing parameter layout. Entry points (line numbers from `crates/nsl-runtime/src/flash_attention.rs`):

  - `nsl_flash_attention:109`
  - `nsl_flash_attention_csha:373`
  - `nsl_flash_attention_csha_with_saves:597`
  - `nsl_flash_attention_csha_backward:883`
  - `nsl_flash_attention_quantized:1459`
  - `nsl_flash_attention_backward:2107`

- [ ] **Step 2:** Extend `nsl_flash_attention_csha:373` first (the largest signature). Find the end of the parameter list and append:

```rust
pub extern "C" fn nsl_flash_attention_csha(
    /* ... existing 36 params ... */
    segment_ids_ptr: i64,
    // Tier B extension (planner spec §4):
    tier_b_ptx_ptr: i64,
    tier_b_name_ptr: i64,
) -> i64 {
    use crate::pca_tier_b_runtime::{
        assert_tier_b_sentinels, should_dispatch_tier_b_at_runtime,
    };

    // Tier B extension entry: assert sentinel agreement (planner spec §4.3).
    assert_tier_b_sentinels(
        "nsl_flash_attention_csha",
        tier_b_ptx_ptr,
        tier_b_name_ptr,
    );

    // Tier B extension: pick PTX/name based on runtime gate.
    let (effective_ptx_ptr, effective_name_ptr) =
        if should_dispatch_tier_b_at_runtime(
            tier_b_ptx_ptr,
            segment_ids_ptr,
            seq_len as u32,
        ) {
            (tier_b_ptx_ptr, tier_b_name_ptr)
        } else {
            (ptx_ptr, name_ptr)
        };

    #[cfg(feature = "cuda")]
    {
        // ... existing body, but replace `ptx_ptr` and `name_ptr` usage with
        //     `effective_ptx_ptr` and `effective_name_ptr`.
    }

    /* return value unchanged */
    0
}
```

Update the body's `cuModuleLoadData(ptx_ptr as *const _)` → `cuModuleLoadData(effective_ptx_ptr as *const _)`. Same for the kernel-name lookup.

- [ ] **Step 3:** Repeat the same extension pattern for the other 5 entry points:

  - `nsl_flash_attention:109` — same shape; uses `ptx_ptr`/`name_ptr` from its base signature. For this non-CSHA entry, `segment_ids_ptr` may not exist as a parameter; if so, the runtime gate's `segment_ids_ptr` argument is always 0 (Tier B never fires for non-CSHA paths since Tier B requires segment masking). Or skip the runtime branch entirely for this entry: pass `0, 0` sentinels at all current call sites.
  - `nsl_flash_attention_csha_with_saves:597` — has `segment_ids_ptr`; full extension applies.
  - `nsl_flash_attention_csha_backward:883` — has `segment_ids_ptr`; full extension applies.
  - `nsl_flash_attention_quantized:1459` — uses `ptx_ptr` for the quantized variant; verify `segment_ids_ptr` availability.
  - `nsl_flash_attention_backward:2107` — backward equivalent of `nsl_flash_attention`.

- [ ] **Step 4:** Add a header documentation block at the top of each extended function documenting the Tier B params per planner spec §4.5:

```rust
/// ... existing docstring ...
///
/// # Tier B extension (planner spec §4)
///
/// The trailing `tier_b_ptx_ptr, tier_b_name_ptr` parameters carry the Tier-B-on
/// variant per the planner spec's case-(β-ii) rehabilitated dispatch.
///
/// **Sentinel encoding:** `(0, 0)` = no Tier-B-on variant available (default for
/// non-`segment_masked` configs). Non-zero pair = codegen emitted a Tier-B-on
/// variant for this config.
///
/// **Precondition:** sentinel pair must agree (both zero or both non-zero).
/// Mismatched pairs trigger `assert_tier_b_sentinels` → process abort with diagnostic.
///
/// **Construction discipline:** Cranelift-side call sites MUST emit the sentinel
/// via `pca_tier_b::tier_b_disabled_sentinel()` or `pca_tier_b::tier_b_enabled(...)`,
/// not inline `0, 0` literals.
///
/// See `docs/superpowers/specs/2026-05-15-pca-tier-b-planner-design.md` §4 and
/// `docs/superpowers/specs/2026-05-15-tier-b-bii-smem-probe-findings.md`.
```

- [ ] **Step 5:** Verify all 6 entry points build:

```bash
cargo build -p nsl-runtime
```

Expected: build fails because existing call sites at `crates/nsl-codegen/src/compiler/kernel.rs:750, 1050, 1173` don't pass the new params. That's expected — Task P-3.5 fixes this.

- [ ] **Step 6:** Commit (even though build fails — the next task in sequence fixes it):

```bash
git add crates/nsl-runtime/src/flash_attention.rs
git commit -m "feat(pca-tier-b-planner): P-3.4 — FFI extension at 6 entry points (build fails until P-3.5)"
```

### Task P-3.5: Production call-site migrations

Migrate the 3 production callers in `crates/nsl-codegen/src/compiler/kernel.rs` to use `emit_tier_b_variants_for_config` + the sentinel helpers.

- [ ] **Step 1:** Inspect `crates/nsl-codegen/src/compiler/kernel.rs:750` (the `maybe_synthesize_csha_training_ptx` enclosing fn). Find the FFI call-argument-list construction (likely just after the PTX synthesis call).

- [ ] **Step 2:** Replace the call-pattern. Before:

```rust
let fwd_ptx_bytes = crate::flash_attention_selector::synthesize_flash_attention_ptx_selected_with_diag(
    &training_config, &mut diags,
);
// ... build FFI call args ending with segment_ids_ptr ...
```

After:

```rust
use crate::pca_tier_b::{emit_tier_b_variants_for_config, tier_b_disabled_sentinel, tier_b_enabled};

let emission = emit_tier_b_variants_for_config(&training_config);

// Embed base PTX as existing pattern.
let base_ptx_data: DataId = /* embed emission.base_ptx as data section */;
let base_name_data: DataId = /* embed emission.base_kernel_name as data section */;

// Embed Tier-B-on PTX if present.
let (tier_b_ptx_data_opt, tier_b_name_data_opt) = match (
    emission.tier_b_on_ptx,
    emission.tier_b_on_kernel_name,
) {
    (Some(on_ptx), Some(on_name)) => {
        let on_ptx_data: DataId = /* embed on_ptx as data section */;
        let on_name_data: DataId = /* embed on_name as data section */;
        (Some(on_ptx_data), Some(on_name_data))
    }
    _ => (None, None),
};

// ... build FFI call args ending with segment_ids_ptr ...

// Tier B extension: emit sentinel pair via helpers (planner spec §4.2).
let [tier_b_ptx_arg, tier_b_name_arg] = match (tier_b_ptx_data_opt, tier_b_name_data_opt) {
    (Some(p), Some(n)) => tier_b_enabled(builder, module, p, n),
    _ => tier_b_disabled_sentinel(builder),
};
ffi_args.push(tier_b_ptx_arg);
ffi_args.push(tier_b_name_arg);
```

Adapt the data-section embedding pattern to match the existing `kernel.rs:750`'s idioms (the existing code already embeds `base_ptx` and `base_kernel_name`; copy that pattern for the Tier-B-on variants).

- [ ] **Step 3:** Repeat the same migration pattern at `kernel.rs:1050` (autotune variant loop) and `kernel.rs:1173` (single-config fallback).

- [ ] **Step 4:** Verify the workspace builds:

```bash
cargo build --workspace
```

Expected: clean build. The 6 FFI entry points and 3 call sites now agree on the extended signature.

- [ ] **Step 5:** Run existing tests to confirm no behavioral regression:

```bash
cargo test --workspace --lib
```

Expected: no failures from existing library tests. Snapshot tests may fail (P-4 scope); ignore for now.

- [ ] **Step 6:** Commit:

```bash
git add crates/nsl-codegen/src/compiler/kernel.rs
git commit -m "feat(pca-tier-b-planner): P-3.5 — migrate 3 production callers to Tier B emission helper"
```

### Task P-3.6: Runtime gate truth-table tests

- [ ] **Step 1:** Create `crates/nsl-runtime/tests/pca_tier_b_dispatch.rs`:

```rust
//! Runtime gate truth-table tests for `should_dispatch_tier_b_at_runtime`.
//! Per planner spec §6.7.

use nsl_runtime::pca_tier_b_runtime::{
    should_dispatch_tier_b_at_runtime, TIER_B_MAX_BAKED_SEQ_LEN, TIER_B_SEQ_LEN_FLOOR,
};

#[test]
fn dispatch_tier_b_on_happy_path() {
    assert!(should_dispatch_tier_b_at_runtime(
        /* tier_b_ptx_ptr */ 0xdeadbeef,
        /* segment_ids_ptr */ 0x12345678,
        /* seq_len */ TIER_B_SEQ_LEN_FLOOR,
    ));
}

#[test]
fn dispatch_tier_b_on_at_max_baked() {
    assert!(should_dispatch_tier_b_at_runtime(
        0xdeadbeef, 0x12345678, TIER_B_MAX_BAKED_SEQ_LEN,
    ));
}

#[test]
fn no_dispatch_when_no_tier_b_emitted() {
    assert!(!should_dispatch_tier_b_at_runtime(
        /* tier_b_ptx_ptr */ 0,
        0x12345678,
        TIER_B_SEQ_LEN_FLOOR,
    ));
}

#[test]
fn no_dispatch_when_no_segment_ids() {
    assert!(!should_dispatch_tier_b_at_runtime(
        0xdeadbeef,
        /* segment_ids_ptr */ 0,
        TIER_B_SEQ_LEN_FLOOR,
    ));
}

#[test]
fn no_dispatch_below_floor() {
    if TIER_B_SEQ_LEN_FLOOR > 0 {
        assert!(!should_dispatch_tier_b_at_runtime(
            0xdeadbeef, 0x12345678, TIER_B_SEQ_LEN_FLOOR - 1,
        ));
    }
}

#[test]
fn no_dispatch_above_max_baked() {
    assert!(!should_dispatch_tier_b_at_runtime(
        0xdeadbeef, 0x12345678, TIER_B_MAX_BAKED_SEQ_LEN + 1,
    ));
}
```

- [ ] **Step 2:** Run the tests:

```bash
cargo test -p nsl-runtime --test pca_tier_b_dispatch
```

Expected: 6/6 PASS.

- [ ] **Step 3:** Commit:

```bash
git add crates/nsl-runtime/tests/pca_tier_b_dispatch.rs
git commit -m "test(pca-tier-b-planner): P-3.6 — runtime gate truth-table tests (6/6)"
```

### Task P-3.7: Dispatcher integration test (branch wiring)

- [ ] **Step 1:** Append to `crates/nsl-runtime/tests/pca_tier_b_dispatch.rs`:

```rust
/// Integration test: verifies the dispatcher's 2-line branch in nsl_flash_attention_csha
/// correctly picks (base_ptx, base_name) when the runtime gate fires OFF for an
/// orthogonal reason (seq_len < FLOOR, sentinel pair consistent).
///
/// Catches inverted-branch failure mode (gate fires correctly but dispatcher picks
/// Tier-B-on instead of base). Distinct surface from the truth-table tests above
/// (IR-006 — distinct test surfaces for distinct failure modes).
#[test]
fn dispatch_branch_picks_base_ptx_when_runtime_gate_false() {
    // Construct: consistent sentinels (both non-zero), but seq_len < FLOOR so gate is OFF.
    let tier_b_ptx_ptr: i64 = 0xdeadbeef;
    let tier_b_name_ptr: i64 = 0xfeedface;
    let segment_ids_ptr: i64 = 0x12345678;
    let base_ptx_ptr: i64 = 0xa0a0a0a0_u64 as i64;
    let base_name_ptr: i64 = 0xb0b0b0b0_u64 as i64;
    let seq_len = if TIER_B_SEQ_LEN_FLOOR > 0 { TIER_B_SEQ_LEN_FLOOR - 1 } else { 0 };

    // Replicate the dispatcher's 2-line branch logic.
    let gate_result = should_dispatch_tier_b_at_runtime(
        tier_b_ptx_ptr, segment_ids_ptr, seq_len,
    );
    let (effective_ptx_ptr, effective_name_ptr) = if gate_result {
        (tier_b_ptx_ptr, tier_b_name_ptr)
    } else {
        (base_ptx_ptr, base_name_ptr)
    };

    // Below-floor seq_len should select base PTX, not Tier-B-on.
    assert!(!gate_result, "Below-floor seq_len should produce false gate result");
    assert_eq!(effective_ptx_ptr, base_ptx_ptr,
        "Gate OFF should select base PTX, not Tier-B-on (inverted-branch regression?)");
    assert_eq!(effective_name_ptr, base_name_ptr);
}
```

- [ ] **Step 2:** Run:

```bash
cargo test -p nsl-runtime --test pca_tier_b_dispatch dispatch_branch_picks_base_ptx_when_runtime_gate_false
```

Expected: PASS.

- [ ] **Step 3:** Commit:

```bash
git add crates/nsl-runtime/tests/pca_tier_b_dispatch.rs
git commit -m "test(pca-tier-b-planner): P-3.7 — dispatcher branch wiring integration test"
```

**Gate to P-4:** Per planner spec §10.1 — runtime parity tests pass (all 16 from PR #169); 6 runtime gate unit tests pass; 1 integration test passes; cargo build clean. Snapshot tests WILL FAIL at this gate (expected; P-4 scope).

- [ ] **Step 4:** Run the PR #169 runtime parity tests:

```bash
cargo test -p nsl-codegen --test pca_tier_b_m3_parity
# plus any forward parity test surfaced by:
git grep -ln 'tier_b.*parity' crates/nsl-codegen/tests
```

Expected: all 16 runtime parity tests PASS (PR #169 inherited unchanged). Snapshot tests may have `.snap.new` files — that's expected; P-4 re-baselines them.

---

## Task P-4: Test Surface + Snapshots + Activation PR

**Scope:** Remaining test surfaces (codegen emission tests, FFI sentinel tests, snapshot re-baselining per §8.3 cascade), institutional-rules registry maintenance per §12.1, activation PR.

**Budget:** ~0.5 day.

### Task P-4.1: Codegen emission-helper tests

- [ ] **Step 1:** Create `crates/nsl-codegen/tests/pca_tier_b_emission.rs`:

```rust
//! Codegen emission-helper tests. Per planner spec §5.8.

use nsl_codegen::flash_attention::FlashAttentionConfig;
use nsl_codegen::pca_tier_b::{
    emit_tier_b_variants_for_config,
    flash_attention_kernel_name_v2_tier_b_on,
    should_emit_tier_b_at_codegen,
    TIER_B_MAX_BAKED_SEQ_LEN,
};

fn segment_masked_config() -> FlashAttentionConfig {
    let mut cfg = FlashAttentionConfig::csha_canonical();
    cfg.segment_masked = true;
    cfg.causal = true;
    cfg
}

#[test]
fn emission_helper_returns_two_blobs_for_segment_masked() {
    let cfg = segment_masked_config();
    let result = emit_tier_b_variants_for_config(&cfg);
    assert!(!result.base_ptx.is_empty(), "base PTX must be non-empty");
    assert!(result.tier_b_on_ptx.is_some(), "Tier-B-on PTX must be emitted for segment_masked config");
    assert!(result.tier_b_on_kernel_name.is_some());
    let on_ptx = result.tier_b_on_ptx.unwrap();
    assert!(!on_ptx.is_empty(), "Tier-B-on PTX must be non-empty");
}

#[test]
fn emission_helper_returns_one_blob_for_non_segment_masked() {
    let mut cfg = segment_masked_config();
    cfg.segment_masked = false;
    let result = emit_tier_b_variants_for_config(&cfg);
    assert!(!result.base_ptx.is_empty());
    assert!(result.tier_b_on_ptx.is_none(), "Tier-B-on must NOT be emitted for non-segment_masked");
    assert!(result.tier_b_on_kernel_name.is_none());
}

#[test]
fn kernel_name_distinctness() {
    let cfg = segment_masked_config();
    let base_name = nsl_codegen::flash_attention_v2::flash_attention_kernel_name_v2(&cfg);
    let on_name = flash_attention_kernel_name_v2_tier_b_on(&cfg);
    assert_ne!(base_name, on_name, "Tier-B-on kernel name must differ from base");
    let expected_suffix = format!("_tier_b_max{}", TIER_B_MAX_BAKED_SEQ_LEN);
    assert!(on_name.ends_with(&expected_suffix),
        "Tier-B-on kernel name must end with {expected_suffix}; got {on_name}");
    assert!(should_emit_tier_b_at_codegen(&cfg), "helper must agree heuristic fires for this config");
}

// Build-time const assertion is already in nsl-tier-b-constants/src/lib.rs.
// Compile failure there is the test; no separate runtime test needed.
```

- [ ] **Step 2:** Run:

```bash
cargo test -p nsl-codegen --test pca_tier_b_emission
```

Expected: 3/3 PASS. The const assertion (4th "test" per planner spec §5.8) is verified at compile time by the workspace build.

- [ ] **Step 3:** Commit:

```bash
git add crates/nsl-codegen/tests/pca_tier_b_emission.rs
git commit -m "test(pca-tier-b-planner): P-4.1 — codegen emission-helper tests (3/3)"
```

### Task P-4.2: FFI sentinel-discipline tests

- [ ] **Step 1:** Add the `bench-internal` Cargo feature to `crates/nsl-runtime/Cargo.toml`:

```toml
[features]
default = []
bench-internal = []  # NEW — enables sentinel-mismatch test that aborts on failure.
```

- [ ] **Step 2:** Create `crates/nsl-runtime/tests/pca_tier_b_ffi_sentinel.rs`:

```rust
//! FFI sentinel-discipline tests. Per planner spec §4.3 + §8.5.
//! Gated behind `bench-internal` to avoid triggering aborts in normal test runs.

#![cfg(feature = "bench-internal")]

use nsl_runtime::pca_tier_b_runtime::assert_tier_b_sentinels;

/// Helper-roundtrip test (planner spec §8.5 item 2).
/// Verifies that consistent sentinel pairs DON'T trigger the assertion.
#[test]
fn helper_roundtrip_disabled_sentinel_passes_assertion() {
    // (0, 0) is the disabled-sentinel pair — must not abort.
    assert_tier_b_sentinels("test_entry", 0, 0);
}

#[test]
fn helper_roundtrip_enabled_sentinel_passes_assertion() {
    // Two non-zero values represent an enabled pair — must not abort.
    assert_tier_b_sentinels("test_entry", 0xdeadbeef, 0xfeedface);
}

/// Sentinel-pair-mismatch test (planner spec §8.5 item 1).
/// Verifies that mismatched sentinels DO trigger the assertion (and process abort).
#[test]
#[should_panic(expected = "FATAL")]
fn helper_bypass_mismatched_sentinel_triggers_abort() {
    // This test will abort the process — but `#[should_panic]` catches the abort
    // when the abort is implemented via panic-and-unwind. If `assert_tier_b_sentinels`
    // uses `std::process::abort()` directly (not unwinding), this test cannot
    // observe the abort and should be skipped or rewritten with a panic-based
    // assertion variant for testing purposes.
    //
    // Implementation note: provide a `cfg(test)` shim that swaps `process::abort`
    // for `panic!` to make this testable. The production behavior remains `abort`.
    assert_tier_b_sentinels("test_entry", 0xdeadbeef, 0);
}
```

(Implementation note: if `std::process::abort()` makes the test untestable, add a `#[cfg(test)] mod test_shims { ... }` in `pca_tier_b_runtime.rs` that replaces the abort with a panic during test builds. The production behavior stays `process::abort`.)

- [ ] **Step 3:** Run:

```bash
cargo test -p nsl-runtime --features bench-internal --test pca_tier_b_ffi_sentinel
```

Expected: 3/3 PASS (or 2/3 + 1 documented skip if abort can't be panic-caught in this Rust version).

- [ ] **Step 4:** Commit:

```bash
git add crates/nsl-runtime/Cargo.toml crates/nsl-runtime/tests/pca_tier_b_ffi_sentinel.rs
git commit -m "test(pca-tier-b-planner): P-4.2 — FFI sentinel-discipline tests"
```

### Task P-4.3: Snapshot re-baseline per §8.3 cascade

The snapshot re-baselining only affects fixtures if `TIER_B_MAX_BAKED_SEQ_LEN` differs from PR #169's MAX=4096 baseline. If P-0.6 resolved MAX=4096 (B-i path), the existing snapshots are byte-identical and re-baselining is a no-op. If MAX=8192 or 16384, the snapshots re-baseline with the new SMEM allocation size.

- [ ] **Step 1:** Run all snapshot tests to surface `.snap.new` files:

```bash
cargo test -p nsl-codegen --tests 2>&1 | tee /tmp/p4_snapshot_test_run.txt
ls crates/nsl-codegen/tests/snapshots/*.snap.new 2>/dev/null | tee /tmp/p4_affected_snapshots.txt
```

Expected: ≤8 `.snap.new` files; exact count varies by `MAX_BAKED` resolution.

- [ ] **Step 2:** Verify the 2 isolation snapshots are NOT in the affected set:

```bash
ls crates/nsl-codegen/tests/snapshots/pca_tier_b_preamble_isolation*.snap.new 2>/dev/null
ls crates/nsl-codegen/tests/snapshots/pca_tier_b_predicate_isolation*.snap.new 2>/dev/null
```

Expected: nothing (no `.snap.new` for isolation tests). If a `.snap.new` IS generated for either, STOP — per planner spec §8.2, isolation-test drift indicates an unexpected call-graph dependency; investigation precedes acceptance.

- [ ] **Step 3:** For each `.snap.new` in the affected set, do a region-by-region diff verification per planner spec §8.3:

```bash
for f in crates/nsl-codegen/tests/snapshots/*.snap.new; do
  old="${f%.new}"
  echo "=== $old ==="
  diff "$old" "$f" || true
done | tee /tmp/p4_diff_review.txt
```

Verify each diff is localized to:

  - `.shared seg_smem[N]` declaration's `[N]` byte count.
  - Range-table indexing offsets in the preamble.
  - `compute_range_table_bytes` constant in SMEM size computation.

If a diff bleeds into any of the 6 should-stay-stable regions (S-compute, skip predicate, tile-skip branch, post-skip body, softmax, writeback), STOP — the re-baseline is structurally wrong.

- [ ] **Step 4:** Accept the snapshot updates:

```bash
cargo insta accept --workspace-root crates/nsl-codegen
```

Or per-file: `for f in crates/nsl-codegen/tests/snapshots/*.snap.new; do mv "$f" "${f%.new}"; done`.

- [ ] **Step 5:** Re-run all tests:

```bash
cargo test --workspace
```

Expected: green across the board.

- [ ] **Step 6:** Commit:

```bash
git add crates/nsl-codegen/tests/snapshots/
git commit -m "test(pca-tier-b-planner): P-4.3 — re-baseline affected snapshots per §8.3 cascade"
```

### Task P-4.4: Institutional-rules registry maintenance

Per planner spec §12.1.

- [ ] **Step 1:** Edit `docs/wiki/institutional-rules.md`. Find IR-008's "Cited from" list and append:

```markdown
- `docs/superpowers/specs/2026-05-15-pca-tier-b-planner-design.md` §8.10 — 1.7× test-LOC-to-implementation-LOC ratio as a concrete instance of IR-008's observational framing (not a budget ceiling per IR-008's original framing).
```

- [ ] **Step 2:** Find IR-011's "Cited from" list and append:

```markdown
- `docs/superpowers/specs/2026-05-15-pca-tier-b-planner-design.md` §9 — V-Bii-SMEM five-outcome decision matrix + SMEM-headroom recording roll up into a four-way v2-trigger decision space (third instance of IR-011's institutional value; first: dispatch spec §13.3 sensitivity tier; second: this spec's trigger-space split).
```

- [ ] **Step 3:** Commit:

```bash
git add docs/wiki/institutional-rules.md
git commit -m "docs(institutional-rules): P-4.4 — extend IR-008 and IR-011 'Cited from' lists for planner spec"
```

### Task P-4.5: Activation PR

- [ ] **Step 1:** Verify the full test plan checklist passes per planner spec §8.9:

```bash
cargo build --workspace
cargo test --workspace
```

Expected: green everywhere.

- [ ] **Step 2:** Push the branch and open the activation PR:

```bash
git push -u origin worktree-feat-pca-tier-b-dispatch
gh pr create --title "feat(pca-tier-b): activate runtime-dispatch Option B (sub-variant: <X>; FLOOR=<F>; MAX_BAKED=<M>)" \
  --body "$(cat <<'EOF'
## Tier B activation — context decisions

- **V-Bii-SMEM outcome:** <B-ii unrestricted | B-ii-restricted (MAX=8192) | B-i (MAX=4096)> per [`2026-05-15-tier-b-bii-smem-probe-findings.md`](docs/superpowers/specs/2026-05-15-tier-b-bii-smem-probe-findings.md).
  - `TIER_B_MAX_BAKED_SEQ_LEN = <value>`
  - FFI extension shape: 2-param (uniform across sub-variants per planner spec §4.1)
- **D-2 outcome:** <clear pass | case-(a) sub-threshold | case-(b) Option B re-evaluation> per [`2026-05-15-tier-b-floor-derivation-findings.md`](docs/superpowers/specs/2026-05-15-tier-b-floor-derivation-findings.md).
  - `TIER_B_SEQ_LEN_FLOOR = <value>`
- **Constant-source location:** <β shared crate | α runtime-as-source> per §6.2 fallback decision (dep-graph inspection: <result>).

## Summary

Activates PCA Tier B in production for `segment_masked` configs at runtime via the kernel-launch wrapper's 2-line dispatch branch. Replaces the dispatch spec's stalled codegen-time activation per the §14 amendment's pivot to Option B.

Spec: [`2026-05-15-pca-tier-b-planner-design.md`](docs/superpowers/specs/2026-05-15-pca-tier-b-planner-design.md).

## Snapshot re-baseline (per planner spec §8.3 cascade)

**N re-baselined:** [list each affected `.snap` file with region-by-region diff summary]

**2 expected-stable (verified byte-identical, NOT re-baselined):**
- `pca_tier_b_preamble_isolation__*`
- `pca_tier_b_predicate_isolation__*`

Cascade narrative per planner spec §8.3:
1. `TIER_B_MAX_BAKED_SEQ_LEN = <value>` (differs from PR #169's MAX=4096 baseline).
2. Tier-B-on PTX's range-table size differs accordingly.
3. Snapshot diffs are localized to: `.shared seg_smem[N]` byte count + range-table indexing offsets + `compute_range_table_bytes` constant.
4. The 6 should-stay-stable regions (S-compute, skip predicate, tile-skip branch, post-skip body, softmax, writeback) are byte-identical.

## Test plan

- [x] Build clean (`cargo build --workspace`)
- [x] All 16 runtime parity tests pass (PR #169 inherited)
- [x] 2 isolation snapshot tests stay byte-identical (no re-baseline)
- [x] Affected PTX snapshots re-baselined per §8.3 cascade (N snapshots; region-by-region diff summary above)
- [x] 6 runtime gate unit tests pass (truth table)
- [x] 1 runtime dispatcher integration test passes (branch wiring)
- [x] 2 FFI sentinel tests pass under `bench-internal` feature
- [x] 3 codegen emission-helper tests pass (const assertion verified at build time)
- [x] Cross-crate constant-agreement test passes (or N/A if (β) shared crate used)
- [x] Constants in `nsl-tier-b-constants` match findings-doc values; both findings docs cited above

## Institutional rules

Extends IR-008 and IR-011 "Cited from" lists per planner spec §12.1.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Fill in actual `<value>` placeholders and the snapshot re-baseline list with the actual file paths from `/tmp/p4_affected_snapshots.txt`.

- [ ] **Step 3:** Verify the PR was created and capture the URL.

**Final gate:** PR opened with cascade narrative + context-decision pinning + test plan complete; reviewer approval triggers merge.

---

## Out-of-scope reminders (per planner spec §2.2 + §9)

If during P-3 or P-4 you encounter any of these, STOP and document as a v2 trigger in `/tmp/p4_v2_triggers_observed.txt` — DO NOT in-line them in this PR:

- TierBPolicy enum (user-overrideable `ForceOn`/`ForceOff`) — dispatch spec §9.1 v2.
- Planner module above codegen — dispatch spec §9.2 v3.
- Per-config `SegmentResidency` selection — dispatch spec §9.3 v2.
- Tier B-extended for seq_len > 16K — planner spec §9 #1/#2.
- B-ii-restricted MAX extension / B-iii migration — planner spec §9 #3/#4.
- Per-launch telemetry — planner spec §9 #5.
- Per-sparsity floor re-derivation — planner spec §9 #7.

Each has a concrete entry point per §9.3; document the entry point's specific trigger if observed.
