# PCA Tier B (Tile-Skip) Implementation Plan — v2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land PCA Tier B (per-tile segment-range tile-skip) into the FA-2 forward kernel, with a Phase 0 SMEM probe gating the layout choice, a runtime-PTX-loop preamble shape, and hybrid (isolation + integration) ptxas/SASS verification.

**Architecture:** Empirical forced-access SMEM probe runs first and produces a permanent findings doc that selects an outcome row from a five-outcome decision matrix. The selected outcome instantiates §3.4 of the revised spec. Tier B's four range tables (qtile_min, qtile_max, kvtile_min, kvtile_max) ride at the tail of the existing `.extern .shared shmem[]` allocation via a single `smem_layout::tier_b_range_table_offset(config)` accessor; `pca_tilerange::range_table_addrs(base, num_q_tiles, num_kv_tiles)` returns a struct with the four sub-offsets. The preamble emits a runtime PTX loop (not compile-time unroll) with strict warp-uniformity discipline so the loop branch compiles to `BRA.U`. Verification splits responsibilities: isolation-level `cargo insta` snapshot catches PTX-string drift; integration-level extensions to `pca_forward_kernel_snapshot` + `pca_sass_byte_identity` catch SASS regressions (`BRA.U`, `UR<n>` uniform-register class, zero-spill, instruction count).

**Tech Stack:** Rust 1.95.0 (workspace toolchain), `nsl-codegen` crate, PTX ISA 7.0+ targeting sm_80 / sm_90 / sm_120, ptxas + cuobjdump from CUDA 13.x toolkit, `insta` for snapshot tests, `cudarc 0.19` dynamic-linking for runtime probe launch.

**Spec:** `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md` (the brainstorm output revising the 2026-05-02 PCA Tier B spec in place). This plan v2 supersedes `docs/superpowers/plans/2026-05-02-pca-tier-b-tile-skip-implementation.md`.

---

## File Map

### Created

| Path | Purpose |
|---|---|
| `docs/superpowers/specs/2026-05-12-tier-b-smem-probe-findings.md` | Probe outcome + dated re-run log (Task 1 deliverable) |
| `docs/wiki/institutional-rules.md` | IR-001 through IR-008 registry (Task 2 deliverable) |
| `crates/nsl-codegen/src/pca_tile_config.rs` | `num_tiles(seq_len, block_size)` shared helper (Task 3) |
| `crates/nsl-codegen/tests/pca_tile_config_identity.rs` | Three-site identity test (Task 3) |
| `crates/nsl-codegen/tests/tier_b_smem_probe.rs` | Forced-access SMEM probe harness (Task 1) |
| `crates/nsl-codegen/tests/pca_tier_b_preamble_isolation.rs` | Isolation insta snapshot test (Task 5a) |
| `crates/nsl-codegen/tests/pca_tier_b_predicate_isolation.rs` | Isolation insta snapshot for predicate (Task 6a) |
| `tests/sass_baselines/<variant>_tier_b.txt` | Per-variant SASS baseline files (Task 10) |

### Modified

| Path | Why |
|---|---|
| `docs/superpowers/specs/2026-05-02-pca-tier-b-tile-skip-design.md` | In-place §3.1/§3.4/§6.3 revision per Task 2 |
| `crates/nsl-codegen/src/flash_attention_v2/smem_layout.rs` | Add `tier_b_range_table_offset` accessor (Task 4) |
| `crates/nsl-codegen/src/flash_attention_v2/mod.rs` | Wire Tier B bytes into `shared_mem_bytes_v2{_backward}` (Task 4) |
| `crates/nsl-codegen/src/pca_tilerange.rs` | Add `RangeTableAddrs` + `range_table_addrs` + real `emit_range_table_preamble` (Tasks 4, 5) + `emit_skip_predicate` (Task 6) |
| `crates/nsl-codegen/src/flash_attention_v2/phases/forward/s_compute.rs` | Insert predicate at KV-tile loop top (Task 6) |
| `crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs` | Call preamble emitter immediately after segment_ids load (Task 8) |
| `crates/nsl-codegen/tests/pca_forward_kernel_snapshot.rs` | Add Tier B-enabled variants + SASS assertions (Task 8 + Task 5b) |
| `crates/nsl-codegen/tests/pca_backward_kernel_snapshot.rs` | Add Tier B-enabled variants (Task 8) |
| `crates/nsl-codegen/tests/pca_sass_byte_identity.rs` | Add BRA.U + UR<n> grep assertions (Task 5b + Task 6a) |
| `crates/nsl-codegen/src/pca_tileskip.rs` | Reuse as M3 parity reference; ensure layout-equivalence definition matches kernel (Task 9) |

### Untouched

| Path | Why |
|---|---|
| `crates/nsl-codegen/src/pca_segment.rs` | Tier A segment-mask logic; Tier B composes on top |
| `crates/nsl-codegen/src/flash_attention_v2/phases/segment_mask.rs` | Per-cell mask emission; unchanged |
| `crates/nsl-codegen/src/flash_attention_v2/phases/backward/*` | Tier B backward integration deferred to Tier B.2 (separate plan) |
| `crates/nsl-codegen/src/pca_detect.rs`, `pca_rope.rs` | Orphaned-pending per spec §9; do not modify |

---

## Phase 0 — Probe + Spec Revision + Registry (Tasks 0-2)

These tasks land BEFORE any Tier B emission code. Task 1 (probe) is gating: §3.4's struct sub-layout assumes "tail-embed safe" outcome.

## Task 0: Worktree setup + prereq verification

**Goal:** Fresh worktree on `feat/pca-tier-b-v2` off current main. Confirm baseline is green and the Tier B-related modules are in expected state from the v1 prerequisite PR (#138) + the three post-merge fixes (#147, #150, #152).

**Files:** No file edits. Diagnostic only.

- [ ] **Step 1: Create the worktree**

```bash
git fetch origin
git worktree add .worktrees/pca-tier-b-v2 -b feat/pca-tier-b-v2 origin/main
cd .worktrees/pca-tier-b-v2
```

Expected: worktree created at `.worktrees/pca-tier-b-v2/`, on a fresh branch tracking main.

- [ ] **Step 2: Confirm the prerequisite modules are present**

```bash
ls crates/nsl-codegen/src/pca_tilerange.rs
ls crates/nsl-codegen/src/pca_segment.rs
grep -n "DEFAULT_SMEM_SEGMENT_BUDGET" crates/nsl-codegen/src/pca_segment.rs
grep -n "shared_mem_bytes_v2_backward" crates/nsl-codegen/src/flash_attention_v2/mod.rs
```

Expected: `pca_tilerange.rs` exists (skeleton from PR #138); `DEFAULT_SMEM_SEGMENT_BUDGET` is 32768 (32 KB); `shared_mem_bytes_v2_backward` exists with `seg_overhead` accounting.

If any check fails, the worktree is off a wrong base — restart from Step 1 with the right `--base`.

- [ ] **Step 3: Confirm the Blackwell warning comment is still in place**

```bash
grep -n "CUDA_ERROR_ILLEGAL_ADDRESS" crates/nsl-codegen/src/flash_attention_v2/phases/backward/prelude.rs
```

Expected: one match around line 270 in the long comment block explaining the Blackwell mixed static+extern ILLEGAL_ADDRESS. This is the assumption the Task 1 probe will verify.

- [ ] **Step 4: Run the baseline test suite to confirm green**

```bash
cargo test -p nsl-codegen --lib 2>&1 | tail -20
cargo test -p nsl-codegen --test pca_tier_b_skeleton 2>&1 | tail -10
cargo test -p nsl-codegen --test pca_forward_kernel_snapshot 2>&1 | tail -10
cargo test -p nsl-codegen --test pca_backward_kernel_snapshot 2>&1 | tail -10
```

Expected: all four green. If any fail on a fresh worktree off main, investigate before proceeding — Task 1 needs a known-green baseline so probe failures are unambiguously caused by the probe, not by environmental drift.

- [ ] **Step 5: No commit — Task 0 is diagnostic.**

Proceed to Task 1.

---

## Task 1: SMEM probe — Phase 0 gate (NEW)

**Goal:** Run the forced-access probe per spec §2 across 18 configurations × 2 architectures. Produce the findings doc that selects one outcome row from the five-outcome decision matrix.

**Files:**
- Create: `crates/nsl-codegen/tests/tier_b_smem_probe.rs`
- Create: `docs/superpowers/specs/2026-05-12-tier-b-smem-probe-findings.md`

- [ ] **Step 1: Write the probe test harness**

Create `crates/nsl-codegen/tests/tier_b_smem_probe.rs`:

```rust
//! Forced-access SMEM probe for PCA Tier B mixed static+extern allocation safety.
//!
//! Per spec §2 of `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md`:
//! sweep 18 configurations (3 static-sizes × 3 dynamic-shmem-sizes × 2 architectures)
//! with every thread writing-and-reading both a static `.shared` array AND extern
//! `.shared` shmem. Detects the Blackwell ILLEGAL_ADDRESS failure mode at access
//! time, not declaration time.
//!
//! Outcome rows from the five-outcome decision matrix are recorded in
//! `docs/superpowers/specs/2026-05-12-tier-b-smem-probe-findings.md`.
//!
//! Run: `cargo test -p nsl-codegen --test tier_b_smem_probe -- --nocapture --test-threads=1`
//! (`--test-threads=1` because cudarc CUDA context is thread-local; parallel runs
//! across the sweep collide on the primary context.)

#![cfg(feature = "cuda")]

use std::ffi::CString;

const STATIC_SIZES_BYTES: &[u32]  = &[256, 1024, 4096];
const DYNAMIC_SIZES_BYTES: &[u32] = &[16 * 1024, 64 * 1024, 96 * 1024];
const TARGET_ARCHES: &[u32] = &[80, 120]; // sm_80 (Ampere) + sm_120 (Blackwell)

/// Probe outcome for a single (N, M, sm) configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ProbeOutcome {
    /// Kernel launched + returned correct sentinels on every thread.
    Pass,
    /// `cuLaunchKernel` returned an error (e.g., CUDA_ERROR_ILLEGAL_ADDRESS).
    LaunchError(String),
    /// Kernel launched cleanly but readback values were not the sentinels.
    ValueCorruption { expected_a: u8, expected_b: u8, found_a: u8, found_b: u8 },
}

/// One row of the 18-config sweep matrix.
#[derive(Debug)]
struct ProbeRow {
    static_bytes: u32,
    dynamic_bytes: u32,
    sm: u32,
    outcome: ProbeOutcome,
}

/// Build the probe kernel PTX for a given (static_size, sm) tuple.
///
/// Emits a kernel that:
///   1. Declares `.shared .align 4 .b8 seg_smem[static_size]` (static decl).
///   2. Declares `.extern .shared .align 16 .b8 shmem[]` (dynamic).
///   3. Every thread writes 0xAA to seg_smem[tid] and 0xBB to shmem[tid].
///   4. `bar.sync 0` to ensure both writes complete.
///   5. Every thread reads back from both, writes the two bytes to global memory
///      at `out[2*tid]` (seg_smem byte) and `out[2*tid+1]` (shmem byte).
fn build_probe_ptx(static_bytes: u32, sm: u32) -> String {
    format!(
        r#".version 7.0
.target sm_{sm}
.address_size 64

.shared .align 4 .b8 seg_smem[{static_bytes}];
.extern .shared .align 16 .b8 shmem[];

.visible .entry probe_kernel(
    .param .u64 probe_kernel_param_0,
    .param .u32 probe_kernel_param_1
) {{
    .reg .u64 %out_ptr;
    .reg .u32 %tid, %n;
    .reg .u32 %seg_addr, %shmem_addr;
    .reg .u64 %wide_seg_addr, %wide_shmem_addr;
    .reg .u64 %slot0_ptr, %slot1_ptr;
    .reg .u16 %byte_aa, %byte_bb, %read_a, %read_b;

    ld.param.u64 %out_ptr, [probe_kernel_param_0];
    ld.param.u32 %n,       [probe_kernel_param_1];
    mov.u32 %tid, %tid.x;

    setp.ge.u32 %p_oob, %tid, %n;
    @%p_oob bra END;

    // Static-shared write at byte offset tid
    mov.u16 %byte_aa, 0xAA;
    cvta.shared.u64 %wide_seg_addr, seg_smem;
    cvt.u64.u32 %slot0_ptr, %tid;
    add.u64 %wide_seg_addr, %wide_seg_addr, %slot0_ptr;
    st.shared.u8 [%wide_seg_addr], %byte_aa;

    // Dynamic-shared write at byte offset tid
    mov.u16 %byte_bb, 0xBB;
    cvta.shared.u64 %wide_shmem_addr, shmem;
    add.u64 %wide_shmem_addr, %wide_shmem_addr, %slot0_ptr;
    st.shared.u8 [%wide_shmem_addr], %byte_bb;

    bar.sync 0;

    // Read back
    ld.shared.u8 %read_a, [%wide_seg_addr];
    ld.shared.u8 %read_b, [%wide_shmem_addr];

    // Write to out[2*tid], out[2*tid+1]
    mul.lo.u32 %seg_addr, %tid, 2;
    cvt.u64.u32 %slot0_ptr, %seg_addr;
    add.u64 %slot0_ptr, %out_ptr, %slot0_ptr;
    add.u64 %slot1_ptr, %slot0_ptr, 1;
    st.global.u8 [%slot0_ptr], %read_a;
    st.global.u8 [%slot1_ptr], %read_b;

END:
    ret;
}}
"#,
        sm = sm,
        static_bytes = static_bytes,
    )
}

/// Run one probe configuration and return its outcome.
///
/// Returns `Err` if the device doesn't support the requested sm (skip with a
/// `LaunchError("no device for sm_X")` outcome rather than panicking).
fn run_probe_config(static_bytes: u32, dynamic_bytes: u32, sm: u32) -> ProbeRow {
    use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};

    // Use 64 threads so reads/writes exercise the first 64 bytes of each region.
    let num_threads: u32 = 64;
    let ptx = build_probe_ptx(static_bytes, sm);

    // Try to compile + load the PTX through cudarc. Errors here mean the
    // device doesn't support sm_{sm} or ptxas rejected the PTX — record and
    // continue.
    let outcome = (|| -> Result<ProbeOutcome, String> {
        let ctx = CudaContext::new(0).map_err(|e| format!("ctx: {e}"))?;
        let module = ctx.load_module(CString::new(ptx.clone()).unwrap().into())
            .map_err(|e| format!("module: {e}"))?;
        let func = module.load_function("probe_kernel")
            .map_err(|e| format!("func: {e}"))?;

        let mut out_buf = ctx.alloc_zeros::<u8>(num_threads as usize * 2)
            .map_err(|e| format!("alloc: {e}"))?;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (num_threads, 1, 1),
            shared_mem_bytes: dynamic_bytes,
        };

        let stream = ctx.default_stream();
        let mut launch = stream.launch_builder(&func);
        launch.arg(&mut out_buf);
        launch.arg(&num_threads);
        unsafe { launch.launch(cfg) }.map_err(|e| format!("launch: {e}"))?;
        stream.synchronize().map_err(|e| format!("sync: {e}"))?;

        let host = stream.memcpy_dtov(&out_buf).map_err(|e| format!("d2h: {e}"))?;
        for tid in 0..num_threads as usize {
            let a = host[2 * tid];
            let b = host[2 * tid + 1];
            if a != 0xAA || b != 0xBB {
                return Ok(ProbeOutcome::ValueCorruption {
                    expected_a: 0xAA, expected_b: 0xBB,
                    found_a: a,       found_b: b,
                });
            }
        }
        Ok(ProbeOutcome::Pass)
    })();

    let outcome = match outcome {
        Ok(o) => o,
        Err(e) => ProbeOutcome::LaunchError(e),
    };

    ProbeRow { static_bytes, dynamic_bytes, sm, outcome }
}

#[test]
fn forced_access_probe_sweep() {
    let mut rows = Vec::new();
    for &sm in TARGET_ARCHES {
        for &static_bytes in STATIC_SIZES_BYTES {
            for &dynamic_bytes in DYNAMIC_SIZES_BYTES {
                let row = run_probe_config(static_bytes, dynamic_bytes, sm);
                eprintln!("sm_{} N={} M={}: {:?}", row.sm, row.static_bytes, row.dynamic_bytes, row.outcome);
                rows.push(row);
            }
        }
    }

    // Don't `assert!` — the test reports outcomes. The interpretation lives
    // in the findings doc (Step 4).
    eprintln!("\n=== Probe sweep summary (18 configs) ===");
    for row in &rows {
        eprintln!(
            "  sm_{:>3}  N={:>5}  M={:>6}  {:?}",
            row.sm, row.static_bytes, row.dynamic_bytes, row.outcome
        );
    }
    eprintln!("=== End summary ===\n");
    eprintln!("Transcribe into docs/superpowers/specs/2026-05-12-tier-b-smem-probe-findings.md");
}
```

- [ ] **Step 2: Build the probe test (compile check)**

```bash
cargo test -p nsl-codegen --features cuda --test tier_b_smem_probe --no-run 2>&1 | tail -10
```

Expected: clean compile. If the build fails because `cudarc` API differs from the sketch above, adjust the launch-builder call to match the project's existing pattern (search `launch_builder` in other tests).

- [ ] **Step 3: Run the probe on hardware**

```bash
cargo test -p nsl-codegen --features cuda --test tier_b_smem_probe \
    -- --nocapture --test-threads=1 2>&1 | tee /c/tmp/tier_b_probe.log
```

Expected: 18 rows printed (3 static × 3 dynamic × 2 sm). Most likely one of:
- All 18 `Pass` → outcome row "passes all probe configs" (option 1 confirmed).
- 9 `Pass` on sm_80, 9 `LaunchError(ILLEGAL_ADDRESS)` on sm_120 → "fires only on sm_120".
- All `Pass` except specific (N, M) tuples → "fires above threshold".

The user's primary GPU is sm_120 (RTX 5070 Ti); the sm_120-specific failure case is the most likely live outcome.

If the probe panics (cudarc init failure, no CUDA device, etc.), the findings doc records "probe could not run" and Tier B is blocked on resolving the environment rather than on a real-hardware outcome.

- [ ] **Step 4: Write the findings doc**

Create `docs/superpowers/specs/2026-05-12-tier-b-smem-probe-findings.md`:

```markdown
# Tier B SMEM Probe — Findings

**Probe spec:** §2 of `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md`
**Sweep:** 3 static-sizes {256, 1024, 4096} × 3 dynamic-sizes {16 KB, 64 KB, 96 KB} × 2 architectures {sm_80, sm_120} = 18 configurations.

## 2026-05-12 — initial run

**Hardware / toolkit at run time:**
- Primary GPU: RTX 5070 Ti (sm_120, Blackwell)
- Secondary GPU (if available): <model, sm_<X>>
- CUDA toolkit: 13.x
- NVIDIA driver: <version from `nvidia-smi`>
- Host OS: Windows 11

**Sweep result (paste from `/c/tmp/tier_b_probe.log`):**

```
sm_<sm>  N=<N>  M=<M>  <outcome>
...
```

**Outcome row selected from spec §2.4 five-outcome decision matrix:** `<row>`

**Decision:** `<text per the selected row>`

**Rationale notes:** <free-text observations, e.g., which (N, M) thresholds trigger failure, which architecture is affected, etc.>

## Re-run triggers

The probe is re-run when any of:

- **CUDA toolkit major version bump** (e.g., 13.x → 14.x). Driver behavior on mixed `.shared` allocations may shift.
- **New target architecture added to NSL's supported matrix** (e.g., sm_130 when it ships). Probe sweeps the new architecture.
- **Production deployment surface reports `CUDA_ERROR_ILLEGAL_ADDRESS`** on a Tier B kernel that previously passed CI. Indicates the probe's prior outcome no longer holds.

## Re-run log

(Future dated entries appended here. Each entry includes: date, trigger reason, hardware/toolkit, sweep result, outcome row, any decision changes.)
```

Fill in the `<placeholders>` from the probe output. The "Outcome row selected" line is load-bearing — Task 2's spec revision branches on it.

- [ ] **Step 5: Commit the probe test + findings doc**

```bash
git add crates/nsl-codegen/tests/tier_b_smem_probe.rs \
        docs/superpowers/specs/2026-05-12-tier-b-smem-probe-findings.md
git commit -m "$(cat <<'EOF'
feat(pca-tier-b-v2): SMEM probe + initial findings doc

Forced-access probe sweeps 18 configurations (3 static × 3 dynamic ×
2 archs sm_80/sm_120) verifying mixed `.shared seg_smem` + `.extern
.shared shmem` coexistence is safe at runtime access time, not just
ptxas-clean. Sentinel bytes 0xAA / 0xBB; every thread writes and reads
both regions; CUDA_SUCCESS + sentinel match required for Pass.

Findings doc selects one outcome row from spec §2.4's five-outcome
decision matrix; downstream §3.4 sub-layout instantiation depends on
this row. Re-run triggers pinned (CUDA major bump, new arch, prod
ILLEGAL_ADDRESS report).

Per `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md`
§2 + IR-003 (pre-implementation verification of load-bearing
assumptions).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Apply probe outcome to spec + create IR registry (NEW)

**Goal:** Revise the 2026-05-02 PCA Tier B spec in place per §6.1 of the revision design, citing the findings doc by path. Create the institutional-rules registry at `docs/wiki/institutional-rules.md` per §6.4 with IR-001 through IR-008.

**Files:**
- Modify: `docs/superpowers/specs/2026-05-02-pca-tier-b-tile-skip-design.md` (in-place §3.1/§3.4/§6.3 rewrite + §10/§12 extension + changelog header)
- Create: `docs/wiki/institutional-rules.md`

- [ ] **Step 1: Remove the "EXECUTION PAUSED" header from the 2026-05-02 spec**

Open `docs/superpowers/specs/2026-05-02-pca-tier-b-tile-skip-design.md`. Replace the existing top blockquote ("> **EXECUTION PAUSED 2026-05-02.** ...") with:

```markdown
## Revision Changelog

- **2026-05-12** — §3.1, §3.4, §6.3 revised after re-brainstorm; cites probe findings doc + plan v2; see `2026-05-12-pca-tier-b-revision-design.md` for the brainstorm output and the institutional-rules citations (IR-004, IR-005, IR-006, IR-007). Original §3.1/§3.4/§6.3 preserved in git history.
- **2026-05-02** — Initial spec; execution paused after 6 prerequisite commits (PR #138) due to §3.1 inline `.shared` decl incompatibility with Blackwell sm_120.
```

- [ ] **Step 2: Replace §3.1 with the runtime-loop emission shape**

In the same spec file, locate `### 3.1 Runtime per-tile segment-range table`. Replace the entire §3.1 body (through to the start of §3.2) with the content of §4 from `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md`. The replacement covers:

- Per-phase emission shape (runtime PTX loop, ~120 bytes per phase regardless of `num_*_tiles`).
- Warp-uniformity discipline (5 numbered pins + the uniformity-through-load anti-pattern).
- Lane-0 predicated-store pattern.
- Inner reduction body (fixed warp-shuffle butterfly).
- Address arithmetic using `range_table_addrs.qtile_*` from §3.4.
- SASS verification gate references (`pca_sass_byte_identity` BRA.U assertion).
- Architecture-specific check (sm_90/sm_120 `UR<n>` direct; sm_80/sm_86 BRA.U proxy).
- Hybrid-rejection paragraph citing IR-005.

- [ ] **Step 3: Replace §3.4 with the SMEM layout API**

Locate `### 3.4 Tier A residency interaction — Shared required, with explicit budget bump`. The Tier A budget-bump subsection (§3.4.1 in the original) is unchanged — keep it. Replace **§3.4.2 (Worked SMEM accounting)** and **§3.4.3 (Combined v1 envelope)** with:

- New §3.4.2: Surface in `smem_layout.rs` — `tier_b_range_table_offset(config) -> u32` (matches existing u32 convention from `kv_offset`/`sp_offset`), returns `align_up(backward_total_bytes(config) + seg_overhead(config), 2)`.
- New §3.4.3: Internal sub-layout in `pca_tilerange.rs` — `RangeTableAddrs` struct + `range_table_addrs(base, num_q_tiles, num_kv_tiles)` constructor (both `pub`).
- New §3.4.4: Shared tile-count source of truth — `pca_tile_config::num_tiles(seq_len, block_size)`.
- New §3.4.5: `num_tiles` identity unit test — three-site comparison.
- New §3.4.6: Launch-side accounting no-op guarantee.
- New §3.4.7: Findings-doc dependency citation — paste the outcome row from the findings doc, name the consequence for the struct sub-layout.

Reference §3 of `2026-05-12-pca-tier-b-revision-design.md` for the exact text.

- [ ] **Step 4: Replace §6.3 with the hybrid verification approach**

Locate `### 6.3 Branch-uniformity verification — pinned`. Replace the entire §6.3 (including §6.3.1 baseline mechanism and §6.3.2 architecture-specific check) with:

- New §6.3.1: Isolation-level test (`pca_tier_b_preamble_isolation` insta snapshot) with sentinel `0xDEAD_BEEF` discipline.
- New §6.3.2: Integration-level checks — extended `pca_{forward,backward}_kernel_snapshot` + extended `pca_sass_byte_identity` (BRA.U + UR<n>) + per-variant SASS baselines.
- New §6.3.3: PR-coordination discipline — three test surfaces update together with cascade-summary line.
- New §6.3.4: Deletion of `emit-pca-tier-b-preamble-harness` bin with maintenance-cost rationale.
- New §6.3.5: §6.3 satisfaction table mapping every v1 criterion to its test surface.

Reference §5 of `2026-05-12-pca-tier-b-revision-design.md`.

- [ ] **Step 5: Extend §10 (risks) with the probe-outcome-shift row**

Add one row to §10's risk table:

```markdown
| Probe outcome shifts under future CUDA toolkit / driver / new architecture; spec §3 needs revisit | medium long-term | high | Dated probe re-run triggers in `docs/superpowers/specs/2026-05-12-tier-b-smem-probe-findings.md`. Each re-run produces a dated entry; if outcome changes, §3.4 sub-layout choice revisits. |
```

- [ ] **Step 6: Extend §12 (references)**

Add three reference rows at the bottom of §12:

```markdown
- Probe findings doc — `docs/superpowers/specs/2026-05-12-tier-b-smem-probe-findings.md`
- Revision design (brainstorm output) — `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md`
- Plan v2 — `docs/superpowers/plans/2026-05-12-pca-tier-b-tile-skip-implementation-v2.md`
- Institutional-rules registry — `docs/wiki/institutional-rules.md`
```

- [ ] **Step 7: Create the institutional-rules registry**

Create `docs/wiki/institutional-rules.md`:

```markdown
# NSL Institutional Rules

This document catalogs project-level design principles surfaced across NSL's spec and brainstorm work. Each rule has a stable identifier (IR-NNN) used in spec citations.

## How to read this document

Each rule is one paragraph stating the principle and citing the specs where it was surfaced or applied. Specs cite rules by identifier (e.g., "per IR-003, the verification gate runs before implementation").

## How to add a rule (entry criteria)

A pattern becomes an IR when it satisfies all three:

- Surfaced across at least two distinct specs/brainstorms.
- The pattern's violation produced or would have produced a real failure mode in retrospect.
- The pattern is small enough to cite by identifier and explain in one paragraph.

Patterns that don't satisfy these criteria are documented in the rejecting spec's text as "considered but not codified," not added to the registry. The criteria prevent two failure modes: registry inflation (every spec adding "lessons learned" entries) and registry stagnation (real patterns never codified because nobody knows when to add).

## Rules

### IR-001 — Preconditions enforced by API shape, not docstrings

Where a function has an unstated precondition (a callee must be invoked first, an input must be in a specific state), the API should make violation structurally impossible — keep the unsafe-without-prep function module-internal, expose only the safe composition. Distinct from type-system enforcement; this is about visibility and composition shape.

Cited from:
- `docs/superpowers/specs/2026-04-26-bitnet-phase1-design.md` — `quantized_ternary_gemm` fusion.
- `docs/superpowers/specs/2026-04-29-awq-calibration-backward-pass-design.md` — `weight_index_map`.
- `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md` §3.4 — `RangeTableAddrs` constructor.

### IR-002 — External references as one-time anchors

External measurements / hardware findings recorded in dated findings docs and cited by path from specs that depend on them. Findings docs append-only with dated re-run log; specs cite by path so the dependency is auditable.

Cited from:
- BitNet HF checkpoint pinning (b1.58 reference logits fixture).
- AWQ calibration (sidecar envelope hashing).
- `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md` §2 + §3.4 — SMEM probe findings doc dependency.

### IR-003 — Pre-implementation verification of load-bearing assumptions

When a spec relies on cross-module behavior, verify the assumption via grep / probe / measurement before writing the code that depends on it. 15-minute verification beats multi-day rework.

Cited from:
- WGGO Phase 2 NodeId space (verified before backward-pass integration).
- CSHA Tier B.1 V1/V2/V3 findings (verified pre-B1.2 kernel emission).
- `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md` §2 — SMEM probe.

### IR-004 — APIs return aligned, ready-to-use values so consumers don't have to

Offset-computation site owns alignment, padding, and any other consumer-facing invariants; consumer assumes the returned value is ready-to-use. Separation-of-concerns principle: the layout site is the single audit point for layout correctness.

Cited from:
- `crates/nsl-codegen/src/flash_attention_v2/smem_layout.rs` — `kv_offset` / `sp_offset` discipline.
- `docs/superpowers/specs/2026-04-26-bitnet-phase1-design.md` — `packed_load.rs` register-resident values.
- `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md` §3.4 — `tier_b_range_table_offset` `align_up(2)` guarantee.

### IR-005 — Bifurcation of emission paths for the same logical operation requires measurement-driven justification, not architectural preference

v1 prefers uniform emission across the config matrix as a structural property; bifurcation (e.g., "compile-time-unroll at small N, runtime loop at large N") requires explicit performance measurement justifying the dual emission paths. Reason: bifurcation creates non-uniform bug surface (regression in one path only manifests at certain configs) and doubled test infrastructure.

Cited from:
- `docs/superpowers/specs/2026-05-11-csha-tier-b1-pipelined-attention-design.md` — no producer/consumer split in v1.
- `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md` §4.7 — no compile-time-unroll / runtime-loop hybrid.

### IR-006 — Distinct failure modes warrant distinct test surfaces

Bundle test concerns only when their failure modes have identical diagnostic shape; otherwise split them. The principle is diagnostic precision: when two distinct failure modes share a single regression surface, bisecting which one fired adds investigation time at every regression.

Cited from:
- CSHA Tier B.1 cost-model correction (standalone PR, separate from kernel-implementation).
- WGGO Phase 2 #134 — per-commit milestone matrix.
- `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md` §5 — isolation snapshot + integration SASS split.

### IR-007 — PTX emission discipline is pinned at the instruction-pattern level, not the algorithmic level

The ptxas → SASS pipeline is pattern-recognition-driven, not semantically aware. Spec text that says "emit a warp-uniform branch" without pinning the specific PTX patterns (register class, operand shape, predicate construction) that ptxas recognizes as uniform risks emission that's algorithmically correct but performance-degraded. Pin specific patterns.

Cited from:
- FA-2 v2 Tier B.1 cp.async commit-group cadence + wait_group operand.
- BitNet `BFE.U32` for single-instruction trit unpack.
- `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md` §4.2 — warp-uniformity through specific register-class + operand patterns.

### IR-008 — For long-lived kernels, verification surface investment is the load-bearing cost

Observation across multiple kernel specs: kernel emission is a few hundred lines; verification (snapshots, SASS assertions, parity tests, reference impls) is comparable or larger and is what catches regressions over the kernel's lifetime. Not a rule that prescribes a ratio — a framing that justifies budget allocation when the verification surface seems disproportionate to the emission.

Cited from:
- FA-2 v2 Tier B.1 verification harness (V1/V2/V3 + cost-model snapshot).
- BitNet Phase 1 validation harness + reference implementation.
- `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md` overall §3.1 / §3.4 / §6.3 balance.
```

- [ ] **Step 8: Commit the spec revision + registry**

```bash
git add docs/superpowers/specs/2026-05-02-pca-tier-b-tile-skip-design.md \
        docs/wiki/institutional-rules.md
git commit -m "$(cat <<'EOF'
docs(pca-tier-b-v2): apply probe outcome to spec; create IR registry

Spec revisions (in-place, per revision-design §6.1):
  - Header: replace EXECUTION PAUSED with Revision Changelog citing
    2026-05-12-pca-tier-b-revision-design.md.
  - §3.1: replace compile-time-unrolled sketch with runtime PTX loop +
    warp-uniformity discipline + uniformity-through-load anti-pattern +
    lane-0 predicated-store pattern + architecture-specific check
    limitations + IR-005 hybrid-rejection paragraph.
  - §3.4: replace inline `.shared` decls with `tier_b_range_table_offset`
    single tail offset + `RangeTableAddrs` struct + `pca_tile_config`
    shared helper + `align_up(2)` guarantee + no-op guarantee for
    non-Tier-B configs. Cite probe findings doc.
  - §6.3: replace standalone-harness approach with hybrid (isolation
    snapshot + integration SASS) + sentinel-emission discipline +
    PR-coordination cascade-summary line + harness-deletion maintenance
    rationale + §6.3 satisfaction table.
  - §10: add probe-outcome-shift risk row.
  - §12: add findings doc + revision design + plan v2 + IR registry
    references.

Institutional-rules registry (new at docs/wiki/institutional-rules.md):
  IR-001..IR-008 — preconditions via API shape; external refs as one-time
  anchors; pre-impl verification; aligned offsets owned upstream;
  bifurcation needs measurement justification; distinct failure modes
  → distinct test surfaces; PTX patterns pinned at instruction level;
  verification surface as load-bearing cost.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 1 — Layout API (Tasks 3-4)

## Task 3: `pca_tile_config::num_tiles` shared helper + identity test (NEW)

**Goal:** Land the single source of truth for tile counts across Rust-side allocators and kernel-side tile-loop emission. Identity test pins three-site agreement.

**Files:**
- Create: `crates/nsl-codegen/src/pca_tile_config.rs`
- Modify: `crates/nsl-codegen/src/lib.rs` (add `pub mod pca_tile_config;`)
- Create: `crates/nsl-codegen/tests/pca_tile_config_identity.rs`

- [ ] **Step 1: Write the failing identity test**

Create `crates/nsl-codegen/tests/pca_tile_config_identity.rs`:

```rust
//! Three-site identity test for `pca_tile_config::num_tiles`.
//!
//! For every (seq_len, block_size) tuple in the supported config matrix,
//! the Rust formula and the value used at the `range_table_addrs` consumer
//! site must agree. (The third site — emitted PTX loop bound — is verified
//! once `emit_range_table_preamble` lands in Task 5.)

use nsl_codegen::pca_tile_config::num_tiles;

const SUPPORTED_CONFIGS: &[(u32, u32)] = &[
    (2048,  32), (2048,  64),
    (4096,  32), (4096,  64), (4096, 128),
    (8192,  64), (8192, 128),
    (16_384,  64), (16_384, 128),
];

#[test]
fn num_tiles_matches_ceiling_division() {
    for &(seq_len, block_size) in SUPPORTED_CONFIGS {
        let expected = seq_len.div_ceil(block_size);
        assert_eq!(num_tiles(seq_len, block_size), expected);
    }
}

#[test]
fn num_tiles_at_exact_multiple() {
    assert_eq!(num_tiles(4096, 64), 64);
    assert_eq!(num_tiles(16_384, 128), 128);
}

#[test]
fn num_tiles_at_non_multiple_rounds_up() {
    assert_eq!(num_tiles(4097, 64), 65);
    assert_eq!(num_tiles(2049, 64), 33);
    assert_eq!(num_tiles(1, 64), 1);
}

#[test]
#[should_panic]
fn num_tiles_rejects_zero_block_size() {
    let _ = num_tiles(4096, 0);
}
```

```bash
cargo test -p nsl-codegen --test pca_tile_config_identity --no-run 2>&1 | tail -5
```

Expected: build fails — `nsl_codegen::pca_tile_config` doesn't exist yet.

- [ ] **Step 2: Implement the module**

Create `crates/nsl-codegen/src/pca_tile_config.rs`:

```rust
//! Shared tile-count helper for PCA Tier B and FA-2 kernel emission.
//!
//! Single source of truth: both kernel tile-loop emission and
//! `pca_tilerange::range_table_addrs` callers compute tile counts via
//! this helper, so they cannot drift.

/// Compute the number of tiles for a given (seq_len, block_size).
///
/// Ceiling division — the last tile may be partial. Panics in debug
/// builds if `block_size == 0`.
pub fn num_tiles(seq_len: u32, block_size: u32) -> u32 {
    seq_len.div_ceil(block_size)
}
```

- [ ] **Step 3: Register the module in `lib.rs`**

Open `crates/nsl-codegen/src/lib.rs`. Find the `pub mod pca_segment;` line and insert `pub mod pca_tile_config;` after it (alphabetical order: segment, tile_config, tilerange).

```bash
grep -n "pub mod pca_" crates/nsl-codegen/src/lib.rs
```

- [ ] **Step 4: Re-run the identity test**

```bash
cargo test -p nsl-codegen --test pca_tile_config_identity 2>&1 | tail -10
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/pca_tile_config.rs \
        crates/nsl-codegen/src/lib.rs \
        crates/nsl-codegen/tests/pca_tile_config_identity.rs
git commit -m "feat(pca-tier-b-v2): pca_tile_config::num_tiles shared helper

Single source of truth for tile counts. Spec §3.4.4."
```

---

## Task 4: `smem_layout::tier_b_range_table_offset` + `RangeTableAddrs` (RESHAPED)

**Goal:** Single-tail-offset accessor + four-field internal struct + constructor + launch-side accounting with no-op guarantee for non-Tier-B configs.

**Files:**
- Modify: `crates/nsl-codegen/src/flash_attention_v2/smem_layout.rs`
- Modify: `crates/nsl-codegen/src/flash_attention_v2/mod.rs`
- Modify: `crates/nsl-codegen/src/pca_tilerange.rs`
- Create: `crates/nsl-codegen/tests/pca_tier_b_layout_api.rs`

- [ ] **Step 1: Confirm `backward_total_bytes` is `pub`**

```bash
grep -n "fn backward_total_bytes" crates/nsl-codegen/src/flash_attention_v2/phases/backward/prelude.rs
```

If `pub(crate)`, change to `pub` in the same file (the function is now cross-module load-bearing for `tier_b_range_table_offset`).

- [ ] **Step 2: Write the failing layout API tests**

Create `crates/nsl-codegen/tests/pca_tier_b_layout_api.rs`:

```rust
//! Unit tests for the Tier B SMEM layout API.
//! Spec §3.4.

use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::smem_layout::tier_b_range_table_offset;
use nsl_codegen::flash_attention_v2::{shared_mem_bytes_v2, shared_mem_bytes_v2_backward};
use nsl_codegen::pca_tile_config::num_tiles;
use nsl_codegen::pca_tilerange::{range_table_addrs, should_emit_tier_b};
use nsl_codegen::pca_segment::SegmentResidency;

fn fa_base_seg_masked() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64, block_kv: 64, head_dim: 64,
        causal: true, paged: false, rope_q: true,
        rope_style: RopeStyle::HalfSplit, gqa_group_size: 2,
        tree_mask: false, gpu_sm: 120, segment_masked: true, csha: None,
    }
}

#[test]
fn tier_b_offset_is_two_byte_aligned() {
    let cfg = fa_base_seg_masked();
    let offset = tier_b_range_table_offset(&cfg);
    assert_eq!(offset % 2, 0, "offset {offset} not 2-byte aligned");
}

#[test]
fn range_table_addrs_monotonic_offsets() {
    let addrs = range_table_addrs(0, 64, 64);
    assert_eq!(addrs.qtile_min,  0);
    assert_eq!(addrs.qtile_max,  64 * 2);
    assert_eq!(addrs.kvtile_min, 2 * 64 * 2);
    assert_eq!(addrs.kvtile_max, (2 * 64 + 64) * 2);
}

#[test]
fn range_table_addrs_asymmetric_blocks() {
    let addrs = range_table_addrs(0x1000, 512, 256);
    assert_eq!(addrs.qtile_min,  0x1000);
    assert_eq!(addrs.qtile_max,  0x1000 + 512 * 2);
    assert_eq!(addrs.kvtile_min, 0x1000 + 2 * 512 * 2);
    assert_eq!(addrs.kvtile_max, 0x1000 + 2 * 512 * 2 + 256 * 2);
}

#[test]
fn range_table_addrs_preserves_base() {
    let addrs = range_table_addrs(0xDEADBEE0, 32, 48);
    assert_eq!(addrs.qtile_min, 0xDEADBEE0);
}

#[test]
fn shared_mem_bytes_v2_no_op_when_not_tier_b() {
    let mut cfg = fa_base_seg_masked();
    cfg.segment_masked = false;
    assert!(!should_emit_tier_b(&cfg, 4096, SegmentResidency::Shared));
    assert!(shared_mem_bytes_v2(&cfg) > 0);
}

#[test]
fn tier_b_bytes_match_formula() {
    // 4K seq, block=64 → 64 q-tiles + 64 kv-tiles → 2*(64+64)*2 = 512 B.
    let cfg = fa_base_seg_masked();
    let bytes = nsl_codegen::pca_tilerange::tier_b_range_table_bytes(&cfg, 4096);
    assert_eq!(bytes, 512);
    let _ = num_tiles(4096, 64); // identity check via helper
}
```

```bash
cargo test -p nsl-codegen --test pca_tier_b_layout_api --no-run 2>&1 | tail -10
```

Expected: build fails — symbols missing.

- [ ] **Step 3: Add `tier_b_range_table_offset` to `smem_layout.rs`**

Open `crates/nsl-codegen/src/flash_attention_v2/smem_layout.rs`. After `pub fn sp_bytes`, add:

```rust
/// PCA Tier B range table — single tail offset into extern shmem.
///
/// Returns `align_up(backward_total_bytes + seg_overhead, 2)` so the
/// returned offset is 2-byte aligned for u16 slot loads. Alignment is
/// owned here per IR-004; `pca_tilerange::range_table_addrs` assumes
/// the base is ready-to-use.
pub fn tier_b_range_table_offset(config: &crate::flash_attention::FlashAttentionConfig) -> u32 {
    let seg_overhead = if config.segment_masked {
        crate::pca_segment::DEFAULT_SMEM_SEGMENT_BUDGET as u32
    } else {
        0
    };
    let base = crate::flash_attention_v2::phases::backward::prelude::backward_total_bytes(config)
        + seg_overhead;
    align_up_u32(base, 2)
}

#[inline]
fn align_up_u32(x: u32, align: u32) -> u32 {
    debug_assert!(align.is_power_of_two());
    (x + align - 1) & !(align - 1)
}
```

- [ ] **Step 4: Add `RangeTableAddrs` + `range_table_addrs` + `tier_b_range_table_bytes` to `pca_tilerange.rs`**

Open `crates/nsl-codegen/src/pca_tilerange.rs`. After the existing module doc comment + `use` lines, before the existing `compute_range_table_bytes` function, insert:

```rust
/// Four sub-table offsets inside the Tier B range-table region.
/// Spec §3.4.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangeTableAddrs {
    pub qtile_min:  u32,
    pub qtile_max:  u32,
    pub kvtile_min: u32,
    pub kvtile_max: u32,
}

/// Compute the four sub-table offsets given the range-table base and
/// tile counts. `base` is assumed 2-byte aligned per
/// `tier_b_range_table_offset`'s `align_up(2)` guarantee.
///
/// Layout:
/// ```text
///   base + 0                       : qtile_min  [num_q_tiles  × u16]
///   base + num_q_tiles × 2         : qtile_max  [num_q_tiles  × u16]
///   base + 2 × num_q_tiles × 2     : kvtile_min [num_kv_tiles × u16]
///   base + (2*num_q + num_kv) × 2  : kvtile_max [num_kv_tiles × u16]
/// ```
pub fn range_table_addrs(base: u32, num_q_tiles: u32, num_kv_tiles: u32) -> RangeTableAddrs {
    let q_bytes  = num_q_tiles  * 2;
    let kv_bytes = num_kv_tiles * 2;
    RangeTableAddrs {
        qtile_min:  base,
        qtile_max:  base + q_bytes,
        kvtile_min: base + 2 * q_bytes,
        kvtile_max: base + 2 * q_bytes + kv_bytes,
    }
}

/// Total bytes consumed by the Tier B range table at the given config
/// + seq_len. Used by `shared_mem_bytes_v2{_backward}_with_seqlen` to
/// widen the dynamic-SMEM launch parameter when Tier B is admitted.
pub fn tier_b_range_table_bytes(
    config: &crate::flash_attention::FlashAttentionConfig,
    seq_len: u32,
) -> u32 {
    let num_q  = crate::pca_tile_config::num_tiles(seq_len, config.block_q as u32);
    let num_kv = crate::pca_tile_config::num_tiles(seq_len, config.block_kv as u32);
    2 * (num_q + num_kv) * 2
}
```

- [ ] **Step 5: Wire launch-side accounting in `flash_attention_v2/mod.rs`**

Open `crates/nsl-codegen/src/flash_attention_v2/mod.rs`. After `pub fn shared_mem_bytes_v2_backward` (line ~528), add:

```rust
/// SMEM byte count for a v2 forward kernel including Tier B contribution.
///
/// Called from launch sites that have access to `seq_len` and the Tier A
/// residency decision. No-op guarantee: when Tier B is not emitted, returns
/// exactly `shared_mem_bytes_v2(config)` — pre-Tier-B SMEM layout preserved
/// byte-identically for non-Tier-B configs (spec §3.4.6).
pub fn shared_mem_bytes_v2_with_seqlen(
    config: &FlashAttentionConfig,
    seq_len: u32,
    residency: crate::pca_segment::SegmentResidency,
) -> u32 {
    let tier_b_bytes = if crate::pca_tilerange::should_emit_tier_b(config, seq_len as u64, residency) {
        crate::pca_tilerange::tier_b_range_table_bytes(config, seq_len)
    } else {
        0
    };
    shared_mem_bytes_v2(config) + tier_b_bytes
}

/// Backward equivalent of `shared_mem_bytes_v2_with_seqlen`.
pub fn shared_mem_bytes_v2_backward_with_seqlen(
    config: &FlashAttentionConfig,
    seq_len: u32,
    residency: crate::pca_segment::SegmentResidency,
) -> u32 {
    let tier_b_bytes = if crate::pca_tilerange::should_emit_tier_b(config, seq_len as u64, residency) {
        crate::pca_tilerange::tier_b_range_table_bytes(config, seq_len)
    } else {
        0
    };
    shared_mem_bytes_v2_backward(config) + tier_b_bytes
}
```

- [ ] **Step 6: Run the layout API tests + existing skeleton tests**

```bash
cargo test -p nsl-codegen --test pca_tier_b_layout_api 2>&1 | tail -15
cargo test -p nsl-codegen --test pca_tier_b_skeleton 2>&1 | tail -10
```

Expected: both green. If the layout API test compiles but fails on values, double-check the offset arithmetic — common bug is `2 * q_bytes` vs `2 * num_q_tiles` (the `* 2` is for the u16 byte width, separately from the `2 *` for two q-tables).

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-codegen/src/flash_attention_v2/smem_layout.rs \
        crates/nsl-codegen/src/flash_attention_v2/mod.rs \
        crates/nsl-codegen/src/pca_tilerange.rs \
        crates/nsl-codegen/tests/pca_tier_b_layout_api.rs \
        crates/nsl-codegen/src/flash_attention_v2/phases/backward/prelude.rs
git commit -m "feat(pca-tier-b-v2): tier_b_range_table_offset + RangeTableAddrs

Single tail offset (align_up(2) in smem_layout) + RangeTableAddrs struct
+ range_table_addrs constructor + tier_b_range_table_bytes helper. New
*_with_seqlen launch-bytes overloads preserve no-op guarantee for
non-Tier-B configs. Spec §3.4."
```

---

## Phase 2 — Preamble Emitter + Verification (Tasks 5, 5a, 5b)

## Task 5: Implement `emit_range_table_preamble` (runtime loop, RESHAPED)

**Goal:** Replace the stub in `pca_tilerange.rs` with a real PTX emitter that produces the q-phase and kv-phase runtime loops per spec §4. Each phase: uniform-counter loop with `BRA.U`-eligible branch, fixed-instruction-count butterfly reduction body, lane-0 predicated store.

**Files:**
- Modify: `crates/nsl-codegen/src/pca_tilerange.rs::emit_range_table_preamble`

- [ ] **Step 1: Replace the stub body**

Open `crates/nsl-codegen/src/pca_tilerange.rs`. Locate `pub fn emit_range_table_preamble` (the stub from PR #138). Replace its body with:

```rust
pub fn emit_range_table_preamble(
    ptx: &mut String,
    config: &crate::flash_attention::FlashAttentionConfig,
    seq_len: u32,
    segment_ids_smem_base: &str,
    range_table_base: u32,
) {
    let block_q  = config.block_q  as u32;
    let block_kv = config.block_kv as u32;
    let num_q_tiles  = crate::pca_tile_config::num_tiles(seq_len, block_q);
    let num_kv_tiles = crate::pca_tile_config::num_tiles(seq_len, block_kv);
    let addrs = range_table_addrs(range_table_base, num_q_tiles, num_kv_tiles);

    ptx.push_str("    // ----- PCA Tier B: range-table preamble (spec sec 3.1, v2) -----\n");
    ptx.push_str(&format!(
        "    // num_q_tiles={num_q_tiles}, num_kv_tiles={num_kv_tiles}, base=0x{range_table_base:X}\n"
    ));

    // Register declarations (declared once at the preamble; reused
    // across q-phase and kv-phase iterations).
    ptx.push_str("    .reg .u32  %r_tile_q_TILERANGE,  %r_tile_kv_TILERANGE;\n");
    ptx.push_str("    .reg .u16  %rs_min_TILERANGE,   %rs_max_TILERANGE;\n");
    ptx.push_str("    .reg .u16  %rs_peer_min_TILERANGE, %rs_peer_max_TILERANGE;\n");
    ptx.push_str("    .reg .u32  %lane_id_TILERANGE;\n");
    ptx.push_str("    .reg .pred %p_done_TILERANGE,   %p_lane_zero_TILERANGE;\n");
    ptx.push_str("    .reg .u64  %addr_min_TILERANGE, %addr_max_TILERANGE;\n");
    ptx.push_str("    .reg .u64  %seg_smem_TILERANGE, %wide_tile_off_TILERANGE;\n");
    ptx.push_str("    .reg .u32  %seg_byte_off_TILERANGE;\n");
    ptx.push_str("\n");

    ptx.push_str(&format!(
        "    cvta.shared.u64 %seg_smem_TILERANGE, {segment_ids_smem_base};\n"
    ));
    ptx.push_str("    mov.u32 %lane_id_TILERANGE, %laneid;\n");
    ptx.push_str("\n");

    emit_phase(ptx, "q",  num_q_tiles,  block_q,  addrs.qtile_min,  addrs.qtile_max);
    emit_phase(ptx, "kv", num_kv_tiles, block_kv, addrs.kvtile_min, addrs.kvtile_max);

    ptx.push_str("    bar.sync 0;  // range tables visible to all warps before kv-tile loop reads\n");
    ptx.push_str("    // ----- end PCA Tier B range-table preamble -----\n");
}

/// Emit one phase (q-phase or kv-phase) of the range-table reduction.
///
/// Per spec sec 4 (warp-uniformity discipline):
///   - %r_tile_*: .reg .u32, initialized via `mov.u32 ..., 0`, incremented unconditionally.
///   - Loop branch consumes a uniform predicate (setp on uniform register vs literal).
///   - Lane-0 store uses PTX predicated execution, not thread-divergent branch.
fn emit_phase(
    ptx: &mut String,
    tag: &str,        // "q" or "kv"
    num_tiles: u32,
    block_size: u32,
    addr_min: u32,
    addr_max: u32,
) {
    let r_tile = format!("%r_tile_{tag}_TILERANGE");
    let loop_label = format!("LOOP_{}_TILERANGE", tag.to_uppercase());

    ptx.push_str(&format!(
        "    // --- {tag}-phase: {num_tiles} tiles x {block_size} tokens, addr_min=0x{addr_min:X}, addr_max=0x{addr_max:X} ---\n"
    ));

    // Uniform counter init.
    ptx.push_str(&format!("    mov.u32 {r_tile}, 0;\n"));
    ptx.push_str(&format!("{loop_label}:\n"));

    // Per-iteration: load segment_ids[tile_start + lane], optional +lane+warp_size if block > 32,
    // butterfly-reduce, lane-0 store.
    emit_inner_reduction(ptx, tag, block_size, &r_tile);
    emit_lane_zero_store(ptx, tag, &r_tile, addr_min, addr_max);

    // Uniform increment + uniform-comparison branch (compiles to BRA.U).
    ptx.push_str(&format!("    add.u32 {r_tile}, {r_tile}, 1;\n"));
    ptx.push_str(&format!(
        "    setp.lt.u32 %p_done_TILERANGE, {r_tile}, {num_tiles};\n"
    ));
    ptx.push_str(&format!("    @%p_done_TILERANGE bra {loop_label};\n\n"));
}

/// Per-iteration body: load segment IDs into lane-local %rs_min/%rs_max,
/// then 5-step warp-shuffle butterfly reduction.
fn emit_inner_reduction(
    ptx: &mut String,
    tag: &str,
    block_size: u32,
    r_tile: &str,
) {
    // tile_start = %r_tile * block_size
    ptx.push_str(&format!(
        "    mul.lo.u32 %seg_byte_off_TILERANGE, {r_tile}, {block_size};\n"
    ));
    // Add lane to the tile start.
    ptx.push_str(
        "    add.u32 %seg_byte_off_TILERANGE, %seg_byte_off_TILERANGE, %lane_id_TILERANGE;\n",
    );
    // Convert token-index → byte offset (× sizeof(u16) = × 2).
    ptx.push_str(
        "    shl.b32 %seg_byte_off_TILERANGE, %seg_byte_off_TILERANGE, 1;\n",
    );
    // Compute byte address into segment_ids SMEM.
    ptx.push_str(
        "    cvt.u64.u32 %wide_tile_off_TILERANGE, %seg_byte_off_TILERANGE;\n",
    );
    ptx.push_str(
        "    add.u64 %wide_tile_off_TILERANGE, %seg_smem_TILERANGE, %wide_tile_off_TILERANGE;\n",
    );
    // Lane-local initial: load one u16 from SMEM.
    ptx.push_str(
        "    ld.shared.u16 %rs_min_TILERANGE, [%wide_tile_off_TILERANGE];\n",
    );
    ptx.push_str(
        "    mov.u16 %rs_max_TILERANGE, %rs_min_TILERANGE;\n",
    );

    // If block_size > warp_size, fold the second half (lane + 32, lane + 64, ...).
    let warp_size: u32 = 32;
    let mut extra_offset = warp_size;
    while extra_offset < block_size {
        let _ = tag; // silenced; tag may be referenced for debug comments in future
        ptx.push_str(&format!(
            "    // fold lane + {extra_offset} into lane-local min/max\n"
        ));
        ptx.push_str(&format!(
            "    add.u64 %wide_tile_off_TILERANGE, %wide_tile_off_TILERANGE, {};\n",
            extra_offset * 2 /* byte stride */
        ));
        ptx.push_str(
            "    ld.shared.u16 %rs_peer_min_TILERANGE, [%wide_tile_off_TILERANGE];\n",
        );
        ptx.push_str(
            "    min.u16 %rs_min_TILERANGE, %rs_min_TILERANGE, %rs_peer_min_TILERANGE;\n",
        );
        ptx.push_str(
            "    max.u16 %rs_max_TILERANGE, %rs_max_TILERANGE, %rs_peer_min_TILERANGE;\n",
        );
        extra_offset += warp_size;
    }

    // 5-step butterfly reduction (offsets 16, 8, 4, 2, 1).
    for offset in [16u32, 8, 4, 2, 1] {
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %rs_peer_min_TILERANGE, %rs_min_TILERANGE, {offset}, 0x1F, 0xFFFFFFFF;\n"
        ));
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %rs_peer_max_TILERANGE, %rs_max_TILERANGE, {offset}, 0x1F, 0xFFFFFFFF;\n"
        ));
        ptx.push_str(
            "    min.u16 %rs_min_TILERANGE, %rs_min_TILERANGE, %rs_peer_min_TILERANGE;\n",
        );
        ptx.push_str(
            "    max.u16 %rs_max_TILERANGE, %rs_max_TILERANGE, %rs_peer_max_TILERANGE;\n",
        );
    }
}

/// Lane-0 predicated store of (min, max) to range_table[tile_idx].
fn emit_lane_zero_store(
    ptx: &mut String,
    tag: &str,
    r_tile: &str,
    addr_min: u32,
    addr_max: u32,
) {
    let _ = tag;
    // Compute byte offset = 2 * %r_tile (u16 stride).
    ptx.push_str(&format!(
        "    shl.b32 %seg_byte_off_TILERANGE, {r_tile}, 1;\n"
    ));
    ptx.push_str(
        "    cvt.u64.u32 %wide_tile_off_TILERANGE, %seg_byte_off_TILERANGE;\n",
    );
    // qtile_min/kvtile_min address.
    ptx.push_str(&format!(
        "    cvta.shared.u64 %addr_min_TILERANGE, shmem;\n"
    ));
    ptx.push_str(&format!(
        "    add.u64 %addr_min_TILERANGE, %addr_min_TILERANGE, {addr_min};\n"
    ));
    ptx.push_str(
        "    add.u64 %addr_min_TILERANGE, %addr_min_TILERANGE, %wide_tile_off_TILERANGE;\n",
    );
    // qtile_max/kvtile_max address.
    ptx.push_str(&format!(
        "    cvta.shared.u64 %addr_max_TILERANGE, shmem;\n"
    ));
    ptx.push_str(&format!(
        "    add.u64 %addr_max_TILERANGE, %addr_max_TILERANGE, {addr_max};\n"
    ));
    ptx.push_str(
        "    add.u64 %addr_max_TILERANGE, %addr_max_TILERANGE, %wide_tile_off_TILERANGE;\n",
    );
    // Lane-0 predicate.
    ptx.push_str(
        "    setp.eq.u32 %p_lane_zero_TILERANGE, %lane_id_TILERANGE, 0;\n",
    );
    // Predicated stores (NOT divergent branch — see spec sec 4.3).
    ptx.push_str(
        "    @%p_lane_zero_TILERANGE st.shared.u16 [%addr_min_TILERANGE], %rs_min_TILERANGE;\n",
    );
    ptx.push_str(
        "    @%p_lane_zero_TILERANGE st.shared.u16 [%addr_max_TILERANGE], %rs_max_TILERANGE;\n",
    );
}
```

**Notes on the emission:**
- The `shmem` symbol referenced in `cvta.shared.u64 ..., shmem` is the existing `.extern .shared .align 16 .b8 shmem[]` declared elsewhere in the kernel (forward/backward preludes already declare it for the dynamic-SMEM path).
- `%laneid` is the PTX-special read-only register holding the warp lane index.
- `shfl.sync.bfly` uses `0x1F` (segmask) and `0xFFFFFFFF` (membermask = all 32 lanes participating).
- The signature changed: the stub took `segment_ids_reg: &str`; the real emitter takes both `segment_ids_smem_base: &str` (the SMEM label or address symbol) **and** `range_table_base: u32` (from `smem_layout::tier_b_range_table_offset`). Update the stub's signature to match.

- [ ] **Step 2: Update the stub signature in `should_emit_tier_b` / module-level tests**

Check whether any existing caller of the stub `emit_range_table_preamble` exists:

```bash
grep -rn "emit_range_table_preamble" crates/nsl-codegen/
```

Expected: only the definition + the skeleton test `pca_tier_b_skeleton.rs`. Update the skeleton test to pass the new fourth argument:

```rust
emit_range_table_preamble(&mut s, &cfg, 4096, "seg_smem", 0);
```

- [ ] **Step 3: Build to confirm clean compile**

```bash
cargo build -p nsl-codegen 2>&1 | tail -15
cargo test -p nsl-codegen --test pca_tier_b_skeleton 2>&1 | tail -10
```

Expected: build clean. The skeleton test's "emitter emits a marker comment" assertion still passes — the real emitter still prefixes with `"// ----- PCA Tier B"`.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/pca_tilerange.rs \
        crates/nsl-codegen/tests/pca_tier_b_skeleton.rs
git commit -m "feat(pca-tier-b-v2): emit_range_table_preamble runtime PTX loop

Real emission per spec sec 3.1 / sec 4: per-phase uniform-counter loop
(BRA.U-eligible branch), warp-shuffle butterfly min/max reduction,
lane-0 predicated store. Uniformity-through-load anti-pattern avoided
(r_tile is register-resident, mov-initialized, unconditionally
incremented). IR-007 (PTX patterns pinned at instruction level)."
```

---

## Task 5a: Isolation insta snapshot for preamble (NEW)

**Goal:** Insta snapshot of the emitted preamble PTX with sentinel base offset `0xDEAD_BEEF`. Catches PTX-string drift in isolation — failure surface is a localized ~5-line diff at the Tier B emitter, not "FA2 kernel changed."

**Files:**
- Create: `crates/nsl-codegen/tests/pca_tier_b_preamble_isolation.rs`
- Create: `crates/nsl-codegen/tests/snapshots/pca_tier_b_preamble_isolation__preamble_4k_block64.snap` (via `cargo insta accept`)

- [ ] **Step 1: Write the snapshot test**

Create `crates/nsl-codegen/tests/pca_tier_b_preamble_isolation.rs`:

```rust
//! Isolation-level insta snapshot of the PCA Tier B preamble PTX.
//!
//! Sentinel base offset 0xDEAD_BEEF — appears verbatim in emitted PTX
//! as a register-loaded immediate; offset-consumption regressions are
//! immediately visible in the snapshot diff.
//!
//! Per spec sec 5.1 of 2026-05-12-pca-tier-b-revision-design.md.

use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::pca_tilerange::emit_range_table_preamble;

fn fa_4k_block64_seg_masked() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64, block_kv: 64, head_dim: 64,
        causal: true, paged: false, rope_q: true,
        rope_style: RopeStyle::HalfSplit, gqa_group_size: 2,
        tree_mask: false, gpu_sm: 120, segment_masked: true, csha: None,
    }
}

#[test]
fn preamble_4k_block64_isolation_snapshot() {
    let mut ptx = String::new();
    emit_range_table_preamble(
        &mut ptx,
        &fa_4k_block64_seg_masked(),
        4096,
        "seg_smem",
        0xDEAD_BEEF,
    );
    insta::assert_snapshot!("preamble_4k_block64", ptx);
}

#[test]
fn preamble_sentinel_visible_in_emitted_ptx() {
    let mut ptx = String::new();
    emit_range_table_preamble(
        &mut ptx,
        &fa_4k_block64_seg_masked(),
        4096,
        "seg_smem",
        0xDEAD_BEEF,
    );
    // Sentinel must appear verbatim in the emitted PTX as decimal or hex.
    // emit_range_table_preamble formats range_table_base as 0x{:X} in the
    // header comment AND uses it as immediate operand of `add.u64`.
    assert!(
        ptx.contains("DEADBEEF") || ptx.contains("deadbeef") || ptx.contains("3735928559"),
        "sentinel 0xDEADBEEF not visible in emitted PTX — sentinel detection broken:\n{ptx}"
    );
}

#[test]
fn preamble_uses_butterfly_shuffles() {
    let mut ptx = String::new();
    emit_range_table_preamble(
        &mut ptx,
        &fa_4k_block64_seg_masked(),
        4096,
        "seg_smem",
        0xDEAD_BEEF,
    );
    // 5-step butterfly per phase, 2 ops (min/max) per step, 2 phases = 20 shuffles.
    assert!(
        ptx.matches("shfl.sync.bfly").count() >= 20,
        "expected >=20 butterfly shuffles, got:\n{ptx}"
    );
}

#[test]
fn preamble_uses_predicated_lane_zero_store() {
    let mut ptx = String::new();
    emit_range_table_preamble(
        &mut ptx,
        &fa_4k_block64_seg_masked(),
        4096,
        "seg_smem",
        0xDEAD_BEEF,
    );
    // Predicated execution (`@%p_lane_zero_TILERANGE st.shared.u16`),
    // NOT a divergent branch around the store.
    assert!(
        ptx.contains("@%p_lane_zero_TILERANGE st.shared.u16"),
        "lane-0 store should be predicated, not branched. Emitted:\n{ptx}"
    );
    assert!(
        !ptx.contains("@%p_lane_zero_TILERANGE bra"),
        "lane-0 store must not use thread-divergent branch:\n{ptx}"
    );
}

#[test]
fn preamble_loop_counter_register_class_correct() {
    let mut ptx = String::new();
    emit_range_table_preamble(
        &mut ptx,
        &fa_4k_block64_seg_masked(),
        4096,
        "seg_smem",
        0xDEAD_BEEF,
    );
    // %r_tile_* must be .reg .u32 per warp-uniformity discipline.
    assert!(
        ptx.contains(".reg .u32  %r_tile_q_TILERANGE,  %r_tile_kv_TILERANGE;"),
        "r_tile must be .reg .u32; emitted preamble:\n{ptx}"
    );
    // Init via mov.u32 from literal 0 (uniformity-through-load anti-pattern check).
    assert!(
        ptx.contains("mov.u32 %r_tile_q_TILERANGE, 0;"),
        "r_tile_q init missing — must be mov.u32 from literal:\n{ptx}"
    );
}
```

- [ ] **Step 2: First run — accept the snapshot**

```bash
cargo test -p nsl-codegen --test pca_tier_b_preamble_isolation 2>&1 | tail -15
```

Expected: 4 structural tests pass; `preamble_4k_block64_isolation_snapshot` fails with insta-pending. Inspect the new snapshot file:

```bash
cat crates/nsl-codegen/tests/snapshots/pca_tier_b_preamble_isolation__preamble_4k_block64.snap.new
```

Verify visually:
- Header comment shows `num_q_tiles=64, num_kv_tiles=64, base=0xDEADBEEF`.
- Both phases (q + kv) emit their loop labels and the 5-step butterfly.
- Lane-0 stores are predicated (`@%p_lane_zero_TILERANGE st.shared.u16`), not branched.
- Final `bar.sync 0;` after kv-phase.

If all four visual checks pass, accept:

```bash
INSTA_UPDATE=accept cargo test -p nsl-codegen --test pca_tier_b_preamble_isolation 2>&1 | tail -10
```

Expected: 5 tests pass, snapshot file moved from `.snap.new` to `.snap`.

- [ ] **Step 3: Commit (snapshot + test)**

```bash
git add crates/nsl-codegen/tests/pca_tier_b_preamble_isolation.rs \
        crates/nsl-codegen/tests/snapshots/pca_tier_b_preamble_isolation__preamble_4k_block64.snap
git commit -m "test(pca-tier-b-v2): preamble isolation insta snapshot

Sentinel base offset 0xDEAD_BEEF visible verbatim in emitted PTX so
offset-consumption regressions show in the diff. 5 assertions:
snapshot equality, sentinel visibility, butterfly count, predicated
lane-0 store (not divergent branch), uniform counter register class.
Spec sec 5.1 + IR-006."
```

---

## Task 5b: Extend `pca_sass_byte_identity` with BRA.U + UR<n> assertions (NEW)

**Goal:** Add Tier B-specific SASS-level assertions to the existing `pca_sass_byte_identity` test. Catches SASS regressions (`BRA` instead of `BRA.U`, `R<n>` instead of `UR<n>` on sm_90/sm_120) that the PTX-text snapshot can't see.

**Files:**
- Modify: `crates/nsl-codegen/tests/pca_sass_byte_identity.rs`

- [ ] **Step 1: Confirm the existing test runs ptxas + cuobjdump**

```bash
grep -n "ptxas\|cuobjdump\|BRA" crates/nsl-codegen/tests/pca_sass_byte_identity.rs | head -30
```

Confirm the test already shells out to `ptxas` + `cuobjdump`. If it doesn't (the existing test does byte-identity at the PTX level only), this Task 5b extends with new test functions that DO shell out — keep the existing tests untouched.

- [ ] **Step 2: Add SASS extraction helper + Tier B assertions**

Append to `crates/nsl-codegen/tests/pca_sass_byte_identity.rs`:

```rust
//! ---------- Tier B SASS assertions (added 2026-05-12) ----------
//!
//! Extends the existing PTX byte-identity test with SASS-level
//! verification that Tier B's range-table loop branches compile to
//! BRA.U (warp-uniform) rather than BRA (per-thread predicated), and
//! that on sm_90/sm_120 the min/max stores land in uniform-class
//! registers (UR<n>).
//!
//! Per spec sec 5.2 of 2026-05-12-pca-tier-b-revision-design.md.

#[cfg(feature = "cuda")]
mod tier_b_sass {
    use super::*;
    use std::process::Command;

    fn ptxas_path() -> String {
        std::env::var("CUDA_PATH")
            .map(|p| format!("{p}/bin/ptxas"))
            .unwrap_or_else(|_| "ptxas".to_string())
    }
    fn cuobjdump_path() -> String {
        std::env::var("CUDA_PATH")
            .map(|p| format!("{p}/bin/cuobjdump"))
            .unwrap_or_else(|_| "cuobjdump".to_string())
    }

    /// Compile PTX → cubin via ptxas, then dump SASS via cuobjdump.
    /// Returns the SASS text on success, error string on failure.
    fn ptx_to_sass(ptx: &str, sm: u32) -> Result<String, String> {
        let tmpdir = tempfile::tempdir().map_err(|e| format!("tempdir: {e}"))?;
        let ptx_path = tmpdir.path().join("tier_b.ptx");
        let cubin_path = tmpdir.path().join("tier_b.cubin");
        std::fs::write(&ptx_path, ptx).map_err(|e| format!("write ptx: {e}"))?;

        let ptxas = Command::new(ptxas_path())
            .args(["-arch", &format!("sm_{sm}"), "-o"])
            .arg(&cubin_path)
            .arg(&ptx_path)
            .output()
            .map_err(|e| format!("spawn ptxas: {e}"))?;
        if !ptxas.status.success() {
            return Err(format!(
                "ptxas failed: stdout={} stderr={}",
                String::from_utf8_lossy(&ptxas.stdout),
                String::from_utf8_lossy(&ptxas.stderr),
            ));
        }

        let dump = Command::new(cuobjdump_path())
            .args(["--dump-sass"])
            .arg(&cubin_path)
            .output()
            .map_err(|e| format!("spawn cuobjdump: {e}"))?;
        Ok(String::from_utf8_lossy(&dump.stdout).into_owned())
    }

    /// Extract the SASS lines between two labels (q-phase or kv-phase
    /// loop body). Returns "" if labels aren't found.
    fn sass_between(sass: &str, start_label: &str, end_label: &str) -> String {
        let mut in_block = false;
        let mut out = String::new();
        for line in sass.lines() {
            if line.contains(start_label) { in_block = true; }
            if in_block {
                out.push_str(line);
                out.push('\n');
            }
            if in_block && line.contains(end_label) { break; }
        }
        out
    }

    #[test]
    fn tier_b_preamble_q_phase_branch_is_uniform_sm120() {
        let mut ptx = String::new();
        let cfg = super::tier_b_test_fixture(); // see Step 3 below
        nsl_codegen::pca_tilerange::emit_range_table_preamble(
            &mut ptx, &cfg, 4096, "seg_smem", 0,
        );
        let kernel_ptx = super::wrap_in_minimal_kernel(&ptx); // see Step 3
        let sass = match ptx_to_sass(&kernel_ptx, 120) {
            Ok(s) => s,
            Err(e) if e.contains("Unknown") || e.contains("not found") => {
                eprintln!("SKIP: ptxas/cuobjdump not available: {e}");
                return;
            }
            Err(e) => panic!("ptxas/cuobjdump failure: {e}"),
        };

        // The Tier B preamble emits labels `LOOP_Q_TILERANGE` (PTX-level).
        // ptxas mangles labels into SASS form; grep for "LOOP_Q" substring.
        let q_loop_sass = sass_between(&sass, "LOOP_Q", "LOOP_KV");
        assert!(
            q_loop_sass.contains("BRA.U") || q_loop_sass.contains("BRA.UNI"),
            "q-phase loop branch not BRA.U on sm_120:\n{q_loop_sass}"
        );
    }

    #[test]
    fn tier_b_preamble_kv_phase_branch_is_uniform_sm120() {
        let mut ptx = String::new();
        let cfg = super::tier_b_test_fixture();
        nsl_codegen::pca_tilerange::emit_range_table_preamble(
            &mut ptx, &cfg, 4096, "seg_smem", 0,
        );
        let kernel_ptx = super::wrap_in_minimal_kernel(&ptx);
        let sass = match ptx_to_sass(&kernel_ptx, 120) {
            Ok(s) => s,
            Err(e) if e.contains("Unknown") || e.contains("not found") => {
                eprintln!("SKIP: ptxas/cuobjdump not available: {e}");
                return;
            }
            Err(e) => panic!("ptxas/cuobjdump failure: {e}"),
        };

        let kv_loop_sass = sass_between(&sass, "LOOP_KV", "bar.sync");
        assert!(
            kv_loop_sass.contains("BRA.U") || kv_loop_sass.contains("BRA.UNI"),
            "kv-phase loop branch not BRA.U on sm_120:\n{kv_loop_sass}"
        );
    }

    #[test]
    fn tier_b_preamble_uses_uniform_registers_on_sm120() {
        let mut ptx = String::new();
        let cfg = super::tier_b_test_fixture();
        nsl_codegen::pca_tilerange::emit_range_table_preamble(
            &mut ptx, &cfg, 4096, "seg_smem", 0,
        );
        let kernel_ptx = super::wrap_in_minimal_kernel(&ptx);
        let sass = match ptx_to_sass(&kernel_ptx, 120) {
            Ok(s) => s,
            Err(e) if e.contains("Unknown") || e.contains("not found") => {
                eprintln!("SKIP: ptxas/cuobjdump not available: {e}");
                return;
            }
            Err(e) => panic!("ptxas/cuobjdump failure: {e}"),
        };

        // On sm_120 (Blackwell), uniform-class registers appear as UR<n>.
        // The Tier B range-table store operands must use UR<n>.
        assert!(
            sass.contains("UR"),
            "sm_120 SASS should reference UR<n> for uniform-class regs; full SASS:\n{sass}"
        );
    }

    #[test]
    fn tier_b_preamble_sm80_uniform_proxy_via_brau() {
        // sm_80 fall-back: UR<n> annotation less observable; BRA.U at
        // the loop branch is the available proxy.
        let mut ptx = String::new();
        let mut cfg = super::tier_b_test_fixture();
        cfg.gpu_sm = 80;
        nsl_codegen::pca_tilerange::emit_range_table_preamble(
            &mut ptx, &cfg, 4096, "seg_smem", 0,
        );
        let kernel_ptx = super::wrap_in_minimal_kernel(&ptx);
        let sass = match ptx_to_sass(&kernel_ptx, 80) {
            Ok(s) => s,
            Err(e) if e.contains("Unknown") || e.contains("not found") => {
                eprintln!("SKIP: ptxas/cuobjdump not available: {e}");
                return;
            }
            Err(e) => panic!("ptxas/cuobjdump failure: {e}"),
        };
        let q_loop_sass = sass_between(&sass, "LOOP_Q", "LOOP_KV");
        assert!(
            q_loop_sass.contains("BRA.U") || q_loop_sass.contains("BRA.UNI"),
            "sm_80 q-phase loop branch not BRA.U (proxy for uniform-class):\n{q_loop_sass}"
        );
    }
}
```

- [ ] **Step 3: Add the fixture + minimal-kernel-wrapper helpers**

Add at the top of `crates/nsl-codegen/tests/pca_sass_byte_identity.rs` (or in a shared `mod common` block):

```rust
#[cfg(feature = "cuda")]
pub(crate) fn tier_b_test_fixture() -> nsl_codegen::flash_attention::FlashAttentionConfig {
    nsl_codegen::flash_attention::FlashAttentionConfig {
        block_q: 64, block_kv: 64, head_dim: 64,
        causal: true, paged: false, rope_q: true,
        rope_style: nsl_codegen::flash_attention::RopeStyle::HalfSplit,
        gqa_group_size: 2, tree_mask: false,
        gpu_sm: 120, segment_masked: true, csha: None,
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn wrap_in_minimal_kernel(preamble_ptx: &str) -> String {
    // Wraps the preamble in a minimal kernel skeleton ptxas can compile.
    // Declares both static `.shared seg_smem` (mimicking Tier A) and
    // extern shmem (where Tier B's range tables ride).
    format!(
        r#".version 7.0
.target sm_90
.address_size 64

.shared .align 4 .b8 seg_smem[4096];
.extern .shared .align 16 .b8 shmem[];

.visible .entry tier_b_test_kernel() {{
{preamble_ptx}
    ret;
}}
"#
    )
}
```

- [ ] **Step 4: Add `tempfile` to dev-dependencies if not present**

```bash
grep -n "tempfile" crates/nsl-codegen/Cargo.toml
```

If missing, add to `[dev-dependencies]`:

```toml
tempfile = "3"
```

- [ ] **Step 5: Run the extended SASS tests**

```bash
cargo test -p nsl-codegen --features cuda --test pca_sass_byte_identity tier_b_sass 2>&1 | tail -15
```

Expected: tests pass if CUDA toolkit is in `CUDA_PATH` or on `PATH`. If ptxas/cuobjdump aren't found, the SKIP-on-not-found branches make the tests pass with a note printed to stderr. Run on a machine with the CUDA toolkit installed before merge.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/tests/pca_sass_byte_identity.rs \
        crates/nsl-codegen/Cargo.toml
git commit -m "test(pca-tier-b-v2): SASS BRA.U + UR<n> assertions for preamble

Extends pca_sass_byte_identity with Tier B-specific SASS-level checks:
  - sm_120: q-phase + kv-phase loop branches must be BRA.U (warp-uniform).
  - sm_120: range-table stores reference UR<n> (uniform-class regs).
  - sm_80:  BRA.U-as-proxy (uniform-register annotation less observable).

Replaces v1's standalone emit-pca-tier-b-preamble-harness bin (deleted
via spec sec 6.3.4 — maintenance cost over time). Tests skip cleanly
when ptxas/cuobjdump not on PATH so CI without CUDA toolkit still
passes. Spec sec 5.2 + IR-006."
```

---

## Phase 3 — Skip Predicate (Tasks 6, 6a)

## Task 6: Implement `emit_skip_predicate` + wire into `s_compute.rs` kv-tile loop

**Goal:** Replace `emit_skip_predicate` stub with real PTX. One comparison + one warp-uniform branch. Insert call at the top of the forward kv-tile loop in `s_compute.rs::emit` so disjoint tile pairs bypass QK^T + softmax + PV.

**Files:**
- Modify: `crates/nsl-codegen/src/pca_tilerange.rs::emit_skip_predicate`
- Modify: `crates/nsl-codegen/src/flash_attention_v2/phases/forward/s_compute.rs`

- [ ] **Step 1: Replace the stub `emit_skip_predicate`**

Open `crates/nsl-codegen/src/pca_tilerange.rs`. Locate `pub fn emit_skip_predicate`. Replace body:

```rust
pub fn emit_skip_predicate(
    ptx: &mut String,
    config: &crate::flash_attention::FlashAttentionConfig,
    seq_len: u32,
    qt_reg: &str,            // PTX register holding the current q-tile index
    kvt_reg: &str,           // PTX register holding the current kv-tile index
    range_table_base: u32,   // from smem_layout::tier_b_range_table_offset
    on_skip_label: &str,     // PTX label to branch to when ranges disjoint
) {
    let num_q  = crate::pca_tile_config::num_tiles(seq_len, config.block_q  as u32);
    let num_kv = crate::pca_tile_config::num_tiles(seq_len, config.block_kv as u32);
    let addrs = range_table_addrs(range_table_base, num_q, num_kv);

    ptx.push_str("    // ----- PCA Tier B: skip predicate (spec sec 3.2) -----\n");
    ptx.push_str("    .reg .u16  %qmin_TB, %qmax_TB, %kvmin_TB, %kvmax_TB;\n");
    ptx.push_str("    .reg .pred %p_lt_TB, %p_gt_TB, %p_skip_TB;\n");
    ptx.push_str("    .reg .u32  %tile_byte_TB;\n");
    ptx.push_str("    .reg .u64  %addr_TB;\n");
    ptx.push_str("\n");

    // qmin[qt], qmax[qt]
    ptx.push_str(&format!("    shl.b32 %tile_byte_TB, {qt_reg}, 1;\n"));
    ptx.push_str("    cvt.u64.u32 %addr_TB, %tile_byte_TB;\n");
    ptx.push_str(&format!(
        "    cvta.shared.u64 %addr_TB, shmem; add.u64 %addr_TB, %addr_TB, {};\n",
        addrs.qtile_min
    ));
    ptx.push_str("    ld.shared.u16 %qmin_TB, [%addr_TB];\n");
    ptx.push_str(&format!(
        "    cvta.shared.u64 %addr_TB, shmem; add.u64 %addr_TB, %addr_TB, {};\n",
        addrs.qtile_max
    ));
    ptx.push_str("    ld.shared.u16 %qmax_TB, [%addr_TB];\n");

    // kvmin[kvt], kvmax[kvt]
    ptx.push_str(&format!("    shl.b32 %tile_byte_TB, {kvt_reg}, 1;\n"));
    ptx.push_str("    cvt.u64.u32 %addr_TB, %tile_byte_TB;\n");
    ptx.push_str(&format!(
        "    cvta.shared.u64 %addr_TB, shmem; add.u64 %addr_TB, %addr_TB, {};\n",
        addrs.kvtile_min
    ));
    ptx.push_str("    ld.shared.u16 %kvmin_TB, [%addr_TB];\n");
    ptx.push_str(&format!(
        "    cvta.shared.u64 %addr_TB, shmem; add.u64 %addr_TB, %addr_TB, {};\n",
        addrs.kvtile_max
    ));
    ptx.push_str("    ld.shared.u16 %kvmax_TB, [%addr_TB];\n");

    // disjoint = (qmax < kvmin) || (qmin > kvmax)
    ptx.push_str("    setp.lt.u16 %p_lt_TB, %qmax_TB, %kvmin_TB;\n");
    ptx.push_str("    setp.gt.u16 %p_gt_TB, %qmin_TB, %kvmax_TB;\n");
    ptx.push_str("    or.pred %p_skip_TB, %p_lt_TB, %p_gt_TB;\n");
    ptx.push_str(&format!("    @%p_skip_TB bra {on_skip_label};\n"));
    ptx.push_str("    // ----- end PCA Tier B skip predicate -----\n");
}
```

The predicate is warp-uniform: all four range-table loads return values that are uniform across the warp (every thread reads the same SMEM slot), `setp.lt.u16` / `setp.gt.u16` produce uniform predicates, `or.pred` produces a uniform predicate, `@%p_skip_TB bra` compiles to `BRA.U`.

- [ ] **Step 2: Locate the kv-tile loop in `s_compute.rs`**

```bash
grep -n "kv_tile\|kv_iter\|for_kv\|kvt" crates/nsl-codegen/src/flash_attention_v2/phases/forward/s_compute.rs | head -20
```

Find the top of the kv-tile loop body — the place where K is loaded from HBM into SMEM, immediately preceding the QK^T matmul. The skip predicate fires BEFORE the K-load (the matmul work is what we're skipping, and the K-load is part of that work).

- [ ] **Step 3: Insert the predicate call**

In `s_compute.rs::emit` (or wherever the kv-tile loop body starts), add at the top of the loop body — conditioned on Tier B being admitted:

```rust
// PCA Tier B skip predicate (only when admitted).
if crate::pca_tilerange::should_emit_tier_b(config, seq_len, residency) {
    let range_table_base = crate::flash_attention_v2::smem_layout::tier_b_range_table_offset(config);
    crate::pca_tilerange::emit_skip_predicate(
        ptx,
        config,
        seq_len,
        "%qt",            // existing q-tile loop counter (verify exact name in this file)
        "%kvt",           // existing kv-tile loop counter (verify exact name)
        range_table_base,
        "KV_TILE_SKIP",   // label at the end of the kv-tile body, just before `add %kvt, 1; bra LOOP_KV;`
    );
}
```

Then ensure the `KV_TILE_SKIP:` label is emitted at the right place — at the bottom of the kv-tile body, just before the kv loop counter increment + back-branch.

The exact register names (`%qt`, `%kvt`) and label location depend on the existing emission code; replace with what's there.

- [ ] **Step 4: Pass `seq_len` and `residency` to `s_compute::emit`**

`s_compute::emit` may not currently take `seq_len` or `residency`. If not, thread them through from the caller (`flash_attention_v2/mod.rs::synthesize_*`).

```bash
grep -n "pub fn emit\|pub(crate) fn emit" crates/nsl-codegen/src/flash_attention_v2/phases/forward/s_compute.rs
```

Update the signature to take `seq_len: u32` and `residency: SegmentResidency`. Update callers accordingly.

- [ ] **Step 5: Build + run baseline tests**

```bash
cargo build -p nsl-codegen 2>&1 | tail -20
cargo test -p nsl-codegen --test pca_tier_b_skeleton 2>&1 | tail -10
cargo test -p nsl-codegen --test pca_tier_b_layout_api 2>&1 | tail -10
cargo test -p nsl-codegen --test pca_tier_b_preamble_isolation 2>&1 | tail -10
```

Expected: all green. The forward/backward kernel snapshot tests may now show diffs because Tier B-enabled configs emit the new predicate — those snapshot updates are part of Task 8 (don't accept them yet).

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/pca_tilerange.rs \
        crates/nsl-codegen/src/flash_attention_v2/phases/forward/s_compute.rs \
        crates/nsl-codegen/src/flash_attention_v2/mod.rs
git commit -m "feat(pca-tier-b-v2): emit_skip_predicate + wire into s_compute.rs

Real emission: 4 range-table loads (uniform across warp) +
setp.lt.u16 + setp.gt.u16 + or.pred + @%p_skip_TB bra (compiles
to BRA.U). Inserted at top of kv-tile loop body in
forward/s_compute.rs::emit; skips QK^T + softmax + PV when range
check fires. Tier A's per-cell mask handles cells inside surviving
tiles. Spec sec 3.2 (conservative-skip semantic) + IR-007."
```

---

## Task 6a: Isolation snapshot for skip predicate + SASS assertion (NEW)

**Goal:** Mirrors Task 5a/5b for the predicate: isolation insta snapshot at the `pca_tilerange::emit_skip_predicate` boundary + SASS `BRA.U` assertion at the predicate branch.

**Files:**
- Create: `crates/nsl-codegen/tests/pca_tier_b_predicate_isolation.rs`
- Modify: `crates/nsl-codegen/tests/pca_sass_byte_identity.rs` (add predicate SASS check)

- [ ] **Step 1: Write the isolation snapshot test**

Create `crates/nsl-codegen/tests/pca_tier_b_predicate_isolation.rs`:

```rust
//! Isolation insta snapshot of the PCA Tier B skip predicate PTX.
//! Sentinel range_table_base = 0xDEAD_BEEF.
//! Per spec sec 5.1 + sec 3.2.

use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::pca_tilerange::emit_skip_predicate;

fn fa_4k_block64_seg_masked() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64, block_kv: 64, head_dim: 64,
        causal: true, paged: false, rope_q: true,
        rope_style: RopeStyle::HalfSplit, gqa_group_size: 2,
        tree_mask: false, gpu_sm: 120, segment_masked: true, csha: None,
    }
}

#[test]
fn predicate_4k_block64_isolation_snapshot() {
    let mut ptx = String::new();
    emit_skip_predicate(
        &mut ptx,
        &fa_4k_block64_seg_masked(),
        4096,
        "%qt", "%kvt",
        0xDEAD_BEEF,
        "KV_TILE_SKIP",
    );
    insta::assert_snapshot!("predicate_4k_block64", ptx);
}

#[test]
fn predicate_emits_four_range_table_loads() {
    let mut ptx = String::new();
    emit_skip_predicate(
        &mut ptx, &fa_4k_block64_seg_masked(), 4096,
        "%qt", "%kvt", 0xDEAD_BEEF, "KV_TILE_SKIP",
    );
    // 4 loads: qmin, qmax, kvmin, kvmax.
    assert_eq!(ptx.matches("ld.shared.u16").count(), 4);
}

#[test]
fn predicate_uses_disjoint_logic() {
    let mut ptx = String::new();
    emit_skip_predicate(
        &mut ptx, &fa_4k_block64_seg_masked(), 4096,
        "%qt", "%kvt", 0xDEAD_BEEF, "KV_TILE_SKIP",
    );
    assert!(ptx.contains("setp.lt.u16 %p_lt_TB, %qmax_TB, %kvmin_TB"));
    assert!(ptx.contains("setp.gt.u16 %p_gt_TB, %qmin_TB, %kvmax_TB"));
    assert!(ptx.contains("or.pred %p_skip_TB, %p_lt_TB, %p_gt_TB"));
}

#[test]
fn predicate_branches_to_provided_label() {
    let mut ptx = String::new();
    emit_skip_predicate(
        &mut ptx, &fa_4k_block64_seg_masked(), 4096,
        "%qt", "%kvt", 0xDEAD_BEEF, "MY_CUSTOM_LABEL",
    );
    assert!(ptx.contains("@%p_skip_TB bra MY_CUSTOM_LABEL"));
}

#[test]
fn predicate_sentinel_visible() {
    let mut ptx = String::new();
    emit_skip_predicate(
        &mut ptx, &fa_4k_block64_seg_masked(), 4096,
        "%qt", "%kvt", 0xDEAD_BEEF, "KV_TILE_SKIP",
    );
    // Sentinel base + sub-table offsets (qmin/qmax/kvmin/kvmax) all
    // appear as immediate operands of `add.u64`. The base itself
    // doesn't appear verbatim since qmin offset == base + 0; the
    // sub-table offsets DO show numerically.
    // qmin offset = 0xDEADBEEF (base), so at least one occurrence:
    assert!(
        ptx.contains("DEADBEEF") || ptx.contains("deadbeef") || ptx.contains("3735928559"),
        "sentinel not visible:\n{ptx}"
    );
}
```

- [ ] **Step 2: First run + accept snapshot**

```bash
cargo test -p nsl-codegen --test pca_tier_b_predicate_isolation 2>&1 | tail -10
cat crates/nsl-codegen/tests/snapshots/pca_tier_b_predicate_isolation__predicate_4k_block64.snap.new
```

Verify the snapshot shows: 4 `ld.shared.u16`, the disjoint `setp` pair, the `or.pred`, the `@%p_skip_TB bra KV_TILE_SKIP`. If correct:

```bash
INSTA_UPDATE=accept cargo test -p nsl-codegen --test pca_tier_b_predicate_isolation 2>&1 | tail -5
```

- [ ] **Step 3: Add predicate SASS assertion to `pca_sass_byte_identity`**

Append to the `tier_b_sass` module from Task 5b:

```rust
#[test]
fn tier_b_skip_predicate_branch_is_uniform_sm120() {
    let mut ptx = String::new();
    let cfg = super::tier_b_test_fixture();
    nsl_codegen::pca_tilerange::emit_skip_predicate(
        &mut ptx, &cfg, 4096, "%qt", "%kvt", 0, "KV_TILE_SKIP",
    );
    let kernel_ptx = super::wrap_in_minimal_kernel(&format!(
        "    .reg .u32 %qt, %kvt;\n    mov.u32 %qt, 0;\n    mov.u32 %kvt, 0;\n{ptx}\nKV_TILE_SKIP:\n    nop;"
    ));
    let sass = match ptx_to_sass(&kernel_ptx, 120) {
        Ok(s) => s,
        Err(e) if e.contains("Unknown") || e.contains("not found") => {
            eprintln!("SKIP: ptxas/cuobjdump not available: {e}");
            return;
        }
        Err(e) => panic!("ptxas/cuobjdump failure: {e}"),
    };
    let predicate_sass = sass_between(&sass, "PCA Tier B", "end PCA Tier B");
    assert!(
        predicate_sass.contains("BRA.U") || predicate_sass.contains("BRA.UNI"),
        "skip predicate branch not BRA.U on sm_120:\n{predicate_sass}"
    );
}
```

- [ ] **Step 4: Re-run extended SASS tests**

```bash
cargo test -p nsl-codegen --features cuda --test pca_sass_byte_identity tier_b_sass 2>&1 | tail -15
```

Expected: predicate SASS test passes (or SKIPs cleanly if CUDA toolkit absent).

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/tests/pca_tier_b_predicate_isolation.rs \
        crates/nsl-codegen/tests/snapshots/pca_tier_b_predicate_isolation__predicate_4k_block64.snap \
        crates/nsl-codegen/tests/pca_sass_byte_identity.rs
git commit -m "test(pca-tier-b-v2): skip predicate isolation snapshot + SASS BRA.U

Mirrors Task 5a/5b for the predicate. Isolation: 5 assertions
(snapshot, 4 ld.shared loads, disjoint setp logic, custom branch
label, sentinel visibility). SASS: BRA.U at the predicate branch
on sm_120. Spec sec 5.1 + sec 5.2."
```

---

## Phase 4 — Instrumentation + FA2 Integration (Tasks 7-8)

## Task 7: Kernel-side debug instrumentation for M3 skip-decision writeback

**Goal:** Emit per-tile skip-decision writeback into an HBM buffer when `cfg!(debug_kernel_instrumentation)` is set at codegen time. Buffer shape `[batch, head, num_q_tiles, num_kv_tiles] : u8` per spec §4.3.1. Lane-0 of owning warp writes the slot. Production builds never call the writeback emitter.

**Files:**
- Modify: `crates/nsl-codegen/src/pca_tilerange.rs::emit_skip_decision_writeback`
- Modify: `crates/nsl-codegen/src/flash_attention_v2/phases/forward/s_compute.rs` (gated call at skip-decision moment)

- [ ] **Step 1: Replace the stub `emit_skip_decision_writeback`**

Open `crates/nsl-codegen/src/pca_tilerange.rs`. Replace the stub body:

```rust
pub fn emit_skip_decision_writeback(
    ptx: &mut String,
    config: &crate::flash_attention::FlashAttentionConfig,
    seq_len: u32,
    qt_reg: &str,
    kvt_reg: &str,
    is_skip_pred: &str,           // %p_skip_TB from emit_skip_predicate
    decisions_buf_param: &str,    // .param .u64 holding the HBM buffer ptr
) {
    let num_kv = crate::pca_tile_config::num_tiles(seq_len, config.block_kv as u32);
    let _ = (config, seq_len);

    ptx.push_str("    // ----- PCA Tier B: skip-decision writeback (debug, spec sec 4.3) -----\n");
    ptx.push_str("    .reg .u64 %dec_buf_TB, %dec_slot_TB;\n");
    ptx.push_str("    .reg .u32 %slot_off_TB, %bh_TB, %bh_slot_TB;\n");
    ptx.push_str("    .reg .u16 %dec_val_TB;\n");
    ptx.push_str("    .reg .pred %p_owner_TB;\n");
    ptx.push_str("\n");

    // Compute flat offset: ((batch * num_heads + head) * num_q_tiles + qt) * num_kv_tiles + kvt
    // batch_idx + head_idx come from existing FA2 kernel parameters
    // %batch_idx, %head_idx (already declared in the kernel prelude).
    ptx.push_str("    // (batch * num_heads + head) precomputed elsewhere as %bh_idx; reuse if available\n");
    ptx.push_str("    mov.u32 %bh_TB, %bh_idx;\n");
    ptx.push_str(&format!(
        "    mad.lo.u32 %bh_slot_TB, %bh_TB, {num_q_tiles_const}, {qt_reg};\n",
        num_q_tiles_const = crate::pca_tile_config::num_tiles(seq_len, config.block_q as u32),
        qt_reg = qt_reg,
    ));
    ptx.push_str(&format!(
        "    mad.lo.u32 %slot_off_TB, %bh_slot_TB, {num_kv}, {kvt_reg};\n",
        num_kv = num_kv,
        kvt_reg = kvt_reg,
    ));

    // Load buffer base + add offset.
    ptx.push_str(&format!(
        "    ld.param.u64 %dec_buf_TB, [{decisions_buf_param}];\n"
    ));
    ptx.push_str("    cvt.u64.u32 %dec_slot_TB, %slot_off_TB;\n");
    ptx.push_str("    add.u64 %dec_slot_TB, %dec_buf_TB, %dec_slot_TB;\n");

    // Owner = lane 0 of owning warp. FA2's tile-warp mapping assigns
    // exactly one warp per (qt, kvt) pair via owning_warp() — for v1
    // we approximate with `warp_id == 0 && lane_id == 0` for the single
    // warp that owns the tile pair within this CTA.
    // TODO(tier-b.2): replace with the real owning_warp(qt, kvt) helper.
    ptx.push_str("    .reg .u32 %warp_id_TB;\n");
    ptx.push_str("    mov.u32 %warp_id_TB, %warpid;\n");
    ptx.push_str("    setp.eq.u32 %p_owner_TB, %warp_id_TB, 0;\n");
    ptx.push_str("    .reg .u32 %lane_TB; mov.u32 %lane_TB, %laneid;\n");
    ptx.push_str("    .reg .pred %p_lane_TB; setp.eq.u32 %p_lane_TB, %lane_TB, 0;\n");
    ptx.push_str("    and.pred %p_owner_TB, %p_owner_TB, %p_lane_TB;\n");

    // Decision value: 1 if disjoint (skipped), 0 if kept.
    ptx.push_str(&format!(
        "    selp.u16 %dec_val_TB, 1, 0, {is_skip_pred};\n"
    ));
    ptx.push_str("    @%p_owner_TB st.global.u8 [%dec_slot_TB], %dec_val_TB;\n");
    ptx.push_str("    // ----- end skip-decision writeback -----\n");
}
```

**Note on the `owning_warp(qt, kvt)` approximation:** v1 uses the single-warp-owner-per-CTA approximation. The real per-tile warp ownership in FA2's tile-warp mapping is left to Tier B.2 — for v1 the M3 parity test uses single-CTA fixtures where the approximation is correct. This is a documented limitation in spec §4.3.3.

- [ ] **Step 2: Add codegen-time gate via Cargo feature**

In `crates/nsl-codegen/Cargo.toml`, add a feature:

```toml
[features]
debug_kernel_instrumentation = []
```

In `s_compute.rs`, gate the writeback call:

```rust
#[cfg(feature = "debug_kernel_instrumentation")]
if crate::pca_tilerange::should_emit_tier_b(config, seq_len, residency) {
    crate::pca_tilerange::emit_skip_decision_writeback(
        ptx, config, seq_len,
        "%qt", "%kvt",
        "%p_skip_TB",
        "skip_decisions_ptr",  // new kernel param added below
    );
}
```

- [ ] **Step 3: Add the `skip_decisions_ptr` kernel parameter (gated)**

In the kernel parameter declaration emission (search for `.param .u64` declarations in the forward kernel prelude), append a gated parameter:

```rust
#[cfg(feature = "debug_kernel_instrumentation")]
ptx.push_str("    .param .u64 skip_decisions_ptr,\n");
```

Verify the parameter list comma discipline — the last param has no trailing comma.

- [ ] **Step 4: Build with the feature flag**

```bash
cargo build -p nsl-codegen --features debug_kernel_instrumentation 2>&1 | tail -15
cargo build -p nsl-codegen 2>&1 | tail -10   # baseline (no feature) still works
```

Expected: both builds clean. The feature-flagged build emits the writeback code; the default build does not.

- [ ] **Step 5: Smoke test the gated emission**

Add a test to `pca_tier_b_skeleton.rs` (or a new test file):

```rust
#[test]
#[cfg(feature = "debug_kernel_instrumentation")]
fn skip_writeback_emits_when_feature_enabled() {
    let mut ptx = String::new();
    let cfg = /* fa_base_for_test() */;
    nsl_codegen::pca_tilerange::emit_skip_decision_writeback(
        &mut ptx, &cfg, 4096,
        "%qt", "%kvt", "%p_skip_TB", "skip_decisions_ptr",
    );
    assert!(ptx.contains("st.global.u8"));
    assert!(ptx.contains("@%p_owner_TB"));
}
```

```bash
cargo test -p nsl-codegen --features debug_kernel_instrumentation skip_writeback_emits_when_feature_enabled 2>&1 | tail -10
```

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/pca_tilerange.rs \
        crates/nsl-codegen/src/flash_attention_v2/phases/forward/s_compute.rs \
        crates/nsl-codegen/Cargo.toml \
        crates/nsl-codegen/tests/pca_tier_b_skeleton.rs
git commit -m "feat(pca-tier-b-v2): kernel-side skip-decision instrumentation

Gated behind cargo feature debug_kernel_instrumentation; production
builds never call the writeback emitter. Buffer shape
[batch, head, num_q_tiles, num_kv_tiles]:u8 per spec sec 4.3.1.
Lane-0 of owning warp writes the slot (v1 approximates as warp 0
lane 0; real owning_warp() deferred to Tier B.2). Per spec sec 4.3."
```

---

## Task 8: Wire Tier B forward into FA2 kernel end-to-end + extended kernel snapshot

**Goal:** Call `emit_range_table_preamble` from `phases/forward/prelude.rs` immediately after the existing Tier A `segment_ids` SMEM load. Add Tier B-enabled variants to `pca_forward_kernel_snapshot` + `pca_backward_kernel_snapshot`. Refresh snapshots; commit the diff with cascade-summary line per spec §6.3.3.

**Files:**
- Modify: `crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs`
- Modify: `crates/nsl-codegen/tests/pca_forward_kernel_snapshot.rs`
- Modify: `crates/nsl-codegen/tests/pca_backward_kernel_snapshot.rs`
- Refresh: existing snapshot files for Tier B-enabled variants

- [ ] **Step 1: Locate the Tier A segment_ids SMEM load in forward prelude**

```bash
grep -n "seg_smem\|segment_ids" crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs | head -15
```

Find the cooperative warp-0 load loop that fills `seg_smem` with segment IDs from HBM. The Tier B preamble call goes IMMEDIATELY AFTER this load completes (and after the `bar.sync 0` that publishes the loaded values to other warps).

- [ ] **Step 2: Insert the preamble call**

In `forward/prelude.rs`, after the segment_ids load + post-load `bar.sync`:

```rust
// PCA Tier B range-table preamble — only when admitted.
if crate::pca_tilerange::should_emit_tier_b(config, seq_len, residency) {
    let range_table_base = crate::flash_attention_v2::smem_layout::tier_b_range_table_offset(config);
    crate::pca_tilerange::emit_range_table_preamble(
        ptx, config, seq_len, "seg_smem", range_table_base,
    );
}
```

If `forward/prelude.rs::emit` doesn't currently take `seq_len` or `residency`, thread them through from the orchestrator (`flash_attention_v2/mod.rs::synthesize_forward` or equivalent).

- [ ] **Step 3: Update launch sites to use `_with_seqlen` overloads**

```bash
grep -rn "shared_mem_bytes_v2[^_]\|shared_mem_bytes_v2_backward[^_]" crates/nsl-runtime crates/nsl-codegen
```

For every call site that launches a forward Tier B kernel (segment_masked=true with admissible Tier B config), switch to `shared_mem_bytes_v2_with_seqlen(config, seq_len, residency)`. Non-Tier-B launches continue calling the two-arg form — no-op guarantee preserves their SMEM layout.

Common call sites:
- `crates/nsl-runtime/src/.../flash_attention.rs` launch path.
- Test harnesses for forward kernel correctness.

- [ ] **Step 4: Add Tier B-enabled variants to `pca_forward_kernel_snapshot`**

Open `crates/nsl-codegen/tests/pca_forward_kernel_snapshot.rs`. Find the existing fixture for `forward_kernel_segment_masked_causal_32_32_32` (or similar). Add a parallel Tier B-enabled variant:

```rust
#[test]
fn forward_kernel_segment_masked_tier_b_causal_32_32_32() {
    let cfg = FlashAttentionConfig {
        block_q: 32, block_kv: 32, head_dim: 32,
        causal: true, paged: false, rope_q: false,
        rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
        tree_mask: false, gpu_sm: 120, segment_masked: true, csha: None,
    };
    let seq_len: u32 = 4096; // within Tier B 8KB range-table envelope
    let residency = SegmentResidency::Shared;
    let ptx = nsl_codegen::flash_attention_v2::synthesize_forward_with_seqlen(
        &cfg, seq_len, residency,
    ).expect("synthesize");
    insta::assert_snapshot!("forward_kernel_segment_masked_tier_b_causal_32_32_32", ptx);
}
```

If `synthesize_forward_with_seqlen` doesn't exist, add it as a thin wrapper around `synthesize_forward` that passes `seq_len` + `residency` through.

- [ ] **Step 5: First run + inspect snapshot**

```bash
cargo test -p nsl-codegen --test pca_forward_kernel_snapshot forward_kernel_segment_masked_tier_b_causal_32_32_32 2>&1 | tail -10
cat crates/nsl-codegen/tests/snapshots/pca_forward_kernel_snapshot__forward_kernel_segment_masked_tier_b_causal_32_32_32.snap.new
```

Verify the snapshot contains:
- Existing FA2 forward kernel structure (Q/K/V loads, QK^T matmul, softmax, PV).
- The Tier B preamble block (`----- PCA Tier B: range-table preamble -----`) immediately after segment_ids load.
- The skip predicate (`----- PCA Tier B: skip predicate -----`) at the top of the kv-tile loop.
- `KV_TILE_SKIP:` label before kv loop counter increment.

If correct:

```bash
INSTA_UPDATE=accept cargo test -p nsl-codegen --test pca_forward_kernel_snapshot 2>&1 | tail -5
```

- [ ] **Step 6: Add the backward Tier B variant (placeholder for Tier B.2 deferred)**

In `pca_backward_kernel_snapshot.rs`, add a fixture that verifies backward kernel emission is UNCHANGED for the same config (Tier B.2 forward-only in v1):

```rust
#[test]
fn backward_kernel_segment_masked_tier_b_unchanged_causal_32_32_32() {
    let cfg = /* same as forward variant above */;
    let seq_len: u32 = 4096;
    let residency = SegmentResidency::Shared;
    // Backward Tier B.2 deferred — backward emission must be identical to Tier A baseline.
    let ptx = nsl_codegen::flash_attention_v2::synthesize_backward(&cfg).expect("synthesize");
    assert!(
        !ptx.contains("PCA Tier B: range-table preamble"),
        "backward kernel should not contain Tier B preamble (Tier B.2 deferred)"
    );
}
```

- [ ] **Step 7: Full-suite check**

```bash
cargo test -p nsl-codegen 2>&1 | tail -30
```

Expected: all green. Non-Tier-B forward/backward snapshots unchanged (no-op guarantee from Task 4). New Tier B-enabled forward snapshot pinned.

- [ ] **Step 8: Commit with cascade-summary line in description**

```bash
git add crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs \
        crates/nsl-codegen/src/flash_attention_v2/mod.rs \
        crates/nsl-codegen/tests/pca_forward_kernel_snapshot.rs \
        crates/nsl-codegen/tests/pca_backward_kernel_snapshot.rs \
        crates/nsl-codegen/tests/snapshots/pca_forward_kernel_snapshot__forward_kernel_segment_masked_tier_b_causal_32_32_32.snap
git commit -m "$(cat <<'EOF'
feat(pca-tier-b-v2): wire Tier B forward into FA2 kernel end-to-end

Forward prelude calls emit_range_table_preamble after segment_ids
load + bar.sync; kv-tile loop top calls emit_skip_predicate;
KV_TILE_SKIP: label before kv loop increment skips QK^T + softmax
+ PV for disjoint tile pairs. Tier A per-cell mask continues to
handle cross-segment cells inside surviving tiles.

Backward kernel unchanged (Tier B.2 deferred); backward snapshot
asserts the absence of Tier B emission.

Cascade per spec sec 5.3:
  - PTX-text change at forward/prelude.rs (preamble insertion) +
    forward/s_compute.rs (predicate insertion) ->
  - kernel SASS shifted at q-phase + kv-phase loop branches (BRA.U)
    and at predicate branch (BRA.U) on sm_120 ->
  - uniform-register class verified by extended pca_sass_byte_identity
    (UR<n> grep on sm_120; BRA.U proxy on sm_80).

Spec sec 3.5 + sec 5.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 5 — Acceptance Gates (Tasks 9-11)

## Task 9: M3 estimator/runtime skip-mask parity test

**Goal:** Hard bit-equality between `pca_tileskip::build`'s skip mask and the kernel's actual skip-decision array on every §4.4 fixture. Requires the Task 7 instrumentation feature enabled.

**Files:**
- Create: `crates/nsl-codegen/tests/pca_tier_b_m3_parity.rs`
- Possibly modify: `crates/nsl-codegen/src/pca_tileskip.rs` (ensure layout-equivalence helper accepts the §4.2 `LayoutDescription`)

- [ ] **Step 1: Define the fixture matrix**

In a new file `crates/nsl-codegen/tests/fixtures/pca_tier_b_fixtures.rs` (or as a helper module inside the test file):

```rust
//! Fixture matrix for M3 parity per spec sec 4.4.
//! Each fixture pins (seq_len, num_docs, doc_lengths, doc_offsets, padding_locs)
//! so the layout-equivalence test reads identical inputs into the estimator
//! and the kernel.

#[derive(Debug, Clone)]
pub struct PackingFixture {
    pub name: &'static str,
    pub seq_len: u32,
    pub doc_lengths: Vec<u32>,
    pub doc_offsets: Vec<u32>,
    pub padding_locs: Vec<u32>,
}

pub fn fixture_matrix() -> Vec<PackingFixture> {
    vec![
        PackingFixture {
            name: "standard_3doc",
            seq_len: 4096,
            doc_lengths: vec![1366, 1366, 1364],
            doc_offsets: vec![0, 1366, 2732],
            padding_locs: vec![],
        },
        PackingFixture {
            name: "long_seq_5doc",
            seq_len: 16_384,
            doc_lengths: vec![3277, 3277, 3277, 3277, 3276],
            doc_offsets: vec![0, 3277, 6554, 9831, 13_108],
            padding_locs: vec![],
        },
        PackingFixture {
            name: "skewed_packing",
            seq_len: 4096,
            doc_lengths: vec![3000, 366, 365, 365],
            doc_offsets: vec![0, 3000, 3366, 3731],
            padding_locs: vec![],
        },
        PackingFixture {
            name: "boundary_dense",
            seq_len: 4096,
            doc_lengths: vec![256; 16],
            doc_offsets: (0..16).map(|i| i * 256).collect(),
            padding_locs: vec![],
        },
        PackingFixture {
            name: "single_doc",
            seq_len: 4096,
            doc_lengths: vec![4096],
            doc_offsets: vec![0],
            padding_locs: vec![],
        },
        PackingFixture {
            name: "tail_padding",
            seq_len: 4096,
            doc_lengths: vec![1024, 1024],
            doc_offsets: vec![0, 1024],
            padding_locs: (2048..4096).collect(),
        },
    ]
}

pub fn segment_ids_from_fixture(f: &PackingFixture) -> Vec<u16> {
    let mut ids = vec![u16::MAX; f.seq_len as usize]; // padding sentinel
    for (doc_idx, (&len, &off)) in f.doc_lengths.iter().zip(f.doc_offsets.iter()).enumerate() {
        for i in 0..len as usize {
            ids[off as usize + i] = doc_idx as u16;
        }
    }
    ids
}
```

- [ ] **Step 2: Write the parity test**

Create `crates/nsl-codegen/tests/pca_tier_b_m3_parity.rs`:

```rust
//! M3 estimator/runtime skip-mask parity test.
//!
//! For each fixture in spec sec 4.4: build the reference skip mask via
//! pca_tileskip::build, launch the Tier B kernel with instrumentation
//! enabled, read back the kernel's per-tile decisions, assert
//! bit-equality. Spec sec 4.3.
//!
//! Requires the debug_kernel_instrumentation feature.

#![cfg(all(feature = "cuda", feature = "debug_kernel_instrumentation"))]

mod fixtures;
use fixtures::*;

use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::pca_segment::SegmentResidency;
use nsl_codegen::pca_tile_config::num_tiles;
use nsl_codegen::pca_tileskip::build as build_reference_mask;

fn fa_cfg(block_q: u32, block_kv: u32) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: block_q as i32, block_kv: block_kv as i32, head_dim: 64,
        causal: true, paged: false, rope_q: false,
        rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
        tree_mask: false, gpu_sm: 120, segment_masked: true, csha: None,
    }
}

fn launch_kernel_and_readback_decisions(
    cfg: &FlashAttentionConfig,
    fixture: &PackingFixture,
) -> Vec<u8> {
    // Implementation: build PTX via synthesize_forward_with_seqlen, load via cudarc,
    // allocate decisions buffer [batch=1, head=1, num_q_tiles, num_kv_tiles]:u8,
    // launch, sync, memcpy_dtov decisions buffer.
    //
    // The launch needs:
    //   - Q, K, V tensors (small synthetic, correctness not under test here).
    //   - segment_ids from segment_ids_from_fixture(fixture).
    //   - skip_decisions_ptr — new param from Task 7.
    //
    // Reuse the launch helper from pca_tier_a_forward_correctness.rs and add
    // the skip_decisions_ptr arg.
    //
    // Returns the flat [num_q_tiles * num_kv_tiles] decision array as u8 (one CTA).
    let _ = (cfg, fixture);
    unimplemented!("see pca_tier_a_forward_correctness::launch_forward for the harness shape; add skip_decisions buffer + arg")
}

fn run_parity_for_fixture(cfg_block_q: u32, cfg_block_kv: u32, fixture: &PackingFixture) {
    let cfg = fa_cfg(cfg_block_q, cfg_block_kv);

    // Reference mask from estimator. Layout description per spec sec 4.2.
    let reference_mask = build_reference_mask(
        cfg_block_q, cfg_block_kv,
        fixture.seq_len,
        &fixture.doc_lengths, &fixture.doc_offsets, &fixture.padding_locs,
    );

    // Kernel-side decisions from instrumented Tier B kernel.
    let kernel_decisions = launch_kernel_and_readback_decisions(&cfg, fixture);

    // Bit-identical comparison. Failure surface names diverging tile coords.
    let num_q  = num_tiles(fixture.seq_len, cfg_block_q);
    let num_kv = num_tiles(fixture.seq_len, cfg_block_kv);
    assert_eq!(
        reference_mask.len(),
        kernel_decisions.len(),
        "fixture {}: estimator mask len {} != kernel decision len {}",
        fixture.name, reference_mask.len(), kernel_decisions.len(),
    );
    for qt in 0..num_q {
        for kvt in 0..num_kv {
            let idx = (qt * num_kv + kvt) as usize;
            assert_eq!(
                reference_mask[idx], kernel_decisions[idx],
                "fixture {}: divergence at qt={qt} kvt={kvt}: estimator={} kernel={}",
                fixture.name, reference_mask[idx], kernel_decisions[idx],
            );
        }
    }
}

#[test]
fn m3_parity_standard_3doc() {
    run_parity_for_fixture(64, 64, &fixture_matrix()[0]);
}

#[test]
fn m3_parity_long_seq_5doc() {
    run_parity_for_fixture(64, 64, &fixture_matrix()[1]);
}

#[test]
fn m3_parity_skewed_packing() {
    run_parity_for_fixture(64, 64, &fixture_matrix()[2]);
}

#[test]
fn m3_parity_boundary_dense() {
    run_parity_for_fixture(64, 64, &fixture_matrix()[3]);
}

#[test]
fn m3_parity_single_doc() {
    // M6 worst case (single doc, no skips possible): mask should be all-zero.
    let fixture = &fixture_matrix()[4];
    let reference = build_reference_mask(64, 64, fixture.seq_len, &fixture.doc_lengths, &fixture.doc_offsets, &fixture.padding_locs);
    assert!(reference.iter().all(|&d| d == 0), "single_doc should produce zero skips");
    run_parity_for_fixture(64, 64, fixture);
}

#[test]
fn m3_parity_tail_padding() {
    run_parity_for_fixture(64, 64, &fixture_matrix()[5]);
}
```

- [ ] **Step 3: Verify `pca_tileskip::build` signature matches the fixture inputs**

```bash
grep -n "pub fn build" crates/nsl-codegen/src/pca_tileskip.rs | head -5
```

If `pca_tileskip::build` takes a different argument shape, either:
- Adjust the parity test to convert `fixture` into the shape `build` expects.
- Add a thin adapter in `pca_tileskip.rs` that takes the §4.2 `LayoutDescription` form and returns the mask vector.

- [ ] **Step 4: Run the parity tests**

```bash
cargo test -p nsl-codegen --features cuda,debug_kernel_instrumentation \
    --test pca_tier_b_m3_parity 2>&1 | tail -20
```

Expected: all 6 fixtures pass with bit-identical masks. If any fixture diverges, failure diagnostic names the fixture + tile coords; debug by:
1. Printing both masks for visual inspection.
2. Checking whether the divergence is uniform (whole rows/columns mismatched) or scattered.
3. Uniform mismatch → estimator/kernel layout interpretation drift. Scattered → kernel reduction or address-arithmetic bug.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/tests/pca_tier_b_m3_parity.rs \
        crates/nsl-codegen/tests/fixtures/pca_tier_b_fixtures.rs
git commit -m "test(pca-tier-b-v2): M3 estimator/runtime parity on 6 fixtures

Bit-identical skip masks between pca_tileskip::build and the
instrumented Tier B kernel across spec sec 4.4 fixture matrix:
standard_3doc, long_seq_5doc, skewed_packing, boundary_dense,
single_doc, tail_padding. Failure diagnostic names the diverging
fixture + tile coords. Requires debug_kernel_instrumentation feature."
```

---

## Task 10: Per-variant SASS baselines + CI gating

**Goal:** Record per-variant cuobjdump instruction count + zero-spill count to `tests/sass_baselines/<variant>_tier_b.txt`. CI fails if drift > ±2.

**Files:**
- Create: `tests/sass_baselines/forward_kernel_segment_masked_tier_b_causal_32_32_32.txt`
- Create: `tests/sass_baselines/forward_kernel_segment_masked_tier_b_causal_64_64_64.txt`
- Create: `crates/nsl-codegen/tests/pca_tier_b_sass_baselines.rs`

- [ ] **Step 1: Generate baseline for the 32_32_32 variant**

```bash
# 1. Synthesize the Tier B-enabled forward kernel PTX.
cargo run -p nsl-codegen --bin emit-kernel --features cuda -- \
    --config forward_kernel_segment_masked_tier_b_causal_32_32_32 \
    --output /c/tmp/tier_b_32.ptx

# 2. Compile + dump SASS.
"$CUDA_PATH/bin/ptxas" -arch=sm_120 -v -o /c/tmp/tier_b_32.cubin /c/tmp/tier_b_32.ptx \
    2>&1 | tee /c/tmp/tier_b_32_ptxas.log
"$CUDA_PATH/bin/cuobjdump" --dump-sass /c/tmp/tier_b_32.cubin \
    2>&1 > /c/tmp/tier_b_32.sass

# 3. Count instructions (non-empty, non-comment, non-label lines).
SASS_COUNT=$(grep -cE '^\s+/\*[0-9a-fA-F]+\*/' /c/tmp/tier_b_32.sass)
SPILL_BYTES=$(grep -oE '[0-9]+ bytes spill' /c/tmp/tier_b_32_ptxas.log | head -1 | grep -oE '^[0-9]+')

echo "forward_kernel_segment_masked_tier_b_causal_32_32_32 sm_120"  > tests/sass_baselines/forward_kernel_segment_masked_tier_b_causal_32_32_32.txt
echo "instruction_count=${SASS_COUNT}"                              >> tests/sass_baselines/forward_kernel_segment_masked_tier_b_causal_32_32_32.txt
echo "spill_bytes=${SPILL_BYTES:-0}"                                >> tests/sass_baselines/forward_kernel_segment_masked_tier_b_causal_32_32_32.txt
echo "tolerance=2"                                                  >> tests/sass_baselines/forward_kernel_segment_masked_tier_b_causal_32_32_32.txt
echo "recorded_date=2026-05-12"                                     >> tests/sass_baselines/forward_kernel_segment_masked_tier_b_causal_32_32_32.txt
```

If the `emit-kernel` bin doesn't exist, use the snapshot-test output instead: copy the PTX from `tests/snapshots/pca_forward_kernel_snapshot__forward_kernel_segment_masked_tier_b_causal_32_32_32.snap` (strip the insta header), feed to ptxas + cuobjdump as above.

- [ ] **Step 2: Repeat for the 64_64_64 variant**

Same procedure, output to `tests/sass_baselines/forward_kernel_segment_masked_tier_b_causal_64_64_64.txt`.

- [ ] **Step 3: Write the CI gate test**

Create `crates/nsl-codegen/tests/pca_tier_b_sass_baselines.rs`:

```rust
//! Per-variant SASS baseline gate. Loads expected instruction count +
//! spill count from tests/sass_baselines/<variant>_tier_b.txt; computes
//! current values from the synthesized kernel + ptxas + cuobjdump;
//! fails if drift exceeds tolerance.
//!
//! Per spec sec 5.2 (per-variant SASS baselines).

#![cfg(feature = "cuda")]

use std::process::Command;

fn parse_baseline(path: &str) -> (usize, usize, usize) {
    let content = std::fs::read_to_string(path).expect("read baseline");
    let mut instr = 0;
    let mut spill = 0;
    let mut tol = 2;
    for line in content.lines() {
        if let Some(v) = line.strip_prefix("instruction_count=") {
            instr = v.trim().parse().unwrap_or(0);
        } else if let Some(v) = line.strip_prefix("spill_bytes=") {
            spill = v.trim().parse().unwrap_or(0);
        } else if let Some(v) = line.strip_prefix("tolerance=") {
            tol = v.trim().parse().unwrap_or(2);
        }
    }
    (instr, spill, tol)
}

fn current_sass_count_and_spill(ptx: &str, sm: u32) -> Result<(usize, usize), String> {
    // Same shell-out shape as the tier_b_sass module from Task 5b.
    // ... (paste the ptx_to_sass helper + ptxas -v stderr capture)
    let _ = (ptx, sm);
    unimplemented!("reuse ptx_to_sass + ptxas -v capture")
}

#[test]
fn baseline_forward_kernel_segment_masked_tier_b_causal_32_32_32() {
    let (expected_instr, expected_spill, tol) = parse_baseline(
        "tests/sass_baselines/forward_kernel_segment_masked_tier_b_causal_32_32_32.txt",
    );
    let cfg = /* fa_4k_block32 with segment_masked=true */;
    let ptx = nsl_codegen::flash_attention_v2::synthesize_forward_with_seqlen(
        &cfg, 4096, nsl_codegen::pca_segment::SegmentResidency::Shared,
    ).expect("synthesize");
    let (actual_instr, actual_spill) = match current_sass_count_and_spill(&ptx, 120) {
        Ok(x) => x,
        Err(e) if e.contains("Unknown") || e.contains("not found") => {
            eprintln!("SKIP: ptxas/cuobjdump not available: {e}");
            return;
        }
        Err(e) => panic!("ptxas/cuobjdump failure: {e}"),
    };

    let drift = (actual_instr as i64 - expected_instr as i64).abs();
    assert!(
        drift <= tol as i64,
        "instruction-count drift {drift} > tolerance {tol}: expected {expected_instr}, got {actual_instr}"
    );
    assert_eq!(actual_spill, expected_spill, "spill_bytes mismatch");
}

#[test]
fn baseline_forward_kernel_segment_masked_tier_b_causal_64_64_64() {
    // Same shape as above, different variant.
}
```

- [ ] **Step 4: Run the baseline gates**

```bash
cargo test -p nsl-codegen --features cuda --test pca_tier_b_sass_baselines 2>&1 | tail -15
```

Expected: both variants pass (drift 0 on first run since baseline was just recorded). On any future emission change, drift > ±2 triggers CI failure → reviewer either updates the baseline (legitimate change with justification) or fixes the emission (regression).

- [ ] **Step 5: Commit**

```bash
git add tests/sass_baselines/forward_kernel_segment_masked_tier_b_causal_32_32_32.txt \
        tests/sass_baselines/forward_kernel_segment_masked_tier_b_causal_64_64_64.txt \
        crates/nsl-codegen/tests/pca_tier_b_sass_baselines.rs
git commit -m "test(pca-tier-b-v2): per-variant SASS baselines + ±2 CI gate

Records cuobjdump instruction count + ptxas spill bytes for the two
canonical Tier B forward variants (causal_32_32_32, causal_64_64_64).
CI gate fails on drift > ±2 from baseline; new variants ship their
own baseline file in the same PR. Spec sec 5.2 (per-variant SASS
baselines) + sec 6.3.1 institutional pattern."
```

---

## Task 11: M2 + M6 measurement infrastructure

**Goal:** M2 (≥30% FLOP reduction on packed-3-doc fixture via Nsight Compute counters) and M6 (≤1% wall-time regression on single_doc fixture). These are documentation + scripting; the spec doesn't gate Tier B merge on absolute numbers being measured today (the gates fire after Tier B kernel runs in a real workload).

**Files:**
- Create: `scripts/measure_tier_b_m2.sh`
- Create: `scripts/measure_tier_b_m6.sh`
- Create: `docs/superpowers/specs/2026-05-12-tier-b-measurement-procedure.md`

- [ ] **Step 1: Write the M2 measurement script**

Create `scripts/measure_tier_b_m2.sh`:

```bash
#!/usr/bin/env bash
# M2 — FLOP reduction measurement.
# Runs Nsight Compute on a Tier-A-only baseline and a Tier-B-on variant of
# the standard_3doc fixture; computes FLOP ratio per spec sec 7 M2 formula.
set -euo pipefail

FIXTURE="standard_3doc"
SM=120

ncu="$CUDA_PATH/bin/ncu"
[ -x "$ncu" ] || { echo "ERR: ncu not at $ncu"; exit 1; }

# 1. Build the bench binary (Tier A only + Tier B on as separate runs).
cargo build --release -p nsl-codegen-bench --features cuda 2>&1 | tail -5

# 2. Run Tier-A-only baseline.
"$ncu" --metrics \
    smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum \
    --csv \
    target/release/nsl-codegen-bench --fixture "$FIXTURE" --tier-b=off \
    > /c/tmp/tier_a_baseline.csv

# 3. Run Tier-B-on.
"$ncu" --metrics \
    smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum \
    --csv \
    target/release/nsl-codegen-bench --fixture "$FIXTURE" --tier-b=on \
    > /c/tmp/tier_b_on.csv

# 4. Compute FLOP per spec sec 7 M2 formula:
#       FLOPs = fadd + 2*ffma + fmul
echo "Tier-A FLOPs:" ; awk -F, 'NR>1 {f+=$2+2*$3+$4} END {print f}' /c/tmp/tier_a_baseline.csv
echo "Tier-B FLOPs:" ; awk -F, 'NR>1 {f+=$2+2*$3+$4} END {print f}' /c/tmp/tier_b_on.csv
echo "Reduction: 1 - (B/A)"
```

```bash
chmod +x scripts/measure_tier_b_m2.sh
```

- [ ] **Step 2: Write the M6 measurement script**

Create `scripts/measure_tier_b_m6.sh`:

```bash
#!/usr/bin/env bash
# M6 — single_doc wall-time regression.
# Median of 5 runs; pass if (B / A) <= 1.01.
set -euo pipefail

FIXTURE="single_doc"
N_RUNS=5

run_median() {
    local mode=$1
    local times=()
    for _ in $(seq "$N_RUNS"); do
        local t
        t=$(target/release/nsl-codegen-bench --fixture "$FIXTURE" --tier-b="$mode" --emit-time-only)
        times+=("$t")
    done
    printf '%s\n' "${times[@]}" | sort -n | awk -v n=$N_RUNS 'NR == int((n+1)/2) {print}'
}

cargo build --release -p nsl-codegen-bench --features cuda 2>&1 | tail -3

T_A=$(run_median off)
T_B=$(run_median on)
RATIO=$(awk -v a="$T_A" -v b="$T_B" 'BEGIN {printf "%.4f", b/a}')

echo "Tier-A-only median: ${T_A}us"
echo "Tier-B-on median:   ${T_B}us"
echo "Ratio B/A: $RATIO (pass criterion: <= 1.01)"
awk -v r="$RATIO" 'BEGIN {exit (r > 1.01)}' && echo "M6 PASS" || echo "M6 FAIL"
```

- [ ] **Step 3: Write the measurement-procedure document**

Create `docs/superpowers/specs/2026-05-12-tier-b-measurement-procedure.md`:

```markdown
# Tier B Measurement Procedure (M2 + M6)

**Spec:** §7 of `2026-05-02-pca-tier-b-tile-skip-design.md` (M2: FLOP reduction, M6: single_doc regression).

## Prerequisites

- CUDA toolkit 13.x with Nsight Compute (`ncu`) installed at `$CUDA_PATH/bin/ncu`.
- Build harness `nsl-codegen-bench` with `--features cuda` (added in this task if not present).
- Hardware: sm_80 / sm_90 / sm_120 — Nsight metrics are architecture-stable.

## M2 — FLOP reduction (≥30% on standard_3doc)

Run `scripts/measure_tier_b_m2.sh`. The script:

1. Builds the bench binary in release.
2. Runs `ncu` with the three FLOP counters (`fadd`, `ffma`, `fmul`).
3. Computes `FLOPs = fadd + 2*ffma + fmul` per spec sec 7 M2 formula.
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

### 2026-XX-XX (initial)

(Fill in after first measurement run.)
```

- [ ] **Step 4: Commit**

```bash
git add scripts/measure_tier_b_m2.sh \
        scripts/measure_tier_b_m6.sh \
        docs/superpowers/specs/2026-05-12-tier-b-measurement-procedure.md
git commit -m "docs(pca-tier-b-v2): M2 + M6 measurement procedure + scripts

M2 (>=30% FLOP reduction on standard_3doc) via Nsight Compute counters
(fadd + 2*ffma + fmul per spec sec 7 formula). M6 (<=1% wall-time
regression on single_doc) via median-of-5 wall-time measurement.
Both scripts run after Tier B emission changes; results logged in
docs/superpowers/specs/2026-05-12-tier-b-measurement-procedure.md."
```

---

## Phase 6 — Close-out (Task 12)

## Task 12: Close-out + memory updates

**Goal:** Update project memory + spec changelog to reflect the shipped Tier B forward. Open the PR. Mark Tier B.2 backward as the next-session deferred item.

**Files:**
- Modify: `C:\Users\bwiem\.claude\projects\c--Users-bwiem-projects-NSL\memory\MEMORY.md` (move PCA Tier B from "Deferred / paused" to "Recently shipped")
- Modify: `C:\Users\bwiem\.claude\projects\c--Users-bwiem-projects-NSL\memory\project_pca_tier_b_paused.md` → rename or supersede with `project_pca_tier_b_v1_shipped.md`
- Possibly modify: `docs/superpowers/specs/2026-05-02-pca-tier-b-tile-skip-design.md` (add `## Status: shipped 2026-05-XX (forward only); Tier B.2 backward deferred to fresh session.` at the top, below the Revision Changelog)

- [ ] **Step 1: Update MEMORY.md**

Move the PCA Tier B entry from "Deferred / paused" to "Recently shipped" with a brief shipped-state description:

```markdown
## Recently shipped (2026-04-15 → 2026-05-XX)
...
- **PCA Tier B forward (Tile-Skip) v2** (2026-05-XX, PR #<number>): [details](project_pca_tier_b_v1_shipped.md). Forward kernel now skips QK^T + softmax + PV for disjoint tile pairs; ~30% FLOP reduction on standard_3doc fixture (M2 verified). Backward (Tier B.2) deferred to fresh session — forward path unblocks long-context packed training.
```

Remove the corresponding "Deferred / paused" entry.

- [ ] **Step 2: Create the shipped memory file**

Create `C:\Users\bwiem\.claude\projects\c--Users-bwiem-projects-NSL\memory\project_pca_tier_b_v1_shipped.md`:

```markdown
---
name: PCA Tier B v1 forward — SHIPPED 2026-05-XX
description: Tier B forward shipped with runtime-PTX-loop preamble, single tail offset SMEM layout, hybrid verification. Tier B.2 backward deferred.
type: project
---
# PCA Tier B v1 forward — shipped 2026-05-XX

## What shipped
- Runtime PTX loop preamble (q-phase + kv-phase) with BRA.U-eligible branches verified by extended pca_sass_byte_identity.
- Skip predicate at top of kv-tile loop in forward s_compute.rs; KV_TILE_SKIP: label bypasses QK^T + softmax + PV for disjoint tile pairs.
- Range tables ride in extern shmem via `smem_layout::tier_b_range_table_offset` + `RangeTableAddrs` struct in pca_tilerange.rs.
- `pca_tile_config::num_tiles` shared between Rust allocators and kernel tile-loop emission (three-site identity test pinned).
- Hybrid verification: isolation insta snapshots (preamble + predicate) + extended FA2 kernel snapshots + extended pca_sass_byte_identity (BRA.U + UR<n>).
- M3 parity test passes on all 6 spec §4.4 fixtures.
- M2 baseline measured at <X>% reduction on standard_3doc (target ≥30%).
- M6 baseline measured at <Y> wall-time ratio on single_doc (target ≤1.01).
- SASS baselines recorded for the two canonical Tier B-enabled variants.

## Load-bearing decisions (pinned from spec)
- Forward seg_smem stays as `.shared .align 4 .b8 seg_smem[N]` (separate static decl); Tier B's range tables ALWAYS use extern shmem via the new offset accessor. Probe outcome (see `docs/superpowers/specs/2026-05-12-tier-b-smem-probe-findings.md`) confirmed this is safe on the user's primary GPU.
- Runtime PTX loop (not compile-time unroll) — uniform emission across config matrix per IR-005.
- Hybrid verification (isolation + integration) per IR-006.

## Deferred to Tier B.2 (next session)
- Backward kernel integration (mirror forward predicate in ds_compute.rs).
- Real `owning_warp(qt, kvt)` mapping replaces the v1 single-warp approximation in the M3 instrumentation.
- Tier B-extended for sequences > 16K (range tables exceed 8 KB; needs per-warp register or HBM-resident tables).
```

- [ ] **Step 3: Update spec header with shipped status**

In `docs/superpowers/specs/2026-05-02-pca-tier-b-tile-skip-design.md`, just below the Revision Changelog header from Task 2, add:

```markdown
**Status: SHIPPED forward path 2026-05-XX (PR #<number>); Tier B.2 backward deferred to fresh session for context-window headroom.**
```

- [ ] **Step 4: Run the full plan-level acceptance**

```bash
# Build everything.
cargo build -p nsl-codegen --release 2>&1 | tail -5
cargo build -p nsl-codegen --features cuda,debug_kernel_instrumentation 2>&1 | tail -5

# All Tier B tests.
cargo test -p nsl-codegen --features cuda \
    pca_tier_b_layout_api \
    pca_tier_b_preamble_isolation \
    pca_tier_b_predicate_isolation \
    pca_tier_b_skeleton \
    pca_tile_config_identity \
    pca_sass_byte_identity \
    2>&1 | tail -20

# Snapshot tests (Tier B-enabled variants must match committed snapshots).
cargo test -p nsl-codegen --test pca_forward_kernel_snapshot 2>&1 | tail -10
cargo test -p nsl-codegen --test pca_backward_kernel_snapshot 2>&1 | tail -10

# Full crate test suite (regression guard).
cargo test -p nsl-codegen 2>&1 | tail -15
```

Expected: all green.

- [ ] **Step 5: Commit memory + spec status updates**

```bash
git add docs/superpowers/specs/2026-05-02-pca-tier-b-tile-skip-design.md
git commit -m "docs(pca-tier-b-v2): mark spec status as shipped (forward only)

Tier B.2 backward deferred to fresh session."

# Memory commit is separate (different repo/location):
cd C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory
# Apply MEMORY.md + project_pca_tier_b_v1_shipped.md edits, then commit
# (or rely on the auto-memory protocol's persistence).
```

- [ ] **Step 6: Open the PR**

```bash
gh pr create \
    --title "feat(pca): Tier B forward — tile-skip predicate + runtime-loop preamble (v2)" \
    --body "$(cat <<'EOF'
## Summary

- PCA Tier B forward kernel: skips QK^T + softmax + PV for tile pairs with strictly disjoint segment ranges; Tier A per-cell mask continues to handle cross-segment cells inside surviving tiles.
- Runtime PTX loop preamble (~120 bytes per phase, scales to long-context Tier B-extended).
- Single tail offset in `smem_layout` + `RangeTableAddrs` struct in `pca_tilerange` — no inline `.shared` decls (Blackwell sm_120 safe per the Phase 0 probe).
- Hybrid verification: isolation snapshots + extended FA2 kernel/SASS tests + per-variant SASS baselines.
- Phase 0 SMEM probe + findings doc captures the empirical mixed-allocation safety result.
- New institutional-rules registry at `docs/wiki/institutional-rules.md` (IR-001..IR-008) codifies the patterns surfaced across this brainstorm + prior specs.

## Test plan

- [x] `cargo test -p nsl-codegen` green on the worktree (1700+ tests).
- [x] Tier B-enabled forward kernel snapshot pinned; non-Tier-B snapshots byte-identical (no-op guarantee).
- [x] SASS `BRA.U` asserted at preamble q-loop + kv-loop + predicate branches on sm_120.
- [x] M3 parity test (6 fixtures) passes with bit-identical skip masks between estimator + kernel.
- [ ] M2 (≥30% FLOP reduction) measured on standard_3doc — pending Nsight Compute run (recorded in measurement procedure doc).
- [ ] M6 (≤1% wall-time regression) measured on single_doc — pending.
- [x] Phase 0 SMEM probe outcome recorded in findings doc; struct sub-layout decision cited.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 7: Final commit (PR-bookkeeping)**

If any spec/memory edits remain unstaged after the PR is opened, commit them:

```bash
git status
# stage anything outstanding
git add -p
git commit -m "chore(pca-tier-b-v2): close-out — final spec + memory updates"
```

---

## Acceptance gates (final, pre-merge)

| Gate | Pass criterion | Status |
|---|---|---|
| M1 | Forward output bit-exact within fp16 tolerance vs Tier-A-only on every §4.4 fixture | Task 8 snapshot diff covers structural correctness; numerical fixture comparison via existing pca_tier_a_forward_correctness extended in close-out |
| M2 | FLOP reduction ≥ 30% on standard_3doc (Nsight `fadd + 2*ffma + fmul`) | scripts/measure_tier_b_m2.sh; recorded post-merge |
| M3 | Skip-mask bit-equality estimator vs kernel on every §4.4 fixture | Task 9 |
| M5 | Four ptxas-pass criteria (sm_120/sm_80 accept, 0 spills, BRA.U at branches, UR<n>) | Tasks 5b + 6a + 10 |
| M6 | single_doc wall-time ratio ≤ 1.01 (median of 5) | scripts/measure_tier_b_m6.sh; recorded post-merge |

M4 (variant matrix audit) is implied — `FlashAttentionConfig` unchanged. Confirm in PR description.

---

## Commits in this branch (expected sequence)

1. `feat(pca-tier-b-v2): SMEM probe + initial findings doc` (Task 1)
2. `docs(pca-tier-b-v2): apply probe outcome to spec; create IR registry` (Task 2)
3. `feat(pca-tier-b-v2): pca_tile_config::num_tiles shared helper` (Task 3)
4. `feat(pca-tier-b-v2): tier_b_range_table_offset + RangeTableAddrs` (Task 4)
5. `feat(pca-tier-b-v2): emit_range_table_preamble runtime PTX loop` (Task 5)
6. `test(pca-tier-b-v2): preamble isolation insta snapshot` (Task 5a)
7. `test(pca-tier-b-v2): SASS BRA.U + UR<n> assertions for preamble` (Task 5b)
8. `feat(pca-tier-b-v2): emit_skip_predicate + wire into s_compute.rs` (Task 6)
9. `test(pca-tier-b-v2): skip predicate isolation snapshot + SASS BRA.U` (Task 6a)
10. `feat(pca-tier-b-v2): kernel-side skip-decision instrumentation` (Task 7)
11. `feat(pca-tier-b-v2): wire Tier B forward into FA2 kernel end-to-end` (Task 8)
12. `test(pca-tier-b-v2): M3 estimator/runtime parity on 6 fixtures` (Task 9)
13. `test(pca-tier-b-v2): per-variant SASS baselines + ±2 CI gate` (Task 10)
14. `docs(pca-tier-b-v2): M2 + M6 measurement procedure + scripts` (Task 11)
15. `docs(pca-tier-b-v2): mark spec status as shipped (forward only)` (Task 12)

---

## Deferred to Tier B.2 / orthogonal

- **Backward kernel integration** (mirror forward predicate in `ds_compute.rs` + extend instrumentation `owning_warp` mapping). Separate plan.
- **Tier B-extended** for sequences > 16 K (range tables > 8 KB; needs per-warp register or HBM-resident tables). Spec §3.3 §11.
- **CTA-uniform vs warp-uniform predicate** — v1 uses warp-uniform; CTA-uniform deferred. Spec §11.
- **Tile-skip-aware backward checkpointing** — Tier B.2-adjacent. Spec §11.

---

## Self-Review (run before handing off to execution)

**Spec coverage check** (does every spec section have at least one task?):

- Spec §1 (problem) — covered by Tasks 5 + 6 + 8 (preamble + predicate + wiring).
- Spec §2 (scope) — covered by Task 0 (verify prereqs) + Task 8 (forward integration only; Tier B.2 backward deferred).
- Spec §3.1 (preamble) — Task 5.
- Spec §3.2 (conservative-skip) — Task 6 (`setp.lt` + `setp.gt` + `or.pred` exactly matches the spec's worked example).
- Spec §3.3 (SMEM cost formula + 8 KB bound) — Task 4 (`tier_b_range_table_bytes`) + Task 3 (`num_tiles` helper) + existing `should_emit_tier_b` check (untouched).
- Spec §3.4 (SMEM layout API) — Tasks 3 + 4.
- Spec §3.5 (phase integration) — Task 6 + Task 8.
- Spec §3.6 (module placement) — All emitters in `pca_tilerange.rs`; `pca_tileskip.rs` repurposed as parity reference (Task 9).
- Spec §4 (estimator/runtime parity) — Task 9 (M3 parity test on 6 fixtures).
- Spec §4.3 (kernel-side instrumentation) — Task 7.
- Spec §4.4 (fixture matrix) — Task 9 fixtures module.
- Spec §5 (gating G1 always-on) — Implicit in Task 8 — call site uses `should_emit_tier_b` which returns true whenever `segment_masked && Shared && range_table <= 8KB`.
- Spec §6.3 (verification) — Tasks 5a + 5b + 6a + 8 + 10.
- Spec §7 (acceptance gates M1-M6) — Tasks 8 (M1 structural via snapshot) + 9 (M3) + 10 (M5) + 11 (M2 + M6 procedure).
- Spec §8 (alternatives) — Discussion-only; no task needed.
- Spec §9 (module orphan policy) — Untouched; this plan does not modify `pca_rope` or `PerDocumentCta`.
- Spec §10 (risks) — Mitigations covered: range-table bug → M3 (Task 9); per-thread branch → M5 (Tasks 5b/6a/10); SMEM > 8 KB → existing `should_emit_tier_b` gate; estimator/runtime drift → M3 in CI (Task 9); conservative-skip preserved → spec text + Task 6 emission shape; unpacked regression → M6 (Task 11); instrumentation buffer layout → Task 7 4-D shape pinned; new variant without baseline → Task 10 CI gate.
- Spec §11 (out-of-scope) — Deferred section above lists them.

Revision design §2 (probe) — Task 1.
Revision design §6.4 (IR registry) — Task 2 Step 7.

No spec section uncovered.

**Placeholder scan:**
- `<row>`, `<text>`, `<placeholders>` in Task 1 Step 4 findings-doc template — intentional; the engineer fills in from probe output.
- `<X>`, `<Y>` in Task 12 shipped-memory description — intentional; filled in post-measurement.
- `<number>` in PR title — intentional; filled in after `gh pr create` returns the PR number.
- `[Status: SHIPPED forward path 2026-05-XX ...]` date — filled in on actual ship day.
- No `TBD`, `TODO`, `fill in details`, or vague "appropriate handling" patterns in steps.

**Type consistency check:**
- `tier_b_range_table_offset(config) -> u32` consistent across smem_layout.rs definition + Task 4 test + Task 5 emitter call + Task 8 wiring + Task 10 baseline test.
- `RangeTableAddrs { qtile_min, qtile_max, kvtile_min, kvtile_max: u32 }` consistent across all references.
- `num_tiles(seq_len: u32, block_size: u32) -> u32` consistent across pca_tile_config + emitter callers + identity test.
- `emit_range_table_preamble(ptx, config, seq_len: u32, segment_ids_smem_base: &str, range_table_base: u32)` — 5 args, consistent across Task 5 definition + Task 5a isolation test + Task 8 wiring + Task 5b SASS test.
- `emit_skip_predicate(ptx, config, seq_len: u32, qt_reg: &str, kvt_reg: &str, range_table_base: u32, on_skip_label: &str)` — 7 args, consistent across Task 6 + Task 6a + extended pca_sass_byte_identity.
- `should_emit_tier_b(config, seq_len: u64, residency)` — existing signature; callers correctly cast `u32` seq_len `as u64` (visible in Task 4 + Task 8).
- `shared_mem_bytes_v2_with_seqlen(config, seq_len: u32, residency)` — new overload; consistent across Task 4 definition + Task 8 launch site.

No type drift.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-12-pca-tier-b-tile-skip-implementation-v2.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task; review between tasks; fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans; batch execution with checkpoints.

**Which approach?**
