# WRGA Fused-LoRA/IA³ PTX Rewrite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `synthesize_fused_lora_ptx` and `synthesize_fused_ia3_ptx` against a new shared `kernel_skeleton/` module extracted from FA v2 so that both emitters produce valid PTX that passes `ptxas` compilation, launches via cudarc, and produces numerically correct output at 1e-4.

**Architecture:** Fine-grained sandwich: a new `kernel_skeleton/` module owns the ~15 lines of PTX boilerplate genuinely shared between FA and WRGA (header, SMEM decl, tid/lane/warp registers, zero-pad helper, param-load helpers). Both synthesizers compose these skeleton primitives with existing `matmul_mma` primitives plus their own WRGA-specific tiling, epilogue math, and store logic. Four-layer test discipline (skeleton snapshots, unit ptxas, integration numerical, E2E launch-counter) replaces B.3's string-pattern tests that let pseudocode ship.

**Tech Stack:** Rust (nsl-codegen, nsl-runtime, nsl-cli crates), PTX 7.0 sm_80 emission, `cudarc::driver::sys::cuModuleLoadData` for ptxas validation, `insta`-style snapshot tests for skeleton variants, existing NSL test harness for integration and E2E.

**Reference spec:** [docs/plans/2026-04-16-wrga-fused-ptx-rewrite-design.md](docs/plans/2026-04-16-wrga-fused-ptx-rewrite-design.md)

**Branch:** The plan executes on a fresh worktree `feat/wrga-fused-ptx-rewrite` branched from `main`. The 6 spec-commits (one per Task Group) can either be preserved individually for bisect granularity or squashed at merge time — executor's choice.

**Commit taxonomy note:** Each Task ends with its own intermediate commit for bisect / rollback granularity. The spec's "6 commits" refer to the logical milestone commits that each Task Group produces. If a clean 6-commit history is preferred at merge time, rebase-squash within each Task Group.

---

## File Structure

**New files:**
- `crates/nsl-codegen/src/kernel_skeleton/mod.rs` — public API re-exports
- `crates/nsl-codegen/src/kernel_skeleton/header.rs` — `emit_ptx_header` + variant enums
- `crates/nsl-codegen/src/kernel_skeleton/smem.rs` — `emit_static_smem_decl`, `emit_dynamic_smem_extern`, `emit_shmem_base_cvta`
- `crates/nsl-codegen/src/kernel_skeleton/indexing.rs` — `emit_thread_lane_warp_registers`
- `crates/nsl-codegen/src/kernel_skeleton/pad.rs` — `emit_smem_zero_pad_predicated`
- `crates/nsl-codegen/src/kernel_skeleton/params.rs` — `emit_param_block`, `emit_ld_param_u64`, `emit_ld_param_f32`
- `crates/nsl-codegen/src/kernel_skeleton/tests/mod.rs` — test module declaration
- `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/*.snap` — per-variant pinned PTX (9 files)
- `crates/nsl-codegen/src/wrga_kernel_helpers.rs` — WRGA-specific tile-staging helpers (not shareable with FA)

**Modified files:**
- `crates/nsl-codegen/src/lib.rs` — add `pub mod kernel_skeleton;`, `pub mod wrga_kernel_helpers;`
- `crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs` — replace 3-line header, SMEM decl, cvta, tid-register emission with skeleton calls
- `crates/nsl-codegen/src/flash_attention_v2/smem_layout.rs` — route `total_bytes`/`needs_dynamic_smem` callers through skeleton (if any FA callers bypass prelude.rs)
- `crates/nsl-codegen/src/wrga_fused_ptx.rs` — full rewrite of `synthesize_fused_lora_ptx` and `synthesize_fused_ia3_ptx` bodies; keep FusedLoraConfig / FusedIa3Config / LoraKernelKey / kernel_key() APIs stable
- `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs` — add `build_4_fused_real_launch`, add IA³ fixtures A and B, flip `build_4_fused_cuda_actually_fires` from `#[ignore]` to `#[cfg(feature="cuda")]`
- `crates/nsl-codegen/tests/wrga_fused_ptx_ptxas.rs` — NEW test file for unit ptxas sweep (LoRA 6 configs, IA³ 5 configs)

**Memory file updates (at milestone close-out):**
- `C:\Users\bwiem\.claude\projects\c--Users-bwiem-projects-NSL\memory\project_wrga_ptx_scaffolding_discovered.md` — prepend retrospective paragraph + CLOSED marker
- `C:\Users\bwiem\.claude\projects\c--Users-bwiem-projects-NSL\memory\project_wrga_fused_ptx_rewrite.md` — NEW, records invariants
- `C:\Users\bwiem\.claude\projects\c--Users-bwiem-projects-NSL\memory\MEMORY.md` — index updates

---

## Prerequisites

Before starting Task A1, create a fresh worktree branched from `main`:

```bash
cd c:/Users/bwiem/projects/NSL
git worktree add .worktrees/wrga-fused-ptx-rewrite -b feat/wrga-fused-ptx-rewrite main
cd .worktrees/wrga-fused-ptx-rewrite
cargo build -p nsl-codegen -p nsl-runtime -p nsl-cli --features cuda
```

Confirm the baseline is clean:
```bash
cargo test -p nsl-codegen --lib 2>&1 | tail -5
cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence -- --test-threads=1 2>&1 | tail -5
```
Expected: `nsl-codegen` lib tests pass (~1167); `wrga_adapter_runtime_equivalence` 7 non-ignored tests pass, 1 ignored.

---

## Task Group A — Commit 1: FA extraction to `kernel_skeleton`

### Task A1: Inventory FA v2 prolog variation points

**Files:**
- Read: `crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs`
- Read: `crates/nsl-codegen/src/flash_attention_v2/smem_layout.rs`
- Read: `crates/nsl-codegen/tests/fa_v2_snapshots.rs`

- [ ] **Step 1: Grep for every `.target` directive in FA v2**

Run:
```bash
grep -rn '\.target\s*sm_' crates/nsl-codegen/src/flash_attention_v2/
```
Record in the commit log: the set of sm targets emitted by FA (expected: `sm_75`).

- [ ] **Step 2: Grep for every `.version` directive in FA v2**

Run:
```bash
grep -rn '\.version\s' crates/nsl-codegen/src/flash_attention_v2/
```
Record the set (expected: `8.7`).

- [ ] **Step 3: Grep for every `.shared` declaration variant in FA v2**

Run:
```bash
grep -rn '\.shared\s\+\.align' crates/nsl-codegen/src/flash_attention_v2/
grep -rn '\.extern\s\+\.shared' crates/nsl-codegen/src/flash_attention_v2/
```
Record: which paths emit which variant, and what `total_bytes` values are seen across `fa_v2_snapshots.rs` test configs.

- [ ] **Step 4: Grep for every `ld.param.*` directive in FA v2**

Run:
```bash
grep -rn 'ld\.param\.' crates/nsl-codegen/src/flash_attention_v2/
```
Record: types used (`.u64`, `.f32`, `.u32`) and register naming conventions (`%rd0..%rd9`, `%scale`, etc.).

- [ ] **Step 5: Grep for `%tid.x` / `%ctaid` usage in FA v2**

Run:
```bash
grep -rn 'mov\.u32\s\+\%.*\s*,\s*%tid\.' crates/nsl-codegen/src/flash_attention_v2/
grep -rn 'mov\.u32\s\+\%.*\s*,\s*%ctaid\.' crates/nsl-codegen/src/flash_attention_v2/
```
Confirm: FA's tid/warp/lane dance matches the 5-line pattern in `prelude.rs:204-208` and uses register names `%tid_x, %warp_id, %lane, %bid_x, %bid_y`.

- [ ] **Step 6: List all FA v2 snapshot test configs**

Run:
```bash
grep -nE 'fn\s+(phase_|kernel_full)' crates/nsl-codegen/tests/fa_v2_snapshots.rs
```
Record: the test names and the configs they cover. This is the ground-truth matrix the extraction must preserve byte-identically.

- [ ] **Step 7: Write the inventory summary as a commit message draft**

Create (do not commit yet) a file `crates/nsl-codegen/src/kernel_skeleton/EXTRACTION_INVENTORY.md` with:
```markdown
# FA v2 Prolog Variation Points (snapshot of 2026-04-XX)

## .version
- 8.7 — all FA v2 paths

## .target
- sm_75 — all FA v2 paths

## .shared decl variants
- Static `.shared .align 16 .b8 shmem[N]` — configs where needs_dynamic_smem(cfg) == false
- Dynamic `.extern .shared .align 16 .b8 shmem[]` — configs where needs_dynamic_smem(cfg) == true
  Emitted at MODULE scope BEFORE .visible .entry

## .param types used
- .u64 — all 30+ pointer params (q_ptr, k_ptr, ...)
- .f32 — scale, csha_eps
- .u32 — csha_active_heads, csha_d_model

## tid/lane/warp pattern
- .reg .u32 %tid_x, %warp_id, %lane, %bid_x, %bid_y;
- mov.u32 %tid_x, %tid.x;
- shr.u32 %warp_id, %tid_x, 5;
- and.b32 %lane, %tid_x, 31;
- mov.u32 %bid_x, %ctaid.x;
- mov.u32 %bid_y, %ctaid.y;

## Skeleton API coverage (proposed)
- emit_ptx_header(PtxVersion::{V7_0, V8_7}, TargetSm::{Sm75, Sm80})  ← covers both FA and WRGA
- emit_static_smem_decl(bytes: usize)                                ← covers all FA static configs
- emit_dynamic_smem_extern()                                         ← covers FA dynamic-SMEM path
- emit_shmem_base_cvta()                                             ← fixed content; no params
- emit_thread_lane_warp_registers()                                  ← fixed content; no params
- emit_param_block(entry_name, &[Param])                             ← covers arbitrary param lists
- emit_ld_param_u64(dest_reg, param_name)                            ← line-level helper
- emit_ld_param_f32(dest_reg, param_name)                            ← line-level helper
- emit_smem_zero_pad_predicated(smem_base_reg, real, padded, dtype_bits)  ← NEW for WRGA; FA can adopt later
```

- [ ] **Step 8: Verify the skeleton API proposal covers every FA variant recorded in steps 1-6**

Cross-check the "Skeleton API coverage" list against every grep result. If any FA emission pattern is NOT covered, STOP and extend the API proposal before continuing. The inventory driving the API (not the reverse) is load-bearing discipline.

- [ ] **Step 9: Commit the inventory as the first commit in the branch**

```bash
git add crates/nsl-codegen/src/kernel_skeleton/EXTRACTION_INVENTORY.md
git commit -m "docs(kernel_skeleton): FA v2 prolog variation inventory (pre-extraction)"
```

### Task A2: Create `kernel_skeleton/` module scaffolding

**Files:**
- Create: `crates/nsl-codegen/src/kernel_skeleton/mod.rs`
- Create: `crates/nsl-codegen/src/kernel_skeleton/header.rs`
- Create: `crates/nsl-codegen/src/kernel_skeleton/smem.rs`
- Create: `crates/nsl-codegen/src/kernel_skeleton/indexing.rs`
- Create: `crates/nsl-codegen/src/kernel_skeleton/pad.rs`
- Create: `crates/nsl-codegen/src/kernel_skeleton/params.rs`
- Modify: `crates/nsl-codegen/src/lib.rs`

- [ ] **Step 1: Create the module root with stub sub-modules**

Create `crates/nsl-codegen/src/kernel_skeleton/mod.rs`:
```rust
//! Shared PTX kernel boilerplate for FA and WRGA.
//!
//! This module owns only the ~15 lines of PTX that both FA and WRGA
//! genuinely emit identically: the file header, SMEM declaration, thread/
//! lane/warp register dance, and a few param-load line-level helpers.
//! Kernel-specific logic (tiling, MMA choreography, epilogue math, store
//! layout) stays in each caller's own emitter.
//!
//! **Extension discipline:** every helper below has a matching set of
//! pinned PTX snapshots in `tests/snapshots/`.  Adding a new variant to
//! FA or WRGA that the skeleton doesn't cover must also add a new snapshot
//! — the gap is caught at the skeleton layer, not requiring trace-back
//! from a failing FA snapshot.

pub mod header;
pub mod indexing;
pub mod pad;
pub mod params;
pub mod smem;

#[cfg(test)]
mod tests;
```

- [ ] **Step 2: Create empty sub-modules with module-level docstrings**

Create `crates/nsl-codegen/src/kernel_skeleton/header.rs`:
```rust
//! PTX file header emission (.version, .target, .address_size).
//!
//! Variants tested:
//! - (V8_7, Sm75) — FA v2 forward and backward
//! - (V7_0, Sm80) — WRGA fused LoRA and IA³
```

Create `crates/nsl-codegen/src/kernel_skeleton/smem.rs`:
```rust
//! Shared memory declarations.
//!
//! Three helpers matching FA's existing three-way split:
//! - emit_static_smem_decl     — inside function body, fixed size
//! - emit_dynamic_smem_extern  — module scope, extern declaration
//! - emit_shmem_base_cvta      — inside function body, casts shmem to %shmem_base
```

Create `crates/nsl-codegen/src/kernel_skeleton/indexing.rs`:
```rust
//! Thread/lane/warp/block-index register initialization.
//!
//! Emits the fixed 6-line tid/warp/lane/bid dance used by both FA and
//! WRGA.  Zero parameters — PTX convention fixes the register names.
```

Create `crates/nsl-codegen/src/kernel_skeleton/pad.rs`:
```rust
//! Predicated SMEM zero-padding for sub-MMA tile regions.
//!
//! Used by WRGA to zero the [real_extent, padded_extent) slice of an
//! SMEM tile when the real rank/m/k is below the MMA tile size.
//! Designed to be adoptable by FA for head_dim edge-case padding later.
```

Create `crates/nsl-codegen/src/kernel_skeleton/params.rs`:
```rust
//! Param block declaration and ld.param helpers.
//!
//! - emit_param_block    — emits `.visible .entry name(\n    .param ...\n)`
//! - emit_ld_param_u64   — line-level helper, caller names dest register
//! - emit_ld_param_f32   — line-level helper, caller names dest register
```

Create `crates/nsl-codegen/src/kernel_skeleton/tests/mod.rs`:
```rust
//! Skeleton primitive snapshot tests — one pinned `.snap` per variant.
```

- [ ] **Step 3: Wire the module into the crate**

Modify `crates/nsl-codegen/src/lib.rs`. Find the module declarations block (near the top). Add:
```rust
pub mod kernel_skeleton;
```
Place it alphabetically after any existing `pub mod k…` declaration.

- [ ] **Step 4: Build and confirm the crate still compiles**

Run:
```bash
cargo build -p nsl-codegen
```
Expected: clean build, no errors, only pre-existing warnings.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/kernel_skeleton/ crates/nsl-codegen/src/lib.rs
git commit -m "feat(kernel_skeleton): scaffold module with empty sub-modules"
```

### Task A3: Extract `emit_ptx_header` with per-variant snapshots

**Files:**
- Modify: `crates/nsl-codegen/src/kernel_skeleton/header.rs`
- Modify: `crates/nsl-codegen/src/kernel_skeleton/tests/mod.rs`
- Create: `crates/nsl-codegen/src/kernel_skeleton/tests/header_tests.rs`
- Create: `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/header__v87_sm75.snap`
- Create: `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/header__v70_sm80.snap`
- Modify: `crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs`

- [ ] **Step 1: Write the header implementation**

Edit `crates/nsl-codegen/src/kernel_skeleton/header.rs` — append:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtxVersion {
    V7_0,
    V8_7,
}

impl PtxVersion {
    fn as_str(self) -> &'static str {
        match self {
            PtxVersion::V7_0 => "7.0",
            PtxVersion::V8_7 => "8.7",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetSm {
    Sm75,
    Sm80,
}

impl TargetSm {
    fn as_str(self) -> &'static str {
        match self {
            TargetSm::Sm75 => "sm_75",
            TargetSm::Sm80 => "sm_80",
        }
    }
}

/// Emit the three-line PTX file header plus trailing blank line.
///
/// Variants tested: (V8_7, Sm75) for FA; (V7_0, Sm80) for WRGA.
/// Extend by adding a PtxVersion / TargetSm variant AND a new snapshot.
pub fn emit_ptx_header(ptx: &mut String, version: PtxVersion, sm: TargetSm) {
    ptx.push_str(&format!(".version {}\n", version.as_str()));
    ptx.push_str(&format!(".target {}\n", sm.as_str()));
    ptx.push_str(".address_size 64\n\n");
}
```

- [ ] **Step 2: Write the snapshot tests**

Create `crates/nsl-codegen/src/kernel_skeleton/tests/header_tests.rs`:
```rust
use crate::kernel_skeleton::header::{emit_ptx_header, PtxVersion, TargetSm};

#[test]
fn header__v87_sm75() {
    let mut ptx = String::new();
    emit_ptx_header(&mut ptx, PtxVersion::V8_7, TargetSm::Sm75);
    let expected = include_str!("snapshots/header__v87_sm75.snap");
    assert_eq!(ptx, expected, "FA-compatible header drift");
}

#[test]
fn header__v70_sm80() {
    let mut ptx = String::new();
    emit_ptx_header(&mut ptx, PtxVersion::V7_0, TargetSm::Sm80);
    let expected = include_str!("snapshots/header__v70_sm80.snap");
    assert_eq!(ptx, expected, "WRGA-compatible header drift");
}
```

Modify `crates/nsl-codegen/src/kernel_skeleton/tests/mod.rs`:
```rust
//! Skeleton primitive snapshot tests — one pinned `.snap` per variant.

mod header_tests;
```

- [ ] **Step 3: Write the snapshot files**

Create `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/header__v87_sm75.snap`:
```
.version 8.7
.target sm_75
.address_size 64

```
(Note: ends with blank line — the trailing `\n\n` from `emit_ptx_header` produces one literal blank line after `.address_size 64`.)

Create `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/header__v70_sm80.snap`:
```
.version 7.0
.target sm_80
.address_size 64

```

- [ ] **Step 4: Run the header tests, verify green**

Run:
```bash
cargo test -p nsl-codegen --lib kernel_skeleton::tests::header_tests
```
Expected: 2 passed, 0 failed.

If snapshot comparison fails, examine diff and either (a) fix the header impl or (b) fix the snapshot — whichever matches the intended spec behavior.

- [ ] **Step 5: Migrate FA v2 prelude.rs to call `emit_ptx_header`**

Modify `crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs` at lines 27-29. Replace:
```rust
    // File header.
    ptx.push_str(".version 8.7\n");
    ptx.push_str(".target sm_75\n");
    ptx.push_str(".address_size 64\n\n");
```
with:
```rust
    // File header.
    use crate::kernel_skeleton::header::{emit_ptx_header, PtxVersion, TargetSm};
    emit_ptx_header(ptx, PtxVersion::V8_7, TargetSm::Sm75);
```

- [ ] **Step 6: Run FA v2 snapshot tests to confirm byte-identical output**

Run:
```bash
cargo test -p nsl-codegen --test fa_v2_snapshots
```
Expected: all snapshots pass. If any snapshot diffs, the migration didn't preserve byte-identical output — fix before continuing.

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-codegen/src/kernel_skeleton/header.rs \
        crates/nsl-codegen/src/kernel_skeleton/tests/ \
        crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs
git commit -m "feat(kernel_skeleton): extract emit_ptx_header + variant snapshots; FA v2 adopts"
```

### Task A4: Extract SMEM helpers with per-variant snapshots

**Files:**
- Modify: `crates/nsl-codegen/src/kernel_skeleton/smem.rs`
- Modify: `crates/nsl-codegen/src/kernel_skeleton/tests/mod.rs`
- Create: `crates/nsl-codegen/src/kernel_skeleton/tests/smem_tests.rs`
- Create: `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/static_smem_decl__1536_bytes.snap`
- Create: `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/static_smem_decl__768_bytes.snap`
- Create: `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/dynamic_smem_extern.snap`
- Create: `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/shmem_base_cvta.snap`
- Modify: `crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs`

- [ ] **Step 1: Write the three SMEM helpers**

Edit `crates/nsl-codegen/src/kernel_skeleton/smem.rs` — append:
```rust
/// Emit static SMEM declaration inside a function body:
///     .shared .align 16 .b8 shmem[{bytes}];
///
/// Caller must emit this AFTER the `.visible .entry ... { ` opener but
/// BEFORE any register decl that uses shmem (via %shmem_base).
pub fn emit_static_smem_decl(ptx: &mut String, bytes: usize) {
    ptx.push_str(&format!(
        "    .shared .align 16 .b8 shmem[{}];\n",
        bytes
    ));
}

/// Emit dynamic SMEM extern declaration at MODULE scope:
///     .extern .shared .align 16 .b8 shmem[];
///
/// Caller must emit this BEFORE the `.visible .entry` directive.  PTX
/// disallows `.extern .shared` inside a function body.
pub fn emit_dynamic_smem_extern(ptx: &mut String) {
    ptx.push_str(".extern .shared .align 16 .b8 shmem[];\n\n");
}

/// Emit the `cvta.shared.u64 %shmem_base, shmem;` line that casts the
/// SMEM array symbol to a register-holdable shared-address value.
///
/// Caller must have already declared `.reg .u64 %shmem_base;` before
/// calling this.
pub fn emit_shmem_base_cvta(ptx: &mut String) {
    ptx.push_str("    cvta.shared.u64 %shmem_base, shmem;\n");
}
```

- [ ] **Step 2: Write the snapshot tests**

Create `crates/nsl-codegen/src/kernel_skeleton/tests/smem_tests.rs`:
```rust
use crate::kernel_skeleton::smem::{
    emit_dynamic_smem_extern, emit_shmem_base_cvta, emit_static_smem_decl,
};

#[test]
fn static_smem_decl__1536_bytes() {
    let mut ptx = String::new();
    emit_static_smem_decl(&mut ptx, 1536);
    assert_eq!(
        ptx,
        include_str!("snapshots/static_smem_decl__1536_bytes.snap"),
        "WRGA LoRA SMEM decl drift"
    );
}

#[test]
fn static_smem_decl__768_bytes() {
    let mut ptx = String::new();
    emit_static_smem_decl(&mut ptx, 768);
    assert_eq!(
        ptx,
        include_str!("snapshots/static_smem_decl__768_bytes.snap"),
        "WRGA IA3 SMEM decl drift"
    );
}

#[test]
fn dynamic_smem_extern() {
    let mut ptx = String::new();
    emit_dynamic_smem_extern(&mut ptx);
    assert_eq!(
        ptx,
        include_str!("snapshots/dynamic_smem_extern.snap"),
        "FA dynamic SMEM extern drift"
    );
}

#[test]
fn shmem_base_cvta() {
    let mut ptx = String::new();
    emit_shmem_base_cvta(&mut ptx);
    assert_eq!(
        ptx,
        include_str!("snapshots/shmem_base_cvta.snap"),
        "shmem base cvta drift"
    );
}
```

Append to `crates/nsl-codegen/src/kernel_skeleton/tests/mod.rs`:
```rust
mod smem_tests;
```

- [ ] **Step 3: Write the snapshot files**

Create `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/static_smem_decl__1536_bytes.snap`:
```
    .shared .align 16 .b8 shmem[1536];
```
(Single trailing newline.)

Create `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/static_smem_decl__768_bytes.snap`:
```
    .shared .align 16 .b8 shmem[768];
```

Create `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/dynamic_smem_extern.snap`:
```
.extern .shared .align 16 .b8 shmem[];

```
(Ends with blank line from `\n\n`.)

Create `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/shmem_base_cvta.snap`:
```
    cvta.shared.u64 %shmem_base, shmem;
```

- [ ] **Step 4: Run SMEM snapshot tests, verify green**

Run:
```bash
cargo test -p nsl-codegen --lib kernel_skeleton::tests::smem_tests
```
Expected: 4 passed.

- [ ] **Step 5: Migrate FA v2 prelude.rs to call the SMEM helpers**

Modify `crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs` at lines 31-38 (dynamic extern path). Replace:
```rust
    if needs_dynamic_smem(config) {
        ptx.push_str(".extern .shared .align 16 .b8 shmem[];\n\n");
    }
```
with:
```rust
    if needs_dynamic_smem(config) {
        use crate::kernel_skeleton::smem::emit_dynamic_smem_extern;
        emit_dynamic_smem_extern(ptx);
    }
```

Modify lines 78-84 (static decl path). Replace:
```rust
    if !needs_dynamic_smem(config) {
        ptx.push_str(&format!(
            "    .shared .align 16 .b8 shmem[{}];\n",
            total_bytes(config)
        ));
    }
```
with:
```rust
    if !needs_dynamic_smem(config) {
        use crate::kernel_skeleton::smem::emit_static_smem_decl;
        emit_static_smem_decl(ptx, total_bytes(config));
    }
```

Modify line 188. Replace:
```rust
    ptx.push_str("    cvta.shared.u64 %shmem_base, shmem;\n");
```
with:
```rust
    crate::kernel_skeleton::smem::emit_shmem_base_cvta(ptx);
```

- [ ] **Step 6: Run FA v2 snapshot tests to confirm byte-identical output**

Run:
```bash
cargo test -p nsl-codegen --test fa_v2_snapshots
```
Expected: all snapshots pass (byte-identical).

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-codegen/src/kernel_skeleton/smem.rs \
        crates/nsl-codegen/src/kernel_skeleton/tests/smem_tests.rs \
        crates/nsl-codegen/src/kernel_skeleton/tests/mod.rs \
        crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/static_smem_decl__1536_bytes.snap \
        crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/static_smem_decl__768_bytes.snap \
        crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/dynamic_smem_extern.snap \
        crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/shmem_base_cvta.snap \
        crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs
git commit -m "feat(kernel_skeleton): extract SMEM helpers (static/dynamic/cvta) + snapshots"
```

### Task A5: Extract `emit_thread_lane_warp_registers` with snapshot

**Files:**
- Modify: `crates/nsl-codegen/src/kernel_skeleton/indexing.rs`
- Modify: `crates/nsl-codegen/src/kernel_skeleton/tests/mod.rs`
- Create: `crates/nsl-codegen/src/kernel_skeleton/tests/indexing_tests.rs`
- Create: `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/thread_lane_warp_registers.snap`
- Modify: `crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs`

- [ ] **Step 1: Write the indexing helper**

Edit `crates/nsl-codegen/src/kernel_skeleton/indexing.rs` — append:
```rust
/// Emit the fixed 6-line tid/warp/lane/bid register declaration and
/// initialization block.
///
/// After this returns, the following registers hold useful values:
///   %tid_x   (u32) = threadIdx.x
///   %warp_id (u32) = tid_x / 32
///   %lane    (u32) = tid_x % 32
///   %bid_x   (u32) = blockIdx.x
///   %bid_y   (u32) = blockIdx.y
///
/// Zero parameters — PTX convention fixes the register names.  Callers
/// needing different names alias locally with `mov`.
pub fn emit_thread_lane_warp_registers(ptx: &mut String) {
    ptx.push_str("    .reg .u32 %tid_x, %warp_id, %lane, %bid_x, %bid_y;\n");
    ptx.push_str("    mov.u32 %tid_x, %tid.x;\n");
    ptx.push_str("    shr.u32 %warp_id, %tid_x, 5;\n");
    ptx.push_str("    and.b32 %lane, %tid_x, 31;\n");
    ptx.push_str("    mov.u32 %bid_x, %ctaid.x;\n");
    ptx.push_str("    mov.u32 %bid_y, %ctaid.y;\n");
}
```

- [ ] **Step 2: Write the snapshot test**

Create `crates/nsl-codegen/src/kernel_skeleton/tests/indexing_tests.rs`:
```rust
use crate::kernel_skeleton::indexing::emit_thread_lane_warp_registers;

#[test]
fn thread_lane_warp_registers() {
    let mut ptx = String::new();
    emit_thread_lane_warp_registers(&mut ptx);
    assert_eq!(
        ptx,
        include_str!("snapshots/thread_lane_warp_registers.snap"),
        "tid/warp/lane dance drift"
    );
}
```

Append to `crates/nsl-codegen/src/kernel_skeleton/tests/mod.rs`:
```rust
mod indexing_tests;
```

- [ ] **Step 3: Write the snapshot file**

Create `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/thread_lane_warp_registers.snap`:
```
    .reg .u32 %tid_x, %warp_id, %lane, %bid_x, %bid_y;
    mov.u32 %tid_x, %tid.x;
    shr.u32 %warp_id, %tid_x, 5;
    and.b32 %lane, %tid_x, 31;
    mov.u32 %bid_x, %ctaid.x;
    mov.u32 %bid_y, %ctaid.y;
```

- [ ] **Step 4: Run the indexing snapshot test, verify green**

Run:
```bash
cargo test -p nsl-codegen --lib kernel_skeleton::tests::indexing_tests
```
Expected: 1 passed.

- [ ] **Step 5: Migrate FA v2 prelude.rs to call the indexing helper**

Modify `crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs`. Find the register declaration line (line ~94):
```rust
    ptx.push_str("    .reg .u32 %tid_x, %warp_id, %lane, %bid_x, %bid_y;\n");
```
and the 5 mov/shr/and lines around line 204-208. Replace the 6 lines total with a single call:
```rust
    use crate::kernel_skeleton::indexing::emit_thread_lane_warp_registers;
    emit_thread_lane_warp_registers(ptx);
```
placed at the SAME position the register declaration originally occupied (line ~94, near the top of the register-decl block) — NOT at line 204 where the mov directives originally lived. The skeleton helper now bundles both parts. FA's existing ordering puts register decls first, then scalar param loads, then tid/warp/lane — since the helper combines decl+init, the block moves to where register decls lived; the scalar param loads now follow AFTER the tid/warp/lane init.

Verify the snapshot still matches after this reordering — if FA's test snapshots encode the original order (decl at top, init after param loads), the new ordering may diff. If it does, fix by keeping the skeleton call at the original init site (line 204) and keeping the bare register decl at line 94 as a separate line (i.e., the skeleton helper emits only the init, not the decl). Adjust the helper accordingly.

**Decision point:** inspect the FA snapshot `phase_prelude__32x32x32` or similar — find the exact line order. If the `.reg .u32 %tid_x, ...` line is adjacent to the `mov.u32 %tid_x, %tid.x;` lines in the snapshot, keep the combined helper. If they're separated by other lines in the snapshot, split the helper into `emit_thread_lane_warp_register_decl()` (declaration only) and `emit_thread_lane_warp_register_init()` (mov/shr/and only) and commit both as separate helpers with separate snapshots.

If splitting is needed, the snapshot `thread_lane_warp_registers.snap` becomes two files:
- `thread_lane_warp_register_decl.snap` — just the `.reg .u32 ...` line
- `thread_lane_warp_register_init.snap` — the 5 mov/shr/and lines

- [ ] **Step 6: Run FA v2 snapshot tests to confirm byte-identical output**

Run:
```bash
cargo test -p nsl-codegen --test fa_v2_snapshots
```
Expected: all snapshots pass. If any diff, revisit Step 5's decision point and split the helper.

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-codegen/src/kernel_skeleton/indexing.rs \
        crates/nsl-codegen/src/kernel_skeleton/tests/indexing_tests.rs \
        crates/nsl-codegen/src/kernel_skeleton/tests/mod.rs \
        crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/thread_lane_warp_registers.snap \
        crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs
git commit -m "feat(kernel_skeleton): extract emit_thread_lane_warp_registers + snapshot"
```

### Task A6: Extract param helpers with snapshots

**Files:**
- Modify: `crates/nsl-codegen/src/kernel_skeleton/params.rs`
- Modify: `crates/nsl-codegen/src/kernel_skeleton/tests/mod.rs`
- Create: `crates/nsl-codegen/src/kernel_skeleton/tests/params_tests.rs`
- Create: `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/ld_param_u64.snap`
- Create: `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/ld_param_f32.snap`
- Create: `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/param_block__wrga_lora.snap`

- [ ] **Step 1: Write the param helpers**

Edit `crates/nsl-codegen/src/kernel_skeleton/params.rs` — append:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamTy {
    U64,
    F32,
    U32,
}

impl ParamTy {
    fn as_str(self) -> &'static str {
        match self {
            ParamTy::U64 => ".u64",
            ParamTy::F32 => ".f32",
            ParamTy::U32 => ".u32",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Param {
    pub ty: ParamTy,
    pub name: &'static str,
}

/// Emit the `.visible .entry <entry_name> ( .param ..., .param ..., ... )` block.
/// The opening brace `{` is NOT emitted — caller opens the body.
pub fn emit_param_block(ptx: &mut String, entry_name: &str, params: &[Param]) {
    ptx.push_str(&format!(".visible .entry {} (\n", entry_name));
    for (i, p) in params.iter().enumerate() {
        let trailing = if i + 1 < params.len() { "," } else { "" };
        ptx.push_str(&format!("    .param {} {}{}\n", p.ty.as_str(), p.name, trailing));
    }
    ptx.push_str(")\n{\n");
}

/// Emit one `ld.param.u64 <dest_reg>, [<param_name>];` line.
pub fn emit_ld_param_u64(ptx: &mut String, dest_reg: &str, param_name: &str) {
    ptx.push_str(&format!("    ld.param.u64 {}, [{}];\n", dest_reg, param_name));
}

/// Emit one `ld.param.f32 <dest_reg>, [<param_name>];` line.
pub fn emit_ld_param_f32(ptx: &mut String, dest_reg: &str, param_name: &str) {
    ptx.push_str(&format!("    ld.param.f32 {}, [{}];\n", dest_reg, param_name));
}
```

- [ ] **Step 2: Write the snapshot tests**

Create `crates/nsl-codegen/src/kernel_skeleton/tests/params_tests.rs`:
```rust
use crate::kernel_skeleton::params::{
    emit_ld_param_f32, emit_ld_param_u64, emit_param_block, Param, ParamTy,
};

#[test]
fn ld_param_u64() {
    let mut ptx = String::new();
    emit_ld_param_u64(&mut ptx, "%rd_x", "x_ptr");
    assert_eq!(
        ptx,
        include_str!("snapshots/ld_param_u64.snap"),
        "ld.param.u64 drift"
    );
}

#[test]
fn ld_param_f32() {
    let mut ptx = String::new();
    emit_ld_param_f32(&mut ptx, "%scale_reg", "scale");
    assert_eq!(
        ptx,
        include_str!("snapshots/ld_param_f32.snap"),
        "ld.param.f32 drift"
    );
}

#[test]
fn param_block__wrga_lora() {
    let mut ptx = String::new();
    let params = [
        Param { ty: ParamTy::U64, name: "x_ptr" },
        Param { ty: ParamTy::U64, name: "w_ptr" },
        Param { ty: ParamTy::U64, name: "a_ptr" },
        Param { ty: ParamTy::U64, name: "b_ptr" },
        Param { ty: ParamTy::F32, name: "scale" },
        Param { ty: ParamTy::U64, name: "y_ptr" },
    ];
    emit_param_block(&mut ptx, "nsl_wrga_fused_lora_test", &params);
    assert_eq!(
        ptx,
        include_str!("snapshots/param_block__wrga_lora.snap"),
        "WRGA LoRA param block drift"
    );
}
```

Append to `crates/nsl-codegen/src/kernel_skeleton/tests/mod.rs`:
```rust
mod params_tests;
```

- [ ] **Step 3: Write the snapshot files**

Create `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/ld_param_u64.snap`:
```
    ld.param.u64 %rd_x, [x_ptr];
```

Create `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/ld_param_f32.snap`:
```
    ld.param.f32 %scale_reg, [scale];
```

Create `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/param_block__wrga_lora.snap`:
```
.visible .entry nsl_wrga_fused_lora_test (
    .param .u64 x_ptr,
    .param .u64 w_ptr,
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .f32 scale,
    .param .u64 y_ptr
)
{
```

- [ ] **Step 4: Run param snapshot tests, verify green**

Run:
```bash
cargo test -p nsl-codegen --lib kernel_skeleton::tests::params_tests
```
Expected: 3 passed.

- [ ] **Step 5: (No FA migration in this task.)**

FA's param block is 30+ params with interleaved comments; migrating it is a separate cleanup that doesn't serve this milestone. The `emit_param_block` helper is designed to be adoptable by FA later but not required here. The `ld_param_*` helpers are also not migrated to FA in this task — FA uses a different register naming scheme (`%rd0..%rd9`) which would need a separate migration pass.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/kernel_skeleton/params.rs \
        crates/nsl-codegen/src/kernel_skeleton/tests/params_tests.rs \
        crates/nsl-codegen/src/kernel_skeleton/tests/mod.rs \
        crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/ld_param_u64.snap \
        crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/ld_param_f32.snap \
        crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/param_block__wrga_lora.snap
git commit -m "feat(kernel_skeleton): extract param helpers (param_block, ld_param_*) + snapshots"
```

### Task A7: Create `emit_smem_zero_pad_predicated` (new helper for WRGA)

**Files:**
- Modify: `crates/nsl-codegen/src/kernel_skeleton/pad.rs`
- Modify: `crates/nsl-codegen/src/kernel_skeleton/tests/mod.rs`
- Create: `crates/nsl-codegen/src/kernel_skeleton/tests/pad_tests.rs`
- Create: `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/smem_zero_pad__rank4_to_16_f16.snap`
- Create: `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/smem_zero_pad__rank16_to_16_f16.snap`

- [ ] **Step 1: Write the pad helper**

Edit `crates/nsl-codegen/src/kernel_skeleton/pad.rs` — append:
```rust
/// Emit a predicated SMEM zero-store loop that clears the slice
/// `[real_extent, padded_extent)` of an SMEM region.
///
/// When `real_extent == padded_extent`, emits zero instructions (no-op).
/// This pure-pass-through is a tested invariant — do not optimize away
/// as `return` or panic, since snapshot tests assert the empty output.
///
/// # Arguments
/// * `smem_base_reg` — register holding the SMEM base for this region,
///   e.g. `"%a_tile_base"`.  Caller must have initialized this register.
/// * `real_extent` — real size of the data dimension (e.g. rank = 4)
/// * `padded_extent` — padded size (e.g. MMA k = 16)
/// * `dtype_bits` — 16 for f16, 32 for f32; controls `st.shared.b{bits}`
///
/// # Emission shape (for real=4, padded=16, dtype=16)
/// ```text
///     // Zero-pad SMEM region [4, 16) at f16 granularity
///     st.shared.b16 [%a_tile_base + 8], 0;
///     st.shared.b16 [%a_tile_base + 10], 0;
///     st.shared.b16 [%a_tile_base + 12], 0;
///     st.shared.b16 [%a_tile_base + 14], 0;
///     st.shared.b16 [%a_tile_base + 16], 0;
///     st.shared.b16 [%a_tile_base + 18], 0;
///     st.shared.b16 [%a_tile_base + 20], 0;
///     st.shared.b16 [%a_tile_base + 22], 0;
///     st.shared.b16 [%a_tile_base + 24], 0;
///     st.shared.b16 [%a_tile_base + 26], 0;
///     st.shared.b16 [%a_tile_base + 28], 0;
///     st.shared.b16 [%a_tile_base + 30], 0;
/// ```
pub fn emit_smem_zero_pad_predicated(
    ptx: &mut String,
    smem_base_reg: &str,
    real_extent: u32,
    padded_extent: u32,
    dtype_bits: u32,
) {
    assert!(
        real_extent <= padded_extent,
        "real_extent ({}) must be ≤ padded_extent ({})",
        real_extent,
        padded_extent,
    );
    if real_extent == padded_extent {
        // No-op path — locked in by smem_zero_pad__rank16_to_16_f16.snap.
        return;
    }
    ptx.push_str(&format!(
        "    // Zero-pad SMEM region [{}, {}) at {}-bit granularity\n",
        real_extent, padded_extent, dtype_bits,
    ));
    let bytes_per_elem = dtype_bits / 8;
    for i in real_extent..padded_extent {
        let offset = i * bytes_per_elem;
        ptx.push_str(&format!(
            "    st.shared.b{} [{} + {}], 0;\n",
            dtype_bits, smem_base_reg, offset,
        ));
    }
}
```

- [ ] **Step 2: Write the snapshot tests**

Create `crates/nsl-codegen/src/kernel_skeleton/tests/pad_tests.rs`:
```rust
use crate::kernel_skeleton::pad::emit_smem_zero_pad_predicated;

#[test]
fn smem_zero_pad__rank4_to_16_f16() {
    let mut ptx = String::new();
    emit_smem_zero_pad_predicated(&mut ptx, "%a_tile_base", 4, 16, 16);
    assert_eq!(
        ptx,
        include_str!("snapshots/smem_zero_pad__rank4_to_16_f16.snap"),
        "smem zero-pad (rank 4→16, f16) drift"
    );
}

#[test]
fn smem_zero_pad__rank16_to_16_f16() {
    // No-op path: real_extent == padded_extent → zero instructions emitted.
    let mut ptx = String::new();
    emit_smem_zero_pad_predicated(&mut ptx, "%a_tile_base", 16, 16, 16);
    assert_eq!(
        ptx,
        include_str!("snapshots/smem_zero_pad__rank16_to_16_f16.snap"),
        "smem zero-pad no-op path drift — helper must emit empty string when real==padded"
    );
}

#[test]
#[should_panic(expected = "must be ≤ padded_extent")]
fn smem_zero_pad__real_gt_padded_panics() {
    let mut ptx = String::new();
    emit_smem_zero_pad_predicated(&mut ptx, "%a_tile_base", 20, 16, 16);
}
```

Append to `crates/nsl-codegen/src/kernel_skeleton/tests/mod.rs`:
```rust
mod pad_tests;
```

- [ ] **Step 3: Write the snapshot files**

Create `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/smem_zero_pad__rank4_to_16_f16.snap`:
```
    // Zero-pad SMEM region [4, 16) at 16-bit granularity
    st.shared.b16 [%a_tile_base + 8], 0;
    st.shared.b16 [%a_tile_base + 10], 0;
    st.shared.b16 [%a_tile_base + 12], 0;
    st.shared.b16 [%a_tile_base + 14], 0;
    st.shared.b16 [%a_tile_base + 16], 0;
    st.shared.b16 [%a_tile_base + 18], 0;
    st.shared.b16 [%a_tile_base + 20], 0;
    st.shared.b16 [%a_tile_base + 22], 0;
    st.shared.b16 [%a_tile_base + 24], 0;
    st.shared.b16 [%a_tile_base + 26], 0;
    st.shared.b16 [%a_tile_base + 28], 0;
    st.shared.b16 [%a_tile_base + 30], 0;
```

Create `crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/smem_zero_pad__rank16_to_16_f16.snap` as an **empty file** (literally zero bytes). This is the no-op snapshot.

- [ ] **Step 4: Run pad snapshot tests, verify green**

Run:
```bash
cargo test -p nsl-codegen --lib kernel_skeleton::tests::pad_tests
```
Expected: 3 passed (2 snapshot + 1 panic).

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/kernel_skeleton/pad.rs \
        crates/nsl-codegen/src/kernel_skeleton/tests/pad_tests.rs \
        crates/nsl-codegen/src/kernel_skeleton/tests/mod.rs \
        crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/smem_zero_pad__rank4_to_16_f16.snap \
        crates/nsl-codegen/src/kernel_skeleton/tests/snapshots/smem_zero_pad__rank16_to_16_f16.snap
git commit -m "feat(kernel_skeleton): add emit_smem_zero_pad_predicated (new helper) + snapshots"
```

### Task A8: Full FA v2 snapshot regression sweep + Task Group A marker commit

**Files:** (none created; verification only)

- [ ] **Step 1: Run the FULL nsl-codegen test suite**

Run:
```bash
cargo test -p nsl-codegen --lib 2>&1 | tail -20
cargo test -p nsl-codegen --test fa_v2_snapshots 2>&1 | tail -20
```
Expected: both pass with NO snapshot diffs anywhere.

- [ ] **Step 2: Run the wider workspace tests to catch ABI regressions**

Run:
```bash
cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence -- --test-threads=1
```
Expected: 7 non-ignored pass, 1 ignored (unchanged from baseline).

- [ ] **Step 3: Inspect remaining FA v2 prolog lines that could still be extracted**

Re-read `crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs` and confirm the remaining FA-specific content (param block with 30+ params, CSHA conditional register blocks, rope_q register blocks, Tier C save_activations blocks, block-index routing math) is genuinely FA-specific and was correctly NOT extracted. If any block is discovered to be shareable, add a new task to extract it BEFORE commit; otherwise proceed.

- [ ] **Step 4: Write the EXTRACTION_INVENTORY.md follow-up entry**

Append to `crates/nsl-codegen/src/kernel_skeleton/EXTRACTION_INVENTORY.md`:
```markdown
## Extraction completed on <commit sha>

Extracted helpers (with snapshot coverage):
- emit_ptx_header ×2 variants
- emit_static_smem_decl ×2 variants (1536 B, 768 B)
- emit_dynamic_smem_extern
- emit_shmem_base_cvta
- emit_thread_lane_warp_registers
- emit_param_block
- emit_ld_param_u64
- emit_ld_param_f32
- emit_smem_zero_pad_predicated ×2 variants (4→16, 16→16 no-op)

NOT extracted (FA-specific, stays in prelude.rs):
- 30+ param declaration block (FA's specific FFI-stable list)
- CSHA conditional register blocks (fused_projections, save_activations, fused_output_proj)
- rope_q register block
- FA's %rd0..%rd9 parameter-register loading (different scheme from WRGA)
- block-index routing math (q_start = bid_x * block_q; head_idx = bid_y % heads)
```

- [ ] **Step 5: Commit the Task Group A closing marker**

```bash
git add crates/nsl-codegen/src/kernel_skeleton/EXTRACTION_INVENTORY.md
git commit -m "docs(kernel_skeleton): record extraction completion; FA snapshot suite byte-identical"
```

**Task Group A complete.** All FA v2 snapshots byte-identical; skeleton has 9 snapshot tests covering 7 helpers across variants.

---

## Task Group B — Commit 2: WRGA LoRA ptxas unit test (red)

### Task B1: Add ptxas validation infrastructure

**Files:**
- Create: `crates/nsl-codegen/src/ptxas_validation.rs`
- Modify: `crates/nsl-codegen/src/lib.rs`

- [ ] **Step 1: Write the validation helper**

Create `crates/nsl-codegen/src/ptxas_validation.rs`:
```rust
//! PTX validation helper for unit tests.
//!
//! Uses `cudarc::driver::sys::cuModuleLoadData` when a CUDA device is
//! available; falls back to shelling out to `nvcc --ptx` otherwise.
//! Returns `Ok(())` when PTX is accepted, `Err(String)` with details
//! when ptxas rejects it.

use std::ffi::CString;

/// Validate that the given PTX string compiles.  Returns Ok on success,
/// Err with ptxas's error message on failure.
pub fn validate_ptx(ptx: &str) -> Result<(), String> {
    // Prefer cudarc when feature is enabled and device is available.
    #[cfg(feature = "cuda")]
    {
        if let Some(res) = try_validate_via_cudarc(ptx) {
            return res;
        }
    }
    // Fallback: nvcc --ptx (compile-only, no device required).
    validate_via_nvcc(ptx)
}

#[cfg(feature = "cuda")]
fn try_validate_via_cudarc(ptx: &str) -> Option<Result<(), String>> {
    use cudarc::driver::sys;
    unsafe {
        // Ensure a context is current; cuCtxGetCurrent returns 0 and
        // *pctx = NULL when none is set.  If none, skip — nvcc fallback
        // will handle it.
        let mut ctx: sys::CUcontext = std::ptr::null_mut();
        let _ = sys::cuCtxGetCurrent(&mut ctx);
        if ctx.is_null() {
            return None;
        }
        let ptx_c = CString::new(ptx).ok()?;
        let mut module: sys::CUmodule = std::ptr::null_mut();
        let rc = sys::cuModuleLoadData(&mut module, ptx_c.as_ptr() as *const _);
        if rc == sys::CUresult::CUDA_SUCCESS {
            let _ = sys::cuModuleUnload(module);
            Some(Ok(()))
        } else {
            let mut name_ptr: *const std::os::raw::c_char = std::ptr::null();
            sys::cuGetErrorString(rc, &mut name_ptr);
            let msg = if !name_ptr.is_null() {
                std::ffi::CStr::from_ptr(name_ptr).to_string_lossy().into_owned()
            } else {
                format!("cudarc rc={:?}", rc)
            };
            Some(Err(format!("cuModuleLoadData rejected PTX: {}", msg)))
        }
    }
}

fn validate_via_nvcc(ptx: &str) -> Result<(), String> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    // Write PTX to a temp file.
    let tmp = std::env::temp_dir().join(format!("nsl_ptx_{}.ptx", std::process::id()));
    {
        let mut f = std::fs::File::create(&tmp).map_err(|e| format!("tmp create: {}", e))?;
        f.write_all(ptx.as_bytes()).map_err(|e| format!("tmp write: {}", e))?;
    }
    // Run `nvcc --ptx` in compile-only mode against the file.
    // `-arch=sm_80` matches WRGA; FA snapshots that hit this path (if any)
    // would pass `-arch=sm_75`.  We infer arch from the `.target` line.
    let arch = if ptx.contains(".target sm_80") {
        "sm_80"
    } else if ptx.contains(".target sm_75") {
        "sm_75"
    } else {
        "sm_80"
    };
    let out = Command::new("nvcc")
        .args(&["--gpu-architecture", arch, "--ptx", "--compile-only", "-o"])
        .arg(std::env::temp_dir().join(format!("nsl_ptx_{}.cubin", std::process::id())))
        .arg(&tmp)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output();
    let _ = std::fs::remove_file(&tmp);
    let output = match out {
        Ok(o) => o,
        Err(e) => return Err(format!("nvcc not available: {}", e)),
    };
    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        Err(format!("nvcc rejected PTX: {}", stderr))
    }
}
```

- [ ] **Step 2: Wire into crate**

Modify `crates/nsl-codegen/src/lib.rs` — add:
```rust
pub mod ptxas_validation;
```
alphabetically in the module block.

- [ ] **Step 3: Build and confirm compilation**

Run:
```bash
cargo build -p nsl-codegen --features cuda
cargo build -p nsl-codegen
```
Expected: both clean builds. The `#[cfg(feature = "cuda")]` block only compiles under the cuda feature; the nvcc fallback compiles unconditionally.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/ptxas_validation.rs crates/nsl-codegen/src/lib.rs
git commit -m "feat(codegen): add ptxas_validation module (cudarc + nvcc fallback)"
```

### Task B2: Write LoRA ptxas validation test with 6 configs (red)

**Files:**
- Create: `crates/nsl-codegen/tests/wrga_fused_ptx_ptxas.rs`

- [ ] **Step 1: Write the validation test**

Create `crates/nsl-codegen/tests/wrga_fused_ptx_ptxas.rs`:
```rust
//! Unit-level ptxas validation for synthesize_fused_lora_ptx and
//! synthesize_fused_ia3_ptx.  Each config's output is fed to
//! cuModuleLoadData (or nvcc --ptx fallback); the test asserts
//! acceptance.
//!
//! This test is the gate that B.3 lacked — string-pattern tests let
//! pseudocode through.  Real PTX validation catches invalid syntax,
//! missing .shared decls, uninitialized registers, operand-type
//! mismatches, .param-as-operand bugs, etc.

use nsl_codegen::ptxas_validation::validate_ptx;
use nsl_codegen::wrga_fused_ptx::{synthesize_fused_lora_ptx, FusedLoraConfig};

fn lora_cfg(m: u32, n: u32, k: u32, rank: u32) -> FusedLoraConfig {
    FusedLoraConfig {
        site_id: format!("test.m{}n{}k{}r{}", m, n, k, rank),
        m,
        n,
        k,
        rank,
        target_sm: 80,
    }
}

fn assert_lora_ptx_valid(cfg: FusedLoraConfig) {
    let ptx = synthesize_fused_lora_ptx(&cfg);
    match validate_ptx(&ptx) {
        Ok(()) => {}
        Err(msg) => panic!(
            "LoRA PTX rejected for config (m={}, n={}, k={}, rank={}):\n{}\n\nEmitted PTX:\n{}",
            cfg.m, cfg.n, cfg.k, cfg.rank, msg, ptx
        ),
    }
}

#[test]
fn lora_ptx_validates__16_8_16_16() {
    assert_lora_ptx_valid(lora_cfg(16, 8, 16, 16));
}

#[test]
fn lora_ptx_validates__16_8_32_4() {
    assert_lora_ptx_valid(lora_cfg(16, 8, 32, 4));
}

#[test]
fn lora_ptx_validates__1_8_8_2() {
    assert_lora_ptx_valid(lora_cfg(1, 8, 8, 2));
}

#[test]
fn lora_ptx_validates__4_8_8_2() {
    // BUILD4_SRC_GPU hardening-test shape.
    assert_lora_ptx_valid(lora_cfg(4, 8, 8, 2));
}

#[test]
fn lora_ptx_validates__32_16_64_8() {
    assert_lora_ptx_valid(lora_cfg(32, 16, 64, 8));
}

#[test]
fn lora_ptx_validates__16_8_8_16() {
    // rank == MMA k: exercises emit_smem_zero_pad_predicated no-op path.
    assert_lora_ptx_valid(lora_cfg(16, 8, 8, 16));
}
```

- [ ] **Step 2: Run the test, confirm red**

Run:
```bash
cargo test -p nsl-codegen --test wrga_fused_ptx_ptxas lora_ptx_validates -- --nocapture 2>&1 | tail -40
```
Expected: **all 6 tests fail.** The current pseudocode PTX doesn't pass ptxas — missing `.shared`, uninitialized registers, etc. This is the correct state at end of Task B2 — it proves the gate catches the bug.

If any test PASSES unexpectedly, investigate — either the validate_ptx helper silently succeeded (e.g., neither cudarc nor nvcc available and fallback returned Ok by mistake) or the pseudocode surprisingly compiles under some flag. Fix the validation harness before continuing.

- [ ] **Step 3: Commit the red test**

```bash
git add crates/nsl-codegen/tests/wrga_fused_ptx_ptxas.rs
git commit -m "test(wrga): ptxas validation unit test for fused LoRA (red against pseudocode)"
```

---

## Task Group C — Commit 3: Rewrite `synthesize_fused_lora_ptx`

### Task C1: WRGA-specific helpers for output coords and register pool

**Files:**
- Create: `crates/nsl-codegen/src/wrga_kernel_helpers.rs`
- Modify: `crates/nsl-codegen/src/lib.rs`

- [ ] **Step 1: Write the WRGA-specific helper module**

Create `crates/nsl-codegen/src/wrga_kernel_helpers.rs`:
```rust
//! WRGA-specific PTX helpers that are not shareable with FA.
//!
//! Tile-staging, register-pool emission, and output-tile coord math.
//! Kept out of `kernel_skeleton/` because these are WRGA-kernel-specific
//! (single-warp-per-tile, m16n8k16 fixed, fp32 output) and would drag
//! FA-irrelevant config surface into the skeleton.

use crate::wrga_fused_ptx::FusedLoraConfig;

/// Register pool spec for WRGA fused LoRA kernel.
#[derive(Debug, Clone)]
pub struct LoraRegisterBudget {
    pub main_accum_count: u32,   // 8 (f32, x@W accumulator)
    pub epi_interm_count: u32,   // 4 (f32, x@A accumulator)
    pub epi_final_count: u32,    // 4 (f32, (x@A)@B accumulator)
    pub main_a_frag_count: u32,  // 4 (b32, x fragment)
    pub main_b_frag_count: u32,  // 2 (b32, W fragment)
    pub epi_a_frag_count: u32,   // 4 (b32, A fragment)
    pub epi_b_frag_count: u32,   // 2 (b32, B fragment)
    pub rd_scratch: u32,         // ~16 (u64)
    pub u32_scratch: u32,        // ~12
    pub pred_count: u32,         // ~4
}

pub fn wrga_lora_register_budget(_cfg: &FusedLoraConfig) -> LoraRegisterBudget {
    LoraRegisterBudget {
        main_accum_count: 8,
        epi_interm_count: 4,
        epi_final_count: 4,
        main_a_frag_count: 4,
        main_b_frag_count: 2,
        epi_a_frag_count: 4,
        epi_b_frag_count: 2,
        rd_scratch: 16,
        u32_scratch: 12,
        pred_count: 4,
    }
}

/// Emit the WRGA-LoRA register pool declarations.  Callers pass the
/// output of `wrga_lora_register_budget`.
pub fn emit_lora_register_pool(ptx: &mut String, b: &LoraRegisterBudget) {
    ptx.push_str(&format!("    .reg .f32 %main_accum<{}>;\n", b.main_accum_count));
    ptx.push_str(&format!("    .reg .f32 %epi_interm<{}>;\n", b.epi_interm_count));
    ptx.push_str(&format!("    .reg .f32 %epi_final<{}>;\n", b.epi_final_count));
    ptx.push_str(&format!("    .reg .b32 %main_a_frag<{}>;\n", b.main_a_frag_count));
    ptx.push_str(&format!("    .reg .b32 %main_b_frag<{}>;\n", b.main_b_frag_count));
    ptx.push_str(&format!("    .reg .b32 %epi_a_frag<{}>;\n", b.epi_a_frag_count));
    ptx.push_str(&format!("    .reg .b32 %epi_b_frag<{}>;\n", b.epi_b_frag_count));
    ptx.push_str(&format!("    .reg .u64 %rd<{}>;\n", b.rd_scratch));
    ptx.push_str(&format!("    .reg .u32 %r<{}>;\n", b.u32_scratch));
    ptx.push_str(&format!("    .reg .pred %p<{}>;\n", b.pred_count));
    // Named pointer registers for clarity (also in %rd<N> pool above).
    ptx.push_str("    .reg .u64 %rd_x, %rd_w, %rd_a, %rd_b, %rd_y;\n");
    ptx.push_str("    .reg .u64 %x_tile_base, %w_tile_base, %a_tile_base, %b_tile_base;\n");
    ptx.push_str("    .reg .u64 %shmem_base;\n");
    ptx.push_str("    .reg .f32 %scale_reg;\n");
    // Lane-derived addressing registers (load-bearing for matmul_mma helpers).
    ptx.push_str("    .reg .u32 %mma_a_row, %mma_b_row, %mma_addr;\n");
    ptx.push_str("    .reg .u64 %smem_base_x, %smem_base_w, %smem_base_a, %smem_base_b;\n");
    // Output tile coord.
    ptx.push_str("    .reg .u32 %row_base, %col_base;\n");
    ptx.push_str("    .reg .u32 %m_real, %n_real;\n");
}

/// Emit the WRGA output-tile coord computation:
///   %row_base = bid_x * 16
///   %col_base = bid_y * 8
///   %m_real = min(m - row_base, 16)   // for tail-predication in store
///   %n_real = min(n - col_base, 8)    // for tail-predication in store
pub fn emit_lora_output_tile_coords(ptx: &mut String, m: u32, n: u32) {
    ptx.push_str("    // WRGA output tile coords\n");
    ptx.push_str("    shl.b32 %row_base, %bid_x, 4;     // * 16\n");
    ptx.push_str("    shl.b32 %col_base, %bid_y, 3;     // * 8\n");
    // m_real and n_real computed as compile-time constants for this kernel.
    // The grid/block launcher ensures grid = ((m+15)/16, (n+7)/8, 1), so
    // per-CTA real extents are known given bid_x/bid_y.  We compute at
    // runtime to handle the last-tile tail generically.
    ptx.push_str(&format!("    mov.u32 %r0, {};\n", m));
    ptx.push_str("    sub.u32 %r0, %r0, %row_base;  // m - row_base\n");
    ptx.push_str("    min.u32 %m_real, %r0, 16;\n");
    ptx.push_str(&format!("    mov.u32 %r1, {};\n", n));
    ptx.push_str("    sub.u32 %r1, %r1, %col_base;  // n - col_base\n");
    ptx.push_str("    min.u32 %n_real, %r1, 8;\n");
}

/// Emit the lane-derivation math that initializes `%mma_a_row` and
/// `%mma_b_row` per the m16n8k16 lane layout spec.  Called ONCE in the
/// prolog before any matmul_mma helper.
///
/// For m16n8k16 with a single warp per tile:
///   A-fragment is 16x16 f16, distributed across 32 lanes as 8 b32 per
///   thread (4 registers × 2 f16 pairs each).  Thread t loads:
///     row = t / 4
///     col_pair = (t % 4) * 2  (each thread owns 8 consecutive f16 cols)
///   For row-major A in SMEM with stride row_stride_bytes, thread t reads
///   from offset (row * row_stride_bytes + col_pair * 2).
///
///   We store row into %mma_a_row.  The matmul_mma helper adds col
///   offsets as it emits the 4 ld.shared.b32 per thread.
///
///   B-fragment for m16n8k16 col-major (k16 rows × n8 cols):
///     Thread t owns 4 consecutive f16 rows (k) × 1 col (n).  Thread
///     t's (row, col) is (t % 8 * 4, t / 8 * ?)... simplified:
///     row = (t % 8) * 4
///     col = t / 8
///     Store "row" into %mma_b_row for B's per-thread row-index.
pub fn emit_matmul_mma_lane_init(ptx: &mut String) {
    ptx.push_str("    // matmul_mma lane-index init (m16n8k16 layout)\n");
    // A-fragment row: tid / 4
    ptx.push_str("    shr.u32 %mma_a_row, %tid_x, 2;\n");
    // B-fragment row: (tid % 8) * 4
    ptx.push_str("    and.b32 %r2, %tid_x, 7;     // tid % 8\n");
    ptx.push_str("    shl.b32 %mma_b_row, %r2, 2; // * 4\n");
}
```

- [ ] **Step 2: Wire into crate**

Modify `crates/nsl-codegen/src/lib.rs` — add:
```rust
pub mod wrga_kernel_helpers;
```
alphabetically.

- [ ] **Step 3: Build and confirm**

Run:
```bash
cargo build -p nsl-codegen
```
Expected: clean build.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/wrga_kernel_helpers.rs crates/nsl-codegen/src/lib.rs
git commit -m "feat(wrga): WRGA-specific PTX helpers (register budget, output coords, lane init)"
```

### Task C2: Tile-staging helpers (x, w, a, b)

**Files:**
- Modify: `crates/nsl-codegen/src/wrga_kernel_helpers.rs`

- [ ] **Step 1: Append tile-staging helpers**

Append to `crates/nsl-codegen/src/wrga_kernel_helpers.rs`:
```rust
/// SMEM byte offsets within the shmem[1536] array for LoRA kernel:
///   [   0, 512) — x_tile  (16×16 f16)
///   [ 512, 768) — w_tile  (16×8  f16)
///   [ 768,1280) — a_tile  (16×16 f16, rank-padded)
///   [1280,1536) — b_tile  (16×8  f16, rank-padded)
pub const LORA_X_TILE_OFFSET: u32 = 0;
pub const LORA_W_TILE_OFFSET: u32 = 512;
pub const LORA_A_TILE_OFFSET: u32 = 768;
pub const LORA_B_TILE_OFFSET: u32 = 1280;

/// Initialize the per-tile SMEM base registers from %shmem_base.
/// Called ONCE in the prolog after `emit_shmem_base_cvta`.
pub fn emit_lora_tile_bases(ptx: &mut String) {
    ptx.push_str(&format!("    add.u64 %x_tile_base, %shmem_base, {};\n", LORA_X_TILE_OFFSET));
    ptx.push_str(&format!("    add.u64 %w_tile_base, %shmem_base, {};\n", LORA_W_TILE_OFFSET));
    ptx.push_str(&format!("    add.u64 %a_tile_base, %shmem_base, {};\n", LORA_A_TILE_OFFSET));
    ptx.push_str(&format!("    add.u64 %b_tile_base, %shmem_base, {};\n", LORA_B_TILE_OFFSET));
    // The matmul_mma helpers look up %smem_base_x/w/a/b by name — alias.
    ptx.push_str("    mov.u64 %smem_base_x, %x_tile_base;\n");
    ptx.push_str("    mov.u64 %smem_base_w, %w_tile_base;\n");
    ptx.push_str("    mov.u64 %smem_base_a, %a_tile_base;\n");
    ptx.push_str("    mov.u64 %smem_base_b, %b_tile_base;\n");
}

/// Stage the x_tile slice for K-iteration `k_tile` from global (%rd_x)
/// into SMEM (%x_tile_base).  Handles k-tail predication when
/// `k_remaining < 16` by zeroing OOB rows via the pad helper.
///
/// Emits one ld.global + st.shared per thread covering the 16×16 tile
/// cooperatively (32 threads × 8 f16 per thread = 16×16 / 32 * 2 = 16
/// f16 per thread, 8 b32 per thread).
///
/// For simplicity of this first rewrite, staging is serialized
/// (one thread does all the loads) — this is provably correct but slow.
/// Cooperative staging is a follow-up perf pass.
pub fn emit_lora_stage_x_tile(ptx: &mut String, k_tile: u32, k_remaining: u32, m: u32, k: u32) {
    ptx.push_str(&format!("    // Stage x_tile for K-tile {}\n", k_tile));
    // Single-thread staging under predication: if %tid_x == 0, do all loads.
    ptx.push_str("    setp.eq.u32 %p0, %tid_x, 0;\n");
    ptx.push_str("    @!%p0 bra lora_x_stage_done_$K;\n".replace("$K", &k_tile.to_string()));
    // Compute base pointers in x: x_gl_base = rd_x + (row_base * k + k_tile*16) * 2 bytes
    // (f16 = 2 bytes).  Per-row offset = row * k * 2.
    let k_start = k_tile * 16;
    for row in 0..m.min(16) {
        for col_pair in 0..8u32 {
            let col = col_pair * 2;
            if k_start + col >= k {
                // OOB on k — store zero.
                let smem_offset = row * 32 + col_pair * 4;
                ptx.push_str(&format!(
                    "    st.shared.b32 [%x_tile_base + {}], 0;\n",
                    smem_offset
                ));
            } else {
                let gl_offset = (row as u64) * (k as u64) * 2 + (k_start as u64 + col as u64) * 2;
                let smem_offset = row * 32 + col_pair * 4;
                ptx.push_str(&format!(
                    "    ld.global.b32 %r3, [%rd_x + {}];\n",
                    gl_offset
                ));
                ptx.push_str(&format!(
                    "    st.shared.b32 [%x_tile_base + {}], %r3;\n",
                    smem_offset
                ));
            }
        }
    }
    // Zero remaining rows if m < 16.
    for row in m.min(16)..16 {
        for col_pair in 0..8u32 {
            let smem_offset = row * 32 + col_pair * 4;
            ptx.push_str(&format!(
                "    st.shared.b32 [%x_tile_base + {}], 0;\n",
                smem_offset
            ));
        }
    }
    ptx.push_str(&format!("lora_x_stage_done_{}:\n", k_tile));
    // Suppress unused-warning pattern.
    let _ = k_remaining;
}

/// Stage the w_tile slice for K-iteration `k_tile`.  Shape: 16×8 f16.
pub fn emit_lora_stage_w_tile(ptx: &mut String, k_tile: u32, n: u32, k: u32) {
    ptx.push_str(&format!("    // Stage w_tile for K-tile {}\n", k_tile));
    ptx.push_str("    setp.eq.u32 %p0, %tid_x, 0;\n");
    ptx.push_str(&format!("    @!%p0 bra lora_w_stage_done_{};\n", k_tile));
    let k_start = k_tile * 16;
    for row in 0..16u32 {
        for col_pair in 0..4u32 {
            let col = col_pair * 2;
            if k_start + row >= k {
                let smem_offset = row * 16 + col_pair * 4;
                ptx.push_str(&format!(
                    "    st.shared.b32 [%w_tile_base + {}], 0;\n",
                    smem_offset
                ));
                continue;
            }
            if col >= n {
                let smem_offset = row * 16 + col_pair * 4;
                ptx.push_str(&format!(
                    "    st.shared.b32 [%w_tile_base + {}], 0;\n",
                    smem_offset
                ));
                continue;
            }
            let gl_offset = ((k_start + row) as u64) * (n as u64) * 2 + (col as u64) * 2;
            let smem_offset = row * 16 + col_pair * 4;
            ptx.push_str(&format!(
                "    ld.global.b32 %r3, [%rd_w + {}];\n",
                gl_offset
            ));
            ptx.push_str(&format!(
                "    st.shared.b32 [%w_tile_base + {}], %r3;\n",
                smem_offset
            ));
        }
    }
    ptx.push_str(&format!("lora_w_stage_done_{}:\n", k_tile));
}

/// Stage the a_tile slice for K-iteration `k_tile`.  A shape: [k, rank].
/// a_tile layout in SMEM: 16 rows × 16 cols (rank-padded to 16), f16.
pub fn emit_lora_stage_a_tile(ptx: &mut String, k_tile: u32, rank: u32, k: u32) {
    ptx.push_str(&format!("    // Stage a_tile for K-tile {} (rank={})\n", k_tile, rank));
    ptx.push_str("    setp.eq.u32 %p0, %tid_x, 0;\n");
    ptx.push_str(&format!("    @!%p0 bra lora_a_stage_done_{};\n", k_tile));
    let k_start = k_tile * 16;
    for row in 0..16u32 {
        for col_pair in 0..8u32 {
            let col = col_pair * 2;
            if k_start + row >= k {
                let smem_offset = row * 32 + col_pair * 4;
                ptx.push_str(&format!(
                    "    st.shared.b32 [%a_tile_base + {}], 0;\n",
                    smem_offset
                ));
                continue;
            }
            if col >= rank {
                // Rank-pad region — zero.
                let smem_offset = row * 32 + col_pair * 4;
                ptx.push_str(&format!(
                    "    st.shared.b32 [%a_tile_base + {}], 0;\n",
                    smem_offset
                ));
                continue;
            }
            let gl_offset = ((k_start + row) as u64) * (rank as u64) * 2 + (col as u64) * 2;
            let smem_offset = row * 32 + col_pair * 4;
            ptx.push_str(&format!(
                "    ld.global.b32 %r3, [%rd_a + {}];\n",
                gl_offset
            ));
            ptx.push_str(&format!(
                "    st.shared.b32 [%a_tile_base + {}], %r3;\n",
                smem_offset
            ));
        }
    }
    ptx.push_str(&format!("lora_a_stage_done_{}:\n", k_tile));
}

/// Stage the b_tile ONCE post-loop.  B shape: [rank, n].  Staged as
/// 16×8 f16 (rank-padded to 16).
pub fn emit_lora_stage_b_tile(ptx: &mut String, rank: u32, n: u32) {
    ptx.push_str("    // Stage b_tile (post-loop, once)\n");
    ptx.push_str("    setp.eq.u32 %p0, %tid_x, 0;\n");
    ptx.push_str("    @!%p0 bra lora_b_stage_done;\n");
    for row in 0..16u32 {
        for col_pair in 0..4u32 {
            let col = col_pair * 2;
            if row >= rank {
                let smem_offset = row * 16 + col_pair * 4;
                ptx.push_str(&format!(
                    "    st.shared.b32 [%b_tile_base + {}], 0;\n",
                    smem_offset
                ));
                continue;
            }
            if col >= n {
                let smem_offset = row * 16 + col_pair * 4;
                ptx.push_str(&format!(
                    "    st.shared.b32 [%b_tile_base + {}], 0;\n",
                    smem_offset
                ));
                continue;
            }
            let gl_offset = (row as u64) * (n as u64) * 2 + (col as u64) * 2;
            let smem_offset = row * 16 + col_pair * 4;
            ptx.push_str(&format!(
                "    ld.global.b32 %r3, [%rd_b + {}];\n",
                gl_offset
            ));
            ptx.push_str(&format!(
                "    st.shared.b32 [%b_tile_base + {}], %r3;\n",
                smem_offset
            ));
        }
    }
    ptx.push_str("lora_b_stage_done:\n");
}
```

- [ ] **Step 2: Build**

Run:
```bash
cargo build -p nsl-codegen
```
Expected: clean build.

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-codegen/src/wrga_kernel_helpers.rs
git commit -m "feat(wrga): tile-staging helpers (x/w/a/b) with k-tail and rank-pad handling"
```

### Task C3: Accumulator init and predicated store helpers

**Files:**
- Modify: `crates/nsl-codegen/src/wrga_kernel_helpers.rs`

- [ ] **Step 1: Append accumulator + store helpers**

Append to `crates/nsl-codegen/src/wrga_kernel_helpers.rs`:
```rust
/// Emit zero-init of N floating-point accumulator registers named
/// `%<base>0..%<base>(N-1)`.
pub fn emit_zero_accumulators(ptx: &mut String, base: &str, count: u32) {
    for i in 0..count {
        ptx.push_str(&format!("    mov.f32 %{}{}, 0f00000000;\n", base, i));
    }
}

/// Emit the output-tile store with predication on m_real and n_real.
/// The main_accum<8> registers hold a 16×8 f32 output tile (distributed
/// across 32 lanes as 4 f32 per thread at 4 fragment rows).
///
/// Per-thread store layout for m16n8k16 D-fragment (f32):
///   Thread t owns 2 consecutive rows × 2 cols out of the 16×8 tile.
///   Row pair: (t / 4) * 2 .. +2
///   Col pair: (t % 4) * 2 .. +2
///   4 f32 registers per thread: main_accum0 = row+0 col+0
///                               main_accum1 = row+0 col+1
///                               main_accum2 = row+1 col+0
///                               main_accum3 = row+1 col+1
///   (main_accum4..7 are reserved for potential multi-tile future use;
///    unused here — still emitted as registers to match kernel launcher
///    register budget.)
///
/// Global y offset for thread t, element i:
///   y[row_base + (t/4)*2 + (i/2), col_base + (t%4)*2 + (i%2)]
///   = rd_y + ((row_base + (t/4)*2 + i/2) * n + col_base + (t%4)*2 + i%2) * 4
pub fn emit_lora_store_output(ptx: &mut String, n: u32) {
    ptx.push_str("    // Store main_accum to y with m_real/n_real predication\n");
    // Compute per-thread (row_in_tile, col_in_tile)
    ptx.push_str("    shr.u32 %r4, %tid_x, 2;        // t / 4\n");
    ptx.push_str("    shl.b32 %r4, %r4, 1;           // * 2\n");  // r4 = row_in_tile_base
    ptx.push_str("    and.b32 %r5, %tid_x, 3;        // t % 4\n");
    ptx.push_str("    shl.b32 %r5, %r5, 1;           // * 2\n");  // r5 = col_in_tile_base
    // For each of 4 main_accum registers (i=0..3), compute (row_in_tile, col_in_tile)
    // and conditionally store.
    for i in 0..4u32 {
        let row_offset_in_tile = i / 2;
        let col_offset_in_tile = i % 2;
        ptx.push_str(&format!("    // Store main_accum{} (row+{}, col+{})\n", i, row_offset_in_tile, col_offset_in_tile));
        // row_tile = r4 + row_offset; col_tile = r5 + col_offset
        ptx.push_str(&format!("    add.u32 %r6, %r4, {};   // row_in_tile\n", row_offset_in_tile));
        ptx.push_str(&format!("    add.u32 %r7, %r5, {};   // col_in_tile\n", col_offset_in_tile));
        // Predicate: row_in_tile < m_real AND col_in_tile < n_real
        ptx.push_str("    setp.lt.u32 %p1, %r6, %m_real;\n");
        ptx.push_str("    setp.lt.and.u32 %p1, %r7, %n_real, %p1;\n");
        // Global offset: ((row_base + row_in_tile) * n + col_base + col_in_tile) * 4
        ptx.push_str("    add.u32 %r8, %row_base, %r6;  // global row\n");
        ptx.push_str(&format!("    mul.lo.u32 %r8, %r8, {};\n", n));
        ptx.push_str("    add.u32 %r8, %r8, %col_base;\n");
        ptx.push_str("    add.u32 %r8, %r8, %r7;\n");
        ptx.push_str("    shl.b32 %r8, %r8, 2;    // * 4 bytes (f32)\n");
        ptx.push_str("    cvt.u64.u32 %rd0, %r8;\n");
        ptx.push_str("    add.u64 %rd0, %rd_y, %rd0;\n");
        ptx.push_str(&format!("    @%p1 st.global.f32 [%rd0], %main_accum{};\n", i));
    }
}
```

- [ ] **Step 2: Build**

Run:
```bash
cargo build -p nsl-codegen
```
Expected: clean build.

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-codegen/src/wrga_kernel_helpers.rs
git commit -m "feat(wrga): accumulator init + predicated output store helpers"
```

### Task C4: Rewrite `synthesize_fused_lora_ptx` body

**Files:**
- Modify: `crates/nsl-codegen/src/wrga_fused_ptx.rs`

- [ ] **Step 1: Delete the old `synthesize_fused_lora_ptx` body and associated helpers**

Open `crates/nsl-codegen/src/wrga_fused_ptx.rs`. Identify and delete:
- The entire body of `synthesize_fused_lora_ptx` (everything between the opening `{` and closing `}` including the assertion block at top).
- Helper fns `emit_load_frag_a_main`, `emit_load_frag_b_main`, `emit_load_frag_a_epi`, `emit_load_frag_b_epi`, `emit_main_mma`, `emit_epi_interm_mma`, `emit_epi_final_mma`, `emit_store_y` — these are the pseudocode wrappers; the rewrite uses `matmul_mma` primitives directly.

Keep:
- `FusedLoraConfig` struct (public API)
- `FusedIa3Config` struct (public API; IA³ rewrite is Task Group E)
- `LoraKernelKey` struct and `FusedLoraConfig::kernel_key` method (public API)
- `const MMA_K_U32: u32 = 16;`
- The `synthesize_fused_ia3_ptx` function AS-IS for now (will be rewritten in Task Group E)

- [ ] **Step 2: Write the new `synthesize_fused_lora_ptx`**

Replace the deleted body with:
```rust
pub fn synthesize_fused_lora_ptx(config: &FusedLoraConfig) -> String {
    assert!(
        config.rank <= 16,
        "B.3 rank ceiling: {} > 16; multi-pass epilogue is a follow-up milestone",
        config.rank,
    );
    assert!(config.target_sm >= 80, "B.3 requires sm_80+");

    use crate::kernel_skeleton::header::{emit_ptx_header, PtxVersion, TargetSm};
    use crate::kernel_skeleton::indexing::emit_thread_lane_warp_registers;
    use crate::kernel_skeleton::params::{
        emit_ld_param_f32, emit_ld_param_u64, emit_param_block, Param, ParamTy,
    };
    use crate::kernel_skeleton::smem::{emit_shmem_base_cvta, emit_static_smem_decl};
    use crate::matmul_mma::{
        emit_load_a_fragment_smem, emit_load_b_fragment_smem, emit_mma_instruction,
    };
    use crate::wrga_kernel_helpers::{
        emit_lora_output_tile_coords, emit_lora_register_pool, emit_lora_stage_a_tile,
        emit_lora_stage_b_tile, emit_lora_stage_w_tile, emit_lora_stage_x_tile,
        emit_lora_store_output, emit_lora_tile_bases, emit_matmul_mma_lane_init,
        emit_zero_accumulators, wrga_lora_register_budget,
    };

    let mut ptx = String::new();
    let budget = wrga_lora_register_budget(config);

    // 1. Header
    emit_ptx_header(&mut ptx, PtxVersion::V7_0, TargetSm::Sm80);

    // 2. Param block
    let entry_name = format!(
        "nsl_wrga_fused_lora_m{}n{}k{}r{}",
        config.m, config.n, config.k, config.rank,
    );
    let params = [
        Param { ty: ParamTy::U64, name: "x_ptr" },
        Param { ty: ParamTy::U64, name: "w_ptr" },
        Param { ty: ParamTy::U64, name: "a_ptr" },
        Param { ty: ParamTy::U64, name: "b_ptr" },
        Param { ty: ParamTy::F32, name: "scale" },
        Param { ty: ParamTy::U64, name: "y_ptr" },
    ];
    emit_param_block(&mut ptx, &entry_name, &params);

    // 3. SMEM decl + register pool + shmem base cvta + tid/warp/lane init
    emit_static_smem_decl(&mut ptx, 1536);
    emit_lora_register_pool(&mut ptx, &budget);
    emit_shmem_base_cvta(&mut ptx);
    emit_thread_lane_warp_registers(&mut ptx);
    emit_lora_tile_bases(&mut ptx);
    emit_matmul_mma_lane_init(&mut ptx);

    // 4. Load params into named registers
    emit_ld_param_u64(&mut ptx, "%rd_x", "x_ptr");
    emit_ld_param_u64(&mut ptx, "%rd_w", "w_ptr");
    emit_ld_param_u64(&mut ptx, "%rd_a", "a_ptr");
    emit_ld_param_u64(&mut ptx, "%rd_b", "b_ptr");
    emit_ld_param_u64(&mut ptx, "%rd_y", "y_ptr");
    emit_ld_param_f32(&mut ptx, "%scale_reg", "scale");

    // 5. Output-tile coords + zero accumulators
    emit_lora_output_tile_coords(&mut ptx, config.m, config.n);
    emit_zero_accumulators(&mut ptx, "main_accum", 8);
    emit_zero_accumulators(&mut ptx, "epi_interm", 4);

    // 6. Main K-loop with interleaved epilogue
    let k_iters = (config.k + MMA_K_U32 - 1) / MMA_K_U32;
    let main_a_frag: [String; 4] = [
        "main_a_frag0".into(), "main_a_frag1".into(),
        "main_a_frag2".into(), "main_a_frag3".into(),
    ];
    let main_b_frag: [String; 2] = [
        "main_b_frag0".into(), "main_b_frag1".into(),
    ];
    let epi_a_frag: [String; 4] = [
        "epi_a_frag0".into(), "epi_a_frag1".into(),
        "epi_a_frag2".into(), "epi_a_frag3".into(),
    ];
    let epi_b_frag: [String; 2] = [
        "epi_b_frag0".into(), "epi_b_frag1".into(),
    ];
    let main_accum: [String; 4] = [
        "main_accum0".into(), "main_accum1".into(),
        "main_accum2".into(), "main_accum3".into(),
    ];
    let epi_interm: [String; 4] = [
        "epi_interm0".into(), "epi_interm1".into(),
        "epi_interm2".into(), "epi_interm3".into(),
    ];
    let epi_final: [String; 4] = [
        "epi_final0".into(), "epi_final1".into(),
        "epi_final2".into(), "epi_final3".into(),
    ];

    for k_tile in 0..k_iters {
        let k_remaining = (config.k - k_tile * MMA_K_U32).min(MMA_K_U32);
        ptx.push_str(&format!("// ===== K-iteration {} =====\n", k_tile));

        // 6a. Stage x, w, a into SMEM
        emit_lora_stage_x_tile(&mut ptx, k_tile, k_remaining, config.m, config.k);
        emit_lora_stage_w_tile(&mut ptx, k_tile, config.n, config.k);
        emit_lora_stage_a_tile(&mut ptx, k_tile, config.rank, config.k);

        ptx.push_str("    bar.sync 0;\n");

        // 6b. Load fragments
        emit_load_a_fragment_smem(&mut ptx, &main_a_frag, "%x_tile_base", 32);
        emit_load_b_fragment_smem(&mut ptx, &main_b_frag, "%w_tile_base", 16);
        emit_load_a_fragment_smem(&mut ptx, &epi_a_frag, "%a_tile_base", 32);

        // 6c. Main MMA: main_accum += x_tile @ w_tile
        emit_mma_instruction(&mut ptx, &main_accum, &main_a_frag, &main_b_frag, &main_accum);

        // 6d. Epilogue MMA: epi_interm += x_tile @ a_tile
        // INVARIANT: m16n8k16 A-fragments encode only the m×k tile and are
        // independent of the B matrix. %main_a_frag loaded from x_tile in
        // step 6b is valid as the A-operand for BOTH x@W (B=w_tile) and
        // x@A (B=a_tile). Reloading x would double the SMEM read cost and
        // risk lane-alignment regressions. See spec §3 invariant (1).
        emit_mma_instruction(&mut ptx, &epi_interm, &main_a_frag, &epi_a_frag, &epi_interm);

        ptx.push_str("    bar.sync 0;\n");
        let _ = k_remaining;
    }

    // 7. Post-loop: stage b_tile, compute (x@A) @ B
    emit_lora_stage_b_tile(&mut ptx, config.rank, config.n);
    ptx.push_str("    bar.sync 0;\n");
    emit_load_b_fragment_smem(&mut ptx, &epi_b_frag, "%b_tile_base", 16);

    // 8. Zero-init epi_final and compute (x@A)@B
    emit_zero_accumulators(&mut ptx, "epi_final", 4);
    emit_mma_instruction(&mut ptx, &epi_final, &epi_interm, &epi_b_frag, &epi_final);

    // 9. Scale epi_final, fold into main_accum
    for i in 0..4u32 {
        ptx.push_str(&format!(
            "    mul.f32 %epi_final{i}, %epi_final{i}, %scale_reg;\n"
        ));
        ptx.push_str(&format!(
            "    add.f32 %main_accum{i}, %main_accum{i}, %epi_final{i};\n"
        ));
    }

    // 10. Store to y with predication
    emit_lora_store_output(&mut ptx, config.n);

    // 11. Close the entry body
    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");

    ptx
}
```

- [ ] **Step 2: Build**

Run:
```bash
cargo build -p nsl-codegen
```
Expected: clean build. If compile errors, iterate on helper names / signatures.

- [ ] **Step 3: Run the unit ptxas test suite**

Run:
```bash
cargo test -p nsl-codegen --test wrga_fused_ptx_ptxas lora_ptx_validates -- --nocapture 2>&1 | tail -50
```
Expected: **all 6 LoRA ptxas tests PASS.** If any fail, read the error message carefully:
- `Missing declaration for %foo` — a register is declared elsewhere or typo'd; fix the declaration in `emit_lora_register_pool`.
- `Unknown symbol` — a param name is misspelled somewhere; check the param block and `ld.param.*` calls.
- `Instruction 'mma.sync' not supported on .target sm_XX` — target_sm mismatch.
- `Operand type mismatch` — check `.u32` vs `.u64` vs `.b32` in the failing instruction.

- [ ] **Step 4: Run the existing WRGA regression suite to confirm no breakage**

Run:
```bash
cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence -- --test-threads=1
```
Expected: 7 passed, 1 ignored. The non-ignored tests all use the CPU-fallback path, so they should be unaffected by the PTX rewrite. If any fail, investigate — the rewrite must not break B.2.1's runtime infrastructure.

- [ ] **Step 5: Run the internal string-pattern tests for wrga_fused_ptx**

Run:
```bash
cargo test -p nsl-codegen --lib wrga_fused_ptx
```
Expected: most tests pass; some existing tests (`lora_ptx_uses_scale_as_param_not_literal`, `lora_ptx_emits_main_and_epilogue_mmas_per_k_tile`, `lora_ptx_folds_epilogue_into_main_accum`) may still hold structurally, but some may need updating if their string-pattern assertions no longer match the new emission.

If any test fails, audit the test: is its assertion testing a meaningful invariant that the rewrite should preserve, or is it testing the old pseudocode structure incidentally? Keep invariant-preserving tests (update their patterns if needed); delete tests that were coupled to pseudocode-specific strings.

- [ ] **Step 6: Commit the rewrite**

```bash
git add crates/nsl-codegen/src/wrga_fused_ptx.rs
git commit -m "feat(wrga): rewrite synthesize_fused_lora_ptx against kernel_skeleton + matmul_mma"
```

### Task C5: Task Group C marker commit

- [ ] **Step 1: Full regression sweep**

Run:
```bash
cargo test -p nsl-codegen --lib 2>&1 | tail -10
cargo test -p nsl-codegen --test wrga_fused_ptx_ptxas 2>&1 | tail -10
cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence -- --test-threads=1 2>&1 | tail -10
```
Expected: all green. 6 LoRA ptxas tests pass; 7/7 WRGA non-ignored tests pass.

- [ ] **Step 2: (Optional) rebase-squash all Task Group C intermediate commits**

If a clean 6-commit history is preferred for the spec's commit-3 boundary, run:
```bash
git rebase -i HEAD~4   # adjust count to cover C1-C4 commits
# Mark all but first as `squash`; keep message: "feat(wrga): rewrite synthesize_fused_lora_ptx against kernel_skeleton"
```
Otherwise leave intermediate commits in place for bisect granularity.

---

## Task Group D — Commit 4: LoRA integration numerical test

### Task D1: Write `build_4_fused_real_launch`

**Files:**
- Modify: `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs`

- [ ] **Step 1: Add the integration test**

At the end of `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs`, add:
```rust
// ─── Commit 4: build_4_fused_real_launch — integration numerical ────
//
// Same source as build_4_fused (BUILD4_SRC_GPU for GPU residency), but
// asserts the REAL cudarc launch produces the correct output — not the
// CPU-fallback.  NSL_WRGA_FUSED_CUDA=1 forces the launch path; tolerance
// is 1e-4 per spec §5.
#[cfg(feature = "cuda")]
#[test]
fn build_4_fused_real_launch() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("build4_real_launch.nsl");
    fs::write(&src_path, BUILD4_SRC_GPU).unwrap();

    let root = workspace_root();
    let stdlib = root.join("stdlib");
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", &stdlib)
        .env("NSL_WRGA_FUSED_CUDA", "1")
        .env("NSL_WRGA_GPU_LAUNCH_COUNTER", "1")
        .arg("run")
        .args(["--source-ad", "--target", "cuda_sm80"])
        .arg(&src_path);
    let output = cmd.output().expect("nsl run failed to spawn");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "nsl run failed.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        stderr,
    );
    // Parse tensor from stdout.
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let tensor = parse_tensor_2d(&stdout);
    assert_eq!(tensor.len(), 4, "expected 4 rows");
    for row in &tensor {
        assert_eq!(row.len(), 8, "expected 8 cols");
    }
    // Assert 16.0 ± 1e-4 per element (LoRA adapter path produces 16.0 uniformly).
    let mut max_diff: f32 = 0.0;
    for row in &tensor {
        for v in row {
            max_diff = max_diff.max((v - 16.0).abs());
        }
    }
    assert!(
        max_diff < 1e-4,
        "build_4_fused_real_launch: max |y - 16.0| = {max_diff:.3e}, want < 1e-4.\n\
         stderr (check for fallback warnings or launch failures):\n{stderr}"
    );
    // Also verify launch counter >= 1 — if this run used CPU fallback, the
    // counter stays 0 and the numerical assertion above would still pass
    // (CPU fallback also produces 16.0).
    let gpu_count_line = stderr.lines().find(|l| l.contains("[nsl-gpu-launch-count]"));
    let count: u64 = gpu_count_line
        .and_then(|l| l.split_whitespace().next_back())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    assert!(
        count >= 1,
        "build_4_fused_real_launch: expected ≥1 real cudarc launch with \
         NSL_WRGA_FUSED_CUDA=1, got {count}. Numerical output was correct \
         but it came from CPU fallback — the fused PTX path didn't fire.\n\
         stderr:\n{stderr}"
    );
}
```

- [ ] **Step 2: Build and confirm compilation under cuda feature**

Run:
```bash
cargo build -p nsl-cli --features cuda
```
Expected: clean build.

- [ ] **Step 3: Run the test on a CUDA machine**

Run:
```bash
cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence --features cuda build_4_fused_real_launch -- --test-threads=1 --nocapture 2>&1 | tail -40
```
Expected on a CUDA machine: test PASSES. Output shows `max |y - 16.0| < 1e-4` and `[nsl-gpu-launch-count] ≥ 1`.

If test fails with numerical divergence (`max_diff >> 1e-4`): PTX is valid but fragment layout or SMEM offsets are wrong. Debug via:
1. Reduce config to `(1, 8, 8, 2)` to minimize surface.
2. Print accumulator values at key points via additional `st.global.f32` debug stores.
3. Compare per-lane fragment layout against m16n8k16 spec.

If test fails with `count == 0`: the cudarc launch isn't firing. Inspect stderr for fallback reason.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs
git commit -m "test(wrga): build_4_fused_real_launch — integration numerical at 1e-4"
```

---

## Task Group E — Commit 5: Rewrite `synthesize_fused_ia3_ptx` + tests

### Task E1: IA³ ptxas validation test (red)

**Files:**
- Modify: `crates/nsl-codegen/tests/wrga_fused_ptx_ptxas.rs`

- [ ] **Step 1: Add IA³ validation test**

At the end of `crates/nsl-codegen/tests/wrga_fused_ptx_ptxas.rs`, add:
```rust
use nsl_codegen::wrga_fused_ptx::{synthesize_fused_ia3_ptx, FusedIa3Config};

fn ia3_cfg(m: u32, n: u32, k: u32) -> FusedIa3Config {
    FusedIa3Config {
        site_id: format!("test.m{}n{}k{}", m, n, k),
        m, n, k,
        target_sm: 80,
    }
}

fn assert_ia3_ptx_valid(cfg: FusedIa3Config) {
    let ptx = synthesize_fused_ia3_ptx(&cfg);
    match validate_ptx(&ptx) {
        Ok(()) => {}
        Err(msg) => panic!(
            "IA3 PTX rejected for config (m={}, n={}, k={}):\n{}\n\nEmitted PTX:\n{}",
            cfg.m, cfg.n, cfg.k, msg, ptx
        ),
    }
}

#[test]
fn ia3_ptx_validates__16_8_16() { assert_ia3_ptx_valid(ia3_cfg(16, 8, 16)); }

#[test]
fn ia3_ptx_validates__16_8_32() { assert_ia3_ptx_valid(ia3_cfg(16, 8, 32)); }

#[test]
fn ia3_ptx_validates__1_8_8() { assert_ia3_ptx_valid(ia3_cfg(1, 8, 8)); }

#[test]
fn ia3_ptx_validates__4_8_8() { assert_ia3_ptx_valid(ia3_cfg(4, 8, 8)); }

#[test]
fn ia3_ptx_validates__32_16_64() { assert_ia3_ptx_valid(ia3_cfg(32, 16, 64)); }
```

- [ ] **Step 2: Run, confirm red**

Run:
```bash
cargo test -p nsl-codegen --test wrga_fused_ptx_ptxas ia3_ptx_validates -- --nocapture 2>&1 | tail -30
```
Expected: **all 5 IA³ tests fail** — current `synthesize_fused_ia3_ptx` is still pseudocode (Task Group E hasn't rewritten it yet).

- [ ] **Step 3: Commit the red test**

```bash
git add crates/nsl-codegen/tests/wrga_fused_ptx_ptxas.rs
git commit -m "test(wrga): ptxas validation unit test for fused IA3 (red against pseudocode)"
```

### Task E2: Add IA³-specific helpers

**Files:**
- Modify: `crates/nsl-codegen/src/wrga_kernel_helpers.rs`

- [ ] **Step 1: Append IA³ helpers**

Append to `crates/nsl-codegen/src/wrga_kernel_helpers.rs`:
```rust
use crate::wrga_fused_ptx::FusedIa3Config;

pub const IA3_X_TILE_OFFSET: u32 = 0;
pub const IA3_W_TILE_OFFSET: u32 = 512;

#[derive(Debug, Clone)]
pub struct Ia3RegisterBudget {
    pub main_accum_count: u32,   // 8
    pub gamma_count: u32,        // 8
    pub main_a_frag_count: u32,  // 4
    pub main_b_frag_count: u32,  // 2
    pub rd_scratch: u32,         // ~12
    pub u32_scratch: u32,        // ~10
    pub pred_count: u32,         // ~4
}

pub fn wrga_ia3_register_budget(_cfg: &FusedIa3Config) -> Ia3RegisterBudget {
    Ia3RegisterBudget {
        main_accum_count: 8,
        gamma_count: 8,
        main_a_frag_count: 4,
        main_b_frag_count: 2,
        rd_scratch: 12,
        u32_scratch: 10,
        pred_count: 4,
    }
}

pub fn emit_ia3_register_pool(ptx: &mut String, b: &Ia3RegisterBudget) {
    ptx.push_str(&format!("    .reg .f32 %main_accum<{}>;\n", b.main_accum_count));
    ptx.push_str(&format!("    .reg .f32 %gamma<{}>;\n", b.gamma_count));
    ptx.push_str(&format!("    .reg .b32 %main_a_frag<{}>;\n", b.main_a_frag_count));
    ptx.push_str(&format!("    .reg .b32 %main_b_frag<{}>;\n", b.main_b_frag_count));
    ptx.push_str(&format!("    .reg .u64 %rd<{}>;\n", b.rd_scratch));
    ptx.push_str(&format!("    .reg .u32 %r<{}>;\n", b.u32_scratch));
    ptx.push_str(&format!("    .reg .pred %p<{}>;\n", b.pred_count));
    ptx.push_str("    .reg .u64 %rd_x, %rd_w, %rd_gamma, %rd_y;\n");
    ptx.push_str("    .reg .u64 %x_tile_base, %w_tile_base;\n");
    ptx.push_str("    .reg .u64 %shmem_base;\n");
    ptx.push_str("    .reg .u32 %mma_a_row, %mma_b_row, %mma_addr;\n");
    ptx.push_str("    .reg .u64 %smem_base_x, %smem_base_w;\n");
    ptx.push_str("    .reg .u32 %row_base, %col_base;\n");
    ptx.push_str("    .reg .u32 %m_real, %n_real;\n");
}

pub fn emit_ia3_tile_bases(ptx: &mut String) {
    ptx.push_str(&format!("    add.u64 %x_tile_base, %shmem_base, {};\n", IA3_X_TILE_OFFSET));
    ptx.push_str(&format!("    add.u64 %w_tile_base, %shmem_base, {};\n", IA3_W_TILE_OFFSET));
    ptx.push_str("    mov.u64 %smem_base_x, %x_tile_base;\n");
    ptx.push_str("    mov.u64 %smem_base_w, %w_tile_base;\n");
}

/// Emit the γ-load sequence: load 8 f32 values from %rd_gamma
/// (offset by col_base * 4) into %gamma0..%gamma7.
pub fn emit_ia3_load_gamma(ptx: &mut String) {
    ptx.push_str("    // Load γ slice for this output tile\n");
    ptx.push_str("    shl.b32 %r3, %col_base, 2;       // col_base * 4 bytes\n");
    ptx.push_str("    cvt.u64.u32 %rd0, %r3;\n");
    ptx.push_str("    add.u64 %rd0, %rd_gamma, %rd0;\n");
    for i in 0..8u32 {
        ptx.push_str(&format!(
            "    ld.global.f32 %gamma{}, [%rd0 + {}];\n", i, i * 4
        ));
    }
}

/// Emit the γ broadcast-multiply over main_accum.
pub fn emit_ia3_gamma_multiply(ptx: &mut String) {
    ptx.push_str("    // Broadcast-multiply main_accum by γ\n");
    for i in 0..8u32 {
        ptx.push_str(&format!(
            "    mul.f32 %main_accum{i}, %main_accum{i}, %gamma{i};\n"
        ));
    }
}

/// Same store semantics as LoRA but no scale/epilogue fold.
/// main_accum[0..3] stores the 4 real f32 outputs per thread per
/// m16n8k16 D-fragment.
pub fn emit_ia3_store_output(ptx: &mut String, n: u32) {
    // Identical structure to emit_lora_store_output — same D-fragment
    // layout because both kernels use the same m16n8k16 output tile.
    emit_lora_store_output(ptx, n);
}
```

- [ ] **Step 2: Build**

Run:
```bash
cargo build -p nsl-codegen
```
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-codegen/src/wrga_kernel_helpers.rs
git commit -m "feat(wrga): IA3-specific kernel helpers (register pool, γ load, γ multiply)"
```

### Task E3: Rewrite `synthesize_fused_ia3_ptx` body

**Files:**
- Modify: `crates/nsl-codegen/src/wrga_fused_ptx.rs`

- [ ] **Step 1: Replace the IA³ body**

Locate `synthesize_fused_ia3_ptx` in `crates/nsl-codegen/src/wrga_fused_ptx.rs`. Delete the entire body between `{` and `}`. Replace with:
```rust
pub fn synthesize_fused_ia3_ptx(config: &FusedIa3Config) -> String {
    assert!(config.target_sm >= 80, "B.3 requires sm_80+");

    use crate::kernel_skeleton::header::{emit_ptx_header, PtxVersion, TargetSm};
    use crate::kernel_skeleton::indexing::emit_thread_lane_warp_registers;
    use crate::kernel_skeleton::params::{
        emit_ld_param_u64, emit_param_block, Param, ParamTy,
    };
    use crate::kernel_skeleton::smem::{emit_shmem_base_cvta, emit_static_smem_decl};
    use crate::matmul_mma::{
        emit_load_a_fragment_smem, emit_load_b_fragment_smem, emit_mma_instruction,
    };
    use crate::wrga_kernel_helpers::{
        emit_ia3_gamma_multiply, emit_ia3_load_gamma, emit_ia3_register_pool,
        emit_ia3_store_output, emit_ia3_tile_bases, emit_lora_output_tile_coords,
        emit_lora_stage_w_tile, emit_lora_stage_x_tile, emit_matmul_mma_lane_init,
        emit_zero_accumulators, wrga_ia3_register_budget,
    };

    let mut ptx = String::new();
    let budget = wrga_ia3_register_budget(config);

    emit_ptx_header(&mut ptx, PtxVersion::V7_0, TargetSm::Sm80);

    let entry_name = format!("nsl_wrga_fused_ia3_m{}n{}k{}", config.m, config.n, config.k);
    let params = [
        Param { ty: ParamTy::U64, name: "x_ptr" },
        Param { ty: ParamTy::U64, name: "w_ptr" },
        Param { ty: ParamTy::U64, name: "gamma_ptr" },
        Param { ty: ParamTy::U64, name: "y_ptr" },
    ];
    emit_param_block(&mut ptx, &entry_name, &params);

    emit_static_smem_decl(&mut ptx, 768);
    emit_ia3_register_pool(&mut ptx, &budget);
    emit_shmem_base_cvta(&mut ptx);
    emit_thread_lane_warp_registers(&mut ptx);
    emit_ia3_tile_bases(&mut ptx);
    emit_matmul_mma_lane_init(&mut ptx);

    emit_ld_param_u64(&mut ptx, "%rd_x", "x_ptr");
    emit_ld_param_u64(&mut ptx, "%rd_w", "w_ptr");
    emit_ld_param_u64(&mut ptx, "%rd_gamma", "gamma_ptr");
    emit_ld_param_u64(&mut ptx, "%rd_y", "y_ptr");

    emit_lora_output_tile_coords(&mut ptx, config.m, config.n);
    emit_zero_accumulators(&mut ptx, "main_accum", 8);

    let main_a_frag: [String; 4] = [
        "main_a_frag0".into(), "main_a_frag1".into(),
        "main_a_frag2".into(), "main_a_frag3".into(),
    ];
    let main_b_frag: [String; 2] = [
        "main_b_frag0".into(), "main_b_frag1".into(),
    ];
    let main_accum: [String; 4] = [
        "main_accum0".into(), "main_accum1".into(),
        "main_accum2".into(), "main_accum3".into(),
    ];

    let k_iters = (config.k + MMA_K_U32 - 1) / MMA_K_U32;
    for k_tile in 0..k_iters {
        ptx.push_str(&format!("// ===== IA3 K-iteration {} =====\n", k_tile));
        emit_lora_stage_x_tile(&mut ptx, k_tile, MMA_K_U32, config.m, config.k);
        emit_lora_stage_w_tile(&mut ptx, k_tile, config.n, config.k);
        ptx.push_str("    bar.sync 0;\n");
        emit_load_a_fragment_smem(&mut ptx, &main_a_frag, "%x_tile_base", 32);
        emit_load_b_fragment_smem(&mut ptx, &main_b_frag, "%w_tile_base", 16);
        emit_mma_instruction(&mut ptx, &main_accum, &main_a_frag, &main_b_frag, &main_accum);
        ptx.push_str("    bar.sync 0;\n");
    }

    emit_ia3_load_gamma(&mut ptx);
    emit_ia3_gamma_multiply(&mut ptx);
    emit_ia3_store_output(&mut ptx, config.n);

    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");
    ptx
}
```

- [ ] **Step 2: Build**

Run:
```bash
cargo build -p nsl-codegen
```
Expected: clean.

- [ ] **Step 3: Run IA³ ptxas tests**

Run:
```bash
cargo test -p nsl-codegen --test wrga_fused_ptx_ptxas ia3_ptx_validates -- --nocapture 2>&1 | tail -30
```
Expected: **all 5 IA³ tests pass.** Diagnose failures same way as LoRA.

- [ ] **Step 4: Full regression sweep (LoRA + IA³ unit + existing tests)**

Run:
```bash
cargo test -p nsl-codegen --test wrga_fused_ptx_ptxas
cargo test -p nsl-codegen --lib wrga_fused_ptx
cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence -- --test-threads=1
```
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/wrga_fused_ptx.rs
git commit -m "feat(wrga): rewrite synthesize_fused_ia3_ptx against kernel_skeleton + matmul_mma"
```

### Task E4: IA³ integration fixtures

**Files:**
- Modify: `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs`

- [ ] **Step 1: Add IA³ test fixtures**

At the end of `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs`, add:
```rust
// ─── Commit 5: IA3 integration fixtures ───────────────────────────

// Fixture A: full compute path (baseline).
//   x = ones([4,8]), W = ones([8,8]), γ = ones([8])
//   y[i,j] = (x@W)[i,j] * γ[j] = 8 * 1 = 8 elementwise
const IA3_FIXTURE_A_SRC: &str = r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = ones([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=ia3, target=["Toy.w"])
let m = Toy()
m.to(cuda)
let x = ones([4, 8]).to(cuda)
let y_target = zeros([4, 8]).to(cuda)
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
m.ia3_scale_Toy_w__ia3 = ones([8]).to(cuda)
let y = m.forward(x)
print(y)
"#;

// Fixture B: γ actually multiplies.
//   x = ones([4,8]), W = ones([8,8]), γ = [2, 2, ..., 2]
//   y[i,j] = 8 * 2 = 16 elementwise
const IA3_FIXTURE_B_SRC: &str = r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = ones([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=ia3, target=["Toy.w"])
let m = Toy()
m.to(cuda)
let x = ones([4, 8]).to(cuda)
let y_target = zeros([4, 8]).to(cuda)
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
m.ia3_scale_Toy_w__ia3 = full([8], 2.0).to(cuda)
let y = m.forward(x)
print(y)
"#;

fn run_ia3_fixture(src: &str, expected: f32, fixture_name: &str) {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join(format!("{}.nsl", fixture_name));
    fs::write(&src_path, src).unwrap();
    let root = workspace_root();
    let stdlib = root.join("stdlib");
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", &stdlib)
        .env("NSL_WRGA_FUSED_CUDA", "1")
        .arg("run")
        .args(["--source-ad", "--target", "cuda_sm80"])
        .arg(&src_path);
    let output = cmd.output().expect("nsl run failed to spawn");
    assert!(
        output.status.success(),
        "{fixture_name} nsl run failed.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let tensor = parse_tensor_2d(&stdout);
    let mut max_diff: f32 = 0.0;
    for row in &tensor {
        for v in row {
            max_diff = max_diff.max((v - expected).abs());
        }
    }
    assert!(
        max_diff < 1e-4,
        "{fixture_name}: max |y - {expected}| = {max_diff:.3e}, want < 1e-4. \
         Tensor: {tensor:?}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn ia3_fixture_a_baseline() {
    run_ia3_fixture(IA3_FIXTURE_A_SRC, 8.0, "ia3_a");
}

#[cfg(feature = "cuda")]
#[test]
fn ia3_fixture_b_gamma_scaling() {
    run_ia3_fixture(IA3_FIXTURE_B_SRC, 16.0, "ia3_b");
}
```

- [ ] **Step 2: Build and run (under CUDA)**

Run:
```bash
cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence --features cuda ia3_fixture -- --test-threads=1 --nocapture 2>&1 | tail -30
```
Expected on CUDA machine: both fixtures pass.

If fixture A fails at `max_diff ≈ 8.0` (output = 0): matmul pipeline isn't running — check the staging + MMA loop.
If fixture A passes, fixture B fails at `max_diff ≈ 8.0` (output still = 8): γ isn't being read or applied — check `emit_ia3_load_gamma` and `emit_ia3_gamma_multiply`.

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs
git commit -m "test(wrga): IA3 integration fixtures (y=8 baseline, y=16 γ-scaling) at 1e-4"
```

---

## Task Group F — Commit 6: Hardening flip + milestone close-out

### Task F1: Flip `build_4_fused_cuda_actually_fires` gate

**Files:**
- Modify: `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs`

- [ ] **Step 1: Flip the test attribute**

In `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs`, locate the test `build_4_fused_cuda_actually_fires`. Replace:
```rust
#[test]
#[ignore]
fn build_4_fused_cuda_actually_fires() {
```
with:
```rust
#[cfg(feature = "cuda")]
#[test]
fn build_4_fused_cuda_actually_fires() {
```

- [ ] **Step 2: Run the hardening test**

Run:
```bash
cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence --features cuda build_4_fused_cuda_actually_fires -- --test-threads=1 --nocapture 2>&1 | tail -20
```
Expected on CUDA machine: PASS. `[nsl-gpu-launch-count] ≥ 1`.

If this fails at `count == 0` after all prior tasks pass, investigate:
- Is `synthesize_fused_lora_ptx` emitting the expected entry name?
- Is the codegen-level `nsl_wrga_register_fused_ptx` registering the kernel at program startup?
- Is `try_cuda_launch_fused_lora` finding the kernel in its registry?

- [ ] **Step 3: Full workspace regression sweep**

Run:
```bash
cargo test -p nsl-codegen --lib 2>&1 | tail -10
cargo test -p nsl-codegen --test wrga_fused_ptx_ptxas 2>&1 | tail -10
cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence --features cuda -- --test-threads=1 2>&1 | tail -15
```
Expected: all green across all test layers. `wrga_adapter_runtime_equivalence` now shows 10 passed (7 original + build_4_fused_real_launch + 2 IA³ + 1 hardening = 10), 0 ignored under `--features cuda`.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs
git commit -m "test(wrga): flip build_4_fused_cuda_actually_fires from ignore to cuda-gated"
```

### Task F2: Memory file updates — close-out

**Files (outside the repo — in `~/.claude/projects/.../memory/`):**
- Modify: `project_wrga_ptx_scaffolding_discovered.md`
- Create: `project_wrga_fused_ptx_rewrite.md`
- Modify: `MEMORY.md`

- [ ] **Step 1: Prepend retrospective paragraph + CLOSED marker to scaffolding memory**

Edit `C:\Users\bwiem\.claude\projects\c--Users-bwiem-projects-NSL\memory\project_wrga_ptx_scaffolding_discovered.md`. At the top of the body (after the frontmatter block), prepend:
```markdown
## CLOSED <YYYY-MM-DD>

**Retrospective:** B.3 shipped PTX scaffolding that looked correct in
string-pattern tests but was never validated against ptxas or real
launches. The 2026-04-16 discovery found this; this milestone closes
the gap. **Future PTX-emitting milestones must include ptxas validation
from the first commit.**

Closed by the "WRGA fused-LoRA/IA³ PTX rewrite" milestone on branch
`feat/wrga-fused-ptx-rewrite`. Four test layers now gate PTX emitters:
skeleton snapshots, unit ptxas sweep, integration numerical at 1e-4,
and E2E launch-counter. See
[project_wrga_fused_ptx_rewrite.md](project_wrga_fused_ptx_rewrite.md)
for the current-state invariants.

---

# (Original discovery content below)
```
(Replace `<YYYY-MM-DD>` with the actual close-out date.)

- [ ] **Step 2: Create the new invariants-memory file**

Create `C:\Users\bwiem\.claude\projects\c--Users-bwiem-projects-NSL\memory\project_wrga_fused_ptx_rewrite.md`:
```markdown
---
name: WRGA fused-LoRA/IA³ PTX rewrite — invariants
description: Load-bearing correctness invariants from the <YYYY-MM-DD> rewrite. Read before editing synthesize_fused_lora_ptx, synthesize_fused_ia3_ptx, matmul_mma, kernel_skeleton, or wrga_kernel_helpers.
type: project
---

# WRGA Fused-LoRA/IA³ PTX Rewrite — Load-Bearing Invariants

Closed <YYYY-MM-DD> on branch `feat/wrga-fused-ptx-rewrite`.
See [docs/plans/2026-04-16-wrga-fused-ptx-rewrite-design.md](../../../../projects/NSL/docs/plans/2026-04-16-wrga-fused-ptx-rewrite-design.md).

## The six invariants that must never regress

1. **Fragment reuse.** `%main_a_frag` (x-tile fragment) serves BOTH the
   main `x@W` MMA and the epilogue `x@A` MMA in each K-iteration. This
   is valid because `m16n8k16` A-fragments encode only the m×k tile and
   are independent of the B matrix's semantic meaning. Reloading x
   would double the SMEM read cost.

2. **Interleaved epilogue inside the K-loop.** `epi_interm += x_tile @
   a_tile` MUST happen INSIDE the main K-loop. Post-loop access sees
   only the LAST x_tile (SMEM is overwritten per iter). The emitter's
   phase 5 call order enforces this structurally.

3. **`bar.sync 0` discipline.** One barrier before fragment loads (SMEM
   writes must complete before any thread reads). One barrier at end
   of iteration (SMEM reads must complete before next iteration
   overwrites). Removing either is silent corruption.

4. **Padding happens in SMEM, not HBM.** Global loads are predicated to
   skip out-of-bounds; SMEM fills those slots via
   `emit_smem_zero_pad_predicated`'s `st.shared.b16 [addr], 0`. Any
   "optimization" that reads zeros from an HBM buffer defeats the
   fusion purpose.

5. **`matmul_mma` register preconditions.** `%mma_a_row`, `%mma_b_row`,
   `%mma_addr`, `%smem_base_*` must be declared and initialized per
   `emit_matmul_mma_lane_init` BEFORE any `matmul_mma` helper call.
   B.3 shipped pseudocode precisely because these preconditions were
   undocumented and uninitialized.

6. **`emit_smem_zero_pad_predicated` no-op path.** When
   `real_extent == padded_extent`, the helper emits zero PTX
   instructions. Locked in by `smem_zero_pad__rank16_to_16_f16.snap`
   being an empty file.

## The four-layer test discipline

- **Skeleton snapshots** in `kernel_skeleton/tests/snapshots/*.snap`:
  per-variant pinned PTX for every helper. 9 snapshot files covering
  7 helpers × variants.
- **Unit ptxas** in `crates/nsl-codegen/tests/wrga_fused_ptx_ptxas.rs`:
  6 LoRA configs + 5 IA³ configs. Each feeds PTX to `cuModuleLoadData`
  (or `nvcc --ptx` fallback). Catches invalid syntax, missing `.shared`,
  uninitialized registers, operand-type mismatches, `.param`-as-operand
  bugs. The layer B.3 lacked.
- **Integration numerical** in `wrga_adapter_runtime_equivalence.rs`
  (`build_4_fused_real_launch`, `ia3_fixture_a/b`): real cudarc launch
  under `NSL_WRGA_FUSED_CUDA=1`, asserts output at 1e-4 tolerance.
  Catches fragment-layout bugs (valid PTX, wrong result).
- **E2E launch-counter** (`build_4_fused_cuda_actually_fires`):
  confirms `[nsl-gpu-launch-count] ≥ 1`. Catches dispatch regressions
  (CPU fallback silently replacing real launch).

## File map

- `crates/nsl-codegen/src/kernel_skeleton/` — shared header/smem/tid/
  pad/params primitives + per-variant snapshots.
- `crates/nsl-codegen/src/wrga_kernel_helpers.rs` — WRGA-specific tile
  staging, register pool, output-tile coord, lane-derivation init.
- `crates/nsl-codegen/src/wrga_fused_ptx.rs` — `synthesize_fused_lora_ptx`
  and `synthesize_fused_ia3_ptx` bodies; both call into skeleton +
  helpers + matmul_mma.
- `crates/nsl-codegen/src/matmul_mma.rs` — m16n8k16 MMA + fragment load
  primitives. **Unchanged** by the rewrite (it was already correct; B.3's
  bug was callers not initializing the registers it assumed existed).
- `crates/nsl-codegen/src/ptxas_validation.rs` — `validate_ptx` helper
  used by unit tests.
- `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs` — all four
  WRGA-visible test layers (build_4_*, ia3_fixture_*).

## Explicitly deferred (NOT in this milestone)

- B.3.1: GatedLoRA epilogue + PTX sigmoid (Taylor or 256-entry LUT).
- B.4 or later: sm_90 WGMMA path, ldmatrix for fragment loads, cp.async
  for SMEM staging overlap, multi-warp-per-tile larger tiles,
  multi-tile γ staging, performance benchmarking.
- Deep `kernel_skeleton` refactor with fusion-callback pattern —
  natural follow-up once both FA and WRGA are stable and provide two
  concrete emitters to abstract over.

## What stays shipped untouched (do not modify)

- B.2.1 runtime adapter side-table, init emission, forward rewrite,
  seed route-through.
- Kernel handle routing, dedup registry, `try_cuda_launch_fused_*`, arg
  marshaling.
- AST-level fusion decision and conditional rewrite in
  `wrga_adapter_rewrite.rs`.
- WGGO→WRGA consumer wiring (`allocate_ranks` honoring overrides).
```
(Replace `<YYYY-MM-DD>` with the actual close-out date.)

- [ ] **Step 3: Update MEMORY.md index**

Edit `C:\Users\bwiem\.claude\projects\c--Users-bwiem-projects-NSL\memory\MEMORY.md`. Find the two lines:
```markdown
## WRGA Session Close-out (2026-04-13 at 49d4e55) — framing outdated
- See [project_wrga_session_2026_04_13_closeout.md](project_wrga_session_2026_04_13_closeout.md) — ...

## WRGA PTX scaffolding discovered (2026-04-16)
- See [project_wrga_ptx_scaffolding_discovered.md](project_wrga_ptx_scaffolding_discovered.md) — ...
```
Replace with:
```markdown
## WRGA Session Close-out (2026-04-13 at 49d4e55) — superseded
- See [project_wrga_session_2026_04_13_closeout.md](project_wrga_session_2026_04_13_closeout.md) — historical; framing superseded by the <YYYY-MM-DD> rewrite.

## WRGA PTX scaffolding (2026-04-16) — CLOSED <YYYY-MM-DD>
- See [project_wrga_ptx_scaffolding_discovered.md](project_wrga_ptx_scaffolding_discovered.md) — the gap was closed by the PTX rewrite milestone.

## WRGA fused-LoRA/IA³ PTX rewrite — invariants (<YYYY-MM-DD>)
- See [project_wrga_fused_ptx_rewrite.md](project_wrga_fused_ptx_rewrite.md) — six load-bearing invariants and the four-layer test discipline. Read before editing any WRGA fused-PTX code.
```

- [ ] **Step 4: No commit for memory files (they live outside the repo)**

Memory files are in `C:\Users\bwiem\.claude\projects\...` which is NOT part of the git repo. Updates are immediate on save.

### Task F3: Final milestone close-out commit

**Files:**
- Create: `docs/plans/2026-04-16-wrga-fused-ptx-rewrite-CLOSEOUT.md`

- [ ] **Step 1: Write the close-out note**

Create `docs/plans/2026-04-16-wrga-fused-ptx-rewrite-CLOSEOUT.md`:
```markdown
# WRGA Fused-LoRA/IA³ PTX Rewrite — Milestone Close-Out

**Closed:** <YYYY-MM-DD>
**Branch:** `feat/wrga-fused-ptx-rewrite`
**Spec:** [2026-04-16-wrga-fused-ptx-rewrite-design.md](2026-04-16-wrga-fused-ptx-rewrite-design.md)
**Plan:** [2026-04-16-wrga-fused-ptx-rewrite-plan.md](2026-04-16-wrga-fused-ptx-rewrite-plan.md)

## Close-out criteria (§5 of spec) — state at close

| # | Criterion | State |
|---|---|---|
| 1 | All 6 commits merged | ✓ |
| 2 | FA v2 snapshot tests byte-identical after commit 1 | ✓ |
| 3 | `kernel_skeleton/tests/snapshots/*.snap` all green | ✓ (9 snapshots across 7 helpers × variants) |
| 4 | WRGA LoRA unit ptxas test green across 6 configs | ✓ |
| 5 | WRGA LoRA integration test green at 1e-4 under NSL_WRGA_FUSED_CUDA=1 | ✓ |
| 6 | WRGA IA³ unit ptxas test green across 5 configs | ✓ |
| 7 | WRGA IA³ integration test green on both fixtures at 1e-4 | ✓ |
| 8 | `build_4_fused_cuda_actually_fires` flipped to `#[cfg(feature="cuda")]`, count ≥ 1 | ✓ |
| 9a | `project_wrga_ptx_scaffolding_discovered.md` prepended retrospective + CLOSED marker | ✓ |
| 9b | `project_wrga_fused_ptx_rewrite.md` created with invariants | ✓ |
| 9c | MEMORY.md index updated | ✓ |

## Institutional lesson (for future PTX milestones)

> B.3 shipped PTX scaffolding that looked correct in string-pattern
> tests but was never validated against ptxas or real launches. The
> 2026-04-16 discovery found this; this milestone closes the gap.
> **Future PTX-emitting milestones must include ptxas validation from
> the first commit.**

## Deferred follow-ups (from §5 of spec, unchanged)

- **B.3.1** — GatedLoRA epilogue + PTX sigmoid (Taylor vs. 256-entry LUT)
- **B.4 or later** — sm_90 WGMMA path, ldmatrix, cp.async staging overlap,
  multi-warp-per-tile, multi-tile γ staging, perf benchmarking
- **Deep kernel_skeleton refactor** — fusion-callback pattern once FA and
  WRGA are both stable
```
(Fill in `<YYYY-MM-DD>` with the actual date.)

- [ ] **Step 2: Commit the close-out note (force-add)**

```bash
git add -f docs/plans/2026-04-16-wrga-fused-ptx-rewrite-CLOSEOUT.md
git commit -m "docs(wrga): milestone close-out — fused-LoRA/IA3 PTX rewrite"
```

- [ ] **Step 3: Final full-workspace regression run**

Run:
```bash
cargo build -p nsl-codegen -p nsl-runtime -p nsl-cli --features cuda
cargo test -p nsl-codegen --lib 2>&1 | tail -5
cargo test -p nsl-codegen --test fa_v2_snapshots 2>&1 | tail -5
cargo test -p nsl-codegen --test wrga_fused_ptx_ptxas 2>&1 | tail -5
cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence --features cuda -- --test-threads=1 2>&1 | tail -10
```
Expected: all green. Milestone is complete.

**Milestone closed.** Ready for PR / merge to main.

---

## Self-Review Results

**Spec coverage check (§1-§5 of spec):**

- §1 architecture: Task Group A covers kernel_skeleton extraction; Task Groups C and E cover LoRA and IA³ rewrites; test infrastructure is in B1 (ptxas_validation module). Six-commit sequence maps to Task Groups A-F. ✓
- §2 kernel_skeleton contract: Tasks A3-A7 extract each helper with per-variant snapshots. Task A1's inventory satisfies the diff-and-parameterize discipline. ✓
- §3 LoRA kernel structure: Tasks C1-C4 cover the register budget, output coords, tile staging, accumulator init, and synthesizer rewrite. All five correctness invariants are mentioned in the C4 rewrite body comment OR enforced by the 6-config ptxas test. ✓
- §4 IA³ kernel structure: Tasks E1-E3 (ptxas test, helpers, synthesizer) plus E4 (integration fixtures). ✓
- §5 test matrix + close-out: Task Group F covers the hardening flip and memory updates. ✓

**Placeholder scan:** No TBD/TODO/placeholder steps remain. Every step has an exact command, exact expected output, or concrete code to write. The `<YYYY-MM-DD>` markers in the memory files are intentional (filled at milestone close).

**Type consistency check:**
- `FusedLoraConfig` / `FusedIa3Config` struct field names (m, n, k, rank, target_sm, site_id) used consistently across Tasks B2, C4, E1, E3.
- Helper names used consistently: `emit_ptx_header`, `emit_static_smem_decl`, `emit_shmem_base_cvta`, `emit_thread_lane_warp_registers`, `emit_param_block`, `emit_ld_param_u64`, `emit_ld_param_f32`, `emit_smem_zero_pad_predicated`.
- WRGA-specific helpers: `wrga_lora_register_budget`, `emit_lora_register_pool`, `emit_lora_output_tile_coords`, `emit_lora_tile_bases`, `emit_matmul_mma_lane_init`, `emit_lora_stage_{x,w,a,b}_tile`, `emit_zero_accumulators`, `emit_lora_store_output`. Matching IA³ prefix `wrga_ia3_*` / `emit_ia3_*`.
- Register names (`%rd_x`, `%rd_w`, `%mma_a_row`, `%main_accum0`, etc.) used consistently in emission code across C1-C4 and E2-E3.

No gaps found.

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-04-16-wrga-fused-ptx-rewrite-plan.md` (force-added per cross-worktree-visibility preference, matching the spec's treatment). Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task (A1, A2, ..., F3), review between tasks, fast iteration. Good for this plan because each Task is self-contained and the TDD gate per task makes per-task verification cheap.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints. Faster wall-clock if no mid-plan surprises, but if a ptxas failure requires debugging PTX emission across multiple helpers, the context can get crowded.

**Which approach?**
