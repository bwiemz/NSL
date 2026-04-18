# WRGA B.3.1 — Fused GatedLoRA Forward PTX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a fused GatedLoRA forward PTX kernel that turns the 10 red ptxas configs from Task Group 3 green and the 4 integration fixtures from Task Group 5 green (3 at 1e-4 saturation tolerance, 1 at 1e-3 mid-range tolerance).

**Architecture:** Sandwich extraction. A new private `emit_fused_adapter_kernel_body` holds LoRA's proven kernel body; both `synthesize_fused_lora_ptx` (refactored) and `synthesize_fused_gatedlora_ptx` (new) call it with a `FoldKind` enum selecting the fold step. PTX sigmoid via fused `rcp.approx.f32(1 + ex2.approx.f32(-x * log2e))` with pinned hex constants. Gate loaded per-thread (2 f32 per thread matching m16n8k16 D-fragment column ownership) at `LastKIter` scheduling to overlap HBM latency with the final main MMA.

**Tech Stack:** Rust (nsl-codegen, nsl-cli crates), PTX 7.0 for sm_80, `cudarc::driver::sys::cuModuleLoadData` and `ptxas` for validation, existing NSL test harness.

**Reference spec:** [docs/plans/2026-04-18-wrga-b31-gatedlora-fused-forward-design.md](docs/plans/2026-04-18-wrga-b31-gatedlora-fused-forward-design.md)

**Branch:** `feat/wrga-b31-gatedlora-fused-forward` (already created; spec committed at `767c9e4`).

**Commit taxonomy:** Each Task ends with its own intermediate commit for bisect / rollback granularity. The spec's 6 logical commits correspond to Task Groups 1-6; intermediate task commits within each group can be preserved for bisect or squashed at merge time.

---

## File Structure

**Modified files:**
- `crates/nsl-codegen/src/wrga_fused_ptx.rs` — add `FoldKind` / `GateLoadPhase` / `PartialTileMask` enums; add `FusedGatedLoraConfig` struct; extract `emit_fused_adapter_kernel_body`; refactor `synthesize_fused_lora_ptx` to call it; add new `synthesize_fused_gatedlora_ptx`
- `crates/nsl-codegen/src/wrga_kernel_helpers.rs` — add `emit_sigmoid_approx_fused`, `emit_gate_load_per_thread`, `emit_gatedlora_fold` helpers; possibly promote `%n_real` into shared pool; possibly confirm `%scale_reg` reuse
- `crates/nsl-codegen/tests/wrga_fused_ptx_ptxas.rs` — add 10 new GatedLoRA ptxas tests
- `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs` — add 4 new integration fixtures (A/B/C/D)

**Created files (docs):**
- `docs/plans/2026-04-18-wrga-b31-gatedlora-CLOSEOUT.md`
- `docs/plans/2026-04-18-wrga-b32-fused-backward-STUB.md`
- WRGA paper Appendix B update (path located at commit 6 prep — likely `docs/research/WRGA.pdf` source or a `.md` equivalent)

**Memory file updates (outside git, in `C:\Users\bwiem\.claude\projects\c--Users-bwiem-projects-NSL\memory\`):**
- `project_wrga_fused_ptx_rewrite.md` — append invariants #13-#16
- `MEMORY.md` — append B.3.1 close-out index entry

---

## Prerequisites

Confirm baseline before starting Task 1:

```bash
cd /c/Users/bwiem/projects/NSL
git checkout feat/wrga-b31-gatedlora-fused-forward
git log --oneline -3
# Expected: 767c9e4 docs(wrga): B.3.1 fused GatedLoRA forward PTX design spec
#           <main tip with B.3 merged>

cargo build -p nsl-codegen -p nsl-cli --features cuda 2>&1 | tail -3
# Expected: clean build (pre-existing warnings OK)

cargo test -p nsl-codegen --test wrga_fused_ptx_ptxas --features cuda 2>&1 | tail -5
# Expected: 11 passed / 0 failed (6 LoRA + 5 IA³ from prior milestone)

cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence --features cuda -- --test-threads=1 2>&1 | tail -5
# Expected: 11 passed / 0 failed / 0 ignored
```

If baseline is not clean, stop and investigate before starting B.3.1.

---

## Task Group 1 — Commit 1: Refactor LoRA to shared body

### Task 1.1: Pre-refactor audits

**Files:** (inspection only; no changes committed in this task)
- Read: `crates/nsl-codegen/src/wrga_fused_ptx.rs`
- Read: `crates/nsl-codegen/src/wrga_kernel_helpers.rs`

- [ ] **Step 1: Register-collision grep pre-audit**

Run:
```bash
grep -n '%fold_tmp\|%r_col1\|%p_col0\|%p_col1\|%sig_tmp\|%rd_gate\|%rd_gate_addr\|%r_gate\|%gate0\|%gate1' \
    crates/nsl-codegen/src/wrga_fused_ptx.rs \
    crates/nsl-codegen/src/wrga_kernel_helpers.rs
```

Expected: **zero matches**. Record the result for the eventual commit message.

If ANY match, all GatedLoRA-additions must prefix with `%gl_` (e.g. `%gl_fold_tmp`, `%gl_p_col0`). Adjust Task 2.1-2.3 and Task 4.x emissions accordingly throughout this plan; record the prefix convention in the commit 1 message.

- [ ] **Step 2: `%scale_reg` reuse check**

Run:
```bash
grep -n '%scale_reg' crates/nsl-codegen/src/wrga_kernel_helpers.rs \
    crates/nsl-codegen/src/wrga_fused_ptx.rs
```

Expected: matches in both files (LoRA declares `%scale_reg` in `emit_lora_register_pool` and uses it in the fold step of `synthesize_fused_lora_ptx`).

If matches are present: note "confirmed: `%scale_reg` reused from LoRA pool" for the commit message. GatedLoRA's fold steps will use the same register.

If NO matches: `%scale_reg` must be added as an 11th register addition in Task 2's helper work. Update Task 2.3's register declarations accordingly AND add `%scale_reg` to Task 1.1 Step 1's collision-grep list (since its name could collide if someone else uses a similar name).

- [ ] **Step 3: `%n_real` sourcing audit**

Run:
```bash
grep -n '%n_real' crates/nsl-codegen/src/wrga_kernel_helpers.rs
```

Expected: matches inside `emit_lora_register_pool` declaration AND inside `emit_lora_store_output`'s body. If the declaration appears BOTH in the pool AND somewhere else (e.g., locally in `emit_lora_store_output`), that's pre-existing duplication to fix.

If `%n_real` is declared ONLY inside `emit_lora_store_output` (not in `emit_lora_register_pool`): promote it. Task 1.3 (extraction step) will need to add `.reg .u32 %n_real;` into `emit_lora_register_pool`. Record the promotion in the commit message.

- [ ] **Step 4: Internal-caller stability grep**

Run:
```bash
grep -rn 'emit_lora_register_pool\|emit_lora_store_output\|emit_lora_output_tile_coords\|emit_lora_tile_bases\|emit_matmul_mma_lane_init' \
    crates/ --include='*.rs' 2>&1 | sort > /tmp/b31_callers_before.txt
wc -l /tmp/b31_callers_before.txt
```

Record the line count. After Task 1.3 extraction completes, re-run this grep and diff against `/tmp/b31_callers_before.txt`. The only new reference should be from inside `emit_fused_adapter_kernel_body`. If any other file picks up or loses references, that's a bug to investigate before commit 1 lands.

- [ ] **Step 5: Inspect `synthesize_fused_lora_ptx` body**

Read `crates/nsl-codegen/src/wrga_fused_ptx.rs` start to finish. Record:
- Exact structure of the current body (prolog → staging → main K-loop → post-loop (x@A)@B → fold → store → ret)
- Line range to extract into `emit_fused_adapter_kernel_body`
- Helper calls it currently makes (for the internal-caller grep)

No file edits in this task. Results captured for Tasks 1.2-1.5.

### Task 1.2: Add `FoldKind` / `GateLoadPhase` / `PartialTileMask` enums

**Files:**
- Modify: `crates/nsl-codegen/src/wrga_fused_ptx.rs`

- [ ] **Step 1: Read current imports and top of file**

```bash
head -40 crates/nsl-codegen/src/wrga_fused_ptx.rs
```

Note where existing struct definitions (`FusedLoraConfig`, `FusedIa3Config`, etc.) live. The new enums go adjacent to those, above any existing public synthesizer functions.

- [ ] **Step 2: Add the three enums**

Insert after the last existing struct/enum and before the first `pub fn synthesize_fused_*`:

```rust
/// Variation point for `emit_fused_adapter_kernel_body`'s fold step.
/// LoRA uses `Scalar`; GatedLoRA uses `PerColumnSigmoid`.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FoldKind {
    /// LoRA: main_accum += epi_final * scale  (uniform scalar scale)
    Scalar { scale: f32 },
    /// GatedLoRA: main_accum += epi_final * sigmoid(gate[col]) * scale  (per-column)
    PerColumnSigmoid {
        scale: f32,
        gate_ptr_param_name: &'static str,
        gate_load_phase: GateLoadPhase,
        partial_tile_mask: PartialTileMask,
    },
}

/// Scheduling hint for when to emit gate-load + sigmoid computation in the
/// PerColumnSigmoid fused body.
///
/// B.3.1 ships with `LastKIter` unconditionally; all shipped configs have
/// k_iters >= 1 for the epilogue path.  `PostLoop` is reserved for a
/// potential future milestone that benchmarks load-phase alternatives.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateLoadPhase {
    /// Emit gate load + sigmoid at the end of the final main K-iteration,
    /// before its bar.sync.  Overlaps HBM load latency with the final MMA.
    LastKIter,
    /// Emit gate load + sigmoid after the main K-loop completes, before
    /// the post-loop (x@A)@B MMA.  Has NO emission site in B.3.1; reserved.
    PostLoop,
}

/// Partial-tile handling strategy for sub-MMA output tiles (n < 8).
///
/// B.3.1 ships `FoldResultMask` unconditionally; `SentinelGate` is reserved
/// for a future variant that stages gate through SMEM with a sentinel value.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PartialTileMask {
    /// Mask fold contribution with @%p_colN predicate; OOB gate reads
    /// are harmless because the fold output is discarded.
    FoldResultMask,
    /// Pre-populate OOB gate SMEM slots with -20.0 such that sigmoid(-20) ≈ 0
    /// naturally zeros the contribution.  Not emitted by B.3.1.
    SentinelGate,
}
```

- [ ] **Step 3: Build**

```bash
cargo build -p nsl-codegen 2>&1 | tail -5
```

Expected: clean build. The enums are unused (no emission site yet) but `#[allow(dead_code)]` suppresses the lint.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/wrga_fused_ptx.rs
git commit -m "feat(wrga): add FoldKind/GateLoadPhase/PartialTileMask enums

Variation-point enum shape for the sandwich extraction coming in the
next task.  #[allow(dead_code)] on all three because B.3.1 ships only
the Scalar / LastKIter / FoldResultMask variants; the others are
reserved for future milestones per spec §1."
```

### Task 1.3: Extract `emit_fused_adapter_kernel_body` from LoRA

**Files:**
- Modify: `crates/nsl-codegen/src/wrga_fused_ptx.rs`
- Modify (if Task 1.1 Step 3 found local `%n_real`): `crates/nsl-codegen/src/wrga_kernel_helpers.rs`

- [ ] **Step 1: (Conditional) Promote `%n_real` into shared pool**

If Task 1.1 Step 3 found `%n_real` declared only inside `emit_lora_store_output` (not in `emit_lora_register_pool`), promote it now. Edit `crates/nsl-codegen/src/wrga_kernel_helpers.rs`:

Find `emit_lora_register_pool` and add `.reg .u32 %n_real;` to its register declaration block. Remove the duplicate declaration (if any) from `emit_lora_store_output`'s body.

If Task 1.1 Step 3 confirmed `%n_real` is already in the shared pool: SKIP this step.

- [ ] **Step 2: Read current `synthesize_fused_lora_ptx` body**

```bash
grep -n 'pub fn synthesize_fused_lora_ptx' crates/nsl-codegen/src/wrga_fused_ptx.rs
```

Note the starting line. The body extends from `{` after the signature to the matching closing `}`. This is the code to extract into a private function.

- [ ] **Step 3: Create private `emit_fused_adapter_kernel_body`**

Add a new private function ABOVE `synthesize_fused_lora_ptx`:

```rust
/// Shared kernel body for fused adapter matmul kernels.  Emits:
///   1. Header (.version, .target, .address_size)
///   2. Entry + param block (6 params for Scalar, 7 for PerColumnSigmoid)
///   3. SMEM decl + register pool + shmem_base_cvta + thread/lane/warp init
///   4. Param loads into named registers (includes %rd_gate under PerColumnSigmoid)
///   5. Output-tile coords + zero accumulators
///   6. Main K-loop with interleaved epilogue; gate load+sigmoid at LastKIter
///   7. Post-loop (x@A)@B MMA
///   8. Fold step — Scalar (main_accum += epi_final * scale) OR
///      PerColumnSigmoid (main_accum += epi_final * sigmoid(gate) * scale, per-thread)
///   9. Predicated store to y + ret
///
/// The variation point is the fold step.  Prolog diffs slightly between
/// Scalar and PerColumnSigmoid (gate_ptr param + its prolog load are
/// conditional on PerColumnSigmoid).  All other phases are identical.
fn emit_fused_adapter_kernel_body(
    ptx: &mut String,
    entry_name: &str,
    m: u32,
    n: u32,
    k: u32,
    rank: u32,
    fold: FoldKind,
) {
    use crate::kernel_skeleton::header::{emit_ptx_header, PtxVersion, TargetSm};
    use crate::kernel_skeleton::indexing::emit_thread_lane_warp_register_init;
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

    // Configuration derived from fold variant
    let is_gated = matches!(fold, FoldKind::PerColumnSigmoid { .. });

    // 1. Header
    emit_ptx_header(ptx, PtxVersion::V7_0, TargetSm::Sm80);

    // 2. Param block — conditionally include gate_ptr
    let mut params = vec![
        Param { ty: ParamTy::U64, name: "x_ptr" },
        Param { ty: ParamTy::U64, name: "w_ptr" },
        Param { ty: ParamTy::U64, name: "a_ptr" },
        Param { ty: ParamTy::U64, name: "b_ptr" },
        Param { ty: ParamTy::F32, name: "scale" },
        Param { ty: ParamTy::U64, name: "y_ptr" },
    ];
    if is_gated {
        params.push(Param { ty: ParamTy::U64, name: "gate_ptr" });
    }
    emit_param_block(ptx, entry_name, &params);

    // 3. SMEM + register pool + base + indexing init
    emit_static_smem_decl(ptx, 1536);
    let budget = wrga_lora_register_budget(&placeholder_config(m, n, k, rank));
    emit_lora_register_pool(ptx, &budget);
    if is_gated {
        // Additional registers for PerColumnSigmoid path
        ptx.push_str("    .reg .f32 %gate0, %gate1;\n");
        ptx.push_str("    .reg .f32 %sig_tmp;\n");
        ptx.push_str("    .reg .f32 %fold_tmp;\n");
        ptx.push_str("    .reg .u64 %rd_gate, %rd_gate_addr;\n");
        ptx.push_str("    .reg .u32 %r_gate, %r_col1;\n");
        ptx.push_str("    .reg .pred %p_col0, %p_col1;\n");
    }
    emit_shmem_base_cvta(ptx);
    emit_thread_lane_warp_register_init(ptx);
    emit_lora_tile_bases(ptx);
    emit_matmul_mma_lane_init(ptx);

    // 4. Param loads
    emit_ld_param_u64(ptx, "%rd_x", "x_ptr");
    emit_ld_param_u64(ptx, "%rd_w", "w_ptr");
    emit_ld_param_u64(ptx, "%rd_a", "a_ptr");
    emit_ld_param_u64(ptx, "%rd_b", "b_ptr");
    emit_ld_param_u64(ptx, "%rd_y", "y_ptr");
    emit_ld_param_f32(ptx, "%scale_reg", "scale");
    if is_gated {
        emit_ld_param_u64(ptx, "%rd_gate", "gate_ptr");
    }

    // 5. Output-tile coords + accumulator init
    emit_lora_output_tile_coords(ptx, m, n);
    emit_zero_accumulators(ptx, "main_accum", 8);
    emit_zero_accumulators(ptx, "epi_interm", 4);

    // 6. Main K-loop (with interleaved epilogue; gate load at LastKIter if gated)
    let k_iters = (k + 15) / 16;
    let main_a_frag: [String; 4] = [
        "%main_a_frag0".into(), "%main_a_frag1".into(),
        "%main_a_frag2".into(), "%main_a_frag3".into(),
    ];
    let main_b_frag: [String; 2] = [
        "%main_b_frag0".into(), "%main_b_frag1".into(),
    ];
    let epi_a_frag: [String; 4] = [
        "%epi_a_frag0".into(), "%epi_a_frag1".into(),
        "%epi_a_frag2".into(), "%epi_a_frag3".into(),
    ];
    let main_accum: [String; 4] = [
        "%main_accum0".into(), "%main_accum1".into(),
        "%main_accum2".into(), "%main_accum3".into(),
    ];
    let epi_interm: [String; 4] = [
        "%epi_interm0".into(), "%epi_interm1".into(),
        "%epi_interm2".into(), "%epi_interm3".into(),
    ];

    for k_tile in 0..k_iters {
        ptx.push_str(&format!("// ===== K-iteration {} =====\n", k_tile));
        emit_lora_stage_x_tile(ptx, k_tile, m, k);
        emit_lora_stage_w_tile(ptx, k_tile, n, k);
        emit_lora_stage_a_tile(ptx, k_tile, rank, k);
        ptx.push_str("    bar.sync 0;\n");
        emit_load_a_fragment_smem(ptx, &main_a_frag, "%x_tile_base", 32);
        emit_load_b_fragment_smem(ptx, &main_b_frag, "%w_tile_base", 16);
        emit_load_a_fragment_smem(ptx, &epi_a_frag, "%a_tile_base", 32);
        emit_mma_instruction(ptx, &main_accum, &main_a_frag, &main_b_frag, &main_accum);
        emit_mma_instruction(ptx, &epi_interm, &main_a_frag, &epi_a_frag, &epi_interm);
        if k_tile == k_iters - 1 {
            if let FoldKind::PerColumnSigmoid {
                gate_load_phase: GateLoadPhase::LastKIter, ..
            } = fold
            {
                crate::wrga_kernel_helpers::emit_gate_load_per_thread(ptx);
                crate::wrga_kernel_helpers::emit_sigmoid_approx_fused(ptx, "%gate0", "%gate0");
                crate::wrga_kernel_helpers::emit_sigmoid_approx_fused(ptx, "%gate1", "%gate1");
            }
        }
        ptx.push_str("    bar.sync 0;\n");
    }

    // 7. Post-loop b-tile stage + (x@A)@B MMA
    let epi_b_frag: [String; 2] = [
        "%epi_b_frag0".into(), "%epi_b_frag1".into(),
    ];
    let epi_final: [String; 4] = [
        "%epi_final0".into(), "%epi_final1".into(),
        "%epi_final2".into(), "%epi_final3".into(),
    ];
    emit_lora_stage_b_tile(ptx, rank, n);
    ptx.push_str("    bar.sync 0;\n");
    emit_load_b_fragment_smem(ptx, &epi_b_frag, "%b_tile_base", 16);
    emit_zero_accumulators(ptx, "epi_final", 4);
    emit_mma_instruction(ptx, &epi_final, &epi_interm, &epi_b_frag, &epi_final);

    // 8. Fold step — variant-dependent
    match fold {
        FoldKind::Scalar { .. } => {
            // LoRA fold: main_accum += epi_final * scale (uniform)
            for i in 0..4u32 {
                ptx.push_str(&format!(
                    "    mul.f32 %epi_final{i}, %epi_final{i}, %scale_reg;\n"
                ));
                ptx.push_str(&format!(
                    "    add.f32 %main_accum{i}, %main_accum{i}, %epi_final{i};\n"
                ));
            }
        }
        FoldKind::PerColumnSigmoid { .. } => {
            // GatedLoRA fold: per-column, predicated.  main_accum0/2 use %gate0;
            // main_accum1/3 use %gate1.  Fold masks on %p_col0/%p_col1.
            crate::wrga_kernel_helpers::emit_gatedlora_fold(ptx);
        }
    }

    // 9. Store + ret
    emit_lora_store_output(ptx, n);
    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");
}

/// Helper to construct a synthetic `FusedLoraConfig` for register budget
/// computation inside the shared body (budget doesn't depend on whether
/// we're LoRA or GatedLoRA; it depends only on dims).
fn placeholder_config(m: u32, n: u32, k: u32, rank: u32) -> FusedLoraConfig {
    FusedLoraConfig {
        site_id: String::new(),
        m,
        n,
        k,
        rank,
        target_sm: 80,
    }
}
```

- [ ] **Step 4: Build**

```bash
cargo build -p nsl-codegen 2>&1 | tail -5
```

Expected: clean build. The body function is defined but nothing calls it yet — the refactor completes in Task 1.4.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/wrga_fused_ptx.rs
git add crates/nsl-codegen/src/wrga_kernel_helpers.rs  # only if %n_real promoted in Step 1
git commit -m "feat(wrga): extract emit_fused_adapter_kernel_body (shared body)

Private function holding the shared kernel body for LoRA and GatedLoRA
synthesizers.  Variation point: fold step (FoldKind::Scalar vs
PerColumnSigmoid).  Prolog diffs on whether gate_ptr param and its
ld.param.u64 are emitted.  All other phases identical.

Still unused at this commit — Task 1.4 refactors synthesize_fused_lora_ptx
to call this body.

%n_real audit outcome: <describe result of Task 1.1 Step 3; either
'already in shared pool' or 'promoted from local scope this commit'>"
```

### Task 1.4: Refactor `synthesize_fused_lora_ptx` to call shared body

**Files:**
- Modify: `crates/nsl-codegen/src/wrga_fused_ptx.rs`

- [ ] **Step 1: Replace LoRA synthesizer body with delegating call**

Find the current `pub fn synthesize_fused_lora_ptx(...) -> String` and replace its body:

```rust
pub fn synthesize_fused_lora_ptx(config: &FusedLoraConfig) -> String {
    assert!(
        config.rank <= 16,
        "B.3 rank ceiling: {} > 16; multi-pass epilogue is a follow-up milestone",
        config.rank,
    );
    assert!(config.target_sm >= 80, "B.3 requires sm_80+");
    let entry_name = format!(
        "nsl_wrga_fused_lora_m{}n{}k{}r{}",
        config.m, config.n, config.k, config.rank,
    );
    let scale = config.alpha as f32 / config.rank as f32;
    let mut ptx = String::new();
    emit_fused_adapter_kernel_body(
        &mut ptx,
        &entry_name,
        config.m,
        config.n,
        config.k,
        config.rank,
        FoldKind::Scalar { scale },
    );
    ptx
}
```

Note: `config.alpha` may or may not exist as a field on `FusedLoraConfig`; check via `grep -n 'alpha' crates/nsl-codegen/src/wrga_fused_ptx.rs`. If the LoRA synthesizer currently computes scale differently (hardcoded `1.0`, or uses a different formula), match that EXACTLY here to preserve byte-identical PTX. The refactor's goal is same-behavior, not same-code-structure.

- [ ] **Step 2: Build**

```bash
cargo build -p nsl-codegen 2>&1 | tail -5
```

Expected: clean build.

- [ ] **Step 3: Run LoRA ptxas tests — BYTE-IDENTITY GATE**

```bash
cargo test -p nsl-codegen --test wrga_fused_ptx_ptxas --features cuda lora_ptx_validates 2>&1 | tail -15
```

Expected: **6 passed / 0 failed.** If ANY LoRA ptxas test fails, the refactor regressed PTX validity; investigate the diff between old and new emission.

**Note on byte-identity:** this test suite is **ptxas-compile** (semantic validation via `cuModuleLoadData` / `ptxas`), NOT snapshot comparison. Two structurally different PTX strings that both compile are equally valid here; the real byte-identity gate is the next step.

- [ ] **Step 4: Run LoRA integration tests — BEHAVIOR-LEVEL GATE**

```bash
cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence --features cuda -- --test-threads=1 2>&1 | tail -10
```

Expected: **11 passed / 0 failed / 0 ignored.** This includes `build_4_fused_real_launch` and `build_4_fused_cuda_actually_fires` — the real-launch + launch-counter tests that prove behavior is byte-identical.

If any test fails: the refactor regressed behavior. The numerical output or launch count differs from baseline.

- [ ] **Step 5: Internal-caller stability re-check**

```bash
grep -rn 'emit_lora_register_pool\|emit_lora_store_output\|emit_lora_output_tile_coords\|emit_lora_tile_bases\|emit_matmul_mma_lane_init' \
    crates/ --include='*.rs' 2>&1 | sort > /tmp/b31_callers_after.txt
diff /tmp/b31_callers_before.txt /tmp/b31_callers_after.txt
```

Expected: the diff shows EXACTLY ONE new line — a reference from `emit_fused_adapter_kernel_body` in `wrga_fused_ptx.rs`. Every pre-existing call site is preserved.

If other diffs appear (lost references from other files, unexpected new callers): investigate before commit.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/wrga_fused_ptx.rs
git commit -m "refactor(wrga): synthesize_fused_lora_ptx calls shared body

LoRA synthesizer now delegates to emit_fused_adapter_kernel_body with
FoldKind::Scalar.  Behavior byte-identical: verified by the full 11-test
wrga_adapter_runtime_equivalence suite including real-launch numerical
and launch-counter hardening tests.

Register-collision pre-audit: <zero matches | matches found, %gl_ prefix applied>
%n_real audit: <already shared | promoted this commit>
Internal-caller stability: diff clean (one new ref inside shared body only)"
```

### Task 1.5: Task Group 1 smoke + marker (optional consolidation)

- [ ] **Step 1: Full test-suite smoke**

```bash
cargo test -p nsl-codegen --lib 2>&1 | tail -5
cargo test -p nsl-codegen --test wrga_fused_ptx_ptxas --features cuda 2>&1 | tail -5
cargo test -p nsl-codegen --test fa_v2_snapshots 2>&1 | tail -5
cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence --features cuda -- --test-threads=1 2>&1 | tail -5
```

Expected:
- nsl-codegen lib: all green
- wrga_fused_ptx_ptxas: 11 passed
- fa_v2_snapshots: 17 passed / 6 pre-existing failures (CSHA Tier C drift, out of scope)
- wrga_adapter_runtime_equivalence: 11 passed

- [ ] **Step 2: (Optional) Squash Task Group 1 intermediate commits**

If a clean single-commit boundary for spec commit 1 is preferred:

```bash
git rebase -i HEAD~3  # Task 1.2, 1.3, 1.4 commits
# Keep Task 1.4's commit message (the behavior-identity proof); squash 1.2 and 1.3 into it
```

Otherwise leave the 3 intermediate commits in place for bisect granularity.

---

## Task Group 2 — Commit 2: Sigmoid + gate helpers

### Task 2.1: `emit_sigmoid_approx_fused` helper

**Files:**
- Modify: `crates/nsl-codegen/src/wrga_kernel_helpers.rs`

- [ ] **Step 1: Append the helper**

Append at the end of `crates/nsl-codegen/src/wrga_kernel_helpers.rs`:

```rust
/// Emit a PTX fp32 sigmoid approximation using the hardware approx units.
///
/// Produces `output = 1 / (1 + exp(-input))` as 4 PTX instructions:
///   mul.f32  %sig_tmp, <input>, 0fBFB8AA3B;   // * -log2(e) = -1.4426950408
///   ex2.approx.f32 %sig_tmp, %sig_tmp;         // e^(-input)
///   add.f32  %sig_tmp, %sig_tmp, 0f3F800000;   // + 1.0
///   rcp.approx.f32 <output>, %sig_tmp;         // sigmoid(input)
///
/// Hex constants:
///   0fBFB8AA3B = -log2(e) as f32 (sign=1, exp=0x7F, mantissa=0x38AA3B).
///     DO NOT use 0f3FB8AA3B (= +log2(e), silently computes sigmoid(-input))
///     DO NOT use 0f3FBE2FB9 (= +1.486, historical bad spec-draft value).
///   0f3F800000 = 1.0 as f32.
///
/// Requires: the caller has declared `%sig_tmp` as `.reg .f32` in the
/// kernel's register pool.  For GatedLoRA, this is declared in
/// `emit_fused_adapter_kernel_body`'s PerColumnSigmoid branch.
pub fn emit_sigmoid_approx_fused(ptx: &mut String, input: &str, output: &str) {
    ptx.push_str(&format!(
        "    mul.f32  %sig_tmp, {input}, 0fBFB8AA3B;   // * -log2(e)\n"
    ));
    ptx.push_str("    ex2.approx.f32 %sig_tmp, %sig_tmp;\n");
    ptx.push_str("    add.f32  %sig_tmp, %sig_tmp, 0f3F800000;   // + 1.0\n");
    ptx.push_str(&format!(
        "    rcp.approx.f32 {output}, %sig_tmp;\n"
    ));
}
```

- [ ] **Step 2: Write structural test**

Append to the same file inside a `#[cfg(test)] mod tests { ... }` block (create one if it doesn't exist):

```rust
#[cfg(test)]
mod sigmoid_tests {
    use super::emit_sigmoid_approx_fused;

    #[test]
    fn emit_sigmoid_approx_fused_has_correct_log2e_constant() {
        let mut ptx = String::new();
        emit_sigmoid_approx_fused(&mut ptx, "%in", "%out");

        // Critical: the mul line must use the NEGATIVE log2(e) constant.
        let mul_line = ptx.lines()
            .find(|l| l.contains("mul.f32") && l.contains("%sig_tmp"))
            .expect("expected mul.f32 %sig_tmp line in emission");
        assert!(
            mul_line.contains("0fBFB8AA3B"),
            "mul line must use -log2(e) = 0fBFB8AA3B; got: {mul_line}"
        );

        // Belt-and-suspenders: reject the historical bad value.
        assert!(
            !ptx.contains("0f3FBE2FB9"),
            "regression to spec-draft bad positive constant 0f3FBE2FB9:\n{ptx}"
        );
        // Also reject the sign-flipped case.
        assert!(
            !ptx.contains("mul.f32  %sig_tmp, %in, 0f3FB8AA3B"),
            "sign-flipped +log2(e) constant detected; would compute sigmoid(-input):\n{ptx}"
        );

        // Structural: exactly one ex2.approx and one rcp.approx.
        assert_eq!(ptx.matches("ex2.approx.f32").count(), 1);
        assert_eq!(ptx.matches("rcp.approx.f32").count(), 1);
        assert!(ptx.contains("0f3F800000"), "missing +1.0 constant");
        assert!(ptx.contains("%in"), "missing input register reference");
        assert!(ptx.contains("%out"), "missing output register reference");
    }
}
```

- [ ] **Step 3: Run the test**

```bash
cargo test -p nsl-codegen --lib wrga_kernel_helpers::sigmoid_tests 2>&1 | tail -5
```

Expected: `1 passed / 0 failed`.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/wrga_kernel_helpers.rs
git commit -m "feat(wrga): emit_sigmoid_approx_fused helper + structural test

Emits 4 PTX instructions for fp32 sigmoid via fused rcp.approx(1 +
ex2.approx(-x * log2e)).  Hex constants pinned: 0fBFB8AA3B for
-log2(e), 0f3F800000 for 1.0.

Structural test asserts positively on 0fBFB8AA3B in the mul-line AND
negatively on the historical bad values 0f3FBE2FB9 and 0f3FB8AA3B.
This is the test that catches both 'regression to draft constant'
and 'sign-flipped constant silently computes sigmoid(-x)' bug classes."
```

### Task 2.2: `emit_gate_load_per_thread` helper

**Files:**
- Modify: `crates/nsl-codegen/src/wrga_kernel_helpers.rs`

- [ ] **Step 1: Append the helper**

Append after `emit_sigmoid_approx_fused`:

```rust
/// Emit a per-thread gate load for the GatedLoRA fused kernel.
///
/// Each thread owns exactly 2 output columns in the m16n8k16 D-fragment
/// (cols `col_base_in_tile + 0` and `+1`, where `col_base_in_tile = (tid%4)*2`).
/// This helper emits the address arithmetic + 2 `ld.global.f32` calls
/// to load those 2 gate values into `%gate0` and `%gate1`.
///
/// Requires (caller must have declared / initialized before this call):
///   %rd_gate      — loaded from .param .u64 gate_ptr at prolog
///   %col_base     — output-tile column base (from emit_lora_output_tile_coords)
///   %r5           — col_base_in_tile = (tid%4)*2 (from emit_matmul_mma_lane_init
///                    OR from the store preamble; verify the setter runs first)
///   %r_gate       — declared .reg .u32
///   %rd_gate_addr — declared .reg .u64
///   %gate0, %gate1 — declared .reg .f32
///
/// Warp-broadcast alternative explicitly avoided: would 4× HBM gate traffic
/// for no benefit since each thread consumes only 2 of 8 columns.  See
/// invariant #13 in project_wrga_fused_ptx_rewrite.md.
///
/// TODO(gate-dtype): the `shl.b32 %r_gate, %r_gate, 2` assumes f32 gate.
/// If gate dtype ever narrows to f16, shift constant becomes 1.
pub fn emit_gate_load_per_thread(ptx: &mut String) {
    ptx.push_str("    // GatedLoRA: per-thread gate load — 2 cols per thread\n");
    ptx.push_str("    add.u32  %r_gate, %col_base, %r5;\n");
    ptx.push_str("    shl.b32  %r_gate, %r_gate, 2;   // * 4 bytes (f32)\n");
    ptx.push_str("    cvt.u64.u32 %rd_gate_addr, %r_gate;\n");
    ptx.push_str("    add.u64  %rd_gate_addr, %rd_gate, %rd_gate_addr;\n");
    ptx.push_str("    ld.global.f32 %gate0, [%rd_gate_addr];\n");
    ptx.push_str("    ld.global.f32 %gate1, [%rd_gate_addr + 4];\n");
}
```

- [ ] **Step 2: Structural test**

Append to `sigmoid_tests` mod (rename to `helpers_tests` or split; here we keep one combined mod):

```rust
    #[test]
    fn emit_gate_load_per_thread_emits_two_f32_loads_with_correct_stride() {
        use super::emit_gate_load_per_thread;
        let mut ptx = String::new();
        emit_gate_load_per_thread(&mut ptx);

        // Exactly 2 ld.global.f32 — one per gate value.
        assert_eq!(
            ptx.matches("ld.global.f32").count(), 2,
            "expected 2 per-thread gate loads; got:\n{ptx}"
        );

        // Byte-stride must be 4 (f32); catches gate-dtype regression.
        assert!(
            ptx.contains("shl.b32  %r_gate, %r_gate, 2"),
            "gate offset must shift-left by 2 (f32 stride); got:\n{ptx}"
        );

        // Second load is at offset +4 (second gate value).
        assert!(
            ptx.contains("ld.global.f32 %gate1, [%rd_gate_addr + 4]"),
            "second gate load missing or wrong offset:\n{ptx}"
        );

        // Base pointer reference.
        assert!(
            ptx.contains("add.u64  %rd_gate_addr, %rd_gate, %rd_gate_addr"),
            "gate address must sum base + offset:\n{ptx}"
        );
    }
```

- [ ] **Step 3: Run the test**

```bash
cargo test -p nsl-codegen --lib wrga_kernel_helpers 2>&1 | tail -5
```

Expected: 2 passed (the sigmoid test from 2.1 + this new one).

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/wrga_kernel_helpers.rs
git commit -m "feat(wrga): emit_gate_load_per_thread helper + structural test

Per-thread gate load for the GatedLoRA fused kernel.  Each thread loads
exactly 2 f32 gate values matching its m16n8k16 D-fragment column
ownership.  NOT a warp-broadcast load — see invariant #13.

Structural test asserts: 2 ld.global.f32 emissions, f32 byte stride
(shl.b32 by 2), second-load offset = +4."
```

### Task 2.3: `emit_gatedlora_fold` helper

**Files:**
- Modify: `crates/nsl-codegen/src/wrga_kernel_helpers.rs`

- [ ] **Step 1: Append the helper**

Append after `emit_gate_load_per_thread`:

```rust
/// Emit the GatedLoRA fold step: per-thread, per-column fold of the
/// epilogue accumulator into `main_accum` with predicated OOB masking.
///
///   main_accum0 += %epi_final0 * %gate0 * %scale_reg   (col 0, row 0)
///   main_accum2 += %epi_final2 * %gate0 * %scale_reg   (col 0, row 8)
///   main_accum1 += %epi_final1 * %gate1 * %scale_reg   (col 1, row 0)
///   main_accum3 += %epi_final3 * %gate1 * %scale_reg   (col 1, row 8)
///
/// Each add is predicated on `%p_col0` / `%p_col1` (computed from `%n_real`)
/// so OOB output columns in partial tiles don't corrupt main_accum.
///
/// Requires (caller must have declared / initialized before this call):
///   %r5           — col_base_in_tile = (tid%4)*2 (from emit_matmul_mma_lane_init)
///   %n_real       — real n for this output tile (from emit_lora_output_tile_coords)
///   %gate0, %gate1 — sigmoid(gate) values (from emit_sigmoid_approx_fused)
///   %scale_reg    — loaded from .param .f32 scale at prolog
///   %epi_final0..3 — declared (4 f32 from the post-loop (x@A)@B MMA)
///   %main_accum0..3 — declared (4 f32 from the main MMA accumulator)
///   %r_col1       — declared .reg .u32
///   %fold_tmp     — declared .reg .f32
///   %p_col0, %p_col1 — declared .reg .pred
///
/// See invariants #13-#15 for the per-thread / LastKIter / FoldResultMask
/// correctness requirements.
pub fn emit_gatedlora_fold(ptx: &mut String) {
    ptx.push_str("    // GatedLoRA fold: main_accum += epi_final * sigmoid(gate) * scale\n");
    ptx.push_str("    //   predicated on n_real bounds to skip OOB cols in partial tiles\n");
    ptx.push_str("    setp.lt.u32  %p_col0, %r5, %n_real;\n");
    ptx.push_str("    add.u32      %r_col1, %r5, 1;\n");
    ptx.push_str("    setp.lt.u32  %p_col1, %r_col1, %n_real;\n");

    // main_accum0 (col 0, row 0), main_accum2 (col 0, row 8) — gate0, p_col0
    ptx.push_str("    mul.f32      %fold_tmp, %epi_final0, %gate0;\n");
    ptx.push_str("    mul.f32      %fold_tmp, %fold_tmp, %scale_reg;\n");
    ptx.push_str("    @%p_col0 add.f32 %main_accum0, %main_accum0, %fold_tmp;\n");
    ptx.push_str("    mul.f32      %fold_tmp, %epi_final2, %gate0;\n");
    ptx.push_str("    mul.f32      %fold_tmp, %fold_tmp, %scale_reg;\n");
    ptx.push_str("    @%p_col0 add.f32 %main_accum2, %main_accum2, %fold_tmp;\n");

    // main_accum1 (col 1, row 0), main_accum3 (col 1, row 8) — gate1, p_col1
    ptx.push_str("    mul.f32      %fold_tmp, %epi_final1, %gate1;\n");
    ptx.push_str("    mul.f32      %fold_tmp, %fold_tmp, %scale_reg;\n");
    ptx.push_str("    @%p_col1 add.f32 %main_accum1, %main_accum1, %fold_tmp;\n");
    ptx.push_str("    mul.f32      %fold_tmp, %epi_final3, %gate1;\n");
    ptx.push_str("    mul.f32      %fold_tmp, %fold_tmp, %scale_reg;\n");
    ptx.push_str("    @%p_col1 add.f32 %main_accum3, %main_accum3, %fold_tmp;\n");
}
```

- [ ] **Step 2: Structural test**

```rust
    #[test]
    fn emit_gatedlora_fold_masks_correctly_per_column() {
        use super::emit_gatedlora_fold;
        let mut ptx = String::new();
        emit_gatedlora_fold(&mut ptx);

        // Predicate setup: two setp.lt.u32 for the two-column fold.
        assert_eq!(
            ptx.matches("setp.lt.u32").count(), 2,
            "expected 2 setp.lt.u32 for %p_col0 and %p_col1; got:\n{ptx}"
        );

        // 4 predicated adds, one per main_accum[0..3].
        assert_eq!(
            ptx.matches("@%p_col").count(), 4,
            "expected 4 predicated adds (main_accum0..3); got:\n{ptx}"
        );

        // Column-parity wiring: main_accum0,2 gated by p_col0; main_accum1,3 gated by p_col1.
        assert!(
            ptx.contains("@%p_col0 add.f32 %main_accum0"),
            "main_accum0 (col 0) must be gated by p_col0"
        );
        assert!(
            ptx.contains("@%p_col0 add.f32 %main_accum2"),
            "main_accum2 (col 0, row 8) must be gated by p_col0"
        );
        assert!(
            ptx.contains("@%p_col1 add.f32 %main_accum1"),
            "main_accum1 (col 1) must be gated by p_col1"
        );
        assert!(
            ptx.contains("@%p_col1 add.f32 %main_accum3"),
            "main_accum3 (col 1, row 8) must be gated by p_col1"
        );

        // Gate wiring: main_accum0/2 use %gate0; main_accum1/3 use %gate1.
        assert!(
            ptx.contains("mul.f32      %fold_tmp, %epi_final0, %gate0"),
            "main_accum0 fold must multiply by %gate0"
        );
        assert!(
            ptx.contains("mul.f32      %fold_tmp, %epi_final1, %gate1"),
            "main_accum1 fold must multiply by %gate1"
        );

        // Scale multiplication present (exactly once per main_accum position = 4 times).
        assert_eq!(
            ptx.matches("mul.f32      %fold_tmp, %fold_tmp, %scale_reg").count(), 4,
            "expected 4 scale multiplications; got:\n{ptx}"
        );
    }
```

- [ ] **Step 3: Run**

```bash
cargo test -p nsl-codegen --lib wrga_kernel_helpers 2>&1 | tail -5
```

Expected: 3 passed (all three structural tests).

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/wrga_kernel_helpers.rs
git commit -m "feat(wrga): emit_gatedlora_fold helper + structural test

Per-thread, per-column fold with %p_col0/%p_col1 predication.  Gate
wiring: main_accum0/2 use %gate0, main_accum1/3 use %gate1 (matching
m16n8k16 D-fragment column ownership).  Scale via %scale_reg (reused
from LoRA's existing prolog load).

Structural test asserts: 2 setp.lt.u32 for predicate setup, 4
predicated adds, correct column-parity wiring (0/2→p_col0,
1/3→p_col1), correct gate wiring (0/2→gate0, 1/3→gate1), and 4
scale multiplications."
```

---

## Task Group 3 — Commit 3: GatedLoRA ptxas red

### Task 3.1: Add `FusedGatedLoraConfig` struct

**Files:**
- Modify: `crates/nsl-codegen/src/wrga_fused_ptx.rs`

- [ ] **Step 1: Append the struct**

Add next to `FusedLoraConfig`:

```rust
/// Config for synthesize_fused_gatedlora_ptx.  Mirrors FusedLoraConfig
/// except the kernel it generates emits the PerColumnSigmoid fold variant.
/// Separate type so kernel-dedup and dispatch don't confuse LoRA and
/// GatedLoRA instances (different kernel_handle = different PTX registry
/// entry).
#[derive(Debug, Clone)]
pub struct FusedGatedLoraConfig {
    pub site_id: String,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub rank: u32,
    pub alpha: u32,
    pub target_sm: u32,
}
```

Check if `FusedLoraConfig` has an `alpha` field; if not, match its exact fields (likely `scale: f32` directly instead of `alpha: u32`). Whatever LoRA uses, mirror it here.

- [ ] **Step 2: Build**

```bash
cargo build -p nsl-codegen 2>&1 | tail -3
```

Expected: clean build (the struct is unused but that's OK — `synthesize_fused_gatedlora_ptx` comes in Step 3).

- [ ] **Step 3: Add stub `synthesize_fused_gatedlora_ptx`**

Append at the end of `wrga_fused_ptx.rs`:

```rust
/// Synthesize PTX for a fused GatedLoRA forward adapter kernel.
///
/// Computes `y[i,j] = (x @ W)[i,j] + sigmoid(gate[j]) * ((x @ A) @ B)[i,j] * scale`
/// where gate is a per-output-column f32 vector.
///
/// Uses the shared `emit_fused_adapter_kernel_body` with
/// `FoldKind::PerColumnSigmoid` to reuse LoRA's proven kernel body plus
/// gate-load + sigmoid + per-column fold logic from wrga_kernel_helpers.
///
/// STUB — current body returns invalid PTX (empty entry) for Task 3's
/// red-state test setup.  Task 4 replaces this with the real emission.
pub fn synthesize_fused_gatedlora_ptx(config: &FusedGatedLoraConfig) -> String {
    assert!(
        config.rank <= 16,
        "B.3 rank ceiling: {} > 16; multi-pass epilogue is a follow-up milestone",
        config.rank,
    );
    assert!(config.target_sm >= 80, "B.3 requires sm_80+");

    // STUB: intentionally invalid PTX to establish red test state.
    // Task 4 replaces this with emit_fused_adapter_kernel_body(.., FoldKind::PerColumnSigmoid).
    let _ = config;
    String::from(".version 7.0\n.target sm_80\n.address_size 64\n\n// STUB — Task 4 replaces this\n")
}
```

- [ ] **Step 4: Build**

```bash
cargo build -p nsl-codegen 2>&1 | tail -3
```

Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/wrga_fused_ptx.rs
git commit -m "feat(wrga): FusedGatedLoraConfig struct + synthesize_fused_gatedlora_ptx stub

Stub returns a header-only PTX to establish red test state for Task 3.
Real emission body lands in Task 4."
```

### Task 3.2: Add 10 GatedLoRA ptxas tests (red)

**Files:**
- Modify: `crates/nsl-codegen/tests/wrga_fused_ptx_ptxas.rs`

- [ ] **Step 1: Read current test file structure**

```bash
wc -l crates/nsl-codegen/tests/wrga_fused_ptx_ptxas.rs
grep -n '^fn \|^#\[test\]' crates/nsl-codegen/tests/wrga_fused_ptx_ptxas.rs | head -30
```

Note the existing patterns for naming and skip-on-no-validator behavior.

- [ ] **Step 2: Append GatedLoRA helper + 10 tests**

Append at the end of `crates/nsl-codegen/tests/wrga_fused_ptx_ptxas.rs`:

```rust
// ─── GatedLoRA ptxas validation — Task Group 3 ─────────────────────

use nsl_codegen::wrga_fused_ptx::{synthesize_fused_gatedlora_ptx, FusedGatedLoraConfig};

fn gated_cfg(m: u32, n: u32, k: u32, rank: u32) -> FusedGatedLoraConfig {
    FusedGatedLoraConfig {
        site_id: format!("gated.m{m}n{n}k{k}r{rank}"),
        m, n, k, rank,
        alpha: rank,  // scale = alpha/rank = 1.0; adjust if FusedGatedLoraConfig uses scale: f32 directly
        target_sm: 80,
    }
}

fn assert_gatedlora_ptx_valid(cfg: FusedGatedLoraConfig) {
    let ptx = synthesize_fused_gatedlora_ptx(&cfg);
    match validate_ptx(&ptx) {
        Ok(()) => {}
        Err(msg) if msg.contains("nvcc not available") => {
            eprintln!(
                "[skip] GatedLoRA ptxas validation for ({},{},{},r={}) — no validator: {}",
                cfg.m, cfg.n, cfg.k, cfg.rank, msg,
            );
        }
        Err(msg) => panic!(
            "GatedLoRA PTX rejected for config (m={}, n={}, k={}, rank={}):\n{}\n\nEmitted PTX:\n{}",
            cfg.m, cfg.n, cfg.k, cfg.rank, msg, ptx
        ),
    }
}

// ── Reused LoRA shapes under FoldKind::PerColumnSigmoid (6) ──
#[test]
fn gatedlora_ptx_validates__16_8_16_16() {
    assert_gatedlora_ptx_valid(gated_cfg(16, 8, 16, 16));
}

#[test]
fn gatedlora_ptx_validates__16_8_32_4() {
    assert_gatedlora_ptx_valid(gated_cfg(16, 8, 32, 4));
}

#[test]
fn gatedlora_ptx_validates__1_8_8_2() {
    assert_gatedlora_ptx_valid(gated_cfg(1, 8, 8, 2));
}

#[test]
fn gatedlora_ptx_validates__4_8_8_2() {
    assert_gatedlora_ptx_valid(gated_cfg(4, 8, 8, 2));
}

#[test]
fn gatedlora_ptx_validates__32_16_64_8() {
    assert_gatedlora_ptx_valid(gated_cfg(32, 16, 64, 8));
}

#[test]
fn gatedlora_ptx_validates__16_8_8_16() {
    assert_gatedlora_ptx_valid(gated_cfg(16, 8, 8, 16));
}

// ── GatedLoRA-distinctive configs (4) ──
// Note: ptxas validation doesn't distinguish gate VALUES (runtime concern);
// these tests validate that the emitted PTX structure is valid across
// shape-and-feature combinations.
#[test]
fn gatedlora_ptx_validates__uniform_gate_zero_16_8_16_16() {
    // Same shape as canonical; gate values are a runtime fixture concern.
    assert_gatedlora_ptx_valid(gated_cfg(16, 8, 16, 16));
}

#[test]
fn gatedlora_ptx_validates__alternating_saturation_16_8_16_16() {
    // Same shape; gate pattern asserted in Task Group 5 integration fixtures.
    assert_gatedlora_ptx_valid(gated_cfg(16, 8, 16, 16));
}

#[test]
fn gatedlora_ptx_validates__partial_n_multi_tile_16_13_16_4() {
    // n=13 → 2 tiles (tile 0 full n=8; tile 1 partial n=5).
    // This is the ONLY config that actually evaluates FoldResultMask
    // predicates false at runtime.  Spec §3.1 fixes config 6.
    assert_gatedlora_ptx_valid(gated_cfg(16, 13, 16, 4));
}

#[test]
fn gatedlora_ptx_validates__sub_mma_k_no_rank_pad_16_8_8_16() {
    // Sub-MMA K (k=8<16) + rank==MMA-k (no rank-pad path).
    assert_gatedlora_ptx_valid(gated_cfg(16, 8, 8, 16));
}
```

Note: the first 6 "reused LoRA shapes" configs intentionally duplicate shape-coverage with the 4 "distinctive" configs. The spec §3.1 labels this as "10 new ptxas tests" total. Three of the first six (`16_8_16_16`, `16_8_16_16`, `16_8_8_16`) reappear in the "distinctive" quartet — that's intentional repetition (different test names but same underlying config) to make the coverage intent explicit in the test listing. If this feels redundant, collapse to the minimal 7 unique configs and label them accordingly; the spec permits either. Keep 10 for strict traceability to spec §3.1's table.

- [ ] **Step 3: Run the tests — confirm RED**

```bash
cargo test -p nsl-codegen --test wrga_fused_ptx_ptxas --features cuda gatedlora_ptx_validates 2>&1 | tail -30
```

Expected: **all 10 GatedLoRA tests FAIL** with ptxas errors (empty entry / missing body). Record the failure mode in the commit message.

If any test PASSES against the stub, the stub somehow produces valid PTX — check the stub body; it MUST be invalid.

If all tests SKIP (`nvcc not available`), the validator isn't available on this machine; the red state is undemonstrated. Document this explicitly; continue.

Run the existing LoRA/IA³ ptxas tests to confirm no regression:

```bash
cargo test -p nsl-codegen --test wrga_fused_ptx_ptxas --features cuda 2>&1 | grep -E "passed|failed"
```

Expected: 11 passed (LoRA 6 + IA³ 5) + 10 failed (GatedLoRA reds) = 21 total / 10 failed.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/tests/wrga_fused_ptx_ptxas.rs
git commit -m "test(wrga): GatedLoRA ptxas validation (red, 10 configs)

10 new ptxas tests invoking public synthesize_fused_gatedlora_ptx.  All
fail against the stub from Task 3.1 with ptxas rejecting the empty
entry body.

Coverage: 6 LoRA shapes under PerColumnSigmoid + 4 GatedLoRA-distinctive
configs (uniform gate=0, alternating ±30 saturation, partial-n (16,13,16,4),
sub-MMA K no rank-pad).  Per spec §3.1.

Task 4 flips these 10 red to green."
```

---

## Task Group 4 — Commit 4: GatedLoRA emitter

### Task 4.1: Implement `synthesize_fused_gatedlora_ptx` body

**Files:**
- Modify: `crates/nsl-codegen/src/wrga_fused_ptx.rs`

- [ ] **Step 1: Replace the stub body**

Find the stub `pub fn synthesize_fused_gatedlora_ptx(config: &FusedGatedLoraConfig) -> String` and replace its body:

```rust
pub fn synthesize_fused_gatedlora_ptx(config: &FusedGatedLoraConfig) -> String {
    assert!(
        config.rank <= 16,
        "B.3 rank ceiling: {} > 16; multi-pass epilogue is a follow-up milestone",
        config.rank,
    );
    assert!(config.target_sm >= 80, "B.3 requires sm_80+");

    let entry_name = format!(
        "nsl_wrga_fused_gatedlora_m{}n{}k{}r{}",
        config.m, config.n, config.k, config.rank,
    );
    let scale = config.alpha as f32 / config.rank as f32;
    let mut ptx = String::new();
    emit_fused_adapter_kernel_body(
        &mut ptx,
        &entry_name,
        config.m,
        config.n,
        config.k,
        config.rank,
        FoldKind::PerColumnSigmoid {
            scale,
            gate_ptr_param_name: "gate_ptr",
            gate_load_phase: GateLoadPhase::LastKIter,
            partial_tile_mask: PartialTileMask::FoldResultMask,
        },
    );
    ptx
}
```

If `FusedGatedLoraConfig` uses `scale: f32` directly instead of `alpha: u32`, replace `config.alpha as f32 / config.rank as f32` with `config.scale`.

- [ ] **Step 2: Build**

```bash
cargo build -p nsl-codegen 2>&1 | tail -5
```

Expected: clean build.

- [ ] **Step 3: Run the 10 GatedLoRA ptxas tests — CRITICAL GATE**

```bash
cargo test -p nsl-codegen --test wrga_fused_ptx_ptxas --features cuda gatedlora_ptx_validates 2>&1 | tail -15
```

Expected: **all 10 tests PASS (red → green)**.

If any fail, read the error carefully:
- `Missing declaration for %foo` — a register name wasn't declared in the pool; check the conditional register block in `emit_fused_adapter_kernel_body` (Task 1.3 Step 3).
- `Unknown symbol 'gate_ptr'` — param not declared; check `is_gated` branch in the param block emission.
- `Illegal expression 'gate0'` — register reference missing `%` prefix; check `emit_gate_load_per_thread` or `emit_gatedlora_fold`.
- `Operand type mismatch` — `u32`/`u64`/`b32` mixing; check the gate-address arithmetic.
- `undefined label` or similar — bar.sync scope or branch target issue.

For each failure, dump the emitted PTX via a one-off `#[test]` that prints it, and compare against the spec §2 emission sequence.

- [ ] **Step 4: Full ptxas regression sweep**

```bash
cargo test -p nsl-codegen --test wrga_fused_ptx_ptxas --features cuda 2>&1 | tail -10
```

Expected: **21 passed / 0 failed** (6 LoRA + 5 IA³ + 10 GatedLoRA).

- [ ] **Step 5: LoRA integration regression**

```bash
cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence --features cuda -- --test-threads=1 2>&1 | tail -5
```

Expected: **11 passed / 0 failed / 0 ignored** (no change from baseline; GatedLoRA integration fixtures come in Task 5).

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/wrga_fused_ptx.rs
git commit -m "feat(wrga): synthesize_fused_gatedlora_ptx against shared body + PerColumnSigmoid

Replaces Task 3.1's stub with the real emission, calling
emit_fused_adapter_kernel_body with FoldKind::PerColumnSigmoid.

10 red ptxas tests from Task 3.2 flip green.  LoRA and IA³ regression
gates unchanged (11/0/0 in wrga_adapter_runtime_equivalence, 11 passed
in wrga_fused_ptx_ptxas including existing 11 LoRA+IA³ configs)."
```

---

## Task Group 5 — Commit 5: Integration fixtures

### Task 5.1: Fixture A — baseline (gate=0, y=16.0 at 1e-4)

**Files:**
- Modify: `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs`

- [ ] **Step 1: Append Fixture A**

Append at the end of `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs`:

```rust
// ─── Task Group 5: GatedLoRA integration fixtures ──────────────────

/// Fixture A — baseline. alpha=2, rank=2, scale=alpha/rank=1.0.
/// x=ones([4,8]), W=ones([8,8]), A=ones([8,2]), B=ones([2,8]), gate=zeros([8]).
/// y[i,j] = (x@W)[i,j] + sigmoid(gate[j]) * ((x@A)@B)[i,j] * scale
///        = 8         + sigmoid(0)       * 16              * 1.0
///        = 8 + 0.5 * 16
///        = 16.0 per element.
const GATEDLORA_FIXTURE_A_SRC: &str = r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = ones([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=gated_lora, target=["Toy.w"], rank=2, alpha=2)
let m = Toy()
m.to(cuda)
let x = ones([4, 8]).to(cuda)
let y_target = zeros([4, 8]).to(cuda)
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
m.lora_A_Toy_w__gatedlora = ones([8, 2]).to(cuda)
m.lora_B_Toy_w__gatedlora = ones([2, 8]).to(cuda)
m.gate_Toy_w__gatedlora = zeros([8]).to(cuda)
let y = m.forward(x)
print(y)
"#;

#[cfg(feature = "cuda")]
#[test]
fn gatedlora_fixture_a_baseline() {
    run_gatedlora_fixture(GATEDLORA_FIXTURE_A_SRC, 16.0, 1e-4, "fixture_a");
}

#[cfg(feature = "cuda")]
fn run_gatedlora_fixture(src: &str, expected: f32, tolerance: f32, fixture_name: &str) {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join(format!("{}.nsl", fixture_name));
    fs::write(&src_path, src).unwrap();
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
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    assert!(
        output.status.success(),
        "{fixture_name} nsl run failed.\nstdout:\n{stdout}\nstderr:\n{stderr}",
    );

    let tensor = parse_tensor_2d(&stdout);
    assert_eq!(tensor.len(), 4, "{fixture_name}: expected 4 rows");

    let mut max_diff: f32 = 0.0;
    for row in &tensor {
        assert_eq!(row.len(), 8, "{fixture_name}: expected 8 cols");
        for v in row {
            max_diff = max_diff.max((v - expected).abs());
        }
    }
    assert!(
        max_diff < tolerance,
        "{fixture_name}: max |y - {expected}| = {max_diff:.3e}, want < {tolerance:.0e}.\n\
         Tensor: {tensor:?}\nstderr:\n{stderr}"
    );

    // Assert fused PTX actually fired (not CPU fallback).
    let gpu_count_line = stderr.lines().find(|l| l.contains("[nsl-gpu-launch-count]"));
    let count: u64 = gpu_count_line
        .and_then(|l| l.split_whitespace().next_back())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    assert!(
        count >= 1,
        "{fixture_name}: expected >=1 fused CUDA launch, got {count}.\nstderr:\n{stderr}"
    );
}
```

Note on `@adapter(type=gated_lora, ...)` syntax: verify via `grep -n 'type=gated_lora\|AdapterKind::GatedLora' crates/nsl-semantic/src/` for the exact NSL-surface-syntax. If the surface form differs (e.g., `type=gatedlora`), adjust the source literal.

Note on `gate_Toy_w__gatedlora` field name: verify via the B.2.1 side-table naming convention grep `grep -n 'gate_' crates/nsl-codegen/src/wrga_adapter_inject.rs`. The format is `gate_<Model>_<field>__<adapter_id>` but the separator exact details matter for seeding.

- [ ] **Step 2: Build**

```bash
cargo build -p nsl-cli --features cuda 2>&1 | tail -3
```

Expected: clean build.

- [ ] **Step 3: Run the fixture**

```bash
cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence --features cuda gatedlora_fixture_a_baseline -- --test-threads=1 --nocapture 2>&1 | tail -30
```

Expected: `1 passed / 0 failed`, `max_diff < 1e-4`, `[nsl-gpu-launch-count] >= 1`.

Debug guide if it fails:
- `max_diff ≈ 16.0` (y=0): adapter path failed entirely — check fused kernel isn't falling back; inspect stderr for `[nsl-wrga] fused PTX launch failed`.
- `max_diff ≈ 8.0` (y=8): adapter delta not folded — check `emit_gatedlora_fold` emission and the scale multiplication.
- `max_diff ≈ 8.0` (y=24): sigmoid returned 1 instead of 0.5 — check the `emit_sigmoid_approx_fused` constants.
- `count = 0`: CPU fallback fired instead of real PTX; check `NSL_WRGA_FUSED_CUDA=1` env, and stderr for fallback warnings.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs
git commit -m "test(wrga): Fixture A baseline (gate=0, y=16.0 at 1e-4)

Proves the GatedLoRA fused kernel runs end-to-end with gate=0 producing
sigmoid(0)=0.5 folded into main_accum with scale=1.0.

y[i,j] = (x@W = 8) + 0.5 * (x@A@B = 16) * 1.0 = 16.0."
```

### Task 5.2: Fixture B — positive saturation (gate=+30, y=24.0 at 1e-4)

**Files:**
- Modify: `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs`

- [ ] **Step 1: Append Fixture B**

Append after Fixture A:

```rust
/// Fixture B — positive saturation. alpha=2, rank=2, scale=1.0.
/// Same model/x as Fixture A; gate = full([8], 30.0).  sigmoid(30) ≈ 1.0 exactly in fp32
/// (ex2.approx(-30 * log2e) = ex2.approx(-43.3) ≈ 2^-43 → underflows to 0 in fp32,
/// yielding 1/(1+0) = 1.0 exactly).
/// y[i,j] = 8 + 1.0 * 16 * 1.0 = 24.0 per element.
const GATEDLORA_FIXTURE_B_SRC: &str = r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = ones([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=gated_lora, target=["Toy.w"], rank=2, alpha=2)
let m = Toy()
m.to(cuda)
let x = ones([4, 8]).to(cuda)
let y_target = zeros([4, 8]).to(cuda)
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
m.lora_A_Toy_w__gatedlora = ones([8, 2]).to(cuda)
m.lora_B_Toy_w__gatedlora = ones([2, 8]).to(cuda)
m.gate_Toy_w__gatedlora = full([8], 30.0).to(cuda)
let y = m.forward(x)
print(y)
"#;

#[cfg(feature = "cuda")]
#[test]
fn gatedlora_fixture_b_positive_saturation() {
    run_gatedlora_fixture(GATEDLORA_FIXTURE_B_SRC, 24.0, 1e-4, "fixture_b");
}
```

- [ ] **Step 2: Run**

```bash
cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence --features cuda gatedlora_fixture_b -- --test-threads=1 --nocapture 2>&1 | tail -10
```

Expected: `1 passed`, `max_diff < 1e-4`, `y = 24.0` per element.

Debug guide:
- `max_diff ≈ 8.0` (y=16): sigmoid returned 0.5 instead of 1 — saturation not reaching the asymptote; check `ex2.approx` constants or input magnitude handling.

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs
git commit -m "test(wrga): Fixture B positive saturation (gate=+30, y=24.0 at 1e-4)

Proves sigmoid saturation reaches 1.0 at fp32 precision.  ex2.approx(-43)
underflows to 0, yielding 1/(1+0) = 1.0 exactly; adapter contributes
full 16.0 atop base 8.0."
```

### Task 5.3: Fixture C — negative saturation (gate=-30, y=8.0 at 1e-4)

- [ ] **Step 1: Append Fixture C**

```rust
/// Fixture C — negative saturation. alpha=2, rank=2, scale=1.0.
/// gate = full([8], -30.0).  sigmoid(-30) ≈ 0.0 exactly (ex2.approx(+43) overflows
/// in denominator direction; 1/huge ≈ 0.0 in fp32).
/// y[i,j] = 8 + 0.0 * 16 * 1.0 = 8.0 per element.
///
/// DIAGNOSTIC: this fixture's base-only value (8.0) specifically tests that
/// the fold predicate path doesn't leak junk into main_accum when sigmoid(gate) ≈ 0.
const GATEDLORA_FIXTURE_C_SRC: &str = r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = ones([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=gated_lora, target=["Toy.w"], rank=2, alpha=2)
let m = Toy()
m.to(cuda)
let x = ones([4, 8]).to(cuda)
let y_target = zeros([4, 8]).to(cuda)
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
m.lora_A_Toy_w__gatedlora = ones([8, 2]).to(cuda)
m.lora_B_Toy_w__gatedlora = ones([2, 8]).to(cuda)
m.gate_Toy_w__gatedlora = full([8], -30.0).to(cuda)
let y = m.forward(x)
print(y)
"#;

#[cfg(feature = "cuda")]
#[test]
fn gatedlora_fixture_c_negative_saturation() {
    run_gatedlora_fixture(GATEDLORA_FIXTURE_C_SRC, 8.0, 1e-4, "fixture_c");
}
```

- [ ] **Step 2: Run**

```bash
cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence --features cuda gatedlora_fixture_c -- --test-threads=1 --nocapture 2>&1 | tail -10
```

Expected: `1 passed`, `max_diff < 1e-4`, `y = 8.0` (base x@W only).

Debug guide:
- `max_diff > 0.01`: adapter path leaked contribution — sigmoid(-30) didn't saturate to 0, or the predicate path is wrong.

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs
git commit -m "test(wrga): Fixture C negative saturation (gate=-30, y=8.0 at 1e-4)

Proves sigmoid negative-saturation reaches 0.0; the adapter path
contribution zeroes out, leaving only the base x@W=8.  This is the
diagnostic fixture for 'fold predicate doesn't leak junk into main_accum
when sigmoid(gate) ≈ 0'."
```

### Task 5.4: Fixture D — mid-range (gate=1.0, y≈19.697 at 1e-3, REQUIRED INLINE COMMENT)

- [ ] **Step 1: Append Fixture D with the full inline ULP comment**

```rust
/// Fixture D — mid-range sigmoid curve validation.
/// alpha=2, rank=2, scale=1.0.  gate = full([8], 1.0f) exact f32.
/// sigmoid(1.0) ≈ 0.7310586.  y[i,j] = 8 + 0.7310586 * 16 * 1.0 ≈ 19.697 per element.
///
/// Tolerance 1e-3 (vs 1e-4 on A/B/C): gate=1.0f exercises the actual
/// ex2.approx + rcp.approx sigmoid curve shape between saturation points,
/// not exact values.
///
/// Error propagation at rank=2 accumulator magnitudes:
///   ex2.approx worst-case ULP:      ~2 ULP (~5e-8 at input -1.4427)
///   rcp.approx worst-case ULP:      ~1 ULP (~6e-8 at input ≈1.368)
///   Combined via 1/(1+e) derivative: ~9e-8 on sigmoid output
///   Through fold at rank=2:          ~5e-6 per element
///   Main-path MMA noise floor:       ~2e-4 (dominates)
///   1e-3 tolerance:                  ~5x over main MMA, ~200x over sigmoid
///
/// Bug class this fixture catches: a sign-based-stub kernel emitting
/// `select(x > 0, 1.0, select(x < 0, 0.0, 0.5))` passes every saturation
/// fixture (A/B/C) but returns 0.5 here — wrong by |0.5 - 0.7310586| ≈ 0.2311
/// on the sigmoid value.
///
///   Correct y:   (x@W = 8) + 0.7310586 · 16 · 1.0 ≈ 19.697
///   Stub   y:   (x@W = 8) + 0.5       · 16 · 1.0 = 16.0
///                                                   ^^^^
///   The (x@W = 8) base contribution is identical in both paths and cancels.
///   y-level error = |19.697 − 16.0| = 3.697, driven entirely by the
///   adapter-path factor 16 · (0.7311 − 0.5) ≈ 3.697. This is ~3700× the
///   1e-3 tolerance — fixture fails loudly.
///
/// This is the ONLY fixture in the B.3.1 matrix that discriminates
/// "emitted rcp.approx(1 + ex2.approx(-x * log2e))" from "silently stubbed."
const GATEDLORA_FIXTURE_D_SRC: &str = r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = ones([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=gated_lora, target=["Toy.w"], rank=2, alpha=2)
let m = Toy()
m.to(cuda)
let x = ones([4, 8]).to(cuda)
let y_target = zeros([4, 8]).to(cuda)
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
m.lora_A_Toy_w__gatedlora = ones([8, 2]).to(cuda)
m.lora_B_Toy_w__gatedlora = ones([2, 8]).to(cuda)
m.gate_Toy_w__gatedlora = full([8], 1.0).to(cuda)
let y = m.forward(x)
print(y)
"#;

#[cfg(feature = "cuda")]
#[test]
fn gatedlora_fixture_d_mid_range_curve() {
    // Expected computed from CPU reference: 8 + sigmoid(1.0) * 16.
    let expected = 8.0_f32 + sigmoid_f64(1.0) as f32 * 16.0;
    run_gatedlora_fixture(GATEDLORA_FIXTURE_D_SRC, expected, 1e-3, "fixture_d");
}

fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
```

- [ ] **Step 2: Run**

```bash
cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence --features cuda gatedlora_fixture_d -- --test-threads=1 --nocapture 2>&1 | tail -10
```

Expected: `1 passed`, `max_diff < 1e-3`, `y ≈ 19.697`.

Debug guide (per fixture comment):
- `y = 16.0` exactly: sign-based-stub bug — sigmoid emits 0.5 at mid-range. Check `emit_sigmoid_approx_fused` — likely a wrong constant.

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs
git commit -m "test(wrga): Fixture D mid-range curve (gate=1.0, y≈19.697 at 1e-3)

The ONLY fixture in B.3.1 that discriminates real rcp.approx(1 +
ex2.approx(-x*log2e)) from a silently-stubbed sigmoid.  Saturation
fixtures (A/B/C) test exact endpoint values and would pass against a
sign-based stub; this fixture catches the stub's 0.5 midpoint response
(wrong by 3.7 at y-level, ~3700x the 1e-3 tolerance).

Inline comment documents the ULP derivation and bug-class attack
vector in full per spec §3.1."
```

---

## Task Group 6 — Commit 6: Close-out

### Task 6.1: Update memory file invariants #13-#16

**Files (outside repo):**
- Modify: `C:\Users\bwiem\.claude\projects\c--Users-bwiem-projects-NSL\memory\project_wrga_fused_ptx_rewrite.md`

- [ ] **Step 1: Audit current last invariant number**

Read the file and grep for `^- \*\*#\d+` (or whatever the invariant numbering format is). Record the highest number currently present. Spec claims #12 is the last post-2026-04-16 close-out entry; verify.

- [ ] **Step 2: Append invariants #13-#16**

Append at the end of the "Load-bearing invariants" section (or equivalent):

```markdown
### Invariants added by B.3.1 (2026-04-18)

> **#13 — Gate load is per-thread, not warp-broadcast.** The m16n8k16 D-fragment distributes output columns so thread t owns cols `(tid%4)*2` and `+1`. `emit_gate_load_per_thread` MUST load exactly these 2 values per thread, not all 8. Warp-broadcast would 4× HBM gate traffic for no benefit since each thread consumes only the 2 values its D-fragment owns.

> **#14 — Gate-load `LastKIter` scheduling hides HBM latency.** The fused-body emitter schedules `emit_gate_load_per_thread + emit_sigmoid_approx_fused` at the end of the final K-iteration (before its `bar.sync`) so the sigmoid computation overlaps the final main MMA's latency. B.3.1 ships with `GateLoadPhase::LastKIter` unconditionally; all shipped ptxas configs have `k_iters ≥ 1` for the epilogue, so the LastKIter emission path covers the full test matrix. The `PostLoop` variant is reserved in the enum for a potential future milestone but has no emission site in B.3.1.

> **#15 — `FoldResultMask` is load-bearing for partial tiles.** OOB gate reads are NOT predicated at load time. The OOB column's fold contribution to `main_accum` is predicated via `@%p_colN`. The existing `%n_real` store-mask in `emit_lora_store_output` is the second bound check. Both masks must remain. The `(16, 13, 16, 4)` ptxas config is the only one that exercises the runtime predicate path — any refactor that removes that config or changes its n value to a multiple of 8 loses runtime coverage of this invariant.

> **#16 — PTX sigmoid immediate constants pinned.** `0fBFB8AA3B = -log₂(e)` (NOT `0f3FB8AA3B` which is +log₂(e) and silently wrong). `0f3F800000 = 1.0`. Structural test in Task 2.1 pins these constants explicitly and rejects the historical bad value `0f3FBE2FB9` as belt-and-suspenders.
```

If Step 1 audit found the highest number is something other than 12, renumber all four invariants accordingly (start at highest+1).

- [ ] **Step 3: (No git commit — memory files are outside the repo)**

Memory updates take effect immediately on save.

### Task 6.2: WRGA paper Appendix B.2 update

**Files (outside repo or in docs/research/):**
- Modify: WRGA paper (path confirmed at this task's prep)

- [ ] **Step 1: Locate the WRGA paper**

```bash
find /c/Users/bwiem/projects/NSL/docs -iname "*wrga*" 2>/dev/null
find /c/Users/bwiem/projects/NSL/docs -name "*.md" | xargs grep -l "WRGA\|Weight Refinement" 2>/dev/null | head -5
```

The paper may be a PDF (source elsewhere), a `.md` (direct edit), or under `docs/research/`. If uncertain, ask the user for the paper path before proceeding.

- [ ] **Step 2: Append Appendix B.2 entry**

Append to the Appendix B section (institutional-memory retrospectives):

```markdown
### B.2 Approximation-based activation helper testing rule

**Discovered during WRGA B.3.1 tolerance design (2026-04-18); generalizes beyond sigmoid and beyond WRGA.**

Any approximation-based activation helper (sigmoid, tanh, GELU, softmax, and related) emitted as a dedicated PTX helper MUST have at least one mid-range integration fixture whose tolerance is set such that a sign-based-stub implementation (one that returns a small set of constants based on sign/zero of the input) would FAIL the fixture.

Pure-saturation fixtures (testing only inputs that produce exact-constant outputs like 0.0, 0.5, 1.0) are necessary but not sufficient; they are unable to distinguish "the approximation is emitted correctly" from "the helper is silently stubbed."

**Concrete B.3.1 instantiation:** the GatedLoRA `gate=1.0f` mid-range fixture at 1e-3 tolerance. A sign-based-stub sigmoid would return 0.5 at gate=1.0, producing y-level error of ~3.7 (>3700× the tolerance) — fixture fails loudly. The 1e-3 tolerance was chosen based on ULP derivation through the fold step: sigmoid ULP ~9e-8, propagates to ~5e-6 per element, main-path MMA noise floor ~2e-4 dominates, 1e-3 provides ~5× headroom.

**Generalization:** the same rule applies to any future PTX helper that implements an approximation-based transcendental (exp, log, tanh, erf, GELU). Each such helper needs a mid-range fixture designed so the specific sign-based-stub attack vector for THAT function is above the chosen tolerance. For sigmoid this is gate=1.0; for tanh it would be input=0.5 (tanh(0.5) ≈ 0.4621, stub returns 0); for GELU it would be input=1.0 (GELU(1.0) ≈ 0.8413, stub returns 1.0 or input).
```

- [ ] **Step 3: Commit the paper update**

```bash
git add <wrga paper path>
git commit -m "docs(wrga): Appendix B.2 — approximation-based activation helper testing rule

Institutional lesson from B.3.1 tolerance design: mid-range integration
fixtures are necessary to catch sign-based-stub implementations of
approximation-based activation helpers.  Saturation fixtures are
necessary but insufficient."
```

If the WRGA paper is a PDF with source elsewhere or a markdown source that's gitignored, adapt the commit accordingly (`git add -f` or update source file out-of-band).

### Task 6.3: B.3.2 deferred-trigger stub

**Files:**
- Create: `docs/plans/2026-04-18-wrga-b32-fused-backward-STUB.md`

- [ ] **Step 1: Write the stub per spec §3.4**

Create the file with the exact content from spec §3.4:

```markdown
# WRGA B.3.2 — Fused GatedLoRA Backward (DEFERRED)

**Status:** deferred pending measurement trigger. Do NOT schedule this
milestone without satisfying the trigger below.

## Measurement-gated promotion trigger

Promote B.3.2 from deferred to scheduled **only when** a GatedLoRA
training benchmark at seq=2048, rank=16 shows backward time materially
larger than forward. Specifically: if `backward_time > 2.5 × forward_time`
on a representative workload (Llama-3-8B-style, batch × seq = 65k tokens
per microbatch, sm_80 or sm_90), schedule B.3.2.

If backward stays within 1.5× to 2× forward and remains matmul-bound
(profiler confirms ≥ 80% of backward time in matmul kernels rather than
elementwise ops), the unfused adapter-triple backward path is adequate;
**B.3.2 remains deferred indefinitely**. Elementwise sigmoid backward
(`σ(x)·(1−σ(x))` via the retained forward-tape value) is already cheap
under source-AD — there is no kernel-launch overhead to eliminate.

## Inherited institutional-memory rules

B.3.2, when scheduled, inherits the institutional lesson from the
2026-04-16 WRGA PTX scaffolding retrospective (see
`project_wrga_ptx_scaffolding_discovered.md`): **ptxas validation MUST
appear in B.3.2's first real-implementation commit, not deferred to a
later test layer.** The commit that introduces the backward PTX emitter
is the same commit that introduces its ptxas unit sweep. This rule is
non-negotiable; it is the WRGA retrospective's load-bearing institutional
correction. B.3.1 further promoted this to the WRGA paper's Appendix B.2
as a generalized rule covering all approximation-based PTX helpers.

## Four-layer test discipline inheritance

B.3.2 MUST ship with all four test layers green at milestone close:

1. **Skeleton snapshots** — any new helpers gated by per-variant pinned
   snapshots (or structural assertions for simple helpers; see B.3.1's
   convention).
2. **Unit ptxas** — backward kernel PTX validated across the same shape
   matrix as forward (6 reused LoRA shapes + any backward-distinctive
   configs), feeding each emission to `cuModuleLoadData` or `ptxas`.
3. **Integration numerical** — gradient correctness verified against
   finite-difference reference at 1e-3 tolerance (looser than forward's
   1e-4 because finite-difference step-size trades off against
   condition-number noise; set the step size and justify the tolerance
   inline at the fixture).
4. **E2E launch-counter** — `backward_fused_cuda_actually_fires` parallel
   to the forward hardening test, asserting `[nsl-gpu-launch-count] ≥ 1`
   during a full forward+backward pass.

## Scope boundary

B.3.2 covers ONLY fused backward for GatedLoRA. Fused backward for LoRA
and IA³ is a separate milestone (B.3.3 or similar) if demanded by the
same measurement trigger applied to those adapter types. Do not conflate.
```

- [ ] **Step 2: Force-add (docs/plans/ is gitignored)**

```bash
git add -f docs/plans/2026-04-18-wrga-b32-fused-backward-STUB.md
git commit -m "docs(wrga): B.3.2 fused GatedLoRA backward deferred-trigger stub

Records the measurement trigger (backward > 2.5x forward at seq=2048,
rank=16; if within 1.5-2x and matmul-bound, stays deferred indefinitely),
the inherited institutional-memory rule (ptxas from first real-impl
commit), and the four-layer test discipline inheritance.

Force-added; docs/plans/ gitignored per milestone precedent."
```

### Task 6.4: B.3.1 close-out note

**Files:**
- Create: `docs/plans/2026-04-18-wrga-b31-gatedlora-CLOSEOUT.md`

- [ ] **Step 1: Write the close-out**

```markdown
# WRGA B.3.1 Fused GatedLoRA Forward PTX — Milestone Close-Out

**Closed:** 2026-04-18
**Branch:** `feat/wrga-b31-gatedlora-fused-forward`
**Spec:** [2026-04-18-wrga-b31-gatedlora-fused-forward-design.md](2026-04-18-wrga-b31-gatedlora-fused-forward-design.md)
**Plan:** [2026-04-18-wrga-b31-gatedlora-fused-forward-plan.md](2026-04-18-wrga-b31-gatedlora-fused-forward-plan.md)

## Close-out criteria (§3.3 of spec) — state at close

| # | Criterion | State |
|---|---|---|
| 1 | All 6 logical commits merged | ✓ |
| 2 | 10 new ptxas tests green on CUDA (6 LoRA shapes × PerColumnSigmoid + 4 GatedLoRA-distinctive) | ✓ |
| 3 | 4 new integration fixtures pass at prescribed tolerances | ✓ (A/B/C at 1e-4, D at 1e-3) |
| 4 | All pre-existing LoRA/IA³/FA regression gates green | ✓ (11 LoRA+IA³ ptxas, 11 runtime_equivalence, fa_v2 baseline preserved) |
| 5 | Memory file invariants #13–#16 appended | ✓ |
| 6 | B.3.2 stub document written | ✓ |
| 7 | WRGA paper Appendix B.2 updated | ✓ |

## Institutional lesson promoted to WRGA paper Appendix B.2

> Any approximation-based activation helper (sigmoid, tanh, GELU, softmax, and related) emitted as a dedicated PTX helper MUST have at least one mid-range integration fixture whose tolerance is set such that a sign-based-stub implementation would FAIL the fixture. Pure-saturation fixtures are necessary but not sufficient.

## Deferred follow-ups

- **B.3.2** — Fused GatedLoRA backward; deferred with measurement trigger (see STUB doc).
- **B.4+** — sm_90 WGMMA, `ldmatrix`, `cp.async`, multi-warp-per-tile, perf benchmarking (same as B.3 milestone's deferred items).
- **Deep `kernel_skeleton` refactor** with fusion-callback pattern.
- **FA param-block migration** to `emit_param_block`.
```

- [ ] **Step 2: Force-add + commit**

```bash
git add -f docs/plans/2026-04-18-wrga-b31-gatedlora-CLOSEOUT.md
git commit -m "docs(wrga): B.3.1 close-out note

All 7 spec §3.3 close-out criteria satisfied.  Institutional lesson
promoted to WRGA paper Appendix B.2.  B.3.2 stub written with
measurement-gated trigger.  Ready to merge."
```

### Task 6.5: MEMORY.md index update

**Files (outside repo):**
- Modify: `C:\Users\bwiem\.claude\projects\c--Users-bwiem-projects-NSL\memory\MEMORY.md`

- [ ] **Step 1: Append close-out entry**

Find the existing WRGA entries and append:

```markdown
## WRGA B.3.1 fused GatedLoRA forward PTX — CLOSED 2026-04-18
- See [project_wrga_fused_ptx_rewrite.md](project_wrga_fused_ptx_rewrite.md) (invariants #13-#16 appended); forward-only, B.3.2 backward deferred with measurement trigger.
```

Place alphabetically/chronologically with other WRGA entries.

- [ ] **Step 2: (No git commit — MEMORY.md is outside the repo)**

### Task 6.6: Final regression sweep

- [ ] **Step 1: Full-workspace build + test**

```bash
cd /c/Users/bwiem/projects/NSL/.worktrees/wrga-b31-gatedlora-fused-forward  # or the worktree name in use
cargo build -p nsl-codegen -p nsl-runtime -p nsl-cli --features cuda 2>&1 | tail -3
cargo test -p nsl-codegen --lib 2>&1 | tail -3
cargo test -p nsl-codegen --test fa_v2_snapshots 2>&1 | tail -3
cargo test -p nsl-codegen --test wrga_fused_ptx_ptxas --features cuda 2>&1 | tail -3
cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence --features cuda -- --test-threads=1 2>&1 | tail -5
```

Expected final-state counts:

| Layer | Target |
|---|---|
| nsl-codegen build | clean (pre-existing warnings OK) |
| nsl-codegen lib tests | all green |
| fa_v2_snapshots | 17 passed / 6 pre-existing failures unchanged |
| wrga_fused_ptx_ptxas | 21 passed / 0 failed (6 LoRA + 5 IA³ + 10 GatedLoRA) |
| wrga_adapter_runtime_equivalence | 15 passed / 0 failed / 0 ignored (11 LoRA+IA³ + 4 GatedLoRA) |

If any unexpected failure, report **BLOCKED** and investigate before tagging milestone as closed.

- [ ] **Step 2: Milestone close marker commit**

```bash
git commit --allow-empty -m "feat(wrga): B.3.1 milestone closed

All 7 spec §3.3 close-out criteria green:
- 21/0 wrga_fused_ptx_ptxas (6 LoRA + 5 IA³ + 10 GatedLoRA)
- 15/0/0 wrga_adapter_runtime_equivalence (11 LoRA+IA³ + 4 GatedLoRA)
- fa_v2 baseline preserved (17 passed / 6 pre-existing)
- Memory invariants #13-#16 appended
- B.3.2 stub + close-out + WRGA paper Appendix B.2 shipped

Ready to merge to main."
```

---

## Self-Review

**Spec coverage check (§1-§3 of spec):**

- §1 architecture + commit sequence: Task Groups 1-6 map directly. ✓
- §2 PTX-level specifics: Tasks 2.1-2.3 cover the three helpers (`emit_sigmoid_approx_fused`, `emit_gate_load_per_thread`, `emit_gatedlora_fold`); Tasks 1.3 + 4.1 cover the shared body + GatedLoRA synthesizer. Fold emission in `emit_fused_adapter_kernel_body` matches spec §2's fold-step code exactly. Register budget additions in the conditional `is_gated` block of Task 1.3 Step 3 match spec §2's list (10 new registers plus `%scale_reg` reused from LoRA). ✓
- §3.1 test matrix: Task 3.2 ships 10 ptxas tests (including `(16, 13, 16, 4)` partial-n at Task 3.2 Step 2); Tasks 5.1-5.4 ship 4 integration fixtures. ✓
- §3.2 commit acceptance: each task ends with a commit following the acceptance-checklist format; the pre-audits from Task 1.1 are explicit in the Task 1.4 commit message. ✓
- §3.3 close-out: Tasks 6.1-6.6 cover all 7 criteria. ✓
- §3.4 B.3.2 stub: Task 6.3 ships verbatim content from spec. ✓

**Placeholder scan:** No TBD/TODO/fill-in markers at the plan level. The `TODO(gate-dtype)` comment in Task 2.2 is a deliberate inline future-work marker, not a plan placeholder.

**Type consistency:** `FoldKind`, `GateLoadPhase`, `PartialTileMask`, `FusedGatedLoraConfig`, `emit_fused_adapter_kernel_body`, `emit_sigmoid_approx_fused`, `emit_gate_load_per_thread`, `emit_gatedlora_fold` all used consistently across Tasks 1-6. Helper signatures in Task 1.3's code body match the signatures declared in Tasks 2.1-2.3.

**Register-name discipline:** `%gate0/%gate1/%sig_tmp/%fold_tmp/%rd_gate/%rd_gate_addr/%r_gate/%r_col1/%p_col0/%p_col1` used consistently across Task 1.3 (register pool), Tasks 2.1-2.3 (helper emissions), and Task 4.1 (synthesizer caller). If Task 1.1 Step 1's collision audit reveals matches, the plan's `%gl_` prefix convention propagates uniformly.

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-04-18-wrga-b31-gatedlora-fused-forward-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task (Task 1.1 through Task 6.6), review between tasks, fast iteration. Good for this plan because each task is self-contained and TDD-gated; PTX debugging in particular benefits from isolated subagent context windows.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`, batch with checkpoints. Faster wall-clock if no mid-plan surprises, but with ~20 tasks and 1 load-bearing ptxas emission task (Task 4.1), the context can get crowded.

Which approach?
