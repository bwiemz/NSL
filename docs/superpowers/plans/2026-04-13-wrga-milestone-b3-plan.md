# WRGA Milestone B.3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an epilogue-fused MMA PTX kernel for LoRA and IA³ adapter sites on sm_80+ that matches the WRGA paper's "+0 memory ops" claim. B.2.1's unfused three-FFI path becomes the sm_75 fallback and the explicit opt-out. GatedLoRA defers to B.3.1.

**Architecture:** Six tasks. Task 1 extracts reusable MMA primitives from `flash_attention.rs` into `matmul_mma.rs` (copy-then-parameterize — FA stays untouched). Task 2 builds `synthesize_fused_lora_ptx` / `synthesize_fused_ia3_ptx` with the CRITICAL **interleaved epilogue** — `x @ A` accumulates during each main K-iteration because post-loop access only has the last K-tile's x fragment. Scale is a `.param .f32`, not a PTX literal, so kernel dedup by `(m, n, k, rank, target_sm)` survives different alpha values. Task 3 adds a verification assertion in the fusion pass (the conditional rewrite in Task 4 means the intermediate VarId never enters the Wengert list for fused sites). Task 4 adds two new clean-signature FFIs (`nsl_adapter_fused_lora_matmul`, `nsl_adapter_fused_ia3_matmul`) and modifies B.2.1's `synthesize_lora_adapted` / `synthesize_ia3_adapted` to emit a single FFI call on sm_80+ when the FusionDecision says EpilogueFusedLora, else fall through to B.2.1's triple. Task 5 adds `build_4_fused` at **1e-4 tolerance** (1e-3 is a bug signal) plus a kernel-launch-count assertion. Task 6 is close-out.

**Tech Stack:** Rust 2021, Cranelift IR, PTX assembly targeting sm_80+, `cargo test -p nsl-codegen` / `-p nsl-cli` / `-p nsl-runtime`. Reference paper: `git show a61db4a:NSL-WRGA-Research.PDF` (file later deleted from repo; commit `a61db4a` is canonical).

---

## Pre-flight

- [ ] **Confirm baseline on main includes B.2.1 + WGGO + CSHA merges.**
  Run: `git log -1 --oneline`
  Expected: commit `64e9e10 docs(wrga): B.3 spec — interleaved epilogue ...` or later.

- [ ] **Record baseline test counts** (these are the floor for every later task):
  ```bash
  cargo test -p nsl-semantic 2>&1 | grep "test result" | tail -3
  cargo test -p nsl-codegen --lib 2>&1 | grep "test result" | tail -3
  cargo test -p nsl-codegen --tests 2>&1 | grep "test result" | tail -3
  cargo test -p nsl-codegen flash_attention 2>&1 | grep "test result" | tail -3
  cargo test -p nsl-cli --test e2e -- --test-threads=1 2>&1 | grep "test result" | tail -3
  cargo test -p nsl-cli --test wrga_report_cli 2>&1 | grep "test result" | tail -3
  cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence 2>&1 | grep "test result" | tail -3
  ```
  Floors: semantic ≥ 260, codegen lib ≥ 1113, nsl-cli e2e = 80, wrga_report_cli = 6, wrga_adapter_runtime_equivalence = 4.

- [ ] **Create the B.3 worktree.**
  ```bash
  cd c:/Users/bwiem/projects/NSL
  git worktree add ../NSL-wrga-b3 -b feat/wrga-milestone-b3
  cd ../NSL-wrga-b3
  cargo build --features cuda 2>&1 | tail -3
  ```
  Expected: clean build. All subsequent work happens in `c:/Users/bwiem/projects/NSL-wrga-b3`.

- [ ] **Read the reference paper's fusion section** so you know what "+0 memory ops" means:
  ```bash
  git show a61db4a:NSL-WRGA-Research.PDF > c:/tmp/wrga-paper.pdf
  # Open in any PDF viewer, find "Innovation 4: Fusion-Integrated Adapters"
  ```

---

## File Structure

**Create:**
- `crates/nsl-codegen/src/matmul_mma.rs` — reusable MMA primitives for arbitrary matmul (Task 1).
- `crates/nsl-codegen/src/wrga_fused_ptx.rs` — LoRA/IA³ epilogue PTX generator (Task 2).
- `crates/nsl-runtime/src/fused_adapter.rs` — two new FFIs that launch the synthesized PTX (Task 4).

**Modify:**
- `crates/nsl-codegen/src/lib.rs` — `pub mod` declarations for the two new nsl-codegen modules (Task 1 + Task 2).
- `crates/nsl-codegen/src/wrga_adapter_inject.rs` — extend `AdapterSite` with `fusion_decision: Option<FusionDecision>` + `rank > 16` hard error (Task 4 + Task 2).
- `crates/nsl-codegen/src/wrga_adapter_rewrite.rs` — conditional emission in `synthesize_lora_adapted` / `synthesize_ia3_adapted` (Task 4).
- `crates/nsl-codegen/src/wrga_fusion.rs` — verification assertion after fusion-plan pass (Task 3).
- `crates/nsl-codegen/src/compiler/entry_points.rs` — PTX kernel registration + embedding + dedup (Task 4).
- `crates/nsl-codegen/src/compiler/mod.rs` — new `fused_ptx_kernels: HashMap<KernelKey, String>` field on `Compiler` (Task 4).
- `crates/nsl-runtime/src/lib.rs` — `pub mod fused_adapter;` (Task 4).
- `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs` — `build_4_fused` + `build_5_kernel_count` (Task 5).
- `crates/nsl-codegen/tests/wrga_adapter_runtime.rs` — `fused_lora_site_leaves_no_intermediate_activation` (Task 3).

---

## Task 1: Standalone MMA matmul infrastructure

**Files:**
- Create: `crates/nsl-codegen/src/matmul_mma.rs`
- Modify: `crates/nsl-codegen/src/lib.rs` (add `pub mod matmul_mma;`)

**Goal:** Extract reusable m16n8k16 f16×f16→f32 MMA primitives from `flash_attention.rs:1323-1422` into a new module with parameterized signatures. FA stays untouched.

**Recon inputs (verified):**
- `flash_attention.rs:1323` — `emit_load_a_fragment_smem(ptx, frag_regs: &[String; 4], smem_base_expr: &str, row_stride: usize)`.
- `flash_attention.rs:1361` — `emit_load_b_fragment_smem(ptx, frag_regs: &[String; 2], smem_base_expr: &str, row_stride: usize)`.
- `flash_attention.rs:1387` — `emit_mma_instruction(ptx, d_regs: &[String; 4], a_regs: &[String; 4], b_regs: &[String; 2], c_regs: &[String; 4])`.
- These emit `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` and the fragment loads.

- [ ] **Step 1: Write the failing golden-output tests**

Create `crates/nsl-codegen/src/matmul_mma.rs`:

```rust
//! WRGA B.3 Task 1: reusable MMA primitives for arbitrary matmul.
//!
//! Copied-then-parameterized from flash_attention.rs:1323-1422.  FA itself
//! stays untouched; a later migration could unify the two paths.

/// MMA shape — B.3 ships only m16n8k16 (Ampere f16 tensor cores).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MmaShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

pub const MMA_M16N8K16: MmaShape = MmaShape { m: 16, n: 8, k: 16 };

/// Fragment layout.  m16n8k16 always uses row-A × col-B in PTX terms,
/// but the caller can remap via SMEM stride if a different source
/// layout is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragmentLayout { Row, Col }

/// Accumulator dtype for MMA.  B.3 uses only F32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccDtype { F32 }

/// Emit PTX for a single m16n8k16 mma.sync.
///
/// Produces exactly one line of the form
/// `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {d...}, {a...}, {b...}, {c...};`.
///
/// Register name conventions match the FA helpers (register names are
/// passed by the caller as strings — this fn doesn't own register
/// allocation).
pub fn emit_mma_instruction(
    ptx: &mut String,
    d_regs: &[String; 4],
    a_regs: &[String; 4],
    b_regs: &[String; 2],
    c_regs: &[String; 4],
) {
    ptx.push_str("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n");
    ptx.push_str(&format!(
        "        {{{}, {}, {}, {}}},\n",
        d_regs[0], d_regs[1], d_regs[2], d_regs[3]
    ));
    ptx.push_str(&format!(
        "        {{{}, {}, {}, {}}},\n",
        a_regs[0], a_regs[1], a_regs[2], a_regs[3]
    ));
    ptx.push_str(&format!("        {{{}, {}}},\n", b_regs[0], b_regs[1]));
    ptx.push_str(&format!(
        "        {{{}, {}, {}, {}}};\n",
        c_regs[0], c_regs[1], c_regs[2], c_regs[3]
    ));
}

/// Emit PTX to load an m16n8k16 A-fragment (row-major f16 x16) from SMEM.
/// Each thread holds 4 .b32 registers covering 8 pairs of f16 values.
pub fn emit_load_a_fragment_smem(
    ptx: &mut String,
    frag_regs: &[String; 4],
    smem_base_expr: &str,
    row_stride_bytes: usize,
) {
    ptx.push_str("    // Load A-fragment (m16xk16 row-major) from shared memory\n");
    for (reg_idx, k_base_pair) in [(0, 0usize), (1, 4), (2, 8), (3, 12)].iter() {
        let byte_col_offset = k_base_pair * 2;
        ptx.push_str(&format!(
            "    mad.lo.u32 %mma_addr, %mma_a_row, {}, {};  // row * stride + base\n",
            row_stride_bytes, smem_base_expr
        ));
        ptx.push_str(&format!(
            "    add.u32 %mma_addr, %mma_addr, {};  // + k byte offset\n",
            byte_col_offset
        ));
        ptx.push_str(&format!(
            "    ld.shared.b32 %{}, [%mma_addr];\n",
            frag_regs[*reg_idx]
        ));
    }
}

/// Emit PTX to load an m16n8k16 B-fragment (col-major f16 x8) from SMEM.
/// Each thread holds 2 .b32 registers covering 4 pairs of f16 values.
pub fn emit_load_b_fragment_smem(
    ptx: &mut String,
    frag_regs: &[String; 2],
    smem_base_expr: &str,
    row_stride_bytes: usize,
) {
    ptx.push_str("    // Load B-fragment (k16xn8 col-major) from shared memory\n");
    for (reg_idx, k_base_pair) in [(0, 0usize), (1, 8)].iter() {
        let byte_col_offset = k_base_pair * 2;
        ptx.push_str(&format!(
            "    mad.lo.u32 %mma_addr, %mma_b_row, {}, {};  // row * stride + base\n",
            row_stride_bytes, smem_base_expr
        ));
        ptx.push_str(&format!(
            "    add.u32 %mma_addr, %mma_addr, {};  // + k byte offset\n",
            byte_col_offset
        ));
        ptx.push_str(&format!(
            "    ld.shared.b32 %{}, [%mma_addr];\n",
            frag_regs[*reg_idx]
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(xs: &[&str; 4]) -> [String; 4] {
        [xs[0].into(), xs[1].into(), xs[2].into(), xs[3].into()]
    }
    fn s2(xs: &[&str; 2]) -> [String; 2] { [xs[0].into(), xs[1].into()] }

    #[test]
    fn emit_mma_instruction_produces_expected_shape() {
        let mut ptx = String::new();
        let d = s(&["%d0", "%d1", "%d2", "%d3"]);
        let a = s(&["%a0", "%a1", "%a2", "%a3"]);
        let b = s2(&["%b0", "%b1"]);
        let c = s(&["%c0", "%c1", "%c2", "%c3"]);
        emit_mma_instruction(&mut ptx, &d, &a, &b, &c);
        assert!(
            ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"),
            "expected MMA shape header, got: {ptx}",
        );
        assert!(ptx.contains("{%d0, %d1, %d2, %d3}"));
        assert!(ptx.contains("{%a0, %a1, %a2, %a3}"));
        assert!(ptx.contains("{%b0, %b1}"));
        assert!(ptx.contains("{%c0, %c1, %c2, %c3}"));
    }

    #[test]
    fn emit_load_a_fragment_emits_four_ld_shared() {
        let mut ptx = String::new();
        let regs = s(&["ra0", "ra1", "ra2", "ra3"]);
        emit_load_a_fragment_smem(&mut ptx, &regs, "%smem_base_x", 16);
        assert_eq!(ptx.matches("ld.shared.b32").count(), 4,
            "should emit 4 ld.shared for A-fragment; got: {ptx}");
        assert!(ptx.contains("%ra0"));
        assert!(ptx.contains("%ra3"));
    }

    #[test]
    fn emit_load_b_fragment_emits_two_ld_shared() {
        let mut ptx = String::new();
        let regs = s2(&["rb0", "rb1"]);
        emit_load_b_fragment_smem(&mut ptx, &regs, "%smem_base_w", 8);
        assert_eq!(ptx.matches("ld.shared.b32").count(), 2);
        assert!(ptx.contains("%rb0"));
        assert!(ptx.contains("%rb1"));
    }
}
```

- [ ] **Step 2: Add `pub mod matmul_mma;` to `crates/nsl-codegen/src/lib.rs`**

Find existing `pub mod wrga_adapter_inject;` / `pub mod wrga_adapter_rewrite;` and add adjacent:

```rust
pub mod matmul_mma;
```

- [ ] **Step 3: Run the tests — expect pass**

```bash
cargo test -p nsl-codegen matmul_mma 2>&1 | tail -10
```

Expected: `test result: ok. 3 passed`.

- [ ] **Step 4: Regression**

FA backward MMA tests + flash_attention snapshot tests must remain green — Task 1 copied code, didn't edit flash_attention.rs:

```bash
cargo test -p nsl-codegen flash_attention 2>&1 | tail -5
cargo build --features cuda 2>&1 | tail -3
```

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/matmul_mma.rs crates/nsl-codegen/src/lib.rs
git commit -m "feat(wrga): matmul_mma — reusable m16n8k16 MMA primitives for B.3"
```

---

## Task 2: Adapter-epilogue PTX generator

**Files:**
- Create: `crates/nsl-codegen/src/wrga_fused_ptx.rs`
- Modify: `crates/nsl-codegen/src/lib.rs` (add `pub mod wrga_fused_ptx;`)
- Modify: `crates/nsl-codegen/src/wrga_adapter_inject.rs` (rank-ceiling hard error)

**Goal:** Synthesize full PTX kernels for LoRA (interleaved-epilogue) and IA³ (broadcast-mul epilogue) fusion. Scale is a `.param .f32`, not a PTX literal. Rank > 16 is a hard compile error.

- [ ] **Step 1: Write the failing invariant tests**

Create `crates/nsl-codegen/src/wrga_fused_ptx.rs`:

```rust
//! WRGA B.3 Task 2: epilogue-fused LoRA/IA³ MMA PTX generator.
//!
//! LoRA kernel structure (CRITICAL — interleaved epilogue):
//! The naive "main loop then epilogue" is WRONG because SMEM is
//! overwritten each K-iteration; post-loop access would only see
//! the last x tile.  Instead: each main K-iteration also performs
//! x_tile @ A_tile accumulating into an `epilogue_intermediate`
//! register, so `epilogue_intermediate == x @ A` after the K-loop.
//! The final (x@A) @ B * scale is then folded into main_accum
//! before storing to y.
//!
//! IA³ is simpler: no interleaving, just a post-loop γ-broadcast-mul.
//!
//! Scale is a `.param .f32` — see the `scale` parameter below.  This
//! enables kernel dedup across sites with different alpha values.

use crate::matmul_mma;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FusedLoraConfig {
    pub site_id: String,
    pub m: u32,                // batch
    pub n: u32,                // d_out
    pub k: u32,                // k_in (shared dim of x@W)
    pub rank: u32,             // ≤ 16
    pub target_sm: u32,        // 80, 86, ...
    // scale is intentionally NOT a field — passed at launch time as
    // .param .f32.  See dedup notes in B.3 spec Risk #5.
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FusedIa3Config {
    pub site_id: String,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub target_sm: u32,
}

/// Kernel cache key for dedup.  Sites with matching key share one PTX.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LoraKernelKey {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub rank: u32,
    pub target_sm: u32,
}

impl FusedLoraConfig {
    pub fn kernel_key(&self) -> LoraKernelKey {
        LoraKernelKey {
            m: self.m,
            n: self.n,
            k: self.k,
            rank: self.rank,
            target_sm: self.target_sm,
        }
    }
}

/// Synthesize the full PTX for an epilogue-fused LoRA matmul kernel.
///
/// **Rank ceiling:** `config.rank <= 16` (single-pass epilogue).  Caller
/// must validate before invoking; this fn panics on violation (caller
/// error — decorator inject pass enforces a proper compile error).
pub fn synthesize_fused_lora_ptx(config: &FusedLoraConfig) -> String {
    assert!(
        config.rank <= 16,
        "B.3 rank ceiling: {} > 16; multi-pass epilogue is a follow-up milestone",
        config.rank,
    );
    assert!(config.target_sm >= 80, "B.3 requires sm_80+");

    let mut ptx = String::new();

    // Header.
    ptx.push_str(".version 7.0\n");
    ptx.push_str(&format!(".target sm_{}\n", config.target_sm));
    ptx.push_str(".address_size 64\n\n");

    ptx.push_str(&format!(
        ".visible .entry nsl_wrga_fused_lora_m{}n{}k{}r{}(\n",
        config.m, config.n, config.k, config.rank,
    ));
    ptx.push_str("    .param .u64 x_ptr,\n");
    ptx.push_str("    .param .u64 w_ptr,\n");
    ptx.push_str("    .param .u64 a_ptr,\n");
    ptx.push_str("    .param .u64 b_ptr,\n");
    ptx.push_str("    .param .f32 scale,       // alpha/rank — launch-time value for dedup\n");
    ptx.push_str("    .param .u64 y_ptr\n");
    ptx.push_str(")\n{\n");

    // Register decls — see spec Register budget analysis §.
    ptx.push_str("    // === Register declarations ===\n");
    ptx.push_str("    .reg .f32 %main_accum<8>;        // x @ W f32 accumulator\n");
    ptx.push_str("    .reg .b32 %main_a_frag<4>;       // current x tile fragment\n");
    ptx.push_str("    .reg .b32 %main_b_frag<2>;       // current W tile fragment\n");
    ptx.push_str("    .reg .b32 %epi_a_frag<4>;        // A-matrix tile (NOT x@A!)\n");
    ptx.push_str("    .reg .f32 %epi_interm<4>;        // incremental x @ A accumulator\n");
    ptx.push_str("    .reg .b32 %epi_b_frag<2>;        // B-matrix tile\n");
    ptx.push_str("    .reg .f32 %epi_final<8>;         // (x@A) @ B accumulator\n");
    ptx.push_str("    .reg .u64 %smem_base_x, %smem_base_w, %smem_base_a, %smem_base_b;\n");
    ptx.push_str("    .reg .u32 %k_idx, %mma_addr, %mma_a_row, %mma_b_row;\n");
    ptx.push_str("    .reg .pred %k_pred;\n");
    ptx.push_str("    .reg .f32 %scale_reg;\n\n");

    // Load scale once before the K-loop.
    ptx.push_str("    // Load scale (alpha/rank) from parameter — dedup-friendly\n");
    ptx.push_str("    ld.param.f32 %scale_reg, [scale];\n\n");

    // Init main_accum + epi_interm to zero.
    ptx.push_str("    // Init accumulators\n");
    for i in 0..8 { ptx.push_str(&format!("    mov.f32 %main_accum{}, 0f00000000;\n", i)); }
    for i in 0..4 { ptx.push_str(&format!("    mov.f32 %epi_interm{}, 0f00000000;\n", i)); }
    ptx.push_str("\n");

    // Interleaved K-tile loop: for each K-tile, (a) main MMA, (b) epilogue MMA into epi_interm.
    let k_iters = config.k / MMA_K_U32;  // number of K-tile iterations
    ptx.push_str(&format!(
        "    // === Interleaved main K-loop: {} tiles ===\n", k_iters,
    ));
    for k_tile in 0..k_iters {
        ptx.push_str(&format!("    // --- K-tile {} ---\n", k_tile));
        // 4a. Load x tile + W tile into SMEM + fragments (caller provides SMEM base exprs).
        ptx.push_str(&format!("    // Load x tile [k={}]\n", k_tile * MMA_K_U32));
        emit_load_frag_a_main(&mut ptx, k_tile);
        ptx.push_str(&format!("    // Load W tile [k={}]\n", k_tile * MMA_K_U32));
        emit_load_frag_b_main(&mut ptx, k_tile);
        // 4b. Main MMA: main_accum += x_tile @ W_tile
        emit_main_mma(&mut ptx);
        // 4c. Load A tile for this K-chunk.
        ptx.push_str(&format!("    // Load A tile [k={}]\n", k_tile * MMA_K_U32));
        emit_load_frag_a_epi(&mut ptx, k_tile);
        // 4d. Epilogue MMA: epi_interm += x_tile @ A_tile
        emit_epi_interm_mma(&mut ptx);
    }
    ptx.push_str("\n");

    // 5. Post-loop epilogue: load B, compute (x@A)@B, scale, add to main_accum.
    ptx.push_str("    // === Post-loop epilogue: (x@A) @ B * scale, fold into main_accum ===\n");
    ptx.push_str("    // Load B tile\n");
    emit_load_frag_b_epi(&mut ptx);
    // epi_final = epi_interm @ epi_b_frag
    emit_epi_final_mma(&mut ptx);
    // epi_final *= scale
    for i in 0..8 {
        ptx.push_str(&format!(
            "    mul.f32 %epi_final{}, %epi_final{}, %scale_reg;\n", i, i,
        ));
    }
    // main_accum += epi_final  (the "epilogue fusion" step)
    for i in 0..8 {
        ptx.push_str(&format!(
            "    add.f32 %main_accum{}, %main_accum{}, %epi_final{};\n", i, i, i,
        ));
    }
    ptx.push_str("\n");

    // 6. Store main_accum to y.
    ptx.push_str("    // === Store y ===\n");
    emit_store_y(&mut ptx);

    ptx.push_str("}\n");
    ptx
}

const MMA_K_U32: u32 = 16;  // m16n8k16

// These emit_* helpers are thin wrappers that delegate to matmul_mma for
// the actual MMA instructions, supplying register-name conventions unique
// to the fused kernel layout.  Details live in the implementation; the
// invariants the golden tests enforce are (1) number of mma.sync lines,
// (2) presence of scale as .param, (3) the "main_accum += epi_final" fold.

fn emit_load_frag_a_main(ptx: &mut String, k_tile: u32) {
    let regs = [
        format!("main_a_frag0"),
        format!("main_a_frag1"),
        format!("main_a_frag2"),
        format!("main_a_frag3"),
    ];
    matmul_mma::emit_load_a_fragment_smem(
        ptx, &regs, &format!("%smem_base_x + {}", k_tile * 32),
        32,
    );
}
fn emit_load_frag_b_main(ptx: &mut String, k_tile: u32) {
    let regs = [format!("main_b_frag0"), format!("main_b_frag1")];
    matmul_mma::emit_load_b_fragment_smem(
        ptx, &regs, &format!("%smem_base_w + {}", k_tile * 16),
        16,
    );
}
fn emit_load_frag_a_epi(ptx: &mut String, k_tile: u32) {
    let regs = [
        format!("epi_a_frag0"), format!("epi_a_frag1"),
        format!("epi_a_frag2"), format!("epi_a_frag3"),
    ];
    matmul_mma::emit_load_a_fragment_smem(
        ptx, &regs, &format!("%smem_base_a + {}", k_tile * 32),
        32,
    );
}
fn emit_load_frag_b_epi(ptx: &mut String) {
    let regs = [format!("epi_b_frag0"), format!("epi_b_frag1")];
    matmul_mma::emit_load_b_fragment_smem(
        ptx, &regs, "%smem_base_b", 16,
    );
}
fn emit_main_mma(ptx: &mut String) {
    let d = [
        "main_accum0".into(), "main_accum1".into(),
        "main_accum2".into(), "main_accum3".into(),
    ];
    let a = [
        "main_a_frag0".into(), "main_a_frag1".into(),
        "main_a_frag2".into(), "main_a_frag3".into(),
    ];
    let b = ["main_b_frag0".into(), "main_b_frag1".into()];
    let c = [
        "main_accum0".into(), "main_accum1".into(),
        "main_accum2".into(), "main_accum3".into(),
    ];
    matmul_mma::emit_mma_instruction(ptx, &d, &a, &b, &c);
}
fn emit_epi_interm_mma(ptx: &mut String) {
    let d = [
        "epi_interm0".into(), "epi_interm1".into(),
        "epi_interm2".into(), "epi_interm3".into(),
    ];
    let a = [
        "main_a_frag0".into(), "main_a_frag1".into(),
        "main_a_frag2".into(), "main_a_frag3".into(),
    ];
    let b = ["epi_a_frag0".into(), "epi_a_frag1".into()];
    let c = [
        "epi_interm0".into(), "epi_interm1".into(),
        "epi_interm2".into(), "epi_interm3".into(),
    ];
    matmul_mma::emit_mma_instruction(ptx, &d, &a, &b, &c);
}
fn emit_epi_final_mma(ptx: &mut String) {
    let d = [
        "epi_final0".into(), "epi_final1".into(),
        "epi_final2".into(), "epi_final3".into(),
    ];
    let a = [
        "epi_interm0".into(), "epi_interm1".into(),
        "epi_interm2".into(), "epi_interm3".into(),
    ];
    let b = ["epi_b_frag0".into(), "epi_b_frag1".into()];
    let c = [
        "epi_final0".into(), "epi_final1".into(),
        "epi_final2".into(), "epi_final3".into(),
    ];
    matmul_mma::emit_mma_instruction(ptx, &d, &a, &b, &c);
}
fn emit_store_y(ptx: &mut String) {
    ptx.push_str("    // (store sequence — per-thread writes of main_accum to global y)\n");
    for i in 0..8 {
        ptx.push_str(&format!(
            "    st.global.f32 [%y_ptr + {}], %main_accum{};\n", i * 4, i,
        ));
    }
}

/// Synthesize the full PTX for an epilogue-fused IA³ matmul kernel.
/// IA³'s epilogue is `y = (x @ W) * γ` — one broadcast-mul after the
/// main matmul.  No epilogue interleaving needed.
pub fn synthesize_fused_ia3_ptx(config: &FusedIa3Config) -> String {
    assert!(config.target_sm >= 80, "B.3 requires sm_80+");
    let mut ptx = String::new();
    ptx.push_str(".version 7.0\n");
    ptx.push_str(&format!(".target sm_{}\n", config.target_sm));
    ptx.push_str(".address_size 64\n\n");
    ptx.push_str(&format!(
        ".visible .entry nsl_wrga_fused_ia3_m{}n{}k{}(\n",
        config.m, config.n, config.k,
    ));
    ptx.push_str("    .param .u64 x_ptr,\n");
    ptx.push_str("    .param .u64 w_ptr,\n");
    ptx.push_str("    .param .u64 gamma_ptr,\n");
    ptx.push_str("    .param .u64 y_ptr\n");
    ptx.push_str(")\n{\n");
    ptx.push_str("    // === Register declarations (IA³ — simpler than LoRA) ===\n");
    ptx.push_str("    .reg .f32 %main_accum<8>;\n");
    ptx.push_str("    .reg .b32 %main_a_frag<4>;\n");
    ptx.push_str("    .reg .b32 %main_b_frag<2>;\n");
    ptx.push_str("    .reg .f32 %gamma<8>;\n");
    ptx.push_str("    .reg .u64 %smem_base_x, %smem_base_w;\n");
    ptx.push_str("    .reg .u32 %mma_addr, %mma_a_row, %mma_b_row;\n\n");
    for i in 0..8 { ptx.push_str(&format!("    mov.f32 %main_accum{}, 0f00000000;\n", i)); }
    // Main K-loop.
    let k_iters = config.k / MMA_K_U32;
    for k_tile in 0..k_iters {
        emit_load_frag_a_main(&mut ptx, k_tile);
        emit_load_frag_b_main(&mut ptx, k_tile);
        emit_main_mma(&mut ptx);
    }
    // Broadcast-mul by γ.
    ptx.push_str("    // === IA³ epilogue: main_accum *= gamma (broadcast) ===\n");
    for i in 0..8 {
        ptx.push_str(&format!(
            "    ld.global.f32 %gamma{}, [%gamma_ptr + {}];\n", i, i * 4,
        ));
        ptx.push_str(&format!(
            "    mul.f32 %main_accum{}, %main_accum{}, %gamma{};\n", i, i, i,
        ));
    }
    emit_store_y(&mut ptx);
    ptx.push_str("}\n");
    ptx
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_lora_config() -> FusedLoraConfig {
        FusedLoraConfig {
            site_id: "test.w".into(),
            m: 16, n: 8, k: 16, rank: 2,
            target_sm: 80,
        }
    }

    #[test]
    fn lora_ptx_uses_scale_as_param_not_literal() {
        let ptx = synthesize_fused_lora_ptx(&mk_lora_config());
        assert!(ptx.contains(".param .f32 scale"),
            "scale must be .param for kernel dedup; got PTX:\n{ptx}");
        assert!(ptx.contains("ld.param.f32 %scale_reg, [scale]"),
            "kernel must load scale from param at entry");
        // Sanity: no bare .f32 immediate passed to mul.f32 where scale is used.
        // The scale register is the only source for the epilogue mul.
        assert!(ptx.contains("mul.f32 %epi_final0, %epi_final0, %scale_reg"),
            "epilogue mul must use %scale_reg");
    }

    #[test]
    fn lora_ptx_emits_main_and_epilogue_mmas_per_k_tile() {
        // k=16 → 1 K-tile iteration; expected: 1 main MMA + 1 epi_interm MMA + 1 epi_final MMA = 3 total
        let ptx = synthesize_fused_lora_ptx(&mk_lora_config());
        let mma_count = ptx.matches("mma.sync.aligned.m16n8k16").count();
        assert_eq!(mma_count, 3,
            "LoRA k=16 expects 3 MMA (1 main + 1 epi_interm + 1 epi_final); got {mma_count}\nPTX:\n{ptx}");
    }

    #[test]
    fn lora_ptx_folds_epilogue_into_main_accum() {
        let ptx = synthesize_fused_lora_ptx(&mk_lora_config());
        // The "epilogue fusion" step: main_accum += epi_final.
        for i in 0..8 {
            let expected = format!("add.f32 %main_accum{}, %main_accum{}, %epi_final{}", i, i, i);
            assert!(ptx.contains(&expected),
                "missing fold step: {expected}\nPTX:\n{ptx}");
        }
    }

    #[test]
    fn lora_ptx_rank_above_16_panics() {
        let cfg = FusedLoraConfig {
            site_id: "test.w".into(),
            m: 16, n: 8, k: 16, rank: 17,
            target_sm: 80,
        };
        let res = std::panic::catch_unwind(|| synthesize_fused_lora_ptx(&cfg));
        assert!(res.is_err(), "rank > 16 must panic — caller must enforce beforehand");
    }

    #[test]
    fn ia3_ptx_emits_single_mma_and_gamma_broadcast() {
        let cfg = FusedIa3Config {
            site_id: "test.w".into(),
            m: 16, n: 8, k: 16, target_sm: 80,
        };
        let ptx = synthesize_fused_ia3_ptx(&cfg);
        assert_eq!(ptx.matches("mma.sync.aligned.m16n8k16").count(), 1,
            "IA³ must emit exactly 1 MMA (no epilogue matmul); got PTX:\n{ptx}");
        assert!(ptx.contains("mul.f32 %main_accum0, %main_accum0, %gamma0"),
            "IA³ must broadcast-mul main_accum by gamma");
        assert!(!ptx.contains(".param .f32 scale"),
            "IA³ has no scale parameter");
    }

    #[test]
    fn dedup_key_ignores_site_id() {
        let a = FusedLoraConfig {
            site_id: "blocks.0.wq".into(),
            m: 16, n: 8, k: 16, rank: 4, target_sm: 80,
        };
        let b = FusedLoraConfig {
            site_id: "blocks.1.wq".into(),
            m: 16, n: 8, k: 16, rank: 4, target_sm: 80,
        };
        assert_eq!(a.kernel_key(), b.kernel_key(),
            "sites with same dims+rank+sm must share kernel — key must exclude site_id");
    }
}
```

- [ ] **Step 2: Add `pub mod wrga_fused_ptx;` to `crates/nsl-codegen/src/lib.rs`**

```rust
pub mod wrga_fused_ptx;
```

- [ ] **Step 3: Add the rank-ceiling hard error in the inject pass**

In `crates/nsl-codegen/src/wrga_adapter_inject.rs`, in `run_with_compiler` (or the per-site loop), after dim resolution succeeds, gate rank > 16:

```rust
// B.3: single-pass epilogue requires rank ≤ 16.  Multi-pass is a
// follow-up milestone.
if site.rank > 16 && matches!(site.kind, crate::AdapterKind::Lora | crate::AdapterKind::GatedLora) {
    eprintln!(
        "[wrga] @adapter(target='{}'): rank={} > 16; B.3 single-pass epilogue \
         does not support rank > 16.  Use rank <= 16 or await multi-pass support.",
        site.target_param, site.rank,
    );
    // Leave site present with sentinel zero dims so downstream skips it
    // (mirrors the dim-resolution failure path from Task 1 of B.2.1).
    site.input_dim = 0;
    site.output_dim = 0;
}
```

- [ ] **Step 4: Run the tests — expect pass**

```bash
cargo test -p nsl-codegen wrga_fused_ptx 2>&1 | tail -15
```

Expected: all 6 tests in `wrga_fused_ptx::tests` pass.

- [ ] **Step 5: Full regression**

```bash
cargo test -p nsl-codegen --lib 2>&1 | tail -3
cargo test -p nsl-codegen --tests 2>&1 | tail -3
cargo build --features cuda 2>&1 | tail -3
```

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/wrga_fused_ptx.rs crates/nsl-codegen/src/lib.rs crates/nsl-codegen/src/wrga_adapter_inject.rs
git commit -m "feat(wrga): synthesize_fused_{lora,ia3}_ptx — interleaved epilogue, scale as .param"
```

---

## Task 3: Verify `activation_live` is clean for fused sites

**Files:**
- Modify: `crates/nsl-codegen/src/wrga_fusion.rs` (add assertion)
- Modify: `crates/nsl-codegen/tests/wrga_adapter_runtime.rs` (new test)

**Goal:** Because Task 4 makes B.2.1's AST rewrite conditional on `FusionDecision`, fused sites emit a single FFI call — the `x @ A` intermediate never enters the Wengert list. This task adds an invariant check that catches regressions where the rewrite accidentally emits the unfused triple for a fused site.

- [ ] **Step 1: Write the failing test**

Append to `crates/nsl-codegen/tests/wrga_adapter_runtime.rs`:

```rust
/// B.3 Task 3: fused LoRA sites must not leave an `x @ A` intermediate
/// in the Wengert list / memory plan.  Task 4's conditional rewrite
/// ensures this; Task 3 is the verification net.
#[test]
fn fused_lora_site_leaves_no_intermediate_activation() {
    const SRC: &str = r#"
@adapter(type=lora, target=["m.w"], rank=2, alpha=2)
let m = Toy()

model Toy:
    let w: Tensor<[8, 8], f32> = zeros([8, 8])

    fn forward(self, x: Tensor<[4, 8], f32>) -> Tensor<[4, 8], f32>:
        return x @ self.w

fn main():
    let x: Tensor<[4, 8], f32> = ones([4, 8])
    let _y = m.forward(x)
"#;
    let opts = nsl_codegen::CompileOptions {
        wrga_inputs: Some(nsl_codegen::WrgaInputs {
            adapter: vec![nsl_codegen::AdapterDecoratorConfig {
                kind: nsl_codegen::AdapterKind::Lora,
                targets: vec!["m.w".into()],
                rank: Some(2),
                alpha: Some(2),
            }],
            ..Default::default()
        }),
        source_ad: true,
        target: "cuda_sm80".into(),  // Force sm_80+ path — fused kernel enabled
        ..Default::default()
    };
    let plan = nsl_codegen::debug_compile_and_return_plan(SRC, &opts)
        .expect("compile must succeed")
        .expect("wrga::run must fire");

    // Find the fusion decision for this site.
    let fused_site_count = plan
        .fusion
        .decisions
        .iter()
        .filter(|d| matches!(d.target, nsl_codegen::wrga_fusion::FusionTarget::EpilogueFusedLora { .. }))
        .count();
    assert!(fused_site_count >= 1, "expected at least one EpilogueFusedLora decision; got decisions={:?}",
        plan.fusion.decisions);

    // Invariant: for each fused site, no memory assignment exists for its `x @ A` intermediate.
    // We detect the intermediate by a naming convention or by Wengert op type; since AdapterPlacement
    // doesn't expose a VarId for the intermediate, we assert the WEAKER property: the memory plan's
    // total assignment count for adapter-related vars is NOT larger than the unfused path would
    // produce — here, fused LoRA adds 0 new activations vs base (1 for x@W), unfused would add 2
    // (x@A and (x@A)@B).
    let adapter_related_assignments: Vec<_> = plan.memory.assignments.iter()
        .filter(|a| plan.prune.pruned.var_names.get(&a.var)
            .map(|n| n.contains("lora_") || n.contains("adapter"))
            .unwrap_or(false))
        .collect();
    // After fusion, these should not include a live `x @ A` intermediate.  A regression where
    // the unfused rewrite fires despite FusionDecision saying fused would leave 2 extra slots
    // with matmul-result liveness; fused should leave 0.
    assert!(
        adapter_related_assignments.len() < 2,
        "fused LoRA should not allocate x@A intermediate; got {} adapter-related assignments: {:?}",
        adapter_related_assignments.len(),
        adapter_related_assignments,
    );
}
```

- [ ] **Step 2: Run — expect failure**

```bash
cargo test -p nsl-codegen --test wrga_adapter_runtime fused_lora_site_leaves_no_intermediate 2>&1 | tail -15
```

Expected: test fails because (a) `FusionTarget` is not currently re-exported at the crate root, or (b) no fusion decisions are produced for adapter sites today, or (c) unfused path emits the triple and adds intermediates.

- [ ] **Step 3: Re-export `FusionTarget` at the crate root**

In `crates/nsl-codegen/src/lib.rs`, add:

```rust
pub use wrga_fusion::{FusionDecision, FusionPlan, FusionTarget};
```

- [ ] **Step 4: Add the verification assertion to `wrga_fusion.rs`**

In `crates/nsl-codegen/src/wrga_fusion.rs`, at the end of the pass that builds `FusionPlan`, add:

```rust
/// B.3 Task 3: verify the `activation_live` invariant for EpilogueFusedLora
/// sites.  Because the AST rewrite (B.2.1 Task 3, modified by B.3 Task 4)
/// emits a SINGLE FFI call for fused sites, the `x @ A` intermediate is
/// never in the AST → never in Wengert → never in activation_live.  If a
/// fused site's synthesized Wengert list somehow contains a matmul op
/// whose name matches `lora_A_<site>` as an input, that's a regression
/// and we panic with a clear message pointing at the rewrite conditional.
///
/// Called at the end of `build_fusion_plan` (or the top-level fusion pass
/// entry) once the plan is finalized.
pub fn verify_fused_sites_have_no_intermediate(
    plan: &FusionPlan,
    wengert: &crate::wengert::WengertList,
) {
    for decision in &plan.decisions {
        if !matches!(decision.target, FusionTarget::EpilogueFusedLora { .. }) {
            continue;
        }
        // Walk ops; look for the tell-tale Matmul whose RHS is a synth
        // lora_A_<site> MemberAccess.  Any such op for a FUSED site means
        // the rewrite regressed to unfused.
        for op in &wengert.ops {
            let op_name = wengert.var_names.get(&op.result).cloned().unwrap_or_default();
            if op_name.starts_with("matmul_") && op_name.contains(&format!("lora_A_{}", decision.site)) {
                panic!(
                    "[wrga B.3 Task 3 invariant] Fused site '{}' has an x @ A \
                     intermediate in the Wengert list ('{}').  This indicates \
                     the AST rewrite (synthesize_lora_adapted) emitted the \
                     unfused triple for a site whose FusionDecision is \
                     EpilogueFusedLora.  Check wrga_adapter_rewrite.rs's \
                     conditional on site.fusion_decision.",
                    decision.site, op_name,
                );
            }
        }
    }
}
```

Call this fn at the end of the existing fusion-plan pass (find where `FusionPlan` is returned and add a `verify_fused_sites_have_no_intermediate(&plan, wengert);` line before return). If the pass doesn't have wengert in scope, thread it through — the caller already has it available (`invoke_wrga_if_enabled` in `stmt.rs`).

- [ ] **Step 5: Run the test — expect pass**

```bash
cargo test -p nsl-codegen --test wrga_adapter_runtime fused_lora_site_leaves_no_intermediate 2>&1 | tail -15
```

Note: the test may pass even before Task 4 lands if (a) no `EpilogueFusedLora` decisions are produced yet (assertion loop is empty) or (b) the count check passes because today's adapter-site names don't start with `lora_`. That's fine — the test is the REGRESSION net, not a proof of fusion. Task 5's Build 4 fused is the proof.

- [ ] **Step 6: Regression**

```bash
cargo test -p nsl-codegen --tests 2>&1 | tail -3
cargo build --features cuda 2>&1 | tail -3
```

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-codegen/src/wrga_fusion.rs crates/nsl-codegen/src/lib.rs crates/nsl-codegen/tests/wrga_adapter_runtime.rs
git commit -m "feat(wrga): verify_fused_sites_have_no_intermediate — B.3 invariant net"
```

---

## Task 4: Dispatch wiring — new FFIs + conditional AST rewrite + PTX registration

**Files:**
- Create: `crates/nsl-runtime/src/fused_adapter.rs`
- Modify: `crates/nsl-runtime/src/lib.rs` (add `pub mod fused_adapter;`)
- Modify: `crates/nsl-codegen/src/wrga_adapter_rewrite.rs` (conditional emission)
- Modify: `crates/nsl-codegen/src/wrga_adapter_inject.rs` (`AdapterSite.fusion_decision` field)
- Modify: `crates/nsl-codegen/src/compiler/mod.rs` (`fused_ptx_kernels` field on Compiler)
- Modify: `crates/nsl-codegen/src/compiler/entry_points.rs` (PTX synth + dedup + embed)
- Modify: `crates/nsl-codegen/src/builtins.rs` (register the two new FFIs)

**Goal:** On sm_80+, LoRA / IA³ sites with `FusionDecision::EpilogueFusedLora` (or `ActivationFusedIa3`) emit a SINGLE FFI call per site. Existing `nsl_tensor_matmul` is untouched.

- [ ] **Step 1: Create the runtime FFI stubs**

Create `crates/nsl-runtime/src/fused_adapter.rs`:

```rust
//! WRGA B.3 Task 4: FFIs that launch the synthesized fused LoRA / IA³ kernels.
//!
//! Kernels are synthesized at compile time (see nsl_codegen::wrga_fused_ptx)
//! and embedded as static strings in the output binary.  The runtime loads
//! the kernel by handle, configures CUDA grid/block dims from tensor shapes,
//! and launches.
//!
//! Existing nsl_tensor_matmul is untouched — these are dedicated new FFIs
//! with clean signatures tailored to adapter fusion.

use std::ffi::c_void;

/// Launch the epilogue-fused LoRA matmul.
///
/// Tensor pointers point to NslTensor structs (device-resident buffers).
/// ptx_kernel_handle is an opaque index into a compile-time kernel table
/// emitted alongside the object file.
#[no_mangle]
pub extern "C" fn nsl_adapter_fused_lora_matmul(
    x_ptr: i64,
    w_ptr: i64,
    a_ptr: i64,
    b_ptr: i64,
    scale: f32,
    ptx_kernel_handle: i64,
) -> i64 {
    // B.3 MVP: allocate the output tensor (same shape as x @ W), load the
    // PTX kernel from the registry, launch with grid/block derived from the
    // tensor's (batch, d_out) dims.  A real implementation goes through the
    // existing CUDA launcher (see flash_attention.rs for the pattern).
    //
    // For B.3's initial commit, implement a CPU fallback that computes
    // the same math (x @ W + (x @ A) @ B * scale) in f32 — this lets the
    // equivalence tests validate the FFI contract without requiring the
    // CUDA launcher to be fully wired.  The CUDA launcher is the immediate
    // follow-up commit inside Task 4.
    //
    // TODO in the implementer's first commit: replace this stub with a
    // real CUDA launch.  The stub is DELIBERATELY a CPU fallback to unblock
    // the dispatch-wiring work; the equivalence test (Task 5 Build 4 fused)
    // exercises the real CUDA path when --target=cuda_sm80 is set.
    let _ = (x_ptr, w_ptr, a_ptr, b_ptr, scale, ptx_kernel_handle);
    // Caller (codegen) must NOT rely on this stub returning a valid tensor
    // pointer until the CUDA launch lands.  The integration test for Task 4
    // asserts only that the FFI is reachable; Task 5 asserts numerical
    // correctness.
    0
}

/// Launch the epilogue-fused IA³ matmul.
#[no_mangle]
pub extern "C" fn nsl_adapter_fused_ia3_matmul(
    x_ptr: i64,
    w_ptr: i64,
    gamma_ptr: i64,
    ptx_kernel_handle: i64,
) -> i64 {
    let _ = (x_ptr, w_ptr, gamma_ptr, ptx_kernel_handle);
    0
}
```

In `crates/nsl-runtime/src/lib.rs`, add:

```rust
pub mod fused_adapter;
```

- [ ] **Step 2: Extend `AdapterSite` with `fusion_decision` field**

In `crates/nsl-codegen/src/wrga_adapter_inject.rs`, extend `AdapterSite`:

```rust
pub struct AdapterSite {
    pub site_id: String,
    pub kind: crate::AdapterKind,
    pub target_param: String,
    pub rank: i64,
    pub alpha: i64,
    pub synthesized_fields: Vec<String>,
    pub input_dim: u32,
    pub output_dim: u32,
    // B.3: Fusion decision for this site, if the fusion pass made one.
    // None means the fusion pass hasn't run yet or decided NoOp.
    pub fusion_decision: Option<crate::wrga_fusion::FusionTarget>,
}
```

Default to `None` in all construction sites. `run_with_compiler` leaves it `None`; the fusion pass (`wrga_fusion.rs`) fills it in after deciding. Thread the fill through the existing `invoke_wrga_if_enabled` flow — after `build_fusion_plan` returns a `FusionPlan`, walk `plan.decisions` and update each matching `AdapterSite.fusion_decision`.

- [ ] **Step 3: Add `fused_ptx_kernels` registry on Compiler**

In `crates/nsl-codegen/src/compiler/mod.rs`, add to `Compiler`:

```rust
/// B.3 Task 4: synthesized fused LoRA/IA³ PTX kernels, keyed by
/// dedup signature (m, n, k, rank, target_sm).  Sites that share a
/// signature share one entry.
pub fused_ptx_kernels: std::collections::HashMap<
    crate::wrga_fused_ptx::LoraKernelKey,
    String,  // PTX source
>,
```

Init to `HashMap::new()` in `Compiler::new`.

- [ ] **Step 4: Modify the LoRA AST rewrite to be conditional**

In `crates/nsl-codegen/src/wrga_adapter_rewrite.rs`, find `synthesize_lora_adapted`. Wrap the existing triple-emission logic with a dispatch on `site.fusion_decision`:

```rust
fn synthesize_lora_adapted(
    original: &Expr,
    lhs: &Expr,
    site: &AdapterSite,
    ctx: &mut RewriteContext,
) -> Expr {
    // B.3: on sm_80+ with EpilogueFusedLora decision, emit a single FFI call.
    let is_fused = matches!(
        site.fusion_decision,
        Some(crate::wrga_fusion::FusionTarget::EpilogueFusedLora { .. })
    );
    let sm_ok = ctx.target_sm.map(|sm| sm >= 80).unwrap_or(false);

    if is_fused && sm_ok {
        return synthesize_lora_fused_call(original, lhs, site, ctx);
    }

    // Fall through to B.2.1's unfused triple-matmul expression (existing code).
    synthesize_lora_unfused_triple(original, lhs, site, ctx)
}

/// B.3: emit a single FFI call to nsl_adapter_fused_lora_matmul.
///
/// Produces the expression:
///   nsl_adapter_fused_lora_matmul(x, self.W, self.lora_A_<site>,
///                                  self.lora_B_<site>, scale, kernel_handle)
fn synthesize_lora_fused_call(
    _original: &Expr,
    lhs: &Expr,
    site: &AdapterSite,
    ctx: &mut RewriteContext,
) -> Expr {
    let scale = (site.alpha as f32) / (site.rank as f32);
    let kernel_handle = ctx.fused_kernel_handle_for(site);  // populated by compiler at PTX reg time
    build_call(
        "nsl_adapter_fused_lora_matmul",
        vec![
            lhs.clone(),
            build_self_member_access(site.target_field(), ctx),
            build_self_member_access(&format!("lora_A_{}", site.site_id), ctx),
            build_self_member_access(&format!("lora_B_{}", site.site_id), ctx),
            Expr::float_literal(scale as f64, _original.span),
            Expr::int_literal(kernel_handle as i64, _original.span),
        ],
        _original.span,
    )
}
```

The existing unfused implementation gets renamed to `synthesize_lora_unfused_triple` but is otherwise unchanged.

Rename the current `synthesize_lora_adapted` body to `synthesize_lora_unfused_triple` as a pure refactor first, verify tests pass, then add the conditional.

- [ ] **Step 5: Mirror the pattern for IA³**

In the same file, `synthesize_ia3_adapted` gets the same conditional-dispatch treatment:

```rust
fn synthesize_ia3_adapted(
    original: &Expr,
    site: &AdapterSite,
    ctx: &mut RewriteContext,
) -> Expr {
    let is_fused = matches!(
        site.fusion_decision,
        Some(crate::wrga_fusion::FusionTarget::ActivationFusedIa3)
    );
    let sm_ok = ctx.target_sm.map(|sm| sm >= 80).unwrap_or(false);

    if is_fused && sm_ok {
        return synthesize_ia3_fused_call(original, site, ctx);
    }
    synthesize_ia3_unfused(original, site, ctx)
}
```

- [ ] **Step 6: Register the new FFIs in the compiler builtins**

In `crates/nsl-codegen/src/builtins.rs` (or wherever `nsl_tensor_matmul` is registered), add:

```rust
// B.3: epilogue-fused adapter matmuls.
registry.declare_runtime_fn(
    "nsl_adapter_fused_lora_matmul",
    &[I64, I64, I64, I64, F32, I64],  // x, W, A, B, scale, kernel_handle
    I64,  // → y_ptr
);
registry.declare_runtime_fn(
    "nsl_adapter_fused_ia3_matmul",
    &[I64, I64, I64, I64],  // x, W, gamma, kernel_handle
    I64,
);
```

- [ ] **Step 7: Emit PTX kernels at compile time**

In `crates/nsl-codegen/src/compiler/entry_points.rs`, after WRGA fusion pass completes, walk `plan.fusion.decisions` and for each `EpilogueFusedLora` (or `ActivationFusedIa3`), synthesize the PTX and store in `compiler.fused_ptx_kernels`:

```rust
for decision in &plan.fusion.decisions {
    match &decision.target {
        crate::wrga_fusion::FusionTarget::EpilogueFusedLora { rank } => {
            let cfg = crate::wrga_fused_ptx::FusedLoraConfig {
                site_id: decision.site.clone(),
                m: /* from AdapterSite */,
                n: /* ... */,
                k: /* ... */,
                rank: *rank as u32,
                target_sm: options.target_sm.unwrap_or(80),
            };
            let key = cfg.kernel_key();
            compiler.fused_ptx_kernels.entry(key).or_insert_with(|| {
                crate::wrga_fused_ptx::synthesize_fused_lora_ptx(&cfg)
            });
        }
        crate::wrga_fusion::FusionTarget::ActivationFusedIa3 => {
            // Similar — use synthesize_fused_ia3_ptx.
        }
        _ => {}
    }
}
```

Kernel-handle assignment is the position in a deterministically-ordered `Vec<LoraKernelKey>` derived from the `HashMap` keys (sort by `(m, n, k, rank, target_sm)`). Handles are stable across a single compile.

- [ ] **Step 8: Write a reachability test**

Add to `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs`:

```rust
/// B.3 Task 4: verify the compiled binary references the new FFI when
/// fusion is enabled.  This is a compile-time proof that the AST rewrite
/// picked the fused branch — runtime correctness is Task 5's Build 4 fused.
#[test]
fn task_4_fused_ffi_is_referenced_when_target_sm80() {
    // Use debug_compile_and_return_plan with target_sm=80 and a LoRA site,
    // then inspect the emitted IR / call graph for nsl_adapter_fused_lora_matmul.
    const SRC: &str = r#"
@adapter(type=lora, target=["m.w"], rank=2, alpha=2)
let m = Toy()

model Toy:
    let w: Tensor<[8, 8], f32> = zeros([8, 8])

    fn forward(self, x: Tensor<[4, 8], f32>) -> Tensor<[4, 8], f32>:
        return x @ self.w

fn main():
    let x: Tensor<[4, 8], f32> = ones([4, 8])
    let _y = m.forward(x)
"#;
    let opts = nsl_codegen::CompileOptions {
        wrga_inputs: Some(nsl_codegen::WrgaInputs {
            adapter: vec![nsl_codegen::AdapterDecoratorConfig {
                kind: nsl_codegen::AdapterKind::Lora,
                targets: vec!["m.w".into()],
                rank: Some(2),
                alpha: Some(2),
            }],
            ..Default::default()
        }),
        source_ad: true,
        target: "cuda_sm80".into(),
        ..Default::default()
    };
    let plan = nsl_codegen::debug_compile_and_return_plan(SRC, &opts)
        .expect("compile must succeed")
        .expect("wrga::run must fire");
    // fused_ptx_kernels on the returned plan must have ≥ 1 entry.
    // (Expose via a new getter if needed — plan may not carry this today;
    // adding a side-channel on the plan is acceptable for Task 4.)
    let fused_count = plan.fusion.decisions.iter()
        .filter(|d| matches!(d.target, nsl_codegen::FusionTarget::EpilogueFusedLora { .. }))
        .count();
    assert!(fused_count >= 1,
        "expected at least one EpilogueFusedLora decision; got: {:?}",
        plan.fusion.decisions);
}
```

- [ ] **Step 9: Run the tests**

```bash
cargo test -p nsl-codegen --tests 2>&1 | tail -3
cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence task_4_fused_ffi 2>&1 | tail -10
cargo build --features cuda 2>&1 | tail -3
```

- [ ] **Step 10: Commit**

```bash
git add crates/nsl-runtime/src/fused_adapter.rs crates/nsl-runtime/src/lib.rs \
        crates/nsl-codegen/src/wrga_adapter_inject.rs \
        crates/nsl-codegen/src/wrga_adapter_rewrite.rs \
        crates/nsl-codegen/src/compiler/mod.rs \
        crates/nsl-codegen/src/compiler/entry_points.rs \
        crates/nsl-codegen/src/builtins.rs \
        crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs
git commit -m "feat(wrga): fused LoRA/IA3 FFIs + conditional AST rewrite + PTX registration"
```

---

## Task 5: Dual-tolerance equivalence test — `build_4_fused` + `build_5_kernel_count`

**Files:**
- Modify: `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs`

**Goal:** Keep B.2.1's Build 4 at 1e-5 (unfused path, sm_75 fallback, and explicit opt-out). Add `build_4_fused` at **1e-4** (NOT 1e-3 — 1e-3 is a bug signal). Add `build_5_kernel_count` asserting one kernel launch per adapter site on sm_80+.

- [ ] **Step 1: Write `build_4_fused`**

Append to `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs`:

```rust
/// B.3 Task 5: fused path numerical correctness.
///
/// Same source as Build 4 (A=ones, B=ones, x=ones, W=zeros, alpha=rank=2).
/// Compiled with target=cuda_sm80 to force the fused epilogue kernel.
/// Expected: every output element = 16.0 ± 1e-4.
///
/// Tolerance rationale (from spec §3):
///   - 1e-4 is correct for f32-accumulated MMA with identical inputs.
///   - 1e-3 is a BUG SIGNAL — likely causes: missing f32→f16 conversion,
///     incorrect accumulator reset between main and epilogue MMA, or
///     register aliasing.  If 1e-4 fails, investigate before loosening.
///   - Failure magnitude is diagnostic:
///       ~1e-3 → register aliasing or double-count bug
///       ~1.0+ → scale factor wrong, or the fused kernel never ran
///       exactly == unfused → test didn't exercise the fused path
///         (check build_5 kernel-count assertion)
#[test]
fn build_4_fused_lora_rewrite_load_bearing_proof() {
    const SRC: &str = r#"
@adapter(type=lora, target=["m.w"], rank=2, alpha=2)
let m = Toy()

model Toy:
    let w: Tensor<[8, 8], f32> = zeros([8, 8])

    fn forward(self, x: Tensor<[4, 8], f32>) -> Tensor<[4, 8], f32>:
        return x @ self.w

fn main():
    m.lora_A_m_w__lora = ones([2, 8])
    m.lora_B_m_w__lora = ones([8, 2])
    let x: Tensor<[4, 8], f32> = ones([4, 8])
    let y = m.forward(x)
    print(y)
"#;
    let tmp = tempfile::TempDir::new().unwrap();
    let src_path = tmp.path().join("build4_fused.nsl");
    std::fs::write(&src_path, SRC).unwrap();

    let stdlib = std::env::var("NSL_STDLIB_PATH")
        .unwrap_or_else(|_| "c:/Users/bwiem/projects/NSL-wrga-b3/stdlib".to_string());

    let mut cmd = assert_cmd::Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", &stdlib)
        .arg("run").arg("--source-ad")
        .arg("--target").arg("cuda_sm80")   // Force fused path
        .arg(&src_path);
    let output = cmd.output().expect("nsl run failed");
    if !output.status.success() {
        panic!("nsl run exit={}\nstdout:{}\nstderr:{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr));
    }
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let tensor = parse_tensor_2d(&stdout);
    let expected = 16.0_f32;
    let tolerance = 1e-4_f32;
    for (i, row) in tensor.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            let err = (v - expected).abs();
            assert!(
                err < tolerance,
                "Build 4 fused failure at [{i},{j}]: expected {expected}, got {v}, diff {err}.\n\
                 Tolerance: {tolerance} (1e-3 is a BUG, do NOT loosen).\n\
                 Diagnostic:\n\
                   - diff ~1e-3: register aliasing or double-count bug\n\
                   - diff ~1.0+: scale factor wrong, or fused kernel never ran\n\
                   - diff == 0 exactly (also failing tolerance): unlikely, \
                     but would indicate test didn't exercise fused path"
            );
        }
    }
}
```

- [ ] **Step 2: Write `build_5_kernel_count`**

```rust
/// B.3 Task 5: fused path launches exactly 1 kernel per adapter site,
/// versus 3 for the unfused path (x@W, x@A, (x@A)@B).
#[test]
fn build_5_fused_launches_one_kernel_per_site() {
    const SRC: &str = r#"
@adapter(type=lora, target=["m.w"], rank=2, alpha=2)
let m = Toy()

model Toy:
    let w: Tensor<[8, 8], f32> = zeros([8, 8])

    fn forward(self, x: Tensor<[4, 8], f32>) -> Tensor<[4, 8], f32>:
        return x @ self.w

fn main():
    let x: Tensor<[4, 8], f32> = ones([4, 8])
    let _y = m.forward(x)
"#;
    let tmp = tempfile::TempDir::new().unwrap();
    let src_path = tmp.path().join("build5.nsl");
    std::fs::write(&src_path, SRC).unwrap();

    let stdlib = std::env::var("NSL_STDLIB_PATH")
        .unwrap_or_else(|_| "c:/Users/bwiem/projects/NSL-wrga-b3/stdlib".to_string());

    // Enable kernel-launch counter via env var (see Task 4's runtime FFI — it
    // increments a thread-local atomic each launch; test reads it after).
    let mut cmd = assert_cmd::Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", &stdlib)
        .env("NSL_KERNEL_LAUNCH_COUNTER", "1")
        .arg("run").arg("--source-ad")
        .arg("--target").arg("cuda_sm80")
        .arg(&src_path);
    let output = cmd.output().expect("nsl run failed");
    if !output.status.success() {
        panic!("nsl run exit={}\nstdout:{}\nstderr:{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr));
    }
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    // Runtime prints "[nsl-kernel-count] <N>" when NSL_KERNEL_LAUNCH_COUNTER=1.
    let count_line = stderr.lines()
        .find(|l| l.contains("[nsl-kernel-count]"))
        .expect(&format!("expected kernel-count line in stderr; got:\n{stderr}"));
    let count: usize = count_line
        .split_whitespace()
        .last()
        .unwrap()
        .parse()
        .unwrap();
    // 1 forward matmul site with LoRA, fused → 1 kernel.  Without fusion, 3.
    assert!(
        count == 1,
        "expected exactly 1 kernel launch for the fused LoRA site; got {count}.\n\
         >1 kernel means fusion didn't fire (AST rewrite emitted the unfused \
         triple despite FusionDecision=EpilogueFusedLora).  Check Task 4's \
         conditional in synthesize_lora_adapted.",
    );
}
```

The runtime-side kernel-count side-channel is a small addition to `nsl-runtime` — every kernel launch increments a thread-local atomic counter, and at program exit (or via an `at_exit` hook) it prints `[nsl-kernel-count] <N>` to stderr iff `NSL_KERNEL_LAUNCH_COUNTER=1` is set. Wire this in if it doesn't exist yet; if it does, reuse.

- [ ] **Step 3: Run — expect failure initially**

```bash
cargo build --bin nsl 2>&1 | tail -3
cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence build_4_fused build_5_fused 2>&1 | tail -20
```

Expected: both new tests fail because the CUDA launcher stub in `fused_adapter.rs` returns `0` (null tensor). Until the real CUDA launch lands, `build_4_fused` gets a parse error on stdout or a crash.

- [ ] **Step 4: Wire the real CUDA launch in `nsl_adapter_fused_lora_matmul`**

In `crates/nsl-runtime/src/fused_adapter.rs`, replace the stub with an actual cudarc-based launch:

```rust
#[no_mangle]
pub extern "C" fn nsl_adapter_fused_lora_matmul(
    x_ptr: i64,
    w_ptr: i64,
    a_ptr: i64,
    b_ptr: i64,
    scale: f32,
    ptx_kernel_handle: i64,
) -> i64 {
    // Resolve tensors.
    let x = crate::tensor::NslTensor::from_ptr(x_ptr);
    let w = crate::tensor::NslTensor::from_ptr(w_ptr);
    let a = crate::tensor::NslTensor::from_ptr(a_ptr);
    let b = crate::tensor::NslTensor::from_ptr(b_ptr);

    // Derive y shape: same as x @ W → [batch, d_out].
    let batch = x.shape[0];
    let d_out = w.shape[0];  // W is [d_out, d_in]
    let y = crate::tensor::nsl_tensor_zeros_on_gpu(&[batch, d_out]);

    // Load PTX kernel from the registry (set up by codegen).
    let module = crate::fused_adapter_registry::get_module(ptx_kernel_handle);
    let kernel = module.get_kernel("nsl_wrga_fused_lora_m16n8k16r2")
        .or_else(|_| panic!("fused LoRA kernel not found for handle {ptx_kernel_handle}"));

    // Configure launch.  Grid: (batch/16, d_out/8, 1).  Block: 128 threads (4 warps).
    let grid = (batch.div_ceil(16), d_out.div_ceil(8), 1);
    let block = (128, 1, 1);
    unsafe {
        kernel.launch(
            grid, block,
            &[
                &x.data_ptr(), &w.data_ptr(), &a.data_ptr(), &b.data_ptr(),
                &scale, &y.data_ptr(),
            ],
        ).expect("fused LoRA kernel launch failed");
    }

    // Kernel-count side channel for Task 5's build_5.
    if std::env::var("NSL_KERNEL_LAUNCH_COUNTER").is_ok() {
        crate::fused_adapter_registry::increment_launch_counter();
    }

    y.as_i64_ptr()
}
```

Full implementation details (module-loading, registry) depend on the existing CUDA launcher pattern in `flash_attention.rs`'s runtime launcher. Mirror that pattern.

- [ ] **Step 5: Run both tests — expect pass**

```bash
cargo build --bin nsl 2>&1 | tail -3
cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence 2>&1 | tail -15
```

Expected: all 6 tests in the equivalence suite pass (Builds 1-4 from B.2.1 + build_4_fused + build_5_fused).

- [ ] **Step 6: Full regression**

```bash
cargo test -p nsl-codegen --tests 2>&1 | tail -3
cargo test -p nsl-cli --test e2e -- --test-threads=1 2>&1 | tail -3
cargo test -p nsl-cli --test wrga_report_cli 2>&1 | tail -3
cargo build --features cuda 2>&1 | tail -3
```

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs crates/nsl-runtime/src/fused_adapter.rs
git commit -m "test(wrga): build_4_fused at 1e-4 + build_5_kernel_count — B.3 load-bearing proofs"
```

---

## Task 6: Close-out

**Files:**
- Create (outside worktree): `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/project_wrga_milestone_b3.md`
- Modify: `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/MEMORY.md` (append pointer)
- **No code changes.**

- [ ] **Step 1: Run the full regression**

```bash
cargo test -p nsl-semantic 2>&1 | grep "test result" | tail -3
cargo test -p nsl-codegen --lib 2>&1 | grep "test result" | tail -3
cargo test -p nsl-codegen --tests 2>&1 | grep "test result" | tail -3
cargo test -p nsl-codegen flash_attention 2>&1 | grep "test result" | tail -3
cargo test -p nsl-cli --test e2e -- --test-threads=1 2>&1 | grep "test result" | tail -3
cargo test -p nsl-cli --test wrga_report_cli 2>&1 | grep "test result" | tail -3
cargo test -p nsl-cli --test wrga_adapter_runtime_equivalence 2>&1 | grep "test result" | tail -3
cargo build --features cuda 2>&1 | tail -3
cargo build --release --features cuda 2>&1 | tail -3
cargo clippy -p nsl-codegen -p nsl-semantic -p nsl-cli --features cuda --all-targets 2>&1 | grep -E "^warning:" | grep -v "nsl-runtime" | head -20
```

Expected: every test count ≥ Pre-flight baseline. Equivalence suite: 6 tests (4 B.2.1 + `build_4_fused` + `build_5_fused`).

- [ ] **Step 2: Write the memory file**

Create `project_wrga_milestone_b3.md`:

```markdown
---
name: WRGA Milestone B.3 — epilogue-fused LoRA/IA³ MMA PTX complete
description: "+0 memory ops" paper claim realized. Fused kernel lives for LoRA and IA³ on sm_80+; sm_75 fallback unchanged. GatedLoRA deferred to B.3.1.
type: project
---

## WRGA Milestone B.3 (landed on branch feat/wrga-milestone-b3)

**Per-task commits:**
- <Task 1 SHA> feat(wrga): matmul_mma — reusable m16n8k16 MMA primitives for B.3
- <Task 2 SHA> feat(wrga): synthesize_fused_{lora,ia3}_ptx — interleaved epilogue, scale as .param
- <Task 3 SHA> feat(wrga): verify_fused_sites_have_no_intermediate — B.3 invariant net
- <Task 4 SHA> feat(wrga): fused LoRA/IA3 FFIs + conditional AST rewrite + PTX registration
- <Task 5 SHA> test(wrga): build_4_fused at 1e-4 + build_5_kernel_count — B.3 load-bearing proofs

**What shipped:**
- `matmul_mma::emit_mma_instruction` + fragment-load helpers copied-then-parameterized from flash_attention.rs (FA untouched).
- `wrga_fused_ptx::synthesize_fused_lora_ptx` and `synthesize_fused_ia3_ptx`.
- LoRA kernel with INTERLEAVED epilogue: each main K-iteration also accumulates `x_tile @ A_tile` into an intermediate register. Post-loop: one more MMA for `(x@A) @ B`, scale-mul, add to main_accum, store.
- Scale passed as `.param .f32` — kernel dedup by `(m, n, k, rank, target_sm)` survives different alpha values.
- Two new FFIs: `nsl_adapter_fused_lora_matmul`, `nsl_adapter_fused_ia3_matmul`. `nsl_tensor_matmul` untouched.
- B.2.1's `synthesize_lora_adapted` / `synthesize_ia3_adapted` now conditional: fused FFI call on sm_80+ with EpilogueFusedLora/ActivationFusedIa3; unfused triple otherwise.
- `build_4_fused` asserts 16.0 ± 1e-4 per element (NOT 1e-3 — 1e-3 is a bug signal).
- `build_5_kernel_count` asserts exactly 1 kernel launch per fused adapter site.
- Rank > 16 is a hard compile error (inject-pass diagnostic).

**Known limitations (documented, deferred):**
- GatedLoRA epilogue fusion (needs PTX sigmoid) → B.3.1.
- Multi-pass epilogue for rank > 16 → follow-up.
- sm_90+ WGMMA path → follow-up (FA has WGMMA but B.3 stays on Ampere m16n8k16).

**Test coverage delta vs B.2.1 baseline:**
- `wrga_adapter_runtime_equivalence`: 4 → 6 tests.
- `matmul_mma` unit tests: 3 new.
- `wrga_fused_ptx` unit tests: 6 new.
- `fused_lora_site_leaves_no_intermediate_activation` (wrga_adapter_runtime.rs): 1 new.
- Total new tests: ~16.

**Windows stack budget:** 16 MB (from B.2.1) still holds; no increase needed.
```

Append to `MEMORY.md`:

```
## WRGA Milestone B.3 (landed on feat/wrga-milestone-b3)
- See [project_wrga_milestone_b3.md](project_wrga_milestone_b3.md) — epilogue-fused LoRA/IA³ MMA PTX; "+0 memory ops" realized; Build 4 fused passes at 1e-4; GatedLoRA deferred to B.3.1
```

- [ ] **Step 3: Do NOT merge**

The controlling session performs the merge after subagent review. Report the branch status (per-task SHAs, test counts, known limitations) and stop.

---

## Final verification (do not skip before reporting DONE)

- [ ] `cargo test --workspace --features cuda` — accept Windows parallel-test flake; single-thread nsl-cli e2e re-run must pass.
- [ ] `cargo clippy --features cuda --all-targets -- -D warnings` on touched crates; no new warnings in B.3 code.
- [ ] `cargo build --release --features cuda` clean.
- [ ] `build_4_fused` passes with every element at 16.0 ± 1e-4. If it fails by ~1e-3, investigate register aliasing / accumulator reset BEFORE loosening tolerance.
- [ ] `build_5_fused` confirms exactly 1 kernel launch per adapter site.

---

## Out of scope

- **GatedLoRA epilogue fusion** — PTX sigmoid (Taylor approx or lookup table); B.3.1.
- **Multi-pass epilogue for rank > 16** — hard compile error today; follow-up milestone.
- **`nsl_tensor_matmul` (Cutlass path) refactor** — untouched.
- **FA backward MMA refactor to use matmul_mma** — Task 1 copies, doesn't migrate.
- **Non-Ampere tensor cores (sm_90+ WGMMA)** — sm_90 falls back to the existing FA WGMMA for attention, but B.3's fused-adapter kernel stays Ampere-only.

---

## Risk log

1. **MMA helper parameterization hidden coupling** — copying from FA may pull attention-specific SMEM-swizzle assumptions. Mitigation: Task 1 unit tests exercise the extracted helpers on plain-matmul inputs (no attention context). If coupling surfaces, fork the helpers rather than unify.

2. **Register budget actually < 16 rank** — the analysis estimates; `ptxas -v` on the first real kernel gives the authoritative number. Mitigation: after Task 2 ships, run `ptxas -v` against a synthesized kernel and compare to the ≤16 ceiling. If actual is lower, tighten the inject-pass error.

3. **Epilogue accumulator double-init** — common bug: freshly zeroing the main accumulator before the epilogue MMA. Mitigation: `lora_ptx_folds_epilogue_into_main_accum` test asserts exactly 8 `add.f32 %main_accum*, %main_accum*, %epi_final*` lines with no intervening `mov.f32 %main_accum*, 0f00000000`.

4. **Associativity tolerance failure** — `build_4_fused` at 1e-4 should pass trivially for unit inputs. Failure magnitude is diagnostic (see test doc). If the failure is ~1e-3, do NOT loosen tolerance; investigate register aliasing or accumulator reset.

5. **PTX kernel launch bringup** — cudarc launch signature matching, PTX JIT compilation, tensor-pointer marshaling. Mitigation: mirror the existing flash_attention.rs launcher pattern exactly. Don't invent a new abstraction.

6. **Kernel-handle invalidation** — handles are assigned by position in a sorted `Vec<LoraKernelKey>`. If a subsequent task reorders or drops a key, handles shift. Mitigation: handles must be stable within a single compile; cross-compile reproducibility comes from sorting keys by `(m, n, k, rank, target_sm)` deterministically.

7. **Windows stack overflow from PTX synth** — `synthesize_fused_lora_ptx` has deep nested loop emission. If tests regress with stack overflow, bump the main-thread bootstrap from 16 MB to 32 MB. Unlikely given the 16 MB held for B.2.1's heavier work.

8. **Runtime launcher is CUDA-only** — on a Windows build without CUDA, the FFI stub returns 0 and tests panic. Mitigation: `build_4_fused` / `build_5_fused` are gated on `cfg(feature = "cuda")` (the same way FA tests are gated). Non-CUDA CI continues to exercise the unfused path only.
