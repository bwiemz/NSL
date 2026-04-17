//! WRGA-specific PTX helpers that are not shareable with FA.
//!
//! Tile-staging, register-pool emission, and output-tile coord math.
//! Kept out of `kernel_skeleton/` because these are WRGA-kernel-specific
//! (single-warp-per-tile, m16n8k16 fixed, fp32 output) and would drag
//! FA-irrelevant config surface into the skeleton.
//!
//! This module is populated incrementally across Tasks C1-C3:
//! - C1: register budget, register pool emitter, output-tile coords,
//!       matmul_mma lane-init.
//! - C2: tile-staging helpers (emit_lora_stage_{x,w,a,b}_tile).
//! - C3: accumulator init, predicated store.

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
    // Named pointer/base/scratch registers.
    ptx.push_str("    .reg .u64 %rd_x, %rd_w, %rd_a, %rd_b, %rd_y;\n");
    ptx.push_str("    .reg .u64 %x_tile_base, %w_tile_base, %a_tile_base, %b_tile_base;\n");
    ptx.push_str("    .reg .u64 %shmem_base;\n");
    ptx.push_str("    .reg .u32 %tid_x, %warp_id, %lane, %bid_x, %bid_y;\n");
    ptx.push_str("    .reg .f32 %scale_reg;\n");
    // Lane-derived addressing registers (load-bearing for matmul_mma helpers).
    ptx.push_str("    .reg .u32 %mma_a_row, %mma_b_row, %mma_addr;\n");
    ptx.push_str("    .reg .u64 %smem_base_x, %smem_base_w, %smem_base_a, %smem_base_b;\n");
    // Output tile coords.
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
    ptx.push_str("    shl.b32 %row_base, %bid_x, 4;\n");
    ptx.push_str("    shl.b32 %col_base, %bid_y, 3;\n");
    // Compute m_real = min(m - row_base, 16) and n_real = min(n - col_base, 8).
    ptx.push_str(&format!("    mov.u32 %r0, {};\n", m));
    ptx.push_str("    sub.u32 %r0, %r0, %row_base;\n");
    ptx.push_str("    min.u32 %m_real, %r0, 16;\n");
    ptx.push_str(&format!("    mov.u32 %r1, {};\n", n));
    ptx.push_str("    sub.u32 %r1, %r1, %col_base;\n");
    ptx.push_str("    min.u32 %n_real, %r1, 8;\n");
}

/// Emit the lane-derivation math that initializes `%mma_a_row` and
/// `%mma_b_row` per the m16n8k16 lane layout spec.  Called ONCE in the
/// prolog before any matmul_mma helper.
///
/// For m16n8k16 with a single warp per tile:
///   A-fragment is 16×16 f16, distributed across 32 lanes.  Thread t's
///   A-fragment row index is `t / 4` (8 threads share 4 consecutive
///   K-elements within a row's 16 columns, 4 threads cover the 4
///   consecutive K-positions a fragment holds).
///
///   B-fragment is 16×8 f16 col-major.  Thread t's B-fragment row is
///   `(t % 8) * 4` — each of 8 threads per column covers 4 consecutive
///   K rows.
///
/// B.3 shipped pseudocode BECAUSE these registers were declared in the
/// `.reg` pool but never initialized.  This helper is the fix.  Callers
/// of matmul_mma::emit_load_a_fragment_smem / emit_load_b_fragment_smem
/// MUST call `emit_matmul_mma_lane_init` before the first such helper.
pub fn emit_matmul_mma_lane_init(ptx: &mut String) {
    ptx.push_str("    // matmul_mma lane-index init (m16n8k16 layout)\n");
    // A-fragment row: tid / 4
    ptx.push_str("    shr.u32 %mma_a_row, %tid_x, 2;\n");
    // B-fragment row: (tid % 8) * 4
    ptx.push_str("    and.b32 %r2, %tid_x, 7;\n");
    ptx.push_str("    shl.b32 %mma_b_row, %r2, 2;\n");
}

// ─── C2: Tile-staging helpers ──────────────────────────────────────────────

/// SMEM byte offsets within the shmem[1536] array for LoRA kernel:
///   [   0,  512) — x_tile  (16×16 f16)
///   [ 512,  768) — w_tile  (16×8  f16)
///   [ 768, 1280) — a_tile  (16×16 f16, rank-padded)
///   [1280, 1536) — b_tile  (16×8  f16, rank-padded)
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
    // matmul_mma helpers look up %smem_base_x/w/a/b by name — alias.
    ptx.push_str("    mov.u64 %smem_base_x, %x_tile_base;\n");
    ptx.push_str("    mov.u64 %smem_base_w, %w_tile_base;\n");
    ptx.push_str("    mov.u64 %smem_base_a, %a_tile_base;\n");
    ptx.push_str("    mov.u64 %smem_base_b, %b_tile_base;\n");
}

/// Stage the x_tile slice for K-iteration `k_tile` from global (%rd_x)
/// into SMEM (%x_tile_base).  Handles m-tail (rows >= m get zeroed) and
/// k-tail (k-positions >= k get zeroed).
///
/// For simplicity (first rewrite), staging is serialized: if %tid_x == 0,
/// thread 0 does all loads.  Cooperative staging is a follow-up perf pass.
///
/// x global layout: [m, k] row-major, f16.  x[row, k_start + col].
pub fn emit_lora_stage_x_tile(ptx: &mut String, k_tile: u32, m: u32, k: u32) {
    ptx.push_str(&format!("    // Stage x_tile for K-tile {}\n", k_tile));
    ptx.push_str("    setp.eq.u32 %p0, %tid_x, 0;\n");
    ptx.push_str(&format!("    @!%p0 bra lora_x_stage_done_{};\n", k_tile));
    let k_start = k_tile * 16;
    for row in 0..16u32 {
        for col_pair in 0..8u32 {
            let col = col_pair * 2;
            let smem_offset = row * 32 + col_pair * 4;
            if row >= m || k_start + col >= k {
                // OOB (m-tail or k-tail) — zero-fill SMEM slot.
                ptx.push_str(&format!(
                    "    st.shared.b32 [%x_tile_base + {}], 0;\n",
                    smem_offset
                ));
            } else {
                let gl_offset = (row as u64) * (k as u64) * 2
                    + (k_start as u64 + col as u64) * 2;
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
    ptx.push_str(&format!("lora_x_stage_done_{}:\n", k_tile));
}

/// Stage the w_tile slice for K-iteration `k_tile`.  Shape: 16×8 f16.
/// W global layout: [k, n] row-major, f16.  W[k_start + row, col].
pub fn emit_lora_stage_w_tile(ptx: &mut String, k_tile: u32, n: u32, k: u32) {
    ptx.push_str(&format!("    // Stage w_tile for K-tile {}\n", k_tile));
    ptx.push_str("    setp.eq.u32 %p0, %tid_x, 0;\n");
    ptx.push_str(&format!("    @!%p0 bra lora_w_stage_done_{};\n", k_tile));
    let k_start = k_tile * 16;
    for row in 0..16u32 {
        for col_pair in 0..4u32 {
            let col = col_pair * 2;
            let smem_offset = row * 16 + col_pair * 4;
            if k_start + row >= k || col >= n {
                ptx.push_str(&format!(
                    "    st.shared.b32 [%w_tile_base + {}], 0;\n",
                    smem_offset
                ));
            } else {
                let gl_offset = ((k_start + row) as u64) * (n as u64) * 2 + (col as u64) * 2;
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
    }
    ptx.push_str(&format!("lora_w_stage_done_{}:\n", k_tile));
}

/// Stage the a_tile slice for K-iteration `k_tile`.  a_tile layout in
/// SMEM: 16 rows × 16 cols (rank-padded to 16), f16.
/// A global layout: [k, rank] row-major, f16.  A[k_start + row, col].
pub fn emit_lora_stage_a_tile(ptx: &mut String, k_tile: u32, rank: u32, k: u32) {
    ptx.push_str(&format!("    // Stage a_tile for K-tile {} (rank={})\n", k_tile, rank));
    ptx.push_str("    setp.eq.u32 %p0, %tid_x, 0;\n");
    ptx.push_str(&format!("    @!%p0 bra lora_a_stage_done_{};\n", k_tile));
    let k_start = k_tile * 16;
    for row in 0..16u32 {
        for col_pair in 0..8u32 {
            let col = col_pair * 2;
            let smem_offset = row * 32 + col_pair * 4;
            if k_start + row >= k || col >= rank {
                // OOB on k OR beyond rank — zero-fill (rank padding).
                ptx.push_str(&format!(
                    "    st.shared.b32 [%a_tile_base + {}], 0;\n",
                    smem_offset
                ));
            } else {
                let gl_offset = ((k_start + row) as u64) * (rank as u64) * 2 + (col as u64) * 2;
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
    }
    ptx.push_str(&format!("lora_a_stage_done_{}:\n", k_tile));
}

/// Stage the b_tile ONCE post-loop.  Shape: 16×8 f16 (rank-padded rows).
/// B global layout: [rank, n] row-major, f16.  B[row, col].
pub fn emit_lora_stage_b_tile(ptx: &mut String, rank: u32, n: u32) {
    ptx.push_str("    // Stage b_tile (post-loop, once)\n");
    ptx.push_str("    setp.eq.u32 %p0, %tid_x, 0;\n");
    ptx.push_str("    @!%p0 bra lora_b_stage_done;\n");
    for row in 0..16u32 {
        for col_pair in 0..4u32 {
            let col = col_pair * 2;
            let smem_offset = row * 16 + col_pair * 4;
            if row >= rank || col >= n {
                ptx.push_str(&format!(
                    "    st.shared.b32 [%b_tile_base + {}], 0;\n",
                    smem_offset
                ));
            } else {
                let gl_offset = (row as u64) * (n as u64) * 2 + (col as u64) * 2;
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
    }
    ptx.push_str("lora_b_stage_done:\n");
}

// ─── C3: Accumulator init + predicated output store ───────────────────────

/// Emit zero-init of N floating-point accumulator registers named
/// `%<base>0..%<base>(N-1)`.
pub fn emit_zero_accumulators(ptx: &mut String, base: &str, count: u32) {
    for i in 0..count {
        ptx.push_str(&format!("    mov.f32 %{}{}, 0f00000000;\n", base, i));
    }
}

/// Emit the output-tile store with predication on m_real and n_real.
///
/// The `main_accum<8>` register pool holds a 16×8 f32 output tile
/// distributed across 32 lanes.  Per the m16n8k16 D-fragment layout,
/// each thread owns 4 f32 values in the (row, col) grid:
///   main_accum0: (row_base_in_tile + 0, col_base_in_tile + 0)
///   main_accum1: (row_base_in_tile + 0, col_base_in_tile + 1)
///   main_accum2: (row_base_in_tile + 8, col_base_in_tile + 0)
///   main_accum3: (row_base_in_tile + 8, col_base_in_tile + 1)
/// where
///   row_base_in_tile = tid / 4      (0..8 depending on lane)
///   col_base_in_tile = (tid % 4) * 2  (0, 2, 4, or 6)
///
/// Global y is [m, n] f32 row-major.  The write address is
///   rd_y + ((row_base + row_in_tile) * n + col_base + col_in_tile) * 4
///
/// Stores are predicated on (row_in_tile < m_real && col_in_tile < n_real)
/// so tail boundary tiles (m%16 != 0 or n%8 != 0) don't overwrite
/// neighboring memory.
pub fn emit_lora_store_output(ptx: &mut String, n: u32) {
    ptx.push_str("    // Store main_accum to y with m_real/n_real predication\n");
    // row_base_in_tile = tid / 4
    ptx.push_str("    shr.u32 %r4, %tid_x, 2;\n");
    // col_base_in_tile = (tid % 4) * 2
    ptx.push_str("    and.b32 %r5, %tid_x, 3;\n");
    ptx.push_str("    shl.b32 %r5, %r5, 1;\n");
    // Per m16n8k16 D-fragment layout, each thread owns 4 outputs at
    // positions: (r,c), (r,c+1), (r+8,c), (r+8,c+1) relative to tile origin.
    let offsets: [(u32, u32); 4] = [(0, 0), (0, 1), (8, 0), (8, 1)];
    for (i, (row_off, col_off)) in offsets.iter().enumerate() {
        ptx.push_str(&format!("    // Store main_accum{} (row+{}, col+{})\n", i, row_off, col_off));
        // row_in_tile = %r4 + row_off
        ptx.push_str(&format!("    add.u32 %r6, %r4, {};\n", row_off));
        // col_in_tile = %r5 + col_off
        ptx.push_str(&format!("    add.u32 %r7, %r5, {};\n", col_off));
        // Predicate: row_in_tile < m_real AND col_in_tile < n_real
        ptx.push_str("    setp.lt.u32 %p1, %r6, %m_real;\n");
        ptx.push_str("    setp.lt.and.u32 %p1, %r7, %n_real, %p1;\n");
        // Global offset in bytes: ((row_base + row_in_tile) * n + col_base + col_in_tile) * 4
        ptx.push_str("    add.u32 %r8, %row_base, %r6;\n");
        ptx.push_str(&format!("    mul.lo.u32 %r8, %r8, {};\n", n));
        ptx.push_str("    add.u32 %r8, %r8, %col_base;\n");
        ptx.push_str("    add.u32 %r8, %r8, %r7;\n");
        ptx.push_str("    shl.b32 %r8, %r8, 2;\n");
        ptx.push_str("    cvt.u64.u32 %rd0, %r8;\n");
        ptx.push_str("    add.u64 %rd0, %rd_y, %rd0;\n");
        ptx.push_str(&format!("    @%p1 st.global.f32 [%rd0], %main_accum{};\n", i));
    }
}
