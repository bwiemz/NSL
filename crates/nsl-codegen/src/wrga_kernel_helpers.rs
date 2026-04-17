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
