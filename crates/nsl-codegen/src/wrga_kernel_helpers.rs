//! WRGA-specific PTX helpers that are not shareable with FA.
//!
//! Tile-staging, register-pool emission, and output-tile coord math.
//! Kept out of `kernel_skeleton/` because these are WRGA-kernel-specific
//! (single-warp-per-tile, m16n8k16 fixed, fp32 output) and would drag
//! FA-irrelevant config surface into the skeleton.
//!
//! This module is populated incrementally across Tasks C1-C3, then E2:
//! - C1: register budget, register pool emitter, output-tile coords,
//!       matmul_mma lane-init.
//! - C2: tile-staging helpers (emit_lora_stage_{x,w,a,b}_tile).
//! - C3: accumulator init, predicated store.
//! - E2: IA³-specific helpers (register budget/pool, tile bases,
//!       γ-load, γ-broadcast-multiply).  IA³ reuses LoRA's staging,
//!       lane-init, and store helpers unchanged.

use crate::wrga_fused_ptx::{FusedIa3Config, FusedLoraConfig};

/// Register pool spec for WRGA fused LoRA kernel.
#[derive(Debug, Clone)]
pub struct LoraRegisterBudget {
    pub main_accum_count: u32,   // 8 (f32, x@W accumulator)
    pub epi_interm_count: u32,   // 4 (f32, x@A accumulator)
    pub epi_final_count: u32,    // 4 (f32, (x@A)@B accumulator)
    pub main_a_frag_count: u32,  // 4 (b32, x fragment)
    pub main_b_frag_count: u32,  // 2 (b32, W fragment)
    pub epi_a_frag_count: u32,   // 2 (b32, A-as-B-operand fragment — col-major B-fragment)
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
        epi_a_frag_count: 2,
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
///
/// Compile-time version: m and n are baked in as immediate constants.
/// Used only when the batch dimension is known at codegen time.
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

/// Dynamic version of `emit_lora_output_tile_coords` that reads m and n
/// from the runtime kernel parameters `%m_param` and `%n_param` instead
/// of baking in compile-time constants.  This is required because the
/// prescan synthesizes the kernel with a placeholder m (to produce a
/// single kernel binary), but the actual batch dimension is only known
/// at launch time.
///
/// Callers must ensure `%m_param` and `%n_param` are loaded from `.param`
/// slots before this is called.
pub fn emit_lora_output_tile_coords_dynamic(ptx: &mut String) {
    ptx.push_str("    // WRGA output tile coords (runtime m/n from kernel params)\n");
    ptx.push_str("    shl.b32 %row_base, %bid_x, 4;\n");
    ptx.push_str("    shl.b32 %col_base, %bid_y, 3;\n");
    // m_real = min(m_param - row_base, 16)
    ptx.push_str("    sub.u32 %r0, %m_param, %row_base;\n");
    ptx.push_str("    min.u32 %m_real, %r0, 16;\n");
    // n_real = min(n_param - col_base, 8)
    ptx.push_str("    sub.u32 %r1, %n_param, %col_base;\n");
    ptx.push_str("    min.u32 %n_real, %r1, 8;\n");
}

/// Emit the lane-derivation math that initializes `%mma_a_row` and
/// `%mma_b_row` per the m16n8k16 lane layout spec.  Called ONCE in the
/// prolog before any matmul_mma helper.
///
/// For m16n8k16 with a single warp per tile:
///   A-fragment is 16×16 f16, distributed across 32 lanes.  Thread t's
///   A-fragment row index is `t / 4` (0..7), which is the row within the
///   16-row tile that this thread owns.
///
///   B-fragment is k=16 × n=8 f16 col-major.  Per PTX ISA §9.7.13.4,
///   thread t owns k-rows {(t%4)*2, (t%4)*2+1} for b-reg[0] and
///   {(t%4)*2+8, (t%4)*2+9} for b-reg[1], at column (t/4).
///   So `mma_b_row = (t % 4) * 2` (k-row of first owned pair).
///   The column byte offset `(t/4)*2` must be added to the SMEM base
///   address BEFORE calling emit_load_b_fragment_smem; the caller is
///   responsible for this — see emit_b_lane_smem_bases in wrga_fused_ptx.
///
/// B.3 shipped pseudocode BECAUSE these registers were declared in the
/// `.reg` pool but never initialized.  This helper is the fix.  Callers
/// of matmul_mma::emit_load_a_fragment_smem / emit_load_b_fragment_smem
/// MUST call `emit_matmul_mma_lane_init` before the first such helper.
pub fn emit_matmul_mma_lane_init(ptx: &mut String) {
    ptx.push_str("    // matmul_mma lane-index init (m16n8k16 layout)\n");
    // A-fragment row: tid / 4   (0..7)
    ptx.push_str("    shr.u32 %mma_a_row, %tid_x, 2;\n");
    // B-fragment byte offset within col-major column: (tid % 4) * 4
    //   = k-pair index * 2 f16s * 2 bytes/f16 = (tid%4)*4 bytes.
    //   Column base offset (tid/4)*32 is baked into smem_base_[w|b]_lane_u32
    //   by callers (see wrga_fused_ptx.rs).
    //   This makes all ld.shared.b32 addresses 4-byte aligned because:
    //     (tid/4)*32 is 32-byte aligned, (tid%4)*4 is 4-byte aligned.
    ptx.push_str("    and.b32 %r2, %tid_x, 3;\n");
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
/// into SMEM (%x_tile_base).  Handles m-tail (rows >= m_param get zeroed) and
/// k-tail (k-positions >= k get zeroed).
///
/// x global layout: [m, k] row-major, **f32** (dtype=1, the NSL GPU tensor dtype).
/// SMEM stores packed f16x2 (required by m16n8k16 A-fragment).
/// Each pair of f32 values is loaded, converted to f16, and packed into a b32.
/// Thread 0 does all work.
///
/// NOTE: The m bound is taken from the **runtime** %m_param register (not the
/// compile-time config.m prescan placeholder, which is always 1).  Each row
/// emits a `setp.lt.u32 %p1, <row_const>, %m_param` before the global load to
/// guard against reading beyond the actual batch dimension.  k-tail bounds are
/// still compile-time constants since k is a fixed kernel dimension.
pub fn emit_lora_stage_x_tile(ptx: &mut String, k_tile: u32, _m: u32, k: u32) {
    ptx.push_str(&format!("    // Stage x_tile for K-tile {} (f32 global -> f16x2 SMEM, m-bound from %m_param)\n", k_tile));
    ptx.push_str("    setp.eq.u32 %p0, %tid_x, 0;\n");
    ptx.push_str(&format!("    @!%p0 bra lora_x_stage_done_{};\n", k_tile));
    let k_start = k_tile * 16;
    for row in 0..16u32 {
        // Emit runtime m-predicate for this row: %p1 = (row < %m_param).
        ptx.push_str(&format!("    setp.lt.u32 %p1, {}, %m_param;\n", row));
        for col_pair in 0..8u32 {
            let col = col_pair * 2;
            let smem_offset = row * 32 + col_pair * 4;
            // k-tail is a compile-time check (k is a fixed kernel dimension).
            if k_start + col >= k {
                // k-OOB: always zero, regardless of row.
                ptx.push_str(&format!(
                    "    st.shared.b32 [%x_tile_base + {}], 0;\n",
                    smem_offset
                ));
            } else {
                // f32 global layout: x[row, col] at byte (row * k + col) * 4.
                let gl_off0 = (row as u64) * (k as u64) * 4
                    + (k_start as u64 + col as u64) * 4;
                let col1_valid = k_start + col + 1 < k;
                let gl_off1 = gl_off0 + 4;
                // Predicated load: row < m_param → load from global; else zero.
                ptx.push_str(&format!(
                    "    @%p1 ld.global.f32 %stg_f0, [%rd_x + {}];\n",
                    gl_off0
                ));
                ptx.push_str("    @%p1 cvt.rn.f16.f32 %stg_h0, %stg_f0;\n");
                ptx.push_str("    @!%p1 mov.b16 %stg_h0, 0;\n");
                if col1_valid {
                    ptx.push_str(&format!(
                        "    @%p1 ld.global.f32 %stg_f1, [%rd_x + {}];\n",
                        gl_off1
                    ));
                    ptx.push_str("    @%p1 cvt.rn.f16.f32 %stg_h1, %stg_f1;\n");
                    ptx.push_str("    @!%p1 mov.b16 %stg_h1, 0;\n");
                } else {
                    ptx.push_str("    mov.b16 %stg_h1, 0;\n");
                }
                // Pack two f16s into b32 and store to SMEM.
                ptx.push_str(&format!(
                    "    mov.b32 %r3, {{%stg_h0, %stg_h1}};\n"
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
/// W global layout: [k, n] row-major, **f32** (NSL GPU tensor dtype=1).
///
/// SMEM layout: col-major so that B-fragment ld.shared.b32 loads are 4-byte aligned.
/// Element [k_row, col] in SMEM is at byte offset `col*32 + k_row*2`.
/// Column stride = 32 bytes (16 f16 k-rows × 2 bytes/f16).
///
/// Each f32 is loaded from global, converted to f16, and stored individually
/// to its col-major SMEM slot using b16 stores (always 2-byte aligned).
/// Thread 0 does all work.
pub fn emit_lora_stage_w_tile(ptx: &mut String, k_tile: u32, n: u32, k: u32) {
    ptx.push_str(&format!("    // Stage w_tile for K-tile {} (f32 global, col-major f16 SMEM)\n", k_tile));
    ptx.push_str("    setp.eq.u32 %p0, %tid_x, 0;\n");
    ptx.push_str(&format!("    @!%p0 bra lora_w_stage_done_{};\n", k_tile));
    let k_start = k_tile * 16;
    // Col-major SMEM: column c at bytes c*32 + k*2.
    const COL_STRIDE: u32 = 32;
    for col in 0..8u32 {
        for k_row in 0..16u32 {
            let smem_offset = col * COL_STRIDE + k_row * 2;
            if k_start + k_row >= k || col >= n {
                ptx.push_str(&format!(
                    "    st.shared.b16 [%w_tile_base + {}], 0;\n",
                    smem_offset
                ));
            } else {
                // f32 global: W[k_start+k_row, col] at byte (k_start+k_row)*n*4 + col*4
                let gl_offset =
                    ((k_start + k_row) as u64) * (n as u64) * 4 + (col as u64) * 4;
                ptx.push_str(&format!(
                    "    ld.global.f32 %stg_f0, [%rd_w + {}];\n",
                    gl_offset
                ));
                ptx.push_str("    cvt.rn.f16.f32 %stg_h0, %stg_f0;\n");
                ptx.push_str(&format!(
                    "    st.shared.b16 [%w_tile_base + {}], %stg_h0;\n",
                    smem_offset
                ));
            }
        }
    }
    ptx.push_str(&format!("lora_w_stage_done_{}:\n", k_tile));
}

/// Stage the a_tile slice for K-iteration `k_tile`.
/// A global layout: [k, rank] row-major, **f32** (NSL GPU tensor dtype=1).
///
/// SMEM layout: col-major so that A-as-B-fragment loads are 4-byte aligned.
/// The epilogue MMA uses a_tile as the B-operand (k=16 × n=rank_padded=8).
/// Col-major layout: element [k_row, col] at byte offset `col*32 + k_row*2`.
/// Column stride = 32 bytes (16 f16 k-rows × 2 bytes/f16).
///
/// Each f32 is loaded from global, converted to f16, and stored individually
/// to its col-major SMEM slot using b16 stores.  Thread 0 does all work.
pub fn emit_lora_stage_a_tile(ptx: &mut String, k_tile: u32, rank: u32, k: u32) {
    ptx.push_str(&format!("    // Stage a_tile for K-tile {} (rank={}, f32 global, col-major f16 SMEM)\n", k_tile, rank));
    ptx.push_str("    setp.eq.u32 %p0, %tid_x, 0;\n");
    ptx.push_str(&format!("    @!%p0 bra lora_a_stage_done_{};\n", k_tile));
    let k_start = k_tile * 16;
    // Col-major: col c at bytes c*32 + k_row*2.
    const COL_STRIDE: u32 = 32;
    for col in 0..8u32 {
        for k_row in 0..16u32 {
            let smem_offset = col * COL_STRIDE + k_row * 2;
            if k_start + k_row >= k || col >= rank {
                ptx.push_str(&format!(
                    "    st.shared.b16 [%a_tile_base + {}], 0;\n",
                    smem_offset
                ));
            } else {
                // f32 global: A[k_start+k_row, col] at byte (k_start+k_row)*rank*4 + col*4
                let gl_offset = ((k_start + k_row) as u64) * (rank as u64) * 4 + (col as u64) * 4;
                ptx.push_str(&format!(
                    "    ld.global.f32 %stg_f0, [%rd_a + {}];\n",
                    gl_offset
                ));
                ptx.push_str("    cvt.rn.f16.f32 %stg_h0, %stg_f0;\n");
                ptx.push_str(&format!(
                    "    st.shared.b16 [%a_tile_base + {}], %stg_h0;\n",
                    smem_offset
                ));
            }
        }
    }
    ptx.push_str(&format!("lora_a_stage_done_{}:\n", k_tile));
}

/// Stage the b_tile ONCE post-loop.  Shape: 16×8 f16 (rank-padded rows = k-dim).
/// B global layout: [rank, n] row-major, **f32** (NSL GPU tensor dtype=1).
///
/// SMEM layout: col-major so that B-fragment ld.shared.b32 loads are 4-byte aligned.
/// Element [k_row, col] is at SMEM byte offset `col*32 + k_row*2`.
/// Column stride = 32 bytes (16 f16 k-rows × 2 bytes/f16).
///
/// Each f32 is loaded from global, converted to f16, and stored individually
/// to its col-major SMEM slot using b16 stores.  Thread 0 does all work.
pub fn emit_lora_stage_b_tile(ptx: &mut String, rank: u32, n: u32) {
    ptx.push_str("    // Stage b_tile (post-loop, once, f32 global, col-major f16 SMEM)\n");
    ptx.push_str("    setp.eq.u32 %p0, %tid_x, 0;\n");
    ptx.push_str("    @!%p0 bra lora_b_stage_done;\n");
    const COL_STRIDE: u32 = 32;
    for col in 0..8u32 {
        for k_row in 0..16u32 {
            let smem_offset = col * COL_STRIDE + k_row * 2;
            if k_row >= rank || col >= n {
                ptx.push_str(&format!(
                    "    st.shared.b16 [%b_tile_base + {}], 0;\n",
                    smem_offset
                ));
            } else {
                // f32 global: B[k_row, col] at byte k_row*n*4 + col*4
                let gl_offset = (k_row as u64) * (n as u64) * 4 + (col as u64) * 4;
                ptx.push_str(&format!(
                    "    ld.global.f32 %stg_f0, [%rd_b + {}];\n",
                    gl_offset
                ));
                ptx.push_str("    cvt.rn.f16.f32 %stg_h0, %stg_f0;\n");
                ptx.push_str(&format!(
                    "    st.shared.b16 [%b_tile_base + {}], %stg_h0;\n",
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
///
/// Shared by both LoRA and IA³ — both use the identical m16n8k16 D-fragment
/// layout and the same main_accum<8> pool.
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

// ─── E2: IA³-specific helpers ─────────────────────────────────────────────
//
// IA³ reuses the following LoRA helpers unchanged:
//   emit_lora_stage_x_tile      — f32 global → f16x2 SMEM, runtime m-predicate
//   emit_lora_stage_w_tile      — f32 global → col-major f16 SMEM
//   emit_matmul_mma_lane_init   — %mma_a_row / %mma_b_row init
//   emit_lora_store_output      — predicated D-fragment store to y
//
// IA³-specific helpers added here:
//   Ia3RegisterBudget           — budget struct (no epilogue regs, 2-reg gamma)
//   wrga_ia3_register_budget    — fills the budget
//   emit_ia3_register_pool      — emits .reg declarations
//   emit_ia3_tile_bases         — sets x/w SMEM base registers (no a/b tiles)
//   emit_ia3_load_gamma         — per-thread γ load (2 f32 values per thread)
//   emit_ia3_gamma_multiply     — broadcast-mul main_accum0..3 by gamma0/gamma1
//   emit_ia3_store_output       — thin wrapper delegating to emit_lora_store_output

/// Register pool spec for WRGA fused IA³ kernel.
///
/// Compared to `LoraRegisterBudget`:
///   - No epi_interm/epi_final/epi_a_frag/epi_b_frag (no epilogue MMA)
///   - No scale_reg (IA³ has no per-site scale)
///   - No rd_a/rd_b/smem_base_a/smem_base_b (no A/B adapter matrices)
///   - gamma_count = 2 (per-thread 2-col load; avoids PTX dynamic reg indexing)
#[derive(Debug, Clone)]
pub struct Ia3RegisterBudget {
    pub main_accum_count: u32,  // 8 (f32, x@W output accumulator)
    pub gamma_count: u32,       // 2 (f32: gamma0 = γ[col+0], gamma1 = γ[col+1])
    pub main_a_frag_count: u32, // 4 (b32, x fragment)
    pub main_b_frag_count: u32, // 2 (b32, W fragment)
    pub rd_scratch: u32,        // 12 (u64 scratch)
    pub u32_scratch: u32,       // 10 (u32 scratch, lane math, tile math)
    pub pred_count: u32,        // 4  (%p0..%p3)
}

pub fn wrga_ia3_register_budget(_cfg: &FusedIa3Config) -> Ia3RegisterBudget {
    Ia3RegisterBudget {
        main_accum_count: 8,
        gamma_count: 2,
        main_a_frag_count: 4,
        main_b_frag_count: 2,
        rd_scratch: 12,
        u32_scratch: 10,
        pred_count: 4,
    }
}

/// Emit the WRGA-IA³ register pool declarations.
///
/// Matches the LoRA pool shape but drops epilogue registers and the scale reg.
/// Keeps all lane-addressing registers (%mma_a_row, %mma_b_row, %mma_addr,
/// %smem_base_x/w, %smem_base_w_lane_u32) because the matmul_mma helpers
/// require them by name.
///
/// %gamma<2> holds the 2 per-thread γ column values loaded by
/// emit_ia3_load_gamma.
pub fn emit_ia3_register_pool(ptx: &mut String, b: &Ia3RegisterBudget) {
    ptx.push_str(&format!("    .reg .f32 %main_accum<{}>;\n", b.main_accum_count));
    ptx.push_str(&format!("    .reg .f32 %gamma<{}>;\n", b.gamma_count));
    ptx.push_str(&format!("    .reg .b32 %main_a_frag<{}>;\n", b.main_a_frag_count));
    ptx.push_str(&format!("    .reg .b32 %main_b_frag<{}>;\n", b.main_b_frag_count));
    ptx.push_str(&format!("    .reg .u64 %rd<{}>;\n", b.rd_scratch));
    ptx.push_str(&format!("    .reg .u32 %r<{}>;\n", b.u32_scratch));
    ptx.push_str(&format!("    .reg .pred %p<{}>;\n", b.pred_count));
    // Named pointer/base registers (subset of LoRA — no rd_a/rd_b).
    ptx.push_str("    .reg .u64 %rd_x, %rd_w, %rd_gamma, %rd_y;\n");
    ptx.push_str("    .reg .u64 %x_tile_base, %w_tile_base;\n");
    ptx.push_str("    .reg .u64 %shmem_base;\n");
    ptx.push_str("    .reg .u32 %tid_x, %warp_id, %lane, %bid_x, %bid_y;\n");
    // Lane-derived addressing registers (required by matmul_mma helpers).
    ptx.push_str("    .reg .u32 %mma_a_row, %mma_b_row, %mma_addr;\n");
    ptx.push_str("    .reg .u64 %smem_base_x, %smem_base_w;\n");
    // Output tile coords.
    ptx.push_str("    .reg .u32 %row_base, %col_base;\n");
    ptx.push_str("    .reg .u32 %m_real, %n_real;\n");
}

/// SMEM byte offsets for the IA³ kernel.
///
/// IA³ only needs x_tile and w_tile (no adapter A/B tiles).
/// Sizes and offsets are identical to the LoRA x/w sub-allocation
/// so the same staging helpers work without modification:
///   [  0,  512) — x_tile  (16×16 f16, 512 bytes)
///   [512,  768) — w_tile  (16× 8 f16, 256 bytes, col-major)
pub const IA3_X_TILE_OFFSET: u32 = 0;
pub const IA3_W_TILE_OFFSET: u32 = 512;

/// Initialize the x/w SMEM base registers for the IA³ kernel.
///
/// Must be called after `emit_shmem_base_cvta` has set %shmem_base.
/// Sets %x_tile_base, %w_tile_base, and their %smem_base_x/w aliases
/// (required by emit_lora_stage_x_tile and emit_lora_stage_w_tile).
///
/// Also sets the per-lane B-fragment column base %smem_base_w_lane_u32
/// needed by emit_load_b_fragment_smem.  The col-major B-tile lane
/// formula is identical to LoRA: (tid/4)*32 bytes.
pub fn emit_ia3_tile_bases(ptx: &mut String) {
    ptx.push_str(&format!("    add.u64 %x_tile_base, %shmem_base, {};\n", IA3_X_TILE_OFFSET));
    ptx.push_str(&format!("    add.u64 %w_tile_base, %shmem_base, {};\n", IA3_W_TILE_OFFSET));
    // matmul_mma helpers look up %smem_base_x/w by name — alias.
    ptx.push_str("    mov.u64 %smem_base_x, %x_tile_base;\n");
    ptx.push_str("    mov.u64 %smem_base_w, %w_tile_base;\n");
}

/// Load the 2 per-thread γ values from global memory into %gamma0 and %gamma1.
///
/// Per-thread column layout (m16n8k16 D-fragment):
///   col_base_in_tile = (tid % 4) * 2   → 0, 2, 4, or 6
/// This thread owns output cols (col_base + col_base_in_tile + 0) and
/// (col_base + col_base_in_tile + 1).  Loading exactly 2 f32 values per
/// thread avoids PTX's lack of dynamic register indexing (option c from spec).
///
/// γ is a 1-D f32 array of length n.  The byte offset for this thread is:
///   (col_base + col_base_in_tile) * 4
///
/// Stores:
///   %gamma0 = γ[col_base + col_base_in_tile + 0]
///   %gamma1 = γ[col_base + col_base_in_tile + 1]
///
/// NOTE: n-bound predication is intentionally omitted here (emit_ia3_store_output
/// already predicates on n_real per column); γ OOB loads on tail columns only
/// produce garbage values that are never written to y.
pub fn emit_ia3_load_gamma(ptx: &mut String) {
    ptx.push_str("    // Load per-thread γ slice (2 cols per thread)\n");
    // col_base_in_tile = (tid % 4) * 2
    ptx.push_str("    and.b32 %r5, %tid_x, 3;\n");
    ptx.push_str("    shl.b32 %r5, %r5, 1;\n");
    // Global gamma byte offset: (col_base + col_base_in_tile) * 4
    ptx.push_str("    add.u32 %r6, %col_base, %r5;\n");
    ptx.push_str("    shl.b32 %r6, %r6, 2;\n");
    ptx.push_str("    cvt.u64.u32 %rd0, %r6;\n");
    ptx.push_str("    add.u64 %rd0, %rd_gamma, %rd0;\n");
    // Load 2 γ values: gamma0 at col+0, gamma1 at col+1.
    ptx.push_str("    ld.global.f32 %gamma0, [%rd0];\n");
    ptx.push_str("    ld.global.f32 %gamma1, [%rd0 + 4];\n");
}

/// Broadcast-multiply main_accum0..3 by the per-thread γ values.
///
/// Per m16n8k16 D-fragment layout each thread owns 4 output values:
///   main_accum0: (row+0, col+0) → multiply by gamma0
///   main_accum1: (row+0, col+1) → multiply by gamma1
///   main_accum2: (row+8, col+0) → multiply by gamma0
///   main_accum3: (row+8, col+1) → multiply by gamma1
///
/// main_accum4..7 are the corresponding values from the second half of the
/// m16n8 output tile (col offset +4 and +5 for the same two-column pair).
/// They use the same gamma0/gamma1 because γ is indexed by the OUTPUT column
/// of the m16n8k16 result, and for a given thread the 4-accum and 8-accum
/// halves differ only in their ROW, not their column.
///
/// Must be called AFTER emit_ia3_load_gamma (uses %gamma0/%gamma1) and
/// BEFORE emit_ia3_store_output.
pub fn emit_ia3_gamma_multiply(ptx: &mut String) {
    ptx.push_str("    // Broadcast-multiply main_accum by γ (per-thread 2-col)\n");
    // First D-fragment half (rows 0 and 8 of the tile):
    ptx.push_str("    mul.f32 %main_accum0, %main_accum0, %gamma0;\n");
    ptx.push_str("    mul.f32 %main_accum1, %main_accum1, %gamma1;\n");
    ptx.push_str("    mul.f32 %main_accum2, %main_accum2, %gamma0;\n");
    ptx.push_str("    mul.f32 %main_accum3, %main_accum3, %gamma1;\n");
    // Second D-fragment half (same column pair, rows 0+8 of upper sub-tile).
    // m16n8k16 has only 4 D-fragment registers per thread for the n=8 output;
    // main_accum4..7 are the second m16n8 tile's results (from the second
    // half of main_accum<8> used when two MMA tiles cover the full 16×8 block).
    // They share the same gamma0/gamma1 because their output columns are
    // col+0 and col+1 relative to this thread's col_base_in_tile.
    ptx.push_str("    mul.f32 %main_accum4, %main_accum4, %gamma0;\n");
    ptx.push_str("    mul.f32 %main_accum5, %main_accum5, %gamma1;\n");
    ptx.push_str("    mul.f32 %main_accum6, %main_accum6, %gamma0;\n");
    ptx.push_str("    mul.f32 %main_accum7, %main_accum7, %gamma1;\n");
}

/// Store the IA³ output tile to y — delegates to the shared LoRA helper.
///
/// Both LoRA and IA³ use the identical m16n8k16 D-fragment layout and the
/// same main_accum<8> pool, so the store sequence is bit-for-bit identical.
pub fn emit_ia3_store_output(ptx: &mut String, n: u32) {
    emit_lora_store_output(ptx, n);
}

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
/// kernel's register pool.  For GatedLoRA, this is declared in the
/// `emit_fused_adapter_kernel_body`'s PerColumnSigmoid register block
/// (added in Task 4.1).
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

/// Emit a per-thread gate load for the GatedLoRA fused kernel.
///
/// Each thread owns exactly 2 output columns in the m16n8k16 D-fragment
/// (cols `col_base_in_tile + 0` and `+1`, where `col_base_in_tile = (tid%4)*2`).
/// This helper emits the address arithmetic + 2 `ld.global.f32` calls to
/// load those 2 gate values into `%gate0` and `%gate1`.
///
/// Requires (caller must have declared / initialized before this call):
///   %rd_gate      — loaded from .param .u64 gate_ptr at prolog
///   %col_base     — output-tile column base (from emit_lora_output_tile_coords_dynamic)
///   %r5           — col_base_in_tile = (tid%4)*2 (computed by emit_matmul_mma_lane_init OR
///                    by the store preamble; caller must ensure a setter has run first)
///   %r_gate       — declared .reg .u32
///   %rd_gate_addr — declared .reg .u64
///   %gate0, %gate1 — declared .reg .f32
///
/// Warp-broadcast alternative explicitly avoided: would 4× HBM gate
/// traffic for no benefit since each thread consumes only 2 of 8 columns.
/// See invariant #13 in project_wrga_fused_ptx_rewrite.md.
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

        // Also reject the sign-flipped case (would silently compute sigmoid(-x)).
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

    #[test]
    fn emit_gate_load_per_thread_emits_two_f32_loads_with_correct_stride() {
        use super::emit_gate_load_per_thread;
        let mut ptx = String::new();
        emit_gate_load_per_thread(&mut ptx);

        // Exactly 2 ld.global.f32 — one per gate value.
        assert_eq!(
            ptx.matches("ld.global.f32").count(),
            2,
            "expected 2 per-thread gate loads; got:\n{ptx}"
        );

        // Byte-stride must be 4 (f32); catches gate-dtype regression.
        assert!(
            ptx.contains("shl.b32  %r_gate, %r_gate, 2"),
            "gate offset must shift-left by 2 (f32 stride); got:\n{ptx}"
        );

        // Second load at offset +4 (second gate value).
        assert!(
            ptx.contains("ld.global.f32 %gate1, [%rd_gate_addr + 4]"),
            "second gate load missing or wrong offset:\n{ptx}"
        );

        // Base pointer summation.
        assert!(
            ptx.contains("add.u64  %rd_gate_addr, %rd_gate, %rd_gate_addr"),
            "gate address must sum base + offset:\n{ptx}"
        );
    }
}
