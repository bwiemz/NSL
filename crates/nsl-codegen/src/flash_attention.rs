// PTX generation uses format!() extensively, including for strings with no interpolation
// (for consistency and readability in the emit_* helpers). Suppress clippy's advice
// to rewrite these as .to_string().
#![allow(clippy::useless_format)]

//! FlashAttention-2 PTX template synthesis.
//!
//! Generates PTX kernel strings at compile time (AOT). Each variant is parameterized
//! by orthogonal flags (paged, rope_q, rope_style, gqa_group_size, causal) and tile
//! sizes (block_q, block_kv). The generated PTX is embedded in .rodata and launched
//! by the runtime wrappers in nsl-runtime/src/flash_attention.rs.

// ---------------------------------------------------------------------------
// MMA (Tensor Core) constants for m16n8k16 on sm_80+
// ---------------------------------------------------------------------------
//
// ## Architecture Overview
//
// FlashAttention uses two matmuls per KV-tile iteration:
//   1. S = Q @ K^T  (score computation)
//   2. O += P @ V   (output accumulation, where P = softmax(S))
//
// Both are implemented using `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`:
//   - A-fragment: 16x16 row-major f16 (Q for S, P for O)
//   - B-fragment: 8x16 col-major f16  (K^T for S, V for O)
//   - C/D accumulator: 16x8 f32
//
// ## Thread-to-Element Mapping (m16n8k16)
//
// Each warp (32 threads) cooperatively computes one 16x8 output tile.
// Thread t holds 4 f32 accumulator values at positions:
//   row = (t % 4) * 2 + (t / 16)       for registers 0, 1
//   row = (t % 4) * 2 + (t / 16) + 8   for registers 2, 3
//   col depends on the register index and n-tile offset
//
// ## Register Pressure Management
//
// Full unrolling of all m-tiles x n-tiles exceeds the 255-register limit.
// Strategy: process one m-tile at a time, immediately feeding S into softmax
// and P@V before advancing. This keeps pressure at O(n_tiles * 4) per phase.
//
// ## Shared Memory
//
// Q and K/V tiles are stored in shared memory as f32 (matching the existing
// global-to-shared load path). Fragment loads convert f32 -> f16 on the fly
// via `cvt.rn.f16.f32`. XOR swizzle avoids bank conflicts.
//
// ## Fallback
//
// GPUs below sm_80 use the existing scalar fma.rn.f32 path. Gate via
// `use_mma_path(gpu_sm)`.

/// MMA tile dimensions for mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
const MMA_M: usize = 16;
const MMA_N: usize = 8;
const MMA_K: usize = 16;

/// Minimum SM version required for f16 MMA tensor core instructions.
const MMA_MIN_SM: u32 = 80;

/// Check whether the MMA path should be used for this GPU.
pub fn use_mma_path(gpu_sm: u32) -> bool {
    gpu_sm >= MMA_MIN_SM
}

/// Validate that FlashAttention tile sizes are compatible with MMA fragment dimensions.
/// Returns Ok(()) if valid, Err with a message describing the constraint violation.
pub fn validate_mma_tile_sizes(
    block_q: usize,
    block_kv: usize,
    head_dim: usize,
) -> Result<(), String> {
    if !block_q.is_multiple_of(MMA_M) {
        return Err(format!(
            "block_q ({}) must be a multiple of MMA_M ({})",
            block_q, MMA_M
        ));
    }
    if !block_kv.is_multiple_of(MMA_N) {
        return Err(format!(
            "block_kv ({}) must be a multiple of MMA_N ({})",
            block_kv, MMA_N
        ));
    }
    if !head_dim.is_multiple_of(MMA_K) {
        return Err(format!(
            "head_dim ({}) must be a multiple of MMA_K ({})",
            head_dim, MMA_K
        ));
    }
    Ok(())
}

/// RoPE interleaving style.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RopeStyle {
    /// (x[i], x[i + head_dim/2]) — LLaMA, Qwen, Mistral
    HalfSplit,
    /// (x[2i], x[2i+1]) — GPT-NeoX, GPT-J
    Adjacent,
}

/// Configuration for a FlashAttention PTX kernel variant.
#[derive(Clone, Debug)]
pub struct FlashAttentionConfig {
    pub block_q: i64,
    pub block_kv: i64,
    pub head_dim: i64,
    pub causal: bool,
    pub paged: bool,
    pub rope_q: bool,
    pub rope_style: RopeStyle,
    pub gqa_group_size: u32,
    /// M33: Whether this attention uses a tree-structured causal mask for speculative decoding.
    pub tree_mask: bool,
    /// Target GPU SM version for PTX target selection (default: 52).
    pub gpu_sm: u32,
}

/// Generate PTX for the FlashAttention-2 kernel with the given configuration.
///
/// Returns null-terminated PTX bytes ready for .rodata embedding.
pub fn synthesize_flash_attention_ptx(config: &FlashAttentionConfig) -> Vec<u8> {
    let mut ptx = String::with_capacity(8192);
    let kernel_name = flash_attention_kernel_name(config);

    // PTX header (dynamic target based on GPU)
    emit_ptx_header(&mut ptx, config.gpu_sm);

    // Kernel entry point
    emit_flash_attention_entry(&mut ptx, &kernel_name, config);

    // Null-terminate
    ptx.push('\0');
    ptx.into_bytes()
}

/// Generate PTX for the rope_cache_write elementwise kernel.
///
/// Returns null-terminated PTX bytes.
pub fn synthesize_rope_cache_write_ptx(
    head_dim: i64,
    rope_style: RopeStyle,
) -> Vec<u8> {
    let mut ptx = String::with_capacity(4096);

    emit_ptx_header(&mut ptx, 52); // RoPE cache write targets sm_52 minimum
    emit_rope_cache_write_entry(&mut ptx, head_dim, rope_style);

    ptx.push('\0');
    ptx.into_bytes()
}

/// Compute the kernel name encoding variant flags and tile sizes.
///
/// Format: `flash_attn_p{paged}_r{rope}_{style}_g{gqa}_c{causal}_t{tree}_q{block_q}_kv{block_kv}`
pub fn flash_attention_kernel_name(config: &FlashAttentionConfig) -> String {
    format!(
        "flash_attn_p{}_r{}_{}_g{}_c{}_t{}_q{}_kv{}",
        config.paged as u8,
        config.rope_q as u8,
        match config.rope_style {
            RopeStyle::HalfSplit => "hs",
            RopeStyle::Adjacent => "adj",
        },
        config.gqa_group_size,
        config.causal as u8,
        config.tree_mask as u8,
        config.block_q,
        config.block_kv,
    )
}

/// Compute shared memory bytes for a given config.
///
/// Formula: (block_q + block_kv) * head_dim * sizeof(f16)
/// where sizeof(f16) = 2.
pub fn shared_mem_bytes(config: &FlashAttentionConfig) -> u32 {
    ((config.block_q + config.block_kv) * config.head_dim * 2) as u32
}

// ── PTX emission helpers ──────────────────────────────────────────

fn emit_ptx_header(ptx: &mut String, gpu_sm: u32) {
    let version = if gpu_sm >= 90 { "8.0" } else { "7.0" };
    let target = if gpu_sm >= 90 { "sm_90" }
        else if gpu_sm >= 80 { "sm_80" }
        else { "sm_52" };
    ptx.push_str(&format!(".version {version}\n"));
    ptx.push_str(&format!(".target {target}\n"));
    ptx.push_str(".address_size 64\n\n");
}

fn emit_flash_attention_entry(
    ptx: &mut String,
    kernel_name: &str,
    config: &FlashAttentionConfig,
) {
    // Parameter declarations — ALWAYS declare ALL params regardless of variant flags.
    // The runtime wrapper always passes the full 17-arg set (unused params are null/zero).
    // Conditional params are simply ignored in the kernel body when the flag is off.
    // This ensures cuLaunchKernel arg alignment is always correct.
    ptx.push_str(&format!(".visible .entry {} (\n", kernel_name));
    ptx.push_str("    .param .u64 q_ptr,\n");
    ptx.push_str("    .param .u64 k_ptr,\n");
    ptx.push_str("    .param .u64 v_ptr,\n");
    ptx.push_str("    .param .u64 out_ptr,\n");
    ptx.push_str("    .param .f32 scale,\n");
    ptx.push_str("    .param .u64 batch,\n");
    ptx.push_str("    .param .u64 heads,\n");
    ptx.push_str("    .param .u64 seq_len,\n");
    ptx.push_str("    .param .u64 head_dim,\n");
    // Paged KV params (null/zero when paged=false, but always declared)
    ptx.push_str("    .param .u64 block_table_ptr,\n");
    ptx.push_str("    .param .u64 k_pool_ptr,\n");
    ptx.push_str("    .param .u64 v_pool_ptr,\n");
    ptx.push_str("    .param .u64 block_size,\n");
    // RoPE params (null when rope_q=false, but always declared)
    ptx.push_str("    .param .u64 cos_ptr,\n");
    ptx.push_str("    .param .u64 sin_ptr,\n");
    // M29-ready ragged batch params
    ptx.push_str("    .param .u64 seq_ids_ptr,\n");
    ptx.push_str("    .param .u64 seq_lens_ptr,\n");
    // M33: Tree attention mask params (null/zero when tree_mask=false)
    ptx.push_str("    .param .u64 dfs_enter_ptr,\n");
    ptx.push_str("    .param .u64 dfs_exit_ptr,\n");
    ptx.push_str("    .param .u64 num_tree_nodes\n");

    ptx.push_str(")\n");
    ptx.push_str("{\n");

    // Shared memory declaration (must be inside kernel body for ptxas)
    let shmem_bytes = shared_mem_bytes(config);
    ptx.push_str(&format!(
        "    .shared .align 16 .b8 shmem[{}];\n",
        shmem_bytes
    ));

    // Register declarations
    emit_register_declarations(ptx, config);

    // Load parameters
    emit_param_loads(ptx, config);

    // Compute thread/block indices
    emit_index_computation(ptx, config);

    // Load Q tile into shared memory
    emit_q_tile_load(ptx, config);

    // Initialize accumulators (O_acc=0, row_max=-inf, row_sum=0)
    emit_accumulator_init(ptx, config);

    // Main K/V tile loop with online softmax
    emit_kv_tile_loop(ptx, config);

    // Finalize: O = O_acc / row_sum
    emit_finalize(ptx, config);

    // Store output tile to global memory
    emit_output_store(ptx, config);

    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");
}

fn emit_register_declarations(ptx: &mut String, config: &FlashAttentionConfig) {
    // Thread indexing registers
    ptx.push_str("    .reg .u32 %tid_x, %bid_x, %bid_y;\n");
    ptx.push_str("    .reg .u64 %rd<64>;\n");
    ptx.push_str("    .reg .f32 %f<128>;\n");
    ptx.push_str("    .reg .b16 %h<32>;\n");  // f16 registers for output conversion
    ptx.push_str("    .reg .pred %p<16>;\n");
    ptx.push_str("    .reg .u32 %r<32>;\n");

    // Scale register
    ptx.push_str("    .reg .f32 %scale;\n");

    // LOG2E constant for exp() via ex2.approx
    ptx.push_str("    .reg .f32 %log2e;\n");
    ptx.push_str("    mov.f32 %log2e, 0x3FB8AA3B;  // 1.4426950408 (log2(e))\n");

    // Loop counter for K/V tile iteration
    ptx.push_str("    .reg .u64 %k_start, %k_max;\n");

    // Accumulator registers for online softmax
    // row_max, row_sum, correction per Q row handled by this thread
    ptx.push_str("    .reg .f32 %row_max, %row_sum, %correction;\n");
    ptx.push_str("    .reg .f32 %new_max, %old_max;\n");

    // Warp reduction temporaries (shfl.sync.bfly)
    ptx.push_str("    .reg .f32 %shfl_tmp;\n");

    if config.rope_q {
        ptx.push_str("    .reg .f32 %cos_val, %sin_val;\n");
        ptx.push_str("    .reg .f32 %q_a, %q_b, %q_rot_a, %q_rot_b;\n");
    }

    if config.tree_mask {
        // M33: DFS enter/exit timestamps for O(1) ancestor checks
        ptx.push_str("    .reg .u64 %dfs_enter_base, %dfs_exit_base, %num_tree_nodes;\n");
        ptx.push_str("    .reg .u32 %dfs_q_enter, %dfs_q_exit, %dfs_k_enter, %dfs_k_exit;\n");
        ptx.push_str("    .reg .pred %p_ancestor;\n");
    }

    // O_acc registers: each thread accumulates a subset of the [block_q, head_dim] output tile
    // Total O_acc regs per thread = (block_q * head_dim) / blockDim.x
    // Example: block_q=64, head_dim=128, blockDim.x=128 → 64 registers
    // Declared dynamically based on config in emit_accumulator_init
    let _ = config;
}

fn emit_param_loads(ptx: &mut String, config: &FlashAttentionConfig) {
    // Always load ALL params — PTX entry declares them all regardless of variant.
    // Unused params (null/zero) are simply never referenced in the kernel body.
    ptx.push_str("    ld.param.u64 %rd0, [q_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd1, [k_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd2, [v_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd3, [out_ptr];\n");
    ptx.push_str("    ld.param.f32 %scale, [scale];\n");
    ptx.push_str("    ld.param.u64 %rd4, [batch];\n");
    ptx.push_str("    ld.param.u64 %rd5, [heads];\n");
    ptx.push_str("    ld.param.u64 %rd6, [seq_len];\n");
    ptx.push_str("    ld.param.u64 %rd7, [head_dim];\n");
    // Paged params (always loaded; only used when paged=true)
    ptx.push_str("    ld.param.u64 %rd8, [block_table_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd9, [k_pool_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd10, [v_pool_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd11, [block_size];\n");
    // RoPE params (always loaded; only used when rope_q=true)
    ptx.push_str("    ld.param.u64 %rd12, [cos_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd13, [sin_ptr];\n");
    // Ragged batch params
    ptx.push_str("    ld.param.u64 %rd14, [seq_ids_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd15, [seq_lens_ptr];\n");

    // M33: Tree mask params (always loaded; only used when tree_mask=true)
    if config.tree_mask {
        ptx.push_str("    ld.param.u64 %dfs_enter_base, [dfs_enter_ptr];\n");
        ptx.push_str("    ld.param.u64 %dfs_exit_base, [dfs_exit_ptr];\n");
        ptx.push_str("    ld.param.u64 %num_tree_nodes, [num_tree_nodes];\n");
    }
}

fn emit_index_computation(ptx: &mut String, config: &FlashAttentionConfig) {
    // threadIdx.x, blockIdx.x (Q tile index), blockIdx.y (batch*head index)
    ptx.push_str("    mov.u32 %tid_x, %tid.x;\n");
    ptx.push_str("    mov.u32 %bid_x, %ctaid.x;\n");
    ptx.push_str("    mov.u32 %bid_y, %ctaid.y;\n");

    // q_start = blockIdx.x * block_q
    ptx.push_str("    // q_start = blockIdx.x * block_q\n");
    ptx.push_str("    cvt.u64.u32 %rd16, %bid_x;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd16, %rd16, {};  // %rd16 = q_start\n",
        config.block_q
    ));

    // batch_head routing computed from blockIdx.y
    ptx.push_str("    // batch_head routing computed from blockIdx.y\n");
    ptx.push_str("    cvt.u64.u32 %rd17, %bid_y;\n");
    ptx.push_str("    rem.u64 %rd18, %rd17, %rd5;  // head_idx = bid_y % heads\n");
    ptx.push_str("    div.u64 %rd19, %rd17, %rd5;  // batch_idx = bid_y / heads\n");

    if config.gqa_group_size > 1 {
        ptx.push_str(&format!(
            "    // GQA: kv_head = q_head / {} (compile-time literal)\n",
            config.gqa_group_size
        ));
        ptx.push_str(&format!(
            "    div.u64 %rd20, %rd18, {};  // kv_head = head_idx / gqa_group_size\n",
            config.gqa_group_size
        ));
    }
}

fn emit_q_tile_load(ptx: &mut String, config: &FlashAttentionConfig) {
    ptx.push_str("    // Load Q tile into shared memory\n");

    // Compute Q base address: q_base = q_ptr + (batch_idx * heads * seq_len * head_dim
    //   + head_idx * seq_len * head_dim + q_start * head_dim) * 4
    ptx.push_str("    // Compute Q base address for this batch/head/q_start\n");
    ptx.push_str("    mul.lo.u64 %rd21, %rd19, %rd5;   // batch_idx * heads\n");
    ptx.push_str("    add.u64 %rd21, %rd21, %rd18;      // + head_idx\n");
    ptx.push_str("    mul.lo.u64 %rd21, %rd21, %rd6;    // * seq_len\n");
    ptx.push_str("    add.u64 %rd21, %rd21, %rd16;      // + q_start\n");
    ptx.push_str("    mul.lo.u64 %rd21, %rd21, %rd7;    // * head_dim\n");
    ptx.push_str("    shl.b64 %rd21, %rd21, 2;          // * 4 (sizeof f32)\n");
    ptx.push_str("    add.u64 %rd21, %rd0, %rd21;       // q_base = q_ptr + offset\n");

    // Each thread loads (block_q * head_dim) / 128 elements cooperatively
    let total_q_elems = config.block_q * config.head_dim;
    let elems_per_thread = total_q_elems / 128;

    ptx.push_str("    // Cooperative Q load: each thread loads ");
    ptx.push_str(&format!("{} elements\n", elems_per_thread));
    ptx.push_str("    cvt.u64.u32 %rd22, %tid_x;        // elem_idx = tid_x\n");
    ptx.push_str(&format!(
        "    mov.u64 %rd23, {};                // total Q elements\n",
        total_q_elems
    ));

    if config.rope_q {
        ptx.push_str("    // RoPE: load Q from global, rotate in registers, store to SRAM\n");
        ptx.push_str("    // cos/sin loaded from global memory into registers (NOT SRAM)\n");
        let (stride_comment, stride_val) = match config.rope_style {
            RopeStyle::HalfSplit => (
                "stride = head_dim/2 (half_split)".to_string(),
                config.head_dim / 2,
            ),
            RopeStyle::Adjacent => ("stride = 1 (adjacent)".to_string(), 1),
        };
        ptx.push_str(&format!("    // RoPE style: {}\n", stride_comment));

        // Compute cos/sin base address for q_start position
        ptx.push_str("    // cos/sin base = cos_ptr + q_start * head_dim * 4\n");
        ptx.push_str("    mul.lo.u64 %rd24, %rd16, %rd7;  // q_start * head_dim\n");
        ptx.push_str("    shl.b64 %rd24, %rd24, 2;        // * 4 bytes\n");
        ptx.push_str("    add.u64 %rd25, %rd12, %rd24;    // cos_base\n");
        ptx.push_str("    add.u64 %rd26, %rd13, %rd24;    // sin_base\n");

        ptx.push_str("LOOP_Q_LOAD_ROPE:\n");
        // Load Q element pair
        ptx.push_str("    // Compute paired dimension index for RoPE rotation\n");
        ptx.push_str(&format!(
            "    rem.u64 %rd27, %rd22, {};          // d = elem_idx % head_dim\n",
            config.head_dim
        ));
        ptx.push_str(&format!(
            "    div.u64 %rd28, %rd22, {};          // row = elem_idx / head_dim\n",
            config.head_dim
        ));

        // Compute pair offset based on rope style
        ptx.push_str(&format!(
            "    // Paired offset: stride = {}\n",
            stride_val
        ));
        // offset_a = elem_idx, offset_b = elem_idx + stride (or elem_idx ^ 1 for adjacent)
        ptx.push_str("    shl.b64 %rd29, %rd22, 2;        // global byte offset\n");
        ptx.push_str("    add.u64 %rd30, %rd21, %rd29;    // q_addr = q_base + offset\n");
        ptx.push_str(&format!(
            "    add.u64 %rd31, %rd30, {};          // q_addr + stride_bytes\n",
            stride_val * 4
        ));

        // Load Q pair
        ptx.push_str("    ld.global.f32 %q_a, [%rd30];\n");
        ptx.push_str("    ld.global.f32 %q_b, [%rd31];\n");

        // Load cos/sin for this position/dimension
        ptx.push_str("    // cos/sin for this row's position and dimension\n");
        ptx.push_str("    mul.lo.u64 %rd32, %rd28, %rd7;  // row * head_dim\n");
        ptx.push_str("    add.u64 %rd32, %rd32, %rd27;    // + d\n");
        ptx.push_str("    shl.b64 %rd32, %rd32, 2;        // * 4 bytes\n");
        ptx.push_str("    add.u64 %rd33, %rd25, %rd32;    // cos_addr\n");
        ptx.push_str("    add.u64 %rd34, %rd26, %rd32;    // sin_addr\n");
        ptx.push_str("    ld.global.f32 %cos_val, [%rd33];\n");
        ptx.push_str("    ld.global.f32 %sin_val, [%rd34];\n");

        // Apply RoPE rotation
        ptx.push_str("    mul.f32 %q_rot_a, %q_a, %cos_val;\n");
        ptx.push_str("    mul.f32 %f0, %q_b, %sin_val;\n");
        ptx.push_str("    sub.f32 %q_rot_a, %q_rot_a, %f0;\n");
        ptx.push_str("    mul.f32 %q_rot_b, %q_a, %sin_val;\n");
        ptx.push_str("    mul.f32 %f0, %q_b, %cos_val;\n");
        ptx.push_str("    add.f32 %q_rot_b, %q_rot_b, %f0;\n");

        // Store rotated values to shared memory
        ptx.push_str("    shl.b64 %rd29, %rd22, 2;        // shmem byte offset\n");
        ptx.push_str("    st.shared.f32 [shmem + %rd29], %q_rot_a;\n");
        ptx.push_str(&format!(
            "    add.u64 %rd35, %rd29, {};          // offset_b in shmem\n",
            stride_val * 4
        ));
        ptx.push_str("    st.shared.f32 [shmem + %rd35], %q_rot_b;\n");

        // Advance and loop
        ptx.push_str("    add.u64 %rd22, %rd22, 128;      // elem_idx += blockDim.x\n");
        ptx.push_str("    setp.lt.u64 %p0, %rd22, %rd23;\n");
        ptx.push_str("    @%p0 bra LOOP_Q_LOAD_ROPE;\n");
    } else {
        // Non-RoPE path: straight copy from global to shared
        ptx.push_str("LOOP_Q_LOAD:\n");
        ptx.push_str("    shl.b64 %rd24, %rd22, 2;        // byte offset = elem_idx * 4\n");
        ptx.push_str("    add.u64 %rd25, %rd21, %rd24;    // global addr\n");
        ptx.push_str("    ld.global.f32 %f0, [%rd25];\n");
        ptx.push_str("    st.shared.f32 [shmem + %rd24], %f0;\n");
        ptx.push_str("    add.u64 %rd22, %rd22, 128;      // elem_idx += blockDim.x\n");
        ptx.push_str("    setp.lt.u64 %p0, %rd22, %rd23;\n");
        ptx.push_str("    @%p0 bra LOOP_Q_LOAD;\n");
    }

    ptx.push_str("    bar.sync 0;\n");
    ptx.push_str("    // Q tile now in shmem[0 .. block_q * head_dim * 4]\n");
}

fn emit_accumulator_init(ptx: &mut String, config: &FlashAttentionConfig) {
    ptx.push_str("    // Initialize accumulators\n");
    ptx.push_str("    // O_acc = 0, row_max = -inf, row_sum = 0\n");
    ptx.push_str("    mov.f32 %row_max, 0xFF800000;  // -inf as IEEE 754\n");
    ptx.push_str("    mov.f32 %row_sum, 0x00000000;  // 0.0\n");

    // Zero O_acc registers: each thread owns (block_q * head_dim) / 128 output elements
    let num_oacc = (config.block_q * config.head_dim / 128) as usize;
    ptx.push_str(&format!(
        "    // O_acc: {} registers per thread (f64..f{})\n",
        num_oacc,
        64 + num_oacc - 1
    ));
    for i in 0..num_oacc {
        ptx.push_str(&format!(
            "    mov.f32 %f{}, 0x00000000;\n",
            64 + i
        ));
    }

    // Compute k_max (upper bound for KV tile loop)
    if config.causal {
        ptx.push_str("    // Causal: k_max = min(q_start + block_q, seq_len)\n");
        ptx.push_str(&format!(
            "    add.u64 %k_max, %rd16, {};  // q_start + block_q\n",
            config.block_q
        ));
        ptx.push_str("    min.u64 %k_max, %k_max, %rd6;  // min(..., seq_len)\n");
    } else {
        ptx.push_str("    // Non-causal: k_max = seq_len\n");
        ptx.push_str("    mov.u64 %k_max, %rd6;\n");
    }

    // Initialize KV loop counter
    ptx.push_str("    mov.u64 %k_start, 0;\n");
}

fn emit_kv_tile_loop(ptx: &mut String, config: &FlashAttentionConfig) {
    ptx.push_str("    // === Main K/V tile loop ===\n");

    if config.causal {
        ptx.push_str("    // Causal: k_max = min(seq_len, q_start + block_q)\n");
        ptx.push_str("    // Zero-divergence — loop naturally terminates at diagonal\n");
    } else {
        ptx.push_str("    // Non-causal: k_max = seq_len\n");
    }

    // Check if loop should execute at all
    ptx.push_str("    setp.ge.u64 %p0, %k_start, %k_max;\n");
    ptx.push_str("    @%p0 bra LOOP_KV_END;\n");

    ptx.push_str("LOOP_KV_START:\n");

    // ── Phase 1: Load K tile into SRAM ──────────────────────────────
    ptx.push_str("    // Phase 1: Load K tile into SRAM\n");

    // shmem_K offset: K tile sits after Q tile in shared memory
    let shmem_k_offset = config.block_q * config.head_dim * 4; // bytes (f32)
    ptx.push_str(&format!(
        "    // shmem_K base = shmem + {} (after Q tile)\n",
        shmem_k_offset
    ));

    // Compute K base address
    if config.paged {
        ptx.push_str("    // Paged: block table indirection per physical block\n");
        ptx.push_str("    // One page table lookup per block_size tokens\n");
        // Paged K: look up physical block from block_table
        // logical_block = k_start / block_size
        ptx.push_str("    div.u64 %rd36, %k_start, %rd11;  // logical_block = k_start / block_size\n");
        // batch/head offset into block table
        ptx.push_str("    // block_table[batch_idx * heads + head_idx_kv, logical_block]\n");
        if config.gqa_group_size > 1 {
            ptx.push_str("    mul.lo.u64 %rd37, %rd19, %rd5;  // batch_idx * heads\n");
            ptx.push_str("    add.u64 %rd37, %rd37, %rd20;    // + kv_head\n");
        } else {
            ptx.push_str("    mul.lo.u64 %rd37, %rd19, %rd5;  // batch_idx * heads\n");
            ptx.push_str("    add.u64 %rd37, %rd37, %rd18;    // + head_idx\n");
        }
        // Read physical block index
        ptx.push_str("    // TODO: compute block_table stride from max_blocks\n");
        ptx.push_str("    shl.b64 %rd38, %rd36, 2;           // * 4 (u32 block indices)\n");
        ptx.push_str("    add.u64 %rd38, %rd8, %rd38;        // block_table_ptr + offset\n");
        ptx.push_str("    ld.global.u32 %r0, [%rd38];        // physical_block\n");
        ptx.push_str("    cvt.u64.u32 %rd39, %r0;            // physical_block as u64\n");
        // k_pool_base = k_pool_ptr + physical_block * block_size * head_dim * 4
        ptx.push_str("    mul.lo.u64 %rd40, %rd39, %rd11;    // phys_block * block_size\n");
        ptx.push_str("    mul.lo.u64 %rd40, %rd40, %rd7;     // * head_dim\n");
        ptx.push_str("    shl.b64 %rd40, %rd40, 2;           // * 4 bytes\n");
        ptx.push_str("    add.u64 %rd40, %rd9, %rd40;        // k_base = k_pool_ptr + offset\n");
    } else {
        // Dense K: k_base = k_ptr + (batch_idx * heads * seq_len * head_dim
        //   + head_idx * seq_len * head_dim + k_start * head_dim) * 4
        if config.gqa_group_size > 1 {
            ptx.push_str(&format!(
                "    // GQA: kv_head = q_head / {} (compile-time literal)\n",
                config.gqa_group_size
            ));
            ptx.push_str("    mul.lo.u64 %rd36, %rd19, %rd5;  // batch_idx * heads\n");
            ptx.push_str("    add.u64 %rd36, %rd36, %rd20;    // + kv_head\n");
        } else {
            ptx.push_str("    mul.lo.u64 %rd36, %rd19, %rd5;  // batch_idx * heads\n");
            ptx.push_str("    add.u64 %rd36, %rd36, %rd18;    // + head_idx\n");
        }
        ptx.push_str("    mul.lo.u64 %rd36, %rd36, %rd6;     // * seq_len\n");
        ptx.push_str("    add.u64 %rd36, %rd36, %k_start;    // + k_start\n");
        ptx.push_str("    mul.lo.u64 %rd36, %rd36, %rd7;     // * head_dim\n");
        ptx.push_str("    shl.b64 %rd36, %rd36, 2;           // * 4 bytes\n");
        ptx.push_str("    add.u64 %rd40, %rd1, %rd36;        // k_base = k_ptr + offset\n");
    }

    // Cooperative K tile load: each thread loads (block_kv * head_dim) / 128 elements
    let total_k_elems = config.block_kv * config.head_dim;
    ptx.push_str(&format!(
        "    // Cooperative K load: {} total elements, {} per thread\n",
        total_k_elems,
        total_k_elems / 128
    ));
    ptx.push_str("    cvt.u64.u32 %rd41, %tid_x;             // elem_idx = tid_x\n");
    ptx.push_str(&format!(
        "    mov.u64 %rd42, {};                    // total K elements\n",
        total_k_elems
    ));
    ptx.push_str("LOOP_K_LOAD:\n");
    ptx.push_str("    shl.b64 %rd43, %rd41, 2;               // byte offset = elem_idx * 4\n");
    ptx.push_str("    add.u64 %rd44, %rd40, %rd43;           // global addr\n");
    ptx.push_str("    ld.global.f32 %f0, [%rd44];\n");
    ptx.push_str(&format!(
        "    add.u64 %rd45, %rd43, {};             // shmem_K byte offset\n",
        shmem_k_offset
    ));
    ptx.push_str("    st.shared.f32 [shmem + %rd45], %f0;\n");
    ptx.push_str("    add.u64 %rd41, %rd41, 128;             // elem_idx += blockDim.x\n");
    ptx.push_str("    setp.lt.u64 %p0, %rd41, %rd42;\n");
    ptx.push_str("    @%p0 bra LOOP_K_LOAD;\n");

    ptx.push_str("    bar.sync 0;  // FENCE 1: K tile fully in SRAM\n");

    // ── Phase 2: S = Q @ K^T ───────────────────────────────────────
    ptx.push_str("    // Phase 2: S = Q_tile @ K_tile^T (registers)\n");

    // Each thread computes S values for its assigned Q rows and K columns
    // Thread tid_x is responsible for a subset of (q_row, k_col) pairs
    // For simplicity: tid_x maps to a linear index over block_q * block_kv
    // q_row = (tid_x * elems_per_thread + iter) / block_kv
    // k_col = (tid_x * elems_per_thread + iter) % block_kv
    let num_s_per_thread = (config.block_q * config.block_kv / 128) as usize;
    ptx.push_str(&format!(
        "    // Each thread computes {} S values\n",
        num_s_per_thread
    ));

    // Compute S values using dot product over head_dim
    ptx.push_str("    mov.u32 %r1, 0;                        // s_iter = 0\n");
    ptx.push_str(&format!(
        "    mov.u32 %r2, {};                      // num_s_per_thread\n",
        num_s_per_thread
    ));
    ptx.push_str("LOOP_S_OUTER:\n");

    // Compute which (q_row, k_col) this S element corresponds to
    ptx.push_str("    // linear_idx = tid_x + s_iter * 128\n");
    ptx.push_str("    mul.lo.u32 %r3, %r1, 128;\n");
    ptx.push_str("    add.u32 %r3, %r3, %tid_x;\n");
    ptx.push_str(&format!(
        "    div.u32 %r4, %r3, {};                 // q_row = linear_idx / block_kv\n",
        config.block_kv
    ));
    ptx.push_str(&format!(
        "    rem.u32 %r5, %r3, {};                 // k_col = linear_idx % block_kv\n",
        config.block_kv
    ));

    // Dot product: S[q_row][k_col] = sum_d(Q[q_row][d] * K[k_col][d])
    ptx.push_str("    mov.f32 %f0, 0x00000000;               // S accumulator = 0\n");
    ptx.push_str("    mov.u32 %r6, 0;                        // d = 0\n");
    ptx.push_str(&format!(
        "    mov.u32 %r7, {};                      // head_dim\n",
        config.head_dim
    ));
    ptx.push_str("LOOP_HD:\n");

    // Q address in shmem: shmem[q_row * head_dim + d] * 4 bytes
    ptx.push_str("    cvt.u64.u32 %rd43, %r4;               // q_row as u64\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd43, %rd43, {};          // q_row * head_dim\n",
        config.head_dim
    ));
    ptx.push_str("    cvt.u64.u32 %rd44, %r6;               // d as u64\n");
    ptx.push_str("    add.u64 %rd43, %rd43, %rd44;           // q_row * head_dim + d\n");
    ptx.push_str("    shl.b64 %rd43, %rd43, 2;              // * 4 bytes\n");
    ptx.push_str("    ld.shared.f32 %f1, [shmem + %rd43];   // Q[q_row][d]\n");

    // K address in shmem: shmem_K[k_col * head_dim + d] * 4 bytes
    ptx.push_str("    cvt.u64.u32 %rd44, %r5;               // k_col as u64\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd44, %rd44, {};          // k_col * head_dim\n",
        config.head_dim
    ));
    ptx.push_str("    cvt.u64.u32 %rd45, %r6;               // d as u64\n");
    ptx.push_str("    add.u64 %rd44, %rd44, %rd45;           // k_col * head_dim + d\n");
    ptx.push_str("    shl.b64 %rd44, %rd44, 2;              // * 4 bytes\n");
    ptx.push_str(&format!(
        "    add.u64 %rd44, %rd44, {};             // + shmem_K base offset\n",
        shmem_k_offset
    ));
    ptx.push_str("    ld.shared.f32 %f2, [shmem + %rd44];   // K[k_col][d]\n");

    ptx.push_str("    fma.rn.f32 %f0, %f1, %f2, %f0;       // S += Q[d] * K[d]\n");
    ptx.push_str("    add.u32 %r6, %r6, 1;                  // d++\n");
    ptx.push_str("    setp.lt.u32 %p0, %r6, %r7;\n");
    ptx.push_str("    @%p0 bra LOOP_HD;\n");

    // S[i][j] *= scale
    ptx.push_str("    // S[i][j] *= scale  (mul.f32)\n");
    ptx.push_str("    mul.f32 %f0, %f0, %scale;\n");

    // Causal masking on diagonal (standard or tree)
    if config.tree_mask {
        // M33: Tree-structured causal mask.
        // Node query_idx attends to node key_idx iff key_idx is an ancestor of query_idx.
        // Ancestor check: dfs_enter[key] <= dfs_enter[query] AND dfs_exit[key] >= dfs_exit[query]
        ptx.push_str("    // M33: Tree mask — ancestor check via DFS timestamps\n");
        ptx.push_str("    cvt.u64.u32 %rd43, %r5;           // k_col as u64\n");
        ptx.push_str("    add.u64 %rd43, %k_start, %rd43;   // key_idx = k_start + k_col\n");
        ptx.push_str("    cvt.u64.u32 %rd44, %r4;           // q_row as u64\n");
        ptx.push_str("    add.u64 %rd44, %rd16, %rd44;      // query_idx = q_start + q_row\n");
        // Load dfs_enter[key_idx]
        ptx.push_str("    mul.lo.u64 %rd45, %rd43, 4;       // key_idx * sizeof(i32)\n");
        ptx.push_str("    add.u64 %rd45, %dfs_enter_base, %rd45;\n");
        ptx.push_str("    ld.global.u32 %dfs_k_enter, [%rd45];\n");
        // Load dfs_enter[query_idx]
        ptx.push_str("    mul.lo.u64 %rd46, %rd44, 4;       // query_idx * sizeof(i32)\n");
        ptx.push_str("    add.u64 %rd46, %dfs_enter_base, %rd46;\n");
        ptx.push_str("    ld.global.u32 %dfs_q_enter, [%rd46];\n");
        // Load dfs_exit[key_idx]
        ptx.push_str("    mul.lo.u64 %rd47, %rd43, 4;\n");
        ptx.push_str("    add.u64 %rd47, %dfs_exit_base, %rd47;\n");
        ptx.push_str("    ld.global.u32 %dfs_k_exit, [%rd47];\n");
        // Load dfs_exit[query_idx]
        ptx.push_str("    mul.lo.u64 %rd48, %rd44, 4;\n");
        ptx.push_str("    add.u64 %rd48, %dfs_exit_base, %rd48;\n");
        ptx.push_str("    ld.global.u32 %dfs_q_exit, [%rd48];\n");
        // Check: is key an ancestor of query?
        // ancestor iff dfs_enter[key] <= dfs_enter[query] AND dfs_exit[key] >= dfs_exit[query]
        ptx.push_str("    setp.gt.u32 %p1, %dfs_k_enter, %dfs_q_enter;  // key enters AFTER query → not ancestor\n");
        ptx.push_str("    setp.lt.u32 %p_ancestor, %dfs_k_exit, %dfs_q_exit;  // key exits BEFORE query → not ancestor\n");
        ptx.push_str("    or.pred %p1, %p1, %p_ancestor;    // either condition → mask out\n");
        ptx.push_str("    @%p1 mov.f32 %f0, 0xFF800000;     // -inf for non-ancestor positions\n");
    } else if config.causal {
        ptx.push_str("    // Partial causal mask on diagonal tile: S[i][j] = -inf where k_start+j > q_start+i\n");
        ptx.push_str("    cvt.u64.u32 %rd43, %r5;           // k_col as u64\n");
        ptx.push_str("    add.u64 %rd43, %k_start, %rd43;   // k_abs = k_start + k_col\n");
        ptx.push_str("    cvt.u64.u32 %rd44, %r4;           // q_row as u64\n");
        ptx.push_str("    add.u64 %rd44, %rd16, %rd44;      // q_abs = q_start + q_row\n");
        ptx.push_str("    setp.gt.u64 %p1, %rd43, %rd44;    // k_abs > q_abs?\n");
        ptx.push_str("    @%p1 mov.f32 %f0, 0xFF800000;     // -inf for masked positions\n");
    }

    // Store S value in a temp register indexed by s_iter (we use %f3..%f3+num_s-1)
    // We'll cap at a manageable number and use a register-indexed store pattern
    ptx.push_str("    // Store S value for this thread's s_iter\n");
    // Use registers %f3..%f(3+num_s_per_thread-1) to hold S values for later phases
    // We emit a chain: if s_iter==0, store to %f3; if s_iter==1 store to %f4, etc.
    for i in 0..num_s_per_thread {
        ptx.push_str(&format!(
            "    setp.eq.u32 %p2, %r1, {};\n",
            i
        ));
        ptx.push_str(&format!(
            "    @%p2 mov.f32 %f{}, %f0;\n",
            3 + i
        ));
    }

    ptx.push_str("    add.u32 %r1, %r1, 1;                  // s_iter++\n");
    ptx.push_str("    setp.lt.u32 %p0, %r1, %r2;\n");
    ptx.push_str("    @%p0 bra LOOP_S_OUTER;\n");

    ptx.push_str("    bar.sync 0;  // FENCE 2: all warps done reading K before SRAM overwrite\n");

    // ── Phase 3: Online softmax ────────────────────────────────────
    ptx.push_str("    // Phase 3: Online softmax — S→P in-place in registers\n");

    // Find local max across this thread's S values
    ptx.push_str("    // Local max from this thread's S values\n");
    ptx.push_str("    mov.f32 %f0, 0xFF800000;               // f_local_max = -inf\n");
    for i in 0..num_s_per_thread {
        ptx.push_str(&format!(
            "    max.f32 %f0, %f0, %f{};              // max with S[{}]\n",
            3 + i, i
        ));
    }

    // Warp-level butterfly reduction for row_max (5 steps)
    ptx.push_str("    // Warp-level reductions via shfl.sync.bfly for row_max, row_sum\n");
    for offset in [16, 8, 4, 2, 1] {
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %shfl_tmp, %f0, {}, 31, 0xFFFFFFFF;\n",
            offset
        ));
        ptx.push_str("    max.f32 %f0, %f0, %shfl_tmp;\n");
    }

    // Correction factor: exp(old_max - new_max)
    ptx.push_str("    // new_max = max(row_max, warp_reduce_max(row_max_of_S))\n");
    ptx.push_str("    mov.f32 %old_max, %row_max;\n");
    ptx.push_str("    max.f32 %new_max, %row_max, %f0;\n");
    ptx.push_str("    mov.f32 %row_max, %new_max;\n");

    // correction = exp(old_max - new_max) via ex2.approx with LOG2E
    ptx.push_str("    // correction = exp(old_max - new_max)  // <= 1.0, no overflow\n");
    ptx.push_str("    sub.f32 %f0, %old_max, %new_max;\n");
    ptx.push_str("    mul.f32 %f0, %f0, %log2e;\n");
    ptx.push_str("    ex2.approx.f32 %correction, %f0;\n");

    // Rescale running accumulators
    ptx.push_str("    // row_sum = row_sum * correction + warp_reduce_sum(exp(S - new_max))\n");
    ptx.push_str("    mul.f32 %row_sum, %row_sum, %correction;\n");

    // Rescale O_acc registers
    ptx.push_str("    // O_acc *= correction\n");
    let num_oacc = (config.block_q * config.head_dim / 128) as usize;
    for i in 0..num_oacc {
        ptx.push_str(&format!(
            "    mul.f32 %f{}, %f{}, %correction;\n",
            64 + i, 64 + i
        ));
    }

    // Compute P = exp(S - new_max) and accumulate row_sum
    ptx.push_str("    // P = exp(S - new_max)  // overwrites S registers in-place\n");
    ptx.push_str("    mov.f32 %f1, 0x00000000;               // partial_sum = 0\n");
    for i in 0..num_s_per_thread {
        ptx.push_str(&format!(
            "    sub.f32 %f{}, %f{}, %new_max;\n",
            3 + i, 3 + i
        ));
        ptx.push_str(&format!(
            "    mul.f32 %f{}, %f{}, %log2e;\n",
            3 + i, 3 + i
        ));
        ptx.push_str(&format!(
            "    ex2.approx.f32 %f{}, %f{};\n",
            3 + i, 3 + i
        ));
        ptx.push_str(&format!(
            "    add.f32 %f1, %f1, %f{};              // partial_sum += P[{}]\n",
            3 + i, i
        ));
    }

    // Warp-level butterfly reduction for sum (5 steps)
    ptx.push_str("    // Sum reduction (5-step butterfly)\n");
    for offset in [16, 8, 4, 2, 1] {
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %shfl_tmp, %f1, {}, 31, 0xFFFFFFFF;\n",
            offset
        ));
        ptx.push_str("    add.f32 %f1, %f1, %shfl_tmp;\n");
    }
    ptx.push_str("    add.f32 %row_sum, %row_sum, %f1;       // row_sum += reduced partial_sum\n");

    // ── Phase 4: Load V tile (reuses K SRAM region) ────────────────
    ptx.push_str("    // Phase 4: Load V tile (reuses shmem_K address)\n");

    if config.paged {
        ptx.push_str("    // Paged V load: same block table indirection as K\n");
        // V uses same physical block but from v_pool
        ptx.push_str("    mul.lo.u64 %rd46, %rd39, %rd11;    // phys_block * block_size\n");
        ptx.push_str("    mul.lo.u64 %rd46, %rd46, %rd7;     // * head_dim\n");
        ptx.push_str("    shl.b64 %rd46, %rd46, 2;           // * 4 bytes\n");
        ptx.push_str("    add.u64 %rd46, %rd10, %rd46;       // v_base = v_pool_ptr + offset\n");
    } else {
        // Dense V: same addressing as K but from v_ptr
        if config.gqa_group_size > 1 {
            ptx.push_str("    mul.lo.u64 %rd46, %rd19, %rd5;  // batch_idx * heads\n");
            ptx.push_str("    add.u64 %rd46, %rd46, %rd20;    // + kv_head\n");
        } else {
            ptx.push_str("    mul.lo.u64 %rd46, %rd19, %rd5;  // batch_idx * heads\n");
            ptx.push_str("    add.u64 %rd46, %rd46, %rd18;    // + head_idx\n");
        }
        ptx.push_str("    mul.lo.u64 %rd46, %rd46, %rd6;     // * seq_len\n");
        ptx.push_str("    add.u64 %rd46, %rd46, %k_start;    // + k_start\n");
        ptx.push_str("    mul.lo.u64 %rd46, %rd46, %rd7;     // * head_dim\n");
        ptx.push_str("    shl.b64 %rd46, %rd46, 2;           // * 4 bytes\n");
        ptx.push_str("    add.u64 %rd46, %rd2, %rd46;        // v_base = v_ptr + offset\n");
    }

    // Cooperative V tile load
    ptx.push_str("    cvt.u64.u32 %rd47, %tid_x;             // elem_idx = tid_x\n");
    ptx.push_str("LOOP_V_LOAD:\n");
    ptx.push_str("    shl.b64 %rd48, %rd47, 2;               // byte offset\n");
    ptx.push_str("    add.u64 %rd49, %rd46, %rd48;           // global addr\n");
    ptx.push_str("    ld.global.f32 %f0, [%rd49];\n");
    ptx.push_str(&format!(
        "    add.u64 %rd50, %rd48, {};             // shmem_K byte offset (V reuses K region)\n",
        shmem_k_offset
    ));
    ptx.push_str("    st.shared.f32 [shmem + %rd50], %f0;\n");
    ptx.push_str("    add.u64 %rd47, %rd47, 128;             // elem_idx += blockDim.x\n");
    ptx.push_str(&format!(
        "    setp.lt.u64 %p0, %rd47, {};\n",
        total_k_elems
    ));
    ptx.push_str("    @%p0 bra LOOP_V_LOAD;\n");

    ptx.push_str("    bar.sync 0;  // FENCE 3: V tile fully in SRAM\n");

    // ── Phase 5: O_acc += P @ V_tile ───────────────────────────────
    ptx.push_str("    // Phase 5: O_acc += P @ V_tile\n");
    ptx.push_str("    // P in registers (Phase 3), V in SRAM\n");

    // For each O_acc element owned by this thread, accumulate P[q_row][k] * V[k][d]
    // O_acc layout: thread owns num_oacc elements spanning (q_row, d_col) pairs
    // Each P value (from %f3..%f(3+num_s-1)) multiplies a row of V
    ptx.push_str("    mov.u32 %r8, 0;                        // oacc_iter = 0\n");
    ptx.push_str(&format!(
        "    mov.u32 %r9, {};                      // num_oacc\n",
        num_oacc
    ));
    ptx.push_str("LOOP_PV_OUTER:\n");

    // Compute which (q_row, d_col) this O_acc element maps to
    ptx.push_str("    // o_linear = tid_x + oacc_iter * 128\n");
    ptx.push_str("    mul.lo.u32 %r10, %r8, 128;\n");
    ptx.push_str("    add.u32 %r10, %r10, %tid_x;\n");
    ptx.push_str(&format!(
        "    rem.u32 %r11, %r10, {};               // d_col = o_linear % head_dim\n",
        config.head_dim
    ));

    // Accumulate: for each k in block_kv, O_acc[q_row][d_col] += P[q_row][k] * V[k][d_col]
    // P values are in %f3..%f(3+num_s-1), each maps to a (q_row, k_col)
    ptx.push_str("    mov.u32 %r12, 0;                       // k_iter = 0\n");
    ptx.push_str(&format!(
        "    mov.u32 %r13, {};                     // block_kv\n",
        config.block_kv
    ));
    ptx.push_str("LOOP_PV:\n");

    // Load V[k_iter][d_col] from shmem_K region
    ptx.push_str("    cvt.u64.u32 %rd48, %r12;              // k_iter as u64\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd48, %rd48, {};          // k_iter * head_dim\n",
        config.head_dim
    ));
    ptx.push_str("    cvt.u64.u32 %rd49, %r11;              // d_col as u64\n");
    ptx.push_str("    add.u64 %rd48, %rd48, %rd49;           // k_iter * head_dim + d_col\n");
    ptx.push_str("    shl.b64 %rd48, %rd48, 2;              // * 4 bytes\n");
    ptx.push_str(&format!(
        "    add.u64 %rd48, %rd48, {};             // + shmem_K base\n",
        shmem_k_offset
    ));
    ptx.push_str("    ld.shared.f32 %f1, [shmem + %rd48];   // V[k][d_col]\n");

    // P value for (q_row, k_iter): stored in %f3..
    // We load from the S/P register using conditional moves
    ptx.push_str("    mov.f32 %f2, 0x00000000;               // default P = 0\n");
    for i in 0..num_s_per_thread {
        ptx.push_str(&format!(
            "    setp.eq.u32 %p2, %r12, {};\n",
            i
        ));
        ptx.push_str(&format!(
            "    @%p2 mov.f32 %f2, %f{};\n",
            3 + i
        ));
    }

    // O_acc[oacc_iter] += P * V
    // We need to index into %f64+oacc_iter — use conditional add pattern
    for i in 0..num_oacc {
        ptx.push_str(&format!(
            "    setp.eq.u32 %p3, %r8, {};\n",
            i
        ));
        ptx.push_str(&format!(
            "    @%p3 fma.rn.f32 %f{}, %f2, %f1, %f{};\n",
            64 + i, 64 + i
        ));
    }

    ptx.push_str("    add.u32 %r12, %r12, 1;                // k_iter++\n");
    ptx.push_str("    setp.lt.u32 %p0, %r12, %r13;\n");
    ptx.push_str("    @%p0 bra LOOP_PV;\n");

    ptx.push_str("    add.u32 %r8, %r8, 1;                  // oacc_iter++\n");
    ptx.push_str("    setp.lt.u32 %p0, %r8, %r9;\n");
    ptx.push_str("    @%p0 bra LOOP_PV_OUTER;\n");

    // Loop back
    ptx.push_str("    // Increment k_start, check loop bound\n");
    ptx.push_str(&format!(
        "    add.u64 %k_start, %k_start, {};       // k_start += block_kv\n",
        config.block_kv
    ));
    ptx.push_str("    setp.lt.u64 %p0, %k_start, %k_max;\n");
    ptx.push_str("    @%p0 bra LOOP_KV_START;\n");
    ptx.push_str("LOOP_KV_END:\n");
}

fn emit_finalize(ptx: &mut String, config: &FlashAttentionConfig) {
    ptx.push_str("    // Finalize: O = O_acc / row_sum\n");

    // Compute reciprocal of row_sum
    ptx.push_str("    rcp.approx.f32 %f0, %row_sum;          // 1.0 / row_sum\n");

    // Multiply each O_acc register by 1/row_sum
    let num_oacc = (config.block_q * config.head_dim / 128) as usize;
    for i in 0..num_oacc {
        ptx.push_str(&format!(
            "    mul.f32 %f{}, %f{}, %f0;\n",
            64 + i, 64 + i
        ));
    }
}

fn emit_output_store(ptx: &mut String, config: &FlashAttentionConfig) {
    ptx.push_str("    // Store output tile to global memory\n");

    // Compute output base address: out_base = out_ptr + (batch_idx * heads * seq_len * head_dim
    //   + head_idx * seq_len * head_dim + q_start * head_dim) * 2  (f16 output)
    ptx.push_str("    // Compute output base address\n");
    ptx.push_str("    mul.lo.u64 %rd51, %rd19, %rd5;         // batch_idx * heads\n");
    ptx.push_str("    add.u64 %rd51, %rd51, %rd18;           // + head_idx\n");
    ptx.push_str("    mul.lo.u64 %rd51, %rd51, %rd6;         // * seq_len\n");
    ptx.push_str("    add.u64 %rd51, %rd51, %rd16;           // + q_start\n");
    ptx.push_str("    mul.lo.u64 %rd51, %rd51, %rd7;         // * head_dim\n");
    ptx.push_str("    shl.b64 %rd51, %rd51, 1;               // * 2 (sizeof f16)\n");
    ptx.push_str("    add.u64 %rd51, %rd3, %rd51;            // out_base = out_ptr + offset\n");

    // Convert f32 O_acc to f16 and store to global memory
    // Each thread stores its num_oacc elements
    let num_oacc = (config.block_q * config.head_dim / 128) as usize;
    ptx.push_str(&format!(
        "    // Convert {} O_acc registers f32→f16, store to global\n",
        num_oacc
    ));

    for i in 0..num_oacc {
        let h_reg = i % 32;
        // Compute the global element index for this O_acc register
        // elem_idx = tid_x + i * 128
        ptx.push_str(&format!(
            "    cvt.rn.f16.f32 %h{}, %f{};              // f32 → f16\n",
            h_reg, 64 + i
        ));
        // Byte offset = (tid_x + i * 128) * 2
        ptx.push_str(&format!(
            "    // elem {} → global offset = (tid_x + {}) * 2\n",
            i,
            i * 128
        ));
        ptx.push_str("    cvt.u64.u32 %rd52, %tid_x;\n");
        ptx.push_str(&format!(
            "    add.u64 %rd52, %rd52, {};              // + i * 128\n",
            i * 128
        ));
        ptx.push_str("    shl.b64 %rd52, %rd52, 1;              // * 2 (f16)\n");
        ptx.push_str("    add.u64 %rd52, %rd51, %rd52;          // out_base + offset\n");
        ptx.push_str(&format!(
            "    st.global.b16 [%rd52], %h{};\n",
            h_reg
        ));
    }
}

fn emit_rope_cache_write_entry(
    ptx: &mut String,
    head_dim: i64,
    rope_style: RopeStyle,
) {
    ptx.push_str(".visible .entry nsl_rope_cache_write (\n");
    ptx.push_str("    .param .u64 k_projected_ptr,\n");
    ptx.push_str("    .param .u64 v_projected_ptr,\n");
    ptx.push_str("    .param .u64 cos_ptr,\n");
    ptx.push_str("    .param .u64 sin_ptr,\n");
    ptx.push_str("    .param .u64 positions_ptr,\n");
    ptx.push_str("    .param .u64 k_pool_ptr,\n");
    ptx.push_str("    .param .u64 v_pool_ptr,\n");
    ptx.push_str("    .param .u64 block_table_ptr,\n");
    ptx.push_str("    .param .u64 seq_ids_ptr,\n");
    ptx.push_str("    .param .u64 seq_lens_ptr,\n");
    ptx.push_str("    .param .u64 num_tokens,\n");
    ptx.push_str("    .param .u64 num_heads,\n");
    ptx.push_str("    .param .u64 head_dim,\n");
    ptx.push_str("    .param .u64 block_size\n");
    ptx.push_str(")\n");
    ptx.push_str("{\n");

    ptx.push_str("    .reg .u32 %tid_x, %bid_x, %bid_y, %bid_z;\n");
    ptx.push_str("    .reg .u64 %rd<32>;\n");
    ptx.push_str("    .reg .f32 %f<16>;\n");
    ptx.push_str("    .reg .f32 %cos_val, %sin_val, %k_a, %k_b, %k_rot_a, %k_rot_b;\n");

    ptx.push_str("    // Grid: (num_tokens, num_heads, ceil(head_dim/2))\n");
    ptx.push_str("    // token_idx = blockIdx.x (up to 2^31-1)\n");
    ptx.push_str("    // head_idx = blockIdx.y\n");
    ptx.push_str("    // dim_pair = blockIdx.z\n");
    ptx.push_str("    mov.u32 %bid_x, %ctaid.x;\n");
    ptx.push_str("    mov.u32 %bid_y, %ctaid.y;\n");
    ptx.push_str("    mov.u32 %bid_z, %ctaid.z;\n");

    let stride_comment = match rope_style {
        RopeStyle::HalfSplit => format!("stride = {} (half_split)", head_dim / 2),
        RopeStyle::Adjacent => "stride = 1 (adjacent)".to_string(),
    };
    ptx.push_str(&format!("    // RoPE style: {}\n", stride_comment));

    ptx.push_str("    // 1. Load K element pair from k_projected into registers\n");
    ptx.push_str("    // 2. Load cos[pos], sin[pos] from frequency table → registers\n");
    ptx.push_str("    // 3. Apply RoPE: k_rot_a = k_a*cos - k_b*sin; k_rot_b = k_a*sin + k_b*cos\n");
    ptx.push_str("    // 4. Look up physical block via block_table[seq_id * max_blocks + logical_idx]\n");
    ptx.push_str("    // 5. Write rotated K into paged K pool\n");
    ptx.push_str("    // 6. Write V directly into paged V pool (no rotation)\n");

    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");

    let _ = head_dim; // used in stride computation
}

// ── MMA PTX emission helpers (sm_80+) ────────────────────────────────
// These helpers are building blocks for the MMA codegen path. They are
// individually tested but will be wired into the main synthesis pipeline
// when the sm_80 feature gate is integrated into emit_kv_tile_loop.

/// Emit PTX to convert f32 registers to packed f16x2 (.b32) for MMA fragments.
///
/// Each destination register holds two f16 values packed into a 32-bit word.
/// `src_f32` names are f32 register names (even count), `dst_b32` are .b32 output names (half count).
#[allow(dead_code)]
fn emit_f32_to_f16_pack(
    ptx: &mut String,
    src_f32: &[String],
    dst_b32: &[String],
) {
    assert_eq!(src_f32.len(), dst_b32.len() * 2, "f16 pack: src must be 2x dst");
    for i in 0..dst_b32.len() {
        let lo = &src_f32[i * 2];
        let hi = &src_f32[i * 2 + 1];
        let dst = &dst_b32[i];
        ptx.push_str(&format!("    cvt.rn.f16.f32 %mma_h0, %{};\n", lo));
        ptx.push_str(&format!("    cvt.rn.f16.f32 %mma_h1, %{};\n", hi));
        ptx.push_str(&format!("    mov.b32 %{}, {{%mma_h0, %mma_h1}};\n", dst));
    }
}

/// Emit PTX to load an MMA A-fragment (m16xk16, row-major f16) from shared memory.
///
/// In m16n8k16, the A-fragment maps as follows:
/// - Thread t holds rows at: row = (t%4)*2 + (t/16), row+8 (for groups 0,1)
/// - Each thread holds 8 f16 values in 4 .b32 registers
/// - Registers a0,a1 cover k=0..7; a2,a3 cover k=8..15
///
/// `frag_regs`: 4 .b32 register names for the fragment output.
/// `smem_base_expr`: PTX expression for shared memory base address.
/// `row_stride`: row stride in bytes (head_dim * 2 for f16, or head_dim * 4 for f32 shmem).
#[allow(dead_code)]
fn emit_load_a_fragment_smem(
    ptx: &mut String,
    frag_regs: &[String; 4],
    smem_base_expr: &str,
    row_stride: usize,
) {
    ptx.push_str("    // Load A-fragment (m16xk16 row-major) from shared memory\n");
    // Each thread's row: row = (laneid % 4) * 2 + (laneid / 16)
    // But we load 4 .b32 registers covering 4 pairs of f16 values
    // Registers 0,1 are k=0..7; registers 2,3 are k=8..15
    // Within each pair, the two f16s are at consecutive addresses
    for (reg_idx, k_base_pair) in [(0, 0usize), (1, 4), (2, 8), (3, 12)].iter() {
        let byte_col_offset = k_base_pair * 2; // each f16 is 2 bytes, pairs of 2 f16 = 4 bytes
        ptx.push_str(&format!(
            "    mad.lo.u32 %mma_addr, %mma_a_row, {}, {};  // row * stride + base\n",
            row_stride, smem_base_expr
        ));
        ptx.push_str(&format!(
            "    add.u32 %mma_addr, %mma_addr, {};  // + k_col byte offset\n",
            byte_col_offset
        ));
        ptx.push_str(&format!(
            "    ld.shared.b32 %{}, [%mma_addr];\n",
            frag_regs[*reg_idx]
        ));
    }
}

/// Emit PTX to load an MMA B-fragment (k16xn8, col-major f16) from shared memory.
///
/// In m16n8k16, the B-fragment maps as follows:
/// - Each thread holds 4 f16 values in 2 .b32 registers
/// - Registers b0 covers k=0..7, b1 covers k=8..15
///
/// `frag_regs`: 2 .b32 register names.
/// `smem_base_expr`: PTX expression for shared memory base address.
/// `row_stride`: row stride in bytes for the K/V matrix in shared memory.
#[allow(dead_code)]
fn emit_load_b_fragment_smem(
    ptx: &mut String,
    frag_regs: &[String; 2],
    smem_base_expr: &str,
    row_stride: usize,
) {
    ptx.push_str("    // Load B-fragment (k16xn8 col-major) from shared memory\n");
    for (reg_idx, k_base_pair) in [(0, 0usize), (1, 8)].iter() {
        let byte_col_offset = k_base_pair * 2;
        ptx.push_str(&format!(
            "    mad.lo.u32 %mma_addr, %mma_b_row, {}, {};  // row * stride + base\n",
            row_stride, smem_base_expr
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

/// Emit PTX for a single MMA instruction: D = A @ B + C (all in-register).
#[allow(dead_code)]
fn emit_mma_instruction(
    ptx: &mut String,
    d_regs: &[String; 4],   // D accumulator (f32 x4)
    a_regs: &[String; 4],   // A fragment (.b32 x4)
    b_regs: &[String; 2],   // B fragment (.b32 x2)
    c_regs: &[String; 4],   // C accumulator (f32 x4)
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
    ptx.push_str(&format!(
        "        {{{}, {}}},\n",
        b_regs[0], b_regs[1]
    ));
    ptx.push_str(&format!(
        "        {{{}, {}, {}, {}}};\n",
        c_regs[0], c_regs[1], c_regs[2], c_regs[3]
    ));
}

/// Emit the Q@K^T matmul using MMA instructions.
///
/// Produces S tile of shape [block_q, block_kv] distributed across warps in MMA
/// accumulator layout. Each warp processes one m-tile-row at a time to keep
/// register pressure manageable (O(n_tiles_s * 4) accumulators per warp).
///
/// Requires: Q tile in shmem[0..block_q*head_dim*4], K tile in shmem[shmem_k_offset..].
/// Both stored as f32 in shared memory — fragment loads convert to f16 on the fly.
///
/// Output: S accumulators in `%acc_s_{nt}_{r}` registers for the current m-tile.
/// The caller must consume S (softmax + P@V) before advancing to the next m-tile.
#[allow(dead_code)]
fn emit_qk_matmul_mma(
    ptx: &mut String,
    _block_q: usize,
    block_kv: usize,
    head_dim: usize,
    shmem_k_offset: usize,
) {
    let n_tiles_s = block_kv / MMA_N;
    let k_iters = head_dim / MMA_K;

    ptx.push_str("    // === Q@K^T via MMA (m16n8k16) ===\n");
    ptx.push_str(&format!(
        "    // n_tiles_s={}, k_iters={}, processing one m-tile at a time\n",
        n_tiles_s, k_iters
    ));

    // Zero S accumulators for current m-tile (n_tiles_s * 4 f32 registers)
    for nt in 0..n_tiles_s {
        for r in 0..4 {
            ptx.push_str(&format!(
                "    mov.f32 %acc_s_{}_{}, 0x00000000;\n", nt, r
            ));
        }
    }

    // K-dimension loop
    ptx.push_str("    mov.u32 %mma_k_iter, 0;\n");
    ptx.push_str("QK_MMA_K_LOOP:\n");

    // Load A-fragment from Q shared memory for current m-tile row
    // Q is in shmem at f32, but MMA needs f16 — we load f32 and convert
    // For simplicity, load the 16x16 A block as f32 from shmem, convert to f16 pairs
    ptx.push_str("    // Load Q A-fragment (f32 from shmem, convert to f16)\n");
    for i in 0..4 {
        let k_pair = i * 4; // each .b32 holds a pair of f16 at k positions k_pair, k_pair+1
        ptx.push_str(&format!(
            "    // A-frag reg {}: k_pair={}\n", i, k_pair
        ));
        // Compute shmem address: q_shmem[a_row * head_dim + k_iter * MMA_K + k_pair]
        ptx.push_str(&format!(
            "    mul.lo.u32 %mma_addr, %mma_a_row, {};  // a_row * head_dim * 4\n",
            head_dim * 4
        ));
        ptx.push_str(&format!(
            "    mad.lo.u32 %mma_addr, %mma_k_iter, {}, %mma_addr;  // + k_iter * MMA_K * 4\n",
            MMA_K * 4
        ));
        ptx.push_str(&format!(
            "    add.u32 %mma_addr, %mma_addr, {};  // + k_pair * 4\n",
            k_pair * 4
        ));
        // Also add m_tile * MMA_M * head_dim * 4 (handled by caller's loop over m-tiles)
        ptx.push_str("    add.u32 %mma_addr, %mma_addr, %mma_m_tile_byte_offset;\n");
        // Load two consecutive f32 values, convert to f16, pack
        ptx.push_str("    ld.shared.f32 %mma_f32_lo, [shmem + %mma_addr];\n");
        ptx.push_str("    add.u32 %mma_addr, %mma_addr, 4;\n");
        ptx.push_str("    ld.shared.f32 %mma_f32_hi, [shmem + %mma_addr];\n");
        ptx.push_str("    cvt.rn.f16.f32 %mma_h0, %mma_f32_lo;\n");
        ptx.push_str("    cvt.rn.f16.f32 %mma_h1, %mma_f32_hi;\n");
        ptx.push_str(&format!(
            "    mov.b32 %aq_{}, {{%mma_h0, %mma_h1}};\n", i
        ));
    }

    // Load B-fragments from K^T shared memory for each n-tile
    for nt in 0..n_tiles_s {
        ptx.push_str(&format!(
            "    // K^T B-fragment for n_tile={}\n", nt
        ));
        for bi in 0..2 {
            let k_pair = bi * 8; // b0 covers k=0..7, b1 covers k=8..15
            // K in shmem at shmem_k_offset, stored as K[k_col][head_dim] row-major
            // For K^T: we need K[k_col=nt*8+col, d=k_iter*16+k] transposed
            // B-fragment wants col-major: B[k, n] where k is the MMA K-dim, n is the N-dim
            // So we load K[nt*MMA_N + b_row, k_iter*MMA_K + k_pair] from shmem
            ptx.push_str(&format!(
                "    mul.lo.u32 %mma_addr, %mma_b_row, {};  // b_row * head_dim * 4\n",
                head_dim * 4
            ));
            ptx.push_str(&format!(
                "    mad.lo.u32 %mma_addr, %mma_k_iter, {}, %mma_addr;  // + k_iter * MMA_K * 4\n",
                MMA_K * 4
            ));
            ptx.push_str(&format!(
                "    add.u32 %mma_addr, %mma_addr, {};  // + k_pair * 4\n",
                k_pair * 4
            ));
            // Add n_tile offset: nt * MMA_N * head_dim * 4
            ptx.push_str(&format!(
                "    add.u32 %mma_addr, %mma_addr, {};  // + n_tile * MMA_N * head_dim * 4\n",
                nt * MMA_N * head_dim * 4
            ));
            // Add shmem_k_offset
            ptx.push_str(&format!(
                "    add.u32 %mma_addr, %mma_addr, {};  // + shmem_K base\n",
                shmem_k_offset
            ));
            ptx.push_str("    ld.shared.f32 %mma_f32_lo, [shmem + %mma_addr];\n");
            ptx.push_str("    add.u32 %mma_addr, %mma_addr, 4;\n");
            ptx.push_str("    ld.shared.f32 %mma_f32_hi, [shmem + %mma_addr];\n");
            ptx.push_str("    cvt.rn.f16.f32 %mma_h0, %mma_f32_lo;\n");
            ptx.push_str("    cvt.rn.f16.f32 %mma_h1, %mma_f32_hi;\n");
            ptx.push_str(&format!(
                "    mov.b32 %bk_{}_{}, {{%mma_h0, %mma_h1}};\n", nt, bi
            ));
        }
    }

    // Issue MMA for each n-tile
    for nt in 0..n_tiles_s {
        ptx.push_str(&format!(
            "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n"
        ));
        ptx.push_str(&format!(
            "        {{%acc_s_{nt}_0, %acc_s_{nt}_1, %acc_s_{nt}_2, %acc_s_{nt}_3}},\n"
        ));
        ptx.push_str(
            "        {%aq_0, %aq_1, %aq_2, %aq_3},\n"
        );
        ptx.push_str(&format!(
            "        {{%bk_{nt}_0, %bk_{nt}_1}},\n"
        ));
        ptx.push_str(&format!(
            "        {{%acc_s_{nt}_0, %acc_s_{nt}_1, %acc_s_{nt}_2, %acc_s_{nt}_3}};\n"
        ));
    }

    // Loop back over K
    ptx.push_str("    add.u32 %mma_k_iter, %mma_k_iter, 1;\n");
    ptx.push_str(&format!(
        "    setp.lt.u32 %mma_pk, %mma_k_iter, {};\n", k_iters
    ));
    ptx.push_str("    @%mma_pk bra QK_MMA_K_LOOP;\n");

    // Scale: S = S * scale
    ptx.push_str("    // Scale S by 1/sqrt(head_dim)\n");
    for nt in 0..n_tiles_s {
        for r in 0..4 {
            ptx.push_str(&format!(
                "    mul.f32 %acc_s_{}_{}, %acc_s_{}_{}, %scale;\n",
                nt, r, nt, r
            ));
        }
    }
}

/// Emit the P@V matmul using MMA instructions.
///
/// P (attention weights after softmax) is in S accumulator registers — must be
/// converted from f32 to packed f16 before use as A-fragment.
/// V is in shared memory at shmem_k_offset (reuses K's region).
///
/// Accumulates into O registers: O += P @ V for the current m-tile.
#[allow(dead_code)]
fn emit_pv_matmul_mma(
    ptx: &mut String,
    block_kv: usize,
    head_dim: usize,
    shmem_k_offset: usize,
) {
    let n_tiles_o = head_dim / MMA_N;
    let k_iters = block_kv / MMA_K; // k-dim for P@V is block_kv

    ptx.push_str("    // === P@V via MMA (m16n8k16) ===\n");
    ptx.push_str(&format!(
        "    // n_tiles_o={}, k_iters={}\n", n_tiles_o, k_iters
    ));

    // K-dimension loop over block_kv
    ptx.push_str("    mov.u32 %mma_k_iter, 0;\n");
    ptx.push_str("PV_MMA_K_LOOP:\n");

    // Load A-fragment from P (attention weights in S accumulator registers)
    // P is in %acc_s_{nt}_{r} registers — need to convert to f16 pairs
    // For the current k_iter, we need P values at k=k_iter*16..k_iter*16+15
    // These come from S accumulators for n_tiles at positions k_iter*2 and k_iter*2+1
    // (since each n-tile covers 8 columns, 16 K values = 2 n-tiles)
    ptx.push_str("    // Convert P registers to f16 A-fragment\n");
    // The 4 A-fragment .b32 registers pack 8 f16 values
    // We take P values from the S accumulators corresponding to this k-range
    for i in 0..4 {
        // Use the S accumulator values directly: acc_s_{k_tile}_{reg}
        // k_tile index depends on k_iter: nt_base = k_iter * (MMA_K / MMA_N) = k_iter * 2
        // register i maps to specific positions in the MMA layout
        ptx.push_str(&format!(
            "    // A-frag P reg {} from S accumulators\n", i
        ));
        // For simplicity, read the S accumulator that maps to this fragment position
        // acc_s_{nt}_{r} where nt = k_iter*2 + (i/2), r = (i%2)*2 + laneid_mapping
        // This is approximate — actual mapping depends on MMA thread layout
        ptx.push_str(&format!(
            "    mul.lo.u32 %mma_addr, %mma_k_iter, 2;\n"  // nt_base = k_iter * 2
        ));
        let nt_offset = i / 2;
        let r_base = (i % 2) * 2;
        ptx.push_str(&format!(
            "    add.u32 %mma_addr, %mma_addr, {};  // nt = nt_base + {}\n",
            nt_offset, nt_offset
        ));
        // Convert the two f32 S values to packed f16
        // Use dynamic register indexing via conditional moves
        ptx.push_str(&format!(
            "    // Pack S acc values for A-frag position {}\n", i
        ));
        ptx.push_str(&format!(
            "    cvt.rn.f16.f32 %mma_h0, %acc_s_scratch_{};\n", r_base
        ));
        ptx.push_str(&format!(
            "    cvt.rn.f16.f32 %mma_h1, %acc_s_scratch_{};\n", r_base + 1
        ));
        ptx.push_str(&format!(
            "    mov.b32 %ap_{}, {{%mma_h0, %mma_h1}};\n", i
        ));
    }

    // Load B-fragments from V shared memory for each output n-tile
    for nt in 0..n_tiles_o {
        ptx.push_str(&format!(
            "    // V B-fragment for n_tile={}\n", nt
        ));
        for bi in 0..2 {
            let k_pair = bi * 8;
            // V[k_col, d] in shmem at shmem_k_offset, row-major
            // B-fragment: V[k=k_iter*16+k_pair, d=nt*8+col]
            ptx.push_str(&format!(
                "    mul.lo.u32 %mma_addr, %mma_b_row, {};  // b_row * head_dim * 4\n",
                head_dim * 4
            ));
            ptx.push_str(&format!(
                "    mad.lo.u32 %mma_addr, %mma_k_iter, {}, %mma_addr;  // + k_iter * MMA_K * head_dim * 4\n",
                MMA_K * head_dim * 4
            ));
            ptx.push_str(&format!(
                "    add.u32 %mma_addr, %mma_addr, {};  // + nt * MMA_N * 4 + k_pair * 4\n",
                nt * MMA_N * 4 + k_pair * 4
            ));
            ptx.push_str(&format!(
                "    add.u32 %mma_addr, %mma_addr, {};  // + shmem_K base\n",
                shmem_k_offset
            ));
            ptx.push_str("    ld.shared.f32 %mma_f32_lo, [shmem + %mma_addr];\n");
            ptx.push_str("    add.u32 %mma_addr, %mma_addr, 4;\n");
            ptx.push_str("    ld.shared.f32 %mma_f32_hi, [shmem + %mma_addr];\n");
            ptx.push_str("    cvt.rn.f16.f32 %mma_h0, %mma_f32_lo;\n");
            ptx.push_str("    cvt.rn.f16.f32 %mma_h1, %mma_f32_hi;\n");
            ptx.push_str(&format!(
                "    mov.b32 %bv_{}_{}, {{%mma_h0, %mma_h1}};\n", nt, bi
            ));
        }
    }

    // Issue MMA for each output n-tile
    for nt in 0..n_tiles_o {
        ptx.push_str("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n");
        ptx.push_str(&format!(
            "        {{%acc_o_{nt}_0, %acc_o_{nt}_1, %acc_o_{nt}_2, %acc_o_{nt}_3}},\n"
        ));
        ptx.push_str(
            "        {%ap_0, %ap_1, %ap_2, %ap_3},\n"
        );
        ptx.push_str(&format!(
            "        {{%bv_{nt}_0, %bv_{nt}_1}},\n"
        ));
        ptx.push_str(&format!(
            "        {{%acc_o_{nt}_0, %acc_o_{nt}_1, %acc_o_{nt}_2, %acc_o_{nt}_3}};\n"
        ));
    }

    // Loop back over K
    ptx.push_str("    add.u32 %mma_k_iter, %mma_k_iter, 1;\n");
    ptx.push_str(&format!(
        "    setp.lt.u32 %mma_pk, %mma_k_iter, {};\n", k_iters
    ));
    ptx.push_str("    @%mma_pk bra PV_MMA_K_LOOP;\n");
}

/// Emit MMA-specific register declarations for the Q@K^T and P@V paths.
#[allow(dead_code)]
fn emit_mma_qk_registers(ptx: &mut String, block_kv: usize, head_dim: usize) {
    let n_tiles_s = block_kv / MMA_N;
    let n_tiles_o = head_dim / MMA_N;

    // S accumulators (for current m-tile only — register pressure managed)
    for nt in 0..n_tiles_s {
        ptx.push_str(&format!(
            "    .reg .f32 %acc_s_{nt}_0, %acc_s_{nt}_1, %acc_s_{nt}_2, %acc_s_{nt}_3;\n"
        ));
    }

    // O accumulators (for current m-tile only)
    for nt in 0..n_tiles_o {
        ptx.push_str(&format!(
            "    .reg .f32 %acc_o_{nt}_0, %acc_o_{nt}_1, %acc_o_{nt}_2, %acc_o_{nt}_3;\n"
        ));
    }

    // A-fragment registers for Q (4 .b32)
    ptx.push_str("    .reg .b32 %aq_0, %aq_1, %aq_2, %aq_3;\n");

    // B-fragment registers for K^T (n_tiles_s * 2 .b32)
    for nt in 0..n_tiles_s {
        ptx.push_str(&format!(
            "    .reg .b32 %bk_{nt}_0, %bk_{nt}_1;\n"
        ));
    }

    // A-fragment registers for P (4 .b32)
    ptx.push_str("    .reg .b32 %ap_0, %ap_1, %ap_2, %ap_3;\n");

    // B-fragment registers for V (n_tiles_o * 2 .b32)
    for nt in 0..n_tiles_o {
        ptx.push_str(&format!(
            "    .reg .b32 %bv_{nt}_0, %bv_{nt}_1;\n"
        ));
    }

    // MMA temporaries
    ptx.push_str("    .reg .f32 %mma_f32_lo, %mma_f32_hi;  // shmem f32 load temps\n");
    ptx.push_str("    .reg .u32 %mma_k_iter;                // K-dimension loop counter\n");
    ptx.push_str("    .reg .u32 %mma_m_tile_byte_offset;    // byte offset for current m-tile\n");
    ptx.push_str("    .reg .pred %mma_pk;                    // K-loop predicate\n");
}

/// Compute XOR-based swizzle offset for shared memory to avoid bank conflicts
/// during MMA fragment loads.
///
/// Standard 32-bank shared memory layout causes conflicts when multiple threads
/// in a warp access the same bank. XOR swizzle distributes accesses across banks.
///
/// `row`: row index in the tile
/// `col_bytes`: column offset in bytes
/// Returns: swizzled byte offset
pub fn swizzle_smem_offset(row: usize, col_bytes: usize) -> usize {
    let base = row * 128 + col_bytes; // assume 128-byte row stride (typical for f16 head_dim=64)
    let bank = (base / 4) % 32;
    let swizzle_bits = (row % 8) ^ (bank % 8);
    base ^ (swizzle_bits << 2) // shift by 2 = multiply by 4 (bank granularity)
}

/// Emit PTX for XOR-based shared memory swizzle during cooperative tile stores.
///
/// Produces PTX that transforms a linear byte offset into a swizzled offset
/// before storing to shared memory. Used when loading Q/K/V tiles from global
/// to shared memory to ensure bank-conflict-free MMA fragment loads.
#[allow(dead_code)]
fn emit_smem_swizzle_store(ptx: &mut String) {
    ptx.push_str("    // XOR swizzle for bank-conflict-free shared memory\n");
    ptx.push_str("    // Input: %smem_linear_off (linear byte offset)\n");
    ptx.push_str("    // Output: %smem_swiz_off (swizzled byte offset)\n");
    ptx.push_str("    shr.u32 %smem_bank, %smem_linear_off, 2;    // bank = offset / 4\n");
    ptx.push_str("    and.b32 %smem_bank, %smem_bank, 31;         // bank = bank % 32\n");
    ptx.push_str("    shr.u32 %smem_row_bits, %smem_linear_off, 7; // row ≈ offset / 128\n");
    ptx.push_str("    and.b32 %smem_row_bits, %smem_row_bits, 7;  // row % 8\n");
    ptx.push_str("    and.b32 %smem_bank_lo, %smem_bank, 7;       // bank % 8\n");
    ptx.push_str("    xor.b32 %smem_swiz, %smem_row_bits, %smem_bank_lo;  // XOR\n");
    ptx.push_str("    shl.u32 %smem_swiz, %smem_swiz, 2;          // * 4 bytes\n");
    ptx.push_str("    xor.b32 %smem_swiz_off, %smem_linear_off, %smem_swiz;  // apply swizzle\n");
}

/// Emit online softmax adapted for MMA accumulator layout.
///
/// In MMA m16n8k16 output layout, thread t holds 4 f32 values at:
///   - Registers 0,1: row = (t%4)*2 + (t/16), cols depend on n-tile
///   - Registers 2,3: row = (t%4)*2 + (t/16) + 8, cols depend on n-tile
///
/// For per-row max/sum, threads sharing a row must communicate via warp shuffles.
/// Within one 16x8 MMA tile, each row is covered by threads with the same
/// (laneid % 4) and (laneid / 16) values — but different n-tiles extend the columns.
///
/// This function emits:
///   1. Per-register local max across all S accumulators
///   2. Warp shuffle to compute per-row global max
///   3. Rescale existing O accumulators by exp(old_max - new_max)
///   4. Compute P = exp(S - new_max), accumulate row_sum
///   5. Warp shuffle to compute per-row global sum
#[allow(dead_code)]
fn emit_mma_online_softmax(
    ptx: &mut String,
    block_kv: usize,
    head_dim: usize,
) {
    let n_tiles_s = block_kv / MMA_N;
    let n_tiles_o = head_dim / MMA_N;

    ptx.push_str("    // === Online softmax (MMA layout) ===\n");

    // Step 1: Find local max across this thread's S accumulator values
    ptx.push_str("    mov.f32 %mma_local_max, 0xFF800000;  // -inf\n");
    for nt in 0..n_tiles_s {
        for r in 0..4 {
            ptx.push_str(&format!(
                "    max.f32 %mma_local_max, %mma_local_max, %acc_s_{}_{};  // S[{}][{}]\n",
                nt, r, nt, r
            ));
        }
    }

    // Step 2: Warp shuffle butterfly reduction for row max
    // Threads sharing a row: in m16n8k16, 4 threads share one row position
    // (those with same laneid%4 and laneid/16 value across n-tiles)
    // Use 5-step butterfly to reduce across all 32 threads in the warp
    ptx.push_str("    // Warp shuffle for row_max\n");
    for offset in [16, 8, 4, 2, 1] {
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %mma_shfl_tmp, %mma_local_max, {}, 31, 0xFFFFFFFF;\n",
            offset
        ));
        ptx.push_str("    max.f32 %mma_local_max, %mma_local_max, %mma_shfl_tmp;\n");
    }

    // Step 3: Rescale existing accumulators
    ptx.push_str("    mov.f32 %mma_old_max, %mma_row_max;\n");
    ptx.push_str("    max.f32 %mma_row_max, %mma_row_max, %mma_local_max;  // new_max\n");
    ptx.push_str("    // correction = exp(old_max - new_max)\n");
    ptx.push_str("    sub.f32 %mma_correction, %mma_old_max, %mma_row_max;\n");
    ptx.push_str("    mul.f32 %mma_correction, %mma_correction, %log2e;\n");
    ptx.push_str("    ex2.approx.f32 %mma_correction, %mma_correction;\n");
    ptx.push_str("    mul.f32 %mma_row_sum, %mma_row_sum, %mma_correction;\n");

    // Rescale O accumulators
    for nt in 0..n_tiles_o {
        for r in 0..4 {
            ptx.push_str(&format!(
                "    mul.f32 %acc_o_{}_{}, %acc_o_{}_{}, %mma_correction;\n",
                nt, r, nt, r
            ));
        }
    }

    // Step 4: P = exp(S - new_max), accumulate row_sum
    ptx.push_str("    mov.f32 %mma_partial_sum, 0x00000000;  // 0.0\n");
    for nt in 0..n_tiles_s {
        for r in 0..4 {
            ptx.push_str(&format!(
                "    sub.f32 %acc_s_{}_{}, %acc_s_{}_{}, %mma_row_max;\n",
                nt, r, nt, r
            ));
            ptx.push_str(&format!(
                "    mul.f32 %acc_s_{}_{}, %acc_s_{}_{}, %log2e;\n",
                nt, r, nt, r
            ));
            ptx.push_str(&format!(
                "    ex2.approx.f32 %acc_s_{}_{}, %acc_s_{}_{};  // P[{}][{}]\n",
                nt, r, nt, r, nt, r
            ));
            ptx.push_str(&format!(
                "    add.f32 %mma_partial_sum, %mma_partial_sum, %acc_s_{}_{};  // += P\n",
                nt, r
            ));
        }
    }

    // Step 5: Warp shuffle for row_sum
    ptx.push_str("    // Warp shuffle for row_sum\n");
    for offset in [16, 8, 4, 2, 1] {
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %mma_shfl_tmp, %mma_partial_sum, {}, 31, 0xFFFFFFFF;\n",
            offset
        ));
        ptx.push_str("    add.f32 %mma_partial_sum, %mma_partial_sum, %mma_shfl_tmp;\n");
    }
    ptx.push_str("    add.f32 %mma_row_sum, %mma_row_sum, %mma_partial_sum;\n");
}

/// Emit register declarations for the MMA online softmax path.
#[allow(dead_code)]
fn emit_mma_softmax_registers(ptx: &mut String) {
    ptx.push_str("    // MMA softmax registers\n");
    ptx.push_str("    .reg .f32 %mma_row_max, %mma_row_sum;\n");
    ptx.push_str("    .reg .f32 %mma_old_max, %mma_local_max;\n");
    ptx.push_str("    .reg .f32 %mma_correction, %mma_partial_sum;\n");
    ptx.push_str("    .reg .f32 %mma_shfl_tmp;\n");
    // Initialize
    ptx.push_str("    mov.f32 %mma_row_max, 0xFF800000;  // -inf\n");
    ptx.push_str("    mov.f32 %mma_row_sum, 0x00000000;  // 0.0\n");
}

/// Emit register declarations for shared memory swizzle temporaries.
#[allow(dead_code)]
fn emit_smem_swizzle_registers(ptx: &mut String) {
    ptx.push_str("    .reg .u32 %smem_linear_off, %smem_swiz_off;\n");
    ptx.push_str("    .reg .u32 %smem_bank, %smem_row_bits, %smem_bank_lo, %smem_swiz;\n");
}

// ── wgmma.mma_async PTX emission helpers (sm_90+ / Hopper) ──────────

/// wgmma tile dimensions: m64n64k16 for f16, m64n64k32 for fp8.
const WGMMA_M: usize = 64;
const WGMMA_N: usize = 64;
const WGMMA_K_F16: usize = 16;

/// Emit PTX for a wgmma shared memory matrix descriptor.
///
/// wgmma reads A and B from shared memory via 64-bit descriptors that encode
/// the base address, leading dimension stride, and swizzle mode. The descriptor
/// is constructed in registers then passed to the wgmma instruction.
///
/// `desc_reg`: name of the .b64 register to hold the descriptor.
/// `smem_base_expr`: PTX expression for the base address of the tile in shared memory.
/// `leading_dim_bytes`: stride in bytes between rows (must be 128-byte aligned for wgmma).
/// `swizzle_mode`: 0=none, 1=32B, 2=64B, 3=128B (128B required for peak performance).
#[allow(dead_code)]
fn emit_wgmma_smem_descriptor(
    ptx: &mut String,
    desc_reg: &str,
    smem_base_expr: &str,
    leading_dim_bytes: usize,
    swizzle_mode: u32,
) {
    use std::fmt::Write;
    // Descriptor layout (64-bit):
    //   [13:0]  = base address (byte offset in shared memory, 16-byte aligned → shift right 4)
    //   [15:14] = swizzle mode
    //   [29:16] = leading dimension stride (in 16-byte units)
    //   [31:30] = reserved
    //   [63:32] = reserved
    writeln!(ptx, "    // Build wgmma shared memory descriptor for {desc_reg}").unwrap();
    writeln!(ptx, "    .reg .b64 %{desc_reg};").unwrap();
    writeln!(ptx, "    .reg .u32 %wgmma_desc_lo, %wgmma_desc_hi;").unwrap();
    // Base address: shift right 4 to get 16-byte units
    writeln!(ptx, "    shr.u32 %wgmma_desc_lo, {smem_base_expr}, 4;  // base in 16B units").unwrap();
    writeln!(ptx, "    and.b32 %wgmma_desc_lo, %wgmma_desc_lo, 0x3FFF;  // 14-bit base").unwrap();
    // Swizzle mode in bits [15:14]
    writeln!(ptx, "    or.b32 %wgmma_desc_lo, %wgmma_desc_lo, {};  // swizzle mode",
        swizzle_mode << 14).unwrap();
    // Leading dimension in 16-byte units, bits [29:16]
    let ld_16b = leading_dim_bytes / 16;
    writeln!(ptx, "    or.b32 %wgmma_desc_lo, %wgmma_desc_lo, {};  // leading dim in 16B units",
        (ld_16b as u32) << 16).unwrap();
    writeln!(ptx, "    mov.u32 %wgmma_desc_hi, 0;  // reserved upper 32 bits").unwrap();
    writeln!(ptx, "    mov.b64 %{desc_reg}, {{%wgmma_desc_lo, %wgmma_desc_hi}};").unwrap();
}

/// Emit the Q@K^T matmul using wgmma.mma_async for Hopper (sm_90).
///
/// wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 uses 128-thread warp groups.
/// Both A (Q) and B (K^T) are read from shared memory via descriptors.
/// The async execution allows overlapping softmax scalar math between
/// commit_group and wait_group.
#[allow(dead_code)]
fn emit_qk_matmul_wgmma(
    ptx: &mut String,
    _block_q: usize,
    block_kv: usize,
    head_dim: usize,
    _shmem_k_offset: usize,
) {
    use std::fmt::Write;

    let n_tiles = block_kv / WGMMA_N;
    let k_iters = head_dim / WGMMA_K_F16;

    writeln!(ptx, "    // === Q@K^T via wgmma.mma_async (m64n64k16, sm_90) ===").unwrap();
    writeln!(ptx, "    // 128-thread warp group, async execution").unwrap();
    writeln!(ptx, "    // n_tiles={n_tiles}, k_iters={k_iters}").unwrap();

    // wgmma accumulator registers: 32 f32 per thread per tile
    for nt in 0..n_tiles {
        for r in 0..32 {
            writeln!(ptx, "    .reg .f32 %wg_acc_s_{nt}_{r};").unwrap();
        }
    }
    writeln!(ptx, "    .reg .u32 %wg_k_iter;").unwrap();
    writeln!(ptx, "    .reg .pred %wg_pk;").unwrap();

    // Zero accumulators
    for nt in 0..n_tiles {
        for r in 0..32 {
            writeln!(ptx, "    mov.f32 %wg_acc_s_{nt}_{r}, 0.0;").unwrap();
        }
    }

    // K-dimension loop
    writeln!(ptx, "    mov.u32 %wg_k_iter, 0;").unwrap();
    writeln!(ptx, "QK_WGMMA_K_LOOP:").unwrap();

    // Issue wgmma for each n-tile
    for nt in 0..n_tiles {
        // wgmma instruction (f16 inputs from shared memory, f32 accumulators)
        writeln!(ptx, "    wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16").unwrap();
        // Accumulators (32 per tile)
        let acc_list: Vec<String> = (0..32).map(|r| format!("%wg_acc_s_{nt}_{r}")).collect();
        writeln!(ptx, "        {{{}}},", acc_list.join(", ")).unwrap();
        // A descriptor (Q tile in shared memory)
        writeln!(ptx, "        [%wg_desc_q],").unwrap();
        // B descriptor (K^T tile in shared memory)
        writeln!(ptx, "        [%wg_desc_kt_{nt}],").unwrap();
        // Scale/negate flags: p=0 (no scale A), q=0 (no scale B), r=0, s=0
        writeln!(ptx, "        0, 0, 0, 0;").unwrap();
    }

    // Commit the wgmma group (allows async execution)
    writeln!(ptx, "    wgmma.commit_group.sync.aligned;").unwrap();
    writeln!(ptx).unwrap();
    writeln!(ptx, "    // --- Softmax scalar math can overlap here (async!) ---").unwrap();
    writeln!(ptx).unwrap();

    // Wait for wgmma to complete before consuming accumulators
    writeln!(ptx, "    wgmma.wait_group.sync.aligned 0;").unwrap();

    // K-loop advancement
    writeln!(ptx, "    add.u32 %wg_k_iter, %wg_k_iter, 1;").unwrap();
    writeln!(ptx, "    setp.lt.u32 %wg_pk, %wg_k_iter, {k_iters};").unwrap();
    writeln!(ptx, "    @%wg_pk bra QK_WGMMA_K_LOOP;").unwrap();

    // Scale S by 1/sqrt(head_dim)
    writeln!(ptx, "    // Scale S accumulators").unwrap();
    for nt in 0..n_tiles {
        for r in 0..32 {
            writeln!(ptx, "    mul.f32 %wg_acc_s_{nt}_{r}, %wg_acc_s_{nt}_{r}, %scale;").unwrap();
        }
    }
}

/// Emit the P@V matmul using wgmma.mma_async for Hopper.
///
/// P (softmax output) must be staged to shared memory first since wgmma
/// requires shared memory inputs (unlike mma.sync which can use registers).
#[allow(dead_code)]
fn emit_pv_matmul_wgmma(
    ptx: &mut String,
    block_kv: usize,
    head_dim: usize,
    _shmem_k_offset: usize,
) {
    use std::fmt::Write;

    let n_tiles = head_dim / WGMMA_N;
    let k_iters = block_kv / WGMMA_K_F16;

    writeln!(ptx, "    // === P@V via wgmma.mma_async (m64n64k16, sm_90) ===").unwrap();
    writeln!(ptx, "    // P staged to shared memory, V already in shared memory").unwrap();

    // O accumulators (persist across KV-tile iterations)
    for nt in 0..n_tiles {
        for r in 0..32 {
            writeln!(ptx, "    .reg .f32 %wg_acc_o_{nt}_{r};").unwrap();
        }
    }

    // P@V K-dimension loop
    writeln!(ptx, "    mov.u32 %wg_k_iter, 0;").unwrap();
    writeln!(ptx, "PV_WGMMA_K_LOOP:").unwrap();

    for nt in 0..n_tiles {
        writeln!(ptx, "    wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16").unwrap();
        let acc_list: Vec<String> = (0..32).map(|r| format!("%wg_acc_o_{nt}_{r}")).collect();
        writeln!(ptx, "        {{{}}},", acc_list.join(", ")).unwrap();
        writeln!(ptx, "        [%wg_desc_p],").unwrap();
        writeln!(ptx, "        [%wg_desc_v_{nt}],").unwrap();
        writeln!(ptx, "        0, 0, 0, 0;").unwrap();
    }

    writeln!(ptx, "    wgmma.commit_group.sync.aligned;").unwrap();
    writeln!(ptx, "    wgmma.wait_group.sync.aligned 0;").unwrap();

    writeln!(ptx, "    add.u32 %wg_k_iter, %wg_k_iter, 1;").unwrap();
    writeln!(ptx, "    setp.lt.u32 %wg_pk, %wg_k_iter, {k_iters};").unwrap();
    writeln!(ptx, "    @%wg_pk bra PV_WGMMA_K_LOOP;").unwrap();
}

/// Emit wgmma register declarations (descriptor registers, accumulators).
#[allow(dead_code)]
fn emit_wgmma_registers(ptx: &mut String, block_kv: usize, head_dim: usize) {
    use std::fmt::Write;

    let n_tiles_s = block_kv / WGMMA_N;
    let n_tiles_o = head_dim / WGMMA_N;

    writeln!(ptx, "    // wgmma descriptor registers").unwrap();
    writeln!(ptx, "    .reg .b64 %wg_desc_q;  // Q tile descriptor").unwrap();
    for nt in 0..n_tiles_s {
        writeln!(ptx, "    .reg .b64 %wg_desc_kt_{nt};  // K^T tile {nt} descriptor").unwrap();
    }
    writeln!(ptx, "    .reg .b64 %wg_desc_p;  // P tile descriptor (softmax output)").unwrap();
    for nt in 0..n_tiles_o {
        writeln!(ptx, "    .reg .b64 %wg_desc_v_{nt};  // V tile {nt} descriptor").unwrap();
    }
}

/// Emit MMA register declarations needed by the fragment load and MMA helpers.
/// These are shared temporaries — the actual accumulator registers are declared
/// separately based on the tiling configuration.
#[allow(dead_code)]
fn emit_mma_temp_registers(ptx: &mut String) {
    ptx.push_str("    // MMA temporary registers\n");
    ptx.push_str("    .reg .f16 %mma_h0, %mma_h1;       // f32→f16 conversion temps\n");
    ptx.push_str("    .reg .u32 %mma_a_row, %mma_b_row;  // fragment row indices\n");
    ptx.push_str("    .reg .u32 %mma_addr;                // shared memory address temp\n");
    ptx.push_str("    .reg .u32 %mma_laneid;              // warp lane ID\n");
    // Compute laneid from tid.x (assuming warp of 32)
    ptx.push_str("    mov.u32 %mma_laneid, %tid.x;\n");
    ptx.push_str("    and.b32 %mma_laneid, %mma_laneid, 31;  // laneid = tid.x % 32\n");
    // A-fragment row mapping: row = (laneid % 4) * 2 + (laneid / 16)
    ptx.push_str("    and.b32 %mma_a_row, %mma_laneid, 3;    // laneid % 4\n");
    ptx.push_str("    shl.b32 %mma_a_row, %mma_a_row, 1;     // * 2\n");
    ptx.push_str("    shr.u32 %mma_addr, %mma_laneid, 4;     // laneid / 16 (reuse mma_addr as temp)\n");
    ptx.push_str("    add.u32 %mma_a_row, %mma_a_row, %mma_addr;  // row = (laneid%4)*2 + laneid/16\n");
    // B-fragment row mapping: same as A row for the k-dimension
    ptx.push_str("    mov.u32 %mma_b_row, %mma_a_row;        // B row mapping matches A for k-dim\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── MMA tile validation tests ─────────────────────────────────────

    #[test]
    fn test_mma_tile_validation_ok() {
        assert!(validate_mma_tile_sizes(64, 64, 64).is_ok());
        assert!(validate_mma_tile_sizes(128, 64, 128).is_ok());
        assert!(validate_mma_tile_sizes(16, 8, 16).is_ok()); // minimum MMA tile
    }

    #[test]
    fn test_mma_tile_validation_block_q_not_aligned() {
        let err = validate_mma_tile_sizes(17, 64, 64).unwrap_err();
        assert!(err.contains("block_q"), "{}", err);
        assert!(err.contains("MMA_M"), "{}", err);
    }

    #[test]
    fn test_mma_tile_validation_block_kv_not_aligned() {
        let err = validate_mma_tile_sizes(64, 7, 64).unwrap_err();
        assert!(err.contains("block_kv"), "{}", err);
    }

    #[test]
    fn test_mma_tile_validation_head_dim_not_aligned() {
        let err = validate_mma_tile_sizes(64, 64, 65).unwrap_err();
        assert!(err.contains("head_dim"), "{}", err);
    }

    #[test]
    fn test_use_mma_path() {
        assert!(!use_mma_path(52));  // Kepler
        assert!(!use_mma_path(70));  // Volta (has wmma but not f16 mma.sync)
        assert!(use_mma_path(80));   // Ampere
        assert!(use_mma_path(89));   // Ada Lovelace
        assert!(use_mma_path(90));   // Hopper
    }

    // ── MMA PTX emission tests ─────────────────────────────────────

    #[test]
    fn test_f32_to_f16_pack_emission() {
        let mut ptx = String::new();
        let src = vec!["f0".to_string(), "f1".to_string(), "f2".to_string(), "f3".to_string()];
        let dst = vec!["a0".to_string(), "a1".to_string()];
        emit_f32_to_f16_pack(&mut ptx, &src, &dst);

        assert!(ptx.contains("cvt.rn.f16.f32 %mma_h0, %f0"), "first lo conversion");
        assert!(ptx.contains("cvt.rn.f16.f32 %mma_h1, %f1"), "first hi conversion");
        assert!(ptx.contains("mov.b32 %a0, {%mma_h0, %mma_h1}"), "first pack");
        assert!(ptx.contains("cvt.rn.f16.f32 %mma_h0, %f2"), "second lo conversion");
        assert!(ptx.contains("mov.b32 %a1, {%mma_h0, %mma_h1}"), "second pack");
    }

    #[test]
    fn test_mma_instruction_emission() {
        let mut ptx = String::new();
        let d = ["d0".into(), "d1".into(), "d2".into(), "d3".into()];
        let a = ["a0".into(), "a1".into(), "a2".into(), "a3".into()];
        let b = ["b0".into(), "b1".into()];
        let c = ["c0".into(), "c1".into(), "c2".into(), "c3".into()];
        emit_mma_instruction(&mut ptx, &d, &a, &b, &c);

        assert!(ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"),
            "must contain MMA instruction");
        assert!(ptx.contains("{d0, d1, d2, d3}"), "D accumulator regs");
        assert!(ptx.contains("{a0, a1, a2, a3}"), "A fragment regs");
        assert!(ptx.contains("{b0, b1}"), "B fragment regs");
        assert!(ptx.contains("{c0, c1, c2, c3}"), "C accumulator regs");
    }

    // ── wgmma (Hopper sm_90) tests ──────────────────────────────────

    #[test]
    fn test_wgmma_smem_descriptor_emission() {
        let mut ptx = String::new();
        emit_wgmma_smem_descriptor(&mut ptx, "desc_q", "%smem_q_base", 128, 3);

        assert!(ptx.contains("wgmma shared memory descriptor"), "comment present");
        assert!(ptx.contains(".reg .b64 %desc_q"), "descriptor register declared");
        assert!(ptx.contains("shr.u32"), "base address shift");
        assert!(ptx.contains("0x3FFF"), "14-bit mask for base");
    }

    #[test]
    fn test_qk_matmul_wgmma_emission() {
        let mut ptx = String::new();
        emit_qk_matmul_wgmma(&mut ptx, 64, 64, 64, 64 * 64 * 4);

        assert!(ptx.contains("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16"),
            "wgmma instruction present");
        assert!(ptx.contains("wgmma.commit_group.sync.aligned"), "commit present");
        assert!(ptx.contains("wgmma.wait_group.sync.aligned"), "wait present");
        assert!(ptx.contains("Softmax scalar math can overlap"), "async overlap comment");
        assert!(ptx.contains("QK_WGMMA_K_LOOP"), "K loop label");
        assert!(ptx.contains("%wg_acc_s_"), "S accumulator registers");
        assert!(ptx.contains("%wg_desc_q"), "Q descriptor");
        assert!(ptx.contains("%wg_desc_kt_"), "K^T descriptor");
        // 64/64 = 1 n-tile, so 1 wgmma instruction per K iteration
        let wgmma_count = ptx.matches("wgmma.mma_async.sync").count();
        assert_eq!(wgmma_count, 1, "1 wgmma per n-tile");
    }

    #[test]
    fn test_pv_matmul_wgmma_emission() {
        let mut ptx = String::new();
        emit_pv_matmul_wgmma(&mut ptx, 64, 64, 64 * 64 * 4);

        assert!(ptx.contains("wgmma.mma_async.sync.aligned"), "wgmma present");
        assert!(ptx.contains("%wg_acc_o_"), "O accumulator registers");
        assert!(ptx.contains("%wg_desc_p"), "P descriptor");
        assert!(ptx.contains("%wg_desc_v_"), "V descriptor");
        assert!(ptx.contains("PV_WGMMA_K_LOOP"), "K loop label");
    }

    #[test]
    fn test_wgmma_registers_emission() {
        let mut ptx = String::new();
        emit_wgmma_registers(&mut ptx, 64, 64);

        assert!(ptx.contains("%wg_desc_q"), "Q descriptor register");
        assert!(ptx.contains("%wg_desc_kt_0"), "K^T descriptor for tile 0");
        assert!(ptx.contains("%wg_desc_p"), "P descriptor register");
        assert!(ptx.contains("%wg_desc_v_0"), "V descriptor for tile 0");
    }

    #[test]
    fn test_wgmma_constants() {
        assert_eq!(WGMMA_M, 64, "wgmma tile M = 64");
        assert_eq!(WGMMA_N, 64, "wgmma tile N = 64");
        assert_eq!(WGMMA_K_F16, 16, "wgmma K for f16 = 16");
    }

    #[test]
    fn test_wgmma_full_pipeline() {
        let mut ptx = String::new();

        // Register declarations
        emit_wgmma_registers(&mut ptx, 64, 64);

        // Q@K^T via wgmma
        emit_qk_matmul_wgmma(&mut ptx, 64, 64, 64, 64 * 64 * 4);

        // P@V via wgmma
        emit_pv_matmul_wgmma(&mut ptx, 64, 64, 64 * 64 * 4);

        // Both matmuls should use wgmma
        let wgmma_count = ptx.matches("wgmma.mma_async.sync").count();
        assert_eq!(wgmma_count, 2, "Q@K^T + P@V = 2 wgmma instructions");

        // Both should have commit/wait pairs
        let commit_count = ptx.matches("wgmma.commit_group").count();
        let wait_count = ptx.matches("wgmma.wait_group").count();
        assert_eq!(commit_count, 2, "2 commit groups");
        assert_eq!(wait_count, 2, "2 wait groups");
    }

    // ── mma.sync (Ampere sm_80) tests ─────────────────────────────────

    #[test]
    fn test_load_a_fragment_emission() {
        let mut ptx = String::new();
        let regs: [String; 4] = ["aq0".into(), "aq1".into(), "aq2".into(), "aq3".into()];
        emit_load_a_fragment_smem(&mut ptx, &regs, "shmem_q", 256);

        assert!(ptx.contains("Load A-fragment"), "comment present");
        assert!(ptx.contains("ld.shared.b32 %aq0"), "loads first register");
        assert!(ptx.contains("ld.shared.b32 %aq3"), "loads last register");
        // Should have 4 load instructions
        assert_eq!(ptx.matches("ld.shared.b32").count(), 4, "4 fragment loads");
    }

    #[test]
    fn test_load_b_fragment_emission() {
        let mut ptx = String::new();
        let regs: [String; 2] = ["bk0".into(), "bk1".into()];
        emit_load_b_fragment_smem(&mut ptx, &regs, "shmem_k", 128);

        assert!(ptx.contains("Load B-fragment"), "comment present");
        assert_eq!(ptx.matches("ld.shared.b32").count(), 2, "2 fragment loads");
    }

    #[test]
    fn test_qk_mma_ptx_emission() {
        let mut ptx = String::new();
        // block_q=64, block_kv=64, head_dim=64
        emit_qk_matmul_mma(&mut ptx, 64, 64, 64, 64 * 64 * 4);

        assert!(ptx.contains("Q@K^T via MMA"), "section comment");
        assert!(ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"),
            "MMA instruction present");
        assert!(ptx.contains("cvt.rn.f16.f32"), "f32->f16 conversion present");
        assert!(ptx.contains("%acc_s_"), "S accumulator registers used");
        assert!(ptx.contains("%aq_"), "A fragment registers used");
        assert!(ptx.contains("%bk_"), "B fragment registers used");
        assert!(ptx.contains("QK_MMA_K_LOOP"), "K-dimension loop label");
        assert!(ptx.contains("mul.f32 %acc_s_"), "scale application");
        // n_tiles_s = 64/8 = 8, so we should have MMA for tiles 0-7
        assert!(ptx.contains("%acc_s_7_3"), "should have n_tile=7");
        // Each n-tile gets one MMA instruction per K iteration
        let mma_count = ptx.matches("mma.sync.aligned").count();
        assert_eq!(mma_count, 8, "8 MMA instructions (one per n-tile)");
    }

    #[test]
    fn test_mma_online_softmax_emission() {
        let mut ptx = String::new();
        emit_mma_online_softmax(&mut ptx, 64, 64);

        assert!(ptx.contains("Online softmax (MMA layout)"), "section comment");
        // Row max reduction
        assert!(ptx.contains("max.f32 %mma_local_max"), "local max computation");
        assert!(ptx.contains("shfl.sync.bfly.b32"), "warp shuffle present");
        // Correction factor
        assert!(ptx.contains("ex2.approx.f32 %mma_correction"), "exp correction");
        // O rescaling
        assert!(ptx.contains("mul.f32 %acc_o_"), "O accumulator rescaling");
        // P = exp(S - max)
        assert!(ptx.contains("ex2.approx.f32 %acc_s_"), "P computation");
        // Row sum
        assert!(ptx.contains("add.f32 %mma_row_sum"), "row sum accumulation");
    }

    #[test]
    fn test_smem_swizzle_emission() {
        let mut ptx = String::new();
        emit_smem_swizzle_store(&mut ptx);

        assert!(ptx.contains("XOR swizzle"), "comment present");
        assert!(ptx.contains("xor.b32 %smem_swiz"), "XOR instruction");
        assert!(ptx.contains("%smem_swiz_off"), "output register");
    }

    #[test]
    fn test_smem_swizzle_offset_no_self_conflict() {
        // Verify that swizzled offsets for consecutive rows don't collide on same bank
        for row in 0..16 {
            let off1 = swizzle_smem_offset(row, 0);
            let off2 = swizzle_smem_offset(row + 1, 0);
            let bank1 = (off1 / 4) % 32;
            let bank2 = (off2 / 4) % 32;
            // Adjacent rows should map to different banks (or same bank is ok if col differs)
            // At minimum, swizzle should not map everything to the same bank
            if row > 0 {
                // Not all banks should be identical
                let bank_prev = (swizzle_smem_offset(row - 1, 0) / 4) % 32;
                assert!(bank1 != bank_prev || bank2 != bank1,
                    "three consecutive rows hit same bank: row={}, banks={},{},{}", row-1, bank_prev, bank1, bank2);
            }
        }
    }

    #[test]
    fn test_mma_softmax_registers_emission() {
        let mut ptx = String::new();
        emit_mma_softmax_registers(&mut ptx);

        assert!(ptx.contains("%mma_row_max"), "row_max declared");
        assert!(ptx.contains("%mma_row_sum"), "row_sum declared");
        assert!(ptx.contains("0xFF800000"), "row_max initialized to -inf");
        assert!(ptx.contains("0x00000000"), "row_sum initialized to 0");
    }

    #[test]
    fn test_mma_full_pipeline_ptx_has_all_components() {
        // Generate a complete MMA pipeline and verify all key components present
        let mut ptx = String::new();

        // Register declarations
        emit_mma_qk_registers(&mut ptx, 64, 64);
        emit_mma_softmax_registers(&mut ptx);
        emit_mma_temp_registers(&mut ptx);
        emit_smem_swizzle_registers(&mut ptx);

        // Q@K^T
        ptx.push_str("    mov.u32 %mma_m_tile_byte_offset, 0;\n");
        emit_qk_matmul_mma(&mut ptx, 64, 64, 64, 64 * 64 * 4);

        // Online softmax
        emit_mma_online_softmax(&mut ptx, 64, 64);

        // P@V
        emit_pv_matmul_mma(&mut ptx, 64, 64, 64 * 64 * 4);

        // Verify all key MMA components are present
        let mma_count = ptx.matches("mma.sync.aligned.m16n8k16").count();
        // Q@K^T: n_tiles_s=8, P@V: n_tiles_o=8 → 16 total
        assert_eq!(mma_count, 16,
            "expected 16 MMA instructions (8 Q@K^T + 8 P@V), got {}", mma_count);

        assert!(ptx.contains("cvt.rn.f16.f32"), "f32→f16 conversion");
        assert!(ptx.contains("shfl.sync.bfly"), "warp shuffles for softmax");
        assert!(ptx.contains("ex2.approx.f32"), "exp via ex2");
        assert!(ptx.contains("%mma_row_max"), "softmax row max");
        assert!(ptx.contains("%acc_s_"), "S accumulators");
        assert!(ptx.contains("%acc_o_"), "O accumulators");
    }

    #[test]
    fn test_pv_mma_ptx_emission() {
        let mut ptx = String::new();
        emit_pv_matmul_mma(&mut ptx, 64, 64, 64 * 64 * 4);

        assert!(ptx.contains("P@V via MMA"), "section comment");
        assert!(ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"),
            "MMA instruction present");
        assert!(ptx.contains("%acc_o_"), "O accumulator registers used");
        assert!(ptx.contains("%ap_"), "P A-fragment registers used");
        assert!(ptx.contains("%bv_"), "V B-fragment registers used");
        assert!(ptx.contains("PV_MMA_K_LOOP"), "K-dimension loop label");
        // n_tiles_o = 64/8 = 8
        let mma_count = ptx.matches("mma.sync.aligned").count();
        assert_eq!(mma_count, 8, "8 MMA instructions (one per output n-tile)");
    }

    #[test]
    fn test_mma_qk_registers_emission() {
        let mut ptx = String::new();
        emit_mma_qk_registers(&mut ptx, 64, 64);

        // n_tiles_s = 64/8 = 8 S accumulators
        assert!(ptx.contains("%acc_s_7_3"), "S acc for n_tile=7 reg 3");
        // n_tiles_o = 64/8 = 8 O accumulators
        assert!(ptx.contains("%acc_o_7_3"), "O acc for n_tile=7 reg 3");
        // Fragment registers
        assert!(ptx.contains("%aq_0"), "A fragment for Q");
        assert!(ptx.contains("%bk_7_1"), "B fragment for K n_tile=7");
        assert!(ptx.contains("%bv_7_1"), "B fragment for V n_tile=7");
        assert!(ptx.contains("%mma_k_iter"), "K loop counter");
    }

    #[test]
    fn test_mma_temp_registers_emission() {
        let mut ptx = String::new();
        emit_mma_temp_registers(&mut ptx);

        assert!(ptx.contains(".reg .f16 %mma_h0, %mma_h1"), "f16 temps declared");
        assert!(ptx.contains(".reg .u32 %mma_a_row"), "A row register declared");
        assert!(ptx.contains("%mma_laneid"), "laneid register used");
        assert!(ptx.contains("and.b32 %mma_laneid, %mma_laneid, 31"), "laneid = tid.x % 32");
    }

    // ── Existing tests ────────────────────────────────────────────────

    #[test]
    fn test_kernel_name_encoding() {
        let config = FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 128,
            causal: true,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: 80,
        };
        assert_eq!(
            flash_attention_kernel_name(&config),
            "flash_attn_p0_r0_hs_g1_c1_t0_q64_kv64"
        );
    }

    #[test]
    fn test_kernel_name_full_variant() {
        let config = FlashAttentionConfig {
            block_q: 128,
            block_kv: 32,
            head_dim: 64,
            causal: true,
            paged: true,
            rope_q: true,
            rope_style: RopeStyle::Adjacent,
            gqa_group_size: 4,
            tree_mask: false,
            gpu_sm: 80,
        };
        assert_eq!(
            flash_attention_kernel_name(&config),
            "flash_attn_p1_r1_adj_g4_c1_t0_q128_kv32"
        );
    }

    #[test]
    fn test_shared_mem_bytes_computation() {
        let config = FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 128,
            causal: true,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: 80,
        };
        // (64 + 64) * 128 * 2 = 32768 bytes (32 KB)
        assert_eq!(shared_mem_bytes(&config), 32768);
    }

    #[test]
    fn test_shared_mem_within_48kb_limit() {
        // block_q=128, block_kv=64, head_dim=128
        // (128 + 64) * 128 * 2 = 49152 > 48KB → would exceed sm_52 default
        let config = FlashAttentionConfig {
            block_q: 128,
            block_kv: 64,
            head_dim: 128,
            causal: true,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: 80,
        };
        assert_eq!(shared_mem_bytes(&config), 49152);
        // This exceeds 48KB — the semantic checker should reject this combination
    }

    #[test]
    fn test_ptx_synthesis_produces_valid_header() {
        let config = FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 128,
            causal: true,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: 80,
        };
        let ptx = synthesize_flash_attention_ptx(&config);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len()-1]).unwrap(); // strip null
        assert!(ptx_str.starts_with(".version 7.0\n"));
        assert!(ptx_str.contains(".target sm_80"));
        assert!(ptx_str.contains(&flash_attention_kernel_name(&config)));
        assert!(ptx_str.contains("bar.sync 0"));
        assert!(ptx_str.contains("FENCE 1"));
        assert!(ptx_str.contains("FENCE 2"));
        assert!(ptx_str.contains("FENCE 3"));
    }

    #[test]
    fn test_ptx_causal_flag() {
        let mut config = FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 128,
            causal: false,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: 80,
        };
        let ptx_no_causal = synthesize_flash_attention_ptx(&config);
        let str_no = std::str::from_utf8(&ptx_no_causal[..ptx_no_causal.len()-1]).unwrap();
        assert!(str_no.contains("Non-causal"));
        assert!(!str_no.contains("Zero-divergence"));

        config.causal = true;
        let ptx_causal = synthesize_flash_attention_ptx(&config);
        let str_c = std::str::from_utf8(&ptx_causal[..ptx_causal.len()-1]).unwrap();
        assert!(str_c.contains("Zero-divergence"));
    }

    #[test]
    fn test_ptx_paged_variant() {
        let config = FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 128,
            causal: true,
            paged: true,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: 80,
        };
        let ptx = synthesize_flash_attention_ptx(&config);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len()-1]).unwrap();
        assert!(ptx_str.contains("block_table_ptr"));
        assert!(ptx_str.contains("k_pool_ptr"));
        assert!(ptx_str.contains("v_pool_ptr"));
    }

    #[test]
    fn test_ptx_rope_variant() {
        let config = FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 128,
            causal: true,
            paged: false,
            rope_q: true,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: 80,
        };
        let ptx = synthesize_flash_attention_ptx(&config);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len()-1]).unwrap();
        assert!(ptx_str.contains("cos_ptr"));
        assert!(ptx_str.contains("sin_ptr"));
        assert!(ptx_str.contains("half_split"));
    }

    #[test]
    fn test_ptx_gqa_variant() {
        let config = FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 128,
            causal: true,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 4,
            tree_mask: false,
            gpu_sm: 80,
        };
        let ptx = synthesize_flash_attention_ptx(&config);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len()-1]).unwrap();
        assert!(ptx_str.contains("kv_head = q_head / 4"));
    }

    #[test]
    fn test_rope_cache_write_ptx() {
        let ptx = synthesize_rope_cache_write_ptx(128, RopeStyle::HalfSplit);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len()-1]).unwrap();
        assert!(ptx_str.contains("nsl_rope_cache_write"));
        assert!(ptx_str.contains("seq_ids_ptr"));
        assert!(ptx_str.contains("seq_lens_ptr"));
        assert!(ptx_str.contains("half_split"));
    }

    #[test]
    fn test_rope_cache_write_adjacent() {
        let ptx = synthesize_rope_cache_write_ptx(128, RopeStyle::Adjacent);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len()-1]).unwrap();
        assert!(ptx_str.contains("adjacent"));
    }

    #[test]
    fn test_tree_mask_variant() {
        let config = FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 128,
            causal: false,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: true,
            gpu_sm: 80,
        };
        let ptx = synthesize_flash_attention_ptx(&config);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len()-1]).unwrap();
        // Verify tree mask params and DFS ancestor check are present
        assert!(ptx_str.contains("dfs_enter_ptr"), "missing dfs_enter_ptr param");
        assert!(ptx_str.contains("dfs_exit_ptr"), "missing dfs_exit_ptr param");
        assert!(ptx_str.contains("num_tree_nodes"), "missing num_tree_nodes param");
        assert!(ptx_str.contains("dfs_enter_base"), "missing DFS register load");
        assert!(ptx_str.contains("Tree mask"), "missing tree mask comment");
        // Verify tree_mask=true kernel name includes t1
        let name = flash_attention_kernel_name(&config);
        assert!(name.contains("_t1_"), "kernel name should contain _t1_ for tree_mask=true, got {}", name);
    }
}
