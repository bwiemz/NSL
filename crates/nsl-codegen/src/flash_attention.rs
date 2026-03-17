//! FlashAttention-2 PTX template synthesis.
//!
//! Generates PTX kernel strings at compile time (AOT). Each variant is parameterized
//! by orthogonal flags (paged, rope_q, rope_style, gqa_group_size, causal) and tile
//! sizes (block_q, block_kv). The generated PTX is embedded in .rodata and launched
//! by the runtime wrappers in nsl-runtime/src/flash_attention.rs.

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
}

/// Generate PTX for the FlashAttention-2 kernel with the given configuration.
///
/// Returns null-terminated PTX bytes ready for .rodata embedding.
pub fn synthesize_flash_attention_ptx(config: &FlashAttentionConfig) -> Vec<u8> {
    let mut ptx = String::with_capacity(8192);
    let kernel_name = flash_attention_kernel_name(config);

    // PTX header
    emit_ptx_header(&mut ptx);

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

    emit_ptx_header(&mut ptx);
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

fn emit_ptx_header(ptx: &mut String) {
    ptx.push_str(".version 7.0\n");
    ptx.push_str(".target sm_52\n");
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

#[cfg(test)]
mod tests {
    use super::*;

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
        };
        let ptx = synthesize_flash_attention_ptx(&config);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len()-1]).unwrap(); // strip null
        assert!(ptx_str.starts_with(".version 7.0\n"));
        assert!(ptx_str.contains(".target sm_52"));
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
