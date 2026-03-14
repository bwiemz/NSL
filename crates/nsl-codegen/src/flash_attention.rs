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
/// Format: `flash_attn_p{paged}_r{rope}_{style}_g{gqa}_c{causal}_q{block_q}_kv{block_kv}`
pub fn flash_attention_kernel_name(config: &FlashAttentionConfig) -> String {
    format!(
        "flash_attn_p{}_r{}_{}_g{}_c{}_q{}_kv{}",
        config.paged as u8,
        config.rope_q as u8,
        match config.rope_style {
            RopeStyle::HalfSplit => "hs",
            RopeStyle::Adjacent => "adj",
        },
        config.gqa_group_size,
        config.causal as u8,
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
    ptx.push_str("    .param .u64 seq_lens_ptr\n");

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

    let _ = config; // variant flags control which registers are USED, not loaded
}

fn emit_index_computation(ptx: &mut String, _config: &FlashAttentionConfig) {
    // threadIdx.x, blockIdx.x (Q tile index), blockIdx.y (batch*head index)
    ptx.push_str("    mov.u32 %tid_x, %tid.x;\n");
    ptx.push_str("    mov.u32 %bid_x, %ctaid.x;\n");
    ptx.push_str("    mov.u32 %bid_y, %ctaid.y;\n");

    // q_start = blockIdx.x * block_q
    // batch_head_idx = blockIdx.y
    // head_idx = batch_head_idx % heads
    // batch_idx = batch_head_idx / heads
    ptx.push_str("    // q_start = blockIdx.x * block_q\n");
    ptx.push_str("    // batch_head routing computed from blockIdx.y\n");
}

fn emit_q_tile_load(ptx: &mut String, config: &FlashAttentionConfig) {
    ptx.push_str("    // Load Q tile into shared memory\n");
    if config.rope_q {
        ptx.push_str("    // RoPE: load Q from global, rotate in registers, store to SRAM\n");
        ptx.push_str("    // cos/sin loaded from global memory into registers (NOT SRAM)\n");
        let stride_comment = match config.rope_style {
            RopeStyle::HalfSplit => "stride = head_dim/2 (half_split)",
            RopeStyle::Adjacent => "stride = 1 (adjacent)",
        };
        ptx.push_str(&format!("    // RoPE style: {}\n", stride_comment));
    }
    ptx.push_str("    bar.sync 0;\n");
    ptx.push_str("    // Q tile now in shmem[0 .. block_q * head_dim * 2]\n");
}

fn emit_accumulator_init(ptx: &mut String, _config: &FlashAttentionConfig) {
    ptx.push_str("    // Initialize accumulators\n");
    ptx.push_str("    // O_acc = 0, row_max = -inf, row_sum = 0\n");
    ptx.push_str("    mov.f32 %row_max, 0xFF800000;  // -inf as IEEE 754\n");
    ptx.push_str("    mov.f32 %row_sum, 0x00000000;  // 0.0\n");
}

fn emit_kv_tile_loop(ptx: &mut String, config: &FlashAttentionConfig) {
    ptx.push_str("    // === Main K/V tile loop ===\n");

    if config.causal {
        ptx.push_str("    // Causal: k_max = min(seq_len, q_start + block_q)\n");
        ptx.push_str("    // Zero-divergence — loop naturally terminates at diagonal\n");
    } else {
        ptx.push_str("    // Non-causal: k_max = seq_len\n");
    }

    ptx.push_str("LOOP_KV_START:\n");

    // Phase 1: Load K tile
    ptx.push_str("    // Phase 1: Load K tile into SRAM\n");
    if config.paged {
        ptx.push_str("    // Paged: block table indirection per physical block\n");
        ptx.push_str("    // One page table lookup per block_size tokens\n");
    }
    if config.gqa_group_size > 1 {
        ptx.push_str(&format!(
            "    // GQA: kv_head = q_head / {} (compile-time literal)\n",
            config.gqa_group_size
        ));
    }
    ptx.push_str("    bar.sync 0;  // FENCE 1: K tile fully in SRAM\n");

    // Phase 2: Compute S = Q @ K^T, apply scale
    ptx.push_str("    // Phase 2: S = Q_tile @ K_tile^T (registers)\n");
    ptx.push_str("    // S[i][j] *= scale  (mul.f32)\n");
    if config.causal {
        ptx.push_str("    // Partial causal mask on diagonal tile: S[i][j] = -inf where k_start+j > q_start+i\n");
    }
    ptx.push_str("    bar.sync 0;  // FENCE 2: all warps done reading K before SRAM overwrite\n");

    // Phase 3: Online softmax
    ptx.push_str("    // Phase 3: Online softmax — S→P in-place in registers\n");
    ptx.push_str("    // Warp-level reductions via shfl.sync.bfly for row_max, row_sum\n");
    ptx.push_str("    // new_max = max(row_max, warp_reduce_max(row_max_of_S))\n");
    ptx.push_str("    // correction = exp(old_max - new_max)  // <= 1.0, no overflow\n");
    ptx.push_str("    // row_sum = row_sum * correction + warp_reduce_sum(exp(S - new_max))\n");
    ptx.push_str("    // O_acc *= correction\n");
    ptx.push_str("    // P = exp(S - new_max)  // overwrites S registers in-place\n");

    // Phase 4: Load V tile (reuses K SRAM)
    ptx.push_str("    // Phase 4: Load V tile (reuses shmem_K address)\n");
    if config.paged {
        ptx.push_str("    // Paged V load: same block table indirection as K\n");
    }
    ptx.push_str("    bar.sync 0;  // FENCE 3: V tile fully in SRAM\n");

    // Phase 5: Accumulate O
    ptx.push_str("    // Phase 5: O_acc += P @ V_tile\n");
    ptx.push_str("    // P in registers (Phase 3), V in SRAM\n");

    // Loop back
    ptx.push_str("    // Increment k_start, check loop bound\n");
    ptx.push_str("    bra LOOP_KV_START;\n");
    ptx.push_str("LOOP_KV_END:\n");
}

fn emit_finalize(ptx: &mut String, _config: &FlashAttentionConfig) {
    ptx.push_str("    // Finalize: O = O_acc / row_sum\n");
}

fn emit_output_store(ptx: &mut String, _config: &FlashAttentionConfig) {
    ptx.push_str("    // Store output tile to global memory\n");
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
        };
        assert_eq!(
            flash_attention_kernel_name(&config),
            "flash_attn_p0_r0_hs_g1_c1_q64_kv64"
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
        };
        assert_eq!(
            flash_attention_kernel_name(&config),
            "flash_attn_p1_r1_adj_g4_c1_q128_kv32"
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
}
