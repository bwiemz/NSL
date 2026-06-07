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

// ---- PTX literal helpers (see docs/superpowers/specs/2026-04-13-fa-emitter-ptx-fix-design.md) ----

/// Format a u32 bit pattern as a PTX f32 literal: `0f########`.
/// PTX rejects `0x########` for f32 immediates.
fn f32_bits(bits: u32) -> String {
    format!("0f{:08X}", bits)
}

const F32_ZERO: u32    = 0x0000_0000;
const F32_NEG_INF: u32 = 0xFF80_0000;
const F32_LOG2E: u32   = 0x3FB8_AA3B;

/// Emit a shared-memory store through `%smem_addr` / `%shmem_base`.
/// Callers must have emitted `.reg .u64 %smem_addr` + `.reg .u64 %shmem_base`
/// and `cvta.shared.u64 %shmem_base, shmem;` in the kernel prolog
/// (see `emit_register_declarations`).
fn emit_smem_store(ptx: &mut String, ty: &str, offset_reg: &str, val_reg: &str) {
    use std::fmt::Write;
    writeln!(ptx, "    add.s64 %smem_addr, %shmem_base, {};", offset_reg).unwrap();
    writeln!(ptx, "    st.shared.{} [%smem_addr], {};", ty, val_reg).unwrap();
}

/// Emit a shared-memory load through `%smem_addr` / `%shmem_base`.
fn emit_smem_load(ptx: &mut String, ty: &str, dst_reg: &str, offset_reg: &str) {
    use std::fmt::Write;
    writeln!(ptx, "    add.s64 %smem_addr, %shmem_base, {};", offset_reg).unwrap();
    writeln!(ptx, "    ld.shared.{} {}, [%smem_addr];", ty, dst_reg).unwrap();
}

/// Emit an `atom.shared.add.f32` through `%smem_addr` / `%shmem_base`.
/// The `offset_reg` must be a `.u64` register (widen with `cvt.u64.u32` first if needed).
fn emit_smem_atom_add_f32(ptx: &mut String, dst_reg: &str, offset_reg: &str, val_reg: &str) {
    use std::fmt::Write;
    writeln!(ptx, "    add.s64 %smem_addr, %shmem_base, {};", offset_reg).unwrap();
    writeln!(ptx, "    atom.shared.add.f32 {}, [%smem_addr], {};", dst_reg, val_reg).unwrap();
}

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
    /// Paper §4.3: attention sinks. Number of always-attended initial
    /// tokens (typically 4) to stabilize streaming attention with rolling
    /// KV cache.
    ///
    /// v0 API-surface state (Sprint 2 cycle-4 + Sprint 2 cycle-5 refusal):
    /// the decorator + config + semantic validation ARE wired through,
    /// but the SMEM-cache codegen is DEFERRED to a future sprint. Configs
    /// with `num_sink_tokens > 0` are REFUSED at codegen with a clear
    /// error rather than silently producing rope-effectively-off output.
    /// The only value that flows through to PTX synthesis today is `0`
    /// (the sentinel "sinks disabled" state — users who want disabled
    /// simply omit the `@attention_sink` decorator entirely). When the
    /// SMEM-emission sprint lands, lift the refusal in
    /// `compiler/kernel.rs::attention_sink` and extend the integration
    /// test in `attention_sink_decorator_integration.rs` with a gated
    /// PTX probe.
    pub num_sink_tokens: u32,
    /// Target GPU SM version for PTX target selection (default: 52).
    pub gpu_sm: u32,
    /// PCA Tier A: when `true`, the emitter produces a segment-aware
    /// attention kernel that masks `S[i, j]` by
    /// `segment_ids[i] == segment_ids[j]` alongside the causal mask.
    /// Mutually exclusive with `paged: true` (spec §3.2).
    pub segment_masked: bool,
    /// CSHA (Compiler-Synthesized Holistic Attention) extensions.  `None`
    /// (the default) leaves the kernel functionally identical to classic
    /// FlashAttention-2.  `Some(..)` selects a CSHA-fused variant per
    /// `NSL-CSHA-Research.PDF`.
    #[doc(hidden)]
    pub csha: Option<CshaExtras>,
}

impl FlashAttentionConfig {
    /// Spec §3.2 invariant: segment_masked + paged are mutually
    /// exclusive; paged KV cache is inference-only, packed
    /// pretraining doesn't use it.
    pub fn validate(&self) -> Result<(), String> {
        if self.segment_masked && self.paged {
            return Err(
                "FlashAttentionConfig: segment_masked and paged are \
                 mutually exclusive (spec §3.2)"
                    .to_string(),
            );
        }
        Ok(())
    }
}

/// CSHA kernel-level fusion extensions.
///
/// Stored as an optional sub-struct so that adding CSHA doesn't force
/// the many existing `FlashAttentionConfig { .. }` construction sites
/// in the codebase to change.  Callers that don't want CSHA leave
/// `csha: None`.
#[derive(Clone, Debug, Default)]
pub struct CshaExtras {
    /// Fusion level this kernel realises (1=boundary, 2=pipeline,
    /// 3=block).  0 is invalid — use `FlashAttentionConfig { csha: None
    /// }` for "no CSHA" instead.
    pub level: u8,
    /// Prologue fusion: read raw `x` and compute `RMSNorm(x)` on tiles
    /// before projecting into Q/K/V (paper §2.1).  Requires
    /// `rmsnorm_eps` and passing the norm weight pointer at launch.
    pub fused_rmsnorm: bool,
    /// Pipelined fusion (paper §2.2): compute Q/K/V projections inside
    /// the attention kernel rather than reading pre-materialised
    /// tensors.  Requires `d_model` to size the projection-weight tile.
    pub fused_projections: bool,
    /// Epilogue fusion: apply `Attn_out @ Wo` before writing out
    /// (§2.2 end, §2.3).
    pub fused_output_proj: bool,
    /// Number of attention heads actually computed.  `0` means "use
    /// the full head count from the launch parameter" — i.e. no
    /// weight-informed pruning.  A non-zero value indicates the
    /// kernel has been specialised to skip pruned heads entirely
    /// (§3.1 / §5.2 dead-head-gradient-elimination).
    pub active_heads: u32,
    /// RMSNorm epsilon baked into the prologue.  Ignored if
    /// `fused_rmsnorm` is false.
    pub rmsnorm_eps: f32,
    /// `d_model` — feature dimension fed into the fused projections.
    /// Ignored unless `fused_projections` is true.
    pub d_model: u32,
    /// When true, forward writes Q_proj, K_proj, V_proj, row_max, row_sum
    /// to HBM for backward consumption. Gated on @train mode by the
    /// compiler. Inference builds leave this false; forward pays zero
    /// extra HBM cost.
    pub save_activations_for_backward: bool,
    /// **Tier B.1 only.** When true, `tier_b1::synthesize` omits the
    /// RMSNorm prologue (`csha_hooks::emit_prologue`) entirely and
    /// expects the caller to provide `csha_x_ptr` as already-normalized,
    /// already-narrowed-to-f16, already-chunkified data in the
    /// `[d_model/chunk, bq | bkv, chunk]` f16 chunks-major HBM layout
    /// the projection cp.async expects. Used by the N4 end-to-end
    /// validation test and by future CSHA pipelines that own the
    /// narrow+chunkify pre-pass externally. Default false (RMSNorm
    /// prologue runs as before; current callers see no behavior change).
    pub skip_rmsnorm_prologue: bool,
    /// **Sprint 8 (paper §4.1) — compile-time seq_len.**  When `Some(s)`,
    /// the Tier B.2 dq/dkdv emitters treat the sequence length as a
    /// compile-time constant and elide the runtime `ceil(seq_len / bq)` /
    /// `ceil(seq_len / bkv)` shift sequence in `emit_q_iter_count_setup` /
    /// `emit_kv_iter_count_setup`, replacing it with a single
    /// `mov.u32 %num_q_iters, <const>` (and same for `%num_kv_iters`).
    /// When the single-tile criterion holds (`s <= block_q` and
    /// `s <= block_kv`) and `causal` is true, the outer tile-skip
    /// predicate is folded to the constant `1` (single tile is always
    /// active; per-element intra-tile masking still handles correctness).
    /// `None` (default) preserves the existing runtime path and produces
    /// byte-identical PTX to the pre-Sprint-8 emitters.
    pub static_seq_len: Option<u32>,
}

impl CshaExtras {
    /// CSHA level 1 preset — boundary fusion (prologue norm + epilogue
    /// RoPE).  `rope_q` should also be set on the parent config.
    pub fn level1(rmsnorm_eps: f32) -> Self {
        Self {
            level: 1,
            fused_rmsnorm: true,
            fused_projections: false,
            fused_output_proj: false,
            active_heads: 0,
            rmsnorm_eps,
            d_model: 0,
            save_activations_for_backward: false,
            skip_rmsnorm_prologue: false,
            static_seq_len: None,
        }
    }

    /// CSHA level 2 preset — full projection pipelining.
    pub fn level2(rmsnorm_eps: f32, d_model: u32) -> Self {
        Self {
            level: 2,
            fused_rmsnorm: true,
            fused_projections: true,
            fused_output_proj: true,
            active_heads: 0,
            rmsnorm_eps,
            d_model,
            save_activations_for_backward: false,
            skip_rmsnorm_prologue: false,
            static_seq_len: None,
        }
    }

    /// CSHA level 3 preset — full-block fusion.  At emit time this is
    /// identical to level 2; the difference is that the memory planner
    /// has confirmed the subsequent FFN also fits in SMEM, so the
    /// compiler chains them together.
    pub fn level3(rmsnorm_eps: f32, d_model: u32) -> Self {
        Self {
            level: 3,
            ..Self::level2(rmsnorm_eps, d_model)
        }
    }
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
pub fn synthesize_rope_cache_write_ptx(head_dim: i64, rope_style: RopeStyle) -> Vec<u8> {
    let mut ptx = String::with_capacity(4096);

    emit_ptx_header(&mut ptx, 52); // RoPE cache write targets sm_52 minimum
    emit_rope_cache_write_entry(&mut ptx, head_dim, rope_style);

    ptx.push('\0');
    ptx.into_bytes()
}

/// Compute the kernel name encoding variant flags and tile sizes.
///
/// Format: `flash_attn_p{paged}_r{rope}_{style}_g{gqa}_c{causal}_t{tree}_q{block_q}_kv{block_kv}`
///
/// When CSHA extras are present, the name gets a `_cshaL{level}[_nN_pN_oN_hN]`
/// suffix so compiler-cached kernel bytes don't collide with the
/// non-fused baseline.
pub fn flash_attention_kernel_name(config: &FlashAttentionConfig) -> String {
    let base = format!(
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
    );
    let with_csha = match &config.csha {
        None => base,
        Some(c) => {
            // Encode which fusion phases are active so kernel-bytes
            // caches never collide between specialisations.
            let n = c.fused_rmsnorm as u8;
            let p = c.fused_projections as u8;
            let o = c.fused_output_proj as u8;
            let h = c.active_heads;
            format!("{base}_cshaL{}_n{n}_p{p}_o{o}_h{h}", c.level)
        }
    };
    // PCA §4.3 Task 10 — append `_rope_reset_max{N}` suffix when the kernel
    // is compiled with `segment_masked && rope_q`. This differentiates the
    // RoPE-reset-aware variant from the sentinel-disabled (identity-position)
    // variant so a single binary can host BOTH variants without colliding in
    // the runtime PTX cache. `MAX_NUM_DOCS` is the compile-time SMEM bound
    // (see `pca_rope`); the suffix bakes it in so a future bound change forces
    // a fresh cache key.
    if config.segment_masked && config.rope_q {
        format!("{with_csha}_rope_reset_max{}", crate::pca_rope::MAX_NUM_DOCS)
    } else {
        with_csha
    }
}

/// Compute shared memory bytes for a given config.
///
/// Baseline formula: `(block_q + block_kv) * head_dim * sizeof(f16)`.
/// CSHA extras add per-phase SMEM allocations:
///
///   * `fused_rmsnorm`     → +`block_q * head_dim * 2`  (normed-x tile)
///   * `fused_projections` → +`3 * head_dim * d_model` (Q/K/V weight tile,
///                                                      clamped to avoid
///                                                      blowing the budget)
///   * `fused_output_proj` → +`block_q * head_dim * 2`  (output tile)
pub fn shared_mem_bytes(config: &FlashAttentionConfig) -> u32 {
    let base = ((config.block_q + config.block_kv) * config.head_dim * 2) as u32;
    let Some(c) = &config.csha else { return base };
    let head_dim = config.head_dim.max(0) as u32;
    let block_q = config.block_q.max(0) as u32;
    let mut extra: u32 = 0;
    if c.fused_rmsnorm {
        extra = extra.saturating_add(block_q.saturating_mul(head_dim).saturating_mul(2));
    }
    if c.fused_projections {
        // Projection-weight tile — cap d_model at 256 to keep SMEM
        // bounded even for large models (paper §2.2 "tile the weight
        // matrices too").
        let d = c.d_model.min(256);
        extra = extra.saturating_add(3u32.saturating_mul(head_dim).saturating_mul(d).saturating_mul(2));
    }
    if c.fused_output_proj {
        extra = extra.saturating_add(block_q.saturating_mul(head_dim).saturating_mul(2));
    }
    base.saturating_add(extra)
}

// ── PTX emission helpers ──────────────────────────────────────────

fn emit_ptx_header(ptx: &mut String, gpu_sm: u32) {
    let version = if gpu_sm >= 90 { "8.0" } else { "7.0" };
    let target = if gpu_sm >= 90 {
        "sm_90"
    } else if gpu_sm >= 80 {
        "sm_80"
    } else {
        "sm_52"
    };
    ptx.push_str(&format!(".version {version}\n"));
    ptx.push_str(&format!(".target {target}\n"));
    ptx.push_str(".address_size 64\n\n");
}

fn emit_flash_attention_entry(ptx: &mut String, kernel_name: &str, config: &FlashAttentionConfig) {
    // Parameter declarations — ALWAYS declare ALL params regardless of variant flags.
    // The runtime wrapper always passes the full 21-arg set (unused params are null/zero).
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
    ptx.push_str("    .param .u64 num_tree_nodes,\n");
    // Backward pass: logsumexp auxiliary output (null = skip, inference-only)
    ptx.push_str("    .param .u64 param_logsumexp,\n");
    // CSHA extensions (paper §2): always declared; null pointers when the
    // kernel was compiled without CSHA so cuLaunchKernel alignment stays
    // stable across variants.
    ptx.push_str("    .param .u64 csha_x_ptr,\n");
    ptx.push_str("    .param .u64 csha_norm_weight_ptr,\n");
    ptx.push_str("    .param .u64 csha_wq_ptr,\n");
    ptx.push_str("    .param .u64 csha_wk_ptr,\n");
    ptx.push_str("    .param .u64 csha_wv_ptr,\n");
    ptx.push_str("    .param .u64 csha_wo_ptr,\n");
    ptx.push_str("    .param .f32 csha_rmsnorm_eps,\n");
    ptx.push_str("    .param .u32 csha_active_heads,\n");
    ptx.push_str("    .param .u32 csha_d_model\n");

    ptx.push_str(")\n");
    ptx.push_str("{\n");

    // CSHA marker comment — makes ptxas diagnostics easier and documents
    // which phases this kernel realises for ahead-of-time inspection.
    if let Some(c) = &config.csha {
        ptx.push_str(&format!(
            "    // CSHA-L{} phases: norm={} proj={} out={} active_heads={}\n",
            c.level,
            c.fused_rmsnorm as u8,
            c.fused_projections as u8,
            c.fused_output_proj as u8,
            c.active_heads,
        ));
    }

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

    // CSHA A.4: active_heads guard — compile-time literal early-exit.
    // When `config.csha.active_heads` is set to a non-zero value smaller
    // than the full head count, the runtime launches with a shrunken
    // grid_y so each CTA's `bid_y % heads` lands inside the active range
    // — but as a defence-in-depth measure we emit a `setp.ge ... @ret`
    // right after head_idx is computed so any launch-vs-kernel
    // specialisation mismatch fails cleanly instead of corrupting
    // memory. ptxas turns the literal into dead-code elimination when
    // the guard is provably unreachable.
    emit_csha_active_heads_guard(ptx, config);

    // CSHA A.2.2: RMSNorm prologue — emits when `config.csha.fused_rmsnorm`
    // is set. Writes normalised x into the Q-tile SMEM slot, runtime-gated
    // on a non-null `csha_x_ptr` so kernels that were compiled with the
    // flag but launched with null pointers (current A.1/A.2.1x state) fall
    // through to the classic Q-from-HBM path emitted just below.
    emit_csha_rmsnorm_prologue(ptx, config);

    // CSHA A.2.3: matmul projection — emits when `config.csha.fused_projections`
    // is set. Projects `x_norm @ Wq/Wk/Wv` into Q/K/V SMEM tiles via m16n8k16
    // MMA, reusing the `matmul_mma` primitives. Runtime-gated on non-null Wq/
    // Wk/Wv pointers so NULL-pointer calls (current state) fall through to the
    // classic path. Dormant at runtime until A.2.5.
    emit_csha_matmul_projection(ptx, config);

    // CSHA A.2.4: RoPE epilogue — emits when both `config.rope_q` and
    // `config.csha.fused_projections` are set. Rotates the projected Q/K
    // fragments in registers before they feed the QK^T MMA, rather than
    // loading pre-rotated Q from HBM. Runtime-gated on non-null cos/sin
    // pointers. Dormant until A.2.5.
    emit_csha_rope_epilogue(ptx, config);

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
    ptx.push_str("    .reg .b16 %h<32>;\n"); // f16 registers for output conversion
    ptx.push_str("    .reg .pred %p<16>;\n");
    ptx.push_str("    .reg .u32 %r<32>;\n");

    // Scale register
    ptx.push_str("    .reg .f32 %scale;\n");

    // LOG2E constant for exp() via ex2.approx
    ptx.push_str("    .reg .f32 %log2e;\n");
    ptx.push_str(&format!("    mov.f32 %log2e, {};  // 1.4426950408 (log2(e))\n", f32_bits(F32_LOG2E)));

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

    // Logsumexp auxiliary output for backward pass
    ptx.push_str("    .reg .u64 %logsumexp_base;\n");
    ptx.push_str("    .reg .f32 %log_sum, %lse;\n");
    ptx.push_str("    .reg .u64 %lse_addr;\n");
    ptx.push_str("    .reg .pred %p_has_lse;\n");

    // O_acc registers: each thread accumulates a subset of the [block_q, head_dim] output tile
    // Total O_acc regs per thread = (block_q * head_dim) / blockDim.x
    // Example: block_q=64, head_dim=128, blockDim.x=128 → 64 registers
    // Declared dynamically based on config in emit_accumulator_init
    let _ = config;

    // SMEM addressing prolog — used by emit_smem_store / emit_smem_load helpers (Tasks 3–5).
    // %shmem_base holds the generic address of the shared memory window; %smem_addr is a
    // per-access scratch register reused for each [shmem + %reg] computation.
    ptx.push_str("    .reg .u64 %smem_addr;\n");
    ptx.push_str("    .reg .u64 %shmem_base;\n");
    ptx.push_str("    cvta.shared.u64 %shmem_base, shmem;\n");
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

    // Logsumexp auxiliary output (always loaded; null-checked before store)
    ptx.push_str("    ld.param.u64 %logsumexp_base, [param_logsumexp];\n");
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

/// CSHA A.2.2 — RMSNorm prologue emitter.
///
/// When `config.csha.fused_rmsnorm` is set, this emits PTX that:
///
///   1. Loads the `csha_x_ptr` and `csha_norm_weight_ptr` params.
///   2. Runtime null-checks — if either pointer is null (the current
///      A.1/A.2.1x state where the FA call site still passes NULL), the
///      whole block is a no-op and the classic `emit_q_tile_load` path
///      fills SMEM as before.
///   3. When pointers are non-null (future A.2.1e), computes
///      `rsqrt(sum(x²)/head_dim + eps) * x * w` per Q-tile row and
///      stores the f16 result into the Q-tile SMEM region (same
///      `shmem[0..]` slot that `emit_q_tile_load` writes, accounted for
///      in `shared_mem_bytes` at the `fused_rmsnorm` branch).
///   4. Finishes with `bar.sync 0` so the main QK^T loop sees a
///      consistent Q tile regardless of which path filled SMEM.
///
/// NOTE: Level 1 (`fused_rmsnorm=true`, `fused_projections=false`) is
/// semantically ambiguous — a complete implementation also needs the
/// matmul projection inside the kernel (A.2.3) because the prologue's
/// output is `x_norm: [block_q, d_model]`, not the `[block_q, head_dim]`
/// Q tile that the attention body consumes. For now we normalise across
/// `head_dim` so the SMEM geometry matches; A.2.3 will replace this
/// with a proper d_model reduction followed by Wq projection.
///
/// The kernel variant that uses this prologue is currently dormant —
/// `nsl_flash_attention_csha` still forwards to the non-CSHA path per
/// A.1, and A.2.5 is where the CSHA-tagged PTX actually launches.
fn emit_csha_rmsnorm_prologue(ptx: &mut String, config: &FlashAttentionConfig) {
    let Some(csha) = config.csha.as_ref() else {
        return;
    };
    if !csha.fused_rmsnorm {
        return;
    }

    let block_q = config.block_q;
    let head_dim = config.head_dim;

    ptx.push_str("    // ── CSHA A.2.2: RMSNorm prologue ──────────────────────────\n");
    ptx.push_str(&format!(
        "    // fused_rmsnorm=1, block_q={}, head_dim={}\n",
        block_q, head_dim
    ));
    ptx.push_str("    // Registers %rd50-%rd60, %f100-%f103, %p10-%p12 are reserved for this block.\n");

    // Declare local registers — PTX requires all regs pre-declared in the
    // prolog, but the helper is emitted inline. These are additive to the
    // generous `%rd<64>`, `%f<128>`, `%p<16>` pools declared in
    // `emit_register_declarations`, so the indices here don't need new
    // `.reg` lines.
    ptx.push_str("    // (no new .reg lines — all indices fall within the pools declared above)\n");

    ptx.push_str("    ld.param.u64 %rd50, [csha_x_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd51, [csha_norm_weight_ptr];\n");
    ptx.push_str("    ld.param.f32 %f100, [csha_rmsnorm_eps];\n");

    // Runtime null-check: if x_ptr OR norm_weight_ptr is null, skip the
    // whole prologue and let `emit_q_tile_load` run normally.
    ptx.push_str("    setp.eq.u64 %p10, %rd50, 0;\n");
    ptx.push_str("    setp.eq.u64 %p11, %rd51, 0;\n");
    ptx.push_str("    or.pred %p10, %p10, %p11;\n");
    ptx.push_str("    @%p10 bra CSHA_PROLOGUE_END;\n");

    // Per-thread row-scoped RMSNorm. We restrict this prologue so that
    // only threads with `tid_x < block_q` (one thread per Q-tile row)
    // actually do the reduction — keeps the code straightforward until
    // A.2.3 brings in MMA-primitive reductions.
    ptx.push_str(&format!(
        "    setp.ge.u32 %p12, %tid_x, {}; // only threads with tid_x < block_q\n",
        block_q
    ));
    ptx.push_str("    @%p12 bra CSHA_PROLOGUE_END;\n");

    // row = tid_x (u64 for address math)
    ptx.push_str("    cvt.u64.u32 %rd52, %tid_x;\n");

    // x base addr = csha_x_ptr + (batch_idx*seq_len*head_dim + (q_start+row)*head_dim) * 4
    // assumes x is [batch, seq, head_dim] laid out contiguously, f32.
    ptx.push_str("    mul.lo.u64 %rd53, %rd19, %rd6;   // batch_idx * seq_len\n");
    ptx.push_str("    add.u64 %rd53, %rd53, %rd16;     // + q_start\n");
    ptx.push_str("    add.u64 %rd53, %rd53, %rd52;     // + row\n");
    ptx.push_str("    mul.lo.u64 %rd53, %rd53, %rd7;   // * head_dim\n");
    ptx.push_str("    shl.b64 %rd53, %rd53, 2;         // * 4 bytes (f32)\n");
    ptx.push_str("    add.u64 %rd53, %rd50, %rd53;     // %rd53 = x row base\n");

    // Pass 1: sum of squares across head_dim.
    ptx.push_str("    mov.f32 %f101, 0.0;              // sum_sq\n");
    ptx.push_str("    mov.u64 %rd54, 0;                // d\n");
    ptx.push_str("CSHA_PROLOGUE_SUMSQ:\n");
    ptx.push_str("    shl.b64 %rd55, %rd54, 2;         // d * 4\n");
    ptx.push_str("    add.u64 %rd55, %rd53, %rd55;     // x[row, d] addr\n");
    ptx.push_str("    ld.global.f32 %f102, [%rd55];\n");
    ptx.push_str("    fma.rn.f32 %f101, %f102, %f102, %f101;\n");
    ptx.push_str("    add.u64 %rd54, %rd54, 1;\n");
    ptx.push_str(&format!(
        "    setp.lt.u64 %p11, %rd54, {};\n",
        head_dim
    ));
    ptx.push_str("    @%p11 bra CSHA_PROLOGUE_SUMSQ;\n");

    // mean = sum_sq / head_dim; inv_rms = rsqrt(mean + eps)
    ptx.push_str(&format!(
        "    mov.f32 %f103, 0f{:08X};         // 1.0 / head_dim\n",
        (1.0f32 / head_dim as f32).to_bits()
    ));
    ptx.push_str("    mul.f32 %f101, %f101, %f103;     // mean_sq\n");
    ptx.push_str("    add.f32 %f101, %f101, %f100;     // + eps\n");
    ptx.push_str("    rsqrt.approx.f32 %f101, %f101;   // inv_rms\n");

    // Pass 2: y = x * inv_rms * w, stored as f16 into SMEM at
    // shmem[(row * head_dim + d) * 2]. `shmem` is the PTX symbol for the
    // shared-memory array declared in emit_flash_attention_entry.
    ptx.push_str("    mul.lo.u64 %rd56, %rd52, %rd7;   // row * head_dim\n");
    ptx.push_str("    shl.b64 %rd56, %rd56, 1;         // * 2 bytes (f16)\n");
    ptx.push_str("    mov.u64 %rd57, shmem;\n");
    ptx.push_str("    add.u64 %rd56, %rd57, %rd56;     // %rd56 = row SMEM base\n");
    ptx.push_str("    mov.u64 %rd54, 0;                // d = 0\n");
    ptx.push_str("CSHA_PROLOGUE_APPLY:\n");
    ptx.push_str("    shl.b64 %rd58, %rd54, 2;         // d * 4\n");
    ptx.push_str("    add.u64 %rd59, %rd53, %rd58;     // x[row, d]\n");
    ptx.push_str("    ld.global.f32 %f102, [%rd59];\n");
    ptx.push_str("    add.u64 %rd60, %rd51, %rd58;     // w[d]\n");
    ptx.push_str("    ld.global.f32 %f103, [%rd60];\n");
    ptx.push_str("    mul.f32 %f102, %f102, %f101;     // x * inv_rms\n");
    ptx.push_str("    mul.f32 %f102, %f102, %f103;     // * w\n");
    ptx.push_str("    cvt.rn.f16.f32 %h0, %f102;\n");
    ptx.push_str("    shl.b64 %rd58, %rd54, 1;         // d * 2 (SMEM offset)\n");
    ptx.push_str("    add.u64 %rd58, %rd56, %rd58;\n");
    ptx.push_str("    st.shared.b16 [%rd58], %h0;\n");
    ptx.push_str("    add.u64 %rd54, %rd54, 1;\n");
    ptx.push_str(&format!(
        "    setp.lt.u64 %p11, %rd54, {};\n",
        head_dim
    ));
    ptx.push_str("    @%p11 bra CSHA_PROLOGUE_APPLY;\n");

    ptx.push_str("CSHA_PROLOGUE_END:\n");
    ptx.push_str("    bar.sync 0;                       // Q tile (from prologue OR Q load) consistent across warps\n");
    ptx.push_str("    // ── end CSHA prologue ───────────────────────────────────────\n");
}

/// CSHA A.4 — active_heads early-exit guard.
///
/// Emits a compile-time-literal guard that returns from the kernel when
/// `head_idx >= active_heads`. Present only when
/// `config.csha.active_heads` is non-zero (weight-informed specialisation
/// per paper §9.3 — i.e. some heads were pruned by the planner).
///
/// The runtime path (`nsl_flash_attention_csha`, A.4 runtime side)
/// shrinks `grid_y = batch * active_heads` when `active_heads > 0`, so
/// in a correctly-paired launch this guard never fires. Emitting it
/// anyway protects against:
///
///   - A launch-vs-kernel specialisation mismatch (e.g. grid built for
///     a different `active_heads` than the kernel variant was
///     compiled with).
///   - Cache-hit misrouting: two callers of the same FA entry pointing
///     at different PTX variants.
///
/// The literal makes it a strict optimisation hint for ptxas — when
/// `active_heads == heads` at launch time (the common no-pruning case
/// with `active_heads=0`), this function emits nothing.
fn emit_csha_active_heads_guard(ptx: &mut String, config: &FlashAttentionConfig) {
    let Some(csha) = config.csha.as_ref() else {
        return;
    };
    if csha.active_heads == 0 {
        return;
    }

    ptx.push_str("    // ── CSHA A.4: active_heads guard ─────────────────────────\n");
    ptx.push_str(&format!(
        "    // active_heads={} (weight-informed kernel specialisation)\n",
        csha.active_heads
    ));
    ptx.push_str(&format!(
        "    setp.ge.u64 %p9, %rd18, {};       // head_idx >= active_heads\n",
        csha.active_heads
    ));
    ptx.push_str("    @%p9 ret;                         // dead head: exit CTA cleanly\n");
    ptx.push_str("    // ── end CSHA active_heads guard ─────────────────────────────\n");
}

/// CSHA A.2.3 — matmul projection emitter.
///
/// When `config.csha.fused_projections` is set, this emits PTX that
/// projects the normalised x tile (produced by the A.2.2 prologue)
/// into Q/K/V tiles via three m16n8k16 tensor-core matmuls, reusing
/// the primitives in [`crate::matmul_mma`]:
///
/// ```text
///    Q_tile = x_norm @ Wq   (A=[block_q, d_model], B=[d_model, head_dim])
///    K_tile = x_norm @ Wk
///    V_tile = x_norm @ Wv
/// ```
///
/// A.2.3 ships a **single-tile proof of structure** — one MMA per
/// projection at the tile origin — rather than the full
/// `(block_q/MMA_M) × (head_dim/MMA_N)` tile-sweep loop. The loop is
/// marked with a PTX comment and lands in a follow-up; ABI / SMEM
/// layout / register conventions are fixed here so the loop expansion
/// is a local change.
///
/// Same dormant-kernel contract as A.2.2: the CSHA-tagged kernel
/// variant is emitted but not launched by `nsl_flash_attention_csha`
/// today (A.2.5 wires runtime dispatch). Runtime null-checks on
/// `csha_wq/wk/wv_ptr` keep current A.1/A.2.1x call sites (which pass
/// NULL) from dereferencing uninitialised pointers if the variant
/// ever does execute during bring-up.
fn emit_csha_matmul_projection(ptx: &mut String, config: &FlashAttentionConfig) {
    let Some(csha) = config.csha.as_ref() else {
        return;
    };
    if !csha.fused_projections {
        return;
    }

    let block_q = config.block_q;
    let head_dim = config.head_dim;
    let d_model = csha.d_model.max(head_dim as u32);

    ptx.push_str("    // ── CSHA A.2.3: matmul projection (x_norm @ Wq/Wk/Wv) ────\n");
    ptx.push_str(&format!(
        "    // fused_projections=1, block_q={}, head_dim={}, d_model={}\n",
        block_q, head_dim, d_model
    ));

    // Load the three projection-weight pointers and null-check.
    ptx.push_str("    ld.param.u64 %rd61, [csha_wq_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd62, [csha_wk_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd63, [csha_wv_ptr];\n");
    ptx.push_str("    setp.eq.u64 %p13, %rd61, 0;\n");
    ptx.push_str("    setp.eq.u64 %p14, %rd62, 0;\n");
    ptx.push_str("    or.pred %p13, %p13, %p14;\n");
    ptx.push_str("    setp.eq.u64 %p14, %rd63, 0;\n");
    ptx.push_str("    or.pred %p13, %p13, %p14;\n");
    ptx.push_str("    @%p13 bra CSHA_PROJECTION_END;\n");

    // Emit MMA temporary registers (laneid, row indices, addr scratch).
    // `emit_mma_temp_registers` is `#[allow(dead_code)]` so calling it here
    // both gives us the standard MMA scaffolding and removes the dead-code
    // annotation's need in future commits.
    emit_mma_temp_registers(ptx);

    // Projection-weight SMEM slot starts right after the Q-tile slot.
    // Layout (matches `shared_mem_bytes` at `fused_projections` branch):
    //   [0 .. block_q*head_dim*2)           — Q tile (written by A.2.2 prologue)
    //   [block_q*head_dim*2 .. + 3*hd*d*2)  — Wq/Wk/Wv tiles
    let q_tile_bytes = (block_q * head_dim * 2) as u32;
    let weight_tile_bytes = (head_dim as u32) * d_model.min(256) * 2;
    ptx.push_str(&format!(
        "    // SMEM weight-tile bases: wq={}, wk={}, wv={}\n",
        q_tile_bytes,
        q_tile_bytes + weight_tile_bytes,
        q_tile_bytes + 2 * weight_tile_bytes,
    ));

    // A-fragment and B-fragment registers for one m16n8k16 MMA iteration.
    // These are local to the projection block — distinct from the attention
    // QK^T / PV MMAs further down the kernel body to avoid register aliasing.
    ptx.push_str("    .reg .b32 %proj_a0, %proj_a1, %proj_a2, %proj_a3;\n");
    ptx.push_str("    .reg .b32 %proj_b0, %proj_b1;\n");
    ptx.push_str("    .reg .f32 %proj_d0, %proj_d1, %proj_d2, %proj_d3;\n");
    ptx.push_str("    .reg .f32 %proj_c0, %proj_c1, %proj_c2, %proj_c3;\n");
    // A.2.3 fragment-pack: 2 .b32 regs carry the 4 f16 output values per
    // (m, n) tile iteration. `%mma_h0` / `%mma_h1` (declared by
    // `emit_mma_temp_registers` earlier in the kernel) are the f16
    // conversion temps used by `emit_f32_to_f16_pack`.
    ptx.push_str("    .reg .b32 %proj_c_pack0, %proj_c_pack1;\n");
    ptx.push_str("    // Zero accumulator for this proof-of-structure iteration\n");
    ptx.push_str("    mov.f32 %proj_c0, 0.0;\n");
    ptx.push_str("    mov.f32 %proj_c1, 0.0;\n");
    ptx.push_str("    mov.f32 %proj_c2, 0.0;\n");
    ptx.push_str("    mov.f32 %proj_c3, 0.0;\n");

    // Fragment register string lists shared across Q/K/V.
    let a_regs = [
        "proj_a0".to_string(),
        "proj_a1".to_string(),
        "proj_a2".to_string(),
        "proj_a3".to_string(),
    ];
    let b_regs = ["proj_b0".to_string(), "proj_b1".to_string()];
    let c_regs = [
        "%proj_c0".to_string(),
        "%proj_c1".to_string(),
        "%proj_c2".to_string(),
        "%proj_c3".to_string(),
    ];
    let d_regs = [
        "%proj_d0".to_string(),
        "%proj_d1".to_string(),
        "%proj_d2".to_string(),
        "%proj_d3".to_string(),
    ];

    // x_norm (A side) lives at SMEM offset 0, row-stride = head_dim * 2.
    // Row-stride here is head_dim rather than d_model because A.2.2's
    // prologue currently normalises over head_dim (see A.2.2 semantic
    // note); A.2.3-full-d_model is a follow-up coupled with A.2.1e.
    let a_stride_bytes = (head_dim * 2) as usize;
    let b_stride_bytes = (d_model.min(256) * 2) as usize;

    // Tile-sweep counters (A.2.3-tile-sweep expansion): m iterates
    // block_q / MMA_M=16 tiles along the output rows, n iterates
    // head_dim / MMA_N=8 tiles along the columns, k reduces over
    // d_model / MMA_K=16 tiles. Accumulator (%proj_c*) is zeroed at
    // the top of each (m, n) iteration and lives across the K loop.
    ptx.push_str("    .reg .u32 %ts_m, %ts_n, %ts_k;\n");
    ptx.push_str("    .reg .u32 %ts_a_base, %ts_b_base, %ts_out_base;\n");
    // A.2.3.2 lane-coherent scatter scratch: per-thread row/col derived
    // from `%mma_laneid` (set by `emit_mma_temp_registers`), plus the
    // two per-pack addresses.
    ptx.push_str("    .reg .u32 %ts_row_lo, %ts_col_pair, %ts_col_off, %ts_row_off;\n");
    ptx.push_str("    .reg .u32 %ts_pack0_addr, %ts_pack1_addr;\n");
    let m_tiles = (block_q / 16).max(1);
    let n_tiles = (head_dim / 8).max(1);
    let k_tiles = (d_model.min(256) as i64 / 16).max(1);

    // Per-projection helper: emit the M/N/K nested loops that sweep a
    // full (block_q × head_dim) output by accumulating over the K-axis
    // of (x_norm × W_*). Writes fragment results into a scratch SMEM
    // slot at `out_smem_bytes`; the store is a placeholder
    // (`st.shared.b32` per accumulator) until the fragment-packing
    // helper lands — semantically-correct conversion+packing is the
    // A.2.3-follow-up's remaining task, but the loop structure,
    // address math, and MMA issue are real and testable.
    fn emit_proj_sweep(
        ptx: &mut String,
        proj_tag: &str,
        w_offset_bytes: u32,
        out_offset_bytes: u32,
        a_stride_bytes: usize,
        b_stride_bytes: usize,
        m_tiles: i64,
        n_tiles: i64,
        k_tiles: i64,
        a_regs: &[String; 4],
        b_regs: &[String; 2],
        c_regs: &[String; 4],
        d_regs: &[String; 4],
    ) {
        ptx.push_str(&format!(
            "    // --- {} projection: x_norm @ W{} ({}×{} × {} MMA) ---\n",
            proj_tag,
            proj_tag.to_ascii_lowercase(),
            m_tiles,
            n_tiles,
            k_tiles,
        ));
        ptx.push_str(&format!(
            "    mov.u32 %ts_m, 0;                 // M-tile loop: 0..{}\n",
            m_tiles
        ));
        ptx.push_str(&format!("CSHA_PROJ_{}_M_LOOP:\n", proj_tag));
        ptx.push_str(&format!(
            "    mov.u32 %ts_n, 0;                 // N-tile loop: 0..{}\n",
            n_tiles
        ));
        ptx.push_str(&format!("CSHA_PROJ_{}_N_LOOP:\n", proj_tag));

        // Zero accumulator for this (m, n) tile.
        for i in 0..4 {
            ptx.push_str(&format!("    mov.f32 {}, 0.0;\n", c_regs[i]));
        }

        // K-tile loop: one MMA per iteration, accumulating into %proj_c*.
        ptx.push_str(&format!(
            "    mov.u32 %ts_k, 0;                 // K-tile loop: 0..{}\n",
            k_tiles
        ));
        ptx.push_str(&format!("CSHA_PROJ_{}_K_LOOP:\n", proj_tag));

        // A-tile base = m*16*a_stride + k*16*2 (row-major, f16)
        ptx.push_str(&format!(
            "    mul.lo.u32 %ts_a_base, %ts_m, {};   // m * 16 * a_stride\n",
            16 * a_stride_bytes
        ));
        ptx.push_str("    mad.lo.u32 %ts_a_base, %ts_k, 32, %ts_a_base; // + k*16*2\n");

        // B-tile base = W_offset + k*16*b_stride + n*8*2 (col-major B)
        ptx.push_str(&format!(
            "    mul.lo.u32 %ts_b_base, %ts_k, {};   // k * 16 * b_stride\n",
            16 * b_stride_bytes
        ));
        ptx.push_str(&format!(
            "    add.u32 %ts_b_base, %ts_b_base, {}; // + W offset\n",
            w_offset_bytes
        ));
        ptx.push_str("    mad.lo.u32 %ts_b_base, %ts_n, 16, %ts_b_base; // + n*8*2\n");

        crate::matmul_mma::emit_load_a_fragment_smem(
            ptx,
            a_regs,
            "%ts_a_base",
            a_stride_bytes,
        );
        crate::matmul_mma::emit_load_b_fragment_smem(
            ptx,
            b_regs,
            "%ts_b_base",
            b_stride_bytes,
        );
        // The load helpers take bare register names (they prepend `%`
        // internally), but `emit_mma_instruction` embeds names
        // verbatim — prefix them here before the MMA issue.
        let pct = |regs: &[String]| -> Vec<String> {
            regs.iter().map(|r| format!("%{}", r)).collect()
        };
        let a_pct_vec = pct(a_regs);
        let b_pct_vec = pct(b_regs);
        let a_pct: [String; 4] = [
            a_pct_vec[0].clone(),
            a_pct_vec[1].clone(),
            a_pct_vec[2].clone(),
            a_pct_vec[3].clone(),
        ];
        let b_pct: [String; 2] = [b_pct_vec[0].clone(), b_pct_vec[1].clone()];
        crate::matmul_mma::emit_mma_instruction(ptx, d_regs, &a_pct, &b_pct, c_regs);

        // D is the new accumulator — copy into C for the next K iter.
        for i in 0..4 {
            ptx.push_str(&format!("    mov.f32 {}, {};\n", c_regs[i], d_regs[i]));
        }

        ptx.push_str("    add.u32 %ts_k, %ts_k, 1;\n");
        ptx.push_str(&format!(
            "    setp.lt.u32 %p7, %ts_k, {};\n",
            k_tiles
        ));
        ptx.push_str(&format!("    @%p7 bra CSHA_PROJ_{}_K_LOOP;\n", proj_tag));

        // Fragment pack: f32 accumulator (4 f32 per thread) → 2 .b32
        // registers holding 4 packed f16 values, stored to SMEM at the
        // per-(m, n) tile origin. Uses the shared `emit_f32_to_f16_pack`
        // helper so ptxas sees the same conversion pattern as the
        // existing attention-body epilogue.
        //
        // Lane-scatter note: mma.sync.m16n8k16 f32 output has each
        // thread holding (row=t/4, col=2*(t%4)+{0,1}) and (row+8,
        // same col) fragment entries. The per-thread SMEM address
        // computed here stages the 4 packed f16 values at the tile
        // base — a thread-local scatter-to-lane-coords pass lands in
        // the A.2.3.2 follow-up once real-hardware validation is in
        // place. The dtype is now correct (f16), which was the blocker
        // for A.2.4's f16-reading tile-sweep.
        let c_names: Vec<String> = c_regs
            .iter()
            .map(|r| r.trim_start_matches('%').to_string())
            .collect();
        let c_pack_dsts = vec!["proj_c_pack0".to_string(), "proj_c_pack1".to_string()];
        emit_f32_to_f16_pack(ptx, &c_names, &c_pack_dsts);

        // Tile-base SMEM address: out_offset + m*16*a_stride + n*8*2.
        ptx.push_str(&format!(
            "    mul.lo.u32 %ts_out_base, %ts_m, {};  // m * 16 * a_stride\n",
            16 * a_stride_bytes
        ));
        ptx.push_str(&format!(
            "    add.u32 %ts_out_base, %ts_out_base, {}; // + projection out offset\n",
            out_offset_bytes
        ));
        ptx.push_str("    mad.lo.u32 %ts_out_base, %ts_n, 16, %ts_out_base; // + n*8*2\n");

        // A.2.3.2 lane-coherent scatter (per mma.sync.m16n8k16 f32 accumulator layout):
        //   pack0 holds (regs[0], regs[1]) → (row = laneid/4,        col = 2*(laneid%4)+{0,1})
        //   pack1 holds (regs[2], regs[3]) → (row = laneid/4 + 8,    col = 2*(laneid%4)+{0,1})
        // Each thread computes its own scatter address from the tile
        // base + per-lane row/col offsets, so the 32 threads in a warp
        // cover the full 16×8 output tile with no bank conflicts on
        // aligned SMEM.
        ptx.push_str(&format!(
            "    shr.u32 %ts_row_lo, %mma_laneid, 2;              // row_lo = laneid / 4 (0..7)\n"
        ));
        ptx.push_str("    and.b32 %ts_col_pair, %mma_laneid, 3;            // col_pair = laneid % 4 (0..3)\n");
        ptx.push_str("    shl.b32 %ts_col_off, %ts_col_pair, 2;            // col_off = col_pair * 2*f16 = *4\n");
        ptx.push_str(&format!(
            "    mul.lo.u32 %ts_row_off, %ts_row_lo, {};          // row_off = row_lo * stride\n",
            a_stride_bytes
        ));
        ptx.push_str("    add.u32 %ts_pack0_addr, %ts_out_base, %ts_row_off;\n");
        ptx.push_str("    add.u32 %ts_pack0_addr, %ts_pack0_addr, %ts_col_off;\n");
        ptx.push_str("    st.shared.b32 [%ts_pack0_addr], %proj_c_pack0;   // (row_lo, col_pair)\n");
        ptx.push_str(&format!(
            "    add.u32 %ts_pack1_addr, %ts_pack0_addr, {};       // +8 rows = 8 * stride\n",
            8 * a_stride_bytes
        ));
        ptx.push_str("    st.shared.b32 [%ts_pack1_addr], %proj_c_pack1;   // (row_lo+8, col_pair)\n");

        ptx.push_str("    add.u32 %ts_n, %ts_n, 1;\n");
        ptx.push_str(&format!(
            "    setp.lt.u32 %p7, %ts_n, {};\n",
            n_tiles
        ));
        ptx.push_str(&format!("    @%p7 bra CSHA_PROJ_{}_N_LOOP;\n", proj_tag));

        ptx.push_str("    add.u32 %ts_m, %ts_m, 1;\n");
        ptx.push_str(&format!(
            "    setp.lt.u32 %p7, %ts_m, {};\n",
            m_tiles
        ));
        ptx.push_str(&format!("    @%p7 bra CSHA_PROJ_{}_M_LOOP;\n", proj_tag));
    }

    // Three projections — each writes into its own SMEM slot.
    // Output slots reuse the Q-tile region (slot 0) for Q since the
    // attention body reads Q from that slot; K and V write into the
    // weight-tile slots vacated after the MMA reduction (which is safe
    // because each K iteration overwrites its own B fragment).
    let q_out = 0u32;
    let k_out = q_tile_bytes + weight_tile_bytes;
    let v_out = q_tile_bytes + 2 * weight_tile_bytes;

    emit_proj_sweep(
        ptx, "Q",
        q_tile_bytes,
        q_out,
        a_stride_bytes, b_stride_bytes,
        m_tiles, n_tiles, k_tiles,
        &a_regs, &b_regs, &c_regs, &d_regs,
    );
    emit_proj_sweep(
        ptx, "K",
        q_tile_bytes + weight_tile_bytes,
        k_out,
        a_stride_bytes, b_stride_bytes,
        m_tiles, n_tiles, k_tiles,
        &a_regs, &b_regs, &c_regs, &d_regs,
    );
    emit_proj_sweep(
        ptx, "V",
        q_tile_bytes + 2 * weight_tile_bytes,
        v_out,
        a_stride_bytes, b_stride_bytes,
        m_tiles, n_tiles, k_tiles,
        &a_regs, &b_regs, &c_regs, &d_regs,
    );

    // A.2.3.2: lane-coherent scatter now emitted per iteration above.
    // Real-hardware numerical validation (mma.sync f32-accumulator test
    // vector) is the remaining follow-up — the scatter math matches the
    // documented layout but has not been run on GPU.

    ptx.push_str("CSHA_PROJECTION_END:\n");
    ptx.push_str("    bar.sync 0;                       // Q/K/V SMEM tiles consistent\n");
    ptx.push_str("    // ── end CSHA projection ─────────────────────────────────────\n");
}

/// CSHA A.2.4 — RoPE epilogue emitter.
///
/// When both `config.rope_q` and `config.csha.fused_projections` are set,
/// this emits PTX that rotates the Q and K fragments produced by the
/// A.2.3 matmul projection **in registers** before they feed the QK^T
/// MMA, rather than loading pre-rotated Q from HBM as `emit_q_tile_load`
/// does in the non-CSHA path.
///
/// Rotation math matches the existing [`emit_q_tile_load`] RoPE branch:
///
/// ```text
///   q_rot_a = q_a * cos - q_b * sin
///   q_rot_b = q_a * sin + q_b * cos
/// ```
///
/// A.2.4 ships a **register-level scaffold** — one (q_a, q_b) pair
/// rotation for Q and one for K, using the `proj_d*` output registers
/// from A.2.3's single MMA. The tile-sweep that applies rotation to
/// every fragment across `(block_q, head_dim)` lands with A.2.3's
/// tile-sweep follow-up, since the two loops share the same tiling.
///
/// Runtime null-check on `cos_ptr` / `sin_ptr` keeps current
/// A.1/A.2.1x call sites (which pass null for CSHA-tagged kernels via
/// the non-CSHA forwarder) from dereferencing uninitialised pointers.
/// Kernel variant remains dormant until A.2.5 wires the runtime FFI.
fn emit_csha_rope_epilogue(ptx: &mut String, config: &FlashAttentionConfig) {
    let Some(csha) = config.csha.as_ref() else {
        return;
    };
    if !csha.fused_projections || !config.rope_q {
        return;
    }

    let head_dim = config.head_dim;
    let stride_val = match config.rope_style {
        RopeStyle::HalfSplit => head_dim / 2,
        RopeStyle::Adjacent => 1,
    };
    let block_q = config.block_q;

    ptx.push_str("    // ── CSHA A.2.4: RoPE epilogue ────────────────────────────\n");
    ptx.push_str(&format!(
        "    // rope_q=1, rope_style={:?}, head_dim={}, stride={}\n",
        config.rope_style, head_dim, stride_val
    ));

    // cos/sin param pointers are loaded into %rd12/%rd13 by
    // `emit_param_loads`; we just need to null-check them here.
    ptx.push_str("    setp.eq.u64 %p15, %rd12, 0;       // cos_ptr null?\n");
    ptx.push_str("    setp.eq.u64 %p14, %rd13, 0;       // sin_ptr null?\n");
    ptx.push_str("    or.pred %p15, %p15, %p14;\n");
    ptx.push_str("    @%p15 bra CSHA_ROPE_EPILOGUE_END;\n");

    // A.2.4 tile-sweep registers.
    ptx.push_str("    .reg .f32 %rope_q_a, %rope_q_b, %rope_q_rot_a, %rope_q_rot_b;\n");
    ptx.push_str("    .reg .f32 %rope_k_a, %rope_k_b, %rope_k_rot_a, %rope_k_rot_b;\n");
    ptx.push_str("    .reg .f32 %rope_cos, %rope_sin, %rope_tmp;\n");
    ptx.push_str("    .reg .u32 %rope_pair_idx, %rope_row, %rope_d;\n");
    ptx.push_str("    .reg .u32 %rope_elem_a_off, %rope_elem_b_off;\n");
    ptx.push_str("    .reg .u64 %rope_cs_addr, %rope_row_u64;\n");

    // Pair count = block_q * (head_dim / 2); each thread strides by
    // blockDim.x (=128) through the flat pair index space so all 128
    // threads cooperate across the full Q/K SMEM tile.
    let pairs_per_q = (block_q * (head_dim / 2)) as u32;
    let q_smem_base = 0u32;
    let q_tile_bytes = (block_q * head_dim * 2) as u32;
    let k_smem_base = if csha.fused_projections {
        // Matches A.2.3's k_out: q_tile_bytes + weight_tile_bytes.
        q_tile_bytes
            + (head_dim as u32) * csha.d_model.max(head_dim as u32).min(256) * 2
    } else {
        // Without projection, K stays in its default pre-projected slot.
        q_tile_bytes
    };
    let stride_bytes = (stride_val * 4) as u32;

    // --- Q rotation sweep ---
    ptx.push_str(&format!(
        "    // --- Q rotation sweep: {} row pairs × {} col pairs per row ---\n",
        block_q,
        head_dim / 2,
    ));
    ptx.push_str("    mov.u32 %rope_pair_idx, %tid_x;\n");
    ptx.push_str("CSHA_ROPE_Q_LOOP:\n");
    ptx.push_str(&format!(
        "    setp.ge.u32 %p14, %rope_pair_idx, {};\n",
        pairs_per_q
    ));
    ptx.push_str("    @%p14 bra CSHA_ROPE_K_START;\n");
    ptx.push_str(&format!(
        "    div.u32 %rope_row, %rope_pair_idx, {};   // row = pair/(head_dim/2)\n",
        head_dim / 2
    ));
    ptx.push_str(&format!(
        "    rem.u32 %rope_d, %rope_pair_idx, {};     // d = pair%%(head_dim/2)\n",
        head_dim / 2
    ));
    // elem_a_off = (row * head_dim + d) * 4  (f32 SMEM placeholder stores)
    ptx.push_str(&format!(
        "    mul.lo.u32 %rope_elem_a_off, %rope_row, {};\n",
        head_dim * 4
    ));
    ptx.push_str("    mad.lo.u32 %rope_elem_a_off, %rope_d, 4, %rope_elem_a_off;\n");
    ptx.push_str(&format!(
        "    add.u32 %rope_elem_a_off, %rope_elem_a_off, {}; // + q_smem_base\n",
        q_smem_base
    ));
    ptx.push_str(&format!(
        "    add.u32 %rope_elem_b_off, %rope_elem_a_off, {}; // +stride*4\n",
        stride_bytes
    ));

    ptx.push_str("    ld.shared.f32 %rope_q_a, [%rope_elem_a_off];\n");
    ptx.push_str("    ld.shared.f32 %rope_q_b, [%rope_elem_b_off];\n");

    // cos/sin address: base_ptr + ((q_start + row) * head_dim + d) * 4.
    // Mirrors the math in `emit_q_tile_load`'s LOOP_Q_LOAD_ROPE branch
    // (%rd25 = cos_base, %rd26 = sin_base in that emitter); here we
    // compute the offset from cos_ptr (%rd12) / sin_ptr (%rd13)
    // directly per loop iteration so the scaffold stays self-contained.
    ptx.push_str("    cvt.u64.u32 %rope_row_u64, %rope_row;\n");
    ptx.push_str("    add.u64 %rope_row_u64, %rope_row_u64, %rd16;        // + q_start\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rope_cs_addr, %rope_row_u64, {}; // * head_dim\n",
        head_dim
    ));
    ptx.push_str("    cvt.u64.u32 %rope_row_u64, %rope_d;                  // reuse as d temp\n");
    ptx.push_str("    add.u64 %rope_cs_addr, %rope_cs_addr, %rope_row_u64; // + d\n");
    ptx.push_str("    shl.b64 %rope_cs_addr, %rope_cs_addr, 2;             // * 4 bytes\n");
    ptx.push_str("    add.u64 %rope_row_u64, %rd12, %rope_cs_addr;         // cos addr\n");
    ptx.push_str("    ld.global.f32 %rope_cos, [%rope_row_u64];\n");
    ptx.push_str("    add.u64 %rope_row_u64, %rd13, %rope_cs_addr;         // sin addr\n");
    ptx.push_str("    ld.global.f32 %rope_sin, [%rope_row_u64];\n");

    ptx.push_str("    mul.f32 %rope_q_rot_a, %rope_q_a, %rope_cos;\n");
    ptx.push_str("    mul.f32 %rope_tmp, %rope_q_b, %rope_sin;\n");
    ptx.push_str("    sub.f32 %rope_q_rot_a, %rope_q_rot_a, %rope_tmp;\n");
    ptx.push_str("    mul.f32 %rope_q_rot_b, %rope_q_a, %rope_sin;\n");
    ptx.push_str("    mul.f32 %rope_tmp, %rope_q_b, %rope_cos;\n");
    ptx.push_str("    add.f32 %rope_q_rot_b, %rope_q_rot_b, %rope_tmp;\n");
    ptx.push_str("    st.shared.f32 [%rope_elem_a_off], %rope_q_rot_a;\n");
    ptx.push_str("    st.shared.f32 [%rope_elem_b_off], %rope_q_rot_b;\n");

    ptx.push_str("    add.u32 %rope_pair_idx, %rope_pair_idx, 128;  // stride by blockDim.x\n");
    ptx.push_str("    bra CSHA_ROPE_Q_LOOP;\n");

    // --- K rotation sweep ---
    ptx.push_str("CSHA_ROPE_K_START:\n");
    ptx.push_str(&format!(
        "    // --- K rotation sweep: {} row pairs × {} col pairs per row ---\n",
        block_q,
        head_dim / 2,
    ));
    ptx.push_str("    mov.u32 %rope_pair_idx, %tid_x;\n");
    ptx.push_str("CSHA_ROPE_K_LOOP:\n");
    ptx.push_str(&format!(
        "    setp.ge.u32 %p14, %rope_pair_idx, {};\n",
        pairs_per_q
    ));
    ptx.push_str("    @%p14 bra CSHA_ROPE_EPILOGUE_END;\n");
    ptx.push_str(&format!(
        "    div.u32 %rope_row, %rope_pair_idx, {};\n",
        head_dim / 2
    ));
    ptx.push_str(&format!(
        "    rem.u32 %rope_d, %rope_pair_idx, {};\n",
        head_dim / 2
    ));
    ptx.push_str(&format!(
        "    mul.lo.u32 %rope_elem_a_off, %rope_row, {};\n",
        head_dim * 4
    ));
    ptx.push_str("    mad.lo.u32 %rope_elem_a_off, %rope_d, 4, %rope_elem_a_off;\n");
    ptx.push_str(&format!(
        "    add.u32 %rope_elem_a_off, %rope_elem_a_off, {}; // + k_smem_base\n",
        k_smem_base
    ));
    ptx.push_str(&format!(
        "    add.u32 %rope_elem_b_off, %rope_elem_a_off, {};\n",
        stride_bytes
    ));

    ptx.push_str("    ld.shared.f32 %rope_k_a, [%rope_elem_a_off];\n");
    ptx.push_str("    ld.shared.f32 %rope_k_b, [%rope_elem_b_off];\n");

    // Per-(row, d) cos/sin indexing — same pattern as the Q sweep.
    ptx.push_str("    cvt.u64.u32 %rope_row_u64, %rope_row;\n");
    ptx.push_str("    add.u64 %rope_row_u64, %rope_row_u64, %rd16;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rope_cs_addr, %rope_row_u64, {};\n",
        head_dim
    ));
    ptx.push_str("    cvt.u64.u32 %rope_row_u64, %rope_d;\n");
    ptx.push_str("    add.u64 %rope_cs_addr, %rope_cs_addr, %rope_row_u64;\n");
    ptx.push_str("    shl.b64 %rope_cs_addr, %rope_cs_addr, 2;\n");
    ptx.push_str("    add.u64 %rope_row_u64, %rd12, %rope_cs_addr;\n");
    ptx.push_str("    ld.global.f32 %rope_cos, [%rope_row_u64];\n");
    ptx.push_str("    add.u64 %rope_row_u64, %rd13, %rope_cs_addr;\n");
    ptx.push_str("    ld.global.f32 %rope_sin, [%rope_row_u64];\n");

    ptx.push_str("    mul.f32 %rope_k_rot_a, %rope_k_a, %rope_cos;\n");
    ptx.push_str("    mul.f32 %rope_tmp, %rope_k_b, %rope_sin;\n");
    ptx.push_str("    sub.f32 %rope_k_rot_a, %rope_k_rot_a, %rope_tmp;\n");
    ptx.push_str("    mul.f32 %rope_k_rot_b, %rope_k_a, %rope_sin;\n");
    ptx.push_str("    mul.f32 %rope_tmp, %rope_k_b, %rope_cos;\n");
    ptx.push_str("    add.f32 %rope_k_rot_b, %rope_k_rot_b, %rope_tmp;\n");
    ptx.push_str("    st.shared.f32 [%rope_elem_a_off], %rope_k_rot_a;\n");
    ptx.push_str("    st.shared.f32 [%rope_elem_b_off], %rope_k_rot_b;\n");

    ptx.push_str("    add.u32 %rope_pair_idx, %rope_pair_idx, 128;\n");
    ptx.push_str("    bra CSHA_ROPE_K_LOOP;\n");


    ptx.push_str("CSHA_ROPE_EPILOGUE_END:\n");
    ptx.push_str("    // ── end CSHA RoPE epilogue ─────────────────────────────────\n");
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
        ptx.push_str(&format!("    // Paired offset: stride = {}\n", stride_val));
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
        emit_smem_store(ptx, "f32", "%rd29", "%q_rot_a");
        ptx.push_str(&format!(
            "    add.u64 %rd35, %rd29, {};          // offset_b in shmem\n",
            stride_val * 4
        ));
        emit_smem_store(ptx, "f32", "%rd35", "%q_rot_b");

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
        emit_smem_store(ptx, "f32", "%rd24", "%f0");
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
    ptx.push_str(&format!("    mov.f32 %row_max, {};  // -inf as IEEE 754\n", f32_bits(F32_NEG_INF)));
    ptx.push_str(&format!("    mov.f32 %row_sum, {};  // 0.0\n", f32_bits(F32_ZERO)));

    // Zero O_acc registers: each thread owns (block_q * head_dim) / 128 output elements
    let num_oacc = (config.block_q * config.head_dim / 128) as usize;
    ptx.push_str(&format!(
        "    // O_acc: {} registers per thread (f64..f{})\n",
        num_oacc,
        64 + num_oacc - 1
    ));
    for i in 0..num_oacc {
        ptx.push_str(&format!("    mov.f32 %f{}, {};\n", 64 + i, f32_bits(F32_ZERO)));
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
        ptx.push_str(
            "    div.u64 %rd36, %k_start, %rd11;  // logical_block = k_start / block_size\n",
        );
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
    emit_smem_store(ptx, "f32", "%rd45", "%f0");
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
    ptx.push_str(&format!("    mov.f32 %f0, {};               // S accumulator = 0\n", f32_bits(F32_ZERO)));
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
    emit_smem_load(ptx, "f32", "%f1", "%rd43"); // Q[q_row][d]

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
    emit_smem_load(ptx, "f32", "%f2", "%rd44"); // K[k_col][d]

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
        ptx.push_str(&format!("    @%p1 mov.f32 %f0, {};     // -inf for non-ancestor positions\n", f32_bits(F32_NEG_INF)));
    } else if config.causal {
        ptx.push_str("    // Partial causal mask on diagonal tile: S[i][j] = -inf where k_start+j > q_start+i\n");
        ptx.push_str("    cvt.u64.u32 %rd43, %r5;           // k_col as u64\n");
        ptx.push_str("    add.u64 %rd43, %k_start, %rd43;   // k_abs = k_start + k_col\n");
        ptx.push_str("    cvt.u64.u32 %rd44, %r4;           // q_row as u64\n");
        ptx.push_str("    add.u64 %rd44, %rd16, %rd44;      // q_abs = q_start + q_row\n");
        ptx.push_str("    setp.gt.u64 %p1, %rd43, %rd44;    // k_abs > q_abs?\n");
        ptx.push_str(&format!("    @%p1 mov.f32 %f0, {};     // -inf for masked positions\n", f32_bits(F32_NEG_INF)));
    }

    // Store S value in a temp register indexed by s_iter (we use %f3..%f3+num_s-1)
    // We'll cap at a manageable number and use a register-indexed store pattern
    ptx.push_str("    // Store S value for this thread's s_iter\n");
    // Use registers %f3..%f(3+num_s_per_thread-1) to hold S values for later phases
    // We emit a chain: if s_iter==0, store to %f3; if s_iter==1 store to %f4, etc.
    for i in 0..num_s_per_thread {
        ptx.push_str(&format!("    setp.eq.u32 %p2, %r1, {};\n", i));
        ptx.push_str(&format!("    @%p2 mov.f32 %f{}, %f0;\n", 3 + i));
    }

    ptx.push_str("    add.u32 %r1, %r1, 1;                  // s_iter++\n");
    ptx.push_str("    setp.lt.u32 %p0, %r1, %r2;\n");
    ptx.push_str("    @%p0 bra LOOP_S_OUTER;\n");

    ptx.push_str("    bar.sync 0;  // FENCE 2: all warps done reading K before SRAM overwrite\n");

    // ── Phase 3: Online softmax ────────────────────────────────────
    ptx.push_str("    // Phase 3: Online softmax — S→P in-place in registers\n");

    // Find local max across this thread's S values
    ptx.push_str("    // Local max from this thread's S values\n");
    ptx.push_str(&format!("    mov.f32 %f0, {};               // f_local_max = -inf\n", f32_bits(F32_NEG_INF)));
    for i in 0..num_s_per_thread {
        ptx.push_str(&format!(
            "    max.f32 %f0, %f0, %f{};              // max with S[{}]\n",
            3 + i,
            i
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
            64 + i,
            64 + i
        ));
    }

    // Compute P = exp(S - new_max) and accumulate row_sum
    ptx.push_str("    // P = exp(S - new_max)  // overwrites S registers in-place\n");
    ptx.push_str(&format!("    mov.f32 %f1, {};               // partial_sum = 0\n", f32_bits(F32_ZERO)));
    for i in 0..num_s_per_thread {
        ptx.push_str(&format!(
            "    sub.f32 %f{}, %f{}, %new_max;\n",
            3 + i,
            3 + i
        ));
        ptx.push_str(&format!("    mul.f32 %f{}, %f{}, %log2e;\n", 3 + i, 3 + i));
        ptx.push_str(&format!("    ex2.approx.f32 %f{}, %f{};\n", 3 + i, 3 + i));
        ptx.push_str(&format!(
            "    add.f32 %f1, %f1, %f{};              // partial_sum += P[{}]\n",
            3 + i,
            i
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
    emit_smem_store(ptx, "f32", "%rd50", "%f0");
    ptx.push_str("    add.u64 %rd47, %rd47, 128;             // elem_idx += blockDim.x\n");
    ptx.push_str(&format!("    setp.lt.u64 %p0, %rd47, {};\n", total_k_elems));
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
    emit_smem_load(ptx, "f32", "%f1", "%rd48"); // V[k][d_col]

    // P value for (q_row, k_iter): stored in %f3..
    // We load from the S/P register using conditional moves
    ptx.push_str(&format!("    mov.f32 %f2, {};               // default P = 0\n", f32_bits(F32_ZERO)));
    for i in 0..num_s_per_thread {
        ptx.push_str(&format!("    setp.eq.u32 %p2, %r12, {};\n", i));
        ptx.push_str(&format!("    @%p2 mov.f32 %f2, %f{};\n", 3 + i));
    }

    // O_acc[oacc_iter] += P * V
    // We need to index into %f64+oacc_iter — use conditional add pattern
    for i in 0..num_oacc {
        ptx.push_str(&format!("    setp.eq.u32 %p3, %r8, {};\n", i));
        ptx.push_str(&format!(
            "    @%p3 fma.rn.f32 %f{}, %f2, %f1, %f{};\n",
            64 + i,
            64 + i
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
        ptx.push_str(&format!("    mul.f32 %f{}, %f{}, %f0;\n", 64 + i, 64 + i));
    }

    // ── Logsumexp auxiliary output ──────────────────────────────────
    // L = row_max + ln(row_sum) = row_max + log2(row_sum) * ln(2)
    // Only store if logsumexp_base != 0 (non-null pointer).
    // Each thread writes one value per Q row it owns.
    // For the tiled layout, thread tid_x covers q_row = tid_x within the tile
    // (simplified: only thread 0 per row writes logsumexp; full per-row
    // assignment uses the same row mapping as O_acc).
    //
    // Address: logsumexp_base + (batch_head_idx * seq_len + q_start + row_offset) * 4
    ptx.push_str("    // Logsumexp: L = row_max + log2(row_sum) * ln(2)\n");
    ptx.push_str("    setp.ne.u64 %p_has_lse, %logsumexp_base, 0;\n");
    ptx.push_str("    @!%p_has_lse bra SKIP_LSE_STORE;\n");
    // Guard: only threads with tid_x < block_q should write logsumexp
    // (128 threads per block, but only block_q rows per tile)
    ptx.push_str(&format!(
        "    setp.ge.u32 %p1, %tid_x, {};            // tid_x >= block_q?\n",
        config.block_q
    ));
    ptx.push_str("    @%p1 bra SKIP_LSE_STORE;\n");
    ptx.push_str("    lg2.approx.f32 %log_sum, %row_sum;\n");
    ptx.push_str("    mul.f32 %log_sum, %log_sum, 0F3F317218;   // * ln(2)\n");
    ptx.push_str("    add.f32 %lse, %row_max, %log_sum;\n");
    // Compute store address: logsumexp_base + (bid_y * seq_len + q_start + tid_x) * 4
    // bid_y = %rd17 (batch_head_idx), seq_len = %rd6, q_start = %rd16
    ptx.push_str("    mul.lo.u64 %lse_addr, %rd17, %rd6;         // batch_head_idx * seq_len\n");
    ptx.push_str("    add.u64 %lse_addr, %lse_addr, %rd16;       // + q_start\n");
    ptx.push_str("    cvt.u64.u32 %rd52, %tid_x;                 // tid_x as u64\n");
    ptx.push_str("    add.u64 %lse_addr, %lse_addr, %rd52;       // + tid_x (row offset)\n");
    // Bounds check: only store if (q_start + tid_x) < seq_len
    ptx.push_str("    add.u64 %rd52, %rd16, %rd52;               // q_start + tid_x\n");
    ptx.push_str("    setp.ge.u64 %p0, %rd52, %rd6;              // >= seq_len?\n");
    ptx.push_str("    @%p0 bra SKIP_LSE_STORE;\n");
    ptx.push_str("    shl.b64 %lse_addr, %lse_addr, 2;           // * 4 (sizeof f32)\n");
    ptx.push_str("    add.u64 %lse_addr, %logsumexp_base, %lse_addr; // final address\n");
    ptx.push_str("    st.global.f32 [%lse_addr], %lse;\n");
    ptx.push_str("SKIP_LSE_STORE:\n");
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
            h_reg,
            64 + i
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
        ptx.push_str(&format!("    st.global.b16 [%rd52], %h{};\n", h_reg));
    }
}

fn emit_rope_cache_write_entry(ptx: &mut String, head_dim: i64, rope_style: RopeStyle) {
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
    ptx.push_str(
        "    // 3. Apply RoPE: k_rot_a = k_a*cos - k_b*sin; k_rot_b = k_a*sin + k_b*cos\n",
    );
    ptx.push_str(
        "    // 4. Look up physical block via block_table[seq_id * max_blocks + logical_idx]\n",
    );
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
fn emit_f32_to_f16_pack(ptx: &mut String, src_f32: &[String], dst_b32: &[String]) {
    assert_eq!(
        src_f32.len(),
        dst_b32.len() * 2,
        "f16 pack: src must be 2x dst"
    );
    for i in 0..dst_b32.len() {
        let lo = &src_f32[i * 2];
        let hi = &src_f32[i * 2 + 1];
        let dst = &dst_b32[i];
        ptx.push_str(&format!("    cvt.rn.f16.f32 %mma_h0, %{};\n", lo));
        ptx.push_str(&format!("    cvt.rn.f16.f32 %mma_h1, %{};\n", hi));
        ptx.push_str(&format!("    mov.b32 %{}, {{%mma_h0, %mma_h1}};\n", dst));
    }
}

// File-local duplicates of `emit_load_a_fragment_smem`,
// `emit_load_b_fragment_smem`, and `emit_mma_instruction` were deleted
// post-N4 helper rewrite (2026-05-15). They were dead production code —
// the live call sites at lines ~1029-1056 use the
// `crate::matmul_mma::emit_load_*_fragment_smem` versions, which the N4
// helper-convention rewrite (commits 1ca2a62f + f6c9958b) brought into
// PTX m16n8k16 spec compliance. The dead local copies still emitted the
// broken single-row-fixed-col-pattern AND the `mad.lo.u32` form that
// causes CUDA_ERROR_INVALID_PTX on PTX ISA 7.0 (see MEMORY.md). Keeping
// them as `#[allow(dead_code)]` was a latent landmine: any future
// reader copying the pattern would inherit both bugs.
//
// The three `#[cfg(test)]` tests that exercised them
// (`test_load_a_fragment_emission`, `test_load_b_fragment_emission`,
// `test_qk_mma_ptx_emission`) now call the public matmul_mma versions
// directly via `crate::matmul_mma::emit_*` — they still validate the
// emission shape (comment present, expected count of `ld.shared.b32` /
// `mma.sync` lines).

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
            ptx.push_str(&format!("    mov.f32 %acc_s_{}_{}, {};\n", nt, r, f32_bits(F32_ZERO)));
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
        ptx.push_str(&format!("    // A-frag reg {}: k_pair={}\n", i, k_pair));
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
        emit_smem_load(ptx, "f32", "%mma_f32_lo", "%mma_addr");
        ptx.push_str("    add.u32 %mma_addr, %mma_addr, 4;\n");
        emit_smem_load(ptx, "f32", "%mma_f32_hi", "%mma_addr");
        ptx.push_str("    cvt.rn.f16.f32 %mma_h0, %mma_f32_lo;\n");
        ptx.push_str("    cvt.rn.f16.f32 %mma_h1, %mma_f32_hi;\n");
        ptx.push_str(&format!("    mov.b32 %aq_{}, {{%mma_h0, %mma_h1}};\n", i));
    }

    // Load B-fragments from K^T shared memory for each n-tile
    for nt in 0..n_tiles_s {
        ptx.push_str(&format!("    // K^T B-fragment for n_tile={}\n", nt));
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
            emit_smem_load(ptx, "f32", "%mma_f32_lo", "%mma_addr");
            ptx.push_str("    add.u32 %mma_addr, %mma_addr, 4;\n");
            emit_smem_load(ptx, "f32", "%mma_f32_hi", "%mma_addr");
            ptx.push_str("    cvt.rn.f16.f32 %mma_h0, %mma_f32_lo;\n");
            ptx.push_str("    cvt.rn.f16.f32 %mma_h1, %mma_f32_hi;\n");
            ptx.push_str(&format!(
                "    mov.b32 %bk_{}_{}, {{%mma_h0, %mma_h1}};\n",
                nt, bi
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
        ptx.push_str("        {%aq_0, %aq_1, %aq_2, %aq_3},\n");
        ptx.push_str(&format!("        {{%bk_{nt}_0, %bk_{nt}_1}},\n"));
        ptx.push_str(&format!(
            "        {{%acc_s_{nt}_0, %acc_s_{nt}_1, %acc_s_{nt}_2, %acc_s_{nt}_3}};\n"
        ));
    }

    // Loop back over K
    ptx.push_str("    add.u32 %mma_k_iter, %mma_k_iter, 1;\n");
    ptx.push_str(&format!(
        "    setp.lt.u32 %mma_pk, %mma_k_iter, {};\n",
        k_iters
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
fn emit_pv_matmul_mma(ptx: &mut String, block_kv: usize, head_dim: usize, shmem_k_offset: usize) {
    let n_tiles_o = head_dim / MMA_N;
    let k_iters = block_kv / MMA_K; // k-dim for P@V is block_kv

    ptx.push_str("    // === P@V via MMA (m16n8k16) ===\n");
    ptx.push_str(&format!(
        "    // n_tiles_o={}, k_iters={}\n",
        n_tiles_o, k_iters
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
        ptx.push_str(&format!("    // A-frag P reg {} from S accumulators\n", i));
        // For simplicity, read the S accumulator that maps to this fragment position
        // acc_s_{nt}_{r} where nt = k_iter*2 + (i/2), r = (i%2)*2 + laneid_mapping
        // This is approximate — actual mapping depends on MMA thread layout
        ptx.push_str(&format!(
            "    mul.lo.u32 %mma_addr, %mma_k_iter, 2;\n" // nt_base = k_iter * 2
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
            "    // Pack S acc values for A-frag position {}\n",
            i
        ));
        ptx.push_str(&format!(
            "    cvt.rn.f16.f32 %mma_h0, %acc_s_scratch_{};\n",
            r_base
        ));
        ptx.push_str(&format!(
            "    cvt.rn.f16.f32 %mma_h1, %acc_s_scratch_{};\n",
            r_base + 1
        ));
        ptx.push_str(&format!("    mov.b32 %ap_{}, {{%mma_h0, %mma_h1}};\n", i));
    }

    // Load B-fragments from V shared memory for each output n-tile
    for nt in 0..n_tiles_o {
        ptx.push_str(&format!("    // V B-fragment for n_tile={}\n", nt));
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
            emit_smem_load(ptx, "f32", "%mma_f32_lo", "%mma_addr");
            ptx.push_str("    add.u32 %mma_addr, %mma_addr, 4;\n");
            emit_smem_load(ptx, "f32", "%mma_f32_hi", "%mma_addr");
            ptx.push_str("    cvt.rn.f16.f32 %mma_h0, %mma_f32_lo;\n");
            ptx.push_str("    cvt.rn.f16.f32 %mma_h1, %mma_f32_hi;\n");
            ptx.push_str(&format!(
                "    mov.b32 %bv_{}_{}, {{%mma_h0, %mma_h1}};\n",
                nt, bi
            ));
        }
    }

    // Issue MMA for each output n-tile
    for nt in 0..n_tiles_o {
        ptx.push_str("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n");
        ptx.push_str(&format!(
            "        {{%acc_o_{nt}_0, %acc_o_{nt}_1, %acc_o_{nt}_2, %acc_o_{nt}_3}},\n"
        ));
        ptx.push_str("        {%ap_0, %ap_1, %ap_2, %ap_3},\n");
        ptx.push_str(&format!("        {{%bv_{nt}_0, %bv_{nt}_1}},\n"));
        ptx.push_str(&format!(
            "        {{%acc_o_{nt}_0, %acc_o_{nt}_1, %acc_o_{nt}_2, %acc_o_{nt}_3}};\n"
        ));
    }

    // Loop back over K
    ptx.push_str("    add.u32 %mma_k_iter, %mma_k_iter, 1;\n");
    ptx.push_str(&format!(
        "    setp.lt.u32 %mma_pk, %mma_k_iter, {};\n",
        k_iters
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
        ptx.push_str(&format!("    .reg .b32 %bk_{nt}_0, %bk_{nt}_1;\n"));
    }

    // A-fragment registers for P (4 .b32)
    ptx.push_str("    .reg .b32 %ap_0, %ap_1, %ap_2, %ap_3;\n");

    // B-fragment registers for V (n_tiles_o * 2 .b32)
    for nt in 0..n_tiles_o {
        ptx.push_str(&format!("    .reg .b32 %bv_{nt}_0, %bv_{nt}_1;\n"));
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
fn emit_mma_online_softmax(ptx: &mut String, block_kv: usize, head_dim: usize) {
    let n_tiles_s = block_kv / MMA_N;
    let n_tiles_o = head_dim / MMA_N;

    ptx.push_str("    // === Online softmax (MMA layout) ===\n");

    // Step 1: Find local max across this thread's S accumulator values
    ptx.push_str(&format!("    mov.f32 %mma_local_max, {};  // -inf\n", f32_bits(F32_NEG_INF)));
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
    ptx.push_str(&format!("    mov.f32 %mma_partial_sum, {};  // 0.0\n", f32_bits(F32_ZERO)));
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
    ptx.push_str(&format!("    mov.f32 %mma_row_max, {};  // -inf\n", f32_bits(F32_NEG_INF)));
    ptx.push_str(&format!("    mov.f32 %mma_row_sum, {};  // 0.0\n", f32_bits(F32_ZERO)));
}

/// Emit register declarations for shared memory swizzle temporaries.
#[allow(dead_code)]
fn emit_smem_swizzle_registers(ptx: &mut String) {
    ptx.push_str("    .reg .u32 %smem_linear_off, %smem_swiz_off;\n");
    ptx.push_str("    .reg .u32 %smem_bank, %smem_row_bits, %smem_bank_lo, %smem_swiz;\n");
}

// ── wgmma.mma_async PTX emission helpers (sm_90+ / Hopper) ──────────

/// wgmma tile dimensions: m64n64k16 for f16, m64n64k32 for fp8.
#[allow(dead_code)]
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
    writeln!(
        ptx,
        "    // Build wgmma shared memory descriptor for {desc_reg}"
    )
    .unwrap();
    writeln!(ptx, "    .reg .b64 %{desc_reg};").unwrap();
    writeln!(ptx, "    .reg .u32 %wgmma_desc_lo, %wgmma_desc_hi;").unwrap();
    // Base address: shift right 4 to get 16-byte units
    writeln!(
        ptx,
        "    shr.u32 %wgmma_desc_lo, {smem_base_expr}, 4;  // base in 16B units"
    )
    .unwrap();
    writeln!(
        ptx,
        "    and.b32 %wgmma_desc_lo, %wgmma_desc_lo, 0x3FFF;  // 14-bit base"
    )
    .unwrap();
    // Swizzle mode in bits [15:14]
    writeln!(
        ptx,
        "    or.b32 %wgmma_desc_lo, %wgmma_desc_lo, {};  // swizzle mode",
        swizzle_mode << 14
    )
    .unwrap();
    // Leading dimension in 16-byte units, bits [29:16]
    let ld_16b = leading_dim_bytes / 16;
    writeln!(
        ptx,
        "    or.b32 %wgmma_desc_lo, %wgmma_desc_lo, {};  // leading dim in 16B units",
        (ld_16b as u32) << 16
    )
    .unwrap();
    writeln!(
        ptx,
        "    mov.u32 %wgmma_desc_hi, 0;  // reserved upper 32 bits"
    )
    .unwrap();
    writeln!(
        ptx,
        "    mov.b64 %{desc_reg}, {{%wgmma_desc_lo, %wgmma_desc_hi}};"
    )
    .unwrap();
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

    writeln!(
        ptx,
        "    // === Q@K^T via wgmma.mma_async (m64n64k16, sm_90) ==="
    )
    .unwrap();
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
        writeln!(
            ptx,
            "    wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16"
        )
        .unwrap();
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
    writeln!(
        ptx,
        "    // --- Softmax scalar math can overlap here (async!) ---"
    )
    .unwrap();
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
            writeln!(
                ptx,
                "    mul.f32 %wg_acc_s_{nt}_{r}, %wg_acc_s_{nt}_{r}, %scale;"
            )
            .unwrap();
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

    writeln!(
        ptx,
        "    // === P@V via wgmma.mma_async (m64n64k16, sm_90) ==="
    )
    .unwrap();
    writeln!(
        ptx,
        "    // P staged to shared memory, V already in shared memory"
    )
    .unwrap();

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
        writeln!(
            ptx,
            "    wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16"
        )
        .unwrap();
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
        writeln!(
            ptx,
            "    .reg .b64 %wg_desc_kt_{nt};  // K^T tile {nt} descriptor"
        )
        .unwrap();
    }
    writeln!(
        ptx,
        "    .reg .b64 %wg_desc_p;  // P tile descriptor (softmax output)"
    )
    .unwrap();
    for nt in 0..n_tiles_o {
        writeln!(
            ptx,
            "    .reg .b64 %wg_desc_v_{nt};  // V tile {nt} descriptor"
        )
        .unwrap();
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
    ptx.push_str(
        "    shr.u32 %mma_addr, %mma_laneid, 4;     // laneid / 16 (reuse mma_addr as temp)\n",
    );
    ptx.push_str(
        "    add.u32 %mma_a_row, %mma_a_row, %mma_addr;  // row = (laneid%4)*2 + laneid/16\n",
    );
    // B-fragment row mapping: same as A row for the k-dimension
    ptx.push_str(
        "    mov.u32 %mma_b_row, %mma_a_row;        // B row mapping matches A for k-dim\n",
    );
}

// ── FlashAttention-2 Backward Pass ───────────────────────────────────────

/// Configuration for a FlashAttention backward PTX kernel variant.
#[derive(Clone, Debug)]
pub struct FlashAttentionBackwardConfig {
    pub block_q: i64,
    pub block_kv: i64,
    pub head_dim: i64,
    pub causal: bool,
    pub gpu_sm: u32,
    /// PCA Tier A: propagated from the upstream
    /// `FlashAttentionConfig::segment_masked`. When `true`, the
    /// backward kernel accepts a `segment_ids: *const u16` parameter
    /// and masks `dS[i, j]` by `segment_ids[i] == segment_ids[j]`
    /// alongside the causal mask (spec §3.4). No mutual-exclusion
    /// check is needed here because backward configs are always
    /// derived from a validated forward config.
    pub segment_masked: bool,
}

impl FlashAttentionBackwardConfig {
    /// Spec §3.2 invariant: segment_masked is propagated from the forward
    /// config. Reserved for future backward emission gating (spec §3.2).
    pub fn validate(&self) -> Result<(), String> {
        Ok(())
    }
}

/// Kernel name for the Phase 1 D-correction vector kernel.
pub fn flash_attention_bwd_d_kernel_name(config: &FlashAttentionBackwardConfig) -> String {
    format!("flash_attn_bwd_d_c{}_q{}", config.causal as u8, config.block_q)
}

/// Kernel name for the Phase 2 main backward kernel (dQ/dK/dV).
pub fn flash_attention_bwd_main_kernel_name(config: &FlashAttentionBackwardConfig) -> String {
    format!(
        "flash_attn_bwd_main_c{}_q{}_kv{}",
        config.causal as u8, config.block_q, config.block_kv
    )
}

/// Compute shared memory bytes needed by the Phase 2 backward kernel.
pub fn backward_shared_mem_bytes(config: &FlashAttentionBackwardConfig) -> u32 {
    let pad = 4i64;
    let hd_padded = config.head_dim + pad;
    let tile_bytes = |rows: i64, cols: i64| -> i64 { rows * cols * 4 };
    let q_tile = tile_bytes(config.block_q, hd_padded);
    let k_tile = tile_bytes(config.block_kv, hd_padded);
    let v_tile = tile_bytes(config.block_kv, hd_padded);
    let do_tile = tile_bytes(config.block_q, hd_padded);
    let dk_local = tile_bytes(config.block_kv, hd_padded);
    let dv_local = tile_bytes(config.block_kv, hd_padded);
    let s_tile = tile_bytes(config.block_q, config.block_kv);
    // dP tile: workspace for MMA dP = dO@V^T results (needed alongside P in S_tile for dS computation)
    let dp_tile = if use_mma_path(config.gpu_sm) {
        tile_bytes(config.block_q, config.block_kv)
    } else {
        0
    };
    let d_vec = config.block_q * 4;
    let l_vec = config.block_q * 4;
    (q_tile + k_tile + v_tile + do_tile + dk_local + dv_local + s_tile + dp_tile + d_vec + l_vec)
        as u32
}

/// Synthesize PTX for the FlashAttention-2 backward pass.
///
/// Returns a tuple of two null-terminated PTX byte vectors:
///   - `.0` — Phase 1: D-correction vector kernel
///   - `.1` — Phase 2: main dQ/dK/dV kernel (placeholder for now)
pub fn synthesize_flash_attention_backward_ptx(
    config: &FlashAttentionBackwardConfig,
) -> (Vec<u8>, Vec<u8>) {
    // Phase 1: D-correction vector
    let mut ptx1 = String::with_capacity(2048);
    emit_ptx_header(&mut ptx1, config.gpu_sm);
    emit_flash_attention_bwd_d(&mut ptx1, config);
    ptx1.push('\0');

    // Phase 2: main backward kernel (dQ/dK/dV)
    let mut ptx2 = String::with_capacity(16384);
    emit_ptx_header(&mut ptx2, config.gpu_sm);
    emit_flash_attention_bwd_main(&mut ptx2, config);
    ptx2.push('\0');

    (ptx1.into_bytes(), ptx2.into_bytes())
}

/// Emit the Phase 1 D-correction vector kernel.
///
/// Computes `D[bh, i] = sum_d( dO[bh, i, d] * O[bh, i, d] )` for each position i.
/// Grid: `(B*nh, ceil(S/block_q), 1)` with `block_q` threads per block.
fn emit_flash_attention_bwd_d(ptx: &mut String, config: &FlashAttentionBackwardConfig) {
    let kernel_name = flash_attention_bwd_d_kernel_name(config);

    // Entry point and parameters
    ptx.push_str(&format!(".visible .entry {} (\n", kernel_name));
    ptx.push_str("    .param .u64 dout_ptr,\n");
    ptx.push_str("    .param .u64 out_ptr,\n");
    ptx.push_str("    .param .u64 d_ptr,\n");
    ptx.push_str("    .param .u64 seq_len,\n");
    ptx.push_str("    .param .u64 head_dim\n");
    ptx.push_str(")\n");
    ptx.push_str("{\n");

    // Register declarations
    ptx.push_str("    .reg .u32 %r<10>;\n");
    ptx.push_str("    .reg .u64 %rd<20>;\n");
    ptx.push_str("    .reg .f32 %f<4>;\n");
    ptx.push_str("    .reg .pred %p<3>;\n\n");

    // Load parameters
    ptx.push_str("    ld.param.u64 %rd1, [dout_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd2, [out_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd3, [d_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd4, [seq_len];\n");
    ptx.push_str("    ld.param.u64 %rd5, [head_dim];\n\n");

    // Compute bh = blockIdx.x
    ptx.push_str("    // bh = blockIdx.x\n");
    ptx.push_str("    mov.u32 %r1, %ctaid.x;\n");
    ptx.push_str("    cvt.u64.u32 %rd6, %r1;\n\n");

    // Compute pos = blockIdx.y * blockDim.x + threadIdx.x
    ptx.push_str("    // pos = blockIdx.y * blockDim.x + threadIdx.x\n");
    ptx.push_str("    mov.u32 %r2, %ctaid.y;\n");
    ptx.push_str("    mov.u32 %r3, %ntid.x;\n");
    ptx.push_str("    mul.lo.u32 %r4, %r2, %r3;\n");
    ptx.push_str("    mov.u32 %r5, %tid.x;\n");
    ptx.push_str("    add.u32 %r4, %r4, %r5;\n");
    ptx.push_str("    cvt.u64.u32 %rd7, %r4;\n\n");

    // Bounds check: if pos >= seq_len, skip
    ptx.push_str("    // Bounds check\n");
    ptx.push_str("    setp.ge.u64 %p1, %rd7, %rd4;\n");
    ptx.push_str(&format!("    @%p1 bra {}_DONE;\n\n", kernel_name));

    // base = (bh * seq_len + pos) * head_dim, byte offset = base * 4
    ptx.push_str("    // base = (bh * seq_len + pos) * head_dim\n");
    ptx.push_str("    mul.lo.u64 %rd8, %rd6, %rd4;\n");  // bh * seq_len
    ptx.push_str("    add.u64 %rd8, %rd8, %rd7;\n");      // + pos
    ptx.push_str("    mul.lo.u64 %rd8, %rd8, %rd5;\n");   // * head_dim
    ptx.push_str("    shl.b64 %rd8, %rd8, 2;\n\n");       // * 4 (sizeof f32)

    // Compute dout_base and out_base pointers
    ptx.push_str("    // dout_base, out_base pointers\n");
    ptx.push_str("    add.u64 %rd9, %rd1, %rd8;\n");
    ptx.push_str("    add.u64 %rd10, %rd2, %rd8;\n\n");

    // Dot product loop: sum = sum_d(dO[base+d] * O[base+d])
    ptx.push_str("    // Dot product across head_dim\n");
    ptx.push_str("    mov.f32 %f1, 0f00000000;\n");        // sum = 0.0
    ptx.push_str("    mov.u64 %rd11, 0;\n");               // d = 0
    ptx.push_str(&format!("{}_LOOP:\n", kernel_name));
    ptx.push_str("    setp.ge.u64 %p2, %rd11, %rd5;\n");
    ptx.push_str(&format!("    @%p2 bra {}_STORE;\n", kernel_name));
    ptx.push_str("    shl.b64 %rd12, %rd11, 2;\n");        // d * 4
    ptx.push_str("    add.u64 %rd13, %rd9, %rd12;\n");
    ptx.push_str("    add.u64 %rd14, %rd10, %rd12;\n");
    ptx.push_str("    ld.global.f32 %f2, [%rd13];\n");
    ptx.push_str("    ld.global.f32 %f3, [%rd14];\n");
    ptx.push_str("    fma.rn.f32 %f1, %f2, %f3, %f1;\n"); // sum += dout * out
    ptx.push_str("    add.u64 %rd11, %rd11, 1;\n");
    ptx.push_str(&format!("    bra {}_LOOP;\n\n", kernel_name));

    // Store D[bh * seq_len + pos]
    ptx.push_str(&format!("{}_STORE:\n", kernel_name));
    ptx.push_str("    // D[bh * seq_len + pos] = sum\n");
    ptx.push_str("    mul.lo.u64 %rd15, %rd6, %rd4;\n");  // bh * seq_len
    ptx.push_str("    add.u64 %rd15, %rd15, %rd7;\n");     // + pos
    ptx.push_str("    shl.b64 %rd15, %rd15, 2;\n");        // * 4
    ptx.push_str("    add.u64 %rd15, %rd3, %rd15;\n");
    ptx.push_str("    st.global.f32 [%rd15], %f1;\n\n");

    ptx.push_str(&format!("{}_DONE:\n", kernel_name));
    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");
}

// ---------------------------------------------------------------------------
// Phase 2: Main backward kernel (dQ / dK / dV)
// ---------------------------------------------------------------------------
//
// Grid: (B*nh, ceil(S/block_kv), 1)
// Each thread-block owns one (batch-head, kv-tile). The inner loop sweeps over
// Q tiles, computing dK/dV locally and atomicAdd-ing into dQ.
//
// Thread assignment: block_q threads per block. Each thread owns one Q-row (mi)
// and iterates over KV positions and head_dim sequentially (scalar FMA path).

/// Shared memory padding (f32 elements) added to head_dim in 2-D tiles to avoid
/// bank conflicts during transpose-style access patterns.
const BWD_PAD: i64 = 4;

/// Emit the complete Phase 2 backward main kernel PTX.
fn emit_flash_attention_bwd_main(ptx: &mut String, config: &FlashAttentionBackwardConfig) {
    let kernel_name = flash_attention_bwd_main_kernel_name(config);

    // Entry point
    ptx.push_str(&format!(".visible .entry {} (\n", kernel_name));
    ptx.push_str("    .param .u64 param_dout,\n");
    ptx.push_str("    .param .u64 param_q,\n");
    ptx.push_str("    .param .u64 param_k,\n");
    ptx.push_str("    .param .u64 param_v,\n");
    ptx.push_str("    .param .u64 param_dq,\n");
    ptx.push_str("    .param .u64 param_dk,\n");
    ptx.push_str("    .param .u64 param_dv,\n");
    ptx.push_str("    .param .u64 param_d,\n");
    ptx.push_str("    .param .u64 param_lse,\n");
    ptx.push_str("    .param .f32 param_scale,\n");
    ptx.push_str("    .param .u64 param_seq_len,\n");
    ptx.push_str("    .param .u64 param_head_dim\n");
    ptx.push_str(")\n");
    ptx.push_str("{\n");

    // Shared memory
    let shmem_bytes = backward_shared_mem_bytes(config);
    ptx.push_str(&format!(
        "    .shared .align 16 .b8 shmem[{}];\n\n",
        shmem_bytes
    ));

    emit_bwd_main_registers(ptx, config);
    if use_mma_path(config.gpu_sm) {
        emit_bwd_mma_registers(ptx, config);
    }
    emit_bwd_main_param_loads(ptx, config);
    emit_bwd_main_index_computation(ptx, config);
    emit_bwd_main_load_kv_tiles(ptx, config);
    emit_bwd_main_init_dk_dv(ptx, config);
    if use_mma_path(config.gpu_sm) {
        emit_bwd_main_q_tile_loop_mma(ptx, config);
    } else {
        emit_bwd_main_q_tile_loop_scalar(ptx, config);
    }
    emit_bwd_main_store_dk_dv(ptx, config);

    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");
}

/// Register declarations for the backward main kernel.
fn emit_bwd_main_registers(ptx: &mut String, _config: &FlashAttentionBackwardConfig) {
    ptx.push_str("    // === Register declarations ===\n");
    ptx.push_str("    .reg .u32 %r<32>;\n");
    ptx.push_str("    .reg .u64 %rd<64>;\n");
    ptx.push_str("    .reg .f32 %f<32>;\n");
    ptx.push_str("    .reg .pred %p<16>;\n");
    ptx.push_str("    .reg .f32 %scale;\n");
    ptx.push_str("    .reg .f32 %log2e;\n");
    ptx.push_str("    mov.f32 %log2e, 0f3FB8AA3B;  // 1.4426950408 (log2(e))\n");
    // Named accumulators for clarity
    ptx.push_str("    .reg .f32 %f_sum;\n");
    ptx.push_str("    .reg .f32 %f_s;\n");
    ptx.push_str("    .reg .f32 %f_p;\n");
    ptx.push_str("    .reg .f32 %f_dp;\n");
    ptx.push_str("    .reg .f32 %f_ds;\n");
    ptx.push_str("    .reg .f32 %f_val;\n");
    ptx.push_str("    .reg .f32 %f_d_val;\n");
    ptx.push_str("    .reg .f32 %f_l_val;\n");
    ptx.push_str("    .reg .f32 %f_tmp;\n");
    ptx.push_str("    .reg .f32 %f_discard;  // sink for atom.shared.add.f32 return value\n\n");
}

/// Load kernel parameters into registers.
fn emit_bwd_main_param_loads(ptx: &mut String, _config: &FlashAttentionBackwardConfig) {
    ptx.push_str("    // === Parameter loads ===\n");
    ptx.push_str("    ld.param.u64 %rd0, [param_dout];\n");
    ptx.push_str("    ld.param.u64 %rd1, [param_q];\n");
    ptx.push_str("    ld.param.u64 %rd2, [param_k];\n");
    ptx.push_str("    ld.param.u64 %rd3, [param_v];\n");
    ptx.push_str("    ld.param.u64 %rd4, [param_dq];\n");
    ptx.push_str("    ld.param.u64 %rd5, [param_dk];\n");
    ptx.push_str("    ld.param.u64 %rd6, [param_dv];\n");
    ptx.push_str("    ld.param.u64 %rd7, [param_d];\n");
    ptx.push_str("    ld.param.u64 %rd8, [param_lse];\n");
    ptx.push_str("    ld.param.f32 %scale, [param_scale];\n");
    ptx.push_str("    ld.param.u64 %rd9, [param_seq_len];\n");
    ptx.push_str("    ld.param.u64 %rd10, [param_head_dim];\n\n");
}

/// Compute thread/block indices and the base pointers for this thread-block's
/// (batch-head, kv-tile) assignment.
fn emit_bwd_main_index_computation(ptx: &mut String, config: &FlashAttentionBackwardConfig) {
    ptx.push_str("    // === Index computation ===\n");
    ptx.push_str("    // blockIdx.x = bh (batch*head), blockIdx.y = j_block (KV tile)\n");
    ptx.push_str("    mov.u32 %r0, %tid.x;\n");
    ptx.push_str("    mov.u32 %r1, %ctaid.x;\n");
    ptx.push_str("    mov.u32 %r2, %ctaid.y;\n");
    ptx.push_str("    cvt.u64.u32 %rd11, %r0;  // tid\n");
    ptx.push_str("    cvt.u64.u32 %rd12, %r1;  // bh\n");
    ptx.push_str("    cvt.u64.u32 %rd13, %r2;  // j_block\n\n");

    // kv_start = j_block * block_kv
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd14, %rd13, {};  // kv_start = j_block * block_kv\n",
        config.block_kv
    ));

    // bh_offset = bh * seq_len * head_dim  (element offset into [B*nh, S, hd])
    ptx.push_str("    mul.lo.u64 %rd15, %rd12, %rd9;   // bh * seq_len\n");
    ptx.push_str("    mul.lo.u64 %rd15, %rd15, %rd10;   // * head_dim  => bh_elem_offset\n");

    // bh_seq_offset = bh * seq_len  (element offset into [B*nh, S])
    ptx.push_str("    mul.lo.u64 %rd16, %rd12, %rd9;    // bh * seq_len => bh_seq_offset\n\n");

    // num_q_tiles = ceil(seq_len / block_q)
    ptx.push_str(&format!(
        "    add.u64 %rd17, %rd9, {};  // seq_len + block_q - 1\n",
        config.block_q - 1
    ));
    ptx.push_str(&format!(
        "    div.u64 %rd17, %rd17, {};  // num_q_tiles = ceil(seq_len / block_q)\n\n",
        config.block_q
    ));
}

/// Load K[j_block] and V[j_block] tiles into shared memory.
fn emit_bwd_main_load_kv_tiles(ptx: &mut String, config: &FlashAttentionBackwardConfig) {
    let hd_padded = config.head_dim + BWD_PAD;
    let k_tile_elems = config.block_kv * hd_padded;
    let _v_tile_elems = config.block_kv * hd_padded;
    let k_tile_bytes = k_tile_elems * 4;
    let v_offset = k_tile_bytes;

    ptx.push_str("    // === Load K[j_block] and V[j_block] into shared memory ===\n");

    // K global base = k_ptr + (bh_elem_offset + kv_start * head_dim) * 4
    ptx.push_str("    mul.lo.u64 %rd18, %rd14, %rd10;   // kv_start * head_dim\n");
    ptx.push_str("    add.u64 %rd18, %rd15, %rd18;       // bh_elem_offset + kv_start*hd\n");
    ptx.push_str("    shl.b64 %rd18, %rd18, 2;           // * 4 bytes\n");
    ptx.push_str("    add.u64 %rd18, %rd2, %rd18;        // k_global_base\n\n");

    // V global base (same layout)
    ptx.push_str("    mul.lo.u64 %rd19, %rd14, %rd10;   // kv_start * head_dim\n");
    ptx.push_str("    add.u64 %rd19, %rd15, %rd19;\n");
    ptx.push_str("    shl.b64 %rd19, %rd19, 2;\n");
    ptx.push_str("    add.u64 %rd19, %rd3, %rd19;        // v_global_base\n\n");

    // Cooperative load K tile: thread tid loads rows starting at tid, stride block_q
    // Each row has head_dim elements in global, stored with hd_padded stride in shmem
    ptx.push_str("    // Load K tile (with padding) into shmem[0..]\n");
    ptx.push_str("    mov.u64 %rd20, %rd11;  // nj = tid\n");
    ptx.push_str("BWD_MAIN_LOAD_K:\n");
    ptx.push_str(&format!(
        "    setp.ge.u64 %p0, %rd20, {};\n",
        config.block_kv
    ));
    ptx.push_str("    @%p0 bra BWD_MAIN_LOAD_K_DONE;\n");

    // For each row nj: load head_dim elements from global, store to shmem with padded stride
    ptx.push_str("    mov.u64 %rd21, 0;  // d = 0\n");
    ptx.push_str("BWD_MAIN_LOAD_K_D:\n");
    ptx.push_str("    setp.ge.u64 %p1, %rd21, %rd10;\n");
    ptx.push_str("    @%p1 bra BWD_MAIN_LOAD_K_D_DONE;\n");
    // global addr: k_global_base + (nj * head_dim + d) * 4
    ptx.push_str("    mul.lo.u64 %rd22, %rd20, %rd10;\n");
    ptx.push_str("    add.u64 %rd22, %rd22, %rd21;\n");
    ptx.push_str("    shl.b64 %rd22, %rd22, 2;\n");
    ptx.push_str("    add.u64 %rd22, %rd18, %rd22;\n");
    ptx.push_str("    ld.global.f32 %f0, [%rd22];\n");
    // shmem addr: (nj * hd_padded + d) * 4
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd23, %rd20, {};\n",
        hd_padded
    ));
    ptx.push_str("    add.u64 %rd23, %rd23, %rd21;\n");
    ptx.push_str("    shl.b64 %rd23, %rd23, 2;\n");
    emit_smem_store(ptx, "f32", "%rd23", "%f0");
    ptx.push_str("    add.u64 %rd21, %rd21, 1;\n");
    ptx.push_str("    bra BWD_MAIN_LOAD_K_D;\n");
    ptx.push_str("BWD_MAIN_LOAD_K_D_DONE:\n");
    ptx.push_str(&format!(
        "    add.u64 %rd20, %rd20, {};  // nj += block_q (stride by num threads)\n",
        config.block_q
    ));
    ptx.push_str("    bra BWD_MAIN_LOAD_K;\n");
    ptx.push_str("BWD_MAIN_LOAD_K_DONE:\n\n");

    // Load V tile: shmem offset = v_offset
    ptx.push_str(&format!(
        "    // Load V tile into shmem[{}..]\n",
        v_offset
    ));
    ptx.push_str("    mov.u64 %rd20, %rd11;  // nj = tid\n");
    ptx.push_str("BWD_MAIN_LOAD_V:\n");
    ptx.push_str(&format!(
        "    setp.ge.u64 %p0, %rd20, {};\n",
        config.block_kv
    ));
    ptx.push_str("    @%p0 bra BWD_MAIN_LOAD_V_DONE;\n");
    ptx.push_str("    mov.u64 %rd21, 0;  // d = 0\n");
    ptx.push_str("BWD_MAIN_LOAD_V_D:\n");
    ptx.push_str("    setp.ge.u64 %p1, %rd21, %rd10;\n");
    ptx.push_str("    @%p1 bra BWD_MAIN_LOAD_V_D_DONE;\n");
    // global addr: v_global_base + (nj * head_dim + d) * 4
    ptx.push_str("    mul.lo.u64 %rd22, %rd20, %rd10;\n");
    ptx.push_str("    add.u64 %rd22, %rd22, %rd21;\n");
    ptx.push_str("    shl.b64 %rd22, %rd22, 2;\n");
    ptx.push_str("    add.u64 %rd22, %rd19, %rd22;\n");
    ptx.push_str("    ld.global.f32 %f0, [%rd22];\n");
    // shmem addr: v_offset + (nj * hd_padded + d) * 4
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd23, %rd20, {};\n",
        hd_padded
    ));
    ptx.push_str("    add.u64 %rd23, %rd23, %rd21;\n");
    ptx.push_str("    shl.b64 %rd23, %rd23, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd23, %rd23, {};  // + V tile shmem offset\n",
        v_offset
    ));
    emit_smem_store(ptx, "f32", "%rd23", "%f0");
    ptx.push_str("    add.u64 %rd21, %rd21, 1;\n");
    ptx.push_str("    bra BWD_MAIN_LOAD_V_D;\n");
    ptx.push_str("BWD_MAIN_LOAD_V_D_DONE:\n");
    ptx.push_str(&format!(
        "    add.u64 %rd20, %rd20, {};  // nj += block_q\n",
        config.block_q
    ));
    ptx.push_str("    bra BWD_MAIN_LOAD_V;\n");
    ptx.push_str("BWD_MAIN_LOAD_V_DONE:\n");

    ptx.push_str("    bar.sync 0;  // K and V tiles loaded\n\n");
}

/// Zero-initialize dK_local and dV_local in shared memory.
fn emit_bwd_main_init_dk_dv(ptx: &mut String, config: &FlashAttentionBackwardConfig) {
    let hd_padded = config.head_dim + BWD_PAD;
    let k_tile_bytes = config.block_kv * hd_padded * 4;
    let v_tile_bytes = config.block_kv * hd_padded * 4;
    let q_tile_bytes = config.block_q * hd_padded * 4;
    let do_tile_bytes = config.block_q * hd_padded * 4;
    let dk_offset = k_tile_bytes + v_tile_bytes + q_tile_bytes + do_tile_bytes;
    let dk_elems = config.block_kv * hd_padded;
    let dv_offset = dk_offset + dk_elems * 4;
    let dv_elems = config.block_kv * hd_padded;

    ptx.push_str("    // === Zero-initialize dK_local and dV_local in shared memory ===\n");
    ptx.push_str("    mov.u64 %rd20, %rd11;  // idx = tid\n");
    ptx.push_str("BWD_MAIN_ZERO_DK:\n");
    ptx.push_str(&format!(
        "    setp.ge.u64 %p0, %rd20, {};\n",
        dk_elems
    ));
    ptx.push_str("    @%p0 bra BWD_MAIN_ZERO_DK_DONE;\n");
    ptx.push_str("    shl.b64 %rd21, %rd20, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd21, %rd21, {};  // + dK shmem offset\n",
        dk_offset
    ));
    emit_smem_store(ptx, "f32", "%rd21", "0f00000000");
    ptx.push_str(&format!(
        "    add.u64 %rd20, %rd20, {};  // stride by block_q threads\n",
        config.block_q
    ));
    ptx.push_str("    bra BWD_MAIN_ZERO_DK;\n");
    ptx.push_str("BWD_MAIN_ZERO_DK_DONE:\n\n");

    ptx.push_str("    mov.u64 %rd20, %rd11;  // idx = tid\n");
    ptx.push_str("BWD_MAIN_ZERO_DV:\n");
    ptx.push_str(&format!(
        "    setp.ge.u64 %p0, %rd20, {};\n",
        dv_elems
    ));
    ptx.push_str("    @%p0 bra BWD_MAIN_ZERO_DV_DONE;\n");
    ptx.push_str("    shl.b64 %rd21, %rd20, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd21, %rd21, {};  // + dV shmem offset\n",
        dv_offset
    ));
    emit_smem_store(ptx, "f32", "%rd21", "0f00000000");
    ptx.push_str(&format!(
        "    add.u64 %rd20, %rd20, {};  // stride by block_q threads\n",
        config.block_q
    ));
    ptx.push_str("    bra BWD_MAIN_ZERO_DV;\n");
    ptx.push_str("BWD_MAIN_ZERO_DV_DONE:\n");

    ptx.push_str("    bar.sync 0;  // dK/dV zeroed\n\n");
}

/// Emit the inner Q-tile loop using scalar FMA path (sm_52+).
/// All 7 computation steps (3a-3g) use per-thread scalar dot products.
fn emit_bwd_main_q_tile_loop_scalar(ptx: &mut String, config: &FlashAttentionBackwardConfig) {
    let hd_padded = config.head_dim + BWD_PAD;
    let k_tile_bytes = config.block_kv * hd_padded * 4;
    let v_tile_bytes = config.block_kv * hd_padded * 4;
    let q_shmem_offset = k_tile_bytes + v_tile_bytes;
    let do_shmem_offset = q_shmem_offset + config.block_q * hd_padded * 4;
    let dk_shmem_offset = do_shmem_offset + config.block_q * hd_padded * 4;
    let dv_shmem_offset = dk_shmem_offset + config.block_kv * hd_padded * 4;
    let s_shmem_offset = dv_shmem_offset + config.block_kv * hd_padded * 4;
    let d_shmem_offset = s_shmem_offset + config.block_q * config.block_kv * 4;
    let l_shmem_offset = d_shmem_offset + config.block_q * 4;

    ptx.push_str("    // === Inner Q-tile loop ===\n");

    // i_block loop: for causal, start at j_block; otherwise start at 0
    if config.causal {
        ptx.push_str("    mov.u64 %rd24, %rd13;  // i_block = j_block (causal)\n");
    } else {
        ptx.push_str("    mov.u64 %rd24, 0;  // i_block = 0 (non-causal)\n");
    }

    ptx.push_str("BWD_MAIN_Q_LOOP:\n");
    ptx.push_str("    setp.ge.u64 %p0, %rd24, %rd17;  // i_block >= num_q_tiles?\n");
    ptx.push_str("    @%p0 bra BWD_MAIN_Q_LOOP_END;\n\n");

    // q_start = i_block * block_q
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd25, %rd24, {};  // q_start = i_block * block_q\n",
        config.block_q
    ));

    // Load Q[i_block] into shmem[q_shmem_offset..]
    ptx.push_str(&format!(
        "    // Load Q[i_block] into shmem[{}..]\n",
        q_shmem_offset
    ));
    // q_global_base = q_ptr + (bh_elem_offset + q_start * head_dim) * 4
    ptx.push_str("    mul.lo.u64 %rd26, %rd25, %rd10;  // q_start * head_dim\n");
    ptx.push_str("    add.u64 %rd26, %rd15, %rd26;\n");
    ptx.push_str("    shl.b64 %rd26, %rd26, 2;\n");
    ptx.push_str("    add.u64 %rd26, %rd1, %rd26;  // q_global_base\n");

    // Cooperative load Q tile rows
    ptx.push_str("    mov.u64 %rd27, %rd11;  // mi = tid\n");
    ptx.push_str("BWD_MAIN_LOAD_Q:\n");
    ptx.push_str(&format!(
        "    setp.ge.u64 %p1, %rd27, {};\n",
        config.block_q
    ));
    ptx.push_str("    @%p1 bra BWD_MAIN_LOAD_Q_DONE;\n");
    ptx.push_str("    mov.u64 %rd28, 0;  // d = 0\n");
    ptx.push_str("BWD_MAIN_LOAD_Q_D:\n");
    ptx.push_str("    setp.ge.u64 %p2, %rd28, %rd10;\n");
    ptx.push_str("    @%p2 bra BWD_MAIN_LOAD_Q_D_DONE;\n");
    ptx.push_str("    mul.lo.u64 %rd29, %rd27, %rd10;\n");
    ptx.push_str("    add.u64 %rd29, %rd29, %rd28;\n");
    ptx.push_str("    shl.b64 %rd29, %rd29, 2;\n");
    ptx.push_str("    add.u64 %rd29, %rd26, %rd29;\n");
    ptx.push_str("    ld.global.f32 %f0, [%rd29];\n");
    // shmem: q_shmem_offset + (mi * hd_padded + d) * 4
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd30, %rd27, {};\n",
        hd_padded
    ));
    ptx.push_str("    add.u64 %rd30, %rd30, %rd28;\n");
    ptx.push_str("    shl.b64 %rd30, %rd30, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd30, %rd30, {};\n",
        q_shmem_offset
    ));
    emit_smem_store(ptx, "f32", "%rd30", "%f0");
    ptx.push_str("    add.u64 %rd28, %rd28, 1;\n");
    ptx.push_str("    bra BWD_MAIN_LOAD_Q_D;\n");
    ptx.push_str("BWD_MAIN_LOAD_Q_D_DONE:\n");
    ptx.push_str(&format!(
        "    add.u64 %rd27, %rd27, {};\n",
        config.block_q
    ));
    ptx.push_str("    bra BWD_MAIN_LOAD_Q;\n");
    ptx.push_str("BWD_MAIN_LOAD_Q_DONE:\n\n");

    // Load dO[i_block] into shmem[do_shmem_offset..]
    ptx.push_str(&format!(
        "    // Load dO[i_block] into shmem[{}..]\n",
        do_shmem_offset
    ));
    ptx.push_str("    mul.lo.u64 %rd26, %rd25, %rd10;\n");
    ptx.push_str("    add.u64 %rd26, %rd15, %rd26;\n");
    ptx.push_str("    shl.b64 %rd26, %rd26, 2;\n");
    ptx.push_str("    add.u64 %rd26, %rd0, %rd26;  // dO_global_base\n");

    ptx.push_str("    mov.u64 %rd27, %rd11;\n");
    ptx.push_str("BWD_MAIN_LOAD_DO:\n");
    ptx.push_str(&format!(
        "    setp.ge.u64 %p1, %rd27, {};\n",
        config.block_q
    ));
    ptx.push_str("    @%p1 bra BWD_MAIN_LOAD_DO_DONE;\n");
    ptx.push_str("    mov.u64 %rd28, 0;\n");
    ptx.push_str("BWD_MAIN_LOAD_DO_D:\n");
    ptx.push_str("    setp.ge.u64 %p2, %rd28, %rd10;\n");
    ptx.push_str("    @%p2 bra BWD_MAIN_LOAD_DO_D_DONE;\n");
    ptx.push_str("    mul.lo.u64 %rd29, %rd27, %rd10;\n");
    ptx.push_str("    add.u64 %rd29, %rd29, %rd28;\n");
    ptx.push_str("    shl.b64 %rd29, %rd29, 2;\n");
    ptx.push_str("    add.u64 %rd29, %rd26, %rd29;\n");
    ptx.push_str("    ld.global.f32 %f0, [%rd29];\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd30, %rd27, {};\n",
        hd_padded
    ));
    ptx.push_str("    add.u64 %rd30, %rd30, %rd28;\n");
    ptx.push_str("    shl.b64 %rd30, %rd30, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd30, %rd30, {};\n",
        do_shmem_offset
    ));
    emit_smem_store(ptx, "f32", "%rd30", "%f0");
    ptx.push_str("    add.u64 %rd28, %rd28, 1;\n");
    ptx.push_str("    bra BWD_MAIN_LOAD_DO_D;\n");
    ptx.push_str("BWD_MAIN_LOAD_DO_D_DONE:\n");
    ptx.push_str(&format!(
        "    add.u64 %rd27, %rd27, {};\n",
        config.block_q
    ));
    ptx.push_str("    bra BWD_MAIN_LOAD_DO;\n");
    ptx.push_str("BWD_MAIN_LOAD_DO_DONE:\n\n");

    // Load D[i_block] and L[i_block] vectors into shmem
    ptx.push_str("    // Load D[i_block] and L[i_block] into shmem\n");
    // Each thread loads one element if tid < block_q
    ptx.push_str(&format!(
        "    setp.ge.u64 %p1, %rd11, {};  // tid >= block_q?\n",
        config.block_q
    ));
    ptx.push_str("    @%p1 bra BWD_MAIN_LOAD_DL_DONE;\n");
    // D[bh * seq_len + q_start + tid]
    ptx.push_str("    add.u64 %rd26, %rd16, %rd25;  // bh_seq_offset + q_start\n");
    ptx.push_str("    add.u64 %rd26, %rd26, %rd11;  // + tid\n");
    ptx.push_str("    shl.b64 %rd26, %rd26, 2;\n");
    ptx.push_str("    add.u64 %rd27, %rd7, %rd26;   // d_ptr + offset\n");
    ptx.push_str("    ld.global.f32 %f0, [%rd27];\n");
    // Store to D_vec in shmem
    ptx.push_str("    shl.b64 %rd28, %rd11, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd28, %rd28, {};  // + D_vec shmem offset\n",
        d_shmem_offset
    ));
    emit_smem_store(ptx, "f32", "%rd28", "%f0");
    // L[bh * seq_len + q_start + tid]
    ptx.push_str("    add.u64 %rd27, %rd8, %rd26;   // lse_ptr + offset\n");
    ptx.push_str("    ld.global.f32 %f0, [%rd27];\n");
    ptx.push_str("    shl.b64 %rd28, %rd11, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd28, %rd28, {};  // + L_vec shmem offset\n",
        l_shmem_offset
    ));
    emit_smem_store(ptx, "f32", "%rd28", "%f0");
    ptx.push_str("BWD_MAIN_LOAD_DL_DONE:\n");
    ptx.push_str("    bar.sync 0;  // Q, dO, D, L loaded\n\n");

    // Each thread is assigned mi = tid (one Q row per thread).
    // Thread iterates over all KV positions nj in the tile.
    ptx.push_str("    // === Compute steps 3a-3g for mi = tid ===\n");
    ptx.push_str("    // Each thread handles one Q row (mi = tid)\n");
    ptx.push_str(&format!(
        "    setp.ge.u64 %p0, %rd11, {};  // skip if tid >= block_q\n",
        config.block_q
    ));
    ptx.push_str("    @%p0 bra BWD_MAIN_STEPS_DONE;\n\n");

    // Load D[mi] and L[mi] from shmem
    ptx.push_str("    shl.b64 %rd26, %rd11, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd27, %rd26, {};\n",
        d_shmem_offset
    ));
    emit_smem_load(ptx, "f32", "%f_d_val", "%rd27"); // D[mi]
    ptx.push_str(&format!(
        "    add.u64 %rd27, %rd26, {};\n",
        l_shmem_offset
    ));
    emit_smem_load(ptx, "f32", "%f_l_val", "%rd27"); // L[mi] (logsumexp)
    ptx.push('\n');

    // global_i = q_start + mi
    ptx.push_str("    add.u64 %rd31, %rd25, %rd11;  // global_i = q_start + tid\n\n");

    // Inner loop over nj (KV positions in tile)
    ptx.push_str("    mov.u64 %rd32, 0;  // nj = 0\n");
    ptx.push_str("BWD_MAIN_NJ_LOOP:\n");
    ptx.push_str(&format!(
        "    setp.ge.u64 %p1, %rd32, {};  // nj >= block_kv?\n",
        config.block_kv
    ));
    ptx.push_str("    @%p1 bra BWD_MAIN_NJ_DONE;\n\n");

    // global_j = kv_start + nj
    ptx.push_str("    add.u64 %rd33, %rd14, %rd32;  // global_j = kv_start + nj\n\n");

    // Step 3a: S[mi][nj] = sum_d(Q[mi,d] * K[nj,d]) * scale
    ptx.push_str("    // Step 3a: S[mi][nj] = Q[mi,:] . K[nj,:] * scale\n");
    ptx.push_str("    mov.f32 %f_sum, 0f00000000;\n");
    ptx.push_str("    mov.u64 %rd34, 0;  // d = 0\n");
    ptx.push_str("BWD_MAIN_S_DOT:\n");
    ptx.push_str("    setp.ge.u64 %p2, %rd34, %rd10;\n");
    ptx.push_str("    @%p2 bra BWD_MAIN_S_DOT_DONE;\n");
    // Q[mi, d] from shmem: q_shmem_offset + (mi * hd_padded + d) * 4
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd35, %rd11, {};  // mi * hd_padded\n",
        hd_padded
    ));
    ptx.push_str("    add.u64 %rd35, %rd35, %rd34;  // + d\n");
    ptx.push_str("    shl.b64 %rd35, %rd35, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd35, %rd35, {};\n",
        q_shmem_offset
    ));
    emit_smem_load(ptx, "f32", "%f_val", "%rd35");
    // K[nj, d] from shmem: (nj * hd_padded + d) * 4
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd36, %rd32, {};  // nj * hd_padded\n",
        hd_padded
    ));
    ptx.push_str("    add.u64 %rd36, %rd36, %rd34;  // + d\n");
    ptx.push_str("    shl.b64 %rd36, %rd36, 2;\n");
    emit_smem_load(ptx, "f32", "%f_tmp", "%rd36");
    ptx.push_str("    fma.rn.f32 %f_sum, %f_val, %f_tmp, %f_sum;\n");
    ptx.push_str("    add.u64 %rd34, %rd34, 1;\n");
    ptx.push_str("    bra BWD_MAIN_S_DOT;\n");
    ptx.push_str("BWD_MAIN_S_DOT_DONE:\n");
    ptx.push_str("    mul.f32 %f_s, %f_sum, %scale;  // S *= scale\n\n");

    // Step 3b: P = exp(S - L), with causal mask
    ptx.push_str("    // Step 3b: P = exp(S - L), causal mask\n");
    if config.causal {
        // If global_i < global_j then P = 0 (causal mask)
        ptx.push_str("    // Causal: P = 0 if global_i < global_j\n");
        ptx.push_str("    setp.lt.u64 %p3, %rd31, %rd33;  // global_i < global_j?\n");
        ptx.push_str("    @%p3 mov.f32 %f_s, 0fFF800000;  // set S = -inf so exp = 0\n");
    }
    ptx.push_str("    sub.f32 %f_tmp, %f_s, %f_l_val;  // S - L\n");
    ptx.push_str("    mul.f32 %f_tmp, %f_tmp, %log2e;   // * log2(e)\n");
    ptx.push_str("    ex2.approx.f32 %f_p, %f_tmp;      // P = exp(S - L)\n");
    if config.causal {
        // Ensure exactly zero for masked positions (exp(-inf) should be 0 but be safe)
        ptx.push_str("    @%p3 mov.f32 %f_p, 0f00000000;  // force P = 0 for masked\n");
    }
    ptx.push('\n');

    // Store S[mi][nj] = P (actually store P, the attention weight) into S_tile in shmem
    // for later use in steps 3c, 3e
    ptx.push_str("    // Store P[mi][nj] into S_tile in shmem\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd35, %rd11, {};  // mi * block_kv\n",
        config.block_kv
    ));
    ptx.push_str("    add.u64 %rd35, %rd35, %rd32;  // + nj\n");
    ptx.push_str("    shl.b64 %rd35, %rd35, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd35, %rd35, {};  // + S_tile shmem offset\n",
        s_shmem_offset
    ));
    emit_smem_store(ptx, "f32", "%rd35", "%f_p");


    // Step 3c: dV_local[nj][d] += P[mi][nj] * dO[mi][d]
    ptx.push_str("    // Step 3c: dV_local[nj][d] += P[mi][nj] * dO[mi][d]\n");
    ptx.push_str("    mov.u64 %rd34, 0;  // d = 0\n");
    ptx.push_str("BWD_MAIN_DV_ACCUM:\n");
    ptx.push_str("    setp.ge.u64 %p2, %rd34, %rd10;\n");
    ptx.push_str("    @%p2 bra BWD_MAIN_DV_ACCUM_DONE;\n");
    // dO[mi, d] from shmem
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd35, %rd11, {};  // mi * hd_padded\n",
        hd_padded
    ));
    ptx.push_str("    add.u64 %rd35, %rd35, %rd34;\n");
    ptx.push_str("    shl.b64 %rd35, %rd35, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd35, %rd35, {};\n",
        do_shmem_offset
    ));
    emit_smem_load(ptx, "f32", "%f_val", "%rd35"); // dO[mi][d]
    // dV_local[nj, d] shmem address
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd36, %rd32, {};  // nj * hd_padded\n",
        hd_padded
    ));
    ptx.push_str("    add.u64 %rd36, %rd36, %rd34;\n");
    ptx.push_str("    shl.b64 %rd36, %rd36, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd36, %rd36, {};\n",
        dv_shmem_offset
    ));
    // Atomic add to shared memory — multiple threads (different mi values)
    // accumulate to the same dV[nj][d] slot concurrently. Without atomics,
    // the read-modify-write would race and produce wrong gradients.
    // atom.shared.add.f32 requires sm_60+ (Pascal); our scalar path targets
    // sm_52 minimum but correctness on sm_52 would need a bar.sync restructure.
    // For the target hardware (RTX 5070 Ti, sm_120) this is optimal.
    ptx.push_str("    mul.f32 %f_tmp, %f_p, %f_val;     // P * dO[d]\n");
    ptx.push_str("    // dV[nj][d] += P * dO[d]\n");
    emit_smem_atom_add_f32(ptx, "%f_discard", "%rd36", "%f_tmp");
    ptx.push_str("    add.u64 %rd34, %rd34, 1;\n");
    ptx.push_str("    bra BWD_MAIN_DV_ACCUM;\n");
    ptx.push_str("BWD_MAIN_DV_ACCUM_DONE:\n\n");

    // Step 3d: dP[mi][nj] = sum_d(dO[mi,d] * V[nj,d])
    ptx.push_str("    // Step 3d: dP[mi][nj] = dO[mi,:] . V[nj,:]\n");
    ptx.push_str("    mov.f32 %f_sum, 0f00000000;\n");
    ptx.push_str("    mov.u64 %rd34, 0;\n");
    ptx.push_str("BWD_MAIN_DP_DOT:\n");
    ptx.push_str("    setp.ge.u64 %p2, %rd34, %rd10;\n");
    ptx.push_str("    @%p2 bra BWD_MAIN_DP_DOT_DONE;\n");
    // dO[mi, d]
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd35, %rd11, {};  // mi * hd_padded\n",
        hd_padded
    ));
    ptx.push_str("    add.u64 %rd35, %rd35, %rd34;\n");
    ptx.push_str("    shl.b64 %rd35, %rd35, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd35, %rd35, {};\n",
        do_shmem_offset
    ));
    emit_smem_load(ptx, "f32", "%f_val", "%rd35");
    // V[nj, d]
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd36, %rd32, {};  // nj * hd_padded\n",
        hd_padded
    ));
    ptx.push_str("    add.u64 %rd36, %rd36, %rd34;\n");
    ptx.push_str("    shl.b64 %rd36, %rd36, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd36, %rd36, {};  // V tile shmem offset\n",
        k_tile_bytes  // V comes right after K
    ));
    emit_smem_load(ptx, "f32", "%f_tmp", "%rd36");
    ptx.push_str("    fma.rn.f32 %f_sum, %f_val, %f_tmp, %f_sum;\n");
    ptx.push_str("    add.u64 %rd34, %rd34, 1;\n");
    ptx.push_str("    bra BWD_MAIN_DP_DOT;\n");
    ptx.push_str("BWD_MAIN_DP_DOT_DONE:\n");
    ptx.push_str("    mov.f32 %f_dp, %f_sum;  // dP[mi][nj]\n\n");

    // Step 3e: dS = P * (dP - D)
    ptx.push_str("    // Step 3e: dS[mi][nj] = P[mi][nj] * (dP[mi][nj] - D[mi])\n");
    ptx.push_str("    sub.f32 %f_tmp, %f_dp, %f_d_val;  // dP - D\n");
    ptx.push_str("    mul.f32 %f_ds, %f_p, %f_tmp;       // P * (dP - D)\n\n");

    // Step 3f: atomicAdd dQ[i_block][mi][d] += dS[mi][nj] * K[nj][d] * scale
    ptx.push_str("    // Step 3f: dQ[global_i][d] += dS * K[nj][d] * scale (atomicAdd)\n");
    // dq_base = dq_ptr + (bh_elem_offset + global_i * head_dim) * 4
    ptx.push_str("    mul.lo.u64 %rd35, %rd31, %rd10;  // global_i * head_dim\n");
    ptx.push_str("    add.u64 %rd35, %rd15, %rd35;\n");
    ptx.push_str("    shl.b64 %rd35, %rd35, 2;\n");
    ptx.push_str("    add.u64 %rd35, %rd4, %rd35;  // dq_global_base for this row\n");
    ptx.push_str("    mov.u64 %rd34, 0;  // d = 0\n");
    ptx.push_str("BWD_MAIN_DQ_ACCUM:\n");
    ptx.push_str("    setp.ge.u64 %p2, %rd34, %rd10;\n");
    ptx.push_str("    @%p2 bra BWD_MAIN_DQ_ACCUM_DONE;\n");
    // K[nj, d] from shmem
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd36, %rd32, {};  // nj * hd_padded\n",
        hd_padded
    ));
    ptx.push_str("    add.u64 %rd36, %rd36, %rd34;\n");
    ptx.push_str("    shl.b64 %rd36, %rd36, 2;\n");
    emit_smem_load(ptx, "f32", "%f_val", "%rd36"); // K[nj][d]
    // dQ_contrib = dS * K[d] * scale
    ptx.push_str("    mul.f32 %f_tmp, %f_ds, %f_val;   // dS * K[d]\n");
    ptx.push_str("    mul.f32 %f_tmp, %f_tmp, %scale;   // * scale\n");
    // atomicAdd to global dQ
    ptx.push_str("    shl.b64 %rd37, %rd34, 2;\n");
    ptx.push_str("    add.u64 %rd37, %rd35, %rd37;  // dQ_addr = dq_base + d*4\n");
    ptx.push_str("    atom.global.add.f32 %f_val, [%rd37], %f_tmp;\n");
    ptx.push_str("    add.u64 %rd34, %rd34, 1;\n");
    ptx.push_str("    bra BWD_MAIN_DQ_ACCUM;\n");
    ptx.push_str("BWD_MAIN_DQ_ACCUM_DONE:\n\n");

    // Step 3g: dK_local[nj][d] += dS[mi][nj] * Q[mi][d] * scale
    ptx.push_str("    // Step 3g: dK_local[nj][d] += dS * Q[mi][d] * scale\n");
    ptx.push_str("    mov.u64 %rd34, 0;  // d = 0\n");
    ptx.push_str("BWD_MAIN_DK_ACCUM:\n");
    ptx.push_str("    setp.ge.u64 %p2, %rd34, %rd10;\n");
    ptx.push_str("    @%p2 bra BWD_MAIN_DK_ACCUM_DONE;\n");
    // Q[mi, d] from shmem
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd36, %rd11, {};  // mi * hd_padded\n",
        hd_padded
    ));
    ptx.push_str("    add.u64 %rd36, %rd36, %rd34;\n");
    ptx.push_str("    shl.b64 %rd36, %rd36, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd36, %rd36, {};\n",
        q_shmem_offset
    ));
    emit_smem_load(ptx, "f32", "%f_val", "%rd36"); // Q[mi][d]
    // dK contribution
    ptx.push_str("    mul.f32 %f_tmp, %f_ds, %f_val;   // dS * Q[d]\n");
    ptx.push_str("    mul.f32 %f_tmp, %f_tmp, %scale;   // * scale\n");
    // dK_local[nj, d] shmem address
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd37, %rd32, {};  // nj * hd_padded\n",
        hd_padded
    ));
    ptx.push_str("    add.u64 %rd37, %rd37, %rd34;\n");
    ptx.push_str("    shl.b64 %rd37, %rd37, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd37, %rd37, {};\n",
        dk_shmem_offset
    ));
    // Atomic add to shared memory — same race condition as dV: multiple
    // threads (different mi) accumulate to the same dK[nj][d] slot.
    ptx.push_str("    // dK[nj][d] += dS * Q[d] * scale\n");
    emit_smem_atom_add_f32(ptx, "%f_discard", "%rd37", "%f_tmp");
    ptx.push_str("    add.u64 %rd34, %rd34, 1;\n");
    ptx.push_str("    bra BWD_MAIN_DK_ACCUM;\n");
    ptx.push_str("BWD_MAIN_DK_ACCUM_DONE:\n\n");

    // End of nj loop
    ptx.push_str("    add.u64 %rd32, %rd32, 1;  // nj++\n");
    ptx.push_str("    bra BWD_MAIN_NJ_LOOP;\n");
    ptx.push_str("BWD_MAIN_NJ_DONE:\n\n");

    ptx.push_str("BWD_MAIN_STEPS_DONE:\n");
    ptx.push_str("    bar.sync 0;  // all threads done with steps 3a-3g\n\n");

    // Advance i_block
    ptx.push_str("    add.u64 %rd24, %rd24, 1;  // i_block++\n");
    ptx.push_str("    bra BWD_MAIN_Q_LOOP;\n");
    ptx.push_str("BWD_MAIN_Q_LOOP_END:\n\n");
}

/// MMA register declarations for the backward kernel (sm_80+).
/// Declares accumulators and fragment registers for all 5 backward matmuls:
///   S = Q@K^T, dP = dO@V^T, dV += P^T@dO, dQ += dS@K, dK += dS^T@Q
fn emit_bwd_mma_registers(ptx: &mut String, config: &FlashAttentionBackwardConfig) {
    let n_tiles_s = config.block_kv as usize / MMA_N; // S/dP: [block_q, block_kv]
    let n_tiles_hd = config.head_dim as usize / MMA_N; // dV/dQ/dK: [*, head_dim]
    let k_iters_qk = config.head_dim as usize / MMA_K;

    ptx.push_str("    // === MMA registers for backward (all 5 matmuls) ===\n");

    // S accumulators: one m-tile at a time, n_tiles_s n-tiles
    for nt in 0..n_tiles_s {
        ptx.push_str(&format!(
            "    .reg .f32 %bwd_acc_s_{nt}_0, %bwd_acc_s_{nt}_1, %bwd_acc_s_{nt}_2, %bwd_acc_s_{nt}_3;\n"
        ));
    }

    // dP accumulators: same shape as S
    for nt in 0..n_tiles_s {
        ptx.push_str(&format!(
            "    .reg .f32 %bwd_acc_dp_{nt}_0, %bwd_acc_dp_{nt}_1, %bwd_acc_dp_{nt}_2, %bwd_acc_dp_{nt}_3;\n"
        ));
    }

    // Generic accumulators for dV/dQ/dK — reused sequentially (n_tiles_hd n-tiles)
    for nt in 0..n_tiles_hd {
        ptx.push_str(&format!(
            "    .reg .f32 %bwd_acc_g_{nt}_0, %bwd_acc_g_{nt}_1, %bwd_acc_g_{nt}_2, %bwd_acc_g_{nt}_3;\n"
        ));
    }

    // A-fragment registers for Q (4 .b32) — used in S = Q@K^T
    ptx.push_str("    .reg .b32 %bwd_aq_0, %bwd_aq_1, %bwd_aq_2, %bwd_aq_3;\n");

    // B-fragment registers for K^T (n_tiles_s * 2 .b32) — used in S = Q@K^T
    for nt in 0..n_tiles_s {
        ptx.push_str(&format!("    .reg .b32 %bwd_bk_{nt}_0, %bwd_bk_{nt}_1;\n"));
    }

    // A-fragment registers for dO (4 .b32) — used in dP = dO@V^T
    ptx.push_str("    .reg .b32 %bwd_ado_0, %bwd_ado_1, %bwd_ado_2, %bwd_ado_3;\n");

    // B-fragment registers for V^T (n_tiles_s * 2 .b32) — used in dP = dO@V^T
    for nt in 0..n_tiles_s {
        ptx.push_str(&format!("    .reg .b32 %bwd_bv_{nt}_0, %bwd_bv_{nt}_1;\n"));
    }

    // Generic A-fragment (4 .b32) — reused for transposed loads (P^T, dS^T) and dS
    ptx.push_str("    .reg .b32 %bwd_ag_0, %bwd_ag_1, %bwd_ag_2, %bwd_ag_3;\n");

    // Generic B-fragment for head_dim output tiles (n_tiles_hd * 2 .b32)
    for nt in 0..n_tiles_hd {
        ptx.push_str(&format!("    .reg .b32 %bwd_bg_{nt}_0, %bwd_bg_{nt}_1;\n"));
    }

    // MMA temporaries
    ptx.push_str("    .reg .f16 %bwd_mma_h0, %bwd_mma_h1;     // f32->f16 conversion temps\n");
    ptx.push_str("    .reg .u32 %bwd_mma_addr;                  // shared memory address temp\n");
    ptx.push_str("    .reg .u32 %bwd_mma_k_iter;                // K-dimension loop counter\n");
    ptx.push_str("    .reg .pred %bwd_mma_pk;                   // K-loop predicate\n");
    ptx.push_str("    .reg .u32 %bwd_mma_laneid;                // warp lane ID\n");
    ptx.push_str("    .reg .u32 %bwd_mma_a_row;                 // A-fragment row\n");
    ptx.push_str("    .reg .u32 %bwd_mma_b_row;                 // B-fragment row\n");
    ptx.push_str("    .reg .f32 %bwd_mma_f32_lo, %bwd_mma_f32_hi;  // shmem load temps\n");
    ptx.push_str("    .reg .u32 %bwd_mma_m_tile;                // m-tile loop counter\n");
    ptx.push_str("    .reg .u32 %bwd_mma_m_byte_off;            // m-tile byte offset\n");
    ptx.push_str("    .reg .u32 %bwd_mma_store_row;             // row for storing MMA results\n");
    ptx.push_str("    .reg .u32 %bwd_mma_store_col;             // col for storing MMA results\n");
    ptx.push_str("    .reg .u32 %bwd_mma_store_addr;            // shmem addr for MMA result store\n");
    ptx.push_str("    .reg .u64 %bwd_mma_store_addr_64;         // widened copy of %bwd_mma_store_addr for add.s64\n");
    // Additional temps for transposed loads and global atomics
    ptx.push_str("    .reg .u32 %bwd_mma_addr2;                 // second address temp\n");
    ptx.push_str("    .reg .u64 %bwd_mma_gaddr;                 // global address for atomicAdd\n");

    // Compute laneid and fragment row mappings
    ptx.push_str("    mov.u32 %bwd_mma_laneid, %tid.x;\n");
    ptx.push_str("    and.b32 %bwd_mma_laneid, %bwd_mma_laneid, 31;  // laneid = tid.x % 32\n");
    // A-fragment row: row = (laneid % 4) * 2 + (laneid / 16)
    ptx.push_str("    and.b32 %bwd_mma_a_row, %bwd_mma_laneid, 3;    // laneid % 4\n");
    ptx.push_str("    shl.b32 %bwd_mma_a_row, %bwd_mma_a_row, 1;     // * 2\n");
    ptx.push_str("    shr.u32 %bwd_mma_addr, %bwd_mma_laneid, 4;     // laneid / 16\n");
    ptx.push_str("    add.u32 %bwd_mma_a_row, %bwd_mma_a_row, %bwd_mma_addr;  // (laneid%4)*2 + laneid/16\n");
    // B-fragment row mapping matches A for the k-dimension
    ptx.push_str("    mov.u32 %bwd_mma_b_row, %bwd_mma_a_row;\n");
    // Store col = (laneid/4) % 4 — precomputed for MMA result store
    ptx.push_str("    shr.u32 %bwd_mma_store_col, %bwd_mma_laneid, 2;   // laneid / 4\n");
    ptx.push_str("    and.b32 %bwd_mma_store_col, %bwd_mma_store_col, 3; // (laneid/4) % 4 = col0\n");

    ptx.push_str(&format!(
        "    // n_tiles_s={}, n_tiles_hd={}, k_iters_qk={}\n\n",
        n_tiles_s, n_tiles_hd, k_iters_qk
    ));
}

/// Emit the inner Q-tile loop using MMA tensor cores for all matmuls (sm_80+).
///
/// All 5 matmuls use `mma.sync.aligned.m16n8k16`:
///   3a: S = Q@K^T * scale
///   3d: dP = dO@V^T
///   3c: dV += P^T@dO
///   3f: dQ += dS@K * scale (atomicAdd to global)
///   3g: dK += dS^T@Q * scale (atomicAdd to dK_local shmem)
///
/// Steps 3b (softmax) and 3e (dS = P*(dP-D)) remain scalar elementwise.
fn emit_bwd_main_q_tile_loop_mma(ptx: &mut String, config: &FlashAttentionBackwardConfig) {
    let hd_padded = config.head_dim + BWD_PAD;
    let k_tile_bytes = config.block_kv * hd_padded * 4;
    let v_tile_bytes = config.block_kv * hd_padded * 4;
    let q_shmem_offset = k_tile_bytes + v_tile_bytes;
    let do_shmem_offset = q_shmem_offset + config.block_q * hd_padded * 4;
    let dk_shmem_offset = do_shmem_offset + config.block_q * hd_padded * 4;
    let dv_shmem_offset = dk_shmem_offset + config.block_kv * hd_padded * 4;
    let s_shmem_offset = dv_shmem_offset + config.block_kv * hd_padded * 4;
    let dp_shmem_offset = s_shmem_offset + config.block_q * config.block_kv * 4;
    let d_shmem_offset = dp_shmem_offset + config.block_q * config.block_kv * 4;
    let l_shmem_offset = d_shmem_offset + config.block_q * 4;

    let block_q_u = config.block_q as usize;
    let block_kv_u = config.block_kv as usize;
    let head_dim_u = config.head_dim as usize;
    let n_tiles_s = block_kv_u / MMA_N;
    let n_tiles_hd = head_dim_u / MMA_N;
    let k_iters_hd = head_dim_u / MMA_K; // k-iters for S/dP (reduction over head_dim)
    let k_iters_q = block_q_u / MMA_K; // k-iters for dV/dK (reduction over block_q)
    let k_iters_kv = block_kv_u / MMA_K; // k-iters for dQ (reduction over block_kv)
    let m_tiles_q = block_q_u / MMA_M; // m-tiles for S/dP/dQ (rows = block_q)
    let m_tiles_kv = block_kv_u / MMA_M; // m-tiles for dV/dK (rows = block_kv)
    let hd_padded_u = hd_padded as usize;

    // Helper: V tile starts right after K tile
    let v_shmem_offset = k_tile_bytes;

    ptx.push_str("    // === Inner Q-tile loop (full MMA path for all matmuls) ===\n");

    // i_block loop: for causal, start at j_block; otherwise start at 0
    if config.causal {
        ptx.push_str("    mov.u64 %rd24, %rd13;  // i_block = j_block (causal)\n");
    } else {
        ptx.push_str("    mov.u64 %rd24, 0;  // i_block = 0 (non-causal)\n");
    }

    ptx.push_str("BWD_MAIN_Q_LOOP:\n");
    ptx.push_str("    setp.ge.u64 %p0, %rd24, %rd17;  // i_block >= num_q_tiles?\n");
    ptx.push_str("    @%p0 bra BWD_MAIN_Q_LOOP_END;\n\n");

    // q_start = i_block * block_q
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd25, %rd24, {};  // q_start = i_block * block_q\n",
        config.block_q
    ));

    // Load Q[i_block] into shmem[q_shmem_offset..]
    ptx.push_str(&format!(
        "    // Load Q[i_block] into shmem[{}..]\n",
        q_shmem_offset
    ));
    ptx.push_str("    mul.lo.u64 %rd26, %rd25, %rd10;  // q_start * head_dim\n");
    ptx.push_str("    add.u64 %rd26, %rd15, %rd26;\n");
    ptx.push_str("    shl.b64 %rd26, %rd26, 2;\n");
    ptx.push_str("    add.u64 %rd26, %rd1, %rd26;  // q_global_base\n");

    ptx.push_str("    mov.u64 %rd27, %rd11;  // mi = tid\n");
    ptx.push_str("BWD_MAIN_LOAD_Q:\n");
    ptx.push_str(&format!(
        "    setp.ge.u64 %p1, %rd27, {};\n",
        config.block_q
    ));
    ptx.push_str("    @%p1 bra BWD_MAIN_LOAD_Q_DONE;\n");
    ptx.push_str("    mov.u64 %rd28, 0;  // d = 0\n");
    ptx.push_str("BWD_MAIN_LOAD_Q_D:\n");
    ptx.push_str("    setp.ge.u64 %p2, %rd28, %rd10;\n");
    ptx.push_str("    @%p2 bra BWD_MAIN_LOAD_Q_D_DONE;\n");
    ptx.push_str("    mul.lo.u64 %rd29, %rd27, %rd10;\n");
    ptx.push_str("    add.u64 %rd29, %rd29, %rd28;\n");
    ptx.push_str("    shl.b64 %rd29, %rd29, 2;\n");
    ptx.push_str("    add.u64 %rd29, %rd26, %rd29;\n");
    ptx.push_str("    ld.global.f32 %f0, [%rd29];\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd30, %rd27, {};\n",
        hd_padded
    ));
    ptx.push_str("    add.u64 %rd30, %rd30, %rd28;\n");
    ptx.push_str("    shl.b64 %rd30, %rd30, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd30, %rd30, {};\n",
        q_shmem_offset
    ));
    emit_smem_store(ptx, "f32", "%rd30", "%f0");
    ptx.push_str("    add.u64 %rd28, %rd28, 1;\n");
    ptx.push_str("    bra BWD_MAIN_LOAD_Q_D;\n");
    ptx.push_str("BWD_MAIN_LOAD_Q_D_DONE:\n");
    ptx.push_str(&format!(
        "    add.u64 %rd27, %rd27, {};\n",
        config.block_q
    ));
    ptx.push_str("    bra BWD_MAIN_LOAD_Q;\n");
    ptx.push_str("BWD_MAIN_LOAD_Q_DONE:\n\n");

    // Load dO[i_block] into shmem[do_shmem_offset..]
    ptx.push_str(&format!(
        "    // Load dO[i_block] into shmem[{}..]\n",
        do_shmem_offset
    ));
    ptx.push_str("    mul.lo.u64 %rd26, %rd25, %rd10;\n");
    ptx.push_str("    add.u64 %rd26, %rd15, %rd26;\n");
    ptx.push_str("    shl.b64 %rd26, %rd26, 2;\n");
    ptx.push_str("    add.u64 %rd26, %rd0, %rd26;  // dO_global_base\n");

    ptx.push_str("    mov.u64 %rd27, %rd11;\n");
    ptx.push_str("BWD_MAIN_LOAD_DO:\n");
    ptx.push_str(&format!(
        "    setp.ge.u64 %p1, %rd27, {};\n",
        config.block_q
    ));
    ptx.push_str("    @%p1 bra BWD_MAIN_LOAD_DO_DONE;\n");
    ptx.push_str("    mov.u64 %rd28, 0;\n");
    ptx.push_str("BWD_MAIN_LOAD_DO_D:\n");
    ptx.push_str("    setp.ge.u64 %p2, %rd28, %rd10;\n");
    ptx.push_str("    @%p2 bra BWD_MAIN_LOAD_DO_D_DONE;\n");
    ptx.push_str("    mul.lo.u64 %rd29, %rd27, %rd10;\n");
    ptx.push_str("    add.u64 %rd29, %rd29, %rd28;\n");
    ptx.push_str("    shl.b64 %rd29, %rd29, 2;\n");
    ptx.push_str("    add.u64 %rd29, %rd26, %rd29;\n");
    ptx.push_str("    ld.global.f32 %f0, [%rd29];\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd30, %rd27, {};\n",
        hd_padded
    ));
    ptx.push_str("    add.u64 %rd30, %rd30, %rd28;\n");
    ptx.push_str("    shl.b64 %rd30, %rd30, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd30, %rd30, {};\n",
        do_shmem_offset
    ));
    emit_smem_store(ptx, "f32", "%rd30", "%f0");
    ptx.push_str("    add.u64 %rd28, %rd28, 1;\n");
    ptx.push_str("    bra BWD_MAIN_LOAD_DO_D;\n");
    ptx.push_str("BWD_MAIN_LOAD_DO_D_DONE:\n");
    ptx.push_str(&format!(
        "    add.u64 %rd27, %rd27, {};\n",
        config.block_q
    ));
    ptx.push_str("    bra BWD_MAIN_LOAD_DO;\n");
    ptx.push_str("BWD_MAIN_LOAD_DO_DONE:\n\n");

    // Load D[i_block] and L[i_block] vectors into shmem
    ptx.push_str("    // Load D[i_block] and L[i_block] into shmem\n");
    ptx.push_str(&format!(
        "    setp.ge.u64 %p1, %rd11, {};  // tid >= block_q?\n",
        config.block_q
    ));
    ptx.push_str("    @%p1 bra BWD_MAIN_LOAD_DL_DONE;\n");
    ptx.push_str("    add.u64 %rd26, %rd16, %rd25;  // bh_seq_offset + q_start\n");
    ptx.push_str("    add.u64 %rd26, %rd26, %rd11;  // + tid\n");
    ptx.push_str("    shl.b64 %rd26, %rd26, 2;\n");
    ptx.push_str("    add.u64 %rd27, %rd7, %rd26;   // d_ptr + offset\n");
    ptx.push_str("    ld.global.f32 %f0, [%rd27];\n");
    ptx.push_str("    shl.b64 %rd28, %rd11, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd28, %rd28, {};  // + D_vec shmem offset\n",
        d_shmem_offset
    ));
    emit_smem_store(ptx, "f32", "%rd28", "%f0");
    ptx.push_str("    add.u64 %rd27, %rd8, %rd26;   // lse_ptr + offset\n");
    ptx.push_str("    ld.global.f32 %f0, [%rd27];\n");
    ptx.push_str("    shl.b64 %rd28, %rd11, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd28, %rd28, {};  // + L_vec shmem offset\n",
        l_shmem_offset
    ));
    emit_smem_store(ptx, "f32", "%rd28", "%f0");
    ptx.push_str("BWD_MAIN_LOAD_DL_DONE:\n");
    ptx.push_str("    bar.sync 0;  // Q, dO, D, L loaded\n\n");

    // ══════════════════════════════════════════════════════════════════
    // MMA Step 3a: S = Q @ K^T * scale  →  S_tile shmem
    // Shape: [block_q, head_dim] @ [head_dim, block_kv] → [block_q, block_kv]
    // ══════════════════════════════════════════════════════════════════
    ptx.push_str("    // === MMA Step 3a: S = Q @ K^T * scale ===\n");
    ptx.push_str(&format!(
        "    // m_tiles={}, n_tiles_s={}, k_iters_hd={}\n",
        m_tiles_q, n_tiles_s, k_iters_hd
    ));

    ptx.push_str("    mov.u32 %bwd_mma_m_tile, 0;\n");
    ptx.push_str("    mov.u32 %bwd_mma_m_byte_off, 0;\n");
    ptx.push_str("BWD_MAIN_MMA_S_M_LOOP:\n");
    ptx.push_str(&format!(
        "    setp.ge.u32 %p0, %bwd_mma_m_tile, {};\n",
        m_tiles_q
    ));
    ptx.push_str("    @%p0 bra BWD_MAIN_MMA_S_M_DONE;\n\n");

    // Zero S accumulators
    for nt in 0..n_tiles_s {
        for r in 0..4 {
            ptx.push_str(&format!(
                "    mov.f32 %bwd_acc_s_{}_{}, {};\n",
                nt, r, f32_bits(F32_ZERO)
            ));
        }
    }

    // K-dimension loop (over head_dim)
    ptx.push_str("    mov.u32 %bwd_mma_k_iter, 0;\n");
    ptx.push_str("BWD_MAIN_MMA_S_K_LOOP:\n");

    // Load A-fragment from Q shmem
    emit_load_a_fragment_row_major(
        ptx,
        "%bwd_aq",
        q_shmem_offset as usize,
        hd_padded_u,
        "%bwd_mma_m_byte_off",
    );

    // Load B-fragments from K shmem for each n-tile (K stored as [block_kv, hd_padded])
    for nt in 0..n_tiles_s {
        emit_load_b_fragment_row_major(ptx, &format!("%bwd_bk_{nt}"), 0, hd_padded_u, nt);
    }

    // Issue MMA for each n-tile
    for nt in 0..n_tiles_s {
        ptx.push_str("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n");
        ptx.push_str(&format!(
            "        {{%bwd_acc_s_{nt}_0, %bwd_acc_s_{nt}_1, %bwd_acc_s_{nt}_2, %bwd_acc_s_{nt}_3}},\n"
        ));
        ptx.push_str("        {%bwd_aq_0, %bwd_aq_1, %bwd_aq_2, %bwd_aq_3},\n");
        ptx.push_str(&format!("        {{%bwd_bk_{nt}_0, %bwd_bk_{nt}_1}},\n"));
        ptx.push_str(&format!(
            "        {{%bwd_acc_s_{nt}_0, %bwd_acc_s_{nt}_1, %bwd_acc_s_{nt}_2, %bwd_acc_s_{nt}_3}};\n"
        ));
    }

    ptx.push_str("    add.u32 %bwd_mma_k_iter, %bwd_mma_k_iter, 1;\n");
    ptx.push_str(&format!(
        "    setp.lt.u32 %bwd_mma_pk, %bwd_mma_k_iter, {};\n",
        k_iters_hd
    ));
    ptx.push_str("    @%bwd_mma_pk bra BWD_MAIN_MMA_S_K_LOOP;\n\n");

    // Scale and store S to S_tile shmem
    emit_scale_and_store_mma_tile(
        ptx,
        "%bwd_acc_s",
        n_tiles_s,
        s_shmem_offset as usize,
        block_kv_u,
        true, // apply scale
    );

    // Advance m-tile
    ptx.push_str(&format!(
        "    add.u32 %bwd_mma_m_byte_off, %bwd_mma_m_byte_off, {};  // += MMA_M * hd_padded * 4\n",
        MMA_M * hd_padded_u * 4
    ));
    ptx.push_str("    add.u32 %bwd_mma_m_tile, %bwd_mma_m_tile, 1;\n");
    ptx.push_str("    bra BWD_MAIN_MMA_S_M_LOOP;\n");
    ptx.push_str("BWD_MAIN_MMA_S_M_DONE:\n");
    ptx.push_str("    bar.sync 0;  // S_tile fully computed via MMA\n\n");

    // ══════════════════════════════════════════════════════════════════
    // Scalar Step 3b: P = exp(S - L), with causal mask → overwrite S_tile with P
    // Each thread handles one row (mi = tid), iterates over columns
    // ══════════════════════════════════════════════════════════════════
    ptx.push_str("    // === Scalar Step 3b: P = exp(S - L) ===\n");
    ptx.push_str(&format!(
        "    setp.ge.u64 %p0, %rd11, {};  // skip if tid >= block_q\n",
        config.block_q
    ));
    ptx.push_str("    @%p0 bra BWD_MAIN_MMA_P_DONE;\n\n");

    // Load D[mi] and L[mi]
    ptx.push_str("    shl.b64 %rd26, %rd11, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd27, %rd26, {};\n",
        d_shmem_offset
    ));
    emit_smem_load(ptx, "f32", "%f_d_val", "%rd27"); // D[mi]
    ptx.push_str(&format!(
        "    add.u64 %rd27, %rd26, {};\n",
        l_shmem_offset
    ));
    emit_smem_load(ptx, "f32", "%f_l_val", "%rd27"); // L[mi]
    ptx.push('\n');

    // global_i = q_start + mi
    ptx.push_str("    add.u64 %rd31, %rd25, %rd11;  // global_i = q_start + tid\n\n");

    ptx.push_str("    mov.u64 %rd32, 0;  // nj = 0\n");
    ptx.push_str("BWD_MAIN_MMA_P_NJ:\n");
    ptx.push_str(&format!(
        "    setp.ge.u64 %p1, %rd32, {};  // nj >= block_kv?\n",
        config.block_kv
    ));
    ptx.push_str("    @%p1 bra BWD_MAIN_MMA_P_NJ_DONE;\n\n");

    // Read S[mi][nj]
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd35, %rd11, {};  // mi * block_kv\n",
        config.block_kv
    ));
    ptx.push_str("    add.u64 %rd35, %rd35, %rd32;\n");
    ptx.push_str("    shl.b64 %rd35, %rd35, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd35, %rd35, {};\n",
        s_shmem_offset
    ));
    emit_smem_load(ptx, "f32", "%f_s", "%rd35");

    // Causal mask
    if config.causal {
        ptx.push_str("    add.u64 %rd33, %rd14, %rd32;  // global_j = kv_start + nj\n");
        ptx.push_str("    setp.lt.u64 %p3, %rd31, %rd33;  // global_i < global_j?\n");
        ptx.push_str("    @%p3 mov.f32 %f_s, 0fFF800000;  // -inf\n");
    }

    // P = exp(S - L)
    ptx.push_str("    sub.f32 %f_tmp, %f_s, %f_l_val;\n");
    ptx.push_str("    mul.f32 %f_tmp, %f_tmp, %log2e;\n");
    ptx.push_str("    ex2.approx.f32 %f_p, %f_tmp;\n");
    if config.causal {
        ptx.push_str("    @%p3 mov.f32 %f_p, 0f00000000;  // force P = 0 for masked\n");
    }

    // Overwrite S_tile with P
    emit_smem_store(ptx, "f32", "%rd35", "%f_p");
    ptx.push_str("    add.u64 %rd32, %rd32, 1;\n");
    ptx.push_str("    bra BWD_MAIN_MMA_P_NJ;\n");
    ptx.push_str("BWD_MAIN_MMA_P_NJ_DONE:\n\n");

    ptx.push_str("BWD_MAIN_MMA_P_DONE:\n");
    ptx.push_str("    bar.sync 0;  // P stored in S_tile\n\n");

    // ══════════════════════════════════════════════════════════════════
    // MMA Step 3d: dP = dO @ V^T → dP_tile shmem
    // Shape: [block_q, head_dim] @ [head_dim, block_kv] → [block_q, block_kv]
    // Same shape as S; A = dO, B = V (transposed via col-major B-fragment read)
    // ══════════════════════════════════════════════════════════════════
    ptx.push_str("    // === MMA Step 3d: dP = dO @ V^T ===\n");

    ptx.push_str("    mov.u32 %bwd_mma_m_tile, 0;\n");
    ptx.push_str("    mov.u32 %bwd_mma_m_byte_off, 0;\n");
    ptx.push_str("BWD_MAIN_MMA_DP_M_LOOP:\n");
    ptx.push_str(&format!(
        "    setp.ge.u32 %p0, %bwd_mma_m_tile, {};\n",
        m_tiles_q
    ));
    ptx.push_str("    @%p0 bra BWD_MAIN_MMA_DP_M_DONE;\n\n");

    // Zero dP accumulators
    for nt in 0..n_tiles_s {
        for r in 0..4 {
            ptx.push_str(&format!(
                "    mov.f32 %bwd_acc_dp_{}_{}, {};\n",
                nt, r, f32_bits(F32_ZERO)
            ));
        }
    }

    // K-dimension loop (over head_dim)
    ptx.push_str("    mov.u32 %bwd_mma_k_iter, 0;\n");
    ptx.push_str("BWD_MAIN_MMA_DP_K_LOOP:\n");

    // A-fragment from dO shmem
    emit_load_a_fragment_row_major(
        ptx,
        "%bwd_ado",
        do_shmem_offset as usize,
        hd_padded_u,
        "%bwd_mma_m_byte_off",
    );

    // B-fragments from V shmem (V stored as [block_kv, hd_padded], same layout as K)
    for nt in 0..n_tiles_s {
        emit_load_b_fragment_row_major(
            ptx,
            &format!("%bwd_bv_{nt}"),
            v_shmem_offset as usize,
            hd_padded_u,
            nt,
        );
    }

    // Issue MMA: dP_acc += dO_frag @ V^T_frag
    for nt in 0..n_tiles_s {
        ptx.push_str("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n");
        ptx.push_str(&format!(
            "        {{%bwd_acc_dp_{nt}_0, %bwd_acc_dp_{nt}_1, %bwd_acc_dp_{nt}_2, %bwd_acc_dp_{nt}_3}},\n"
        ));
        ptx.push_str("        {%bwd_ado_0, %bwd_ado_1, %bwd_ado_2, %bwd_ado_3},\n");
        ptx.push_str(&format!("        {{%bwd_bv_{nt}_0, %bwd_bv_{nt}_1}},\n"));
        ptx.push_str(&format!(
            "        {{%bwd_acc_dp_{nt}_0, %bwd_acc_dp_{nt}_1, %bwd_acc_dp_{nt}_2, %bwd_acc_dp_{nt}_3}};\n"
        ));
    }

    ptx.push_str("    add.u32 %bwd_mma_k_iter, %bwd_mma_k_iter, 1;\n");
    ptx.push_str(&format!(
        "    setp.lt.u32 %bwd_mma_pk, %bwd_mma_k_iter, {};\n",
        k_iters_hd
    ));
    ptx.push_str("    @%bwd_mma_pk bra BWD_MAIN_MMA_DP_K_LOOP;\n\n");

    // Store dP to dP_tile shmem (no scaling needed)
    emit_scale_and_store_mma_tile(
        ptx,
        "%bwd_acc_dp",
        n_tiles_s,
        dp_shmem_offset as usize,
        block_kv_u,
        false, // no scale
    );

    // Advance m-tile
    ptx.push_str(&format!(
        "    add.u32 %bwd_mma_m_byte_off, %bwd_mma_m_byte_off, {};  // += MMA_M * hd_padded * 4\n",
        MMA_M * hd_padded_u * 4
    ));
    ptx.push_str("    add.u32 %bwd_mma_m_tile, %bwd_mma_m_tile, 1;\n");
    ptx.push_str("    bra BWD_MAIN_MMA_DP_M_LOOP;\n");
    ptx.push_str("BWD_MAIN_MMA_DP_M_DONE:\n");
    ptx.push_str("    bar.sync 0;  // dP_tile fully computed via MMA\n\n");

    // ══════════════════════════════════════════════════════════════════
    // MMA Step 3c: dV += P^T @ dO → accumulate to dV_local shmem
    // Shape: [block_kv, block_q] @ [block_q, head_dim] → [block_kv, head_dim]
    // A = P^T (transposed read from S_tile[block_q, block_kv])
    // B = dO (row-major [block_q, hd_padded])
    // ══════════════════════════════════════════════════════════════════
    ptx.push_str("    // === MMA Step 3c: dV += P^T @ dO ===\n");
    ptx.push_str(&format!(
        "    // m_tiles_kv={}, n_tiles_hd={}, k_iters_q={}\n",
        m_tiles_kv, n_tiles_hd, k_iters_q
    ));

    ptx.push_str("    mov.u32 %bwd_mma_m_tile, 0;\n");
    ptx.push_str("    mov.u32 %bwd_mma_m_byte_off, 0;\n");
    ptx.push_str("BWD_MAIN_MMA_DV_M_LOOP:\n");
    ptx.push_str(&format!(
        "    setp.ge.u32 %p0, %bwd_mma_m_tile, {};\n",
        m_tiles_kv
    ));
    ptx.push_str("    @%p0 bra BWD_MAIN_MMA_DV_M_DONE;\n\n");

    // Zero accumulators
    for nt in 0..n_tiles_hd {
        for r in 0..4 {
            ptx.push_str(&format!(
                "    mov.f32 %bwd_acc_g_{}_{}, {};\n",
                nt, r, f32_bits(F32_ZERO)
            ));
        }
    }

    // K-dimension loop (over block_q, reduction dimension)
    ptx.push_str("    mov.u32 %bwd_mma_k_iter, 0;\n");
    ptx.push_str("BWD_MAIN_MMA_DV_K_LOOP:\n");

    // A-fragment from P^T: load transposed from S_tile[block_q, block_kv]
    // P^T[kv_row, q_col] = S_tile[q_col * block_kv + kv_row]
    // m_tile indexes rows of P^T = rows of KV dimension
    // k_iter indexes columns of P^T = rows of Q dimension
    emit_load_a_fragment_transposed(
        ptx,
        "%bwd_ag",
        s_shmem_offset as usize,
        block_kv_u, // col stride of source matrix (S_tile row width)
    );

    // B-fragments from dO shmem [block_q, hd_padded]
    // k_iter indexes rows of dO (Q dimension)
    for nt in 0..n_tiles_hd {
        emit_load_b_fragment_row_major_k_is_q(
            ptx,
            &format!("%bwd_bg_{nt}"),
            do_shmem_offset as usize,
            hd_padded_u,
            nt,
        );
    }

    // Issue MMA: dV_acc += P^T_frag @ dO_frag
    for nt in 0..n_tiles_hd {
        ptx.push_str("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n");
        ptx.push_str(&format!(
            "        {{%bwd_acc_g_{nt}_0, %bwd_acc_g_{nt}_1, %bwd_acc_g_{nt}_2, %bwd_acc_g_{nt}_3}},\n"
        ));
        ptx.push_str("        {%bwd_ag_0, %bwd_ag_1, %bwd_ag_2, %bwd_ag_3},\n");
        ptx.push_str(&format!("        {{%bwd_bg_{nt}_0, %bwd_bg_{nt}_1}},\n"));
        ptx.push_str(&format!(
            "        {{%bwd_acc_g_{nt}_0, %bwd_acc_g_{nt}_1, %bwd_acc_g_{nt}_2, %bwd_acc_g_{nt}_3}};\n"
        ));
    }

    ptx.push_str("    add.u32 %bwd_mma_k_iter, %bwd_mma_k_iter, 1;\n");
    ptx.push_str(&format!(
        "    setp.lt.u32 %bwd_mma_pk, %bwd_mma_k_iter, {};\n",
        k_iters_q
    ));
    ptx.push_str("    @%bwd_mma_pk bra BWD_MAIN_MMA_DV_K_LOOP;\n\n");

    // Accumulate dV results to dV_local shmem via atom.shared.add.f32
    emit_accumulate_mma_to_shmem(
        ptx,
        "%bwd_acc_g",
        n_tiles_hd,
        dv_shmem_offset as usize,
        hd_padded_u,
    );

    // Advance m-tile (row stride for dV_local is hd_padded)
    ptx.push_str(&format!(
        "    add.u32 %bwd_mma_m_byte_off, %bwd_mma_m_byte_off, {};  // += MMA_M * hd_padded * 4\n",
        MMA_M * hd_padded_u * 4
    ));
    ptx.push_str("    add.u32 %bwd_mma_m_tile, %bwd_mma_m_tile, 1;\n");
    ptx.push_str("    bra BWD_MAIN_MMA_DV_M_LOOP;\n");
    ptx.push_str("BWD_MAIN_MMA_DV_M_DONE:\n");
    ptx.push_str("    bar.sync 0;  // dV accumulation done\n\n");

    // ══════════════════════════════════════════════════════════════════
    // Scalar Step 3e: dS = P * (dP - D) → overwrite S_tile with dS
    // Now P is in S_tile and dP is in dP_tile
    // ══════════════════════════════════════════════════════════════════
    ptx.push_str("    // === Scalar Step 3e: dS = P * (dP - D) ===\n");
    ptx.push_str(&format!(
        "    setp.ge.u64 %p0, %rd11, {};  // skip if tid >= block_q\n",
        config.block_q
    ));
    ptx.push_str("    @%p0 bra BWD_MAIN_MMA_DS_DONE;\n\n");

    // Reload D[mi]
    ptx.push_str("    shl.b64 %rd26, %rd11, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd27, %rd26, {};\n",
        d_shmem_offset
    ));
    emit_smem_load(ptx, "f32", "%f_d_val", "%rd27");
    ptx.push('\n');

    ptx.push_str("    mov.u64 %rd32, 0;  // nj = 0\n");
    ptx.push_str("BWD_MAIN_MMA_DS_NJ:\n");
    ptx.push_str(&format!(
        "    setp.ge.u64 %p1, %rd32, {};  // nj >= block_kv?\n",
        config.block_kv
    ));
    ptx.push_str("    @%p1 bra BWD_MAIN_MMA_DS_NJ_DONE;\n\n");

    // S_tile addr for [mi][nj]
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd35, %rd11, {};  // mi * block_kv\n",
        config.block_kv
    ));
    ptx.push_str("    add.u64 %rd35, %rd35, %rd32;\n");
    ptx.push_str("    shl.b64 %rd35, %rd35, 2;\n");

    // Read P[mi][nj] from S_tile
    ptx.push_str(&format!(
        "    add.u64 %rd36, %rd35, {};  // S_tile offset\n",
        s_shmem_offset
    ));
    emit_smem_load(ptx, "f32", "%f_p", "%rd36");

    // Read dP[mi][nj] from dP_tile
    ptx.push_str(&format!(
        "    add.u64 %rd37, %rd35, {};  // dP_tile offset\n",
        dp_shmem_offset
    ));
    emit_smem_load(ptx, "f32", "%f_dp", "%rd37");

    // dS = P * (dP - D)
    ptx.push_str("    sub.f32 %f_tmp, %f_dp, %f_d_val;\n");
    ptx.push_str("    mul.f32 %f_ds, %f_p, %f_tmp;\n");

    // Overwrite S_tile with dS (reuse same location)
    emit_smem_store(ptx, "f32", "%rd36", "%f_ds");
    ptx.push_str("    add.u64 %rd32, %rd32, 1;\n");
    ptx.push_str("    bra BWD_MAIN_MMA_DS_NJ;\n");
    ptx.push_str("BWD_MAIN_MMA_DS_NJ_DONE:\n\n");

    ptx.push_str("BWD_MAIN_MMA_DS_DONE:\n");
    ptx.push_str("    bar.sync 0;  // dS stored in S_tile\n\n");

    // ══════════════════════════════════════════════════════════════════
    // MMA Step 3f: dQ += dS @ K * scale → atomicAdd to global dQ
    // Shape: [block_q, block_kv] @ [block_kv, head_dim] → [block_q, head_dim]
    // A = dS (row-major from S_tile[block_q, block_kv])
    // B = K (row-major [block_kv, hd_padded])
    // ══════════════════════════════════════════════════════════════════
    ptx.push_str("    // === MMA Step 3f: dQ += dS @ K * scale ===\n");
    ptx.push_str(&format!(
        "    // m_tiles_q={}, n_tiles_hd={}, k_iters_kv={}\n",
        m_tiles_q, n_tiles_hd, k_iters_kv
    ));

    ptx.push_str("    mov.u32 %bwd_mma_m_tile, 0;\n");
    ptx.push_str("    mov.u32 %bwd_mma_m_byte_off, 0;\n");
    ptx.push_str("BWD_MAIN_MMA_DQ_M_LOOP:\n");
    ptx.push_str(&format!(
        "    setp.ge.u32 %p0, %bwd_mma_m_tile, {};\n",
        m_tiles_q
    ));
    ptx.push_str("    @%p0 bra BWD_MAIN_MMA_DQ_M_DONE;\n\n");

    // Zero accumulators
    for nt in 0..n_tiles_hd {
        for r in 0..4 {
            ptx.push_str(&format!(
                "    mov.f32 %bwd_acc_g_{}_{}, {};\n",
                nt, r, f32_bits(F32_ZERO)
            ));
        }
    }

    // K-dimension loop (over block_kv, reduction)
    ptx.push_str("    mov.u32 %bwd_mma_k_iter, 0;\n");
    ptx.push_str("BWD_MAIN_MMA_DQ_K_LOOP:\n");

    // A-fragment from dS (stored in S_tile [block_q, block_kv], row-major)
    // row stride = block_kv (not hd_padded!)
    emit_load_a_fragment_s_tile(
        ptx,
        "%bwd_ag",
        s_shmem_offset as usize,
        block_kv_u,
    );

    // B-fragments from K shmem [block_kv, hd_padded]
    // k_iter indexes rows of K (KV dimension = reduction)
    for nt in 0..n_tiles_hd {
        emit_load_b_fragment_row_major_k_is_q(
            ptx,
            &format!("%bwd_bg_{nt}"),
            0, // K at shmem[0..]
            hd_padded_u,
            nt,
        );
    }

    // Issue MMA: dQ_acc += dS_frag @ K_frag
    for nt in 0..n_tiles_hd {
        ptx.push_str("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n");
        ptx.push_str(&format!(
            "        {{%bwd_acc_g_{nt}_0, %bwd_acc_g_{nt}_1, %bwd_acc_g_{nt}_2, %bwd_acc_g_{nt}_3}},\n"
        ));
        ptx.push_str("        {%bwd_ag_0, %bwd_ag_1, %bwd_ag_2, %bwd_ag_3},\n");
        ptx.push_str(&format!("        {{%bwd_bg_{nt}_0, %bwd_bg_{nt}_1}},\n"));
        ptx.push_str(&format!(
            "        {{%bwd_acc_g_{nt}_0, %bwd_acc_g_{nt}_1, %bwd_acc_g_{nt}_2, %bwd_acc_g_{nt}_3}};\n"
        ));
    }

    ptx.push_str("    add.u32 %bwd_mma_k_iter, %bwd_mma_k_iter, 1;\n");
    ptx.push_str(&format!(
        "    setp.lt.u32 %bwd_mma_pk, %bwd_mma_k_iter, {};\n",
        k_iters_kv
    ));
    ptx.push_str("    @%bwd_mma_pk bra BWD_MAIN_MMA_DQ_K_LOOP;\n\n");

    // Scale and atomicAdd dQ results to global dQ
    // dQ[global_q_row, d] += acc * scale
    // global_q_row = q_start + m_tile*16 + thread_row
    emit_atomicadd_mma_to_global_dq(
        ptx,
        "%bwd_acc_g",
        n_tiles_hd,
        config,
    );

    // Advance m-tile (row stride for S_tile A-fragment is block_kv, but m_byte_off
    // tracks the Q/dO m-tile offset which uses hd_padded stride — we need a separate
    // offset for S_tile. Use m_tile counter directly in the helper.)
    ptx.push_str(&format!(
        "    add.u32 %bwd_mma_m_byte_off, %bwd_mma_m_byte_off, {};  // += MMA_M * block_kv * 4 (S_tile row stride)\n",
        MMA_M * block_kv_u * 4
    ));
    ptx.push_str("    add.u32 %bwd_mma_m_tile, %bwd_mma_m_tile, 1;\n");
    ptx.push_str("    bra BWD_MAIN_MMA_DQ_M_LOOP;\n");
    ptx.push_str("BWD_MAIN_MMA_DQ_M_DONE:\n");
    ptx.push_str("    bar.sync 0;  // dQ accumulation done\n\n");

    // ══════════════════════════════════════════════════════════════════
    // MMA Step 3g: dK += dS^T @ Q * scale → accumulate to dK_local shmem
    // Shape: [block_kv, block_q] @ [block_q, head_dim] → [block_kv, head_dim]
    // A = dS^T (transposed read from S_tile[block_q, block_kv])
    // B = Q (row-major [block_q, hd_padded])
    // ══════════════════════════════════════════════════════════════════
    ptx.push_str("    // === MMA Step 3g: dK += dS^T @ Q * scale ===\n");
    ptx.push_str(&format!(
        "    // m_tiles_kv={}, n_tiles_hd={}, k_iters_q={}\n",
        m_tiles_kv, n_tiles_hd, k_iters_q
    ));

    ptx.push_str("    mov.u32 %bwd_mma_m_tile, 0;\n");
    ptx.push_str("    mov.u32 %bwd_mma_m_byte_off, 0;\n");
    ptx.push_str("BWD_MAIN_MMA_DK_M_LOOP:\n");
    ptx.push_str(&format!(
        "    setp.ge.u32 %p0, %bwd_mma_m_tile, {};\n",
        m_tiles_kv
    ));
    ptx.push_str("    @%p0 bra BWD_MAIN_MMA_DK_M_DONE;\n\n");

    // Zero accumulators
    for nt in 0..n_tiles_hd {
        for r in 0..4 {
            ptx.push_str(&format!(
                "    mov.f32 %bwd_acc_g_{}_{}, {};\n",
                nt, r, f32_bits(F32_ZERO)
            ));
        }
    }

    // K-dimension loop (over block_q, reduction)
    ptx.push_str("    mov.u32 %bwd_mma_k_iter, 0;\n");
    ptx.push_str("BWD_MAIN_MMA_DK_K_LOOP:\n");

    // A-fragment from dS^T: transposed from S_tile[block_q, block_kv]
    // Same transpose pattern as P^T in step 3c
    emit_load_a_fragment_transposed(
        ptx,
        "%bwd_ag",
        s_shmem_offset as usize,
        block_kv_u,
    );

    // B-fragments from Q shmem [block_q, hd_padded]
    for nt in 0..n_tiles_hd {
        emit_load_b_fragment_row_major_k_is_q(
            ptx,
            &format!("%bwd_bg_{nt}"),
            q_shmem_offset as usize,
            hd_padded_u,
            nt,
        );
    }

    // Issue MMA: dK_acc += dS^T_frag @ Q_frag
    for nt in 0..n_tiles_hd {
        ptx.push_str("    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32\n");
        ptx.push_str(&format!(
            "        {{%bwd_acc_g_{nt}_0, %bwd_acc_g_{nt}_1, %bwd_acc_g_{nt}_2, %bwd_acc_g_{nt}_3}},\n"
        ));
        ptx.push_str("        {%bwd_ag_0, %bwd_ag_1, %bwd_ag_2, %bwd_ag_3},\n");
        ptx.push_str(&format!("        {{%bwd_bg_{nt}_0, %bwd_bg_{nt}_1}},\n"));
        ptx.push_str(&format!(
            "        {{%bwd_acc_g_{nt}_0, %bwd_acc_g_{nt}_1, %bwd_acc_g_{nt}_2, %bwd_acc_g_{nt}_3}};\n"
        ));
    }

    ptx.push_str("    add.u32 %bwd_mma_k_iter, %bwd_mma_k_iter, 1;\n");
    ptx.push_str(&format!(
        "    setp.lt.u32 %bwd_mma_pk, %bwd_mma_k_iter, {};\n",
        k_iters_q
    ));
    ptx.push_str("    @%bwd_mma_pk bra BWD_MAIN_MMA_DK_K_LOOP;\n\n");

    // Scale and accumulate dK results to dK_local shmem via atom.shared.add.f32
    emit_accumulate_mma_to_shmem_scaled(
        ptx,
        "%bwd_acc_g",
        n_tiles_hd,
        dk_shmem_offset as usize,
        hd_padded_u,
    );

    // Advance m-tile
    ptx.push_str(&format!(
        "    add.u32 %bwd_mma_m_byte_off, %bwd_mma_m_byte_off, {};  // += MMA_M * hd_padded * 4\n",
        MMA_M * hd_padded_u * 4
    ));
    ptx.push_str("    add.u32 %bwd_mma_m_tile, %bwd_mma_m_tile, 1;\n");
    ptx.push_str("    bra BWD_MAIN_MMA_DK_M_LOOP;\n");
    ptx.push_str("BWD_MAIN_MMA_DK_M_DONE:\n");
    ptx.push_str("BWD_MAIN_STEPS_DONE:\n");
    ptx.push_str("    bar.sync 0;  // all steps 3a-3g done (MMA path)\n\n");

    // Advance i_block
    ptx.push_str("    add.u64 %rd24, %rd24, 1;  // i_block++\n");
    ptx.push_str("    bra BWD_MAIN_Q_LOOP;\n");
    ptx.push_str("BWD_MAIN_Q_LOOP_END:\n\n");
}

// ────────────────────────────────────────────────────────────────────────
// MMA fragment load / store helpers for backward kernel
// ────────────────────────────────────────────────────────────────────────

/// Load A-fragment from a row-major tile in shmem.
/// The tile is stored as f32 and converted to f16 on load.
/// `frag_prefix`: e.g. "%bwd_aq" → registers %bwd_aq_0..3
/// `shmem_base`: byte offset of the tile in shmem
/// `row_stride_elems`: row stride in elements (e.g. hd_padded)
/// `m_byte_off_reg`: register holding m_tile * MMA_M * row_stride * 4
fn emit_load_a_fragment_row_major(
    ptx: &mut String,
    frag_prefix: &str,
    shmem_base: usize,
    row_stride_elems: usize,
    m_byte_off_reg: &str,
) {
    ptx.push_str(&format!(
        "    // Load A-fragment (row-major) from shmem[{}..]\n",
        shmem_base
    ));
    for i in 0..4 {
        let k_pair = i * 4; // byte offset within the k-dimension: i*2 elements * 4 bytes (f32)
        ptx.push_str(&format!(
            "    mul.lo.u32 %bwd_mma_addr, %bwd_mma_a_row, {};  // a_row * row_stride * 4\n",
            row_stride_elems * 4
        ));
        ptx.push_str(&format!(
            "    mad.lo.u32 %bwd_mma_addr, %bwd_mma_k_iter, {}, %bwd_mma_addr;  // + k_iter * MMA_K * 4\n",
            MMA_K * 4
        ));
        ptx.push_str(&format!(
            "    add.u32 %bwd_mma_addr, %bwd_mma_addr, {};  // + k_pair * 4\n",
            k_pair * 4
        ));
        ptx.push_str(&format!(
            "    add.u32 %bwd_mma_addr, %bwd_mma_addr, {};  // + m_tile byte offset\n",
            m_byte_off_reg
        ));
        ptx.push_str(&format!(
            "    add.u32 %bwd_mma_addr, %bwd_mma_addr, {};  // + shmem base\n",
            shmem_base
        ));
        emit_smem_load(ptx, "f32", "%bwd_mma_f32_lo", "%bwd_mma_addr");
        ptx.push_str("    add.u32 %bwd_mma_addr, %bwd_mma_addr, 4;\n");
        emit_smem_load(ptx, "f32", "%bwd_mma_f32_hi", "%bwd_mma_addr");
        ptx.push_str("    cvt.rn.f16.f32 %bwd_mma_h0, %bwd_mma_f32_lo;\n");
        ptx.push_str("    cvt.rn.f16.f32 %bwd_mma_h1, %bwd_mma_f32_hi;\n");
        ptx.push_str(&format!(
            "    mov.b32 {frag_prefix}_{i}, {{%bwd_mma_h0, %bwd_mma_h1}};\n"
        ));
    }
}

/// Load A-fragment from S_tile (row-major [block_q, block_kv]) where the row stride
/// is block_kv (not hd_padded). Used for dS in step 3f.
fn emit_load_a_fragment_s_tile(
    ptx: &mut String,
    frag_prefix: &str,
    shmem_base: usize,
    row_stride_elems: usize, // block_kv
) {
    ptx.push_str(&format!(
        "    // Load A-fragment from S_tile (row stride={}) at shmem[{}..]\n",
        row_stride_elems, shmem_base
    ));
    for i in 0..4 {
        let k_pair = i * 4;
        // row address: a_row * row_stride * 4 + m_byte_off (which is m_tile * MMA_M * row_stride * 4)
        ptx.push_str(&format!(
            "    mul.lo.u32 %bwd_mma_addr, %bwd_mma_a_row, {};  // a_row * row_stride * 4\n",
            row_stride_elems * 4
        ));
        ptx.push_str(&format!(
            "    mad.lo.u32 %bwd_mma_addr, %bwd_mma_k_iter, {}, %bwd_mma_addr;  // + k_iter * MMA_K * 4\n",
            MMA_K * 4
        ));
        ptx.push_str(&format!(
            "    add.u32 %bwd_mma_addr, %bwd_mma_addr, {};  // + k_pair * 4\n",
            k_pair * 4
        ));
        ptx.push_str(&format!(
            "    add.u32 %bwd_mma_addr, %bwd_mma_addr, {};  // + m_tile byte offset\n",
            "%bwd_mma_m_byte_off"
        ));
        ptx.push_str(&format!(
            "    add.u32 %bwd_mma_addr, %bwd_mma_addr, {};  // + shmem base\n",
            shmem_base
        ));
        emit_smem_load(ptx, "f32", "%bwd_mma_f32_lo", "%bwd_mma_addr");
        ptx.push_str("    add.u32 %bwd_mma_addr, %bwd_mma_addr, 4;\n");
        emit_smem_load(ptx, "f32", "%bwd_mma_f32_hi", "%bwd_mma_addr");
        ptx.push_str("    cvt.rn.f16.f32 %bwd_mma_h0, %bwd_mma_f32_lo;\n");
        ptx.push_str("    cvt.rn.f16.f32 %bwd_mma_h1, %bwd_mma_f32_hi;\n");
        ptx.push_str(&format!(
            "    mov.b32 {frag_prefix}_{i}, {{%bwd_mma_h0, %bwd_mma_h1}};\n"
        ));
    }
}

/// Load A-fragment from a TRANSPOSED matrix stored in S_tile shmem.
///
/// Source matrix: S_tile[block_q, block_kv] stored row-major.
/// We want A^T[kv_row, q_col] = S_tile[q_col, kv_row].
///
/// m_tile indexes rows of A^T (= columns of S_tile = KV dimension)
/// k_iter indexes cols of A^T (= rows of S_tile = Q dimension)
///
/// For A-fragment, thread needs:
///   row_in_tile = a_row (from laneid mapping)
///   actual_m_row = m_tile*16 + a_row
///   For register i, k values are at k_iter*16 + k_pair (k_pair = i*4, i*4+1)
///
/// S_tile address for A^T[actual_m_row, k_val]:
///   = S_tile[k_val][actual_m_row]
///   = shmem_base + (k_val * col_stride + actual_m_row) * 4
///
/// `col_stride`: block_kv (S_tile row width)
fn emit_load_a_fragment_transposed(
    ptx: &mut String,
    frag_prefix: &str,
    shmem_base: usize,
    col_stride: usize, // block_kv
) {
    ptx.push_str(&format!(
        "    // Load A-fragment TRANSPOSED from S_tile at shmem[{}..] (col_stride={})\n",
        shmem_base, col_stride
    ));
    // Compute actual_m_row = m_tile*16 + a_row
    // m_byte_off encodes m_tile * MMA_M * something — but for transposed reads,
    // we just need m_tile*16 + a_row as the column index into S_tile
    // So we compute: m_col = m_tile * 16 + a_row
    ptx.push_str(&format!(
        "    mul.lo.u32 %bwd_mma_addr2, %bwd_mma_m_tile, {};  // m_tile * MMA_M\n",
        MMA_M
    ));
    ptx.push_str("    add.u32 %bwd_mma_addr2, %bwd_mma_addr2, %bwd_mma_a_row;  // + a_row = m_col\n");

    for i in 0..4 {
        let k_pair = i * 4; // within MMA_K=16, register i covers elements [k_pair, k_pair+1]
        // For each pair of k-values (2 consecutive):
        //   k_val_lo = k_iter * 16 + k_pair
        //   k_val_hi = k_iter * 16 + k_pair + 1
        // addr_lo = shmem_base + (k_val_lo * col_stride + m_col) * 4
        // addr_hi = shmem_base + (k_val_hi * col_stride + m_col) * 4

        // k_val_lo = k_iter * 16 + k_pair
        ptx.push_str(&format!(
            "    mul.lo.u32 %bwd_mma_addr, %bwd_mma_k_iter, {};  // k_iter * MMA_K\n",
            MMA_K
        ));
        if k_pair > 0 {
            ptx.push_str(&format!(
                "    add.u32 %bwd_mma_addr, %bwd_mma_addr, {};  // + k_pair\n",
                k_pair
            ));
        }
        // addr = (k_val * col_stride + m_col) * 4 + shmem_base
        ptx.push_str(&format!(
            "    mul.lo.u32 %bwd_mma_addr, %bwd_mma_addr, {};  // k_val * col_stride\n",
            col_stride
        ));
        ptx.push_str("    add.u32 %bwd_mma_addr, %bwd_mma_addr, %bwd_mma_addr2;  // + m_col\n");
        ptx.push_str("    shl.b32 %bwd_mma_addr, %bwd_mma_addr, 2;  // * 4\n");
        ptx.push_str(&format!(
            "    add.u32 %bwd_mma_addr, %bwd_mma_addr, {};  // + shmem base\n",
            shmem_base
        ));
        emit_smem_load(ptx, "f32", "%bwd_mma_f32_lo", "%bwd_mma_addr");

        // k_val_hi: stride by col_stride * 4 bytes
        ptx.push_str(&format!(
            "    add.u32 %bwd_mma_addr, %bwd_mma_addr, {};  // + col_stride * 4 (next row)\n",
            col_stride * 4
        ));
        emit_smem_load(ptx, "f32", "%bwd_mma_f32_hi", "%bwd_mma_addr");

        ptx.push_str("    cvt.rn.f16.f32 %bwd_mma_h0, %bwd_mma_f32_lo;\n");
        ptx.push_str("    cvt.rn.f16.f32 %bwd_mma_h1, %bwd_mma_f32_hi;\n");
        ptx.push_str(&format!(
            "    mov.b32 {frag_prefix}_{i}, {{%bwd_mma_h0, %bwd_mma_h1}};\n"
        ));
    }
}

/// Load B-fragment from a row-major tile where the B-operand's row dimension is indexed
/// by b_row (from laneid mapping) and the k-iteration indexes rows of the source tile.
///
/// Used for K/V tiles stored as [block_kv, hd_padded] where the k-dimension iterates over
/// the head_dim (columns) and n-tiles index different row groups.
///
/// This is the standard B-fragment load from the forward kernel.
fn emit_load_b_fragment_row_major(
    ptx: &mut String,
    frag_name: &str,
    shmem_base: usize,
    row_stride_elems: usize,
    n_tile: usize,
) {
    ptx.push_str(&format!(
        "    // B-fragment for n_tile={} from shmem[{}..]\n",
        n_tile, shmem_base
    ));
    for bi in 0..2 {
        let k_pair = bi * 8;
        ptx.push_str(&format!(
            "    mul.lo.u32 %bwd_mma_addr, %bwd_mma_b_row, {};  // b_row * row_stride * 4\n",
            row_stride_elems * 4
        ));
        ptx.push_str(&format!(
            "    mad.lo.u32 %bwd_mma_addr, %bwd_mma_k_iter, {}, %bwd_mma_addr;  // + k_iter * MMA_K * 4\n",
            MMA_K * 4
        ));
        ptx.push_str(&format!(
            "    add.u32 %bwd_mma_addr, %bwd_mma_addr, {};  // + k_pair * 4\n",
            k_pair * 4
        ));
        ptx.push_str(&format!(
            "    add.u32 %bwd_mma_addr, %bwd_mma_addr, {};  // + n_tile * MMA_N * row_stride * 4\n",
            n_tile * MMA_N * row_stride_elems * 4
        ));
        if shmem_base > 0 {
            ptx.push_str(&format!(
                "    add.u32 %bwd_mma_addr, %bwd_mma_addr, {};  // + shmem base\n",
                shmem_base
            ));
        }
        emit_smem_load(ptx, "f32", "%bwd_mma_f32_lo", "%bwd_mma_addr");
        ptx.push_str("    add.u32 %bwd_mma_addr, %bwd_mma_addr, 4;\n");
        emit_smem_load(ptx, "f32", "%bwd_mma_f32_hi", "%bwd_mma_addr");
        ptx.push_str("    cvt.rn.f16.f32 %bwd_mma_h0, %bwd_mma_f32_lo;\n");
        ptx.push_str("    cvt.rn.f16.f32 %bwd_mma_h1, %bwd_mma_f32_hi;\n");
        ptx.push_str(&format!(
            "    mov.b32 {frag_name}_{bi}, {{%bwd_mma_h0, %bwd_mma_h1}};\n"
        ));
    }
}

/// Load B-fragment from a row-major tile where k_iter indexes ROWS of the source
/// (used for dV/dK/dQ where the reduction dimension is block_q or block_kv and
/// the output columns come from head_dim).
///
/// Source tile: [reduction_dim, hd_padded]
/// B-fragment for n_tile nt covers columns [nt*8 .. nt*8+8)
/// k_iter covers rows [k_iter*16 .. k_iter*16+16)
/// b_row maps to the thread's row within the 16-row k-tile
///
/// Unlike the standard B load, here the k_iter multiplies by row_stride (not column stride)
/// because we're iterating over rows of the source tile.
fn emit_load_b_fragment_row_major_k_is_q(
    ptx: &mut String,
    frag_name: &str,
    shmem_base: usize,
    row_stride_elems: usize,
    n_tile: usize,
) {
    ptx.push_str(&format!(
        "    // B-fragment (k=row) for n_tile={} from shmem[{}..]\n",
        n_tile, shmem_base
    ));
    for bi in 0..2 {
        let k_row_offset = bi * 8; // b_row offset within the 16-row tile: 0 or 8
        // row = k_iter * 16 + b_row + k_row_offset
        // col = n_tile * 8 + (element within pair: 0, 1)
        // addr = (row * row_stride + col) * 4 + shmem_base
        ptx.push_str(&format!(
            "    mad.lo.u32 %bwd_mma_addr, %bwd_mma_k_iter, {}, 0;  // k_iter * MMA_K * row_stride * 4\n",
            MMA_K * row_stride_elems * 4
        ));
        ptx.push_str(&format!(
            "    mad.lo.u32 %bwd_mma_addr, %bwd_mma_b_row, {}, %bwd_mma_addr;  // + b_row * row_stride * 4\n",
            row_stride_elems * 4
        ));
        if k_row_offset > 0 {
            ptx.push_str(&format!(
                "    add.u32 %bwd_mma_addr, %bwd_mma_addr, {};  // + k_row_offset * row_stride * 4\n",
                k_row_offset * row_stride_elems * 4
            ));
        }
        // Column offset: n_tile * MMA_N * 4 (bytes)
        ptx.push_str(&format!(
            "    add.u32 %bwd_mma_addr, %bwd_mma_addr, {};  // + n_tile * MMA_N * 4\n",
            n_tile * MMA_N * 4
        ));
        if shmem_base > 0 {
            ptx.push_str(&format!(
                "    add.u32 %bwd_mma_addr, %bwd_mma_addr, {};  // + shmem base\n",
                shmem_base
            ));
        }
        emit_smem_load(ptx, "f32", "%bwd_mma_f32_lo", "%bwd_mma_addr");
        ptx.push_str("    add.u32 %bwd_mma_addr, %bwd_mma_addr, 4;\n");
        emit_smem_load(ptx, "f32", "%bwd_mma_f32_hi", "%bwd_mma_addr");
        ptx.push_str("    cvt.rn.f16.f32 %bwd_mma_h0, %bwd_mma_f32_lo;\n");
        ptx.push_str("    cvt.rn.f16.f32 %bwd_mma_h1, %bwd_mma_f32_hi;\n");
        ptx.push_str(&format!(
            "    mov.b32 {frag_name}_{bi}, {{%bwd_mma_h0, %bwd_mma_h1}};\n"
        ));
    }
}

/// Store MMA accumulator tile results to shared memory.
/// Optionally applies the %scale factor before storing.
fn emit_scale_and_store_mma_tile(
    ptx: &mut String,
    acc_prefix: &str,
    n_tiles: usize,
    shmem_base: usize,
    row_width: usize, // number of columns in the output tile (e.g. block_kv)
    apply_scale: bool,
) {
    ptx.push_str("    // Store MMA accumulator results to shmem\n");

    for nt in 0..n_tiles {
        for r in 0..4 {
            if apply_scale {
                ptx.push_str(&format!(
                    "    mul.f32 {acc_prefix}_{nt}_{r}, {acc_prefix}_{nt}_{r}, %scale;\n"
                ));
            }
            let row_offset = if r >= 2 { 8 } else { 0 };
            let col_offset = if r % 2 == 1 { 4 } else { 0 };
            // row = m_tile*16 + a_row + row_offset
            ptx.push_str(&format!(
                "    mul.lo.u32 %bwd_mma_store_row, %bwd_mma_m_tile, {};\n",
                MMA_M
            ));
            ptx.push_str(
                "    add.u32 %bwd_mma_store_row, %bwd_mma_store_row, %bwd_mma_a_row;\n",
            );
            if row_offset > 0 {
                ptx.push_str(&format!(
                    "    add.u32 %bwd_mma_store_row, %bwd_mma_store_row, {};\n",
                    row_offset
                ));
            }
            // addr = shmem_base + (row * row_width + nt*8 + col0 + col_offset) * 4
            ptx.push_str(&format!(
                "    mul.lo.u32 %bwd_mma_store_addr, %bwd_mma_store_row, {};\n",
                row_width
            ));
            ptx.push_str(&format!(
                "    add.u32 %bwd_mma_store_addr, %bwd_mma_store_addr, {};  // + nt*MMA_N + col_offset\n",
                nt * MMA_N + col_offset
            ));
            ptx.push_str(
                "    add.u32 %bwd_mma_store_addr, %bwd_mma_store_addr, %bwd_mma_store_col;\n",
            );
            ptx.push_str("    shl.b32 %bwd_mma_store_addr, %bwd_mma_store_addr, 2;\n");
            ptx.push_str(&format!(
                "    add.u32 %bwd_mma_store_addr, %bwd_mma_store_addr, {};\n",
                shmem_base
            ));
            ptx.push_str("    cvt.u64.u32 %bwd_mma_store_addr_64, %bwd_mma_store_addr;\n");
            emit_smem_store(ptx, "f32", "%bwd_mma_store_addr_64", &format!("{acc_prefix}_{nt}_{r}"));
        }
    }
}

/// Accumulate MMA results to shared memory via atom.shared.add.f32 (no scaling).
/// Used for dV += P^T @ dO.
fn emit_accumulate_mma_to_shmem(
    ptx: &mut String,
    acc_prefix: &str,
    n_tiles: usize,
    shmem_base: usize,
    row_stride_elems: usize, // hd_padded
) {
    ptx.push_str("    // Accumulate MMA results to shmem via atomicAdd\n");

    for nt in 0..n_tiles {
        for r in 0..4 {
            let row_offset = if r >= 2 { 8 } else { 0 };
            let col_offset = if r % 2 == 1 { 4 } else { 0 };
            // row = m_tile*16 + a_row + row_offset
            ptx.push_str(&format!(
                "    mul.lo.u32 %bwd_mma_store_row, %bwd_mma_m_tile, {};\n",
                MMA_M
            ));
            ptx.push_str(
                "    add.u32 %bwd_mma_store_row, %bwd_mma_store_row, %bwd_mma_a_row;\n",
            );
            if row_offset > 0 {
                ptx.push_str(&format!(
                    "    add.u32 %bwd_mma_store_row, %bwd_mma_store_row, {};\n",
                    row_offset
                ));
            }
            // col = nt*8 + col0 + col_offset
            // addr = shmem_base + (row * row_stride + col) * 4
            ptx.push_str(&format!(
                "    mul.lo.u32 %bwd_mma_store_addr, %bwd_mma_store_row, {};\n",
                row_stride_elems
            ));
            ptx.push_str(&format!(
                "    add.u32 %bwd_mma_store_addr, %bwd_mma_store_addr, {};  // + nt*MMA_N + col_offset\n",
                nt * MMA_N + col_offset
            ));
            ptx.push_str(
                "    add.u32 %bwd_mma_store_addr, %bwd_mma_store_addr, %bwd_mma_store_col;\n",
            );
            ptx.push_str("    shl.b32 %bwd_mma_store_addr, %bwd_mma_store_addr, 2;\n");
            ptx.push_str(&format!(
                "    add.u32 %bwd_mma_store_addr, %bwd_mma_store_addr, {};\n",
                shmem_base
            ));
            ptx.push_str("    cvt.u64.u32 %bwd_mma_store_addr_64, %bwd_mma_store_addr;\n");
            emit_smem_atom_add_f32(ptx, "%bwd_mma_f32_lo", "%bwd_mma_store_addr_64", &format!("{acc_prefix}_{nt}_{r}"));
        }
    }
}

/// Accumulate MMA results to shared memory via atom.shared.add.f32 WITH scale factor.
/// Used for dK += dS^T @ Q * scale.
fn emit_accumulate_mma_to_shmem_scaled(
    ptx: &mut String,
    acc_prefix: &str,
    n_tiles: usize,
    shmem_base: usize,
    row_stride_elems: usize,
) {
    ptx.push_str("    // Accumulate MMA results to shmem via atomicAdd (with scale)\n");

    for nt in 0..n_tiles {
        for r in 0..4 {
            // Scale first
            ptx.push_str(&format!(
                "    mul.f32 {acc_prefix}_{nt}_{r}, {acc_prefix}_{nt}_{r}, %scale;\n"
            ));
            let row_offset = if r >= 2 { 8 } else { 0 };
            let col_offset = if r % 2 == 1 { 4 } else { 0 };
            ptx.push_str(&format!(
                "    mul.lo.u32 %bwd_mma_store_row, %bwd_mma_m_tile, {};\n",
                MMA_M
            ));
            ptx.push_str(
                "    add.u32 %bwd_mma_store_row, %bwd_mma_store_row, %bwd_mma_a_row;\n",
            );
            if row_offset > 0 {
                ptx.push_str(&format!(
                    "    add.u32 %bwd_mma_store_row, %bwd_mma_store_row, {};\n",
                    row_offset
                ));
            }
            ptx.push_str(&format!(
                "    mul.lo.u32 %bwd_mma_store_addr, %bwd_mma_store_row, {};\n",
                row_stride_elems
            ));
            ptx.push_str(&format!(
                "    add.u32 %bwd_mma_store_addr, %bwd_mma_store_addr, {};  // + nt*MMA_N + col_offset\n",
                nt * MMA_N + col_offset
            ));
            ptx.push_str(
                "    add.u32 %bwd_mma_store_addr, %bwd_mma_store_addr, %bwd_mma_store_col;\n",
            );
            ptx.push_str("    shl.b32 %bwd_mma_store_addr, %bwd_mma_store_addr, 2;\n");
            ptx.push_str(&format!(
                "    add.u32 %bwd_mma_store_addr, %bwd_mma_store_addr, {};\n",
                shmem_base
            ));
            ptx.push_str("    cvt.u64.u32 %bwd_mma_store_addr_64, %bwd_mma_store_addr;\n");
            emit_smem_atom_add_f32(ptx, "%bwd_mma_f32_lo", "%bwd_mma_store_addr_64", &format!("{acc_prefix}_{nt}_{r}"));
        }
    }
}

/// Emit atomicAdd of MMA dQ results to global dQ memory (with scale).
/// dQ[global_q_row, d] += acc * scale
/// global_q_row = q_start + m_tile*16 + thread_row
fn emit_atomicadd_mma_to_global_dq(
    ptx: &mut String,
    acc_prefix: &str,
    n_tiles: usize,
    config: &FlashAttentionBackwardConfig,
) {
    ptx.push_str("    // AtomicAdd MMA dQ results to global dQ (with scale)\n");

    for nt in 0..n_tiles {
        for r in 0..4 {
            // Scale
            ptx.push_str(&format!(
                "    mul.f32 {acc_prefix}_{nt}_{r}, {acc_prefix}_{nt}_{r}, %scale;\n"
            ));
            let row_offset: usize = if r >= 2 { 8 } else { 0 };
            let col_offset: usize = if r % 2 == 1 { 4 } else { 0 };

            // global_q_row = q_start + m_tile*16 + a_row + row_offset
            // We compute the 64-bit global address:
            // dQ_addr = dq_ptr + (bh_elem_offset + global_q_row * head_dim + col) * 4
            ptx.push_str(&format!(
                "    mul.lo.u32 %bwd_mma_store_row, %bwd_mma_m_tile, {};\n",
                MMA_M
            ));
            ptx.push_str(
                "    add.u32 %bwd_mma_store_row, %bwd_mma_store_row, %bwd_mma_a_row;\n",
            );
            if row_offset > 0 {
                ptx.push_str(&format!(
                    "    add.u32 %bwd_mma_store_row, %bwd_mma_store_row, {};\n",
                    row_offset
                ));
            }
            // Convert to u64 and add q_start (%rd25)
            ptx.push_str(
                "    cvt.u64.u32 %bwd_mma_gaddr, %bwd_mma_store_row;  // row as u64\n",
            );
            ptx.push_str(
                "    add.u64 %bwd_mma_gaddr, %bwd_mma_gaddr, %rd25;   // + q_start\n",
            );
            // global_q_row * head_dim
            ptx.push_str(
                "    mul.lo.u64 %bwd_mma_gaddr, %bwd_mma_gaddr, %rd10;  // * head_dim\n",
            );
            // + bh_elem_offset (%rd15)
            ptx.push_str(
                "    add.u64 %bwd_mma_gaddr, %bwd_mma_gaddr, %rd15;\n",
            );
            // + col (nt*8 + col0 + col_offset)
            ptx.push_str(&format!(
                "    cvt.u64.u32 %rd35, %bwd_mma_store_col;  // col0 as u64\n"
            ));
            ptx.push_str(&format!(
                "    add.u64 %bwd_mma_gaddr, %bwd_mma_gaddr, {};  // + nt*MMA_N + col_offset\n",
                nt * MMA_N + col_offset
            ));
            ptx.push_str(
                "    add.u64 %bwd_mma_gaddr, %bwd_mma_gaddr, %rd35;  // + col0\n",
            );
            // * 4 bytes
            ptx.push_str(
                "    shl.b64 %bwd_mma_gaddr, %bwd_mma_gaddr, 2;\n",
            );
            // + dq_ptr (%rd4)
            ptx.push_str(
                "    add.u64 %bwd_mma_gaddr, %bwd_mma_gaddr, %rd4;  // + dq_ptr\n",
            );
            ptx.push_str(&format!(
                "    atom.global.add.f32 %bwd_mma_f32_lo, [%bwd_mma_gaddr], {acc_prefix}_{nt}_{r};\n"
            ));
        }
    }

    let _ = config; // config used for documentation/future extensions
}

/// Store dK_local and dV_local from shared memory to global dK/dV arrays.
fn emit_bwd_main_store_dk_dv(ptx: &mut String, config: &FlashAttentionBackwardConfig) {
    let hd_padded = config.head_dim + BWD_PAD;
    let k_tile_bytes = config.block_kv * hd_padded * 4;
    let v_tile_bytes = config.block_kv * hd_padded * 4;
    let q_tile_bytes = config.block_q * hd_padded * 4;
    let do_tile_bytes = config.block_q * hd_padded * 4;
    let dk_shmem_offset = k_tile_bytes + v_tile_bytes + q_tile_bytes + do_tile_bytes;
    let dv_shmem_offset = dk_shmem_offset + config.block_kv * hd_padded * 4;

    ptx.push_str("    // === Store dK_local and dV_local to global memory ===\n");

    // dK global base: dk_ptr + (bh_elem_offset + kv_start * head_dim) * 4
    ptx.push_str("    mul.lo.u64 %rd20, %rd14, %rd10;  // kv_start * head_dim\n");
    ptx.push_str("    add.u64 %rd20, %rd15, %rd20;\n");
    ptx.push_str("    shl.b64 %rd20, %rd20, 2;\n");
    ptx.push_str("    add.u64 %rd20, %rd5, %rd20;  // dk_global_base\n\n");

    // dV global base
    ptx.push_str("    mul.lo.u64 %rd21, %rd14, %rd10;\n");
    ptx.push_str("    add.u64 %rd21, %rd15, %rd21;\n");
    ptx.push_str("    shl.b64 %rd21, %rd21, 2;\n");
    ptx.push_str("    add.u64 %rd21, %rd6, %rd21;  // dv_global_base\n\n");

    // Store dK: thread tid stores rows starting at tid, stride block_q
    ptx.push_str("    mov.u64 %rd22, %rd11;  // nj = tid\n");
    ptx.push_str("BWD_MAIN_STORE_DK:\n");
    ptx.push_str(&format!(
        "    setp.ge.u64 %p0, %rd22, {};\n",
        config.block_kv
    ));
    ptx.push_str("    @%p0 bra BWD_MAIN_STORE_DK_DONE;\n");
    ptx.push_str("    mov.u64 %rd23, 0;  // d = 0\n");
    ptx.push_str("BWD_MAIN_STORE_DK_D:\n");
    ptx.push_str("    setp.ge.u64 %p1, %rd23, %rd10;\n");
    ptx.push_str("    @%p1 bra BWD_MAIN_STORE_DK_D_DONE;\n");
    // shmem addr: dk_shmem_offset + (nj * hd_padded + d) * 4
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd24, %rd22, {};\n",
        hd_padded
    ));
    ptx.push_str("    add.u64 %rd24, %rd24, %rd23;\n");
    ptx.push_str("    shl.b64 %rd24, %rd24, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd24, %rd24, {};\n",
        dk_shmem_offset
    ));
    emit_smem_load(ptx, "f32", "%f0", "%rd24");
    // global addr: dk_global_base + (nj * head_dim + d) * 4
    ptx.push_str("    mul.lo.u64 %rd25, %rd22, %rd10;\n");
    ptx.push_str("    add.u64 %rd25, %rd25, %rd23;\n");
    ptx.push_str("    shl.b64 %rd25, %rd25, 2;\n");
    ptx.push_str("    add.u64 %rd25, %rd20, %rd25;\n");
    ptx.push_str("    st.global.f32 [%rd25], %f0;\n");
    ptx.push_str("    add.u64 %rd23, %rd23, 1;\n");
    ptx.push_str("    bra BWD_MAIN_STORE_DK_D;\n");
    ptx.push_str("BWD_MAIN_STORE_DK_D_DONE:\n");
    ptx.push_str(&format!(
        "    add.u64 %rd22, %rd22, {};\n",
        config.block_q
    ));
    ptx.push_str("    bra BWD_MAIN_STORE_DK;\n");
    ptx.push_str("BWD_MAIN_STORE_DK_DONE:\n\n");

    // Store dV
    ptx.push_str("    mov.u64 %rd22, %rd11;  // nj = tid\n");
    ptx.push_str("BWD_MAIN_STORE_DV:\n");
    ptx.push_str(&format!(
        "    setp.ge.u64 %p0, %rd22, {};\n",
        config.block_kv
    ));
    ptx.push_str("    @%p0 bra BWD_MAIN_STORE_DV_DONE;\n");
    ptx.push_str("    mov.u64 %rd23, 0;  // d = 0\n");
    ptx.push_str("BWD_MAIN_STORE_DV_D:\n");
    ptx.push_str("    setp.ge.u64 %p1, %rd23, %rd10;\n");
    ptx.push_str("    @%p1 bra BWD_MAIN_STORE_DV_D_DONE;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd24, %rd22, {};\n",
        hd_padded
    ));
    ptx.push_str("    add.u64 %rd24, %rd24, %rd23;\n");
    ptx.push_str("    shl.b64 %rd24, %rd24, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd24, %rd24, {};\n",
        dv_shmem_offset
    ));
    emit_smem_load(ptx, "f32", "%f0", "%rd24");
    ptx.push_str("    mul.lo.u64 %rd25, %rd22, %rd10;\n");
    ptx.push_str("    add.u64 %rd25, %rd25, %rd23;\n");
    ptx.push_str("    shl.b64 %rd25, %rd25, 2;\n");
    ptx.push_str("    add.u64 %rd25, %rd21, %rd25;\n");
    ptx.push_str("    st.global.f32 [%rd25], %f0;\n");
    ptx.push_str("    add.u64 %rd23, %rd23, 1;\n");
    ptx.push_str("    bra BWD_MAIN_STORE_DV_D;\n");
    ptx.push_str("BWD_MAIN_STORE_DV_D_DONE:\n");
    ptx.push_str(&format!(
        "    add.u64 %rd22, %rd22, {};\n",
        config.block_q
    ));
    ptx.push_str("    bra BWD_MAIN_STORE_DV;\n");
    ptx.push_str("BWD_MAIN_STORE_DV_DONE:\n\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── CshaExtras field tests ─────────────────────────────────────────

    #[test]
    fn cshaextras_save_activations_defaults_false() {
        let e = CshaExtras::default();
        assert!(!e.save_activations_for_backward);
    }

    #[test]
    fn cshaextras_save_activations_independent_of_fused_projections() {
        let mut e = CshaExtras::default();
        e.save_activations_for_backward = true;
        assert!(e.save_activations_for_backward);
        assert!(!e.fused_projections); // independent flags
    }

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
        assert!(!use_mma_path(52)); // Kepler
        assert!(!use_mma_path(70)); // Volta (has wmma but not f16 mma.sync)
        assert!(use_mma_path(80)); // Ampere
        assert!(use_mma_path(89)); // Ada Lovelace
        assert!(use_mma_path(90)); // Hopper
    }

    // ── MMA PTX emission tests ─────────────────────────────────────

    #[test]
    fn test_f32_to_f16_pack_emission() {
        let mut ptx = String::new();
        let src = vec![
            "f0".to_string(),
            "f1".to_string(),
            "f2".to_string(),
            "f3".to_string(),
        ];
        let dst = vec!["a0".to_string(), "a1".to_string()];
        emit_f32_to_f16_pack(&mut ptx, &src, &dst);

        assert!(
            ptx.contains("cvt.rn.f16.f32 %mma_h0, %f0"),
            "first lo conversion"
        );
        assert!(
            ptx.contains("cvt.rn.f16.f32 %mma_h1, %f1"),
            "first hi conversion"
        );
        assert!(
            ptx.contains("mov.b32 %a0, {%mma_h0, %mma_h1}"),
            "first pack"
        );
        assert!(
            ptx.contains("cvt.rn.f16.f32 %mma_h0, %f2"),
            "second lo conversion"
        );
        assert!(
            ptx.contains("mov.b32 %a1, {%mma_h0, %mma_h1}"),
            "second pack"
        );
    }

    #[test]
    fn test_mma_instruction_emission() {
        let mut ptx = String::new();
        let d = ["d0".into(), "d1".into(), "d2".into(), "d3".into()];
        let a = ["a0".into(), "a1".into(), "a2".into(), "a3".into()];
        let b = ["b0".into(), "b1".into()];
        let c = ["c0".into(), "c1".into(), "c2".into(), "c3".into()];
        crate::matmul_mma::emit_mma_instruction(&mut ptx, &d, &a, &b, &c);

        assert!(
            ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"),
            "must contain MMA instruction"
        );
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

        assert!(
            ptx.contains("wgmma shared memory descriptor"),
            "comment present"
        );
        assert!(
            ptx.contains(".reg .b64 %desc_q"),
            "descriptor register declared"
        );
        assert!(ptx.contains("shr.u32"), "base address shift");
        assert!(ptx.contains("0x3FFF"), "14-bit mask for base");
    }

    #[test]
    fn test_qk_matmul_wgmma_emission() {
        let mut ptx = String::new();
        emit_qk_matmul_wgmma(&mut ptx, 64, 64, 64, 64 * 64 * 4);

        assert!(
            ptx.contains("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16"),
            "wgmma instruction present"
        );
        assert!(
            ptx.contains("wgmma.commit_group.sync.aligned"),
            "commit present"
        );
        assert!(
            ptx.contains("wgmma.wait_group.sync.aligned"),
            "wait present"
        );
        assert!(
            ptx.contains("Softmax scalar math can overlap"),
            "async overlap comment"
        );
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

        assert!(
            ptx.contains("wgmma.mma_async.sync.aligned"),
            "wgmma present"
        );
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

    #[test]
    fn test_sm80_path_does_not_emit_wgmma() {
        // Regression: sm_80 (A100) should use mma.sync, NOT wgmma
        let gpu = crate::gpu_specs::find_gpu("A100-SXM").unwrap();
        assert!(!gpu.supports_wgmma(), "A100 should not support wgmma");
        assert!(gpu.supports_fp16_mma(), "A100 should support mma.sync");
        assert_eq!(gpu.warp_group_size(), 32, "A100 uses 32-thread warps");
    }

    #[test]
    fn test_sm90_selects_wgmma_over_mma() {
        let gpu = crate::gpu_specs::find_gpu("H100-SXM").unwrap();
        assert!(gpu.supports_wgmma(), "H100 supports wgmma");
        assert!(gpu.supports_fp16_mma(), "H100 also supports mma.sync");
        assert_eq!(
            gpu.warp_group_size(),
            128,
            "H100 uses 128-thread warp groups"
        );
        // Dispatch should prefer wgmma when available
        // (tested via FP8 compile_fp8_matmul in fp8.rs)
    }

    #[test]
    fn test_wgmma_tile_constraints() {
        // wgmma m64n64k16: block sizes must be multiples of 64
        assert_eq!(64 % WGMMA_N, 0, "64 is valid");
        assert_eq!(128 % WGMMA_N, 0, "128 is valid");
        // Validate k alignment
        assert_eq!(64 % WGMMA_K_F16, 0, "head_dim=64 is k-aligned");
        assert_eq!(128 % WGMMA_K_F16, 0, "head_dim=128 is k-aligned");
    }

    // ── mma.sync (Ampere sm_80) tests ─────────────────────────────────

    #[test]
    fn test_load_a_fragment_emission() {
        let mut ptx = String::new();
        let regs: [String; 4] = ["aq0".into(), "aq1".into(), "aq2".into(), "aq3".into()];
        crate::matmul_mma::emit_load_a_fragment_smem(&mut ptx, &regs, "shmem_q", 256);

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
        crate::matmul_mma::emit_load_b_fragment_smem(&mut ptx, &regs, "shmem_k", 128);

        assert!(ptx.contains("Load B-fragment"), "comment present");
        assert_eq!(ptx.matches("ld.shared.b32").count(), 2, "2 fragment loads");
    }

    #[test]
    fn test_qk_mma_ptx_emission() {
        let mut ptx = String::new();
        // block_q=64, block_kv=64, head_dim=64
        emit_qk_matmul_mma(&mut ptx, 64, 64, 64, 64 * 64 * 4);

        assert!(ptx.contains("Q@K^T via MMA"), "section comment");
        assert!(
            ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"),
            "MMA instruction present"
        );
        assert!(
            ptx.contains("cvt.rn.f16.f32"),
            "f32->f16 conversion present"
        );
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

        assert!(
            ptx.contains("Online softmax (MMA layout)"),
            "section comment"
        );
        // Row max reduction
        assert!(
            ptx.contains("max.f32 %mma_local_max"),
            "local max computation"
        );
        assert!(ptx.contains("shfl.sync.bfly.b32"), "warp shuffle present");
        // Correction factor
        assert!(
            ptx.contains("ex2.approx.f32 %mma_correction"),
            "exp correction"
        );
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
                assert!(
                    bank1 != bank_prev || bank2 != bank1,
                    "three consecutive rows hit same bank: row={}, banks={},{},{}",
                    row - 1,
                    bank_prev,
                    bank1,
                    bank2
                );
            }
        }
    }

    #[test]
    fn test_mma_softmax_registers_emission() {
        let mut ptx = String::new();
        emit_mma_softmax_registers(&mut ptx);

        assert!(ptx.contains("%mma_row_max"), "row_max declared");
        assert!(ptx.contains("%mma_row_sum"), "row_sum declared");
        assert!(ptx.contains("0fFF800000"), "row_max initialized to -inf");
        assert!(ptx.contains("0f00000000"), "row_sum initialized to 0");
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
        assert_eq!(
            mma_count, 16,
            "expected 16 MMA instructions (8 Q@K^T + 8 P@V), got {}",
            mma_count
        );

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
        assert!(
            ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"),
            "MMA instruction present"
        );
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

        assert!(
            ptx.contains(".reg .f16 %mma_h0, %mma_h1"),
            "f16 temps declared"
        );
        assert!(
            ptx.contains(".reg .u32 %mma_a_row"),
            "A row register declared"
        );
        assert!(ptx.contains("%mma_laneid"), "laneid register used");
        assert!(
            ptx.contains("and.b32 %mma_laneid, %mma_laneid, 31"),
            "laneid = tid.x % 32"
        );
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
            num_sink_tokens: 0,
            gpu_sm: 80,
        segment_masked: false,
        csha: None,
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
            num_sink_tokens: 0,
            gpu_sm: 80,
        segment_masked: false,
        csha: None,
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
            num_sink_tokens: 0,
            gpu_sm: 80,
        segment_masked: false,
        csha: None,
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
            num_sink_tokens: 0,
            gpu_sm: 80,
        segment_masked: false,
        csha: None,
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
            num_sink_tokens: 0,
            gpu_sm: 80,
        segment_masked: false,
        csha: None,
        };
        let ptx = synthesize_flash_attention_ptx(&config);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len() - 1]).unwrap(); // strip null
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
            num_sink_tokens: 0,
            gpu_sm: 80,
        segment_masked: false,
        csha: None,
        };
        let ptx_no_causal = synthesize_flash_attention_ptx(&config);
        let str_no = std::str::from_utf8(&ptx_no_causal[..ptx_no_causal.len() - 1]).unwrap();
        assert!(str_no.contains("Non-causal"));
        assert!(!str_no.contains("Zero-divergence"));

        config.causal = true;
        let ptx_causal = synthesize_flash_attention_ptx(&config);
        let str_c = std::str::from_utf8(&ptx_causal[..ptx_causal.len() - 1]).unwrap();
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
            num_sink_tokens: 0,
            gpu_sm: 80,
        segment_masked: false,
        csha: None,
        };
        let ptx = synthesize_flash_attention_ptx(&config);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len() - 1]).unwrap();
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
            num_sink_tokens: 0,
            gpu_sm: 80,
        segment_masked: false,
        csha: None,
        };
        let ptx = synthesize_flash_attention_ptx(&config);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len() - 1]).unwrap();
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
            num_sink_tokens: 0,
            gpu_sm: 80,
        segment_masked: false,
        csha: None,
        };
        let ptx = synthesize_flash_attention_ptx(&config);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len() - 1]).unwrap();
        assert!(ptx_str.contains("kv_head = q_head / 4"));
    }

    #[test]
    fn test_rope_cache_write_ptx() {
        let ptx = synthesize_rope_cache_write_ptx(128, RopeStyle::HalfSplit);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len() - 1]).unwrap();
        assert!(ptx_str.contains("nsl_rope_cache_write"));
        assert!(ptx_str.contains("seq_ids_ptr"));
        assert!(ptx_str.contains("seq_lens_ptr"));
        assert!(ptx_str.contains("half_split"));
    }

    #[test]
    fn test_rope_cache_write_adjacent() {
        let ptx = synthesize_rope_cache_write_ptx(128, RopeStyle::Adjacent);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len() - 1]).unwrap();
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
            num_sink_tokens: 0,
            gpu_sm: 80,
        segment_masked: false,
        csha: None,
        };
        let ptx = synthesize_flash_attention_ptx(&config);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len() - 1]).unwrap();
        // Verify tree mask params and DFS ancestor check are present
        assert!(
            ptx_str.contains("dfs_enter_ptr"),
            "missing dfs_enter_ptr param"
        );
        assert!(
            ptx_str.contains("dfs_exit_ptr"),
            "missing dfs_exit_ptr param"
        );
        assert!(
            ptx_str.contains("num_tree_nodes"),
            "missing num_tree_nodes param"
        );
        assert!(
            ptx_str.contains("dfs_enter_base"),
            "missing DFS register load"
        );
        assert!(ptx_str.contains("Tree mask"), "missing tree mask comment");
        // Verify tree_mask=true kernel name includes t1
        let name = flash_attention_kernel_name(&config);
        assert!(
            name.contains("_t1_"),
            "kernel name should contain _t1_ for tree_mask=true, got {}",
            name
        );
    }

    // ── Backward main kernel (Phase 2) tests ──────────────────────────

    fn bwd_config(causal: bool) -> FlashAttentionBackwardConfig {
        FlashAttentionBackwardConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 64,
            causal,
            gpu_sm: 80,
            segment_masked: false,
        }
    }

    #[test]
    fn test_bwd_main_kernel_name() {
        let cfg = bwd_config(true);
        assert_eq!(
            flash_attention_bwd_main_kernel_name(&cfg),
            "flash_attn_bwd_main_c1_q64_kv64"
        );
        let cfg2 = bwd_config(false);
        assert_eq!(
            flash_attention_bwd_main_kernel_name(&cfg2),
            "flash_attn_bwd_main_c0_q64_kv64"
        );
    }

    #[test]
    fn test_bwd_main_synthesize_noncausal() {
        let cfg = bwd_config(false);
        let (_, ptx2_bytes) = synthesize_flash_attention_backward_ptx(&cfg);
        let ptx = std::str::from_utf8(&ptx2_bytes[..ptx2_bytes.len() - 1]).unwrap();

        // Entry point
        assert!(
            ptx.contains("flash_attn_bwd_main_c0_q64_kv64"),
            "kernel name in entry"
        );
        // Parameters
        assert!(ptx.contains("param_dout"), "dout param");
        assert!(ptx.contains("param_dq"), "dq param");
        assert!(ptx.contains("param_dk"), "dk param");
        assert!(ptx.contains("param_dv"), "dv param");
        assert!(ptx.contains("param_lse"), "lse param");
        assert!(ptx.contains("param_scale"), "scale param");
        assert!(ptx.contains("param_seq_len"), "seq_len param");
        assert!(ptx.contains("param_head_dim"), "head_dim param");
    }

    #[test]
    fn test_bwd_main_has_atomic_dq() {
        let cfg = bwd_config(false);
        let (_, ptx2_bytes) = synthesize_flash_attention_backward_ptx(&cfg);
        let ptx = std::str::from_utf8(&ptx2_bytes[..ptx2_bytes.len() - 1]).unwrap();
        assert!(
            ptx.contains("atom.global.add.f32"),
            "dQ must use atomicAdd"
        );
    }

    #[test]
    fn test_bwd_main_has_regular_store_dk_dv() {
        let cfg = bwd_config(false);
        let (_, ptx2_bytes) = synthesize_flash_attention_backward_ptx(&cfg);
        let ptx = std::str::from_utf8(&ptx2_bytes[..ptx2_bytes.len() - 1]).unwrap();
        assert!(
            ptx.contains("BWD_MAIN_STORE_DK"),
            "dK store section"
        );
        assert!(
            ptx.contains("BWD_MAIN_STORE_DV"),
            "dV store section"
        );
        // dK/dV use regular st.global, not atomic
        assert!(
            ptx.contains("st.global.f32"),
            "dK/dV use regular global store"
        );
    }

    #[test]
    fn test_bwd_main_has_barriers() {
        let cfg = bwd_config(false);
        let (_, ptx2_bytes) = synthesize_flash_attention_backward_ptx(&cfg);
        let ptx = std::str::from_utf8(&ptx2_bytes[..ptx2_bytes.len() - 1]).unwrap();
        let barrier_count = ptx.matches("bar.sync 0").count();
        // At least 3: after KV load, after dK/dV zero init, after steps 3a-3g
        assert!(
            barrier_count >= 3,
            "need at least 3 bar.sync 0, got {}",
            barrier_count
        );
    }

    #[test]
    fn test_bwd_main_has_exp_via_ex2() {
        let cfg = bwd_config(false);
        let (_, ptx2_bytes) = synthesize_flash_attention_backward_ptx(&cfg);
        let ptx = std::str::from_utf8(&ptx2_bytes[..ptx2_bytes.len() - 1]).unwrap();
        assert!(ptx.contains("ex2.approx.f32"), "exp via ex2.approx");
        assert!(
            ptx.contains("0f3FB8AA3B") || ptx.contains("0x3FB8AA3B"),
            "log2(e) constant"
        );
    }

    #[test]
    fn test_bwd_main_causal_mask() {
        let cfg = bwd_config(true);
        let (_, ptx2_bytes) = synthesize_flash_attention_backward_ptx(&cfg);
        let ptx = std::str::from_utf8(&ptx2_bytes[..ptx2_bytes.len() - 1]).unwrap();
        // Causal: i_block starts at j_block
        assert!(
            ptx.contains("i_block = j_block (causal)"),
            "causal loop start"
        );
        // Causal mask check
        assert!(
            ptx.contains("global_i < global_j"),
            "causal mask predicate"
        );
        // Force P = 0
        assert!(
            ptx.contains("force P = 0 for masked"),
            "masked P zeroing"
        );
    }

    #[test]
    fn test_bwd_main_noncausal_no_mask() {
        let cfg = bwd_config(false);
        let (_, ptx2_bytes) = synthesize_flash_attention_backward_ptx(&cfg);
        let ptx = std::str::from_utf8(&ptx2_bytes[..ptx2_bytes.len() - 1]).unwrap();
        // Non-causal: i_block starts at 0
        assert!(
            ptx.contains("i_block = 0 (non-causal)"),
            "non-causal loop start"
        );
        assert!(
            !ptx.contains("force P = 0 for masked"),
            "no causal mask in non-causal variant"
        );
    }

    #[test]
    fn test_bwd_main_shared_mem_size() {
        let cfg = bwd_config(false);
        let shmem = backward_shared_mem_bytes(&cfg);
        // block_q=64, block_kv=64, head_dim=64, pad=4, hd_padded=68
        // K: 64*68*4=17408, V: same, Q: 64*68*4=17408, dO: same
        // dK: 17408, dV: 17408
        // S_tile: 64*64*4=16384, D: 64*4=256, L: 64*4=256
        // Total: 6*17408 + 16384 + 256 + 256 = 104448 + 16896 = 121344
        // ... just check it's a reasonable size
        assert!(shmem > 0, "shmem must be > 0");
        let (_, ptx2_bytes) = synthesize_flash_attention_backward_ptx(&cfg);
        let ptx = std::str::from_utf8(&ptx2_bytes[..ptx2_bytes.len() - 1]).unwrap();
        assert!(
            ptx.contains(&format!("shmem[{}]", shmem)),
            "shared memory declaration matches computed size"
        );
    }

    #[test]
    fn test_bwd_main_all_7_steps_present() {
        let cfg = bwd_config(true);
        let (_, ptx2_bytes) = synthesize_flash_attention_backward_ptx(&cfg);
        let ptx = std::str::from_utf8(&ptx2_bytes[..ptx2_bytes.len() - 1]).unwrap();
        assert!(ptx.contains("Step 3a"), "step 3a: S = Q.K * scale");
        assert!(ptx.contains("Step 3b"), "step 3b: P = exp(S - L)");
        assert!(ptx.contains("Step 3c"), "step 3c: dV += P^T @ dO");
        assert!(ptx.contains("Step 3d"), "step 3d: dP = dO @ V^T");
        assert!(ptx.contains("Step 3e"), "step 3e: dS = P * (dP - D)");
        assert!(ptx.contains("Step 3f"), "step 3f: dQ += dS @ K * scale");
        assert!(ptx.contains("Step 3g"), "step 3g: dK += dS^T @ Q * scale");
    }

    #[test]
    fn test_bwd_main_null_terminated() {
        let cfg = bwd_config(false);
        let (_, ptx2_bytes) = synthesize_flash_attention_backward_ptx(&cfg);
        assert_eq!(
            *ptx2_bytes.last().unwrap(),
            0,
            "PTX must be null-terminated"
        );
    }

    #[test]
    fn test_bwd_main_mma_path_sm80() {
        let cfg = FlashAttentionBackwardConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 64,
            causal: false,
            gpu_sm: 80,
            segment_masked: false,
        };
        assert!(use_mma_path(cfg.gpu_sm), "sm_80 should use MMA path");
        let (_, ptx2_bytes) = synthesize_flash_attention_backward_ptx(&cfg);
        let ptx = std::str::from_utf8(&ptx2_bytes[..ptx2_bytes.len() - 1]).unwrap();

        // MMA path indicators
        assert!(
            ptx.contains("MMA Step 3a"),
            "MMA Step 3a comment (S = Q@K^T)"
        );
        assert!(
            ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"),
            "MMA instruction present in backward kernel"
        );
        assert!(
            ptx.contains("cvt.rn.f16.f32"),
            "f32->f16 conversion for MMA fragments"
        );
        assert!(
            ptx.contains("%bwd_acc_s_"),
            "backward S accumulator registers"
        );
        assert!(
            ptx.contains("%bwd_aq_"),
            "backward A-fragment registers for Q"
        );
        assert!(
            ptx.contains("%bwd_bk_"),
            "backward B-fragment registers for K"
        );
        assert!(
            ptx.contains("BWD_MAIN_MMA_S_K_LOOP"),
            "MMA K-dimension loop label"
        );
        // MMA results stored to S_tile shmem
        assert!(
            ptx.contains("S_tile fully computed via MMA"),
            "S_tile store barrier comment"
        );
        // All 7 steps still present
        assert!(ptx.contains("Step 3a"), "step 3a");
        assert!(ptx.contains("Step 3b"), "step 3b");
        assert!(ptx.contains("Step 3c"), "step 3c");
        assert!(ptx.contains("Step 3d"), "step 3d");
        assert!(ptx.contains("Step 3e"), "step 3e");
        assert!(ptx.contains("Step 3f"), "step 3f");
        assert!(ptx.contains("Step 3g"), "step 3g");
        // MMA path should NOT contain the scalar S dot product loop
        assert!(
            !ptx.contains("BWD_MAIN_S_DOT:"),
            "scalar S dot loop should NOT be in MMA path"
        );
    }

    #[test]
    fn test_bwd_main_scalar_path_sm52() {
        let cfg = FlashAttentionBackwardConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 64,
            causal: false,
            gpu_sm: 52,
            segment_masked: false,
        };
        assert!(!use_mma_path(cfg.gpu_sm), "sm_52 should NOT use MMA path");
        let (_, ptx2_bytes) = synthesize_flash_attention_backward_ptx(&cfg);
        let ptx = std::str::from_utf8(&ptx2_bytes[..ptx2_bytes.len() - 1]).unwrap();

        // Scalar path indicators
        assert!(
            ptx.contains("BWD_MAIN_S_DOT:"),
            "scalar S dot product loop"
        );
        assert!(
            ptx.contains("BWD_MAIN_DP_DOT:"),
            "scalar dP dot product loop"
        );
        // Should NOT contain MMA instructions
        assert!(
            !ptx.contains("mma.sync.aligned"),
            "no MMA instructions in scalar path"
        );
        assert!(
            !ptx.contains("%bwd_acc_s_"),
            "no MMA accumulator registers in scalar path"
        );
        // All 7 steps still present
        assert!(ptx.contains("Step 3a"), "step 3a");
        assert!(ptx.contains("Step 3b"), "step 3b");
        assert!(ptx.contains("Step 3c"), "step 3c");
        assert!(ptx.contains("Step 3d"), "step 3d");
        assert!(ptx.contains("Step 3e"), "step 3e");
        assert!(ptx.contains("Step 3f"), "step 3f");
        assert!(ptx.contains("Step 3g"), "step 3g");
    }

    #[test]
    fn test_bwd_main_mma_tile_count() {
        // Verify correct number of MMA instructions for block_kv=64, head_dim=64
        // 5 matmuls: S, dP (n_tiles_s=8 each), dV, dQ, dK (n_tiles_hd=8 each)
        // Total = 5 * 8 = 40
        let cfg = bwd_config(false);
        let (_, ptx2_bytes) = synthesize_flash_attention_backward_ptx(&cfg);
        let ptx = std::str::from_utf8(&ptx2_bytes[..ptx2_bytes.len() - 1]).unwrap();
        let mma_count = ptx.matches("mma.sync.aligned.m16n8k16").count();
        assert_eq!(
            mma_count, 40,
            "expected 40 MMA instructions (5 matmuls * 8 n-tiles), got {}",
            mma_count
        );
    }

    #[test]
    fn test_bwd_main_labels_unique_prefix() {
        let cfg = bwd_config(false);
        let (_, ptx2_bytes) = synthesize_flash_attention_backward_ptx(&cfg);
        let ptx = std::str::from_utf8(&ptx2_bytes[..ptx2_bytes.len() - 1]).unwrap();
        // All labels should start with BWD_MAIN_
        for line in ptx.lines() {
            let trimmed = line.trim();
            if trimmed.ends_with(':') && !trimmed.starts_with("//") && !trimmed.starts_with('.') {
                assert!(
                    trimmed.starts_with("BWD_MAIN_"),
                    "label '{}' does not have BWD_MAIN_ prefix",
                    trimmed
                );
            }
        }
    }

    // ── CSHA A.2.2: RMSNorm prologue emission tests ──────────────────

    fn csha_l1_config() -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 128,
            causal: true,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            num_sink_tokens: 0,
            gpu_sm: 80,
            segment_masked: false,
            csha: Some(CshaExtras::level1(1e-5)),
        }
    }

    fn non_csha_config() -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 128,
            causal: true,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            num_sink_tokens: 0,
            gpu_sm: 80,
            segment_masked: false,
            csha: None,
        }
    }

    #[test]
    fn a22_rmsnorm_prologue_appears_with_fused_rmsnorm() {
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&csha_l1_config())).unwrap();
        // Prologue marker comment must appear — this is the anchor
        // downstream assertions use to confirm the block was emitted.
        assert!(
            ptx.contains("CSHA A.2.2: RMSNorm prologue"),
            "prologue block missing for CSHA L1 config"
        );
        // Core PTX ops that the prologue emits.
        assert!(ptx.contains("ld.param.u64 %rd50, [csha_x_ptr];"));
        assert!(ptx.contains("ld.param.u64 %rd51, [csha_norm_weight_ptr];"));
        assert!(ptx.contains("ld.param.f32 %f100, [csha_rmsnorm_eps];"));
        assert!(ptx.contains("rsqrt.approx.f32 %f101, %f101;"));
        assert!(ptx.contains("st.shared.b16 [%rd58], %h0;"));
        // Labels must be present for the runtime null-check branch.
        assert!(ptx.contains("CSHA_PROLOGUE_SUMSQ:"));
        assert!(ptx.contains("CSHA_PROLOGUE_APPLY:"));
        assert!(ptx.contains("CSHA_PROLOGUE_END:"));
    }

    #[test]
    fn a22_rmsnorm_prologue_absent_without_csha() {
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&non_csha_config())).unwrap();
        assert!(
            !ptx.contains("CSHA A.2.2: RMSNorm prologue"),
            "prologue must not appear in non-CSHA PTX"
        );
        assert!(!ptx.contains("CSHA_PROLOGUE_SUMSQ:"));
        assert!(!ptx.contains("CSHA_PROLOGUE_APPLY:"));
    }

    #[test]
    fn a22_rmsnorm_prologue_absent_when_fused_rmsnorm_off() {
        // CSHA plumbing enabled but prologue fusion explicitly off —
        // the kernel variant registers the CSHA param list (so ABI stays
        // stable across cshaL*/non-CSHA variants) but the prologue
        // PTX body is still elided.
        let mut cfg = csha_l1_config();
        cfg.csha.as_mut().unwrap().fused_rmsnorm = false;
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&cfg)).unwrap();
        assert!(
            !ptx.contains("CSHA A.2.2: RMSNorm prologue"),
            "prologue must respect the fused_rmsnorm gate"
        );
    }

    #[test]
    fn a22_rmsnorm_prologue_runtime_null_check_uses_or_pred() {
        // The prologue must short-circuit if EITHER csha_x_ptr OR
        // csha_norm_weight_ptr is null — otherwise threading only one
        // of them (possible during A.2.1e rollout) would dereference a
        // null pointer in the other. Check the or.pred guard.
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&csha_l1_config())).unwrap();
        assert!(
            ptx.contains("setp.eq.u64 %p10, %rd50, 0;")
                && ptx.contains("setp.eq.u64 %p11, %rd51, 0;")
                && ptx.contains("or.pred %p10, %p10, %p11;")
                && ptx.contains("@%p10 bra CSHA_PROLOGUE_END;"),
            "runtime null-check on x_ptr/norm_weight_ptr is missing"
        );
    }

    // ── CSHA A.2.3: matmul projection emission tests ─────────────────

    fn csha_l2_config() -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 128,
            causal: true,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            num_sink_tokens: 0,
            gpu_sm: 80,
            segment_masked: false,
            csha: Some(CshaExtras::level2(1e-5, 512)),
        }
    }

    #[test]
    fn a23_projection_appears_with_fused_projections() {
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&csha_l2_config())).unwrap();
        assert!(
            ptx.contains("CSHA A.2.3: matmul projection"),
            "projection block missing for CSHA L2 config"
        );
        // Tile-sweep: three MMAs (one per projection iteration, reused
        // across M/N/K loops at runtime). Exactly three `mma.sync`
        // statically-emitted instructions — the loop structure above
        // dispatches them many times at runtime.
        assert_eq!(
            ptx.matches("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32")
                .count(),
            3,
            "exactly three statically-emitted m16n8k16 MMAs (Q/K/V loop bodies)"
        );
        // Section markers for each projection.
        assert!(ptx.contains("--- Q projection:"));
        assert!(ptx.contains("--- K projection:"));
        assert!(ptx.contains("--- V projection:"));
        // End label + bar.sync.
        assert!(ptx.contains("CSHA_PROJECTION_END:"));
    }

    #[test]
    fn a23_projection_tile_sweep_has_nested_mnk_loops() {
        // Tile-sweep expansion: every projection has an M-loop, N-loop,
        // and K-loop, each a runtime branch backed by a unique label.
        // The label count verifies the loop structure is generated for
        // all three projections (9 labels total: 3 × {M, N, K}).
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&csha_l2_config())).unwrap();
        for tag in ["Q", "K", "V"] {
            for dim in ["M", "N", "K"] {
                let label = format!("CSHA_PROJ_{}_{}_LOOP:", tag, dim);
                assert!(
                    ptx.contains(&label),
                    "missing loop label {} in CSHA L2 PTX",
                    label
                );
            }
        }
        // Each projection increments its counters by 1 per iteration,
        // so we expect 9 `add.u32 %ts_{m,n,k}, ..., 1;` increments
        // (one per loop per projection).
        assert_eq!(
            ptx.matches("add.u32 %ts_m, %ts_m, 1;").count(),
            3,
            "one M-loop increment per projection"
        );
        assert_eq!(
            ptx.matches("add.u32 %ts_n, %ts_n, 1;").count(),
            3,
            "one N-loop increment per projection"
        );
        assert_eq!(
            ptx.matches("add.u32 %ts_k, %ts_k, 1;").count(),
            3,
            "one K-loop increment per projection"
        );
    }

    #[test]
    fn a23_fragment_pack_converts_f32_accumulator_to_f16_before_smem_store() {
        // A.2.3 follow-up: per-(m, n) tile closure packs the 4 f32
        // accumulator lanes into 2 .b32 registers (each holding 2
        // packed f16 values) via `emit_f32_to_f16_pack`. Verifies the
        // conversion is actually emitted rather than the old
        // placeholder `st.shared.b32 %proj_c0` raw-f32 store.
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&csha_l2_config())).unwrap();
        // Three projections × 2 f32→f16 conversion pairs each
        // (emit_f32_to_f16_pack emits one cvt per f32 src, 4 per call).
        assert_eq!(
            ptx.matches("cvt.rn.f16.f32 %mma_h0, %proj_c").count(),
            6,
            "two f32→f16 conversion calls per projection × 3 projections"
        );
        // A.2.3.2 lane-coherent scatter: each pack is stored via a
        // per-lane address (%ts_pack0_addr / %ts_pack1_addr), not the
        // bare tile origin. The tile-origin stores of the old
        // scaffold must be gone.
        assert!(ptx.contains("st.shared.b32 [%ts_pack0_addr], %proj_c_pack0;"));
        assert!(ptx.contains("st.shared.b32 [%ts_pack1_addr], %proj_c_pack1;"));
        assert!(!ptx.contains("st.shared.b32 [%ts_out_base + 0], %proj_c_pack0"));
        assert!(!ptx.contains("st.shared.b32 [%ts_out_base + 4], %proj_c_pack1"));
    }

    #[test]
    fn a232_lane_coherent_scatter_matches_mma_layout() {
        // A.2.3.2: the per-(m, n) tile closure must emit the scatter
        // address math documented for mma.sync.m16n8k16 f32-accumulator
        // layout — thread t holds (row=t/4, col=2*(t%4)+{0,1}) and
        // (row+8, same col). Pack0 stores the low row, pack1 the high
        // row.
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&csha_l2_config())).unwrap();
        // Lane → row / col math
        assert!(ptx.contains("shr.u32 %ts_row_lo, %mma_laneid, 2;"));
        assert!(ptx.contains("and.b32 %ts_col_pair, %mma_laneid, 3;"));
        assert!(ptx.contains("shl.b32 %ts_col_off, %ts_col_pair, 2;"));
        // Row offset = row_lo * stride (stride = head_dim*2 = 256 here).
        assert!(ptx.contains("mul.lo.u32 %ts_row_off, %ts_row_lo, 256;"));
        // pack0 = out_base + row_off + col_off
        assert!(ptx.contains("add.u32 %ts_pack0_addr, %ts_out_base, %ts_row_off;"));
        assert!(ptx.contains("add.u32 %ts_pack0_addr, %ts_pack0_addr, %ts_col_off;"));
        // pack1 = pack0 + 8*stride (8 rows lower)
        assert!(ptx.contains("add.u32 %ts_pack1_addr, %ts_pack0_addr, 2048;"));
        // Per-lane scatter stores (not tile-origin stores)
        assert_eq!(
            ptx.matches("st.shared.b32 [%ts_pack0_addr], %proj_c_pack0;").count(),
            3,
            "one pack0 scatter store per projection (Q/K/V)"
        );
        assert_eq!(
            ptx.matches("st.shared.b32 [%ts_pack1_addr], %proj_c_pack1;").count(),
            3,
            "one pack1 scatter store per projection"
        );
    }

    #[test]
    fn a24_rope_cos_sin_indexing_uses_per_row_per_d_addressing() {
        // A.2.4 follow-up: cos/sin addresses now compute
        // `cos_ptr + ((q_start + row) * head_dim + d) * 4` per pair,
        // rather than sampling position 0 like the original scaffold.
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&csha_l2_rope_config())).unwrap();
        // Two sweeps (Q, K), each computes `(q_start + row) * head_dim`,
        // adds `d`, shifts << 2, and adds to cos_ptr / sin_ptr.
        assert_eq!(
            ptx.matches("add.u64 %rope_row_u64, %rope_row_u64, %rd16;").count(),
            2,
            "q_start added once per sweep (Q + K)"
        );
        assert_eq!(
            ptx.matches("shl.b64 %rope_cs_addr, %rope_cs_addr, 2;").count(),
            2,
            "* 4 bytes indexing shift per sweep"
        );
        assert_eq!(
            ptx.matches("add.u64 %rope_row_u64, %rd12, %rope_cs_addr;").count(),
            2,
            "cos_ptr-relative addr computed per sweep"
        );
        assert_eq!(
            ptx.matches("add.u64 %rope_row_u64, %rd13, %rope_cs_addr;").count(),
            2,
            "sin_ptr-relative addr computed per sweep"
        );
        // The old placeholder-load pattern (sampling %rd12 directly
        // without offset computation) must no longer appear.
        assert!(!ptx.contains("ld.global.f32 %rope_cos, [%rd12];"));
        assert!(!ptx.contains("ld.global.f32 %rope_sin, [%rd13];"));
    }

    #[test]
    fn a23_projection_tile_sweep_uses_register_based_smem_addresses() {
        // Per-iteration A and B fragment loads must address the per-tile
        // base register (%ts_a_base / %ts_b_base), not a compile-time
        // literal, so the MMA primitive walks the correct tile each
        // iteration. The load is emitted as `mul.lo.u32` + `add.u32`
        // rather than `mad.lo.u32` because the fused multiply-add form is
        // rejected by CUDA at runtime on PTX ISA 7.0 (see MEMORY.md).
        //
        // PTX-spec-rewrite update (post N4 helper rewrite, 2026-05-15): the
        // A-fragment helper now computes per-lane addressing internally
        // from `%lane` and writes to `%mma_a_row` as a scratch (rather
        // than expecting callers to pre-set `%mma_a_row`). Emission is:
        //   shr.u32 %mma_addr, %lane, 2          ; row_lo = lane / 4
        //   mul.lo.u32 %mma_a_row, %mma_addr, <stride>
        //   and.b32 %mma_addr, %lane, 3
        //   shl.b32 %mma_addr, %mma_addr, 2      ; col_lo_bytes
        //   add.u32 %mma_a_row, %mma_a_row, %mma_addr
        //   add.u32 %mma_a_row, %mma_a_row, <base>
        //   ld.shared.b32 ...                    ; reg 0
        // Test assertions track the new pattern (multiplicand order swap;
        // accumulator reg is %mma_a_row not %mma_addr; base add target
        // is %mma_a_row).
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&csha_l2_config())).unwrap();
        // A stride = head_dim * 2 = 128 * 2 = 256 bytes.
        // The new helper computes row_lo in %mma_a_row in-place
        // (`shr.u32 %mma_a_row, %mma_addr, 2`) then multiplies that
        // accumulator by the stride. Pattern is therefore
        // `mul.lo.u32 %mma_a_row, %mma_a_row, <stride>`.
        assert!(
            ptx.contains("mul.lo.u32 %mma_a_row, %mma_a_row, 256;"),
            "A-fragment load must multiply row offset by the tile stride; got:\n{}",
            &ptx[..ptx.len().min(4000)]
        );
        assert!(
            ptx.contains("add.u32 %mma_a_row, %mma_a_row, %ts_a_base;"),
            "A-fragment load must add the tile-base register to the row+col offset"
        );
        // B-fragment helper post-rewrite: same self-contained per-lane
        // derivation as A-frag. Emission stores n_col*col_stride in
        // %mma_b_row (in-place after shr → mul) then adds k_lo bytes +
        // base. Pattern is `mul.lo.u32 %mma_b_row, %mma_b_row, <stride>`.
        assert!(
            ptx.contains("mul.lo.u32 %mma_b_row, %mma_b_row, 512;"),
            "B-fragment load must multiply n_col by the col stride"
        );
        assert!(
            ptx.contains("add.u32 %mma_b_row, %mma_b_row, %ts_b_base;"),
            "B-fragment load must add the tile-base register to the n_col+k_lo offset"
        );
    }

    #[test]
    fn a23_projection_absent_without_fused_projections() {
        // CSHA L1 only has fused_rmsnorm; projections stay external.
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&csha_l1_config())).unwrap();
        assert!(
            !ptx.contains("CSHA A.2.3: matmul projection"),
            "projection must not appear in CSHA L1 PTX (fused_projections=false)"
        );
        assert!(!ptx.contains("CSHA_PROJECTION_END:"));
    }

    #[test]
    fn a23_projection_absent_in_non_csha_ptx() {
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&non_csha_config())).unwrap();
        assert!(!ptx.contains("CSHA A.2.3: matmul projection"));
        // And the non-CSHA path emits exactly the attention-body MMAs it
        // already had (sanity: no CSHA-tagged MMAs leaking through).
        assert!(!ptx.contains("CSHA_PROJECTION_END:"));
    }

    #[test]
    fn a23_projection_runtime_null_check_covers_all_three_weights() {
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&csha_l2_config())).unwrap();
        // Must set-predicate-eq for each of Wq/Wk/Wv and or.pred-combine
        // them, then branch to PROJECTION_END if any is null.
        assert!(ptx.contains("ld.param.u64 %rd61, [csha_wq_ptr];"));
        assert!(ptx.contains("ld.param.u64 %rd62, [csha_wk_ptr];"));
        assert!(ptx.contains("ld.param.u64 %rd63, [csha_wv_ptr];"));
        assert!(ptx.contains("setp.eq.u64 %p13, %rd61, 0;"));
        assert!(ptx.contains("setp.eq.u64 %p14, %rd62, 0;"));
        assert!(ptx.contains("setp.eq.u64 %p14, %rd63, 0;"));
        assert!(ptx.contains("@%p13 bra CSHA_PROJECTION_END;"));
    }

    // ── CSHA A.4: active_heads guard emission tests ──────────────────

    fn csha_pruned_config(active_heads: u32) -> FlashAttentionConfig {
        let mut cfg = csha_l1_config();
        cfg.csha.as_mut().unwrap().active_heads = active_heads;
        cfg
    }

    #[test]
    fn a4_active_heads_guard_absent_when_zero() {
        // active_heads=0 is the "no pruning" signal — kernel runs the
        // full head count and the guard must not appear (otherwise ptxas
        // sees a `bid_y >= 0` check which is never false but still
        // emits the branch instruction).
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&csha_l1_config())).unwrap();
        assert!(!ptx.contains("CSHA A.4: active_heads guard"));
        assert!(!ptx.contains("@%p9 ret;"));
    }

    #[test]
    fn a4_active_heads_guard_emits_literal_bound() {
        let cfg = csha_pruned_config(5);
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&cfg)).unwrap();
        assert!(
            ptx.contains("CSHA A.4: active_heads guard"),
            "guard must appear when active_heads > 0"
        );
        // The compile-time literal (5) appears in the setp operand,
        // letting ptxas fold the comparison against a constant rather
        // than a runtime-loaded param.
        assert!(
            ptx.contains("setp.ge.u64 %p9, %rd18, 5;"),
            "guard must use the compile-time active_heads literal"
        );
        assert!(ptx.contains("@%p9 ret;"));
    }

    #[test]
    fn a4_active_heads_guard_follows_index_computation() {
        // The guard must appear AFTER `emit_index_computation` (which
        // writes %rd18 = head_idx) but BEFORE the A.2.2 prologue — so
        // dead heads exit without doing any SMEM work.
        let cfg = csha_pruned_config(3);
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&cfg)).unwrap();
        let head_idx = ptx
            .find("head_idx = bid_y % heads")
            .expect("index computation writes head_idx");
        let guard_idx = ptx.find("CSHA A.4: active_heads guard").unwrap();
        let prologue_idx = ptx.find("CSHA A.2.2: RMSNorm prologue").unwrap();
        assert!(head_idx < guard_idx, "guard must follow head_idx computation");
        assert!(
            guard_idx < prologue_idx,
            "guard must precede the prologue so dead heads skip SMEM work"
        );
    }

    #[test]
    fn a4_active_heads_guard_absent_in_non_csha_ptx() {
        // Sanity: the guard is purely a CSHA feature.
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&non_csha_config())).unwrap();
        assert!(!ptx.contains("CSHA A.4: active_heads guard"));
    }

    // ── CSHA A.2.4: RoPE epilogue emission tests ────────────────────

    fn csha_l2_rope_config() -> FlashAttentionConfig {
        let mut cfg = csha_l2_config();
        cfg.rope_q = true;
        cfg.rope_style = RopeStyle::HalfSplit;
        cfg
    }

    #[test]
    fn a24_rope_epilogue_appears_with_rope_plus_fused_projections() {
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&csha_l2_rope_config())).unwrap();
        assert!(
            ptx.contains("CSHA A.2.4: RoPE epilogue"),
            "RoPE epilogue block missing for CSHA L2 + rope_q config"
        );
        // Tile-sweep markers — both Q and K now rotate across the full
        // (block_q × head_dim/2) pair grid, not just a single element.
        assert!(ptx.contains("--- Q rotation sweep:"));
        assert!(ptx.contains("--- K rotation sweep:"));
        // Rotation math: Q and K each do cos*a - sin*b and sin*a + cos*b.
        assert!(ptx.contains("sub.f32 %rope_q_rot_a"));
        assert!(ptx.contains("add.f32 %rope_q_rot_b"));
        assert!(ptx.contains("sub.f32 %rope_k_rot_a"));
        assert!(ptx.contains("add.f32 %rope_k_rot_b"));
        // Write-back into SMEM (tile-sweep replaces the previous
        // register-only %proj_d* write-back).
        assert!(ptx.contains("st.shared.f32 [%rope_elem_a_off], %rope_q_rot_a;"));
        assert!(ptx.contains("st.shared.f32 [%rope_elem_a_off], %rope_k_rot_a;"));
        // End label.
        assert!(ptx.contains("CSHA_ROPE_EPILOGUE_END:"));
    }

    #[test]
    fn a24_rope_tile_sweep_has_cooperative_pair_loop() {
        // The Q-sweep and K-sweep are each implemented as a runtime
        // loop strided by blockDim.x (128) so all 128 threads
        // cooperate over the flat pair index space. Check the loop
        // labels + the `+128` stride increment are present.
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&csha_l2_rope_config())).unwrap();
        assert!(ptx.contains("CSHA_ROPE_Q_LOOP:"));
        assert!(ptx.contains("CSHA_ROPE_K_LOOP:"));
        assert!(ptx.contains("CSHA_ROPE_K_START:"));
        // blockDim.x stride appears in both loops.
        assert_eq!(
            ptx.matches("add.u32 %rope_pair_idx, %rope_pair_idx, 128;").count(),
            2,
            "one stride increment each for Q and K sweeps"
        );
    }

    #[test]
    fn a24_rope_epilogue_absent_without_rope() {
        // fused_projections on, rope_q off → epilogue must not appear;
        // the A.2.3 projection still does.
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&csha_l2_config())).unwrap();
        assert!(
            !ptx.contains("CSHA A.2.4: RoPE epilogue"),
            "RoPE epilogue must respect the rope_q gate"
        );
        assert!(ptx.contains("CSHA A.2.3: matmul projection"));
    }

    #[test]
    fn a24_rope_epilogue_absent_without_fused_projections() {
        // CSHA L1 (fused_rmsnorm only) + rope_q on. Pre-CSHA RoPE path
        // in `emit_q_tile_load` handles rotation from HBM; the A.2.4
        // epilogue is only meaningful when Q/K come from A.2.3's
        // in-kernel projection.
        let mut cfg = csha_l1_config();
        cfg.rope_q = true;
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&cfg)).unwrap();
        assert!(
            !ptx.contains("CSHA A.2.4: RoPE epilogue"),
            "RoPE epilogue requires fused_projections — L1 must not emit it"
        );
    }

    #[test]
    fn a24_rope_epilogue_runtime_null_check_on_cos_sin() {
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&csha_l2_rope_config())).unwrap();
        // cos_ptr is %rd12, sin_ptr is %rd13 per `emit_param_loads`.
        assert!(ptx.contains("setp.eq.u64 %p15, %rd12, 0;"));
        assert!(ptx.contains("setp.eq.u64 %p14, %rd13, 0;"));
        assert!(ptx.contains("or.pred %p15, %p15, %p14;"));
        assert!(ptx.contains("@%p15 bra CSHA_ROPE_EPILOGUE_END;"));
    }

    #[test]
    fn a24_rope_epilogue_follows_projection() {
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&csha_l2_rope_config())).unwrap();
        let projection_idx = ptx
            .find("CSHA A.2.3: matmul projection")
            .expect("projection present");
        let epilogue_idx = ptx
            .find("CSHA A.2.4: RoPE epilogue")
            .expect("epilogue present");
        let q_load_idx = ptx
            .find("Load Q tile into shared memory")
            .expect("Q tile load present");
        assert!(
            projection_idx < epilogue_idx,
            "RoPE epilogue must follow A.2.3 projection"
        );
        assert!(
            epilogue_idx < q_load_idx,
            "RoPE epilogue must precede the classic Q-from-HBM load"
        );
    }

    #[test]
    fn a24_rope_epilogue_absent_in_non_csha_ptx() {
        // Non-CSHA + rope_q uses the original in-Q-load RoPE path; the
        // CSHA-specific epilogue must NOT leak in.
        let mut cfg = non_csha_config();
        cfg.rope_q = true;
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&cfg)).unwrap();
        assert!(!ptx.contains("CSHA A.2.4: RoPE epilogue"));
        assert!(!ptx.contains("CSHA_ROPE_EPILOGUE_END:"));
        // But the original RoPE path must still be active.
        assert!(ptx.contains("LOOP_Q_LOAD_ROPE:"));
    }

    #[test]
    fn a23_projection_follows_prologue_precedes_q_tile_load() {
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&csha_l2_config())).unwrap();
        let prologue_idx = ptx
            .find("CSHA A.2.2: RMSNorm prologue")
            .expect("prologue present (L2 turns on fused_rmsnorm too)");
        let projection_idx = ptx
            .find("CSHA A.2.3: matmul projection")
            .expect("projection present");
        let q_load_idx = ptx
            .find("Load Q tile into shared memory")
            .expect("Q tile load present");
        assert!(
            prologue_idx < projection_idx,
            "prologue must precede projection"
        );
        assert!(
            projection_idx < q_load_idx,
            "projection must precede classic Q load"
        );
    }

    #[test]
    fn a22_rmsnorm_prologue_precedes_q_tile_load() {
        // The prologue must emit BEFORE the classic Q-from-HBM load so
        // that when null pointers short-circuit the prologue, Q load
        // still fills SMEM and the kernel behaves identically to the
        // non-CSHA variant at runtime (current A.1/A.2.1x state).
        let ptx = String::from_utf8(synthesize_flash_attention_ptx(&csha_l1_config())).unwrap();
        let prologue_idx = ptx.find("CSHA A.2.2: RMSNorm prologue").expect("prologue present");
        let q_load_idx = ptx
            .find("Load Q tile into shared memory")
            .expect("Q tile load present");
        assert!(
            prologue_idx < q_load_idx,
            "prologue at offset {} must precede Q tile load at offset {}",
            prologue_idx,
            q_load_idx
        );
    }

    // ── FlashAttentionConfig::validate() tests ────────────────────────

    #[test]
    fn flash_attention_config_validate_rejects_paged_segment_masked_combo() {
        // Spec §3.2 invariant: segment_masked + paged are mutually
        // exclusive.
        let mut cfg = FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 64,
            causal: true,
            paged: true,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            num_sink_tokens: 0,
            gpu_sm: 80,
            segment_masked: false,
            csha: None,
        };

        // Case 1: paged=true alone is fine.
        assert!(cfg.validate().is_ok(), "paged alone should validate");

        // Case 2: segment_masked alone is fine.
        cfg.paged = false;
        cfg.segment_masked = true;
        assert!(cfg.validate().is_ok(), "segment_masked alone should validate");

        // Case 3: both set together must error.
        cfg.paged = true;
        let result = cfg.validate();
        assert!(
            result.is_err(),
            "segment_masked=true AND paged=true must error per spec §3.2"
        );
        let err = result.unwrap_err();
        assert!(
            err.contains("mutually exclusive"),
            "error message should mention mutual exclusion, got: {err}"
        );
    }
}
