//! BitNet b1.58 ternary quantization kernel family.
//!
//! Spec: `docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md`
//!
//! ## Subsystem layout (Phase 1)
//!
//! - `config.rs`: `BitNetKernelConfig` — tile shapes, dtypes, fused-RMSNorm /
//!   fused-bias / fused-residual flags.
//! - `mod.rs`: orchestrator (`synthesize_kernel`); composes phase emitters into a
//!   standalone BitNet GEMM kernel.
//! - `pack.rs`: packed/unpacked conversion ops (host-side, pure Rust). Added in Task 2.
//! - `reference.rs`: CPU reference impl (`#[cfg(test)]`-gated). Added in Task 3.
//! - `phases/`:
//!   - `packed_load.rs` — PUBLIC; HBM→SMEM load + on-the-fly unpack.
//!   - `absmean_quant.rs` — `pub` + `#[doc(hidden)]`; activation quant prologue.
//!     BitNet b1.58 uses absmax per-row for activations; file name is historical.
//!     IR-001 invariant is documentation-enforced (not visibility-enforced) since
//!     integration tests in `tests/` are separate crates and cannot see `pub(super)`.
//!   - `ternary_gemm.rs` — `pub` + `#[doc(hidden)]`; GEMM body (input invariant:
//!     activations must be quantized; documentation-enforced per IR-001).
//!   - `quantized_ternary_gemm.rs` — PUBLIC; fused activation-quant + ternary GEMM.
//!   - `finalize.rs` — PUBLIC; dequant + epilogue.
//!
//! See `phases/README.md` for the Phase 2 (M35.2) planned additions.

pub mod config;
pub mod loader;
pub mod pack;
pub mod phases;

// M35.2a training-mode wrapper — gated on V-P1-D pass per BLOCKED_ON_V_P1_D.md.
// Real implementation lands in M35.2a implementation Stage D.5.
#[doc(hidden)]
pub mod orchestrator_train;

pub use config::BitNetKernelConfig;

use crate::kernel_ir::KirType;

/// Number of `f32` slots reserved in SMEM for the per-warp absmax partials.
/// Sized for the maximum CTA width CUDA permits (1024 threads = 32 warps).
const RED_SLOTS: u32 = 32;

/// Per-CTA shared-memory budget assumed by `synthesize_kernel`.
///
/// 48 KiB is the portable static limit across sm_80 through sm_120; the larger
/// opt-in limits require `cuFuncSetAttribute` at launch, which the current
/// host contract does not do.
const SMEM_BUDGET_BYTES: u64 = 48 * 1024;

/// Launch geometry required by the kernel emitted by [`synthesize_kernel`].
///
/// The host MUST launch with exactly this shape; the emitted PTX derives every
/// index from `%ctaid`/`%tid`/`%ntid` under these assumptions.
///
/// - `grid  = (rows, ceil(out_dim / block_n), 1)`
/// - `block = (block_n, 1, 1)`
///
/// where `rows` is the number of activation rows (tokens) being projected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitNetLaunchGeometry {
    /// `blockDim.x` — one thread per output column within a column tile.
    pub threads_per_cta: u32,
    /// `gridDim.y` — number of column tiles spanning `out_dim`.
    pub column_tiles: u32,
}

impl BitNetKernelConfig {
    /// Launch geometry for a given number of activation rows.
    ///
    /// `gridDim.x` is `rows` and is supplied by the caller; the other two
    /// dimensions are fully determined by the config.
    pub fn launch_geometry(&self) -> BitNetLaunchGeometry {
        BitNetLaunchGeometry {
            threads_per_cta: self.block_n,
            column_tiles: self.out_dim.div_ceil(self.block_n),
        }
    }
}

/// Synthesize a complete BitNet GEMM kernel as PTX bytes.
///
/// Composes the four public phase emitters: `packed_load`,
/// `quantized_ternary_gemm` (which itself fuses activation-quant + ternary GEMM),
/// `finalize`. The bare `absmean_quant` and `ternary_gemm` phases are
/// subsystem-internal and not exposed here per IR-001.
///
/// # Runtime contract
///
/// The emitted module is a complete, assemblable, launchable kernel: it
/// assembles under `ptxas` for sm_80 / sm_90 / sm_120 and reproduces the
/// `tests/bitnet_reference_impl.rs` forward pass bit-exactly (see
/// `tests/bitnet_gpu_correctness.rs`).
///
/// Buffer layouts:
/// - `act_in`: `[rows, hidden_dim]` row-major, `activation_dtype` (F16/BF16).
/// - `weights_packed`: `[out_dim, hidden_dim]` row-major **in trits**, packed
///   4 trits per byte along the `hidden_dim` (k) axis, HIGH-BITS-FIRST per
///   `PACKED_BYTE_LAYOUT.md`. This is the safetensors-native HF `nn.Linear`
///   order (`[out_features, in_features]`), so `loader.rs` output is directly
///   consumable with no transpose. **NOTE** this is the transpose of the
///   `[hidden_dim, out_dim]` convention used by the CPU reference's
///   `forward_reference`; the reference is a math oracle, not a layout spec.
/// - `y_out`: `[rows, out_dim]` row-major, `output_dtype`.
/// - `weight_scale`: per-tensor BitLinear absmean scale (`loader.rs`).
/// - `bias` (only when `fused_bias_add`): `[out_dim]`, **FP32** — note this is
///   NOT `output_dtype`. BitNet b1.58's Llama-style projections carry no bias,
///   so this path is unused in the pinned model; it exists for the optional
///   spec-§4.4 capability.
/// - `residual` (only when `fused_residual_add`): `[rows, out_dim]` row-major,
///   **FP32**, again NOT `output_dtype`. Passing an F16 residual (the natural
///   transformer layout) would be silently misread.
///
/// Requires `hidden_dim % 4 == 0` so that each thread's 4-trit unpack lands on
/// a byte boundary (`loader.rs` enforces the same constraint at pack time).
///
/// # Register contract
///
/// All `.reg` declarations live here, in one block, so phase emitters stay
/// pure text producers and cannot disagree about types. Phases document which
/// registers they read and write in their own module docs.
///
/// # Known performance limitation (correctness-first)
///
/// Each CTA redundantly recomputes the row absmax and re-quantizes the whole
/// activation row into its own SMEM copy, so that work is duplicated
/// `ceil(out_dim / block_n)` times per row. Weights stream straight from HBM
/// with no SMEM staging or `cp.async` pipelining, and the GEMM is scalar
/// (no `mma.sync`). `block_m` and `block_k` are therefore currently carried
/// for the dispatch key and for M35.2a backward tiling only — the forward
/// kernel does not tile over them. Fixing all of this is the tiled follow-up;
/// correctness and a numerical oracle come first.
pub fn synthesize_kernel(config: &BitNetKernelConfig) -> Vec<u8> {
    assert!(
        config.hidden_dim.is_multiple_of(4),
        "BitNet synthesize_kernel: hidden_dim must be a multiple of 4 (4 trits per \
         packed byte), got {}",
        config.hidden_dim
    );
    assert!(
        config.block_n > 0 && config.block_n.is_multiple_of(32) && config.block_n <= 1024,
        "BitNet synthesize_kernel: block_n is the CTA width and must be a positive \
         multiple of 32 no greater than 1024, got {}",
        config.block_n
    );
    // The quantized activation row lives in SMEM (1 byte per hidden element),
    // alongside the reduction scratch and the broadcast scale. Without this
    // check an oversized hidden_dim surfaces as a ptxas "uses too much shared
    // data" error at a distance rather than a named refusal here.
    let smem_bytes = config.hidden_dim as u64 + (RED_SLOTS as u64 * 4) + 4;
    assert!(
        smem_bytes <= SMEM_BUDGET_BYTES,
        "BitNet synthesize_kernel: hidden_dim={} needs {smem_bytes} B of SMEM \
         (quantized row + {RED_SLOTS} reduction slots + scale), over the \
         {SMEM_BUDGET_BYTES} B per-CTA budget. Tiling over hidden_dim is the \
         follow-up that lifts this.",
        config.hidden_dim
    );
    // The flat trit index `col_id * hidden_dim + k` is computed in 32 bits and
    // consumed by packed_load's `shr.b32`. Refuse rather than silently wrap.
    let trit_count = config.out_dim as u64 * config.hidden_dim as u64;
    assert!(
        trit_count <= u32::MAX as u64,
        "BitNet synthesize_kernel: out_dim({}) * hidden_dim({}) = {trit_count} trits \
         exceeds the 32-bit flat weight index",
        config.out_dim,
        config.hidden_dim
    );

    let mut ptx = String::new();

    // PTX module preamble: target sm_80+ (Ampere baseline; matches NSL
    // architecture floor). Convention matches backend_ptx.rs /
    // epilogue_fusion.rs preamble shape.
    //
    // The bf16 `cvt` mnemonics require a newer PTX ISA than 7.0 — ptxas
    // rejects `cvt.f32.bf16` at .version 7.0 with "requires PTX ISA .version
    // 7.1 or later". Mirror the established dtype dispatch from
    // `fused_linear_ce.rs::ptx_header`: F16 stays at 7.0, Bf16 bumps to 8.0.
    let version = match config.activation_dtype {
        KirType::Bf16 => "8.0",
        _ => "7.0",
    };
    ptx.push_str(&format!(".version {version}\n"));
    ptx.push_str(".target sm_80\n");
    ptx.push_str(".address_size 64\n");
    ptx.push_str(&format!("// BitNet kernel: {}\n", config.kernel_name()));

    // ---- SMEM ----
    // qact: the quantized activation row for this CTA's row (int8, one byte
    // per hidden element). red: per-warp absmax partials. scale: the reduced
    // row scale, broadcast to every thread.
    ptx.push_str("// SMEM: quantized activation row + warp-reduction scratch + broadcast scale.\n");
    ptx.push_str(&format!(
        ".shared .align 4 .b8 bitnet_qact[{}];\n",
        config.hidden_dim
    ));
    ptx.push_str(&format!(
        ".shared .align 4 .b8 bitnet_red[{}];\n",
        RED_SLOTS * 4
    ));
    ptx.push_str(".shared .align 4 .b8 bitnet_scale[4];\n");

    ptx.push_str(&format!(".visible .entry {} (\n", config.kernel_name()));
    ptx.push_str("  .param .u64 act_in,\n");
    ptx.push_str("  .param .u64 weights_packed,\n");
    ptx.push_str("  .param .u64 y_out,\n");
    // Per-tensor BitLinear b1.58 absmean scale (one scalar per layer; loaded
    // once by finalize.rs::emit and multiplied into the FP32 accumulator
    // before bias/residual). Required for forward correctness; without it
    // the emitted kernel is off by a per-layer constant factor.
    ptx.push_str("  .param .f32 weight_scale");
    if config.fused_bias_add {
        ptx.push_str(",\n  .param .u64 bias");
    }
    if config.fused_residual_add {
        ptx.push_str(",\n  .param .u64 residual");
    }
    ptx.push_str("\n) {\n");

    emit_register_decls(&mut ptx, config);
    emit_prologue(&mut ptx, config);

    // Compose the phases.
    crate::bitnet::phases::packed_load::emit_prologue(&mut ptx, config);
    crate::bitnet::phases::quantized_ternary_gemm::emit(&mut ptx, config);
    crate::bitnet::phases::finalize::emit(&mut ptx, config);

    // Threads whose column is past out_dim jump here. The label must sit after
    // every `bar.sync` in the kernel — see emit_prologue for why.
    ptx.push_str("BITNET_EXIT:\n");
    ptx.push_str("ret;\n");
    ptx.push_str("}\n");

    ptx.into_bytes()
}

/// Emit every `.reg` declaration the composed phases rely on.
fn emit_register_decls(ptx: &mut String, config: &BitNetKernelConfig) {
    ptx.push_str("// --- register file (declared centrally; see mod.rs register contract) ---\n");
    ptx.push_str(
        ".reg .pred %p_amdone, %p_qdone, %p_gdone, %p_zero, %p_lane0, %p_t0, %p_rdone, %p_cguard;\n",
    );
    ptx.push_str(".reg .b16 %h_x, %h_xq, %h_y, %rs_qi8, %rs_qact;\n");
    ptx.push_str(".reg .b32 %r_row_id, %r_tid, %r_nthreads, %r_col_id, %r_k;\n");
    ptx.push_str(".reg .b32 %r_hidden_dim, %r_out_dim, %r_off, %r_qoff;\n");
    ptx.push_str(".reg .b32 %r_lane, %r_warp, %r_nwarps, %r_shfl_a, %r_shfl_b, %r_i;\n");
    ptx.push_str(".reg .b32 %r_qact_base, %r_red_base, %r_scale_base, %r_addr32;\n");
    ptx.push_str(".reg .b32 %r_byte_offset, %r_trit_index, %r_wbase, %r_packed_word;\n");
    ptx.push_str(".reg .b32 %r_trit0, %r_trit1, %r_trit2, %r_trit3;\n");
    ptx.push_str(".reg .b32 %r_t0_val, %r_t1_val, %r_t2_val, %r_t3_val;\n");
    ptx.push_str(".reg .b32 %r_acc, %r_prod, %r_qact_i32, %r_qi, %r_yoff;\n");
    ptx.push_str(".reg .f32 %f_x, %f_absx, %f_max, %f_partial, %f_scale, %f_inv_scale;\n");
    ptx.push_str(".reg .f32 %f_scaled, %f_clip, %f_half, %f_rndbias, %f_rnd;\n");
    ptx.push_str(".reg .f32 %f_acc, %f_127, %f_inv127_scale, %f_y_out, %f_w_scale;\n");
    ptx.push_str(".reg .b64 %rd_act_in, %rd_w_packed, %rd_y_out;\n");
    ptx.push_str(".reg .b64 %rd_off, %rd_addr, %rd_weight_addr, %rd_byte_off_64, %rd_tmp;\n");
    ptx.push_str(".reg .b64 %rd_yoff, %rd_y_addr;\n");
    if config.fused_bias_add {
        ptx.push_str(".reg .b64 %rd_bias, %rd_bias_off, %rd_bias_addr;\n");
        ptx.push_str(".reg .f32 %f_bias;\n");
    }
    if config.fused_residual_add {
        ptx.push_str(".reg .b64 %rd_residual, %rd_res_off, %rd_res_addr;\n");
        ptx.push_str(".reg .f32 %f_res;\n");
    }
}

/// Load kernel params, derive thread/CTA indices, and take SMEM base addresses.
fn emit_prologue(ptx: &mut String, config: &BitNetKernelConfig) {
    ptx.push_str("// --- prologue: params, indices, SMEM bases ---\n");
    ptx.push_str("ld.param.u64 %rd_act_in, [act_in];\n");
    ptx.push_str("ld.param.u64 %rd_w_packed, [weights_packed];\n");
    ptx.push_str("ld.param.u64 %rd_y_out, [y_out];\n");
    ptx.push_str("cvta.to.global.u64 %rd_act_in, %rd_act_in;\n");
    ptx.push_str("cvta.to.global.u64 %rd_w_packed, %rd_w_packed;\n");
    ptx.push_str("cvta.to.global.u64 %rd_y_out, %rd_y_out;\n");
    if config.fused_bias_add {
        ptx.push_str("ld.param.u64 %rd_bias, [bias];\n");
        ptx.push_str("cvta.to.global.u64 %rd_bias, %rd_bias;\n");
    }
    if config.fused_residual_add {
        ptx.push_str("ld.param.u64 %rd_residual, [residual];\n");
        ptx.push_str("cvta.to.global.u64 %rd_residual, %rd_residual;\n");
    }

    ptx.push_str(&format!(
        "mov.u32 %r_hidden_dim, {};\n",
        config.hidden_dim
    ));
    ptx.push_str(&format!("mov.u32 %r_out_dim, {};\n", config.out_dim));

    // gridDim.x indexes the activation row; gridDim.y tiles the output columns.
    ptx.push_str("mov.u32 %r_row_id, %ctaid.x;\n");
    ptx.push_str("mov.u32 %r_tid, %tid.x;\n");
    ptx.push_str("mov.u32 %r_nthreads, %ntid.x;\n");
    ptx.push_str("mov.u32 %r_col_id, %ctaid.y;\n");
    ptx.push_str("mul.lo.u32 %r_col_id, %r_col_id, %r_nthreads;\n");
    ptx.push_str("add.u32 %r_col_id, %r_col_id, %r_tid;\n");

    ptx.push_str("mov.u32 %r_qact_base, bitnet_qact;\n");
    ptx.push_str("mov.u32 %r_red_base, bitnet_red;\n");
    ptx.push_str("mov.u32 %r_scale_base, bitnet_scale;\n");
    // NOTE: the out-of-range column guard is deliberately NOT emitted here.
    // Every thread must reach the `bar.sync`s inside absmax_quant, so the
    // guard is emitted by ternary_gemm::emit AFTER the last barrier.
}
