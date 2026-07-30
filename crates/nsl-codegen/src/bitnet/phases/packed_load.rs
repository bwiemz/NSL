//! BitNet packed-ternary weight load + on-the-fly unpack.
//!
//! Spec: `docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md` §4.1.
//! Public phase emitter (callable from CSHA-fused mode in a future M35.x).
//!
//! Bit ordering: HIGH-BITS-FIRST per `PACKED_BYTE_LAYOUT.md` (PI.1).
//! trit[0] at bits [7:6]; trit[3] at bits [1:0]. Per-trit unpack must emit a
//! single SASS instruction (`BFE.U32` or equivalent) at sm_80+ — asserted by
//! `tests/bitnet_sass_discipline.rs`.
//!
//! ## Weight layout
//!
//! `weights_packed` is `[out_dim, hidden_dim]` row-major **in trits**, packed
//! 4 trits per byte along the `hidden_dim` (k) axis. That is the
//! safetensors-native HF `nn.Linear` order (`[out_features, in_features]`), so
//! `loader.rs` output is consumable with no transpose, and the four trits in a
//! byte are four *consecutive k* for one output column — which is what makes a
//! single `ld.global.u8` + four `bfe.u32` the natural inner step.
//!
//! ## Why there is no `cp.async` here any more
//!
//! The Phase 1 draft emitted a `cp.async` ping-pong that staged a single 4-byte
//! word into SMEM and then dropped the unpacked trits on the floor — nothing
//! consumed them, and the surrounding code read weights from an SMEM region no
//! one ever wrote. It also did not assemble. This emitter now streams weights
//! straight from HBM per thread. SMEM staging plus `cp.async` pipelining is a
//! performance follow-up, to be reintroduced together with real K/N tiling.

use crate::bitnet::config::BitNetKernelConfig;

/// Emit the packed-weight prologue: the phase marker and a note on layout.
///
/// Called once by `mod.rs::synthesize_kernel` before the quantization phase, so
/// that the `BitNet packed_load` marker precedes `BitNet absmax_quant` in the
/// emitted module (phase-ordering invariant asserted by
/// `tests/bitnet_orchestrator.rs`).
///
/// The global weight base pointer `%rd_w_packed` is already materialised by
/// `mod.rs::emit_prologue`; this emitter only documents the contract.
pub fn emit_prologue(ptx: &mut String, config: &BitNetKernelConfig) {
    let block_k = config.block_k;
    let block_n = config.block_n;
    ptx.push_str(&format!(
        "// === BitNet packed_load (block_k={block_k}, block_n={block_n}) ===\n"
    ));
    ptx.push_str("// weights_packed is [out_dim, hidden_dim] row-major in trits, 4 trits/byte\n");
    ptx.push_str("// along k, HIGH-BITS-FIRST. Base pointer %rd_w_packed set in the prologue.\n");
    ptx.push_str("// === end BitNet packed_load ===\n");
}

/// Emit the inner unpack step: load one packed byte and decode its 4 trits.
///
/// Reads:
/// - `%r_trit_index` — flat trit index (`col_id * hidden_dim + k`), 4-aligned.
/// - `%rd_w_packed`  — global base pointer to the packed weight bytes.
///
/// Writes:
/// - `%r_t0_val` … `%r_t3_val` — the four decoded trit values in `{-1, 0, +1}`,
///   corresponding to `k+0` … `k+3`.
///
/// Clobbers `%r_byte_offset`, `%rd_byte_off_64`, `%rd_weight_addr`,
/// `%r_packed_word`, `%r_trit0` … `%r_trit3`.
///
/// Emits exactly four `bfe.u32` and exactly one `shr.b32`, and no `and.b32`
/// (the SHR+AND fallback is sm_75 and earlier, out of v1 scope).
pub fn emit(ptx: &mut String, _config: &BitNetKernelConfig) {
    ptx.push_str("// === BitNet packed_load unpack ===\n");
    // Compute byte offset from trit offset: byte_offset = trit_offset / 4.
    ptx.push_str("// Compute byte offset = trit_offset >> 2 (4 trits per byte).\n");
    ptx.push_str("shr.b32 %r_byte_offset, %r_trit_index, 2;\n");
    ptx.push_str("cvt.u64.u32 %rd_byte_off_64, %r_byte_offset;\n");
    ptx.push_str("add.s64 %rd_weight_addr, %rd_w_packed, %rd_byte_off_64;\n");
    ptx.push_str("ld.global.u8 %r_packed_word, [%rd_weight_addr];\n");

    // HIGH-BITS-FIRST per PACKED_BYTE_LAYOUT.md:
    //   trit[0] at bits [7:6], trit[1] at [5:4], trit[2] at [3:2], trit[3] at [1:0].
    ptx.push_str("// Unpack 4 trits per byte using bit-field extract (single SASS at sm_80+).\n");
    ptx.push_str("// HIGH-BITS-FIRST: trit[0] at bits [7:6]; trit[3] at bits [1:0].\n");
    ptx.push_str("bfe.u32 %r_trit0, %r_packed_word, 6, 2;\n");
    ptx.push_str("bfe.u32 %r_trit1, %r_packed_word, 4, 2;\n");
    ptx.push_str("bfe.u32 %r_trit2, %r_packed_word, 2, 2;\n");
    ptx.push_str("bfe.u32 %r_trit3, %r_packed_word, 0, 2;\n");

    // Decode 2-bit → trit value: 00 → -1, 01 → 0, 10 → +1.
    // 0b11 is rejected upstream by the loader (try_unpack_byte).
    ptx.push_str("// Decode: trit_value = bits - 1 (00 -> -1, 01 -> 0, 10 -> +1).\n");
    ptx.push_str("sub.s32 %r_t0_val, %r_trit0, 1;\n");
    ptx.push_str("sub.s32 %r_t1_val, %r_trit1, 1;\n");
    ptx.push_str("sub.s32 %r_t2_val, %r_trit2, 1;\n");
    ptx.push_str("sub.s32 %r_t3_val, %r_trit3, 1;\n");

    ptx.push_str("// === end BitNet packed_load unpack ===\n");
}
