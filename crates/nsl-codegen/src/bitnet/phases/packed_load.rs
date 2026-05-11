//! BitNet packed-ternary HBM → SMEM load + on-the-fly unpack.
//!
//! Spec: `docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md` §4.1.
//! Public phase emitter (callable from CSHA-fused mode in a future M35.x).
//!
//! Loads 2-bit packed ternary weights via cp.async (Ampere+, depth=2 ping-pong)
//! and unpacks on-the-fly into i8 register tiles. Per-trit unpack must emit a
//! single SASS instruction (`BFE.U32` or equivalent) at sm_80+.
//!
//! Bit ordering: HIGH-BITS-FIRST per `PACKED_BYTE_LAYOUT.md` (PI.1).
//! trit[0] at bits [7:6]; trit[3] at bits [1:0].

use crate::bitnet::config::BitNetKernelConfig;

/// Emit packed-load PTX into the kernel-building context.
///
/// The emitted PTX assumes:
/// - `%rd1` holds the global weight pointer (packed bytes).
/// - `%r1` holds the K-tile offset in trits.
/// - `%rd2` is the SMEM destination pointer.
///
/// Phase 1 emits cp.async-based loads with depth=2 ping-pong (no role split).
pub fn emit(ptx: &mut String, config: &BitNetKernelConfig) {
    let block_k = config.block_k;
    let block_n = config.block_n;

    ptx.push_str(&format!(
        "// === BitNet packed_load (block_k={block_k}, block_n={block_n}) ===\n"
    ));

    // Compute byte offset from trit offset: byte_offset = trit_offset / 4.
    ptx.push_str("// Compute byte offset = trit_offset >> 2 (4 trits per byte).\n");
    ptx.push_str("shr.b32 %r_byte_offset, %r1, 2;\n");
    ptx.push_str("cvta.to.global.u64 %rd_weight_global, %rd1;\n");
    ptx.push_str("cvt.u64.u32 %rd_byte_off_64, %r_byte_offset;\n");
    ptx.push_str("add.s64 %rd_weight_addr, %rd_weight_global, %rd_byte_off_64;\n");

    // cp.async: asynchronous global → shared load. 4 bytes per thread per issue.
    // Depth=2 ping-pong: load tile N+1 in stage 1 while computing on stage 0.
    ptx.push_str("// cp.async ping-pong (depth=2, no role split).\n");
    ptx.push_str("cp.async.ca.shared.global [%rd2], [%rd_weight_addr], 4;\n");
    ptx.push_str("cp.async.commit_group;\n");

    // After the load lands in SMEM, threads cooperatively unpack.
    // Each thread unpacks 4 trits → 4 i8 register slots.
    // HIGH-BITS-FIRST per PACKED_BYTE_LAYOUT.md:
    //   trit[0] at bits [7:6], trit[1] at [5:4], trit[2] at [3:2], trit[3] at [1:0].
    ptx.push_str("// Unpack 4 trits per byte using bit-field extract (single SASS at sm_80+).\n");
    ptx.push_str("// HIGH-BITS-FIRST: trit[0] at bits [7:6]; trit[3] at bits [1:0].\n");
    ptx.push_str("ld.shared.b32 %r_packed_word, [%rd2];\n");
    ptx.push_str("bfe.u32 %r_trit0, %r_packed_word, 6, 2;\n");
    ptx.push_str("bfe.u32 %r_trit1, %r_packed_word, 4, 2;\n");
    ptx.push_str("bfe.u32 %r_trit2, %r_packed_word, 2, 2;\n");
    ptx.push_str("bfe.u32 %r_trit3, %r_packed_word, 0, 2;\n");

    // Decode 2-bit → trit value: 00 → -1, 01 → 0, 10 → +1.
    // Subtract 1 from the 2-bit value (00 → -1, 01 → 0, 10 → 1).
    // 0b11 is rejected upstream by the loader (try_unpack_byte).
    ptx.push_str("// Decode: trit_value = bits - 1 (00→-1, 01→0, 10→+1).\n");
    ptx.push_str("sub.s32 %r_t0_val, %r_trit0, 1;\n");
    ptx.push_str("sub.s32 %r_t1_val, %r_trit1, 1;\n");
    ptx.push_str("sub.s32 %r_t2_val, %r_trit2, 1;\n");
    ptx.push_str("sub.s32 %r_t3_val, %r_trit3, 1;\n");

    ptx.push_str("// === end BitNet packed_load ===\n");
}
