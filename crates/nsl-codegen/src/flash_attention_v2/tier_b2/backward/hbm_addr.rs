//! HBM byte-offset PTX emission helpers for the dQ-kernel.
//!
//! Emits the mul/add chain to compute byte offsets for row-major
//! [B, H, S, D] (4D) and [B, H, S] (3D) tensors per Phase 2.5 spec ss2.1.
//!
//! - 4D formula: byte_offset = (((b*H + h)*S + s)*D + d) * sizeof(dtype)
//! - 3D formula: byte_offset = ((b*H + h)*S + s) * sizeof(dtype)
//!
//! D is a compile-time constant; sizeof(dtype) is compile-time (f16=2, f32=4).
//!
//! Callers must declare a `%row_index_tmp` u32 scratch register before invoking.
//!
//! GQA zero-copy stride pattern (paper §4.2): when `gqa_group_size > 1` the
//! K and V tensors share `head_idx` slots across Q-heads — for K/V address
//! computations the head register must be divided by `gqa_group_size` before
//! the stride formula is applied. The `emit_kv_head_divisor` helper emits
//! the compile-time-literal `div.u32` and returns the kv-head register name;
//! when `gqa_group_size == 1` it is a no-op that returns the Q-head register
//! unchanged, preserving byte-identical PTX for the non-GQA path.

/// Emit PTX that computes a 4D byte offset into `out_reg` (u64).
///
/// Row-major [B, H, S, D] byte offset formula:
///   byte_offset = (((b*H + h)*S + s)*D + d) * sizeof(dtype)
///
/// Caller declares `.reg .u32 %row_index_tmp` before calling.
/// `out_reg` is the u64 destination; widening from u32 happens via `mul.wide.u32`.
pub fn emit_4d_byte_offset(
    ptx: &mut String,
    out_reg: &str,    // u64
    b_reg: &str,      // u32
    h_reg: &str,      // u32
    s_reg: &str,      // u32
    d_reg: &str,      // u32 (pass literal "0" string for d=0)
    heads_reg: &str,  // u32 (H param)
    seq_reg: &str,    // u32 (S param)
    hd_const: u32,    // D (compile-time constant)
    sizeof_dtype: u32,// 2 for f16, 4 for f32
) {
    // row_index = (((b*H + h)*S + s)*D + d) (all u32)
    ptx.push_str(&format!("    mul.lo.u32 %row_index_tmp, {b_reg}, {heads_reg};\n"));
    ptx.push_str(&format!("    add.u32    %row_index_tmp, %row_index_tmp, {h_reg};\n"));
    ptx.push_str(&format!("    mul.lo.u32 %row_index_tmp, %row_index_tmp, {seq_reg};\n"));
    ptx.push_str(&format!("    add.u32    %row_index_tmp, %row_index_tmp, {s_reg};\n"));
    ptx.push_str(&format!("    mul.lo.u32 %row_index_tmp, %row_index_tmp, {hd_const};\n"));
    ptx.push_str(&format!("    add.u32    %row_index_tmp, %row_index_tmp, {d_reg};\n"));
    // byte offset = row_index * sizeof(dtype), widened to u64
    ptx.push_str(&format!(
        "    mul.wide.u32 {out_reg}, %row_index_tmp, {sizeof_dtype};\n"
    ));
}

/// Emit PTX that computes a 3D byte offset into `out_reg` (u64).
///
/// Row-major [B, H, S] byte offset formula:
///   byte_offset = ((b*H + h)*S + s) * sizeof(dtype)
///
/// Same as `emit_4d_byte_offset` but without the D dimension.
/// Caller declares `.reg .u32 %row_index_tmp` before calling.
pub fn emit_3d_byte_offset(
    ptx: &mut String,
    out_reg: &str,
    b_reg: &str,
    h_reg: &str,
    s_reg: &str,
    heads_reg: &str,
    seq_reg: &str,
    sizeof_dtype: u32,
) {
    ptx.push_str(&format!("    mul.lo.u32 %row_index_tmp, {b_reg}, {heads_reg};\n"));
    ptx.push_str(&format!("    add.u32    %row_index_tmp, %row_index_tmp, {h_reg};\n"));
    ptx.push_str(&format!("    mul.lo.u32 %row_index_tmp, %row_index_tmp, {seq_reg};\n"));
    ptx.push_str(&format!("    add.u32    %row_index_tmp, %row_index_tmp, {s_reg};\n"));
    ptx.push_str(&format!(
        "    mul.wide.u32 {out_reg}, %row_index_tmp, {sizeof_dtype};\n"
    ));
}

/// Emit GQA kv-head divisor as a compile-time literal `div.u32`.
///
/// When `gqa_group_size == 1` this is a no-op and returns `q_head_reg`
/// unchanged so that callers' subsequent `emit_4d_byte_offset(..., kv_reg, ...)`
/// passes the same register as before — byte-identical PTX is preserved for
/// the non-GQA path.
///
/// When `gqa_group_size > 1` this emits
///   `div.u32 %dst, %q_head_reg, {gqa_group_size};`
/// and returns the destination register name `dst_reg`.
///
/// Paper §4.2: zero-copy GQA expansion — the same KV data is addressed with
/// a different thread-to-output mapping via integer-division on head index.
/// `dst_reg` MUST be a caller-pre-declared u32 scratch register; callers are
/// expected to add the `.reg .u32 %xxx_kv_head;` decl alongside their other
/// register declarations (this helper assumes the decl already exists).
pub fn emit_kv_head_divisor<'a>(
    ptx: &mut String,
    q_head_reg: &'a str,
    dst_reg: &'a str,
    gqa_group_size: u32,
) -> &'a str {
    if gqa_group_size <= 1 {
        // Byte-identical no-op path: caller continues to use the Q-head register.
        q_head_reg
    } else {
        ptx.push_str(&format!(
            "    // GQA: kv_head = q_head / {gqa_group_size} (paper s4.2 zero-copy)\n"
        ));
        ptx.push_str(&format!(
            "    div.u32 {dst_reg}, {q_head_reg}, {gqa_group_size};\n"
        ));
        dst_reg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emit_kv_head_divisor_noop_when_gqa_one() {
        let mut ptx = String::new();
        let out = emit_kv_head_divisor(&mut ptx, "%head", "%kv_head", 1);
        assert_eq!(out, "%head", "gqa=1 must return the q_head register");
        assert!(ptx.is_empty(), "gqa=1 must emit no PTX (byte-identity)");
    }

    #[test]
    fn emit_kv_head_divisor_emits_div_when_gqa_gt_one() {
        let mut ptx = String::new();
        let out = emit_kv_head_divisor(&mut ptx, "%head", "%kv_head", 4);
        assert_eq!(out, "%kv_head", "gqa>1 must return the kv_head register");
        assert!(ptx.contains("div.u32 %kv_head, %head, 4;"),
            "expected `div.u32 %kv_head, %head, 4;` in PTX, got:\n{}", ptx);
    }

    #[test]
    fn emit_4d_byte_offset_canonical() {
        let mut ptx = String::new();
        emit_4d_byte_offset(&mut ptx, "%off", "%b", "%h", "%s", "%d", "%H", "%S", 32, 2);
        assert!(ptx.contains("mul.lo.u32"));
        assert!(ptx.contains("mul.wide.u32 %off"));
        assert!(ptx.contains(", 32;"), "expected hd constant 32");
        assert!(ptx.contains(", 2;"), "expected sizeof f16 = 2");
    }

    #[test]
    fn emit_3d_byte_offset_no_hd_term() {
        let mut ptx = String::new();
        emit_3d_byte_offset(&mut ptx, "%off", "%b", "%h", "%s", "%H", "%S", 4);
        assert!(ptx.contains("mul.wide.u32 %off"));
        assert!(ptx.contains(", 4;"), "expected sizeof f32 = 4");
        // 3D should have exactly 2 mul.lo.u32 (for *H and *S), no *D step
        let mul_lo_count = ptx.matches("mul.lo.u32").count();
        assert_eq!(mul_lo_count, 2, "3D should emit exactly 2 mul.lo.u32 (*H, *S)");
    }
}
