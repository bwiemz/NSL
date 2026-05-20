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

#[cfg(test)]
mod tests {
    use super::*;

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
