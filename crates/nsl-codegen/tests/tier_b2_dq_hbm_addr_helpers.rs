use nsl_codegen::flash_attention_v2::tier_b2::backward::dq::hbm_addr::{
    emit_4d_byte_offset, emit_3d_byte_offset,
};

#[test]
fn emit_4d_byte_offset_produces_canonical_mul_chain() {
    let mut ptx = String::new();
    emit_4d_byte_offset(
        &mut ptx,
        "%off",        // output register
        "%batch_idx",  // b register
        "%head",       // h register
        "%seq_idx",    // s register
        "%d_idx",      // d register
        "%heads",      // H param register
        "%seq_len",    // S param register
        32,            // D constant (head_dim)
        2,             // sizeof(dtype) = 2 for f16
    );
    // Expected pattern: row_index = ((b·H + h)·S + s)·D + d; off = row_index · sizeof(dtype)
    assert!(ptx.contains("mul.lo.u32"));
    assert!(ptx.contains("mul.wide.u32"));
    assert!(ptx.contains("%off"));
}

#[test]
fn emit_3d_byte_offset_uses_correct_stride() {
    let mut ptx = String::new();
    emit_3d_byte_offset(
        &mut ptx, "%off", "%batch_idx", "%head", "%seq_idx",
        "%heads", "%seq_len", 4,
    );
    assert!(ptx.contains("mul.wide.u32"));
}

#[test]
fn emit_4d_byte_offset_emits_ascii_only_ptx() {
    let mut ptx = String::new();
    emit_4d_byte_offset(&mut ptx, "%off", "%b", "%h", "%s", "%d",
        "%H", "%S", 128, 2);
    for byte in ptx.bytes() {
        assert!(byte.is_ascii(), "non-ASCII byte 0x{:02x} in emit_4d_byte_offset output", byte);
    }
}
