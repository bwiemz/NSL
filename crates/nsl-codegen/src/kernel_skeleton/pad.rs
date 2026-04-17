//! Predicated SMEM zero-padding for sub-MMA tile regions.
//!
//! Used by WRGA to zero the [real_extent, padded_extent) slice of an
//! SMEM tile when the real rank/m/k is below the MMA tile size.
//! Designed to be adoptable by FA for head_dim edge-case padding later.

/// Emit a SMEM zero-store loop that clears the slice
/// `[real_extent, padded_extent)` of an SMEM region.
///
/// When `real_extent == padded_extent`, emits zero PTX instructions (no-op).
/// This pure-pass-through is a tested invariant — do not optimize away
/// as `return` or panic, since snapshot tests assert the empty output.
///
/// # Arguments
/// * `smem_base_reg` — register holding the SMEM base for this region,
///   e.g. `"%a_tile_base"`.  Caller must have initialized this register.
/// * `real_extent` — real size of the data dimension (e.g. rank = 4)
/// * `padded_extent` — padded size (e.g. MMA k = 16)
/// * `dtype_bits` — 16 for f16, 32 for f32; controls `st.shared.b{bits}`
///
/// # Panics
/// Panics if `real_extent > padded_extent`.
pub fn emit_smem_zero_pad_predicated(
    ptx: &mut String,
    smem_base_reg: &str,
    real_extent: u32,
    padded_extent: u32,
    dtype_bits: u32,
) {
    assert!(
        real_extent <= padded_extent,
        "real_extent ({}) must be ≤ padded_extent ({})",
        real_extent,
        padded_extent,
    );
    if real_extent == padded_extent {
        // No-op path — locked in by smem_zero_pad__rank16_to_16_f16.snap (empty file).
        return;
    }
    ptx.push_str(&format!(
        "    // Zero-pad SMEM region [{}, {}) at {}-bit granularity\n",
        real_extent, padded_extent, dtype_bits,
    ));
    let bytes_per_elem = dtype_bits / 8;
    for i in real_extent..padded_extent {
        let offset = i * bytes_per_elem;
        ptx.push_str(&format!(
            "    st.shared.b{} [{} + {}], 0;\n",
            dtype_bits, smem_base_reg, offset,
        ));
    }
}
