use crate::kernel_skeleton::pad::emit_smem_zero_pad_predicated;

#[test]
fn smem_zero_pad_rank4_to_16_f16_test() {
    let mut ptx = String::new();
    emit_smem_zero_pad_predicated(&mut ptx, "%a_tile_base", 4, 16, 16);
    assert_eq!(
        ptx,
        include_str!("snapshots/smem_zero_pad__rank4_to_16_f16.snap"),
        "smem zero-pad (rank 4→16, f16) drift"
    );
}

#[test]
fn smem_zero_pad_rank16_to_16_f16_noop_test() {
    // No-op path: real_extent == padded_extent → zero instructions emitted.
    let mut ptx = String::new();
    emit_smem_zero_pad_predicated(&mut ptx, "%a_tile_base", 16, 16, 16);
    assert_eq!(
        ptx,
        include_str!("snapshots/smem_zero_pad__rank16_to_16_f16.snap"),
        "smem zero-pad no-op path drift — helper must emit empty string when real==padded"
    );
}

#[test]
#[should_panic(expected = "must be ≤ padded_extent")]
fn smem_zero_pad_real_gt_padded_panics_test() {
    let mut ptx = String::new();
    emit_smem_zero_pad_predicated(&mut ptx, "%a_tile_base", 20, 16, 16);
}
