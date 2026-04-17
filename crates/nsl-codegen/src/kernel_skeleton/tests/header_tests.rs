use crate::kernel_skeleton::header::{emit_ptx_header, PtxVersion, TargetSm};

#[test]
fn header_v87_sm75() {
    let mut ptx = String::new();
    emit_ptx_header(&mut ptx, PtxVersion::V8_7, TargetSm::Sm75);
    let expected = include_str!("snapshots/header__v87_sm75.snap");
    assert_eq!(ptx, expected, "FA-compatible header drift");
}

#[test]
fn header_v70_sm80() {
    let mut ptx = String::new();
    emit_ptx_header(&mut ptx, PtxVersion::V7_0, TargetSm::Sm80);
    let expected = include_str!("snapshots/header__v70_sm80.snap");
    assert_eq!(ptx, expected, "WRGA-compatible header drift");
}
