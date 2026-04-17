use crate::kernel_skeleton::smem::{
    emit_dynamic_smem_extern, emit_shmem_base_cvta, emit_static_smem_decl,
};

#[test]
fn static_smem_decl_1536_bytes() {
    let mut ptx = String::new();
    emit_static_smem_decl(&mut ptx, 1536);
    assert_eq!(
        ptx,
        include_str!("snapshots/static_smem_decl__1536_bytes.snap"),
        "WRGA LoRA SMEM decl drift"
    );
}

#[test]
fn static_smem_decl_768_bytes() {
    let mut ptx = String::new();
    emit_static_smem_decl(&mut ptx, 768);
    assert_eq!(
        ptx,
        include_str!("snapshots/static_smem_decl__768_bytes.snap"),
        "WRGA IA3 SMEM decl drift"
    );
}

#[test]
fn dynamic_smem_extern_test() {
    let mut ptx = String::new();
    emit_dynamic_smem_extern(&mut ptx);
    assert_eq!(
        ptx,
        include_str!("snapshots/dynamic_smem_extern.snap"),
        "FA dynamic SMEM extern drift"
    );
}

#[test]
fn shmem_base_cvta_test() {
    let mut ptx = String::new();
    emit_shmem_base_cvta(&mut ptx);
    assert_eq!(
        ptx,
        include_str!("snapshots/shmem_base_cvta.snap"),
        "shmem base cvta drift"
    );
}
