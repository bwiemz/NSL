use crate::kernel_skeleton::params::{
    emit_ld_param_f32, emit_ld_param_u32, emit_ld_param_u64, emit_param_block, Param, ParamTy,
};

#[test]
fn ld_param_u64_test() {
    let mut ptx = String::new();
    emit_ld_param_u64(&mut ptx, "%rd_x", "x_ptr");
    assert_eq!(
        ptx,
        include_str!("snapshots/ld_param_u64.snap"),
        "ld.param.u64 drift"
    );
}

#[test]
fn ld_param_u32_test() {
    let mut ptx = String::new();
    emit_ld_param_u32(&mut ptx, "%r10", "csha_active_heads");
    assert_eq!(
        ptx,
        include_str!("snapshots/ld_param_u32.snap"),
        "ld.param.u32 drift"
    );
}

#[test]
fn ld_param_f32_test() {
    let mut ptx = String::new();
    emit_ld_param_f32(&mut ptx, "%scale_reg", "scale");
    assert_eq!(
        ptx,
        include_str!("snapshots/ld_param_f32.snap"),
        "ld.param.f32 drift"
    );
}

#[test]
fn param_block_wrga_lora_test() {
    let mut ptx = String::new();
    let params = [
        Param {
            ty: ParamTy::U64,
            name: "x_ptr",
        },
        Param {
            ty: ParamTy::U64,
            name: "w_ptr",
        },
        Param {
            ty: ParamTy::U64,
            name: "a_ptr",
        },
        Param {
            ty: ParamTy::U64,
            name: "b_ptr",
        },
        Param {
            ty: ParamTy::F32,
            name: "scale",
        },
        Param {
            ty: ParamTy::U64,
            name: "y_ptr",
        },
    ];
    emit_param_block(&mut ptx, "nsl_wrga_fused_lora_test", &params);
    assert_eq!(
        ptx,
        include_str!("snapshots/param_block__wrga_lora.snap"),
        "WRGA LoRA param block drift"
    );
}
