//! Layer 1 (a) — per-HIR-node Verilog template snapshots per spec §7.1.
//! Layer 1 (c) — v1 MLP structural-skeleton snapshot (localparam elided).

mod common;

use insta::assert_snapshot;
use nsl_codegen::backend_verilog::templates::*;
use nsl_codegen::hir::*;
use nsl_codegen::hir::ids::*;
use nsl_codegen::hir::nodes::*;

#[test]
fn snapshot_mul_template() {
    let m = Mul {
        a: SignalRef::port("x"), b: SignalRef::port("w"),
        out: WireId(1), a_width: 8, b_width: 8, out_width: 16,
    };
    assert_snapshot!(emit_mul(&m));
}

#[test]
fn snapshot_add_template() {
    let a = Add {
        a: SignalRef::port("acc"), b: SignalRef::port("prod"),
        out: WireId(1), width: 32,
    };
    assert_snapshot!(emit_add(&a));
}

#[test]
fn snapshot_max0_template() {
    let m = Max0 { a: SignalRef::wire(WireId(5)), out: WireId(6), width: 32 };
    assert_snapshot!(emit_max0(&m));
}

#[test]
fn snapshot_sign_extend_template_i16_to_i32() {
    let s = SignExtend {
        src: SignalRef::wire(WireId(3)), dst: WireId(4),
        src_width: 16, dst_width: 32,
    };
    assert_snapshot!(emit_sign_extend(&s));
}

#[test]
fn snapshot_local_param_i32() {
    let lp = LocalParam::new("W1_0_0", 32, -1234);
    assert_snapshot!(emit_local_param(&lp));
}

#[test]
fn snapshot_wire_i32() {
    let w = Wire { id: WireId(9), width: 32 };
    assert_snapshot!(emit_wire(&w));
}

// ---------------------------------------------------------------------------
// Layer 1 (c) — v1 MLP structural-skeleton snapshot
// ---------------------------------------------------------------------------

use nsl_codegen::backend_verilog::filter::elide_localparams;
use nsl_codegen::backend_verilog::lower::VerilogEmitter;
use nsl_codegen::kernel_ir::*;

fn v1_mlp_kir() -> KernelIR {
    common::kir_builder::kir_with_ops(
        "tiny_mlp",
        vec![
            // Layer 1: 784 → 128 (i8×i8→i32)
            KirOp::Matmul {
                a: 1, b: 2, out: 3,
                a_dtype: KirType::I8, b_dtype: KirType::I8, out_dtype: KirType::I32,
                a_shape: [1, 784], b_shape: [784, 128],
            },
            KirOp::ElementwiseAdd {
                a: 3, b: 4, out: 5,
                dtype: KirType::I32, shape: [128],
            },
            KirOp::Relu {
                a: 5, out: 6,
                dtype: KirType::I32, shape: [128],
            },
            // Layer 2: 128 → 10 (i32×i32→i64)
            KirOp::Matmul {
                a: 6, b: 7, out: 8,
                a_dtype: KirType::I32, b_dtype: KirType::I32, out_dtype: KirType::I64,
                a_shape: [1, 128], b_shape: [128, 10],
            },
            KirOp::ElementwiseAdd {
                a: 8, b: 9, out: 10,
                dtype: KirType::I64, shape: [10],
            },
            KirOp::Relu {
                a: 10, out: 11,
                dtype: KirType::I64, shape: [10],
            },
        ],
    )
}

#[test]
fn snapshot_v1_mlp_structural_skeleton() {
    let kir = v1_mlp_kir();
    let pass = KirToHirPass::new(true);
    let module = pass.lower(&kir, "tiny_mlp").unwrap();
    let verilog = VerilogEmitter::emit_module(&module);
    let skeleton = elide_localparams(&verilog);
    assert_snapshot!(skeleton);
}
