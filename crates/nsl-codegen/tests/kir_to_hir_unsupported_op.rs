//! Test that the KIR → HIR pass errors on unsupported ops with the §2.5 contract.

use nsl_codegen::fpga_error::FpgaLoweringError;
use nsl_codegen::hir::KirToHirPass;
use nsl_codegen::kernel_ir::*;

/// Build a KernelIR from a list of ops.
fn kir_with_ops(name: &str, ops: Vec<KirOp>) -> KernelIR {
    let mut b = KirBuilder::new(name);
    let blk = b.new_block();
    b.set_block(blk);
    for op in ops {
        b.emit(op);
    }
    b.terminate(KirTerminator::Return);
    b.finalize()
}

#[test]
fn tanh_op_errors_with_supported_list() {
    let kir = kir_with_ops("tanh_test", vec![KirOp::Tanh(1, 2)]);

    let pass = KirToHirPass::new(true);
    let result = pass.lower(&kir, "tanh_test");

    let err = result.unwrap_err();
    match err {
        FpgaLoweringError::UnsupportedKirOp { op_kind, supported } => {
            assert_eq!(op_kind, "Tanh");
            assert!(supported.contains(&"Matmul"));
            assert!(supported.contains(&"ElementwiseAdd"));
            assert!(supported.contains(&"Relu"));
        }
        _ => panic!("expected UnsupportedKirOp error"),
    }
}

#[test]
fn pass_errors_at_first_unsupported_op_not_at_end() {
    let kir = kir_with_ops(
        "first_fail",
        vec![
            // Supported op first
            KirOp::Matmul {
                a: 1, b: 2, out: 3,
                a_dtype: KirType::I8, b_dtype: KirType::I8, out_dtype: KirType::I32,
                a_shape: [1, 784], b_shape: [784, 128],
            },
            // Unsupported op — should trigger error here
            KirOp::Sqrt(3, 4),
            // Also supported but never reached
            KirOp::Relu {
                a: 4, out: 5,
                dtype: KirType::I32, shape: [128],
            },
        ],
    );

    let pass = KirToHirPass::new(true);
    let err = pass.lower(&kir, "first_fail").unwrap_err();
    match err {
        FpgaLoweringError::UnsupportedKirOp { op_kind, .. } => {
            assert_eq!(op_kind, "Sqrt"); // first unsupported, not last
        }
        _ => panic!("expected UnsupportedKirOp"),
    }
}
