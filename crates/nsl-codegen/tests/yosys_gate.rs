//! Layer 3 — end-to-end Yosys gate test.
//!
//! Skipped if `yosys` isn't installed on the system. CI installs it via apt.

mod common;

use nsl_codegen::backend_verilog::lower::VerilogEmitter;
use nsl_codegen::backend_verilog::yosys::YosysGate;
use nsl_codegen::hir::KirToHirPass;
use nsl_codegen::kernel_ir::*;

fn v1_mlp_kir() -> KernelIR {
    common::kir_builder::kir_with_ops(
        "tiny_mlp",
        vec![
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
fn yosys_gate_v1_mlp_clean() {
    if !YosysGate::is_available() {
        eprintln!("SKIPPED: yosys not installed");
        return;
    }

    let kir = v1_mlp_kir();
    let pass = KirToHirPass::new(true);
    let module = pass.lower(&kir, "tiny_mlp").unwrap();
    let verilog = VerilogEmitter::emit_module(&module);

    let tmp_dir = std::env::temp_dir().join("nsl_fpga_test");
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let v_path = tmp_dir.join("tiny_mlp.v");
    let log_path = tmp_dir.join("yosys-full.log");
    std::fs::write(&v_path, &verilog).unwrap();

    let result = YosysGate::run(&v_path, &log_path);
    if let Err(e) = &result {
        eprintln!("yosys gate failed: {e}");
        eprintln!("emitted Verilog snippet (first 200 lines):\n{}",
                  verilog.lines().take(200).collect::<Vec<_>>().join("\n"));
    }
    result.expect("yosys gate must pass with zero warnings on v1 MLP");
}
