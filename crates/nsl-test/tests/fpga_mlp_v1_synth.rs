//! Layer 3 — Yosys synthesis gate for the v1 MLP, run on tap-enabled Verilog.
//!
//! Test is `#[ignore]`'d pending M57.1 (v1 closure).
//! Prerequisites identical to fpga_mlp_v1_parity.rs:
//!   1. AST->KIR dispatch for Matmul/Relu/ElementwiseAdd structured ops
//!   2. HIR port/wire declaration emission
//!   3. End-to-end `nsl fpga-compile` CLI dispatch
//!
//! Per spec §2.7, the synthesis gate runs on tap-enabled Verilog (the same
//! file Layer 2 simulates) to close the structural gap: if the tap ports
//! make the design unsynthesizable, Layer 3 catches it.
//!
//! See spec §7.3 and §5 for the zero-warnings-as-errors discipline.

use std::path::Path;
use std::process::Command;

use nsl_codegen::backend_verilog::yosys::YosysGate;

/// Layer 3 Yosys synthesis gate on tap-enabled v1 MLP Verilog.
///
/// Runs `yosys -p "read_verilog ...; synth -noabc -flatten; check"` with
/// warnings-as-errors per spec §5.
#[test]
#[ignore = "M57.1 v1 closure prerequisite: AST->KIR dispatch, HIR port/wire generation, \
            end-to-end CLI wiring (nsl fpga-compile), and Verilog synthesizability. \
            See spec §1.5 deferred-roadmap / crates/nsl-test/tests/fpga_mlp_v1_synth.rs header."]
fn fpga_mlp_v1_yosys_clean() {
    if !YosysGate::is_available() {
        eprintln!("SKIPPED: yosys not installed");
        return;
    }

    // Compile NSL source to tap-enabled Verilog
    let output_dir = Path::new("target/fpga/test");
    std::fs::create_dir_all(output_dir).unwrap();

    let status = Command::new("cargo")
        .args([
            "run",
            "-p",
            "nsl-cli",
            "--release",
            "--",
            "fpga-compile",
            "crates/nsl-test/fixtures/v1_mlp.nsl",
            "--fixture",
            "crates/nsl-test/fixtures/mlp_int8_weights_v1.bin",
            "--output-dir",
            output_dir.to_str().unwrap(),
            "--test-taps", // tap-enabled per spec §2.7
        ])
        .status()
        .unwrap();
    assert!(status.success(), "nsl fpga-compile failed");

    let v_path = output_dir.join("tiny_mlp.v");
    let log_path = output_dir.join("yosys-full.log");

    let result = YosysGate::run(&v_path, &log_path);
    if let Err(ref e) = result {
        eprintln!("Yosys gate failed: {e}");
    }
    result.expect(
        "Yosys gate must pass with zero warnings on tap-enabled v1 MLP \
         (warnings-as-errors per spec §5)",
    );
}
