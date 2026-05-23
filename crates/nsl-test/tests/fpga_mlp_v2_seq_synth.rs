//! Layer 3 — Yosys synthesis gate for the v2 sequential FSM (M57.2).
//!
//! ## Purpose
//!
//! Synthesizes `tiny_mlp_seq.v` (the `always_ff`-based sequential FSM emitted
//! by `nsl fpga-compile --seq`) with Yosys and asserts:
//!
//!   - Zero Yosys warnings or errors (warnings-as-errors per spec §5).
//!   - No implicit nets (would indicate undriven port or wiring error).
//!   - No inferred latches (the design uses `always_ff` not `always_comb`;
//!     any latch inference means a missing `else` branch or unreset register).
//!   - All registers are reset (including `x_buf` and `h_buf` per-element
//!     pipeline registers; failure here → flip-flop may hold stale state
//!     across inference calls).
//!
//! ## Graceful skip policy
//!
//! When `YosysGate::is_available()` returns false this test prints
//! "SKIPPED: yosys not installed" to stderr and returns successfully.
//! No Yosys invocation or assertion is attempted.

use std::path::Path;
use std::process::Command;

use nsl_codegen::backend_verilog::yosys::YosysGate;

const BIN_PATH: &str = "crates/nsl-test/fixtures/mlp_int8_weights_v1.bin";
const NSL_PATH: &str = "crates/nsl-test/fixtures/v1_mlp.nsl";

/// Layer 3 Yosys synthesis gate on the v2 sequential FSM.
///
/// Generates `tiny_mlp_seq.v` with a baked fixture (so all `localparam` weight
/// constants are non-zero), then runs `yosys -p "read_verilog -sv ...; synth
/// -noabc -flatten; check"` with warnings-as-errors per spec §5.
///
/// Asserts: zero warnings/errors, no implicit nets, no inferred latches, all
/// registers reset (covering `x_buf`, `h_buf` pipeline regs per M57.2 design).
///
/// Gracefully skips when Yosys is not installed.
#[test]
fn fpga_mlp_v2_seq_yosys_clean() {
    if !YosysGate::is_available() {
        eprintln!("SKIPPED: yosys not installed");
        return;
    }

    // -----------------------------------------------------------------------
    // Compile sequential Verilog with baked fixture so localparam weights are
    // non-zero (mirrors how the combinational synth test prepares the .v)
    // -----------------------------------------------------------------------
    let output_dir = Path::new("target/fpga/test-seq-synth");
    std::fs::create_dir_all(output_dir).unwrap();

    let out = Command::new("cargo")
        .args([
            "run",
            "-p",
            "nsl-cli",
            "--release",
            "--",
            "fpga-compile",
            NSL_PATH,
            "--fixture",
            BIN_PATH,
            "--output-dir",
            output_dir.to_str().unwrap(),
            "--test-taps", // consistent with spec §2.7; seq module ignores taps but flag is harmless
            "--seq",       // emit tiny_mlp_seq.v
        ])
        .output()
        .expect("cargo run nsl-cli failed to spawn");

    assert!(
        out.status.success(),
        "nsl fpga-compile --seq failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );

    let v_path   = output_dir.join("tiny_mlp_seq.v");
    let log_path = output_dir.join("yosys-seq.log");

    // -----------------------------------------------------------------------
    // Run the Yosys gate (warnings-as-errors, zero-warnings policy per §5)
    // -----------------------------------------------------------------------
    let result = YosysGate::run(&v_path, &log_path);
    if let Err(ref e) = result {
        eprintln!("Yosys sequential gate failed: {e}");
    }
    result.expect(
        "Yosys gate must pass with zero warnings on the sequential v2 MLP FSM \
         (warnings-as-errors per spec §5). \
         Expected: no implicit nets, no inferred latches (always_ff design), \
         all registers reset including x_buf/h_buf pipeline regs.",
    );
}
