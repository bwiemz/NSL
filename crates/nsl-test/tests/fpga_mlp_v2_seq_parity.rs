//! Layer 2 — Verilator clocked-parity test for the v2 sequential FSM (M57.2).
//!
//! ## Purpose
//!
//! This test realizes the SQ-9 anti-staleness intent: it regenerates BOTH
//! `tiny_mlp_seq.v` (sequential FSM) and `tiny_mlp.v` (combinational v1) fresh
//! in-test, runs both under Verilator, and performs three assertions per
//! stimulus:
//!
//!   1. **Bit-exact CPU** — `v2.taps["out"]` decoded as 10 × i64 matches the
//!      CPU reference (`cpu_reference_layer2_relu` of the full v1 forward pass).
//!   2. **v1 ↔ v2 cross-check (SQ-9)** — the `out` port byte-exact matches the
//!      combinational v1 harness output for the same stimulus.
//!   3. **Cycle-count cross-check** — `v2.measured_cycles` equals the
//!      `total_cycles=` emitted by the CLI `--seq` code generator.
//!
//! ## Documented deviation from a literal SQ-9 dual-module testbench
//!
//! SQ-9 in the spec imagines a single Verilog testbench that instantiates both
//! `tiny_mlp` and `tiny_mlp_seq` side-by-side and asserts their outputs match
//! cycle-by-cycle. That approach is not used here for the following reasons:
//!
//!   a. A single mixed testbench would need hand-written Verilog glue that could
//!      silently paper over a naming divergence (e.g., an SQ-9 false-negative
//!      if the glue drives the wrong port).
//!   b. This Rust-level cross-check achieves the same goal: both `.v` files are
//!      generated fresh within the test (not cached artifacts), both are built
//!      into separate Verilator simulations, and their final-output bytes are
//!      compared in Rust.  A v1/v2 datapath divergence still causes a localized
//!      failure pointing to the sequential FSM scheduler as the culprit.
//!
//! Failure localization guide (printed on first failure):
//!   - `v2 != v1` → FSM scheduler bug (sequential pipeline produces wrong value
//!     that combinational design gets right).
//!   - `v2 == v1 != CPU` → datapath/CPU-reference bug (both hardware designs
//!     agree but diverge from the bit-exact reference).
//!   - `measured_cycles != emitted_total_cycles` → cycle-count mismatch between
//!     the code generator's declared cycle budget and the FSM's actual run.
//!
//! ## Skip policy
//!
//! When `VerilatorHarness::is_available()` returns false this test prints
//! "SKIPPED: verilator not installed" to stderr and returns successfully.
//! No assertions are evaluated and the test is not marked as a failure.

use std::path::Path;
use std::process::Command;

use nsl_test::cpu_reference::*;
use nsl_test::fixture::{parse, verify_hash};
use nsl_test::fpga_harness::{TapDescriptor, VerilatorHarness};
use nsl_test::stimuli::v1_mlp_stimuli;

const BIN_PATH: &str = "crates/nsl-test/fixtures/mlp_int8_weights_v1.bin";
const TOML_PATH: &str = "crates/nsl-test/fixtures/mlp_int8_weights_v1.toml";
const NSL_PATH: &str = "crates/nsl-test/fixtures/v1_mlp.nsl";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse `total_cycles=<n>` from CLI stdout.
///
/// The `--seq` codegen prints "total_cycles=<integer>" to stdout.
/// Returns `None` if the line is absent (unexpected; CLI contract violation).
fn parse_total_cycles(stdout: &str) -> Option<u64> {
    for line in stdout.lines() {
        if let Some(rest) = line.trim().strip_prefix("total_cycles=") {
            if let Ok(n) = rest.trim().parse::<u64>() {
                return Some(n);
            }
        }
    }
    None
}

/// Compile the NSL source to Verilog via `cargo run -p nsl-cli -- fpga-compile`.
///
/// `seq` controls whether `--seq` is appended (sequential module).
/// Returns `(output_path, cli_stdout)`.
fn compile_to_verilog(
    output_dir: &Path,
    seq: bool,
) -> (std::path::PathBuf, String) {
    let mut args = vec![
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
        "--test-taps",
    ];
    if seq {
        args.push("--seq");
    }
    let out = Command::new("cargo")
        .args(&args)
        .output()
        .expect("cargo run nsl-cli failed to spawn");
    assert!(
        out.status.success(),
        "nsl fpga-compile{} failed:\nstdout: {}\nstderr: {}",
        if seq { " --seq" } else { "" },
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let cli_stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let v_name = if seq { "tiny_mlp_seq.v" } else { "tiny_mlp.v" };
    (output_dir.join(v_name), cli_stdout)
}

// ---------------------------------------------------------------------------
// Clocked parity test
// ---------------------------------------------------------------------------

/// Layer 2 clocked-parity test (triple-check, SQ-9) — first-failure-stops.
///
/// Regenerates both `tiny_mlp_seq.v` (sequential, `--seq`) and `tiny_mlp.v`
/// (combinational, no `--seq`) fresh, builds Verilator harnesses for each,
/// then for all 100 deterministic stimuli asserts:
///
///   1. v2 `out` == CPU reference (bit-exact, 10 × i64).
///   2. v2 `out` == v1 `out` (SQ-9 cross-check; locates FSM-vs-datapath bugs).
///   3. `v2.measured_cycles` == `total_cycles` emitted by `--seq` CLI.
///
/// Gracefully skips when Verilator is not installed.
#[test]
fn fpga_mlp_v2_seq_parity() {
    if !VerilatorHarness::is_available() {
        eprintln!("SKIPPED: verilator not installed");
        return;
    }

    // -----------------------------------------------------------------------
    // Load and verify fixture (same fixture as combinational test)
    // -----------------------------------------------------------------------
    let bytes = std::fs::read(BIN_PATH).unwrap();
    let manifest_str = std::fs::read_to_string(TOML_PATH).unwrap();
    let manifest: toml::Value = toml::from_str(&manifest_str).unwrap();
    let expected_hash = manifest
        .get("sha256")
        .and_then(|v| v.as_str())
        .expect("sha256 field in manifest");
    verify_hash(&bytes, expected_hash).expect("fixture hash check");
    let fixture = parse(&bytes).expect("fixture parse");

    let w1 = fixture.blocks[0].as_i8();
    let b1 = fixture.blocks[1].as_i32();
    let w2 = fixture.blocks[2].as_i8();
    let b2 = fixture.blocks[3].as_i64();

    // -----------------------------------------------------------------------
    // Compile sequential Verilog + extract emitted_total_cycles
    // -----------------------------------------------------------------------
    let seq_dir = Path::new("target/fpga/test-seq");
    std::fs::create_dir_all(seq_dir).unwrap();
    let (seq_v_path, cli_stdout) = compile_to_verilog(seq_dir, true);

    let emitted_total_cycles = parse_total_cycles(&cli_stdout).unwrap_or_else(|| {
        panic!(
            "CLI --seq did not emit 'total_cycles=<n>' line.\nCLI stdout was:\n{cli_stdout}"
        )
    });

    // -----------------------------------------------------------------------
    // Compile combinational Verilog (for v1 ↔ v2 cross-check, SQ-9)
    // -----------------------------------------------------------------------
    let comb_dir = Path::new("target/fpga/test-comb");
    std::fs::create_dir_all(comb_dir).unwrap();
    let (comb_v_path, _) = compile_to_verilog(comb_dir, false);

    // -----------------------------------------------------------------------
    // Build Verilator harnesses
    // -----------------------------------------------------------------------
    let harness_seq = VerilatorHarness::build_clocked(
        &seq_v_path,
        "tiny_mlp_seq",
        TapDescriptor::v1_mlp_seq(),
    )
    .expect("verilator clocked harness build");

    let harness_comb = VerilatorHarness::build(
        &comb_v_path,
        "tiny_mlp",
        TapDescriptor::v1_mlp(),
    )
    .expect("verilator combinational harness build");

    // -----------------------------------------------------------------------
    // Per-stimulus triple-check: first-failure-stops
    // -----------------------------------------------------------------------
    let stimuli = v1_mlp_stimuli();
    for (stim_idx, x) in stimuli.iter().enumerate() {
        // --- CPU reference (reusing unmodified combinational-test functions) ---
        let cpu_l1_matmul = cpu_reference_layer1_matmul_with_bias(x, &w1, &b1);
        let cpu_l1_relu   = cpu_reference_layer1_relu(&cpu_l1_matmul);
        let cpu_l2_matmul = cpu_reference_layer2_matmul_with_bias(&cpu_l1_relu, &w2, &b2);
        let cpu_out:  Vec<i64> = cpu_reference_layer2_relu(&cpu_l2_matmul);

        // --- Run v2 sequential harness ---
        let v2 = harness_seq
            .run_clocked(x)
            .unwrap_or_else(|e| {
                panic!("stimulus #{stim_idx}: sequential simulator error: {e}")
            });
        let v2_out = v2.taps.tap_i64("out", 10);

        // --- Assertion 1: bit-exact CPU ---
        for (i, (&hw, &cpu)) in v2_out.iter().zip(cpu_out.iter()).enumerate() {
            if hw != cpu {
                panic!(
                    "Assertion 1 (v2 bit-exact CPU) FAILED:\n  \
                     stimulus #{stim_idx}, element [{i}]\n  \
                     v2 hw     = {hw}  (0x{hw:016x})\n  \
                     cpu ref   = {cpu}  (0x{cpu:016x})\n  \
                     Diagnosis: datapath/CPU-reference bug (both v1 and v2 hardware \
                     agree, but diverge from the reference, OR the sequential \
                     datapath has a standalone bug)."
                );
            }
        }

        // --- Run v1 combinational harness (SQ-9 cross-check) ---
        let v1 = harness_comb
            .run(x)
            .unwrap_or_else(|e| {
                panic!("stimulus #{stim_idx}: combinational simulator error: {e}")
            });
        let v1_out = v1.tap_i64("out", 10);

        // --- Assertion 2: v1 ↔ v2 cross-check (SQ-9) ---
        for (i, (&hw2, &hw1)) in v2_out.iter().zip(v1_out.iter()).enumerate() {
            if hw2 != hw1 {
                panic!(
                    "Assertion 2 (SQ-9 v1↔v2 cross-check) FAILED:\n  \
                     stimulus #{stim_idx}, element [{i}]\n  \
                     v2 (seq)  = {hw2}  (0x{hw2:016x})\n  \
                     v1 (comb) = {hw1}  (0x{hw1:016x})\n  \
                     Diagnosis: FSM scheduler bug — the sequential design reaches \
                     a different final output than the combinational design for the \
                     same input. Both agree with the CPU reference after the v2 \
                     datapath is fixed."
                );
            }
        }

        // --- Assertion 3: cycle-count cross-check ---
        if v2.measured_cycles != emitted_total_cycles {
            panic!(
                "Assertion 3 (cycle-count cross-check) FAILED:\n  \
                 stimulus #{stim_idx}\n  \
                 measured_cycles        = {}\n  \
                 emitted_total_cycles   = {emitted_total_cycles}\n  \
                 Diagnosis: the sequential FSM ran for a different number of cycles \
                 than the code generator declared in 'total_cycles=<n>'. This \
                 indicates a code-generator / FSM scheduling mismatch.",
                v2.measured_cycles
            );
        }
    }
}
