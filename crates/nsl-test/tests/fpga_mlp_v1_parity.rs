//! Layer 2 — Verilator parity test for v1 MLP per M57 spec §4.5 + §7.2.
//!
//! Both tests are `#[ignore]`'d pending M57.1 (v1 closure).
//! Prerequisites that must land before enabling these tests:
//!
//!   1. AST -> structured KIR dispatch: NSL frontend must lower `@` matmul
//!      and `relu()` to KirOp::Matmul / KirOp::Relu when target = Fpga.
//!   2. HIR port/wire declarations: GenerateFor loops need matching Port::Input
//!      declarations for `__matmul_a`, `__matmul_b` signal refs.
//!   3. MAC-chain genvar indexing: `acc[i] = acc[i-1] + product` needs
//!      correct GenerateFor loop variable threading.
//!   4. `nsl fpga-compile` CLI dispatch must call the HIR→Verilog pipeline
//!      instead of returning the deferred error.
//!
//! See spec §1.5 deferred-roadmap table for the M57.1 scope boundary.

use std::path::Path;
use std::process::Command;

use nsl_test::cpu_reference::*;
use nsl_test::fixture::{parse, verify_hash};
use nsl_test::fpga_harness::{TapDescriptor, VerilatorHarness};
use nsl_test::stimuli::v1_mlp_stimuli;

fn assert_bit_exact<T>(hw: &[T], cpu: &[T], stim_idx: usize, op_label: &str)
where
    T: PartialEq + std::fmt::Debug + std::fmt::LowerHex + Copy,
{
    for (i, (h, c)) in hw.iter().zip(cpu).enumerate() {
        if h != c {
            panic!(
                "Layer 2 parity mismatch:\n  \
                 stimulus #{stim_idx}, op `{op_label}`, element [{i}]\n  \
                 hw        = {h:?}  (0x{h:x})\n  \
                 reference = {c:?}  (0x{c:x})",
            );
        }
    }
}

/// Layer 2 parity test — first-failure-stops.
///
/// Compiles v1_mlp.nsl to Verilog via `nsl fpga-compile`, builds a
/// Verilator simulation, and asserts bit-exact per-op and final-output
/// match across all 100 deterministic stimuli.
#[test]
#[ignore = "M57.1 v1 closure prerequisite: AST->KIR dispatch, HIR port/wire generation, \
            end-to-end CLI wiring (nsl fpga-compile), and Verilog synthesizability. \
            See spec §1.5 deferred-roadmap / crates/nsl-test/tests/fpga_mlp_v1_parity.rs header."]
fn fpga_mlp_v1_parity() {
    if !VerilatorHarness::is_available() {
        eprintln!("SKIPPED: verilator not installed");
        return;
    }

    // -----------------------------------------------------------------------
    // Load and verify fixture
    // -----------------------------------------------------------------------
    let bin_path = "crates/nsl-test/fixtures/mlp_int8_weights_v1.bin";
    let toml_path = "crates/nsl-test/fixtures/mlp_int8_weights_v1.toml";

    let bytes = std::fs::read(bin_path).unwrap();
    let manifest_str = std::fs::read_to_string(toml_path).unwrap();
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
    // Compile NSL source to Verilog via `nsl fpga-compile`
    // -----------------------------------------------------------------------
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
            bin_path,
            "--output-dir",
            output_dir.to_str().unwrap(),
            "--test-taps",
        ])
        .status()
        .unwrap();
    assert!(status.success(), "nsl fpga-compile failed");

    // -----------------------------------------------------------------------
    // Build Verilator harness
    // -----------------------------------------------------------------------
    let v_path = output_dir.join("tiny_mlp.v");
    let harness =
        VerilatorHarness::build(&v_path, "tiny_mlp", TapDescriptor::v1_mlp())
            .expect("verilator harness build");

    // -----------------------------------------------------------------------
    // Per-stimulus parity: first-failure-stops
    // -----------------------------------------------------------------------
    let stimuli = v1_mlp_stimuli();
    for (stim_idx, x) in stimuli.iter().enumerate() {
        let hw = harness.run(x).expect("simulator run");

        // M57.1 §3.5: bias is the ripple-chain seed, so `*_matmul` holds the
        // post-bias accumulator value. Both `tap_l*_matmul_out` and the legacy
        // `tap_l*_bias_out` should compare against the same value bit-exactly
        // (the bias-tap descriptors are removed in Task 3.6).
        let cpu_l1_matmul = cpu_reference_layer1_matmul_with_bias(x, &w1, &b1);
        let cpu_l1_relu = cpu_reference_layer1_relu(&cpu_l1_matmul);
        let cpu_l2_matmul = cpu_reference_layer2_matmul_with_bias(&cpu_l1_relu, &w2, &b2);
        let cpu_out = cpu_reference_layer2_relu(&cpu_l2_matmul);

        assert_bit_exact(
            &hw.tap_i32("tap_l1_matmul_out", 128),
            &cpu_l1_matmul,
            stim_idx,
            "layer1_matmul",
        );
        assert_bit_exact(
            &hw.tap_i32("tap_l1_bias_out", 128),
            &cpu_l1_matmul,
            stim_idx,
            "layer1_bias",
        );
        assert_bit_exact(
            &hw.tap_i32("tap_l1_relu_out", 128),
            &cpu_l1_relu,
            stim_idx,
            "layer1_relu",
        );
        assert_bit_exact(
            &hw.tap_i64("tap_l2_matmul_out", 10),
            &cpu_l2_matmul,
            stim_idx,
            "layer2_matmul",
        );
        assert_bit_exact(
            &hw.tap_i64("tap_l2_bias_out", 10),
            &cpu_l2_matmul,
            stim_idx,
            "layer2_bias",
        );
        assert_bit_exact(
            &hw.tap_i64("out", 10),
            &cpu_out,
            stim_idx,
            "final_output",
        );
    }
}

/// Full-diagnostic variant — runs all 100 stimuli and accumulates failures.
///
/// Recommended pre-PR developer step per spec §7.2: after fixing a Layer 2
/// failure, run this to surface any masked second bug before opening the PR.
#[test]
#[ignore = "M57.1 v1 closure prerequisite: AST->KIR dispatch, HIR port/wire generation, \
            end-to-end CLI wiring (nsl fpga-compile), and Verilog synthesizability. \
            See spec §1.5 deferred-roadmap / crates/nsl-test/tests/fpga_mlp_v1_parity.rs header."]
fn fpga_mlp_v1_parity_full_diagnostic() {
    if !VerilatorHarness::is_available() {
        eprintln!("SKIPPED: verilator not installed");
        return;
    }

    let bin_path = "crates/nsl-test/fixtures/mlp_int8_weights_v1.bin";
    let toml_path = "crates/nsl-test/fixtures/mlp_int8_weights_v1.toml";
    let bytes = std::fs::read(bin_path).unwrap();
    let manifest_str = std::fs::read_to_string(toml_path).unwrap();
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
            bin_path,
            "--output-dir",
            output_dir.to_str().unwrap(),
            "--test-taps",
        ])
        .status()
        .unwrap();
    assert!(status.success(), "nsl fpga-compile failed");

    let v_path = output_dir.join("tiny_mlp.v");
    let harness =
        VerilatorHarness::build(&v_path, "tiny_mlp", TapDescriptor::v1_mlp())
            .expect("verilator harness build");

    let stimuli = v1_mlp_stimuli();
    let mut failures: Vec<String> = Vec::new();
    const MAX_REPORT: usize = 10;

    for (stim_idx, x) in stimuli.iter().enumerate() {
        if failures.len() >= MAX_REPORT {
            break;
        }
        let hw = match harness.run(x) {
            Ok(h) => h,
            Err(e) => {
                failures.push(format!("stimulus #{stim_idx}: simulator error: {e}"));
                continue;
            }
        };

        // M57.1 §3.5: bias-as-seed (see note in fpga_mlp_v1_parity above).
        let cpu_l1_matmul = cpu_reference_layer1_matmul_with_bias(x, &w1, &b1);
        let cpu_l1_relu = cpu_reference_layer1_relu(&cpu_l1_matmul);
        let cpu_l2_matmul = cpu_reference_layer2_matmul_with_bias(&cpu_l1_relu, &w2, &b2);
        let cpu_out = cpu_reference_layer2_relu(&cpu_l2_matmul);

        let check_i32 = |hw_tap: Vec<i32>, cpu_ref: &[i32], op: &str| {
            for (i, (h, c)) in hw_tap.iter().zip(cpu_ref).enumerate() {
                if h != c {
                    return Some(format!(
                        "stimulus #{stim_idx}, op `{op}`, element [{i}]: hw=0x{h:x} cpu=0x{c:x}"
                    ));
                }
            }
            None
        };
        let check_i64 = |hw_tap: Vec<i64>, cpu_ref: &[i64], op: &str| {
            for (i, (h, c)) in hw_tap.iter().zip(cpu_ref).enumerate() {
                if h != c {
                    return Some(format!(
                        "stimulus #{stim_idx}, op `{op}`, element [{i}]: hw=0x{h:x} cpu=0x{c:x}"
                    ));
                }
            }
            None
        };

        for fail in [
            check_i32(hw.tap_i32("tap_l1_matmul_out", 128), &cpu_l1_matmul, "layer1_matmul"),
            check_i32(hw.tap_i32("tap_l1_bias_out", 128), &cpu_l1_matmul, "layer1_bias"),
            check_i32(hw.tap_i32("tap_l1_relu_out", 128), &cpu_l1_relu, "layer1_relu"),
            check_i64(hw.tap_i64("tap_l2_matmul_out", 10), &cpu_l2_matmul, "layer2_matmul"),
            check_i64(hw.tap_i64("tap_l2_bias_out", 10), &cpu_l2_matmul, "layer2_bias"),
            check_i64(hw.tap_i64("out", 10), &cpu_out, "final_output"),
        ]
        .into_iter()
        .flatten()
        {
            failures.push(fail);
            if failures.len() >= MAX_REPORT {
                break;
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "Layer 2 full-diagnostic: {} failure(s) (showing up to {MAX_REPORT}):\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}
