//! Verilator external-process harness per M57 spec §6.6.
//!
//! `VerilatorHarness::build` compiles a Verilog file using Verilator's
//! `--binary` mode, then `VerilatorHarness::run` spawns the resulting
//! simulation binary with the test vector on stdin and reads tap-port
//! values from stdout.
//!
//! NOTE: This module compiles cleanly but the harness is NOT exercised on
//! this branch — the v1 MLP Verilog is not yet synthesizable end-to-end.
//! The Layer 2 parity test that uses this harness is `#[ignore]`'d in
//! `crates/nsl-test/tests/fpga_mlp_v1_parity.rs` pending M57.1.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum HarnessError {
    #[error("Verilator compile failed:\nstdout: {stdout}\nstderr: {stderr}")]
    CompileFailed { stdout: String, stderr: String },

    #[error("Verilator simulation crashed (exit {exit_code:?}):\n{stderr}")]
    SimulationCrashed {
        exit_code: Option<i32>,
        stderr: String,
    },

    #[error("Output parse failure on line {unparsed_line:?}; expected {expected_format}")]
    OutputParseFailed {
        unparsed_line: String,
        expected_format: String,
    },

    /// Per spec §2.6: 2× budget (4 min) is the hard abort threshold.
    #[error(
        "Verilator gate exceeded 2x budget: {elapsed:?} > 4 min.\n\
         Per spec §2.6, the v1 architecture commitment stays; the fixture scope adapts.\n\
         Three next steps: (1) re-run for noise check; (2) compare against historical timing; \
         (3) activate fixture downsize (28->16->10 shape for CI)."
    )]
    BudgetExceeded { elapsed: Duration },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// ---------------------------------------------------------------------------
// Tap descriptor
// ---------------------------------------------------------------------------

/// Describes the tap-output port layout for one DUT.
pub struct TapDescriptor {
    /// `(port_name, width_in_bits)` — must match the emitted Verilog.
    pub taps: Vec<(String, usize)>,
}

impl TapDescriptor {
    /// Tap descriptor for the v1 MLP.
    ///
    /// Per spec §2.7: test-tap ports expose every per-op intermediate signal
    /// when `--test-taps` is passed at codegen time.
    pub fn v1_mlp() -> Self {
        Self {
            taps: vec![
                ("tap_l1_matmul_out".into(), 128 * 32), // 128 × i32
                ("tap_l1_bias_out".into(), 128 * 32),
                ("tap_l1_relu_out".into(), 128 * 32),
                ("tap_l2_matmul_out".into(), 10 * 64), // 10 × i64
                ("tap_l2_bias_out".into(), 10 * 64),
                ("out".into(), 10 * 64),
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// HarnessOutput
// ---------------------------------------------------------------------------

/// Per-stimulus output from the Verilator simulation.
#[derive(Debug)]
pub struct HarnessOutput {
    /// Map from tap-port name → raw bytes in little-endian order.
    pub taps: std::collections::HashMap<String, Vec<u8>>,
}

impl HarnessOutput {
    /// Decode a tap as `count` little-endian i32 values.
    pub fn tap_i32(&self, name: &str, count: usize) -> Vec<i32> {
        let bytes = self.taps.get(name).expect("tap not found");
        bytes
            .chunks_exact(4)
            .take(count)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Decode a tap as `count` little-endian i64 values.
    pub fn tap_i64(&self, name: &str, count: usize) -> Vec<i64> {
        let bytes = self.taps.get(name).expect("tap not found");
        bytes
            .chunks_exact(8)
            .take(count)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// VerilatorHarness
// ---------------------------------------------------------------------------

pub struct VerilatorHarness {
    sim_binary: PathBuf,
    #[allow(dead_code)] // used at M57.1 when run() is wired to tap parsing
    tap_descriptor: TapDescriptor,
}

impl VerilatorHarness {
    /// Compile `verilog_path` with Verilator and return a ready harness.
    ///
    /// Verilator must be on `$PATH`; call `is_available()` first if the test
    /// should skip gracefully when Verilator is not installed.
    pub fn build(
        verilog_path: &Path,
        top: &str,
        taps: TapDescriptor,
    ) -> Result<Self, HarnessError> {
        let obj_dir = Path::new("target/fpga/verilator-obj");
        std::fs::create_dir_all(obj_dir)?;

        let testbench = Path::new("crates/nsl-test/src/testbench.cpp");
        let sim_name = format!("{top}_sim");

        let output = Command::new("verilator")
            .args([
                "--binary",
                "--top-module",
                top,
                "-Wall",
                "-Wno-fatal",
                "--Mdir",
                obj_dir.to_str().unwrap(),
                "-o",
                &sim_name,
                verilog_path.to_str().unwrap(),
                testbench.to_str().unwrap(),
            ])
            .output()?;

        if !output.status.success() {
            return Err(HarnessError::CompileFailed {
                stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }

        Ok(Self {
            sim_binary: obj_dir.join(&sim_name),
            tap_descriptor: taps,
        })
    }

    /// Run the simulation with one 784-element i8 input vector.
    ///
    /// Spawns the compiled simulation binary, writes the input to stdin,
    /// and parses the tap-port hex values from stdout.
    pub fn run(&self, input: &[i8]) -> Result<HarnessOutput, HarnessError> {
        let start = Instant::now();

        let mut child = Command::new(&self.sim_binary)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let stdin_bytes: Vec<u8> = input.iter().map(|&b| b as u8).collect();
        child
            .stdin
            .as_mut()
            .expect("stdin piped")
            .write_all(&stdin_bytes)?;

        let output = child.wait_with_output()?;
        let elapsed = start.elapsed();

        // Warn at 1× budget (2 min), abort at 2× (4 min).
        if elapsed > Duration::from_secs(120) {
            eprintln!(
                "WARNING: Verilator simulation exceeded 2-min target ({elapsed:?}). \
                 Per §2.6, this is the '5-10 min slow-but-acceptable zone'. \
                 If consistently exceeded, activate fixture downsize."
            );
        }
        if elapsed > Duration::from_secs(240) {
            return Err(HarnessError::BudgetExceeded { elapsed });
        }

        if !output.status.success() {
            return Err(HarnessError::SimulationCrashed {
                exit_code: output.status.code(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }

        self.parse_output(&String::from_utf8_lossy(&output.stdout))
    }

    fn parse_output(&self, stdout: &str) -> Result<HarnessOutput, HarnessError> {
        let mut taps = std::collections::HashMap::new();
        for line in stdout.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let (name, hex) = line
                .split_once('=')
                .ok_or_else(|| HarnessError::OutputParseFailed {
                    unparsed_line: line.to_string(),
                    expected_format: "<name>=0x<hex>".to_string(),
                })?;
            let hex = hex.trim_start_matches("0x");
            // Hex is big-endian (MSB first); decode to little-endian bytes.
            let bytes: Vec<u8> = (0..hex.len())
                .step_by(2)
                .map(|i| {
                    u8::from_str_radix(&hex[i..i + 2], 16).unwrap_or(0)
                })
                .rev()
                .collect();
            taps.insert(name.to_string(), bytes);
        }
        Ok(HarnessOutput { taps })
    }

    /// Returns `true` if `verilator --version` succeeds.
    ///
    /// Tests that need Verilator should call this and skip if it returns false.
    pub fn is_available() -> bool {
        Command::new("verilator")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tap_descriptor_v1_mlp_shape() {
        let desc = TapDescriptor::v1_mlp();
        assert_eq!(desc.taps.len(), 6);
        assert_eq!(desc.taps[0].0, "tap_l1_matmul_out");
        assert_eq!(desc.taps[0].1, 128 * 32);
        assert_eq!(desc.taps[5].0, "out");
        assert_eq!(desc.taps[5].1, 10 * 64);
    }

    #[test]
    fn parse_output_valid_hex() {
        // Build a minimal harness instance for testing parse_output.
        let harness = VerilatorHarness {
            sim_binary: PathBuf::from("/nonexistent"),
            tap_descriptor: TapDescriptor::v1_mlp(),
        };
        // Single byte value 0x42 → little-endian: [0x42]
        let stdout = "tap_l1_matmul_out=0x42\n";
        let result = harness.parse_output(stdout).unwrap();
        // Hex "42" is 1 byte; reversed = still [0x42]
        assert_eq!(result.taps["tap_l1_matmul_out"], vec![0x42u8]);
    }

    #[test]
    fn parse_output_round_trips_two_i32_words() {
        // Testbench dumps multi-word taps high-word-first as big-endian hex
        // bytes, then parse_output reverses the whole byte sequence so
        // tap_i32 reads little-endian. Verify with two i32 words: [1, 2].
        //
        // Testbench would emit dut[1]=0x00000002 first, then dut[0]=0x00000001
        // → "0x0000000200000001". Hex bytes msb-first:
        //   [0x00,0x00,0x00,0x02,0x00,0x00,0x00,0x01]
        // After flat reverse:
        //   [0x01,0x00,0x00,0x00,0x02,0x00,0x00,0x00]
        // tap_i32 reads as LE pairs → [1, 2]. ✓
        let harness = VerilatorHarness {
            sim_binary: PathBuf::from("/nonexistent"),
            tap_descriptor: TapDescriptor::v1_mlp(),
        };
        let stdout = "tap_l1_matmul_out=0x0000000200000001\n";
        let result = harness.parse_output(stdout).unwrap();
        let words = result.tap_i32("tap_l1_matmul_out", 2);
        assert_eq!(words, vec![1i32, 2i32]);
    }

    #[test]
    fn parse_output_rejects_malformed_line() {
        let harness = VerilatorHarness {
            sim_binary: PathBuf::from("/nonexistent"),
            tap_descriptor: TapDescriptor::v1_mlp(),
        };
        let stdout = "no_equals_sign_here\n";
        match harness.parse_output(stdout) {
            Err(HarnessError::OutputParseFailed { .. }) => {}
            other => panic!("expected OutputParseFailed, got {other:?}"),
        }
    }

    #[test]
    fn is_available_does_not_panic() {
        // Just verify the call doesn't panic regardless of whether Verilator
        // is installed in this environment.
        let _ = VerilatorHarness::is_available();
    }
}
