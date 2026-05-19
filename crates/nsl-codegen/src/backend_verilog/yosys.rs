//! Yosys synthesis gate per spec §5.2-§5.4.
//! Dual-stream capture (stdout + stderr); permissive regex; warnings-as-errors.

use std::path::Path;
use std::process::Command;
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use regex::Regex;
use thiserror::Error;

/// Permissive warning/error scanner per spec §5.3 Correction 1.
/// Hoisted to a static so `YosysGate::run` doesn't rebuild the DFA on every
/// invocation (the run loop scans the full Yosys log line-by-line).
static WARN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:^|\s)(Warning|Error):\s").unwrap());

#[derive(Debug, Error)]
pub enum YosysGateError {
    #[error("Yosys exited non-zero (exit={exit_code}). Script: {script}\nstderr:\n{stderr}")]
    YosysFailed { exit_code: i32, script: String, stderr: String },
    #[error("Yosys gate failed: {count} warning(s)/error(s) (zero-warnings policy).\n\n{lines}\n\nFull Yosys log: {log_path}")]
    WarningsEmitted { count: usize, lines: String, log_path: String },
    // Post-mortem detection only: `Command::output()` blocks until the child
    // exits, so this fires after a slow-but-completed run, not during a hang.
    // The CI `timeout-minutes: 15` is the actual backstop for hangs. A true
    // wall-clock interrupt would require Command::spawn + wait_timeout.
    #[error("Yosys gate exceeded 2x budget: {elapsed_secs}s > 600s. CI-cost-policy abort per §5.4. \
             Three next steps: (1) re-run for noise check; (2) compare against historical timing curve; \
             (3) activate fixture downsize (28→16→10 CI, full on nightly).")]
    BudgetExceeded { elapsed_secs: u64 },
    #[error("IO error invoking yosys: {0}")]
    Io(#[from] std::io::Error),
}

pub struct YosysGate;

impl YosysGate {
    /// Run yosys against the given Verilog file. Returns Ok(duration) on clean
    /// synthesis with zero warnings; error otherwise.
    pub fn run(verilog_path: &Path, log_path: &Path) -> Result<Duration, YosysGateError> {
        let start = Instant::now();

        let script = format!(
            "read_verilog -sv {}; synth -noabc -flatten; check",
            verilog_path.display()
        );
        let output = Command::new("yosys")
            .args(["-p", &script, "-l", &log_path.display().to_string()])
            .output()?;

        let elapsed = start.elapsed();

        // Budget gate per §5.4 (post-mortem; see BudgetExceeded comment)
        if elapsed > Duration::from_secs(300) {
            eprintln!("WARNING: Yosys gate exceeded 5-min target ({elapsed:?}).");
        }
        if elapsed > Duration::from_secs(600) {
            return Err(YosysGateError::BudgetExceeded { elapsed_secs: elapsed.as_secs() });
        }

        if !output.status.success() {
            return Err(YosysGateError::YosysFailed {
                exit_code: output.status.code().unwrap_or(-1),
                script,
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            });
        }

        // Dual-stream capture + permissive regex per §5.3 Correction 1
        let combined = format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        let offending: Vec<&str> = combined.lines().filter(|l| WARN_RE.is_match(l)).collect();

        if !offending.is_empty() {
            return Err(YosysGateError::WarningsEmitted {
                count: offending.len(),
                lines: offending.join("\n"),
                log_path: log_path.display().to_string(),
            });
        }

        Ok(elapsed)
    }

    /// Returns true if yosys is installed and accessible on PATH.
    pub fn is_available() -> bool {
        Command::new("yosys").arg("-V").output()
            .map(|o| o.status.success()).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regex_matches_bare_warning() {
        assert!(WARN_RE.is_match("Warning: foo"));
        assert!(WARN_RE.is_match("Error: bar"));
    }

    #[test]
    fn regex_matches_file_prefixed_warning() {
        assert!(WARN_RE.is_match("/tmp/tiny_mlp.v:1234: Warning: foo"));
    }

    #[test]
    fn regex_rejects_identifier_prefix_false_positive() {
        assert!(!WARN_RE.is_match("WarningExample: foo"));   // identifier, not prefix
        assert!(!WARN_RE.is_match("MyError:foo"));            // no space after colon
    }
}
