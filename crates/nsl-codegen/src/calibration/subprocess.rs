//! Spawn the calibration binary with a timeout and classify the exit
//! according to the spec §6.1 status-byte protocol:
//!
//!   0 → Clean           (all hooks returned Ok)
//!   1 → Degenerate      (at least one hook returned Degenerate)
//!   other / timeout / I/O error → Infrastructure

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Debug)]
pub enum SubprocessOutcome {
    Clean,
    Degenerate,
    Infrastructure(String),
}

/// Spawn `binary` and wait up to `timeout`.  Kills the process on
/// timeout and returns `Infrastructure`.  Status mapping:
/// 0 = Clean, 1 = Degenerate, anything else = Infrastructure.  Stdout
/// and stderr are inherited so the user sees subprocess diagnostics
/// inline with the build's own output.
pub fn run_subprocess(binary: &Path, timeout: Duration) -> std::io::Result<SubprocessOutcome> {
    let mut child = Command::new(binary)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()?;
    let start = Instant::now();
    loop {
        match child.try_wait()? {
            Some(status) => {
                return Ok(classify_exit(status.code()));
            }
            None => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Ok(SubprocessOutcome::Infrastructure(format!(
                        "timeout after {:?}",
                        timeout
                    )));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

fn classify_exit(code: Option<i32>) -> SubprocessOutcome {
    match code {
        Some(0) => SubprocessOutcome::Clean,
        Some(1) => SubprocessOutcome::Degenerate,
        Some(other) => SubprocessOutcome::Infrastructure(format!(
            "calibration subprocess exited with status {other}"
        )),
        None => SubprocessOutcome::Infrastructure(
            "calibration subprocess terminated by signal".into(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::Duration;

    fn echo_binary_that_exits(code: i32) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("nsl-calib-subproc-{code}.{}", std::process::id()));
        #[cfg(windows)]
        {
            p.set_extension("cmd");
            std::fs::write(&p, format!("@exit {code}\r\n")).unwrap();
        }
        #[cfg(not(windows))]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::write(&p, format!("#!/bin/sh\nexit {code}\n")).unwrap();
            let mut perms = std::fs::metadata(&p).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&p, perms).unwrap();
        }
        p
    }

    #[test]
    fn exit_0_maps_to_clean() {
        let b = echo_binary_that_exits(0);
        let r = run_subprocess(&b, Duration::from_secs(5));
        let _ = std::fs::remove_file(&b);
        assert!(matches!(r, Ok(SubprocessOutcome::Clean)));
    }

    #[test]
    fn exit_1_maps_to_degenerate() {
        let b = echo_binary_that_exits(1);
        let r = run_subprocess(&b, Duration::from_secs(5));
        let _ = std::fs::remove_file(&b);
        assert!(matches!(r, Ok(SubprocessOutcome::Degenerate)));
    }

    #[test]
    fn exit_2_maps_to_infrastructure() {
        let b = echo_binary_that_exits(2);
        let r = run_subprocess(&b, Duration::from_secs(5));
        let _ = std::fs::remove_file(&b);
        assert!(matches!(r, Ok(SubprocessOutcome::Infrastructure(_))));
    }

    #[test]
    fn timeout_maps_to_infrastructure() {
        let mut p = std::env::temp_dir();
        p.push(format!("nsl-calib-sleep.{}", std::process::id()));
        #[cfg(windows)]
        {
            p.set_extension("cmd");
            // Use ping for sleeping on Windows — `timeout /t` requires interactive console.
            std::fs::write(&p, "@ping -n 11 127.0.0.1 >nul\r\n").unwrap();
        }
        #[cfg(not(windows))]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::write(&p, "#!/bin/sh\nsleep 10\n").unwrap();
            let mut perms = std::fs::metadata(&p).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&p, perms).unwrap();
        }
        let r = run_subprocess(&p, Duration::from_millis(500));
        let _ = std::fs::remove_file(&p);
        match r {
            Ok(SubprocessOutcome::Infrastructure(msg)) => {
                assert!(msg.to_lowercase().contains("timeout"), "msg: {msg}")
            }
            other => panic!("expected Infrastructure(timeout), got {other:?}"),
        }
    }
}
