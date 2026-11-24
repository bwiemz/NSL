//! The four MXFP8 quantizers assemble (when `ptxas` is on the PATH) for
//! `sm_89`, the target their header names, and for `sm_120`, the GPU they
//! run on. Their bytes are checked against the host reference by the GPU
//! test (nsl-runtime `tests/mx_quant_gpu.rs`).

use std::io::Write;
use std::process::{Command, Stdio};

use nsl_kir::kernels::mx_quant::{MxQuant, ptx};
use nsl_kir::kernels::storage::Storage;

fn find_ptxas() -> Option<String> {
    for name in ["ptxas", "ptxas.exe"] {
        if Command::new(name)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
        {
            return Some(name.to_string());
        }
    }
    None
}

fn assemble(ptxas: &str, ptx: &str, gpu: &str) -> Result<(), String> {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let cubin = std::env::temp_dir().join(format!(
        "nsl_mx_quant_{}_{}.cubin",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let mut child = Command::new(ptxas)
        .args(["--gpu-name", gpu, "-o"])
        .arg(&cubin)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn ptxas: {e}"))?;
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(ptx.as_bytes())
        .map_err(|e| e.to_string())?;
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&cubin);
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).into_owned())
    }
}

#[test]
fn every_quantizer_assembles_for_sm_89_and_sm_120() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("ptxas not on PATH; skipping the assembly check");
        return;
    };
    let mut failures = Vec::new();
    for q in MxQuant::ALL {
        for src in Storage::ALL {
            let text = String::from_utf8(ptx(q, src))
                .expect("ASCII")
                .trim_end_matches('\0')
                .to_string();
            for gpu in ["sm_89", "sm_120"] {
                if let Err(e) = assemble(&ptxas, &text, gpu) {
                    failures.push(format!("{} on {gpu}:\n{e}", q.kernel_name(src)));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}
