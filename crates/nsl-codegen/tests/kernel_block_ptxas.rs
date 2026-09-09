//! Roadmap A2 step 3: user `kernel` blocks compile through KIR on the CUDA
//! target, and the PTX must assemble.
//!
//! Until step 3 the CUDA target had its own AST→PTX text emitter
//! (`kernel.rs`); this gate is what replaced its proof. Each kernel below is
//! parsed from NSL source exactly as `nsl build` would see it, lowered by
//! `kernel_lower::lower_kernel_to_ir` (verified), printed by `backend_ptx`
//! and piped through `ptxas --gpu-name sm_80` (the text says `.target sm_70`,
//! which an `sm_80` assembly accepts; the CUDA 13 toolkit CI installs no longer
//! takes `sm_70` as a `--gpu-name`). The set covers everything
//! the lowering accepts: the e2e fixtures' element-wise kernels, an
//! `if`/`elif`/`else` chain assigning a local (a join with a block
//! parameter), a `for ... in range(...)` accumulation (a loop header with
//! two parameters and a parallel-copy back edge), a `while` with `break`
//! and `continue`, compound stores, integer `%`, and the index builtins.
//!
//! `tests/snapshot_tests.rs` pins the same kernels' text
//! (`kernel_block_*`); this file proves the text is PTX.
//!
//! Skipped with a note when `ptxas` is not in PATH; CI's cuda-feature lane
//! installs the toolkit and runs this file in its "PTX emitter ptxas
//! validation" step. Same harness as `kir_scalar_isa_ptxas.rs`.

mod common;

use std::io::Write;
use std::process::{Command, Stdio};

use nsl_codegen::kernel_lower::compile_kernel_ptx;

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
    let win_default = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\ptxas.exe";
    if std::path::Path::new(win_default).exists() {
        return Some(win_default.to_string());
    }
    None
}

fn assemble_ptx(ptxas_path: &str, ptx_bytes: &[u8], sm_arch: &str) -> Result<(), String> {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let cubin = std::env::temp_dir().join(format!(
        "nsl_kernel_block_{sm_arch}_{}_{}.cubin",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let _cleanup = Cleanup(cubin.clone());
    let mut child = Command::new(ptxas_path)
        .args(["--gpu-name", sm_arch, "-O0", "-o"])
        .arg(&cubin)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to spawn ptxas: {e}"))?;
    // The backend null-terminates for cuModuleLoadData; ptxas wants text.
    let text = &ptx_bytes[..ptx_bytes.len().saturating_sub(1)];
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(text)
        .map_err(|e| format!("failed to write PTX to ptxas stdin: {e}"))?;
    let out = child.wait_with_output().map_err(|e| format!("ptxas did not exit: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).into_owned())
    }
}

#[test]
fn every_kernel_block_shape_assembles() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("kernel_block_ptxas: ptxas not found in PATH, skipping (CI's cuda lane runs this)");
        return;
    };
    for (name, src) in common::kernel_blocks::ALL {
        let (kernel, interner) = common::kernel_blocks::parse_first_kernel(src);
        let ptx = compile_kernel_ptx(&kernel, &interner)
            .unwrap_or_else(|e| panic!("{name}: kernel must compile: {e}"));
        let text = String::from_utf8_lossy(&ptx[..ptx.len() - 1]).into_owned();
        assemble_ptx(&ptxas, &ptx, "sm_80")
            .unwrap_or_else(|e| panic!("{name}: ptxas rejected the KIR lowering:\n{e}\nPTX:\n{text}"));
    }
}
