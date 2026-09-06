//! Roadmap A2 step 2: the KIR async-copy group must assemble.
//!
//! `backend_ptx` lowers `SharedBase` / `CpAsync` / `CpAsyncCommit` /
//! `CpAsyncWait` to `mov.u64 %rd, shared_mem` / `cp.async.ca.shared.global`
//! / `cp.async.commit_group` / `cp.async.wait_group` under an `sm_80`
//! target. The verifier proves the IR is well formed; only `ptxas` proves
//! the text it becomes is an instruction stream. This gate pipes a
//! one-stage pipeline kernel — stage 16 bytes of a global row into shared
//! memory, wait, barrier, read it back, store the sum to global — through
//! `ptxas --gpu-name sm_80` and asserts acceptance.
//!
//! Skipped with a note when `ptxas` is not in PATH (the CUDA toolkit is
//! not installed here); CI's cuda-feature lane installs the toolkit and
//! runs this file in its "PTX emitter ptxas validation" step. Same harness
//! as `csha_ptx_ptxas_validation.rs`.

use std::io::Write;
use std::process::{Command, Stdio};

use nsl_codegen::backend_ptx::lower_kir_to_ptx;
use nsl_codegen::kernel_ir::{AddressSpace, KernelIR, KirBuilder, KirOp, KirTerminator, KirType};

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
        "nsl_kir_async_copy_{sm_arch}_{}_{}.cubin",
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

/// One pipeline stage: cp.async 16 bytes (four f32) of `src` into shared
/// memory, commit, wait for everything, barrier, then read the first
/// element back and store it to `out`.
fn one_stage_pipeline() -> KernelIR {
    let mut b = KirBuilder::new("kir_async_copy_stage");
    let src = b.add_param(
        "src",
        KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global),
        AddressSpace::Global,
    );
    let out = b.add_param(
        "out",
        KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global),
        AddressSpace::Global,
    );
    b.set_shared_mem(64);
    let entry = b.new_block();
    b.set_block(entry);
    let smem = b.new_typed_var(KirType::Ptr(Box::new(KirType::F32), AddressSpace::Shared));
    b.emit(KirOp::SharedBase(smem));
    b.emit(KirOp::CpAsync { dst: smem, src, bytes: 16 });
    b.emit(KirOp::CpAsyncCommit);
    b.emit(KirOp::CpAsyncWait { pending: 0 });
    b.emit(KirOp::Barrier);
    let v = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Load(v, smem, AddressSpace::Shared));
    b.emit(KirOp::Store(out, v, AddressSpace::Global));
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([32, 1, 1]);
    b.finalize()
}

#[test]
fn the_async_copy_stage_verifies_and_lowers_to_cp_async() {
    let ir = one_stage_pipeline();
    assert_eq!(ir.verify(), Ok(()));
    let ptx = String::from_utf8(lower_kir_to_ptx(&ir)).unwrap();
    assert!(ptx.contains(".target sm_80"), "{ptx}");
    assert!(ptx.contains("cp.async.ca.shared.global"), "{ptx}");
    assert!(ptx.contains("cp.async.commit_group;"), "{ptx}");
    assert!(ptx.contains("cp.async.wait_group 0;"), "{ptx}");
}

#[test]
fn ptxas_accepts_the_kir_async_copy_stage() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("ptxas not found; skipping (install the CUDA toolkit to run this gate)");
        return;
    };
    let ir = one_stage_pipeline();
    let ptx = lower_kir_to_ptx(&ir);
    match assemble_ptx(&ptxas, &ptx, "sm_80") {
        Ok(()) => println!("ptxas accepted the KIR async-copy stage (sm_80)"),
        Err(stderr) => panic!(
            "ptxas rejected the KIR async-copy stage:\n{stderr}\n--- PTX ---\n{}",
            String::from_utf8_lossy(&ptx)
        ),
    }
}
