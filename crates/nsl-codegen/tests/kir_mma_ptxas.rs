//! Roadmap A2 step 2: the KIR tensor-core tile must assemble.
//!
//! `backend_ptx` lowers `LdMatrixX4` / `MmaF16M16N8K16` to
//! `ldmatrix.sync.aligned.m8n8.x4[.trans].shared.b16` /
//! `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` under an `sm_80`
//! target, with the packed fragments in a `.reg .b32 %v<N>` class. The
//! verifier holds every fragment register to its type; only `ptxas` proves
//! the text is an instruction stream. This gate builds one tile — stage
//! A and B fragments from shared memory with two `ldmatrix`, one `mma`
//! into a zero accumulator, store the first accumulator lane — and pipes
//! it through `ptxas --gpu-name sm_80`.
//!
//! Skipped with a note when `ptxas` is not in PATH; CI's cuda-feature lane
//! installs the toolkit and runs this file in its "PTX emitter ptxas
//! validation" step. Same harness as `kir_async_copy_ptxas.rs`.

use std::io::Write;
use std::process::{Command, Stdio};

use nsl_codegen::backend_ptx::lower_kir_to_ptx;
use nsl_codegen::kernel_ir::{
    AddressSpace, ConstValue, KernelIR, KirBuilder, KirConst, KirOp, KirTerminator, KirType,
};

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
        "nsl_kir_mma_{sm_arch}_{}_{}.cubin",
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

/// One m16n8k16 tile: A and B fragments via `ldmatrix.x4` (B transposed),
/// a zero accumulator, one `mma.sync`, the first accumulator lane stored.
fn one_tile() -> KernelIR {
    let mut b = KirBuilder::new("kir_mma_tile");
    let out = b.add_param(
        "out",
        KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global),
        AddressSpace::Global,
    );
    b.set_shared_mem(1024);
    let entry = b.new_block();
    b.set_block(entry);
    let frag = || KirType::Vec(Box::new(KirType::F16), 2);
    let smem = b.new_typed_var(KirType::Ptr(Box::new(KirType::F16), AddressSpace::Shared));
    b.emit(KirOp::SharedBase(smem));
    let a = [
        b.new_typed_var(frag()),
        b.new_typed_var(frag()),
        b.new_typed_var(frag()),
        b.new_typed_var(frag()),
    ];
    b.emit(KirOp::LdMatrixX4 { dst: a, addr: smem, trans: false });
    let bb = [
        b.new_typed_var(frag()),
        b.new_typed_var(frag()),
        b.new_typed_var(frag()),
        b.new_typed_var(frag()),
    ];
    b.emit(KirOp::LdMatrixX4 { dst: bb, addr: smem, trans: true });
    let mut c = [0; 4];
    for slot in &mut c {
        *slot = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Const(
            *slot,
            KirConst { ty: KirType::F32, value: ConstValue::F32(0.0) },
        ));
    }
    let d = [
        b.new_typed_var(KirType::F32),
        b.new_typed_var(KirType::F32),
        b.new_typed_var(KirType::F32),
        b.new_typed_var(KirType::F32),
    ];
    b.emit(KirOp::MmaF16M16N8K16 { d, a, b: [bb[0], bb[1]], c });
    b.emit(KirOp::Store(out, d[0], AddressSpace::Global));
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([32, 1, 1]);
    b.finalize()
}

#[test]
fn the_tile_verifies_and_lowers_to_ldmatrix_and_mma() {
    let ir = one_tile();
    assert_eq!(ir.verify(), Ok(()));
    let ptx = String::from_utf8(lower_kir_to_ptx(&ir)).unwrap();
    assert!(ptx.contains(".target sm_80"), "{ptx}");
    assert!(ptx.contains(".reg .b32 %v<"), "{ptx}");
    assert!(ptx.contains("ldmatrix.sync.aligned.m8n8.x4.shared.b16"), "{ptx}");
    assert!(ptx.contains("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"), "{ptx}");
    assert!(ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"), "{ptx}");
}

#[test]
fn ptxas_accepts_the_kir_mma_tile() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("ptxas not found; skipping (install the CUDA toolkit to run this gate)");
        return;
    };
    let ir = one_tile();
    let ptx = lower_kir_to_ptx(&ir);
    match assemble_ptx(&ptxas, &ptx, "sm_80") {
        Ok(()) => println!("ptxas accepted the KIR mma tile (sm_80)"),
        Err(stderr) => panic!(
            "ptxas rejected the KIR mma tile:\n{stderr}\n--- PTX ---\n{}",
            String::from_utf8_lossy(&ptx)
        ),
    }
}
