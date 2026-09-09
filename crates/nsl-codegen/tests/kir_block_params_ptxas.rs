//! Roadmap A2 step 2: a KIR loop written with block parameters must assemble.
//!
//! A loop-carried value in KIR is a block parameter: the header block lists
//! it, the entry edge passes its first value and the back edge passes the
//! next one (`KirEdge::with`). `backend_ptx` implements each edge as a
//! parallel copy into the parameter registers before the jump, and a
//! conditional branch whose edges carry arguments gets its own `_else`
//! label. The verifier holds the argument counts and types to the target's
//! parameters (rule 7); only `ptxas` proves the text is an instruction
//! stream. This gate builds the grid-stride copy every element-wise kernel
//! is made of and pipes it through `ptxas --gpu-name sm_70`.
//!
//! Skipped with a note when `ptxas` is not in PATH; CI's cuda-feature lane
//! installs the toolkit and runs this file in its "PTX emitter ptxas
//! validation" step. Same harness as `kir_async_copy_ptxas.rs`.

use std::io::Write;
use std::process::{Command, Stdio};

use nsl_codegen::backend_ptx::lower_kir_to_ptx;
use nsl_codegen::kernel_ir::{
    AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp,
    KirTerminator, KirType,
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
        "nsl_kir_block_params_{sm_arch}_{}_{}.cubin",
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

/// `out[i] = a[i] * 2` for `i` in a grid-stride loop, plus a running
/// `f32` sum carried alongside the index so the header has two
/// parameters of different classes and the back edge copies both.
fn grid_stride_scale() -> KernelIR {
    let f32_ptr = KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global);
    let mut b = KirBuilder::new("kir_grid_stride_scale");
    let a = b.add_param("a", f32_ptr.clone(), AddressSpace::Global);
    let out = b.add_param("out", f32_ptr.clone(), AddressSpace::Global);
    let sum_out = b.add_param("sum_out", f32_ptr.clone(), AddressSpace::Global);
    let n = b.add_param("n", KirType::U32, AddressSpace::Local);

    let entry = b.new_block();
    let header = b.new_block();
    let body = b.new_block();
    let exit = b.new_block();
    let i = b.add_block_param(header, KirType::U32);
    let acc = b.add_block_param(header, KirType::F32);

    b.set_block(entry);
    let start = b.new_typed_var(KirType::U32);
    b.emit(KirOp::GlobalId(start, 0));
    let zero = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Const(zero, KirConst { ty: KirType::F32, value: ConstValue::F32(0.0) }));
    b.terminate(KirTerminator::Branch(KirEdge::with(header, vec![start, zero])));

    b.set_block(header);
    let more = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(more, i, n, CmpOp::Lt));
    b.terminate(KirTerminator::CondBranch(more, body.into(), exit.into()));

    b.set_block(body);
    let src = b.new_typed_var(f32_ptr.clone());
    b.emit(KirOp::PtrOffset(src, a, i));
    let v = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Load(v, src, AddressSpace::Global));
    let two = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Const(two, KirConst { ty: KirType::F32, value: ConstValue::F32(2.0) }));
    let scaled = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Mul(scaled, v, two));
    let dst = b.new_typed_var(f32_ptr);
    b.emit(KirOp::PtrOffset(dst, out, i));
    b.emit(KirOp::Store(dst, scaled, AddressSpace::Global));
    let acc_next = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Add(acc_next, acc, scaled));
    let bdim = b.new_typed_var(KirType::U32);
    b.emit(KirOp::BlockDim(bdim, 0));
    let gdim = b.new_typed_var(KirType::U32);
    b.emit(KirOp::GridDim(gdim, 0));
    let stride = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Mul(stride, bdim, gdim));
    let next = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Add(next, i, stride));
    b.terminate(KirTerminator::Branch(KirEdge::with(header, vec![next, acc_next])));

    b.set_block(exit);
    // `acc` is the header's parameter and the header dominates the exit.
    b.emit(KirOp::AtomicAdd(sum_out, acc, AddressSpace::Global));
    b.terminate(KirTerminator::Return);

    b.set_workgroup_size([256, 1, 1]);
    b.finalize()
}

#[test]
fn the_grid_stride_loop_verifies_and_lowers_to_edge_copies() {
    let ir = grid_stride_scale();
    assert_eq!(ir.verify(), Ok(()));
    let ptx = String::from_utf8(lower_kir_to_ptx(&ir)).unwrap();
    assert!(ptx.contains(".target sm_70"), "{ptx}");
    // Two back-edge copies (index and accumulator) precede the jump.
    let back_edge = ptx.rsplit("bra BB1;").nth(1).unwrap();
    assert!(back_edge.contains("mov.u32 %r"), "{ptx}");
    assert!(back_edge.contains("mov.f32 %f"), "{ptx}");
    // The scratch class is declared because the kernel has block parameters.
    assert!(ptx.contains(".reg .u32 %edge_r;"), "{ptx}");
}

#[test]
fn ptxas_accepts_the_kir_grid_stride_loop() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("ptxas not found; skipping (install the CUDA toolkit to run this gate)");
        return;
    };
    let ir = grid_stride_scale();
    let ptx = lower_kir_to_ptx(&ir);
    match assemble_ptx(&ptxas, &ptx, "sm_70") {
        Ok(()) => println!("ptxas accepted the KIR grid-stride loop (sm_70)"),
        Err(stderr) => panic!(
            "ptxas rejected the KIR grid-stride loop:\n{stderr}\n--- PTX ---\n{}",
            String::from_utf8_lossy(&ptx)
        ),
    }
}
