//! Roadmap A2 step 4: the KIR scalar ISA must assemble.
//!
//! The hand-PTX estate's index math and reductions are shifts, masks,
//! `rem`, `min`/`max`, warp shuffles and votes; its 16-bit paths are
//! `ld/st.global.b16` around `cvt.rn.{f16,bf16}.f32`; its bandwidth kernels
//! are `.v4` loads; its guarded stores are `@%p st`. This gate builds one
//! kernel that uses every family `backend_ptx` gained in step 4 — a bf16
//! conversion puts it on PTX 7.8 / `sm_80` — verifies it, and pipes the
//! text through `ptxas --gpu-name sm_80`.
//!
//! Skipped with a note when `ptxas` is not in PATH; CI's cuda-feature lane
//! installs the toolkit and runs this file in its "PTX emitter ptxas
//! validation" step. Same harness as `kir_async_copy_ptxas.rs`.

use std::io::Write;
use std::process::{Command, Stdio};

use nsl_codegen::backend_ptx::lower_kir_to_ptx;
use nsl_codegen::kernel_ir::{
    AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirOp, KirTerminator,
    KirType, RoundMode, ShuffleMode, VoteMode,
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
        "nsl_kir_scalar_isa_{sm_arch}_{}_{}.cubin",
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

/// One thread per lane: load four f32 from `src` at a shifted/masked
/// index, warp-reduce the first with a butterfly shuffle and `max`, cast
/// through bf16 and back with the default and an explicit rounding, vote,
/// and store the results — the second store predicated on the vote.
fn scalar_isa_kernel() -> KernelIR {
    let f32_ptr = KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global);
    let bf16_ptr = KirType::Ptr(Box::new(KirType::Bf16), AddressSpace::Global);
    let mut b = KirBuilder::new("kir_scalar_isa");
    let src = b.add_param("src", f32_ptr.clone(), AddressSpace::Global);
    let out = b.add_param("out", f32_ptr.clone(), AddressSpace::Global);
    let out16 = b.add_param("out16", bf16_ptr, AddressSpace::Global);
    let flags = b.add_param("flags", KirType::Ptr(Box::new(KirType::U32), AddressSpace::Global), AddressSpace::Global);

    let entry = b.new_block();
    b.set_block(entry);
    let u32c = |b: &mut KirBuilder, v: u32| {
        let d = b.new_typed_var(KirType::U32);
        b.emit(KirOp::Const(d, KirConst { ty: KirType::U32, value: ConstValue::U32(v) }));
        d
    };

    // idx = ((tid << 2) & 0xfffc) | (lane % 4); base = src + idx (u64 math).
    let tid = b.new_typed_var(KirType::U32);
    b.emit(KirOp::GlobalId(tid, 0));
    let two = u32c(&mut b, 2);
    let shifted = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Shl(shifted, tid, two));
    let mask = u32c(&mut b, 0xfffc);
    let masked = b.new_typed_var(KirType::U32);
    b.emit(KirOp::And(masked, shifted, mask));
    let lane = b.new_typed_var(KirType::U32);
    b.emit(KirOp::LaneId(lane));
    let four = u32c(&mut b, 4);
    let low = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Rem(low, lane, four));
    let idx = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Or(idx, masked, low));
    let idx64 = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Cast(idx64, idx, KirType::U64));
    let bytes = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Const(bytes, KirConst { ty: KirType::U64, value: ConstValue::U64(2) }));
    let off = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Shl(off, idx64, two));
    let _ = bytes;
    let base = b.new_typed_var(f32_ptr.clone());
    b.emit(KirOp::PtrOffset(base, src, idx));
    let _ = off;

    // Four values at once.
    let v: Vec<_> = (0..4).map(|_| b.new_typed_var(KirType::F32)).collect();
    b.emit(KirOp::LoadVec { dsts: v.clone(), ptr: base, space: AddressSpace::Global });

    // Butterfly max-reduce v0 across the warp; 1/sqrt of the result.
    let one = u32c(&mut b, 1);
    let peer = b.new_typed_var(KirType::F32);
    b.emit(KirOp::WarpShuffle { dst: peer, val: v[0], lane: one, mode: ShuffleMode::Xor, width: 32 });
    let m = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Max(m, v[0], peer));
    let rs = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Rsqrt(rs, m));
    let rc = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Rcp(rc, v[1]));

    // Through bf16 and back (default .rn), and an explicit .rz to f16.
    let h = b.new_typed_var(KirType::Bf16);
    b.emit(KirOp::Cast(h, rs, KirType::Bf16));
    let dst16 = b.new_typed_var(KirType::Ptr(Box::new(KirType::Bf16), AddressSpace::Global));
    b.emit(KirOp::PtrOffset(dst16, out16, tid));
    b.emit(KirOp::Store(dst16, h, AddressSpace::Global));
    let back = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Cast(back, h, KirType::F32));
    let z = b.new_typed_var(KirType::F16);
    b.emit(KirOp::CastRounded { dst: z, src: rc, ty: KirType::F16, mode: RoundMode::Rz });
    let zf = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Cast(zf, z, KirType::F32));
    let trunc = b.new_typed_var(KirType::I32);
    b.emit(KirOp::Cast(trunc, zf, KirType::I32));
    let truncf = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Cast(truncf, trunc, KirType::F32));

    // Vote on whether the lane's value is positive; select and store.
    let zero = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Const(zero, KirConst { ty: KirType::F32, value: ConstValue::F32(0.0) }));
    let pos = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(pos, back, zero, CmpOp::Gt));
    let any = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Vote { dst: any, pred: pos, mode: VoteMode::Any });
    let ballot = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Vote { dst: ballot, pred: pos, mode: VoteMode::Ballot });
    let notpos = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Not(notpos, pos));
    let either = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Select(either, any, pos, notpos));
    let chosen = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Select(chosen, either, truncf, v[2]));
    let sum = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Add(sum, chosen, v[3]));
    let dst = b.new_typed_var(f32_ptr);
    b.emit(KirOp::PtrOffset(dst, out, tid));
    b.emit(KirOp::StoreVec { ptr: dst, vals: vec![sum, chosen], space: AddressSpace::Global });
    let fdst = b.new_typed_var(KirType::Ptr(Box::new(KirType::U32), AddressSpace::Global));
    b.emit(KirOp::PtrOffset(fdst, flags, tid));
    b.emit(KirOp::Predicated {
        pred: any,
        negate: false,
        op: Box::new(KirOp::Store(fdst, ballot, AddressSpace::Global)),
    });
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([32, 1, 1]);
    b.finalize()
}

#[test]
fn the_scalar_isa_kernel_verifies_and_lowers() {
    let ir = scalar_isa_kernel();
    assert_eq!(ir.verify(), Ok(()));
    let ptx = String::from_utf8(lower_kir_to_ptx(&ir)).unwrap();
    assert!(ptx.contains(".version 7.8\n.target sm_80"), "{ptx}");
    for family in [
        "shl.b32", "and.b32", "rem.u32", "or.b32", "shl.b64", "ld.global.v4.f32 {",
        "shfl.sync.bfly.b32", "max.f32", "rsqrt.approx.f32", "rcp.approx.f32",
        "cvt.rn.bf16.f32", "st.global.b16", "cvt.f32.bf16", "cvt.rz.f16.f32",
        "cvt.rzi.s32.f32", "cvt.rn.f32.s32", "vote.sync.any.pred", "vote.sync.ballot.b32",
        "not.pred", "selp.f32", "st.global.v2.f32 [", "@%p",
    ] {
        assert!(ptx.contains(family), "missing `{family}` in\n{ptx}");
    }
}

#[test]
fn ptxas_accepts_the_kir_scalar_isa_kernel() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("ptxas not found; skipping (install the CUDA toolkit to run this gate)");
        return;
    };
    let ir = scalar_isa_kernel();
    let ptx = lower_kir_to_ptx(&ir);
    match assemble_ptx(&ptxas, &ptx, "sm_80") {
        Ok(()) => println!("ptxas accepted the KIR scalar-ISA kernel (sm_80)"),
        Err(stderr) => panic!(
            "ptxas rejected the KIR scalar-ISA kernel:\n{stderr}\n--- PTX ---\n{}",
            String::from_utf8_lossy(&ptx)
        ),
    }
}
