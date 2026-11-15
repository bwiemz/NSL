//! Every bf16-storage instantiation of the storage-generic families verifies,
//! is named by `storage::variant_name`, reads and writes 16-bit operands
//! where its storages say so, and (when `ptxas` is on the PATH) assembles for
//! `sm_80`, the target the KIR printer emits. Numerical correctness is the
//! GPU bracket tests' job (nsl-runtime `tests/bf16_storage_kernels_gpu.rs`).

use std::io::Write;
use std::process::{Command, Stdio};

use nsl_kir::kernels::elementwise::{
    backward_ptx_st, binary_ptx_st, rotate_half_ptx_st, scalar_ptx_st, unary_ptx_st, BackwardOp, BinaryOp,
    RotateHalfOp, ScalarOp, UnaryOp,
};
use nsl_kir::kernels::storage::{variant_name, Storage};

fn find_ptxas() -> Option<String> {
    for name in ["ptxas", "ptxas.exe"] {
        if Command::new(name).arg("--version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok() {
            return Some(name.to_string());
        }
    }
    None
}

fn assemble(ptxas: &str, ptx: &str) -> Result<(), String> {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let cubin = std::env::temp_dir().join(format!(
        "nsl_bf16_storage_{}_{}.cubin",
        std::process::id(),
        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let mut child = Command::new(ptxas)
        .args(["--gpu-name", "sm_80", "-o"])
        .arg(&cubin)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn ptxas: {e}"))?;
    child.stdin.take().expect("stdin").write_all(ptx.as_bytes()).map_err(|e| e.to_string())?;
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&cubin);
    if out.status.success() { Ok(()) } else { Err(String::from_utf8_lossy(&out.stderr).into_owned()) }
}

/// (expected entry name, PTX text, input storages, output storage)
fn instantiations() -> Vec<(String, String, Vec<Storage>, Storage)> {
    use Storage::*;
    let text = |v: Vec<u8>| String::from_utf8(v).expect("ASCII").trim_end_matches('\0').to_string();
    let mut out = Vec::new();
    let pairs = [[Bf16, Bf16], [F32, Bf16], [Bf16, F32], [F32, F32]];
    for op in BinaryOp::ALL {
        for ins in pairs {
            for o in Storage::ALL {
                if ins == [F32, F32] && o == F32 {
                    continue;
                }
                let name = variant_name(op.kernel_name(), &ins, o);
                out.push((name, text(binary_ptx_st(op, ins, o)), ins.to_vec(), o));
            }
        }
    }
    for i in Storage::ALL {
        for o in Storage::ALL {
            if i == F32 && o == F32 {
                continue;
            }
            for op in ScalarOp::ALL {
                out.push((variant_name(op.kernel_name(), &[i], o), text(scalar_ptx_st(op, i, o)), vec![i], o));
            }
            for op in UnaryOp::ALL {
                out.push((variant_name(op.kernel_name(), &[i], o), text(unary_ptx_st(op, i, o)), vec![i], o));
            }
            for op in RotateHalfOp::ALL {
                out.push((variant_name(op.kernel_name(), &[i], o), text(rotate_half_ptx_st(op, i, o)), vec![i], o));
            }
        }
    }
    for op in BackwardOp::ALL {
        let n = op.param_names().len() - 2;
        for o in Storage::ALL {
            let ins = vec![Bf16; n];
            out.push((variant_name(op.kernel_name(), &ins, o), text(backward_ptx_st(op, &ins, o)), ins, o));
        }
    }
    out
}

#[test]
fn every_bf16_instantiation_is_named_typed_and_assembles() {
    let ptxas = find_ptxas();
    if ptxas.is_none() {
        eprintln!("ptxas not on PATH: checking names and instructions only");
    }
    let all = instantiations();
    // binary 4 ops x 7 storage triples, scalar 4 x 3, unary 12 x 3,
    // rotate_half 2 x 3, backward 8 x 2 (all-bf16 inputs, either output).
    assert_eq!(all.len(), 4 * 7 + 4 * 3 + 12 * 3 + 2 * 3 + 8 * 2, "instantiation count");
    let mut failures = Vec::new();
    for (name, ptx, ins, out) in &all {
        assert!(ptx.contains(&format!(".visible .entry {name}(")), "{name}: entry not found:\n{ptx}");
        let any_bf16 = ins.contains(&Storage::Bf16) || *out == Storage::Bf16;
        assert!(any_bf16, "{name}: an all-f32 instantiation in the bf16 list");
        assert!(ptx.contains(".version 7.8"), "{name}: a bf16 kernel needs PTX 7.8");
        if ins.contains(&Storage::Bf16) {
            assert!(ptx.contains("ld.global.b16") && ptx.contains("cvt.f32.bf16"), "{name}: no widening bf16 load");
        }
        if *out == Storage::Bf16 {
            assert!(ptx.contains("cvt.rn.bf16.f32") && ptx.contains("st.global.b16"), "{name}: no rounded bf16 store");
            assert!(!ptx.contains("st.global.f32"), "{name}: stores f32 into a bf16 output");
        } else {
            assert!(!ptx.contains("st.global.b16"), "{name}: stores bf16 into an f32 output");
        }
        if let Some(p) = &ptxas
            && let Err(e) = assemble(p, ptx)
        {
            failures.push(format!("{name}: {e}"));
        }
    }
    assert!(failures.is_empty(), "ptxas rejected:\n{}", failures.join("\n"));
}
