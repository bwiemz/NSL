//! New-roadmap item 5: the fused adjoint elementwise-chain kernel, now KIR
//! (`nsl_codegen::ew_chain_ptx`), must assemble for every architecture the
//! serving GPU table names.
//!
//! The module targets the KIR backend's floor (`.version 7.0` / `.target
//! sm_70`) and the driver JIT-compiles it forward; this gate assembles it
//! with `ptxas --gpu-name sm_XX` for each architecture, the offline form of
//! that JIT, over chains that use every opcode and operand kind. Its
//! behaviour is proved against the frozen hand emitter by
//! `ew_chain_kir_equivalence`.
//!
//! Skipped with a note when `ptxas` is not in PATH; CI's cuda-feature lane
//! installs the toolkit and runs this file in its "PTX emitter ptxas
//! validation" step.

use std::io::Write;
use std::process::{Command, Stdio};

use nsl_codegen::ew_chain_fusion::{ChainSig, ChainStep, EwOpcode, Operand};
use nsl_codegen::fusion::synthesize_fused_chain_ptx;

fn find_ptxas() -> Option<String> {
    for name in ["ptxas", "ptxas.exe"] {
        if Command::new(name).arg("--version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok() {
            return Some(name.to_string());
        }
    }
    None
}

/// Assemble `ptx` (text, no NUL) for `sm_arch`; `Err` carries ptxas's
/// stderr.
fn assemble_ptx(ptxas_path: &str, ptx: &str, sm_arch: &str) -> Result<(), String> {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let cubin = std::env::temp_dir().join(format!(
        "nsl_ew_chain_{sm_arch}_{}_{}.cubin",
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
        .args(["--gpu-name", sm_arch, "-o"])
        .arg(&cubin)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to spawn ptxas: {e}"))?;
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(ptx.as_bytes())
        .map_err(|e| format!("failed to write PTX to ptxas stdin: {e}"))?;
    let out = child.wait_with_output().map_err(|e| format!("ptxas did not exit: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).into_owned())
    }
}

/// Every `sm_version` in `gpu_specs`' table that CUDA 13 still assembles
/// for, and the RTX 50 series.
const ARCHES: [u32; 8] = [75, 80, 86, 87, 89, 90, 100, 120];

fn step(op: EwOpcode, lhs: Operand, rhs: Option<Operand>) -> ChainStep {
    ChainStep { op, lhs, rhs }
}

fn sigs() -> Vec<ChainSig> {
    use EwOpcode::*;
    use Operand::{Imm, Input as I, Prev as P};
    vec![
        // The fuser's canonical chain: every opcode, an immediate, an unread slot.
        ChainSig {
            n_inputs: 4,
            steps: vec![
                step(Mul, I(0), Some(I(1))),
                step(RtsCheck, P(0), Some(I(2))),
                step(Add, P(1), Some(Imm(0x3F00_0000))),
                step(Neg, P(2), None),
                step(Div, P(3), Some(I(3))),
            ],
        },
        // The widest chain the fuser builds.
        ChainSig {
            n_inputs: 6,
            steps: vec![
                step(Mul, I(0), Some(I(1))),
                step(Add, P(0), Some(I(2))),
                step(Sub, P(1), Some(I(3))),
                step(Div, P(2), Some(I(4))),
                step(Mul, P(3), Some(I(5))),
                step(Add, P(4), Some(P(0))),
            ],
        },
        // One input; the result an RtsCheck's pass-through.
        ChainSig { n_inputs: 1, steps: vec![step(Sub, I(0), Some(Imm(0xBF80_0000))), step(RtsCheck, P(0), Some(I(0)))] },
    ]
}

#[test]
fn ptxas_accepts_the_chain_kernel_for_every_serving_arch() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("ptxas not found; skipping (install the CUDA toolkit to run this gate)");
        return;
    };
    for sig in sigs() {
        let bytes = synthesize_fused_chain_ptx(&sig, &sig.kernel_name());
        let ptx = std::str::from_utf8(&bytes[..bytes.len() - 1]).expect("ASCII");
        for sm in ARCHES {
            let arch = format!("sm_{sm}");
            if let Err(stderr) = assemble_ptx(&ptxas, ptx, &arch) {
                panic!("ptxas rejected the chain kernel for {arch} ({sig:?}):\n{stderr}\n--- PTX ---\n{ptx}");
            }
        }
    }
}
