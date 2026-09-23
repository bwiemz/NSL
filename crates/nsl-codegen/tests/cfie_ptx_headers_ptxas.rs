//! The CFIE hand-written emitters must assemble for every architecture the
//! serving GPU table names.
//!
//! Each of these emitters writes `.target sm_{N}` for the serving GPU with
//! the ISA `gpu_specs::ptx_isa_for_sm(N)` picks. Until that table existed
//! each carried its own copy of a three-row one (7.0 below sm_90, 8.4 below
//! sm_100, 8.6 above), and `ptxas` refused the result for sm_86, sm_87 and
//! sm_89 (`PTX .version 7.0 does not support .target sm_89`) and for
//! sm_120 (`.version 8.6 does not support .target sm_120`) — the RTX 30
//! and 40 series, Jetson Orin, and the RTX 50 series. The driver's JIT
//! makes the same check when it loads the module.
//!
//! This gate builds every kernel of the five emitters at every `sm_version`
//! in `gpu_specs::GPU_DATABASE` that CUDA 13 still assembles for, and
//! assembles each with `ptxas --gpu-name sm_{N}`.
//!
//! `cfie_kv_quant_ptx` has since moved onto KIR (roadmap A2 step 9): its
//! per-layer kernels target the KIR floor (`sm_70`) whatever GPU serves
//! them and take no `sm_version`, so the gate builds them once and still
//! assembles them for every architecture — the driver JIT-compiles them
//! forward, and this is the offline form of that.
//!
//! Skipped with a note when `ptxas` is not in PATH; CI's cuda-feature lane
//! installs the toolkit and runs this file in its "PTX emitter ptxas
//! validation" step. Same harness as `kir_block_params_ptxas.rs`.

use std::io::Write;
use std::process::{Command, Stdio};

use nsl_codegen::cfie_fused_sample::{emit_program, LmHeadShape, SamplingParams};
use nsl_codegen::cfie_kv_quant::KvPrecision;
use nsl_codegen::gpu_specs::GPU_DATABASE;
use nsl_codegen::{
    cfie_kv_quant_ptx, cfie_persistent_ptx, cfie_sample_ptx, cfie_spec_sampler_ptx,
    cfie_speculative_ptx,
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

/// Assemble `ptx` (text, no NUL) for `sm_arch`; `Err` carries ptxas's
/// stderr.
fn assemble_ptx(ptxas_path: &str, ptx: &str, sm_arch: &str) -> Result<(), String> {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let cubin = std::env::temp_dir().join(format!(
        "nsl_cfie_headers_{sm_arch}_{}_{}.cubin",
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

/// Every distinct `sm_version` in the GPU table, sorted. CUDA 13 dropped
/// the architectures below sm_75, and the table has none.
fn serving_arches() -> Vec<u32> {
    let mut sms: Vec<u32> = GPU_DATABASE.iter().map(|g| g.sm_version).collect();
    sms.sort_unstable();
    sms.dedup();
    assert!(sms.iter().all(|&sm| sm >= 75), "{sms:?}");
    sms
}

/// `(kernel, PTX)` for every kernel the five emitters build, at `sm`, over
/// the paper-shaped configurations their own tests use.
fn modules(sm: u32) -> Vec<(String, String)> {
    let mut out = Vec::new();

    let kv = cfie_kv_quant_ptx::QuantDecodeAttentionConfig {
        n_layers: 2,
        n_heads: 8,
        n_kv_heads: 4,
        head_dim: 64,
        per_slot_max_tokens: 256,
        max_slots: 4,
        layer_precisions: vec![(KvPrecision::Fp16, KvPrecision::Int8), (KvPrecision::Int8, KvPrecision::Int8)],
    };
    for (ptx, meta) in cfie_kv_quant_ptx::emit_all(&kv) {
        out.push((meta.kernel_name, ptx));
    }

    let shape = LmHeadShape { d_model: 512, vocab_size: 49_152, vocab_tile: 128, dtype_bytes: 2 };
    let sample = cfie_sample_ptx::FusedSampleKernelConfig {
        d_model: 512,
        vocab_size: 49_152,
        vocab_tile: 128,
        top_k: 50,
        sm_version: sm,
        grammar_states: 0,
    };
    let (ptx, meta) = cfie_sample_ptx::emit(&emit_program(SamplingParams::default(), shape), &sample);
    out.push((meta.kernel_name, ptx));

    let block = cfie_persistent_ptx::DecodeBlockConfig {
        d_model: 512,
        head_dim: 64,
        n_heads: 8,
        n_kv_heads: 4,
        d_ff: 1408,
        per_slot_max_tokens: 2048,
        max_slots: 64,
        n_layers: 8,
        rope_theta: 10000.0,
        eps: 1e-5,
        sm_version: sm,
    };
    let (ptx, meta) = cfie_persistent_ptx::emit(&block);
    out.push((meta.kernel_name, ptx));

    let spec = cfie_spec_sampler_ptx::SpecSamplerConfig {
        d_model: 512,
        vocab_size: 49_152,
        vocab_tile: 128,
        sm_version: sm,
    };
    let (ptx, meta) = cfie_spec_sampler_ptx::emit_draft_sample(&spec);
    out.push((meta.kernel_name, ptx));
    let (ptx, meta) = cfie_spec_sampler_ptx::emit_verify_probs(&spec);
    out.push((meta.kernel_name, ptx));

    let verify = cfie_speculative_ptx::VerifyAttentionConfig {
        n_heads: 8,
        n_kv_heads: 4,
        head_dim: 128,
        per_slot_max_tokens: 2048,
        max_slots: 64,
        num_nodes: 6,
        // A chain: node r attends itself and every earlier node.
        mask_bits: (0..6).map(|r| (1u64 << (r + 1)) - 1).collect(),
        sm_version: sm,
    };
    let (ptx, meta) = cfie_speculative_ptx::emit_verify_attention(&verify);
    out.push((meta.kernel_name, ptx));
    let reject = cfie_speculative_ptx::RejectionConfig { k_tokens: 5, vocab_size: 49_152, sm_version: sm };
    let (ptx, meta) = cfie_speculative_ptx::emit_rejection_kernel(&reject);
    out.push((meta.kernel_name, ptx));

    out
}

#[test]
fn ptxas_accepts_every_cfie_hand_emitter_for_every_serving_arch() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("ptxas not found; skipping (install the CUDA toolkit to run this gate)");
        return;
    };
    let arches = serving_arches();
    let mut assembled = 0;
    for &sm in &arches {
        for (kernel, ptx) in modules(sm) {
            let header: Vec<&str> =
                ptx.lines().filter(|l| l.starts_with(".version") || l.starts_with(".target")).collect();
            assert_eq!(header.len(), 2, "{kernel} at sm_{sm}: one .version and one .target");
            if let Err(stderr) = assemble_ptx(&ptxas, &ptx, &format!("sm_{sm}")) {
                panic!("ptxas rejected {kernel} for sm_{sm} (header {header:?}):\n{stderr}");
            }
            assembled += 1;
        }
    }
    println!("ptxas accepted {assembled} CFIE modules across {arches:?}");
}
