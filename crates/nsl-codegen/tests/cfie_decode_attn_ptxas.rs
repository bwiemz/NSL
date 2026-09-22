//! Roadmap A2 step 9: the KIR decode-attention module must assemble, for
//! every architecture the serving GPU table names.
//!
//! The KIR module targets the backend floor (`.version 7.0` /
//! `.target sm_70`) whatever GPU it is compiled for, and the driver
//! JIT-compiles it forward. This gate assembles it with
//! `ptxas --gpu-name sm_XX` for each of those architectures — the offline
//! form of that JIT — over the geometries the module's own tests sweep.
//!
//! It also records why the header changed. The hand kernel paired
//! `.target sm_{N}` with the ISA `gpu_specs::GpuSpec::ptx_version` picks
//! for `N`: 7.0 below sm_90, 8.4 below sm_100, 8.6 above. PTX introduced
//! sm_86 in ISA 7.1, sm_87 in 7.4, sm_89 in 7.8 and sm_120 in 8.7, so for
//! those parts the hand module named a target its own `.version` cannot.
//! The frozen
//! hand emitter is assembled here for each, and must be refused — while
//! the same body under a header its ISA can name (sm_80, sm_90, sm_100) is
//! accepted, so the refusal is the header's and not the kernel's.
//!
//! Skipped with a note when `ptxas` is not in PATH; CI's cuda-feature lane
//! installs the toolkit and runs this file in its "PTX emitter ptxas
//! validation" step. Same harness as `kir_block_params_ptxas.rs` (CUDA 13
//! no longer takes `sm_70` as a `--gpu-name`, so the floor module is
//! assembled for the real parts only).

use std::io::Write;
use std::process::{Command, Stdio};

use nsl_codegen::cfie_decode_attention::{emit_decode_attention_ptx, DecodeAttentionConfig};

#[allow(dead_code)]
#[path = "fixtures/cfie_decode_attn_hand.rs"]
mod hand;

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
        "nsl_cfie_decode_attn_{sm_arch}_{}_{}.cubin",
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

/// Every `sm_version` in `gpu_specs`' table that CUDA 13 still assembles
/// for.
const SERVING_ARCHES: [u32; 7] = [75, 80, 86, 87, 89, 90, 100];

/// Blackwell consumer parts (the RTX 50 series). Kept separate: it is the
/// development target, and the case the hand header got wrong on it.
const SM_120: u32 = 120;

fn geometries() -> Vec<DecodeAttentionConfig> {
    [(8, 8, 4, 128, 2048, 64), (1, 6, 2, 40, 300, 3), (3, 4, 1, 1, 129, 2)]
        .into_iter()
        .map(|(n_layers, n_heads, n_kv_heads, head_dim, per_slot, slots)| DecodeAttentionConfig {
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            per_slot_max_tokens: per_slot,
            max_slots: slots,
            kv_dtype_bytes: 2,
        })
        .collect()
}

fn hand_cfg(cfg: &DecodeAttentionConfig, sm_version: u32) -> hand::DecodeAttentionConfig {
    hand::DecodeAttentionConfig {
        n_layers: cfg.n_layers,
        n_heads: cfg.n_heads,
        n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim,
        per_slot_max_tokens: cfg.per_slot_max_tokens,
        max_slots: cfg.max_slots,
        kv_dtype_bytes: cfg.kv_dtype_bytes,
        sm_version,
    }
}

#[test]
fn ptxas_accepts_the_kir_decode_attention_module_for_every_serving_arch() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("ptxas not found; skipping (install the CUDA toolkit to run this gate)");
        return;
    };
    for cfg in geometries() {
        let ptx = emit_decode_attention_ptx(&cfg);
        for sm in SERVING_ARCHES.into_iter().chain([SM_120]) {
            let arch = format!("sm_{sm}");
            if let Err(stderr) = assemble_ptx(&ptxas, &ptx, &arch) {
                panic!("ptxas rejected the KIR decode-attention module for {arch} ({cfg:?}):\n{stderr}\n--- PTX ---\n{ptx}");
            }
        }
    }
    println!("ptxas accepted the KIR decode-attention module (sm_75..sm_120)");
}

#[test]
fn the_hand_header_named_targets_its_own_isa_could_not() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("ptxas not found; skipping (install the CUDA toolkit to run this gate)");
        return;
    };
    let cfg = &geometries()[0];
    // Control: where `ptx_version` names an ISA that has the target, the
    // hand module assembles — so the body is sound and a refusal below is
    // the header's.
    for sm in [80, 90, 100] {
        let ptx = hand::emit_decode_attention_ptx(&hand_cfg(cfg, sm));
        if let Err(stderr) = assemble_ptx(&ptxas, &ptx, &format!("sm_{sm}")) {
            panic!("control: ptxas rejected the hand module for sm_{sm}:\n{stderr}");
        }
    }
    // sm_86 needs ISA 7.1, sm_87 7.4, sm_89 7.8, sm_120 8.7; the hand
    // header gave them 7.0, 7.0, 7.0 and 8.6.
    for sm in [86, 87, 89, SM_120] {
        let ptx = hand::emit_decode_attention_ptx(&hand_cfg(cfg, sm));
        let header: Vec<&str> = ptx.lines().filter(|l| l.starts_with(".version") || l.starts_with(".target")).collect();
        match assemble_ptx(&ptxas, &ptx, &format!("sm_{sm}")) {
            Ok(()) => panic!("ptxas accepted the hand header {header:?}; the KIR module's docs are wrong"),
            Err(stderr) => println!("hand header {header:?} refused as expected: {}", stderr.trim()),
        }
    }
}
