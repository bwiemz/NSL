//! Tier C — ptxas clean-assembly gate for backward PTX on
//! sm_75 / sm_90 / sm_120. Mirrors `csha_ptx_ptxas_validation.rs` but
//! synthesises the T3.1 backward prelude plus a trailing `ret` so
//! the kernel body is a legal (if trivial) PTX function.
//!
//! Full backward synth (`synthesize_backward`) lands in T4.1; until
//! then, this test composes the prelude emission with a minimal
//! body so Phase 3 phase emitters can be validated as they land
//! without waiting on the orchestrator.

#![cfg(feature = "test-helpers")]

use std::io::Write;
use std::process::{Command, Stdio};

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::phases::backward;

fn find_ptxas() -> Option<String> {
    if let Ok(p) = std::env::var("PTXAS") {
        if std::path::Path::new(&p).is_file() {
            return Some(p);
        }
    }
    for cand in [
        "ptxas",
        "ptxas.exe",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2/bin/ptxas.exe",
        "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/ptxas.exe",
        "/usr/local/cuda/bin/ptxas",
    ] {
        if Command::new(cand).arg("--version").output().is_ok() {
            return Some(cand.into());
        }
    }
    None
}

fn assemble_ptx(ptxas: &str, ptx: &[u8], sm: &str) -> Result<(), String> {
    let out = std::env::temp_dir().join(format!("bwd_prelude_{sm}.cubin"));
    let mut cmd = Command::new(ptxas)
        .arg(format!("--gpu-name={sm}"))
        .arg("-o").arg(&out)
        .arg("-")
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn ptxas: {e}"))?;
    cmd.stdin.as_mut().unwrap().write_all(ptx).map_err(|e| e.to_string())?;
    let fin = cmd.wait_with_output().map_err(|e| e.to_string())?;
    if !fin.status.success() {
        return Err(String::from_utf8_lossy(&fin.stderr).into_owned());
    }
    Ok(())
}

fn synth_prelude_with_ret(config: &FlashAttentionConfig) -> Vec<u8> {
    let mut ptx = String::new();
    backward::prelude::emit(&mut ptx, config);
    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");
    // ptxas wants NUL-terminated input when loaded via cuModuleLoadData,
    // but the stdin path does not require it. Keep trailing newline only.
    ptx.into_bytes()
}

fn synth_prelude_q_load_with_ret(config: &FlashAttentionConfig) -> Vec<u8> {
    let mut ptx = String::new();
    backward::prelude::emit(&mut ptx, config);
    backward::q_load::emit(&mut ptx, config, 0);
    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");
    ptx.into_bytes()
}

fn synth_prelude_qload_ds_with_ret(config: &FlashAttentionConfig) -> Vec<u8> {
    let mut ptx = String::new();
    backward::prelude::emit(&mut ptx, config);
    backward::q_load::emit(&mut ptx, config, 0);
    backward::ds_compute::emit(&mut ptx, config, 0);
    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");
    ptx.into_bytes()
}

#[test]
fn backward_prelude_ptxas_clean_sm75_sm90_sm120() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("skipping: ptxas not found in PATH / standard locations");
        return;
    };

    let base = FlashAttentionConfig {
        block_q: 32, block_kv: 32, head_dim: 32,
        causal: false, paged: false, rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1, tree_mask: false, gpu_sm: 75,
        csha: Some(CshaExtras {
            fused_projections: true,
            save_activations_for_backward: true,
            d_model: 32,
            ..CshaExtras::default()
        }),
    };

    let mut failures = Vec::new();
    for sm in &["sm_75", "sm_90", "sm_120"] {
        let mut c = base.clone();
        c.gpu_sm = sm.trim_start_matches("sm_").parse().unwrap_or(75);
        let ptx = synth_prelude_with_ret(&c);
        let dump = std::env::temp_dir().join(format!("bwd_prelude_{sm}.ptx"));
        std::fs::write(&dump, &ptx).ok();
        if let Err(err) = assemble_ptx(&ptxas, &ptx, sm) {
            failures.push(format!(
                "sm={sm} dump={} ptxas:\n{err}",
                dump.display()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "backward prelude ptxas failures:\n{}",
        failures.join("\n---\n")
    );
}

#[test]
fn backward_prelude_plus_q_load_ptxas_clean_sm75_sm90_sm120() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("skipping: ptxas not found");
        return;
    };
    let base = FlashAttentionConfig {
        block_q: 32, block_kv: 32, head_dim: 32,
        causal: false, paged: false, rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1, tree_mask: false, gpu_sm: 75,
        csha: Some(CshaExtras {
            fused_projections: true,
            save_activations_for_backward: true,
            d_model: 32,
            ..CshaExtras::default()
        }),
    };

    let mut failures = Vec::new();
    for sm in &["sm_75", "sm_90", "sm_120"] {
        let mut c = base.clone();
        c.gpu_sm = sm.trim_start_matches("sm_").parse().unwrap_or(75);
        let ptx = synth_prelude_q_load_with_ret(&c);
        let dump = std::env::temp_dir().join(format!("bwd_prelude_qload_{sm}.ptx"));
        std::fs::write(&dump, &ptx).ok();
        if let Err(err) = assemble_ptx(&ptxas, &ptx, sm) {
            failures.push(format!(
                "sm={sm} dump={} ptxas:\n{err}",
                dump.display()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "backward prelude+q_load ptxas failures:\n{}",
        failures.join("\n---\n")
    );
}

#[test]
fn backward_prelude_qload_ds_compute_ptxas_clean_sm75_sm90_sm120() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("skipping: ptxas not found");
        return;
    };
    for (causal, tag) in [(false, "nocausal"), (true, "causal")] {
        let base = FlashAttentionConfig {
            block_q: 32, block_kv: 32, head_dim: 32,
            causal, paged: false, rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1, tree_mask: false, gpu_sm: 75,
            csha: Some(CshaExtras {
                fused_projections: true,
                save_activations_for_backward: true,
                d_model: 32,
                ..CshaExtras::default()
            }),
        };
        let mut failures = Vec::new();
        for sm in &["sm_75", "sm_90", "sm_120"] {
            let mut c = base.clone();
            c.gpu_sm = sm.trim_start_matches("sm_").parse().unwrap_or(75);
            let ptx = synth_prelude_qload_ds_with_ret(&c);
            let dump = std::env::temp_dir()
                .join(format!("bwd_prelude_qload_ds_{tag}_{sm}.ptx"));
            std::fs::write(&dump, &ptx).ok();
            if let Err(err) = assemble_ptx(&ptxas, &ptx, sm) {
                failures.push(format!("tag={tag} sm={sm} dump={} ptxas:\n{err}",
                    dump.display()));
            }
        }
        assert!(
            failures.is_empty(),
            "backward prelude+qload+ds ({tag}) ptxas failures:\n{}",
            failures.join("\n---\n")
        );
    }
}
