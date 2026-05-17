//! PCA §4.3 — T5+T6 RoPE-reset PTX emission tests.
//!
//! Verifies the CTA-prologue SMEM load and supporting register
//! declarations only appear when `segment_masked && rope_q`. Sentinel
//! (segment_masked=false) callers MUST remain byte-stable.

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::synthesize_flash_attention_ptx_v2;

fn config_segment_masked_with_rope() -> FlashAttentionConfig {
    // Use the CSHA L2 RoPE preset matching the existing ptxas-validated
    // `csha_l2_rope_ptx_assembles_on_sm120` test, but flip `segment_masked`
    // on. CshaExtras::default() leaves smem-base registers undeclared, so
    // the CSHA RoPE-Q epilogue's references to `%q_smem_base` fail ptxas;
    // L2 (fused_projections=true, fused_output_proj=true) sets that up.
    FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 32,
        causal: true,
        paged: false,
        rope_q: true,
        rope_style: RopeStyle::Adjacent,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 75,
        segment_masked: true,
        csha: Some(CshaExtras::level2(1e-5, 32)),
    }
}

fn ptx_string(cfg: &FlashAttentionConfig) -> String {
    String::from_utf8(synthesize_flash_attention_ptx_v2(cfg))
        .expect("PTX must be valid UTF-8")
}

#[test]
fn rope_reset_enabled_declares_registers() {
    let cfg = config_segment_masked_with_rope();
    let ptx = ptx_string(&cfg);
    assert!(
        ptx.contains("%r_effective_pos_q") || ptx.contains("%r_effective_pos_k"),
        "RoPE-reset effective_pos registers must be declared"
    );
}

#[test]
fn rope_reset_enabled_declares_doc_starts_param() {
    let cfg = config_segment_masked_with_rope();
    let ptx = ptx_string(&cfg);
    assert!(
        ptx.contains(".param .u64 doc_starts_ptr"),
        "PTX kernel signature must declare doc_starts_ptr param"
    );
}

#[test]
fn rope_reset_enabled_cta_prologue_loads_doc_starts_to_smem() {
    let cfg = config_segment_masked_with_rope();
    let ptx = ptx_string(&cfg);
    assert!(
        ptx.contains(".shared .align 4 .b8 smem_doc_starts[1028]"),
        "PTX must declare a 1028-byte SMEM region for doc_starts"
    );
    assert!(
        ptx.contains("V2_PCA_ROPE_DOC_STARTS_LOAD_LOOP:"),
        "PTX must include the doc_starts SMEM load loop"
    );
    assert!(
        ptx.contains("mul.lo.u32 %r_row_offset_elems"),
        "Prologue must compute per-row offset"
    );
}

#[test]
fn rope_reset_disabled_does_not_emit_prologue() {
    let mut cfg = config_segment_masked_with_rope();
    cfg.segment_masked = false;
    let ptx = ptx_string(&cfg);
    assert!(
        !ptx.contains("smem_doc_starts"),
        "PTX must NOT declare smem_doc_starts when segment_masked=false"
    );
    assert!(
        !ptx.contains("V2_PCA_ROPE_DOC_STARTS_LOAD_LOOP"),
        "PTX must NOT emit the load loop when segment_masked=false"
    );
    assert!(
        !ptx.contains(".param .u64 doc_starts_ptr"),
        "PTX must NOT declare doc_starts_ptr param when segment_masked=false"
    );
}

// ---------------------------------------------------------------------------
// ptxas integration check — assembles the emitted PTX on sm_75 / sm_120.
//
// Locates ptxas via $PATH or the default Windows CUDA path. Skipped (with a
// stderr note) when neither is found, so it stays green on dev boxes without
// CUDA. On a CUDA-equipped box, this is the one e2e gate that proves the
// T5+T6 emission is syntactically valid PTX.
// ---------------------------------------------------------------------------

fn find_ptxas() -> Option<String> {
    use std::process::{Command, Stdio};
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
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut child = Command::new(ptxas_path)
        .args(["--gpu-name", sm_arch, "-O0", "-o", "-", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn ptxas failed: {e}"))?;
    child
        .stdin
        .as_mut()
        .ok_or("ptxas stdin closed")?
        .write_all(ptx_bytes)
        .map_err(|e| format!("write ptx failed: {e}"))?;
    let out = child.wait_with_output().map_err(|e| format!("ptxas wait: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).into_owned())
    }
}

#[test]
fn rope_reset_enabled_ptx_assembles_on_sm75_and_sm120() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("ptxas not found — skipping");
        return;
    };
    let cfg = config_segment_masked_with_rope();
    let ptx = synthesize_flash_attention_ptx_v2(&cfg);
    for sm in ["sm_75", "sm_120"] {
        if let Err(stderr) = assemble_ptx(&ptxas, &ptx, sm) {
            panic!("ptxas rejected PTX for {sm}:\n{stderr}");
        }
    }
}

#[test]
#[ignore]
fn dump_rope_reset_ptx() {
    let cfg = config_segment_masked_with_rope();
    let ptx = ptx_string(&cfg);
    // Write the FULL PTX to a fixed temp path for inspection.
    let path = std::env::temp_dir().join("rope_reset_full.ptx");
    let _ = std::fs::write(&path, &ptx);
    eprintln!("wrote full PTX to {}", path.display());
    eprintln!("--- doc_starts section ---");
    let mut in_section = false;
    for line in ptx.lines() {
        if line.contains("doc_starts_ptr") || line.contains("smem_doc_starts")
            || line.contains("V2_PCA_ROPE_DOC_STARTS") || line.contains("RoPE-reset")
            || line.contains("CTA prologue") || line.contains("doc_starts to SMEM") {
            in_section = true;
        }
        if in_section {
            println!("{}", line);
        }
        if in_section && line.contains("end PCA Tier A segment_ids load") {
            in_section = false;
        }
    }
}
