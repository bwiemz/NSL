//! PCA §4.3 — T5+T6 RoPE-reset PTX emission tests.
//!
//! Verifies the CTA-prologue SMEM load and supporting register
//! declarations only appear when `segment_masked && rope_q`. Sentinel
//! (segment_masked=false) callers MUST remain byte-stable.

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::{
    flash_attention_kernel_name_v2, synthesize_backward, synthesize_flash_attention_ptx_v2,
};

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
        num_sink_tokens: 0,
        gpu_sm: 75,
        segment_masked: true,
        csha: Some(CshaExtras::level2(1e-5, 32)),
        checkpoint: None,
    }
}

/// Backward-capable variant: same gating as the forward fixture, but with
/// `save_activations_for_backward=true` so `synthesize_backward` accepts
/// the config (the backward path reads from the saved post-RoPE Q/K/V).
fn config_segment_masked_with_rope_for_backward() -> FlashAttentionConfig {
    let mut cfg = config_segment_masked_with_rope();
    if let Some(csha) = cfg.csha.as_mut() {
        csha.save_activations_for_backward = true;
    }
    cfg
}

fn ptx_string(cfg: &FlashAttentionConfig) -> String {
    String::from_utf8(synthesize_flash_attention_ptx_v2(cfg))
        .expect("PTX must be valid UTF-8")
}

fn backward_ptx_string(cfg: &FlashAttentionConfig) -> String {
    synthesize_backward(cfg).expect("backward synthesis must succeed")
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

// ---------------------------------------------------------------------------
// T7 + T8 — forward Q+K rotation sites consume the doc_starts SMEM table.
//
// effective_pos_q / effective_pos_k are computed inside emit_rope_pair_sweep
// (inside the per-pair loop) when the gate is active. cs_idx (cos/sin table
// lookup) is rerouted through the effective_pos register; %r_rope_row stays
// as the tile-local row for SMEM addressing.
//
// Sentinel-disabled (segment_masked=false) PTX MUST be byte-stable: no
// effective_pos register references, no sub.s32 against doc_start.
// ---------------------------------------------------------------------------

#[test]
fn forward_q_rotation_computes_effective_pos_when_doc_starts_active() {
    let cfg = config_segment_masked_with_rope();
    let ptx = ptx_string(&cfg);
    assert!(
        ptx.contains("%r_effective_pos_q"),
        "Forward Q rotation must compute effective_pos_q"
    );
    assert!(
        ptx.contains("sub.s32 %r_effective_pos_q"),
        "Forward Q rotation must use sub.s32 to compute effective_pos_q from q_pos and doc_start"
    );
}

#[test]
fn forward_k_rotation_computes_effective_pos_when_doc_starts_active() {
    let cfg = config_segment_masked_with_rope();
    let ptx = ptx_string(&cfg);
    assert!(
        ptx.contains("%r_effective_pos_k"),
        "Forward K rotation must compute effective_pos_k"
    );
    assert!(
        ptx.contains("sub.s32 %r_effective_pos_k"),
        "Forward K rotation must use sub.s32 to compute effective_pos_k from kv_pos and doc_start"
    );
}

#[test]
fn forward_rotation_sites_skip_effective_pos_when_disabled() {
    let mut cfg = config_segment_masked_with_rope();
    cfg.segment_masked = false;
    let ptx = ptx_string(&cfg);
    assert!(
        !ptx.contains("%r_effective_pos_q") && !ptx.contains("%r_effective_pos_k"),
        "Sentinel-disabled path must NOT use effective_pos registers"
    );
}

// ---------------------------------------------------------------------------
// Non-CSHA inline q_load.rs path — when csha is None (or fused_projections is
// false), the inline RoPE applies via emit_rope_rotation_inline in q_load.rs.
// PCA §4.3 site 1 reroutes cos/sin row through effective_pos_q just like the
// CSHA fused path does (csha_hooks.rs). Sentinel paths stay byte-stable.
// ---------------------------------------------------------------------------

fn config_segment_masked_with_rope_no_csha() -> FlashAttentionConfig {
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
        num_sink_tokens: 0,
        gpu_sm: 75,
        segment_masked: true,
        checkpoint: None,
        csha: None,
    }
}

#[test]
fn inline_q_load_computes_effective_pos_when_reset_active_and_csha_off() {
    let cfg = config_segment_masked_with_rope_no_csha();
    let ptx = ptx_string(&cfg);
    assert!(
        ptx.contains("// PCA sec.4.3 site 1: forward Q effective_pos"),
        "Inline q_load path must emit the site 1 marker when segment_masked && rope_q && csha=None"
    );
    assert!(
        ptx.contains("sub.s32 %r_effective_pos_q"),
        "Inline q_load path must compute effective_pos_q via sub.s32"
    );
    assert!(
        ptx.contains("mul.lo.u64 %rd27, %rd27, "),
        "Inline q_load path must reroute the cos/sin row through %rd27 (effective_pos*head_dim)"
    );
}

#[test]
fn inline_q_load_skips_effective_pos_when_segment_masked_off() {
    let mut cfg = config_segment_masked_with_rope_no_csha();
    cfg.segment_masked = false;
    let ptx = ptx_string(&cfg);
    assert!(
        !ptx.contains("// PCA sec.4.3 site 1: forward Q effective_pos"),
        "Inline q_load path must NOT emit the site 1 marker when segment_masked=false"
    );
    assert!(
        !ptx.contains("%r_effective_pos_q"),
        "Inline q_load path must NOT reference effective_pos_q when segment_masked=false"
    );
}

#[test]
fn inline_q_load_ptx_assembles_on_sm75_and_sm120() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("ptxas not found — skipping");
        return;
    };
    let cfg = config_segment_masked_with_rope_no_csha();
    let ptx = synthesize_flash_attention_ptx_v2(&cfg);
    for sm in ["sm_75", "sm_120"] {
        if let Err(stderr) = assemble_ptx(&ptxas, &ptx, sm) {
            panic!("ptxas rejected inline q_load PTX for {sm}:\n{stderr}");
        }
    }
}

// ---------------------------------------------------------------------------
// T9 — backward dQ + dK de-rotation sites consume the doc_starts SMEM table.
//
// Inverse-rotation math:
//   dx0 =  dY[2i]*cos + dY[2i+1]*sin
//   dx1 = -dY[2i]*sin + dY[2i+1]*cos
// The cs_idx (cos/sin lookup) is identical to the forward — only the sign
// of `sin` flips when applying. Therefore the effective_pos formula must be
// bit-identical to T7+T8: effective_pos = abs_row - smem_doc_starts[seg_id].
//
// Sentinel-disabled (segment_masked=false) backward PTX MUST be byte-stable:
// no effective_pos refs, no smem_doc_starts decl/load.
// ---------------------------------------------------------------------------

#[test]
fn backward_dq_de_rotation_computes_effective_pos_when_doc_starts_active() {
    let cfg = config_segment_masked_with_rope_for_backward();
    let ptx = backward_ptx_string(&cfg);
    assert!(
        ptx.contains("%r_effective_pos_q"),
        "Backward dQ de-rotation must compute effective_pos_q"
    );
}

#[test]
fn backward_dk_de_rotation_computes_effective_pos_when_doc_starts_active() {
    let cfg = config_segment_masked_with_rope_for_backward();
    let ptx = backward_ptx_string(&cfg);
    assert!(
        ptx.contains("%r_effective_pos_k"),
        "Backward dK de-rotation must compute effective_pos_k"
    );
}

#[test]
fn backward_loads_doc_starts_to_smem_when_segment_masked_and_rope_q() {
    let cfg = config_segment_masked_with_rope_for_backward();
    let ptx = backward_ptx_string(&cfg);
    assert!(
        ptx.contains("V2_PCA_ROPE_DOC_STARTS_LOAD_LOOP")
            || ptx.contains("smem_doc_starts"),
        "Backward CTA prologue must populate smem_doc_starts \
         (either via emit_doc_starts_smem_load or equivalent)"
    );
}

#[test]
fn backward_rope_reset_enabled_ptx_assembles_on_sm75_and_sm120() {
    let Some(ptxas) = find_ptxas() else {
        eprintln!("ptxas not found — skipping");
        return;
    };
    let cfg = config_segment_masked_with_rope_for_backward();
    let ptx = backward_ptx_string(&cfg);
    for sm in ["sm_75", "sm_120"] {
        if let Err(stderr) = assemble_ptx(&ptxas, ptx.as_bytes(), sm) {
            panic!("ptxas rejected backward PTX for {sm}:\n{stderr}");
        }
    }
}

#[test]
fn backward_disabled_path_has_no_effective_pos_or_smem_load() {
    let mut cfg = config_segment_masked_with_rope_for_backward();
    cfg.segment_masked = false;
    let ptx = backward_ptx_string(&cfg);
    assert!(
        !ptx.contains("%r_effective_pos_q") && !ptx.contains("%r_effective_pos_k"),
        "Backward sentinel-disabled path must NOT use effective_pos"
    );
    assert!(
        !ptx.contains("smem_doc_starts"),
        "Backward sentinel-disabled path must NOT load smem_doc_starts"
    );
}

// ---------------------------------------------------------------------------
// T10 — kernel-name suffix `_rope_reset_max256` differentiates the live-
// activation variant from the sentinel-disabled variant. Without this suffix
// cached PTX bytes would collide between the two variants on a single binary.
//
// Backward kernel name derives from the forward via
// `flash_attention_v2::phases::backward::prelude::kernel_name` (it strip-
// prefixes `flash_attn_` → `flash_attn_backward_`), so the suffix inherits
// automatically — but we still assert on the backward path to lock that in.
// ---------------------------------------------------------------------------

#[test]
fn kernel_name_has_rope_reset_max256_suffix_when_segment_masked_and_rope_q() {
    let cfg = config_segment_masked_with_rope();
    let name = flash_attention_kernel_name_v2(&cfg);
    assert!(
        name.contains("rope_reset_max256"),
        "Kernel name {} must include rope_reset_max256 suffix when segment_masked && rope_q",
        name
    );
}

#[test]
fn kernel_name_omits_rope_reset_suffix_when_segment_masked_off() {
    let mut cfg = config_segment_masked_with_rope();
    cfg.segment_masked = false;
    let name = flash_attention_kernel_name_v2(&cfg);
    assert!(
        !name.contains("rope_reset_max"),
        "Kernel name {} must NOT include rope_reset suffix when segment_masked=false",
        name
    );
}

#[test]
fn kernel_name_omits_rope_reset_suffix_when_rope_q_off() {
    let mut cfg = config_segment_masked_with_rope();
    cfg.rope_q = false;
    let name = flash_attention_kernel_name_v2(&cfg);
    assert!(
        !name.contains("rope_reset_max"),
        "Kernel name {} must NOT include rope_reset suffix when rope_q=false",
        name
    );
}

#[test]
fn backward_kernel_name_inherits_rope_reset_suffix() {
    // The backward kernel name is derived from the forward name (strip
    // `flash_attn_` → `flash_attn_backward_`), so the rope_reset suffix
    // must appear in the backward variant too — proving the two kernels
    // don't collide in the module cache.
    let cfg = config_segment_masked_with_rope_for_backward();
    let fwd_name = flash_attention_kernel_name_v2(&cfg);
    assert!(
        fwd_name.contains("rope_reset_max256"),
        "Forward name {} must carry the suffix for the backward to inherit it",
        fwd_name
    );
    // Sanity: the backward PTX text should contain a kernel-entry name that
    // also carries the suffix. (The backward `kernel_name` helper is private
    // to the backward module, so we check by scanning the synthesized PTX.)
    let ptx = backward_ptx_string(&cfg);
    let entry_line_count = ptx
        .lines()
        .filter(|l| l.contains(".visible .entry") && l.contains("rope_reset_max256"))
        .count();
    assert!(
        entry_line_count >= 1,
        "Backward PTX must declare a `.visible .entry` whose name contains \
         rope_reset_max256; found {} matching entry lines",
        entry_line_count
    );
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
