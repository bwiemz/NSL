//! F2 STRONGEST lane-mapping test per Phase 2.5 spec §4.4.
//! This is the load_transposed-bug-class site: the test must verify the
//! `lane/4` n-column derivation that the original load_transposed bug
//! omitted (the term whose absence caused the all-32-lanes-same-first-byte
//! symptom).

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b2::backward::dq::synthesize_dq_kernel;

fn canonical_cfg() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64, block_kv: 64, head_dim: 32,
        causal: false, paged: false,
        rope_q: false, rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1, tree_mask: false,
        gpu_sm: 80, segment_masked: false,
        csha: Some(CshaExtras { level: 2, ..Default::default() }),
    }
}

fn canonical_hd128_cfg() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 32, block_kv: 32, head_dim: 128,
        causal: false, paged: false,
        rope_q: false, rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1, tree_mask: false,
        gpu_sm: 80, segment_masked: false,
        csha: Some(CshaExtras { level: 2, ..Default::default() }),
    }
}

#[test]
fn f2_kcol_restage_emits_real_ld_shared_b16() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    assert!(ptx.contains("ld.shared.b16"),
        "expected real ld.shared.b16 in K col-major re-stage (Path A source read)");
}

#[test]
fn f2_kcol_restage_emits_real_st_shared_b16() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    assert!(ptx.contains("st.shared.b16"),
        "expected real st.shared.b16 in K col-major re-stage (Path A col-major write)");
}

#[test]
fn f2_kcol_restage_all_four_address_terms_present() {
    // THE INSTITUTIONAL PIN test: assert all four of {lane%4, lane/4, smem_base, stride}
    // contribute to the scatter address.
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();

    // Term 1: lane%4 derivation (and.b32 ..., 3)
    let has_lane_mod_4 = ptx.lines().any(|l| {
        let t = l.trim();
        t.starts_with("and.b32") && t.contains("%lane_id") && t.ends_with(", 3;")
    });
    assert!(has_lane_mod_4,
        "F2 destination address must derive from lane%4 (and.b32 %... %lane_id, 3)");

    // Term 2: lane/4 derivation (shr.u32 ..., 2) -- THE LOAD_TRANSPOSED-BUG TERM
    let has_lane_div_4 = ptx.lines().any(|l| {
        let t = l.trim();
        (t.starts_with("shr.u32") || t.starts_with("shr.b32"))
            && t.contains("%lane_id") && t.ends_with(", 2;")
    });
    assert!(has_lane_div_4,
        "F2 destination address must derive from lane/4 (shr ..., 2) -- \
         this is the institutional pin from spec §5.5; the original \
         load_transposed bug omitted this term and produced all-32-lanes-\
         same-first-byte. Phase 2.5 must verify the term is present.");

    // Term 3: smem_base (the k_colmajor_offset constant)
    use nsl_codegen::flash_attention_v2::smem_layout::tier_b2_dq_k_colmajor_offset;
    let cfg = canonical_cfg();
    let kcol_off = tier_b2_dq_k_colmajor_offset(&cfg);
    assert!(ptx.contains(&format!("{kcol_off}")),
        "F2 must reference k_colmajor_offset {kcol_off} as the SMEM base for the col-major band");

    // Term 4: stride (bkv_eff * 2 for f16 col-major dst)
    use nsl_codegen::flash_attention_v2::smem_layout::tier_b2_effective_bkv;
    let bkv = tier_b2_effective_bkv(&cfg);
    let col_stride_bytes = bkv * 2;  // col-major dst stride between hd-cols
    assert!(ptx.contains(&format!("{col_stride_bytes}")),
        "F2 must reference col_stride_bytes {col_stride_bytes} (bkv*2) for col-major dst addressing");
}

#[test]
fn f2_kcol_restage_warp_0_gates_the_scatter() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    // Warp 0 gate present + DQ_KCOL_RESTAGE_DONE label present
    assert!(ptx.contains("@!%p_producer bra DQ_KCOL_RESTAGE_DONE"),
        "F2 scatter must be warp-0 gated (avoids 4x write amplification)");
    assert!(ptx.contains("DQ_KCOL_RESTAGE_DONE"),
        "F2 must have DQ_KCOL_RESTAGE_DONE label for the warp-0 gate target");
}

#[test]
fn f2_kcol_restage_bar_sync_after_warp_0_gate() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    // After the warp-0-gated scatter, all 4 warps must sync before MMA reads from k_colmajor band.
    // Find the LABEL (DQ_KCOL_RESTAGE_DONE:) not the bra reference, so the search starts
    // after the label itself (which ends the scatter body, not before it).
    let label = "DQ_KCOL_RESTAGE_DONE:";
    let kcol_done_idx = ptx.find(label)
        .expect("DQ_KCOL_RESTAGE_DONE: label must be present");
    let after = &ptx[kcol_done_idx + label.len()..];
    let next_200 = &after[..after.len().min(200)];
    assert!(next_200.contains("bar.sync"),
        "F2 must emit bar.sync immediately after DQ_KCOL_RESTAGE_DONE: label so col-major K is visible to all warps");
}

#[test]
fn f2_kcol_restage_scales_with_per_lane_pairs() {
    // At hd=128 bkv=32: pairs_per_lane = (32*128)/32 = 128
    // At hd=32 bkv=64: pairs_per_lane = (64*32)/32 = 64
    // The hd=128 config must emit more scatter pairs than hd=32.
    let ptx_32 = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    let ptx_128 = synthesize_dq_kernel(&canonical_hd128_cfg()).unwrap();

    let ld_32 = ptx_32.matches("ld.shared.b16").count();
    let ld_128 = ptx_128.matches("ld.shared.b16").count();

    // ld_128 should be at least double ld_32 (128 pairs vs 64 pairs per lane)
    assert!(ld_128 > ld_32,
        "F2 must emit more ld.shared.b16 at hd=128 (pairs=128) than hd=32 (pairs=64); got {ld_32} vs {ld_128}");
}

#[test]
fn f2_kcol_restage_ascii_only_ptx_invariant() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    for (i, byte) in ptx.bytes().enumerate() {
        assert!(byte.is_ascii(), "non-ASCII byte 0x{:02x} at position {}", byte, i);
    }
    let ptx_128 = synthesize_dq_kernel(&canonical_hd128_cfg()).unwrap();
    for (i, byte) in ptx_128.bytes().enumerate() {
        assert!(byte.is_ascii(), "non-ASCII byte 0x{:02x} at position {} (hd=128)", byte, i);
    }
}

#[test]
fn f2_kcol_restage_ptxas_clean() {
    // Reactivated ptxas smoke test for F2 -- the col-major K re-stage must
    // produce assembler-clean PTX, exercising the all-four-terms address chain.
    use std::process::Command;
    let cfg = canonical_cfg();
    let ptx = synthesize_dq_kernel(&cfg).unwrap();
    let tmp = std::env::temp_dir().join("f2_test.ptx");
    std::fs::write(&tmp, &ptx).unwrap();
    let result = Command::new("ptxas")
        .args(["-arch=sm_80", tmp.to_str().unwrap(), "-o"])
        .arg(if cfg!(windows) { "NUL" } else { "/dev/null" })
        .output();
    match result {
        Ok(out) if out.status.success() => {}
        Ok(out) => panic!(
            "ptxas failed:\nstdout: {}\nstderr: {}\nPTX file: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
            tmp.display(),
        ),
        Err(e) => eprintln!("[ptxas not on PATH, skipping]: {e}"),
    }
    // Also test hd=128
    let cfg_128 = canonical_hd128_cfg();
    let ptx_128 = synthesize_dq_kernel(&cfg_128).unwrap();
    let tmp_128 = std::env::temp_dir().join("f2_test_hd128.ptx");
    std::fs::write(&tmp_128, &ptx_128).unwrap();
    let result_128 = Command::new("ptxas")
        .args(["-arch=sm_80", tmp_128.to_str().unwrap(), "-o"])
        .arg(if cfg!(windows) { "NUL" } else { "/dev/null" })
        .output();
    match result_128 {
        Ok(out) if out.status.success() => {}
        Ok(out) => panic!(
            "ptxas failed (hd=128):\nstdout: {}\nstderr: {}\nPTX file: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
            tmp_128.display(),
        ),
        Err(e) => eprintln!("[ptxas not on PATH, skipping (hd=128)]: {e}"),
    }
}
