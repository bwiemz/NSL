//! G1 lane-mapping test per Phase 2.5 spec §4.4.
//! dQ HBM finalize: each lane scatters 4 f32 dq_acc values per fragment
//! to row-major [B,H,S,D] dQ output via emit_4d_byte_offset chain.

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b2::backward::dq::synthesize_dq_kernel;

fn canonical_cfg() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 32,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 80,
        segment_masked: false,
        csha: Some(CshaExtras { level: 2, ..Default::default() }),
    }
}

#[test]
fn g1_finalize_emits_real_st_global_f32() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    let st_count = ptx.matches("st.global.f32").count();
    // At hd=32 with 1 accumulator fragment (32/32), 4 stores per lane per fragment = 4 emissions
    assert!(
        st_count >= 4,
        "expected at least 4 st.global.f32 emissions for dq_acc finalize, got {st_count}"
    );
}

#[test]
fn g1_finalize_references_d_q_out_ptr() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    assert!(
        ptx.contains("d_q_out_ptr"),
        "G1 must load dQ HBM base from d_q_out_ptr param"
    );
}

#[test]
fn g1_finalize_uses_mul_wide_for_f32_byte_offset() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    // f32 sizeof=4: byte offset uses mul.wide.u32 ..., 4
    let has_mul_wide_4 = ptx.lines().any(|l| {
        let t = l.trim();
        t.starts_with("mul.wide.u32") && t.ends_with(", 4;")
    });
    assert!(
        has_mul_wide_4,
        "G1 must use mul.wide.u32 with sizeof=4 for f32 dQ byte offset (via emit_4d_byte_offset)"
    );
}

#[test]
fn g1_finalize_address_chain_has_lane_terms() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    // dQ row = q_iter * bq + (lane_id / 4 [+ 8 for hi half])
    // dQ col = fragment_col_base + (lane_id % 4) * 2 [+ 1 for hi col]
    // So the chain must reference both lane%4 and lane/4
    let has_lane_mod_4 = ptx.lines().any(|l| {
        let t = l.trim();
        t.starts_with("and.b32") && t.contains("%lane_id") && t.ends_with(", 3;")
    });
    let has_lane_div_4 = ptx.lines().any(|l| {
        let t = l.trim();
        (t.starts_with("shr.u32") || t.starts_with("shr.b32"))
            && t.contains("%lane_id")
            && t.ends_with(", 2;")
    });
    assert!(has_lane_mod_4, "G1 address chain must derive from lane%4");
    assert!(has_lane_div_4, "G1 address chain must derive from lane/4");
}

#[test]
fn g1_finalize_address_chain_references_q_iter_and_batch_head() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    // emit_4d_byte_offset chain references batch_idx, head, q_iter (via q_tile_start)
    assert!(ptx.contains("%q_iter"), "G1 address must reference q_iter");
    assert!(ptx.contains("%batch_idx"), "G1 address must reference batch_idx");
    assert!(ptx.contains("%head"), "G1 address must reference head");
}

#[test]
fn g1_finalize_ascii_only() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    for (i, byte) in ptx.bytes().enumerate() {
        assert!(
            byte.is_ascii(),
            "non-ASCII byte 0x{:02x} at position {}",
            byte,
            i
        );
    }
}

#[test]
fn g1_finalize_ptxas_clean() {
    use std::process::Command;
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    let tmp = std::env::temp_dir().join("g1_test.ptx");
    std::fs::write(&tmp, &ptx).unwrap();
    let result = Command::new("ptxas")
        .args(["-arch=sm_80", tmp.to_str().unwrap(), "-o"])
        .arg(if cfg!(windows) { "NUL" } else { "/dev/null" })
        .output();
    match result {
        Ok(out) if out.status.success() => {}
        Ok(out) => panic!(
            "ptxas failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        ),
        Err(e) => eprintln!("[ptxas not on PATH, skipping]: {e}"),
    }
}
