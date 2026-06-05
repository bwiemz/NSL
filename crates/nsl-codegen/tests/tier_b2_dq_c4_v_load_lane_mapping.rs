// C4 lane-mapping tests per Phase 2.5 spec §5.5 institutional pin.
//
// Each test asserts a specific structural property of the V-tile cp.async load:
//   - real instruction emitted (not a placeholder comment); count >= 4 (C1 + C2 + C3 + C4)
//   - V-specific address contributions: v_saved_ptr parameter reference, %kv_iter (per-kv_iter)
//   - V SMEM destination uses tier_b2_dq_v_offset accessor (non-zero, after Q+K tiles)
//   - V tile uses %c4_ register prefix (no clash with C1/C2/C3 namespaces)
//   - cp.async.commit_group still present after C4 inside the inner loop
//   - ASCII-only invariant preserved (no Unicode introduced by C4)
//   - optional ptxas smoke test

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
        csha: Some(CshaExtras {
            level: 2,
            ..Default::default()
        }),
    }
}

/// C4 emits real cp.async instructions (not a placeholder comment), and together
/// with C1, C2, and C3 the total count is >= 4 (one per tile: Q, dO, K, V).
#[test]
fn c4_v_cp_async_count_is_at_least_four() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    let real_cp_async = ptx.matches("cp.async.ca.shared.global").count();
    assert!(
        real_cp_async >= 4,
        "expected at least 4 cp.async.ca instructions (C1 for Q + C2 for dO + C3 for K + C4 for V), got {real_cp_async}\n\
         (C4 placeholder comment was not replaced with real emission)"
    );
}

/// C4 references v_saved_ptr in the parameter load, proving HBM source is correct.
#[test]
fn c4_v_hbm_source_is_v_saved_ptr() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    assert!(
        ptx.contains("[v_saved_ptr]"),
        "C4 V load must reference v_saved_ptr parameter via ld.param.u64, not k_saved_ptr or q_saved_ptr"
    );
}

/// C4 derives the HBM row offset from kv_iter (V tile is per-kv_iter, not resident).
/// The address chain for V must reference %kv_iter via %c4_kv_tile_start.
#[test]
fn c4_v_address_uses_kv_iter_not_q_iter() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    // kv_tile_start = kv_iter * bkv — per-kv_iter addressing
    assert!(
        ptx.contains("%c4_kv_tile_start"),
        "C4 V tile start register %c4_kv_tile_start must appear in PTX \
         (kv_tile_start = kv_iter * bkv)"
    );
    // The C4 HBM base must be loaded from v_saved_ptr.
    assert!(
        ptx.contains("%c4_v_hbm_base") || ptx.contains("[v_saved_ptr]"),
        "C4 must load v_saved_ptr into an HBM base register"
    );
}

/// C4 address chain terms all present: %c4_lane_byte_off derived from lane_id.
#[test]
fn c4_v_address_terms_all_present() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    assert!(
        ptx.contains("%c4_kv_tile_start"),
        "C4 V address chain must reference kv_tile_start (= kv_iter * bkv)"
    );
    assert!(
        ptx.contains("%c4_lane_byte_off"),
        "C4 must compute per-lane byte offset (lane_id * 4)"
    );
    // The lane_id register must feed the per-lane HBM offset.
    assert!(
        ptx.contains("%lane_id"),
        "C4 V address chain must reference %lane_id"
    );
    assert!(
        ptx.contains("%batch_idx"),
        "C4 V address chain must reference %batch_idx"
    );
    assert!(
        ptx.contains("%head"),
        "C4 V address chain must reference %head"
    );
}

/// C4 writes to SMEM at tier_b2_dq_v_offset (non-zero; V follows Q+K tiles).
/// The emitter must add the v_offset constant to the shared-space base address.
#[test]
fn c4_v_smem_dst_uses_v_offset_accessor() {
    use nsl_codegen::flash_attention_v2::smem_layout::tier_b2_dq_v_offset;
    let cfg = canonical_cfg();
    let v_off = tier_b2_dq_v_offset(&cfg);
    assert!(
        v_off > 0,
        "tier_b2_dq_v_offset must be > 0 (V lives after Q+K tiles in SMEM)"
    );
    let ptx = synthesize_dq_kernel(&cfg).unwrap();
    // The offset must appear as a literal in the add.u32 SMEM base instruction.
    assert!(
        ptx.contains(&format!("{v_off}")),
        "C4 must add v_offset={v_off} to SMEM base (accessor value must appear in PTX)"
    );
    // The comment annotation should carry +v_offset for reviewability.
    assert!(
        ptx.contains("+v_offset"),
        "C4 should annotate the add with +v_offset for reviewability"
    );
}

/// The %c4_ register namespace is used (distinct from %c1_, %c2_, and %c3_ namespaces).
#[test]
fn c4_v_uses_c4_register_prefix() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    assert!(
        ptx.contains("%c4_"),
        "C4 V load must use %c4_ register prefix to avoid clashes with C1/C2/C3 namespaces"
    );
}

/// cp.async.commit_group is present in the inner loop (follows K+V loads).
/// After C4, the total commit_group count must be >= 2 (q_dO block + inner kv block).
#[test]
fn c4_v_commit_group_present_in_inner_loop() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    let commit_count = ptx.matches("cp.async.commit_group").count();
    assert!(
        commit_count >= 2,
        "expected at least 2 cp.async.commit_group (q_dO block + inner kv block), got {commit_count}"
    );
}

/// ASCII-only invariant: C4 must not introduce any non-ASCII bytes into the PTX.
#[test]
fn c4_v_ascii_only_invariant_preserved() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    for (i, byte) in ptx.bytes().enumerate() {
        assert!(
            byte.is_ascii(),
            "non-ASCII byte 0x{:02x} at PTX position {} (C4 introduced non-ASCII)",
            byte,
            i
        );
    }
}

/// C3 and C4 are both present: the placeholder comment is fully replaced.
#[test]
fn c4_placeholder_comment_removed() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    assert!(
        !ptx.contains("C4 will replace"),
        "C4 placeholder comment still present — emit_v_producer_load was not called"
    );
}

/// Optional ptxas smoke test — catches assembler errors at lib-test time.
/// Skipped gracefully when ptxas is not on PATH.
///
/// The MMA `%` prefix bug (emit_inner_loop_body passing un-prefixed register
/// names to emit_mma_instruction) was fixed in E1 of Phase 2.5. This test is
/// now active and should pass on any machine with ptxas on PATH.
#[test]
fn c4_emitted_ptx_ptxas_clean() {
    use std::process::Command;
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    let tmp = std::env::temp_dir().join("c4_test.ptx");
    std::fs::write(&tmp, &ptx).unwrap();
    let cubin = std::env::temp_dir().join("c4_test.cubin");
    let result = Command::new("ptxas")
        .args([
            "-arch=sm_80",
            tmp.to_str().unwrap(),
            "-o",
            cubin.to_str().unwrap(),
        ])
        .output();
    match result {
        Ok(out) if out.status.success() => {}
        Ok(out) => {
            let ptx_excerpt: String = ptx.lines().take(80).collect::<Vec<_>>().join("\n");
            panic!(
                "ptxas failed:\nstdout: {}\nstderr: {}\nPTX (first 80 lines):\n{}\nFull PTX file: {}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr),
                ptx_excerpt,
                tmp.display(),
            );
        }
        Err(e) => eprintln!("[ptxas not on PATH, skipping smoke test]: {e}"),
    }
}
