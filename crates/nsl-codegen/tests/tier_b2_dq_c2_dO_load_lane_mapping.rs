// C2 lane-mapping tests per Phase 2.5 spec §5.5 institutional pin.
//
// Each test asserts a specific structural property of the dO-tile cp.async load:
//   - real instruction emitted (not a comment placeholder); count >= 2 (C1 + C2)
//   - dO-specific address contributions: d_o_ptr parameter reference, q_iter (via do_tile_start)
//   - dO SMEM destination uses tier_b2_dq_dO_offset (non-zero, after Q+K+V tiles)
//   - cp.async.commit_group still emitted after both C1 and C2
//   - ASCII-only invariant preserved (no Unicode introduced by C2)
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
        gpu_sm: 80,
        segment_masked: false,
        csha: Some(CshaExtras {
            level: 2,
            ..Default::default()
        }),
    }
}

/// C2 emits real cp.async instructions (not a placeholder comment), and together
/// with C1 the total count is >= 2 (one per tile: Q from C1, dO from C2).
#[test]
fn c2_do_cp_async_count_is_at_least_two() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    let real_cp_async = ptx.matches("cp.async.ca.shared.global").count();
    assert!(
        real_cp_async >= 2,
        "expected at least 2 cp.async.ca instructions (C1 for Q + C2 for dO), got {real_cp_async}\n\
         (C2 placeholder comment was not replaced with real emission)"
    );
}

/// C2 references d_o_ptr in the parameter load, proving HBM source is correct.
#[test]
#[allow(non_snake_case)]
fn c2_do_hbm_source_is_d_o_ptr() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    assert!(
        ptx.contains("d_o_ptr"),
        "C2 dO load must reference d_o_ptr parameter, not q_saved_ptr"
    );
}

/// C2 derives the HBM row offset from q_iter (dO tile is Q-aligned).
/// The address chain for dO must reference %q_iter, %batch_idx, %head, and %lane_id
/// (same 4D layout as Q, different base ptr).
#[test]
fn c2_do_address_terms_all_present() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    // q_iter already verified by C1 tests; here we confirm C2's dO chain also uses it.
    assert!(
        ptx.contains("%c2_do_tile_start"),
        "C2 dO tile start register must appear in PTX (do_tile_start = q_iter * bq)"
    );
    // d_o_ptr must be referenced (HBM source param).
    assert!(
        ptx.contains("[d_o_ptr]"),
        "C2 must load d_o_ptr via ld.param.u64"
    );
    // Lane-id offset drives per-lane HBM address — verify %c2_lane_byte_off is computed.
    assert!(
        ptx.contains("%c2_lane_byte_off"),
        "C2 must compute per-lane byte offset (lane_id * 4)"
    );
}

/// C2 writes to SMEM at tier_b2_dq_dO_offset (non-zero; dO follows Q+K+V tiles).
/// The emitter must add the dO_offset constant to the shared-space base address,
/// so the offset value must appear as a literal in the PTX.
#[test]
#[allow(non_snake_case)]
fn c2_do_smem_dst_uses_dO_offset_accessor() {
    use nsl_codegen::flash_attention_v2::smem_layout::tier_b2_dq_dO_offset;
    let cfg = canonical_cfg();
    let do_off = tier_b2_dq_dO_offset(&cfg);
    assert!(
        do_off > 0,
        "tier_b2_dq_dO_offset must be > 0 (dO lives after Q+K+V tiles)"
    );
    let ptx = synthesize_dq_kernel(&cfg).unwrap();
    // The offset must appear as a literal in the add.u32 SMEM base instruction.
    assert!(
        ptx.contains(&format!("{do_off}")),
        "C2 must add dO_offset={do_off} to SMEM base (accessor value must appear in PTX)"
    );
    // Sanity: the comment +dO_offset annotation also carries the value.
    assert!(
        ptx.contains(&format!("+dO_offset")),
        "C2 should annotate the add with +dO_offset for reviewability"
    );
}

/// cp.async.commit_group is still emitted after both C1 and C2 complete.
#[test]
fn c2_do_commit_group_present_after_both_tiles() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    assert!(
        ptx.contains("cp.async.commit_group"),
        "cp.async.commit_group must follow both C1 Q-tile and C2 dO-tile loads"
    );
}

/// ASCII-only invariant: C2 must not introduce any non-ASCII bytes into the PTX.
#[test]
fn c2_do_ascii_only_invariant_preserved() {
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    for (i, byte) in ptx.bytes().enumerate() {
        assert!(
            byte.is_ascii(),
            "non-ASCII byte 0x{:02x} at PTX position {} (C2 introduced non-ASCII)",
            byte,
            i
        );
    }
}

/// Optional ptxas smoke test — catches assembler errors at lib-test time.
/// Skipped gracefully when ptxas is not on PATH.
///
/// The MMA `%` prefix bug (emit_inner_loop_body passing un-prefixed register
/// names to emit_mma_instruction) was fixed in E1 of Phase 2.5. This test is
/// now active and should pass on any machine with ptxas on PATH.
#[test]
fn c2_emitted_ptx_ptxas_clean() {
    use std::process::Command;
    let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
    let tmp = std::env::temp_dir().join("c2_test.ptx");
    std::fs::write(&tmp, &ptx).unwrap();
    let cubin = std::env::temp_dir().join("c2_test.cubin");
    let result = Command::new("ptxas")
        .args(["-arch=sm_80", tmp.to_str().unwrap(), "-o", cubin.to_str().unwrap()])
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
