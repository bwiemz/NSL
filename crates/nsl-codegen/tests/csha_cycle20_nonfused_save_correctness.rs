//! CSHA Cycle 20 T3 — non-fused save-path correctness regressions.
//!
//! Two bugs on the non-fused / non-CSHA save path in
//! `crates/nsl-codegen/src/flash_attention_v2/mod.rs`:
//!
//! **BUG A (K-save order)**: `SaveSet::QK` was emitted BEFORE
//! `emit_k_tile_load` — the K save read uninitialised SMEM at
//! `%k_smem_base`. The in-file comment explicitly states K save must run
//! AFTER K tile load.
//!
//! **BUG B (V-save Q-indexing)**: `SaveSet::V` used the shared emitter
//! `emit_save_activations_subset` whose HBM row math bakes in
//! `%q_start + warp_row` (Q-indexed). Under a multi-KV-tile run
//! (`k_start > 0`), this writes to the wrong HBM rows.
//!
//! Both tests are text-level assertions over the emitted PTX for a
//! non-fused config with `save_activations_for_backward=true`. The
//! block_q/block_kv split is symmetric here; the multi-tile behaviour
//! we care about is runtime (%k_start advances per KV loop iter), so
//! the correctness fix is a textual property of the emitted PTX
//! (address math references %k_start, not %q_start; K-save skip label
//! appears AFTER the K tile load marker).

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::synthesize_flash_attention_ptx_v2;

/// Non-fused save-path config: CSHA extras with
/// `save_activations_for_backward=true` and `fused_projections=false`.
/// Uses `block_q=block_kv=32` so the standard-path branch (line ~312+)
/// is taken instead of the fused-projections orchestrator.
fn non_fused_save_config() -> FlashAttentionConfig {
    // Build CSHA extras with fused_projections=false so the standard
    // non-fused orchestrator branch runs. save_activations=true so the
    // Tier C save sites actually emit.
    let csha = CshaExtras {
        level: 1,
        fused_rmsnorm: false,
        fused_projections: false,
        fused_output_proj: false,
        active_heads: 0,
        rmsnorm_eps: 1e-5,
        d_model: 128,
        save_activations_for_backward: true,
        skip_rmsnorm_prologue: true,
        static_seq_len: None,
    };
    FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 32,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 75,
        segment_masked: false,
        csha: Some(csha),
        checkpoint: None,
    }
}

/// BUG A regression: the K save-skip label must appear AFTER the K tile
/// load body. Pre-fix, the QK save was emitted BEFORE emit_k_tile_load,
/// so `V2_CSHA_SAVE_K_SKIP_0` would appear before the "K tile load"
/// comment in the emitted PTX. Post-fix, the order is swapped.
#[test]
fn t3_bug_a_k_save_runs_after_k_tile_load() {
    let ptx_bytes = synthesize_flash_attention_ptx_v2(&non_fused_save_config());
    let ptx = String::from_utf8(ptx_bytes).expect("PTX must be valid UTF-8");

    // Locate the standard-path (non-fused) K tile load and the K save
    // skip label for q_iter=0. If either is missing, the config didn't
    // route through the intended branch — fail loudly.
    let k_tile_load_marker = "// K tile load:";
    let k_save_skip_label = "V2_CSHA_SAVE_K_SKIP_0:";

    let k_load_pos = ptx.find(k_tile_load_marker).unwrap_or_else(|| {
        panic!("PTX did not emit '{k_tile_load_marker}' — config routed wrong");
    });
    let k_save_pos = ptx.find(k_save_skip_label).unwrap_or_else(|| {
        panic!(
            "PTX did not emit '{k_save_skip_label}' — save_activations gate off?"
        );
    });

    assert!(
        k_save_pos > k_load_pos,
        "BUG A regressed: '{k_save_skip_label}' at byte {k_save_pos} came BEFORE '{k_tile_load_marker}' at byte {k_load_pos} — K save reads uninitialised SMEM"
    );
}

/// BUG B regression: the V-save address math must use `%k_start` (K/V
/// indexed), not `%q_start` (Q indexed). Post-fix, the inline V-save
/// block emits `add.u64 %rd_save_off, %rd_save_bh, %k_start;`. Pre-fix
/// (via `emit_save_activations_subset`) the V-save block instead
/// emits `add.u64 %rd_save_off, %rd_save_off, %q_start;`.
#[test]
fn t3_bug_b_v_save_uses_k_start_not_q_start() {
    let ptx_bytes = synthesize_flash_attention_ptx_v2(&non_fused_save_config());
    let ptx = String::from_utf8(ptx_bytes).expect("PTX must be valid UTF-8");

    // The T3-inlined V-save carries a fixed marker comment so we can
    // locate it precisely without confusing it with the K-save block.
    let t3_marker = "-- Cycle 20 T3: inline V-save (k_start-indexed) --";
    let t3_end_label = "V2_CSHA_SAVE_V_T3_SKIP_0:";

    let start = ptx.find(t3_marker).unwrap_or_else(|| {
        panic!("PTX did not emit T3 V-save marker '{t3_marker}' — BUG B regressed")
    });
    let end = ptx[start..].find(t3_end_label).map(|off| start + off).unwrap_or_else(|| {
        panic!("PTX did not emit T3 V-save end label '{t3_end_label}'")
    });
    let block = &ptx[start..end];

    assert!(
        block.contains("%k_start"),
        "BUG B regressed: T3 V-save block missing '%k_start' — got:\n{block}"
    );
    assert!(
        !block.contains("%q_start"),
        "BUG B regressed: T3 V-save block references '%q_start' (Q-indexed) — got:\n{block}"
    );
}

/// R3 regression: every `%rd_save_*` virtual register referenced in the
/// emitted PTX must have a matching `.reg .u64 %rd_save_<name>`
/// declaration. Missed declarations (e.g. `%rd_save_bh` in the T3
/// inline V-save block) fail `ptxas` at runtime `cuModuleLoadData` but
/// pass snapshot tests because no snapshot exercises the non-fused-save
/// config path.
#[test]
fn t3_all_rd_save_regs_declared() {
    use std::collections::HashSet;

    let ptx_bytes = synthesize_flash_attention_ptx_v2(&non_fused_save_config());
    let ptx = String::from_utf8(ptx_bytes).expect("PTX must be valid UTF-8");

    // Collect referenced %rd_save_<name> identifiers (excluding declaration lines).
    let mut referenced: HashSet<String> = HashSet::new();
    let mut declared: HashSet<String> = HashSet::new();
    for line in ptx.lines() {
        let trimmed = line.trim_start();
        let is_decl = trimmed.starts_with(".reg ");
        // Manual scan: find every "%rd_save_" occurrence and grab the identifier chars.
        let bytes = line.as_bytes();
        let needle = b"%rd_save_";
        let mut i = 0usize;
        while i + needle.len() <= bytes.len() {
            if &bytes[i..i + needle.len()] == needle {
                let start = i + 1; // skip '%'
                let mut end = start + b"rd_save_".len();
                while end < bytes.len() {
                    let c = bytes[end];
                    if c.is_ascii_alphanumeric() || c == b'_' {
                        end += 1;
                    } else {
                        break;
                    }
                }
                let ident = std::str::from_utf8(&bytes[start..end]).unwrap().to_string();
                if is_decl {
                    declared.insert(ident);
                } else {
                    referenced.insert(ident);
                }
                i = end;
            } else {
                i += 1;
            }
        }
    }

    let missing: Vec<&String> = referenced.difference(&declared).collect();
    assert!(
        missing.is_empty(),
        "R3 regressed: emitted PTX references undeclared %rd_save_* registers: {missing:?}\n\
         declared={declared:?}\n\
         referenced={referenced:?}"
    );
}

/// Cross-check: the fused-projections path is UNAFFECTED by the T3
/// fixes. Emit a fused config and confirm no T3 marker appears (the
/// fused orchestrator uses its own K/V save sites that route through
/// the fused Step 3c + V pre-pass, not the standard-path V-save).
#[test]
fn t3_fused_path_unaffected_no_t3_marker() {
    let csha = CshaExtras {
        level: 2,
        fused_rmsnorm: true,
        fused_projections: true,
        fused_output_proj: true,
        active_heads: 0,
        rmsnorm_eps: 1e-5,
        d_model: 128,
        save_activations_for_backward: true,
        skip_rmsnorm_prologue: false,
        static_seq_len: None,
    };
    let cfg = FlashAttentionConfig {
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
        segment_masked: false,
        csha: Some(csha),
        checkpoint: None,
    };
    let ptx_bytes = synthesize_flash_attention_ptx_v2(&cfg);
    let ptx = String::from_utf8(ptx_bytes).expect("PTX must be valid UTF-8");
    assert!(
        !ptx.contains("Cycle 20 T3: inline V-save"),
        "T3 inline V-save leaked into the fused-projections path — scope creep"
    );
}
