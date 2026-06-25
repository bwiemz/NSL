//! Sprint 1 T1.4: combined scalar + Tier B.2 hybrid backward PTX module.
//!
//! `synthesize_backward_combined` at module scope returns either:
//!   * unchanged scalar PTX (compile-time-ineligible config), OR
//!   * a union module containing BOTH the scalar v2 backward kernel AND
//!     the four Tier B.2 hybrid kernels under a single `.version` /
//!     `.target` / `.extern .shared shmem[]` header.
//!
//! The runtime branch on `tier_b2_active` (Sprint 1 T1.3) chooses which
//! entry to launch; both must be present in the loaded module so
//! `cuModuleGetFunction` succeeds for either path.

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::{
    phases::backward::prelude::kernel_name as scalar_backward_kernel_name,
    synthesize_backward, synthesize_backward_combined,
};

fn smoke_eligible_cfg() -> FlashAttentionConfig {
    // hd=64, heads=1, d_model=64, csha level=2, gpu_sm=80, rope_q=false.
    // Compile-time-eligible per `tier_b2_hybrid_backward_compile_time_eligible`.
    //
    // block_q/block_kv set to 32 (not the 64 the runtime hybrid path would
    // use at smoke seq_len) because the SCALAR backward validator caps at
    // 99 KB and (bq=64, bkv=64, hd=64, d_model=64) needs ~140 KB. The
    // compile-time predicate ignores block_q (`seq_len == block_q` is a
    // RUNTIME check), so any bq that fits the scalar validator works for
    // structural assertions on the combined module. `tier_b2_can_dispatch`
    // also overrides bq with its ladder-pinned value (64 for hd=64) before
    // synthesising the hybrid bodies.
    FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 64,
        causal: true,
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
            d_model: 64,
            active_heads: 1,
            ..Default::default()
        }),
        checkpoint: None,
    }
}

fn ineligible_level1_cfg() -> FlashAttentionConfig {
    // csha.level=1 fails the Tier B.2 dispatch precondition; combined PTX
    // must equal the scalar baseline exactly.
    let mut c = smoke_eligible_cfg();
    if let Some(csha) = c.csha.as_mut() {
        csha.level = 1;
    }
    c
}

#[test]
fn ineligible_config_returns_scalar_unchanged() {
    let cfg = ineligible_level1_cfg();
    let scalar = synthesize_backward(&cfg).expect("scalar synth ok");
    let combined = synthesize_backward_combined(&cfg).expect("combined synth ok");
    assert_eq!(
        scalar, combined,
        "ineligible config: combined PTX must be byte-identical to scalar",
    );
    // Sanity: no Tier B.2 entries leak in.
    for name in [
        "tier_b2_d_prepass",
        "tier_b2_dq_kernel",
        "tier_b2_dkdv_kernel",
        "tier_b2_proj_backward",
    ] {
        assert!(
            !combined.contains(name),
            "ineligible config must not contain Tier B.2 entry {name}",
        );
    }
}

#[test]
fn eligible_config_emits_all_five_entries() {
    let cfg = smoke_eligible_cfg();
    let combined = synthesize_backward_combined(&cfg).expect("combined synth ok");

    // The scalar backward entry name is computed by the same helper
    // production codegen at kernel.rs:822-837 uses for FFI name lookup.
    let scalar_name = scalar_backward_kernel_name(&cfg);
    assert!(
        combined.contains(&format!(".visible .entry {scalar_name}")),
        "combined PTX missing scalar entry .visible .entry {scalar_name}",
    );

    // All four Tier B.2 hybrid entries must be present.
    for name in [
        "tier_b2_d_prepass",
        "tier_b2_dq_kernel",
        "tier_b2_dkdv_kernel",
        "tier_b2_proj_backward",
    ] {
        assert!(
            combined.contains(&format!(".visible .entry {name}")),
            "combined PTX missing Tier B.2 entry .visible .entry {name}",
        );
    }
}

#[test]
fn eligible_config_has_exactly_one_module_header() {
    let cfg = smoke_eligible_cfg();
    let combined = synthesize_backward_combined(&cfg).expect("combined synth ok");

    // Exactly one `.version` directive (header union).
    let version_count = combined.match_indices(".version ").count();
    assert_eq!(
        version_count, 1,
        "expected exactly one .version directive, got {version_count}\nPTX:\n{combined}",
    );

    // Exactly one `.target` directive.
    let target_count = combined.match_indices(".target ").count();
    assert_eq!(
        target_count, 1,
        "expected exactly one .target directive, got {target_count}",
    );

    // Exactly one `.extern .shared .align ... shmem[]` directive.
    let shmem_extern_count = combined.match_indices(".extern .shared").count();
    assert_eq!(
        shmem_extern_count, 1,
        "expected exactly one .extern .shared shmem[] directive, got {shmem_extern_count}",
    );
}

#[test]
fn combined_ptx_is_ascii_only() {
    // Mirrors `tier_b2_ascii_only_ptx.rs` invariant — Unicode in PTX
    // causes ptxas to reject the module with CUDA_ERROR_INVALID_PTX.
    let cfg = smoke_eligible_cfg();
    let combined = synthesize_backward_combined(&cfg).expect("combined synth ok");
    for (line_idx, line) in combined.lines().enumerate() {
        for (col, byte) in line.bytes().enumerate() {
            assert!(
                byte.is_ascii(),
                "non-ASCII byte 0x{byte:02x} at line {} col {}:\n  {line}",
                line_idx + 1,
                col + 1,
            );
        }
    }
}
