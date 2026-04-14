/// Regression tests that guard against the two PTX defect classes fixed in
/// Tasks 2-5:
///   1. Hex-float literals in `mov.f32` instructions  (e.g. `mov.f32 %r, 0x3F800000`)
///   2. Shared-memory pointer arithmetic expressed as `[%smem_ptr + offset]`
///      instead of the correct `add.s64 %smem_addr, %shmem_base, offset` form.
use nsl_codegen::flash_attention::{
    synthesize_flash_attention_ptx, CshaExtras, FlashAttentionConfig, RopeStyle,
};

fn base_config() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 64,
        causal: true,
        paged: false,
        rope_q: true,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 80,
        csha: None,
    }
}

fn configs_to_lint() -> Vec<FlashAttentionConfig> {
    vec![
        // Plain FlashAttention (no CSHA)
        base_config(),
        // CSHA level 1 — boundary fusion (prologue norm + epilogue RoPE)
        FlashAttentionConfig {
            csha: Some(CshaExtras::level1(1e-5_f32)),
            ..base_config()
        },
        // CSHA level 2 — full projection pipelining
        FlashAttentionConfig {
            csha: Some(CshaExtras::level2(1e-5_f32, 128)),
            ..base_config()
        },
        // CSHA level 3 — full-block fusion
        FlashAttentionConfig {
            csha: Some(CshaExtras::level3(1e-5_f32, 128)),
            ..base_config()
        },
    ]
}

/// Defect class 1: `mov.f32 %reg, 0xXXXXXXXX` — hex-float literals are
/// illegal in PTX `mov` instructions and cause driver-side compilation errors.
/// The fix replaces them with the `0f########` decimal-hex form.
#[test]
fn no_hex_float_literals_in_mov_f32() {
    for (i, cfg) in configs_to_lint().iter().enumerate() {
        let ptx = synthesize_flash_attention_ptx(cfg);
        let src = String::from_utf8_lossy(&ptx);

        // Pattern: `mov.f32 %<anything>, 0x` — illegal hex-float in mov
        let bad_count = src
            .lines()
            .filter(|l| l.contains("mov.f32") && l.contains(", 0x"))
            .count();

        assert_eq!(
            bad_count, 0,
            "Config[{i}]: found {bad_count} illegal hex-float literal(s) in mov.f32 instructions"
        );
    }
}

/// Defect class 2: `[%smem_ptr + offset]` addressing — the PTX ISA does not
/// allow register + immediate in a bracket address expression for shared memory.
/// The fix materialises the address first via `add.s64 %smem_addr, %shmem_base, offset`.
#[test]
fn no_shmem_plus_reg_addressing() {
    for (i, cfg) in configs_to_lint().iter().enumerate() {
        let ptx = synthesize_flash_attention_ptx(cfg);
        let src = String::from_utf8_lossy(&ptx);

        // Match `[%<ident> + ` which is the illegal pattern
        let bad_count = src
            .lines()
            .filter(|l| {
                let l = l.trim();
                // `[%reg + N]` or `[%reg +N]` — illegal computed bracket address
                l.contains("[%") && l.contains(" + ")
            })
            .count();

        assert_eq!(
            bad_count, 0,
            "Config[{i}]: found {bad_count} illegal `[%reg + offset]` shared-memory address expression(s)"
        );
    }
}

/// Smoke-test that the helper patterns introduced in Tasks 2-5 actually appear
/// in the emitted PTX, confirming the migration was applied (not dead-code).
/// Thresholds: >= 10 occurrences of `, 0f` (decimal f32 immediate) and
///             >= 20 occurrences of `add.s64 %smem_addr, %shmem_base,`.
#[test]
fn helpers_actually_fired() {
    // Aggregate across all configs so at least one heavy config (level2/3) hits the floor.
    let all_ptx: Vec<u8> = configs_to_lint()
        .iter()
        .flat_map(|cfg| synthesize_flash_attention_ptx(cfg))
        .collect();
    let src = String::from_utf8_lossy(&all_ptx);

    let decimal_f32_count = src.lines().filter(|l| l.contains(", 0f")).count();
    let smem_addr_count = src
        .lines()
        .filter(|l| l.contains("add.s64 %smem_addr, %shmem_base,"))
        .count();

    assert!(
        decimal_f32_count >= 10,
        "Expected >= 10 decimal-f32 immediates (`, 0f`), found {decimal_f32_count}. \
         This suggests the hex-float migration helpers were not called."
    );

    assert!(
        smem_addr_count >= 20,
        "Expected >= 20 `add.s64 %smem_addr, %shmem_base,` lines, found {smem_addr_count}. \
         This suggests the SMEM addressing helpers were not called."
    );
}
